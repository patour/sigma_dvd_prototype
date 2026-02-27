"""
Graph builder for PDN netlists: edge materialization & boundary stitching.

Contains GraphBuilder class and _use_optimized_edges ContextVar.
"""

import logging
from collections import defaultdict
from contextvars import ContextVar
from typing import Dict, List, Optional, Set, Tuple

from graph.rx_graph import RustworkxMultiDiGraphWrapper
from graph.rx_algorithms import contract_nodes, node_connected_component
from .edge_attrs import (
    create_element_edge,
    _get_net_type_index as _get_edge_net_type_index,
    reset_net_type_table as reset_edge_net_type_table,
    BaseElementEdge,
)
from .metadata import (
    _get_net_type_index, _reset_net_type_tables,
    _FLAG_BOUNDARY, _FLAG_PACKAGE, _FLAG_VSRC, _FLAG_DIE,
    PDNNodeAttrs, ParseStats,
)
from .spice_lexer import (
    GMAX, SHORT_THRESHOLD,
    _ELEM_R, _ELEM_C, _ELEM_L, _ELEM_V, _ELEM_I, _ELEM_TYPES,
    _KEY_TYPE, _KEY_VALUE, _KEY_ELEM_NAME, _KEY_TILE_ID, _KEY_NET_TYPE,
    _CAT_DIE, _CAT_PACKAGE, _CAT_UNMAPPED,
    _RE_COORD_EXTRACT,
    _parse_spice_value,
    _check_net_filter,
)
from .current_sources import CurrentSource


# =============================================================================
# Optimized Edge Storage ContextVar
# =============================================================================

_use_optimized_edges: ContextVar[bool] = ContextVar('_use_optimized_edges', default=True)


def get_use_optimized_edges() -> bool:
    """Return whether optimized edge attribute classes are used.

    Returns:
        True if edge attributes use specialized slotted dataclasses (default),
        False if all edges use dict-based attributes (legacy mode).
    """
    return _use_optimized_edges.get()


def set_use_optimized_edges(value: bool) -> None:
    """Enable/disable optimized edge attribute storage.

    When enabled (default), edge attributes use specialized slotted dataclasses
    (ResistorEdge, CapacitorEdge, etc.) that reduce memory by ~90-95% per edge.
    This is critical for 100M+ edge netlists.

    When disabled, edges use dict-based attributes for backward compatibility.

    Args:
        value: If True, use memory-optimized edge classes.
               If False, use legacy dict-based attributes.
    """
    _use_optimized_edges.set(value)


class GraphBuilder:
    """
    Builds and manages the rustworkx MultiDiGraph representation of the PDN.
    """

    def __init__(self, validate: bool = False, strict: bool = False, net_filter: Optional[str] = None,
                 store_instance_sources: bool = False, vsrc_resistor_pattern: str = 'rs'):
        self.graph = RustworkxMultiDiGraphWrapper()
        self.validate = validate
        self.strict = strict
        self.net_filter = net_filter.lower() if net_filter else None  # Store lowercase for case-insensitive comparison
        self.store_instance_sources = store_instance_sources
        self.vsrc_resistor_pattern = vsrc_resistor_pattern  # For determining which R edges need elem_name
        # Union-Find structure for package/main netlist connectivity
        self.uf_parent: Dict[str, str] = {}  # Union-Find parent pointers for package nodes
        self.uf_net: Dict[str, str] = {}  # Net type for each union-find root
        self.package_edges: List[Tuple[str, str]] = []  # Deferred package edges for union-find
        self.stats = ParseStats()
        
        # Metadata dictionaries
        self.vsrc_dict: Dict[str, Dict] = {}
        self.parameters: Dict[str, str] = {}
        self.instance_node_map: Dict[str, List[str]] = {}  # Backward compat: name -> [node+, node-]
        self.instance_sources: Dict[str, CurrentSource] = {}  # Full current source data
        self.merged_nodes: List[Tuple[str, str, int]] = []
        self.mutual_inductors: Dict[str, Tuple[str, str, float]] = {}
        self.node_net_map: Dict[str, str] = {}  # Die node name -> net name from .nd files
        self.node_net_map_lower: Dict[str, str] = {}  # Die node name -> lowercase net name for filtering
        
        # Parsing context
        self.current_tile_id: Optional[Tuple[int, int]] = None
        self.current_file_type: str = 'die'  # 'die', 'package', or 'instance'
        self.tile_grid: Optional[Tuple[int, int]] = None  # (N, M) from .partition_info
        
        # Node tracking
        self.boundary_nodes: Set[str] = set()
        self.node_attributes: Dict[str, Dict] = defaultdict(dict)

        # Edge index tracking for efficient filtering (avoid full graph iteration)
        self.package_edge_indices: List[int] = []  # Edge indices from package.ckt
        self.vsrc_edge_indices: List[int] = []     # Voltage source edge indices

        self.logger = logging.getLogger(__name__)
        
    def _uf_find(self, node: str) -> str:
        """Union-Find: find root with path compression"""
        if node not in self.uf_parent:
            self.uf_parent[node] = node  # Initialize
            return node
        
        # Path compression
        if self.uf_parent[node] != node:
            self.uf_parent[node] = self._uf_find(self.uf_parent[node])
        return self.uf_parent[node]
    
    def _uf_union(self, node1: str, node2: str) -> None:
        """Union-Find: union two nodes, propagating net type from die nodes"""
        root1 = self._uf_find(node1)
        root2 = self._uf_find(node2)
        
        if root1 == root2:
            return  # Already in same set
        
        # Get net types (from .nd file for die nodes, or from existing union)
        # Use original case net names for union-find
        net1 = self.node_net_map.get(node1)  # Die node from .nd file
        net2 = self.node_net_map.get(node2)  # Die node from .nd file
        root1_net = net1 or self.uf_net.get(root1)  # Existing net from root
        root2_net = net2 or self.uf_net.get(root2)  # Existing net from root
        
        # Union: prefer root with net type from die node
        if root1_net:
            self.uf_parent[root2] = root1
            self.uf_net[root1] = root1_net
        elif root2_net:
            self.uf_parent[root1] = root2
            self.uf_net[root2] = root2_net
        else:
            # Neither has net type yet, arbitrary union
            self.uf_parent[root2] = root1
    
    def _get_node_net(self, node: str) -> Optional[str]:
        """Get effective net type for a node (from .nd file or union-find)"""
        # First check if it's a die node with explicit mapping
        net = self.node_net_map.get(node)
        if net:
            return net
        
        # Check union-find for package/unmapped nodes
        root = self._uf_find(node)
        return self.uf_net.get(root)
    
    def add_node(self, name: str, **attrs):
        """Add node with attributes, merging with existing if present"""
        if name not in self.graph:
            # Node '0' is special: never package, always unmapped, excluded from statistics
            if name == '0':
                is_package_node = False
            else:
                # Determine if this is a package node (in package file but not in die node map)
                is_package_node = (self.current_file_type == 'package' and
                                 name not in self.node_net_map)

            # Build flags for compact storage
            flags = 0
            if name in self.boundary_nodes:
                flags |= PDNNodeAttrs.FLAG_BOUNDARY
            if is_package_node:
                flags |= PDNNodeAttrs.FLAG_PACKAGE
            # Check if this is a die node (X_Y_LAYER pattern)
            # Die nodes have x, y, layer extractable from name
            if not is_package_node and name != '0':
                # Quick check for die node pattern: starts with digits_digits
                parts = name.split('_')
                if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
                    flags |= PDNNodeAttrs.FLAG_DIE

            # Create compact node attributes
            node_attrs = PDNNodeAttrs(
                name=name,
                tile_id=self.current_tile_id,
                flags=flags
            )
            # Apply any additional attributes from caller (skip read-only computed properties)
            _readonly_attrs = {'x', 'y', 'layer', 'name', 'is_die'}
            for k, v in attrs.items():
                if k not in _readonly_attrs and hasattr(node_attrs, k) and not k.startswith('_'):
                    setattr(node_attrs, k, v)

            self.graph.add_node(name, node_attrs)

            # Track unmapped nodes (not in .nd file, not package)
            # Node '0' is global ground and always counts as unmapped (but excluded from statistics)
            if name == '0':
                pass  # Node '0' is unmapped but not counted in statistics
            elif (name not in self.node_net_map and
                  not is_package_node and
                  self.current_file_type != 'package'):
                self.stats.unmapped_nodes += 1
            # No longer need to call _extract_coordinates - x, y, layer are computed from name
        else:
            # Update existing node attributes
            self.graph.nodes_dict[name].update(attrs)

    def _infer_net_type(self, node_name: str) -> Optional[str]:
        """
        Infer power net type from node name.
        For die nodes: use exact mapping from .nd files
        For package nodes or unmapped nodes: return None (will be categorized separately)
        Node '0' is global ground, not a net type.
        """
        # Only use exact mapping from .nd file (die nodes)
        if node_name in self.node_net_map:
            return self.node_net_map[node_name]
        
        # No pattern matching - if not in .nd file, it's unmapped or package
        # Node '0' is global ground, not mapped to any net
        return None
            
    def add_element(self, elem_type: str, node1: str, node2: str, 
                   value: float, name: str, **attrs) -> bool:
        """Add circuit element as edge between two nodes.
        
        Returns:
            True if element was added, False if filtered out by net_filter.
        """
        # For package/main netlist elements (not in tiles), defer union-find processing
        # Tile elements have die nodes with explicit net types from .nd files
        if self.current_file_type in ['package', 'die'] and self.current_tile_id is None:
            # Defer package connectivity (will process after all parsing)
            # Skip node '0' to prevent cross-net contamination
            if node1 != '0' and node2 != '0':
                self.package_edges.append((node1, node2))
        
        # Get net type from .nd file (die nodes) or union-find (package nodes)
        node1_net = self._get_node_net(node1)
        node2_net = self._get_node_net(node2)
        net_type = node1_net or node2_net
        
        # Apply net filter if active (case-insensitive)
        # Exception: Don't filter voltage sources or package elements during parsing
        # These will be filtered post-processing based on connectivity to filtered net
        if self.net_filter is not None and elem_type != 'V' and self.current_file_type != 'package':
            # Get lowercase net names for comparison
            node1_net_lower = self.node_net_map_lower.get(node1) or (node1_net.lower() if node1_net else None)
            node2_net_lower = self.node_net_map_lower.get(node2) or (node2_net.lower() if node2_net else None)
            # Include element if either node belongs to filtered net
            if node1_net_lower != self.net_filter and node2_net_lower != self.net_filter:
                return False  # Skip this element
        
        # Ensure nodes exist
        self.add_node(node1)
        self.add_node(node2)

        # Determine if this resistor needs elem_name stored
        # - Always for V (voltage sources)
        # - For R only if it matches vsrc_resistor_pattern (for vsrc node identification)
        # This optimization saves ~160 bytes per die resistor (99.9% of resistors)
        needs_elem_name = (
            elem_type == 'V' or  # Always for voltage sources
            (elem_type == 'R' and name.lower() == self.vsrc_resistor_pattern.lower())
        )

        # Create edge attributes
        if get_use_optimized_edges():
            # Use memory-optimized edge attribute classes
            edge_obj = create_element_edge(
                elem_type=elem_type,
                value=value,
                elem_name=name if needs_elem_name else None,
                tile_id=self.current_tile_id,
                net_type=net_type,
                needs_elem_name=needs_elem_name,
            )
            # Add edge with optimized object
            edge_idx = self.graph.add_edge(node1, node2, edge_obj=edge_obj)
        else:
            # Legacy dict-based edge attributes with interned keys
            interned_type = _ELEM_TYPES.get(elem_type, elem_type)
            edge_attrs = {
                _KEY_TYPE: interned_type,
                _KEY_VALUE: value,
                _KEY_TILE_ID: self.current_tile_id,
                _KEY_NET_TYPE: net_type
            }
            # Only store elem_name for R and V types to save memory on C/L/I edges
            if elem_type in ('R', 'V'):
                edge_attrs[_KEY_ELEM_NAME] = name
            edge_attrs.update(attrs)
            # Add edge (MultiDiGraph allows multiple edges between same nodes)
            edge_idx = self.graph.add_edge(node1, node2, **edge_attrs)

        # Track edge indices for efficient post-processing filtering
        if elem_type == 'V':
            self.vsrc_edge_indices.append(edge_idx)
        if self.current_file_type == 'package':
            self.package_edge_indices.append(edge_idx)

        # Update global statistics
        self.stats.elements_total += 1
        if elem_type == 'R':
            self.stats.resistors += 1
        elif elem_type == 'C':
            self.stats.capacitors += 1
        elif elem_type == 'L':
            self.stats.inductors += 1
        elif elem_type == 'V':
            self.stats.vsources += 1
        elif elem_type == 'I':
            self.stats.isources += 1
        
        # Update per-net statistics (separate die, package, and unmapped)
        if net_type:
            # Determine category: package nodes are in package file and not in die node map
            node1_is_package = (node1 not in self.node_net_map and 
                              self.current_file_type == 'package')
            node2_is_package = (node2 not in self.node_net_map and 
                              self.current_file_type == 'package')
            
            # Element is package if either node is a package node
            is_package_elem = node1_is_package or node2_is_package
            category = _CAT_PACKAGE if is_package_elem else _CAT_DIE
            
            if net_type not in self.stats.net_stats:
                self.stats.net_stats[net_type] = {
                    'die': {
                        'nodes': set(),
                        'resistors': 0,
                        'capacitors': 0,
                        'inductors': 0,
                        'vsources': 0,
                        'isources': 0,
                        'isources_with_waveforms': 0,
                        'wscale_values': [],
                        'total_resistance': 0.0,
                        'total_capacitance': 0.0,
                        'total_inductance': 0.0,
                        'total_current': 0.0
                    },
                    'package': {
                        'nodes': set(),
                        'resistors': 0,
                        'capacitors': 0,
                        'inductors': 0,
                        'vsources': 0,
                        'isources': 0,
                        'isources_with_waveforms': 0,
                        'wscale_values': [],
                        'total_resistance': 0.0,
                        'total_capacitance': 0.0,
                        'total_inductance': 0.0,
                        'total_current': 0.0
                    },
                    'unmapped': {
                        'nodes': set(),
                        'resistors': 0,
                        'capacitors': 0,
                        'inductors': 0,
                        'vsources': 0,
                        'isources': 0,
                        'isources_with_waveforms': 0,
                        'wscale_values': [],
                        'total_resistance': 0.0,
                        'total_capacitance': 0.0,
                        'total_inductance': 0.0,
                        'total_current': 0.0
                    }
                }

            net_stat = self.stats.net_stats[net_type][category]
            # Exclude node '0' from statistics
            if node1 != '0':
                net_stat['nodes'].add(node1)
            if node2 != '0':
                net_stat['nodes'].add(node2)
            
            if elem_type == 'R':
                net_stat['resistors'] += 1
                net_stat['total_resistance'] += value
            elif elem_type == 'C':
                net_stat['capacitors'] += 1
                net_stat['total_capacitance'] += value
            elif elem_type == 'L':
                net_stat['inductors'] += 1
                net_stat['total_inductance'] += value
            elif elem_type == 'V':
                net_stat['vsources'] += 1
            elif elem_type == 'I':
                net_stat['isources'] += 1
                net_stat['total_current'] += abs(value)
        else:
            # No net_type means unmapped element
            if 'unmapped' not in self.stats.net_stats:
                self.stats.net_stats['unmapped'] = {
                    'die': {
                        'nodes': set(),
                        'resistors': 0,
                        'capacitors': 0,
                        'inductors': 0,
                        'vsources': 0,
                        'isources': 0,
                        'isources_with_waveforms': 0,
                        'wscale_values': [],
                        'total_resistance': 0.0,
                        'total_capacitance': 0.0,
                        'total_inductance': 0.0,
                        'total_current': 0.0
                    },
                    'package': {
                        'nodes': set(),
                        'resistors': 0,
                        'capacitors': 0,
                        'inductors': 0,
                        'vsources': 0,
                        'isources': 0,
                        'isources_with_waveforms': 0,
                        'wscale_values': [],
                        'total_resistance': 0.0,
                        'total_capacitance': 0.0,
                        'total_inductance': 0.0,
                        'total_current': 0.0
                    },
                    'unmapped': {
                        'nodes': set(),
                        'resistors': 0,
                        'capacitors': 0,
                        'inductors': 0,
                        'vsources': 0,
                        'isources': 0,
                        'isources_with_waveforms': 0,
                        'wscale_values': [],
                        'total_resistance': 0.0,
                        'total_capacitance': 0.0,
                        'total_inductance': 0.0,
                        'total_current': 0.0
                    }
                }

            # Determine if unmapped element is in package or die
            is_package_elem = self.current_file_type == 'package'
            category = 'package' if is_package_elem else 'unmapped'
            
            net_stat = self.stats.net_stats['unmapped'][category]
            # Exclude node '0' from statistics
            if node1 != '0':
                net_stat['nodes'].add(node1)
            if node2 != '0':
                net_stat['nodes'].add(node2)
            
            if elem_type == 'R':
                net_stat['resistors'] += 1
                net_stat['total_resistance'] += value
            elif elem_type == 'C':
                net_stat['capacitors'] += 1
                net_stat['total_capacitance'] += value
            elif elem_type == 'L':
                net_stat['inductors'] += 1
                net_stat['total_inductance'] += value
            elif elem_type == 'V':
                net_stat['vsources'] += 1
            elif elem_type == 'I':
                net_stat['isources'] += 1
                net_stat['total_current'] += abs(value)
        
        return True
            
    def add_grounded_element(self, elem_type: str, node: str, value: float, 
                            name: str, **attrs) -> bool:
        """Add element connected to ground (node '0').
        
        Returns:
            True if element was added, False if filtered out by net_filter.
        """
        return self.add_element(elem_type, node, '0', value, name, **attrs)
        
    def mark_boundary_node(self, name: str):
        """Mark node as boundary node (needs stitching)"""
        self.boundary_nodes.add(name)
        if name in self.graph:
            self.graph.nodes_dict[name]['is_boundary'] = True
        self.stats.boundary_nodes += 1
        
    def stitch_nodes(self, name1: str, name2: str):
        """
        Stitch two nodes together (merge them).
        This is used for boundary nodes across tiles.
        """
        if name1 not in self.graph or name2 not in self.graph:
            self.logger.warning(f"Cannot stitch nodes {name1} and {name2}: one or both not found")
            return

        # Merge name2 into name1
        try:
            contract_nodes(self.graph, name1, name2, self_loops=False)
            self.merged_nodes.append((name2, name1, 0))  # 0 = stitch merge type
            self.logger.debug(f"Stitched nodes: {name2} -> {name1}")
        except Exception as e:
            self.logger.error(f"Error stitching nodes {name1} and {name2}: {e}")
            
    def validate_node_uniqueness(self):
        """Check for node name collisions and report detailed errors"""
        # This is mostly already handled by the wrapper, but we can add custom checks
        node_sources = defaultdict(list)

        for node in self.graph.nodes():
            tile_id = self.graph.nodes_dict[node].get('tile_id')
            is_package = self.graph.nodes_dict[node].get('is_package')
            source = f"tile_{tile_id}" if tile_id else ('package' if is_package else 'main')
            node_sources[node].append(source)
        
        # Check for actual duplicates (shouldn't happen with NetworkX, but check our tracking)
        duplicates = {node: sources for node, sources in node_sources.items() if len(sources) > 1}
        
        if duplicates:
            error_msg = "Node name collisions detected:\n"
            for node, sources in duplicates.items():
                error_msg += f"  Node '{node}' appears in: {', '.join(sources)}\n"
            
            if self.strict:
                raise ValueError(error_msg)
            else:
                self.logger.warning(error_msg)
                
    def finalize(self):
        """Finalize graph and add metadata"""
        self.stats.nodes_after_cleanup = self.graph.number_of_nodes()
        
        # Convert net_stats node sets to counts (separate die, package, and unmapped)
        net_stats_serializable = {}
        for net, categories in self.stats.net_stats.items():
            net_stats_serializable[net] = {}
            for category in ['die', 'package', 'unmapped']:
                if category in categories:
                    stats = categories[category]
                    net_stats_serializable[net][category] = {
                        'nodes': len(stats['nodes']),
                        'resistors': stats['resistors'],
                        'capacitors': stats['capacitors'],
                        'inductors': stats['inductors'],
                        'vsources': stats['vsources'],
                        'isources': stats['isources'],
                        'total_resistance_kohm': stats['total_resistance'],
                        'total_capacitance_ff': stats['total_capacitance'],
                        'total_inductance_nh': stats['total_inductance'],
                        'total_current_ma': stats['total_current']
                    }
        
        # Build net_connectivity from union-find results and die node mappings
        net_connectivity = defaultdict(list)
        
        # Add die nodes from .nd files
        for node, net in self.node_net_map.items():
            if node in self.graph:
                net_connectivity[net].append(node)
        
        # Add package nodes from union-find
        for node in self.graph.nodes():
            if node not in self.node_net_map and node != '0':
                net = self._get_node_net(node)
                if net:
                    net_connectivity[net].append(node)
        
        # Add metadata to graph (vsrc_nodes and layer_stats already added by compute methods)
        self.graph.graph['vsrc_dict'] = self.vsrc_dict
        self.graph.graph['parameters'] = self.parameters
        self.graph.graph['tile_grid'] = self.tile_grid
        self.graph.graph['instance_node_map'] = self.instance_node_map

        # Store instance_sources - either serialized (for pickle compatibility) or raw objects (for memory efficiency)
        if self.store_instance_sources:
            # Serialize for storage (backward compat, pickle-safe)
            instance_sources_serialized = {
                name: src.to_dict() for name, src in self.instance_sources.items()
            }
            self.graph.graph['instance_sources'] = instance_sources_serialized
        else:
            # Store raw CurrentSource objects directly (memory efficient, not pickle-safe)
            # Solvers can access via '_instance_sources_objects' key
            self.graph.graph['_instance_sources_objects'] = self.instance_sources
        self.graph.graph['merged_nodes'] = self.merged_nodes
        self.graph.graph['mutual_inductors'] = self.mutual_inductors
        self.graph.graph['net_connectivity'] = dict(net_connectivity)
        self.graph.graph['stats'] = {
            'nodes': self.stats.nodes_after_cleanup,
            'edges': self.graph.number_of_edges(),
            'resistors': self.stats.resistors,
            'capacitors': self.stats.capacitors,
            'inductors': self.stats.inductors,
            'vsources': self.stats.vsources,
            'isources': self.stats.isources,
            'mutual_inductors': self.stats.mutual_inductors,
            'boundary_nodes': self.stats.boundary_nodes,
            'package_nodes': self.stats.package_nodes,
            'vsrc_nodes': self.stats.vsrc_nodes,
            'tiles_parsed': self.stats.tiles_parsed,
            'tiles_failed': self.stats.tiles_failed,
            'unmapped_nodes': self.stats.unmapped_nodes,
            'instances_with_waveforms': self.stats.instances_with_waveforms,
            'total_static_current_ma': self.stats.total_static_current_ma
        }
        self.graph.graph['net_stats'] = net_stats_serializable

        self.logger.info(f"Graph finalized: {self.stats.nodes_after_cleanup} nodes, "
                        f"{self.graph.number_of_edges()} edges")

        # Memory optimization: Clear temporary data structures that are no longer needed
        # These are only used during parsing and net connectivity propagation
        self.node_net_map.clear()
        self.node_net_map_lower.clear()
        self.uf_parent.clear()
        self.uf_net.clear()
        self.package_edges.clear()
        # Clear node attributes cache
        self.node_attributes.clear()
        # Note: instance_node_map and instance_sources are NOT cleared here because
        # they are stored by reference in graph.graph. Clearing them would clear
        # the graph's data as well. The graph now owns these data structures.



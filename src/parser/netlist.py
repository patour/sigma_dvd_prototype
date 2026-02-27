"""
PDN Netlist Parser - facade module.

Contains NetlistParser (main parsing entry point), _PDNUnpickler
(legacy pickle resolution), and load_pdn_pickle() helper.
"""

import bisect
import gzip
import logging
import os
import pickle
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

from graph.rx_graph import RustworkxMultiDiGraphWrapper
from graph.rx_algorithms import contract_nodes, node_connected_component
from .edge_attrs import (
    create_element_edge,
    _get_net_type_index as _get_edge_net_type_index,
    reset_net_type_table as reset_edge_net_type_table,
    BaseElementEdge,
)

try:
    from tqdm import tqdm
except ImportError:
    class tqdm:
        def __init__(self, iterable=None, desc=None, total=None, **kwargs):
            self.iterable = iterable
            self.n = 0
            self.total = total
        def __iter__(self):
            return iter(self.iterable) if self.iterable else iter([])
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def update(self, n=1):
            self.n += n
        def set_description(self, desc):
            pass

from .spice_lexer import (
    GMAX, SHORT_THRESHOLD, INVALID_STATIC_CURRENT,
    R_TO_KOHM, C_TO_FF, L_TO_NH, I_TO_MA,
    _ELEM_R, _ELEM_C, _ELEM_L, _ELEM_V, _ELEM_I, _ELEM_TYPES,
    _KEY_TYPE, _KEY_VALUE, _KEY_ELEM_NAME, _KEY_TILE_ID, _KEY_NET_TYPE,
    _CAT_DIE, _CAT_PACKAGE, _CAT_UNMAPPED,
    _RE_SPICE_VALUE, _RE_PULSE, _RE_PWL, _RE_COORD_EXTRACT, _RE_TILE_FILE, _RE_INST_FILE,
    _SPICE_MULTIPLIERS, _parse_spice_value,
    SpiceLineReader,
    _check_net_filter, _fast_instance_net_filter, _has_structured_instance_names,
)

from .current_sources import (
    _apply_wscale, get_apply_wscale, set_apply_wscale,
    _optimize_dc_only, get_optimize_dc_only, set_optimize_dc_only,
    InstanceInfo, Pulse, PWL, CurrentSource, _DCOnlyCurrentSource,
    _parse_pulse, _parse_pwl, _parse_current_source_line,
    PreparedSource, _prepare_instance_source,
)

from .graph_builder import (
    _use_optimized_edges, get_use_optimized_edges, set_use_optimized_edges,
    GraphBuilder,
)

from .metadata import (
    _NET_TYPE_TABLE, _NET_TYPE_INDEX,
    _get_net_type_index, _reset_net_type_tables,
    _FLAG_BOUNDARY, _FLAG_PACKAGE, _FLAG_VSRC, _FLAG_DIE,
    PDNNodeAttrs, ParseStats,
)


class _PDNUnpickler(pickle.Unpickler):
    """Custom unpickler that resolves PDN classes from __main__ to pdn.pdn_parser.

    This handles legacy pickle files created when pdn_parser.py was run directly
    as __main__, which causes classes to be pickled with __main__ module reference.
    """

    def find_class(self, module: str, name: str):
        # Remap __main__ references to pdn.netlist (canonical) or pdn.pdn_parser
        if module == '__main__':
            import parser.netlist as netlist_module
            if hasattr(netlist_module, name):
                return getattr(netlist_module, name)
        return super().find_class(module, name)


def load_pdn_pickle(filepath: str):
    """Load a pickled PDN graph with proper class resolution.

    This function handles pickle files that contain PDNNodeAttrs,
    _DCOnlyCurrentSource, or other classes that were pickled from
    __main__ context (when running pdn_parser.py directly).

    Args:
        filepath: Path to the pickle file

    Returns:
        The unpickled object (typically RustworkxMultiDiGraphWrapper)

    Example:
        from parser.netlist import load_pdn_pickle
        graph = load_pdn_pickle('pdn_graph.pkl')
    """
    with open(filepath, 'rb') as f:
        return _PDNUnpickler(f).load()


# =============================================================================
# PDN Node Attributes & Parse Statistics - imported from parser.metadata
# =============================================================================
from .metadata import (  # noqa: F401 — re-exported for backward compatibility
    _NET_TYPE_TABLE, _NET_TYPE_INDEX,
    _get_net_type_index, _reset_net_type_tables,
    _FLAG_BOUNDARY, _FLAG_PACKAGE, _FLAG_VSRC, _FLAG_DIE,
    PDNNodeAttrs, ParseStats,
)

class NetlistParser:
    """
    Main parser for PDN netlists. Orchestrates the parsing process including
    tile-based parsing, subcircuit expansion, and validation.
    """
    
    def __init__(self, netlist_dir: str, validate: bool = False, strict: bool = False,
                 net_filter: Optional[str] = None, verbose: bool = False,
                 vsrc_resistor_pattern: str = 'rs', vsrc_depth_limit: int = 3,
                 store_instance_sources: bool = False,
                 parallel: bool = False, n_workers: Optional[int] = None,
                 chunk_size: int = 10000):
        """
        Initialize PDN netlist parser.

        Args:
            netlist_dir: Path to directory containing netlist files
            validate: Enable validation checks during parsing
            strict: Raise errors on validation failures (vs warnings)
            net_filter: Only parse elements for this net (e.g., 'VDD')
            verbose: Enable debug logging
            vsrc_resistor_pattern: Pattern for identifying voltage source resistors
            vsrc_depth_limit: Max depth for voltage source node traversal
            store_instance_sources: If True, serialize instance_sources to graph metadata
                                   (needed for pickling). If False (default), store raw
                                   CurrentSource objects for memory efficiency (~60% savings
                                   for large netlists with 1M+ sources).
            parallel: Enable parallel tile parsing using multiprocessing (default: False)
            n_workers: Number of parallel workers (default: min(cpu_count, 16))
            chunk_size: Lines per chunk for file reading in parallel mode (default: 10000)
        """
        self.netlist_dir = Path(netlist_dir)
        self.validate = validate
        self.strict = strict
        self.net_filter = net_filter
        self.vsrc_resistor_pattern = vsrc_resistor_pattern
        self.vsrc_depth_limit = vsrc_depth_limit
        self.store_instance_sources = store_instance_sources

        # Parallel parsing configuration
        self.parallel = parallel
        self.n_workers = n_workers or min(os.cpu_count() or 4, 16)
        self.chunk_size = chunk_size

        # Setup logging
        log_level = logging.DEBUG if verbose else logging.INFO
        logging.basicConfig(level=log_level,
                          format='%(levelname)s: %(message)s')
        self.logger = logging.getLogger(__name__)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = self.netlist_dir / f'pdn_parser_{timestamp}.log'
        if not any(isinstance(h, logging.FileHandler) for h in self.logger.handlers):
            file_handler = logging.FileHandler(log_file, mode='a')
            file_handler.setLevel(log_level)
            file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
            self.logger.addHandler(file_handler)
            self.logger.info(f"Logging to file: {log_file}")

        # Initialize graph builder
        self.builder = GraphBuilder(validate=validate, strict=strict, net_filter=net_filter,
                                    store_instance_sources=store_instance_sources,
                                    vsrc_resistor_pattern=vsrc_resistor_pattern)
        
        # Parsing state
        self.subcircuits: Dict[str, Dict] = {}  # name -> {pins: [...], body: [...]}
        self.tile_queue: List[Tuple[int, int, str]] = []  # (x, y, filepath)
        self.instance_queue: List[Tuple[int, int, str]] = []  # (x, y, filepath)
        self.include_stack: List[str] = []  # Track nested includes
        
        # Check for main netlist file
        self.main_netlist = self.netlist_dir / 'ckt.sp'
        if not self.main_netlist.exists():
            raise FileNotFoundError(f"Main netlist file not found: {self.main_netlist}")
            
    def parse(self) -> RustworkxMultiDiGraphWrapper:
        """
        Main parsing entry point. Returns populated rustworkx graph wrapper.
        """
        self.logger.info(f"Parsing PDN netlist from: {self.netlist_dir}")
        timings: Dict[str, float] = {}
        parse_start = time.perf_counter()
        
        try:
            # Parse main netlist file
            t0 = time.perf_counter()
            self._parse_file(str(self.main_netlist), is_main=True)
            timings["parse_main"] = time.perf_counter() - t0
            
            # Parse tiles if present
            if self.tile_queue:
                t0 = time.perf_counter()
                self._parse_tiles()
                timings["parse_tiles"] = time.perf_counter() - t0
                
            # Parse instance models
            if self.instance_queue:
                t0 = time.perf_counter()
                self._parse_instance_models()
                timings["parse_instance_models"] = time.perf_counter() - t0
            
            # Propagate net connectivity through package elements
            # This handles cases where package elements were parsed before die elements
            t0 = time.perf_counter()
            self._propagate_net_connectivity()
            timings["propagate_net_connectivity"] = time.perf_counter() - t0
            
            # Update per-net statistics for package elements now that net types are known
            t0 = time.perf_counter()
            self._update_package_statistics()
            timings["update_package_statistics"] = time.perf_counter() - t0
            
            # Filter voltage sources based on connectivity to filtered net
            t0 = time.perf_counter()
            self._filter_voltage_sources_by_net()
            timings["filter_voltage_sources_by_net"] = time.perf_counter() - t0
            
            # Identify voltage source nodes (must be before validation for floating node check)
            t0 = time.perf_counter()
            self._identify_vsrc_nodes()
            timings["identify_vsrc_nodes"] = time.perf_counter() - t0
            
            # Compute layer statistics
            t0 = time.perf_counter()
            self._compute_layer_stats()
            timings["compute_layer_stats"] = time.perf_counter() - t0
                
            # Perform validation if requested (after vsrc node identification)
            if self.validate:
                t0 = time.perf_counter()
                self._perform_validation()
                timings["perform_validation"] = time.perf_counter() - t0
                
            # Validate node uniqueness
            t0 = time.perf_counter()
            self.builder.validate_node_uniqueness()
            timings["validate_node_uniqueness"] = time.perf_counter() - t0
            
            # Finalize graph
            t0 = time.perf_counter()
            self.builder.finalize()
            timings["finalize"] = time.perf_counter() - t0
            
            # Print statistics
            t0 = time.perf_counter()
            self._print_statistics()
            timings["print_statistics"] = time.perf_counter() - t0

            timings["total_parse"] = time.perf_counter() - parse_start

            self.logger.info("Parse timing breakdown (s):")
            for key in sorted(timings.keys()):
                self.logger.info(f"  {key}: {timings[key]:.4f}")
            
            return self.builder.graph
            
        except Exception as e:
            self.logger.error(f"Parsing failed: {e}")
            if self.strict:
                raise
            return self.builder.graph
            
    def _parse_file(self, filepath: str, is_main: bool = False):
        """Parse a single SPICE file"""
        self.logger.debug(f"Parsing file: {filepath}")
        self.include_stack.append(filepath)
        
        try:
            with SpiceLineReader(filepath) as reader:
                while True:
                    line = reader.read_line()
                    if line is None:
                        break
                        
                    # Process line
                    try:
                        self._process_line(line, filepath)
                    except Exception as e:
                        msg = f"Error parsing line {reader.line_number} in {filepath}: {e}\n  Line: {line}"
                        if self.strict:
                            raise RuntimeError(msg) from e
                        else:
                            self.logger.warning(msg)
                            
        finally:
            self.include_stack.pop()
            
    def _process_line(self, line: str, source_file: str):
        """Process a single logical line from netlist"""
        if not line:
            return
            
        # Get first token (case-insensitive)
        tokens = line.split()
        if not tokens:
            return
            
        first_token = tokens[0]
        first_char = first_token[0].upper()
        
        # Dot commands
        if first_char == '.':
            self._process_dot_command(line, source_file)
        # Circuit elements - check for lowercase single char (simplified format) or standard format
        elif first_token == 'r':  # Exactly 'r' for simplified format
            self._parse_resistor(line)
        elif first_char == 'R' or (first_token.startswith('r') and len(first_token) > 1):
            # R<name> or r<name> format
            self._parse_resistor(line)
        elif first_token == 'c':  # Exactly 'c' for simplified format
            self._parse_capacitor(line)
        elif first_char == 'C' or (first_token.startswith('c') and len(first_token) > 1):
            # C<name> or c<name> format
            self._parse_capacitor(line)
        elif first_token == 'l':  # Exactly 'l' for simplified format
            self._parse_inductor(line)
        elif first_char == 'L' or (first_token.startswith('l') and len(first_token) > 1):
            # L<name> or l<name> format
            self._parse_inductor(line)
        elif first_char == 'K':
            self._parse_mutual_inductor(line)
        elif first_char == 'V':
            self._parse_vsource(line)
        elif first_char == 'I':
            self._parse_isource(line)
        elif first_char in ['E', 'F', 'G', 'H']:
            self._parse_controlled_source(line, first_char)
        elif first_char == 'X':
            self._parse_subcircuit_instance(line)
        elif first_char == 'M':
            self.logger.debug(f"Skipping transistor: {tokens[0]}")
        elif first_char == 'P':
            self.logger.debug(f"Skipping power gate: {tokens[0]}")
        else:
            self.logger.debug(f"Unknown element type: {first_char} in line: {line[:50]}")
            
    def _process_dot_command(self, line: str, source_file: str):
        """Process dot commands (.include, .subckt, .parameter, etc.)"""
        tokens = line.split()
        cmd = tokens[0].lower()
        
        if cmd == '.partition_info' and len(tokens) >= 3:
            # .partition_info N M
            n, m = int(tokens[1]), int(tokens[2])
            self.builder.tile_grid = (n, m)
            self.logger.info(f"Detected tile grid: {n} x {m}")
            
        elif cmd == '.include' and len(tokens) >= 2:
            include_file = tokens[1]
            self._process_include(include_file, source_file)
            
        elif cmd == '.parameter' or cmd == '.param':
            # .parameter name=value or .parameter name value
            if len(tokens) == 3 and '=' not in tokens[1]:
                # Space-separated format: .parameter VDD 0.75
                name, value = tokens[1], tokens[2]
                self.builder.parameters[name.strip().upper()] = value.strip()
            else:
                # Equals-separated format: .parameter VDD=0.75
                for token in tokens[1:]:
                    if '=' in token:
                        name, value = token.split('=', 1)
                        self.builder.parameters[name.strip().upper()] = value.strip()
                    
        elif cmd == '.subckt':
            self._parse_subcircuit_definition(line)
            
        elif cmd == '.ends':
            pass  # Handled in _parse_subcircuit_definition
            
        elif cmd == '.flag_boundary':
            # Mark following nodes as boundary nodes (handled during parsing)
            pass
            
        elif cmd in ['.print', '.tran', '.ac', '.dc', '.die_area', '.model']:
            # Analysis commands - store but don't process
            self.logger.debug(f"Skipping command: {cmd}")
            
    def _process_include(self, include_path: str, source_file: str):
        """Process .include directive"""
        # Resolve path relative to source file directory
        source_dir = Path(source_file).parent
        full_path = source_dir / include_path
        
        if not full_path.exists():
            # Try relative to netlist_dir
            full_path = self.netlist_dir / include_path
            
        if not full_path.exists():
            msg = f"Include file not found: {include_path}"
            if self.strict:
                raise FileNotFoundError(msg)
            else:
                self.logger.warning(msg)
                return
                
        # Check for tile or instance model files
        filename = full_path.name
        
        # Pattern: tile_X_Y.ckt or tile_X_Y.sp
        tile_match = _RE_TILE_FILE.match(filename)
        if tile_match:
            x, y = int(tile_match.group(1)), int(tile_match.group(2))
            self.tile_queue.append((x, y, str(full_path)))
            self.logger.debug(f"Queued tile {x}_{y}: {full_path}")
            return

        # Pattern: instanceModels_X_Y.sp
        inst_match = _RE_INST_FILE.match(filename)
        if inst_match:
            x, y = int(inst_match.group(1)), int(inst_match.group(2))
            self.instance_queue.append((x, y, str(full_path)))
            self.logger.debug(f"Queued instance models {x}_{y}: {full_path}")
            return
            
        # Check for package files
        if 'package' in filename.lower():
            old_file_type = self.builder.current_file_type
            self.builder.current_file_type = 'package'
            self.builder.stats.package_nodes += 1
            self._parse_file(str(full_path))
            self.builder.current_file_type = old_file_type
        else:
            # Regular include
            self._parse_file(str(full_path))
            
    def _load_node_net_map(self, nd_filepath: str) -> Dict[str, str]:
        """
        Load node-to-net mapping from .nd file.
        Format: <node_name> <val1> <val2> <val3> <val4> <net_name>
        """
        node_map = {}
        
        try:
            with SpiceLineReader(nd_filepath) as reader:
                while True:
                    line = reader.read_line()
                    if line is None:
                        break
                    
                    tokens = line.split()
                    if len(tokens) >= 6:
                        node_name = tokens[0]
                        net_name = tokens[5]
                        node_map[node_name] = net_name
                        # Also store lowercase version for case-insensitive filtering
                        self.builder.node_net_map_lower[node_name] = net_name.lower()
                    elif len(tokens) > 0:
                        self.logger.warning(f"Invalid .nd line (expected 6 tokens, got {len(tokens)}): {line}")
        except FileNotFoundError:
            raise FileNotFoundError(f"Required .nd file not found: {nd_filepath}")
        except Exception as e:
            raise RuntimeError(f"Error reading .nd file {nd_filepath}: {e}")
        
        return node_map
    
    def _propagate_net_connectivity(self):
        """
        Process deferred package edges using union-find to efficiently trace connectivity.
        Die nodes from .nd files serve as anchors with explicit net types.
        This also assigns net types to all package nodes based on their connectivity to die nodes.
        """
        if not self.builder.package_edges:
            return
        
        self.logger.info(f"Processing {len(self.builder.package_edges)} package edges with union-find...")
        
        # Process all package edges with union-find
        for node1, node2 in self.builder.package_edges:
            self.builder._uf_union(node1, node2)
        
        # After union-find, assign net types to all package nodes based on their root
        for node in self.builder.graph.nodes():
            if node not in self.builder.node_net_map and node != '0':
                # This is a package node - get its net type from union-find
                net_type = self.builder._get_node_net(node)
                if net_type:
                    # Update node attributes and lowercase map for filtering
                    self.builder.graph.nodes_dict[node]['net_type'] = net_type
                    self.builder.node_net_map_lower[node] = net_type.lower()
        
        self.logger.debug(f"Union-find processing complete - net types propagated to package nodes")
    
    def _update_package_statistics(self):
        """
        Update per-net statistics for package elements after net types have been propagated.
        This is called after _propagate_net_connectivity assigns net types to package nodes.
        """
        self.logger.debug("Updating package element statistics...")
        
        # Iterate through all edges and update statistics for package elements
        for u, v, d in self.builder.graph.edges(data=True):
            elem_type = d.get('type')
            value = d.get('value', 0.0)
            
            # Check if this is a package element
            u_is_die = u in self.builder.node_net_map
            v_is_die = v in self.builder.node_net_map
            u_is_ground = u == '0'
            v_is_ground = v == '0'
            u_is_package = not u_is_die and not u_is_ground
            v_is_package = not v_is_die and not v_is_ground
            
            # Determine if this is a package element:
            # 1. Voltage sources: at least one package node (can connect to ground or die)
            # 2. Other elements (R,C,L): at least one package node
            if elem_type == 'V':
                # Package voltage source: at least one terminal is package node
                if not (u_is_package or v_is_package):
                    continue
            else:
                # For R,C,L: include if at least one node is a package node
                # This includes die-to-package connections (like "rs" resistors)
                if not (u_is_package or v_is_package):
                    continue
            
            # This is a package element - get its net type from the propagated data
            net_type = d.get('net_type')
            if not net_type:
                # Try to get from node attributes (set by union-find)
                u_net = self.builder.graph.nodes_dict[u].get('net_type') if u != '0' else None
                v_net = self.builder.graph.nodes_dict[v].get('net_type') if v != '0' else None
                net_type = u_net or v_net
            
            if net_type:
                # Initialize net stats if needed
                if net_type not in self.builder.stats.net_stats:
                    self.builder.stats.net_stats[net_type] = {
                        'die': {
                            'nodes': set(),
                            'resistors': 0, 'capacitors': 0, 'inductors': 0,
                            'vsources': 0, 'isources': 0,
                            'total_resistance': 0.0, 'total_capacitance': 0.0,
                            'total_inductance': 0.0, 'total_current': 0.0
                        },
                        'package': {
                            'nodes': set(),
                            'resistors': 0, 'capacitors': 0, 'inductors': 0,
                            'vsources': 0, 'isources': 0,
                            'total_resistance': 0.0, 'total_capacitance': 0.0,
                            'total_inductance': 0.0, 'total_current': 0.0
                        }
                    }
                
                # Add to package statistics (only package nodes, not die nodes)
                net_stat = self.builder.stats.net_stats[net_type]['package']
                if u != '0' and u not in self.builder.node_net_map:
                    net_stat['nodes'].add(u)
                if v != '0' and v not in self.builder.node_net_map:
                    net_stat['nodes'].add(v)
                
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
        
        # Print package node names in verbose mode
        if self.logger.isEnabledFor(logging.DEBUG):
            for net_type, net_categories in self.builder.stats.net_stats.items():
                if net_type == 'unmapped':
                    continue
                pkg_nodes = net_categories.get('package', {}).get('nodes', set())
                if pkg_nodes:
                    self.logger.debug(f"Package nodes for net {net_type}: {sorted(pkg_nodes)}")
        
        self.logger.debug("Package element statistics updated")
    
    def _filter_voltage_sources_by_net(self):
        """
        Remove voltage sources and package elements that don't connect to the filtered net.
        This is called after _propagate_net_connectivity has assigned net types to package nodes.

        Strategy: Use tracked edge indices from parsing to avoid iterating over all edges.
        Only voltage sources and package elements (from package.ckt) need to be checked.
        """
        if self.builder.net_filter is None:
            return

        self.logger.info(f"Filtering voltage sources and package elements by net '{self.builder.net_filter}'...")
        self.logger.debug(f"Checking {len(self.builder.vsrc_edge_indices)} vsrc edges, "
                         f"{len(self.builder.package_edge_indices)} package edges")

        # Access internal rustworkx graph for efficient index-based operations
        rx_graph = self.builder.graph._graph
        idx_to_node = self.builder.graph._idx_to_node

        # Collect edge indices to remove and statistics updates
        edges_to_remove = []
        vsrc_remove_count = 0
        pkg_remove_count = 0
        stats_updates = {'R': 0, 'C': 0, 'L': 0, 'V': 0}
        per_net_updates = []  # [(net_type, elem_type, value, u, v), ...]

        # Check voltage source edges (only iterate over tracked vsrc indices)
        for edge_idx in self.builder.vsrc_edge_indices:
            try:
                u_idx, v_idx = rx_graph.get_edge_endpoints_by_index(edge_idx)
                u = idx_to_node[u_idx]
                v = idx_to_node[v_idx]
            except Exception:
                # Edge may have been removed by earlier processing
                continue

            # Get positive terminal (non-ground node)
            vsrc_pos_node = u if u != '0' else v
            if vsrc_pos_node == '0':
                # Both terminals grounded - remove
                edges_to_remove.append(edge_idx)
                vsrc_remove_count += 1
                stats_updates['V'] += 1
                continue

            # Check if positive terminal has the filtered net type
            node_net_lower = self.builder.node_net_map_lower.get(vsrc_pos_node)
            if node_net_lower != self.builder.net_filter:
                edges_to_remove.append(edge_idx)
                vsrc_remove_count += 1
                stats_updates['V'] += 1

        # Check package edges (only iterate over tracked package indices)
        # Skip edges already marked for removal (vsrc edges that are also in package)
        edges_to_remove_set = set(edges_to_remove)

        for edge_idx in self.builder.package_edge_indices:
            if edge_idx in edges_to_remove_set:
                continue  # Already marked for removal as vsrc

            try:
                u_idx, v_idx = rx_graph.get_edge_endpoints_by_index(edge_idx)
                u = idx_to_node[u_idx]
                v = idx_to_node[v_idx]
                edge_data = rx_graph.get_edge_data_by_index(edge_idx)
            except Exception:
                continue

            # Package edges: both nodes are not in die node map
            if u not in self.builder.node_net_map and u != '0' and \
               v not in self.builder.node_net_map and v != '0':
                u_net_lower = self.builder.node_net_map_lower.get(u)
                v_net_lower = self.builder.node_net_map_lower.get(v)

                # Remove if neither node has the filtered net type
                if u_net_lower != self.builder.net_filter and v_net_lower != self.builder.net_filter:
                    edges_to_remove.append(edge_idx)
                    pkg_remove_count += 1

                    elem_type = edge_data.get('type') if edge_data else None
                    if elem_type and elem_type in stats_updates:
                        stats_updates[elem_type] += 1

                    # Track per-net statistics update
                    net_type = edge_data.get('net_type') if edge_data else None
                    value = edge_data.get('value', 0.0) if edge_data else 0.0
                    if net_type:
                        per_net_updates.append((net_type, elem_type, value, u, v))

        # Batch remove edges (sort descending to avoid index invalidation issues)
        for edge_idx in sorted(edges_to_remove, reverse=True):
            try:
                rx_graph.remove_edge_from_index(edge_idx)
            except Exception:
                pass  # Edge already removed

        # Update global statistics
        self.builder.stats.vsources -= stats_updates['V']
        self.builder.stats.resistors -= stats_updates['R']
        self.builder.stats.capacitors -= stats_updates['C']
        self.builder.stats.inductors -= stats_updates['L']
        self.builder.stats.elements_total -= len(edges_to_remove)

        # Update per-net statistics
        for net_type, elem_type, value, u, v in per_net_updates:
            if net_type in self.builder.stats.net_stats:
                net_stat = self.builder.stats.net_stats[net_type].get('package')
                if net_stat:
                    net_stat['nodes'].discard(u)
                    net_stat['nodes'].discard(v)
                    if elem_type == 'R':
                        net_stat['resistors'] -= 1
                        net_stat['total_resistance'] -= value
                    elif elem_type == 'C':
                        net_stat['capacitors'] -= 1
                        net_stat['total_capacitance'] -= value
                    elif elem_type == 'L':
                        net_stat['inductors'] -= 1
                        net_stat['total_inductance'] -= value

        # Remove isolated package nodes (only check package nodes, not all nodes)
        # Build set of package nodes from the tracked edges for efficiency
        package_nodes_to_check = set()
        for edge_idx in self.builder.package_edge_indices:
            try:
                u_idx, v_idx = rx_graph.get_edge_endpoints_by_index(edge_idx)
                u = idx_to_node.get(u_idx)
                v = idx_to_node.get(v_idx)
                if u and u not in self.builder.node_net_map and u != '0':
                    package_nodes_to_check.add(u)
                if v and v not in self.builder.node_net_map and v != '0':
                    package_nodes_to_check.add(v)
            except Exception:
                pass

        package_nodes_to_remove = []
        for node in package_nodes_to_check:
            if node in self.builder.graph and self.builder.graph.degree(node) == 0:
                package_nodes_to_remove.append(node)

        for node in package_nodes_to_remove:
            self.builder.graph.remove_node(node)

        self.logger.info(f"Filtered out {vsrc_remove_count} voltage sources, "
                        f"{pkg_remove_count} package edges, {len(package_nodes_to_remove)} package nodes")
    
    def _parse_tiles(self):
        """Parse all queued tile files with progress bar"""
        if not self.tile_queue:
            return

        if self.parallel and len(self.tile_queue) > 1:
            self._parse_tiles_parallel()
        else:
            self._parse_tiles_sequential()

    def _parse_tiles_sequential(self):
        """Parse tiles sequentially (original implementation)"""
        self.logger.info(f"Parsing {len(self.tile_queue)} tile files (sequential)...")

        with tqdm(total=len(self.tile_queue), desc="Parsing tiles") as pbar:
            for x, y, filepath in self.tile_queue:
                pbar.set_description(f"Parsing tile {x}_{y}")

                try:
                    self.builder.current_tile_id = (x, y)

                    # Load corresponding .nd file for node-to-net mapping
                    nd_filepath = Path(filepath).parent / f"tile_{x}_{y}.nd"
                    if not nd_filepath.exists():
                        # Try with .gz extension
                        nd_filepath = Path(filepath).parent / f"tile_{x}_{y}.nd.gz"

                    self.logger.debug(f"Loading node map from {nd_filepath}")
                    tile_node_map = self._load_node_net_map(str(nd_filepath))
                    self.builder.node_net_map.update(tile_node_map)
                    self.logger.debug(f"Loaded {len(tile_node_map)} node mappings for tile {x}_{y}")

                    # Now parse the tile netlist
                    self._parse_file(filepath)
                    self.builder.stats.tiles_parsed += 1
                except Exception as e:
                    self.builder.stats.tiles_failed += 1
                    msg = f"Failed to parse tile {x}_{y}: {e}"
                    if self.strict:
                        raise RuntimeError(msg) from e
                    else:
                        self.logger.warning(msg)
                finally:
                    pbar.update(1)

        self.builder.current_tile_id = None

    def _parse_tiles_parallel(self):
        """Parse tiles in parallel using multiprocessing.Pool"""
        import multiprocessing
        from concurrent.futures import ThreadPoolExecutor
        from parser.parallel import (
            _parse_tile_worker, TileParseResult,
            _build_tile_worker_args
        )

        self.logger.info(f"Parsing {len(self.tile_queue)} tile files (parallel, {self.n_workers} workers)...")

        # Build worker arguments
        net_filter_lower = self.net_filter.lower() if self.net_filter else None
        worker_args = _build_tile_worker_args(
            self.tile_queue,
            self.netlist_dir,
            net_filter_lower,
            self.chunk_size,
            {'strict': self.strict}
        )

        # Execute in parallel using multiprocessing Pool
        results: List[TileParseResult] = []
        with multiprocessing.Pool(processes=self.n_workers) as pool:
            with tqdm(total=len(worker_args), desc="Parsing tiles") as pbar:
                for result in pool.imap_unordered(_parse_tile_worker, worker_args):
                    results.append(result)
                    pbar.update(1)
                    pbar.set_description(f"Parsed tile {result.tile_id}")

                    # Log any errors
                    for error in result.errors:
                        if self.strict:
                            raise RuntimeError(error)
                        self.logger.warning(error)
                    for warning in result.warnings:
                        self.logger.warning(warning)

        # Merge results into builder
        self._merge_tile_results_parallel(results)

    def _merge_tile_results_parallel(self, results: List):
        """Merge tile results using bulk graph operations."""
        from concurrent.futures import ThreadPoolExecutor
        from parser.parallel import TileParseResult

        self.logger.info(f"Merging {len(results)} tile results...")

        # Phase 1: Collect all data from results
        all_nodes = []  # [(name, attrs), ...]
        all_edges = []  # [(u, v, attrs), ...]
        all_node_net_map = {}
        all_node_net_map_lower = {}
        total_stats = defaultdict(int)

        for result in results:
            # Update node-net mappings
            all_node_net_map.update(result.node_net_map)
            for node, net in result.node_net_map.items():
                all_node_net_map_lower[node] = net.lower()

            # Collect nodes
            for node_name, node_attrs in result.nodes.items():
                # Add net type from the mapping
                net_type = result.node_net_map.get(node_name)
                if net_type:
                    node_attrs['net_type'] = net_type
                all_nodes.append((node_name, node_attrs))

            # Collect edges
            for node1, node2, elem_type, value, name, attrs in result.edges:
                # Get net_type from attrs if present
                net_type = attrs.get('net_type')

                if get_use_optimized_edges():
                    # Determine if this resistor needs elem_name
                    needs_elem_name = (
                        elem_type == 'V' or
                        (elem_type == 'R' and name.lower() == self.vsrc_resistor_pattern.lower())
                    )
                    # Create optimized edge object
                    edge_obj = create_element_edge(
                        elem_type=elem_type,
                        value=value,
                        elem_name=name if needs_elem_name else None,
                        tile_id=result.tile_id,
                        net_type=net_type,
                        needs_elem_name=needs_elem_name,
                    )
                    all_edges.append((node1, node2, edge_obj))
                else:
                    # Legacy dict-based edge attributes
                    edge_attrs = {
                        'type': elem_type,
                        'value': value,
                        'tile_id': result.tile_id,
                        **attrs
                    }
                    # Only store elem_name for R and V types to save memory
                    if elem_type in ('R', 'V'):
                        edge_attrs['elem_name'] = name
                    all_edges.append((node1, node2, edge_attrs))
                total_stats[elem_type] += 1

            # Track boundary nodes
            for boundary_node in result.boundary_nodes:
                self.builder.boundary_nodes.add(boundary_node)

            # Accumulate statistics
            self.builder.stats.tiles_parsed += 1
            if result.errors:
                self.builder.stats.tiles_failed += 1

        # Phase 2: Update builder state
        self.builder.node_net_map.update(all_node_net_map)
        self.builder.node_net_map_lower.update(all_node_net_map_lower)

        # Phase 3: Bulk insert into graph
        self.logger.info(f"Bulk inserting {len(all_nodes)} nodes, {len(all_edges)} edges...")

        # Add nodes with attributes
        for node_name, node_attrs in all_nodes:
            if node_name not in self.builder.graph:
                # Determine if this is a package node
                is_package_node = (node_name not in self.builder.node_net_map and
                                   node_name != '0')

                # Build flags for compact storage
                flags = 0
                if node_name in self.builder.boundary_nodes:
                    flags |= PDNNodeAttrs.FLAG_BOUNDARY
                if is_package_node:
                    flags |= PDNNodeAttrs.FLAG_PACKAGE
                # Check if die node pattern
                if not is_package_node and node_name != '0':
                    parts = node_name.split('_')
                    if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
                        flags |= PDNNodeAttrs.FLAG_DIE

                # Create compact node attributes
                attrs_obj = PDNNodeAttrs(
                    name=node_name,
                    tile_id=node_attrs.get('tile_id'),
                    flags=flags
                )
                # Apply net_type from merged dict if present
                if 'net_type' in node_attrs:
                    attrs_obj.net_type = node_attrs['net_type']

                self.builder.graph.add_node(node_name, attrs_obj)
                # No longer need to call _extract_coordinates - x, y, layer computed from name
            else:
                # Update existing node
                self.builder.graph.nodes_dict[node_name].update(node_attrs)

        # Add edges with attributes
        edge_indices = self.builder.graph.add_edges_from(all_edges)

        # Update global statistics
        self.builder.stats.resistors += total_stats.get('R', 0)
        self.builder.stats.capacitors += total_stats.get('C', 0)
        self.builder.stats.inductors += total_stats.get('L', 0)
        self.builder.stats.vsources += total_stats.get('V', 0)
        self.builder.stats.isources += total_stats.get('I', 0)
        self.builder.stats.elements_total += sum(total_stats.values())

        # Track voltage source edge indices
        for i, (u, v, attrs) in enumerate(all_edges):
            if attrs.get('type') == 'V':
                self.builder.vsrc_edge_indices.append(edge_indices[i])

        # Update per-net statistics
        self._update_parallel_net_statistics(all_edges, all_node_net_map)

        self.logger.info(f"Tile merge complete: {self.builder.stats.tiles_parsed} tiles, "
                        f"{len(all_edges)} elements")

    def _update_parallel_net_statistics(self, edges: List, node_net_map: Dict):
        """Update per-net statistics from parallel parsing results."""
        for node1, node2, attrs in edges:
            elem_type = attrs.get('type')
            value = attrs.get('value', 0.0)
            net_type = attrs.get('net_type')

            if not net_type:
                net_type = node_net_map.get(node1) or node_net_map.get(node2)

            if net_type:
                if net_type not in self.builder.stats.net_stats:
                    self.builder.stats.net_stats[net_type] = {
                        'die': {
                            'nodes': set(), 'resistors': 0, 'capacitors': 0,
                            'inductors': 0, 'vsources': 0, 'isources': 0,
                            'isources_with_waveforms': 0, 'wscale_values': [],
                            'total_resistance': 0.0, 'total_capacitance': 0.0,
                            'total_inductance': 0.0, 'total_current': 0.0
                        },
                        'package': {
                            'nodes': set(), 'resistors': 0, 'capacitors': 0,
                            'inductors': 0, 'vsources': 0, 'isources': 0,
                            'isources_with_waveforms': 0, 'wscale_values': [],
                            'total_resistance': 0.0, 'total_capacitance': 0.0,
                            'total_inductance': 0.0, 'total_current': 0.0
                        },
                        'unmapped': {
                            'nodes': set(), 'resistors': 0, 'capacitors': 0,
                            'inductors': 0, 'vsources': 0, 'isources': 0,
                            'isources_with_waveforms': 0, 'wscale_values': [],
                            'total_resistance': 0.0, 'total_capacitance': 0.0,
                            'total_inductance': 0.0, 'total_current': 0.0
                        }
                    }

                # Tiles are always 'die' category
                net_stat = self.builder.stats.net_stats[net_type]['die']
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
        
    def _parse_instance_models(self):
        """Parse instance model files (current sources)"""
        if not self.instance_queue:
            return

        if self.parallel and len(self.instance_queue) > 1:
            self._parse_instance_models_parallel()
        else:
            self._parse_instance_models_sequential()

    def _parse_instance_models_sequential(self):
        """Parse instance models sequentially (original implementation)"""
        self.logger.info(f"Parsing {len(self.instance_queue)} instance model files (sequential)...")

        # Detect structured names from first instance file for fast net filtering
        if self.net_filter and self.instance_queue:
            _, _, first_path = self.instance_queue[0]
            self._use_fast_instance_filter = _has_structured_instance_names(first_path)
        else:
            self._use_fast_instance_filter = False

        old_file_type = self.builder.current_file_type
        self.builder.current_file_type = 'instance'

        with tqdm(total=len(self.instance_queue), desc="Parsing instance models") as pbar:
            for x, y, filepath in self.instance_queue:
                pbar.set_description(f"Parsing instances {x}_{y}")

                try:
                    self.builder.current_tile_id = (x, y)
                    self._parse_file(filepath)
                except Exception as e:
                    msg = f"Failed to parse instance models {x}_{y}: {e}"
                    if self.strict:
                        raise RuntimeError(msg) from e
                    else:
                        self.logger.warning(msg)
                finally:
                    pbar.update(1)

        self.builder.current_file_type = old_file_type
        self.builder.current_tile_id = None

    def _parse_instance_models_parallel(self):
        """Parse instance models in parallel using multiprocessing.Pool"""
        import multiprocessing
        from parser.parallel import (
            _parse_instance_worker, InstanceParseResult,
            _build_instance_worker_args
        )

        self.logger.info(f"Parsing {len(self.instance_queue)} instance model files (parallel, {self.n_workers} workers)...")

        # Build worker arguments
        net_filter_lower = self.net_filter.lower() if self.net_filter else None
        worker_args = _build_instance_worker_args(
            self.instance_queue,
            net_filter_lower,
            self.chunk_size,
            {'strict': self.strict}
        )

        # Execute in parallel using multiprocessing Pool
        results: List[InstanceParseResult] = []
        with multiprocessing.Pool(processes=self.n_workers) as pool:
            with tqdm(total=len(worker_args), desc="Parsing instance models") as pbar:
                for result in pool.imap_unordered(_parse_instance_worker, worker_args):
                    results.append(result)
                    pbar.update(1)
                    pbar.set_description(f"Parsed instances {result.tile_id}")

                    # Log any warnings
                    for warning in result.warnings:
                        self.logger.warning(warning)

        # Merge results
        self._merge_instance_results_parallel(results)

    def _merge_instance_results_parallel(self, results: List):
        """Merge instance model results from parallel parsing."""
        from parser.parallel import InstanceParseResult

        self.logger.info(f"Merging {len(results)} instance model results...")

        total_instances = 0
        total_with_waveforms = 0
        total_static_current = 0.0

        for result in results:
            # Add current sources to builder
            for name, isrc_dict in result.current_sources.items():
                # Reconstruct CurrentSource object
                isrc = CurrentSource.from_dict(isrc_dict)
                nodes = result.instance_node_map.get(name, ['', ''])
                node_pos, node_neg = nodes[0], nodes[1]

                # Check net filter
                if self.net_filter:
                    node_pos_net = self.builder.node_net_map_lower.get(node_pos)
                    node_neg_net = self.builder.node_net_map_lower.get(node_neg)
                    if node_pos_net != self.net_filter.lower() and node_neg_net != self.net_filter.lower():
                        continue

                # Store in builder
                self.builder.instance_node_map[name] = nodes
                self.builder.instance_sources[name] = isrc

                # Add edge to graph if nodes exist
                if node_pos in self.builder.graph or node_neg in self.builder.graph:
                    static_current_ma = isrc.get_static_current()
                    # Note: elem_name not stored for I-type edges to save memory
                    attrs = {
                        'type': 'I',
                        'value': static_current_ma,
                        'tile_id': result.tile_id,
                        'dc': static_current_ma,
                        'has_waveform': isrc.has_waveform_data(),
                        'net_type': self.builder.node_net_map.get(node_pos) or self.builder.node_net_map.get(node_neg)
                    }

                    # Ensure nodes exist
                    self.builder.add_node(node_pos)
                    self.builder.add_node(node_neg)
                    self.builder.graph.add_edge(node_pos, node_neg, **attrs)

                    self.builder.stats.isources += 1
                    self.builder.stats.elements_total += 1

                    if isrc.has_waveform_data():
                        total_with_waveforms += 1

                        # Update per-net waveform statistics
                        net_type = self.builder.node_net_map.get(node_pos) or self.builder.node_net_map.get(node_neg)
                        if net_type and net_type in self.builder.stats.net_stats:
                            if 'die' in self.builder.stats.net_stats[net_type]:
                                die_stats = self.builder.stats.net_stats[net_type]['die']
                                die_stats['isources_with_waveforms'] += 1
                                if 'wscale_values' not in die_stats:
                                    die_stats['wscale_values'] = []
                                die_stats['wscale_values'].append(isrc.wscale)

                    total_static_current += abs(static_current_ma)

                total_instances += 1

        # Statistics are already accumulated in the per-instance loop above
        self.builder.stats.instances_with_waveforms = total_with_waveforms
        self.builder.stats.total_static_current_ma = total_static_current

        self.logger.info(f"Instance merge complete: {total_instances} current sources, "
                        f"{total_with_waveforms} with waveforms")
        
    def _parse_resistor(self, line: str):
        """Parse resistor: R<name> <node1> <node2> <value> OR r <node1> <node2> <value>"""
        tokens = line.split()
        if len(tokens) < 3:
            self.logger.warning(f"Invalid resistor line: {line}")
            return
        
        # Check format: 'r <node1> <node2> <value>' (no name) or 'R<name> <node1> <node2> <value>'
        first_token = tokens[0]
        if first_token.lower() == 'r':
            # Format: r <node1> <node2> <value>
            if len(tokens) < 4:
                self.logger.warning(f"Invalid resistor line: {line}")
                return
            # Generate unique name
            name = f"r_{tokens[1]}_{tokens[2]}_{id(line)}"
            node1 = tokens[1]
            node2 = tokens[2]
            value_token = tokens[3]
        else:
            # Format: R<name> <node1> <node2> <value>
            if len(tokens) < 4:
                self.logger.warning(f"Invalid resistor line: {line}")
                return
            name = tokens[0]
            node1 = tokens[1]
            node2 = tokens[2]
            value_token = tokens[3]
        
        # Handle boundary nodes (marked with *)
        node1_is_boundary = node1.startswith('*')
        node2_is_boundary = node2.startswith('*')
        
        if node1_is_boundary:
            node1 = node1[1:]  # Remove *
            self.builder.mark_boundary_node(node1)
        if node2_is_boundary:
            node2 = node2[1:]
            self.builder.mark_boundary_node(node2)
            
        try:
            value = self._parse_value(value_token)
            # Convert to KOhm
            value_kohm = value * R_TO_KOHM
            self.builder.add_element('R', node1, node2, value_kohm, name)
        except ValueError as e:
            self.logger.warning(f"Error parsing resistor value in line: {line}: {e}")
            
    def _parse_capacitor(self, line: str):
        """Parse capacitor: C<name> <node1> <node2> <value> [model] OR c <node1> <node2> <value> [model]"""
        tokens = line.split()
        if len(tokens) < 3:
            self.logger.warning(f"Invalid capacitor line: {line}")
            return
        
        # Check format: 'c <node1> <node2> <value>' (no name) or 'C<name> <node1> <node2> <value>'
        first_token = tokens[0]
        if first_token.lower() == 'c':
            # Format: c <node1> <node2> <value> [model]
            if len(tokens) < 4:
                self.logger.warning(f"Invalid capacitor line: {line}")
                return
            # Generate unique name
            name = f"c_{tokens[1]}_{tokens[2]}_{id(line)}"
            node1 = tokens[1]
            node2 = tokens[2]
            value_token = tokens[3]
            model_idx = 4
        else:
            # Format: C<name> <node1> <node2> <value> [model]
            if len(tokens) < 4:
                self.logger.warning(f"Invalid capacitor line: {line}")
                return
            name = tokens[0]
            node1 = tokens[1]
            node2 = tokens[2]
            value_token = tokens[3]
            model_idx = 4
        
        # Handle boundary nodes
        if node1.startswith('*'):
            node1 = node1[1:]
            self.builder.mark_boundary_node(node1)
        if node2.startswith('*'):
            node2 = node2[1:]
            self.builder.mark_boundary_node(node2)
            
        try:
            value = self._parse_value(value_token)
            # Convert to fF
            value_ff = value * C_TO_FF
            
            # Check for nonlinear cap model
            attrs = {}
            if len(tokens) > model_idx and not tokens[model_idx].startswith('*'):
                attrs['nlcap_model'] = tokens[model_idx]
                
            self.builder.add_element('C', node1, node2, value_ff, name, **attrs)
        except ValueError as e:
            self.logger.warning(f"Error parsing capacitor value in line: {line}: {e}")
            
    def _parse_inductor(self, line: str):
        """Parse inductor: L<name> <node1> <node2> <value> OR l <node1> <node2> <value>"""
        tokens = line.split()
        if len(tokens) < 3:
            self.logger.warning(f"Invalid inductor line: {line}")
            return
        
        # Check format: 'l <node1> <node2> <value>' (no name) or 'L<name> <node1> <node2> <value>'
        first_token = tokens[0]
        if first_token.lower() == 'l':
            # Format: l <node1> <node2> <value>
            if len(tokens) < 4:
                self.logger.warning(f"Invalid inductor line: {line}")
                return
            # Generate unique name
            name = f"l_{tokens[1]}_{tokens[2]}_{id(line)}"
            node1 = tokens[1]
            node2 = tokens[2]
            value_token = tokens[3]
        else:
            # Format: L<name> <node1> <node2> <value>
            if len(tokens) < 4:
                self.logger.warning(f"Invalid inductor line: {line}")
                return
            name = tokens[0]
            node1 = tokens[1]
            node2 = tokens[2]
            value_token = tokens[3]
        
        # Handle boundary nodes
        if node1.startswith('*'):
            node1 = node1[1:]
            self.builder.mark_boundary_node(node1)
        if node2.startswith('*'):
            node2 = node2[1:]
            self.builder.mark_boundary_node(node2)
            
        try:
            value = self._parse_value(value_token)
            # Convert to nH
            value_nh = value * L_TO_NH
            self.builder.add_element('L', node1, node2, value_nh, name)
        except ValueError as e:
            self.logger.warning(f"Error parsing inductor value in line: {line}: {e}")
            
    def _parse_mutual_inductor(self, line: str):
        """Parse mutual inductor: K<name> L<name1> L<name2> <coupling>"""
        tokens = line.split()
        if len(tokens) < 4:
            self.logger.warning(f"Invalid mutual inductor line: {line}")
            return
            
        name = tokens[0]
        l1_name = tokens[1]
        l2_name = tokens[2]
        
        try:
            coupling = float(tokens[3])
            self.builder.mutual_inductors[name] = (l1_name, l2_name, coupling)
            self.builder.stats.mutual_inductors += 1
            self.logger.debug(f"Added mutual inductor {name}: {l1_name} <-> {l2_name}, k={coupling}")
        except ValueError as e:
            self.logger.warning(f"Error parsing mutual inductor in line: {line}: {e}")
            
    def _parse_vsource(self, line: str):
        """Parse voltage source: V<name> <node+> <node-> <dc_value> [AC ...] [PWL ...]"""
        tokens = line.split()
        if len(tokens) < 4:
            self.logger.warning(f"Invalid voltage source line: {line}")
            return
            
        name = tokens[0]
        node_pos = tokens[1]
        node_neg = tokens[2]
        
        # Handle boundary nodes
        if node_pos.startswith('*'):
            node_pos = node_pos[1:]
            self.builder.mark_boundary_node(node_pos)
        if node_neg.startswith('*'):
            node_neg = node_neg[1:]
            self.builder.mark_boundary_node(node_neg)
            
        try:
            dc_value = self._parse_value(tokens[3])
            
            # Parse additional parameters (AC, PWL, etc.)
            attrs = {'dc': dc_value}
            i = 4
            while i < len(tokens):
                token = tokens[i].upper()
                if token == 'AC' and i + 1 < len(tokens):
                    attrs['ac'] = float(tokens[i + 1])
                    i += 2
                elif token == 'PORTID' and i + 1 < len(tokens):
                    port_str = tokens[i + 1]
                    if '=' in port_str:
                        attrs['portid'] = port_str.split('=')[1]
                    i += 2
                elif token.startswith('PWL'):
                    # PWL(...) - extract but don't parse fully
                    attrs['pwl'] = line[line.upper().find('PWL'):]
                    break
                else:
                    i += 1
                    
            self.builder.add_element('V', node_pos, node_neg, dc_value, name, **attrs)
            self.builder.vsrc_dict[name] = attrs
            
        except ValueError as e:
            self.logger.warning(f"Error parsing voltage source in line: {line}: {e}")
            
    def _parse_isource(self, line: str):
        """
        Parse current source with full waveform support.

        Handles:
        - I<name> <node+> <node-> <dc_value> [static_value=...] [pulse(...)] [pwl(...)]

        All current values are converted to mA and stored in CurrentSource objects
        for both static DC analysis and time-domain evaluation.
        """
        # Fast net pre-filter (avoids full parsing for wrong-net lines)
        if (getattr(self, '_use_fast_instance_filter', False)
                and not _fast_instance_net_filter(line, self.net_filter.lower())):
            return

        prepared = _prepare_instance_source(line)
        if prepared is None:
            return

        isrc = prepared.cs
        node_pos = prepared.node_pos
        node_neg = prepared.node_neg
        name = isrc.name
        static_current_ma = prepared.static_current_ma

        if prepared.is_boundary_pos:
            self.builder.mark_boundary_node(node_pos)
        if prepared.is_boundary_neg:
            self.builder.mark_boundary_node(node_neg)

        # Build edge attributes for graph element
        attrs = {
            'dc': static_current_ma,
            'has_waveform': isrc.has_waveform_data()
        }

        # Extract coordinates from instance name (e.g., i_cell:1000_2000:vdd)
        coord_match = _RE_COORD_EXTRACT.search(name)
        if coord_match:
            attrs['inst_x'] = int(coord_match.group(1))
            attrs['inst_y'] = int(coord_match.group(2))

        # Add element to graph
        added = self.builder.add_element('I', node_pos, node_neg, static_current_ma, name, **attrs)

        # Only store if element was actually added (not filtered by net_filter)
        if added:
            # Backward compatibility: instance_node_map
            self.builder.instance_node_map[name] = [node_pos, node_neg]

            # Store full CurrentSource for time-domain analysis
            self.builder.instance_sources[name] = isrc

            # Update statistics
            if isrc.has_waveform_data():
                self.builder.stats.instances_with_waveforms += 1

                # Update per-net waveform statistics
                net_type = self.builder.node_net_map.get(node_pos) or self.builder.node_net_map.get(node_neg)
                if net_type and net_type in self.builder.stats.net_stats:
                    # Current sources from instanceModels are always 'die' category
                    if 'die' in self.builder.stats.net_stats[net_type]:
                        die_stats = self.builder.stats.net_stats[net_type]['die']
                        die_stats['isources_with_waveforms'] += 1
                        if 'wscale_values' not in die_stats:
                            die_stats['wscale_values'] = []
                        die_stats['wscale_values'].append(isrc.wscale)

            self.builder.stats.total_static_current_ma += abs(static_current_ma)
            
    def _parse_controlled_source(self, line: str, source_type: str):
        """Parse controlled sources (E/F/G/H)"""
        # E: VCVS, F: CCCS, G: VCCS, H: CCVS
        tokens = line.split()
        if len(tokens) < 6:
            self.logger.warning(f"Invalid controlled source line: {line}")
            return
            
        name = tokens[0]
        out_pos = tokens[1]
        out_neg = tokens[2]
        
        # Handle boundary nodes
        for i in [1, 2, 3, 4]:
            if i < len(tokens) and tokens[i].startswith('*'):
                tokens[i] = tokens[i][1:]
                self.builder.mark_boundary_node(tokens[i])
        
        attrs = {
            'source_type': source_type,
            'ctrl_pos': tokens[3],
            'ctrl_neg': tokens[4]
        }
        
        try:
            gain = float(tokens[5])
            # Store as special edge with type indicating controlled source
            self.builder.add_element(source_type, out_pos, out_neg, gain, name, **attrs)
        except ValueError as e:
            self.logger.warning(f"Error parsing controlled source in line: {line}: {e}")
            
    def _parse_subcircuit_definition(self, line: str):
        """Parse .subckt definition (store for later expansion)"""
        tokens = line.split()
        if len(tokens) < 3:
            return
            
        subckt_name = tokens[1]
        pins = tokens[2:]
        
        # Read body until .ends
        body_lines = []
        # Note: This is simplified - in full implementation, would need to handle
        # nested subcircuits properly
        
        self.subcircuits[subckt_name] = {
            'pins': pins,
            'body': body_lines
        }
        self.logger.debug(f"Defined subcircuit: {subckt_name} with pins: {pins}")
        
    def _parse_subcircuit_instance(self, line: str):
        """Parse subcircuit instance and expand it (flatten hierarchy)"""
        tokens = line.split()
        if len(tokens) < 3:
            self.logger.warning(f"Invalid subcircuit instance: {line}")
            return
            
        inst_name = tokens[0]
        subckt_name = tokens[-1]
        node_list = tokens[1:-1]
        
        if subckt_name not in self.subcircuits:
            self.logger.warning(f"Subcircuit {subckt_name} not defined for instance {inst_name}")
            return
            
        # Flatten: expand subcircuit with hierarchical naming
        subckt = self.subcircuits[subckt_name]
        pins = subckt['pins']
        
        if len(node_list) != len(pins):
            self.logger.warning(f"Pin count mismatch for instance {inst_name}")
            return
            
        # Create pin mapping
        pin_map = dict(zip(pins, node_list))
        
        # Process subcircuit body (simplified - would need full implementation)
        # For now, just log the expansion
        self.logger.debug(f"Expanding subcircuit instance {inst_name} of type {subckt_name}")
        
    def _parse_value(self, value_str: str) -> float:
        """Parse SPICE value with suffix (K, M, G, etc.)"""
        value_str = value_str.strip()
        
        # Check if this is a parameter reference
        # Parameters are case-insensitive in SPICE
        param_key = value_str.upper()
        if param_key in self.builder.parameters:
            value_str = self.builder.parameters[param_key]
        # Also check case-preserving lookup
        elif value_str in self.builder.parameters:
            value_str = self.builder.parameters[value_str]
        
        value_str = value_str.strip().upper()
        
        # SPICE suffixes - order matters! Check longer suffixes first (MEG before M)
        suffixes = [
            ('MEG', 1e6), ('T', 1e12), ('G', 1e9), ('X', 1e6), ('K', 1e3),
            ('M', 1e-3), ('U', 1e-6), ('N', 1e-9), ('P', 1e-12), ('F', 1e-15)
        ]
        
        # Check for suffix
        for suffix, multiplier in suffixes:
            if value_str.endswith(suffix):
                base = value_str[:-len(suffix)]
                return float(base) * multiplier
                
        # No suffix
        return float(value_str)
        
    def _perform_validation(self):
        """Perform sanity checks on the parsed netlist"""
        self.logger.info("Performing netlist validation...")
        
        self.builder.stats.nodes_before_cleanup = self.builder.graph.number_of_nodes()
        
        # Check for shorted resistors
        self._check_shorts()
        
        # Check for floating nodes
        self._check_floating_nodes()
        
        # Check for grounded nodes
        self._check_grounded_nodes()
        
    def _check_shorts(self):
        """Detect and report shorted resistors.

        Optimized to use rustworkx directly instead of wrapper iteration.
        """
        rx_graph = self.builder.graph._graph
        idx_to_node = self.builder.graph._idx_to_node
        shorts = []

        for edge_idx in rx_graph.edge_indices():
            data = rx_graph.get_edge_data_by_index(edge_idx)
            if data and data.get('type') == 'R':
                value = data.get('value', float('inf'))
                if value < SHORT_THRESHOLD:
                    u_idx, v_idx = rx_graph.get_edge_endpoints_by_index(edge_idx)
                    shorts.append((idx_to_node[u_idx], idx_to_node[v_idx],
                                  data.get('elem_name'), value))
                    self.builder.stats.shorted_elements += 1

        if shorts:
            msg = f"Found {len(shorts)} shorted resistors:\n"
            for u, v, name, value in shorts[:10]:  # Show first 10
                display_name = name if name else "(die resistor)"
                msg += f"  {display_name}: {u} <-> {v} = {value:.2e} KOhm\n"
            if len(shorts) > 10:
                msg += f"  ... and {len(shorts) - 10} more\n"
                
            if self.strict:
                raise ValueError(msg)
            else:
                self.logger.warning(msg)
                
    def _check_floating_nodes(self):
        """Detect nodes not connected to any voltage source.

        Optimized to:
        1. Use tracked vsrc_edge_indices instead of iterating all edges
        2. Use directed graph directly - node_connected_component already
           handles weak connectivity for directed graphs via
           rx.weakly_connected_components(), so to_undirected() is unnecessary.
        """
        rx_graph = self.builder.graph._graph
        idx_to_node = self.builder.graph._idx_to_node

        # Find all nodes connected to voltage sources using tracked indices
        grounded_nodes = set()
        for edge_idx in self.builder.vsrc_edge_indices:
            try:
                u_idx, v_idx = rx_graph.get_edge_endpoints_by_index(edge_idx)
                grounded_nodes.add(idx_to_node[u_idx])
                grounded_nodes.add(idx_to_node[v_idx])
            except Exception:
                # Edge may have been removed by earlier processing
                pass

        # BFS from grounded nodes using weak connectivity on directed graph
        connected_nodes = set()
        for node in grounded_nodes:
            if node not in connected_nodes:
                component = node_connected_component(self.builder.graph, node)
                connected_nodes.update(component)

        # Find floating nodes
        all_nodes = set(self.builder.graph.nodes())
        floating = all_nodes - connected_nodes

        self.builder.stats.floating_nodes = len(floating)

        if floating:
            msg = f"Found {len(floating)} floating nodes (not connected to voltage source)"
            if len(floating) <= 10:
                msg += f": {', '.join(list(floating)[:10])}"
            else:
                msg += f". First 10: {', '.join(list(floating)[:10])}"

            self.logger.warning(msg)
            
    def _check_grounded_nodes(self):
        """Check for nodes directly grounded via shorts"""
        # This would be more complex - simplified version
        ground_node = '0'
        if ground_node in self.builder.graph:
            neighbors = list(self.builder.graph.neighbors(ground_node))
            self.logger.debug(f"Ground node '0' has {len(neighbors)} direct connections")
    
    def _identify_vsrc_nodes(self):
        """
        Identify nodes connected to voltage sources via zero-valued resistors.
        These are typically package nodes connected to ideal voltage sources through
        resistors named 'rs' (or custom pattern) with value == 0.0.
        
        The identification propagates through zero-resistance paths up to a 
        configurable depth limit.
        """
        self.logger.info("Identifying voltage source nodes...")
        
        vsrc_nodes = set()
        
        # Step 1: Find all nodes directly connected to voltage sources
        for u, v, data in self.builder.graph.edges(data=True):
            if data.get('type') == 'V':
                vsrc_nodes.add(u)
                vsrc_nodes.add(v)
        
        # Step 2: Find zero-valued resistors matching pattern
        zero_resistors = []
        for u, v, key, data in self.builder.graph.edges(keys=True, data=True):
            if data.get('type') == 'R':
                elem_name = data.get('elem_name', '')
                value = data.get('value', float('inf'))
                # Check for exact pattern match and zero resistance
                if elem_name == self.vsrc_resistor_pattern and value == 0.0:
                    zero_resistors.append((u, v))
        
        # Step 3: Propagate through zero-resistance paths using BFS with depth limit
        from collections import deque
        
        visited = set(vsrc_nodes)
        queue = deque([(node, 0) for node in vsrc_nodes])
        
        while queue:
            node, depth = queue.popleft()
            
            if depth >= self.vsrc_depth_limit:
                continue
            
            # Check all zero-resistance connections from this node
            for u, v in zero_resistors:
                neighbor = None
                if u == node and v not in visited:
                    neighbor = v
                elif v == node and u not in visited:
                    neighbor = u
                
                if neighbor:
                    visited.add(neighbor)
                    vsrc_nodes.add(neighbor)
                    queue.append((neighbor, depth + 1))
        
        # Step 4: Mark nodes in graph
        for node in vsrc_nodes:
            if node in self.builder.graph:
                self.builder.graph.nodes_dict[node]['is_vsrc_node'] = True
        
        # Update statistics
        self.builder.stats.vsrc_nodes = len(vsrc_nodes)
        
        # Store in graph metadata
        self.builder.graph.graph['vsrc_nodes'] = vsrc_nodes
        
        self.logger.info(f"Identified {len(vsrc_nodes)} voltage source nodes")
        if zero_resistors:
            self.logger.debug(f"Found {len(zero_resistors)} zero-valued '{self.vsrc_resistor_pattern}' resistors")
    
    def _compute_layer_stats(self):
        """
        Compute per-layer and per-net statistics for nodes and elements.
        Aggregates counts by layer identifier and net type.
        """
        self.logger.info("Computing layer statistics...")
        
        # Structure: layer_stats_by_net[net][layer] = stats
        layer_stats_by_net = defaultdict(lambda: defaultdict(lambda: {
            'nodes': 0,
            'vsrc_nodes': 0,
            'resistors': 0,
            'capacitors': 0,
            'inductors': 0,
            'vsources': 0,
            'isources': 0
        }))
        
        # Count nodes per layer per net
        for node, data in self.builder.graph.nodes(data=True):
            net_type = data.get('net_type') or self.builder._get_node_net(node)
            
            if net_type:
                # Determine if this is a package node
                is_package_node = node not in self.builder.node_net_map and node != '0'
                
                if is_package_node:
                    layer = 'package'
                else:
                    layer = data.get('layer')
                
                layer_stats_by_net[net_type][layer]['nodes'] += 1
                if data.get('is_vsrc_node', False):
                    layer_stats_by_net[net_type][layer]['vsrc_nodes'] += 1
        
        # Count elements per layer per net
        # Count both intra-layer and inter-layer elements
        for u, v, data in self.builder.graph.edges(data=True):
            elem_type = data.get('type')
            net_type = data.get('net_type') or self.builder._get_node_net(u)
            
            # Determine layer for this element
            u_is_die = u in self.builder.node_net_map
            v_is_die = v in self.builder.node_net_map
            u_is_package = not u_is_die and u != '0'
            v_is_package = not v_is_die and v != '0'
            
            # Determine layer based on node types
            if u_is_package or v_is_package:
                # At least one node is package - count as package layer
                layer = 'package'
            else:
                # Both are die nodes or ground, get their layers
                # Treat node '0' (ground) as having no layer - use the other node's layer
                u_layer = self.builder.graph.nodes_dict[u].get('layer') if u in self.builder.graph and u != '0' else None
                v_layer = self.builder.graph.nodes_dict[v].get('layer') if v in self.builder.graph and v != '0' else None
                
                # Handle ground node cases
                if u == '0':
                    layer = v_layer
                elif v == '0':
                    layer = u_layer
                elif u_layer == v_layer:
                    # Same layer (intra-layer)
                    layer = u_layer
                elif u_layer and v_layer:
                    # Inter-layer connection - create combined layer name
                    # Sort to ensure consistency (e.g., "19-21" not "21-19")
                    layers_sorted = sorted([u_layer, v_layer], key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else x))
                    layer = f"{layers_sorted[0]}-{layers_sorted[1]}"
                else:
                    # One or both layers unknown
                    layer = u_layer or v_layer
            
            if net_type:
                if elem_type == 'R':
                    layer_stats_by_net[net_type][layer]['resistors'] += 1
                elif elem_type == 'C':
                    layer_stats_by_net[net_type][layer]['capacitors'] += 1
                elif elem_type == 'L':
                    layer_stats_by_net[net_type][layer]['inductors'] += 1
                elif elem_type == 'V':
                    layer_stats_by_net[net_type][layer]['vsources'] += 1
                elif elem_type == 'I':
                    layer_stats_by_net[net_type][layer]['isources'] += 1
        
        # Convert to regular dict and store
        self.builder.stats.layer_stats_by_net = {net: dict(layers) for net, layers in layer_stats_by_net.items()}
        self.builder.graph.graph['layer_stats_by_net'] = self.builder.stats.layer_stats_by_net
        
        # Also compute global layer stats for backwards compatibility
        layer_stats = defaultdict(lambda: {
            'nodes': 0, 'vsrc_nodes': 0, 'resistors': 0,
            'capacitors': 0, 'inductors': 0, 'vsources': 0, 'isources': 0
        })
        for net_layers in layer_stats_by_net.values():
            for layer, stats in net_layers.items():
                for key in stats:
                    layer_stats[layer][key] += stats[key]
        self.builder.stats.layer_stats = dict(layer_stats)
        self.builder.graph.graph['layer_stats'] = dict(layer_stats)
        
        # Log summary
        total_layers = set()
        for net_layers in layer_stats_by_net.values():
            total_layers.update(k for k in net_layers.keys() if k is not None and k != 'package')
        if total_layers:
            self.logger.info(f"Found {len(total_layers)} layers")
            for net, net_layers in sorted(layer_stats_by_net.items()):
                for layer, stats in sorted(net_layers.items(), key=lambda x: (x[0] is None, x[0] == 'package', x[0])):
                    if layer is not None and layer != 'package':
                        self.logger.debug(f"  Net {net}, Layer {layer}: {stats['nodes']} nodes, "
                                        f"{stats['resistors']} resistors, {stats['capacitors']} capacitors")
            
    def _print_statistics(self):
        """Print parsing statistics"""
        stats = self.builder.stats
        
        # Per-net statistics
        if stats.net_stats:
            print("\n" + "-"*70)
            print("Per-Net Statistics:")
            print("-"*70)
            
            # Sort nets by name, exclude 'unmapped'
            # If net filter is active, only show that net
            if self.net_filter:
                sorted_nets = [net for net in stats.net_stats.keys() if net != 'unmapped' and net.lower() == self.net_filter.lower()]
            else:
                sorted_nets = sorted([net for net in stats.net_stats.keys() if net != 'unmapped'])
            
            for net in sorted_nets:
                net_categories = stats.net_stats[net]
                print(f"\n  Net: {net}")
                
                for category in ['die', 'package']:
                    if category not in net_categories:
                        continue
                    
                    net_stat = net_categories[category]
                    
                    # Skip if no elements in this category
                    total_elems = (net_stat['resistors'] + net_stat['capacitors'] + 
                                 net_stat['inductors'] + net_stat['vsources'] + net_stat['isources'])
                    if total_elems == 0:
                        continue
                    
                    print(f"    [{category.upper()}]")
                    print(f"      Nodes: {len(net_stat['nodes']):,}")
                    print(f"      Resistors: {net_stat['resistors']:,}")
                    if net_stat['resistors'] > 0:
                        avg_r = net_stat['total_resistance'] / net_stat['resistors']
                        print(f"        Total: {net_stat['total_resistance']:.3f} KOhm")
                        print(f"        Average: {avg_r:.6f} KOhm")
                    
                    print(f"      Capacitors: {net_stat['capacitors']:,}")
                    if net_stat['capacitors'] > 0:
                        avg_c = net_stat['total_capacitance'] / net_stat['capacitors']
                        print(f"        Total: {net_stat['total_capacitance']:.3f} fF")
                        print(f"        Average: {avg_c:.6f} fF")
                    
                    if net_stat['inductors'] > 0:
                        print(f"      Inductors: {net_stat['inductors']:,}")
                        avg_l = net_stat['total_inductance'] / net_stat['inductors']
                        print(f"        Total: {net_stat['total_inductance']:.3f} nH")
                        print(f"        Average: {avg_l:.6f} nH")
                    
                    if net_stat['vsources'] > 0:
                        print(f"      Voltage Sources: {net_stat['vsources']:,}")
                    
                    if net_stat['isources'] > 0:
                        print(f"      Current Sources: {net_stat['isources']:,}")
                        waveform_count = net_stat.get('isources_with_waveforms', 0)
                        if waveform_count > 0:
                            print(f"        With Waveforms: {waveform_count:,}")
                            # Display wscale distribution
                            wscale_values = net_stat.get('wscale_values', [])
                            if wscale_values:
                                print(f"        Wscale Distribution:")
                                # Count by unique wscale value
                                wscale_counts = {}
                                for ws in wscale_values:
                                    wscale_counts[ws] = wscale_counts.get(ws, 0) + 1
                                # Sort by wscale descending, print with percentages
                                total = len(wscale_values)
                                for ws in sorted(wscale_counts.keys(), reverse=True):
                                    count = wscale_counts[ws]
                                    pct = 100.0 * count / total
                                    print(f"          {ws:.2f}: {count:,} ({pct:.1f}%)")
                        avg_i = net_stat['total_current'] / net_stat['isources']
                        print(f"        Total Current: {net_stat['total_current']:.3f} mA")
                        print(f"        Average: {avg_i:.6f} mA")
        
        print("\n" + "="*70 + "\n")
        stats = self.builder.stats
        
        self.logger.info("=" * 60)
        self.logger.info("Netlist Parsing Statistics")
        self.logger.info("=" * 60)
        self.logger.info(f"Nodes: {stats.nodes_after_cleanup}")
        if self.validate and stats.nodes_before_cleanup > 0:
            self.logger.info(f"  (before cleanup: {stats.nodes_before_cleanup})")
        if stats.vsrc_nodes > 0:
            self.logger.info(f"  Voltage Source Nodes: {stats.vsrc_nodes}")
        self.logger.info(f"Elements: {stats.elements_total}")
        self.logger.info(f"  Resistors: {stats.resistors}")
        self.logger.info(f"  Capacitors: {stats.capacitors}")
        self.logger.info(f"  Inductors: {stats.inductors}")
        self.logger.info(f"  Voltage Sources: {stats.vsources}")
        self.logger.info(f"  Current Sources: {stats.isources}")
        self.logger.info(f"  Mutual Inductors: {stats.mutual_inductors}")
        if self.builder.tile_grid:
            self.logger.info(f"Tile Grid: {self.builder.tile_grid[0]} x {self.builder.tile_grid[1]}")
            self.logger.info(f"  Tiles Parsed: {stats.tiles_parsed}")
            if stats.tiles_failed > 0:
                self.logger.info(f"  Tiles Failed: {stats.tiles_failed}")
        self.logger.info(f"Boundary Nodes: {stats.boundary_nodes}")
        if stats.package_nodes > 0:
            self.logger.info(f"Package Nodes: {stats.package_nodes}")
        if self.validate:
            self.logger.info(f"Shorted Elements: {stats.shorted_elements}")
            self.logger.info(f"Floating Nodes: {stats.floating_nodes}")
        self.logger.info(f"Instance-Node Mappings: {len(self.builder.instance_node_map)}")
        
        # Print layer statistics by net
        if hasattr(stats, 'layer_stats_by_net') and stats.layer_stats_by_net:
            # Filter nets if net_filter is active
            if self.net_filter:
                nets_to_show = [net for net in stats.layer_stats_by_net.keys() if net.lower() == self.net_filter.lower()]
            else:
                nets_to_show = sorted(stats.layer_stats_by_net.keys())
            
            for net in nets_to_show:
                net_layers = stats.layer_stats_by_net[net]
                
                # Separate single layers and inter-layer connections
                single_layers = []
                inter_layers = []
                for k in net_layers.keys():
                    if k is not None and k != 'package':
                        if '-' in str(k):
                            inter_layers.append(k)
                        else:
                            single_layers.append(k)
                
                if single_layers or inter_layers or 'package' in net_layers:
                    self.logger.info("=" * 60)
                    self.logger.info(f"Layer Statistics for Net: {net} ({len(single_layers)} layers)")
                    self.logger.info("=" * 60)
                    
                    # Print header
                    self.logger.info(f"{'Layer':<15} {'Nodes':>8} {'Vsrc':>6} {'Res':>8} {'Cap':>8} {'Ind':>6} {'Isrc':>6}")
                    self.logger.info("-" * 60)
                    
                    # Sort: numeric layers first (as ints), then alphabetic layers
                    def layer_sort_key(x):
                        if isinstance(x, str) and x.isdigit():
                            return (0, int(x))
                        else:
                            return (1, str(x))
                    
                    # Initialize totals
                    total_stats = {
                        'nodes': 0, 'vsrc_nodes': 0, 'resistors': 0,
                        'capacitors': 0, 'inductors': 0, 'vsources': 0, 'isources': 0
                    }
                    
                    # Print single-layer stats (sorted by layer name)
                    for layer in sorted(single_layers, key=layer_sort_key):
                        layer_stat = net_layers[layer]
                        self.logger.info(
                            f"{str(layer):<15} "
                            f"{layer_stat['nodes']:>8} "
                            f"{layer_stat['vsrc_nodes']:>6} "
                            f"{layer_stat['resistors']:>8} "
                            f"{layer_stat['capacitors']:>8} "
                            f"{layer_stat['inductors']:>6} "
                            f"{layer_stat['isources']:>6}"
                        )
                        # Accumulate totals
                        for key in total_stats:
                            total_stats[key] += layer_stat[key]
                    
                    # Print inter-layer stats (sorted)
                    for layer in sorted(inter_layers):
                        layer_stat = net_layers[layer]
                        self.logger.info(
                            f"{str(layer):<15} "
                            f"{layer_stat['nodes']:>8} "
                            f"{layer_stat['vsrc_nodes']:>6} "
                            f"{layer_stat['resistors']:>8} "
                            f"{layer_stat['capacitors']:>8} "
                            f"{layer_stat['inductors']:>6} "
                            f"{layer_stat['isources']:>6}"
                        )
                        # Accumulate totals
                        for key in total_stats:
                            total_stats[key] += layer_stat[key]
                    
                    # Print Package row if exists
                    if 'package' in net_layers:
                        layer_stat = net_layers['package']
                        self.logger.info(
                            f"{'Package':<15} "
                            f"{layer_stat['nodes']:>8} "
                            f"{layer_stat['vsrc_nodes']:>6} "
                            f"{layer_stat['resistors']:>8} "
                            f"{layer_stat['capacitors']:>8} "
                            f"{layer_stat['inductors']:>6} "
                            f"{layer_stat['isources']:>6}"
                        )
                        # Add package to totals
                        for key in total_stats:
                            total_stats[key] += layer_stat[key]
                    
                    # Print Total row
                    self.logger.info("-" * 60)
                    self.logger.info(
                        f"{'Total':<15} "
                        f"{total_stats['nodes']:>8} "
                        f"{total_stats['vsrc_nodes']:>6} "
                        f"{total_stats['resistors']:>8} "
                        f"{total_stats['capacitors']:>8} "
                        f"{total_stats['inductors']:>6} "
                        f"{total_stats['isources']:>6}"
                    )
        
        self.logger.info("=" * 60)


#!/usr/bin/env python3
"""
Generate a sampled multi-tile PDN netlist from netlist_minion.

This script creates a realistic multi-die PDN netlist for fast performance
evaluation of the PDN parser, solver, and analysis tools. The generated
netlist is sampled from netlist_minion (VDD_XLV net) to preserve
realistic resistor/capacitor distributions and current source characteristics.

Target netlist:
- ~100K nodes, ~160K resistors, ~150K capacitors
- ~60K current sources (50% with pulse waveforms)
- 25-30 voltage sources
- 3x3 tiles with connected boundary nodes
- ~10% of original current (~16 mA) and capacitance (~220 fF)
- Expected IR-drop: ~5 mV

Usage:
    python generate_sampled_netlist.py

Output:
    netlist_sampled/
"""

import os
import re
import sys
import random
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional, Any, FrozenSet
from collections import defaultdict, deque

# Add prototype directory to path for core imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Use parser.current_sources to match the classes stored in pickle files
from .netlist import load_pdn_pickle
from .current_sources import CurrentSource, Pulse, PWL


def _get_script_dir() -> str:
    """Get the directory containing this script."""
    return os.path.dirname(os.path.abspath(__file__))


@dataclass
class SamplingConfig:
    """Configuration for netlist sampling."""
    source_pickle: str = field(default_factory=lambda: os.path.join(_get_script_dir(), 'netlist_minion/pdn_graph.pkl'))
    output_dir: str = field(default_factory=lambda: os.path.join(_get_script_dir(), 'netlist_sampled'))
    net_name: str = 'VDD_XLV'
    vdd_voltage: float = 0.66

    # Sampling parameters
    x_sample_fraction: float = 0.40  # ~40% of x range
    y_sample_fraction: float = 0.40  # ~40% of y range

    # Tile grid
    n_tiles_x: int = 3
    n_tiles_y: int = 3

    # Voltage source sampling
    target_vsrc_count: int = 27  # ~25-30

    # Current source sampling
    target_current_sources: int = 60000  # ~60K
    waveform_ratio: float = 0.50  # 50% with waveforms

    # Random seed
    seed: int = 42

    # Island creation (for testing floating island handling)
    # 3-5 islands with 100-200 nodes each = ~800 max total (< 1000)
    n_islands: int = 4  # Number of M0 islands to create
    island_size_range: Tuple[int, int] = (100, 200)  # Min/max nodes per island

    # Layer rename map
    layer_rename: Dict[str, str] = field(default_factory=lambda: {
        '24': 'M0', '26': 'M1', '28': 'M2', '30': 'M3',
        '32': 'M4', '34': 'M5', '36': 'M6', '38': 'M7',
        '40': 'M8', '42': 'M9', '44': 'M10', '46': 'M11',
        '48': 'M12', '50': 'M13'
    })


class SampledNetlistGenerator:
    """Generate a sampled multi-tile PDN netlist."""

    def __init__(self, config: SamplingConfig):
        self.config = config
        random.seed(config.seed)

        # Data structures
        self.graph = None
        self.net_nodes: Set[str] = set()
        self.vsrc_nodes: Set[str] = set()

        # Sampled data
        self.sampled_nodes: Set[str] = set()
        self.sampled_resistors: List[Tuple[str, str, float]] = []  # (node1, node2, value)
        self.sampled_capacitors: List[Tuple[str, str, float]] = []
        self.node_to_tile: Dict[str, Tuple[int, int]] = {}
        self.tile_nodes: Dict[Tuple[int, int], Set[str]] = defaultdict(set)
        self.tile_resistors: Dict[Tuple[int, int], List] = defaultdict(list)
        self.tile_capacitors: Dict[Tuple[int, int], List] = defaultdict(list)
        self.sampled_vsrcs: List[Tuple[str, str]] = []  # (vsrc_node, die_node)
        self.sampled_current_sources: Dict[str, List[CurrentSource]] = defaultdict(list)
        self._nodes_with_resistors: Set[str] = set()  # Nodes that have resistor edges
        self._island_nodes: Set[str] = set()  # Nodes in floating islands
        self._boundary_nodes: Set[str] = set()  # Nodes shared across tile boundaries

        # Bounds
        self.sample_x_min = 0
        self.sample_x_max = 0
        self.sample_y_min = 0
        self.sample_y_max = 0

        # Statistics
        self.stats = {}

    def load_source_graph(self):
        """Load the source PDN graph from pickle."""
        print(f"Loading source graph from {self.config.source_pickle}...")
        self.graph = load_pdn_pickle(self.config.source_pickle)

        # Get net nodes
        net_connectivity = self.graph.graph.get('net_connectivity', {})
        self.net_nodes = set(net_connectivity.get(self.config.net_name, []))
        print(f"  Net {self.config.net_name} has {len(self.net_nodes)} nodes")

        # Get vsrc nodes
        self.vsrc_nodes = set(self.graph.graph.get('vsrc_nodes', []))
        print(f"  Found {len(self.vsrc_nodes)} voltage source nodes")

    def _parse_die_node(self, node: str) -> Optional[Tuple[int, int, str]]:
        """Parse die node name into (x, y, layer). Returns None for non-die nodes."""
        # Die nodes have format: X_Y_LAYER (e.g., '1000_2000_24')
        parts = node.split('_')
        if len(parts) == 3:
            try:
                x = int(parts[0])
                y = int(parts[1])
                layer = parts[2]
                return (x, y, layer)
            except ValueError:
                pass
        return None

    def _rename_layer(self, layer: str) -> str:
        """Rename layer using config mapping."""
        return self.config.layer_rename.get(layer, layer)

    def _rename_node(self, node: str) -> str:
        """Rename node with new layer names."""
        parsed = self._parse_die_node(node)
        if parsed:
            x, y, layer = parsed
            new_layer = self._rename_layer(layer)
            return f"{x}_{y}_{new_layer}"
        return node

    def select_sampling_region(self):
        """Select the spatial region to sample from."""
        print("Selecting sampling region...")

        # Collect coordinates from die nodes in the net
        coords = []
        for node in self.net_nodes:
            parsed = self._parse_die_node(node)
            if parsed:
                x, y, _ = parsed
                coords.append((x, y))

        if not coords:
            raise ValueError(f"No die nodes found in net {self.config.net_name}")

        x_coords = [c[0] for c in coords]
        y_coords = [c[1] for c in coords]

        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)

        print(f"  Original bounding box: x=[{x_min}, {x_max}], y=[{y_min}, {y_max}]")

        # Calculate sample region (centered)
        x_range = x_max - x_min
        y_range = y_max - y_min

        sample_x_size = int(x_range * self.config.x_sample_fraction)
        sample_y_size = int(y_range * self.config.y_sample_fraction)

        # Center the region
        x_center = (x_min + x_max) // 2
        y_center = (y_min + y_max) // 2

        self.sample_x_min = x_center - sample_x_size // 2
        self.sample_x_max = x_center + sample_x_size // 2
        self.sample_y_min = y_center - sample_y_size // 2
        self.sample_y_max = y_center + sample_y_size // 2

        print(f"  Sample region: x=[{self.sample_x_min}, {self.sample_x_max}], "
              f"y=[{self.sample_y_min}, {self.sample_y_max}]")
        print(f"  Sample size: {sample_x_size} x {sample_y_size} "
              f"({self.config.x_sample_fraction*100:.0f}% x {self.config.y_sample_fraction*100:.0f}%)")

    def _is_in_sample_region(self, node: str) -> bool:
        """Check if a die node is in the sample region."""
        parsed = self._parse_die_node(node)
        if parsed:
            x, y, _ = parsed
            return (self.sample_x_min <= x <= self.sample_x_max and
                    self.sample_y_min <= y <= self.sample_y_max)
        return False

    def extract_sampled_subgraph(self):
        """Extract nodes and edges within the sample region."""
        print("Extracting sampled subgraph...")

        # First pass: collect nodes in sample region
        for node in self.net_nodes:
            if self._is_in_sample_region(node):
                self.sampled_nodes.add(node)

        print(f"  Sampled {len(self.sampled_nodes)} nodes in region")

        # Second pass: extract edges where both endpoints are in sample region
        # Use keys=True for multigraph compatibility
        for u, v, k, data in self.graph.edges(keys=True, data=True):
            elem_type = data.get('type', '')  # Use 'type' not 'elem_type'
            value = data.get('value', 0)

            # For resistors: both endpoints must be in sampled nodes
            if elem_type == 'R':
                if u in self.sampled_nodes and v in self.sampled_nodes:
                    self.sampled_resistors.append((u, v, value))
                elif u in self.sampled_nodes or v in self.sampled_nodes:
                    # Edge crosses boundary - include both nodes
                    if u not in self.sampled_nodes and self._parse_die_node(u):
                        self.sampled_nodes.add(u)
                    if v not in self.sampled_nodes and self._parse_die_node(v):
                        self.sampled_nodes.add(v)
            # For capacitors: include ground caps where non-ground node is sampled
            elif elem_type == 'C':
                if v == '0' and u in self.sampled_nodes:
                    # Ground cap - node u connects to ground
                    self.sampled_capacitors.append((u, v, value))
                elif u == '0' and v in self.sampled_nodes:
                    # Ground cap - node v connects to ground
                    self.sampled_capacitors.append((v, '0', value))
                elif u in self.sampled_nodes and v in self.sampled_nodes:
                    # Coupling cap between two sampled nodes
                    self.sampled_capacitors.append((u, v, value))

        print(f"  Extracted {len(self.sampled_resistors)} resistors")
        print(f"  Extracted {len(self.sampled_capacitors)} capacitors")

    def partition_into_tiles(self):
        """Partition sampled nodes into 3x3 tiles."""
        print("Partitioning into tiles...")

        # Calculate tile boundaries
        x_range = self.sample_x_max - self.sample_x_min
        y_range = self.sample_y_max - self.sample_y_min

        tile_x_size = x_range / self.config.n_tiles_x
        tile_y_size = y_range / self.config.n_tiles_y

        print(f"  Tile size: {tile_x_size:.0f} x {tile_y_size:.0f}")

        # Build adjacency list from resistor edges
        adj: Dict[str, List[str]] = {}
        for u, v, _ in self.sampled_resistors:
            if u not in adj:
                adj[u] = []
            if v not in adj:
                adj[v] = []
            adj[u].append(v)
            adj[v].append(u)

        # Find M13 (layer 50) nodes that will have vsrc connections
        m13_seed_nodes = []
        for node in adj.keys():
            parsed = self._parse_die_node(node)
            if parsed:
                _, _, layer = parsed
                if layer == '50':  # M13 layer
                    m13_seed_nodes.append(node)

        print(f"  M13 nodes in resistor network: {len(m13_seed_nodes)}")

        # BFS from M13 nodes to find connected component
        nodes_with_resistors = set()
        visited = set()
        queue = deque(m13_seed_nodes)
        for seed in m13_seed_nodes:
            visited.add(seed)

        while queue:
            node = queue.popleft()
            nodes_with_resistors.add(node)
            for neighbor in adj.get(node, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        print(f"  Nodes connected to M13: {len(nodes_with_resistors)}")

        # Filter resistors to only those between connected nodes
        filtered_resistors = [(u, v, val) for u, v, val in self.sampled_resistors
                              if u in nodes_with_resistors and v in nodes_with_resistors]
        self.sampled_resistors = filtered_resistors
        print(f"  Resistors in connected component: {len(self.sampled_resistors)}")

        # Assign nodes to tiles (only those in connected component)
        for node in nodes_with_resistors:
            parsed = self._parse_die_node(node)
            if parsed:
                x, y, _ = parsed
                tile_x = min(int((x - self.sample_x_min) / tile_x_size),
                            self.config.n_tiles_x - 1)
                tile_y = min(int((y - self.sample_y_min) / tile_y_size),
                            self.config.n_tiles_y - 1)
                tile_x = max(0, tile_x)
                tile_y = max(0, tile_y)

                self.node_to_tile[node] = (tile_x, tile_y)
                self.tile_nodes[(tile_x, tile_y)].add(node)

        # Assign resistors to tiles (assign to lower tile_id if crossing boundary)
        for u, v, value in self.sampled_resistors:
            tile_u = self.node_to_tile.get(u)
            tile_v = self.node_to_tile.get(v)

            if tile_u and tile_v:
                # Assign to the "owning" tile (lower tile_id)
                if tile_u <= tile_v:
                    self.tile_resistors[tile_u].append((u, v, value))
                else:
                    self.tile_resistors[tile_v].append((u, v, value))

        # Assign capacitors to tiles (only for nodes with resistor connections)
        for u, v, value in self.sampled_capacitors:
            if v == '0':  # Ground cap
                if u in nodes_with_resistors:  # Only include if node has resistor connections
                    tile = self.node_to_tile.get(u)
                    if tile:
                        self.tile_capacitors[tile].append((u, v, value))
            else:
                # Coupling cap - both nodes must have resistor connections
                if u in nodes_with_resistors and v in nodes_with_resistors:
                    tile_u = self.node_to_tile.get(u)
                    tile_v = self.node_to_tile.get(v)
                    if tile_u and tile_v:
                        if tile_u <= tile_v:
                            self.tile_capacitors[tile_u].append((u, v, value))
                        else:
                            self.tile_capacitors[tile_v].append((u, v, value))

        # Store nodes_with_resistors for later use
        self._nodes_with_resistors = nodes_with_resistors

        # Print tile statistics
        for tile_y in range(self.config.n_tiles_y):
            for tile_x in range(self.config.n_tiles_x):
                tile = (tile_x, tile_y)
                print(f"  Tile ({tile_x},{tile_y}): {len(self.tile_nodes[tile])} nodes, "
                      f"{len(self.tile_resistors[tile])} R, {len(self.tile_capacitors[tile])} C")

    def _compute_boundary_nodes(self):
        """Identify nodes that appear in cross-tile resistors.

        These nodes need a '*' prefix in the .ckt files to signal the parser
        that they are shared across tile boundaries and need to be stitched.
        """
        print("Computing boundary nodes...")

        for tile, resistors in self.tile_resistors.items():
            for u, v, _ in resistors:
                tile_u = self.node_to_tile.get(u)
                tile_v = self.node_to_tile.get(v)

                # Check if this resistor crosses a tile boundary
                if tile_u != tile_v:
                    # Both nodes in a cross-tile resistor are boundary nodes
                    self._boundary_nodes.add(u)
                    self._boundary_nodes.add(v)
                else:
                    # Also mark if a node's home tile differs from resistor's tile
                    # (node appears in a neighboring tile's resistor list)
                    if tile_u is not None and tile_u != tile:
                        self._boundary_nodes.add(u)
                    if tile_v is not None and tile_v != tile:
                        self._boundary_nodes.add(v)

        print(f"  Found {len(self._boundary_nodes)} boundary nodes")

    def create_islands(self):
        """Create small floating islands on M0 by dropping resistors.

        This creates test cases for floating island detection by isolating
        small clusters of M0 nodes from the main connected component.
        """
        if self.config.n_islands <= 0:
            return

        print(f"Creating {self.config.n_islands} M0 islands...")

        # Find M0 nodes in each tile
        m0_nodes_by_tile: Dict[Tuple[int, int], List[str]] = defaultdict(list)
        for node, tile in self.node_to_tile.items():
            parsed = self._parse_die_node(node)
            if parsed:
                _, _, layer = parsed
                if layer == '24':  # M0 layer in original naming
                    m0_nodes_by_tile[tile].append(node)

        # Build M0-only adjacency from tile resistors
        m0_adj: Dict[str, Set[str]] = defaultdict(set)
        for tile in self.tile_resistors:
            for u, v, _ in self.tile_resistors[tile]:
                # Check if both are M0 nodes
                u_parsed = self._parse_die_node(u)
                v_parsed = self._parse_die_node(v)
                if u_parsed and v_parsed and u_parsed[2] == '24' and v_parsed[2] == '24':
                    m0_adj[u].add(v)
                    m0_adj[v].add(u)

        # Select tiles to place islands (distribute across tiles)
        tiles_with_m0 = [(tile, nodes) for tile, nodes in m0_nodes_by_tile.items()
                        if len(nodes) >= self.config.island_size_range[1]]
        random.shuffle(tiles_with_m0)

        islands_created = 0
        total_island_nodes = 0
        island_nodes_set: Set[str] = set()

        for tile, m0_nodes in tiles_with_m0:
            if islands_created >= self.config.n_islands:
                break

            # Find a seed node that's not already in an island
            available_seeds = [n for n in m0_nodes if n not in island_nodes_set and n in m0_adj]
            if not available_seeds:
                continue

            # Pick a random seed and grow a small cluster via BFS
            seed = random.choice(available_seeds)
            target_size = random.randint(*self.config.island_size_range)

            # BFS to find a cluster of M0 nodes
            cluster = set()
            queue = deque([seed])
            visited = {seed}

            while queue and len(cluster) < target_size:
                node = queue.popleft()
                if node in island_nodes_set:
                    continue
                cluster.add(node)
                for neighbor in m0_adj.get(node, set()):
                    if neighbor not in visited and neighbor not in island_nodes_set:
                        visited.add(neighbor)
                        queue.append(neighbor)

            if len(cluster) < self.config.island_size_range[0]:
                continue  # Cluster too small, skip

            # Remove ALL resistors connecting cluster to outside (including cross-layer vias)
            # Need to check ALL tiles since vias might be in different tiles
            removed_resistors = 0
            for check_tile in list(self.tile_resistors.keys()):
                new_tile_resistors = []
                for u, v, value in self.tile_resistors[check_tile]:
                    u_in_cluster = u in cluster
                    v_in_cluster = v in cluster
                    # Keep if both in cluster (internal) or both outside cluster
                    if u_in_cluster == v_in_cluster:
                        new_tile_resistors.append((u, v, value))
                    else:
                        # This resistor connects cluster to outside - remove it
                        removed_resistors += 1
                self.tile_resistors[check_tile] = new_tile_resistors

            island_nodes_set.update(cluster)
            total_island_nodes += len(cluster)
            islands_created += 1

            print(f"  Island {islands_created} in tile {tile}: {len(cluster)} nodes, "
                  f"removed {removed_resistors} boundary resistors (including vias)")

        self._island_nodes = island_nodes_set
        print(f"  Total island nodes: {total_island_nodes}")

    def _verify_islands(self):
        """Verify that created islands are truly disconnected from the main component.

        This does a BFS from M13 (vsrc-connected) nodes to verify no island nodes
        are reachable from the main connected component.
        """
        if not self._island_nodes:
            print("No islands to verify")
            return

        print("Verifying island connectivity...")

        # Build adjacency from all tile resistors (after island creation)
        adj: Dict[str, Set[str]] = defaultdict(set)
        for tile, resistors in self.tile_resistors.items():
            for u, v, _ in resistors:
                adj[u].add(v)
                adj[v].add(u)

        # Find M13 nodes (will have vsrc connections)
        m13_nodes = []
        for node in adj.keys():
            parsed = self._parse_die_node(node)
            if parsed and parsed[2] == '50':  # M13 layer
                m13_nodes.append(node)

        # BFS from M13 to find main connected component
        main_component = set()
        visited = set(m13_nodes)
        queue = deque(m13_nodes)

        while queue:
            node = queue.popleft()
            main_component.add(node)
            for neighbor in adj.get(node, set()):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        # Check for island nodes in main component
        leaked_nodes = self._island_nodes & main_component
        if leaked_nodes:
            print(f"  WARNING: {len(leaked_nodes)} island nodes still connected to main!")
            for node in list(leaked_nodes)[:5]:
                print(f"    {node}")
        else:
            print(f"  ✓ All {len(self._island_nodes)} island nodes are disconnected")

        # Count island distribution across tiles
        island_by_tile: Dict[Tuple[int, int], int] = defaultdict(int)
        for node in self._island_nodes:
            tile = self.node_to_tile.get(node)
            if tile:
                island_by_tile[tile] += 1

        print(f"  Island distribution across {len(island_by_tile)} tiles:")
        for tile in sorted(island_by_tile.keys()):
            print(f"    Tile {tile}: {island_by_tile[tile]} island nodes")

    def sample_voltage_sources(self):
        """Sample voltage sources connecting to M13 nodes in the sample region.

        Note: The original vsrc bumps may be at die edges (outside sample region).
        We create synthetic vsrc connections to M13 nodes that ARE in the sample region.
        """
        print("Sampling voltage sources...")

        # Find M13 (layer 50) nodes with resistor connections
        m13_nodes_in_region = []
        for node in self._nodes_with_resistors:
            parsed = self._parse_die_node(node)
            if parsed:
                _, _, layer = parsed
                if layer == '50':  # M13 layer in original naming
                    m13_nodes_in_region.append(node)

        print(f"  Found {len(m13_nodes_in_region)} M13 nodes in sample region")

        if not m13_nodes_in_region:
            print("  WARNING: No M13 nodes in sample region - no vsrc connections possible")
            self.sampled_vsrcs = []
            return

        # Sample M13 nodes to connect vsrc to (spread across the region)
        # Sort by coordinates to get spatial distribution
        def get_coords(node):
            parsed = self._parse_die_node(node)
            if parsed:
                return (parsed[0], parsed[1])
            return (0, 0)

        m13_sorted = sorted(m13_nodes_in_region, key=get_coords)

        # Sample uniformly spaced M13 nodes
        n_vsrc = min(self.config.target_vsrc_count, len(m13_sorted))
        step = max(1, len(m13_sorted) // n_vsrc)

        selected_m13 = []
        for i in range(0, len(m13_sorted), step):
            if len(selected_m13) >= n_vsrc:
                break
            selected_m13.append(m13_sorted[i])

        # Create synthetic vsrc names
        self.sampled_vsrcs = []
        for i, m13 in enumerate(selected_m13):
            vsrc_name = f"{self.config.net_name}_tap_{i:05d}_vsrc"
            self.sampled_vsrcs.append((vsrc_name, m13))

        print(f"  Created {len(self.sampled_vsrcs)} voltage source connections")

    def sample_current_sources(self):
        """Sample current sources from nodes in the sample region."""
        print("Sampling current sources...")

        # Get instance sources from graph
        instance_sources = self.graph.graph.get('_instance_sources_objects', {})
        if not instance_sources:
            # Try serialized format
            instance_sources_dict = self.graph.graph.get('instance_sources', {})
            print(f"  Warning: Using serialized instance_sources format")
            # Convert back to CurrentSource objects if needed
            instance_sources = instance_sources_dict

        print(f"  Found {len(instance_sources)} total current sources in graph")

        # Categorize sources by whether they're in sample region and have waveforms
        sources_with_waveform = []
        sources_dc_only = []

        for name, src in instance_sources.items():
            # Get the node this source connects to
            if isinstance(src, CurrentSource):
                node = src.node1
                has_waveform = src.has_waveform_data()
            elif isinstance(src, dict):
                node = src.get('node1', '')
                has_waveform = bool(src.get('pulses') or src.get('pwls'))
            else:
                continue

            # Check if node is in sample region AND has resistor connections
            if node in self._nodes_with_resistors:
                if has_waveform:
                    sources_with_waveform.append((name, src))
                else:
                    sources_dc_only.append((name, src))

        print(f"  Sources in sample region: {len(sources_with_waveform)} with waveforms, "
              f"{len(sources_dc_only)} DC-only")

        # Calculate how many to sample from each category
        target_waveform = int(self.config.target_current_sources * self.config.waveform_ratio)
        target_dc = self.config.target_current_sources - target_waveform

        # Sample from each category
        sampled_waveform = random.sample(sources_with_waveform,
                                         min(target_waveform, len(sources_with_waveform)))
        sampled_dc = random.sample(sources_dc_only,
                                   min(target_dc, len(sources_dc_only)))

        # If we didn't get enough from one category, take more from the other
        total_sampled = len(sampled_waveform) + len(sampled_dc)
        if total_sampled < self.config.target_current_sources:
            remaining = self.config.target_current_sources - total_sampled
            if len(sampled_waveform) < target_waveform:
                # Take more DC sources
                remaining_dc = [s for s in sources_dc_only if s not in sampled_dc]
                sampled_dc.extend(random.sample(remaining_dc, min(remaining, len(remaining_dc))))
            else:
                # Take more waveform sources
                remaining_wf = [s for s in sources_with_waveform if s not in sampled_waveform]
                sampled_waveform.extend(random.sample(remaining_wf, min(remaining, len(remaining_wf))))

        all_sampled = sampled_waveform + sampled_dc
        random.shuffle(all_sampled)  # Mix them up

        # Assign to tiles
        for i, (name, src) in enumerate(all_sampled):
            if isinstance(src, CurrentSource):
                node = src.node1
            else:
                node = src.get('node1', '')

            tile = self.node_to_tile.get(node)
            if tile:
                # Create new source with renamed name (include net/pin suffix)
                new_name = f"i_Inst{i:06d}:{self.config.net_name}:VDD:0:0:0:0:0"
                if isinstance(src, CurrentSource):
                    new_src = CurrentSource(
                        name=new_name,
                        node1=node,
                        node2=src.node2,
                        dc_value=src.dc_value,
                        static_value=src.static_value,
                        pulses=src.pulses.copy() if src.pulses else [],
                        pwls=src.pwls.copy() if src.pwls else [],
                        wscale=src.wscale
                    )
                    self.sampled_current_sources[tile].append(new_src)
                else:
                    # Handle dict format
                    self.sampled_current_sources[tile].append({
                        'name': new_name,
                        'node1': node,
                        'node2': src.get('node2', '0'),
                        'dc_value': src.get('dc_value', 0),
                        'static_value': src.get('static_value'),
                        'pulses': src.get('pulses', []),
                        'pwls': src.get('pwls', []),
                        'wscale': src.get('wscale', 1.0)
                    })

        total_sources = sum(len(v) for v in self.sampled_current_sources.values())
        waveform_count = sum(1 for tile in self.sampled_current_sources.values()
                            for src in tile if self._has_waveform(src))

        print(f"  Sampled {total_sources} current sources total")
        print(f"  Waveform ratio: {waveform_count/total_sources*100:.1f}%")

    def _has_waveform(self, src) -> bool:
        """Check if source has waveform data."""
        if isinstance(src, CurrentSource):
            return src.has_waveform_data()
        elif isinstance(src, dict):
            return bool(src.get('pulses') or src.get('pwls'))
        return False

    def generate_output_files(self):
        """Generate all output files."""
        print(f"Generating output files in {self.config.output_dir}...")

        os.makedirs(self.config.output_dir, exist_ok=True)

        self._generate_ckt_sp()
        self._generate_pg_net_voltage()
        self._generate_tile_files()
        self._generate_instance_models()
        self._generate_package_ckt()

        print("Done generating files!")

    def _generate_ckt_sp(self):
        """Generate main ckt.sp file."""
        filepath = os.path.join(self.config.output_dir, 'ckt.sp')

        lines = [
            f"* Sampled PDN netlist for {self.config.net_name}",
            f"* Generated from {self.config.source_pickle}",
            f"* {self.config.n_tiles_x}x{self.config.n_tiles_y} tile grid",
            "",
            f".partition_info {self.config.n_tiles_x} {self.config.n_tiles_y}",
            "",
            "* Power net voltage definitions",
            f".parameter {self.config.net_name} {self.config.vdd_voltage}",
            ".parameter VSS 0.0",
            "",
            "* Include tile netlists",
        ]

        for tile_y in range(self.config.n_tiles_y):
            for tile_x in range(self.config.n_tiles_x):
                lines.append(f".include tile_{tile_x}_{tile_y}.ckt")

        lines.append("")
        lines.append("* Include instance models")

        for tile_y in range(self.config.n_tiles_y):
            for tile_x in range(self.config.n_tiles_x):
                lines.append(f".include instanceModels_{tile_x}_{tile_y}.sp")

        lines.append("")
        lines.append("* Include package model")
        lines.append(".include package.ckt")
        lines.append("")
        lines.append(".end")
        lines.append("")

        with open(filepath, 'w') as f:
            f.write('\n'.join(lines))

        print(f"  Generated {filepath}")

    def _generate_pg_net_voltage(self):
        """Generate pg_net_voltage file."""
        filepath = os.path.join(self.config.output_dir, 'pg_net_voltage')

        with open(filepath, 'w') as f:
            f.write(f"{self.config.net_name} {self.config.vdd_voltage}\n")
            f.write("VSS 0.0\n")

        print(f"  Generated {filepath}")

    def _generate_tile_files(self):
        """Generate tile_X_Y.ckt and tile_X_Y.nd files."""
        for tile_y in range(self.config.n_tiles_y):
            for tile_x in range(self.config.n_tiles_x):
                tile = (tile_x, tile_y)
                self._generate_single_tile_ckt(tile)
                self._generate_single_tile_nd(tile)

    def _format_node_for_ckt(self, orig_node: str) -> str:
        """Format node name for .ckt file, adding '*' prefix for boundary nodes."""
        renamed = self._rename_node(orig_node)
        if orig_node in self._boundary_nodes:
            return f"*{renamed}"
        return renamed

    def _generate_single_tile_ckt(self, tile: Tuple[int, int]):
        """Generate a single tile .ckt file."""
        tile_x, tile_y = tile
        filepath = os.path.join(self.config.output_dir, f'tile_{tile_x}_{tile_y}.ckt')

        lines = [
            f"* Tile {tile_x}_{tile_y} - Die resistor mesh",
            f"* Sampled from {self.config.net_name}",
            "",
        ]

        # Write resistors (convert from kOhm to Ohms for SPICE format)
        lines.append("* Resistors")
        KOHM_TO_OHM = 1e3  # kOhm to Ohms conversion
        for u, v, value in self.tile_resistors[tile]:
            new_u = self._format_node_for_ckt(u)
            new_v = self._format_node_for_ckt(v)
            # Graph stores values in kOhm, SPICE expects Ohms
            value_ohm = value * KOHM_TO_OHM
            lines.append(f"r {new_u} {new_v} {value_ohm:.6g}")

        lines.append("")

        # Write capacitors (convert from fF to Farads for SPICE format)
        lines.append("* Capacitors")
        for u, v, value in self.tile_capacitors[tile]:
            new_u = self._format_node_for_ckt(u)
            new_v = '0' if v == '0' else self._format_node_for_ckt(v)
            # Graph stores values in fF, SPICE expects Farads
            value_farads = value * 1e-15
            lines.append(f"c {new_u} {new_v} {value_farads:.6e}")

        lines.append("")

        with open(filepath, 'w') as f:
            f.write('\n'.join(lines))

        print(f"  Generated {filepath} ({len(self.tile_resistors[tile])} R, "
              f"{len(self.tile_capacitors[tile])} C)")

    def _generate_single_tile_nd(self, tile: Tuple[int, int]):
        """Generate a single tile .nd file."""
        tile_x, tile_y = tile
        filepath = os.path.join(self.config.output_dir, f'tile_{tile_x}_{tile_y}.nd')

        lines = []

        for node in sorted(self.tile_nodes[tile]):
            parsed = self._parse_die_node(node)
            if parsed:
                x, y, layer = parsed
                new_layer = self._rename_layer(layer)
                new_node = f"{x}_{y}_{new_layer}"
                # Format: node_name x y layer 0 net_name
                lines.append(f"{new_node} {x} {y} {new_layer} 0 {self.config.net_name}")

        with open(filepath, 'w') as f:
            f.write('\n'.join(lines))
            f.write('\n')

        print(f"  Generated {filepath} ({len(self.tile_nodes[tile])} nodes)")

    def _generate_instance_models(self):
        """Generate instanceModels_X_Y.sp files."""
        for tile_y in range(self.config.n_tiles_y):
            for tile_x in range(self.config.n_tiles_x):
                tile = (tile_x, tile_y)
                self._generate_single_instance_model(tile)

    def _generate_single_instance_model(self, tile: Tuple[int, int]):
        """Generate a single instanceModels .sp file."""
        tile_x, tile_y = tile
        filepath = os.path.join(self.config.output_dir, f'instanceModels_{tile_x}_{tile_y}.sp')

        sources = self.sampled_current_sources[tile]

        lines = [
            f"* Instance current sources for tile {tile_x}_{tile_y}",
            f"* {len(sources)} current sources",
            "",
        ]

        for src in sources:
            line = self._format_current_source(src)
            if line:
                lines.append(line)

        lines.append("")

        with open(filepath, 'w') as f:
            f.write('\n'.join(lines))

        waveform_count = sum(1 for src in sources if self._has_waveform(src))
        print(f"  Generated {filepath} ({len(sources)} sources, "
              f"{waveform_count} with waveforms)")

    def _format_current_source(self, src) -> str:
        """Format a current source as SPICE string.

        Note: Graph stores current values in mA, but SPICE format expects Amperes.
        All current values are converted from mA to A (divide by 1000).
        """
        MA_TO_A = 1e-3  # mA to Amperes conversion

        if isinstance(src, CurrentSource):
            name = src.name
            node1 = self._rename_node(src.node1)
            node2 = src.node2
            dc_value = src.dc_value * MA_TO_A  # Convert mA to A
            static_value = src.static_value * MA_TO_A if src.static_value is not None else None
            pulses = src.pulses
            pwls = src.pwls
            wscale = src.wscale
        else:
            name = src['name']
            node1 = self._rename_node(src['node1'])
            node2 = src.get('node2', '0')
            dc_value = src.get('dc_value', 0) * MA_TO_A  # Convert mA to A
            sv = src.get('static_value')
            static_value = sv * MA_TO_A if sv is not None else None
            pulses = src.get('pulses', [])
            pwls = src.get('pwls', [])
            wscale = src.get('wscale', 1.0)

        # Format: i_name node1 node2 dc_value [static_value=X] [pulse(...)] [pwl(...)] [wscale=X]
        parts = [f"{name} {node1} {node2} {dc_value:.6e}"]

        if static_value is not None:
            parts.append(f"static_value={static_value:.6e}")

        for pulse in pulses:
            if isinstance(pulse, Pulse):
                # Convert pulse v1, v2 from mA to A
                v1_a = pulse.v1 * MA_TO_A
                v2_a = pulse.v2 * MA_TO_A
                parts.append(f"pulse({v1_a},{v2_a},{pulse.delay},"
                           f"{pulse.rt},{pulse.ft},{pulse.width},{pulse.period})")
            elif isinstance(pulse, dict):
                v1_a = pulse.get('v1', 0) * MA_TO_A
                v2_a = pulse.get('v2', 0) * MA_TO_A
                parts.append(f"pulse({v1_a},{v2_a},"
                           f"{pulse.get('delay', 0)},{pulse.get('rt', 0)},"
                           f"{pulse.get('ft', 0)},{pulse.get('width', 0)},"
                           f"{pulse.get('period', 0)})")

        for pwl in pwls:
            if isinstance(pwl, PWL):
                # Convert PWL values from mA to A
                points_str = ' '.join(f"{t} {v * MA_TO_A}" for t, v in pwl.points)
                parts.append(f"pwl({points_str})")
                if pwl.period > 0:
                    parts.append(f"pwl_period={pwl.period}")
                if pwl.delay > 0:
                    parts.append(f"pwl_delay={pwl.delay}")
            elif isinstance(pwl, dict):
                points = pwl.get('points', [])
                points_str = ' '.join(f"{t} {v * MA_TO_A}" for t, v in points)
                parts.append(f"pwl({points_str})")
                if pwl.get('period', 0) > 0:
                    parts.append(f"pwl_period={pwl['period']}")
                if pwl.get('delay', 0) > 0:
                    parts.append(f"pwl_delay={pwl['delay']}")

        if wscale != 1.0:
            parts.append(f"wscale={wscale}")

        return ' '.join(parts)

    def _generate_package_ckt(self):
        """Generate package.ckt file with voltage sources and bumps."""
        filepath = os.path.join(self.config.output_dir, 'package.ckt')

        lines = [
            f"* Package model for {self.config.net_name}",
            f"* {len(self.sampled_vsrcs)} voltage sources",
            "",
            f"* Voltage source for {self.config.net_name}",
            f"v_{self.config.net_name} {self.config.net_name}_vsrc 0 {self.config.net_name}",
            "",
            "* Bump connections (vsrc to die)",
        ]

        for i, (vsrc_node, die_node) in enumerate(self.sampled_vsrcs):
            tap_name = f"{self.config.net_name}_tap_{i:05d}"
            new_die_node = self._rename_node(die_node)
            # Two resistors: vsrc -> tap, tap -> die
            lines.append(f"r {self.config.net_name}_vsrc {tap_name} 0.001")
            lines.append(f"r {tap_name} {new_die_node} 0.001")

        lines.append("")

        with open(filepath, 'w') as f:
            f.write('\n'.join(lines))

        print(f"  Generated {filepath} ({len(self.sampled_vsrcs)} bumps)")

    def compute_statistics(self):
        """Compute and print statistics."""
        print("\n=== Final Statistics ===")

        # Count from actual tile data structures (after filtering)
        total_nodes = sum(len(nodes) for nodes in self.tile_nodes.values())
        total_resistors = sum(len(r) for r in self.tile_resistors.values())
        total_capacitors = sum(len(c) for c in self.tile_capacitors.values())
        total_vsrcs = len(self.sampled_vsrcs)
        total_isrcs = sum(len(v) for v in self.sampled_current_sources.values())

        waveform_count = sum(1 for tile in self.sampled_current_sources.values()
                            for src in tile if self._has_waveform(src))

        # Compute total current and capacitance
        total_current = 0
        total_capacitance = 0

        for tile in self.sampled_current_sources.values():
            for src in tile:
                if isinstance(src, CurrentSource):
                    total_current += src.get_static_current() if hasattr(src, 'get_static_current') else src.dc_value
                else:
                    total_current += src.get('static_value') or src.get('dc_value', 0)

        for tile in self.tile_capacitors.values():
            for u, v, value in tile:
                total_capacitance += value

        self.stats = {
            'nodes': total_nodes,
            'resistors': total_resistors,
            'capacitors': total_capacitors,
            'voltage_sources': total_vsrcs,
            'current_sources': total_isrcs,
            'waveform_ratio': waveform_count / total_isrcs if total_isrcs > 0 else 0,
            'total_current_mA': total_current,
            'total_capacitance_fF': total_capacitance,
            'boundary_nodes': len(self._boundary_nodes),
            'island_nodes': len(self._island_nodes),
        }

        print(f"  Nodes: {total_nodes:,}")
        print(f"  Resistors: {total_resistors:,}")
        print(f"  Capacitors: {total_capacitors:,}")
        print(f"  Voltage sources: {total_vsrcs}")
        print(f"  Current sources: {total_isrcs:,}")
        print(f"  Waveform ratio: {waveform_count/total_isrcs*100:.1f}%")
        print(f"  Total current: {total_current:.2f} mA")
        print(f"  Total capacitance: {total_capacitance:.2f} fF")
        print(f"  Boundary nodes: {len(self._boundary_nodes):,}")
        print(f"  Island nodes: {len(self._island_nodes):,}")

    def validate_output(self):
        """Validate the generated netlist by parsing and solving."""
        print("\n=== Validation ===")

        from .netlist import NetlistParser
        from model.factory import create_model_from_pdn
        from solver.unified_solver import UnifiedIRDropSolver

        # Parse with 1 process
        print("Parsing with 1 process...")
        parser = NetlistParser(self.config.output_dir, parallel=False)
        try:
            graph = parser.parse()
            print(f"  Success! {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
        except Exception as e:
            print(f"  FAILED: {e}")
            return False

        # Parse with 3 processes
        print("Parsing with 3 processes...")
        parser = NetlistParser(self.config.output_dir, parallel=True, n_workers=3)
        try:
            graph = parser.parse()
            print(f"  Success! {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
        except Exception as e:
            print(f"  FAILED: {e}")
            return False

        # Solve
        print("Solving...")
        try:
            model = create_model_from_pdn(graph, self.config.net_name)
            load_currents = model.extract_current_sources()
            solver = UnifiedIRDropSolver(model)
            result = solver.solve(load_currents)

            max_ir_drop_mV = max(result.ir_drop.values()) * 1000
            print(f"  Success! Max IR-drop: {max_ir_drop_mV:.2f} mV")

            if max_ir_drop_mV > 50:
                print(f"  Warning: IR-drop is higher than expected")

        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
            return False

        print("\n=== Validation PASSED ===")
        return True

    def run(self):
        """Run the full generation pipeline."""
        print("=" * 60)
        print("Generating Sampled Multi-Tile PDN Netlist")
        print("=" * 60)
        print()

        self.load_source_graph()
        self.select_sampling_region()
        self.extract_sampled_subgraph()
        self.partition_into_tiles()
        self._compute_boundary_nodes()
        self.create_islands()
        self._verify_islands()
        self.sample_voltage_sources()
        self.sample_current_sources()
        self.generate_output_files()
        self.compute_statistics()

        print()
        success = self.validate_output()

        return success


def main():
    config = SamplingConfig()
    generator = SampledNetlistGenerator(config)
    success = generator.run()

    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())

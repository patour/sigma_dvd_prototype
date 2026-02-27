"""Tile worker for distributed DDM solver.

Thin stateful actor that holds per-tile BlockMatrixSystem and delegates
ALL math to solver/coupled_system.py building blocks.

Parsing helpers (_parse_spice_value, _parse_current_source_line) are
imported lazily from parser sub-modules to avoid circular imports
(parser -> solver -> distributed).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Unit conversions matching pdn_parser.py
R_TO_KOHM = 1e-3  # Ohm to kOhm
I_TO_MA = 1e3  # Ampere to mA


@dataclass
class TileData:
    """Parsed tile data. Serializable for Ray."""

    tile_id: Tuple[int, int]
    resistive_edges: List[Tuple[str, str, float]]  # (u, v, conductance_mS)
    all_nodes: Set[str]
    boundary_nodes: Set[str]
    current_injections: Dict[str, float]  # node -> current in mA (positive = sink)


def _load_nd_file(nd_path: str) -> Dict[str, str]:
    """Load .nd file for node-net mapping (lowercase net names).

    Returns:
        node -> lowercase net name mapping
    """
    import gzip

    node_net_map_lower: Dict[str, str] = {}

    if nd_path is None:
        return node_net_map_lower

    open_fn = gzip.open if nd_path.endswith('.gz') else open
    mode = 'rt' if nd_path.endswith('.gz') else 'r'

    with open_fn(nd_path, mode) as f:
        for line in f:
            parts = line.strip().split()
            # Format: node_name x y layer tile_id net_name
            if len(parts) >= 6:
                node_net_map_lower[parts[0]] = parts[5].lower()

    return node_net_map_lower


def _parse_tile_ckt(
    ckt_path: str,
    nd_path: Optional[str],
    net_filter: Optional[str],
    tile_id: Tuple[int, int],
) -> TileData:
    """Parse a tile .ckt file to extract resistive edges and current injections.

    Args:
        ckt_path: Path to tile .ckt file
        nd_path: Path to tile .nd file (for net filtering)
        net_filter: Optional lowercase net name to filter by
        tile_id: Tile identifier

    Returns:
        TileData with parsed edges and nodes
    """
    import gzip
    from parser.spice_lexer import _parse_spice_value

    node_net_map_lower = _load_nd_file(nd_path)

    resistive_edges: List[Tuple[str, str, float]] = []
    current_injections: Dict[str, float] = {}
    all_nodes: Set[str] = set()
    boundary_nodes: Set[str] = set()

    GMAX = 1e5
    SHORT_THRESHOLD = 1e-6

    open_fn = gzip.open if ckt_path.endswith('.gz') else open
    mode = 'rt' if ckt_path.endswith('.gz') else 'r'

    with open_fn(ckt_path, mode) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('*'):
                continue

            tokens = line.split()
            if len(tokens) < 4:
                continue

            first = tokens[0].lower()

            # Resistors
            if first[0] == 'r':
                if first == 'r':
                    # Unnamed: r node1 node2 value
                    node1, node2, value_token = tokens[1], tokens[2], tokens[3]
                else:
                    # Named: R_name node1 node2 value
                    node1, node2, value_token = tokens[1], tokens[2], tokens[3]

                # Handle boundary markers
                is_bnd1 = node1.startswith('*')
                is_bnd2 = node2.startswith('*')
                if is_bnd1:
                    node1 = node1[1:]
                    boundary_nodes.add(node1)
                if is_bnd2:
                    node2 = node2[1:]
                    boundary_nodes.add(node2)

                # Net filter
                if net_filter is not None:
                    n1_net = node_net_map_lower.get(node1)
                    n2_net = node_net_map_lower.get(node2)
                    if n1_net != net_filter and n2_net != net_filter:
                        continue

                try:
                    r_value = _parse_spice_value(value_token)
                except ValueError:
                    continue

                # Convert to kOhm then conductance (mS)
                r_kohm = r_value * R_TO_KOHM
                if r_kohm <= 0 or r_kohm < SHORT_THRESHOLD:
                    g = GMAX
                else:
                    g = 1.0 / r_kohm

                resistive_edges.append((node1, node2, g))
                all_nodes.add(node1)
                all_nodes.add(node2)

            # Current sources (for DC value extraction)
            elif first[0] == 'i':
                if first == 'i':
                    continue  # Unnamed current source without enough info
                # Named: I_name node+ node- dc_value ...
                node_pos = tokens[1]
                node_neg = tokens[2]

                if node_pos.startswith('*'):
                    node_pos = node_pos[1:]
                if node_neg.startswith('*'):
                    node_neg = node_neg[1:]

                # Net filter
                if net_filter is not None:
                    n1_net = node_net_map_lower.get(node_pos)
                    n2_net = node_net_map_lower.get(node_neg)
                    if n1_net != net_filter and n2_net != net_filter:
                        continue

                try:
                    dc_value = _parse_spice_value(tokens[3])
                    dc_ma = dc_value * I_TO_MA  # Convert A to mA
                except (ValueError, IndexError):
                    continue

                # Current source from node_pos to node_neg (pos = net node, neg = ground)
                if node_pos != '0':
                    current_injections[node_pos] = current_injections.get(node_pos, 0.0) + dc_ma

    return TileData(
        tile_id=tile_id,
        resistive_edges=resistive_edges,
        all_nodes=all_nodes,
        boundary_nodes=boundary_nodes,
        current_injections=current_injections,
    )


def _parse_instance_models(
    instance_path: str,
    net_filter: Optional[str],
    nd_path: Optional[str] = None,
) -> Dict[str, float]:
    """Parse instanceModels file for current source DC values.

    Uses shared _prepare_instance_source() for parsing + A→mA conversion,
    matching the flat parser's handling exactly.

    Args:
        instance_path: Path to instanceModels*.sp file
        net_filter: Optional lowercase net name to filter by
        nd_path: Path to .nd file for net filtering (required when net_filter is set)

    Returns:
        Dict mapping node -> current in mA (positive = sink)
    """
    import gzip
    from parser.current_sources import _prepare_instance_source
    from parser.spice_lexer import (
        _check_net_filter,
        _fast_instance_net_filter, _has_structured_instance_names,
    )

    current_injections: Dict[str, float] = {}

    # Detect structured names for fast net filtering
    use_fast_filter = (
        net_filter is not None
        and _has_structured_instance_names(instance_path)
    )

    # Only load .nd file when net_filter is set AND names are not structured
    if net_filter and not use_fast_filter:
        node_net_map_lower = _load_nd_file(nd_path)
    else:
        node_net_map_lower = {}

    open_fn = gzip.open if instance_path.endswith('.gz') else open
    mode = 'rt' if instance_path.endswith('.gz') else 'r'

    with open_fn(instance_path, mode) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('*') or line.startswith('.'):
                continue

            # Fast path: reject non-matching nets before expensive parsing
            if use_fast_filter and not _fast_instance_net_filter(line, net_filter):
                continue

            prepared = _prepare_instance_source(line)
            if prepared is None:
                continue

            # Slow path fallback: .nd-based filtering for unstructured names
            if net_filter and not use_fast_filter:
                if not _check_net_filter(prepared.node_pos, prepared.node_neg,
                                         node_net_map_lower, net_filter):
                    continue

            # Inject at positive terminal only: instance model current sources
            # are always node+ -> ground ('0'), so the negative terminal is
            # eliminated from the nodal system and needs no entry.
            if prepared.node_pos != '0':
                current_injections[prepared.node_pos] = (
                    current_injections.get(prepared.node_pos, 0.0) + prepared.static_current_ma
                )

    return current_injections


def parse_tile_with_instances(
    ckt_path: str,
    nd_path: Optional[str],
    net_filter: Optional[str],
    tile_id: Tuple[int, int],
    instance_path: Optional[str] = None,
) -> TileData:
    """Parse a tile .ckt file and merge instance model current injections.

    Combines _parse_tile_ckt() + _parse_instance_models() + merge into a
    single call, eliminating duplication between TileWorker.setup() and
    DistributedNetlistParser.parse_and_dump().

    Args:
        ckt_path: Path to tile .ckt file
        nd_path: Path to tile .nd file (for net filtering)
        net_filter: Optional lowercase net name to filter by
        tile_id: Tile identifier tuple (x, y)
        instance_path: Optional path to instanceModels*.sp file

    Returns:
        TileData with parsed edges, nodes, and merged current injections
    """
    tile_data = _parse_tile_ckt(ckt_path, nd_path, net_filter, tile_id)

    if instance_path:
        inst_currents = _parse_instance_models(instance_path, net_filter, nd_path)
        for node, current in inst_currents.items():
            if node in tile_data.all_nodes:
                tile_data.current_injections[node] = (
                    tile_data.current_injections.get(node, 0.0) + current
                )

    return tile_data


class TileWorker:
    """Per-tile actor for distributed DDM. Thin wrapper that delegates
    all math to coupled_system.py building blocks.

    Intentionally thin - manages state, not algorithms.
    """

    def __init__(self):
        self._tile_data: Optional[TileData] = None
        self._block_system = None
        self._rhs_dirichlet = None
        self._interface_nodes: Optional[Set[str]] = None

    def setup(
        self,
        tile_config_dict: Dict[str, Any],
        interface_nodes: Set[str],
    ) -> Dict[str, Any]:
        """Parse tile and build block system (without factoring).

        Args:
            tile_config_dict: Dict with tile_id, ckt_path, nd_path, instance_path, net_filter
            interface_nodes: Set of all global interface nodes (boundary + die attachment)

        Returns:
            Dict with tile metadata: {tile_id, boundary_nodes, n_interior, n_boundary, islands_removed}
        """
        tc = tile_config_dict
        tile_id = tuple(tc['tile_id'])

        self._tile_data = parse_tile_with_instances(
            ckt_path=tc['ckt_path'],
            nd_path=tc.get('nd_path'),
            net_filter=tc.get('net_filter'),
            tile_id=tile_id,
            instance_path=tc.get('instance_path'),
        )

        return self._build_block_system(interface_nodes)

    def setup_from_tile_data(
        self,
        tile_data: TileData,
        interface_nodes: Set[str],
    ) -> Dict[str, Any]:
        """Build block system from pre-parsed TileData (no file I/O).

        Accepts a TileData that was previously produced by _parse_tile_ckt()
        and _parse_instance_models() (e.g., loaded from a .pkl file).
        Performs island detection and builds BlockMatrixSystem, identical to
        the bottom half of setup().

        Args:
            tile_data: Pre-parsed tile data (edges, nodes, currents)
            interface_nodes: Set of all global interface nodes (boundary + die attachment)

        Returns:
            Dict with tile metadata: {tile_id, boundary_nodes, n_interior, n_boundary, islands_removed}
        """
        self._tile_data = tile_data
        return self._build_block_system(interface_nodes)

    def _build_block_system(
        self,
        interface_nodes: Set[str],
    ) -> Dict[str, Any]:
        """Shared logic: island detection + BlockMatrixSystem construction.

        Called by both setup() and setup_from_tile_data() after tile_data
        has been populated.

        Args:
            interface_nodes: Set of all global interface nodes

        Returns:
            Dict with tile metadata: {tile_id, boundary_nodes, n_interior, n_boundary, islands_removed}
        """
        self._interface_nodes = interface_nodes

        # Classify: port nodes = interface nodes present in this tile
        port_nodes_local = interface_nodes & self._tile_data.all_nodes

        # Floating island detection: remove components not connected to any port/boundary
        islands_removed = 0
        if port_nodes_local:
            islands_removed = self._remove_floating_islands(port_nodes_local)

        # Build BlockMatrixSystem from edges (no factorization yet)
        from solver.coupled_system import build_block_system_from_edges
        self._block_system, self._rhs_dirichlet = build_block_system_from_edges(
            edges=self._tile_data.resistive_edges,
            port_nodes=port_nodes_local,
            dirichlet_nodes=None,
            dirichlet_voltage=0.0,
            ground_node='0',
        )

        return {
            'tile_id': self._tile_data.tile_id,
            'boundary_nodes': self._block_system.port_nodes,
            'n_interior': self._block_system.n_interior,
            'n_boundary': self._block_system.n_ports,
            'islands_removed': islands_removed,
        }

    # Minimum interface node count for a non-largest component to be kept.
    # Components with fewer interface nodes are likely small fragments whose
    # boundary nodes form near-singular chains in the interface system.
    # Components with many interface nodes (>= threshold) are legitimate
    # cross-tile strips that connect to the main network through other tiles.
    MIN_INTERFACE_NODES_KEEP = 5

    def _remove_floating_islands(self, port_nodes: Set[str]) -> int:
        """Remove disconnected components that are isolated fragments.

        Keeps the largest component plus any component with enough interface
        node connectivity (>= MIN_INTERFACE_NODES_KEEP interface nodes) to
        be a legitimate cross-tile strip. Removes small fragments whose few
        boundary nodes would create near-singular interface blocks.

        Returns:
            Number of islands removed
        """
        # Build adjacency
        adj: Dict[str, Set[str]] = {}
        for u, v, g in self._tile_data.resistive_edges:
            if u == '0' or v == '0':
                continue
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)

        # Find connected components via BFS (exclude ground node — handled separately)
        visited: Set[str] = set()
        components: List[Set[str]] = []
        for start_node in self._tile_data.all_nodes:
            if start_node in visited or start_node == '0':
                continue
            comp: Set[str] = set()
            queue = [start_node]
            while queue:
                node = queue.pop()
                if node in visited:
                    continue
                visited.add(node)
                comp.add(node)
                for nb in adj.get(node, set()):
                    if nb not in visited:
                        queue.append(nb)
            components.append(comp)

        if len(components) <= 1:
            return 0

        # Keep largest component + components with sufficient interface connectivity
        largest = max(components, key=len)
        removed_nodes: Set[str] = set()
        islands_removed = 0
        for comp in components:
            if comp is largest:
                continue
            n_interface = len(comp & port_nodes)
            if n_interface >= self.MIN_INTERFACE_NODES_KEEP:
                continue  # Legitimate cross-tile strip
            removed_nodes.update(comp)
            islands_removed += 1

        if not removed_nodes:
            return 0

        self._tile_data.all_nodes -= removed_nodes
        self._tile_data.boundary_nodes -= removed_nodes
        self._tile_data.resistive_edges = [
            (u, v, g) for u, v, g in self._tile_data.resistive_edges
            if u not in removed_nodes and v not in removed_nodes
        ]
        for node in removed_nodes:
            self._tile_data.current_injections.pop(node, None)

        return islands_removed

    def factor_and_compute_schur(self) -> Tuple[Any, List[str]]:
        """Factor interior and compute explicit Schur complement.

        Returns:
            Tuple of (S_i as numpy array, boundary_node_list)
        """
        from solver.coupled_system import compute_explicit_schur

        self._block_system.factor_interior()
        S = compute_explicit_schur(self._block_system)
        return S, list(self._block_system.port_nodes)

    def get_reduced_rhs(self, current_injections: Optional[Dict[str, float]] = None) -> Any:
        """Compute reduced RHS for this tile.

        Args:
            current_injections: Override current injections. If None, uses tile's own.

        Returns:
            Reduced RHS numpy array (n_ports,)
        """
        from solver.coupled_system import compute_reduced_rhs

        if current_injections is None:
            current_injections = self._tile_data.current_injections

        return compute_reduced_rhs(
            self._block_system, current_injections, self._rhs_dirichlet,
        )

    def get_interior_voltages(
        self,
        boundary_voltages_dict: Dict[str, float],
        current_injections: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """Recover interior voltages from boundary voltages.

        Args:
            boundary_voltages_dict: Dict mapping boundary node -> voltage
            current_injections: Optional override currents (node -> mA).
                If None, uses tile's own current sources from parsing.

        Returns:
            Dict mapping all tile nodes (interior + boundary) -> voltage
        """
        from solver.coupled_system import recover_bottom_voltages

        currents = current_injections if current_injections is not None else self._tile_data.current_injections

        # Convert boundary dict to array in port_nodes order
        port_voltages = np.zeros(self._block_system.n_ports, dtype=np.float64)
        for node, idx in self._block_system.port_to_idx.items():
            if node in boundary_voltages_dict:
                port_voltages[idx] = boundary_voltages_dict[node]

        # Recover interior
        interior_voltages = recover_bottom_voltages(
            self._block_system,
            port_voltages,
            currents,
            self._rhs_dirichlet,
        )

        # Combine boundary + interior voltages
        all_voltages = dict(interior_voltages)
        for node in self._block_system.port_nodes:
            if node in boundary_voltages_dict:
                all_voltages[node] = boundary_voltages_dict[node]

        return all_voltages

    @property
    def tile_id(self) -> Tuple[int, int]:
        return self._tile_data.tile_id if self._tile_data else None

    @property
    def n_interior(self) -> int:
        return self._block_system.n_interior if self._block_system else 0

    @property
    def n_boundary(self) -> int:
        return self._block_system.n_ports if self._block_system else 0

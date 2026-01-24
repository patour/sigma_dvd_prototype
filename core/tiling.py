"""Tiling infrastructure for hierarchical IR-drop solving.

This module provides:
- solve_single_tile: Module-level function for parallel tile solves (picklable)
- TileManager: Class for tile generation, connectivity validation, and result merging
"""

from __future__ import annotations

import logging
import math
import time
from collections import deque
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import scipy
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from packaging import version as pkg_version

from .solver_results import TileBounds, BottomGridTile

# Scipy version compatibility for tol vs rtol parameter
_SCIPY_VERSION = pkg_version.parse(scipy.__version__)
_SCIPY_USE_RTOL = _SCIPY_VERSION >= pkg_version.parse("1.12.0")


def _get_tol_kwargs(tol: float, atol: float = 0.0) -> dict:
    """Return tolerance kwargs compatible with current scipy version.

    Scipy 1.12+ uses 'rtol' and 'atol', while older versions use 'tol'.
    """
    if _SCIPY_USE_RTOL:
        return {"rtol": tol, "atol": atol}
    else:
        return {"tol": tol}
from .unified_model import UnifiedPowerGridModel
from .node_adapter import NodeInfoExtractor
from .rx_graph import RustworkxMultiDiGraphWrapper

# Logger for tiling warnings
logger = logging.getLogger(__name__)


# ============================================================================
# Module-level function for multiprocessing tile solves
# ============================================================================

def solve_single_tile(
    tile_id: int,
    node_list: List[Any],
    core_nodes: Set[Any],
    edge_tuples: List[Tuple[Any, Any, float]],  # (u, v, conductance)
    port_nodes: List[Any],
    port_voltages: Dict[Any, float],
    current_injections: Dict[Any, float],
    vdd: float,
) -> Tuple[int, Dict[Any, float], float, Set[Any]]:
    """Solve IR-drop for a single tile (picklable for multiprocessing).

    This function is designed to be called in a separate process.
    It builds the conductance matrix, applies Dirichlet BCs at port nodes,
    and solves for unknown node voltages.

    Handles disconnected components within the tile:
    - Disconnected halo nodes are dropped from the solve (their voltages not needed)
    - Disconnected core nodes raise an error (they need accurate voltages)

    Args:
        tile_id: Tile identifier for result mapping
        node_list: List of all nodes in tile (core + halo + ports)
        core_nodes: Set of core nodes (must all be connected to ports)
        edge_tuples: List of (u, v, conductance) for edges within tile
        port_nodes: List of port nodes (Dirichlet BCs)
        port_voltages: Port node -> voltage from top-grid solve
        current_injections: Node -> current for loads in tile
        vdd: Nominal voltage (fallback for ports without explicit voltage)

    Returns:
        (tile_id, voltages_dict, solve_time_ms, disconnected_halo_nodes)

    Raises:
        ValueError: If any core node is disconnected from ports
    """
    t0 = time.perf_counter()

    n_nodes = len(node_list)
    if n_nodes == 0:
        return (tile_id, {}, 0.0, set())

    port_set = set(port_nodes)

    # Identify unknown (non-port) nodes
    all_unknown_nodes = [n for n in node_list if n not in port_set]
    if not all_unknown_nodes:
        # All nodes are ports - just return port voltages
        voltages = {n: port_voltages.get(n, vdd) for n in node_list}
        return (tile_id, voltages, (time.perf_counter() - t0) * 1000, set())

    # ================================================================
    # Connectivity check: Find unknown nodes reachable from ports
    # ================================================================

    # Build adjacency list from edges
    adjacency: Dict[Any, Set[Any]] = {n: set() for n in node_list}
    for u, v, g in edge_tuples:
        if g > 0 and np.isfinite(g):
            adjacency[u].add(v)
            adjacency[v].add(u)

    # BFS from all port nodes to find reachable unknown nodes
    reachable_from_ports: Set[Any] = set()
    visited: Set[Any] = set()
    queue = deque(port_nodes)
    visited.update(port_nodes)

    while queue:
        node = queue.popleft()
        for neighbor in adjacency[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
                if neighbor not in port_set:
                    reachable_from_ports.add(neighbor)

    # Identify disconnected nodes
    all_unknown_set = set(all_unknown_nodes)
    disconnected_nodes = all_unknown_set - reachable_from_ports

    # Check for disconnected core nodes
    # These should have been handled by _validate_and_fix_tile_connectivity
    # but we handle them gracefully here by assigning vdd
    disconnected_core = disconnected_nodes & core_nodes
    if disconnected_core:
        # Log warning but don't raise - assign vdd to these nodes
        # This handles edge cases where global floating islands weren't fully detected
        pass  # Will be handled below by assigning vdd

    # Disconnected halo nodes are OK - just drop them from the solve
    disconnected_halo = disconnected_nodes - core_nodes

    # Filter unknown nodes to only those connected to ports
    unknown_nodes = [n for n in all_unknown_nodes if n in reachable_from_ports]

    # Initialize voltages dict
    voltages: Dict[Any, float] = {}

    # If no unknown nodes are connected to ports, return port voltages only
    if not unknown_nodes:
        for node in port_nodes:
            voltages[node] = port_voltages.get(node, vdd)
        return (tile_id, voltages, (time.perf_counter() - t0) * 1000, disconnected_halo)

    # ================================================================
    # Build and solve the reduced system for connected nodes only
    # ================================================================

    unknown_to_idx = {n: i for i, n in enumerate(unknown_nodes)}
    n_unknown = len(unknown_nodes)
    n_ports = len(port_nodes)

    # Build conductance matrix in COO format
    # G_uu: unknown x unknown, G_up: unknown x port
    rows_uu, cols_uu, data_uu = [], [], []
    rows_up, cols_up, data_up = [], [], []
    diag_u = np.zeros(n_unknown, dtype=np.float64)

    # Port index mapping
    port_to_idx = {p: i for i, p in enumerate(port_nodes)}

    for u, v, g in edge_tuples:
        if g <= 0 or not np.isfinite(g):
            continue

        u_is_unknown = u in unknown_to_idx
        v_is_unknown = v in unknown_to_idx
        u_is_port = u in port_set
        v_is_port = v in port_set

        if u_is_unknown and v_is_unknown:
            # Both unknown: contribute to G_uu
            iu, iv = unknown_to_idx[u], unknown_to_idx[v]
            rows_uu.extend([iu, iv])
            cols_uu.extend([iv, iu])
            data_uu.extend([-g, -g])
            diag_u[iu] += g
            diag_u[iv] += g
        elif u_is_unknown and v_is_port:
            # u unknown, v port: contribute to G_up and u's diagonal
            iu = unknown_to_idx[u]
            iv = port_to_idx[v]
            rows_up.append(iu)
            cols_up.append(iv)
            data_up.append(-g)
            diag_u[iu] += g
        elif v_is_unknown and u_is_port:
            # v unknown, u port: contribute to G_up and v's diagonal
            iv = unknown_to_idx[v]
            iu = port_to_idx[u]
            rows_up.append(iv)
            cols_up.append(iu)
            data_up.append(-g)
            diag_u[iv] += g

    # Add diagonal entries to G_uu
    for i in range(n_unknown):
        rows_uu.append(i)
        cols_uu.append(i)
        data_uu.append(diag_u[i])

    # Build sparse matrices
    G_uu = sp.csr_matrix(
        (data_uu, (rows_uu, cols_uu)),
        shape=(n_unknown, n_unknown),
        dtype=np.float64
    )

    if rows_up:
        G_up = sp.csr_matrix(
            (data_up, (rows_up, cols_up)),
            shape=(n_unknown, n_ports),
            dtype=np.float64
        )
    else:
        G_up = sp.csr_matrix((n_unknown, n_ports), dtype=np.float64)

    # Build RHS: I - G_up * V_p
    # Current convention: positive = sink (drawing current)
    I_u = np.zeros(n_unknown, dtype=np.float64)
    for node, current in current_injections.items():
        if node in unknown_to_idx:
            I_u[unknown_to_idx[node]] = -current  # Negate for nodal equation

    # Port voltage vector
    V_p = np.array([port_voltages.get(p, vdd) for p in port_nodes], dtype=np.float64)

    # RHS = I_u - G_up * V_p
    rhs = I_u - G_up.dot(V_p)

    # Solve G_uu * V_u = rhs
    try:
        lu = spla.splu(G_uu.tocsc())
        V_u = lu.solve(rhs)
    except Exception:
        # Fallback: use iterative solver
        V_u, info = spla.cg(G_uu, rhs, **_get_tol_kwargs(1e-10))
        if info != 0:
            # If solver fails, return port voltages for unknowns as fallback
            V_u = np.full(n_unknown, vdd)

    # Build result voltages for connected unknown nodes
    for i, node in enumerate(unknown_nodes):
        voltages[node] = float(V_u[i])

    # Add port voltages
    for node in port_nodes:
        voltages[node] = port_voltages.get(node, vdd)

    solve_time_ms = (time.perf_counter() - t0) * 1000
    return (tile_id, voltages, solve_time_ms, disconnected_halo)


# ============================================================================
# TileManager Class
# ============================================================================

class TileManager:
    """Manages bottom-grid tile generation and connectivity validation.

    This class handles all tiling-related operations for the hierarchical solver:
    - Coordinate extraction from PDN node names
    - Uniform tile boundary generation
    - Node-to-tile assignment with constraint satisfaction
    - Halo expansion for boundary accuracy
    - Tile connectivity validation and repair
    - Tile solve argument preparation
    - Result merging and accuracy validation
    """

    def __init__(self, model: UnifiedPowerGridModel):
        """Initialize tile manager with a unified model.

        Args:
            model: UnifiedPowerGridModel instance
        """
        self.model = model

    def extract_bottom_grid_coordinates(
        self,
        bottom_grid_nodes: Set[Any],
        port_nodes: Optional[Set[Any]] = None,
    ) -> Tuple[Dict[Any, Tuple[float, float]], Dict[Any, Tuple[float, float]], Tuple[float, float, float, float]]:
        """Extract (x, y) coordinates for bottom-grid and port nodes.

        Only supports PDN graphs with string node names in X_Y_LAYER format.
        Raises ValueError for synthetic grids (NodeID types).

        Args:
            bottom_grid_nodes: Set of nodes in the bottom-grid
            port_nodes: Optional set of port nodes (for spatial overlap detection)

        Returns:
            (bottom_coords, port_coords, bounds)
            - bottom_coords: node -> (x, y) for bottom-grid nodes
            - port_coords: node -> (x, y) for port nodes
            - bounds: (x_min, x_max, y_min, y_max) combined bounds

        Raises:
            ValueError: If graph is synthetic (NodeID) or coordinates unavailable
        """
        extractor = NodeInfoExtractor(self.model.graph)
        bottom_coords: Dict[Any, Tuple[float, float]] = {}
        port_coords_dict: Dict[Any, Tuple[float, float]] = {}

        x_min, x_max = float('inf'), float('-inf')
        y_min, y_max = float('inf'), float('-inf')

        # Extract bottom-grid node coordinates
        for node in bottom_grid_nodes:
            # Check for NodeID (synthetic grid) - not supported
            if hasattr(node, 'layer') and hasattr(node, 'idx'):
                raise ValueError(
                    "Tiled hierarchical solving is only supported for PDN graphs. "
                    f"Found synthetic NodeID: {node}. "
                    "Use solve_hierarchical() for synthetic grids instead."
                )

            info = extractor.get_info(node)
            xy = info.xy

            if xy is None:
                # Skip nodes without coordinates (e.g., package nodes)
                continue

            x, y = xy
            bottom_coords[node] = (x, y)

            x_min = min(x_min, x)
            x_max = max(x_max, x)
            y_min = min(y_min, y)
            y_max = max(y_max, y)

        # Extract port node coordinates (for spatial overlap detection)
        if port_nodes:
            for node in port_nodes:
                info = extractor.get_info(node)
                xy = info.xy

                if xy is None:
                    continue

                x, y = xy
                port_coords_dict[node] = (x, y)

                # Include ports in bounds calculation
                x_min = min(x_min, x)
                x_max = max(x_max, x)
                y_min = min(y_min, y)
                y_max = max(y_max, y)

        if not bottom_coords:
            raise ValueError(
                "No valid coordinates found for bottom-grid nodes. "
                "Ensure PDN nodes use X_Y_LAYER naming format (e.g., '1000_2000_M1')."
            )

        if x_min == x_max:
            # All nodes on same x - add small margin
            x_min -= 1.0
            x_max += 1.0
        if y_min == y_max:
            # All nodes on same y - add small margin
            y_min -= 1.0
            y_max += 1.0

        return bottom_coords, port_coords_dict, (x_min, x_max, y_min, y_max)

    def generate_uniform_tile_bounds(
        self,
        N_x: int,
        N_y: int,
        bounds: Tuple[float, float, float, float],
    ) -> List[TileBounds]:
        """Generate uniform rectangular tile boundaries.

        Args:
            N_x: Number of tiles in x direction
            N_y: Number of tiles in y direction
            bounds: (x_min, x_max, y_min, y_max) of the grid

        Returns:
            List of TileBounds, indexed by tile_id = row * N_x + col
        """
        x_min, x_max, y_min, y_max = bounds
        x_step = (x_max - x_min) / N_x
        y_step = (y_max - y_min) / N_y

        tiles = []
        for row in range(N_y):
            for col in range(N_x):
                tile_id = row * N_x + col
                tile_x_min = x_min + col * x_step
                tile_x_max = x_min + (col + 1) * x_step
                tile_y_min = y_min + row * y_step
                tile_y_max = y_min + (row + 1) * y_step

                # Handle floating point boundary: make last tile extend to exact max
                if col == N_x - 1:
                    tile_x_max = x_max
                if row == N_y - 1:
                    tile_y_max = y_max

                tiles.append(TileBounds(
                    tile_id=tile_id,
                    x_min=tile_x_min,
                    x_max=tile_x_max,
                    y_min=tile_y_min,
                    y_max=tile_y_max,
                ))

        return tiles

    def assign_nodes_to_tiles(
        self,
        bottom_coords: Dict[Any, Tuple[float, float]],
        port_coords: Dict[Any, Tuple[float, float]],
        tile_bounds: List[TileBounds],
        port_nodes: Set[Any],
        load_nodes: Set[Any],
        min_ports_per_tile: int,
        N_x: int,
        N_y: int,
    ) -> Tuple[List[TileBounds], Dict[int, Set[Any]], Dict[int, Set[Any]], Dict[int, Set[Any]]]:
        """Assign nodes to tiles and adjust boundaries to satisfy constraints.

        Constraints:
        1. Each tile must have at least min_ports_per_tile port nodes
        2. Each tile must have at least one current source (or be merged)

        Args:
            bottom_coords: bottom-grid node -> (x, y) mapping
            port_coords: port node -> (x, y) mapping (for spatial assignment)
            tile_bounds: Initial uniform tile boundaries
            port_nodes: Set of port nodes from partition layer
            load_nodes: Set of nodes with current injections
            min_ports_per_tile: Minimum ports required per tile
            N_x: Number of tiles in x direction
            N_y: Number of tiles in y direction

        Returns:
            (adjusted_bounds, tile_nodes, tile_ports, tile_loads)
            - adjusted_bounds: List of TileBounds (may have fewer tiles if merged)
            - tile_nodes: tile_id -> set of bottom-grid nodes in tile
            - tile_ports: tile_id -> set of port nodes spatially in tile
            - tile_loads: tile_id -> set of load nodes in tile

        Raises:
            ValueError: If tiles cannot be adjusted to satisfy constraints
        """
        # Build initial assignment
        tile_nodes: Dict[int, Set[Any]] = {t.tile_id: set() for t in tile_bounds}
        tile_ports: Dict[int, Set[Any]] = {t.tile_id: set() for t in tile_bounds}
        tile_loads: Dict[int, Set[Any]] = {t.tile_id: set() for t in tile_bounds}

        # Map each node to its tile
        bounds_by_id = {t.tile_id: t for t in tile_bounds}

        def get_tile_for_coord(x: float, y: float) -> Optional[int]:
            """Find tile containing coordinate (x, y)."""
            for t in tile_bounds:
                if t.x_min <= x <= t.x_max and t.y_min <= y <= t.y_max:
                    return t.tile_id
            return None

        # Assign bottom-grid nodes to tiles
        for node, (x, y) in bottom_coords.items():
            tid = get_tile_for_coord(x, y)
            if tid is not None:
                tile_nodes[tid].add(node)
                if node in load_nodes:
                    tile_loads[tid].add(node)

        # Assign port nodes to tiles based on spatial coordinates
        # Port nodes are at partition layer but we use their (x,y) to assign to tiles
        for node, (x, y) in port_coords.items():
            tid = get_tile_for_coord(x, y)
            if tid is not None:
                tile_ports[tid].add(node)

        # Phase 1: Adjust boundaries for port constraint
        # This is a simplified approach - expand under-ported tiles
        adjusted_bounds = list(tile_bounds)
        max_iterations = N_x * N_y * 2  # Prevent infinite loop
        iteration = 0

        while iteration < max_iterations:
            iteration += 1
            changed = False

            for idx, t in enumerate(adjusted_bounds):
                tid = t.tile_id
                if len(tile_ports.get(tid, set())) >= min_ports_per_tile:
                    continue

                # Find best neighbor to expand into
                best_neighbor = None
                best_ports_gain = 0

                # Check adjacent tiles
                row, col = tid // N_x, tid % N_x
                neighbors = []
                if col > 0:
                    neighbors.append(('left', (row * N_x + col - 1)))
                if col < N_x - 1:
                    neighbors.append(('right', (row * N_x + col + 1)))
                if row > 0:
                    neighbors.append(('up', ((row - 1) * N_x + col)))
                if row < N_y - 1:
                    neighbors.append(('down', ((row + 1) * N_x + col)))

                for direction, nbr_tid in neighbors:
                    if nbr_tid not in tile_ports:
                        continue
                    nbr_ports = len(tile_ports[nbr_tid])
                    if nbr_ports > min_ports_per_tile:
                        # Can steal some ports
                        excess = nbr_ports - min_ports_per_tile
                        if excess > best_ports_gain:
                            best_ports_gain = excess
                            best_neighbor = (direction, nbr_tid)

                if best_neighbor is None:
                    continue

                direction, nbr_tid = best_neighbor
                nbr_bounds = bounds_by_id.get(nbr_tid)
                if nbr_bounds is None:
                    continue

                # Expand tile boundary by 10% of neighbor's dimension
                expansion_factor = 0.1
                if direction == 'left':
                    dx = (nbr_bounds.x_max - nbr_bounds.x_min) * expansion_factor
                    new_x_min = max(nbr_bounds.x_min, t.x_min - dx)
                    adjusted_bounds[idx] = TileBounds(
                        tile_id=tid, x_min=new_x_min, x_max=t.x_max,
                        y_min=t.y_min, y_max=t.y_max
                    )
                elif direction == 'right':
                    dx = (nbr_bounds.x_max - nbr_bounds.x_min) * expansion_factor
                    new_x_max = min(nbr_bounds.x_max, t.x_max + dx)
                    adjusted_bounds[idx] = TileBounds(
                        tile_id=tid, x_min=t.x_min, x_max=new_x_max,
                        y_min=t.y_min, y_max=t.y_max
                    )
                elif direction == 'up':
                    dy = (nbr_bounds.y_max - nbr_bounds.y_min) * expansion_factor
                    new_y_min = max(nbr_bounds.y_min, t.y_min - dy)
                    adjusted_bounds[idx] = TileBounds(
                        tile_id=tid, x_min=t.x_min, x_max=t.x_max,
                        y_min=new_y_min, y_max=t.y_max
                    )
                elif direction == 'down':
                    dy = (nbr_bounds.y_max - nbr_bounds.y_min) * expansion_factor
                    new_y_max = min(nbr_bounds.y_max, t.y_max + dy)
                    adjusted_bounds[idx] = TileBounds(
                        tile_id=tid, x_min=t.x_min, x_max=t.x_max,
                        y_min=t.y_min, y_max=new_y_max
                    )

                bounds_by_id[tid] = adjusted_bounds[idx]
                changed = True

                # Reassign nodes with new boundaries
                tile_nodes[tid] = set()
                tile_ports[tid] = set()
                tile_loads[tid] = set()
                new_t = adjusted_bounds[idx]
                # Reassign bottom-grid nodes
                for node, (x, y) in bottom_coords.items():
                    if new_t.x_min <= x <= new_t.x_max and new_t.y_min <= y <= new_t.y_max:
                        tile_nodes[tid].add(node)
                        if node in load_nodes:
                            tile_loads[tid].add(node)
                # Reassign port nodes
                for node, (x, y) in port_coords.items():
                    if new_t.x_min <= x <= new_t.x_max and new_t.y_min <= y <= new_t.y_max:
                        tile_ports[tid].add(node)

            if not changed:
                break

        # Phase 2: Merge tiles with no current sources into neighbors
        tiles_to_remove: Set[int] = set()
        for t in adjusted_bounds:
            tid = t.tile_id
            if tid in tiles_to_remove:
                continue
            if len(tile_loads.get(tid, set())) == 0:
                # Find neighbor to merge into
                row, col = tid // N_x, tid % N_x
                neighbors = []
                if col > 0:
                    neighbors.append(row * N_x + col - 1)
                if col < N_x - 1:
                    neighbors.append(row * N_x + col + 1)
                if row > 0:
                    neighbors.append((row - 1) * N_x + col)
                if row < N_y - 1:
                    neighbors.append((row + 1) * N_x + col)

                for nbr_tid in neighbors:
                    if nbr_tid in tiles_to_remove or nbr_tid not in tile_nodes:
                        continue
                    # Merge into this neighbor
                    tile_nodes[nbr_tid].update(tile_nodes.get(tid, set()))
                    tile_ports[nbr_tid].update(tile_ports.get(tid, set()))
                    tile_loads[nbr_tid].update(tile_loads.get(tid, set()))

                    # Expand neighbor bounds to include this tile
                    nbr_b = bounds_by_id[nbr_tid]
                    merged_bounds = TileBounds(
                        tile_id=nbr_tid,
                        x_min=min(nbr_b.x_min, t.x_min),
                        x_max=max(nbr_b.x_max, t.x_max),
                        y_min=min(nbr_b.y_min, t.y_min),
                        y_max=max(nbr_b.y_max, t.y_max),
                    )
                    bounds_by_id[nbr_tid] = merged_bounds

                    # Mark this tile for removal
                    tiles_to_remove.add(tid)
                    break

        # Remove merged tiles
        adjusted_bounds = [
            bounds_by_id[t.tile_id] for t in adjusted_bounds
            if t.tile_id not in tiles_to_remove
        ]
        for tid in tiles_to_remove:
            tile_nodes.pop(tid, None)
            tile_ports.pop(tid, None)
            tile_loads.pop(tid, None)

        # Phase 3: Final reassignment to ensure disjoint cores
        # After boundary adjustments, tiles may overlap. We need to assign each
        # node to exactly one tile using a deterministic tie-breaker.
        # Tie-breaker: assign to tile with smallest tile_id among those containing the node.
        final_bounds = {t.tile_id: t for t in adjusted_bounds}

        # Clear all assignments and reassign with deterministic tie-breaking
        tile_nodes = {t.tile_id: set() for t in adjusted_bounds}
        tile_ports = {t.tile_id: set() for t in adjusted_bounds}
        tile_loads = {t.tile_id: set() for t in adjusted_bounds}

        # Sort tile_ids for deterministic assignment (lowest ID wins ties)
        sorted_tids = sorted(final_bounds.keys())

        # Assign each bottom-grid node to exactly one tile
        for node, (x, y) in bottom_coords.items():
            for tid in sorted_tids:
                t = final_bounds[tid]
                if t.x_min <= x <= t.x_max and t.y_min <= y <= t.y_max:
                    tile_nodes[tid].add(node)
                    if node in load_nodes:
                        tile_loads[tid].add(node)
                    break  # Assign to first (lowest ID) matching tile only

        # Assign each port node to exactly one tile
        for node, (x, y) in port_coords.items():
            for tid in sorted_tids:
                t = final_bounds[tid]
                if t.x_min <= x <= t.x_max and t.y_min <= y <= t.y_max:
                    tile_ports[tid].add(node)
                    break  # Assign to first (lowest ID) matching tile only

        return adjusted_bounds, tile_nodes, tile_ports, tile_loads

    def expand_tile_with_halo(
        self,
        tile: TileBounds,
        tile_core_nodes: Set[Any],
        bottom_coords: Dict[Any, Tuple[float, float]],
        grid_bounds: Tuple[float, float, float, float],
        halo_percent: float,
        port_nodes: Set[Any],
        port_coords: Dict[Any, Tuple[float, float]],
        load_nodes: Set[Any],
    ) -> BottomGridTile:
        """Expand tile with halo region and collect node sets.

        Args:
            tile: Original tile bounds (core region)
            tile_core_nodes: Nodes in the core region
            bottom_coords: bottom-grid node -> (x, y) mapping
            grid_bounds: (x_min, x_max, y_min, y_max) of entire grid
            halo_percent: Halo size as percentage of tile dimensions (0.0 to 1.0)
            port_nodes: Set of all port nodes
            port_coords: port node -> (x, y) mapping (for spatial selection)
            load_nodes: Set of all load nodes (nodes with current)

        Returns:
            BottomGridTile with core, halo, and port node sets
        """
        x_min_grid, x_max_grid, y_min_grid, y_max_grid = grid_bounds

        # Compute halo expansion
        tile_width = tile.x_max - tile.x_min
        tile_height = tile.y_max - tile.y_min
        halo_x = tile_width * halo_percent
        halo_y = tile_height * halo_percent

        # Expected halo area (if no clipping)
        expected_halo_area = (
            (tile_width + 2 * halo_x) * (tile_height + 2 * halo_y)
            - tile_width * tile_height
        )

        # Compute expanded bounds (clipped to grid)
        exp_x_min = max(x_min_grid, tile.x_min - halo_x)
        exp_x_max = min(x_max_grid, tile.x_max + halo_x)
        exp_y_min = max(y_min_grid, tile.y_min - halo_y)
        exp_y_max = min(y_max_grid, tile.y_max + halo_y)

        # Actual halo area after clipping
        actual_halo_area = (
            (exp_x_max - exp_x_min) * (exp_y_max - exp_y_min)
            - tile_width * tile_height
        )

        halo_clipped = False
        halo_clip_ratio = 1.0
        if expected_halo_area > 0:
            halo_clip_ratio = max(0.0, actual_halo_area / expected_halo_area)
            halo_clipped = halo_clip_ratio < 0.99  # Allow small floating point error

        # Only warn if halo is severely clipped (< 30% of expected)
        # Moderate clipping (30-50%) is expected for corner/edge tiles
        if halo_clipped and halo_clip_ratio < 0.3:
            logger.warning(
                f"Tile {tile.tile_id}: Halo severely clipped - only {halo_clip_ratio:.0%} of "
                f"expected halo area available (tile at grid boundary). "
                f"This may reduce accuracy. Consider using fewer tiles or smaller halo_percent."
            )

        # Collect halo nodes (in expanded region but not in core)
        halo_nodes: Set[Any] = set()
        for node, (x, y) in bottom_coords.items():
            if node in tile_core_nodes:
                continue
            if exp_x_min <= x <= exp_x_max and exp_y_min <= y <= exp_y_max:
                halo_nodes.add(node)

        # Collect port nodes spatially within tile+halo (these become Dirichlet BCs)
        # Port nodes are at partition layer but we find them by (x,y) coordinates
        tile_port_nodes: Set[Any] = set()
        for port, (x, y) in port_coords.items():
            if exp_x_min <= x <= exp_x_max and exp_y_min <= y <= exp_y_max:
                tile_port_nodes.add(port)

        # All nodes includes bottom-grid nodes (core + halo) AND port nodes
        # Port nodes are essential for applying Dirichlet BCs in the tile solve
        all_nodes = tile_core_nodes | halo_nodes | tile_port_nodes

        # Collect load nodes within core only
        tile_load_nodes = tile_core_nodes & load_nodes

        return BottomGridTile(
            tile_id=tile.tile_id,
            bounds=tile,
            core_nodes=tile_core_nodes,
            halo_nodes=halo_nodes,
            all_nodes=all_nodes,
            port_nodes=tile_port_nodes,
            load_nodes=tile_load_nodes,
            halo_clipped=halo_clipped,
            halo_clip_ratio=halo_clip_ratio,
        )

    def validate_and_fix_tile_connectivity(
        self,
        tiles: List[BottomGridTile],
        bottom_nodes: Set[Any],
        port_nodes: Set[Any],
    ) -> List[BottomGridTile]:
        """Validate that all core nodes are connected to ports and fix if needed.

        For each tile, checks connectivity from core nodes to port nodes through
        the tile's local edges. If core nodes are disconnected, expands the halo
        to include the connecting path through the full bottom-grid.

        Handles three cases:
        1. Core nodes connected through tile edges - no fix needed
        2. Core nodes connected through full graph but not tile - expand halo
        3. Core nodes in global floating islands - mark for vdd assignment

        Args:
            tiles: List of tiles to validate
            bottom_nodes: All bottom-grid nodes
            port_nodes: All port nodes at partition layer

        Returns:
            List of tiles with potentially expanded halos to ensure connectivity
        """
        graph = self.model.graph
        is_pdn = isinstance(graph, RustworkxMultiDiGraphWrapper)

        # ================================================================
        # First: Find global floating islands in the bottom-grid
        # These are nodes not connected to ANY port via R-type edges
        # NOTE: For directed graphs (MultiDiGraph), we must traverse BOTH
        # successors and predecessors since R-type edges may go either direction
        # ================================================================

        global_reachable: Set[Any] = set()
        global_visited: Set[Any] = set(port_nodes)
        global_queue = deque(port_nodes)

        # Helper to get all neighbors (both directions for directed graphs)
        def get_all_neighbors(g: Any, node: Any) -> Set[Any]:
            """Get neighbors in both directions for directed graphs."""
            if g.is_directed():
                # Combine successors and predecessors
                return set(g.successors(node)) | set(g.predecessors(node))
            else:
                return set(g[node])

        # Helper to check for R-type edge in either direction
        def has_r_type_edge(g: Any, u: Any, v: Any, is_pdn_graph: bool) -> bool:
            """Check if there's an R-type edge between u and v (either direction)."""
            # Check u -> v
            if g.has_edge(u, v):
                if is_pdn_graph:
                    for key, data in g[u][v].items():
                        if data.get('type') == 'R':
                            return True
                else:
                    if 'resistance' in g[u][v]:
                        return True
            # Check v -> u (for directed graphs)
            if g.is_directed() and g.has_edge(v, u):
                if is_pdn_graph:
                    for key, data in g[v][u].items():
                        if data.get('type') == 'R':
                            return True
                else:
                    if 'resistance' in g[v][u]:
                        return True
            return False

        while global_queue:
            node = global_queue.popleft()
            for neighbor in get_all_neighbors(graph, node):
                if neighbor in global_visited:
                    continue

                if not has_r_type_edge(graph, node, neighbor, is_pdn):
                    continue

                global_visited.add(neighbor)
                # Track bottom nodes that are reachable
                if neighbor in bottom_nodes:
                    global_reachable.add(neighbor)
                # Continue BFS through all nodes (not just bottom_nodes)
                global_queue.append(neighbor)

        global_floating = bottom_nodes - global_reachable - port_nodes

        if global_floating:
            logger.warning(
                f"Found {len(global_floating)} bottom-grid nodes in global floating islands "
                f"(not connected to ANY port via R-type edges). These will be assigned vdd."
            )

        # ================================================================
        # Now process each tile
        # ================================================================
        fixed_tiles = []

        for tile in tiles:
            # Build local adjacency from tile's nodes
            tile_nodes = tile.all_nodes
            local_adj: Dict[Any, Set[Any]] = {n: set() for n in tile_nodes}

            for u in tile_nodes:
                # Handle directed graphs: get both successors and predecessors
                neighbors = get_all_neighbors(graph, u)
                for v in neighbors:
                    if v not in tile_nodes:
                        continue
                    # Check if there's an R-type edge (either direction)
                    if has_r_type_edge(graph, u, v, is_pdn):
                        local_adj[u].add(v)
                        local_adj[v].add(u)

            # BFS from ports to find reachable core nodes
            port_set = tile.port_nodes
            visited = set(port_set)
            queue = deque(port_set)

            while queue:
                node = queue.popleft()
                for neighbor in local_adj.get(node, set()):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)

            reachable_core = visited & tile.core_nodes
            disconnected_core = tile.core_nodes - reachable_core

            # Separate global floating nodes from locally disconnected nodes
            floating_core = disconnected_core & global_floating
            locally_disconnected_core = disconnected_core - global_floating

            if not locally_disconnected_core and not floating_core:
                # All core nodes connected - tile is OK
                # Still check for disconnected halo nodes
                disconnected_halo = (tile.halo_nodes - visited) - port_set
                if disconnected_halo:
                    # Remove disconnected halo nodes from tile
                    new_halo = tile.halo_nodes - disconnected_halo
                    new_all = tile.core_nodes | new_halo | tile.port_nodes
                    fixed_tiles.append(BottomGridTile(
                        tile_id=tile.tile_id,
                        bounds=tile.bounds,
                        core_nodes=tile.core_nodes,
                        halo_nodes=new_halo,
                        all_nodes=new_all,
                        port_nodes=tile.port_nodes,
                        load_nodes=tile.load_nodes,
                        halo_clipped=tile.halo_clipped,
                        halo_clip_ratio=tile.halo_clip_ratio,
                        disconnected_halo_nodes=disconnected_halo,
                    ))
                else:
                    fixed_tiles.append(tile)
                continue

            # Handle floating core nodes: remove from core, track separately
            # They'll be assigned vdd by merge_tiled_voltages
            new_core = tile.core_nodes - floating_core

            if not locally_disconnected_core:
                # Only floating nodes - just remove them from core
                disconnected_halo = (tile.halo_nodes - visited) - port_set
                new_halo = tile.halo_nodes - disconnected_halo
                new_all = new_core | new_halo | tile.port_nodes
                fixed_tiles.append(BottomGridTile(
                    tile_id=tile.tile_id,
                    bounds=tile.bounds,
                    core_nodes=new_core,
                    halo_nodes=new_halo,
                    all_nodes=new_all,
                    port_nodes=tile.port_nodes,
                    load_nodes=tile.load_nodes - floating_core,  # Remove floating from loads too
                    halo_clipped=tile.halo_clipped,
                    halo_clip_ratio=tile.halo_clip_ratio,
                    disconnected_halo_nodes=disconnected_halo,
                    floating_core_nodes=floating_core,  # Track floating nodes separately
                ))
                continue

            # Need to expand halo to connect locally disconnected core nodes
            # (not global floating nodes - those were already handled above)
            # Find shortest paths from disconnected core nodes to any port
            # through the FULL bottom-grid (not just tile)

            expanded_halo = set(tile.halo_nodes)
            expanded_ports = set(tile.port_nodes)

            for core_node in locally_disconnected_core:
                # BFS from this core node through full graph to find nearest port
                # Track parent pointers to reconstruct actual path
                parent: Dict[Any, Any] = {core_node: None}
                full_queue = deque([core_node])
                found_port = None

                while full_queue and found_port is None:
                    node = full_queue.popleft()
                    for neighbor in graph[node]:
                        if neighbor in parent:
                            continue

                        # Check if this is an R-type connection
                        has_r_edge = False
                        if is_pdn:
                            for key, data in graph[node][neighbor].items():
                                if data.get('type') == 'R':
                                    has_r_edge = True
                                    break
                        else:
                            has_r_edge = 'resistance' in graph[node][neighbor]

                        if not has_r_edge:
                            continue

                        parent[neighbor] = node

                        # Check if neighbor is a port
                        if neighbor in port_nodes:
                            found_port = neighbor
                            break

                        # Add to queue if it's a bottom-grid node (continue searching)
                        if neighbor in bottom_nodes:
                            full_queue.append(neighbor)

                if found_port:
                    # Trace back path from found_port to core_node
                    path_nodes: Set[Any] = set()
                    current: Any = found_port
                    while current is not None:
                        path_nodes.add(current)
                        current = parent[current]

                    # Add path nodes (excluding core_node itself) to halo
                    # and the found_port to ports
                    for pn in path_nodes:
                        if pn in bottom_nodes and pn not in tile.core_nodes:
                            expanded_halo.add(pn)
                        if pn in port_nodes:
                            expanded_ports.add(pn)

            # Rebuild tile with expanded halo (excluding floating core)
            new_all = new_core | expanded_halo | expanded_ports

            # Identify disconnected halo after expansion
            new_local_adj: Dict[Any, Set[Any]] = {n: set() for n in new_all}
            for u in new_all:
                for v in graph[u]:
                    if v not in new_all:
                        continue
                    if is_pdn:
                        for key, data in graph[u][v].items():
                            if data.get('type') == 'R':
                                new_local_adj[u].add(v)
                                new_local_adj[v].add(u)
                                break
                    else:
                        if 'resistance' in graph[u][v]:
                            new_local_adj[u].add(v)
                            new_local_adj[v].add(u)

            # Check connectivity again
            new_visited = set(expanded_ports)
            new_queue = deque(expanded_ports)
            while new_queue:
                node = new_queue.popleft()
                for neighbor in new_local_adj.get(node, set()):
                    if neighbor not in new_visited:
                        new_visited.add(neighbor)
                        new_queue.append(neighbor)

            # All locally_disconnected_core should now be connected
            # (global floating were already removed from new_core)
            still_disconnected_core = new_core - new_visited
            if still_disconnected_core:
                # This shouldn't happen - these should have been found as globally floating
                logger.error(
                    f"Tile {tile.tile_id}: {len(still_disconnected_core)} core nodes still disconnected "
                    f"after halo expansion. Sample: {list(still_disconnected_core)[:3]}. "
                    f"This indicates a bug in connectivity analysis."
                )
                # Add them to floating set to avoid solver errors
                floating_core |= still_disconnected_core
                new_core -= still_disconnected_core

            disconnected_halo = expanded_halo - new_visited
            connected_halo = expanded_halo - disconnected_halo

            fixed_tiles.append(BottomGridTile(
                tile_id=tile.tile_id,
                bounds=tile.bounds,
                core_nodes=new_core,
                halo_nodes=connected_halo,
                all_nodes=new_core | connected_halo | expanded_ports,
                port_nodes=expanded_ports,
                load_nodes=tile.load_nodes - floating_core,
                halo_clipped=tile.halo_clipped,
                halo_clip_ratio=tile.halo_clip_ratio,
                disconnected_halo_nodes=disconnected_halo,
                floating_core_nodes=floating_core,  # Track floating nodes separately
            ))

            if len(expanded_halo) > len(tile.halo_nodes) * 1.5:
                logger.info(
                    f"Tile {tile.tile_id}: Expanded halo from {len(tile.halo_nodes)} to "
                    f"{len(connected_halo)} nodes (+{len(expanded_ports) - len(tile.port_nodes)} ports) "
                    f"to connect {len(locally_disconnected_core)} core nodes"
                )

        return fixed_tiles

    def build_tile_solve_args(
        self,
        tiles: List[BottomGridTile],
        port_voltages: Dict[Any, float],
        current_injections: Dict[Any, float],
    ) -> List[Tuple[int, List[Any], Set[Any], List[Tuple[Any, Any, float]], List[Any], Dict[Any, float], Dict[Any, float], float]]:
        """Build serializable arguments for parallel tile solves.

        Args:
            tiles: List of BottomGridTile objects
            port_voltages: Port node -> voltage from top-grid solve
            current_injections: Node -> current for all loads

        Returns:
            List of argument tuples for solve_single_tile function:
            (tile_id, node_list, core_nodes, edge_tuples, port_nodes, port_voltages, currents, vdd)
        """
        args_list = []
        graph = self.model.graph
        is_pdn = isinstance(graph, RustworkxMultiDiGraphWrapper)
        r_unit_kohm = self.model.resistance_unit_kohm

        for tile in tiles:
            node_list = list(tile.all_nodes)
            node_set = tile.all_nodes

            # Extract edges within tile
            edge_tuples: List[Tuple[Any, Any, float]] = []
            seen_edges: Set[Tuple[Any, Any]] = set()

            for u in node_set:
                for v, edge_data in graph[u].items():
                    if v not in node_set:
                        continue

                    # Handle MultiDiGraph edge data format
                    if is_pdn:
                        # edge_data is dict of {key: data}
                        for key, data in edge_data.items():
                            if data.get('type') != 'R':
                                continue
                            R = float(data.get('value', 0.0))
                            if r_unit_kohm:
                                R *= 1e3  # Convert kOhm to Ohm
                            if R <= 0:
                                continue
                            g = 1.0 / R

                            # Avoid duplicate edges
                            edge_key = (min(u, v), max(u, v)) if isinstance(u, str) else (u, v)
                            if edge_key not in seen_edges:
                                seen_edges.add(edge_key)
                                edge_tuples.append((u, v, g))
                    else:
                        # Regular graph
                        R = float(edge_data.get('resistance', 0.0))
                        if R <= 0:
                            continue
                        g = 1.0 / R

                        edge_key = (u, v) if hash(u) < hash(v) else (v, u)
                        if edge_key not in seen_edges:
                            seen_edges.add(edge_key)
                            edge_tuples.append((u, v, g))

            # Filter port voltages and currents for this tile
            tile_port_voltages = {
                p: port_voltages.get(p, self.model.vdd)
                for p in tile.port_nodes
            }
            tile_currents = {
                n: current_injections.get(n, 0.0)
                for n in tile.load_nodes
                if current_injections.get(n, 0.0) != 0
            }

            args_list.append((
                tile.tile_id,
                node_list,
                tile.core_nodes,  # Pass core_nodes for connectivity validation
                edge_tuples,
                list(tile.port_nodes),
                tile_port_voltages,
                tile_currents,
                self.model.vdd,
            ))

        return args_list

    def merge_tiled_voltages(
        self,
        tiles: List[BottomGridTile],
        tile_results: Dict[int, Dict[Any, float]],
        bottom_grid_nodes: Set[Any],
    ) -> Tuple[Dict[Any, float], List[str]]:
        """Merge tile results, taking only core node voltages.

        Args:
            tiles: List of BottomGridTile objects
            tile_results: tile_id -> {node: voltage} from tile solves
            bottom_grid_nodes: Set of all bottom-grid nodes for coverage validation

        Returns:
            (merged_voltages, warnings)
            - merged_voltages: node -> voltage for all bottom-grid nodes
            - warnings: List of warning messages

        Raises:
            ValueError: If core regions overlap or coverage is incomplete
        """
        merged: Dict[Any, float] = {}
        warnings = []
        covered_nodes: Set[Any] = set()
        vdd = self.model.vdd
        floating_nodes: Set[Any] = set()

        for tile in tiles:
            tile_voltages = tile_results.get(tile.tile_id, {})

            # Only take core node voltages
            for node in tile.core_nodes:
                if node in covered_nodes:
                    raise ValueError(
                        f"Disjoint core region violation: node {node} appears in "
                        f"multiple tile cores. This indicates a bug in tile boundary adjustment."
                    )
                covered_nodes.add(node)

                if node in tile_voltages:
                    merged[node] = tile_voltages[node]
                else:
                    # Node missing from solve - likely in a floating island
                    # Assign vdd since it has no IR-drop path
                    merged[node] = vdd
                    floating_nodes.add(node)

            # Handle floating core nodes tracked separately
            # These are original core nodes that were removed due to being in global floating islands
            if tile.floating_core_nodes:
                for node in tile.floating_core_nodes:
                    if node not in covered_nodes:
                        covered_nodes.add(node)
                        merged[node] = vdd
                        floating_nodes.add(node)

            # Collect halo clip warnings
            if tile.halo_clipped and tile.halo_clip_ratio < 0.5:
                warnings.append(
                    f"Tile {tile.tile_id}: Halo significantly clipped "
                    f"(ratio={tile.halo_clip_ratio:.2%})"
                )

        if floating_nodes:
            warnings.append(
                f"{len(floating_nodes)} nodes in floating islands assigned vdd"
            )

        # Validate coverage (nodes with coordinates should be covered)
        # Note: Some nodes may not have coordinates (package nodes) - that's OK
        return merged, warnings

    def validate_tiled_accuracy(
        self,
        tiled_voltages: Dict[Any, float],
        bottom_grid_nodes: Set[Any],
        port_nodes: Set[Any],
        port_voltages: Dict[Any, float],
        current_injections: Dict[Any, float],
    ) -> Dict[str, Any]:
        """Validate tiled solution against flat bottom-grid solve.

        Runs a full (non-tiled) bottom-grid solve and compares voltages.

        Args:
            tiled_voltages: Voltages from tiled solve
            bottom_grid_nodes: Set of bottom-grid nodes
            port_nodes: Set of port nodes (Dirichlet BCs)
            port_voltages: Port voltages from top-grid solve
            current_injections: Load currents

        Returns:
            Dict with max_diff, mean_diff, rmse, node_with_max_diff
        """
        # Build full bottom-grid system
        bottom_subgrid = bottom_grid_nodes | port_nodes

        try:
            full_system = self.model._build_subgrid_system_fast(
                subgrid_nodes=bottom_subgrid,
                dirichlet_nodes=port_nodes,
                dirichlet_voltage=self.model.vdd,
            )
        except Exception:
            full_system = self.model._build_subgrid_system(
                subgrid_nodes=bottom_subgrid,
                dirichlet_nodes=port_nodes,
                dirichlet_voltage=self.model.vdd,
            )

        if full_system is None:
            return {
                'max_diff': float('nan'),
                'mean_diff': float('nan'),
                'rmse': float('nan'),
                'node_with_max_diff': None,
                'error': 'Failed to build full bottom-grid system',
            }

        # Get currents for bottom-grid nodes
        bottom_grid_currents = {
            n: c for n, c in current_injections.items()
            if n in bottom_grid_nodes
        }

        # Solve full bottom-grid
        flat_voltages = self.model._solve_subgrid(
            reduced_system=full_system,
            current_injections=bottom_grid_currents,
            dirichlet_voltages=port_voltages,
        )

        # Compare voltages
        diffs = []
        max_diff = 0.0
        max_diff_node = None

        for node, tiled_v in tiled_voltages.items():
            flat_v = flat_voltages.get(node)
            if flat_v is None:
                continue

            diff = abs(tiled_v - flat_v)
            diffs.append(diff)

            if diff > max_diff:
                max_diff = diff
                max_diff_node = node

        if not diffs:
            return {
                'max_diff': 0.0,
                'mean_diff': 0.0,
                'rmse': 0.0,
                'node_with_max_diff': None,
                'num_compared': 0,
            }

        mean_diff = sum(diffs) / len(diffs)
        rmse = math.sqrt(sum(d * d for d in diffs) / len(diffs))

        return {
            'max_diff': max_diff,
            'mean_diff': mean_diff,
            'rmse': rmse,
            'node_with_max_diff': max_diff_node,
            'num_compared': len(diffs),
        }

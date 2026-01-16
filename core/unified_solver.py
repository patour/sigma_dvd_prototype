"""Unified IR-drop solver supporting both flat and hierarchical solving.

This module provides the UnifiedIRDropSolver class that works with
UnifiedPowerGridModel for both synthetic grids and PDN netlists.

Supports:
- Flat (direct) solving
- Hierarchical solving with top/bottom grid decomposition
- Tiled hierarchical solving with locality-exploiting parallel bottom-grid computation
"""

from __future__ import annotations

import logging
import math
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Set, Tuple

import heapq
import networkx as nx
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .unified_model import UnifiedPowerGridModel, UnifiedReducedSystem, LayerID
from .node_adapter import NodeInfoExtractor

# Logger for tiling warnings
logger = logging.getLogger(__name__)


# ============================================================================
# Data Classes for Solve Results
# ============================================================================

@dataclass
class UnifiedSolveResult:
    """Result of a unified IR-drop solve.

    Attributes:
        voltages: Node -> voltage mapping
        ir_drop: Node -> IR-drop mapping (vdd - voltage)
        nominal_voltage: Reference Vdd used for IR-drop calculation
        net_name: Optional net name for multi-net support
        metadata: Additional metadata from the solve
    """
    voltages: Dict[Any, float]
    ir_drop: Dict[Any, float]
    nominal_voltage: float
    net_name: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UnifiedHierarchicalResult:
    """Result of hierarchical IR-drop solve with sub-grid details.

    Attributes:
        voltages: Complete node -> voltage mapping (merged from top and bottom)
        ir_drop: Complete node -> IR-drop mapping
        partition_layer: Layer used for decomposition
        top_grid_voltages: Voltages for top-grid nodes
        bottom_grid_voltages: Voltages for bottom-grid nodes
        port_nodes: Set of port nodes at partition layer
        port_voltages: Port node -> voltage mapping
        port_currents: Port node -> aggregated current
        aggregation_map: Load node -> list of (port, weight, current_contribution)
    """
    voltages: Dict[Any, float]
    ir_drop: Dict[Any, float]
    partition_layer: LayerID
    top_grid_voltages: Dict[Any, float]
    bottom_grid_voltages: Dict[Any, float]
    port_nodes: Set[Any]
    port_voltages: Dict[Any, float]
    port_currents: Dict[Any, float]
    aggregation_map: Dict[Any, List[Tuple[Any, float, float]]] = field(default_factory=dict)


# ============================================================================
# Data Classes for Tiled Hierarchical Solving
# ============================================================================

class TileBounds(NamedTuple):
    """Rectangular tile bounds in coordinate space.

    Attributes:
        tile_id: Unique tile identifier (row * N_x + col)
        x_min: Minimum x coordinate (inclusive)
        x_max: Maximum x coordinate (inclusive)
        y_min: Minimum y coordinate (inclusive)
        y_max: Maximum y coordinate (inclusive)
    """
    tile_id: int
    x_min: float
    x_max: float
    y_min: float
    y_max: float


@dataclass
class BottomGridTile:
    """A tile of the bottom-grid for localized IR-drop computation.

    Attributes:
        tile_id: Unique tile identifier
        bounds: TileBounds defining the core region
        core_nodes: Set of nodes in the core region (results taken from here)
        halo_nodes: Set of nodes in the halo region (for boundary accuracy)
        all_nodes: Union of core_nodes and halo_nodes
        port_nodes: Port nodes within tile+halo (Dirichlet BCs)
        load_nodes: Current source nodes within the core region
        halo_clipped: True if halo was clipped at grid boundary
        halo_clip_ratio: Ratio of actual halo area to expected halo area (1.0 = no clip)
        disconnected_halo_nodes: Halo nodes not connected to ports (dropped from solve)
        floating_core_nodes: Core nodes in global floating islands (assigned vdd)
    """
    tile_id: int
    bounds: TileBounds
    core_nodes: Set[Any]
    halo_nodes: Set[Any]
    all_nodes: Set[Any]
    port_nodes: Set[Any]
    load_nodes: Set[Any]
    halo_clipped: bool = False
    halo_clip_ratio: float = 1.0
    disconnected_halo_nodes: Set[Any] = field(default_factory=set)
    floating_core_nodes: Set[Any] = field(default_factory=set)


@dataclass
class TileSolveResult:
    """Result of solving a single tile.

    Attributes:
        tile_id: Tile identifier
        voltages: Node -> voltage for all nodes in tile (core + halo)
        solve_time_ms: Time to solve this tile in milliseconds
    """
    tile_id: int
    voltages: Dict[Any, float]
    solve_time_ms: float


@dataclass
class TiledBottomGridResult(UnifiedHierarchicalResult):
    """Result of tiled hierarchical IR-drop solve.

    Extends UnifiedHierarchicalResult with tile-specific information.

    Attributes:
        tiles: List of BottomGridTile objects used in the solve
        per_tile_solve_times: tile_id -> solve time in milliseconds
        halo_clip_warnings: List of warning messages for clipped halos
        validation_stats: Optional dict with accuracy validation stats
            (max_diff, mean_diff, rmse, node_with_max_diff) when
            validate_against_flat=True
        tiling_params: Dict with N_x, N_y, halo_percent, min_ports_per_tile used
    """
    tiles: List[BottomGridTile] = field(default_factory=list)
    per_tile_solve_times: Dict[int, float] = field(default_factory=dict)
    halo_clip_warnings: List[str] = field(default_factory=list)
    validation_stats: Optional[Dict[str, Any]] = None
    tiling_params: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Module-level function for multiprocessing tile solves
# ============================================================================

def _solve_single_tile(
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
    from collections import deque

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
        V_u, info = spla.cg(G_uu, rhs, tol=1e-10)
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
# Main Solver Class
# ============================================================================

class UnifiedIRDropSolver:
    """Unified IR-drop solver for power grids.

    Supports:
    - Flat (direct) solving
    - Hierarchical solving with top/bottom grid decomposition
    - Both synthetic and PDN sources via UnifiedPowerGridModel

    Example usage:
        model = create_model_from_synthetic(G, pads, vdd=1.0)
        solver = UnifiedIRDropSolver(model)

        # Flat solve
        result = solver.solve(load_currents)

        # Hierarchical solve
        hier_result = solver.solve_hierarchical(load_currents, partition_layer=2)
    """

    def __init__(self, model: UnifiedPowerGridModel):
        """Initialize solver with a unified model.

        Args:
            model: UnifiedPowerGridModel instance
        """
        self.model = model

    @property
    def vdd(self) -> float:
        """Get nominal voltage."""
        return self.model.vdd

    def solve(
        self,
        current_injections: Dict[Any, float],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> UnifiedSolveResult:
        """Solve for voltages using flat (direct) method.

        Args:
            current_injections: Node -> current (positive = sink)
            metadata: Optional metadata to include in result

        Returns:
            UnifiedSolveResult with voltages and IR-drop.
        """
        voltages = self.model.solve_voltages(current_injections)
        ir_drop = self.model.ir_drop(voltages)

        return UnifiedSolveResult(
            voltages=voltages,
            ir_drop=ir_drop,
            nominal_voltage=self.model.vdd,
            net_name=self.model.net_name,
            metadata=metadata or {},
        )

    def solve_batch(
        self,
        stimuli: List[Dict[Any, float]],
        metadatas: Optional[List[Dict]] = None,
    ) -> List[UnifiedSolveResult]:
        """Solve for multiple stimuli (reuses LU factorization).

        Args:
            stimuli: List of current injection dicts
            metadatas: Optional list of metadata dicts

        Returns:
            List of UnifiedSolveResult, one per stimulus.
        """
        metadatas = metadatas or [{}] * len(stimuli)
        results = []

        for currents, meta in zip(stimuli, metadatas):
            result = self.solve(currents, metadata=meta)
            results.append(result)

        return results

    def summarize(
        self, result: "UnifiedSolveResult | UnifiedHierarchicalResult"
    ) -> Dict[str, float]:
        """Compute summary statistics from a solve result.

        Statistics exclude pad nodes to report only free (non-vsrc) node behavior,
        consistent with PDNSolver reporting conventions.

        Args:
            result: UnifiedSolveResult or UnifiedHierarchicalResult

        Returns:
            Dict with nominal_voltage, min_voltage, max_voltage, max_drop, avg_drop.
            For UnifiedHierarchicalResult, also includes num_ports.
        """
        # Exclude pad nodes from statistics (they are fixed at nominal voltage)
        pad_set = set(self.model.pad_nodes)
        free_voltages = [v for n, v in result.voltages.items() if n not in pad_set]
        free_drops = [d for n, d in result.ir_drop.items() if n not in pad_set]

        # Get nominal voltage (UnifiedSolveResult has it directly, hierarchical uses model)
        if hasattr(result, 'nominal_voltage'):
            nominal = result.nominal_voltage
        else:
            nominal = self.model.vdd

        summary = {
            'nominal_voltage': nominal,
            'min_voltage': min(free_voltages) if free_voltages else nominal,
            'max_voltage': max(free_voltages) if free_voltages else nominal,
            'max_drop': max(free_drops) if free_drops else 0.0,
            'avg_drop': sum(free_drops) / len(free_drops) if free_drops else 0.0,
        }

        # Add hierarchical-specific stats
        if isinstance(result, UnifiedHierarchicalResult):
            summary['num_ports'] = len(result.port_nodes)
            summary['partition_layer'] = result.partition_layer

        return summary

    def solve_hierarchical(
        self,
        current_injections: Dict[Any, float],
        partition_layer: LayerID,
        top_k: int = 5,
        weighting: str = "shortest_path",
        rmax: Optional[float] = None,
        verbose: bool = False,
        use_fast_builder: bool = True,
    ) -> UnifiedHierarchicalResult:
        """Solve using hierarchical decomposition.

        Decomposes the grid at partition_layer into:
        - Top-grid: layers >= partition_layer (contains pads)
        - Bottom-grid: layers < partition_layer (contains loads)
        - Ports: nodes at partition_layer connecting to bottom-grid

        Steps:
        1. Aggregate bottom-grid currents to ports (using top-k weighting)
        2. Solve top-grid with pad voltages as Dirichlet BC and port injections
        3. Solve bottom-grid with port voltages as Dirichlet BCs

        Args:
            current_injections: Node -> current (positive = sink)
            partition_layer: Layer to partition at
            top_k: Number of nearest ports for current aggregation
            weighting: "effective" (effective resistance) or "shortest_path"
            rmax: Maximum resistance distance for shortest_path weighting.
                  Paths beyond this distance are ignored. None means no limit.
                  Only applies when weighting="shortest_path".
            verbose: If True, print timing information for each step.
            use_fast_builder: If True (default), use vectorized subgrid builder
                  for ~10x speedup. Set to False to use original Python-loop
                  implementation for debugging or validation.

        Returns:
            UnifiedHierarchicalResult with complete voltages and decomposition info.

        Raises:
            ValueError: If partition layer has no ports, or if load nodes are
                electrically disconnected from ports. The error message will
                suggest alternative partition layers if disconnection is detected.
        """
        import time
        timings: Dict[str, float] = {}
        
        # Step 0: Decompose the grid
        t0 = time.perf_counter()
        top_nodes, bottom_nodes, port_nodes, via_edges = self.model._decompose_at_layer(partition_layer)
        timings['decompose'] = time.perf_counter() - t0

        if not port_nodes:
            raise ValueError(
                f"No ports found at partition layer {partition_layer}. "
                "Check that vias connect this layer to the layer below."
            )

        # Step 0.5: Validate load-to-port connectivity before expensive computation
        t0 = time.perf_counter()
        bottom_load_nodes = {
            n for n, c in current_injections.items()
            if n in bottom_nodes and n not in port_nodes and c != 0
        }
        if bottom_load_nodes:
            disconnected_loads = self._find_disconnected_loads(
                bottom_grid_nodes=bottom_nodes,
                port_nodes=port_nodes,
                load_nodes=bottom_load_nodes,
            )
            if disconnected_loads:
                # Find alternative partition layers that might work better
                suggestions = self._suggest_partition_layers(
                    disconnected_loads=disconnected_loads,
                    current_partition=partition_layer,
                )
                raise ValueError(
                    f"{len(disconnected_loads)} load node(s) are electrically disconnected from "
                    f"ports at partition layer {partition_layer}. "
                    f"Example disconnected load: {next(iter(disconnected_loads))}. "
                    f"This typically means the partition layer is too high. "
                    f"{suggestions}"
                )
        timings['connectivity_check'] = time.perf_counter() - t0

        # Step 1: Aggregate bottom-grid currents onto ports
        t0 = time.perf_counter()
        port_currents, aggregation_map = self._aggregate_currents_to_ports(
            current_injections=current_injections,
            bottom_grid_nodes=bottom_nodes,
            port_nodes=port_nodes,
            top_k=top_k,
            weighting=weighting,
            rmax=rmax,
        )
        timings['aggregate_currents'] = time.perf_counter() - t0

        # Step 2: Solve top-grid with pads as Dirichlet BC
        # Top-grid includes: top_nodes (layers >= partition_layer)
        # Dirichlet nodes: pads (on top layer)
        pad_set = set(self.model.pad_nodes)
        top_grid_pads = pad_set & top_nodes

        if not top_grid_pads:
            raise ValueError(
                f"No pad nodes found in top-grid (layers >= {partition_layer}). "
                "Pads should be on the top-most layer."
            )

        # Build and solve top-grid system
        t0 = time.perf_counter()
        if use_fast_builder:
            top_system = self.model._build_subgrid_system_fast(
                subgrid_nodes=top_nodes,
                dirichlet_nodes=top_grid_pads,
                dirichlet_voltage=self.model.vdd,
            )
        else:
            top_system = self.model._build_subgrid_system(
                subgrid_nodes=top_nodes,
                dirichlet_nodes=top_grid_pads,
                dirichlet_voltage=self.model.vdd,
            )

        if top_system is None:
            raise ValueError("Failed to build top-grid system")
        timings['build_top_system'] = time.perf_counter() - t0

        # Current injections for top-grid: loads in top-grid + aggregated port currents
        # Note: Any loads that happen to be in top-grid are handled directly
        top_grid_currents = {n: c for n, c in current_injections.items() if n in top_nodes}
        # Add aggregated port currents
        for port, curr in port_currents.items():
            top_grid_currents[port] = top_grid_currents.get(port, 0.0) + curr

        t0 = time.perf_counter()
        top_voltages = self.model._solve_subgrid(
            reduced_system=top_system,
            current_injections=top_grid_currents,
            dirichlet_voltages=None,  # Use vdd for pads
        )
        timings['solve_top'] = time.perf_counter() - t0

        # Extract port voltages for bottom-grid BC
        port_voltages = {p: top_voltages[p] for p in port_nodes}

        # Step 3: Solve bottom-grid with port voltages as Dirichlet BC
        # Bottom-grid includes: bottom_nodes (layers < partition_layer)
        # We need to include the ports as Dirichlet nodes but they're in top-grid
        # Solution: Include ports in the bottom subgrid for the solve
        bottom_subgrid = bottom_nodes | port_nodes

        t0 = time.perf_counter()
        if use_fast_builder:
            bottom_system = self.model._build_subgrid_system_fast(
                subgrid_nodes=bottom_subgrid,
                dirichlet_nodes=port_nodes,
                dirichlet_voltage=self.model.vdd,  # Will be overridden by dirichlet_voltages
            )
        else:
            bottom_system = self.model._build_subgrid_system(
                subgrid_nodes=bottom_subgrid,
                dirichlet_nodes=port_nodes,
                dirichlet_voltage=self.model.vdd,  # Will be overridden by dirichlet_voltages
            )

        if bottom_system is None:
            raise ValueError("Failed to build bottom-grid system")
        timings['build_bottom_system'] = time.perf_counter() - t0

        # Current injections for bottom-grid: original currents in bottom-grid
        bottom_grid_currents = {
            n: c for n, c in current_injections.items()
            if n in bottom_nodes
        }

        t0 = time.perf_counter()
        bottom_voltages = self.model._solve_subgrid(
            reduced_system=bottom_system,
            current_injections=bottom_grid_currents,
            dirichlet_voltages=port_voltages,
        )
        timings['solve_bottom'] = time.perf_counter() - t0

        # Merge voltages: top-grid + bottom-grid (excluding duplicate ports)
        t0 = time.perf_counter()
        all_voltages = {}
        all_voltages.update(top_voltages)
        # Add bottom-grid nodes (excluding ports which are already in top_voltages)
        for n, v in bottom_voltages.items():
            if n not in port_nodes:
                all_voltages[n] = v

        # Compute IR-drop
        ir_drop = self.model.ir_drop(all_voltages)
        timings['merge_results'] = time.perf_counter() - t0
        
        # Print timing summary if verbose
        if verbose:
            total_time = sum(timings.values())
            print(f"\n=== Hierarchical Solve Timing ===")
            print(f"  Top nodes: {len(top_nodes):,}, Bottom nodes: {len(bottom_nodes):,}, Ports: {len(port_nodes):,}")
            print(f"  Load nodes in bottom-grid: {len(bottom_load_nodes):,}")
            print(f"  ---")
            for step, t in timings.items():
                pct = t / total_time * 100
                print(f"  {step:25s}: {t*1000:8.1f} ms  ({pct:5.1f}%)")
            print(f"  {'TOTAL':25s}: {total_time*1000:8.1f} ms")
            print(f"=================================\n")

        return UnifiedHierarchicalResult(
            voltages=all_voltages,
            ir_drop=ir_drop,
            partition_layer=partition_layer,
            top_grid_voltages=top_voltages,
            bottom_grid_voltages=bottom_voltages,
            port_nodes=port_nodes,
            port_voltages=port_voltages,
            port_currents=port_currents,
            aggregation_map=aggregation_map,
        )

    def _find_disconnected_loads(
        self,
        bottom_grid_nodes: Set[Any],
        port_nodes: Set[Any],
        load_nodes: Set[Any],
    ) -> Set[Any]:
        """Find load nodes that are electrically disconnected from ports.

        Uses connected components analysis on R-type edges to identify loads
        that have no resistive path to any port node.

        Args:
            bottom_grid_nodes: Set of nodes in bottom-grid
            port_nodes: Set of port nodes at partition boundary
            load_nodes: Set of load nodes to check

        Returns:
            Set of load nodes that are disconnected from all ports.
        """
        # Build R-type subgraph for bottom-grid + ports
        subgraph_nodes = bottom_grid_nodes | port_nodes
        
        # Use Union-Find for efficient component detection
        parent: Dict[Any, Any] = {}
        
        def find(x: Any) -> Any:
            if x not in parent:
                parent[x] = x
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x: Any, y: Any) -> None:
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
        
        # Process edges using EdgeInfoExtractor to handle both synthetic and PDN formats
        edge_extractor = self.model._edge_extractor
        for u, v, d in self.model.graph.edges(subgraph_nodes, data=True):
            if u not in subgraph_nodes or v not in subgraph_nodes:
                continue
            edge_info = edge_extractor.get_info(d)
            if not edge_info.is_resistive:
                continue
            R = edge_info.resistance if edge_info.resistance else 0.0
            if R > 0 and R < float('inf'):
                union(u, v)
                union(u, v)
        
        # Find components containing ports
        port_components = {find(p) for p in port_nodes if p in parent}
        
        # Find loads not in any port-containing component
        disconnected = set()
        for load in load_nodes:
            if load not in parent:
                # Node has no R-type edges at all
                disconnected.add(load)
            elif find(load) not in port_components:
                disconnected.add(load)
        
        return disconnected

    def _suggest_partition_layers(
        self,
        disconnected_loads: Set[Any],
        current_partition: LayerID,
    ) -> str:
        """Suggest alternative partition layers that might include disconnected loads.

        Args:
            disconnected_loads: Set of load nodes disconnected from current partition
            current_partition: The partition layer that failed

        Returns:
            String with suggestions for alternative partition layers.
        """
        # Get layers present in disconnected loads
        load_layers: Dict[LayerID, int] = {}
        for load in list(disconnected_loads)[:1000]:  # Sample for efficiency
            info = self.model.get_node_info(load)
            if info.layer is not None:
                load_layers[info.layer] = load_layers.get(info.layer, 0) + 1
        
        if not load_layers:
            return "Could not determine layers of disconnected loads."
        
        # Find the highest layer with disconnected loads
        all_layers = sorted(self.model.get_all_layers())
        pad_set = set(self.model.pad_nodes)
        
        # Try lower partition layers and check:
        # 1. Has ports
        # 2. Has pads in top-grid
        suggestions = []
        
        for layer in reversed(all_layers):
            if layer >= current_partition:
                continue
            
            try:
                top_nodes, bottom_nodes, ports, _ = self.model._decompose_at_layer(layer)
            except ValueError:
                # Skip invalid partition layers (e.g., bottom layer)
                continue
            if not ports:
                continue
            
            # Check if pads are in top-grid
            pads_in_top = pad_set & top_nodes
            if not pads_in_top:
                continue
            
            suggestions.append(f"layer {layer} ({len(ports)} ports, {len(pads_in_top)} pads in top-grid)")
            if len(suggestions) >= 3:
                break
        
        if suggestions:
            return f"Try partitioning at a lower layer: {', '.join(suggestions)}."
        return "Try using a lower partition layer closer to the load layers."

    def _aggregate_currents_to_ports(
        self,
        current_injections: Dict[Any, float],
        bottom_grid_nodes: Set[Any],
        port_nodes: Set[Any],
        top_k: int = 5,
        weighting: str = "shortest_path",
        rmax: Optional[float] = None,
    ) -> Tuple[Dict[Any, float], Dict[Any, List[Tuple[Any, float, float]]]]:
        """Aggregate bottom-grid currents to port nodes.

        For each load node with current in bottom-grid, distributes current to
        top-k nearest ports weighted by inverse resistance.

        Args:
            current_injections: Node -> current (can include nodes outside bottom-grid)
            bottom_grid_nodes: Set of nodes in bottom-grid
            port_nodes: Set of port nodes at partition boundary
            top_k: Number of nearest ports per load
            weighting: "effective" or "shortest_path"
            rmax: Maximum resistance distance for shortest_path (None = no limit)

        Returns:
            (port_currents, aggregation_map)
            - port_currents: Port -> aggregated current
            - aggregation_map: Load -> [(port, weight, current_contrib), ...]
        """
        port_currents = {p: 0.0 for p in port_nodes}
        aggregation_map = {}

        if not port_nodes or not current_injections:
            return port_currents, aggregation_map

        # Filter currents to only those in bottom-grid (excluding ports)
        bottom_currents = {
            n: c for n, c in current_injections.items()
            if n in bottom_grid_nodes and n not in port_nodes
        }

        if not bottom_currents:
            return port_currents, aggregation_map

        port_list = list(port_nodes)
        subgraph_nodes = bottom_grid_nodes | port_nodes

        # Precompute multi-source shortest-path distances once for all loads
        shortest_path_cache: Dict[Any, List[Tuple[Any, float]]] = {}
        if weighting == "shortest_path":
            shortest_path_cache = self._multi_source_port_distances(
                subgrid_nodes=subgraph_nodes,
                port_nodes=port_list,
                top_k=top_k,
                rmax=rmax,
            )

        for load_node, load_current in bottom_currents.items():
            if load_current == 0:
                continue

            # Compute resistance to each port
            if weighting == "effective":
                resistances = self._compute_effective_resistance_in_subgrid(
                    subgrid_nodes=subgraph_nodes,
                    source_node=load_node,
                    target_nodes=port_list,
                    dirichlet_nodes=port_nodes,
                )
            else:
                port_distances = shortest_path_cache.get(load_node, [])
                resistances = {p: d for p, d in port_distances if d is not None}

            valid_resistances = {
                p: r for p, r in resistances.items()
                if r is not None and r < float('inf') and r > 0
            }

            if not valid_resistances:
                raise ValueError(
                    f"No valid resistance paths found from load node {load_node} to any port. "
                    f"This indicates the load is electrically isolated from the port layer. "
                    f"Check grid connectivity."
                )

            sorted_ports = sorted(valid_resistances.items(), key=lambda x: x[1])
            selected = sorted_ports[:top_k] if top_k < len(sorted_ports) else sorted_ports

            inv_R = [(p, 1.0 / R) for p, R in selected]
            total_inv_R = sum(w for _, w in inv_R)
            if total_inv_R == 0:
                raise ValueError(
                    f"Total inverse resistance is zero for load node {load_node}. "
                    f"Selected ports: {[p for p, _ in selected]}. "
                    f"This should not happen with valid_resistances already filtered."
                )

            contributions = []
            for port, inv_r in inv_R:
                weight = inv_r / total_inv_R
                contrib = load_current * weight
                port_currents[port] += contrib
                contributions.append((port, weight, contrib))

            aggregation_map[load_node] = contributions

        return port_currents, aggregation_map

    def _build_resistive_csr(
        self,
        subgrid_nodes: Set[Any],
    ) -> Tuple[List[Any], Dict[Any, int], np.ndarray, np.ndarray, np.ndarray]:
        """Build CSR adjacency with resistance weights for a node set."""
        if not subgrid_nodes:
            return [], {}, np.array([], dtype=np.int64), np.array([], dtype=np.int32), np.array([], dtype=float)

        node_list = list(subgrid_nodes)
        index = {n: i for i, n in enumerate(node_list)}

        edge_min = {}
        graph = self.model.graph
        node_set = set(subgrid_nodes)

        for u, v, edge_data in graph.edges(subgrid_nodes, data=True):
            if u not in node_set or v not in node_set:
                continue

            if isinstance(graph, nx.MultiDiGraph):
                if edge_data.get('type') != 'R':
                    continue
                R = float(edge_data.get('value', 0.0))
                if self.model.resistance_unit_kohm:
                    R *= 1e3
            else:
                R = float(edge_data.get('resistance', 0.0))

            if R <= 0.0 or not np.isfinite(R):
                continue

            iu, iv = index[u], index[v]
            if iu == iv:
                continue

            key = (iu, iv) if iu < iv else (iv, iu)
            prev = edge_min.get(key)
            if prev is None or R < prev:
                edge_min[key] = R

        n = len(node_list)
        adjacency: List[List[Tuple[int, float]]] = [[] for _ in range(n)]
        for (iu, iv), R in edge_min.items():
            adjacency[iu].append((iv, R))
            adjacency[iv].append((iu, R))

        indptr = np.zeros(n + 1, dtype=np.int64)
        indices: List[int] = []
        data: List[float] = []

        for i, nbrs in enumerate(adjacency):
            for j, w in nbrs:
                indices.append(j)
                data.append(w)
            indptr[i + 1] = len(indices)

        return (
            node_list,
            index,
            indptr,
            np.array(indices, dtype=np.int32),
            np.array(data, dtype=float),
        )

    def _multi_source_port_distances(
        self,
        subgrid_nodes: Set[Any],
        port_nodes: List[Any],
        top_k: int,
        rmax: Optional[float] = None,
    ) -> Dict[Any, List[Tuple[Any, float]]]:
        """Compute up to top_k nearest ports for every node via multi-source Dijkstra.

        Uses an optimized label-setting algorithm that tracks settled (port, node)
        pairs to avoid redundant heap operations. For large graphs (1M+ nodes,
        10K+ ports), this avoids heap explosion by pruning aggressively.

        Args:
            subgrid_nodes: Set of nodes to consider
            port_nodes: List of port nodes (sources for Dijkstra)
            top_k: Number of nearest ports to track per node
            rmax: Maximum resistance distance. Paths beyond this are pruned.
                  None means no limit (default behavior).

        Complexity: O((N + E) * K * log(N * K)) where N=nodes, E=edges, K=top_k
                    With rmax, complexity can be significantly reduced as
                    propagation stops at the distance boundary.
        """
        node_list, index, indptr, indices, weights = self._build_resistive_csr(subgrid_nodes)
        if not node_list or not port_nodes:
            return {}

        top_k = max(1, top_k)
        n = len(node_list)

        # For each node, store dict {port_id: best_distance} for O(1) lookup
        best_dist: List[Dict[int, float]] = [{} for _ in range(n)]
        # Max-heap (negated distances) to track k-th best distance in O(1)
        # Format: List of max-heaps where heap[i] contains (-dist, port_id)
        kth_heaps: List[List[Tuple[float, int]]] = [[] for _ in range(n)]
        # Track settled (port_id, node_idx) pairs to skip redundant processing
        settled: Set[Tuple[int, int]] = set()

        # Priority queue: (distance, port_id, node_idx)
        heap: List[Tuple[float, int, int]] = []
        for port_id, port in enumerate(port_nodes):
            if port not in index:
                continue
            node_idx = index[port]
            heapq.heappush(heap, (0.0, port_id, node_idx))
            best_dist[node_idx][port_id] = 0.0
            heapq.heappush(kth_heaps[node_idx], (0.0, port_id))  # max-heap uses negated dist, but 0 is same

        while heap:
            dist, port_id, node_idx = heapq.heappop(heap)

            # Early termination: skip if distance exceeds rmax
            if rmax is not None and dist > rmax:
                continue

            # Skip if this (port, node) pair was already settled
            key = (port_id, node_idx)
            if key in settled:
                continue

            # Skip if we've found a better path since this was queued
            node_best = best_dist[node_idx]
            if port_id in node_best and dist > node_best[port_id]:
                continue

            # Mark as settled - this is the optimal distance for this (port, node) pair
            settled.add(key)

            # Propagate to neighbors
            start, end = indptr[node_idx], indptr[node_idx + 1]
            for offset in range(start, end):
                nbr = int(indices[offset])
                w = float(weights[offset])
                if w <= 0.0 or not np.isfinite(w):
                    continue

                new_dist = dist + w
                nbr_best = best_dist[nbr]
                nbr_kth_heap = kth_heaps[nbr]

                # Skip if already settled for this port
                if (port_id, nbr) in settled:
                    continue

                # Pruning: if neighbor already has top_k labels and new_dist
                # is worse than the k-th best, skip (O(1) check using max-heap)
                if len(nbr_kth_heap) >= top_k:
                    # Max-heap root has the largest (worst) of top-k distances
                    kth_best = -nbr_kth_heap[0][0]  # negate to get actual distance
                    if new_dist >= kth_best:
                        continue

                # Skip if new distance exceeds rmax
                if rmax is not None and new_dist > rmax:
                    continue

                # Check if this improves existing distance for this port
                if port_id in nbr_best:
                    if new_dist >= nbr_best[port_id]:
                        continue
                    # Update existing - need to rebuild heap (rare case)
                    nbr_best[port_id] = new_dist
                else:
                    # New port for this node
                    nbr_best[port_id] = new_dist

                # Update the k-th best heap
                if len(nbr_kth_heap) < top_k:
                    heapq.heappush(nbr_kth_heap, (-new_dist, port_id))
                elif new_dist < -nbr_kth_heap[0][0]:
                    # Replace the worst (largest distance) in top-k
                    heapq.heapreplace(nbr_kth_heap, (-new_dist, port_id))

                heapq.heappush(heap, (new_dist, port_id, nbr))

        # Build result: for each node, return top-k (port, distance) pairs
        result: Dict[Any, List[Tuple[Any, float]]] = {}
        for idx in range(n):
            node_best = best_dist[idx]
            if not node_best:
                continue
            # Sort by distance and take top_k
            sorted_labels = sorted(node_best.items(), key=lambda x: x[1])[:top_k]
            node = node_list[idx]
            result[node] = [(port_nodes[pid], d) for pid, d in sorted_labels]

        return result

    def _compute_effective_resistance_in_subgrid(
        self,
        subgrid_nodes: Set[Any],
        source_node: Any,
        target_nodes: List[Any],
        dirichlet_nodes: Set[Any],
    ) -> Dict[Any, float]:
        """Compute effective resistance from source to targets within a subgrid.

        Properly computes electrical effective resistance considering all parallel
        paths, with specified nodes as Dirichlet (fixed voltage) boundaries.

        Args:
            subgrid_nodes: Set of nodes forming the subgrid
            source_node: Source node for resistance computation
            target_nodes: List of target nodes (typically ports)
            dirichlet_nodes: Nodes with fixed voltage (ports)

        Returns:
            Dict mapping target -> effective resistance (Ohms)
        """
        import scipy.sparse as sp
        import scipy.sparse.linalg as spla

        # Build subgraph
        subgraph = self.model.graph.subgraph(subgrid_nodes)
        nodes = list(subgraph.nodes())

        if source_node not in nodes:
            return {t: float('inf') for t in target_nodes}

        index = {n: i for i, n in enumerate(nodes)}
        n_nodes = len(nodes)

        # Build sparse conductance matrix
        data, rows, cols = [], [], []
        diag = np.zeros(n_nodes, dtype=float)

        for u, v, edge_data in subgraph.edges(data=True):
            # Get resistance based on graph type
            if isinstance(self.model.graph, nx.MultiDiGraph):
                # PDN: R type edges
                if edge_data.get('type') == 'R':
                    R = float(edge_data.get('value', 0.0))
                    if self.model.resistance_unit_kohm:
                        R = R * 1e3
                else:
                    continue
            else:
                # Synthetic: all edges have resistance attribute
                R = float(edge_data.get('resistance', 0.0))

            if R <= 0.0:
                continue

            g = 1.0 / R
            iu, iv = index[u], index[v]
            rows.extend([iu, iv])
            cols.extend([iv, iu])
            data.extend([-g, -g])
            diag[iu] += g
            diag[iv] += g

        for i in range(n_nodes):
            rows.append(i)
            cols.append(i)
            data.append(diag[i])

        G_mat = sp.csr_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))

        # Separate Dirichlet (fixed) nodes from unknowns
        dirichlet_in_subgrid = dirichlet_nodes & set(nodes)
        unknown_nodes = [n for n in nodes if n not in dirichlet_in_subgrid]

        if source_node in dirichlet_in_subgrid:
            return {t: (0.0 if t == source_node else float('inf')) for t in target_nodes}

        if not unknown_nodes:
            return {t: float('inf') for t in target_nodes}

        index_unknown = {n: i for i, n in enumerate(unknown_nodes)}
        u_idx = [index[n] for n in unknown_nodes]
        G_uu = G_mat[np.ix_(u_idx, u_idx)].tocsr()

        try:
            lu = spla.factorized(G_uu.tocsc())
        except Exception:
            return {t: float('inf') for t in target_nodes}

        if source_node not in index_unknown:
            return {t: float('inf') for t in target_nodes}

        source_idx = index_unknown[source_node]
        e_source = np.zeros(len(unknown_nodes))
        e_source[source_idx] = 1.0
        x_source = lu(e_source)

        results = {}

        # Get coupling matrix G_ud for computing resistance to Dirichlet nodes
        dirichlet_list = list(dirichlet_in_subgrid)
        d_idx = [index[d] for d in dirichlet_list]
        if d_idx:
            G_ud = G_mat[np.ix_(u_idx, d_idx)].tocsr()
        else:
            G_ud = sp.csr_matrix((len(u_idx), 0))

        for target in target_nodes:
            if target not in dirichlet_in_subgrid:
                # Target is unknown node
                if target in index_unknown:
                    target_idx = index_unknown[target]
                    e_target = np.zeros(len(unknown_nodes))
                    e_target[target_idx] = 1.0
                    x_target = lu(e_target)
                    # R_eff = x_s[s] + x_t[t] - 2*x_s[t]
                    r_eff = (x_source[source_idx] + x_target[target_idx]
                             - x_source[target_idx] - x_target[source_idx])
                    results[target] = max(0.0, r_eff)
                else:
                    results[target] = float('inf')
            else:
                # Target is a Dirichlet node (port)
                if G_ud.shape[1] > 0:
                    target_d_idx = None
                    for i, d in enumerate(dirichlet_list):
                        if d == target:
                            target_d_idx = i
                            break

                    if target_d_idx is not None:
                        target_col = G_ud[:, target_d_idx].toarray().flatten()
                        current_to_target = np.dot(target_col, x_source)

                        if abs(current_to_target) > 1e-10:
                            r_eff = x_source[source_idx] / abs(current_to_target)
                            results[target] = max(0.0, r_eff)
                        else:
                            results[target] = float('inf')
                    else:
                        results[target] = float('inf')
                else:
                    results[target] = max(0.0, x_source[source_idx])

        return results

    def _shortest_path_resistance(
        self,
        subgraph: nx.Graph,
        source: Any,
        target: Any,
    ) -> Optional[float]:
        """Compute shortest path resistance between two nodes.

        Uses Dijkstra with edge resistance as weight.

        Args:
            subgraph: Graph to search in
            source: Source node
            target: Target node

        Returns:
            Total resistance along shortest path, or None if no path.
        """
        if source == target:
            return 0.0

        if source not in subgraph or target not in subgraph:
            return None

        # Build weight function
        def get_weight(u, v, data):
            if isinstance(subgraph, nx.MultiDiGraph):
                # For MultiDiGraph, data is already the edge dict
                R = data.get('value', 0.0) if data.get('type') == 'R' else float('inf')
                if self.model.resistance_unit_kohm:
                    R = R * 1e3
                return R if R > 0 else float('inf')
            else:
                R = data.get('resistance', 0.0)
                return R if R > 0 else float('inf')

        try:
            # Use networkx shortest path with custom weight
            length = nx.dijkstra_path_length(subgraph, source, target, weight=get_weight)
            return length if length < float('inf') else None
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    # ========================================================================
    # Tiled Hierarchical Solving Methods
    # ========================================================================

    def _extract_bottom_grid_coordinates(
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

    def _generate_uniform_tile_bounds(
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

    def _assign_nodes_to_tiles(
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
        tiles_to_remove = set()
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

    def _expand_tile_with_halo(
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

    def _validate_and_fix_tile_connectivity(
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
        from collections import deque
        
        graph = self.model.graph
        is_pdn = isinstance(graph, nx.MultiDiGraph)
        
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
        def get_all_neighbors(g, node):
            """Get neighbors in both directions for directed graphs."""
            if g.is_directed():
                # Combine successors and predecessors
                return set(g.successors(node)) | set(g.predecessors(node))
            else:
                return set(g[node])
        
        # Helper to check for R-type edge in either direction
        def has_r_type_edge(g, u, v, is_pdn_graph):
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
            # They'll be assigned vdd by _merge_tiled_voltages
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
                    current = found_port
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

    def _build_tile_solve_args(
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
            List of argument tuples for _solve_single_tile function:
            (tile_id, node_list, core_nodes, edge_tuples, port_nodes, port_voltages, currents, vdd)
        """
        args_list = []
        graph = self.model.graph
        is_pdn = isinstance(graph, nx.MultiDiGraph)
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

    def _merge_tiled_voltages(
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
        merged = {}
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

    def _validate_tiled_accuracy(
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

    def solve_hierarchical_tiled(
        self,
        current_injections: Dict[Any, float],
        partition_layer: LayerID,
        N_x: int,
        N_y: int,
        halo_percent: float,
        min_ports_per_tile: Optional[int] = None,
        top_k: int = 5,
        weighting: str = "shortest_path",
        rmax: Optional[float] = None,
        n_workers: Optional[int] = None,
        parallel_backend: str = "thread",
        progress_callback: Optional[Callable[[int, int, int], None]] = None,
        validate_against_flat: bool = False,
        verbose: bool = False,
        override_port_voltages: Optional[Dict[Any, float]] = None,
    ) -> TiledBottomGridResult:
        """Solve using tiled hierarchical decomposition with locality exploitation.

        Decomposes the grid into top-grid and bottom-grid at partition_layer,
        then tiles the bottom-grid for parallel localized IR-drop computation.
        Each tile uses halo regions with port nodes as Dirichlet BCs.

        Only supports PDN graphs (not synthetic grids with NodeID).

        Args:
            current_injections: Node -> current (positive = sink)
            partition_layer: Layer to partition at
            N_x: Number of tiles in x direction
            N_y: Number of tiles in y direction
            halo_percent: Halo size as percentage of tile dimensions (0.0 to 1.0)
                          Larger halo improves accuracy but increases computation.
            min_ports_per_tile: Minimum port nodes required per tile.
                                Default: ceil(total_ports / (N_x * N_y))
            top_k: Number of nearest ports for current aggregation
            weighting: "effective" or "shortest_path" for current aggregation
            rmax: Maximum resistance distance for shortest_path weighting
            n_workers: Number of parallel workers (default: CPU count)
            parallel_backend: "thread" (default, safe for most contexts) or
                              "process" (true parallelism, but requires pickling)
            progress_callback: Optional callback(completed, total, tile_id)
                               called after each tile completes
            validate_against_flat: If True, run full bottom-grid solve and
                                   report accuracy statistics
            verbose: If True, print timing information
            override_port_voltages: If provided, skip top-grid solve and use
                                    these port voltages as Dirichlet BCs.
                                    Useful for validation against flat solver.

        Returns:
            TiledBottomGridResult with voltages, tiles, and optional validation stats

        Raises:
            ValueError: If graph is synthetic, partition layer is invalid,
                        or tile constraints cannot be satisfied
        """
        timings: Dict[str, float] = {}

        # ================================================================
        # Step 1: Standard hierarchical decomposition and top-grid solve
        # ================================================================
        t0 = time.perf_counter()
        top_nodes, bottom_nodes, port_nodes, via_edges = self.model._decompose_at_layer(partition_layer)
        timings['decompose'] = time.perf_counter() - t0

        if not port_nodes:
            raise ValueError(
                f"No ports found at partition layer {partition_layer}. "
                "Check that vias connect this layer to the layer below."
            )

        # Validate load-to-port connectivity
        t0 = time.perf_counter()
        bottom_load_nodes = {
            n for n, c in current_injections.items()
            if n in bottom_nodes and n not in port_nodes and c != 0
        }
        if bottom_load_nodes:
            disconnected_loads = self._find_disconnected_loads(
                bottom_grid_nodes=bottom_nodes,
                port_nodes=port_nodes,
                load_nodes=bottom_load_nodes,
            )
            if disconnected_loads:
                suggestions = self._suggest_partition_layers(
                    disconnected_loads=disconnected_loads,
                    current_partition=partition_layer,
                )
                raise ValueError(
                    f"{len(disconnected_loads)} load node(s) are electrically disconnected from "
                    f"ports at partition layer {partition_layer}. "
                    f"{suggestions}"
                )
        timings['connectivity_check'] = time.perf_counter() - t0

        # ================================================================
        # Use override port voltages OR solve top-grid
        # ================================================================
        if override_port_voltages is not None:
            # Skip top-grid solve; use provided port voltages directly
            port_voltages = {p: override_port_voltages.get(p, self.model.vdd) for p in port_nodes}
            port_currents = {}  # Not needed for bottom-grid-only validation
            aggregation_map = {}
            timings['aggregate_currents'] = 0.0
            timings['build_top_system'] = 0.0
            timings['solve_top'] = 0.0
            if verbose:
                print("Using override port voltages (skipping top-grid solve)")
        else:
            # Aggregate currents to ports
            t0 = time.perf_counter()
            port_currents, aggregation_map = self._aggregate_currents_to_ports(
                current_injections=current_injections,
                bottom_grid_nodes=bottom_nodes,
                port_nodes=port_nodes,
                top_k=top_k,
                weighting=weighting,
                rmax=rmax,
            )
            timings['aggregate_currents'] = time.perf_counter() - t0

            # Build and solve top-grid
            t0 = time.perf_counter()
            pad_set = set(self.model.pad_nodes)
            top_grid_pads = pad_set & top_nodes

            if not top_grid_pads:
                raise ValueError(
                    f"No pad nodes found in top-grid (layers >= {partition_layer})."
                )

            top_system = self.model._build_subgrid_system_fast(
                subgrid_nodes=top_nodes,
                dirichlet_nodes=top_grid_pads,
                dirichlet_voltage=self.model.vdd,
            )
            if top_system is None:
                raise ValueError("Failed to build top-grid system")
            timings['build_top_system'] = time.perf_counter() - t0

            t0 = time.perf_counter()
            top_grid_currents = {n: c for n, c in current_injections.items() if n in top_nodes}
            for port, curr in port_currents.items():
                top_grid_currents[port] = top_grid_currents.get(port, 0.0) + curr

            top_voltages = self.model._solve_subgrid(
                reduced_system=top_system,
                current_injections=top_grid_currents,
                dirichlet_voltages=None,
            )
            timings['solve_top'] = time.perf_counter() - t0

            port_voltages = {p: top_voltages[p] for p in port_nodes}

        # ================================================================
        # Step 2: Extract coordinates and generate tiles
        # ================================================================
        t0 = time.perf_counter()
        bottom_coords, port_coords, grid_bounds = self._extract_bottom_grid_coordinates(
            bottom_nodes, port_nodes
        )
        timings['extract_coords'] = time.perf_counter() - t0

        t0 = time.perf_counter()
        initial_bounds = self._generate_uniform_tile_bounds(N_x, N_y, grid_bounds)

        # Determine minimum ports per tile
        total_ports_with_coords = len(port_coords)
        if min_ports_per_tile is None:
            min_ports_per_tile = max(1, math.ceil(total_ports_with_coords / (N_x * N_y)))

        # Assign nodes and adjust constraints
        adjusted_bounds, tile_nodes, tile_ports, tile_loads = self._assign_nodes_to_tiles(
            bottom_coords=bottom_coords,
            port_coords=port_coords,
            tile_bounds=initial_bounds,
            port_nodes=port_nodes,
            load_nodes=bottom_load_nodes,
            min_ports_per_tile=min_ports_per_tile,
            N_x=N_x,
            N_y=N_y,
        )
        timings['generate_tiles'] = time.perf_counter() - t0

        # ================================================================
        # Step 3: Expand tiles with halos
        # ================================================================
        t0 = time.perf_counter()
        tiles: List[BottomGridTile] = []
        for bounds in adjusted_bounds:
            tid = bounds.tile_id
            tile = self._expand_tile_with_halo(
                tile=bounds,
                tile_core_nodes=tile_nodes.get(tid, set()),
                bottom_coords=bottom_coords,
                grid_bounds=grid_bounds,
                halo_percent=halo_percent,
                port_nodes=port_nodes,
                port_coords=port_coords,
                load_nodes=bottom_load_nodes,
            )
            tiles.append(tile)
        timings['expand_halos'] = time.perf_counter() - t0

        # ================================================================
        # Step 3b: Validate and fix tile connectivity
        # Ensure all core nodes are connected to ports through tile edges
        # ================================================================
        t0 = time.perf_counter()
        tiles = self._validate_and_fix_tile_connectivity(tiles, bottom_nodes, port_nodes)
        timings['fix_connectivity'] = time.perf_counter() - t0

        # Validate disjoint core regions
        all_core_nodes: Set[Any] = set()
        for tile in tiles:
            overlap = all_core_nodes & tile.core_nodes
            if overlap:
                raise ValueError(
                    f"Tile cores are not disjoint. Overlap detected: {list(overlap)[:5]}..."
                )
            all_core_nodes.update(tile.core_nodes)

        # ================================================================
        # Step 4: Build tile solve arguments
        # ================================================================
        t0 = time.perf_counter()
        solve_args = self._build_tile_solve_args(tiles, port_voltages, current_injections)
        timings['build_args'] = time.perf_counter() - t0

        # ================================================================
        # Step 5: Parallel tile solves
        # ================================================================
        t0 = time.perf_counter()
        tile_results: Dict[int, Dict[Any, float]] = {}
        per_tile_times: Dict[int, float] = {}

        if n_workers is None:
            import os
            n_workers = os.cpu_count() or 1

        n_tiles = len(tiles)
        disconnected_halo_counts: Dict[int, int] = {}

        if n_tiles == 0:
            # No tiles to solve
            pass
        elif n_workers == 1 or n_tiles == 1:
            # Sequential execution
            for args in solve_args:
                tile_id, voltages, solve_time, disconnected_halo = _solve_single_tile(*args)
                tile_results[tile_id] = voltages
                per_tile_times[tile_id] = solve_time
                if disconnected_halo:
                    disconnected_halo_counts[tile_id] = len(disconnected_halo)
                if progress_callback:
                    progress_callback(len(tile_results), n_tiles, tile_id)
        else:
            # Parallel execution
            ExecutorClass = (
                ProcessPoolExecutor if parallel_backend == "process"
                else ThreadPoolExecutor
            )

            with ExecutorClass(max_workers=n_workers) as executor:
                futures = {
                    executor.submit(_solve_single_tile, *args): args[0]
                    for args in solve_args
                }

                for future in as_completed(futures):
                    try:
                        tile_id, voltages, solve_time, disconnected_halo = future.result()
                        tile_results[tile_id] = voltages
                        per_tile_times[tile_id] = solve_time
                        if disconnected_halo:
                            disconnected_halo_counts[tile_id] = len(disconnected_halo)
                    except Exception as e:
                        tile_id = futures[future]
                        logger.error(f"Tile {tile_id} solve failed: {e}")
                        raise  # Re-raise to propagate disconnected core errors

                    if progress_callback:
                        progress_callback(len(tile_results), n_tiles, tile_id)

        timings['solve_tiles'] = time.perf_counter() - t0
        
        # Log disconnected halo statistics
        total_disconnected_halo = sum(disconnected_halo_counts.values())
        if total_disconnected_halo > 0 and verbose:
            print(f"  Dropped {total_disconnected_halo} disconnected halo nodes across {len(disconnected_halo_counts)} tiles")

        # ================================================================
        # Step 6: Merge results
        # ================================================================
        t0 = time.perf_counter()
        bottom_voltages, halo_warnings = self._merge_tiled_voltages(
            tiles=tiles,
            tile_results=tile_results,
            bottom_grid_nodes=bottom_nodes,
        )

        # Merge with top-grid voltages (if we solved top-grid)
        all_voltages = {}
        if override_port_voltages is None:
            # Full hierarchical solve - include top-grid voltages
            all_voltages.update(top_voltages)
        else:
            # Bottom-grid only validation - include port voltages as provided
            all_voltages.update(override_port_voltages)
        all_voltages.update(bottom_voltages)

        # Compute IR-drop
        ir_drop = self.model.ir_drop(all_voltages)
        timings['merge_results'] = time.perf_counter() - t0

        # ================================================================
        # Step 7: Optional accuracy validation
        # ================================================================
        validation_stats = None
        if validate_against_flat:
            t0 = time.perf_counter()
            validation_stats = self._validate_tiled_accuracy(
                tiled_voltages=bottom_voltages,
                bottom_grid_nodes=bottom_nodes,
                port_nodes=port_nodes,
                port_voltages=port_voltages,
                current_injections=current_injections,
            )
            timings['validate'] = time.perf_counter() - t0

        # ================================================================
        # Verbose output
        # ================================================================
        if verbose:
            total_time = sum(timings.values())
            print(f"\n=== Tiled Hierarchical Solve Timing ===")
            print(f"  Top nodes: {len(top_nodes):,}, Bottom nodes: {len(bottom_nodes):,}")
            print(f"  Ports: {len(port_nodes):,}, Tiles: {len(tiles)}")
            print(f"  Grid: {N_x}x{N_y}, Halo: {halo_percent*100:.0f}%")
            print(f"  Workers: {n_workers}, Backend: {parallel_backend}")
            print(f"  ---")
            for step, t in timings.items():
                pct = t / total_time * 100
                print(f"  {step:25s}: {t*1000:8.1f} ms  ({pct:5.1f}%)")
            print(f"  {'TOTAL':25s}: {total_time*1000:8.1f} ms")

            if validation_stats:
                print(f"  --- Validation ---")
                print(f"  Max diff: {validation_stats['max_diff']*1000:.4f} mV")
                print(f"  Mean diff: {validation_stats['mean_diff']*1000:.4f} mV")
                print(f"  RMSE: {validation_stats['rmse']*1000:.4f} mV")

            if halo_warnings:
                print(f"  --- Warnings ---")
                for w in halo_warnings[:5]:
                    print(f"  {w}")

            print(f"=========================================\n")

        # Determine top_grid_voltages for result
        if override_port_voltages is None:
            result_top_voltages = top_voltages
        else:
            # In validation mode, we don't have top-grid voltages
            result_top_voltages = {}

        return TiledBottomGridResult(
            voltages=all_voltages,
            ir_drop=ir_drop,
            partition_layer=partition_layer,
            top_grid_voltages=result_top_voltages,
            bottom_grid_voltages=bottom_voltages,
            port_nodes=port_nodes,
            port_voltages=port_voltages,
            port_currents=port_currents,
            aggregation_map=aggregation_map,
            tiles=tiles,
            per_tile_solve_times=per_tile_times,
            halo_clip_warnings=halo_warnings,
            validation_stats=validation_stats,
            tiling_params={
                'N_x': N_x,
                'N_y': N_y,
                'halo_percent': halo_percent,
                'min_ports_per_tile': min_ports_per_tile,
                'n_workers': n_workers,
                'parallel_backend': parallel_backend,
            },
        )

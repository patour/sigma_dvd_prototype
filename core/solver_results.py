"""Data classes for IR-drop solve results and solver contexts.

This module contains result types used by UnifiedIRDropSolver:
- UnifiedSolveResult: Basic solve result
- UnifiedHierarchicalResult: Hierarchical solve result with port information
- TileBounds, BottomGridTile, TileSolveResult: Tiling infrastructure
- TiledBottomGridResult: Tiled hierarchical solve result
- UnifiedCoupledHierarchicalResult: Coupled hierarchical solve result

And context types for efficient batch solving:
- FlatSolverContext: Cached artifacts for flat solving
- HierarchicalSolverContext: Cached artifacts for hierarchical solving
- CoupledHierarchicalSolverContext: Cached artifacts for coupled hierarchical solving
- TiledHierarchicalSolverContext: Cached artifacts for tiled hierarchical solving
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Set, Tuple, TYPE_CHECKING

import numpy as np
import scipy.sparse.linalg as spla

from .unified_model import LayerID

if TYPE_CHECKING:
    from .unified_model import UnifiedReducedSystem
    from .coupled_system import BlockMatrixSystem, SchurComplementOperator, CoupledSystemOperator


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
# Data Classes for Coupled Hierarchical Solving
# ============================================================================

@dataclass
class UnifiedCoupledHierarchicalResult:
    """Result of coupled hierarchical IR-drop solve.

    The coupled solver solves the top-grid + bottom-grid system exactly
    (up to iterative tolerance) using a matrix-free Schur complement approach,
    rather than approximating port currents via weighted distribution.

    Attributes:
        voltages: Complete node -> voltage mapping
        ir_drop: Complete node -> IR-drop mapping (vdd - voltage)
        partition_layer: Layer used for decomposition
        top_grid_voltages: Voltages for top-grid nodes
        bottom_grid_voltages: Voltages for bottom-grid nodes
        port_nodes: Set of port nodes at partition layer
        port_voltages: Port node -> voltage mapping
        iterations: Number of iterative solver iterations
        final_residual: Final residual norm from iterative solver
        converged: True if solver converged within tolerance
        preconditioner_type: Type of preconditioner used ('none', 'block_diagonal', 'ilu')
        timings: Dict of timing information for each step (in seconds)
    """
    voltages: Dict[Any, float]
    ir_drop: Dict[Any, float]
    partition_layer: LayerID
    top_grid_voltages: Dict[Any, float]
    bottom_grid_voltages: Dict[Any, float]
    port_nodes: Set[Any]
    port_voltages: Dict[Any, float]
    iterations: int
    final_residual: float
    converged: bool
    preconditioner_type: str = 'none'
    timings: Dict[str, float] = field(default_factory=dict)


# ============================================================================
# Solver Context Classes for Efficient Batch Solving
# ============================================================================

@dataclass
class FlatSolverContext:
    """Cached artifacts for efficient batch flat solving.

    Holds the pre-computed reduced system with LU factorization, enabling
    fast subsequent solves that only require forward/backward substitution.

    Example usage:
        solver = UnifiedIRDropSolver(model)
        ctx = solver.prepare_flat()
        results = [solver.solve_prepared(stim, ctx) for stim in stimuli]

    Attributes:
        reduced_system: Pre-built reduced system with LU factorization
        vdd: Nominal voltage
        net_name: Optional net name for multi-net support
        pad_nodes: Set of voltage source nodes
    """
    reduced_system: 'UnifiedReducedSystem'
    vdd: float
    net_name: Optional[str]
    pad_nodes: Set[Any]


@dataclass
class HierarchicalSolverContext:
    """Cached artifacts for efficient batch hierarchical solving.

    Holds pre-computed grid decomposition, factored top/bottom systems, and
    shortest-path cache for current aggregation. Subsequent solves only need
    to aggregate currents and perform forward/backward substitution.

    Example usage:
        solver = UnifiedIRDropSolver(model)
        ctx = solver.prepare_hierarchical(partition_layer='M2', top_k=5)
        results = [solver.solve_hierarchical_prepared(stim, ctx) for stim in stimuli]

    Attributes:
        partition_layer: Layer used for grid decomposition
        top_nodes: Set of nodes in top-grid (layers >= partition_layer)
        bottom_nodes: Set of nodes in bottom-grid (layers < partition_layer)
        port_nodes: Set of port nodes at partition boundary
        via_edges: Set of (u, v) edges crossing the partition
        top_system: LU-factored reduced system for top-grid
        bottom_system: LU-factored reduced system for bottom-grid
        top_k: Number of nearest ports for current aggregation
        weighting: Weighting scheme ("shortest_path" or "effective")
        rmax: Maximum resistance distance for shortest_path weighting
        shortest_path_cache: Pre-computed distances from nodes to ports
        pad_nodes: Set of voltage source nodes
        top_grid_pads: Pads in the top-grid
        vdd: Nominal voltage
    """
    partition_layer: LayerID
    top_nodes: Set[Any]
    bottom_nodes: Set[Any]
    port_nodes: Set[Any]
    via_edges: Set[Tuple[Any, Any]]
    top_system: 'UnifiedReducedSystem'
    bottom_system: 'UnifiedReducedSystem'
    top_k: int
    weighting: str
    rmax: Optional[float]
    shortest_path_cache: Dict[Any, List[Tuple[Any, float]]]
    pad_nodes: Set[Any]
    top_grid_pads: Set[Any]
    vdd: float


@dataclass
class CoupledHierarchicalSolverContext:
    """Cached artifacts for efficient batch coupled hierarchical solving.

    Holds pre-computed block matrices, LU factorizations, Schur complement
    operator, coupled system operator, and preconditioner. Subsequent solves
    only need to build RHS and run the iterative solver.

    Example usage:
        solver = UnifiedIRDropSolver(model)
        ctx = solver.prepare_hierarchical_coupled(partition_layer='M2', tol=1e-8)
        results = [solver.solve_hierarchical_coupled_prepared(stim, ctx) for stim in stimuli]

    Attributes:
        partition_layer: Layer used for grid decomposition
        top_nodes: Set of nodes in top-grid
        bottom_nodes: Set of nodes in bottom-grid
        port_nodes: Set of port nodes at partition boundary
        bottom_subgrid: bottom_nodes | port_nodes
        top_blocks: Block matrices for top-grid (with optional LU for interior)
        bottom_blocks: Block matrices for bottom-grid (with lu_ii factored)
        rhs_dirichlet_top: Dirichlet contribution from pads to RHS
        rhs_dirichlet_bottom: Dirichlet contribution for bottom (typically zeros)
        schur_B: Matrix-free Schur complement operator for bottom-grid
        coupled_op: Full coupled system linear operator
        preconditioner: Preconditioner linear operator (or None)
        preconditioner_type: Type of preconditioner used
        solver: Iterative solver name ('gmres' or 'bicgstab')
        tol: Convergence tolerance
        maxiter: Maximum iterations
        vdd: Nominal voltage
        top_grid_pads: Pads in the top-grid
        n_ports: Number of port nodes
        n_top_interior: Number of top-grid interior nodes
    """
    partition_layer: LayerID
    top_nodes: Set[Any]
    bottom_nodes: Set[Any]
    port_nodes: Set[Any]
    bottom_subgrid: Set[Any]
    top_blocks: 'BlockMatrixSystem'
    bottom_blocks: 'BlockMatrixSystem'
    rhs_dirichlet_top: np.ndarray
    rhs_dirichlet_bottom: np.ndarray
    schur_B: 'SchurComplementOperator'
    coupled_op: 'CoupledSystemOperator'
    preconditioner: Optional[spla.LinearOperator]
    preconditioner_type: str
    solver: str
    tol: float
    maxiter: int
    vdd: float
    top_grid_pads: Set[Any]
    n_ports: int
    n_top_interior: int


@dataclass
class TiledHierarchicalSolverContext:
    """Cached artifacts for efficient batch tiled hierarchical solving.

    Holds pre-computed grid decomposition, top-grid system, tile structure,
    and shortest-path cache. Subsequent solves reuse tiles and top-grid LU.

    Note: Individual tile solves still perform their own LU factorization
    per solve, as tile systems change with port voltages. For maximum
    efficiency with identical tile structures, consider caching tile
    systems separately.

    Example usage:
        solver = UnifiedIRDropSolver(model)
        ctx = solver.prepare_hierarchical_tiled(partition_layer='M2', N_x=4, N_y=4)
        results = [solver.solve_hierarchical_tiled_prepared(stim, ctx) for stim in stimuli]

    Attributes:
        partition_layer: Layer used for grid decomposition
        top_nodes: Set of nodes in top-grid
        bottom_nodes: Set of nodes in bottom-grid
        port_nodes: Set of port nodes at partition boundary
        top_system: LU-factored reduced system for top-grid
        top_grid_pads: Pads in the top-grid
        tiles: List of BottomGridTile objects defining tile structure
        bottom_coords: Node -> (x, y) coordinates for bottom-grid nodes
        port_coords: Node -> (x, y) coordinates for port nodes
        grid_bounds: (x_min, x_max, y_min, y_max) of bottom-grid
        top_k: Number of nearest ports for current aggregation
        weighting: Weighting scheme ("shortest_path" or "effective")
        rmax: Maximum resistance distance for shortest_path weighting
        shortest_path_cache: Pre-computed distances from nodes to ports
        N_x: Number of tiles in x direction
        N_y: Number of tiles in y direction
        halo_percent: Halo size as fraction of tile dimensions
        min_ports_per_tile: Minimum port nodes per tile
        n_workers: Number of parallel workers
        parallel_backend: 'thread' or 'process'
        vdd: Nominal voltage
        pad_nodes: Set of voltage source nodes
    """
    partition_layer: LayerID
    top_nodes: Set[Any]
    bottom_nodes: Set[Any]
    port_nodes: Set[Any]
    top_system: 'UnifiedReducedSystem'
    top_grid_pads: Set[Any]
    tiles: List[BottomGridTile]
    bottom_coords: Dict[Any, Tuple[float, float]]
    port_coords: Dict[Any, Tuple[float, float]]
    grid_bounds: Tuple[float, float, float, float]
    top_k: int
    weighting: str
    rmax: Optional[float]
    shortest_path_cache: Dict[Any, List[Tuple[Any, float]]]
    N_x: int
    N_y: int
    halo_percent: float
    min_ports_per_tile: int
    n_workers: int
    parallel_backend: str
    vdd: float
    pad_nodes: Set[Any]

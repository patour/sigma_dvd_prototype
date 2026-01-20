"""Data classes for IR-drop solve results.

This module contains result types used by UnifiedIRDropSolver:
- UnifiedSolveResult: Basic solve result
- UnifiedHierarchicalResult: Hierarchical solve result with port information
- TileBounds, BottomGridTile, TileSolveResult: Tiling infrastructure
- TiledBottomGridResult: Tiled hierarchical solve result
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, NamedTuple, Optional, Set, Tuple

from .unified_model import LayerID


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

"""Result and context dataclasses for distributed DDM solver.

Follows the same pattern as core/solver_results.py for the unified solver.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np


@dataclass
class TileSolveResult:
    """Per-tile solve result. Voltages for one tile (interior + boundary)."""

    tile_id: Tuple[int, int]
    voltages: Dict[str, float]  # All nodes in this tile (interior + boundary)
    n_interior: int
    n_boundary: int


@dataclass
class DistributedSolveResult:
    """Result of distributed DDM solve. Per-tile results stay distributed.

    Interface voltages are stored separately. Per-tile results are accessed
    by tile_id. A flatten() method merges into a global dict when needed
    (e.g., for validation against flat solver). IR-drop is computed on demand.
    """

    tile_results: Dict[Tuple[int, int], TileSolveResult]
    interface_voltages: Dict[str, float]  # Boundary node voltages (from interface solve)
    pad_voltages: Dict[str, float]  # Dirichlet node voltages (= vdd)
    nominal_voltage: float
    net_name: Optional[str] = None
    interface_size: int = 0
    solve_metadata: Dict[str, Any] = field(default_factory=dict)

    def flatten(self) -> Dict[str, float]:
        """Merge all tile voltages + interface + pads into a single global dict."""
        merged = dict(self.pad_voltages)
        merged.update(self.interface_voltages)
        for tr in self.tile_results.values():
            merged.update(tr.voltages)
        return merged

    @property
    def ir_drop(self) -> Dict[str, float]:
        """Compute IR-drop (vdd - voltage) for all nodes. Computed on demand."""
        return {n: self.nominal_voltage - v for n, v in self.flatten().items()}


@dataclass
class DistributedSolverContext:
    """Cached solver artifacts. Analogous to CoupledHierarchicalSolverContext.

    Contains ONLY solver-derived state (factorizations, operators, index maps).
    Model data (workers, metadata, package) lives in DistributedPowerGridModel.
    """

    # Interface system (coordinator-side factorization)
    interface_lu: Callable  # Factored interface matrix .solve()
    interface_nodes: List[str]  # Global boundary node ordering
    interface_node_to_idx: Dict[str, int]
    rhs_dirichlet_interface: np.ndarray  # Package Dirichlet RHS contribution

    # Per-tile index maps (local -> global for scatter-add assembly)
    tile_index_maps: Dict[Tuple[int, int], np.ndarray]

    # Interface island nodes penalized during prepare (shorted to Vdd)
    removed_interface_nodes: Set[str] = field(default_factory=set)

    # Timing breakdown
    timings: Dict[str, float] = field(default_factory=dict)

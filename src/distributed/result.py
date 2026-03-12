"""Result and context dataclasses for distributed DDM solver.

Follows the same pattern as core/solver_results.py for the unified solver.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import scipy.sparse as sp


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


@dataclass
class DistributedSmoothedSources:
    """Lightweight coordinator-side handle for preprocessed current sources.

    Actual VectorizedCurrentSources data lives on workers (or disk as .pkl).
    This handle tracks the preprocessing parameters so the coordinator can
    verify compatibility without touching worker data.
    """

    time_step: float
    t_start: float
    t_end: float
    smoothed: bool
    n_tiles: int
    per_tile_stats: Dict[Tuple[int, int], Dict[str, int]]

    def is_compatible(
        self, time_step: float, t_start: float, t_end: float, tol: float = 1e-12
    ) -> bool:
        """Check if these sources can be reused for the given time parameters.

        Args:
            time_step: Desired simulation time step.
            t_start: Desired simulation start time.
            t_end: Desired simulation end time.
            tol: Absolute tolerance for floating-point comparison.

        Returns:
            True if all three parameters match within tolerance.
        """
        return (
            abs(self.time_step - time_step) <= tol
            and abs(self.t_start - t_start) <= tol
            and abs(self.t_end - t_end) <= tol
        )

    def validate_against_model(self, model: Any) -> None:
        """Verify this handle matches the given DistributedPowerGridModel.

        Args:
            model: A DistributedPowerGridModel instance.

        Raises:
            ValueError: If the number of tiles does not match the model.
        """
        n_model_tiles = len(model.metadata.tile_configs)
        if self.n_tiles != n_model_tiles:
            raise ValueError(
                f"DistributedSmoothedSources has {self.n_tiles} tiles but "
                f"model has {n_model_tiles} tiles"
            )


@dataclass
class DistributedTransientContext:
    """Cached solver artifacts for distributed transient (RC) analysis.

    Extends DistributedSolverContext with time-integration state.
    The dc_context field holds the base DC factorization; this context
    adds capacitance-related matrices and integration parameters.
    """

    # Interface system (coordinator-side factorization for A-based system)
    interface_lu: Callable
    interface_nodes: List[str]
    interface_node_to_idx: Dict[str, int]
    rhs_dirichlet_interface: np.ndarray

    # Per-tile index maps (local -> global for scatter-add assembly)
    tile_index_maps: Dict[Tuple[int, int], np.ndarray]

    # DC context from the base prepare step
    dc_context: DistributedSolverContext

    # Time integration parameters
    dt_scaled: float  # dt in ps (dt = dt_seconds * 1e12)
    integration_method: str  # 'be' or 'trap'
    has_capacitance: bool

    # G-only Dirichlet RHS (without cap contributions), used in transient loop
    rhs_dirichlet_G: Optional[np.ndarray] = None

    # Optional package-level matrices (coordinator-side)
    C_package_uu: Optional[sp.csr_matrix] = None
    G_package_uu: Optional[sp.csr_matrix] = None

    # Interface island nodes penalized during prepare
    removed_interface_nodes: Set[str] = field(default_factory=set)

    # Timing breakdown
    timings: Dict[str, float] = field(default_factory=dict)

    @property
    def dt(self) -> float:
        """Time step in seconds (converted from ps)."""
        return self.dt_scaled * 1e-12

    @property
    def C_coeff(self) -> float:
        """Capacitance coefficient: 1/dt_scaled (BE) or 2/dt_scaled (TR)."""
        return (2.0 if self.integration_method == 'trap' else 1.0) / self.dt_scaled


@dataclass
class DistributedQuasiStaticResult:
    """Result of distributed quasi-static (batch DC) time-domain analysis.

    Peak data stays on workers during the time loop and is NOT eagerly
    collected. Collection happens lazily only when the user calls
    .as_flat(), .as_per_tile(), or .dump().

    Each worker tracks per-node (max_ir_drop, time_of_max) tuples.

    Global peaks and per-step summaries are always available since they
    are tracked via lightweight per-step scalar reductions.
    """

    t_array: np.ndarray
    nominal_voltage: float
    net_name: Optional[str] = None

    # Global peaks (always available -- tracked via per-step scalar max)
    peak_ir_drop: float = 0.0
    peak_ir_drop_time: float = 0.0
    peak_ir_drop_node: Optional[str] = None
    worst_nodes: List[Tuple[str, float, float]] = field(default_factory=list)

    # Per-time-step summaries (always available -- one scalar per step)
    max_ir_drop_per_time: np.ndarray = field(
        default_factory=lambda: np.array([])
    )
    total_current_per_time: np.ndarray = field(
        default_factory=lambda: np.array([])
    )

    # Optional tracked waveforms
    tracked_waveforms: Dict[str, np.ndarray] = field(default_factory=dict)
    tracked_ir_drop: Dict[str, np.ndarray] = field(default_factory=dict)

    solve_metadata: Dict[str, Any] = field(default_factory=dict)

    # --- Lazy peak data (stays on workers until requested) ---
    _model: Optional[Any] = field(default=None, repr=False)
    _peak_collected: bool = field(default=False, repr=False)
    _peak_cache: Optional[
        Dict[Tuple[int, int], Dict[str, Tuple[float, float]]]
    ] = field(default=None, repr=False)

    def _collect_peaks(self) -> None:
        """Lazily collect peak data from workers.

        Raises:
            RuntimeError: If workers are disconnected or result was loaded
                from disk without peaks.
        """
        if self._peak_collected:
            return
        if self._model is None:
            raise RuntimeError(
                "Peak data not available (workers disconnected or result "
                "loaded from disk without peaks)"
            )
        stats = self._model.backend.call_all(
            self._model.workers, 'get_peak_stats'  # TileWorker method added in Phase 4
        )
        tile_configs = self._model.metadata.tile_configs
        self._peak_cache = {
            tc.tile_id: s for tc, s in zip(tile_configs, stats)
        }
        self._peak_collected = True

    def as_per_tile(
        self,
    ) -> Dict[Tuple[int, int], Dict[str, Tuple[float, float]]]:
        """Collect peak data from workers.

        Returns:
            Dict mapping tile_id -> {node: (max_drop, time_of_max)}.
        """
        self._collect_peaks()
        return self._peak_cache  # type: ignore[return-value]

    def as_flat(self) -> Dict[str, Tuple[float, float]]:
        """Collect and flatten peak data.

        For boundary nodes present in multiple tiles, keeps the entry
        with the highest drop.

        Returns:
            Dict mapping node -> (max_drop, time_of_max).
        """
        self._collect_peaks()
        flat: Dict[str, Tuple[float, float]] = {}
        for tile_peaks in self._peak_cache.values():  # type: ignore[union-attr]
            for node, (drop, t) in tile_peaks.items():
                prev = flat.get(node)
                if prev is None or drop > prev[0]:
                    flat[node] = (drop, t)
        return flat

    def dump(self, path: str) -> None:
        """Serialize to disk. Collects peaks from workers first (if available).

        Args:
            path: File path for the pickle output.
        """
        if not self._peak_collected and self._model is not None:
            self._collect_peaks()
        import copy
        snapshot = copy.copy(self)
        snapshot._model = None  # Don't serialize worker references
        with open(path, 'wb') as f:
            pickle.dump(snapshot, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str) -> 'DistributedQuasiStaticResult':
        """Deserialize from disk.

        Note: The loaded result will not have a live model reference,
        so .as_flat() / .as_per_tile() will only work if peaks were
        collected before dump().

        Args:
            path: File path to load from.

        Returns:
            Deserialized DistributedQuasiStaticResult.
        """
        with open(path, 'rb') as f:
            return pickle.load(f)


@dataclass
class DistributedTransientResult(DistributedQuasiStaticResult):
    """Result of distributed transient (RC) time-domain analysis.

    Extends DistributedQuasiStaticResult with transient-specific metadata.
    Inherits all lazy peak collection machinery from the parent class.
    """

    integration_method: str = 'be'
    has_capacitance: bool = False
    total_capacitance_fF: float = 0.0

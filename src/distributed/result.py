"""Result and context classes for distributed DDM solver.

Follows the same pattern as core/solver_results.py for the unified solver.
Context classes are active objects with factor() / release() lifecycle methods.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import scipy.sparse as sp

if TYPE_CHECKING:
    from .model import DistributedPowerGridModel

logger = logging.getLogger(__name__)


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
class DistributedTopologyContext:
    """Topology data shared by DC and transient contexts.

    Computed once during the first factorization (prepare or prepare_transient),
    then reused by subsequent context creations. Intended to be treated as
    read-only after construction. Contains interface node ordering, tile index
    maps, and G-only Dirichlet RHS.
    """

    interface_nodes: List[str]
    interface_node_to_idx: Dict[str, int]
    tile_index_maps: Dict[Tuple[int, int], np.ndarray]
    rhs_dirichlet_G: np.ndarray  # G-only Dirichlet RHS (used by both DC and transient)
    G_package_uu: Optional[sp.csr_matrix]  # Resistive package matrix (unknown-unknown)
    removed_interface_nodes: Set[str] = field(default_factory=set)


class DistributedSolverContext:
    """Active DC solver context. Manages coordinator + worker LU lifecycle.

    Created by DistributedDDMSolver.prepare(). Owns a reference to the model
    and orchestrates factorization on coordinator and workers.

    For backward compatibility, can also be constructed with explicit field
    values (used by tests that don't have a full model).
    """

    def __init__(
        self,
        model: Optional['DistributedPowerGridModel'] = None,
        topology: Optional[DistributedTopologyContext] = None,
        # Backward-compat kwargs (used by tests and direct construction)
        interface_lu: Optional[Callable] = None,
        interface_nodes: Optional[List[str]] = None,
        interface_node_to_idx: Optional[Dict[str, int]] = None,
        rhs_dirichlet_interface: Optional[np.ndarray] = None,
        tile_index_maps: Optional[Dict[Tuple[int, int], np.ndarray]] = None,
        removed_interface_nodes: Optional[Set[str]] = None,
        timings: Optional[Dict[str, Any]] = None,
    ):
        self.model = model
        self.topology: Optional[DistributedTopologyContext] = topology
        self.is_factored: bool = False
        self.timings: Dict[str, Any] = timings if timings is not None else {}
        self._S_global: Optional[sp.csc_matrix] = None  # saved for potential re-factor

        # Direct field storage (populated by factor() or backward-compat constructor)
        self._interface_lu: Optional[Callable] = interface_lu
        self._interface_nodes: Optional[List[str]] = interface_nodes
        self._interface_node_to_idx: Optional[Dict[str, int]] = interface_node_to_idx
        self._rhs_dirichlet_interface: Optional[np.ndarray] = rhs_dirichlet_interface
        self._tile_index_maps: Optional[Dict[Tuple[int, int], np.ndarray]] = tile_index_maps
        self._removed_interface_nodes: Set[str] = (
            removed_interface_nodes if removed_interface_nodes is not None else set()
        )

        # If constructed with explicit fields, mark as factored
        if interface_lu is not None:
            self.is_factored = True

    def factor(self, verbose: bool = False) -> None:
        """Factor tiles + assemble/factor interface system.

        Moves all the logic previously in solver.prepare() into this method.
        After calling, self.is_factored=True and topology is populated.

        Args:
            verbose: Log timing and statistics info.

        Raises:
            RuntimeError: If model is not set.
        """
        from .result_factorization import _factor_dc_context
        _factor_dc_context(self, verbose)

    def release(self) -> None:
        """Free coordinator LU + worker factorizations. Topology preserved."""
        if self._interface_lu is not None:
            self._interface_lu = None
        self._S_global = None
        self.is_factored = False
        # Clear worker DC factorizations if we have a model reference
        if self.model is not None:
            self.model.backend.call_all(
                self.model.workers, 'clear_dc_factorization'
            )

    # --- Checkpoint: save / load / refactor ---

    def save(self, path: Optional[str] = None) -> str:
        """Save coordinator-side metadata to disk.

        Serializes topology, S_global sparse matrix, and timings.
        LU factorizations are NOT saved (not picklable). Call
        ``factor()`` or ``refactor()`` after ``load()`` to rebuild LU
        from the saved S_global.

        Args:
            path: File path for pickle output. If None, uses a default
                checkpoint directory derived from model metadata.

        Returns:
            The absolute path where the checkpoint was saved.
        """
        from .result_factorization import _save_dc_context
        return _save_dc_context(self, path)

    @classmethod
    def load(
        cls,
        model: 'DistributedPowerGridModel',
        path: str,
    ) -> 'DistributedSolverContext':
        """Load coordinator-side metadata from disk.

        Restores topology and S_global. The context is NOT factored
        after load -- call ``refactor()`` to rebuild LU from the saved
        S_global, or ``factor()`` for a full re-factorization that also
        re-runs workers.

        Args:
            model: A live DistributedPowerGridModel with active workers.
            path: Path to saved checkpoint file.

        Returns:
            DistributedSolverContext with restored metadata,
            ``is_factored=False``.

        Raises:
            ValueError: If the checkpoint type does not match.
        """
        from .result_factorization import _load_dc_context
        return _load_dc_context(cls, model, path)

    def refactor(self, verbose: bool = False) -> None:
        """Rebuild coordinator-side LU from saved S_global.

        .. warning::
            This only rebuilds the coordinator interface LU. Workers must
            already have their block systems factored (e.g., from a prior
            ``factor()`` call in this session). If workers are fresh or
            have been released, call ``factor()`` instead for a full
            factorization that includes worker-side setup.

        Args:
            verbose: Log timing info.

        Raises:
            RuntimeError: If S_global is not available (e.g. after
                ``release()`` without a prior ``save()``).
        """
        from .result_factorization import _refactor_dc_context
        _refactor_dc_context(self, verbose)

    def _default_checkpoint_path(self, filename: str) -> str:
        """Derive default checkpoint path from model metadata."""
        from .result_factorization import _default_checkpoint_path
        return _default_checkpoint_path(self, filename)

    # --- Backward-compat property access ---
    # These allow existing code like ctx.interface_nodes, ctx.interface_lu, etc.
    # to work whether the context was populated via factor() or direct construction.

    @property
    def interface_lu(self) -> Optional[Callable]:
        return self._interface_lu

    @interface_lu.setter
    def interface_lu(self, value: Optional[Callable]) -> None:
        self._interface_lu = value

    @property
    def interface_nodes(self) -> List[str]:
        if self._interface_nodes is not None:
            return self._interface_nodes
        if self.topology is not None:
            return self.topology.interface_nodes
        return []

    @property
    def interface_node_to_idx(self) -> Dict[str, int]:
        if self._interface_node_to_idx is not None:
            return self._interface_node_to_idx
        if self.topology is not None:
            return self.topology.interface_node_to_idx
        return {}

    @property
    def rhs_dirichlet_interface(self) -> Optional[np.ndarray]:
        """For DC, this IS the G-only Dirichlet RHS."""
        if self._rhs_dirichlet_interface is not None:
            return self._rhs_dirichlet_interface
        if self.topology is not None:
            return self.topology.rhs_dirichlet_G
        return None

    @property
    def tile_index_maps(self) -> Dict[Tuple[int, int], np.ndarray]:
        if self._tile_index_maps is not None:
            return self._tile_index_maps
        if self.topology is not None:
            return self.topology.tile_index_maps
        return {}

    @property
    def removed_interface_nodes(self) -> Set[str]:
        return self._removed_interface_nodes

    @removed_interface_nodes.setter
    def removed_interface_nodes(self, value: Set[str]) -> None:
        self._removed_interface_nodes = value


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


class DistributedTransientContext:
    """Active transient solver context. Manages coordinator + worker LU lifecycle.

    Created by DistributedDDMSolver.prepare_transient(). Owns a reference to
    the model and orchestrates transient factorization on coordinator and workers.

    For backward compatibility, can also be constructed with explicit field
    values (used by tests that don't have a full model).
    """

    def __init__(
        self,
        model: Optional['DistributedPowerGridModel'] = None,
        topology: Optional[DistributedTopologyContext] = None,
        dt: float = 0.1e-9,
        method: str = 'be',
        # Backward-compat kwargs (used by tests and direct construction)
        interface_lu: Optional[Callable] = None,
        interface_nodes: Optional[List[str]] = None,
        interface_node_to_idx: Optional[Dict[str, int]] = None,
        rhs_dirichlet_interface: Optional[np.ndarray] = None,
        tile_index_maps: Optional[Dict[Tuple[int, int], np.ndarray]] = None,
        dt_scaled: Optional[float] = None,
        integration_method: Optional[str] = None,
        has_capacitance: bool = False,
        rhs_dirichlet_G: Optional[np.ndarray] = None,
        C_package_uu: Optional[sp.csr_matrix] = None,
        G_package_uu: Optional[sp.csr_matrix] = None,
        removed_interface_nodes: Optional[Set[str]] = None,
        timings: Optional[Dict[str, Any]] = None,
    ):
        self.model = model
        self.topology: Optional[DistributedTopologyContext] = topology
        self.is_factored: bool = False
        self.timings: Dict[str, Any] = timings if timings is not None else {}

        # Time integration parameters: prefer explicit dt_scaled/integration_method
        # if provided (backward compat), otherwise derive from dt/method
        if dt_scaled is not None:
            self.dt_scaled: float = dt_scaled
        else:
            self.dt_scaled = dt * 1e12
        if integration_method is not None:
            self.integration_method: str = integration_method
        else:
            self.integration_method = method
        self.has_capacitance: bool = has_capacitance

        # Transient-specific coordinator state
        self.rhs_dirichlet_A: Optional[np.ndarray] = rhs_dirichlet_interface
        self._rhs_dirichlet_G: Optional[np.ndarray] = rhs_dirichlet_G
        self.C_package_uu: Optional[sp.csr_matrix] = C_package_uu
        self._G_package_uu: Optional[sp.csr_matrix] = G_package_uu

        self._S_global: Optional[sp.csc_matrix] = None

        # Direct field storage (populated by factor() or backward-compat constructor)
        self._interface_lu: Optional[Callable] = interface_lu
        self._interface_nodes: Optional[List[str]] = interface_nodes
        self._interface_node_to_idx: Optional[Dict[str, int]] = interface_node_to_idx
        self._tile_index_maps: Optional[Dict[Tuple[int, int], np.ndarray]] = tile_index_maps
        self._removed_interface_nodes: Set[str] = (
            removed_interface_nodes if removed_interface_nodes is not None else set()
        )

        # If constructed with explicit fields, mark as factored
        if interface_lu is not None:
            self.is_factored = True

    def factor(self, verbose: bool = False) -> None:
        """Factor transient A-system on tiles and assemble interface.

        Builds A = G + C_coeff * C on each tile, computes Schur complements,
        and assembles the global transient interface system including package
        capacitance contributions.

        Args:
            verbose: Log timing info.

        Raises:
            RuntimeError: If model is not set.
        """
        from .result_factorization import _factor_transient_context
        _factor_transient_context(self, verbose)

    def release(self) -> None:
        """Free coordinator LU + worker transient factorizations."""
        if self._interface_lu is not None:
            self._interface_lu = None
        self._S_global = None
        self.rhs_dirichlet_A = None
        self._rhs_dirichlet_G = None
        self.C_package_uu = None
        self._G_package_uu = None
        self.is_factored = False
        # Clear worker transient factorizations if we have a model reference
        if self.model is not None:
            self.model.backend.call_all(
                self.model.workers, 'clear_transient_factorization'
            )

    # --- Checkpoint: save / load / refactor ---

    def save(self, path: Optional[str] = None) -> str:
        """Save coordinator-side transient metadata to disk.

        Serializes topology, S_global sparse matrix, integration params,
        transient-specific matrices, and timings.  LU factorizations are
        NOT saved (not picklable). Call ``refactor()`` after ``load()``
        to rebuild LU from the saved S_global.

        Args:
            path: File path for pickle output. If None, uses a default
                checkpoint directory derived from model metadata.

        Returns:
            The absolute path where the checkpoint was saved.
        """
        from .result_factorization import _save_transient_context
        return _save_transient_context(self, path)

    @classmethod
    def load(
        cls,
        model: 'DistributedPowerGridModel',
        path: str,
    ) -> 'DistributedTransientContext':
        """Load coordinator-side transient metadata from disk.

        Restores topology, S_global, integration params, and transient
        matrices. The context is NOT factored after load -- call
        ``refactor()`` to rebuild LU from the saved S_global, or
        ``factor()`` for a full re-factorization.

        Args:
            model: A live DistributedPowerGridModel with active workers.
            path: Path to saved checkpoint file.

        Returns:
            DistributedTransientContext with restored metadata,
            ``is_factored=False``.

        Raises:
            ValueError: If the checkpoint type does not match.
        """
        from .result_factorization import _load_transient_context
        return _load_transient_context(cls, model, path)

    def refactor(self, verbose: bool = False) -> None:
        """Rebuild coordinator-side LU from saved S_global.

        .. warning::
            This only rebuilds the coordinator interface LU. Workers must
            already have their transient block systems factored (e.g.,
            from a prior ``factor()`` call in this session). If workers
            are fresh or have been released, call ``factor()`` instead
            for a full factorization that includes worker-side setup.

        Args:
            verbose: Log timing info.

        Raises:
            RuntimeError: If S_global is not available.
        """
        from .result_factorization import _refactor_transient_context
        _refactor_transient_context(self, verbose)

    def _default_checkpoint_path(self, filename: str) -> str:
        """Derive default checkpoint path from model metadata."""
        from .result_factorization import _default_checkpoint_path
        return _default_checkpoint_path(self, filename)

    # --- Computed properties ---

    @property
    def dt(self) -> float:
        """Time step in seconds (converted from ps)."""
        return self.dt_scaled * 1e-12

    @property
    def C_coeff(self) -> float:
        """Capacitance coefficient: 1/dt_scaled (BE) or 2/dt_scaled (TR)."""
        return (2.0 if self.integration_method == 'trap' else 1.0) / self.dt_scaled

    # --- Backward-compat property access ---

    @property
    def interface_lu(self) -> Optional[Callable]:
        return self._interface_lu

    @interface_lu.setter
    def interface_lu(self, value: Optional[Callable]) -> None:
        self._interface_lu = value

    @property
    def interface_nodes(self) -> List[str]:
        if self._interface_nodes is not None:
            return self._interface_nodes
        if self.topology is not None:
            return self.topology.interface_nodes
        return []

    @property
    def interface_node_to_idx(self) -> Dict[str, int]:
        if self._interface_node_to_idx is not None:
            return self._interface_node_to_idx
        if self.topology is not None:
            return self.topology.interface_node_to_idx
        return {}

    @property
    def rhs_dirichlet_interface(self) -> Optional[np.ndarray]:
        """A-based Dirichlet RHS (includes cap contributions)."""
        return self.rhs_dirichlet_A

    @property
    def rhs_dirichlet_G(self) -> Optional[np.ndarray]:
        """G-only Dirichlet RHS (without cap contributions)."""
        if self._rhs_dirichlet_G is not None:
            return self._rhs_dirichlet_G
        if self.topology is not None:
            return self.topology.rhs_dirichlet_G
        return None

    @rhs_dirichlet_G.setter
    def rhs_dirichlet_G(self, value: Optional[np.ndarray]) -> None:
        self._rhs_dirichlet_G = value

    @property
    def tile_index_maps(self) -> Dict[Tuple[int, int], np.ndarray]:
        if self._tile_index_maps is not None:
            return self._tile_index_maps
        if self.topology is not None:
            return self.topology.tile_index_maps
        return {}

    @property
    def removed_interface_nodes(self) -> Set[str]:
        return self._removed_interface_nodes

    @removed_interface_nodes.setter
    def removed_interface_nodes(self, value: Set[str]) -> None:
        self._removed_interface_nodes = value

    @property
    def G_package_uu(self) -> Optional[sp.csr_matrix]:
        return self._G_package_uu

    @G_package_uu.setter
    def G_package_uu(self, value: Optional[sp.csr_matrix]) -> None:
        self._G_package_uu = value


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

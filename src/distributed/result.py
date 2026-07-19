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

    tile_id: tuple  # 2-tuple (x, y) for original tiles; 3-tuple (x, y, k) for sub-tiles
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

    tile_results: Dict[tuple, TileSolveResult]
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
    tile_index_maps: Dict[tuple, np.ndarray]
    rhs_dirichlet_G: np.ndarray  # G-only Dirichlet RHS (used by both DC and transient)
    G_package_uu: Optional[sp.csr_matrix]  # Resistive package matrix (unknown-unknown)
    removed_interface_nodes: Set[str] = field(default_factory=set)
    island_nodes: Optional[Set[str]] = field(default=None)
    """Cached DC-mode island detection result (resistive-only extra_edges).

    ``None``  — not yet computed (or loaded from a pre-A7 checkpoint).
    ``set()`` — computed; no islands found.
    ``{...}`` — computed; these nodes are islands.

    Populated on the first DC prepare call.  When ``package_cap_edges``
    is empty, ``island_nodes_td`` is also set to the same value on that
    same call (identical BFS inputs → identical result).

    Persisted through context save/load so that a load()+refactor() cycle
    does not recompute the BFS.  Old-format checkpoints (field absent)
    degrade gracefully: the attribute will be missing from the unpickled
    instance, and all access uses ``getattr(topology, 'island_nodes', None)``
    to fall back to recomputation.

    .. warning::
        Cross-mode reuse (using this DC result for transient) is only safe
        when ``model.package_data.package_cap_edges`` is empty.  When cap
        edges are present, cap connectivity can bridge components that are
        resistively-disconnected, making the transient island set a strict
        subset of the DC island set.  Using the DC set for transient would
        over-penalise those cap-bridged nodes.  Use ``island_nodes_td`` for
        transient-mode cache lookups; ``_factor_transient_context`` enforces
        this guard automatically.
    """

    island_nodes_td: Optional[Set[str]] = field(default=None)
    """Cached transient-mode island detection result (resistive + cap extra_edges).

    ``None``  — not yet computed (or not applicable — e.g. old checkpoint).
    ``set()`` — computed; no islands found.
    ``{...}`` — computed; these nodes are islands.

    Populated on the first transient prepare call.  When ``package_cap_edges``
    is empty (so ``combined_edges == resistive_edges``), DC and transient BFS
    produce identical island sets and both ``island_nodes`` and
    ``island_nodes_td`` are set to the same value; the transient path can then
    reuse the DC result without running a second BFS, and vice-versa.

    When ``package_cap_edges`` is non-empty this field is populated
    independently — the transient BFS uses ``combined_edges`` (resistive
    + weighted cap), which may have fewer islands than the DC BFS.
    Cross-mode reuse is suppressed by ``_factor_transient_context``.
    """

    tile_kept_port_pos: Dict[tuple, np.ndarray] = field(default_factory=dict)
    """Stage 2 (D1 fix, item 1): per-tile POSITIONS (int64) within the tile's
    full port-order arrays (``get_reduced_rhs()``'s return value,
    ``bs.port_nodes``) that are kept (non-Dirichlet) -- i.e. the positions
    that were sliced INTO ``S_i`` (via ``kept_position_slice``) and into
    ``tile_index_maps[tid]``.  Consumed by ``solve_dc``'s RHS scatter
    (``solver.py``) to filter a full-port-length reduced-RHS vector down to
    the same length as ``tile_index_maps[tid]`` for tiles with a Dirichlet
    pad among their own ports (the D1 RHS-scatter manifestation).  Absent
    (empty dict) on pre-Stage-2 checkpoints -- callers must use
    ``getattr(topology, 'tile_kept_port_pos', {})`` and fall back to
    unfiltered behaviour when a tile id is missing.
    """

    _assembly_cache: Optional[Dict[str, Any]] = field(default=None)
    """A4: Mutable assembly-pattern cache for ``assemble_schur_complement_system``.

    Stores ``full_node_to_idx``, per-tile ``local_to_global`` index arrays,
    and dimension bookkeeping so that the second ``assemble_schur_complement_system``
    call (transient prepare) can skip per-port dict lookups.

    This field is intentionally NOT included in ``save()`` / ``load()``
    checkpoints — it can be rebuilt cheaply and the Factor objects it
    references are not picklable across processes anyway.

    Initialised lazily as an empty ``{}`` on the first ``_factor_dc_context``
    or ``_factor_transient_context`` call; ``None`` means "not yet
    initialised".
    """

    tile_port_count: Dict[tuple, int] = field(default_factory=dict)
    """S2/S13: per-tile FULL port count (``len(tile's port node list)`` --
    including any Dirichlet/pad ports -- as recorded at the same
    ``kept_position_slice``/streaming-gather call that produced
    ``tile_kept_port_pos[tid]``), i.e. the expected length of a full-port-
    order reduced-RHS vector (``get_reduced_rhs()``/``evaluate_and_get_
    reduced_rhs()``/adjoint terminal-RHS return values) for that tile.

    Used by ``interface_iterative.filter_kept_rhs`` to VALIDATE a live
    worker's returned vector length against what was recorded when
    ``tile_kept_port_pos``/``tile_index_maps`` were built, before slicing
    with ``kept_pos`` -- catching a drifted/stale context (e.g. worker
    re-setup or retile between factor() and a scatter call) with a loud
    error instead of ``g_i[kept_pos]`` silently indexing into the wrong
    positions whenever the lengths happen to coincide.  Empty dict on
    pre-Stage-2 checkpoints -- callers must fall back to the length-only
    fast-path check when a tile id is missing.
    """


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
        tile_index_maps: Optional[Dict[tuple, np.ndarray]] = None,
        removed_interface_nodes: Optional[Set[str]] = None,
        timings: Optional[Dict[str, Any]] = None,
    ):
        self.model = model
        self.topology: Optional[DistributedTopologyContext] = topology
        self.is_factored: bool = False
        self.timings: Dict[str, Any] = timings if timings is not None else {}
        self._S_global: Optional[sp.csc_matrix] = None  # saved for potential re-factor
        # B2: CG solver handle (InterfaceCGSolver or None); set by factor(), cleared by release().
        self._cg_solver = None
        # Item 3: True iff the most recent factor() used the "never assemble
        # S_global" path (interface_drop_s_global).  save() checks this and
        # raises with guidance instead of silently saving nothing useful.
        self._s_global_dropped: bool = False

        # Direct field storage (populated by factor() or backward-compat constructor)
        self._interface_lu: Optional[Callable] = interface_lu
        self._interface_nodes: Optional[List[str]] = interface_nodes
        self._interface_node_to_idx: Optional[Dict[str, int]] = interface_node_to_idx
        self._rhs_dirichlet_interface: Optional[np.ndarray] = rhs_dirichlet_interface
        self._tile_index_maps: Optional[Dict[tuple, np.ndarray]] = tile_index_maps
        # Stage 2 (D1 fix, item 1): always freshly overwritten by factor()
        # (mirrors _tile_index_maps -- the CONTEXT's own last-factor() value
        # wins over the shared topology's, which is only set at first
        # construction).  Not part of the backward-compat constructor kwargs
        # (no pre-Stage-2 caller could have passed it).
        self._tile_kept_port_pos: Dict[tuple, np.ndarray] = {}
        # S2/S13: parity with _tile_kept_port_pos -- see
        # DistributedTopologyContext.tile_port_count.
        self._tile_port_count: Dict[tuple, int] = {}
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
        # B2: CG solver holds S_global + preconditioner factors — free it too
        # so that release() actually reclaims memory in CG mode.
        # Stage 2 lifecycle-parity checklist: close its persistent matvec/
        # BJ-apply thread pool BEFORE dropping the reference, or repeated
        # prepare()/release() cycles accumulate live thread pools (a
        # weakref.finalize safety net exists too, but is not the primary
        # release path -- GC timing is not guaranteed).
        if self._cg_solver is not None:
            _close = getattr(self._cg_solver, 'close', None)
            if callable(_close):
                _close()
        self._cg_solver = None
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
    def tile_index_maps(self) -> Dict[tuple, np.ndarray]:
        if self._tile_index_maps is not None:
            return self._tile_index_maps
        if self.topology is not None:
            return self.topology.tile_index_maps
        return {}

    @property
    def tile_kept_port_pos(self) -> Dict[tuple, np.ndarray]:
        """Stage 2 (D1 fix): see ``DistributedTopologyContext.tile_kept_port_pos``.

        Prefers the context's own last-``factor()`` value; falls back to the
        (possibly stale, from a different mode's first construction) value
        cached on the shared topology, then to ``{}`` for pre-Stage-2
        checkpoints/backward-compat direct construction.
        """
        if self._tile_kept_port_pos:
            return self._tile_kept_port_pos
        if self.topology is not None:
            return getattr(self.topology, 'tile_kept_port_pos', {})
        return {}

    @property
    def tile_port_count(self) -> Dict[tuple, int]:
        """S2/S13: see ``DistributedTopologyContext.tile_port_count``.

        Same precedence as ``tile_kept_port_pos``: context's own last-
        ``factor()`` value, then the shared topology's, then ``{}``.
        """
        if self._tile_port_count:
            return self._tile_port_count
        if self.topology is not None:
            return getattr(self.topology, 'tile_port_count', {})
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
    per_tile_stats: Dict[tuple, Dict[str, int]]
    timings: Dict[str, float] = field(default_factory=dict)
    """Internal phase timings populated by ``preprocess_sources()`` in ``solver_td.py``.

    Keys (when present):

    - ``init_vcs``: wall time for parallel VCS init across all workers.
    - ``smooth_sources``: wall time for parallel PWL smoothing (only if
      ``smooth=True`` was passed to ``preprocess_sources``).

    This field was added to allow benchmark/regression scripts (see
    ``scripts/benchmark/run_perf_baseline.py``) to report VCS init and
    smoothing as separate phases, matching the spec's JSON timing schema.
    It is additive — absent or empty has no effect on solve correctness.
    """

    phase_table_info: Optional[Dict[str, Any]] = field(default=None)
    """Per-tile step-column table metadata from A2 precompute_step_columns.

    Set after ``precompute_step_columns`` is called on all workers.  Each
    entry is the info dict returned by the worker (tier, m, phase0,
    n_src_nodes, memory_mb, build_path, build_time_s).  None = table not
    built or use_step_columns=False.
    """

    per_tile_smooth_time_s: Dict[tuple, float] = field(default_factory=dict)
    """A5: Per-tile PWL-smoothing wall-clock time in seconds.

    Keys are tile_ids ``(x, y)``.  Populated only when ``smooth=True`` or
    ``smooth='auto'`` with the auto-decision to smooth.  Zero for tiles
    that loaded from the smoothed-VCS disk cache.  Empty dict when
    smoothing was skipped.
    """

    smooth_cache_hits: int = 0
    """A5: Number of tiles that loaded their smoothed VCS from disk cache.

    Populated only when ``smooth=True`` or ``smooth='auto'`` with
    smoothing enabled.  Zero when caching is disabled or all tiles had to
    recompute.
    """

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
        tile_index_maps: Optional[Dict[tuple, np.ndarray]] = None,
        dt_scaled: Optional[float] = None,
        integration_method: Optional[str] = None,
        has_capacitance: bool = False,
        rhs_dirichlet_G: Optional[np.ndarray] = None,
        C_package_uu: Optional[sp.csr_matrix] = None,
        G_package_uu: Optional[sp.csr_matrix] = None,
        removed_interface_nodes: Optional[Set[str]] = None,
        timings: Optional[Dict[str, Any]] = None,
        island_penalty_rhs: Optional[np.ndarray] = None,
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
        # T1 fix: separate per-step island-penalty RHS vector (penalty *
        # vdd at penalized interface-island rows, zero elsewhere). Must be
        # added EXACTLY ONCE per time step (both BE and TR) -- NOT folded
        # into rhs_dirichlet_G, which TR scales by 2 (that would double the
        # penalty forcing term under TR). None when there are no islands
        # (or on pre-fix checkpoints -- solve_transient degrades gracefully
        # to the old omission for those, same as before this fix).
        self.island_penalty_rhs: Optional[np.ndarray] = island_penalty_rhs

        self._S_global: Optional[sp.csc_matrix] = None
        # B2: CG solver handle (InterfaceCGSolver or None); set by factor(), cleared by release().
        self._cg_solver = None
        # Item 3 (TD extension): True iff the most recent factor() used the
        # "never assemble S_global" path (interface_drop_s_global). save()
        # checks this and raises with guidance instead of silently saving
        # nothing useful. Mirrors DistributedSolverContext's identical field.
        self._s_global_dropped: bool = False

        # Direct field storage (populated by factor() or backward-compat constructor)
        self._interface_lu: Optional[Callable] = interface_lu
        self._interface_nodes: Optional[List[str]] = interface_nodes
        self._interface_node_to_idx: Optional[Dict[str, int]] = interface_node_to_idx
        self._tile_index_maps: Optional[Dict[tuple, np.ndarray]] = tile_index_maps
        # Stage 2 (D1 fix): parity with DistributedSolverContext; see the
        # matching comment there.  Consumed by solve_quasi_static/
        # solve_transient's D1 RHS-scatter filtering (S2 fix).
        self._tile_kept_port_pos: Dict[tuple, np.ndarray] = {}
        # S2/S13: parity with _tile_kept_port_pos -- see
        # DistributedTopologyContext.tile_port_count.
        self._tile_port_count: Dict[tuple, int] = {}
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
        # B2: CG solver holds S_global + preconditioner factors — free it too
        # so that release() actually reclaims memory in CG mode.
        # Stage 2 lifecycle-parity checklist: see the matching comment in
        # DistributedSolverContext.release().
        if self._cg_solver is not None:
            _close = getattr(self._cg_solver, 'close', None)
            if callable(_close):
                _close()
        self._cg_solver = None
        self.rhs_dirichlet_A = None
        self._rhs_dirichlet_G = None
        self.island_penalty_rhs = None
        self.C_package_uu = None
        self._G_package_uu = None
        self.is_factored = False
        # Clear worker transient factorizations if we have a model reference
        if self.model is not None:
            self.model.backend.call_all(
                self.model.workers, 'clear_transient_factorization'
            )
            # Finding 4 (round 2 review): this call_all just cleared EVERY
            # worker's transient factorization unconditionally (regardless
            # of which context's dt/method the model's stamp currently
            # names) -- clear the stamp alongside it, or a subsequent
            # solve_transient()/analyze_adjoint() on some OTHER live
            # context would read a stamp claiming the workers are still
            # factored for a (dt, method) they no longer hold at all,
            # producing a misleading "currently factored for dt=..."
            # diagnostic instead of the accurate "state unknown/absent"
            # one (see _check_worker_transient_stamp in solver_td.py --
            # None is what makes that context raise loudly here instead of
            # either passing silently or dying later with a confusing
            # worker-side "requires factor_transient_system()" error).
            self.model._worker_transient_factor_key = None

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
        """Rebuild the coordinator-side interface solve.

        Two supported contracts, selected automatically by how this
        context was last ``factor()``'d (``ctx._s_global_dropped``):

        **Assembled contract** (``factor()`` ran with
        ``interface_drop_s_global`` off/unmet, or a checkpoint was
        ``load()``'d): ``ctx._S_global`` must be present. Workers must
        already have their transient block systems factored (e.g. from a
        prior ``factor()`` call in this session, or a subsequent
        ``factor()`` on workers after ``load()``) if
        ``interface_matvec_mode`` resolves to ``'tilewise'``; the
        ``'assembled'`` matvec mode needs no worker involvement at all. If
        workers are fresh or have been released and tilewise is needed,
        call ``factor()`` instead for a full factorization.

        **Never-assembled contract** (``factor()`` ran with
        ``interface_drop_s_global=True`` and its preconditions were met):
        ``ctx._S_global`` is never assembled at all (by design), so there
        is no saved-S_global fallback -- ``save()`` refuses to persist
        such a context (see its error message). The ONLY supported
        recovery path is ``release()`` -> ``refactor()`` with workers
        still attached (``ctx.model.workers`` non-empty): this
        unconditionally re-runs ``factor_transient_system(dt, method)`` on
        every attached worker (a streaming re-gather of the dense
        transient Schur blocks, NOT a cheap coordinator-only rebuild) and
        rebuilds ``S_extra``/Dirichlet vectors/island penalty from
        scratch. If workers are detached, this cannot be recovered by
        ``refactor()`` at all -- only a fresh ``prepare_transient()`` /
        ``factor()`` with live, attached workers can rebuild it.

        .. warning::
            **Cross-context worker clobbering**: because workers are
            shared, single-owner state for the whole model, a
            never-assembled ``refactor()`` call re-factors ALL workers at
            *this* context's ``(dt_scaled, integration_method)`` --
            silently invalidating the worker-side transient factorization
            any OTHER live ``DistributedTransientContext`` (prepared at a
            different dt/method on the same model) depends on. A
            subsequent ``solve_transient()`` on that other context now
            raises a loud ``RuntimeError`` naming the dt/method mismatch
            (see ``solve_transient``'s dt-stamp guard in ``solver_td.py``)
            instead of silently mixing mismatched coordinator/worker
            state. This is a deliberate behavior change: the same hazard
            exists for the assembled contract's implicit-tilewise 'auto'
            resolution (see the WARNING logged in that branch), but there
            it degrades an existing footgun into a documented one rather
            than converting it into a hard error.

        Args:
            verbose: Log timing info.

        Raises:
            RuntimeError: If S_global is not available (assembled
                contract) and this is not a never-assembled context with
                workers still attached; or if refactoring to
                ``interface_solver='direct'`` is requested for a
                never-assembled context (impossible -- no S_global was
                ever built).
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
    def tile_index_maps(self) -> Dict[tuple, np.ndarray]:
        if self._tile_index_maps is not None:
            return self._tile_index_maps
        if self.topology is not None:
            return self.topology.tile_index_maps
        return {}

    @property
    def tile_kept_port_pos(self) -> Dict[tuple, np.ndarray]:
        """Stage 2 (D1 fix): see ``DistributedTopologyContext.tile_kept_port_pos``.

        Same precedence as the DC context's property: context's own last-
        ``factor()`` value, then the shared topology's, then ``{}``.
        Consumed by ``solve_quasi_static``/``solve_transient``'s D1 RHS-
        scatter filtering (S2 fix) via ``interface_iterative.filter_kept_rhs``.
        """
        if self._tile_kept_port_pos:
            return self._tile_kept_port_pos
        if self.topology is not None:
            return getattr(self.topology, 'tile_kept_port_pos', {})
        return {}

    @property
    def tile_port_count(self) -> Dict[tuple, int]:
        """S2/S13: see ``DistributedTopologyContext.tile_port_count``."""
        if self._tile_port_count:
            return self._tile_port_count
        if self.topology is not None:
            return getattr(self.topology, 'tile_port_count', {})
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
        Dict[tuple, Dict[str, Tuple[float, float]]]
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
    ) -> Dict[tuple, Dict[str, Tuple[float, float]]]:
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

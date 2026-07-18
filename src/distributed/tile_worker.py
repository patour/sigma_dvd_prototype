"""Tile worker for distributed DDM solver.

Thin stateful actor that holds per-tile BlockMatrixSystem and delegates
ALL math to pgmath (block_system, schur, factor) building blocks.

Parsing helpers (TileData, _parse_tile_ckt, etc.) live in tile_parsing.py
and are re-exported here for backward compatibility.

Time-domain methods (VCS, smoothing, transient, peak tracking) live in
tile_worker_td.py and are mixed in via _TimeDomainMixin.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Re-exports from tile_parsing for backward compatibility.
# All existing ``from distributed.tile_worker import X`` continue to work.
# ---------------------------------------------------------------------------

from .tile_parsing import (  # noqa: F401
    TileData,
    R_TO_KOHM,
    C_TO_FF,
    I_TO_MA,
    _is_gzip_file,
    _load_nd_file,
    _parse_tile_ckt,
    _iter_instance_sources,
    _parse_instance_models,
    _iter_instance_capacitors,
    _parse_instance_capacitors,
    parse_tile_with_instances,
    parse_and_dump_tile,
)

from .tile_worker_td import _TimeDomainMixin
from .tile_worker_adjoint import _AdjointWorkerMixin


class TileWorker(_AdjointWorkerMixin, _TimeDomainMixin):
    """Per-tile actor for distributed DDM. Thin wrapper that delegates
    all math to coupled_system.py building blocks.

    Time-domain methods (init_vectorized_sources, smooth_sources,
    evaluate_and_get_reduced_rhs, factor_transient_system,
    get_transient_reduced_rhs, get_transient_interior_voltages,
    set_initial_voltages, init_peak_tracking, update_peak_stats,
    get_peak_stats, get_tracked_waveforms, use_smoothed_sources) are
    provided by _TimeDomainMixin in tile_worker_td.py.

    Adjoint sensitivity methods (has_node, init_adjoint_source_mappings,
    filter_adjoint_sources, reset_adjoint_state) are provided by
    _AdjointWorkerMixin in tile_worker_adjoint.py.
    """

    def __init__(self):
        self._tile_data: Optional[TileData] = None
        self._block_system = None
        self._rhs_dirichlet = None
        self._interface_nodes: Optional[Set[str]] = None
        self._removed_island_nodes: Set[str] = set()
        # Stage 1e: set via configure({'pre_cleaned_full_trusted': ...}) from
        # the SAME model-creation-time decision the coordinator resolved
        # (model._resolve_island_detection).  _build_block_system() skips
        # _remove_floating_islands entirely only when this is True AND the
        # tile's own TileData.pre_cleaned_full is True.
        self._trust_pre_cleaned_full: bool = False

        # --- Time-domain state (Phase 4) ---
        # VectorizedCurrentSources (raw and smoothed)
        self._vec_sources = None
        self._smoothed_sources = None
        self._active_sources = None  # Points to _vec_sources or _smoothed_sources

        # Transient system (A = G + C_coeff * C)
        self._transient_block_system = None
        self._c_pp_diag: Optional[np.ndarray] = None
        self._c_ii_diag: Optional[np.ndarray] = None
        self._total_cap: float = 0.0
        self._C_coeff: float = 0.0
        self._dt_scaled: float = 0.0
        self._transient_method: str = 'be'
        # Transient time-stepping state
        self._v_interior_old: Optional[np.ndarray] = None
        self._last_f_i: Optional[np.ndarray] = None
        # Quasi-static: cached interior RHS from last evaluate_and_get_reduced_rhs
        self._last_qs_rhs_i: Optional[np.ndarray] = None
        # Per-node current mask (float64): 1.0 = keep, 0.0 = zero out
        self._current_node_mask: Optional[np.ndarray] = None

        # Peak tracking (dict-based for QS, array-based for transient)
        self._peak_per_node: Dict[str, Tuple[float, float]] = {}
        self._peak_vdd: float = 0.0
        self._peak_tracking_active: bool = False
        self._tracked_nodes: Optional[Set[str]] = None
        self._tracked_waveforms: Dict[str, List[float]] = {}
        # Vectorized peak arrays (init_peak_tracking sets these)
        self._peak_drops_array: Optional[np.ndarray] = None
        self._peak_times_array: Optional[np.ndarray] = None
        self._peak_use_arrays: bool = False
        self._tracked_node_indices: Dict[str, int] = {}

        # --- A2: Phase-folded step-column table (tile_worker_td.py) --------
        # _step_col_table: the ACTIVE table used by _get_current_array_for_step.
        # None means per-step evaluate_at_time fallback is used.
        # F6: separate from the CACHED slot so a Change-C skip or a
        # use_step_columns=False path can deactivate the table without
        # evicting a valid cached table built for a different call (e.g. the
        # phase table from the main transient dt).
        self._step_col_table: Optional[Dict] = None
        # _step_col_cached_table: the most recently BUILT table (may differ
        # from _step_col_table when the active table was deactivated by a
        # Change-C skip or use_step_columns=False path).  precompute reuse
        # check reads this slot, not _step_col_table.  Both slots point to the
        # SAME dict object after a successful build; on a reuse hit the active
        # slot is updated to the cached slot.
        self._step_col_cached_table: Optional[Dict] = None
        # Settings (propagated via configure() from coordinator settings).
        self._use_step_columns: bool = True
        self._max_table_mb: float = 512.0
        # Pre-allocated current buffer for buffer-reuse per step (n_nodes,).
        # Allocated lazily on first use.
        self._current_buf: Optional[np.ndarray] = None

        # --- A2 Change A: Step-column table cross-transient reuse cache -----
        # _sources_version is bumped on every source-mutating call so that
        # the cache key is invalidated automatically without having to inspect
        # source identity.
        self._sources_version: int = 0
        # Tuple key describing the CACHED slot (_step_col_cached_table).
        # None = no valid cached table.
        self._step_col_cache_key: Optional[tuple] = None
        # The info dict returned by the most recent build (stored so callers
        # can retrieve it without a round-trip on a cache hit).
        self._step_col_info: Optional[Dict] = None
        # F1/F5: memoised per-row grid-alignment probe result.
        # Keyed on (sources_version, dt); cleared by _invalidate_step_columns().
        # None = no cached result yet.
        self._grid_alignment_cache = None

        # --- F7: Smoothed-VCS disk-cache identity tracking ------------------
        # The hash string embedded in the smoothed-VCS cache filename for the
        # currently loaded smoothed sources.  Set on both compute and disk-hit
        # paths in smooth_sources().  When smooth_sources() is called again with
        # identical params (same hash) AND _active_sources is already the
        # smoothed sources, we short-circuit and return cached stats WITHOUT
        # reloading or bumping _sources_version.
        self._smoothed_cache_hash: Optional[str] = None

        # --- A4: Symbolic-reuse cache for factor_and_compute_schur ----------
        # Holds {'P_ii', 'factor', 'ref_indptr', 'ref_indices'} from the
        # most recent _compute_schur_partial call.  Passed to subsequent
        # calls (DC→transient or transient→DC) so the CHOLMOD AMD analysis
        # is skipped and only numeric values are updated.
        # None = not yet initialised; enabled by default when CHOLMOD active.
        self._symbolic_ii: Optional[Dict] = None
        self._use_symbolic_reuse: bool = True

        # --- B3: Worker-side Schur cache for streaming assembly --------------
        # Set by factor_and_cache_schur(); consumed and cleared by
        # get_schur_coo_shards().  None when not in streaming mode.
        self._cached_schur: Optional[np.ndarray] = None

        # --- Adjoint sensitivity state ---
        self._init_adjoint_state()

    def configure(self, settings: Dict[str, Any]) -> None:
        """Apply solver configuration on this worker (call once after creation).

        Propagates solver backend settings (CHOLMOD mode, ordering, etc.)
        so that Ray workers match the driver-side configuration.
        """
        from pgmath.block_system import set_partial_factor_reg_resistance
        from pgmath.factor import (
            set_use_cholmod,
            set_cholmod_mode,
            set_cholmod_ordering,
            set_cholmod_use_long,
        )
        if 'partial_factor_reg_ohms' in settings:
            set_partial_factor_reg_resistance(settings['partial_factor_reg_ohms'])
        if 'use_cholmod' in settings:
            set_use_cholmod(settings['use_cholmod'])
        if 'cholmod_mode' in settings:
            set_cholmod_mode(settings['cholmod_mode'])
        if 'cholmod_ordering' in settings:
            set_cholmod_ordering(settings['cholmod_ordering'])
        if 'cholmod_use_long' in settings:
            set_cholmod_use_long(settings['cholmod_use_long'])
        # A2 step-column table settings
        if 'use_step_columns' in settings:
            self._use_step_columns = bool(settings['use_step_columns'])
        if 'max_table_mb' in settings:
            self._max_table_mb = float(settings['max_table_mb'])
        # A4 symbolic-reuse setting
        if 'use_symbolic_reuse' in settings:
            self._use_symbolic_reuse = bool(settings['use_symbolic_reuse'])
        # Stage 1e: parse-time pre-clean trust flag (all-new-or-all-legacy,
        # resolved once at model creation -- see model._resolve_island_detection)
        if 'pre_cleaned_full_trusted' in settings:
            self._trust_pre_cleaned_full = bool(settings['pre_cleaned_full_trusted'])

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

    def setup_from_pkl(self, pkl_path, interface_nodes):
        """Load TileData from .pkl, build block system. Worker-side I/O."""
        import pickle
        with open(pkl_path, 'rb') as f:
            self._tile_data = pickle.load(f)
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

        # Floating island detection: remove components not connected to any port/boundary.
        # Stage 1e: skip entirely when this exact tile was already fully
        # pre-cleaned at parse time (TileData.pre_cleaned_full) AND the
        # model-creation-time trust assertion held (_trust_pre_cleaned_full,
        # set via configure() from model._resolve_island_detection) -- the
        # pkl is already clean, so no adjacency dict / BFS / mutation is
        # needed.  Legacy bundles (pre_cleaned_full=False, e.g. parsed before
        # Stage 1e) run today's path unchanged regardless of the trust flag.
        islands_removed = 0
        kept_nonlargest_iface: Set[str] = set()
        _skip_removal = (
            getattr(self._tile_data, 'pre_cleaned_full', False)
            and self._trust_pre_cleaned_full
        )
        if port_nodes_local and not _skip_removal:
            islands_removed, kept_nonlargest_iface = self._remove_floating_islands(port_nodes_local)
            # Finding R2: this is the legacy/trust-off branch -- the bundle
            # may already have arrived PRE-CLEANED at parse time (the
            # universal whole-tile pre-clean runs regardless of which
            # island_detection mode a LATER model creation requests), so a
            # fresh _remove_floating_islands call finds nothing left to
            # remove and reports 0 even though the tile genuinely lost
            # component(s) before this worker ever saw it.  The shared
            # _decompose_and_remove_floating core UNIONS the fresh count
            # into tile_data.n_floating_components_removed (never
            # overwrites), so after the call above that field already holds
            # parse-time + fresh -- use it directly (NOT
            # `islands_removed +=`, which would double-count the fresh
            # removal) so 'auto'/'summaries' and 'schur_bfs' report
            # IDENTICAL totals for the same bundle (the F8/F9 fix's stated
            # goal) instead of schur_bfs silently under-reporting.
            islands_removed = getattr(
                self._tile_data, 'n_floating_components_removed', islands_removed)
        elif _skip_removal:
            # Stage 1e findings F8/F9: removal already ran at PARSE time (not
            # here) -- the resulting diagnostics were persisted onto this
            # exact tile's own TileData (tile-local, in its own .pkl; see
            # tile_parsing._decompose_and_remove_floating).  Surface them
            # here exactly as the legacy worker-time removal would have, so
            # model.island_stats / tile_kept_nonlargest_iface / the
            # floating-nodes report / the verbose global-island cross-check
            # produce the SAME output in summaries mode as in schur_bfs mode.
            islands_removed = getattr(
                self._tile_data, 'n_floating_components_removed', 0)
            kept_nonlargest_iface = set(
                getattr(self._tile_data, 'kept_nonlargest_iface_cached', None) or ())
            # Finding R3: the cached value was computed against the
            # PARSE-time port-candidate set (raw shared_bnd at the
            # whole-tile pass, or sub.boundary_nodes-derived candidates at
            # the sub-tile pass), which can contain names ABSENT from the
            # model-creation-time FINAL interface set (e.g. a raw-shared
            # node demoted by a split neighbor's parse-time clean -- see
            # finding R1).  Intersect with the FINAL interface set this
            # worker actually received so no non-interface names leak into
            # model.tile_kept_nonlargest_iface / the verbose global-island
            # cross-check, regardless of whether R1's bundle-level
            # consistency check caught this particular bundle.
            kept_nonlargest_iface &= interface_nodes

        # Stage 1e F8/F9: unconditionally sync from
        # TileData.removed_floating_nodes -- populated either by
        # _remove_floating_islands() just now (via the shared
        # _decompose_and_remove_floating helper) or by parse-time
        # persistence (the _skip_removal branch above).
        self._removed_island_nodes = set(
            getattr(self._tile_data, 'removed_floating_nodes', None) or ())

        # Build BlockMatrixSystem from edges (no factorization yet)
        from pgmath.block_system import build_block_system_from_edges
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
            'kept_nonlargest_iface': sorted(kept_nonlargest_iface),
        }

    # Minimum interface node count for a non-largest component to be kept
    # in a tile that has NOT been pre-cleaned at the parent level.
    MIN_INTERFACE_NODES_KEEP = 5
    # Threshold used for sub-tiles created by split_tile().  After parent-level
    # pre-cleaning, any component connected to ≥1 port node is legitimate
    # (it was part of the parent's clean network and became visually
    # disconnected only because of the geometric cut).
    MIN_INTERFACE_NODES_KEEP_PRE_CLEANED = 1

    def _remove_floating_islands(self, port_nodes: Set[str]) -> Tuple[int, Set[str]]:
        """Remove disconnected components that are isolated fragments.

        For ordinary tiles (``tile_data.pre_cleaned=False``) keeps the
        largest component plus any component with ``>= MIN_INTERFACE_NODES_KEEP``
        interface nodes.

        For sub-tiles produced by :func:`~distributed.retile.split_tile`
        (``tile_data.pre_cleaned=True``) uses ``MIN_INTERFACE_NODES_KEEP_PRE_CLEANED=1``
        instead: genuinely floating nodes were already removed at the parent
        level before the split, so any component that connects to even one
        port node (including cut-interface nodes) is legitimate and must not
        be dropped.

        Returns:
            Tuple of (islands_removed, kept_nonlargest_iface).
        """
        # Select threshold: pre-cleaned sub-tiles use threshold=1.
        is_pre_cleaned = getattr(self._tile_data, 'pre_cleaned', False)
        min_iface_keep = (
            self.MIN_INTERFACE_NODES_KEEP_PRE_CLEANED
            if is_pre_cleaned
            else self.MIN_INTERFACE_NODES_KEEP
        )

        # Stage 1e finding F14: delegate to the shared decomposition/removal
        # core also used by retile._pre_clean_tile_data (parse-time
        # pre-clean), so the two removal decisions are bit-identical by
        # construction.  The helper also persists removal diagnostics onto
        # self._tile_data (findings F8/F9); _build_block_system() syncs
        # self._removed_island_nodes from there after this call returns.
        from .tile_parsing import _decompose_and_remove_floating
        _components, _largest, kept_nonlargest_iface, _removed_nodes, islands_removed = (
            _decompose_and_remove_floating(self._tile_data, port_nodes, min_iface_keep)
        )

        return islands_removed, kept_nonlargest_iface

    def factor_and_compute_schur(self) -> Tuple[Any, List[str], Dict[str, Any]]:
        """Factor interior and compute explicit Schur complement.

        Path selection (partial Cholesky vs chunked multi-RHS) is handled
        automatically by ``compute_explicit_schur()`` based on the active
        solver backend (CHOLMOD vs splu).  Both paths set ``lu_ii`` as a
        side-effect, so downstream RHS/recovery calls work transparently.

        A4: When ``_use_symbolic_reuse`` is True and CHOLMOD is active, the
        cached symbolic analysis from a prior DC or transient factorization is
        passed to ``compute_explicit_schur`` so the AMD analysis is skipped
        and only numeric values are updated.

        Returns:
            Tuple of (S_i as numpy array, boundary_node_list, stats dict)
        """
        from pgmath.schur import compute_explicit_schur
        from pgmath.block_system import _format_bytes, _sparse_mem_bytes

        bs = self._block_system
        t_total_start = time.perf_counter()

        # A4: Pass the worker's symbolic cache (may be None on first call).
        sym_cache = self._symbolic_ii if self._use_symbolic_reuse else None

        t0 = time.perf_counter()
        S, schur_stats = compute_explicit_schur(bs, symbolic_cache=sym_cache)
        schur_time = time.perf_counter() - t0

        # A4: Persist the new symbolic cache if the full analyze path ran.
        if '_new_symbolic_cache' in schur_stats:
            self._symbolic_ii = schur_stats.pop('_new_symbolic_cache')
        # On symbolic-reuse path, self._symbolic_ii['factor'] was updated
        # in-place inside _compute_schur_partial — no assignment needed.

        # Both paths report factor timing in schur_stats: partial path
        # via 'analyze_s'+'factor_s', chunked path via 'factor_s' alone.
        factor_time = schur_stats.get('factor_s', 0) + schur_stats.get('analyze_s', 0)

        total_time = time.perf_counter() - t_total_start

        schur_path = schur_stats['path']

        # Build stats dict
        stats = dict(bs.stats())  # n_ports, n_interior, G_ii_nnz, etc.
        stats.update({
            'factor_interior_s': factor_time,
            'compute_schur_s': schur_time,
            'total_s': total_time,
            'schur_shape': S.shape,
            'schur_mem_bytes': schur_stats['schur_mem_bytes'],
            'schur_path': schur_path,
        })

        # Backend info from factor_adapter (may be None if n_interior == 0)
        fa = bs.factor_adapter
        if fa is not None:
            stats['factorization_backend'] = fa.backend
            stats['factorization_backend_info'] = fa.backend_info
        else:
            stats['factorization_backend'] = 'n/a'
            stats['factorization_backend_info'] = 'n/a'

        # DEBUG logging: per-tile matrix characteristics
        tid = self._tile_data.tile_id if self._tile_data else '?'
        n_ii = stats['n_interior']
        n_pp = stats['n_ports']
        G_ii_nnz = stats['G_ii_nnz']
        density = (G_ii_nnz / (n_ii * n_ii) * 100) if n_ii > 0 else 0.0
        G_ii_mem = _sparse_mem_bytes(bs.G_ii) if hasattr(bs.G_ii, 'data') else 0
        logger.debug(
            "Tile %s factor_and_compute_schur:\n"
            "  G_ii: %s x %s, nnz=%s (density %.5f%%), %s\n"
            "  G_pp: %s x %s, nnz=%s\n"
            "  G_pi: %s x %s, nnz=%s  |  G_ip: %s x %s, nnz=%s\n"
            "  Block system memory: %s\n"
            "  factor_interior: %.3fs  |  path: %s  |  backend: %s\n"
            "  compute_schur: %.3fs  |  Schur: %s x %s dense (%s)",
            tid,
            f"{n_ii:,}", f"{n_ii:,}", f"{G_ii_nnz:,}", density, _format_bytes(G_ii_mem),
            f"{n_pp:,}", f"{n_pp:,}", f"{stats['G_pp_nnz']:,}",
            f"{n_pp:,}", f"{n_ii:,}", f"{stats['G_pi_nnz']:,}",
            f"{n_ii:,}", f"{n_pp:,}", f"{stats['G_ip_nnz']:,}",
            _format_bytes(stats['mem_bytes']),
            factor_time, schur_path, stats['factorization_backend_info'],
            schur_time, f"{S.shape[0]:,}", f"{S.shape[1]:,}",
            _format_bytes(schur_stats['schur_mem_bytes']),
        )

        return S, list(bs.port_nodes), stats

    def get_reduced_rhs(self, current_injections: Optional[Dict[str, float]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Compute reduced RHS for this tile.

        Args:
            current_injections: Override current injections. If None, uses tile's own.

        Returns:
            Tuple of (reduced RHS numpy array (n_ports,), stats dict)
        """
        from pgmath.block_system import compute_reduced_rhs

        if current_injections is None:
            current_injections = self._tile_data.current_injections

        t0 = time.perf_counter()
        rhs = compute_reduced_rhs(
            self._block_system, current_injections, self._rhs_dirichlet,
        )
        rhs_time = time.perf_counter() - t0

        n_currents = sum(1 for v in current_injections.values() if v != 0.0)
        rhs_norm = float(np.linalg.norm(rhs))

        stats = {
            'rhs_time_s': rhs_time,
            'n_currents': n_currents,
            'rhs_norm': rhs_norm,
        }

        tid = self._tile_data.tile_id if self._tile_data else '?'
        logger.debug(
            "Tile %s get_reduced_rhs:\n"
            "  time: %.3fs  |  n_currents: %s  |  rhs_norm: %.2f",
            tid, rhs_time, f"{n_currents:,}", rhs_norm,
        )

        return rhs, stats

    def get_interior_voltages(
        self,
        boundary_voltages_dict: Dict[str, float],
        current_injections: Optional[Dict[str, float]] = None,
    ) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """Recover interior voltages from boundary voltages.

        Args:
            boundary_voltages_dict: Dict mapping boundary node -> voltage
            current_injections: Optional override currents (node -> mA).

        Returns:
            Tuple of (Dict mapping all tile nodes -> voltage, stats dict)
        """
        from pgmath.block_system import recover_bottom_voltages

        t0 = time.perf_counter()

        currents = current_injections if current_injections is not None else self._tile_data.current_injections

        port_voltages = np.zeros(self._block_system.n_ports, dtype=np.float64)
        for node, idx in self._block_system.port_to_idx.items():
            if node in boundary_voltages_dict:
                port_voltages[idx] = boundary_voltages_dict[node]

        interior_voltages = recover_bottom_voltages(
            self._block_system,
            port_voltages,
            currents,
            self._rhs_dirichlet,
        )

        all_voltages = dict(interior_voltages)
        for node in self._block_system.port_nodes:
            if node in boundary_voltages_dict:
                all_voltages[node] = boundary_voltages_dict[node]

        recovery_time = time.perf_counter() - t0

        n_nodes = len(all_voltages)
        if all_voltages:
            vals = list(all_voltages.values())
            v_min = min(vals)
            v_max = max(vals)
        else:
            v_min = 0.0
            v_max = 0.0

        stats = {
            'recovery_time_s': recovery_time,
            'n_nodes': n_nodes,
            'v_min': v_min,
            'v_max': v_max,
        }

        tid = self._tile_data.tile_id if self._tile_data else '?'
        logger.debug(
            "Tile %s get_interior_voltages:\n"
            "  time: %.3fs  |  n_nodes: %s  |  v_range: [%.3f, %.3f]",
            tid, recovery_time, f"{n_nodes:,}", v_min, v_max,
        )

        return all_voltages, stats

    def get_layer_metadata(self) -> Dict[str, Dict]:
        """Per-layer spatial extents, orientation, and stripe coordinates."""
        if self._tile_data is None:
            raise RuntimeError(
                "get_layer_metadata() called before setup(); no tile data loaded"
            )

        from visualization.stripe_heatmap import parse_node_info

        node_info: Dict[str, Tuple[float, float, str]] = {}
        layer_xs: Dict[str, List[float]] = {}
        layer_ys: Dict[str, List[float]] = {}

        for node in self._tile_data.all_nodes:
            x, y, layer = parse_node_info(node)
            if x is None:
                continue
            node_info[node] = (x, y, layer)
            layer_xs.setdefault(layer, []).append(x)
            layer_ys.setdefault(layer, []).append(y)

        layer_h: Dict[str, int] = {}
        layer_v: Dict[str, int] = {}
        layer_d: Dict[str, int] = {}

        for u, v, _g in self._tile_data.resistive_edges:
            u_info = node_info.get(u)
            v_info = node_info.get(v)
            if u_info is None or v_info is None:
                continue
            ux, uy, u_layer = u_info
            vx, vy, v_layer = v_info
            if u_layer != v_layer:
                continue

            dx = abs(vx - ux)
            dy = abs(vy - uy)
            if dx == 0 and dy == 0:
                continue

            if dy == 0:
                layer_h[u_layer] = layer_h.get(u_layer, 0) + 1
            elif dx == 0:
                layer_v[u_layer] = layer_v.get(u_layer, 0) + 1
            else:
                layer_d[u_layer] = layer_d.get(u_layer, 0) + 1

        result: Dict[str, Dict] = {}
        for layer in sorted(layer_xs):
            xs = layer_xs[layer]
            ys = layer_ys[layer]
            result[layer] = {
                'bbox': (min(xs), max(xs), min(ys), max(ys)),
                'n_nodes': len(xs),
                'stripe_coords_h': sorted(set(ys)),
                'stripe_coords_v': sorted(set(xs)),
                'edge_orientation': (
                    layer_h.get(layer, 0),
                    layer_v.get(layer, 0),
                    layer_d.get(layer, 0),
                ),
            }

        return result

    def get_schur_coo_shards(
        self,
        n_shards: int,
        tile_index_map: Optional[np.ndarray],
    ) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Return the tile's S_i contribution as COO shards with global indices.

        B3: Streaming Schur assembly — first-call (COO accumulation) path only.
        Used when the CSR scatter pattern has not yet been computed.  The
        coordinator needs global (rows, cols) to build the first-call COO matrix
        and infer the CSR sparsity pattern.

        IMPORTANT — wire-cost note:
        Each shard contains ``int32 global_rows + int32 global_cols + float64 data``
        which is 16 bytes per entry, i.e. *2× the 8 bytes/entry* of the dense S_i.
        All ``n_shards`` shards are built eagerly into a single list worker-side
        and returned in one call, so the Ray boundary is crossed once per tile
        (not once per shard).  ``n_shards`` therefore controls ONLY the
        coordinator's scatter granularity — it does NOT bound worker-side peak
        or per-shard transfer size.

        For the fast path (CSR scatter pattern already cached), call
        ``get_schur_data_flat`` instead: it returns only the float64 data
        (8 bytes/entry, no rows/cols), since the scatter pattern already encodes
        where each entry goes.

        Callers first invoke ``factor_and_cache_schur`` (which factors interior
        and caches S_i worker-side without sending it to the coordinator), then
        call this method so the coordinator can stream the COO data.

        Args:
            n_shards: Number of row-shards to split S_i into (>=1).
                Each shard covers ``ceil(n_ports / n_shards)`` rows.
                The last shard may be smaller.
                Controls scatter granularity on the coordinator; does NOT reduce
                the total bytes transferred across the Ray boundary.
            tile_index_map: int32 array of shape ``(n_ports,)`` mapping local
                port index ``i`` to global interface index ``tile_index_map[i]``.
                Must be consistent with the local port ordering in
                ``self._block_system.port_nodes``.

        Returns:
            List of (global_rows, global_cols, data) tuples — one per shard.
            global_rows and global_cols are int32; data is float64.
            All shards are materialized worker-side before return.

        Raises:
            RuntimeError: if the block system is not yet built / factored.
        """
        if self._block_system is None:
            raise RuntimeError(
                "get_schur_coo_shards() called before block system is built. "
                "Call setup() or setup_from_pkl() first."
            )

        # Use cached S_i from factor_and_cache_schur() if available; otherwise compute fresh.
        _cached = getattr(self, '_cached_schur', None)
        if _cached is not None:
            S_i = _cached
            # Free the cache after use — the coordinator reads shards and discards them.
            self._cached_schur = None
        else:
            from pgmath.schur import compute_explicit_schur
            sym_cache = self._symbolic_ii if self._use_symbolic_reuse else None
            S_i, schur_stats = compute_explicit_schur(self._block_system, symbolic_cache=sym_cache)
            # Persist symbolic cache if the full analyze path ran
            if '_new_symbolic_cache' in schur_stats:
                self._symbolic_ii = schur_stats.pop('_new_symbolic_cache')

        # Build global index map
        if tile_index_map is None:
            # Build from block_system port_nodes order
            # This is a fallback; callers should always pass tile_index_map.
            raise ValueError(
                "tile_index_map must be provided to get_schur_coo_shards(). "
                "It must map local port index -> global interface index and "
                "be consistent with block_system.port_nodes ordering."
            )

        idx = np.asarray(tile_index_map, dtype=np.int32)
        n_ports = S_i.shape[0]
        assert len(idx) == n_ports, (
            f"tile_index_map length {len(idx)} != n_ports {n_ports}"
        )

        n_shards = max(1, n_shards)
        shard_size = max(1, (n_ports + n_shards - 1) // n_shards)

        shards: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for shard_start in range(0, n_ports, shard_size):
            shard_end = min(shard_start + shard_size, n_ports)
            # Row-slice of S_i: shape (shard_rows, n_ports)
            S_slice = S_i[shard_start:shard_end, :]  # (shard_rows, n_ports)
            shard_rows_local = np.arange(shard_start, shard_end, dtype=np.int32)

            # COO global indices
            # For each row r in shard, column c in [0, n_ports):
            #   global_row = idx[r],  global_col = idx[c]
            n_shard_rows = shard_end - shard_start
            global_rows = np.repeat(idx[shard_rows_local], n_ports)
            global_cols = np.tile(idx, n_shard_rows)
            data = S_slice.ravel().astype(np.float64)

            shards.append((global_rows, global_cols, data))

        return shards

    def get_schur_coo_indices_only(
        self,
        tile_index_map: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return the tile's S_i COO global indices WITHOUT float64 data.

        B3 two-pass first-call path: used by the coordinator's index-only pre-pass
        to build the CSR sparsity pattern (indptr/indices) without transferring any
        float64 values.  The per-tile float64 data is later streamed separately via
        ``get_schur_data_flat`` once the scatter pattern is known.

        Wire cost: ``n_ports^2 * 8 bytes`` (two int32 arrays) vs 16 bytes/entry for
        ``get_schur_coo_shards`` (rows + cols + float64 data).  Since the data is NOT
        sent, this call halves first-call wire cost and — critically — the coordinator
        accumulates only index arrays (int32, 4 B/entry each) during the pre-pass,
        NOT float64 values (8 B/entry).  After the pre-pass builds the CSR pattern,
        the accumulated index arrays are freed and data is streamed via the fast path.

        First-call coordinator peak with two-pass approach:
          Pre-pass:   sum_tiles(n_ports_i^2) * 8 bytes (two int32 arrays, no float64)
          Data pass:  one tile's n_ports^2 * 8 bytes (float64) + preallocated CSR buffer
          Overall:    O(nnz * 8 B) rather than O(nnz * 16 B) with old single-pass COO.
          (nnz here refers to the sum of all tile S_i entries = sum n_ports_i^2.)

        IMPORTANT: does NOT consume ``_cached_schur`` — the cached S_i remains after
        this call so ``get_schur_data_flat`` can use it in the data pass.

        Args:
            tile_index_map: int32 array of shape ``(n_ports,)`` mapping local port
                index ``i`` to global interface index ``tile_index_map[i]``.

        Returns:
            (global_rows, global_cols) both int32 of length ``n_ports^2``, in
            row-major order consistent with ``S_i.ravel()``.

        Raises:
            RuntimeError: if ``factor_and_cache_schur`` has not been called yet
                          (``_cached_schur`` is None).
        """
        if self._block_system is None:
            raise RuntimeError(
                "get_schur_coo_indices_only() called before block system is built. "
                "Call setup() or setup_from_pkl() first."
            )

        _cached = getattr(self, '_cached_schur', None)
        if _cached is None:
            raise RuntimeError(
                "get_schur_coo_indices_only() called before factor_and_cache_schur(). "
                "S_i is not cached on this worker.  Call factor_and_cache_schur() "
                "first, then get_schur_coo_indices_only() for the index pre-pass "
                "followed by get_schur_data_flat() for the data pass."
            )

        idx = np.asarray(tile_index_map, dtype=np.int32)
        n_ports = _cached.shape[0]
        if len(idx) != n_ports:
            raise ValueError(
                f"tile_index_map length {len(idx)} != n_ports {n_ports}"
            )

        # Build (global_rows, global_cols) in row-major order (same as S_i.ravel()).
        # global_rows[k] = idx[row(k)], global_cols[k] = idx[col(k)]
        # where k = row * n_ports + col.
        global_rows = np.repeat(idx, n_ports)   # shape (n_ports^2,), int32
        global_cols = np.tile(idx, n_ports)     # shape (n_ports^2,), int32
        # NOTE: _cached_schur is NOT cleared here — it will be consumed by
        # get_schur_data_flat() in the subsequent data pass.
        return global_rows, global_cols

    def get_schur_data_flat(self) -> np.ndarray:
        """Return S_i as a flat float64 array (row-major) WITHOUT global indices.

        B3 fast-path: used by the coordinator when the CSR scatter pattern is
        already cached from a prior streaming call.  The scatter pattern encodes
        exactly where each entry of ``S_i.ravel()`` lands in the preallocated
        ``G_full_data`` buffer, so the coordinator needs only the *values* — not
        the (rows, cols) indices that ``get_schur_coo_shards`` also sends.

        Wire cost: exactly ``n_ports * n_ports * 8`` bytes (float64 only), which
        is **half** the 16 bytes/entry that ``get_schur_coo_shards`` transfers
        (int32 rows + int32 cols + float64 data).

        Memory note:
          - Worker peak: S_i is in ``_cached_schur``; ``ravel()`` on a C-contiguous
            array returns a view (no extra copy), so worker peak = S_i size only.
          - Coordinator peak: one tile's float64 array is live, then immediately
            scatter-added and freed before the next tile's result arrives.

        Returns:
            1-D float64 array of length ``n_ports * n_ports`` in row-major order
            (same as ``S_i.ravel(order='C')``).  The cached S_i is freed after
            this call to release worker-side memory.

        Raises:
            RuntimeError: if ``factor_and_cache_schur`` has not been called yet.
        """
        if self._block_system is None:
            raise RuntimeError(
                "get_schur_data_flat() called before block system is built. "
                "Call setup() or setup_from_pkl() first."
            )

        _cached = getattr(self, '_cached_schur', None)
        if _cached is None:
            raise RuntimeError(
                "get_schur_data_flat() called before factor_and_cache_schur(). "
                "S_i is not cached on this worker.  Call factor_and_cache_schur() "
                "first, then get_schur_data_flat() to retrieve the flat data."
            )

        # ravel on a C-contiguous array is a no-copy view; ensure float64.
        data_flat = _cached.ravel(order='C').astype(np.float64, copy=False)

        # Free the cached S_i — caller is done with it.
        self._cached_schur = None

        return data_flat

    def factor_and_cache_schur(self) -> Tuple[List[str], Dict[str, Any]]:
        """Factor interior AND compute S_i, caching it worker-side without returning it.

        B3: Used in the streaming assembly path.  The coordinator calls this
        first (factors all tiles in parallel), then calls
        ``get_schur_coo_shards()`` tile-by-tile to stream the COO data.
        S_i lives in worker-process memory only — it never crosses the
        coordinator boundary as a dense matrix.

        Returns:
            Tuple of (boundary_node_list, stats dict) — same layout as
            ``factor_and_compute_schur`` minus the dense S_i.
        """
        from pgmath.schur import compute_explicit_schur
        from pgmath.block_system import _format_bytes, _sparse_mem_bytes
        import time as _time_inner

        bs = self._block_system
        sym_cache = self._symbolic_ii if self._use_symbolic_reuse else None

        t0 = _time_inner.perf_counter()
        S, schur_stats = compute_explicit_schur(bs, symbolic_cache=sym_cache)
        elapsed = _time_inner.perf_counter() - t0

        # Persist symbolic cache
        if '_new_symbolic_cache' in schur_stats:
            self._symbolic_ii = schur_stats.pop('_new_symbolic_cache')

        # Cache S_i worker-side for subsequent get_schur_coo_shards calls
        self._cached_schur: np.ndarray = S

        factor_time = schur_stats.get('factor_s', 0) + schur_stats.get('analyze_s', 0)

        stats = dict(bs.stats())
        stats.update({
            'factor_interior_s': factor_time,
            'compute_schur_s': elapsed - factor_time,
            'total_s': elapsed,
            'schur_shape': S.shape,
            'schur_mem_bytes': schur_stats['schur_mem_bytes'],
            'schur_path': schur_stats['path'],
        })
        fa = bs.factor_adapter
        if fa is not None:
            stats['factorization_backend'] = fa.backend
            stats['factorization_backend_info'] = fa.backend_info
        else:
            stats['factorization_backend'] = 'n/a'
            stats['factorization_backend_info'] = 'n/a'

        return list(bs.port_nodes), stats

    def get_schur_size_stats(self) -> Dict[str, Any]:
        """Return lightweight per-tile size stats without computing S_i.

        B3: Used by the 'auto' streaming decision in ``_factor_dc_context``
        so the coordinator can estimate peak S_i memory BEFORE factoring
        interior.  This avoids the catch-22 where 'auto' previously had to
        run ``factor_and_compute_schur`` (gathering all dense S_i) just to
        decide whether streaming would have been beneficial.

        The ``schur_mem_bytes`` estimate is ``n_ports^2 * 8`` (float64),
        which equals the actual S_i memory when it is a dense n_ports×n_ports
        array.  This is exact for the default (non-sparse) Schur path.

        Returns a dict with:
          'n_interior'       int
          'n_ports'          int
          'schur_mem_bytes'  int  -- estimate: n_ports^2 * 8
        """
        if self._block_system is None:
            return {'n_interior': 0, 'n_ports': 0, 'schur_mem_bytes': 0}
        bs = self._block_system
        n_p = bs.n_ports
        return {
            'n_interior': bs.n_interior,
            'n_ports': n_p,
            'schur_mem_bytes': n_p * n_p * 8,
        }

    def get_port_node_list(self) -> List[str]:
        """Return this tile's port (boundary) node NAME list, no computation.

        Stage 2 item 3 ("tilewise without ever assembling S_global"): the
        coordinator needs every tile's port name list to compute the
        interface node ordering (unknown vs. Dirichlet split) and Dirichlet
        RHS BEFORE it gathers any per-tile Schur VALUES -- this lightweight
        round-trip provides exactly that, without touching G_ii/factoring/
        Schur computation.  ``self._block_system.port_nodes`` is already
        built by ``setup()``/``setup_from_pkl()`` (island removal + block
        assembly), so this is a free attribute read.
        """
        if self._block_system is None:
            raise RuntimeError(
                "get_port_node_list() called before setup(); no block "
                "system built yet."
            )
        return list(self._block_system.port_nodes)

    def get_current_injections(self) -> Dict[str, float]:
        """Return per-tile current injection dict ``{node: mA}``."""
        if self._tile_data is None:
            raise RuntimeError(
                "get_current_injections() called before setup(); no tile data loaded"
            )
        return dict(self._tile_data.current_injections)

    @property
    def tile_id(self) -> Tuple[int, int]:
        return self._tile_data.tile_id if self._tile_data else None

    @property
    def n_interior(self) -> int:
        return self._block_system.n_interior if self._block_system else 0

    @property
    def n_boundary(self) -> int:
        return self._block_system.n_ports if self._block_system else 0

    def get_floating_nodes_data(self) -> Dict[str, Any]:
        """Return floating node data for report aggregation."""
        if self._tile_data is None:
            raise RuntimeError(
                "get_floating_nodes_data() called before setup(); no tile data loaded"
            )
        return {
            'removed_nodes': list(self._removed_island_nodes),
            'tile_id': self._tile_data.tile_id,
        }

    def lookup_instance_names(
        self,
        target_nodes: Set[str],
        instance_path: Optional[str],
        nd_path: Optional[str] = None,
        net_filter: Optional[str] = None,
    ) -> Dict[str, str]:
        """Look up instance names for a set of target nodes from instanceModels file."""
        node_to_instance: Dict[str, str] = {}

        for prepared in _iter_instance_sources(instance_path, net_filter, nd_path):
            if prepared.node_pos in target_nodes:
                node_to_instance[prepared.node_pos] = prepared.cs.name

        return node_to_instance

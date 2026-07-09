"""Time-domain mixin for TileWorker.

Provides VCS creation, PWL smoothing, quasi-static evaluation, transient
factorization/RHS/recovery, and peak tracking.  Kept in a separate file
to keep tile_worker.py under 500 lines; the public API is surfaced on
TileWorker via inheritance.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np

from .tile_worker_peak import _PeakTrackingMixin

logger = logging.getLogger(__name__)

# A5: bump this constant whenever the smoothing algorithm changes so that
# existing smoothed-VCS caches are automatically invalidated.
SMOOTHING_CODE_VERSION: int = 1


class _TimeDomainMixin(_PeakTrackingMixin):
    """Mixin providing time-domain methods for TileWorker.

    Peak tracking methods (init_peak_tracking, update_peak_stats,
    get_peak_stats, prebin_peak_data, get_top_k_peaks,
    get_tracked_waveforms, recover_and_update_peaks,
    recover_transient_and_update_peaks) are provided by
    _PeakTrackingMixin in tile_worker_peak.py.

    Expects the host class to provide:
        _block_system, _rhs_dirichlet, _tile_data, _vec_sources,
        _smoothed_sources, _active_sources, _transient_block_system,
        _c_pp_diag, _c_ii_diag, _total_cap, _C_coeff, _dt_scaled,
        _transient_method, _v_interior_old,
        _last_f_i, _peak_per_node, _peak_vdd,
        _peak_tracking_active, _tracked_nodes, _tracked_waveforms,
        get_reduced_rhs()
    """

    # --- 4a. Vectorized current source creation + disk caching ---------

    def init_vectorized_sources(
        self,
        instance_path: Optional[str],
        nd_path: Optional[str] = None,
        net_filter: Optional[str] = None,
        pkl_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Build VectorizedCurrentSources from instance model file.

        Supports optional disk caching via pkl_dir.

        Args:
            instance_path: Path to instanceModels*.sp file.
            nd_path: Path to .nd file for net filtering.
            net_filter: Optional lowercase net name to filter by.
            pkl_dir: Optional directory for VCS pickle cache.

        Returns:
            Stats dict: ``{n_sources, n_nodes, cached}``.
        """
        n_ports = self._block_system.n_ports
        n_interior = self._block_system.n_interior
        n_nodes = n_ports + n_interior

        if self._vec_sources is not None:
            stats = self._vec_sources.get_statistics()
            stats['n_nodes'] = n_nodes
            stats['cached'] = True
            return stats

        import hashlib as _hashlib
        import os
        import pickle
        from analysis.vectorized_sources import VectorizedCurrentSources
        from .tile_parsing import _iter_instance_sources

        tile_str = '_'.join(str(c) for c in self._tile_data.tile_id)

        # Compute a content signature from sorted interior node names.
        # This discriminates across different splits of the same parent tile (which
        # share the same (x,y,k) tile_id) even when n_nodes coincides, preventing
        # silent injection of wrong current sources after --max-interior changes.
        # Using interior nodes (not ports) because interior nodes uniquely identify
        # the partition: each interior node belongs to exactly one tile.
        _interior_keys = sorted(self._block_system.interior_to_idx.keys())
        node_sig = _hashlib.md5('|'.join(_interior_keys).encode()).hexdigest()[:16]

        # Try loading from disk cache.
        # Staleness check: node-content signature is the primary guard; n_nodes is
        # a secondary sanity check.  Sub-tile IDs (x, y, k) are deterministic per
        # splitting run but collide across different --max-interior values when the
        # resulting sub-tiles coincidentally have the same n_nodes — the signature
        # detects this case and forces a rebuild.
        if pkl_dir is not None:
            cache_path = os.path.join(pkl_dir, f'vcs_tile_{tile_str}.pkl')
            if os.path.isfile(cache_path):
                logger.debug("Tile %s: loading VCS from %s", tile_str, cache_path)
                try:
                    with open(cache_path, 'rb') as f:
                        raw = pickle.load(f)

                    # New format: wrapper dict {'vcs':…, 'n_nodes':…, 'node_sig':…}
                    # Old/legacy format: direct VCS object (no signature → always stale).
                    if isinstance(raw, dict) and 'vcs' in raw:
                        cached_vcs = raw['vcs']
                        cached_n = raw.get('n_nodes')
                        cached_sig = raw.get('node_sig')
                    else:
                        cached_vcs = raw
                        cached_n = getattr(raw, 'n_nodes', None)
                        cached_sig = None  # legacy: no signature → will be rejected

                    if cached_sig != node_sig:
                        logger.warning(
                            "Tile %s: stale VCS cache rejected "
                            "(node-content signature mismatch). Rebuilding.",
                            tile_str,
                        )
                        # fall through to rebuild
                    elif cached_n is not None and cached_n != n_nodes:
                        logger.warning(
                            "Tile %s: stale VCS cache rejected "
                            "(n_nodes mismatch: cached=%d, current=%d). Rebuilding.",
                            tile_str, cached_n, n_nodes,
                        )
                        # fall through to rebuild
                    else:
                        self._vec_sources = cached_vcs
                        self._active_sources = self._vec_sources
                        stats = self._vec_sources.get_statistics()
                        stats['n_nodes'] = n_nodes
                        stats['cached'] = True
                        return stats
                except Exception as _exc:
                    logger.warning(
                        "Tile %s: failed to load VCS cache (%s); rebuilding.",
                        tile_str, _exc,
                    )

        # Build node_to_idx: ports [0..n_ports), interior [n_ports..n_total)
        # For PORT nodes that appear in multiple tiles (cut nodes), only include
        # the VCS source in the tile that owns the node (i.e., that tile's
        # current_injections dict has the node as a key).  Interior nodes are
        # always unique to one tile, so they are always included.
        owned_nodes = set(
            self._tile_data.current_injections.keys()
            if self._tile_data is not None else []
        )
        node_to_idx: Dict[str, int] = {
            node: idx
            for node, idx in self._block_system.port_to_idx.items()
            if node in owned_nodes
        }
        for node, idx in self._block_system.interior_to_idx.items():
            node_to_idx[node] = idx + n_ports

        # Collect full CurrentSource objects from instance file
        sources_dict: Dict[str, Any] = {}
        for prepared in _iter_instance_sources(instance_path, net_filter, nd_path):
            sources_dict[prepared.cs.name] = prepared.cs

        self._vec_sources = VectorizedCurrentSources.from_current_sources(
            sources_dict, node_to_idx, n_nodes,
        )
        self._active_sources = self._vec_sources
        # A2: new raw sources invalidate any existing step-column table
        self._step_col_table = None

        # Save to disk cache using wrapper dict that includes the node signature.
        # The smoothed VCS cache (vcs_tile_X_Y_smoothed_<hash>.pkl) validates the
        # raw cache via mtime+size, so changing the wrapper format here automatically
        # invalidates any existing smoothed caches on the next run — correct behaviour.
        if pkl_dir is not None:
            os.makedirs(pkl_dir, exist_ok=True)
            cache_path = os.path.join(pkl_dir, f'vcs_tile_{tile_str}.pkl')
            logger.debug("Tile %s: saving VCS to %s", tile_str, cache_path)
            cache_payload = {
                'vcs': self._vec_sources,
                'n_nodes': n_nodes,
                'node_sig': node_sig,
            }
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_payload, f, protocol=pickle.HIGHEST_PROTOCOL)

        stats = self._vec_sources.get_statistics()
        stats['n_nodes'] = n_nodes
        stats['cached'] = False
        return stats

    # --- 4b. PWL smoothing ---------------------------------------------

    def get_min_pwl_segment_duration(self) -> float:
        """Return minimum PWL segment / pulse feature duration across raw sources.

        Used by the coordinator's ``smooth='auto'`` logic to decide whether
        smoothing is necessary.  Scans ``_vec_sources`` (raw VCS) only —
        must be called after ``init_vectorized_sources()``.

        For PWL waveforms: minimum ``diff(times)`` across all breakpoints.
        For pulse waveforms: minimum of rise_time, fall_time, width (> 0).

        Returns:
            Minimum positive duration in seconds.  Returns ``float('inf')``
            when no time-varying sources are loaded or none have positive
            segment durations.
        """
        vcs = self._vec_sources
        if vcs is None:
            return float('inf')

        min_dur = float('inf')

        # PWL: min inter-breakpoint spacing across all waveforms
        if vcs.n_pwls > 0 and vcs.n_pwl_points > 1:
            for i in range(vcs.n_pwls):
                off = int(vcs.pwl_offset[i])
                cnt = int(vcs.pwl_count[i])
                if cnt < 2:
                    continue
                times_seg = vcs.pwl_times[off:off + cnt]
                diffs = np.diff(times_seg)
                pos = diffs[diffs > 0.0]
                if len(pos) > 0:
                    min_dur = min(min_dur, float(pos.min()))

        # Pulse: min of rise_time, fall_time, width (> 0)
        if vcs.n_pulses > 0:
            for arr in (vcs.pulse_rt, vcs.pulse_ft, vcs.pulse_width):
                if len(arr) > 0:
                    pos = arr[arr > 0.0]
                    if len(pos) > 0:
                        min_dur = min(min_dur, float(pos.min()))

        return min_dur

    def smooth_sources(
        self,
        time_step: float,
        t_start: float,
        t_end: float,
        pkl_dir: Optional[str] = None,
        compact_threshold: float = 1e-12,
        chunk_size: int = 10000,
    ) -> Dict[str, Any]:
        """Apply triangular low-pass smoothing to vectorized sources.

        Supports optional disk caching via ``pkl_dir``.  The smoothed VCS
        is saved next to the raw cache as
        ``vcs_tile_X_Y_smoothed_<key>.pkl`` and reloaded on subsequent
        calls with identical parameters.  The cached payload records the
        raw pkl's mtime + size; if the raw cache is modified or replaced
        the smoothed cache is invalidated automatically.

        Cache key is a 12-char MD5 prefix of
        ``(time_step, t_start, t_end, compact_threshold, chunk_size,
        SMOOTHING_CODE_VERSION)``.  Bump :data:`SMOOTHING_CODE_VERSION`
        whenever the smoothing algorithm changes.

        Args:
            time_step: Filter window = 2 * time_step (seconds).
            t_start: Simulation start time (seconds).
            t_end: Simulation end time (seconds).
            pkl_dir: Optional directory for smoothed VCS cache (same dir
                used by ``init_vectorized_sources``).
            compact_threshold: PWL compaction threshold passed to
                ``create_smoothed_copy`` (default 1e-12).
            chunk_size: Smoothing chunk size passed to
                ``create_smoothed_copy`` (default 10000).

        Returns:
            Stats dict::

                {
                    'time_step': float,
                    't_start': float,
                    't_end': float,
                    'smooth_time_s': float,   # 0.0 on cache hit
                    'cached': bool,
                }
        """
        import hashlib
        import os
        import pickle as _pickle

        if self._vec_sources is None:
            raise RuntimeError(
                "smooth_sources() called before init_vectorized_sources()"
            )

        tile_str = '_'.join(str(c) for c in self._tile_data.tile_id)

        # Build a stable key from all parameters that affect smoothing output.
        key_str = (
            f"{time_step:.17g}:{t_start:.17g}:{t_end:.17g}"
            f":{compact_threshold:.17g}:{chunk_size:d}"
            f":{SMOOTHING_CODE_VERSION:d}"
        )
        key_hash = hashlib.md5(key_str.encode()).hexdigest()[:12]

        # --- Try loading from disk cache ---
        smoothed_cache_path: Optional[str] = None
        raw_cache_path: Optional[str] = None
        if pkl_dir is not None:
            raw_cache_path = os.path.join(pkl_dir, f'vcs_tile_{tile_str}.pkl')
            smoothed_cache_path = os.path.join(
                pkl_dir, f'vcs_tile_{tile_str}_smoothed_{key_hash}.pkl'
            )
            if (
                os.path.isfile(smoothed_cache_path)
                and os.path.isfile(raw_cache_path)
            ):
                try:
                    raw_stat = os.stat(raw_cache_path)
                    raw_mtime = raw_stat.st_mtime
                    raw_size = raw_stat.st_size
                    with open(smoothed_cache_path, 'rb') as _f:
                        payload = _pickle.load(_f)
                    # Verify raw cache identity (mtime + size)
                    if (
                        payload.get('raw_mtime') == raw_mtime
                        and payload.get('raw_size') == raw_size
                    ):
                        self._smoothed_sources = payload['smoothed_vcs']
                        self._active_sources = self._smoothed_sources
                        # A2: active sources changed → invalidate step-column table
                        self._step_col_table = None
                        logger.debug(
                            "Tile %s: smoothed VCS cache HIT %s",
                            tile_str, smoothed_cache_path,
                        )
                        return {
                            'time_step': time_step,
                            't_start': t_start,
                            't_end': t_end,
                            'smooth_time_s': 0.0,
                            'cached': True,
                        }
                    else:
                        logger.debug(
                            "Tile %s: smoothed cache stale "
                            "(raw mtime/size changed), recomputing",
                            tile_str,
                        )
                except Exception as _exc:
                    logger.debug(
                        "Tile %s: smoothed cache load failed (%s), "
                        "recomputing",
                        tile_str, _exc,
                    )

        # --- Cache miss: compute smoothed VCS ---
        t0 = time.perf_counter()
        self._smoothed_sources = self._vec_sources.create_smoothed_copy(
            time_step, t_start, t_end,
            compact_threshold=compact_threshold,
            chunk_size=chunk_size,
        )
        smooth_time = time.perf_counter() - t0
        self._active_sources = self._smoothed_sources
        # A2: smoothing changes active sources → invalidate step-column table
        self._step_col_table = None

        logger.debug(
            "Tile %s: smooth_sources computed in %.3fs",
            tile_str, smooth_time,
        )

        # --- Save to smoothed cache ---
        if (
            pkl_dir is not None
            and smoothed_cache_path is not None
            and raw_cache_path is not None
            and os.path.isfile(raw_cache_path)
        ):
            try:
                raw_stat = os.stat(raw_cache_path)
                payload = {
                    'raw_mtime': raw_stat.st_mtime,
                    'raw_size': raw_stat.st_size,
                    'smoothed_vcs': self._smoothed_sources,
                }
                os.makedirs(pkl_dir, exist_ok=True)
                with open(smoothed_cache_path, 'wb') as _f:
                    _pickle.dump(payload, _f, protocol=_pickle.HIGHEST_PROTOCOL)
                logger.debug(
                    "Tile %s: saved smoothed VCS to %s (%.3fs)",
                    tile_str, smoothed_cache_path, smooth_time,
                )
            except Exception as _exc:
                logger.warning(
                    "Tile %s: smoothed cache save failed (%s)",
                    tile_str, _exc,
                )

        return {
            'time_step': time_step,
            't_start': t_start,
            't_end': t_end,
            'smooth_time_s': smooth_time,
            'cached': False,
        }

    # --- 4c. Source selection -------------------------------------------

    def use_smoothed_sources(self, use_smoothed: bool = True) -> None:
        """Switch between raw and smoothed vectorized sources."""
        if use_smoothed:
            if self._smoothed_sources is None:
                raise RuntimeError(
                    "No smoothed sources; call smooth_sources() first"
                )
            self._active_sources = self._smoothed_sources
        else:
            if self._vec_sources is None:
                raise RuntimeError(
                    "No raw sources; call init_vectorized_sources() first"
                )
            self._active_sources = self._vec_sources
        # Changing active sources invalidates the step-column table.
        self._step_col_table = None

    # --- A2: Step-column table methods ---------------------------------

    def get_period_info(self) -> Dict[str, Any]:
        """Return period info from active VCS (for coordinator tier selection).

        Returns empty dict when no active sources are loaded.
        """
        if self._active_sources is None:
            return {'has_single_period': False, 'n_active_source_rows': 0}
        return self._active_sources.get_period_info()

    def precompute_step_columns(
        self,
        t_start: float,
        dt: float,
        n_steps: int,
        use_step_columns: Optional[bool] = None,
        max_table_mb: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Build per-tile step-column current table for the time loop.

        Stores source-carrying rows only (sparse row index + dense values).
        DC contributions are baked into every column.  The table is reused
        across steps via :meth:`_get_current_array_for_step`.

        Build paths (in priority order):
        1. Smoothed-grid direct scatter (production flow): when active sources
           are smoothed PWLs on the uniform ``actual_step`` grid, reads values
           directly from the existing smoothed arrays — no evaluate_at_time.
        2. Vectorized batch build: ``evaluate_at_times(t_grid)`` evaluating
           all sources at all ``m`` phase-grid times in one vectorized pass.
        3. Scalar fallback: ``m`` ``evaluate_at_time`` calls (used only when
           ``n_steps < 32``).

        Tier selection:
        - **Tier a** (single period, integral P/dt, within memory budget):
          full phase table of ``m = round(P/dt)`` columns; per step uses
          column ``(step_idx + phase0) % m``.
        - **Tier c** (everything else): chunked window, ``W`` steps at a time,
          rebuilt when the step leaves the window.

        Args:
            t_start: Simulation start time (s).
            dt: Time step (s).
            n_steps: Total number of time steps.
            use_step_columns: Override the worker's ``_use_step_columns``
                setting for this call.  None = use worker setting.
            max_table_mb: Override ``_max_table_mb``.  None = use worker
                setting.

        Returns:
            Info dict: ``{tier, m, phase0, memory_mb, build_path, n_src_nodes}``.
        """
        # Apply overrides
        use_sc = self._use_step_columns if use_step_columns is None else bool(use_step_columns)
        max_mb = self._max_table_mb if max_table_mb is None else float(max_table_mb)

        if not use_sc or self._active_sources is None:
            self._step_col_table = None
            return {'tier': 'disabled', 'n_src_nodes': 0, 'memory_mb': 0.0}

        vcs = self._active_sources
        pinfo = vcs.get_period_info()

        # Determine tier
        tier = 'chunked'  # default
        m = 0
        phase0 = 0

        if pinfo['has_single_period']:
            P = pinfo['single_period']
            integral_fn = pinfo['p_over_dt_is_integral']
            is_int, m_candidate = integral_fn(dt)
            if is_int and m_candidate > 0:
                est_mb_fn = pinfo['est_table_mb']
                if est_mb_fn(m_candidate) <= max_mb:
                    tier = 'phase'
                    m = m_candidate
                    # Phase offset: which column corresponds to step_idx=0?
                    # Step 0 evaluates at t = t_start + dt.
                    # t % P = (t_start + dt) % P.
                    # Column k corresponds to t = k*dt (one-indexed from dt).
                    # For t_start=0: column for step_idx s is s % m.
                    # For general t_start: phase0 = round(t_start/dt) % m
                    if P > 0:
                        phase0 = int(round(t_start / dt)) % m
                    else:
                        phase0 = 0

        if tier == 'phase':
            info = self._build_phase_table(t_start, dt, m, phase0, n_steps)
        else:
            # Tier c: chunked window
            W = min(512, n_steps)
            info = self._build_chunked_window(t_start, dt, n_steps, W)

        return info

    def _build_phase_table(
        self,
        t_start: float,
        dt: float,
        m: int,
        phase0: int,
        n_steps: int,
    ) -> Dict[str, Any]:
        """Build a full phase table of m columns (tier a).

        Selects build path:
        1. Smoothed-grid direct scatter if sources are aligned.
        2. evaluate_at_times_for_rows — row-sparse vectorized batch build.
        3. Scalar fallback (m < 32).

        All paths work on the pre-computed ``src_rows_candidate`` set so that
        only source-carrying rows are ever allocated.  Peak build memory is
        ``O(max(n_pulses, n_pwls) * m + n_src_rows * m)`` rather than
        ``O(n_nodes * m)``.  The stored table is always row-sparse.
        """
        import time as _time
        t0 = _time.perf_counter()

        vcs = self._active_sources
        n_nodes = vcs.n_nodes

        # Pre-compute candidate source-carrying rows from VCS metadata.
        # This is cheap (array ops, no evaluation) and avoids the O(n_nodes×m)
        # dense intermediate for the batch and direct-scatter build paths.
        src_rows_candidate = vcs.get_src_node_indices()

        # Build path selection
        build_path = self._select_build_path(dt, m)

        # True build-time peak (what a naive full build would cost)
        peak_build_mb_full = n_nodes * m * 8 / 1e6

        if build_path == 'direct_scatter':
            # Sparse direct scatter: uses src_rows_candidate, returns (n_src, m)
            C_sparse, src_rows = self._build_via_direct_scatter(
                m, phase0, dt, t_start, src_rows_candidate
            )
            peak_build_mb = len(src_rows_candidate) * m * 8 / 1e6
        elif m < 32 or build_path == 'scalar':
            # Scalar fallback: small m → full build acceptable
            C_dense_full = self._build_via_scalar_fallback(dt, m)
            row_max = np.abs(C_dense_full).max(axis=1)
            src_rows = np.where(row_max > 0)[0].astype(np.int32)
            C_sparse = np.asfortranarray(C_dense_full[src_rows, :])
            peak_build_mb = peak_build_mb_full  # full array was materialized
        else:
            # Path ii: row-sparse vectorized batch build.
            # Build ONE FULL PERIOD [dt, 2*dt, ..., m*dt] (origin at t=0)
            # but ONLY for src_rows_candidate — avoids the n_nodes×m spike.
            # phase0 encodes where t_start falls in this period; per-step
            # column lookup uses col = (step_idx + phase0) % m.
            t_grid = np.arange(1, m + 1, dtype=np.float64) * dt
            # (n_candidate, m) — much smaller than (n_nodes, m)
            C_pre = vcs.evaluate_at_times_for_rows(t_grid, src_rows_candidate)
            # Trim to actually-nonzero rows
            row_max = np.abs(C_pre).max(axis=1)
            nonzero = row_max > 0
            src_rows = src_rows_candidate[nonzero]
            C_sparse = np.asfortranarray(C_pre[nonzero, :])
            peak_build_mb = len(src_rows_candidate) * m * 8 / 1e6

        if len(src_rows) == 0:
            # No sources: store empty table
            self._step_col_table = {
                'tier': 'phase',
                'm': m,
                'phase0': phase0,
                'src_rows': src_rows,
                'C_dense': np.empty((0, m), dtype=np.float64, order='F'),
            }
        else:
            self._step_col_table = {
                'tier': 'phase',
                'm': m,
                'phase0': phase0,
                'src_rows': src_rows,
                'C_dense': C_sparse,  # (n_src, m), Fortran order
            }

        # Pre-allocate current buffer for buffer reuse (avoid np.zeros per step)
        if self._current_buf is None or len(self._current_buf) != n_nodes:
            self._current_buf = np.zeros(n_nodes, dtype=np.float64)

        build_time = _time.perf_counter() - t0
        n_src = len(src_rows)
        mem_mb = n_src * m * 8 / 1e6
        logger.debug(
            "Tile %s precompute_step_columns: tier=phase m=%d phase0=%d "
            "n_src_nodes=%d mem=%.2fMB peak_build=%.2fMB (full=%.2fMB) "
            "path=%s build=%.3fs",
            self._tile_data.tile_id if self._tile_data else '?',
            m, phase0, n_src, mem_mb, peak_build_mb, peak_build_mb_full,
            build_path, build_time,
        )
        return {
            'tier': 'phase',
            'm': m,
            'phase0': phase0,
            'n_src_nodes': n_src,
            'memory_mb': mem_mb,
            'peak_build_mb': peak_build_mb,
            'build_path': build_path,
            'build_time_s': build_time,
        }

    def _build_chunked_window(
        self,
        t_start: float,
        dt: float,
        n_steps: int,
        W: int,
    ) -> Dict[str, Any]:
        """Build a chunked-window table (tier c).

        The first window of W steps is built eagerly; subsequent windows are
        built on demand in :meth:`_get_current_array_for_step`.

        Uses the row-sparse build to avoid the ``(n_nodes, W)`` full-dense
        intermediate: only ``src_rows_candidate`` rows are allocated.
        Peak build memory is ``O(max(n_pulses, n_pwls) * W + n_src * W)``
        rather than ``O(n_nodes * W)``.
        """
        import time as _time
        t0 = _time.perf_counter()

        vcs = self._active_sources
        n_nodes = vcs.n_nodes

        W = min(W, n_steps)

        # Pre-compute candidate source rows (cheap, metadata-only)
        src_rows_candidate = vcs.get_src_node_indices()

        # Build first window row-sparse to avoid n_nodes × W allocation
        t_grid = t_start + np.arange(1, W + 1, dtype=np.float64) * dt
        # (n_candidate, W) — much smaller than (n_nodes, W)
        C_pre = vcs.evaluate_at_times_for_rows(t_grid, src_rows_candidate)

        row_max = np.abs(C_pre).max(axis=1)
        nonzero = row_max > 0
        src_rows = src_rows_candidate[nonzero]

        if len(src_rows) == 0:
            C_sparse = np.empty((0, W), dtype=np.float64, order='F')
        else:
            C_sparse = np.asfortranarray(C_pre[nonzero, :])

        self._step_col_table = {
            'tier': 'chunked',
            'W': W,
            't_start': t_start,
            'dt': dt,
            'n_steps': n_steps,
            # Store candidate rows so window rebuilds use the same candidate set.
            # Per-window src_rows may differ (late-active sources), but
            # candidate set never shrinks.
            '_src_rows_candidate': src_rows_candidate,
            'src_rows': src_rows,
            'C_dense': C_sparse,
            '_chunk_start': 0,   # step_idx of first column in current chunk
        }

        if self._current_buf is None or len(self._current_buf) != n_nodes:
            self._current_buf = np.zeros(n_nodes, dtype=np.float64)

        build_time = _time.perf_counter() - t0
        n_src = len(src_rows)
        mem_mb = n_src * W * 8 / 1e6
        peak_build_mb = len(src_rows_candidate) * W * 8 / 1e6
        peak_build_mb_full = n_nodes * W * 8 / 1e6
        logger.debug(
            "Tile %s precompute_step_columns: tier=chunked W=%d "
            "n_src_nodes=%d mem=%.2fMB peak_build=%.2fMB (full=%.2fMB) build=%.3fs",
            self._tile_data.tile_id if self._tile_data else '?',
            W, n_src, mem_mb, peak_build_mb, peak_build_mb_full, build_time,
        )
        return {
            'tier': 'chunked',
            'W': W,
            'n_src_nodes': n_src,
            'memory_mb': mem_mb,
            'peak_build_mb': peak_build_mb,
            'build_path': 'evaluate_at_times_for_rows',
            'build_time_s': build_time,
        }

    def _select_build_path(self, dt: float, m: int) -> str:
        """Select build path for phase table: 'direct_scatter', 'batch', or 'scalar'."""
        if m < 32:
            return 'scalar'
        vcs = self._active_sources
        if vcs is None:
            return 'scalar'
        # Path i: smoothed-grid direct scatter
        # Conditions: n_pulses==0 (all converted to PWL after smoothing),
        # all_zero_delay, single period, and PWL times align to dt grid.
        if vcs.n_pulses == 0 and vcs.n_pwls > 0:
            # Check delay and period
            pinfo = vcs.get_period_info()
            if pinfo['all_zero_pwl_delay'] and pinfo['has_single_period']:
                P = pinfo['single_period']
                # Check integrality
                is_int, m_check = pinfo['p_over_dt_is_integral'](dt)
                if is_int and m_check == m:
                    # Check grid alignment: first PWL time ≈ 0 (t_start offset baked in)
                    # Actually we check if the step between consecutive samples ≈ dt
                    if vcs.n_pwl_points > 0 and vcs.n_pwls > 0:
                        # Sample the first PWL row's times
                        off0 = int(vcs.pwl_offset[0])
                        cnt0 = int(vcs.pwl_count[0])
                        if cnt0 >= 2:
                            sample_step = float(vcs.pwl_times[off0 + 1] - vcs.pwl_times[off0])
                            actual_step = P / m
                            # Also require that the first sample is at t≈0 so
                            # that sample index k+1 maps to time (k+1)*dt from
                            # origin (matching the batch/scalar build convention).
                            # If smooth_sources was called with t_start!=0 the
                            # first sample would be at t_start and this guard
                            # would fall through to the batch path.
                            first_time = float(vcs.pwl_times[off0])
                            if (abs(sample_step - actual_step) <= 1e-9 * actual_step
                                    and abs(first_time) <= 0.5 * actual_step):
                                return 'direct_scatter'
        return 'batch'

    def _build_via_direct_scatter(
        self,
        m: int,
        phase0: int,
        dt: float,
        t_start: float,
        src_rows_candidate: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build phase table via scatter-add of smoothed value arrays.

        Row-sparse: only allocates ``(n_candidate, m)`` instead of
        ``(n_nodes, m)``.  Returns ``(C_sparse, src_rows)`` where
        ``C_sparse`` is ``(n_src, m)`` Fortran-order and ``src_rows``
        is the int32 index array into the full node vector.

        Each column k stores the value at time (k+1)*dt from origin (t=0)
        for each source row.  ``phase0`` is NOT applied here; it is applied
        at lookup time in ``_get_current_array_for_step`` via
        ``col = (step_idx + phase0) % m``.

        Falls back to ``evaluate_at_times_for_rows`` on any mismatch.
        """
        try:
            from parser.current_sources import get_apply_wscale
            apply_wscale = get_apply_wscale()
        except ImportError:
            apply_wscale = True

        vcs = self._active_sources
        n_nodes = vcs.n_nodes
        n_candidate = len(src_rows_candidate)

        # node → position in src_rows_candidate (-1 if absent)
        row_positions = -np.ones(n_nodes, dtype=np.int32)
        if n_candidate > 0:
            row_positions[src_rows_candidate] = np.arange(n_candidate, dtype=np.int32)

        # DC base for candidate rows only
        dc_per_node = np.zeros(n_nodes, dtype=np.float64)
        if vcs.n_sources > 0:
            np.add.at(dc_per_node, vcs.source_node_idx, vcs.dc_values)

        # Allocate (n_candidate, m) instead of (n_nodes, m)
        C = np.empty((n_candidate, m), dtype=np.float64)
        if n_candidate > 0:
            C[:] = dc_per_node[src_rows_candidate, np.newaxis]  # DC init
        else:
            C[:] = 0.0

        # Scatter each PWL row's values into the appropriate column.
        # After smoothing, pwl_times starts near t=0 (the t_start offset is
        # "baked into" the waveform shape — pwl_times[off + i] ≈ i * actual_step).
        # The phase table is built at ORIGIN (same as the batch and scalar paths):
        # column k stores the value at time (k+1)*dt from t=0, i.e. sample index
        # (k+1) % cnt.  phase0 is NOT applied here; it is applied at lookup time
        # in _get_current_array_for_step via col = (step_idx + phase0) % m.
        try:
            for pwl_i in range(vcs.n_pwls):
                off = int(vcs.pwl_offset[pwl_i])
                cnt = int(vcs.pwl_count[pwl_i])
                node = int(vcs.pwl_node_idx[pwl_i])
                row = int(row_positions[node])
                if row < 0:
                    continue  # node not in candidate set (guard)
                wsc = (float(vcs.pwl_wscale[pwl_i])
                       if (apply_wscale and len(vcs.pwl_wscale) == vcs.n_pwls)
                       else 1.0)
                if cnt < m + 1:
                    # Not enough samples — fall back
                    raise ValueError("PWL has fewer samples than m+1")
                # Columns 0..m-1 use samples at indices 1..m (origin-based).
                for k in range(m):
                    sample_idx = (k + 1) % cnt
                    C[row, k] += wsc * float(vcs.pwl_values[off + sample_idx])
        except (ValueError, IndexError):
            # Fall back to evaluate_at_times_for_rows — row-sparse, builds at
            # ORIGIN so that col = (step_idx + phase0) % m remains valid.
            t_grid = np.arange(1, m + 1, dtype=np.float64) * dt
            C_pre = vcs.evaluate_at_times_for_rows(t_grid, src_rows_candidate)
            row_max = np.abs(C_pre).max(axis=1)
            nonzero = row_max > 0
            final_rows = src_rows_candidate[nonzero]
            return np.asfortranarray(C_pre[nonzero, :]), final_rows

        # Trim to actually-nonzero rows
        row_max = np.abs(C).max(axis=1)
        nonzero = row_max > 0
        final_rows = src_rows_candidate[nonzero]
        return np.asfortranarray(C[nonzero, :]), final_rows

    def _build_via_scalar_fallback(
        self, dt: float, m: int,
    ) -> np.ndarray:
        """Build phase table via m scalar evaluate_at_time calls (reference path).

        Evaluates at ONE FULL PERIOD [dt, 2*dt, ..., m*dt] from the origin.
        The per-step column lookup uses ``(step_idx + phase0) % m`` to map
        simulation steps back into this period grid.
        """
        vcs = self._active_sources
        n_nodes = vcs.n_nodes
        C = np.empty((n_nodes, m), dtype=np.float64)
        for k in range(m):
            t = (k + 1) * dt
            C[:, k] = vcs.evaluate_at_time(t)
        return C

    def _get_current_array_for_step(
        self, step_idx: int, t: float,
    ) -> np.ndarray:
        """Return current array for the given step (table or fallback).

        Uses the step-column table when available for O(n_src_nodes) gather
        instead of a full source evaluation.  Falls back to evaluate_at_time
        when the table is disabled or not built.

        The returned array is either ``self._current_buf`` (mutated in-place
        for buffer reuse) or a freshly allocated array from evaluate_at_time.
        Callers must NOT cache the returned reference across steps.

        Args:
            step_idx: 0-based index of the current time step.
            t: Actual time in seconds (used for fallback path only).

        Returns:
            shape ``(n_nodes,)`` current array.
        """
        tbl = self._step_col_table
        if tbl is None or self._active_sources is None:
            # Fallback: per-step evaluate_at_time
            return self._active_sources.evaluate_at_time(t) if self._active_sources else np.zeros(
                self._block_system.n_ports + self._block_system.n_interior,
                dtype=np.float64,
            )

        tier = tbl['tier']
        src_rows = tbl['src_rows']
        n_src = len(src_rows)

        if tier == 'phase':
            m = tbl['m']
            phase0 = tbl['phase0']
            col = (step_idx + phase0) % m
        else:
            # Chunked: rebuild window if step_idx is outside current chunk
            W = tbl['W']
            chunk_start = tbl['_chunk_start']
            if step_idx < chunk_start or step_idx >= chunk_start + W:
                # Rebuild window starting at step_idx using stored candidate rows.
                # The candidate set from _build_chunked_window never shrinks, so
                # sources active in later windows (zero in window 0) are included.
                t_start = tbl['t_start']
                dt = tbl['dt']
                n_steps = tbl['n_steps']
                src_rows_candidate = tbl.get('_src_rows_candidate')
                new_start = step_idx
                new_end = min(new_start + W, n_steps)
                W_actual = new_end - new_start
                t_grid = t_start + np.arange(
                    new_start + 1, new_end + 1, dtype=np.float64
                ) * dt
                vcs = self._active_sources
                if src_rows_candidate is not None and len(src_rows_candidate) > 0:
                    # Row-sparse rebuild: only evaluate candidate rows
                    C_pre = vcs.evaluate_at_times_for_rows(t_grid, src_rows_candidate)
                    new_row_max = np.abs(C_pre).max(axis=1)
                    nonzero = new_row_max > 0
                    src_rows = src_rows_candidate[nonzero]
                    if len(src_rows) > 0:
                        tbl['C_dense'] = np.asfortranarray(C_pre[nonzero, :W_actual])
                    else:
                        tbl['C_dense'] = np.empty((0, W_actual), dtype=np.float64, order='F')
                else:
                    # Fallback: full evaluation (no candidate set stored)
                    C_win = vcs.evaluate_at_times(t_grid)
                    new_row_max = np.abs(C_win).max(axis=1)
                    src_rows = np.where(new_row_max > 0)[0].astype(np.int32)
                    if len(src_rows) > 0:
                        tbl['C_dense'] = np.asfortranarray(C_win[src_rows, :W_actual])
                    else:
                        tbl['C_dense'] = np.empty((0, W_actual), dtype=np.float64, order='F')
                tbl['src_rows'] = src_rows
                tbl['_chunk_start'] = new_start
                chunk_start = new_start
            col = step_idx - chunk_start

        # Dense fill into pre-allocated buffer (no np.zeros, no np.add.at)
        buf = self._current_buf
        buf[:] = 0.0
        if n_src > 0:
            C_dense = tbl['C_dense']
            buf[src_rows] = C_dense[:, col]

        return buf

    # --- 4c'. Current node masking (for near/far decomposition) --------

    def set_current_node_mask(self, mask: Optional[np.ndarray]) -> None:
        """Set per-node current mask for spatial source filtering.

        When set, both :meth:`evaluate_and_get_reduced_rhs` and
        :meth:`get_transient_reduced_rhs` multiply ``current_array`` by
        the mask immediately after evaluating sources.

        Args:
            mask: Float64 array of shape ``(n_ports + n_interior,)``
                where ``1.0`` keeps and ``0.0`` zeroes the current at
                that node index.  Pass ``None`` to clear.
        """
        self._current_node_mask = mask

    def build_node_mask_for_window(
        self,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        inside: bool = True,
    ) -> np.ndarray:
        """Build a per-node current mask from a spatial window.

        Iterates ``port_to_idx`` and ``interior_to_idx``, parses
        coordinates from node names via
        :func:`~distributed.tile_parsing._parse_node_xy`, and returns
        a float64 mask where ``1.0`` selects the node.

        Args:
            x_min: Left edge of the spatial window.
            x_max: Right edge of the spatial window.
            y_min: Bottom edge of the spatial window.
            y_max: Top edge of the spatial window.
            inside: If ``True``, select nodes **inside** the window.
                If ``False``, select nodes **outside** the window.

        Returns:
            Float64 array of shape ``(n_ports + n_interior,)``
            suitable for :meth:`set_current_node_mask`.
        """
        from .tile_parsing import _parse_node_xy

        bs = self._block_system
        n_ports = bs.n_ports
        n_interior = bs.n_interior
        n_total = n_ports + n_interior

        mask = np.zeros(n_total, dtype=np.float64)

        for node, idx in bs.port_to_idx.items():
            x, y = _parse_node_xy(node)
            if x is None:
                # Can't determine position; include in "outside" set
                if not inside:
                    mask[idx] = 1.0
                continue
            in_window = (x_min <= x <= x_max) and (y_min <= y <= y_max)
            if in_window == inside:
                mask[idx] = 1.0

        for node, idx in bs.interior_to_idx.items():
            offset_idx = idx + n_ports
            x, y = _parse_node_xy(node)
            if x is None:
                if not inside:
                    mask[offset_idx] = 1.0
                continue
            in_window = (x_min <= x <= x_max) and (y_min <= y <= y_max)
            if in_window == inside:
                mask[offset_idx] = 1.0

        return mask

    def build_and_set_window_mask(
        self,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        inside: bool = True,
    ) -> int:
        """Build spatial mask and apply it in one call.

        Combines :meth:`build_node_mask_for_window` and
        :meth:`set_current_node_mask` to avoid two worker round-trips.

        Args:
            x_min: Left edge of the spatial window.
            x_max: Right edge of the spatial window.
            y_min: Bottom edge of the spatial window.
            y_max: Top edge of the spatial window.
            inside: If True, keep nodes inside the window. If False, keep outside.

        Returns:
            Number of active (masked-in) nodes.
        """
        mask = self.build_node_mask_for_window(x_min, x_max, y_min, y_max, inside)
        self.set_current_node_mask(mask)
        return int(np.sum(mask > 0))

    # --- 4d. Quasi-static evaluate + reduced RHS -----------------------

    def evaluate_and_get_reduced_rhs(
        self,
        t: float,
        step_idx: Optional[int] = None,
    ) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        """Evaluate time-domain currents and compute reduced RHS.

        Falls back to static DC path when no VCS is loaded.  Caches the
        interior RHS vector so that the subsequent
        :meth:`recover_and_update_peaks` call recovers interior voltages
        using the same time-varying currents (not the static DC values).

        When ``step_idx`` is provided and a step-column table has been built
        (via :meth:`precompute_step_columns`), the current array is read from
        the table instead of evaluating sources at time ``t``.

        Args:
            t: Time in seconds (used when table is absent or step_idx=None).
            step_idx: 0-based step index for table lookup.  None = force
                per-step evaluate_at_time path (disables table).

        Returns:
            ``(g, total_current, stats)`` -- reduced RHS ``(n_ports,)``,
            scalar sum of currents (mA), and stats dict.
        """
        t0 = time.perf_counter()

        n_ports = self._block_system.n_ports
        n_interior = self._block_system.n_interior

        if self._active_sources is None:
            self._last_qs_rhs_i = None
            rhs, _rhs_stats = self.get_reduced_rhs()
            total = sum(self._tile_data.current_injections.values())
            eval_time = time.perf_counter() - t0
            stats = {
                'eval_time_s': eval_time,
                'n_active_sources': sum(
                    1 for v in self._tile_data.current_injections.values()
                    if v != 0.0
                ),
            }
            return rhs, total, stats

        # A2: use table lookup when step_idx provided and table is built
        if step_idx is not None and self._step_col_table is not None:
            current_array = self._get_current_array_for_step(step_idx, t)
        else:
            current_array = self._active_sources.evaluate_at_time(t)
        if self._current_node_mask is not None:
            current_array = current_array * self._current_node_mask
        total_current = float(np.sum(current_array))

        I_p = current_array[:n_ports]
        I_i = current_array[n_ports:n_ports + n_interior]

        rhs_d_p = self._rhs_dirichlet[:n_ports]
        rhs_d_i = self._rhs_dirichlet[n_ports:n_ports + n_interior]

        rhs_p = -I_p + rhs_d_p
        rhs_i = -I_i + rhs_d_i

        # Cache interior RHS for consistent recovery in recover_and_update_peaks
        self._last_qs_rhs_i = rhs_i

        if n_interior > 0 and self._block_system.lu_ii is not None:
            v_i = self._block_system.lu_ii(rhs_i)
            g = rhs_p - self._block_system.G_pi @ v_i
        else:
            g = rhs_p

        eval_time = time.perf_counter() - t0
        n_active = int(np.count_nonzero(current_array))
        stats = {
            'eval_time_s': eval_time,
            'n_active_sources': n_active,
        }

        tid = self._tile_data.tile_id if self._tile_data else '?'
        logger.debug(
            "Tile %s evaluate_and_get_reduced_rhs:\n"
            "  time: %.3fs  |  n_active_sources: %s  |  total_current: %.2f mA",
            tid, eval_time, f"{n_active:,}", total_current,
        )

        return g, total_current, stats

    # --- 4e. Transient system factorization ----------------------------

    def factor_transient_system(
        self,
        dt_scaled: float,
        method: str = 'be',
    ) -> Tuple[np.ndarray, List[str], float, Dict[str, Any]]:
        """Build and factor A = G + C_coeff * diag(C), return Schur complement.

        Args:
            dt_scaled: Time step in pico-seconds (dt_seconds * 1e12).
            method: ``'be'`` (Backward Euler) or ``'trap'`` (Trapezoidal).

        Returns:
            ``(S_A_dense, port_list, total_cap, stats)`` -- dense Schur
            complement, ordered port node names, total capacitance (fF),
            and stats dict.
        """
        from pgmath.block_system import (
            BlockMatrixSystem,
            build_grounded_capacitance_diags,
            _format_bytes,
        )
        from pgmath.schur import compute_explicit_schur
        import scipy.sparse as sp_mod

        bs = self._block_system
        n_ports = bs.n_ports
        n_interior = bs.n_interior

        # Build capacitance diagonals
        self._c_pp_diag, self._c_ii_diag, self._total_cap = (
            build_grounded_capacitance_diags(
                self._tile_data.capacitive_edges,
                bs.port_to_idx, bs.interior_to_idx,
                n_ports, n_interior,
            )
        )

        # Integration coefficient
        C_coeff = (2.0 if method == 'trap' else 1.0) / dt_scaled
        self._C_coeff = C_coeff
        self._dt_scaled = dt_scaled
        self._transient_method = method

        # _pad_port_mask is set by the coordinator via set_pad_port_mask() after
        # _precompute_port_gathers() runs in solve_transient (using the authoritative
        # interface_node_to_idx from the transient context, not _interface_nodes
        # which may include pads in some test setups).  Reset to None here so a
        # stale mask from a previous run does not carry over when
        # factor_transient_system is called again with a different port layout.
        self._pad_port_mask: Optional[np.ndarray] = None

        # A_ii = G_ii + C_coeff * diag(c_ii)
        if n_interior > 0:
            A_ii = bs.G_ii + sp_mod.diags(self._c_ii_diag * C_coeff, format='csr')
        else:
            A_ii = bs.G_ii

        # A_pp = G_pp + C_coeff * diag(c_pp)
        if n_ports > 0:
            A_pp = bs.G_pp + sp_mod.diags(self._c_pp_diag * C_coeff, format='csr')
        else:
            A_pp = bs.G_pp

        # Off-diagonal blocks unchanged (grounded caps -> C_pi = C_ip = 0)
        transient_bs = BlockMatrixSystem(
            G_pp=A_pp, G_pi=bs.G_pi, G_ip=bs.G_ip, G_ii=A_ii,
            port_nodes=list(bs.port_nodes),
            interior_nodes=list(bs.interior_nodes),
            port_to_idx=dict(bs.port_to_idx),
            interior_to_idx=dict(bs.interior_to_idx),
            lu_ii=None,
        )

        self._transient_block_system = transient_bs

        # A4: Pass the worker's symbolic cache (populated by DC factor_and_compute_schur).
        sym_cache = self._symbolic_ii if self._use_symbolic_reuse else None

        t0 = time.perf_counter()
        S_A, schur_stats = compute_explicit_schur(transient_bs, symbolic_cache=sym_cache)
        schur_time = time.perf_counter() - t0

        # A4: Persist the new symbolic cache if the full analyze path ran.
        if '_new_symbolic_cache' in schur_stats:
            self._symbolic_ii = schur_stats.pop('_new_symbolic_cache')
        # On symbolic-reuse path, self._symbolic_ii['factor'] was updated
        # in-place inside _compute_schur_partial — no assignment needed.

        # Both paths report factor timing in schur_stats: partial path
        # via 'analyze_s'+'factor_s', chunked path via 'factor_s' alone.
        factor_time = schur_stats.get('factor_s', 0) + schur_stats.get('analyze_s', 0)

        # Capacitance stats
        c_ii_cap_nodes = int(np.count_nonzero(self._c_ii_diag))
        c_pp_cap_nodes = int(np.count_nonzero(self._c_pp_diag))

        # Backend info from factor_adapter
        fa = transient_bs.factor_adapter
        if fa is not None:
            backend = fa.backend
            backend_info = fa.backend_info
        else:
            backend = 'n/a'
            backend_info = 'n/a'

        A_ii_nnz = A_ii.nnz if sp_mod.issparse(A_ii) else 0
        A_pp_nnz = A_pp.nnz if sp_mod.issparse(A_pp) else 0

        stats = {
            'factor_interior_s': factor_time,
            'compute_schur_s': schur_time,
            'total_cap_fF': self._total_cap,
            'A_ii_nnz': A_ii_nnz,
            'A_pp_nnz': A_pp_nnz,
            'schur_mem_bytes': schur_stats['schur_mem_bytes'],
            'factorization_backend': backend,
            'factorization_backend_info': backend_info,
            'n_ports': n_ports,
            'n_interior': n_interior,
            'c_ii_cap_nodes': c_ii_cap_nodes,
            'c_pp_cap_nodes': c_pp_cap_nodes,
            'C_coeff': C_coeff,
            'schur_path': schur_stats['path'],
        }

        schur_path = stats['schur_path']
        tid = self._tile_data.tile_id if self._tile_data else '?'
        logger.debug(
            "Tile %s factor_transient_system:\n"
            "  A_ii: %s x %s, nnz=%s  |  A_pp: %s x %s, nnz=%s\n"
            "  C_ii cap nodes: %s / %s  |  C_pp cap nodes: %s / %s\n"
            "  Total tile cap: %.0f fF  |  C_coeff: %.4f\n"
            "  factor_interior: %.3fs  |  backend: %s  |  path: %s\n"
            "  compute_schur: %.3fs  |  Schur: %s x %s dense (%s)",
            tid,
            f"{n_interior:,}", f"{n_interior:,}", f"{A_ii_nnz:,}",
            f"{n_ports:,}", f"{n_ports:,}", f"{A_pp_nnz:,}",
            f"{c_ii_cap_nodes:,}", f"{n_interior:,}",
            f"{c_pp_cap_nodes:,}", f"{n_ports:,}",
            self._total_cap, C_coeff,
            factor_time, backend_info, schur_path,
            schur_time, f"{S_A.shape[0]:,}", f"{S_A.shape[1]:,}",
            _format_bytes(schur_stats['schur_mem_bytes']),
        )

        return S_A, list(bs.port_nodes), self._total_cap, stats

    # --- 4f. Transient reduced RHS ------------------------------------

    def get_transient_reduced_rhs(
        self,
        t: float,
        boundary_v_old: Dict[str, float],
    ) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        """Compute reduced RHS for one transient time step (BE or TR).

        Args:
            t: Current time in seconds.
            boundary_v_old: Previous-step port voltages ``{node: V}``.

        Returns:
            ``(g, total_current, stats)`` -- reduced RHS ``(n_ports,)``,
            scalar sum of currents (mA), and stats dict.
        """
        t0 = time.perf_counter()

        if self._transient_block_system is None:
            raise RuntimeError(
                "get_transient_reduced_rhs() requires factor_transient_system()"
            )

        bs = self._block_system
        tbs = self._transient_block_system
        n_ports = bs.n_ports
        n_interior = bs.n_interior

        # Evaluate currents
        if self._active_sources is not None:
            current_array = self._active_sources.evaluate_at_time(t)
            if self._current_node_mask is not None:
                current_array = current_array * self._current_node_mask
            I_p = current_array[:n_ports]
            I_i = current_array[n_ports:n_ports + n_interior]
        else:
            I_p = np.zeros(n_ports, dtype=np.float64)
            I_i = np.zeros(n_interior, dtype=np.float64)
            for node, cur in self._tile_data.current_injections.items():
                if node in bs.port_to_idx:
                    I_p[bs.port_to_idx[node]] += cur
                elif node in bs.interior_to_idx:
                    I_i[bs.interior_to_idx[node]] += cur
            current_array = np.concatenate([I_p, I_i])

        total_current = float(np.sum(current_array))

        # Previous-step voltages
        v_p_old = np.zeros(n_ports, dtype=np.float64)
        for node, idx in bs.port_to_idx.items():
            if node in boundary_v_old:
                v_p_old[idx] = boundary_v_old[node]

        v_i_old = self._v_interior_old if self._v_interior_old is not None else np.zeros(n_interior, dtype=np.float64)

        rhs_d_p = self._rhs_dirichlet[:n_ports]
        rhs_d_i = self._rhs_dirichlet[n_ports:n_ports + n_interior]

        if self._transient_method == 'trap':
            # Bug 1 fix: zero pad-port entries in the G-history vector so that
            # pad contributions (+2*rhs_d) are not triple-counted.  The cap term
            # (C_coeff*c_pp*v_p_old) is unaffected — pad caps are always zero.
            pad_mask = getattr(self, '_pad_port_mask', None)
            if pad_mask is not None:
                v_p_hist = v_p_old.copy()
                v_p_hist[pad_mask] = 0.0
            else:
                v_p_hist = v_p_old
            f_i = (
                -2.0 * I_i
                + self._C_coeff * self._c_ii_diag * v_i_old
                - bs.G_ii @ v_i_old - bs.G_ip @ v_p_hist
                + 2.0 * rhs_d_i
            )
            f_p = (
                -2.0 * I_p
                + self._C_coeff * self._c_pp_diag * v_p_old
                - bs.G_pi @ v_i_old - bs.G_pp @ v_p_hist
                + 2.0 * rhs_d_p
            )
        else:  # Backward Euler
            f_i = -I_i + self._C_coeff * self._c_ii_diag * v_i_old + rhs_d_i
            f_p = -I_p + self._C_coeff * self._c_pp_diag * v_p_old + rhs_d_p

        self._last_f_i = f_i

        if n_interior > 0 and tbs.lu_ii is not None:
            g = f_p - tbs.G_pi @ tbs.lu_ii(f_i)
        else:
            g = f_p

        rhs_time = time.perf_counter() - t0
        rhs_norm = float(np.linalg.norm(g))
        stats = {
            'rhs_time_s': rhs_time,
            'total_current': total_current,
            'rhs_norm': rhs_norm,
        }

        tid = self._tile_data.tile_id if self._tile_data else '?'
        logger.debug(
            "Tile %s get_transient_reduced_rhs:\n"
            "  time: %.3fs  |  total_current: %.2f mA  |  rhs_norm: %.2f",
            tid, rhs_time, total_current, rhs_norm,
        )

        return g, total_current, stats

    # --- 4f'. Transient reduced RHS (array-based) ----------------------

    def get_transient_reduced_rhs_arr(
        self,
        t: float,
        v_p_old: np.ndarray,
        step_idx: Optional[int] = None,
    ) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        """Compute reduced RHS for one transient step (array-based variant).

        Array-based variant of :meth:`get_transient_reduced_rhs`: takes
        ``v_p_old`` as a pre-gathered float64 ndarray ``(n_ports,)``
        instead of a boundary voltage dict, eliminating the per-port
        Python dict-lookup loop for port voltage construction.

        When ``step_idx`` is provided and a step-column table has been built
        (via :meth:`precompute_step_columns`), the current array is gathered
        from the table instead of calling evaluate_at_time.

        Args:
            t: Current time in seconds (used when table is absent).
            v_p_old: Previous-step port voltages, shape ``(n_ports,)``.
                ``v_p_old[j]`` is the voltage at the tile's j-th port (in
                ``bs.port_nodes`` order). Pad/Dirichlet ports should
                already be filled with ``vdd`` by the coordinator.
            step_idx: 0-based step index for table lookup.  None = force
                per-step evaluate_at_time path.

        Returns:
            ``(g, total_current, stats)`` -- reduced RHS ``(n_ports,)``,
            scalar sum of currents (mA), and stats dict.
        """
        t0 = time.perf_counter()

        if self._transient_block_system is None:
            raise RuntimeError(
                "get_transient_reduced_rhs_arr() requires factor_transient_system()"
            )

        bs = self._block_system
        tbs = self._transient_block_system
        n_ports = bs.n_ports
        n_interior = bs.n_interior

        # Evaluate / gather currents
        if self._active_sources is not None:
            # A2: use table lookup when step_idx provided and table built
            if step_idx is not None and self._step_col_table is not None:
                current_array = self._get_current_array_for_step(step_idx, t)
            else:
                current_array = self._active_sources.evaluate_at_time(t)
            if self._current_node_mask is not None:
                current_array = current_array * self._current_node_mask
            I_p = current_array[:n_ports]
            I_i = current_array[n_ports:n_ports + n_interior]
        else:
            I_p = np.zeros(n_ports, dtype=np.float64)
            I_i = np.zeros(n_interior, dtype=np.float64)
            for node, cur in self._tile_data.current_injections.items():
                if node in bs.port_to_idx:
                    I_p[bs.port_to_idx[node]] += cur
                elif node in bs.interior_to_idx:
                    I_i[bs.interior_to_idx[node]] += cur
            current_array = np.concatenate([I_p, I_i])

        total_current = float(np.sum(current_array))

        # v_p_old: direct ndarray input — no per-port dict-lookup loop needed
        v_i_old = (
            self._v_interior_old
            if self._v_interior_old is not None
            else np.zeros(n_interior, dtype=np.float64)
        )

        rhs_d_p = self._rhs_dirichlet[:n_ports]
        rhs_d_i = self._rhs_dirichlet[n_ports:n_ports + n_interior]

        if self._transient_method == 'trap':
            # Bug 1 fix: zero pad-port entries in the G-history vector so that
            # pad contributions (+2*rhs_d) are not triple-counted.  The cap term
            # (C_coeff*c_pp*v_p_old) is unaffected — pad caps are always zero.
            pad_mask = getattr(self, '_pad_port_mask', None)
            if pad_mask is not None:
                v_p_hist = v_p_old.copy()
                v_p_hist[pad_mask] = 0.0
            else:
                v_p_hist = v_p_old
            f_i = (
                -2.0 * I_i
                + self._C_coeff * self._c_ii_diag * v_i_old
                - bs.G_ii @ v_i_old - bs.G_ip @ v_p_hist
                + 2.0 * rhs_d_i
            )
            f_p = (
                -2.0 * I_p
                + self._C_coeff * self._c_pp_diag * v_p_old
                - bs.G_pi @ v_i_old - bs.G_pp @ v_p_hist
                + 2.0 * rhs_d_p
            )
        else:  # Backward Euler
            f_i = -I_i + self._C_coeff * self._c_ii_diag * v_i_old + rhs_d_i
            f_p = -I_p + self._C_coeff * self._c_pp_diag * v_p_old + rhs_d_p

        self._last_f_i = f_i

        if n_interior > 0 and tbs.lu_ii is not None:
            g = f_p - tbs.G_pi @ tbs.lu_ii(f_i)
        else:
            g = f_p

        rhs_time = time.perf_counter() - t0
        rhs_norm = float(np.linalg.norm(g))
        stats = {
            'rhs_time_s': rhs_time,
            'total_current': total_current,
            'rhs_norm': rhs_norm,
        }

        return g, total_current, stats

    # --- 4g. Transient interior recovery + state update ----------------

    def get_transient_interior_voltages(
        self,
        boundary_voltages_dict: Dict[str, float],
    ) -> Dict[str, float]:
        """Recover interior voltages and update state for next step.

        Args:
            boundary_voltages_dict: Solved port voltages for current step.

        Returns:
            Dict mapping all tile nodes -> voltage.
        """
        if self._transient_block_system is None:
            raise RuntimeError(
                "get_transient_interior_voltages() requires factor_transient_system()"
            )
        if self._last_f_i is None:
            raise RuntimeError(
                "get_transient_interior_voltages() needs a preceding "
                "get_transient_reduced_rhs() call"
            )

        tbs = self._transient_block_system
        bs = self._block_system
        n_ports = bs.n_ports

        v_p = np.zeros(n_ports, dtype=np.float64)
        for node, idx in bs.port_to_idx.items():
            if node in boundary_voltages_dict:
                v_p[idx] = boundary_voltages_dict[node]

        if bs.n_interior > 0 and tbs.lu_ii is not None:
            v_i = tbs.lu_ii(self._last_f_i - tbs.G_ip @ v_p)
        else:
            v_i = np.array([], dtype=np.float64)

        self._v_interior_old = v_i.copy()

        all_voltages: Dict[str, float] = {}
        for i, node in enumerate(bs.interior_nodes):
            all_voltages[node] = float(v_i[i])
        for node in bs.port_nodes:
            if node in boundary_voltages_dict:
                all_voltages[node] = boundary_voltages_dict[node]
        return all_voltages

    # --- 4h. State management ------------------------------------------

    def set_initial_voltages(self, voltages: Dict[str, float]) -> None:
        """Initialize time-stepping state from a voltage dict."""
        bs = self._block_system
        v_i = np.zeros(bs.n_interior, dtype=np.float64)
        for node, idx in bs.interior_to_idx.items():
            if node in voltages:
                v_i[idx] = voltages[node]
        self._v_interior_old = v_i

    def recover_and_set_initial_voltages(
        self,
        boundary_voltages_dict: Dict[str, float],
    ) -> Dict[str, Any]:
        """Recover DC interior voltages and set as transient IC in one step.

        Combines get_interior_voltages + set_initial_voltages without the
        coordinator round-trip.  The interior voltage array stays on the
        worker.

        Args:
            boundary_voltages_dict: Port/boundary voltages from DC interface
                solve.

        Returns:
            Stats dict with recovery metadata.
        """
        bs = self._block_system
        n_interior = bs.n_interior

        port_voltages = np.zeros(bs.n_ports, dtype=np.float64)
        for node, idx in bs.port_to_idx.items():
            if node in boundary_voltages_dict:
                port_voltages[idx] = boundary_voltages_dict[node]

        # Bug 2 fix: use the interior RHS cached by the preceding
        # evaluate_and_get_reduced_rhs(t=0).  That RHS embeds VCS currents at
        # t_start, exactly matching what the coordinator used for the interface
        # solve.  Without this fix, the interior IC uses static
        # tile_data.current_injections while the interface IC already reflects
        # VCS(t_start) — an inconsistency when VCS(t_start) != static.
        #
        # Spec-deviation note (Bug 2 root cause): the original spec described
        # Bug 2 as a "Schur-consistent time-loop history reformulation" issue.
        # Investigation confirmed the time-loop history formula was already
        # correct (matching flat transient_solver.py:858-859).  The actual
        # root cause was this IC inconsistency between the interface and interior
        # recoveries, which is what this fix addresses.  Empirically validated:
        # distributed TR diff on netlist_sampled ~1.4e-15 V (machine precision)
        # vs the pre-fix ~0.5 mV.
        #
        # INTENTIONAL SCOPE EXPANSION: this fix is NOT method-gated and applies
        # to both TR and BE IC recovery — a deliberate decision because the IC
        # inconsistency is equally present in both methods (TR amplifies it via
        # the stiff-node period-2 mode; BE damps it quickly).  On netlist_sampled
        # BE improved ~3.6e-11 -> ~1.6e-15 V post-fix (within the 1e-8 V spec;
        # baseline diff_qs_vs_be_uV updated from 11657.45 to 11657.47 µV).
        # With static DC loads (no VCS), both paths give identical results.
        if n_interior > 0 and bs.lu_ii is not None and self._last_qs_rhs_i is not None:
            rhs_i = self._last_qs_rhs_i - bs.G_ip @ port_voltages
            v_i = bs.lu_ii(rhs_i)
        else:
            from pgmath.block_system import recover_bottom_voltages
            interior_voltages = recover_bottom_voltages(
                bs, port_voltages,
                self._tile_data.current_injections,
                self._rhs_dirichlet,
            )
            v_i = np.zeros(n_interior, dtype=np.float64)
            for node, idx in bs.interior_to_idx.items():
                if node in interior_voltages:
                    v_i[idx] = interior_voltages[node]

        self._v_interior_old = v_i
        return {'n_interior': n_interior, 'n_ports': bs.n_ports}

    def recover_and_set_initial_voltages_arr(
        self,
        v_p: np.ndarray,
    ) -> Dict[str, Any]:
        """Recover DC interior voltages and set as transient IC (array-based).

        Array-based variant of :meth:`recover_and_set_initial_voltages`:
        takes ``v_p`` as a pre-gathered float64 ndarray ``(n_ports,)``
        instead of a boundary voltage dict, eliminating the per-port
        Python dict-lookup loop for port voltage construction.

        Args:
            v_p: Pre-gathered port voltages from the DC interface solve,
                shape ``(n_ports,)``. ``v_p[j]`` is the voltage at the
                tile's j-th port (in ``bs.port_nodes`` order).
                Pad/Dirichlet ports should be filled with ``vdd``.

        Returns:
            Stats dict with recovery metadata.
        """
        bs = self._block_system
        n_interior = bs.n_interior

        # Bug 2 fix: use the interior RHS cached by the preceding
        # evaluate_and_get_reduced_rhs(t=0).  That RHS embeds VCS currents at
        # t_start, exactly matching what the coordinator used for the interface
        # solve.  Without this fix, the interior IC uses static
        # tile_data.current_injections while the interface IC already reflects
        # VCS(t_start) — an inconsistency when VCS(t_start) != static.
        # See recover_and_set_initial_voltages docstring for the full
        # spec-deviation note (IC root cause vs spec's history diagnosis) and
        # the intentional-scope-expansion note (BE + TR both fixed).
        if n_interior > 0 and bs.lu_ii is not None and self._last_qs_rhs_i is not None:
            rhs_i = self._last_qs_rhs_i - bs.G_ip @ v_p
            v_i = bs.lu_ii(rhs_i)
        else:
            from pgmath.block_system import recover_bottom_voltages
            interior_voltages = recover_bottom_voltages(
                bs, v_p,
                self._tile_data.current_injections,
                self._rhs_dirichlet,
            )
            v_i = np.zeros(n_interior, dtype=np.float64)
            for node, idx in bs.interior_to_idx.items():
                if node in interior_voltages:
                    v_i[idx] = interior_voltages[node]

        self._v_interior_old = v_i
        return {'n_interior': n_interior, 'n_ports': bs.n_ports}

    # --- 4j-bis. Pad-port mask (Bug 1 fix) --------------------------------

    def set_pad_port_mask(self, mask: np.ndarray) -> Dict[str, Any]:
        """Store the coordinator-supplied pad-port boolean mask.

        Called by solve_transient immediately after _precompute_port_gathers()
        so that the TR G-history fix knows which ports are Dirichlet pads.

        When mask[j] is True, port j is a Dirichlet pad (fixed at vdd).
        In the TR update, the coordinator already accounts for the pad
        contribution via ``+2*rhs_d_G``, so the G-history terms
        ``-G_ip@v_p_old`` and ``-G_pp@v_p_old`` must zero those entries
        to avoid triple-counting.

        **Dead-code note for coordinator-driven models**: The coordinator
        derives ``mask`` from :func:`_precompute_port_gathers`, which sets
        ``mask[j] = True`` only when a tile's port node is NOT in
        ``interface_node_to_idx``.  In a correctly-constructed
        ``DistributedPowerGridModel`` (via ``create_distributed_model`` or the
        synthetic helpers), pad nodes are placed in the package layer only and
        are not tile boundary nodes.  Therefore every element of ``mask`` is
        False and ``_pad_port_mask`` remains ``None`` for all production runs.
        A model with a pad as both a tile boundary node and absent from
        ``interface_node_to_idx`` would also trigger a length mismatch in the
        coordinator's bincount scatter (``tile_index_maps`` only covers
        interface nodes), making that configuration structurally unreachable.

        Args:
            mask: Boolean ndarray of shape ``(n_ports,)``.  True = Dirichlet.
                Expected to be all-False for coordinator-driven models.

        Returns:
            Stats dict with ``n_pad_ports`` count.
        """
        n_pad = int(mask.sum())
        if n_pad > 0:
            tile_id = getattr(getattr(self, '_tile_data', None), 'tile_id', '?')
            logger.warning(
                "set_pad_port_mask: %d pad-port(s) detected for tile %s. "
                "This path is unreachable for correctly-constructed distributed "
                "models (pad nodes should live in PackageData, not tile "
                "boundary_nodes).  _pad_port_mask will be set non-None; "
                "interior recovery via _last_f_i may be inconsistent.",
                n_pad,
                tile_id,
            )
        self._pad_port_mask = mask if mask.any() else None
        return {'n_pad_ports': n_pad}

    # --- 4k. Factorization lifecycle -------------------------------------

    def clear_dc_factorization(self) -> Dict[str, Any]:
        """Free DC LU factorization from _block_system. G matrices preserved.

        Clears lu_ii and factor_adapter from the block system, freeing the
        LU memory while preserving G_ii, G_pp, G_pi, G_ip matrices for
        potential re-factorization.

        Returns:
            Stats dict with keys: had_lu (bool), n_interior (int).
        """
        had_lu = False
        n_interior = 0
        if self._block_system is not None:
            n_interior = self._block_system.n_interior
            if self._block_system.lu_ii is not None:
                had_lu = True
                self._block_system.lu_ii = None
            if self._block_system.factor_adapter is not None:
                self._block_system.factor_adapter = None
        return {'had_lu': had_lu, 'n_interior': n_interior}

    def clear_transient_factorization(self) -> Dict[str, Any]:
        """Free transient block system entirely.

        Removes the _transient_block_system (A = G + C_coeff * C), freeing
        both matrices and LU factorization. The base _block_system (DC G matrices)
        is preserved.

        Also clears transient time-stepping state (_last_f_i and _v_interior_old).
        Callers must call set_initial_voltages() before restarting transient
        integration.

        Returns:
            Stats dict with keys: had_transient_system (bool).
        """
        had = self._transient_block_system is not None
        self._transient_block_system = None
        # Clear transient time-stepping state that depends on transient system
        self._last_f_i = None
        self._v_interior_old = None
        return {'had_transient_system': had}

    # Peak tracking methods (init_peak_tracking, update_peak_stats,
    # get_peak_stats, prebin_peak_data, get_top_k_peaks,
    # get_tracked_waveforms, recover_and_update_peaks,
    # recover_transient_and_update_peaks) are provided by
    # _PeakTrackingMixin in tile_worker_peak.py.

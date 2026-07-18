"""Time-domain coordinator methods for DistributedDDMSolver.

Provides preprocess_sources(), solve_quasi_static(), prepare_transient(),
and solve_transient(). Kept in a separate file to keep solver.py under
800 lines; surfaced on DistributedDDMSolver via _SolverTimeDomainMixin.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .interface_iterative import filter_kept_rhs
from .result import (
    DistributedQuasiStaticResult,
    DistributedSmoothedSources,
    DistributedSolveResult,
    DistributedSolverContext,
    DistributedTransientContext,
    DistributedTransientResult,
)

logger = logging.getLogger(__name__)


class _SolverTimeDomainMixin:
    """Mixin providing coordinator-side time-domain methods.

    Expects the host class to provide:
        self.model  (DistributedPowerGridModel)
        self.prepare(verbose) -> DistributedSolverContext
        self.solve_dc(context, verbose) -> DistributedSolveResult
    """

    def preprocess_sources(
        self,
        time_step: float,
        t_start: float = 0.0,
        t_end: float = 100e-9,
        smooth: Union[bool, str] = True,
        pkl_dir: Optional[str] = None,
        compact_threshold: float = 1e-12,
        chunk_size: int = 10000,
        verbose: bool = False,
    ) -> DistributedSmoothedSources:
        """Initialize vectorized current sources on all tile workers.

        Loads instanceModels files in parallel, optionally applies PWL
        smoothing, and returns a lightweight coordinator-side handle.

        Args:
            time_step: Simulation time step in seconds (also used as
                smoothing filter half-width when ``smooth=True`` or
                ``smooth='auto'``).
            t_start: Simulation start time in seconds.
            t_end: Simulation end time in seconds.
            smooth: Smoothing policy.

                - ``True`` *(default)*: always apply smoothing.
                - ``False``: skip smoothing.
                - ``'auto'``: skip smoothing when ``time_step`` ≤ the
                  smallest PWL segment duration (inter-breakpoint gap for
                  PWL waveforms; min of rise_time / fall_time / width for
                  pulses) across **all** active sources across **all**
                  tiles.  The rationale is that if the simulation step is
                  already as fine as or finer than the fastest waveform
                  feature, no aliasing can occur and the triangular
                  low-pass filter adds no information.  When all tiles
                  have only DC sources (no PWL/pulse), the minimum
                  segment duration is ``inf`` and smoothing is always
                  skipped by the ``auto`` rule.

            pkl_dir: Optional directory for VCS and smoothed-VCS pickle
                caches.  If None, derives from the first tile's ``.ckt``
                parent directory.
            compact_threshold: PWL compaction threshold passed to
                ``create_smoothed_copy`` (default 1e-12).
            chunk_size: Smoothing chunk size passed to
                ``create_smoothed_copy`` (default 10000).
            verbose: Log per-tile stats and smoothing summary.

        Returns:
            DistributedSmoothedSources handle.  The handle's
            ``per_tile_smooth_time_s`` and ``smooth_cache_hits`` fields
            are populated when smoothing runs.
        """
        timings: Dict[str, float] = {}
        model = self.model
        tile_configs = model.metadata.tile_configs

        # Derive pkl_dir for VCS cache.  Priority:
        #   1. Explicit caller argument (highest priority).
        #   2. model.pkl_dir — the output directory that contains tile_*.pkl files.
        #      This is set from bundle.pkl_dir during create_distributed_model() so
        #      that sub-tile VCS caches land in the PKL output dir, NOT in the source
        #      netlist directory (which would pollute read-only netlists and collide
        #      across different splits that share parent tile_id tuples).
        #   3. os.path.dirname(first_ckt) — legacy fallback for models created from
        #      PowerGridMetaData without a bundle (deprecated path).
        if pkl_dir is None:
            if getattr(model, 'pkl_dir', None):
                pkl_dir = model.pkl_dir
            else:
                first_ckt = tile_configs[0].ckt_path
                pkl_dir = os.path.dirname(first_ckt)

        # Derive net_filter from the first tile config
        net_filter = tile_configs[0].net_filter

        # 1. Init vectorized sources on all workers (parallel)
        t0 = time.perf_counter()
        init_args = [
            (tc.instance_path, tc.nd_path, net_filter, pkl_dir)
            for tc in tile_configs
        ]
        init_results = model.backend.call_all(
            model.workers, 'init_vectorized_sources', init_args,
        )
        timings['init_vcs'] = time.perf_counter() - t0

        # 2. Resolve smooth='auto' via a cheap worker-side probe
        if smooth == 'auto':
            t0_probe = time.perf_counter()
            min_durs = model.backend.call_all(
                model.workers, 'get_min_pwl_segment_duration',
            )
            global_min_dur = min(min_durs) if min_durs else float('inf')
            smooth_actual: bool = time_step > global_min_dur
            timings['auto_probe'] = time.perf_counter() - t0_probe
            if verbose:
                logger.info(
                    "smooth='auto': time_step=%.3gs, "
                    "global_min_segment=%.3gs, smoothing=%s",
                    time_step, global_min_dur, smooth_actual,
                )
        else:
            smooth_actual = bool(smooth)

        # 3. Optionally smooth sources on all workers (parallel)
        per_tile_smooth_time_s: Dict[Tuple[int, int], float] = {}
        smooth_cache_hits = 0
        if not smooth_actual:
            # Explicitly reset workers to raw sources so that a worker
            # previously smoothed by an earlier preprocess_sources(smooth=True)
            # call does not keep serving stale smoothed data.  This also
            # invalidates any A2 step-column table on each worker.
            model.backend.call_all(model.workers, 'use_raw_sources')

        if smooth_actual:
            t0 = time.perf_counter()
            smooth_args = [
                (time_step, t_start, t_end, pkl_dir, compact_threshold, chunk_size)
            ] * len(tile_configs)
            smooth_results = model.backend.call_all(
                model.workers, 'smooth_sources', smooth_args,
            )
            timings['smooth_sources'] = time.perf_counter() - t0

            # Collect per-tile stats
            for i, s_stats in enumerate(smooth_results):
                tid = tile_configs[i].tile_id
                per_tile_smooth_time_s[tid] = s_stats.get('smooth_time_s', 0.0)
                if s_stats.get('cached', False):
                    smooth_cache_hits += 1

            if verbose:
                if per_tile_smooth_time_s:
                    times_list = list(per_tile_smooth_time_s.values())
                    n_tiles_sm = len(times_list)
                    logger.info(
                        "Smoothing: %d tiles, cache hits=%d/%d, "
                        "per-tile wall: max=%.3fs mean=%.3fs",
                        n_tiles_sm, smooth_cache_hits, n_tiles_sm,
                        max(times_list),
                        sum(times_list) / max(n_tiles_sm, 1),
                    )

        # 4. Collect per-tile VCS init stats
        per_tile_stats: Dict[Tuple[int, int], Dict[str, int]] = {}
        total_sources = 0
        for i, stats in enumerate(init_results):
            tid = tile_configs[i].tile_id
            per_tile_stats[tid] = stats
            total_sources += stats.get('n_sources', 0)

        if verbose:
            cached_count = sum(
                1 for s in init_results if s.get('cached', False)
            )
            logger.info(
                "VCS init: %d total sources across %d tiles "
                "(%d from cache)",
                total_sources, len(tile_configs), cached_count,
            )
            for k, v in sorted(timings.items()):
                if isinstance(v, (int, float)):
                    logger.info("  %s: %.4fs", k, v)

        return DistributedSmoothedSources(
            time_step=time_step,
            t_start=t_start,
            t_end=t_end,
            smoothed=smooth_actual,
            n_tiles=len(tile_configs),
            per_tile_stats=per_tile_stats,
            timings=timings,
            per_tile_smooth_time_s=per_tile_smooth_time_s,
            smooth_cache_hits=smooth_cache_hits,
        )

    def solve_quasi_static(
        self,
        context: DistributedSolverContext,
        t_start: float = 0.0,
        t_end: float = 100e-9,
        n_points: int = 101,
        t_array: Optional[np.ndarray] = None,
        smoothed_sources: Optional[DistributedSmoothedSources] = None,
        n_worst_nodes: int = 10,
        track_nodes: Optional[List[str]] = None,
        use_step_columns: bool = True,
        verbose: bool = False,
    ) -> DistributedQuasiStaticResult:
        """Distributed quasi-static (batch DC) time-domain analysis.

        Evaluates time-varying current sources at each time point and
        solves the DC system via domain decomposition. Peak IR-drop
        tracking happens on workers; scalar summaries are reduced to
        the coordinator each step.

        Args:
            context: Pre-computed DC solver context (from ``prepare()``).
                Must be factored.
            t_start: Simulation start time in seconds.
            t_end: Simulation end time in seconds.
            n_points: Number of time points (used when ``t_array`` is None).
            t_array: Explicit time array. Overrides ``t_start/t_end/n_points``.
            smoothed_sources: Pre-processed sources handle. If None, calls
                ``preprocess_sources()`` with defaults.
            n_worst_nodes: Number of worst-drop nodes to report.
            track_nodes: Optional node names for per-step waveform recording.
            verbose: Log timing and progress.

        Returns:
            DistributedQuasiStaticResult with per-step summaries and
            lazy peak collection.

        Raises:
            ValueError: If context is not factored.
        """
        if not context.is_factored:
            raise ValueError(
                "Context is not factored. Call context.factor() or use "
                "solver.prepare() to obtain a factored context."
            )

        timings: Dict[str, Any] = {}
        model = self.model
        tile_configs = model.metadata.tile_configs

        ctx = context

        # 2. Preprocess sources if needed
        if smoothed_sources is None:
            dt = (t_end - t_start) / max(n_points - 1, 1)
            t0 = time.perf_counter()
            smoothed_sources = self.preprocess_sources(
                time_step=dt, t_start=t_start, t_end=t_end, verbose=verbose,
            )
            timings['preprocess_sources'] = time.perf_counter() - t0

        # 3. Build time array
        if t_array is None:
            t_array = np.linspace(t_start, t_end, n_points)

        # 4. Init peak tracking on workers (parallel)
        vdd = model.vdd
        t0 = time.perf_counter()
        peak_args = [(track_nodes, vdd)] * len(tile_configs)
        model.backend.call_all(
            model.workers, 'init_peak_tracking', peak_args,
        )
        timings['init_peak_tracking'] = time.perf_counter() - t0

        # 5. Time loop
        n_interface = len(ctx.interface_nodes)
        max_drops = np.zeros(len(t_array), dtype=np.float64)
        total_currents = np.zeros(len(t_array), dtype=np.float64)

        # A2: pre-compute step-column tables on workers (if enabled).
        # GUARD: step-column tables assume a UNIFORM time grid (constant dt).
        # An explicit t_array may be non-uniform (e.g. log-spaced); in that
        # case the phase/chunked column mapping silently evaluates at the wrong
        # times.  Check uniformity before enabling the table; fall back to the
        # per-step evaluate_at_time path for non-uniform grids.
        n_steps_qs = len(t_array)
        _qs_step_col_infos: List[Dict] = []
        if use_step_columns and n_steps_qs > 1:
            dt_qs = float(t_array[1] - t_array[0])
            _qs_grid_uniform = (
                dt_qs > 0 and bool(
                    np.allclose(np.diff(t_array), dt_qs, rtol=1e-9, atol=0.0)
                )
            )
            if _qs_grid_uniform:
                t0_sc = time.perf_counter()
                # QS time points are t_array[k]; step_idx=k corresponds to t_array[k].
                # precompute_step_columns builds columns for t_col_start + (k+1)*dt,
                # so we set t_col_start = t_array[0] - dt_qs to get column k → t_array[k].
                t_col_start_qs = float(t_array[0]) - dt_qs
                sc_qs_args = [
                    (t_col_start_qs, dt_qs, n_steps_qs) for _ in tile_configs
                ]
                _qs_step_col_infos = model.backend.call_all(
                    model.workers, 'precompute_step_columns', sc_qs_args,
                )
                timings['precompute_step_columns'] = time.perf_counter() - t0_sc
                if verbose:
                    tiers = [info.get('tier', '?') for info in _qs_step_col_infos]
                    logger.info(
                        "A2 QS step columns: %d tiles, tiers=%s, %.3fs",
                        len(tile_configs), tiers,
                        timings['precompute_step_columns'],
                    )
            else:
                if verbose:
                    logger.info(
                        "A2 QS step columns skipped: non-uniform t_array "
                        "(step-column table requires constant dt; dt[0]=%.4g s "
                        "but diffs range [%.4g, %.4g] s)",
                        dt_qs,
                        float(np.diff(t_array).min()),
                        float(np.diff(t_array).max()),
                    )

        # Pre-loop: build concatenated scatter index (ONCE, reused every step).
        # tile_index_maps[tid][j] = global interface idx for local port j.
        # This is the SCATTER index (interface nodes only, excludes pads) and
        # is correct for np.bincount. It is NOT used as the gather index.
        tile_ids_qs = [tc.tile_id for tc in tile_configs]
        tile_idx_maps_qs = [ctx.tile_index_maps[tid] for tid in tile_ids_qs]
        all_idx_qs = np.concatenate(tile_idx_maps_qs)
        # S2/S13: cache the D1 kept-position maps once (not per step) --
        # see filter_kept_rhs's docstring for why the pad-on-port scatter
        # needs BOTH tile_kept_port_pos AND tile_port_count, not just a
        # length check against idx_map.
        _kept_pos_map_qs = ctx.tile_kept_port_pos
        _port_count_map_qs = ctx.tile_port_count

        # Pre-loop: per-tile port-gather arrays + pad mask for boundary voltage
        # exchange. _precompute_port_gathers yields port_gather[j] = global
        # interface index for port j (or 0 for pad ports), and pad_mask[j] =
        # True where port j is a pad/Dirichlet node that must be filled with
        # vdd. Per step: v_arr = np.where(pad_mask, vdd, v_gamma[port_gather]).
        # This is the correct GATHER path; tile_idx_maps_qs / all_idx_qs above
        # is the SCATTER path only.
        port_gathers_qs, pad_masks_qs = _precompute_port_gathers(
            tile_configs, ctx.interface_node_to_idx, model.tile_boundary_nodes,
        )

        # Cumulative timing accumulators for the time loop
        cum_rhs_time = 0.0
        cum_asm_solve_time = 0.0
        cum_recovery_time = 0.0
        # Stage 1a: finer split of the "Assemble + solve" timer.  These are
        # ADDITIVE new accumulators -- cum_asm_solve_time keeps its exact
        # existing meaning (pure interface_lu/CG call time in the QS loop).
        cum_rhs_final_time = 0.0  # RHS finalization (Dirichlet add)
        cum_solve_time = 0.0      # pure interface_lu/CG call
        all_step_eval_times: List[List[float]] = []
        cg_iters_per_step: List[int] = []

        t0_loop = time.perf_counter()
        _qs_pass_step_idx = use_step_columns and bool(_qs_step_col_infos)
        for step_idx, t_val in enumerate(t_array):
            # 5a. Workers: evaluate sources + compute reduced RHS (parallel)
            # A2: pass step_idx when table is active
            t0_rhs = time.perf_counter()
            if _qs_pass_step_idx:
                rhs_results = model.backend.call_all(
                    model.workers, 'evaluate_and_get_reduced_rhs',
                    [(t_val, step_idx)] * len(tile_configs),
                )
            else:
                rhs_results = model.backend.call_all(
                    model.workers, 'evaluate_and_get_reduced_rhs',
                    [(t_val,)] * len(tile_configs),
                )

            # 5b. Coordinator: assemble global RHS via bincount scatter.
            # Concatenate per-tile g_i in tile iteration order (same order
            # as the old np.add.at loop) so bincount is bit-identical.
            step_total_current = 0.0
            step_eval_times: List[float] = []
            all_g_list_qs: List[np.ndarray] = []
            for tid, (g_i, tile_current, tile_rhs_stats) in zip(tile_ids_qs, rhs_results):
                # S2/S13 (D1 follow-up): evaluate_and_get_reduced_rhs returns
                # g_i in FULL port order (may include a tile-resident pad
                # port); tile_idx_maps_qs[i] (and hence all_idx_qs) is
                # pad-FILTERED.  Filter+validate before concatenating so a
                # pad-on-tile-port model doesn't crash bincount (or, worse,
                # silently misalign) -- see solve_dc's identical fix.
                g_i = filter_kept_rhs(
                    g_i, tid, ctx.tile_index_maps[tid],
                    _kept_pos_map_qs, _port_count_map_qs,
                    caller='solve_quasi_static',
                )
                all_g_list_qs.append(g_i)
                step_total_current += tile_current
                step_eval_times.append(
                    tile_rhs_stats.get('eval_time_s', 0.0)
                )
            all_step_eval_times.append(step_eval_times)
            global_rhs = np.bincount(
                all_idx_qs,
                weights=np.concatenate(all_g_list_qs),
                minlength=n_interface,
            )
            cum_rhs_time += time.perf_counter() - t0_rhs

            # 5c. RHS finalization (Dirichlet add) -- Stage 1a: timed separately.
            t0_rhs_final = time.perf_counter()
            global_rhs += ctx.rhs_dirichlet_interface
            total_currents[step_idx] = step_total_current
            cum_rhs_final_time += time.perf_counter() - t0_rhs_final

            # Coordinator: solve interface (pure interface_lu/CG call).
            # t0_asm starts here (post-Dirichlet-add), preserving the
            # pre-Stage-1a meaning of cum_asm_solve_time_s in the QS loop
            # (pure solve time only -- unlike the transient loop below,
            # where cum_asm_solve_time_s has always bundled rhs_final +
            # solve). See rhs_final_total_s / solve_total_s for the new
            # additive split that is consistent across both loops.
            t0_asm = time.perf_counter()
            t0_solve = time.perf_counter()
            v_gamma = ctx.interface_lu(global_rhs)
            cum_solve_time += time.perf_counter() - t0_solve
            cum_asm_solve_time += time.perf_counter() - t0_asm

            # Stage 1a: capture per-step CG iteration count when CG is active.
            if ctx._cg_solver is not None:
                cg_iters_per_step.append(
                    int(ctx._cg_solver.stats.get('last_cg_iters', 0))
                )

            # 5d. Gather per-tile boundary voltage arrays (replacing dict exchange).
            # v_arr[j] = v_gamma[port_gather[j]] for interface ports; vdd for
            # pad/Dirichlet ports (pad_mask[j]=True). Shape always (n_ports,).
            bv_per_tile_t = [
                (np.where(pad_mask, vdd, v_gamma[port_gather]), t_val)
                for port_gather, pad_mask in zip(port_gathers_qs, pad_masks_qs)
            ]

            # 5e. Workers: recover interior + update peaks (parallel)
            t0_rec = time.perf_counter()
            step_max_drops = model.backend.call_all(
                model.workers, 'recover_and_update_peaks_arr', bv_per_tile_t,
            )
            cum_recovery_time += time.perf_counter() - t0_rec

            max_drops[step_idx] = max(step_max_drops) if step_max_drops else 0.0

            if verbose and (step_idx % 10 == 0 or step_idx == len(t_array) - 1):
                _cg_str = (
                    f", cg_iters={cg_iters_per_step[-1]}"
                    if cg_iters_per_step else ""
                )
                logger.info(
                    "Step %d/%d (t=%.3e s): max_drop=%.4f V, total_I=%.2f mA%s",
                    step_idx + 1, len(t_array), t_val,
                    max_drops[step_idx], step_total_current, _cg_str,
                )

        timings['time_loop'] = time.perf_counter() - t0_loop

        # 6. Collect tracked waveforms from workers (if any)
        tracked_waveforms, tracked_ir_drop = _collect_tracked_waveforms(
            model, track_nodes, vdd, timings,
        )

        # 7. Determine global peak from per-step max drops
        peak_idx = int(np.argmax(max_drops))
        peak_ir_drop = float(max_drops[peak_idx])
        peak_time = float(t_array[peak_idx])

        timings['total'] = sum(
            v for k, v in timings.items()
            if k != 'total' and isinstance(v, (int, float))
        )

        # --- Build time-loop stats ---
        n_steps = len(t_array)
        from distributed.solver import _minmeanmax

        # Per-tile eval times: flatten all per-step lists, then min/mean/max
        flat_eval_times = [
            et for step_list in all_step_eval_times for et in step_list
        ]
        if flat_eval_times:
            elo, eavg, ehi = _minmeanmax(flat_eval_times)
        else:
            elo = eavg = ehi = 0.0

        loop_stats: Dict[str, Any] = {
            'n_steps': n_steps,
            'total_loop_s': timings['time_loop'],
            'cum_rhs_time_s': cum_rhs_time,
            'cum_asm_solve_time_s': cum_asm_solve_time,
            'cum_recovery_time_s': cum_recovery_time,
            'per_tile_eval_time': {'min': elo, 'mean': eavg, 'max': ehi},
            # Stage 1a: additive fine-grained split of the "Assemble + solve"
            # timer (cum_asm_solve_time_s keeps its exact existing meaning).
            'rhs_final_total_s': cum_rhs_final_time,
            'solve_total_s': cum_solve_time,
        }
        if cg_iters_per_step:
            loop_stats['cg_iters_per_step'] = cg_iters_per_step
            loop_stats['cg_iters_mean'] = float(
                sum(cg_iters_per_step) / len(cg_iters_per_step)
            )
            loop_stats['cg_iters_max'] = int(max(cg_iters_per_step))
        timings['loop_stats'] = loop_stats

        if verbose:
            logger.info("=== Distributed DDM Quasi-Static Solve Statistics ===")
            t_first = float(t_array[0]) * 1e9 if n_steps > 0 else t_start * 1e9
            t_last = float(t_array[-1]) * 1e9 if n_steps > 0 else t_end * 1e9
            logger.info("Time steps: %d  |  t: [%.3gns, %.3gns]",
                        n_steps, t_first, t_last)
            avg_step = timings['time_loop'] / max(n_steps, 1)
            logger.info("Time loop: %.2fs (%d steps, avg %.3fs/step)",
                        timings['time_loop'], n_steps, avg_step)
            rhs_per = cum_rhs_time / max(n_steps, 1)
            logger.info("  Evaluate + RHS:    %.3fs/step (per-tile: %.3f / %.3f / %.3f)",
                        rhs_per, elo, eavg, ehi)
            asm_per = cum_asm_solve_time / max(n_steps, 1)
            logger.info("  Assemble + solve:  %.3fs/step "
                        "(rhs_final: %.4fs/step, pure solve: %.4fs/step)",
                        asm_per, cum_rhs_final_time / max(n_steps, 1),
                        cum_solve_time / max(n_steps, 1))
            rec_per = cum_recovery_time / max(n_steps, 1)
            logger.info("  Recovery + peaks:   %.3fs/step", rec_per)
            if cg_iters_per_step:
                logger.info(
                    "  CG iterations:      mean=%.1f, max=%d",
                    loop_stats['cg_iters_mean'], loop_stats['cg_iters_max'],
                )
            logger.info("Peak IR-drop: %.4f V at t=%.3fns",
                        peak_ir_drop, peak_time * 1e9)

        return DistributedQuasiStaticResult(
            t_array=t_array,
            nominal_voltage=vdd,
            net_name=model.net_name,
            peak_ir_drop=peak_ir_drop,
            peak_ir_drop_time=peak_time,
            max_ir_drop_per_time=max_drops,
            total_current_per_time=total_currents,
            tracked_waveforms=tracked_waveforms,
            tracked_ir_drop=tracked_ir_drop,
            worst_nodes=[],
            solve_metadata={'timings': timings},
            _model=model,
        )

    def prepare_transient(
        self,
        dt: float,
        method: str = 'be',
        verbose: bool = False,
    ) -> DistributedTransientContext:
        """Factor transient A-system on tiles and assemble interface.

        Creates a DistributedTransientContext with transient factorization
        only.  Does NOT create a DC context -- callers must create one
        separately via ``prepare()`` if needed for the initial condition.

        Args:
            dt: Time step in seconds.
            method: Integration method -- ``'be'`` (Backward Euler) or
                ``'trap'`` (Trapezoidal).
            verbose: Log timing info.

        Returns:
            DistributedTransientContext with cached transient factorizations.
        """
        trans_ctx = DistributedTransientContext(
            model=self.model, topology=self._topology, dt=dt, method=method,
        )
        trans_ctx.factor(verbose=verbose)

        # Cache topology for reuse
        if self._topology is None:
            self._topology = trans_ctx.topology

        return trans_ctx

    def solve_transient(
        self,
        context: DistributedTransientContext,
        dc_context: Optional[DistributedSolverContext] = None,
        ic_voltages: Optional[Union[Dict[str, float], DistributedSolveResult]] = None,
        t_start: float = 0.0,
        t_end: float = 100e-9,
        smoothed_sources: Optional[DistributedSmoothedSources] = None,
        n_worst_nodes: int = 10,
        track_nodes: Optional[List[str]] = None,
        use_step_columns: bool = True,
        verbose: bool = False,
    ) -> DistributedTransientResult:
        """Distributed transient (RC) time-domain analysis.

        Uses DC initial condition at t=t_start, then time-steps forward
        using Backward Euler or Trapezoidal integration.

        The initial condition is determined by one of two paths:
        - **dc_context path**: Evaluates sources at t_start and solves the DC
          system using the provided dc_context.
        - **ic_voltages path**: Directly uses the provided voltage dict as
          initial condition (skips DC solve entirely).

        Args:
            context: Pre-computed transient context (from
                ``prepare_transient()``). Must be factored.
            dc_context: DC solver context for initial condition solve.
                Mutually exclusive alternative to ``ic_voltages``. Not
                consumed — caller is responsible for releasing it.
            ic_voltages: Initial voltages for all nodes. Accepts either a
                ``DistributedSolveResult`` (preferred -- extracts per-tile
                voltages without flattening) or a flat ``Dict[str, float]``
                (backward compat). Mutually exclusive with ``dc_context``.
            t_start: Simulation start time in seconds.
            t_end: Simulation end time in seconds.
            smoothed_sources: Pre-processed sources. If None, calls
                ``preprocess_sources()``.
            n_worst_nodes: Number of worst-drop nodes to report.
            track_nodes: Optional node names for per-step waveform recording.
            verbose: Log timing and progress.

        Returns:
            DistributedTransientResult with per-step summaries and
            lazy peak collection.

        Raises:
            ValueError: If context is not factored, if neither dc_context
                nor ic_voltages is provided, or if both are provided
                (they are mutually exclusive).
        """
        if not context.is_factored:
            raise ValueError(
                "Context is not factored. Call context.factor() or use "
                "solver.prepare_transient() to obtain a factored context."
            )
        if dc_context is None and ic_voltages is None:
            raise ValueError(
                "Either dc_context or ic_voltages must be provided for "
                "the initial condition."
            )
        if dc_context is not None and ic_voltages is not None:
            raise ValueError(
                "dc_context and ic_voltages are mutually exclusive. "
                "Provide one or the other, not both."
            )

        timings: Dict[str, Any] = {}
        model = self.model
        tile_configs = model.metadata.tile_configs
        vdd = model.vdd
        trans_ctx = context
        dt = trans_ctx.dt_scaled / 1e12  # Convert back to seconds
        method = trans_ctx.integration_method

        n_interface = len(trans_ctx.interface_nodes)

        # 2. Preprocess sources if needed
        if smoothed_sources is None:
            t0 = time.perf_counter()
            smoothed_sources = self.preprocess_sources(
                time_step=dt, t_start=t_start, t_end=t_end, verbose=verbose,
            )
            timings['preprocess_sources'] = time.perf_counter() - t0

        # 3. Initial condition
        if ic_voltages is not None:
            t0 = time.perf_counter()

            if isinstance(ic_voltages, DistributedSolveResult):
                # --- DistributedSolveResult path: per-tile extraction ---
                v_gamma_init = np.zeros(n_interface, dtype=np.float64)
                for i, node in enumerate(trans_ctx.interface_nodes):
                    v_gamma_init[i] = ic_voltages.interface_voltages.get(
                        node, ic_voltages.pad_voltages.get(node, vdd)
                    )
                # Per-tile voltage dicts (no flattening needed)
                init_v_args = []
                for tc in tile_configs:
                    tile_res = ic_voltages.tile_results.get(tc.tile_id)
                    if tile_res is None:
                        raise ValueError(
                            f"ic_voltages (DistributedSolveResult) is missing "
                            f"tile_id {tc.tile_id}. Ensure ic_voltages was "
                            f"produced by the same model/tiling configuration."
                        )
                    init_v_args.append((tile_res.voltages,))
            else:
                # --- Flat dict path (backward compat) ---
                v_gamma_init = np.zeros(n_interface, dtype=np.float64)
                for i, node in enumerate(trans_ctx.interface_nodes):
                    v_gamma_init[i] = ic_voltages.get(node, vdd)
                init_v_args = [(ic_voltages,)] * len(tile_configs)

            # Workers: set initial voltages from provided data
            model.backend.call_all(
                model.workers, 'set_initial_voltages', init_v_args,
            )
            timings['dc_initial'] = time.perf_counter() - t0
        else:
            # --- dc_context path: solve DC at t=t_start for initial condition ---
            t0 = time.perf_counter()
            dc_ctx = dc_context

            # Evaluate sources at t_start on workers and solve DC system
            eval_args = [(t_start,)] * len(tile_configs)
            rhs_results = model.backend.call_all(
                model.workers, 'evaluate_and_get_reduced_rhs', eval_args,
            )

            # Assemble global DC RHS via bincount scatter.
            # tile_index_maps[tid][j] = global interface idx for local port j,
            # in the same order as g_i (both derived from boundary_list order).
            dc_tile_idx_maps_ic = [
                dc_ctx.tile_index_maps[tc.tile_id] for tc in tile_configs
            ]
            # S2/S13 (D1 follow-up): g_i is FULL port order; filter+validate
            # against dc_ctx's own kept-position maps before concatenating
            # (see solve_dc's identical fix).
            _dc_kept_pos_ic = dc_ctx.tile_kept_port_pos
            _dc_port_count_ic = dc_ctx.tile_port_count
            all_g_dc_ic = np.concatenate([
                filter_kept_rhs(
                    g_i, tc.tile_id, dc_ctx.tile_index_maps[tc.tile_id],
                    _dc_kept_pos_ic, _dc_port_count_ic,
                    caller='solve_transient(dc_initial_condition)',
                )
                for tc, (g_i, _cur, _stats) in zip(tile_configs, rhs_results)
            ])
            global_rhs_init = np.bincount(
                np.concatenate(dc_tile_idx_maps_ic),
                weights=all_g_dc_ic,
                minlength=n_interface,
            )
            global_rhs_init += dc_ctx.rhs_dirichlet_interface

            # Solve DC interface for initial voltages
            v_gamma_init = dc_ctx.interface_lu(global_rhs_init)

            # Workers: recover interior voltages and set as IC (array-based).
            # Use _precompute_port_gathers so pad/Dirichlet ports (not in
            # interface_node_to_idx) are correctly filled with vdd rather than
            # gathered from v_gamma_init with the wrong (shorter) index array.
            dc_port_gathers_ic, dc_pad_masks_ic = _precompute_port_gathers(
                tile_configs, dc_ctx.interface_node_to_idx, model.tile_boundary_nodes,
            )
            bv_init_arr_args = [
                (np.where(pad_mask, vdd, v_gamma_init[port_gather]),)
                for port_gather, pad_mask in zip(dc_port_gathers_ic, dc_pad_masks_ic)
            ]
            model.backend.call_all(
                model.workers, 'recover_and_set_initial_voltages_arr',
                bv_init_arr_args,
            )
            timings['dc_initial'] = time.perf_counter() - t0

            del dc_ctx

        # 5. Init peak tracking on workers (parallel)
        t0 = time.perf_counter()
        peak_args = [(track_nodes, vdd)] * len(tile_configs)
        model.backend.call_all(
            model.workers, 'init_peak_tracking', peak_args,
        )
        timings['init_peak_tracking'] = time.perf_counter() - t0

        # Pre-loop: build concatenated scatter index (ONCE, reused every step).
        # tile_index_maps[tid][j] = global interface idx for local port j.
        # This is the SCATTER index (interface nodes only, excludes pads) and
        # is correct for np.bincount. It is NOT used as the gather index.
        tile_ids_trans = [tc.tile_id for tc in tile_configs]
        tile_idx_maps_trans = [
            trans_ctx.tile_index_maps[tid] for tid in tile_ids_trans
        ]
        all_idx_trans = np.concatenate(tile_idx_maps_trans)
        # S2/S13: cache the D1 kept-position maps once (not per step).
        _kept_pos_map_trans = trans_ctx.tile_kept_port_pos
        _port_count_map_trans = trans_ctx.tile_port_count

        # Pre-loop: per-tile port-gather arrays + pad mask for boundary voltage
        # exchange. Tiles whose port list includes pad/Dirichlet nodes must
        # have those ports filled with vdd, not gathered from v_gamma_new/old.
        port_gathers_trans, pad_masks_trans = _precompute_port_gathers(
            tile_configs, trans_ctx.interface_node_to_idx, model.tile_boundary_nodes,
        )

        # Bug 1 fix: push the authoritative pad-port masks to workers so the TR
        # G-history terms (-G_ip@v_p_old, -G_pp@v_p_old) can zero pad entries.
        # This must happen AFTER factor_transient_system (which resets the mask)
        # and BEFORE the first call to get_transient_reduced_rhs_arr.
        set_mask_args = [(mask,) for mask in pad_masks_trans]
        model.backend.call_all(model.workers, 'set_pad_port_mask', set_mask_args)

        # 6. Compute initial bv_old arrays from initial interface voltages.
        # np.where(pad_mask, vdd, v_gamma_old[port_gather]) fills pads with vdd.
        v_gamma_old = v_gamma_init.copy()
        bv_old_arr_list = [
            np.where(pad_mask, vdd, v_gamma_old[port_gather])
            for port_gather, pad_mask in zip(port_gathers_trans, pad_masks_trans)
        ]

        # 7. Time loop
        t_array = np.arange(
            t_start + dt, t_end + dt / 2, dt, dtype=np.float64,
        )

        # A2: pre-compute step-column tables on workers (if enabled).
        # Each worker builds its own phase table for current source lookup.
        n_steps_total = len(t_array)
        _step_col_table_infos: List[Dict] = []
        if use_step_columns and n_steps_total > 0:
            t0 = time.perf_counter()
            sc_args = [
                (t_start, dt, n_steps_total) for _ in tile_configs
            ]
            _step_col_table_infos = model.backend.call_all(
                model.workers, 'precompute_step_columns', sc_args,
            )
            timings['precompute_step_columns'] = time.perf_counter() - t0
            if verbose:
                tiers = [
                    (info.get('tier', '?') + '(reused)')
                    if info.get('reused') else info.get('tier', '?')
                    for info in _step_col_table_infos
                ]
                n_reused = sum(1 for info in _step_col_table_infos if info.get('reused'))
                logger.info(
                    "A2 step columns: %d tiles, tiers=%s, %.3fs%s",
                    len(tile_configs), tiers,
                    timings['precompute_step_columns'],
                    f" ({n_reused} reused)" if n_reused else "",
                )
        max_drops = np.zeros(len(t_array), dtype=np.float64)
        total_currents = np.zeros(len(t_array), dtype=np.float64)

        # Cumulative timing accumulators for the time loop
        cum_rhs_time = 0.0
        cum_asm_solve_time = 0.0
        cum_recovery_time = 0.0
        # Stage 1a: finer split of the "Assemble + solve" timer.  These are
        # ADDITIVE new accumulators -- cum_asm_solve_time keeps its exact
        # existing meaning (RHS finalization + interface_lu/CG call bundled).
        cum_rhs_final_time = 0.0  # Dirichlet + package cap/G history terms
        cum_solve_time = 0.0      # pure interface_lu/CG call
        all_step_rhs_times: List[List[float]] = []
        cg_iters_per_step: List[int] = []

        t0_loop = time.perf_counter()
        # A2: whether to pass step_idx to workers (enables table lookup)
        _pass_step_idx = use_step_columns and bool(_step_col_table_infos)
        for step_idx, t_val in enumerate(t_array):
            # 7a. Use cached bv_old_arr_list (built before loop or from prev step)
            # A2: include step_idx when table is active so workers can
            # do a direct column gather instead of evaluate_at_time.
            if _pass_step_idx:
                bv_old_per_tile = [(t_val, v_arr, step_idx) for v_arr in bv_old_arr_list]
            else:
                bv_old_per_tile = [(t_val, v_arr) for v_arr in bv_old_arr_list]

            # 7b. Workers: compute transient reduced RHS (array-based)
            t0_rhs = time.perf_counter()
            rhs_results = model.backend.call_all(
                model.workers, 'get_transient_reduced_rhs_arr', bv_old_per_tile,
            )

            # 7c. Coordinator: assemble global RHS via bincount scatter.
            # Concatenate per-tile g_i in tile iteration order (same as the
            # old np.add.at loop) so bincount produces bit-identical results.
            step_total_current = 0.0
            step_rhs_times: List[float] = []
            all_g_list_trans: List[np.ndarray] = []
            for tid, (g_i, tile_current, tile_rhs_stats) in zip(tile_ids_trans, rhs_results):
                # S2/S13 (D1 follow-up): get_transient_reduced_rhs_arr
                # returns g in FULL port order; filter+validate before
                # concatenating (see solve_dc's identical fix).
                g_i = filter_kept_rhs(
                    g_i, tid, trans_ctx.tile_index_maps[tid],
                    _kept_pos_map_trans, _port_count_map_trans,
                    caller='solve_transient',
                )
                all_g_list_trans.append(g_i)
                step_total_current += tile_current
                step_rhs_times.append(
                    tile_rhs_stats.get('rhs_time_s', 0.0)
                )
            all_step_rhs_times.append(step_rhs_times)
            global_rhs = np.bincount(
                all_idx_trans,
                weights=np.concatenate(all_g_list_trans),
                minlength=n_interface,
            )
            cum_rhs_time += time.perf_counter() - t0_rhs

            total_currents[step_idx] = step_total_current

            # Dirichlet RHS: use G-only (no cap contribution from ud block)
            # BE: + rhs_dirichlet_G, TR: + 2 * rhs_dirichlet_G
            t0_asm = time.perf_counter()
            t0_rhs_final = time.perf_counter()
            rhs_d_G = trans_ctx.rhs_dirichlet_G
            if method == 'trap':
                global_rhs += 2.0 * rhs_d_G
            else:
                global_rhs += rhs_d_G

            # T1 fix: island-penalty RHS is a SEPARATE vector (penalty*vdd
            # at penalized interface-island rows), added exactly ONCE per
            # step for BOTH BE and TR -- NOT folded into rhs_d_G above,
            # which TR already scales by 2 (that would double-count the
            # penalty forcing term under TR). Without this, penalized
            # island rows carry the 1e5 mS penalty on the diagonal with no
            # matching RHS term and decay from Vdd toward 0 within a few
            # steps. See result_factorization.py's island detection block
            # in _factor_transient_context for where this is built.
            if trans_ctx.island_penalty_rhs is not None:
                global_rhs += trans_ctx.island_penalty_rhs

            # Package capacitance history term: C_coeff * C_pkg_uu @ v_old
            if trans_ctx.C_package_uu is not None:
                global_rhs += trans_ctx.C_coeff * (
                    trans_ctx.C_package_uu @ v_gamma_old
                )

            # Trapezoidal: subtract package G_uu contribution from old step
            if method == 'trap' and trans_ctx.G_package_uu is not None:
                global_rhs -= trans_ctx.G_package_uu @ v_gamma_old
            cum_rhs_final_time += time.perf_counter() - t0_rhs_final

            # 7d. Coordinator: solve transient interface (pure interface_lu/CG call)
            t0_solve = time.perf_counter()
            v_gamma_new = trans_ctx.interface_lu(global_rhs)
            cum_solve_time += time.perf_counter() - t0_solve
            cum_asm_solve_time += time.perf_counter() - t0_asm

            # Stage 1a: capture per-step CG iteration count when CG is active.
            if trans_ctx._cg_solver is not None:
                cg_iters_per_step.append(
                    int(trans_ctx._cg_solver.stats.get('last_cg_iters', 0))
                )

            # 7e. Gather per-tile boundary voltage arrays for new step.
            # np.where(pad_mask, vdd, v_gamma_new[port_gather]) fills pads
            # with vdd and interface ports with their solved voltage.
            bv_new_arr_list = [
                np.where(pad_mask, vdd, v_gamma_new[port_gather])
                for port_gather, pad_mask in zip(port_gathers_trans, pad_masks_trans)
            ]
            bv_per_tile_t = [(v_arr, t_val) for v_arr in bv_new_arr_list]

            # 7f. Workers: recover transient interior + update peaks (array-based)
            t0_rec = time.perf_counter()
            step_max_drops = model.backend.call_all(
                model.workers,
                'recover_transient_and_update_peaks_arr',
                bv_per_tile_t,
            )
            cum_recovery_time += time.perf_counter() - t0_rec

            max_drops[step_idx] = (
                max(step_max_drops) if step_max_drops else 0.0
            )

            # 7g. Advance state
            v_gamma_old = v_gamma_new
            bv_old_arr_list = bv_new_arr_list  # cache for next step

            if verbose and (step_idx % 10 == 0 or step_idx == len(t_array) - 1):
                _cg_str = (
                    f", cg_iters={cg_iters_per_step[-1]}"
                    if cg_iters_per_step else ""
                )
                logger.info(
                    "Step %d/%d (t=%.3e s): max_drop=%.4f V, total_I=%.2f mA%s",
                    step_idx + 1, len(t_array), t_val,
                    max_drops[step_idx], step_total_current, _cg_str,
                )

        timings['time_loop'] = time.perf_counter() - t0_loop

        # 8. Collect tracked waveforms from workers (if any)
        tracked_waveforms, tracked_ir_drop = _collect_tracked_waveforms(
            model, track_nodes, vdd, timings,
        )

        # 9. Determine global peak from per-step max drops
        peak_idx = int(np.argmax(max_drops)) if len(max_drops) > 0 else 0
        peak_ir_drop = float(max_drops[peak_idx]) if len(max_drops) > 0 else 0.0
        peak_time = float(t_array[peak_idx]) if len(t_array) > 0 else t_start

        timings['total'] = sum(
            v for k, v in timings.items()
            if k != 'total' and isinstance(v, (int, float))
        )

        # --- Build time-loop stats ---
        n_steps = len(t_array)
        from distributed.solver import _minmeanmax

        # Per-tile rhs times: flatten all per-step lists, then min/mean/max
        flat_rhs_times = [
            rt for step_list in all_step_rhs_times for rt in step_list
        ]
        if flat_rhs_times:
            rlo, ravg, rhi = _minmeanmax(flat_rhs_times)
        else:
            rlo = ravg = rhi = 0.0

        loop_stats: Dict[str, Any] = {
            'n_steps': n_steps,
            'total_loop_s': timings['time_loop'],
            'cum_rhs_time_s': cum_rhs_time,
            'cum_asm_solve_time_s': cum_asm_solve_time,
            'cum_recovery_time_s': cum_recovery_time,
            'per_tile_rhs_time': {'min': rlo, 'mean': ravg, 'max': rhi},
            # Stage 1a: additive fine-grained split of the "Assemble + solve"
            # timer (cum_asm_solve_time_s keeps its exact existing meaning).
            'rhs_final_total_s': cum_rhs_final_time,
            'solve_total_s': cum_solve_time,
        }
        if cg_iters_per_step:
            loop_stats['cg_iters_per_step'] = cg_iters_per_step
            loop_stats['cg_iters_mean'] = float(
                sum(cg_iters_per_step) / len(cg_iters_per_step)
            )
            loop_stats['cg_iters_max'] = int(max(cg_iters_per_step))
        timings['loop_stats'] = loop_stats

        if verbose:
            logger.info("=== Distributed DDM Transient Solve Statistics ===")
            t_first = float(t_array[0]) if len(t_array) > 0 else t_start + dt
            t_last = float(t_array[-1]) if len(t_array) > 0 else t_end
            logger.info(
                "Time steps: %d  |  t: [%.3gps, %.3gns]  |  dt: %.3gps  |  method: %s",
                n_steps, t_first * 1e12, t_last * 1e9, dt * 1e12, method,
            )
            logger.info("DC initial condition: %.3fs", timings.get('dc_initial', 0.0))
            avg_step = timings['time_loop'] / max(n_steps, 1)
            logger.info("Time loop: %.2fs (%d steps, avg %.3fs/step)",
                        timings['time_loop'], n_steps, avg_step)
            rhs_per = cum_rhs_time / max(n_steps, 1)
            logger.info("  Transient RHS:     %.3fs/step (per-tile: %.3f / %.3f / %.3f)",
                        rhs_per, rlo, ravg, rhi)
            asm_per = cum_asm_solve_time / max(n_steps, 1)
            logger.info("  Assemble + solve:  %.3fs/step "
                        "(rhs_final: %.4fs/step, pure solve: %.4fs/step)",
                        asm_per, cum_rhs_final_time / max(n_steps, 1),
                        cum_solve_time / max(n_steps, 1))
            rec_per = cum_recovery_time / max(n_steps, 1)
            logger.info("  Interior recovery:  %.3fs/step", rec_per)
            if cg_iters_per_step:
                logger.info(
                    "  CG iterations:      mean=%.1f, max=%d",
                    loop_stats['cg_iters_mean'], loop_stats['cg_iters_max'],
                )
            logger.info("Peak IR-drop: %.4f V at t=%.3fns",
                        peak_ir_drop, peak_time * 1e9)

        return DistributedTransientResult(
            t_array=t_array,
            nominal_voltage=vdd,
            net_name=model.net_name,
            peak_ir_drop=peak_ir_drop,
            peak_ir_drop_time=peak_time,
            max_ir_drop_per_time=max_drops,
            total_current_per_time=total_currents,
            tracked_waveforms=tracked_waveforms,
            tracked_ir_drop=tracked_ir_drop,
            worst_nodes=[],
            solve_metadata={'timings': timings},
            _model=model,
            integration_method=method,
            has_capacitance=trans_ctx.has_capacitance,
        )


# ---------------------------------------------------------------------------
# Shared helpers (module-private)
# ---------------------------------------------------------------------------


def _build_bv_dicts(
    tile_configs,
    tile_index_maps,
    interface_nodes,
    tile_boundary_nodes,
    v_gamma,
    vdd,
) -> List[Dict[str, float]]:
    """Build per-tile boundary voltage dicts from solved v_gamma.

    Dead in the per-step hot loops (replaced by the array-based exchange;
    see ``recover_transient_and_update_peaks_arr``). Kept as the reference
    implementation of the gather semantics and for its unit tests. The
    adjoint path builds its own dicts independently and does not call this.
    """
    bv_list = []
    for tc in tile_configs:
        tid = tc.tile_id
        P_i = tile_index_maps[tid]
        bv = {
            interface_nodes[idx]: float(v_gamma[idx])
            for idx in P_i
        }
        for n in tile_boundary_nodes[tid]:
            if n not in bv:
                bv[n] = vdd
        bv_list.append(bv)
    return bv_list


def _precompute_port_gathers(
    tile_configs,
    interface_node_to_idx: Dict[str, int],
    tile_boundary_nodes,
):
    """Precompute per-tile port gather arrays for array-based boundary exchange.

    Computed ONCE per solve (before the time loop) and reused each step.

    For each tile i, computes:
    - ``port_gather[j]`` = global interface index for the tile's j-th port
      (in ``bs.port_nodes`` order), OR 0 (dummy) for pad/Dirichlet ports.
    - ``pad_mask[j]`` = True where port j is a pad (Dirichlet) node that
      should be filled with ``vdd`` rather than gathered from ``v_gamma``.

    Per step, the coordinator computes::

        v_arr = np.where(pad_mask, vdd, v_gamma[port_gather])

    and sends it to the worker, which uses it directly as ``v_p``.

    CRITICAL for bit-exactness: the resulting ``v_arr[j]`` == the value
    that ``_build_bv_dicts`` / dict-lookup would have placed at local port
    index j, so the numerical path is identical.

    Args:
        tile_configs: Sequence of tile config objects (with ``.tile_id``).
        interface_node_to_idx: Mapping node name → global interface index.
        tile_boundary_nodes: Per-tile list of port node names in
            ``bs.port_nodes`` order (from ``model.tile_boundary_nodes``).

    Returns:
        ``(port_gather_list, pad_mask_list)`` — one array per tile.
    """
    port_gather_list = []
    pad_mask_list = []
    for tc in tile_configs:
        tid = tc.tile_id
        boundary_list = tile_boundary_nodes[tid]  # list(bs.port_nodes)
        n_ports = len(boundary_list)

        port_gather = np.zeros(n_ports, dtype=np.int32)
        pad_mask = np.zeros(n_ports, dtype=bool)

        for j, node in enumerate(boundary_list):
            if node in interface_node_to_idx:
                port_gather[j] = interface_node_to_idx[node]
            else:
                # Pad/Dirichlet: dummy index 0 (overridden by pad_mask fill)
                pad_mask[j] = True

        port_gather_list.append(port_gather)
        pad_mask_list.append(pad_mask)

    return port_gather_list, pad_mask_list


def _collect_tracked_waveforms(
    model, track_nodes, vdd, timings,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Collect tracked waveforms from workers."""
    tracked_waveforms: Dict[str, np.ndarray] = {}
    tracked_ir_drop: Dict[str, np.ndarray] = {}
    if track_nodes:
        t0 = time.perf_counter()
        waveform_results = model.backend.call_all(
            model.workers, 'get_tracked_waveforms',
        )
        for wf_dict in waveform_results:
            for node, vals in wf_dict.items():
                arr = np.array(vals, dtype=np.float64)
                prev = tracked_waveforms.get(node)
                if prev is None or len(arr) > len(prev):
                    tracked_waveforms[node] = arr
                    tracked_ir_drop[node] = vdd - arr
        timings['collect_waveforms'] = time.perf_counter() - t0
    return tracked_waveforms, tracked_ir_drop

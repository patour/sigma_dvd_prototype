"""Time-domain coordinator methods for DistributedDDMSolver.

Provides preprocess_sources(), solve_quasi_static(), prepare_transient(),
and solve_transient(). Kept in a separate file to keep solver.py under
800 lines; surfaced on DistributedDDMSolver via _SolverTimeDomainMixin.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .result import (
    DistributedQuasiStaticResult,
    DistributedSmoothedSources,
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
        smooth: bool = True,
        pkl_dir: Optional[str] = None,
        verbose: bool = False,
    ) -> DistributedSmoothedSources:
        """Initialize vectorized current sources on all tile workers.

        Loads instanceModels files in parallel, optionally applies PWL
        smoothing, and returns a lightweight coordinator-side handle.

        Args:
            time_step: Simulation time step in seconds (also used as
                smoothing filter half-width when ``smooth=True``).
            t_start: Simulation start time in seconds.
            t_end: Simulation end time in seconds.
            smooth: If True, apply triangular low-pass smoothing.
            pkl_dir: Optional directory for VCS pickle cache. If None,
                derives from the first tile's ``.ckt`` parent directory.
            verbose: Log per-tile stats.

        Returns:
            DistributedSmoothedSources handle.
        """
        timings: Dict[str, float] = {}
        model = self.model
        tile_configs = model.metadata.tile_configs

        # Derive pkl_dir for VCS cache from tile ckt path if not given
        if pkl_dir is None:
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

        # 2. Optionally smooth sources on all workers (parallel)
        if smooth:
            t0 = time.perf_counter()
            smooth_args = [(time_step, t_start, t_end)] * len(tile_configs)
            model.backend.call_all(
                model.workers, 'smooth_sources', smooth_args,
            )
            timings['smooth_sources'] = time.perf_counter() - t0

        # 3. Collect per-tile stats
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
                logger.info("  %s: %.4fs", k, v)

        return DistributedSmoothedSources(
            time_step=time_step,
            t_start=t_start,
            t_end=t_end,
            smoothed=smooth,
            n_tiles=len(tile_configs),
            per_tile_stats=per_tile_stats,
        )

    def solve_quasi_static(
        self,
        t_start: float = 0.0,
        t_end: float = 100e-9,
        n_points: int = 101,
        t_array: Optional[np.ndarray] = None,
        context: Optional[DistributedSolverContext] = None,
        smoothed_sources: Optional[DistributedSmoothedSources] = None,
        n_worst_nodes: int = 10,
        track_nodes: Optional[List[str]] = None,
        verbose: bool = False,
    ) -> DistributedQuasiStaticResult:
        """Distributed quasi-static (batch DC) time-domain analysis.

        Evaluates time-varying current sources at each time point and
        solves the DC system via domain decomposition. Peak IR-drop
        tracking happens on workers; scalar summaries are reduced to
        the coordinator each step.

        Args:
            t_start: Simulation start time in seconds.
            t_end: Simulation end time in seconds.
            n_points: Number of time points (used when ``t_array`` is None).
            t_array: Explicit time array. Overrides ``t_start/t_end/n_points``.
            context: Pre-computed DC solver context. If None, calls ``prepare()``.
            smoothed_sources: Pre-processed sources handle. If None, calls
                ``preprocess_sources()`` with defaults.
            n_worst_nodes: Number of worst-drop nodes to report.
            track_nodes: Optional node names for per-step waveform recording.
            verbose: Log timing and progress.

        Returns:
            DistributedQuasiStaticResult with per-step summaries and
            lazy peak collection.
        """
        timings: Dict[str, Any] = {}
        model = self.model
        tile_configs = model.metadata.tile_configs

        # 1. Prepare DC context if needed
        if context is None:
            t0 = time.perf_counter()
            context = self.prepare(verbose=verbose)
            timings['prepare'] = time.perf_counter() - t0
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

        # Cumulative timing accumulators for the time loop
        cum_rhs_time = 0.0
        cum_asm_solve_time = 0.0
        cum_recovery_time = 0.0
        all_step_eval_times: List[List[float]] = []

        t0_loop = time.perf_counter()
        for step_idx, t_val in enumerate(t_array):
            # 5a. Workers: evaluate sources + compute reduced RHS (parallel)
            t0_rhs = time.perf_counter()
            rhs_results = model.backend.call_all(
                model.workers, 'evaluate_and_get_reduced_rhs',
                [(t_val,)] * len(tile_configs),
            )

            # 5b. Coordinator: assemble global RHS
            global_rhs = np.zeros(n_interface, dtype=np.float64)
            step_total_current = 0.0
            step_eval_times: List[float] = []
            for i, (g_i, tile_current, tile_rhs_stats) in enumerate(rhs_results):
                tid = tile_configs[i].tile_id
                idx_map = ctx.tile_index_maps[tid]
                np.add.at(global_rhs, idx_map, g_i)
                step_total_current += tile_current
                step_eval_times.append(
                    tile_rhs_stats.get('eval_time_s', 0.0)
                )
            all_step_eval_times.append(step_eval_times)
            cum_rhs_time += time.perf_counter() - t0_rhs

            global_rhs += ctx.rhs_dirichlet_interface
            total_currents[step_idx] = step_total_current

            # 5c. Coordinator: solve interface
            t0_asm = time.perf_counter()
            v_gamma = ctx.interface_lu(global_rhs)
            cum_asm_solve_time += time.perf_counter() - t0_asm

            # 5d. Build per-tile boundary voltage dicts
            bv_per_tile = _build_bv_dicts(
                tile_configs, ctx.tile_index_maps, ctx.interface_nodes,
                model.tile_boundary_nodes, v_gamma, vdd,
            )
            bv_per_tile_t = [(bv, t_val) for bv in bv_per_tile]

            # 5e. Workers: recover interior + update peaks (parallel)
            t0_rec = time.perf_counter()
            step_max_drops = model.backend.call_all(
                model.workers, 'recover_and_update_peaks', bv_per_tile_t,
            )
            cum_recovery_time += time.perf_counter() - t0_rec

            max_drops[step_idx] = max(step_max_drops) if step_max_drops else 0.0

            if verbose and (step_idx % 10 == 0 or step_idx == len(t_array) - 1):
                logger.info(
                    "Step %d/%d (t=%.3e s): max_drop=%.4f V, total_I=%.2f mA",
                    step_idx + 1, len(t_array), t_val,
                    max_drops[step_idx], step_total_current,
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
        }
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
            logger.info("  Assemble + solve:  %.3fs/step", asm_per)
            rec_per = cum_recovery_time / max(n_steps, 1)
            logger.info("  Recovery + peaks:   %.3fs/step", rec_per)
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

        Builds A = G + C_coeff * C on each tile, computes Schur complements,
        and assembles the global transient interface system including package
        capacitance contributions.

        Args:
            dt: Time step in seconds.
            method: Integration method -- ``'be'`` (Backward Euler) or
                ``'trap'`` (Trapezoidal).
            verbose: Log timing info.

        Returns:
            DistributedTransientContext with cached transient factorizations.
        """
        timings: Dict[str, Any] = {}
        model = self.model
        tile_configs = model.metadata.tile_configs

        # 1. DC prepare (needed for initial condition)
        t0 = time.perf_counter()
        dc_ctx = self.prepare(verbose=verbose)
        timings['dc_prepare'] = time.perf_counter() - t0

        # 2. Factor transient system on all workers (parallel)
        dt_scaled = dt * 1e12  # seconds -> ps
        C_coeff = (2.0 if method == 'trap' else 1.0) / dt_scaled

        t0 = time.perf_counter()
        trans_args = [(dt_scaled, method)] * len(tile_configs)
        schur_results = model.backend.call_all(
            model.workers, 'factor_transient_system', trans_args,
        )
        timings['factor_transient_tiles'] = time.perf_counter() - t0

        # Organize results by tile
        tile_schur_complements: Dict[Any, np.ndarray] = {}
        tile_port_node_lists: Dict[Any, List[str]] = {}
        total_tile_cap = 0.0
        per_tile_stats: List[Dict[str, Any]] = []
        for i, (S_A_i, port_list, tile_cap, tile_stats) in enumerate(schur_results):
            tid = tile_configs[i].tile_id
            tile_schur_complements[tid] = S_A_i
            tile_port_node_lists[tid] = port_list
            total_tile_cap += tile_cap
            per_tile_stats.append(tile_stats)

        # Coordinator-side DEBUG: per-tile transient factor details
        from solver.coupled_system import _format_bytes
        from distributed.solver import _fmt_count
        for i, ts in enumerate(per_tile_stats):
            tid = tile_configs[i].tile_id
            n_pp = ts.get('n_ports', 0)
            logger.debug(
                "Tile %s factor_transient_system:\n"
                "  A_ii: nnz=%s  |  A_pp: nnz=%s\n"
                "  C_ii cap nodes: %d / %d  |  C_pp cap nodes: %d / %d\n"
                "  Total tile cap: %.1f fF  |  C_coeff: %.4f\n"
                "  factor_interior: %.3fs  |  backend: %s\n"
                "  compute_schur: %.3fs  |  Schur: %s x %s dense (%s)  |  chunk_size: %s",
                tid,
                _fmt_count(ts.get('A_ii_nnz', 0)), _fmt_count(ts.get('A_pp_nnz', 0)),
                ts.get('c_ii_cap_nodes', 0), ts.get('n_interior', 0),
                ts.get('c_pp_cap_nodes', 0), n_pp,
                ts.get('total_cap_fF', 0), ts.get('C_coeff', 0),
                ts.get('factor_interior_s', 0), ts.get('factorization_backend_info', 'n/a'),
                ts.get('compute_schur_s', 0),
                f"{n_pp:,}", f"{n_pp:,}",
                _format_bytes(ts.get('schur_mem_bytes', 0)),
                ts.get('schur_chunk_size', 0),
            )

        # 3. Build combined package edges: resistive + effective cap edges
        t0 = time.perf_counter()
        pkg_res_edges = model.package_data.package_edges
        pkg_cap_edges = model.package_data.package_cap_edges

        combined_edges = list(pkg_res_edges)
        has_cap = total_tile_cap > 0
        for u, v, c_fF in pkg_cap_edges:
            if c_fF > 0:
                combined_edges.append((u, v, C_coeff * c_fF))
                has_cap = True

        # 4. Assemble transient interface system
        from solver.coupled_system import (
            assemble_schur_complement_system,
            build_interface_package_matrices,
            detect_interface_islands,
        )

        S_global, rhs_dirichlet_A, interface_nodes, interface_node_to_idx = (
            assemble_schur_complement_system(
                tile_schur_complements=tile_schur_complements,
                tile_port_node_lists=tile_port_node_lists,
                extra_edges=combined_edges,
                dirichlet_nodes=model.pad_nodes,
                dirichlet_voltage=model.vdd,
            )
        )

        # Also compute G-only Dirichlet RHS (without cap contributions)
        # needed for correct transient RHS formulation
        _, rhs_dirichlet_G, _, _ = assemble_schur_complement_system(
            tile_schur_complements=tile_schur_complements,
            tile_port_node_lists=tile_port_node_lists,
            extra_edges=list(pkg_res_edges),
            dirichlet_nodes=model.pad_nodes,
            dirichlet_voltage=model.vdd,
        )

        # Island detection on transient system
        S_global, rhs_dirichlet_A, island_nodes = detect_interface_islands(
            S_global, rhs_dirichlet_A, interface_nodes, interface_node_to_idx,
            pad_nodes=model.pad_nodes,
            extra_edges=combined_edges,
            dirichlet_voltage=model.vdd,
        )

        if island_nodes:
            logger.warning(
                "Transient: penalized %d interface island nodes",
                len(island_nodes),
            )

        timings['assemble_transient_interface'] = time.perf_counter() - t0

        # 5. Build package G and C matrices for transient RHS history terms
        n_interface = len(interface_nodes)
        G_pkg_uu, C_pkg_uu = build_interface_package_matrices(
            package_edges=pkg_res_edges,
            package_cap_edges=pkg_cap_edges,
            interface_node_to_idx=interface_node_to_idx,
            n_interface=n_interface,
            dirichlet_nodes=model.pad_nodes,
        )

        # 6. Factor transient interface system
        t0 = time.perf_counter()
        from solver.unified_solver import _factor_conductance_matrix

        interface_lu = _factor_conductance_matrix(S_global, verbose=verbose)
        timings['factor_transient_interface'] = time.perf_counter() - t0

        # 7. Build tile index maps
        tile_index_maps: Dict[Tuple[int, int], np.ndarray] = {}
        for tid, port_list in tile_port_node_lists.items():
            local_to_global = np.array(
                [interface_node_to_idx[n] for n in port_list
                 if n in interface_node_to_idx],
                dtype=np.int32,
            )
            tile_index_maps[tid] = local_to_global

        timings['total_prepare_transient'] = sum(
            v for k, v in timings.items()
            if k != 'total_prepare_transient' and isinstance(v, (int, float))
        )

        # --- Build solver_stats ---
        from distributed.solver import _minmeanmax, _fmt_count
        from solver.coupled_system import _sparse_mem_bytes, _format_bytes

        # Interface system stats
        n_unknowns = S_global.shape[0]
        iface_nnz = S_global.nnz
        iface_density_pct = (
            (iface_nnz / (n_unknowns * n_unknowns) * 100)
            if n_unknowns > 0 else 0.0
        )
        iface_mem_bytes = _sparse_mem_bytes(S_global)

        interface_stats: Dict[str, Any] = {
            'n_unknowns': n_unknowns,
            'nnz': iface_nnz,
            'density_pct': iface_density_pct,
            'mem_bytes': iface_mem_bytes,
            'factor_time_s': timings['factor_transient_interface'],
            'backend': interface_lu.backend,
            'backend_info': interface_lu.backend_info,
            'has_capacitance': has_cap,
            'pkg_C_uu_nnz': C_pkg_uu.nnz if C_pkg_uu is not None else 0,
            'pkg_G_uu_nnz': G_pkg_uu.nnz if G_pkg_uu is not None else 0,
        }

        # Aggregate per-tile stats (min/mean/max)
        aggregate_stats: Dict[str, Any] = {}
        if per_tile_stats:
            for key in ('A_ii_nnz', 'A_pp_nnz', 'total_cap_fF',
                        'schur_mem_bytes', 'factor_interior_s'):
                values = [s.get(key, 0) for s in per_tile_stats]
                lo, avg, hi = _minmeanmax(values)
                aggregate_stats[key] = {'min': lo, 'mean': avg, 'max': hi,
                                        'total': float(np.sum(values))}

        solver_stats: Dict[str, Any] = {
            'per_tile': per_tile_stats,
            'interface': interface_stats,
            'aggregate': aggregate_stats,
        }
        timings['solver_stats'] = solver_stats

        # --- Verbose INFO logging ---
        if verbose:
            n_tiles = len(per_tile_stats)
            logger.info("=== Distributed DDM Prepare Transient Statistics ===")
            logger.info("Method: %s  |  dt: %.1f ps  |  C_coeff: %.4f",
                        method, dt_scaled, C_coeff)
            logger.info("Tiles: %d", n_tiles)
            if aggregate_stats:
                _ag = aggregate_stats
                lo, avg, hi = _ag['A_ii_nnz']['min'], _ag['A_ii_nnz']['mean'], _ag['A_ii_nnz']['max']
                logger.info("  A_ii nnz:         %s / %s / %s  (min/mean/max)",
                            _fmt_count(lo), _fmt_count(avg), _fmt_count(hi))
                lo, avg, hi = _ag['A_pp_nnz']['min'], _ag['A_pp_nnz']['mean'], _ag['A_pp_nnz']['max']
                logger.info("  A_pp nnz:         %s / %s / %s",
                            _fmt_count(lo), _fmt_count(avg), _fmt_count(hi))
                lo, avg, hi = _ag['total_cap_fF']['min'], _ag['total_cap_fF']['mean'], _ag['total_cap_fF']['max']
                total_cap = _ag['total_cap_fF']['total']
                logger.info("  Total cap:        %.0f fF / %.0f fF / %.0f fF  (total: %.0f fF)",
                            lo, avg, hi, total_cap)
                lo, avg, hi = _ag['schur_mem_bytes']['min'], _ag['schur_mem_bytes']['mean'], _ag['schur_mem_bytes']['max']
                total_sm = _ag['schur_mem_bytes']['total']
                logger.info("  Schur memory:     %s / %s / %s  (total: %s)",
                            _format_bytes(lo), _format_bytes(avg),
                            _format_bytes(hi), _format_bytes(total_sm))
                lo, avg, hi = _ag['factor_interior_s']['min'], _ag['factor_interior_s']['mean'], _ag['factor_interior_s']['max']
                logger.info("  Factor time:      %.3fs / %.3fs / %.3fs", lo, avg, hi)
                # Backend: use the first tile's backend_info as representative
                backend_info = per_tile_stats[0].get('factorization_backend_info', 'n/a')
                logger.info("  Factor backend:   %s", backend_info)

            logger.info("Transient interface: %s unknowns, %s nnz (density %.3f%%), %s",
                        _fmt_count(n_unknowns), _fmt_count(iface_nnz),
                        iface_density_pct, _format_bytes(iface_mem_bytes))
            logger.info("  Backend: %s", interface_lu.backend_info)
            logger.info("  Factor time: %.3fs", timings['factor_transient_interface'])
            logger.info("  Has capacitance: %s", has_cap)
            logger.info("  Package C_uu nnz: %d  |  Package G_uu nnz: %d",
                        interface_stats['pkg_C_uu_nnz'],
                        interface_stats['pkg_G_uu_nnz'])

        return DistributedTransientContext(
            interface_lu=interface_lu.solve,
            interface_nodes=interface_nodes,
            interface_node_to_idx=interface_node_to_idx,
            rhs_dirichlet_interface=rhs_dirichlet_A,
            tile_index_maps=tile_index_maps,
            dc_context=dc_ctx,
            dt_scaled=dt_scaled,
            integration_method=method,
            has_capacitance=has_cap,
            rhs_dirichlet_G=rhs_dirichlet_G,
            C_package_uu=C_pkg_uu if C_pkg_uu.nnz > 0 else None,
            G_package_uu=G_pkg_uu if G_pkg_uu.nnz > 0 else None,
            removed_interface_nodes=island_nodes,
            timings=timings,
        )

    def solve_transient(
        self,
        t_start: float = 0.0,
        t_end: float = 100e-9,
        dt: float = 0.1e-9,
        method: str = 'be',
        context: Optional[DistributedTransientContext] = None,
        smoothed_sources: Optional[DistributedSmoothedSources] = None,
        n_worst_nodes: int = 10,
        track_nodes: Optional[List[str]] = None,
        verbose: bool = False,
    ) -> DistributedTransientResult:
        """Distributed transient (RC) time-domain analysis.

        Uses DC initial condition at t=t_start, then time-steps forward
        using Backward Euler or Trapezoidal integration.

        Args:
            t_start: Simulation start time in seconds.
            t_end: Simulation end time in seconds.
            dt: Time step in seconds.
            method: ``'be'`` (Backward Euler) or ``'trap'`` (Trapezoidal).
            context: Pre-computed transient context. If None, calls
                ``prepare_transient()``.
            smoothed_sources: Pre-processed sources. If None, calls
                ``preprocess_sources()``.
            n_worst_nodes: Number of worst-drop nodes to report.
            track_nodes: Optional node names for per-step waveform recording.
            verbose: Log timing and progress.

        Returns:
            DistributedTransientResult with per-step summaries and
            lazy peak collection.
        """
        timings: Dict[str, Any] = {}
        model = self.model
        tile_configs = model.metadata.tile_configs
        vdd = model.vdd

        # 1. Prepare transient context if needed
        if context is None:
            t0 = time.perf_counter()
            context = self.prepare_transient(dt, method, verbose=verbose)
            timings['prepare_transient'] = time.perf_counter() - t0
        trans_ctx = context

        # 2. Preprocess sources if needed
        if smoothed_sources is None:
            t0 = time.perf_counter()
            smoothed_sources = self.preprocess_sources(
                time_step=dt, t_start=t_start, t_end=t_end, verbose=verbose,
            )
            timings['preprocess_sources'] = time.perf_counter() - t0

        # 3. Initial condition at t=t_start using time-varying sources
        t0 = time.perf_counter()
        dc_ctx = trans_ctx.dc_context
        n_interface = len(dc_ctx.interface_nodes)

        # Evaluate sources at t_start on workers and solve DC system
        eval_args = [(t_start,)] * len(tile_configs)
        rhs_results = model.backend.call_all(
            model.workers, 'evaluate_and_get_reduced_rhs', eval_args,
        )

        # Assemble global DC RHS
        global_rhs_init = np.zeros(n_interface, dtype=np.float64)
        for i, (g_i, _, _stats) in enumerate(rhs_results):
            tid = tile_configs[i].tile_id
            idx_map = dc_ctx.tile_index_maps[tid]
            np.add.at(global_rhs_init, idx_map, g_i)
        global_rhs_init += dc_ctx.rhs_dirichlet_interface

        # Solve DC interface for initial voltages
        v_gamma_init = dc_ctx.interface_lu(global_rhs_init)

        # Recover interior voltages on workers
        bv_init_list = _build_bv_dicts(
            tile_configs, dc_ctx.tile_index_maps,
            dc_ctx.interface_nodes, model.tile_boundary_nodes,
            v_gamma_init, vdd,
        )
        bv_init_args = [(bv,) for bv in bv_init_list]
        init_voltages_results = model.backend.call_all(
            model.workers, 'get_interior_voltages', bv_init_args,
        )
        init_voltages_list = [v for v, _stats in init_voltages_results]
        timings['dc_initial'] = time.perf_counter() - t0

        # 4. Set initial voltages on workers (parallel)
        t0 = time.perf_counter()
        init_v_args = [(v,) for v in init_voltages_list]
        model.backend.call_all(
            model.workers, 'set_initial_voltages', init_v_args,
        )
        timings['set_initial_voltages'] = time.perf_counter() - t0

        # 5. Init peak tracking on workers (parallel)
        t0 = time.perf_counter()
        peak_args = [(track_nodes, vdd)] * len(tile_configs)
        model.backend.call_all(
            model.workers, 'init_peak_tracking', peak_args,
        )
        timings['init_peak_tracking'] = time.perf_counter() - t0

        # 6. Compute initial v_gamma_old from initial interface voltages
        v_gamma_old = v_gamma_init.copy()

        # 7. Time loop
        t_array = np.arange(
            t_start + dt, t_end + dt / 2, dt, dtype=np.float64,
        )
        max_drops = np.zeros(len(t_array), dtype=np.float64)
        total_currents = np.zeros(len(t_array), dtype=np.float64)

        # Cumulative timing accumulators for the time loop
        cum_rhs_time = 0.0
        cum_asm_solve_time = 0.0
        cum_recovery_time = 0.0
        all_step_rhs_times: List[List[float]] = []

        t0_loop = time.perf_counter()
        for step_idx, t_val in enumerate(t_array):
            # 7a. Build per-tile boundary_v_old dicts from v_gamma_old
            bv_old_list = _build_bv_dicts(
                tile_configs, trans_ctx.tile_index_maps,
                trans_ctx.interface_nodes, model.tile_boundary_nodes,
                v_gamma_old, vdd,
            )
            bv_old_per_tile = [(t_val, bv) for bv in bv_old_list]

            # 7b. Workers: compute transient reduced RHS (parallel)
            t0_rhs = time.perf_counter()
            rhs_results = model.backend.call_all(
                model.workers, 'get_transient_reduced_rhs', bv_old_per_tile,
            )

            # 7c. Coordinator: assemble global RHS
            global_rhs = np.zeros(n_interface, dtype=np.float64)
            step_total_current = 0.0
            step_rhs_times: List[float] = []
            for i, (g_i, tile_current, tile_rhs_stats) in enumerate(rhs_results):
                tid = tile_configs[i].tile_id
                idx_map = trans_ctx.tile_index_maps[tid]
                np.add.at(global_rhs, idx_map, g_i)
                step_total_current += tile_current
                step_rhs_times.append(
                    tile_rhs_stats.get('rhs_time_s', 0.0)
                )
            all_step_rhs_times.append(step_rhs_times)
            cum_rhs_time += time.perf_counter() - t0_rhs

            total_currents[step_idx] = step_total_current

            # Dirichlet RHS: use G-only (no cap contribution from ud block)
            # BE: + rhs_dirichlet_G, TR: + 2 * rhs_dirichlet_G
            t0_asm = time.perf_counter()
            rhs_d_G = trans_ctx.rhs_dirichlet_G
            if method == 'trap':
                global_rhs += 2.0 * rhs_d_G
            else:
                global_rhs += rhs_d_G

            # Package capacitance history term: C_coeff * C_pkg_uu @ v_old
            if trans_ctx.C_package_uu is not None:
                global_rhs += trans_ctx.C_coeff * (
                    trans_ctx.C_package_uu @ v_gamma_old
                )

            # Trapezoidal: subtract package G_uu contribution from old step
            if method == 'trap' and trans_ctx.G_package_uu is not None:
                global_rhs -= trans_ctx.G_package_uu @ v_gamma_old

            # 7d. Coordinator: solve transient interface
            v_gamma_new = trans_ctx.interface_lu(global_rhs)
            cum_asm_solve_time += time.perf_counter() - t0_asm

            # 7e. Build per-tile boundary voltage dicts for new step
            bv_new_list = _build_bv_dicts(
                tile_configs, trans_ctx.tile_index_maps,
                trans_ctx.interface_nodes, model.tile_boundary_nodes,
                v_gamma_new, vdd,
            )
            bv_per_tile_t = [(bv, t_val) for bv in bv_new_list]

            # 7f. Workers: recover transient interior + update peaks (parallel)
            t0_rec = time.perf_counter()
            step_max_drops = model.backend.call_all(
                model.workers,
                'recover_transient_and_update_peaks',
                bv_per_tile_t,
            )
            cum_recovery_time += time.perf_counter() - t0_rec

            max_drops[step_idx] = (
                max(step_max_drops) if step_max_drops else 0.0
            )

            # 7g. Advance state
            v_gamma_old = v_gamma_new

            if verbose and (step_idx % 10 == 0 or step_idx == len(t_array) - 1):
                logger.info(
                    "Step %d/%d (t=%.3e s): max_drop=%.4f V, total_I=%.2f mA",
                    step_idx + 1, len(t_array), t_val,
                    max_drops[step_idx], step_total_current,
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
        }
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
            logger.info("  Assemble + solve:  %.3fs/step", asm_per)
            rec_per = cum_recovery_time / max(n_steps, 1)
            logger.info("  Interior recovery:  %.3fs/step", rec_per)
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
    """Build per-tile boundary voltage dicts from solved v_gamma."""
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

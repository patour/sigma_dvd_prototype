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
        timings: Dict[str, float] = {}
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

        t0_loop = time.perf_counter()
        for step_idx, t_val in enumerate(t_array):
            # 5a. Workers: evaluate sources + compute reduced RHS (parallel)
            rhs_results = model.backend.call_all(
                model.workers, 'evaluate_and_get_reduced_rhs',
                [(t_val,)] * len(tile_configs),
            )

            # 5b. Coordinator: assemble global RHS
            global_rhs = np.zeros(n_interface, dtype=np.float64)
            step_total_current = 0.0
            for i, (g_i, tile_current) in enumerate(rhs_results):
                tid = tile_configs[i].tile_id
                idx_map = ctx.tile_index_maps[tid]
                np.add.at(global_rhs, idx_map, g_i)
                step_total_current += tile_current

            global_rhs += ctx.rhs_dirichlet_interface
            total_currents[step_idx] = step_total_current

            # 5c. Coordinator: solve interface
            v_gamma = ctx.interface_lu(global_rhs)

            # 5d. Build per-tile boundary voltage dicts
            bv_per_tile = _build_bv_dicts(
                tile_configs, ctx.tile_index_maps, ctx.interface_nodes,
                model.tile_boundary_nodes, v_gamma, vdd,
            )
            bv_per_tile_t = [(bv, t_val) for bv in bv_per_tile]

            # 5e. Workers: recover interior + update peaks (parallel)
            step_max_drops = model.backend.call_all(
                model.workers, 'recover_and_update_peaks', bv_per_tile_t,
            )

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
            v for k, v in timings.items() if k != 'total'
        )

        if verbose:
            logger.info("Quasi-static timing breakdown:")
            for k, v in sorted(timings.items()):
                logger.info("  %s: %.4fs", k, v)
            logger.info(
                "Peak IR-drop: %.4f V at t=%.3e s",
                peak_ir_drop, peak_time,
            )

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
        timings: Dict[str, float] = {}
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
        for i, (S_A_i, port_list, tile_cap) in enumerate(schur_results):
            tid = tile_configs[i].tile_id
            tile_schur_complements[tid] = S_A_i
            tile_port_node_lists[tid] = port_list
            total_tile_cap += tile_cap

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
            v for k, v in timings.items() if k != 'total_prepare_transient'
        )

        if verbose:
            logger.info("Prepare transient timing breakdown:")
            for k, v in sorted(timings.items()):
                logger.info("  %s: %.4fs", k, v)
            logger.info(
                "Transient interface: %d unknowns, has_cap=%s",
                n_interface, has_cap,
            )

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
        timings: Dict[str, float] = {}
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
        for i, (g_i, _) in enumerate(rhs_results):
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
        init_voltages_list = model.backend.call_all(
            model.workers, 'get_interior_voltages', bv_init_args,
        )
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
            rhs_results = model.backend.call_all(
                model.workers, 'get_transient_reduced_rhs', bv_old_per_tile,
            )

            # 7c. Coordinator: assemble global RHS
            global_rhs = np.zeros(n_interface, dtype=np.float64)
            step_total_current = 0.0
            for i, (g_i, tile_current) in enumerate(rhs_results):
                tid = tile_configs[i].tile_id
                idx_map = trans_ctx.tile_index_maps[tid]
                np.add.at(global_rhs, idx_map, g_i)
                step_total_current += tile_current

            total_currents[step_idx] = step_total_current

            # Dirichlet RHS: use G-only (no cap contribution from ud block)
            # BE: + rhs_dirichlet_G, TR: + 2 * rhs_dirichlet_G
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

            # 7e. Build per-tile boundary voltage dicts for new step
            bv_new_list = _build_bv_dicts(
                tile_configs, trans_ctx.tile_index_maps,
                trans_ctx.interface_nodes, model.tile_boundary_nodes,
                v_gamma_new, vdd,
            )
            bv_per_tile_t = [(bv, t_val) for bv in bv_new_list]

            # 7f. Workers: recover transient interior + update peaks (parallel)
            step_max_drops = model.backend.call_all(
                model.workers,
                'recover_transient_and_update_peaks',
                bv_per_tile_t,
            )

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
            v for k, v in timings.items() if k != 'total'
        )

        if verbose:
            logger.info("Transient timing breakdown:")
            for k, v in sorted(timings.items()):
                logger.info("  %s: %.4fs", k, v)
            logger.info(
                "Peak IR-drop: %.4f V at t=%.3e s", peak_ir_drop, peak_time,
            )

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

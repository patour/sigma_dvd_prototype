"""Distributed DDM solver.

Orchestrates the Schur complement domain decomposition across tiles.
Follows the prepare/solve pattern matching the existing unified solver.
Time-domain methods (quasi-static, transient) are provided by the
_SolverTimeDomainMixin in solver_td.py.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .model import DistributedPowerGridModel
from .result import (
    DistributedSolveResult,
    DistributedSolverContext,
    TileSolveResult,
)
from .solver_td import _SolverTimeDomainMixin

logger = logging.getLogger(__name__)


def _minmeanmax(values):
    """Return (min, mean, max) of a sequence of numbers."""
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.min()), float(arr.mean()), float(arr.max())


def _fmt_count(n: float) -> str:
    """Format a count with comma separators or SI suffix for large values."""
    if abs(n) >= 1e6:
        return f"{n / 1e6:.1f}M"
    return f"{n:,.0f}"


class DistributedDDMSolver(_SolverTimeDomainMixin):
    """Solver for distributed DDM. Takes DistributedPowerGridModel.

    Follows the prepare/solve pattern:
        model = create_distributed_model(metadata, backend='ray')
        solver = DistributedDDMSolver(model)
        ctx = solver.prepare()
        result = solver.solve_dc(ctx)
    """

    def __init__(self, model: DistributedPowerGridModel):
        self.model = model
        self._topology = None  # Cached DistributedTopologyContext for reuse

    def prepare(self, verbose: bool = False) -> DistributedSolverContext:
        """Factor tiles + assemble/factor interface. Expensive, reusable for batch.

        Creates a DistributedSolverContext and calls factor(). Topology is
        cached for reuse by prepare_transient().

        Returns:
            DistributedSolverContext with cached factorizations
        """
        ctx = DistributedSolverContext(
            model=self.model, topology=self._topology,
        )
        ctx.factor(verbose=verbose)
        # Cache topology for reuse by prepare_transient
        if self._topology is None:
            self._topology = ctx.topology
        return ctx

    def solve_dc(
        self,
        context: DistributedSolverContext,
        per_tile_currents: Optional[List[Dict[str, float]]] = None,
        verbose: bool = False,
    ) -> DistributedSolveResult:
        """DC solve using domain decomposition.

        Args:
            context: Pre-computed solver context (from prepare()). Must be
                factored (``context.is_factored == True``).
            per_tile_currents: Optional pre-partitioned currents, one dict per
                tile (node -> mA). If None, uses each tile's own current
                sources from parsing. Caller is responsible for partitioning
                boundary node currents to exactly one tile.
            verbose: Print timing info

        Returns:
            DistributedSolveResult with per-tile voltages

        Raises:
            ValueError: If context is not factored.
        """
        if not context.is_factored:
            raise ValueError(
                "Context is not factored. Call context.factor() or use "
                "solver.prepare() to obtain a factored context."
            )

        timings: Dict[str, float] = {}
        model = self.model

        ctx = context

        # 1. Get reduced RHS from each tile (parallel on workers)
        t0 = time.perf_counter()
        if per_tile_currents is not None:
            rhs_results = model.backend.call_all(
                model.workers, 'get_reduced_rhs',
                [(ptc,) for ptc in per_tile_currents],
            )
        else:
            rhs_results = model.backend.call_all(
                model.workers, 'get_reduced_rhs',
            )
        timings['compute_reduced_rhs'] = time.perf_counter() - t0

        # 2. Assemble global RHS: f = sum_i P_i^T g_i + rhs_dirichlet_interface
        t0 = time.perf_counter()
        n_interface = len(ctx.interface_nodes)
        global_rhs = np.zeros(n_interface, dtype=np.float64)

        per_tile_rhs_stats: List[Dict[str, Any]] = []
        for i, (g_i, rhs_stats) in enumerate(rhs_results):
            tid = model.metadata.tile_configs[i].tile_id
            idx_map = ctx.tile_index_maps[tid]
            # g_i has shape (n_ports,) from compute_reduced_rhs
            assert len(g_i) == len(idx_map), (
                f"Tile {tid}: reduced RHS length {len(g_i)} != "
                f"index map length {len(idx_map)}"
            )
            # Vectorized scatter-add
            np.add.at(global_rhs, idx_map, g_i)
            per_tile_rhs_stats.append(rhs_stats)

        # Coordinator-side DEBUG: per-tile RHS details
        for i, rs in enumerate(per_tile_rhs_stats):
            tid = model.metadata.tile_configs[i].tile_id
            logger.debug(
                "Tile %s get_reduced_rhs:\n"
                "  time: %.3fs  |  n_currents: %d  |  rhs_norm: %.4f",
                tid, rs.get('rhs_time_s', 0), rs.get('n_currents', 0),
                rs.get('rhs_norm', 0),
            )

        # Add Dirichlet contribution
        global_rhs += ctx.rhs_dirichlet_interface
        timings['assemble_rhs'] = time.perf_counter() - t0

        # 3. Solve interface system: v_Gamma = S_global^-1 * f
        t0 = time.perf_counter()
        v_gamma = ctx.interface_lu(global_rhs)
        timings['solve_interface'] = time.perf_counter() - t0

        # Build interface voltages dict
        interface_voltages: Dict[str, float] = {}
        for i, node in enumerate(ctx.interface_nodes):
            interface_voltages[node] = float(v_gamma[i])

        # 4. Distribute boundary voltages and recover interior (parallel on workers)
        t0 = time.perf_counter()
        # Build per-tile boundary voltage dicts (+ optional current overrides)
        bv_per_tile = []
        for i, tc in enumerate(model.metadata.tile_configs):
            tid = tc.tile_id
            boundary_list = model.tile_boundary_nodes[tid]
            tile_bv = {n: interface_voltages.get(n, model.vdd) for n in boundary_list}
            if per_tile_currents is not None:
                bv_per_tile.append((tile_bv, per_tile_currents[i]))
            else:
                bv_per_tile.append((tile_bv,))

        interior_results = model.backend.call_all(
            model.workers, 'get_interior_voltages', bv_per_tile,
        )
        timings['recover_interior'] = time.perf_counter() - t0

        # 5. Build result
        tile_results: Dict[Tuple[int, int], TileSolveResult] = {}
        per_tile_recovery_stats: List[Dict[str, Any]] = []
        for i, (voltages, recovery_stats) in enumerate(interior_results):
            tid = model.metadata.tile_configs[i].tile_id
            n_bnd = len(model.tile_boundary_nodes[tid])
            tile_results[tid] = TileSolveResult(
                tile_id=tid,
                voltages=voltages,
                n_interior=model.tile_interior_counts[tid],
                n_boundary=n_bnd,
            )
            per_tile_recovery_stats.append(recovery_stats)

        # Coordinator-side DEBUG: per-tile recovery details
        for i, rs in enumerate(per_tile_recovery_stats):
            tid = model.metadata.tile_configs[i].tile_id
            logger.debug(
                "Tile %s get_interior_voltages:\n"
                "  time: %.3fs  |  n_nodes: %d  |  v_range: [%.3f, %.3f]",
                tid, rs.get('recovery_time_s', 0), rs.get('n_nodes', 0),
                rs.get('v_min', 0), rs.get('v_max', 0),
            )

        pad_voltages = {n: model.vdd for n in model.pad_nodes}

        timings['total_solve'] = sum(timings.values())

        # --- Build solve_stats ---
        rhs_norm = float(np.linalg.norm(global_rhs))
        v_min = float(v_gamma.min()) if v_gamma.size > 0 else 0.0
        v_max = float(v_gamma.max()) if v_gamma.size > 0 else 0.0

        rhs_times = [s.get('rhs_time_s', 0.0) for s in per_tile_rhs_stats]
        recovery_times = [s.get('recovery_time_s', 0.0) for s in per_tile_recovery_stats]

        solve_stats: Dict[str, Any] = {
            'reduced_rhs': {
                'total_time_s': timings['compute_reduced_rhs'],
                'per_tile_times': rhs_times,
            },
            'assemble_rhs_s': timings['assemble_rhs'],
            'interface_solve': {
                'time_s': timings['solve_interface'],
                'n_unknowns': n_interface,
                'v_min': v_min,
                'v_max': v_max,
                'rhs_norm': rhs_norm,
            },
            'interior_recovery': {
                'total_time_s': timings['recover_interior'],
                'per_tile_times': recovery_times,
            },
            'per_tile_rhs_stats': per_tile_rhs_stats,
            'per_tile_recovery_stats': per_tile_recovery_stats,
        }

        # --- Verbose INFO logging ---
        if verbose:
            logger.info("=== Distributed DDM Solve Statistics ===")
            if rhs_times:
                rlo, ravg, rhi = _minmeanmax(rhs_times)
                logger.info("Reduced RHS:       %.3fs (per-tile: %.3f / %.3f / %.3f)",
                            timings['compute_reduced_rhs'], rlo, ravg, rhi)
            else:
                logger.info("Reduced RHS:       %.3fs", timings['compute_reduced_rhs'])
            logger.info("Assemble RHS:      %.3fs", timings['assemble_rhs'])
            logger.info("Interface solve:    %.3fs  |  %s unknowns  |  solution range [%.3f, %.3f]",
                        timings['solve_interface'], _fmt_count(n_interface), v_min, v_max)
            if recovery_times:
                ilo, iavg, ihi = _minmeanmax(recovery_times)
                logger.info("Interior recovery:  %.3fs (per-tile: %.3f / %.3f / %.3f)",
                            timings['recover_interior'], ilo, iavg, ihi)
            else:
                logger.info("Interior recovery:  %.3fs", timings['recover_interior'])

        return DistributedSolveResult(
            tile_results=tile_results,
            interface_voltages=interface_voltages,
            pad_voltages=pad_voltages,
            nominal_voltage=model.vdd,
            net_name=model.net_name,
            interface_size=n_interface,
            solve_metadata={'timings': timings, 'solve_stats': solve_stats},
        )

    def generate_reports(
        self,
        result: DistributedSolveResult,
        context: Optional[DistributedSolverContext] = None,
        output_dir: str = './results',
        plot_layers: Optional[List[str]] = None,
        max_stripes: int = 2000,
        stripe_bin_size: Optional[int] = None,
        show_irdrop: bool = True,
        top_k: int = 100,
        verbose: bool = False,
    ) -> None:
        """Generate per-layer stripe heatmaps from distributed solve results.

        Delegates to ``plot_distributed_heatmaps()`` from the heatmap module.
        When ``show_irdrop`` is True, generates IR-drop heatmaps (max
        aggregation). Always generates current heatmaps (sum aggregation).

        Parameters
        ----------
        result : DistributedSolveResult
            Per-tile solve result from ``solve_dc()``.
        context : DistributedSolverContext or None
            Solver context from ``prepare()``. If provided, floating nodes
            report will be generated. If None, no floating nodes report.
        output_dir : str
            Directory to write PNG files.
        plot_layers : list of str or None
            Layers to plot. ``None`` = all detected layers.
        max_stripes : int
            Maximum display stripes before consolidation.
        stripe_bin_size : int or None
            Bins per stripe along the parallel axis. ``None`` = auto.
        show_irdrop : bool
            If True (default), generate IR-drop heatmaps in addition to
            current heatmaps.
        top_k : int
            Number of worst IR-drop nodes to include in the top-K report
            (default 100).
        verbose : bool
            Log progress during heatmap generation.
        """
        from pathlib import Path
        from .heatmap import plot_distributed_heatmaps
        from reports.floating_nodes import (
            collect_floating_nodes_distributed,
            generate_floating_nodes_report,
        )
        from reports.topk_irdrop import generate_topk_report

        # Generate floating nodes report if context is provided
        if context is None:
            logger.warning(
                "No solver context provided. Skipping floating nodes report."
            )
        else:
            floating_data = collect_floating_nodes_distributed(
                self.model, context, self.model.workers,
                net_name=self.model.net_name,
            )
            generate_floating_nodes_report(floating_data, Path(output_dir), verbose=verbose)

        # Generate top-K worst IR-drop report
        v_all = result.flatten()
        pad_nodes = self.model.pad_nodes
        vdd = self.model.vdd

        # Compute IR-drop for each non-pad, non-ground node and find top-K
        ir_items = []
        for node, voltage in v_all.items():
            if node == '0' or node in pad_nodes:
                continue
            ir_items.append((node, abs(vdd - voltage)))
        ir_items.sort(key=lambda x: x[1], reverse=True)
        target_nodes = {node for node, _ in ir_items[:top_k]}

        # Parallel instance name lookup across all tile workers
        tile_configs = self.model.metadata.tile_configs
        lookup_args = [
            (target_nodes, tc.instance_path, tc.nd_path, tc.net_filter)
            for tc in tile_configs
        ]
        try:
            per_tile_maps = self.model.backend.call_all(
                self.model.workers, 'lookup_instance_names', lookup_args,
            )
        except Exception:
            logger.warning(
                "Instance name lookup failed; report will show N/A for instances",
                exc_info=True,
            )
            per_tile_maps = []

        # Merge per-tile instance dicts into a single mapping
        node_to_instance: Dict[str, str] = {}
        for tile_map in per_tile_maps:
            node_to_instance.update(tile_map)

        generate_topk_report(
            voltages=v_all,
            nominal_voltage=vdd,
            net_name=self.model.net_name,
            pad_nodes=pad_nodes,
            output_dir=output_dir,
            top_k=top_k,
            node_to_instance=node_to_instance,
        )

        common_kwargs = dict(
            model=self.model,
            output_dir=output_dir,
            plot_layers=plot_layers,
            max_stripes=max_stripes,
            stripe_bin_size=stripe_bin_size,
            verbose=verbose,
        )

        if show_irdrop:
            plot_distributed_heatmaps(
                result=result, is_current=False, **common_kwargs
            )

        plot_distributed_heatmaps(
            result=result, is_current=True, **common_kwargs
        )

    def generate_td_reports(
        self,
        result,  # DistributedQuasiStaticResult or DistributedTransientResult
        output_dir: str = './results',
        plot_layers: Optional[List[str]] = None,
        max_stripes: int = 2000,
        stripe_bin_size: Optional[int] = None,
        top_k: int = 100,
        verbose: bool = False,
    ) -> None:
        """Generate top-K report and peak IR-drop heatmaps for time-domain results.

        Worker-distributed: top-K peaks are collected from workers (each
        returns its local top-K), merged on the coordinator, and written
        as a ranked report. Peak IR-drop heatmaps are generated via
        worker-side pre-binning (no full voltage dict transfer).

        Parameters
        ----------
        result : DistributedQuasiStaticResult or DistributedTransientResult
            Time-domain result from ``solve_quasi_static()`` or
            ``solve_transient()``.
        output_dir : str
            Directory to write report files and PNG heatmaps.
        plot_layers : list of str or None
            Layers to plot. ``None`` = all detected layers.
        max_stripes : int
            Maximum display stripes before consolidation.
        stripe_bin_size : int or None
            Bins per stripe along the parallel axis. ``None`` = auto.
        top_k : int
            Number of worst IR-drop nodes to include in the top-K report.
        verbose : bool
            Log progress during report generation.
        """
        from pathlib import Path
        from .heatmap import plot_distributed_td_heatmaps
        from .result import DistributedTransientResult
        from reports.topk_irdrop import generate_topk_report

        model = self.model
        vdd = model.vdd
        net_name = model.net_name or 'unknown'
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # --- Top-K report (worker-distributed) ---
        # NOTE: This is an approximate distributed top-K. Each tile returns its
        # local top-K, which are then merged. This is exact when K >= max nodes
        # per tile, and a very good approximation in practice since the global
        # top-K nodes tend to cluster in a few hot-spot tiles.

        # 1. Collect local top-K from each worker
        topk_args = [(top_k,)] * len(model.metadata.tile_configs)
        try:
            per_tile_topk = model.backend.call_all(
                model.workers, 'get_top_k_peaks', args_per_actor=topk_args,
            )
        except Exception:
            logger.error(
                "Failed to collect peak data from workers. "
                "Was peak tracking initialized during the time-domain solve?",
                exc_info=True,
            )
            return

        # 2. Merge per-tile top-K lists: combine, sort by drop desc, take global top-K
        merged_items: List[Tuple[str, float, float]] = []
        seen: Dict[str, int] = {}  # node -> index in merged_items (keep highest drop)
        for tile_list in per_tile_topk:
            for node, drop, t in tile_list:
                idx = seen.get(node)
                if idx is not None:
                    if drop > merged_items[idx][1]:
                        merged_items[idx] = (node, drop, t)
                else:
                    seen[node] = len(merged_items)
                    merged_items.append((node, drop, t))

        merged_items.sort(key=lambda x: x[1], reverse=True)
        top_items = merged_items[:top_k]

        # 3. Convert to voltage dict for generate_topk_report
        voltages: Dict[str, float] = {node: vdd - drop for node, drop, _ in top_items}
        # Add pad nodes at nominal voltage
        for pad in model.pad_nodes:
            voltages[pad] = vdd

        # 4. Instance name lookup (parallel across tile workers)
        target_nodes = {node for node, _, _ in top_items}
        tile_configs = model.metadata.tile_configs
        lookup_args = [
            (target_nodes, tc.instance_path, tc.nd_path, tc.net_filter)
            for tc in tile_configs
        ]
        try:
            per_tile_maps = model.backend.call_all(
                model.workers, 'lookup_instance_names', lookup_args,
            )
        except Exception:
            logger.warning(
                "Instance name lookup failed; report will show N/A for instances",
                exc_info=True,
            )
            per_tile_maps = []

        node_to_instance: Dict[str, str] = {}
        for tile_map in per_tile_maps:
            node_to_instance.update(tile_map)

        # 5. Build extra header lines with mode and peak info
        is_transient = isinstance(result, DistributedTransientResult)
        mode_label = 'Transient (RC)' if is_transient else 'Quasi-Static (batch DC)'
        extra_header_lines = [
            f"Mode: {mode_label}",
            f"Time steps: {len(result.t_array)}",
            f"Peak IR-drop: {result.peak_ir_drop * 1000:.3f} mV "
            f"at t = {result.peak_ir_drop_time * 1e9:.3f} ns",
        ]

        generate_topk_report(
            voltages=voltages,
            nominal_voltage=vdd,
            net_name=net_name,
            pad_nodes=model.pad_nodes,
            output_dir=output_dir,
            top_k=top_k,
            node_to_instance=node_to_instance,
            extra_header_lines=extra_header_lines,
        )

        # --- Peak IR-drop heatmaps ---
        plot_distributed_td_heatmaps(
            model=model,
            nominal_voltage=vdd,
            net_name=net_name,
            output_dir=output_dir,
            plot_layers=plot_layers,
            max_stripes=max_stripes,
            stripe_bin_size=stripe_bin_size,
            title_prefix='Peak IR-Drop',
            verbose=verbose,
        )

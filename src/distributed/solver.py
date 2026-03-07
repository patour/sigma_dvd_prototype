"""Distributed DDM solver.

Orchestrates the Schur complement domain decomposition across tiles.
Follows the prepare/solve pattern matching the existing unified solver.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from .model import DistributedPowerGridModel
from .result import DistributedSolveResult, DistributedSolverContext, TileSolveResult

logger = logging.getLogger(__name__)


class DistributedDDMSolver:
    """Solver for distributed DDM. Takes DistributedPowerGridModel.

    Follows the prepare/solve pattern:
        model = create_distributed_model(metadata, backend='ray')
        solver = DistributedDDMSolver(model)
        ctx = solver.prepare()
        result = solver.solve_dc(context=ctx)
    """

    def __init__(self, model: DistributedPowerGridModel):
        self.model = model

    def prepare(self, verbose: bool = False) -> DistributedSolverContext:
        """Factor tiles + assemble/factor interface. Expensive, reusable for batch.

        Steps:
        1. Workers factor interior and compute explicit Schur complements
        2. Assemble global interface system from tile Schurs + package edges
        3. Factor interface system
        4. Build tile index maps (local -> global)

        Returns:
            DistributedSolverContext with cached factorizations
        """
        timings: Dict[str, float] = {}
        model = self.model

        # 1. Factor tiles and compute Schur complements (parallel on workers)
        t0 = time.perf_counter()
        schur_results = model.backend.call_all(
            model.workers, 'factor_and_compute_schur'
        )
        timings['factor_tiles'] = time.perf_counter() - t0

        # Organize results by tile
        tile_schur_complements: Dict[Any, np.ndarray] = {}
        tile_port_node_lists: Dict[Any, List[str]] = {}

        for i, (S_i, boundary_list) in enumerate(schur_results):
            tid = model.metadata.tile_configs[i].tile_id
            tile_schur_complements[tid] = S_i
            tile_port_node_lists[tid] = boundary_list

        # 2. Assemble global interface system
        t0 = time.perf_counter()
        from solver.coupled_system import assemble_schur_complement_system

        S_global, rhs_dirichlet, interface_nodes, interface_node_to_idx = (
            assemble_schur_complement_system(
                tile_schur_complements=tile_schur_complements,
                tile_port_node_lists=tile_port_node_lists,
                extra_edges=model.package_data.package_edges,
                dirichlet_nodes=model.pad_nodes,
                dirichlet_voltage=model.vdd,
            )
        )
        timings['assemble_interface'] = time.perf_counter() - t0

        # 2b. Global interface island detection
        t0 = time.perf_counter()
        from solver.coupled_system import detect_interface_islands

        S_global, rhs_dirichlet, island_nodes = detect_interface_islands(
            S_global, rhs_dirichlet, interface_nodes, interface_node_to_idx,
            pad_nodes=model.pad_nodes,
            extra_edges=model.package_data.package_edges,
            dirichlet_voltage=model.vdd,
        )
        timings['detect_interface_islands'] = time.perf_counter() - t0

        if island_nodes:
            logger.warning(
                "Penalized %d interface island nodes (shorted to %.3f V)",
                len(island_nodes), model.vdd,
            )

        if verbose and island_nodes:
            # Cross-check: tile-level kept non-largest components vs global islands
            tile_flagged: Set[str] = set()
            for nodes in model.tile_kept_nonlargest_iface.values():
                tile_flagged.update(nodes)

            confirmed = tile_flagged & island_nodes
            saved = tile_flagged - island_nodes
            if confirmed:
                logger.info(
                    "%d tile-flagged interface nodes confirmed as global islands",
                    len(confirmed),
                )
            if saved:
                logger.info(
                    "%d tile-flagged interface nodes connected to pads through other tiles",
                    len(saved),
                )

        if verbose:
            logger.info(
                f"Interface system: {len(interface_nodes)} unknowns, "
                f"{S_global.nnz} nonzeros"
            )

        # 3. Factor interface system
        t0 = time.perf_counter()
        from solver.unified_solver import _factor_conductance_matrix

        interface_lu = _factor_conductance_matrix(S_global, verbose=verbose)
        timings['factor_interface'] = time.perf_counter() - t0

        # 4. Build tile index maps (local port indices -> global interface indices)
        tile_index_maps: Dict[Tuple[int, int], np.ndarray] = {}
        for tid, boundary_list in tile_port_node_lists.items():
            local_to_global = np.array(
                [interface_node_to_idx[n] for n in boundary_list if n in interface_node_to_idx],
                dtype=np.int32,
            )
            tile_index_maps[tid] = local_to_global

        timings['total_prepare'] = sum(timings.values())

        if verbose:
            logger.info("Prepare timing breakdown:")
            for key, val in sorted(timings.items()):
                logger.info(f"  {key}: {val:.4f}s")

        return DistributedSolverContext(
            interface_lu=interface_lu.solve,
            interface_nodes=interface_nodes,
            interface_node_to_idx=interface_node_to_idx,
            rhs_dirichlet_interface=rhs_dirichlet,
            tile_index_maps=tile_index_maps,
            removed_interface_nodes=island_nodes,
            timings=timings,
        )

    def solve_dc(
        self,
        per_tile_currents: Optional[List[Dict[str, float]]] = None,
        context: Optional[DistributedSolverContext] = None,
        verbose: bool = False,
    ) -> DistributedSolveResult:
        """DC solve using domain decomposition.

        If context is None, calls prepare() internally.

        Args:
            per_tile_currents: Optional pre-partitioned currents, one dict per
                tile (node -> mA). If None, uses each tile's own current
                sources from parsing. Caller is responsible for partitioning
                boundary node currents to exactly one tile.
            context: Pre-computed solver context (from prepare())
            verbose: Print timing info

        Returns:
            DistributedSolveResult with per-tile voltages
        """
        timings: Dict[str, float] = {}
        model = self.model

        if context is None:
            context = self.prepare(verbose=verbose)

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

        for i, g_i in enumerate(rhs_results):
            tid = model.metadata.tile_configs[i].tile_id
            idx_map = ctx.tile_index_maps[tid]
            # g_i has shape (n_ports,) from compute_reduced_rhs
            assert len(g_i) == len(idx_map), (
                f"Tile {tid}: reduced RHS length {len(g_i)} != "
                f"index map length {len(idx_map)}"
            )
            # Vectorized scatter-add
            np.add.at(global_rhs, idx_map, g_i)

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
        for i, voltages in enumerate(interior_results):
            tid = model.metadata.tile_configs[i].tile_id
            n_bnd = len(model.tile_boundary_nodes[tid])
            tile_results[tid] = TileSolveResult(
                tile_id=tid,
                voltages=voltages,
                n_interior=model.tile_interior_counts[tid],
                n_boundary=n_bnd,
            )

        pad_voltages = {n: model.vdd for n in model.pad_nodes}

        timings['total_solve'] = sum(timings.values())

        if verbose:
            logger.info("Solve timing breakdown:")
            for key, val in sorted(timings.items()):
                logger.info(f"  {key}: {val:.4f}s")

        return DistributedSolveResult(
            tile_results=tile_results,
            interface_voltages=interface_voltages,
            pad_voltages=pad_voltages,
            nominal_voltage=model.vdd,
            net_name=model.net_name,
            interface_size=n_interface,
            solve_metadata={'timings': timings},
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

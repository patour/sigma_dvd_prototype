"""Adjoint sensitivity mixin for DistributedDDMSolver.

Provides static and dynamic adjoint sensitivity analysis for distributed
IR-drop attribution.  Kept in a separate file following the mixin pattern
established by solver_td.py.

The public API is surfaced on DistributedDDMSolver via inheritance.

Static Adjoint Algorithm (analyze_adjoint_static)
-------------------------------------------------
For DC/quasi-static analysis, the sensitivity of voltage at the victim node
to current injection at any other node is given by G^{-1}.  Specifically,
for a unit current injected at node j, the voltage change at the victim is
G^{-1}[victim, j].

The DDM approach computes this via the Schur complement:
1. Build terminal RHS: e_victim (unit vector at victim's interface node,
   or reduced RHS if victim is interior)
2. Coordinator solves: S_dc @ lambda_p = global_rhs
3. Workers recover interior lambda and accumulate contributions:
   contribution[source_j] = lambda[node_j] * I_j(t_observation)

This is equivalent to solving G @ lambda = e_victim and computing
lambda^T @ I, but distributed across tiles.
"""

from __future__ import annotations

import logging
import os
import time as time_module
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from analysis.adjoint_sensitivity import (
    AdjointAttribution,
    AggressorContribution,
)
from .interface_iterative import filter_kept_rhs
from .solver_td import _check_worker_transient_stamp
from .result import (
    DistributedQuasiStaticResult,
    DistributedSolveResult,
    DistributedSolverContext,
    DistributedTransientContext,
    DistributedTransientResult,
)
from .tile_parsing import _parse_node_xy

logger = logging.getLogger(__name__)


def _reset_interface_cg_warm_start(ctx: Any) -> None:
    """Reset the interface CG solver's warm-start state (plain ``_x0`` seed
    AND the two-point linear-extrapolation history -- see
    ``InterfaceCGSolver.reset_warm_start``) before a solve that starts a new
    RHS "family" on a context whose ``interface_lu`` may be a warm-started
    ``InterfaceCGSolver`` shared with an unrelated prior solve.

    Round-3 code review finding 2: ``ctx``/``trans_ctx`` are long-lived and
    shared across very different linear-system families that all route
    through the SAME ``ctx._cg_solver`` instance -- forward transient/QS
    voltage solves, the adjoint terminal condition (``e_victim``), and the
    adjoint backward-sweep steps for one victim vs. the NEXT victim's
    terminal condition. ``push_solution_history`` (called automatically at
    the end of every converged ``InterfaceCGSolver.__call__``) has no way
    to know a family boundary was crossed -- it will happily extrapolate
    ``2*x_prev - x_prev2`` across two solutions of entirely different
    systems when ``interface_warm_start_extrapolation=True``, a seed that
    can be strictly worse than a cold start. Call this at every genuine
    family boundary (adjoint terminal solve, and the forward solve that
    follows an adjoint solve on the same context) so each family starts
    from a clean plain-previous-or-cold seed instead. A no-op when the
    context is not using the CG interface solver (``_cg_solver is None`` --
    e.g. 'direct' mode has no warm-start state) or has no interface solver
    configured yet.
    """
    cg_solver = getattr(ctx, '_cg_solver', None)
    if cg_solver is not None:
        cg_solver.reset_warm_start()


class _AdjointMixin:
    """Mixin providing adjoint sensitivity methods for DistributedDDMSolver.

    Expects the host class to provide:
        self.model (DistributedPowerGridModel)
        self.solve_dc(context, ...) -> DistributedSolveResult
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze_adjoint_static(
        self,
        victim_nodes: Union[str, List[str]],
        observation_time: float,
        dc_context: DistributedSolverContext,
        top_k: int = 10,
        spatial_window: Optional[Tuple[float, float, float, float]] = None,
        window_margin: float = 0.0,
        include_waveforms: bool = False,
        verbose: bool = False,
    ) -> Union[AdjointAttribution, List[AdjointAttribution]]:
        """Static sensitivity analysis using G^{-1}.

        Computes which aggressor current sources contribute most to the
        IR-drop at the victim node(s) at the observation time, using the
        DC Schur complement factorization.

        For stiff RC systems (tau << dt), the static method provides a good
        approximation to the full dynamic adjoint and is computationally
        efficient since it reuses the DC factorization.

        Args:
            victim_nodes: Single victim node name or list of victim node names.
            observation_time: Time T at which to evaluate source currents
                (seconds).
            dc_context: Pre-computed DC solver context (from ``prepare()``).
                Must be factored.
            top_k: Number of top aggressors to return per victim.
            spatial_window: Optional ``(x_min, x_max, y_min, y_max)`` bounding
                box to limit candidate sources.  Coordinates are in the same
                units as PDN node names (um).
            window_margin: If > 0 and ``spatial_window`` is None, auto-creates
                a window around the victim with this margin (um).
            include_waveforms: If True, include single-point current values
                in the results.
            verbose: Log timing and progress.

        Returns:
            If ``victim_nodes`` is a single string: ``AdjointAttribution``.
            If ``victim_nodes`` is a list: ``List[AdjointAttribution]``,
            one per victim in the same order.

        Raises:
            ValueError: If dc_context is not factored.
            RuntimeError: If victim node is not found in any tile.
        """
        if not dc_context.is_factored:
            raise ValueError(
                "dc_context is not factored. Call context.factor() or use "
                "solver.prepare() to obtain a factored context."
            )

        # Normalize input
        single_victim = isinstance(victim_nodes, str)
        victims = [victim_nodes] if single_victim else list(victim_nodes)

        # Initialize source mappings on workers (idempotent)
        self._ensure_adjoint_source_mappings()

        results: List[AdjointAttribution] = []
        for victim_node in victims:
            result = self._analyze_single_victim_static(
                victim_node=victim_node,
                observation_time=observation_time,
                dc_context=dc_context,
                top_k=top_k,
                spatial_window=spatial_window,
                window_margin=window_margin,
                include_waveforms=include_waveforms,
                verbose=verbose,
            )
            results.append(result)

        return results[0] if single_victim else results

    def analyze_adjoint(
        self,
        victim_nodes: Union[str, List[str]],
        observation_time: float,
        trans_context: DistributedTransientContext,
        transient_result: DistributedTransientResult,
        memory_window: int = 20,
        top_k: int = 10,
        spatial_window: Optional[Tuple[float, float, float, float]] = None,
        window_margin: float = 0.0,
        include_waveforms: bool = True,
        initial_condition: str = 'zero',
        dc_context: Optional[DistributedSolverContext] = None,
        verbose: bool = False,
    ) -> Union[AdjointAttribution, List[AdjointAttribution]]:
        """Full dynamic adjoint sensitivity analysis using backward sweep.

        Propagates sensitivities backward through the RC network's memory
        to compute which aggressor current sources contribute most to the
        IR-drop at the victim node(s) at the observation time.

        For stiff RC systems (tau << dt), this converges to the static result.
        For systems with significant decoupling, this captures time-varying
        memory effects.

        Args:
            victim_nodes: Single victim node name or list of victim node names.
            observation_time: Time T at which to observe IR-drop (seconds).
            trans_context: Pre-computed transient context (from
                ``prepare_transient()``). Must be factored.
            transient_result: Result from ``solve_transient()``. Used to
                extract IR-drop at T for the victim.
            memory_window: Number of time steps to look back from T. The actual
                time step is taken from ``trans_context.dt`` to ensure the
                backward sweep uses the same time grid as the forward solve.
            top_k: Number of top aggressors to return per victim.
            spatial_window: Optional ``(x_min, x_max, y_min, y_max)`` bounding
                box to limit candidate sources (coordinates in um).
            window_margin: If > 0 and ``spatial_window`` is None, auto-creates
                a window around the victim with this margin (um).
            include_waveforms: If True, include current waveforms in results.
            initial_condition: ``'zero'`` (default) or ``'dc'``. ``'dc'`` mode
                adds DC contribution at t=0 (requires ``dc_context``).
            dc_context: DC solver context for ``'dc'`` initial condition.
                Required when ``initial_condition='dc'``.
            verbose: Log timing and progress.

        Returns:
            If ``victim_nodes`` is a single string: ``AdjointAttribution``.
            If ``victim_nodes`` is a list: ``List[AdjointAttribution]``,
            one per victim in the same order.

        Raises:
            ValueError: If trans_context is not factored, or if
                ``initial_condition='dc'`` but ``dc_context`` is None.
            RuntimeError: If victim node is not found in any tile.
        """
        if not trans_context.is_factored:
            raise ValueError(
                "trans_context is not factored. Call context.factor() or use "
                "solver.prepare_transient() to obtain a factored context."
            )
        # Finding 2 (round 2 review): analyze_adjoint's backward sweep
        # consumes the SAME shared, single-owner worker-side transient
        # factorization solve_transient() does (tile_worker_adjoint's
        # lambda-recovery RPCs require factor_transient_system() to have
        # already run on the workers) -- guard it identically, BEFORE the
        # backward sweep, so a stale/mismatched/unknown worker factorization
        # raises loudly here too instead of silently producing wrong
        # sensitivities/blame attributions. analyze_adjoint_static (DC-only,
        # no transient context) does not need this guard.
        _check_worker_transient_stamp(self.model, trans_context, caller='analyze_adjoint')
        if initial_condition == 'dc' and dc_context is None:
            raise ValueError(
                "initial_condition='dc' requires dc_context to be provided"
            )
        if initial_condition not in ('zero', 'dc'):
            raise ValueError(
                f"initial_condition must be 'zero' or 'dc', got '{initial_condition}'"
            )

        # Normalize input
        single_victim = isinstance(victim_nodes, str)
        victims = [victim_nodes] if single_victim else list(victim_nodes)

        # Initialize source mappings on workers (idempotent)
        self._ensure_adjoint_source_mappings()

        results: List[AdjointAttribution] = []
        for victim_node in victims:
            result = self._analyze_single_victim_dynamic(
                victim_node=victim_node,
                observation_time=observation_time,
                trans_context=trans_context,
                transient_result=transient_result,
                memory_window=memory_window,
                top_k=top_k,
                spatial_window=spatial_window,
                window_margin=window_margin,
                include_waveforms=include_waveforms,
                initial_condition=initial_condition,
                dc_context=dc_context,
                verbose=verbose,
            )
            results.append(result)

        return results[0] if single_victim else results

    # ------------------------------------------------------------------
    # Helper: locate victim tiles
    # ------------------------------------------------------------------

    def _locate_victim_tiles(self, victim_node: str) -> List[Tuple[int, int]]:
        """Find which tiles contain the victim node.

        Broadcasts ``has_node(victim_node)`` to all workers and returns
        the list of tile_ids where the victim is found.

        Args:
            victim_node: Node name to locate.

        Returns:
            List of tile_ids (tuples) that contain the victim node.
        """
        model = self.model
        tile_configs = model.metadata.tile_configs

        # Call has_node on all workers
        has_node_args = [(victim_node,)] * len(tile_configs)
        has_results = model.backend.call_all(
            model.workers, 'has_node', has_node_args,
        )

        # Collect tile_ids where victim was found
        victim_tiles = []
        for i, has in enumerate(has_results):
            if has:
                victim_tiles.append(tile_configs[i].tile_id)

        return victim_tiles

    # ------------------------------------------------------------------
    # Helper: extract IR-drop from result
    # ------------------------------------------------------------------

    def _get_ir_drop_from_result(
        self,
        victim_node: str,
        observation_time: float,
        result: Union[
            DistributedSolveResult,
            DistributedQuasiStaticResult,
            DistributedTransientResult,
        ],
    ) -> float:
        """Extract IR-drop at victim node at observation time from a result.

        Args:
            victim_node: The victim node name.
            observation_time: Time T at which IR-drop is observed (seconds).
                For DC results, this is ignored.
            result: A DC, quasi-static, or transient result object.

        Returns:
            IR-drop in millivolts (mV).
        """
        vdd = self.model.vdd

        if isinstance(result, DistributedSolveResult):
            # DC result: just get voltage from flattened result
            voltages = result.flatten()
            if victim_node not in voltages:
                raise ValueError(
                    f"Victim node '{victim_node}' not found in DC result"
                )
            return (vdd - voltages[victim_node]) * 1000.0

        # Time-domain result: use peak_ir_drop from as_flat()
        # Note: as_flat() returns {node: (max_drop, time_of_max)}
        peak_data = result.as_flat()
        if victim_node in peak_data:
            max_drop, _time = peak_data[victim_node]
            return max_drop * 1000.0

        # Fallback: victim might be a pad node
        if victim_node in self.model.pad_nodes:
            return 0.0

        raise ValueError(
            f"Victim node '{victim_node}' not found in time-domain result"
        )

    # ------------------------------------------------------------------
    # Helper: build attribution result
    # ------------------------------------------------------------------

    def _build_attribution_result(
        self,
        victim_node: str,
        ir_drop_at_T: float,
        contributions: Dict[str, float],
        source_to_node: Dict[str, str],
        waveforms: Dict[str, List[Tuple[float, float]]],
        top_k: int,
        observation_time: float,
        spatial_window: Optional[Tuple[float, float, float, float]],
        n_candidate_sources: int,
        timings: Dict[str, float],
        *,
        memory_window: Optional[Tuple[float, float]] = None,
        t_array: Optional[np.ndarray] = None,
        initial_condition: str = 'zero',
        dc_context: Optional[DistributedSolverContext] = None,
    ) -> AdjointAttribution:
        """Build AdjointAttribution from merged per-source contributions.

        Used by both static and dynamic adjoint paths.  The dynamic path
        passes the extra keyword arguments to populate time-domain fields.

        Args:
            victim_node: The victim node name.
            ir_drop_at_T: Total IR-drop at victim at T (mV).
            contributions: Dict mapping source_name -> contribution in mV.
            source_to_node: Dict mapping source_name -> node_name.
            waveforms: Dict mapping source_name -> list of (time, current).
            top_k: Number of top aggressors to return.
            observation_time: Time T at which IR-drop was observed.
            spatial_window: Spatial window used for filtering.
            n_candidate_sources: Number of sources considered.
            timings: Timing breakdown dict.
            memory_window: ``(t_start, t_end)`` for dynamic adjoint, or
                ``None`` to default to ``(observation_time, observation_time)``.
            t_array: Time array for dynamic adjoint, or ``None`` to default
                to a single-point array ``[observation_time]``.
            initial_condition: ``'zero'`` (default) or ``'dc'``.
            dc_context: DC solver context, used when
                ``initial_condition='dc'`` to compute DC IR-drop.

        Returns:
            AdjointAttribution result.
        """
        # Defaults for static path
        if memory_window is None:
            memory_window = (observation_time, observation_time)
        if t_array is None:
            t_array = np.array([observation_time])

        # Build node_to_sources mapping for grouping
        node_to_sources: Dict[str, List[str]] = {}
        for src_name, node in source_to_node.items():
            if node not in node_to_sources:
                node_to_sources[node] = []
            node_to_sources[node].append(src_name)

        # Aggregate contributions by node
        node_contributions: Dict[str, float] = {}
        for src_name, contrib in contributions.items():
            node = source_to_node.get(src_name)
            if node is None:
                continue
            node_contributions[node] = node_contributions.get(node, 0.0) + contrib

        # Separate self-contribution (victim's own sources)
        self_contribution_mV = node_contributions.pop(victim_node, 0.0)
        self_sources = node_to_sources.get(victim_node, [])

        # Compute total attributed from full dict BEFORE top-K truncation (O(N) dict sum)
        total_attributed = self_contribution_mV + sum(node_contributions.values())

        # Snapshot all node contributions for downstream spatial partitioning
        all_node_contribs = dict(node_contributions)

        # Compute self percentage
        self_contribution_pct = (
            100.0 * self_contribution_mV / ir_drop_at_T
            if ir_drop_at_T > 0 else 0.0
        )

        # Build victim current waveform if available
        victim_current_waveform = None
        if waveforms and self_sources:
            # Aggregate all victim sources
            combined = []
            for src in self_sources:
                if src in waveforms:
                    combined.extend(waveforms[src])
            if combined:
                combined.sort(key=lambda x: x[0])
                victim_current_waveform = np.array([c for _, c in combined])

        # Sort by |contribution|, take top K
        sorted_nodes = sorted(
            node_contributions.items(),
            key=lambda x: abs(x[1]),
            reverse=True,
        )[:top_k]

        # Build AggressorContribution list
        top_aggressors: List[AggressorContribution] = []
        for node, contrib_mV in sorted_nodes:
            contribution_pct = (
                100.0 * contrib_mV / ir_drop_at_T if ir_drop_at_T > 0 else 0.0
            )

            # Aggregate waveforms for this node
            node_waveform = None
            node_sources = node_to_sources.get(node, [])
            if waveforms and node_sources:
                combined = []
                for src in node_sources:
                    if src in waveforms:
                        combined.extend(waveforms[src])
                if combined:
                    combined.sort(key=lambda x: x[0])
                    node_waveform = np.array([c for _, c in combined])

            top_aggressors.append(AggressorContribution(
                node=node,
                contribution_mV=contrib_mV,
                contribution_pct=contribution_pct,
                source_names=node_sources,
                current_waveform=node_waveform,
            ))

        attribution_efficiency = (
            total_attributed / ir_drop_at_T if ir_drop_at_T > 0 else 0.0
        )

        # DC IR-drop for 'dc' initial condition
        dc_ir_drop_mV = None
        if initial_condition == 'dc' and dc_context is not None:
            dc_result = self.solve_dc(dc_context, verbose=False)
            dc_ir_drop_mV = self._get_ir_drop_from_result(
                victim_node, 0.0, dc_result,
            )

        return AdjointAttribution(
            victim_node=victim_node,
            observation_time=observation_time,
            ir_drop_at_T=ir_drop_at_T,
            memory_window=memory_window,
            t_array=t_array,
            spatial_window=spatial_window,
            n_candidate_sources=n_candidate_sources,
            self_contribution_mV=self_contribution_mV,
            self_contribution_pct=self_contribution_pct,
            victim_current_waveform=victim_current_waveform,
            top_aggressors=top_aggressors,
            total_attributed_ir_drop=total_attributed,
            attribution_efficiency=attribution_efficiency,
            timings=timings,
            initial_condition=initial_condition,
            dc_ir_drop_mV=dc_ir_drop_mV,
            all_node_contributions=all_node_contribs,
        )

    # ------------------------------------------------------------------
    # Internal: ensure source mappings are initialized
    # ------------------------------------------------------------------

    def _ensure_adjoint_source_mappings(self) -> None:
        """Initialize VCS and source mappings on workers (idempotent).

        Called once per adjoint analysis session.  Safe to call multiple
        times -- workers track whether mappings have been built.

        This method initializes both:
        1. VectorizedCurrentSources (VCS) for current evaluation
        2. Adjoint source name-to-index mappings for contribution accumulation

        The VCS initialization must happen before adjoint analysis because
        the contribution accumulation uses VCS.evaluate_per_source_at_time().
        """
        if getattr(self, '_adjoint_sources_initialized', False):
            return

        model = self.model
        tile_configs = model.metadata.tile_configs

        first_ckt = tile_configs[0].ckt_path
        pkl_dir = os.path.dirname(first_ckt)
        net_filter = tile_configs[0].net_filter

        # 1. Initialize VCS on all workers (needed for current evaluation)
        vcs_args = [
            (tc.instance_path, tc.nd_path, net_filter, pkl_dir)
            for tc in tile_configs
        ]
        model.backend.call_all(
            model.workers, 'init_vectorized_sources', vcs_args,
        )

        # 2. Initialize adjoint source name-to-index mappings on all workers
        mapping_args = [
            (tc.instance_path, tc.nd_path, tc.net_filter)
            for tc in tile_configs
        ]
        model.backend.call_all(
            model.workers, 'init_adjoint_source_mappings', mapping_args,
        )

        self._adjoint_sources_initialized = True

    # ------------------------------------------------------------------
    # Internal: single victim static analysis
    # ------------------------------------------------------------------

    def _analyze_single_victim_static(
        self,
        victim_node: str,
        observation_time: float,
        dc_context: DistributedSolverContext,
        top_k: int,
        spatial_window: Optional[Tuple[float, float, float, float]],
        window_margin: float,
        include_waveforms: bool,
        verbose: bool,
    ) -> AdjointAttribution:
        """Analyze a single victim using static sensitivity.

        Implements the DDM adjoint pattern:
        1. Workers compute terminal RHS from e_victim
        2. Coordinator assembles and solves interface system
        3. Workers recover interior lambda and accumulate contributions
        """
        timings: Dict[str, float] = {}
        t0_total = time_module.perf_counter()

        model = self.model
        tile_configs = model.metadata.tile_configs
        ctx = dc_context
        vdd = model.vdd

        # 1. Locate victim tiles
        t0 = time_module.perf_counter()
        victim_tiles = self._locate_victim_tiles(victim_node)
        timings['locate_victim'] = time_module.perf_counter() - t0

        if not victim_tiles:
            raise RuntimeError(
                f"Victim node '{victim_node}' not found in any tile"
            )

        if verbose:
            logger.info(
                "Victim '%s' found in tiles: %s",
                victim_node, victim_tiles,
            )

        # 2. Auto-create spatial window if margin specified
        t0 = time_module.perf_counter()
        actual_window = spatial_window
        if actual_window is None and window_margin > 0:
            x_v, y_v = _parse_node_xy(victim_node)
            if x_v is not None:
                actual_window = (
                    x_v - window_margin,
                    x_v + window_margin,
                    y_v - window_margin,
                    y_v + window_margin,
                )

        # 3. Apply spatial filtering on workers
        if actual_window is not None:
            filter_args = [(actual_window,)] * len(tile_configs)
            n_active_per_tile = model.backend.call_all(
                model.workers, 'filter_adjoint_sources', filter_args,
            )
            n_candidate_sources = sum(n_active_per_tile)
        else:
            # Clear any previous filter
            filter_args = [(None,)] * len(tile_configs)
            model.backend.call_all(
                model.workers, 'filter_adjoint_sources', filter_args,
            )
            # Count total sources (approximate from per-tile stats)
            # For now, set to -1 indicating unknown
            n_candidate_sources = -1
        timings['filter'] = time_module.perf_counter() - t0

        if verbose and actual_window is not None:
            logger.info(
                "Spatial window: (%.0f, %.0f, %.0f, %.0f), candidates: %d",
                actual_window[0], actual_window[1],
                actual_window[2], actual_window[3],
                n_candidate_sources,
            )

        # 4. Workers: compute terminal reduced RHS (adjoint terminal condition)
        # For static adjoint: A = G, so use_transient=False
        t0 = time_module.perf_counter()
        rhs_args = [(victim_node, False)] * len(tile_configs)
        rhs_results = model.backend.call_all(
            model.workers, 'get_adjoint_terminal_reduced_rhs', rhs_args,
        )
        timings['compute_terminal_rhs'] = time_module.perf_counter() - t0

        # 5. Coordinator: assemble global RHS
        t0 = time_module.perf_counter()
        n_interface = len(ctx.interface_nodes)
        global_rhs = np.zeros(n_interface, dtype=np.float64)

        victim_stats = []
        for i, (g_i, stats) in enumerate(rhs_results):
            tid = tile_configs[i].tile_id
            idx_map = ctx.tile_index_maps[tid]
            # S2/S13 (D1 follow-up): get_adjoint_terminal_reduced_rhs returns
            # g_i in FULL port order (may include a tile-resident pad port);
            # filter+validate before scattering (see solve_dc's identical fix).
            g_i = filter_kept_rhs(
                g_i, tid, idx_map, ctx.tile_kept_port_pos, ctx.tile_port_count,
                caller='analyze_adjoint_static',
            )
            # Scatter-add reduced RHS
            np.add.at(global_rhs, idx_map, g_i)
            victim_stats.append(stats)

        # B1 fix: Port victims appear on multiple tiles, causing scatter-add to
        # sum multiple unit vectors (e.g., global_rhs[victim_idx] = 2.0).
        # Reset to 1.0 for correct adjoint terminal condition.
        victim_global_idx = ctx.interface_node_to_idx.get(victim_node)
        if victim_global_idx is not None:
            global_rhs[victim_global_idx] = 1.0

        if verbose:
            for i, stats in enumerate(victim_stats):
                if stats.get('victim_type') != 'absent':
                    logger.info(
                        "Tile %s: victim is %s (idx=%s)",
                        tile_configs[i].tile_id,
                        stats['victim_type'],
                        stats['victim_idx'],
                    )
        timings['assemble_rhs'] = time_module.perf_counter() - t0

        # Note: For static adjoint (G-based), no Dirichlet RHS correction
        # is needed because pads have fixed voltage and zero adjoint.
        # The adjoint equation is G @ lambda = e_victim, not the forward
        # problem with Dirichlet BCs.

        # 6. Coordinator: solve interface system
        # Finding 2 (round 3): this ctx's interface CG solver may still be
        # warm-started from an unrelated prior solve (a forward DC solve,
        # or -- across victims in the caller's loop -- the previous
        # victim's own forward QS solve below) -- reset before starting
        # this victim's adjoint family (see _reset_interface_cg_warm_start).
        _reset_interface_cg_warm_start(ctx)
        t0 = time_module.perf_counter()
        lambda_p = ctx.interface_lu(global_rhs)
        timings['solve_interface'] = time_module.perf_counter() - t0

        # Build lambda_p dict
        lambda_p_dict: Dict[str, float] = {}
        for i, node in enumerate(ctx.interface_nodes):
            lambda_p_dict[node] = float(lambda_p[i])

        # 7. Workers: recover interior lambda and accumulate contributions
        # Static adjoint uses DC block system (G), not transient (A = G + C*C).
        # Worker reconstructs e_i internally from victim_node.
        t0 = time_module.perf_counter()

        recover_results = model.backend.call_all(
            model.workers, 'recover_and_accumulate_adjoint_static',
            [(lambda_p_dict, victim_node, observation_time, include_waveforms)]
            * len(tile_configs),
        )
        timings['recover_accumulate'] = time_module.perf_counter() - t0

        # 8. Collect results from workers
        t0 = time_module.perf_counter()
        collect_results = model.backend.call_all(
            model.workers, 'get_adjoint_results',
        )

        # Merge contributions across tiles
        merged_contributions: Dict[str, float] = {}
        merged_source_to_node: Dict[str, str] = {}
        merged_waveforms: Dict[str, List[Tuple[float, float]]] = {}

        for contrib_dict, s2n_dict, wf_dict in collect_results:
            # Contributions: convert V to mV and sum across tiles
            for src_name, contrib_V in contrib_dict.items():
                contrib_mV = contrib_V * 1000.0
                merged_contributions[src_name] = (
                    merged_contributions.get(src_name, 0.0) + contrib_mV
                )
            # Source-to-node mapping (should be consistent across tiles)
            merged_source_to_node.update(s2n_dict)
            # Waveforms: merge lists
            for src_name, wf_list in wf_dict.items():
                if src_name not in merged_waveforms:
                    merged_waveforms[src_name] = []
                merged_waveforms[src_name].extend(wf_list)
        timings['collect_results'] = time_module.perf_counter() - t0

        # 9. Compute IR-drop at victim via a quasi-static solve at observation_time.
        #    This evaluates I(T) on workers and solves G @ V = I(T), matching the
        #    flat solver's behavior (which uses _evaluate_currents_from_raw_sources(T)).
        #    Pass a dummy smoothed_sources to prevent solve_quasi_static from
        #    re-calling preprocess_sources (which would apply smoothing by default
        #    and overwrite _active_sources).
        # Finding 2 (round 3): this forward voltage solve reuses the SAME
        # ctx (and, in CG mode, the SAME warm-started _cg_solver) as the
        # adjoint lambda solve just above -- a different RHS family
        # (forward currents vs. e_victim) -- reset before crossing back.
        _reset_interface_cg_warm_start(ctx)
        t0 = time_module.perf_counter()
        from .result import DistributedSmoothedSources
        _already_init = DistributedSmoothedSources(
            time_step=0, t_start=0, t_end=0, smoothed=False,
            n_tiles=len(tile_configs), per_tile_stats={},
        )
        qs_result = self.solve_quasi_static(
            dc_context,
            t_array=np.array([observation_time]),
            smoothed_sources=_already_init,
            n_worst_nodes=1,
            verbose=False,
        )
        qs_flat = qs_result.as_flat()
        if victim_node in qs_flat:
            ir_drop_at_T = qs_flat[victim_node][0] * 1000.0  # V -> mV
        else:
            ir_drop_at_T = qs_result.peak_ir_drop * 1000.0
        timings['forward_solve'] = time_module.perf_counter() - t0

        # 10. Build attribution result
        t0 = time_module.perf_counter()
        attribution = self._build_attribution_result(
            victim_node=victim_node,
            ir_drop_at_T=ir_drop_at_T,
            contributions=merged_contributions,
            source_to_node=merged_source_to_node,
            waveforms=merged_waveforms,
            top_k=top_k,
            observation_time=observation_time,
            spatial_window=actual_window,
            n_candidate_sources=n_candidate_sources if n_candidate_sources >= 0 else len(merged_contributions),
            timings=timings,
        )
        timings['build_result'] = time_module.perf_counter() - t0
        timings['total'] = time_module.perf_counter() - t0_total

        # Update timings in result
        attribution.timings.update(timings)

        if verbose:
            logger.info(
                "Adjoint static analysis for '%s':\n"
                "  IR-drop at T: %.3f mV\n"
                "  Self contribution: %.3f mV (%.1f%%)\n"
                "  Top aggressor: %s (%.3f mV, %.1f%%)\n"
                "  Attribution efficiency: %.3f\n"
                "  Total time: %.3fs",
                victim_node,
                attribution.ir_drop_at_T,
                attribution.self_contribution_mV,
                attribution.self_contribution_pct,
                attribution.top_aggressors[0].node if attribution.top_aggressors else 'N/A',
                attribution.top_aggressors[0].contribution_mV if attribution.top_aggressors else 0.0,
                attribution.top_aggressors[0].contribution_pct if attribution.top_aggressors else 0.0,
                attribution.attribution_efficiency,
                timings['total'],
            )

        # 11. Reset adjoint state on workers for next analysis
        model.backend.call_all(model.workers, 'reset_adjoint_state')

        return attribution

    # ------------------------------------------------------------------
    # Internal: single victim dynamic analysis (backward sweep)
    # ------------------------------------------------------------------

    def _analyze_single_victim_dynamic(
        self,
        victim_node: str,
        observation_time: float,
        trans_context: DistributedTransientContext,
        transient_result: DistributedTransientResult,
        memory_window: int,
        top_k: int,
        spatial_window: Optional[Tuple[float, float, float, float]],
        window_margin: float,
        include_waveforms: bool,
        initial_condition: str,
        dc_context: Optional[DistributedSolverContext],
        verbose: bool,
    ) -> AdjointAttribution:
        """Analyze a single victim using dynamic backward sweep.

        Implements the DDM adjoint backward sweep:
        1. Terminal step (k = L-1): A @ lambda = e_victim
        2. Backward steps (k = L-2 down to 0): A @ lambda_k = B @ lambda_{k+1}
           where B = C_coeff * C (capacitive mass matrix)
        3. At each step, accumulate contributions: lambda[j] * I_j(t_k)

        The Schur complement reduction is used at each step:
        - Workers compute reduced RHS: g = s_p - A_pi @ lu_ii(s_i)
        - Coordinator assembles global RHS + package cap history
        - Coordinator solves: S_transient @ lambda_p = global_rhs
        - Workers recover interior lambda and accumulate
        """
        timings: Dict[str, float] = {}
        t0_total = time_module.perf_counter()

        model = self.model
        tile_configs = model.metadata.tile_configs
        trans_ctx = trans_context

        # 1. Locate victim tiles
        t0 = time_module.perf_counter()
        victim_tiles = self._locate_victim_tiles(victim_node)
        timings['locate_victim'] = time_module.perf_counter() - t0

        if not victim_tiles:
            raise RuntimeError(
                f"Victim node '{victim_node}' not found in any tile"
            )

        if verbose:
            logger.info(
                "Victim '%s' found in tiles: %s",
                victim_node, victim_tiles,
            )

        # 2. Extract IR-drop at T from transient result
        ir_drop_at_T = self._get_ir_drop_from_result(
            victim_node, observation_time, transient_result,
        )

        # 3. Auto-create spatial window if margin specified
        t0 = time_module.perf_counter()
        actual_window = spatial_window
        if actual_window is None and window_margin > 0:
            x_v, y_v = _parse_node_xy(victim_node)
            if x_v is not None:
                actual_window = (
                    x_v - window_margin,
                    x_v + window_margin,
                    y_v - window_margin,
                    y_v + window_margin,
                )

        # 4. Apply spatial filtering on workers
        if actual_window is not None:
            filter_args = [(actual_window,)] * len(tile_configs)
            n_active_per_tile = model.backend.call_all(
                model.workers, 'filter_adjoint_sources', filter_args,
            )
            n_candidate_sources = sum(n_active_per_tile)
        else:
            # Clear any previous filter
            filter_args = [(None,)] * len(tile_configs)
            model.backend.call_all(
                model.workers, 'filter_adjoint_sources', filter_args,
            )
            n_candidate_sources = -1
        timings['filter'] = time_module.perf_counter() - t0

        if verbose and actual_window is not None:
            logger.info(
                "Spatial window: (%.0f, %.0f, %.0f, %.0f), candidates: %d",
                actual_window[0], actual_window[1],
                actual_window[2], actual_window[3],
                n_candidate_sources,
            )

        # 5. Compute time array using trans_context.dt to match forward solve
        # The backward sweep must use the same dt as the transient factorization
        dt = trans_ctx.dt  # Time step in seconds from transient context
        L = memory_window + 1  # Number of time points
        t_start = max(0.0, observation_time - memory_window * dt)
        t_array = np.arange(t_start, observation_time + dt * 0.5, dt)
        L = len(t_array)

        if verbose:
            logger.info(
                "Backward sweep: %d steps, dt=%.2e s, t_start=%.2e s, T=%.2e s",
                L, dt, t_start, observation_time,
            )

        n_interface = len(trans_ctx.interface_nodes)

        # === Terminal step (k = L-1) ===
        t0 = time_module.perf_counter()

        # 6. Workers compute terminal RHS: g = -A_pi @ lu_A_ii(e_i) for victim
        # use_transient=True to use A = G + C*C block system
        rhs_args = [(victim_node, True)] * len(tile_configs)
        rhs_results = model.backend.call_all(
            model.workers, 'get_adjoint_terminal_reduced_rhs', rhs_args,
        )

        # 7. Coordinator assembles global_rhs
        global_rhs = self._build_adjoint_global_rhs(
            tile_rhs_results=rhs_results,
            trans_ctx=trans_ctx,
            lambda_p_old=None,  # Terminal step: no history
            victim_node=victim_node,  # For port victim fix
        )

        # 8. Coordinator solves S_transient @ lambda_p = global_rhs
        # Finding 2 (round 3): trans_ctx's interface CG solver may still be
        # warm-started from the forward transient sweep that produced
        # ``transient_result`` (a different RHS family: forward voltages,
        # not adjoint lambdas) -- OR, when this method runs for the NEXT
        # victim in analyze_adjoint's loop, from the PREVIOUS victim's own
        # terminal/backward-sweep solves (a different e_victim). Resetting
        # here, at the top of every terminal step, covers both boundaries
        # (see _reset_interface_cg_warm_start) without touching the
        # legitimate warm start WITHIN this victim's backward sweep below
        # (each step's lambda_p is a genuine continuation of the SAME
        # adjoint family, one time step apart -- exactly analogous to the
        # forward loop's own warm start across time steps).
        _reset_interface_cg_warm_start(trans_ctx)
        lambda_p = trans_ctx.interface_lu(global_rhs)
        timings['terminal_step'] = time_module.perf_counter() - t0

        # Build lambda_p dict for workers
        lambda_p_dict: Dict[str, float] = {}
        for i, node in enumerate(trans_ctx.interface_nodes):
            lambda_p_dict[node] = float(lambda_p[i])

        # 9. Workers recover lambda_i and accumulate at t = observation_time
        # Terminal step uses e_victim as source term. Worker reconstructs from victim_node.
        terminal_recover_args = [
            (lambda_p_dict, observation_time, victim_node, include_waveforms)
        ] * len(tile_configs)
        model.backend.call_all(
            model.workers, 'recover_and_accumulate_adjoint_terminal',
            terminal_recover_args,
        )

        # Store lambda_p for next step
        lambda_p_old = lambda_p.copy()

        # === Backward steps (k = L-2 down to 0) ===
        t0_loop = time_module.perf_counter()
        cum_rhs_time = 0.0
        cum_solve_time = 0.0
        cum_recover_time = 0.0

        for k in range(L - 2, -1, -1):
            t_val = t_array[k]

            # 10. Workers compute step RHS: g = s_p - A_pi @ lu_ii(s_i)
            # where s_i = C_coeff * c_ii * lambda_i_old (element-wise)
            #       s_p = C_coeff * c_pp * lambda_p_old (element-wise)
            t0_rhs = time_module.perf_counter()
            step_rhs_args = [(lambda_p_dict,)] * len(tile_configs)
            step_rhs_results = model.backend.call_all(
                model.workers, 'get_adjoint_step_reduced_rhs', step_rhs_args,
            )
            cum_rhs_time += time_module.perf_counter() - t0_rhs

            # 11. Coordinator assembles global_rhs + package cap history
            t0_solve = time_module.perf_counter()
            global_rhs = self._build_adjoint_global_rhs(
                tile_rhs_results=step_rhs_results,
                trans_ctx=trans_ctx,
                lambda_p_old=lambda_p_old,
                victim_node=None,  # Not terminal step
            )

            # 12. Coordinator solves interface
            lambda_p = trans_ctx.interface_lu(global_rhs)
            cum_solve_time += time_module.perf_counter() - t0_solve

            # Update lambda_p dict
            for i, node in enumerate(trans_ctx.interface_nodes):
                lambda_p_dict[node] = float(lambda_p[i])

            # 13. Workers recover lambda_i and accumulate at t = k * dt
            # Uses cached s_i/s_p from get_adjoint_step_reduced_rhs to avoid
            # transferring large arrays back through the coordinator.
            t0_recover = time_module.perf_counter()
            cached_recover_args = [
                (lambda_p_dict, t_val, include_waveforms)
            ] * len(tile_configs)

            model.backend.call_all(
                model.workers, 'recover_and_accumulate_adjoint_cached',
                cached_recover_args,
            )
            cum_recover_time += time_module.perf_counter() - t0_recover

            # Update lambda_p_old for next iteration
            lambda_p_old = lambda_p.copy()

            if verbose and (k % 10 == 0 or k == 0):
                logger.info(
                    "Backward step %d/%d (t=%.3e s)",
                    L - 1 - k, L - 1, t_val,
                )

        timings['backward_loop'] = time_module.perf_counter() - t0_loop
        timings['cum_rhs_time'] = cum_rhs_time
        timings['cum_solve_time'] = cum_solve_time
        timings['cum_recover_time'] = cum_recover_time

        # 14. Collect contributions from workers
        t0 = time_module.perf_counter()
        collect_results = model.backend.call_all(
            model.workers, 'get_adjoint_results',
        )

        # 15. Merge across tiles by node
        merged_contributions: Dict[str, float] = {}
        merged_source_to_node: Dict[str, str] = {}
        merged_waveforms: Dict[str, List[Tuple[float, float]]] = {}

        for contrib_dict, s2n_dict, wf_dict in collect_results:
            # Contributions: convert V to mV and sum across tiles
            for src_name, contrib_V in contrib_dict.items():
                contrib_mV = contrib_V * 1000.0
                merged_contributions[src_name] = (
                    merged_contributions.get(src_name, 0.0) + contrib_mV
                )
            # Source-to-node mapping (should be consistent across tiles)
            merged_source_to_node.update(s2n_dict)
            # Waveforms: merge lists
            for src_name, wf_list in wf_dict.items():
                if src_name not in merged_waveforms:
                    merged_waveforms[src_name] = []
                merged_waveforms[src_name].extend(wf_list)
        timings['collect_results'] = time_module.perf_counter() - t0

        # 16. Build AdjointAttribution
        t0 = time_module.perf_counter()
        attribution = self._build_attribution_result(
            victim_node=victim_node,
            ir_drop_at_T=ir_drop_at_T,
            contributions=merged_contributions,
            source_to_node=merged_source_to_node,
            waveforms=merged_waveforms,
            top_k=top_k,
            observation_time=observation_time,
            spatial_window=actual_window,
            n_candidate_sources=n_candidate_sources if n_candidate_sources >= 0 else len(merged_contributions),
            timings=timings,
            memory_window=(float(t_array[0]), float(t_array[-1])),
            t_array=t_array,
            initial_condition=initial_condition,
            dc_context=dc_context,
        )
        timings['build_result'] = time_module.perf_counter() - t0
        timings['total'] = time_module.perf_counter() - t0_total

        # Update timings in result
        attribution.timings.update(timings)

        if verbose:
            logger.info(
                "Adjoint dynamic analysis for '%s':\n"
                "  IR-drop at T: %.3f mV\n"
                "  Self contribution: %.3f mV (%.1f%%)\n"
                "  Top aggressor: %s (%.3f mV, %.1f%%)\n"
                "  Attribution efficiency: %.3f\n"
                "  Total time: %.3fs",
                victim_node,
                attribution.ir_drop_at_T,
                attribution.self_contribution_mV,
                attribution.self_contribution_pct,
                attribution.top_aggressors[0].node if attribution.top_aggressors else 'N/A',
                attribution.top_aggressors[0].contribution_mV if attribution.top_aggressors else 0.0,
                attribution.top_aggressors[0].contribution_pct if attribution.top_aggressors else 0.0,
                attribution.attribution_efficiency,
                timings['total'],
            )

        # Reset adjoint state on workers for next analysis
        model.backend.call_all(model.workers, 'reset_adjoint_state')

        return attribution

    # ------------------------------------------------------------------
    # Helper: build global RHS for adjoint interface solve
    # ------------------------------------------------------------------

    def _build_adjoint_global_rhs(
        self,
        tile_rhs_results: List[Tuple[np.ndarray, Dict]],
        trans_ctx: DistributedTransientContext,
        lambda_p_old: Optional[np.ndarray],
        victim_node: Optional[str] = None,
    ) -> np.ndarray:
        """Assemble global RHS for adjoint interface solve.

        For terminal step (lambda_p_old is None):
            - Scatter-add tile reduced RHS
            - Fix port victim double-counting (reset to 1.0)

        For non-terminal steps (lambda_p_old is not None):
            - Scatter-add tile reduced RHS
            - Add package cap history: C_coeff * C_package_uu @ lambda_p_old
            - NO Dirichlet RHS (pads have fixed voltage, zero adjoint)
            - NO G_package subtraction (adjoint B = C only)

        Args:
            tile_rhs_results: List of (reduced_rhs, stats) from workers.
            trans_ctx: Transient context with interface info.
            lambda_p_old: Previous step interface lambda values (None for terminal).
            victim_node: Victim node name for port fix (terminal step only).

        Returns:
            Global RHS array of shape (n_interface,).
        """
        model = self.model
        tile_configs = model.metadata.tile_configs

        n_interface = len(trans_ctx.interface_nodes)
        global_rhs = np.zeros(n_interface, dtype=np.float64)

        # Scatter-add reduced RHS from each tile.
        # S2/S13 (D1 follow-up): both get_adjoint_terminal_reduced_rhs and
        # get_adjoint_step_reduced_rhs return g_i in FULL port order (may
        # include a tile-resident pad port); filter+validate before
        # scattering (see solve_dc's identical fix).
        _kept_pos_map_adj = trans_ctx.tile_kept_port_pos
        _port_count_map_adj = trans_ctx.tile_port_count
        for i, (g_i, _stats) in enumerate(tile_rhs_results):
            tid = tile_configs[i].tile_id
            idx_map = trans_ctx.tile_index_maps[tid]
            g_i = filter_kept_rhs(
                g_i, tid, idx_map, _kept_pos_map_adj, _port_count_map_adj,
                caller='analyze_adjoint',
            )
            np.add.at(global_rhs, idx_map, g_i)

        if lambda_p_old is None:
            # Terminal step: fix port victim double-counting
            # Port victims appear on multiple tiles, causing scatter-add to
            # sum multiple unit vectors (e.g., global_rhs[victim_idx] = 2.0).
            # Reset to 1.0 for correct adjoint terminal condition.
            if victim_node is not None:
                victim_global_idx = trans_ctx.interface_node_to_idx.get(victim_node)
                if victim_global_idx is not None:
                    global_rhs[victim_global_idx] = 1.0
        else:
            # Non-terminal step: add package capacitance history term
            # global_rhs += C_coeff * C_package_uu @ lambda_p_old
            # NO Dirichlet RHS (pads have fixed voltage, zero adjoint contribution)
            # NO G_package subtraction (adjoint B = C only, not A = G + C*C)
            if trans_ctx.C_package_uu is not None:
                global_rhs += trans_ctx.C_coeff * (
                    trans_ctx.C_package_uu @ lambda_p_old
                )

        return global_rhs



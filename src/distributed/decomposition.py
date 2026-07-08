"""Distributed near/far IR-drop decomposition analysis.

Orchestrates the full decomposition workflow using the distributed DDM
solver:

1. Load pre-parsed tile data and create the distributed model.
2. DC solve (shared IC) + transient RC simulation to identify worst
   victims.
3. For each victim, run a near-only masked transient and compute
   far = all - near (by linearity).
4. Optionally run adjoint sensitivity to identify top aggressors
   within the near window.

The transient factorization (A = G + C_coeff * C) is current-independent
and is reused across all masked solves -- only the RHS changes.

Reuses data classes from :mod:`analysis.dynamic_irdrop_decomposition`
(``InstanceDecomposition``, ``DecompositionResult``, ``AggressorResult``)
and helpers from the notebook prototype.
"""

from __future__ import annotations

import logging
import math
import time as time_module
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from pgmath.factor import SolverBackendConfig

import numpy as np

from analysis.adjoint_sensitivity import AggressorContribution
from analysis.dynamic_irdrop_decomposition import (
    AggressorResult,
    DecompositionResult,
    InstanceDecomposition,
    compute_window_for_instance,
)
from distributed.tile_parsing import _parse_node_xy

logger = logging.getLogger(__name__)


# =========================================================================
# Helper functions (adapted from notebook)
# =========================================================================


def extract_instance_locations_from_peaks(
    peak_data: Dict[str, Tuple[float, float]],
) -> Tuple[Dict[str, Tuple[float, float]], Tuple[float, float, float, float]]:
    """Extract (x, y) coordinates for each node from peak data.

    The distributed solver does not expose a flat graph with instance
    source objects, so coordinates are parsed from node names directly
    using ``_parse_node_xy``.

    Args:
        peak_data: Dict mapping ``node -> (peak_drop, peak_time)``
            as returned by ``DistributedTransientResult.as_flat()``.

    Returns:
        Tuple of:
        - node_coords: Dict mapping node name to ``(x, y)`` tuple.
        - grid_bounds: ``(x_min, x_max, y_min, y_max)`` bounding box.
    """
    node_coords: Dict[str, Tuple[float, float]] = {}

    for node in peak_data:
        x, y = _parse_node_xy(node)
        if x is not None and y is not None:
            node_coords[node] = (float(x), float(y))

    if node_coords:
        xs = [c[0] for c in node_coords.values()]
        ys = [c[1] for c in node_coords.values()]
        grid_bounds = (min(xs), max(xs), min(ys), max(ys))
    else:
        grid_bounds = (0.0, 1.0, 0.0, 1.0)

    return node_coords, grid_bounds


def find_worst_nodes_separated(
    peak_data: Dict[str, Tuple[float, float]],
    node_coords: Dict[str, Tuple[float, float]],
    grid_bounds: Tuple[float, float, float, float],
    top_k: int,
    window_percent: float = 5.0,
) -> List[Tuple[str, float, float, float, float]]:
    """Find top-K worst nodes with spatial separation.

    Uses greedy selection: picks the worst node, then the next worst
    whose window does not overlap with any already-selected window, etc.

    Args:
        peak_data: Dict mapping ``node -> (peak_drop, peak_time)``.
        node_coords: Dict mapping ``node -> (x, y)``.
        grid_bounds: ``(x_min, x_max, y_min, y_max)`` bounding box.
        top_k: Number of worst nodes to select.
        window_percent: Window size as percentage of grid extent.

    Returns:
        List of ``(node, x, y, peak_drop, peak_time)`` tuples.
    """
    x_min_g, x_max_g, y_min_g, y_max_g = grid_bounds
    w = (x_max_g - x_min_g) * window_percent / 100.0
    h = (y_max_g - y_min_g) * window_percent / 100.0

    candidates = []
    for node, (drop, t) in peak_data.items():
        if node in node_coords:
            x, y = node_coords[node]
            candidates.append((node, x, y, drop, t))

    candidates.sort(key=lambda c: c[3], reverse=True)

    selected: List[Tuple[str, float, float, float, float]] = []
    selected_windows: List[Tuple[float, float, float, float]] = []

    for node, x, y, drop, t in candidates:
        if len(selected) >= top_k:
            break

        win = (x - w / 2, x + w / 2, y - h / 2, y + h / 2)
        overlaps = any(
            not (win[1] < sw[0] or sw[1] < win[0]
                 or win[3] < sw[2] or sw[3] < win[2])
            for sw in selected_windows
        )

        if not overlaps:
            selected.append((node, x, y, drop, t))
            selected_windows.append(win)

    return selected


def decompose_near_far(
    attr: Any,
    window: Tuple[float, float, float, float],
) -> Tuple[float, float, int, int, list]:
    """Partition adjoint contributions into near/far by spatial window.

    Self-contribution (victim's own sources) is classified as *near*.

    When ``attr.all_node_contributions`` is available (populated by the
    distributed adjoint solver), iterates the raw ``Dict[str, float]``
    for fast spatial totals.  ``AggressorContribution`` objects are built
    only for near-window nodes, avoiding the cost of materialising
    structured objects for all 200K+ sources.

    Falls back to iterating ``attr.top_aggressors`` when the raw dict is
    not present (flat adjoint path).

    Args:
        attr: ``AdjointAttribution`` result (from dynamic or static
            adjoint analysis).
        window: ``(x_min, x_max, y_min, y_max)`` spatial window.

    Returns:
        ``(near_mV, far_mV, n_near, n_far, near_aggressors_sorted)``
        where ``near_aggressors_sorted`` is a list of
        ``AggressorContribution`` objects inside the window, sorted by
        ``|contribution|`` descending.
    """
    x_min, x_max, y_min, y_max = window
    near_mV = attr.self_contribution_mV
    far_mV = 0.0
    n_near = 0
    n_far = 0

    # Fast path: iterate raw dict when available (distributed adjoint).
    node_contribs = attr.all_node_contributions
    if node_contribs is not None:
        near_nodes: List[Tuple[str, float]] = []
        for node, contrib_mV in node_contribs.items():
            x, y = _parse_node_xy(node)
            if (x is not None
                    and x_min <= x <= x_max
                    and y_min <= y <= y_max):
                near_mV += contrib_mV
                n_near += 1
                near_nodes.append((node, contrib_mV))
            else:
                # Outside window or coordinates not parseable -> classify as far
                far_mV += contrib_mV
                n_far += 1

        # Build AggressorContribution objects only for near aggressors.
        near_nodes.sort(key=lambda t: abs(t[1]), reverse=True)
        ir_drop = attr.ir_drop_at_T
        near_aggressors = []
        for node, contrib_mV in near_nodes:
            pct = 100.0 * contrib_mV / ir_drop if ir_drop > 0 else 0.0
            near_aggressors.append(AggressorContribution(
                node=node,
                contribution_mV=contrib_mV,
                contribution_pct=pct,
                source_names=[],
                current_waveform=None,
            ))
        return near_mV, far_mV, n_near, n_far, near_aggressors

    # Fallback: iterate top_aggressors (flat adjoint path).
    near_aggressors = []
    for agg in attr.top_aggressors:
        x, y = _parse_node_xy(agg.node)
        if x is not None and x_min <= x <= x_max and y_min <= y <= y_max:
            near_mV += agg.contribution_mV
            n_near += 1
            near_aggressors.append(agg)
        else:
            far_mV += agg.contribution_mV
            n_far += 1

    near_aggressors.sort(key=lambda a: abs(a.contribution_mV), reverse=True)
    return near_mV, far_mV, n_near, n_far, near_aggressors


def _grid_bounds_from_layer_metadata(
    tile_metadatas: List[Dict[str, Dict]],
) -> Tuple[float, float, float, float]:
    """Compute design-wide (x_min, x_max, y_min, y_max) from worker metadata.

    Merges per-tile per-layer bounding boxes into a single overall extent.
    Falls back to ``(0.0, 1.0, 0.0, 1.0)`` if no metadata is available.
    """
    x_mins: List[float] = []
    x_maxs: List[float] = []
    y_mins: List[float] = []
    y_maxs: List[float] = []

    for tile_meta in tile_metadatas:
        for _layer, info in tile_meta.items():
            bbox = info['bbox']  # (x_min, x_max, y_min, y_max)
            x_mins.append(bbox[0])
            x_maxs.append(bbox[1])
            y_mins.append(bbox[2])
            y_maxs.append(bbox[3])

    if x_mins:
        return (min(x_mins), max(x_maxs), min(y_mins), max(y_maxs))
    return (0.0, 1.0, 0.0, 1.0)


# =========================================================================
# Main orchestration function
# =========================================================================


def analyze_distributed_decomposition(
    pkl_dir: str,
    backend: str = 'local',
    t_start: float = 0.0,
    t_end: float = 100e-9,
    dt: float = 0.1e-9,
    top_k: int = 5,
    window_percent: float = 10.0,
    integration_method: str = 'be',
    instances: Optional[List[str]] = None,
    smooth_sources: bool = True,
    aggressor_top_k: int = 0,
    adjoint_method: str = 'dynamic',
    adjoint_memory_window: int = 20,
    verbose: bool = False,
    coordinator_solver_config: Optional[SolverBackendConfig] = None,
    worker_solver_config: Optional[SolverBackendConfig] = None,
    threads_per_worker: Optional[Any] = None,
) -> Tuple[DecompositionResult, Any, Any]:
    """Run distributed near/far IR-drop decomposition analysis.

    Performs the following phases:

    0. **Setup** -- Load tiles, create model and solver, preprocess
       current sources.
    1. **Factor** -- DC and transient factorization.
    2a. **DC solve** -- Shared initial condition for all transient solves.
    2b. **Initial transient** -- Full all-sources transient to identify
       the top-K worst victim nodes (skipped when ``instances`` is
       provided).  Reuses the DC result from 2a as ``ic_voltages``
       to avoid a redundant internal DC solve.
    3. **Waveform decomposition** -- For each victim, run a near-only
       masked transient, compute far = all - near.
    4. **Adjoint aggressor analysis** (optional) -- Identify top near-
       window aggressors per victim.
    5. **Build result** -- Assemble ``DecompositionResult`` and clean up
       contexts.

    Args:
        pkl_dir: Path to pre-parsed distributed pkl directory.
        backend: Compute backend (``'local'`` or ``'ray'``).
        t_start: Simulation start time in seconds.
        t_end: Simulation end time in seconds.
        dt: Time step in seconds.
        top_k: Number of worst victim nodes to analyze.
        window_percent: Near-window size as percentage of design extent.
        integration_method: ``'be'`` (Backward Euler) or ``'trap'``
            (Trapezoidal).
        instances: Pre-defined victim node names.  When provided, the
            initial transient (Phase 2) is skipped.
        smooth_sources: Apply PWL smoothing to current sources.
        aggressor_top_k: Number of top near-window aggressors per
            victim.  Set to 0 to disable adjoint analysis.
        adjoint_method: ``'static'`` or ``'dynamic'`` adjoint.
        adjoint_memory_window: Backward sweep memory window (number of
            time steps) for dynamic adjoint.
        verbose: Log progress.

    Returns:
        ``(result, solver, model)`` -- the ``DecompositionResult``,
        the ``DistributedDDMSolver``, and the
        ``DistributedPowerGridModel``.  Caller should call
        ``model.shutdown()`` when done.
    """
    # Lazy imports to avoid circular dependencies
    from distributed import (
        DistributedDDMSolver,
        create_distributed_model,
        load_distributed_partitions,
    )

    timings: Dict[str, float] = {}
    t0_total = time_module.perf_counter()

    # =================================================================
    # Phase 0: Setup
    # =================================================================
    t0 = time_module.perf_counter()
    if verbose:
        logger.info("Phase 0: Loading distributed partitions from %s", pkl_dir)

    bundle = load_distributed_partitions(pkl_dir)
    model = create_distributed_model(
        bundle, backend=backend,
        coordinator_solver_config=coordinator_solver_config,
        worker_solver_config=worker_solver_config,
        threads_per_worker=threads_per_worker,
    )
    solver = DistributedDDMSolver(model)
    timings['load_model'] = time_module.perf_counter() - t0

    # Guard: if anything below raises before the normal return,
    # shut down the model so Ray actors (or local workers) don't leak.
    # On the success path the caller owns the model and must call
    # model.shutdown() itself.
    success = False
    try:
        if verbose:
            logger.info(
                "Model: %d tiles (%dx%d), %d interface nodes, Vdd=%.3f V, net=%s",
                model.n_tiles, model.tile_grid[0], model.tile_grid[1],
                len(model.interface_nodes), model.vdd,
                model.net_name or 'unknown',
            )

        t0 = time_module.perf_counter()
        smoothed = solver.preprocess_sources(
            time_step=dt,
            t_start=t_start,
            t_end=t_end,
            smooth=smooth_sources,
            verbose=verbose,
        )
        timings['preprocess_sources'] = time_module.perf_counter() - t0

        # =============================================================
        # Phase 1: Factor
        # =============================================================
        if verbose:
            logger.info("Phase 1: Factoring DC and transient systems")

        t0 = time_module.perf_counter()
        dc_ctx = solver.prepare(verbose=verbose)
        timings['prepare_dc'] = time_module.perf_counter() - t0

        t0 = time_module.perf_counter()
        trans_ctx = solver.prepare_transient(
            dt=dt,
            method=integration_method,
            verbose=verbose,
        )
        timings['prepare_transient'] = time_module.perf_counter() - t0
        # =============================================================
        # Phase 2a: DC solve for shared IC (moved before transient to
        # avoid redundant DC solve inside solve_transient)
        # =============================================================
        if verbose:
            logger.info("Phase 2a: DC solve for shared initial condition")

        t0 = time_module.perf_counter()
        dc_result = solver.solve_dc(dc_ctx, verbose=False)
        timings['dc_solve'] = time_module.perf_counter() - t0

        # =============================================================
        # Phase 2b: Initial transient (find worst instances)
        # =============================================================
        peak_data: Dict[str, Tuple[float, float]] = {}
        node_coords: Dict[str, Tuple[float, float]] = {}
        grid_bounds: Tuple[float, float, float, float] = (0.0, 1.0, 0.0, 1.0)
        worst: List[Tuple[str, float, float, float, float]] = []
        peak_ir_drop_per_node: Dict[str, float] = {}
        trans_result = None  # Needed later for dynamic adjoint

        if instances is not None:
            # Skip initial transient -- build worst list from provided names
            if verbose:
                logger.info(
                    "Phase 2b: Skipped (using %d pre-defined instances)",
                    len(instances),
                )
            _coords: List[Tuple[float, float]] = []
            for node in instances:
                x, y = _parse_node_xy(node)
                if x is not None and y is not None:
                    _coords.append((float(x), float(y)))
                    worst.append((node, float(x), float(y), 0.0, 0.0))
                else:
                    logger.warning(
                        "Cannot parse coordinates for instance '%s'; "
                        "skipping",
                        node,
                    )

            # Compute design-wide grid_bounds from worker layer metadata.
            # Using only the victim coordinates would collapse the bounds
            # when few instances are provided (e.g. single point -> zero
            # extent).
            tile_metadatas = model.backend.call_all(
                model.workers, 'get_layer_metadata',
            )
            grid_bounds = _grid_bounds_from_layer_metadata(tile_metadatas)
        else:
            if verbose:
                logger.info(
                    "Phase 2b: Running initial transient "
                    "(%.2f ns to %.2f ns, dt=%.3f ns)",
                    t_start * 1e9, t_end * 1e9, dt * 1e9,
                )

            t0 = time_module.perf_counter()
            trans_result = solver.solve_transient(
                context=trans_ctx,
                ic_voltages=dc_result,
                t_start=t_start,
                t_end=t_end,
                smoothed_sources=smoothed,
                verbose=verbose,
            )
            timings['initial_transient'] = time_module.perf_counter() - t0

            if verbose:
                logger.info(
                    "  Peak IR-drop: %.3f mV at t=%.3f ns",
                    trans_result.peak_ir_drop * 1000,
                    trans_result.peak_ir_drop_time * 1e9,
                )

            # Collect peak data from workers
            t0 = time_module.perf_counter()
            peak_data = trans_result.as_flat()
            timings['collect_peaks'] = time_module.perf_counter() - t0

            node_coords, grid_bounds = extract_instance_locations_from_peaks(
                peak_data,
            )

            worst = find_worst_nodes_separated(
                peak_data, node_coords, grid_bounds, top_k, window_percent,
            )

            if verbose:
                logger.info(
                    "  Found %d spatially-separated worst nodes "
                    "(grid: X=[%.0f, %.0f], Y=[%.0f, %.0f])",
                    len(worst),
                    grid_bounds[0], grid_bounds[1],
                    grid_bounds[2], grid_bounds[3],
                )

            # Preserve peak IR-drop per node for heatmap generation
            peak_ir_drop_per_node = {
                node: drop for node, (drop, _) in peak_data.items()
            }

        if not worst:
            logger.warning("No victim nodes found; returning empty result")
            result = DecompositionResult(
                netlist_dir=pkl_dir,
                net_name=model.net_name or 'unknown',
                method='transient',
                integration_method=integration_method,
                t_start_ns=t_start * 1e9,
                t_end_ns=t_end * 1e9,
                dt_ns=dt * 1e9,
                window_percent=window_percent,
                grid_bounds=grid_bounds,
                worst_instances=[],
                timings=timings,
                smooth_sources=smooth_sources,
            )
            success = True
            return result, solver, model

        # =============================================================
        # Phase 3: Waveform decomposition
        # =============================================================
        if verbose:
            logger.info(
                "Phase 3: Waveform decomposition for %d victims",
                len(worst),
            )

        victim_nodes = [node for node, *_ in worst]
        n_tiles = model.n_tiles

        # All-sources transient tracking victim nodes
        t0 = time_module.perf_counter()
        if verbose:
            logger.info(
                "  Running all-sources transient (tracking %d victims)",
                len(victim_nodes),
            )

        all_trans = solver.solve_transient(
            context=trans_ctx,
            ic_voltages=dc_result,
            t_start=t_start,
            t_end=t_end,
            smoothed_sources=smoothed,
            track_nodes=victim_nodes,
            verbose=False,
        )
        timings['all_sources_transient'] = time_module.perf_counter() - t0

        decompositions: List[InstanceDecomposition] = []

        for rank, (node, x, y, peak_drop, peak_time) in enumerate(worst, 1):
            if verbose:
                logger.info("  Victim %d/%d: %s", rank, len(worst), node)

            window = compute_window_for_instance(x, y, grid_bounds, window_percent)

            # Build and set near mask on all workers (single round-trip).
            # Returns the active (masked-in) node count per tile.
            mask_args = [
                (window[0], window[1], window[2], window[3], True)
            ] * n_tiles
            active_counts = model.backend.call_all(
                model.workers, 'build_and_set_window_mask', mask_args,
            )

            # Near-only transient
            t0_near = time_module.perf_counter()
            near_trans = solver.solve_transient(
                context=trans_ctx,
                ic_voltages=dc_result,
                t_start=t_start,
                t_end=t_end,
                smoothed_sources=smoothed,
                track_nodes=[node],
                verbose=False,
            )
            if verbose:
                logger.info(
                    "    Near-only transient: %.2f s",
                    time_module.perf_counter() - t0_near,
                )

            # Clear mask (per-victim; also done defensively in finally)
            model.backend.call_all(
                model.workers,
                'set_current_node_mask',
                [(None,)] * n_tiles,
            )

            # Extract and align waveforms
            ir_all = all_trans.tracked_ir_drop.get(node)
            ir_near = near_trans.tracked_ir_drop.get(node)

            if ir_all is None or ir_near is None:
                logger.warning(
                    "    Waveform not tracked for %s; skipping", node,
                )
                continue

            n_common = min(len(ir_all), len(ir_near))
            if len(ir_all) != len(ir_near):
                logger.warning(
                    "    Waveform length mismatch for %s: "
                    "all=%d, near=%d; truncating to %d",
                    node, len(ir_all), len(ir_near), n_common,
                )
            ir_all_c = ir_all[:n_common]
            ir_near_c = ir_near[:n_common]
            ir_far_c = ir_all_c - ir_near_c
            t_arr = all_trans.t_array[:n_common]

            # Peak statistics
            peak_total = float(np.max(ir_all_c))
            peak_idx = int(np.argmax(ir_all_c))
            peak_time_s = float(t_arr[peak_idx])
            near_at_peak = float(ir_near_c[peak_idx])
            far_at_peak = float(ir_far_c[peak_idx])

            # Near/far source counts from worker mask active counts.
            # active_counts[i] is the number of current-source nodes
            # inside the near window for tile i (returned by
            # build_and_set_window_mask).
            n_near_srcs = sum(active_counts)
            total_interior = sum(model.tile_interior_counts.values())
            n_far_srcs = max(0, total_interior - n_near_srcs)

            # Fraction at peak
            if peak_total > 0:
                near_frac_peak = near_at_peak / peak_total * 100
                far_frac_peak = far_at_peak / peak_total * 100
            else:
                near_frac_peak = far_frac_peak = 0.0

            # Average statistics
            avg_total = float(np.mean(ir_all_c))
            avg_near = float(np.mean(ir_near_c))
            avg_far = float(np.mean(ir_far_c))

            # Average fraction over time
            safe_total = np.where(ir_all_c > 1e-15, ir_all_c, 1e-15)
            avg_near_frac = float(np.mean(ir_near_c / safe_total * 100))
            avg_far_frac = float(np.mean(ir_far_c / safe_total * 100))

            decomp = InstanceDecomposition(
                instance_name=node,
                node=node,
                x=x,
                y=y,
                window_bounds=window,
                n_near_sources=n_near_srcs,
                n_far_sources=n_far_srcs,
                t_array=t_arr,
                ir_drop_total=ir_all_c,
                ir_drop_near=ir_near_c,
                ir_drop_far=ir_far_c,
                peak_total_mV=peak_total * 1000,
                peak_near_mV=near_at_peak * 1000,
                peak_far_mV=far_at_peak * 1000,
                peak_time_ns=peak_time_s * 1e9,
                avg_total_mV=avg_total * 1000,
                avg_near_mV=avg_near * 1000,
                avg_far_mV=avg_far * 1000,
                near_fraction_at_peak=near_frac_peak,
                far_fraction_at_peak=far_frac_peak,
                avg_near_fraction=avg_near_frac,
                avg_far_fraction=avg_far_frac,
            )
            decompositions.append(decomp)

            if verbose:
                logger.info(
                    "    Peak: %.3f mV  Near: %.3f mV (%.1f%%)  "
                    "Far: %.3f mV (%.1f%%)",
                    peak_total * 1000, near_at_peak * 1000, near_frac_peak,
                    far_at_peak * 1000, far_frac_peak,
                )

        timings['waveform_decomposition'] = (
            time_module.perf_counter()
            - t0
            - timings.get('all_sources_transient', 0)
        )

        # =============================================================
        # Phase 4: Optional adjoint aggressor analysis
        # =============================================================
        if aggressor_top_k > 0 and decompositions:
            if verbose:
                logger.info(
                    "Phase 4: Adjoint aggressor analysis "
                    "(top-%d, method=%s)",
                    aggressor_top_k, adjoint_method,
                )

            t0_adjoint = time_module.perf_counter()

            # If we skipped the initial transient (instances provided)
            # but need a transient result for dynamic adjoint, run one.
            if trans_result is None and adjoint_method == 'dynamic':
                if verbose:
                    logger.info(
                        "  Running transient for dynamic adjoint "
                        "reference",
                    )
                trans_result = solver.solve_transient(
                    context=trans_ctx,
                    ic_voltages=dc_result,
                    t_start=t_start,
                    t_end=t_end,
                    smoothed_sources=smoothed,
                    verbose=False,
                )

            for rank, decomp in enumerate(decompositions, 1):
                if verbose:
                    logger.info(
                        "  Adjoint for victim %d/%d: %s",
                        rank, len(decompositions), decomp.node,
                    )

                peak_time_s = decomp.peak_time_ns * 1e-9
                window = decomp.window_bounds

                try:
                    if adjoint_method == 'static':
                        attr = solver.analyze_adjoint_static(
                            victim_nodes=decomp.node,
                            observation_time=peak_time_s,
                            dc_context=dc_ctx,
                            top_k=aggressor_top_k,
                            verbose=verbose,
                        )
                    else:
                        attr = solver.analyze_adjoint(
                            victim_nodes=decomp.node,
                            observation_time=peak_time_s,
                            trans_context=trans_ctx,
                            transient_result=trans_result,
                            memory_window=adjoint_memory_window,
                            top_k=aggressor_top_k,
                            verbose=verbose,
                        )

                    # Partition into near/far
                    near_mV, far_mV, n_near, n_far, near_aggressors = (
                        decompose_near_far(attr, window)
                    )

                    # Build AggressorResult list from top-K near
                    aggressor_results: List[AggressorResult] = []
                    for agg in near_aggressors[:aggressor_top_k]:
                        agg_x, agg_y = _parse_node_xy(agg.node)
                        if agg_x is not None:
                            distance = math.sqrt(
                                (agg_x - decomp.x) ** 2
                                + (agg_y - decomp.y) ** 2
                            )
                        else:
                            distance = 0.0

                        aggressor_results.append(
                            AggressorResult(
                                node=agg.node,
                                contribution_mV=agg.contribution_mV,
                                contribution_pct=agg.contribution_pct,
                                source_names=agg.source_names,
                                distance_um=distance,
                            )
                        )

                    decomp.top_aggressors = aggressor_results
                    decomp.near_total_mV = near_mV
                    decomp.self_contribution_mV = (
                        attr.self_contribution_mV
                    )
                    decomp.self_contribution_pct = (
                        attr.self_contribution_pct
                    )
                    decomp.attribution_efficiency = (
                        attr.attribution_efficiency
                    )

                    if verbose:
                        logger.info(
                            "    Adjoint: self=%.3f mV (%.1f%%), "
                            "near=%.3f mV, far=%.3f mV, "
                            "eff=%.3f",
                            attr.self_contribution_mV,
                            attr.self_contribution_pct,
                            near_mV, far_mV,
                            attr.attribution_efficiency,
                        )

                except Exception:
                    logger.exception(
                        "    Adjoint analysis failed for %s", decomp.node,
                    )

            timings['adjoint_analysis'] = (
                time_module.perf_counter() - t0_adjoint
            )

        # =============================================================
        # Phase 5: Build result and return
        # =============================================================
        if verbose:
            logger.info("Phase 5: Building result")

        # Total current waveform from the all-sources transient
        total_current = np.array(
            all_trans.total_current_per_time
        ) if all_trans.total_current_per_time is not None else np.array([])

        result_t_array = (
            all_trans.t_array
            if all_trans.t_array is not None
            else np.array([])
        )

        timings['total'] = time_module.perf_counter() - t0_total

        if verbose:
            logger.info("Analysis complete. Timing breakdown:")
            for key, val in timings.items():
                logger.info("  %s: %.3f s", key, val)

        result = DecompositionResult(
            netlist_dir=pkl_dir,
            net_name=model.net_name or 'unknown',
            method='transient',
            integration_method=integration_method,
            t_start_ns=t_start * 1e9,
            t_end_ns=t_end * 1e9,
            dt_ns=dt * 1e9,
            window_percent=window_percent,
            grid_bounds=grid_bounds,
            worst_instances=decompositions,
            timings=timings,
            peak_ir_drop_per_node=peak_ir_drop_per_node,
            total_current_waveform=total_current,
            t_array=result_t_array,
            smooth_sources=smooth_sources,
        )

        success = True
        return result, solver, model

    finally:
        # Defensive mask clear -- ignore errors (workers may already
        # be shut down if the exception occurred during teardown).
        try:
            model.backend.call_all(
                model.workers,
                'set_current_node_mask',
                [(None,)] * model.n_tiles,
            )
        except Exception:
            pass

        # Release contexts.  Each release() is idempotent, so calling
        # it even after a normal-path return (where the try block
        # already returned) is safe.  If prepare() or prepare_transient()
        # never ran, the NameError is caught harmlessly.
        try:
            trans_ctx.release()
        except Exception:
            pass
        try:
            dc_ctx.release()
        except Exception:
            pass

        # If we never reached the success return, shut down the model
        # to prevent leaked Ray actors / local workers.
        if not success:
            try:
                model.shutdown()
            except Exception:
                pass

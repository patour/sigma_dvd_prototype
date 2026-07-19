"""CLI for distributed PDN parser/solver pipeline.

Subcommands:
    parse      - Parse netlist, dump per-tile .pkl files
    solve      - Load .pkl partitions, run DDM solver
    run        - Parse + dump + solve in one shot
    decompose  - Near/far IR-drop decomposition analysis

Usage:
    python -m distributed parse  ./netlist/netlist_sampled --net VDD_XLV -o ./pkl_out
    python -m distributed solve  ./pkl_out -o ./results
    python -m distributed solve  ./pkl_out --mode quasi-static --t-end 100e-9 --n-points 51
    python -m distributed solve  ./pkl_out --mode transient --t-end 100e-9 --dt 0.1e-9
    python -m distributed run    ./netlist/netlist_sampled --net VDD_XLV -o ./results
    python -m distributed decompose ./netlist/netlist_sampled --net VDD_XLV --top-k 2 --verbose
"""

from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_LOG_FORMAT = '%(asctime)s %(name)s %(levelname)s: %(message)s'
_LOG_DATEFMT = '%H:%M:%S'


def _setup_logging(verbose: bool) -> None:
    """Configure logging for CLI usage."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format=_LOG_FORMAT, datefmt=_LOG_DATEFMT)


def _add_file_logging(output_dir: str, mode: str) -> Optional[logging.FileHandler]:
    """Add a file handler to the root logger.

    The caller should close the handler when done via
    ``_close_file_logging(handler)``.

    Args:
        output_dir: Directory for the log file.  Created if it does not exist.
        mode: Analysis mode label (e.g. ``'dc'``, ``'quasi-static'``).

    Returns:
        The ``FileHandler``, or ``None`` if *output_dir* is falsy.
    """
    if not output_dir:
        return None

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    mode_slug = mode.replace('-', '_')
    log_path = out_path / f'{mode_slug}_{timestamp}.log'

    file_handler = logging.FileHandler(str(log_path), mode='w')
    file_handler.setLevel(logging.NOTSET)
    file_handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATEFMT))
    logging.getLogger().addHandler(file_handler)
    logger.info("Logging to file: %s", log_path)
    return file_handler


def _close_file_logging(handler: Optional[logging.FileHandler]) -> None:
    """Remove and close a file handler previously added by ``_add_file_logging``."""
    if handler is None:
        return
    logging.getLogger().removeHandler(handler)
    handler.close()


def cmd_parse(args: argparse.Namespace) -> None:
    """Parse netlist and dump per-tile .pkl files."""
    from .parser import DistributedNetlistParser

    _setup_logging(args.verbose)

    out_dir = args.output or str(Path(args.netlist_dir) / 'distributed_pkl')
    fh = _add_file_logging(out_dir, 'parse')

    try:
        t0 = time.perf_counter()

        parser = DistributedNetlistParser(args.netlist_dir, net_filter=args.net)
        max_interior = getattr(args, 'max_interior', None)
        out_path, _bundle = parser.parse_and_dump(
            out_dir, backend=args.backend, max_interior=max_interior,
        )

        elapsed = time.perf_counter() - t0
        logger.info(f"parse_and_dump completed in {elapsed:.3f}s -> {out_path}")
    finally:
        _close_file_logging(fh)


# Built-in defaults for the interface-solver settings, applied by
# _load_and_apply_config() AFTER the explicit-CLI > YAML precedence check
# resolves to "neither was set" (finding 2).  Also used as the getattr()
# fallback in _build_interface_settings() below for callers (tests, direct
# analyze_distributed_decomposition() calls) that construct a bare
# argparse.Namespace without going through _load_and_apply_config.
#
# Stage 2 additions: interface_matvec_mode default 'assembled' -> 'auto'
# (item 8 -- tilewise whenever per-tile Schur blocks are available);
# matvec_threads (no 'interface_' prefix, matching the plan's literal
# naming), interface_matvec_dtype, interface_strict_dtype_rtol,
# interface_drop_s_global (item 3).
#
# Stage 3: interface_preconditioner default 'block_jacobi' -> 'auto' (same
# pattern as the Stage 2 matvec_mode change above) -- 'auto' resolves to
# 'two_level' whenever CG + tilewise matvec is selected (the regime Stage 2
# measured plain block_jacobi CG stagnating in), else the legacy
# 'block_jacobi' default (see interface_iterative.resolve_preconditioner,
# the ONE place 'auto' is resolved). New interface_coarse_* knobs control
# the coarse-space build; see InterfaceCGSolver's docstring.
_IFACE_SETTING_DEFAULTS: Dict[str, Any] = {
    'interface_solver': 'auto',
    'interface_matvec_mode': 'auto',
    'interface_preconditioner': 'auto',
    'interface_cg_rtol': 1e-8,
    'interface_cg_atol': 1e-14,
    'interface_cg_maxiter': None,
    'interface_cg_strict': True,
    'interface_factor_memory_budget': 'auto',
    'interface_block_jacobi_max_bytes': 'auto',
    'matvec_threads': 'auto',
    'interface_matvec_dtype': 'float64',
    'interface_strict_dtype_rtol': True,
    'interface_drop_s_global': False,
    # Finding 15: these four literals are resolved lazily from
    # interface_coarse.py's own DEFAULT_GENEO_K/DEFAULT_GENEO_TOL/
    # DEFAULT_EPS_RANK/DEFAULT_MAX_COLS by _iface_default() below, NOT
    # re-hardcoded here -- see that function's docstring for why this stays
    # a lazy (not module-level) import.
    'interface_coarse_geneo_k': None,
    'interface_coarse_geneo_tol': None,
    'interface_coarse_eps_rank': None,
    'interface_coarse_max_cols': None,
    'interface_coarse_max_bytes': 'auto',
}

# Finding 15: keys whose real default is resolved lazily via
# interface_coarse.py's own DEFAULT_* constants (see _iface_default()) --
# the corresponding _IFACE_SETTING_DEFAULTS entries above are placeholder
# Nones, never read directly.
_COARSE_DEFAULT_KEYS = frozenset((
    'interface_coarse_geneo_k', 'interface_coarse_geneo_tol',
    'interface_coarse_eps_rank', 'interface_coarse_max_cols',
))


def _iface_default(key: str) -> Any:
    """Resolve an ``_IFACE_SETTING_DEFAULTS`` entry (Finding 15).

    The four Stage-3 coarse-space column/rank knobs are resolved from
    ``interface_coarse.DEFAULT_GENEO_K`` et al. (the single canonical
    source) instead of being re-hardcoded a third time here.
    ``interface_iterative.py``'s own ``InterfaceCGSolver.__init__``/
    ``build_interface_solver`` signatures use ``None``-sentinel defaults and
    resolve from the SAME canonical source dynamically, at call time
    (Finding 9, round 2) -- not a def-time-bound copy, which would defeat
    ``monkeypatch.setattr(interface_coarse, 'DEFAULT_GENEO_K', ...)``.  The
    import here is lazy
    (function-local, not module-level) to preserve cli.py's existing
    convention of deferring every internal-package import so a plain
    ``--help``/argparse-only invocation doesn't pay for pulling in
    numpy/scipy (``interface_coarse.py`` imports both) -- see every other
    ``from .xxx import ...`` in this file, all function-local for the same
    reason.
    """
    if key in _COARSE_DEFAULT_KEYS:
        from . import interface_coarse
        return {
            'interface_coarse_geneo_k': interface_coarse.DEFAULT_GENEO_K,
            'interface_coarse_geneo_tol': interface_coarse.DEFAULT_GENEO_TOL,
            'interface_coarse_eps_rank': interface_coarse.DEFAULT_EPS_RANK,
            'interface_coarse_max_cols': interface_coarse.DEFAULT_MAX_COLS,
        }[key]
    return _IFACE_SETTING_DEFAULTS[key]


def _build_interface_settings(args: argparse.Namespace) -> Dict[str, Any]:
    """Build the ``interface_*`` model.settings dict from parsed CLI args.

    Shared by ``cmd_solve``, ``cmd_run``, and ``cmd_decompose`` so all three
    subcommands push the same six interface-CG flags (plus
    interface_solver/matvec_mode/preconditioner/rtol) into model.settings
    identically.  Previously ``cmd_decompose`` accepted these flags on its
    subparser (via ``_add_config_and_solver_args``) but never pushed them
    anywhere -- ``analyze_distributed_decomposition`` built the model without
    them, so they were silently ignored (finding 3).  Factoring the push
    into one helper (used via ``analyze_distributed_decomposition``'s
    ``interface_settings=`` kwarg for decompose, and directly on
    ``model.settings`` for solve/run) avoids re-duplicating the same nine
    keys a third time.

    ``interface_cg_maxiter`` is the one key whose real default (``None``)
    is indistinguishable from "unset" -- that is correct (``None`` IS the
    default), so it is read as-is rather than substituted.
    """
    def _get(key: str) -> Any:
        val = getattr(args, key, None)
        return val if val is not None else _iface_default(key)

    return {
        'interface_solver': _get('interface_solver'),
        'interface_matvec_mode': _get('interface_matvec_mode'),
        'interface_preconditioner': _get('interface_preconditioner'),
        'interface_cg_rtol': _get('interface_cg_rtol'),
        'interface_cg_atol': _get('interface_cg_atol'),
        'interface_cg_maxiter': getattr(args, 'interface_cg_maxiter', None),
        'interface_cg_strict': _get('interface_cg_strict'),
        'interface_factor_memory_budget': _get('interface_factor_memory_budget'),
        'interface_block_jacobi_max_bytes': _get('interface_block_jacobi_max_bytes'),
        # Stage 2
        'matvec_threads': _get('matvec_threads'),
        'interface_matvec_dtype': _get('interface_matvec_dtype'),
        'interface_strict_dtype_rtol': _get('interface_strict_dtype_rtol'),
        'interface_drop_s_global': _get('interface_drop_s_global'),
        # Stage 3: two-level coarse-space preconditioner knobs.
        'interface_coarse_geneo_k': _get('interface_coarse_geneo_k'),
        'interface_coarse_geneo_tol': _get('interface_coarse_geneo_tol'),
        'interface_coarse_eps_rank': _get('interface_coarse_eps_rank'),
        'interface_coarse_max_cols': _get('interface_coarse_max_cols'),
        'interface_coarse_max_bytes': _get('interface_coarse_max_bytes'),
    }


def _push_interface_settings(
    model: Any, args: argparse.Namespace, verbose: bool = False,
) -> Dict[str, Any]:
    """Push interface-solver settings into ``model.settings`` (coordinator-side).

    Returns the settings dict that was pushed (callers that want the
    resolved values for logging can reuse it instead of re-reading args).
    """
    settings = _build_interface_settings(args)
    model.settings.update(settings)
    if verbose:
        logger.info(
            "Interface solver: %s (matvec=%s, precond=%s, rtol=%.2e)",
            settings['interface_solver'], settings['interface_matvec_mode'],
            settings['interface_preconditioner'], settings['interface_cg_rtol'],
        )
    return settings


def _resolve_island_detection_arg(args: argparse.Namespace) -> str:
    """Read ``args.island_detection`` without falsy-coercing it (finding F7).

    ``_load_and_apply_config`` already resolves ``args.island_detection`` to
    a definite value (explicit CLI flag > YAML > built-in default 'auto') for
    every subcommand that calls it before reaching ``create_distributed_model``
    -- so this is a plain ``None``-check passthrough for that normal case.

    The bug this replaces: ``getattr(args, 'island_detection', None) or
    'auto'`` silently coerced ANY falsy-but-non-None value (e.g. ``False``,
    which PyYAML 1.1 parses ``island_detection: off``/``no`` to) to 'auto',
    defeating ``model._resolve_island_detection``'s loud ``ValueError`` for
    invalid settings.  Only a genuine ``None`` (island_detection truly never
    set -- e.g. a bare ``argparse.Namespace`` built directly by a test/caller
    that skipped ``_load_and_apply_config``) falls back to 'auto' here.
    """
    val = getattr(args, 'island_detection', None)
    return val if val is not None else 'auto'


def cmd_solve(args: argparse.Namespace) -> None:
    """Load .pkl partitions and run DDM solver."""
    from .model import create_distributed_model, load_distributed_partitions
    from .solver import DistributedDDMSolver

    _setup_logging(args.verbose)
    args = _load_and_apply_config(args)

    mode = getattr(args, 'mode', 'dc')
    fh = _add_file_logging(args.output, mode)

    t0 = time.perf_counter()

    bundle = load_distributed_partitions(args.pkl_dir)

    model = create_distributed_model(
        bundle,
        backend=args.backend,
        coordinator_solver_config=getattr(args, 'coordinator_solver_config', None),
        worker_solver_config=getattr(args, 'worker_solver_config', None),
        threads_per_worker=_parse_threads_per_worker(
            getattr(args, 'threads_per_worker', None)
        ),
        tiles_per_worker=_parse_threads_per_worker(
            getattr(args, 'tiles_per_worker', None)
        ),
        island_detection=_resolve_island_detection_arg(args),
    )

    # B2/Stage 1: Push interface-solver settings into model.settings (coordinator-side)
    _push_interface_settings(model, args, verbose=args.verbose)

    # B3: Push streaming_assembly / use_step_columns / max_table_mb into model.settings
    _push_b3_settings(model, args)

    try:
        solver = DistributedDDMSolver(model)

        if mode == 'dc':
            ctx = solver.prepare(verbose=args.verbose)
            _solve_dc(solver, ctx, args, t0)
        elif mode == 'quasi-static':
            ctx = solver.prepare(verbose=args.verbose)
            _solve_quasi_static(solver, ctx, args, t0)
        elif mode == 'transient':
            _solve_transient(solver, args, t0)
        else:
            logger.error("Unknown mode: %s", mode)
            raise SystemExit(1)
    finally:
        model.shutdown()
        _close_file_logging(fh)


def _push_b3_settings(model: Any, args: argparse.Namespace) -> None:
    """Push B3 streaming-assembly and A2 step-column settings into model.settings.

    Reads from args attributes populated by YAML or CLI:
        streaming_assembly  — None (skip) | False | True | 'auto' | 'false'|'true' string
        use_step_columns    — bool (optional, only set when present on args)
        max_table_mb        — float (optional, only set when present on args)
    """
    sa_raw = getattr(args, 'streaming_assembly', None)
    if sa_raw is not None:
        # Normalise string choices from CLI ('false'/'true'/'auto') to bool/'auto'
        if isinstance(sa_raw, str):
            if sa_raw.lower() == 'false':
                sa_val: Any = False
            elif sa_raw.lower() == 'true':
                sa_val = True
            else:
                sa_val = 'auto'
        else:
            sa_val = bool(sa_raw)
        model.settings['streaming_assembly'] = sa_val

    usc = getattr(args, 'use_step_columns', None)
    if usc is not None:
        model.settings['use_step_columns'] = bool(usc)

    mtm = getattr(args, 'max_table_mb', None)
    if mtm is not None:
        model.settings['max_table_mb'] = float(mtm)


def _solve_dc(
    solver: 'DistributedDDMSolver',
    ctx: 'DistributedSolverContext',
    args: argparse.Namespace,
    t0: float,
) -> None:
    """Run DC solve and report results (original behavior)."""
    result = solver.solve_dc(ctx, verbose=args.verbose)

    # Report summary
    v_all = result.flatten()
    ir_drop = result.ir_drop

    elapsed = time.perf_counter() - t0
    if ir_drop:
        max_drop_node = max(ir_drop, key=ir_drop.get)
        max_drop_mv = ir_drop[max_drop_node] * 1e3
        logger.info(
            f"Solve completed in {elapsed:.3f}s: "
            f"{len(v_all)} nodes, max IR-drop = {max_drop_mv:.3f} mV "
            f"at {max_drop_node}"
        )
    else:
        logger.warning(
            f"Solve completed in {elapsed:.3f}s: "
            f"{len(v_all)} nodes, but ir_drop dict is empty "
            f"(no non-pad nodes solved?)"
        )

    # Optionally save results
    if args.output:
        import pickle
        out_path = Path(args.output)
        out_path.mkdir(parents=True, exist_ok=True)
        result_pkl = out_path / 'ddm_result.pkl'
        with open(result_pkl, 'wb') as f:
            pickle.dump(
                {
                    'voltages': v_all,
                    'ir_drop': ir_drop,
                    'nominal_voltage': result.nominal_voltage,
                    'net_name': result.net_name,
                    'timings': result.solve_metadata.get('timings', {}),
                },
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        logger.info(f"Results saved to {result_pkl}")

    # Optionally generate heatmaps (must happen before model.shutdown())
    if args.plot:
        plot_layers = args.plot_layers.split(',') if args.plot_layers else None
        solver.generate_reports(
            result,
            context=ctx,
            output_dir=args.output or './results',
            plot_layers=plot_layers,
            max_stripes=args.max_stripes,
            stripe_bin_size=args.stripe_bin_size,
            show_irdrop=args.show_irdrop,
            top_k=args.top_k,
            verbose=args.verbose,
        )


def _solve_quasi_static(
    solver: 'DistributedDDMSolver',
    ctx: 'DistributedSolverContext',
    args: argparse.Namespace,
    t0: float,
) -> None:
    """Run quasi-static (batch DC) time-domain solve."""
    smooth = getattr(args, 'smooth', True)

    # Preprocess sources with smoothing controlled by CLI flag
    dt = (args.t_end - args.t_start) / max(args.n_points - 1, 1)
    smoothed_sources = solver.preprocess_sources(
        time_step=dt,
        t_start=args.t_start,
        t_end=args.t_end,
        smooth=smooth,
        verbose=args.verbose,
    )

    result = solver.solve_quasi_static(
        ctx,
        t_start=args.t_start,
        t_end=args.t_end,
        n_points=args.n_points,
        smoothed_sources=smoothed_sources,
        verbose=args.verbose,
    )

    _report_time_domain_result(result, args, t0, mode='quasi-static', solver=solver)


def _solve_transient(
    solver: 'DistributedDDMSolver',
    args: argparse.Namespace,
    t0: float,
) -> None:
    """Run transient (RC) time-domain solve."""
    smooth = getattr(args, 'smooth', True)

    # Preprocess sources with smoothing controlled by CLI flag
    smoothed_sources = solver.preprocess_sources(
        time_step=args.dt,
        t_start=args.t_start,
        t_end=args.t_end,
        smooth=smooth,
        verbose=args.verbose,
    )

    # Create DC context for initial condition, then transient context
    dc_ctx = solver.prepare(verbose=args.verbose)
    trans_ctx = solver.prepare_transient(
        dt=args.dt, method=args.method, verbose=args.verbose,
    )

    result = solver.solve_transient(
        trans_ctx,
        dc_context=dc_ctx,
        t_start=args.t_start,
        t_end=args.t_end,
        smoothed_sources=smoothed_sources,
        verbose=args.verbose,
    )

    dc_ctx.release()
    trans_ctx.release()
    _report_time_domain_result(result, args, t0, mode='transient', solver=solver)


def _report_time_domain_result(
    result: Any,
    args: argparse.Namespace,
    t0: float,
    mode: str,
    solver: Any = None,
) -> None:
    """Print summary and optionally save a time-domain result.

    Works for both DistributedQuasiStaticResult and DistributedTransientResult.
    """
    elapsed = time.perf_counter() - t0
    n_steps = len(result.t_array)

    logger.info(
        f"{mode} solve completed in {elapsed:.3f}s: "
        f"{n_steps} time steps, "
        f"peak IR-drop = {result.peak_ir_drop * 1e3:.3f} mV "
        f"at t = {result.peak_ir_drop_time:.3e} s"
    )

    # Optionally save results
    if args.output:
        out_path = Path(args.output)
        out_path.mkdir(parents=True, exist_ok=True)
        result_pkl = out_path / f'ddm_{mode.replace("-", "_")}_result.pkl'
        result.dump(str(result_pkl))
        logger.info(f"Results saved to {result_pkl}")

    if args.plot and solver is not None:
        plot_layers = args.plot_layers.split(',') if args.plot_layers else None
        solver.generate_td_reports(
            result,
            output_dir=args.output or './results',
            plot_layers=plot_layers,
            max_stripes=args.max_stripes,
            stripe_bin_size=args.stripe_bin_size,
            top_k=args.top_k,
            verbose=args.verbose,
        )
    elif args.plot:
        logger.warning("Cannot generate heatmaps: solver reference not available.")


def cmd_run(args: argparse.Namespace) -> None:
    """Parse + dump + solve in one shot."""
    from .model import create_distributed_model
    from .parser import DistributedNetlistParser
    from .solver import DistributedDDMSolver

    _setup_logging(args.verbose)
    args = _load_and_apply_config(args)

    mode = getattr(args, 'mode', 'dc')
    fh = _add_file_logging(args.output, mode)

    t_total = time.perf_counter()

    # Parse and dump
    t0 = time.perf_counter()
    parser = DistributedNetlistParser(args.netlist_dir, net_filter=args.net)
    pkl_dir = args.pkl_dir or str(Path(args.netlist_dir) / 'distributed_pkl')
    max_interior = getattr(args, 'max_interior', None)
    _out_path, bundle = parser.parse_and_dump(
        pkl_dir, backend=args.backend, max_interior=max_interior,
    )
    t_parse = time.perf_counter() - t0
    logger.info(f"Parse phase: {t_parse:.3f}s")

    # Create model from bundle
    t0 = time.perf_counter()
    model = create_distributed_model(
        bundle,
        backend=args.backend,
        coordinator_solver_config=getattr(args, 'coordinator_solver_config', None),
        worker_solver_config=getattr(args, 'worker_solver_config', None),
        threads_per_worker=_parse_threads_per_worker(
            getattr(args, 'threads_per_worker', None)
        ),
        tiles_per_worker=_parse_threads_per_worker(
            getattr(args, 'tiles_per_worker', None)
        ),
        island_detection=_resolve_island_detection_arg(args),
    )

    # B2/Stage 1: Push interface-solver settings into model.settings (coordinator-side)
    _push_interface_settings(model, args, verbose=args.verbose)

    # B3: Push streaming_assembly / use_step_columns / max_table_mb into model.settings
    _push_b3_settings(model, args)

    t_model = time.perf_counter() - t0
    logger.info(f"Model creation: {t_model:.3f}s")

    try:
        solver = DistributedDDMSolver(model)

        if mode == 'dc':
            t0_prepare = time.perf_counter()
            ctx = solver.prepare(verbose=args.verbose)
            t_prepare = time.perf_counter() - t0_prepare
            logger.info(f"Prepare phase: {t_prepare:.3f}s")
            _solve_dc(solver, ctx, args, t_total)
        elif mode == 'quasi-static':
            t0_prepare = time.perf_counter()
            ctx = solver.prepare(verbose=args.verbose)
            t_prepare = time.perf_counter() - t0_prepare
            logger.info(f"Prepare phase: {t_prepare:.3f}s")
            _solve_quasi_static(solver, ctx, args, t_total)
        elif mode == 'transient':
            _solve_transient(solver, args, t_total)
        else:
            logger.error("Unknown mode: %s", mode)
            raise SystemExit(1)
    finally:
        model.shutdown()
        _close_file_logging(fh)


def cmd_decompose(args: argparse.Namespace) -> None:
    """Run distributed near/far IR-drop decomposition analysis."""
    from .decomposition import analyze_distributed_decomposition
    from analysis.dynamic_irdrop_decomposition import (
        Logger,
        generate_plots,
        print_results,
    )

    _setup_logging(args.verbose)

    # Load decompose-specific YAML config before the shared cholmod handler.
    decompose_config: dict = {}
    if getattr(args, 'config', None):
        decompose_config = _load_decompose_config(args.config)
        # Prevent _load_and_apply_config from re-loading the same file
        # via pdn_solver.merge_config_with_args (wrong schema).  The
        # solver/cholmod keys are already handled by _merge_decompose_config.
        args.config = None

    # Merge decompose-specific parameters (CLI takes precedence).
    # This also resolves args.smooth from None -> True/False.
    if decompose_config:
        args = _merge_decompose_config(decompose_config, args)

    # Resolve smooth sentinel when no config was loaded.
    if args.smooth is None:
        args.smooth = True

    # Apply cholmod / solver backend settings (reads args.use_cholmod etc.).
    args = _load_and_apply_config(args)
    t0 = time.perf_counter()

    # Derive pkl_dir from netlist_dir.  Auto-parse if tile pkls don't exist.
    netlist_dir = args.netlist_dir
    net_filter = getattr(args, 'net', None)
    pkl_dir = str(Path(netlist_dir) / 'distributed_pkl')

    tile_pkls = list(Path(pkl_dir).glob('tile_*.pkl'))
    if not tile_pkls:
        from .parser import DistributedNetlistParser

        logger.info(
            "No tile pkls in %s — parsing netlist %s (net=%s)",
            pkl_dir, netlist_dir, net_filter or 'all',
        )
        t_parse = time.perf_counter()
        parser_obj = DistributedNetlistParser(
            netlist_dir, net_filter=net_filter,
        )
        parser_obj.parse_and_dump(pkl_dir, backend=args.backend)
        logger.info("Parse phase: %.3fs", time.perf_counter() - t_parse)
    else:
        logger.info(
            "Using %d cached tile pkls from %s",
            len(tile_pkls), pkl_dir,
        )

    # Parse --instances if provided (comma-separated node names)
    instances = None
    if args.instances:
        instances = [s.strip() for s in args.instances.split(',') if s.strip()]

    output_dir = args.output
    fh = _add_file_logging(output_dir, 'decompose')

    result, solver, model = None, None, None
    try:
        result, solver, model = analyze_distributed_decomposition(
            pkl_dir=pkl_dir,
            backend=args.backend,
            t_start=args.t_start,
            t_end=args.t_end,
            dt=args.dt,
            top_k=args.top_k,
            window_percent=args.window_percent,
            integration_method=args.method,
            instances=instances,
            smooth_sources=args.smooth,
            aggressor_top_k=args.aggressor_top_k,
            adjoint_method=args.adjoint_method,
            adjoint_memory_window=args.adjoint_memory_window,
            qs_candidate_factor=getattr(args, 'qs_candidate_factor', 3000),
            max_qs_candidates=getattr(args, 'max_qs_candidates', 10000),
            verbose=args.verbose,
            coordinator_solver_config=getattr(args, 'coordinator_solver_config', None),
            worker_solver_config=getattr(args, 'worker_solver_config', None),
            threads_per_worker=_parse_threads_per_worker(
                getattr(args, 'threads_per_worker', None)
            ),
            # B2/Stage 1 (finding 3): push the same interface_* settings as
            # cmd_solve/cmd_run so --interface-cg-* flags actually reach the
            # decompose-side model instead of being silently accepted and
            # dropped.
            interface_settings=_build_interface_settings(args),
            island_detection=_resolve_island_detection_arg(args),
        )

        elapsed = time.perf_counter() - t0
        logger.info(
            "Decomposition completed in %.3fs: %d victims analyzed",
            elapsed, len(result.worst_instances),
        )

        # Print results to console and log file
        log_file = str(Path(output_dir) / 'analysis.log')
        log = Logger(log_file)
        try:
            print_results(result, log)
        finally:
            log.close()

        # Save JSON
        json_file = str(Path(output_dir) / 'results.json')
        result.save_json(json_file)
        logger.info("Results saved to %s", json_file)

        # Generate plots
        if not args.no_plot:
            plot_dir = str(Path(output_dir) / 'plots')
            heatmap_layers = (
                args.plot_layers.split(',') if args.plot_layers else None
            )
            generate_plots(
                result,
                plot_dir,
                show=False,
                heatmap_layers=heatmap_layers,
                max_stripes=args.max_stripes,
                verbose=args.verbose,
            )

    finally:
        if model is not None:
            model.shutdown()
        _close_file_logging(fh)


def _add_config_and_solver_args(parser: argparse.ArgumentParser) -> None:
    """Add --config, cholmod backend, and profiling flags to a subparser.

    Shared between the ``solve`` and ``run`` subcommands so the flag set is
    consistent with ``pdn_solver.py``'s CLI.
    """
    # Config file
    parser.add_argument('--config', '-c', type=str, default=None,
                        help='Config file path (.yaml, .yml, or .json)')

    # Solver backend (cholmod / splu)
    from pgmath.factor import _VALID_CHOLMOD_MODES, _VALID_CHOLMOD_ORDERINGS
    _modes = list(_VALID_CHOLMOD_MODES)
    _orderings = list(_VALID_CHOLMOD_ORDERINGS)

    parser.add_argument('--use-cholmod', action='store_true', default=None,
                        help='Force cholmod backend (requires sksparse)')
    parser.add_argument('--use-splu', action='store_true',
                        help='Force splu backend (scipy)')
    parser.add_argument('--cholmod-mode', type=str, default='auto',
                        choices=_modes,
                        help='Cholmod factorization mode (default: auto)')
    parser.add_argument('--cholmod-ordering', type=str, default='default',
                        choices=_orderings,
                        help='Cholmod fill-reducing ordering (default: default)')
    parser.add_argument('--cholmod-use-long', action='store_true', default=None,
                        help='Force 64-bit indices in cholmod')

    # Per-role solver backend overrides (coordinator / worker)
    for role in ('coordinator', 'worker'):
        grp = parser.add_argument_group(f'{role} solver backend overrides')
        grp.add_argument(f'--{role}-use-cholmod', action='store_true', default=None,
                         help=f'Force cholmod backend for {role}')
        grp.add_argument(f'--{role}-use-splu', action='store_true',
                         help=f'Force splu backend for {role}')
        grp.add_argument(f'--{role}-cholmod-mode', type=str, default=None,
                         choices=_modes,
                         help=f'Cholmod mode for {role}')
        grp.add_argument(f'--{role}-cholmod-ordering', type=str, default=None,
                         choices=_orderings,
                         help=f'Cholmod ordering for {role}')
        grp.add_argument(f'--{role}-cholmod-use-long', action='store_true', default=None,
                         help=f'Force 64-bit indices for {role}')

    # Per-actor threading (Ray backend only)
    parser.add_argument(
        '--threads-per-worker', type=str, default=None,
        dest='threads_per_worker',
        help=(
            'Threads per Ray worker actor for BLAS/OMP (int or "auto"). '
            '"auto" = max(1, cpus // n_workers). '
            'No effect on local backend. (default: system default)'
        ),
    )

    # Worker packing (B1: useful when retiling produces tile count >> cores)
    parser.add_argument(
        '--tiles-per-worker', type=str, default=None,
        dest='tiles_per_worker',
        help=(
            'Pack multiple tiles per worker (int or "auto"). '
            '"auto" = ceil(n_tiles / n_cpus). '
            'Reduces actor overhead when --max-interior splitting produces '
            'many sub-tiles. (default: one tile per worker)'
        ),
    )

    # B2: Interface solver selection
    #
    # NOTE on defaults: every flag in this group defaults to None at the
    # argparse level, NOT its real default value (documented in each --help
    # string below).  This is required for correct YAML-vs-CLI precedence
    # (finding 2): _load_and_apply_config() distinguishes "the user passed
    # this flag" (argparse value is not None) from "the user left it unset"
    # (argparse value is None) and only then falls through to YAML, then to
    # the real built-in default (see _IFACE_SETTING_DEFAULTS below).  A
    # fixed default value (e.g. True for interface_cg_strict) would be
    # indistinguishable from an explicit CLI flag equal to that same value,
    # which previously let YAML silently override explicit flags whenever
    # the explicit value happened to equal the argparse default (or, worse,
    # equal ANY other key's default -- 1 == True in Python, so
    # --interface-cg-maxiter 1 collided with the interface_cg_strict
    # default).
    iface_grp = parser.add_argument_group('interface solver (B2)')
    iface_grp.add_argument(
        '--interface-solver', type=str, default=None,
        choices=['direct', 'cg', 'auto'],
        dest='interface_solver',
        help=(
            'Interface solve method.  '
            "'direct' = CHOLMD/SuperLU factorization (existing behaviour). "
            "'cg' = iterative CG (no factor; saves ~100-300 GB at 1M interface nodes). "
            "'auto' (default) = direct when n_interface < 200K and factor fits "
            "memory budget; else CG.  'auto' is backwards-compatible (existing "
            "netlists with small interface systems get 'direct')."
        ),
    )
    iface_grp.add_argument(
        '--interface-matvec-mode', type=str, default=None,
        choices=['auto', 'assembled', 'tilewise'],
        dest='interface_matvec_mode',
        help=(
            'CG matvec mode (only used when --interface-solver=cg or auto selects CG). '
            "'auto' = tilewise when per-tile dense Schur blocks are available, "
            "else assembled. "
            "'assembled' = matvec on assembled sparse S_global. "
            "'tilewise' = sum_i P_i^T S_i P_i x using per-tile dense Schur blocks "
            "(avoids global assembly entirely; coordinator O(n*k) memory). "
            "(default: auto)"
        ),
    )
    iface_grp.add_argument(
        '--interface-preconditioner', type=str, default=None,
        choices=['auto', 'block_jacobi', 'jacobi', 'none', 'amg', 'two_level'],
        dest='interface_preconditioner',
        help=(
            'CG preconditioner (only used when CG is selected).  Default '
            "(omit this flag) resolves to 'two_level' when CG + tilewise "
            "matvec is selected (Stage 3 -- fixes block_jacobi CG "
            "stagnation at large split regimes), else 'block_jacobi'. "
            "'block_jacobi' = block-diagonal from per-tile Schur submatrices. "
            "'jacobi' = diagonal of S_global.  'none' = identity.  "
            "'amg' = algebraic multigrid via pyamg (requires pyamg).  "
            "'two_level' = block_jacobi PLUS an additive partition-of-unity "
            "+ GenEO-lite coarse-space correction -- see "
            "--interface-coarse-geneo-k/-tol/--interface-coarse-eps-rank/"
            "--interface-coarse-max-cols."
        ),
    )
    iface_grp.add_argument(
        '--interface-cg-rtol', type=float, default=None,
        dest='interface_cg_rtol',
        help=(
            'CG relative convergence tolerance (default: 1e-8; validated by '
            'the Stage 0 rtol sweep -- 166 nV max error vs direct on the '
            'BRCM-class proxy, see docs Sec 7.7).'
        ),
    )
    iface_grp.add_argument(
        '--interface-cg-atol', type=float, default=None,
        dest='interface_cg_atol',
        help=(
            'CG absolute convergence tolerance floor (default: 1e-14). '
            'Prevents CG from burning maxiter iterations when the RHS is '
            'near-zero (e.g. early transient steps with no active sources).'
        ),
    )
    iface_grp.add_argument(
        '--interface-cg-maxiter', type=int, default=None,
        dest='interface_cg_maxiter',
        help='Max CG iterations (default: None -> 3 * n_interface).',
    )
    iface_grp.add_argument(
        '--interface-cg-strict', dest='interface_cg_strict',
        action='store_true', default=None,
        help='Raise RuntimeError on CG non-convergence (default: enabled).',
    )
    iface_grp.add_argument(
        '--interface-cg-no-strict', dest='interface_cg_strict',
        action='store_false',
        help=(
            'Demote CG non-convergence to a warning instead of raising '
            '(not recommended for production).'
        ),
    )
    iface_grp.add_argument(
        '--interface-factor-memory-budget', type=str, default=None,
        dest='interface_factor_memory_budget',
        help=(
            "Coordinator direct-factor memory budget used by "
            "--interface-solver=auto ('auto' = min(32 GB, 0.4x total RAM) "
            "via psutil, or an explicit integer byte count). "
            "(default: auto)"
        ),
    )
    iface_grp.add_argument(
        '--interface-block-jacobi-max-bytes', type=str, default=None,
        dest='interface_block_jacobi_max_bytes',
        help=(
            "Block-Jacobi preconditioner factor-memory budget "
            "('auto' = min(8 GB, 0.1x total RAM) via psutil, or an "
            "explicit integer byte count).  Exceeding it downgrades to the "
            "'jacobi' diagonal preconditioner (logged as a loud WARNING "
            "naming this setting and the iteration-count consequence). "
            "(default: auto)"
        ),
    )
    # Stage 3: two-level coarse-space preconditioner knobs (only used when
    # the resolved preconditioner is 'two_level').
    iface_grp.add_argument(
        '--interface-coarse-geneo-k', type=int, default=None,
        dest='interface_coarse_geneo_k',
        help=(
            "Max GenEO-lite eigenpairs enriched per block-Jacobi ownership "
            "block (default: 4; 0 disables GenEO, leaving a partition-of-"
            "unity-only coarse space)."
        ),
    )
    iface_grp.add_argument(
        '--interface-coarse-geneo-tol', type=float, default=None,
        dest='interface_coarse_geneo_tol',
        help=(
            "Relative eigenvalue threshold (fraction of a block's own "
            "lambda_max) below which an eigenpair is GenEO-enriched "
            "(default: 1e-6)."
        ),
    )
    iface_grp.add_argument(
        '--interface-coarse-eps-rank', type=float, default=None,
        dest='interface_coarse_eps_rank',
        help=(
            "S_c eigenvalues <= this fraction of S_c's own lambda_max are "
            "treated as structural rank deficiency (e.g. the checkerboard "
            "null space of an even-multiplicity partition-of-unity basis) "
            "and dropped from the coarse pseudo-inverse (default: 1e-12; "
            "distinct knob from --interface-coarse-geneo-tol)."
        ),
    )
    iface_grp.add_argument(
        '--interface-coarse-max-cols', type=int, default=None,
        dest='interface_coarse_max_cols',
        help=(
            "Hard cap on the coarse-space column count T' (default: 4096); "
            "exceeding it first falls back to a PoU-only coarse space "
            "(WARNING, GenEO columns dropped) -- the coarse space is "
            "disabled entirely (degrades to plain block_jacobi) only if "
            "the PoU-only column count ALONE still exceeds the cap."
        ),
    )
    iface_grp.add_argument(
        '--interface-coarse-max-bytes', type=str, default=None,
        dest='interface_coarse_max_bytes',
        help=(
            "Byte-based guard on the two dense (n x T') fp64 arrays the "
            "coarse build allocates ('auto' = min(8 GB, 0.1x total RAM) "
            "via psutil, or an explicit integer byte count).  Distinct "
            "from --interface-coarse-max-cols (a column-count cap that "
            "does not scale with n); same two-rung degradation (PoU-only, "
            "then disable) when exceeded. (default: auto)"
        ),
    )
    # Stage 2: threaded tilewise matvec / block-Jacobi apply
    iface_grp.add_argument(
        '--matvec-threads', type=str, default=None,
        dest='matvec_threads',
        help=(
            "Thread count for the tilewise CG matvec and the block-Jacobi "
            "apply (persistent ThreadPoolExecutor, lazily built). "
            "'auto' (default) = min(8, cpu_count, n_tiles) -- Stage 0 "
            "measured best throughput at 8 threads on the BRCM-class proxy "
            "(inverted scaling above 8), or an explicit positive integer."
        ),
    )
    iface_grp.add_argument(
        '--interface-matvec-dtype', type=str, default=None,
        choices=['float64', 'float32'],
        dest='interface_matvec_dtype',
        help=(
            "Tilewise per-tile Schur block storage dtype (default: "
            "float64). 'float32' roughly doubles GEMV throughput on a "
            "CPU-only host at the cost of a ~1e-7 relative residual floor "
            "-- must be paired with --interface-cg-rtol >= 1e-7 (enforced "
            "unless --interface-no-strict-dtype-rtol is passed)."
        ),
    )
    iface_grp.add_argument(
        '--interface-strict-dtype-rtol', dest='interface_strict_dtype_rtol',
        action='store_true', default=None,
        help=(
            "Enforce the matvec_dtype='float32'/rtol>=1e-7 pairing "
            "(default: enabled)."
        ),
    )
    iface_grp.add_argument(
        '--interface-no-strict-dtype-rtol', dest='interface_strict_dtype_rtol',
        action='store_false',
        help=(
            "Allow matvec_dtype='float32' with rtol < 1e-7 (not "
            "recommended; for accuracy studies only)."
        ),
    )
    iface_grp.add_argument(
        '--interface-drop-s-global', dest='interface_drop_s_global',
        action='store_true', default=None,
        help=(
            "Never assemble S_global at all (item 3, Stage 0 Finding 0) -- "
            "requires --interface-solver=cg (explicit, not auto), "
            "--interface-matvec-mode in {tilewise, auto}, and "
            "--island-detection resolving to the 'summaries' union-find "
            "(the legacy Schur-BFS needs S_global's nonzero structure). "
            "Falls back to the normal assembling path with a WARNING when "
            "preconditions are not met. save() raises with guidance for a "
            "context factored this way (default: disabled)."
        ),
    )
    iface_grp.add_argument(
        '--interface-no-drop-s-global', dest='interface_drop_s_global',
        action='store_false',
        help=(
            "Finding 10: explicit negation of --interface-drop-s-global, "
            "matching the paired true/false flags on the other iface-group "
            "booleans (--interface-cg-strict/-no-strict, "
            "--interface-strict-dtype-rtol/-no-strict-dtype-rtol). Without "
            "this, a shared solver.yaml with interface_drop_s_global: true "
            "could never be overridden to False from the CLI -- there was "
            "no flag that produced an explicit False for this key, so "
            "explicit-CLI > YAML precedence was unexpressable in the False "
            "direction. Forces the normal S_global-assembling factor path "
            "(needed to save() a checkpoint, which never-assemble contexts "
            "cannot do)."
        ),
    )

    # Stage 1e: island detection (parse-time summaries vs legacy Schur-BFS)
    iface_grp.add_argument(
        '--island-detection', type=str, default=None,
        choices=['auto', 'summaries', 'schur_bfs'],
        dest='island_detection',
        help=(
            "Interface island-detection strategy, resolved ONCE at model "
            "creation (all-new-or-all-legacy).  'auto'/'summaries' (default: "
            "auto) use the bundle's parse-time connectivity summaries via a "
            "cheap union-find and skip worker-side island removal, falling "
            "back to 'schur_bfs' automatically when the bundle lacks "
            "summaries, the summary version is stale, or the model-creation "
            "trust assertion fails.  'schur_bfs' forces the legacy "
            "O(S.nnz) Schur-complement BFS path unconditionally. "
            "(default: auto)"
        ),
    )

    # B3: streaming Schur assembly
    asm_grp = parser.add_argument_group('streaming assembly (B3)')
    asm_grp.add_argument(
        '--streaming-assembly', type=str, default=None,
        choices=['false', 'true', 'auto'],
        dest='streaming_assembly',
        help=(
            'Streaming Schur assembly mode.  '
            "'false' (default): assemble S_global fully in memory before factoring. "
            "'true': workers cache S_i shards and stream them one tile at a time; "
            "peak memory is O(one tile's shard) instead of O(sum of all S_i). "
            "'auto': switch to streaming when estimated S_i peak exceeds 512 MB. "
            'Incompatible with interface_solver=cg when matvec_mode=assembled. '
            '(default: None → use model.settings default, which is False)'
        ),
    )

    # Reporting / profiling
    parser.add_argument('--top-k', type=int, default=100,
                        help='Number of worst nodes to report (default: 100)')
    parser.add_argument('--profile-memory', action='store_true',
                        help='Enable memory profiling (slower)')


def _add_time_domain_args(parser: argparse.ArgumentParser) -> None:
    """Add --mode and time-domain parameters to a subparser.

    Shared between the ``solve`` and ``run`` subcommands.
    """
    td = parser.add_argument_group('time-domain analysis')
    td.add_argument('--mode', type=str, default='dc',
                    choices=['dc', 'quasi-static', 'transient'],
                    help='Analysis mode (default: dc)')
    td.add_argument('--t-start', type=float, default=0.0,
                    help='Start time in seconds (default: 0.0)')
    td.add_argument('--t-end', type=float, default=100e-9,
                    help='End time in seconds (default: 100e-9)')
    td.add_argument('--dt', type=float, default=0.1e-9,
                    help='Time step for transient in seconds (default: 0.1e-9)')
    td.add_argument('--n-points', type=int, default=101,
                    help='Number of time points for quasi-static (default: 101)')
    td.add_argument('--method', type=str, default='be',
                    choices=['be', 'trap'],
                    help='Integration method for transient (default: be)')
    td.add_argument('--smooth', action='store_true', default=True,
                    help='Enable PWL smoothing (default: enabled)')
    td.add_argument('--no-smooth', dest='smooth', action='store_false',
                    help='Disable PWL smoothing')


def _load_decompose_config(config_path: str) -> dict:
    """Load a YAML config file and return the raw dict.

    Parameters
    ----------
    config_path : str
        Path to a ``.yaml`` / ``.yml`` config file.

    Returns
    -------
    dict
        Parsed config dict (may be empty if file is empty).

    Raises
    ------
    SystemExit
        If the file is missing or unparseable.
    """
    try:
        import yaml
    except ImportError:
        logger.error(
            "PyYAML is required for config file support. "
            "Install with: pip install pyyaml"
        )
        raise SystemExit(1)

    p = Path(config_path)
    if not p.exists():
        logger.error("Config file not found: %s", config_path)
        raise SystemExit(1)
    try:
        with open(p, 'r') as f:
            data = yaml.safe_load(f)
        return data if data is not None else {}
    except Exception as exc:
        logger.error("Failed to load config %s: %s", config_path, exc)
        raise SystemExit(1)


def _merge_decompose_config(
    config: dict,
    args: argparse.Namespace,
) -> argparse.Namespace:
    """Merge YAML config with CLI args for the ``decompose`` subcommand.

    CLI arguments take precedence over config values.  The precedence rule is:
    "if the CLI value differs from the argparse default, the user explicitly
    provided it, so it wins."

    Parameters
    ----------
    config : dict
        Raw dict loaded from the YAML config file.
    args : argparse.Namespace
        Parsed CLI arguments (mutated in-place and returned).

    Returns
    -------
    argparse.Namespace
        The updated arguments namespace.
    """
    from analysis.dynamic_irdrop_decomposition import parse_time_value

    # Validate top-level config sections
    unknown_top = set(config.keys()) - _VALID_DECOMPOSE_TOP_KEYS
    if unknown_top:
        raise ValueError(
            f"Unknown top-level key(s) in decompose config: "
            f"{', '.join(sorted(unknown_top))}. "
            f"Valid keys: {', '.join(sorted(_VALID_DECOMPOSE_TOP_KEYS))}"
        )

    # -- netlist_dir / net / backend -----------------------------------------
    # netlist_dir is the primary positional arg; config can override it.
    if config.get('netlist_dir') and not getattr(args, 'netlist_dir', None):
        args.netlist_dir = str(config['netlist_dir'])
    if config.get('net') and not getattr(args, 'net', None):
        args.net = str(config['net'])

    if config.get('backend') and args.backend == 'local':
        args.backend = config['backend']

    # -- time section ----------------------------------------------------
    time_cfg = config.get('time', {})

    # Argparse defaults for the decompose subcommand.
    # Keep in sync with build_parser() decompose section + set_defaults().
    # See test_defaults_dict_matches_argparse for automated validation.
    _DEFAULTS = {
        't_start': 0.0,
        't_end': 100e-9,
        'dt': 0.1e-9,
        'method': 'be',
        'top_k': 5,
        'window_percent': 10.0,
        'aggressor_top_k': 0,
        'adjoint_method': 'dynamic',
        'adjoint_memory_window': 20,
        'qs_candidate_factor': 3000,
        'max_qs_candidates': 10000,
        'output': './irdrop_decomp_results',
        'no_plot': False,
        'plot_layers': None,
        'max_stripes': 500,
        'verbose': False,
        'instances': None,
    }

    def _cli_is_default(attr: str) -> bool:
        """Return True if the CLI value is the argparse default."""
        return getattr(args, attr, None) == _DEFAULTS.get(attr)

    if time_cfg.get('start') is not None and _cli_is_default('t_start'):
        args.t_start = parse_time_value(time_cfg['start'])
    if time_cfg.get('end') is not None and _cli_is_default('t_end'):
        args.t_end = parse_time_value(time_cfg['end'])
    if time_cfg.get('dt') is not None and _cli_is_default('dt'):
        args.dt = parse_time_value(time_cfg['dt'])

    # -- analysis section ------------------------------------------------
    analysis_cfg = config.get('analysis', {})

    if analysis_cfg.get('top_k') is not None and _cli_is_default('top_k'):
        args.top_k = int(analysis_cfg['top_k'])
    if analysis_cfg.get('window_percent') is not None and _cli_is_default('window_percent'):
        args.window_percent = float(analysis_cfg['window_percent'])
    if analysis_cfg.get('integration') is not None and _cli_is_default('method'):
        args.method = str(analysis_cfg['integration'])
    if analysis_cfg.get('qs_candidate_factor') is not None and _cli_is_default('qs_candidate_factor'):
        args.qs_candidate_factor = int(analysis_cfg['qs_candidate_factor'])
    if analysis_cfg.get('max_qs_candidates') is not None and _cli_is_default('max_qs_candidates'):
        args.max_qs_candidates = int(analysis_cfg['max_qs_candidates'])

    # Smooth: three-way merge (CLI explicit > config > default True).
    # args.smooth is None when user didn't pass --smooth or --no-smooth.
    if args.smooth is None:
        cfg_smooth = analysis_cfg.get('smooth_sources')
        if cfg_smooth is not None:
            args.smooth = bool(cfg_smooth)
        else:
            args.smooth = True  # ultimate default
    # else: CLI was explicit (True from --smooth, False from --no-smooth)

    # Instances: CLI string > config list > config instances_file > None
    if args.instances is None:
        cfg_instances = analysis_cfg.get('instances')
        cfg_instances_file = analysis_cfg.get('instances_file')
        if cfg_instances and isinstance(cfg_instances, list):
            # Store as comma-separated string to match CLI format
            args.instances = ','.join(str(i) for i in cfg_instances)
        elif cfg_instances_file:
            p = Path(cfg_instances_file)
            if p.exists():
                with open(p, 'r') as f:
                    nodes = [line.strip() for line in f if line.strip()]
                if nodes:
                    args.instances = ','.join(nodes)
            else:
                logger.warning(
                    "instances_file not found: %s", cfg_instances_file
                )

    # -- aggressor section -----------------------------------------------
    # Accept both 'aggressor:' top-level section and 'analysis.aggressor_top_k'
    # (flat config compat).  Dedicated 'aggressor:' section takes priority.
    agg_cfg = config.get('aggressor', {})

    if agg_cfg.get('top_k') is not None and _cli_is_default('aggressor_top_k'):
        args.aggressor_top_k = int(agg_cfg['top_k'])
    if agg_cfg.get('method') is not None and _cli_is_default('adjoint_method'):
        args.adjoint_method = str(agg_cfg['method'])
    if agg_cfg.get('memory_window') is not None and _cli_is_default('adjoint_memory_window'):
        args.adjoint_memory_window = int(agg_cfg['memory_window'])

    # -- output section --------------------------------------------------
    output_cfg = config.get('output', {})

    if output_cfg.get('output_dir') is not None and _cli_is_default('output'):
        args.output = str(output_cfg['output_dir'])
    if output_cfg.get('no_plot') is not None and _cli_is_default('no_plot'):
        args.no_plot = bool(output_cfg['no_plot'])
    if output_cfg.get('plot_layers') is not None and _cli_is_default('plot_layers'):
        layers = output_cfg['plot_layers']
        if isinstance(layers, list):
            args.plot_layers = ','.join(str(layer) for layer in layers)
        else:
            args.plot_layers = str(layers)
    if output_cfg.get('max_stripes') is not None and _cli_is_default('max_stripes'):
        args.max_stripes = int(output_cfg['max_stripes'])
    if output_cfg.get('verbose') is not None and _cli_is_default('verbose'):
        args.verbose = bool(output_cfg['verbose'])

    # -- solver section (cholmod settings) --------------------------------
    solver_cfg = config.get('solver', {})
    _validate_solver_yaml_keys(solver_cfg, 'solver')

    if solver_cfg.get('use_cholmod') is not None:
        if getattr(args, 'use_cholmod', None) is None:
            args.use_cholmod = bool(solver_cfg['use_cholmod'])
    if solver_cfg.get('ordering') is not None:
        if getattr(args, 'cholmod_ordering', 'default') == 'default':
            args.cholmod_ordering = str(solver_cfg['ordering'])
    if solver_cfg.get('mode') is not None:
        if getattr(args, 'cholmod_mode', 'auto') == 'auto':
            args.cholmod_mode = str(solver_cfg['mode'])

    # Per-role overrides from solver: coordinator: / worker: sub-dicts
    _apply_yaml_role_configs(solver_cfg, args)

    # threads_per_worker (YAML only; CLI flag --threads-per-worker takes precedence)
    if solver_cfg.get('threads_per_worker') is not None:
        if getattr(args, 'threads_per_worker', None) is None:
            yaml_tpw = solver_cfg['threads_per_worker']
            args.threads_per_worker = (
                'auto' if yaml_tpw == 'auto' else int(yaml_tpw)
            )

    # island_detection (YAML only here; explicit CLI flag takes precedence --
    # finding F6).  cmd_decompose nulls args.config before calling
    # _load_and_apply_config (to stop it from re-loading this same file via
    # pdn_solver's unrelated schema), which means _load_and_apply_config's
    # OWN island_detection resolution (CLI > YAML > 'auto') never sees this
    # YAML's solver.island_detection -- so it must be resolved HERE, in the
    # decompose-specific config merge, exactly like every other solver: key
    # above.  _load_and_apply_config's later resolution step only fills in
    # 'auto' when args.island_detection is still None, so it will not
    # override whatever is set here.
    if solver_cfg.get('island_detection') is not None:
        if getattr(args, 'island_detection', None) is None:
            args.island_detection = str(solver_cfg['island_detection'])

    logger.info("Merged decompose config from YAML")
    return args


# Valid keys for solver: YAML section and coordinator:/worker: sub-dicts
_VALID_SOLVER_YAML_KEYS = frozenset({
    'use_cholmod', 'mode', 'cholmod_mode', 'ordering', 'cholmod_ordering',
    'cholmod_use_long', 'use_long', 'coordinator', 'worker',
    'threads_per_worker',
    # B2: interface solver settings
    'interface_solver', 'interface_matvec_mode',
    'interface_preconditioner', 'interface_cg_rtol',
    # Stage 1: CG tolerance/budget plumbing
    'interface_cg_atol', 'interface_cg_maxiter', 'interface_cg_strict',
    'interface_factor_memory_budget', 'interface_block_jacobi_max_bytes',
    # Stage 2: threaded tilewise matvec / fp32 / never-assemble-S_global
    'matvec_threads', 'interface_matvec_dtype', 'interface_strict_dtype_rtol',
    'interface_drop_s_global',
    # Stage 3: two-level coarse-space preconditioner knobs
    'interface_coarse_geneo_k', 'interface_coarse_geneo_tol',
    'interface_coarse_eps_rank', 'interface_coarse_max_cols',
    'interface_coarse_max_bytes',
    # B3: streaming Schur assembly + A2 step-column table
    'streaming_assembly', 'use_step_columns', 'max_table_mb',
    # Stage 1e: island detection strategy
    'island_detection',
})
_VALID_ROLE_YAML_KEYS = _VALID_SOLVER_YAML_KEYS - {
    'coordinator', 'worker', 'threads_per_worker',
    # Stage 1e finding F12: island_detection is a top-level-solver-only
    # setting (resolved once at model creation, all-new-or-all-legacy across
    # the WHOLE model) -- it has no per-role (coordinator/worker) meaning, so
    # it must not silently validate inside a coordinator:/worker: sub-dict.
    'island_detection',
}
_VALID_DECOMPOSE_TOP_KEYS = frozenset({
    'netlist_dir', 'net', 'backend', 'time', 'analysis',
    'aggressor', 'output', 'solver',
})


def _validate_solver_yaml_keys(
    solver_cfg: dict, section_name: str = 'solver',
) -> None:
    """Raise ValueError if solver YAML section contains unknown keys."""
    if not solver_cfg:
        return
    unknown = set(solver_cfg.keys()) - _VALID_SOLVER_YAML_KEYS
    if unknown:
        raise ValueError(
            f"Unknown key(s) in {section_name}: {', '.join(sorted(unknown))}. "
            f"Valid keys: {', '.join(sorted(_VALID_SOLVER_YAML_KEYS))}"
        )
    for role in ('coordinator', 'worker'):
        sub = solver_cfg.get(role)
        if sub and isinstance(sub, dict):
            role_unknown = set(sub.keys()) - _VALID_ROLE_YAML_KEYS
            if role_unknown:
                raise ValueError(
                    f"Unknown key(s) in {section_name}.{role}: "
                    f"{', '.join(sorted(role_unknown))}. "
                    f"Valid keys: {', '.join(sorted(_VALID_ROLE_YAML_KEYS))}"
                )


def _build_config_from_yaml_section(
    section: dict,
    parent: dict,
) -> Optional[SolverBackendConfig]:
    """Build a SolverBackendConfig from a YAML role sub-dict.

    Values in *section* override values from *parent* (the top-level
    ``solver:`` dict).  Returns ``None`` if the section is empty or
    absent.
    """
    if not section:
        return None
    from pgmath.factor import SolverBackendConfig

    # Merge: parent scalar settings provide defaults, section overrides.
    # Filter parent to exclude sub-dicts (coordinator/worker).
    parent_scalars = {k: v for k, v in parent.items() if not isinstance(v, dict)}
    merged = {**parent_scalars, **section}

    # Normalize YAML key names (ordering/mode) -> config field names.
    # Use explicit None checks to avoid treating False as falsy.
    use_cholmod = merged.get('use_cholmod')

    mode = merged.get('mode')
    if mode is None:
        mode = merged.get('cholmod_mode')
    mode = mode or 'auto'

    ordering = merged.get('ordering')
    if ordering is None:
        ordering = merged.get('cholmod_ordering')
    ordering = ordering or 'default'

    use_long = merged.get('cholmod_use_long')
    if use_long is None:
        use_long = merged.get('use_long')

    return SolverBackendConfig(
        use_cholmod=bool(use_cholmod) if use_cholmod is not None else None,
        cholmod_mode=mode,
        cholmod_ordering=ordering,
        cholmod_use_long=bool(use_long) if use_long is not None else None,
    )


def _apply_yaml_role_configs(
    solver_cfg: dict,
    args: argparse.Namespace,
) -> None:
    """Extract coordinator/worker sub-dicts from solver YAML section.

    Only sets ``args.coordinator_solver_config`` / ``args.worker_solver_config``
    if the corresponding sub-dict is present AND no CLI per-role flag already
    produced a config.
    """
    for role in ('coordinator', 'worker'):
        attr = f'{role}_solver_config'
        # CLI flags take precedence
        if getattr(args, attr, None) is not None:
            continue
        sub = solver_cfg.get(role)
        if sub and isinstance(sub, dict):
            cfg = _build_config_from_yaml_section(sub, solver_cfg)
            if cfg is not None:
                setattr(args, attr, cfg)


def _load_and_apply_config(args: argparse.Namespace) -> argparse.Namespace:
    """Load config file (if specified) and apply cholmod backend settings.

    Reuses ``load_config`` / ``merge_config_with_args`` from
    ``solver.pdn_solver`` and the global cholmod setters from
    ``pgmath.factor``, mirroring ``PDNSolver._configure_solver_backend``.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments (may be mutated).

    Returns
    -------
    argparse.Namespace
        The (potentially merged) arguments.
    """
    # -- config file --------------------------------------------------------
    _raw_config: Optional[dict] = None
    if getattr(args, 'config', None):
        from solver.pdn_solver import load_config, merge_config_with_args

        try:
            _raw_config = load_config(args.config)
        except FileNotFoundError:
            logger.error("Config file not found: %s", args.config)
            raise SystemExit(1)
        except (ValueError, Exception) as exc:
            logger.error("Failed to load config %s: %s", args.config, exc)
            raise SystemExit(1)
        args = merge_config_with_args(_raw_config, args)
        # Validate solver: sub-section if present
        if 'solver' in _raw_config and isinstance(_raw_config['solver'], dict):
            _validate_solver_yaml_keys(_raw_config['solver'], 'solver')
        logger.info("Loaded config from: %s", args.config)

    # -- cholmod backend ----------------------------------------------------
    from pgmath.factor import (
        set_use_cholmod,
        set_cholmod_mode,
        set_cholmod_ordering,
        set_cholmod_use_long,
        get_active_backend,
    )

    # Resolve --use-cholmod / --use-splu into a single value
    use_cholmod = None
    if getattr(args, 'use_cholmod', None) is True:
        use_cholmod = True
    elif getattr(args, 'use_splu', False) is True:
        use_cholmod = False
    elif getattr(args, 'use_cholmod', None) is False:
        use_cholmod = False

    if use_cholmod is not None:
        try:
            set_use_cholmod(use_cholmod)
        except ImportError:
            if use_cholmod:
                raise  # user explicitly asked for cholmod but it is missing

    cholmod_mode = getattr(args, 'cholmod_mode', 'auto')
    if cholmod_mode != 'auto':
        set_cholmod_mode(cholmod_mode)

    cholmod_ordering = getattr(args, 'cholmod_ordering', 'default')
    if cholmod_ordering != 'default':
        set_cholmod_ordering(cholmod_ordering)

    cholmod_use_long = getattr(args, 'cholmod_use_long', None)
    if cholmod_use_long is not None:
        set_cholmod_use_long(cholmod_use_long)

    backend_name = get_active_backend()
    logger.info("Solver backend: %s", backend_name)

    # -- per-role overrides (YAML first, then CLI flags take precedence) ----
    # Only initialize if not already set (e.g., by _merge_decompose_config)
    if not hasattr(args, 'coordinator_solver_config'):
        args.coordinator_solver_config = None
    if not hasattr(args, 'worker_solver_config'):
        args.worker_solver_config = None
    if _raw_config is not None:
        _apply_yaml_role_configs(_raw_config.get('solver', {}), args)

    # CLI --coordinator-*/--worker-* flags override YAML
    cli_coord = _collect_role_config(args, 'coordinator')
    cli_worker = _collect_role_config(args, 'worker')
    if cli_coord is not None:
        args.coordinator_solver_config = cli_coord
    if cli_worker is not None:
        args.worker_solver_config = cli_worker

    if args.coordinator_solver_config is not None:
        logger.info("Coordinator backend override: %s",
                     args.coordinator_solver_config)
    if args.worker_solver_config is not None:
        logger.info("Worker backend override: %s",
                     args.worker_solver_config)

    # -- threads_per_worker (YAML first; CLI flag takes precedence) ----------
    if not hasattr(args, 'threads_per_worker'):
        args.threads_per_worker = None
    if _raw_config is not None:
        solver_cfg_tpw = _raw_config.get('solver', {})
        yaml_tpw = solver_cfg_tpw.get('threads_per_worker')
        if yaml_tpw is not None and args.threads_per_worker is None:
            if yaml_tpw == 'auto':
                args.threads_per_worker = 'auto'
            else:
                args.threads_per_worker = int(yaml_tpw)

    if args.threads_per_worker is not None:
        logger.info("threads_per_worker: %s", args.threads_per_worker)

    # -- B2/Stage 1: interface_solver settings -------------------------------
    # Precedence: explicit CLI flag > YAML > built-in default (finding 2).
    #
    # All nine argparse flags default to None (see _add_config_and_solver_args),
    # so "getattr(args, k) is not None" means exactly "the user passed this
    # flag on the command line" -- there is no longer any ambiguity between an
    # explicit CLI value and some OTHER key's default (the old shared-tuple
    # sentinel check let e.g. --interface-cg-maxiter 1 collide with the
    # interface_cg_strict default of True, since 1 == True in Python), nor
    # between an explicit CLI value that happens to equal its OWN default
    # (e.g. --interface-cg-strict, whose value True is also the default,
    # could never beat a YAML interface_cg_strict: false).
    _iface_yaml_keys = (
        'interface_solver', 'interface_matvec_mode',
        'interface_preconditioner', 'interface_cg_rtol',
        'interface_cg_atol', 'interface_cg_maxiter', 'interface_cg_strict',
        'interface_factor_memory_budget', 'interface_block_jacobi_max_bytes',
        # Stage 2
        'matvec_threads', 'interface_matvec_dtype',
        'interface_strict_dtype_rtol', 'interface_drop_s_global',
        # Stage 3
        'interface_coarse_geneo_k', 'interface_coarse_geneo_tol',
        'interface_coarse_eps_rank', 'interface_coarse_max_cols',
        'interface_coarse_max_bytes',
    )
    solver_cfg_iface = (
        _raw_config.get('solver', {}) if _raw_config is not None else {}
    )
    for _k in _iface_yaml_keys:
        _cli_val = getattr(args, _k, None)
        if _cli_val is not None:
            # Explicit CLI flag always wins, regardless of YAML.
            continue
        _yaml_val = solver_cfg_iface.get(_k)
        setattr(
            args, _k,
            _yaml_val if _yaml_val is not None else _iface_default(_k),
        )

    # -- B3: streaming_assembly, use_step_columns, max_table_mb (YAML first) --
    # These map directly into model.settings; store on args so cmd_solve /
    # cmd_run can push them with the same pattern as interface_solver.
    if _raw_config is not None:
        solver_cfg_b3 = _raw_config.get('solver', {})
        # streaming_assembly: bool or 'auto' in YAML; store as-is (resolved later)
        _yaml_sa = solver_cfg_b3.get('streaming_assembly')
        if _yaml_sa is not None and getattr(args, 'streaming_assembly', None) is None:
            setattr(args, 'streaming_assembly', _yaml_sa)
        # use_step_columns: bool
        _yaml_usc = solver_cfg_b3.get('use_step_columns')
        if _yaml_usc is not None and not hasattr(args, 'use_step_columns'):
            setattr(args, 'use_step_columns', bool(_yaml_usc))
        # max_table_mb: float
        _yaml_mtm = solver_cfg_b3.get('max_table_mb')
        if _yaml_mtm is not None and not hasattr(args, 'max_table_mb'):
            setattr(args, 'max_table_mb', float(_yaml_mtm))

    # -- Stage 1e: island_detection (explicit CLI > YAML > 'auto' default) --
    # Resolved to a concrete value here (same precedence pattern as the
    # interface_solver loop above) so cmd_solve/cmd_run/cmd_decompose can
    # read args.island_detection unconditionally and pass it straight into
    # create_distributed_model(island_detection=...).
    if getattr(args, 'island_detection', None) is None:
        _yaml_id = solver_cfg_iface.get('island_detection')
        args.island_detection = _yaml_id if _yaml_id is not None else 'auto'

    return args


def _collect_role_config(
    args: argparse.Namespace,
    role: str,
) -> Optional[SolverBackendConfig]:
    """Build a SolverBackendConfig from ``--<role>-*`` CLI flags.

    Returns ``None`` if no role-specific flags were set.
    """
    from pgmath.factor import SolverBackendConfig

    prefix = role.replace('-', '_')
    use_cholmod_flag = getattr(args, f'{prefix}_use_cholmod', None)
    use_splu_flag = getattr(args, f'{prefix}_use_splu', False)
    mode = getattr(args, f'{prefix}_cholmod_mode', None)
    ordering = getattr(args, f'{prefix}_cholmod_ordering', None)
    use_long = getattr(args, f'{prefix}_cholmod_use_long', None)

    # Check if any role-specific flag was explicitly provided
    has_override = (
        use_cholmod_flag is not None
        or use_splu_flag
        or mode is not None
        or ordering is not None
        or use_long is not None
    )
    if not has_override:
        return None

    # Resolve use_cholmod from the two boolean flags
    use_cholmod = None
    if use_cholmod_flag is True:
        use_cholmod = True
    elif use_splu_flag:
        use_cholmod = False

    return SolverBackendConfig(
        use_cholmod=use_cholmod,
        cholmod_mode=mode or 'auto',
        cholmod_ordering=ordering or 'default',
        cholmod_use_long=use_long,
    )


def _parse_threads_per_worker(value: Optional[str]) -> Optional[Any]:
    """Convert a CLI/YAML threads_per_worker value to int or 'auto'.

    Args:
        value: ``None``, ``'auto'``, or a numeric string (e.g. ``'4'``).

    Returns:
        ``None`` when not set, ``'auto'`` for 'auto', or ``int`` otherwise.

    Raises:
        ValueError: If value is not None, 'auto', or a valid integer string.
    """
    if value is None:
        return None
    if isinstance(value, int):
        return value
    s = str(value).strip()
    if s == 'auto':
        return 'auto'
    try:
        return int(s)
    except ValueError:
        raise ValueError(
            f"Invalid threads_per_worker {value!r}. "
            "Must be an integer or 'auto'."
        )


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser with subcommands."""
    top = argparse.ArgumentParser(
        prog='python -m distributed',
        description='Distributed PDN parser/solver pipeline',
    )
    sub = top.add_subparsers(dest='command', required=True)

    # ── parse ──────────────────────────────────────────────────────
    p_parse = sub.add_parser('parse', help='Parse netlist, dump per-tile .pkl files')
    p_parse.add_argument('netlist_dir', help='Path to netlist directory')
    p_parse.add_argument('--net', '-n', default=None, help='Net name to filter (e.g., VDD_XLV)')
    p_parse.add_argument('--backend', '-b', default='local', choices=['local', 'ray'],
                         help='Compute backend (default: local)')
    p_parse.add_argument('--output', '-o', default=None, help='Output directory for .pkl files')
    p_parse.add_argument(
        '--max-interior', type=int, default=None, dest='max_interior',
        help=(
            'B1 balanced retiling: split tiles with more than this many interior '
            'nodes via recursive geometric bisection.  None = disabled (default).'
        ),
    )
    p_parse.add_argument('--verbose', '-v', action='store_true')
    p_parse.set_defaults(func=cmd_parse)

    # ── solve ──────────────────────────────────────────────────────
    p_solve = sub.add_parser('solve', help='Load .pkl partitions and run DDM solver')
    p_solve.add_argument('pkl_dir', help='Directory containing tile_X_Y.pkl and metadata.pkl')
    p_solve.add_argument('--backend', '-b', default='local', choices=['local', 'ray'],
                         help='Compute backend (default: local)')
    p_solve.add_argument('--output', '-o', default=None, help='Output directory for results')
    p_solve.add_argument('--verbose', '-v', action='store_true')
    p_solve.add_argument('--plot', action='store_true', help='Generate heatmaps after solve')
    p_solve.add_argument('--plot-layers', type=str, default=None,
                         help='Layers to plot (comma-separated, e.g. M1,M2)')
    p_solve.add_argument('--max-stripes', type=int, default=2000,
                         help='Maximum number of stripes per heatmap')
    p_solve.add_argument('--stripe-bin-size', type=int, default=None,
                         help='Bin size for within-stripe aggregation (auto if not set)')
    p_solve.add_argument('--show-voltage', dest='show_irdrop', action='store_false',
                         default=True, help='Show voltage instead of IR-drop')
    _add_config_and_solver_args(p_solve)
    _add_time_domain_args(p_solve)
    p_solve.set_defaults(func=cmd_solve)

    # ── run ────────────────────────────────────────────────────────
    p_run = sub.add_parser('run', help='Parse + dump + solve in one shot')
    p_run.add_argument('netlist_dir', help='Path to netlist directory')
    p_run.add_argument('--net', '-n', default=None, help='Net name to filter (e.g., VDD_XLV)')
    p_run.add_argument('--backend', '-b', default='local', choices=['local', 'ray'],
                       help='Compute backend (default: local)')
    p_run.add_argument('--pkl-dir', default=None,
                       help='Directory for intermediate .pkl files (default: <netlist_dir>/distributed_pkl)')
    p_run.add_argument(
        '--max-interior', type=int, default=None, dest='max_interior',
        help=(
            'B1 balanced retiling: split tiles with more than this many interior '
            'nodes via recursive geometric bisection.  None = disabled (default).'
        ),
    )
    p_run.add_argument('--output', '-o', default=None, help='Output directory for results')
    p_run.add_argument('--verbose', '-v', action='store_true')
    p_run.add_argument('--plot', action='store_true', help='Generate heatmaps after solve')
    p_run.add_argument('--plot-layers', type=str, default=None,
                       help='Layers to plot (comma-separated, e.g. M1,M2)')
    p_run.add_argument('--max-stripes', type=int, default=2000,
                       help='Maximum number of stripes per heatmap')
    p_run.add_argument('--stripe-bin-size', type=int, default=None,
                       help='Bin size for within-stripe aggregation (auto if not set)')
    p_run.add_argument('--show-voltage', dest='show_irdrop', action='store_false',
                       default=True, help='Show voltage instead of IR-drop')
    _add_config_and_solver_args(p_run)
    _add_time_domain_args(p_run)
    p_run.set_defaults(func=cmd_run)

    # ── decompose ─────────────────────────────────────────────────
    p_decompose = sub.add_parser(
        'decompose',
        help='Near/far IR-drop decomposition analysis',
    )
    p_decompose.add_argument(
        'netlist_dir',
        help='Path to netlist directory (auto-parses if '
             '<netlist_dir>/distributed_pkl does not exist)',
    )
    p_decompose.add_argument(
        '--net', '-n', type=str, default=None,
        help='Net name to filter (e.g., VDD_XLV)',
    )
    p_decompose.add_argument(
        '--backend', '-b', default='local', choices=['local', 'ray'],
        help='Compute backend (default: local)',
    )
    p_decompose.add_argument(
        '--output', '-o', default='./irdrop_decomp_results',
        help='Output directory for results (default: ./irdrop_decomp_results)',
    )
    p_decompose.add_argument('--verbose', '-v', action='store_true')
    p_decompose.add_argument(
        '--no-plot', action='store_true',
        help='Skip plot generation',
    )

    # Shared solver/config args (adds --top-k with default=100)
    _add_config_and_solver_args(p_decompose)

    # Time-domain args relevant to decompose (no --mode or --n-points).
    td = p_decompose.add_argument_group('time-domain analysis')
    td.add_argument('--t-start', type=float, default=0.0,
                     help='Start time in seconds (default: 0.0)')
    td.add_argument('--t-end', type=float, default=100e-9,
                     help='End time in seconds (default: 100e-9)')
    td.add_argument('--dt', type=float, default=0.1e-9,
                     help='Time step for transient in seconds (default: 0.1e-9)')
    td.add_argument('--method', type=str, default='be',
                     choices=['be', 'trap'],
                     help='Integration method for transient (default: be)')
    td.add_argument('--smooth', action='store_true', default=None,
                     help='Enable PWL smoothing (default: enabled)')
    td.add_argument('--no-smooth', dest='smooth', action='store_false',
                     help='Disable PWL smoothing')

    # Decomposition parameters
    decomp = p_decompose.add_argument_group('decomposition')
    decomp.add_argument(
        '--window-percent', type=float, default=10.0,
        help='Near-window size as %% of design extent (default: 10.0)',
    )
    decomp.add_argument(
        '--instances', type=str, default=None,
        help='Comma-separated victim node names (skips initial transient)',
    )
    decomp.add_argument(
        '--qs-candidate-factor', type=int, default=3000,
        dest='qs_candidate_factor',
        help=(
            'Multiplier on top_k for QS pre-selection candidates (default: 3000). '
            'Increase when capacitive-history effects push transient victims '
            'below the QS ranking cutoff; a safety-net transient will warn '
            'and fire automatically, but raising this factor avoids it.'
        ),
    )
    decomp.add_argument(
        '--max-qs-candidates', type=int, default=10000,
        dest='max_qs_candidates',
        help=(
            'Absolute cap on QS pre-selection candidates tracked per Phase 2b '
            'transient (default: 10000). Bounds per-step Python-loop overhead '
            'and worker-side waveform memory (each tracked node stores one float '
            'per time step). Victims outside the cap are caught by the safety-net '
            'targeted transient. Increase for netlists where top-K transient '
            'victims rank beyond this position in the quasi-static ranking.'
        ),
    )

    # Aggressor parameters
    agg_group = p_decompose.add_argument_group('aggressor analysis')
    agg_group.add_argument(
        '--aggressor-top-k', type=int, default=0,
        help='Number of top aggressors per victim (0=disabled, default: 0)',
    )
    agg_group.add_argument(
        '--adjoint-method', choices=['static', 'dynamic'], default='dynamic',
        help='Adjoint analysis method (default: dynamic)',
    )
    agg_group.add_argument(
        '--adjoint-memory-window', type=int, default=20,
        help='Backward sweep memory window in time steps (default: 20)',
    )

    # Plot parameters
    p_decompose.add_argument(
        '--plot-layers', type=str, default=None,
        help='Layers for heatmaps (comma-separated, e.g. M1,M2)',
    )
    p_decompose.add_argument(
        '--max-stripes', type=int, default=500,
        help='Maximum stripes for heatmap (default: 500)',
    )

    # Override --top-k default for decompose (5 victims, not 100 report nodes)
    p_decompose.set_defaults(func=cmd_decompose, top_k=5)

    return top


def main() -> None:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)

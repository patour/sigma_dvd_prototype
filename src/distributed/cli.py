"""CLI for distributed PDN parser/solver pipeline.

Subcommands:
    parse  - Parse netlist, dump per-tile .pkl files
    solve  - Load .pkl partitions, run DDM solver
    run    - Parse + dump + solve in one shot

Usage:
    python -m distributed parse  ./netlist/netlist_sampled --net VDD_XLV -o ./pkl_out
    python -m distributed solve  ./pkl_out -o ./results
    python -m distributed solve  ./pkl_out --mode quasi-static --t-end 100e-9 --n-points 51
    python -m distributed solve  ./pkl_out --mode transient --t-end 100e-9 --dt 0.1e-9
    python -m distributed run    ./netlist/netlist_sampled --net VDD_XLV -o ./results
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _setup_logging(verbose: bool) -> None:
    """Configure logging for CLI usage."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s %(name)s %(levelname)s: %(message)s',
        datefmt='%H:%M:%S',
    )


def cmd_parse(args: argparse.Namespace) -> None:
    """Parse netlist and dump per-tile .pkl files."""
    from .parser import DistributedNetlistParser

    _setup_logging(args.verbose)
    t0 = time.perf_counter()

    parser = DistributedNetlistParser(args.netlist_dir, net_filter=args.net)
    out_dir = args.output or str(Path(args.netlist_dir) / 'distributed_pkl')
    out_path, _bundle = parser.parse_and_dump(out_dir, backend=args.backend)

    elapsed = time.perf_counter() - t0
    logger.info(f"parse_and_dump completed in {elapsed:.3f}s -> {out_path}")


def cmd_solve(args: argparse.Namespace) -> None:
    """Load .pkl partitions and run DDM solver."""
    from .model import create_distributed_model, load_distributed_partitions
    from .solver import DistributedDDMSolver

    _setup_logging(args.verbose)
    args = _load_and_apply_config(args)
    t0 = time.perf_counter()

    mode = getattr(args, 'mode', 'dc')

    bundle = load_distributed_partitions(args.pkl_dir)

    model = create_distributed_model(
        bundle,
        backend=args.backend,
    )

    try:
        solver = DistributedDDMSolver(model)
        ctx = solver.prepare(verbose=args.verbose)

        if mode == 'dc':
            _solve_dc(solver, ctx, args, t0)
        elif mode == 'quasi-static':
            _solve_quasi_static(solver, ctx, args, t0)
        elif mode == 'transient':
            _solve_transient(solver, ctx, args, t0)
        else:
            logger.error("Unknown mode: %s", mode)
            raise SystemExit(1)
    finally:
        model.shutdown()


def _solve_dc(
    solver: 'DistributedDDMSolver',
    ctx: 'DistributedSolverContext',
    args: argparse.Namespace,
    t0: float,
) -> None:
    """Run DC solve and report results (original behavior)."""
    result = solver.solve_dc(context=ctx, verbose=args.verbose)

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
        t_start=args.t_start,
        t_end=args.t_end,
        n_points=args.n_points,
        context=ctx,
        smoothed_sources=smoothed_sources,
        verbose=args.verbose,
    )

    _report_time_domain_result(result, args, t0, mode='quasi-static')


def _solve_transient(
    solver: 'DistributedDDMSolver',
    ctx: 'DistributedSolverContext',  # unused: transient builds its own context
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

    result = solver.solve_transient(
        t_start=args.t_start,
        t_end=args.t_end,
        dt=args.dt,
        method=args.method,
        smoothed_sources=smoothed_sources,
        verbose=args.verbose,
    )

    _report_time_domain_result(result, args, t0, mode='transient')


def _report_time_domain_result(
    result: Any,
    args: argparse.Namespace,
    t0: float,
    mode: str,
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

    # Heatmaps not yet supported for time-domain modes
    if args.plot:
        logger.info(
            "Heatmap generation is not yet supported for %s mode. "
            "Skipping --plot.", mode,
        )


def cmd_run(args: argparse.Namespace) -> None:
    """Parse + dump + solve in one shot."""
    from .model import create_distributed_model
    from .parser import DistributedNetlistParser
    from .solver import DistributedDDMSolver

    _setup_logging(args.verbose)
    args = _load_and_apply_config(args)
    t_total = time.perf_counter()

    mode = getattr(args, 'mode', 'dc')

    # Parse and dump
    t0 = time.perf_counter()
    parser = DistributedNetlistParser(args.netlist_dir, net_filter=args.net)
    pkl_dir = args.pkl_dir or str(Path(args.netlist_dir) / 'distributed_pkl')
    _out_path, bundle = parser.parse_and_dump(pkl_dir, backend=args.backend)
    t_parse = time.perf_counter() - t0
    logger.info(f"Parse phase: {t_parse:.3f}s")

    # Create model from bundle
    t0 = time.perf_counter()
    model = create_distributed_model(
        bundle,
        backend=args.backend,
    )
    t_model = time.perf_counter() - t0
    logger.info(f"Model creation: {t_model:.3f}s")

    try:
        solver = DistributedDDMSolver(model)

        t0_prepare = time.perf_counter()
        ctx = solver.prepare(verbose=args.verbose)
        t_prepare = time.perf_counter() - t0_prepare
        logger.info(f"Prepare phase: {t_prepare:.3f}s")

        if mode == 'dc':
            _solve_dc(solver, ctx, args, t_total)
        elif mode == 'quasi-static':
            _solve_quasi_static(solver, ctx, args, t_total)
        elif mode == 'transient':
            _solve_transient(solver, ctx, args, t_total)
        else:
            logger.error("Unknown mode: %s", mode)
            raise SystemExit(1)
    finally:
        model.shutdown()


def _add_config_and_solver_args(parser: argparse.ArgumentParser) -> None:
    """Add --config, cholmod backend, and profiling flags to a subparser.

    Shared between the ``solve`` and ``run`` subcommands so the flag set is
    consistent with ``pdn_solver.py``'s CLI.
    """
    # Config file
    parser.add_argument('--config', '-c', type=str, default=None,
                        help='Config file path (.yaml, .yml, or .json)')

    # Solver backend (cholmod / splu)
    parser.add_argument('--use-cholmod', action='store_true', default=None,
                        help='Force cholmod backend (requires sksparse)')
    parser.add_argument('--use-splu', action='store_true',
                        help='Force splu backend (scipy)')
    parser.add_argument('--cholmod-mode', type=str, default='auto',
                        choices=['auto', 'simplicial', 'supernodal'],
                        help='Cholmod factorization mode (default: auto)')
    parser.add_argument('--cholmod-ordering', type=str, default='default',
                        choices=['default', 'natural', 'amd', 'metis',
                                 'nesdis', 'colamd', 'best'],
                        help='Cholmod fill-reducing ordering (default: default)')
    parser.add_argument('--cholmod-use-long', action='store_true', default=None,
                        help='Force 64-bit indices in cholmod')

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


def _load_and_apply_config(args: argparse.Namespace) -> argparse.Namespace:
    """Load config file (if specified) and apply cholmod backend settings.

    Reuses ``load_config`` / ``merge_config_with_args`` from
    ``solver.pdn_solver`` and the global cholmod setters from
    ``solver.unified_solver``, mirroring ``PDNSolver._configure_solver_backend``.

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
    if getattr(args, 'config', None):
        from solver.pdn_solver import load_config, merge_config_with_args

        try:
            config = load_config(args.config)
        except FileNotFoundError:
            logger.error("Config file not found: %s", args.config)
            raise SystemExit(1)
        except (ValueError, Exception) as exc:
            logger.error("Failed to load config %s: %s", args.config, exc)
            raise SystemExit(1)
        args = merge_config_with_args(config, args)
        logger.info("Loaded config from: %s", args.config)

    # -- cholmod backend ----------------------------------------------------
    from solver.unified_solver import (
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

    # Cholmod globals only affect the driver process; Ray workers inherit
    # their own defaults.  Warn if the user explicitly set cholmod options
    # with a Ray backend so they aren't surprised.
    backend_arg = getattr(args, 'backend', 'local')
    if backend_arg == 'ray' and use_cholmod is not None:
        logger.warning(
            "Cholmod settings are applied to the driver process only. "
            "Ray tile workers use their own defaults and may not inherit "
            "these settings."
        )

    return args


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
                         help='Number of bins per stripe (auto if not set)')
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
    p_run.add_argument('--output', '-o', default=None, help='Output directory for results')
    p_run.add_argument('--verbose', '-v', action='store_true')
    p_run.add_argument('--plot', action='store_true', help='Generate heatmaps after solve')
    p_run.add_argument('--plot-layers', type=str, default=None,
                       help='Layers to plot (comma-separated, e.g. M1,M2)')
    p_run.add_argument('--max-stripes', type=int, default=2000,
                       help='Maximum number of stripes per heatmap')
    p_run.add_argument('--stripe-bin-size', type=int, default=None,
                       help='Number of bins per stripe (auto if not set)')
    p_run.add_argument('--show-voltage', dest='show_irdrop', action='store_false',
                       default=True, help='Show voltage instead of IR-drop')
    _add_config_and_solver_args(p_run)
    _add_time_domain_args(p_run)
    p_run.set_defaults(func=cmd_run)

    return top


def main() -> None:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)

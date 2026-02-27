"""CLI for distributed PDN parser/solver pipeline.

Subcommands:
    parse  - Parse netlist, dump per-tile .pkl files
    solve  - Load .pkl partitions, run DDM solver
    run    - Parse + dump + solve in one shot

Usage:
    python -m distributed parse  ./netlist/netlist_sampled --net VDD_XLV -o ./pkl_out
    python -m distributed solve  ./pkl_out -o ./results
    python -m distributed run    ./netlist/netlist_sampled --net VDD_XLV -o ./results
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional

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
    parser.parse_and_dump(out_dir)

    elapsed = time.perf_counter() - t0
    logger.info(f"parse_and_dump completed in {elapsed:.3f}s -> {out_dir}")


def cmd_solve(args: argparse.Namespace) -> None:
    """Load .pkl partitions and run DDM solver."""
    from .model import create_distributed_model, load_distributed_partitions
    from .solver import DistributedDDMSolver

    _setup_logging(args.verbose)
    t0 = time.perf_counter()

    metadata, boundary_nodes, tile_data_dict = load_distributed_partitions(args.pkl_dir)

    model = create_distributed_model(
        metadata,
        backend=args.backend,
        use_pkl=True,
        boundary_nodes=boundary_nodes,
        tile_data_dict=tile_data_dict,
    )

    try:
        solver = DistributedDDMSolver(model)
        ctx = solver.prepare(verbose=args.verbose)
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
    finally:
        model.shutdown()


def cmd_run(args: argparse.Namespace) -> None:
    """Parse + dump + solve in one shot."""
    from .model import create_distributed_model, load_distributed_partitions
    from .parser import DistributedNetlistParser
    from .solver import DistributedDDMSolver

    _setup_logging(args.verbose)
    t_total = time.perf_counter()

    # Parse and dump
    t0 = time.perf_counter()
    parser = DistributedNetlistParser(args.netlist_dir, net_filter=args.net)
    pkl_dir = args.pkl_dir or str(Path(args.netlist_dir) / 'distributed_pkl')
    parser.parse_and_dump(pkl_dir)
    t_parse = time.perf_counter() - t0
    logger.info(f"Parse phase: {t_parse:.3f}s")

    # Load and solve
    t0 = time.perf_counter()
    metadata, boundary_nodes, tile_data_dict = load_distributed_partitions(pkl_dir)

    model = create_distributed_model(
        metadata,
        backend=args.backend,
        use_pkl=True,
        boundary_nodes=boundary_nodes,
        tile_data_dict=tile_data_dict,
    )
    t_model = time.perf_counter() - t0
    logger.info(f"Model creation: {t_model:.3f}s")

    try:
        solver = DistributedDDMSolver(model)

        t0 = time.perf_counter()
        ctx = solver.prepare(verbose=args.verbose)
        t_prepare = time.perf_counter() - t0
        logger.info(f"Prepare phase: {t_prepare:.3f}s")

        t0 = time.perf_counter()
        result = solver.solve_dc(context=ctx, verbose=args.verbose)
        t_solve = time.perf_counter() - t0
        logger.info(f"Solve phase: {t_solve:.3f}s")

        # Report summary
        v_all = result.flatten()
        ir_drop = result.ir_drop

        elapsed = time.perf_counter() - t_total
        if ir_drop:
            max_drop_node = max(ir_drop, key=ir_drop.get)
            max_drop_mv = ir_drop[max_drop_node] * 1e3
            logger.info(
                f"Total pipeline: {elapsed:.3f}s, "
                f"{len(v_all)} nodes, max IR-drop = {max_drop_mv:.3f} mV "
                f"at {max_drop_node}"
            )
        else:
            logger.warning(
                f"Total pipeline: {elapsed:.3f}s, "
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
    finally:
        model.shutdown()


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
    p_run.set_defaults(func=cmd_run)

    return top


def main() -> None:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)

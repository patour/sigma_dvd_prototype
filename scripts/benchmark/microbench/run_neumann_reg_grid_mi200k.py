#!/usr/bin/env python
"""NN/BDD work package: cold-DC-only Tikhonov-reg grid on mi200k_v2.

The first NN run (eigclip pseudo-inverse, reg=0) stagnated cold at
rel-res 1.6e-5 / 2000 iters: every tile Schur block at the split regime is
numerically singular, and the 1/(1e-10*lambda_max) clipped-direction
response amplifies a broad near-null cluster the 65-column PoU deflation
cannot cover.  This grid measures cold DC @1e-8 for a ladder of relative
Tikhonov shifts (bounding the amplification at ~1/reg and putting every
block on the fast Cholesky path) -- picking the reg for the full 100-step
protocol run, or killing the NN-reg idea if no cell beats the champion's
34 cold iters by enough to matter.

Usage:
    venv/bin/python -u scripts/benchmark/microbench/run_neumann_reg_grid_mi200k.py \
        netlist/netlist_brcm_sampled/distributed_pkl_mi200k_v2 \
        [--regs 1e-3,1e-4,1e-5,1e-6] [--json results_neumann_reg_grid_mi200k.json]
"""
from __future__ import annotations

import argparse
import json
import time
from typing import Dict, List

import psutil
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)s %(levelname)s: %(message)s',
                    datefmt='%H:%M:%S')


def _rss_gb() -> float:
    return psutil.Process().memory_info().rss / 1e9


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('pkl_dir')
    ap.add_argument('--regs', default='1e-3,1e-4,1e-5,1e-6')
    ap.add_argument('--tiles-per-worker', type=int, default=4)
    ap.add_argument('--rtol', type=float, default=1e-8)
    ap.add_argument('--maxiter', type=int, default=1500)
    ap.add_argument('--json', default=None)
    args = ap.parse_args()

    from distributed.model import (
        create_distributed_model, load_distributed_partitions,
    )
    from distributed.solver import DistributedDDMSolver

    regs = [float(r) for r in args.regs.split(',')]
    rows: List[Dict[str, object]] = []

    bundle = load_distributed_partitions(args.pkl_dir)
    model = create_distributed_model(
        bundle, backend='ray', tiles_per_worker=args.tiles_per_worker)

    base_settings = {
        'interface_solver': 'cg',
        'interface_matvec_mode': 'auto',
        'interface_preconditioner': 'two_level',
        'interface_two_level_base': 'neumann',
        'interface_coarse_apply_mode': 'deflated',
        'interface_coarse_geneo_k': 0,
        'interface_cg_rtol': args.rtol,
        'interface_cg_maxiter': args.maxiter,
        'interface_cg_strict': False,      # grid: record, don't abort
        'streaming_assembly': True,
        'interface_drop_s_global': True,
        'interface_warm_start_extrapolation': True,
    }

    for reg in regs:
        model.settings.update(dict(base_settings, interface_neumann_reg=reg))
        solver = DistributedDDMSolver(model)
        t0 = time.perf_counter()
        ctx = solver.prepare(verbose=False)
        prep = time.perf_counter() - t0
        cg = ctx._cg_solver
        t0 = time.perf_counter()
        solver.solve_dc(ctx)
        solve_s = time.perf_counter() - t0
        iters = cg.stats.get('last_cg_iters') if cg else None
        converged = iters is not None and iters < args.maxiter
        row = {
            'reg': reg, 'cold_iters': iters, 'converged': bool(converged),
            'prepare_s': prep, 'solve_s': solve_s, 'rss_gb': _rss_gb(),
            'label': getattr(cg, 'preconditioner_label', None),
        }
        rows.append(row)
        print(f"[grid] reg={reg:g}: iters={iters} converged={converged} "
              f"prepare={prep:.0f}s solve={solve_s:.1f}s rss={_rss_gb():.1f}GB",
              flush=True)
        ctx.release()
        if args.json:
            with open(args.json, 'w') as f:
                json.dump({'pkl_dir': args.pkl_dir, 'rtol': args.rtol,
                           'maxiter': args.maxiter, 'rows': rows},
                          f, indent=2, default=str)

    print('\n=== grid summary (champion cold reference: 34 iters @1e-8) ===')
    for row in rows:
        print(json.dumps(row, default=str))
    if args.json:
        print(f'wrote {args.json}')


if __name__ == '__main__':
    main()

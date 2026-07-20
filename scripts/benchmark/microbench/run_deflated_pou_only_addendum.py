#!/usr/bin/env python
"""Addendum to the deflation matrix: deflated + PoU-only (geneo_k=0) at mi200k.

Completes two cells the main matrix (run_deflated_measurement_matrix.py)
left open:
  - does the deflated apply's warm gain (23.4 -> 20.0 iters/step with GenEO)
    survive WITHOUT the 60 GenEO columns (i.e. is the +70 s GenEO eigsolve
    buying anything at this regime)?
  - cold DC under the deflated apply at scale (the matrix ran DC additive).

Usage:
    venv/bin/python -u scripts/benchmark/microbench/run_deflated_pou_only_addendum.py \
        netlist/netlist_brcm_sampled/distributed_pkl_mi200k_v2 [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import time
from typing import Dict

import numpy as np
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
    ap.add_argument('--dt', type=float, default=5e-12)
    ap.add_argument('--t-end', type=float, default=1e-10)
    ap.add_argument('--tiles-per-worker', type=int, default=4)
    ap.add_argument('--json', default=None)
    args = ap.parse_args()

    from distributed.model import (
        create_distributed_model, load_distributed_partitions,
    )
    from distributed.solver import DistributedDDMSolver

    out: Dict[str, object] = {'pkl_dir': args.pkl_dir, 'runs': {}}
    runs: Dict[str, dict] = out['runs']  # type: ignore
    n_steps = int(round(args.t_end / args.dt))

    bundle = load_distributed_partitions(args.pkl_dir)
    model = create_distributed_model(
        bundle, backend='ray', tiles_per_worker=args.tiles_per_worker)
    model.settings.update({
        'interface_solver': 'cg',
        'interface_matvec_mode': 'auto',
        'interface_preconditioner': 'two_level',
        'interface_coarse_apply_mode': 'deflated',
        'interface_coarse_geneo_k': 0,
        'interface_cg_rtol': 1e-12,
        'streaming_assembly': True,
        'interface_drop_s_global': True,
    })
    solver = DistributedDDMSolver(model)

    t0 = time.perf_counter()
    dc_ctx = solver.prepare(verbose=True)
    dc_prep = time.perf_counter() - t0
    cg = dc_ctx._cg_solver
    print(f'[DC] prepare {dc_prep:.1f}s rss={_rss_gb():.1f}GB '
          f'label={getattr(cg, "preconditioner_label", None)}', flush=True)
    if cg is not None:
        cg.progress_every = 100
    t0 = time.perf_counter()
    dc_ref = solver.solve_dc(dc_ctx)
    s12 = time.perf_counter() - t0
    it12 = cg.stats.get('last_cg_iters') if cg else None
    print(f'[DC] cold @1e-12: {it12} iters {s12:.1f}s', flush=True)
    if cg is not None:
        cg.rtol = 1e-8
        cg.reset_warm_start()
    t0 = time.perf_counter()
    solver.solve_dc(dc_ctx)
    s8 = time.perf_counter() - t0
    it8 = cg.stats.get('last_cg_iters') if cg else None
    print(f'[DC] cold @1e-8: {it8} iters {s8:.1f}s', flush=True)
    runs['dc_deflated_pou_only'] = {
        'prepare_s': dc_prep, 'cold_iters_1e12': it12, 'cold_s_1e12': s12,
        'cold_iters_1e8': it8, 'cold_s_1e8': s8, 'rss_gb': _rss_gb(),
    }
    dc_ctx.release()
    del dc_ctx

    model.settings['interface_cg_rtol'] = 1e-8
    t0 = time.perf_counter()
    tr_ctx = solver.prepare_transient(dt=args.dt, method='be', verbose=True)
    tr_prep = time.perf_counter() - t0
    cg = tr_ctx._cg_solver
    print(f'[TR] prepare {tr_prep:.1f}s rss={_rss_gb():.1f}GB '
          f'label={getattr(cg, "preconditioner_label", None)}', flush=True)
    sources = solver.preprocess_sources(
        time_step=args.dt, t_end=args.t_end, smooth=True,
        pkl_dir=args.pkl_dir, verbose=True)
    for extrap in (False, True):
        if cg is not None:
            cg._warm_start_extrapolation = extrap
            cg.reset_warm_start()
        iters_before = cg.total_iterations if cg else 0
        t0 = time.perf_counter()
        res = solver.solve_transient(
            tr_ctx, ic_voltages=dc_ref, t_end=args.t_end,
            smoothed_sources=sources, n_worst_nodes=50, verbose=True)
        loop = time.perf_counter() - t0
        iters = (cg.total_iterations if cg else 0) - iters_before
        key = 'tr_deflated_pou_only_1e-8' + ('_extrap' if extrap else '')
        runs[key] = {
            'loop_s_per_step': loop / n_steps,
            'iters_per_step': iters / n_steps,
            'peak_mV': res.peak_ir_drop * 1e3,
            'rss_gb': _rss_gb(),
        }
        print(f'[TR] 1e-8 extrap={extrap}: {iters/n_steps:.1f} iters/step, '
              f'{loop/n_steps:.2f} s/step, peak={res.peak_ir_drop*1e3:.4f} mV',
              flush=True)
        if cg is not None:
            cg._warm_start_extrapolation = False
    tr_ctx.release()

    print('\n=== summary ===')
    for k, v in runs.items():
        print(k, json.dumps(v, default=str))
    if args.json:
        with open(args.json, 'w') as f:
            json.dump(out, f, indent=2, default=str)
        print(f'wrote {args.json}')


if __name__ == '__main__':
    main()

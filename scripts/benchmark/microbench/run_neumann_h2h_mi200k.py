#!/usr/bin/env python
"""NN/BDD work package: candidate-vs-champion head-to-head on the proxy.

Runs ONE interface-preconditioner configuration through the campaign's
100-step protocol (cold DC @1e-8 + 100-step BE dt=5ps transient,
IC = DC solve) so its row is directly comparable to the champion baseline
``results_champion_100step_mi200k.json`` (two_level[deflated](jacobi+PoU):
cold 34 iters / warm 17.34 iters/step / 5.254 s/step / driver RSS 40 GB).

Usage:
    venv/bin/python -u scripts/benchmark/microbench/run_neumann_h2h_mi200k.py \
        netlist/netlist_brcm_sampled/distributed_pkl_mi200k_v2 \
        --two-level-base neumann --apply-mode deflated \
        [--t-end 5e-10] [--json results_neumann_deflated_mi200k.json]

Run under the campaign watchdog (mem_watchdog_attach.sh) -- the NN base
adds a second dense-block footprint per live context.
"""
from __future__ import annotations

import argparse
import json
import time
from typing import Dict

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
    ap.add_argument('--t-end', type=float, default=5e-10)
    ap.add_argument('--tiles-per-worker', type=int, default=4)
    ap.add_argument('--two-level-base', default='neumann',
                    choices=['auto', 'block_jacobi', 'jacobi', 'neumann'])
    ap.add_argument('--apply-mode', default='deflated',
                    choices=['deflated', 'additive'])
    ap.add_argument('--neumann-weight', default='stiffness',
                    choices=['stiffness', 'multiplicity'])
    ap.add_argument('--neumann-reg', type=float, default=0.0)
    ap.add_argument('--rtol', type=float, default=1e-8)
    ap.add_argument('--maxiter', type=int, default=2000)
    ap.add_argument('--reproject-every', type=int, default=None,
                    help='Override interface_deflated_reproject_every '
                         '(0 disables; default: library default of 50).')
    ap.add_argument('--dc-only', action='store_true',
                    help='Skip the transient phase (memory-envelope fallback '
                         'for large bundles: cold-DC row only).')
    ap.add_argument('--json', default=None)
    args = ap.parse_args()

    from distributed.model import (
        create_distributed_model, load_distributed_partitions,
    )
    from distributed.solver import DistributedDDMSolver

    n_steps = int(round(args.t_end / args.dt))
    config = (f"two_level[{args.apply_mode}]({args.two_level_base}+PoU) "
              f"geneo_k=0, tilewise CG, never-assemble, rtol {args.rtol:g}, "
              f"extrapolation on, weight={args.neumann_weight}, "
              f"reg={args.neumann_reg:g}")
    out: Dict[str, object] = {
        'pkl_dir': args.pkl_dir,
        'dt': args.dt, 't_end': args.t_end, 'n_steps': n_steps,
        'config': config,
    }
    print(f'[cfg] {config}', flush=True)

    t_all = time.perf_counter()
    bundle = load_distributed_partitions(args.pkl_dir)
    model = create_distributed_model(
        bundle, backend='ray', tiles_per_worker=args.tiles_per_worker)
    model.settings.update({
        'interface_solver': 'cg',
        'interface_matvec_mode': 'auto',
        'interface_preconditioner': 'two_level',
        'interface_two_level_base': args.two_level_base,
        'interface_neumann_weight': args.neumann_weight,
        'interface_neumann_reg': args.neumann_reg,
        'interface_coarse_apply_mode': args.apply_mode,
        'interface_coarse_geneo_k': 0,
        'interface_cg_rtol': args.rtol,
        'interface_cg_maxiter': args.maxiter,
        'streaming_assembly': True,
        'interface_drop_s_global': True,
        'interface_warm_start_extrapolation': True,
    })
    if args.reproject_every is not None:
        model.settings['interface_deflated_reproject_every'] = (
            args.reproject_every
        )
    solver = DistributedDDMSolver(model)

    t0 = time.perf_counter()
    dc_ctx = solver.prepare(verbose=True)
    dc_prep = time.perf_counter() - t0
    cg = dc_ctx._cg_solver
    print(f'[DC] prepare {dc_prep:.1f}s rss={_rss_gb():.1f}GB '
          f'label={getattr(cg, "preconditioner_label", None)}', flush=True)
    t0 = time.perf_counter()
    dc_ref = solver.solve_dc(dc_ctx)
    dc_solve = time.perf_counter() - t0
    dc_iters = cg.stats.get('last_cg_iters') if cg else None
    print(f'[DC] cold @{args.rtol:g}: {dc_iters} iters {dc_solve:.1f}s',
          flush=True)
    out['dc'] = {'prepare_s': dc_prep, 'cold_solve_s': dc_solve,
                 'cold_iters': dc_iters, 'rss_gb': _rss_gb(),
                 'label': getattr(cg, 'preconditioner_label', None)}

    if args.dc_only:
        dc_ctx.release()
        print('\n=== summary (dc-only) ===')
        print(json.dumps(out, default=str))
        if args.json:
            with open(args.json, 'w') as f:
                json.dump(out, f, indent=2, default=str)
            print(f'wrote {args.json}')
        return

    t0 = time.perf_counter()
    tr_ctx = solver.prepare_transient(dt=args.dt, method='be', verbose=True)
    tr_prep = time.perf_counter() - t0
    cg = tr_ctx._cg_solver
    print(f'[TR] prepare {tr_prep:.1f}s rss={_rss_gb():.1f}GB '
          f'label={getattr(cg, "preconditioner_label", None)}', flush=True)

    t0 = time.perf_counter()
    sources = solver.preprocess_sources(
        time_step=args.dt, t_end=args.t_end, smooth=True,
        pkl_dir=args.pkl_dir, verbose=True)
    smooth_s = time.perf_counter() - t0
    print(f'[TR] preprocess_sources {smooth_s:.1f}s', flush=True)

    iters_before = cg.total_iterations if cg else 0
    t0 = time.perf_counter()
    res = solver.solve_transient(
        tr_ctx, ic_voltages=dc_ref, t_end=args.t_end,
        smoothed_sources=sources, n_worst_nodes=50, verbose=True)
    loop = time.perf_counter() - t0
    iters = (cg.total_iterations if cg else 0) - iters_before
    end_to_end = time.perf_counter() - t_all

    ls = (res.solve_metadata.get('timings', {}) or {}).get('loop_stats', {})
    out['transient'] = {
        'prepare_s': tr_prep,
        'preprocess_sources_s': smooth_s,
        'loop_s': loop,
        'loop_s_per_step': loop / n_steps,
        'iters_per_step': iters / n_steps,
        'peak_mV': res.peak_ir_drop * 1e3,
        'peak_time_ns': res.peak_ir_drop_time * 1e9,
        'peak_node': res.peak_ir_drop_node,
        'rss_gb': _rss_gb(),
        'end_to_end_s': end_to_end,
        'label': getattr(cg, 'preconditioner_label', None),
        'loop_stats': ls,
    }
    print(f'[TR] {n_steps}-step: {iters/n_steps:.2f} iters/step, '
          f'{loop/n_steps:.3f} s/step, loop {loop:.0f}s, '
          f'peak={res.peak_ir_drop*1e3:.4f} mV, end-to-end {end_to_end:.0f}s',
          flush=True)
    if ls:
        print(f"[TR] loop_stats: rhs {ls.get('cum_rhs_time_s', 0)/n_steps:.3f} "
              f"solve {ls.get('solve_total_s', 0)/n_steps:.3f} "
              f"recovery {ls.get('cum_recovery_time_s', 0)/n_steps:.3f} s/step, "
              f"cg mean {ls.get('cg_iters_mean')} max {ls.get('cg_iters_max')}",
              flush=True)

    tr_ctx.release()
    dc_ctx.release()

    print('\n=== summary ===')
    print(json.dumps({k: v for k, v in out.items() if k != 'transient'},
                     default=str))
    tr_summary = {k: v for k, v in out['transient'].items()  # type: ignore
                  if k != 'loop_stats'}
    print(json.dumps(tr_summary, default=str))
    if args.json:
        with open(args.json, 'w') as f:
            json.dump(out, f, indent=2, default=str)
        print(f'wrote {args.json}')


if __name__ == '__main__':
    main()

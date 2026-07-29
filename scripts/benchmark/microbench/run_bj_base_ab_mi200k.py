#!/usr/bin/env python
"""A/B: does the BRCM-stuck preconditioner config work on the mi200k proxy?

The 2026-07-20 production BRCM run (logs/brcm_transient_20260720_123333.log)
built `two_level[deflated](bj+geneo k=0, T'=56, rank=56)` -- a BLOCK-JACOBI
base -- because its BJ factor estimate (3186.8 MB) sits below the 4 GB floor
of `resolve_block_jacobi_max_bytes('auto')`, so the memory-budget downgrade
never fired.  Every proxy measurement behind docs 7.9-7.12 instead ran
`two_level[deflated](jacobi+PoU)`, because at the mi200k_v2 regime the same
estimate is 10.6 GB > the 8 GB auto budget and the guard DID downgrade.

That makes the bj-base + deflated-coarse combination an unmeasured cell.
This script measures it, changing exactly one variable:

  A (control)   interface_block_jacobi_max_bytes = 'auto'  -> jacobi base
  B (BRCM cfg)  interface_block_jacobi_max_bytes = 16 GiB  -> bj base survives

Everything else is the docs-7.12 winning config (deflated apply, geneo_k=0,
tilewise CG, never-assemble, rtol 1e-8, warm-start extrapolation).

Primary metric is the COLD DC interface solve -- that is the operation the
BRCM run is stuck in (solver_td.py:913, reached via solve_transient's
dc_context initial-condition path).  A short transient follows for any
variant whose DC converged.

maxiter is bounded (default 1500) so a stagnating variant fails in minutes
instead of burning the production default of 3*n = 505,758 iterations.

Usage:
    venv/bin/python -u scripts/benchmark/microbench/run_bj_base_ab_mi200k.py \
        netlist/netlist_brcm_sampled/distributed_pkl_mi200k_v2 \
        [--json scripts/benchmark/microbench/results_bj_base_ab_mi200k.json]
"""
from __future__ import annotations

import argparse
import json
import logging
import time
import traceback
from typing import Dict, List, Optional

import psutil

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)s %(levelname)s: %(message)s',
                    datefmt='%H:%M:%S')

GIB = 1024 ** 3


def _rss_gb() -> float:
    return psutil.Process().memory_info().rss / 1e9


def _label(cg) -> Optional[str]:
    if cg is None:
        return None
    lab = getattr(cg, 'preconditioner_label', None)
    return lab() if callable(lab) else lab


def _precond_info(cg) -> Dict[str, object]:
    info: Dict[str, object] = {
        'label': _label(cg),
        'preconditioner': getattr(cg, 'preconditioner', None),
        'requested': getattr(cg, 'requested_preconditioner', None),
        'base_label': getattr(cg, '_base_precond_label', None),
        'bj_downgraded': getattr(cg, '_bj_downgraded', None),
        'apply_mode': getattr(cg, '_apply_mode', None),
        'maxiter': getattr(cg, 'maxiter', None),
        'rtol': getattr(cg, 'rtol', None),
    }
    coarse = getattr(cg, '_coarse', None)
    if coarse is not None:
        for attr in ('n_pou_cols', 'n_geneo_cols', 'rank'):
            info[attr] = getattr(coarse, attr, None)
        info['SZ_retained'] = getattr(coarse, 'SZ', None) is not None
    return info


def run_variant(
    name: str,
    bj_max_bytes: Optional[int],
    args,
    model,
    solver,
    sources,
) -> Dict[str, object]:
    """One A/B cell: DC prepare + cold DC solve (+ short transient)."""
    print(f'\n{"=" * 72}\n[{name}] bj_max_bytes='
          f'{bj_max_bytes if bj_max_bytes is not None else "auto"}\n{"=" * 72}',
          flush=True)
    rec: Dict[str, object] = {'variant': name, 'bj_max_bytes': bj_max_bytes}

    if bj_max_bytes is None:
        model.settings['interface_block_jacobi_max_bytes'] = 'auto'
    else:
        model.settings['interface_block_jacobi_max_bytes'] = int(bj_max_bytes)

    # ---- DC prepare (never-assemble) ----
    t0 = time.perf_counter()
    dc_ctx = solver.prepare(verbose=True)
    rec['dc_prepare_s'] = time.perf_counter() - t0
    cg = dc_ctx._cg_solver
    rec['dc_precond'] = _precond_info(cg)
    rec['dc_prepare_rss_gb'] = _rss_gb()
    print(f'[{name}] DC prepare {rec["dc_prepare_s"]:.1f}s '
          f'rss={rec["dc_prepare_rss_gb"]:.1f}GB label={_label(cg)}', flush=True)
    if cg is not None:
        cg.progress_every = args.progress_every

    # ---- COLD DC solve @1e-8 (the operation BRCM is stuck in) ----
    print(f'[{name}] COLD DC solve @{args.rtol:.0e} '
          f'(maxiter={getattr(cg, "maxiter", None)}) starting...', flush=True)
    t0 = time.perf_counter()
    dc_result = None
    try:
        dc_result = solver.solve_dc(dc_ctx)
        rec['dc_converged'] = True
        rec['dc_error'] = None
    except RuntimeError as exc:
        rec['dc_converged'] = False
        rec['dc_error'] = str(exc)
        print(f'[{name}] COLD DC FAILED: {exc}', flush=True)
    rec['dc_solve_s'] = time.perf_counter() - t0
    if cg is not None:
        rec['dc_iters'] = cg.stats.get('last_cg_iters')
        rec['dc_rel_residual'] = cg.stats.get('last_cg_rel_residual')
    print(f'[{name}] COLD DC: converged={rec["dc_converged"]} '
          f'iters={rec.get("dc_iters")} {rec["dc_solve_s"]:.1f}s '
          f'rel_res={rec.get("dc_rel_residual")}', flush=True)

    dc_ctx.release()
    del dc_ctx

    if not rec['dc_converged'] or args.skip_transient:
        rec['transient'] = None
        return rec

    # ---- short transient, IC = the DC solution ----
    n_steps = int(round(args.t_end / args.dt))
    t0 = time.perf_counter()
    tr_ctx = solver.prepare_transient(dt=args.dt, method='be', verbose=True)
    rec['tr_prepare_s'] = time.perf_counter() - t0
    cg_tr = tr_ctx._cg_solver
    rec['tr_precond'] = _precond_info(cg_tr)
    print(f'[{name}] TR prepare {rec["tr_prepare_s"]:.1f}s '
          f'rss={_rss_gb():.1f}GB label={_label(cg_tr)}', flush=True)
    if cg_tr is not None:
        cg_tr.progress_every = args.progress_every

    iters_before = cg_tr.total_iterations if cg_tr else 0
    t0 = time.perf_counter()
    tr: Dict[str, object] = {'n_steps': n_steps}
    try:
        res = solver.solve_transient(
            tr_ctx, ic_voltages=dc_result, t_end=args.t_end,
            smoothed_sources=sources, n_worst_nodes=50, verbose=True)
        loop = time.perf_counter() - t0
        iters = (cg_tr.total_iterations if cg_tr else 0) - iters_before
        tr.update({
            'converged': True,
            'loop_s': loop,
            's_per_step': loop / n_steps,
            'iters_per_step': iters / n_steps,
            'peak_mV': res.peak_ir_drop * 1e3,
            'peak_time_ns': res.peak_ir_drop_time * 1e9,
            'peak_node': res.peak_ir_drop_node,
        })
        print(f'[{name}] TR {n_steps} steps: {tr["iters_per_step"]:.1f} '
              f'iters/step, {tr["s_per_step"]:.3f} s/step, '
              f'peak={tr["peak_mV"]:.4f} mV', flush=True)
    except RuntimeError as exc:
        tr.update({'converged': False, 'error': str(exc),
                   'loop_s': time.perf_counter() - t0})
        print(f'[{name}] TR FAILED: {exc}', flush=True)
    rec['transient'] = tr
    rec['peak_rss_gb'] = _rss_gb()

    tr_ctx.release()
    del tr_ctx
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('pkl_dir')
    ap.add_argument('--dt', type=float, default=5e-12)
    ap.add_argument('--t-end', type=float, default=1e-10)
    ap.add_argument('--tiles-per-worker', type=int, default=4)
    ap.add_argument('--rtol', type=float, default=1e-8)
    ap.add_argument('--maxiter', type=int, default=1500,
                    help='Bounded so a stagnating variant fails in minutes '
                         '(production default is 3*n = 505,758 here).')
    ap.add_argument('--progress-every', type=int, default=25,
                    help='Log true rel-residual every N CG iterations.')
    ap.add_argument('--bj-max-bytes', type=float, default=16 * GIB,
                    help='Budget for variant B (must exceed the ~10.6 GB '
                         'estimate so the bj base survives).')
    ap.add_argument('--skip-transient', action='store_true')
    ap.add_argument('--json', default=None)
    args = ap.parse_args()

    from distributed.model import (
        create_distributed_model, load_distributed_partitions,
    )
    from distributed.solver import DistributedDDMSolver

    out: Dict[str, object] = {
        'pkl_dir': args.pkl_dir, 'dt': args.dt, 't_end': args.t_end,
        'rtol': args.rtol, 'maxiter': args.maxiter,
        'question': 'Does two_level[deflated](bj+PoU) -- the config the stuck '
                    'BRCM run built -- converge at the split regime, vs the '
                    'proxy-validated two_level[deflated](jacobi+PoU)?',
        'variants': [],
    }
    variants: List[Dict[str, object]] = out['variants']  # type: ignore

    t_all = time.perf_counter()
    bundle = load_distributed_partitions(args.pkl_dir)
    model = create_distributed_model(
        bundle, backend='ray', tiles_per_worker=args.tiles_per_worker)
    model.settings.update({
        'interface_solver': 'cg',
        'interface_matvec_mode': 'auto',
        'interface_preconditioner': 'two_level',
        'interface_coarse_apply_mode': 'deflated',
        'interface_coarse_geneo_k': 0,
        'interface_cg_rtol': args.rtol,
        'interface_cg_maxiter': args.maxiter,
        'streaming_assembly': True,
        'interface_drop_s_global': True,
        'interface_warm_start_extrapolation': True,
    })
    solver = DistributedDDMSolver(model)

    sources = None
    if not args.skip_transient:
        t0 = time.perf_counter()
        sources = solver.preprocess_sources(
            time_step=args.dt, t_end=args.t_end, smooth=True,
            pkl_dir=args.pkl_dir, verbose=True)
        print(f'[setup] preprocess_sources {time.perf_counter() - t0:.1f}s',
              flush=True)

    for name, bj in (('A_jacobi_base_control', None),
                     ('B_bj_base_brcm_config', int(args.bj_max_bytes))):
        try:
            variants.append(
                run_variant(name, bj, args, model, solver, sources))
        except Exception as exc:  # noqa: BLE001 - record and continue to B
            print(f'[{name}] UNEXPECTED FAILURE: {exc}', flush=True)
            traceback.print_exc()
            variants.append({'variant': name, 'bj_max_bytes': bj,
                             'unexpected_error': str(exc)})

    out['end_to_end_s'] = time.perf_counter() - t_all

    print('\n' + '=' * 72)
    print('=== A/B SUMMARY: cold DC interface solve (the stuck operation) ===')
    print('=' * 72)
    for v in variants:
        print(f"{v['variant']:<26} label={v.get('dc_precond', {}).get('label')}")
        print(f"{'':<26} converged={v.get('dc_converged')} "
              f"iters={v.get('dc_iters')} t={v.get('dc_solve_s', 0):.1f}s "
              f"rel_res={v.get('dc_rel_residual')}")
        tr = v.get('transient')
        if tr:
            print(f"{'':<26} transient: converged={tr.get('converged')} "
                  f"{tr.get('iters_per_step')} iters/step "
                  f"peak={tr.get('peak_mV')} mV")
    if args.json:
        with open(args.json, 'w') as fh:
            json.dump(out, fh, indent=2, default=str)
        print(f'\nWrote {args.json}')


if __name__ == '__main__':
    main()

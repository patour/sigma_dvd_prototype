#!/usr/bin/env python
"""Direct baseline + CG rtol sweep on a split proxy bundle (Stage 0).

Runs, on the same bundle and sources:
  1. direct-solver 20-step transient  -> reference waveforms + baseline timings
  2. CG (assembled matvec, block-Jacobi) 20-step transient at each rtol in
     the sweep, warm-start reset between runs

and reports, per rtol: mean CG iters/step, loop time/step, max|dV| vs the
direct reference over ~600 tracked nodes (200 worst + boundary sample),
and the peak-drop delta.  This chooses the production rtol default
(<= 1 uV target per the plan).

Usage:
    venv/bin/python scripts/benchmark/microbench/run_rtol_sweep.py \
        netlist/netlist_brcm_sampled/distributed_pkl_mi100k \
        [--dt 5e-12] [--t-end 1e-10] \
        [--rtols 1e-12,1e-10,1e-8,1e-7,1e-6] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from typing import Dict, List

import numpy as np

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)s %(levelname)s: %(message)s',
                    datefmt='%H:%M:%S')


def _pick_track_nodes(worst: List, boundary_nodes, n_bnd: int = 400) -> List[str]:
    track = [w[0] for w in worst]
    bnd = sorted(boundary_nodes)
    stride = max(1, len(bnd) // n_bnd)
    track += bnd[::stride][:n_bnd]
    return list(dict.fromkeys(track))


def _waveform_err(ref: Dict[str, np.ndarray],
                  got: Dict[str, np.ndarray]) -> float:
    err = 0.0
    for node, w in ref.items():
        g = got.get(node)
        if g is None:
            continue
        m = min(len(w), len(g))
        err = max(err, float(np.max(np.abs(np.asarray(w[:m]) -
                                           np.asarray(g[:m])))))
    return err


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('pkl_dir')
    ap.add_argument('--dt', type=float, default=5e-12)
    ap.add_argument('--t-end', type=float, default=1e-10)
    ap.add_argument('--rtols', default='1e-12,1e-10,1e-8,1e-7,1e-6')
    ap.add_argument('--json', default=None)
    args = ap.parse_args()

    from distributed.model import (
        create_distributed_model, load_distributed_partitions,
    )
    from distributed.solver import DistributedDDMSolver
    import distributed.interface_iterative as ii

    # Stage 1c preview: keep genuine block-Jacobi at 356K unknowns (default
    # 4 GB cap would silently fall back to plain Jacobi and skew the sweep).
    ii.BLOCK_JACOBI_MAX_FACTOR_BYTES = 32 * 1024 ** 3

    out: Dict[str, object] = {'pkl_dir': args.pkl_dir, 'dt': args.dt,
                              't_end': args.t_end, 'runs': {}}
    runs: Dict[str, dict] = out['runs']  # type: ignore

    bundle = load_distributed_partitions(args.pkl_dir)
    model = create_distributed_model(bundle, backend='ray')
    model.settings['interface_solver'] = 'direct'
    solver = DistributedDDMSolver(model)

    t0 = time.perf_counter()
    dc_ctx = solver.prepare(verbose=True)
    dc_prep = time.perf_counter() - t0
    t0 = time.perf_counter()
    tr_ctx = solver.prepare_transient(dt=args.dt, method='be', verbose=True)
    tr_prep = time.perf_counter() - t0
    sources = solver.preprocess_sources(
        time_step=args.dt, t_end=args.t_end, smooth=True,
        pkl_dir=args.pkl_dir, verbose=True)

    # Pass 1: find worst nodes
    t0 = time.perf_counter()
    res = solver.solve_transient(
        tr_ctx, dc_context=dc_ctx, t_end=args.t_end,
        smoothed_sources=sources, n_worst_nodes=200, verbose=True)
    direct1_s = time.perf_counter() - t0
    track = _pick_track_nodes(res.worst_nodes, bundle.shared_boundary_nodes)
    print(f'direct pass 1: {direct1_s:.1f}s  peak={res.peak_ir_drop*1e3:.3f} mV '
          f'@ {res.peak_ir_drop_time*1e9:.3f} ns  tracking {len(track)} nodes')

    # Pass 2: direct reference with waveforms
    t0 = time.perf_counter()
    res_ref = solver.solve_transient(
        tr_ctx, dc_context=dc_ctx, t_end=args.t_end,
        smoothed_sources=sources, n_worst_nodes=50,
        track_nodes=track, verbose=True)
    runs['direct'] = {
        'loop_s': time.perf_counter() - t0,
        'dc_prepare_s': dc_prep,
        'trans_prepare_s': tr_prep,
        'peak_mV': res_ref.peak_ir_drop * 1e3,
        'peak_t_ns': res_ref.peak_ir_drop_time * 1e9,
        'peak_node': res_ref.peak_ir_drop_node,
    }
    ref_wf = {k: np.asarray(v) for k, v in res_ref.tracked_waveforms.items()}

    tr_ctx.release()
    dc_ctx.release()

    # CG contexts (prepared once; rtol mutated per run)
    model.settings.update({
        'interface_solver': 'cg',
        'interface_matvec_mode': 'assembled',
        'interface_preconditioner': 'block_jacobi',
        'interface_cg_rtol': 1e-12,
    })
    t0 = time.perf_counter()
    dc2 = solver.prepare(verbose=True)
    cg_dc_prep = time.perf_counter() - t0
    t0 = time.perf_counter()
    tr2 = solver.prepare_transient(dt=args.dt, method='be', verbose=True)
    cg_tr_prep = time.perf_counter() - t0
    runs['cg_prepare'] = {'dc_prepare_s': cg_dc_prep,
                          'trans_prepare_s': cg_tr_prep}

    n_steps = int(round(args.t_end / args.dt))
    for rtol_s in args.rtols.split(','):
        rtol = float(rtol_s)
        for ctx in (dc2, tr2):
            cg = ctx._cg_solver
            if cg is not None:
                cg.rtol = rtol
                cg.reset_warm_start()
        cg_tr = tr2._cg_solver
        iters_before = cg_tr.total_iterations if cg_tr else 0
        t0 = time.perf_counter()
        res_cg = solver.solve_transient(
            tr2, dc_context=dc2, t_end=args.t_end,
            smoothed_sources=sources, n_worst_nodes=50,
            track_nodes=track, verbose=True)
        loop_s = time.perf_counter() - t0
        iters = (cg_tr.total_iterations if cg_tr else 0) - iters_before
        wf_err = _waveform_err(ref_wf, {k: np.asarray(v) for k, v in
                                        res_cg.tracked_waveforms.items()})
        runs[f'cg_{rtol_s}'] = {
            'rtol': rtol,
            'loop_s': loop_s,
            'iters_total': int(iters),
            'iters_per_step': iters / n_steps,
            'max_dV_V': wf_err,
            'peak_mV': res_cg.peak_ir_drop * 1e3,
            'peak_delta_mV': (res_cg.peak_ir_drop -
                              res_ref.peak_ir_drop) * 1e3,
            'peak_node_match': (res_cg.peak_ir_drop_node ==
                                res_ref.peak_ir_drop_node),
        }
        print(f"rtol={rtol_s}: loop={loop_s:.1f}s "
              f"iters/step={iters / n_steps:.1f} "
              f"max|dV|={wf_err:.3e} V "
              f"peak_delta={runs[f'cg_{rtol_s}']['peak_delta_mV']:.6f} mV")

    tr2.release()
    dc2.release()

    print('\n=== summary ===')
    for k, v in runs.items():
        print(k, json.dumps(v, default=str))
    if args.json:
        with open(args.json, 'w') as f:
            json.dump(out, f, indent=2, default=str)
        print(f'wrote {args.json}')


if __name__ == '__main__':
    main()

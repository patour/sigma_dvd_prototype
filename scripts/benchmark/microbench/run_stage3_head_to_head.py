#!/usr/bin/env python
"""Stage 3 gate measurement: two_level vs block_jacobi on the mi200k proxy.

Head-to-head at the split regime (64 tiles / 167,659 interface unknowns):

  A. two_level DC (never-assemble): cold solve @1e-8 must CONVERGE (Stage 2
     measured cold block_jacobi CG stagnating at rel-res ~0.27 — §7.8), plus
     a @1e-12 solve for the accuracy diff (direct factor is infeasible here;
     CG@1e-12 == direct to 1.9e-11 V per the Stage 0 36-tile sweep, §7.7).
  B. two_level transient, 20 steps BE, IC = the DC result: warm iters/step
     (GATE: <= 30 at rtol 1e-8), s/step, accuracy vs a @1e-12 run.
  C. block_jacobi transient, same 20 steps, same IC: the BJ warm-iteration
     baseline that Stage 2 deliberately deferred to this head-to-head.

Memory discipline (Stage 2 lessons): DC and TD CG contexts do NOT fit
simultaneously on this host — each phase releases its context before the
next prepare; the driver should run under the external memory watchdog.

Usage:
    venv/bin/python -u scripts/benchmark/microbench/run_stage3_head_to_head.py \
        netlist/netlist_brcm_sampled/distributed_pkl_mi200k_v2 \
        [--dt 5e-12] [--t-end 1e-10] [--tiles-per-worker 4] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from typing import Dict, List

import numpy as np
import psutil

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)s %(levelname)s: %(message)s',
                    datefmt='%H:%M:%S')


def _rss_gb() -> float:
    return psutil.Process().memory_info().rss / 1e9


def _pick_track_nodes(worst: List, boundary_nodes, n_bnd: int = 400) -> List[str]:
    track = [w[0] for w in worst]
    bnd = sorted(boundary_nodes)
    stride = max(1, len(bnd) // n_bnd)
    track += bnd[::stride][:n_bnd]
    return list(dict.fromkeys(track))


def _waveform_err(ref: Dict[str, np.ndarray], got: Dict[str, np.ndarray]) -> float:
    err = 0.0
    for node, w in ref.items():
        g = got.get(node)
        if g is None:
            continue
        m = min(len(w), len(g))
        err = max(err, float(np.max(np.abs(np.asarray(w[:m]) - np.asarray(g[:m])))))
    return err


def _cg_precond_info(cg) -> Dict[str, object]:
    info: Dict[str, object] = {
        'preconditioner': getattr(cg, 'preconditioner', None),
        'label': None,
    }
    label_fn = getattr(cg, 'preconditioner_label', None)
    if callable(label_fn):
        info['label'] = label_fn()
    elif label_fn is not None:
        info['label'] = label_fn
    coarse = getattr(cg, '_coarse', None)
    if coarse is not None:
        for attr in ('n_pou_cols', 'n_geneo_cols', 'rank', 'n_dropped_cols'):
            info[attr] = getattr(coarse, attr, None)
    return info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('pkl_dir')
    ap.add_argument('--dt', type=float, default=5e-12)
    ap.add_argument('--t-end', type=float, default=1e-10)
    ap.add_argument('--tiles-per-worker', type=int, default=4)
    ap.add_argument('--dc-maxiter', type=int, default=4000)
    ap.add_argument('--bj-max-bytes', type=float, default=None,
                    help='Override interface_block_jacobi_max_bytes (bytes). '
                         'The default 8 GB auto budget downgrades the base to '
                         'diagonal jacobi at this regime (est 10.6 GB); pass '
                         'e.g. 17179869184 to measure the bj+geneo variant.')
    ap.add_argument('--skip-tr-reference', action='store_true',
                    help='Skip the rtol-1e-12 transient reference (accuracy '
                         'already established by a prior run); report only '
                         'iters/step, s/step and peak for the 1e-8 runs.')
    ap.add_argument('--json', default=None)
    args = ap.parse_args()

    from distributed.model import (
        create_distributed_model, load_distributed_partitions,
    )
    from distributed.solver import DistributedDDMSolver

    out: Dict[str, object] = {'pkl_dir': args.pkl_dir, 'dt': args.dt,
                              't_end': args.t_end, 'runs': {}}
    runs: Dict[str, dict] = out['runs']  # type: ignore
    n_steps = int(round(args.t_end / args.dt))

    bundle = load_distributed_partitions(args.pkl_dir)
    model = create_distributed_model(
        bundle, backend='ray', tiles_per_worker=args.tiles_per_worker)
    model.settings.update({
        'interface_solver': 'cg',
        'interface_matvec_mode': 'auto',
        'interface_preconditioner': 'two_level',
        'interface_cg_rtol': 1e-12,
        'interface_cg_maxiter': args.dc_maxiter,
        'streaming_assembly': True,
        'interface_drop_s_global': True,   # DC: never assemble S_global
    })
    if args.bj_max_bytes is not None:
        model.settings['interface_block_jacobi_max_bytes'] = int(args.bj_max_bytes)
        out['bj_max_bytes'] = int(args.bj_max_bytes)
    solver = DistributedDDMSolver(model)

    # ============ Phase A: two_level DC (never-assemble, COLD) ============
    t0 = time.perf_counter()
    dc_ctx = solver.prepare(verbose=True)
    dc_prep = time.perf_counter() - t0
    cg_dc = dc_ctx._cg_solver
    print(f'[A] DC prepare {dc_prep:.1f}s rss={_rss_gb():.1f}GB '
          f'precond={_cg_precond_info(cg_dc)}', flush=True)
    if cg_dc is not None:
        cg_dc.progress_every = 100

    print('[A] DC cold solve @1e-12 starting...', flush=True)
    t0 = time.perf_counter()
    dc_1e12_failed = None
    try:
        dc_ref = solver.solve_dc(dc_ctx)
    except RuntimeError as exc:
        # Cold stagnation at 1e-12 (the Stage 2 block-jacobi failure mode).
        # Record it and fall back to rtol 1e-8 so the transient phases still
        # get an IC; accuracy vs 1e-12 is then unavailable for this variant.
        dc_1e12_failed = str(exc)
        print(f'[A] DC @1e-12 FAILED after {time.perf_counter() - t0:.1f}s: '
              f'{exc}', flush=True)
        if cg_dc is not None:
            cg_dc.rtol = 1e-8
            cg_dc.reset_warm_start()
        t0 = time.perf_counter()
        dc_ref = solver.solve_dc(dc_ctx)
    dc_ref_s = time.perf_counter() - t0
    dc_ref_iters = cg_dc.stats.get('last_cg_iters') if cg_dc else None
    print(f'[A] DC @1e-12 done: {dc_ref_s:.1f}s, {dc_ref_iters} iters', flush=True)
    ref_flat = dc_ref.flatten()

    if cg_dc is not None:
        cg_dc.rtol = 1e-8
        cg_dc.reset_warm_start()
    print('[A] DC cold solve @1e-8 starting...', flush=True)
    t0 = time.perf_counter()
    dc_b = solver.solve_dc(dc_ctx)
    dc_b_s = time.perf_counter() - t0
    dc_b_iters = cg_dc.stats.get('last_cg_iters') if cg_dc else None
    b_flat = dc_b.flatten()
    dc_err = max(abs(ref_flat[n] - b_flat[n]) for n in ref_flat)
    runs['dc_two_level'] = {
        'prepare_s': dc_prep,
        'precond_info': _cg_precond_info(cg_dc),
        'cold_solve_1e12_s': dc_ref_s, 'cold_iters_1e12': dc_ref_iters,
        'dc_1e12_failed': dc_1e12_failed,
        'cold_solve_1e8_s': dc_b_s, 'cold_iters_1e8': dc_b_iters,
        'max_dV_V_1e8_vs_1e12': dc_err,
        'rss_gb': _rss_gb(),
    }
    print(f'[A] DC @1e-8 done: {dc_b_s:.1f}s, {dc_b_iters} iters, '
          f'max|dV|={dc_err:.3e} V rss={_rss_gb():.1f}GB', flush=True)
    del b_flat, ref_flat, dc_b
    dc_ctx.release()
    del dc_ctx

    # ============ Phase B: two_level transient (IC = DC result) ============
    model.settings['interface_cg_rtol'] = 1e-12
    t0 = time.perf_counter()
    tr_ctx = solver.prepare_transient(dt=args.dt, method='be', verbose=True)
    tr_prep = time.perf_counter() - t0
    cg_tr = tr_ctx._cg_solver
    print(f'[B] TR prepare (two_level) {tr_prep:.1f}s rss={_rss_gb():.1f}GB '
          f'precond={_cg_precond_info(cg_tr)}', flush=True)
    sources = solver.preprocess_sources(
        time_step=args.dt, t_end=args.t_end, smooth=True,
        pkl_dir=args.pkl_dir, verbose=True)
    if cg_tr is not None:
        cg_tr.progress_every = 200

    if args.skip_tr_reference:
        track = []
        ref_wf: Dict[str, np.ndarray] = {}
        res_a = None
    else:
        res_a1 = solver.solve_transient(
            tr_ctx, ic_voltages=dc_ref, t_end=args.t_end,
            smoothed_sources=sources, n_worst_nodes=200, verbose=True)
        track = _pick_track_nodes(res_a1.worst_nodes, bundle.shared_boundary_nodes)
        if cg_tr is not None:
            cg_tr.reset_warm_start()
        t0 = time.perf_counter()
        res_a = solver.solve_transient(
            tr_ctx, ic_voltages=dc_ref, t_end=args.t_end,
            smoothed_sources=sources, n_worst_nodes=50,
            track_nodes=track, verbose=True)
        loop_a = time.perf_counter() - t0
        stats_a = dict(getattr(res_a, 'loop_stats', {}) or {})
        runs['tr_two_level_1e-12_reference'] = {
            'tr_prepare_s': tr_prep,
            'precond_info': _cg_precond_info(cg_tr),
            'loop_s': loop_a, 'loop_s_per_step': loop_a / n_steps,
            'solve_total_s': stats_a.get('solve_total_s'),
            'cg_iters_mean': stats_a.get('cg_iters_mean'),
            'cg_iters_max': stats_a.get('cg_iters_max'),
            'peak_mV': res_a.peak_ir_drop * 1e3,
            'peak_node': res_a.peak_ir_drop_node,
            'rss_gb': _rss_gb(),
        }
        print(f"[B] tr two_level@1e-12 loop {loop_a:.1f}s ({loop_a/n_steps:.2f} s/step) "
              f"iters mean={stats_a.get('cg_iters_mean')}", flush=True)
        ref_wf = {k: np.asarray(v) for k, v in res_a.tracked_waveforms.items()}

    if cg_tr is not None:
        cg_tr.rtol = 1e-8
        cg_tr.reset_warm_start()
    iters_before = cg_tr.total_iterations if cg_tr else 0
    t0 = time.perf_counter()
    res_b = solver.solve_transient(
        tr_ctx, ic_voltages=dc_ref, t_end=args.t_end,
        smoothed_sources=sources, n_worst_nodes=50,
        track_nodes=track, verbose=True)
    loop_b = time.perf_counter() - t0
    iters_b = (cg_tr.total_iterations if cg_tr else 0) - iters_before
    stats_b = dict(getattr(res_b, 'loop_stats', {}) or {})
    wf_err = _waveform_err(ref_wf, {k: np.asarray(v) for k, v in
                                    res_b.tracked_waveforms.items()})
    runs['tr_two_level_1e-8'] = {
        'loop_s': loop_b, 'loop_s_per_step': loop_b / n_steps,
        'solve_total_s': stats_b.get('solve_total_s'),
        'cg_iters_per_step': iters_b / n_steps,
        'cg_iters_mean': stats_b.get('cg_iters_mean'),
        'cg_iters_max': stats_b.get('cg_iters_max'),
        'max_dV_V': wf_err if ref_wf else None,
        'peak_mV': res_b.peak_ir_drop * 1e3,
        'peak_delta_mV': ((res_b.peak_ir_drop - res_a.peak_ir_drop) * 1e3
                          if res_a is not None else None),
        'peak_node_match': (res_b.peak_ir_drop_node == res_a.peak_ir_drop_node
                            if res_a is not None else None),
        'rss_gb': _rss_gb(),
    }
    print(f"[B] tr two_level@1e-8 loop {loop_b:.1f}s ({loop_b/n_steps:.2f} s/step) "
          f"iters/step={iters_b/n_steps:.1f} (GATE <= 30) "
          f"max|dV|={wf_err:.3e} V" if ref_wf else
          f"[B] tr two_level@1e-8 loop {loop_b:.1f}s ({loop_b/n_steps:.2f} s/step) "
          f"iters/step={iters_b/n_steps:.1f} (GATE <= 30)", flush=True)
    tr_ctx.release()
    del tr_ctx, res_b

    # ============ Phase C: block_jacobi transient (same IC) ============
    model.settings['interface_preconditioner'] = 'block_jacobi'
    model.settings['interface_cg_rtol'] = 1e-8
    t0 = time.perf_counter()
    bj_ctx = solver.prepare_transient(dt=args.dt, method='be', verbose=True)
    bj_prep = time.perf_counter() - t0
    cg_bj = bj_ctx._cg_solver
    print(f'[C] TR prepare (block_jacobi) {bj_prep:.1f}s rss={_rss_gb():.1f}GB '
          f'precond={_cg_precond_info(cg_bj)}', flush=True)
    if cg_bj is not None:
        cg_bj.progress_every = 200
    iters_before = cg_bj.total_iterations if cg_bj else 0
    t0 = time.perf_counter()
    res_c = solver.solve_transient(
        bj_ctx, ic_voltages=dc_ref, t_end=args.t_end,
        smoothed_sources=sources, n_worst_nodes=50,
        track_nodes=track, verbose=True)
    loop_c = time.perf_counter() - t0
    iters_c = (cg_bj.total_iterations if cg_bj else 0) - iters_before
    stats_c = dict(getattr(res_c, 'loop_stats', {}) or {})
    wf_err_c = (_waveform_err(ref_wf, {k: np.asarray(v) for k, v in
                                       res_c.tracked_waveforms.items()})
                if ref_wf else None)
    runs['tr_block_jacobi_1e-8'] = {
        'tr_prepare_s': bj_prep,
        'precond_info': _cg_precond_info(cg_bj),
        'loop_s': loop_c, 'loop_s_per_step': loop_c / n_steps,
        'solve_total_s': stats_c.get('solve_total_s'),
        'cg_iters_per_step': iters_c / n_steps,
        'cg_iters_mean': stats_c.get('cg_iters_mean'),
        'cg_iters_max': stats_c.get('cg_iters_max'),
        'max_dV_V': wf_err_c,
        'rss_gb': _rss_gb(),
    }
    print(f"[C] tr block_jacobi@1e-8 loop {loop_c:.1f}s ({loop_c/n_steps:.2f} s/step) "
          f"iters/step={iters_c/n_steps:.1f} max|dV|="
          + (f"{wf_err_c:.3e} V" if wf_err_c is not None else "n/a"), flush=True)
    bj_ctx.release()

    print('\n=== summary ===')
    for k, v in runs.items():
        print(k, json.dumps(v, default=str))
    if args.json:
        with open(args.json, 'w') as f:
            json.dump(out, f, indent=2, default=str)
        print(f'wrote {args.json}')


if __name__ == '__main__':
    main()

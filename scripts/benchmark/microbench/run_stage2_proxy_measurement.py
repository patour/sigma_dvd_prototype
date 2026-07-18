#!/usr/bin/env python
"""Stage 2 gate measurement on the mi200k split-regime proxy (64 tiles, 167K).

Runs, on netlist_brcm_sampled/distributed_pkl_mi200k (the closest local
analog of the BRCM 107-tile/190K regime):

  A. direct reference with streaming_assembly=True (Finding 0 workaround:
     pre-allocated CSR keeps assembly peak bounded) — 20-step BE transient,
     tracked waveforms for the accuracy diff.
  B. CG tilewise (Stage 2 production config): threaded tilewise matvec +
     threaded block-Jacobi, rtol 1e-8, streaming assembly — same 20 steps.

Reports: prepare times, per-step solve_pure (Stage 1a timer), CG iters
mean/max, max|dV| B vs A over tracked nodes, peak-drop delta, RSS peaks.
Gate (plan Stage 2): solve/step >= 5x vs the Stage 0 CG baseline
(329 s/step BRCM 107-tile assembled-1t; local Stage 0 microbench floor:
serial tilewise 570 ms/matvec) and accuracy <= 1 uV at rtol 1e-8.

Usage:
    venv/bin/python -u scripts/benchmark/microbench/run_stage2_proxy_measurement.py \
        netlist/netlist_brcm_sampled/distributed_pkl_mi200k \
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

    out: Dict[str, object] = {'pkl_dir': args.pkl_dir, 'dt': args.dt,
                              't_end': args.t_end, 'runs': {}}
    runs: Dict[str, dict] = out['runs']  # type: ignore
    n_steps = int(round(args.t_end / args.dt))

    bundle = load_distributed_partitions(args.pkl_dir)
    model = create_distributed_model(
        bundle, backend='ray', tiles_per_worker=args.tiles_per_worker)

    # Direct reference is INFEASIBLE at the split regime on this host
    # (measured 2026-07-18: workers + S_global CSR + supernodal interface
    # factor > 195 GB — watchdog-killed). Holding DC AND TD CG contexts
    # simultaneously ALSO overflows (two S_globals + two block sets + both
    # worker factor sets). So: two sequential phases, one mode live at a
    # time; accuracy chain = CG rtol 1e-12 == direct to 1.9e-11 V (Stage 0
    # 36-tile sweep, docs §7.7).
    model.settings.update({
        'interface_solver': 'cg',
        'interface_matvec_mode': 'auto',
        'interface_preconditioner': 'block_jacobi',
        'interface_cg_rtol': 1e-12,
        'streaming_assembly': True,
        'interface_drop_s_global': True,   # DC: never assemble S_global
    })
    solver = DistributedDDMSolver(model)

    # ================= Phase 1: DC (never-assemble) ======================
    t0 = time.perf_counter()
    dc_ctx = solver.prepare(verbose=True)
    dc_prep = time.perf_counter() - t0
    print(f'[A] DC prepare (CG never-assemble) {dc_prep:.1f}s '
          f'rss={_rss_gb():.1f}GB', flush=True)
    cg_dc = dc_ctx._cg_solver
    if cg_dc is not None:
        cg_dc.progress_every = 200  # log true rel-res every 200 CG iters

    print('[A] DC solve @ rtol 1e-12 (cold reference) starting...', flush=True)
    t0 = time.perf_counter()
    dc_ref = solver.solve_dc(dc_ctx)
    dc_solve_ref_s = time.perf_counter() - t0
    print(f'[A] DC solve @1e-12 done: {dc_solve_ref_s:.1f}s, '
          f'{cg_dc.stats.get("last_cg_iters") if cg_dc else "?"} iters', flush=True)
    ref_flat = dc_ref.flatten()
    if cg_dc is not None:
        cg_dc.rtol = 1e-8
        cg_dc.reset_warm_start()
    print('[A] DC solve @ rtol 1e-8 (cold) starting...', flush=True)
    t0 = time.perf_counter()
    dc_b = solver.solve_dc(dc_ctx)
    dc_solve_b_s = time.perf_counter() - t0
    print(f'[A] DC solve @1e-8 done: {dc_solve_b_s:.1f}s, '
          f'{cg_dc.stats.get("last_cg_iters") if cg_dc else "?"} iters', flush=True)
    b_flat = dc_b.flatten()
    dc_err = max(abs(ref_flat[n] - b_flat[n]) for n in ref_flat)
    runs['dc'] = {
        'dc_prepare_s': dc_prep,
        'solve_1e12_s': dc_solve_ref_s,
        'solve_1e8_s': dc_solve_b_s,
        'max_dV_V_1e8_vs_1e12': dc_err,
        'rss_gb': _rss_gb(),
    }
    print(f"[A] DC solves: 1e-12 {dc_solve_ref_s:.1f}s, 1e-8 {dc_solve_b_s:.1f}s, "
          f"max|dV|={dc_err:.3e} V rss={_rss_gb():.1f}GB", flush=True)
    del b_flat, ref_flat, dc_b
    dc_ctx.release()
    del dc_ctx

    # ================= Phase 2: transient (ic_voltages) ==================
    model.settings['interface_cg_rtol'] = 1e-12
    t0 = time.perf_counter()
    tr_ctx = solver.prepare_transient(dt=args.dt, method='be', verbose=True)
    tr_prep = time.perf_counter() - t0
    print(f'[B] TR prepare (CG) {tr_prep:.1f}s rss={_rss_gb():.1f}GB', flush=True)
    sources = solver.preprocess_sources(
        time_step=args.dt, t_end=args.t_end, smooth=True,
        pkl_dir=args.pkl_dir, verbose=True)
    cg_tr = tr_ctx._cg_solver
    if cg_tr is not None:
        cg_tr.progress_every = 200

    # Run A: rtol 1e-12 transient reference (IC = 1e-12 DC result)
    iters_before = cg_tr.total_iterations if cg_tr else 0
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
    iters_a = (cg_tr.total_iterations if cg_tr else 0) - iters_before
    stats_a = dict(getattr(res_a, 'loop_stats', {}) or {})
    runs['tr_1e-12_reference'] = {
        'tr_prepare_s': tr_prep,
        'loop_s': loop_a, 'loop_s_per_step': loop_a / n_steps,
        'solve_total_s': stats_a.get('solve_total_s'),
        'cg_iters_mean': stats_a.get('cg_iters_mean'),
        'cg_iters_max': stats_a.get('cg_iters_max'),
        'peak_mV': res_a.peak_ir_drop * 1e3,
        'peak_node': res_a.peak_ir_drop_node,
        'matvec_mode_resolved': getattr(cg_tr, 'matvec_mode', None),
        'precond': getattr(cg_tr, 'preconditioner', None),
        'rss_gb': _rss_gb(),
    }
    print(f"[B] tr@1e-12 loop {loop_a:.1f}s ({loop_a/n_steps:.2f} s/step) "
          f"peak={res_a.peak_ir_drop*1e3:.4f} mV rss={_rss_gb():.1f}GB", flush=True)
    ref_wf = {k: np.asarray(v) for k, v in res_a.tracked_waveforms.items()}

    # Run B: rtol 1e-8 production default (same IC)
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
    iters = (cg_tr.total_iterations if cg_tr else 0) - iters_before
    stats_b = dict(getattr(res_b, 'loop_stats', {}) or {})
    wf_err = _waveform_err(ref_wf, {k: np.asarray(v) for k, v in
                                    res_b.tracked_waveforms.items()})
    runs['tr_1e-8'] = {
        'loop_s': loop_b, 'loop_s_per_step': loop_b / n_steps,
        'solve_total_s': stats_b.get('solve_total_s'),
        'cg_iters_total': int(iters),
        'cg_iters_per_step': iters / n_steps,
        'cg_iters_mean': stats_b.get('cg_iters_mean'),
        'cg_iters_max': stats_b.get('cg_iters_max'),
        'max_dV_V': wf_err,
        'peak_mV': res_b.peak_ir_drop * 1e3,
        'peak_delta_mV': (res_b.peak_ir_drop - res_a.peak_ir_drop) * 1e3,
        'peak_node_match': res_b.peak_ir_drop_node == res_a.peak_ir_drop_node,
        'rss_gb': _rss_gb(),
    }
    print(f"[B] tr@1e-8 loop {loop_b:.1f}s ({loop_b/n_steps:.2f} s/step) "
          f"iters/step={iters/n_steps:.1f} max|dV|={wf_err:.3e} V "
          f"peak_delta={runs['tr_1e-8']['peak_delta_mV']:.6f} mV", flush=True)

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

#!/usr/bin/env python
"""Deflation-package gate matrix on the mi200k proxy (warm iters <= 10 target).

Measures, at the split regime (64 tiles / 168,586 interface unknowns,
never-assemble DC+TD, tilewise CG, jacobi-downgraded base per §7.9):

  A. DC (apply_mode=additive, decoupled GenEO now ACTIVE by default):
     cold @1e-12 + @1e-8 — comparable to §7.9's PoU-only 118/70 iters.
  B. TD additive: 1e-12 tracked reference, then 1e-8 with extrapolation
     off/on (extrapolation toggled on the live solver, no re-prepare).
  C. TD deflated: prepare (SZ retained), 1e-8 extrapolation off/on.
  D. TD additive PoU-only (geneo_k=0): isolates GenEO's warm contribution
     vs the §7.10 baseline config (23.6 iters/step).

Accuracy: every 1e-8 run diffs tracked waveforms vs the phase-B 1e-12
reference (budget <= 1 uV).  Memory discipline: one TD context alive at a
time; run under the external watchdog.

Usage:
    venv/bin/python -u scripts/benchmark/microbench/run_deflated_measurement_matrix.py \
        netlist/netlist_brcm_sampled/distributed_pkl_mi200k_v2 [--json out.json]
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


def _cg_info(cg) -> Dict[str, object]:
    info: Dict[str, object] = {'label': getattr(cg, 'preconditioner_label', None)}
    coarse = getattr(cg, '_coarse', None)
    if coarse is not None:
        for attr in ('n_pou_cols', 'n_geneo_cols', 'rank'):
            info[attr] = getattr(coarse, attr, None)
        info['sz_retained'] = getattr(coarse, 'SZ', None) is not None
    return info


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
    model.settings.update({
        'interface_solver': 'cg',
        'interface_matvec_mode': 'auto',
        'interface_preconditioner': 'two_level',
        'interface_coarse_apply_mode': 'additive',
        'interface_cg_rtol': 1e-12,
        'streaming_assembly': True,
        'interface_drop_s_global': True,
    })
    solver = DistributedDDMSolver(model)

    # ================= A: DC additive (GenEO active by default) ============
    t0 = time.perf_counter()
    dc_ctx = solver.prepare(verbose=True)
    dc_prep = time.perf_counter() - t0
    cg_dc = dc_ctx._cg_solver
    print(f'[A] DC prepare {dc_prep:.1f}s rss={_rss_gb():.1f}GB '
          f'{_cg_info(cg_dc)}', flush=True)
    if cg_dc is not None:
        cg_dc.progress_every = 100
    t0 = time.perf_counter()
    dc_ref = solver.solve_dc(dc_ctx)
    dc_ref_s = time.perf_counter() - t0
    it12 = cg_dc.stats.get('last_cg_iters') if cg_dc else None
    print(f'[A] DC @1e-12 done: {dc_ref_s:.1f}s, {it12} iters', flush=True)
    ref_flat = dc_ref.flatten()
    if cg_dc is not None:
        cg_dc.rtol = 1e-8
        cg_dc.reset_warm_start()
    t0 = time.perf_counter()
    dc_b = solver.solve_dc(dc_ctx)
    dc_b_s = time.perf_counter() - t0
    it8 = cg_dc.stats.get('last_cg_iters') if cg_dc else None
    b_flat = dc_b.flatten()
    dc_err = max(abs(ref_flat[n] - b_flat[n]) for n in ref_flat)
    runs['dc_additive_geneo'] = {
        'prepare_s': dc_prep, 'cg': _cg_info(cg_dc),
        'cold_iters_1e12': it12, 'cold_s_1e12': dc_ref_s,
        'cold_iters_1e8': it8, 'cold_s_1e8': dc_b_s,
        'max_dV_V': dc_err, 'rss_gb': _rss_gb(),
    }
    print(f'[A] DC @1e-8: {it8} iters {dc_b_s:.1f}s max|dV|={dc_err:.3e} V',
          flush=True)
    del b_flat, ref_flat, dc_b
    dc_ctx.release()
    del dc_ctx

    sources = None
    track: List[str] = []
    ref_wf: Dict[str, np.ndarray] = {}

    def _td_phase(tag: str, apply_mode: str, geneo_k=None,
                  make_reference: bool = False) -> None:
        nonlocal sources, track, ref_wf
        model.settings['interface_coarse_apply_mode'] = apply_mode
        if geneo_k is not None:
            model.settings['interface_coarse_geneo_k'] = geneo_k
        model.settings['interface_cg_rtol'] = 1e-12
        t0 = time.perf_counter()
        tr_ctx = solver.prepare_transient(dt=args.dt, method='be', verbose=True)
        tr_prep = time.perf_counter() - t0
        cg = tr_ctx._cg_solver
        print(f'[{tag}] TR prepare {tr_prep:.1f}s rss={_rss_gb():.1f}GB '
              f'{_cg_info(cg)}', flush=True)
        if sources is None:
            sources = solver.preprocess_sources(
                time_step=args.dt, t_end=args.t_end, smooth=True,
                pkl_dir=args.pkl_dir, verbose=True)
        if cg is not None:
            cg.progress_every = 200

        if make_reference:
            res_w = solver.solve_transient(
                tr_ctx, ic_voltages=dc_ref, t_end=args.t_end,
                smoothed_sources=sources, n_worst_nodes=200, verbose=True)
            track.extend(_pick_track_nodes(res_w.worst_nodes,
                                           bundle.shared_boundary_nodes))
            track[:] = list(dict.fromkeys(track))
            if cg is not None:
                cg.reset_warm_start()
            t0 = time.perf_counter()
            res_r = solver.solve_transient(
                tr_ctx, ic_voltages=dc_ref, t_end=args.t_end,
                smoothed_sources=sources, n_worst_nodes=50,
                track_nodes=track, verbose=True)
            loop_r = time.perf_counter() - t0
            ref_wf.update({k: np.asarray(v)
                           for k, v in res_r.tracked_waveforms.items()})
            runs[f'tr_{tag}_1e-12_ref'] = {
                'tr_prepare_s': tr_prep,
                'loop_s_per_step': loop_r / n_steps,
                'peak_mV': res_r.peak_ir_drop * 1e3,
            }
            print(f'[{tag}] 1e-12 ref {loop_r/n_steps:.2f} s/step', flush=True)

        for extrap in (False, True):
            if cg is not None:
                cg.rtol = 1e-8
                cg._warm_start_extrapolation = extrap
                cg.reset_warm_start()
            iters_before = cg.total_iterations if cg else 0
            t0 = time.perf_counter()
            res = solver.solve_transient(
                tr_ctx, ic_voltages=dc_ref, t_end=args.t_end,
                smoothed_sources=sources, n_worst_nodes=50,
                track_nodes=track, verbose=True)
            loop = time.perf_counter() - t0
            iters = (cg.total_iterations if cg else 0) - iters_before
            wf_err = _waveform_err(ref_wf, {k: np.asarray(v) for k, v in
                                            res.tracked_waveforms.items()})
            key = f'tr_{tag}_1e-8' + ('_extrap' if extrap else '')
            runs[key] = {
                'tr_prepare_s': tr_prep, 'cg': _cg_info(cg),
                'loop_s_per_step': loop / n_steps,
                'iters_per_step': iters / n_steps,
                'max_dV_V': wf_err,
                'peak_mV': res.peak_ir_drop * 1e3,
                'rss_gb': _rss_gb(),
            }
            print(f'[{tag}] 1e-8 extrap={extrap}: '
                  f'{iters/n_steps:.1f} iters/step, {loop/n_steps:.2f} s/step, '
                  f'max|dV|={wf_err:.3e} V', flush=True)
            if cg is not None:
                cg._warm_start_extrapolation = False
        tr_ctx.release()

    # B: additive + GenEO (production default), with 1e-12 reference
    _td_phase('B_additive_geneo', 'additive', make_reference=True)
    # C: deflated + GenEO
    _td_phase('C_deflated_geneo', 'deflated')
    # D: additive PoU-only (geneo_k=0) — the §7.10 baseline config
    _td_phase('D_additive_pou_only', 'additive', geneo_k=0)

    print('\n=== summary ===')
    for k, v in runs.items():
        print(k, json.dumps(v, default=str))
    if args.json:
        with open(args.json, 'w') as f:
            json.dump(out, f, indent=2, default=str)
        print(f'wrote {args.json}')


if __name__ == '__main__':
    main()

#!/usr/bin/env python
"""Decompose the production CG per-iteration cost on the mi200k proxy.

The Stage 2 measurement observed ~0.85 s/iteration where the Stage 0 kernel
benchmarks predicted ~0.27 s (matvec 150 ms + threaded-BJ ~120 ms).  This
probe rebuilds the production never-assemble DC context and times each
component of one CG iteration separately, on the live production objects:

  1. cg._linear_op.matvec(x)   -- production tilewise matvec (incl. S_extra)
  2. cg._M.matvec(x)           -- production block-Jacobi apply
  3. bench-style serial tilewise loop over the SAME kept blocks (reference)
  4. CG vector-op baseline (axpy/dot on n-length vectors)
  5. STREAM triad (bandwidth context)

Prints a reconciliation against the measured solver s/iter.

Usage:
    venv/bin/python -u scripts/benchmark/microbench/probe_iter_decomposition.py \
        netlist/netlist_brcm_sampled/distributed_pkl_mi200k_v2 [--reps 10]
"""
from __future__ import annotations

import argparse
import json
import logging
import time

import numpy as np

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)s %(levelname)s: %(message)s',
                    datefmt='%H:%M:%S')


def _time_op(fn, reps: int):
    fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return {'mean_ms': 1e3 * float(np.mean(ts)),
            'min_ms': 1e3 * float(np.min(ts))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('pkl_dir')
    ap.add_argument('--reps', type=int, default=10)
    ap.add_argument('--tiles-per-worker', type=int, default=4)
    ap.add_argument('--json', default=None)
    args = ap.parse_args()

    from distributed.model import (
        create_distributed_model, load_distributed_partitions,
    )
    from distributed.solver import DistributedDDMSolver

    bundle = load_distributed_partitions(args.pkl_dir)
    model = create_distributed_model(
        bundle, backend='ray', tiles_per_worker=args.tiles_per_worker)
    model.settings.update({
        'interface_solver': 'cg',
        'interface_matvec_mode': 'auto',
        'interface_preconditioner': 'block_jacobi',
        'interface_cg_rtol': 1e-8,
        'streaming_assembly': True,
        'interface_drop_s_global': True,
    })
    solver = DistributedDDMSolver(model)
    t0 = time.perf_counter()
    ctx = solver.prepare(verbose=True)
    print(f'prepare {time.perf_counter() - t0:.1f}s', flush=True)
    cg = ctx._cg_solver
    assert cg is not None
    n = cg.n_interface if hasattr(cg, 'n_interface') else len(ctx._interface_node_to_idx)

    rng = np.random.default_rng(0)
    x = rng.standard_normal(n)
    results = {'n': int(n),
               'matvec_mode': getattr(cg, 'matvec_mode', None),
               'matvec_threads': getattr(cg, 'matvec_threads', None),
               'preconditioner': getattr(cg, 'preconditioner', None),
               'timings': {}}
    T = results['timings']

    # 1. production matvec
    T['production_matvec'] = _time_op(lambda: cg._linear_op.matvec(x), args.reps)
    print(f"production matvec     : {T['production_matvec']['mean_ms']:8.1f} ms", flush=True)

    # 2. production BJ apply
    if cg._M is not None:
        T['production_bj_apply'] = _time_op(lambda: cg._M.matvec(x), args.reps)
        print(f"production BJ apply   : {T['production_bj_apply']['mean_ms']:8.1f} ms", flush=True)

    # 3. bench-style serial tilewise reference over the same kept blocks
    tiles = [(np.asarray(cg.tile_index_maps[tid]), S)
             for tid, S in cg.tile_schur_complements.items()] \
        if getattr(cg, 'tile_schur_complements', None) else []
    if tiles:
        def serial_mv():
            y = np.zeros(n)
            for idx, S_i in tiles:
                y += np.bincount(idx, weights=S_i @ x[idx], minlength=n)
            return y
        T['serial_tilewise_ref'] = _time_op(serial_mv, max(3, args.reps // 2))
        print(f"serial tilewise ref   : {T['serial_tilewise_ref']['mean_ms']:8.1f} ms", flush=True)
        sizes = sorted((S.shape[0] for _, S in tiles), reverse=True)
        results['block_sizes_top8'] = sizes[:8]
        results['sum_np2'] = int(sum(s * s for s in sizes))

    # 3b. CORRECTNESS diagnostics — distinguish kappa-stagnation from an
    # operator bug (the observed cold-solve stagnation at rel_res ~0.27).
    print('\n=== operator diagnostics ===', flush=True)
    u = rng.standard_normal(n); v = rng.standard_normal(n)
    Au = cg._linear_op.matvec(u); Av = cg._linear_op.matvec(v)
    sym_rel = abs(float(u @ Av) - float(v @ Au)) / max(abs(float(u @ Av)), 1e-300)
    results['symmetry_rel_err'] = sym_rel
    print(f'symmetry <u,Av> vs <v,Au> rel err: {sym_rel:.3e}', flush=True)
    spd_vals = []
    for _ in range(5):
        w = rng.standard_normal(n)
        spd_vals.append(float(w @ cg._linear_op.matvec(w)))
    results['xAx_samples'] = spd_vals
    print(f'x^T A x samples (must all be > 0): {["%.3e" % s for s in spd_vals]}', flush=True)
    if cg._M is not None:
        m_sym = abs(float(u @ cg._M.matvec(v)) - float(v @ cg._M.matvec(u))) \
            / max(abs(float(u @ cg._M.matvec(v))), 1e-300)
        m_spd = [float(w @ cg._M.matvec(w)) for w in
                 (rng.standard_normal(n) for _ in range(3))]
        results['M_symmetry_rel_err'] = m_sym
        results['xMx_samples'] = m_spd
        print(f'M symmetry rel err: {m_sym:.3e}; x^T M x samples: '
              f'{["%.3e" % s for s in m_spd]}', flush=True)
    # Diagonal sampling via e_i probes: taps/die nodes (rows living ONLY in
    # S_extra under never-assemble) vs ordinary boundary rows.
    node_to_idx = ctx._interface_node_to_idx
    tap_nodes = sorted(getattr(model.package_data, 'tap_nodes', set()) or [])
    die_nodes = sorted(model.package_data.die_attachment_nodes)[:5]
    bnd_sample = sorted(bundle.shared_boundary_nodes)[:5]
    diag_report = {}
    for label, nodes in (('tap', tap_nodes[:5]), ('die', die_nodes),
                         ('boundary', bnd_sample)):
        for nd in nodes:
            i = node_to_idx.get(nd)
            if i is None:
                continue
            e = np.zeros(n); e[i] = 1.0
            col = cg._linear_op.matvec(e)
            diag_report[f'{label}:{nd}'] = {
                'diag': float(col[i]),
                'col_norm': float(np.linalg.norm(col)),
                'offdiag_nnz_gt_1e-12': int(np.sum(np.abs(col) > 1e-12) - 1),
            }
    results['row_probes'] = diag_report
    for k2, v2 in diag_report.items():
        print(f'row {k2}: diag={v2["diag"]:.3e} col_norm={v2["col_norm"]:.3e} '
              f'offdiag_nnz={v2["offdiag_nnz_gt_1e-12"]}', flush=True)

    # 4. CG vector-op baseline (~7 n-vector ops per iteration)
    a = rng.standard_normal(n); b = rng.standard_normal(n)
    def vec_ops():
        c = a + 0.5 * b
        d = float(a @ b)
        e = b - 0.5 * a
        f = float(c @ e)
        return d + f
    T['cg_vector_ops_x2'] = _time_op(vec_ops, args.reps)

    # 5. STREAM triad context
    m = 50_000_000
    p = np.zeros(m); q = rng.standard_normal(m); r = rng.standard_normal(m)
    T['stream_triad_1t'] = _time_op(lambda: np.add(q, 2.5 * r, out=p), 5)
    T['stream_triad_1t']['gbps'] = 3 * m * 8 / (T['stream_triad_1t']['min_ms'] / 1e3) / 1e9

    mv = T['production_matvec']['mean_ms']
    bj = T.get('production_bj_apply', {}).get('mean_ms', 0.0)
    vec = T['cg_vector_ops_x2']['mean_ms'] * 2
    print('\n=== reconciliation (per CG iteration) ===')
    print(f'matvec {mv:.0f} ms + BJ {bj:.0f} ms + vec ops ~{vec:.0f} ms '
          f'= {mv + bj + vec:.0f} ms  (solver measured ~850 ms/iter)')
    if args.json:
        with open(args.json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f'wrote {args.json}')
    ctx.release()


if __name__ == '__main__':
    main()

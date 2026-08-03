#!/usr/bin/env python
"""NN/BDD work package: tile-Schur-block spectrum probe (mi200k_v2).

The reg grid measured plain/regularized NN strictly worse than the jacobi
base (282..maxiter cold iters vs 34), implying a BROAD per-block near-null
cluster.  This probe eigendecomposes (values only) a stratified sample of
the gathered DC tile Schur blocks and reports, per block, the count of
eigenvalues below tau * lambda_max for a ladder of tau -- sizing the
spectral (GenEO-style) coarse space that would be needed, or killing that
direction if the band exceeds the T' <= 4096 wall.

Usage:
    venv/bin/python -u scripts/benchmark/microbench/probe_tile_block_spectra_mi200k.py \
        netlist/netlist_brcm_sampled/distributed_pkl_mi200k_v2 \
        [--sample 16] [--json results_tile_block_spectra_mi200k.json]
"""
from __future__ import annotations

import argparse
import json
import time
from typing import Dict, List

import numpy as np
import scipy.linalg as la
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)s %(levelname)s: %(message)s',
                    datefmt='%H:%M:%S')

TAUS = (1e-2, 1e-4, 1e-6, 1e-8, 1e-10)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('pkl_dir')
    ap.add_argument('--tiles-per-worker', type=int, default=4)
    ap.add_argument('--sample', type=int, default=16)
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
        'interface_preconditioner': 'two_level',   # jacobi+PoU default base
        'interface_coarse_geneo_k': 0,
        'streaming_assembly': True,
        'interface_drop_s_global': True,
    })
    solver = DistributedDDMSolver(model)
    t0 = time.perf_counter()
    ctx = solver.prepare(verbose=False)
    print(f'[probe] prepare {time.perf_counter()-t0:.0f}s', flush=True)

    cg = ctx._cg_solver
    blocks = cg.tile_schur_complements
    idx_maps = cg.tile_index_maps
    sizes = sorted(
        ((tid, len(idx_maps[tid])) for tid in blocks), key=lambda kv: -kv[1]
    )
    print(f'[probe] {len(sizes)} blocks; n_p quartiles: '
          f'{np.percentile([s for _, s in sizes], [0, 25, 50, 75, 100])}',
          flush=True)

    # Stratified sample: 4 largest + spread over the rest.
    n_sample = min(args.sample, len(sizes))
    picks = [tid for tid, _ in sizes[:4]]
    rest = [tid for tid, _ in sizes[4:]]
    if rest and n_sample > 4:
        step = max(1, len(rest) // (n_sample - 4))
        picks += rest[::step][:n_sample - 4]

    rows: List[Dict[str, object]] = []
    for tid in picks:
        S_i = np.asarray(blocks[tid], dtype=np.float64)
        k = S_i.shape[0]
        t0 = time.perf_counter()
        w = la.eigvalsh(0.5 * (S_i + S_i.T), check_finite=False)
        dt = time.perf_counter() - t0
        lam_max = float(w[-1])
        rel = w / lam_max
        counts = {f'tau_{tau:g}': int(np.sum(rel < tau)) for tau in TAUS}
        row = {
            'tile': str(tid), 'n_p': k, 'lam_max': lam_max,
            'lam_min_rel': float(rel[0]),
            'smallest20_rel': [float(x) for x in rel[:20]],
            'counts_below': counts, 'eig_s': dt,
        }
        rows.append(row)
        print(f"[probe] tile {tid} n_p={k}: lam_min_rel={rel[0]:.2e} "
              f"counts<tau {counts} ({dt:.0f}s)", flush=True)
        if args.json:
            with open(args.json, 'w') as f:
                json.dump({'pkl_dir': args.pkl_dir, 'rows': rows},
                          f, indent=2, default=str)

    # Extrapolate the total GenEO column count per tau over ALL blocks by
    # scaling each sampled count class to the full population (crude: mean
    # fraction of n_p, applied to total sum n_p).
    total_np = sum(s for _, s in sizes)
    print('\n=== extrapolated total coarse columns (all 64 blocks) ===')
    for tau in TAUS:
        fracs = [r['counts_below'][f'tau_{tau:g}'] / r['n_p'] for r in rows]
        est = int(np.mean(fracs) * total_np)
        print(f'tau={tau:g}: mean frac {np.mean(fracs):.4f} -> ~{est} columns '
              f'(cap 4096)', flush=True)

    ctx.release()
    if args.json:
        print(f'wrote {args.json}')


if __name__ == '__main__':
    main()

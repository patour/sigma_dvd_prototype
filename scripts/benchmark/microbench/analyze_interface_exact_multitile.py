#!/usr/bin/env python
"""netlist_multi_tile exact interface-preconditioner analysis (2026-08-03).

The 9-tile PDN is small enough (interface system n=112) to compute EXACTLY
what the BRCM proxy campaign (docs/brcm_distributed_runtime_optimization.md
Sections 7.16-7.17) could only sample or infer from iteration counts:

  1. Full spectrum of the assembled interface Schur complement S and of the
     diag-scaled system (exact one-level jacobi kappa).
  2. Per-tile-block near-null (tearing) modes: local generalized Rayleigh
     quotient vs the SAME vector embedded in assembled S -- the direct proof
     that the modes are tile-tearing artifacts, healthy in S.
  3. Exact one-level kappa(M^-1 S) for every base in the h2h matrix:
     jacobi, NN/BDD (stiffness weights, reg in {0/eigclip, 1e-3, 1e-5}),
     never-assemble BJ (single-owner S_i slices), true-block BJ (assembled
     principal blocks, same ownership).
  4. Exact DEFLATED effective kappas: spectrum of P_def M^-1 S with the
     production PoU coarse space Z (T'=10) -- the numbers behind the h2h
     iteration counts.
  5. PoU coverage of the embedded tearing modes (why deflation fixes them).

Usage:
    venv/bin/python -u scripts/benchmark/microbench/analyze_interface_exact_multitile.py \
        netlist/netlist_multi_tile/distributed_pkl \
        [--json results_interface_exact_multitile.json]
"""
from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List

import numpy as np
import scipy.linalg as la
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)s %(levelname)s: %(message)s',
                    datefmt='%H:%M:%S')

NN_EIGCLIP_EPS_REL = 1e-10   # mirrors interface_iterative.NEUMANN_EIGCLIP_EPS_REL
REGS = (0.0, 1e-3, 1e-5)


def _sym(M: np.ndarray) -> np.ndarray:
    return 0.5 * (M + M.T)


def _w_half(W: np.ndarray) -> np.ndarray:
    """Symmetric PSD square root via eigh (clips tiny negatives)."""
    lam, U = la.eigh(_sym(W))
    lam = np.clip(lam, 0.0, None)
    return (U * np.sqrt(lam)) @ U.T


def _kappa_psd(W: np.ndarray, A: np.ndarray, zero_tol_rel: float = 1e-10
               ) -> Dict[str, Any]:
    """Spectrum of M^-1 S computed as eigvalsh(W^1/2 A W^1/2), W = M^-1 PSD.

    Returns kappa over the numerically-nonzero part plus the zero-mode count
    (nonzero only for the eigclip'd reg=0 NN, whose W annihilates the
    per-tile constant directions).
    """
    B = _sym(_w_half(W) @ A @ _w_half(W).T)
    w = la.eigvalsh(B)
    lam_max = float(w[-1])
    nz = w[w > zero_tol_rel * lam_max]
    return {
        'lam_min': float(nz[0]), 'lam_max': lam_max,
        'kappa': float(lam_max / nz[0]), 'n_zero_modes': int(len(w) - len(nz)),
    }


def _deflated_kappa(W: np.ndarray, A: np.ndarray, Z: np.ndarray
                    ) -> Dict[str, Any]:
    """Effective spectrum of the DEF-deflated operator P_def M^-1 S.

    P_def = I - Z (Z^T A Z)^-1 Z^T A (A-orthogonal projector onto the
    complement of range(Z)); spectrum has T' zeros plus the effective
    (deflated) eigenvalues that govern PCG convergence.
    """
    n, t = Z.shape
    Sc = _sym(Z.T @ A @ Z)
    P = np.eye(n) - Z @ la.solve(Sc, Z.T @ A, assume_a='pos')
    lam = la.eig(P @ (W @ A))[0]
    lam = np.real(lam[np.abs(np.imag(lam)) < 1e-8 * np.max(np.abs(lam))])
    lam = np.sort(lam)
    lam_max = float(lam[-1])
    nz = lam[lam > 1e-8 * lam_max]
    return {
        'lam_min': float(nz[0]), 'lam_max': lam_max,
        'kappa_eff': float(lam_max / nz[0]),
        'n_deflated': int(len(lam) - len(nz)), 't_cols': int(t),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('pkl_dir')
    ap.add_argument('--json', default=None)
    args = ap.parse_args()

    from distributed.model import (
        create_distributed_model, load_distributed_partitions,
    )
    from distributed.solver import DistributedDDMSolver

    bundle = load_distributed_partitions(args.pkl_dir)
    model = create_distributed_model(bundle, backend='local')
    model.settings.update({
        'interface_solver': 'cg',
        'interface_matvec_mode': 'tilewise',
        'interface_preconditioner': 'two_level',
        'interface_two_level_base': 'jacobi',
        'interface_coarse_geneo_k': 0,
        # Keep S_global: the whole point is exact assembled-vs-torn algebra.
        'interface_drop_s_global': False,
        'streaming_assembly': False,
    })
    solver = DistributedDDMSolver(model)
    ctx = solver.prepare(verbose=False)
    cg = ctx._cg_solver
    A = _sym(np.asarray(ctx._S_global.todense(), dtype=np.float64))
    n = A.shape[0]
    blocks = {tid: np.asarray(S_i, dtype=np.float64)
              for tid, S_i in cg.tile_schur_complements.items()}
    idx_maps = {tid: np.asarray(idx, dtype=np.int64)
                for tid, idx in cg.tile_index_maps.items()}
    Z = cg._coarse.Z
    Z = np.asarray(Z.todense() if hasattr(Z, 'todense') else Z,
                   dtype=np.float64)
    out: Dict[str, Any] = {'pkl_dir': args.pkl_dir, 'n': n,
                           'n_tiles': len(blocks), 't_cols': Z.shape[1]}

    # ---- 0. multiplicity ------------------------------------------------
    mult = np.zeros(n, dtype=np.int64)
    for idx in idx_maps.values():
        mult[idx] += 1
    covered = mult > 0
    out['multiplicity'] = {
        'hist': {int(m): int(np.sum(mult == m)) for m in np.unique(mult)},
        'mean_over_covered': float(mult[covered].mean()),
        'sum_np': int(mult.sum()),
    }
    print(f'[exact] n={n}, T={len(blocks)} tiles, T\'={Z.shape[1]} PoU cols, '
          f'multiplicity hist={out["multiplicity"]["hist"]} '
          f'(0 = package-only rows)', flush=True)

    # ---- 1. raw + jacobi ------------------------------------------------
    eigs_A = la.eigvalsh(A)
    d = np.diag(A).copy()
    Dm = 1.0 / np.sqrt(d)
    eigs_jac = la.eigvalsh(_sym(Dm[:, None] * A * Dm[None, :]))
    out['raw'] = {'kappa': float(eigs_A[-1] / eigs_A[0]),
                  'lam_min': float(eigs_A[0]), 'lam_max': float(eigs_A[-1])}
    out['jacobi_one_level'] = {'kappa': float(eigs_jac[-1] / eigs_jac[0]),
                               'lam_min': float(eigs_jac[0]),
                               'lam_max': float(eigs_jac[-1])}
    print(f'[exact] kappa(S)={out["raw"]["kappa"]:.3e}  '
          f'kappa(D^-1S)={out["jacobi_one_level"]["kappa"]:.3e}', flush=True)

    # ---- 2. tearing modes: local vs embedded Rayleigh + PoU coverage ----
    # Orthonormal basis of range(Z) for coverage numbers.
    Qz = la.qr(Z, mode='economic')[0]
    rows: List[Dict[str, Any]] = []
    for tid, S_i in blocks.items():
        idx = idx_maps[tid]
        lam, V = la.eigh(_sym(S_i))
        v = V[:, 0]
        w = np.zeros(n)
        w[idx] = v
        # Generalized Rayleigh vs the diagonal (what jacobi-PCG sees):
        # torn-block value ~1e-16 (floating mode), embedded-in-S value O(1).
        ray_local = float(v @ S_i @ v) / float(v @ (np.diag(S_i) * v))
        ray_global = float(w @ A @ w) / float(w @ (d * w))
        cov = float(np.linalg.norm(Qz.T @ w) / np.linalg.norm(w))
        rows.append({'tile': str(tid), 'n_p': int(S_i.shape[0]),
                     'lam_min_rel': float(lam[0] / lam[-1]),
                     'rayleigh_local': ray_local,
                     'rayleigh_embedded': ray_global,
                     'pou_coverage': cov})
        print(f'[tearing] tile {tid} n_p={S_i.shape[0]:3d}: '
              f'local R={ray_local:+.2e}  embedded R={ray_global:.3f}  '
              f'||P_Z w||/||w||={cov:.4f}', flush=True)
    out['tearing_modes'] = rows

    # ---- 3. one-level + deflated kappas per base ------------------------
    # Stiffness weights (Mandel-Brezina): w_i(g) = diag(S_i)/sum_tiles diag.
    tot_diag = np.zeros(n)
    for tid, S_i in blocks.items():
        tot_diag[idx_maps[tid]] += np.diag(S_i)
    bases: Dict[str, np.ndarray] = {}

    for reg in REGS:
        W = np.zeros((n, n))
        n_clipped = 0
        for tid, S_i in blocks.items():
            idx = idx_maps[tid]
            w_loc = np.diag(S_i) / tot_diag[idx]
            sub = S_i.copy()
            if reg > 0.0:
                sub[np.diag_indices_from(sub)] += reg * np.diag(S_i)
                Sinv = la.inv(_sym(sub))
            else:
                lam, U = la.eigh(_sym(sub))
                keep = lam > NN_EIGCLIP_EPS_REL * lam[-1]
                n_clipped += int(np.sum(~keep))
                Sinv = (U[:, keep] / lam[keep]) @ U[:, keep].T
            Sinv = Sinv * w_loc[np.newaxis, :] * w_loc[:, np.newaxis]
            W[np.ix_(idx, idx)] += Sinv
        W[~covered, ~covered] += 1.0 / d[~covered]
        bases[f'neumann_reg{reg:g}'] = W
        if reg == 0.0:
            out['nn_reg0_clipped_dirs'] = n_clipped

    # Ownership: first-seen tile in tile_index_maps iteration order --
    # replicates interface_iterative._build_block_jacobi exactly.
    owner = -np.ones(n, dtype=np.int64)
    tile_order = list(idx_maps)
    for k, tid in enumerate(tile_order):
        idx = idx_maps[tid]
        unset = idx[owner[idx] < 0]
        owner[unset] = k
    W_bj = np.zeros((n, n))
    W_true = np.zeros((n, n))
    for k, tid in enumerate(tile_order):
        idx = idx_maps[tid]
        own_g = idx[owner[idx] == k]
        if own_g.size == 0:
            continue
        pos = {int(g): j for j, g in enumerate(idx)}
        loc = np.array([pos[int(g)] for g in own_g], dtype=np.int64)
        for W_dst, src in ((W_bj, blocks[tid][np.ix_(loc, loc)]),
                           (W_true, A[np.ix_(own_g, own_g)])):
            sub = _sym(np.asarray(src))
            lam, U = la.eigh(sub)
            keep = lam > NN_EIGCLIP_EPS_REL * lam[-1]
            W_dst[np.ix_(own_g, own_g)] = (U[:, keep] / lam[keep]) @ U[:, keep].T
    W_bj[~covered, ~covered] += 1.0 / d[~covered]
    W_true[~covered, ~covered] += 1.0 / d[~covered]
    bases['bj_never_assemble'] = W_bj
    bases['bj_true_block'] = W_true
    bases['jacobi'] = np.diag(1.0 / d)

    out['bases'] = {}
    print(f'\n[kappa] {"base":<22}{"one-level":>14}{"zero":>6}'
          f'{"deflated(PoU)":>16}{"defl":>6}', flush=True)
    for name, W in bases.items():
        one = _kappa_psd(W, A)
        two = _deflated_kappa(W, A, Z)
        out['bases'][name] = {'one_level': one, 'deflated': two}
        print(f'[kappa] {name:<22}{one["kappa"]:>14.3e}'
              f'{one["n_zero_modes"]:>6d}{two["kappa_eff"]:>16.3e}'
              f'{two["n_deflated"]:>6d}', flush=True)

    ctx.release()
    if args.json:
        with open(args.json, 'w') as f:
            json.dump(out, f, indent=2, default=str)
        print(f'wrote {args.json}')


if __name__ == '__main__':
    main()

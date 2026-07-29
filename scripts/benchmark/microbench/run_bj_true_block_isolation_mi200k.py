#!/usr/bin/env python
"""Isolate BLOCK CONSTRUCTION as the root cause of block-Jacobi CG stagnation.

Context (docs/brcm_distributed_runtime_optimization.md §7.8/§7.9/§7.13): every
recorded block-Jacobi stagnation at the split regime ran the NEVER-ASSEMBLE
path, where `_form_owned_block` builds each ownership block from the single
OWNER tile's dense Schur block `S_i` (+ S_extra) -- path 2 of
`_build_block_jacobi`'s docstring.  For a boundary node shared with a
neighbor tile, that block is missing the neighbor's stiffness contribution
entirely.  The assembled path (path 1, true principal submatrices of
S_global) was never measured at the split regime on the proxy; the closest
datum is real-BRCM 107-tile assembled BJ converging slowly (§7.4).

This script isolates the variable SURGICALLY: it monkeypatches
`InterfaceCGSolver._form_owned_block` to return the TRUE principal submatrix

    S[O_i, O_i] = sum_t (P_t S_t P_t^T)[O_i, O_i] + S_extra[O_i, O_i]

accumulated over ALL tiles (not just the owner), leaving ownership
assignment, factoring, apply, matvec (tilewise, never-assemble), coarse
space, and memory profile byte-identical.  No S_global is ever assembled.

Cells (all: cold DC interface solve, rtol 1e-8, bounded maxiter):

  0 control_jacobi_pou       auto BJ budget -> guard downgrades to jacobi;
                             two_level[deflated](jacobi+PoU).  Harness anchor:
                             must reproduce §7.13 variant A (34 iters, ~10 s).
  1 path2_plain_bj           plain block_jacobi, single-owner-tile blocks,
                             16 GiB budget.  Expect stagnation (§7.8).
  2 true_plain_bj            plain block_jacobi, TRUE blocks (patched).
                             THE experiment cell.
  3 true_twolevel_deflated   two_level[deflated](bj_true+PoU), 16 GiB.
                             The §7.13 variant-B twin with true blocks --
                             variant B stagnated at rel-res ~1e-5/1500 iters.

Results JSON is (re)written after every cell so a watchdog kill preserves
completed cells.

Usage:
    venv/bin/python -u scripts/benchmark/microbench/run_bj_true_block_isolation_mi200k.py \
        netlist/netlist_brcm_sampled/distributed_pkl_mi200k_v2 \
        --json scripts/benchmark/microbench/results_bj_true_block_isolation_mi200k.json

    # quick correctness check of the true-block builder (no netlist needed):
    venv/bin/python scripts/benchmark/microbench/run_bj_true_block_isolation_mi200k.py --self-test
"""
from __future__ import annotations

import argparse
import json
import logging
import time
import traceback
from typing import Any, Dict, List, Optional

import numpy as np
import scipy.sparse as sp

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)s %(levelname)s: %(message)s',
                    datefmt='%H:%M:%S')

GIB = 1024 ** 3


# ---------------------------------------------------------------------------
# True-block builder (pure function; self-tested below)
# ---------------------------------------------------------------------------

def build_true_block(
    tile_index_maps: Dict[Any, np.ndarray],
    tile_schur_complements: Dict[Any, np.ndarray],
    owned: np.ndarray,
    S_extra_csr: Optional[sp.spmatrix],
) -> np.ndarray:
    """True principal submatrix S[owned, owned] of the never-assembled S.

    S = sum_t P_t S_t P_t^T + S_extra, so S[O,O] accumulates, for every tile
    t, S_t restricted to (O intersect ports_t) -- NOT just the owner tile's
    slice.  ``owned`` must be sorted ascending (guaranteed by
    ``_build_block_jacobi``'s ``np.array(sorted(owned_global))``).
    Mirrors path 2's 1e-12 diagonal jitter so factoring sees the same
    regularization.
    """
    owned = np.asarray(owned, dtype=np.int64)
    k = owned.shape[0]
    sub = np.zeros((k, k), dtype=np.float64)
    for tid, idx_full in tile_index_maps.items():
        idx_full = np.asarray(idx_full, dtype=np.int64)
        pos = np.searchsorted(owned, idx_full)
        pos_c = np.minimum(pos, k - 1)
        member = owned[pos_c] == idx_full
        if not member.any():
            continue
        loc = np.nonzero(member)[0]          # tile-local port positions
        ow = pos_c[member]                    # positions within the owned block
        S_i = np.asarray(tile_schur_complements[tid], dtype=np.float64)
        sub[np.ix_(ow, ow)] += S_i[np.ix_(loc, loc)]
    if S_extra_csr is not None:
        sub += S_extra_csr[np.ix_(owned, owned)].toarray()
    sub += 1e-12 * np.eye(k)
    return sub


def self_test() -> None:
    """Verify build_true_block against explicit dense assembly (3 tiles)."""
    rng = np.random.default_rng(0)
    n = 12
    maps = {
        't0': np.array([0, 1, 2, 3, 4, 5]),
        't1': np.array([4, 5, 6, 7, 8]),
        't2': np.array([2, 3, 8, 9, 10, 11]),
    }
    tiles = {}
    for tid, idx in maps.items():
        A = rng.standard_normal((len(idx), len(idx)))
        tiles[tid] = A @ A.T
    S_extra = sp.random(n, n, density=0.3, random_state=1)
    S_extra = (S_extra + S_extra.T).tocsr()

    S_dense = np.zeros((n, n))
    for tid, idx in maps.items():
        S_dense[np.ix_(idx, idx)] += tiles[tid]
    S_dense += S_extra.toarray()

    for owned_set in ([1, 3, 4, 8, 9], [0], [6, 7], list(range(n))):
        owned = np.array(sorted(owned_set), dtype=np.int32)
        ref = S_dense[np.ix_(owned, owned)] + 1e-12 * np.eye(len(owned))
        got = build_true_block(maps, tiles, owned, S_extra)
        assert np.allclose(got, ref, rtol=1e-12, atol=1e-12), owned_set
        # and it must differ from the single-owner path-2 block whenever the
        # owned set touches a shared node (sanity that the treatment is real)
    print('self-test: build_true_block matches explicit dense assembly -- OK')


# ---------------------------------------------------------------------------
# Patch machinery
# ---------------------------------------------------------------------------

_PATCH_STATS: List[Dict[str, Any]] = []

# Contexts not yet released -- drained by main() on unexpected cell failure
# so a mid-cell exception does not leak a prepared context (worker factors +
# coordinator blocks) into subsequent cells' memory budget.
_LIVE_CTX: List[Any] = []


def _make_patched_form(orig_form):
    def _patched(self, tile_or_node_group, owned_global_arr, S_csr,
                 S_extra_csr, use_s_global):
        if use_s_global:  # not our regime; defer to the original
            return orig_form(self, tile_or_node_group, owned_global_arr,
                             S_csr, S_extra_csr, use_s_global)
        t0 = time.perf_counter()
        sub_true = build_true_block(
            self.tile_index_maps, self.tile_schur_complements,
            owned_global_arr, S_extra_csr)
        # path-2 block for the record: how much stiffness was missing?
        sub_p2 = orig_form(self, tile_or_node_group, owned_global_arr,
                           S_csr, S_extra_csr, use_s_global)
        d_true = np.diag(sub_true).copy()
        d_p2 = np.diag(sub_p2).copy()
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.where(d_p2 > 0, d_true / d_p2, np.inf)
        fro_p2 = float(np.linalg.norm(sub_p2))
        _PATCH_STATS.append({
            'tile': str(tile_or_node_group),
            'k': int(owned_global_arr.shape[0]),
            'rel_fro_added': float(np.linalg.norm(sub_true - sub_p2))
                             / fro_p2 if fro_p2 > 0 else None,
            'diag_ratio_min': float(np.min(ratio)),
            'diag_ratio_median': float(np.median(ratio)),
            'diag_ratio_max': float(np.max(ratio)) if np.isfinite(ratio).all()
                              else 'inf',
            'build_s': time.perf_counter() - t0,
        })
        return sub_true
    return _patched


# ---------------------------------------------------------------------------
# Cell runner
# ---------------------------------------------------------------------------

def _rss_gb() -> float:
    import psutil
    return psutil.Process().memory_info().rss / 1e9


def _sys_used_gb() -> float:
    with open('/proc/meminfo') as fh:
        mem = {l.split(':')[0]: int(l.split()[1]) for l in fh}
    return (mem['MemTotal'] - mem['MemAvailable']) / 1048576.0


def _precond_info(cg) -> Dict[str, Any]:
    if cg is None:
        return {}
    lab = getattr(cg, 'preconditioner_label', None)
    info: Dict[str, Any] = {
        'label': lab() if callable(lab) else lab,
        'preconditioner': getattr(cg, 'preconditioner', None),
        'requested': getattr(cg, 'requested_preconditioner', None),
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


def run_cell(name: str, model, solver, args, *, preconditioner: str,
             bj_max_bytes, patched: bool, apply_mode: str = 'deflated',
             reproject_every: Optional[int] = None,
             instrument: bool = False) -> Dict[str, Any]:
    import distributed.interface_iterative as ii

    print(f'\n{"=" * 72}\n[{name}] preconditioner={preconditioner} '
          f'bj_max_bytes={bj_max_bytes} true_blocks={patched} '
          f'apply_mode={apply_mode} reproject_every={reproject_every}'
          f'\n{"=" * 72}', flush=True)
    rec: Dict[str, Any] = {
        'cell': name, 'preconditioner': preconditioner,
        'bj_max_bytes': str(bj_max_bytes), 'true_blocks': patched,
        'apply_mode': apply_mode, 'reproject_every': reproject_every,
    }
    model.settings['interface_preconditioner'] = preconditioner
    model.settings['interface_block_jacobi_max_bytes'] = bj_max_bytes
    model.settings['interface_coarse_apply_mode'] = apply_mode
    model.settings['interface_deflated_reproject_every'] = (
        50 if reproject_every is None else int(reproject_every))

    _PATCH_STATS.clear()
    orig = ii.InterfaceCGSolver._form_owned_block
    if patched:
        ii.InterfaceCGSolver._form_owned_block = _make_patched_form(orig)
    try:
        t0 = time.perf_counter()
        dc_ctx = solver.prepare(verbose=True)
        rec['dc_prepare_s'] = time.perf_counter() - t0
        _LIVE_CTX.append(dc_ctx)
    finally:
        ii.InterfaceCGSolver._form_owned_block = orig

    if patched:
        ks = np.array([s['k'] for s in _PATCH_STATS])
        rec['patch'] = {
            'n_blocks': len(_PATCH_STATS),
            'sum_k': int(ks.sum()) if len(ks) else 0,
            'max_k': int(ks.max()) if len(ks) else 0,
            'total_build_s': float(sum(s['build_s'] for s in _PATCH_STATS)),
            'rel_fro_added_max': max((s['rel_fro_added'] or 0)
                                     for s in _PATCH_STATS) if _PATCH_STATS else None,
            'diag_ratio_min_overall': min(s['diag_ratio_min']
                                          for s in _PATCH_STATS) if _PATCH_STATS else None,
            'per_block': _PATCH_STATS[:],
        }
        print(f'[{name}] patch: {len(_PATCH_STATS)} true blocks built, '
              f'sum_k={rec["patch"]["sum_k"]}, '
              f'build={rec["patch"]["total_build_s"]:.1f}s, '
              f'max rel added mass={rec["patch"]["rel_fro_added_max"]:.3g}',
              flush=True)

    cg = dc_ctx._cg_solver
    rec['dc_precond'] = _precond_info(cg)
    rec['dc_prepare_rss_gb'] = _rss_gb()
    rec['dc_prepare_sys_used_gb'] = _sys_used_gb()
    print(f'[{name}] DC prepare {rec["dc_prepare_s"]:.1f}s '
          f'rss={rec["dc_prepare_rss_gb"]:.1f}GB '
          f'sys_used={rec["dc_prepare_sys_used_gb"]:.1f}GB '
          f'label={rec["dc_precond"].get("label")}', flush=True)
    if cg is not None:
        cg.progress_every = args.progress_every

    # --- instrumentation (deflated path only; instance-level wrappers) ----
    # _M_base_apply is called exactly once per CG iteration with the TRACKED
    # residual as its argument -> its norms are the tracked-residual
    # trajectory.  coarse.apply is called by _recover_x (per progress-logged
    # iteration) and by _try_accept (acceptance attempts) -> its call
    # positions mark acceptance activity.  The matvec shim counts full-n
    # matvecs (each acceptance attempt pays two extras).
    real_op = cg._linear_op if cg is not None else None
    inst: Optional[Dict[str, Any]] = None
    if instrument and cg is not None and getattr(cg, '_coarse', None) is not None:
        import types
        # CoarseSpace has __slots__, so coarse.apply cannot be wrapped.
        # Acceptance attempts are instead detected from matvec bursts:
        # normal pattern is 1 matvec/iteration (S p) plus 1 extra at each
        # progress-logged iteration (the callback's true-residual check goes
        # through this same shim); each _try_accept adds 2 extras at one
        # iteration index -> matvec_at_iter positions with >= 2 surplus
        # calls mark acceptance attempts.
        inst = {'tracked_rnorm': [], 'matvec_at_iter': [], 'n_matvec': 0}
        _orig_base = cg._M_base_apply

        def _base_wrap(r, _orig=_orig_base, _i=inst):
            _i['tracked_rnorm'].append(float(np.linalg.norm(r)))
            return _orig(r)

        cg._M_base_apply = _base_wrap

        def _mv_wrap(x, _op=real_op, _i=inst):
            _i['n_matvec'] += 1
            _i['matvec_at_iter'].append(len(_i['tracked_rnorm']))
            return _op.matvec(x)

        cg._linear_op = types.SimpleNamespace(matvec=_mv_wrap)

    # --- final fresh true-residual check (ALL cells; H-C criterion probe) --
    _final: Dict[str, float] = {}
    _orig_lu = dc_ctx.interface_lu
    _true_mv = real_op.matvec if real_op is not None else None

    def _lu_wrap(rhs, _orig=_orig_lu, _cap=_final, _mv=_true_mv):
        x = _orig(rhs)
        if _mv is not None:
            bn = float(np.linalg.norm(rhs))
            rn = float(np.linalg.norm(rhs - _mv(x)))
            _cap['rhs_norm'] = bn
            _cap['final_true_rel_res'] = rn / bn if bn else rn
        return x

    dc_ctx.interface_lu = _lu_wrap

    print(f'[{name}] COLD DC solve @{args.rtol:.0e} '
          f'(maxiter={getattr(cg, "maxiter", None)}) starting...', flush=True)
    t0 = time.perf_counter()
    try:
        solver.solve_dc(dc_ctx)
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
        rec['apply_algorithm'] = cg.stats.get('apply_algorithm')
    rec['peak_rss_gb'] = _rss_gb()
    rec['sys_used_gb'] = _sys_used_gb()
    rec.update(_final)  # rhs_norm + final_true_rel_res when solve succeeded

    if inst is not None:
        tr = np.array(inst['tracked_rnorm'])
        bn = _final.get('rhs_norm')
        atol_eff = max(1e-14, args.rtol * bn) if bn else None
        first_below = (int(np.argmax(tr <= atol_eff)) + 1
                       if atol_eff is not None and np.any(tr <= atol_eff)
                       else None)
        n_iters = len(tr)
        # Acceptance attempts from matvec bursts (see wrapper comment):
        # baseline 1 matvec/iter (recorded at index k for iteration k-1's
        # Sp), +1 at progress-logged indices; a surplus >= 2 at one index
        # is a _try_accept event (fresh Sy + candidate residual check).
        mv_at = np.array(inst['matvec_at_iter'], dtype=np.int64)
        counts = np.bincount(mv_at, minlength=n_iters + 2)
        expected = np.ones_like(counts)
        expected[0] = 0
        if args.progress_every > 0:
            logged = np.arange(0, len(counts), args.progress_every)
            expected[logged[logged > 0]] += 1
        surplus = counts - expected
        attempt_iters = np.nonzero(surplus >= 2)[0].tolist()
        rec['instrument'] = {
            'n_base_applies': n_iters,
            'n_matvec': inst['n_matvec'],
            'first_tracked_below_tol_iter': first_below,
            'accept_attempt_iters': attempt_iters,
            # tracked rel-res trajectory, subsampled (every 10) + last 20
            'tracked_rel_res_every10': [
                float(v / bn) for v in tr[::10]] if bn else None,
            'tracked_rel_res_last20': [
                float(v / bn) for v in tr[-20:]] if bn else None,
        }
        print(f'[{name}] instrument: iters={n_iters} matvecs={inst["n_matvec"]} '
              f'first_tracked_below_tol={first_below} '
              f'accept_attempts@{attempt_iters}', flush=True)

    print(f'[{name}] COLD DC: converged={rec["dc_converged"]} '
          f'iters={rec.get("dc_iters")} {rec["dc_solve_s"]:.1f}s '
          f'rel_res={rec.get("dc_rel_residual")} '
          f'final_true_rel_res={rec.get("final_true_rel_res")}', flush=True)

    dc_ctx.release()
    if dc_ctx in _LIVE_CTX:
        _LIVE_CTX.remove(dc_ctx)
    del dc_ctx
    return rec


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('pkl_dir', nargs='?')
    ap.add_argument('--self-test', action='store_true')
    ap.add_argument('--tiles-per-worker', type=int, default=4)
    ap.add_argument('--rtol', type=float, default=1e-8)
    ap.add_argument('--maxiter', type=int, default=1500)
    ap.add_argument('--progress-every', type=int, default=25)
    ap.add_argument('--bj-max-bytes', type=float, default=16 * GIB)
    ap.add_argument('--cells', default='all',
                    help='comma list from: control_jacobi_pou,path2_plain_bj,'
                         'true_plain_bj,true_twolevel_deflated')
    ap.add_argument('--json', default=None)
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return
    if not args.pkl_dir:
        ap.error('pkl_dir required unless --self-test')

    from distributed.model import (
        create_distributed_model, load_distributed_partitions,
    )
    from distributed.solver import DistributedDDMSolver

    out: Dict[str, Any] = {
        'pkl_dir': args.pkl_dir, 'rtol': args.rtol, 'maxiter': args.maxiter,
        'question': 'Is BLOCK CONSTRUCTION (single-owner-tile S_i slice vs '
                    'true principal submatrix of S) the root cause of '
                    'block-Jacobi CG stagnation at the split regime?',
        'cells': [],
    }
    cells_out: List[Dict[str, Any]] = out['cells']  # type: ignore

    t_all = time.perf_counter()
    bundle = load_distributed_partitions(args.pkl_dir)
    model = create_distributed_model(
        bundle, backend='ray', tiles_per_worker=args.tiles_per_worker)
    model.settings.update({
        'interface_solver': 'cg',
        'interface_matvec_mode': 'auto',          # -> tilewise
        'interface_coarse_apply_mode': 'deflated',
        'interface_coarse_geneo_k': 0,
        'interface_cg_rtol': args.rtol,
        'interface_cg_maxiter': args.maxiter,
        'streaming_assembly': True,
        'interface_drop_s_global': True,          # never-assemble throughout
        'interface_warm_start_extrapolation': True,
    })
    solver = DistributedDDMSolver(model)

    bj16 = int(args.bj_max_bytes)
    plan = [
        # --- phase 1: block-construction isolation (2026-07-27 run) -------
        ('control_jacobi_pou', dict(preconditioner='two_level',
                                    bj_max_bytes='auto', patched=False)),
        ('path2_plain_bj', dict(preconditioner='block_jacobi',
                                bj_max_bytes=bj16, patched=False)),
        ('true_plain_bj', dict(preconditioner='block_jacobi',
                               bj_max_bytes=bj16, patched=True)),
        ('true_twolevel_deflated', dict(preconditioner='two_level',
                                        bj_max_bytes=bj16, patched=True)),
        # --- phase 2: why does two_level make true-BJ worse (262 -> 311)? -
        # H-C probe: plain BJ rerun, now with the fresh final-true-residual
        # check (was the scipy tracked-residual stop a discount?)
        ('true_plain_bj_check', dict(preconditioner='block_jacobi',
                                     bj_max_bytes=bj16, patched=True)),
        # instrumented reproduction of the 311-iter cell (H-A/H-R traces)
        ('true_deflated_r50', dict(preconditioner='two_level',
                                   bj_max_bytes=bj16, patched=True,
                                   instrument=True)),
        # H-R dose-response: no reprojection / aggressive reprojection
        ('true_deflated_r0', dict(preconditioner='two_level',
                                  bj_max_bytes=bj16, patched=True,
                                  reproject_every=0, instrument=True)),
        ('true_deflated_r10', dict(preconditioner='two_level',
                                   bj_max_bytes=bj16, patched=True,
                                   reproject_every=10, instrument=True)),
        # H-S probe: additive apply (scipy loop, no projection machinery)
        ('true_additive', dict(preconditioner='two_level',
                               bj_max_bytes=bj16, patched=True,
                               apply_mode='additive')),
    ]
    wanted = (None if args.cells == 'all'
              else {c.strip() for c in args.cells.split(',')})

    def _flush() -> None:
        out['end_to_end_s'] = time.perf_counter() - t_all
        if args.json:
            with open(args.json, 'w') as fh:
                json.dump(out, fh, indent=2, default=str)

    for name, kw in plan:
        if wanted is not None and name not in wanted:
            continue
        try:
            cells_out.append(run_cell(name, model, solver, args, **kw))
        except Exception as exc:  # noqa: BLE001 - record and continue
            print(f'[{name}] UNEXPECTED FAILURE: {exc}', flush=True)
            traceback.print_exc()
            cells_out.append({'cell': name, 'unexpected_error': str(exc),
                              'traceback': traceback.format_exc()})
            while _LIVE_CTX:
                try:
                    _LIVE_CTX.pop().release()
                except Exception:
                    pass
        _flush()

    print('\n' + '=' * 72)
    print('=== SUMMARY: cold DC interface solve @rtol %.0e ===' % args.rtol)
    print('=' * 72)
    for c in cells_out:
        print(f"{c['cell']:<26} label={c.get('dc_precond', {}).get('label')}")
        print(f"{'':<26} converged={c.get('dc_converged')} "
              f"iters={c.get('dc_iters')} t={c.get('dc_solve_s', 0):.1f}s "
              f"rel_res={c.get('dc_rel_residual')}")
    _flush()
    if args.json:
        print(f'\nWrote {args.json}')


if __name__ == '__main__':
    main()

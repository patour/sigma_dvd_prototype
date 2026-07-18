#!/usr/bin/env python
"""Interface matvec microbenchmark (Stage 0, interface-solve plan).

Loads a distributed pkl bundle and gathers the per-tile dense Schur blocks
S_i directly from the tile workers (``factor_and_compute_schur``), bypassing
the coordinator's S_global COO assembly entirely — the production non-
streaming assembly needs >190 GB of driver RAM at the 64-tile/167K-interface
split regime (measured 2026-07-17; the naive gather+COO path was watchdog-
killed at 51 GB available), which is itself a Stage 0 finding feeding the
Stage 2 S_extra/streaming redesign.

Timed kernels:

  1. synthetic CSR SpMV, 1 thread   (proxy for the assembled-mode matvec;
                                     bandwidth-bound, so a synthetic pattern
                                     with matching n/nnz is representative)
  2. serial tilewise matvec         (current matvec_mode='tilewise' math)
  3. threaded tilewise matvec       (Stage 2 design: LPT partition, per-thread
                                     accumulators, GEMV releases the GIL)
  4. threaded tilewise fp32         (storage fp32, accumulate fp64)
  5. ownership-block-Jacobi apply   (approximates the production BJ cost:
                                     cho_solve over per-tile owned blocks)
  6. threaded matmat, T+1 cols      (Stage 3 coarse-setup S·Z probe)
  7. STREAM-style triad             (memory-bandwidth ceiling reference)

Pad ports (Dirichlet nodes absent from the interface set) are dropped from
each S_i via np.ix_ kept-slicing — the D1-correct convention.

BLAS threads are pinned to 1 (set before numpy import) so the thread pool
is the only parallelism — matching the Stage 2 production design.

Usage:
    venv/bin/python -u scripts/benchmark/microbench/bench_interface_matvec.py \
        netlist/netlist_brcm_sampled/distributed_pkl_mi200k \
        [--tiles-per-worker 4] [--threads 8,16,32,48] [--reps 10] [--json out.json]
"""
from __future__ import annotations

import os

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple

import numpy as np

_PHASE = {'name': 'init'}


def _set_phase(name: str) -> None:
    _PHASE['name'] = name
    print(f'[phase] {name}', flush=True)


def _mem_logger(interval_s: float = 15.0) -> None:
    """Daemon thread: log driver RSS + system available every interval."""
    import psutil
    proc = psutil.Process()
    while True:
        rss = proc.memory_info().rss / 1e9
        avail = psutil.virtual_memory().available / 1e9
        print(f"[mem] phase={_PHASE['name']} rss={rss:.1f}GB "
              f"avail={avail:.1f}GB", flush=True)
        time.sleep(interval_s)


def _time_op(fn, reps: int) -> Dict[str, float]:
    fn()  # warmup
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return {
        'mean_ms': 1e3 * float(np.mean(times)),
        'min_ms': 1e3 * float(np.min(times)),
        'max_ms': 1e3 * float(np.max(times)),
    }


def _lpt_partition(costs: List[float], n_bins: int) -> List[List[int]]:
    """Longest-processing-time greedy: returns per-bin lists of item indices."""
    order = np.argsort(costs)[::-1]
    bins: List[List[int]] = [[] for _ in range(n_bins)]
    loads = np.zeros(n_bins)
    for i in order:
        b = int(np.argmin(loads))
        bins[b].append(int(i))
        loads[b] += costs[i]
    return bins


def build_threaded_matvec(tiles: List[Tuple[np.ndarray, np.ndarray]],
                          n: int, n_threads: int, pool: ThreadPoolExecutor):
    """Stage 2 prototype: LPT partition + per-thread accumulator rows."""
    costs = [float(S.shape[0]) ** 2 for _, S in tiles]
    part = _lpt_partition(costs, n_threads)
    acc = np.zeros((n_threads, n), dtype=np.float64)

    def matvec(x: np.ndarray) -> np.ndarray:
        acc.fill(0.0)

        def work(t: int) -> None:
            row = acc[t]
            for i in part[t]:
                idx, S_i = tiles[i]
                y_local = S_i @ x[idx]
                row += np.bincount(idx, weights=y_local, minlength=n)

        list(pool.map(work, range(n_threads)))
        return acc.sum(axis=0)

    return matvec


def serial_tilewise_matvec(tiles, n):
    """Reference: production-style serial tilewise loop (gather/GEMV/scatter)."""
    def matvec(x: np.ndarray) -> np.ndarray:
        y = np.zeros(n)
        for idx, S_i in tiles:
            y += np.bincount(idx, weights=S_i @ x[idx], minlength=n)
        return y
    return matvec


def gather_tiles(args) -> tuple:
    """Create the model and pull kept-port dense S_i tile blocks.

    Returns (tiles, n_interface, gather_stats).  tiles[i] = (global_idx int64,
    S_kept float64) with pad/Dirichlet ports already sliced out.
    """
    from distributed.model import (
        create_distributed_model, load_distributed_partitions,
    )

    _set_phase('load_bundle')
    bundle = load_distributed_partitions(args.pkl_dir)
    _set_phase('create_model')
    model = create_distributed_model(
        bundle, backend='ray', tiles_per_worker=args.tiles_per_worker)
    interface_sorted = sorted(model.interface_nodes)
    node_to_idx = {nd: i for i, nd in enumerate(interface_sorted)}
    n = len(interface_sorted)

    _set_phase('factor_and_gather')
    t0 = time.perf_counter()
    be = model.backend
    tiles = []
    n_pad_ports_dropped = 0
    factor_stats = []
    # B3-style streaming gather: one tile's dense S_i in flight at a time,
    # sliced+copied then dropped — driver peak ~= kept blocks + one shard.
    for _i, (S_i, port_nodes, stats) in be.call_all_streaming(
            model.workers, 'factor_and_compute_schur',
            [() for _ in model.workers]):
        kept_pos = [p for p, nd in enumerate(port_nodes) if nd in node_to_idx]
        n_pad_ports_dropped += len(port_nodes) - len(kept_pos)
        idx = np.array([node_to_idx[port_nodes[p]] for p in kept_pos],
                       dtype=np.int64)
        if len(kept_pos) == len(port_nodes):
            S_kept = np.array(S_i, dtype=np.float64)  # contiguous copy
        else:
            pos = np.array(kept_pos, dtype=np.int64)
            S_kept = np.ascontiguousarray(
                np.asarray(S_i, dtype=np.float64)[np.ix_(pos, pos)])
        del S_i
        tiles.append((idx, S_kept))
        factor_stats.append({k: stats.get(k) for k in
                             ('n_ports', 'n_interior', 'schur_path',
                              'factor_interior_s', 'compute_schur_s')})
    gather_s = time.perf_counter() - t0
    gstats = {
        'gather_s': gather_s,
        'n_pad_ports_dropped': int(n_pad_ports_dropped),
        'factor_stats': factor_stats,
    }
    return tiles, n, model, gstats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('pkl_dir')
    ap.add_argument('--threads', default='8,16,32,48')
    ap.add_argument('--reps', type=int, default=10)
    ap.add_argument('--json', default=None)
    ap.add_argument('--tiles-per-worker', type=int, default=4,
                    help='pack k tiles per Ray actor to cap concurrent '
                         'tile-factor working memory')
    args = ap.parse_args()

    threading.Thread(target=_mem_logger, daemon=True).start()

    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(name)s %(levelname)s: %(message)s',
        datefmt='%H:%M:%S')

    tiles, n, model, gstats = gather_tiles(args)
    _set_phase('bench')

    sum_np2 = sum(S.shape[0] ** 2 for _, S in tiles)
    dense_gb = sum_np2 * 8 / 1e9
    # Union nnz of overlapping blocks is not computed exactly (that IS the
    # expensive assembly); ~0.85 corrects for same-cut double-counted pairs.
    s_nnz_est = int(0.85 * sum_np2)
    results: Dict[str, object] = {
        'pkl_dir': args.pkl_dir,
        'n_interface': int(n),
        'n_tiles': len(tiles),
        's_nnz': s_nnz_est,
        's_nnz_is_estimate': True,
        'sum_np2': int(sum_np2),
        'tile_port_counts': [int(S.shape[0]) for _, S in tiles],
        'dense_blocks_gb': dense_gb,
        'gather': gstats,
        'timings': {},
    }
    timings: Dict[str, Dict[str, float]] = results['timings']  # type: ignore

    rng = np.random.default_rng(42)
    x = rng.standard_normal(n)

    mv_serial = serial_tilewise_matvec(tiles, n)
    y_serial = mv_serial(x)

    print(f"n={n}  tiles={len(tiles)}  sum_np2={sum_np2/1e9:.2f}B "
          f"(dense blocks {dense_gb:.1f} GB)  nnz_est={s_nnz_est/1e6:.0f}M  "
          f"pad ports dropped={gstats['n_pad_ports_dropped']}", flush=True)

    # --- 1. synthetic CSR SpMV (matching n and estimated nnz) ---
    from scipy.sparse import csr_matrix
    nnz_per_row = max(1, s_nnz_est // n)
    cols = np.minimum(
        n - 1,
        (np.repeat(np.arange(n), nnz_per_row) +
         rng.integers(-5000, 5000, size=n * nnz_per_row))).clip(0)
    indptr = np.arange(0, n * nnz_per_row + 1, nnz_per_row)
    data = rng.standard_normal(n * nnz_per_row)
    S_csr = csr_matrix((data, cols.astype(np.int32), indptr), shape=(n, n))
    del data, cols
    timings['csr_1thread_synthetic'] = _time_op(lambda: S_csr @ x, args.reps)
    timings['csr_1thread_synthetic']['traffic_gb'] = S_csr.nnz * 12 / 1e9
    del S_csr

    # --- 2. serial tilewise ---
    timings['tilewise_serial'] = _time_op(lambda: mv_serial(x), 5)

    # --- 3. threaded tilewise ---
    thread_counts = [int(t) for t in args.threads.split(',')]
    for nt in thread_counts:
        pool = ThreadPoolExecutor(max_workers=nt)
        mv = build_threaded_matvec(tiles, n, nt, pool)
        y_thr = mv(x)
        err = float(np.max(np.abs(y_thr - y_serial)) /
                    max(1e-300, float(np.max(np.abs(y_serial)))))
        timings[f'tilewise_threaded_{nt}'] = _time_op(lambda: mv(x), args.reps)
        timings[f'tilewise_threaded_{nt}']['rel_err_vs_serial'] = err
        pool.shutdown()

    # --- 4. fp32 storage variant at the best thread count ---
    best_nt = thread_counts[-1]
    tiles32 = [(idx, S.astype(np.float32)) for idx, S in tiles]
    pool = ThreadPoolExecutor(max_workers=best_nt)
    mv32 = build_threaded_matvec(tiles32, n, best_nt, pool)
    y32 = mv32(x)
    timings[f'tilewise_fp32_{best_nt}'] = _time_op(lambda: mv32(x), args.reps)
    timings[f'tilewise_fp32_{best_nt}']['rel_err_vs_fp64'] = float(
        np.max(np.abs(y32 - y_serial)) /
        max(1e-300, float(np.max(np.abs(y_serial)))))
    pool.shutdown()
    del tiles32

    # --- 5. ownership-block-Jacobi apply (approx of production BJ cost) ---
    _set_phase('bench_bj')
    from scipy.linalg import cho_factor, cho_solve
    owner = np.full(n, -1, dtype=np.int64)
    bj_blocks = []
    for t, (idx, S_i) in enumerate(tiles):
        own_pos = np.where(owner[idx] == -1)[0]
        owner[idx[own_pos]] = t
        if len(own_pos) == 0:
            continue
        blk = S_i[np.ix_(own_pos, own_pos)].copy()
        blk[np.diag_indices_from(blk)] += 1e-8 * np.abs(blk).max()
        try:
            bj_blocks.append((idx[own_pos], cho_factor(blk)))
        except np.linalg.LinAlgError:
            pass  # skip indefinite block — timing probe only

    def bj_apply(r: np.ndarray) -> np.ndarray:
        z = np.zeros(n)
        for gidx, cf in bj_blocks:
            z[gidx] = cho_solve(cf, r[gidx])
        return z

    timings['bj_apply_ownership'] = _time_op(lambda: bj_apply(x), args.reps)
    timings['bj_apply_ownership']['traffic_gb'] = sum(
        cf[0].nbytes for _, cf in bj_blocks) / 1e9
    del bj_blocks

    # --- 6. coarse-setup probe: threaded matmat with T'~=n_tiles+1 columns ---
    _set_phase('bench_matmat')
    n_cols = len(tiles) + 1
    Z = rng.standard_normal((n, n_cols))
    pool = ThreadPoolExecutor(max_workers=best_nt)
    costs = [float(S.shape[0]) ** 2 for _, S in tiles]
    part = _lpt_partition(costs, best_nt)
    acc_mm = np.zeros((best_nt, n, n_cols))

    def matmat() -> np.ndarray:
        acc_mm.fill(0.0)

        def work(t: int) -> None:
            out = acc_mm[t]
            for i in part[t]:
                idx, S_i = tiles[i]
                np.add.at(out, idx, S_i @ Z[idx, :])

        list(pool.map(work, range(best_nt)))
        return acc_mm.sum(axis=0)

    try:
        timings[f'matmat_{n_cols}cols_{best_nt}t'] = _time_op(matmat, 3)
    except MemoryError:
        timings[f'matmat_{n_cols}cols_{best_nt}t'] = {'mean_ms': float('nan')}
    pool.shutdown()
    del acc_mm, Z

    # --- 7. STREAM triad ceiling: single-thread and threaded ---
    _set_phase('bench_stream')
    m = 100_000_000  # 3 arrays x 0.8 GB
    a = np.zeros(m)
    b = rng.standard_normal(m)
    c = rng.standard_normal(m)

    def triad() -> None:
        np.add(b, 2.5 * c, out=a)

    t = _time_op(triad, 5)
    bw = 3 * m * 8 / (t['min_ms'] / 1e3) / 1e9
    timings['stream_triad_1t'] = {**t, 'gbps': bw}

    pool = ThreadPoolExecutor(max_workers=best_nt)
    chunks = np.array_split(np.arange(m), best_nt)

    def triad_mt() -> None:
        def work(ch):
            np.add(b[ch[0]:ch[-1] + 1], 2.5 * c[ch[0]:ch[-1] + 1],
                   out=a[ch[0]:ch[-1] + 1])
        list(pool.map(work, chunks))

    t = _time_op(triad_mt, 5)
    bw_mt = 3 * m * 8 / (t['min_ms'] / 1e3) / 1e9
    timings[f'stream_triad_{best_nt}t'] = {**t, 'gbps': bw_mt}
    pool.shutdown()

    print('\n=== results ===', flush=True)
    for k, v in timings.items():
        extra = ''.join(f'  {ek}={ev:.3g}' for ek, ev in v.items()
                        if ek not in ('mean_ms', 'min_ms', 'max_ms'))
        print(f"{k:28s} mean={v['mean_ms']:9.2f} ms  "
              f"min={v.get('min_ms', float('nan')):9.2f} ms{extra}", flush=True)

    if args.json:
        with open(args.json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f'wrote {args.json}', flush=True)


if __name__ == '__main__':
    main()

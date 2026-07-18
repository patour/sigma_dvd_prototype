#!/usr/bin/env python
"""GPU interface-matvec microbenchmark (Stage 0/Stage 4 layout decision).

Synthesizes an interface system with the SAME shapes as the measured bundle
(reads the JSON emitted by bench_interface_matvec.py: n_interface, per-tile
port counts via sum_np2/n_tiles, S nnz) and times on the GPU:

  1. device CSR SpMV, fp64 and fp32        (candidate Stage 4 default)
  2. size-bucketed batched dense GEMV over padded S_i blocks, fp64/fp32
  3. H2D/D2H transfer of the CG vectors (~2.8 MB)
  4. device batched block-Jacobi apply (padded batched GEMM with inverses)

Bandwidth-bound kernels care about shape/nnz, not values, so synthetic data
decides the layout without a second 30-minute prepare.

Usage:
    venv/bin/python scripts/benchmark/microbench/bench_gpu_matvec.py \
        --shapes scripts/benchmark/microbench/results_matvec_mi100k.json \
        [--reps 20]
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np


def _time_gpu(fn, reps: int, sync) -> dict:
    fn(); sync()
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn(); sync()
        times.append(time.perf_counter() - t0)
    return {'mean_ms': 1e3 * float(np.mean(times)),
            'min_ms': 1e3 * float(np.min(times))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--shapes', required=True)
    ap.add_argument('--reps', type=int, default=20)
    ap.add_argument('--json', default=None)
    args = ap.parse_args()

    with open(args.shapes) as f:
        shapes = json.load(f)
    n = shapes['n_interface']
    n_tiles = shapes['n_tiles']
    s_nnz = shapes['s_nnz']
    tile_np = shapes.get('tile_port_counts')
    if not tile_np:
        avg = int(np.sqrt(shapes['sum_np2'] / n_tiles))
        tile_np = [avg] * n_tiles

    import cupy as cp
    import cupyx.scipy.sparse as csp

    dev = cp.cuda.Device(0)
    free0, total = dev.mem_info
    print(f"GPU: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()} "
          f"free={free0/1e9:.1f}/{total/1e9:.1f} GB")
    print(f"shapes: n={n} tiles={n_tiles} nnz={s_nnz/1e6:.0f}M "
          f"sum_np2={sum(p*p for p in tile_np)/1e6:.0f}M")
    sync = cp.cuda.Stream.null.synchronize
    results = {'timings': {}}
    T = results['timings']
    rng = np.random.default_rng(0)

    # --- 1. CSR SpMV (synthetic banded-ish pattern with matching nnz/row) ---
    nnz_per_row = max(1, s_nnz // n)
    cols = np.minimum(
        n - 1,
        (np.repeat(np.arange(n), nnz_per_row) +
         rng.integers(-5000, 5000, size=n * nnz_per_row))).clip(0)
    # Row-sort and mark canonical: without this, cupyx's __mul__ triggers
    # sum_duplicates() -> full-COO lexsort of ~2B entries -> device OOM.
    cols = np.sort(cols.reshape(n, nnz_per_row), axis=1).ravel()
    indptr = np.arange(0, n * nnz_per_row + 1, nnz_per_row)
    for dt, tag in ((np.float64, 'fp64'), (np.float32, 'fp32')):
        data = rng.standard_normal(n * nnz_per_row).astype(dt)
        S_gpu = csp.csr_matrix(
            (cp.asarray(data), cp.asarray(cols.astype(np.int32)),
             cp.asarray(indptr.astype(np.int32))), shape=(n, n))
        S_gpu.has_canonical_format = True
        x = cp.asarray(rng.standard_normal(n).astype(dt))
        T[f'csr_spmv_{tag}'] = _time_gpu(lambda: S_gpu @ x, args.reps, sync)
        gb = (data.nbytes + cols.nbytes) / 1e9
        T[f'csr_spmv_{tag}']['traffic_gb'] = gb
        del S_gpu, data, x
        cp.get_default_memory_pool().free_all_blocks()

    # --- 2. size-bucketed batched dense GEMV ---
    # 16 buckets over the sorted (skewed) tile sizes keep padding waste low;
    # 4 buckets padded one bucket alone to ~31 GB -> device OOM.
    cp.get_default_memory_pool().free_all_blocks()
    order = np.argsort(tile_np)
    n_buckets = 16
    buckets = [b for b in np.array_split(order, n_buckets) if len(b)]
    for dt, tag in ((np.float64, 'fp64'), (np.float32, 'fp32')):
        blocks, idxs = [], []
        for b in buckets:
            pad = max(tile_np[i] for i in b)
            arr = cp.asarray(
                rng.standard_normal((len(b), pad, pad)).astype(dt))
            gidx = cp.asarray(
                rng.integers(0, n, size=(len(b), pad)).astype(np.int32))
            blocks.append(arr)
            idxs.append(gidx)
        x = cp.asarray(rng.standard_normal(n).astype(dt))
        y = cp.zeros(n, dtype=cp.float64)

        def batched():
            y.fill(0.0)
            for arr, gidx in zip(blocks, idxs):
                xl = x[gidx]                       # (nb, pad)
                yl = cp.matmul(arr, xl[:, :, None])[:, :, 0]
                cp.add.at(y, gidx, yl.astype(cp.float64))
            return y

        T[f'batched_gemv_{tag}'] = _time_gpu(batched, args.reps, sync)
        T[f'batched_gemv_{tag}']['traffic_gb'] = sum(
            a.nbytes for a in blocks) / 1e9
        del blocks, idxs, x, y
        cp.get_default_memory_pool().free_all_blocks()

    # --- 3. H2D/D2H of the CG vector ---
    hx = rng.standard_normal(n)
    T['h2d_vector'] = _time_gpu(lambda: cp.asarray(hx), args.reps, sync)
    gx = cp.asarray(hx)
    T['d2h_vector'] = _time_gpu(lambda: gx.get(), args.reps, sync)

    # --- 4. batched block-Jacobi apply (padded inverses, disjoint scatter) ---
    own = np.array_split(np.arange(n), n_tiles)
    pad = max(len(o) for o in own)
    inv = cp.asarray(rng.standard_normal((n_tiles, pad, pad)))
    oidx = cp.asarray(np.stack([
        np.pad(o, (0, pad - len(o))) for o in own]).astype(np.int64))
    r = cp.asarray(rng.standard_normal(n))

    def bj():
        rl = r[oidx]
        zl = cp.einsum('bij,bj->bi', inv, rl)
        out = cp.zeros(n)
        out[oidx.ravel()] = zl.ravel()
        return out

    T['bj_apply_fp64'] = _time_gpu(bj, args.reps, sync)
    T['bj_apply_fp64']['traffic_gb'] = inv.nbytes / 1e9

    print('\n=== results ===')
    for k, v in T.items():
        extra = f"  {v['traffic_gb']:.1f} GB -> {v['traffic_gb']/(v['min_ms']/1e3):.0f} GB/s" \
            if 'traffic_gb' in v else ''
        print(f"{k:22s} mean={v['mean_ms']:8.3f} ms  min={v['min_ms']:8.3f} ms{extra}")

    if args.json:
        with open(args.json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"wrote {args.json}")


if __name__ == '__main__':
    main()

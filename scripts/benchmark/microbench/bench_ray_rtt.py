#!/usr/bin/env python
"""Ray round-trip latency microbenchmark (Stage 0, interface-solve plan).

Measures the coordinator<->actor communication cost that a worker-side
interface matvec would pay per CG iteration:

  A. broadcast: ray.put one (n_interface,) float64 vector (~1.5 MB at 190K),
     call all actors with the shared ObjectRef, each returns a small slice,
     coordinator ray.get's all results.
  B. sliced: send each actor only its own ~30 KB port-slice (no shared put).

Reports p50/p95/mean per round over --rounds rounds, after warmup.

Usage:
    venv/bin/python scripts/benchmark/microbench/bench_ray_rtt.py \
        [--actors 107] [--rounds 100] [--n 190867] [--slice-bytes 30000]
"""
from __future__ import annotations

import argparse
import statistics
import time

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--actors', type=int, default=107)
    ap.add_argument('--rounds', type=int, default=100)
    ap.add_argument('--n', type=int, default=190_867,
                    help='global vector length (float64)')
    ap.add_argument('--slice-bytes', type=int, default=30_000,
                    help='per-actor slice size for mode B')
    args = ap.parse_args()

    import ray
    ray.init(ignore_reinit_error=True, log_to_driver=False)

    @ray.remote
    class Echo:
        def __init__(self, lo: int, hi: int):
            self.lo = lo
            self.hi = hi

        def apply_shared(self, x: np.ndarray) -> np.ndarray:
            # Simulate gather + trivial compute + return slice
            return x[self.lo:self.hi] * 2.0

        def apply_slice(self, x_local: np.ndarray) -> np.ndarray:
            return x_local * 2.0

    n = args.n
    n_act = args.actors
    bounds = np.linspace(0, n, n_act + 1).astype(int)
    actors = [Echo.remote(int(bounds[i]), int(bounds[i + 1]))
              for i in range(n_act)]
    ray.get([a.apply_slice.remote(np.zeros(4)) for a in actors])  # spawn

    x = np.random.default_rng(0).standard_normal(n)
    slice_len = args.slice_bytes // 8
    x_slices = [np.ascontiguousarray(x[:slice_len]) for _ in range(n_act)]

    def bench(fn, label: str) -> None:
        for _ in range(5):
            fn()
        times = []
        for _ in range(args.rounds):
            t0 = time.perf_counter()
            fn()
            times.append(time.perf_counter() - t0)
        ms = sorted(t * 1e3 for t in times)
        p50 = ms[len(ms) // 2]
        p95 = ms[int(len(ms) * 0.95)]
        print(f"{label:28s} p50={p50:8.3f} ms  p95={p95:8.3f} ms  "
              f"mean={statistics.mean(ms):8.3f} ms  (rounds={args.rounds})")

    def round_shared():
        ref = ray.put(x)
        ray.get([a.apply_shared.remote(ref) for a in actors])

    def round_sliced():
        ray.get([a.apply_slice.remote(s)
                 for a, s in zip(actors, x_slices)])

    print(f"actors={n_act}  n={n} ({n * 8 / 1e6:.1f} MB broadcast)  "
          f"slice={slice_len * 8 / 1e3:.0f} KB")
    bench(round_shared, 'A broadcast+gather (put)')
    bench(round_sliced, 'B per-actor slices')

    ray.shutdown()


if __name__ == '__main__':
    main()

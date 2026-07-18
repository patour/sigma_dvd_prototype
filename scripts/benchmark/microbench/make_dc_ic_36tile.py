#!/usr/bin/env python
"""Produce a full-chip DC voltage dict from the 36-tile bundle (direct solve).

The 64-tile mi200k cold BJ-CG DC solve stagnates (kappa-limited) and the
direct factor does not fit at the split regime on this host — but the SAME
netlist's 36-tile bundle solves DC directly in minutes (§7.6). Node voltages
are tiling-independent, so this dict seeds the 64-tile transient measurement
via solve_transient(ic_voltages=...).

Usage:
    venv/bin/python -u scripts/benchmark/microbench/make_dc_ic_36tile.py \
        netlist/netlist_brcm_sampled/distributed_pkl <out.pkl>
"""
from __future__ import annotations

import logging
import pickle
import sys
import time

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)s %(levelname)s: %(message)s',
                    datefmt='%H:%M:%S')


def main() -> None:
    pkl_dir, out_path = sys.argv[1], sys.argv[2]
    from distributed.model import (
        create_distributed_model, load_distributed_partitions,
    )
    from distributed.solver import DistributedDDMSolver

    bundle = load_distributed_partitions(pkl_dir)
    model = create_distributed_model(bundle, backend='ray')
    model.settings['interface_solver'] = 'direct'
    solver = DistributedDDMSolver(model)
    t0 = time.perf_counter()
    ctx = solver.prepare(verbose=True)
    print(f'prepare {time.perf_counter() - t0:.1f}s', flush=True)
    t0 = time.perf_counter()
    res = solver.solve_dc(ctx)
    print(f'solve_dc {time.perf_counter() - t0:.1f}s', flush=True)
    flat = res.flatten()
    print(f'{len(flat)} node voltages', flush=True)
    with open(out_path, 'wb') as f:
        pickle.dump(flat, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'wrote {out_path}', flush=True)
    ctx.release()


if __name__ == '__main__':
    main()

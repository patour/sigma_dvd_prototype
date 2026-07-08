#!/usr/bin/env python3
"""Distributed DDM end-to-end performance baseline capture and comparison.

Runs the distributed solver on netlist/netlist_sampled in transient mode via
the API (mirrors what cli.py's `solve` subcommand does) and extracts phase
timings into a JSON file.

Captured timings schema::

    {
        "load":          float,   # model load time (s)
        "vcs":           float,   # VCS init time (s)      [*]
        "smooth":        float,   # PWL smoothing time (s) [*]
        "dc_prepare":    float,   # DC prepare (factor tiles + interface) (s)
        "trans_prepare": float,   # Transient prepare (s)
        "loop_total":    float,   # Total time-loop wall time (s)
        "per_step": {
            "rhs":       float,   # Cumulative RHS time / n_steps (s/step)
            "solve":     float,   # Cumulative assemble+solve time / n_steps (s/step)
            "recovery":  float    # Cumulative recovery time / n_steps (s/step)
        },
        "n_steps":       int,
        "peak_ir_drop_mV": float,
        "peak_time_ns":  float
    }

[*] ``vcs`` and ``smooth`` are extracted from the ``timings`` attribute on
``DistributedSmoothedSources`` (``result.py``), which records ``init_vcs`` and
``smooth_sources`` durations separately inside ``preprocess_sources()``
(``solver_td.py``).  Those two fields were added to the production source as
the minimal change needed to expose the split — they are additive (default
``{}``) and have no effect on solve correctness.  If the field is missing
(e.g. on an older build), the script falls back to reporting the combined
wall-clock under ``vcs`` and ``smooth=0.0``.

Usage::

    # Capture baseline
    python scripts/benchmark/run_perf_baseline.py --output scripts/benchmark/baselines/perf.json

    # Compare against baseline (exit 1 if any phase regresses by more than --max-regress %)
    python scripts/benchmark/run_perf_baseline.py --compare scripts/benchmark/baselines/perf.json --max-regress 20
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Project root / Python path bootstrap
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
if str(_PROJECT_ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / 'src'))

# Notebooks (and Ray workers) require cwd == project root so that relative
# paths like 'netlist/netlist_sampled/distributed_pkl' resolve correctly.
os.chdir(_PROJECT_ROOT)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
_DEFAULT_PKL_DIR = 'netlist/netlist_sampled/distributed_pkl'
_DEFAULT_T_END   = 1e-9    # 1 ns  (spec: ~minutes on netlist_sampled)
_DEFAULT_DT      = 10e-12  # 10 ps
_DEFAULT_BACKEND = 'ray'


# ---------------------------------------------------------------------------
# Core benchmark
# ---------------------------------------------------------------------------

def run_benchmark(
    pkl_dir: str,
    t_end: float,
    dt: float,
    backend: str,
    verbose: bool,
    use_step_columns: bool = True,
) -> dict:
    """Run end-to-end transient solve and return timing dict."""
    from distributed.model import create_distributed_model, load_distributed_partitions
    from distributed.solver import DistributedDDMSolver

    timings: dict = {}

    # 1. Load partitions / create model
    t0 = time.perf_counter()
    bundle = load_distributed_partitions(pkl_dir)
    model  = create_distributed_model(bundle, backend=backend)
    timings['load'] = time.perf_counter() - t0
    if verbose:
        print(f"[perf] load: {timings['load']:.3f}s", flush=True)

    solver = DistributedDDMSolver(model)

    try:
        # 2. Preprocess sources (VCS init + smoothing)
        t0 = time.perf_counter()
        smoothed = solver.preprocess_sources(
            time_step=dt,
            t_start=0.0,
            t_end=t_end,
            smooth=True,
            verbose=verbose,
        )
        t_vcs_total = time.perf_counter() - t0

        # Split VCS init from smoothing: preprocess_sources attaches its
        # internal timings dict to the returned handle as .timings.
        # Keys: 'init_vcs' (parallel load / pickle cache) and optionally
        # 'smooth_sources' (triangular PWL smoothing pass).
        vcs_timings = getattr(smoothed, 'timings', {})
        if vcs_timings:
            timings['vcs']    = vcs_timings.get('init_vcs', t_vcs_total)
            timings['smooth'] = vcs_timings.get('smooth_sources', 0.0)
        else:
            # Fallback: whole block under vcs, smooth unknown
            timings['vcs']    = t_vcs_total
            timings['smooth'] = 0.0

        if verbose:
            print(f"[perf] vcs:    {timings['vcs']:.3f}s  "
                  f"smooth: {timings['smooth']:.3f}s  "
                  f"(total {t_vcs_total:.3f}s)", flush=True)

        # 3. DC prepare
        t0 = time.perf_counter()
        dc_ctx = solver.prepare(verbose=verbose)
        timings['dc_prepare'] = time.perf_counter() - t0
        if verbose:
            print(f"[perf] dc_prepare: {timings['dc_prepare']:.3f}s", flush=True)

        # Pull internal DC-prepare phase details from the context if available.
        dc_timings = getattr(dc_ctx, 'timings', {})
        if dc_timings:
            timings['dc_prepare_detail'] = {
                k: v for k, v in dc_timings.items()
                if isinstance(v, (int, float))
            }

        # 4. Transient prepare
        t0 = time.perf_counter()
        trans_ctx = solver.prepare_transient(dt=dt, method='be', verbose=verbose)
        timings['trans_prepare'] = time.perf_counter() - t0
        if verbose:
            print(f"[perf] trans_prepare: {timings['trans_prepare']:.3f}s", flush=True)

        # 5. Transient solve
        result = solver.solve_transient(
            trans_ctx,
            dc_context=dc_ctx,
            t_start=0.0,
            t_end=t_end,
            smoothed_sources=smoothed,
            use_step_columns=use_step_columns,
            verbose=verbose,
        )

        # Extract per-step stats from result metadata
        solve_timings = result.solve_metadata.get('timings', {})
        loop_stats    = solve_timings.get('loop_stats', {})
        n_steps       = loop_stats.get('n_steps', 0) or 1

        timings['loop_total'] = solve_timings.get('time_loop', 0.0)
        timings['per_step'] = {
            'rhs':      loop_stats.get('cum_rhs_time_s', 0.0)      / n_steps,
            'solve':    loop_stats.get('cum_asm_solve_time_s', 0.0) / n_steps,
            'recovery': loop_stats.get('cum_recovery_time_s', 0.0)  / n_steps,
        }
        timings['n_steps']         = int(n_steps)
        timings['peak_ir_drop_mV'] = float(result.peak_ir_drop * 1000)
        timings['peak_time_ns']    = float(result.peak_ir_drop_time * 1e9)

        if verbose:
            print(f"[perf] loop_total: {timings['loop_total']:.3f}s  "
                  f"({n_steps} steps @ {timings['loop_total']/n_steps:.4f}s/step)", flush=True)
            print(f"[perf] peak_ir_drop: {timings['peak_ir_drop_mV']:.3f} mV "
                  f"at {timings['peak_time_ns']:.3f} ns", flush=True)

        dc_ctx.release()
        trans_ctx.release()

    finally:
        model.shutdown()

    return timings


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def compare_timings(current: dict, baseline: dict, max_regress_pct: float) -> bool:
    """Compare current vs baseline.  Returns True if all phases pass.

    A key present in the baseline but absent from *current* is flagged as FAIL
    (not treated as an improvement).  A zero current value that is present in
    both dicts is still compared normally.
    """
    # Phase keys to compare (skip nested dicts for now)
    top_keys = ['load', 'vcs', 'smooth', 'dc_prepare', 'trans_prepare', 'loop_total']
    step_keys = ['rhs', 'solve', 'recovery']

    all_pass = True
    lines = []
    lines.append(f"\n{'Phase':<25} {'Baseline':>12} {'Current':>12} {'Change':>10}  Status")
    lines.append('-' * 70)

    def _check(name: str, base: float, cur: float | None) -> None:
        nonlocal all_pass
        if cur is None:
            # Key present in baseline but missing from current run — always fail.
            lines.append(
                f"  {name:<23} {base:>11.3f}s {'MISSING':>12}  "
                f"FAIL (key absent from current results)"
            )
            all_pass = False
            return
        if base <= 0:
            lines.append(f"  {name:<23} {'N/A':>12} {cur:>11.3f}s  {'SKIP (no baseline)':}")
            return
        pct = (cur - base) / base * 100
        status = 'OK' if pct <= max_regress_pct else f'REGRESS (+{pct:.1f}% > {max_regress_pct:.0f}%)'
        if pct > max_regress_pct:
            all_pass = False
        lines.append(f"  {name:<23} {base:>11.3f}s {cur:>11.3f}s {pct:>+9.1f}%  {status}")

    for k in top_keys:
        base_val = baseline.get(k, 0.0)
        # Use sentinel None to distinguish "key absent" from "key present but 0"
        cur_val = current.get(k) if k in current else None
        _check(k, base_val, cur_val)

    base_ps = baseline.get('per_step', {})
    cur_ps  = current.get('per_step', {})
    for k in step_keys:
        base_val = base_ps.get(k, 0.0)
        cur_val = cur_ps.get(k) if k in cur_ps else None
        _check(f'per_step/{k}', base_val, cur_val)

    print('\n'.join(lines))

    base_drop = baseline.get('peak_ir_drop_mV', 0.0)
    cur_drop  = current.get('peak_ir_drop_mV', 0.0)
    if base_drop:
        diff_mv = abs(cur_drop - base_drop)
        print(f"\npeak_ir_drop: baseline={base_drop:.4f} mV, current={cur_drop:.4f} mV, diff={diff_mv:.4f} mV")

    return all_pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description='Distributed DDM performance baseline capture / comparison.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--pkl-dir', default=_DEFAULT_PKL_DIR,
                   help='Path to parsed tile pkl directory')
    p.add_argument('--t-end', type=float, default=_DEFAULT_T_END,
                   help='Simulation end time in seconds')
    p.add_argument('--dt', type=float, default=_DEFAULT_DT,
                   help='Transient time step in seconds')
    p.add_argument('--backend', default=_DEFAULT_BACKEND, choices=['ray', 'local'],
                   help='Compute backend')
    p.add_argument('--output', '-o', type=str, default=None,
                   help='Write captured timings to this JSON file')
    p.add_argument('--compare', type=str, default=None,
                   help='Baseline JSON to compare against')
    p.add_argument('--max-regress', type=float, default=20.0,
                   help='Max allowed regression percent before exit-1')
    p.add_argument('--verbose', '-v', action='store_true',
                   help='Verbose solver output')
    p.add_argument('--no-step-columns', action='store_true',
                   help='Disable A2 phase-folded step-column table (flag-off A/B)')
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    pkl_dir = str(_PROJECT_ROOT / args.pkl_dir) if not Path(args.pkl_dir).is_absolute() else args.pkl_dir

    print(f"Running distributed transient benchmark on {pkl_dir}", flush=True)
    print(f"  backend={args.backend}  t_end={args.t_end*1e9:.2f}ns  dt={args.dt*1e12:.1f}ps", flush=True)

    t_wall = time.perf_counter()
    use_sc = not args.no_step_columns
    if not use_sc:
        print("  [A2] --no-step-columns: step-column table disabled", flush=True)
    timings = run_benchmark(
        pkl_dir=pkl_dir,
        t_end=args.t_end,
        dt=args.dt,
        backend=args.backend,
        verbose=args.verbose,
        use_step_columns=use_sc,
    )
    timings['wall_time_s'] = time.perf_counter() - t_wall

    print(f"\n=== Phase Timings ===")
    for k, v in timings.items():
        if k == 'per_step':
            for sk, sv in v.items():
                print(f"  per_step/{sk:<18} {sv:.4f} s/step")
        elif k == 'dc_prepare_detail':
            for sk, sv in v.items():
                print(f"  dc_prepare/{sk:<16} {sv:.3f} s")
        elif isinstance(v, float):
            print(f"  {k:<25} {v:.3f} s")
        elif isinstance(v, int):
            print(f"  {k:<25} {v}")
    print(flush=True)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump(timings, f, indent=2)
        print(f"Timings written to {out_path}")

    if args.compare:
        cmp_path = Path(args.compare)
        if not cmp_path.exists():
            print(f"ERROR: baseline file not found: {cmp_path}", file=sys.stderr)
            sys.exit(1)
        with open(cmp_path) as f:
            baseline = json.load(f)
        print(f"\nComparing against baseline: {cmp_path}")
        passed = compare_timings(timings, baseline, args.max_regress)
        if not passed:
            print(f"\nREGRESSION DETECTED (threshold={args.max_regress:.0f}%)", file=sys.stderr)
            sys.exit(1)
        print("\nAll phases within regression budget.")


if __name__ == '__main__':
    main()

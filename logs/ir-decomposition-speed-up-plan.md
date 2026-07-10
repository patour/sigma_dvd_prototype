# Speed up A2 step-column build in distributed IR-drop decomposition

> **Status (2026-07-10): IMPLEMENTED** — Changes A + B + C landed on `distributed-10x` as
> `11478ce` (A+C reuse cache + amortization guard), `5e8182e` (VCS-backed end-to-end tests),
> `3d7abcc` (B chunked direct-scatter windows). Gates: 34+24 new unit tests, full distributed
> unit suite, equivalence suite (68 passed, 6 expected xfails), netlist_sampled perf compare
> (loop_total −25%, results exact) and four-notebook regression. See
> `docs/brcm_distributed_runtime_optimization.md` §7.2. End-to-end minion re-run pending
> (netlist_minion not on this host).

## Context

The `20260710` decompose run on `netlist_minion` regressed to **1117s vs 857s** (old
`20260512`) despite the transient *time loop* getting 2.3× faster (0.449→0.198 s/step).
Root cause analysis of both logs isolated three regressions; the largest is the
**A2 step-column build (314s)** logged as `A2 step columns: 1 tiles, tiers=['chunked'], 314.029s`
inside `initial_transient` (394.6s total).

The user asked to draft + benchmark **Option 1: vectorize `_evaluate_pwls_batch`**
(the per-row Python loop over ~778K PWL rows suspected of being 2× slower than the
per-step `evaluate_at_time` path).

## Benchmark results — Option 1 is REJECTED

All benchmarks run against the real smoothed VCS on disk
(`netlist/netlist_minion/distributed_pkl/vcs_tile_0_0_smoothed_b90c2c37c215.pkl`):
`n_pwls=777,992`, `n_pwl_points=4,281,258`, **mean 5.5 knots/PWL** (min 2, max 364),
`n_pulses=0`, single period `P=1e-8`, `dt=1e-11` → `m=1000`.

| Path | Time (isolated, 1 build / 400 steps) | Notes |
|---|---|---|
| On-the-fly `evaluate_at_time` × 400 | **77.2s** (193 ms/step) | old-run behavior; dispatches to `_evaluate_pwls_binned` |
| Chunked build `evaluate_at_times_for_rows` (W=400) | **74.7s** (ratio 0.97×) | new-run behavior; NOT 2× slower in isolation |
| `_evaluate_pwls_batch` alone (m=400) | 54.6s, allocates 2.49 GB | the dominant cost |
| **Option 1 prototype** (vectorized binned batch over m) | **98.6s (0.54× — SLOWER)**, bit-exact (0 diff) | REJECTED |

**Why Option 1 fails:** fully vectorizing the row loop builds a
`(n_group, max_count, m)` 3-D boolean intermediate for the segment search. With
PWLs already **compacted to ~5.5 knots each**, the existing per-row loop
(`vectorized_sources.py:1233`) is near-optimal — the Python loop overhead is small
relative to the per-row numpy work, and the 3-D broadcast's memory traffic dominates.
Vectorizing across `m` is a net loss. Correctness was verified (max abs diff = 0.0).

**Why the real run showed 314s (~4× the isolated 75s):** the Ray tile actor runs
thread-capped (`OMP_NUM_THREADS`/`OPENBLAS_NUM_THREADS` per worker via `runtime_env`)
and under memory pressure from the multi-GB intermediates — not an algorithmic issue
in `_evaluate_pwls_batch`. So the fix is **not to make one build faster**, but to
**stop rebuilding**, and to use the fast per-step path.

## The fix the data supports

Two independent, additive changes. **Change A is the primary win.**

### Change A — Reuse the step-column table across transients (biggest lever)

`solve_transient` rebuilds the step-column table on **every** call
(`solver_td.py:754-762`, `precompute_step_columns` on all workers). A single
`decompose` run calls `solve_transient` **5 times** on the *same smoothed sources*
(`decomposition.py:575/658/748/821/959` — initial, targeted-missing, 2× near-only,
adjoint). Each rebuild is a full ~75s (isolated) / ~314s (Ray) build.

Because the near/far mask is applied **post-gather** (per CLAUDE.md — the same `C`
table serves all victims), the table is a pure function of `(active_sources, t_start,
dt, n_steps)`. It can be built once and reused.

**Reuse math (6 transient passes on same sources):**
- On-the-fly: 6 × 77s = **462s**
- Build once + reuse: 135s (phase, m=1000) + 6 × 0.69s = **139s** → **~3.3× fewer build-seconds**

Implementation:
- Cache the built table keyed by `(id(active_sources), t_start, dt, n_steps)` on the
  worker. `precompute_step_columns` (`tile_worker_td.py:459`) returns the cached table
  when the key matches instead of rebuilding.
- The existing invalidation hooks already null `self._step_col_table` on every
  source-mutating call (`init_vectorized_sources:173`, `smooth_sources:332/367`,
  `use_smoothed_sources:426`, and the disabled path `:506`). Extend these to also clear
  the cache key so correctness is preserved automatically.
- `solve_transient` should skip the `precompute_step_columns` round-trip entirely when
  the coordinator knows the key is unchanged, OR let the worker no-op cheaply. Prefer
  the worker-side cache (simpler, robust to the multiple call sites).

### Change B — Let the phase tier win by raising `max_table_mb` (secondary)

The phase tier (column-gather, **1.7 ms/step** vs 193 ms/step on-the-fly) is
disqualified purely by memory budget: `est_table_mb(m=1000)=1488 MB > max_table_mb=512`
(`tile_worker_td.py:533`), forcing the chunked tier. The phase table is only 1.49 GB
and, once built, makes the 400-step loop essentially free (0.69s total).

Options (pick one, expose via YAML `solver.max_table_mb` — already wired per
CLAUDE.md B-features):
- Raise the default/config `max_table_mb` (e.g. 2048) for hosts with RAM headroom, so
  single-tile large-node runs use the phase tier.
- Or teach the **chunked tier a direct-scatter fast path** mirroring
  `_build_via_direct_scatter` (`tile_worker_td.py:802`): smoothed PWLs sit on the
  uniform `actual_step` grid, so window columns can be gathered from `pwl_values` by
  index instead of re-evaluated — removing the `evaluate_at_times_for_rows` call from
  both the eager build (`:713`) and the on-demand rebuild (`:978`). This is the
  memory-safe version of the phase-tier win and also benefits multi-window runs.

### Change C — Gate the precompute when it can't amortize (cheap guard)

When the resolved tier is `chunked` **and** `n_steps ≤ W` (single window, no reuse
*within* one solve) **and** table reuse (Change A) is not in play, skip the precompute
and fall back to per-step `evaluate_at_time` — it is strictly no-slower (0.97×) and
avoids the multi-GB intermediate. Low risk; keeps behavior sane if A/B are deferred.

## Files to modify

- `src/distributed/tile_worker_td.py`
  - `precompute_step_columns` (`:459`) — add build-key cache + reuse; gate logic (Change C).
  - `_build_chunked_window` (`:682`) + on-demand rebuild in `_get_current_array_for_step`
    (`:958-996`) — optional direct-scatter fast path (Change B alt).
  - invalidation sites (`:173/332/367/426/506`) — clear cache key alongside `_step_col_table`.
- `src/distributed/solver_td.py`
  - `solve_transient` step-column setup (`:750-769`) — avoid redundant rebuild across the
    5 decompose calls; pass/trust the worker cache.
- `src/distributed/decomposition.py` — no logic change required if reuse is worker-side;
  optionally log when a cached table is reused.
- Config plumbing for `max_table_mb` already exists (CLAUDE.md B-features / `_apply_yaml_role_configs`).

## Verification

1. **Unit correctness** (bit-exact table): the equivalence already holds (prototype diff
   0.0). Add/extend a test in `tests/distributed/test_time_domain.py` asserting a
   reused table produces identical waveforms to a freshly-built one, and that a
   source-mutating call invalidates it.
   ```bash
   pytest tests/distributed/test_time_domain.py -k "step_column" -v
   pytest -m unit
   ```
2. **Equivalence gate** unchanged:
   ```bash
   pytest -m validation tests/validation/test_equivalence.py
   ```
3. **End-to-end decompose** on the same command; confirm `initial_transient` drops and
   the 4 subsequent transients no longer each pay a full build:
   ```bash
   python -m distributed decompose --backend ray \
     --config results/minion/vcd_sdc/ir_drop_decomp/config.yaml \
     netlist/netlist_minion -v
   ```
   Compare the new `Timing breakdown` block against the `20260710` baseline in
   `results/minion/vcd_sdc/ir_drop_decomp/decompose_20260710_110053.log`. Expected:
   `initial_transient` build 314s→(one build), `targeted_transient`/near-only/adjoint
   builds ≈ 0. Target total ~700-750s (also folds in the separate QS-preselection and
   cold-cache findings noted in the log analysis, which are out of scope here).

## Out of scope (tracked separately from the log analysis)

- QS pre-selection mispredicting victims → 44.7s QS + 139s targeted transient
  (`decomposition.py` Phase 2). Consider bypassing QS pre-selection since global peak
  tracking already covers all nodes.
- Cold smoothed-VCS cache (+55s one-time; recovers on re-run).

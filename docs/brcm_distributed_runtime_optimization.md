# BRCM Distributed Solve — Runtime Analysis & Optimization Plan

Source material: `brcm_decompose.log` (IR-drop decomposition run) and `brcm_transient.log`
(transient-only run) on the BRCM testcase — a 36-tile (6×6) PDN, 30.67 M interior nodes,
138,209 interface nodes, `net=VDD_VAR`, `Vdd=0.760 V`. Transient config: BE, `dt=5 ps`,
10,000 steps to 50 ns, 12,393,200 current sources.

The testcase itself is not available; this plan is derived entirely from the two runtime logs
plus the current source in `src/`.

---

## 1. Executive summary

Both runs are dominated by the transient time loop (**12.7 h** and **17.1 h**), and within each
step the **per-step RHS assembly is 54–70 %** of the cost. The RHS is expensive because the
vectorized current-source evaluator recomputes every source from its PWL breakpoints on **every one
of the 10,000 steps** via a brute-force segment search — even though PWL smoothing has already put
all waveforms on a uniform, periodic grid that makes the per-step result a pure table lookup.

The headline fix (**Fix 3 — phase-folded precompute + per-step gather**) collapses the 3.2–3.3 s/step
RHS to microseconds and is *exact* (columns are built from the existing evaluator, reproduced
bit-for-bit). Two configuration wins in the prepare phase (interface factor backend, unified
per-tile CHOLMOD settings) and several caching wins are lower-risk and independently shippable.

Estimated impact: each ~12–17 h time loop drops to the interface-solve floor (~3–5 h), then further
with the supernodal-interface fix. Prepare phase drops by ~55 min/run (decompose) from the backend fix alone.

---

## 2. Where the wall-clock goes (evidence)

### 2.1 `brcm_transient.log` — transient-only, ~19.2 h total

| Phase | Time | Notes |
|---|---|---|
| Model load + create | 47 s | 36 tiles, 138,209 interface nodes |
| VCS init + **smooth_sources** | 181 s + **2697 s** | 12.4 M sources, 0 from cache |
| DC prepare | 2087 s | per-tile factor wall 1240 s, assemble 369 s, islands 404 s, interface factor **74 s (supernodal)** |
| Transient prepare | 2459 s | per-tile factor wall 1250 s, assemble 1130 s, interface factor **79 s (supernodal)** |
| **Transient time loop** | **61,430 s (17.1 h)** | 10,000 steps @ **6.143 s/step** |
| Plot / heatmap | ~min | matplotlib tail |

Per-step split: **RHS 3.313 s** · assemble+solve 1.688 s · interior recovery 1.047 s.
Per-tile RHS min/mean/max = `0.025 / 0.896 / 37.579 s`.

### 2.2 `brcm_decompose.log` — decomposition run, **>26 h and did not finish** (ended at Victim 1/10)

| Phase | Time | Notes |
|---|---|---|
| Model load + create | 48 s | — |
| smooth_sources | 2842 s | VCS cached ("36 from cache") but **smoothing re-ran anyway** |
| DC prepare | 2819 s | interface factor **1668 s (simplicial!)**, assemble 473 s, islands 401 s, per-tile wall 276 s |
| Transient prepare | 3176 s | interface factor **1760 s (simplicial!)**, assemble 1134 s, per-tile wall 282 s |
| DC solve | 15 s | — |
| **Phase 2b initial transient** | **45,755 s (12.7 h)** | 10,000 steps @ **4.576 s/step** |
| **Phase 3 waveform decomposition** | **~11.5 h to reach Victim 1/10** | re-runs a *full* all-sources transient, then per-victim |

Per-step split: **RHS 3.194 s** · assemble+solve 0.673 s · interior recovery 0.620 s.
Per-tile RHS min/mean/max = `0.025 / 0.710 / 98.123 s`.

### 2.3 Shared structure

- Interface system: 70,734 unknowns, 479.4 M nnz, 9.581 % density, 5.4 GB factored.
- Per-tile: interior nodes 201,199 / 852,024 / 1.6 M (min/mean/max) — an **8× imbalance**;
  factor time 6 s → 268 s — a **42× imbalance**. Both the factor wall and the per-step RHS are
  gated by the single densest tile.

---

## 3. Root cause of the time loop: the per-step RHS

### 3.1 The hot path, end to end

Per step, `solve_transient` (`src/distributed/solver_td.py:556`) fans out `get_transient_reduced_rhs`
to all 36 tiles (`:562`). Each tile (`src/distributed/tile_worker_td.py:509`) calls
`self._active_sources.evaluate_at_time(t)`, then slices `[:n_ports]` / `[n_ports:]` (`:512–513`).

`evaluate_at_time` (`src/analysis/vectorized_sources.py:385`) does, **for every step**:

```python
currents = np.zeros(self.n_nodes)                            # :401  full tile-node alloc (up to 1.6M) every step
np.add.at(currents, self.source_node_idx, self.dc_values)   # :405  DC scatter — constant, yet re-done every step
pwl_values = self._evaluate_pwls(t)                         # :416  the expensive part
np.add.at(currents, self.pwl_node_idx, pwl_values)          # :419  unbuffered scatter
```

and `_evaluate_pwls` → `_evaluate_pwls_padded`/`_binned` runs an **O(n_pwls × max_count)** breakpoint
search on every step:

```python
# vectorized_sources.py:634 (padded) and :838 (binned)
seg_idx = np.sum(times_2d <= t_clamped_2d, axis=1) - 1
```

For a heavy tile (~120 k PWLs, max_count 16–128 after smoothing) that materializes and reduces a
multi-million-element boolean matrix per step × 10,000 steps — the multi-second RHS the logs show.

### 3.2 It detects the uniform grid but does not exploit it

Smoothing **retains each waveform's period** (`src/analysis/pwl_smoothing.py:2333`) and **folds delay
into the breakpoint times** (`:2312`, so output delay → 0), snapping samples to
`actual_step = period/round(period/time_step)` (`:347–348`). The evaluator even takes a fast path for
this "no delay, uniform period" case (`vectorized_sources.py:613`, `:827`) reducing `t` to
`t_mod = t % P` — but then **still runs the full matrix search** to find each PWL's segment.

The solver visits only the uniform grid `t_k = t_start + (k+1)·dt` (`solver_td.py:543`). Since the
evaluator is a pure function of `t_mod = t % P`, and `P = m·dt`, there are only **m distinct source
vectors** in the entire run, repeating with stride `m`. The code recomputes one from scratch 10,000×.

### 3.3 The four inefficiencies (per tile, per step)

1. **Full `n_nodes` zero-alloc every step** (`:401`).
2. **Constant DC scatter re-done every step** (`:405`) — DC current is time-invariant.
3. **Brute-force PWL segment search every step** (`:634`/`:838`) — the dominant cost.
4. **`np.add.at` unbuffered scatter** (`:405`, `:419`, and coordinator `solver_td.py:573`) — the slow
   path; `np.bincount` (already used at `vectorized_sources.py:1068`) is 5–30× faster.

---

## 4. Fix 3 (headline) — phase-folded precompute + per-step gather

### 4.1 Core idea and correctness

Because `evaluate_at_time` is periodic with `P`:

```
evaluate_at_time(t_k)  ==  evaluate_at_time(t_start + (1 + (k mod m))·dt)     when P = m·dt
```

So precompute the `m` distinct source vectors once, then per step gather a column:

1. **Precompute (once, per tile, on the worker):**
   `C[:, j] = evaluate_at_time(t_start + (j+1)·dt)`, for `j = 0 … m-1`.
2. **Per step:** `current_array = C[:, k mod m]`.

Columns are built by calling the **existing evaluator** at `m` phases, so they reproduce
`evaluate_at_time` **bit-for-bit** for periodic content — no new numerics, no approximation. The only
precondition is that the evaluator is genuinely periodic with `P`, which holds by construction of `t % P`.

### 4.2 Tier 1 — periodic phase-folding (recommended)

**Data structure (per tile worker):** a `float64` array `C` of shape `(n_active_nodes, m)` in the
tile's port-first node order, with the constant DC contribution already embedded in each column.
Optionally split into `C_port (n_ports, m)` and `C_int (n_interior, m)` to skip the per-step slice.
Store over source-carrying rows only (≤ ~120 k) plus a scatter index, not all `n_nodes`.

**New worker method** (`_TimeDomainMixin`, `tile_worker_td.py`), called once after `smooth_sources`:

```python
def precompute_step_columns(self, t_start, dt, m):
    src = self._active_sources
    self._step_cols = np.empty((src.n_nodes, m), dtype=np.float64)
    for j in range(m):
        self._step_cols[:, j] = src.evaluate_at_time(t_start + (j + 1) * dt)
    self._step_m = m
```

The first `evaluate_at_time` builds the padded/binned cache once — this also moves the suspected
98 s first-step spike (`_build_pwl_padded_cache`/`_build_pwl_binned_groups`, `vectorized_sources.py:684`/`:757`)
out of the timed loop and into setup.

**Per-step replacement** in `get_transient_reduced_rhs` (`tile_worker_td.py:509`) and
`evaluate_and_get_reduced_rhs` (`:307`):

```python
if self._step_cols is not None:
    current_array = self._step_cols[:, step_idx % self._step_m]
else:
    current_array = self._active_sources.evaluate_at_time(t)
```

Pass `step_idx` alongside `t` in the per-tile arg tuples (`solver_td.py:558`, `:612`) rather than
re-deriving `k` from the float `t`.

**Masking (near/far, Phase 3):** keep the post-eval multiply (`tile_worker_td.py:511`):
`current_array = C[:, j] * self._current_node_mask`. The mask is constant in time per victim and
columns are unchanged, so the **same `C` serves all 10 victims** — no per-victim source recompute.

**Memory:** `n_active_nodes × m × 8 B`. Heaviest tile ≈ 120 k sources; at `m ≈ 200` → **~190 MB on
the busiest worker**, trivial elsewhere, per-process (Ray actors), not aggregated.

**Precompute cost:** `m` evaluations instead of `N = 10,000` — at `m ≈ 200` a **50× reduction**, done
once, parallel across workers (~minutes), reused across the DC-IC solve, Phase 2b, and all of Phase 3.

**Per-step cost after change:** a column gather + one `bincount` scatter into the reduced RHS. The
3.2 s/step RHS collapses to ms; the loop becomes gated by `interface_lu(global_rhs)`
(`solver_td.py:603`) and interior recovery — the correct floor.

### 4.3 Tier 2 — fallbacks when not cleanly single-period

- **Multiple distinct periods** (`np.unique(pwl_period)` > 1 value → `_pwl_single_period == 0`):
  group sources by period `P_g`, fold each group at its own `m_g = P_g/dt`, sum per step:
  `current = Σ_g C_g[:, k mod m_g]`. Memory `Σ_g n_g × m_g`.

- **Non-periodic / period = 0** (breakpoints span the full window): precompute the on-grid time series
  in **time chunks** of `W ≈ 500–1000` steps — build `C_chunk (n_active_nodes, W)` once via `W`
  evaluations, stream through it, discard, advance. Still hoists the segment search out of the
  per-step loop; caps memory at `n_active_nodes × W`.

### 4.4 Minimal-diff variant (most conservative)

Keep interpolation, kill only the search: precompute per phase the per-PWL `(seg_idx, frac)` once
(`m × n_pwls` int32 + float32), then per step do the existing gather+interp
(`vectorized_sources.py:638–647`) indexed by `phase = k mod m`, skipping
`np.sum(times_2d <= …)`. Smaller behavior change; still removes the dominant cost but not the per-step scatter.

### 4.5 Correctness guards & edge cases

- **Integrality:** assert `abs(P/dt - round(P/dt)) < eps`; else Tier 2. (Both derive from the same
  `time_step`; smoothing snaps to `actual_step`, `pwl_smoothing.py:347–348`, so this should hold.)
- **Zero delay:** assert `_pwl_all_zero_delay` (guaranteed by delay-fold, `:2312`).
- **`t_start` phase offset:** general `j_k = round(((t_start + (k+1)·dt) % P)/dt)`; with `t_start=0`
  it is `(k+1) mod m`.
- **Periodic wrap seam:** building columns via the real evaluator bakes the `after_periodic` wrap
  (`vectorized_sources.py:657`) into column `m-1 → 0` — no special handling.
- **DC / constant-hold sources:** fold into a single constant added to every column (or add once per step).

### 4.6 Validation

- **Unit equivalence:** for random `k`, assert `C[:, k mod m] == evaluate_at_time(t_start+(k+1)·dt)`
  to fp tolerance (~1e-12, since columns *are* evaluator outputs).
- **End-to-end:** `solve_transient` peak IR-drop / tracked waveforms match the pre-change run on
  `netlist_sampled` and `netlist_test`. Existing coverage: 68 tests in
  `tests/analysis/test_pwl_smoothing.py` (the `TestSparseVsDenseEquivalence` backbone) + distributed
  transient validation in `tests/distributed/test_time_domain*.py`.

### 4.7 Step 0 — probe before implementing (selects Tier 1 vs 2, sizes `m`)

Run against one smoothed worker's `VectorizedCurrentSources`:

```python
print(np.unique(vcs.pwl_period))          # one value P? -> Tier 1
P = ...; dt = 5e-12
print(P/dt, round(P/dt))                   # integer & small?
print(bool(np.all(vcs.pwl_delay == 0)))    # zero-delay fast case?
print(int(vcs.pwl_count.max()), vcs.n_pwls)# column-build cost / memory
```

The BRCM `max_drop` signature repeats on a fixed cadence over the 50 ns window (the periodic-stimulus
fingerprint), so Tier 1 is expected — but confirm with the probe rather than assume.

---

## 5. Companion fixes 1, 2, 4 (cheap, ship alongside Fix 3)

- **Fix 1 — hoist constant DC out of the loop** (`vectorized_sources.py:405`): precompute `current_dc`
  once, add it; stop re-scattering `dc_values` every step. (Subsumed by Fix 3's column bake, but valid
  independently.)
- **Fix 2 — `np.add.at` → `np.bincount`** for scatters (`:405`, `:419`, `solver_td.py:573`). Same
  pattern already used at `vectorized_sources.py:1068`.
- **Fix 4 — build the padded/binned cache at setup**, not lazily on the first solve step
  (`vectorized_sources.py:684`/`:757`) — removes the first-step spike from the timed loop.

---

## 6. Prepare-phase & structural opportunities (from the same logs)

### 6.1 Interface factor backend is mis-set in the decompose run (config bug)

Identical interface system (70,734 unknowns, 479 M nnz, 9.58 % density, 5.4 GB), two backends:

| Run | Interface backend | DC / transient factor |
|---|---|---|
| transient | `cholmod(supernodal, metis)` | **74 s / 79 s** |
| decompose | `cholmod(simplicial, metis)` | **1668 s / 1760 s** |

A **22× regression**, ~3275 s (~55 min) of avoidable prepare in the decompose run. Switch the
coordinator/interface factor to **supernodal** (`cholmod_mode`; propagated to workers via
`TileWorker.configure`, see `src/distributed/CLAUDE.md`). Caveat: supernodal needs more peak RAM — if
simplicial was a memory-driven fallback, release per-tile `G_ii`/Schur before the interface factor or
provision more RAM.

### 6.2 Per-tile CHOLMOD settings differ — neither run is optimal

| Run | Per-tile backend | factor min/mean/max |
|---|---|---|
| transient | `idx=32-bit, ordering=amd` | 8 / 373 / **1234 s** |
| decompose | `idx=64-bit, ordering=default` | 6 / 92 / **268 s** |

Decompose's per-tile config is **4.6× faster**. Unify on **decompose's per-tile settings +
transient's supernodal interface** — the best of each.

### 6.3 Cache the partition-static work

- **smooth_sources (2700–2842 s):** in the decompose run the VCS was cached ("36 from cache") yet
  smoothing re-ran for 2842 s. Persist the smoothed waveforms keyed on (sources, smoothing params);
  also confirm smoothing is parallelized across the 36 workers (230 µs/source suggests partly serial).
- **Detect islands (401–404 s, every prepare):** topology is fixed per partition set — cache and
  reuse across DC and transient prepare (and across runs).

### 6.4 Per-tile load balance

Interior nodes 201 k → 1.6 M (8×); factor 6 s → 268 s (42×). Both the factor wall and the per-step RHS
straggler are gated by the single densest tile — adding workers won't help. Use **density-aware
partitioning** (balance interior size for factor, current-source count for RHS) instead of the uniform
6×6 geometric grid.

### 6.5 Phase 3 re-runs the entire transient (decompose-only)

Phase 2b runs the full 10,000-step transient (12.7 h); Phase 3's "all-sources transient (tracking 10
victims)" then ran **~11.5 h just to reach Victim 1/10** — re-solving essentially the same trajectory
to record waveforms at 10 nodes. Capture victim/candidate waveforms **during** the Phase 2b pass, or
fold near/far decomposition into a single forward solve, instead of re-running the loop per stage.
(Fix 3 also makes each such pass far cheaper.)

---

## 7. Status tracker — landed fixes vs BRCM projection

> **Measurement note**: BRCM netlist (`brcm_transient.log`, 30.67 M nodes, 36 tiles, 10 K steps @ 5 ps) is **not available on this host**. netlist_sampled deltas are measured; BRCM projections use the plan arithmetic from the baseline data. BRCM re-measurement is pending bundle access.

| Fix # | Description | Commit | Branch | netlist_sampled measured delta | BRCM projected saving | Status |
|--------|------------|--------|--------|-------------------------------|----------------------|--------|
| 1 | `bincount` scatter (Fix 2) + hoist DC constant (Fix 1) | `f72bd88` (A1) | distributed-10x | loop_total –31% vs pre-A baseline | ~10,000–14,000 s (array exchange + bincount) | **Landed** |
| 2 | Phase-folded RHS precompute (`use_step_columns`, Fix 3 Tier 1) | `7fb92bc` (A2) | distributed-10x | rhs/step ≈ 28 ms vs ~276 ms pre-A (10×); loop –31% accumulated with Fix 1 | ~29,000 s (3.3 s/step → ~0.4 s/step straggler backsolve) | **Landed** |
| 3 | CHOLMOD supernodal guardrails + per-actor threads (A3) | `d79ca2c` | distributed-10x | dc_prepare –12% | ~3,275 s (supernodal trap removal) + ~1,000 s tile factors | **Landed** |
| 4 | Symbolic + assembly-pattern reuse DC→transient (A4) | `0f4ac88` | distributed-10x | trans_prepare –70% vs pre-A baseline | ~1,500–1,800 s of 2,459 s transient prepare | **Landed** |
| 5 | Smoothed-VCS disk cache + `smooth='auto'` (A5) | `0f4ac88` | distributed-10x | smooth ≈ 0.17 s first / ~seconds cached (vs ~150 s uncached on netlist_sampled) | ~2,700 s first run → ~30 s cached | **Landed** |
| 6 | Island detection cache (A7) | `0f4ac88` | distributed-10x | detect_interface_islands shared across DC+transient prepare | ~400 s/run (404 s × 2 in baseline) | **Landed** |
| 7 | Decomposition: capture victims in main sweep, drop Phase-3 redo (A6) | `df256df` | distributed-10x | N/A (decompose workflow; not in per-step perf JSON) | ~12 h/decompose run eliminated | **Landed** |
| 8 | Balanced retiling via parser-side tile splitting (B1) | `80fae0c` | distributed-10x | not yet measured separately on netlist_sampled | ~1,240 s tile factor wall → ~300 s; straggler backsolve –70% | **Landed** |
| 9 | Iterative interface CG + block-Jacobi preconditioner (B2) | `e66f65a` | distributed-10x | auto selects direct for n_interface < 200K on netlist_sampled | removes ~200 GB coordinator memory wall at 1 M+ interface nodes | **Landed** |
| 10 | Streaming Schur shard assembly (B3) | `9818280` + `10695f5` | distributed-10x | peak 3.5 MB vs 15.6 MB bulk on netlist_sampled; auto at 512 MB+ estimated S_i peak | caps coordinator peak memory; enables 100M+ node PDNs | **Landed** |
| 11 | Multi-node task-dataflow design + `TaskDataflowBackend` prototype (B4) | `f25d209` | distributed-10x | DC actor 0.858 s vs task 0.797 s, max \|ΔV\| = 0.0 V (exact) | enables k-machine deployment (see §7.1); actor mode remains single-node default | **Landed (prototype)** |
| 12 | Step-column table reuse across transients + chunked direct-scatter windows (minion plan, see §7.2) | `11478ce` + `5e8182e` + `3d7abcc`; review-hardened by `a147883` + `5319dd1` + `b0c781b` | distributed-10x | loop_total –25 to –31% vs checked-in baseline; results exact (peak diff 0.0000 mV) | netlist_minion: kills the 314 s/solve rebuild ×6 solve_transient calls per decompose → `initial_transient` 394.6 s → ~85–90 s, total 1,117 s → ~700–750 s projected | **Landed** |

### 7.1 B4 findings — multi-node task dataflow (full analysis: `docs/multinode_task_dataflow_design.md`)

Measured on this host (48-CPU, single machine; 2-node runs use resource-labeled virtual nodes —
optimistic for network transfer, documented as such):

- **CHOLMOD/SuperLU factors are NOT picklable** (experimentally confirmed) — the central constraint.
  Factors must stay pinned in the worker process. Serializing triangular factor arrays instead
  degrades per-step solves 3–10× (**~+21,000 s on the BRCM transient**) — rejected. Refactor-on-
  session-start with process-resident factors is the only viable multi-node persistence strategy.
- **Tile-pkl distribution via the Ray object store works and is cheap**: 12.3 ms/tile `ray.put`
  (~108 ms for 36 BRCM-scale tiles) — removes the shared-NFS assumption. Recommended
  **unconditionally**, independent of solve mode.
- **Per-step task-submission overhead bounds task-mode transients**: 6.1 ms/step (task) vs
  4.0 ms/step (actor) at 9 tiles; at 250 tiles × 2 barriers ≈ 95 ms/step → **~950 s of pure
  scheduling over 10 K steps** (~13 % of the ~7,400 s end-to-end target). Actor mode therefore
  remains the default for single-node transient loops.
- **Task mode wins**: (a) multi-node DC prepare (factor tasks placed by
  `NodeAffinitySchedulingStrategy` where their tile pkl lives), (b) stateless CG tilewise-matvec
  tasks once `n_interface` > ~500 K (per-tile `S_i` IS picklable and object-store cacheable).
- **100M-node / 4-machine arithmetic**: ~250 tiles at `max_interior=400 K` → ~36 GB of CHOLMOD
  factors per machine (needs 64–96 GB nodes); the interface system itself is small (~26 MB at
  160 K interface nodes under CG).
- **Prototype validation**: `TaskDataflowBackend` DC prepare+solve on netlist_sampled is
  algebraically exact vs actor mode (max |ΔV| = 0.0 V) at parity wall time (0.797 s vs 0.858 s).

### 7.2 Minion decompose findings — A2 rebuild regression (plan: `logs/ir-decomposition-speed-up-plan.md`)

The `netlist_minion` decompose run (`logs/decompose_20260710_110053.log` vs pre-refactor
`logs/decompose_20260512_182310.log`) regressed **857 s → 1,117 s** despite the transient loop
getting 2.3× faster (0.449 → 0.198 s/step). Root causes and dispositions:

- **A2 step-column rebuild per `solve_transient` call (fixed, Fix 12)**: a decompose run calls
  `solve_transient` ~6× on the *same* smoothed sources; each call paid a full chunked-tier build
  (314 s under Ray — thread-capped actor + multi-GB `evaluate_at_times_for_rows` intermediates;
  75 s isolated). The phase tier (1.7 ms/step gather) was disqualified purely by memory:
  `est_table_mb(m=1000) = 1,488 MB > max_table_mb = 512`. Three additive fixes landed:
  **Change A** — worker-side table cache keyed on a sources-version counter + `(dt, t_start,
  max_table_mb)`; phase tables additionally reusable across any dt-grid-aligned `t_start`
  (phase0 recomputed cheaply). **Change B** — direct-scatter fast path for chunked *window*
  builds (index gather from the smoothed uniform-grid PWL arrays, wrap-at-m convention); the
  memory-safe version of the phase-tier win, keeps `max_table_mb` at 512. **Change C** — skip
  the precompute when a single-window chunked build cannot amortize and no fast path applies
  (per-step `evaluate_at_time` is 0.97× of a single-window build without the multi-GB
  intermediate).
- **Vectorizing `_evaluate_pwls_batch` (Option 1) was benchmarked and REJECTED**: 0.54×
  (slower) — smoothed PWLs are compacted to ~5.5 knots, so the per-row loop is near-optimal and
  the 3-D broadcast's memory traffic dominates.
- **Out of scope, tracked**: QS victim pre-selection mispredicted → +44.7 s QS + 139 s targeted
  transient (Phase 2 decomposition follow-up); cold smoothed-VCS cache +55 s (one-time,
  recovers on re-run).

Projected minion re-run: `initial_transient` 394.6 s → ~85–90 s (one cheap build + 79 s loop);
the 5 subsequent transients no longer pay builds; total ~700–750 s vs 1,117 s.

A follow-up xhigh adversarial review of the A/B/C changes confirmed 15 findings (4 silent
wrong-result bugs on input shapes the fixed netlists never exercise), all fixed in `a147883` +
`5319dd1` + `b0c781b`: the smoothed-grid probe is now a full vectorized per-row eligibility check
(non-uniform / partially-compacted rows disable the fast path instead of gathering wrong values —
on heavily compacted VCS like minion's the fast path correctly self-disables and Change A reuse
carries the win); the stale-window `n_src` capture on reuse is fixed at root; `apply_wscale` is in
the cache key; skips no longer evict a valid cached table (active/cached slot split); identical-
params smoothed-cache re-hits no longer bump the sources version; negative on-grid `t_start`
(the QS convention `t_col_start = -dt`) is accepted by both tiers; short-then-long reuse rebuilds
with proper W. A 40-test guard-matrix suite
(`tests/distributed/test_step_column_guard_matrix.py`) now pins the tier-selection guards and
cache-validity checks across these input shapes.

### 7.3 First BRCM re-measurement (2026-07-10 run, `logs/brcm_transient_20260710_221705.log`)

First post-refactor run on the real BRCM bundle (36 tiles unchanged — **B1 retiling NOT enabled**;
smoothed-VCS cache cold 0/36; `threads_per_worker` not set). Results **bit-identical** to baseline:
peak 104.945 mV at t=35.575 ns, same top-10 nodes — exactness holds at 30.67M nodes.

| Phase | Baseline | Re-run | Δ | Notes |
|---|---|---|---|---|
| Model load | 47 s | 85 s | +80% | host-side |
| VCS init | 181 s | 224 s | +24% | host-side |
| Smoothing | 2,697 s | 3,616 s | +34% | cold cache (0/36); straggler-bound: max tile 3,614 s ≈ wall, mean 549 s |
| DC prepare | 2,087 s | 1,933 s | −7% | assemble 369→149 s (A4); islands 404→414 s; tile factor wall 1,240→1,286 s (no B1) |
| Transient prepare | 2,459 s | 1,431 s | **−42%** | A4 pattern reuse (assemble ≈0) + A7 islands 0 s; tile numeric refactor remains ~1,250 s (simplicial ⇒ symbolic reuse saves little) |
| A2 step-column build | — | 575 s | new | all 36 tiles → **chunked** (m=2000; straggler table ~20 GB ≫ 512 MB cap) |
| Loop (10K steps) | 61,430 s (6.143/step) | 55,704 s (5.570/step) | −9% | see decomposition below |
| — RHS | 3.313/step | 1.798/step | −46% | steady ≈0.9/step (straggler interior backsolve) + **19 window-rebuild spikes ≈500 s each ≈9,500 s** (every 512 steps; compacted smoothed rows make the direct-scatter gather ineligible, so rebuilds pay `evaluate_at_times_for_rows`) |
| — Assemble+solve | 1.688/step | 2.858/step | **+70%** | same factor (supernodal/METIS, 70,734 unknowns, factor 74→83 s); segment actually *shrank* (assembly moved to RHS timer) ⇒ backsolve itself slower — environmental (all host-side phases +24–80%; also A2 tables + 5 GB/window rebuild allocations pressure memory bandwidth). DC-IC solve identical (34.2 vs 34.0 s) |
| — Recovery | 1.047/step | 0.913/step | −13% | straggler-gated |
| **End-to-end** | **68,940 s** | **63,605 s** | **1.08×** | |

**Why only 1.08× instead of the projected ~4×:** (1) B1 retiling was not applied — the 1.6M-interior
straggler still gates RHS steady-state, recovery, tile factor, transient refactor, and smoothing wall
(the plan's arithmetic REQUIRES B1 for tile-side terms); (2) cold smoothing cache (+3,616 s, one-time);
(3) A2 chunked rebuild spikes (+9,500 s) — BRCM's m=2000 phase table cannot fit and compacted rows
disqualify the gather; (4) the coordinator backsolve regression (+11,700 s vs baseline) appears
environmental, not algorithmic.

**Next actions (ranked):**
1. **Re-parse with `--max-interior 400000` (B1)** — splits the straggler; expected: RHS steady 0.9→~0.3,
   recovery 0.9→~0.35, tile factor 1,286→~400 s, transient refactor ~1,250→~400 s, smoothing straggler
   3,614→<900 s, rebuild spikes shrink and parallelize.
2. Re-run as-is gets smoothing ≈ free (cache 36/36).
3. New item — **overlap chunked window prefetch** with the time loop (rebuilds are per-tile independent;
   ~9,500 s hideable behind the interface solve barrier).
4. Interface solve remains the post-B1 floor (~17–29 K s/10K steps): threaded/multi-RHS backsolve or
   warm-started CG (`--interface-solver cg`, block-Jacobi, 70,734 unknowns @ 9.9% density) is the
   remaining lever the plan always attributed the last ~2× to. Verify host BLAS threads/contention on
   the next run before concluding anything algorithmic.

With B1 + warm cache + unchanged interface solve at baseline speed, projected ≈19–21 K s (~3.4×);
reaching ~10× still runs through the interface-solve line, as §7's cumulative table always showed.

### 7.4 B1 400K-split re-measurement — interface system explodes (2026-07-12/13 runs, 20 steps)

Setup: BRCM re-parsed with `--max-interior 400000` → **107 tiles**, interior 197,977 / 285,540 /
376,298 (min/mean/max) — balance fully achieved vs the old 201K / 852K / 1.6M. Two 20-step transient
runs (dt=5ps, t_end=0.1ns): **direct** interface solve (`logs/brcm_transient_20260713_102026.log`,
forced `--interface-solver direct`) and **CG** (`logs/brcm_transient_20260712_181719.log`, `auto`
resolved to CG: estimated factor 47.8 GB > the 32 GB `AUTO_CG_FACTOR_MEMORY_BUDGET_BYTES`).
Both runs bit-identical to each other (peak 104.932 mV @ t=0.095 ns, same max_drop at every logged
step) — exactness holds through the 107-tile split *and* through CG at rtol 1e-12.

| Metric | 36-tile (§7.3) | 107-tile direct | 107-tile CG |
|---|---|---|---|
| Interface unknowns | 70,734 | 190,867 | 190,867 |
| S_global nnz / density / size | 493.5M / 9.9% | 1,282.1M / 3.5% / 14.3 GB | same |
| Interface factor | 83 s (supernodal/METIS) | 297 s DC + 313 s transient | n/a (CG setup 12–14 s) |
| Detect islands (DC) | 414 s | 900 s | 882 s |
| Tile factor wall | 1,286 s | 433 s | 401 s |
| Smoothing | 3,616 s cold, straggler 3,614 s | warm (0 s) | **cold 254 s wall** (max tile 253 s) |
| RHS /step | 1.798 (steady ≈0.9) | 0.885 (per-tile max 1.35) | 0.939 |
| Assemble+solve /step | 2.858 | **11.537** | **329.1** |
| Recovery /step | 0.913 | 0.609 | 0.594 |
| Loop /step | 5.570 | **13.034** | **330.6** |
| DC initial condition | 34 s | 21 s | 641 s (one cold CG solve) |
| 10K-step loop extrapolation | 55.7 K s | ~130 K s → **~134 K s total (0.51×)** | ~3.3 M s (non-starter) |

**What B1 delivered, exactly as designed (tile-side):** per-step RHS+recovery
4.36 (baseline) → 2.71 (36t) → **1.49 s/step**; tile factor wall 1,286 → 433 s; transient tile
refactor similarly down; cold smoothing wall 3,616 → 254 s (the straggler is gone: per-tile max
253 s vs 3,614 s). Every tile-side projection in §7.3 action 1 was met or beaten.

**What broke: the interface system, structurally.** Unknowns 70,734 → 190,867 (2.7×); S nnz
493.5M → 1,282.1M (2.6×). Σ n_ports² over the 107 tiles = 1,561M — the assembled S is essentially
the union of the *dense per-tile port blocks* (overlap dedupe only 1.22×). Every cut plane adds
ports, and Schur complements are dense in the ports, so S nnz grows superlinearly with splitting.
Downstream:
- **Direct:** factor ≈ 47.8 GB (vs ~12 GB implied at 36 tiles). The backsolve is memory-bandwidth
  bound: 2.858 → 11.537 s/step = 4.0×, matching the factor-size ratio almost exactly.
- **CG:** block-Jacobi at rtol 1e-12 needs ~175–190 iters/step even warm-started (330 cold), and
  each iteration is one single-threaded CSR matvec over the 14.3 GB assembled S ≈ 1.75 s →
  329 s/step. Warm start helps only 1.8× because the tolerance is validation-grade.
- Detect islands doubled (414 → 900 s) — scales with interface size; still uncached across runs.

**Interpretation — per-step cost is U-shaped in tile count.** Tile-side terms fall with splitting;
the interface term (unknowns × density × bandwidth-bound solve) grows faster. At max-interior 400K
the interface line dominates everything and the end-to-end is **2× worse than baseline**. §7.3's
conclusion is now a measurement: the interface solve is the floor — even at 36 tiles it is
2.86 s/step = 28.6 K s of a ~6.9 K s 10× budget. **No max-interior setting reaches 10×; the
interface solver must change.**

**Next actions (ranked):**
1. **Re-parse at `--max-interior 1000000`** (splits only the 1.6M straggler + a couple more,
   ~40 tiles). Expected: RHS ~0.5, recovery ~0.6, interface unknowns +10–20% over 70,734 →
   backsolve ~3.2–3.5 → loop ~4.3–4.6 s/step ≈ 43–46 K s (~1.5–1.6×), plus most of the B1
   prepare/smoothing wins. Best available configuration without interface work; also the right
   baseline for measuring item 2.
2. **Interface solve engineering — the only path to 10×** (target ≤0.3–0.5 s/step):
   (a) CG `--interface-matvec-mode tilewise` moved worker-side: the per-tile dense `S_i @ x` is a
   BLAS GEMV, Ray-parallel across 107 workers (~tens of ms/matvec vs 1.75 s assembled), but the
   per-iteration round-trip must be batched — likely needs a fused "run k iterations worker-side"
   or coordinator-side dense blocks; (b) relax `--interface-cg-rtol` to 1e-8/1e-6 with an accuracy
   study (1e-12 proved bit-identical — there is headroom); (c) **two-level preconditioner** (coarse
   one-node-per-tile Galerkin space, already sketched in B2): block-Jacobi alone plateaus at ~180
   iters; a coarse space typically cuts DDM interface CG to tens; (d) for direct: threaded /
   multi-RHS CHOLMOD backsolve — both the 11.5 s and the 36-tile 2.86 s are single-RHS
   bandwidth-bound.
3. **Persist island detection with the pkl bundle** (partition-static; 900 s/run at 107 tiles,
   invalidate on re-parse only). A7 already makes the transient reuse free; the DC one still pays
   full price every run.
4. Make the direct-path factor-memory budget host-aware/configurable
   (`AUTO_CG_FACTOR_MEMORY_BUDGET_BYTES` = 32 GB hardcoded in `interface_iterative.py`; the host
   factored 47.8 GB without trouble when forced).

Caveat: 20-step runs skip the A2 table build for most tiles (Change C guard: n_steps ≤ 512 →
tiers mostly 'skipped', 7 'phase'), so full-run RHS would additionally pay/amortize chunked-window
builds — second-order next to the interface line.

### 7.5 Sampled-BRCM proxy evaluation (2026-07-16, `netlist/netlist_brcm_sampled/`)

Question: is the 10×-node-sampled BRCM PDN (3.08M nodes, same 36-tile layout, straggler tile (2,3)
preserved) a valid performance proxy — i.e., does a 10× speedup on it predict a 10× on real BRCM?
Measured: parse (13 s) + transient solve 2000 steps BE dt=5ps (`logs/brcm_sampled_transient_20260716_164303.log`,
default solver config, Ray).

**Answer: not in its current form.** The sampler reduced nodes 10× but **resistors 26×**
(R/node 1.87 → 0.72 — below the ≥1.0 needed for connectivity), so the graph fragments: **83% of
sampled nodes were dropped as resistively floating** (10,899 global islands, 11,034 penalized),
leaving 514K interior of the intended 3.08M (an effective 60× reduction, not 10×), 242K of 1.24M
current sources, and non-physical electrical results (max_drop 1.29 V on a 0.76 V rail;
total_I ~2% of BRCM's). Fragmentation also destroys the dense port-block structure of the tile
Schur complements, which is precisely the structure that makes the interface solve the BRCM
bottleneck.

| Metric | BRCM 36-tile (re-run) | Sampled | Ratio |
|---|---|---|---|
| Interior nodes (kept) | 30.67M | 514,151 | 60× |
| Interface unknowns | 70,734 | 20,695 | 3.4× |
| S_global nnz / density | 493.5M / 9.9% | 2.2M / 0.51% | **224×** |
| Interface factor | 83 s | 0.19 s | 437× |
| Tile factor wall | 1,286 s | 0.16 s | ~8,000× |
| Smoothing (cold) | 3,616 s | 24.4 s | 148× |
| DC / transient prepare | 1,933 / 1,431 s | 5.9 / 3.1 s | 330 / 460× |
| RHS /step | 1.798 (32%) | 0.048 (**51%**) | 37× |
| Assemble+solve /step | 2.858 (**51%**) | 0.005 (**5%**) | **572×** |
| Recovery /step | 0.913 (16%) | 0.040 (43%) | 23× |
| Loop /step | 5.570 | 0.094 | 59× |

**The bottleneck profile is inverted.** On BRCM the interface backsolve is 51% of the step — the
entire remaining 10× problem (§7.4). On the sample it is 5%; RHS+recovery are 94% — the phases B1
already fixed. Consequences: an interface-solve optimization (lockstep multi-RHS, CG coarse space,
threaded backsolve) would measure ≈1.05× on the sample but ~2× on BRCM; the B1 U-shape (§7.4)
cannot be reproduced at all because splitting fragmented tiles does not densify the interface.
Nothing tuned on this testcase transfers; nothing that matters on BRCM is measurable on it.

**What a valid proxy requires (sampler fixes, in order):**
1. **Preserve R/node ≈ 1.9** — reduce nodes by *contraction/coarsening* (merge nodes, combine
   conductances — the R_eff-preserving analogue of `netlist_sampled`) rather than independent
   element dropping. Acceptance: ~0 floating nodes after parse (`Islands penalized ≈ 0`),
   kept-interior ≈ sampled-node count.
2. **Preserve interface structure** — boundary sampling was actually reasonable (138K → 69.5K
   pre-island; cut planes are 2-D so ~2–3× is expected); what matters is that ports stay densely
   coupled through tile interiors so S density returns to ~10% (target S nnz ≈ 50–120M, interface
   factor seconds-not-milliseconds).
3. **Acceptance test for the proxy itself**: per-step shares within ~±10 points of BRCM's
   32/51/16 (RHS/solve/recovery), max_drop < Vdd, total_I ≈ scaled BRCM.
Until then, the cheap-and-faithful alternatives are what §7.4 already used: short (20-step) runs on
the real BRCM bundle (~50 min wall, exact per-step structure) plus netlist_sampled for regression
gating.

### 7.6 Contraction-sampled proxy re-evaluation (2026-07-17, `netlist/netlist_brcm_sampled/` regenerated)

The sampler was rewritten to contraction/coarsening (2f60c46 + 795094d review fixes) and re-run on
the BRCM host against the real 30.8M-node PDN: nodes 30,812,194 → 8,275,709 (3.72×), resistors
57.6M → 31.2M, sources 12.4M → 1.24M (10×), R/node 1.868 → **3.765**, `isolated_optional_dropped=0`,
`islanded_reps_dropped=0`, pads 309/309. Measured here: parse (17.7 s, zero island warnings) +
transient 2000 steps BE dt=5ps, default config, Ray
(`logs/brcm_sampled2_transient_20260717_002043.log`, end-to-end 1,778 s).

**Answer to §7.5's question: yes, with per-phase projection.** The fragmentation failure is gone
and — the decisive property — **the interface system is reproduced exactly**: boundary, pad, and
die-attachment nodes are mandatory (never contracted), so the proxy has the *same* 70,734 interface
unknowns and the *same* S structure (493.5M nnz, 9.864% dense, 5.5 GB) as full BRCM. The
interface-solve problem being optimized for the remaining 10× is bit-identical in structure, with
realistic values from merged conductances.

| Metric | BRCM 36-tile (re-run §7.3) | Contraction proxy | Ratio | §7.5 old sampler |
|---|---|---|---|---|
| Interior nodes (kept) | 30.67M | 8.14M (98.5% of sampled) | 3.8× | 514K (17%) |
| Tile interior min/mean/max | 201K/852K/1.6M | 70K/226K/552K | ~3.7× | — |
| Interface unknowns | 70,734 | **70,734** | **1×** | 20,695 |
| S_global nnz / density | 493.5M / 9.9% | **493.5M / 9.864%** | **1×** | 2.2M / 0.51% |
| Interface factor (supernodal) | 83 s | 27.9 s DC / 30.1 s tr | 2.8–3.0× | 0.19 s |
| Islands penalized | 0 | 0 | — | 11,034 |
| Tile factor wall | 1,286 s | 15.7 s (max tile 9.6 s) | 82× | 0.16 s |
| Smoothing (cold) | 3,616 s (straggler 3,614) | 57.4 s (straggler 57.3) | 63× | 24.4 s |
| DC / transient prepare | 1,933 / 1,431 s | 316 / 96 s | 6.1 / 14.9× | 5.9 / 3.1 s |
| RHS /step | 1.798 (32%) | 0.232 (37%) | 7.8× | 0.048 (51%) |
| Assemble+solve /step | 2.858 (**51%**) | 0.156 (**25%**) | 18.3× | 0.005 (5%) |
| Recovery /step | 0.913 (16%) | 0.237 (38%) | 3.9× | 0.040 (43%) |
| Loop /step | 5.570 | 0.626 | 8.9× | 0.094 |
| 10K-step extrapolation | 63,605 s | ~6,790 s | 9.4× | — |
| Peak drop | 104.9 mV @ 35.6 ns | 76.2 mV @ 6.6 ns (10 ns window) | physical | 1.29 V (> rail) |

**Acceptance criteria from §7.5:** (1) islands ≈ 0 — **met** (0 penalized, 0 parse warnings,
98.5% of sampled nodes survive as interior vs 17% before). (2) Interface structure — **exceeded**
(exact, not approximate). (3) R/node ±30% of 1.87 — missed high (3.765): contraction merged nodes
3.72× but inter-cluster resistors only 1.85×, so the coarse graph is ~2× denser per node. This errs
in the safe direction (no fragmentation) but makes tile factor/recovery relatively costlier per
node. (4) Per-step shares ≈ 32/51/16 ±10 pts — RHS 37% met; solve 25% / recovery 38% skewed toward
the tile side.

**Cross-host confound (all full-BRCM numbers come from the BRCM host; the proxy runs here):** the
one identically-structured operation measured on both — the supernodal interface factor — runs
2.8–3.0× faster here (27.9 s vs 83 s). The per-step backsolve gap (18.3×) exceeds that anchor
because the backsolve is bandwidth-bound and gains more from this host than the GEMM-bound factor;
treat absolute per-step times as host-specific and compare **shares and per-phase ratios only**.

**How to use the proxy (projection rule):** every phase is now present at ≥25% of the step, so any
optimization is measurable — but end-to-end loop speedup on the proxy *under*-predicts BRCM for
interface work and *over*-predicts for tile work. Measure the per-phase improvement factor k on the
proxy, then project onto BRCM's shares: e.g. an interface-solve k× gives at most 1.33× on the proxy
loop (share 25%) but 2.04× on BRCM (share 51%). The structural exactness of S means interface
findings (matvec modes, s-step CG, coarse-space preconditioners, multi-RHS backsolve) transfer
directly; iteration counts of CG-type methods use realistic contracted conductance values, though
spectrum-sensitive results should be spot-checked on a 20-step real-BRCM run before landing.
Practical win: one optimization iteration costs ~30 min wall on the proxy vs ~18 h on BRCM.

Secondary confirmations: island detection is 238 s = 75% of the proxy's DC prepare — ranked
action (3) of §7.4 (persist islands with the pkl bundle) is now the top prepare-phase item on both
netlists. The smoothing straggler shape survives sampling (one tile = 99.8% of the wall in both),
so straggler-oriented smoothing work is also testable here.

### 7.7 Interface-solve acceleration — Stage 0 baselines & microbenchmarks (2026-07-17, dev host)

Measurement campaign for the interface-solve plan (`plans/interface_solve_acceleration_plan.md`).
Dev host: 48 cores, 251 GB RAM, RTX 6000 Ada 48 GB. Proxy bundles parsed from
`netlist/netlist_brcm_sampled` (§7.6): `distributed_pkl_mi100k` (116 tiles, 355,623 shared
boundary) and `distributed_pkl_mi200k` (64 tiles, 167,493 shared boundary → n_interface 167,659
with die nodes — the closest local analog of the BRCM 107-tile/190,867 split regime). Scripts in
`scripts/benchmark/microbench/`; raw JSON: `results_matvec_mi200k.json`, `results_gpu_mi200k.json`.

**Finding 0 (blocking, feeds Stage 2): the non-streaming S_i gather + COO assembly does not fit
this host at the split regime.** Two attempts at a production CG-tilewise `prepare()` on mi200k
were watchdog-killed: the coordinator ballooned to >190 GB driver RSS (dense S_i retained + COO
triplets at 24 B/entry + CSR-conversion temporaries) while the 16 packed workers sat at a healthy
4–8 GB each. A B3-style streaming gather of the same blocks (slice-and-copy per tile, one shard in
flight) peaks at **26.5 GB** — the D2 direct-stamping fix plus streaming/tilewise-without-S_global
is a hard requirement for the 107-tile BRCM run, not an optimization. mi100k additionally OOM'd
with 116 *unpacked* concurrent tile factors; `tiles_per_worker=4` (16 actors) kept factor pressure
trivial. Also measured: no tile-resident pad ports in this bundle (D1 doesn't bite here; it remains
latent for netlists with pads on tile ports).

**Ray RTT at 116 actors** (`bench_ray_rtt.py`, Stage 5 go/no-go): broadcast 1.5 MB + gather
p50 ≈ **15.9 ms**, per-actor 30 KB slices p50 ≈ 16.2 ms — latency-bound, not payload-bound.
Per-iteration worker-side matvec would pay ~16 ms/iteration of pure RTT on top of compute; on one
box it adds zero aggregate bandwidth, so Stage 5 stays gated on multi-node need.

**CPU matvec kernels at n = 167,659** (mi200k, sum n_p² = 2.34B, dense blocks 18.7 GB, est.
S nnz ≈ 2.0B — denser than BRCM's 1.28 B at 190K; BLAS pinned to 1 thread, thread pool only):

| Kernel | mean | note |
|---|---|---|
| CSR SpMV, 1 thread (synthetic, matched nnz) | 1384 ms | current assembled-mode cost |
| tilewise serial (production math) | 570 ms | |
| tilewise threaded, 8 threads | **150 ms** | best; ~125 GB/s effective |
| tilewise threaded, 16 / 32 / 48 | 172 / 200 / 225 ms | inverted scaling — accumulator zero-fill + reduction grow with thread count |
| tilewise fp32 threaded, 48 | (1597 ms — invalid) | mixed fp32·fp64 GEMV fell off the BLAS path; redo in Stage 2 if fp32 promoted |
| block-Jacobi apply, serial (as production) | 990 ms | 6.8 GB/s — Stage 2 must thread it (~120 ms projected) |
| matmat, 65 cols, threaded (S·Z coarse setup) | 1272 ms | one-time per factored context — within Stage 3 budget |
| STREAM triad 1t / 48t | 6.5 / 56.9 GB/s | ceiling reference (NUMA-affected) |

Stage 2 CPU projection: ~150 ms matvec + ~120 ms threaded BJ ≈ **0.27 s/iteration** → even at 15
warm iterations ≈ 4 s/step on dev — ~3× above the plan's CPU-path estimate (the plan's 45–60 ms
floor assumed higher effective bandwidth). The CPU path remains the mandatory fallback but is
firmly off-target; the iteration-count cut (Stage 3) and the GPU matvec (Stage 4) are *jointly*
load-bearing.

**GPU kernels, RTX 6000 Ada** (`bench_gpu_matvec.py`, synthetic shapes matched to mi200k):

| Kernel | mean | effective BW |
|---|---|---|
| device CSR SpMV fp64 | **26.8 ms** | 1188 GB/s |
| device CSR SpMV fp32 | 17.9 ms | 1335 GB/s |
| batched dense GEMV fp64 (16 buckets) | 25.3 ms | 886 GB/s |
| batched dense GEMV fp32 | 12.8 ms | 879 GB/s |
| batched BJ apply fp64 (padded inverses) | 16.1 ms | 218 GB/s — unoptimized; ~4 ms plausible bucketed |
| H2D + D2H of CG vector | 0.31 + 0.11 ms | negligible per solve |

Stage 4 layout decision confirmed: device CSR ties batched dense (within 6%) → **device-resident
assembled CSR fp64**, far simpler. GPU PCG iteration ≈ 30–45 ms → target ≤0.2 s/step on dev needs
warm iterations ≤ ~5–10, i.e. the Stage 3 coarse space + warm start must deliver its stretch goal;
whole-loop-on-device (BJ inverses + coarse solve on GPU) avoids the 0.4 ms/iteration transfer tax
entirely. fp64 fits comfortably (~24 GB of 48 GB), so no fp32 accuracy caveats on this card.

**rtol sweep** (`run_rtol_sweep.py`, 36-tile bundle `distributed_pkl` — production prepare fits
there; the rtol→error curve is spectrum-driven, so the default choice transfers; split-regime
iters/step re-measured at Stage 2 gates). 20-step BE dt=5ps transient, CG assembled + block-Jacobi
(4 GB cap raised to 32 GB — the default would have silently degraded to plain Jacobi), warm-start
reset per run, errors = max|ΔV| over 400 tracked nodes (200 worst + boundary sample) vs the direct
reference (direct: loop 12.9 s, peak 76.1606 mV @ 0.095 ns; raw JSON `results_rtol_36t.json`):

| rtol | iters/step (warm) | max\|ΔV\| vs direct | peak-drop Δ | peak node |
|---|---|---|---|---|
| 1e-12 | 130.3 | 1.9e-11 V | −8e-10 mV | match |
| 1e-10 | 86.4 | 1.9e-9 V | +8e-8 mV | match |
| **1e-8** | **42.1** | **166 nV** | −1.2e-6 mV | match |
| 1e-7 | 21.6 | 1.66 µV — **over budget** | −1.2e-4 mV | match |
| 1e-6 | 7.3 | 135 µV | −7.6e-3 mV | match |

Error tracks ≈ rtol × 10–100 mV. **Production default confirmed: `interface_cg_rtol = 1e-8`**
(166 nV, 6× inside the ≤1 µV budget; 1e-7 is just over — the margin is real but not lavish, so
1e-8 stands and every later proxy measurement re-reports max|ΔV| as the standing accuracy gate).
Iteration payoff at 36 tiles: 130 → 42/step (3.1×) from rtol alone; block-Jacobi CG scales
~130 iters at 36 tiles vs ~180 at 107 tiles (§7.4), consistent with the κ ~ 1/H² growth the
Stage 3 coarse space removes. Bonus data point: CG prepare (42 s DC) vs direct (317 s DC incl.
238 s island detection + factor) — the direct factor cost CG avoids is already visible at 36 tiles.

BRCM-host GPU availability and node count: **unconfirmed — required before Stage 4/5 scoping**
(if CPU-only, fp32 tilewise study is promoted to critical path; if single-node, Stage 5 descopes
to a design note).

### Cumulative projected BRCM end-to-end (from plan arithmetic)

| Phase complete | Projected total | vs baseline 68,900 s |
|---|---|---|
| Baseline | 68,900 s | 1× |
| After Phase A (Fixes 1–7) | ~16,900 s | ~4.1× |
| + B1 balanced tiling (Fix 8) | ~7,400–9,400 s | ~7.3–9.3× |
| + threaded interface backsolve | ≤ 7,400 s | ~10× target |

BRCM re-measurement pending. Proxies: netlist_sampled perf JSON in
`scripts/benchmark/baselines/perf_netlist_sampled.json`; per-op microbenchmarks recorded in Phase V
equivalence suite (`tests/validation/test_equivalence.py`, marker `validation`).

### Original implementation sequencing (for reference)

| # | Change | Risk | Payoff | Depends on |
|---|---|---|---|---|
| 0 | Step-0 probe (period `P`, `m`, zero-delay) | none | selects tier | — |
| 1 | Fix 2 (`bincount`) + Fix 1 (hoist DC) | low | modest, immediate | — |
| 2 | **Fix 3 Tier 1** (`precompute_step_columns` + gather, behind `use_step_columns` flag) | med | **the big one** (hours → ms/step) | probe |
| 3 | Fix 4 (warm cache at setup) | low | removes first-step spike | — |
| 4 | Interface backend → supernodal (decompose path) | low | ~55 min/run | RAM check |
| 5 | Unify per-tile CHOLMOD settings | low | ~1000 s (transient) | — |
| 6 | Cache smoothed sources + island detection | low | ~50 min/run on repeats | — |
| 7 | Density-aware tile partitioning | med | factor wall + RHS straggler | — |
| 8 | Phase 3: avoid duplicate full transient | med | ~12 h/decompose run | Fix 3 |

All items above are now landed on branch `distributed-10x`.

---

## 8. Key file/line references

| Location | Role |
|---|---|
| `src/distributed/solver_td.py:556–629` | transient time loop |
| `src/distributed/solver_td.py:543` | uniform step grid `t_array` |
| `src/distributed/solver_td.py:562`, `:603`, `:573` | RHS fanout · interface solve · coordinator scatter |
| `src/distributed/tile_worker_td.py:509`, `:307` | per-step `evaluate_at_time` (transient / QS) |
| `src/distributed/tile_worker_td.py:511–513` | mask multiply · `I_p`/`I_i` slice |
| `src/analysis/vectorized_sources.py:385` | `evaluate_at_time` |
| `src/analysis/vectorized_sources.py:401`, `:405`, `:419` | zero-alloc · DC scatter · PWL scatter |
| `src/analysis/vectorized_sources.py:634`, `:838` | brute-force segment search |
| `src/analysis/vectorized_sources.py:613`, `:827`, `:704–711` | uniform-period fast path + detection |
| `src/analysis/vectorized_sources.py:684`, `:757` | lazy padded/binned cache build |
| `src/analysis/vectorized_sources.py:1068` | `bincount` scatter exemplar |
| `src/analysis/pwl_smoothing.py:2333`, `:2312`, `:347–348` | period retained · delay folded · grid snap |

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

# Conjugate Gradient in sigma-dvd — a new-hire field guide

This document explains every Conjugate Gradient (CG) implementation in the codebase: what
each one is, why it exists, what it replaced, how fast it is, and how the pieces fit
together. It is written for someone joining the project who knows basic numerical linear
algebra (what CG, preconditioning, and a Schur complement are) but has never seen this
repo.

Primary sources, if you want the raw evidence behind every number quoted here:

- `docs/brcm_distributed_runtime_optimization.md` — the full measurement campaign (§7.4–§7.13)
- `src/distributed/interface_iterative.py` — the CG solver itself (module docstring is excellent)
- `src/distributed/interface_coarse.py` — two-level coarse-space preconditioner
- `src/distributed/interface_deflated_pcg.py` — the hand-rolled deflated PCG loop
- `src/distributed/interface_deflation_notes.md` — measurement history & algorithm-selection records
- `plans/interface_solve_acceleration_plan.md` — the staged plan the work followed

---

## 1. The problem CG solves here

The distributed DDM solver partitions a PDN into tiles. Each tile eliminates its interior
nodes via a Schur complement, leaving one global system over the **interface nodes** (the
nodes shared between tiles, plus package/die/tap unknowns):

```
S · v_Γ = b        where   S = Σ_i P_iᵀ S_i P_i  +  S_extra
```

- `S_i` is tile *i*'s **dense** Schur complement over its ports (Schur complements are
  dense in the ports — this is the structural fact that drives everything below).
- `P_i` scatters tile-local port indices into the global interface ordering
  (`tile_index_maps` in the code).
- `S_extra` holds what the per-tile blocks can't see: package-edge conductances and the
  1e5 mS diagonal penalties stamped on interface-island nodes.
- `S` is **symmetric positive definite** (verified to 7e-16 symmetry in §7.8) — which is
  exactly the class of matrix CG is designed for.

Every solve phase funnels through this system: the DC solve does it once, the transient
loop does it **once per time step** (10,000 steps on the BRCM production testcase), and
the adjoint sweeps reuse the same callable. The interface solve is therefore the
scalability chokepoint of the whole distributed solver.

### Why not just factor S directly?

We do, when it fits — the direct CHOLMOD/SuperLU factorization is still the default for
small systems and is bit-exact. It stops being viable at scale for two independent
reasons, both measured on the BRCM testcase (30.6M nodes):

1. **Memory wall.** At 36 tiles, S has 70,734 unknowns and 493M nonzeros (~9.9% dense);
   the supernodal factor is ~5.4 GB — fine. Re-tile to 107 tiles (needed to balance the
   tile-side work, §7.4) and S grows to 190,867 unknowns / 1.28B nnz, with an estimated
   **47.8 GB factor**. Splitting tiles adds ports, and Schur complements are dense in the
   ports, so S's nnz grows *superlinearly* with tile count.
2. **Bandwidth-bound backsolve.** Even when the factor fits, the per-step triangular
   backsolve is single-RHS, memory-bandwidth-bound: 2.86 s/step at 36 tiles, 11.5 s/step
   at 107 tiles — 51% of the entire time step. §7.4's conclusion: *"No max-interior
   setting reaches 10×; the interface solver must change."*

That sentence is why everything in this document exists. Per-step cost is **U-shaped in
tile count**: tile-side work (factor, RHS, recovery) falls as you split; the interface
system grows faster. CG is the tool that flattens the interface side of the U.

---

## 2. Inventory — every CG in the codebase

| # | What | Where | Role |
|---|------|-------|------|
| 1 | **`InterfaceCGSolver`** — PCG on the interface Schur system | `distributed/interface_iterative.py` | **The production CG.** Everything below is a mode or component of it |
| 1a | scipy-loop path (`spla.cg`) with pluggable matvec + preconditioner | same | Used for every non-deflated configuration |
| 1b | **`_deflated_pcg`** — hand-rolled deflated ("DEF") PCG loop | `distributed/interface_deflated_pcg.py` | Production default at the split regime; can't be expressed through scipy's `M=` interface |
| 2 | Preconditioners: `none` / `jacobi` / `block_jacobi` / `amg` / `two_level` (+ coarse space) | `interface_iterative.py`, `interface_coarse.py` | The real story — CG's convergence lives or dies here |
| 3 | Direct interface factor (CHOLMOD/SuperLU) | via `pgmath.factor` | What CG replaces at scale; still default for small systems; also the bit-exact reference |
| 4 | `solve_hierarchical_coupled(solver='cg'\|'gmres'\|'bicgstab')` | `solver/unified_solver.py` | **Validation oracle only** — matrix-free iterative solve of the flat coupled system. Not a production path; don't confuse it with the interface CG |

The rest of this document is about #1 and #2. One orientation fact that saves a lot of
confusion: **`ctx.interface_lu` is always a plain callable** `solve(rhs) -> x`. Whether it
is backed by a direct LU factor, a CG solver over assembled S, or a CG solver over
per-tile blocks is decided at `prepare()` time and invisible to the time loop, the
adjoint, and everything downstream. `build_interface_solver()` (bottom of
`interface_iterative.py`) is the **single factory** that makes that decision.

---

## 3. How we got here — the chronological why

Each iteration of the CG machinery was a response to a specific measurement. Reading this
section in order is the fastest way to understand why the code looks the way it does.

### 3.1 B2 (first version): scipy CG + block-Jacobi, auto-selected

The original motivation was **coordinator memory**, not speed: at 1M+ interface nodes the
direct factor is ~200 GB. `InterfaceCGSolver` wrapped `scipy.sparse.linalg.cg` around the
assembled `S_global` with a block-Jacobi preconditioner (each interface node's owner tile
contributes its principal `S_i` submatrix; blocks are Cholesky-factored). Warm-starting
from the previous time step's solution exploits the slowly-varying transient RHS.
`auto_select_interface_solver()` picked `direct` below 200K unknowns (and within a factor
memory budget), `cg` above.

### 3.2 §7.4: the first honest measurement — both options fail at the split regime

The 107-tile BRCM run: direct = 11.5 s/step (bandwidth-bound backsolve on a 47.8 GB
factor); CG(assembled, block-Jacobi, rtol 1e-12) = **329 s/step** — ~180 warm iterations,
each one a single-threaded CSR matvec over a 14.3 GB assembled S. Both are non-starters.
This run also proved CG's *accuracy* (bit-identical peak drop vs direct at rtol 1e-12) —
so the problem was purely cost: too many iterations × too expensive an iteration.

The whole subsequent program is those two factors attacked separately:

```
cost/step = (iterations/step) × (cost/iteration)
             └── preconditioner,          └── matvec mode, threading,
                 warm start, rtol,            fp32, never-assemble
                 deflation
```

### 3.3 §7.7 (Stage 0): microbenchmarks + the rtol sweep

Two decisions that still stand came out of pure measurement:

- **`rtol = 1e-8` is the production default.** The sweep on the 36-tile proxy: 1e-12 →
  130 iters/step; 1e-8 → 42 iters/step with max|ΔV| = **166 nV** vs the direct reference
  (budget: ≤1 µV; 1e-7 was just over at 1.66 µV). Error tracks ≈ rtol × 10–100 mV. Every
  later measurement re-reports max|ΔV| as a standing accuracy gate.
- **The tilewise matvec is the right kernel.** At the mi200k split regime: assembled CSR
  SpMV = 1384 ms (single-threaded, bandwidth-bound); per-tile dense GEMVs threaded at 8
  threads = **150 ms**. Thread scaling *inverts* above 8 threads (accumulator zero-fill +
  reduction grows with thread count) — which is why `matvec_threads='auto'` caps at 8.

### 3.4 Stage 2 (§7.8): fast iterations — and the discovery that block-Jacobi is broken

Stage 2 landed the machinery for cheap iterations: threaded tilewise matvec (LPT
partition + compact per-thread buffers), fp32 storage path, D1 pad-port slicing, D2
direct-stamped `S_extra`, and never-assemble-S_global for DC (§4.7). Iteration cost hit
the target (176 ms matvec).

Then the measurement that changed the plan: **cold block-Jacobi CG stagnates outright**
at the split regime — relative residual 0.32 → 0.27 over iterations 200 → 1000.
Diagnostics proved the operator is a healthy SPD matrix (**not a bug**); the
preconditioner is the pathology: `x·M⁻¹x / x·Ax ~ 1e6` on random vectors. The Cholesky-
factored ownership blocks have genuine ~1e-10-relative near-null eigendirections
(weakly-grounded port subsets), so κ(M⁻¹S) ≳ 1e6. **Block-Jacobi intrinsically collapses
at split-regime granularity** — the block boundaries cut the physical grounding paths the
blocks would need to be well-conditioned.

*Update (§7.14, 2026-07-27): a surgical intervention experiment later pinned the collapse
on the never-assemble **block construction**, not on block-Jacobi per se. The blocks that
stagnate are built from the single owner tile's `S_i` slice and are missing the
neighbor-tile stiffness — up to 4.5× the Frobenius mass of what they keep. The same
solver with true principal-submatrix blocks (everything else byte-identical) converges
cold in 262 iterations. Still 7.7× worse than `two_level(jacobi+PoU)`'s 34, so the
production default is unaffected — but "intrinsic" was too strong; read §3.4's collapse
mechanism as a property of the path-2 blocks.*

### 3.5 Stage 3 (§7.9): the two-level coarse space

The classical DDM answer to per-subdomain preconditioner collapse: add a coarse space.
`interface_coarse.py` builds an additive correction

```
M⁻¹ = M_base⁻¹ + Z S_c⁺ Zᵀ,     S_c = Zᵀ S Z   (T'×T', tiny)
```

where Z's columns are Nicolaides **partition-of-unity** vectors (one per tile, weight
1/multiplicity on shared nodes, plus one indicator column for unowned package/die/tap
unknowns) optionally enriched with **GenEO-lite** columns (the k lowest eigenpairs of each
block-Jacobi ownership block — precisely the near-null directions measured in §7.8).

Measured results reshaped the design:

- The 65-column PoU space alone **completely repairs the cold solve**: 118 iterations at
  rtol 1e-12 where Stage 2 stagnated forever. Warm transient: 29.2 → 23.6 iters/step.
- It also makes iteration counts **tile-count-independent** (chain fixture: block-Jacobi
  grows 34→67→165 iterations as tiles go 15→60→150; two_level stays flat at ~27). This is
  the textbook κ ~ 1/H² argument showing up in real data.
- **The surprise: `two_level(block_jacobi + GenEO)` — the originally-specified design —
  fails.** With the BJ base kept alive (16 GiB budget), cold DC stagnates at rel-res
  ~1e-1 after 4000 iterations. An *additive* coarse term can only add a PSD correction;
  it deflates S's small eigenvalues but cannot remove M_BJ⁻¹'s ~1e6× amplification along
  its own near-null block directions — and those form a broad cluster, far more than a
  few GenEO vectors per block capture. A diagonal (Jacobi) base has no such amplification.
- So the production configuration became `two_level(jacobi + PoU)` — reached at the time
  *by accident*: the BJ memory-budget guard happened to downgrade the base to diagonal at
  the split regime. Remember this; it becomes a production incident in §3.8.

Stage 3 also rewrote the block-Jacobi *apply* (one global permuted gather/scatter +
per-block precomputed dense inverses applied as contiguous GEMVs, instead of per-block
fancy-indexed `cho_solve`): 701 ms → ~50–120 ms. The old design's per-block gather/scatter
held the GIL and was cache-miss-bound, serializing the threads.

### 3.6 TD never-assemble (§7.10): the transient loop joins the fast path

Stage 3's transient path still assembled S_global and ran the 1.4 s/iter assembled
matvec. Extending never-assemble + tilewise CG to the transient factor
(`_factor_transient_context_no_s_global`) was the single dominant lever: **31.1 →
6.25 s/step** at rtol 1e-8 (identical iteration count and accuracy), transient prepare
489 s / 93 GB → 125 s / 39.6 GB.

### 3.7 Deflation (§7.11): squeezing the warm-start floor

The additive form's weakness is *warm* solves: the coarse and fine spaces stay coupled,
so warm iterations only dropped modestly. `interface_coarse_apply_mode='deflated'`
removes `range(Z)` from the Krylov iteration exactly (projected matvec — see §4.6 for the
math and for why two "literal" textbook variants were implemented, measured, and
rejected). Results on mi200k_v2:

| config | cold DC 1e-12 / 1e-8 | warm iters/step | + extrapolation |
|---|---|---|---|
| additive, PoU-only | 118 / 70 | 23.6 | 20.9 |
| additive + GenEO | 118 / 70 | 23.4 | 20.9 |
| **deflated, PoU-only** | **79 / 34** | **20.0** | **17.7** |

Two defaults were flipped on this data (recorded in `interface_deflation_notes.md`):
`DEFAULT_APPLY_MODE = 'deflated'` (won every cell, better accuracy — 183 nV vs 253 nV)
and `DEFAULT_GENEO_K = 0` (GenEO contributed *zero* iteration benefit in every cell while
costing ~70 s per prepare; the machinery remains opt-in). Warm-start extrapolation
(`2·x_prev − x_prev2`) is a real 1.13× but stays opt-in.

The ≤10 warm-iteration stretch goal was **not** met: 17.7 iters/step is the floor for
this coarse space. Deflation extracted everything Z contains (its gain equals the
theoretical maximum for this Z); the residual error is locally-varying fine-space error
that no ~100-column coarse space can represent. Remaining levers are a fundamentally
stronger fine-space preconditioner (no candidate survives the §3.4 collapse analysis) or
cheaper iterations (GPU matvec: measured 27 ms SpMV on an RTX 6000 Ada — optional
backend; the BRCM host is CPU-only).

§7.12 validated the winning configuration at full length (2000 steps): 17.3 iters/step,
stationary (no drift, no stagnation, flat 40 GB RSS), 5.03 s/step of which **87.8% is the
interface solve**, peak IR-drop matching the direct-solver baseline (76.176 mV vs
76.2 mV).

### 3.8 §7.13: the BRCM hang — a memory guard was doing duty as a numerics guard

The first production BRCM run of the winning config **never completed a single time
step** — silent for hours. Root cause chain, worth internalizing:

1. Every validated measurement ran `two_level(jacobi+PoU)` because the proxy's BJ
   estimate (10.6 GB) exceeded the 8 GB auto budget → guard downgraded the base.
2. On BRCM the BJ estimate is 3.1 GiB — *below even the 4 GiB legacy floor* — so the
   guard **cannot fire**, and the run built `two_level[deflated](bj+…)`: the base §3.5
   proved collapses at split regimes.
3. The cold DC solve stagnated (A/B-confirmed on the proxy: control converges in 34
   iterations / 10 s; the BJ-base variant is flat at rel-res 1e-5 after 1500 iterations,
   having also spent +45 GB building the preconditioner that then fails).
4. It *looked* like a hang because `maxiter` defaults to `3·n` = 362,883 (≈ days at this
   regime) and the CG progress knob (`progress_every`) has no CLI flag.

Immediate unblock: `--interface-block-jacobi-max-bytes 1` (explicit values bypass the
auto floor → forces the jacobi base) plus `--interface-cg-maxiter 2000` (fail in minutes,
not days). The recorded follow-ups (base selected explicitly by regime rather than by
byte-budget accident, a CLI progress flag, a sane maxiter cap) are in §7.13 — check
whether they've landed before relying on this paragraph.

Also flagged there and still standing: **D1 pad-port coverage risk**. BRCM has 38–49% of
its interface nodes as Dirichlet pad ports; every proxy has ≈0. The kept-port slicing
machinery (`kept_position_slice`, `filter_kept_rhs`, PoU columns over the same maps) has
only toy-fixture coverage, and an indexing bug there presents the *same symptom* as the
preconditioner failure (non-convergent CG). Watch the first cold DC solve on any new
pad-heavy bundle.

---

### 3.9 The NN/BDD campaign (§7.16–§7.17): the fine-space question closed

§3.7 ended with "the remaining lever is a fundamentally stronger fine-space
preconditioner — no candidate identified." A SOTA research pass
(`docs/interface_precond_sota_research.md`) identified the classical candidate the
literature ranks first: the weighted Neumann–Neumann/BDD base, using the FULL per-tile
dense Schur blocks we already hold (categorically different from the failed BJ owner
slices — it keeps the neighbor coupling BJ discards). It was implemented
(`interface_two_level_base='neumann'`), toy-validated (6–17× iteration win over the
jacobi base on chain fixtures), and then measured dead on the proxy: cold DC 282 iters
at best regularization vs the champion's 34, stagnation as reg→0, and — the decisive
diagnostic — a spectrum probe showing every natural tile has exactly ONE near-null
mode (the textbook floating constant) while B1-split sub-tiles carry 6–460 apiece,
~2,905 total: **tearing artifacts** (modes grounded through neighbor tiles in
assembled S) that any local-solve base must amplify and no affordable coarse space can
cover. Verification cells after review: in-family ordering matches classical theory
(NN 111 < true-BJ 206 at 36 tiles), so the finding is precisely "the diagonal beats
the entire local-solve family on torn PDN operators," not "NN is broken." Full
mechanism with a hand-checkable two-port derivation:
`docs/neumann_neumann_pathology.md`; 24-node reproduction:
`scripts/benchmark/microbench/nn_pathology_demo.py`. The champion configuration is
unchanged; the fine-space chapter is closed with a proof of *why*.

## 4. Component deep-dives

### 4.1 Solver selection — `auto_select_interface_solver` / `build_interface_solver`

`interface_solver` setting: `'auto'` (default) | `'direct'` | `'cg'`.

Auto rule: `direct` iff `n_interface < 200,000` (`AUTO_CG_N_INTERFACE_THRESHOLD`) **and**
estimated factor memory (`S.nnz × 8 × fill_ratio 5`) fits the budget — resolved
host-aware as `min(32 GB, 0.4 × RAM)` (`resolve_factor_memory_budget_bytes`). Both
branches log the decision at INFO. Small systems (all unit fixtures, notebooks,
netlist_sampled) therefore never see CG; nothing in the validation suite changed when CG
landed.

`build_interface_solver()` is the one factory: it resolves `auto` for the solver, the
matvec mode (`'tilewise'` whenever per-tile blocks are available, else `'assembled'`),
and the preconditioner (`resolve_preconditioner`: `'two_level'` for CG+tilewise, legacy
`'block_jacobi'` otherwise), then returns `(solve_callable, resolved_mode, cg_solver)`.
All four factor/refactor paths in `result_factorization.py` route through it.

### 4.2 `InterfaceCGSolver` anatomy

Construction wires: the linear operator (§4.3), the preconditioner (§4.4/4.5), the coarse
space (built *after* the linear op exists, because `S_c = Zᵀ(SZ)` uses the solver's own
matmat), and the warm-start state. Key contract points in `__call__`:

- **Dispatch**: the hand-rolled `_deflated_pcg` runs only when `apply_mode='deflated'` AND
  a coarse space with retained `SZ` exists; every other configuration goes through
  `scipy.sparse.linalg.cg` unchanged. A degraded coarse build silently and safely lands
  you on the scipy path.
- **Warm start** lives *inside* the solver (`_x0`), updated after every converged solve
  (`push_solution_history`); the time loop just calls `ctx.interface_lu(rhs)`. With
  extrapolation enabled, the seed is `2·x_prev − x_prev2`. Hygiene rules that came out of
  code review and are worth knowing: failed solves clear the extrapolation history (so a
  later success never extrapolates across a failure), and zero-RHS solves don't touch it
  (a `2·0 − x_prev` seed is worse than cold). `reset_warm_start()` forces a cold start —
  the adjoint sweeps use it when switching solve families.
- **Strict mode** (default on): non-convergence raises `RuntimeError` with the *true*
  relative residual (recomputed via a fresh matvec, never the tracked recurrence value).
  `maxiter` defaults to `3·n_interface` — see §3.8 for why you should bound it on big
  runs.
- **Stats** (`cg_stats_dict` / `solver.stats`): per-solve iterations, time, `info`,
  `apply_algorithm` (`'deflated'` / `'additive'` / base name), cumulative totals, failure
  counts. The per-step iteration counts in the benchmark JSONs come from here.
- **Observability**: `solver.progress_every = 50` logs true-residual progress every N
  iterations (costs one matvec per report). Debug attribute only — no CLI flag as of
  §7.13.

### 4.3 The two matvec modes

**`assembled`** — CSR SpMV on the assembled `S_global`. Simple, single-threaded,
bandwidth-bound (1384 ms at mi200k scale). Requires assembling (and holding) S_global.
Kept as the fallback when per-tile blocks aren't available.

**`tilewise`** — `S x = S_extra·x + Σ_i P_iᵀ(S_i · x[idx_i])`, i.e. one dense GEMV per
tile plus scatter-adds. This is the production mode; the auto default whenever tile
blocks exist. Implementation details that matter:

- **Threading**: static LPT (longest-processing-time) partition of tiles by `n_ports²`
  across `matvec_threads` bins (auto: `min(8, cpus, n_tiles)` — 8 is a measured optimum,
  see §3.3). Each thread scatters into a **compact buffer** sized to the union of
  interface indices its tiles touch — not a full `(n_threads, n)` accumulator, whose
  zero-fill + reduction is what inverted thread scaling in the naive design. BLAS is
  pinned to 1 thread inside the pool region (`threadpoolctl`) to avoid nested
  parallelism. Scatter-adds use `np.bincount` (10–30× over `np.add.at`).
- **fp32 storage** (`interface_matvec_dtype='float32'`): halves memory and ~1.7–2×
  throughput, but *both* GEMV operands must be float32 — a mixed-dtype call silently
  falls off the BLAS fast path (~10× slower). fp32's ~1e-7 residual floor means it is
  guard-railed against tight tolerances: `rtol ≥ 1e-7` enforced (`≥ 1e-6` in deflated
  mode, which double-evaluates the floor through the fresh acceptance check).
- **`_tilewise_matmat`**: the multi-column generalization, used to build the coarse
  space (`S·Z`). Always accumulates in fp64 regardless of storage dtype — so `S_c` and
  `SZ` are fp64 even on the fp32 path.
- **D1 invariant**: each `S_i` must be sliced to kept (non-Dirichlet) port positions
  *before* reaching the solver; `__init__` asserts `S_i.shape[0] == len(tile_index_maps[tid])`
  so a forgotten slice fails loudly rather than corrupting indices.

### 4.4 The preconditioner ladder

`interface_preconditioner`: `'auto'` | `'none'` | `'jacobi'` | `'block_jacobi'` | `'amg'` | `'two_level'`.

- **`jacobi`** — diagonal of S. Weak but unconditionally cheap, scales to any size, and —
  the §3.5 lesson — has no near-null amplification, which makes it the *right base* under
  the two-level coarse space at split regimes.
- **`block_jacobi`** — for each interface node, the first tile whose index map contains
  it is the "owner"; each owner's principal `S_i` submatrix is factored (Cholesky;
  SPD-safe eigendecomposition fallback that clips eigenvalues — a `pinv` fallback could
  retain negative FP-noise eigenvalues and silently void CG's convergence guarantee).
  Memory scales as Σk_i² ≈ n²/T, so a byte budget (`interface_block_jacobi_max_bytes`,
  auto: `max(4 GB, min(8 GB, 0.1×RAM))`) downgrades to `jacobi` with a WARNING when
  exceeded. Apply = one global permuted gather + per-block contiguous GEMVs (§3.5).
  Fine at small/well-conditioned regimes; **collapses at split regimes** (§3.4) — and the
  budget guard is *not* a reliable proxy for that collapse (§3.8).
- **`amg`** — pyamg smoothed aggregation, lazy import, skipped gracefully if missing.
  Never became a production path.
- **`neumann`** — NN/BDD work package (§3.9): weighted Neumann–Neumann fine space
  `M⁻¹ = Σᵢ RᵢᵀDᵢS̃ᵢ⁺DᵢRᵢ` over the FULL tile Schur blocks (scatter-add tilewise
  apply; knobs `interface_neumann_{weight,reg,max_bytes}`). **Measured dead at both
  proxy regimes** (111 vs jacobi's 27 at 36 tiles; 282→stagnation at 64) — kept for
  well-grounded-block netlists, where it is a genuine 6–17× iteration win. Full
  mechanism: `docs/neumann_neumann_pathology.md`.
- **`two_level`** — base plus the coarse-space correction. The auto default for
  CG+tilewise. The base is selected by `interface_two_level_base`
  (`'auto'`→block_jacobi with its budget downgrade | `'jacobi'` | `'neumann'`);
  two apply modes — §4.5/§4.6.

### 4.5 The coarse space (`interface_coarse.py`)

- **Z (partition of unity)**: one column per tile, `1/multiplicity` on shared boundary
  nodes; one indicator column for unowned unknowns (package/die/tap — block-Jacobi treats
  those as identity, so the coarse space must "see" them); island-penalized rows zeroed
  in *every* column (PoU and GenEO alike — otherwise the 1e5 penalty diagonal leaks into
  and corrupts `S_c`); all-zero columns dropped. T' ≈ n_tiles + 1 — 65 columns at the
  mi200k regime.
- **GenEO-lite enrichment** (opt-in since the §3.7 flip, `interface_coarse_geneo_k`): k
  lowest eigenpairs per ownership block via shift-invert `eigsh` reusing the existing
  Cholesky factor (dense `eigh` below 500 rows); a decoupled one-block-at-a-time path
  runs it even when the BJ base was budget-downgraded. Measured contribution at both
  regimes tested: zero — hence `DEFAULT_GENEO_K = 0`.
- **S_c** = `Zᵀ(SZ)` via the solver's own fp64-accumulating matmat (so `S_extra` — package
  edges and island penalties — is automatically included), pseudo-inverted via `eigh`
  with rank truncation. Structural rank deficiency (the alternating-sign checkerboard
  combination of PoU columns) is expected, not an error; the rank is logged.
- **Degradation ladder** — build failures never kill `prepare()`: PoU+GenEO → PoU-only
  (column-count cap `interface_coarse_max_cols`=4096 or byte cap
  `interface_coarse_max_bytes`, auto `min(8 GB, 0.1×RAM)` — the byte cap is the one that
  scales with n and actually protects the never-assemble regime; true peak is ~3·n·T'·8
  bytes) → base preconditioner only, each rung with a WARNING.
- Never persisted: rebuilt by every `factor()`/`refactor()`, like the tile Schur blocks
  it derives from.

### 4.6 The deflated apply mode — the hand-rolled PCG

**Why hand-rolled:** the deflated operator combination (projected matvec + plain base
preconditioner) is not expressible as a single symmetric `M⁻¹`, so it cannot be passed as
scipy's `M=`. `_deflated_pcg` mirrors scipy's contract exactly (stopping criterion,
`bnrm2==0` short-circuit, `(x, info)` return) so both paths share the caller.

**The math** (notation: `Q = Z S_c⁺ Zᵀ`, never materialized; `P = I − SQ`, idempotent):

- The **matvec itself is projected**: `w = P(S p) = S p − SQ(S p)`. Because
  `Zᵀ(P v) = 0` *identically*, the deflation invariant `Zᵀ r_k = 0` holds by
  construction at every iterate — not "usually".
- The preconditioner apply is the **plain base** (`z = M_base⁻¹ r`, no Q term).
- The solution is **recovered**, not accumulated: the CG iterate `y` solves the projected
  system `(PS) y = Pb`; the answer is `x = y + Q(b − S y)`. Accumulating `x += α·p`
  against a projected matvec is precisely the "r no longer equals b − Sx" bug the
  rejected variants hit. `S y` is tracked incrementally from the already-computed `S p`,
  so recovery costs one coarse apply, not a matvec. Cold start with zero iterations
  degenerates to `x = Qb` — the intuitive coarse-only solve.
- `SZ` (dense `n × T'` fp64) is retained on the `CoarseSpace` so every `SQ` application
  is a GEMV pair, never a full matvec.

**DEF's win is base-dependent** (§7.15): on the production *jacobi* base, deflated beats
additive in every measured cell (§7.11); on a *block-Jacobi* base the ranking flips —
additive helps (225 vs plain BJ's 262 cold iters at the split regime with true blocks)
while deflated hurts (283 even with reprojection off). Plausible mechanism: a BJ base
re-amplifies exactly the span(Z)-overlapping directions each iteration, so the projected
matvec keeps cancelling large components (fp noise floor in the tail); a diagonal base
doesn't. If a bj-base two_level is ever shipped, default it to additive.

**Why DEF and not the textbook A-DEF2** (full record:
`interface_deflation_notes.md`): three candidate formulas from the
Tang/Nabben/Vuik/Erlangga taxonomy were implemented and measured. The spec's literal
formula (actually A-DEF1) **stalls** on the real `netlist_multi_tile` fixture; the
P-after-base variant **silently stagnates** (`info=0` with the tracked residual satisfied
while the true residual never converges — the most dangerous failure mode in this whole
file); true A-DEF2 ties DEF when the base is diagonal (they're mathematically equivalent
there) but regresses 18% on BJ bases and hits maxiter on the ill-conditioned fixture.
Root cause for the rejects: with an un-projected matvec, `Zᵀ r_k = 0` is not preserved
past the first iterate and the effective operator isn't symmetric. DEF won on data; the
setting is honestly named `'deflated'`. The rejected implementations live on in
`tests/distributed/test_interface_coarse.py` as regression coverage.

**Safety machinery** (each guard earned by a review finding or a measured failure):

- *Re-projection* every `interface_deflated_reproject_every` (50) iterations recomputes
  `r = P(b − Sy)` from scratch — FP hygiene for `Zᵀr` drift, verified not to affect
  convergence on any fixture. (§7.15 later measured it *dose-dependently harmful* on
  long ill-conditioned solves — each residual replacement perturbs CG conjugacy, +28
  iters at the default interval on a 300-iter BJ-base solve. Production jacobi-base
  solves finish in ≤34 iters and never reach it.)
- *True-residual acceptance*: every tentative convergence (and the strict-mode gate)
  recomputes `b − S·x_recovered` via a fresh matvec — the tracked `r` belongs to the
  *projected* system and is a genuinely different quantity. Disagreement isn't fatal; the
  loop just keeps iterating (debounced to at most one attempt per re-arm period, fallback
  10 iterations).
- *Breakdown guards*: `ρ` and `pᵀw` are tested against a scaled FP noise floor
  (`eps × ‖a‖‖b‖`), not exact zero — near-PSD penalty-heavy systems land *near* zero,
  and dividing by a tiny value flips signs instead of triggering the restart an exact
  zero would.

### 4.7 Never-assemble S_global (`interface_drop_s_global=True`)

The tilewise matvec only needs the per-tile dense blocks — so at scale we never form
S_global at all. Interface ordering and Dirichlet RHS vectors are derived from tile port
name lists; blocks are gathered via a streaming one-tile-in-flight protocol with D1
slicing applied per tile (peak coordinator memory never holds an unsliced sum or an
assembled CSR); `S_extra` is direct-stamped (D2) instead of computed as a giant
subtraction. Requires the union-find island detector (`island_detection_mode='summaries'`
— the legacy Schur-BFS needs S_global's structure and triggers a WARNING fallback).
Applies to both the DC and transient factor paths, each with its own block set (DC's are
G-based, transient's are `A = G + C_coeff·C`-based), so both contexts can coexist.
Measured: the >190 GB watchdog-killed coordinator became 18–40 GB (§7.8, §7.10);
`save()` raises with guidance since there's nothing assembled to save.

### 4.8 Accuracy discipline

CG at rtol 1e-8 is an *approximation* where direct was exact, so the project carries a
standing accuracy budget: **≤1 µV max|ΔV|** on tracked nodes vs a reference. Every
measurement section re-reports it (166 nV at the sweep; ≤300 nV through every Stage 3–
deflation cell; peak-drop node match every time). rtol 1e-12 CG was proven *bit-identical*
to direct at BRCM scale (§7.4) — so there's a validation-grade setting when you need it.
When touching numerics, keep this pattern: measure iterations AND max|ΔV|, never
iterations alone.

---

## 5. Performance summary (mi200k_v2 proxy, 64 tiles / 168,586 unknowns, warm transient @ rtol 1e-8)

| Configuration | Cold DC iters | Warm iters/step | s/step | Note |
|---|---|---|---|---|
| CG assembled + BJ @1e-12 (§7.4, 107t BRCM) | — | ~180 | 329 | the starting point |
| Stage 2: tilewise + BJ | stagnates | — | — | BJ collapse discovered |
| Stage 3: two_level additive (jacobi+PoU), assembled TD | 70 | 23.6 | 31.1 | cold solve fixed |
| + TD never-assemble | 70 | 23.6 | 6.25 | 5× from matvec cost |
| + deflated apply | 34 | 20.0 | ~5.6 | defaults flipped here |
| + warm-start extrapolation (opt-in) | 34 | **17.7** | **~5.0** | current best |
| Full-length 2000-step check (§7.12) | 34 | 17.3 (stationary) | 5.03 | 88% of step = interface solve |

BRCM-host projection at the equivalent split regime: ~8–15 s/step (≈3× kernel-speed
ratio), vs the 329 s/step CG baseline and the direct path that no longer fits.

---

## 6. Operational lessons (the ones that cost real time)

1. **A preconditioner can fail as numerics while succeeding as code.** Block-Jacobi at
   split regimes converges nowhere, with a perfectly healthy SPD operator (§3.4). Check
   `x·M⁻¹x / x·Ax` before blaming the matrix. (§7.14 later proved the failure was the
   single-owner-tile block *construction*, isolated by direct intervention — see §3.4.)
2. **Never let a memory guard double as a numerics guard** (§3.8). The base
   preconditioner choice must be explicit policy, not a budget side-effect.
3. **Distrust tracked residuals.** Two separate incidents (the P-after-base variant's
   `info=0` stagnation; the fresh-acceptance/fp32 interaction) — always gate acceptance
   on a freshly recomputed `b − Sx`.
4. **Bound `maxiter` and turn on progress logging for anything big.** The default `3·n`
   turns "did not converge" into "silent for days, then discards the 21-minute prepare".
5. **Warm-start numbers don't transfer to cold solves** (130 warm vs stagnation cold on
   the same config), and **proxy structure must match the failure you're studying** —
   §7.5's fragmented sampler inverted the bottleneck profile entirely; the D1 pad-port
   population still isn't represented by any proxy.
6. **Measure before shipping a default.** Every default in this stack (rtol 1e-8, 8
   threads, deflated apply, geneo_k=0, extrapolation off) is the direct output of a
   recorded measurement, and two of them *reversed* the design intuition (GenEO, A-DEF2).

---

## 7. Knobs quick reference (model.settings / YAML / CLI `--interface-*`)

| Setting | Default | Meaning |
|---|---|---|
| `interface_solver` | `auto` | `direct` < 200K unknowns & factor fits budget; else `cg` |
| `interface_matvec_mode` | `auto` | `tilewise` when tile blocks exist, else `assembled` |
| `interface_preconditioner` | `auto` | `two_level` for CG+tilewise, else `block_jacobi` |
| `interface_cg_rtol` / `atol` | `1e-8` / `1e-14` | §3.3 sweep; 1e-12 = validation grade |
| `interface_cg_maxiter` | `None` → 3·n | **bound this on production runs** (§3.8) |
| `interface_cg_strict` | `True` | raise vs warn on non-convergence |
| `interface_factor_memory_budget` | `auto` | min(32 GB, 0.4·RAM) — direct-path gate |
| `interface_block_jacobi_max_bytes` | `auto` | max(4 GB, min(8 GB, 0.1·RAM)); explicit value bypasses the floor |
| `matvec_threads` | `auto` | min(8, cpus, n_tiles) |
| `interface_matvec_dtype` | `float64` | `float32` = ~2× matvec, needs rtol ≥ 1e-7 (1e-6 deflated) |
| `interface_drop_s_global` | `False` | never-assemble path; needs `island_detection_mode='summaries'` |
| `interface_coarse_geneo_k` / `_tol` | 0 / 1e-6 | GenEO enrichment (opt-in) |
| `interface_coarse_max_cols` / `_max_bytes` | 4096 / auto | coarse-space caps (degradation ladder) |
| `interface_coarse_apply_mode` | `deflated` | or `additive` |
| `interface_deflated_reproject_every` | 50 | ≤0 disables re-projection |
| `interface_warm_start_extrapolation` | `False` | opt-in `2x_prev − x_prev2` seeding |
| `solver.progress_every` (attribute) | 0 | per-N-iteration true-residual logging (no CLI flag yet) |

## 8. File & test map

| File | Contents |
|---|---|
| `src/distributed/interface_iterative.py` | `InterfaceCGSolver`, matvec modes, preconditioner builds, auto-selection, `build_interface_solver` factory |
| `src/distributed/interface_coarse.py` | PoU basis, GenEO eigenpairs, `CoarseSpace`, `build_coarse_space` |
| `src/distributed/interface_deflated_pcg.py` | `_deflated_pcg` loop, breakdown guards |
| `src/distributed/interface_deflation_notes.md` | measurement history, algorithm ratification records |
| `src/distributed/result_factorization.py` | factor/refactor paths that call the factory; never-assemble gathers; settings reader |
| `tests/distributed/test_interface_iterative*.py`, `test_interface_coarse.py` | unit + regression coverage, incl. the rejected A-DEF1/A-DEF2 implementations |
| `scripts/benchmark/microbench/` | Stage 0–3 measurement scripts + raw JSONs behind every number in §5 |

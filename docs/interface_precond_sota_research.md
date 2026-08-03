# Two-Level Interface Preconditioners for SPD Schur Systems: Landscape, Candidates, and a Benchmark Plan for `InterfaceCGSolver`

**Scope.** The interface solve `S u_Γ = g` (S = Σᵢ RᵢᵀSᵢRᵢ, per-tile dense Schur blocks Sᵢ, SPD after tap/Dirichlet handling) is now 88% of a transient step at the split regime, at ~20.0 warm iters/step × ~0.19 s/iter under the shipped champion `two_level[deflated](jacobi+PoU)` (cold DC 34 iters @1e-8, 79 @1e-12; 17.7 iters/step with opt-in extrapolation). The measured verdict of the deflation campaign was: *deflation has extracted everything the current coarse space Z contains; the residual warm floor is fine-space error that a diagonal (Jacobi) base cannot damp, and enlarging Z hits the T′² coarse-solve and SZ-memory walls first.* This report maps what the domain-decomposition literature says the next fine-space rung should be, ranks implementable candidates against our per-tile dense Sᵢ data layout, and specifies the benchmark that would ratify or kill each one.

A note on evidence discipline up front: two of the three foundational papers were verified against their full text; the claims below distinguish **proved bounds** (theorems, cited precisely), **library-reported engineering results** ([6][8], taken from paper summaries, not re-verified line-by-line), and **local measurements** (our §7.x campaign logs). One frequently miscited point is corrected explicitly in §1.2.

---

## 1. Landscape: the two-level substructuring family tree

All serious interface preconditioners for SPD Schur systems have the same skeleton — a **fine-space component** built from local (per-subdomain/per-tile) Schur solves, plus a **coarse component** that removes the O(1/H²) low-frequency modes that make one-level methods degrade with subdomain count. The families differ in (a) what the local solve is, (b) how local corrections are *weighted* back onto shared interface nodes, and (c) how the coarse space is built and applied.

### 1.1 One-level baselines (where we started)

Diagonal (Jacobi) or block-Jacobi preconditioning of S has no coarse component and no mechanism to propagate information across subdomains faster than one layer per iteration; condition number grows like 1/H² with subdomain count at fixed problem size. This is not just theory for us: assembled block-Jacobi CG measured ~130 iters at 36 tiles → ~180 at 107 tiles on BRCM, consistent with the κ ~ 1/H² growth the coarse space removes (local log, §7.7), and *never-assemble* BJ — whose blocks are single-owner-tile Sᵢ slices, not true diagonal blocks of S — **stagnates outright cold** at the split regime (§7.13–7.15). One-level methods are the floor of the ladder, kept only as the base smoother under a coarse space.

### 1.2 Neumann–Neumann / Balancing Domain Decomposition (BDD)

The classical two-level primal family. Fine space: a **weighted sum of local Schur (pseudo-)inverses**, M⁻¹ ≈ Σᵢ RᵢᵀDᵢSᵢ⁺DᵢRᵢ, where Dᵢ is a diagonal weighting forming a partition of unity across tiles sharing an interface node. Coarse space: the "balancing" step projects out the near-null spaces of the singular Sᵢ (floating subdomains) — structurally the same job our deflation Q does. Mandel & Brezina's key result is that with **coefficient-weighted** (not simple counting/multiplicity) Dᵢ, BDD's convergence bound is independent of the number of subdomains and robust to large inter-subdomain coefficient jumps, degrading only polylogarithmically in H/h [4]. This is the direct theoretical ancestor of "weighted DᵢSᵢ⁺Dᵢ over tiles + PoU-flavored coarse projection" — i.e., of exactly the upgrade path open to us, since we already hold every Sᵢ dense.

### 1.3 FETI-DP (dual) and BDDC (primal)

The modern pair. FETI-DP iterates on Lagrange multipliers enforcing interface continuity, with selected "primal" constraints (corner values, edge/face averages) enforced exactly at every iteration; BDDC is its primal mirror, built by Dohrmann from **constrained energy minimization**: coarse basis functions are the energy-minimal extensions of coarse degrees of freedom (corner values + weighted interface averages), and the preconditioner combines weighted local corrections with that coarse solve [2]. The proved headline for the family, from Klawonn–Widlund–Dryja's analysis of dual–primal FETI in 3D:

- With edge/face-average constraints and coefficient-aware scaling, **κ(M⁻¹F) ≤ C(1+log(H/h))²**, with C independent of h, H, the number of subdomains N, and the coefficient values ρᵢ (Theorems 1 and 2 of [3]; the abstract states the bounds are "otherwise independent of the number of subdomains, the mesh size, and jumps in the coefficients") [3].
- With **vertex-only constraints** (their Algorithm A), the bound degrades to **κ ≤ C(H/h)(1+log(H/h))²** — linear, not polylogarithmic, in H/h — and Algorithm A is explicitly noted as non-competitive in 3D practice (citing prior numerical work; [3] itself is purely theoretical, containing no experiments) [3].

That vertex-only-vs-averages gap is the single most actionable theoretical fact for us: it says *what you put in the coarse/constraint space determines whether you get (H/h)·polylog or pure polylog*, and our geometric PoU columns are closer in spirit to the weak end of that spectrum than to energy-minimized averages.

**Correction of a common miscitation.** The Mandel–Dohrmann–Tezaur "algebraic theory" paper [1] is often cited as the source of a κ ≤ C(1+log(H/h))² bound for BDDC/FETI-DP. It contains **no such bound and no numerical experiments whatsoever** — it is purely algebraic, ending at a model-problem setup section [1]. What it actually proves is sharper and more abstract:

- **P-FETI-DP and BDDC are the same operator**: M_BDDC = E Š⁻¹ Eᵀ (their Eq. 15) equals the P-FETI-DP preconditioner (Theorem 3.1) [1].
- Both preconditioned operators reduce to the abstract form (L A⁻¹ Lᵀ)(Tᵀ A T) with LT = I, and **all eigenvalues satisfy 1 ≤ λ ≤ ‖TL‖²_A** (Lemma 4.1); concretely 1 ≤ λ ≤ ω_BDDC = ‖E‖²_S and 1 ≤ λ ≤ ω_FETI-DP = ‖B_Dᵀ B‖²_S, and under the natural compatibility assumption B_DᵀB + E = I the two ω's coincide (Theorem 4.1) [1].
- **The spectra of preconditioned BDDC and FETI-DP are identical except possibly at eigenvalue 1** (Theorem 4.2), so the dual and primal methods are interchangeable in convergence terms — pick whichever fits the data layout [1].
- The entire theory needs only algebraic assumptions (intermediate space W̃, projection E with E² = E, jump operator B with null B = Ŵ, generalized inverse B_Dᵀ with BB_Dᵀ = I) — **no mesh, no H, no h, no N** [1]. This matters for us because our "tiles" are electrical partitions of an extracted PDN, not FEM substructures: the algebraic framing guarantees the machinery applies as-is to our S = Σ RᵢᵀSᵢRᵢ with PSD per-tile Sᵢ, which is exactly the setting their model section assumes (Sᵢ symmetric positive *semi*-definite, S block-diagonal over substructures) [1].

The concrete condition number one buys is then governed by ‖E‖²_S — i.e., **by the quality of the weighted averaging operator E**, which is where scaling choices enter. (The O(log²(1+H/h)) proof for BDDC itself is attributed, via secondary sources, to the companion Mandel–Dohrmann convergence paper rather than to [1] or verifiably to [2] — see reliability notes, §6.) The standard PCG bound ‖e⁽ᵏ⁾‖_K ≤ 2((√κ−1)/(√κ+1))ᵏ‖e⁽⁰⁾‖_K [1] converts these κ's into iteration predictions: κ ≈ 10 → ~14 iters for 1e-8, κ ≈ 100 → ~45.

### 1.4 Scaling/weighting refinements: ρ-scaling and deluxe scaling

Within BDD/BDDC/FETI-DP, the diagonal weights Dᵢ can be: multiplicity (counting) weights, coefficient (ρ-/stiffness-diagonal) weights, or **deluxe scaling** — replacing the diagonal weight on each shared interface object by a small dense solve combining the two (or more) adjacent tiles' local Schur blocks on that object [7]. Deluxe scaling was introduced/analyzed for badly-behaved discretizations (isogeometric) precisely because diagonal scalings degrade under strong local stiffness contrast; it makes the condition number far less sensitive to coefficient jumps and geometric distortion, at the cost of extra small dense local solves at setup [7]. For a PDN, inter-layer conductance contrast (M1 vs. thick top metals, via stacks) is the analog of coefficient jumps — this is the knob to reach for if a plain NN/BDDC fine space works but shows tile-to-tile iteration sensitivity.

### 1.5 Adaptive/spectral coarse spaces and multilevel extensions

GenEO-style enrichment solves local generalized eigenproblems to *discover* the coarse vectors a given problem needs, rather than prescribing corners/averages; HPDDM is the production embodiment (GenEO two-level Schwarz + FETI/BDD, with deflated/recycled Krylov and multi-RHS support, exposed in PETSc as KSPHPDDM/PCHPDDM) [8]. PCBDDC is the production BDDC in PETSc, with deluxe scaling and adaptive coarse selection, reporting robust weak scaling to 8232+ MPI ranks on 3D multi-material problems with 10⁸ coefficient contrast and >0.5B DOFs, using an overlapped **multilevel** coarse solve [6]. Multilevel matters at ~1000 blocks: the coarse problem itself grows with block count and eventually needs its own DD treatment [6].

Our local data point cuts against naive spectral enrichment, though: GenEO-lite (eigs of the *owned* cho-factored blocks) contributed **zero** iteration benefit in every measured cell at +70 s prepare, and now ships disabled (`geneo_k=0`). The honest reading is that our GenEO-lite eigenproblem (owned-block near-null modes) is not the literature's GenEO eigenproblem (local Neumann vs. weighted-assembled pencils), and that our PoU deflation already captures the same near-null content — not that adaptive coarse spaces are refuted. See §5.

### 1.6 Expected block-count scaling per family (64 → 107 → ~1000 blocks)

| Family | κ behavior in N (fixed total n) | Expected iters, 64 → 107 → ~1000 blocks | Coarse-problem growth |
|---|---|---|---|
| Jacobi / BJ (one-level) | grows ~1/H² | measured 130 → 180 (BJ, 36→107t); stagnates cold at split regime | none |
| jacobi + PoU deflation (champion) | N-independent-ish, but fine-space quality weak (vertex-like end of [3]'s spectrum) | 34 cold @1e-8 today; expect mild growth with N since local H/h *shrinks* but Jacobi base doesn't exploit it | T′ ≈ O(N); dense T′² solve, capped at 4096 cols |
| BDD / NN + coarse [4] | independent of N; polylog in local H/h | should *improve* per-block conditioning as blocks shrink (smaller H/h); flat-to-falling iters across 64→1000 | same T′ growth as above |
| BDDC / FETI-DP (energy-minimized constraints) [2][3][1] | κ ≤ C(1+log(H/h))², independent of N and jumps [3] | flat across block counts; interchangeable dual/primal spectra [1] | coarse dof count O(N·objects); multilevel needed at very large N [6] |
| + deluxe scaling [7] | as above, contrast-robust | flat, and robust to layer-contrast outliers | + small local dense solves at setup |
| Adaptive (GenEO) [8] | provably targetable κ | flat, at eigsolve setup cost | data-driven size; our lite variant measured useless locally |

The strategic conclusion from the table: **at ~1000 blocks the champion's weakness is not the coarse space (PoU T′ ~ 1000–4000 still fits the dense-coarse-solve cap) but the Jacobi fine space**, which leaves per-block interior conditioning on the table that BDD/BDDC-class local solves would harvest — precisely matching the measured deflation-campaign verdict that the warm floor is fine-space error.

---

## 2. Ranked shortlist

Notation: tiles i = 1..N with dense Schur blocks Sᵢ (nₚᵢ ports; Σnₚᵢ² = 2.34B → 18.7 GB fp64 at mi200k_v2), global interface size n = 168,586, PoU diagonal weights Dᵢ (Σᵢ RᵢᵀDᵢRᵢ = I, already implicit in our PoU columns), coarse matrix Z (T′ = 65 PoU cols today), deflation projector P = I − SQ, Q = Z(ZᵀSZ)⁻¹Zᵀ, base apply M⁻¹_base. Champion economics: ~0.19 s/iter (tilewise matvec + jacobi base + coarse), warm 20.0 iters/step, cold DC 34 @1e-8, step ≈ 20×0.19 + 1.5 RHS + 0.35 recovery ≈ 5.6 s. All "predicted" numbers below are estimates to be ratified by §3's protocol, not measurements.

### Candidate 1 — Neumann–Neumann fine space: M⁻¹_NN = Σᵢ RᵢᵀDᵢS̃ᵢ⁺DᵢRᵢ, under the existing DEF loop  ★ top pick

**What it is.** Replace the Jacobi base apply with a weighted sum of per-tile dense Schur solves — the BDD fine space [4], balanced by our existing PoU deflation (which plays exactly the role of BDD's coarse "balancing" of floating-tile null spaces [4], and the algebraic framework of [1] guarantees the combination is analyzable without any FEM assumptions). **This is categorically different from the failed BJ base**: BJ partitioned interface nodes into disjoint owner slices, discarding up to 4.5× the Frobenius mass of neighbor-tile stiffness (§7.14); NN applies every tile's *full* Sᵢ⁺ and reconciles shared nodes through Dᵢ, so no coupling is discarded — the information BJ threw away is exactly what NN keeps.

**Algorithm sketch.**
- Setup, per tile (worker-side, embarrassingly parallel): symmetric-eigendecompose or Cholesky-with-clipping S̃ᵢ = Sᵢ (+ tap/regularization handling); materialize the dense (pseudo-)inverse *replacing* the factor payload, per the §7.15 permuted-GEMV lesson — apply must be one contiguous GEMV, not triangular solves. The SPD-safe eigenclip fallback already shipped for BJ blocks is reused verbatim for singular Sᵢ (floating tiles), whose true kernel is handled by deflation.
- Apply, per iteration: gather x slice per tile (same gathers as the matvec), scale by Dᵢ, GEMV with Sᵢ⁺, scale by Dᵢ, scatter-add — structurally a second tilewise-matvec pass over the same LPT thread partition and compact scatter buffers.
- Everything else (DEF projection, coarse solve, fresh-true-residual acceptance, maxiter bound) unchanged. **Caution:** DEF-vs-additive preference is base-dependent (measured: deflated wins on jacobi, additive wins on bj bases — §7.15), so the DEF/additive A/B must be re-run on the NN base rather than assumed.

**Setup cost.** Dense factor+invert Σᵢ O(nₚᵢ³): dominated by the largest block (13,834 rows → ~10¹²·2.5 flops); the §7.9-measured 97 s factor+eigsh sweep over the same 64 blocks (including that block) is the right order — budget **~2–4 min added to `prepare()`**, embarrassingly parallel across Ray workers. Memory: +18.7 GB fp64 for the inverses (or +9.4 GB fp32 apply, legal at rtol ≥ 1e-7 per the enforced dtype/rtol pairing) on top of the never-assemble path's 19–40 GB coordinator peak — worker-resident, so it lands on the same hosts already holding Sᵢ.

**Per-iteration delta.** +1 tilewise dense pass ≈ +0.15–0.19 s → **~0.35–0.40 s/iter** (matches the §7.8 Stage-2 projection of ~0.27 s/iter for matvec + threaded dense apply). Break-even therefore requires ≥ ~2× iteration cut.

**Predicted iterations (mi200k_v2).** The family's bound is κ ≤ C(1+log(H/h))²-class, N-independent [3][4]; with tile port counts ~2.7k, (1+log H/h)² is a small constant multiple. Prediction: **cold DC 10–18 @1e-8** (vs 34), **warm 5–10 iters/step** (vs 20.0) — warm gains should exceed cold ratio because the warm floor was specifically fine-space error. Predicted step: ~8 × 0.38 + 1.5 + 0.35 ≈ **4.0 s/step**, with the real prize being the ≤10-warm-iters target (§7.11's declared make-or-break for the CPU path) coming into range, and flat scaling to 107/1000 blocks where the champion is unproven. Kill criterion: if warm iters land ≥ 12, the 2× per-iter cost makes it a wash — stop.

**Effort.** Moderate (~1–2 weeks): new base mode `'neumann'` in the preconditioner ladder, reusing `_bj_perm`-style single-gather/single-scatter machinery, the eigenclip fallback, and the LPT thread partition; plus the DEF/additive re-A/B. No new distributed plumbing.

### Candidate 2 — BDDC-style energy-minimized coarse columns (constrained energy minimization) replacing geometric PoU

**What it is.** Keep T′ and the deflation machinery; replace each geometric PoU column (flat indicator over a tile's interface footprint) with its **energy-minimal representative**: per tile, solve a small saddle system with Sᵢ so the column is the minimum-Sᵢ-energy interface function with unit weighted average on the tile's objects — Dohrmann's constrained-energy-minimization construction [2], which per [1] makes the resulting E the operator whose ‖E‖²_S *is* the condition-number bound. This directly attacks the vertex-like-vs-averages gap of [3]: same coarse dimension, strictly better coarse quality.

**Setup cost.** Per tile: one dense factor of Sᵢ (shared with Candidate 1 if both land) + a few RHS solves per coarse column — minutes, parallel. Coarse dimensions unchanged → no new T′² pressure. **Per-iteration delta: ~0** (Z changes content, not shape; SZ storage identical).

**Predicted iterations.** On the current jacobi base alone: cold DC 34 → **~20–28**, warm 20.0 → **~16–19** — real but modest, because the measured warm floor is fine-space, which this doesn't touch. Stacked on Candidate 1 it is the difference between "BDD-flavored" and "BDDC-proper," plausibly the last 1.3–1.5× (warm **4–8**). **Effort:** low-moderate (ic~1 week); ship as `interface_coarse_mode='energy'` A/B-able against `'pou'`.

### Candidate 3 — Deluxe scaling for the Dᵢ weights (contrast robustness)  [7]

**What it is.** On each shared interface object between tiles i, j, replace diagonal PoU weights by the deluxe average: w = (S⁽ⁱ⁾_ΓΓ + S⁽ʲ⁾_ΓΓ)⁻¹S⁽ⁱ⁾_ΓΓ applied to tile i's contribution (small dense solves on the object's dofs, extracted from the already-dense Sᵢ). Folds into the setup of Candidates 1–2; the per-iteration apply keeps the same shape (weights become small dense blocks applied during gather/scatter, or are pre-multiplied into the stored Sᵢ⁺).

**When it pays.** Only if Candidate 1 shows tile-contrast pathology — e.g., iteration counts dominated by a few tiles spanning high-contrast layer stacks, the PDN analog of the coefficient jumps that motivate deluxe scaling [7] and the coefficient-weighted BDD scaling before it [4]. **Predicted:** no change on benign bundles; recovers Candidate-1-level iterations on contrast-pathological ones. **Effort:** moderate; **rank it as a contingency, gated on Candidate 1's per-tile iteration diagnostics** — do not build speculatively.

### Candidate 4 — Iteration-cost reduction as the orthogonal axis: fp32 NN apply + fused worker-side iteration batching (with GPU as opt-in backend)

**What it is.** Not a preconditioner: accept ~8–20 iters and shrink the 0.19–0.38 s/iter. (a) fp32 storage for both Sᵢ (already supported) *and* Sᵢ⁺ apply — both GEMV operands must share dtype to stay on the BLAS fast path (measured: mixed dtype is ~10× slower than fp64); halves memory and roughly doubles throughput, legal at production rtol 1e-8 under the enforced rtol ≥ 1e-7 guard. (b) Batch k CG iterations worker-side to amortize Ray RTT (~16 ms/round-trip measured) — relevant mainly at 107+ tiles multi-host. (c) The measured GPU numbers (~0.03 s/iter projection) stay an optional backend since the BRCM host is CPU-only. **Predicted:** 1.5–2× on s/iter, multiplicative with Candidates 1–2. **Effort:** (a) days; (b) 1–2 weeks of protocol work.

### Candidate 5 — Multilevel coarse solve (three-level BDDC) — *deferred until ~1000 blocks is real*

At N ≈ 1000, PoU T′ ≈ 1000–4000 approaches the 4096-column cap and the dense T′² coarse solve plus SZ storage (n × T′) start to bite; the production answer at extreme N is a multilevel coarse problem, as in PCBDDC's overlapped multilevel implementation scaled to 8232 ranks [6], with HPDDM's recycled/multi-RHS Krylov machinery the reference for warm-start-heavy transient workloads like ours [8]. **Effort:** high; **trigger:** coarse-solve + SZ-apply time exceeding ~20% of an iteration in the 1000-block scaling run, not before. In the nearer term, PETSc PCBDDC/HPDDM are best used as *oracles* — cross-checking our achieved iteration counts against a reference BDDC on an exported S — rather than as production dependencies inside the Ray/numpy stack.

---

## 3. Benchmark protocol vs. the champion

Every cell runs under the standing discipline: used-memory watchdog (`run_bj_true_block_watchdog.sh` pattern), bounded `--interface-cg-maxiter` (never the 3·n default — the measured multi-day-silent-hang failure mode), `progress_every` true-residual logging, acceptance gated on freshly recomputed ‖b − Sx‖ (two measured incidents of tracked-residual lies), and results recorded as a new §7.x in `docs/brcm_distributed_runtime_optimization.md` with scripts + raw JSONs in `scripts/benchmark/microbench/`.

**P0 — Correctness gate.** Flat-vs-distributed equivalence suite at rtol 1e-12 (bit-identical validation grade); `netlist_multi_tile` smoke; the ill-conditioned realistic-T′/n-ratio fixture that killed true-A-DEF2 must converge.

**P1 — Champion head-to-head (mi200k_v2, 64 tiles / 168,586 unknowns, Ray, tiles_per_worker=4).**
- Cold DC at rtol 1e-8 and 1e-12: iters, s/iter, prepare-time delta, peak RSS (coordinator + worker).
- 100-step BE transient, dt = 5 ps, IC = DC solve, rtol 1e-8, extrapolation both off and on: warm iters/step trajectory (report mean of steps 20–100; §7.12 showed stationarity matters), s/step decomposition (iters × s/iter vs RHS ~1.5 s vs recovery ~0.35 s).
- Accuracy: max|ΔV| vs the rtol 1e-12 tracked reference, budget ≤ 1 µV (champion: 183 nV) — re-reported for every cell, per standing practice.
- DEF vs additive A/B **re-run on the new base** (base-dependence is measured fact, §7.15).
- Champion reference cells to beat: cold 34 @1e-8, warm 20.0 (17.7 extrap), ~5.6 (~5.0) s/step.

**P2 — Block-count scaling (36 / 64 / 116-tile bundles).** Same physical netlist re-parsed at three `--max-interior` settings. Success criterion for any two-level candidate: cold and warm iteration counts **flat (±20%) across block counts** — the N-independence the theory promises [3][4] and the champion has only partially demonstrated; the one-level control row (BJ: 130 → 180 → stagnation) anchors the plot. Also track T′, coarse-solve ms/iter, and prepare time vs N (setup is Σnₚᵢ³-dominated and should *fall* as blocks shrink).

**P3 — Process/thread scaling.** matvec+apply s/iter at matvec_threads ∈ {1, 4, 8, 16} × tiles_per_worker ∈ {1, 4, 8}; verify the NN apply inherits the compact-scatter scaling (the naive full-row accumulator design measured *inverting* above 8 threads) and that BLAS is pinned (threadpool_limits) as in the §7.15 microbenches. fp32 cells per Candidate 4a.

**P4 — Decision rule.** Promote a candidate iff: (i) P0 clean; (ii) P1 s/step ≤ 0.8× champion at equal-or-better max|ΔV|; (iii) P2 flat; (iv) no memory cell exceeds the never-assemble envelope by >25%. Otherwise record the §7.x and stop — per the standing rule that every default in this stack is the output of a recorded measurement.

---

## 4. Explicitly rejected directions (tied to measured dead ends)

1. **Any block-Jacobi base at the split regime.** Never-assemble BJ blocks are single-owner-tile Sᵢ slices — cold CG stagnates (rel-res 0.32 → 0.27 over 800 iterations); even *true* principal-submatrix blocks, isolated by direct intervention, converge (262 iters) but lose 7.7× to jacobi+PoU (34), at 107 GB peak. The byte-budget guard is a memory guard, not a numerics guard (the BRCM silent-hang incident). BJ is dead as a base; its only legacy is the permuted-GEMV apply machinery, which Candidate 1 reuses.
2. **GenEO-lite as shipped.** Zero iteration benefit in every measured cell (cold and warm, additive and deflated), +70 s prepare → `geneo_k=0` shipped. Re-opening spectral enrichment requires a *different eigenproblem* (proper Neumann/weighted pencils per the GenEO literature [8]), and only after Candidate 1's base changes the fine space — not a re-run of the same experiment.
3. **A-DEF1 / literal-formula deflation variants and true A-DEF2.** The un-projected-matvec taxonomy members stall or silently stagnate on the real PDN fixture (tracked residual satisfied, true residual not — twice, independently reimplemented); true A-DEF2 ties DEF at the production regime, regresses 31% on bj bases, and maxiter-fails on the ill-conditioned fixture. DEF ships; the rejected variants live on only as regression tests.
4. **Growing Z within the current architecture.** Deflation already extracts the theoretical maximum for the current Z (measured: warm gain equals it); enlarging Z hits the dense T′² coarse-solve and SZ-memory walls long before the ≤10-warm-iters target. Better *content* (Candidate 2), not more columns — until the P2/1000-block trigger flips Candidate 5 on.
5. **Assembled-path revival.** Never-assemble won 5× on s/step and 3.9×/2.3× on prepare/RSS; the assembled S is essentially dense at split regimes (Σnₚᵢ² comparable to n²). Any candidate requiring assembled S (including "just use scipy/PETSc on S_global") is out at this scale on this host.
6. **Periodic re-projection as a robustness feature.** `interface_deflated_reproject_every=50` measured dose-dependently harmful on solves long enough to reach it; it never fires at production iteration counts and must not be leaned on by any new candidate.
7. **fp32 below its rtol floor, and mixed-dtype GEMV.** fp32 residual floor ~1e-7 relative (guard enforced); mixed dtype falls off the BLAS fast path at ~10× cost. Constraints, not options.

---

## 5. Bottom line

The literature and our measurements point the same direction: the champion's coarse level is done (deflation provably extracted it), and the next factor of ~2–4× in iterations lives in the fine space, exactly where BDD → BDDC history put it — weighted per-tile Schur solves [4][2], energy-minimized averaging (the ‖E‖²_S that *is* the bound [1]), average-type constraints rather than vertex-type ([3]'s polylog-vs-linear gap), and contrast-aware scaling held in reserve [7][4]. Candidate 1 (NN base under the existing DEF loop) is the highest-expected-value step: moderate effort, reuses the two hardest-won pieces of local engineering (tilewise threading, permuted-GEMV apply), doubles per-iteration cost against a predicted 2–4× iteration cut, and is the first configuration with a theoretical claim to flat 64 → 107 → 1000-block scaling [3][1]. Candidates 2–4 stack on it; Candidate 5 waits for the 1000-block trigger; PCBDDC/HPDDM serve as oracles, not dependencies [6][8].

---

## 6. References and reliability notes

Verification status: [1] and [3] verified against full text this session; [2] primary text **inaccessible** (DOI → epubs.siam.org returned 403; no legal mirror found) — its claims here are drawn from a secondary source (Sousedík & Mandel, arXiv:0802.4328) that characterizes it as introducing BDDC as a primal FETI-DP alternative with corner/average constraints, and attributes the O(log²(1+H/h)) BDDC proof to the companion Mandel–Dohrmann 2003 convergence paper; which of the two 2003 papers formally contains the proof could not be verified. [3] is purely theoretical (no experiments; its "Algorithm A is not competitive" remark cites others' numerics). [1] is purely algebraic (no experiments, no log² bound — see §1.2). [4]–[8] are cited from titles/abstracts and library documentation summaries, not full-text re-verification; the engineering figures quoted for [6] (8232 ranks, 10⁸ contrast, >0.5B DOFs) and [8] (GenEO + recycled Krylov in PETSc) are the papers' reported results.

- [1] Mandel, Dohrmann, Tezaur — *An Algebraic Theory for Primal and Dual Substructuring Methods by Constraints* (BDDC/FETI-DP under minimalist assumptions). https://arxiv.org/pdf/0708.4031
- [2] Dohrmann — *A Preconditioner for Substructuring Based on Constrained Energy Minimization*, SIAM J. Sci. Comput., 2003. https://doi.org/10.1137/S1064827502412887
- [3] Klawonn, Widlund, Dryja — *Dual-Primal FETI Methods for Three-Dimensional Elliptic Problems with Heterogeneous Coefficients*, SIAM J. Numer. Anal., 2002. https://epubs.siam.org/doi/10.1137/S0036142901388081
- [4] Mandel, Brezina — *Balancing Domain Decomposition for Problems with Large Jumps in Coefficients*, Math. Comp., 1996. https://www.ams.org/journals/mcom/1996-65-216/S0025-5718-96-00757-0/S0025-5718-96-00757-0.pdf
- [5] Toselli, Widlund — *Domain Decomposition Methods — Algorithms and Theory*, Springer, 2005. https://link.springer.com/book/10.1007/b137868
- [6] Zampini — *PCBDDC: A Class of Robust Dual-Primal Methods in PETSc*, SIAM J. Sci. Comput., 2016. https://epubs.siam.org/doi/10.1137/15M1025785
- [7] Beirão da Veiga, Pavarino, Scacchi, Widlund, Zampini — *Isogeometric BDDC Preconditioners with Deluxe Scaling*, 2014. https://doi.org/10.1137/130917399
- [8] Jolivet, Roman, Zampini — *KSPHPDDM and PCHPDDM: Extending PETSc with advanced Krylov methods and robust multilevel overlapping Schwarz preconditioners*, Comput. Math. Appl., 2021. https://www.sciencedirect.com/science/article/pii/S0898122121000055

Local measurement sources: `docs/brcm_distributed_runtime_optimization.md` §7.4–§7.15, `docs/cg_implementations_guide.md` §3–§6, `src/distributed/interface_deflation_notes.md`, `src/distributed/CLAUDE.md` (B2/Stages 1–3, block-Jacobi hazards).
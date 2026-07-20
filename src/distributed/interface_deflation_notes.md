# `interface_iterative.py` — measurement history & ratification records

Round-2 code review finding 12 (file size discipline, `CLAUDE.md`'s "~800 lines
per file" guideline): this file holds the detailed measurement/ratification
narrative that used to live inline in `interface_iterative.py`'s module
docstring — decision history, rejected-formula walkthroughs, and specific
timing numbers that are useful institutional memory but not needed to *use*
or *maintain* the shipped code day to day. The module docstring keeps a short
summary of each topic with a pointer here; nothing here changes behavior.

Round-3 code review finding 11 (same file-size discipline, continued): the
hand-rolled deflated-PCG *code* itself (`_deflated_pcg`, `_is_breakdown`,
`_BREAKDOWN_EPS`, `DEFLATED_DEBOUNCE_REARM_FALLBACK_ITERS`) has since moved
out of `interface_iterative.py` into its own module,
[`interface_deflated_pcg.py`](interface_deflated_pcg.py) (pure mechanical
move, zero logic change — it was already a self-contained, module-level
function of its own arguments with no `InterfaceCGSolver` dependency).
`interface_iterative.py` imports and re-exports these names unchanged, so
every reference to `_deflated_pcg`/etc. below (and in the docstrings) still
resolves the same way whether read via `interface_iterative` or
`interface_deflated_pcg` directly. This notes file's algorithm/measurement
narrative is unaffected by the move — it describes the numerics, not the
file layout.

## Stage 2 — threaded tilewise matvec / block-Jacobi apply (measured detail)

**Threaded tilewise matvec (design + measured scatter decision).**
`matvec_threads` controls a persistent `ThreadPoolExecutor` (lazy-built on
first tilewise matvec/BJ-apply call, closed via `InterfaceCGSolver.close` or
a `weakref.finalize` safety net). Work is a static LPT (longest-processing-
time) partition of tiles by `n_ports**2` (proxy for per-tile GEMV cost)
across `matvec_threads` bins, computed once at construction.

Scatter design: each thread accumulates into a **compact buffer** sized to
the UNION of global interface indices its assigned tiles touch (precomputed
once), not a full `(n_threads, n)` array that must be zero-filled and
reduced via `acc.sum(axis=0)` every call. Stage 0 measured the naive
full-row-accumulator design INVERT above 8 threads (zero-fill + reduction
cost growing with thread count outpacing the GEMV work); the compact-
touched-index design matched the naive design's throughput at ≤8 threads and
pulled ahead at 16-32 threads (e.g. ~26.6 ms vs ~31.6 ms at 32 threads on a
150K-interface/60-tile synthetic), while being architecturally immune to the
zero-fill-scales-with-n_threads·n failure mode. `matvec_threads='auto'`
resolves to `min(8, cpu_count, n_tiles)` — 8, not 32 — because Stage 0
measured best throughput at 8 threads on the BRCM-class proxy; the original
Stage 2 sketch's `min(32, ...)` predates that measurement.

**fp32 critical path (BRCM host is CPU-only).** `matvec_dtype='float32'`
stores each tile's `S_i` as float32 and casts the (small) gathered `x` slice
to float32 per tile so the GEMV itself (`S_i @ x_local`) stays entirely in
float32 and hits the BLAS `sgemv` fast path; the float32 result is then
accumulated into the float64 running total. A naive mixed-dtype `S_i`
(float32) times `x` (float64) call falls OFF the BLAS fast path in numpy
(silently promotes to a slow elementwise loop) — Stage 0 measured this ~10x
SLOWER than fp64, which is why both operands of the GEMV must share dtype.
Measured on this host (150K-interface/60-tile synthetic, 8 threads): fp32
~1.7-2.0x the fp64 throughput, matching the plan's "at least ~2x"
expectation. fp32 residual floor is ~1e-7 relative, so
`matvec_dtype='float32'` is enforced to pair with `rtol >= 1e-7` (raises
`ValueError` otherwise; override with `strict_dtype_rtol=False`).

**SPD-safe block-Jacobi fallback.** Singular/indefinite owned blocks
previously fell back to `np.linalg.pinv`, which can retain small NEGATIVE
eigenvalues from FP noise on a numerically indefinite block — silently
voiding CG's convergence guarantee (a preconditioner must be SPD). The
fallback now eigendecomposes the (symmetric) block, clips eigenvalues to
`>= eps_rel * lambda_max`, and applies `V @ diag(1/w_clipped) @ V.T` —
guaranteed PSD (in fact SPD after clipping away non-positive modes).

**BJ-apply perf fix (permuted-contiguous GEMV).** Stage 2 measured the
threaded block-Jacobi apply at only 1.4x speedup (701 ms vs 990 ms serial,
64 blocks / n=167,659) despite `cho_solve` itself releasing the GIL (LAPACK
`dpotrs`): the per-block fancy-index gather (`x[global_idx]`) and scatter
(`result[global_idx] = ...`) do NOT release the GIL and are RANDOM-access
(cache-miss-bound) at a cost comparable to the O(k²) solve itself at these
block sizes — serializing across threads even though the solve itself
parallelizes fine. The fix: do exactly ONE gather and ONE scatter per
`apply()` call (not one per block) via a single global permutation array
(block-Jacobi ownership is a partition, so concatenating every block's
global index array is a valid partial permutation of `[0, n)`), and
materialize each block's dense (pseudo-)inverse ONCE at build time
(replacing — not duplicating — the cho factor payload, so total memory
stays the same order as before) so each block's apply is a single
contiguous-slice GEMV (BLAS-2, releases the GIL) instead of two triangular
solves plus scattered indexing. Measured on a synthetic proportional to the
mi200k_v2 regime (64 blocks, ~2674 avg block size, n~171K, 8 threads, BLAS
pinned to 1 thread throughout via `threadpool_limits`): the OLD design's own
threading is NEGATIVE (serial ~255 ms, 8-thread ~470-490 ms — concurrent
`cho_solve` calls across threads contend rather than parallelize, even with
per-thread disjoint data); the fix is faster BOTH serially (~120 ms, ~2.1x —
a streaming GEMV beats two triangular-solve passes even single-threaded) AND
when threaded (~47-52 ms, a further ~2.3-2.5x over its own serial, ~9-10x
over the OLD design's threaded number). See `_build_block_jacobi`'s
`_bj_perm`/`_bj_offsets`/`_bj_solve_threaded` for the implementation.

## A-DEF2 work package — naming history / ratification record

The original work-package spec's enum called the deflated apply mode
`'adef2'` and wrote out an in-line formula that turned out to be a
TRANSCRIPTION ERROR (see "Why DEF, not the literal formula" below) — in the
Tang/Nabben/Vuik/Erlangga taxonomy, "apply `P` to `r` before `M_base^-1`
with an un-projected matvec" is actually **A-DEF1**, known in the
literature to be non-robust with inexact/pseudo-inverse coarse solves
(matches the observed stall on the real PDN fixture).

The coordinator ruled that the TRUE A-DEF2 preconditioner (a genuinely
different combination: `z = (I - QS) M_base^-1 r + Q r` inside a STANDARD
unprojected-matvec PCG, with the mandatory projected starting vector
`x0' = Q b + (I - QS) x0`) be implemented as a third candidate and measured
head-to-head against DEF and the additive form on the `netlist_multi_tile`
gate script (both the 'natural' block-Jacobi-base and 'jacobi-forced'
production-mirroring scenarios) plus the repo's ill-conditioned realistic-
`T'/n`-ratio unit fixture.

Result: true A-DEF2 TIES DEF exactly in the production-representative
'jacobi-forced' scenario (16.75 == 16.75 warm iters/step — consistent with
the DEF1/A-DEF2 mathematical-equivalence theorem when `x0` is correctly
projected and the base preconditioner is diagonal), but REGRESSES badly in
the 'natural' (block-Jacobi-base) scenario (98.10 warm iters/step vs. DEF's
83.00 and additive's 74.65 — fails the `<= additive * 1.05` non-regression
bar) and FAILS TO CONVERGE outright (hits `maxiter`) on the realistic-
`T'/n`-ratio ill-conditioned PoU-only fixture where DEF and additive both
converge in a few hundred to ~2000 iterations.

Selected BY DATA: DEF ships, under the honest name `'deflated'` (not
`'adef2'` — the setting no longer claims to be running an algorithm it
isn't, so the self-disclosure machinery an earlier revision of this module
needed — `ADEF2_ACTUAL_ALGORITHM`, a `[adef2:def1]` label tag, a one-time
runtime WARNING — has been removed as unnecessary). The true-A-DEF2
implementation that was measured and rejected (`_true_adef2_pcg`) lives in
`tests/distributed/test_interface_coarse.py` alongside the other previously-
rejected variant (`_literal_spec_adef2_pcg`) — kept for the regression
coverage of the "warm `x0` must be projected" lesson (see
`TestTrueADef2X0ProjectionRegression`), not shipped as a selectable mode.
Its `Q S v` helper (`apply_QS`) moved out of `src` alongside it (round-2
code review finding 10: it had no other caller) — it is now a test-local
free function (`_apply_QS(coarse, v)`) in the same test file, not a
`CoarseSpace` method.

## Why DEF, not the literal formula

Two literal-taxonomy candidates were tried and rejected — both keep the
matvec `S p` un-projected and instead fold `Q` into the preconditioner
apply.

1. The spec's in-line formula (`P` applied to `r` before `M_base^-1`)
   STALLS on the real `netlist_multi_tile` PDN fixture (plateaus at rel-res
   ~1e-4, never reaching rtol).
2. Applying `P` AFTER `M_base^-1` instead reproduces the direct solution on
   every SYNTHETIC chain fixture tried, but SILENTLY STAGNATES on the same
   real fixture: `info=0` is returned with the tracked residual satisfied
   while the true residual (freshly recomputed via `b - S @ x`) never
   actually approaches the rtol target.

Root cause: for a non-DEF member of this taxonomy, `Z^T r_k = 0` is not
preserved by the plain (un-projected) CG recurrence beyond the first
iterate, because the resulting `M^-1_ADEF2` is not symmetric as a matrix,
so the standard 3-term PCG recurrence has no guaranteed conjugate search
directions; chain fixtures mask this because their partition-of-unity
column count T' sits close to n (near-degenerate), so deflation alone does
most of the work regardless of formula.

A transpose-corrected variant (`P^T = I - QS` instead of `P = I - SQ`) was
also tried and diverges outright on the same fixture — see
`tests/distributed/test_interface_coarse.py`'s
`TestTransposeCorrectedADef2AlsoFails`. Both the original check and a
from-scratch independent reimplementation (`_literal_spec_adef2_pcg` /
`TestLiteralADef2FormulaIndependentlyReverified` in the same test file)
reproduce the stall, so this is not an implementation artifact of one
attempt.

What IS implemented ("DEF") — see the module docstring's "What is actually
implemented" paragraph and `_deflated_pcg`'s own docstring for the live
formula and per-iteration cost accounting.

## Decoupled GenEO extraction — measured detail

`InterfaceCGSolver._extract_geneo_decoupled` (Deliverable 1 of the A-DEF2
work package) runs GenEO-lite enrichment even when the base preconditioner
downgrades to diagonal `'jacobi'` (previously the byte-budget-downgrade path
never cho-factored any block, so `self._geneo_pairs` stayed permanently
empty). §7.8's probe independently PROVED the cho-factored ownership blocks
at the mi200k_v2 split regime (64 tiles, giant skewed ownership blocks)
carry genuine ~1e-10-relative near-null eigendirections — real GenEO
material that was being discarded unused for exactly the split regime that
benefits from it most.

The per-block memory guard exists because §7.8/§7.9 measured ~100k-row
single ownership blocks at mi200k_v2 — an unguarded dense formation there
would attempt an ~80+160 GB allocation. §7.9 measured 97 s for the
equivalent factor+eigsh sweep over 64 blocks including a 13,834-row block —
an acceptable one-time `prepare()` cost, logged as its own phase.

## Defaults flipped by measurement (2026-07-20)

A second coordinator ruling, measured on `mi200k_v2` (64 tiles, 168,586
unknowns; full matrix in `scripts/benchmark/microbench/results_deflated_matrix_mi200k.json`
and the PoU-only addendum in
`scripts/benchmark/microbench/results_deflated_pou_only_addendum_mi200k.json`),
flipped two `interface_coarse.py` defaults:

* **`DEFAULT_APPLY_MODE`: `'additive'` → `'deflated'`.** `'deflated'` beat
  `'additive'` in EVERY measured cell:
  * Cold DC: 118 → 79 iters @ rtol 1e-12; 70 → 34 iters @ rtol 1e-8
    (PoU-only, no GenEO enrichment on either side).
  * Warm transient: 23.6 → 20.0 iters/step @ rtol 1e-8.
  * Accuracy equal or better: 183 nV vs 253 nV max `dV`.

  `'additive'` remains fully supported and selectable explicitly via
  `interface_coarse_apply_mode='additive'`.

* **`DEFAULT_GENEO_K`: `4` → `0`.** GenEO enrichment contributed ZERO
  iteration benefit in every measured cell, cold and warm, additive and
  deflated alike (118≡118, 70≡70, 23.6≈23.4, 20.0≡20.0 iters — GenEO vs.
  PoU-only, same apply mode), while costing ~70 s per `prepare()` at the
  split regime (198 s GenEO-enabled `dc_prepare` vs. 125–129 s PoU-only).
  The GenEO machinery (`_extract_geneo_decoupled`, the per-block memory
  guard, the GenEO-then-disable degrade ladder) stays fully functional and
  opt-in via `interface_coarse_geneo_k > 0` — nothing about how GenEO
  enrichment itself works changed, only which value ships by default.

* **`interface_warm_start_extrapolation` stays `False`** (opt-in) — measured
  a modest 1.13x additional speedup on top of the deflated default, but with
  cross-family seeding subtleties (linear extrapolation across a
  cold-start-vs-warm-transient boundary) that did not clear the bar for a
  silent default flip.

No solver logic changed in this pass — both constants are simple defaults,
and every mode/knob they gate (`apply_mode='additive'`,
`interface_coarse_geneo_k=<n>`) remains reachable and behaves exactly as
before.

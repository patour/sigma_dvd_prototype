"""Iterative interface solve for large-scale DDM.

Provides InterfaceCGSolver, a wrapper around scipy.sparse.linalg.cg that
replaces the direct CHOLMOD/SuperLU factorization of the global Schur
complement S_global.  Two coordinator-side matvec modes are supported:

  'assembled'  -- matvec on the assembled sparse S_global (removes the
                  dominant ~100-300 GB CHOLMOD factor; keeps ~50 GB S_global).
  'tilewise'   -- matvec = sum_i P_i^T (S_i @ (P_i @ x)) using the per-tile
                  dense S_i blocks + tile_index_maps.  Avoids global assembly
                  entirely (removes both the ~50 GB S_global and its factor).

Worker-side RPC matvec (S_i lives on the worker) is out of scope -- noted as
future work for B4.

Preconditioners (pluggable, via ``preconditioner`` keyword):
  'block_jacobi' (default) -- block-Jacobi from the owner's principal S_i
                              submatrix for each interface node.  Ownership:
                              first tile whose tile_index_map includes the node.

                              Memory note: each owned block is a k_i x k_i
                              dense Cholesky factor.  Total factor memory scales
                              as Sum(k_i^2) ≈ n^2/T for T balanced tiles.  At
                              n=1M interface nodes with T=1000 tiles this is
                              ~8 GB for factors alone — the same coordinator-
                              memory class CG is designed to avoid.  For
                              netlist_sampled (n~2K, max block 478) the factor
                              is ~5.9 MB (negligible).

                              When the estimated factor memory exceeds
                              ``BLOCK_JACOBI_MAX_FACTOR_BYTES`` (default 4 GB),
                              _build_block_jacobi() automatically falls back to
                              'jacobi' (diagonal) and logs a WARNING.  For
                              1M-interface systems with few tiles, prefer
                              'jacobi' explicitly or reduce tile size via B1
                              splitting so T is large enough that individual
                              block sizes are manageable.
  'jacobi'                  -- diagonal of S_global (assembled) or the summed
                              tile S_i diagonals (tilewise).  Cheap, scales
                              to any interface size; weaker than block_jacobi
                              but does not use O(n^2/T) memory.
  'none'                    -- identity (no preconditioning).
  'amg'                     -- algebraic multigrid via pyamg (lazy import,
                              skipped gracefully when pyamg is not installed).
  'two_level'               -- Stage 3: block_jacobi PLUS an additive coarse-
                              space correction (partition-of-unity + GenEO-
                              lite columns) -- fixes the block_jacobi CG
                              stagnation Stage 2 measured at large split
                              regimes (genuine near-null eigendirections in
                              the cho-factored ownership blocks).  See
                              ``interface_coarse.py`` and this class's
                              "Stage 3" docstring section below.  'auto' (the
                              resolved default for CG+tilewise, see
                              :func:`resolve_preconditioner`) picks this
                              automatically; small/'assembled' systems keep
                              the legacy 'block_jacobi' default.

Save / load semantics for CG mode
-----------------------------------
CG mode has no factor to save (that is the whole point).  On save():
  - S_global is stored as usual (used by assembled mode; can also be used to
    refactor into direct mode later).
  - The ``interface_solver`` setting is stored so that load() + refactor()
    reconstructs a CG solve callable, not an LU factor.
On load() + refactor() in CG mode:
  - ``refactor`` rebuilds a CG callable from S_global (for assembled mode) or
    from S_global + tile_schur_complements (for tilewise mode).
  - Per-tile S_i blocks are NOT saved (too large; always recomputed).
  - Simplest correct behaviour: after load(), callers must call factor()
    for tilewise mode (which re-runs factor_and_compute_schur on workers),
    or refactor() for assembled mode (which rebuilds the CG op from S_global).

Adjoint note
------------
``solver_adjoint.py``'s ``analyze_adjoint_static``/``analyze_adjoint`` call
``ctx.interface_lu(global_rhs)`` generically -- there is no special-casing of
CG vs direct, and no check of ``ctx._interface_solver_mode``.  This works
transparently in EVERY mode (direct, CG/assembled, CG/tilewise) because
``interface_lu`` is always a plain linear-solve callable with the same call
signature, regardless of what builds it: a direct LU factorization, an
``InterfaceCGSolver`` wrapping the assembled ``S_global``, or an
``InterfaceCGSolver`` wrapping per-tile ``S_i`` blocks via the tilewise
matvec.  Tilewise CG adjoint is not a special/unsupported case that needs
forcing to direct -- the D1 (pad-port kept-position slicing) and D2
(direct-stamped ``S_extra``) fixes make the tilewise matvec exactly correct,
so ``ctx.interface_lu`` returns the same answer as direct/assembled mode for
the adjoint's DC-shaped solves too (same coefficient matrix as the forward
DC solve).

Stage 2 -- threaded tilewise matvec + threaded block-Jacobi apply
-------------------------------------------------------------------
D1 fix (pad-port corruption)
    ``tile_index_maps`` (built by ``result_factorization.py``) filters
    Dirichlet/pad nodes out of each tile's port list, but the per-tile dense
    Schur block ``S_i`` returned by the tile worker still has ALL ports
    (including pads) as rows/columns.  Pairing the filtered map with the
    full ``S_i`` is a dimension mismatch (or, worse, silent index
    corruption if sizes happen to coincide).  The fix lives at the call
    site (``result_factorization.py``'s ``_kept_position_slice``, reused
    here as ``kept_position_slice``): each tile's ``S_i`` is sliced to
    ``S_i[np.ix_(pos, pos)]`` -- dropping exactly the rows/columns whose
    node is NOT in ``interface_node_to_idx`` -- BEFORE it ever reaches this
    class.  ``InterfaceCGSolver.__init__`` additionally asserts
    ``S_i.shape[0] == len(tile_index_maps[tid])`` per tile so any caller
    that forgets the slicing step fails loudly (a clear ``ValueError``)
    instead of a cryptic numpy broadcast error deep inside a matvec.

Threaded tilewise matvec (design + measured scatter decision)
    ``matvec_threads`` controls a **persistent** ``ThreadPoolExecutor``
    (lazy-built on first tilewise matvec/BJ-apply call, closed via
    :meth:`InterfaceCGSolver.close` or a ``weakref.finalize`` safety net),
    partitioning tiles across threads by a static LPT (longest-processing-
    time) heuristic on ``n_ports**2``.  ``matvec_threads='auto'`` resolves
    to ``min(8, cpu_count, n_tiles)`` (Stage 0 measured best throughput at
    8 threads on the BRCM-class proxy).  Full scatter-design rationale and
    measured numbers: ``interface_deflation_notes.md``, "Stage 2" section.

fp32 critical path (BRCM host is CPU-only)
    ``matvec_dtype='float32'`` stores each tile's ``S_i`` as float32 (both
    GEMV operands must share dtype to hit BLAS's fast path -- a naive
    mixed-dtype call falls off it, ~10x slower).  fp32 residual floor is
    ~1e-7 relative, so ``matvec_dtype='float32'`` is enforced to pair with
    ``rtol >= 1e-7`` (``ValueError`` otherwise; override with
    ``strict_dtype_rtol=False``).  Measured throughput: see notes.md.

SPD-safe block-Jacobi fallback
    Singular/indefinite owned blocks previously fell back to
    ``np.linalg.pinv``, which can retain small NEGATIVE eigenvalues from FP
    noise on a numerically indefinite block -- silently voiding CG's
    convergence guarantee (a preconditioner must be SPD).  The fallback now
    eigendecomposes the (symmetric) block, clips eigenvalues to
    ``>= eps_rel * lambda_max``, and applies ``V @ diag(1/w_clipped) @ V.T``
    -- guaranteed PSD (in fact SPD after clipping away non-positive modes).

Stage 3 -- two-level coarse-space preconditioner ('two_level')
-------------------------------------------------------------------
``preconditioner='two_level'`` layers an additive coarse-space correction on
top of block-Jacobi: ``M^-1 = M_bj^-1 + Z S_c^+ Z^T`` (see
``interface_coarse.py`` for the full derivation/motivation -- Stage 2
measured cold block-Jacobi CG STAGNATING at the mi200k_v2 split regime due to
genuine near-null eigendirections in the cho-factored ownership blocks). The
coarse space (``Z``, partition-of-unity + GenEO-lite columns; ``S_c``, its
small eigh-factored pseudo-inverse) is built AFTER ``self._linear_op`` exists
(``_augment_with_coarse_space``, called from ``__init__`` right after
``_build_linear_op()``) because ``S Z`` uses the solver's own matmat --
building the block-Jacobi component first (as for plain ``'block_jacobi'``)
and layering the coarse term on second avoids disturbing the existing
fp64-read-before-fp32-cast ordering invariant documented above. Never
persisted (rebuilt on every ``factor()``/``refactor()``, like the per-tile
Schur blocks it derives from). If T' (PoU + GenEO columns) exceeds ``interface_coarse_max_cols``, falls
back to the PoU-only rung first (WARNING, GenEO columns dropped for this
solve -- PoU-only is a strictly smaller space and typically still fits).
Degrades further to the plain (possibly memory-budget-downgraded-to-'jacobi')
base preconditioner with a WARNING -- never raises prepare() -- only when the
coarse build itself fails outright (PoU-only T' ALSO exceeds
``interface_coarse_max_cols``, no usable ``tile_index_maps``, or ``S_c`` has
no positive spectrum).

A-DEF2 work package -- deflated apply mode ('interface_coarse_apply_mode=deflated')
-------------------------------------------------------------------------------------
The additive two-level form (above) is the weak link *warm*: the coarse and
fine (block-Jacobi/jacobi) spaces stay coupled, so a warm-started transient
step's iteration count only drops modestly (measured 29.2 -> 23.6 iters/step,
jacobi-alone -> two_level(jacobi+PoU)) even though the coarse space alone
repairs cold-solve stagnation completely. This apply mode removes
``range(Z)`` from the iteration exactly via a PROJECTED matvec, rather than
adding a correction on top of it -- the deflation/"DEF" member of the
Tang/Nabben/Vuik/Erlangga taxonomy (Tang, Nabben, Vuik & Erlangga, "A
comparison of two-level preconditioners based on multigrid and deflation",
2009).

**Naming / ratification record** (full measured numbers and the true-A-DEF2
head-to-head comparison: ``interface_deflation_notes.md``): the setting
ships as ``'deflated'``, not ``'adef2'`` -- the coordinator measured the
literal Tang/Nabben/Vuik/Erlangga A-DEF2 preconditioner head-to-head against
the deflation ("DEF") member of the same taxonomy and DEF won on data (true
A-DEF2 regressed warm iterations on the 'natural' block-Jacobi-base scenario
and failed to converge outright on the realistic-ratio ill-conditioned
fixture). The true-A-DEF2 implementation that was measured and rejected
(``_true_adef2_pcg``) lives in ``tests/distributed/test_interface_coarse.py``
alongside the other previously-rejected variant (``_literal_spec_adef2_pcg``)
-- kept for the regression coverage of the "warm ``x0`` must be projected"
lesson (see ``TestTrueADef2X0ProjectionRegression``), not shipped as a
selectable mode. Its ``Q S v`` helper (``apply_QS``) moved out of src
alongside it (round-2 code review finding 10: it had no other caller) -- now
a test-local free function (``_apply_QS(coarse, v)``), not a ``CoarseSpace``
method.

Notation: ``Q = Z S_c^+ Z^T`` (never materialized; every application goes
through :meth:`interface_coarse.CoarseSpace.apply`/``apply_with_SQ``, reusing
the same eigenfactorization the additive ``two_level`` mode already
computes), ``P = I - S Q``.

**Implementation location**: the hand-rolled loop itself
(:func:`_deflated_pcg`, plus :func:`_is_breakdown`/``_BREAKDOWN_EPS`` and
``DEFLATED_DEBOUNCE_REARM_FALLBACK_ITERS``) lives in
``interface_deflated_pcg.py`` (round-3 code review finding 11 -- moved out
of this file, which had grown well past the repo's ~800-line-per-file
convention, once the numerics stabilized; pure mechanical move, no logic
change) and is imported back into this module's namespace below, so every
``_deflated_pcg``/``_is_breakdown``/``_BREAKDOWN_EPS`` reference in this
docstring and the rest of this file resolves unchanged.

**Why DEF, not the literal formula** (full derivation/rejected-formula
walkthrough: ``interface_deflation_notes.md``): two literal-taxonomy
candidates were tried and rejected, both keeping the matvec ``S p``
un-projected and folding ``Q`` into the preconditioner apply instead -- the
spec's in-line formula STALLS on the real ``netlist_multi_tile`` PDN
fixture, and applying ``P`` after ``M_base^-1`` SILENTLY STAGNATES on the
same fixture (``info=0`` with the tracked residual satisfied while the true
residual never approaches rtol). Root cause: for a non-DEF member of this
taxonomy, ``Z^T r_k = 0`` is not preserved by the plain (un-projected) CG
recurrence beyond the first iterate (the resulting operator is not
symmetric, so standard 3-term PCG has no guaranteed conjugate search
directions); chain fixtures mask this since their PoU column count T' sits
close to n. See ``tests/distributed/test_interface_coarse.py``'s
``TestTransposeCorrectedADef2AlsoFails`` /
``TestLiteralADef2FormulaIndependentlyReverified`` for the independent
reproductions.

**What is actually implemented ("DEF")**: the matvec ITSELF is projected --
``w = P(S p) = S p - S Q(S p)`` (one ``apply_with_SQ`` call on the ALREADY-
computed ``S p``, so this is CHEAPER per iteration than either rejected
formula, not more expensive) -- the preconditioner apply is the PLAIN base
(``z = M_base^-1 r``, no ``Q`` term; deflation is carried entirely by the
projected matvec), and the solution is RECOVERED from the CG iterate ``y``
(which solves the projected system ``(P S) y = P b``) via ``x = y + Q(b - S
y)`` rather than accumulated directly as ``x += alpha * p`` (that
accumulation is exactly what breaks the ``r = b - S x`` identity once the
matvec is projected). Because the matvec's own projection makes ``Z^T (P v)
= 0`` an IDENTITY for any ``v``, ``Z^T r_k = 0`` holds BY CONSTRUCTION at
every CG iterate -- not merely "usually true" -- which is why re-projection
(below) is genuine hygiene here, not a correctness requirement.

* **Initial setup** (inside :func:`_deflated_pcg`): ``y0 = x0`` (cold start:
  zero), ``r0 = P(b - S y0)`` via one ``apply_with_SQ`` call. The final
  answer is always recovered as ``x = y + Q(b - S y)`` -- for a cold start
  with 0 CG iterations this degenerates to exactly ``Q b``, the intuitive
  "coarse-only" solve. ``S y`` is tracked incrementally from each
  iteration's already-computed ``S p`` (``S y += alpha * S p``), so recovery
  is one cheap ``coarse.apply`` call, never an extra full matvec.
* **Re-projection** (numerical hygiene, not correctness): every
  ``interface_deflated_reproject_every`` iterations (default 50, ``<= 0``
  disables), :func:`_deflated_pcg` recomputes ``r = P(b - S y)`` from
  scratch instead of trusting the incrementally-updated ``r``, correcting
  floating-point drift in ``Z^T r -> 0`` accumulated over many iterations.
  Verified to make no difference to whether a given fixture converges
  (``reproject_every=0`` still converges correctly on every fixture this
  module's tests exercise) -- only, potentially, to the exact iteration
  count on a very long solve.
* ``SZ`` **retention**: ``S Q v = (S Z) (S_c^+ (Z^T v))`` for whatever vector
  ``v`` the projected matvec/re-projection needs it on (``S p`` per
  iteration, or ``b - S y`` at re-projection/setup) -- apply_mode='deflated'
  retains the dense ``(n, T')`` fp64 ``S Z`` array on the ``CoarseSpace``
  (``interface_coarse.build_coarse_space(..., retain_sz=True)``) so every
  ``S Q`` application in the hot loop is a GEMV pair, never a full-``n``
  matvec of ``S`` itself. Persistent memory cost: one more ``n * T' * 8``
  bytes for the coarse space's lifetime -- folded into the
  ``interface_coarse_max_bytes`` byte guard (see ``build_coarse_space``'s
  ``retain_sz`` docstring). Round-2 code review finding 6 (corrects an
  earlier revision of this paragraph, which claimed the opposite order):
  the existing GenEO-then-disable ladder (drop GenEO columns, then disable
  the coarse space outright) runs FIRST and determines the FINAL T'; the
  retained-SZ byte check is evaluated AFTER, against that final T', using
  the SAME ``3*n*T'*8`` formula/limit the ladder itself just verified fits
  -- so in practice a too-tight budget is caught by the ladder (PoU-only
  degrade or outright disablement) before SZ retention is ever separately
  at risk, and the dedicated "drop SZ, keep additive" degrade is defensive
  (documents the invariant, guards against the accounting ever changing)
  rather than a distinct degradation step an operator should expect to
  observe. fp32 tilewise matvec storage
  (``matvec_dtype='float32'``) does not weaken this: ``SZ`` and ``S_c`` are
  both formed via the SAME fp64-accumulating ``matmat`` the additive mode
  already uses (each tile's fp32 GEMV result is cast back to float64 before
  accumulating into the running ``(n, T')`` sum -- see
  ``_tilewise_matmat``), so the retained ``SZ`` GEMV inside every ``S Q``
  application is fp64 regardless of ``matvec_dtype``.
* **True-residual acceptance**: exactly like the additive/scipy path, the
  strict-mode non-convergence check (and its ``last_cg_rel_residual`` stat)
  recomputes ``rhs - S @ x`` via a FRESH full matvec on the final RECOVERED
  iterate, never trusting the internally-tracked recurrence residual (``r``
  tracks the PROJECTED system, ``P b - P S y``, a genuinely different
  quantity from ``b - S x``) -- shared code between both apply modes (see
  ``InterfaceCGSolver.__call__``). This gate applies on EVERY tentative-
  convergence event inside :func:`_deflated_pcg` (top-of-loop check, the
  final post-update check after ``maxiter`` iterations, and the CG-breakdown
  early-exit), via the internal ``_try_accept`` helper: a disagreement
  (tracked ``r`` says converged, the fresh check does not) is not a hard
  failure, it just means the loop keeps iterating. This is a safety net
  against ``Sy``'s incremental accumulation (``Sy = Sy + alpha * Sp`` every
  iteration, never a fresh ``matvec(y)``) drifting from the true ``S @ y``
  over a very long solve -- re-projection (above) corrects ``Z^T r`` drift
  but not ``Sy`` drift itself. Each acceptance attempt costs two extra
  matvecs (fresh ``Sy = matvec(y)`` plus the ``matvec(x_candidate)``
  residual check) and one ``coarse.apply``; attempts fire when the tracked
  residual first looks converged and then at most once per
  ``_rearm_period`` iterations -- bounded, never per-iteration.
* Degradation: if ``preconditioner`` never resolves to (or degrades away
  from) ``'two_level'``, or the coarse build itself fails/degrades such that
  no ``SZ`` was retained, ``apply_mode='deflated'`` is moot -- ``__call__``
  dispatches to the plain ``scipy.sparse.linalg.cg`` path (whatever base/
  additive ``self._M`` was actually built), so a degraded coarse space is
  zero risk to any existing (non-deflated) code path.
* Composes with ``interface_warm_start_extrapolation`` (below) and with
  every base the ``two_level`` machinery accepts (jacobi-downgraded,
  block_jacobi, or a full GenEO-enriched block_jacobi).
* :attr:`InterfaceCGSolver.preconditioner_label` tags an active deflated
  apply as ``[deflated]`` (``stats['apply_algorithm'] == 'deflated'``);
  the additive label format is byte-identical to pre-work-package Stage 3
  (no tag at all) when ``apply_mode='additive'`` or the coarse build never
  retained ``SZ``.

Warm-start extrapolation ('interface_warm_start_extrapolation')
-------------------------------------------------------------------
Optional, composes with either apply mode. When enabled,
:meth:`InterfaceCGSolver.push_solution_history` (called at the end of every
successful solve in place of the plain ``self._x0 = result.copy()``) seeds
the NEXT solve's warm start with the linear extrapolation ``2*x_prev -
x_prev2`` instead of just ``x_prev`` -- falls back to ``x_prev`` until two
solves have been recorded. Anticipates a slowly-varying transient RHS's next
solution rather than assuming it equals the current one; the transient time
loop itself (``solver_td.py``) is unchanged -- this lives entirely inside
``InterfaceCGSolver``, which already owns ``_x0``.

BJ-apply perf fix (permuted-contiguous GEMV)
-------------------------------------------------------------------
Stage 2 measured the threaded block-Jacobi apply at only 1.4x speedup despite
``cho_solve`` itself releasing the GIL: the PER-BLOCK fancy-index gather/
scatter does NOT release the GIL and is RANDOM-access (cache-miss-bound) at
a cost comparable to the O(k^2) solve itself, serializing across threads.
The fix: exactly ONE gather and ONE scatter per ``apply()`` call (not one per
block) via a single global permutation array, with each block's dense
(pseudo-)inverse materialized ONCE at build time so each block's apply is a
single contiguous-slice GEMV (BLAS-2) instead of two triangular solves plus
scattered indexing -- measured ~9-10x over the old design's threaded number
on a mi200k_v2-proportional synthetic. Full measured numbers:
``interface_deflation_notes.md``. See ``_build_block_jacobi``'s
``_bj_perm``/``_bj_offsets``/``_bj_solve_threaded`` for the implementation.

Tilewise without ever assembling S_global (Finding 0 upgrade; extended to the
transient factor path by the TD never-assemble work package)
    ``interface_drop_s_global`` (bool, default False) changes the CG+tilewise
    factor path from "assemble S_global then optionally free it" to "never
    assemble it at all": interface node ordering + Dirichlet RHS are derived
    from tile PORT NAME LISTS alone (no S_i values), per-tile dense Schur
    blocks are gathered via the streaming (one-tile-in-flight)
    ``call_all_streaming`` protocol with kept-position slicing applied
    immediately per tile (peak coordinator memory never includes an
    unsliced full-S_i-sum or an assembled S_global CSR), and S_extra is
    built by direct stamping (D2, below) rather than a giant
    S_global-minus-sum subtraction.  Requires
    ``model.island_detection_mode == 'summaries'`` (the union-find island
    detector, which never touches S_global) -- with the legacy Schur-BFS
    detector this setting falls back to the normal assembled path with a
    WARNING, since that BFS fundamentally needs S_global's nonzero
    structure.  ``save()`` raises with guidance when S_global was never
    assembled; ``refactor()`` re-gathers via the same streaming protocol.
    Applies identically to the DC factor (``_factor_dc_context_no_s_global``)
    and the transient factor (``_factor_transient_context_no_s_global``) --
    each keeps its own dense per-tile Schur block set (DC's S_i are
    G-based, transient's are A = G + C_coeff*C based; never shared or
    aliased), so both never-assemble contexts can be alive simultaneously
    at the cost of the sum of both block sets.
"""

from __future__ import annotations

import logging
import os
import threading
import time
import weakref
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import threadpoolctl

from . import interface_coarse
# Round-3 code review finding 11: the self-contained A-DEF2 hand-rolled
# deflated-PCG machinery (module-level, no InterfaceCGSolver dependency --
# see that module's docstring) now lives in its own file to keep this one
# under the repo's ~800-line-per-file convention for src/distributed/.
# Re-exported here under their original names so existing call sites
# (``_deflated_pcg(...)`` below) and test imports (``from
# distributed.interface_iterative import _deflated_pcg, _is_breakdown,
# _BREAKDOWN_EPS``) keep working unchanged -- a pure mechanical move.
from .interface_deflated_pcg import (
    DEFLATED_DEBOUNCE_REARM_FALLBACK_ITERS,
    _BREAKDOWN_EPS,
    _deflated_pcg,
    _is_breakdown,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stage 2: threaded matvec/BJ-apply thread-count resolution.
#
# Stage 0 microbenchmark (docs §7.7 / plan report) measured the threaded
# tilewise matvec on the BRCM-class proxy: best throughput at 8 threads,
# INVERTED scaling above 8 (per-thread accumulator zero-fill + final
# reduction growing with thread count outpaces the GEMV work per thread on
# this host's memory subsystem).  8, not 32, is therefore the 'auto' cap.
# ---------------------------------------------------------------------------
DEFAULT_MATVEC_THREADS_CAP: int = 8

# fp32 tilewise matvec residual floor (Stage 0 empirical + this host's
# fp32-vs-fp64 comparison): pairing matvec_dtype='float32' with a CG rtol
# tighter than this is not meaningful (the matvec itself introduces ~1e-7
# relative error) and risks CG failing to converge to its requested
# tolerance.  Enforced in InterfaceCGSolver.__init__ unless
# strict_dtype_rtol=False.
FP32_MATVEC_MIN_RTOL: float = 1e-7

# Round-3 code review finding 1 (CONFIRMED, option (a) chosen -- see the
# module's "A-DEF2" docstring section for the record): apply_mode='deflated'
# gates EVERY acceptance -- not just the scipy path's failure-branch
# diagnostic -- on a FRESH true residual (``b - matvec(x_candidate)``,
# ``_try_accept`` in ``_deflated_pcg``) computed through the fp32 tilewise
# matvec.  At ``rtol == FP32_MATVEC_MIN_RTOL`` (1e-7, the base guard's own
# floor), ``atol_eff`` sits exactly at the fp32 matvec's own ~1e-7 relative
# noise floor, so the fresh check can disagree with the tracked/deflated
# recurrence residual on EVERY attempt (not merely occasionally, as on the
# additive/scipy path, which only recomputes a fresh residual on the
# failure branch and otherwise accepts scipy's own tracked quantity) --
# turning a genuinely-converging deflated solve into a guaranteed
# strict-mode ``RuntimeError`` after burning the full ``maxiter`` budget.
# One extra decade of headroom between the CG target and the matvec's own
# noise floor is enough for the fresh check to reliably agree (the same
# margin the base guard already assumes is sufficient FOR THE TRACKED
# residual alone -- deflated mode adds a second, independent fp32
# evaluation of that same floor via the fresh check, so it needs the
# floor to be one decade further from the target, not the same distance).
# Simpler, more conservative than a bounded-disagreement/WARNING escape
# hatch (rejected: it would let a marginal, potentially-silent accuracy
# regression through on the highest-risk apply mode instead of failing
# loudly at construction time, and duplicates none of the existing
# ``strict_dtype_rtol=False`` override machinery for users who deliberately
# want the tighter floor).
FP32_MATVEC_MIN_RTOL_DEFLATED: float = 1e-6

# A-DEF2 work package: warm-start extrapolation default (also the
# model.settings / CLI default -- interface_warm_start_extrapolation).  This
# is a general CG warm-start refinement (composes with either apply mode),
# not part of the coarse-space build, so its default lives here rather than
# in interface_coarse.py alongside DEFAULT_APPLY_MODE/DEFAULT_DEFLATED_
# REPROJECT_EVERY.  False (unchanged behaviour: next solve's x0 seed is
# simply the just-computed solution) until the coordinator's on/off
# measurement (see InterfaceCGSolver.push_solution_history) motivates
# flipping it.
DEFAULT_WARM_START_EXTRAPOLATION: bool = False

# Round-3 code review finding 11: DEFLATED_DEBOUNCE_REARM_FALLBACK_ITERS
# (fallback re-arm period for _deflated_pcg's debounced fresh-true-residual
# acceptance check) now lives in interface_deflated_pcg.py alongside the
# rest of the self-contained deflated-PCG machinery it exclusively serves --
# imported back below for backward-compatible access as
# ``interface_iterative.DEFLATED_DEBOUNCE_REARM_FALLBACK_ITERS``.

# Historical note (no longer a live code path): an earlier revision of this
# module shipped the deflated apply mode under the setting value 'adef2'
# even though the algorithm actually dispatched was DEF, not literal A-DEF2
# (see the "A-DEF2 work package" docstring section above for the full
# ratification/selection record) -- that mismatch required runtime self-
# disclosure (a module constant `ADEF2_ACTUAL_ALGORITHM = 'def1'`, a
# `[adef2:def1]` preconditioner_label tag, a one-time WARNING). The setting
# is now honestly named 'deflated', so none of that disclosure machinery is
# needed; it has been removed rather than kept as dead code.

# ---------------------------------------------------------------------------
# Threshold for auto-select: use CG when n_interface >= this value.
# Overridable via monkeypatch in tests.
# ---------------------------------------------------------------------------
AUTO_CG_N_INTERFACE_THRESHOLD: int = 200_000

# ---------------------------------------------------------------------------
# Sentinel: factor-memory budget (bytes) for auto-select.
# CHOLMOD supernodal factor is roughly 5-10x S_global memory.
# At 50 GB S_global (138K interface) CHOLMOD factor = 100-300 GB.
# We use a 32 GB budget as the auto cutover (conservative for coordinator).
#
# This module-level constant is the FALLBACK used when no settings dict is
# available (e.g. direct calls to auto_select_interface_solver() without an
# explicit factor_memory_budget_bytes).  When a model.settings dict IS
# available, the ``interface_factor_memory_budget`` setting (Stage 1c)
# resolves to ``min(AUTO_CG_FACTOR_MEMORY_BUDGET_BYTES, 0.4 * total_RAM)``
# via ``resolve_factor_memory_budget_bytes()`` and is passed explicitly by
# callers (result_factorization.py).  See that function for the host-aware
# 'auto' default.
# ---------------------------------------------------------------------------
AUTO_CG_FACTOR_MEMORY_BUDGET_BYTES: int = 32 * 1024 ** 3  # 32 GB

# ---------------------------------------------------------------------------
# Block-Jacobi preconditioner memory budget.
#
# Each owned block is a k_i x k_i dense float64 Cholesky factor.  Total
# factor memory scales as Sum(k_i^2) = n^2/T for T balanced tiles.
# At n=1M interface nodes with T=1000 tiles: ~8 GB just for factors.
# With fewer or larger tiles the cost grows quickly into the coordinator-
# memory class that CG is designed to avoid.
#
# When the estimated block-Jacobi factor memory exceeds this threshold,
# _build_block_jacobi() falls back to 'jacobi' (diagonal) automatically
# and logs a WARNING.  Override via monkeypatch in tests.
#
# Default 4 GB is conservative: keeps the preconditioner build out of the
# multi-GB danger zone while still allowing block_jacobi on all realistic
# netlist_sampled / BRCM-class systems where n_interface is 2-140K.
# (Measured 5.9 MB on netlist_sampled 2015-interface max-block-478 system.)
#
# This module-level constant is the FALLBACK used by InterfaceCGSolver
# instances that don't receive an explicit ``block_jacobi_max_bytes``
# constructor argument (preserves legacy/monkeypatch-based test behaviour).
# When a model.settings dict IS available, the
# ``interface_block_jacobi_max_bytes`` setting (Stage 1c) resolves to
# ``min(8 GB, 0.1 * total_RAM)`` via ``resolve_block_jacobi_max_bytes()``
# and is passed explicitly through the ``block_jacobi_max_bytes`` constructor
# argument by callers (result_factorization.py).
# ---------------------------------------------------------------------------
BLOCK_JACOBI_MAX_FACTOR_BYTES: int = 4 * 1024 ** 3  # 4 GB

# A-DEF2 work package finding 1 (round-1 code review): floor for the
# DECOUPLED GenEO pass's per-block memory cap (see
# InterfaceCGSolver._extract_geneo_decoupled).  That pass forms one
# ownership block dense (~5*k^2*8 bytes worst-case -- covers the
# eigh-fallback path's peak, not just the cheaper cho_factor-succeeds 2x;
# see Finding 7, round 2, and _extract_geneo_decoupled's own estimate
# comment) at a time on the block_jacobi memory-downgrade
# path -- the resolved ``interface_block_jacobi_max_bytes`` is the natural
# per-block ceiling (peak memory during the pass is exactly one block, by
# design), but that setting is sometimes configured far below any real
# per-block cost purely to force the SUM-based downgrade deterministically
# (this module's own test suite does this extensively, e.g.
# ``block_jacobi_max_bytes=1``) rather than to express a genuine memory
# ceiling.  ``max(resolved_max_bytes, this floor)`` -- mirroring the
# existing 'auto' floor pattern in ``resolve_block_jacobi_max_bytes`` --
# means the cap never rejects a block too small to plausibly threaten
# coordinator memory (64 MiB is ~1000x smaller than the ~80-160 GB
# single-block failure scenario the guard exists to catch, and comfortably
# above every block this module's tests ever form).
GENEO_DECOUPLED_MIN_BLOCK_CAP_BYTES: int = 64 * 1024 ** 2  # 64 MiB

# Host-aware 'auto' caps (Stage 1c) — independent of the legacy module
# constants above so that changing these does not alter the fallback path
# monkeypatched by existing tests.
_AUTO_FACTOR_MEMORY_BUDGET_CAP_BYTES: int = 32 * 1024 ** 3   # 32 GB
_AUTO_FACTOR_MEMORY_BUDGET_FRACTION: float = 0.4              # of total RAM
_AUTO_BLOCK_JACOBI_MAX_BYTES_CAP: int = 8 * 1024 ** 3          # 8 GB
_AUTO_BLOCK_JACOBI_MAX_BYTES_FRACTION: float = 0.1             # of total RAM

# Finding 5 (Stage 3): host-aware 'auto' cap/fraction for
# interface_coarse_max_bytes -- same order as the block-Jacobi budget above
# (both guard coordinator-resident dense arrays), no legacy floor (brand
# new setting, no pre-existing fixed constant to never regress below).
_AUTO_COARSE_MAX_BYTES_CAP: int = 8 * 1024 ** 3                # 8 GB
_AUTO_COARSE_MAX_BYTES_FRACTION: float = 0.1                   # of total RAM
# Finding 9 (round 2): NO module-level COARSE_MAX_BYTES_DEFAULT constant
# here -- a prior version snapshotted interface_coarse.DEFAULT_MAX_BYTES at
# IMPORT time (a plain module-level assignment), which defeats
# `monkeypatch.setattr(interface_coarse, 'DEFAULT_MAX_BYTES', ...)` despite
# a comment on the old constant claiming the opposite ("read dynamically so
# test monkeypatches keep working" -- it did not).  The fallback used when
# InterfaceCGSolver receives no explicit interface_coarse_max_bytes now
# reads ``interface_coarse.DEFAULT_MAX_BYTES`` directly at the point of use
# (``_augment_with_coarse_space``, below) -- genuinely dynamic, mirroring
# BLOCK_JACOBI_MAX_FACTOR_BYTES's role for block_jacobi (that constant IS
# safe to snapshot at module level because it is defined in THIS module, so
# tests monkeypatch it directly with no import-time-copy indirection).


_warned_no_psutil = False


def _get_total_ram_bytes() -> int:
    """Return total system RAM in bytes via psutil.

    Falls back to a conservative 64 GB assumption (logging a WARNING once)
    if psutil is unavailable, so host-aware 'auto' sizing degrades
    gracefully rather than crashing.
    """
    global _warned_no_psutil
    try:
        import psutil  # lazy import; declared in pyproject.toml [project] deps
        return int(psutil.virtual_memory().total)
    except Exception:
        if _warned_no_psutil:
            return 64 * 1024 ** 3
        _warned_no_psutil = True
        logger.warning(
            "psutil unavailable; falling back to a conservative 64 GB RAM "
            "assumption for interface-solver memory-budget auto-sizing. "
            "Install psutil or set interface_factor_memory_budget / "
            "interface_block_jacobi_max_bytes explicitly."
        )
        return 64 * 1024 ** 3


def _resolve_memory_budget_bytes(
    setting: Any,
    auto_cap_bytes: int,
    auto_fraction: float,
    floor_bytes: int = 0,
) -> int:
    """Resolve a 'auto' | int | float | numeric-string memory-budget setting.

    'auto' (or None) -> max(floor_bytes, min(auto_cap_bytes,
    auto_fraction * total_RAM_bytes)) via psutil.  ``floor_bytes`` lets a
    caller enforce a legacy minimum so the host-aware auto formula never
    regresses below a previously-fixed constant (finding 9: the block-Jacobi
    budget must never auto-resolve below the legacy 4 GB floor).

    Any other value is coerced via ``int(float(setting))`` so plausible
    numeric strings (including exponent notation like '2e9', which PyYAML
    1.1 parses as a *string* rather than a float) are accepted the same way
    an actual float would be.  Bools are rejected explicitly BEFORE this
    coercion: Python's ``bool`` is a subclass of ``int``, so
    ``int(float(True))`` would silently resolve to 1 byte instead of
    raising -- a classic YAML-1.1 pitfall (``interface_factor_memory_budget:
    yes`` parses to Python ``True``).
    """
    if setting is None or (
        isinstance(setting, str) and setting.strip().lower() == 'auto'
    ):
        total_ram = _get_total_ram_bytes()
        return max(floor_bytes, min(auto_cap_bytes, int(auto_fraction * total_ram)))
    if isinstance(setting, bool):
        raise ValueError(
            f"Invalid memory-budget setting {setting!r}: expected 'auto' or "
            f"an integer byte count (bool is not a valid byte count)."
        )
    try:
        # Finding 8: a YAML '.inf' value (PyYAML parses to float('inf'), a
        # plausible way to spell "unlimited budget") makes int(float(...))
        # raise OverflowError ("cannot convert float infinity to integer"),
        # not TypeError/ValueError -- must be caught here too, or prepare()
        # crashes with a raw traceback instead of this actionable message.
        return int(float(setting))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"Invalid memory-budget setting {setting!r}: expected 'auto' or "
            f"an integer byte count."
        ) from exc


def resolve_factor_memory_budget_bytes(setting: Any = 'auto') -> int:
    """Resolve the ``interface_factor_memory_budget`` setting to bytes.

    'auto' (default) = min(32 GB, 0.4 * total system RAM), computed via
    psutil.  Consumed by ``auto_select_interface_solver`` to decide between
    'direct' and 'cg'.
    """
    return _resolve_memory_budget_bytes(
        setting,
        _AUTO_FACTOR_MEMORY_BUDGET_CAP_BYTES,
        _AUTO_FACTOR_MEMORY_BUDGET_FRACTION,
    )


def resolve_block_jacobi_max_bytes(setting: Any = 'auto') -> int:
    """Resolve the ``interface_block_jacobi_max_bytes`` setting to bytes.

    'auto' (default) = max(4 GB legacy floor, min(8 GB, 0.1 * total system
    RAM)), computed via psutil.  The legacy 4 GB floor (the fixed
    pre-Stage-1c ``BLOCK_JACOBI_MAX_FACTOR_BYTES`` constant) ensures the
    host-aware auto formula never resolves to a SMALLER budget than the
    previously-hardcoded value on hosts with <40 GB RAM -- i.e. 'auto' can
    only ever be as-good-or-better than the legacy behaviour, never a
    silent downgrade of block_jacobi -> diagonal jacobi on modest hosts.
    Consumed by ``InterfaceCGSolver._build_block_jacobi`` to decide whether
    to fall back from 'block_jacobi' to the 'jacobi' diagonal
    preconditioner.
    """
    return _resolve_memory_budget_bytes(
        setting,
        _AUTO_BLOCK_JACOBI_MAX_BYTES_CAP,
        _AUTO_BLOCK_JACOBI_MAX_BYTES_FRACTION,
        floor_bytes=BLOCK_JACOBI_MAX_FACTOR_BYTES,
    )


def resolve_coarse_max_bytes(setting: Any = 'auto') -> int:
    """Resolve the ``interface_coarse_max_bytes`` setting to bytes (Finding 5).

    'auto' (default) = min(8 GB, 0.1 * total system RAM), computed via
    psutil -- same cap/fraction as :func:`resolve_block_jacobi_max_bytes`
    (both bound coordinator-resident dense-array allocations of the same
    order). Guards the two dense ``(n, T')`` fp64 arrays
    ``interface_coarse.build_coarse_space`` allocates (``Z_dense`` + ``SZ``)
    -- unlike ``interface_coarse_max_cols`` (a column-count cap), this scales
    with ``n`` too, so it is the guard that actually bounds coordinator
    memory at the "never assemble S_global" / large-n regime this feature
    must not regress.
    """
    return _resolve_memory_budget_bytes(
        setting,
        _AUTO_COARSE_MAX_BYTES_CAP,
        _AUTO_COARSE_MAX_BYTES_FRACTION,
    )


# NN/BDD work package (Candidate 1 of docs/interface_precond_sota_research.md):
# Neumann-Neumann fine-space memory budget.  The NN base materializes one
# dense (pseudo-)inverse PER TILE, of exactly the tile Schur block's own
# shape -- Sum(n_p_i^2) * itemsize, the SAME order as the
# tile_schur_complements dict the tilewise matvec already retains (18.7 GB
# fp64 at mi200k_v2) -- NOT the block-Jacobi budget's disjoint owned slices
# (~n^2/T).  Like the BJ guard, this is a MEMORY guard, not a numerics
# guard (the §7.13 lesson); the NN base has no known numeric collapse mode
# to guard (it keeps the neighbor-tile coupling BJ discards).
NEUMANN_MAX_FACTOR_BYTES = 64 * 1024 ** 3
_AUTO_NEUMANN_MAX_BYTES_CAP = NEUMANN_MAX_FACTOR_BYTES
_AUTO_NEUMANN_MAX_BYTES_FRACTION = 0.25
# Eigenvalue clip (relative to the block's own lambda_max) for the eigh
# pseudo-inverse fallback on singular tile Schur blocks (floating tiles /
# weakly-grounded port subsets) -- same default as the block-Jacobi
# fallback's _spd_safe_pseudo_solve_factor_ex.  The clipped tile-kernel
# (near-constant) directions are exactly what the PoU coarse space
# balances/deflates, per classical BDD.
NEUMANN_EIGCLIP_EPS_REL = 1e-10
# A numerically-singular PSD block can PASS cho_factor (tiny positive
# pivots) and yield a finite but ~1/pivot^2-amplifying inverse -- the same
# amplification pathology as the §7.8 BJ collapse, just along tile-kernel
# directions.  Route blocks whose Cholesky pivot ratio implies
# cond >~ 1e12 to the eigclip pseudo-inverse instead (which caps the
# response at 1/(eps_rel*lambda_max)).
NEUMANN_CHO_RCOND_MIN = 1e-12
# Coefficient (stiffness) weighting is the default: Mandel-Brezina's BDD
# subdomain-count-independence result requires coefficient-weighted D_i,
# not plain multiplicity counting (docs/interface_precond_sota_research.md
# §1.2 [4]).
DEFAULT_NEUMANN_WEIGHT = 'stiffness'
# Relative Tikhonov regularization for the NN local solves:
# S~_i = S_i + reg * diag(diag(S_i)).  MEASURED NEED (first mi200k_v2 NN
# run, 2026-08-01): every tile Schur block at the split regime is
# numerically singular (cond >~ 1e12 -- the §7.8 weakly-grounded-port
# cluster), so the raw eigclip pseudo-inverse amplifies ~1e10 along a
# broad near-null cluster the 65-column PoU deflation cannot cover, and
# cold DC stagnates (rel-res 1.6e-5 @ 2000 iters).  A reg > 0 bounds the
# amplification at ~1/reg AND lets the block factor via fast Cholesky
# (the all-eigh build measured 574 s vs ~2 min budget).  0.0 = off.
DEFAULT_NEUMANN_REG = 0.0


def resolve_neumann_max_bytes(setting: Any = 'auto') -> int:
    """Resolve the ``interface_neumann_max_bytes`` setting to bytes.

    'auto' (default) = min(64 GB, 0.25 * total system RAM), via psutil --
    deliberately much larger than the block-Jacobi budget because the NN
    inverses cost the same order as the already-retained tile Schur blocks
    (a second ~Sum(n_p_i^2)*itemsize footprint), not disjoint owned
    slices.  Consumed by ``InterfaceCGSolver._build_neumann`` to decide
    whether to degrade to the 'jacobi' diagonal base.
    """
    return _resolve_memory_budget_bytes(
        setting,
        _AUTO_NEUMANN_MAX_BYTES_CAP,
        _AUTO_NEUMANN_MAX_BYTES_FRACTION,
    )


# ---------------------------------------------------------------------------
# Stage 2 helpers: thread-count / dtype / matvec-mode resolution, LPT
# partitioning, kept-position (D1) slicing, SPD-safe pseudo-inverse.
# ---------------------------------------------------------------------------


def resolve_matvec_threads(setting: Any, n_tiles: int) -> int:
    """Resolve the ``matvec_threads`` setting to a concrete thread count.

    'auto' (default) = ``min(DEFAULT_MATVEC_THREADS_CAP, cpu_count, n_tiles)``
    -- capped at 8 (not 32), per the Stage 0 measurement documented in the
    module docstring.  An explicit integer is honoured as-is (not capped) --
    callers who override the default get exactly what they asked for.

    Args:
        setting: ``None``/``'auto'``, or an explicit positive integer
            (also accepts numeric strings, matching the other Stage 1
            memory-budget coercion helpers).
        n_tiles: Number of tiles contributing to the tilewise matvec.  Used
            only for the 'auto' cap (no point spinning up more threads than
            tiles to partition across).

    Returns:
        A positive integer thread count (never less than 1).
    """
    if setting is None or (
        isinstance(setting, str) and setting.strip().lower() == 'auto'
    ):
        cpu = os.cpu_count() or 1
        return max(1, min(DEFAULT_MATVEC_THREADS_CAP, cpu, max(1, n_tiles)))
    if isinstance(setting, bool):
        raise ValueError(
            f"Invalid matvec_threads setting {setting!r}: expected 'auto' "
            f"or a positive integer (bool is not valid)."
        )
    try:
        # Finding 8: same OverflowError-on-.inf gap as
        # _resolve_memory_budget_bytes (e.g. 'matvec_threads: .inf').
        val = int(float(setting))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"Invalid matvec_threads setting {setting!r}: expected 'auto' "
            f"or a positive integer."
        ) from exc
    if val < 1:
        raise ValueError(
            f"Invalid matvec_threads setting {setting!r}: must be >= 1."
        )
    return val


def resolve_matvec_dtype(setting: Any) -> np.dtype:
    """Resolve the ``interface_matvec_dtype`` setting to a numpy dtype.

    Accepts ``'float64'``/``'float32'`` (strings), or the numpy dtype /
    type objects directly.  Default (``None``) is float64.
    """
    if setting is None:
        return np.dtype(np.float64)
    if isinstance(setting, str):
        key = setting.strip().lower()
        if key in ('float64', 'fp64', 'f64', 'double'):
            return np.dtype(np.float64)
        if key in ('float32', 'fp32', 'f32', 'single'):
            return np.dtype(np.float32)
        raise ValueError(
            f"Invalid interface_matvec_dtype {setting!r}: expected "
            f"'float64' or 'float32'."
        )
    dt = np.dtype(setting)
    if dt not in (np.dtype(np.float64), np.dtype(np.float32)):
        raise ValueError(
            f"Invalid interface_matvec_dtype {setting!r}: expected "
            f"float64 or float32."
        )
    return dt


def resolve_matvec_mode(setting: Any, has_tile_blocks: bool) -> str:
    """Resolve ``interface_matvec_mode`` -- 'auto' | 'assembled' | 'tilewise'.

    Item 8: default changed from a hardcoded 'assembled' to 'auto', which
    picks 'tilewise' whenever per-tile dense Schur blocks are available
    (the common, memory-cheaper, now-threaded-and-D1-safe path) and falls
    back to 'assembled' only when they are not (e.g. streaming assembly
    without ``interface_drop_s_global``).  Explicit values are honoured
    verbatim -- 'auto' never overrides an explicit caller choice.
    """
    if setting is None or (
        isinstance(setting, str) and setting.strip().lower() == 'auto'
    ):
        return 'tilewise' if has_tile_blocks else 'assembled'
    if setting not in ('assembled', 'tilewise'):
        raise ValueError(
            f"Invalid interface_matvec_mode {setting!r}: expected 'auto', "
            f"'assembled', or 'tilewise'."
        )
    return setting


def resolve_preconditioner(
    setting: Any, interface_solver_resolved: str, matvec_mode_resolved: str,
) -> str:
    """Resolve the ``interface_preconditioner`` setting -- 'auto' | explicit.

    Stage 3 resolved default: 'auto' (or ``None``) resolves to 'two_level'
    when the resolved interface solver is 'cg' AND the resolved matvec mode
    is 'tilewise' (the regime Stage 2 measured block-Jacobi CG stagnating
    in); otherwise ('assembled' CG, or -- moot, since this is only called
    from the CG branch -- 'direct') it resolves to the legacy 'block_jacobi'
    default. An explicit (non-'auto') value is always honoured verbatim, so
    small systems that already resolve to 'direct' (never reaching this
    function) and any caller/YAML/CLI setting that names a preconditioner
    explicitly are completely unaffected -- this is the ONLY place 'auto' is
    resolved (see ``build_interface_solver``'s docstring).
    """
    # Finding 12: normalize (strip + lower) EVERY string value before
    # validation, not just the 'auto' sentinel -- the pre-fix code left an
    # inconsistent-coercion trap where 'auto'/' AUTO ' was silently accepted
    # but an equivalently-sloppy explicit value (e.g. 'Two_Level ', a
    # trailing space from a quoted YAML scalar) raised ValueError deep
    # inside prepare()/factor() at solve time.
    _norm = setting.strip().lower() if isinstance(setting, str) else setting
    if _norm is None or _norm == 'auto':
        if interface_solver_resolved == 'cg' and matvec_mode_resolved == 'tilewise':
            return 'two_level'
        return 'block_jacobi'
    if _norm not in (
        'block_jacobi', 'jacobi', 'none', 'amg', 'two_level', 'neumann',
    ):
        raise ValueError(
            f"Invalid interface_preconditioner {setting!r}: expected 'auto', "
            f"'block_jacobi', 'jacobi', 'none', 'amg', 'two_level', or "
            f"'neumann'."
        )
    return _norm


def _lpt_partition(costs: List[float], n_bins: int) -> List[List[int]]:
    """Longest-processing-time greedy load-balancing partition.

    Returns a list of ``n_bins`` lists, each containing the indices (into
    ``costs``) assigned to that bin, chosen greedily by always adding the
    next-most-expensive remaining item to the currently-least-loaded bin.
    Same algorithm as the Stage 0 prototype
    (``scripts/benchmark/microbench/bench_interface_matvec.py``).
    """
    n_bins = max(1, n_bins)
    if not costs:
        return [[] for _ in range(n_bins)]
    order = np.argsort(costs)[::-1]
    bins: List[List[int]] = [[] for _ in range(n_bins)]
    loads = np.zeros(n_bins)
    for i in order:
        b = int(np.argmin(loads))
        bins[b].append(int(i))
        loads[b] += costs[i]
    return bins


def kept_position_slice(
    S_i: np.ndarray,
    port_nodes: List[str],
    interface_node_to_idx: Dict[str, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """D1 fix: drop Dirichlet/pad rows+columns from a tile's dense Schur block.

    ``port_nodes`` is the tile's FULL port list (as returned by
    ``factor_and_compute_schur`` / ``bs.port_nodes``), which may include
    Dirichlet (pad) nodes -- those are not part of the interface unknown
    set (``interface_node_to_idx``), so ``S_i`` (shape
    ``(len(port_nodes), len(port_nodes))``) is one row/column too big per
    pad port for the tilewise-CG matvec, which indexes with the
    pad-filtered ``tile_index_maps[tid]``.

    Mathematically exact (not an approximation): slicing away a Dirichlet
    row/column from a SUMMED matrix (``S_global = sum_i S_i_embedded``) is
    identical to summing the sliced blocks, because slicing is a linear
    projection that commutes with addition.  The dropped pad's contribution
    to the unknown-unknown block already lives entirely in the assembly's
    ``rhs_dirichlet`` (via ``-G_ud @ V_d``), not in ``S_i``'s own
    unknown-unknown entries -- see interface_solve_acceleration_plan.md D1.

    Args:
        S_i: Dense ``(n_ports, n_ports)`` Schur block (full port order,
            same order as ``port_nodes``).
        port_nodes: Tile's full port name list, same order as ``S_i``'s
            rows/columns.
        interface_node_to_idx: Global interface (unknown-only) ordering.

    Returns:
        ``(idx, S_kept, kept_pos)`` -- ``idx`` (int32, global interface
        indices, kept order), ``S_kept`` (float64, contiguous, ``(k, k)``
        where ``k <= n_ports``), and ``kept_pos`` (int64, POSITIONS within
        ``port_nodes``/the original full-port-order arrays such as
        ``get_reduced_rhs()``'s return value -- callers filtering a
        full-port-length vector, e.g. ``solve_dc``'s RHS scatter, index with
        this).  ``k`` may be 0 (a tile whose entire port list is Dirichlet
        pads contributes nothing to the tilewise matvec).
    """
    n_ports = len(port_nodes)
    if n_ports == 0:
        empty = np.empty(0, dtype=np.int64)
        return np.empty(0, dtype=np.int32), np.zeros((0, 0), dtype=np.float64), empty

    kept_pos_list = [p for p, nd in enumerate(port_nodes) if nd in interface_node_to_idx]
    kept_pos = np.array(kept_pos_list, dtype=np.int64)
    idx = np.array(
        [interface_node_to_idx[port_nodes[p]] for p in kept_pos_list], dtype=np.int32,
    )
    S_arr = np.asarray(S_i, dtype=np.float64)
    if len(kept_pos_list) == n_ports:
        S_kept = np.ascontiguousarray(S_arr)
    else:
        S_kept = np.ascontiguousarray(S_arr[np.ix_(kept_pos, kept_pos)])
    return idx, S_kept, kept_pos


def filter_kept_rhs(
    g_i: np.ndarray,
    tid: Any,
    idx_map: np.ndarray,
    kept_pos_map: Dict[Any, np.ndarray],
    port_count_map: Optional[Dict[Any, int]] = None,
    *,
    caller: str = "",
) -> np.ndarray:
    """D1/S2/S13-safe filter: reduce a full-port-order per-tile vector to kept positions.

    Shared by every RHS/terminal-vector scatter site that pairs a tile's
    reduced vector (``get_reduced_rhs``/``evaluate_and_get_reduced_rhs``/
    the adjoint terminal-RHS, all returned in the tile's FULL port order,
    possibly including a Dirichlet/pad port directly on the tile's own port
    list) with the pad-FILTERED ``idx_map`` (``tile_index_maps[tid]``) --
    see ``kept_position_slice`` and interface_solve_acceleration_plan.md D1.
    ``solve_dc`` originally had this filter inline; Stage 2's D1 follow-up
    (S2) found the sibling QS/transient/adjoint scatter sites lacked it
    entirely, and Stage 2's own inline version (S13) under-validated: it
    accepted any ``kept_pos`` whose length matched ``idx_map`` without
    checking that ``kept_pos`` was actually built for THIS ``g_i`` -- a
    stale/drifted context (e.g. workers re-set-up after a retile, or a
    mixed pre/post-Stage-2 session) with a coincidentally same-length,
    wrong-valued ``kept_pos`` would silently scatter ``g_i`` to the wrong
    interface rows instead of raising.

    Args:
        g_i: Full-port-order per-tile vector as returned by the worker RPC.
        tid: Tile id (dict key into ``idx_map``'s owning ``tile_index_maps``,
            ``kept_pos_map``, and ``port_count_map``).
        idx_map: This tile's (pad-filtered) global interface index array
            (``ctx.tile_index_maps[tid]``).
        kept_pos_map: ``ctx.tile_kept_port_pos`` -- per-tile positions within
            the FULL port order that survived pad-filtering.
        port_count_map: ``ctx.tile_port_count`` -- per-tile FULL port count
            recorded at the same factor()-time call that built
            ``kept_pos_map[tid]``.  When provided (non-empty) and the tile
            has an entry, ``len(g_i)`` is validated against it before
            slicing.  Pass ``None`` (or an empty/missing-tid dict) only for
            legacy pre-Stage-2 contexts that never recorded it -- the
            length-only fast path and the ``kept_pos``-length check still
            catch the common cases, just without the extra drift guard.
        caller: Short label (e.g. ``"solve_quasi_static"``) included in the
            error message for easier diagnosis.

    Returns:
        ``g_i`` unchanged (fast path, no tile-resident pad port) or
        ``g_i[kept_pos]`` (validated slice down to ``len(idx_map)``).

    Raises:
        ValueError: On any length mismatch that ``kept_pos``/``port_count``
            cannot resolve -- a genuinely inconsistent context. No silent
            'repair' is attempted beyond the documented kept-position slice.
    """
    if len(g_i) == len(idx_map):
        # Fast path: tile has no Dirichlet/pad port on its own port list
        # (the overwhelmingly common case) -- already matches. T4: still
        # validate against port_count_map when available -- a drifted tile
        # (e.g. re-set-up after a retile) whose CURRENT full port count no
        # longer matches what was recorded at factor() time can produce a
        # g_i whose length only COINCIDENTALLY equals len(idx_map); this is
        # exactly the "lengths happen to coincide" corruption the S13
        # slow-path check below exists to catch, except the fast path used
        # to return before ever consulting port_count_map. A consistent
        # (non-drifted) full-count tile always satisfies
        # len(g_i) == port_count == len(idx_map); if port_count_map has an
        # entry for this tile and it differs from len(g_i), that invariant
        # is broken regardless of the len(g_i) == len(idx_map) coincidence.
        _expected_full_fast = port_count_map.get(tid) if port_count_map else None
        if _expected_full_fast is not None and _expected_full_fast != len(g_i):
            raise ValueError(
                f"{caller}: tile {tid!r} reduced-vector length {len(g_i)} "
                f"coincidentally equals index map length {len(idx_map)} "
                f"(fast path), but the FULL port count recorded when "
                f"tile_kept_port_pos[{tid!r}] was built at factor() time is "
                f"{_expected_full_fast} != {len(g_i)}. Slicing this g_i "
                f"unfiltered would silently scatter to the wrong interface "
                f"rows -- this indicates a drifted/stale context (e.g. "
                f"workers re-set-up after a retile). Re-run factor() (not "
                f"just refactor()) to rebuild a consistent context."
            )
        return g_i

    kept_pos = kept_pos_map.get(tid) if kept_pos_map else None
    if kept_pos is None or len(kept_pos) != len(idx_map):
        raise ValueError(
            f"{caller}: tile {tid!r} reduced-vector length {len(g_i)} != "
            f"index map length {len(idx_map)}, and no usable "
            f"tile_kept_port_pos entry is available to recover (D1 "
            f"pad-on-port fix requires a Stage-2 factor() context; a "
            f"pre-Stage-2 checkpoint or backward-compat direct "
            f"construction cannot recover here)."
        )

    expected_full = port_count_map.get(tid) if port_count_map else None
    if expected_full is not None and len(g_i) != expected_full:
        raise ValueError(
            f"{caller}: tile {tid!r} reduced-vector length {len(g_i)} != "
            f"the FULL port count {expected_full} recorded when "
            f"tile_kept_port_pos[{tid!r}] was built at factor() time. "
            f"kept_pos happens to have the right length ({len(kept_pos)}) "
            f"to match idx_map, but slicing with it would silently "
            f"scatter to the wrong interface rows -- this indicates a "
            f"drifted/stale context (e.g. workers re-set-up after a "
            f"retile). Re-run factor() (not just refactor()) to rebuild "
            f"a consistent context."
        )
    if kept_pos.size and int(kept_pos.max()) >= len(g_i):
        raise ValueError(
            f"{caller}: tile {tid!r} kept_pos max index {int(kept_pos.max())} "
            f"is out of bounds for reduced-vector length {len(g_i)} -- "
            f"inconsistent context (see tile_kept_port_pos docs)."
        )
    return g_i[kept_pos]


def _spd_safe_pseudo_solve_factor(
    sub: np.ndarray, eps_rel: float = 1e-10,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Eigh-based PSD-safe pseudo-inverse factor for an indefinite BJ block.

    Replaces the old ``np.linalg.pinv`` fallback (item 7): ``pinv`` of a
    numerically indefinite symmetric matrix can retain tiny NEGATIVE
    eigenvalues from floating-point noise, so the resulting "preconditioner"
    is not actually PSD -- silently voiding CG's convergence guarantee
    (a valid preconditioner for SPD CG must itself be SPD).

    Thin wrapper over :func:`_spd_safe_pseudo_solve_factor_ex` that drops
    its third return value (the raw, unclipped spectrum ``w``).  Kept as a
    separate name -- rather than inlined at call sites -- because it is
    directly unit-tested (``TestSPDSafeFallback``) against the 2-tuple
    ``(V, inv_w)`` contract; production (``_build_block_jacobi``) calls the
    ``_ex`` form directly so it can reuse ``w`` for GenEO-lite enrichment
    without a second eigendecomposition of the same block. Because this is
    a thin wrapper, it is NOT on `_build_block_jacobi`'s call path -- tests
    that need to intercept the eigh-fallback failure mode in that path must
    monkeypatch :func:`_spd_safe_pseudo_solve_factor_ex`, not this function.

    Returns ``None`` only if the block has no positive spectrum at all
    (``lambda_max <= 0``) -- callers should skip the block (identity
    fallback) in that degenerate case.
    """
    result = _spd_safe_pseudo_solve_factor_ex(sub, eps_rel=eps_rel)
    if result is None:
        return None
    V, inv_w, _w = result
    return V, inv_w


def _spd_safe_pseudo_solve_factor_ex(
    sub: np.ndarray, eps_rel: float = 1e-10,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Same as :func:`_spd_safe_pseudo_solve_factor` but also returns the
    raw (unclipped) eigenvalues ``w``.

    A separate function -- not a change to the original's return contract --
    because :func:`_spd_safe_pseudo_solve_factor` is directly unit-tested
    (``TestSPDSafeFallback``) against a 2-tuple ``(V, inv_w)`` return, and
    :func:`_spd_safe_pseudo_solve_factor` now delegates HERE (dropping
    ``w``) so there is exactly one implementation.  Stage 3's GenEO-lite
    enrichment needs the block's full spectrum to pick its lowest eigenpairs
    WITHOUT a second eigendecomposition of the same block (the "don't
    recompute" requirement -- a block that already fell into this
    indefinite-block fallback path has already paid for a full ``eigh``);
    ``_build_block_jacobi`` calls THIS ``_ex`` form directly (always, not
    only when GenEO reuse is needed) so there is exactly one eigh call per
    indefinite block either way.
    """
    sub_sym = 0.5 * (sub + sub.T)
    w, V = np.linalg.eigh(sub_sym)
    lam_max = float(np.max(w)) if w.size else 0.0
    if lam_max <= 0:
        return None
    eps = eps_rel * lam_max
    w_clipped = np.clip(w, eps, None)
    inv_w = 1.0 / w_clipped
    return V, inv_w, w


def _finalize_pool(pool: ThreadPoolExecutor) -> None:
    """Module-level finalizer callback (must not close over ``self``).

    Used as the ``weakref.finalize`` target so a garbage-collected
    ``InterfaceCGSolver`` that never called :meth:`InterfaceCGSolver.close`
    explicitly still releases its thread pool -- a safety net, not the
    primary release mechanism (``release()`` on the owning context calls
    ``close()`` directly; see the lifecycle-parity checklist in the plan).
    """
    pool.shutdown(wait=False)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class InterfaceCGSolver:
    """Iterative CG solver for the global Schur complement interface system.

    Wraps scipy.sparse.linalg.cg with a LinearOperator.  The returned
    callable has the same signature as the direct LU callable:
        v_gamma = solver(rhs)

    Parameters
    ----------
    n_interface : int
        Size of the interface system (number of unknowns).
    matvec_mode : str
        'assembled' or 'tilewise'.
    S_global : sp.csc_matrix, optional
        Assembled sparse Schur matrix.  Required for mode='assembled'.
        Also used for 'jacobi' preconditioner in tilewise mode.
    tile_schur_complements : dict, optional
        {tile_id: S_i (dense np.ndarray)}.  Required for mode='tilewise'.
    tile_index_maps : dict, optional
        {tile_id: np.ndarray(int32)} mapping local port index -> global
        interface index.  Required for mode='tilewise'.
    preconditioner : str
        'block_jacobi' (default), 'jacobi', 'none', 'amg'.
    rtol : float
        Relative tolerance for CG convergence (default 1e-8, validated by the
        Stage 0 sweep: 166 nV max error vs direct on the BRCM-class proxy —
        see docs §7.7).
    maxiter : int or None
        Maximum CG iterations.  None = 3 * n_interface.
    x0 : np.ndarray or None
        Initial guess for the next solve.  Warm-start support.
    stats_dict : dict or None
        If provided, iteration counts and timing are written here.
        Updated in-place on each solve call.
    block_jacobi_max_bytes : int or None
        Per-instance override for the block-Jacobi factor-memory budget
        (bytes).  ``None`` (default) falls back to the module-level
        ``BLOCK_JACOBI_MAX_FACTOR_BYTES`` constant (read dynamically, so
        test monkeypatches of that module attribute still work).  Callers
        that have a ``model.settings['interface_block_jacobi_max_bytes']``
        value should resolve it via ``resolve_block_jacobi_max_bytes()``
        and pass the result here.
    """

    def __init__(
        self,
        n_interface: int,
        matvec_mode: str = 'assembled',
        S_global: Optional[sp.spmatrix] = None,
        tile_schur_complements: Optional[Dict[Any, np.ndarray]] = None,
        tile_index_maps: Optional[Dict[Any, np.ndarray]] = None,
        S_extra: Optional[sp.spmatrix] = None,
        preconditioner: str = 'block_jacobi',
        rtol: float = 1e-8,
        atol: float = 1e-14,
        maxiter: Optional[int] = None,
        x0: Optional[np.ndarray] = None,
        stats_dict: Optional[Dict[str, Any]] = None,
        strict: bool = True,
        block_jacobi_max_bytes: Optional[int] = None,
        # NN/BDD work package: which base component 'two_level' builds --
        # None/'auto' (legacy block_jacobi with its byte-budget jacobi
        # downgrade), 'block_jacobi', 'jacobi' (skip the BJ estimate
        # entirely), or 'neumann' (weighted Neumann-Neumann fine space,
        # _build_neumann).  Ignored unless preconditioner=='two_level'.
        two_level_base: Optional[str] = None,
        # None-sentinel like block_jacobi_max_bytes: None = read the
        # module-level NEUMANN_MAX_FACTOR_BYTES dynamically at build time
        # (keeps monkeypatch-based test behaviour).
        neumann_max_bytes: Optional[int] = None,
        # 'stiffness' (default, Mandel-Brezina coefficient weights) or
        # 'multiplicity' (plain 1/count PoU) -- resolved from
        # DEFAULT_NEUMANN_WEIGHT dynamically when None.
        neumann_weight: Optional[str] = None,
        # Relative Tikhonov shift for the NN local solves (see
        # DEFAULT_NEUMANN_REG's comment for the measured rationale) --
        # resolved dynamically when None.
        neumann_reg: Optional[float] = None,
        matvec_threads: Any = 'auto',
        matvec_dtype: Any = 'float64',
        strict_dtype_rtol: bool = True,
        island_idx: Optional[np.ndarray] = None,
        # Finding 9: None-sentinel defaults, NOT interface_coarse.DEFAULT_*
        # bound at def time (module-import time) -- resolved dynamically in
        # the constructor body instead, so
        # monkeypatch.setattr(interface_coarse, 'DEFAULT_GENEO_K', ...)
        # (etc.) actually takes effect for callers that don't pass these
        # explicitly. See the constructor body's Finding 9 comment.
        interface_coarse_geneo_k: Optional[int] = None,
        interface_coarse_geneo_tol: Optional[float] = None,
        interface_coarse_eps_rank: Optional[float] = None,
        interface_coarse_max_cols: Optional[int] = None,
        interface_coarse_max_bytes: Optional[int] = None,
        # A-DEF2 work package: same None-sentinel dynamic-default pattern as
        # the Stage 3 coarse knobs above (apply_mode/reproject_every are
        # resolved from interface_coarse.DEFAULT_APPLY_MODE/DEFAULT_DEFLATED_
        # REPROJECT_EVERY dynamically in the constructor body, NOT bound at
        # def time). Round-2 code review finding 5 (regression -- corrects a
        # prior revision of this comment, which claimed a plain bool
        # def-time default was fine here because "its canonical default
        # lives in THIS module": that reasoning is exactly backwards --
        # DEFAULT_WARM_START_EXTRAPOLATION living in this module is why it
        # MUST be resolved dynamically too, not why a def-time snapshot is
        # safe. `warm_start_extrapolation: bool = DEFAULT_WARM_START_
        # EXTRAPOLATION` binds the constant's value ONCE, at module-import
        # time -- monkeypatch.setattr(interface_iterative,
        # 'DEFAULT_WARM_START_EXTRAPOLATION', ...) (the exact pattern
        # cli.py's _iface_default docstring promises works, and
        # TestFinding9DynamicCoarseDefaults exercises for the sibling coarse
        # knobs) would silently have NO effect on a caller that omits this
        # kwarg -- a def-time-bound copy, same class of bug Finding 9
        # (round 1) fixed for the coarse knobs above. Same None-sentinel
        # fix here: resolved dynamically in the constructor body, see the
        # Finding-5 comment there.
        interface_coarse_apply_mode: Optional[str] = None,
        interface_deflated_reproject_every: Optional[int] = None,
        warm_start_extrapolation: Optional[bool] = None,
    ) -> None:
        """
        Parameters
        ----------
        island_idx : np.ndarray, optional
            Stage 3: global interface indices of interface-island nodes
            (penalized via ``S_extra``'s 1e5 mS diagonal).  Only consumed by
            ``preconditioner='two_level'`` -- zeroed out of every coarse-
            space column (PoU and GenEO alike) so the penalty does not leak
            into ``S_c``.  See ``interface_coarse.py``'s module docstring.
        interface_coarse_geneo_k : int
            Stage 3: max GenEO-lite eigenpairs enriched per block-Jacobi
            ownership block (default 0 as of the 2026-07-20 measurement-
            driven flip -- GenEO measured ZERO iteration benefit in every
            cold/warm x additive/deflated cell on mi200k_v2 while costing
            ~70 s/prepare, see ``interface_deflation_notes.md``; the
            machinery stays fully functional and opt-in via geneo_k > 0).
            0 disables GenEO, leaving a PoU-only coarse space.  Only used by
            ``preconditioner='two_level'``.
        interface_coarse_geneo_tol : float
            Stage 3: relative eigenvalue threshold (fraction of the block's
            own ``lambda_max``) below which an eigenpair is enriched
            (default 1e-6).
        interface_coarse_eps_rank : float
            Stage 3: ``S_c`` eigenvalues <= this fraction of ``S_c``'s
            ``lambda_max`` are treated as structural rank deficiency (e.g.
            the checkerboard null space of an even-multiplicity PoU basis)
            and dropped from the pseudo-inverse (default 1e-12) -- distinct
            knob from ``interface_coarse_geneo_tol``.
        interface_coarse_max_cols : int
            Stage 3: hard cap on T' (default 4096); exceeding it first
            falls back to a PoU-only coarse space (WARNING, GenEO columns
            dropped) -- the coarse space is disabled entirely (degrades to
            plain block-Jacobi) only if the PoU-only column count ALONE
            still exceeds the cap.
        interface_coarse_max_bytes : int
            Stage 3 (Finding 5): byte-based guard on the two dense
            ``(n, T')`` fp64 arrays the coarse build allocates (default
            'auto' -> ``resolve_coarse_max_bytes``, min(8 GB, 0.1x total
            RAM)); same two-rung PoU-only-then-disable degradation as
            ``interface_coarse_max_cols`` above, since a large-n/modest-T'
            system can exceed it even when comfortably under the column cap.
        interface_coarse_apply_mode : 'additive' or 'deflated'
            A-DEF2 work package: how the ``two_level`` coarse-space
            correction is applied inside CG (default 'deflated' as of the
            2026-07-20 measurement-driven flip -- 'deflated' beat 'additive'
            in EVERY cell of the mi200k_v2 head-to-head matrix, see
            ``interface_deflation_notes.md``'s "Defaults flipped by
            measurement" section; see this module's "A-DEF2 work package"
            docstring section for the full derivation and the ratification
            record of why the non-additive value is named 'deflated', not
            'adef2'). 'additive' is the Stage 3 ``M^-1 = M_base^-1 + Z S_c^+
            Z^T`` form (unchanged, still fully supported, just no longer the
            default). 'deflated' deflects ``range(Z)`` out of the iteration exactly
            (``M^-1_DEF r = M_base^-1(r - S Q r) + Q r``), routed through the
            hand-rolled :func:`_deflated_pcg` loop instead of
            ``scipy.sparse.linalg.cg`` -- only takes effect when
            ``preconditioner`` resolves to ``'two_level'`` AND the coarse
            build actually retains ``SZ`` (see ``interface_coarse_max_bytes``
            -- an over-budget coarse space degrades via the GenEO-then-
            disable ladder FIRST; the dedicated SZ-only "keep the coarse
            space, drop SZ, fall back to additive" degrade only fires in the
            (defensive, not normally reachable) case where the ladder's own
            pass already fits but a persistent SZ still would not -- see
            this module's "A-DEF2 work package" docstring's "SZ retention"
            bullet). Ignored (no-op) for every other ``preconditioner``
            value.
        interface_deflated_reproject_every : int
            A-DEF2 work package: re-project the deflated residual every this
            many CG iterations to control finite-precision drift in the
            deflation invariant ``Z^T r -> 0`` (default 50; ``<= 0``
            disables reprojection). Only consumed when
            ``interface_coarse_apply_mode == 'deflated'`` and the coarse
            build retained ``SZ``.
        warm_start_extrapolation : bool
            When True, seeds each solve's warm start with the linear
            extrapolation ``2*x_prev - x_prev2`` of the last two solutions
            (falls back to ``x_prev`` until two solves have been recorded)
            instead of the plain previous solution (default False). Composes
            with either apply mode; see :meth:`push_solution_history`.
        S_extra : sp.spmatrix, optional
            Additional sparse contribution to the matvec, added on top of the
            per-tile Schur sum.  For ``matvec_mode='tilewise'`` this carries the
            package-edge contribution (G_pkg_uu) that is NOT included in the
            per-tile Schur complements.  Ignored in assembled mode (where S_global
            already contains all contributions).
        atol : float
            Absolute tolerance for CG convergence.  The CG stopping criterion is
            ``||r|| <= max(rtol * ||b||, atol)``.  The default (1e-14) prevents
            CG from burning all maxiter iterations when the RHS is near-zero
            (e.g. early transient steps with no active sources), where
            ``rtol * ||b|| -> 0`` would otherwise make convergence unreachable.
            Set to 0 to recover the pure rtol behaviour.
        strict : bool
            If True (default), raise ``RuntimeError`` when CG does not converge
            within maxiter iterations.  The error message includes the relative
            residual so it appears clearly in logs.  Set to False to demote to a
            warning (not recommended for production; useful only for unit-testing
            the non-convergence path).
        matvec_threads : int or 'auto'
            Stage 2: number of threads for the tilewise matvec and the
            block-Jacobi apply (shared persistent ``ThreadPoolExecutor``,
            lazily built).  'auto' (default) = ``min(8, cpu_count, n_tiles)``
            -- see :func:`resolve_matvec_threads` and the module docstring
            for the Stage 0 measurement behind the cap of 8.  Only used in
            'tilewise' matvec mode (and for block-Jacobi regardless of
            matvec mode).
        matvec_dtype : 'float64' or 'float32'
            Stage 2: storage dtype for each tile's ``S_i`` block in tilewise
            mode ('float64' default).  'float32' halves per-tile storage
            and roughly doubles GEMV throughput on a CPU-only host (see
            module docstring) at the cost of a ~1e-7 relative residual
            floor -- paired with ``strict_dtype_rtol`` (below).  Ignored in
            'assembled' mode.
        strict_dtype_rtol : bool
            If True (default) and ``matvec_dtype='float32'``, raise
            ``ValueError`` when ``rtol < FP32_MATVEC_MIN_RTOL`` (1e-7) --
            fp32's own matvec error floor makes a tighter CG tolerance
            unachievable/meaningless.  With ``interface_coarse_apply_mode
            ='deflated'``, the floor is ``FP32_MATVEC_MIN_RTOL_DEFLATED``
            (1e-6, one decade looser) instead -- the deflated loop's
            acceptance gate (``_try_accept`` in :func:`_deflated_pcg`)
            recomputes a FRESH true residual through the same fp32 matvec
            on every attempt (not just the failure branch, as the scipy/
            additive path does), so at the plain 1e-7 floor that gate can
            sit persistently at/above the fp32 noise floor and never
            accept, even though CG's own tracked residual has genuinely
            converged (round-3 code review finding 1).  Set False to
            override (e.g. for a deliberate accuracy study).
        """
        if matvec_mode not in ('assembled', 'tilewise'):
            raise ValueError(
                f"matvec_mode must be 'assembled' or 'tilewise', got {matvec_mode!r}"
            )
        if matvec_mode == 'assembled' and S_global is None:
            raise ValueError("matvec_mode='assembled' requires S_global")
        if matvec_mode == 'tilewise' and (
            tile_schur_complements is None or tile_index_maps is None
        ):
            raise ValueError(
                "matvec_mode='tilewise' requires tile_schur_complements "
                "and tile_index_maps"
            )
        if preconditioner not in (
            'block_jacobi', 'jacobi', 'none', 'amg', 'two_level', 'neumann',
        ):
            raise ValueError(
                f"preconditioner must be one of 'block_jacobi', 'jacobi', "
                f"'none', 'amg', 'two_level', 'neumann'; got {preconditioner!r}"
            )

        # D1 (item 1): fail loudly, HERE, if a caller passed a tile Schur
        # block whose shape doesn't match its (pad-filtered) index map --
        # this is exactly the corruption pattern the kept-position slicing
        # (result_factorization.py, kept_position_slice() above) fixes at
        # the source; catching it here turns a silent numpy broadcast
        # failure deep inside the CG matvec into an actionable error at
        # construction time.
        if matvec_mode == 'tilewise':
            for tid, S_i in tile_schur_complements.items():
                idx = tile_index_maps[tid]
                if S_i.shape != (len(idx), len(idx)):
                    raise ValueError(
                        f"InterfaceCGSolver: tile {tid!r} Schur block shape "
                        f"{S_i.shape} != (len(tile_index_maps[tid]),)*2 = "
                        f"({len(idx)}, {len(idx)}). This is the D1 pad-port "
                        f"corruption pattern (interface_solve_acceleration_"
                        f"plan.md) -- the caller must slice S_i to kept "
                        f"(non-Dirichlet) port positions via "
                        f"kept_position_slice() before constructing "
                        f"InterfaceCGSolver."
                    )

        self.n = n_interface
        self.matvec_mode = matvec_mode
        self.S_global = S_global
        self.tile_schur_complements = tile_schur_complements
        self.tile_index_maps = tile_index_maps
        self.S_extra = S_extra  # package-edge contribution (tilewise mode)
        self.preconditioner = preconditioner
        # The originally-requested preconditioner, kept even if a runtime
        # memory-budget downgrade (see _build_block_jacobi) replaces
        # self.preconditioner with a weaker one.  Callers/logs that need to
        # know what the USER asked for (as opposed to what is actually
        # active) should read this attribute (finding 11).
        self.requested_preconditioner = preconditioner
        self.rtol = rtol
        self.atol = atol
        self.strict = strict
        self.maxiter = maxiter if maxiter is not None else max(3 * n_interface, 100)
        self._x0: Optional[np.ndarray] = x0
        self._stats: Dict[str, Any] = stats_dict if stats_dict is not None else {}
        # Stage 1c: per-instance block-Jacobi memory budget override.  None
        # means "use the module-level BLOCK_JACOBI_MAX_FACTOR_BYTES constant
        # dynamically" (preserves monkeypatch-based test behaviour).
        self.block_jacobi_max_bytes: Optional[int] = block_jacobi_max_bytes

        # --- Stage 3: two-level coarse-space state --------------------------
        # ``_two_level_requested`` is captured BEFORE _build_block_jacobi()
        # runs because that builder may itself downgrade self.preconditioner
        # 'block_jacobi' -> 'jacobi' (memory budget) -- the coarse-space
        # augmentation step (after _build_linear_op(), below) must still
        # fire in that case (diagonal + coarse is a valid, useful
        # combination), so it cannot key off self.preconditioner alone.
        self._two_level_requested: bool = preconditioner == 'two_level'
        self._island_idx: Optional[np.ndarray] = (
            np.asarray(island_idx, dtype=np.int64)
            if island_idx is not None and len(island_idx) else None
        )
        # Finding 9: None-sentinel constructor defaults (see the signature
        # above) resolved HERE, via a live attribute lookup on
        # interface_coarse, rather than binding interface_coarse.DEFAULT_*
        # as def-time default parameter values -- def-time defaults are
        # evaluated exactly once, at module-import time, so
        # `monkeypatch.setattr(interface_coarse, 'DEFAULT_GENEO_K', ...)`
        # would have no effect on them.
        self._geneo_k: int = int(
            interface_coarse_geneo_k if interface_coarse_geneo_k is not None
            else interface_coarse.DEFAULT_GENEO_K
        )
        self._geneo_tol: float = float(
            interface_coarse_geneo_tol if interface_coarse_geneo_tol is not None
            else interface_coarse.DEFAULT_GENEO_TOL
        )
        self._eps_rank: float = float(
            interface_coarse_eps_rank if interface_coarse_eps_rank is not None
            else interface_coarse.DEFAULT_EPS_RANK
        )
        self._max_cols: int = int(
            interface_coarse_max_cols if interface_coarse_max_cols is not None
            else interface_coarse.DEFAULT_MAX_COLS
        )
        # Stage 1c-style per-instance override (Finding 5): None means "use
        # interface_coarse.DEFAULT_MAX_BYTES dynamically" (read at the point
        # of use in _augment_with_coarse_space, mirrors
        # block_jacobi_max_bytes above -- also a None-sentinel).
        self._coarse_max_bytes: Optional[int] = interface_coarse_max_bytes
        # Whether _build_block_jacobi() should pay for GenEO eigendecomposition
        # at all -- both a 'two_level' request AND a nonzero k are required
        # (k=0 is the documented "GenEO disabled, PoU-only" knob).
        self._want_geneo: bool = self._two_level_requested and self._geneo_k > 0
        # Per-block GenEO eigenpairs, ``[(global_idx, V_k, w_k), ...]`` --
        # populated by _build_block_jacobi() (only when
        # self._two_level_requested and self._geneo_k > 0), consumed by
        # _augment_with_coarse_space() after _build_linear_op() runs.
        self._geneo_pairs: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        self._coarse: Optional[interface_coarse.CoarseSpace] = None
        # Set True inside _build_block_jacobi() iff the 'two_level' base
        # component itself downgraded to the diagonal 'jacobi' fallback
        # (memory budget) -- used only for the human-readable
        # preconditioner_label below.
        self._bj_downgraded: bool = False
        # Finding 8: set inside _augment_with_coarse_space() to whether the
        # base (pre-coarse) preconditioner builder returned None outright
        # (e.g. all block-Jacobi blocks failed) -- distinct from
        # _bj_downgraded (block_jacobi -> jacobi); used so the degrade
        # path/label never mislabels a truly unpreconditioned base as
        # 'block_jacobi'.
        self._bj_was_none: bool = False
        # Finding 13: the base component's ACTUAL name ('none' | 'jacobi' |
        # 'block_jacobi'), derived ONCE from _bj_was_none/_bj_downgraded in
        # _augment_with_coarse_space (the only place both flags are known)
        # instead of being independently re-derived by the same if/elif
        # chain in both the degrade-path warning and preconditioner_label.
        # Placeholder default here; always overwritten before being read
        # (both consumers only run after _augment_with_coarse_space has).
        self._base_precond_label: str = 'block_jacobi'

        # --- NN/BDD work package: Neumann-Neumann base state ---------------
        _tlb_norm = (
            two_level_base.strip().lower()
            if isinstance(two_level_base, str) else two_level_base
        )
        if _tlb_norm is None or _tlb_norm == 'auto':
            _tlb_norm = 'block_jacobi'
        if _tlb_norm not in ('block_jacobi', 'jacobi', 'neumann'):
            raise ValueError(
                f"two_level_base must be 'auto', 'block_jacobi', 'jacobi', "
                f"or 'neumann'; got {two_level_base!r}."
            )
        self._two_level_base: str = _tlb_norm
        self.neumann_max_bytes: Optional[int] = neumann_max_bytes
        _nw_norm = (
            neumann_weight.strip().lower()
            if isinstance(neumann_weight, str) else neumann_weight
        )
        if _nw_norm is None:
            _nw_norm = DEFAULT_NEUMANN_WEIGHT
        if _nw_norm not in ('stiffness', 'multiplicity'):
            raise ValueError(
                f"neumann_weight must be 'stiffness' or 'multiplicity'; "
                f"got {neumann_weight!r}."
            )
        self._neumann_weight: str = _nw_norm
        self._neumann_reg: float = float(
            neumann_reg if neumann_reg is not None else DEFAULT_NEUMANN_REG
        )
        if self._neumann_reg < 0.0:
            raise ValueError(
                f"neumann_reg must be >= 0; got {neumann_reg!r}."
            )
        # Set by the base builder that actually ran ('neumann' on NN
        # success, 'jacobi' for an explicit two_level_base='jacobi'; None
        # otherwise) -- consulted by _augment_with_coarse_space's label
        # derivation ahead of the legacy _bj_downgraded/_bj_was_none flags.
        self._base_builder_label: Optional[str] = None

        # --- A-DEF2 work package: apply-mode / warm-start-extrapolation state ---
        _apply_mode_norm = (
            interface_coarse_apply_mode.strip().lower()
            if isinstance(interface_coarse_apply_mode, str)
            else interface_coarse_apply_mode
        )
        self._apply_mode: str = (
            _apply_mode_norm if _apply_mode_norm is not None
            else interface_coarse.DEFAULT_APPLY_MODE
        )
        # A-DEF2 work package -- RATIFIED (coordinator ruling; see this
        # module's "A-DEF2" docstring section for the full selection
        # record). The non-additive value is named 'deflated', NOT 'adef2':
        # the coordinator measured the true Tang/Nabben/Vuik/Erlangga A-DEF2
        # preconditioner head-to-head against the deflation ("DEF") member
        # of the same taxonomy this module actually ships, and DEF won (true
        # A-DEF2 regressed warm iterations on the netlist_multi_tile
        # 'natural' scenario and failed to converge outright on the
        # realistic-ratio ill-conditioned fixture -- see the docstring
        # section for the numbers). Shipping the setting as 'deflated'
        # rather than 'adef2' means the name no longer claims an algorithm
        # that isn't what runs -- no runtime self-disclosure machinery is
        # needed as a result (contrast the removed ADEF2_ACTUAL_ALGORITHM /
        # one-time-WARNING pattern an earlier revision of this module used
        # while the setting was still misleadingly named 'adef2').
        if self._apply_mode not in ('additive', 'deflated'):
            raise ValueError(
                f"interface_coarse_apply_mode must be 'additive' or "
                f"'deflated'; got {interface_coarse_apply_mode!r}."
            )
        self._deflated_reproject_every: int = int(
            interface_deflated_reproject_every
            if interface_deflated_reproject_every is not None
            else interface_coarse.DEFAULT_DEFLATED_REPROJECT_EVERY
        )
        # Base (pre-coarse-augmentation) preconditioner apply, used ONLY by
        # the 'deflated' hand-rolled loop (self._M holds the ADDITIVE
        # combination base_apply(r) + coarse.apply(r) once two_level is
        # active, which is the wrong operator for the deflated apply's own
        # M_base^-1(r - S Q r) + Q r formula) -- set inside
        # _augment_with_coarse_space, read here only for the type
        # annotation/default (never called before that runs when
        # _two_level_requested; __call__ only dispatches to the deflated
        # loop when self._coarse is not None, which implies
        # _augment_with_coarse_space already ran and set this).
        self._M_base_apply: Callable[[np.ndarray], np.ndarray] = (
            lambda r: r.copy()
        )
        # Finding 5 (round 2, mirrors Finding 9's round-1 pattern for the
        # coarse knobs): resolve the None sentinel HERE, dynamically, not
        # at def time -- see the parameter's Finding-5 comment above.
        self._warm_start_extrapolation: bool = bool(
            warm_start_extrapolation if warm_start_extrapolation is not None
            else DEFAULT_WARM_START_EXTRAPOLATION
        )
        # push_solution_history's two-point history (see its docstring).
        self._x_hist_prev: Optional[np.ndarray] = None
        self._x_hist_prev2: Optional[np.ndarray] = None

        # --- Stage 2: threaded matvec/BJ-apply state -----------------------
        self.matvec_dtype: np.dtype = resolve_matvec_dtype(matvec_dtype)
        # S9: matvec_dtype is documented as IGNORED in 'assembled' mode (pure
        # fp64 sparse matvec on S_global -- see _build_linear_op) -- only
        # 'tilewise' mode actually casts S_i to fp32 and hits the ~1e-7
        # residual floor this strict check exists to prevent an unachievable
        # rtol for.  Gate on matvec_mode so a shared solver.yaml with
        # interface_matvec_dtype='float32' (intended for tilewise runs)
        # doesn't abort an assembled-mode fallback run that would have
        # converged fine in fp64.
        if (
            self.matvec_mode == 'tilewise'
            and self.matvec_dtype == np.dtype(np.float32)
            and strict_dtype_rtol
        ):
            # Round-3 code review finding 1: apply_mode='deflated' gates
            # EVERY acceptance (not just a failure-branch diagnostic) on a
            # FRESH true residual computed through this same fp32 matvec
            # (see FP32_MATVEC_MIN_RTOL_DEFLATED's module-level comment
            # above for the full reasoning) -- needs one decade more
            # headroom above the matvec's own noise floor than the plain
            # additive/scipy path does.
            _fp32_min_rtol = (
                FP32_MATVEC_MIN_RTOL_DEFLATED if self._apply_mode == 'deflated'
                else FP32_MATVEC_MIN_RTOL
            )
            if rtol < _fp32_min_rtol:
                raise ValueError(
                    f"matvec_dtype='float32' requires rtol >= "
                    f"{_fp32_min_rtol:.0e} (fp32 tilewise matvec has a "
                    f"~1e-7 relative residual floor -- a tighter CG rtol is "
                    f"unachievable). Got rtol={rtol:.2e}. "
                    + (
                        "interface_coarse_apply_mode='deflated' additionally "
                        "gates EVERY acceptance on a fresh true residual "
                        "through this same fp32 matvec, so it needs rtol "
                        "comfortably above the plain fp32 floor "
                        f"({FP32_MATVEC_MIN_RTOL:.0e}), not merely at it. "
                        if self._apply_mode == 'deflated' else ""
                    )
                    + "Raise rtol, use matvec_dtype='float64', use "
                    f"apply_mode='additive', or pass "
                    f"strict_dtype_rtol=False to override."
                )
        # n_tiles for the 'auto' thread-count cap: prefer tile_index_maps
        # (present whenever block-Jacobi ownership is computed, in BOTH
        # matvec modes) over tile_schur_complements (tilewise-only) so an
        # assembled-mode solver with many tiles' worth of BJ ownership
        # blocks still gets a sensible 'auto' thread count for the
        # threaded BJ apply (item 5), not just the threaded matvec.
        if tile_index_maps:
            n_tiles = len(tile_index_maps)
        elif tile_schur_complements:
            n_tiles = len(tile_schur_complements)
        else:
            n_tiles = 0
        self.matvec_threads: int = resolve_matvec_threads(matvec_threads, n_tiles)
        self._pool: Optional[ThreadPoolExecutor] = None
        self._pool_lock = threading.Lock()

        # Build preconditioner BEFORE the linear op (order matters for fp32
        # tilewise storage, see below).  The block_jacobi/jacobi fallback
        # paths (S_global unavailable, tilewise-only) read
        # self.tile_schur_complements in fp64 -- they must run before
        # _build_linear_op()'s tilewise state prep converts those blocks to
        # fp32 and frees the fp64 originals in place.  Neither builder
        # depends on the other's *output* (only this ordering of reads),
        # so swapping them is safe.
        self._M: Optional[spla.LinearOperator] = self._build_preconditioner()
        # Build operator (tilewise mode: converts/frees tile_schur_complements
        # to matvec_dtype in place -- see _prepare_tilewise_matvec_state).
        self._linear_op: spla.LinearOperator = self._build_linear_op()

        # Stage 3: layer the coarse-space correction on top of the just-built
        # base (block_jacobi, or its jacobi downgrade) preconditioner -- must
        # run AFTER self._linear_op exists (S @ Z uses its matmat).  Degrades
        # to the plain base preconditioner (WARNING, never raises) if the
        # coarse build itself fails.
        if self._two_level_requested:
            self._M = self._augment_with_coarse_space()

        # Track cumulative iteration stats
        self._total_iters: int = 0
        self._total_solves: int = 0

    # ------------------------------------------------------------------
    # Thread pool lifecycle (Stage 2)
    # ------------------------------------------------------------------

    def _get_pool(self) -> ThreadPoolExecutor:
        """Lazily build the persistent matvec/BJ-apply thread pool.

        A ``weakref.finalize`` safety net is registered on first build so a
        garbage-collected solver that never called :meth:`close` explicitly
        still releases its threads; the primary release path is still
        ``close()`` (called by the owning context's ``release()``).
        """
        if self._pool is None:
            with self._pool_lock:
                if self._pool is None:
                    pool = ThreadPoolExecutor(
                        max_workers=self.matvec_threads,
                        thread_name_prefix='ifacecg-mv',
                    )
                    self._pool = pool
                    weakref.finalize(self, _finalize_pool, pool)
        return self._pool

    def close(self) -> None:
        """Shut down the persistent thread pool, if one was built.

        Idempotent.  Must be called by the owning context's ``release()``
        (lifecycle-parity checklist) so repeated prepare/release cycles do
        not accumulate live thread pools.
        """
        pool = self._pool
        self._pool = None
        if pool is not None:
            pool.shutdown(wait=True)

    # ------------------------------------------------------------------
    # LinearOperator / preconditioner construction
    # ------------------------------------------------------------------

    def _build_linear_op(self) -> spla.LinearOperator:
        """Build the matvec/matmat LinearOperator for S_global.

        'assembled' mode is unchanged (sparse matvec on S_global).
        'tilewise' mode (Stage 2) precomputes an LPT partition of tiles by
        ``n_ports**2`` across ``self.matvec_threads`` bins and, for each
        bin, the UNION of global interface indices its tiles touch (the
        compact-accumulator scatter design -- see module docstring for the
        measured decision).  ``matvec_threads <= 1`` uses a plain serial
        loop (no pool overhead).
        """
        n = self.n

        if self.matvec_mode == 'assembled':
            # Convert to CSR for efficient matvec
            S = self.S_global.tocsr()

            def _matvec(x: np.ndarray) -> np.ndarray:
                return S @ x

            def _matmat(X: np.ndarray) -> np.ndarray:
                return S @ X

            return spla.LinearOperator(
                shape=(n, n), matvec=_matvec, matmat=_matmat, dtype=np.float64,
            )

        # --- 'tilewise' mode -------------------------------------------------
        self._prepare_tilewise_matvec_state()
        return spla.LinearOperator(
            shape=(n, n),
            matvec=self._tilewise_matvec,
            matmat=self._tilewise_matmat,
            dtype=np.float64,
        )

    def _prepare_tilewise_matvec_state(self) -> None:
        """Precompute the tilewise-matvec working set (Stage 2).

        Populates:
          ``self._tiles``       -- list of (idx int64, S_i dtype-cast) pairs.
          ``self._mv_partition``-- LPT partition of tile indices into
                                   ``self.matvec_threads`` bins by cost n_p^2.
          ``self._mv_touched``  -- per-bin sorted-unique array of global
                                   interface indices the bin's tiles touch
                                   (the compact-accumulator scatter target).
          ``self._S_extra_csr`` -- CSR S_extra (or None).

        Finding 3 (fp32 storage): when ``dtype`` is float32, each tile's
        fp64 block is cast and the fp64 original is immediately freed by
        overwriting the ``self.tile_schur_complements[tid]`` entry in
        place, so peak coordinator memory during this loop is one tile's
        block (fp64 + fp32 transiently), not the whole dict held in both
        dtypes at once -- at BRCM scale, keeping both would be 12.5 GB +
        6.2 GB, strictly worse than fp64-only.  After this method returns,
        ``self.tile_schur_complements`` holds only fp32 arrays; the only
        readers of that dict are the block_jacobi/jacobi preconditioner
        builders, which are guaranteed to have already run (see the
        build-order comment in ``__init__``), and no caller keeps its own
        reference to the fp64 arrays past constructing this solver (the
        dict object itself is shared with the caller, but callers never
        read it again after passing it in -- see the callers in
        ``result_factorization.py``).
        """
        dtype = self.matvec_dtype
        free_fp64 = dtype != np.float64
        tiles: List[Tuple[np.ndarray, np.ndarray]] = []
        for tid, S_i in self.tile_schur_complements.items():
            idx = np.asarray(self.tile_index_maps[tid], dtype=np.int64)
            S_arr = np.asarray(S_i)
            S_cast = np.ascontiguousarray(S_arr.astype(dtype, copy=False))
            tiles.append((idx, S_cast))
            if free_fp64:
                self.tile_schur_complements[tid] = S_cast
        self._tiles = tiles

        costs = [float(S.shape[0]) ** 2 for _, S in tiles]
        # Finding 15: cap the bin count at len(tiles) -- an explicit
        # matvec_threads > n_tiles otherwise produces permanently empty LPT
        # bins whose no-op work items are still dispatched through the
        # thread pool on EVERY matvec call (pure dispatch overhead in the
        # innermost hot loop, exactly the regime the module's own Stage 0
        # note says inverts scaling). Pool size itself (self.matvec_threads,
        # via _get_pool()) is left uncapped -- only the partition/dispatch
        # bin count shrinks to match the actual amount of parallelizable
        # work.
        self._mv_n_bins = min(self.matvec_threads, len(tiles)) if tiles else 1
        self._mv_partition = _lpt_partition(costs, self._mv_n_bins)

        touched: List[np.ndarray] = []
        for t in range(self._mv_n_bins):
            part = self._mv_partition[t]
            if part:
                touched.append(
                    np.unique(np.concatenate([tiles[i][0] for i in part]))
                )
            else:
                touched.append(np.empty(0, dtype=np.int64))
        self._mv_touched = touched

        self._S_extra_csr = (
            self.S_extra.tocsr() if self.S_extra is not None else None
        )

    def _tilewise_extra_term(self, X: np.ndarray) -> np.ndarray:
        """S_extra @ X (or a zero array), broadcasting to X's shape."""
        if self._S_extra_csr is not None:
            return np.asarray(self._S_extra_csr @ X, dtype=np.float64)
        return np.zeros_like(X, dtype=np.float64)

    def _tilewise_matvec(self, x: np.ndarray) -> np.ndarray:
        if self.matvec_threads <= 1 or len(self._tiles) < 2:
            return self._tilewise_matvec_serial(x)
        return self._tilewise_matvec_threaded(x)

    def _tilewise_matvec_serial(self, x: np.ndarray) -> np.ndarray:
        """Serial tilewise matvec (also the n_threads<=1 code path).

        Scatter-add via ``np.bincount``: ~10-30x faster than ``np.add.at``
        for dense index maps with repeated global indices (same bottleneck
        A1 replaced in ``solver_td.py`` for the transient RHS).
        """
        n = self.n
        dtype = self.matvec_dtype
        result = self._tilewise_extra_term(x)
        for idx, S_i in self._tiles:
            x_local = x[idx]
            if dtype != np.float64:
                x_local = x_local.astype(dtype, copy=False)
            y_local = np.asarray(S_i @ x_local, dtype=np.float64)
            result += np.bincount(idx, weights=y_local, minlength=n)
        return result

    def _tilewise_matvec_threaded(self, x: np.ndarray) -> np.ndarray:
        """Threaded tilewise matvec: LPT partition + compact per-thread buffers.

        Each thread's work is gather (``x[idx]``) -> GEMV
        (``S_i @ x_local``, releases the GIL inside BLAS) -> scatter into
        its OWN compact buffer (sized to its touched-index union, not the
        full interface) -- avoiding the Stage-0-flagged inverted scaling
        from a full ``(n_threads, n)`` accumulator's zero-fill + reduction
        cost growing with thread count.  BLAS is pinned to 1 thread for the
        duration of the pool region to avoid nested-parallelism oversubscription.
        """
        n = self.n
        dtype = self.matvec_dtype
        tiles = self._tiles
        part = self._mv_partition
        touched = self._mv_touched
        # Finding 15: dispatch over the CAPPED bin count (min(matvec_
        # threads, n_tiles), computed once in _prepare_tilewise_matvec_
        # state), not self.matvec_threads -- otherwise an explicit
        # matvec_threads > n_tiles dispatches permanently-empty no-op work
        # items through the pool on every matvec call.
        n_threads = self._mv_n_bins
        local_bufs: List[np.ndarray] = [None] * n_threads  # type: ignore[list-item]

        def work(t: int) -> None:
            u = touched[t]
            if len(u) == 0:
                local_bufs[t] = np.zeros(0, dtype=np.float64)
                return
            buf = np.zeros(len(u), dtype=np.float64)
            for i in part[t]:
                idx, S_i = tiles[i]
                x_local = x[idx]
                if dtype != np.float64:
                    x_local = x_local.astype(dtype, copy=False)
                y_local = np.asarray(S_i @ x_local, dtype=np.float64)
                pos = np.searchsorted(u, idx)
                buf += np.bincount(pos, weights=y_local, minlength=len(u))
            local_bufs[t] = buf

        pool = self._get_pool()
        with threadpoolctl.threadpool_limits(1):
            list(pool.map(work, range(n_threads)))

        result = self._tilewise_extra_term(x)
        for t in range(n_threads):
            u = touched[t]
            if len(u):
                result[u] += local_bufs[t]
        return result

    def _tilewise_matmat(self, X: np.ndarray) -> np.ndarray:
        """Per-tile GEMM + scatter (item 6): ``S_global-equivalent @ X``.

        Used by Stage 3's coarse-space setup (``S @ Z``, T'+1 columns) and
        exercised directly by the matmat-vs-column-matvecs equivalence
        test.  Threaded when ``matvec_threads > 1`` (same LPT
        partition/touched-index buffers as the matvec, generalised to
        multiple columns).
        """
        X = np.asarray(X, dtype=np.float64)
        squeeze = X.ndim == 1
        X2 = X.reshape(self.n, 1) if squeeze else X
        n, k = X2.shape[0], X2.shape[1]

        if self.matvec_threads <= 1 or len(self._tiles) < 2:
            result = self._tilewise_extra_term(X2)
            dtype = self.matvec_dtype
            for idx, S_i in self._tiles:
                X_local = X2[idx, :]
                if dtype != np.float64:
                    X_local = X_local.astype(dtype, copy=False)
                Y_local = np.asarray(S_i @ X_local, dtype=np.float64)
                np.add.at(result, idx, Y_local)
            return result[:, 0] if squeeze else result

        dtype = self.matvec_dtype
        tiles = self._tiles
        part = self._mv_partition
        touched = self._mv_touched
        # Finding 15: see the matching comment in _tilewise_matvec_threaded.
        n_threads = self._mv_n_bins
        local_bufs: List[np.ndarray] = [None] * n_threads  # type: ignore[list-item]

        def work(t: int) -> None:
            u = touched[t]
            if len(u) == 0:
                local_bufs[t] = np.zeros((0, k), dtype=np.float64)
                return
            buf = np.zeros((len(u), k), dtype=np.float64)
            for i in part[t]:
                idx, S_i = tiles[i]
                X_local = X2[idx, :]
                if dtype != np.float64:
                    X_local = X_local.astype(dtype, copy=False)
                Y_local = np.asarray(S_i @ X_local, dtype=np.float64)
                pos = np.searchsorted(u, idx)
                np.add.at(buf, pos, Y_local)
            local_bufs[t] = buf

        pool = self._get_pool()
        with threadpoolctl.threadpool_limits(1):
            list(pool.map(work, range(n_threads)))

        result = self._tilewise_extra_term(X2)
        for t in range(n_threads):
            u = touched[t]
            if len(u):
                result[u, :] += local_bufs[t]
        return result[:, 0] if squeeze else result

    def _build_preconditioner(self) -> Optional[spla.LinearOperator]:
        """Build preconditioner LinearOperator."""
        n = self.n

        if self.preconditioner == 'none':
            return None

        if self.preconditioner == 'jacobi':
            # Extract diagonal from S_global (assembled) or sum tile diagonals
            if self.S_global is not None:
                diag = np.array(self.S_global.diagonal(), dtype=np.float64)
            else:
                diag = np.zeros(n, dtype=np.float64)
                for tid, S_i in self.tile_schur_complements.items():
                    idx = self.tile_index_maps[tid]
                    np.add.at(diag, idx, np.diag(S_i))
                # T2/S4: never-assemble mode (S_global is None) needs
                # S_extra's diagonal (island 1e5 penalties + package
                # conductances) added in too -- S = sum_i P_i^T S_i P_i +
                # S_extra, so the tile-diagonal sum alone omits exactly the
                # terms that, in assembled mode, live in S_global's
                # diagonal. This is the same S4 fix already applied to the
                # sibling _build_jacobi_fallback (the memory-budget-downgrade
                # path); this is the EXPLICIT preconditioner='jacobi' branch,
                # which was missed. Use self.S_extra (set in __init__,
                # before this builder runs), not self._S_extra_csr (only
                # populated later by _build_linear_op()).
                if self.S_extra is not None:
                    diag += np.asarray(
                        self.S_extra.tocsr().diagonal(), dtype=np.float64
                    )
            diag = np.where(diag > 0, diag, 1.0)  # guard against zero diagonal

            def _msolve(x: np.ndarray) -> np.ndarray:
                return x / diag

            return spla.LinearOperator(shape=(n, n), matvec=_msolve, dtype=np.float64)

        if self.preconditioner == 'block_jacobi':
            return self._build_block_jacobi()

        if self.preconditioner == 'neumann':
            # NN/BDD work package: standalone one-level NN base (exists for
            # tests/ablation; production use is as the 'two_level' base
            # below, where the PoU coarse space provides BDD's "balancing").
            return self._build_neumann()

        if self.preconditioner == 'two_level':
            # Stage 3: the coarse-space TERM is layered on top AFTER
            # self._linear_op exists (see _augment_with_coarse_space, called
            # from __init__ right after _build_linear_op()) -- this builder
            # only produces the BASE component here (including any
            # memory-budget downgrade to 'jacobi', which the base builders
            # apply to self.preconditioner directly; _two_level_requested --
            # captured before this call -- is what the post-linear-op step
            # keys off of, not self.preconditioner, so the coarse term still
            # gets added even after a downgrade).  NN/BDD work package: the
            # base component is selected by two_level_base ('auto' resolves
            # to the legacy block_jacobi-with-downgrade path).
            if self._two_level_base == 'neumann':
                return self._build_neumann()
            if self._two_level_base == 'jacobi':
                # Explicitly-requested diagonal base -- not a downgrade
                # (skips the BJ ownership/estimate work entirely).
                self._base_builder_label = 'jacobi'
                return self._build_jacobi_fallback()
            return self._build_block_jacobi()

        if self.preconditioner == 'amg':
            return self._build_amg()

        return None

    def _augment_with_coarse_space(self) -> Optional[spla.LinearOperator]:
        """Stage 3: layer ``Z S_c^+ Z^T`` on top of the already-built base
        (block_jacobi, or its jacobi memory-budget downgrade) preconditioner.

        Must run AFTER ``self._linear_op`` exists (the coarse build's
        ``S @ Z`` matmat needs the fully-prepared tilewise/assembled matvec
        state -- see the module docstring's Stage 3 ordering note).
        ``build_coarse_space`` itself falls back to a PoU-only coarse space
        (still returned here, not ``None``) when GenEO pushes T' over the
        cap/byte budget; this method only degrades to the plain base
        preconditioner (``self._M``, already built) with a WARNING -- NEVER
        RAISES -- both when ``build_coarse_space`` returns ``None`` (PoU-only
        T'/bytes ALSO exceed budget, no usable ``tile_index_maps``, ``S_c``
        has no positive spectrum, ...) and (Finding 6) when it raises ANY
        exception outright (e.g. ``MemoryError``/``LinAlgError`` from the
        dense ``Z.toarray()``/``eigh(S_c)`` calls) -- a pathological S_c must
        degrade this (new-default, 'auto'-resolved) code path the same way
        the sibling block-Jacobi eigh fallback degrades (see the "Finding 7"
        comment in ``_build_block_jacobi`` for the matching precedent).
        """
        base_M = self._M
        # Finding 8: remember whether the base builder returned None
        # OUTRIGHT (e.g. every block-Jacobi block failed) -- distinct from
        # self._bj_downgraded (block_jacobi -> jacobi) -- so the degrade
        # path/label below can report 'none' instead of falsely claiming
        # 'block_jacobi' is active.
        self._bj_was_none = base_M is None
        # Finding 13: derive the base label ONCE here (the only place
        # _bj_was_none and _bj_downgraded are both known) -- see
        # self._base_precond_label's docstring in __init__.
        if self._bj_was_none:
            self._base_precond_label = 'none'
        elif self._base_builder_label is not None:
            # NN/BDD work package: a non-BJ base builder ran and recorded
            # its own name ('neumann', or the explicit-'jacobi' base) --
            # the _bj_downgraded/else chain below only describes the
            # legacy block_jacobi builder's outcomes.
            self._base_precond_label = self._base_builder_label
        elif self._bj_downgraded:
            self._base_precond_label = 'jacobi'
        else:
            self._base_precond_label = 'block_jacobi'
        base_apply = base_M.matvec if base_M is not None else (lambda r: r.copy())
        # A-DEF2: remember the RAW base apply (independent of whatever
        # self._M ends up holding below -- the additive combination for a
        # successful build, or base_M unchanged on a degrade) -- the
        # hand-rolled A-DEF2 loop needs M_base^-1 alone, never the additive
        # sum.  Set unconditionally (even on the degrade-to-base path) so
        # self._M_base_apply is always consistent with "the base
        # preconditioner actually in effect", though it is read only when
        # self._coarse is not None (the __call__ dispatch condition).
        self._M_base_apply = base_apply

        # Finding 9: read interface_coarse.DEFAULT_MAX_BYTES dynamically
        # (attribute lookup at call time), NOT a module-level snapshot --
        # see the removed COARSE_MAX_BYTES_DEFAULT constant's comment.
        _max_bytes = (
            self._coarse_max_bytes if self._coarse_max_bytes is not None
            else interface_coarse.DEFAULT_MAX_BYTES
        )
        # Finding 7: matvec_dtype only actually affects the matmat's fp
        # precision in 'tilewise' mode (S9 -- 'assembled' mode's sparse
        # matvec on S_global is always fp64 regardless of the dtype
        # setting), so only floor eps_rank when that matmat is genuinely
        # running in fp32.
        _coarse_matvec_dtype = (
            self.matvec_dtype if self.matvec_mode == 'tilewise'
            else np.dtype(np.float64)
        )
        try:
            coarse = interface_coarse.build_coarse_space(
                matmat=self._linear_op.matmat,
                tile_index_maps=self.tile_index_maps or {},
                n=self.n,
                island_idx=self._island_idx,
                geneo_pairs=self._geneo_pairs,
                max_cols=self._max_cols,
                eps_rank=self._eps_rank,
                max_bytes=_max_bytes,
                matvec_dtype=_coarse_matvec_dtype,
                retain_sz=(self._apply_mode == 'deflated'),
            )
        except Exception as exc:
            # Finding 6: build_coarse_space is documented "never raises" at
            # this call site, but its own dense allocation/eigh calls can
            # raise MemoryError/LinAlgError/ValueError on a pathological S_c
            # -- catch-all here (matching the sibling block-Jacobi eigh
            # fallback's convention) so a coarse-build failure degrades to
            # the base preconditioner instead of aborting prepare()/factor()
            # entirely.
            logger.warning(
                "InterfaceCGSolver: two_level coarse-space build raised "
                "%s: %s; degrading to the plain base preconditioner for "
                "this solve.",
                type(exc).__name__, exc,
            )
            coarse = None

        if coarse is None:
            # The base component's ACTUAL name -- self.preconditioner is
            # still 'two_level' at this point (only the success path below
            # overwrites it; _build_block_jacobi() only ever rewrites it to
            # 'jacobi' on a memory-budget downgrade, never back to
            # 'block_jacobi'), so use self._base_precond_label (Finding 13:
            # derived once, above) rather than trusting self.preconditioner
            # (Finding 8: must not claim 'block_jacobi' when the base
            # builder actually returned None).
            _actual = self._base_precond_label
            logger.warning(
                "InterfaceCGSolver: two_level coarse-space build failed or "
                "was disabled (see preceding WARNING); degrading to the "
                "plain %r preconditioner for this solve.",
                _actual,
            )
            self._coarse = None
            self.preconditioner = _actual
            return base_M

        self._coarse = coarse
        self.preconditioner = 'two_level'
        n = self.n

        def _msolve(r: np.ndarray) -> np.ndarray:
            return base_apply(r) + coarse.apply(r)

        logger.info(
            "InterfaceCGSolver: two_level preconditioner active -- %s",
            self.preconditioner_label,
        )
        return spla.LinearOperator(shape=(n, n), matvec=_msolve, dtype=np.float64)

    @property
    def preconditioner_label(self) -> str:
        """Human-readable preconditioner description for stats/logs, e.g.
        ``"two_level(bj+geneo k=4, T'=321, rank=320)"`` (additive, the
        byte-identical Stage 3 format -- unchanged) or
        ``"two_level[deflated](jacobi+geneo k=61, T'=126, rank=126)"``
        (A-DEF2 work package, apply_mode='deflated' AND the coarse build
        actually retained SZ -- see ``coarse.SZ``).  Falls back to the
        plain ``self.preconditioner`` string when no coarse space is active
        (including a degraded two_level request).

        The setting is named 'deflated', not 'adef2' (see this module's
        "A-DEF2 work package" docstring section for the ratification/
        selection record) -- the algorithm shipped IS what the setting name
        says, so (unlike an earlier revision of this module) no separate
        actual-vs-requested self-disclosure tag is needed;
        ``InterfaceCGSolver.stats['apply_algorithm']`` carries the same
        plain 'deflated' value in machine-readable form."""
        if self._coarse is not None:
            # Finding 13: self._base_precond_label (derived once in
            # _augment_with_coarse_space -- see its docstring) is 'none' |
            # 'jacobi' | 'block_jacobi'; the label uses the short 'bj' form
            # for the 'block_jacobi' case only (Finding 8: 'none' must not
            # fall through to 'bj' when the base builder returned None
            # outright -- coarse-only correction, no block/diagonal base at
            # all).
            base = (
                'bj' if self._base_precond_label == 'block_jacobi'
                else self._base_precond_label
            )
            # Only tag the label when apply_mode='deflated' genuinely took
            # effect (SZ retained) -- a byte-guard SZ-drop silently runs the
            # additive apply for this solve (see
            # interface_coarse.build_coarse_space's retain_sz docstring), so
            # the label must say so too, keeping the additive format
            # byte-identical whenever that is what is actually running.
            _tag = (
                '[deflated]'
                if self._apply_mode == 'deflated' and self._coarse.SZ is not None
                else ''
            )
            return (
                f"two_level{_tag}({base}+geneo k={self._coarse.n_geneo_cols}, "
                f"T'={self._coarse.n_cols}, rank={self._coarse.rank})"
            )
        return self.preconditioner

    def _form_owned_block(
        self,
        tile_or_node_group: Any,
        owned_global_arr: np.ndarray,
        S_csr: Optional[sp.csr_matrix],
        S_extra_csr: Optional[sp.csr_matrix],
        use_s_global: bool,
    ) -> Optional[np.ndarray]:
        """Dense, 1e-12-jittered ownership block for ``tile_or_node_group``.

        A-DEF2 work package (Deliverable 1): factored out of
        ``_build_block_jacobi``'s per-block loop so the EXACT same block
        (true ``S_global`` principal submatrix when available, else the
        per-tile Schur block + ``S_extra``) is used both there (the
        retained-factor path) and by :meth:`_extract_geneo_decoupled` (the
        new decoupled-GenEO path, which runs even when the base
        preconditioner downgrades to diagonal 'jacobi' and never reaches
        the retained-factor loop at all) -- one block-formation
        implementation, not two.

        Returns ``None`` only when neither ``S_global`` nor
        ``tile_schur_complements``/``tile_index_maps`` are available
        (mirrors the original inline ``continue`` case -- callers should
        skip this ownership group).
        """
        if use_s_global:
            sub = S_csr[np.ix_(owned_global_arr, owned_global_arr)].toarray()
        elif self.tile_schur_complements is not None and self.tile_index_maps is not None:
            tid = tile_or_node_group
            S_i = np.asarray(self.tile_schur_complements[tid], dtype=np.float64)
            idx_full = self.tile_index_maps[tid]
            global_to_local: Dict[int, int] = {
                int(g): loc for loc, g in enumerate(idx_full)
            }
            owned_local = np.array(
                [global_to_local[g] for g in owned_global_arr], dtype=np.int32
            )
            sub = S_i[np.ix_(owned_local, owned_local)].copy()
            if S_extra_csr is not None:
                sub += S_extra_csr[
                    np.ix_(owned_global_arr, owned_global_arr)
                ].toarray()
        else:
            return None

        # Regularize to ensure SPD.
        sub += 1e-12 * np.eye(sub.shape[0])
        return sub

    def _cho_or_eigh_with_geneo(
        self,
        sub: np.ndarray,
        island_local_mask: Optional[np.ndarray],
        owned_global_arr: np.ndarray,
        log_prefix: str,
        run_geneo: bool,
    ) -> Tuple[Optional[Tuple[str, Any]], bool]:
        """Cholesky-factor ``sub`` (SPD-safe eigh fallback on
        ``LinAlgError``), optionally running GenEO-lite enrichment on
        whichever factor succeeds.

        Finding 13 (A-DEF2 code review, round 1): shared by
        ``_build_block_jacobi``'s retained-factor loop
        (``run_geneo=self._want_geneo``; caller retains the returned
        factor payload for preconditioner application) and
        :meth:`_extract_geneo_decoupled` (``run_geneo=True`` always; the
        returned factor payload is discarded by the caller instead of
        retained) -- these two call sites used to duplicate this cascade
        almost verbatim. ``log_prefix`` distinguishes each call site's log
        messages ('Block-Jacobi' / 'Decoupled GenEO').

        Error handling (S11 / Finding 7, pre-existing, preserved unchanged):
        ``la.cho_factor`` failing with ``LinAlgError`` falls back to
        ``_spd_safe_pseudo_solve_factor_ex`` (eigh-based SPD projection, NOT
        ``np.linalg.pinv`` -- pinv of a numerically indefinite symmetric
        block can retain tiny negative eigenvalues from FP noise, silently
        voiding CG's SPD-preconditioner assumption). The eigh fallback
        ITSELF is wrapped in a catch-all (``except Exception``) since
        ``np.linalg.eigh`` can raise ``MemoryError``/``ValueError`` on a
        pathological block -- degrades to "skip this block" rather than
        aborting the whole ``prepare()``/``factor()`` call. GenEO itself
        gets its own narrow ``try/except`` so a GenEO-only failure never
        lands in the ``LinAlgError`` handler above and double-appends a
        factor for the same ``owned_global_arr`` (Finding 2, round 1).

        NOTE: does NOT itself guard ``la.cho_factor``/``np.linalg.eigh``
        against ``MemoryError`` on the FIRST attempt (only the eigh
        fallback's own re-entry is wrapped, per the paragraph above).
        ``_extract_geneo_decoupled`` (an optional enrichment pass) wraps
        this method's call in its own ``except MemoryError`` to degrade
        gracefully; ``_build_block_jacobi``'s retained-factor loop (the
        BASE preconditioner) deliberately does NOT -- see round-2 code
        review finding 2 -- a MemoryError there must propagate and abort
        ``prepare()``/``factor()`` fail-fast, not silently skip a block
        that's needed for the preconditioner CG actually uses.

        Returns:
            ``(factor_repr, geneo_contributed)`` where ``factor_repr`` is
            ``('cho', cho)`` / ``('eigh', (V, inv_w))`` on success or
            ``None`` when even the eigh fallback fails outright, and
            ``geneo_contributed`` is ``True`` iff a GenEO call actually
            appended near-null columns to ``self._geneo_pairs``.
        """
        try:
            cho = la.cho_factor(sub, lower=False, check_finite=False)
        except la.LinAlgError:
            logger.warning(
                "%s: owned block size %d is singular/indefinite; using "
                "eigh-based SPD-safe pseudo-inverse fallback (clipped to "
                ">= eps*lambda_max).",
                log_prefix, sub.shape[0],
            )
            try:
                eigh_factor_ex = _spd_safe_pseudo_solve_factor_ex(sub)
            except Exception as exc:
                logger.warning(
                    "%s: eigh-based SPD-safe fallback itself failed on "
                    "block size %d (%s: %s); skipping this block.",
                    log_prefix, sub.shape[0], type(exc).__name__, exc,
                )
                return None, False
            if eigh_factor_ex is None:
                logger.warning(
                    "%s: block size %d has no positive spectrum "
                    "(lambda_max <= 0); skipping this block.",
                    log_prefix, sub.shape[0],
                )
                return None, False
            V_eigh, inv_w_eigh, w_eigh = eigh_factor_ex
            contributed = False
            if run_geneo:
                try:
                    V_k, w_k = interface_coarse.geneo_lowest_eigenpairs(
                        sub, k=self._geneo_k, tol=self._geneo_tol,
                        precomputed=(w_eigh, V_eigh),
                        island_local_mask=island_local_mask,
                    )
                    if V_k.shape[1] > 0:
                        self._geneo_pairs.append((owned_global_arr, V_k, w_k))
                        contributed = True
                except Exception as exc:
                    logger.warning(
                        "%s: GenEO enrichment failed on the eigh-fallback "
                        "block size %d (%s: %s); skipping enrichment for "
                        "this block (keeps the eigh factor).",
                        log_prefix, sub.shape[0], type(exc).__name__, exc,
                    )
            return ('eigh', (V_eigh, inv_w_eigh)), contributed
        else:
            contributed = False
            if run_geneo:
                try:
                    V_k, w_k = interface_coarse.geneo_lowest_eigenpairs(
                        sub, cho=cho, k=self._geneo_k, tol=self._geneo_tol,
                        island_local_mask=island_local_mask,
                    )
                    if V_k.shape[1] > 0:
                        self._geneo_pairs.append((owned_global_arr, V_k, w_k))
                        contributed = True
                except Exception as exc:
                    logger.warning(
                        "%s: GenEO enrichment failed on block size %d (%s: "
                        "%s); skipping enrichment for this block (keeps "
                        "the cho factor -- block-Jacobi itself is "
                        "unaffected).",
                        log_prefix, sub.shape[0], type(exc).__name__, exc,
                    )
            return ('cho', cho), contributed

    def _extract_geneo_decoupled(self, tile_owned: Dict[Any, List[int]]) -> None:
        """A-DEF2 work package (Deliverable 1): GenEO-lite extraction
        decoupled from which base preconditioner actually gets built.

        Before this method existed, GenEO enrichment ran ONLY inside
        ``_build_block_jacobi``'s retained-factor loop (below) -- when the
        byte-budget guard downgraded the base to diagonal 'jacobi' (the
        ``_build_jacobi_fallback`` early-return, triggered BEFORE that
        loop), no block was ever cho-factored, so ``self._geneo_pairs``
        stayed permanently empty, discarding real near-null eigendirection
        material at exactly the split regime that benefits from it most
        (measured detail: docs §7.8/§7.9, ``interface_deflation_notes.md``).

        Called with the SAME ``tile_owned`` (ownership assignment) the
        retained-factor loop uses, so column labeling/ordering in the
        eventual coarse space is unaffected by which code path populated
        ``self._geneo_pairs``. Each block is formed via
        :meth:`_form_owned_block` (identical to the retained loop), then
        factored ONE BLOCK AT A TIME via the SHARED
        :meth:`_cho_or_eigh_with_geneo` helper (Finding 13) -- the factor is
        DISCARDED at the end of each iteration (never appended anywhere)
        instead of retained for preconditioner application, so peak
        coordinator memory during this pass is bounded by the single
        largest PROCESSED block's factor, not the sum over all blocks (this
        pass must not reintroduce the pressure the byte-budget guard just
        downgraded away from).

        Memory guard (Finding 1, round 1; multiplier corrected by Finding 7,
        round 2): before forming a block, its estimated dense
        formation+cho/eigh-fallback peak (``5 * k^2 * 8`` bytes -- covers
        the WORST of the two paths ``_cho_or_eigh_with_geneo`` can take, not
        just the cheaper 2x cho_factor-succeeds case; see the estimate's own
        inline comment for the accounting) is checked against a per-block
        cap derived from the resolved ``interface_block_jacobi_max_bytes``
        (see ``GENEO_DECOUPLED_MIN_BLOCK_CAP_BYTES`` for the floor rationale);
        an over-cap block is skipped (WARNING) WITHOUT ever calling
        :meth:`_form_owned_block`. Block formation and the cho-factor-or-
        eigh-fallback call are ALSO wrapped in a ``MemoryError``-tolerant
        guard (unlike ``_build_block_jacobi``'s retained-factor loop, which
        must stay fail-fast -- see round-2 code review finding 2): even a
        block that passes the pre-flight estimate can still exhaust memory
        in practice, and that must skip the block (this is an optional
        enrichment pass), not abort ``prepare()``/``factor()``.

        Processes blocks SEQUENTIALLY, not via the thread pool (spec-
        permitted choice -- this method only runs on the infrequent
        memory-downgrade path, so build-time throughput is secondary to
        simplicity here). Timed as its own phase (INFO log) -- see
        ``interface_deflation_notes.md`` for the measured one-time cost.
        """
        t0 = time.perf_counter()
        _use_s_global = self.S_global is not None
        S_csr = self.S_global.tocsr() if _use_s_global else None
        _S_extra_csr_early = (
            self.S_extra.tocsr() if self.S_extra is not None else None
        )
        _max_bytes = (
            self.block_jacobi_max_bytes
            if self.block_jacobi_max_bytes is not None
            else BLOCK_JACOBI_MAX_FACTOR_BYTES
        )
        # See GENEO_DECOUPLED_MIN_BLOCK_CAP_BYTES's module-level docstring
        # for why the floor is necessary (a resolved budget configured far
        # below any real per-block cost -- e.g. this module's own tests --
        # must not reject every block outright).
        _per_block_cap = max(_max_bytes, GENEO_DECOUPLED_MIN_BLOCK_CAP_BYTES)
        _n_blocks_enriched = 0
        for tile_or_node_group, owned_global in tile_owned.items():
            owned_global_arr = np.array(sorted(owned_global), dtype=np.int32)
            k = owned_global_arr.shape[0]
            # Finding 1 (round 1) / Finding 7 (round 2, corrects an
            # under-count in the round-1 estimate): this pre-flight must
            # cover the WORST-CASE peak among the two paths
            # _cho_or_eigh_with_geneo can take, not just the common
            # (cho_factor succeeds) one:
            #   - cho_factor path: sub (k^2*8) + cho_factor's own internal
            #     working copy (another k^2*8) = 2*k^2*8.
            #   - eigh-fallback path (singular/indefinite block -- the exact
            #     pathological case this guard targets): sub (k^2*8) is
            #     still live, PLUS _spd_safe_pseudo_solve_factor_ex's own
            #     ``0.5*(sub+sub.T)`` allocation (another k^2*8) AND
            #     ``np.linalg.eigh``'s internal LAPACK workspace (divide-
            #     and-conquer ``syevd`` needs roughly 2*k^2 doubles of
            #     scratch beyond its output arrays) -- roughly 4*k^2*8
            #     concurrent with ``sub`` itself, ~5*k^2*8 total. Sizing the
            #     pre-flight for the cho_factor path's cheaper 2x therefore
            #     admits blocks that, on hitting the eigh fallback, attempt
            #     ~2-2.5x the intended per-block cap -- risking an OOM-kill
            #     (SIGKILL, not a catchable MemoryError) rather than the
            #     graceful WARNING-and-skip degrade this guard exists to
            #     guarantee. Use the eigh-path multiplier (5x) so the guard
            #     stays honest for the pathological blocks it targets; this
            #     is conservative for the (more common) cho_factor-succeeds
            #     case, which is the intended direction to err.
            est_block_bytes = 5 * k * k * 8
            if est_block_bytes > _per_block_cap:
                logger.warning(
                    "Decoupled GenEO: skipping ownership block of size %d "
                    "(~%.1f MB estimated dense formation+cho/eigh-fallback "
                    "peak exceeds the %.1f MB per-block cap derived from "
                    "'interface_block_jacobi_max_bytes'); this block's "
                    "rows enter the coarse space as PoU-only (no GenEO "
                    "columns).",
                    k, est_block_bytes / 1024.0 ** 2,
                    _per_block_cap / 1024.0 ** 2,
                )
                continue

            try:
                sub = self._form_owned_block(
                    tile_or_node_group, owned_global_arr, S_csr,
                    _S_extra_csr_early, _use_s_global,
                )
            except MemoryError as exc:
                logger.warning(
                    "Decoupled GenEO: forming ownership block of size %d "
                    "raised MemoryError (%s); skipping this block (PoU-"
                    "only for its rows) rather than aborting prepare().",
                    k, exc,
                )
                continue
            if sub is None:
                continue

            island_local_mask = (
                np.isin(owned_global_arr, self._island_idx)
                if self._island_idx is not None else None
            )

            try:
                _factor_repr, contributed = self._cho_or_eigh_with_geneo(
                    sub, island_local_mask, owned_global_arr,
                    log_prefix="Decoupled GenEO", run_geneo=True,
                )
            except MemoryError as exc:
                logger.warning(
                    "Decoupled GenEO: factoring ownership block of size "
                    "%d raised MemoryError (%s); skipping this block "
                    "(PoU-only for its rows) rather than aborting "
                    "prepare().",
                    k, exc,
                )
                continue
            # `_factor_repr` (the 'cho'/'eigh' payload) is intentionally
            # discarded here -- never retained, see this method's
            # docstring -- it falls out of scope at the end of this
            # iteration (peak memory = one block's factor, not the sum).
            if contributed:
                _n_blocks_enriched += 1

        elapsed = time.perf_counter() - t0
        if _n_blocks_enriched > 0:
            # Finding 12: only claim the coarse space "stays PoU+GenEO"
            # when at least one block actually contributed columns --
            # otherwise (every block failed/skipped/had no near-null
            # spectrum) it degrades to PoU-only exactly like the
            # geneo_k=0/disabled case, and the log must say so.
            logger.info(
                "Decoupled GenEO: %d/%d ownership block(s) contributed "
                "columns (base preconditioner downgraded to diagonal "
                "'jacobi'; coarse space stays PoU+GenEO instead of "
                "degrading to PoU-only), %.3fs.",
                _n_blocks_enriched, len(tile_owned), elapsed,
            )
        else:
            logger.info(
                "Decoupled GenEO: 0/%d ownership block(s) contributed "
                "columns (base preconditioner downgraded to diagonal "
                "'jacobi'; coarse space stays PoU-only), %.3fs.",
                len(tile_owned), elapsed,
            )

    def _build_block_jacobi(self) -> Optional[spla.LinearOperator]:
        """Block-Jacobi preconditioner.

        Two implementation paths:
        1. S_global available (assembled mode or tilewise with S_global passed):
           uses the actual principal submatrices of S_global for each tile's
           owned nodes.  These are the TRUE block-diagonal blocks of the global
           system.  Much better preconditioner than path 2.
        2. S_global not available (tilewise-only): falls back to per-tile
           Schur complement submatrices.  These are NOT the diagonal blocks of
           S_global (they miss cross-tile couplings and package edges), so they
           are a weaker approximation.

        Ownership assignment: each interface node is assigned to the FIRST
        tile (in iteration order of tile_index_maps keys) whose tile_index_map
        includes it.  The owner contributes the block.

        Each per-owner submatrix is factored via scipy.linalg.cho_factor (dense
        Cholesky) for numerically robust small-block solves.  Falls back to
        pinv on singular blocks (1e-12 diagonal regularization applied first).

        Memory guard: when the estimated factor memory across all blocks exceeds
        ``BLOCK_JACOBI_MAX_FACTOR_BYTES`` (default 4 GB), falls back to the
        'jacobi' (diagonal) preconditioner automatically and logs a WARNING.
        This prevents the block-Jacobi from re-introducing the coordinator-
        memory pressure that CG mode is designed to remove.  Scaling analysis:
        for T balanced tiles with n total interface nodes, total factor memory
        is Sum(k_i^2) * 8 bytes ≈ n^2/T * 8 bytes.  At n=1M, T=1000 this is
        ~8 GB; with fewer/larger tiles it grows rapidly.  Use 'jacobi' or
        increase T (via B1 splitting) when targeting 1M+ interface nodes.

        Returns None only on a complete build failure; triggers logged warning.
        """
        n = self.n

        # ---- 1. Assign ownership (first-seen tile for each global index) ----
        owner_tile: Dict[int, Any] = {}
        if self.tile_index_maps:
            for tid, idx in self.tile_index_maps.items():
                for g in idx:
                    gi = int(g)
                    if gi not in owner_tile:
                        owner_tile[gi] = tid
        elif self.tile_schur_complements is None:
            # Neither tile_index_maps nor tile_schur_complements: cannot build
            return None

        # ---- 2. Collect owned global indices per tile ----
        tile_owned: Dict[Any, List[int]] = {}
        for g, tid in owner_tile.items():
            tile_owned.setdefault(tid, []).append(g)

        # ---- 2b. Pre-flight memory estimate ----
        # Sum(k_i^2) * 8 bytes = total dense Cholesky factor memory.
        # Each k_i is the number of nodes owned by tile i.  Use the
        # owned-count dict (already built) to estimate without allocating.
        #
        # Stage 1c: use the per-instance override when provided (resolved
        # from the 'interface_block_jacobi_max_bytes' setting), else fall
        # back to the module-level constant (read dynamically so test
        # monkeypatches of the module attribute keep working).
        _max_bytes = (
            self.block_jacobi_max_bytes
            if self.block_jacobi_max_bytes is not None
            else BLOCK_JACOBI_MAX_FACTOR_BYTES
        )
        _max_block_pre = max(len(v) for v in tile_owned.values()) if tile_owned else 0
        est_factor_bytes = sum(
            len(owned) * len(owned) * 8 for owned in tile_owned.values()
        )
        # Finding 10 (round 1) / Finding 8 (round 2): the materialization
        # loop below (section 4) transiently holds, for the block currently
        # being symmetrized (`Sinv = 0.5 * (Sinv + Sinv.T)`), FOUR k x k
        # arrays at once: the not-yet-replaced cho factor (`payload`,
        # already counted in est_factor_bytes above), the freshly-built
        # dense inverse from cho_solve (`Sinv`), the `Sinv + Sinv.T` sum
        # temporary, and the `0.5 * (...)` result array that `Sinv` gets
        # reassigned to (the OLD Sinv object from cho_solve is not
        # reclaimed until this expression finishes evaluating) -- three
        # arrays BEYOND the counted cho factor, not two.  Add the
        # worst-case extra (the single largest block's three transient
        # arrays) so this pre-flight estimate does not underestimate the
        # actual build-time peak.
        est_factor_bytes += 3 * _max_block_pre * _max_block_pre * 8
        if est_factor_bytes > _max_bytes:
            max_block = _max_block_pre
            # Finding 5: this builder also serves preconditioner='two_level'
            # requests (dispatched from _build_preconditioner's two_level
            # branch, which calls this method to build the BASE component --
            # see that branch's comment) -- name the ACTUAL requested
            # preconditioner (self.requested_preconditioner), not a
            # hardcoded 'block_jacobi', and -- when the request was
            # 'two_level' -- say so explicitly.
            #
            # A-DEF2 work package (Deliverable 1): this fallback path
            # (_build_jacobi_fallback) never reaches the RETAINED-factor
            # per-block cho_factor loop below, but GenEO enrichment is NO
            # LONGER silently skipped as a result -- _extract_geneo_
            # decoupled runs its OWN one-block-at-a-time factor+discard
            # pass right here when self._want_geneo, so the eventual
            # coarse space (if two_level) still gets PoU+GenEO, not
            # PoU-only, even with a diagonal base.
            _is_two_level = self.requested_preconditioner == 'two_level'
            logger.warning(
                "Block-Jacobi: estimated factor memory %.1f GB exceeds budget "
                "%.1f GB (setting 'interface_block_jacobi_max_bytes', "
                "n_interface=%d, T=%d tiles, max_block=%d). Downgrading "
                "requested preconditioner %r -> 'jacobi' "
                "(diagonal) for this solve -- expect MORE CG iterations per "
                "solve than block_jacobi would have needed (diagonal is a "
                "strictly weaker preconditioner).%s "
                "Scaling: Sum(k_i^2)*8 bytes ~ n^2/T*8; at n=1M, T=1000 this "
                "is ~8 GB. To keep block_jacobi, reduce interface size via B1 "
                "tile splitting or raise 'interface_block_jacobi_max_bytes'.",
                est_factor_bytes / 1024 ** 3,
                _max_bytes / 1024 ** 3,
                n,
                len(tile_owned),
                max_block,
                self.requested_preconditioner,
                (
                    " GenEO enrichment runs via a DECOUPLED one-block-at-a-"
                    "time pass (factor discarded after each block's GenEO "
                    "call, so peak memory stays bounded) -- the two_level "
                    "coarse space -- if it can still be built at all -- "
                    "stays PoU+GenEO, not PoU-only."
                    if _is_two_level and self._want_geneo else
                    " GenEO enrichment is disabled "
                    "(interface_coarse_geneo_k=0); the two_level coarse "
                    "space -- if it can still be built at all -- will be "
                    "PoU-only."
                    if _is_two_level else ""
                ),
            )
            # Fall back to diagonal preconditioner (cheap, always safe).
            # Keep self.requested_preconditioner == 'block_jacobi' (set in
            # __init__) so callers/logs can distinguish "asked for" from
            # "actually in use"; downgrade self.preconditioner so every
            # downstream report (backend_info, verbose logs) reflects reality.
            self.preconditioner = 'jacobi'
            self._bj_downgraded = True
            # A-DEF2 work package (Deliverable 1): GenEO extraction
            # decoupled from the base -- runs even though this path never
            # reaches the retained-factor loop below.  Guarded by
            # self._want_geneo (== _two_level_requested and geneo_k > 0)
            # so a plain 'block_jacobi' request (not 'two_level') or an
            # explicit interface_coarse_geneo_k=0 correctly pays nothing.
            if self._want_geneo:
                self._extract_geneo_decoupled(tile_owned)
            return self._build_jacobi_fallback()

        logger.debug(
            "Block-Jacobi: estimated factor memory %.1f MB "
            "(n=%d, T=%d tiles, max_block=%d).",
            est_factor_bytes / 1024 ** 2,
            n,
            len(tile_owned),
            _max_block_pre,
        )

        # ---- 3. Extract submatrices and build Cholesky factors ----
        _block_factors: List[Tuple[np.ndarray, Any, bool]] = []

        # Path 1: use actual diagonal blocks of S_global (preferred)
        _use_s_global = self.S_global is not None
        S_csr = self.S_global.tocsr() if _use_s_global else None
        # S4: never-assemble mode (S_global is None) needs S_extra's
        # contribution (island 1e5 penalties + package conductances) added
        # into each owned block too -- S = sum_i P_i^T S_i P_i + S_extra, so
        # a block built from tile S_i alone omits exactly the terms that, in
        # assembled mode, live in S_global's diagonal blocks.  Use
        # self.S_extra (available from __init__, before self._S_extra_csr
        # is populated by _build_linear_op() later).
        _S_extra_csr_early = (
            self.S_extra.tocsr() if self.S_extra is not None else None
        )

        # Round-2 code review finding 2: unlike _extract_geneo_decoupled
        # (below), this retained-factor loop builds the BASE preconditioner
        # itself -- the thing CG correctness/performance depends on, not an
        # optional enrichment pass -- so block formation/factoring here must
        # stay FAIL-FAST on MemoryError, exactly as it was before Finding 1
        # (round 1) introduced a shared _form_owned_block/_cho_or_eigh_with_
        # geneo helper pair. Round 1 correctly wrapped the NEW decoupled-
        # GenEO pass's calls to those helpers in `except MemoryError`
        # (that pass is a graceful-degradation enrichment step -- see its
        # own docstring), but the shared refactor also accidentally added
        # the SAME guard around THESE two call sites, silently converting a
        # coordinator-under-memory-pressure MemoryError that used to abort
        # prepare()/factor() immediately (with a clear signal to raise the
        # budget or retile) into "skip this block, fall back to identity
        # preconditioning for its nodes" -- masking the real out-of-memory
        # root cause behind a slow, confusing CG stall or strict-mode
        # non-convergence error, potentially hours into a transient run.
        # Neither call is wrapped here anymore; a MemoryError propagates
        # out of this method (and __init__) uncaught, matching pre-Finding-1
        # behavior. GenEO enrichment failures (a narrower, already-caught
        # concern) still degrade gracefully -- see
        # _cho_or_eigh_with_geneo's own inner `except Exception` around
        # each `geneo_lowest_eigenpairs` call, which never lets a GenEO-only
        # failure reach this loop as an exception at all.
        for tile_or_node_group, owned_global in tile_owned.items():
            owned_global_arr = np.array(sorted(owned_global), dtype=np.int32)

            sub = self._form_owned_block(
                tile_or_node_group, owned_global_arr, S_csr,
                _S_extra_csr_early, _use_s_global,
            )
            if sub is None:
                continue

            # Finding 3: local (within-this-block) island mask, used by
            # BOTH GenEO call sites below so the eigen-analysis runs on the
            # physical (non-penalty-inflated) spectrum -- see
            # interface_coarse.geneo_lowest_eigenpairs's island_local_mask
            # docstring for the full rationale.  Island nodes CAN appear in
            # an ownership block (see interface_coarse.py's module
            # docstring, "Island rows in GenEO columns").
            island_local_mask = (
                np.isin(owned_global_arr, self._island_idx)
                if self._island_idx is not None else None
            )

            # Finding 13: the cho_factor-then-eigh-fallback-then-GenEO
            # cascade (S11's catch-all resilience, Finding 2's
            # double-append fix, Finding 7's MemoryError/ValueError
            # catch-all on the eigh fallback) lives ONCE in the shared
            # _cho_or_eigh_with_geneo helper -- also used by
            # _extract_geneo_decoupled (the jacobi-downgrade path).
            _factor_repr, _contributed = self._cho_or_eigh_with_geneo(
                sub, island_local_mask, owned_global_arr,
                log_prefix="Block-Jacobi", run_geneo=self._want_geneo,
            )
            if _factor_repr is not None:
                _kind, _payload = _factor_repr
                _block_factors.append((owned_global_arr, _kind, _payload))

        if not _block_factors:
            logger.warning(
                "Block-Jacobi: all blocks failed; falling back to no preconditioner."
            )
            return None

        # ---- 4. Materialize each block's dense (pseudo-)inverse, in place --
        # BJ-apply perf fix (Stage 3): the OLD apply (kept as _bj_apply_one's
        # cho_solve/two-GEMV forms up through Stage 2) gathered
        # ``x[global_idx]`` and scattered ``result[global_idx] = ...`` once
        # PER BLOCK -- fancy-index random-access numpy ops that do not
        # release the GIL, measured (Stage 2) to serialize threads down to a
        # 1.4x speedup despite cho_solve's own LAPACK call releasing the
        # GIL fine. Replaced by materializing each block's dense SPD
        # (pseudo-)inverse HERE (cho_solve(cho, eye(k)) for 'cho' blocks; the
        # already-available V @ diag(inv_w) @ V.T for 'eigh' blocks) --
        # REPLACING (not duplicating) the cho-factor payload in
        # _block_factors, so total memory stays Sum(k_i^2)*8 bytes, the same
        # order as before -- so apply becomes ONE global gather + a loop of
        # CONTIGUOUS-slice GEMVs (BLAS-2, releases the GIL) + ONE global
        # scatter (see _bj_solve_serial/_bj_solve_threaded below).
        _bj_perm_parts: List[np.ndarray] = []
        _bj_inv_blocks: List[np.ndarray] = []
        _offsets = [0]
        for _bi, (global_idx, kind, payload) in enumerate(_block_factors):
            k = len(global_idx)
            if kind == 'cho':
                Sinv = la.cho_solve(payload, np.eye(k), check_finite=False)
                # Finding 9: cho_solve's explicitly-formed inverse is not
                # exactly symmetric (unlike the backward-stable triangular
                # solves the old apply-time cho_solve(cho, x) design used
                # directly) -- on the ill-conditioned blocks this feature
                # targets (near-null eigenvalues ~1e-10 relative, cond up to
                # ~1e10 per the module docstring) the asymmetry is large
                # enough to weaken PCG's SPD-preconditioner assumption.
                # Cheap, exact fix: symmetrize in place.  ('eigh' blocks
                # below are already exactly symmetric by construction --
                # V @ diag(inv_w) @ V.T -- so this is only needed here.)
                Sinv = 0.5 * (Sinv + Sinv.T)
            else:
                V, inv_w = payload
                Sinv = (V * inv_w) @ V.T
            Sinv = np.ascontiguousarray(Sinv)
            _block_factors[_bi] = (global_idx, kind, Sinv)
            _bj_perm_parts.append(global_idx)
            _bj_inv_blocks.append(Sinv)
            _offsets.append(_offsets[-1] + k)

        self._bj_block_factors = _block_factors
        self._bj_perm = (
            np.concatenate(_bj_perm_parts) if _bj_perm_parts
            else np.empty(0, dtype=np.int64)
        )
        self._bj_offsets = np.array(_offsets, dtype=np.int64)
        self._bj_inv_blocks = _bj_inv_blocks

        # ---- 5. LPT partition of blocks by k^2 for threaded apply ----------
        # Threading is safe WITHOUT accumulation: block ownership is a
        # partition (each interface node owned by exactly one block), so
        # concurrent threads write to disjoint SLICES of y_perm below --
        # no race, no reduction step needed (item 5).
        # Finding 15: cap the bin count at len(_block_factors) -- see the
        # matching comment in _prepare_tilewise_matvec_state.
        self._bj_n_bins = min(self.matvec_threads, len(_block_factors))
        self._bj_partition = _lpt_partition(
            [float(len(idx)) ** 2 for idx, _, _ in _block_factors],
            self._bj_n_bins,
        )

        def _bj_solve_serial(x: np.ndarray) -> np.ndarray:
            result = x.copy()  # identity for any unowned nodes
            perm = self._bj_perm
            if len(perm):
                offs = self._bj_offsets
                x_perm = x[perm]                      # ONE gather
                y_perm = np.empty_like(x_perm)
                for i, Sinv in enumerate(self._bj_inv_blocks):
                    s, e = offs[i], offs[i + 1]
                    y_perm[s:e] = Sinv @ x_perm[s:e]   # contiguous GEMV
                result[perm] = y_perm                  # ONE scatter
            return result

        def _bj_solve_threaded(x: np.ndarray) -> np.ndarray:
            result = x.copy()  # identity for any unowned nodes
            perm = self._bj_perm
            if len(perm) == 0:
                return result
            offs = self._bj_offsets
            inv_blocks = self._bj_inv_blocks
            x_perm = x[perm]                           # ONE gather
            y_perm = np.empty_like(x_perm)

            def work(t: int) -> None:
                for bi in self._bj_partition[t]:
                    s, e = offs[bi], offs[bi + 1]
                    # Disjoint contiguous slice per block -- no lock needed;
                    # dgemv releases the GIL so this genuinely parallelizes,
                    # unlike the old per-block fancy-index gather/scatter.
                    y_perm[s:e] = inv_blocks[bi] @ x_perm[s:e]

            pool = self._get_pool()
            with threadpoolctl.threadpool_limits(1):
                # Finding 15: dispatch over the CAPPED bin count, not
                # self.matvec_threads -- see the matching comment in
                # _prepare_tilewise_matvec_state.
                list(pool.map(work, range(self._bj_n_bins)))
            result[perm] = y_perm                       # ONE scatter
            return result

        def _bj_solve(x: np.ndarray) -> np.ndarray:
            if self.matvec_threads <= 1 or len(_block_factors) < 2:
                return _bj_solve_serial(x)
            return _bj_solve_threaded(x)

        return spla.LinearOperator(
            shape=(n, n), matvec=_bj_solve, dtype=np.float64
        )

    def _build_jacobi_fallback(self) -> Optional[spla.LinearOperator]:
        """Build a diagonal (Jacobi) preconditioner as a fallback.

        Used when block_jacobi memory exceeds the budget threshold, or as an
        explicit alternative. Cheap and scalable to any interface size.
        """
        n = self.n
        if self.S_global is not None:
            diag = np.array(self.S_global.diagonal(), dtype=np.float64)
        elif self.tile_schur_complements is not None and self.tile_index_maps is not None:
            diag = np.zeros(n, dtype=np.float64)
            for tid, S_i in self.tile_schur_complements.items():
                idx = self.tile_index_maps[tid]
                # Scatter-add diagonal entries using bincount (fast path)
                diag += np.bincount(idx, weights=np.diag(S_i), minlength=n)
            # S4: S_global is None in never-assemble mode -- S = sum_i P_i^T
            # S_i P_i + S_extra, so the true diagonal also needs S_extra's
            # diagonal (island 1e5 penalties + package conductances), not
            # just the summed tile S_i diagonals.  Omitting it under-scales
            # the preconditioner on every islanded/package-coupled node.
            # NB: use self.S_extra (set in __init__ before this builder
            # runs), NOT self._S_extra_csr -- that cache is only populated
            # later, inside _build_linear_op()/_prepare_tilewise_matvec_
            # state(), which __init__ calls AFTER the preconditioner.
            if self.S_extra is not None:
                diag += np.asarray(
                    self.S_extra.tocsr().diagonal(), dtype=np.float64
                )
        else:
            return None  # cannot build any preconditioner

        diag = np.where(diag > 0, diag, 1.0)

        def _diag_solve(x: np.ndarray) -> np.ndarray:
            return x / diag

        return spla.LinearOperator(shape=(n, n), matvec=_diag_solve, dtype=np.float64)

    def _build_neumann(self) -> Optional[spla.LinearOperator]:
        """Weighted Neumann-Neumann / BDD fine-space preconditioner.

        NN/BDD work package (Candidate 1 of
        ``docs/interface_precond_sota_research.md``):

            M^-1 = sum_i R_i^T D_i S~_i^+ D_i R_i   (+ diagonal complement)

        One dense (pseudo-)inverse per FULL tile Schur block ``S_i`` -- not
        the block-Jacobi owned slice, so the neighbor-tile coupling BJ
        discards (up to 4.5x the kept Frobenius mass, §7.14) is retained --
        reconciled across tiles by partition-of-unity weights ``D_i``
        (``sum_i D_i = I`` on every tile-covered node).  Coefficient
        (stiffness) weights by default: Mandel-Brezina's subdomain-count-
        independence result requires coefficient weighting, not
        multiplicity counting.

        Conventions mirrored from the sibling builders:

        * Dense inverses are MATERIALIZED at build time (cho_solve(eye),
          symmetrized, or the eigh pseudo-inverse) so the apply is
          contiguous GEMVs -- the §7.15 permuted-GEMV lesson.
        * Tile blocks OVERLAP on shared interface nodes (unlike BJ's
          disjoint ownership partition), so the apply is a scatter-ADD
          tilewise pass: structurally ``_tilewise_matvec_serial/_threaded``
          with ``B_i = D_i S~_i^+ D_i`` in place of ``S_i``.  The code
          duplication between those matvec methods and ``_nn_apply_*``
          below follows the existing matvec/matmat precedent in this file
          -- the hot matvec is measurement-frozen and not refactored here.
        * Island nodes (S_extra 1e5 penalty diagonal) are sliced OUT of
          every block before factoring -- keeping them would inject
          gratuitous near-null modes into ``S~_i`` (the same penalty-
          inflation trap the GenEO ``island_local_mask`` handles) -- and
          are served by the diagonal complement instead, whose full_diag
          INCLUDES the S_extra penalty (so ``M^-1 S ~ 1`` there).
        * Singular blocks (floating tiles / weakly-grounded port subsets)
          fall back to the SPD-safe eigenclip pseudo-inverse; the clipped
          tile-kernel directions are exactly what the PoU coarse space
          balances/deflates (BDD's "balancing" step) -- this base is
          intended to run under ``'two_level'``.
        * The byte guard mirrors ``_build_block_jacobi``'s: a MEMORY
          guard, not a numerics guard (§7.13); on breach the base degrades
          to 'jacobi' with a WARNING.

        Returns the LinearOperator, or degrades to
        ``_build_jacobi_fallback()`` (never returns None while a diagonal
        fallback is still buildable).
        """
        n = self.n
        if not self.tile_schur_complements or self.tile_index_maps is None:
            logger.warning(
                "Neumann base: tile Schur blocks/index maps unavailable "
                "(assembled-only construction); degrading %r -> 'jacobi' "
                "(diagonal) for this solve.",
                self.requested_preconditioner,
            )
            self.preconditioner = 'jacobi'
            self._bj_downgraded = True
            return self._build_jacobi_fallback()

        _max_bytes = (
            self.neumann_max_bytes if self.neumann_max_bytes is not None
            else NEUMANN_MAX_FACTOR_BYTES
        )
        itemsize = np.dtype(self.matvec_dtype).itemsize
        sizes = [
            len(self.tile_index_maps[tid])
            for tid in self.tile_schur_complements
        ]
        _max_k = max(sizes) if sizes else 0
        # Retained inverses (matvec_dtype) + the largest block's transient
        # fp64 build set (sub copy, cho_solve inverse, symmetrize temp,
        # reassignment target -- the same 3-extra-arrays accounting as the
        # BJ pre-flight, plus the sub copy the BJ path does not need).
        est_bytes = sum(k * k * itemsize for k in sizes)
        est_bytes += 4 * _max_k * _max_k * 8
        if est_bytes > _max_bytes:
            logger.warning(
                "Neumann base: estimated inverse memory %.1f GB exceeds "
                "budget %.1f GB (setting 'interface_neumann_max_bytes', "
                "n_interface=%d, T=%d tiles, max_block=%d). Downgrading "
                "requested preconditioner %r -> 'jacobi' (diagonal) for "
                "this solve -- expect MORE CG iterations. The NN inverses "
                "cost the same order as the retained tile Schur blocks "
                "themselves (Sum(n_p_i^2) * %d bytes); raise the budget if "
                "the host already holds the blocks comfortably.",
                est_bytes / 1024 ** 3, _max_bytes / 1024 ** 3, n,
                len(sizes), _max_k,
                self.requested_preconditioner, itemsize,
            )
            self.preconditioner = 'jacobi'
            self._bj_downgraded = True
            return self._build_jacobi_fallback()

        island = self._island_idx

        # ---- pass 1: stiffness totals / multiplicities over ALL tiles ----
        total_diag = np.zeros(n, dtype=np.float64)
        counts = np.zeros(n, dtype=np.float64)
        for tid, S_i in self.tile_schur_complements.items():
            idx = np.asarray(self.tile_index_maps[tid], dtype=np.int64)
            total_diag += np.bincount(
                idx,
                weights=np.ascontiguousarray(np.diag(S_i), dtype=np.float64),
                minlength=n,
            )
            counts += np.bincount(idx, minlength=n)
        # Diagonal for the complement term: the TRUE diagonal of S (tile
        # sum + S_extra's island-penalty/package contributions -- same S4
        # reasoning as _build_jacobi_fallback).
        full_diag = total_diag.copy()
        if self.S_extra is not None:
            full_diag += np.asarray(
                self.S_extra.tocsr().diagonal(), dtype=np.float64
            )
        full_diag = np.where(full_diag > 0, full_diag, 1.0)

        # ---- pass 2: factor each tile block, fold in the PoU weights -----
        covered = np.zeros(n, dtype=bool)
        nn_tiles: List[Tuple[np.ndarray, np.ndarray]] = []
        n_eigclip = 0
        n_clip_dirs = 0
        n_failed = 0
        for tid, S_i in self.tile_schur_complements.items():
            idx = np.asarray(self.tile_index_maps[tid], dtype=np.int64)
            keep_mask: Optional[np.ndarray] = None
            if island is not None:
                keep_mask = ~np.isin(idx, island)
                if keep_mask.all():
                    keep_mask = None
                else:
                    idx = idx[keep_mask]
            if len(idx) == 0:
                continue
            S_arr = np.asarray(S_i, dtype=np.float64)

            def _extract_sub() -> np.ndarray:
                if keep_mask is not None:
                    return np.ascontiguousarray(
                        S_arr[np.ix_(keep_mask, keep_mask)]
                    )
                return np.array(S_arr, dtype=np.float64, copy=True)

            sub = _extract_sub()
            diag_vals = np.ascontiguousarray(np.diag(sub))
            if self._neumann_reg > 0.0:
                # Relative Tikhonov shift: bounds the local-solve response
                # along the block's near-null cluster at ~1/reg (relative
                # to the diagonal scale) AND makes the Cholesky path
                # succeed on the numerically-singular split-regime blocks
                # (see DEFAULT_NEUMANN_REG's measured rationale).
                sub[np.diag_indices_from(sub)] += (
                    self._neumann_reg * diag_vals
                )
            Sinv: Optional[np.ndarray] = None
            try:
                cho = la.cho_factor(
                    sub, check_finite=False, overwrite_a=True,
                )
                # Pivot-ratio condition estimate: cond(S_i) ~ (d_max/d_min)^2
                # for Cholesky pivots d.  A PSD-singular block can pass
                # cho_factor with tiny positive pivots and yield a finite
                # but kernel-amplifying inverse (the §7.8 BJ pathology) --
                # route those to the eigclip pseudo-inverse below.
                _piv = np.abs(np.diag(cho[0]))
                if (
                    _piv.min() <= 0.0
                    or (_piv.min() / _piv.max()) ** 2 < NEUMANN_CHO_RCOND_MIN
                ):
                    Sinv = None
                else:
                    Sinv = la.cho_solve(
                        cho, np.eye(len(idx)), check_finite=False,
                    )
                    # cho_solve's explicitly-formed inverse is not exactly
                    # symmetric -- same Finding-9 fix as the BJ
                    # materialization loop: symmetrize so PCG's SPD-
                    # preconditioner assumption holds.
                    Sinv = 0.5 * (Sinv + Sinv.T)
                    if not np.isfinite(Sinv).all():
                        Sinv = None  # numerically-singular cho slipped through
            except (la.LinAlgError, ValueError):
                Sinv = None
            if Sinv is None:
                # overwrite_a=True destroyed `sub`; re-extract for eigh
                # (re-applying the same Tikhonov shift, if any).
                sub2 = _extract_sub()
                if self._neumann_reg > 0.0:
                    sub2[np.diag_indices_from(sub2)] += (
                        self._neumann_reg * diag_vals
                    )
                res = _spd_safe_pseudo_solve_factor_ex(
                    sub2, eps_rel=NEUMANN_EIGCLIP_EPS_REL,
                )
                if res is None:
                    n_failed += 1
                    logger.warning(
                        "Neumann base: tile %r block (%d rows) has no "
                        "positive spectrum; skipping it (its nodes fall "
                        "back to neighbor tiles / the diagonal "
                        "complement).",
                        tid, len(idx),
                    )
                    continue
                V, inv_w, _w = res
                Sinv = (V * inv_w) @ V.T
                n_eigclip += 1
                # Diagnostic for the coarse-space-enrichment decision: how
                # many directions per block sit below the clip (i.e. get
                # the maximal 1/(eps_rel*lambda_max) response).
                n_clip_dirs += int(
                    np.sum(_w < NEUMANN_EIGCLIP_EPS_REL * float(_w.max()))
                )
            if self._neumann_weight == 'stiffness':
                _t = total_diag[idx]
                w_loc = diag_vals / np.where(_t > 0.0, _t, 1.0)
            else:  # 'multiplicity'
                _c = counts[idx]
                w_loc = 1.0 / np.where(_c > 0.0, _c, 1.0)
            # B_i = D_i S~_i^+ D_i, folded in place (row + column scaling).
            Sinv *= w_loc[np.newaxis, :]
            Sinv *= w_loc[:, np.newaxis]
            B = np.ascontiguousarray(Sinv.astype(self.matvec_dtype, copy=False))
            nn_tiles.append((idx, B))
            covered[idx] = True

        if not nn_tiles:
            logger.warning(
                "Neumann base: every tile block failed to factor; "
                "degrading %r -> 'jacobi' (diagonal) for this solve.",
                self.requested_preconditioner,
            )
            self.preconditioner = 'jacobi'
            self._bj_downgraded = True
            return self._build_jacobi_fallback()

        # Exact-jacobi response on nodes NO tile covers (taps/package-only
        # unknowns, islands, failed-block-only nodes); zero on covered
        # nodes.  Keeps M SPD on all of R^n.
        self._nn_comp_scale = np.where(covered, 0.0, 1.0 / full_diag)

        # LPT partition + per-bin touched-index unions -- mirrors
        # _prepare_tilewise_matvec_state (costs ~ k^2 per GEMV).
        self._nn_n_bins = max(1, min(self.matvec_threads, len(nn_tiles)))
        self._nn_partition = _lpt_partition(
            [float(len(idx)) ** 2 for idx, _ in nn_tiles], self._nn_n_bins,
        )
        touched: List[np.ndarray] = []
        for t in range(self._nn_n_bins):
            part = self._nn_partition[t]
            if part:
                touched.append(
                    np.unique(np.concatenate([nn_tiles[i][0] for i in part]))
                )
            else:
                touched.append(np.empty(0, dtype=np.int64))
        self._nn_touched = touched
        self._nn_tiles = nn_tiles

        self._base_builder_label = 'neumann'
        logger.info(
            "Neumann base: %d tile inverses (%d via eigclip pseudo-inverse "
            "with %d total clipped directions, %d skipped), %.2f GB (%s), "
            "weight='%s', reg=%g, %d/%d nodes covered.",
            len(nn_tiles), n_eigclip, n_clip_dirs, n_failed,
            sum(B.nbytes for _, B in nn_tiles) / 1024 ** 3,
            np.dtype(self.matvec_dtype).name, self._neumann_weight,
            self._neumann_reg, int(covered.sum()), n,
        )
        return spla.LinearOperator(
            shape=(n, n), matvec=self._nn_apply, dtype=np.float64,
        )

    def _nn_apply(self, x: np.ndarray) -> np.ndarray:
        """Neumann-Neumann apply dispatch (serial vs threaded)."""
        if self.matvec_threads <= 1 or len(self._nn_tiles) < 2:
            return self._nn_apply_serial(x)
        return self._nn_apply_threaded(x)

    def _nn_apply_serial(self, x: np.ndarray) -> np.ndarray:
        """Serial NN apply: diagonal complement + per-tile gather / GEMV /
        bincount scatter-add -- mirrors ``_tilewise_matvec_serial`` (see
        ``_build_neumann``'s docstring for why the pattern is duplicated
        rather than shared with the measurement-frozen matvec)."""
        n = self.n
        dtype = self.matvec_dtype
        result = x * self._nn_comp_scale
        for idx, B in self._nn_tiles:
            x_local = x[idx]
            if dtype != np.float64:
                x_local = x_local.astype(dtype, copy=False)
            y_local = np.asarray(B @ x_local, dtype=np.float64)
            result += np.bincount(idx, weights=y_local, minlength=n)
        return result

    def _nn_apply_threaded(self, x: np.ndarray) -> np.ndarray:
        """Threaded NN apply: LPT partition + compact per-bin buffers --
        mirrors ``_tilewise_matvec_threaded``.  Tile blocks OVERLAP on
        shared nodes, so unlike the disjoint-slice BJ apply this needs the
        scatter-ADD design (compact touched-index unions, not disjoint
        contiguous slices)."""
        dtype = self.matvec_dtype
        tiles = self._nn_tiles
        part = self._nn_partition
        touched = self._nn_touched
        n_threads = self._nn_n_bins
        local_bufs: List[np.ndarray] = [None] * n_threads  # type: ignore[list-item]

        def work(t: int) -> None:
            u = touched[t]
            if len(u) == 0:
                local_bufs[t] = np.zeros(0, dtype=np.float64)
                return
            buf = np.zeros(len(u), dtype=np.float64)
            for i in part[t]:
                idx, B = tiles[i]
                x_local = x[idx]
                if dtype != np.float64:
                    x_local = x_local.astype(dtype, copy=False)
                y_local = np.asarray(B @ x_local, dtype=np.float64)
                pos = np.searchsorted(u, idx)
                buf += np.bincount(pos, weights=y_local, minlength=len(u))
            local_bufs[t] = buf

        pool = self._get_pool()
        with threadpoolctl.threadpool_limits(1):
            list(pool.map(work, range(n_threads)))

        result = x * self._nn_comp_scale
        for t in range(n_threads):
            u = touched[t]
            if len(u):
                result[u] += local_bufs[t]
        return result

    def _build_amg(self) -> Optional[spla.LinearOperator]:
        """AMG preconditioner via pyamg (optional dependency).

        Requires the assembled S_global (CSR format).  Returns None if
        pyamg is not installed or S_global is not available.
        """
        try:
            import pyamg  # type: ignore[import]
        except ImportError:
            logger.info(
                "pyamg not installed; falling back to block_jacobi preconditioner."
            )
            return self._build_block_jacobi()

        if self.S_global is None:
            logger.warning(
                "AMG preconditioner requires S_global (assembled mode); "
                "falling back to block_jacobi."
            )
            return self._build_block_jacobi()

        try:
            ml = pyamg.smoothed_aggregation_solver(self.S_global.tocsr())
            M_op = ml.aspreconditioner(cycle='V')
            return M_op
        except Exception as exc:
            logger.warning(
                "AMG setup failed (%s); falling back to block_jacobi.", exc
            )
            return self._build_block_jacobi()

    # ------------------------------------------------------------------
    # Solve (callable interface matching direct LU)
    # ------------------------------------------------------------------

    def __call__(self, rhs: np.ndarray) -> np.ndarray:
        """Solve S_global @ x = rhs via CG.

        Args:
            rhs: Right-hand-side vector, shape (n_interface,).

        Returns:
            Solution vector x, shape (n_interface,).

        Raises:
            RuntimeError: If CG fails to converge within maxiter.
        """
        t0 = time.perf_counter()

        x0 = self._x0

        iters: List[int] = [0]
        # Debug/observability knob (set attribute directly, e.g. from a
        # measurement script): log progress every `progress_every` CG
        # iterations, including the TRUE relative residual (costs one extra
        # matvec per report -- e.g. 0.5% overhead at progress_every=200).
        # <= 0 (0 is the default) disables.
        _progress_every = int(getattr(self, 'progress_every', 0) or 0)
        _rhs_norm = float(np.linalg.norm(rhs)) if _progress_every > 0 else 0.0

        def _callback(xk: Optional[np.ndarray]) -> None:
            # xk is None when called from _deflated_pcg's progress_every
            # gate on an iteration it decided not to log (see that
            # function's `progress_every` Args entry) -- guaranteed to line
            # up with the `iters[0] % _progress_every == 0` check below
            # (same modulus, same value), so xk is never None on an
            # iteration this branch actually enters.
            #
            # Finding 3 (A-DEF2 code review, round 1): gate on
            # `_progress_every > 0`, not plain truthiness -- a NEGATIVE
            # value is truthy in Python, and `k % -m == 0` is True whenever
            # k is a multiple of m, so the old `if _progress_every:` gate
            # entered this branch on a negative setting while
            # _deflated_pcg's own gate (below, `progress_every > 0`) still
            # delivered `xk=None` for every iteration (its "logging
            # disabled" contract) -- `self._linear_op.matvec(None)` then
            # raised TypeError, a deflated-only crash on a value the
            # sibling `reproject_every` knob documents as "<= 0 disables".
            iters[0] += 1
            if _progress_every > 0 and iters[0] % _progress_every == 0:
                rel = (float(np.linalg.norm(rhs - self._linear_op.matvec(xk)))
                       / max(_rhs_norm, 1e-300))
                logger.info(
                    "InterfaceCG progress: iter %d, true rel_res=%.3e, "
                    "elapsed=%.1fs (target rtol=%.1e)",
                    iters[0], rel, time.perf_counter() - t0, self.rtol,
                )

        # A-DEF2 work package: the hand-rolled deflated-PCG loop is used
        # ONLY when apply_mode='deflated' actually has a coarse space WITH
        # retained SZ to deflect against (see module docstring's "A-DEF2
        # work package" section) -- every other combination (additive
        # two_level, plain block_jacobi/jacobi/none/amg, or a two_level
        # request that degraded/never retained SZ) goes through the
        # unchanged scipy path, so this is zero risk to any pre-existing
        # mode (spec's "simplest correct" degradation choice).
        _use_deflated = (
            self._apply_mode == 'deflated'
            and self._coarse is not None
            and self._coarse.SZ is not None
        )
        if _use_deflated:
            result, info = _deflated_pcg(
                matvec=self._linear_op.matvec,
                base_apply=self._M_base_apply,
                coarse=self._coarse,
                b=rhs,
                x0=x0,
                rtol=self.rtol,
                atol=self.atol,
                maxiter=self.maxiter,
                reproject_every=self._deflated_reproject_every,
                callback=_callback,
                # Perf finding: without this, _deflated_pcg recovers the
                # full x (two O(n*T') dense GEMVs) every iteration solely
                # to feed a callback that only USES it once every
                # `_progress_every` iterations (0 = never, the default) --
                # gate the recovery on the same interval so the default
                # (progress logging off) skips it entirely.
                progress_every=_progress_every,
            )
        else:
            result, info = spla.cg(
                self._linear_op,
                rhs,
                x0=x0,
                rtol=self.rtol,
                atol=self.atol,
                maxiter=self.maxiter,
                M=self._M,
                callback=_callback,
            )

        elapsed = time.perf_counter() - t0
        n_iter = iters[0]

        if info < 0:
            raise RuntimeError(
                f"InterfaceCGSolver: CG failed with illegal input (info={info})."
            )

        # Accumulate stats before any possible raise so callers that catch the
        # RuntimeError can still inspect stats['cg_failed'] / 'total_cg_failures'.
        self._total_iters += n_iter
        self._total_solves += 1
        self._stats['last_cg_iters'] = n_iter
        self._stats['last_cg_time_s'] = elapsed
        self._stats['last_cg_info'] = info
        # Machine-readable disclosure of the algorithm actually dispatched
        # this call -- 'deflated' when apply_mode='deflated' genuinely took
        # effect, 'additive' for the Stage 3 two_level combination, or the
        # plain base preconditioner name otherwise. A measurement script
        # (e.g. the coordinator's mi200k/Gate-4 harness) can key off this
        # without parsing preconditioner_label's string.
        self._stats['apply_algorithm'] = (
            'deflated' if _use_deflated
            else ('additive' if self._coarse is not None else self.preconditioner)
        )
        self._stats.setdefault('total_cg_iters', 0)
        self._stats['total_cg_iters'] = self._total_iters
        self._stats.setdefault('total_cg_solves', 0)
        self._stats['total_cg_solves'] = self._total_solves

        _failed = info > 0
        self._stats['cg_failed'] = _failed
        self._stats.setdefault('total_cg_failures', 0)

        rel_res: Optional[float] = None
        if info > 0:
            # Compute actual relative residual so the error message is informative.
            r_norm = float(np.linalg.norm(rhs - self._linear_op.matvec(result)))
            b_norm = float(np.linalg.norm(rhs))
            rel_res = r_norm / b_norm if b_norm > 0 else r_norm
            self._stats['last_cg_rel_residual'] = rel_res
            self._stats['total_cg_failures'] += 1

        # Warm-start: update x0 for next call (plain previous-solution seed,
        # or the linear-extrapolation seed when warm_start_extrapolation is
        # enabled -- see push_solution_history).
        #
        # Finding 2 (A-DEF2 code review, round 1): only feed a CONVERGED
        # solution (info == 0) into the extrapolation history. Pushing a
        # non-converged iterate (info != 0, reachable with strict=False)
        # would have its error linearly AMPLIFIED by the next step's
        # ``2*x_prev - x_prev2`` seed instead of merely being reused
        # as-is -- on a transient run that repeatedly hits maxiter this
        # compounds step over step, producing progressively worse warm
        # starts. On a failed solve: seed the NEXT call with the plain
        # best iterate (never worse than the pre-extrapolation behaviour)
        # and CLEAR the two-point history so a later converged solve does
        # not extrapolate across the gap using the bad iterate as one of
        # its two history entries.
        #
        # Round-2 code review finding 8 (PLAUSIBLE, confirmed): this
        # hygiene MUST run BEFORE the strict-mode raise below, not after
        # it -- a prior revision of this method ran the raise first, so
        # with strict=True (the production default) a failed solve's
        # RuntimeError propagated out of __call__ WITHOUT ever clearing
        # ``_x_hist_prev``/``_x_hist_prev2`` or reseeding ``_x0`` from the
        # best iterate. A caller that catches that RuntimeError and
        # retries (the documented recovery path short of strict=False)
        # would then re-solve from the SAME pre-failure ``_x0``/history
        # that just failed -- and if a LATER solve converges, it would
        # extrapolate ``2*x_prev - x_prev2`` across the failed step using
        # stale pre-failure history, exactly the across-the-gap
        # extrapolation this hygiene exists to prevent. Reordering costs
        # nothing (the raise below still fires with the same message/stats
        # either way) and makes the hygiene hold unconditionally.
        #
        # Round-3 code review finding 3 (CONFIRMED): a ``bnrm2 == 0`` RHS
        # is a special-cased immediate return (both the scipy path and
        # ``_deflated_pcg`` short-circuit to ``x=0, info=0`` WITHOUT
        # running a single iteration -- see their own ``bnrm2 == 0``
        # branches) -- it is not a genuine converged SOLUTION of a
        # "family" the extrapolation history should track. Pushing it in
        # would seed the NEXT solve with ``2*0 - x_prev == -x_prev`` (via
        # ``push_solution_history``), a seed reliably WORSE than a cold
        # start. Skip the push entirely for this case -- leave ``_x0``/
        # ``_x_hist_prev``/``_x_hist_prev2`` exactly as they were before
        # this call, so the NEXT solve warm-starts from whatever the state
        # was prior to the zero-RHS solve (unaffected by it), matching the
        # intuition that "solve nothing" should be a no-op for warm-start
        # purposes rather than actively corrupting it.
        _bnrm2 = float(np.linalg.norm(rhs))
        if info == 0 and _bnrm2 != 0.0:
            self.push_solution_history(result)
        elif info != 0:
            self._x_hist_prev = None
            self._x_hist_prev2 = None
            self._x0 = result.copy()

        if info > 0:
            if self.strict:
                raise RuntimeError(
                    f"InterfaceCGSolver: CG did not converge after {n_iter} "
                    f"iterations (rtol={self.rtol:.2e}, atol={self.atol:.2e}, "
                    f"rel_residual={rel_res:.3e}). "
                    f"Increase maxiter, relax tolerances, or improve the "
                    f"preconditioner. Set strict=False to demote to a warning."
                )
            else:
                logger.warning(
                    "InterfaceCGSolver: CG did not converge after %d iterations "
                    "(rtol=%.2e, rel_residual=%.3e). "
                    "Using best iterate (strict=False).",
                    n_iter, self.rtol, rel_res,
                )

        logger.debug(
            "InterfaceCGSolver: %d iters, info=%d, %.4fs (warm=%s)",
            n_iter, info, elapsed, x0 is not None,
        )

        return result

    # ------------------------------------------------------------------
    # Warm-start control
    # ------------------------------------------------------------------

    def reset_warm_start(self) -> None:
        """Clear the warm-start initial guess (force cold start next call),
        including the linear-extrapolation history (see
        :meth:`push_solution_history`) -- the next call after a reset gets
        neither the plain previous solution nor an extrapolated seed."""
        self._x0 = None
        self._x_hist_prev = None
        self._x_hist_prev2 = None

    def set_x0(self, x0: Optional[np.ndarray]) -> None:
        """Explicitly set the warm-start guess for the next solve call."""
        self._x0 = x0 if x0 is None else np.asarray(x0, dtype=np.float64).copy()

    def push_solution_history(self, x: np.ndarray) -> None:
        """Update the warm-start seed for the NEXT solve from a just-computed
        solution ``x``.

        When ``warm_start_extrapolation`` is False (default), this is
        byte-identical to the pre-A-DEF2-work-package behaviour: the seed is
        simply ``x`` itself. When True, seeds with the linear extrapolation
        ``2*x_prev - x_prev2`` of the last two solutions (falls back to
        ``x_prev`` -- i.e. the plain previous-solution seed -- until two
        solves have been recorded) -- anticipates a slowly-varying
        transient RHS's next solution rather than assuming it equals the
        current one. Called automatically at the end of every CONVERGED
        :meth:`__call__` (Finding 2, round 1: a non-converged solve
        instead clears the history and seeds plainly -- see the call
        site); exposed as a public method so a caller with its own solve
        loop (bypassing ``__call__``) can still opt in.
        """
        x = np.asarray(x, dtype=np.float64)
        if not self._warm_start_extrapolation:
            self._x0 = x.copy()
            return
        if self._x_hist_prev is not None:
            self._x_hist_prev2 = self._x_hist_prev
        self._x_hist_prev = x.copy()
        if self._x_hist_prev2 is not None:
            self._x0 = 2.0 * self._x_hist_prev - self._x_hist_prev2
        else:
            self._x0 = self._x_hist_prev.copy()

    # ------------------------------------------------------------------
    # Stats access
    # ------------------------------------------------------------------

    @property
    def stats(self) -> Dict[str, Any]:
        """Read-only view of accumulated stats (same dict as stats_dict)."""
        return self._stats

    @property
    def total_iterations(self) -> int:
        """Total CG iterations across all calls."""
        return self._total_iters

    @property
    def total_solves(self) -> int:
        """Total solve calls."""
        return self._total_solves


# Round-3 code review finding 11: the hand-rolled A-DEF2 deflated-PCG
# machinery (_deflated_pcg, _is_breakdown, _BREAKDOWN_EPS,
# DEFLATED_DEBOUNCE_REARM_FALLBACK_ITERS) moved to interface_deflated_pcg.py
# (pure mechanical move, zero logic change) -- imported above and
# re-exported under their original names so every existing call site and
# test import keeps working unchanged. See that module's docstring for
# the full algorithm/dependency record.


# ---------------------------------------------------------------------------
# Auto-select helper (used by result_factorization)
# ---------------------------------------------------------------------------


def auto_select_interface_solver(
    n_interface: int,
    S_global: Optional[sp.spmatrix] = None,
    n_interface_threshold: int = AUTO_CG_N_INTERFACE_THRESHOLD,
    factor_memory_budget_bytes: int = AUTO_CG_FACTOR_MEMORY_BUDGET_BYTES,
) -> str:
    """Decide whether to use 'direct' or 'cg' for the interface solve.

    Rules:
      1. If n_interface < n_interface_threshold AND estimated factor memory
         fits within factor_memory_budget_bytes: return 'direct'.
      2. Otherwise: return 'cg'.

    The factor memory estimate uses S_global.nnz * 8 * factor_fill_ratio
    (empirical fill ratio 5 for supernodal CHOLMOD on PDN-class matrices).

    Args:
        n_interface: Number of interface unknowns.
        S_global: Assembled Schur matrix (optional; used for memory estimate).
        n_interface_threshold: Cutover size (default 200_000).
        factor_memory_budget_bytes: Memory budget in bytes (default 32 GB).

    Returns:
        'direct' or 'cg'.
    """
    # Finding 4 (SKIPPED -- host-RAM-dependent auto selection is intended
    # Stage 1c behaviour) is mitigated by always logging the resolved
    # budget and resulting mode at INFO level, regardless of which branch
    # is taken, so a host-dependent flip from 'direct' to 'cg' (or vice
    # versa) is visible in the log rather than silent.
    if n_interface < n_interface_threshold:
        if S_global is not None:
            # Rough factor memory: nnz * bytes_per_double * fill_ratio
            fill_ratio = 5.0
            est_bytes = S_global.nnz * 8 * fill_ratio
            if est_bytes > factor_memory_budget_bytes:
                logger.info(
                    "auto_select: n_interface=%d < threshold but estimated "
                    "factor memory %.1f GB > budget %d bytes (%.1f GB); "
                    "resolved_mode=cg.",
                    n_interface,
                    est_bytes / 1024 ** 3,
                    factor_memory_budget_bytes,
                    factor_memory_budget_bytes / 1024 ** 3,
                )
                return 'cg'
            logger.info(
                "auto_select: n_interface=%d < threshold %d, estimated "
                "factor memory %.1f GB fits budget %d bytes (%.1f GB); "
                "resolved_mode=direct.",
                n_interface, n_interface_threshold,
                est_bytes / 1024 ** 3,
                factor_memory_budget_bytes,
                factor_memory_budget_bytes / 1024 ** 3,
            )
        else:
            logger.info(
                "auto_select: n_interface=%d < threshold %d (no S_global "
                "for memory estimate); resolved_mode=direct.",
                n_interface, n_interface_threshold,
            )
        return 'direct'

    logger.info(
        "auto_select: n_interface=%d >= threshold %d; budget=%d bytes "
        "(%.1f GB); resolved_mode=cg.",
        n_interface, n_interface_threshold,
        factor_memory_budget_bytes, factor_memory_budget_bytes / 1024 ** 3,
    )
    return 'cg'


# ---------------------------------------------------------------------------
# Factory: build an interface_lu callable from CG or direct
# ---------------------------------------------------------------------------


def build_interface_solver(
    S_global: Optional[sp.spmatrix],
    interface_solver: str = 'auto',
    tile_schur_complements: Optional[Dict[Any, np.ndarray]] = None,
    tile_index_maps: Optional[Dict[Any, np.ndarray]] = None,
    S_extra: Optional[sp.spmatrix] = None,
    matvec_mode: str = 'auto',
    preconditioner: str = 'block_jacobi',
    rtol: float = 1e-8,
    atol: float = 1e-14,
    maxiter: Optional[int] = None,
    strict: bool = True,
    x0: Optional[np.ndarray] = None,
    block_jacobi_max_bytes: Optional[int] = None,
    # NN/BDD work package: forwarded as-is to InterfaceCGSolver (same
    # None-sentinel pattern as the coarse knobs below).
    two_level_base: Optional[str] = None,
    neumann_max_bytes: Optional[int] = None,
    neumann_weight: Optional[str] = None,
    neumann_reg: Optional[float] = None,
    factor_memory_budget_bytes: Optional[int] = None,
    n_interface_threshold: int = AUTO_CG_N_INTERFACE_THRESHOLD,
    coordinator_solver_config: Optional[Any] = None,
    verbose: bool = False,
    cg_stats_dict: Optional[Dict[str, Any]] = None,
    n_interface: Optional[int] = None,
    matvec_threads: Any = 'auto',
    matvec_dtype: Any = 'float64',
    strict_dtype_rtol: bool = True,
    island_idx: Optional[np.ndarray] = None,
    # Finding 9: None-sentinel defaults (NOT interface_coarse.DEFAULT_*
    # bound at def/import time) -- forwarded as-is to InterfaceCGSolver,
    # which resolves them dynamically (see its constructor's Finding 9
    # comment). A def-time snapshot here would silently defeat
    # monkeypatch.setattr(interface_coarse, 'DEFAULT_GENEO_K', ...) even
    # after InterfaceCGSolver itself was fixed, since this factory's own
    # default would already be a concrete (stale) int/float by the time
    # InterfaceCGSolver.__init__ ever sees it.
    interface_coarse_geneo_k: Optional[int] = None,
    interface_coarse_geneo_tol: Optional[float] = None,
    interface_coarse_eps_rank: Optional[float] = None,
    interface_coarse_max_cols: Optional[int] = None,
    interface_coarse_max_bytes: Optional[int] = None,
    # A-DEF2 work package: same None-sentinel forwarding pattern as the
    # Stage 3 coarse knobs above -- including warm_start_extrapolation
    # (round-2 code review finding 5: a prior revision of this signature
    # bound it to DEFAULT_WARM_START_EXTRAPOLATION at def time instead,
    # same class of bug as a def-time-bound interface_coarse_geneo_k
    # default would be). None is forwarded as-is to InterfaceCGSolver,
    # which resolves it dynamically -- never resolved here, for the same
    # reason the four coarse knobs above are not.
    interface_coarse_apply_mode: Optional[str] = None,
    interface_deflated_reproject_every: Optional[int] = None,
    warm_start_extrapolation: Optional[bool] = None,
) -> Tuple[Callable, str, Optional['InterfaceCGSolver']]:
    """Build an interface solve callable (direct LU or CG).

    Stage 1d: this is the single entry point for constructing an
    ``InterfaceCGSolver`` (or a direct factor).  ``_factor_dc_context``,
    ``_factor_transient_context``, ``_refactor_dc_context``, and
    ``_refactor_transient_context`` (all in ``result_factorization.py``) are
    routed through this factory so there is exactly one place that builds
    the CG solve callable.

    Args:
        S_global: Assembled sparse Schur matrix (CSR or CSC).  May be
            ``None`` ONLY when ``interface_solver='cg'`` (not 'auto' or
            'direct' -- both need S_global) and ``matvec_mode`` resolves to
            'tilewise' with ``tile_schur_complements`` provided -- the
            "never assemble S_global" path (item 3).  ``n_interface`` must
            be passed explicitly in that case.
        interface_solver: 'direct', 'cg', or 'auto'.
        tile_schur_complements: Per-tile dense Schur blocks (for tilewise).
        tile_index_maps: Per-tile index maps (for tilewise and block_jacobi
            ownership).
        S_extra: Package-edge contribution not captured by per-tile Schur
            complements (tilewise mode only; ignored otherwise).
        matvec_mode: 'auto' (default; item 8 -- 'tilewise' when
            ``tile_schur_complements`` is provided, else 'assembled'),
            'assembled', or 'tilewise' (only used when cg selected).
        preconditioner: 'auto' (Stage 3 -- resolves to 'two_level' when the
            resolved CG matvec mode is 'tilewise', else 'block_jacobi'; see
            :func:`resolve_preconditioner`, the ONE place this resolution
            happens), 'block_jacobi', 'jacobi', 'none', 'amg', or
            'two_level'.
        rtol: CG relative convergence tolerance (default 1e-8).
        atol: CG absolute convergence tolerance floor (default 1e-14).
        maxiter: Max CG iterations (default None -> 3 * n_interface).
        strict: Raise on CG non-convergence (default True).
        x0: Optional initial warm-start guess.
        block_jacobi_max_bytes: Resolved block-Jacobi memory budget (bytes);
            None falls back to the module-level constant.
        factor_memory_budget_bytes: Direct-factor memory budget (bytes),
            used only when ``interface_solver='auto'``.  May be a resolved
            byte count, ``'auto'``, or ``None`` -- all are routed through
            ``resolve_factor_memory_budget_bytes()`` so the factory's own
            'auto' branch is host-aware (matches the host-aware resolution
            already performed by callers), not the raw module-level
            ``AUTO_CG_FACTOR_MEMORY_BUDGET_BYTES`` constant.
        n_interface_threshold: Cutover size used only when
            ``interface_solver='auto'``.
        coordinator_solver_config: SolverBackendConfig (for direct path).
        verbose: Log the resolved backend.
        cg_stats_dict: Mutable dict for CG statistics.
        n_interface: Interface system size.  Required when ``S_global`` is
            ``None`` (the "never assemble" path); otherwise inferred from
            ``S_global.shape[0]`` and used only as a consistency check.
        matvec_threads: Stage 2 ``InterfaceCGSolver`` thread count ('auto'
            default -- see :func:`resolve_matvec_threads`).
        matvec_dtype: Stage 2 tilewise storage dtype ('float64' default).
        strict_dtype_rtol: Enforce the fp32-matvec/rtol pairing (default
            True) -- see :class:`InterfaceCGSolver`.
        island_idx: Stage 3: global interface indices of interface-island
            nodes, for the 'two_level' preconditioner's coarse-space row
            zeroing (ignored otherwise).
        interface_coarse_geneo_k, interface_coarse_geneo_tol,
        interface_coarse_eps_rank, interface_coarse_max_cols,
        interface_coarse_max_bytes: Stage 3 'two_level' knobs -- see
            :class:`InterfaceCGSolver`.
        interface_coarse_apply_mode, interface_deflated_reproject_every,
        warm_start_extrapolation: A-DEF2 work package knobs -- see
            :class:`InterfaceCGSolver`.

    Returns:
        (solve_callable, resolved_mode, cg_solver_or_none)
        resolved_mode is 'direct' or 'cg'.
        cg_solver_or_none is the InterfaceCGSolver when cg is used, else None.
    """
    if S_global is not None:
        n = S_global.shape[0]
        if n_interface is not None and n_interface != n:
            raise ValueError(
                f"build_interface_solver: n_interface={n_interface} != "
                f"S_global.shape[0]={n}."
            )
    else:
        if n_interface is None:
            raise ValueError(
                "build_interface_solver: n_interface must be provided "
                "explicitly when S_global is None."
            )
        n = n_interface
        if interface_solver != 'cg':
            raise ValueError(
                f"build_interface_solver: S_global is None, which is only "
                f"supported for interface_solver='cg' (got "
                f"{interface_solver!r}) -- 'direct' and 'auto' both need "
                f"S_global for the direct-factor memory estimate/factor."
            )

    # Resolve 'auto'.  Route through resolve_factor_memory_budget_bytes()
    # (host-aware: min(32 GB, 0.4*RAM) via psutil) rather than the raw
    # AUTO_CG_FACTOR_MEMORY_BUDGET_BYTES constant, so the factory's own
    # 'auto' branch behaves identically to callers that pre-resolve the
    # budget externally (finding 12).  factor_memory_budget_bytes may
    # already be a resolved int (idempotent through this call) or an
    # unresolved 'auto'/None passthrough.
    if interface_solver == 'auto':
        _budget = resolve_factor_memory_budget_bytes(
            factor_memory_budget_bytes
            if factor_memory_budget_bytes is not None
            else 'auto'
        )
        resolved = auto_select_interface_solver(
            n, S_global,
            n_interface_threshold=n_interface_threshold,
            factor_memory_budget_bytes=_budget,
        )
    else:
        resolved = interface_solver

    if resolved == 'direct':
        from pgmath.factor import _factor_conductance_matrix
        t0 = time.perf_counter()
        result = _factor_conductance_matrix(
            S_global, verbose=verbose, config=coordinator_solver_config,
        )
        elapsed = time.perf_counter() - t0
        if verbose:
            logger.info(
                "Interface direct factor: %s, %.3fs, n=%d",
                result.backend_info, elapsed, n,
            )
        return result.solve, 'direct', None

    # CG path
    if cg_stats_dict is None:
        cg_stats_dict = {}

    # Item 8: resolve 'auto' matvec_mode -- 'tilewise' when tile Schur
    # blocks are available, else 'assembled'.  Explicit values pass through
    # resolve_matvec_mode() unchanged.
    _has_tile_blocks = bool(tile_schur_complements) and tile_index_maps is not None
    matvec_mode = resolve_matvec_mode(matvec_mode, _has_tile_blocks)

    # For tilewise mode: require tile_schur_complements + tile_index_maps
    # (unless S_global is None, in which case 'assembled' is impossible --
    # this can only happen via a caller bug, since the "never assemble"
    # path always provides tile blocks; raise instead of silently building
    # a solver with no matvec at all).
    if matvec_mode == 'tilewise' and not _has_tile_blocks:
        if S_global is None:
            raise ValueError(
                "build_interface_solver: S_global is None and "
                "tile_schur_complements/tile_index_maps were not provided "
                "-- cannot build any matvec."
            )
        logger.warning(
            "build_interface_solver: matvec_mode='tilewise' requested but "
            "tile_schur_complements or tile_index_maps not provided; "
            "falling back to 'assembled' mode."
        )
        matvec_mode = 'assembled'
    if matvec_mode == 'assembled' and S_global is None:
        raise ValueError(
            "build_interface_solver: matvec_mode resolved to 'assembled' "
            "but S_global is None."
        )

    # Stage 3: the ONE place 'auto' preconditioner resolution happens --
    # 'two_level' when CG + tilewise, else the legacy 'block_jacobi' default.
    # An explicit (non-'auto') caller value passes through unchanged.
    preconditioner = resolve_preconditioner(preconditioner, resolved, matvec_mode)

    t0 = time.perf_counter()
    cg_solver = InterfaceCGSolver(
        n_interface=n,
        matvec_mode=matvec_mode,
        S_global=S_global,
        tile_schur_complements=tile_schur_complements,
        tile_index_maps=tile_index_maps,
        S_extra=S_extra,
        preconditioner=preconditioner,
        rtol=rtol,
        atol=atol,
        maxiter=maxiter,
        x0=x0,
        stats_dict=cg_stats_dict,
        strict=strict,
        block_jacobi_max_bytes=block_jacobi_max_bytes,
        two_level_base=two_level_base,
        neumann_max_bytes=neumann_max_bytes,
        neumann_weight=neumann_weight,
        neumann_reg=neumann_reg,
        matvec_threads=matvec_threads,
        matvec_dtype=matvec_dtype,
        strict_dtype_rtol=strict_dtype_rtol,
        island_idx=island_idx,
        interface_coarse_geneo_k=interface_coarse_geneo_k,
        interface_coarse_geneo_tol=interface_coarse_geneo_tol,
        interface_coarse_eps_rank=interface_coarse_eps_rank,
        interface_coarse_max_cols=interface_coarse_max_cols,
        interface_coarse_max_bytes=interface_coarse_max_bytes,
        interface_coarse_apply_mode=interface_coarse_apply_mode,
        interface_deflated_reproject_every=interface_deflated_reproject_every,
        warm_start_extrapolation=warm_start_extrapolation,
    )
    elapsed = time.perf_counter() - t0
    if verbose:
        logger.info(
            "Interface CG solver built: mode=%s, precond=%s, rtol=%.2e, n=%d, "
            "threads=%d, dtype=%s, build_time=%.3fs",
            matvec_mode, cg_solver.preconditioner_label, rtol, n,
            cg_solver.matvec_threads, cg_solver.matvec_dtype, elapsed,
        )

    return cg_solver, 'cg', cg_solver

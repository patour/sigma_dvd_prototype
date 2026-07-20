"""A-DEF2 hand-rolled deflated PCG -- module-level, dependency-free of
:class:`~distributed.interface_iterative.InterfaceCGSolver`.

Round-3 code review finding 11 (CONFIRMED): this machinery (``_deflated_pcg``,
``_is_breakdown``, ``_BREAKDOWN_EPS``, ``DEFLATED_DEBOUNCE_REARM_FALLBACK_
ITERS``) was a pure function of its own arguments from the day it was added
(per its own docstring: "module-level ... independently unit-testable
against a small dense system") but lived inline in ``interface_iterative.py``,
which had grown to ~4.8x the repo's ~800-line-per-file convention for
``src/distributed/`` (see root ``CLAUDE.md``, "Files use a mixin pattern to
keep each under ~800 lines"). Split out here as a PURE MECHANICAL MOVE (zero
logic change -- diffed against the pre-move code) once the deflated-PCG
numerics stabilized (three code-review rounds with no further algorithmic
findings against this block, only the accompanying cleanup/hygiene fixes
that are already reflected in the moved code below).

Dependencies actually referenced by this module (verified before moving, to
avoid a spurious circular import): ``numpy``, and
:mod:`distributed.interface_coarse` for exactly one read
(``interface_coarse.DEFAULT_DEFLATED_REPROJECT_EVERY``, the dynamic default
for ``reproject_every=None`` -- same "read the canonical default at call
time, not at def time" pattern the rest of the coarse-space knobs use).
``interface_coarse.py`` does not import this module (or
``interface_iterative.py``) at module level, so this import direction is
safe. Everything else this module needs (``matvec``, ``base_apply``,
``coarse``) is passed in by the caller as plain callables/a duck-typed
``CoarseSpace`` object -- ``interface_iterative.py`` imports ``_deflated_pcg``
etc. back from here (see its own import block) so existing call sites and
test imports (``from distributed.interface_iterative import _deflated_pcg,
_is_breakdown, _BREAKDOWN_EPS``) keep working unchanged.

See ``interface_deflation_notes.md`` for the full A-DEF2/DEF algorithm
selection record (why DEF, not the literal A-DEF2 formula) and
``interface_iterative.py``'s own "A-DEF2" docstring section for the
preconditioner-label/settings-wiring context this module plugs into.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np

from . import interface_coarse

# Round-2 code review finding 3: fallback re-arm period (in CG iterations)
# for _deflated_pcg's debounced fresh-true-residual acceptance check when
# ``reproject_every`` is disabled (<= 0).  When reprojection IS enabled, its
# own interval doubles as the re-arm period (both exist to bound how long a
# tracked-vs-true-residual disagreement can persist unexamined -- reusing one
# knob for both avoids introducing a redundant setting).  10 is a small
# constant chosen to keep the worst-case extra-matvec overhead low (one
# fresh accept-check every 10 iterations, ~10% overhead) while still
# guaranteeing prompt termination once the true residual actually meets
# tolerance -- see _deflated_pcg's debounce/re-arm docstring paragraph.
DEFLATED_DEBOUNCE_REARM_FALLBACK_ITERS: int = 10


# ---------------------------------------------------------------------------
# A-DEF2 hand-rolled deflated PCG (module-level: pure function of its
# arguments, independently unit-testable against a small dense system).
# ---------------------------------------------------------------------------


# Finding 10 (A-DEF2 code review, round 1): magnitude/sign-aware breakdown
# guard for the CG scalars ``rho`` (= r^T z, z = M_base^-1 r with M_base
# SPD) and ``pw`` (= p^T (P S p)) inside ``_deflated_pcg`` -- both should be
# strictly positive for a (near-)PSD operator, so exact float equality to
# 0.0 is not a reliable breakdown test: an ill-conditioned/penalty-heavy
# system lands NEAR, not AT, zero (occasionally slightly NEGATIVE from FP
# noise), and dividing by a tiny-but-nonzero value overflows or flips sign
# instead of triggering the restart path exactly the way an exact zero
# would have. ``eps * scale`` (``scale`` = ``norm(a) * norm(b)`` for a dot
# product ``a . b``) is the natural noise floor below which a value is
# indistinguishable from rounding error. Module-level (not a closure
# inside ``_deflated_pcg``) so it is independently unit-testable.
_BREAKDOWN_EPS: float = float(np.finfo(np.float64).eps)


def _is_breakdown(value: float, scale: float) -> bool:
    """``True`` iff ``value`` is at or below the FP noise floor
    ``_BREAKDOWN_EPS * scale`` -- treated as a CG breakdown (degenerate
    search direction / degenerate preconditioned-residual dot product)
    rather than requiring exact equality to ``0.0`` (see the module-level
    comment above for why). ``scale`` should be the natural magnitude
    scale of the dot product being tested (``norm(a) * norm(b)`` for
    ``value = a . b``); a non-positive ``value`` is always a breakdown
    regardless of ``scale`` (the operator this guards is expected to be
    (near-)PSD, so a non-positive value is itself already outside the
    expected sign)."""
    return value <= _BREAKDOWN_EPS * scale


def _deflated_pcg(
    matvec: Callable[[np.ndarray], np.ndarray],
    base_apply: Callable[[np.ndarray], np.ndarray],
    coarse: 'interface_coarse.CoarseSpace',
    b: np.ndarray,
    x0: Optional[np.ndarray],
    rtol: float,
    atol: float,
    maxiter: int,
    reproject_every: Optional[int] = None,
    callback: Optional[Callable[[Optional[np.ndarray]], None]] = None,
    progress_every: Optional[int] = None,
) -> Tuple[np.ndarray, int]:
    r"""A-DEF2 deflated preconditioned CG on ``S x = b``.

    ``Q = Z S_c^+ Z^T`` (never materialized -- every application goes
    through ``coarse.apply``/``coarse.apply_with_SQ``), ``P = I - S Q``
    (idempotent: ``P^2 = P``; ``Z^T P v = 0`` identically for ANY ``v`` --
    the "deflation invariant" this whole algorithm hinges on).

    **Two formulas were implemented and empirically REJECTED before this
    one** (motivated by Tang, Nabben, Vuik & Erlangga 2009's taxonomy, which
    did not survive contact with a real PDN's ``S_extra`` structure) --
    both applied ``P`` around a PLAIN un-projected matvec instead of
    projecting the matvec itself: the spec's literal formula STALLS on the
    real ``netlist_multi_tile`` fixture, and applying ``P`` after
    ``M_base^-1`` instead SILENTLY STAGNATES on the same fixture (a
    dangerous false-negative: ``info=0`` with the tracked residual
    satisfied while the true residual never approaches rtol). Full
    numbers and root-cause analysis: this module's docstring ("Why DEF,
    not the literal formula") and ``interface_deflation_notes.md``.

    **What IS implemented below** -- the deflation/"DEF" member of the same
    taxonomy: the matvec ITSELF is projected (``P (S p) = S p - S Q(S p)``,
    one extra ``apply_with_SQ`` call on ``S p`` -- which is already computed
    for the standard CG recurrence, so this is CHEAPER per iteration than
    either formula above, not more expensive), the preconditioner apply is
    the PLAIN base (``z = M_base^-1 r``, no ``Q`` term at all -- the
    deflation is carried entirely by the projected matvec), and ``x`` is
    recovered from the CG iterate ``y`` via ``x = y + Q(b - S y)`` rather
    than accumulated directly (accumulating ``x += alpha * p`` against a
    PROJECTED ``w`` is exactly the "r no longer equals b - S x" bug the
    rejected formulas above avoided by using an un-projected matvec, and
    exactly the bug this design avoids by never conflating ``y`` --  the
    solution of the projected system ``(P S) y = P b`` -- with ``x``).
    Because ``Z^T (P v) = 0`` identically for the projected matvec, ``Z^T
    r_k = 0`` is now maintained BY CONSTRUCTION at every iteration (an
    algebraic identity, not a hope) -- re-projection (below) is therefore
    genuine floating-point hygiene, exactly as originally intended, rather
    than a correctness necessity. Verified: reproduces ``x_direct``/direct-
    solve references to machine/near-machine precision on the
    ``netlist_multi_tile`` real PDN fixture, the chain fixture at every
    tested length, the disjoint-near-null-block GenEO fixture, and the
    penalized-island fixture -- all with the TRUE residual (``b - S x``
    recomputed via a fresh matvec) checked explicitly, not merely the
    tracked recurrence quantity (the mechanism that let the rejected
    formula #2 above silently pass a naive residual check while actually
    wrong).

    As a matrix this operator is still not literally symmetric (the
    combination of a projected matvec with a plain preconditioner is not
    expressible as a single symmetric ``M^-1``), so it still cannot be
    passed as ``scipy.sparse.linalg.cg``'s ``M=`` parameter -- hence the
    hand-rolled loop. Contract otherwise mirrors
    ``scipy.sparse.linalg.cg`` exactly: same stopping criterion, same
    ``bnrm2 == 0`` early exit, same "loop exhausted without converging" ->
    ``info=maxiter``, same ``(x, info)`` return shape, so callers share all
    downstream stats/strict-mode/warm-start handling with the scipy-backed
    additive path (see ``InterfaceCGSolver.__call__``).

    Per-iteration cost: one full ``matvec`` (``S p`` -- same as any CG),
    one ``apply_with_SQ`` (projects that matvec), one ``base_apply``, and
    one cheap ``coarse.apply`` (for the incrementally-tracked recovered
    ``x`` the ``callback`` receives -- ``S y`` is tracked incrementally
    from the already-computed ``S p``, so this recovery needs no extra
    full-``n`` matvec). Every acceptance attempt (see ``_try_accept``
    below, debounced/re-armed -- module docstring's "True-residual
    acceptance" bullet) pays two EXTRA full ``matvec`` calls (a fresh
    ``Sy = matvec(y)`` plus the ``matvec(x_candidate)`` residual check)
    on top of the per-iteration cost above -- bounded, not per-iteration.

    Re-projection (every ``reproject_every`` iterations, ``<= 0``
    disables): recomputes ``r = P(b - S y)`` from scratch (rather than
    trusting the incrementally-updated ``r``) to correct floating-point
    drift in ``Z^T r -> 0`` accumulated over many iterations -- pure
    hygiene now (see the class-level "Because Z^T (P v) = 0..." note
    above), verified to make no difference to whether a given fixture
    converges (only, potentially, to how many iterations a VERY long
    solve needs) -- reproject_every=0 (disabled) still converges
    correctly on every fixture this module's tests exercise. Deliberately
    does NOT also refresh the recurrence's incrementally-tracked ``Sy``
    (round-3 code review finding 4 -- see ``_try_accept``'s docstring for
    where that refresh actually lives and why): an earlier revision
    overwrote the SHARED ``Sy`` here too, and while that is a no-op on
    every synthetic fixture (``matvec`` = identity, so ``matvec(y) ==``
    the incrementally-tracked ``Sy`` to machine precision), on a REAL,
    very ill-conditioned tilewise-Schur fixture (cond ~1e12) it perturbs
    the ongoing CG recurrence's conjugacy just enough, every
    ``reproject_every`` iterations, to measurably slow convergence -- see
    ``TestTrueADef2SelectionRecord::test_true_adef2_fails_to_converge_
    on_realistic_ratio_fixture`` (regression discovered verifying this
    finding, not by the review). The drift ``_try_accept`` actually needs
    repaired only matters at the ACCEPTANCE decision, not the ongoing
    recurrence -- refreshing it there instead gets the repair with zero
    effect on convergence speed.

    Args:
        matvec: The ORIGINAL system's matvec (``S @ x``) -- the SAME
            ``InterfaceCGSolver._linear_op.matvec`` the scipy path uses.
        base_apply: ``M_base^-1`` alone (NOT the additive ``M_base^-1 +
            Q`` combination ``two_level``'s ``self._M`` holds) --
            ``InterfaceCGSolver._M_base_apply``.
        coarse: A :class:`interface_coarse.CoarseSpace` with ``SZ``
            retained (``coarse.SZ is not None`` -- callers must check this
            before selecting this function over the scipy path).
        b: Right-hand side, shape ``(n,)``.
        x0: Initial guess (seeds ``y0``), or ``None`` for a cold (zero)
            start.
        rtol, atol: Same stopping criterion scipy.sparse.linalg.cg documents:
            ``||r|| <= max(rtol * ||b||, atol)``.
        maxiter: Maximum CG iterations.
        reproject_every: Residual re-projection interval (see above).
            ``None`` resolves dynamically to
            ``interface_coarse.DEFAULT_DEFLATED_REPROJECT_EVERY`` at call
            time (Finding 9 pattern -- not bound at def time, so
            ``monkeypatch.setattr(interface_coarse, 'DEFAULT_DEFLATED_
            REPROJECT_EVERY', ...)`` still takes effect for callers that
            don't pass this explicitly); ``<= 0`` disables reprojection.
        callback: Called as ``callback(xk)`` once per completed iteration
            with the fully-recovered iterate (after updating ``y``),
            matching scipy's callback contract -- NOT called for the
            ``bnrm2 == 0`` or already-converged-at-x0' early exits (0
            iterations performed). When ``progress_every`` gates recovery
            off for a given iteration (see below), ``callback`` still fires
            every iteration but receives ``None`` instead of ``xk`` --
            callers that only count completed iterations (e.g.
            ``InterfaceCGSolver``'s own ``iters[0]`` bookkeeping) are
            unaffected; callers that need the real iterate every time
            (test harnesses checking drift/residuals) must leave
            ``progress_every`` at its default (``None``).
        progress_every: Perf gate for the (otherwise unconditional)
            ``callback(_recover_x())`` call every iteration -- recovering
            ``x`` costs two dense ``O(n * T')`` GEMVs via
            ``coarse.apply``/``apply_with_SQ``, real money at T' ~ 100+.
            ``None`` (default) preserves the original unconditional
            behaviour: recover and invoke ``callback`` with the real
            iterate every iteration, regardless of whether anything
            downstream consumes it. A caller that only wants ``callback``
            invoked for periodic true-residual logging (mirroring
            ``InterfaceCGSolver.__call__``'s own ``_callback`` gate on
            ``iters[0] % progress_every == 0``) should pass the SAME
            interval here: an iteration that will actually be logged (per
            that same modulus) gets the real recovered iterate; every other
            iteration -- including ALL of them when ``progress_every <= 0``
            ("logging disabled", the production default) -- calls
            ``callback(None)`` instead, skipping ``_recover_x()`` entirely.
            This has no effect on the returned solution, the convergence
            check, or warm-start -- ``_recover_x`` is still called
            (unconditionally) at every actual accept/return point; only the
            per-iteration callback delivery is gated.

    Returns:
        ``(x, info)`` -- ``info=0`` on convergence, else the number of
        iterations performed (== ``maxiter`` when the loop is exhausted
        without converging), mirroring ``scipy.sparse.linalg.cg``'s
        contract.
    """
    if reproject_every is None:
        reproject_every = interface_coarse.DEFAULT_DEFLATED_REPROJECT_EVERY
    b = np.asarray(b, dtype=np.float64)
    n = b.shape[0]
    bnrm2 = float(np.linalg.norm(b))
    # Same stopping-criterion normalization scipy.sparse.linalg.cg documents:
    # ||r|| <= max(rtol * ||b||, atol).
    atol_eff = max(float(atol), float(rtol) * bnrm2)

    if bnrm2 == 0.0:
        # Mirror scipy.sparse.linalg.cg's own bnrm2==0 special case: for an
        # SPD S, the unique solution to S x = 0 is x = 0, independent of
        # x0/warm-start -- return immediately, 0 iterations.
        return np.zeros(n, dtype=np.float64), 0

    # y solves the projected system (P S) y = P b; x is recovered from y
    # (never accumulated directly -- see this function's docstring for why).
    y = (
        np.zeros(n, dtype=np.float64) if x0 is None
        else np.array(x0, dtype=np.float64, copy=True)
    )
    Sy = matvec(y) if np.any(y) else np.zeros(n, dtype=np.float64)
    # Round-3 code review finding 8 (cleanup): compute `b - Sy` once and
    # reuse -- the pre-fix code materialized this O(n) temporary twice
    # (once as the apply_with_SQ argument, once for the subtraction).
    _resid0 = b - Sy
    _, SQr0 = coarse.apply_with_SQ(_resid0)
    r = _resid0 - SQr0  # = P(b - S y0)

    def _recover_x() -> np.ndarray:
        # x = y + Q(b - S y) -- Sy is tracked incrementally (see the alpha
        # step below), so this is one cheap coarse.apply, never a fresh
        # full-n matvec.
        return y + coarse.apply(b - Sy)

    def _try_accept() -> Optional[Tuple[np.ndarray, int]]:
        """Spec numerical-hygiene (b): a tentative-convergence event (the
        tracked/deflated recurrence residual ``r`` meeting ``atol_eff``)
        is only ACCEPTED after a fresh true-residual check on the ORIGINAL
        system (``b - S @ x_recovered``, a real matvec -- never the tracked
        recurrence quantity), mirroring the strict-mode acceptance gate the
        scipy/additive path already applies on the failure branch (see
        ``InterfaceCGSolver.__call__``). Mathematically ``b - S x_recovered
        == r`` exactly (``x = y + Q(b-Sy)`` => ``b - Sx = P(b-Sy) = r``), so
        this fresh check should virtually always agree with the tracked
        value -- it exists as a safety net against ``Sy``'s INCREMENTAL
        accumulation (``Sy = Sy + alpha*Sp`` every iteration between
        re-projection points) drifting from the true ``S @ y`` over a very
        long solve.

        Round-3 code review finding 4 (PLAUSIBLE, confirmed -- fix scoped
        here after verification, not into the shared recurrence state):
        ``x_candidate`` is recovered using a FRESHLY recomputed ``Sy``
        (``matvec(y)``, one extra full-``n`` matvec beyond the fresh check
        below), not the loop's incrementally-tracked ``Sy`` -- WITHOUT
        this, a large enough drift between the incremental ``Sy`` and the
        true ``S @ y`` made this fresh check disagree with the tracked
        residual FOREVER (``x_candidate`` and hence the "fresh" check
        derived from it were both built from the SAME stale ``Sy``,
        so the disagreement could never resolve), a permanent,
        unrepairable acceptance failure that burns the whole ``maxiter``
        budget. Scoped to exactly this function (does NOT write back to
        the enclosing ``Sy``, unlike an earlier revision that refreshed
        it inside the periodic re-projection block instead) -- refreshing
        the SHARED recurrence ``Sy`` measurably slowed convergence on a
        real, very ill-conditioned fixture (perturbs CG conjugacy every
        ``reproject_every`` iterations; see this function's enclosing
        docstring's "Re-projection" paragraph for the regression that
        caught it) even though it is a no-op on every synthetic
        (identity-``matvec``) test fixture. Confining the refresh to the
        acceptance decision alone repairs exactly the drift this
        docstring describes with zero effect on the ongoing recurrence.
        Returns ``(x, 0)`` when the fresh check confirms convergence, else
        ``None`` (caller keeps iterating
        -- a disagreement here is not a hard failure, just "not actually
        converged yet"). Reads ``r``/``Sy``/``y`` from the enclosing scope
        (Finding 14, round 1: this used to take an unused ``rnorm_tracked``
        parameter -- churn residue from an earlier revision that compared
        the tracked and fresh norms directly -- removed; only the fresh
        check has ever driven the accept/reject decision).
        """
        Sy_fresh = matvec(y)
        x_candidate = y + coarse.apply(b - Sy_fresh)
        true_rnorm = float(np.linalg.norm(b - matvec(x_candidate)))
        if true_rnorm <= atol_eff:
            return x_candidate, 0
        return None

    rho_prev: Optional[float] = None
    rho_prev_scale: float = 0.0
    p: Optional[np.ndarray] = None
    # Debounced + re-armed fresh true-residual check -- see this module's
    # docstring ("True-residual acceptance" bullet) for the full contract
    # and Findings 9 (round 1) / 3 (round 2) for why: edge-triggered alone
    # (Finding 9) avoids paying the extra matvec every iteration once the
    # tracked residual sits below tolerance, but a PURE edge-trigger never
    # retries after one disagreement (Finding 3 regression) -- re-arm every
    # `_rearm_period` iterations while still stuck below tolerance, bounding
    # the cost to O(1) extra matvecs per period instead of the full maxiter
    # budget. `_rearm_period` = `reproject_every` when reprojection is
    # enabled (a reprojected residual is the likeliest moment a stale
    # disagreement resolves), else `DEFLATED_DEBOUNCE_REARM_FALLBACK_ITERS`.
    #
    # Round-3 code review finding 6 (PLAUSIBLE, confirmed): the round-2
    # re-arm condition was `not _was_below_tol or _iters_since_accept_
    # attempt >= _rearm_period` -- an EDGE trigger (`not _was_below_tol`)
    # ORed with the period throttle, not a pure period throttle. CG's
    # Euclidean residual is not monotone, so on an ill-conditioned/fp32
    # solve the tracked residual can cross above and back below
    # `atol_eff` every couple of iterations; EVERY such re-crossing is a
    # fresh "rising edge" that fires `not _was_below_tol` regardless of
    # how recently the last attempt ran, defeating the O(1)-per-period
    # bound the debounce exists to guarantee (up to ~50% matvec overhead
    # on an oscillating residual instead of one attempt per
    # `_rearm_period`). Fix: track iterations since the LAST attempt
    # continuously (incremented whether above or below tolerance) and gate
    # PURELY on that counter -- `_iters_since_accept_attempt` initialized
    # to `_rearm_period` so the very first below-tolerance iteration still
    # attempts immediately (preserving the original "check as soon as
    # tolerance is first met" behaviour), with no separate edge-trigger
    # path to bypass the spacing.
    _rearm_period = (
        reproject_every if reproject_every > 0
        else DEFLATED_DEBOUNCE_REARM_FALLBACK_ITERS
    )
    _iters_since_accept_attempt = _rearm_period

    for iteration in range(maxiter):
        rnorm = float(np.linalg.norm(r))
        _is_below_tol = rnorm <= atol_eff
        # Round-3 code review finding 5 (CONFIRMED): tracks whether
        # `_try_accept()` already ran THIS iteration -- the double-
        # breakdown exit below must not repeat a byte-identical fresh-
        # true-residual check (a wasted full-n matvec + O(n*T') recovery
        # for a provably unchanged answer). `_try_accept()` depends only
        # on `y` (it now recomputes its own fresh `Sy` locally -- see its
        # docstring, finding 4) and `b`, and `y` is not modified anywhere
        # between the top-of-loop attempt and the exit below (only the
        # bottom-of-loop `alpha` step updates it, which runs AFTER both)
        # -- so within one iteration, a second attempt is ALWAYS a
        # byte-identical duplicate of the first; no invalidation case is
        # needed. Reset every iteration.
        _accept_attempted_this_iter = False
        if _is_below_tol and _iters_since_accept_attempt >= _rearm_period:
            accepted = _try_accept()
            _accept_attempted_this_iter = True
            _iters_since_accept_attempt = 0
            if accepted is not None:
                return accepted
        else:
            _iters_since_accept_attempt += 1

        # Re-projection (hygiene only -- see docstring): recompute r from
        # scratch via (b - Sy) rather than trusting the incrementally
        # updated r, correcting any accumulated floating-point drift.
        # Round-3 spec-compliance review finding: `if reproject_every` alone
        # only treats exactly 0 as "disabled" -- a NEGATIVE value is truthy
        # in Python, and `iteration % (-k)` is 0 whenever `iteration` is a
        # multiple of `k` (== every iteration for -1, since x % -1 == 0 for
        # ALL integers x), so a caller passing e.g. -1 to "disable" per the
        # documented "<= 0 disables" contract got MAXIMUM reprojection
        # instead. Gate on `reproject_every > 0` explicitly so both 0 and
        # any negative value disable, matching the docstring/CLI/param docs.
        if reproject_every > 0 and iteration > 0 and iteration % reproject_every == 0:
            # Round-3 code review finding 4 (PLAUSIBLE, confirmed -- fix
            # scoped to `_try_accept` instead, see its docstring): this
            # deliberately reuses the loop's incrementally-tracked `Sy`
            # (NOT a fresh `matvec(y)`) -- an earlier revision refreshed
            # `Sy` here too and it measurably slowed convergence on a very
            # ill-conditioned real fixture by perturbing the ongoing CG
            # recurrence's conjugacy every `reproject_every` iterations.
            # The drift that actually needs repairing only matters at the
            # acceptance decision (`_try_accept`, which now recomputes its
            # own fresh `Sy` locally, without writing back here) -- this
            # block stays exactly what it always was: `Z^T r -> 0`
            # floating-point hygiene on `r`, using the SAME `Sy` the
            # ongoing recurrence already trusts.
            #
            # Round-3 code review finding 8 (cleanup): compute `b - Sy`
            # once and reuse (see the identical setup-block comment above).
            _resid_rp = b - Sy
            _, _SQr_rp = coarse.apply_with_SQ(_resid_rp)
            r = _resid_rp - _SQr_rp
            # Finding 4 (round 2): keep `rnorm` in sync with `r` for the
            # rest of THIS iteration -- the double-breakdown exit below
            # must see the reprojected residual, not the stale
            # top-of-loop value from before this block ran.
            rnorm = float(np.linalg.norm(r))

        z = base_apply(r)  # plain base apply -- no Q term (see docstring)

        rho_cur = float(np.dot(r, z))
        # Round-3 code review finding 8 (cleanup): `rnorm` already holds
        # `norm(r)` exactly (computed at the top of this iteration, and
        # refreshed above if reprojection just ran) -- reuse it instead of
        # a redundant O(n) `np.linalg.norm(r)` reduction every iteration.
        _rho_cur_scale = rnorm * float(np.linalg.norm(z))
        # Finding 9 (round 2): track whether `p` was ALREADY `z` this
        # iteration -- the pw-breakdown retry below must not redundantly
        # recompute an identical `p=z` attempt when it was.
        _p_was_z: bool
        if (
            iteration > 0
            and rho_prev is not None
            and not _is_breakdown(rho_prev, rho_prev_scale)
        ):
            beta = rho_cur / rho_prev
            p = z + beta * p
            _p_was_z = False
        else:
            # First iteration, OR a CG breakdown (rho_prev degenerate
            # despite r not yet meeting the convergence tolerance -- can
            # occur on ill-conditioned/penalty-heavy systems). Restart the
            # search direction (beta=0) instead of dividing by a
            # degenerate value -- standard CG-breakdown handling; costs at
            # most a few extra iterations, never produces an incorrect
            # result.
            p = z.copy()
            _p_was_z = True

        Sp = matvec(p)
        _, SQp = coarse.apply_with_SQ(Sp)
        w = Sp - SQp  # = P(S p) -- the projected matvec
        pw = float(np.dot(p, w))
        _pw_scale = float(np.linalg.norm(p)) * float(np.linalg.norm(w))
        if _is_breakdown(pw, _pw_scale):
            # Finding 9 (round 2): skip the retry when `p` was ALREADY `z`
            # above -- recomputing `p=z.copy()` would call
            # `matvec`/`apply_with_SQ` on the byte-identical `p` a second
            # time, deterministically reproducing the same `pw` (wasting a
            # full-n matvec + two O(n*T') GEMVs) before falling through to
            # the "genuinely stuck" exit below anyway. Go straight there,
            # reusing the already-computed `pw`/`_pw_scale`.
            if not _p_was_z:
                # Degenerate search direction under the SAME breakdown
                # scenario -- restart once more with the plain
                # preconditioned residual before giving up.
                p = z.copy()
                Sp = matvec(p)
                _, SQp = coarse.apply_with_SQ(Sp)
                w = Sp - SQp
                pw = float(np.dot(p, w))
                _pw_scale = float(np.linalg.norm(p)) * float(np.linalg.norm(w))
            if _is_breakdown(pw, _pw_scale):
                # Genuinely stuck (z itself is S-null this iteration) --
                # nothing more this iteration can do; report success if r
                # already meets the bar (fresh true-residual gated, same as
                # every other acceptance point), else stop rather than
                # raise ZeroDivisionError.
                #
                # Finding 5 (round 3, CONFIRMED): skip this attempt when
                # a top-of-loop attempt already ran THIS iteration
                # (`_accept_attempted_this_iter`) -- `_try_accept()`
                # depends only on `y` and `b` (it recomputes its own fresh
                # `Sy` locally -- finding 4), neither of which the
                # intervening base_apply/CG-direction/breakdown work above
                # touches, so a repeat here would recompute the
                # byte-identical candidate and matvecs and, having already
                # returned `None` once, deterministically return `None`
                # again -- two wasted full-n matvecs + O(n*T') recovery
                # for a provably unchanged answer.
                if rnorm <= atol_eff and not _accept_attempted_this_iter:
                    accepted = _try_accept()
                    if accepted is not None:
                        return accepted
                # Finding 7 (round 1): report the ACTUAL iterations
                # performed (`iteration`), not a hardcoded `maxiter` --
                # scipy's contract is info = iterations performed on
                # failure, and `stats['last_cg_info']` must agree with
                # `stats['last_cg_iters']`/the strict-mode error message.
                #
                # Finding 1 (round 2, CRITICAL -- corrects the above): a
                # plain `iteration` value collides with scipy's SUCCESS
                # code (0) when this breakdown strikes on the very first
                # iteration (e.g. an ill-conditioned/penalty-heavy system
                # whose first search direction is already S-null).
                # Reaching this line means the solve did NOT converge (the
                # `rnorm <= atol_eff` gate above already returned on the
                # real success path) -- returning info=0 here falsely
                # reports it as converged: __call__ skips the strict-mode
                # raise, clears `cg_failed`, and pushes the bad iterate
                # into the warm-start history. scipy's info-is-iterations-
                # performed contract has no representation for "0
                # iterations performed but NOT converged" that isn't also
                # its success code, so floor at 1 -- the smallest value
                # that still signals failure. This slightly overstates
                # `last_cg_iters` in the rare iteration==0 case (a cosmetic
                # mismatch, same class already tolerated elsewhere for
                # `last_cg_info`), a strictly better trade than a
                # false-success info=0 silently corrupting DC/QS/transient
                # results.
                return _recover_x(), max(iteration, 1)
        alpha = rho_cur / pw
        y = y + alpha * p
        Sy = Sy + alpha * Sp  # incremental -- no extra full-n matvec
        r = r - alpha * w
        rho_prev = rho_cur
        rho_prev_scale = _rho_cur_scale

        if callback is not None:
            if progress_every is None:
                # Default/back-compat contract: always deliver the real
                # recovered iterate (see the ``progress_every`` Args entry
                # above).
                callback(_recover_x())
            elif progress_every > 0 and (iteration + 1) % progress_every == 0:
                # This iteration is one InterfaceCGSolver.__call__'s
                # ``_callback`` will actually log (same modulus, so the two
                # stay in lockstep) -- recovery is needed.
                callback(_recover_x())
            else:
                # Recovery would be thrown away (progress_every<=0 disables
                # logging entirely, or this iteration falls between log
                # points) -- skip the two O(n*T') GEMVs and just signal
                # "one more iteration completed" so callers that use the
                # callback purely as an iteration counter stay correct.
                callback(None)

    # Round-2 spec-compliance review finding: the loop above only checks
    # convergence at the TOP of each iteration (before that iteration's
    # update), so a solve that first satisfies the tolerance on the FINAL
    # (maxiter-th) update was previously falling through to `maxiter`
    # (reported as non-convergence) without ever being checked -- a real
    # deviation from scipy.sparse.linalg.cg, which checks after every
    # update including the last. Check the final post-update residual here,
    # through the same fresh-true-residual gate as every other acceptance
    # point, before conceding maxiter.
    final_rnorm = float(np.linalg.norm(r))
    if final_rnorm <= atol_eff:
        accepted = _try_accept()
        if accepted is not None:
            return accepted
    # Finding 1 (round 2): same info=0-false-success collision as the
    # double-breakdown exit above, but for `maxiter == 0` (the loop above
    # never runs). `max(maxiter, 1)` is a no-op for every real maxiter.
    return _recover_x(), max(maxiter, 1)


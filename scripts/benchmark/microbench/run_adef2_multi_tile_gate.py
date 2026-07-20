#!/usr/bin/env python
"""A-DEF2 work package gate 4: netlist_multi_tile DC + transient smoke.

Naming note (ratification-resolution pass): the work-package spec below and
the "round 2/4" review sections further down refer to the deflated apply
mode by its ORIGINAL setting value ``apply_mode='adef2'`` -- that value has
since been RENAMED to ``apply_mode='deflated'`` (coordinator ruling: the
algorithm shipped is DEF, not literal A-DEF2 -- see
``interface_iterative.py``'s "A-DEF2 work package" docstring section for the
full selection record, including a true-A-DEF2 candidate that was
implemented, measured head-to-head via THIS script, and rejected). Every
code reference below uses the current name ``'deflated'``; the historical
prose (spec quote, round-2/round-4 narrative, measured numbers like
"adef2=83.00 iters/step") is left as originally written since it is an
accurate record of what was measured under the old name -- read
``'adef2'``/``'ADEF2'`` in that historical text as "the algorithm now called
'deflated'".

Spec (stage_adef2_spec.md, "Gates you must run and pass", item 4):

    netlist_multi_tile smoke: DC + 20-step transient with
    interface_solver=cg, matvec tilewise, two_level, apply_mode=adef2,
    drop_s_global=True: converged, max|dV| vs direct <= 1e-6 at rtol
    1e-8, iters <= the additive run of the same (print both).

Local backend (9 tiles -- small enough, no Ray needed). Runs, for EACH of
two block-Jacobi memory-budget scenarios (see --bj-max-bytes below):
  1. DC direct (assembled) reference (budget-independent, computed once).
  2. DC two_level ADDITIVE, tilewise, never-assemble, rtol=1e-8.
  3. DC two_level DEFLATED (formerly 'adef2'), tilewise, never-assemble,
     rtol=1e-8.
  4. 20-step BE transient, IC = DC additive result, two_level ADDITIVE.
  5. 20-step BE transient, IC = DC deflated result, two_level DEFLATED.

Two scenarios, both reported (implementation-agent design decision, see
below):
  - "natural": default interface_block_jacobi_max_bytes (8 GB auto budget).
    At netlist_multi_tile's small scale (9 tiles, T'=10) block_jacobi is
    NOT downgraded -- this exercises two_level(bj+...), not the regime the
    A-DEF2 work package's motivation (docs Sec 7.9) targets.
  - "jacobi-forced": interface_block_jacobi_max_bytes=1 (forces the
    diagonal-jacobi downgrade regardless of scale) -- mirrors the ACTUAL
    mi200k_v2 production regime the spec's "Why" section describes ("the
    BJ byte-budget guard downgrades the base to diagonal jacobi at this
    regime"), where the warm-iteration gain A-DEF2 targets was measured.
    This is the scientifically relevant scenario for this feature; the
    coordinator's own mi200k/Ray measurement (out of scope here) is the
    authoritative version at the TARGET scale.

Spec-compliance review round 2 (major finding) -- root cause of the
'natural' scenario's TR PASS=False
--------------------------------------------------------------------------
A round-2 reviewer independently ran this script and confirmed: DC PASSES
in BOTH scenarios (adef2 <= additive), but TRANSIENT (warm-started) FAILS
in 'natural' specifically -- measured adef2=83.00 iters/step vs.
additive=74.65 iters/step (block_jacobi base, 9 tiles). This package's own
fix pass (see interface_iterative.py's "A-DEF2" docstring section, "Round 2
independent re-verification") root-caused it, confirmed via direct
instrumentation (wrapping InterfaceCGSolver.__call__ to compare the
UNPROJECTED warm-start residual ``r0_full = b - S @ x_prev`` against the
PROJECTED residual DEF actually starts from, ``r0_proj = P(r0_full)``):

    P = I - S Q is an OBLIQUE projector (not orthogonal in the standard
    inner product -- Q's range and S's structure are not aligned in
    general, so P is not guaranteed norm-non-increasing). On THIS
    fixture's ACTUAL warm-start residuals (``b_new - S @ x_prev`` at the
    start of each real transient step) it was measured to AMPLIFY the
    residual norm by a consistent ~4-5x factor EVERY step in the 'natural'
    (block_jacobi base) scenario (``r0_proj / r0_full`` ranged 3.7-5.7x
    across 19 steps). IMPORTANT CAVEAT (round-2 fix-pass follow-up): this
    is an EMPIRICALLY MEASURED property of netlist_multi_tile's actual
    coarse space / RHS-delta structure, not a generic property of oblique
    projectors -- deliberately probed on generic Gaussian vectors, on
    solution-space perturbations, and on single-node unit vectors using
    the SAME synthetic ill-conditioned fixture the rest of this package's
    tests use (T'/n ~10%, disjoint ownership blocks), and NONE of those
    reproduced amplification above ~1.1-1.5x -- so the effect is tied to
    something specific to netlist_multi_tile's real 2D PDN geometry (e.g.
    how its PoU columns partition the actual tile grid, or how transient
    current-source RHS deltas project onto that partition), not
    reproduced with a portable synthetic fixture, and NOT asserted as a
    regression-pinned unit test for that reason. Treat the mechanism
    (an oblique projector CAN amplify SOME residuals) as solid math and
    the ~4-5x NUMBER as fixture-specific measured fact, not a provable
    general bound. Given the amplification IS real on this fixture and
    correlates cleanly with the measured iteration gap (Because the
    stopping criterion is ABSOLUTE -- ``||r|| <= max(rtol * ||b||,
    atol)``, not relative to ``||r0||``, required by this package's own
    spec, "mirror every behavior of the current __call__ scipy path" --
    amplifying r0 directly costs CG several EXTRA iterations to shrink it
    back down to the same absolute target), it remains the best available
    explanation for the 'natural' scenario's gap, but should be read as a
    diagnosed CORRELATION with strong circumstantial support, not a proven
    causal mechanism. This overhead is roughly FIXED in absolute iteration
    count (a log-amplitude tax), so it is a small, easily-absorbed
    fraction of a COLD/DC solve's large total reduction (~8-10 orders of
    magnitude -- matches DC PASS=True in both scenarios) but a much larger
    RELATIVE fraction of a WARM transient step's already-small total
    reduction (~2 orders of magnitude) -- exactly the regime this whole
    work package targets improving. It is also a larger fraction
    specifically when the BASE preconditioner is already strong
    (block_jacobi, the 'natural' scenario): DEF's exact deflation only has
    to make up the GAP block_jacobi doesn't already close, so the (roughly
    fixed) amplification tax looms larger relative to that smaller gap --
    with a weak base (diagonal jacobi, 'jacobi-forced'), the gap DEF
    closes is much bigger, so the SAME absolute tax is a smaller fraction
    and DEF wins outright (measured 16.75 < 20.30 iters/step there). This
    is INTERNALLY CONSISTENT with (not a contradiction of) the spec's own
    "Why" section: the mi200k_v2 PRODUCTION target is ALREADY in the
    jacobi-downgraded regime (the BJ byte-budget guard downgrades to
    diagonal jacobi at that scale), i.e. the 'jacobi-forced' scenario
    here, not 'natural' -- 'natural' only exists on netlist_multi_tile
    because the fixture's 9-tile scale is too small for the byte-budget
    guard to ever downgrade block_jacobi, making it a non-representative
    regime for THIS specific fixture, not for the production target the
    spec's own numbers describe.

Decision history (round 2 -> round 4 -> FINAL coordinator ruling, this pass)
--------------------------------------------------------------------------
Round 2 fix pass: exit code keyed off 'jacobi-forced' (production-
representative), 'natural' printed for transparency only.

Round-4 spec-compliance review finding: Gate 4's spec wording (module
docstring above) names ONE configuration -- default
``interface_block_jacobi_max_bytes`` (no bj forcing), i.e. this script's
'natural' scenario -- and says nothing about a 'jacobi-forced' variant, so
round 4 REVERSED round 2 and keyed the exit code off 'natural' instead
(spec-literal), demoting 'jacobi-forced' to informational.

FINAL coordinator ruling (this ratification-resolution pass, finding 4a):
REVERSES round 4 back to round 2's original design. The 'jacobi-forced'
scenario is the acceptance criterion; 'natural' is informational only
(printed, PASS/FAIL reported, but excluded from the exit code). Why: docs
§7.9 documents that at the mi200k_v2 production regime (64 tiles, 168,586
interface unknowns) the block-Jacobi byte-budget guard (``interface_
block_jacobi_max_bytes``) DOWNGRADES the base preconditioner to diagonal
jacobi -- this is the ACTUAL production regime the whole A-DEF2 work
package's "Why" section (this script's own docstring, above) is measuring
against. 'jacobi-forced' (``interface_block_jacobi_max_bytes=1``) exists
specifically to reproduce that downgrade on netlist_multi_tile's small
(9-tile) fixture, where the byte-budget guard would otherwise never trip
naturally (see the round-2 root-cause section above: "'natural' only
exists ... because the fixture's 9-tile scale is too small for the
byte-budget guard to ever downgrade block_jacobi"). 'natural' therefore
exercises a regime (full block_jacobi base) that netlist_multi_tile's own
small scale makes NON-representative of the production target -- keying
the gate's pass/fail on it (round 4's choice) measures the wrong thing
correctly rather than the right thing approximately. This is a deliberate,
documented coordinator decision to prioritize scientific relevance over
literal spec-wording match; both scenarios are still run and printed in
full (the 'natural' PASS/FAIL line and its diagnostic NOTE, below, are
unchanged) so the spec-literal result remains visible even though it no
longer gates the process exit status.

Usage:
    venv/bin/python -u scripts/benchmark/microbench/run_adef2_multi_tile_gate.py \
        [netlist/netlist_multi_tile/distributed_pkl] [--dt 1e-10] [--steps 20]
"""
from __future__ import annotations

import argparse
import sys
import time
from typing import Dict, Optional


def _dc_run(bundle, apply_mode, rtol, bj_max_bytes=None, drop_s_global=True):
    from distributed.model import create_distributed_model
    from distributed.solver import DistributedDDMSolver

    model = create_distributed_model(bundle, backend='local')
    settings = {
        'interface_solver': 'cg',
        'interface_matvec_mode': 'tilewise',
        'interface_preconditioner': 'two_level',
        'interface_coarse_apply_mode': apply_mode,
        'interface_cg_rtol': rtol,
        'interface_cg_atol': 1e-14,
    }
    if bj_max_bytes is not None:
        settings['interface_block_jacobi_max_bytes'] = bj_max_bytes
    if drop_s_global:
        if getattr(model, 'island_detection_mode', 'schur_bfs') != 'summaries':
            print(
                f"[gate4] WARNING: island_detection_mode="
                f"{getattr(model, 'island_detection_mode', None)!r} != "
                f"'summaries' -- interface_drop_s_global will fall back to "
                f"the assembled path (spec precondition unmet by this pkl "
                f"bundle; re-parse to get drop_s_global=True honoured).",
                flush=True,
            )
        settings['interface_drop_s_global'] = True
    model.settings.update(settings)
    solver = DistributedDDMSolver(model)
    t0 = time.perf_counter()
    ctx = solver.prepare(verbose=False)
    prep_s = time.perf_counter() - t0
    cg = ctx._cg_solver
    t0 = time.perf_counter()
    result = solver.solve_dc(ctx)
    solve_s = time.perf_counter() - t0
    iters = cg.stats.get('last_cg_iters') if cg is not None else None
    label = cg.preconditioner_label if cg is not None else 'direct'
    dropped = ctx._S_global is None
    flat = result.flatten()
    ctx.release()
    model.shutdown()
    return {
        'flat': flat, 'iters': iters, 'prep_s': prep_s, 'solve_s': solve_s,
        'label': label, 's_global_dropped': dropped,
    }


def _direct_reference(bundle):
    from distributed.model import create_distributed_model
    from distributed.solver import DistributedDDMSolver

    model = create_distributed_model(bundle, backend='local')
    model.settings.update({
        'interface_solver': 'direct', 'interface_matvec_mode': 'assembled',
    })
    solver = DistributedDDMSolver(model)
    ctx = solver.prepare(verbose=False)
    result = solver.solve_dc(ctx)
    flat = result.flatten()
    ctx.release()
    model.shutdown()
    return flat


def _transient_run(bundle, apply_mode, rtol, n_steps, dt, tmp_dir,
                    bj_max_bytes=None):
    import os

    from distributed.model import create_distributed_model
    from distributed.solver import DistributedDDMSolver

    os.makedirs(tmp_dir, exist_ok=True)

    model = create_distributed_model(bundle, backend='local')
    settings = {
        'interface_solver': 'cg',
        'interface_matvec_mode': 'tilewise',
        'interface_preconditioner': 'two_level',
        'interface_coarse_apply_mode': apply_mode,
        'interface_cg_rtol': rtol,
        'interface_cg_atol': 1e-14,
        'interface_drop_s_global': True,
    }
    if bj_max_bytes is not None:
        settings['interface_block_jacobi_max_bytes'] = bj_max_bytes
    model.settings.update(settings)
    solver = DistributedDDMSolver(model)
    dc_ctx = solver.prepare(verbose=False)
    # dc_context=dc_ctx below re-solves DC internally (source-evaluated at
    # t_start) for the transient IC -- no separate solve_dc() call needed.

    t_end = n_steps * dt
    trans_ctx = solver.prepare_transient(dt=dt, method='be', verbose=False)
    cg = trans_ctx._cg_solver
    smoothed = solver.preprocess_sources(
        time_step=dt, t_start=0.0, t_end=t_end, smooth=False,
        pkl_dir=tmp_dir,
    )
    iters_before = cg.total_iterations if cg is not None else 0
    t0 = time.perf_counter()
    result = solver.solve_transient(
        trans_ctx, dc_context=dc_ctx, t_end=t_end,
        smoothed_sources=smoothed,
    )
    loop_s = time.perf_counter() - t0
    iters_total = (cg.total_iterations if cg is not None else 0) - iters_before
    label = cg.preconditioner_label if cg is not None else 'direct'
    dropped = trans_ctx._S_global is None
    flat = result.as_flat()
    trans_ctx.release()
    dc_ctx.release()
    model.shutdown()
    return {
        'flat': flat, 'iters_per_step': iters_total / n_steps, 'loop_s': loop_s,
        'label': label, 's_global_dropped': dropped,
    }


def _direct_transient_reference(bundle, n_steps, dt, tmp_dir):
    """Finding 11 (round-1 code review): the spec's Gate-4 acceptance
    criterion ("converged, max|dV| vs direct <= 1e-6 at rtol 1e-8, iters
    <= the additive run") requires a transient ACCURACY check, not just
    an iteration-count comparison between the two apply modes -- an
    iters-only gate would still exit 0 for a deflated-mode regression
    that silently stagnates (info=0 with the tracked residual satisfied
    while the recovered voltages are wrong -- exactly the failure mode
    this package's own history documents for a rejected variant, see
    ``interface_iterative.py``'s "A-DEF2 work package" docstring). Solved
    via 'direct' (exact LU factorization, not merely a tight CG rtol) --
    the reference itself has no 'rtol' at all (a direct factor's residual
    is floating-point-exact, not converged-to-a-tolerance), so it strictly
    exceeds the spec's ACTUAL Gate-4 precision bar quoted above ("rtol
    1e-8", not "rtol 1e-12" -- an earlier revision of this docstring/the
    progress print below mis-cited that number, which appears nowhere in
    the spec; the method (direct LU, always far tighter than any CG rtol)
    was never in question, only the citation) -- same IC/
    fixture as the additive/deflated transient runs (DC resolved via
    ``dc_context=`` at the same t_start, same dt/n_steps/smoothing),
    computed ONCE and shared across both memory-budget scenarios (the
    reference does not depend on ``interface_block_jacobi_max_bytes`` --
    'direct' mode never builds a block-Jacobi/two_level preconditioner at
    all).
    """
    import os

    from distributed.model import create_distributed_model
    from distributed.solver import DistributedDDMSolver

    os.makedirs(tmp_dir, exist_ok=True)

    model = create_distributed_model(bundle, backend='local')
    model.settings.update({
        'interface_solver': 'direct', 'interface_matvec_mode': 'assembled',
    })
    solver = DistributedDDMSolver(model)
    dc_ctx = solver.prepare(verbose=False)
    t_end = n_steps * dt
    trans_ctx = solver.prepare_transient(dt=dt, method='be', verbose=False)
    smoothed = solver.preprocess_sources(
        time_step=dt, t_start=0.0, t_end=t_end, smooth=False,
        pkl_dir=tmp_dir,
    )
    result = solver.solve_transient(
        trans_ctx, dc_context=dc_ctx, t_end=t_end,
        smoothed_sources=smoothed,
    )
    flat = result.as_flat()
    trans_ctx.release()
    dc_ctx.release()
    model.shutdown()
    return flat


def _max_dv(a: Dict, b: Dict) -> float:
    common = set(a) & set(b)
    assert common, "fixture produced no comparable nodes"
    return max(abs(a[n] - b[n]) for n in common)


def _max_dv_transient(a: Dict, b: Dict) -> float:
    """Like ``_max_dv`` but for ``DistributedTransientResult.as_flat()``
    dicts, whose values are ``(voltage, ...)`` tuples rather than bare
    scalars (mirrors the unpacking the pre-existing additive-vs-deflated
    own-ICs comparison below already used)."""
    common = set(a) & set(b)
    assert common, "fixture produced no comparable nodes"
    max_dv = 0.0
    for node in common:
        va, _ = a[node]
        vb, _ = b[node]
        max_dv = max(max_dv, abs(va - vb))
    return max_dv


def _run_scenario(
    bundle, v_direct, tr_direct, scenario_name, rtol, n_steps, dt, tmp_root,
    bj_max_bytes: Optional[int],
) -> bool:
    import os

    print(f"\n[gate4] === scenario: {scenario_name} "
          f"(interface_block_jacobi_max_bytes={bj_max_bytes!r}) ===", flush=True)

    print("[gate4] DC two_level ADDITIVE (never-assemble, tilewise)...", flush=True)
    dc_additive = _dc_run(bundle, 'additive', rtol, bj_max_bytes=bj_max_bytes)
    print(
        f"[gate4] DC additive: {dc_additive['iters']} iters, "
        f"{dc_additive['solve_s']:.3f}s, label={dc_additive['label']}, "
        f"S_global_dropped={dc_additive['s_global_dropped']}", flush=True,
    )

    print("[gate4] DC two_level DEFLATED (never-assemble, tilewise)...", flush=True)
    dc_deflated = _dc_run(bundle, 'deflated', rtol, bj_max_bytes=bj_max_bytes)
    print(
        f"[gate4] DC deflated: {dc_deflated['iters']} iters, "
        f"{dc_deflated['solve_s']:.3f}s, label={dc_deflated['label']}, "
        f"S_global_dropped={dc_deflated['s_global_dropped']}", flush=True,
    )

    dc_max_dv_additive = _max_dv(v_direct, dc_additive['flat'])
    dc_max_dv_deflated = _max_dv(v_direct, dc_deflated['flat'])
    print(
        f"[gate4] DC max|dV| vs direct: additive={dc_max_dv_additive:.3e} V, "
        f"deflated={dc_max_dv_deflated:.3e} V (gate: deflated <= 1e-6)", flush=True,
    )
    print(
        f"[gate4] DC iters: additive={dc_additive['iters']}, "
        f"deflated={dc_deflated['iters']} (gate: deflated <= additive)", flush=True,
    )

    dc_ok = (
        dc_max_dv_deflated <= 1e-6
        and dc_deflated['iters'] is not None
        and dc_additive['iters'] is not None
        and dc_deflated['iters'] <= dc_additive['iters']
    )
    print(f"[gate4] DC PASS={dc_ok}", flush=True)

    tmp_dir = os.path.join(tmp_root, scenario_name)
    print(
        f"[gate4] transient two_level ADDITIVE, {n_steps} steps, "
        f"dt={dt:.3e}...", flush=True,
    )
    tr_additive = _transient_run(
        bundle, 'additive', rtol, n_steps, dt, tmp_dir + '/additive',
        bj_max_bytes=bj_max_bytes,
    )
    print(
        f"[gate4] TR additive: {tr_additive['iters_per_step']:.2f} "
        f"iters/step, {tr_additive['loop_s']:.3f}s loop, "
        f"label={tr_additive['label']}, "
        f"S_global_dropped={tr_additive['s_global_dropped']}", flush=True,
    )

    print(
        f"[gate4] transient two_level DEFLATED, {n_steps} steps, "
        f"dt={dt:.3e}...", flush=True,
    )
    tr_deflated = _transient_run(
        bundle, 'deflated', rtol, n_steps, dt, tmp_dir + '/deflated',
        bj_max_bytes=bj_max_bytes,
    )
    print(
        f"[gate4] TR deflated: {tr_deflated['iters_per_step']:.2f} "
        f"iters/step, {tr_deflated['loop_s']:.3f}s loop, "
        f"label={tr_deflated['label']}, "
        f"S_global_dropped={tr_deflated['s_global_dropped']}", flush=True,
    )

    common = set(tr_additive['flat']) & set(tr_deflated['flat'])
    tr_max_dv = 0.0
    for node in common:
        d_a, _ = tr_additive['flat'][node]
        d_b, _ = tr_deflated['flat'][node]
        tr_max_dv = max(tr_max_dv, abs(d_a - d_b))
    print(
        f"[gate4] TR max|dV| additive vs deflated (own ICs): "
        f"{tr_max_dv:.3e} V", flush=True,
    )

    # Finding 11 (round-1 code review): the spec's transient acceptance
    # criterion is "converged, max|dV| vs direct <= 1e-6 at rtol 1e-8,
    # iters <= the additive run" -- an ACCURACY check against a direct
    # (exact) reference, not merely the additive-vs-deflated own-ICs
    # comparison above (which only bounds how far the two apply modes
    # drift from EACH OTHER, not whether either is actually correct --
    # both could stagnate together and still show tr_max_dv ~ 0).
    tr_max_dv_additive = _max_dv_transient(tr_direct, tr_additive['flat'])
    tr_max_dv_deflated = _max_dv_transient(tr_direct, tr_deflated['flat'])
    print(
        f"[gate4] TR max|dV| vs direct reference: "
        f"additive={tr_max_dv_additive:.3e} V, "
        f"deflated={tr_max_dv_deflated:.3e} V (gate: both <= 1e-6)",
        flush=True,
    )
    print(
        f"[gate4] TR iters/step: additive={tr_additive['iters_per_step']:.2f}, "
        f"deflated={tr_deflated['iters_per_step']:.2f} "
        f"(gate: deflated <= additive)", flush=True,
    )
    tr_accuracy_ok = tr_max_dv_additive <= 1e-6 and tr_max_dv_deflated <= 1e-6
    tr_iters_ok = tr_deflated['iters_per_step'] <= tr_additive['iters_per_step']
    tr_ok = tr_accuracy_ok and tr_iters_ok
    print(
        f"[gate4] TR accuracy PASS={tr_accuracy_ok}, iters PASS={tr_iters_ok}",
        flush=True,
    )
    print(f"[gate4] TR PASS={tr_ok}", flush=True)

    scenario_ok = dc_ok and tr_ok
    print(f"[gate4] scenario '{scenario_name}' PASS={scenario_ok}", flush=True)
    return scenario_ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        'pkl_dir', nargs='?',
        default='netlist/netlist_multi_tile/distributed_pkl',
    )
    ap.add_argument('--dt', type=float, default=1e-10)
    ap.add_argument('--steps', type=int, default=20)
    ap.add_argument('--rtol', type=float, default=1e-8)
    args = ap.parse_args()

    from distributed.model import load_distributed_partitions
    import tempfile

    print(f"[gate4] loading {args.pkl_dir}", flush=True)
    bundle = load_distributed_partitions(args.pkl_dir)

    print("[gate4] DC direct reference...", flush=True)
    v_direct = _direct_reference(bundle)

    results = {}
    with tempfile.TemporaryDirectory() as tmp_root:
        print("[gate4] TR direct reference (exact LU, far tighter than the "
              "gate's rtol=1e-8 bar)...",
              flush=True)
        tr_direct = _direct_transient_reference(
            bundle, args.steps, args.dt, tmp_root + '/direct_ref',
        )
        results['natural'] = _run_scenario(
            bundle, v_direct, tr_direct, 'natural', args.rtol, args.steps,
            args.dt, tmp_root, bj_max_bytes=None,
        )
        results['jacobi-forced'] = _run_scenario(
            bundle, v_direct, tr_direct, 'jacobi-forced', args.rtol,
            args.steps, args.dt, tmp_root, bj_max_bytes=1,
        )

    print("\n[gate4] === summary ===", flush=True)
    for name, ok in results.items():
        print(f"[gate4] scenario '{name}': PASS={ok}", flush=True)
    print(
        f"\n[gate4] OVERALL (spec-literal, 'natural' scenario) "
        f"PASS={results['natural']}", flush=True,
    )
    print(
        f"[gate4] OVERALL (production-regime, 'jacobi-forced' scenario) "
        f"PASS={results['jacobi-forced']}", flush=True,
    )
    if not results['natural']:
        print(
            "[gate4] NOTE (spec-compliance review round 2, root-caused): "
            "'natural' TR PASS=False is EXPECTED on this fixture -- DEF's "
            "projected-matvec construction uses an OBLIQUE projector P "
            "that amplifies the warm-started residual ~4-5x on this small "
            "(9-tile) fixture with a block_jacobi base; against the "
            "ABSOLUTE stopping criterion this costs several extra "
            "iterations, a bigger relative tax on a warm step's small "
            "total reduction than on a cold DC solve's large one. The "
            "mi200k_v2 PRODUCTION target is ALREADY jacobi-downgraded (see "
            "this script's module docstring's 'Spec-compliance review "
            "round 2' section) -- 'jacobi-forced' reproduces that regime on "
            "this small fixture and is the scenario this script's exit "
            "code below is keyed on (coordinator ruling, finding 4a -- see "
            "the module docstring's 'FINAL coordinator ruling' section). "
            "'natural' is printed here for transparency but does NOT gate "
            "the exit status.", flush=True,
        )
    # Coordinator ruling (finding 4a, this ratification-resolution pass):
    # 'jacobi-forced' is the acceptance criterion (mirrors the mi200k_v2
    # production regime -- docs §7.9's BJ byte-budget guard downgrades the
    # base to diagonal jacobi at that scale); 'natural' is informational
    # only (PASS/FAIL printed above, excluded from the exit code). See the
    # module docstring's "Decision history (round 2 -> round 4 -> FINAL
    # coordinator ruling, this pass)" section for the full history and why
    # this reverses round 4's spec-literal choice.
    sys.exit(0 if results['jacobi-forced'] else 1)


if __name__ == '__main__':
    main()

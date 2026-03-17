#!/usr/bin/env python
"""Validate distributed solver context lifecycle and result correctness.

Checks:
  1. DC -> release -> transient flow (LUs never coexist)
  2. Flat solver comparison (DDM DC vs flat DC)
  3. Save/load round-trip (checkpoint -> release -> load -> refactor -> solve)
  4. Both transient IC paths produce identical results

Uses backend='local' (no Ray) and pre-parsed tiles in distributed_pkl/.
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
import time

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NETLIST_SAMPLED_DIR = os.path.join(REPO, 'netlist', 'netlist_sampled')
PKL_DIR = os.path.join(NETLIST_SAMPLED_DIR, 'distributed_pkl')

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.WARNING,
    format='%(levelname)s %(name)s: %(message)s',
)
logger = logging.getLogger('validate')
logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TOLERANCE = 1e-6  # 1 uV (for flat vs DDM DC comparison)

# The dc_context and ic_voltages paths use different numerical pathways
# for the initial condition (Schur-complement interior recovery vs direct
# dict assignment), so transient steps accumulate FP differences.
TRANSIENT_IC_PATH_TOL = 1e-5  # 10 uV (for dc_context vs ic_voltages path)


def _banner(title: str) -> None:
    sep = '=' * 70
    print(f'\n{sep}')
    print(f'  {title}')
    print(sep)


def _pass(msg: str) -> None:
    print(f'  PASS  {msg}')


def _fail(msg: str) -> None:
    print(f'  FAIL  {msg}')
    sys.exit(1)


def _check(condition: bool, msg: str, detail: str = '') -> None:
    if condition:
        _pass(msg)
    else:
        _fail(f'{msg} -- {detail}')


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def create_ddm():
    """Create distributed model + solver from pre-parsed pkl files."""
    from distributed.model import load_distributed_partitions, create_distributed_model
    from distributed.solver import DistributedDDMSolver

    bundle = load_distributed_partitions(PKL_DIR)
    model = create_distributed_model(bundle, backend='local')
    solver = DistributedDDMSolver(model)
    return model, solver


def create_flat():
    """Create flat solver from sampled netlist."""
    from parser.netlist import NetlistParser
    from model.factory import create_model_from_pdn
    from solver.unified_solver import UnifiedIRDropSolver

    parser = NetlistParser(NETLIST_SAMPLED_DIR)
    graph = parser.parse()
    model = create_model_from_pdn(graph, 'VDD_XLV')
    load_currents = model.extract_current_sources()
    solver = UnifiedIRDropSolver(model)
    return model, solver, load_currents


# ===================================================================
# Validation 1: DC -> release -> transient flow
# ===================================================================

def validate_dc_release_transient(model, solver):
    """Validate that DC and transient LUs never coexist.

    Flow:
      prepare() -> solve_dc() -> extract IC voltages -> release()
      -> prepare_transient() -> solve_transient(ic_voltages=...)
    Also runs the dc_context path for comparison.
    """
    _banner('Validation 1: DC -> release -> transient (LUs never coexist)')

    # Transient parameters
    t_end = 2e-9
    dt = 100e-12  # 100 ps

    # -- Step 1: DC solve --
    t0 = time.perf_counter()
    dc_ctx = solver.prepare()
    dc_result = solver.solve_dc(dc_ctx)
    t_dc = time.perf_counter() - t0

    _check(dc_ctx.is_factored, 'DC context is factored after prepare+solve')

    # Extract IC voltages from DC result (pass DistributedSolveResult directly)
    vdd = model.vdd
    max_ir_drop_dc = max(dc_result.ir_drop.values())
    print(f'  INFO  DC solve: {len(dc_result.flatten())} nodes, max IR-drop {max_ir_drop_dc*1e3:.3f} mV, {t_dc:.2f}s')

    # -- Step 2: Save DC context (for validation 3) then release --
    # We will save in validate_save_load, so just release here.
    dc_ctx.release()
    _check(not dc_ctx.is_factored, 'DC context released (is_factored=False)')

    # -- Step 3: Preprocess sources --
    t0 = time.perf_counter()
    smoothed = solver.preprocess_sources(
        time_step=dt, t_start=0.0, t_end=t_end, smooth=False,
    )
    t_sources = time.perf_counter() - t0
    print(f'  INFO  Source preprocessing: {t_sources:.2f}s')

    # -- Step 4: Prepare transient (builds new LU for A = G + C/dt) --
    t0 = time.perf_counter()
    trans_ctx = solver.prepare_transient(dt=dt, method='be')
    t_prep = time.perf_counter() - t0
    _check(trans_ctx.is_factored, 'Transient context is factored')
    print(f'  INFO  Transient prepare: {t_prep:.2f}s')

    # DC context is still released
    _check(not dc_ctx.is_factored, 'DC context still released while transient is active')

    # -- Step 5: Solve transient with ic_voltages path --
    t0 = time.perf_counter()
    trans_result_ic = solver.solve_transient(
        trans_ctx,
        ic_voltages=dc_result,  # Pass DistributedSolveResult directly (per-tile, no flattening)
        t_start=0.0,
        t_end=t_end,
        smoothed_sources=smoothed,
    )
    t_trans = time.perf_counter() - t0
    print(f'  INFO  Transient solve (ic_voltages path): {t_trans:.2f}s, '
          f'peak IR-drop {trans_result_ic.peak_ir_drop*1e3:.3f} mV at '
          f't={trans_result_ic.peak_ir_drop_time*1e12:.1f} ps')

    _check(
        trans_result_ic.peak_ir_drop >= 0,
        'Transient peak IR-drop is non-negative',
    )

    # -- Step 6: Run dc_context path for comparison (needs DC re-prepare) --
    #    Re-prepare DC so we can pass dc_context to solve_transient.
    dc_ctx2 = solver.prepare()
    _check(dc_ctx2.is_factored, 'Fresh DC context is factored for dc_context path')

    trans_result_dc = solver.solve_transient(
        trans_ctx,
        dc_context=dc_ctx2,
        t_start=0.0,
        t_end=t_end,
        smoothed_sources=smoothed,
    )

    # Compare peak IR-drop between both paths
    peak_diff = abs(trans_result_ic.peak_ir_drop - trans_result_dc.peak_ir_drop)
    _check(
        peak_diff < TOLERANCE,
        f'Both IC paths produce identical peak IR-drop (diff={peak_diff:.2e} V)',
        f'diff={peak_diff:.2e} V exceeds {TOLERANCE:.0e} V tolerance',
    )

    # Compare per-step max IR-drop arrays (relaxed tolerance: the two IC
    # paths use different numerical pathways for interior recovery, so FP
    # differences accumulate across transient steps).
    step_diff = np.max(np.abs(
        trans_result_ic.max_ir_drop_per_time - trans_result_dc.max_ir_drop_per_time
    ))
    _check(
        step_diff < TRANSIENT_IC_PATH_TOL,
        f'Per-step max IR-drop arrays match (max diff={step_diff:.2e} V)',
        f'max step diff={step_diff:.2e} V exceeds {TRANSIENT_IC_PATH_TOL:.0e} V tolerance',
    )

    # Clean up: release both contexts
    dc_ctx2.release()
    trans_ctx.release()
    _check(not trans_ctx.is_factored, 'Transient context released')

    return dc_result, trans_result_ic, trans_result_dc


# ===================================================================
# Validation 2: Flat solver comparison
# ===================================================================

def validate_flat_comparison(model, solver):
    """Compare distributed DDM DC vs flat DC solve."""
    _banner('Validation 2: Flat solver comparison (DDM DC vs flat DC)')

    # DDM
    t0 = time.perf_counter()
    dc_ctx = solver.prepare()
    ddm_result = solver.solve_dc(dc_ctx)
    ddm_voltages = ddm_result.flatten()
    t_ddm = time.perf_counter() - t0
    print(f'  INFO  DDM DC: {len(ddm_voltages)} nodes, {t_ddm:.2f}s')

    # Flat
    t0 = time.perf_counter()
    flat_model, flat_solver, load_currents = create_flat()
    flat_result = flat_solver.solve(load_currents)
    t_flat = time.perf_counter() - t0
    print(f'  INFO  Flat DC: {len(flat_result.voltages)} nodes, {t_flat:.2f}s')

    # Common nodes
    common = set(ddm_voltages.keys()) & set(flat_result.voltages.keys())
    print(f'  INFO  Common nodes: {len(common)}')
    _check(len(common) > 1000, f'Common nodes count ({len(common)}) > 1000')

    # Voltage comparison
    diffs = [abs(ddm_voltages[n] - flat_result.voltages[n]) for n in common]
    max_diff = max(diffs)
    mean_diff = np.mean(diffs)
    print(f'  INFO  Max voltage diff:  {max_diff*1e6:.3f} uV')
    print(f'  INFO  Mean voltage diff: {mean_diff*1e6:.3f} uV')

    _check(
        max_diff < TOLERANCE,
        f'Max voltage diff ({max_diff*1e6:.3f} uV) < 1 uV',
        f'max_diff={max_diff*1e6:.3f} uV',
    )

    # Worst node rankings
    vdd = model.vdd
    ddm_drops = sorted(
        [(n, vdd - ddm_voltages[n]) for n in common],
        key=lambda x: x[1], reverse=True,
    )
    flat_drops = sorted(
        [(n, vdd - flat_result.voltages[n]) for n in common],
        key=lambda x: x[1], reverse=True,
    )
    ddm_top10 = [n for n, _ in ddm_drops[:10]]
    flat_top10 = [n for n, _ in flat_drops[:10]]
    overlap = len(set(ddm_top10) & set(flat_top10))
    print(f'  INFO  Top-10 worst nodes overlap: {overlap}/10')
    _check(overlap >= 8, f'Top-10 worst nodes overlap ({overlap}/10) >= 8')

    dc_ctx.release()
    return ddm_voltages


# ===================================================================
# Validation 3: Save/load round-trip
# ===================================================================

def validate_save_load(model, solver):
    """Save -> release -> load -> refactor -> verify identical results."""
    _banner('Validation 3: Save/load round-trip')

    # -- Step 1: prepare + solve DC (reference) --
    dc_ctx = solver.prepare()
    ref_result = solver.solve_dc(dc_ctx)
    ref_voltages = ref_result.flatten()
    print(f'  INFO  Reference solve: {len(ref_voltages)} nodes')

    # -- Step 2: save context to temp file --
    tmpdir = tempfile.mkdtemp(prefix='ctx_lifecycle_')
    save_path = os.path.join(tmpdir, 'dc_context.pkl')
    actual_path = dc_ctx.save(path=save_path)
    _check(os.path.exists(actual_path), f'Checkpoint saved to {actual_path}')

    # -- Step 3: release (frees coordinator LU + worker LUs) --
    dc_ctx.release()
    _check(not dc_ctx.is_factored, 'DC context released after save')

    # -- Step 4: load from disk --
    from distributed.result import DistributedSolverContext
    loaded_ctx = DistributedSolverContext.load(model, actual_path)
    _check(not loaded_ctx.is_factored, 'Loaded context starts un-factored')
    _check(loaded_ctx.topology is not None, 'Loaded context has topology')

    # -- Step 5: refactor (rebuild coordinator LU from saved S_global) --
    #    NOTE: refactor only rebuilds coordinator LU. Workers need their
    #    block systems factored to do a solve. Since we released worker LUs
    #    in Step 3, we need a full factor() for a real solve.
    #    Let's test refactor first (just coordinator), then full factor.

    # Full factor: worker + coordinator, uses the loaded topology
    loaded_ctx.factor()
    _check(loaded_ctx.is_factored, 'Loaded context factored after factor()')

    # Solve and compare
    loaded_result = solver.solve_dc(loaded_ctx)
    loaded_voltages = loaded_result.flatten()

    common = set(ref_voltages.keys()) & set(loaded_voltages.keys())
    max_diff = max(abs(ref_voltages[n] - loaded_voltages[n]) for n in common)
    _check(
        max_diff < 1e-12,
        f'Load + factor produces identical results (max diff={max_diff:.2e} V)',
        f'max_diff={max_diff:.2e} exceeds tolerance',
    )
    loaded_ctx.release()

    # -- Step 6: prepare from scratch and compare --
    fresh_ctx = solver.prepare()
    fresh_result = solver.solve_dc(fresh_ctx)
    fresh_voltages = fresh_result.flatten()

    common2 = set(ref_voltages.keys()) & set(fresh_voltages.keys())
    max_diff2 = max(abs(ref_voltages[n] - fresh_voltages[n]) for n in common2)
    _check(
        max_diff2 < 1e-12,
        f'Fresh prepare produces identical results (max diff={max_diff2:.2e} V)',
        f'max_diff={max_diff2:.2e} exceeds tolerance',
    )
    fresh_ctx.release()

    # Clean up temp dir
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)
    print(f'  INFO  Cleaned up {tmpdir}')


# ===================================================================
# Validation 4: Both transient IC paths produce identical results
# ===================================================================

def validate_transient_ic_paths(model, solver):
    """Compare dc_context vs ic_voltages paths for transient IC."""
    _banner('Validation 4: Transient IC paths produce identical results')

    t_end = 2e-9
    dt = 100e-12

    # Preprocess sources once
    smoothed = solver.preprocess_sources(
        time_step=dt, t_start=0.0, t_end=t_end, smooth=False,
    )

    # -- Path A: dc_context path --
    dc_ctx = solver.prepare()
    trans_ctx = solver.prepare_transient(dt=dt, method='be')

    t0 = time.perf_counter()
    result_a = solver.solve_transient(
        trans_ctx,
        dc_context=dc_ctx,
        t_start=0.0,
        t_end=t_end,
        smoothed_sources=smoothed,
    )
    t_a = time.perf_counter() - t0
    print(f'  INFO  Path A (dc_context): peak={result_a.peak_ir_drop*1e3:.3f} mV, {t_a:.2f}s')

    # Extract IC voltages by solving DC separately
    dc_result = solver.solve_dc(dc_ctx)

    # Release DC context (not needed for Path B)
    dc_ctx.release()

    # -- Path B: ic_voltages path --
    # Need fresh transient context since transient state was modified by path A
    trans_ctx.release()
    trans_ctx_b = solver.prepare_transient(dt=dt, method='be')

    t0 = time.perf_counter()
    result_b = solver.solve_transient(
        trans_ctx_b,
        ic_voltages=dc_result,  # Pass DistributedSolveResult directly (per-tile, no flattening)
        t_start=0.0,
        t_end=t_end,
        smoothed_sources=smoothed,
    )
    t_b = time.perf_counter() - t0
    print(f'  INFO  Path B (ic_voltages): peak={result_b.peak_ir_drop*1e3:.3f} mV, {t_b:.2f}s')

    # Compare peaks (relaxed: different IC numerical paths)
    peak_diff = abs(result_a.peak_ir_drop - result_b.peak_ir_drop)
    _check(
        peak_diff < TRANSIENT_IC_PATH_TOL,
        f'Peak IR-drop matches (diff={peak_diff:.2e} V)',
        f'diff={peak_diff:.2e} V exceeds {TRANSIENT_IC_PATH_TOL:.0e} V tolerance',
    )

    # Compare per-step max IR-drop (relaxed: FP accumulation across steps)
    step_diff = np.max(np.abs(
        result_a.max_ir_drop_per_time - result_b.max_ir_drop_per_time
    ))
    _check(
        step_diff < TRANSIENT_IC_PATH_TOL,
        f'Per-step max IR-drop arrays match (max diff={step_diff:.2e} V)',
        f'max step diff={step_diff:.2e} exceeds {TRANSIENT_IC_PATH_TOL:.0e} V tolerance',
    )

    # Compare per-step total current (should be identical)
    current_diff = np.max(np.abs(
        result_a.total_current_per_time - result_b.total_current_per_time
    ))
    _check(
        current_diff < 1e-10,
        f'Per-step total current arrays match (max diff={current_diff:.2e} mA)',
        f'max current diff={current_diff:.2e} mA',
    )

    trans_ctx_b.release()


# ===================================================================
# Main
# ===================================================================

def main():
    t_start = time.perf_counter()
    print('Context Lifecycle Validation Script')
    print(f'Netlist: {NETLIST_SAMPLED_DIR}')
    print(f'PKL dir: {PKL_DIR}')

    # Pre-flight checks
    if not os.path.isdir(NETLIST_SAMPLED_DIR):
        _fail(f'Netlist directory not found: {NETLIST_SAMPLED_DIR}')
    if not os.path.isdir(PKL_DIR):
        _fail(f'PKL directory not found: {PKL_DIR}')

    model, solver = create_ddm()
    print(f'Model: {model.n_tiles} tiles, VDD={model.vdd:.3f} V, net={model.net_name}')

    try:
        validate_dc_release_transient(model, solver)
        validate_flat_comparison(model, solver)
        validate_save_load(model, solver)
        validate_transient_ic_paths(model, solver)
    finally:
        model.shutdown()

    t_total = time.perf_counter() - t_start

    _banner('ALL VALIDATIONS PASSED')
    print(f'  Total time: {t_total:.1f}s')


if __name__ == '__main__':
    main()

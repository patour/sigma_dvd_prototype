"""Shared fixtures for tests/validation/ equivalence tests.

Module-scoped fixtures amortize the heavy parse + factorization overhead.
All *unit* fixtures use the pre-parsed netlist_sampled/distributed_pkl
directory so no re-parsing is needed.

Design note on DistributedSmoothedSources
------------------------------------------
`preprocess_sources()` returns a lightweight coordinator-side handle whose
actual payload lives in each worker's `_active_sources`.  Calling
`preprocess_sources()` again overwrites that payload on all workers.  Tests
that re-run a distributed solve must therefore call `preprocess_sources()`
themselves immediately before the solve — they must NOT rely on a handle
stored in the fixture.  The fixture provides pre-computed FLAT results
(which are stateless) and the distributed model/contexts (whose
factorizations are reusable and read-only during a solve).

Design note on synthetic-fixture QS/transient
----------------------------------------------
The synthetic 2-tile model has no instance model files (instance_path=None),
so its current sources live only in TileData.current_injections (static DC
loads, not VCS).  `preprocess_sources()` on such a model initialises an
*empty* VCS on each worker, causing subsequent QS/transient steps to evaluate
zero currents — which differs from DC.

To avoid this, synthetic-fixture QS and transient tests bypass the internal
`preprocess_sources()` call by passing a pre-built dummy
`DistributedSmoothedSources` handle.  With a non-None handle the solver
skips the internal call, workers keep `_active_sources=None`, and
`evaluate_and_get_reduced_rhs` falls back to the static DC loads from
`current_injections`.  This ensures QS ≡ DC at each step (constant loads),
and BE/TR transient at DC initial condition ≡ DC forever (standard RC result
for constant forcing from DC steady state).
"""

import logging
import warnings
from pathlib import Path

import pytest

# ── Canonical paths ────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).parent.parent.parent
_PKL_DIR = _PROJECT_ROOT / "netlist" / "netlist_sampled" / "distributed_pkl"
_SAMPLED_DIR = _PROJECT_ROOT / "netlist" / "netlist_sampled"
_NETLIST_TEST_DIR = _PROJECT_ROOT / "netlist" / "netlist_test"

PKL_DIR_EXISTS = _PKL_DIR.is_dir()
SAMPLED_DIR_EXISTS = _SAMPLED_DIR.is_dir()
NETLIST_TEST_EXISTS = _NETLIST_TEST_DIR.is_dir()


# ── Lightweight bundle ─────────────────────────────────────────────────

class _Setup:
    """Attribute bag for fixture-yielded bundles."""
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


# ══════════════════════════════════════════════════════════════════════
# Fixture 1: netlist_sampled / distributed_pkl (unit + integration)
# ══════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def sampled_setup(request):
    """Build both distributed (pre-parsed pkl) and flat solvers for
    netlist_sampled.

    Provides
    --------
    Distributed:
        dist_model, dist_solver — model and solver instances
        dc_ctx  — factored DC context (read-only during solve; released in teardown)

    Flat (pre-computed, stateless — safe to read in any test order):
        flat_model, flat_graph, flat_dyn, flat_trans
        flat_sm    — pre-smoothed VCS (used for both BE and TR flat solves)
        flat_dc_v  — Dict[node, voltage]
        flat_qs    — QuasiStaticResult (raw sources, no smoothing)
        flat_be_result, flat_tr_result — TransientResult

    Timing parameters (used in both flat pre-computation and each test):
        dt, t_start, t_end, n_points

    NOTE: the fixture does NOT pre-compute distributed QS or transient
    results because those require `preprocess_sources()` which mutates
    per-worker `_active_sources`.  Each test that needs a distributed QS
    or transient result must call `preprocess_sources()` itself before
    the solve to guarantee correct worker state.

    Skipped when distributed_pkl or netlist_sampled are missing.
    """
    if not PKL_DIR_EXISTS:
        pytest.skip("netlist_sampled/distributed_pkl not available")
    if not SAMPLED_DIR_EXISTS:
        pytest.skip("netlist_sampled not available")

    logging.disable(logging.WARNING)
    warnings.filterwarnings("ignore")

    # ── Distributed model + contexts ───────────────────────────────────
    from distributed.model import load_distributed_partitions, create_distributed_model
    from distributed.solver import DistributedDDMSolver

    bundle = load_distributed_partitions(str(_PKL_DIR))
    dist_model = create_distributed_model(bundle, backend="local")
    dist_solver = DistributedDDMSolver(dist_model)

    # Time-domain parameters (short window: fits 60 s unit budget)
    dt = 1e-9       # 1 ns time step
    t_start = 0.0
    t_end = 3e-9    # 3 ns → 3 transient steps
    n_points = 5    # QS: 5 batch-DC points

    # DC context — read-only during solves
    dc_ctx = dist_solver.prepare(verbose=False)
    dist_dc_result = dist_solver.solve_dc(dc_ctx, verbose=False)
    dist_dc_v = dist_dc_result.flatten()

    # ── Flat solver (all results are stateless dicts/arrays) ───────────
    from parser.netlist import NetlistParser
    from model.factory import create_model_from_pdn
    from solver.unified_solver import UnifiedIRDropSolver
    from analysis.dynamic_solver import DynamicIRDropSolver
    from analysis.transient_solver import TransientIRDropSolver, IntegrationMethod

    flat_parser = NetlistParser(str(_SAMPLED_DIR))
    flat_graph = flat_parser.parse()
    flat_model = create_model_from_pdn(flat_graph, "VDD_XLV")
    flat_solver = UnifiedIRDropSolver(flat_model)
    flat_load = flat_model.extract_current_sources()
    flat_dc_result = flat_solver.solve(flat_load)
    flat_dc_v = flat_dc_result.voltages

    # Flat QS — raw sources (no smoothed_sources passed → uses VCS directly)
    flat_dyn = DynamicIRDropSolver(flat_model, flat_graph)
    flat_qs = flat_dyn.solve_quasi_static(
        t_start=t_start,
        t_end=t_end,
        n_points=n_points,
        method="flat",
    )

    # Flat transient — raw (unsmoothed) sources to match distributed smooth=False.
    # flat_sm is still computed and kept for adjoint tests (analyze_victim needs it).
    # flat_be_result / flat_tr_result use raw VCS (no smoothed_sources argument)
    # so that both sides of the comparison use identical raw sources, matching
    # the QS discipline.  BE peak diff on sampled is ~3.6e-11 V with raw=both.
    flat_trans = TransientIRDropSolver(flat_model, flat_graph)
    flat_sm = flat_trans.preprocess_sources(dt=dt, t_start=t_start, t_end=t_end)
    flat_be_result = flat_trans.solve_transient(
        t_start=t_start, t_end=t_end, dt=dt,
        method=IntegrationMethod.BACKWARD_EULER,
        # No smoothed_sources: uses raw VCS to match distributed smooth=False
    )
    flat_tr_result = flat_trans.solve_transient(
        t_start=t_start, t_end=t_end, dt=dt,
        method=IntegrationMethod.TRAPEZOIDAL,
        # No smoothed_sources: uses raw VCS to match distributed smooth=False
    )

    logging.disable(logging.NOTSET)

    s = _Setup(
        # distributed
        dist_model=dist_model,
        dist_solver=dist_solver,
        dc_ctx=dc_ctx,
        dist_dc_v=dist_dc_v,
        # flat (stateless results)
        flat_model=flat_model,
        flat_graph=flat_graph,
        flat_solver=flat_solver,
        flat_dyn=flat_dyn,
        flat_trans=flat_trans,
        flat_sm=flat_sm,
        flat_dc_v=flat_dc_v,
        flat_qs=flat_qs,
        flat_be_result=flat_be_result,
        flat_tr_result=flat_tr_result,
        # params
        dt=dt,
        t_start=t_start,
        t_end=t_end,
        n_points=n_points,
    )

    yield s

    # Teardown — release DC context, then model.
    # Transient contexts are managed by individual tests (_run_dist_transient).
    # No save() needed: tests do not checkpoint.
    try:
        dc_ctx.release()
    except Exception:
        pass
    dist_model.shutdown()


# ══════════════════════════════════════════════════════════════════════
# Fixture 2: netlist_test (integration)
# ══════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def netlist_test_setup(request):
    """Build both distributed and flat solvers for netlist_test.

    Uses a short 2-step time window (t_end=2 ns, dt=1 ns, n_points=3) to
    keep the fixture fast while exercising all solve paths.

    Provides
    --------
    Distributed:
        dist_model, dist_solver — model and solver instances
        dc_ctx  — factored DC context (read-only during solve; released in teardown)
        dist_dc_v — Dict[node, voltage]

    Flat (pre-computed, stateless):
        flat_model, flat_graph, flat_solver, flat_dyn, flat_trans
        flat_sm    — pre-smoothed VCS
        flat_dc_v  — Dict[node, voltage]
        flat_qs    — QuasiStaticResult (raw sources)
        flat_be_result, flat_tr_result — TransientResult

    victim_node: interior node with VCS current sources (for adjoint tests)

    Timing parameters:
        dt, t_start, t_end, n_points

    NOTE: individual tests calling distributed QS or transient must call
    preprocess_sources() themselves (same design as sampled_setup).

    Skipped when netlist_test is missing.
    """
    if not NETLIST_TEST_EXISTS:
        pytest.skip("netlist_test not available")

    logging.disable(logging.WARNING)
    warnings.filterwarnings("ignore")

    # ── Distributed model + DC context ────────────────────────────────
    from distributed.parser import DistributedNetlistParser
    from distributed.model import create_distributed_model
    from distributed.solver import DistributedDDMSolver

    dist_parser = DistributedNetlistParser(str(_NETLIST_TEST_DIR), net_filter="VDD")
    metadata = dist_parser.parse_metadata()
    dist_model = create_distributed_model(metadata, backend="local")
    dist_solver = DistributedDDMSolver(dist_model)
    dc_ctx = dist_solver.prepare(verbose=False)
    dist_dc_result = dist_solver.solve_dc(dc_ctx, verbose=False)
    dist_dc_v = dist_dc_result.flatten()

    # ── Flat model + DC solve ─────────────────────────────────────────
    from parser.netlist import NetlistParser
    from model.factory import create_model_from_pdn
    from solver.unified_solver import UnifiedIRDropSolver
    from analysis.dynamic_solver import DynamicIRDropSolver
    from analysis.transient_solver import TransientIRDropSolver, IntegrationMethod

    flat_parser = NetlistParser(str(_NETLIST_TEST_DIR))
    flat_graph = flat_parser.parse()
    flat_model = create_model_from_pdn(flat_graph, "VDD")
    flat_solver = UnifiedIRDropSolver(flat_model)
    flat_load = flat_model.extract_current_sources()
    flat_dc_result = flat_solver.solve(flat_load)
    flat_dc_v = flat_dc_result.voltages

    # Timing params — short window, fast for integration budget
    dt = 1e-9
    t_start = 0.0
    t_end = 2e-9       # 2 ns → 2 time steps
    n_points = 3

    # Flat QS (raw sources — no smoothed_sources → raw VCS)
    flat_dyn = DynamicIRDropSolver(flat_model, flat_graph)
    flat_qs = flat_dyn.solve_quasi_static(
        t_start=t_start,
        t_end=t_end,
        n_points=n_points,
        method="flat",
    )

    # Flat transient — raw (unsmoothed) sources to match distributed smooth=False.
    # vectorize_threshold=0: force VCS even for small netlists (netlist_test
    # has only ~17 sources, below the default 10000 threshold).
    # flat_sm is still kept for adjoint tests; flat_be_result / flat_tr_result
    # use raw VCS (no smoothed_sources) so both sides use identical raw sources.
    flat_trans = TransientIRDropSolver(flat_model, flat_graph, vectorize_threshold=0)
    flat_sm = flat_trans.preprocess_sources(dt=dt, t_start=t_start, t_end=t_end)
    flat_be_result = flat_trans.solve_transient(
        t_start=t_start, t_end=t_end, dt=dt,
        method=IntegrationMethod.BACKWARD_EULER,
        # No smoothed_sources: raw VCS to match distributed smooth=False
    )
    flat_tr_result = flat_trans.solve_transient(
        t_start=t_start, t_end=t_end, dt=dt,
        method=IntegrationMethod.TRAPEZOIDAL,
        # No smoothed_sources: raw VCS to match distributed smooth=False
    )

    # ── Victim node for adjoint tests ─────────────────────────────────
    # Pick an interior node (not interface/pad) that has VCS current sources.
    from distributed.tile_parsing import _iter_instance_sources

    interface_set = set(dc_ctx.interface_nodes)
    nodes_with_sources: set = set()
    for tc in dist_model.metadata.tile_configs:
        if tc.instance_path is None:
            continue
        for prep in _iter_instance_sources(tc.instance_path, tc.net_filter, tc.nd_path):
            n = prep.cs.node1
            if n and n != "0":
                nodes_with_sources.add(n)
    interior_sources = sorted(
        n for n in nodes_with_sources
        if n not in interface_set and not n.endswith("_vsrc")
    )
    assert interior_sources, (
        "No interior nodes with VCS current sources found in netlist_test; "
        "check node naming and net filter — adjoint tests require at least one "
        "interior node with instance sources"
    )
    victim_node = interior_sources[0]

    logging.disable(logging.NOTSET)

    s = _Setup(
        # distributed
        dist_model=dist_model,
        dist_solver=dist_solver,
        dc_ctx=dc_ctx,
        dist_dc_v=dist_dc_v,
        # flat (stateless results)
        flat_model=flat_model,
        flat_graph=flat_graph,
        flat_solver=flat_solver,
        flat_dyn=flat_dyn,
        flat_trans=flat_trans,
        flat_sm=flat_sm,
        flat_dc_v=flat_dc_v,
        flat_qs=flat_qs,
        flat_be_result=flat_be_result,
        flat_tr_result=flat_tr_result,
        # adjoint
        victim_node=victim_node,
        # params
        dt=dt,
        t_start=t_start,
        t_end=t_end,
        n_points=n_points,
    )

    yield s

    try:
        dc_ctx.release()
    except Exception:
        pass
    dist_model.shutdown()

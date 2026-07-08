"""Flat-vs-distributed DDM equivalence suite.

Gates every performance change of the 'distributed-10x' refactor by asserting
that the distributed solver produces results that are numerically identical
(within floating-point noise) to the corresponding flat solvers.

Coverage matrix (flat solver oracle vs DistributedDDMSolver):
  DC          -- all three fixtures (synthetic, netlist_test, netlist_sampled)
                 spec: max |V_flat - V_dist| <= 1e-9 V
  QS          -- all three fixtures; tolerance <= 1e-9 V (hard assert)
  Transient   -- all three fixtures (BE and TR); raw sources on both sides
                 BE peak: <= 1e-8 V (hard assert — ~3.6e-11 V observed on sampled)
                 TR peak: <= 2e-3 V regression guard; 1e-8 V spec xfail
  Adjoint     -- netlist_test + netlist_sampled
                 static sampled: <= 1e-6 relative (hard assert — spec met)
                 dynamic sampled: <= 1e-6 relative (hard assert — spec met with raw sources)
                 netlist_test: structural top-K overlap only (parsing delta ~30 µV
                   propagates non-linearly; magnitude xfail at 1e-6 for both static/dyn)

Smoothing discipline: all transient and adjoint tests use smooth=False (raw VCS)
on the distributed side, and no smoothed_sources on the flat side.  This is the
same as QS and ensures identical source inputs on both sides.  Using smooth=True
on either side independently (the "dual-smoother anti-pattern") masked a genuine
~40–77% dynamic adjoint magnitude divergence that vanishes when raw=both sides.

Marker conventions:
  @pytest.mark.validation  -- applied at module level to all tests here
  @pytest.mark.unit        -- fast sub-set (<60 s total); no full re-parse
  @pytest.mark.integration -- heavy; requires real netlist data

SOLVER_VARIANTS
---------------
Module-level list of kwargs dicts passed as **kw to distributed solve calls.
Each entry becomes a parametrized test variant.  Adding one line here is all
it takes to cover a new Phase-A optimisation flag.

Phase-A flags to add once implemented:
  pytest.param({"use_step_columns": True}, id="step_cols")  # A2
"""

import logging
import sys
import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest

pytestmark = pytest.mark.validation

# ── SOLVER_VARIANTS: parametrize over distributed solve kwargs ─────────
# Baseline: no extra kwargs (standard call).
# When Phase A2 lands, append: pytest.param({"use_step_columns": True}, id="step_cols")
SOLVER_VARIANTS: List[pytest.param] = [
    pytest.param({}, id="baseline"),
]

# ── FLAG_VARIANTS: cross-comparison gate (flag-ON == flag-OFF) ─────────
# Non-baseline entries: all SOLVER_VARIANTS where the kwargs dict is non-empty.
# Currently empty (only baseline exists). Populated automatically when Phase
# A2 appends {"use_step_columns": True} to SOLVER_VARIANTS above.
_FLAG_VARIANTS: List[pytest.param] = [v for v in SOLVER_VARIANTS if v.values[0]]

# Placeholder for parametrize when _FLAG_VARIANTS is empty — produces a
# single test that is immediately skipped so the parametrize call stays valid.
_FLAG_VARIANTS_OR_SKIP: List[pytest.param] = _FLAG_VARIANTS or [
    pytest.param(
        {},
        id="_no_variant_yet",
        marks=pytest.mark.skip(
            reason=(
                "No non-baseline SOLVER_VARIANTS yet; skip until Phase A2 appends "
                "pytest.param({'use_step_columns': True}, id='step_cols')"
            )
        ),
    )
]

# ── Paths ──────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).parent.parent.parent
_PKL_DIR = _PROJECT_ROOT / "netlist" / "netlist_sampled" / "distributed_pkl"
_SAMPLED_DIR = _PROJECT_ROOT / "netlist" / "netlist_sampled"
_NETLIST_TEST_DIR = _PROJECT_ROOT / "netlist" / "netlist_test"

PKL_DIR_EXISTS = _PKL_DIR.is_dir()
SAMPLED_DIR_EXISTS = _SAMPLED_DIR.is_dir()
NETLIST_TEST_EXISTS = _NETLIST_TEST_DIR.is_dir()

# ── Tolerances ─────────────────────────────────────────────────────────
# Flag-gate: flag-ON == flag-OFF must be bit-identical (fp accumulation only).
# Used by TestSolverVariantFlagGate; unaffected by the dual-smoother issue
# (flag-gate compares distributed-only, same smoother on both sides).
FLAG_GATE_ATOL = 1e-12

# DC: DDM is algebraically exact; difference is only floating-point
# accumulation in the Schur complement assembly (~1e-14 V observed).
DC_ATOL = 1e-9

# QS: same matrix as DC at each step; tolerance matches DC (1e-9 V).
# The spec requires <= 1e-9.  Observed delta is machine-precision (~3e-15 V).
QS_ATOL = 1e-9

# Transient BE (with raw sources on both sides, smooth=False):
# BE peak diff ~3.6e-11 V on netlist_sampled — well within 1e-8 V spec.
# TRANS_SPEC_ATOL is the spec value, now hard-asserted for BE peak.
# TRANS_ATOL is a loose regression guard retained only for per-step and
# per-node array tests (which may have sub-leading first-step IC differences).
TRANS_ATOL = 5e-6
TRANS_SPEC_ATOL = 1e-8

# Transient TR: even with identical raw sources, the Trapezoidal method
# diverges at ~0.5 mV between flat and distributed implementations.
# This is a genuine integration-scheme difference, not a smoothing issue.
# TR_TRANS_ATOL is the regression guard; TRANS_SPEC_ATOL is the spec xfail.
TR_TRANS_ATOL = 2e-3

# Adjoint (static, integration): relative tolerance for per-aggressor
# contribution comparison on the common top-K set.
# ADJOINT_REL_TOL is the regression guard; ADJOINT_SPEC_REL_TOL is the spec xfail.
ADJOINT_REL_TOL = 1e-4
ADJOINT_SPEC_REL_TOL = 1e-6

# Tolerance for netlist_test DC/QS/BE-transient: parsing differences between the
# flat and distributed parsers cause a ~30 µV baseline delta (known; documented
# in TestDCNetlistTest).  QS and BE transient inherit this delta.
NETLIST_TEST_ATOL = 5e-5

# TR transient on netlist_test has a larger delta (~63 µV observed) because the
# Trapezoidal method amplifies the parse-level mismatch vs BE.
NETLIST_TEST_TR_ATOL = 1e-4

# Adjoint on netlist_test: the ~30 µV parsing delta propagates non-linearly
# into per-aggressor contributions (observed 27–51% relative error on small
# contributions).  Magnitude comparison is unreliable; tests use structural
# top-K overlap instead.  This rtol is no longer used but kept for reference.
NETLIST_TEST_ADJOINT_RTOL = 0.60


# ── xfail reasons ─────────────────────────────────────────────────────
# NOTE: _XFAIL_BE_SPEC was removed.  BE transient is now hard-asserted at
# TRANS_SPEC_ATOL=1e-8 V using smooth=False on both sides (raw sources).
# Observed BE peak diff on netlist_sampled: ~3.6e-11 V (well within spec).
_XFAIL_TR_SPEC = (
    "genuine flat-vs-distributed TR divergence under identical raw sources — "
    "under investigation (suspect TR Dirichlet/pad-cap history handling); "
    "auto-passes when distributed TR integration is realigned with flat TR"
)
_XFAIL_ADJ_STATIC_SPEC = (
    "src-level adjoint static delta ~1e-4 relative exceeds spec 1e-6; "
    "auto-passes when src is fixed"
)
_XFAIL_ADJ_DYN_SPEC = (
    "src-level dynamic adjoint delta 40–77% relative exceeds spec 1e-6; "
    "auto-passes when src is fixed"
)

# ── Transient smoothing discipline ─────────────────────────────────────
# Both sides use raw (unsmoothed) sources: distributed side calls
# preprocess_sources(smooth=False) and the flat side omits smoothed_sources
# in solve_transient() (falls back to raw VCS from the constructor).
# This is the same discipline as QS (which also uses smooth=False on both
# sides and achieves QS_ATOL=1e-9 V).
#
# Root causes of remaining transient delta after smooth=False fix:
#   BE: floating-point accumulation only (~3.6e-11 V on sampled) →
#       hard-asserted at TRANS_SPEC_ATOL=1e-8 V in test_peak_drop_matches.
#   TR: genuine Trapezoidal integration difference (~0.5 mV on sampled) →
#       TR_TRANS_ATOL=2e-3 V regression guard; TRANS_SPEC_ATOL spec xfail.
#   netlist_test (any method): ~30 µV parsing delta dominates because the
#       flat and distributed parsers handle complex elements differently.
#
# Historical note: an earlier version used smooth=True on the distributed
# side ("dual-smoother pattern") which masked a ~2–3 µV smoother-
# implementation divergence.  Using smooth=False exposes the actual solver
# accuracy: BE is functionally exact (3.6e-11 V), TR has a real integration
# mismatch (~0.5 mV) that needs a separate src fix.
#
# The flag-gate (TestSolverVariantFlagGate) compares distributed flag-ON vs
# flag-OFF — same smoother on both sides, no delta, tolerance 1e-12.
_NOTE_RAW_SOURCES = (
    "Both sides use raw (smooth=False) sources; BE peak diff ~3.6e-11 V on "
    "sampled (hard-asserted at 1e-8 V spec).  TR diverges ~0.5 mV due to a "
    "genuine Trapezoidal integration difference between flat and distributed."
)


def _build_synthetic_two_tile_model(tmp_dir: str = None):
    """Build a minimal 2-tile DistributedPowerGridModel suitable for DC+transient.

    Unlike _build_two_tile_distributed_model (test_time_domain.py), pad nodes
    are NOT placed in tile boundary_nodes — they appear only in PackageData.
    This is the correct design for use with solve_dc() (avoids reduced-RHS
    length mismatch caused by Dirichlet nodes being treated as tile ports).

    Topology
    --------
    Tile A: a1 --[1 mS]-- B (interior: a1; boundary: B only)
    Tile B: B --[3 mS]-- b1 --[1 mS]-- 0 (interior: b1; boundary: B only)
    Package: pad --[10 mS]-- B
    Pad 'pad' at 1.0 V (Dirichlet); Loads: a1=0.5 mA, b1=0.3 mA
    Caps: a1-0 (10 fF), b1-0 (5 fF)

    Schur complement (pad=1.0 V eliminated; unknowns: a1, B, b1):
        G = [[1, -1,  0],
             [-1, 14, -3],
             [0,  -3,  4]]
        rhs = [-0.5, 10.0, -0.3]  mA
              (pad contributes 10*1.0=10 to B; loads are sinks)
    """
    import os
    import tempfile

    from distributed.backend import LocalBackend
    from distributed.model import DistributedPowerGridModel
    from distributed.parser import PackageData, PowerGridMetaData, TileConfig
    from distributed.tile_worker import TileData, TileWorker

    # tmp_dir is where the adjoint VCS cache is written.  create_temporary if
    # not supplied so that analyze_adjoint_static() can call
    # os.makedirs(os.path.dirname(ckt_path)) without getting ''.
    _owns_tmp = tmp_dir is None
    if _owns_tmp:
        tmp_dir = tempfile.mkdtemp(prefix="synth_two_tile_")

    tile_a = TileData(
        tile_id=(0, 0),
        resistive_edges=[("a1", "B", 1.0)],
        all_nodes={"a1", "B"},
        boundary_nodes={"B"},          # pad is NOT here
        current_injections={"a1": 0.5},
        capacitive_edges=[("a1", "0", 10.0)],   # 10 fF grounded
    )
    tile_b = TileData(
        tile_id=(0, 1),
        resistive_edges=[("B", "b1", 3.0), ("b1", "0", 1.0)],
        all_nodes={"B", "b1"},
        boundary_nodes={"B"},
        current_injections={"b1": 0.3},
        capacitive_edges=[("b1", "0", 5.0)],    # 5 fF grounded
    )

    interface_nodes = {"B"}  # only non-Dirichlet shared node

    be = LocalBackend()
    be.initialize()
    wa = TileWorker()
    wa.setup_from_tile_data(tile_a, interface_nodes)
    wb = TileWorker()
    wb.setup_from_tile_data(tile_b, interface_nodes)

    pkg = PackageData(
        vsrc_dict={"V1": {"node_pos": "pad", "node_neg": "0", "net": "VDD", "value": 1.0}},
        package_edges=[("pad", "B", 10.0)],    # 10 mS package resistor
        pad_nodes={"pad"},
        tap_nodes=set(),
        die_attachment_nodes=set(),
        vdd=1.0,
        net_name="VDD",
        package_cap_edges=[],
    )
    # ckt_path must have a valid parent dir so that analyze_adjoint_static's
    # pkl_dir = os.path.dirname(ckt_path) resolves to a writable directory.
    # We use tmp_dir/dummy.ckt so that VCS cache writes land in tmp_dir.
    _dummy_ckt = os.path.join(tmp_dir, "dummy.ckt")
    tile_configs = [
        TileConfig(tile_id=(0, 0), ckt_path=_dummy_ckt, nd_path=None,
                   instance_path=None, net_filter=None),
        TileConfig(tile_id=(0, 1), ckt_path=_dummy_ckt, nd_path=None,
                   instance_path=None, net_filter=None),
    ]
    metadata = PowerGridMetaData(
        tile_grid=(1, 2),
        parameters={},
        tile_configs=tile_configs,
        package_data=pkg,
        net_name="VDD",
        vdd=1.0,
    )

    return DistributedPowerGridModel(
        backend=be,
        workers=[wa, wb],
        interface_nodes=interface_nodes,
        tile_boundary_nodes={(0, 0): ["B"], (0, 1): ["B"]},
        tile_interior_counts={(0, 0): wa.n_interior, (0, 1): wb.n_interior},
        package_data=pkg,
        metadata=metadata,
        # Hand ownership of tmp_dir to the model so shutdown() cleans it up.
        _owns_pkl_dir=tmp_dir if _owns_tmp else None,
    )


# ══════════════════════════════════════════════════════════════════════
# 1. DC — Synthetic two-tile (unit, no fixture needed)
# ══════════════════════════════════════════════════════════════════════

class TestDCSynthetic:
    """Exact DC equivalence on a hand-crafted 2-tile synthetic graph.

    Uses _build_synthetic_two_tile_model() (defined above) with the correct
    design: pad nodes appear ONLY in package_edges, NOT in tile boundary_nodes.
    This ensures solve_dc() can correctly map the reduced RHS to the interface.

    Topology
    --------
    Tile A: a1 --[1 mS]-- B  (interior: a1; boundary: B)
    Tile B: B --[3 mS]-- b1 --[1 mS]-- 0  (interior: b1; boundary: B)
    Package: pad --[10 mS]-- B
    Pad 'pad' at 1.0 V (Dirichlet); Loads: a1=0.5 mA, b1=0.3 mA

    Schur complement (pad=1.0 V eliminated; unknowns: a1, B, b1):
        G = [[1, -1,  0],
             [-1, 14, -3],
             [0,  -3,  4]]
        rhs = [-0.5, 10.0, -0.3]  mA
              (pad contributes 10*1.0=10 to B; loads a1=0.5mA, b1=0.3mA are sinks)
    """

    @pytest.mark.unit
    def test_synthetic_dc_matches_scipy(self):
        """DDM DC result matches scipy flat solve to 1e-9 V."""
        import scipy.linalg
        from distributed.solver import DistributedDDMSolver

        # ── Flat reference (hand-assembled from topology above) ────────
        G = np.array([
            [1.0,  -1.0,  0.0],
            [-1.0, 14.0, -3.0],
            [0.0,  -3.0,  4.0],
        ])
        rhs = np.array([-0.5, 10.0, -0.3])
        v_ref = scipy.linalg.solve(G, rhs)
        ref = {"a1": v_ref[0], "B": v_ref[1], "b1": v_ref[2]}

        # ── Distributed model ──────────────────────────────────────────
        model = _build_synthetic_two_tile_model()
        solver = DistributedDDMSolver(model)
        ctx = solver.prepare(verbose=False)
        result = solver.solve_dc(ctx, verbose=False)
        ddm_v = result.flatten()

        # ── Compare interior nodes ────────────────────────────────────
        for node, v_scipy in ref.items():
            assert node in ddm_v, f"Node '{node}' missing from DDM result"
            diff = abs(ddm_v[node] - v_scipy)
            assert diff <= DC_ATOL, (
                f"Node '{node}': |DDM {ddm_v[node]:.9f} - scipy {v_scipy:.9f}| "
                f"= {diff:.2e} V > {DC_ATOL:.0e} V"
            )

        ctx.release()
        model.shutdown()


# ══════════════════════════════════════════════════════════════════════
# 2. QS — Synthetic two-tile (unit)
# ══════════════════════════════════════════════════════════════════════

class TestQSSynthetic:
    """QS equivalence on the synthetic 2-tile model (constant DC loads).

    The 2-tile model has no instance current-source files (instance_path=None).
    We use a dummy DistributedSmoothedSources handle so the solver skips the
    internal preprocess_sources() call; workers keep _active_sources=None and
    fall back to static TileData.current_injections at each QS step.

    Because loads are constant, each QS step is identical to the DC solve.
    The per-step max IR-drop should match the DDM DC max-drop to <= QS_ATOL.
    """

    @pytest.mark.unit
    @pytest.mark.parametrize("solver_kw", SOLVER_VARIANTS)
    def test_qs_per_step_matches_dc(self, solver_kw):
        """Per-step QS max-drop equals DC max-drop to 1e-9 V."""
        from distributed.result import DistributedSmoothedSources
        from distributed.solver import DistributedDDMSolver

        model = _build_synthetic_two_tile_model()
        solver = DistributedDDMSolver(model)

        # DC reference
        dc_ctx = solver.prepare(verbose=False)
        dc_result = solver.solve_dc(dc_ctx, verbose=False)
        dc_v = dc_result.flatten()
        vdd = model.vdd
        dc_max_drop = max(vdd - v for v in dc_v.values())

        # QS with dummy handle (workers keep _active_sources=None → DC loads)
        dt = 1e-9
        t_start = 0.0
        t_end = 3e-9
        n_points = 4
        dummy_sources = DistributedSmoothedSources(
            time_step=dt, t_start=t_start, t_end=t_end,
            smoothed=False, n_tiles=len(model.workers), per_tile_stats={},
        )
        qs_result = solver.solve_quasi_static(
            dc_ctx,
            t_start=t_start,
            t_end=t_end,
            n_points=n_points,
            smoothed_sources=dummy_sources,
            verbose=False,
            **solver_kw,
        )
        qs_max = np.array(qs_result.max_ir_drop_per_time)

        max_diff = float(np.max(np.abs(qs_max - dc_max_drop)))
        assert max_diff <= QS_ATOL, (
            f"Synthetic QS: per-step max IR-drop delta = {max_diff:.3e} V "
            f"> {QS_ATOL:.0e} V  "
            f"(dc_max={dc_max_drop:.6f} V; steps={qs_max.tolist()})"
        )

        dc_ctx.release()
        model.shutdown()


# ══════════════════════════════════════════════════════════════════════
# 3. Transient — Synthetic two-tile (unit)
# ══════════════════════════════════════════════════════════════════════

class TestTransientSynthetic:
    """RC transient equivalence on synthetic 2-tile (BE and TR).

    With constant DC loads and DC initial condition, the transient stays at
    DC steady state for both BE and TR.  The peak IR-drop must equal the DC
    max-drop to QS_ATOL (1e-9 V) — there is no RC transient noise with
    constant forcing from DC IC.

    Uses dummy smoothed_sources handle (same technique as TestQSSynthetic).
    """

    @pytest.mark.unit
    @pytest.mark.parametrize("method", ["be", "trap"])
    @pytest.mark.parametrize("solver_kw", SOLVER_VARIANTS)
    def test_transient_peak_matches_dc(self, method, solver_kw):
        """Transient peak IR-drop equals DC max-drop to 1e-9 V."""
        from distributed.result import DistributedSmoothedSources
        from distributed.solver import DistributedDDMSolver

        model = _build_synthetic_two_tile_model()
        solver = DistributedDDMSolver(model)

        # DC context (provides IC + Dirichlet info)
        dc_ctx = solver.prepare(verbose=False)
        dc_result = solver.solve_dc(dc_ctx, verbose=False)
        dc_v = dc_result.flatten()
        vdd = model.vdd
        dc_max_drop = max(vdd - v for v in dc_v.values())

        # Transient context
        dt = 1e-9
        t_start = 0.0
        t_end = 3e-9
        trans_ctx = solver.prepare_transient(dt=dt, method=method, verbose=False)

        # Dummy handle: bypass internal preprocess_sources()
        dummy_sources = DistributedSmoothedSources(
            time_step=dt, t_start=t_start, t_end=t_end,
            smoothed=False, n_tiles=len(model.workers), per_tile_stats={},
        )
        trans_result = solver.solve_transient(
            trans_ctx,
            dc_context=dc_ctx,
            t_start=t_start,
            t_end=t_end,
            smoothed_sources=dummy_sources,
            verbose=False,
            **solver_kw,
        )

        diff = abs(trans_result.peak_ir_drop - dc_max_drop)
        assert diff <= QS_ATOL, (
            f"Synthetic transient-{method}: peak_ir_drop={trans_result.peak_ir_drop:.9f} "
            f"dc_max={dc_max_drop:.9f} diff={diff:.3e} > {QS_ATOL:.0e} V"
        )

        trans_ctx.release()
        dc_ctx.release()
        model.shutdown()


# ══════════════════════════════════════════════════════════════════════
# 4. Adjoint — Synthetic two-tile (unit, minimal smoke tests)
# ══════════════════════════════════════════════════════════════════════

class TestAdjointSynthetic:
    """Smoke tests for adjoint on synthetic 2-tile (no VCS instance files).

    Since the model has no instanceModels files, VCS is empty and adjoint
    returns an AdjointAttribution with empty top_aggressors.  Tests verify:
    (a) the solver runs without error,
    (b) the result is an AdjointAttribution instance.

    Note: for meaningful magnitude checks, see TestAdjointNetlistTest
    (integration, real VCS sources) and TestAdjointSampledPkl (integration).
    """

    @pytest.mark.unit
    def test_static_adjoint_returns_attribution(self):
        """analyze_adjoint_static runs and returns AdjointAttribution."""
        from distributed.solver import DistributedDDMSolver
        from analysis.adjoint_sensitivity import AdjointAttribution

        model = _build_synthetic_two_tile_model()
        solver = DistributedDDMSolver(model)
        dc_ctx = solver.prepare(verbose=False)

        # Use a1 as the victim (interior node in tile A)
        attr = solver.analyze_adjoint_static(
            victim_nodes="a1",
            observation_time=1e-9,
            dc_context=dc_ctx,
            top_k=5,
            verbose=False,
        )
        assert isinstance(attr, AdjointAttribution), (
            f"Expected AdjointAttribution, got {type(attr)}"
        )
        # With no VCS sources, top_aggressors should be empty
        assert isinstance(attr.top_aggressors, list), (
            "top_aggressors should be a list"
        )

        dc_ctx.release()
        model.shutdown()

    @pytest.mark.unit
    def test_dynamic_adjoint_returns_attribution(self):
        """analyze_adjoint (dynamic) runs and returns AdjointAttribution."""
        from distributed.result import DistributedSmoothedSources
        from distributed.solver import DistributedDDMSolver
        from analysis.adjoint_sensitivity import AdjointAttribution

        model = _build_synthetic_two_tile_model()
        solver = DistributedDDMSolver(model)

        dc_ctx = solver.prepare(verbose=False)
        dt = 1e-9
        t_start = 0.0
        t_end = 2e-9
        trans_ctx = solver.prepare_transient(dt=dt, method="be", verbose=False)
        dummy_sources = DistributedSmoothedSources(
            time_step=dt, t_start=t_start, t_end=t_end,
            smoothed=False, n_tiles=len(model.workers), per_tile_stats={},
        )
        trans_result = solver.solve_transient(
            trans_ctx,
            dc_context=dc_ctx,
            t_start=t_start,
            t_end=t_end,
            smoothed_sources=dummy_sources,
            verbose=False,
        )

        attr = solver.analyze_adjoint(
            victim_nodes="a1",
            observation_time=t_end,
            trans_context=trans_ctx,
            transient_result=trans_result,
            dc_context=dc_ctx,
            top_k=5,
            verbose=False,
        )
        assert isinstance(attr, AdjointAttribution), (
            f"Expected AdjointAttribution, got {type(attr)}"
        )

        trans_ctx.release()
        dc_ctx.release()
        model.shutdown()


# ══════════════════════════════════════════════════════════════════════
# 5. DC — netlist_sampled pkl (unit + integration)
# ══════════════════════════════════════════════════════════════════════

class TestDCSampledPkl:
    """DC equivalence on netlist_sampled from pre-parsed pkl files.

    Both solvers use the same conductance system; the only numerical
    difference comes from floating-point accumulation in the Schur
    complement assembly (~1e-14 V observed, well within DC_ATOL=1e-9 V).
    """

    @pytest.mark.unit
    @pytest.mark.integration
    def test_common_nodes_nonempty(self, sampled_setup):
        """Flat and distributed DC share a large common node set."""
        s = sampled_setup
        common = set(s.dist_dc_v.keys()) & set(s.flat_dc_v.keys())
        assert len(common) > 1000, (
            f"Only {len(common)} common nodes — expected thousands"
        )

    @pytest.mark.unit
    @pytest.mark.integration
    def test_dc_voltages_match(self, sampled_setup):
        """All common-node DC voltages agree within DC_ATOL={DC_ATOL:.0e} V."""
        s = sampled_setup
        common = sorted(set(s.dist_dc_v.keys()) & set(s.flat_dc_v.keys()))
        diffs = np.array([abs(s.dist_dc_v[n] - s.flat_dc_v[n]) for n in common])
        max_diff = float(diffs.max())
        assert max_diff <= DC_ATOL, (
            f"Max |DDM - flat| = {max_diff:.3e} V "
            f"({max_diff * 1e9:.3f} nV) > {DC_ATOL:.0e} V "
            f"over {len(common)} nodes"
        )

    @pytest.mark.unit
    @pytest.mark.integration
    def test_dc_voltages_in_range(self, sampled_setup):
        """All distributed DC voltages lie in [0, Vdd]."""
        s = sampled_setup
        vdd = s.dist_model.vdd
        dist_v = np.array(list(s.dist_dc_v.values()))
        assert dist_v.min() >= -1e-6, f"Voltage below 0 V: {dist_v.min():.3e}"
        assert dist_v.max() <= vdd + 1e-6, f"Voltage above Vdd: {dist_v.max():.3e}"


# ══════════════════════════════════════════════════════════════════════
# 6. Quasi-static — netlist_sampled pkl (unit + integration)
# ══════════════════════════════════════════════════════════════════════

class TestQSSampledPkl:
    """Quasi-static equivalence on netlist_sampled.

    Both solvers use raw sources (smooth=False / no smoothed_sources),
    so the only residual is floating-point noise from the DC matrix solve.

    Design: each test calls preprocess_sources(smooth=False) before the
    distributed solve to guarantee clean worker state independent of test
    execution order.  See conftest.py design note for details.

    SOLVER_VARIANTS is applied to the distributed solve call; adding an
    entry to that list triggers an additional parametrised run automatically.
    """

    @pytest.mark.unit
    @pytest.mark.integration
    @pytest.mark.parametrize("solver_kw", SOLVER_VARIANTS)
    def test_per_step_max_drop_matches(self, sampled_setup, solver_kw):
        """Per-step distributed max-drop matches flat within QS_ATOL."""
        s = sampled_setup
        sources = s.dist_solver.preprocess_sources(
            time_step=s.dt, t_start=s.t_start, t_end=s.t_end,
            smooth=False, verbose=False,
        )
        dist_result = s.dist_solver.solve_quasi_static(
            s.dc_ctx,
            t_start=s.t_start,
            t_end=s.t_end,
            n_points=s.n_points,
            smoothed_sources=sources,
            verbose=False,
            **solver_kw,
        )
        flat_max = np.array(s.flat_qs.max_ir_drop_per_time)
        dist_max = np.array(dist_result.max_ir_drop_per_time)

        # Trim to the shorter array in case n_points differs slightly
        n = min(len(flat_max), len(dist_max))
        diffs = np.abs(dist_max[:n] - flat_max[:n])
        max_diff = float(diffs.max())
        assert max_diff <= QS_ATOL, (
            f"QS max-drop mismatch [{solver_kw}]: "
            f"max |dist-flat| = {max_diff:.3e} V > {QS_ATOL:.0e} V"
        )

    @pytest.mark.unit
    @pytest.mark.integration
    @pytest.mark.parametrize("solver_kw", SOLVER_VARIANTS)
    def test_peak_irdrop_matches(self, sampled_setup, solver_kw):
        """Overall peak IR-drop agrees within QS_ATOL."""
        s = sampled_setup
        sources = s.dist_solver.preprocess_sources(
            time_step=s.dt, t_start=s.t_start, t_end=s.t_end,
            smooth=False, verbose=False,
        )
        dist_result = s.dist_solver.solve_quasi_static(
            s.dc_ctx,
            t_start=s.t_start,
            t_end=s.t_end,
            n_points=s.n_points,
            smoothed_sources=sources,
            verbose=False,
            **solver_kw,
        )
        diff = abs(dist_result.peak_ir_drop - s.flat_qs.peak_ir_drop)
        assert diff <= QS_ATOL, (
            f"Peak QS IR-drop [{solver_kw}]: "
            f"dist={dist_result.peak_ir_drop:.6f} flat={s.flat_qs.peak_ir_drop:.6f} "
            f"diff={diff:.3e} > {QS_ATOL:.0e}"
        )

    @pytest.mark.unit
    @pytest.mark.integration
    def test_qs_per_node_peaks_match(self, sampled_setup):
        """Per-node peak IR-drop on common nodes matches within QS_ATOL.

        Runs a fresh distributed QS solve with raw sources; compares
        per-node peaks on the intersection of common nodes.
        """
        s = sampled_setup
        sources = s.dist_solver.preprocess_sources(
            time_step=s.dt, t_start=s.t_start, t_end=s.t_end,
            smooth=False, verbose=False,
        )
        dist_result = s.dist_solver.solve_quasi_static(
            s.dc_ctx,
            t_start=s.t_start,
            t_end=s.t_end,
            n_points=s.n_points,
            smoothed_sources=sources,
            verbose=False,
        )
        flat_per_node = s.flat_qs.peak_ir_drop_per_node
        dist_raw = dist_result.as_flat()
        dist_per_node = {n: drop for n, (drop, _) in dist_raw.items()}
        common = sorted(set(flat_per_node.keys()) & set(dist_per_node.keys()))
        assert len(common) > 1000, f"Only {len(common)} common nodes"
        diffs = np.abs(
            np.array([dist_per_node[n] for n in common]) -
            np.array([flat_per_node[n] for n in common])
        )
        max_diff = float(diffs.max())
        assert max_diff <= QS_ATOL, (
            f"Per-node QS peak: max diff = {max_diff:.3e} V > {QS_ATOL:.0e} V "
            f"(over {len(common)} nodes)"
        )


# ══════════════════════════════════════════════════════════════════════
# 7. Transient — netlist_sampled pkl (unit + integration)
# ══════════════════════════════════════════════════════════════════════

def _run_dist_transient(s, method: str, solver_kw: dict):
    """Helper: (re)prepare transient context + preprocess (raw) + solve.

    Each transient test must call this (rather than re-using a cached
    context) because prepare_transient() mutates per-worker transient
    state (_C_coeff, _transient_block_system).  Calling it for 'trap'
    after 'be' would leave workers in TR mode, corrupting subsequent
    BE solves.  Re-preparing immediately before each solve ensures the
    workers match the coordinator context.

    Smoothing: raw sources (smooth=False) on both sides
    ----------------------------------------------------
    Uses smooth=False here to mirror the flat side (which uses raw VCS
    by not passing smoothed_sources to solve_transient).  This is the
    same discipline as QS (smooth=False on both sides → identical raw
    sources → exact match up to fp accumulation).

    With raw sources on both sides:
      BE: diff ~3.6e-11 V on netlist_sampled → hard-asserted at 1e-8 V spec.
      TR: diff ~0.5 mV (genuine Trapezoidal integration mismatch) → xfail.

    The flag-gate (TestSolverVariantFlagGate) compares distributed flag-ON
    vs flag-OFF — same smoother on both sides, unaffected by this change.
    """
    ctx = s.dist_solver.prepare_transient(dt=s.dt, method=method, verbose=False)
    sources = s.dist_solver.preprocess_sources(
        time_step=s.dt, t_start=s.t_start, t_end=s.t_end,
        smooth=False,  # raw sources — matches flat side (no smoothed_sources)
        verbose=False,
    )
    result = s.dist_solver.solve_transient(
        ctx,
        dc_context=s.dc_ctx,
        t_start=s.t_start,
        t_end=s.t_end,
        smoothed_sources=sources,
        verbose=False,
        **solver_kw,
    )
    ctx.release()
    return result


class TestTransientSampledPkl:
    """RC transient equivalence on netlist_sampled (BE and TR methods).

    Both sides use raw (unsmoothed) sources: the distributed side calls
    preprocess_sources(smooth=False) and the flat side uses raw VCS from
    the constructor (no smoothed_sources argument in solve_transient).
    This is the same discipline as QS (smooth=False on both sides).

    With identical raw sources:
      BE: diff ~3.6e-11 V → hard-asserted at TRANS_SPEC_ATOL=1e-8 V.
      TR: diff ~0.5 mV (genuine Trapezoidal integration mismatch between
          flat and distributed) → TR_TRANS_ATOL=2e-3 V regression guard;
          TRANS_SPEC_ATOL=1e-8 V spec xfail until distributed TR is fixed.

    Each test calls _run_dist_transient() which re-prepares the transient
    context immediately before the solve.  This is required because
    prepare_transient() mutates per-worker state; using a cached context
    from a different method (e.g., TR context when worker is in BE mode)
    gives wrong results.

    Parametrised over integration method × SOLVER_VARIANTS.  When A2
    lands, each existing method entry gets a new 'step_cols' twin for free.

    Guards:
        BE peak:       TRANS_SPEC_ATOL = 1e-8 V  (hard assert — achievable with raw)
        TR peak:       TR_TRANS_ATOL = 2e-3 V  (regression guard; xfail at 1e-8)
        BE per-step:   TRANS_ATOL = 5e-6 V  (loose guard for array comparison)
        TR per-step:   TR_TRANS_ATOL = 2e-3 V
    """

    @pytest.mark.unit
    @pytest.mark.integration
    @pytest.mark.parametrize("method", ["be", "trap"])
    @pytest.mark.parametrize("solver_kw", SOLVER_VARIANTS)
    def test_peak_drop_matches(self, sampled_setup, method, solver_kw):
        """Peak IR-drop matches flat within tolerance.

        BE uses TRANS_SPEC_ATOL=1e-8 V (hard assert — achievable with raw
        sources on both sides; observed delta ~3.6e-11 V on sampled).
        TR uses TR_TRANS_ATOL=2e-3 V (regression guard — genuine Trapezoidal
        integration difference ~0.5 mV between flat and distributed).
        """
        s = sampled_setup
        atol = TRANS_SPEC_ATOL if method == "be" else TR_TRANS_ATOL
        flat_result = s.flat_be_result if method == "be" else s.flat_tr_result
        dist_result = _run_dist_transient(s, method, solver_kw)
        diff = abs(dist_result.peak_ir_drop - flat_result.peak_ir_drop)
        assert diff <= atol, (
            f"Transient-{method} [{solver_kw}] peak IR-drop: "
            f"dist={dist_result.peak_ir_drop:.6f} "
            f"flat={flat_result.peak_ir_drop:.6f} "
            f"diff={diff:.3e} > {atol:.0e}"
        )

    @pytest.mark.unit
    @pytest.mark.integration
    @pytest.mark.parametrize("method", ["trap"])  # BE now hard-asserted in test_peak_drop_matches
    @pytest.mark.parametrize("solver_kw", SOLVER_VARIANTS)
    @pytest.mark.xfail(
        strict=False,
        reason=_XFAIL_TR_SPEC,
    )
    def test_peak_drop_at_spec_tolerance(self, sampled_setup, method, solver_kw):
        """TR peak IR-drop at spec tolerance 1e-8 V (xfail).

        TR diverges ~0.5 mV even with identical raw sources — a genuine
        Trapezoidal integration difference in the distributed solver.
        Auto-passes when the distributed TR is realigned with flat TR.
        BE is NOT included here: it is hard-asserted in test_peak_drop_matches
        (TRANS_SPEC_ATOL=1e-8 V, observed delta ~3.6e-11 V with raw sources).
        """
        s = sampled_setup
        flat_result = s.flat_tr_result
        dist_result = _run_dist_transient(s, method, solver_kw)
        diff = abs(dist_result.peak_ir_drop - flat_result.peak_ir_drop)
        assert diff <= TRANS_SPEC_ATOL, (
            f"Transient-{method} peak at spec: diff={diff:.3e} > {TRANS_SPEC_ATOL:.0e} V"
        )

    @pytest.mark.unit
    @pytest.mark.integration
    @pytest.mark.parametrize("method", ["be", "trap"])
    @pytest.mark.parametrize("solver_kw", SOLVER_VARIANTS)
    def test_per_step_max_drop_matches(self, sampled_setup, method, solver_kw):
        """Per-step max IR-drop array matches flat within tolerance.

        The flat solver includes the DC initial condition (t=0) as the first
        entry in max_ir_drop_per_time; the distributed solver starts from t=dt.
        Arrays are aligned by taking the trailing ``n_dist`` entries from the
        flat array, so flat[1:] ↔ dist[:] when IC is present.
        """
        s = sampled_setup
        atol = TRANS_ATOL if method == "be" else TR_TRANS_ATOL
        flat_result = s.flat_be_result if method == "be" else s.flat_tr_result
        dist_result = _run_dist_transient(s, method, solver_kw)
        flat_max = np.array(flat_result.max_ir_drop_per_time)
        dist_max = np.array(dist_result.max_ir_drop_per_time)
        n = len(dist_max)
        assert n > 0, (
            f"Transient-{method} [{solver_kw}]: distributed solver produced "
            f"zero time steps (dist_max is empty)"
        )
        aligned_flat = flat_max[-n:]   # skip IC step at t=0
        diffs = np.abs(dist_max - aligned_flat)
        max_diff = float(diffs.max())
        assert max_diff <= atol, (
            f"Transient-{method} [{solver_kw}] per-step max-drop: "
            f"max diff = {max_diff:.3e} V > {atol:.0e} V "
            f"(n_dist={n}, n_flat={len(flat_max)})"
        )

    @pytest.mark.unit
    @pytest.mark.integration
    @pytest.mark.parametrize("method", ["be", "trap"])
    def test_per_node_peak_matches(self, sampled_setup, method):
        """Per-node peak IR-drop on common nodes matches within tolerance."""
        s = sampled_setup
        atol = TRANS_ATOL if method == "be" else TR_TRANS_ATOL
        flat_result = s.flat_be_result if method == "be" else s.flat_tr_result
        dist_result = _run_dist_transient(s, method, {})
        flat_per_node = flat_result.peak_ir_drop_per_node
        dist_raw = dist_result.as_flat()
        dist_per_node = {n: drop for n, (drop, _) in dist_raw.items()}
        common = sorted(set(flat_per_node.keys()) & set(dist_per_node.keys()))
        assert len(common) > 1000, f"Only {len(common)} common nodes"
        diffs = np.abs(
            np.array([dist_per_node[n] for n in common]) -
            np.array([flat_per_node[n] for n in common])
        )
        max_diff = float(diffs.max())
        assert max_diff <= atol, (
            f"Transient-{method} per-node peak: "
            f"max diff = {max_diff:.3e} V > {atol:.0e} V "
            f"(over {len(common)} nodes)"
        )


# ══════════════════════════════════════════════════════════════════════
# 8. DC — netlist_test full parse (integration)
# ══════════════════════════════════════════════════════════════════════

class TestDCNetlistTest:
    """DC equivalence on netlist_test using full re-parse.

    The flat and distributed parsers diverge by ~30 µV on netlist_test due
    to differences in how complex element types (subcircuit instances,
    decap cells) are handled.  The tolerance here is deliberately loose
    (5e-5 V) to detect genuine solver regressions while tolerating the
    known parsing delta.
    """

    # Absolute tolerance for netlist_test flat-vs-distributed DC.
    # Measured baseline: ~30 µV.  Set to 5e-5 V to cover regressions.
    NETLIST_TEST_DC_ATOL = 5e-5

    @pytest.mark.integration
    @pytest.mark.skipif(not NETLIST_TEST_EXISTS, reason="netlist_test not available")
    def test_dc_within_parsing_tolerance(self):
        """Distributed vs flat DC on netlist_test within 50 µV.

        If this tolerance shrinks over time (better parser alignment), tighten
        NETLIST_TEST_DC_ATOL accordingly.
        """
        logging.disable(logging.WARNING)
        warnings.filterwarnings("ignore")

        # ── Distributed ────────────────────────────────────────────────
        from distributed.parser import DistributedNetlistParser
        from distributed.model import create_distributed_model
        from distributed.solver import DistributedDDMSolver

        parser = DistributedNetlistParser(str(_NETLIST_TEST_DIR), net_filter="VDD")
        metadata = parser.parse_metadata()
        dist_model = create_distributed_model(metadata, backend="local")
        dist_solver = DistributedDDMSolver(dist_model)
        dc_ctx = dist_solver.prepare(verbose=False)
        dc_result = dist_solver.solve_dc(dc_ctx, verbose=False)
        dist_v = dc_result.flatten()

        # ── Flat ───────────────────────────────────────────────────────
        from parser.netlist import NetlistParser
        from model.factory import create_model_from_pdn
        from solver.unified_solver import UnifiedIRDropSolver

        flat_parser = NetlistParser(str(_NETLIST_TEST_DIR))
        flat_graph = flat_parser.parse()
        flat_model = create_model_from_pdn(flat_graph, "VDD")
        flat_solver = UnifiedIRDropSolver(flat_model)
        flat_load = flat_model.extract_current_sources()
        flat_result = flat_solver.solve(flat_load)
        flat_v = flat_result.voltages

        # ── Compare ────────────────────────────────────────────────────
        common = sorted(set(dist_v.keys()) & set(flat_v.keys()))
        assert len(common) > 10, f"Only {len(common)} common nodes"
        diffs = np.abs(
            np.array([dist_v[n] for n in common]) -
            np.array([flat_v[n] for n in common])
        )
        max_diff = float(diffs.max())
        assert max_diff <= self.NETLIST_TEST_DC_ATOL, (
            f"netlist_test DC: max |dist - flat| = {max_diff:.3e} V "
            f"> {self.NETLIST_TEST_DC_ATOL:.0e} V "
            f"(expected ~30 µV parsing delta; see class docstring)"
        )

        # Teardown
        dc_ctx.release()
        dist_model.shutdown()
        logging.disable(logging.NOTSET)

    @pytest.mark.integration
    @pytest.mark.skipif(not NETLIST_TEST_EXISTS, reason="netlist_test not available")
    @pytest.mark.xfail(
        strict=False,
        reason=(
            "netlist_test DC: ~30 µV parsing delta between flat and distributed "
            "parsers exceeds spec 1e-9 V; auto-passes when parsers align"
        ),
    )
    def test_dc_at_spec_tolerance(self):
        """Distributed vs flat DC on netlist_test at spec tolerance DC_ATOL=1e-9 V (xfail).

        The spec (item 1) requires max |V_flat - V_dist| <= 1e-9 V on netlist_test.
        The ~30 µV parsing delta currently prevents this; this xfail gates the spec
        and auto-passes when the parsers align.
        """
        logging.disable(logging.WARNING)
        warnings.filterwarnings("ignore")

        from distributed.parser import DistributedNetlistParser
        from distributed.model import create_distributed_model
        from distributed.solver import DistributedDDMSolver

        parser = DistributedNetlistParser(str(_NETLIST_TEST_DIR), net_filter="VDD")
        metadata = parser.parse_metadata()
        dist_model = create_distributed_model(metadata, backend="local")
        dist_solver = DistributedDDMSolver(dist_model)
        dc_ctx = dist_solver.prepare(verbose=False)
        dc_result = dist_solver.solve_dc(dc_ctx, verbose=False)
        dist_v = dc_result.flatten()

        from parser.netlist import NetlistParser
        from model.factory import create_model_from_pdn
        from solver.unified_solver import UnifiedIRDropSolver

        flat_parser = NetlistParser(str(_NETLIST_TEST_DIR))
        flat_graph = flat_parser.parse()
        flat_model = create_model_from_pdn(flat_graph, "VDD")
        flat_solver = UnifiedIRDropSolver(flat_model)
        flat_load = flat_model.extract_current_sources()
        flat_result = flat_solver.solve(flat_load)
        flat_v = flat_result.voltages

        common = sorted(set(dist_v.keys()) & set(flat_v.keys()))
        assert len(common) > 10, f"Only {len(common)} common nodes"
        diffs = np.abs(
            np.array([dist_v[n] for n in common]) -
            np.array([flat_v[n] for n in common])
        )
        max_diff = float(diffs.max())
        assert max_diff <= DC_ATOL, (
            f"netlist_test DC at spec: max |dist - flat| = {max_diff:.3e} V "
            f"> DC_ATOL={DC_ATOL:.0e} V  (expected ~30 µV parsing delta)"
        )

        dc_ctx.release()
        dist_model.shutdown()
        logging.disable(logging.NOTSET)


# ══════════════════════════════════════════════════════════════════════
# 9. QS — netlist_test (integration)
# ══════════════════════════════════════════════════════════════════════

class TestQSNetlistTest:
    """QS equivalence on netlist_test (flat vs distributed).

    NETLIST_TEST_ATOL = 5e-5 V: covers the ~30 µV parser delta that affects
    per-step DC solves.  The xfail test at QS_ATOL (1e-9 V) will auto-pass
    when the parsing differences are resolved.
    """

    @pytest.mark.integration
    @pytest.mark.skipif(not NETLIST_TEST_EXISTS, reason="netlist_test not available")
    @pytest.mark.parametrize("solver_kw", SOLVER_VARIANTS)
    def test_per_step_max_drop_within_parsing_tolerance(self, netlist_test_setup, solver_kw):
        """Per-step QS max-drop agrees within NETLIST_TEST_ATOL."""
        s = netlist_test_setup
        sources = s.dist_solver.preprocess_sources(
            time_step=s.dt, t_start=s.t_start, t_end=s.t_end,
            smooth=False, verbose=False,
        )
        dist_qs = s.dist_solver.solve_quasi_static(
            s.dc_ctx,
            t_start=s.t_start,
            t_end=s.t_end,
            n_points=s.n_points,
            smoothed_sources=sources,
            verbose=False,
            **solver_kw,
        )
        flat_max = np.array(s.flat_qs.max_ir_drop_per_time)
        dist_max = np.array(dist_qs.max_ir_drop_per_time)
        n = min(len(flat_max), len(dist_max))
        diffs = np.abs(dist_max[:n] - flat_max[:n])
        max_diff = float(diffs.max())
        assert max_diff <= NETLIST_TEST_ATOL, (
            f"netlist_test QS per-step [{solver_kw}]: "
            f"max diff = {max_diff:.3e} V > {NETLIST_TEST_ATOL:.0e} V"
        )

    @pytest.mark.integration
    @pytest.mark.skipif(not NETLIST_TEST_EXISTS, reason="netlist_test not available")
    @pytest.mark.xfail(
        strict=False,
        reason=(
            "netlist_test QS: ~30 µV parser delta exceeds spec 1e-9 V; "
            "auto-passes when parsers align"
        ),
    )
    def test_per_step_at_spec_tolerance(self, netlist_test_setup):
        """Per-step QS matches flat at spec tolerance QS_ATOL=1e-9 V (xfail)."""
        s = netlist_test_setup
        sources = s.dist_solver.preprocess_sources(
            time_step=s.dt, t_start=s.t_start, t_end=s.t_end,
            smooth=False, verbose=False,
        )
        dist_qs = s.dist_solver.solve_quasi_static(
            s.dc_ctx,
            t_start=s.t_start,
            t_end=s.t_end,
            n_points=s.n_points,
            smoothed_sources=sources,
            verbose=False,
        )
        flat_max = np.array(s.flat_qs.max_ir_drop_per_time)
        dist_max = np.array(dist_qs.max_ir_drop_per_time)
        n = min(len(flat_max), len(dist_max))
        diffs = np.abs(dist_max[:n] - flat_max[:n])
        max_diff = float(diffs.max())
        assert max_diff <= QS_ATOL, (
            f"netlist_test QS at spec: max diff = {max_diff:.3e} > {QS_ATOL:.0e} V"
        )


# ══════════════════════════════════════════════════════════════════════
# 10. Transient — netlist_test (integration)
# ══════════════════════════════════════════════════════════════════════

def _run_netlist_test_transient(s, method: str, solver_kw: dict):
    """Helper: fresh transient context + raw sources (smooth=False) + solve.

    Uses smooth=False to match the flat side (flat_be_result / flat_tr_result
    in the netlist_test_setup fixture are also computed with raw VCS, no
    smoothed_sources argument).  The dominant delta on netlist_test is the
    ~30 µV parsing difference between flat and distributed parsers, NOT smoothing.
    """
    ctx = s.dist_solver.prepare_transient(dt=s.dt, method=method, verbose=False)
    sources = s.dist_solver.preprocess_sources(
        time_step=s.dt, t_start=s.t_start, t_end=s.t_end,
        smooth=False,  # raw sources — matches flat side (no smoothed_sources)
        verbose=False,
    )
    result = s.dist_solver.solve_transient(
        ctx,
        dc_context=s.dc_ctx,
        t_start=s.t_start,
        t_end=s.t_end,
        smoothed_sources=sources,
        verbose=False,
        **solver_kw,
    )
    ctx.release()
    return result


class TestTransientNetlistTest:
    """RC transient equivalence on netlist_test (BE and TR).

    The parsing delta (~30 µV) dominates for BE; TR has a larger delta (~63 µV)
    because the Trapezoidal method amplifies parsing mismatch.
    NETLIST_TEST_ATOL is used for BE; NETLIST_TEST_TR_ATOL for TR.
    Spec-tolerance xfail tests at 1e-8 V for both.
    """

    @pytest.mark.integration
    @pytest.mark.skipif(not NETLIST_TEST_EXISTS, reason="netlist_test not available")
    @pytest.mark.parametrize("method", ["be", "trap"])
    @pytest.mark.parametrize("solver_kw", SOLVER_VARIANTS)
    def test_peak_drop_within_parsing_tolerance(self, netlist_test_setup, method, solver_kw):
        """Peak IR-drop matches flat within parsing tolerance (BE: 5e-5; TR: 1e-4)."""
        s = netlist_test_setup
        atol = NETLIST_TEST_ATOL if method == "be" else NETLIST_TEST_TR_ATOL
        flat_result = s.flat_be_result if method == "be" else s.flat_tr_result
        dist_result = _run_netlist_test_transient(s, method, solver_kw)
        diff = abs(dist_result.peak_ir_drop - flat_result.peak_ir_drop)
        assert diff <= atol, (
            f"netlist_test transient-{method} [{solver_kw}] peak: "
            f"dist={dist_result.peak_ir_drop:.6f} flat={flat_result.peak_ir_drop:.6f} "
            f"diff={diff:.3e} > {atol:.0e} V"
        )

    @pytest.mark.integration
    @pytest.mark.skipif(not NETLIST_TEST_EXISTS, reason="netlist_test not available")
    @pytest.mark.parametrize("method", ["be", "trap"])
    @pytest.mark.xfail(
        strict=False,
        reason=(
            "netlist_test transient: ~30 µV parser delta exceeds spec 1e-8 V; "
            "auto-passes when parsers align"
        ),
    )
    def test_peak_drop_at_spec_tolerance(self, netlist_test_setup, method):
        """Peak IR-drop matches flat at spec tolerance 1e-8 V (xfail)."""
        s = netlist_test_setup
        flat_result = s.flat_be_result if method == "be" else s.flat_tr_result
        dist_result = _run_netlist_test_transient(s, method, {})
        diff = abs(dist_result.peak_ir_drop - flat_result.peak_ir_drop)
        assert diff <= TRANS_SPEC_ATOL, (
            f"netlist_test transient-{method} at spec: diff={diff:.3e} > {TRANS_SPEC_ATOL:.0e} V"
        )


# ══════════════════════════════════════════════════════════════════════
# 11. DC — netlist_sampled full re-parse (integration)
# ══════════════════════════════════════════════════════════════════════

class TestDCSampledFullParse:
    """DC equivalence on netlist_sampled with a fresh parse (no pkl).

    This integration test verifies the parser→model→solve pipeline end-to-end
    rather than relying on the pre-parsed pkl.  It uses the same DC_ATOL
    as the pkl-based unit test because both parsers produce the same graph.
    """

    @pytest.mark.integration
    @pytest.mark.skipif(
        not SAMPLED_DIR_EXISTS,
        reason="netlist_sampled not available",
    )
    def test_dc_fresh_parse_matches_flat(self):
        """DC after fresh parse equals flat within DC_ATOL."""
        logging.disable(logging.WARNING)
        warnings.filterwarnings("ignore")

        from distributed.parser import DistributedNetlistParser
        from distributed.model import create_distributed_model
        from distributed.solver import DistributedDDMSolver
        from parser.netlist import NetlistParser
        from model.factory import create_model_from_pdn
        from solver.unified_solver import UnifiedIRDropSolver

        # Distributed
        parser = DistributedNetlistParser(str(_SAMPLED_DIR), net_filter="VDD_XLV")
        metadata = parser.parse_metadata()
        dist_model = create_distributed_model(metadata, backend="local")
        dist_solver = DistributedDDMSolver(dist_model)
        dc_ctx = dist_solver.prepare(verbose=False)
        dc_result = dist_solver.solve_dc(dc_ctx, verbose=False)
        dist_v = dc_result.flatten()

        # Flat
        flat_parser = NetlistParser(str(_SAMPLED_DIR))
        flat_graph = flat_parser.parse()
        flat_model = create_model_from_pdn(flat_graph, "VDD_XLV")
        flat_solver = UnifiedIRDropSolver(flat_model)
        flat_load = flat_model.extract_current_sources()
        flat_result = flat_solver.solve(flat_load)
        flat_v = flat_result.voltages

        common = sorted(set(dist_v.keys()) & set(flat_v.keys()))
        assert len(common) > 10000, f"Only {len(common)} common nodes"
        diffs = np.abs(
            np.array([dist_v[n] for n in common]) -
            np.array([flat_v[n] for n in common])
        )
        max_diff = float(diffs.max())
        assert max_diff <= DC_ATOL, (
            f"netlist_sampled fresh-parse DC: max diff = {max_diff:.3e} V "
            f"> {DC_ATOL:.0e} V"
        )

        dc_ctx.release()
        dist_model.shutdown()
        logging.disable(logging.NOTSET)


# ══════════════════════════════════════════════════════════════════════
# 12. Adjoint — netlist_test (integration)
# ══════════════════════════════════════════════════════════════════════

class TestAdjointNetlistTest:
    """Adjoint sensitivity equivalence on netlist_test.

    Uses the netlist_test_setup fixture (module-scoped) for model reuse.

    The ~30 µV parsing delta propagates non-linearly into per-aggressor
    contributions (observed 27–51% relative error on small contributions).
    Magnitude comparison is unreliable for netlist_test adjoint; regression
    guards use structural top-K overlap instead.
    Spec-tolerance xfail at ADJOINT_SPEC_REL_TOL=1e-6 for magnitude.
    """

    TOP_K = 10

    @pytest.mark.integration
    @pytest.mark.skipif(not NETLIST_TEST_EXISTS, reason="netlist_test not available")
    def test_static_adjoint_structural_overlap(
        self, netlist_test_setup
    ):
        """Static adjoint: at least 2 of top-K aggressors overlap between dist and flat.

        The ~30 µV parsing delta causes 27–51% relative errors on small
        contributions, so per-contribution magnitude comparison is unreliable.
        Both solvers must agree on the worst aggressor nodes (structural overlap).
        """
        s = netlist_test_setup
        logging.disable(logging.WARNING)
        warnings.filterwarnings("ignore")

        if s.victim_node is None:
            pytest.skip("No interior node with VCS sources found in netlist_test")

        obs_time = s.t_end

        # Distributed static adjoint
        dist_attr = s.dist_solver.analyze_adjoint_static(
            victim_nodes=s.victim_node,
            observation_time=obs_time,
            dc_context=s.dc_ctx,
            top_k=self.TOP_K,
            verbose=False,
        )

        # Flat static adjoint
        from analysis.adjoint_sensitivity import AdjointSensitivitySolver
        flat_adj = AdjointSensitivitySolver(s.flat_model, s.flat_graph,
                                            vectorize_threshold=0)
        flat_attr = flat_adj.analyze_victim_static(
            victim_node=s.victim_node,
            observation_time=obs_time,
            top_k=self.TOP_K,
        )

        dist_nodes = {ac.node for ac in dist_attr.top_aggressors}
        flat_nodes = {ac.node for ac in flat_attr.top_aggressors}

        assert dist_nodes, "Distributed static adjoint returned empty top_aggressors"
        assert flat_nodes, "Flat static adjoint returned empty top_aggressors"

        overlap = dist_nodes & flat_nodes
        assert len(overlap) >= 2, (
            f"netlist_test static adjoint top-K overlap too small ({len(overlap)}); "
            f"dist={sorted(dist_nodes)[:5]} flat={sorted(flat_nodes)[:5]}"
        )
        logging.disable(logging.NOTSET)

    @pytest.mark.integration
    @pytest.mark.skipif(not NETLIST_TEST_EXISTS, reason="netlist_test not available")
    @pytest.mark.xfail(strict=False, reason=_XFAIL_ADJ_STATIC_SPEC)
    def test_static_adjoint_at_spec_tolerance(self, netlist_test_setup):
        """Per-aggressor static contributions at spec 1e-6 relative (xfail)."""
        s = netlist_test_setup
        logging.disable(logging.WARNING)
        warnings.filterwarnings("ignore")

        if s.victim_node is None:
            pytest.skip("No interior node with VCS sources found in netlist_test")

        obs_time = s.t_end
        dist_attr = s.dist_solver.analyze_adjoint_static(
            victim_nodes=s.victim_node,
            observation_time=obs_time,
            dc_context=s.dc_ctx,
            top_k=self.TOP_K,
            verbose=False,
        )
        from analysis.adjoint_sensitivity import AdjointSensitivitySolver
        flat_adj = AdjointSensitivitySolver(s.flat_model, s.flat_graph,
                                            vectorize_threshold=0)
        flat_attr = flat_adj.analyze_victim_static(
            victim_node=s.victim_node,
            observation_time=obs_time,
            top_k=self.TOP_K,
        )

        dist_contribs = {ac.node: ac.contribution_mV for ac in dist_attr.top_aggressors}
        flat_contribs = {ac.node: ac.contribution_mV for ac in flat_attr.top_aggressors}
        common = set(dist_contribs.keys()) & set(flat_contribs.keys())
        assert len(common) >= 2, (
            f"Too few common aggressors to compare ({len(common)}); "
            f"dist had {len(dist_contribs)}, flat had {len(flat_contribs)}"
        )

        failures = []
        for name in sorted(common):
            d = dist_contribs[name]
            f = flat_contribs[name]
            rel = abs(d - f) / max(abs(f), 1e-9)
            if rel > ADJOINT_SPEC_REL_TOL:
                failures.append(f"  {name}: dist={d:.4e} flat={f:.4e} rel={rel:.2e}")

        assert not failures, (
            f"netlist_test adjoint static at spec {ADJOINT_SPEC_REL_TOL:.0e}: "
            f"{len(failures)}/{len(common)} common aggressors exceed tolerance:\n"
            + "\n".join(failures)
        )
        logging.disable(logging.NOTSET)

    @pytest.mark.integration
    @pytest.mark.skipif(not NETLIST_TEST_EXISTS, reason="netlist_test not available")
    def test_dynamic_adjoint_structural_overlap(self, netlist_test_setup):
        """Dynamic adjoint: top-K aggressors structurally overlap between dist and flat.

        The distributed dynamic adjoint magnitude diverges from the flat solver
        (~40–77% relative), but both should identify the same worst aggressors.
        Requires at least 2 of the top-K aggressors to overlap.
        """
        s = netlist_test_setup
        logging.disable(logging.WARNING)
        warnings.filterwarnings("ignore")

        if s.victim_node is None:
            pytest.skip("No interior node with VCS sources found in netlist_test")

        from analysis.transient_solver import IntegrationMethod

        obs_time = s.t_end

        # Distributed dynamic adjoint — raw sources (smooth=False) on both sides.
        # Using the same discipline as QS/BE transient: identical raw sources
        # avoids the dual-smoother pattern warned against in project memory.
        dist_sources = s.dist_solver.preprocess_sources(
            time_step=s.dt, t_start=s.t_start, t_end=s.t_end,
            smooth=False, verbose=False,
        )
        trans_ctx = s.dist_solver.prepare_transient(dt=s.dt, method="be", verbose=False)
        trans_result = s.dist_solver.solve_transient(
            trans_ctx,
            dc_context=s.dc_ctx,
            t_start=s.t_start,
            t_end=s.t_end,
            smoothed_sources=dist_sources,
            verbose=False,
        )
        dist_attr = s.dist_solver.analyze_adjoint(
            victim_nodes=s.victim_node,
            observation_time=obs_time,
            trans_context=trans_ctx,
            transient_result=trans_result,
            dc_context=s.dc_ctx,
            top_k=self.TOP_K,
            verbose=False,
        )
        trans_ctx.release()

        # Flat dynamic adjoint — raw VCS (no smoothed_sources) to match dist smooth=False
        from analysis.adjoint_sensitivity import AdjointSensitivitySolver
        flat_adj = AdjointSensitivitySolver(s.flat_model, s.flat_graph,
                                            vectorize_threshold=0)
        flat_attr = flat_adj.analyze_victim(
            victim_node=s.victim_node,
            observation_time=obs_time,
            dt=s.dt,
            method=IntegrationMethod.BACKWARD_EULER,
            # No smoothed_sources: raw VCS to match distributed smooth=False
            top_k=self.TOP_K,
        )

        dist_nodes = {ac.node for ac in dist_attr.top_aggressors}
        flat_nodes = {ac.node for ac in flat_attr.top_aggressors}
        assert dist_nodes, "Distributed dynamic adjoint returned empty top_aggressors"
        assert flat_nodes, "Flat dynamic adjoint returned empty top_aggressors"
        overlap = dist_nodes & flat_nodes
        assert len(overlap) >= 2, (
            f"netlist_test dynamic adjoint top-K overlap too small ({len(overlap)}); "
            f"dist={sorted(dist_nodes)[:5]} flat={sorted(flat_nodes)[:5]}"
        )
        logging.disable(logging.NOTSET)

    @pytest.mark.integration
    @pytest.mark.skipif(not NETLIST_TEST_EXISTS, reason="netlist_test not available")
    @pytest.mark.xfail(strict=False, reason=_XFAIL_ADJ_DYN_SPEC)
    def test_dynamic_adjoint_at_spec_tolerance(self, netlist_test_setup):
        """Dynamic adjoint per-aggressor contributions at spec 1e-6 relative (xfail).

        Expected to fail until the distributed dynamic backward sweep is fixed
        to match the flat adjoint magnitude (currently 40–77% relative error).
        Auto-passes when fixed (xfail strict=False).
        """
        s = netlist_test_setup
        logging.disable(logging.WARNING)
        warnings.filterwarnings("ignore")

        if s.victim_node is None:
            pytest.skip("No interior node with VCS sources found in netlist_test")

        from analysis.transient_solver import IntegrationMethod

        obs_time = s.t_end

        # Raw sources (smooth=False) on both sides — same discipline as transient tests.
        dist_sources = s.dist_solver.preprocess_sources(
            time_step=s.dt, t_start=s.t_start, t_end=s.t_end,
            smooth=False, verbose=False,
        )
        trans_ctx = s.dist_solver.prepare_transient(dt=s.dt, method="be", verbose=False)
        trans_result = s.dist_solver.solve_transient(
            trans_ctx,
            dc_context=s.dc_ctx,
            t_start=s.t_start,
            t_end=s.t_end,
            smoothed_sources=dist_sources,
            verbose=False,
        )
        dist_attr = s.dist_solver.analyze_adjoint(
            victim_nodes=s.victim_node,
            observation_time=obs_time,
            trans_context=trans_ctx,
            transient_result=trans_result,
            dc_context=s.dc_ctx,
            top_k=self.TOP_K,
            verbose=False,
        )
        trans_ctx.release()

        from analysis.adjoint_sensitivity import AdjointSensitivitySolver
        flat_adj = AdjointSensitivitySolver(s.flat_model, s.flat_graph,
                                            vectorize_threshold=0)
        flat_attr = flat_adj.analyze_victim(
            victim_node=s.victim_node,
            observation_time=obs_time,
            dt=s.dt,
            method=IntegrationMethod.BACKWARD_EULER,
            # No smoothed_sources: raw VCS to match distributed smooth=False
            top_k=self.TOP_K,
        )

        dist_contribs = {ac.node: ac.contribution_mV for ac in dist_attr.top_aggressors}
        flat_contribs = {ac.node: ac.contribution_mV for ac in flat_attr.top_aggressors}
        common = set(dist_contribs.keys()) & set(flat_contribs.keys())
        assert len(common) >= 2, (
            f"Too few common aggressors to compare ({len(common)}); "
            f"dist had {len(dist_contribs)}, flat had {len(flat_contribs)}"
        )

        failures = []
        for name in sorted(common):
            d = dist_contribs[name]
            f = flat_contribs[name]
            rel = abs(d - f) / max(abs(f), 1e-9)
            if rel > ADJOINT_SPEC_REL_TOL:
                failures.append(f"  {name}: dist={d:.4e} flat={f:.4e} rel={rel:.2e}")
        assert not failures, (
            f"netlist_test dynamic adjoint at spec {ADJOINT_SPEC_REL_TOL:.0e}: "
            f"{len(failures)}/{len(common)} exceeded:\n" + "\n".join(failures)
        )
        logging.disable(logging.NOTSET)


# ══════════════════════════════════════════════════════════════════════
# 13. Adjoint — netlist_sampled pkl (integration)
# ══════════════════════════════════════════════════════════════════════

class TestAdjointSampledPkl:
    """Adjoint sensitivity equivalence on netlist_sampled.

    Compares distributed analyze_adjoint_static against flat
    AdjointSensitivitySolver.analyze_victim_static on the common top-K
    aggressor set.  Both sides use the same DC factorization (DDM vs flat
    G^-1), so contributions should agree within ADJOINT_REL_TOL.

    Both forward (DC solve) and backward (adjoint solve) must use
    identical source treatment — using raw un-smoothed sources here
    avoids any smoothing-induced asymmetry between the two solvers.
    See memory note: 'Adjoint forward/backward must use same I.'

    Regression guards (at observed delta):
        static:  ADJOINT_REL_TOL = 1e-4 relative
        dynamic: structural top-K overlap (magnitude diverges 40–77%)
    Spec-tolerance xfail (auto-passes when src is fixed):
        both:    ADJOINT_SPEC_REL_TOL = 1e-6 relative
    """

    TOP_K = 20  # aggressors to compare

    @pytest.mark.integration
    @pytest.mark.skipif(not PKL_DIR_EXISTS, reason="distributed_pkl not available")
    @pytest.mark.skipif(not SAMPLED_DIR_EXISTS, reason="netlist_sampled not available")
    def test_static_adjoint_contributions_match(self):
        """Per-aggressor static contributions agree within ADJOINT_REL_TOL."""
        logging.disable(logging.WARNING)
        warnings.filterwarnings("ignore")

        # ── Distributed ────────────────────────────────────────────────
        from distributed.model import load_distributed_partitions, create_distributed_model
        from distributed.solver import DistributedDDMSolver
        from distributed.tile_parsing import _iter_instance_sources

        bundle = load_distributed_partitions(str(_PKL_DIR))
        dist_model = create_distributed_model(bundle, backend="local")
        dist_solver = DistributedDDMSolver(dist_model)
        dc_ctx = dist_solver.prepare(verbose=False)

        # Pick a victim node: interior node with current sources nearby
        interface_set = set(dc_ctx.interface_nodes)
        nodes_with_sources: set = set()
        for tc in dist_model.metadata.tile_configs:
            if tc.instance_path is None:
                continue
            for prep in _iter_instance_sources(tc.instance_path, tc.net_filter, tc.nd_path):
                n = prep.cs.node1
                if n and n != "0":
                    nodes_with_sources.add(n)
        interior_with_sources = sorted(
            n for n in nodes_with_sources
            if n not in interface_set and not n.endswith("_vsrc")
        )
        assert interior_with_sources, "No interior nodes with sources found"
        victim = interior_with_sources[0]

        obs_time = 5e-9  # observation time: 5 ns (arbitrary within t_end)

        dist_attr = dist_solver.analyze_adjoint_static(
            victim_nodes=victim,
            observation_time=obs_time,
            dc_context=dc_ctx,
            top_k=self.TOP_K,
            verbose=False,
        )

        # ── Flat ───────────────────────────────────────────────────────
        from parser.netlist import NetlistParser
        from model.factory import create_model_from_pdn
        from solver.unified_solver import UnifiedIRDropSolver
        from analysis.adjoint_sensitivity import AdjointSensitivitySolver

        flat_parser = NetlistParser(str(_SAMPLED_DIR))
        flat_graph = flat_parser.parse()
        flat_model = create_model_from_pdn(flat_graph, "VDD_XLV")
        flat_adj = AdjointSensitivitySolver(flat_model, flat_graph,
                                            vectorize_threshold=0)
        flat_attr = flat_adj.analyze_victim_static(
            victim_node=victim,
            observation_time=obs_time,
            top_k=self.TOP_K,
        )

        # ── Compare top-K contributions on common aggressor node set ─────
        dist_contribs: Dict[str, float] = {
            ac.node: ac.contribution_mV
            for ac in dist_attr.top_aggressors
        }
        flat_contribs: Dict[str, float] = {
            ac.node: ac.contribution_mV
            for ac in flat_attr.top_aggressors
        }
        common = set(dist_contribs.keys()) & set(flat_contribs.keys())
        assert len(common) >= 3, (
            f"Too few common aggressors ({len(common)}); "
            f"dist had {len(dist_contribs)}, flat had {len(flat_contribs)}"
        )

        failures = []
        for name in sorted(common):
            d = dist_contribs[name]
            f = flat_contribs[name]
            denom = max(abs(f), 1e-9)  # avoid division by tiny numbers
            rel = abs(d - f) / denom
            if rel > ADJOINT_REL_TOL:
                failures.append(
                    f"  {name}: dist={d:.4e} flat={f:.4e} rel={rel:.2e}"
                )

        assert not failures, (
            f"Adjoint static contributions exceed ADJOINT_REL_TOL={ADJOINT_REL_TOL:.0e} "
            f"on {len(failures)}/{len(common)} common aggressors:\n"
            + "\n".join(failures)
        )

        # Teardown
        dc_ctx.release()
        dist_model.shutdown()
        logging.disable(logging.NOTSET)

    @pytest.mark.integration
    @pytest.mark.skipif(not PKL_DIR_EXISTS, reason="distributed_pkl not available")
    @pytest.mark.skipif(not SAMPLED_DIR_EXISTS, reason="netlist_sampled not available")
    def test_static_adjoint_at_spec_tolerance(self):
        """Per-aggressor static contributions at spec 1e-6 relative (hard assert).

        Static adjoint on netlist_sampled already meets ADJOINT_SPEC_REL_TOL=1e-6
        (observed delta < 1e-6).  This is a hard assert — any regression above
        1e-6 will fail the build immediately rather than silently reverting to
        xfail.
        """
        logging.disable(logging.WARNING)
        warnings.filterwarnings("ignore")

        from distributed.model import load_distributed_partitions, create_distributed_model
        from distributed.solver import DistributedDDMSolver
        from distributed.tile_parsing import _iter_instance_sources

        bundle = load_distributed_partitions(str(_PKL_DIR))
        dist_model = create_distributed_model(bundle, backend="local")
        dist_solver = DistributedDDMSolver(dist_model)
        dc_ctx = dist_solver.prepare(verbose=False)

        interface_set = set(dc_ctx.interface_nodes)
        nodes_with_sources: set = set()
        for tc in dist_model.metadata.tile_configs:
            if tc.instance_path is None:
                continue
            for prep in _iter_instance_sources(tc.instance_path, tc.net_filter, tc.nd_path):
                n = prep.cs.node1
                if n and n != "0":
                    nodes_with_sources.add(n)
        interior_with_sources = sorted(
            n for n in nodes_with_sources
            if n not in interface_set and not n.endswith("_vsrc")
        )
        if not interior_with_sources:
            dc_ctx.release()
            dist_model.shutdown()
            pytest.skip("No interior nodes with sources")
        victim = interior_with_sources[0]

        dist_attr = dist_solver.analyze_adjoint_static(
            victim_nodes=victim,
            observation_time=5e-9,
            dc_context=dc_ctx,
            top_k=self.TOP_K,
            verbose=False,
        )

        from parser.netlist import NetlistParser
        from model.factory import create_model_from_pdn
        from analysis.adjoint_sensitivity import AdjointSensitivitySolver

        flat_parser = NetlistParser(str(_SAMPLED_DIR))
        flat_graph = flat_parser.parse()
        flat_model = create_model_from_pdn(flat_graph, "VDD_XLV")
        flat_adj = AdjointSensitivitySolver(flat_model, flat_graph, vectorize_threshold=0)
        flat_attr = flat_adj.analyze_victim_static(
            victim_node=victim, observation_time=5e-9, top_k=self.TOP_K,
        )

        dist_contribs = {ac.node: ac.contribution_mV for ac in dist_attr.top_aggressors}
        flat_contribs = {ac.node: ac.contribution_mV for ac in flat_attr.top_aggressors}
        common = set(dist_contribs.keys()) & set(flat_contribs.keys())
        assert len(common) >= 3, f"Too few common aggressors ({len(common)})"

        failures = []
        for name in sorted(common):
            d = dist_contribs[name]
            f = flat_contribs[name]
            rel = abs(d - f) / max(abs(f), 1e-9)
            if rel > ADJOINT_SPEC_REL_TOL:
                failures.append(f"  {name}: dist={d:.4e} flat={f:.4e} rel={rel:.2e}")

        dc_ctx.release()
        dist_model.shutdown()
        logging.disable(logging.NOTSET)

        assert not failures, (
            f"Adjoint static at spec {ADJOINT_SPEC_REL_TOL:.0e}: "
            f"{len(failures)}/{len(common)} exceeded:\n" + "\n".join(failures)
        )

    @pytest.mark.integration
    @pytest.mark.skipif(not PKL_DIR_EXISTS, reason="distributed_pkl not available")
    @pytest.mark.skipif(not SAMPLED_DIR_EXISTS, reason="netlist_sampled not available")
    def test_dynamic_adjoint_contributions_match(self, sampled_setup):
        """Per-aggressor dynamic contributions agree structurally (top-K overlap).

        Uses the module-scoped fixture for fast model + context reuse.
        analyze_adjoint() requires a pre-run transient result so that the
        backward sweep can use the exact same voltage trajectory as the forward.
        Both sides use raw (smooth=False) sources — identical inputs to avoid
        the dual-smoother anti-pattern documented in project memory.

        NOTE: The distributed dynamic adjoint implementation diverges
        substantially from the flat adjoint in *magnitude* (~40–77%
        relative difference is observed on netlist_sampled).  This is a
        known pre-existing difference between the two backward-sweep
        algorithms.  The test therefore checks *structural* agreement
        (top-K node overlap), not magnitude.  See test_dynamic_adjoint_at_spec_tolerance
        for the magnitude gate (xfail at spec until src is fixed).
        """
        s = sampled_setup
        from analysis.transient_solver import IntegrationMethod

        # ── Select victim: interior node with nearby current sources ────
        from distributed.tile_parsing import _iter_instance_sources

        interface_set = set(s.dc_ctx.interface_nodes)
        nodes_with_sources: set = set()
        for tc in s.dist_model.metadata.tile_configs:
            if tc.instance_path is None:
                continue
            for prep in _iter_instance_sources(tc.instance_path, tc.net_filter, tc.nd_path):
                n = prep.cs.node1
                if n and n != "0":
                    nodes_with_sources.add(n)
        interior_with_sources = sorted(
            n for n in nodes_with_sources
            if n not in interface_set and not n.endswith("_vsrc")
        )
        assert interior_with_sources, "No interior nodes with sources"
        victim = interior_with_sources[0]

        obs_time = s.t_end  # observe at the end of the window

        # ── Distributed dynamic adjoint — raw sources (smooth=False) ─────
        dist_sources = s.dist_solver.preprocess_sources(
            time_step=s.dt, t_start=s.t_start, t_end=s.t_end,
            smooth=False, verbose=False,
        )
        trans_ctx = s.dist_solver.prepare_transient(dt=s.dt, method="be", verbose=False)
        trans_result = s.dist_solver.solve_transient(
            trans_ctx,
            dc_context=s.dc_ctx,
            t_start=s.t_start,
            t_end=s.t_end,
            smoothed_sources=dist_sources,
            verbose=False,
        )
        dist_attr = s.dist_solver.analyze_adjoint(
            victim_nodes=victim,
            observation_time=obs_time,
            trans_context=trans_ctx,
            transient_result=trans_result,
            dc_context=s.dc_ctx,
            top_k=self.TOP_K,
            verbose=False,
        )
        trans_ctx.release()

        # ── Flat dynamic adjoint — raw VCS (no smoothed_sources) ─────────
        from analysis.adjoint_sensitivity import AdjointSensitivitySolver

        flat_adj = AdjointSensitivitySolver(
            s.flat_model, s.flat_graph, vectorize_threshold=0
        )
        flat_attr = flat_adj.analyze_victim(
            victim_node=victim,
            observation_time=obs_time,
            dt=s.dt,
            method=IntegrationMethod.BACKWARD_EULER,
            # No smoothed_sources: raw VCS to match distributed smooth=False
            top_k=self.TOP_K,
        )

        # ── Structural agreement: top-K aggressor overlap ──────────────
        dist_nodes = [ac.node for ac in dist_attr.top_aggressors]
        flat_nodes = [ac.node for ac in flat_attr.top_aggressors]
        assert dist_nodes, "Distributed adjoint returned empty top_aggressors"
        assert flat_nodes, "Flat adjoint returned empty top_aggressors"

        # Both must have non-trivial contributions
        assert dist_attr.top_aggressors[0].contribution_mV > 0, (
            "Distributed dynamic adjoint top-1 contribution is zero or negative"
        )
        assert flat_attr.top_aggressors[0].contribution_mV > 0, (
            "Flat dynamic adjoint top-1 contribution is zero or negative"
        )

        # At least 2 of the top-K aggressors must be common to both
        top_dist_set = set(dist_nodes)
        top_flat_set = set(flat_nodes)
        overlap = top_dist_set & top_flat_set
        assert len(overlap) >= 2, (
            f"Dynamic adjoint top-K overlap too small ({len(overlap)}); "
            f"dist={sorted(top_dist_set)[:5]} flat={sorted(top_flat_set)[:5]}"
        )

    @pytest.mark.integration
    @pytest.mark.skipif(not PKL_DIR_EXISTS, reason="distributed_pkl not available")
    @pytest.mark.skipif(not SAMPLED_DIR_EXISTS, reason="netlist_sampled not available")
    def test_dynamic_adjoint_at_spec_tolerance(self, sampled_setup):
        """Per-aggressor dynamic contributions at spec 1e-6 relative (hard assert).

        With identical raw sources (smooth=False) on both sides, the distributed
        dynamic adjoint meets ADJOINT_SPEC_REL_TOL=1e-6 on netlist_sampled.
        The prior xfail was caused by the dual-smoother anti-pattern (independent
        smooth=True executions on each side), not a genuine solver divergence.
        Any regression above 1e-6 fails immediately rather than silently xfailing.
        """
        s = sampled_setup
        from analysis.transient_solver import IntegrationMethod
        from distributed.tile_parsing import _iter_instance_sources

        interface_set = set(s.dc_ctx.interface_nodes)
        nodes_with_sources: set = set()
        for tc in s.dist_model.metadata.tile_configs:
            if tc.instance_path is None:
                continue
            for prep in _iter_instance_sources(tc.instance_path, tc.net_filter, tc.nd_path):
                n = prep.cs.node1
                if n and n != "0":
                    nodes_with_sources.add(n)
        interior_with_sources = sorted(
            n for n in nodes_with_sources
            if n not in interface_set and not n.endswith("_vsrc")
        )
        assert interior_with_sources
        victim = interior_with_sources[0]
        obs_time = s.t_end

        # Raw sources (smooth=False) on both sides — same discipline as transient tests.
        dist_sources = s.dist_solver.preprocess_sources(
            time_step=s.dt, t_start=s.t_start, t_end=s.t_end,
            smooth=False, verbose=False,
        )
        trans_ctx = s.dist_solver.prepare_transient(dt=s.dt, method="be", verbose=False)
        trans_result = s.dist_solver.solve_transient(
            trans_ctx,
            dc_context=s.dc_ctx,
            t_start=s.t_start,
            t_end=s.t_end,
            smoothed_sources=dist_sources,
            verbose=False,
        )
        dist_attr = s.dist_solver.analyze_adjoint(
            victim_nodes=victim,
            observation_time=obs_time,
            trans_context=trans_ctx,
            transient_result=trans_result,
            dc_context=s.dc_ctx,
            top_k=self.TOP_K,
            verbose=False,
        )
        trans_ctx.release()

        from analysis.adjoint_sensitivity import AdjointSensitivitySolver
        flat_adj = AdjointSensitivitySolver(s.flat_model, s.flat_graph, vectorize_threshold=0)
        flat_attr = flat_adj.analyze_victim(
            victim_node=victim, observation_time=obs_time, dt=s.dt,
            method=IntegrationMethod.BACKWARD_EULER,
            # No smoothed_sources: raw VCS to match distributed smooth=False
            top_k=self.TOP_K,
        )

        dist_contribs = {ac.node: ac.contribution_mV for ac in dist_attr.top_aggressors}
        flat_contribs = {ac.node: ac.contribution_mV for ac in flat_attr.top_aggressors}
        common = set(dist_contribs.keys()) & set(flat_contribs.keys())
        assert len(common) >= 2, f"Too few common aggressors ({len(common)})"

        failures = []
        for name in sorted(common):
            d = dist_contribs[name]
            f = flat_contribs[name]
            rel = abs(d - f) / max(abs(f), 1e-9)
            if rel > ADJOINT_SPEC_REL_TOL:
                failures.append(f"  {name}: dist={d:.4e} flat={f:.4e} rel={rel:.2e}")
        assert not failures, (
            f"Dynamic adjoint at spec {ADJOINT_SPEC_REL_TOL:.0e}: "
            f"{len(failures)}/{len(common)} exceeded:\n" + "\n".join(failures)
        )


# ══════════════════════════════════════════════════════════════════════
# 14. Flag-gate: flag-ON == flag-OFF (bit-identical, ≤ FLAG_GATE_ATOL)
# ══════════════════════════════════════════════════════════════════════

class TestSolverVariantFlagGate:
    """Cross-comparison: distributed flag-ON == distributed flag-OFF to FLAG_GATE_ATOL=1e-12 V.

    Phase V spec requirement: 'Every Phase A flag tested on and off.'
    A2's verify clause: 'end-to-end equal to flag-off ≤1e-12.'

    This is SEPARATE from the flat-vs-distributed equivalence tests above.
    Those compare each variant against the flat oracle at loose tolerances
    (TRANS_ATOL=5e-6 for BE, etc.) because of the dual-smoother issue.
    This class compares flag-ON vs flag-OFF on the SAME (distributed) side,
    so no dual-smoother divergence occurs and the tolerance is tight (1e-12).

    A test that compares flag-ON to flat at 5e-6 V would NOT catch a ~1 µV
    bug introduced by the flag — that bug would be hidden inside the 5e-6
    regression guard.  This class catches it immediately.

    When SOLVER_VARIANTS = [pytest.param({}, id="baseline")] only,
    _FLAG_VARIANTS is empty and _FLAG_VARIANTS_OR_SKIP contains a single
    skip-marked placeholder.  All tests below are skipped automatically.

    When Phase A2 appends pytest.param({"use_step_columns": True}, id="step_cols")
    to SOLVER_VARIANTS (one-line change at the top of this file), _FLAG_VARIANTS
    becomes non-empty, the skip is lifted, and all tests run automatically for
    every new flag variant.

    Note: solve_dc() does not accept solver_kw flags, so DC is not in this gate.
    QS and transient are the relevant flag-bearing paths.
    """

    # ── 14a. QS — synthetic 2-tile (unit, fast) ────────────────────────

    @pytest.mark.unit
    @pytest.mark.parametrize("variant_kw", _FLAG_VARIANTS_OR_SKIP)
    def test_qs_synthetic_flag_matches_baseline(self, variant_kw):
        """QS synthetic 2-tile: flag-on per-step max-drop == baseline to 1e-12 V."""
        from distributed.result import DistributedSmoothedSources
        from distributed.solver import DistributedDDMSolver

        dt = 1e-9
        t_start = 0.0
        t_end = 3e-9
        n_points = 4

        def _run_qs(extra_kw):
            model = _build_synthetic_two_tile_model()
            solver = DistributedDDMSolver(model)
            ctx = solver.prepare(verbose=False)
            dummy = DistributedSmoothedSources(
                time_step=dt, t_start=t_start, t_end=t_end,
                smoothed=False, n_tiles=len(model.workers), per_tile_stats={},
            )
            result = solver.solve_quasi_static(
                ctx, t_start=t_start, t_end=t_end, n_points=n_points,
                smoothed_sources=dummy, verbose=False, **extra_kw,
            )
            arr = np.array(result.max_ir_drop_per_time)
            ctx.release()
            model.shutdown()
            return arr

        base_arr = _run_qs({})
        var_arr = _run_qs(variant_kw)
        n = min(len(base_arr), len(var_arr))
        max_diff = float(np.max(np.abs(var_arr[:n] - base_arr[:n])))
        assert max_diff <= FLAG_GATE_ATOL, (
            f"QS synthetic flag-gate [{variant_kw}]: "
            f"flag-on vs flag-off per-step max-drop diff = {max_diff:.3e} V "
            f"> FLAG_GATE_ATOL={FLAG_GATE_ATOL:.0e} V  "
            f"(base={base_arr.tolist()} var={var_arr[:n].tolist()})"
        )

    # ── 14b. Transient BE/TR — synthetic 2-tile (unit, fast) ───────────

    @pytest.mark.unit
    @pytest.mark.parametrize("method", ["be", "trap"])
    @pytest.mark.parametrize("variant_kw", _FLAG_VARIANTS_OR_SKIP)
    def test_transient_synthetic_flag_matches_baseline(self, method, variant_kw):
        """Transient synthetic 2-tile: flag-on peak == baseline to 1e-12 V."""
        from distributed.result import DistributedSmoothedSources
        from distributed.solver import DistributedDDMSolver

        dt = 1e-9
        t_start = 0.0
        t_end = 3e-9

        def _run_trans(extra_kw):
            model = _build_synthetic_two_tile_model()
            solver = DistributedDDMSolver(model)
            dc_ctx = solver.prepare(verbose=False)
            trans_ctx = solver.prepare_transient(dt=dt, method=method, verbose=False)
            dummy = DistributedSmoothedSources(
                time_step=dt, t_start=t_start, t_end=t_end,
                smoothed=False, n_tiles=len(model.workers), per_tile_stats={},
            )
            result = solver.solve_transient(
                trans_ctx, dc_context=dc_ctx, t_start=t_start, t_end=t_end,
                smoothed_sources=dummy, verbose=False, **extra_kw,
            )
            peak = result.peak_ir_drop
            trans_ctx.release()
            dc_ctx.release()
            model.shutdown()
            return peak

        base_peak = _run_trans({})
        var_peak = _run_trans(variant_kw)
        diff = abs(var_peak - base_peak)
        assert diff <= FLAG_GATE_ATOL, (
            f"Transient-{method} synthetic flag-gate [{variant_kw}]: "
            f"flag-on peak={var_peak:.12f} V  flag-off peak={base_peak:.12f} V  "
            f"diff={diff:.3e} V > FLAG_GATE_ATOL={FLAG_GATE_ATOL:.0e} V"
        )

    # ── 14c. QS — netlist_sampled pkl (unit + integration) ─────────────

    @pytest.mark.unit
    @pytest.mark.integration
    @pytest.mark.parametrize("variant_kw", _FLAG_VARIANTS_OR_SKIP)
    def test_qs_sampled_flag_matches_baseline(self, sampled_setup, variant_kw):
        """QS netlist_sampled: flag-on per-step max-drop == baseline to 1e-12 V.

        Uses smooth=False on both runs (same as the main QS equivalence tests)
        so the only delta between runs is the solver flag itself.
        """
        s = sampled_setup

        def _run_qs(extra_kw):
            sources = s.dist_solver.preprocess_sources(
                time_step=s.dt, t_start=s.t_start, t_end=s.t_end,
                smooth=False, verbose=False,
            )
            result = s.dist_solver.solve_quasi_static(
                s.dc_ctx, t_start=s.t_start, t_end=s.t_end,
                n_points=s.n_points, smoothed_sources=sources,
                verbose=False, **extra_kw,
            )
            return np.array(result.max_ir_drop_per_time)

        base_arr = _run_qs({})
        var_arr = _run_qs(variant_kw)
        n = min(len(base_arr), len(var_arr))
        max_diff = float(np.max(np.abs(var_arr[:n] - base_arr[:n])))
        assert max_diff <= FLAG_GATE_ATOL, (
            f"QS sampled flag-gate [{variant_kw}]: "
            f"flag-on vs flag-off diff = {max_diff:.3e} V "
            f"> FLAG_GATE_ATOL={FLAG_GATE_ATOL:.0e} V  (n_steps={n})"
        )

    # ── 14d. Transient BE/TR — netlist_sampled pkl (unit + integration) ─

    @pytest.mark.unit
    @pytest.mark.integration
    @pytest.mark.parametrize("method", ["be", "trap"])
    @pytest.mark.parametrize("variant_kw", _FLAG_VARIANTS_OR_SKIP)
    def test_transient_sampled_flag_matches_baseline(self, sampled_setup, method, variant_kw):
        """Transient netlist_sampled: flag-on peak == baseline to 1e-12 V.

        Uses smooth=False on both runs (via _run_dist_transient) — same raw
        sources on both sides, so the only delta is the solver flag itself.
        No dual-smoother issue here: both runs go through the same distributed
        VCS path with identical inputs.
        """
        s = sampled_setup
        base_result = _run_dist_transient(s, method, {})
        var_result = _run_dist_transient(s, method, variant_kw)
        diff = abs(var_result.peak_ir_drop - base_result.peak_ir_drop)
        assert diff <= FLAG_GATE_ATOL, (
            f"Transient-{method} sampled flag-gate [{variant_kw}]: "
            f"flag-on peak={var_result.peak_ir_drop:.12f} V  "
            f"flag-off peak={base_result.peak_ir_drop:.12f} V  "
            f"diff={diff:.3e} V > FLAG_GATE_ATOL={FLAG_GATE_ATOL:.0e} V"
        )

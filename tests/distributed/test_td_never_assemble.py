"""TD never-assemble work package: transient tilewise CG without S_global.

Covers ``_factor_transient_context_no_s_global`` (the transient extension of
Stage 2's ``_factor_dc_context_no_s_global``) end-to-end:

  - Exactness matrix: never-assemble vs assembled, BE/TR, multiple dt,
    package caps present (so S_extra^TD's dt-dependence is exercised).
  - The rhs_dirichlet_G linearity-delta correctness (the highest-risk piece
    -- a wrong Dirichlet vector shows up as a small constant voltage bias,
    not a crash), checked directly on the vectors, not just end-to-end.
  - Island parity on a cap-coupled-only component (DC island, TD not).
  - The T1 island-penalty RHS regression, extended to the never-assemble path.
  - Lifecycle: release->refactor rebuilds tilewise; stale-ordering drift
    raises; save() raises with guidance.
  - Both never-assemble contexts (DC + TD) alive simultaneously.
  - Adjoint on a never-assemble TD context.
  - Warm-start iteration-count determinism across repeated solve_transient
    calls on one context.

See also ``test_interface_iterative_stage2.py``'s ``TestInterfaceDropSGlobal``
(DC-only coverage, item 3 baseline) and its
``TestS15ForcedCGEquivalenceMatrix.test_transient_tilewise_never_assemble_matches_direct``
(the forced-CG equivalence matrix's TD row).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.unit

sys.path.insert(0, str(Path(__file__).parent))
from test_time_domain import _build_two_tile_distributed_model  # noqa: E402
from test_interface_iterative_stage2 import (  # noqa: E402
    _build_three_tile_island_model,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _summaries_two_tile_model(pkg_caps, tmp_path, subdir='m'):
    """Two-tile fixture (optionally with package caps) + summaries island
    detection (interface_drop_s_global's precondition) + a real ckt_path
    parent dir (required by preprocess_sources' VCS cache location)."""
    model = _build_two_tile_distributed_model(package_cap_edges=pkg_caps)
    for tc in model.metadata.tile_configs:
        tc.ckt_path = str(tmp_path / subdir / 'dummy.ckt')
    model.island_detection_mode = 'summaries'
    model.component_summaries = []
    return model


def _never_assemble_settings(rtol=1e-12, atol=1e-16, preconditioner='block_jacobi'):
    return {
        'interface_solver': 'cg',
        'interface_matvec_mode': 'tilewise',
        'interface_preconditioner': preconditioner,
        'interface_cg_rtol': rtol,
        'interface_cg_atol': atol,
        'interface_drop_s_global': True,
    }


def _assembled_cg_settings(rtol=1e-12, atol=1e-16, preconditioner='block_jacobi'):
    return {
        'interface_solver': 'cg',
        'interface_matvec_mode': 'tilewise',
        'interface_preconditioner': preconditioner,
        'interface_cg_rtol': rtol,
        'interface_cg_atol': atol,
    }


def _solve_transient_two_tile(
    model, method, dt, t_end, *, pkl_dir, track_nodes=None,
):
    """Prepare + solve_transient on the two-tile fixture; returns
    (result, trans_ctx, dc_ctx, solver), with the caller owning release()
    of both trans_ctx and dc_ctx (and model.shutdown())."""
    from distributed.solver import DistributedDDMSolver

    solver = DistributedDDMSolver(model)
    dc_ctx = solver.prepare()
    trans_ctx = solver.prepare_transient(dt=dt, method=method)
    smoothed = solver.preprocess_sources(
        time_step=dt, t_start=0.0, t_end=t_end, smooth=False,
        pkl_dir=str(pkl_dir),
    )
    result = solver.solve_transient(
        trans_ctx, dc_context=dc_ctx, t_end=t_end,
        smoothed_sources=smoothed, track_nodes=track_nodes,
    )
    return result, trans_ctx, dc_ctx, solver


# ---------------------------------------------------------------------------
# 1. Exactness matrix: never-assemble vs assembled, BE/TR, two dt values,
#    package caps present.
# ---------------------------------------------------------------------------


class TestExactnessMatrix:

    PKG_CAPS = [('pad', 'shared', 100.0)]

    def _run(self, method, dt, drop_s_global, tmp_path, subdir):
        model = _summaries_two_tile_model(self.PKG_CAPS, tmp_path, subdir)
        settings = (
            _never_assemble_settings() if drop_s_global
            else _assembled_cg_settings()
        )
        model.settings.update(settings)
        result, trans_ctx, dc_ctx, solver = _solve_transient_two_tile(
            model, method, dt, t_end=5 * dt,
            pkl_dir=tmp_path / subdir,
            track_nodes=['a1', 'b1', 'shared'],
        )
        try:
            assert (trans_ctx._S_global is None) == drop_s_global
            flat = result.as_flat()
            waveforms = {k: np.array(v) for k, v in result.tracked_waveforms.items()}
            peak = (result.peak_ir_drop, result.peak_ir_drop_time)
        finally:
            trans_ctx.release()
            dc_ctx.release()
            model.shutdown()
        return flat, waveforms, peak

    @pytest.mark.parametrize('method', ['be', 'trap'])
    @pytest.mark.parametrize('dt', [1e-10, 3e-10])
    def test_never_assemble_matches_assembled(self, method, dt, tmp_path):
        flat_a, wave_a, peak_a = self._run(method, dt, False, tmp_path, 'assembled')
        flat_n, wave_n, peak_n = self._run(method, dt, True, tmp_path, 'never_assemble')

        # Peak IR-drop.
        assert abs(peak_a[0] - peak_n[0]) < 1e-10, (
            f"[{method}, dt={dt:.1e}] peak IR-drop assembled={peak_a[0]!r} "
            f"vs never-assemble={peak_n[0]!r}"
        )

        # Waveforms.
        assert set(wave_a) == set(wave_n)
        for node in wave_a:
            np.testing.assert_allclose(
                wave_a[node], wave_n[node], atol=1e-10,
                err_msg=f"[{method}, dt={dt:.1e}] waveform mismatch at {node!r}",
            )

        # Per-node peak voltages (as_flat: node -> (max_drop, time_of_max)).
        common = set(flat_a) & set(flat_n)
        assert common, "fixture produced no comparable nodes"
        for node in common:
            drop_a, _t_a = flat_a[node]
            drop_n, _t_n = flat_n[node]
            assert abs(drop_a - drop_n) < 1e-10, (
                f"[{method}, dt={dt:.1e}] node {node}: assembled drop="
                f"{drop_a!r} vs never-assemble drop={drop_n!r}"
            )


# ---------------------------------------------------------------------------
# 1b. netlist_multi_tile end-to-end smoke (integration -- real netlist data)
# lives in test_td_never_assemble_integration.py (repo convention: a
# _integration.py file, not a marked class inside the unit file -- a
# module-level `pytestmark = pytest.mark.unit` here would otherwise also
# tag an integration-marked class, since pytest merges module- and
# class-level marks, making `-m unit` incorrectly pick it up too).
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 2. rhs_dirichlet_G delta correctness (highest-risk piece): compare the
#    actual vectors, not just end-to-end voltages.
# ---------------------------------------------------------------------------


class TestRhsDirichletGDeltaCorrectness:

    @pytest.mark.parametrize('method', ['be', 'trap'])
    @pytest.mark.parametrize('dt', [1e-10, 4e-10])
    def test_vectors_match_assembled_path(self, method, dt, tmp_path):
        pkg_caps = [('pad', 'shared', 100.0)]

        model_a = _summaries_two_tile_model(pkg_caps, tmp_path, 'a')
        model_a.settings.update(_assembled_cg_settings())
        from distributed.solver import DistributedDDMSolver
        solver_a = DistributedDDMSolver(model_a)
        trans_a = solver_a.prepare_transient(dt=dt, method=method)
        try:
            rhs_A_a = trans_a.rhs_dirichlet_interface.copy()
            rhs_G_a = trans_a.rhs_dirichlet_G.copy()
            island_pen_a = trans_a.island_penalty_rhs
        finally:
            trans_a.release()
            model_a.shutdown()

        model_n = _summaries_two_tile_model(pkg_caps, tmp_path, 'n')
        model_n.settings.update(_never_assemble_settings())
        solver_n = DistributedDDMSolver(model_n)
        trans_n = solver_n.prepare_transient(dt=dt, method=method)
        try:
            # Finding 13: this is exactly the assertion that fires if the
            # never-assemble dispatch silently regresses to the assembled
            # path -- keep it inside the try/finally so a regression here
            # still releases trans_n/model_n instead of leaking LocalBackend
            # workers for the rest of the pytest session.
            assert trans_n._S_global is None
            rhs_A_n = trans_n.rhs_dirichlet_interface.copy()
            rhs_G_n = trans_n.rhs_dirichlet_G.copy()
            island_pen_n = trans_n.island_penalty_rhs
        finally:
            trans_n.release()
            model_n.shutdown()

        # Sanity: this fixture genuinely exercises rhs_dirichlet_G !=
        # rhs_dirichlet_A (the linearity-delta path, not the no-op copy).
        assert not np.allclose(rhs_A_a, rhs_G_a), (
            "fixture sanity: package cap edge must make rhs_dirichlet_G "
            "diverge from rhs_dirichlet_A on the assembled path"
        )

        assert rhs_A_a.shape == rhs_A_n.shape
        np.testing.assert_allclose(
            rhs_A_a, rhs_A_n, rtol=1e-12, atol=1e-14,
            err_msg=(
                f"[{method}, dt={dt:.1e}] rhs_dirichlet_A (never-assemble) "
                f"!= rhs_dirichlet_A (assembled)"
            ),
        )
        np.testing.assert_allclose(
            rhs_G_a, rhs_G_n, rtol=1e-12, atol=1e-14,
            err_msg=(
                f"[{method}, dt={dt:.1e}] rhs_dirichlet_G (never-assemble) "
                f"!= rhs_dirichlet_G (assembled) -- a wrong Dirichlet vector "
                f"shows up as a small constant voltage bias, not a crash"
            ),
        )
        assert island_pen_a is None and island_pen_n is None, (
            "fixture sanity: no islands expected on the plain two-tile fixture"
        )

    def test_no_caps_rhs_g_equals_rhs_a_never_assemble(self, tmp_path):
        """Without package caps, rhs_dirichlet_G == rhs_dirichlet_A on the
        never-assemble path too (mirrors TestTransientDirichletRHS's
        assembled-path twin in test_time_domain.py)."""
        model = _summaries_two_tile_model([], tmp_path, 'nocaps')
        model.settings.update(_never_assemble_settings())
        from distributed.solver import DistributedDDMSolver
        solver = DistributedDDMSolver(model)
        trans_ctx = solver.prepare_transient(dt=1e-9, method='be')
        try:
            assert trans_ctx._S_global is None
            rhs_A = trans_ctx.rhs_dirichlet_interface
            rhs_G = trans_ctx.rhs_dirichlet_G
            np.testing.assert_allclose(rhs_G, rhs_A, atol=1e-12)
        finally:
            trans_ctx.release()
            model.shutdown()


# ---------------------------------------------------------------------------
# 3. Island parity: never-assemble TD island set == assembled TD island set,
#    on a cap-coupled-only component (DC island, TD not).
# ---------------------------------------------------------------------------


class TestIslandParity:

    def test_cap_bridged_component_matches_assembled(self, tmp_path):
        """Finding 11 (round 2 review): wrap the assertions in try/finally
        (mirroring this same file's Finding-13/14 pattern) so a regression
        in a fixture-sanity assert doesn't leak the CHOLMOD factor / CG
        thread pool / model workers into the rest of the pytest session."""
        from distributed.solver import DistributedDDMSolver

        model_a = _build_three_tile_island_model(cap_c_fF=20.0)
        solver_a = DistributedDDMSolver(model_a)
        trans_a = solver_a.prepare_transient(dt=1e-10, method='be')
        # Finding 14: capture + release() the inline DC context -- it was
        # previously discarded unreleased, leaking the CHOLMOD factor / CG
        # thread pool it built purely to check topology.island_nodes.
        dc_a = solver_a.prepare()
        try:
            # Fixture sanity: DC-mode islands 'iso' (no resistive path), but
            # the cap bridge rescues it in TD mode.
            assert 'iso' in dc_a.topology.island_nodes
            islands_a = set(trans_a.topology.island_nodes_td)
        finally:
            dc_a.release()
            trans_a.release()
            model_a.shutdown()

        model_n = _build_three_tile_island_model(cap_c_fF=20.0)
        model_n.island_detection_mode = 'summaries'
        model_n.component_summaries = []
        model_n.settings.update(_never_assemble_settings())
        solver_n = DistributedDDMSolver(model_n)
        trans_n = solver_n.prepare_transient(dt=1e-10, method='be')
        try:
            assert trans_n._S_global is None
            islands_n = set(trans_n.topology.island_nodes_td)
        finally:
            trans_n.release()
            model_n.shutdown()

        assert 'iso' not in islands_a, (
            "fixture sanity: the cap bridge must rescue 'iso' in TD mode "
            "on the assembled/schur_bfs path"
        )
        assert islands_a == islands_n, (
            f"TD island set mismatch: assembled={islands_a!r} vs "
            f"never-assemble={islands_n!r}"
        )


# ---------------------------------------------------------------------------
# 4. Penalty RHS regression (Stage 2 T1), extended to never-assemble TD.
# ---------------------------------------------------------------------------


class TestPenaltyRhsRegressionNeverAssemble:

    @pytest.mark.parametrize('method', ['be', 'trap'])
    def test_island_node_stays_pinned_near_vdd(self, method, tmp_path):
        from distributed.solver import DistributedDDMSolver

        model = _build_three_tile_island_model(cap_c_fF=0.0)
        model.island_detection_mode = 'summaries'
        model.component_summaries = []
        model.settings.update(_never_assemble_settings())
        vdd = model.vdd

        solver = DistributedDDMSolver(model)
        dc_ctx = solver.prepare()
        trans_ctx = solver.prepare_transient(dt=1e-10, method=method)
        assert trans_ctx._S_global is None
        # Fixture sanity: 'iso' really is a TRANSIENT-mode island (no cap
        # bridge -- cap_c_fF=0.0), so the T1 penalty RHS path is exercised.
        assert 'iso' in trans_ctx.topology.island_nodes_td
        try:
            smoothed = solver.preprocess_sources(
                time_step=1e-10, t_start=0.0, t_end=2e-9, smooth=False,
                pkl_dir=str(tmp_path),
            )
            result = solver.solve_transient(
                trans_ctx, dc_context=dc_ctx, t_end=2e-9,
                smoothed_sources=smoothed,
                track_nodes=['iso', 'shared', 'b1'],
            )
        finally:
            trans_ctx.release()
            dc_ctx.release()
            model.shutdown()

        iso_wave = result.tracked_waveforms.get('iso')
        assert iso_wave is not None and len(iso_wave) > 1
        for step_idx, v in enumerate(iso_wave):
            assert abs(vdd - v) < 1e-3, (
                f"[method={method}, never-assemble] step {step_idx}: island "
                f"node 'iso' drifted to v={v!r} (expected ~{vdd!r} V, "
                f"pinned)."
            )
        assert iso_wave[-1] > 0.9 * vdd


# ---------------------------------------------------------------------------
# 5. Refactor: release->refactor rebuilds tilewise + coarse space;
#    stale-ordering drift raises; save() raises with guidance.
# ---------------------------------------------------------------------------


class TestRefactorLifecycle:

    def test_save_raises_with_guidance(self, tmp_path):
        from distributed.solver import DistributedDDMSolver

        model = _summaries_two_tile_model([], tmp_path, 'save')
        model.settings.update(_never_assemble_settings())
        solver = DistributedDDMSolver(model)
        trans_ctx = solver.prepare_transient(dt=1e-10, method='be')
        try:
            with pytest.raises(RuntimeError, match='never assemble'):
                trans_ctx.save(str(tmp_path / 'should_fail.pkl'))
        finally:
            trans_ctx.release()
            model.shutdown()

    def test_release_then_refactor_rebuilds_tilewise(self, tmp_path):
        from distributed.solver import DistributedDDMSolver

        model = _summaries_two_tile_model([], tmp_path, 'rf')
        model.settings.update(_never_assemble_settings())
        solver = DistributedDDMSolver(model)
        trans_ctx = solver.prepare_transient(dt=1e-10, method='be')
        try:
            assert trans_ctx._cg_solver.matvec_mode == 'tilewise'
            v_before = trans_ctx._cg_solver.preconditioner_label

            trans_ctx.release()  # clears coordinator LU + worker TD factors
            assert trans_ctx._interface_lu is None
            assert trans_ctx._cg_solver is None

            # Workers are still attached (same model) -- refactor() must
            # re-gather S_A_i via streaming and rebuild tilewise, not raise
            # or silently downgrade (no 'assembled' fallback exists here).
            trans_ctx.refactor()
            assert trans_ctx.is_factored
            assert trans_ctx._S_global is None, (
                "refactor() of a never-assembled context must stay "
                "never-assembled"
            )
            assert getattr(trans_ctx, '_s_global_dropped', False) is True
            assert trans_ctx._cg_solver is not None
            assert trans_ctx._cg_solver.matvec_mode == 'tilewise'
            # Coarse space (or whatever the preconditioner resolved to) is
            # rebuilt fresh -- same label as before release/refactor.
            assert trans_ctx._cg_solver.preconditioner_label == v_before

            dc_ctx = solver.prepare()
            smoothed = solver.preprocess_sources(
                time_step=1e-10, t_start=0.0, t_end=1e-9, smooth=False,
                pkl_dir=str(tmp_path / 'rf'),
            )
            result = solver.solve_transient(
                trans_ctx, dc_context=dc_ctx, t_end=1e-9,
                smoothed_sources=smoothed,
            )
            dc_ctx.release()
            assert result.as_flat()  # sanity: solve actually runs
        finally:
            trans_ctx.release()
            model.shutdown()

    def test_release_then_refactor_preserves_package_cap_history(self, tmp_path):
        """Review round-1 major finding: ``_refactor_transient_context``'s
        never-assemble tilewise re-gather rebuilt the interface solver but
        never ``ctx.C_package_uu`` / ``ctx._G_package_uu``, which
        ``release()`` clears to ``None`` -- so a post-refactor
        ``solve_transient`` silently dropped the package-cap history term.
        Compares a release()->refactor()'d run against a fresh
        (never-released) reference run on an identical model with a
        pad-adjacent package cap."""
        from distributed.solver import DistributedDDMSolver

        pkg_caps = [('pad', 'shared', 100.0)]

        # Reference: prepared and solved without ever releasing/refactoring.
        model_ref = _summaries_two_tile_model(pkg_caps, tmp_path, 'ref')
        model_ref.settings.update(_never_assemble_settings())
        result_ref, trans_ref, dc_ref, _ = _solve_transient_two_tile(
            model_ref, 'be', 1e-10, t_end=5e-10,
            pkl_dir=tmp_path / 'ref', track_nodes=['a1', 'b1', 'shared'],
        )
        wave_ref = {k: np.array(v) for k, v in result_ref.tracked_waveforms.items()}
        trans_ref.release()
        dc_ref.release()
        model_ref.shutdown()

        # Refactored: release() then refactor() before the real solve.
        model_rf = _summaries_two_tile_model(pkg_caps, tmp_path, 'rf')
        model_rf.settings.update(_never_assemble_settings())
        solver_rf = DistributedDDMSolver(model_rf)
        trans_rf = solver_rf.prepare_transient(dt=1e-10, method='be')
        assert trans_rf.C_package_uu is not None, (
            "fixture sanity: a nonzero pad-adjacent package cap must "
            "produce a nonzero C_package_uu"
        )
        trans_rf.release()
        assert trans_rf.C_package_uu is None  # release() clears it
        trans_rf.refactor()
        try:
            assert trans_rf.C_package_uu is not None, (
                "refactor() of a never-assembled TD context must rebuild "
                "C_package_uu, not leave it None (silently drops the "
                "package-cap history term)"
            )
            dc_rf = solver_rf.prepare()
            smoothed = solver_rf.preprocess_sources(
                time_step=1e-10, t_start=0.0, t_end=5e-10, smooth=False,
                pkl_dir=str(tmp_path / 'rf'),
            )
            result_rf = solver_rf.solve_transient(
                trans_rf, dc_context=dc_rf, t_end=5e-10,
                smoothed_sources=smoothed, track_nodes=['a1', 'b1', 'shared'],
            )
            dc_rf.release()
        finally:
            trans_rf.release()
            model_rf.shutdown()

        wave_rf = {k: np.array(v) for k, v in result_rf.tracked_waveforms.items()}
        assert set(wave_ref) == set(wave_rf)
        for node in wave_ref:
            np.testing.assert_allclose(
                wave_ref[node], wave_rf[node], atol=1e-9,
                err_msg=(
                    f"node {node!r}: release->refactor waveform diverged "
                    f"from the never-released reference (package-cap "
                    f"history term dropped after refactor?)"
                ),
            )

    def test_release_then_refactor_preserves_island_penalty(self, tmp_path):
        """Same finding as above, island-penalty half: mirrors
        ``TestPenaltyRhsRegressionNeverAssemble`` but exercises the
        release()->refactor() cycle. Pre-fix, ``island_penalty_rhs`` goes
        ``[1e5*vdd, ...] -> None`` on release() and STAYED ``None`` after
        refactor(), so the penalized 'iso' node lost its Vdd forcing term
        and decayed instead of holding near Vdd."""
        from distributed.solver import DistributedDDMSolver

        model = _build_three_tile_island_model(cap_c_fF=0.0)
        model.island_detection_mode = 'summaries'
        model.component_summaries = []
        model.settings.update(_never_assemble_settings())
        vdd = model.vdd

        solver = DistributedDDMSolver(model)
        trans_ctx = solver.prepare_transient(dt=1e-10, method='be')
        assert trans_ctx.island_penalty_rhs is not None, (
            "fixture sanity: 'iso' must be a TD-mode island requiring the "
            "penalty RHS"
        )
        trans_ctx.release()
        assert trans_ctx.island_penalty_rhs is None  # release() clears it
        trans_ctx.refactor()
        try:
            assert trans_ctx.island_penalty_rhs is not None, (
                "refactor() of a never-assembled TD context must rebuild "
                "island_penalty_rhs, not leave it None (the penalized "
                "island node loses its Vdd forcing term and decays)"
            )
            dc_ctx = solver.prepare()
            smoothed = solver.preprocess_sources(
                time_step=1e-10, t_start=0.0, t_end=2e-9, smooth=False,
                pkl_dir=str(tmp_path),
            )
            result = solver.solve_transient(
                trans_ctx, dc_context=dc_ctx, t_end=2e-9,
                smoothed_sources=smoothed,
                track_nodes=['iso', 'shared', 'b1'],
            )
            dc_ctx.release()
        finally:
            trans_ctx.release()
            model.shutdown()

        iso_wave = result.tracked_waveforms.get('iso')
        assert iso_wave is not None and len(iso_wave) > 1
        for step_idx, v in enumerate(iso_wave):
            assert abs(vdd - v) < 1e-3, (
                f"[release->refactor] step {step_idx}: island node 'iso' "
                f"drifted to v={v!r} (expected ~{vdd!r} V, pinned)."
            )
        assert iso_wave[-1] > 0.9 * vdd

    def test_dc_first_release_then_refactor_no_double_counted_island_penalty(
        self, tmp_path,
    ):
        """Major finding (review round 2): the documented DC-first lifecycle
        (CLAUDE.md: ``dc_ctx = solver.prepare()`` BEFORE
        ``solver.prepare_transient()``) makes ``solver._topology`` (and
        hence ``trans_ctx.topology``) the DC never-assemble topology, whose
        ``rhs_dirichlet_G`` ALREADY has the island penalty folded in (see
        ``_factor_dc_context_no_s_global``). Pre-fix,
        ``_refactor_transient_context``'s never-assemble branch rebuilt
        ``C_package_uu``/``_G_package_uu``/``island_penalty_rhs`` but NOT
        ``rhs_dirichlet_A``/``_rhs_dirichlet_G`` -- which ``release()``
        (result.py) also clears to ``None`` -- so the ``rhs_dirichlet_G``
        property (result.py) silently fell back to the shared DC topology's
        penalty-included value after a release()->refactor() cycle, on top
        of the separate per-step ``island_penalty_rhs`` term the time loop
        (solver_td.py) already adds -- double-counting the penalty. Exact
        repro (matches the reviewer's reported numbers): on
        ``_build_three_tile_island_model(cap_c_fF=0.0)``, pre-fix
        ``rhs_dirichlet_G['iso']`` == ``island_penalty_rhs['iso']`` ==
        100000.0 after DC-first release()->refactor(), and node 'iso'
        settles at 2*vdd instead of vdd. TD-first (no DC prepare() before
        prepare_transient()) does NOT reproduce this -- see
        ``test_release_then_refactor_preserves_island_penalty`` above, which
        passes both pre- and post-fix.
        """
        from distributed.solver import DistributedDDMSolver

        model = _build_three_tile_island_model(cap_c_fF=0.0)
        model.island_detection_mode = 'summaries'
        model.component_summaries = []
        model.settings.update(_never_assemble_settings())
        vdd = model.vdd

        solver = DistributedDDMSolver(model)
        # DC-FIRST: populates solver._topology with the DC never-assemble
        # topology (rhs_dirichlet_G penalty-included) BEFORE
        # prepare_transient() ever runs -- the documented lifecycle order.
        dc_first = solver.prepare()
        dc_first.release()
        trans_ctx = solver.prepare_transient(dt=1e-10, method='be')
        assert trans_ctx.island_penalty_rhs is not None, (
            "fixture sanity: 'iso' must be a TD-mode island requiring the "
            "penalty RHS"
        )
        iso_idx = trans_ctx.interface_node_to_idx['iso']
        # Sanity on the freshly-factored (not yet released/refactored)
        # context: the penalty must live ONLY in island_penalty_rhs, never
        # folded into rhs_dirichlet_G (the vector the time loop reads
        # directly, BE +1x / TR +2x per step).
        assert abs(trans_ctx.rhs_dirichlet_G[iso_idx]) < 1e-6
        assert abs(trans_ctx.island_penalty_rhs[iso_idx] - 1e5 * vdd) < 1e-6

        trans_ctx.release()
        assert trans_ctx.rhs_dirichlet_A is None  # release() clears it
        assert trans_ctx._rhs_dirichlet_G is None  # release() clears it
        trans_ctx.refactor()
        try:
            # The bug under test: post-refactor, rhs_dirichlet_G must STILL
            # be penalty-free -- not silently inherit the shared DC
            # topology's penalty-included value via the stale-fallback
            # property (result.py's rhs_dirichlet_G getter).
            assert abs(trans_ctx.rhs_dirichlet_G[iso_idx]) < 1e-6, (
                f"rhs_dirichlet_G['iso']={trans_ctx.rhs_dirichlet_G[iso_idx]!r} "
                f"after DC-first release->refactor -- expected ~0.0 (the "
                f"penalty belongs only in island_penalty_rhs; a nonzero "
                f"value here means the time loop double-counts it)."
            )
            assert trans_ctx.island_penalty_rhs is not None
            assert abs(trans_ctx.island_penalty_rhs[iso_idx] - 1e5 * vdd) < 1e-6

            dc_ctx = solver.prepare()
            smoothed = solver.preprocess_sources(
                time_step=1e-10, t_start=0.0, t_end=2e-9, smooth=False,
                pkl_dir=str(tmp_path),
            )
            result = solver.solve_transient(
                trans_ctx, dc_context=dc_ctx, t_end=2e-9,
                smoothed_sources=smoothed,
                track_nodes=['iso', 'shared', 'b1'],
            )
            dc_ctx.release()
        finally:
            trans_ctx.release()
            model.shutdown()

        iso_wave = result.tracked_waveforms.get('iso')
        assert iso_wave is not None and len(iso_wave) > 1
        for step_idx, v in enumerate(iso_wave):
            assert abs(vdd - v) < 1e-3, (
                f"[DC-first release->refactor] step {step_idx}: island "
                f"node 'iso' drifted to v={v!r} (expected ~{vdd!r} V -- "
                f"2x vdd indicates the double-counted island-penalty "
                f"regression)."
            )

    def test_refactor_without_workers_raises(self, tmp_path):
        """A never-assembled context has no S_global to fall back to -- if
        workers are detached, refactor() must raise (not silently produce a
        broken 'assembled' solver, which is impossible here).

        Unlike the assembled DC/TD siblings (TestRefactorRebuildsTilewise),
        there is no save()/load() route to a fresh "no workers" stand-in for
        a never-assembled context -- save() itself raises (tested above).
        model.shutdown() alone does not empty model.workers (it only tears
        down the backend), so "workers detached" is simulated directly by
        clearing the live model's own .workers list post-release().
        """
        from distributed.solver import DistributedDDMSolver

        model = _summaries_two_tile_model([], tmp_path, 'noworkers')
        model.settings.update(_never_assemble_settings())
        solver = DistributedDDMSolver(model)
        trans_ctx = solver.prepare_transient(dt=1e-10, method='be')
        try:
            trans_ctx.release()
            model.workers = []  # simulate "no workers attached"

            with pytest.raises(RuntimeError, match='Cannot refactor'):
                trans_ctx.refactor()
        finally:
            # Finding 11 (round 2 review): teardown parity with this file's
            # other tests -- shutdown() is a LocalBackend no-op given
            # model.workers was already cleared above, but keeps this test
            # consistent (and safe if run against a backend where shutdown
            # does more than tear down the backend handle).
            model.shutdown()

    def test_stale_ordering_drift_raises(self, tmp_path):
        """Mirrors TestT3RefactorStaleOrderingRaisesLoudly (DC, assembled)
        for the never-assemble TD path: a LIVE worker's port list containing
        a node absent from the context's own (stale) interface_node_to_idx
        must raise a loud ValueError, not silently drop a row/column."""
        from distributed.solver import DistributedDDMSolver

        model = _summaries_two_tile_model([], tmp_path, 'drift')
        model.settings.update(_never_assemble_settings(preconditioner='jacobi'))
        solver = DistributedDDMSolver(model)
        trans_ctx = solver.prepare_transient(dt=1e-10, method='be')
        assert trans_ctx._cg_solver.matvec_mode == 'tilewise'

        trans_ctx.release()

        worker0 = model.workers[0]
        assert model.metadata.tile_configs[0].tile_id == (0, 0)
        orig_method = worker0.factor_transient_system

        def _tampered_factor_transient_system(dt_scaled, method='be'):
            S, port_nodes, total_cap, stats = orig_method(dt_scaled, method)
            S_arr = np.asarray(S, dtype=np.float64)
            n = S_arr.shape[0]
            S_new = np.zeros((n + 1, n + 1), dtype=np.float64)
            S_new[:n, :n] = S_arr
            S_new[n, n] = 1.0
            port_nodes_new = list(port_nodes) + ['drifted_phantom_node']
            return S_new, port_nodes_new, total_cap, stats

        worker0.factor_transient_system = _tampered_factor_transient_system
        try:
            with pytest.raises(ValueError, match='drifted_phantom_node'):
                trans_ctx.refactor()
        finally:
            worker0.factor_transient_system = orig_method
            trans_ctx.release()
            model.shutdown()


# ---------------------------------------------------------------------------
# 6. Both-contexts coexistence: DC + TD never-assemble contexts alive
#    together; solve DC then transient via dc_context= AND ic_voltages=.
# ---------------------------------------------------------------------------


class TestBothContextsCoexistence:

    def test_dc_and_td_never_assemble_coexist(self, tmp_path):
        from distributed.solver import DistributedDDMSolver

        model = _summaries_two_tile_model(
            [('pad', 'shared', 50.0)], tmp_path, 'coexist',
        )
        model.settings.update(_never_assemble_settings())
        solver = DistributedDDMSolver(model)

        dc_ctx = solver.prepare()
        trans_ctx = solver.prepare_transient(dt=1e-10, method='be')
        try:
            assert dc_ctx._S_global is None
            assert trans_ctx._S_global is None
            assert dc_ctx._cg_solver is not None
            assert trans_ctx._cg_solver is not None
            assert dc_ctx._cg_solver is not trans_ctx._cg_solver
            # DC ≠ TD blocks: distinct dense per-tile block sets (S_G vs
            # S_A), never shared/aliased.
            assert (
                dc_ctx._cg_solver.tile_schur_complements
                is not trans_ctx._cg_solver.tile_schur_complements
            )
            for tid in dc_ctx._cg_solver.tile_schur_complements:
                S_dc = dc_ctx._cg_solver.tile_schur_complements[tid]
                S_td = trans_ctx._cg_solver.tile_schur_complements[tid]
                assert S_dc is not S_td

            dc_result = solver.solve_dc(dc_ctx)
            smoothed = solver.preprocess_sources(
                time_step=1e-10, t_start=0.0, t_end=1e-9, smooth=False,
                pkl_dir=str(tmp_path / 'coexist'),
            )

            # Path A: dc_context= handoff.
            result_a = solver.solve_transient(
                trans_ctx, dc_context=dc_ctx, t_end=1e-9,
                smoothed_sources=smoothed,
            )
            # Path B: ic_voltages= (skip DC solve inside solve_transient,
            # reuse the dc_result computed above).
            trans_ctx._cg_solver.reset_warm_start()
            result_b = solver.solve_transient(
                trans_ctx, ic_voltages=dc_result, t_end=1e-9,
                smoothed_sources=smoothed,
            )

            flat_a, flat_b = result_a.as_flat(), result_b.as_flat()
            common = set(flat_a) & set(flat_b)
            assert common
            for node in common:
                d_a, _ = flat_a[node]
                d_b, _ = flat_b[node]
                assert abs(d_a - d_b) < 1e-9, (
                    f"node {node}: dc_context path drop={d_a!r} vs "
                    f"ic_voltages path drop={d_b!r}"
                )

            # Both DC and TD contexts are still independently usable after
            # the transient solves (no state bleed).
            assert dc_ctx._S_global is None
            assert trans_ctx._S_global is None
        finally:
            trans_ctx.release()
            dc_ctx.release()
            model.shutdown()

    def test_matches_single_context_runs(self, tmp_path):
        """Results from a session with BOTH never-assemble contexts alive
        must match a session that only ever has one alive at a time."""
        from distributed.solver import DistributedDDMSolver

        pkg_caps = [('pad', 'shared', 50.0)]

        # Combined session.
        model_c = _summaries_two_tile_model(pkg_caps, tmp_path, 'combined')
        model_c.settings.update(_never_assemble_settings())
        solver_c = DistributedDDMSolver(model_c)
        dc_c = solver_c.prepare()
        trans_c = solver_c.prepare_transient(dt=1e-10, method='be')
        try:
            smoothed_c = solver_c.preprocess_sources(
                time_step=1e-10, t_start=0.0, t_end=1e-9, smooth=False,
                pkl_dir=str(tmp_path / 'combined'),
            )
            result_c = solver_c.solve_transient(
                trans_c, dc_context=dc_c, t_end=1e-9,
                smoothed_sources=smoothed_c,
            )
            flat_c = result_c.as_flat()
        finally:
            trans_c.release()
            dc_c.release()
            model_c.shutdown()

        # TD-only session (DC context released before TD is even prepared).
        model_s = _summaries_two_tile_model(pkg_caps, tmp_path, 'solo')
        model_s.settings.update(_never_assemble_settings())
        solver_s = DistributedDDMSolver(model_s)
        dc_s = solver_s.prepare()
        dc_s.release()
        trans_s = solver_s.prepare_transient(dt=1e-10, method='be')
        dc_s2 = solver_s.prepare()  # re-factor DC for the IC solve
        try:
            smoothed_s = solver_s.preprocess_sources(
                time_step=1e-10, t_start=0.0, t_end=1e-9, smooth=False,
                pkl_dir=str(tmp_path / 'solo'),
            )
            result_s = solver_s.solve_transient(
                trans_s, dc_context=dc_s2, t_end=1e-9,
                smoothed_sources=smoothed_s,
            )
            flat_s = result_s.as_flat()
        finally:
            trans_s.release()
            dc_s2.release()
            model_s.shutdown()

        common = set(flat_c) & set(flat_s)
        assert common
        for node in common:
            d_c, _ = flat_c[node]
            d_s, _ = flat_s[node]
            assert abs(d_c - d_s) < 1e-9, (
                f"node {node}: combined-session drop={d_c!r} vs "
                f"solo-session drop={d_s!r}"
            )


# ---------------------------------------------------------------------------
# 7. Adjoint on a never-assemble TD context: the adjoint code (solver_
#    adjoint.py) has NO ctx._interface_solver_mode / ctx._S_global check at
#    all -- it calls ctx.interface_lu(...) generically (see interface_
#    iterative.py's "Adjoint note"), which already works transparently in
#    direct/CG-assembled/CG-tilewise mode.  A never-assembled context's
#    interface_lu is built by the exact same build_interface_solver() CG
#    tilewise path as an explicitly-tilewise assembled context, so this
#    "just works" with no special-casing -- verified here against the
#    assembled-CG adjoint result (S is symmetric either way).
# ---------------------------------------------------------------------------


class TestAdjointNeverAssemble:

    def _run(self, drop_s_global, tmp_path, subdir):
        from distributed.solver import DistributedDDMSolver

        model = _summaries_two_tile_model(
            [('pad', 'shared', 50.0)], tmp_path, subdir,
        )
        settings = (
            _never_assemble_settings() if drop_s_global
            else _assembled_cg_settings()
        )
        model.settings.update(settings)
        solver = DistributedDDMSolver(model)
        dc_ctx = solver.prepare()
        trans_ctx = solver.prepare_transient(dt=1e-10, method='be')
        try:
            assert (trans_ctx._S_global is None) == drop_s_global
            smoothed = solver.preprocess_sources(
                time_step=1e-10, t_start=0.0, t_end=2e-9, smooth=False,
                pkl_dir=str(tmp_path / subdir),
            )
            result = solver.solve_transient(
                trans_ctx, dc_context=dc_ctx, t_end=2e-9,
                smoothed_sources=smoothed,
            )
            attribution = solver.analyze_adjoint(
                victim_nodes='b1', observation_time=1.5e-9,
                trans_context=trans_ctx, transient_result=result,
                memory_window=10, initial_condition='zero',
            )
        finally:
            trans_ctx.release()
            dc_ctx.release()
            model.shutdown()
        return attribution

    def test_matches_assembled_cg_adjoint(self, tmp_path):
        attr_assembled = self._run(False, tmp_path, 'adj_assembled')
        attr_never_assemble = self._run(True, tmp_path, 'adj_never_assemble')

        assert abs(
            attr_assembled.ir_drop_at_T - attr_never_assemble.ir_drop_at_T
        ) < 1e-6, (
            f"ir_drop_at_T: assembled={attr_assembled.ir_drop_at_T!r} vs "
            f"never-assemble={attr_never_assemble.ir_drop_at_T!r}"
        )
        assert abs(
            attr_assembled.total_attributed_ir_drop
            - attr_never_assemble.total_attributed_ir_drop
        ) < 1e-6


# ---------------------------------------------------------------------------
# 8. Decompose/warm-start determinism: two sequential solve_transient calls
#    on one TD never-assemble ctx -> reproducible iteration counts.
# ---------------------------------------------------------------------------


class TestWarmStartDeterminism:

    def test_repeated_solve_transient_reproducible_iters(self, tmp_path):
        from distributed.solver import DistributedDDMSolver

        model = _summaries_two_tile_model(
            [('pad', 'shared', 50.0)], tmp_path, 'warm',
        )
        model.settings.update(_never_assemble_settings())
        solver = DistributedDDMSolver(model)
        dc_ctx = solver.prepare()
        trans_ctx = solver.prepare_transient(dt=1e-10, method='be')
        try:
            assert trans_ctx._S_global is None
            smoothed = solver.preprocess_sources(
                time_step=1e-10, t_start=0.0, t_end=1e-9, smooth=False,
                pkl_dir=str(tmp_path / 'warm'),
            )

            trans_ctx._cg_solver.reset_warm_start()
            result1 = solver.solve_transient(
                trans_ctx, dc_context=dc_ctx, t_end=1e-9,
                smoothed_sources=smoothed,
            )
            iters1 = result1.solve_metadata['timings']['loop_stats'].get(
                'cg_iters_per_step',
            )

            trans_ctx._cg_solver.reset_warm_start()
            result2 = solver.solve_transient(
                trans_ctx, dc_context=dc_ctx, t_end=1e-9,
                smoothed_sources=smoothed,
            )
            iters2 = result2.solve_metadata['timings']['loop_stats'].get(
                'cg_iters_per_step',
            )

            assert iters1 is not None and iters2 is not None, (
                "sanity: CG solver must report per-step iteration counts"
            )
            assert iters1 == iters2, (
                f"iteration counts not reproducible across repeated "
                f"solve_transient calls (reset_warm_start between): "
                f"{iters1!r} vs {iters2!r}"
            )

            flat1, flat2 = result1.as_flat(), result2.as_flat()
            for node in set(flat1) & set(flat2):
                d1, _ = flat1[node]
                d2, _ = flat2[node]
                assert abs(d1 - d2) < 1e-12, (
                    f"node {node}: repeated solve produced different "
                    f"results: {d1!r} vs {d2!r}"
                )
        finally:
            trans_ctx.release()
            dc_ctx.release()
            model.shutdown()


# ---------------------------------------------------------------------------
# 9. Finding 4: two_level coarse-space preconditioner in never-assemble TD
#    mode. Every test above (and in test_interface_iterative_stage2.py)
#    pins interface_preconditioner='block_jacobi' -- the GenEO-lite coarse
#    build on the dt-scaled transient A blocks (interface_coarse_geneo_*
#    knobs + island_idx, threaded through build_interface_solver in
#    _factor_transient_context_no_s_global) was never exercised.
# ---------------------------------------------------------------------------


class TestTwoLevelPreconditionerNeverAssemble:

    @pytest.mark.parametrize('method', ['be', 'trap'])
    def test_two_level_converges_and_matches_assembled(self, method, tmp_path):
        # Package cap present so S_extra^TD's dt-dependence (and hence the
        # coarse build's input operator) is genuinely dt/method-scaled, not
        # just the DC-identical resistive-only case.
        pkg_caps = [('pad', 'shared', 100.0)]
        dt = 1e-10
        t_end = 5 * dt

        model_a = _summaries_two_tile_model(pkg_caps, tmp_path, 'tl_a')
        model_a.settings.update(_assembled_cg_settings(preconditioner='two_level'))
        result_a, trans_a, dc_a, _ = _solve_transient_two_tile(
            model_a, method, dt, t_end=t_end, pkl_dir=tmp_path / 'tl_a',
            track_nodes=['a1', 'b1', 'shared'],
        )
        try:
            # 'two_level(' (additive, untagged) or 'two_level[deflated](' --
            # this test's invariant is "coarse space actually built, not
            # silently degraded to block_jacobi/jacobi", independent of
            # which apply_mode interface_coarse.DEFAULT_APPLY_MODE resolves
            # to (2026-07-20 flip: 'deflated'). Neither settings dict above
            # pins interface_coarse_apply_mode, so both paths pick up
            # whatever the canonical default currently is.
            label_a = trans_a._cg_solver.preconditioner_label
            assert label_a.startswith('two_level(') or label_a.startswith('two_level['), (
                f"fixture sanity: assembled reference must resolve to "
                f"two_level, got {label_a!r}"
            )
            assert trans_a._cg_solver._coarse is not None
            n_geneo_cols_a = trans_a._cg_solver._coarse.n_geneo_cols
            flat_a = result_a.as_flat()
        finally:
            trans_a.release()
            dc_a.release()
            model_a.shutdown()

        model_n = _summaries_two_tile_model(pkg_caps, tmp_path, 'tl_n')
        model_n.settings.update(_never_assemble_settings(preconditioner='two_level'))
        result_n, trans_n, dc_n, _ = _solve_transient_two_tile(
            model_n, method, dt, t_end=t_end, pkl_dir=tmp_path / 'tl_n',
            track_nodes=['a1', 'b1', 'shared'],
        )
        try:
            assert trans_n._S_global is None
            # The regression this test guards against: a mishandled
            # transient A = G + C_coeff*C tile block (vs. the DC G-based
            # blocks the coarse build was originally validated against)
            # silently degrading two_level to block_jacobi/jacobi, or
            # crashing inside prepare_transient.
            label_n = trans_n._cg_solver.preconditioner_label
            assert label_n.startswith('two_level(') or label_n.startswith('two_level['), (
                f"never-assemble path must also resolve to two_level, not "
                f"silently degrade -- got {label_n!r}"
            )
            assert trans_n._cg_solver._coarse is not None
            # This fixture's tiles have only 1 interior DOF each, so GenEO
            # itself contributes 0 columns beyond the partition-of-unity
            # base on EITHER path (confirmed empirically) -- exercising a
            # genuine near-null GenEO eigendirection needs a much larger
            # per-tile interior (see test_interface_coarse.py's
            # test_additive_M_spd_random_vectors for what that requires).
            # What this DOES verify: the never-assemble path's coarse build
            # reads the SAME dt-scaled transient A blocks as the assembled
            # path and produces an IDENTICAL geneo column count (parity),
            # not just "some" coarse space of unknown provenance.
            assert trans_n._cg_solver._coarse.n_geneo_cols == n_geneo_cols_a, (
                f"never-assemble n_geneo_cols="
                f"{trans_n._cg_solver._coarse.n_geneo_cols!r} != assembled "
                f"n_geneo_cols={n_geneo_cols_a!r} -- the GenEO build must "
                f"read identical dt-scaled transient A blocks on both paths"
            )
            flat_n = result_n.as_flat()
        finally:
            trans_n.release()
            dc_n.release()
            model_n.shutdown()

        common = set(flat_a) & set(flat_n)
        assert common, "fixture produced no comparable nodes"
        for node in common:
            d_a, _ = flat_a[node]
            d_n, _ = flat_n[node]
            assert abs(d_a - d_n) < 1e-9, (
                f"[{method}] node {node}: assembled two_level drop={d_a!r} "
                f"vs never-assemble two_level drop={d_n!r}"
            )


# ---------------------------------------------------------------------------
# 10. Findings 1 & 2: dt-change invalidation across repeated prepare_transient
#     calls on ONE solver/model (never-assemble mode). Workers are shared,
#     single-owner state -- only ONE (dt, method) transient factorization can
#     be resident on them at a time. A second prepare_transient()/refactor()
#     at a different dt silently invalidated the FIRST context's worker-side
#     state pre-fix; solve_transient() now raises a loud RuntimeError (the
#     dt/method stamp guard in solver_td.py) instead of silently mixing
#     mismatched coordinator/worker state.
# ---------------------------------------------------------------------------


class TestDtChangeInvalidation:

    def test_second_prepare_transient_stales_first_context(self, tmp_path):
        """Finding 2: two prepare_transient() calls with different dt on ONE
        solver/model -- the SECOND context must solve correctly (exactness
        vs a clean single-context assembled reference), and the FIRST
        context must now raise the Finding-1 guard error instead of
        silently solving with stale worker-side state."""
        from distributed.solver import DistributedDDMSolver

        pkg_caps = [('pad', 'shared', 50.0)]
        dt1, dt2 = 1e-10, 2e-10
        t_end = 5 * dt2

        # Reference: dt2 solved in a clean, single-context, ASSEMBLED
        # session (no interleaving) -- the exactness oracle for ctx2 below.
        model_ref = _summaries_two_tile_model(pkg_caps, tmp_path, 'dtchg_ref')
        model_ref.settings.update(_assembled_cg_settings())
        result_ref, trans_ref, dc_ref, _ = _solve_transient_two_tile(
            model_ref, 'be', dt2, t_end=t_end, pkl_dir=tmp_path / 'dtchg_ref',
        )
        flat_ref = result_ref.as_flat()
        trans_ref.release()
        dc_ref.release()
        model_ref.shutdown()

        # Interleaved session: two never-assemble contexts on ONE model,
        # dt1 prepared first, dt2 second (workers now factored for dt2).
        model = _summaries_two_tile_model(pkg_caps, tmp_path, 'dtchg')
        model.settings.update(_never_assemble_settings())
        solver = DistributedDDMSolver(model)
        dc_ctx = solver.prepare()
        trans_1 = solver.prepare_transient(dt=dt1, method='be')
        trans_2 = solver.prepare_transient(dt=dt2, method='be')
        try:
            assert trans_1._S_global is None
            assert trans_2._S_global is None
            # Workers are single-owner state: the SECOND prepare_transient()
            # call re-stamped them at dt2, not dt1.
            assert model._worker_transient_factor_key == (trans_2.dt_scaled, 'be')

            smoothed = solver.preprocess_sources(
                time_step=dt2, t_start=0.0, t_end=t_end, smooth=False,
                pkl_dir=str(tmp_path / 'dtchg'),
            )

            # ctx2 (workers' CURRENT dt) solves correctly, matching the
            # clean single-context reference.
            result_2 = solver.solve_transient(
                trans_2, dc_context=dc_ctx, t_end=t_end,
                smoothed_sources=smoothed,
            )
            flat_2 = result_2.as_flat()
            common = set(flat_ref) & set(flat_2)
            assert common, "fixture produced no comparable nodes"
            for node in common:
                d_ref, _ = flat_ref[node]
                d_2, _ = flat_2[node]
                assert abs(d_ref - d_2) < 1e-9, (
                    f"node {node}: ctx2 (dt2, interleaved session) "
                    f"drop={d_2!r} vs clean single-context reference="
                    f"{d_ref!r}"
                )

            # ctx1 (workers no longer factored for dt1) must raise loudly.
            with pytest.raises(RuntimeError, match=r'currently factored'):
                solver.solve_transient(
                    trans_1, dc_context=dc_ctx, t_end=5 * dt1,
                    smoothed_sources=smoothed,
                )
        finally:
            trans_1.release()
            trans_2.release()
            dc_ctx.release()
            model.shutdown()

    def test_refactor_of_stale_context_clobbers_other_context_raises(self, tmp_path):
        """Finding 1 regression -- the reviewer's exact confirmed scenario:
        ctx1 = prepare_transient(dt1); ctx2 = prepare_transient(dt2);
        ctx1.release(); ctx1.refactor() (the ONLY refactor path a
        never-assembled context has) re-gathers/re-factors ALL workers at
        ctx1's dt1, silently clobbering the dt2 worker-side transient
        factorization ctx2 depends on. solve_transient(ctx2) must now raise
        the dt/method stamp-mismatch RuntimeError -- without the Finding-1
        fix, this scenario produces NO error and silently mixes ctx2's
        coordinator-side dt2 vectors with worker-side dt1 per-step
        RHS/interior recovery."""
        from distributed.solver import DistributedDDMSolver

        model = _summaries_two_tile_model(
            [('pad', 'shared', 50.0)], tmp_path, 'refactor_clobber',
        )
        model.settings.update(_never_assemble_settings())
        solver = DistributedDDMSolver(model)
        dc_ctx = solver.prepare()

        dt1, dt2 = 1e-10, 2e-10
        trans_1 = solver.prepare_transient(dt=dt1, method='be')
        trans_2 = solver.prepare_transient(dt=dt2, method='be')
        try:
            assert model._worker_transient_factor_key == (trans_2.dt_scaled, 'be')

            trans_1.release()
            trans_1.refactor()  # re-gathers/re-factors workers at dt1
            assert model._worker_transient_factor_key == (trans_1.dt_scaled, 'be'), (
                "fixture sanity: refactor() must re-stamp the workers at "
                "ITS OWN dt (dt1), overwriting ctx2's dt2 stamp -- this is "
                "the clobbering step this regression targets."
            )

            smoothed = solver.preprocess_sources(
                time_step=dt2, t_start=0.0, t_end=5 * dt2, smooth=False,
                pkl_dir=str(tmp_path / 'refactor_clobber'),
            )
            with pytest.raises(RuntimeError, match=r'currently factored'):
                solver.solve_transient(
                    trans_2, dc_context=dc_ctx, t_end=5 * dt2,
                    smoothed_sources=smoothed,
                )
        finally:
            trans_1.release()
            trans_2.release()
            dc_ctx.release()
            model.shutdown()


# ---------------------------------------------------------------------------
# 11. Round-2 review findings 1-4: worker dt/method stamp guard hardening.
#     Finding 1: stamp written AFTER the full worker gather, invalidated
#       (None) BEFORE dispatch -- a mid-gather failure must leave the stamp
#       at None ("state unknown"), not a stale-but-plausible value.
#     Finding 2: analyze_adjoint() consumes the identical shared worker-side
#       transient factorization solve_transient() does -- must be guarded
#       too.
#     Finding 3: method comparison is case-canonicalized ('be' == 'BE').
#     Finding 4: release() must clear the stamp alongside the worker-side
#       clear_transient_factorization() it already unconditionally issues.
# ---------------------------------------------------------------------------


class TestRound2StampGuardHardening:

    def test_mid_gather_failure_invalidates_stamp_to_none(self, tmp_path):
        """Finding 1: a worker failure PARTWAY THROUGH a second
        prepare_transient()'s streamed tile gather (tile 0 already
        re-factored at dt2, tile 1 never reached) must leave the model's
        stamp at None -- not the stale dt1 value a naive "only write after
        success" implementation (without invalidate-BEFORE-dispatch) would
        leave behind. solve_transient() on the FIRST (still is_factored=
        True) context must then raise loudly instead of silently passing
        a not-None guard against a worker set that is no longer at dt1
        (tile 0 was re-factored to dt2; only tile 1's dt1 state survives).

        Negative-test evidence (manual, see the round-2 fix report): with
        the invalidate-before-dispatch removed (i.e. only calling
        _stamp_worker_transient_factor after a successful gather, as
        round-1 code did), this same mid-gather failure leaves the stamp
        at its STALE (dt1, 'be') value (never touched by the failed dt2
        dispatch) -- the `model._worker_transient_factor_key is None`
        assertion below fails, and solve_transient(trans_1) does NOT raise
        (the stale stamp matches trans_1's own key), reproducing exactly
        the silent-wrong-voltage hazard this fix closes.
        """
        from distributed.solver import DistributedDDMSolver

        model = _summaries_two_tile_model([], tmp_path, 'midgather')
        model.settings.update(_never_assemble_settings())
        solver = DistributedDDMSolver(model)
        dc_ctx = solver.prepare()

        dt1, dt2 = 1e-10, 2e-10
        trans_1 = solver.prepare_transient(dt=dt1, method='be')
        try:
            assert model._worker_transient_factor_key == (trans_1.dt_scaled, 'be')

            # LocalBackend.call_all_streaming processes tiles in order (0
            # then 1) and yields lazily -- monkeypatching tile 1's
            # factor_transient_system to raise means tile 0's RPC has
            # ALREADY completed (re-factored at dt2) by the time the
            # failure occurs: a genuine partial/mid-gather failure, not a
            # simulated one.
            assert model.metadata.tile_configs[1].tile_id != model.metadata.tile_configs[0].tile_id
            worker1 = model.workers[1]

            def _boom(*args, **kwargs):
                raise RuntimeError("simulated mid-gather worker failure")

            worker1.factor_transient_system = _boom
            try:
                with pytest.raises(RuntimeError, match='simulated mid-gather'):
                    solver.prepare_transient(dt=dt2, method='be')
            finally:
                del worker1.factor_transient_system

            # Finding 1: the failed dispatch must have invalidated the
            # stamp to None -- NOT left it at the stale dt1 value.
            assert model._worker_transient_factor_key is None

            # Finding 1: solve_transient(trans_1) must now raise loudly:
            # the workers' actual state is an inconsistent, unknown mix
            # (tile 0 re-factored at dt2, tile 1 still at dt1) -- a stale
            # not-None dt1 stamp would otherwise let this pass silently.
            smoothed = solver.preprocess_sources(
                time_step=dt1, t_start=0.0, t_end=5 * dt1, smooth=False,
                pkl_dir=str(tmp_path / 'midgather'),
            )
            with pytest.raises(RuntimeError, match='unknown or absent'):
                solver.solve_transient(
                    trans_1, dc_context=dc_ctx, t_end=5 * dt1,
                    smoothed_sources=smoothed,
                )
        finally:
            trans_1.release()
            dc_ctx.release()
            model.shutdown()

    def test_stale_context_analyze_adjoint_raises(self, tmp_path):
        """Finding 2: analyze_adjoint()'s backward sweep consumes the
        SAME shared, single-owner worker-side transient factorization
        solve_transient() does (tile_worker_adjoint's lambda-recovery RPCs
        require factor_transient_system() to already have run on the
        workers) -- the reviewer's exact confirmed scenario: ctx1 =
        prepare_transient(dt1); ctx2 = prepare_transient(dt2) (workers now
        factored for dt2); analyze_adjoint(trans_context=ctx1) must now
        raise the SAME stamp-mismatch RuntimeError solve_transient(ctx1)
        does, instead of silently running the backward sweep against
        workers factored at dt2 using ctx1's dt1 coordinator-side vectors.
        """
        from distributed.solver import DistributedDDMSolver

        model = _summaries_two_tile_model(
            [('pad', 'shared', 50.0)], tmp_path, 'adj_stamp_guard',
        )
        model.settings.update(_never_assemble_settings())
        solver = DistributedDDMSolver(model)
        dc_ctx = solver.prepare()

        dt1, dt2 = 1e-10, 2e-10
        trans_1 = solver.prepare_transient(dt=dt1, method='be')
        trans_2 = None
        try:
            smoothed_1 = solver.preprocess_sources(
                time_step=dt1, t_start=0.0, t_end=5 * dt1, smooth=False,
                pkl_dir=str(tmp_path / 'adj_stamp_guard'),
            )
            result_1 = solver.solve_transient(
                trans_1, dc_context=dc_ctx, t_end=5 * dt1,
                smoothed_sources=smoothed_1,
            )

            trans_2 = solver.prepare_transient(dt=dt2, method='be')
            assert model._worker_transient_factor_key == (trans_2.dt_scaled, 'be')

            with pytest.raises(RuntimeError, match=r'currently factored'):
                solver.analyze_adjoint(
                    victim_nodes='b1', observation_time=3 * dt1,
                    trans_context=trans_1, transient_result=result_1,
                    memory_window=3, initial_condition='zero',
                )
        finally:
            trans_1.release()
            if trans_2 is not None:
                trans_2.release()
            dc_ctx.release()
            model.shutdown()

    def test_case_variant_method_no_spurious_mismatch(self, tmp_path):
        """Finding 3: 'be' and 'BE' are numerically identical
        configurations -- every numeric consumer treats any method other
        than 'trap' as Backward Euler (``C_coeff = 2.0 if method == 'trap'
        else 1.0``). Preparing one context with method='be' and a second
        with method='BE' at the SAME dt must NOT raise a stamp-mismatch
        error on the first context's solve_transient() -- both describe
        the identical worker factorization, and the stamp is written
        case-canonicalized (.lower()) regardless of which spelling was
        used to prepare.

        Negative-test evidence (manual, see the round-2 fix report): with
        the `.lower()` canonicalization removed from BOTH
        _stamp_worker_transient_factor and _check_worker_transient_stamp,
        this scenario raises RuntimeError('currently factored ... method=
        'BE' ... but this context was prepared for ... method='be'') even
        though both contexts are numerically identical -- the
        `model._worker_transient_factor_key == (..., 'be')` assertion
        below (pre-fix the raw, non-canonicalized 'BE' would be stored
        instead) and the subsequent solve_transient() call both fail.
        """
        from distributed.solver import DistributedDDMSolver

        model = _summaries_two_tile_model([], tmp_path, 'casevariant')
        model.settings.update(_never_assemble_settings())
        solver = DistributedDDMSolver(model)
        dc_ctx = solver.prepare()

        dt = 1e-10
        trans_lower = solver.prepare_transient(dt=dt, method='be')
        trans_upper = solver.prepare_transient(dt=dt, method='BE')
        try:
            # Finding 3: the stamp is written case-canonicalized regardless
            # of which spelling ('be' or 'BE') was used to prepare.
            assert model._worker_transient_factor_key == (trans_upper.dt_scaled, 'be')

            smoothed = solver.preprocess_sources(
                time_step=dt, t_start=0.0, t_end=5 * dt, smooth=False,
                pkl_dir=str(tmp_path / 'casevariant'),
            )
            # Must NOT raise: trans_lower's own method ('be') and the
            # workers' current method ('BE', from trans_upper's more
            # recent prepare_transient()) are numerically identical.
            result = solver.solve_transient(
                trans_lower, dc_context=dc_ctx, t_end=5 * dt,
                smoothed_sources=smoothed,
            )
            assert result is not None
        finally:
            trans_lower.release()
            trans_upper.release()
            dc_ctx.release()
            model.shutdown()

    def test_release_clears_stamp_then_other_context_raises_unknown_state(
        self, tmp_path,
    ):
        """Finding 4: release() must clear
        model._worker_transient_factor_key alongside the worker-side
        clear_transient_factorization() RPC it already unconditionally
        issues -- otherwise the stamp keeps asserting the workers hold a
        (dt, method) factorization they no longer have at all.
        ctx2.release() (called while ctx2 is the model's CURRENT stamp)
        must clear the stamp; solve_transient(ctx1) then raises with the
        accurate "state unknown/absent" wording, not a stale "currently
        factored for dt=<ctx2's dt>" claim about a factorization that no
        longer exists.
        """
        from distributed.solver import DistributedDDMSolver

        model = _summaries_two_tile_model([], tmp_path, 'release_clears')
        model.settings.update(_never_assemble_settings())
        solver = DistributedDDMSolver(model)
        dc_ctx = solver.prepare()

        dt1, dt2 = 1e-10, 2e-10
        trans_1 = solver.prepare_transient(dt=dt1, method='be')
        trans_2 = solver.prepare_transient(dt=dt2, method='be')
        try:
            assert model._worker_transient_factor_key == (trans_2.dt_scaled, 'be')

            trans_2.release()
            assert model._worker_transient_factor_key is None, (
                "Finding 4: release() must clear the stamp, not just the "
                "worker-side transient factorization"
            )

            smoothed = solver.preprocess_sources(
                time_step=dt1, t_start=0.0, t_end=5 * dt1, smooth=False,
                pkl_dir=str(tmp_path / 'release_clears'),
            )
            with pytest.raises(RuntimeError, match='unknown or absent'):
                solver.solve_transient(
                    trans_1, dc_context=dc_ctx, t_end=5 * dt1,
                    smoothed_sources=smoothed,
                )
        finally:
            trans_1.release()
            dc_ctx.release()
            model.shutdown()

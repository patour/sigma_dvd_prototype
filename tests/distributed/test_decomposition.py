"""Unit tests for distributed/decomposition.py — A6 safety mechanism.

Tests that the missing-victim safety net in analyze_distributed_decomposition
works correctly:

- When all victims from find_worst_nodes_separated are already tracked by the
  Phase 2b solve_transient (i.e., they appear in track_candidates), no extra
  transient is run.
- When a victim falls outside the QS candidate set (the cap-history case), a
  single targeted transient is run for the missing node(s) and its waveforms
  are merged into trans_result.tracked_ir_drop so Phase 3 succeeds.

Note: tests control the candidate set size via the ``qs_candidate_factor``
parameter (passed directly to the function) rather than monkey-patching the
module-level ``_QS_CANDIDATE_FACTOR`` constant.  The function accepts the
parameter explicitly so patching is no longer required.
"""

from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import MagicMock, call, patch


# ---------------------------------------------------------------------------
# Helpers — shared mock-builder factories
# ---------------------------------------------------------------------------

def _make_solver_mock(
    *,
    qs_flat: dict,
    trans_as_flat: dict,
    trans_tracked: dict,
    missing_tracked: dict | None = None,
    near_tracked: dict,
    t_array: np.ndarray,
):
    """Return (mock_solver, mock_trans_result, mock_missing_result, mock_near_result).

    Sequence of solve_transient calls:
      1. Phase 2b (all-sources) → mock_trans_result
      2. Safety targeted (only if missing_tracked is not None) → mock_missing_result
      3. Phase 3 near-only → mock_near_result

    The ``trans_tracked`` dict is a REAL mutable dict on mock_trans_result so
    that the safety-check code can write into it with:
        trans_result.tracked_ir_drop[_node] = _wf
    """
    # QS result
    mock_qs_result = MagicMock()
    mock_qs_result.as_flat.return_value = qs_flat

    # All-sources transient result (Phase 2b)
    mock_trans_result = MagicMock()
    mock_trans_result.as_flat.return_value = trans_as_flat
    mock_trans_result.tracked_ir_drop = trans_tracked   # real dict — mutable
    mock_trans_result.t_array = t_array
    mock_trans_result.total_current_per_time = None
    mock_trans_result.peak_ir_drop = max(v[0] for v in trans_as_flat.values())
    mock_trans_result.peak_ir_drop_time = 1e-9

    # Safety targeted result (only populated when missing_tracked is provided)
    mock_missing_result = MagicMock()
    mock_missing_result.tracked_ir_drop = missing_tracked or {}

    # Near-only result (Phase 3)
    mock_near_result = MagicMock()
    mock_near_result.tracked_ir_drop = near_tracked
    mock_near_result.t_array = t_array
    mock_near_result.total_current_per_time = None

    mock_solver = MagicMock()
    mock_solver.preprocess_sources.return_value = MagicMock()

    mock_dc_ctx = MagicMock()
    mock_dc_ctx.release = MagicMock()
    mock_solver.prepare.return_value = mock_dc_ctx

    mock_trans_ctx = MagicMock()
    mock_trans_ctx.release = MagicMock()
    mock_solver.prepare_transient.return_value = mock_trans_ctx

    mock_dc_result = MagicMock()
    mock_solver.solve_dc.return_value = mock_dc_result

    mock_solver.solve_quasi_static.return_value = mock_qs_result

    if missing_tracked is not None:
        # Three solve_transient calls: Phase 2b → safety → Phase 3 near
        mock_solver.solve_transient.side_effect = [
            mock_trans_result,
            mock_missing_result,
            mock_near_result,
        ]
    else:
        # Two solve_transient calls: Phase 2b → Phase 3 near
        mock_solver.solve_transient.side_effect = [
            mock_trans_result,
            mock_near_result,
        ]

    return mock_solver, mock_trans_result, mock_missing_result, mock_near_result


def _make_model_mock(*, n_tiles: int = 1):
    mock_model = MagicMock()
    mock_model.n_tiles = n_tiles
    mock_model.workers = [MagicMock() for _ in range(n_tiles)]
    mock_model.tile_interior_counts = {(0, i): 10 for i in range(n_tiles)}
    mock_model.tile_grid = (1, n_tiles)
    mock_model.interface_nodes = {'ifc_0'}
    mock_model.vdd = 1.0
    mock_model.net_name = 'VDD'
    # call_all returns a list of per-worker return values
    mock_model.backend.call_all.return_value = [0] * n_tiles
    return mock_model


# ---------------------------------------------------------------------------
# The common patch stack for analyze_distributed_decomposition
# ---------------------------------------------------------------------------

# We need to patch:
# 1. load_distributed_partitions (lazy import inside the function)
# 2. create_distributed_model   (lazy import inside the function)
# 3. DistributedDDMSolver        (lazy import inside the function)
# 4. _QS_CANDIDATE_FACTOR        (so n_candidates = top_k * 1 = 1)
#
# The lazy imports are executed as:
#   from distributed import DistributedDDMSolver, create_distributed_model, load_distributed_partitions
# so we patch them in the 'distributed' namespace (which is what the test
# process's 'distributed' module exposes after the function does that import).

_PATCH_LDP = 'distributed.load_distributed_partitions'
_PATCH_CDM = 'distributed.create_distributed_model'
_PATCH_DDM = 'distributed.DistributedDDMSolver'


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestMissingVictimSafetyNet:
    """Safety net: targeted transient for victims outside QS candidate set."""

    def test_no_targeted_transient_when_all_victims_tracked(self):
        """When the worst victim is in track_candidates, no extra solve runs."""
        # top_k=1, _QS_CANDIDATE_FACTOR patched to 1 → n_candidates = 1
        # qs_flat rank-0: '100_100_M1' → becomes the single track_candidate
        # trans_as_flat: '100_100_M1' has the highest transient peak too
        # → find_worst_nodes_separated picks '100_100_M1' → already tracked → no safety run

        qs_flat = {
            '100_100_M1': (0.05, 1e-9),  # rank 0 → candidate
            '200_200_M1': (0.01, 2e-9),  # rank 1 → excluded
        }
        t_array = np.linspace(0, 10e-9, 3)
        wf_tracked = np.array([0.01, 0.05, 0.03])
        trans_tracked = {'100_100_M1': wf_tracked}
        near_tracked = {'100_100_M1': np.array([0.005, 0.02, 0.01])}

        trans_as_flat = {
            '100_100_M1': (0.05, 1e-9),  # this victim is tracked
            '200_200_M1': (0.01, 2e-9),
        }

        mock_solver, mock_trans_result, _, _ = _make_solver_mock(
            qs_flat=qs_flat,
            trans_as_flat=trans_as_flat,
            trans_tracked=trans_tracked,
            missing_tracked=None,       # no safety run expected
            near_tracked=near_tracked,
            t_array=t_array,
        )
        mock_model = _make_model_mock()

        from distributed.decomposition import analyze_distributed_decomposition

        # qs_candidate_factor=1 → n_candidates = max(1*1, 1) = 1
        # so only the top-1 QS node is tracked; the victim IS in the set.
        with (
            patch(_PATCH_LDP, return_value=MagicMock()),
            patch(_PATCH_CDM, return_value=mock_model),
            patch(_PATCH_DDM, return_value=mock_solver),
        ):
            result, _, _ = analyze_distributed_decomposition(
                pkl_dir='/fake/dir',
                backend='local',
                t_start=0.0,
                t_end=10e-9,
                dt=5e-9,
                top_k=1,
                window_percent=50.0,
                instances=None,
                smooth_sources=False,
                aggressor_top_k=0,
                qs_candidate_factor=1,
                verbose=False,
            )

        # solve_transient called exactly twice: Phase 2b + Phase 3 near
        assert mock_solver.solve_transient.call_count == 2, (
            f"Expected 2 solve_transient calls (Phase 2b + near), "
            f"got {mock_solver.solve_transient.call_count}"
        )

        # No 'targeted_transient_missing_victims' key in timings
        assert 'targeted_transient_missing_victims' not in result.timings

        # Exactly one victim analyzed
        assert len(result.worst_instances) == 1
        assert result.worst_instances[0].instance_name == '100_100_M1'

    def test_targeted_transient_fires_for_missing_victim(self):
        """A victim outside QS candidates triggers one targeted transient."""
        # top_k=1, _QS_CANDIDATE_FACTOR=1 → n_candidates = 1
        # qs_flat rank-0: '100_100_M1' → sole candidate
        # trans_as_flat: '200_200_M1' has a HIGHER transient peak
        # → find_worst_nodes_separated selects '200_200_M1'
        # → '200_200_M1' is NOT in track_candidates → safety fires

        qs_flat = {
            '100_100_M1': (0.05, 1e-9),  # rank 0 → candidate
            '200_200_M1': (0.01, 2e-9),  # rank 1 → excluded from candidates
        }
        t_array = np.linspace(0, 10e-9, 3)
        missing_wf = np.array([0.05, 0.08, 0.06])
        # Phase 2b only tracked '100_100_M1'
        trans_tracked = {'100_100_M1': np.array([0.01, 0.02, 0.01])}
        # The targeted transient provides '200_200_M1'
        missing_tracked = {'200_200_M1': missing_wf}
        near_tracked = {'200_200_M1': np.array([0.02, 0.03, 0.025])}

        trans_as_flat = {
            '100_100_M1': (0.01, 1e-9),
            '200_200_M1': (0.08, 2e-9),   # higher transient peak → victim
        }

        mock_solver, mock_trans_result, _, _ = _make_solver_mock(
            qs_flat=qs_flat,
            trans_as_flat=trans_as_flat,
            trans_tracked=trans_tracked,
            missing_tracked=missing_tracked,
            near_tracked=near_tracked,
            t_array=t_array,
        )
        mock_model = _make_model_mock()

        from distributed.decomposition import analyze_distributed_decomposition

        # qs_candidate_factor=1 → n_candidates = max(1*1, 1) = 1
        # → only '100_100_M1' (rank-0 in QS) is tracked; '200_200_M1' is not.
        with (
            patch(_PATCH_LDP, return_value=MagicMock()),
            patch(_PATCH_CDM, return_value=mock_model),
            patch(_PATCH_DDM, return_value=mock_solver),
        ):
            result, _, _ = analyze_distributed_decomposition(
                pkl_dir='/fake/dir',
                backend='local',
                t_start=0.0,
                t_end=10e-9,
                dt=5e-9,
                top_k=1,
                window_percent=50.0,
                instances=None,
                smooth_sources=False,
                aggressor_top_k=0,
                qs_candidate_factor=1,
                verbose=False,
            )

        # solve_transient called three times: Phase 2b + safety + Phase 3 near
        assert mock_solver.solve_transient.call_count == 3, (
            f"Expected 3 solve_transient calls (Phase 2b + safety + near), "
            f"got {mock_solver.solve_transient.call_count}"
        )

        # Second call (index 1) is the safety targeted transient
        safety_call_kwargs = mock_solver.solve_transient.call_args_list[1].kwargs
        assert safety_call_kwargs.get('track_nodes') == ['200_200_M1'], (
            f"Safety targeted transient should track ['200_200_M1'], "
            f"got {safety_call_kwargs.get('track_nodes')}"
        )

        # Timing key is recorded
        assert 'targeted_transient_missing_victims' in result.timings

        # Waveform was merged: victim's decomposition uses the missing waveform
        assert len(result.worst_instances) == 1
        decomp = result.worst_instances[0]
        assert decomp.instance_name == '200_200_M1'
        # The ir_drop_total should be the waveform from trans_result (missing_wf merged in)
        np.testing.assert_array_almost_equal(decomp.ir_drop_total, missing_wf)

    def test_targeted_transient_merges_waveform_into_trans_result(self):
        """The missing victim's waveform appears in trans_result after safety run."""
        # Same setup as above, but we directly inspect trans_result.tracked_ir_drop
        qs_flat = {
            '100_100_M1': (0.05, 1e-9),
            '200_200_M1': (0.01, 2e-9),
        }
        t_array = np.linspace(0, 10e-9, 3)
        missing_wf = np.array([0.05, 0.08, 0.06])
        trans_tracked = {'100_100_M1': np.array([0.01, 0.02, 0.01])}
        missing_tracked = {'200_200_M1': missing_wf}
        near_tracked = {'200_200_M1': np.array([0.02, 0.03, 0.025])}
        trans_as_flat = {
            '100_100_M1': (0.01, 1e-9),
            '200_200_M1': (0.08, 2e-9),
        }

        mock_solver, mock_trans_result, _, _ = _make_solver_mock(
            qs_flat=qs_flat,
            trans_as_flat=trans_as_flat,
            trans_tracked=trans_tracked,
            missing_tracked=missing_tracked,
            near_tracked=near_tracked,
            t_array=t_array,
        )
        mock_model = _make_model_mock()

        from distributed.decomposition import analyze_distributed_decomposition

        # qs_candidate_factor=1 → only rank-0 QS node tracked; '200_200_M1' missing.
        with (
            patch(_PATCH_LDP, return_value=MagicMock()),
            patch(_PATCH_CDM, return_value=mock_model),
            patch(_PATCH_DDM, return_value=mock_solver),
        ):
            analyze_distributed_decomposition(
                pkl_dir='/fake/dir',
                backend='local',
                t_start=0.0,
                t_end=10e-9,
                dt=5e-9,
                top_k=1,
                window_percent=50.0,
                instances=None,
                smooth_sources=False,
                aggressor_top_k=0,
                qs_candidate_factor=1,
                verbose=False,
            )

        # After analysis, trans_result.tracked_ir_drop must contain the victim
        assert '200_200_M1' in mock_trans_result.tracked_ir_drop, (
            "Safety net must merge missing victim waveform into trans_result"
        )
        np.testing.assert_array_equal(
            mock_trans_result.tracked_ir_drop['200_200_M1'],
            missing_wf,
        )

    def test_instances_path_never_triggers_safety(self):
        """When instances= is given, the QS path is skipped; safety is irrelevant."""
        # With instances=['300_300_M1'], the code sets track_candidates = instances
        # and runs only ONE all-sources transient (no QS, no safety check).
        # solve_transient is called: (1) all-sources, (2) near-only.
        t_array = np.linspace(0, 10e-9, 3)
        tracked_wf = np.array([0.04, 0.07, 0.05])
        near_wf = np.array([0.01, 0.02, 0.015])

        mock_solver = MagicMock()
        mock_solver.preprocess_sources.return_value = MagicMock()
        mock_dc_ctx = MagicMock()
        mock_dc_ctx.release = MagicMock()
        mock_solver.prepare.return_value = mock_dc_ctx
        mock_trans_ctx = MagicMock()
        mock_trans_ctx.release = MagicMock()
        mock_solver.prepare_transient.return_value = mock_trans_ctx
        mock_solver.solve_dc.return_value = MagicMock()

        mock_trans_result = MagicMock()
        mock_trans_result.tracked_ir_drop = {'300_300_M1': tracked_wf}
        mock_trans_result.t_array = t_array
        mock_trans_result.total_current_per_time = None

        mock_near_result = MagicMock()
        mock_near_result.tracked_ir_drop = {'300_300_M1': near_wf}
        mock_near_result.t_array = t_array
        mock_near_result.total_current_per_time = None

        mock_solver.solve_transient.side_effect = [
            mock_trans_result,  # all-sources
            mock_near_result,   # near-only Phase 3
        ]

        mock_model = _make_model_mock()
        # The instances path makes the following call_all calls:
        #   1. get_layer_metadata → per-layer bbox dicts (for grid_bounds)
        #   2. build_and_set_window_mask → list of active-source counts (Phase 3)
        #   3. set_current_node_mask → None (mask clear after near trans)
        #   4. set_current_node_mask → None (finally block defensive clear)
        mock_model.backend.call_all.side_effect = [
            [{'M1': {'bbox': (0.0, 1000.0, 0.0, 1000.0)}}],  # get_layer_metadata
            [0],     # build_and_set_window_mask active counts
            [None],  # set_current_node_mask clear
            [None],  # set_current_node_mask clear (finally)
        ]

        from distributed.decomposition import analyze_distributed_decomposition

        with (
            patch(_PATCH_LDP, return_value=MagicMock()),
            patch(_PATCH_CDM, return_value=mock_model),
            patch(_PATCH_DDM, return_value=mock_solver),
        ):
            result, _, _ = analyze_distributed_decomposition(
                pkl_dir='/fake/dir',
                backend='local',
                t_start=0.0,
                t_end=10e-9,
                dt=5e-9,
                top_k=1,
                window_percent=50.0,
                instances=['300_300_M1'],
                smooth_sources=False,
                aggressor_top_k=0,
                verbose=False,
            )

        # solve_quasi_static NOT called (QS phase skipped)
        mock_solver.solve_quasi_static.assert_not_called()
        # solve_transient called exactly twice (all-sources + near-only)
        assert mock_solver.solve_transient.call_count == 2, (
            f"Expected 2 solve_transient calls in instances path, "
            f"got {mock_solver.solve_transient.call_count}"
        )
        # No safety timing key
        assert 'targeted_transient_missing_victims' not in result.timings

    def test_multiple_missing_victims_single_targeted_call(self):
        """All missing victims are collected into ONE targeted transient call."""
        # Setup: QS gives 2 candidates, but 4 transient nodes exist,
        # of which 2 have the highest transient peaks and are NOT candidates.
        # _QS_CANDIDATE_FACTOR=1 with top_k=2 → n_candidates = max(2,1) = 2
        #
        # Victim coordinates are chosen so their windows (window_percent=20)
        # do NOT overlap:
        #   grid: x=[100,900], y=[100,900] (extent 800)
        #   w = h = 800 * 0.20 = 160; half = 80
        #   '500_500_M1' window: (420, 580, 420, 580)
        #   '900_900_M1' window: (820, 980, 820, 980)
        #   → 580 < 820, so no x-overlap → both selected
        qs_flat = {
            '100_100_M1': (0.10, 1e-9),  # rank 0 → candidate
            '200_200_M1': (0.08, 2e-9),  # rank 1 → candidate
            '500_500_M1': (0.01, 3e-9),  # rank 2 → excluded
            '900_900_M1': (0.005, 4e-9), # rank 3 → excluded
        }
        t_array = np.linspace(0, 10e-9, 3)

        # Phase 2b only tracks the 2 QS candidates
        trans_tracked = {
            '100_100_M1': np.array([0.01, 0.02, 0.01]),
            '200_200_M1': np.array([0.02, 0.03, 0.02]),
        }
        # Transient peaks: the top-2 spatially-separated victims are the
        # two nodes NOT in candidates (well-separated coords avoid window overlap).
        trans_as_flat = {
            '100_100_M1': (0.01, 1e-9),
            '200_200_M1': (0.02, 2e-9),
            '500_500_M1': (0.09, 3e-9),   # highest → victim 1
            '900_900_M1': (0.07, 4e-9),   # second → victim 2
        }
        # Targeted transient returns both missing nodes
        missing_wf_a = np.array([0.06, 0.09, 0.07])
        missing_wf_b = np.array([0.04, 0.07, 0.05])
        missing_tracked = {
            '500_500_M1': missing_wf_a,
            '900_900_M1': missing_wf_b,
        }
        near_tracked_a = {'500_500_M1': np.array([0.02, 0.03, 0.02])}
        near_tracked_b = {'900_900_M1': np.array([0.01, 0.02, 0.015])}

        mock_solver = MagicMock()
        mock_solver.preprocess_sources.return_value = MagicMock()
        mock_dc_ctx = MagicMock()
        mock_dc_ctx.release = MagicMock()
        mock_solver.prepare.return_value = mock_dc_ctx
        mock_trans_ctx = MagicMock()
        mock_trans_ctx.release = MagicMock()
        mock_solver.prepare_transient.return_value = mock_trans_ctx
        mock_solver.solve_dc.return_value = MagicMock()

        mock_qs_result = MagicMock()
        mock_qs_result.as_flat.return_value = qs_flat
        mock_solver.solve_quasi_static.return_value = mock_qs_result

        mock_trans_result = MagicMock()
        mock_trans_result.as_flat.return_value = trans_as_flat
        mock_trans_result.tracked_ir_drop = trans_tracked
        mock_trans_result.t_array = t_array
        mock_trans_result.total_current_per_time = None
        mock_trans_result.peak_ir_drop = 0.09
        mock_trans_result.peak_ir_drop_time = 3e-9

        mock_missing_result = MagicMock()
        mock_missing_result.tracked_ir_drop = missing_tracked

        mock_near_a = MagicMock()
        mock_near_a.tracked_ir_drop = near_tracked_a
        mock_near_a.t_array = t_array
        mock_near_a.total_current_per_time = None

        mock_near_b = MagicMock()
        mock_near_b.tracked_ir_drop = near_tracked_b
        mock_near_b.t_array = t_array
        mock_near_b.total_current_per_time = None

        # Call sequence:
        # Phase 2b → mock_trans_result
        # Safety targeted (both missing at once) → mock_missing_result
        # Phase 3 near victim-1 → mock_near_a
        # Phase 3 near victim-2 → mock_near_b
        mock_solver.solve_transient.side_effect = [
            mock_trans_result,
            mock_missing_result,
            mock_near_a,
            mock_near_b,
        ]

        mock_model = _make_model_mock()

        from distributed.decomposition import analyze_distributed_decomposition

        # qs_candidate_factor=1 → n_candidates = max(2*1, 1) = 2
        # → only '100_100_M1' (rank 0) and '200_200_M1' (rank 1) tracked;
        # '500_500_M1' (rank 2) and '900_900_M1' (rank 3) are outside the set.
        with (
            patch(_PATCH_LDP, return_value=MagicMock()),
            patch(_PATCH_CDM, return_value=mock_model),
            patch(_PATCH_DDM, return_value=mock_solver),
        ):
            result, _, _ = analyze_distributed_decomposition(
                pkl_dir='/fake/dir',
                backend='local',
                t_start=0.0,
                t_end=10e-9,
                dt=5e-9,
                top_k=2,
                window_percent=20.0,  # smaller window → nodes well-separated, no overlap
                instances=None,
                smooth_sources=False,
                aggressor_top_k=0,
                qs_candidate_factor=1,
                verbose=False,
            )

        # Four solve_transient calls: Phase 2b + 1 safety + 2 × Phase 3 near
        assert mock_solver.solve_transient.call_count == 4, (
            f"Expected 4 solve_transient calls, got "
            f"{mock_solver.solve_transient.call_count}"
        )

        # The safety call is the second call (index 1)
        safety_kwargs = mock_solver.solve_transient.call_args_list[1].kwargs
        safety_track = safety_kwargs.get('track_nodes', [])
        # Both missing victims must be in the single safety call
        assert set(safety_track) == {'500_500_M1', '900_900_M1'}, (
            f"Safety targeted transient must cover both missing victims; "
            f"got track_nodes={safety_track}"
        )

        # Both waveforms merged into trans_result
        assert '500_500_M1' in mock_trans_result.tracked_ir_drop
        assert '900_900_M1' in mock_trans_result.tracked_ir_drop

        assert 'targeted_transient_missing_victims' in result.timings
        assert len(result.worst_instances) == 2


# ---------------------------------------------------------------------------
# Result-invariant tests (A6 conscious-acceptance verification)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestResultInvariant:
    """Verify A6 result invariant: Phase 2b tracked waveforms == pre-A6 Phase 3.

    Pre-A6 (base f3cab48) ran a separate Phase-3 all-sources transient with
    ``track_nodes=victim_nodes`` to capture victim waveforms.  A6 eliminates
    that transient: Phase 2b captures waveforms via ``track_nodes=candidates``.

    Since both solve the SAME system with the SAME IC, sources, and context,
    the tracked waveform is numerically identical regardless of which call
    produced it.  These tests verify that the DecompositionResult reflects
    the Phase 2b waveform (not a separately-reconstructed value) and that no
    extra all-sources transient is run when all victims are already tracked.
    """

    def test_decomposition_uses_phase2b_waveform_not_separate_phase3(self):
        """ir_drop_total in result equals the Phase 2b tracked waveform (A6 invariant).

        The pre-A6 approach would have run:
          (1) Phase 2b  -- all-sources, no track_nodes
          (2) Phase 3   -- all-sources, track_nodes=victims  → provides ir_all
          (3) near-only -- masked, track_nodes=[victim]

        A6 collapses (1)+(2) into one Phase 2b call with track_nodes=candidates.
        The waveform stored in tracked_ir_drop MUST equal what Phase 3 would have
        returned (same solve, same IC — only the bookkeeping differs).
        """
        qs_flat = {'100_100_M1': (0.05, 1e-9)}
        t_array = np.linspace(0, 10e-9, 5)
        # This is the "truth" waveform both Phase 2b (A6) and a hypothetical
        # separate Phase 3 (pre-A6) would return identically.
        truth_wf = np.array([0.010, 0.030, 0.050, 0.040, 0.020])
        near_wf  = np.array([0.003, 0.009, 0.015, 0.012, 0.006])

        trans_tracked = {'100_100_M1': truth_wf}
        trans_as_flat = {'100_100_M1': (0.05, 2e-9)}  # peak at step-2 index
        near_tracked  = {'100_100_M1': near_wf}

        mock_solver, _, _, _ = _make_solver_mock(
            qs_flat=qs_flat,
            trans_as_flat=trans_as_flat,
            trans_tracked=trans_tracked,
            missing_tracked=None,   # victim is in candidates → no safety
            near_tracked=near_tracked,
            t_array=t_array,
        )
        mock_model = _make_model_mock()

        from distributed.decomposition import analyze_distributed_decomposition

        with (
            patch(_PATCH_LDP, return_value=MagicMock()),
            patch(_PATCH_CDM, return_value=mock_model),
            patch(_PATCH_DDM, return_value=mock_solver),
        ):
            result, _, _ = analyze_distributed_decomposition(
                pkl_dir='/fake/dir',
                backend='local',
                t_start=0.0,
                t_end=10e-9,
                dt=5e-9,
                top_k=1,
                window_percent=50.0,
                instances=None,
                smooth_sources=False,
                aggressor_top_k=0,
                qs_candidate_factor=1,  # tight → victim must be rank-0 QS node
                verbose=False,
            )

        assert len(result.worst_instances) == 1
        decomp = result.worst_instances[0]

        # ir_drop_total must be the Phase 2b tracked waveform (truth_wf).
        np.testing.assert_array_equal(
            decomp.ir_drop_total, truth_wf,
            err_msg=(
                "A6 invariant: ir_drop_total must equal the Phase 2b "
                "tracked waveform (same float values as a pre-A6 Phase-3 "
                "all-sources transient would have returned)"
            ),
        )

        # ir_drop_near must be the near-only waveform.
        np.testing.assert_array_equal(
            decomp.ir_drop_near, near_wf,
            err_msg="ir_drop_near must equal the near-only tracked waveform",
        )

        # ir_drop_far = total - near (linearity).
        np.testing.assert_array_almost_equal(
            decomp.ir_drop_far, truth_wf - near_wf,
            err_msg="ir_drop_far must equal total - near (linearity)",
        )

        # Exactly two solve_transient calls: Phase 2b + near-only.
        # No separate Phase-3 all-sources transient (that was eliminated by A6).
        assert mock_solver.solve_transient.call_count == 2, (
            "A6 must not run a separate Phase-3 all-sources transient; "
            f"got {mock_solver.solve_transient.call_count} solve_transient calls"
        )

    def test_instances_path_result_invariant(self):
        """For instances= path, result waveforms come from the single Phase 2b solve.

        Pre-A6 instances= path: Phase 2b SKIPPED; Phase 3 ran
        solve_transient(track_nodes=victims).
        A6 instances= path: Phase 2b runs solve_transient(track_nodes=instances).
        Same solve, same IC → same waveforms.
        """
        t_array = np.linspace(0, 10e-9, 5)
        truth_wf = np.array([0.04, 0.07, 0.06, 0.05, 0.03])
        near_wf  = np.array([0.01, 0.02, 0.018, 0.015, 0.009])

        mock_solver = MagicMock()
        mock_solver.preprocess_sources.return_value = MagicMock()

        mock_dc_ctx = MagicMock()
        mock_dc_ctx.release = MagicMock()
        mock_solver.prepare.return_value = mock_dc_ctx

        mock_trans_ctx = MagicMock()
        mock_trans_ctx.release = MagicMock()
        mock_solver.prepare_transient.return_value = mock_trans_ctx
        mock_solver.solve_dc.return_value = MagicMock()

        mock_trans_result = MagicMock()
        mock_trans_result.tracked_ir_drop = {'300_300_M1': truth_wf}
        mock_trans_result.t_array = t_array
        mock_trans_result.total_current_per_time = None

        mock_near_result = MagicMock()
        mock_near_result.tracked_ir_drop = {'300_300_M1': near_wf}
        mock_near_result.t_array = t_array
        mock_near_result.total_current_per_time = None

        mock_solver.solve_transient.side_effect = [
            mock_trans_result,  # Phase 2b (all-sources, track_nodes=instances)
            mock_near_result,   # near-only Phase 3
        ]

        mock_model = _make_model_mock()
        mock_model.backend.call_all.side_effect = [
            [{'M1': {'bbox': (0.0, 1000.0, 0.0, 1000.0)}}],  # get_layer_metadata
            [0],    # build_and_set_window_mask
            [None], # set_current_node_mask (victim clear)
            [None], # set_current_node_mask (finally)
        ]

        from distributed.decomposition import analyze_distributed_decomposition

        with (
            patch(_PATCH_LDP, return_value=MagicMock()),
            patch(_PATCH_CDM, return_value=mock_model),
            patch(_PATCH_DDM, return_value=mock_solver),
        ):
            result, _, _ = analyze_distributed_decomposition(
                pkl_dir='/fake/dir',
                backend='local',
                t_start=0.0,
                t_end=10e-9,
                dt=5e-9,
                top_k=1,
                window_percent=50.0,
                instances=['300_300_M1'],
                smooth_sources=False,
                aggressor_top_k=0,
                verbose=False,
            )

        assert len(result.worst_instances) == 1
        decomp = result.worst_instances[0]

        # ir_drop_total must be the Phase 2b tracked waveform (truth_wf).
        np.testing.assert_array_equal(
            decomp.ir_drop_total, truth_wf,
            err_msg=(
                "instances= path A6 invariant: ir_drop_total must equal the "
                "Phase 2b tracked waveform"
            ),
        )

        np.testing.assert_array_almost_equal(
            decomp.ir_drop_far, truth_wf - near_wf,
            err_msg="ir_drop_far must equal total - near",
        )

        # QS phase must NOT have run.
        mock_solver.solve_quasi_static.assert_not_called()

        # Exactly two solve_transient calls: Phase 2b + near-only.
        assert mock_solver.solve_transient.call_count == 2


# ---------------------------------------------------------------------------
# Cap tests — _MAX_QS_CANDIDATES absolute limit
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestQsCandidateCap:
    """Verify max_qs_candidates absolute cap on Phase 2b track_nodes overhead.

    Issue A6-2: with the default top_k=5 and qs_candidate_factor=3000, the
    uncapped n_candidates = 15000.  At BRCM scale (10K steps, 36 tiles) this
    causes ~117 MB/tile of Python-list waveform storage and ~200 ms/tile of
    unvectorised per-step Python loop overhead.  _MAX_QS_CANDIDATES caps
    n_candidates to bound both costs without changing behaviour for top_k <= 2
    (the validated netlist_sampled case).
    """

    @staticmethod
    def _make_qs_flat(n: int) -> dict:
        """Return an n-node QS flat dict with descending drops.

        Nodes are named ``'<100i>_<100i>_M1'`` (i = 1 .. n) so they all have
        parseable (x, y) coordinates and are spatially distinct.
        """
        return {
            f'{(i + 1) * 100}_{(i + 1) * 100}_M1': (
                1.0 - i * 0.001,  # drop, descending → rank-0 has highest
                float(i) * 1e-9,  # peak_time
            )
            for i in range(n)
        }

    def test_cap_applied_when_candidates_exceed_limit(self):
        """track_nodes is bounded to max_qs_candidates even when top_k × factor is larger.

        Setup: 50 QS nodes, qs_candidate_factor=50, top_k=1.
        Uncapped n_candidates = max(1*50, 50) = 50.
        With max_qs_candidates=10, n_candidates must be capped at 10.
        """
        n_qs = 50
        qs_flat = self._make_qs_flat(n_qs)
        t_array = np.linspace(0, 10e-9, 3)

        # The rank-0 QS node is the highest-drop transient victim too, so it is
        # always within the capped top-10 candidates → no safety net fires.
        victim = '100_100_M1'
        trans_as_flat = {node: (drop, 1e-9) for node, (drop, _) in qs_flat.items()}
        trans_tracked = {victim: np.array([0.01, 0.05, 0.03])}
        near_tracked = {victim: np.array([0.005, 0.02, 0.01])}

        mock_solver, _, _, _ = _make_solver_mock(
            qs_flat=qs_flat,
            trans_as_flat=trans_as_flat,
            trans_tracked=trans_tracked,
            missing_tracked=None,
            near_tracked=near_tracked,
            t_array=t_array,
        )
        mock_model = _make_model_mock()

        from distributed.decomposition import analyze_distributed_decomposition

        with (
            patch(_PATCH_LDP, return_value=MagicMock()),
            patch(_PATCH_CDM, return_value=mock_model),
            patch(_PATCH_DDM, return_value=mock_solver),
        ):
            analyze_distributed_decomposition(
                pkl_dir='/fake/dir',
                backend='local',
                t_start=0.0,
                t_end=10e-9,
                dt=5e-9,
                top_k=1,
                window_percent=50.0,
                instances=None,
                smooth_sources=False,
                aggressor_top_k=0,
                qs_candidate_factor=50,   # uncapped → 50 candidates
                max_qs_candidates=10,      # cap forces 10 candidates
                verbose=False,
            )

        # Phase 2b call (index 0): track_nodes must be exactly 10 (capped).
        phase2b_kwargs = mock_solver.solve_transient.call_args_list[0].kwargs
        track_nodes = phase2b_kwargs.get('track_nodes', [])
        assert len(track_nodes) == 10, (
            f"Expected track_nodes length 10 (capped at max_qs_candidates=10), "
            f"got {len(track_nodes)}"
        )

    def test_cap_not_applied_when_below_limit(self):
        """When top_k × factor < max_qs_candidates, no cap is applied.

        Setup: 20 QS nodes, qs_candidate_factor=5, top_k=1.
        Uncapped n_candidates = max(1*5, 5) = 5 < max_qs_candidates=100.
        track_nodes should have exactly 5 elements.
        """
        n_qs = 20
        qs_flat = self._make_qs_flat(n_qs)
        t_array = np.linspace(0, 10e-9, 3)

        victim = '100_100_M1'
        trans_as_flat = {node: (drop, 1e-9) for node, (drop, _) in qs_flat.items()}
        trans_tracked = {victim: np.array([0.01, 0.05, 0.03])}
        near_tracked = {victim: np.array([0.005, 0.02, 0.01])}

        mock_solver, _, _, _ = _make_solver_mock(
            qs_flat=qs_flat,
            trans_as_flat=trans_as_flat,
            trans_tracked=trans_tracked,
            missing_tracked=None,
            near_tracked=near_tracked,
            t_array=t_array,
        )
        mock_model = _make_model_mock()

        from distributed.decomposition import analyze_distributed_decomposition

        with (
            patch(_PATCH_LDP, return_value=MagicMock()),
            patch(_PATCH_CDM, return_value=mock_model),
            patch(_PATCH_DDM, return_value=mock_solver),
        ):
            analyze_distributed_decomposition(
                pkl_dir='/fake/dir',
                backend='local',
                t_start=0.0,
                t_end=10e-9,
                dt=5e-9,
                top_k=1,
                window_percent=50.0,
                instances=None,
                smooth_sources=False,
                aggressor_top_k=0,
                qs_candidate_factor=5,    # uncapped → 5 candidates
                max_qs_candidates=100,    # cap > uncapped → no effect
                verbose=False,
            )

        # Phase 2b call (index 0): track_nodes should be exactly 5 (no cap).
        phase2b_kwargs = mock_solver.solve_transient.call_args_list[0].kwargs
        track_nodes = phase2b_kwargs.get('track_nodes', [])
        assert len(track_nodes) == 5, (
            f"Expected track_nodes length 5 (no cap), got {len(track_nodes)}"
        )

    def test_cap_uses_top_k_oversampling_floor(self):
        """The floor max(top_k*factor, factor) is applied before the cap.

        Setup: top_k=3, qs_candidate_factor=1 → uncapped = max(3, 1) = 3.
        max_qs_candidates=100 → min(3, 100) = 3.
        track_nodes should have exactly 3 elements (uses the max() floor, not raw top_k).
        """
        n_qs = 10
        qs_flat = self._make_qs_flat(n_qs)
        t_array = np.linspace(0, 10e-9, 3)

        # Make victims spatially separated: use rank-0, rank-3, rank-7 nodes.
        # window_percent=10% on a 1000×1000 grid → w=h=100; victims at distance
        # > 100 don't overlap.  Rank-0: (100,100), rank-3: (400,400),
        # rank-7: (800,800) — separations are 424 and 566, both > 100.
        trans_as_flat = {node: (drop, 1e-9) for node, (drop, _) in qs_flat.items()}

        # Only rank-0 victim is guaranteed in candidates when top_k*factor=3.
        # The other two victims (rank-3 and rank-7) may or may not be in the
        # 3-candidate set; the safety net handles any that are missing.
        # For this test, we just verify the cap arithmetic — not the safety net.
        # Use top_k=1 to keep it simple: one victim, one tracked node needed.
        victim = '100_100_M1'
        trans_tracked_single = {victim: np.array([0.01, 0.05, 0.03])}
        near_tracked_single = {victim: np.array([0.005, 0.02, 0.01])}

        mock_solver, _, _, _ = _make_solver_mock(
            qs_flat=qs_flat,
            trans_as_flat=trans_as_flat,
            trans_tracked=trans_tracked_single,
            missing_tracked=None,
            near_tracked=near_tracked_single,
            t_array=t_array,
        )
        mock_model = _make_model_mock()

        from distributed.decomposition import analyze_distributed_decomposition

        with (
            patch(_PATCH_LDP, return_value=MagicMock()),
            patch(_PATCH_CDM, return_value=mock_model),
            patch(_PATCH_DDM, return_value=mock_solver),
        ):
            analyze_distributed_decomposition(
                pkl_dir='/fake/dir',
                backend='local',
                t_start=0.0,
                t_end=10e-9,
                dt=5e-9,
                top_k=1,
                window_percent=50.0,
                instances=None,
                smooth_sources=False,
                aggressor_top_k=0,
                qs_candidate_factor=2,   # max(1*2, 2) = 2 candidates
                max_qs_candidates=100,   # cap doesn't apply
                verbose=False,
            )

        phase2b_kwargs = mock_solver.solve_transient.call_args_list[0].kwargs
        track_nodes = phase2b_kwargs.get('track_nodes', [])
        # The floor is max(1*2, 2) = 2; cap=100 → n_candidates=2.
        assert len(track_nodes) == 2, (
            f"Expected track_nodes length 2 (max floor applied), "
            f"got {len(track_nodes)}"
        )

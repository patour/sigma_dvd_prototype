"""Tests for pwl_source_idx propagation through smoothing pipeline.

Validates the fix for the bug where SmoothedWaveformCache did not store
pwl_source_idx, causing all PWL entries to map to source index 0 when
apply_cache_to_sources() or from_smoothed_cache() reconstructed a
VectorizedCurrentSources.  This made evaluate_per_source_at_time()
(used by multi-RHS) dump ALL PWL current into source 0 → 700 V IR-drop.

Also validates the spatial-peaks fix where ir_drop_arr was recomputed
per mask instead of reusing a stale value from the statistics loop.

Usage:
    python -m unittest tests.test_smoothing_source_idx
    python -m unittest tests.test_smoothing_source_idx -v
"""

import os
import sys
import unittest
from pathlib import Path

import numpy as np
import pytest

from analysis.vectorized_sources import VectorizedCurrentSources
from analysis.transient_solver import TransientIRDropSolver, IntegrationMethod
from model.factory import create_model_from_pdn

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _netlist_test_dir() -> Path:
    return Path(__file__).parent.parent.parent / 'netlist' / 'netlist_test'


def _have_netlist_test() -> bool:
    return _netlist_test_dir().exists()


def _parse_pdn():
    """Parse the test PDN netlist and return (graph, model)."""
    from parser.netlist import NetlistParser
    parser = NetlistParser(str(_netlist_test_dir()), validate=False)
    graph = parser.parse()
    model = create_model_from_pdn(graph, 'VDD')
    return graph, model


# ---------------------------------------------------------------------------
# Test 1 – pwl_source_idx preservation through smoothing (synthetic data)
# ---------------------------------------------------------------------------

class TestPwlSourceIdxPreservation(unittest.TestCase):
    """Verify pwl_source_idx survives the smoothing pipeline."""

    def test_source_idx_preserved_through_smoothing(self):
        """pwl_source_idx should be preserved after create_smoothed_copy.

        Before the fix, SmoothedWaveformCache did not store pwl_source_idx,
        so apply_cache_to_sources set it to all-zeros.  This test constructs
        3 sources (1 DC-only, 2 PWL) with known source indices and verifies
        that after smoothing the per-source evaluation still attributes
        current to the correct source.
        """
        n_sources = 3
        node_to_idx = {'A': 0, 'B': 1, 'C': 2}

        # Source 0 → DC only (node A)
        # Source 1 → PWL on node B, source_idx=1
        # Source 2 → PWL on node C, source_idx=2
        pwl_times = np.array([0.0, 5e-9, 10e-9, 0.0, 5e-9, 10e-9])
        pwl_values = np.array([1.0, 2.0, 1.0, 0.5, 1.5, 0.5])

        vec = VectorizedCurrentSources(
            n_nodes=3,
            node_to_idx=node_to_idx,
            n_sources=n_sources,
            dc_values=np.array([0.1, 0.0, 0.0]),
            source_node_idx=np.array([0, 1, 2], dtype=np.int32),
            n_pulses=0,
            pulse_node_idx=np.array([], dtype=np.int32),
            pulse_source_idx=np.array([], dtype=np.int32),
            pulse_v1=np.array([]),
            pulse_v2=np.array([]),
            pulse_delay=np.array([]),
            pulse_rt=np.array([]),
            pulse_ft=np.array([]),
            pulse_width=np.array([]),
            pulse_period=np.array([]),
            n_pwls=2,
            n_pwl_points=6,
            pwl_node_idx=np.array([1, 2], dtype=np.int32),
            pwl_source_idx=np.array([1, 2], dtype=np.int32),
            pwl_period=np.array([0.0, 0.0]),
            pwl_delay=np.array([0.0, 0.0]),
            pwl_offset=np.array([0, 3], dtype=np.int32),
            pwl_count=np.array([3, 3], dtype=np.int32),
            pwl_times=pwl_times,
            pwl_values=pwl_values,
        )

        smoothed = vec.create_smoothed_copy(
            time_step=1e-9,
            t_start=0,
            t_end=10e-9,
        )

        # pwl_source_idx must still be [1, 2] (not [0, 0])
        self.assertEqual(len(smoothed.pwl_source_idx), smoothed.n_pwls)
        unique_src = set(smoothed.pwl_source_idx.tolist())
        self.assertIn(1, unique_src, "Source 1 must appear in pwl_source_idx")
        self.assertIn(2, unique_src, "Source 2 must appear in pwl_source_idx")
        self.assertNotIn(0, unique_src,
                         "Source 0 (DC-only) must NOT appear in pwl_source_idx")

        # Per-source evaluation at mid-point should attribute current correctly
        per_src = smoothed.evaluate_per_source_at_time(5e-9)
        self.assertAlmostEqual(per_src[0], 0.1, places=5,
                               msg="Source 0 should only have DC=0.1")

    def test_pulse_source_idx_preserved_through_smoothing(self):
        """Pulse sources converted to PWL during smoothing should keep source_idx."""
        n_sources = 2
        node_to_idx = {'X': 0, 'Y': 1}

        vec = VectorizedCurrentSources(
            n_nodes=2,
            node_to_idx=node_to_idx,
            n_sources=n_sources,
            dc_values=np.array([0.0, 0.0]),
            source_node_idx=np.array([0, 1], dtype=np.int32),
            n_pulses=2,
            pulse_node_idx=np.array([0, 1], dtype=np.int32),
            pulse_source_idx=np.array([0, 1], dtype=np.int32),
            pulse_v1=np.zeros(2),
            pulse_v2=np.array([1.0, 2.0]),
            pulse_delay=np.array([0.0, 0.0]),
            pulse_rt=np.array([0.1e-9, 0.1e-9]),
            pulse_ft=np.array([0.1e-9, 0.1e-9]),
            pulse_width=np.array([2e-9, 2e-9]),
            pulse_period=np.array([10e-9, 10e-9]),
            n_pwls=0, n_pwl_points=0,
            pwl_node_idx=np.array([], dtype=np.int32),
            pwl_source_idx=np.array([], dtype=np.int32),
            pwl_period=np.array([]),
            pwl_delay=np.array([]),
            pwl_offset=np.array([], dtype=np.int32),
            pwl_count=np.array([], dtype=np.int32),
            pwl_times=np.array([]),
            pwl_values=np.array([]),
        )

        smoothed = vec.create_smoothed_copy(
            time_step=0.1e-9, t_start=0, t_end=10e-9,
        )

        # Pulses become PWLs after smoothing
        self.assertEqual(smoothed.n_pulses, 0)
        self.assertGreaterEqual(smoothed.n_pwls, 2)

        # Source indices must include both 0 and 1
        unique_src = set(smoothed.pwl_source_idx.tolist())
        self.assertIn(0, unique_src)
        self.assertIn(1, unique_src)


# ---------------------------------------------------------------------------
# Test 2 – multi-RHS with smoothing matches single-RHS
# ---------------------------------------------------------------------------

@unittest.skipUnless(_have_netlist_test(), "netlist/netlist_test not available")
class TestMultiRHSSmoothedMatchesSingleRHS(unittest.TestCase):
    """Multi-RHS with smoothed sources must match single-RHS solve."""

    @classmethod
    def setUpClass(cls):
        cls.graph, cls.model = _parse_pdn()
        cls.solver = TransientIRDropSolver(
            cls.model, cls.graph, vectorize_threshold=0,
        )

    def test_smoothed_multi_rhs_matches_single_rhs(self):
        """All-true mask multi-RHS with smoothing ≈ single-RHS with smoothing.

        Before the fix, pwl_source_idx was all-zeros after smoothing, causing
        evaluate_per_source_at_time() (multi-RHS path) to dump all PWL current
        into source 0 → 700 V.  The single-RHS path used evaluate_at_time()
        which uses pwl_node_idx (correct), so it gave the right ~0.4 mV.
        """
        t_start, t_end, dt = 0.0, 5e-9, 1e-9

        smoothed = self.solver.preprocess_sources(
            dt=dt, t_start=t_start, t_end=t_end,
        )

        # Single-RHS solve (uses evaluate_at_time → pwl_node_idx)
        result_single = self.solver.solve_transient(
            t_start=t_start, t_end=t_end, dt=dt,
            method=IntegrationMethod.BACKWARD_EULER,
            smoothed_sources=smoothed,
        )

        # Multi-RHS solve with all-true mask (uses evaluate_per_source_at_time → pwl_source_idx)
        n_src = self.solver._vec_sources.n_sources
        mask_all = np.ones((1, n_src), dtype=bool)
        results_multi = self.solver.solve_transient_multi_rhs(
            t_start=t_start, t_end=t_end, dt=dt,
            source_masks=mask_all,
            method=IntegrationMethod.BACKWARD_EULER,
            smoothed_sources=smoothed,
        )

        result_multi = results_multi[0]
        peak_single = result_single.peak_ir_drop * 1000  # mV
        peak_multi = result_multi.peak_ir_drop * 1000  # mV

        # They must match within micro-volt precision
        self.assertAlmostEqual(
            peak_single, peak_multi, places=3,
            msg=(f"Smoothed single-RHS ({peak_single:.4f} mV) must match "
                 f"smoothed multi-RHS ({peak_multi:.4f} mV)"),
        )

        # Both must be in the sub-mV range, not 700+ V
        self.assertLess(peak_single, 10.0,
                        "Peak IR-drop should be sub-10 mV for test netlist")
        self.assertLess(peak_multi, 10.0,
                        "Peak IR-drop should be sub-10 mV for test netlist")


# ---------------------------------------------------------------------------
# Test 3 – near/far decomposition linearity with smoothing
# ---------------------------------------------------------------------------

@unittest.skipUnless(_have_netlist_test(), "netlist/netlist_test not available")
class TestNearFarDecompositionSmoothed(unittest.TestCase):
    """Near + far ≈ total when using smoothed sources."""

    @classmethod
    def setUpClass(cls):
        cls.graph, cls.model = _parse_pdn()
        cls.solver = TransientIRDropSolver(
            cls.model, cls.graph, vectorize_threshold=0,
        )

    def test_linearity_with_smoothing(self):
        """Verify V_total ≈ V_near + V_far with smoothed sources.

        Superposition holds point-by-point on the time-domain IR-drop
        waveform (max_ir_drop_per_time) and on tracked node waveforms.
        Note: peak_ir_drop = max over time is NOT linear (triangle ineq.),
        so we check the arrays, not the scalar peaks.

        If pwl_source_idx is wrong, each mask misattributes current, and
        near + far ≠ total.
        """
        t_start, t_end, dt = 0.0, 5e-9, 1e-9

        smoothed = self.solver.preprocess_sources(
            dt=dt, t_start=t_start, t_end=t_end,
        )

        n_src = self.solver._vec_sources.n_sources
        half = n_src // 2

        mask_all = np.ones(n_src, dtype=bool)
        mask_near = np.zeros(n_src, dtype=bool)
        mask_near[:half] = True
        mask_far = np.zeros(n_src, dtype=bool)
        mask_far[half:] = True

        source_masks = np.stack([mask_all, mask_near, mask_far])

        # Pick a track node that has current
        graph_dict = self.graph.graph if hasattr(self.graph, 'graph') else self.graph._attrs
        current_sources = graph_dict.get('_instance_sources_objects', {})
        track_node = None
        for name, src in current_sources.items():
            node = getattr(src, 'node1', None)
            if node and node != '0':
                track_node = node
                break
        track_nodes = [track_node] if track_node else []

        results = self.solver.solve_transient_multi_rhs(
            t_start=t_start, t_end=t_end, dt=dt,
            source_masks=source_masks,
            method=IntegrationMethod.BACKWARD_EULER,
            smoothed_sources=smoothed,
            track_nodes=track_nodes,
        )

        r_total, r_near, r_far = results

        # Point-by-point superposition on tracked node waveform
        # (voltage at a specific node is linear in current excitation)
        self.assertTrue(
            track_node is not None,
            "Need at least one current source node to track",
        )
        self.assertIn(track_node, r_total.tracked_ir_drop)

        wave_total = r_total.tracked_ir_drop[track_node]
        wave_near = r_near.tracked_ir_drop[track_node]
        wave_far = r_far.tracked_ir_drop[track_node]

        np.testing.assert_allclose(
            wave_total, wave_near + wave_far, atol=1e-10,
            err_msg="Waveform superposition should hold point-by-point",
        )


# ---------------------------------------------------------------------------
# Test 4 – spatial peaks correctly computed per-mask
# ---------------------------------------------------------------------------

@unittest.skipUnless(_have_netlist_test(), "netlist/netlist_test not available")
class TestSpatialPeaksPerMask(unittest.TestCase):
    """Spatial peaks (peak_ir_drop_per_node) must differ between distinct masks.

    The bug was that ir_drop_arr was computed once in the statistics loop (for
    the last mask) and reused for ALL masks in the spatial-peaks loop.  After
    the fix, each mask gets its own recomputed ir_drop_arr.
    """

    @classmethod
    def setUpClass(cls):
        cls.graph, cls.model = _parse_pdn()
        cls.solver = TransientIRDropSolver(
            cls.model, cls.graph, vectorize_threshold=0,
        )

    def test_spatial_peaks_differ_between_masks(self):
        """Two disjoint masks should produce different spatial peak maps."""
        n_src = self.solver._vec_sources.n_sources
        half = n_src // 2

        mask0 = np.zeros(n_src, dtype=bool)
        mask0[:half] = True
        mask1 = np.zeros(n_src, dtype=bool)
        mask1[half:] = True
        source_masks = np.stack([mask0, mask1])

        results = self.solver.solve_transient_multi_rhs(
            t_start=0.0, t_end=3e-9, dt=1e-9,
            source_masks=source_masks,
            method=IntegrationMethod.BACKWARD_EULER,
            n_worst_nodes=5,
        )

        self.assertEqual(len(results), 2)

        peak0 = results[0].peak_ir_drop
        peak1 = results[1].peak_ir_drop

        # Peak IR-drop should differ because different sources are active
        self.assertNotAlmostEqual(
            peak0, peak1, places=12,
            msg="Disjoint masks should produce different peak IR-drops",
        )

        # Spatial peak maps should differ on most nodes
        pn0 = results[0].peak_ir_drop_per_node
        pn1 = results[1].peak_ir_drop_per_node

        self.assertTrue(pn0, "peak_ir_drop_per_node should be non-empty for mask 0")
        self.assertTrue(pn1, "peak_ir_drop_per_node should be non-empty for mask 1")

        common = set(pn0.keys()) & set(pn1.keys())
        n_different = sum(1 for n in common if abs(pn0[n] - pn1[n]) > 1e-15)

        self.assertGreater(
            n_different, len(common) // 2,
            msg=(f"Only {n_different}/{len(common)} nodes have different "
                 f"spatial peaks — stale ir_drop_arr bug may persist"),
        )

    def test_spatial_peaks_consistent_with_global_peak(self):
        """Worst spatial peak should match global peak_ir_drop."""
        n_src = self.solver._vec_sources.n_sources
        mask_all = np.ones((1, n_src), dtype=bool)

        results = self.solver.solve_transient_multi_rhs(
            t_start=0.0, t_end=3e-9, dt=1e-9,
            source_masks=mask_all,
            method=IntegrationMethod.BACKWARD_EULER,
        )

        r = results[0]
        if r.peak_ir_drop_per_node:
            worst_spatial = max(r.peak_ir_drop_per_node.values())
            self.assertAlmostEqual(
                r.peak_ir_drop, worst_spatial, places=10,
                msg="Global peak should equal worst spatial peak",
            )


if __name__ == '__main__':
    unittest.main()

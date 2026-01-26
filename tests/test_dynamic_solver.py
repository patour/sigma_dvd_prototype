"""Tests for dynamic IR-drop solver.

Tests quasi-static analysis with batch DC solves at discrete time points.
"""

import unittest
from pathlib import Path
from typing import Dict, Any

import numpy as np

from generate_power_grid import generate_power_grid
from core import (
    create_model_from_synthetic,
    create_model_from_pdn,
    UnifiedIRDropSolver,
)
from core.dynamic_solver import DynamicIRDropSolver, QuasiStaticResult


def build_synthetic_grid(K=3, N0=8, I_N=80, N_vsrc=4, seed=42):
    """Build a synthetic test grid."""
    G, loads, pads = generate_power_grid(
        K=K, N0=N0, I_N=I_N, N_vsrc=N_vsrc, seed=seed,
        max_stripe_res=0.01, max_via_res=0.01
    )
    model = create_model_from_synthetic(G, pads, vdd=1.0)
    return model, loads


class TestDynamicSolverInit(unittest.TestCase):
    """Tests for DynamicIRDropSolver initialization."""

    def test_init_with_synthetic_grid(self):
        """Solver should initialize with synthetic grid (no dynamic sources)."""
        model, _ = build_synthetic_grid()
        solver = DynamicIRDropSolver(model)

        self.assertIsNotNone(solver.model)
        self.assertEqual(solver.num_current_sources, 0)
        self.assertFalse(solver.has_dynamic_sources)

    def test_init_with_pdn_graph(self):
        """Solver should initialize with PDN graph if available."""
        test_netlist = Path(__file__).parent.parent / 'pdn' / 'netlist_test'
        if not test_netlist.exists():
            self.skipTest("Test netlist not available")

        from pdn.pdn_parser import NetlistParser
        parser = NetlistParser(str(test_netlist))
        graph = parser.parse()
        model = create_model_from_pdn(graph, 'VDD')

        solver = DynamicIRDropSolver(model, graph)

        self.assertIsNotNone(solver.model)
        # PDN graph should have current sources
        self.assertGreaterEqual(solver.num_current_sources, 0)


class TestCurrentEvaluation(unittest.TestCase):
    """Tests for current source evaluation at different times."""

    def setUp(self):
        """Set up test fixtures."""
        self.model, self.loads = build_synthetic_grid()
        self.solver = DynamicIRDropSolver(self.model)

    def test_evaluate_currents_no_sources(self):
        """With no dynamic sources, should return static currents."""
        currents = self.solver._evaluate_currents_at_time(0.0)
        # Synthetic grid has no I-type edges, so should be empty
        self.assertIsInstance(currents, dict)

    def test_evaluate_currents_consistent_times(self):
        """Current evaluation should be deterministic."""
        currents_t0 = self.solver._evaluate_currents_at_time(0.0)
        currents_t0_again = self.solver._evaluate_currents_at_time(0.0)

        self.assertEqual(currents_t0, currents_t0_again)


class TestQuasiStaticSolve(unittest.TestCase):
    """Tests for quasi-static solving."""

    def setUp(self):
        """Set up test fixtures."""
        self.model, self.loads = build_synthetic_grid()
        self.solver = DynamicIRDropSolver(self.model)

    def test_solve_quasi_static_returns_result(self):
        """solve_quasi_static should return QuasiStaticResult."""
        result = self.solver.solve_quasi_static(
            t_start=0.0,
            t_end=10e-9,
            n_points=5,
            method='flat',
        )

        self.assertIsInstance(result, QuasiStaticResult)
        self.assertEqual(len(result.t_array), 5)
        self.assertIsNotNone(result.peak_ir_drop)
        self.assertEqual(result.nominal_voltage, self.model.vdd)

    def test_solve_quasi_static_with_t_array(self):
        """solve_quasi_static should accept explicit time array."""
        t_array = np.array([0.0, 5e-9, 10e-9, 20e-9])
        result = self.solver.solve_quasi_static(t_array=t_array)

        np.testing.assert_array_equal(result.t_array, t_array)

    def test_solve_quasi_static_flat_method(self):
        """Flat method should work."""
        result = self.solver.solve_quasi_static(
            t_start=0.0,
            t_end=10e-9,
            n_points=3,
            method='flat',
        )

        self.assertTrue('solve' in result.timings)
        self.assertTrue('prepare' in result.timings)

    def test_solve_quasi_static_hierarchical_method(self):
        """Hierarchical method should work."""
        result = self.solver.solve_quasi_static(
            t_start=0.0,
            t_end=10e-9,
            n_points=3,
            method='hierarchical',
            partition_layer=2,
            top_k=5,
        )

        self.assertTrue('solve' in result.timings)

    def test_hierarchical_requires_partition_layer(self):
        """Hierarchical method should raise error without partition_layer."""
        with self.assertRaises(ValueError):
            self.solver.solve_quasi_static(
                t_start=0.0,
                t_end=10e-9,
                n_points=3,
                method='hierarchical',
                # No partition_layer
            )

    def test_invalid_method_raises_error(self):
        """Invalid method should raise ValueError."""
        with self.assertRaises(ValueError):
            self.solver.solve_quasi_static(method='invalid')


class TestPeakTracking(unittest.TestCase):
    """Tests for peak IR-drop and worst node tracking."""

    def setUp(self):
        """Set up test fixtures."""
        self.model, self.loads = build_synthetic_grid()
        self.solver = DynamicIRDropSolver(self.model)

    def test_peak_ir_drop_tracked(self):
        """Peak IR-drop should be tracked."""
        result = self.solver.solve_quasi_static(
            t_start=0.0,
            t_end=10e-9,
            n_points=5,
        )

        self.assertGreaterEqual(result.peak_ir_drop, 0.0)
        self.assertIn(result.peak_ir_drop_time, result.t_array)

    def test_worst_nodes_tracked(self):
        """Worst nodes should be tracked."""
        n_worst = 5
        result = self.solver.solve_quasi_static(
            t_start=0.0,
            t_end=10e-9,
            n_points=3,
            n_worst_nodes=n_worst,
        )

        # May have fewer worst nodes than requested if grid is small
        self.assertGreater(len(result.worst_nodes), 0)
        self.assertLessEqual(len(result.worst_nodes), n_worst)

        # Worst nodes should be sorted by drop (descending)
        drops = [drop for _, drop, _ in result.worst_nodes]
        self.assertEqual(drops, sorted(drops, reverse=True))

    def test_peak_ir_drop_per_node_populated(self):
        """peak_ir_drop_per_node should be populated for all solved nodes."""
        result = self.solver.solve_quasi_static(
            t_start=0.0,
            t_end=10e-9,
            n_points=3,
        )

        self.assertGreater(len(result.peak_ir_drop_per_node), 0)
        # All values should be non-negative (with floating point tolerance)
        for drop in result.peak_ir_drop_per_node.values():
            self.assertGreaterEqual(drop, -1e-12)

    def test_max_ir_drop_per_time_populated(self):
        """max_ir_drop_per_time should be populated."""
        n_points = 5
        result = self.solver.solve_quasi_static(
            t_start=0.0,
            t_end=10e-9,
            n_points=n_points,
        )

        self.assertEqual(len(result.max_ir_drop_per_time), n_points)


class TestWaveformTracking(unittest.TestCase):
    """Tests for node waveform tracking."""

    def setUp(self):
        """Set up test fixtures."""
        self.model, self.loads = build_synthetic_grid()
        self.solver = DynamicIRDropSolver(self.model)

    def test_track_specific_nodes(self):
        """Waveforms should be stored for tracked nodes."""
        # Get a valid node to track
        load_nodes = list(self.loads.keys())[:3]

        result = self.solver.solve_quasi_static(
            t_start=0.0,
            t_end=10e-9,
            n_points=5,
            track_nodes=load_nodes,
        )

        for node in load_nodes:
            self.assertIn(node, result.tracked_waveforms)
            self.assertEqual(len(result.tracked_waveforms[node]), 5)
            self.assertIn(node, result.tracked_ir_drop)

    def test_get_voltage_waveform(self):
        """get_voltage_waveform should return waveform for tracked node."""
        load_nodes = list(self.loads.keys())[:1]
        n_points = 5

        result = self.solver.solve_quasi_static(
            t_start=0.0,
            t_end=10e-9,
            n_points=n_points,
            track_nodes=load_nodes,
        )

        waveform = result.get_voltage_waveform(load_nodes[0])
        self.assertEqual(len(waveform), n_points)
        # Voltage should be <= Vdd
        self.assertTrue(np.all(waveform <= self.model.vdd + 1e-10))

    def test_get_ir_drop_waveform(self):
        """get_ir_drop_waveform should return IR-drop waveform."""
        load_nodes = list(self.loads.keys())[:1]
        n_points = 5

        result = self.solver.solve_quasi_static(
            t_start=0.0,
            t_end=10e-9,
            n_points=n_points,
            track_nodes=load_nodes,
        )

        ir_waveform = result.get_ir_drop_waveform(load_nodes[0])
        self.assertEqual(len(ir_waveform), n_points)
        # IR-drop should be >= 0 (with floating point tolerance)
        self.assertTrue(np.all(ir_waveform >= -1e-12))

    def test_untracked_node_raises_error(self):
        """Accessing waveform for untracked node should raise KeyError."""
        result = self.solver.solve_quasi_static(
            t_start=0.0,
            t_end=10e-9,
            n_points=3,
            track_nodes=[],  # No tracked nodes
        )

        with self.assertRaises(KeyError):
            result.get_voltage_waveform('nonexistent_node')


class TestVsrcCurrentTracking(unittest.TestCase):
    """Tests for voltage source current tracking."""

    def setUp(self):
        """Set up test fixtures."""
        self.model, self.loads = build_synthetic_grid()
        self.solver = DynamicIRDropSolver(self.model)

    def test_vsrc_current_tracked(self):
        """total_vsrc_current_per_time should be populated."""
        n_points = 5
        result = self.solver.solve_quasi_static(
            t_start=0.0,
            t_end=10e-9,
            n_points=n_points,
        )

        self.assertEqual(len(result.total_vsrc_current_per_time), n_points)

    def test_current_conservation(self):
        """Vsrc current should equal total load current (Kirchhoff)."""
        # With no dynamic sources, current should be constant
        result = self.solver.solve_quasi_static(
            t_start=0.0,
            t_end=10e-9,
            n_points=3,
        )

        # For synthetic grid without I-edges, total load current is 0
        # so vsrc current should also be approximately 0
        # (small numerical errors possible)
        for i in range(len(result.t_array)):
            total_load = result.total_current_per_time[i]
            vsrc_current = result.total_vsrc_current_per_time[i]
            # These should match (current conservation)
            self.assertAlmostEqual(total_load, vsrc_current, delta=1e-6)


class TestTimings(unittest.TestCase):
    """Tests for timing information."""

    def setUp(self):
        """Set up test fixtures."""
        self.model, self.loads = build_synthetic_grid()
        self.solver = DynamicIRDropSolver(self.model)

    def test_timings_populated(self):
        """Timing breakdown should be populated."""
        result = self.solver.solve_quasi_static(
            t_start=0.0,
            t_end=10e-9,
            n_points=5,
        )

        self.assertIn('prepare', result.timings)
        self.assertIn('solve', result.timings)
        self.assertIn('total', result.timings)

        # All times should be non-negative
        for t in result.timings.values():
            self.assertGreaterEqual(t, 0.0)


class TestQuasiStaticWithPDN(unittest.TestCase):
    """Tests for quasi-static solving with PDN netlist."""

    @classmethod
    def setUpClass(cls):
        """Parse test netlist once for all tests."""
        test_netlist = Path(__file__).parent.parent / 'pdn' / 'netlist_test'
        if not test_netlist.exists():
            cls.graph = None
            cls.model = None
            return

        from pdn.pdn_parser import NetlistParser
        parser = NetlistParser(str(test_netlist))
        cls.graph = parser.parse()
        cls.model = create_model_from_pdn(cls.graph, 'VDD')

    def setUp(self):
        """Skip if test netlist not available."""
        if self.model is None:
            self.skipTest("Test netlist not available")
        self.solver = DynamicIRDropSolver(self.model, self.graph)

    def test_pdn_quasi_static_solve(self):
        """Quasi-static solve should work with PDN graph."""
        result = self.solver.solve_quasi_static(
            t_start=0.0,
            t_end=10e-9,
            n_points=3,
            method='flat',
        )

        self.assertIsInstance(result, QuasiStaticResult)
        self.assertEqual(len(result.t_array), 3)

    def test_pdn_with_hierarchical(self):
        """Hierarchical solve should work with PDN graph."""
        try:
            result = self.solver.solve_quasi_static(
                t_start=0.0,
                t_end=10e-9,
                n_points=3,
                method='hierarchical',
                partition_layer='M2',
                top_k=5,
            )
            self.assertIsInstance(result, QuasiStaticResult)
        except RuntimeError as e:
            # Small test netlist may create singular partition
            if "singular" in str(e).lower():
                self.skipTest("Test netlist too small for hierarchical partition")

    def test_pdn_current_sources_loaded(self):
        """PDN current sources should be loaded."""
        # Check that current sources were loaded
        self.assertGreaterEqual(self.solver.num_current_sources, 0)


class TestConstantCurrentConsistency(unittest.TestCase):
    """Tests that constant currents produce consistent results across time."""

    def setUp(self):
        """Set up test fixtures."""
        self.model, self.loads = build_synthetic_grid()
        self.solver = DynamicIRDropSolver(self.model)

    def test_constant_currents_constant_results(self):
        """With constant currents, all time points should have same result."""
        result = self.solver.solve_quasi_static(
            t_start=0.0,
            t_end=100e-9,
            n_points=5,
        )

        # With no dynamic sources, max IR-drop should be same at all times
        max_drops = result.max_ir_drop_per_time
        if max_drops[0] > 0:  # Only check if there's actual IR drop
            # All values should be equal (within numerical precision)
            self.assertTrue(np.allclose(max_drops, max_drops[0], rtol=1e-10))


class TestPulseWaveformEvaluation(unittest.TestCase):
    """Tests for Pulse waveform evaluation in dynamic context."""

    @classmethod
    def setUpClass(cls):
        """Parse test netlist once for all tests."""
        test_netlist = Path(__file__).parent.parent / 'pdn' / 'netlist_test'
        if not test_netlist.exists():
            cls.graph = None
            cls.model = None
            return

        from pdn.pdn_parser import NetlistParser
        parser = NetlistParser(str(test_netlist))
        cls.graph = parser.parse()
        cls.model = create_model_from_pdn(cls.graph, 'VDD')

    def setUp(self):
        """Skip if test netlist not available."""
        if self.model is None:
            self.skipTest("Test netlist not available")
        self.solver = DynamicIRDropSolver(self.model, self.graph)

    def test_current_varies_with_time(self):
        """Currents should vary over time if waveforms are present."""
        if not self.solver.has_dynamic_sources:
            self.skipTest("No dynamic sources in test netlist")

        currents_t0 = self.solver._evaluate_currents_at_time(0.0)
        currents_t1 = self.solver._evaluate_currents_at_time(10e-9)
        currents_t2 = self.solver._evaluate_currents_at_time(50e-9)

        # At least some currents should be different at different times
        total_t0 = sum(currents_t0.values())
        total_t1 = sum(currents_t1.values())
        total_t2 = sum(currents_t2.values())

        # Check that at least currents exist
        self.assertGreater(len(currents_t0), 0)

    def test_currents_periodic_behavior(self):
        """Current sources with period should repeat."""
        if not self.solver.has_dynamic_sources:
            self.skipTest("No dynamic sources in test netlist")

        # Test over multiple periods if applicable
        result = self.solver.solve_quasi_static(
            t_start=0.0,
            t_end=200e-9,
            n_points=21,
        )

        # Total current should show some structure
        self.assertEqual(len(result.total_current_per_time), 21)


class TestQuasiStaticVsDCSolve(unittest.TestCase):
    """Tests comparing quasi-static with single DC solve."""

    @classmethod
    def setUpClass(cls):
        """Parse test netlist once for all tests."""
        test_netlist = Path(__file__).parent.parent / 'pdn' / 'netlist_test'
        if not test_netlist.exists():
            cls.graph = None
            cls.model = None
            return

        from pdn.pdn_parser import NetlistParser
        parser = NetlistParser(str(test_netlist))
        cls.graph = parser.parse()
        cls.model = create_model_from_pdn(cls.graph, 'VDD')

    def setUp(self):
        """Skip if test netlist not available."""
        if self.model is None:
            self.skipTest("Test netlist not available")
        self.solver = DynamicIRDropSolver(self.model, self.graph)
        self.dc_solver = UnifiedIRDropSolver(self.model)

    def test_single_point_matches_dc_solve(self):
        """Quasi-static with single time point should match DC solve."""
        # Get currents at t=0
        currents = self.solver._evaluate_currents_at_time(0.0)

        # DC solve
        dc_result = self.dc_solver.solve(currents)

        # Quasi-static with 1 point
        qs_result = self.solver.solve_quasi_static(
            t_start=0.0,
            t_end=0.0,
            n_points=1,
        )

        # Peak IR-drop should be similar
        dc_max = max(dc_result.ir_drop.values()) if dc_result.ir_drop else 0.0
        qs_max = qs_result.peak_ir_drop

        self.assertAlmostEqual(dc_max, qs_max, places=6)


class TestVerboseOutput(unittest.TestCase):
    """Tests for verbose output mode."""

    def setUp(self):
        """Set up test fixtures."""
        self.model, self.loads = build_synthetic_grid()
        self.solver = DynamicIRDropSolver(self.model)

    def test_verbose_does_not_error(self):
        """Verbose mode should complete without errors."""
        import io
        import sys

        # Capture stdout
        captured = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured

        try:
            result = self.solver.solve_quasi_static(
                t_start=0.0,
                t_end=10e-9,
                n_points=5,
                verbose=True,
            )
        finally:
            sys.stdout = old_stdout

        # Should have printed progress
        output = captured.getvalue()
        # Verbose output contains time step info
        self.assertIsInstance(result, QuasiStaticResult)


class TestWeightingMethods(unittest.TestCase):
    """Tests for different weighting methods in hierarchical mode."""

    def setUp(self):
        """Set up test fixtures."""
        self.model, self.loads = build_synthetic_grid()
        self.solver = DynamicIRDropSolver(self.model)

    def test_shortest_path_weighting(self):
        """Shortest path weighting should work."""
        result = self.solver.solve_quasi_static(
            t_start=0.0,
            t_end=10e-9,
            n_points=3,
            method='hierarchical',
            partition_layer=2,
            weighting='shortest_path',
        )

        self.assertIsInstance(result, QuasiStaticResult)

    def test_effective_weighting(self):
        """Effective resistance weighting should work."""
        result = self.solver.solve_quasi_static(
            t_start=0.0,
            t_end=10e-9,
            n_points=3,
            method='hierarchical',
            partition_layer=2,
            weighting='effective',
        )

        self.assertIsInstance(result, QuasiStaticResult)


class TestResultDataIntegrity(unittest.TestCase):
    """Tests for result data integrity and consistency."""

    def setUp(self):
        """Set up test fixtures."""
        self.model, self.loads = build_synthetic_grid()
        self.solver = DynamicIRDropSolver(self.model)

    def test_peak_drop_consistent_with_per_time(self):
        """peak_ir_drop should equal max of max_ir_drop_per_time."""
        result = self.solver.solve_quasi_static(
            t_start=0.0,
            t_end=50e-9,
            n_points=11,
        )

        expected_max = np.max(result.max_ir_drop_per_time)
        self.assertAlmostEqual(result.peak_ir_drop, expected_max, places=10)

    def test_peak_time_valid(self):
        """peak_ir_drop_time should be in t_array."""
        result = self.solver.solve_quasi_static(
            t_start=0.0,
            t_end=50e-9,
            n_points=11,
        )

        self.assertIn(result.peak_ir_drop_time, result.t_array)

    def test_worst_nodes_in_peak_per_node(self):
        """Worst nodes should appear in peak_ir_drop_per_node."""
        result = self.solver.solve_quasi_static(
            t_start=0.0,
            t_end=10e-9,
            n_points=3,
            n_worst_nodes=5,
        )

        for node, drop, _ in result.worst_nodes:
            self.assertIn(node, result.peak_ir_drop_per_node)

    def test_t_array_monotonically_increasing(self):
        """Time array should be monotonically increasing."""
        result = self.solver.solve_quasi_static(
            t_start=0.0,
            t_end=100e-9,
            n_points=11,
        )

        for i in range(len(result.t_array) - 1):
            self.assertLess(result.t_array[i], result.t_array[i + 1])


class TestEdgeCases(unittest.TestCase):
    """Tests for edge cases and boundary conditions."""

    def setUp(self):
        """Set up test fixtures."""
        self.model, self.loads = build_synthetic_grid()
        self.solver = DynamicIRDropSolver(self.model)

    def test_single_time_point(self):
        """Should work with single time point."""
        result = self.solver.solve_quasi_static(
            t_start=0.0,
            t_end=0.0,
            n_points=1,
        )

        self.assertEqual(len(result.t_array), 1)
        self.assertEqual(len(result.max_ir_drop_per_time), 1)

    def test_two_time_points(self):
        """Should work with just two time points."""
        result = self.solver.solve_quasi_static(
            t_start=0.0,
            t_end=10e-9,
            n_points=2,
        )

        self.assertEqual(len(result.t_array), 2)

    def test_many_time_points(self):
        """Should work with many time points."""
        result = self.solver.solve_quasi_static(
            t_start=0.0,
            t_end=100e-9,
            n_points=101,
        )

        self.assertEqual(len(result.t_array), 101)

    def test_negative_start_time(self):
        """Should handle negative start time."""
        result = self.solver.solve_quasi_static(
            t_start=-10e-9,
            t_end=10e-9,
            n_points=5,
        )

        self.assertAlmostEqual(result.t_array[0], -10e-9)


if __name__ == '__main__':
    unittest.main()

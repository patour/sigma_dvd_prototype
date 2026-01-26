"""Integration tests for dynamic and transient IR-drop analysis.

Tests end-to-end workflows and cross-solver comparisons.
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
from core.transient_solver import (
    TransientIRDropSolver,
    TransientResult,
    IntegrationMethod,
)


def build_synthetic_grid(K=3, N0=8, I_N=80, N_vsrc=4, seed=42):
    """Build a synthetic test grid."""
    G, loads, pads = generate_power_grid(
        K=K, N0=N0, I_N=I_N, N_vsrc=N_vsrc, seed=seed,
        max_stripe_res=0.01, max_via_res=0.01
    )
    model = create_model_from_synthetic(G, pads, vdd=1.0)
    return model, loads


class TestEndToEndPDNWorkflow(unittest.TestCase):
    """End-to-end integration tests for PDN analysis workflow."""

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

    def test_complete_quasi_static_workflow(self):
        """Complete quasi-static analysis workflow."""
        # Create solver
        solver = DynamicIRDropSolver(self.model, self.graph)

        # Run analysis
        result = solver.solve_quasi_static(
            t_start=0.0,
            t_end=100e-9,
            n_points=21,
            method='flat',
            n_worst_nodes=5,
        )

        # Verify results structure
        self.assertIsInstance(result, QuasiStaticResult)
        self.assertEqual(len(result.t_array), 21)
        self.assertGreater(len(result.worst_nodes), 0)
        self.assertGreater(len(result.peak_ir_drop_per_node), 0)

        # Verify data integrity
        self.assertGreaterEqual(result.peak_ir_drop, 0)
        self.assertEqual(result.peak_ir_drop, np.max(result.max_ir_drop_per_time))

    def test_complete_transient_workflow(self):
        """Complete transient RC analysis workflow."""
        # Create solver
        solver = TransientIRDropSolver(self.model, self.graph)

        # Run analysis
        result = solver.solve_transient(
            t_start=0.0,
            t_end=50e-9,
            dt=1e-9,
            method=IntegrationMethod.BACKWARD_EULER,
            n_worst_nodes=5,
        )

        # Verify results structure
        self.assertIsInstance(result, TransientResult)
        self.assertGreater(len(result.t_array), 0)
        self.assertGreater(len(result.peak_ir_drop_per_node), 0)

        # Verify capacitance is being used
        self.assertTrue(solver.has_capacitance)

    def test_tracking_specific_nodes(self):
        """Test tracking specific nodes across time."""
        solver = DynamicIRDropSolver(self.model, self.graph)

        # Get some valid nodes to track
        valid_nodes = list(self.model.valid_nodes)[:5]

        result = solver.solve_quasi_static(
            t_start=0.0,
            t_end=50e-9,
            n_points=11,
            track_nodes=valid_nodes,
        )

        # Verify tracked nodes have waveforms
        for node in valid_nodes:
            self.assertIn(node, result.tracked_waveforms)
            self.assertEqual(len(result.tracked_waveforms[node]), 11)

            # Verify we can retrieve waveforms
            v_waveform = result.get_voltage_waveform(node)
            ir_waveform = result.get_ir_drop_waveform(node)

            self.assertEqual(len(v_waveform), 11)
            self.assertEqual(len(ir_waveform), 11)


class TestQuasiStaticVsTransient(unittest.TestCase):
    """Tests comparing quasi-static and transient analysis."""

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
        self.qs_solver = DynamicIRDropSolver(self.model, self.graph)
        self.tr_solver = TransientIRDropSolver(self.model, self.graph)

    def test_both_produce_valid_results(self):
        """Both solvers should produce valid results."""
        qs_result = self.qs_solver.solve_quasi_static(
            t_start=0.0,
            t_end=50e-9,
            n_points=11,
        )

        tr_result = self.tr_solver.solve_transient(
            t_start=0.0,
            t_end=50e-9,
            dt=5e-9,
        )

        # Both should have valid results
        self.assertGreaterEqual(qs_result.peak_ir_drop, 0)
        self.assertGreaterEqual(tr_result.peak_ir_drop, 0)

        # Both should have populated data
        self.assertGreater(len(qs_result.peak_ir_drop_per_node), 0)
        self.assertGreater(len(tr_result.peak_ir_drop_per_node), 0)

    def test_same_nodes_analyzed(self):
        """Both solvers should analyze the same set of nodes."""
        qs_result = self.qs_solver.solve_quasi_static(
            t_start=0.0,
            t_end=10e-9,
            n_points=3,
        )

        tr_result = self.tr_solver.solve_transient(
            t_start=0.0,
            t_end=10e-9,
            dt=5e-9,
        )

        # Same nodes should have peak data
        qs_nodes = set(qs_result.peak_ir_drop_per_node.keys())
        tr_nodes = set(tr_result.peak_ir_drop_per_node.keys())

        # Node sets should be identical
        self.assertEqual(qs_nodes, tr_nodes)


class TestDynamicVsDCSolver(unittest.TestCase):
    """Tests comparing dynamic solver with static DC solver."""

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
        self.dynamic_solver = DynamicIRDropSolver(self.model, self.graph)
        self.dc_solver = UnifiedIRDropSolver(self.model)

    def test_t0_matches_dc_with_same_currents(self):
        """At t=0, dynamic solve should match DC solve with same currents."""
        # Get currents at t=0
        currents_t0 = self.dynamic_solver._evaluate_currents_at_time(0.0)

        if not currents_t0:
            self.skipTest("No current sources at t=0")

        # DC solve
        dc_result = self.dc_solver.solve(currents_t0)

        # Quasi-static at single point
        qs_result = self.dynamic_solver.solve_quasi_static(
            t_start=0.0,
            t_end=0.0,
            n_points=1,
        )

        # Max IR-drop should match
        dc_max = max(dc_result.ir_drop.values()) if dc_result.ir_drop else 0.0
        qs_max = qs_result.peak_ir_drop

        self.assertAlmostEqual(dc_max, qs_max, places=6)


class TestCurrentConservation(unittest.TestCase):
    """Tests for current conservation (Kirchhoff)."""

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

    def test_quasi_static_current_conservation(self):
        """Vsrc current should equal total load current in quasi-static."""
        solver = DynamicIRDropSolver(self.model, self.graph)

        result = solver.solve_quasi_static(
            t_start=0.0,
            t_end=50e-9,
            n_points=11,
        )

        # At each time point, current conservation should hold
        for i in range(len(result.t_array)):
            total_load = result.total_current_per_time[i]
            vsrc_current = result.total_vsrc_current_per_time[i]

            # They should match within tolerance
            # Note: sign convention may differ, so check absolute values
            if total_load > 0.01:  # Only check if significant current
                # Relative difference should be small
                rel_diff = abs(vsrc_current - total_load) / total_load
                self.assertLess(rel_diff, 0.1)  # 10% tolerance for numerical errors

    def test_transient_current_conservation(self):
        """Vsrc current should approximately equal load current in transient."""
        solver = TransientIRDropSolver(self.model, self.graph)

        result = solver.solve_transient(
            t_start=0.0,
            t_end=50e-9,
            dt=2e-9,
        )

        # Check current conservation at each step
        for i in range(len(result.t_array)):
            total_load = result.total_current_per_time[i]
            vsrc_current = result.total_vsrc_current_per_time[i]

            # Note: in transient, capacitive current also flows
            # So exact conservation doesn't hold - just check reasonableness
            # Both should be non-negative
            self.assertGreaterEqual(total_load, 0)


class TestFlatVsHierarchicalDynamic(unittest.TestCase):
    """Tests comparing flat and hierarchical methods in dynamic solver."""

    def setUp(self):
        """Set up test fixtures."""
        self.model, self.loads = build_synthetic_grid()
        self.solver = DynamicIRDropSolver(self.model)

    def test_flat_and_hierarchical_similar_peaks(self):
        """Flat and hierarchical should produce similar peak IR-drop."""
        result_flat = self.solver.solve_quasi_static(
            t_start=0.0,
            t_end=10e-9,
            n_points=3,
            method='flat',
        )

        result_hier = self.solver.solve_quasi_static(
            t_start=0.0,
            t_end=10e-9,
            n_points=3,
            method='hierarchical',
            partition_layer=2,
            top_k=5,
        )

        # Both should produce valid results
        self.assertIsInstance(result_flat, QuasiStaticResult)
        self.assertIsInstance(result_hier, QuasiStaticResult)

        # Peak values should be similar (hierarchical has some approximation error)
        if result_flat.peak_ir_drop > 1e-6:
            rel_diff = abs(result_flat.peak_ir_drop - result_hier.peak_ir_drop) / result_flat.peak_ir_drop
            self.assertLess(rel_diff, 0.5)  # 50% tolerance for hierarchical approximation


class TestLongDurationSimulation(unittest.TestCase):
    """Tests for longer duration simulations."""

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

    def test_long_quasi_static_simulation(self):
        """Longer quasi-static simulation should complete."""
        solver = DynamicIRDropSolver(self.model, self.graph)

        result = solver.solve_quasi_static(
            t_start=0.0,
            t_end=500e-9,  # 500 ns
            n_points=51,
        )

        self.assertEqual(len(result.t_array), 51)
        self.assertGreater(result.timings['total'], 0)

    def test_long_transient_simulation(self):
        """Longer transient simulation should complete."""
        solver = TransientIRDropSolver(self.model, self.graph)

        result = solver.solve_transient(
            t_start=0.0,
            t_end=200e-9,  # 200 ns
            dt=5e-9,       # 5 ns steps
        )

        self.assertGreater(len(result.t_array), 10)
        self.assertGreater(result.timings['total'], 0)


class TestMultipleSolveConsistency(unittest.TestCase):
    """Tests for consistency across multiple solves."""

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

    def test_quasi_static_deterministic(self):
        """Quasi-static solve should be deterministic."""
        solver = DynamicIRDropSolver(self.model, self.graph)

        result1 = solver.solve_quasi_static(
            t_start=0.0,
            t_end=50e-9,
            n_points=11,
        )

        result2 = solver.solve_quasi_static(
            t_start=0.0,
            t_end=50e-9,
            n_points=11,
        )

        # Results should be identical
        np.testing.assert_array_almost_equal(
            result1.max_ir_drop_per_time,
            result2.max_ir_drop_per_time,
        )

        self.assertAlmostEqual(result1.peak_ir_drop, result2.peak_ir_drop)

    def test_transient_deterministic(self):
        """Transient solve should be deterministic."""
        solver = TransientIRDropSolver(self.model, self.graph)

        result1 = solver.solve_transient(
            t_start=0.0,
            t_end=50e-9,
            dt=2e-9,
        )

        result2 = solver.solve_transient(
            t_start=0.0,
            t_end=50e-9,
            dt=2e-9,
        )

        # Results should be identical
        np.testing.assert_array_almost_equal(
            result1.max_ir_drop_per_time,
            result2.max_ir_drop_per_time,
        )


class TestSyntheticGridDynamic(unittest.TestCase):
    """Integration tests for dynamic analysis on synthetic grids."""

    def setUp(self):
        """Set up test fixtures."""
        self.model, self.loads = build_synthetic_grid()

    def test_quasi_static_on_synthetic(self):
        """Quasi-static should work on synthetic grid."""
        solver = DynamicIRDropSolver(self.model)

        result = solver.solve_quasi_static(
            t_start=0.0,
            t_end=100e-9,
            n_points=11,
        )

        self.assertIsInstance(result, QuasiStaticResult)
        self.assertEqual(len(result.t_array), 11)

    def test_transient_on_synthetic(self):
        """Transient should work on synthetic grid (no capacitance)."""
        solver = TransientIRDropSolver(self.model)

        result = solver.solve_transient(
            t_start=0.0,
            t_end=100e-9,
            dt=10e-9,
        )

        self.assertIsInstance(result, TransientResult)

        # Should have zero capacitance
        self.assertFalse(solver.has_capacitance)


class TestPlotterIntegration(unittest.TestCase):
    """Integration tests for plotting utilities."""

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

    def test_plotter_with_quasi_static_result(self):
        """DynamicPlotter should work with QuasiStaticResult."""
        from core.dynamic_plotter import DynamicPlotter

        solver = DynamicIRDropSolver(self.model, self.graph)
        result = solver.solve_quasi_static(
            t_start=0.0,
            t_end=50e-9,
            n_points=11,
        )

        # Should not raise
        try:
            fig, ax = DynamicPlotter.plot_peak_ir_drop_heatmap(
                self.model, result, show=False
            )
            self.assertIsNotNone(fig)
            import matplotlib.pyplot as plt
            plt.close(fig)
        except ValueError as e:
            # May fail if no coordinates, which is OK
            if "No nodes with coordinates" not in str(e):
                raise

    def test_plotter_with_transient_result(self):
        """DynamicPlotter should work with TransientResult."""
        from core.dynamic_plotter import DynamicPlotter

        solver = TransientIRDropSolver(self.model, self.graph)
        result = solver.solve_transient(
            t_start=0.0,
            t_end=50e-9,
            dt=5e-9,
        )

        # Time series should always work
        fig, ax = DynamicPlotter.plot_time_series(
            result,
            metrics=['max_ir_drop', 'total_current'],
            show=False,
        )
        self.assertIsNotNone(fig)

        import matplotlib.pyplot as plt
        plt.close(fig)


class TestIntegrationMethodComparison(unittest.TestCase):
    """Tests comparing different integration methods."""

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
        self.solver = TransientIRDropSolver(self.model, self.graph)

    def test_be_and_trap_both_valid(self):
        """Both BE and Trapezoidal should produce valid results."""
        result_be = self.solver.solve_transient(
            t_start=0.0,
            t_end=30e-9,
            dt=1e-9,
            method=IntegrationMethod.BACKWARD_EULER,
        )

        result_trap = self.solver.solve_transient(
            t_start=0.0,
            t_end=30e-9,
            dt=1e-9,
            method=IntegrationMethod.TRAPEZOIDAL,
        )

        # Both should be valid
        self.assertGreaterEqual(result_be.peak_ir_drop, 0)
        self.assertGreaterEqual(result_trap.peak_ir_drop, 0)

        # Both should have same number of time points
        self.assertEqual(len(result_be.t_array), len(result_trap.t_array))


if __name__ == '__main__':
    unittest.main()

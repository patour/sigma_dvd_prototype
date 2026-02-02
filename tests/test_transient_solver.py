"""Tests for transient IR-drop solver with RC support.

Tests time-domain simulation with capacitance for accurate decoupling effects.
"""

import unittest
from pathlib import Path
from typing import Dict, Any

import numpy as np

from generate_power_grid import generate_power_grid
from core import (
    create_model_from_synthetic,
    create_model_from_pdn,
)
from core.transient_solver import (
    TransientIRDropSolver,
    TransientResult,
    IntegrationMethod,
    RCSystem,
)
from core.dynamic_solver import DynamicIRDropSolver


def build_synthetic_grid(K=3, N0=8, I_N=80, N_vsrc=4, seed=42):
    """Build a synthetic test grid."""
    G, loads, pads = generate_power_grid(
        K=K, N0=N0, I_N=I_N, N_vsrc=N_vsrc, seed=seed,
        max_stripe_res=0.01, max_via_res=0.01
    )
    model = create_model_from_synthetic(G, pads, vdd=1.0)
    return model, loads


class TestTransientSolverInit(unittest.TestCase):
    """Tests for TransientIRDropSolver initialization."""

    def test_init_with_synthetic_grid(self):
        """Solver should initialize with synthetic grid."""
        model, _ = build_synthetic_grid()
        solver = TransientIRDropSolver(model)

        self.assertIsNotNone(solver.model)
        # Synthetic grids don't have capacitors
        self.assertFalse(solver.has_capacitance)

    def test_init_with_pdn_graph(self):
        """Solver should initialize with PDN graph if available."""
        test_netlist = Path(__file__).parent.parent / 'pdn' / 'netlist_test'
        if not test_netlist.exists():
            self.skipTest("Test netlist not available")

        from pdn.pdn_parser import NetlistParser
        parser = NetlistParser(str(test_netlist))
        graph = parser.parse()
        model = create_model_from_pdn(graph, 'VDD')

        solver = TransientIRDropSolver(model, graph)

        self.assertIsNotNone(solver.model)
        # PDN graph should have capacitors
        self.assertTrue(solver.has_capacitance)


class TestRCSystemBuilding(unittest.TestCase):
    """Tests for RC system matrix building."""

    def setUp(self):
        """Set up test fixtures."""
        self.model, self.loads = build_synthetic_grid()
        self.solver = TransientIRDropSolver(self.model)

    def test_rc_system_built(self):
        """RC system should be built correctly."""
        rc = self.solver._ensure_rc_system()

        self.assertIsInstance(rc, RCSystem)
        self.assertGreater(rc.n_nodes, 0)
        self.assertGreater(rc.n_unknown, 0)
        self.assertEqual(len(rc.node_order), rc.n_nodes)

    def test_g_matrix_symmetric(self):
        """Reduced conductance matrix (G_uu) should be symmetric."""
        rc = self.solver._ensure_rc_system()
        G_dense = rc.G_uu.toarray()
        np.testing.assert_array_almost_equal(G_dense, G_dense.T)

    def test_g_matrix_positive_diagonal(self):
        """Reduced conductance matrix diagonal should be positive."""
        rc = self.solver._ensure_rc_system()
        diag = rc.G_uu.diagonal()
        self.assertTrue(np.all(diag > 0))

    def test_c_matrix_symmetric(self):
        """Reduced capacitance matrix (C_uu) should be symmetric."""
        rc = self.solver._ensure_rc_system()
        C_dense = rc.C_uu.toarray()
        np.testing.assert_array_almost_equal(C_dense, C_dense.T)

    def test_c_matrix_nonnegative_diagonal(self):
        """Reduced capacitance matrix diagonal should be non-negative."""
        rc = self.solver._ensure_rc_system()
        diag = rc.C_uu.diagonal()
        self.assertTrue(np.all(diag >= 0))


class TestTransientSolve(unittest.TestCase):
    """Tests for transient solving."""

    def setUp(self):
        """Set up test fixtures."""
        self.model, self.loads = build_synthetic_grid()
        self.solver = TransientIRDropSolver(self.model)

    def test_solve_transient_returns_result(self):
        """solve_transient should return TransientResult."""
        result = self.solver.solve_transient(
            t_start=0.0,
            t_end=10e-9,
            dt=1e-9,
        )

        self.assertIsInstance(result, TransientResult)
        self.assertIsNotNone(result.peak_ir_drop)
        self.assertEqual(result.nominal_voltage, self.model.vdd)

    def test_solve_with_backward_euler(self):
        """Backward Euler method should work."""
        result = self.solver.solve_transient(
            t_start=0.0,
            t_end=10e-9,
            dt=1e-9,
            method=IntegrationMethod.BACKWARD_EULER,
        )

        self.assertEqual(result.integration_method, IntegrationMethod.BACKWARD_EULER)
        self.assertTrue('solve' in result.timings)

    def test_solve_with_trapezoidal(self):
        """Trapezoidal method should work."""
        result = self.solver.solve_transient(
            t_start=0.0,
            t_end=10e-9,
            dt=1e-9,
            method=IntegrationMethod.TRAPEZOIDAL,
        )

        self.assertEqual(result.integration_method, IntegrationMethod.TRAPEZOIDAL)

    def test_time_array_correct(self):
        """Time array should have correct range and step."""
        result = self.solver.solve_transient(
            t_start=0.0,
            t_end=10e-9,
            dt=2e-9,
        )

        self.assertAlmostEqual(result.t_array[0], 0.0)
        self.assertAlmostEqual(result.t_array[-1], 10e-9)
        # Check step size
        dt_actual = result.t_array[1] - result.t_array[0]
        self.assertAlmostEqual(dt_actual, 2e-9)


class TestTransientWithPDN(unittest.TestCase):
    """Tests for transient solving with PDN netlist (with capacitors)."""

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

    def test_pdn_has_capacitance(self):
        """PDN model should have capacitance."""
        self.assertTrue(self.solver.has_capacitance)
        self.assertGreater(self.solver.total_capacitance, 0)

    def test_pdn_transient_solve(self):
        """Transient solve should work with PDN graph."""
        result = self.solver.solve_transient(
            t_start=0.0,
            t_end=10e-9,
            dt=1e-9,
        )

        self.assertIsInstance(result, TransientResult)
        self.assertGreater(len(result.t_array), 0)

    def test_capacitance_matrix_built(self):
        """Capacitance matrix should be non-empty for PDN."""
        rc = self.solver._ensure_rc_system()
        self.assertTrue(rc.has_capacitance)
        self.assertGreater(rc.C_uu.nnz, 0)


class TestTransientVsQuasiStatic(unittest.TestCase):
    """Tests comparing transient and quasi-static results."""

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
        self.transient_solver = TransientIRDropSolver(self.model, self.graph)
        self.quasi_static_solver = DynamicIRDropSolver(self.model, self.graph)

    def test_transient_produces_valid_results(self):
        """Transient solve should produce valid results."""
        result = self.transient_solver.solve_transient(
            t_start=0.0,
            t_end=50e-9,
            dt=1e-9,
            method=IntegrationMethod.BACKWARD_EULER,
        )

        # Results should be valid
        self.assertGreater(len(result.t_array), 0)
        self.assertTrue(np.all(result.max_ir_drop_per_time >= 0))

        # Peak IR-drop should be within the max range
        self.assertEqual(result.peak_ir_drop, np.max(result.max_ir_drop_per_time))


class TestPeakTracking(unittest.TestCase):
    """Tests for peak IR-drop and worst node tracking."""

    def setUp(self):
        """Set up test fixtures."""
        self.model, self.loads = build_synthetic_grid()
        self.solver = TransientIRDropSolver(self.model)

    def test_peak_ir_drop_tracked(self):
        """Peak IR-drop should be tracked."""
        result = self.solver.solve_transient(
            t_start=0.0,
            t_end=10e-9,
            dt=1e-9,
        )

        self.assertGreaterEqual(result.peak_ir_drop, 0.0)

    def test_worst_nodes_tracked(self):
        """Worst nodes should be tracked."""
        n_worst = 5
        result = self.solver.solve_transient(
            t_start=0.0,
            t_end=10e-9,
            dt=1e-9,
            n_worst_nodes=n_worst,
        )

        # May have fewer worst nodes than requested if grid is small
        self.assertLessEqual(len(result.worst_nodes), n_worst)

        # Worst nodes should be sorted by drop (descending)
        if result.worst_nodes:
            drops = [drop for _, drop, _ in result.worst_nodes]
            self.assertEqual(drops, sorted(drops, reverse=True))

    def test_peak_ir_drop_per_node_populated(self):
        """peak_ir_drop_per_node should be populated."""
        result = self.solver.solve_transient(
            t_start=0.0,
            t_end=10e-9,
            dt=1e-9,
        )

        self.assertGreater(len(result.peak_ir_drop_per_node), 0)


class TestWaveformTracking(unittest.TestCase):
    """Tests for node waveform tracking."""

    def setUp(self):
        """Set up test fixtures."""
        self.model, self.loads = build_synthetic_grid()
        self.solver = TransientIRDropSolver(self.model)

    def test_track_specific_nodes(self):
        """Waveforms should be stored for tracked nodes."""
        load_nodes = list(self.loads.keys())[:3]

        result = self.solver.solve_transient(
            t_start=0.0,
            t_end=10e-9,
            dt=2e-9,
            track_nodes=load_nodes,
        )

        for node in load_nodes:
            self.assertIn(node, result.tracked_waveforms)
            self.assertIn(node, result.tracked_ir_drop)

    def test_get_voltage_waveform(self):
        """get_voltage_waveform should return waveform for tracked node."""
        load_nodes = list(self.loads.keys())[:1]

        result = self.solver.solve_transient(
            t_start=0.0,
            t_end=10e-9,
            dt=2e-9,
            track_nodes=load_nodes,
        )

        waveform = result.get_voltage_waveform(load_nodes[0])
        self.assertEqual(len(waveform), len(result.t_array))
        # Voltage should be <= Vdd (with small tolerance for numerical errors)
        self.assertTrue(np.all(waveform <= self.model.vdd + 1e-10))

    def test_untracked_node_raises_error(self):
        """Accessing waveform for untracked node should raise KeyError."""
        result = self.solver.solve_transient(
            t_start=0.0,
            t_end=10e-9,
            dt=2e-9,
            track_nodes=[],
        )

        with self.assertRaises(KeyError):
            result.get_voltage_waveform('nonexistent_node')


class TestVsrcCurrentTracking(unittest.TestCase):
    """Tests for voltage source current tracking."""

    def setUp(self):
        """Set up test fixtures."""
        self.model, self.loads = build_synthetic_grid()
        self.solver = TransientIRDropSolver(self.model)

    def test_vsrc_current_tracked(self):
        """total_vsrc_current_per_time should be populated."""
        result = self.solver.solve_transient(
            t_start=0.0,
            t_end=10e-9,
            dt=2e-9,
        )

        self.assertEqual(len(result.total_vsrc_current_per_time), len(result.t_array))


class TestTimings(unittest.TestCase):
    """Tests for timing information."""

    def setUp(self):
        """Set up test fixtures."""
        self.model, self.loads = build_synthetic_grid()
        self.solver = TransientIRDropSolver(self.model)

    def test_timings_populated(self):
        """Timing breakdown should be populated."""
        result = self.solver.solve_transient(
            t_start=0.0,
            t_end=10e-9,
            dt=2e-9,
        )

        self.assertIn('build_rc', result.timings)
        self.assertIn('factor', result.timings)
        self.assertIn('solve', result.timings)
        self.assertIn('total', result.timings)

        # All times should be non-negative
        for t in result.timings.values():
            self.assertGreaterEqual(t, 0.0)


class TestIntegrationMethodEnum(unittest.TestCase):
    """Tests for IntegrationMethod enum."""

    def test_enum_values(self):
        """Enum should have expected values."""
        self.assertEqual(IntegrationMethod.BACKWARD_EULER.value, 'be')
        self.assertEqual(IntegrationMethod.TRAPEZOIDAL.value, 'trap')


class TestRCSystemMatrixProperties(unittest.TestCase):
    """Tests for RC system matrix properties."""

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

    def test_reduced_matrices_built(self):
        """Reduced matrices (G_uu, C_uu) should be built."""
        rc = self.solver._ensure_rc_system()

        self.assertIsNotNone(rc.G_uu)
        self.assertIsNotNone(rc.C_uu)
        self.assertGreater(rc.n_unknown, 0)

    def test_g_uu_positive_definite(self):
        """Reduced G matrix should be positive semi-definite."""
        rc = self.solver._ensure_rc_system()
        G_uu_dense = rc.G_uu.toarray()

        # Check symmetry
        np.testing.assert_array_almost_equal(G_uu_dense, G_uu_dense.T)

        # Check eigenvalues are non-negative
        eigenvalues = np.linalg.eigvalsh(G_uu_dense)
        self.assertTrue(np.all(eigenvalues >= -1e-10))

    def test_c_uu_positive_semidefinite(self):
        """Reduced C matrix should be positive semi-definite."""
        rc = self.solver._ensure_rc_system()
        C_uu_dense = rc.C_uu.toarray()

        # Check symmetry
        np.testing.assert_array_almost_equal(C_uu_dense, C_uu_dense.T)

        # Check eigenvalues are non-negative
        eigenvalues = np.linalg.eigvalsh(C_uu_dense)
        self.assertTrue(np.all(eigenvalues >= -1e-10))

    def test_node_ordering_consistent(self):
        """Node ordering should be consistent across structures."""
        rc = self.solver._ensure_rc_system()

        # All unknown nodes should be in node_order
        for node in rc.unknown_nodes:
            self.assertIn(node, rc.node_order)
            self.assertIn(node, rc.node_to_idx)

        # All pad nodes should be in node_order
        for node in rc.pad_nodes:
            self.assertIn(node, rc.node_order)


class TestBackwardEulerStability(unittest.TestCase):
    """Tests for Backward Euler stability properties."""

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

    def test_stable_with_large_dt(self):
        """Backward Euler should be stable even with large time steps."""
        # Large dt should still produce bounded results
        result = self.solver.solve_transient(
            t_start=0.0,
            t_end=100e-9,
            dt=10e-9,  # Large time step
            method=IntegrationMethod.BACKWARD_EULER,
        )

        # Voltages should remain bounded (not blow up)
        self.assertTrue(np.all(result.max_ir_drop_per_time < self.model.vdd))
        self.assertTrue(np.all(result.max_ir_drop_per_time >= 0))

    def test_voltage_bounded_by_vdd(self):
        """Voltages should never exceed Vdd."""
        result = self.solver.solve_transient(
            t_start=0.0,
            t_end=50e-9,
            dt=1e-9,
        )

        # IR-drop is Vdd - V, so should be non-negative (V <= Vdd)
        for drop in result.peak_ir_drop_per_node.values():
            self.assertGreaterEqual(drop, -1e-10)


class TestTrapezoidalMethod(unittest.TestCase):
    """Tests for Trapezoidal integration method."""

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

    def test_trapezoidal_produces_valid_results(self):
        """Trapezoidal method should produce valid results with appropriate step size.
        
        Note: Trapezoidal can exhibit oscillations with large step sizes on stiff
        RC systems. Using 100ps step size avoids numerical oscillations.
        """
        result = self.solver.solve_transient(
            t_start=0.0,
            t_end=50e-9,
            dt=100e-12,
            method=IntegrationMethod.TRAPEZOIDAL,
        )

        self.assertTrue(np.all(result.max_ir_drop_per_time >= 0))
        self.assertGreater(len(result.t_array), 0)
        # Verify expected IR-drop range for this netlist
        self.assertAlmostEqual(result.max_ir_drop_per_time.min() * 1000, 0.4063, places=2)
        self.assertAlmostEqual(result.max_ir_drop_per_time.max() * 1000, 0.7743, places=2)

    def test_be_vs_trap_both_complete(self):
        """Both BE and Trapezoidal should complete without errors."""
        # Note: Trapezoidal can exhibit oscillations on stiff RC systems,
        # especially with rapid current transients. This is expected behavior
        # for the Trapezoidal rule (it preserves oscillations while BE damps them).
        # We only verify both methods complete and produce valid results.

        result_be = self.solver.solve_transient(
            t_start=0.0,
            t_end=20e-9,
            dt=0.5e-9,
            method=IntegrationMethod.BACKWARD_EULER,
        )

        result_trap = self.solver.solve_transient(
            t_start=0.0,
            t_end=20e-9,
            dt=0.5e-9,
            method=IntegrationMethod.TRAPEZOIDAL,
        )

        # Both should complete and have same array sizes
        self.assertEqual(len(result_be.t_array), len(result_trap.t_array))

        # Both should produce non-negative IR-drop values
        self.assertTrue(np.all(result_be.max_ir_drop_per_time >= -1e-10))
        # Trapezoidal may have oscillations but IR-drop should be bounded
        self.assertTrue(np.all(result_trap.max_ir_drop_per_time < 2.0))  # Bounded by 2*Vdd


class TestCapacitiveSmoothing(unittest.TestCase):
    """Tests for capacitive smoothing effects."""

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
        self.transient_solver = TransientIRDropSolver(self.model, self.graph)
        self.quasi_static_solver = DynamicIRDropSolver(self.model, self.graph)

    def test_capacitance_present(self):
        """Test netlist should have capacitance."""
        self.assertTrue(self.transient_solver.has_capacitance)
        total_cap = self.transient_solver.total_capacitance
        self.assertGreater(total_cap, 0)


class TestTransientResultIntegrity(unittest.TestCase):
    """Tests for transient result data integrity."""

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

    def test_peak_consistent_with_per_time(self):
        """peak_ir_drop should equal max of max_ir_drop_per_time."""
        result = self.solver.solve_transient(
            t_start=0.0,
            t_end=30e-9,
            dt=1e-9,
        )

        expected_max = np.max(result.max_ir_drop_per_time)
        self.assertAlmostEqual(result.peak_ir_drop, expected_max, places=10)

    def test_arrays_same_length(self):
        """All time-indexed arrays should have same length."""
        result = self.solver.solve_transient(
            t_start=0.0,
            t_end=30e-9,
            dt=1e-9,
        )

        n_steps = len(result.t_array)
        self.assertEqual(len(result.max_ir_drop_per_time), n_steps)
        self.assertEqual(len(result.total_current_per_time), n_steps)
        self.assertEqual(len(result.total_vsrc_current_per_time), n_steps)


class TestTransientEdgeCases(unittest.TestCase):
    """Tests for transient edge cases."""

    def setUp(self):
        """Set up test fixtures."""
        self.model, self.loads = build_synthetic_grid()
        self.solver = TransientIRDropSolver(self.model)

    def test_single_time_step(self):
        """Should handle single time step."""
        result = self.solver.solve_transient(
            t_start=0.0,
            t_end=1e-9,
            dt=1e-9,
        )

        self.assertGreaterEqual(len(result.t_array), 1)

    def test_very_small_dt(self):
        """Should handle very small time steps."""
        result = self.solver.solve_transient(
            t_start=0.0,
            t_end=1e-9,
            dt=0.1e-9,
        )

        self.assertGreater(len(result.t_array), 5)

    def test_verbose_mode(self):
        """Verbose mode should complete without errors."""
        import io
        import sys

        captured = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured

        try:
            result = self.solver.solve_transient(
                t_start=0.0,
                t_end=10e-9,
                dt=2e-9,
                verbose=True,
            )
        finally:
            sys.stdout = old_stdout

        self.assertIsInstance(result, TransientResult)


class TestSyntheticGridTransient(unittest.TestCase):
    """Tests for transient solving on synthetic grids (no capacitance)."""

    def setUp(self):
        """Set up test fixtures."""
        self.model, self.loads = build_synthetic_grid()
        self.solver = TransientIRDropSolver(self.model)

    def test_no_capacitance_in_synthetic(self):
        """Synthetic grid should have no capacitance."""
        self.assertFalse(self.solver.has_capacitance)
        self.assertEqual(self.solver.total_capacitance, 0.0)

    def test_transient_without_capacitance(self):
        """Transient solve should work even without capacitance."""
        result = self.solver.solve_transient(
            t_start=0.0,
            t_end=10e-9,
            dt=1e-9,
        )

        self.assertIsInstance(result, TransientResult)

    def test_rc_system_zero_capacitance(self):
        """RC system C matrix should be zero for synthetic grid."""
        rc = self.solver._ensure_rc_system()

        # C matrix should be all zeros
        self.assertFalse(rc.has_capacitance)
        self.assertEqual(rc.C_uu.nnz, 0)


class TestCapacitanceEffects(unittest.TestCase):
    """Tests verifying capacitance effects are correctly modeled.
    
    These tests validate that:
    1. Capacitance values are correctly reported in fF units
    2. The transient solver correctly handles RC time constants
    3. Capacitive smoothing reduces IR-drop compared to quasi-static
    """

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
        self.transient_solver = TransientIRDropSolver(self.model, self.graph)
        self.quasi_static_solver = DynamicIRDropSolver(self.model, self.graph)

    def test_total_capacitance_value(self):
        """Total capacitance should be reported in fF with correct value.
        
        netlist_test has 50 capacitors of 1000 fF each = 50000 fF total.
        After nodal stamping (divide by 2), total_capacitance = 25000 fF.
        """
        total_cap = self.transient_solver.total_capacitance
        
        # Value should be in fF (not F or other units)
        # Expected: ~25000 fF (50 caps * 1000 fF / 2 for nodal stamping)
        self.assertGreater(total_cap, 1000)  # Should be thousands of fF, not tiny values
        self.assertLess(total_cap, 1e9)  # Should not be in impossibly large range
        
        # More specific: should be around 25000 fF (allowing for numerical precision)
        self.assertAlmostEqual(total_cap, 25000, delta=1000)

    def test_capacitance_matrix_units_consistency(self):
        """Verify G and C matrices have consistent units (mS and fF).
        
        For PDN netlists:
        - G is in mS (from R in kOhm): G = 1/R_kOhm
        - C is in fF (native PDN units)
        - Time constant tau = R*C = kOhm * fF = ps
        """
        rc = self.transient_solver._ensure_rc_system()
        
        # G diagonal should be reasonable mS values (not tiny or huge)
        G_diag_mean = rc.G_uu.diagonal().mean()
        self.assertGreater(G_diag_mean, 1)  # At least 1 mS
        self.assertLess(G_diag_mean, 1e10)  # Not unreasonably large
        
        # C diagonal should be reasonable fF values
        C_diag_mean = rc.C_uu.diagonal().mean()
        self.assertGreater(C_diag_mean, 1)  # At least 1 fF per node
        self.assertLess(C_diag_mean, 1e9)  # Not unreasonably large

    def test_transient_converges_to_quasi_static_at_large_dt(self):
        """At large timesteps (dt >> tau), transient should match quasi-static.
        
        When the timestep is much larger than the RC time constant, 
        capacitors respond essentially instantaneously, so transient
        analysis should give the same steady-state result as quasi-static.
        """
        # Run quasi-static analysis
        qs_result = self.quasi_static_solver.solve_quasi_static(
            t_start=0, t_end=10e-9, n_points=11,
            method='flat', verbose=False
        )
        
        # Run transient with large timestep (1 ns >> tau which is ~ps)
        trans_result = self.transient_solver.solve_transient(
            t_start=0, t_end=10e-9, dt=1e-9,
            method=IntegrationMethod.BACKWARD_EULER,
            verbose=False
        )
        
        # After initial transient, results should converge
        # Compare IR-drop at later time points (skip t=0 due to initial condition)
        qs_drops = qs_result.max_ir_drop_per_time[2:]  # Skip first 2 points
        trans_drops = trans_result.max_ir_drop_per_time[2:]
        
        # Should be within 1% of each other at steady state
        for qs_drop, trans_drop in zip(qs_drops, trans_drops):
            if qs_drop > 0:
                rel_diff = abs(qs_drop - trans_drop) / qs_drop
                self.assertLess(rel_diff, 0.01, 
                    f"Transient should converge to quasi-static: QS={qs_drop:.6f}, Trans={trans_drop:.6f}")

    def test_capacitive_smoothing_at_small_dt(self):
        """At small timesteps (dt ~ tau), capacitors should provide smoothing.
        
        When the timestep is comparable to or smaller than the RC time constant,
        capacitors resist rapid voltage changes, resulting in lower peak IR-drop
        compared to instantaneous (quasi-static) response.
        """
        # For this test, we need a very small timestep to see the effect
        # since tau is in the ps range for this netlist
        
        # Run transient with decreasing timesteps
        results = {}
        for dt in [1e-9, 1e-10, 1e-11, 1e-12]:
            result = self.transient_solver.solve_transient(
                t_start=0, t_end=5*dt, dt=dt,
                method=IntegrationMethod.BACKWARD_EULER,
                verbose=False
            )
            # Get peak excluding t=0 (which has zero drop due to initial condition)
            peak = np.max(result.max_ir_drop_per_time[1:]) if len(result.max_ir_drop_per_time) > 1 else 0
            results[dt] = peak
        
        # With smaller timesteps, we should see more smoothing (lower peak IR-drop)
        # Compare 1ns vs 1ps results
        peak_1ns = results[1e-9]
        peak_1ps = results[1e-12]
        
        # At 1ps timestep, capacitive smoothing should reduce the peak
        # (by at least 10% compared to 1ns timestep)
        if peak_1ns > 0:
            smoothing_percent = (peak_1ns - peak_1ps) / peak_1ns * 100
            self.assertGreater(smoothing_percent, 10,
                f"Capacitive smoothing should reduce peak IR-drop: 1ns={peak_1ns*1000:.4f}mV, 1ps={peak_1ps*1000:.4f}mV")

    def test_c_matrix_in_native_ff_units(self):
        """C matrix should store values in native fF units (not converted to F).

        This ensures the RC time constant calculation is correct:
        tau = R(kOhm) * C(fF) = ps
        """
        rc = self.transient_solver._ensure_rc_system()

        # The netlist has 50 caps of 1000 fF each = 50000 fF total
        # total_capacitance_fF stores total capacitance (diagonal sum / 2)
        total_cap = rc.total_capacitance_fF

        # If C were in Farads, the total would be ~5e-11
        # If C is in fF, the total should be ~50000
        self.assertGreater(total_cap, 1000,
            "C matrix should be in fF units, not Farads")
        self.assertLess(total_cap, 1e6,
            "C matrix values should be reasonable fF values")

    def test_backward_euler_vs_trapezoidal_consistency(self):
        """Both integration methods should give similar results.

        While Backward Euler and Trapezoidal have different numerical properties,
        they should converge to similar results for well-resolved simulations.

        Note: For this netlist, the RC time constant tau ~ 0.6 ps. With dt >> tau,
        trapezoidal can exhibit overshoot (expected behavior for stiff systems).
        Using dt=5ps gives <1% difference; dt=10ps gives ~18% difference.
        """
        dt = 5e-12  # 5 ps - provides well-resolved behavior (dt ~ 8*tau)
        t_end = 1e-9  # 1 ns total simulation

        be_result = self.transient_solver.solve_transient(
            t_start=0, t_end=t_end, dt=dt,
            method=IntegrationMethod.BACKWARD_EULER,
            verbose=False
        )

        trap_result = self.transient_solver.solve_transient(
            t_start=0, t_end=t_end, dt=dt,
            method=IntegrationMethod.TRAPEZOIDAL,
            verbose=False
        )

        # With dt=5ps, both methods should give nearly identical results (<5% diff)
        be_peak = be_result.peak_ir_drop
        trap_peak = trap_result.peak_ir_drop

        if be_peak > 0:
            rel_diff = abs(be_peak - trap_peak) / be_peak
            self.assertLess(rel_diff, 0.05,
                f"BE and Trap should give similar results: BE={be_peak*1000:.4f}mV, Trap={trap_peak*1000:.4f}mV")


class TestMultiRHSSolver(unittest.TestCase):
    """Tests for solve_transient_multi_rhs with multiple current source masks."""

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
        # Use vectorize_threshold=0 to force vectorized mode
        self.solver = TransientIRDropSolver(self.model, self.graph, vectorize_threshold=0)

    def test_multi_rhs_requires_vectorized_sources(self):
        """solve_transient_multi_rhs should require vectorized current sources."""
        # Create solver without vectorized sources
        solver_no_vec = TransientIRDropSolver(self.model, self.graph, vectorize_threshold=-1)

        n_sources = 17  # Known size for test netlist
        masks = np.ones((1, n_sources), dtype=bool)

        with self.assertRaises(RuntimeError):
            solver_no_vec.solve_transient_multi_rhs(
                t_start=0, t_end=1e-9, dt=1e-9,
                source_masks=masks
            )

    def test_multi_rhs_validates_mask_shape(self):
        """solve_transient_multi_rhs should validate mask shape."""
        # Wrong number of sources in mask
        wrong_masks = np.ones((2, 5), dtype=bool)  # Wrong n_sources

        with self.assertRaises(ValueError):
            self.solver.solve_transient_multi_rhs(
                t_start=0, t_end=1e-9, dt=1e-9,
                source_masks=wrong_masks
            )

    def test_multi_rhs_returns_list_of_results(self):
        """solve_transient_multi_rhs should return one result per mask."""
        n_sources = self.solver._vec_sources.n_sources
        masks = np.stack([
            np.ones(n_sources, dtype=bool),
            np.zeros(n_sources, dtype=bool),
            np.array([i % 2 == 0 for i in range(n_sources)], dtype=bool),
        ])

        results = self.solver.solve_transient_multi_rhs(
            t_start=0, t_end=5e-9, dt=1e-9,
            source_masks=masks
        )

        self.assertEqual(len(results), 3)
        for result in results:
            self.assertIsInstance(result, TransientResult)

    def test_multi_rhs_all_sources_mask(self):
        """All-sources mask should match single-RHS solve."""
        n_sources = self.solver._vec_sources.n_sources
        all_mask = np.ones((1, n_sources), dtype=bool)

        multi_results = self.solver.solve_transient_multi_rhs(
            t_start=0, t_end=10e-9, dt=1e-9,
            source_masks=all_mask,
            method=IntegrationMethod.TRAPEZOIDAL
        )

        single_result = self.solver.solve_transient(
            t_start=0, t_end=10e-9, dt=1e-9,
            method=IntegrationMethod.TRAPEZOIDAL
        )

        # Peak IR-drop should match
        self.assertAlmostEqual(
            multi_results[0].peak_ir_drop,
            single_result.peak_ir_drop,
            places=10,
            msg="Multi-RHS with all sources should match single-RHS"
        )

        # Max IR-drop per time should match
        np.testing.assert_allclose(
            multi_results[0].max_ir_drop_per_time,
            single_result.max_ir_drop_per_time,
            rtol=1e-10,
            err_msg="Max IR-drop per time should match"
        )

    def test_multi_rhs_zero_sources_mask(self):
        """Zero-sources mask should give zero IR-drop."""
        n_sources = self.solver._vec_sources.n_sources
        zero_mask = np.zeros((1, n_sources), dtype=bool)

        results = self.solver.solve_transient_multi_rhs(
            t_start=0, t_end=5e-9, dt=1e-9,
            source_masks=zero_mask
        )

        # With no current sources, IR-drop should be zero (within floating point tolerance)
        self.assertAlmostEqual(results[0].peak_ir_drop, 0.0, places=10)
        np.testing.assert_allclose(
            results[0].max_ir_drop_per_time,
            np.zeros_like(results[0].max_ir_drop_per_time),
            atol=1e-12  # Allow for floating point rounding errors
        )

    def test_multi_rhs_linearity(self):
        """IR-drop should be linear: total ≈ subset1 + subset2 (superposition).

        Superposition holds for individual node voltages/IR-drops, not for
        max() across nodes. So we test linearity on tracked node waveforms.
        """
        n_sources = self.solver._vec_sources.n_sources

        # Split sources into two disjoint sets
        mask_all = np.ones(n_sources, dtype=bool)
        mask_even = np.array([i % 2 == 0 for i in range(n_sources)], dtype=bool)
        mask_odd = ~mask_even

        masks = np.stack([mask_all, mask_even, mask_odd])

        # Track a specific node to verify linearity
        rc = self.solver._ensure_rc_system()
        track_node = rc.unknown_nodes[0]

        results = self.solver.solve_transient_multi_rhs(
            t_start=0, t_end=10e-9, dt=1e-9,
            source_masks=masks,
            method=IntegrationMethod.TRAPEZOIDAL,
            track_nodes=[track_node]
        )

        result_all, result_even, result_odd = results

        # By superposition: IR_all[node] ≈ IR_even[node] + IR_odd[node]
        ir_all = result_all.tracked_ir_drop[track_node]
        ir_even = result_even.tracked_ir_drop[track_node]
        ir_odd = result_odd.tracked_ir_drop[track_node]

        np.testing.assert_allclose(
            ir_all,
            ir_even + ir_odd,
            rtol=1e-8,
            err_msg="Linearity violated: node IR-drop should equal sum of disjoint subsets"
        )

        # Also verify total current linearity
        np.testing.assert_allclose(
            result_all.total_current_per_time,
            result_even.total_current_per_time + result_odd.total_current_per_time,
            rtol=1e-10,
            err_msg="Total current linearity violated"
        )

    def test_multi_rhs_timing_keys(self):
        """solve_transient_multi_rhs should include all timing keys."""
        n_sources = self.solver._vec_sources.n_sources
        masks = np.ones((1, n_sources), dtype=bool)

        results = self.solver.solve_transient_multi_rhs(
            t_start=0, t_end=5e-9, dt=1e-9,
            source_masks=masks
        )

        expected_keys = {
            'build_rc', 'build_map', 'factor', 'dc_init',
            'time_stepping', 'solve', 'evaluate_currents', 'statistics',
            'vsrc_current', 'spatial_peaks', 'tracked_waveforms',
            'convert_results', 'total'
        }

        actual_keys = set(results[0].timings.keys())
        self.assertEqual(actual_keys, expected_keys,
            f"Missing timing keys: {expected_keys - actual_keys}")

    def test_multi_rhs_tracked_waveforms(self):
        """Tracked waveforms should be stored correctly for multi-RHS."""
        n_sources = self.solver._vec_sources.n_sources
        masks = np.stack([
            np.ones(n_sources, dtype=bool),
            np.array([i % 2 == 0 for i in range(n_sources)], dtype=bool),
        ])

        # Get a valid node to track
        rc = self.solver._ensure_rc_system()
        track_node = rc.unknown_nodes[0]

        results = self.solver.solve_transient_multi_rhs(
            t_start=0, t_end=5e-9, dt=1e-9,
            source_masks=masks,
            track_nodes=[track_node]
        )

        # Both results should have tracked waveforms
        for result in results:
            self.assertIn(track_node, result.tracked_waveforms)
            self.assertIn(track_node, result.tracked_ir_drop)
            self.assertEqual(len(result.tracked_waveforms[track_node]), len(result.t_array))

    def test_multi_rhs_total_current_tracking(self):
        """Total current per time should be tracked correctly for each mask."""
        n_sources = self.solver._vec_sources.n_sources
        mask_all = np.ones(n_sources, dtype=bool)
        mask_half = np.array([i < n_sources // 2 for i in range(n_sources)], dtype=bool)

        masks = np.stack([mask_all, mask_half])

        results = self.solver.solve_transient_multi_rhs(
            t_start=0, t_end=5e-9, dt=1e-9,
            source_masks=masks
        )

        # All-sources should have >= half-sources current at each time
        for i in range(len(results[0].t_array)):
            self.assertGreaterEqual(
                results[0].total_current_per_time[i],
                results[1].total_current_per_time[i] - 1e-10,
                msg=f"Total current for all sources should be >= half sources at t={results[0].t_array[i]}"
            )


class TestVectorizedSourcesMultiRHS(unittest.TestCase):
    """Tests for vectorized source methods used by multi-RHS solver."""

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
        self.solver = TransientIRDropSolver(self.model, self.graph, vectorize_threshold=0)

    def test_build_source_to_unknown_direct_from_dict(self):
        """build_source_to_unknown_direct_from_dict should create valid mappings."""
        rc = self.solver._ensure_rc_system()
        vec_sources = self.solver._vec_sources

        source_to_unknown_direct, valid_sources = vec_sources.build_source_to_unknown_direct_from_dict(
            rc.unknown_to_idx,
            self.model.edge_cache.idx_to_node,
        )

        # Should have n_sources entries
        self.assertEqual(len(source_to_unknown_direct), vec_sources.n_sources)
        self.assertEqual(len(valid_sources), vec_sources.n_sources)

        # Valid sources should map to valid unknown indices
        n_unknown = len(rc.unknown_nodes)
        for i in range(vec_sources.n_sources):
            if valid_sources[i]:
                self.assertGreaterEqual(source_to_unknown_direct[i], 0)
                self.assertLess(source_to_unknown_direct[i], n_unknown)
            else:
                self.assertEqual(source_to_unknown_direct[i], -1)

    def test_evaluate_to_multi_rhs_correctness(self):
        """evaluate_to_multi_rhs should produce correct aggregated currents."""
        rc = self.solver._ensure_rc_system()
        vec_sources = self.solver._vec_sources
        n_unknown = len(rc.unknown_nodes)

        # Build direct mapping
        source_to_unknown_direct, valid_sources = vec_sources.build_source_to_unknown_direct_from_dict(
            rc.unknown_to_idx,
            self.model.edge_cache.idx_to_node,
        )

        # Create test masks
        n_sources = vec_sources.n_sources
        masks = np.stack([
            np.ones(n_sources, dtype=bool),
            np.zeros(n_sources, dtype=bool),
        ])

        # Evaluate
        rhs_multi = np.zeros((n_unknown, 2), dtype=np.float64)
        total_currents = vec_sources.evaluate_to_multi_rhs(
            t=0.0, rhs_multi=rhs_multi, source_masks=masks,
            source_to_unknown_direct=source_to_unknown_direct,
            valid_sources=valid_sources
        )

        # All-sources should have non-zero RHS, zero-sources should have zero RHS
        self.assertGreater(np.abs(rhs_multi[:, 0]).sum(), 0,
            "All-sources mask should produce non-zero RHS")
        np.testing.assert_allclose(rhs_multi[:, 1], 0,
            err_msg="Zero-sources mask should produce zero RHS")

        # Total current for zero mask should be zero
        self.assertEqual(total_currents[1], 0.0)

    def test_evaluate_to_multi_rhs_matches_evaluate_to_rhs_array(self):
        """Multi-RHS with single all-ones mask should match single-RHS evaluation."""
        rc = self.solver._ensure_rc_system()
        vec_sources = self.solver._vec_sources
        n_unknown = len(rc.unknown_nodes)

        # Build node-level mapping for single-RHS
        source_to_unknown, valid_mask = vec_sources.build_source_to_unknown_map(
            rc.unknown_to_idx,
            self.model.edge_cache.idx_to_node,
        )

        # Build direct mapping for multi-RHS
        source_to_unknown_direct, valid_sources = vec_sources.build_source_to_unknown_direct_from_dict(
            rc.unknown_to_idx,
            self.model.edge_cache.idx_to_node,
        )

        # Single-RHS evaluation
        rhs_single = np.zeros(n_unknown, dtype=np.float64)
        total_single, _ = vec_sources.evaluate_to_rhs_array(
            t=5e-9, rhs=rhs_single,
            source_to_unknown=source_to_unknown,
            valid_mask=valid_mask
        )

        # Multi-RHS evaluation with all-ones mask
        n_sources = vec_sources.n_sources
        masks = np.ones((1, n_sources), dtype=bool)
        rhs_multi = np.zeros((n_unknown, 1), dtype=np.float64)
        total_multi = vec_sources.evaluate_to_multi_rhs(
            t=5e-9, rhs_multi=rhs_multi, source_masks=masks,
            source_to_unknown_direct=source_to_unknown_direct,
            valid_sources=valid_sources
        )

        # Results should match
        np.testing.assert_allclose(
            rhs_multi[:, 0], rhs_single,
            rtol=1e-10,
            err_msg="Multi-RHS should match single-RHS for all-ones mask"
        )
        self.assertAlmostEqual(total_multi[0], total_single, places=10)


if __name__ == '__main__':
    unittest.main()

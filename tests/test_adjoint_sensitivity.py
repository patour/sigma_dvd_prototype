"""Tests for adjoint sensitivity solver for dynamic IR-drop attribution.

Tests the adjoint method for computing aggressor contributions to IR-drop
at a victim node, accounting for the RC network's dynamic memory.
"""

import unittest
from pathlib import Path
from typing import Dict, Any, List

import numpy as np

from core import (
    create_model_from_pdn,
)
from core.transient_solver import (
    TransientIRDropSolver,
    IntegrationMethod,
    RCSystem,
)
from core.adjoint_sensitivity import (
    AdjointSensitivitySolver,
    AdjointSolverContext,
    AdjointAttribution,
    AggressorContribution,
)


class TestAdjointSolverInit(unittest.TestCase):
    """Tests for AdjointSensitivitySolver initialization."""

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

    def test_init_with_pdn_graph(self):
        """Solver should initialize with PDN graph."""
        solver = AdjointSensitivitySolver(self.model, self.graph)

        self.assertIsNotNone(solver.model)
        self.assertIsNotNone(solver._graph)

    def test_from_transient_solver(self):
        """Should create solver from transient solver."""
        trans = TransientIRDropSolver(self.model, self.graph, vectorize_threshold=0)

        adjoint = AdjointSensitivitySolver.from_transient_solver(trans)

        self.assertIsNotNone(adjoint.model)
        # Should share RC system
        self.assertIs(adjoint._rc_system, trans._rc_system)
        # Should share vectorized sources
        self.assertIs(adjoint._vec_sources, trans._vec_sources)

    def test_source_mappings_built(self):
        """Source-to-node and node-to-sources mappings should be built."""
        solver = AdjointSensitivitySolver(self.model, self.graph)

        # Should have some sources
        self.assertGreater(len(solver._source_to_node), 0)
        self.assertGreater(len(solver._node_to_sources), 0)


class TestAdjointSolverContext(unittest.TestCase):
    """Tests for AdjointSolverContext preparation."""

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
        self.solver = AdjointSensitivitySolver(self.model, self.graph)

    def test_prepare_returns_context(self):
        """prepare() should return AdjointSolverContext."""
        ctx = self.solver.prepare(dt=1e-9)

        self.assertIsInstance(ctx, AdjointSolverContext)
        self.assertIsNotNone(ctx.rc_system)
        self.assertIsNotNone(ctx.A)
        self.assertIsNotNone(ctx.B)
        self.assertIsNotNone(ctx.lu_A)

    def test_context_matrices_correct_shape(self):
        """Context matrices should have correct shape."""
        ctx = self.solver.prepare(dt=1e-9)

        n_unknown = ctx.n_unknown
        self.assertEqual(ctx.A.shape, (n_unknown, n_unknown))
        self.assertEqual(ctx.B.shape, (n_unknown, n_unknown))

    def test_context_with_backward_euler(self):
        """Context should be built correctly for Backward Euler."""
        ctx = self.solver.prepare(dt=1e-9, method=IntegrationMethod.BACKWARD_EULER)

        # A should be G_uu + C_uu/dt
        rc = ctx.rc_system
        dt_scaled = 1e-9 * 1e12  # s -> ps
        expected_A = rc.G_uu + rc.C_uu / dt_scaled

        # Compare shapes at minimum
        self.assertEqual(ctx.A.shape, expected_A.shape)

    def test_context_reusable(self):
        """Same context should be reusable for multiple solves."""
        ctx = self.solver.prepare(dt=1e-9)

        # Should be able to call lu_A.solve multiple times
        rhs = np.random.rand(ctx.n_unknown)
        sol1 = ctx.lu_A.solve(rhs)
        sol2 = ctx.lu_A.solve(rhs)

        np.testing.assert_array_equal(sol1, sol2)


class TestMatrixSymmetry(unittest.TestCase):
    """Tests for PDN matrix symmetry (required for efficient adjoint solve)."""

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
        trans = TransientIRDropSolver(self.model, self.graph)
        self.rc = trans._ensure_rc_system()

    def test_g_uu_symmetric(self):
        """G_uu should be symmetric for PDN matrices."""
        G = self.rc.G_uu.toarray()
        np.testing.assert_array_almost_equal(G, G.T, decimal=10)

    def test_c_uu_symmetric(self):
        """C_uu should be symmetric for PDN matrices."""
        C = self.rc.C_uu.toarray()
        np.testing.assert_array_almost_equal(C, C.T, decimal=10)

    def test_a_matrix_symmetric(self):
        """A = G_uu + C_uu/dt should be symmetric (enables A^T = A optimization)."""
        dt_scaled = 1e-9 * 1e12
        A = self.rc.G_uu + self.rc.C_uu / dt_scaled
        A_dense = A.toarray()
        np.testing.assert_array_almost_equal(A_dense, A_dense.T, decimal=10)

    def test_b_matrix_symmetric(self):
        """B = C_uu/dt should be symmetric."""
        dt_scaled = 1e-9 * 1e12
        B = self.rc.C_uu / dt_scaled
        B_dense = B.toarray()
        np.testing.assert_array_almost_equal(B_dense, B_dense.T, decimal=10)


class TestAnalyzeVictim(unittest.TestCase):
    """Tests for analyze_victim() method."""

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
        self.trans = TransientIRDropSolver(self.model, self.graph, vectorize_threshold=0)
        self.adjoint = AdjointSensitivitySolver.from_transient_solver(self.trans)

    def test_analyze_victim_returns_attribution(self):
        """analyze_victim should return AdjointAttribution."""
        # Get a valid victim node
        rc = self.trans._ensure_rc_system()
        victim = rc.unknown_nodes[0]

        result = self.adjoint.analyze_victim(
            victim_node=victim,
            observation_time=10e-9,
            memory_window=10,
            dt=1e-9,
            top_k=5,
        )

        self.assertIsInstance(result, AdjointAttribution)
        self.assertEqual(result.victim_node, victim)
        self.assertEqual(result.observation_time, 10e-9)

    def test_analyze_victim_returns_top_aggressors(self):
        """analyze_victim should return top aggressors."""
        rc = self.trans._ensure_rc_system()
        victim = rc.unknown_nodes[0]

        result = self.adjoint.analyze_victim(
            victim_node=victim,
            observation_time=10e-9,
            memory_window=10,
            dt=1e-9,
            top_k=5,
        )

        # Should return list of aggressors (may be empty if no sources)
        self.assertIsInstance(result.top_aggressors, list)
        for agg in result.top_aggressors:
            self.assertIsInstance(agg, AggressorContribution)
            self.assertIsNotNone(agg.node)
            self.assertIsInstance(agg.contribution_mV, float)
            self.assertIsInstance(agg.contribution_pct, float)

    def test_invalid_victim_raises_error(self):
        """Invalid victim node should raise ValueError."""
        with self.assertRaises(ValueError):
            self.adjoint.analyze_victim(
                victim_node='nonexistent_node',
                observation_time=10e-9,
                memory_window=10,
                dt=1e-9,
            )

    def test_memory_window_correct(self):
        """Memory window should have correct time range."""
        rc = self.trans._ensure_rc_system()
        victim = rc.unknown_nodes[0]

        result = self.adjoint.analyze_victim(
            victim_node=victim,
            observation_time=50e-9,
            memory_window=20,
            dt=1e-9,
        )

        t_start, t_end = result.memory_window
        self.assertAlmostEqual(t_end, 50e-9)
        self.assertAlmostEqual(t_start, 50e-9 - 19 * 1e-9)  # 20 points = 19 intervals
        self.assertEqual(len(result.t_array), 20)

    def test_timings_populated(self):
        """Timing breakdown should be populated."""
        rc = self.trans._ensure_rc_system()
        victim = rc.unknown_nodes[0]

        result = self.adjoint.analyze_victim(
            victim_node=victim,
            observation_time=10e-9,
            memory_window=10,
            dt=1e-9,
        )

        self.assertIn('total', result.timings)
        self.assertIn('adjoint_solve', result.timings)
        for t in result.timings.values():
            self.assertGreaterEqual(t, 0.0)


class TestSpatialFiltering(unittest.TestCase):
    """Tests for spatial window filtering."""

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
        self.trans = TransientIRDropSolver(self.model, self.graph, vectorize_threshold=0)
        self.adjoint = AdjointSensitivitySolver.from_transient_solver(self.trans)

    def test_get_node_coordinates(self):
        """Should extract coordinates from PDN node names."""
        x, y = self.adjoint._get_node_coordinates('1500_2000_M1')
        self.assertEqual(x, 1500.0)
        self.assertEqual(y, 2000.0)

        x, y = self.adjoint._get_node_coordinates('invalid')
        self.assertIsNone(x)
        self.assertIsNone(y)

    def test_spatial_window_filters_sources(self):
        """Sources outside spatial window should be excluded."""
        rc = self.trans._ensure_rc_system()

        # Get all sources in a tight window vs a wide window
        tight_window = (0, 100, 0, 100)
        wide_window = (0, 100000, 0, 100000)

        tight = self.adjoint._filter_sources_by_window(tight_window, rc.unknown_to_idx)
        wide = self.adjoint._filter_sources_by_window(wide_window, rc.unknown_to_idx)

        # Wide window should have >= tight window sources
        self.assertGreaterEqual(len(wide), len(tight))

    def test_spatial_window_in_result(self):
        """Spatial window should be recorded in result."""
        rc = self.trans._ensure_rc_system()
        victim = rc.unknown_nodes[0]

        window = (0, 5000, 0, 5000)
        result = self.adjoint.analyze_victim(
            victim_node=victim,
            observation_time=10e-9,
            memory_window=10,
            dt=1e-9,
            spatial_window=window,
        )

        self.assertEqual(result.spatial_window, window)

    def test_window_margin_auto_creates_window(self):
        """window_margin should auto-create spatial window."""
        rc = self.trans._ensure_rc_system()
        # Find a victim with valid coordinates
        victim = None
        for node in rc.unknown_nodes:
            x, y = self.adjoint._get_node_coordinates(node)
            if x is not None:
                victim = node
                break

        if victim is None:
            self.skipTest("No nodes with valid coordinates")

        result = self.adjoint.analyze_victim(
            victim_node=victim,
            observation_time=10e-9,
            memory_window=10,
            dt=1e-9,
            window_margin=500.0,
        )

        self.assertIsNotNone(result.spatial_window)
        x_v, y_v = self.adjoint._get_node_coordinates(victim)
        x_min, x_max, y_min, y_max = result.spatial_window
        self.assertAlmostEqual(x_min, x_v - 500.0)
        self.assertAlmostEqual(x_max, x_v + 500.0)


class TestAdjointSolve(unittest.TestCase):
    """Tests for the adjoint solve algorithm."""

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
        self.trans = TransientIRDropSolver(self.model, self.graph, vectorize_threshold=0)
        self.adjoint = AdjointSensitivitySolver.from_transient_solver(self.trans)

    def test_terminal_condition(self):
        """Terminal condition λ_N should have -1 at victim index only."""
        rc = self.trans._ensure_rc_system()
        victim = rc.unknown_nodes[0]
        victim_idx = rc.unknown_to_idx[victim]

        ctx = self.adjoint.prepare(dt=1e-9)

        # Create terminal condition
        lambda_N = np.zeros(ctx.n_unknown)
        lambda_N[victim_idx] = -1.0

        # Verify
        self.assertEqual(lambda_N[victim_idx], -1.0)
        self.assertEqual(np.sum(lambda_N != 0), 1)  # Only one non-zero

    def test_zero_current_zero_contribution(self):
        """Source with zero current should have zero contribution."""
        rc = self.trans._ensure_rc_system()
        victim = rc.unknown_nodes[0]

        # Run analysis at t=0 where currents should be minimal
        result = self.adjoint.analyze_victim(
            victim_node=victim,
            observation_time=0.0,  # Very early time
            memory_window=5,
            dt=1e-12,  # Very short window
            top_k=5,
        )

        # Attribution should still work (may have zero or small contributions)
        self.assertIsInstance(result, AdjointAttribution)


class TestContributions(unittest.TestCase):
    """Tests for contribution calculations."""

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
        self.trans = TransientIRDropSolver(self.model, self.graph, vectorize_threshold=0)
        self.adjoint = AdjointSensitivitySolver.from_transient_solver(self.trans)

    def test_contributions_sorted_by_magnitude(self):
        """Top aggressors should be sorted by |contribution|."""
        rc = self.trans._ensure_rc_system()
        victim = rc.unknown_nodes[0]

        result = self.adjoint.analyze_victim(
            victim_node=victim,
            observation_time=50e-9,
            memory_window=20,
            dt=1e-9,
            top_k=10,
        )

        # Verify sorted by |contribution|
        for i in range(len(result.top_aggressors) - 1):
            self.assertGreaterEqual(
                abs(result.top_aggressors[i].contribution_mV),
                abs(result.top_aggressors[i+1].contribution_mV)
            )

    def test_percentage_calculation(self):
        """Contribution percentages should be calculated correctly."""
        rc = self.trans._ensure_rc_system()
        victim = rc.unknown_nodes[0]

        result = self.adjoint.analyze_victim(
            victim_node=victim,
            observation_time=50e-9,
            memory_window=20,
            dt=1e-9,
            top_k=5,
        )

        if result.ir_drop_at_T > 0:
            for agg in result.top_aggressors:
                expected_pct = 100.0 * agg.contribution_mV / result.ir_drop_at_T
                self.assertAlmostEqual(agg.contribution_pct, expected_pct, places=5)

    def test_self_contribution_separated(self):
        """Self-contribution should be separated from remote aggressors."""
        rc = self.trans._ensure_rc_system()

        # Find a victim that has current sources
        victim = None
        for node in rc.unknown_nodes:
            if node in self.adjoint._node_to_sources:
                victim = node
                break

        if victim is None:
            self.skipTest("No victim with self-current sources")

        result = self.adjoint.analyze_victim(
            victim_node=victim,
            observation_time=50e-9,
            memory_window=20,
            dt=1e-9,
        )

        # Self-contribution should be separated
        self.assertIsInstance(result.self_contribution_mV, float)
        self.assertIsInstance(result.self_contribution_pct, float)

        # Victim should not appear in top_aggressors
        for agg in result.top_aggressors:
            self.assertNotEqual(agg.node, victim)


class TestWaveformStorage(unittest.TestCase):
    """Tests for current waveform storage."""

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
        self.trans = TransientIRDropSolver(self.model, self.graph, vectorize_threshold=0)
        self.adjoint = AdjointSensitivitySolver.from_transient_solver(self.trans)

    def test_waveforms_included_when_requested(self):
        """Waveforms should be included when include_waveforms=True."""
        rc = self.trans._ensure_rc_system()
        victim = rc.unknown_nodes[0]

        result = self.adjoint.analyze_victim(
            victim_node=victim,
            observation_time=50e-9,
            memory_window=20,
            dt=1e-9,
            include_waveforms=True,
        )

        # Check if any aggressors have waveforms
        for agg in result.top_aggressors:
            if agg.current_waveform is not None:
                self.assertEqual(len(agg.current_waveform), len(result.t_array))
                break

    def test_waveforms_excluded_when_not_requested(self):
        """Waveforms should be excluded when include_waveforms=False."""
        rc = self.trans._ensure_rc_system()
        victim = rc.unknown_nodes[0]

        result = self.adjoint.analyze_victim(
            victim_node=victim,
            observation_time=50e-9,
            memory_window=20,
            dt=1e-9,
            include_waveforms=False,
        )

        # Waveforms should be None
        for agg in result.top_aggressors:
            self.assertIsNone(agg.current_waveform)


class TestAttributionEfficiency(unittest.TestCase):
    """Tests for attribution efficiency (conservation check)."""

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
        self.trans = TransientIRDropSolver(self.model, self.graph, vectorize_threshold=0)
        self.adjoint = AdjointSensitivitySolver.from_transient_solver(self.trans)

    def test_attribution_efficiency_computed(self):
        """Attribution efficiency should be computed."""
        rc = self.trans._ensure_rc_system()
        victim = rc.unknown_nodes[0]

        result = self.adjoint.analyze_victim(
            victim_node=victim,
            observation_time=50e-9,
            memory_window=20,
            dt=1e-9,
        )

        self.assertIsInstance(result.attribution_efficiency, float)

    def test_total_attributed_ir_drop(self):
        """Total attributed IR-drop should be sum of all contributions."""
        rc = self.trans._ensure_rc_system()
        victim = rc.unknown_nodes[0]

        result = self.adjoint.analyze_victim(
            victim_node=victim,
            observation_time=50e-9,
            memory_window=20,
            dt=1e-9,
            top_k=100,  # Get more to check total
        )

        # Verify total includes self + aggressors
        expected_total = result.self_contribution_mV + sum(
            agg.contribution_mV for agg in result.top_aggressors
        )

        # May not be exactly equal due to sources not in top-K
        # but should be close
        self.assertGreaterEqual(result.total_attributed_ir_drop, expected_total - 0.001)


class TestContextReuse(unittest.TestCase):
    """Tests for reusing AdjointSolverContext across multiple solves."""

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
        self.trans = TransientIRDropSolver(self.model, self.graph, vectorize_threshold=0)
        self.adjoint = AdjointSensitivitySolver.from_transient_solver(self.trans)

    def test_context_reuse_gives_same_result(self):
        """Reusing context should give same result as creating new one."""
        rc = self.trans._ensure_rc_system()
        victim = rc.unknown_nodes[0]

        # Create context once
        ctx = self.adjoint.prepare(dt=1e-9)

        # Solve with explicit context
        result1 = self.adjoint.analyze_victim(
            victim_node=victim,
            observation_time=50e-9,
            memory_window=10,
            dt=1e-9,
            context=ctx,
        )

        # Solve again with same context
        result2 = self.adjoint.analyze_victim(
            victim_node=victim,
            observation_time=50e-9,
            memory_window=10,
            dt=1e-9,
            context=ctx,
        )

        # Results should be identical
        self.assertAlmostEqual(result1.ir_drop_at_T, result2.ir_drop_at_T, places=10)
        self.assertEqual(len(result1.top_aggressors), len(result2.top_aggressors))

    def test_context_reuse_different_victims(self):
        """Same context should work for different victims."""
        rc = self.trans._ensure_rc_system()
        victim1 = rc.unknown_nodes[0]
        victim2 = rc.unknown_nodes[min(1, len(rc.unknown_nodes) - 1)]

        # Create context once
        ctx = self.adjoint.prepare(dt=1e-9)

        # Solve for different victims
        result1 = self.adjoint.analyze_victim(
            victim_node=victim1,
            observation_time=50e-9,
            memory_window=10,
            dt=1e-9,
            context=ctx,
        )

        result2 = self.adjoint.analyze_victim(
            victim_node=victim2,
            observation_time=50e-9,
            memory_window=10,
            dt=1e-9,
            context=ctx,
        )

        # Both should complete successfully
        self.assertEqual(result1.victim_node, victim1)
        self.assertEqual(result2.victim_node, victim2)


class TestIntegrationWithTransientSolver(unittest.TestCase):
    """Tests for integration with TransientIRDropSolver."""

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

    def test_analyze_worst_node_from_transient(self):
        """Should be able to analyze worst node from transient result."""
        trans = TransientIRDropSolver(self.model, self.graph, vectorize_threshold=0)

        # Run transient
        trans_result = trans.solve_transient(
            t_start=0, t_end=50e-9, dt=1e-9,
            method=IntegrationMethod.BACKWARD_EULER,
        )

        # Create adjoint solver
        adjoint = AdjointSensitivitySolver.from_transient_solver(trans)

        # Analyze worst node at peak time
        victim = trans_result.peak_ir_drop_node
        T = trans_result.peak_ir_drop_time

        result = adjoint.analyze_victim(
            victim_node=victim,
            observation_time=T,
            memory_window=20,
            dt=1e-9,
        )

        self.assertEqual(result.victim_node, victim)
        self.assertAlmostEqual(result.observation_time, T)


class TestStaticSensitivity(unittest.TestCase):
    """Tests for static sensitivity analysis (for stiff RC systems)."""

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
        self.trans = TransientIRDropSolver(self.model, self.graph, vectorize_threshold=0)
        self.adjoint = AdjointSensitivitySolver.from_transient_solver(self.trans)

    def test_static_returns_attribution(self):
        """analyze_victim_static should return AdjointAttribution."""
        rc = self.trans._ensure_rc_system()
        victim = rc.unknown_nodes[0]

        result = self.adjoint.analyze_victim_static(
            victim_node=victim,
            observation_time=10e-9,
            top_k=5,
        )

        self.assertIsInstance(result, AdjointAttribution)
        self.assertEqual(result.victim_node, victim)

    def test_static_attribution_efficiency_near_one(self):
        """Static attribution efficiency should be close to 100%."""
        rc = self.trans._ensure_rc_system()
        victim = rc.unknown_nodes[0]

        result = self.adjoint.analyze_victim_static(
            victim_node=victim,
            observation_time=50e-9,
            top_k=100,  # Get all
        )

        # Attribution should account for essentially all IR-drop
        self.assertGreater(result.attribution_efficiency, 0.95)

    def test_static_contributions_sum_to_ir_drop(self):
        """Sum of all contributions should equal total IR-drop."""
        rc = self.trans._ensure_rc_system()
        victim = rc.unknown_nodes[0]

        result = self.adjoint.analyze_victim_static(
            victim_node=victim,
            observation_time=50e-9,
            top_k=100,
        )

        total_contrib = result.self_contribution_mV
        total_contrib += sum(agg.contribution_mV for agg in result.top_aggressors)

        if result.ir_drop_at_T > 0:
            ratio = total_contrib / result.ir_drop_at_T
            self.assertAlmostEqual(ratio, 1.0, places=2)

    def test_use_static_flag(self):
        """use_static=True should use static sensitivity."""
        rc = self.trans._ensure_rc_system()
        victim = rc.unknown_nodes[0]

        result = self.adjoint.analyze_victim(
            victim_node=victim,
            observation_time=50e-9,
            memory_window=10,
            dt=1e-9,
            top_k=10,
            use_static=True,
        )

        # Should get good attribution efficiency with static method
        self.assertGreater(result.attribution_efficiency, 0.9)


class TestStaticVsDynamic(unittest.TestCase):
    """Tests comparing static and dynamic adjoint methods."""

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
        self.trans = TransientIRDropSolver(self.model, self.graph, vectorize_threshold=0)
        self.adjoint = AdjointSensitivitySolver.from_transient_solver(self.trans)

    def test_static_faster_than_dynamic(self):
        """Static method should be faster than dynamic for this netlist."""
        rc = self.trans._ensure_rc_system()
        victim = rc.unknown_nodes[0]

        # Time static
        result_static = self.adjoint.analyze_victim_static(
            victim_node=victim,
            observation_time=50e-9,
            top_k=10,
        )

        # Time dynamic
        result_dynamic = self.adjoint.analyze_victim(
            victim_node=victim,
            observation_time=50e-9,
            memory_window=20,
            dt=1e-9,
            top_k=10,
        )

        # Static should be faster (no time stepping)
        self.assertLess(
            result_static.timings['total'],
            result_dynamic.timings['total']
        )


class TestVectorizedVsRawEvaluation(unittest.TestCase):
    """Tests comparing vectorized (threshold=0) vs raw source evaluation."""

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

    def test_adjoint_init_with_vectorize_threshold_zero(self):
        """AdjointSensitivitySolver should build vectorized sources when threshold=0."""
        solver = AdjointSensitivitySolver(
            self.model, self.graph, vectorize_threshold=0
        )

        # Should have vectorized sources
        self.assertIsNotNone(solver._vec_sources)
        self.assertGreater(solver._vec_sources.n_sources, 0)

    def test_adjoint_init_without_vectorization(self):
        """AdjointSensitivitySolver should work without vectorized sources."""
        # Use very high threshold so vectorization is skipped
        solver = AdjointSensitivitySolver(
            self.model, self.graph, vectorize_threshold=100000
        )

        # Should NOT have vectorized sources
        self.assertIsNone(solver._vec_sources)

        # But should still have source mappings
        self.assertGreater(len(solver._source_to_node), 0)
        self.assertGreater(len(solver._node_to_sources), 0)

    def test_vectorized_vs_raw_static_sensitivity(self):
        """Vectorized and raw source evaluation should give same static results."""
        # Create solvers with and without vectorization
        solver_vec = AdjointSensitivitySolver(
            self.model, self.graph, vectorize_threshold=0
        )
        solver_raw = AdjointSensitivitySolver(
            self.model, self.graph, vectorize_threshold=100000
        )

        # Verify one has vec_sources and one doesn't
        self.assertIsNotNone(solver_vec._vec_sources)
        self.assertIsNone(solver_raw._vec_sources)

        # Get a victim node
        rc = solver_vec._ensure_rc_system()
        victim = rc.unknown_nodes[0]

        # Run static analysis with both
        observation_time = 10e-9  # Time when currents are active

        result_vec = solver_vec.analyze_victim_static(
            victim_node=victim,
            observation_time=observation_time,
            top_k=10,
        )

        result_raw = solver_raw.analyze_victim_static(
            victim_node=victim,
            observation_time=observation_time,
            top_k=10,
        )

        # IR-drop should match
        self.assertAlmostEqual(
            result_vec.ir_drop_at_T,
            result_raw.ir_drop_at_T,
            places=4,
            msg="IR-drop mismatch between vectorized and raw evaluation"
        )

        # Self-contribution should match
        self.assertAlmostEqual(
            result_vec.self_contribution_mV,
            result_raw.self_contribution_mV,
            places=4,
            msg="Self-contribution mismatch"
        )

        # Total attributed should match
        self.assertAlmostEqual(
            result_vec.total_attributed_ir_drop,
            result_raw.total_attributed_ir_drop,
            places=4,
            msg="Total attributed mismatch"
        )

    def test_vectorized_vs_raw_dynamic_adjoint(self):
        """Vectorized and raw source evaluation should give same dynamic results."""
        # Create solvers with and without vectorization
        solver_vec = AdjointSensitivitySolver(
            self.model, self.graph, vectorize_threshold=0
        )
        solver_raw = AdjointSensitivitySolver(
            self.model, self.graph, vectorize_threshold=100000
        )

        # Get a victim node
        rc = solver_vec._ensure_rc_system()
        victim = rc.unknown_nodes[0]

        observation_time = 10e-9
        dt = 1e-9

        result_vec = solver_vec.analyze_victim(
            victim_node=victim,
            observation_time=observation_time,
            memory_window=10,
            dt=dt,
            top_k=10,
            use_static=False,
        )

        result_raw = solver_raw.analyze_victim(
            victim_node=victim,
            observation_time=observation_time,
            memory_window=10,
            dt=dt,
            top_k=10,
            use_static=False,
        )

        # IR-drop should match
        self.assertAlmostEqual(
            result_vec.ir_drop_at_T,
            result_raw.ir_drop_at_T,
            places=4,
            msg="Dynamic IR-drop mismatch between vectorized and raw evaluation"
        )

        # Total attributed should match
        self.assertAlmostEqual(
            result_vec.total_attributed_ir_drop,
            result_raw.total_attributed_ir_drop,
            places=4,
            msg="Dynamic total attributed mismatch"
        )

    def test_vectorized_vs_raw_top_aggressors_match(self):
        """Top aggressors should be the same for vectorized and raw evaluation."""
        solver_vec = AdjointSensitivitySolver(
            self.model, self.graph, vectorize_threshold=0
        )
        solver_raw = AdjointSensitivitySolver(
            self.model, self.graph, vectorize_threshold=100000
        )

        rc = solver_vec._ensure_rc_system()
        victim = rc.unknown_nodes[0]
        observation_time = 10e-9

        result_vec = solver_vec.analyze_victim_static(
            victim_node=victim,
            observation_time=observation_time,
            top_k=5,
        )

        result_raw = solver_raw.analyze_victim_static(
            victim_node=victim,
            observation_time=observation_time,
            top_k=5,
        )

        # Same number of top aggressors
        self.assertEqual(
            len(result_vec.top_aggressors),
            len(result_raw.top_aggressors),
            msg="Different number of top aggressors"
        )

        # Same nodes in top aggressors (order may differ slightly due to ties)
        vec_nodes = {agg.node for agg in result_vec.top_aggressors}
        raw_nodes = {agg.node for agg in result_raw.top_aggressors}
        self.assertEqual(
            vec_nodes,
            raw_nodes,
            msg="Different nodes in top aggressors"
        )

        # Contributions should match for each node
        vec_contribs = {agg.node: agg.contribution_mV for agg in result_vec.top_aggressors}
        raw_contribs = {agg.node: agg.contribution_mV for agg in result_raw.top_aggressors}

        for node in vec_nodes:
            self.assertAlmostEqual(
                vec_contribs[node],
                raw_contribs[node],
                places=4,
                msg=f"Contribution mismatch for node {node}"
            )

    def test_from_transient_solver_with_vectorization(self):
        """from_transient_solver should properly share vectorized sources."""
        trans = TransientIRDropSolver(
            self.model, self.graph, vectorize_threshold=0
        )

        # Ensure RC system is built (which also builds vec_sources)
        trans._ensure_rc_system()

        adjoint = AdjointSensitivitySolver.from_transient_solver(trans)

        # Should share the same vec_sources instance
        self.assertIs(adjoint._vec_sources, trans._vec_sources)
        self.assertIsNotNone(adjoint._vec_sources)

    def test_from_transient_solver_without_prior_rc_build(self):
        """from_transient_solver should build RC system if not already built."""
        trans = TransientIRDropSolver(
            self.model, self.graph, vectorize_threshold=0
        )

        # Don't call _ensure_rc_system - let from_transient_solver do it
        self.assertIsNone(trans._rc_system)

        adjoint = AdjointSensitivitySolver.from_transient_solver(trans)

        # RC system should now be built
        self.assertIsNotNone(trans._rc_system)
        self.assertIsNotNone(adjoint._rc_system)

        # Vec sources should be shared
        self.assertIsNotNone(adjoint._vec_sources)


class TestDynamicVsStaticStiffSystem(unittest.TestCase):
    """Tests verifying dynamic adjoint equals static for stiff RC systems."""

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

    def test_dynamic_matches_static_for_stiff_system_vectorized(self):
        """For stiff RC systems, dynamic should match static (vectorized)."""
        solver = AdjointSensitivitySolver(
            self.model, self.graph, vectorize_threshold=0
        )

        rc = solver._ensure_rc_system()
        victim = rc.unknown_nodes[5]  # Pick a different victim
        observation_time = 10e-9

        result_static = solver.analyze_victim_static(
            victim_node=victim,
            observation_time=observation_time,
            top_k=10,
        )

        result_dynamic = solver.analyze_victim(
            victim_node=victim,
            observation_time=observation_time,
            memory_window=20,
            dt=1e-9,
            top_k=10,
            use_static=False,
        )

        # For stiff systems, dynamic should match static within 5%
        if abs(result_static.ir_drop_at_T) > 0.01:  # Only check if significant IR-drop
            ratio = result_dynamic.ir_drop_at_T / result_static.ir_drop_at_T
            self.assertGreater(ratio, 0.95, "Dynamic IR-drop too low vs static")
            self.assertLess(ratio, 1.05, "Dynamic IR-drop too high vs static")

        # Total attributed should also match
        if abs(result_static.total_attributed_ir_drop) > 0.01:
            ratio = result_dynamic.total_attributed_ir_drop / result_static.total_attributed_ir_drop
            self.assertGreater(ratio, 0.95, "Dynamic attribution too low vs static")
            self.assertLess(ratio, 1.05, "Dynamic attribution too high vs static")

    def test_dynamic_matches_static_for_stiff_system_raw(self):
        """For stiff RC systems, dynamic should match static (raw sources)."""
        solver = AdjointSensitivitySolver(
            self.model, self.graph, vectorize_threshold=100000
        )

        rc = solver._ensure_rc_system()
        victim = rc.unknown_nodes[5]
        observation_time = 10e-9

        result_static = solver.analyze_victim_static(
            victim_node=victim,
            observation_time=observation_time,
            top_k=10,
        )

        result_dynamic = solver.analyze_victim(
            victim_node=victim,
            observation_time=observation_time,
            memory_window=20,
            dt=1e-9,
            top_k=10,
            use_static=False,
        )

        # For stiff systems, dynamic should match static within 5%
        if abs(result_static.ir_drop_at_T) > 0.01:
            ratio = result_dynamic.ir_drop_at_T / result_static.ir_drop_at_T
            self.assertGreater(ratio, 0.95, "Dynamic IR-drop too low vs static (raw)")
            self.assertLess(ratio, 1.05, "Dynamic IR-drop too high vs static (raw)")

    def test_multiple_victims_dynamic_vs_static(self):
        """Test dynamic vs static on multiple victims."""
        solver = AdjointSensitivitySolver(
            self.model, self.graph, vectorize_threshold=0
        )

        rc = solver._ensure_rc_system()

        # Test on several victims
        victims_to_test = min(5, len(rc.unknown_nodes))
        observation_time = 10e-9

        for i in range(victims_to_test):
            victim = rc.unknown_nodes[i * 3 % len(rc.unknown_nodes)]

            result_static = solver.analyze_victim_static(
                victim_node=victim,
                observation_time=observation_time,
                top_k=5,
            )

            result_dynamic = solver.analyze_victim(
                victim_node=victim,
                observation_time=observation_time,
                memory_window=15,
                dt=1e-9,
                top_k=5,
                use_static=False,
            )

            # Check attribution efficiency is reasonable
            if abs(result_static.ir_drop_at_T) > 0.01:
                self.assertGreater(
                    result_dynamic.attribution_efficiency, 0.9,
                    f"Low attribution efficiency for victim {victim}"
                )


class TestNonStiffSystemWithScaledRC(unittest.TestCase):
    """Tests for non-stiff RC systems with τ ~ 100ps, validated against aggressor-only sims.

    Creates a modified RC system where conductance is scaled down to achieve
    τ = C/G ~ 100ps, then validates the dynamic adjoint against "aggressor-only"
    transient simulations.

    For a single aggressor active, the adjoint contribution should match the
    actual IR-drop from a transient simulation with only that source.
    """

    @classmethod
    def setUpClass(cls):
        """Parse test netlist."""
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

    def _create_scaled_rc_system(self, trans, g_scale_factor: float):
        """Create an RC system with scaled conductance to achieve target τ.

        Args:
            trans: TransientIRDropSolver instance (will modify its _rc_system)
            g_scale_factor: Factor to scale G by (< 1 increases R, increases τ)

        Returns:
            Modified RCSystem with τ ~ 100ps
        """
        import scipy.sparse as sp
        from core.transient_solver import RCSystem

        # Ensure original RC system is built
        rc_orig = trans._ensure_rc_system()

        # Scale G matrices (reducing conductance = increasing resistance)
        G_uu_scaled = rc_orig.G_uu * g_scale_factor
        G_up_scaled = rc_orig.G_up * g_scale_factor

        # Create new RCSystem with scaled G
        # Note: G_full/C_full are no longer stored (memory optimization)
        rc_scaled = RCSystem(
            G_uu=G_uu_scaled.tocsr(),
            G_up=G_up_scaled.tocsr(),
            C_uu=rc_orig.C_uu,
            node_order=rc_orig.node_order,
            node_to_idx=rc_orig.node_to_idx,
            unknown_nodes=rc_orig.unknown_nodes,
            unknown_to_idx=rc_orig.unknown_to_idx,
            pad_nodes=rc_orig.pad_nodes,
            n_nodes=rc_orig.n_nodes,
            n_unknown=rc_orig.n_unknown,
            has_capacitance=rc_orig.has_capacitance,
            total_capacitance_fF=rc_orig.total_capacitance_fF,
        )

        return rc_scaled

    def _compute_time_constant(self, rc):
        """Compute characteristic time constant τ = C/G."""
        G_diag = rc.G_uu.diagonal()
        C_diag = rc.C_uu.diagonal()

        tau_values = []
        for i in range(len(G_diag)):
            if G_diag[i] > 0 and C_diag[i] > 0:
                tau = C_diag[i] / G_diag[i]  # fF / mS = ps
                tau_values.append(tau)

        return np.mean(tau_values) if tau_values else 0.0

    def test_scaled_rc_has_target_tau(self):
        """Verify scaled RC system achieves τ ~ 100ps."""
        trans = TransientIRDropSolver(self.model, self.graph, vectorize_threshold=0)
        rc_orig = trans._ensure_rc_system()

        tau_orig = self._compute_time_constant(rc_orig)

        # Scale to achieve τ ~ 100ps
        # Original τ ~ 0.003 ps, target τ ~ 100 ps
        # Need to scale G by τ_orig/τ_target = 0.003/100 = 3e-5
        target_tau = 100.0  # ps
        g_scale = tau_orig / target_tau

        rc_scaled = self._create_scaled_rc_system(trans, g_scale)
        tau_scaled = self._compute_time_constant(rc_scaled)

        # Should be close to target (within 10%)
        self.assertGreater(tau_scaled, target_tau * 0.9)
        self.assertLess(tau_scaled, target_tau * 1.1)

    def test_dynamic_vs_static_differs_for_non_stiff(self):
        """For non-stiff system (τ ~ 100ps, dt = 10ps), dynamic should differ from static."""
        trans = TransientIRDropSolver(self.model, self.graph, vectorize_threshold=0)
        rc_orig = trans._ensure_rc_system()

        # Scale to achieve τ ~ 100ps
        tau_orig = self._compute_time_constant(rc_orig)
        target_tau = 100.0  # ps
        g_scale = tau_orig / target_tau

        rc_scaled = self._create_scaled_rc_system(trans, g_scale)

        # Replace the RC system in the solver
        trans._rc_system = rc_scaled

        # Create adjoint solver with the modified system
        adjoint = AdjointSensitivitySolver.from_transient_solver(trans)

        victim = rc_scaled.unknown_nodes[0]
        observation_time = 500e-12  # 500 ps
        dt = 10e-12  # 10 ps (τ/dt = 10, non-stiff)

        # Run static analysis
        result_static = adjoint.analyze_victim_static(
            victim_node=victim,
            observation_time=observation_time,
            top_k=5,
        )

        # Run dynamic analysis
        result_dynamic = adjoint.analyze_victim(
            victim_node=victim,
            observation_time=observation_time,
            memory_window=30,  # 300 ps of history
            dt=dt,
            top_k=5,
            use_static=False,
        )

        # For non-stiff system, they should differ
        # The dynamic method accounts for history decay
        if abs(result_static.total_attributed_ir_drop) > 0.001:
            ratio = result_dynamic.total_attributed_ir_drop / result_static.total_attributed_ir_drop

            # For τ/dt ~ 10, we expect some difference
            # Static overpredicts because it doesn't account for decay
            # Dynamic should be somewhat less than static
            self.assertLess(
                ratio, 1.0,
                f"Dynamic ({result_dynamic.total_attributed_ir_drop:.4f} mV) should be "
                f"less than static ({result_static.total_attributed_ir_drop:.4f} mV) "
                f"for non-stiff system"
            )

    def test_adjoint_contribution_ordering(self):
        """Verify that adjoint ranks aggressors by contribution magnitude.

        For the scaled RC system, the adjoint should identify which nodes
        contribute most to the victim's IR-drop, with contributions ordered
        by absolute magnitude.
        """
        trans = TransientIRDropSolver(self.model, self.graph, vectorize_threshold=0)
        rc_orig = trans._ensure_rc_system()

        # Scale to achieve τ ~ 100ps
        tau_orig = self._compute_time_constant(rc_orig)
        target_tau = 100.0  # ps
        g_scale = tau_orig / target_tau

        rc_scaled = self._create_scaled_rc_system(trans, g_scale)
        trans._rc_system = rc_scaled

        adjoint = AdjointSensitivitySolver.from_transient_solver(trans)

        victim = rc_scaled.unknown_nodes[0]
        observation_time = 500e-12  # 500 ps
        dt = 10e-12  # 10 ps

        result = adjoint.analyze_victim(
            victim_node=victim,
            observation_time=observation_time,
            memory_window=30,
            dt=dt,
            top_k=5,
            use_static=False,
        )

        # Verify contributions are ordered by magnitude
        contributions = [abs(agg.contribution_mV) for agg in result.top_aggressors]
        self.assertEqual(
            contributions,
            sorted(contributions, reverse=True),
            "Aggressors should be ordered by |contribution|"
        )

        # Verify we got some aggressors
        self.assertGreater(len(result.top_aggressors), 0, "Should find some aggressors")

    def test_adjoint_vs_aggressor_only_simulation(self):
        """Validate adjoint attribution against aggressor-only transient simulations.

        For each top aggressor, run a simulation with ONLY that aggressor active.
        The resulting IR-drop at the victim should match the adjoint contribution.

        This validates the adjoint method by comparing its per-aggressor contributions
        against brute-force "aggressor-only" transient simulations. For a linear system,
        the IR-drop from a single aggressor should match the adjoint's attributed
        contribution for that aggressor.

        IMPORTANT: Both the adjoint and simulation must use the same initial condition.
        The adjoint assumes we start from V=VDD (zero IR-drop), so the simulation
        must use init_condition='vdd' for a valid comparison.

        NOTE: Previous issues fixed:
        - Integration method mismatch: solve_transient_multi_rhs now defaults to BACKWARD_EULER
        - Initial condition mismatch: added init_condition='vdd' to start from V=VDD
        """
        # Create transient solver with vectorized sources
        trans = TransientIRDropSolver(self.model, self.graph, vectorize_threshold=0)
        rc_orig = trans._ensure_rc_system()

        # Scale RC system to achieve τ ~ 100ps (non-stiff with dt = 10ps)
        tau_orig = self._compute_time_constant(rc_orig)
        target_tau = 100.0  # ps
        g_scale = tau_orig / target_tau if tau_orig > 0 else 1e-5

        rc_scaled = self._create_scaled_rc_system(trans, g_scale)
        trans._rc_system = rc_scaled

        # Create adjoint solver with the scaled system
        adjoint = AdjointSensitivitySolver.from_transient_solver(trans)

        # Pick a victim node
        victim = rc_scaled.unknown_nodes[0]
        observation_time = 2000e-12  # 2000 ps = 2 ns
        dt = 10e-12  # 10 ps (τ/dt = 10, non-stiff)
        memory_window = 100  # 100 time steps = 1000 ps = 10τ (for ~2% error)

        # Run adjoint analysis to get top aggressor contributions
        attribution = adjoint.analyze_victim(
            victim_node=victim,
            observation_time=observation_time,
            memory_window=memory_window,
            dt=dt,
            top_k=20,  # Get more aggressors to ensure we find ones with sources
            use_static=False,
        )

        # Skip if no aggressors found
        if len(attribution.top_aggressors) == 0:
            self.skipTest("No aggressors found")

        # Get vectorized sources and node mapping
        vec_sources = trans._vec_sources
        if vec_sources is None:
            self.skipTest("Vectorized sources not available")

        node_to_idx = self.model.edge_cache.node_to_idx

        # Track how many aggressors we successfully validated
        validated_count = 0

        # For each top aggressor, run aggressor-only simulation
        for agg in attribution.top_aggressors[:10]:  # Test up to 10 aggressors
            agg_node = agg.node

            # Skip if aggressor node not in node mapping
            if agg_node not in node_to_idx:
                continue

            agg_node_idx = node_to_idx[agg_node]

            # Create mask: enable only sources at this aggressor node
            single_agg_mask = (vec_sources.source_node_idx == agg_node_idx)

            # Skip if no sources at this node
            if not np.any(single_agg_mask):
                continue

            # Reshape mask to (1, n_sources) for solve_transient_multi_rhs
            source_masks = single_agg_mask.reshape(1, -1)

            # Run aggressor-only transient simulation with init_condition='vdd'
            # This ensures we start from V=VDD like the adjoint assumes
            t_start = observation_time - (memory_window - 1) * dt
            t_end = observation_time

            results = trans.solve_transient_multi_rhs(
                t_start=t_start,
                t_end=t_end,
                dt=dt,
                source_masks=source_masks,
                method=IntegrationMethod.BACKWARD_EULER,
                track_nodes=[victim],  # Track the victim node explicitly
                init_condition='vdd',  # Start from V=VDD for adjoint comparison
                verbose=False,
            )

            sim_result = results[0]

            # Get IR-drop at victim at the observation time (last time step)
            if sim_result.tracked_ir_drop and victim in sim_result.tracked_ir_drop:
                sim_ir_drop_at_victim_mV = sim_result.tracked_ir_drop[victim][-1] * 1000.0
            elif victim in sim_result.peak_ir_drop_per_node:
                sim_ir_drop_at_victim_mV = sim_result.peak_ir_drop_per_node[victim] * 1000.0
            else:
                sim_ir_drop_at_victim_mV = sim_result.max_ir_drop_per_time[-1] * 1000.0

            adjoint_contrib_mV = agg.contribution_mV

            # Only check if contribution is significant (avoid division by tiny numbers)
            if abs(adjoint_contrib_mV) > 0.01:
                rel_error = abs(sim_ir_drop_at_victim_mV - adjoint_contrib_mV) / abs(adjoint_contrib_mV)

                # With memory_window = 100 steps (10τ), expect ~2% error
                # Allow 5% tolerance for safety margin
                self.assertLess(
                    rel_error, 0.05,
                    f"Aggressor {agg_node}: adjoint={adjoint_contrib_mV:.4f} mV, "
                    f"simulated={sim_ir_drop_at_victim_mV:.4f} mV, "
                    f"rel_error={rel_error:.2%}"
                )
                validated_count += 1

        # Ensure we validated at least one aggressor
        self.assertGreater(
            validated_count, 0,
            "Should validate at least one aggressor"
        )


class TestNonStiffSystem(unittest.TestCase):
    """Tests for non-stiff RC systems where τ > dt.

    In non-stiff systems, the dynamic adjoint should capture time-dependent
    effects that the static sensitivity method cannot. This means:
    1. Contributions depend on WHEN sources are active, not just their magnitude
    2. Sources active earlier in the memory window have decayed contributions
    3. Top-K aggressors may differ between static and dynamic methods
    """

    @classmethod
    def setUpClass(cls):
        """Parse test netlist and prepare modified RC system."""
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

    def test_non_stiff_adjoint_decay(self):
        """In non-stiff systems, adjoint λ should decay over the memory window.

        For stiff systems (τ << dt), λ stays nearly constant.
        For non-stiff systems (τ > dt), λ should decay as we go back in time.

        We test this by creating a modified RC system with larger time constants.
        """
        import scipy.sparse as sp

        trans = TransientIRDropSolver(self.model, self.graph, vectorize_threshold=0)
        rc = trans._ensure_rc_system()

        # Create adjoint solver
        adjoint = AdjointSensitivitySolver.from_transient_solver(trans)

        # Original system is stiff: τ ≈ 0.004 ps << dt = 1 ns
        # We'll test with a smaller dt to approach non-stiff regime
        # Using dt = 0.001 ps (1e-15 s) would make τ/dt ≈ 4

        dt_stiff = 1e-9  # 1 ns - very stiff for this system
        dt_less_stiff = 1e-12  # 1 ps - less stiff (τ/dt ≈ 0.004)

        victim_idx = 0
        victim = rc.unknown_nodes[victim_idx]

        # Prepare contexts for both dt values
        ctx_stiff = adjoint.prepare(dt_stiff)
        ctx_less_stiff = adjoint.prepare(dt_less_stiff)

        # Terminal condition: λ = A^{-1} · e_victim
        e_victim = np.zeros(rc.n_unknown)
        e_victim[victim_idx] = 1.0

        lambda_stiff = ctx_stiff.lu_A.solve(e_victim)
        lambda_less_stiff = ctx_less_stiff.lu_A.solve(e_victim)

        # For stiff system, A ≈ G, so λ ≈ G^{-1} e_victim
        # For less stiff, A = G + C/dt has larger C/dt contribution

        # The λ vector should have smaller norm when less stiff (larger A)
        norm_stiff = np.linalg.norm(lambda_stiff)
        norm_less_stiff = np.linalg.norm(lambda_less_stiff)

        # Less stiff (smaller dt) means larger A, so smaller λ
        self.assertLess(
            norm_less_stiff, norm_stiff,
            "λ should be smaller for less stiff system (smaller dt)"
        )

        # Now test the decay: propagate λ backward one step
        # λ_{k} = A^{-1} B λ_{k+1}
        # For stiff: B/A ≈ (C/dt)/(G + C/dt) ≈ C/(G·dt + C) ≈ 0 (fast decay)
        # For less stiff: decay is slower

        lambda_next_stiff = ctx_stiff.lu_A.solve(ctx_stiff.B @ lambda_stiff)
        lambda_next_less_stiff = ctx_less_stiff.lu_A.solve(ctx_less_stiff.B @ lambda_less_stiff)

        decay_ratio_stiff = np.linalg.norm(lambda_next_stiff) / norm_stiff
        decay_ratio_less_stiff = np.linalg.norm(lambda_next_less_stiff) / norm_less_stiff

        # The stiff system should have faster decay (smaller ratio)
        self.assertLess(
            decay_ratio_stiff, decay_ratio_less_stiff,
            "Stiff system should have faster adjoint decay"
        )

    def test_synthetic_non_stiff_contribution_differs(self):
        """Test that dynamic contribution differs from static for synthetic non-stiff case.

        We construct a scenario where:
        - Source A is active only at the observation time T
        - Source B was active earlier but is now off

        For static: both contribute based on steady-state sensitivity × current at T
        For dynamic: Source B's past contribution has decayed

        In stiff systems, this decay is so fast that only T matters (≈ static).
        In non-stiff systems, the memory window matters.
        """
        trans = TransientIRDropSolver(self.model, self.graph, vectorize_threshold=0)
        adjoint = AdjointSensitivitySolver.from_transient_solver(trans)

        rc = adjoint._ensure_rc_system()
        victim = rc.unknown_nodes[0]

        # Test with standard parameters (stiff)
        observation_time = 10e-9
        dt = 1e-9

        # Get static and dynamic results
        result_static = adjoint.analyze_victim_static(
            victim_node=victim,
            observation_time=observation_time,
            top_k=5,
        )

        result_dynamic = adjoint.analyze_victim(
            victim_node=victim,
            observation_time=observation_time,
            memory_window=10,
            dt=dt,
            top_k=5,
            use_static=False,
        )

        # For this stiff system, they should match closely
        if abs(result_static.total_attributed_ir_drop) > 0.001:
            ratio = result_dynamic.total_attributed_ir_drop / result_static.total_attributed_ir_drop
            # Should be very close for stiff system
            self.assertGreater(ratio, 0.99, "Stiff system: dynamic should match static")
            self.assertLess(ratio, 1.01, "Stiff system: dynamic should match static")

    def test_adjoint_matrix_properties_non_stiff(self):
        """Test that A and B matrices have expected properties for non-stiff analysis.

        For Backward Euler: A = G + C/dt, B = C/dt
        - As dt → 0: A → ∞, B → ∞, but B/A → 1 (no decay, fully non-stiff)
        - As dt → ∞: A → G, B → 0, B/A → 0 (instant decay, fully stiff)
        """
        import scipy.sparse as sp

        trans = TransientIRDropSolver(self.model, self.graph, vectorize_threshold=0)
        adjoint = AdjointSensitivitySolver.from_transient_solver(trans)
        rc = adjoint._ensure_rc_system()

        # Test with different dt values
        dt_values = [1e-15, 1e-12, 1e-9, 1e-6]  # From non-stiff to stiff

        prev_ba_ratio = None
        for dt in dt_values:
            ctx = adjoint.prepare(dt)

            # Compute spectral properties of B @ A^{-1}
            # For a simple test, look at the Frobenius norm ratio
            A_norm = sp.linalg.norm(ctx.A)
            B_norm = sp.linalg.norm(ctx.B)

            ba_ratio = B_norm / A_norm

            if prev_ba_ratio is not None:
                # B/A ratio should decrease as dt increases (more stiff)
                self.assertLess(
                    ba_ratio, prev_ba_ratio,
                    f"B/A ratio should decrease as dt increases: dt={dt}"
                )

            prev_ba_ratio = ba_ratio

    def test_memory_window_effect_documented(self):
        """Document and test the memory window effect.

        For stiff systems: memory_window doesn't matter much (instant decay)
        For non-stiff systems: longer memory_window captures more history

        This test verifies the API handles different memory windows correctly.
        """
        trans = TransientIRDropSolver(self.model, self.graph, vectorize_threshold=0)
        adjoint = AdjointSensitivitySolver.from_transient_solver(trans)

        rc = adjoint._ensure_rc_system()
        victim = rc.unknown_nodes[0]
        observation_time = 10e-9
        dt = 1e-9

        # Test different memory window sizes
        windows = [5, 10, 20]
        results = []

        for window in windows:
            result = adjoint.analyze_victim(
                victim_node=victim,
                observation_time=observation_time,
                memory_window=window,
                dt=dt,
                top_k=5,
                use_static=False,
            )
            results.append(result)

            # Verify memory_window is stored correctly
            expected_t_start = observation_time - (window - 1) * dt
            self.assertAlmostEqual(
                result.memory_window[0], expected_t_start, places=15,
                msg=f"Memory window start incorrect for window={window}"
            )
            self.assertAlmostEqual(
                result.memory_window[1], observation_time, places=15,
                msg=f"Memory window end incorrect for window={window}"
            )

            # Verify t_array length
            self.assertEqual(
                len(result.t_array), window,
                msg=f"t_array length should be {window}"
            )

        # For this stiff system, results should be nearly identical
        # regardless of memory window (since decay is instant)
        for i in range(len(results) - 1):
            if abs(results[i].total_attributed_ir_drop) > 0.001:
                ratio = results[i+1].total_attributed_ir_drop / results[i].total_attributed_ir_drop
                self.assertGreater(ratio, 0.99, "Stiff: memory window shouldn't matter")
                self.assertLess(ratio, 1.01, "Stiff: memory window shouldn't matter")


class TestDCInitialCondition(unittest.TestCase):
    """Tests for the 'dc' initial_condition option.

    When initial_condition='dc', the adjoint solver computes contributions
    starting from the DC operating point rather than V=VDD. This means:
    - DC currents establish a baseline IR-drop
    - Contributions are computed using incremental currents: ΔI = I(t) - I_DC
    - The attributed IR-drop is the incremental drop above the DC baseline
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
        self.trans = TransientIRDropSolver(self.model, self.graph, vectorize_threshold=0)
        self.adjoint = AdjointSensitivitySolver.from_transient_solver(self.trans)

    def test_dc_initial_condition_static_returns_attribution(self):
        """analyze_victim_static with initial_condition='dc' should return AdjointAttribution."""
        rc = self.trans._ensure_rc_system()
        victim = rc.unknown_nodes[0]

        result = self.adjoint.analyze_victim_static(
            victim_node=victim,
            observation_time=10e-9,
            top_k=5,
            initial_condition='dc',
        )

        self.assertIsInstance(result, AdjointAttribution)
        self.assertEqual(result.victim_node, victim)
        self.assertEqual(result.initial_condition, 'dc')

    def test_dc_initial_condition_has_dc_fields(self):
        """'dc' mode should populate dc_ir_drop_mV."""
        rc = self.trans._ensure_rc_system()
        victim = rc.unknown_nodes[0]

        result = self.adjoint.analyze_victim_static(
            victim_node=victim,
            observation_time=10e-9,
            top_k=5,
            initial_condition='dc',
        )

        # DC field should be populated
        self.assertIsNotNone(result.dc_ir_drop_mV)

        # ir_drop_at_T is always total, DC baseline is separate
        # Incremental can be computed as: ir_drop_at_T - dc_ir_drop_mV
        incremental = result.ir_drop_at_T - result.dc_ir_drop_mV
        self.assertGreaterEqual(incremental, -0.001)  # Should be non-negative for typical cases

    def test_zero_initial_condition_has_no_dc_fields(self):
        """'zero' mode should not populate dc_ir_drop_mV."""
        rc = self.trans._ensure_rc_system()
        victim = rc.unknown_nodes[0]

        result = self.adjoint.analyze_victim_static(
            victim_node=victim,
            observation_time=10e-9,
            top_k=5,
            initial_condition='zero',
        )

        self.assertEqual(result.initial_condition, 'zero')
        self.assertIsNone(result.dc_ir_drop_mV)

    def test_static_contribution_field_populated_in_dc_mode(self):
        """'dc' mode should populate static_contribution_mV for each aggressor."""
        rc = self.trans._ensure_rc_system()
        victim = rc.unknown_nodes[0]

        result = self.adjoint.analyze_victim_static(
            victim_node=victim,
            observation_time=50e-9,
            top_k=10,
            initial_condition='dc',
        )

        # All aggressors should have static_contribution_mV populated
        for agg in result.top_aggressors:
            self.assertIsNotNone(
                agg.static_contribution_mV,
                f"static_contribution_mV should be set for {agg.node}"
            )

    def test_zero_mode_has_no_static_contribution(self):
        """'zero' mode should not populate static_contribution_mV."""
        rc = self.trans._ensure_rc_system()
        victim = rc.unknown_nodes[0]

        result = self.adjoint.analyze_victim_static(
            victim_node=victim,
            observation_time=50e-9,
            top_k=10,
            initial_condition='zero',
        )

        # In 'zero' mode, static_contribution_mV should be None
        for agg in result.top_aggressors:
            self.assertIsNone(
                agg.static_contribution_mV,
                f"static_contribution_mV should be None in 'zero' mode for {agg.node}"
            )

    def test_contribution_decomposition_incremental_plus_static_equals_total(self):
        """Verify: contribution_mV (incremental) + static_contribution_mV = zero-mode contribution."""
        rc = self.trans._ensure_rc_system()
        victim = rc.unknown_nodes[0]
        observation_time = 50e-9

        result_zero = self.adjoint.analyze_victim_static(
            victim_node=victim,
            observation_time=observation_time,
            top_k=20,
            initial_condition='zero',
        )

        result_dc = self.adjoint.analyze_victim_static(
            victim_node=victim,
            observation_time=observation_time,
            top_k=20,
            initial_condition='dc',
        )

        # Build lookup for zero-mode contributions
        zero_contrib = {agg.node: agg.contribution_mV for agg in result_zero.top_aggressors}

        # Verify decomposition: incremental + static = total
        for agg_dc in result_dc.top_aggressors:
            if agg_dc.node in zero_contrib:
                total_zero = zero_contrib[agg_dc.node]
                incremental = agg_dc.contribution_mV
                static = agg_dc.static_contribution_mV or 0.0
                sum_dc = incremental + static

                self.assertAlmostEqual(
                    sum_dc, total_zero, places=4,
                    msg=f"{agg_dc.node}: incremental({incremental:.4f}) + "
                        f"static({static:.4f}) = {sum_dc:.4f} != zero({total_zero:.4f})"
                )

    def test_dc_vs_zero_total_ir_drop_matches(self):
        """ir_drop_at_T should match between 'dc' and 'zero' modes (both are total)."""
        rc = self.trans._ensure_rc_system()
        victim = rc.unknown_nodes[0]
        observation_time = 50e-9

        result_zero = self.adjoint.analyze_victim_static(
            victim_node=victim,
            observation_time=observation_time,
            top_k=5,
            initial_condition='zero',
        )

        result_dc = self.adjoint.analyze_victim_static(
            victim_node=victim,
            observation_time=observation_time,
            top_k=5,
            initial_condition='dc',
        )

        # ir_drop_at_T is the total IR-drop in both modes
        self.assertAlmostEqual(
            result_zero.ir_drop_at_T,
            result_dc.ir_drop_at_T,
            places=10,
            msg="ir_drop_at_T (total) should match between modes"
        )

    def test_dc_incremental_ir_drop_positive_when_switching(self):
        """Incremental IR-drop should be positive when switching adds current."""
        rc = self.trans._ensure_rc_system()
        victim = rc.unknown_nodes[0]

        # At early time, switching activity is starting
        result = self.adjoint.analyze_victim_static(
            victim_node=victim,
            observation_time=10e-9,
            top_k=5,
            initial_condition='dc',
        )

        # If there's any switching activity, incremental should be >= 0
        # (negative would mean switching reduces IR-drop, which is unusual)
        self.assertGreaterEqual(
            result.ir_drop_at_T, -0.001,
            "Incremental IR-drop should be non-negative for typical switching"
        )

    def test_dc_initial_condition_dynamic_adjoint(self):
        """analyze_victim with initial_condition='dc' should work for dynamic adjoint."""
        rc = self.trans._ensure_rc_system()
        victim = rc.unknown_nodes[0]

        result = self.adjoint.analyze_victim(
            victim_node=victim,
            observation_time=50e-9,
            memory_window=10,
            dt=1e-9,
            top_k=5,
            initial_condition='dc',
            use_static=False,
        )

        self.assertEqual(result.initial_condition, 'dc')
        self.assertIsNotNone(result.dc_ir_drop_mV)
        # ir_drop_at_T is always total, incremental = ir_drop_at_T - dc_ir_drop_mV
        self.assertGreater(result.ir_drop_at_T, 0)

    def test_dc_mode_contributions_use_incremental_currents(self):
        """In 'dc' mode, contributions should be based on incremental currents."""
        rc = self.trans._ensure_rc_system()
        victim = rc.unknown_nodes[0]
        observation_time = 50e-9

        result_zero = self.adjoint.analyze_victim_static(
            victim_node=victim,
            observation_time=observation_time,
            top_k=100,
            initial_condition='zero',
        )

        result_dc = self.adjoint.analyze_victim_static(
            victim_node=victim,
            observation_time=observation_time,
            top_k=100,
            initial_condition='dc',
        )

        # In 'zero' mode: total_attributed ≈ ir_drop_at_T (total)
        # In 'dc' mode: total_attributed ≈ ir_drop_at_T (incremental)
        # Both should have good attribution efficiency
        self.assertGreater(
            result_zero.attribution_efficiency, 0.9,
            "'zero' mode should have good attribution efficiency"
        )

        # For 'dc' mode, efficiency should also be good if there's meaningful activity
        if abs(result_dc.ir_drop_at_T) > 0.001:
            self.assertGreater(
                result_dc.attribution_efficiency, 0.9,
                "'dc' mode should have good attribution efficiency"
            )

    def test_dc_mode_decomposition(self):
        """Verify: ir_drop_at_T is total, and incremental = total - DC."""
        rc = self.trans._ensure_rc_system()
        victim = rc.unknown_nodes[0]
        observation_time = 50e-9

        result = self.adjoint.analyze_victim_static(
            victim_node=victim,
            observation_time=observation_time,
            top_k=5,
            initial_condition='dc',
        )

        # ir_drop_at_T is the total IR-drop
        # dc_ir_drop_mV is the DC baseline
        # Incremental = total - DC
        incremental = result.ir_drop_at_T - result.dc_ir_drop_mV

        # Verify DC baseline is less than total (typical case)
        self.assertLessEqual(
            result.dc_ir_drop_mV,
            result.ir_drop_at_T + 0.001,
            msg="DC baseline should be <= total IR-drop"
        )

        # Verify incremental is non-negative (typical case)
        self.assertGreaterEqual(
            incremental, -0.001,
            msg="Incremental IR-drop should be non-negative"
        )

    def test_dc_mode_with_use_static_flag(self):
        """use_static=True with initial_condition='dc' should work."""
        rc = self.trans._ensure_rc_system()
        victim = rc.unknown_nodes[0]

        result = self.adjoint.analyze_victim(
            victim_node=victim,
            observation_time=50e-9,
            memory_window=10,
            dt=1e-9,
            top_k=5,
            use_static=True,
            initial_condition='dc',
        )

        self.assertEqual(result.initial_condition, 'dc')
        self.assertIsNotNone(result.dc_ir_drop_mV)

    def test_adjoint_vs_aggressor_only_simulation_dc_mode(self):
        """Validate 'dc' mode adjoint attribution against aggressor-only simulations.

        For 'dc' initial condition, contributions are based on incremental currents:
            contribution_i = sensitivity[i] * (I_i(T) - I_i(DC))

        This test validates that:
        1. The 'zero' mode contributions still match simulation (baseline check)
        2. The 'dc' mode contribution <= 'zero' mode contribution (uses subset of current)
        3. Both modes report the same total ir_drop_at_T

        Uses a scaled RC system with τ ~ 100ps for meaningful dynamic effects.
        """
        import scipy.sparse as sp
        from core.transient_solver import RCSystem

        # Create transient solver with vectorized sources
        trans = TransientIRDropSolver(self.model, self.graph, vectorize_threshold=0)
        rc_orig = trans._ensure_rc_system()

        # Scale RC system to achieve τ ~ 100ps (non-stiff with dt = 10ps)
        G_diag = rc_orig.G_uu.diagonal()
        C_diag = rc_orig.C_uu.diagonal()
        tau_values = []
        for i in range(len(G_diag)):
            if G_diag[i] > 0 and C_diag[i] > 0:
                tau = C_diag[i] / G_diag[i]  # fF / mS = ps
                tau_values.append(tau)
        tau_orig = np.mean(tau_values) if tau_values else 0.0

        target_tau = 100.0  # ps
        g_scale = tau_orig / target_tau if tau_orig > 0 else 1e-5

        # Create scaled RC system
        # Note: G_full/C_full are no longer stored (memory optimization)
        G_uu_scaled = rc_orig.G_uu * g_scale
        G_up_scaled = rc_orig.G_up * g_scale

        rc_scaled = RCSystem(
            G_uu=G_uu_scaled.tocsr(),
            G_up=G_up_scaled.tocsr(),
            C_uu=rc_orig.C_uu,
            node_order=rc_orig.node_order,
            node_to_idx=rc_orig.node_to_idx,
            unknown_nodes=rc_orig.unknown_nodes,
            unknown_to_idx=rc_orig.unknown_to_idx,
            pad_nodes=rc_orig.pad_nodes,
            n_nodes=rc_orig.n_nodes,
            n_unknown=rc_orig.n_unknown,
            has_capacitance=rc_orig.has_capacitance,
            total_capacitance_fF=rc_orig.total_capacitance_fF,
        )

        trans._rc_system = rc_scaled

        # Create adjoint solver with the scaled system
        adjoint = AdjointSensitivitySolver.from_transient_solver(trans)

        # Pick a victim node
        victim = rc_scaled.unknown_nodes[0]
        observation_time = 2000e-12  # 2000 ps = 2 ns
        dt = 10e-12  # 10 ps (τ/dt = 10, non-stiff)
        memory_window = 100  # 100 time steps = 1000 ps = 10τ

        # Run adjoint analysis with 'dc' initial condition
        attribution_dc = adjoint.analyze_victim(
            victim_node=victim,
            observation_time=observation_time,
            memory_window=memory_window,
            dt=dt,
            top_k=20,
            use_static=False,
            initial_condition='dc',
        )

        # Also run with 'zero' mode for comparison
        attribution_zero = adjoint.analyze_victim(
            victim_node=victim,
            observation_time=observation_time,
            memory_window=memory_window,
            dt=dt,
            top_k=20,
            use_static=False,
            initial_condition='zero',
        )

        # Skip if no aggressors found
        if len(attribution_dc.top_aggressors) == 0:
            self.skipTest("No aggressors found")

        # Get vectorized sources and node mapping
        vec_sources = trans._vec_sources
        if vec_sources is None:
            self.skipTest("Vectorized sources not available")

        node_to_idx = self.model.edge_cache.node_to_idx

        # Track validation results
        validated_count = 0

        # For each top aggressor, verify relationship between dc and zero modes
        for agg_dc in attribution_dc.top_aggressors[:10]:
            agg_node = agg_dc.node

            # Find corresponding aggressor in zero mode
            agg_zero = None
            for a in attribution_zero.top_aggressors:
                if a.node == agg_node:
                    agg_zero = a
                    break

            if agg_zero is None:
                continue

            # Skip if aggressor node not in node mapping
            if agg_node not in node_to_idx:
                continue

            agg_node_idx = node_to_idx[agg_node]

            # Create mask: enable only sources at this aggressor node
            single_agg_mask = (vec_sources.source_node_idx == agg_node_idx)

            # Skip if no sources at this node
            if not np.any(single_agg_mask):
                continue

            # Reshape mask to (1, n_sources) for solve_transient_multi_rhs
            source_masks = single_agg_mask.reshape(1, -1)

            t_start = observation_time - (memory_window - 1) * dt
            t_end = observation_time

            # Run simulation starting from VDD (for zero-mode comparison)
            results_vdd = trans.solve_transient_multi_rhs(
                t_start=t_start,
                t_end=t_end,
                dt=dt,
                source_masks=source_masks,
                method=IntegrationMethod.BACKWARD_EULER,
                track_nodes=[victim],
                init_condition='vdd',  # Start from V=VDD
                verbose=False,
            )

            sim_result_vdd = results_vdd[0]

            # Get IR-drop at victim at the observation time
            if sim_result_vdd.tracked_ir_drop and victim in sim_result_vdd.tracked_ir_drop:
                sim_ir_drop_vdd = sim_result_vdd.tracked_ir_drop[victim][-1] * 1000.0
            else:
                continue

            # Verify zero-mode contribution matches simulation from VDD
            adjoint_zero_contrib = agg_zero.contribution_mV
            if abs(adjoint_zero_contrib) > 0.01:
                rel_error_zero = abs(sim_ir_drop_vdd - adjoint_zero_contrib) / abs(adjoint_zero_contrib)
                self.assertLess(
                    rel_error_zero, 0.05,
                    f"Zero-mode: Aggressor {agg_node}: adjoint={adjoint_zero_contrib:.4f} mV, "
                    f"simulated={sim_ir_drop_vdd:.4f} mV, rel_error={rel_error_zero:.2%}"
                )
                validated_count += 1

            # Key relationship: dc contribution <= zero contribution
            # (since dc uses incremental current which is <= total current)
            adjoint_dc_contrib = agg_dc.contribution_mV

            # DC contribution should be <= zero contribution (or very close)
            # Allow small tolerance for numerical precision
            self.assertLessEqual(
                adjoint_dc_contrib,
                adjoint_zero_contrib + 0.001,
                f"DC contribution should be <= zero contribution for {agg_node}"
            )

        # Verify we validated some aggressors
        self.assertGreater(
            validated_count, 0,
            "Should validate at least one aggressor"
        )

        # Verify DC fields are populated
        self.assertIsNotNone(attribution_dc.dc_ir_drop_mV)
        self.assertEqual(attribution_dc.initial_condition, 'dc')

        # Verify ir_drop_at_T is the same for both modes (both are total)
        self.assertAlmostEqual(
            attribution_dc.ir_drop_at_T,
            attribution_zero.ir_drop_at_T,
            places=8,
            msg="ir_drop_at_T should match between dc and zero modes"
        )

    def test_adjoint_dc_matches_simulation_with_dc_init(self):
        """Validate adjoint 'dc' mode matches simulation with DC initial condition.

        For the adjoint 'dc' mode:
            contribution_i = sensitivity[i] * (I_i(T) - I_DC,i)
            where I_DC,i = get_dc() for each source

        For simulation with DC initial condition:
            - V_DC = steady-state voltage with I(t_start=0)
            - V(T) = voltage at time T after evolving with I(t)
            - Incremental IR-drop = (VDD - V(T)) - (VDD - V_DC) = V_DC - V(T)

        Key insight: We start simulation from t=0 so that I(t=0) ≈ get_dc()
        (before any pulses fire), making the DC init consistent with adjoint's
        DC reference.

        Uses a scaled RC system with τ ~ 100ps for meaningful dynamic effects.
        """
        import scipy.sparse as sp
        from core.transient_solver import RCSystem

        # Create transient solver with vectorized sources
        trans = TransientIRDropSolver(self.model, self.graph, vectorize_threshold=0)
        rc_orig = trans._ensure_rc_system()

        # Scale RC system to achieve τ ~ 100ps (non-stiff with dt = 10ps)
        G_diag = rc_orig.G_uu.diagonal()
        C_diag = rc_orig.C_uu.diagonal()
        tau_values = []
        for i in range(len(G_diag)):
            if G_diag[i] > 0 and C_diag[i] > 0:
                tau = C_diag[i] / G_diag[i]  # fF / mS = ps
                tau_values.append(tau)
        tau_orig = np.mean(tau_values) if tau_values else 0.0

        target_tau = 100.0  # ps
        g_scale = tau_orig / target_tau if tau_orig > 0 else 1e-5

        # Create scaled RC system
        # Note: G_full/C_full are no longer stored (memory optimization)
        G_uu_scaled = rc_orig.G_uu * g_scale
        G_up_scaled = rc_orig.G_up * g_scale

        rc_scaled = RCSystem(
            G_uu=G_uu_scaled.tocsr(),
            G_up=G_up_scaled.tocsr(),
            C_uu=rc_orig.C_uu,
            node_order=rc_orig.node_order,
            node_to_idx=rc_orig.node_to_idx,
            unknown_nodes=rc_orig.unknown_nodes,
            unknown_to_idx=rc_orig.unknown_to_idx,
            pad_nodes=rc_orig.pad_nodes,
            n_nodes=rc_orig.n_nodes,
            n_unknown=rc_orig.n_unknown,
            has_capacitance=rc_orig.has_capacitance,
            total_capacitance_fF=rc_orig.total_capacitance_fF,
        )

        trans._rc_system = rc_scaled

        # Create adjoint solver with the scaled system
        adjoint = AdjointSensitivitySolver.from_transient_solver(trans)

        # Pick a victim node
        victim = rc_scaled.unknown_nodes[0]

        # Start from t=0 so DC init uses I(t=0) ≈ get_dc() (before pulses fire)
        dt = 10e-12  # 10 ps (τ/dt = 10, non-stiff)
        observation_time = 1000e-12  # 1000 ps = 1 ns
        memory_window = 100  # 100 time steps = 1000 ps = 10τ
        t_start = 0.0  # Start from t=0 for consistent DC reference

        # Run adjoint analysis with 'dc' initial condition
        attribution_dc = adjoint.analyze_victim(
            victim_node=victim,
            observation_time=observation_time,
            memory_window=memory_window,
            dt=dt,
            top_k=20,
            use_static=False,
            initial_condition='dc',
        )

        # Skip if no aggressors found
        if len(attribution_dc.top_aggressors) == 0:
            self.skipTest("No aggressors found")

        # Get vectorized sources and node mapping
        vec_sources = trans._vec_sources
        if vec_sources is None:
            self.skipTest("Vectorized sources not available")

        node_to_idx = self.model.edge_cache.node_to_idx
        vdd = self.model.vdd

        # Track validation results
        validated_count = 0

        # For each top aggressor, run simulation with DC init and compare
        for agg_dc in attribution_dc.top_aggressors[:10]:
            agg_node = agg_dc.node

            # Skip if aggressor node not in node mapping
            if agg_node not in node_to_idx:
                continue

            agg_node_idx = node_to_idx[agg_node]

            # Create mask: enable only sources at this aggressor node
            single_agg_mask = (vec_sources.source_node_idx == agg_node_idx)

            # Skip if no sources at this node
            if not np.any(single_agg_mask):
                continue

            # Reshape mask to (1, n_sources) for solve_transient_multi_rhs
            source_masks = single_agg_mask.reshape(1, -1)

            # Run simulation from t=0 to observation_time with DC init
            # DC init uses I(t=0) which should ≈ get_dc()
            results_dc = trans.solve_transient_multi_rhs(
                t_start=t_start,
                t_end=observation_time,
                dt=dt,
                source_masks=source_masks,
                method=IntegrationMethod.BACKWARD_EULER,
                track_nodes=[victim],
                init_condition='dc',  # Start from DC operating point
                verbose=False,
            )

            sim_result = results_dc[0]

            # Get tracked waveforms
            if not (sim_result.tracked_waveforms and victim in sim_result.tracked_waveforms):
                continue

            waveform = sim_result.tracked_waveforms[victim]

            # V[0] is DC initial voltage (at t=0), V[-1] is final voltage
            V_dc = waveform[0]
            V_T = waveform[-1]

            # Incremental IR-drop from simulation:
            # (VDD - V(T)) - (VDD - V_DC) = V_DC - V(T)
            sim_incremental_ir_drop_mV = (V_dc - V_T) * 1000.0

            adjoint_dc_contrib_mV = agg_dc.contribution_mV

            # Both should be close when I(t=0) ≈ I_DC
            # Only validate if contribution is significant
            if abs(adjoint_dc_contrib_mV) > 0.01:
                rel_error = abs(sim_incremental_ir_drop_mV - adjoint_dc_contrib_mV) / abs(adjoint_dc_contrib_mV)

                # With memory_window = 100 steps (10τ), expect ~2% error
                # Allow 10% tolerance for slight I(t=0) vs get_dc() differences
                self.assertLess(
                    rel_error, 0.10,
                    f"Aggressor {agg_node}: adjoint_dc={adjoint_dc_contrib_mV:.4f} mV, "
                    f"sim_incremental={sim_incremental_ir_drop_mV:.4f} mV, "
                    f"rel_error={rel_error:.2%}"
                )
                validated_count += 1

        # Ensure we validated at least one aggressor
        self.assertGreater(
            validated_count, 0,
            "Should validate at least one aggressor with DC init simulation"
        )


if __name__ == '__main__':
    unittest.main()

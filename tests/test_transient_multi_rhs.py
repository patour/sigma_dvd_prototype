"""Tests for solve_transient_multi_rhs() method.

Validates that the multi-RHS transient solver:
1. Produces identical results to standard solve_transient() for single mask
2. Correctly applies multiple masks independently
3. Maintains linearity (total = near + far)
4. Handles edge cases (empty mask, single time step)

Usage:
    # Run with default netlist (netlist_test, fast)
    python -m unittest tests.test_transient_multi_rhs

    # Run with larger netlist (netlist_small, slower but more thorough)
    python -m tests.test_transient_multi_rhs --small
"""

import os
import sys
import unittest
from pathlib import Path
from typing import Dict, Any, List, Set

import numpy as np

from core import create_model_from_pdn
from core.transient_solver import (
    TransientIRDropSolver,
    TransientResult,
    IntegrationMethod,
)

# Check for --small flag to use netlist_small instead of netlist_test
USE_SMALL_NETLIST = '--small' in sys.argv or os.environ.get('USE_SMALL_NETLIST', '').lower() in ('1', 'true', 'yes')

# Remove --small from sys.argv so unittest doesn't complain
if '--small' in sys.argv:
    sys.argv.remove('--small')


class TestTransientMultiRHSBase(unittest.TestCase):
    """Base test class with shared fixtures for multi-RHS tests."""

    @classmethod
    def setUpClass(cls):
        """Parse test netlist once for all tests."""
        if USE_SMALL_NETLIST:
            test_netlist = Path(__file__).parent.parent / 'pdn' / 'netlist_small'
            net_name = 'VDD_XLV'
        else:
            test_netlist = Path(__file__).parent.parent / 'pdn' / 'netlist_test'
            net_name = 'VDD'

        if not test_netlist.exists():
            cls.graph = None
            cls.model = None
            cls.current_sources = None
            return

        from pdn.pdn_parser import NetlistParser
        parser = NetlistParser(str(test_netlist))
        cls.graph = parser.parse()
        cls.model = create_model_from_pdn(cls.graph, net_name)

        # Get current sources
        graph_dict = cls.graph.graph if hasattr(cls.graph, 'graph') else cls.graph._attrs
        cls.current_sources = graph_dict.get('_instance_sources_objects', {})

        # Build source list and masks
        cls.source_names = list(cls.current_sources.keys())
        cls.n_sources = len(cls.source_names)

        # Find a suitable track node (one with a current source)
        cls.track_node = None
        for name, src in cls.current_sources.items():
            node = getattr(src, 'node1', None)
            if node and node != '0':
                cls.track_node = node
                break

    def setUp(self):
        """Skip if test netlist not available."""
        if self.model is None:
            self.skipTest("Test netlist not available")
        # Use vectorize_threshold=0 to force vectorization (required for solve_transient_multi_rhs)
        self.solver = TransientIRDropSolver(self.model, self.graph, vectorize_threshold=0)


class TestSingleMaskMatchesStandardSolve(TestTransientMultiRHSBase):
    """Test that single mask (all True) matches standard solve_transient()."""

    def test_single_mask_all_true_matches_standard(self):
        """Single mask with all sources should produce identical results to solve_transient()."""
        t_start, t_end, dt = 0.0, 5e-9, 1e-9
        track_nodes = [self.track_node] if self.track_node else []

        # Standard solve
        result_std = self.solver.solve_transient(
            t_start=t_start,
            t_end=t_end,
            dt=dt,
            method=IntegrationMethod.BACKWARD_EULER,
            track_nodes=track_nodes,
        )

        # Multi-RHS with single all-True mask
        mask_all = np.ones(self.n_sources, dtype=bool)
        results_multi = self.solver.solve_transient_multi_rhs(
            t_start=t_start,
            t_end=t_end,
            dt=dt,
            source_masks=mask_all[np.newaxis, :],  # (1, n_sources)
            method=IntegrationMethod.BACKWARD_EULER,
            track_nodes=track_nodes,
        )

        self.assertEqual(len(results_multi), 1)
        result_multi = results_multi[0]

        # Compare peak IR-drop
        self.assertAlmostEqual(
            result_std.peak_ir_drop,
            result_multi.peak_ir_drop,
            places=10,
            msg="Peak IR-drop should match"
        )

        # Compare max_ir_drop_per_time
        np.testing.assert_allclose(
            result_std.max_ir_drop_per_time,
            result_multi.max_ir_drop_per_time,
            rtol=1e-10,
            err_msg="max_ir_drop_per_time should match"
        )

        # Compare tracked waveforms if available
        if track_nodes:
            for node in track_nodes:
                if node in result_std.tracked_ir_drop and node in result_multi.tracked_ir_drop:
                    np.testing.assert_allclose(
                        result_std.tracked_ir_drop[node],
                        result_multi.tracked_ir_drop[node],
                        rtol=1e-10,
                        err_msg=f"Tracked IR-drop for {node} should match"
                    )


class TestMultipleMasks(TestTransientMultiRHSBase):
    """Test multiple masks are processed correctly."""

    def test_multiple_masks_independent(self):
        """Multiple masks should produce independent results."""
        t_start, t_end, dt = 0.0, 5e-9, 1e-9
        track_nodes = [self.track_node] if self.track_node else []

        # Create three different masks
        n = self.n_sources
        mask_all = np.ones(n, dtype=bool)
        mask_half = np.zeros(n, dtype=bool)
        mask_half[:n//2] = True
        mask_other = ~mask_half

        source_masks = np.stack([mask_all, mask_half, mask_other])  # (3, n)

        results = self.solver.solve_transient_multi_rhs(
            t_start=t_start,
            t_end=t_end,
            dt=dt,
            source_masks=source_masks,
            method=IntegrationMethod.BACKWARD_EULER,
            track_nodes=track_nodes,
        )

        self.assertEqual(len(results), 3)

        # Each result should have same time array
        for result in results:
            self.assertEqual(len(result.t_array), len(results[0].t_array))

        # Half + other should approximately equal all (if currents are additive)
        # This validates correct mask application
        if track_nodes and self.track_node in results[0].tracked_ir_drop:
            total = results[0].get_ir_drop_waveform(self.track_node)
            half = results[1].get_ir_drop_waveform(self.track_node)
            other = results[2].get_ir_drop_waveform(self.track_node)

            # Due to linearity, half + other should equal total
            np.testing.assert_allclose(
                total, half + other,
                rtol=1e-8,
                err_msg="half + other should equal total (linearity)"
            )


class TestLinearitySuperposition(TestTransientMultiRHSBase):
    """Test linearity: V_total = V_near + V_far."""

    def test_superposition_near_plus_far_equals_total(self):
        """Verify linearity: V_total ≈ V_near + V_far."""
        t_start, t_end, dt = 0.0, 10e-9, 1e-9
        track_nodes = [self.track_node] if self.track_node else []

        if not track_nodes:
            self.skipTest("No track node available")

        # Create near/far split (arbitrary: first 30% vs rest)
        n = self.n_sources
        split = max(1, n // 3)

        mask_all = np.ones(n, dtype=bool)
        mask_near = np.zeros(n, dtype=bool)
        mask_near[:split] = True
        mask_far = ~mask_near

        source_masks = np.stack([mask_all, mask_near, mask_far])

        results = self.solver.solve_transient_multi_rhs(
            t_start=t_start,
            t_end=t_end,
            dt=dt,
            source_masks=source_masks,
            method=IntegrationMethod.BACKWARD_EULER,
            track_nodes=track_nodes,
        )

        total = results[0].get_ir_drop_waveform(self.track_node)
        near = results[1].get_ir_drop_waveform(self.track_node)
        far = results[2].get_ir_drop_waveform(self.track_node)

        # Verify superposition
        np.testing.assert_allclose(
            total, near + far,
            rtol=1e-8,
            err_msg="Superposition: total should equal near + far"
        )


class TestEmptyMask(TestTransientMultiRHSBase):
    """Test empty mask (all False) returns zero IR-drop."""

    def test_empty_mask_returns_zero_ir_drop(self):
        """All-False mask should produce zero IR-drop."""
        t_start, t_end, dt = 0.0, 5e-9, 1e-9
        track_nodes = [self.track_node] if self.track_node else []

        if not track_nodes:
            self.skipTest("No track node available")

        mask_empty = np.zeros(self.n_sources, dtype=bool)

        results = self.solver.solve_transient_multi_rhs(
            t_start=t_start,
            t_end=t_end,
            dt=dt,
            source_masks=mask_empty[np.newaxis, :],  # (1, n_sources)
            method=IntegrationMethod.BACKWARD_EULER,
            track_nodes=track_nodes,
        )

        result = results[0]

        # Peak IR-drop should be essentially zero (within numerical precision)
        self.assertLess(result.peak_ir_drop, 1e-10,
                        f"Peak IR-drop should be ~0 but got {result.peak_ir_drop}")

        # Tracked waveform should be all essentially zeros
        if self.track_node in result.tracked_ir_drop:
            ir_drop = result.tracked_ir_drop[self.track_node]
            np.testing.assert_allclose(
                ir_drop, np.zeros_like(ir_drop),
                atol=1e-10,
                err_msg="Empty mask should produce essentially zero IR-drop"
            )


class TestIntegrationMethods(TestTransientMultiRHSBase):
    """Test both integration methods work correctly."""

    def test_backward_euler_produces_valid_results(self):
        """Backward Euler should produce valid results."""
        t_start, t_end, dt = 0.0, 10e-9, 1e-9
        mask_all = np.ones(self.n_sources, dtype=bool)

        results = self.solver.solve_transient_multi_rhs(
            t_start=t_start,
            t_end=t_end,
            dt=dt,
            source_masks=mask_all[np.newaxis, :],
            method=IntegrationMethod.BACKWARD_EULER,
        )

        result = results[0]
        self.assertEqual(result.integration_method, IntegrationMethod.BACKWARD_EULER)
        self.assertGreater(len(result.t_array), 0)
        self.assertTrue(np.all(result.max_ir_drop_per_time >= 0))

    def test_trapezoidal_produces_valid_results(self):
        """Trapezoidal should produce valid results."""
        t_start, t_end, dt = 0.0, 10e-9, 0.5e-9  # Smaller dt for stability
        mask_all = np.ones(self.n_sources, dtype=bool)

        results = self.solver.solve_transient_multi_rhs(
            t_start=t_start,
            t_end=t_end,
            dt=dt,
            source_masks=mask_all[np.newaxis, :],
            method=IntegrationMethod.TRAPEZOIDAL,
        )

        result = results[0]
        self.assertEqual(result.integration_method, IntegrationMethod.TRAPEZOIDAL)
        self.assertGreater(len(result.t_array), 0)

    def test_be_vs_trap_similar_results(self):
        """Both methods should converge to similar results with small dt.

        Uses t_end=1ns and dt=10ps to ensure both methods are well-resolved
        and produce similar results (within 10%).
        """
        t_start, t_end, dt = 0.0, 1e-9, 10e-12  # 1ns with 10ps steps
        mask_all = np.ones(self.n_sources, dtype=bool)

        results_be = self.solver.solve_transient_multi_rhs(
            t_start=t_start,
            t_end=t_end,
            dt=dt,
            source_masks=mask_all[np.newaxis, :],
            method=IntegrationMethod.BACKWARD_EULER,
        )

        results_trap = self.solver.solve_transient_multi_rhs(
            t_start=t_start,
            t_end=t_end,
            dt=dt,
            source_masks=mask_all[np.newaxis, :],
            method=IntegrationMethod.TRAPEZOIDAL,
        )

        # With small dt, results should be similar (within 10%)
        be_peak = results_be[0].peak_ir_drop
        trap_peak = results_trap[0].peak_ir_drop

        if be_peak > 0:
            rel_diff = abs(be_peak - trap_peak) / be_peak
            self.assertLessEqual(rel_diff, 0.1,
                                 f"BE and TRAP should give similar results: "
                                 f"BE={be_peak*1000:.4f}mV, TRAP={trap_peak*1000:.4f}mV, diff={rel_diff*100:.1f}%")


class TestMultipleTrackedNodes(TestTransientMultiRHSBase):
    """Test tracking multiple nodes across all masks."""

    def test_multiple_tracked_nodes(self):
        """Should correctly track multiple nodes across all masks."""
        # Get multiple track nodes
        track_nodes = []
        for name, src in self.current_sources.items():
            node = getattr(src, 'node1', None)
            if node and node != '0' and node not in track_nodes:
                track_nodes.append(node)
            if len(track_nodes) >= 3:
                break

        if len(track_nodes) < 2:
            self.skipTest("Not enough track nodes available")

        t_start, t_end, dt = 0.0, 5e-9, 1e-9
        n = self.n_sources
        mask_all = np.ones(n, dtype=bool)
        mask_half = np.zeros(n, dtype=bool)
        mask_half[:n//2] = True

        source_masks = np.stack([mask_all, mask_half])

        results = self.solver.solve_transient_multi_rhs(
            t_start=t_start,
            t_end=t_end,
            dt=dt,
            source_masks=source_masks,
            method=IntegrationMethod.BACKWARD_EULER,
            track_nodes=track_nodes,
        )

        n_steps = len(results[0].t_array)

        for result in results:
            for node in track_nodes:
                # Check node is tracked
                self.assertIn(node, result.tracked_ir_drop)
                self.assertIn(node, result.tracked_waveforms)

                # Check waveform length
                self.assertEqual(len(result.tracked_ir_drop[node]), n_steps)
                self.assertEqual(len(result.tracked_waveforms[node]), n_steps)


class TestResultMetadata(TestTransientMultiRHSBase):
    """Test result metadata is correct for each mask."""

    def test_result_metadata_per_mask(self):
        """Each result should have correct metadata."""
        t_start, t_end, dt = 0.0, 10e-9, 1e-9
        n = self.n_sources

        mask_all = np.ones(n, dtype=bool)
        mask_half = np.zeros(n, dtype=bool)
        mask_half[:n//2] = True

        source_masks = np.stack([mask_all, mask_half])

        results = self.solver.solve_transient_multi_rhs(
            t_start=t_start,
            t_end=t_end,
            dt=dt,
            source_masks=source_masks,
            method=IntegrationMethod.BACKWARD_EULER,
        )

        for result in results:
            # Time array
            self.assertGreater(len(result.t_array), 0)
            self.assertAlmostEqual(result.t_array[0], t_start)
            self.assertAlmostEqual(result.t_array[-1], t_end)

            # Peak info
            self.assertIsNotNone(result.peak_ir_drop)
            self.assertIsNotNone(result.peak_ir_drop_time)

            # Arrays have correct length
            n_steps = len(result.t_array)
            self.assertEqual(len(result.max_ir_drop_per_time), n_steps)
            self.assertEqual(len(result.total_current_per_time), n_steps)
            self.assertEqual(len(result.total_vsrc_current_per_time), n_steps)


class TestMaskShapeValidation(TestTransientMultiRHSBase):
    """Test that invalid mask shapes raise errors."""

    def test_wrong_n_sources_raises_error(self):
        """Mask with wrong number of sources should raise ValueError."""
        t_start, t_end, dt = 0.0, 5e-9, 1e-9

        # Wrong number of sources
        bad_mask = np.ones((1, self.n_sources + 1), dtype=bool)

        with self.assertRaises(ValueError):
            self.solver.solve_transient_multi_rhs(
                t_start=t_start,
                t_end=t_end,
                dt=dt,
                source_masks=bad_mask,
                method=IntegrationMethod.BACKWARD_EULER,
            )


class TestEdgeCases(TestTransientMultiRHSBase):
    """Test edge cases."""

    def test_single_time_step(self):
        """Should handle single time step."""
        t_start, t_end, dt = 0.0, 1e-9, 1e-9
        mask_all = np.ones(self.n_sources, dtype=bool)

        results = self.solver.solve_transient_multi_rhs(
            t_start=t_start,
            t_end=t_end,
            dt=dt,
            source_masks=mask_all[np.newaxis, :],
            method=IntegrationMethod.BACKWARD_EULER,
        )

        self.assertEqual(len(results), 1)
        self.assertGreaterEqual(len(results[0].t_array), 1)

    def test_many_masks(self):
        """Should handle many masks efficiently."""
        t_start, t_end, dt = 0.0, 5e-9, 1e-9
        n = self.n_sources

        # Create 10 random masks
        np.random.seed(42)
        n_masks = 10
        source_masks = np.random.rand(n_masks, n) > 0.5

        results = self.solver.solve_transient_multi_rhs(
            t_start=t_start,
            t_end=t_end,
            dt=dt,
            source_masks=source_masks,
            method=IntegrationMethod.BACKWARD_EULER,
        )

        self.assertEqual(len(results), n_masks)

        # Each should have valid results
        for result in results:
            self.assertGreater(len(result.t_array), 0)


class TestCurrentTracking(TestTransientMultiRHSBase):
    """Test total current tracking with masks."""

    def test_total_current_reflects_mask(self):
        """Total current should reflect which sources are masked in."""
        t_start, t_end, dt = 0.0, 5e-9, 1e-9
        n = self.n_sources

        mask_all = np.ones(n, dtype=bool)
        mask_half = np.zeros(n, dtype=bool)
        mask_half[:n//2] = True

        source_masks = np.stack([mask_all, mask_half])

        results = self.solver.solve_transient_multi_rhs(
            t_start=t_start,
            t_end=t_end,
            dt=dt,
            source_masks=source_masks,
            method=IntegrationMethod.BACKWARD_EULER,
        )

        result_all = results[0]
        result_half = results[1]

        # Total current with half mask should be less than or equal to all mask
        # (assuming positive currents)
        for i in range(len(result_all.t_array)):
            curr_all = abs(result_all.total_current_per_time[i])
            curr_half = abs(result_half.total_current_per_time[i])
            # Half should have less or equal current (allowing small numerical tolerance)
            self.assertLessEqual(
                curr_half, curr_all + 1e-10,
                f"Half mask should have <= current at t[{i}]"
            )


class TestSharedNodeSourceMasking(unittest.TestCase):
    """Test that shared-node sources are correctly masked at the source level.

    This test validates the bug fix for per-source masking: before the fix,
    solve_transient_multi_rhs would produce incorrect results when multiple
    sources shared a node because it was masking at the node level (after
    aggregation) instead of at the source level (before aggregation).

    The test creates synthetic current sources with multiple sources on the
    same node, then verifies that solve_transient() and solve_transient_multi_rhs()
    (with an all-True mask) produce identical results.
    """

    @classmethod
    def setUpClass(cls):
        """Create a synthetic test scenario with shared-node sources."""
        from pdn.pdn_parser import CurrentSource, Pulse, PWL

        # Build a minimal PDN graph with shared-node current sources
        if USE_SMALL_NETLIST:
            test_netlist = Path(__file__).parent.parent / 'pdn' / 'netlist_small'
            net_name = 'VDD_XLV'
        else:
            test_netlist = Path(__file__).parent.parent / 'pdn' / 'netlist_test'
            net_name = 'VDD'

        if not test_netlist.exists():
            cls.graph = None
            cls.model = None
            return

        from pdn.pdn_parser import NetlistParser
        parser = NetlistParser(str(test_netlist))
        cls.graph = parser.parse()
        cls.model = create_model_from_pdn(cls.graph, net_name)

        # Get existing current sources to find valid nodes
        graph_dict = cls.graph.graph if hasattr(cls.graph, 'graph') else cls.graph._attrs
        original_sources = graph_dict.get('_instance_sources_objects', {})

        if not original_sources:
            cls.graph = None
            cls.model = None
            return

        # Find a valid node that has a current source
        shared_node = None
        for name, src in original_sources.items():
            node = getattr(src, 'node1', None)
            if node and node != '0':
                shared_node = node
                break

        if shared_node is None:
            cls.graph = None
            cls.model = None
            return

        cls.shared_node = shared_node

        # Create synthetic sources with multiple sources on the same node
        # Include DC, pulse, and PWL to exercise all code paths
        synthetic_sources = {}

        # Source 1: DC only on shared node
        src1 = CurrentSource(
            name='I_shared_dc1',
            node1=shared_node,
            node2='0',
            dc_value=0.5,  # 0.5 mA
        )
        synthetic_sources['I_shared_dc1'] = src1

        # Source 2: DC + pulse on same shared node
        pulse = Pulse(
            v1=0.0,
            v2=1.0,  # 1 mA peak
            delay=2e-9,
            rt=0.5e-9,
            ft=0.5e-9,
            width=3e-9,
            period=10e-9,
        )
        src2 = CurrentSource(
            name='I_shared_pulse',
            node1=shared_node,
            node2='0',
            dc_value=0.2,
            pulses=[pulse],
        )
        synthetic_sources['I_shared_pulse'] = src2

        # Source 3: PWL on same shared node
        pwl = PWL(
            delay=0.0,
            period=0.0,  # Non-periodic
            points=[(0.0, 0.0), (3e-9, 0.8), (6e-9, 0.3), (10e-9, 0.3)],
        )
        src3 = CurrentSource(
            name='I_shared_pwl',
            node1=shared_node,
            node2='0',
            dc_value=0.0,
            pwls=[pwl],
        )
        synthetic_sources['I_shared_pwl'] = src3

        # Replace original sources with synthetic sources
        graph_dict['_instance_sources_objects'] = synthetic_sources

        cls.n_sources = len(synthetic_sources)
        cls.source_names = list(synthetic_sources.keys())
        cls.track_node = shared_node

    def setUp(self):
        """Skip if test setup failed."""
        if self.model is None:
            self.skipTest("Test setup failed - no valid sources or nodes")
        # Use vectorize_threshold=0 to force vectorization
        self.solver = TransientIRDropSolver(self.model, self.graph, vectorize_threshold=0)

    def test_shared_node_sources_match_standard_solve(self):
        """All-True mask in multi_rhs should match standard solve_transient().

        This is the key test for the bug fix: before the fix, when multiple
        sources shared a node, the multi_rhs solver would incorrectly mask
        at the node level (after aggregation), producing different results
        from the standard solver. After the fix, both should be identical.
        """
        t_start, t_end, dt = 0.0, 8e-9, 0.5e-9
        track_nodes = [self.track_node]

        # Standard solve (unaffected by the bug)
        result_std = self.solver.solve_transient(
            t_start=t_start,
            t_end=t_end,
            dt=dt,
            method=IntegrationMethod.BACKWARD_EULER,
            track_nodes=track_nodes,
        )

        # Multi-RHS with all-True mask (this was buggy before the fix)
        mask_all = np.ones(self.n_sources, dtype=bool)
        results_multi = self.solver.solve_transient_multi_rhs(
            t_start=t_start,
            t_end=t_end,
            dt=dt,
            source_masks=mask_all[np.newaxis, :],  # (1, n_sources)
            method=IntegrationMethod.BACKWARD_EULER,
            track_nodes=track_nodes,
        )

        self.assertEqual(len(results_multi), 1)
        result_multi = results_multi[0]

        # Compare peak IR-drop - should be identical
        self.assertAlmostEqual(
            result_std.peak_ir_drop,
            result_multi.peak_ir_drop,
            places=10,
            msg=f"Peak IR-drop mismatch: std={result_std.peak_ir_drop*1000:.6f}mV, "
                f"multi={result_multi.peak_ir_drop*1000:.6f}mV"
        )

        # Compare max_ir_drop_per_time arrays
        np.testing.assert_allclose(
            result_std.max_ir_drop_per_time,
            result_multi.max_ir_drop_per_time,
            rtol=1e-10,
            err_msg="max_ir_drop_per_time should be identical"
        )

        # Compare total current per time
        np.testing.assert_allclose(
            result_std.total_current_per_time,
            result_multi.total_current_per_time,
            rtol=1e-10,
            err_msg="total_current_per_time should be identical"
        )

        # Compare tracked waveforms for the shared node
        if self.track_node in result_std.tracked_ir_drop:
            np.testing.assert_allclose(
                result_std.tracked_ir_drop[self.track_node],
                result_multi.tracked_ir_drop[self.track_node],
                rtol=1e-10,
                err_msg=f"Tracked IR-drop for shared node {self.track_node} should match"
            )

    def test_selective_mask_excludes_sources_correctly(self):
        """Masking out some sources on a shared node should reduce IR-drop.

        When we mask out some sources on a shared node, the IR-drop should
        decrease (since less current is being drawn). This validates that
        masking is applied per-source, not per-node.
        """
        t_start, t_end, dt = 0.0, 8e-9, 0.5e-9

        # All sources mask
        mask_all = np.ones(self.n_sources, dtype=bool)

        # Only first source mask
        mask_first_only = np.zeros(self.n_sources, dtype=bool)
        mask_first_only[0] = True

        source_masks = np.stack([mask_all, mask_first_only])

        results = self.solver.solve_transient_multi_rhs(
            t_start=t_start,
            t_end=t_end,
            dt=dt,
            source_masks=source_masks,
            method=IntegrationMethod.BACKWARD_EULER,
        )

        result_all = results[0]
        result_first = results[1]

        # With only one source (instead of three on the same node),
        # the IR-drop should be smaller
        self.assertLess(
            result_first.peak_ir_drop,
            result_all.peak_ir_drop,
            msg=f"Single source should have lower IR-drop than all sources: "
                f"first={result_first.peak_ir_drop*1000:.4f}mV, "
                f"all={result_all.peak_ir_drop*1000:.4f}mV"
        )

        # Total current should also be lower
        for i in range(len(result_all.t_array)):
            self.assertLessEqual(
                abs(result_first.total_current_per_time[i]),
                abs(result_all.total_current_per_time[i]) + 1e-10,
                f"Single source should have <= current at t[{i}]"
            )

    def test_superposition_with_shared_nodes(self):
        """Verify superposition holds when sources share a node.

        For a linear system, V_total = V_src1 + V_src2 + V_src3.
        This validates that masking is correctly applied to each source
        independently, even when they share a node.
        """
        t_start, t_end, dt = 0.0, 8e-9, 0.5e-9
        track_nodes = [self.track_node]

        # Create individual masks for each source
        masks = []
        for i in range(self.n_sources):
            mask = np.zeros(self.n_sources, dtype=bool)
            mask[i] = True
            masks.append(mask)

        # Add all-sources mask
        mask_all = np.ones(self.n_sources, dtype=bool)
        masks.append(mask_all)

        source_masks = np.stack(masks)

        results = self.solver.solve_transient_multi_rhs(
            t_start=t_start,
            t_end=t_end,
            dt=dt,
            source_masks=source_masks,
            method=IntegrationMethod.BACKWARD_EULER,
            track_nodes=track_nodes,
        )

        # Results: [src0_only, src1_only, src2_only, all_sources]
        result_all = results[-1]

        # Sum of individual source contributions
        summed_ir_drop = np.zeros_like(result_all.tracked_ir_drop[self.track_node])
        for i in range(self.n_sources):
            if self.track_node in results[i].tracked_ir_drop:
                summed_ir_drop += results[i].tracked_ir_drop[self.track_node]

        # Verify superposition: sum of individuals ≈ all together
        np.testing.assert_allclose(
            summed_ir_drop,
            result_all.tracked_ir_drop[self.track_node],
            rtol=1e-8,
            err_msg="Superposition should hold: sum of individual sources = all sources"
        )


if __name__ == '__main__':
    unittest.main()

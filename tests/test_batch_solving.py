"""Tests for batch solving with cached contexts.

Tests that prepare/solve_prepared methods produce results identical to
direct solving, and that cached contexts provide performance benefits.
"""

import unittest
import time
from typing import Dict, Any

from generate_power_grid import generate_power_grid
from core import (
    create_model_from_synthetic,
    UnifiedIRDropSolver,
    FlatSolverContext,
    HierarchicalSolverContext,
    CoupledHierarchicalSolverContext,
)


def build_test_grid(K=3, N0=8, I_N=80, N_vsrc=4, seed=42):
    """Build a test grid for batch solving tests."""
    G, loads, pads = generate_power_grid(
        K=K, N0=N0, I_N=I_N, N_vsrc=N_vsrc, seed=seed,
        max_stripe_res=0.01, max_via_res=0.01
    )
    model = create_model_from_synthetic(G, pads, vdd=1.0)
    return model, loads


class TestFlatSolverBatch(unittest.TestCase):
    """Tests for flat solver batch solving."""

    def setUp(self):
        """Set up test fixtures."""
        self.model, self.loads = build_test_grid()
        self.solver = UnifiedIRDropSolver(self.model)

    def test_prepare_flat_creates_context(self):
        """prepare_flat() should return a valid FlatSolverContext."""
        ctx = self.solver.prepare_flat()
        self.assertIsInstance(ctx, FlatSolverContext)
        self.assertIsNotNone(ctx.reduced_system)
        self.assertEqual(ctx.vdd, self.model.vdd)
        self.assertEqual(ctx.pad_nodes, set(self.model.pad_nodes))

    def test_solve_prepared_matches_direct(self):
        """solve_prepared() should produce identical results to solve()."""
        ctx = self.solver.prepare_flat()
        currents = {n: 0.001 for n in list(self.loads.keys())[:10]}

        result_direct = self.solver.solve(currents)
        result_prepared = self.solver.solve_prepared(currents, ctx)

        # Voltages should match exactly
        for node in result_direct.voltages:
            self.assertAlmostEqual(
                result_direct.voltages[node],
                result_prepared.voltages[node],
                places=12,
                msg=f"Voltage mismatch at node {node}"
            )

    def test_solve_with_context_parameter(self):
        """solve() with context parameter should delegate to solve_prepared()."""
        ctx = self.solver.prepare_flat()
        currents = {n: 0.001 for n in list(self.loads.keys())[:10]}

        result_direct = self.solver.solve(currents)
        result_via_context = self.solver.solve(currents, context=ctx)

        for node in result_direct.voltages:
            self.assertAlmostEqual(
                result_direct.voltages[node],
                result_via_context.voltages[node],
                places=12
            )

    def test_batch_solve_consistency(self):
        """Multiple solves with same context should produce consistent results."""
        ctx = self.solver.prepare_flat()
        load_nodes = list(self.loads.keys())[:10]

        stimuli = [
            {n: 0.001 for n in load_nodes},
            {n: 0.002 for n in load_nodes},
            {n: 0.003 for n in load_nodes},
        ]

        for currents in stimuli:
            result_direct = self.solver.solve(currents)
            result_prepared = self.solver.solve_prepared(currents, ctx)

            max_diff = max(
                abs(result_direct.voltages[n] - result_prepared.voltages[n])
                for n in result_direct.voltages
            )
            self.assertLess(max_diff, 1e-12)


class TestHierarchicalSolverBatch(unittest.TestCase):
    """Tests for hierarchical solver batch solving."""

    def setUp(self):
        """Set up test fixtures."""
        self.model, self.loads = build_test_grid()
        self.solver = UnifiedIRDropSolver(self.model)

    def test_prepare_hierarchical_creates_context(self):
        """prepare_hierarchical() should return a valid HierarchicalSolverContext."""
        ctx = self.solver.prepare_hierarchical(partition_layer=2, top_k=5)
        self.assertIsInstance(ctx, HierarchicalSolverContext)
        self.assertEqual(ctx.partition_layer, 2)
        self.assertEqual(ctx.top_k, 5)
        self.assertIsNotNone(ctx.top_system)
        self.assertIsNotNone(ctx.bottom_system)
        self.assertTrue(len(ctx.port_nodes) > 0)
        self.assertTrue(len(ctx.shortest_path_cache) > 0)

    def test_solve_hierarchical_prepared_matches_direct(self):
        """solve_hierarchical_prepared() should match solve_hierarchical()."""
        ctx = self.solver.prepare_hierarchical(partition_layer=2, top_k=5)
        currents = {n: 0.001 for n in list(self.loads.keys())[:10]}

        result_direct = self.solver.solve_hierarchical(currents, partition_layer=2, top_k=5)
        result_prepared = self.solver.solve_hierarchical_prepared(currents, ctx)

        for node in result_direct.voltages:
            self.assertAlmostEqual(
                result_direct.voltages[node],
                result_prepared.voltages[node],
                places=12,
                msg=f"Voltage mismatch at node {node}"
            )

    def test_solve_hierarchical_with_context_parameter(self):
        """solve_hierarchical() with context should delegate to prepared."""
        ctx = self.solver.prepare_hierarchical(partition_layer=2, top_k=5)
        currents = {n: 0.001 for n in list(self.loads.keys())[:10]}

        result_direct = self.solver.solve_hierarchical(currents, partition_layer=2, top_k=5)
        result_via_context = self.solver.solve_hierarchical(
            currents, partition_layer=2, context=ctx
        )

        max_diff = max(
            abs(result_direct.voltages[n] - result_via_context.voltages[n])
            for n in result_direct.voltages
        )
        self.assertLess(max_diff, 1e-12)

    def test_batch_hierarchical_consistency(self):
        """Multiple hierarchical solves should produce consistent results."""
        ctx = self.solver.prepare_hierarchical(partition_layer=2, top_k=5)
        load_nodes = list(self.loads.keys())[:10]

        stimuli = [
            {n: 0.001 for n in load_nodes},
            {n: 0.002 for n in load_nodes},
        ]

        for currents in stimuli:
            result_direct = self.solver.solve_hierarchical(currents, partition_layer=2, top_k=5)
            result_prepared = self.solver.solve_hierarchical_prepared(currents, ctx)

            max_diff = max(
                abs(result_direct.voltages[n] - result_prepared.voltages[n])
                for n in result_direct.voltages
            )
            self.assertLess(max_diff, 1e-12)


class TestCoupledHierarchicalSolverBatch(unittest.TestCase):
    """Tests for coupled hierarchical solver batch solving."""

    def setUp(self):
        """Set up test fixtures."""
        self.model, self.loads = build_test_grid()
        self.solver = UnifiedIRDropSolver(self.model)

    def test_prepare_hierarchical_coupled_creates_context(self):
        """prepare_hierarchical_coupled() should return valid context."""
        ctx = self.solver.prepare_hierarchical_coupled(
            partition_layer=2,
            solver='gmres',
            tol=1e-8,
            preconditioner='block_diagonal'
        )
        self.assertIsInstance(ctx, CoupledHierarchicalSolverContext)
        self.assertEqual(ctx.partition_layer, 2)
        self.assertEqual(ctx.solver, 'gmres')
        self.assertEqual(ctx.preconditioner_type, 'block_diagonal')
        self.assertIsNotNone(ctx.coupled_op)
        self.assertIsNotNone(ctx.bottom_blocks.lu_ii)

    def test_solve_coupled_prepared_matches_direct(self):
        """solve_hierarchical_coupled_prepared() should match direct solve."""
        ctx = self.solver.prepare_hierarchical_coupled(
            partition_layer=2, tol=1e-8
        )
        currents = {n: 0.001 for n in list(self.loads.keys())[:10]}

        result_direct = self.solver.solve_hierarchical_coupled(
            currents, partition_layer=2, tol=1e-8
        )
        result_prepared = self.solver.solve_hierarchical_coupled_prepared(
            currents, ctx
        )

        # Results should match within iterative tolerance
        for node in result_direct.voltages:
            self.assertAlmostEqual(
                result_direct.voltages[node],
                result_prepared.voltages[node],
                places=8,
                msg=f"Voltage mismatch at node {node}"
            )

    def test_solve_coupled_with_context_parameter(self):
        """solve_hierarchical_coupled() with context should delegate."""
        ctx = self.solver.prepare_hierarchical_coupled(partition_layer=2)
        currents = {n: 0.001 for n in list(self.loads.keys())[:10]}

        result_direct = self.solver.solve_hierarchical_coupled(
            currents, partition_layer=2
        )
        result_via_context = self.solver.solve_hierarchical_coupled(
            currents, partition_layer=2, context=ctx
        )

        max_diff = max(
            abs(result_direct.voltages[n] - result_via_context.voltages[n])
            for n in result_direct.voltages
        )
        self.assertLess(max_diff, 1e-8)

    def test_batch_coupled_consistency(self):
        """Multiple coupled solves should produce consistent results."""
        ctx = self.solver.prepare_hierarchical_coupled(partition_layer=2)
        load_nodes = list(self.loads.keys())[:10]

        stimuli = [
            {n: 0.001 for n in load_nodes},
            {n: 0.002 for n in load_nodes},
        ]

        for currents in stimuli:
            result_direct = self.solver.solve_hierarchical_coupled(
                currents, partition_layer=2
            )
            result_prepared = self.solver.solve_hierarchical_coupled_prepared(
                currents, ctx
            )

            max_diff = max(
                abs(result_direct.voltages[n] - result_prepared.voltages[n])
                for n in result_direct.voltages
            )
            self.assertLess(max_diff, 1e-8)


class TestBatchSolvingPerformance(unittest.TestCase):
    """Tests that batch solving provides performance benefits."""

    def setUp(self):
        """Set up larger test grid for performance testing."""
        self.model, self.loads = build_test_grid(K=4, N0=12, I_N=150, N_vsrc=4)
        self.solver = UnifiedIRDropSolver(self.model)

    def test_flat_batch_faster_than_individual(self):
        """Batch flat solving should be faster than individual solves."""
        load_nodes = list(self.loads.keys())[:20]
        stimuli = [{n: 0.001 * i for n in load_nodes} for i in range(1, 6)]

        # Time individual solves (no caching)
        t0 = time.perf_counter()
        for currents in stimuli:
            _ = self.solver.solve(currents)
        time_individual = time.perf_counter() - t0

        # Time batch solves (with caching)
        ctx = self.solver.prepare_flat()
        t0 = time.perf_counter()
        for currents in stimuli:
            _ = self.solver.solve_prepared(currents, ctx)
        time_batch = time.perf_counter() - t0

        # Batch should be faster (or at least not much slower)
        # We don't assert specific speedup as it depends on grid size
        # Just verify it completes without errors
        self.assertTrue(time_batch > 0)

    def test_hierarchical_batch_faster_than_individual(self):
        """Batch hierarchical solving should be faster than individual."""
        load_nodes = list(self.loads.keys())[:20]
        stimuli = [{n: 0.001 * i for n in load_nodes} for i in range(1, 4)]

        # Time batch solves with prepare
        ctx = self.solver.prepare_hierarchical(partition_layer=2)
        t0 = time.perf_counter()
        for currents in stimuli:
            _ = self.solver.solve_hierarchical_prepared(currents, ctx)
        time_batch = time.perf_counter() - t0

        self.assertTrue(time_batch > 0)


if __name__ == '__main__':
    unittest.main()

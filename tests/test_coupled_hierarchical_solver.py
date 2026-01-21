"""Unit tests for coupled hierarchical IR-drop solver.

Tests the solve_hierarchical_coupled() method which solves the coupled
top-grid + bottom-grid system exactly (up to iterative tolerance) using
a matrix-free Schur complement approach.
"""

import math
import signal
import unittest
from functools import wraps

import numpy as np

from generate_power_grid import generate_power_grid, NodeID


def timeout(seconds):
    """Decorator to add timeout to test methods (Unix only)."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            def handler(signum, frame):
                raise TimeoutError(f"Test {func.__name__} timed out after {seconds}s")
            old_handler = signal.signal(signal.SIGALRM, handler)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
            return result
        return wrapper
    return decorator


def build_small_grid():
    """Build a small test grid with 3 layers."""
    G, loads, pads = generate_power_grid(
        K=3, N0=8, I_N=10, N_vsrc=2,
        max_stripe_res=1.0, max_via_res=0.1,
        load_current=0.1, seed=42, plot=False
    )
    return G, loads, pads


def build_medium_grid():
    """Build a medium test grid with 5 layers."""
    G, loads, pads = generate_power_grid(
        K=5, N0=8, I_N=15, N_vsrc=4,
        max_stripe_res=1.0, max_via_res=0.1,
        load_current=0.1, seed=42, plot=False
    )
    return G, loads, pads


class TestCoupledHierarchicalSolverSynthetic(unittest.TestCase):
    """Tests for solve_hierarchical_coupled with synthetic grids."""

    def test_basic_coupled_solve(self):
        """Test that coupled solver completes without error."""
        from core import create_model_from_synthetic, UnifiedIRDropSolver

        G, loads, pads = build_small_grid()
        model = create_model_from_synthetic(G, pads, vdd=1.0)
        solver = UnifiedIRDropSolver(model)

        result = solver.solve_hierarchical_coupled(
            current_injections=loads,
            partition_layer=1,
            solver='gmres',
            tol=1e-8,
            preconditioner='block_diagonal',
        )

        self.assertTrue(result.converged)
        self.assertGreater(len(result.voltages), 0)
        self.assertGreater(result.iterations, 0)

    def test_coupled_matches_flat_tightly(self):
        """Coupled solution should match flat solve within tolerance."""
        from core import create_model_from_synthetic, UnifiedIRDropSolver

        G, loads, pads = build_small_grid()
        model = create_model_from_synthetic(G, pads, vdd=1.0)
        solver = UnifiedIRDropSolver(model)

        # Flat solve (ground truth)
        flat_result = solver.solve(loads)

        # Coupled hierarchical solve
        coupled_result = solver.solve_hierarchical_coupled(
            current_injections=loads,
            partition_layer=1,
            tol=1e-10,
            maxiter=1000,
            preconditioner='block_diagonal',
        )

        # Compare voltages
        max_diff = 0.0
        for node in flat_result.voltages:
            if node in coupled_result.voltages:
                diff = abs(flat_result.voltages[node] - coupled_result.voltages[node])
                max_diff = max(max_diff, diff)

        # Should be nearly exact (within iterative tolerance)
        self.assertLess(max_diff, 1e-6, f"Max voltage difference {max_diff} exceeds 1e-6")

    def test_coupled_more_accurate_than_uncoupled(self):
        """Coupled solve should be more accurate than uncoupled hierarchical."""
        from core import create_model_from_synthetic, UnifiedIRDropSolver

        G, loads, pads = build_medium_grid()
        model = create_model_from_synthetic(G, pads, vdd=1.0)
        solver = UnifiedIRDropSolver(model)

        # Flat solve (ground truth)
        flat_result = solver.solve(loads)

        # Uncoupled hierarchical solve
        uncoupled_result = solver.solve_hierarchical(
            current_injections=loads,
            partition_layer=2,
            top_k=5,
            weighting='effective',
        )

        # Coupled hierarchical solve
        coupled_result = solver.solve_hierarchical_coupled(
            current_injections=loads,
            partition_layer=2,
            tol=1e-8,
            preconditioner='block_diagonal',
        )

        # Compute errors
        uncoupled_errors = []
        coupled_errors = []
        for node in flat_result.voltages:
            if node in uncoupled_result.voltages and node in coupled_result.voltages:
                uncoupled_errors.append(abs(flat_result.voltages[node] - uncoupled_result.voltages[node]))
                coupled_errors.append(abs(flat_result.voltages[node] - coupled_result.voltages[node]))

        uncoupled_max = max(uncoupled_errors)
        coupled_max = max(coupled_errors)

        # Coupled should be significantly more accurate
        self.assertLess(coupled_max, uncoupled_max,
                       f"Coupled max error {coupled_max} should be < uncoupled {uncoupled_max}")

    def test_different_solvers(self):
        """Test both GMRES and BiCGSTAB solvers."""
        from core import create_model_from_synthetic, UnifiedIRDropSolver

        G, loads, pads = build_small_grid()
        model = create_model_from_synthetic(G, pads, vdd=1.0)
        solver = UnifiedIRDropSolver(model)

        # GMRES
        result_gmres = solver.solve_hierarchical_coupled(
            current_injections=loads,
            partition_layer=1,
            solver='gmres',
            tol=1e-8,
        )
        self.assertTrue(result_gmres.converged)

        # BiCGSTAB
        result_bicgstab = solver.solve_hierarchical_coupled(
            current_injections=loads,
            partition_layer=1,
            solver='bicgstab',
            tol=1e-8,
        )
        self.assertTrue(result_bicgstab.converged)

        # Both should produce similar results
        max_diff = 0.0
        for node in result_gmres.voltages:
            if node in result_bicgstab.voltages:
                diff = abs(result_gmres.voltages[node] - result_bicgstab.voltages[node])
                max_diff = max(max_diff, diff)

        self.assertLess(max_diff, 1e-6)

    def test_different_preconditioners(self):
        """Test different preconditioner options."""
        from core import create_model_from_synthetic, UnifiedIRDropSolver

        G, loads, pads = build_small_grid()
        model = create_model_from_synthetic(G, pads, vdd=1.0)
        solver = UnifiedIRDropSolver(model)

        # No preconditioner
        result_none = solver.solve_hierarchical_coupled(
            current_injections=loads,
            partition_layer=1,
            preconditioner='none',
            tol=1e-8,
            maxiter=1000,
        )
        self.assertTrue(result_none.converged)
        self.assertEqual(result_none.preconditioner_type, 'none')

        # Block diagonal preconditioner
        result_bd = solver.solve_hierarchical_coupled(
            current_injections=loads,
            partition_layer=1,
            preconditioner='block_diagonal',
            tol=1e-8,
        )
        self.assertTrue(result_bd.converged)
        self.assertEqual(result_bd.preconditioner_type, 'block_diagonal')

        # Block diagonal should converge faster
        self.assertLessEqual(result_bd.iterations, result_none.iterations)

    def test_convergence_tolerance(self):
        """Verify solution improves with tighter tolerance."""
        from core import create_model_from_synthetic, UnifiedIRDropSolver

        G, loads, pads = build_small_grid()
        model = create_model_from_synthetic(G, pads, vdd=1.0)
        solver = UnifiedIRDropSolver(model)

        flat_result = solver.solve(loads)

        # Loose tolerance
        result_loose = solver.solve_hierarchical_coupled(
            current_injections=loads,
            partition_layer=1,
            tol=1e-4,
        )

        # Tight tolerance
        result_tight = solver.solve_hierarchical_coupled(
            current_injections=loads,
            partition_layer=1,
            tol=1e-10,
            maxiter=2000,
        )

        # Compute errors
        def max_error(result):
            errs = [abs(flat_result.voltages[n] - result.voltages[n])
                   for n in flat_result.voltages if n in result.voltages]
            return max(errs)

        loose_err = max_error(result_loose)
        tight_err = max_error(result_tight)

        self.assertLess(tight_err, loose_err,
                       f"Tight tol error {tight_err} should be < loose {loose_err}")

    def test_no_currents_all_at_vdd(self):
        """With no currents, all nodes should be at Vdd."""
        from core import create_model_from_synthetic, UnifiedIRDropSolver

        G, loads, pads = build_small_grid()
        model = create_model_from_synthetic(G, pads, vdd=1.0)
        solver = UnifiedIRDropSolver(model)

        result = solver.solve_hierarchical_coupled(
            current_injections={},
            partition_layer=1,
        )

        # Use places=6 since iterative solver has tolerance of 1e-8
        for v in result.voltages.values():
            self.assertAlmostEqual(v, 1.0, places=6)

    def test_result_structure(self):
        """Test that result has all expected fields."""
        from core import create_model_from_synthetic, UnifiedIRDropSolver
        from core import UnifiedCoupledHierarchicalResult

        G, loads, pads = build_small_grid()
        model = create_model_from_synthetic(G, pads, vdd=1.0)
        solver = UnifiedIRDropSolver(model)

        result = solver.solve_hierarchical_coupled(
            current_injections=loads,
            partition_layer=1,
        )

        # Check type
        self.assertIsInstance(result, UnifiedCoupledHierarchicalResult)

        # Check fields exist
        self.assertIsInstance(result.voltages, dict)
        self.assertIsInstance(result.ir_drop, dict)
        self.assertEqual(result.partition_layer, 1)
        self.assertIsInstance(result.top_grid_voltages, dict)
        self.assertIsInstance(result.bottom_grid_voltages, dict)
        self.assertIsInstance(result.port_nodes, set)
        self.assertIsInstance(result.port_voltages, dict)
        self.assertIsInstance(result.iterations, int)
        self.assertIsInstance(result.final_residual, float)
        self.assertIsInstance(result.converged, bool)
        self.assertIsInstance(result.preconditioner_type, str)
        self.assertIsInstance(result.timings, dict)

    def test_different_partition_layers(self):
        """Test with different partition layers."""
        from core import create_model_from_synthetic, UnifiedIRDropSolver

        G, loads, pads = build_medium_grid()
        model = create_model_from_synthetic(G, pads, vdd=1.0)
        solver = UnifiedIRDropSolver(model)

        for partition_layer in [1, 2, 3]:
            result = solver.solve_hierarchical_coupled(
                current_injections=loads,
                partition_layer=partition_layer,
            )

            self.assertTrue(result.converged)
            self.assertEqual(result.partition_layer, partition_layer)
            self.assertGreater(len(result.voltages), 0)

    def test_invalid_partition_layer_raises(self):
        """Invalid partition layer should raise ValueError."""
        from core import create_model_from_synthetic, UnifiedIRDropSolver

        G, loads, pads = build_small_grid()
        model = create_model_from_synthetic(G, pads, vdd=1.0)
        solver = UnifiedIRDropSolver(model)

        with self.assertRaises(ValueError):
            solver.solve_hierarchical_coupled(loads, partition_layer=0)

        with self.assertRaises(ValueError):
            solver.solve_hierarchical_coupled(loads, partition_layer=100)

    def test_invalid_solver_raises(self):
        """Invalid solver name should raise ValueError."""
        from core import create_model_from_synthetic, UnifiedIRDropSolver

        G, loads, pads = build_small_grid()
        model = create_model_from_synthetic(G, pads, vdd=1.0)
        solver = UnifiedIRDropSolver(model)

        with self.assertRaises(ValueError):
            solver.solve_hierarchical_coupled(
                loads, partition_layer=1, solver='invalid_solver'
            )

    def test_invalid_preconditioner_raises(self):
        """Invalid preconditioner name should raise ValueError."""
        from core import create_model_from_synthetic, UnifiedIRDropSolver

        G, loads, pads = build_small_grid()
        model = create_model_from_synthetic(G, pads, vdd=1.0)
        solver = UnifiedIRDropSolver(model)

        with self.assertRaises(ValueError):
            solver.solve_hierarchical_coupled(
                loads, partition_layer=1, preconditioner='invalid'
            )

    def test_non_convergence_raises_runtime_error(self):
        """Non-convergence should raise RuntimeError."""
        from core import create_model_from_synthetic, UnifiedIRDropSolver

        G, loads, pads = build_small_grid()
        model = create_model_from_synthetic(G, pads, vdd=1.0)
        solver = UnifiedIRDropSolver(model)

        # Use maxiter=1 to force non-convergence
        with self.assertRaises(RuntimeError) as ctx:
            solver.solve_hierarchical_coupled(
                current_injections=loads,
                partition_layer=1,
                maxiter=1,
                tol=1e-15,  # Very tight tolerance
            )

        self.assertIn('did not converge', str(ctx.exception))

    def test_verbose_output(self):
        """Test verbose mode prints timing info (no assertion, just coverage)."""
        from core import create_model_from_synthetic, UnifiedIRDropSolver
        import io
        import sys

        G, loads, pads = build_small_grid()
        model = create_model_from_synthetic(G, pads, vdd=1.0)
        solver = UnifiedIRDropSolver(model)

        # Capture stdout
        captured = io.StringIO()
        sys.stdout = captured

        try:
            result = solver.solve_hierarchical_coupled(
                current_injections=loads,
                partition_layer=1,
                verbose=True,
            )
        finally:
            sys.stdout = sys.__stdout__

        output = captured.getvalue()
        self.assertIn('Coupled Hierarchical Solve', output)
        self.assertIn('Iterations', output)


class TestSchurComplementOperator(unittest.TestCase):
    """Tests for the Schur complement operator."""

    def test_schur_operator_matches_explicit(self):
        """Schur complement operator should match explicit matrix formation."""
        from core import create_model_from_synthetic, UnifiedIRDropSolver
        from core.coupled_system import extract_block_matrices, SchurComplementOperator
        import scipy.sparse as sp
        import scipy.sparse.linalg as spla

        G, loads, pads = build_small_grid()
        model = create_model_from_synthetic(G, pads, vdd=1.0)

        # Decompose
        top_nodes, bottom_nodes, port_nodes, _ = model._decompose_at_layer(1)
        bottom_subgrid = bottom_nodes | port_nodes

        # Extract block matrices
        blocks, _ = extract_block_matrices(
            model=model,
            grid_nodes=bottom_subgrid,
            dirichlet_nodes=set(),
            port_nodes=port_nodes,
            dirichlet_voltage=1.0,
        )

        if blocks.n_interior == 0:
            self.skipTest("No interior nodes in bottom grid")

        # Factor interior
        blocks.factor_interior()

        # Create Schur complement operator
        schur_op = SchurComplementOperator(
            blocks.G_pp, blocks.G_pi, blocks.G_ip, blocks.lu_ii
        )

        # Compute explicit Schur complement
        # S = G_pp - G_pi * inv(G_ii) * G_ip
        G_ii_inv = spla.inv(blocks.G_ii.tocsc())
        S_explicit = blocks.G_pp - blocks.G_pi @ G_ii_inv @ blocks.G_ip

        # Test on random vectors
        np.random.seed(42)
        for _ in range(5):
            x = np.random.randn(blocks.n_ports)

            y_op = schur_op @ x
            y_explicit = S_explicit @ x

            np.testing.assert_allclose(y_op, y_explicit, rtol=1e-10)


class TestCoupledHierarchicalSolverPDN(unittest.TestCase):
    """Tests for solve_hierarchical_coupled with PDN graphs."""

    @classmethod
    def setUpClass(cls):
        """Load PDN netlist once for all tests."""
        import os
        import sys

        prototype_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sys.path.insert(0, prototype_root)

        netlist_path = os.path.join(prototype_root, 'pdn', 'netlist_small')
        if not os.path.exists(netlist_path):
            cls.model = None
            cls.load_currents = None
            cls.solver = None
            return

        try:
            from pdn.pdn_parser import NetlistParser
            from core import create_model_from_pdn, UnifiedIRDropSolver

            parser = NetlistParser(netlist_path, validate=False)
            graph = parser.parse()
            cls.model = create_model_from_pdn(graph, 'VDD_XLV')
            cls.load_currents = cls.model.extract_current_sources()
            cls.solver = UnifiedIRDropSolver(cls.model)
        except Exception:
            cls.model = None
            cls.load_currents = None
            cls.solver = None

    def setUp(self):
        """Skip tests if PDN netlist not available."""
        if self.model is None:
            self.skipTest("PDN netlist_small not available")

    @timeout(120)
    def test_pdn_basic_coupled_solve(self):
        """Test coupled solver on PDN graph."""
        result = self.solver.solve_hierarchical_coupled(
            current_injections=self.load_currents,
            partition_layer='M2',
            solver='gmres',
            tol=1e-8,
            preconditioner='block_diagonal',
        )

        self.assertTrue(result.converged)
        self.assertGreater(len(result.voltages), 0)

    @timeout(120)
    def test_pdn_coupled_matches_flat(self):
        """PDN coupled solution should match flat solve."""
        flat_result = self.solver.solve(self.load_currents)

        coupled_result = self.solver.solve_hierarchical_coupled(
            current_injections=self.load_currents,
            partition_layer='M2',
            tol=1e-10,
            maxiter=1000,
        )

        # Compare voltages
        errors = []
        for node in flat_result.voltages:
            if node in coupled_result.voltages:
                diff = abs(flat_result.voltages[node] - coupled_result.voltages[node])
                errors.append(diff)

        max_error = max(errors)
        mean_error = sum(errors) / len(errors)

        # Should match flat solve closely (up to iterative solver tolerance)
        self.assertLess(max_error, 1e-6, f"Max error {max_error} too large")
        self.assertLess(mean_error, 1e-6, f"Mean error {mean_error} too large")


class TestBlockMatrixExtraction(unittest.TestCase):
    """Tests for extract_block_matrices function."""

    def test_block_dimensions(self):
        """Test that block matrix dimensions are consistent."""
        from core import create_model_from_synthetic
        from core.coupled_system import extract_block_matrices

        G, loads, pads = build_small_grid()
        model = create_model_from_synthetic(G, pads, vdd=1.0)

        top_nodes, bottom_nodes, port_nodes, _ = model._decompose_at_layer(1)
        pad_set = set(pads) & top_nodes

        blocks, rhs = extract_block_matrices(
            model=model,
            grid_nodes=top_nodes,
            dirichlet_nodes=pad_set,
            port_nodes=port_nodes,
            dirichlet_voltage=1.0,
        )

        # Check dimensions
        self.assertEqual(blocks.G_pp.shape, (blocks.n_ports, blocks.n_ports))
        self.assertEqual(blocks.G_pi.shape, (blocks.n_ports, blocks.n_interior))
        self.assertEqual(blocks.G_ip.shape, (blocks.n_interior, blocks.n_ports))
        self.assertEqual(blocks.G_ii.shape, (blocks.n_interior, blocks.n_interior))

        # Check RHS dimension
        self.assertEqual(len(rhs), blocks.n_ports + blocks.n_interior)

    def test_node_ordering_consistency(self):
        """Test that node ordering is consistent across blocks."""
        from core import create_model_from_synthetic
        from core.coupled_system import extract_block_matrices

        G, loads, pads = build_small_grid()
        model = create_model_from_synthetic(G, pads, vdd=1.0)

        top_nodes, bottom_nodes, port_nodes, _ = model._decompose_at_layer(1)
        pad_set = set(pads) & top_nodes

        blocks, _ = extract_block_matrices(
            model=model,
            grid_nodes=top_nodes,
            dirichlet_nodes=pad_set,
            port_nodes=port_nodes,
            dirichlet_voltage=1.0,
        )

        # Port nodes should match
        self.assertEqual(len(blocks.port_nodes), blocks.n_ports)
        self.assertEqual(len(blocks.port_to_idx), blocks.n_ports)

        # Interior nodes should match
        self.assertEqual(len(blocks.interior_nodes), blocks.n_interior)
        self.assertEqual(len(blocks.interior_to_idx), blocks.n_interior)


class TestILUPreconditioner(unittest.TestCase):
    """Tests for ILU preconditioner."""

    def test_ilu_preconditioner_converges(self):
        """Test that ILU preconditioner enables convergence."""
        from core import create_model_from_synthetic, UnifiedIRDropSolver

        G, loads, pads = build_small_grid()
        model = create_model_from_synthetic(G, pads, vdd=1.0)
        solver = UnifiedIRDropSolver(model)

        result = solver.solve_hierarchical_coupled(
            current_injections=loads,
            partition_layer=1,
            preconditioner='ilu',
            tol=1e-8,
        )

        self.assertTrue(result.converged)
        self.assertEqual(result.preconditioner_type, 'ilu')


class TestCoupledSystemWithFixtures(unittest.TestCase):
    """Tests using fixture graphs for edge cases."""

    def test_basic_fixture_coupled_solve(self):
        """Test coupled solve on basic fixture graph."""
        from tests.fixtures import create_minimal_pdn_graph
        from core import create_model_from_pdn, UnifiedIRDropSolver

        graph, pads, load_currents = create_minimal_pdn_graph('basic')
        model = create_model_from_pdn(graph, 'VDD')
        solver = UnifiedIRDropSolver(model)

        # Run coupled solve
        result = solver.solve_hierarchical_coupled(
            current_injections=load_currents,
            partition_layer='M2',
            tol=1e-8,
        )

        self.assertTrue(result.converged)
        self.assertGreater(len(result.voltages), 0)

    def test_fixture_coupled_vs_flat(self):
        """Test that coupled matches flat on fixture graph."""
        from tests.fixtures import create_minimal_pdn_graph
        from core import create_model_from_pdn, UnifiedIRDropSolver

        graph, pads, load_currents = create_minimal_pdn_graph('basic')
        model = create_model_from_pdn(graph, 'VDD')
        solver = UnifiedIRDropSolver(model)

        flat_result = solver.solve(load_currents)
        coupled_result = solver.solve_hierarchical_coupled(
            current_injections=load_currents,
            partition_layer='M2',
            tol=1e-10,
        )

        # Compare
        max_diff = 0.0
        for node in flat_result.voltages:
            if node in coupled_result.voltages:
                diff = abs(flat_result.voltages[node] - coupled_result.voltages[node])
                max_diff = max(max_diff, diff)

        self.assertLess(max_diff, 1e-6)


if __name__ == '__main__':
    unittest.main()

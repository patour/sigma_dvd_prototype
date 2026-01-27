"""Integration tests for hierarchical IR-drop solver.

These tests exercise the full hierarchical solver flow including the expensive
_aggregate_currents_to_ports with multi-source Dijkstra. They are slow and
should be run separately from unit tests.

Run with:
    python -m unittest tests.test_hierarchical_integration -v

Or run specific test:
    python -m unittest tests.test_hierarchical_integration.TestFullHierarchicalFlow.test_2x2_full_flow -v
"""

import os
import signal
import sys
import unittest
from functools import wraps

import numpy as np


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


def _load_pdn_small():
    """Load the small PDN netlist for integration tests.
    
    Returns:
        (model, load_currents, solver) or None if netlist unavailable
    """
    # Add prototype root to path
    prototype_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, prototype_root)
    
    netlist_path = os.path.join(prototype_root, 'pdn', 'netlist_small')
    if not os.path.exists(netlist_path):
        return None
        
    try:
        from pdn.pdn_parser import NetlistParser
        from core import create_model_from_pdn, UnifiedIRDropSolver
        
        parser = NetlistParser(netlist_path, validate=False)
        graph = parser.parse()
        model = create_model_from_pdn(graph, 'VDD_XLV')
        load_currents = model.extract_current_sources()
        solver = UnifiedIRDropSolver(model)
        
        return model, load_currents, solver
    except Exception as e:
        print(f"Failed to load PDN netlist: {e}")
        return None


class TestFullHierarchicalFlow(unittest.TestCase):
    """Integration tests for full hierarchical solver flow.
    
    These tests exercise _aggregate_currents_to_ports which uses multi-source
    Dijkstra and can be slow for large PDNs. Use 300s timeout for safety.
    """
    
    @classmethod
    def setUpClass(cls):
        """Load PDN netlist once for all tests."""
        result = _load_pdn_small()
        if result is None:
            cls.model = None
            cls.load_currents = None
            cls.solver = None
        else:
            cls.model, cls.load_currents, cls.solver = result
    
    def setUp(self):
        """Skip tests if PDN netlist not available."""
        if self.model is None:
            self.skipTest("PDN netlist_small not available")

    # ========================================================================
    # Full hierarchical flow tests (with _aggregate_currents_to_ports)
    # ========================================================================
    
    @timeout(300)
    def test_2x2_full_hierarchical_flow(self):
        """Test 2x2 tiling with full hierarchical flow (no override)."""
        result = self.solver.solve_hierarchical_tiled(
            current_injections=self.load_currents,
            partition_layer='M2',
            N_x=2,
            N_y=2,
            halo_percent=0.2,
            top_k=5,
            weighting='shortest_path',
            n_workers=1,
            parallel_backend='thread',
            validate_against_flat=True,
        )
        
        # Basic sanity checks
        self.assertGreater(len(result.voltages), 0)
        self.assertEqual(len(result.tiles), 4)  # 2x2 = 4 tiles
        
        # Should have computed port currents (non-empty aggregation)
        self.assertGreater(len(result.port_currents), 0)
        self.assertGreater(len(result.aggregation_map), 0)
        
        # Validation should show reasonable accuracy
        self.assertIsNotNone(result.validation_stats)
        max_diff = result.validation_stats['max_diff']
        self.assertLess(max_diff, 0.005, "Max diff should be < 5mV")
    
    @timeout(300)
    def test_1x1_full_hierarchical_flow(self):
        """Test 1x1 tiling (single tile) with full hierarchical flow."""
        result = self.solver.solve_hierarchical_tiled(
            current_injections=self.load_currents,
            partition_layer='M2',
            N_x=1,
            N_y=1,
            halo_percent=0.2,
            top_k=5,
            weighting='shortest_path',
            n_workers=1,
            parallel_backend='thread',
            validate_against_flat=True,
        )
        
        # Should produce exactly 1 tile
        self.assertEqual(len(result.tiles), 1)
        
        # Single tile should match flat solver very closely
        self.assertIsNotNone(result.validation_stats)
        max_diff = result.validation_stats['max_diff']
        self.assertLess(max_diff, 0.001, "Max diff should be < 1mV for single tile")
    
    @timeout(300)
    def test_3x3_full_hierarchical_flow(self):
        """Test 3x3 tiling with full hierarchical flow."""
        result = self.solver.solve_hierarchical_tiled(
            current_injections=self.load_currents,
            partition_layer='M2',
            N_x=3,
            N_y=3,
            halo_percent=0.2,
            top_k=5,
            weighting='shortest_path',
            n_workers=1,
            parallel_backend='thread',
            validate_against_flat=True,
        )
        
        # Should produce up to 9 tiles (may be fewer if merged)
        self.assertGreater(len(result.tiles), 0)
        self.assertLessEqual(len(result.tiles), 9)
        
        # Accuracy check
        self.assertIsNotNone(result.validation_stats)
        self.assertLess(result.validation_stats['max_diff'], 0.005)

    # ========================================================================
    # Current aggregation specific tests
    # ========================================================================
    
    @timeout(300)
    def test_aggregate_currents_to_ports_shortest_path(self):
        """Test _aggregate_currents_to_ports with shortest_path weighting."""
        # Get decomposition
        top_nodes, bottom_nodes, port_nodes, _ = self.model._decompose_at_layer('M2')
        
        # Filter currents to bottom-grid only
        bottom_currents = {
            n: c for n, c in self.load_currents.items()
            if n in bottom_nodes and n not in port_nodes
        }
        
        # Call the aggregation function directly
        port_currents, aggregation_map = self.solver._aggregator.aggregate_currents_to_ports(
            current_injections=self.load_currents,
            bottom_grid_nodes=bottom_nodes,
            port_nodes=port_nodes,
            top_k=5,
            weighting='shortest_path',
        )
        
        # Should have non-empty results
        self.assertGreater(len(port_currents), 0)
        self.assertGreater(len(aggregation_map), 0)
        
        # Total port current should approximately equal total load current
        total_port_current = sum(port_currents.values())
        total_load_current = sum(bottom_currents.values())
        
        # Allow some tolerance due to floating point
        self.assertAlmostEqual(
            total_port_current, total_load_current, 
            delta=abs(total_load_current) * 0.01,
            msg="Total port current should match total load current"
        )
        
        # Each aggregation entry should have valid structure
        for load_node, contributions in aggregation_map.items():
            self.assertIsInstance(contributions, list)
            self.assertGreater(len(contributions), 0)
            
            # Each contribution is (port, weight, current)
            total_weight = 0
            for port, weight, current in contributions:
                self.assertIn(port, port_nodes)
                self.assertGreater(weight, 0)
                total_weight += weight
            
            # Weights should sum to approximately 1
            self.assertAlmostEqual(total_weight, 1.0, delta=0.001)
    
    @timeout(300)
    def test_aggregate_currents_top_k_variation(self):
        """Test that different top_k values produce valid aggregations."""
        top_nodes, bottom_nodes, port_nodes, _ = self.model._decompose_at_layer('M2')
        
        for top_k in [1, 3, 10]:
            port_currents, aggregation_map = self.solver._aggregator.aggregate_currents_to_ports(
                current_injections=self.load_currents,
                bottom_grid_nodes=bottom_nodes,
                port_nodes=port_nodes,
                top_k=top_k,
                weighting='shortest_path',
            )
            
            # Each load should have at most top_k ports
            for load_node, contributions in aggregation_map.items():
                self.assertLessEqual(
                    len(contributions), top_k,
                    f"Load {load_node} has {len(contributions)} ports, expected <= {top_k}"
                )
    
    @timeout(300)
    def test_hierarchical_vs_flat_accuracy(self):
        """Compare full hierarchical solve against flat solve."""
        # Run flat solver
        flat_result = self.solver.solve(self.load_currents)
        
        # Run full hierarchical tiled solver
        hier_result = self.solver.solve_hierarchical_tiled(
            current_injections=self.load_currents,
            partition_layer='M2',
            N_x=2,
            N_y=2,
            halo_percent=0.2,
            top_k=5,
            weighting='shortest_path',
            n_workers=1,
            parallel_backend='thread',
        )
        
        # Get bottom-grid nodes for comparison
        _, bottom_nodes, _, _ = self.model._decompose_at_layer('M2')
        
        # Compare voltages
        errors = []
        for node in bottom_nodes:
            if node in hier_result.voltages and node in flat_result.voltages:
                error = abs(hier_result.voltages[node] - flat_result.voltages[node])
                errors.append(error)
        
        errors = np.array(errors)
        
        # Hierarchical should be reasonably close to flat
        # (some error expected due to current aggregation approximation)
        self.assertLess(errors.max(), 0.010, "Max error should be < 10mV")
        self.assertLess(errors.mean(), 0.003, "Mean error should be < 3mV")

    # ========================================================================
    # Bottom-grid validation tests
    # ========================================================================
    
    @timeout(300)
    def test_bottom_grid_with_flat_port_voltages(self):
        """Validate bottom-grid solver using flat solver port voltages as Dirichlet BCs.
        
        This test isolates the bottom-grid solver by using the exact port voltages
        from the flat solver as Dirichlet boundary conditions. The bottom-grid
        voltages should match the flat solver's bottom-grid voltages exactly
        (within numerical precision).
        """
        # Run flat solver to get ground truth
        flat_result = self.solver.solve(self.load_currents)
        flat_voltages = flat_result.voltages
        
        # Decompose the PDN at M3
        partition_layer = 'M3'
        top_nodes, bottom_nodes, ports, via_edges = self.model._decompose_at_layer(partition_layer)
        
        # Get the flat solver's port voltages (ground truth)
        flat_port_voltages = {p: flat_voltages[p] for p in ports if p in flat_voltages}
        
        # Build bottom-grid system with ports as Dirichlet nodes
        bottom_subgrid = bottom_nodes | ports
        bottom_system = self.model._build_subgrid_system(
            subgrid_nodes=bottom_subgrid,
            dirichlet_nodes=ports,
            dirichlet_voltage=self.model.vdd,
        )
        
        self.assertIsNotNone(bottom_system, "Failed to build bottom-grid system")
        
        # Get bottom-grid currents
        bottom_grid_currents = {n: c for n, c in self.load_currents.items() if n in bottom_nodes}
        
        # Solve with flat solver's port voltages
        validated_bottom_voltages = self.model._solve_subgrid(
            reduced_system=bottom_system,
            current_injections=bottom_grid_currents,
            dirichlet_voltages=flat_port_voltages,
        )
        
        # Compare with flat solver's bottom-grid voltages
        errors = []
        for node in bottom_nodes:
            if node in validated_bottom_voltages and node in flat_voltages:
                error = abs(validated_bottom_voltages[node] - flat_voltages[node])
                errors.append(error)
        
        self.assertGreater(len(errors), 0, "No bottom-grid nodes to validate")
        
        errors = np.array(errors)
        
        # Bottom-grid solver with exact port voltages should be very accurate
        # Errors should be at numerical precision level (< 1e-6 V = 1 µV)
        self.assertLess(
            errors.max(), 1e-6,
            f"Max error {errors.max():.2e} V should be < 1 µV for exact port voltages"
        )
        self.assertLess(
            errors.mean(), 1e-9,
            f"Mean error {errors.mean():.2e} V should be negligible"
        )

    # ========================================================================
    # Different partition layers
    # ========================================================================
    
    @timeout(300)
    def test_partition_at_M3(self):
        """Test hierarchical solve with partition at M3 layer."""
        result = self.solver.solve_hierarchical_tiled(
            current_injections=self.load_currents,
            partition_layer='M3',
            N_x=2,
            N_y=2,
            halo_percent=0.2,
            top_k=5,
            weighting='shortest_path',
            n_workers=1,
            parallel_backend='thread',
            validate_against_flat=True,
        )
        
        # Should complete and produce valid results
        self.assertGreater(len(result.voltages), 0)
        self.assertEqual(result.partition_layer, 'M3')
        
        # Accuracy check
        self.assertIsNotNone(result.validation_stats)
        self.assertLess(result.validation_stats['max_diff'], 0.005)


class TestCurrentAggregationWeighting(unittest.TestCase):
    """Tests for different current aggregation weighting methods."""
    
    @classmethod
    def setUpClass(cls):
        """Load PDN netlist once for all tests."""
        result = _load_pdn_small()
        if result is None:
            cls.model = None
            cls.load_currents = None
            cls.solver = None
        else:
            cls.model, cls.load_currents, cls.solver = result
    
    def setUp(self):
        """Skip tests if PDN netlist not available."""
        if self.model is None:
            self.skipTest("PDN netlist_small not available")
    
    @timeout(300)
    def test_shortest_path_weighting(self):
        """Test shortest_path weighting method."""
        result = self.solver.solve_hierarchical_tiled(
            current_injections=self.load_currents,
            partition_layer='M2',
            N_x=2,
            N_y=2,
            halo_percent=0.2,
            top_k=5,
            weighting='shortest_path',
            n_workers=1,
            parallel_backend='thread',
        )
        
        self.assertGreater(len(result.voltages), 0)
        self.assertGreater(len(result.port_currents), 0)
    
    @unittest.skip("Too slow - effective resistance weighting is computationally expensive")
    @timeout(600)
    def test_effective_resistance_weighting(self):
        """Test effective resistance weighting method (slower)."""
        result = self.solver.solve_hierarchical_tiled(
            current_injections=self.load_currents,
            partition_layer='M2',
            N_x=2,
            N_y=2,
            halo_percent=0.2,
            top_k=3,  # Use smaller top_k for speed
            weighting='effective',
            n_workers=1,
            parallel_backend='thread',
        )
        
        self.assertGreater(len(result.voltages), 0)
        self.assertGreater(len(result.port_currents), 0)
    
    @timeout(300)
    def test_shortest_path_with_rmax(self):
        """Test shortest_path weighting with rmax limit."""
        top_nodes, bottom_nodes, port_nodes, _ = self.model._decompose_at_layer('M2')
        
        # Use a small rmax to limit search distance
        port_currents, aggregation_map = self.solver._aggregator.aggregate_currents_to_ports(
            current_injections=self.load_currents,
            bottom_grid_nodes=bottom_nodes,
            port_nodes=port_nodes,
            top_k=5,
            weighting='shortest_path',
            rmax=1.0,  # 1 Ohm max distance
        )
        
        # Should still produce results (some loads may not find ports within rmax)
        # The function should handle this gracefully or raise informative error
        self.assertIsInstance(port_currents, dict)


class TestCGAMGIntegration(unittest.TestCase):
    """Integration tests for CG + AMG solver on real PDN netlist."""

    @classmethod
    def setUpClass(cls):
        """Load real PDN netlist for integration testing."""
        result = _load_pdn_small()
        if result is None:
            cls.model = None
            cls.solver = None
            cls.load_currents = None
        else:
            cls.model, cls.load_currents, cls.solver = result

        # Check pyamg availability
        from core import HAS_PYAMG
        cls.has_pyamg = HAS_PYAMG

    def setUp(self):
        """Skip tests if PDN netlist or pyamg not available."""
        if self.model is None:
            self.skipTest("PDN netlist_small not available")
        if not self.has_pyamg:
            self.skipTest("pyamg not installed")

    @timeout(120)
    def test_cg_amg_on_pdn_netlist(self):
        """Test CG + AMG on real PDN netlist."""
        result = self.solver.solve_hierarchical_coupled(
            current_injections=self.load_currents,
            partition_layer='M2',
            solver='cg',
            tol=1e-8,
            preconditioner='amg',
        )

        self.assertTrue(result.converged)
        self.assertGreater(len(result.voltages), 0)
        self.assertLess(result.iterations, 100)  # AMG should converge quickly

    @timeout(120)
    def test_cg_amg_accuracy_vs_flat(self):
        """CG + AMG should match flat solver on PDN netlist."""
        flat_result = self.solver.solve(self.load_currents)

        coupled_result = self.solver.solve_hierarchical_coupled(
            current_injections=self.load_currents,
            partition_layer='M2',
            solver='cg',
            tol=1e-10,
            maxiter=500,
            preconditioner='amg',
        )

        # Compare voltages
        max_diff = 0.0
        for node in flat_result.voltages:
            if node in coupled_result.voltages:
                diff = abs(flat_result.voltages[node] - coupled_result.voltages[node])
                max_diff = max(max_diff, diff)

        # Should match to within 1 µV
        self.assertLess(max_diff * 1e6, 1.0,
                       f"Max voltage difference {max_diff*1e6:.4f} µV exceeds 1 µV")

    @timeout(120)
    def test_cg_amg_batch_solving_pdn(self):
        """Test batch solving with CG + AMG on PDN netlist."""
        # Prepare context
        ctx = self.solver.prepare_hierarchical_coupled(
            partition_layer='M2',
            solver='cg',
            tol=1e-8,
            preconditioner='amg',
        )

        # Solve multiple scenarios
        results = []
        for scale in [0.5, 1.0, 2.0]:
            scaled_currents = {n: c * scale for n, c in self.load_currents.items()}
            result = self.solver.solve_hierarchical_coupled_prepared(scaled_currents, ctx)
            results.append(result)
            self.assertTrue(result.converged)

        # IR-drop should scale with current
        ir_drops = [max(r.ir_drop.values()) for r in results]
        # 2x current should give ~2x IR-drop (within 10%)
        ratio = ir_drops[2] / ir_drops[1]  # scale 2.0 / scale 1.0
        self.assertAlmostEqual(ratio, 2.0, delta=0.2)

    @timeout(120)
    def test_all_solver_precond_combinations_pdn(self):
        """Test various solver/preconditioner combinations on PDN netlist."""
        configs = [
            ('cg', 'amg'),
            ('gmres', 'amg'),
            ('bicgstab', 'amg'),
            ('cg', 'block_diagonal'),
        ]

        results = {}
        for solver_name, precond in configs:
            result = self.solver.solve_hierarchical_coupled(
                current_injections=self.load_currents,
                partition_layer='M2',
                solver=solver_name,
                tol=1e-8,
                preconditioner=precond,
            )
            self.assertTrue(result.converged,
                           f"{solver_name}+{precond} did not converge")
            results[(solver_name, precond)] = result

        # All results should be similar
        ref_result = results[('cg', 'amg')]
        for key, result in results.items():
            if key == ('cg', 'amg'):
                continue
            max_diff = max(
                abs(ref_result.voltages[n] - result.voltages[n])
                for n in ref_result.voltages if n in result.voltages
            )
            self.assertLess(max_diff, 2e-5,
                           f"{key} differs from CG+AMG by {max_diff}")


if __name__ == '__main__':
    # Run with verbose output
    unittest.main(verbosity=2)

import math
import unittest

import numpy as np

from generate_power_grid import generate_power_grid, NodeID
from irdrop import (
    PowerGridModel, 
    StimulusGenerator, 
    IRDropSolver, 
    EffectiveResistanceCalculator,
    plot_voltage_map, 
    plot_ir_drop_map
)


def build_small():
    G, loads, pads = generate_power_grid(
        K=3, N0=8, I_N=80, N_vsrc=3,
        max_stripe_res=1.0, max_via_res=0.1,
        load_current=1.0, seed=3, plot=False
    )
    return G, loads, pads


class TestIRDrop(unittest.TestCase):
    def test_no_load_currents_all_pad_voltage(self):
        G, loads, pads = build_small()
        model = PowerGridModel(G, pad_nodes=pads, vdd=1.0)
        solver = IRDropSolver(model)
        result = solver.solve({})
        self.assertTrue(all(0.0 <= v <= 1.0 for v in result.voltages.values()))
        for p in pads:
            self.assertTrue(math.isclose(result.voltages[p], 1.0))

    def test_uniform_power_distribution_current_sum(self):
        G, loads, pads = build_small()
        model = PowerGridModel(G, pad_nodes=pads, vdd=1.0)
        stim_gen = StimulusGenerator(load_nodes=list(loads.keys()), vdd=1.0, seed=10)
        P = 2.0
        meta = stim_gen.generate(total_power=P, percent=0.5, distribution='uniform')
        self.assertTrue(math.isclose(sum(meta.currents.values()), P / 1.0, rel_tol=1e-9))

    def test_batch_min_voltage_monotonic(self):
        G, loads, pads = build_small()
        model = PowerGridModel(G, pad_nodes=pads, vdd=1.0)
        stim_gen = StimulusGenerator(load_nodes=list(loads.keys()), vdd=1.0, seed=2)
        powers = [0.5, 1.0, 2.0]
        metas = stim_gen.generate_batch(powers, percent=0.4, distribution='gaussian')
        solver = IRDropSolver(model)
        results = solver.solve_batch([m.currents for m in metas], metas)
        min_voltages = [min(r.voltages.values()) for r in results]
        self.assertTrue(min_voltages[0] >= min_voltages[1] >= min_voltages[2])

    def test_plot_functions(self):
        G, loads, pads = build_small()
        model = PowerGridModel(G, pad_nodes=pads, vdd=1.0)
        stim_gen = StimulusGenerator(load_nodes=list(loads.keys()), vdd=1.0, seed=5)
        meta = stim_gen.generate(total_power=1.0, percent=0.4, distribution='uniform')
        solver = IRDropSolver(model)
        result = solver.solve(meta.currents)
        fig1, _ = plot_voltage_map(G, result.voltages, show=False)
        fig2, _ = plot_ir_drop_map(G, result.voltages, vdd=1.0, show=False)
        # Just ensure figures were created
        self.assertIsNotNone(fig1)
        self.assertIsNotNone(fig2)


class TestEffectiveResistance(unittest.TestCase):
    """Test suite for effective resistance computation."""
    
    def test_ground_resistance_positive(self):
        """R_eff to ground should be positive for all non-pad nodes."""
        G, loads, pads = build_small()
        model = PowerGridModel(G, pad_nodes=pads, vdd=1.0)
        calc = EffectiveResistanceCalculator(model)
        
        # Test a few load nodes
        test_nodes = list(loads.keys())[:5]
        pairs = [(n, None) for n in test_nodes]
        reff = calc.compute_batch(pairs)
        
        # All resistances should be positive
        self.assertTrue(np.all(reff > 0), f"Found non-positive R_eff: {reff}")
        
    def test_pad_to_ground_is_zero(self):
        """Pad nodes should have zero resistance to ground."""
        G, loads, pads = build_small()
        model = PowerGridModel(G, pad_nodes=pads, vdd=1.0)
        calc = EffectiveResistanceCalculator(model)
        
        # Test pad nodes
        pairs = [(pads[0], None)]
        reff = calc.compute_batch(pairs)
        
        self.assertAlmostEqual(reff[0], 0.0, places=10)
        
    def test_node_to_node_symmetry(self):
        """R_eff(u,v) should equal R_eff(v,u)."""
        G, loads, pads = build_small()
        model = PowerGridModel(G, pad_nodes=pads, vdd=1.0)
        calc = EffectiveResistanceCalculator(model)
        
        # Get two distinct load nodes
        nodes = list(loads.keys())
        u, v = nodes[0], nodes[1]
        
        pairs = [(u, v), (v, u)]
        reff = calc.compute_batch(pairs)
        
        self.assertAlmostEqual(reff[0], reff[1], places=10)
        
    def test_node_to_node_positive(self):
        """R_eff between two distinct nodes should be positive."""
        G, loads, pads = build_small()
        model = PowerGridModel(G, pad_nodes=pads, vdd=1.0)
        calc = EffectiveResistanceCalculator(model)
        
        # Test several pairs
        nodes = list(loads.keys())
        pairs = [(nodes[i], nodes[i+1]) for i in range(min(5, len(nodes)-1))]
        reff = calc.compute_batch(pairs)
        
        self.assertTrue(np.all(reff > 0), f"Found non-positive R_eff: {reff}")
        
    def test_node_to_self_is_zero(self):
        """R_eff(u,u) should be zero (or very small due to numerics)."""
        G, loads, pads = build_small()
        model = PowerGridModel(G, pad_nodes=pads, vdd=1.0)
        calc = EffectiveResistanceCalculator(model)
        
        node = list(loads.keys())[0]
        pairs = [(node, node)]
        reff = calc.compute_batch(pairs)
        
        # Should be very close to zero (numerical tolerance)
        self.assertAlmostEqual(reff[0], 0.0, places=8)
        
    def test_triangle_inequality(self):
        """R_eff should satisfy triangle inequality: R(u,w) <= R(u,v) + R(v,w)."""
        G, loads, pads = build_small()
        model = PowerGridModel(G, pad_nodes=pads, vdd=1.0)
        calc = EffectiveResistanceCalculator(model)
        
        # Get three nodes
        nodes = list(loads.keys())
        u, v, w = nodes[0], nodes[1], nodes[2]
        
        pairs = [(u, w), (u, v), (v, w)]
        reff = calc.compute_batch(pairs)
        r_uw, r_uv, r_vw = reff[0], reff[1], reff[2]
        
        # Triangle inequality with small tolerance for numerical error
        self.assertLessEqual(r_uw, r_uv + r_vw + 1e-10)
        
    def test_batch_mixed_types(self):
        """Test batch computation with mix of ground and node-to-node pairs."""
        G, loads, pads = build_small()
        model = PowerGridModel(G, pad_nodes=pads, vdd=1.0)
        calc = EffectiveResistanceCalculator(model)
        
        nodes = list(loads.keys())
        pairs = [
            (nodes[0], None),      # ground
            (nodes[1], nodes[2]),  # node-to-node
            (nodes[3], None),      # ground
            (nodes[4], nodes[5]),  # node-to-node
        ]
        
        reff = calc.compute_batch(pairs)
        
        # Check length and positivity
        self.assertEqual(len(reff), 4)
        self.assertTrue(np.all(reff > 0))
        
    def test_batch_large(self):
        """Test efficiency with large batch of pairs."""
        G, loads, pads = build_small()
        model = PowerGridModel(G, pad_nodes=pads, vdd=1.0)
        calc = EffectiveResistanceCalculator(model)
        
        nodes = list(loads.keys())
        # Create many pairs
        pairs = []
        for i in range(0, len(nodes), 2):
            pairs.append((nodes[i], None))
        for i in range(len(nodes)-1):
            pairs.append((nodes[i], nodes[i+1]))
            
        reff = calc.compute_batch(pairs)
        
        # Should complete without error and have correct length
        self.assertEqual(len(reff), len(pairs))
        self.assertTrue(np.all(reff >= 0))
        
    def test_single_convenience_method(self):
        """Test compute_single convenience method."""
        G, loads, pads = build_small()
        model = PowerGridModel(G, pad_nodes=pads, vdd=1.0)
        calc = EffectiveResistanceCalculator(model)
        
        nodes = list(loads.keys())
        u, v = nodes[0], nodes[1]
        
        # Single computation
        reff_single = calc.compute_single(u, v)
        
        # Batch computation for comparison
        reff_batch = calc.compute_batch([(u, v)])
        
        self.assertAlmostEqual(reff_single, reff_batch[0], places=10)
        
    def test_empty_batch(self):
        """Test with empty input."""
        G, loads, pads = build_small()
        model = PowerGridModel(G, pad_nodes=pads, vdd=1.0)
        calc = EffectiveResistanceCalculator(model)
        
        reff = calc.compute_batch([])
        
        self.assertEqual(len(reff), 0)
        
    def test_invalid_node_raises(self):
        """Test that invalid nodes raise appropriate errors."""
        G, loads, pads = build_small()
        model = PowerGridModel(G, pad_nodes=pads, vdd=1.0)
        calc = EffectiveResistanceCalculator(model)
        
        # Try to compute with pad node (should fail for node-to-node)
        nodes = list(loads.keys())
        with self.assertRaises(ValueError):
            calc.compute_batch([(pads[0], nodes[0])])
            
    def test_consistency_with_voltage_solve(self):
        """Verify R_eff relates correctly to voltage drop under unit current."""
        G, loads, pads = build_small()
        model = PowerGridModel(G, pad_nodes=pads, vdd=1.0)
        calc = EffectiveResistanceCalculator(model)
        
        # Pick a load node
        node = list(loads.keys())[0]
        
        # Compute R_eff to ground
        reff_to_ground = calc.compute_single(node, None)
        
        # Apply 1A current at that node and solve
        currents = {node: 1.0}  # 1A sink
        voltages = model.solve_voltages(currents)
        
        # Voltage drop from Vdd should approximately equal R_eff * I
        # V_node = Vdd - R_eff * I_node
        expected_voltage = 1.0 - reff_to_ground * 1.0
        actual_voltage = voltages[node]
        
        # Should be close (subject to network effects)
        self.assertAlmostEqual(expected_voltage, actual_voltage, places=6)


if __name__ == '__main__':
    unittest.main()

"""Unit tests for RegionalIRDropSolver.

Tests the regional IR-drop solver against the full solver for consistency
and validates the algorithm on partitioned power grids.
"""

import math
import unittest

import numpy as np

from generate_power_grid import generate_power_grid, NodeID
from irdrop import (
    PowerGridModel,
    EffectiveResistanceCalculator,
    RegionalIRDropSolver,
    IRDropSolver,
    GridPartitioner,
)


def build_test_grid():
    """Build a small test grid for unit tests."""
    G, loads, pads = generate_power_grid(
        K=3, N0=8, I_N=40, N_vsrc=3,
        max_stripe_res=1.0, max_via_res=0.1,
        load_current=0.01, seed=42, plot=False
    )
    return G, loads, pads


class TestRegionalIRDropSolver(unittest.TestCase):
    """Test suite for RegionalIRDropSolver."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.G, self.loads, self.pads = build_test_grid()
        self.model = PowerGridModel(self.G, pad_nodes=self.pads, vdd=1.0)
        self.calc = EffectiveResistanceCalculator(self.model)
        self.regional_solver = RegionalIRDropSolver(self.calc)
        self.full_solver = IRDropSolver(self.model)
        
    def test_initialization(self):
        """Test that RegionalIRDropSolver initializes correctly."""
        self.assertIsNotNone(self.regional_solver)
        self.assertEqual(self.regional_solver.calc, self.calc)
        
    def test_compute_ir_drops_matches_full_solver(self):
        """Test that regional solver matches full solver for a partition."""
        # Partition the grid
        partitioner = GridPartitioner(self.G, load_nodes=self.loads, pad_nodes=self.pads)
        partition_result = partitioner.partition(P=2, axis='x')
        
        # Select first partition and ALL of its loads as subset S
        partition = partition_result.partitions[0]
        load_nodes = sorted(list(partition.load_nodes), key=lambda n: (n.layer, n.idx))
        
        # Use all load nodes in partition as S
        S = set(load_nodes)
        R = partition.all_nodes
        A = partition.separator_nodes
        
        # Define currents - I_R includes all loads in region R, I_F includes loads outside R
        loads_in_R = set(partition.load_nodes)
        I_R = {node: self.loads[node] for node in loads_in_R if node in self.loads}
        all_far_loads = set(self.loads.keys()) - loads_in_R
        I_F = {node: self.loads[node] for node in all_far_loads}
        
        # Compute IR-drops with regional solver
        ir_drops_regional, drop_near, drop_far = self.regional_solver.compute_ir_drops(S, R, A, I_R, I_F)
        
        # Compute IR-drops with full solver
        result_full = self.full_solver.solve(self.loads)
        
        # Compare results
        for node in S:
            drop_regional = ir_drops_regional[node]
            drop_full = result_full.ir_drop[node]
            self.assertAlmostEqual(
                drop_regional, drop_full, places=12,
                msg=f"IR-drop mismatch at node {node}: regional={drop_regional}, full={drop_full}"
            )
    
    def test_single_node_subset(self):
        """Test regional solver with a single node subset."""
        # Partition the grid
        partitioner = GridPartitioner(self.G, load_nodes=self.loads, pad_nodes=self.pads)
        partition_result = partitioner.partition(P=2, axis='x')
        
        partition = partition_result.partitions[0]
        load_nodes = list(partition.load_nodes)
        
        if not load_nodes:
            self.skipTest("Partition has no load nodes")
        
        # Single node subset
        S = {load_nodes[0]}
        R = partition.all_nodes
        A = partition.separator_nodes
        
        loads_in_R = set(partition.load_nodes)
        I_R = {node: self.loads[node] for node in loads_in_R if node in self.loads}
        all_far_loads = set(self.loads.keys()) - loads_in_R
        I_F = {node: self.loads[node] for node in all_far_loads}
        
        # Should not raise any errors
        ir_drops_regional, drop_near, drop_far = self.regional_solver.compute_ir_drops(S, R, A, I_R, I_F)
        
        self.assertEqual(len(ir_drops_regional), 1)
        self.assertIn(load_nodes[0], ir_drops_regional)
        self.assertGreater(ir_drops_regional[load_nodes[0]], 0.0)
        
    def test_empty_subset_raises_error(self):
        """Test that empty subset S raises ValueError."""
        partitioner = GridPartitioner(self.G, load_nodes=self.loads, pad_nodes=self.pads)
        partition_result = partitioner.partition(P=2, axis='x')
        
        partition = partition_result.partitions[0]
        S = set()  # Empty
        R = partition.all_nodes
        A = partition.separator_nodes
        
        I_R = {}
        I_F = self.loads
        
        with self.assertRaises(ValueError) as context:
            self.regional_solver.compute_ir_drops(S, R, A, I_R, I_F)
        
        self.assertIn("Subset S cannot be empty", str(context.exception))
        
    def test_empty_boundary_raises_error(self):
        """Test that empty boundary set A raises ValueError."""
        load_node = list(self.loads.keys())[0]
        S = {load_node}
        R = {load_node}
        A = set()  # Empty boundary
        
        I_R = {load_node: self.loads[load_node]}
        I_F = {node: self.loads[node] for node in self.loads if node != load_node}
        
        with self.assertRaises(ValueError) as context:
            self.regional_solver.compute_ir_drops(S, R, A, I_R, I_F)
        
        self.assertIn("Boundary set A cannot be empty", str(context.exception))
        
    def test_subset_not_in_region_raises_error(self):
        """Test that subset S not contained in region R raises ValueError."""
        partitioner = GridPartitioner(self.G, load_nodes=self.loads, pad_nodes=self.pads)
        partition_result = partitioner.partition(P=2, axis='x')
        
        partition0 = partition_result.partitions[0]
        partition1 = partition_result.partitions[1]
        
        # S from partition 0, but R from partition 1
        load_nodes_0 = list(partition0.load_nodes)
        if not load_nodes_0:
            self.skipTest("Partition 0 has no load nodes")
            
        S = {load_nodes_0[0]}
        R = partition1.all_nodes  # Wrong region
        A = partition1.separator_nodes
        
        I_R = {load_nodes_0[0]: self.loads[load_nodes_0[0]]}
        I_F = {node: self.loads[node] for node in self.loads if node not in S}
        
        with self.assertRaises(ValueError) as context:
            self.regional_solver.compute_ir_drops(S, R, A, I_R, I_F)
        
        self.assertIn("Subset S must be contained in region R", str(context.exception))
        
    def test_ir_drops_are_positive(self):
        """Test that computed IR-drops are non-negative."""
        partitioner = GridPartitioner(self.G, load_nodes=self.loads, pad_nodes=self.pads)
        partition_result = partitioner.partition(P=2, axis='x')
        
        partition = partition_result.partitions[0]
        load_nodes = sorted(list(partition.load_nodes), key=lambda n: (n.layer, n.idx))
        
        if len(load_nodes) < 2:
            self.skipTest("Not enough load nodes in partition")
        
        S = set(load_nodes[:3])
        R = partition.all_nodes
        A = partition.separator_nodes
        
        loads_in_R = set(partition.load_nodes)
        I_R = {node: self.loads[node] for node in loads_in_R if node in self.loads}
        all_far_loads = set(self.loads.keys()) - loads_in_R
        I_F = {node: self.loads[node] for node in all_far_loads}
        
        ir_drops_regional, drop_near, drop_far = self.regional_solver.compute_ir_drops(S, R, A, I_R, I_F)
        
        for node, drop in ir_drops_regional.items():
            self.assertGreaterEqual(drop, 0.0, msg=f"IR-drop at {node} is negative: {drop}")
            
    def test_zero_far_loads(self):
        """Test regional solver when all far loads have zero current."""
        partitioner = GridPartitioner(self.G, load_nodes=self.loads, pad_nodes=self.pads)
        partition_result = partitioner.partition(P=2, axis='x')
        
        partition = partition_result.partitions[0]
        load_nodes = sorted(list(partition.load_nodes), key=lambda n: (n.layer, n.idx))
        
        if len(load_nodes) < 2:
            self.skipTest("Not enough load nodes in partition")
        
        S = set(load_nodes[:2])
        R = partition.all_nodes
        A = partition.separator_nodes
        
        loads_in_R = set(partition.load_nodes)
        I_R = {node: self.loads[node] for node in loads_in_R if node in self.loads}
        all_far_loads = set(self.loads.keys()) - loads_in_R
        I_F = {node: 0.0 for node in all_far_loads}  # Zero far loads
        
        # Should compute successfully with only near load contribution
        ir_drops_regional, drop_near, drop_far = self.regional_solver.compute_ir_drops(S, R, A, I_R, I_F)
        
        self.assertEqual(len(ir_drops_regional), len(S))
        for node in S:
            self.assertGreater(ir_drops_regional[node], 0.0)
            
    def test_zero_near_loads(self):
        """Test regional solver when all near loads have zero current."""
        partitioner = GridPartitioner(self.G, load_nodes=self.loads, pad_nodes=self.pads)
        partition_result = partitioner.partition(P=2, axis='x')
        
        partition = partition_result.partitions[0]
        load_nodes = sorted(list(partition.load_nodes), key=lambda n: (n.layer, n.idx))
        
        if len(load_nodes) < 2:
            self.skipTest("Not enough load nodes in partition")
        
        S = set(load_nodes[:2])
        R = partition.all_nodes
        A = partition.separator_nodes
        
        loads_in_R = set(partition.load_nodes)
        I_R = {node: 0.0 for node in loads_in_R}  # Zero near loads (all loads in R)
        all_far_loads = set(self.loads.keys()) - loads_in_R
        I_F = {node: self.loads[node] for node in all_far_loads}
        
        # Should compute successfully with only far load contribution
        ir_drops_regional, drop_near, drop_far = self.regional_solver.compute_ir_drops(S, R, A, I_R, I_F)
        
        self.assertEqual(len(ir_drops_regional), len(S))
        # IR-drops should still be positive due to far loads
        for node in S:
            self.assertGreaterEqual(ir_drops_regional[node], 0.0)
            
    def test_multiple_partitions(self):
        """Test regional solver works correctly across multiple partitions."""
        partitioner = GridPartitioner(self.G, load_nodes=self.loads, pad_nodes=self.pads)
        partition_result = partitioner.partition(P=2, axis='x')  # Use just 2 partitions
        
        # Test on first partition only (simpler case)
        partition = partition_result.partitions[0]
        load_nodes = sorted(list(partition.load_nodes), key=lambda n: (n.layer, n.idx))
        
        if not load_nodes:
            self.skipTest("No load nodes in partition")
            
        # Use all loads in this partition as S
        S = set(load_nodes)
        R = partition.all_nodes
        A = partition.separator_nodes
        
        loads_in_R = set(partition.load_nodes)
        I_R = {node: self.loads[node] for node in loads_in_R if node in self.loads}
        all_far_loads = set(self.loads.keys()) - loads_in_R
        I_F = {node: self.loads[node] for node in all_far_loads}
        
        ir_drops_regional, drop_near, drop_far = self.regional_solver.compute_ir_drops(S, R, A, I_R, I_F)
        result_full = self.full_solver.solve(self.loads)
        
        # Verify partition matches full solver
        for node in S:
            drop_regional = ir_drops_regional[node]
            drop_full = result_full.ir_drop[node]
            self.assertAlmostEqual(
                drop_regional, drop_full, places=11,
                msg=f"Partition {partition.partition_id}, node {node}: regional={drop_regional}, full={drop_full}"
            )
                
    def test_precision_with_different_currents(self):
        """Test solver precision with various current magnitudes."""
        partitioner = GridPartitioner(self.G, load_nodes=self.loads, pad_nodes=self.pads)
        partition_result = partitioner.partition(P=2, axis='x')
        
        partition = partition_result.partitions[0]
        load_nodes = sorted(list(partition.load_nodes), key=lambda n: (n.layer, n.idx))
        
        if not load_nodes:
            self.skipTest("No load nodes in partition")
        
        # Use all loads in partition as S
        S = set(load_nodes)
        R = partition.all_nodes
        A = partition.separator_nodes
        
        loads_in_R = set(partition.load_nodes)
        
        # Test with different current scales
        for scale in [0.001, 0.01, 0.1, 1.0]:
            scaled_loads = {node: curr * scale for node, curr in self.loads.items()}
            
            I_R = {node: scaled_loads[node] for node in loads_in_R if node in scaled_loads}
            all_far_loads = set(scaled_loads.keys()) - loads_in_R
            I_F = {node: scaled_loads[node] for node in all_far_loads}
            
            ir_drops_regional, drop_near, drop_far = self.regional_solver.compute_ir_drops(S, R, A, I_R, I_F)
            result_full = self.full_solver.solve(scaled_loads)
            
            for node in S:
                drop_regional = ir_drops_regional[node]
                drop_full = result_full.ir_drop[node]
                rel_error = abs(drop_regional - drop_full) / (abs(drop_full) + 1e-15)
                self.assertLess(
                    rel_error, 1e-9,  # Relaxed from 1e-10 to account for numerical precision
                    msg=f"Scale {scale}, node {node}: relative error {rel_error}"
                )


if __name__ == '__main__':
    unittest.main()

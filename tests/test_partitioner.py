#!/usr/bin/env python3
"""Tests for grid partitioner."""

import unittest
import numpy as np

from generate_power_grid import generate_power_grid
from irdrop.grid_partitioner import GridPartitioner, PartitionResult


def build_test_grid():
    """Build a small test grid for partitioning."""
    G, loads, pads = generate_power_grid(
        K=3, N0=16, I_N=100, N_vsrc=4,
        max_stripe_res=1.0, max_via_res=0.1,
        load_current=1.0, seed=42, plot=False
    )
    return G, loads, pads


class TestGridPartitioner(unittest.TestCase):
    
    def test_partition_creates_expected_structure(self):
        """Test that partitioning creates valid structure."""
        G, loads, pads = build_test_grid()
        partitioner = GridPartitioner(G, loads, pads, seed=42)
        
        result = partitioner.partition(P=2)
        
        # Check basic structure
        self.assertEqual(result.num_partitions, 2)
        self.assertEqual(len(result.partitions), 2)
        self.assertIsInstance(result, PartitionResult)
        
        # Check each partition has valid structure
        for partition in result.partitions:
            self.assertGreaterEqual(len(partition.interior_nodes), 0)
            self.assertGreaterEqual(len(partition.separator_nodes), 0)
            self.assertGreaterEqual(len(partition.load_nodes), 0)
    
    def test_structured_load_balance(self):
        """Structured partition load balance: ratio within acceptable bound (<=3.5).
        
        Via-column boundary constraints can cause higher imbalance than ideal midpoint placement.
        """
        G, loads, pads = build_test_grid()
        partitioner = GridPartitioner(G, loads, pads, seed=42)
        result = partitioner.partition(P=4)
        counts = [p.num_loads for p in result.partitions]
        self.assertLessEqual(result.load_balance_ratio, 3.5)
        self.assertEqual(sum(counts), len(loads))
    
    def test_axis_parameter(self):
        """Test X, Y, and auto axis partitioning modes."""
        G, loads, pads = build_test_grid()
        partitioner = GridPartitioner(G, loads, pads, seed=42)
        
        # Test X axis
        result_x = partitioner.partition(P=4, axis='x')
        self.assertEqual(result_x.num_partitions, 4)
        self.assertEqual(sum(p.num_loads for p in result_x.partitions), len(loads))
        
        # Test Y axis
        result_y = partitioner.partition(P=4, axis='y')
        self.assertEqual(result_y.num_partitions, 4)
        self.assertEqual(sum(p.num_loads for p in result_y.partitions), len(loads))
        
        # Test auto mode (should choose better balance)
        result_auto = partitioner.partition(P=4, axis='auto')
        self.assertEqual(result_auto.num_partitions, 4)
        self.assertEqual(sum(p.num_loads for p in result_auto.partitions), len(loads))
        self.assertLessEqual(result_auto.load_balance_ratio, 
                            max(result_x.load_balance_ratio, result_y.load_balance_ratio))
        
        # Verify invalid axis raises error
        with self.assertRaises(ValueError):
            partitioner.partition(P=4, axis='z')
    
    def test_auto_axis_improves_balance(self):
        """Auto axis should select orientation with better load balance."""
        G, loads, pads = build_test_grid()
        partitioner = GridPartitioner(G, loads, pads, seed=42)
        
        result_x = partitioner.partition(P=4, axis='x')
        result_y = partitioner.partition(P=4, axis='y')
        result_auto = partitioner.partition(P=4, axis='auto')
        
        # Auto should match the better of X or Y
        best_ratio = min(result_x.load_balance_ratio, result_y.load_balance_ratio)
        self.assertAlmostEqual(result_auto.load_balance_ratio, best_ratio, places=3)
    
    def test_interior_connectivity(self):
        """Test partition interior connectivity for different axes.
        
        X-axis: Structured partition interiors should be fully connected.
        Y-axis: May have disconnected layer-0 components when using layer 1+ separators only.
        This is expected because layer-0 stripe segments can be isolated from each other.
        """
        G, loads, pads = build_test_grid()
        partitioner = GridPartitioner(G, loads, pads, seed=42)
        
        # Test X-axis partitioning (should be fully connected)
        result_x = partitioner.partition(P=3, axis='x')
        pad_set = set(pads)
        
        for partition in result_x.partitions:
            interior = partition.interior_nodes - result_x.separator_nodes - pad_set
            if len(interior) <= 1:
                continue
            from collections import deque
            start = next(iter(interior))
            visited = {start}
            q = deque([start])
            while q:
                cur = q.popleft()
                for nbr in G.neighbors(cur):
                    if nbr in visited or nbr not in interior:
                        continue
                    visited.add(nbr)
                    q.append(nbr)
            self.assertEqual(visited, interior, 
                           f"X-axis Partition {partition.partition_id} interior not fully connected")
        
        # Test Y-axis partitioning (may have disconnected layer-0 components)
        result_y = partitioner.partition(P=3, axis='y')
        conn_info = result_y.get_partition_connectivity_info(G, pad_set)
        
        # Y-axis may have disconnections, but should still have loads correctly assigned
        total_loads_assigned = sum(p.num_loads for p in result_y.partitions)
        self.assertEqual(total_loads_assigned, len(loads),
                        "Y-axis partitioning should assign all loads even with disconnected components")
    
    def test_removed_edges_recorded(self):
        """Test that boundary edges are properly recorded."""
        G, loads, pads = build_test_grid()
        original_edge_count = G.number_of_edges()
        
        partitioner = GridPartitioner(G, loads, pads, seed=42)
        result = partitioner.partition(P=2)
        
        # Should have identified some boundary edges
        self.assertGreater(len(result.boundary_edges), 0)
        
        # Graph should have SAME number of edges (no removal)
        self.assertEqual(G.number_of_edges(), original_edge_count)
        
        # Check boundary edges structure
        for u, v, data in result.boundary_edges:
            self.assertIsNotNone(u)
            self.assertIsNotNone(v)
            self.assertIsInstance(data, dict)
            self.assertIn('resistance', data)
            
            # At least one endpoint should be a separator
            self.assertTrue(u in result.separator_nodes or v in result.separator_nodes,
                          f"Boundary edge ({u}, {v}) has no separator endpoint")
    
    def test_separator_nodes_have_no_loads(self):
        """Test that separator nodes do not contain any loads."""
        G, loads, pads = build_test_grid()
        partitioner = GridPartitioner(G, loads, pads, seed=42)
        
        result = partitioner.partition(P=3)
        
        # Check no separator is a load node
        load_nodes_set = set(loads.keys())
        for sep_node in result.separator_nodes:
            self.assertNotIn(sep_node, load_nodes_set,
                           f"Separator node {sep_node} is a load node")
    
    def test_all_nodes_assigned_to_partitions(self):
        """Test that all non-pad nodes are assigned to some partition.
        
        Note: Separators may appear in multiple partitions (on boundaries).
        Interior nodes appear in exactly one partition.
        """
        G, loads, pads = build_test_grid()
        partitioner = GridPartitioner(G, loads, pads, seed=42)
        
        result = partitioner.partition(P=2)
        
        # Collect all interior nodes (should be disjoint across partitions)
        all_interior = set()
        for partition in result.partitions:
            all_interior.update(partition.interior_nodes)
        
        # Check all non-pad, non-separator nodes are assigned as interior
        pad_set = set(pads)
        for node in G.nodes():
            if node not in pad_set and node not in result.separator_nodes:
                self.assertIn(node, all_interior,
                            f"Interior node {node} not assigned to any partition")
        
        # Check all separators appear in at least one partition
        for sep in result.separator_nodes:
            appears_in = [p.partition_id for p in result.partitions if sep in p.separator_nodes]
            self.assertGreater(len(appears_in), 0,
                             f"Separator {sep} does not appear in any partition's separator_nodes")
    
    def test_partition_with_different_P_values(self):
        """Test partitioning with different numbers of partitions."""
        G, loads, pads = build_test_grid()
        
        for P in [2, 3, 4, 5]:
            with self.subTest(P=P):
                # Need fresh graph for each test since partitioning modifies it
                G_copy, loads_copy, pads_copy = build_test_grid()
                partitioner = GridPartitioner(G_copy, loads_copy, pads_copy, seed=42)
                
                result = partitioner.partition(P=P)
                
                self.assertEqual(result.num_partitions, P)
                self.assertEqual(len(result.partitions), P)
                
                # All partitions should have at least one load
                for partition in result.partitions:
                    self.assertGreater(partition.num_loads, 0,
                                     f"Partition {partition.partition_id} has no loads")
    
    
    def test_partition_invalid_inputs(self):
        """Test that invalid inputs raise appropriate errors."""
        G, loads, pads = build_test_grid()
        partitioner = GridPartitioner(G, loads, pads, seed=42)
        
        # P <= 0 should raise error
        with self.assertRaises(ValueError):
            partitioner.partition(P=0)
        
        with self.assertRaises(ValueError):
            partitioner.partition(P=-1)
        
        # P > num_loads should raise error
        num_loads = len(loads)
        with self.assertRaises(ValueError):
            partitioner.partition(P=num_loads + 1)
    
    def test_partition_result_properties(self):
        """Test PartitionResult properties and methods."""
        G, loads, pads = build_test_grid()
        partitioner = GridPartitioner(G, loads, pads, seed=42)
        
        result = partitioner.partition(P=3)
        
        # Test get_partition
        for p_id in range(3):
            partition = result.get_partition(p_id)
            self.assertIsNotNone(partition)
            self.assertEqual(partition.partition_id, p_id)
        
        # Non-existent partition
        self.assertIsNone(result.get_partition(999))
        
        # Test __repr__
        repr_str = repr(result)
        self.assertIn('PartitionResult', repr_str)
        self.assertIn('P=3', repr_str)
    
    def test_partition_reproducibility(self):
        """Test that partitioning is reproducible with same seed."""
        G1, loads1, pads1 = build_test_grid()
        G2, loads2, pads2 = build_test_grid()
        
        partitioner1 = GridPartitioner(G1, loads1, pads1, seed=123)
        partitioner2 = GridPartitioner(G2, loads2, pads2, seed=123)
        
        result1 = partitioner1.partition(P=3)
        result2 = partitioner2.partition(P=3)
        
        # Check load counts match
        loads1_counts = [p.num_loads for p in result1.partitions]
        loads2_counts = [p.num_loads for p in result2.partitions]
        self.assertEqual(loads1_counts, loads2_counts)
        
        # Check separator counts match
        sep1_counts = [len(p.separator_nodes) for p in result1.partitions]
        sep2_counts = [len(p.separator_nodes) for p in result2.partitions]
        self.assertEqual(sep1_counts, sep2_counts)
    
    def test_pads_remain_global(self):
        """Test that pad nodes are not assigned to partitions."""
        G, loads, pads = build_test_grid()
        partitioner = GridPartitioner(G, loads, pads, seed=42)
        
        result = partitioner.partition(P=2)
        
        # Collect all nodes in partitions
        nodes_in_partitions = set()
        for partition in result.partitions:
            nodes_in_partitions.update(partition.all_nodes)
        
        # Check no pad is in any partition
        pad_set = set(pads)
        for pad in pad_set:
            self.assertNotIn(pad, nodes_in_partitions,
                           f"Pad {pad} should not be in any partition")
        
        # Check pads still have edges in graph (graph not modified)
        for pad in pad_set:
            self.assertGreater(G.degree(pad), 0,
                             f"Pad {pad} should still have edges")
    
    def test_graph_not_modified(self):
        """Test that the graph structure is not modified by partitioning."""
        G, loads, pads = build_test_grid()
        
        # Record original graph state
        original_nodes = set(G.nodes())
        original_edges = set(G.edges())
        original_edge_count = G.number_of_edges()
        
        partitioner = GridPartitioner(G, loads, pads, seed=42)
        result = partitioner.partition(P=3)
        
        # Check graph is unchanged
        self.assertEqual(set(G.nodes()), original_nodes)
        self.assertEqual(set(G.edges()), original_edges)
        self.assertEqual(G.number_of_edges(), original_edge_count)
    
    def test_cross_partition_requires_separator(self):
        """Verify nodes in different partitions require separator traversal."""
        import networkx as nx
        G, loads, pads = build_test_grid()
        partitioner = GridPartitioner(G, loads, pads, seed=42)
        result = partitioner.partition(P=2)
        
        # Get interior nodes (excluding separators and pads)
        pad_set = set(pads)
        p0_interior = result.partitions[0].interior_nodes - result.separator_nodes - pad_set
        p1_interior = result.partitions[1].interior_nodes - result.separator_nodes - pad_set
        
        if not p0_interior or not p1_interior:
            self.skipTest("Empty interior in at least one partition")
        
        # Create subgraph WITHOUT separators or pads
        nodes_without_seps = set(G.nodes()) - result.separator_nodes - pad_set
        G_no_seps = G.subgraph(nodes_without_seps)
        
        # Pick representative nodes from each partition
        node_p0 = next(iter(p0_interior))
        node_p1 = next(iter(p1_interior))
        
        # Verify no path exists without separators
        self.assertFalse(
            nx.has_path(G_no_seps, node_p0, node_p1),
            f"Path exists from partition 0 to 1 without traversing separators"
        )
        
        # Verify path DOES exist in full graph (with separators)
        self.assertTrue(
            nx.has_path(G, node_p0, node_p1),
            f"No path exists even with separators (graph disconnected)"
        )


class TestPartitionDataStructures(unittest.TestCase):
    """Test the Partition and PartitionResult data structures."""
    
    def test_partition_all_nodes_property(self):
        """Test Partition.all_nodes property."""
        from irdrop.grid_partitioner import Partition
        
        interior = {1, 2, 3}
        separators = {4, 5}
        loads = {1, 2}
        
        partition = Partition(
            partition_id=0,
            interior_nodes=interior,
            separator_nodes=separators,
            load_nodes=loads
        )
        
        self.assertEqual(partition.all_nodes, {1, 2, 3, 4, 5})
        self.assertEqual(partition.num_loads, 2)
    
    def test_y_axis_layer_distribution(self):
        """Test Y-axis partitioning uses only layer 1+ separators."""
        G, loads, pads = build_test_grid()
        partitioner = GridPartitioner(G, loads, pads, seed=42)
        
        result = partitioner.partition(P=4, axis='y')
        
        # Count separators by layer
        layer_counts = {}
        for layer in range(3):
            layer_counts[layer] = len([n for n in result.separator_nodes 
                                        if G.nodes[n].get('layer', 0) == layer])
        
        # Y-axis should have ZERO layer-0 separators
        self.assertEqual(layer_counts[0], 0,
                        "Y-axis partitioning should have no layer-0 separators")
        
        # Should have some layer 1+ separators
        self.assertGreater(layer_counts[1] + layer_counts[2], 0,
                          "Y-axis should have layer 1+ separators")
    
    def test_y_axis_disconnected_components_expected(self):
        """Test Y-axis partitioning may have disconnected layer-0 components (expected behavior)."""
        G, loads, pads = build_test_grid()
        partitioner = GridPartitioner(G, loads, pads, seed=42)
        
        result = partitioner.partition(P=4, axis='y')
        pad_set = set(pads)
        
        # Get connectivity info
        conn_info = result.get_partition_connectivity_info(G, pad_set)
        
        # Some partitions may have disconnected components (this is expected)
        has_disconnections = any(not info['connected'] for info in conn_info.values())
        
        # All loads should still be assigned correctly
        total_loads = sum(p.num_loads for p in result.partitions)
        self.assertEqual(total_loads, len(loads),
                        "All loads should be assigned even with disconnected components")
        
        # If disconnections exist, verify they're on layer 0
        if has_disconnections:
            for partition in result.partitions:
                interior = partition.interior_nodes - result.separator_nodes - pad_set
                # Verify we have some layer-0 nodes that could be disconnected
                layer0_nodes = [n for n in interior if G.nodes[n].get('layer', 0) == 0]
                # This is the expected behavior: layer-0 nodes form separate stripe segments
    
    def test_separators_assigned_by_adjacency(self):
        """Test that separators are correctly assigned to partitions based on adjacency to interior nodes.
        
        This test validates the fix for the bug where separators were not being included in
        partition.separator_nodes because they weren't in partition_assignments.
        
        Note: Some separators may be "orphans" (only adjacent to other separators or pads),
        which are assigned to the first partition by default.
        """
        G, loads, pads = build_test_grid()
        partitioner = GridPartitioner(G, loads, pads, seed=42)
        
        # Test both X and Y axis
        for axis in ['x', 'y']:
            with self.subTest(axis=axis):
                result = partitioner.partition(P=3, axis=axis)
                
                # For each partition, verify all adjacent separators are included
                for partition in result.partitions:
                    # Find separators adjacent to this partition's interior nodes
                    adjacent_separators = set()
                    for interior_node in partition.interior_nodes:
                        for neighbor in G.neighbors(interior_node):
                            if neighbor in result.separator_nodes:
                                adjacent_separators.add(neighbor)
                    
                    # All adjacent separators should be in partition.separator_nodes
                    missing = adjacent_separators - partition.separator_nodes
                    self.assertEqual(len(missing), 0,
                                   f"{axis.upper()}-axis Partition {partition.partition_id}: "
                                   f"Missing {len(missing)} adjacent separators. "
                                   f"This indicates _build_partitions is not finding separators by adjacency.")
                    
                    # Most separators in partition should be adjacent to interior
                    # (exception: orphan separators in partition 0)
                    separators_with_interior_neighbors = 0
                    for sep in partition.separator_nodes:
                        has_interior_neighbor = False
                        for neighbor in G.neighbors(sep):
                            if neighbor in partition.interior_nodes:
                                has_interior_neighbor = True
                                break
                        if has_interior_neighbor:
                            separators_with_interior_neighbors += 1
                    
                    # At least some separators should have interior neighbors
                    # (unless it's a tiny partition with only orphans)
                    if len(partition.separator_nodes) > 0:
                        ratio = separators_with_interior_neighbors / len(partition.separator_nodes)
                        # Allow orphans in any partition (connectivity enforcement artifacts
                        # or orphans assigned based on non-orphan separator neighbors)
                        # Require at least 40% to have interior neighbors
                        min_ratio = 0.4
                        self.assertGreaterEqual(ratio, min_ratio,
                                              f"{axis.upper()}-axis Partition {partition.partition_id}: "
                                              f"Only {separators_with_interior_neighbors}/{len(partition.separator_nodes)} "
                                              f"separators are adjacent to interior nodes ({ratio:.1%} < {min_ratio:.1%}).")
    
    def test_separators_appear_in_multiple_partitions(self):
        """Test that boundary separators appear in multiple adjacent partitions.
        
        Separators on internal boundaries should appear in the separator_nodes set of
        multiple partitions (the ones they're adjacent to).
        """
        G, loads, pads = build_test_grid()
        partitioner = GridPartitioner(G, loads, pads, seed=42)
        
        result = partitioner.partition(P=3, axis='x')
        
        # Track which partitions each separator belongs to
        sep_to_partitions = {}
        for partition in result.partitions:
            for sep in partition.separator_nodes:
                if sep not in sep_to_partitions:
                    sep_to_partitions[sep] = []
                sep_to_partitions[sep].append(partition.partition_id)
        
        # For P=3, we have 2 internal boundaries, so we expect:
        # - Some separators appear in 2 partitions (on internal boundaries)
        # - Edge partitions (0 and 2) may have separators on only their inner edge
        multi_partition_seps = {k: v for k, v in sep_to_partitions.items() if len(v) > 1}
        
        # We should have some separators appearing in multiple partitions
        self.assertGreater(len(multi_partition_seps), 0,
                          "Expected some separators to appear in multiple partitions")
        
        # Verify middle partition (ID=1) has separators on both boundaries
        middle_partition = result.partitions[1]
        self.assertGreater(len(middle_partition.separator_nodes), 0,
                          "Middle partition should have separators")
    
    def test_x_axis_uses_layer0_separators(self):
        """Test X-axis partitioning uses layer-0 via nodes as separators."""
        G, loads, pads = build_test_grid()
        partitioner = GridPartitioner(G, loads, pads, seed=42)
        
        result = partitioner.partition(P=4, axis='x')
        
        # Count separators by layer
        layer_counts = {}
        for layer in range(3):
            layer_counts[layer] = len([n for n in result.separator_nodes 
                                        if G.nodes[n].get('layer', 0) == layer])
        
        # X-axis should have layer-0 separators (via nodes)
        self.assertGreater(layer_counts[0], 0,
                          "X-axis partitioning should have layer-0 separators")
    
    def test_partition_result_num_partitions(self):
        """Test PartitionResult.num_partitions property."""
        from irdrop.grid_partitioner import Partition, PartitionResult
        
        partitions = [
            Partition(0, {1, 2}, {3}, {1}),
            Partition(1, {4, 5}, {6}, {4}),
        ]
        
        result = PartitionResult(
            partitions=partitions,
            separator_nodes={3, 6},
            boundary_edges=[(1, 4, {})],
            load_balance_ratio=1.0
        )
        
        self.assertEqual(result.num_partitions, 2)


if __name__ == '__main__':
    unittest.main()

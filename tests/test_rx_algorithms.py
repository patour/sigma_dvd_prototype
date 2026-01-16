"""Unit tests for custom rustworkx algorithms.

Tests that contract_nodes, node_connected_component, and other
algorithms match NetworkX behavior.
"""

import unittest
from core.rx_graph import RustworkxGraphWrapper, RustworkxMultiDiGraphWrapper
from core.rx_algorithms import (
    contract_nodes,
    node_connected_component,
    connected_components,
    is_connected,
    number_connected_components,
)


class TestContractNodesUndirected(unittest.TestCase):
    """Test contract_nodes on undirected graphs."""

    def test_contract_basic(self):
        """Contract two adjacent nodes."""
        g = RustworkxGraphWrapper()
        g.add_edge('a', 'b', weight=1.0)
        g.add_edge('b', 'c', weight=2.0)
        g.add_edge('c', 'd', weight=3.0)

        # Contract b into a
        contract_nodes(g, 'a', 'b')

        self.assertIn('a', g)
        self.assertNotIn('b', g)
        self.assertTrue(g.has_edge('a', 'c'))
        self.assertEqual(g.number_of_nodes(), 3)

    def test_contract_preserves_edges(self):
        """All edges from removed node redirect to kept node."""
        g = RustworkxGraphWrapper()
        g.add_edge('center', 'a')
        g.add_edge('center', 'b')
        g.add_edge('center', 'c')
        g.add_edge('keep', 'd')

        contract_nodes(g, 'keep', 'center')

        self.assertTrue(g.has_edge('keep', 'a'))
        self.assertTrue(g.has_edge('keep', 'b'))
        self.assertTrue(g.has_edge('keep', 'c'))
        self.assertTrue(g.has_edge('keep', 'd'))

    def test_contract_no_self_loops_default(self):
        """Self-loops not created by default."""
        g = RustworkxGraphWrapper()
        g.add_edge('a', 'b')

        contract_nodes(g, 'a', 'b', self_loops=False)

        # Should not have self-loop on 'a'
        self.assertFalse(g.has_edge('a', 'a'))

    def test_contract_with_self_loops(self):
        """Self-loops created when requested."""
        g = RustworkxGraphWrapper()
        g.add_edge('a', 'b')

        contract_nodes(g, 'a', 'b', self_loops=True)

        # Should have self-loop on 'a'
        self.assertTrue(g.has_edge('a', 'a'))

    def test_contract_merges_node_attrs(self):
        """Node attributes are merged, keep_node takes precedence."""
        g = RustworkxGraphWrapper()
        g.add_node('keep', x=100, y=200)
        g.add_node('remove', x=999, z=300)
        g.add_edge('keep', 'remove')

        contract_nodes(g, 'keep', 'remove')

        attrs = g.nodes_dict['keep']
        self.assertEqual(attrs['x'], 100)  # keep_node wins
        self.assertEqual(attrs['y'], 200)
        self.assertEqual(attrs['z'], 300)  # inherited from remove

    def test_contract_star_topology(self):
        """Contract center of star topology."""
        g = RustworkxGraphWrapper()
        # Create star: center connected to a, b, c, d
        for n in ['a', 'b', 'c', 'd']:
            g.add_edge('center', n)

        contract_nodes(g, 'a', 'center')

        # 'a' should now be connected to b, c, d
        self.assertEqual(g.number_of_nodes(), 4)
        for n in ['b', 'c', 'd']:
            self.assertTrue(g.has_edge('a', n))

    def test_contract_non_adjacent_nodes(self):
        """Contract non-adjacent nodes."""
        g = RustworkxGraphWrapper()
        g.add_edge('a', 'b')
        g.add_edge('c', 'd')

        contract_nodes(g, 'a', 'c')

        # 'a' should now have 'c's neighbor 'd'
        self.assertTrue(g.has_edge('a', 'd'))
        self.assertNotIn('c', g)

    def test_contract_missing_keep_node_raises(self):
        """Raises KeyError if keep_node not in graph."""
        g = RustworkxGraphWrapper()
        g.add_node('a')

        with self.assertRaises(KeyError):
            contract_nodes(g, 'missing', 'a')

    def test_contract_missing_remove_node_raises(self):
        """Raises KeyError if remove_node not in graph."""
        g = RustworkxGraphWrapper()
        g.add_node('a')

        with self.assertRaises(KeyError):
            contract_nodes(g, 'a', 'missing')


class TestContractNodesDirected(unittest.TestCase):
    """Test contract_nodes on directed graphs."""

    def test_contract_directed_preserves_direction(self):
        """Directed edges maintain direction after contraction."""
        g = RustworkxMultiDiGraphWrapper()
        g.add_edge('a', 'b')  # a -> b
        g.add_edge('b', 'c')  # b -> c
        g.add_edge('d', 'b')  # d -> b

        contract_nodes(g, 'a', 'b')

        # a -> c (was b -> c)
        self.assertTrue(g.has_edge('a', 'c'))
        # d -> a (was d -> b)
        self.assertTrue(g.has_edge('d', 'a'))

    def test_contract_directed_incoming_outgoing(self):
        """Both incoming and outgoing edges are redirected."""
        g = RustworkxMultiDiGraphWrapper()
        g.add_node('keep')  # keep_node must exist
        g.add_edge('x', 'remove')  # incoming to remove
        g.add_edge('remove', 'y')  # outgoing from remove

        contract_nodes(g, 'keep', 'remove')

        self.assertTrue(g.has_edge('x', 'keep'))
        self.assertTrue(g.has_edge('keep', 'y'))

    def test_contract_directed_no_self_loop(self):
        """No self-loop when contracting adjacent directed nodes."""
        g = RustworkxMultiDiGraphWrapper()
        g.add_edge('a', 'b')

        contract_nodes(g, 'a', 'b', self_loops=False)

        self.assertFalse(g.has_edge('a', 'a'))


class TestNodeConnectedComponent(unittest.TestCase):
    """Test node_connected_component function."""

    def test_single_component(self):
        """All nodes in single component."""
        g = RustworkxGraphWrapper()
        g.add_edge('a', 'b')
        g.add_edge('b', 'c')
        g.add_edge('c', 'd')

        component = node_connected_component(g, 'b')
        self.assertEqual(component, {'a', 'b', 'c', 'd'})

    def test_multiple_components(self):
        """Correctly identifies separate components."""
        g = RustworkxGraphWrapper()
        g.add_edge('a', 'b')
        g.add_edge('c', 'd')  # Separate component

        comp_a = node_connected_component(g, 'a')
        comp_c = node_connected_component(g, 'c')

        self.assertEqual(comp_a, {'a', 'b'})
        self.assertEqual(comp_c, {'c', 'd'})

    def test_isolated_node(self):
        """Isolated node returns singleton set."""
        g = RustworkxGraphWrapper()
        g.add_node('isolated')
        g.add_edge('a', 'b')

        component = node_connected_component(g, 'isolated')
        self.assertEqual(component, {'isolated'})

    def test_node_not_in_graph_raises(self):
        """Raises KeyError for missing node."""
        g = RustworkxGraphWrapper()
        g.add_node('a')

        with self.assertRaises(KeyError):
            node_connected_component(g, 'missing')

    def test_directed_graph_weak_connectivity(self):
        """Directed graph uses weak connectivity."""
        g = RustworkxMultiDiGraphWrapper()
        g.add_edge('a', 'b')  # a -> b
        g.add_edge('c', 'b')  # c -> b

        # All three should be in same weakly connected component
        comp = node_connected_component(g, 'a')
        self.assertEqual(comp, {'a', 'b', 'c'})


class TestConnectedComponents(unittest.TestCase):
    """Test connected_components function."""

    def test_single_component(self):
        """Single connected component."""
        g = RustworkxGraphWrapper()
        g.add_edge('a', 'b')
        g.add_edge('b', 'c')

        comps = connected_components(g)
        self.assertEqual(len(comps), 1)
        self.assertEqual(comps[0], {'a', 'b', 'c'})

    def test_multiple_components(self):
        """Multiple connected components."""
        g = RustworkxGraphWrapper()
        g.add_edge('a', 'b')
        g.add_edge('c', 'd')
        g.add_edge('e', 'f')

        comps = connected_components(g)
        self.assertEqual(len(comps), 3)

        # Check each component
        all_nodes = set()
        for comp in comps:
            all_nodes.update(comp)
        self.assertEqual(all_nodes, {'a', 'b', 'c', 'd', 'e', 'f'})

    def test_isolated_nodes(self):
        """Isolated nodes are their own components."""
        g = RustworkxGraphWrapper()
        g.add_node('isolated1')
        g.add_node('isolated2')
        g.add_edge('a', 'b')

        comps = connected_components(g)
        self.assertEqual(len(comps), 3)

    def test_empty_graph(self):
        """Empty graph has no components."""
        g = RustworkxGraphWrapper()
        comps = connected_components(g)
        self.assertEqual(len(comps), 0)


class TestIsConnected(unittest.TestCase):
    """Test is_connected function."""

    def test_connected_graph(self):
        """Connected graph returns True."""
        g = RustworkxGraphWrapper()
        g.add_edge('a', 'b')
        g.add_edge('b', 'c')

        self.assertTrue(is_connected(g))

    def test_disconnected_graph(self):
        """Disconnected graph returns False."""
        g = RustworkxGraphWrapper()
        g.add_edge('a', 'b')
        g.add_edge('c', 'd')

        self.assertFalse(is_connected(g))

    def test_single_node(self):
        """Single node is connected."""
        g = RustworkxGraphWrapper()
        g.add_node('a')

        self.assertTrue(is_connected(g))

    def test_empty_graph(self):
        """Empty graph is considered connected."""
        g = RustworkxGraphWrapper()
        self.assertTrue(is_connected(g))


class TestNumberConnectedComponents(unittest.TestCase):
    """Test number_connected_components function."""

    def test_count_components(self):
        """Count connected components."""
        g = RustworkxGraphWrapper()
        g.add_edge('a', 'b')
        g.add_edge('c', 'd')
        g.add_node('isolated')

        self.assertEqual(number_connected_components(g), 3)

    def test_single_component(self):
        """Single component returns 1."""
        g = RustworkxGraphWrapper()
        g.add_edge('a', 'b')
        g.add_edge('b', 'c')

        self.assertEqual(number_connected_components(g), 1)


class TestContractNodesEdgeCases(unittest.TestCase):
    """Edge case tests for contract_nodes."""

    def test_contract_preserves_other_edges(self):
        """Edges not involving contracted nodes are preserved."""
        g = RustworkxGraphWrapper()
        g.add_edge('a', 'b')
        g.add_edge('c', 'd')
        g.add_edge('d', 'e')

        contract_nodes(g, 'c', 'd')

        # Edge d-e should now be c-e
        self.assertTrue(g.has_edge('c', 'e'))
        # Edge a-b should be unchanged
        self.assertTrue(g.has_edge('a', 'b'))

    def test_contract_chain(self):
        """Contract multiple times in a chain."""
        g = RustworkxGraphWrapper()
        g.add_edge('a', 'b')
        g.add_edge('b', 'c')
        g.add_edge('c', 'd')

        contract_nodes(g, 'a', 'b')
        contract_nodes(g, 'a', 'c')

        self.assertEqual(g.number_of_nodes(), 2)
        self.assertTrue(g.has_edge('a', 'd'))

    def test_contract_with_edge_attrs(self):
        """Edge attributes are preserved during contraction."""
        g = RustworkxGraphWrapper()
        g.add_edge('a', 'b', weight=1.0)
        g.add_edge('b', 'c', weight=2.0)

        contract_nodes(g, 'a', 'b')

        # The b-c edge (now a-c) should have its attributes
        data = g.get_edge_data('a', 'c')
        self.assertEqual(data['weight'], 2.0)


if __name__ == '__main__':
    unittest.main()

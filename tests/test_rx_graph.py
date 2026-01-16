"""Unit tests for rustworkx graph wrappers.

Tests that RustworkxGraphWrapper and RustworkxMultiDiGraphWrapper
provide NetworkX-compatible API behavior.
"""

import unittest
from core.rx_graph import RustworkxGraphWrapper, RustworkxMultiDiGraphWrapper


class TestRustworkxGraphWrapperNodes(unittest.TestCase):
    """Test node operations on undirected graph wrapper."""

    def test_add_node_basic(self):
        """Add nodes to empty graph."""
        g = RustworkxGraphWrapper()
        g.add_node('a')
        g.add_node('b')

        self.assertIn('a', g)
        self.assertIn('b', g)
        self.assertEqual(g.number_of_nodes(), 2)

    def test_add_node_with_attrs(self):
        """Add node with attributes."""
        g = RustworkxGraphWrapper()
        g.add_node('a', x=100, y=200, layer='M1')

        self.assertIn('a', g)
        self.assertEqual(g.nodes_dict['a']['x'], 100)
        self.assertEqual(g.nodes_dict['a']['y'], 200)
        self.assertEqual(g.nodes_dict['a']['layer'], 'M1')

    def test_add_duplicate_node_merges_attrs(self):
        """Adding existing node updates attributes."""
        g = RustworkxGraphWrapper()
        g.add_node('a', x=100)
        g.add_node('a', y=200)

        self.assertEqual(g.number_of_nodes(), 1)
        self.assertEqual(g.nodes_dict['a']['x'], 100)
        self.assertEqual(g.nodes_dict['a']['y'], 200)

    def test_remove_node(self):
        """Remove node from graph."""
        g = RustworkxGraphWrapper()
        g.add_node('a')
        g.add_node('b')
        g.remove_node('a')

        self.assertNotIn('a', g)
        self.assertIn('b', g)
        self.assertEqual(g.number_of_nodes(), 1)

    def test_remove_node_removes_edges(self):
        """Removing node also removes incident edges."""
        g = RustworkxGraphWrapper()
        g.add_edge('a', 'b')
        g.add_edge('a', 'c')
        g.remove_node('a')

        self.assertFalse(g.has_edge('a', 'b'))
        self.assertFalse(g.has_edge('a', 'c'))
        self.assertEqual(g.number_of_edges(), 0)

    def test_has_node(self):
        """Check node existence."""
        g = RustworkxGraphWrapper()
        g.add_node('a')

        self.assertTrue(g.has_node('a'))
        self.assertFalse(g.has_node('b'))

    def test_nodes_iteration(self):
        """Iterate over nodes without data."""
        g = RustworkxGraphWrapper()
        g.add_node('a')
        g.add_node('b')
        g.add_node('c')

        nodes = set(g.nodes())
        self.assertEqual(nodes, {'a', 'b', 'c'})

    def test_nodes_iteration_with_data(self):
        """Iterate over nodes with data."""
        g = RustworkxGraphWrapper()
        g.add_node('a', x=1)
        g.add_node('b', x=2)

        nodes_data = dict(g.nodes(data=True))
        self.assertEqual(nodes_data['a']['x'], 1)
        self.assertEqual(nodes_data['b']['x'], 2)


class TestRustworkxGraphWrapperEdges(unittest.TestCase):
    """Test edge operations on undirected graph wrapper."""

    def test_add_edge_basic(self):
        """Add edge between two nodes."""
        g = RustworkxGraphWrapper()
        g.add_edge('a', 'b')

        self.assertTrue(g.has_edge('a', 'b'))
        self.assertTrue(g.has_edge('b', 'a'))  # Undirected

    def test_add_edge_creates_nodes(self):
        """Adding edge auto-creates nodes."""
        g = RustworkxGraphWrapper()
        g.add_edge('a', 'b')

        self.assertIn('a', g)
        self.assertIn('b', g)

    def test_add_edge_with_attrs(self):
        """Add edge with attributes."""
        g = RustworkxGraphWrapper()
        g.add_edge('a', 'b', resistance=1.5, kind='stripe')

        data = g.get_edge_data('a', 'b')
        self.assertEqual(data['resistance'], 1.5)
        self.assertEqual(data['kind'], 'stripe')

    def test_remove_edge(self):
        """Remove edge from graph."""
        g = RustworkxGraphWrapper()
        g.add_edge('a', 'b')
        g.remove_edge('a', 'b')

        self.assertFalse(g.has_edge('a', 'b'))
        # Nodes should still exist
        self.assertIn('a', g)
        self.assertIn('b', g)

    def test_edges_iteration(self):
        """Iterate over edges without data."""
        g = RustworkxGraphWrapper()
        g.add_edge('a', 'b')
        g.add_edge('b', 'c')

        edges = list(g.edges())
        self.assertEqual(len(edges), 2)

    def test_edges_iteration_with_data(self):
        """Iterate over edges with data."""
        g = RustworkxGraphWrapper()
        g.add_edge('a', 'b', weight=1.0)
        g.add_edge('b', 'c', weight=2.0)

        edges = list(g.edges(data=True))
        self.assertEqual(len(edges), 2)

        # Check data is present
        for u, v, data in edges:
            self.assertIn('weight', data)

    def test_edges_for_node(self):
        """Get edges for specific node."""
        g = RustworkxGraphWrapper()
        g.add_edge('a', 'b')
        g.add_edge('a', 'c')
        g.add_edge('b', 'd')

        a_edges = list(g.edges(nbunch='a'))
        self.assertEqual(len(a_edges), 2)

    def test_get_edge_data(self):
        """Get edge data for specific edge."""
        g = RustworkxGraphWrapper()
        g.add_edge('a', 'b', weight=5.0)

        data = g.get_edge_data('a', 'b')
        self.assertEqual(data['weight'], 5.0)

        # Non-existent edge returns None
        self.assertIsNone(g.get_edge_data('a', 'c'))


class TestRustworkxGraphWrapperNeighbors(unittest.TestCase):
    """Test neighbor/adjacency operations."""

    def test_neighbors(self):
        """Get neighbors of a node."""
        g = RustworkxGraphWrapper()
        g.add_edge('a', 'b')
        g.add_edge('a', 'c')
        g.add_edge('a', 'd')

        neighbors = set(g.neighbors('a'))
        self.assertEqual(neighbors, {'b', 'c', 'd'})

    def test_degree(self):
        """Get degree of a node."""
        g = RustworkxGraphWrapper()
        g.add_edge('a', 'b')
        g.add_edge('a', 'c')

        self.assertEqual(g.degree('a'), 2)
        self.assertEqual(g.degree('b'), 1)

    def test_adjacency_dict(self):
        """Access adjacency via graph[node]."""
        g = RustworkxGraphWrapper()
        g.add_edge('a', 'b', w=1)
        g.add_edge('a', 'c', w=2)

        adj = g['a']
        self.assertIn('b', adj)
        self.assertIn('c', adj)
        self.assertEqual(adj['b']['w'], 1)


class TestRustworkxGraphWrapperSubgraph(unittest.TestCase):
    """Test subgraph operations."""

    def test_subgraph_basic(self):
        """Extract subgraph with subset of nodes."""
        g = RustworkxGraphWrapper()
        g.add_edge('a', 'b')
        g.add_edge('b', 'c')
        g.add_edge('c', 'd')

        sub = g.subgraph({'a', 'b', 'c'})

        self.assertEqual(sub.number_of_nodes(), 3)
        self.assertEqual(sub.number_of_edges(), 2)
        self.assertNotIn('d', sub)
        self.assertFalse(sub.has_edge('c', 'd'))

    def test_subgraph_preserves_attrs(self):
        """Subgraph preserves node and edge attributes."""
        g = RustworkxGraphWrapper()
        g.add_node('a', x=100)
        g.add_node('b', x=200)
        g.add_edge('a', 'b', weight=5.0)

        sub = g.subgraph({'a', 'b'})

        self.assertEqual(sub.nodes_dict['a']['x'], 100)
        self.assertEqual(sub.get_edge_data('a', 'b')['weight'], 5.0)

    def test_subgraph_copies_metadata(self):
        """Subgraph copies graph-level metadata."""
        g = RustworkxGraphWrapper()
        g.graph['name'] = 'test_graph'
        g.add_node('a')

        sub = g.subgraph({'a'})
        self.assertEqual(sub.graph['name'], 'test_graph')

    def test_copy(self):
        """Make a complete copy of the graph."""
        g = RustworkxGraphWrapper()
        g.add_edge('a', 'b', w=1)
        g.graph['key'] = 'value'

        copy = g.copy()

        self.assertEqual(copy.number_of_nodes(), 2)
        self.assertTrue(copy.has_edge('a', 'b'))
        self.assertEqual(copy.graph['key'], 'value')


class TestRustworkxMultiDiGraphWrapperBasic(unittest.TestCase):
    """Test basic operations on directed multigraph wrapper."""

    def test_add_edge_directed(self):
        """Edges are directed."""
        g = RustworkxMultiDiGraphWrapper()
        g.add_edge('a', 'b')

        self.assertTrue(g.has_edge('a', 'b'))
        self.assertFalse(g.has_edge('b', 'a'))

    def test_multi_edges(self):
        """Multiple edges between same nodes."""
        g = RustworkxMultiDiGraphWrapper()
        g.add_edge('a', 'b', type='R', value=1.0)
        g.add_edge('a', 'b', type='C', value=100.0)

        edges = list(g.edges(data=True))
        self.assertEqual(len(edges), 2)

        # Verify both edge types exist
        types = {e[2]['type'] for e in edges}
        self.assertEqual(types, {'R', 'C'})

    def test_get_all_edge_data(self):
        """Get all edges between two nodes."""
        g = RustworkxMultiDiGraphWrapper()
        g.add_edge('a', 'b', type='R')
        g.add_edge('a', 'b', type='C')

        all_data = g.get_all_edge_data('a', 'b')
        self.assertEqual(len(all_data), 2)

    def test_in_edges(self):
        """Get incoming edges to a node."""
        g = RustworkxMultiDiGraphWrapper()
        g.add_edge('a', 'c')
        g.add_edge('b', 'c')

        in_edges = list(g.in_edges('c'))
        sources = {e[0] for e in in_edges}
        self.assertEqual(sources, {'a', 'b'})

    def test_out_edges(self):
        """Get outgoing edges from a node."""
        g = RustworkxMultiDiGraphWrapper()
        g.add_edge('a', 'b')
        g.add_edge('a', 'c')

        out_edges = list(g.out_edges('a'))
        targets = {e[1] for e in out_edges}
        self.assertEqual(targets, {'b', 'c'})

    def test_predecessors_successors(self):
        """Get predecessors and successors."""
        g = RustworkxMultiDiGraphWrapper()
        g.add_edge('a', 'b')
        g.add_edge('c', 'b')
        g.add_edge('b', 'd')

        preds = set(g.predecessors('b'))
        succs = set(g.successors('b'))

        self.assertEqual(preds, {'a', 'c'})
        self.assertEqual(succs, {'d'})

    def test_in_out_degree(self):
        """Get in-degree and out-degree."""
        g = RustworkxMultiDiGraphWrapper()
        g.add_edge('a', 'b')
        g.add_edge('c', 'b')
        g.add_edge('b', 'd')

        self.assertEqual(g.in_degree('b'), 2)
        self.assertEqual(g.out_degree('b'), 1)
        self.assertEqual(g.degree('b'), 3)

    def test_adjacency_dict_multiedge(self):
        """Adjacency dict with multi-edges."""
        g = RustworkxMultiDiGraphWrapper()
        g.add_edge('a', 'b', type='R')
        g.add_edge('a', 'b', type='C')

        adj = g['a']
        self.assertIn('b', adj)
        # Should have 2 edges with different keys
        self.assertEqual(len(adj['b']), 2)

    def test_to_undirected(self):
        """Convert to undirected graph."""
        g = RustworkxMultiDiGraphWrapper()
        g.add_edge('a', 'b')
        g.add_edge('b', 'a')  # Reverse edge

        undirected = g.to_undirected()

        self.assertIsInstance(undirected, RustworkxGraphWrapper)
        self.assertTrue(undirected.has_edge('a', 'b'))
        # Should have only 1 edge (duplicates merged)
        self.assertEqual(undirected.number_of_edges(), 1)

    def test_edges_with_keys(self):
        """Iterate edges with keys."""
        g = RustworkxMultiDiGraphWrapper()
        g.add_edge('a', 'b', val=1)
        g.add_edge('a', 'b', val=2)

        edges = list(g.edges(keys=True, data=True))
        self.assertEqual(len(edges), 2)

        # Each edge should have (u, v, key, data)
        for u, v, key, data in edges:
            self.assertEqual(u, 'a')
            self.assertEqual(v, 'b')
            self.assertIsInstance(key, int)
            self.assertIn('val', data)


class TestGraphMetadata(unittest.TestCase):
    """Test graph-level metadata (replaces nx graph.graph)."""

    def test_undirected_metadata(self):
        """Undirected graph metadata access."""
        g = RustworkxGraphWrapper()
        g.graph['net_connectivity'] = {'VDD': ['n1', 'n2']}
        g.graph['parameters'] = {'VDD': 1.0}

        self.assertEqual(g.graph['net_connectivity'], {'VDD': ['n1', 'n2']})
        self.assertEqual(g.graph['parameters']['VDD'], 1.0)

    def test_directed_metadata(self):
        """Directed graph metadata access."""
        g = RustworkxMultiDiGraphWrapper()
        g.graph['vsrc_nodes'] = {'v1', 'v2'}

        self.assertEqual(g.graph['vsrc_nodes'], {'v1', 'v2'})


class TestIndexAccess(unittest.TestCase):
    """Test direct index access for low-level operations."""

    def test_get_index_undirected(self):
        """Get rustworkx index for node."""
        g = RustworkxGraphWrapper()
        g.add_node('a')
        g.add_node('b')

        idx_a = g.get_index('a')
        idx_b = g.get_index('b')

        self.assertIsNotNone(idx_a)
        self.assertIsNotNone(idx_b)
        self.assertNotEqual(idx_a, idx_b)

    def test_get_node_from_index(self):
        """Get node key from rustworkx index."""
        g = RustworkxGraphWrapper()
        g.add_node('test_node')

        idx = g.get_index('test_node')
        node = g.get_node(idx)

        self.assertEqual(node, 'test_node')

    def test_rx_graph_access(self):
        """Access underlying rustworkx graph."""
        g = RustworkxGraphWrapper()
        g.add_node('a')

        rx_g = g.rx_graph
        self.assertEqual(rx_g.num_nodes(), 1)


if __name__ == '__main__':
    unittest.main()

"""Unit tests for interface island detection functions in coupled_system.py.

Tests find_interface_islands(), apply_island_penalty(), and
detect_interface_islands() — used by the distributed DDM solver to
handle singular global Schur complement systems.
"""

import unittest

import numpy as np
import scipy.sparse as sp

from solver.coupled_system import (
    find_interface_islands,
    apply_island_penalty,
    detect_interface_islands,
)


def _make_laplacian(n, edges):
    """Build a sparse Laplacian from node count and (i, j, g) edge list.

    Returns CSR matrix of shape (n, n).
    """
    data, rows, cols = [], [], []
    diag = np.zeros(n, dtype=np.float64)
    for i, j, g in edges:
        rows.extend([i, j])
        cols.extend([j, i])
        data.extend([-g, -g])
        diag[i] += g
        diag[j] += g
    for k in range(n):
        rows.append(k)
        cols.append(k)
        data.append(diag[k])
    return sp.csr_matrix((data, (rows, cols)), shape=(n, n))


class TestFindInterfaceIslands(unittest.TestCase):
    """Tests for find_interface_islands()."""

    def test_no_islands_fully_connected_to_pad(self):
        """All interface nodes connected to a pad via extra_edges -> no islands."""
        nodes = ['A', 'B', 'C']
        idx = {n: i for i, n in enumerate(nodes)}
        # A--B--C chain in the Schur complement
        S = _make_laplacian(3, [(0, 1, 1.0), (1, 2, 1.0)])
        # Pad connected to A
        extra = [('A', 'PAD1', 10.0)]
        pads = {'PAD1'}

        islands = find_interface_islands(S, nodes, idx, pads, extra)
        self.assertEqual(islands, set())

    def test_single_disconnected_island(self):
        """Node C is disconnected from A-B which connects to pad."""
        nodes = ['A', 'B', 'C']
        idx = {n: i for i, n in enumerate(nodes)}
        # A--B connected, C isolated (no off-diagonal entries for C)
        S = _make_laplacian(3, [(0, 1, 1.0)])
        extra = [('A', 'PAD1', 10.0)]
        pads = {'PAD1'}

        islands = find_interface_islands(S, nodes, idx, pads, extra)
        self.assertEqual(islands, {'C'})

    def test_multiple_islands(self):
        """Two disconnected clusters, neither connected to pad."""
        nodes = ['A', 'B', 'C', 'D']
        idx = {n: i for i, n in enumerate(nodes)}
        # A--B cluster, C--D cluster, no pads at all
        S = _make_laplacian(4, [(0, 1, 1.0), (2, 3, 1.0)])
        pads = set()

        islands = find_interface_islands(S, nodes, idx, pads, extra_edges=None)
        self.assertEqual(islands, {'A', 'B', 'C', 'D'})

    def test_pad_connected_via_extra_edges_only(self):
        """Node connected to pad only through extra_edges, not S_global."""
        nodes = ['A', 'B']
        idx = {n: i for i, n in enumerate(nodes)}
        # No off-diagonal in S (both isolated in Schur complement)
        S = sp.csr_matrix(np.diag([1.0, 1.0]))
        # PAD connects to A via extra_edges
        extra = [('A', 'PAD1', 5.0)]
        pads = {'PAD1'}

        islands = find_interface_islands(S, nodes, idx, pads, extra)
        # A is connected to PAD1, B is an island
        self.assertEqual(islands, {'B'})

    def test_empty_interface(self):
        """Empty interface nodes -> empty island set."""
        S = sp.csr_matrix((0, 0))
        islands = find_interface_islands(S, [], {}, {'PAD'}, None)
        self.assertEqual(islands, set())

    def test_ground_node_excluded(self):
        """Extra edges touching ground node '0' are skipped."""
        nodes = ['A']
        idx = {'A': 0}
        S = sp.csr_matrix(np.array([[1.0]]))
        # Edge from A to ground should NOT create adjacency
        extra = [('A', '0', 10.0)]
        pads = set()

        islands = find_interface_islands(S, nodes, idx, pads, extra)
        self.assertEqual(islands, {'A'})

    def test_extra_edge_between_interface_nodes(self):
        """Extra edge connecting two interface unknowns (not in S_global)."""
        nodes = ['A', 'B']
        idx = {n: i for i, n in enumerate(nodes)}
        # No off-diagonal in S
        S = sp.csr_matrix(np.diag([1.0, 1.0]))
        # Extra edge connects A to B; pad connects to A
        extra = [('A', 'B', 2.0), ('A', 'PAD1', 5.0)]
        pads = {'PAD1'}

        islands = find_interface_islands(S, nodes, idx, pads, extra)
        # Both A and B should reach PAD1 via extra edges
        self.assertEqual(islands, set())

    def test_negative_conductance_extra_edge_skipped(self):
        """Extra edges with g <= 0 should be ignored."""
        nodes = ['A']
        idx = {'A': 0}
        S = sp.csr_matrix(np.array([[1.0]]))
        extra = [('A', 'PAD1', -1.0), ('A', 'PAD1', 0.0)]
        pads = {'PAD1'}

        islands = find_interface_islands(S, nodes, idx, pads, extra)
        self.assertEqual(islands, {'A'})

    def test_large_connected_graph_no_islands(self):
        """Chain of 100 nodes all connected to one pad -> no islands."""
        n = 100
        nodes = [f'N{i}' for i in range(n)]
        idx = {nd: i for i, nd in enumerate(nodes)}
        edges = [(i, i + 1, 1.0) for i in range(n - 1)]
        S = _make_laplacian(n, edges)
        extra = [('N0', 'PAD', 10.0)]
        pads = {'PAD'}

        islands = find_interface_islands(S, nodes, idx, pads, extra)
        self.assertEqual(islands, set())

    def test_two_components_one_has_pad(self):
        """Two components: one with pad (valid), one without (island)."""
        nodes = ['A', 'B', 'C', 'D']
        idx = {n: i for i, n in enumerate(nodes)}
        # A--B cluster, C--D cluster
        S = _make_laplacian(4, [(0, 1, 1.0), (2, 3, 1.0)])
        extra = [('A', 'PAD1', 5.0)]
        pads = {'PAD1'}

        islands = find_interface_islands(S, nodes, idx, pads, extra)
        self.assertEqual(islands, {'C', 'D'})


class TestApplyIslandPenalty(unittest.TestCase):
    """Tests for apply_island_penalty()."""

    def test_empty_islands_returns_unchanged(self):
        """Empty island set returns inputs unchanged (zero-copy)."""
        S = sp.csr_matrix(np.diag([1.0, 2.0]))
        rhs = np.array([0.5, 1.0])
        idx = {'A': 0, 'B': 1}

        S_out, rhs_out = apply_island_penalty(S, rhs, set(), idx, 1.0)
        self.assertIs(S_out, S)  # Zero-copy: same object
        self.assertIs(rhs_out, rhs)

    def test_single_island_penalty(self):
        """Single island node gets penalty on diagonal and RHS."""
        S = sp.csr_matrix(np.diag([1.0, 2.0]))
        rhs = np.array([0.0, 0.0])
        idx = {'A': 0, 'B': 1}
        vdd = 0.8
        penalty = 1e5

        S_out, rhs_out = apply_island_penalty(
            S, rhs, {'B'}, idx, vdd, penalty
        )

        # Diagonal of B (index 1) should have penalty added
        self.assertAlmostEqual(S_out[1, 1], 2.0 + penalty)
        # A unchanged
        self.assertAlmostEqual(S_out[0, 0], 1.0)
        # Off-diagonal unchanged
        self.assertAlmostEqual(S_out[0, 1], 0.0)

        # RHS
        self.assertAlmostEqual(rhs_out[1], penalty * vdd)
        self.assertAlmostEqual(rhs_out[0], 0.0)

    def test_multiple_island_penalties(self):
        """Multiple island nodes each get penalty."""
        n = 4
        S = sp.csr_matrix(np.eye(n))
        rhs = np.zeros(n)
        idx = {f'N{i}': i for i in range(n)}
        islands = {'N1', 'N3'}
        vdd = 1.0
        penalty = 100.0

        S_out, rhs_out = apply_island_penalty(S, rhs, islands, idx, vdd, penalty)

        for i in range(n):
            expected_diag = 1.0 + (penalty if f'N{i}' in islands else 0.0)
            self.assertAlmostEqual(S_out[i, i], expected_diag)

        for i in range(n):
            expected_rhs = penalty * vdd if f'N{i}' in islands else 0.0
            self.assertAlmostEqual(rhs_out[i], expected_rhs)

    def test_rhs_not_mutated_in_place(self):
        """Original rhs array must not be modified."""
        S = sp.csr_matrix(np.diag([1.0]))
        rhs = np.array([5.0])
        idx = {'A': 0}

        _, rhs_out = apply_island_penalty(S, rhs, {'A'}, idx, 1.0, 100.0)
        self.assertAlmostEqual(rhs[0], 5.0)  # Original unchanged
        self.assertAlmostEqual(rhs_out[0], 5.0 + 100.0)

    def test_default_penalty_is_1e5(self):
        """Default penalty_conductance should be 1e5 (matching GMAX)."""
        S = sp.csr_matrix(np.diag([0.0]))
        rhs = np.array([0.0])
        idx = {'A': 0}

        S_out, _ = apply_island_penalty(S, rhs, {'A'}, idx, 1.0)
        self.assertAlmostEqual(S_out[0, 0], 1e5)


class TestDetectInterfaceIslands(unittest.TestCase):
    """Tests for detect_interface_islands() convenience wrapper."""

    def test_no_islands_fast_path(self):
        """When no islands exist, returns inputs unchanged (zero-copy)."""
        nodes = ['A', 'B']
        idx = {n: i for i, n in enumerate(nodes)}
        S = _make_laplacian(2, [(0, 1, 1.0)])
        rhs = np.array([1.0, 2.0])
        extra = [('A', 'PAD1', 5.0)]
        pads = {'PAD1'}

        S_out, rhs_out, islands = detect_interface_islands(
            S, rhs, nodes, idx, pads, extra, dirichlet_voltage=1.0,
        )
        self.assertIs(S_out, S)
        self.assertIs(rhs_out, rhs)
        self.assertEqual(islands, set())

    def test_island_detected_and_fixed(self):
        """Island node detected and diagonal penalty applied."""
        nodes = ['A', 'B', 'C']
        idx = {n: i for i, n in enumerate(nodes)}
        # A--B connected, C isolated
        S = _make_laplacian(3, [(0, 1, 1.0)])
        rhs = np.zeros(3)
        extra = [('A', 'PAD1', 5.0)]
        pads = {'PAD1'}
        vdd = 0.9

        S_out, rhs_out, islands = detect_interface_islands(
            S, rhs, nodes, idx, pads, extra,
            dirichlet_voltage=vdd, penalty_conductance=1e4,
        )

        self.assertEqual(islands, {'C'})
        # C (index 2) should have penalty
        self.assertAlmostEqual(S_out[2, 2], S[2, 2] + 1e4)
        self.assertAlmostEqual(rhs_out[2], 1e4 * vdd)
        # A, B unchanged
        self.assertAlmostEqual(S_out[0, 0], S[0, 0])
        self.assertAlmostEqual(rhs_out[0], 0.0)

    def test_penalised_system_is_solvable(self):
        """After penalty, the system should be non-singular and solvable."""
        nodes = ['A', 'B', 'C']
        idx = {n: i for i, n in enumerate(nodes)}
        # A--B connected via off-diag. A is grounded by pad Dirichlet
        # elimination (adds 5.0 to A's diagonal + RHS contribution).
        # C is isolated (island).
        S = _make_laplacian(3, [(0, 1, 1.0)])
        # Simulate pad Dirichlet elimination: adds pad conductance to
        # A's diagonal (just like assemble_schur_complement_system does)
        pad_g = 5.0
        vdd = 1.0
        S_base = S.toarray()
        S_base[0, 0] += pad_g
        S = sp.csr_matrix(S_base)
        rhs = np.zeros(3)
        rhs[0] = pad_g * vdd  # Dirichlet RHS contribution

        extra = [('A', 'PAD1', pad_g)]
        pads = {'PAD1'}

        S_out, rhs_out, islands = detect_interface_islands(
            S, rhs, nodes, idx, pads, extra,
            dirichlet_voltage=vdd, penalty_conductance=1e5,
        )

        self.assertEqual(islands, {'C'})

        # Solve: the penalised system should be non-singular
        import scipy.sparse.linalg as spla
        v = spla.spsolve(S_out.tocsc(), rhs_out)
        self.assertTrue(np.all(np.isfinite(v)))
        # Island node C should be pinned close to vdd
        self.assertAlmostEqual(v[idx['C']], vdd, places=3)
        # Non-island nodes should also have finite voltages near vdd
        self.assertAlmostEqual(v[idx['A']], vdd, places=3)
        self.assertAlmostEqual(v[idx['B']], vdd, places=3)

    def test_returns_three_element_tuple(self):
        """Return type is always (csr_matrix, ndarray, set)."""
        nodes = ['A']
        idx = {'A': 0}
        S = sp.csr_matrix(np.array([[1.0]]))
        rhs = np.array([0.0])

        result = detect_interface_islands(S, rhs, nodes, idx, set(),
                                         dirichlet_voltage=0.0)
        self.assertEqual(len(result), 3)
        self.assertIsInstance(result[0], sp.csr_matrix)
        self.assertIsInstance(result[1], np.ndarray)
        self.assertIsInstance(result[2], set)


if __name__ == '__main__':
    unittest.main()

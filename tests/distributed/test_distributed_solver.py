"""Unit tests for distributed DDM solver building blocks.

Tests Schur complements, tile workers, local backend, island detection,
instance model parsing, and other fast unit-level functionality.
"""

import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pytest

from solver.coupled_system import (
    build_block_system_from_edges,
    compute_explicit_schur,
    assemble_schur_complement_system,
)

pytestmark = pytest.mark.unit

# ──────────────────────────────────────────────────────────────────────
# Test data
# ──────────────────────────────────────────────────────────────────────


def _build_toy_graph():
    """Build a minimal 4-node graph for Schur complement unit tests.

    Graph:
        a ──[2mS]── b ──[3mS]── c
                     |
                   [1mS]
                     |
                     0 (ground)

    Port nodes: {a, c}
    Interior nodes: {b}
    Ground: '0'
    """
    edges = [
        ('a', 'b', 2.0),   # 2 mS
        ('b', 'c', 3.0),   # 3 mS
        ('b', '0', 1.0),   # 1 mS to ground
    ]
    port_nodes = {'a', 'c'}
    return edges, port_nodes


def _build_two_tile_graph():
    """Build a 2-tile decomposition for DDM unit tests.

    Tile A: a1 ──[1mS]── A ──[2mS]── B
    Tile B: B ──[3mS]── b1

    Boundary: {A, B}
    A is Dirichlet at 1.0V (voltage source)
    b1 has 0.5 mA current sink
    """
    tile_a_edges = [
        ('a1', 'A', 1.0),
        ('A', 'B', 2.0),
    ]
    tile_b_edges = [
        ('B', 'b1', 3.0),
    ]
    return tile_a_edges, tile_b_edges


# ──────────────────────────────────────────────────────────────────────
# Building Block Tests
# ──────────────────────────────────────────────────────────────────────

class TestComputeExplicitSchur(unittest.TestCase):
    """Test compute_explicit_schur against SchurComplementOperator matvec."""

    def test_explicit_vs_matvec_toy(self):
        """Explicit S must match matrix-free S @ e_i for all basis vectors."""
        edges, port_nodes = _build_toy_graph()
        block, rhs_d = build_block_system_from_edges(
            edges, port_nodes, ground_node='0',
        )
        block.factor_interior()

        S_explicit = compute_explicit_schur(block)
        n = block.n_ports

        self.assertEqual(S_explicit.shape, (n, n))

        # Compare against matvec for each basis vector
        for j in range(n):
            e_j = np.zeros(n)
            e_j[j] = 1.0
            # Manual Schur matvec: S @ e_j = G_pp @ e_j - G_pi @ (G_ii^{-1} @ (G_ip @ e_j))
            Gip_ej = block.G_ip.dot(e_j)
            inv_Gii_Gip_ej = block.lu_ii(Gip_ej)
            S_ej_manual = block.G_pp.dot(e_j) - block.G_pi.dot(inv_Gii_Gip_ej)

            np.testing.assert_allclose(
                S_explicit[:, j], S_ej_manual, atol=1e-12,
                err_msg=f"Schur column {j} mismatch",
            )

    def test_explicit_schur_symmetric(self):
        """Schur complement of symmetric system must be symmetric."""
        edges, port_nodes = _build_toy_graph()
        block, _ = build_block_system_from_edges(
            edges, port_nodes, ground_node='0',
        )
        block.factor_interior()
        S = compute_explicit_schur(block)
        np.testing.assert_allclose(S, S.T, atol=1e-12)

    def test_chunked_path_matches_single_solve(self):
        """Force chunked path via tiny memory budget; result must match single solve."""
        # Build 20x20 grid: 400 nodes, 76 ports on perimeter, 324 interior
        # Need n_ports > 32 so the BLAS floor (32) allows chunking
        N = 20
        edges = []
        for i in range(N):
            for j in range(N):
                node = f'n{i}_{j}'
                if j < N - 1:
                    edges.append((node, f'n{i}_{j+1}', 1.0))
                if i < N - 1:
                    edges.append((node, f'n{i+1}_{j}', 1.0))
        edges.append((f'n{N//2}_{N//2}', '0', 0.1))  # Ground at center

        # Perimeter as ports: 4*N - 4 = 76 nodes
        port_nodes = set()
        for i in range(N):
            port_nodes.add(f'n0_{i}')
            port_nodes.add(f'n{N-1}_{i}')
            port_nodes.add(f'n{i}_0')
            port_nodes.add(f'n{i}_{N-1}')

        block, _ = build_block_system_from_edges(edges, port_nodes, ground_node='0')
        block.factor_interior()

        n_ports = block.n_ports
        n_interior = block.n_interior
        self.assertGreater(n_ports, 32, "Need >32 ports for chunked path test")

        # Single solve (large memory budget)
        S_single = compute_explicit_schur(block, max_memory_gb=100.0)

        # Force chunked path: tiny memory budget -> chunk_size = 32 (BLAS floor)
        # With 76 ports and chunk_size=32, we get 3 chunks
        S_chunked = compute_explicit_schur(block, max_memory_gb=1e-9)

        # Verify chunked path was actually taken
        # chunk_size = max(min(memory_chunk, ..., 256), 32) = 32 < 76 = n_ports
        INT_MAX = 2**31 - 1
        bytes_per_col = n_interior * 8 * 0.5
        memory_chunk = max(1, int(1e-9 * 1e9 / bytes_per_col))
        index_chunk = max(1, INT_MAX // max(n_interior, 1))
        chunk_size = min(memory_chunk, index_chunk, n_ports, 256)
        chunk_size = max(chunk_size, min(32, n_ports))
        self.assertLess(chunk_size, n_ports,
                        f"Expected chunked path: chunk_size={chunk_size}, n_ports={n_ports}")

        np.testing.assert_allclose(S_chunked, S_single, atol=1e-12,
                                   err_msg="Chunked path result differs from single solve")
        np.testing.assert_allclose(S_chunked, S_chunked.T, atol=1e-12,
                                   err_msg="Chunked Schur complement not symmetric")




class TestBuildBlockSystemFromEdges(unittest.TestCase):
    """Test build_block_system_from_edges against extract_block_matrices."""

    def test_toy_graph_dimensions(self):
        """Block system has correct interior/port counts."""
        edges, port_nodes = _build_toy_graph()
        block, rhs_d = build_block_system_from_edges(
            edges, port_nodes, ground_node='0',
        )
        self.assertEqual(block.n_ports, 2)
        self.assertEqual(block.n_interior, 1)
        self.assertEqual(len(block.port_nodes), 2)
        self.assertIn('a', block.port_nodes)
        self.assertIn('c', block.port_nodes)
        # Ground should not be in interior or port
        self.assertNotIn('0', block.interior_to_idx)
        self.assertNotIn('0', block.port_to_idx)
        # rhs_dirichlet should be zero (no Dirichlet nodes)
        # Shape is (n_ports + n_interior,) = (2 + 1,) = (3,)
        np.testing.assert_array_equal(rhs_d, np.zeros(block.n_ports + block.n_interior))

    def test_dirichlet_elimination(self):
        """Dirichlet nodes eliminated: not in port/interior, RHS contribution nonzero."""
        edges = [
            ('a', 'b', 2.0),
            ('b', 'c', 3.0),
            ('c', '0', 1.0),
        ]
        port_nodes = {'b'}
        dirichlet_nodes = {'a'}
        block, rhs_d = build_block_system_from_edges(
            edges, port_nodes, dirichlet_nodes=dirichlet_nodes,
            dirichlet_voltage=1.0, ground_node='0',
        )
        # a is Dirichlet, b is port, c is interior, 0 is ground
        self.assertEqual(block.n_ports, 1)
        self.assertEqual(block.n_interior, 1)
        self.assertNotIn('a', block.port_to_idx)
        self.assertNotIn('a', block.interior_to_idx)
        # RHS should have nonzero contribution from Dirichlet
        self.assertGreater(np.abs(rhs_d).sum(), 0)

    def test_conductance_conservation(self):
        """Row sums of full conductance matrix should be zero (for non-ground rows)."""
        edges, port_nodes = _build_toy_graph()
        block, _ = build_block_system_from_edges(
            edges, port_nodes, ground_node='0',
        )
        # Check that G_pp + G_pi has correct row sums (should include ground contribution)
        # For port node 'a': connected to b(2mS) → diagonal=2, off-diag to b=-2
        # For port node 'c': connected to b(3mS) → diagonal=3, off-diag to b=-3
        # Port-port block diagonal entries
        G_pp = block.G_pp.toarray()
        G_pi = block.G_pi.toarray()
        G_ip = block.G_ip.toarray()
        G_ii = block.G_ii.toarray()

        # Full system should have zero row sums (for nodes not connected to ground)
        # Port 'a' has no ground connection: G_pp[a,:] sum + G_pi[a,:] sum = 0
        a_idx = block.port_to_idx['a']
        a_row_sum = G_pp[a_idx, :].sum() + G_pi[a_idx, :].sum()
        self.assertAlmostEqual(a_row_sum, 0.0, places=12)


class TestAssembleSchurComplementSystem(unittest.TestCase):
    """Test global Schur complement assembly from per-tile contributions."""

    def test_two_tile_assembly(self):
        """Two tile Schurs assembled correctly with Dirichlet elimination."""
        tile_a_edges, tile_b_edges = _build_two_tile_graph()

        # Build per-tile Schur complements
        block_a, _ = build_block_system_from_edges(
            tile_a_edges, {'A', 'B'}, ground_node='0',
        )
        block_a.factor_interior()
        S_a = compute_explicit_schur(block_a)

        block_b, _ = build_block_system_from_edges(
            tile_b_edges, {'B'}, ground_node='0',
        )
        block_b.factor_interior()
        S_b = compute_explicit_schur(block_b)

        # Assemble global interface
        tile_schurs = {
            'A': S_a,
            'B': S_b,
        }
        tile_ports = {
            'A': list(block_a.port_nodes),
            'B': list(block_b.port_nodes),
        }
        S_global, rhs_d, nodes, node_to_idx = assemble_schur_complement_system(
            tile_schurs, tile_ports,
            dirichlet_nodes={'A'}, dirichlet_voltage=1.0,
        )

        # Only 'B' should remain as unknown (A is Dirichlet)
        self.assertEqual(len(nodes), 1)
        self.assertIn('B', nodes)
        self.assertEqual(S_global.shape, (1, 1))
        # RHS should be nonzero (Dirichlet contribution from A)
        self.assertGreater(np.abs(rhs_d).sum(), 0)

    def test_symmetry(self):
        """Assembled global S should be symmetric."""
        tile_a_edges, tile_b_edges = _build_two_tile_graph()

        block_a, _ = build_block_system_from_edges(
            tile_a_edges, {'A', 'B'}, ground_node='0',
        )
        block_a.factor_interior()
        S_a = compute_explicit_schur(block_a)

        block_b, _ = build_block_system_from_edges(
            tile_b_edges, {'B'}, ground_node='0',
        )
        block_b.factor_interior()
        S_b = compute_explicit_schur(block_b)

        S_global, _, _, _ = assemble_schur_complement_system(
            {'A': S_a, 'B': S_b},
            {'A': list(block_a.port_nodes), 'B': list(block_b.port_nodes)},
        )
        S_dense = S_global.toarray()
        np.testing.assert_allclose(S_dense, S_dense.T, atol=1e-12)

    def test_extra_edges(self):
        """Extra edges (e.g., package) add conductance to global S."""
        edges = [('a', 'b', 1.0)]
        block, _ = build_block_system_from_edges(edges, {'a', 'b'}, ground_node='0')
        block.factor_interior()
        S = compute_explicit_schur(block)

        # Without extra edges
        S1, _, _, _ = assemble_schur_complement_system(
            {'t': S}, {'t': list(block.port_nodes)},
        )
        # With extra edge adding 5mS between a and b
        S2, _, _, _ = assemble_schur_complement_system(
            {'t': S}, {'t': list(block.port_nodes)},
            extra_edges=[('a', 'b', 5.0)],
        )
        # S2 should have larger diagonal entries
        self.assertGreater(S2.toarray().diagonal().sum(), S1.toarray().diagonal().sum())


# ──────────────────────────────────────────────────────────────────────
# Tile Worker Tests
# ──────────────────────────────────────────────────────────────────────

class TestTileWorker(unittest.TestCase):
    """Test TileWorker setup and solve operations."""

    def test_setup_returns_metadata(self):
        """Worker setup returns tile_id, boundary_nodes, counts."""
        from distributed.tile_worker import TileWorker

        # Create a temp tile .ckt file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.ckt', delete=False) as f:
            f.write("* Test tile\n")
            f.write("R1 a *b 1000\n")  # 1kOhm, *b is boundary
            f.write("R2 *b c 2000\n")  # 2kOhm
            f.write("R3 c 0 500\n")    # 500 Ohm to ground
            temp_ckt = f.name

        try:
            worker = TileWorker()
            result = worker.setup(
                {
                    'tile_id': [0, 0],
                    'ckt_path': temp_ckt,
                    'nd_path': None,
                    'instance_path': None,
                    'net_filter': None,
                },
                interface_nodes={'b'},  # b is the only interface node
            )

            self.assertEqual(result['tile_id'], (0, 0))
            self.assertIn('b', result['boundary_nodes'])
            self.assertGreater(result['n_interior'], 0)
            self.assertEqual(result['n_boundary'], 1)
        finally:
            os.unlink(temp_ckt)

    def test_factor_and_schur(self):
        """Worker computes Schur complement after factoring interior."""
        from distributed.tile_worker import TileWorker

        # Build tile with boundary node 'b' and two interior branches
        # b connects to a (1kOhm) and c (2kOhm), c connects to ground (500 Ohm)
        # S(b,b) = g_ba + g_bc - g_bc^2 / (g_bc + g_c0) > 0
        with tempfile.NamedTemporaryFile(mode='w', suffix='.ckt', delete=False) as f:
            f.write("R1 a *b 1000\n")   # a-b: 1kOhm → 1mS → g=1.0 mS
            f.write("R2 *b c 2000\n")   # b-c: 2kOhm → 0.5mS → g=0.5 mS
            f.write("R3 c 0 500\n")     # c-0: 500 Ohm → 2mS → g=2.0 mS
            temp_ckt = f.name

        try:
            worker = TileWorker()
            worker.setup(
                {'tile_id': [0, 0], 'ckt_path': temp_ckt, 'nd_path': None,
                 'instance_path': None, 'net_filter': None},
                interface_nodes={'b'},
            )
            S, boundary_list = worker.factor_and_compute_schur()

            self.assertEqual(S.shape[0], S.shape[1])
            self.assertEqual(len(boundary_list), S.shape[0])
            self.assertGreater(S.shape[0], 0)
            # Schur complement should be positive definite
            # S(b,b) > 0 because there's a path from b to ground through c
            # and a path from b to interior node a
            if S.shape[0] == 1:
                self.assertGreater(S[0, 0], 0)
            else:
                # Multi-port: check positive diagonal
                for i in range(S.shape[0]):
                    self.assertGreater(S[i, i], 0)
        finally:
            os.unlink(temp_ckt)

    def test_get_reduced_rhs(self):
        """Worker computes reduced RHS for current injection."""
        from distributed.tile_worker import TileWorker

        with tempfile.NamedTemporaryFile(mode='w', suffix='.ckt', delete=False) as f:
            f.write("R1 a *b 1000\n")
            f.write("R2 *b c 2000\n")
            f.write("R3 c 0 500\n")
            temp_ckt = f.name

        try:
            worker = TileWorker()
            worker.setup(
                {'tile_id': [0, 0], 'ckt_path': temp_ckt, 'nd_path': None,
                 'instance_path': None, 'net_filter': None},
                interface_nodes={'b'},
            )
            worker.factor_and_compute_schur()

            # With no current, RHS should be zero
            rhs_zero = worker.get_reduced_rhs()
            np.testing.assert_allclose(rhs_zero, 0.0, atol=1e-15)

            # With current at interior node c
            rhs_with_current = worker.get_reduced_rhs({'c': 1.0})  # 1 mA at c
            self.assertGreater(np.abs(rhs_with_current).sum(), 0)
        finally:
            os.unlink(temp_ckt)


# ──────────────────────────────────────────────────────────────────────
# Boundary Node Filtering Tests
# ──────────────────────────────────────────────────────────────────────

class TestBoundaryNodeFiltering(unittest.TestCase):
    """Test single-tile-only boundary node filtering in parser."""

    def test_single_tile_boundary_nodes_filtered(self):
        """collect_shared_boundary_nodes returns only nodes in 2+ tiles."""
        from distributed.parser import DistributedNetlistParser

        # Create two tile .ckt files:
        #   Tile A: has boundary nodes *shared1, *shared2, *only_a
        #   Tile B: has boundary nodes *shared1, *shared2, *only_b
        # Expected shared: {shared1, shared2}
        with tempfile.TemporaryDirectory() as tmpdir:
            tile_a = os.path.join(tmpdir, 'tile_0_0.ckt')
            with open(tile_a, 'w') as f:
                f.write("R1 a1 *shared1 1000\n")
                f.write("R2 *shared1 *shared2 2000\n")
                f.write("R3 *shared2 *only_a 500\n")
                f.write("R4 *only_a 0 1000\n")

            tile_b = os.path.join(tmpdir, 'tile_0_1.ckt')
            with open(tile_b, 'w') as f:
                f.write("R1 b1 *shared1 1000\n")
                f.write("R2 *shared1 *shared2 2000\n")
                f.write("R3 *shared2 *only_b 500\n")
                f.write("R4 *only_b 0 1000\n")

            from distributed.parser import TileConfig
            tile_configs = [
                TileConfig(tile_id=(0, 0), ckt_path=tile_a, nd_path=None,
                           instance_path=None, net_filter=None),
                TileConfig(tile_id=(0, 1), ckt_path=tile_b, nd_path=None,
                           instance_path=None, net_filter=None),
            ]

            parser = DistributedNetlistParser(tmpdir)

            # Old API returns all boundary nodes (deprecated)
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                all_bnd = parser.collect_boundary_nodes(tile_configs)
            self.assertEqual(all_bnd, {'shared1', 'shared2', 'only_a', 'only_b'})

            # New API returns only shared nodes
            shared_bnd = parser.collect_shared_boundary_nodes(tile_configs)
            self.assertEqual(shared_bnd, {'shared1', 'shared2'})

    def test_all_shared_returns_all(self):
        """When all boundary nodes appear in 2+ tiles, nothing is filtered."""
        from distributed.parser import DistributedNetlistParser, TileConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            tile_a = os.path.join(tmpdir, 'tile_0_0.ckt')
            with open(tile_a, 'w') as f:
                f.write("R1 a1 *shared1 1000\n")
                f.write("R2 *shared1 0 500\n")

            tile_b = os.path.join(tmpdir, 'tile_0_1.ckt')
            with open(tile_b, 'w') as f:
                f.write("R1 b1 *shared1 1000\n")
                f.write("R2 *shared1 0 500\n")

            tile_configs = [
                TileConfig(tile_id=(0, 0), ckt_path=tile_a, nd_path=None,
                           instance_path=None, net_filter=None),
                TileConfig(tile_id=(0, 1), ckt_path=tile_b, nd_path=None,
                           instance_path=None, net_filter=None),
            ]

            parser = DistributedNetlistParser(tmpdir)
            shared = parser.collect_shared_boundary_nodes(tile_configs)
            self.assertEqual(shared, {'shared1'})

    def test_no_boundary_nodes_returns_empty(self):
        """Tiles with no boundary nodes return empty set."""
        from distributed.parser import DistributedNetlistParser, TileConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            tile_a = os.path.join(tmpdir, 'tile_0_0.ckt')
            with open(tile_a, 'w') as f:
                f.write("R1 a1 a2 1000\n")
                f.write("R2 a2 0 500\n")

            tile_configs = [
                TileConfig(tile_id=(0, 0), ckt_path=tile_a, nd_path=None,
                           instance_path=None, net_filter=None),
            ]

            parser = DistributedNetlistParser(tmpdir)
            shared = parser.collect_shared_boundary_nodes(tile_configs)
            self.assertEqual(shared, set())


# ──────────────────────────────────────────────────────────────────────
# Backend Tests
# ──────────────────────────────────────────────────────────────────────

class TestLocalBackend(unittest.TestCase):
    """Test LocalBackend for correctness."""

    def test_create_actors_and_call(self):
        """LocalBackend creates objects and calls methods."""
        from distributed.backend import LocalBackend

        class DummyActor:
            def __init__(self):
                self.x = 0
            def inc(self, n):
                self.x += n
                return self.x

        be = LocalBackend()
        be.initialize()

        actors = be.create_actors(DummyActor, [None, None])
        self.assertEqual(len(actors), 2)

        result = be.call(actors[0], 'inc', 5)
        self.assertEqual(result, 5)

        results = be.call_all(actors, 'inc', [(3,), (7,)])
        self.assertEqual(results, [8, 7])

        be.shutdown()

    def test_call_all_no_args(self):
        """call_all without args works correctly."""
        from distributed.backend import LocalBackend

        class Adder:
            def __init__(self):
                self.total = 0
            def get_total(self):
                return self.total

        be = LocalBackend()
        be.initialize()
        actors = be.create_actors(Adder, [None])
        results = be.call_all(actors, 'get_total')
        self.assertEqual(results, [0])
        be.shutdown()

    def test_map_func(self):
        """map_func applies function sequentially to each args tuple."""
        from distributed.backend import LocalBackend

        def add(a, b):
            return a + b

        be = LocalBackend()
        be.initialize()
        results = be.map_func(add, [(1, 2), (3, 4), (10, 20)])
        self.assertEqual(results, [3, 7, 30])
        be.shutdown()

    def test_map_func_empty(self):
        """map_func with empty args_list returns empty list."""
        from distributed.backend import LocalBackend

        def noop():
            return 42

        be = LocalBackend()
        be.initialize()
        results = be.map_func(noop, [])
        self.assertEqual(results, [])
        be.shutdown()


# ──────────────────────────────────────────────────────────────────────
# Island Detection Tests
# ──────────────────────────────────────────────────────────────────────

class TestIslandDetection(unittest.TestCase):
    """Test floating island detection in TileWorker."""

    def test_removes_small_fragments(self):
        """Small fragments with few boundary nodes are removed."""
        from distributed.tile_worker import TileWorker

        # Main component: a-b-c-d chain (4 nodes, 'b' is interface)
        # Fragment: f1-f2 chain (2 nodes, 'f1' is interface but only 1 → removed)
        # Note: ground node '0' edges only add diagonal, don't create adjacency
        lines = [
            "R1 a b 1000\n",
            "R2 b c 1000\n",
            "R3 c d 1000\n",
            "R4 d 0 1000\n",
            "R5 f1 f2 1000\n",  # Disconnected fragment
        ]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.ckt', delete=False) as f:
            f.writelines(lines)
            temp_ckt = f.name

        try:
            worker = TileWorker()
            result = worker.setup(
                {'tile_id': [0, 0], 'ckt_path': temp_ckt, 'nd_path': None,
                 'instance_path': None, 'net_filter': None},
                interface_nodes={'b', 'f1'},
            )
            # Fragment has 1 interface node (< MIN_INTERFACE_NODES_KEEP=5) → removed
            self.assertGreaterEqual(result['islands_removed'], 1)
            # f2 should be removed (part of small fragment)
            self.assertNotIn('f2', worker._tile_data.all_nodes)
            # Main component's node 'a' should be kept
            self.assertIn('a', worker._tile_data.all_nodes)
        finally:
            os.unlink(temp_ckt)

    def test_keeps_well_connected_components(self):
        """Components with >= MIN_INTERFACE_NODES_KEEP interface nodes are kept."""
        from distributed.tile_worker import TileWorker

        threshold = TileWorker.MIN_INTERFACE_NODES_KEEP

        # Main component: long chain (20 nodes, much larger than strip)
        lines = []
        main_nodes = [f'm{i}' for i in range(20)]
        for i in range(len(main_nodes) - 1):
            lines.append(f"R{i} {main_nodes[i]} {main_nodes[i+1]} 1000\n")
        lines.append(f"R_gnd {main_nodes[-1]} 0 1000\n")

        # Second component: strip with many interface nodes
        strip_nodes = [f's{i}' for i in range(threshold + 2)]
        for i in range(len(strip_nodes) - 1):
            lines.append(f"R{100+i} {strip_nodes[i]} {strip_nodes[i+1]} 1000\n")

        with tempfile.NamedTemporaryFile(mode='w', suffix='.ckt', delete=False) as f:
            f.writelines(lines)
            temp_ckt = f.name

        try:
            worker = TileWorker()
            # Make all strip nodes + m0 interface nodes
            interface = {'m0'} | set(strip_nodes)
            result = worker.setup(
                {'tile_id': [0, 0], 'ckt_path': temp_ckt, 'nd_path': None,
                 'instance_path': None, 'net_filter': None},
                interface_nodes=interface,
            )
            # Strip has >= threshold interface nodes → should be kept
            self.assertEqual(result['islands_removed'], 0)
            # All strip nodes should be present
            for sn in strip_nodes:
                self.assertIn(sn, worker._tile_data.all_nodes)
        finally:
            os.unlink(temp_ckt)


# ──────────────────────────────────────────────────────────────────────
# Instance Model Parsing Tests
# ──────────────────────────────────────────────────────────────────────

class TestInstanceModelParsing(unittest.TestCase):
    """Test _parse_instance_models for static_value and pulse handling."""

    def test_dc_only(self):
        """Parse DC-only current source."""
        from distributed.tile_worker import _parse_instance_models

        with tempfile.NamedTemporaryFile(mode='w', suffix='.sp', delete=False) as f:
            f.write("I_test n1 0 1e-3\n")  # 1mA in Amps
            temp = f.name

        try:
            currents = _parse_instance_models(temp, None)
            self.assertIn('n1', currents)
            self.assertAlmostEqual(currents['n1'], 1.0, places=6)  # 1mA
        finally:
            os.unlink(temp)

    def test_static_value(self):
        """Parse DC + static_value (summed per flat parser behavior)."""
        from distributed.tile_worker import _parse_instance_models

        with tempfile.NamedTemporaryFile(mode='w', suffix='.sp', delete=False) as f:
            # dc=1e-3A, static_value=2e-3A → total=3mA
            f.write("I_test n1 0 1e-3 static_value=2e-3\n")
            temp = f.name

        try:
            currents = _parse_instance_models(temp, None)
            self.assertAlmostEqual(currents['n1'], 3.0, places=6)  # 3mA
        finally:
            os.unlink(temp)

    def test_pulse_dc_average_without_static(self):
        """Pulse DC average included when no static_value."""
        from distributed.tile_worker import _parse_instance_models

        with tempfile.NamedTemporaryFile(mode='w', suffix='.sp', delete=False) as f:
            # dc=0, pulse(0, 1e-3, 0, 0, 0, 5e-9, 10e-9)
            # DC average = 0 + (1e-3 - 0) * (0/2 + 5e-9 + 0/2) / 10e-9 = 0.5e-3 A
            f.write("I_test n1 0 0 pulse(0,1e-3,0,0,0,5e-9,10e-9)\n")
            temp = f.name

        try:
            currents = _parse_instance_models(temp, None)
            self.assertIn('n1', currents)
            # 0.5e-3 A * 1e3 = 0.5 mA
            self.assertAlmostEqual(currents['n1'], 0.5, places=4)
        finally:
            os.unlink(temp)

    def test_pulse_ignored_with_static(self):
        """Pulse DC average NOT included when static_value is set."""
        from distributed.tile_worker import _parse_instance_models

        with tempfile.NamedTemporaryFile(mode='w', suffix='.sp', delete=False) as f:
            # dc=1e-3, static_value=2e-3, pulse present but ignored
            f.write("I_test n1 0 1e-3 static_value=2e-3 pulse(0,1e-3,0,0,0,5e-9,10e-9)\n")
            temp = f.name

        try:
            currents = _parse_instance_models(temp, None)
            # Should be dc + static = 3mA, NOT dc + static + pulse_avg
            self.assertAlmostEqual(currents['n1'], 3.0, places=4)
        finally:
            os.unlink(temp)

    def test_ground_node_excluded(self):
        """Current sources with node_pos='0' are excluded."""
        from distributed.tile_worker import _parse_instance_models

        with tempfile.NamedTemporaryFile(mode='w', suffix='.sp', delete=False) as f:
            f.write("I_test 0 n1 1e-3\n")  # node_pos is ground
            temp = f.name

        try:
            currents = _parse_instance_models(temp, None)
            self.assertEqual(len(currents), 0)
        finally:
            os.unlink(temp)

    def test_pwl_dc_average_included(self):
        """PWL DC average included when no static_value."""
        from distributed.tile_worker import _parse_instance_models

        with tempfile.NamedTemporaryFile(mode='w', suffix='.sp', delete=False) as f:
            # dc=0, pwl triangle: (0,0) -> (5ns,2mA) -> (10ns,0) with period=10ns
            # Area = 0.5 * base * height = 0.5 * 10e-9 * 2e-3 = 1e-11
            # DC avg = 1e-11 / 10e-9 = 1e-3 A = 1.0 mA
            f.write("I_test n1 0 0 pwl(0 0 5e-9 2e-3 10e-9 0) pwl_period=10e-9\n")
            temp = f.name

        try:
            currents = _parse_instance_models(temp, None)
            self.assertIn('n1', currents)
            self.assertAlmostEqual(currents['n1'], 1.0, places=4)
        finally:
            os.unlink(temp)

    def test_pwl_dc_ignored_with_static(self):
        """PWL DC average NOT included when static_value is set."""
        from distributed.tile_worker import _parse_instance_models

        with tempfile.NamedTemporaryFile(mode='w', suffix='.sp', delete=False) as f:
            # dc=1e-3, static_value=2e-3 → 3mA; PWL present but ignored
            f.write("I_test n1 0 1e-3 static_value=2e-3 "
                    "pwl(0 0 5e-9 2e-3 10e-9 0) pwl_period=10e-9\n")
            temp = f.name

        try:
            currents = _parse_instance_models(temp, None)
            self.assertAlmostEqual(currents['n1'], 3.0, places=4)
        finally:
            os.unlink(temp)

    def test_net_filter_excludes_wrong_net(self):
        """Sources on a different net are excluded when net_filter is set."""
        from distributed.tile_worker import _parse_instance_models

        # Instance models file with sources on both VDD and VSS nodes
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sp', delete=False) as f:
            f.write("I_vdd vdd_node 0 1e-3\n")
            f.write("I_vss vss_node 0 2e-3\n")
            temp_sp = f.name

        # .nd file mapping nodes to nets
        with tempfile.NamedTemporaryFile(mode='w', suffix='.nd', delete=False) as f:
            # Format: node_name x y layer tile_id net_name
            f.write("vdd_node 100 200 M1 0_0 VDD\n")
            f.write("vss_node 300 400 M1 0_0 VSS\n")
            temp_nd = f.name

        try:
            # Filter for VDD only
            currents = _parse_instance_models(temp_sp, 'vdd', temp_nd)
            self.assertIn('vdd_node', currents)
            self.assertAlmostEqual(currents['vdd_node'], 1.0, places=6)
            self.assertNotIn('vss_node', currents)  # VSS source excluded

            # Filter for VSS only
            currents = _parse_instance_models(temp_sp, 'vss', temp_nd)
            self.assertNotIn('vdd_node', currents)
            self.assertIn('vss_node', currents)
            self.assertAlmostEqual(currents['vss_node'], 2.0, places=6)

            # No filter → both included
            currents = _parse_instance_models(temp_sp, None)
            self.assertIn('vdd_node', currents)
            self.assertIn('vss_node', currents)
        finally:
            os.unlink(temp_sp)
            os.unlink(temp_nd)

    def test_net_filter_without_nd_file_includes_all(self):
        """When net_filter is set but nd_path is None, no sources are matched."""
        from distributed.tile_worker import _parse_instance_models

        with tempfile.NamedTemporaryFile(mode='w', suffix='.sp', delete=False) as f:
            f.write("I_test n1 0 1e-3\n")
            temp = f.name

        try:
            # net_filter set but no .nd file → node_net_map is empty → nothing matches
            currents = _parse_instance_models(temp, 'vdd')
            self.assertEqual(len(currents), 0)
        finally:
            os.unlink(temp)


# ──────────────────────────────────────────────────────────────────────
# Interface Island Detection Tests
# ──────────────────────────────────────────────────────────────────────

class TestInterfaceIslandDetection(unittest.TestCase):
    """Integration tests for interface island detection in the distributed pipeline.

    Tests find_interface_islands(), apply_island_penalty(), and
    detect_interface_islands() in the context of assembled Schur complement
    systems (global interface level), complementing the unit-level tests
    in tests/solver/test_interface_islands.py.
    """

    # ── helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _make_healthy_two_tile_system(vdd: float = 1.0):
        """Build a small 2-tile system where all interface nodes reach a pad.

        Tile A:  a1 ──[1]── I1 ──[2]── I2
        Tile B:  I1 ──[3]── I2 ──[4]── b1

        Interface (boundary): {I1, I2}
        Package edge: I1 ── PAD1 (10 mS)  — connects interface to Dirichlet pad
        Pad: PAD1 at vdd

        Returns (S_global, rhs, nodes, idx, extra_edges, pad_nodes, vdd).
        """
        # Tile A: interior={a1}, ports={I1, I2}
        tile_a_edges = [('a1', 'I1', 1.0), ('I1', 'I2', 2.0)]
        # Tile B: interior={b1}, ports={I1, I2}
        tile_b_edges = [('I1', 'I2', 3.0), ('I2', 'b1', 4.0)]

        block_a, _ = build_block_system_from_edges(
            tile_a_edges, {'I1', 'I2'}, ground_node='0',
        )
        block_a.factor_interior()
        S_a = compute_explicit_schur(block_a)

        block_b, _ = build_block_system_from_edges(
            tile_b_edges, {'I1', 'I2'}, ground_node='0',
        )
        block_b.factor_interior()
        S_b = compute_explicit_schur(block_b)

        pad_nodes = {'PAD1'}
        extra_edges = [('I1', 'PAD1', 10.0)]

        S_global, rhs, nodes, idx = assemble_schur_complement_system(
            tile_schur_complements={'A': S_a, 'B': S_b},
            tile_port_node_lists={
                'A': list(block_a.port_nodes),
                'B': list(block_b.port_nodes),
            },
            extra_edges=extra_edges,
            dirichlet_nodes=pad_nodes,
            dirichlet_voltage=vdd,
        )
        return S_global, rhs, nodes, idx, extra_edges, pad_nodes, vdd

    @staticmethod
    def _make_island_two_tile_system(vdd: float = 1.0):
        """Build a 2-tile system where group B is disconnected from any pad.

        Tile A:  a1 ──[1]── I1 ──[2]── I2
        Tile B:  I3 ──[3]── I4 ──[4]── b1

        Interface (boundary): {I1, I2, I3, I4}
        Package edge: I1 ── PAD1 (10 mS) — only I1-I2 group reaches pad
        Group A: {I1, I2} — connected to PAD1
        Group B: {I3, I4} — disconnected island

        Returns (S_global, rhs, nodes, idx, extra_edges, pad_nodes, vdd).
        """
        tile_a_edges = [('a1', 'I1', 1.0), ('I1', 'I2', 2.0)]
        tile_b_edges = [('I3', 'I4', 3.0), ('I4', 'b1', 4.0)]

        block_a, _ = build_block_system_from_edges(
            tile_a_edges, {'I1', 'I2'}, ground_node='0',
        )
        block_a.factor_interior()
        S_a = compute_explicit_schur(block_a)

        block_b, _ = build_block_system_from_edges(
            tile_b_edges, {'I3', 'I4'}, ground_node='0',
        )
        block_b.factor_interior()
        S_b = compute_explicit_schur(block_b)

        pad_nodes = {'PAD1'}
        extra_edges = [('I1', 'PAD1', 10.0)]

        S_global, rhs, nodes, idx = assemble_schur_complement_system(
            tile_schur_complements={'A': S_a, 'B': S_b},
            tile_port_node_lists={
                'A': list(block_a.port_nodes),
                'B': list(block_b.port_nodes),
            },
            extra_edges=extra_edges,
            dirichlet_nodes=pad_nodes,
            dirichlet_voltage=vdd,
        )
        return S_global, rhs, nodes, idx, extra_edges, pad_nodes, vdd

    # ── tests ────────────────────────────────────────────────────────

    def test_no_islands_noop(self):
        """Healthy system: detect_interface_islands returns inputs unchanged."""
        from solver.coupled_system import detect_interface_islands

        S_global, rhs, nodes, idx, extra, pads, vdd = (
            self._make_healthy_two_tile_system()
        )

        S_fixed, rhs_fixed, islands = detect_interface_islands(
            S_global, rhs, nodes, idx,
            pad_nodes=pads, extra_edges=extra,
            dirichlet_voltage=vdd,
        )

        # Zero-copy fast path: objects returned unchanged
        self.assertIs(S_fixed, S_global)
        self.assertIs(rhs_fixed, rhs)
        self.assertEqual(islands, set())

    def test_find_interface_islands_detects_disconnected(self):
        """Group B (no pad path) detected as islands; group A not."""
        from solver.coupled_system import find_interface_islands

        S_global, rhs, nodes, idx, extra, pads, vdd = (
            self._make_island_two_tile_system()
        )

        islands = find_interface_islands(
            S_global, nodes, idx, pads, extra,
        )

        # Group B nodes are islands
        self.assertIn('I3', islands)
        self.assertIn('I4', islands)
        # Group A nodes are NOT islands
        self.assertNotIn('I1', islands)
        self.assertNotIn('I2', islands)

    def test_apply_island_penalty_modifies_diagonal_and_rhs(self):
        """Island penalty adds GMAX to diagonal and GMAX*vdd to RHS."""
        import scipy.sparse as sp
        from solver.coupled_system import apply_island_penalty

        GMAX = 1e5
        vdd = 0.85

        S_global, rhs, nodes, idx, extra, pads, _ = (
            self._make_island_two_tile_system(vdd=vdd)
        )

        # Identify islands first
        from solver.coupled_system import find_interface_islands
        islands = find_interface_islands(S_global, nodes, idx, pads, extra)
        self.assertGreater(len(islands), 0, "Need islands for this test")

        S_orig_diag = S_global.toarray().diagonal().copy()
        rhs_orig = rhs.copy()

        S_fixed, rhs_fixed = apply_island_penalty(
            S_global, rhs, islands, idx, vdd,
        )

        S_fixed_diag = S_fixed.toarray().diagonal()

        for node in nodes:
            i = idx[node]
            if node in islands:
                # Diagonal increased by GMAX
                np.testing.assert_allclose(
                    S_fixed_diag[i], S_orig_diag[i] + GMAX, rtol=1e-12,
                    err_msg=f"Island node {node}: diagonal not increased by GMAX",
                )
                # RHS increased by GMAX * vdd
                np.testing.assert_allclose(
                    rhs_fixed[i], rhs_orig[i] + GMAX * vdd, rtol=1e-12,
                    err_msg=f"Island node {node}: RHS not increased by GMAX*vdd",
                )
            else:
                # Non-island entries unchanged (exact equality)
                self.assertEqual(
                    S_fixed_diag[i], S_orig_diag[i],
                    f"Non-island node {node}: diagonal should be unchanged",
                )
                self.assertEqual(
                    rhs_fixed[i], rhs_orig[i],
                    f"Non-island node {node}: RHS should be unchanged",
                )

    def test_island_voltages_near_vdd(self):
        """After penalty + solve, island nodes within 0.1% of VDD,
        non-island voltages unperturbed by penalty (checked under load)."""
        import scipy.sparse.linalg as spla
        from solver.coupled_system import detect_interface_islands

        vdd = 0.85
        S_global, rhs, nodes, idx, extra, pads, _ = (
            self._make_island_two_tile_system(vdd=vdd)
        )

        # Add load current at a non-island node so voltages deviate from Vdd
        rhs_loaded = rhs.copy()
        rhs_loaded[idx['I2']] -= 0.5  # 0.5 mA sink

        S_fixed, rhs_fixed, islands = detect_interface_islands(
            S_global, rhs_loaded, nodes, idx,
            pad_nodes=pads, extra_edges=extra,
            dirichlet_voltage=vdd,
        )
        self.assertGreater(len(islands), 0)

        v = spla.spsolve(S_fixed.tocsc(), rhs_fixed)

        # Island voltages close to Vdd
        for node in islands:
            self.assertAlmostEqual(
                v[idx[node]], vdd, delta=vdd * 0.001,
                msg=f"Island node {node}: {v[idx[node]]:.6f} not near Vdd",
            )

        # Non-island voltages: compare with-penalty vs without-penalty.
        # Since islands are disconnected, the penalty only modifies the
        # island diagonal block. Non-island submatrix is identical.
        non_island_indices = np.array(
            [idx[n] for n in nodes if n not in islands], dtype=np.int32,
        )
        S_sub = S_global[np.ix_(non_island_indices, non_island_indices)].tocsc()
        rhs_sub = rhs_loaded[non_island_indices]
        v_ref = spla.spsolve(S_sub, rhs_sub)

        for i, gi in enumerate(non_island_indices):
            np.testing.assert_allclose(
                v[gi], v_ref[i], atol=1e-10,
                err_msg=f"Non-island node {nodes[gi]}: perturbed by penalty",
            )

        # Verify non-island voltages actually differ from Vdd (test is non-trivial)
        non_island_voltages = [v[gi] for gi in non_island_indices]
        self.assertTrue(
            any(abs(vi - vdd) > 1e-6 for vi in non_island_voltages),
            "Non-island voltages should deviate from Vdd under load",
        )

    def test_default_penalty_is_gmax(self):
        """apply_island_penalty default penalty_conductance equals 1e5 (GMAX)."""
        import inspect
        from solver.coupled_system import apply_island_penalty

        sig = inspect.signature(apply_island_penalty)
        default = sig.parameters['penalty_conductance'].default
        self.assertEqual(default, 1e5)

    def test_kept_nonlargest_iface_returned_by_worker(self):
        """TileWorker returns kept_nonlargest_iface for qualifying components."""
        from distributed.tile_worker import TileWorker

        threshold = TileWorker.MIN_INTERFACE_NODES_KEEP

        # Build two disconnected components:
        # Main: m0-m1-...-m9 chain (10 nodes)
        # Strip: s0-s1-...-s(threshold+1) chain, with all s-nodes as interface
        lines = []
        main_nodes = [f'm{i}' for i in range(10)]
        for i in range(len(main_nodes) - 1):
            lines.append(f"R{i} {main_nodes[i]} {main_nodes[i+1]} 1000\n")
        lines.append(f"R_gnd {main_nodes[-1]} 0 1000\n")

        strip_nodes = [f's{i}' for i in range(threshold + 2)]
        for i in range(len(strip_nodes) - 1):
            lines.append(f"R{100+i} {strip_nodes[i]} {strip_nodes[i+1]} 1000\n")

        with tempfile.NamedTemporaryFile(mode='w', suffix='.ckt', delete=False) as f:
            f.writelines(lines)
            temp_ckt = f.name

        try:
            worker = TileWorker()
            interface = {'m0'} | set(strip_nodes)
            result = worker.setup(
                {'tile_id': [0, 0], 'ckt_path': temp_ckt, 'nd_path': None,
                 'instance_path': None, 'net_filter': None},
                interface_nodes=interface,
            )

            self.assertIn('kept_nonlargest_iface', result)
            kept = set(result['kept_nonlargest_iface'])
            # All strip nodes that are interface should be flagged
            for sn in strip_nodes:
                self.assertIn(sn, kept,
                              f"Strip interface node {sn} should be in kept_nonlargest_iface")
            # m0 belongs to the largest component — should NOT be flagged
            self.assertNotIn('m0', kept)
        finally:
            os.unlink(temp_ckt)


# ──────────────────────────────────────────────────────────────────────
# Ray Backend Tests
# ──────────────────────────────────────────────────────────────────────

try:
    import ray
    HAS_RAY = True
except ImportError:
    HAS_RAY = False


@unittest.skipUnless(HAS_RAY, "ray not installed")
class TestRayBackendMapFunc(unittest.TestCase):
    """Test RayBackend.map_func for correctness."""

    def test_map_func(self):
        """RayBackend.map_func returns correct results via Ray."""
        from distributed.backend import RayBackend

        def multiply(a, b):
            return a * b

        be = RayBackend()
        be.initialize()
        results = be.map_func(multiply, [(2, 3), (5, 7), (10, 0)])
        self.assertEqual(results, [6, 35, 0])


class TestLoadDistributedPartitions(unittest.TestCase):
    """Unit tests for load_distributed_partitions() error handling and basic loading."""

    def test_load_missing_metadata_raises(self):
        """load_distributed_partitions() raises FileNotFoundError on empty dir."""
        from distributed.model import load_distributed_partitions

        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(FileNotFoundError):
                load_distributed_partitions(tmpdir)

    def test_load_returns_bundle(self):
        """load_distributed_partitions() returns ParsedTileBundle from metadata.pkl."""
        import pickle
        from distributed.model import load_distributed_partitions, ParsedTileBundle
        from distributed.parser import PowerGridMetaData, PackageData

        with tempfile.TemporaryDirectory() as tmpdir:
            # Write a minimal metadata.pkl
            dummy_meta = PowerGridMetaData(
                tile_grid=(1, 1),
                parameters={},
                tile_configs=[],
                package_data=PackageData(
                    vsrc_dict={}, package_edges=[], pad_nodes=set(),
                    tap_nodes=set(), die_attachment_nodes=set(),
                    vdd=1.0, net_name='VDD',
                ),
                net_name='VDD',
                vdd=1.0,
            )
            meta_path = Path(tmpdir) / 'metadata.pkl'
            with open(meta_path, 'wb') as f:
                pickle.dump({'metadata': dummy_meta, 'boundary_nodes': set()}, f)

            bundle = load_distributed_partitions(tmpdir)
            self.assertIsInstance(bundle, ParsedTileBundle)
            self.assertEqual(bundle.metadata.net_name, 'VDD')
            self.assertEqual(bundle.shared_boundary_nodes, set())
            self.assertEqual(bundle.pkl_dir, tmpdir)


class TestTileWorkerGetLayerMetadata(unittest.TestCase):
    """Tests for TileWorker.get_layer_metadata()."""

    def _make_worker(self, tile_data):
        """Create a TileWorker with pre-loaded TileData (no block system needed)."""
        from distributed.tile_worker import TileWorker
        worker = TileWorker()
        worker._tile_data = tile_data
        return worker

    def test_single_layer_horizontal_edges(self):
        """Single layer with horizontal edges returns correct bbox and orientation."""
        from distributed.tile_worker import TileData

        td = TileData(
            tile_id=(0, 0),
            resistive_edges=[
                ('100_200_M1', '200_200_M1', 1.0),  # horizontal
                ('200_200_M1', '300_200_M1', 1.0),  # horizontal
            ],
            all_nodes={'100_200_M1', '200_200_M1', '300_200_M1'},
            boundary_nodes=set(),
            current_injections={},
        )
        worker = self._make_worker(td)
        result = worker.get_layer_metadata()

        self.assertIn('M1', result)
        m1 = result['M1']
        self.assertEqual(m1['bbox'], (100.0, 300.0, 200.0, 200.0))
        self.assertEqual(m1['n_nodes'], 3)
        # 2 horizontal edges, 0 vertical, 0 diagonal
        self.assertEqual(m1['edge_orientation'], (2, 0, 0))
        # stripe_coords_h: sorted unique Y values
        self.assertEqual(list(m1['stripe_coords_h']), [200.0])
        # stripe_coords_v: sorted unique X values
        self.assertEqual(list(m1['stripe_coords_v']), [100.0, 200.0, 300.0])

    def test_single_layer_vertical_edges(self):
        """Single layer with vertical edges returns correct orientation."""
        from distributed.tile_worker import TileData

        td = TileData(
            tile_id=(0, 0),
            resistive_edges=[
                ('100_200_M2', '100_300_M2', 1.0),  # vertical
                ('100_300_M2', '100_400_M2', 1.0),  # vertical
            ],
            all_nodes={'100_200_M2', '100_300_M2', '100_400_M2'},
            boundary_nodes=set(),
            current_injections={},
        )
        worker = self._make_worker(td)
        result = worker.get_layer_metadata()

        self.assertIn('M2', result)
        m2 = result['M2']
        self.assertEqual(m2['bbox'], (100.0, 100.0, 200.0, 400.0))
        self.assertEqual(m2['n_nodes'], 3)
        self.assertEqual(m2['edge_orientation'], (0, 2, 0))

    def test_multi_layer(self):
        """Multiple layers each get separate metadata."""
        from distributed.tile_worker import TileData

        td = TileData(
            tile_id=(1, 2),
            resistive_edges=[
                ('10_20_M1', '20_20_M1', 1.0),
                ('10_20_M3', '10_30_M3', 2.0),
            ],
            all_nodes={'10_20_M1', '20_20_M1', '10_20_M3', '10_30_M3'},
            boundary_nodes=set(),
            current_injections={},
        )
        worker = self._make_worker(td)
        result = worker.get_layer_metadata()

        self.assertEqual(set(result.keys()), {'M1', 'M3'})
        self.assertEqual(result['M1']['n_nodes'], 2)
        self.assertEqual(result['M3']['n_nodes'], 2)

    def test_diagonal_edges_counted(self):
        """Edges where both X and Y differ are classified as diagonal."""
        from distributed.tile_worker import TileData

        td = TileData(
            tile_id=(0, 0),
            resistive_edges=[
                ('100_200_M1', '200_300_M1', 1.0),  # diagonal
            ],
            all_nodes={'100_200_M1', '200_300_M1'},
            boundary_nodes=set(),
            current_injections={},
        )
        worker = self._make_worker(td)
        result = worker.get_layer_metadata()

        self.assertEqual(result['M1']['edge_orientation'], (0, 0, 1))

    def test_cross_layer_edges_ignored_in_orientation(self):
        """Edges between different layers (vias) are not counted in orientation."""
        from distributed.tile_worker import TileData

        td = TileData(
            tile_id=(0, 0),
            resistive_edges=[
                ('100_200_M1', '100_200_M2', 5.0),  # via: same X,Y, different layer
                ('100_200_M1', '200_200_M1', 1.0),  # horizontal M1 edge
            ],
            all_nodes={'100_200_M1', '100_200_M2', '200_200_M1'},
            boundary_nodes=set(),
            current_injections={},
        )
        worker = self._make_worker(td)
        result = worker.get_layer_metadata()

        # Via should NOT be counted in either layer's orientation
        self.assertEqual(result['M1']['edge_orientation'], (1, 0, 0))
        # M2 has no same-layer edges
        self.assertEqual(result['M2']['edge_orientation'], (0, 0, 0))

    def test_unparseable_nodes_excluded(self):
        """Nodes that don't match X_Y_LAYER format are silently skipped."""
        from distributed.tile_worker import TileData

        td = TileData(
            tile_id=(0, 0),
            resistive_edges=[
                ('100_200_M1', 'VDD_vsrc', 1.0),
                ('100_200_M1', '0', 0.5),
            ],
            all_nodes={'100_200_M1', 'VDD_vsrc', '0'},
            boundary_nodes=set(),
            current_injections={},
        )
        worker = self._make_worker(td)
        result = worker.get_layer_metadata()

        self.assertIn('M1', result)
        self.assertEqual(result['M1']['n_nodes'], 1)
        # No same-layer edges (VDD_vsrc and 0 don't parse to any layer)
        self.assertEqual(result['M1']['edge_orientation'], (0, 0, 0))

    def test_empty_tile(self):
        """Empty tile returns empty dict."""
        from distributed.tile_worker import TileData

        td = TileData(
            tile_id=(0, 0),
            resistive_edges=[],
            all_nodes=set(),
            boundary_nodes=set(),
            current_injections={},
        )
        worker = self._make_worker(td)
        result = worker.get_layer_metadata()

        self.assertEqual(result, {})

    def test_stripe_coords_sorted_unique(self):
        """Stripe coords are sorted and deduplicated."""
        from distributed.tile_worker import TileData

        td = TileData(
            tile_id=(0, 0),
            resistive_edges=[
                ('100_200_M1', '200_200_M1', 1.0),
                ('100_300_M1', '200_300_M1', 1.0),
                ('100_200_M1', '100_300_M1', 1.0),
            ],
            all_nodes={
                '100_200_M1', '200_200_M1',
                '100_300_M1', '200_300_M1',
            },
            boundary_nodes=set(),
            current_injections={},
        )
        worker = self._make_worker(td)
        result = worker.get_layer_metadata()

        m1 = result['M1']
        self.assertEqual(list(m1['stripe_coords_h']), [200.0, 300.0])
        self.assertEqual(list(m1['stripe_coords_v']), [100.0, 200.0])

    def test_not_setup_raises(self):
        """Calling get_layer_metadata before setup raises RuntimeError."""
        from distributed.tile_worker import TileWorker

        worker = TileWorker()
        with self.assertRaises(RuntimeError):
            worker.get_layer_metadata()


class TestTileWorkerGetCurrentInjections(unittest.TestCase):
    """Tests for TileWorker.get_current_injections()."""

    def _make_worker(self, tile_data):
        from distributed.tile_worker import TileWorker
        worker = TileWorker()
        worker._tile_data = tile_data
        return worker

    def test_returns_copy(self):
        """get_current_injections returns a new dict (not a reference)."""
        from distributed.tile_worker import TileData

        td = TileData(
            tile_id=(0, 0),
            resistive_edges=[],
            all_nodes={'a'},
            boundary_nodes=set(),
            current_injections={'a': 1.5},
        )
        worker = self._make_worker(td)
        result = worker.get_current_injections()

        self.assertEqual(result, {'a': 1.5})
        # Mutating the returned dict should not affect the tile data
        result['a'] = 999.0
        self.assertEqual(worker.get_current_injections(), {'a': 1.5})

    def test_empty_injections(self):
        """Tile with no current sources returns empty dict."""
        from distributed.tile_worker import TileData

        td = TileData(
            tile_id=(0, 0),
            resistive_edges=[],
            all_nodes=set(),
            boundary_nodes=set(),
            current_injections={},
        )
        worker = self._make_worker(td)
        self.assertEqual(worker.get_current_injections(), {})

    def test_not_setup_raises(self):
        """Calling get_current_injections before setup raises RuntimeError."""
        from distributed.tile_worker import TileWorker

        worker = TileWorker()
        with self.assertRaises(RuntimeError):
            worker.get_current_injections()

    def test_preserves_values(self):
        """Returned dict preserves exact injection values."""
        from distributed.tile_worker import TileData

        injections = {
            '100_200_M1': 0.5,
            '200_300_M1': -0.123,
            '300_400_M2': 42.0,
        }
        td = TileData(
            tile_id=(1, 1),
            resistive_edges=[],
            all_nodes=set(injections.keys()),
            boundary_nodes=set(),
            current_injections=injections,
        )
        worker = self._make_worker(td)
        result = worker.get_current_injections()

        for node, expected_val in injections.items():
            self.assertEqual(result[node], expected_val)


# ──────────────────────────────────────────────────────────────────────
# Package Parsing Tests
# ──────────────────────────────────────────────────────────────────────

class TestPackageParsing(unittest.TestCase):
    """Tests for _is_die_coordinate_node, _uf_find/_uf_union, and _parse_package."""

    # ------------------------------------------------------------------
    # Test 1: _is_die_coordinate_node
    # ------------------------------------------------------------------

    def test_is_die_coordinate_node(self):
        """Verify die coordinate detection for various node name formats."""
        from distributed.parser import _is_die_coordinate_node

        # True cases: first two _-delimited parts are digits
        self.assertTrue(_is_die_coordinate_node('1094400_1123200_M13'))  # sampled
        self.assertTrue(_is_die_coordinate_node('1197000_449800_86'))    # brcm
        self.assertTrue(_is_die_coordinate_node('0_0_M3'))              # multi_tile
        self.assertTrue(_is_die_coordinate_node('123_456'))             # minimal

        # False cases: non-coordinate patterns
        self.assertFalse(_is_die_coordinate_node('bmpary_bmp_VDD_VAR_0_1'))  # brcm pkg infra
        self.assertFalse(_is_die_coordinate_node('VDD_XLV_tap_00000'))       # sampled tap
        self.assertFalse(_is_die_coordinate_node('VDD_XLV_vsrc'))            # sampled vsrc
        self.assertFalse(_is_die_coordinate_node('0'))                       # ground
        self.assertFalse(_is_die_coordinate_node(''))                        # empty

    # ------------------------------------------------------------------
    # Test 2: _uf_find and _uf_union
    # ------------------------------------------------------------------

    def test_uf_find_and_union(self):
        """Union-find operations: find, union, net propagation, isolation, compression."""
        from distributed.parser import _uf_find, _uf_union

        parent = {}
        uf_net = {}

        # Basic find: new node returns itself as root
        self.assertEqual(_uf_find(parent, 'A'), 'A')
        self.assertIn('A', parent)
        self.assertEqual(parent['A'], 'A')

        # Union two nodes: they share the same root after union
        _uf_find(parent, 'B')
        _uf_union(parent, uf_net, 'A', 'B')
        self.assertEqual(_uf_find(parent, 'A'), _uf_find(parent, 'B'))

        # Net propagation: label one root, union propagates it
        parent2 = {}
        uf_net2 = {}
        _uf_find(parent2, 'X')
        uf_net2[_uf_find(parent2, 'X')] = 'VDD'
        _uf_find(parent2, 'Y')
        _uf_union(parent2, uf_net2, 'X', 'Y')
        root_y = _uf_find(parent2, 'Y')
        self.assertEqual(uf_net2.get(root_y), 'VDD')

        # Multi-component isolation: two disconnected groups don't share roots
        parent3 = {}
        uf_net3 = {}
        _uf_union(parent3, uf_net3, 'P', 'Q')
        _uf_union(parent3, uf_net3, 'R', 'S')
        self.assertNotEqual(_uf_find(parent3, 'P'), _uf_find(parent3, 'R'))

        # Path compression: after find, parent points directly to root
        parent4 = {}
        uf_net4 = {}
        # Build a chain: A -> B -> C -> D
        _uf_find(parent4, 'D')
        _uf_find(parent4, 'C')
        _uf_find(parent4, 'B')
        _uf_find(parent4, 'A')
        _uf_union(parent4, uf_net4, 'C', 'D')
        _uf_union(parent4, uf_net4, 'B', 'C')
        _uf_union(parent4, uf_net4, 'A', 'B')
        root = _uf_find(parent4, 'A')
        # After find, A should point directly to root (path compression)
        self.assertEqual(parent4['A'], root)

        # Conflicting nets: first (root1) net wins, second is silently dropped
        parent5 = {}
        uf_net5 = {}
        _uf_find(parent5, 'V')
        uf_net5['V'] = 'VDD'
        _uf_find(parent5, 'S')
        uf_net5['S'] = 'VSS'
        _uf_union(parent5, uf_net5, 'V', 'S')
        merged_root = _uf_find(parent5, 'S')
        self.assertEqual(uf_net5[merged_root], 'VDD')

    # ------------------------------------------------------------------
    # Helper: create parser + package.ckt in tmpdir
    # ------------------------------------------------------------------

    def _make_parser_with_package(self, package_content):
        """Create a tmpdir with package.ckt and return (tmpdir, parser).

        Caller must manage tmpdir lifetime (use as context manager or
        call cleanup).
        """
        from distributed.parser import DistributedNetlistParser

        tmpdir = tempfile.mkdtemp()
        pkg_path = os.path.join(tmpdir, 'package.ckt')
        with open(pkg_path, 'w') as f:
            f.write(package_content)
        parser = DistributedNetlistParser(tmpdir)
        return tmpdir, parser

    # ------------------------------------------------------------------
    # Test 3: _parse_package — sampled format
    # ------------------------------------------------------------------

    def test_parse_package_sampled_format(self):
        """Parse sampled-style package with vsrc -> tap -> die node chain."""
        content = (
            "* Package model for VDD_XLV\n"
            "v_VDD_XLV VDD_XLV_vsrc 0 VDD_XLV\n"
            "r VDD_XLV_vsrc VDD_XLV_tap_00000 0.001\n"
            "r VDD_XLV_tap_00000 1094400_1123200_M13 0.001\n"
        )
        tmpdir, parser = self._make_parser_with_package(content)
        try:
            pkg = parser._parse_package('VDD_XLV', 0.66)

            self.assertIn('VDD_XLV_vsrc', pkg.pad_nodes)
            self.assertEqual(len(pkg.vsrc_dict), 1)
            self.assertIn('v_VDD_XLV', pkg.vsrc_dict)
            self.assertEqual(len(pkg.package_edges), 2)
            self.assertIn('1094400_1123200_M13', pkg.die_attachment_nodes)
            self.assertIn('VDD_XLV_tap_00000', pkg.tap_nodes)
        finally:
            import shutil
            shutil.rmtree(tmpdir)

    # ------------------------------------------------------------------
    # Test 4: _parse_package — brcm format
    # ------------------------------------------------------------------

    def test_parse_package_brcm_format(self):
        """Parse brcm-style package with probe/int/vsrc infra nodes."""
        content = (
            "v_bmpary_bmp_VDD_VAR_0_1 bmpary_bmp_VDD_VAR_0_1_vsrc 0 VDD_VAR\n"
            "r bmpary_bmp_VDD_VAR_0_1 bmpary_bmp_VDD_VAR_0_1_probe 0\n"
            "r bmpary_bmp_VDD_VAR_0_1_probe bmpary_bmp_VDD_VAR_0_1_int 0.001\n"
            "r bmpary_bmp_VDD_VAR_0_1_int bmpary_bmp_VDD_VAR_0_1_vsrc 0\n"
            "rs 1197000_449800_86 bmpary_bmp_VDD_VAR_0_1 0\n"
        )
        tmpdir, parser = self._make_parser_with_package(content)
        try:
            pkg = parser._parse_package('VDD_VAR', 0.75)

            self.assertIn('bmpary_bmp_VDD_VAR_0_1_vsrc', pkg.pad_nodes)
            self.assertIn('1197000_449800_86', pkg.die_attachment_nodes)
            self.assertIn('bmpary_bmp_VDD_VAR_0_1_probe', pkg.tap_nodes)
            self.assertIn('bmpary_bmp_VDD_VAR_0_1_int', pkg.tap_nodes)
            self.assertIn('bmpary_bmp_VDD_VAR_0_1', pkg.tap_nodes)
            self.assertEqual(len(pkg.package_edges), 4)  # 3 r + 1 rs
        finally:
            import shutil
            shutil.rmtree(tmpdir)

    # ------------------------------------------------------------------
    # Test 5: _parse_package — multi-net separation
    # ------------------------------------------------------------------

    def test_parse_package_multi_net_separation(self):
        """VDD elements filtered; VSS excluded. Die nodes are global candidates."""
        content = (
            "v_VDD VDD_vsrc 0 VDD\n"
            "r VDD_vsrc VDD_tap 0.001\n"
            "r VDD_tap 100_200_M13 0.001\n"
            "v_VSS VSS_vsrc 0 VSS\n"
            "r VSS_vsrc VSS_tap 0.001\n"
            "r VSS_tap 300_400_M13 0.001\n"
        )
        tmpdir, parser = self._make_parser_with_package(content)
        try:
            pkg = parser._parse_package('VDD', 1.0)

            # Only VDD vsrc
            self.assertEqual(pkg.pad_nodes, {'VDD_vsrc'})
            # Only VDD resistor edges
            self.assertEqual(len(pkg.package_edges), 2)
            # Die attachment nodes are ALL coordinate nodes (global candidates)
            self.assertIn('100_200_M13', pkg.die_attachment_nodes)
            self.assertIn('300_400_M13', pkg.die_attachment_nodes)
            # Tap nodes only from VDD
            self.assertEqual(pkg.tap_nodes, {'VDD_tap'})
        finally:
            import shutil
            shutil.rmtree(tmpdir)

    # ------------------------------------------------------------------
    # Test 6: _parse_package — die_net_map fallback
    # ------------------------------------------------------------------

    def test_parse_package_die_net_map_seeding(self):
        """die_net_map seeds net labels on vsrc-less components, enabling edge filtering."""
        # Two disconnected components in the package:
        #   Component A: vsrc for VDD -> VDD_vsrc -- 100_200_M13 (has net label)
        #   Component B: extra_node -- 200_300_M13 (NO vsrc, no net label)
        content = (
            "v_VDD VDD_vsrc 0 VDD\n"
            "r VDD_vsrc 100_200_M13 0.001\n"
            "r extra_node 200_300_M13 0.001\n"
        )
        tmpdir, parser = self._make_parser_with_package(content)
        try:
            # Without die_net_map: only component A (VDD) edges are filtered
            pkg_no_map = parser._parse_package('VDD', 1.0)
            self.assertEqual(len(pkg_no_map.package_edges), 1)
            self.assertIn('VDD_vsrc', pkg_no_map.pad_nodes)

            # With die_net_map: component B's die node seeds VDD label,
            # so component B's edge is now also included
            pkg_with_map = parser._parse_package(
                'VDD', 1.0,
                die_net_map={'200_300_M13': 'VDD'},
            )
            self.assertEqual(len(pkg_with_map.package_edges), 2)
            # Die attachment nodes include both coordinate nodes
            self.assertIn('100_200_M13', pkg_with_map.die_attachment_nodes)
            self.assertIn('200_300_M13', pkg_with_map.die_attachment_nodes)
            # Tap nodes: extra_node is now filtered in (non-die, non-pad)
            self.assertIn('extra_node', pkg_with_map.tap_nodes)
        finally:
            import shutil
            shutil.rmtree(tmpdir)

    # ------------------------------------------------------------------
    # Test 6b: _parse_package — true zero-pads fallback with die_net_map
    # ------------------------------------------------------------------

    def test_parse_package_die_net_map_no_vsrc_fallback(self):
        """When no vsrc matches the target net, die_net_map enables filtering."""
        # vsrc declares VDD_CORE, but we're looking for VDD_IO
        content = (
            "v_VDD_CORE VDD_CORE_vsrc 0 VDD_CORE\n"
            "r VDD_CORE_vsrc 100_200_M13 0.001\n"
            "r pkg_node 300_400_M13 0.001\n"
        )
        tmpdir, parser = self._make_parser_with_package(content)
        try:
            # Without die_net_map: no VDD_IO pads or edges
            pkg_no_map = parser._parse_package('VDD_IO', 1.0)
            self.assertEqual(len(pkg_no_map.pad_nodes), 0)
            self.assertEqual(len(pkg_no_map.package_edges), 0)

            # With die_net_map seeding VDD_IO on component B
            pkg_with_map = parser._parse_package(
                'VDD_IO', 1.0,
                die_net_map={'300_400_M13': 'VDD_IO'},
            )
            # Component B's edge is now included
            self.assertEqual(len(pkg_with_map.package_edges), 1)
            # Still no vsrc for VDD_IO, so pad_nodes remains empty
            self.assertEqual(len(pkg_with_map.pad_nodes), 0)
        finally:
            import shutil
            shutil.rmtree(tmpdir)

    # ------------------------------------------------------------------
    # Test 7: _parse_package — numeric vsrc value (4th token is float)
    # ------------------------------------------------------------------

    def test_parse_package_numeric_vsrc_value(self):
        """Numeric 4th token on vsrc: net inferred from element name."""
        content = (
            "V_VDD VDD_vrm 0 0.75\n"
            "R_vrm VDD_vrm VDD_pkg 0.001\n"
            "R_conn VDD_pkg 0_0_M3 0.02\n"
        )
        tmpdir, parser = self._make_parser_with_package(content)
        try:
            pkg = parser._parse_package('VDD', 0.75)

            self.assertIn('VDD_vrm', pkg.pad_nodes)
            self.assertEqual(len(pkg.package_edges), 2)
            self.assertIn('0_0_M3', pkg.die_attachment_nodes)
        finally:
            import shutil
            shutil.rmtree(tmpdir)

    # ------------------------------------------------------------------
    # Test 8: _parse_package — inductor union-find propagation
    # ------------------------------------------------------------------

    def test_parse_package_inductor_union(self):
        """Inductors participate in union-find and become GMAX shorts in package_edges."""
        content = (
            "V_VDD VDD_vrm 0 0.75\n"
            "R_vrm VDD_vrm pkg_bump 0.001\n"
            "L_bump pkg_bump pkg_bump_l 0.05e-9\n"
            "R_bump pkg_bump_l 100_200_M3 0.02\n"
        )
        tmpdir, parser = self._make_parser_with_package(content)
        try:
            pkg = parser._parse_package('VDD', 0.75)

            # 2 R + 1 L (as GMAX short)
            self.assertEqual(len(pkg.package_edges), 3)
            self.assertIn('100_200_M3', pkg.die_attachment_nodes)

            # All nodes reachable from VDD_vrm through inductor
            self.assertIn('VDD_vrm', pkg.pad_nodes)

            # Verify inductor edge has GMAX conductance (1e5 mS)
            inductor_edges = [
                (u, v, g) for u, v, g in pkg.package_edges
                if {u, v} == {'pkg_bump', 'pkg_bump_l'}
            ]
            self.assertEqual(len(inductor_edges), 1)
            self.assertAlmostEqual(inductor_edges[0][2], 1e5)
        finally:
            import shutil
            shutil.rmtree(tmpdir)


class TestLookupInstanceNames(unittest.TestCase):
    """Tests for TileWorker.lookup_instance_names()."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix='lookup_inst_test_')
        self.worker = None

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _get_worker(self):
        """Lazy-create a TileWorker (no setup() needed for lookup_instance_names)."""
        if self.worker is None:
            from distributed.tile_worker import TileWorker
            self.worker = TileWorker()
        return self.worker

    def _write_instance_file(self, lines):
        """Write instance model lines to a temp file and return its path."""
        path = os.path.join(self.tmpdir, 'instanceModels_0_0.sp')
        with open(path, 'w') as f:
            f.write('\n'.join(lines) + '\n')
        return path

    # ------------------------------------------------------------------
    # 1. Basic lookup
    # ------------------------------------------------------------------
    def test_basic_lookup(self):
        """Known instance file content maps to correct node->instance entries."""
        instance_path = self._write_instance_file([
            'I_inst1 1000_2000_M1 0 DC 0.001',
            'I_inst2 1000_2100_M1 0 DC 0.002',
            'I_inst3 2000_3000_M2 0 DC 0.003',
        ])
        target_nodes = {'1000_2000_M1', '1000_2100_M1', '2000_3000_M2'}

        result = self._get_worker().lookup_instance_names(
            target_nodes=target_nodes,
            instance_path=instance_path,
        )

        self.assertEqual(len(result), 3)
        self.assertEqual(result['1000_2000_M1'], 'I_inst1')
        self.assertEqual(result['1000_2100_M1'], 'I_inst2')
        self.assertEqual(result['2000_3000_M2'], 'I_inst3')

    # ------------------------------------------------------------------
    # 2. None instance_path returns empty dict
    # ------------------------------------------------------------------
    def test_none_instance_path_returns_empty(self):
        """When instance_path is None, return empty dict immediately."""
        result = self._get_worker().lookup_instance_names(
            target_nodes={'1000_2000_M1'},
            instance_path=None,
        )
        self.assertEqual(result, {})

    # ------------------------------------------------------------------
    # 3. Target node filtering
    # ------------------------------------------------------------------
    def test_target_node_filtering(self):
        """Only nodes in target_nodes appear in the result."""
        instance_path = self._write_instance_file([
            'I_inst1 1000_2000_M1 0 DC 0.001',
            'I_inst2 1000_2100_M1 0 DC 0.002',
            'I_inst3 2000_3000_M2 0 DC 0.003',
        ])
        # Only ask for one of the three nodes
        target_nodes = {'1000_2100_M1'}

        result = self._get_worker().lookup_instance_names(
            target_nodes=target_nodes,
            instance_path=instance_path,
        )

        self.assertEqual(len(result), 1)
        self.assertIn('1000_2100_M1', result)
        self.assertNotIn('1000_2000_M1', result)
        self.assertNotIn('2000_3000_M2', result)

    # ------------------------------------------------------------------
    # 4. No matching nodes
    # ------------------------------------------------------------------
    def test_no_matching_nodes_returns_empty(self):
        """When target_nodes has no overlap with file content, return empty."""
        instance_path = self._write_instance_file([
            'I_inst1 1000_2000_M1 0 DC 0.001',
            'I_inst2 1000_2100_M1 0 DC 0.002',
        ])
        target_nodes = {'9999_9999_M9', 'nonexistent_node'}

        result = self._get_worker().lookup_instance_names(
            target_nodes=target_nodes,
            instance_path=instance_path,
        )

        self.assertEqual(result, {})

    # ------------------------------------------------------------------
    # 5. Last-wins semantics for duplicate nodes
    # ------------------------------------------------------------------
    def test_duplicate_node_last_wins(self):
        """Multiple instances mapping to the same node: last one wins."""
        instance_path = self._write_instance_file([
            'I_first 1000_2000_M1 0 DC 0.001',
            'I_second 1000_2000_M1 0 DC 0.002',
        ])
        target_nodes = {'1000_2000_M1'}

        result = self._get_worker().lookup_instance_names(
            target_nodes=target_nodes,
            instance_path=instance_path,
        )

        self.assertEqual(result['1000_2000_M1'], 'I_second')

    # ------------------------------------------------------------------
    # 6. Comments and blank lines are skipped
    # ------------------------------------------------------------------
    def test_comments_and_blanks_skipped(self):
        """Lines starting with * or . and blank lines should be ignored."""
        instance_path = self._write_instance_file([
            '* This is a comment',
            '.subckt something',
            '',
            'I_inst1 1000_2000_M1 0 DC 0.001',
            '* Another comment',
            'I_inst2 2000_3000_M2 0 DC 0.002',
        ])
        target_nodes = {'1000_2000_M1', '2000_3000_M2'}

        result = self._get_worker().lookup_instance_names(
            target_nodes=target_nodes,
            instance_path=instance_path,
        )

        self.assertEqual(len(result), 2)
        self.assertEqual(result['1000_2000_M1'], 'I_inst1')
        self.assertEqual(result['2000_3000_M2'], 'I_inst2')


if __name__ == '__main__':
    unittest.main()

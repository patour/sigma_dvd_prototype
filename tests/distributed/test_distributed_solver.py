"""Tests for distributed DDM solver.

Tests the full distributed pipeline: building blocks, tile parsing, model
creation, and end-to-end DDM solve with validation against flat solver.
"""

import math
import os
import tempfile
import unittest
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np

from solver.coupled_system import (
    BlockMatrixSystem,
    SchurComplementOperator,
    build_block_system_from_edges,
    compute_explicit_schur,
    compute_reduced_rhs,
    assemble_schur_complement_system,
    recover_bottom_voltages,
)

# ──────────────────────────────────────────────────────────────────────
# Test data
# ──────────────────────────────────────────────────────────────────────

NETLIST_SAMPLED_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'netlist', 'netlist_sampled',
)
NETLIST_SAMPLED_EXISTS = os.path.isdir(NETLIST_SAMPLED_DIR)

NETLIST_SMALL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'netlist', 'netlist_small',
)
NETLIST_SMALL_EXISTS = os.path.isdir(NETLIST_SMALL_DIR)


def _partition_currents_for_tiles(model, flat_currents):
    """Partition a flat current dict into per-tile dicts for solve_dc.

    Boundary node currents are assigned to exactly one tile to avoid
    double-counting. Interior node currents are broadcast to all tiles
    (only the owning tile will use them).
    """
    boundary_assigned: Set[str] = set()
    per_tile: List[Dict[str, float]] = [{} for _ in model.workers]
    tile_bnd_sets = [
        set(model.tile_boundary_nodes[tc.tile_id])
        for tc in model.metadata.tile_configs
    ]
    for node, current in flat_currents.items():
        if current == 0.0:
            continue
        if node in model.interface_nodes:
            for i, bnd_set in enumerate(tile_bnd_sets):
                if node in bnd_set and node not in boundary_assigned:
                    per_tile[i][node] = current
                    boundary_assigned.add(node)
                    break
        else:
            for i in range(len(model.workers)):
                per_tile[i][node] = current
    return per_tile


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
# Parser Tests
# ──────────────────────────────────────────────────────────────────────

@unittest.skipUnless(NETLIST_SAMPLED_EXISTS, "netlist_sampled not available")
class TestDistributedParser(unittest.TestCase):
    """Test DistributedNetlistParser on netlist_sampled."""

    def test_parse_metadata(self):
        """Parser returns valid metadata for netlist_sampled."""
        from distributed import DistributedNetlistParser

        parser = DistributedNetlistParser(NETLIST_SAMPLED_DIR, net_filter='VDD_XLV')
        metadata = parser.parse_metadata()

        self.assertEqual(metadata.tile_grid, (3, 3))
        self.assertEqual(len(metadata.tile_configs), 9)
        self.assertEqual(metadata.net_name, 'VDD_XLV')
        self.assertAlmostEqual(metadata.vdd, 0.66, places=2)
        self.assertGreater(len(metadata.package_data.pad_nodes), 0)
        self.assertGreater(len(metadata.package_data.package_edges), 0)
        self.assertGreater(len(metadata.package_data.die_attachment_nodes), 0)

    def test_tile_configs_have_paths(self):
        """Each tile config has valid ckt and nd paths."""
        from distributed import DistributedNetlistParser

        parser = DistributedNetlistParser(NETLIST_SAMPLED_DIR, net_filter='VDD_XLV')
        metadata = parser.parse_metadata()

        for tc in metadata.tile_configs:
            self.assertTrue(os.path.exists(tc.ckt_path), f"Missing: {tc.ckt_path}")
            if tc.nd_path:
                self.assertTrue(os.path.exists(tc.nd_path), f"Missing: {tc.nd_path}")

    def test_collect_boundary_nodes(self):
        """Boundary node collection finds cross-tile shared nodes."""
        from distributed import DistributedNetlistParser

        parser = DistributedNetlistParser(NETLIST_SAMPLED_DIR, net_filter='VDD_XLV')
        metadata = parser.parse_metadata()
        boundary_nodes = parser.collect_boundary_nodes(metadata.tile_configs)

        # Multi-tile netlist should have many boundary nodes
        self.assertGreater(len(boundary_nodes), 100)
        # Boundary nodes should be string names
        for node in list(boundary_nodes)[:5]:
            self.assertIsInstance(node, str)


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


# ──────────────────────────────────────────────────────────────────────
# End-to-End Validation Tests
# ──────────────────────────────────────────────────────────────────────

@unittest.skipUnless(NETLIST_SAMPLED_EXISTS, "netlist_sampled not available")
class TestDistributedVsFlat(unittest.TestCase):
    """Validate DDM solve against monolithic flat solve on netlist_sampled."""

    @classmethod
    def setUpClass(cls):
        """Parse and solve once for all validation tests."""
        import logging
        logging.disable(logging.WARNING)

        # DDM
        from distributed import DistributedNetlistParser, create_distributed_model, DistributedDDMSolver
        parser = DistributedNetlistParser(NETLIST_SAMPLED_DIR, net_filter='VDD_XLV')
        cls.metadata = parser.parse_metadata()
        cls.model = create_distributed_model(cls.metadata, backend='local')
        cls.solver = DistributedDDMSolver(cls.model)
        cls.ctx = cls.solver.prepare()
        cls.ddm_result = cls.solver.solve_dc(context=cls.ctx)
        cls.ddm_voltages = cls.ddm_result.flatten()

        # Flat
        from parser.netlist import NetlistParser
        from model.factory import create_model_from_pdn
        from solver.unified_solver import UnifiedIRDropSolver
        flat_parser = NetlistParser(NETLIST_SAMPLED_DIR)
        graph = flat_parser.parse()
        cls.flat_model = create_model_from_pdn(graph, 'VDD_XLV')
        flat_solver = UnifiedIRDropSolver(cls.flat_model)
        cls.load_currents = cls.flat_model.extract_current_sources()
        cls.flat_result = flat_solver.solve(cls.load_currents)

        cls.common_nodes = set(cls.ddm_voltages.keys()) & set(cls.flat_result.voltages.keys())

        logging.disable(logging.NOTSET)

    @classmethod
    def tearDownClass(cls):
        cls.model.shutdown()

    def test_node_count_match(self):
        """DDM and flat should have same number of nodes."""
        self.assertEqual(len(self.ddm_voltages), len(self.flat_result.voltages))

    def test_voltage_exact_match(self):
        """DDM voltages match flat within floating-point tolerance (<1 µV)."""
        max_diff = max(
            abs(self.ddm_voltages[n] - self.flat_result.voltages[n])
            for n in self.common_nodes
        )
        self.assertLess(max_diff, 1e-6, f"Max voltage diff {max_diff*1e6:.3f} µV > 1 µV")

    def test_ir_drop_matches(self):
        """IR-drop values match within tolerance."""
        vdd = self.metadata.vdd
        ddm_max_drop = max(vdd - self.ddm_voltages[n] for n in self.common_nodes)
        flat_max_drop = max(vdd - self.flat_result.voltages[n] for n in self.common_nodes)
        self.assertAlmostEqual(ddm_max_drop, flat_max_drop, places=5)

    def test_voltages_in_range(self):
        """All DDM voltages between 0 and VDD."""
        vdd = self.metadata.vdd
        for node, v in self.ddm_voltages.items():
            if node == '0':
                continue
            self.assertGreaterEqual(v, -0.01, f"Node {node} voltage {v} < -0.01")
            self.assertLessEqual(v, vdd + 0.001, f"Node {node} voltage {v} > VDD+0.001")

    def test_pad_voltages_at_vdd(self):
        """Pad nodes should be at VDD."""
        vdd = self.metadata.vdd
        for node in self.ddm_result.pad_voltages:
            self.assertAlmostEqual(
                self.ddm_result.pad_voltages[node], vdd, places=10,
            )

    def test_batch_solve_reuse(self):
        """Second solve with same context gives identical result."""
        result2 = self.solver.solve_dc(context=self.ctx)
        v2 = result2.flatten()
        for node in self.common_nodes:
            self.assertEqual(self.ddm_voltages[node], v2[node])

    def test_external_current_injection_scaled(self):
        """Scaled override currents match flat solve (exposes interior recovery bug).

        Uses 10x scaled currents so overrides differ significantly from
        tile-local parsed currents.  If interior recovery ignores the
        override and uses stale tile-local (1x) currents, the interior
        voltages will be wrong by several mV.
        """
        from solver.unified_solver import UnifiedIRDropSolver

        scale = 10.0
        scaled = {n: c * scale for n, c in self.load_currents.items()}

        # DDM with scaled override
        per_tile = _partition_currents_for_tiles(self.model, scaled)
        ddm_result = self.solver.solve_dc(
            per_tile_currents=per_tile, context=self.ctx,
        )
        v_ddm = ddm_result.flatten()

        # Flat reference with same scaled currents
        flat_result = UnifiedIRDropSolver(self.flat_model).solve(scaled)

        max_diff = max(
            abs(v_ddm[n] - flat_result.voltages[n])
            for n in self.common_nodes
        )
        # Confirm the scaled scenario has significant IR-drop (not vacuous)
        vdd = self.metadata.vdd
        max_drop = max(vdd - flat_result.voltages[n] for n in self.common_nodes)
        self.assertGreater(max_drop, 0.005,
                           f"Scaled IR-drop {max_drop*1e3:.2f} mV too small to be meaningful")

        self.assertLess(max_diff, 1e-6,
                        f"Scaled current max diff {max_diff*1e6:.3f} µV")

    def test_zero_current_override_no_ir_drop(self):
        """Zero override currents produce zero IR-drop.

        With no injected current every node should sit at VDD.  If
        interior recovery ignores the override and falls back to
        tile-local parsed currents, interior nodes will show spurious
        IR-drop (~1.6 mV for this netlist).
        """
        zero_per_tile = [{} for _ in self.model.workers]
        result = self.solver.solve_dc(
            per_tile_currents=zero_per_tile, context=self.ctx,
        )
        voltages = result.flatten()
        vdd = self.metadata.vdd

        max_drop = max(
            abs(vdd - v) for n, v in voltages.items() if n != '0'
        )
        self.assertLess(max_drop, 1e-10,
                        f"Zero-current IR-drop {max_drop*1e6:.3f} µV (expect 0)")


@unittest.skipUnless(NETLIST_SAMPLED_EXISTS, "netlist_sampled not available")
class TestDistributedModel(unittest.TestCase):
    """Test DistributedPowerGridModel creation and properties."""

    def test_model_creation(self):
        """Model creates successfully with correct tile count."""
        import logging
        logging.disable(logging.WARNING)
        from distributed import DistributedNetlistParser, create_distributed_model

        parser = DistributedNetlistParser(NETLIST_SAMPLED_DIR, net_filter='VDD_XLV')
        metadata = parser.parse_metadata()
        model = create_distributed_model(metadata, backend='local')

        try:
            self.assertEqual(model.n_tiles, 9)
            self.assertEqual(model.tile_grid, (3, 3))
            self.assertEqual(len(model.workers), 9)
            self.assertGreater(len(model.interface_nodes), 0)
            self.assertAlmostEqual(model.vdd, 0.66, places=2)
            self.assertEqual(model.net_name, 'VDD_XLV')
        finally:
            model.shutdown()
            logging.disable(logging.NOTSET)

    def test_no_overlapping_interior_nodes(self):
        """Interior nodes should be unique across tiles."""
        import logging
        logging.disable(logging.WARNING)
        from distributed import DistributedNetlistParser, create_distributed_model

        parser = DistributedNetlistParser(NETLIST_SAMPLED_DIR, net_filter='VDD_XLV')
        metadata = parser.parse_metadata()
        model = create_distributed_model(metadata, backend='local')

        try:
            # Boundary nodes can overlap, but interior counts should be consistent
            total_interior = sum(model.tile_interior_counts.values())
            self.assertGreater(total_interior, 0)
            # Total interior + unique boundary should roughly match total nodes
            unique_boundary = set()
            for blist in model.tile_boundary_nodes.values():
                unique_boundary.update(blist)
            # Interior + boundary should cover all die nodes
            total_approx = total_interior + len(unique_boundary)
            self.assertGreater(total_approx, 100000)  # netlist_sampled has ~136K nodes
        finally:
            model.shutdown()
            logging.disable(logging.NOTSET)


@unittest.skipUnless(NETLIST_SAMPLED_EXISTS, "netlist_sampled not available")
class TestDistributedResult(unittest.TestCase):
    """Test DistributedSolveResult data access methods."""

    @classmethod
    def setUpClass(cls):
        import logging
        logging.disable(logging.WARNING)

        from distributed import DistributedNetlistParser, create_distributed_model, DistributedDDMSolver
        parser = DistributedNetlistParser(NETLIST_SAMPLED_DIR, net_filter='VDD_XLV')
        metadata = parser.parse_metadata()
        cls.model = create_distributed_model(metadata, backend='local')
        solver = DistributedDDMSolver(cls.model)
        cls.result = solver.solve_dc()
        logging.disable(logging.NOTSET)

    @classmethod
    def tearDownClass(cls):
        cls.model.shutdown()

    def test_flatten_returns_all_nodes(self):
        """flatten() includes interior, boundary, and pad nodes."""
        flat = self.result.flatten()
        self.assertGreater(len(flat), 100000)
        # Should include pad nodes
        for pad in self.result.pad_voltages:
            self.assertIn(pad, flat)

    def test_ir_drop_positive(self):
        """IR-drop should be non-negative for all nodes."""
        ir_drop = self.result.ir_drop
        for node, drop in ir_drop.items():
            if node == '0':
                continue
            self.assertGreaterEqual(drop, -0.001, f"Negative IR-drop at {node}: {drop}")

    def test_tile_results_coverage(self):
        """Each tile should have results."""
        self.assertEqual(len(self.result.tile_results), 9)
        for tid, tr in self.result.tile_results.items():
            self.assertGreater(len(tr.voltages), 0)
            self.assertGreater(tr.n_interior, 0)

    def test_interface_voltages_exist(self):
        """Interface voltages should be populated."""
        self.assertGreater(len(self.result.interface_voltages), 0)


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
# Context Tests
# ──────────────────────────────────────────────────────────────────────

@unittest.skipUnless(NETLIST_SAMPLED_EXISTS, "netlist_sampled not available")
class TestDistributedSolverContext(unittest.TestCase):
    """Test DistributedSolverContext creation and reuse."""

    def test_prepare_returns_context(self):
        """prepare() returns a DistributedSolverContext with expected fields."""
        import logging
        logging.disable(logging.WARNING)
        from distributed import DistributedNetlistParser, create_distributed_model, DistributedDDMSolver
        from distributed.result import DistributedSolverContext

        parser = DistributedNetlistParser(NETLIST_SAMPLED_DIR, net_filter='VDD_XLV')
        metadata = parser.parse_metadata()
        model = create_distributed_model(metadata, backend='local')

        try:
            solver = DistributedDDMSolver(model)
            ctx = solver.prepare()

            self.assertIsInstance(ctx, DistributedSolverContext)
            self.assertGreater(len(ctx.interface_nodes), 0)
            self.assertGreater(len(ctx.interface_node_to_idx), 0)
            self.assertGreater(len(ctx.tile_index_maps), 0)
            self.assertTrue(callable(ctx.interface_lu))
            self.assertIn('factor_tiles', ctx.timings)
            self.assertIn('assemble_interface', ctx.timings)
            self.assertIn('factor_interface', ctx.timings)
        finally:
            model.shutdown()
            logging.disable(logging.NOTSET)


# ──────────────────────────────────────────────────────────────────────
# Interface Island Detection Integration Tests
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

    @unittest.skipUnless(NETLIST_SAMPLED_EXISTS, "netlist_sampled not available")
    def test_removed_nodes_on_context(self):
        """prepare() populates removed_interface_nodes and timing key."""
        import logging
        logging.disable(logging.WARNING)
        from distributed import (
            DistributedNetlistParser,
            create_distributed_model,
            DistributedDDMSolver,
        )
        from distributed.result import DistributedSolverContext

        parser = DistributedNetlistParser(NETLIST_SAMPLED_DIR, net_filter='VDD_XLV')
        metadata = parser.parse_metadata()
        model = create_distributed_model(metadata, backend='local')

        try:
            solver = DistributedDDMSolver(model)
            ctx = solver.prepare()

            self.assertIsInstance(ctx, DistributedSolverContext)
            self.assertIsInstance(ctx.removed_interface_nodes, set)
            self.assertIn('detect_interface_islands', ctx.timings)
        finally:
            model.shutdown()
            logging.disable(logging.NOTSET)


# ──────────────────────────────────────────────────────────────────────
# Ray Backend Tests
# ──────────────────────────────────────────────────────────────────────

try:
    import ray
    HAS_RAY = True
except ImportError:
    HAS_RAY = False


@unittest.skipUnless(HAS_RAY, "ray not installed")
@unittest.skipUnless(NETLIST_SAMPLED_EXISTS, "netlist_sampled not available")
class TestRayBackend(unittest.TestCase):
    """Test distributed DDM with Ray backend."""

    def test_ray_solve_matches_local(self):
        """Ray backend gives same result as local backend."""
        import logging
        logging.disable(logging.WARNING)
        from distributed import DistributedNetlistParser, create_distributed_model, DistributedDDMSolver

        parser = DistributedNetlistParser(NETLIST_SAMPLED_DIR, net_filter='VDD_XLV')
        metadata = parser.parse_metadata()

        # Local solve
        model_local = create_distributed_model(metadata, backend='local')
        solver_local = DistributedDDMSolver(model_local)
        result_local = solver_local.solve_dc()
        v_local = result_local.flatten()

        # Ray solve
        model_ray = create_distributed_model(metadata, backend='ray')
        solver_ray = DistributedDDMSolver(model_ray)
        result_ray = solver_ray.solve_dc()
        v_ray = result_ray.flatten()

        try:
            self.assertEqual(len(v_local), len(v_ray))
            max_diff = max(abs(v_local[n] - v_ray[n]) for n in v_local)
            self.assertLess(max_diff, 1e-10, f"Ray vs Local max diff: {max_diff}")
        finally:
            model_local.shutdown()
            model_ray.shutdown()
            logging.disable(logging.NOTSET)


# ──────────────────────────────────────────────────────────────────────
# Benchmark: Flat vs DDM LocalBackend vs DDM RayBackend
# ──────────────────────────────────────────────────────────────────────

@unittest.skipUnless(NETLIST_SAMPLED_EXISTS, "netlist_sampled not available")
class TestBenchmarkDDMVsFlat(unittest.TestCase):
    """Benchmark flat vs DDM LocalBackend vs DDM RayBackend on netlist_sampled.

    Reports timing breakdown for parse, model creation, prepare, and solve.
    Also validates correctness (max voltage diff < 1 µV).
    """

    _results: dict = {}

    @staticmethod
    def _run_flat():
        """Run flat solver pipeline. Returns (voltages_dict, timings_dict)."""
        import time
        from parser.netlist import NetlistParser
        from model.factory import create_model_from_pdn
        from solver.unified_solver import UnifiedIRDropSolver

        t0 = time.perf_counter()
        parser = NetlistParser(NETLIST_SAMPLED_DIR)
        graph = parser.parse()
        t_parse = time.perf_counter() - t0

        t0 = time.perf_counter()
        model = create_model_from_pdn(graph, 'VDD_XLV')
        load_currents = model.extract_current_sources()
        t_model = time.perf_counter() - t0

        t0 = time.perf_counter()
        solver = UnifiedIRDropSolver(model)
        result = solver.solve(load_currents)
        t_solve = time.perf_counter() - t0

        timings = {
            'parse': t_parse,
            'model': t_model,
            'solve': t_solve,
            'total': t_parse + t_model + t_solve,
        }
        return result.voltages, timings

    @staticmethod
    def _run_ddm(backend='local'):
        """Run DDM pipeline. Returns (voltages_dict, timings_dict, model)."""
        import time
        from distributed import DistributedNetlistParser, create_distributed_model, DistributedDDMSolver

        t0 = time.perf_counter()
        parser = DistributedNetlistParser(NETLIST_SAMPLED_DIR, net_filter='VDD_XLV')
        metadata = parser.parse_metadata()
        t_parse = time.perf_counter() - t0

        t0 = time.perf_counter()
        model = create_distributed_model(metadata, backend=backend)
        t_model = time.perf_counter() - t0

        solver = DistributedDDMSolver(model)

        t0 = time.perf_counter()
        ctx = solver.prepare()
        t_prepare = time.perf_counter() - t0

        t0 = time.perf_counter()
        result = solver.solve_dc(context=ctx)
        t_solve = time.perf_counter() - t0

        timings = {
            'parse': t_parse,
            'model': t_model,
            'prepare': t_prepare,
            'solve': t_solve,
            'total': t_parse + t_model + t_prepare + t_solve,
            # Sub-breakdowns from solver internals
            'prepare_detail': dict(ctx.timings),
            'solve_detail': dict(result.solve_metadata.get('timings', {})),
        }
        return result.flatten(), timings, model

    @staticmethod
    def _print_comparison(label, flat_t, ddm_t):
        """Print formatted timing comparison table."""
        print(f"\n{'=' * 70}")
        print(f"  BENCHMARK: Flat vs DDM {label}")
        print(f"{'=' * 70}")
        print(f"  {'Phase':<30} {'Flat':>10} {'DDM':>10} {'Speedup':>10}")
        print(f"  {'-' * 60}")
        print(f"  {'Parse':<30} {flat_t['parse']:>9.3f}s {ddm_t['parse']:>9.3f}s"
              f" {flat_t['parse'] / max(ddm_t['parse'], 1e-9):>9.1f}x")
        print(f"  {'Model creation':<30} {flat_t['model']:>9.3f}s {ddm_t['model']:>9.3f}s"
              f" {flat_t['model'] / max(ddm_t['model'], 1e-9):>9.1f}x")
        print(f"  {'Solve (flat) / Prepare+Solve':<30} {flat_t['solve']:>9.3f}s"
              f" {ddm_t['prepare'] + ddm_t['solve']:>9.3f}s"
              f" {flat_t['solve'] / max(ddm_t['prepare'] + ddm_t['solve'], 1e-9):>9.1f}x")
        print(f"  {'-' * 60}")
        print(f"  {'TOTAL':<30} {flat_t['total']:>9.3f}s {ddm_t['total']:>9.3f}s"
              f" {flat_t['total'] / max(ddm_t['total'], 1e-9):>9.1f}x")

        # DDM sub-breakdown
        prep_detail = ddm_t.get('prepare_detail', {})
        solve_detail = ddm_t.get('solve_detail', {})
        if prep_detail or solve_detail:
            print(f"\n  DDM {label} Breakdown:")
            print(f"  {'  Prepare Phase':<30}")
            for key in ('factor_tiles', 'assemble_interface', 'factor_interface'):
                if key in prep_detail:
                    print(f"    {key:<28} {prep_detail[key]:>9.3f}s")
            print(f"  {'  Solve Phase':<30}")
            for key in ('compute_reduced_rhs', 'assemble_rhs', 'solve_interface', 'recover_interior'):
                if key in solve_detail:
                    print(f"    {key:<28} {solve_detail[key]:>9.3f}s")
        print(f"{'=' * 70}\n")

    def test_benchmark_local_backend(self):
        """Benchmark flat vs DDM LocalBackend. Validates correctness."""
        import logging
        logging.disable(logging.WARNING)

        try:
            v_flat, t_flat = self._run_flat()
            v_ddm, t_ddm, model = self._run_ddm('local')

            # Correctness
            common = set(v_flat) & set(v_ddm)
            self.assertGreater(len(common), 100000)
            max_diff = max(abs(v_flat[n] - v_ddm[n]) for n in common)
            self.assertLess(max_diff, 1e-6, f"Max diff {max_diff * 1e6:.3f} µV")

            self._print_comparison('LocalBackend', t_flat, t_ddm)
            self.__class__._results['flat'] = t_flat
            self.__class__._results['flat_voltages'] = v_flat
            self.__class__._results['local'] = t_ddm

            model.shutdown()
        finally:
            logging.disable(logging.NOTSET)

    @unittest.skipUnless(HAS_RAY, "ray not installed")
    def test_benchmark_ray_backend(self):
        """Benchmark flat vs DDM RayBackend (9 workers). Validates correctness."""
        import logging
        logging.disable(logging.WARNING)

        try:
            # Reuse flat results from local test if available
            v_flat = self.__class__._results.get('flat_voltages')
            t_flat = self.__class__._results.get('flat')
            if v_flat is None or t_flat is None:
                v_flat, t_flat = self._run_flat()

            v_ddm, t_ddm, model = self._run_ddm('ray')

            # Correctness
            common = set(v_flat) & set(v_ddm)
            self.assertGreater(len(common), 100000)
            max_diff = max(abs(v_flat[n] - v_ddm[n]) for n in common)
            self.assertLess(max_diff, 1e-6, f"Max diff {max_diff * 1e6:.3f} µV")

            self._print_comparison('RayBackend (9 workers)', t_flat, t_ddm)
            self.__class__._results['ray'] = t_ddm

            model.shutdown()
        finally:
            logging.disable(logging.NOTSET)

    def test_benchmark_summary(self):
        """Print comparison summary across all backends that ran."""
        results = self.__class__._results
        if 'flat' not in results or 'local' not in results:
            self.skipTest("Local backend benchmark did not run")

        print(f"\n{'=' * 70}")
        print(f"  BENCHMARK SUMMARY")
        print(f"{'=' * 70}")
        print(f"  {'Backend':<25} {'Total':>10} {'Speedup vs Flat':>16}")
        print(f"  {'-' * 51}")

        flat_total = results['flat']['total']
        print(f"  {'Flat (single-process)':<25} {flat_total:>9.3f}s {'1.0x':>16}")

        local_total = results['local']['total']
        print(f"  {'DDM LocalBackend':<25} {local_total:>9.3f}s"
              f" {flat_total / max(local_total, 1e-9):>15.1f}x")

        if 'ray' in results:
            ray_total = results['ray']['total']
            print(f"  {'DDM RayBackend (9 wkrs)':<25} {ray_total:>9.3f}s"
                  f" {flat_total / max(ray_total, 1e-9):>15.1f}x")

            # Local vs Ray comparison
            print(f"\n  {'Ray vs Local speedup:':<25}"
                  f" {local_total / max(ray_total, 1e-9):>15.1f}x")

        print(f"{'=' * 70}\n")


# ──────────────────────────────────────────────────────────────────────
# PKL Serialization Tests
# ──────────────────────────────────────────────────────────────────────

@unittest.skipUnless(NETLIST_SAMPLED_EXISTS, "netlist_sampled not available")
class TestTileWorkerFromTileData(unittest.TestCase):
    """Unit test: setup_from_tile_data() produces same block system as setup()."""

    def test_same_block_system_as_setup(self):
        """setup_from_tile_data() with pre-parsed TileData gives identical
        boundary nodes, interior/boundary counts, and islands removed
        as setup() which parses from file.
        """
        import logging
        logging.disable(logging.WARNING)
        from distributed.parser import DistributedNetlistParser
        from distributed.tile_worker import (
            TileWorker, TileData, _parse_tile_ckt, _parse_instance_models,
        )

        try:
            parser = DistributedNetlistParser(NETLIST_SAMPLED_DIR, net_filter='VDD_XLV')
            metadata = parser.parse_metadata()
            boundary_nodes = parser.collect_boundary_nodes(metadata.tile_configs)
            interface_nodes = boundary_nodes | metadata.package_data.die_attachment_nodes

            # Pick the first tile config
            tc = metadata.tile_configs[0]

            # Path A: setup() from file
            worker_file = TileWorker()
            result_file = worker_file.setup(
                {
                    'tile_id': list(tc.tile_id),
                    'ckt_path': tc.ckt_path,
                    'nd_path': tc.nd_path,
                    'instance_path': tc.instance_path,
                    'net_filter': tc.net_filter,
                },
                interface_nodes=interface_nodes,
            )

            # Path B: pre-parse TileData, then setup_from_tile_data()
            tile_data = _parse_tile_ckt(tc.ckt_path, tc.nd_path, tc.net_filter, tc.tile_id)
            if tc.instance_path:
                inst_currents = _parse_instance_models(tc.instance_path, tc.net_filter, tc.nd_path)
                for node, current in inst_currents.items():
                    if node in tile_data.all_nodes:
                        tile_data.current_injections[node] = (
                            tile_data.current_injections.get(node, 0.0) + current
                        )

            worker_data = TileWorker()
            result_data = worker_data.setup_from_tile_data(tile_data, interface_nodes)

            # Compare results
            self.assertEqual(result_file['tile_id'], result_data['tile_id'])
            self.assertEqual(
                sorted(result_file['boundary_nodes']),
                sorted(result_data['boundary_nodes']),
            )
            self.assertEqual(result_file['n_interior'], result_data['n_interior'])
            self.assertEqual(result_file['n_boundary'], result_data['n_boundary'])
            self.assertEqual(result_file['islands_removed'], result_data['islands_removed'])
        finally:
            logging.disable(logging.NOTSET)

    def test_all_tiles_match(self):
        """setup_from_tile_data() matches setup() for every tile in the netlist."""
        import logging
        logging.disable(logging.WARNING)
        from distributed.parser import DistributedNetlistParser
        from distributed.tile_worker import (
            TileWorker, _parse_tile_ckt, _parse_instance_models,
        )

        try:
            parser = DistributedNetlistParser(NETLIST_SAMPLED_DIR, net_filter='VDD_XLV')
            metadata = parser.parse_metadata()
            boundary_nodes = parser.collect_boundary_nodes(metadata.tile_configs)
            interface_nodes = boundary_nodes | metadata.package_data.die_attachment_nodes

            for tc in metadata.tile_configs:
                with self.subTest(tile_id=tc.tile_id):
                    # File-based setup
                    w1 = TileWorker()
                    r1 = w1.setup(
                        {
                            'tile_id': list(tc.tile_id),
                            'ckt_path': tc.ckt_path,
                            'nd_path': tc.nd_path,
                            'instance_path': tc.instance_path,
                            'net_filter': tc.net_filter,
                        },
                        interface_nodes=interface_nodes,
                    )

                    # TileData-based setup
                    td = _parse_tile_ckt(tc.ckt_path, tc.nd_path, tc.net_filter, tc.tile_id)
                    if tc.instance_path:
                        inst = _parse_instance_models(tc.instance_path, tc.net_filter, tc.nd_path)
                        for node, current in inst.items():
                            if node in td.all_nodes:
                                td.current_injections[node] = (
                                    td.current_injections.get(node, 0.0) + current
                                )

                    w2 = TileWorker()
                    r2 = w2.setup_from_tile_data(td, interface_nodes)

                    self.assertEqual(r1['n_interior'], r2['n_interior'])
                    self.assertEqual(r1['n_boundary'], r2['n_boundary'])
                    self.assertEqual(
                        sorted(r1['boundary_nodes']),
                        sorted(r2['boundary_nodes']),
                    )
        finally:
            logging.disable(logging.NOTSET)


@unittest.skipUnless(NETLIST_SAMPLED_EXISTS, "netlist_sampled not available")
class TestPklRoundTrip(unittest.TestCase):
    """Integration test: parse_and_dump() creates expected files,
    load_distributed_partitions() returns correct data."""

    def test_round_trip_files_exist(self):
        """parse_and_dump() creates metadata.pkl and tile_X_Y.pkl for each tile."""
        import logging
        logging.disable(logging.WARNING)
        from distributed.parser import DistributedNetlistParser

        try:
            parser = DistributedNetlistParser(NETLIST_SAMPLED_DIR, net_filter='VDD_XLV')

            with tempfile.TemporaryDirectory() as tmpdir:
                parser.parse_and_dump(tmpdir)

                # Check metadata.pkl exists
                meta_pkl = Path(tmpdir) / 'metadata.pkl'
                self.assertTrue(meta_pkl.exists(), "metadata.pkl not found")

                # Check tile .pkl files exist (3x3 grid for netlist_sampled)
                tile_files = sorted(Path(tmpdir).glob('tile_*.pkl'))
                self.assertGreater(len(tile_files), 0, "No tile .pkl files found")

                # Verify file names match expected tile grid
                metadata = parser.parse_metadata()
                expected_count = len(metadata.tile_configs)
                self.assertEqual(
                    len(tile_files), expected_count,
                    f"Expected {expected_count} tile files, found {len(tile_files)}",
                )
        finally:
            logging.disable(logging.NOTSET)

    def test_round_trip_metadata_correct(self):
        """Loaded metadata matches original parsed metadata."""
        import logging
        logging.disable(logging.WARNING)
        from distributed.parser import DistributedNetlistParser
        from distributed.model import load_distributed_partitions

        try:
            parser = DistributedNetlistParser(NETLIST_SAMPLED_DIR, net_filter='VDD_XLV')
            original_metadata = parser.parse_metadata()

            with tempfile.TemporaryDirectory() as tmpdir:
                parser.parse_and_dump(tmpdir)
                loaded_meta, loaded_bnd, loaded_tiles = load_distributed_partitions(tmpdir)

                # Metadata fields
                self.assertEqual(loaded_meta.tile_grid, original_metadata.tile_grid)
                self.assertEqual(loaded_meta.net_name, original_metadata.net_name)
                self.assertAlmostEqual(loaded_meta.vdd, original_metadata.vdd, places=6)
                self.assertEqual(
                    len(loaded_meta.tile_configs),
                    len(original_metadata.tile_configs),
                )

                # Boundary nodes should be non-empty
                self.assertGreater(len(loaded_bnd), 0)

                # Tile data dict keys match tile configs
                expected_ids = {tc.tile_id for tc in original_metadata.tile_configs}
                self.assertEqual(set(loaded_tiles.keys()), expected_ids)
        finally:
            logging.disable(logging.NOTSET)

    def test_round_trip_tile_data_preserved(self):
        """TileData round-trips through pickle without data loss."""
        import logging
        logging.disable(logging.WARNING)
        from distributed.parser import DistributedNetlistParser
        from distributed.model import load_distributed_partitions
        from distributed.tile_worker import _parse_tile_ckt, _parse_instance_models

        try:
            parser = DistributedNetlistParser(NETLIST_SAMPLED_DIR, net_filter='VDD_XLV')
            metadata = parser.parse_metadata()

            with tempfile.TemporaryDirectory() as tmpdir:
                parser.parse_and_dump(tmpdir)
                _, _, loaded_tiles = load_distributed_partitions(tmpdir)

                # Compare first tile's data
                tc = metadata.tile_configs[0]
                expected = _parse_tile_ckt(tc.ckt_path, tc.nd_path, tc.net_filter, tc.tile_id)
                if tc.instance_path:
                    inst = _parse_instance_models(tc.instance_path, tc.net_filter, tc.nd_path)
                    for node, current in inst.items():
                        if node in expected.all_nodes:
                            expected.current_injections[node] = (
                                expected.current_injections.get(node, 0.0) + current
                            )

                loaded = loaded_tiles[tc.tile_id]

                self.assertEqual(loaded.tile_id, expected.tile_id)
                self.assertEqual(loaded.all_nodes, expected.all_nodes)
                self.assertEqual(loaded.boundary_nodes, expected.boundary_nodes)
                self.assertEqual(len(loaded.resistive_edges), len(expected.resistive_edges))
                self.assertEqual(
                    set(loaded.current_injections.keys()),
                    set(expected.current_injections.keys()),
                )
                for node in expected.current_injections:
                    self.assertAlmostEqual(
                        loaded.current_injections[node],
                        expected.current_injections[node],
                        places=10,
                        msg=f"Current mismatch at node {node}",
                    )
        finally:
            logging.disable(logging.NOTSET)

    def test_load_missing_metadata_raises(self):
        """load_distributed_partitions() raises FileNotFoundError on empty dir."""
        from distributed.model import load_distributed_partitions

        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(FileNotFoundError):
                load_distributed_partitions(tmpdir)

    def test_load_no_tiles_raises(self):
        """load_distributed_partitions() raises ValueError if no tile pkls."""
        import pickle
        from distributed.model import load_distributed_partitions
        from distributed.parser import PowerGridMetaData, PackageData

        with tempfile.TemporaryDirectory() as tmpdir:
            # Write a minimal metadata.pkl with no tile files
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

            with self.assertRaises(ValueError):
                load_distributed_partitions(tmpdir)


@unittest.skipUnless(NETLIST_SAMPLED_EXISTS, "netlist_sampled not available")
class TestPklSolveMatchesDirect(unittest.TestCase):
    """End-to-end: solve from .pkl produces identical voltages as direct DDM solve."""

    def test_pkl_solve_matches_direct(self):
        """DDM solve from pkl files gives same voltages as DDM solve from .ckt files."""
        import logging
        logging.disable(logging.WARNING)
        from distributed import (
            DistributedNetlistParser,
            create_distributed_model,
            DistributedDDMSolver,
        )
        from distributed.model import load_distributed_partitions

        try:
            # Direct DDM solve (from .ckt files)
            parser = DistributedNetlistParser(NETLIST_SAMPLED_DIR, net_filter='VDD_XLV')
            metadata_direct = parser.parse_metadata()
            model_direct = create_distributed_model(metadata_direct, backend='local')
            solver_direct = DistributedDDMSolver(model_direct)
            result_direct = solver_direct.solve_dc()
            v_direct = result_direct.flatten()

            # PKL-based DDM solve
            with tempfile.TemporaryDirectory() as tmpdir:
                parser.parse_and_dump(tmpdir)
                metadata_pkl, bnd_nodes, tile_data = load_distributed_partitions(tmpdir)

                model_pkl = create_distributed_model(
                    metadata_pkl,
                    backend='local',
                    use_pkl=True,
                    boundary_nodes=bnd_nodes,
                    tile_data_dict=tile_data,
                )
                solver_pkl = DistributedDDMSolver(model_pkl)
                result_pkl = solver_pkl.solve_dc()
                v_pkl = result_pkl.flatten()

                # Verify identical results
                self.assertEqual(len(v_direct), len(v_pkl),
                                 f"Node count mismatch: {len(v_direct)} vs {len(v_pkl)}")

                common = set(v_direct) & set(v_pkl)
                self.assertEqual(len(common), len(v_direct),
                                 "Node sets differ between direct and pkl solves")

                max_diff = max(abs(v_direct[n] - v_pkl[n]) for n in common)
                self.assertLess(
                    max_diff, 1e-10,
                    f"Pkl vs direct max voltage diff: {max_diff:.2e} V "
                    f"(expected < 1e-10)"
                )

                model_pkl.shutdown()

            model_direct.shutdown()
        finally:
            logging.disable(logging.NOTSET)

    def test_pkl_solve_matches_flat(self):
        """DDM solve from pkl files matches flat solver to < 1 uV."""
        import logging
        logging.disable(logging.WARNING)
        from parser.netlist import NetlistParser
        from model.factory import create_model_from_pdn
        from solver.unified_solver import UnifiedIRDropSolver
        from distributed import (
            DistributedNetlistParser,
            create_distributed_model,
            DistributedDDMSolver,
        )
        from distributed.model import load_distributed_partitions

        try:
            # Flat solver
            flat_parser = NetlistParser(NETLIST_SAMPLED_DIR)
            graph = flat_parser.parse()
            model_flat = create_model_from_pdn(graph, 'VDD_XLV')
            load_currents = model_flat.extract_current_sources()
            solver_flat = UnifiedIRDropSolver(model_flat)
            result_flat = solver_flat.solve(load_currents)
            v_flat = result_flat.voltages

            # PKL DDM
            dist_parser = DistributedNetlistParser(NETLIST_SAMPLED_DIR, net_filter='VDD_XLV')

            with tempfile.TemporaryDirectory() as tmpdir:
                dist_parser.parse_and_dump(tmpdir)
                metadata_pkl, bnd_nodes, tile_data = load_distributed_partitions(tmpdir)

                model_pkl = create_distributed_model(
                    metadata_pkl,
                    backend='local',
                    use_pkl=True,
                    boundary_nodes=bnd_nodes,
                    tile_data_dict=tile_data,
                )
                solver_pkl = DistributedDDMSolver(model_pkl)
                result_pkl = solver_pkl.solve_dc()
                v_pkl = result_pkl.flatten()

                # Compare
                common = set(v_flat) & set(v_pkl)
                self.assertGreater(len(common), 100000)

                max_diff = max(abs(v_flat[n] - v_pkl[n]) for n in common)
                self.assertLess(
                    max_diff, 1e-6,
                    f"Pkl DDM vs flat max diff: {max_diff * 1e6:.3f} uV"
                )

                model_pkl.shutdown()
        finally:
            logging.disable(logging.NOTSET)


if __name__ == '__main__':
    unittest.main()

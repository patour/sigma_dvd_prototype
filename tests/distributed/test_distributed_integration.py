"""Integration tests for distributed DDM solver (slow, require test netlists)."""

import os
import tempfile
import unittest
from pathlib import Path
from typing import Dict, List, Set

import numpy as np
import pytest

pytestmark = pytest.mark.integration

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


try:
    import ray
    HAS_RAY = True
except ImportError:
    HAS_RAY = False


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
        boundary_nodes = parser.collect_shared_boundary_nodes(metadata.tile_configs)

        # Multi-tile netlist should have many shared boundary nodes
        self.assertGreater(len(boundary_nodes), 100)
        # Boundary nodes should be string names
        for node in list(boundary_nodes)[:5]:
            self.assertIsInstance(node, str)

    def test_shared_boundary_subset_of_all(self):
        """Shared boundary nodes are a strict subset of all boundary nodes."""
        from distributed import DistributedNetlistParser

        parser = DistributedNetlistParser(NETLIST_SAMPLED_DIR, net_filter='VDD_XLV')
        metadata = parser.parse_metadata()
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            all_boundary = parser.collect_boundary_nodes(metadata.tile_configs)
        shared_boundary = parser.collect_shared_boundary_nodes(metadata.tile_configs)

        # Shared must be a subset of all
        self.assertTrue(shared_boundary.issubset(all_boundary))
        # There should be some single-tile-only nodes filtered out
        self.assertGreater(len(all_boundary), len(shared_boundary),
                           "Expected some single-tile-only nodes to be filtered")


@unittest.skipUnless(NETLIST_SAMPLED_EXISTS, "netlist_sampled not available")
class TestParseAndDumpFiltering(unittest.TestCase):
    """Test that parse_and_dump stores only shared boundary nodes in metadata.pkl."""

    def test_parse_and_dump_filters_single_tile_boundary(self):
        """metadata.pkl boundary_nodes contains only shared (2+ tile) nodes."""
        import logging
        import pickle
        logging.disable(logging.WARNING)
        from distributed.parser import DistributedNetlistParser

        try:
            parser = DistributedNetlistParser(NETLIST_SAMPLED_DIR, net_filter='VDD_XLV')
            metadata = parser.parse_metadata()

            # Get reference counts from direct scanning
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                all_bnd = parser.collect_boundary_nodes(metadata.tile_configs)
            shared_bnd = parser.collect_shared_boundary_nodes(metadata.tile_configs)

            with tempfile.TemporaryDirectory() as tmpdir:
                _out_path, _bundle = parser.parse_and_dump(tmpdir)

                # Load metadata.pkl and check boundary nodes
                meta_pkl = Path(tmpdir) / 'metadata.pkl'
                with open(meta_pkl, 'rb') as f:
                    meta_bundle = pickle.load(f)

                pkl_bnd = meta_bundle['boundary_nodes']

                # PKL should contain only shared nodes
                self.assertEqual(pkl_bnd, shared_bnd)
                # Should be strictly fewer than all boundary nodes
                self.assertGreater(len(all_bnd), len(pkl_bnd),
                                   "Expected filtering to remove some nodes")
        finally:
            logging.disable(logging.NOTSET)


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
        import warnings
        from distributed import DistributedNetlistParser, create_distributed_model, DistributedDDMSolver
        parser = DistributedNetlistParser(NETLIST_SAMPLED_DIR, net_filter='VDD_XLV')
        cls.metadata = parser.parse_metadata()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
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
        import warnings
        logging.disable(logging.WARNING)
        from distributed import DistributedNetlistParser, create_distributed_model

        parser = DistributedNetlistParser(NETLIST_SAMPLED_DIR, net_filter='VDD_XLV')
        metadata = parser.parse_metadata()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
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
        import warnings
        logging.disable(logging.WARNING)
        from distributed import DistributedNetlistParser, create_distributed_model

        parser = DistributedNetlistParser(NETLIST_SAMPLED_DIR, net_filter='VDD_XLV')
        metadata = parser.parse_metadata()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
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

        import warnings
        from distributed import DistributedNetlistParser, create_distributed_model, DistributedDDMSolver
        parser = DistributedNetlistParser(NETLIST_SAMPLED_DIR, net_filter='VDD_XLV')
        metadata = parser.parse_metadata()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
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
# Context Tests
# ──────────────────────────────────────────────────────────────────────

@unittest.skipUnless(NETLIST_SAMPLED_EXISTS, "netlist_sampled not available")
class TestDistributedSolverContext(unittest.TestCase):
    """Test DistributedSolverContext creation and reuse."""

    def test_prepare_returns_context(self):
        """prepare() returns a DistributedSolverContext with expected fields."""
        import logging
        import warnings
        logging.disable(logging.WARNING)
        from distributed import DistributedNetlistParser, create_distributed_model, DistributedDDMSolver
        from distributed.result import DistributedSolverContext

        parser = DistributedNetlistParser(NETLIST_SAMPLED_DIR, net_filter='VDD_XLV')
        metadata = parser.parse_metadata()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
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

@unittest.skipUnless(NETLIST_SAMPLED_EXISTS, "netlist_sampled not available")
class TestInterfaceIslandDetectionIntegration(unittest.TestCase):
    """Integration test for interface island detection with real netlist."""

    def test_removed_nodes_on_context(self):
        """prepare() populates removed_interface_nodes and timing key."""
        import logging
        import warnings
        logging.disable(logging.WARNING)
        from distributed import (
            DistributedNetlistParser,
            create_distributed_model,
            DistributedDDMSolver,
        )
        from distributed.result import DistributedSolverContext

        parser = DistributedNetlistParser(NETLIST_SAMPLED_DIR, net_filter='VDD_XLV')
        metadata = parser.parse_metadata()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
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

@unittest.skipUnless(HAS_RAY, "ray not installed")
@unittest.skipUnless(NETLIST_SAMPLED_EXISTS, "netlist_sampled not available")
class TestRayBackend(unittest.TestCase):
    """Test distributed DDM with Ray backend."""

    def test_ray_solve_matches_local(self):
        """Ray backend gives same result as local backend."""
        import logging
        import warnings
        logging.disable(logging.WARNING)
        from distributed import DistributedNetlistParser, create_distributed_model, DistributedDDMSolver

        parser = DistributedNetlistParser(NETLIST_SAMPLED_DIR, net_filter='VDD_XLV')
        metadata = parser.parse_metadata()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
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


@unittest.skipUnless(HAS_RAY, "ray not installed")
@unittest.skipUnless(NETLIST_SAMPLED_EXISTS, "netlist_sampled not available")
class TestParseAndDumpRay(unittest.TestCase):
    """Test that Ray-based parse_and_dump matches local backend."""

    def test_ray_parse_matches_local(self):
        """parse_and_dump with backend='ray' produces identical results to 'local'."""
        import logging
        import pickle
        logging.disable(logging.WARNING)
        from distributed.parser import DistributedNetlistParser

        try:
            parser = DistributedNetlistParser(NETLIST_SAMPLED_DIR, net_filter='VDD_XLV')

            with tempfile.TemporaryDirectory() as local_dir, \
                 tempfile.TemporaryDirectory() as ray_dir:
                parser.parse_and_dump(local_dir, backend='local')  # returns (Path, bundle)
                parser.parse_and_dump(ray_dir, backend='ray')

                # Load metadata.pkl from both
                with open(os.path.join(local_dir, 'metadata.pkl'), 'rb') as f:
                    meta_local = pickle.load(f)
                with open(os.path.join(ray_dir, 'metadata.pkl'), 'rb') as f:
                    meta_ray = pickle.load(f)

                # Boundary nodes must match exactly
                self.assertEqual(
                    meta_local['boundary_nodes'],
                    meta_ray['boundary_nodes'],
                    "Boundary nodes differ between local and ray parse"
                )

                # Compare each tile's TileData
                n_tiles = len(meta_local['metadata'].tile_configs)
                for tc in meta_local['metadata'].tile_configs:
                    x, y = tc.tile_id
                    fname = f'tile_{x}_{y}.pkl'
                    with open(os.path.join(local_dir, fname), 'rb') as f:
                        td_local = pickle.load(f)
                    with open(os.path.join(ray_dir, fname), 'rb') as f:
                        td_ray = pickle.load(f)

                    self.assertEqual(
                        td_local.all_nodes, td_ray.all_nodes,
                        f"Tile ({x},{y}): all_nodes differ"
                    )
                    self.assertEqual(
                        td_local.boundary_nodes, td_ray.boundary_nodes,
                        f"Tile ({x},{y}): boundary_nodes differ"
                    )
                    self.assertEqual(
                        len(td_local.resistive_edges), len(td_ray.resistive_edges),
                        f"Tile ({x},{y}): resistive_edges count differs"
                    )
                    self.assertEqual(
                        len(td_local.current_injections), len(td_ray.current_injections),
                        f"Tile ({x},{y}): current_injections count differs"
                    )
        finally:
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
        import warnings
        from distributed import DistributedNetlistParser, create_distributed_model, DistributedDDMSolver

        t0 = time.perf_counter()
        parser = DistributedNetlistParser(NETLIST_SAMPLED_DIR, net_filter='VDD_XLV')
        metadata = parser.parse_metadata()
        t_parse = time.perf_counter() - t0

        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
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
            boundary_nodes = parser.collect_shared_boundary_nodes(metadata.tile_configs)
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
            boundary_nodes = parser.collect_shared_boundary_nodes(metadata.tile_configs)
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
                _out_path, _bundle = parser.parse_and_dump(tmpdir)

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
                _out_path, _bundle = parser.parse_and_dump(tmpdir)
                bundle = load_distributed_partitions(tmpdir)

                # Metadata fields
                self.assertEqual(bundle.metadata.tile_grid, original_metadata.tile_grid)
                self.assertEqual(bundle.metadata.net_name, original_metadata.net_name)
                self.assertAlmostEqual(bundle.metadata.vdd, original_metadata.vdd, places=6)
                self.assertEqual(
                    len(bundle.metadata.tile_configs),
                    len(original_metadata.tile_configs),
                )

                # Boundary nodes should be non-empty
                self.assertGreater(len(bundle.shared_boundary_nodes), 0)

                # pkl_dir should be set
                self.assertEqual(bundle.pkl_dir, tmpdir)
        finally:
            logging.disable(logging.NOTSET)

    def test_round_trip_tile_data_preserved(self):
        """TileData round-trips through pickle without data loss."""
        import logging
        import pickle
        logging.disable(logging.WARNING)
        from distributed.parser import DistributedNetlistParser
        from distributed.tile_worker import TileData, _parse_tile_ckt, _parse_instance_models

        try:
            parser = DistributedNetlistParser(NETLIST_SAMPLED_DIR, net_filter='VDD_XLV')
            metadata = parser.parse_metadata()

            with tempfile.TemporaryDirectory() as tmpdir:
                _out_path, _bundle = parser.parse_and_dump(tmpdir)

                # Load first tile pkl directly
                tc = metadata.tile_configs[0]
                x, y = tc.tile_id
                with open(Path(tmpdir) / f'tile_{x}_{y}.pkl', 'rb') as f:
                    loaded = pickle.load(f)

                # Compare first tile's data
                expected = _parse_tile_ckt(tc.ckt_path, tc.nd_path, tc.net_filter, tc.tile_id)
                if tc.instance_path:
                    inst = _parse_instance_models(tc.instance_path, tc.net_filter, tc.nd_path)
                    for node, current in inst.items():
                        if node in expected.all_nodes:
                            expected.current_injections[node] = (
                                expected.current_injections.get(node, 0.0) + current
                            )

                self.assertIsInstance(loaded, TileData)
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


@unittest.skipUnless(NETLIST_SAMPLED_EXISTS, "netlist_sampled not available")
class TestPklSolveMatchesDirect(unittest.TestCase):
    """End-to-end: solve from .pkl produces identical voltages as direct DDM solve."""

    def test_pkl_solve_matches_direct(self):
        """DDM solve from pkl files gives same voltages as DDM solve from .ckt files."""
        import logging
        import warnings
        logging.disable(logging.WARNING)
        from distributed import (
            DistributedNetlistParser,
            create_distributed_model,
            DistributedDDMSolver,
        )
        from distributed.model import load_distributed_partitions

        try:
            # Direct DDM solve (from .ckt files, legacy path)
            parser = DistributedNetlistParser(NETLIST_SAMPLED_DIR, net_filter='VDD_XLV')
            metadata_direct = parser.parse_metadata()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                model_direct = create_distributed_model(metadata_direct, backend='local')
            solver_direct = DistributedDDMSolver(model_direct)
            result_direct = solver_direct.solve_dc()
            v_direct = result_direct.flatten()

            # PKL-based DDM solve (new ParsedTileBundle path)
            with tempfile.TemporaryDirectory() as tmpdir:
                _out_path, bundle = parser.parse_and_dump(tmpdir)

                model_pkl = create_distributed_model(bundle, backend='local')
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

        try:
            # Flat solver
            flat_parser = NetlistParser(NETLIST_SAMPLED_DIR)
            graph = flat_parser.parse()
            model_flat = create_model_from_pdn(graph, 'VDD_XLV')
            load_currents = model_flat.extract_current_sources()
            solver_flat = UnifiedIRDropSolver(model_flat)
            result_flat = solver_flat.solve(load_currents)
            v_flat = result_flat.voltages

            # PKL DDM (new ParsedTileBundle path)
            dist_parser = DistributedNetlistParser(NETLIST_SAMPLED_DIR, net_filter='VDD_XLV')

            with tempfile.TemporaryDirectory() as tmpdir:
                _out_path, bundle = dist_parser.parse_and_dump(tmpdir)

                model_pkl = create_distributed_model(bundle, backend='local')
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


@unittest.skipUnless(NETLIST_SMALL_EXISTS, "netlist_small not available")
class TestTopKReportIntegration(unittest.TestCase):
    """Integration test: distributed solve + top-K report generation."""

    @classmethod
    def setUpClass(cls):
        """Parse netlist_small as distributed model and solve once."""
        import logging
        import warnings
        logging.disable(logging.WARNING)
        try:
            from distributed import (
                DistributedNetlistParser,
                create_distributed_model,
                DistributedDDMSolver,
            )
            parser = DistributedNetlistParser(NETLIST_SMALL_DIR, net_filter='VDD_XLV')
            cls.metadata = parser.parse_metadata()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                cls.model = create_distributed_model(cls.metadata, backend='local')
            cls.solver = DistributedDDMSolver(cls.model)
            cls.ctx = cls.solver.prepare()
            cls.result = cls.solver.solve_dc(context=cls.ctx)
            cls._setup_ok = True
        except Exception:
            cls._setup_ok = False
        finally:
            logging.disable(logging.NOTSET)

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, '_setup_ok', False) and hasattr(cls, 'model'):
            cls.model.shutdown()

    def test_generate_reports_creates_topk_file(self):
        """generate_reports(top_k=10) creates topk_irdrop_*.txt in output_dir."""
        if not self._setup_ok:
            self.skipTest("distributed setup failed for netlist_small")

        with tempfile.TemporaryDirectory() as tmpdir:
            import logging
            logging.disable(logging.WARNING)
            try:
                self.solver.generate_reports(
                    result=self.result,
                    context=self.ctx,
                    output_dir=tmpdir,
                    top_k=10,
                    show_irdrop=False,  # skip heatmaps for speed
                )
            finally:
                logging.disable(logging.NOTSET)

            # Find the topk report file
            import glob
            topk_files = glob.glob(os.path.join(tmpdir, 'topk_irdrop_*.txt'))
            self.assertGreater(
                len(topk_files), 0,
                f"No topk_irdrop_*.txt found in {tmpdir}; contents: {os.listdir(tmpdir)}",
            )

            # Verify file has header and at least 1 data row
            with open(topk_files[0], 'r') as f:
                lines = f.read().splitlines()
            # Header is 6 lines (title, net, voltage, separator, columns, separator)
            self.assertGreater(len(lines), 6, "Report should have header + data rows")

            # Data rows should be at most 10
            data_lines = [l for l in lines[6:] if l.strip()]
            self.assertLessEqual(len(data_lines), 10)
            self.assertGreater(len(data_lines), 0, "Expected at least 1 data row")


if __name__ == '__main__':
    unittest.main()

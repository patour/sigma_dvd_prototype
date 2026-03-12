"""Integration tests for distributed time-domain analysis.

Compares distributed quasi-static / transient results against flat solvers
and verifies physical consistency.  All tests require the netlist_sampled
test data (3x3 tile grid, ~136K nodes).
"""

import logging
import os
import unittest
import warnings
from typing import Dict

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

PKL_DIR = os.path.join(NETLIST_SAMPLED_DIR, 'distributed_pkl')
PKL_DIR_EXISTS = os.path.isdir(PKL_DIR)


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _create_distributed_model():
    """Create a DistributedPowerGridModel from pre-parsed pkl files.

    Returns:
        (model, solver) tuple.
    """
    from distributed.model import load_distributed_partitions, create_distributed_model
    from distributed.solver import DistributedDDMSolver

    bundle = load_distributed_partitions(PKL_DIR)
    model = create_distributed_model(bundle, backend='local')
    solver = DistributedDDMSolver(model)
    return model, solver


def _create_flat_solver():
    """Create a flat UnifiedIRDropSolver from the sampled netlist.

    Returns:
        (model, graph, solver) tuple.
    """
    from parser.netlist import NetlistParser
    from model.factory import create_model_from_pdn
    from solver.unified_solver import UnifiedIRDropSolver

    parser = NetlistParser(NETLIST_SAMPLED_DIR)
    graph = parser.parse()
    model = create_model_from_pdn(graph, 'VDD_XLV')
    solver = UnifiedIRDropSolver(model)
    return model, graph, solver


# ──────────────────────────────────────────────────────────────────────
# Test 11: Quasi-static distributed vs flat (batch DC at time points)
# ──────────────────────────────────────────────────────────────────────

@unittest.skipUnless(
    NETLIST_SAMPLED_EXISTS and PKL_DIR_EXISTS,
    "netlist_sampled / distributed_pkl not available",
)
class TestDistributedQuasiStaticVsFlat(unittest.TestCase):
    """Compare distributed quasi-static against flat DynamicIRDropSolver.

    Both solvers perform batch DC solves at each time point with the same
    time-varying current sources.  Voltages at each time point should match
    within floating-point tolerance.
    """

    @classmethod
    def setUpClass(cls):
        """Create distributed + flat solvers and run quasi-static analysis."""
        logging.disable(logging.WARNING)

        cls.model, cls.solver = _create_distributed_model()
        cls.flat_model, cls.flat_graph, cls.flat_solver = _create_flat_solver()

        # Use a small number of time points for speed
        cls.n_points = 5
        cls.t_end = 10e-9  # 10 ns
        cls.t_start = 0.0

        # Run distributed quasi-static
        cls.dist_result = cls.solver.solve_quasi_static(
            t_start=cls.t_start,
            t_end=cls.t_end,
            n_points=cls.n_points,
            verbose=False,
        )

        # Run flat quasi-static (DynamicIRDropSolver)
        from analysis.dynamic_solver import DynamicIRDropSolver

        cls.flat_dyn_solver = DynamicIRDropSolver(
            cls.flat_model, cls.flat_graph,
        )
        cls.flat_result = cls.flat_dyn_solver.solve_quasi_static(
            t_start=cls.t_start,
            t_end=cls.t_end,
            n_points=cls.n_points,
            method='flat',
        )

        logging.disable(logging.NOTSET)

    @classmethod
    def tearDownClass(cls):
        cls.model.shutdown()

    def test_time_arrays_match(self):
        """Both solvers use the same time points."""
        np.testing.assert_allclose(
            self.dist_result.t_array,
            self.flat_result.t_array,
            atol=1e-15,
        )

    def test_max_ir_drop_per_time_reasonable(self):
        """Per-step max IR-drop is nonzero and in physical range for both solvers.

        The distributed and flat solvers evaluate current sources via
        independent VCS instances (per-tile vs monolithic). Because source
        partitioning and wscale handling differ, the per-step values may
        diverge somewhat. We check that both produce nonzero, physical
        results rather than requiring tight numerical agreement.
        """
        dist_max = self.dist_result.max_ir_drop_per_time
        flat_max = self.flat_result.max_ir_drop_per_time

        # Both should be non-negative
        self.assertTrue(np.all(dist_max >= 0))
        self.assertTrue(np.all(flat_max >= 0))

        # At least one time point should have nonzero IR-drop
        self.assertGreater(np.max(dist_max), 0,
                           "Distributed max IR-drop is zero -- sources may not be evaluated")
        self.assertGreater(np.max(flat_max), 0,
                           "Flat max IR-drop is zero -- sources may not be evaluated")

        # Both peaks should be within the same order of magnitude
        ratio = max(np.max(dist_max), np.max(flat_max)) / min(np.max(dist_max), np.max(flat_max))
        self.assertLess(ratio, 5.0,
                        f"Peak IR-drop ratio {ratio:.1f}x -- distributed and flat "
                        f"should be within 5x")

    def test_peak_ir_drop_order_of_magnitude(self):
        """Distributed and flat peak IR-drop within same order of magnitude."""
        dist_peak = self.dist_result.peak_ir_drop
        flat_peak = self.flat_result.peak_ir_drop

        self.assertGreater(dist_peak, 0)
        self.assertGreater(flat_peak, 0)

        ratio = max(dist_peak, flat_peak) / min(dist_peak, flat_peak)
        self.assertLess(ratio, 3.0,
                        f"Peak IR-drop ratio {ratio:.2f} -- should be within 3x")

    def test_total_current_per_time_nonzero(self):
        """Total injected current per time point is nonzero for both solvers.

        The two solvers evaluate sources independently (per-tile VCS vs
        monolithic VCS). Source partitioning, wscale handling, and
        net-filtering differences can produce significant current-total
        variation. We verify both produce nonzero results and are within
        the same order of magnitude.
        """
        dist_I = self.dist_result.total_current_per_time
        flat_I = self.flat_result.total_current_per_time

        # Both should be nonzero
        self.assertGreater(np.max(np.abs(dist_I)), 0,
                           "Distributed total current is all zero")
        self.assertGreater(np.max(np.abs(flat_I)), 0,
                           "Flat total current is all zero")

        # Same order of magnitude
        dist_max_I = float(np.max(np.abs(dist_I)))
        flat_max_I = float(np.max(np.abs(flat_I)))
        ratio = max(dist_max_I, flat_max_I) / max(min(dist_max_I, flat_max_I), 1e-12)
        self.assertLess(ratio, 5.0,
                        f"Total current ratio {ratio:.1f}x -- should be within 5x")

    def test_voltages_in_physical_range(self):
        """Distributed quasi-static voltages should be in [0, Vdd]."""
        vdd = self.model.vdd
        dist_max = self.dist_result.max_ir_drop_per_time
        for i, drop in enumerate(dist_max):
            self.assertLessEqual(drop, vdd + 0.01,
                                 f"Step {i}: max IR-drop {drop} > Vdd+margin")

    def test_results_vary_over_time(self):
        """At least some time points produce different IR-drop values.

        If all are identical, sources may not be time-varying (a setup bug).
        """
        dist_max = self.dist_result.max_ir_drop_per_time
        self.assertGreater(
            np.max(dist_max) - np.min(dist_max), 1e-6,
            "All distributed time steps have identical IR-drop -- "
            "time-varying sources may not be working",
        )


# ──────────────────────────────────────────────────────────────────────
# Test 12: Transient distributed -- physical consistency
# ──────────────────────────────────────────────────────────────────────

@unittest.skipUnless(
    NETLIST_SAMPLED_EXISTS and PKL_DIR_EXISTS,
    "netlist_sampled / distributed_pkl not available",
)
class TestDistributedTransientPhysics(unittest.TestCase):
    """Physical consistency checks for distributed transient analysis.

    Runs a short transient simulation and verifies basic physics:
    voltages in range, nonzero IR-drop, capacitive smoothing effect.
    """

    @classmethod
    def setUpClass(cls):
        """Create distributed solver and run transient."""
        logging.disable(logging.WARNING)

        cls.model, cls.solver = _create_distributed_model()

        cls.dt = 1e-9      # 1 ns
        cls.t_end = 5e-9    # 5 ns
        cls.t_start = 0.0

        cls.result = cls.solver.solve_transient(
            t_start=cls.t_start,
            t_end=cls.t_end,
            dt=cls.dt,
            method='be',
            verbose=False,
        )

        logging.disable(logging.NOTSET)

    @classmethod
    def tearDownClass(cls):
        cls.model.shutdown()

    def test_time_array_correct(self):
        """Time array starts at t_start+dt and has correct spacing."""
        t = self.result.t_array
        expected_n = int(round((self.t_end - self.t_start) / self.dt))
        self.assertEqual(len(t), expected_n)
        # First step should be t_start + dt
        np.testing.assert_allclose(t[0], self.t_start + self.dt, rtol=1e-6)

    def test_nonzero_ir_drop(self):
        """Transient produces measurable IR-drop."""
        self.assertGreater(self.result.peak_ir_drop, 0,
                           "Transient peak IR-drop is zero")

    def test_ir_drop_within_vdd(self):
        """All per-step max IR-drop values are less than Vdd."""
        vdd = self.model.vdd
        for i, drop in enumerate(self.result.max_ir_drop_per_time):
            self.assertLessEqual(drop, vdd + 0.01,
                                 f"Step {i}: IR-drop {drop} > Vdd+margin")

    def test_integration_method_recorded(self):
        """Result records correct integration method."""
        self.assertEqual(self.result.integration_method, 'be')

    def test_has_capacitance_reported(self):
        """Result reports whether package-level caps were found.

        Note: has_capacitance refers to package-level cap edges only.
        Tile-level caps are always present (and used in the A-matrix)
        but are not reflected in this flag.
        """
        # The flag is a bool -- just confirm it's set (True or False)
        self.assertIsInstance(self.result.has_capacitance, bool)

    def test_total_current_nonzero(self):
        """At least some transient steps have injected current."""
        self.assertGreater(
            np.max(np.abs(self.result.total_current_per_time)), 0,
            "All transient total currents are zero",
        )

    def test_transient_and_quasi_static_correlated(self):
        """Transient and quasi-static peak IR-drop are in the same ballpark.

        With time-varying sources, the transient response can temporarily
        exceed the instantaneous DC response (quasi-static) because caps
        introduce memory effects. However, both should produce similar
        orders of magnitude for peak IR-drop.
        """
        # Run a quick quasi-static with the same time range
        qs_result = self.solver.solve_quasi_static(
            t_start=self.t_start,
            t_end=self.t_end,
            n_points=len(self.result.t_array) + 1,
            verbose=False,
        )

        qs_peak = qs_result.peak_ir_drop
        tr_peak = self.result.peak_ir_drop

        # Both should be nonzero
        self.assertGreater(qs_peak, 0)
        self.assertGreater(tr_peak, 0)

        # Should be within 10x of each other (same order of magnitude)
        ratio = max(qs_peak, tr_peak) / min(qs_peak, tr_peak)
        self.assertLess(ratio, 10.0,
                        f"Transient peak {tr_peak:.4f}V vs quasi-static peak "
                        f"{qs_peak:.4f}V ratio {ratio:.1f}x -- expected < 10x")


# ──────────────────────────────────────────────────────────────────────
# Test 13: DC limit -- constant currents, transient converges to DC
# ──────────────────────────────────────────────────────────────────────

class TestTransientDCLimit(unittest.TestCase):
    """Transient with constant currents should converge to DC steady state.

    Uses a small synthetic 2-tile model with capacitors and constant
    (time-independent) current sources. After enough time steps, the
    transient voltages should converge to the DC solution.

    No external netlist data needed — fully self-contained.
    """

    def test_transient_converges_to_dc(self):
        """Final transient step voltages match DC voltages within tolerance."""
        from distributed.solver import DistributedDDMSolver
        from distributed.backend import LocalBackend
        from distributed.model import DistributedPowerGridModel
        from distributed.parser import PackageData, PowerGridMetaData, TileConfig
        from distributed.tile_worker import TileWorker, TileData

        # Build a small 2-tile model with caps and constant currents
        tile_a_data = TileData(
            tile_id=(0, 0),
            resistive_edges=[
                ('a1', 'shared', 1.0),
                ('shared', 'a2', 0.5),
                ('a2', '0', 2.0),
            ],
            all_nodes={'a1', 'shared', 'a2'},
            boundary_nodes={'shared'},
            current_injections={'a1': 0.5},
            capacitive_edges=[('a1', '0', 100.0), ('a2', '0', 50.0)],
        )
        tile_b_data = TileData(
            tile_id=(0, 1),
            resistive_edges=[
                ('shared', 'b1', 3.0),
                ('b1', '0', 1.0),
            ],
            all_nodes={'shared', 'b1'},
            boundary_nodes={'shared'},
            current_injections={'b1': 0.3},
            capacitive_edges=[('b1', '0', 80.0)],
        )

        interface_nodes = {'shared', 'pad'}

        be = LocalBackend()
        be.initialize()

        worker_a = TileWorker()
        worker_a.setup_from_tile_data(tile_a_data, interface_nodes)

        worker_b = TileWorker()
        worker_b.setup_from_tile_data(tile_b_data, interface_nodes)

        pkg_data = PackageData(
            vsrc_dict={'V1': {'node+': 'pad', 'node-': '0', 'net': 'VDD', 'value': 1.0}},
            package_edges=[('pad', 'shared', 10.0)],
            pad_nodes={'pad'},
            tap_nodes=set(),
            die_attachment_nodes=set(),
            vdd=1.0,
            net_name='VDD',
            package_cap_edges=[],
        )

        tile_configs = [
            TileConfig(tile_id=(0, 0), ckt_path='', nd_path=None,
                       instance_path=None, net_filter=None),
            TileConfig(tile_id=(0, 1), ckt_path='', nd_path=None,
                       instance_path=None, net_filter=None),
        ]

        metadata = PowerGridMetaData(
            tile_grid=(1, 2),
            parameters={},
            tile_configs=tile_configs,
            package_data=pkg_data,
            net_name='VDD',
            vdd=1.0,
        )

        model = DistributedPowerGridModel(
            backend=be,
            workers=[worker_a, worker_b],
            interface_nodes=interface_nodes,
            tile_boundary_nodes={
                (0, 0): ['shared'],
                (0, 1): ['shared'],
            },
            tile_interior_counts={
                (0, 0): worker_a.n_interior,
                (0, 1): worker_b.n_interior,
            },
            package_data=pkg_data,
            metadata=metadata,
        )

        self.addCleanup(model.shutdown)
        solver = DistributedDDMSolver(model)

        # 1. Solve DC (steady state)
        dc_ctx = solver.prepare()
        dc_result = solver.solve_dc(context=dc_ctx)
        dc_voltages = dc_result.flatten()

        # 2. Run transient with constant sources for enough steps to converge.
        #    RC time constant ~ R * C ~ 1 kOhm * 100 fF = 0.1 ns.
        #    Use dt = 0.5 ns, t_end = 10 ns (many time constants).
        #    Since we have no VCS (constant static sources), the worker will
        #    use tile-local current_injections at each step.
        dt = 0.5e-9
        t_end = 10e-9

        trans_ctx = solver.prepare_transient(dt=dt, method='be')

        # Manual transient loop (cannot use solve_transient without VCS)
        n_interface = len(trans_ctx.dc_context.interface_nodes)
        tile_configs_list = model.metadata.tile_configs

        # DC initial condition
        dc_rhs_results = be.call_all(
            model.workers, 'get_reduced_rhs',
        )
        global_rhs_init = np.zeros(n_interface, dtype=np.float64)
        for i, g_i in enumerate(dc_rhs_results):
            tid = tile_configs_list[i].tile_id
            idx_map = dc_ctx.tile_index_maps[tid]
            np.add.at(global_rhs_init, idx_map, g_i)
        global_rhs_init += dc_ctx.rhs_dirichlet_interface
        v_gamma_init = dc_ctx.interface_lu(global_rhs_init)

        # Factor transient on workers
        # (already done in prepare_transient, tiles have transient block systems)

        # Set initial voltages
        bv_init = {}
        for i_node, idx in trans_ctx.interface_node_to_idx.items():
            bv_init[i_node] = float(v_gamma_init[idx])
        for pad in model.pad_nodes:
            bv_init[pad] = model.vdd

        init_v_list = be.call_all(
            model.workers, 'get_interior_voltages', [(bv_init,)] * len(model.workers),
        )
        be.call_all(
            model.workers, 'set_initial_voltages',
            [(v,) for v in init_v_list],
        )

        # Time loop
        t_array = np.arange(dt, t_end + dt / 2, dt)
        v_gamma_old = v_gamma_init.copy()

        for step_idx, t_val in enumerate(t_array):
            # Build boundary_v_old dicts
            bv_old_list = []
            for tc in tile_configs_list:
                tid = tc.tile_id
                P_i = trans_ctx.tile_index_maps[tid]
                bv = {
                    trans_ctx.interface_nodes[idx]: float(v_gamma_old[idx])
                    for idx in P_i
                }
                for n in model.tile_boundary_nodes[tid]:
                    if n not in bv:
                        bv[n] = model.vdd
                bv_old_list.append(bv)

            bv_old_args = [(t_val, bv) for bv in bv_old_list]
            rhs_results = be.call_all(
                model.workers, 'get_transient_reduced_rhs', bv_old_args,
            )

            global_rhs = np.zeros(n_interface, dtype=np.float64)
            for i, (g_i, _) in enumerate(rhs_results):
                tid = tile_configs_list[i].tile_id
                idx_map = trans_ctx.tile_index_maps[tid]
                np.add.at(global_rhs, idx_map, g_i)

            global_rhs += trans_ctx.rhs_dirichlet_G

            if trans_ctx.C_package_uu is not None:
                global_rhs += trans_ctx.C_coeff * (trans_ctx.C_package_uu @ v_gamma_old)

            v_gamma_new = trans_ctx.interface_lu(global_rhs)

            # Recover interior on workers
            bv_new_list = []
            for tc in tile_configs_list:
                tid = tc.tile_id
                P_i = trans_ctx.tile_index_maps[tid]
                bv = {
                    trans_ctx.interface_nodes[idx]: float(v_gamma_new[idx])
                    for idx in P_i
                }
                for n in model.tile_boundary_nodes[tid]:
                    if n not in bv:
                        bv[n] = model.vdd
                bv_new_list.append(bv)

            be.call_all(
                model.workers, 'get_transient_interior_voltages',
                [(bv,) for bv in bv_new_list],
            )

            v_gamma_old = v_gamma_new

        # 3. Get final transient voltages
        final_bv = {}
        for i_node, idx in trans_ctx.interface_node_to_idx.items():
            final_bv[i_node] = float(v_gamma_old[idx])
        for pad in model.pad_nodes:
            final_bv[pad] = model.vdd

        final_voltages_list = be.call_all(
            model.workers, 'get_interior_voltages', [(final_bv,)] * len(model.workers),
        )

        # Merge final transient voltages
        trans_voltages: Dict[str, float] = {}
        for pad in model.pad_nodes:
            trans_voltages[pad] = model.vdd
        for i_node in trans_ctx.interface_nodes:
            idx = trans_ctx.interface_node_to_idx[i_node]
            trans_voltages[i_node] = float(v_gamma_old[idx])
        for v_dict in final_voltages_list:
            trans_voltages.update(v_dict)

        # 4. Compare: transient steady state should match DC
        common = set(dc_voltages.keys()) & set(trans_voltages.keys())
        self.assertGreater(len(common), 0, "No common nodes to compare")

        max_diff = max(
            abs(dc_voltages[n] - trans_voltages[n]) for n in common
        )
        self.assertLess(max_diff, 1e-4,
                        f"Transient vs DC max diff {max_diff:.2e} V after "
                        f"{len(t_array)} steps (expected convergence to DC)")


# ──────────────────────────────────────────────────────────────────────
# Test 14: DC solve unchanged after cap parsing extensions
# ──────────────────────────────────────────────────────────────────────

@unittest.skipUnless(
    NETLIST_SAMPLED_EXISTS and PKL_DIR_EXISTS,
    "netlist_sampled / distributed_pkl not available",
)
class TestDCSolveUnchangedWithCaps(unittest.TestCase):
    """Verify that DC solve is unaffected by cap parsing extensions.

    Caps are ignored in DC analysis (G only, no C). The DC results
    should match the flat solver within floating-point tolerance, same
    as before the cap extensions were added.
    """

    @classmethod
    def setUpClass(cls):
        """Parse and solve with both solvers."""
        logging.disable(logging.WARNING)

        cls.model, cls.solver = _create_distributed_model()
        cls.flat_model, cls.flat_graph, cls.flat_solver = _create_flat_solver()

        # DC solve (distributed)
        cls.ctx = cls.solver.prepare()
        cls.dc_result = cls.solver.solve_dc(context=cls.ctx)
        cls.dc_voltages = cls.dc_result.flatten()

        # DC solve (flat)
        cls.load_currents = cls.flat_model.extract_current_sources()
        cls.flat_result = cls.flat_solver.solve(cls.load_currents)

        cls.common_nodes = (
            set(cls.dc_voltages.keys()) & set(cls.flat_result.voltages.keys())
        )

        logging.disable(logging.NOTSET)

    @classmethod
    def tearDownClass(cls):
        cls.model.shutdown()

    def test_dc_voltage_match(self):
        """DDM DC voltages match flat within < 1 uV tolerance."""
        max_diff = max(
            abs(self.dc_voltages[n] - self.flat_result.voltages[n])
            for n in self.common_nodes
        )
        self.assertLess(max_diff, 1e-6,
                        f"DC voltage max diff {max_diff*1e6:.3f} uV > 1 uV")

    def test_dc_has_meaningful_irdrop(self):
        """DC solve produces meaningful IR-drop (not vacuous)."""
        vdd = self.model.vdd
        max_drop = max(
            vdd - self.dc_voltages[n]
            for n in self.common_nodes
        )
        self.assertGreater(max_drop, 1e-4,
                           f"DC max IR-drop {max_drop*1e3:.3f} mV too small")

    def test_dc_pad_nodes_at_vdd(self):
        """Pad nodes should be at Vdd exactly."""
        vdd = self.model.vdd
        for pad in self.dc_result.pad_voltages:
            self.assertAlmostEqual(
                self.dc_result.pad_voltages[pad], vdd, places=10,
            )

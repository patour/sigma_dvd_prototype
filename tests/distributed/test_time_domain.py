"""Unit tests for distributed time-domain analysis building blocks.

Tests capacitor parsing (C lines, instance caps), grounded capacitance
diagonals, package interface matrices, result types, peak tracking, and
island detection with capacitive-only paths.
"""

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

pytestmark = pytest.mark.unit


# ──────────────────────────────────────────────────────────────────────
# 1. C line parsing tests
# ──────────────────────────────────────────────────────────────────────


class TestCapacitorParsing:
    """Test C-line handling in _parse_tile_ckt."""

    def test_parse_tile_ckt_with_caps(self, tmp_path):
        """R, I, and C lines parsed; capacitive_edges populated correctly."""
        from distributed.tile_parsing import _parse_tile_ckt

        ckt = tmp_path / "tile_0_0.ckt"
        ckt.write_text(
            "* test tile with caps\n"
            "R1 a *b 1000\n"             # 1 kOhm resistor
            "Ifoo *b 0 1e-3\n"           # 1 mA current source
            "Cfoo *b 0 1e-15\n"          # Named cap, 1 fF
            "c a 0 2e-15\n"              # Unnamed cap, 2 fF
            "C_big *c 0 10e-15\n"        # Named cap on boundary c, 10 fF
        )

        td = _parse_tile_ckt(str(ckt), nd_path=None, net_filter=None,
                             tile_id=(0, 0))

        # Capacitive edges: 3 C lines
        assert len(td.capacitive_edges) == 3
        # Values in fF: 1, 2, 10
        cap_values = sorted(c for _, _, c in td.capacitive_edges)
        np.testing.assert_allclose(cap_values, [1.0, 2.0, 10.0], rtol=1e-6)

        # Ground excluded from all_nodes
        assert '0' not in td.all_nodes
        # Interior + boundary nodes all present
        assert 'a' in td.all_nodes
        assert 'b' in td.all_nodes
        assert 'c' in td.all_nodes

        # *-prefixed nodes on C lines register as boundary
        assert 'b' in td.boundary_nodes
        assert 'c' in td.boundary_nodes

    def test_parse_tile_ckt_caps_net_filter(self, tmp_path):
        """C lines not matching net filter are excluded."""
        from distributed.tile_parsing import _parse_tile_ckt

        # .nd file: a -> vdd, x -> vss
        nd = tmp_path / "tile_0_0.nd"
        nd.write_text(
            "a 0 0 M1 (0,0) vdd\n"
            "x 0 0 M1 (0,0) vss\n"
        )

        ckt = tmp_path / "tile_0_0.ckt"
        ckt.write_text(
            "R1 a 0 1000\n"
            "Cvdd a 0 5e-15\n"    # On vdd net
            "Cvss x 0 3e-15\n"    # On vss net
        )

        td = _parse_tile_ckt(str(ckt), nd_path=str(nd), net_filter='vdd',
                             tile_id=(0, 0))

        # Only the vdd-net cap should be included
        assert len(td.capacitive_edges) == 1
        node, gnd, c_fF = td.capacitive_edges[0]
        assert node == 'a'
        np.testing.assert_allclose(c_fF, 5.0, rtol=1e-6)

    def test_parse_tile_ckt_no_caps(self, tmp_path):
        """A .ckt with only R lines has empty capacitive_edges."""
        from distributed.tile_parsing import _parse_tile_ckt

        ckt = tmp_path / "tile_0_0.ckt"
        ckt.write_text(
            "R1 a b 1000\n"
            "R2 b 0 500\n"
        )

        td = _parse_tile_ckt(str(ckt), nd_path=None, net_filter=None,
                             tile_id=(0, 0))

        assert td.capacitive_edges == []


# ──────────────────────────────────────────────────────────────────────
# 2. Instance model capacitor parsing tests
# ──────────────────────────────────────────────────────────────────────


class TestInstanceCapacitorParsing:
    """Test _iter_instance_capacitors and _parse_instance_capacitors."""

    def test_iter_instance_capacitors_grounded(self, tmp_path):
        """Grounded cap lines yield (node, c_fF)."""
        from distributed.tile_parsing import _iter_instance_capacitors

        inst = tmp_path / "instanceModels.sp"
        inst.write_text(
            "cr0 node1 0 8.59e-15\n"
        )

        results = list(_iter_instance_capacitors(str(inst), net_filter=None))
        assert len(results) == 1
        node, c_fF = results[0]
        assert node == 'node1'
        np.testing.assert_allclose(c_fF, 8.59, rtol=1e-6)

    def test_iter_instance_capacitors_skip_coupling(self, tmp_path):
        """Coupling caps (both non-ground) are skipped."""
        from distributed.tile_parsing import _iter_instance_capacitors

        inst = tmp_path / "instanceModels.sp"
        inst.write_text(
            "cfoo n1 n2 1e-15\n"
        )

        results = list(_iter_instance_capacitors(str(inst), net_filter=None))
        assert results == []

    def test_parse_instance_capacitors_accumulates(self, tmp_path):
        """Multiple caps on same node are summed."""
        from distributed.tile_parsing import _parse_instance_capacitors

        inst = tmp_path / "instanceModels.sp"
        inst.write_text(
            "c1 node1 0 3e-15\n"
            "c2 node1 0 7e-15\n"
            "c3 node2 0 5e-15\n"
        )

        result = _parse_instance_capacitors(str(inst), net_filter=None)

        # Convert to dict for easy checking: {node: total}
        cap_dict = {node: total for node, gnd, total in result}
        assert len(cap_dict) == 2
        np.testing.assert_allclose(cap_dict['node1'], 10.0, rtol=1e-6)
        np.testing.assert_allclose(cap_dict['node2'], 5.0, rtol=1e-6)


# ──────────────────────────────────────────────────────────────────────
# 3. Grounded capacitance diagonal tests
# ──────────────────────────────────────────────────────────────────────


class TestGroundedCapacitanceDiags:
    """Test build_grounded_capacitance_diags."""

    def _build_diags(self, cap_edges, port_to_idx, interior_to_idx,
                     n_ports, n_interior):
        from solver.coupled_system import build_grounded_capacitance_diags
        return build_grounded_capacitance_diags(
            cap_edges, port_to_idx, interior_to_idx, n_ports, n_interior,
        )

    def test_basic_grounded_cap(self):
        """One port cap, one interior cap -- verify diagonal values and total."""
        port_to_idx = {'p1': 0}
        interior_to_idx = {'i1': 0}
        cap_edges = [
            ('p1', '0', 5.0),   # 5 fF on port
            ('i1', '0', 10.0),  # 10 fF on interior
        ]

        c_pp, c_ii, total = self._build_diags(
            cap_edges, port_to_idx, interior_to_idx, 1, 1,
        )

        np.testing.assert_allclose(c_pp, [5.0])
        np.testing.assert_allclose(c_ii, [10.0])
        np.testing.assert_allclose(total, 15.0)

    def test_both_ground_skipped(self):
        """(0, 0, value) edge is skipped."""
        cap_edges = [('0', '0', 10.0)]

        c_pp, c_ii, total = self._build_diags(
            cap_edges, {}, {}, 0, 0,
        )

        assert total == 0.0

    def test_coupling_cap_skipped(self):
        """Both terminals non-ground is not a grounded cap -- skipped."""
        port_to_idx = {'n1': 0}
        interior_to_idx = {'n2': 0}
        cap_edges = [('n1', 'n2', 10.0)]

        c_pp, c_ii, total = self._build_diags(
            cap_edges, port_to_idx, interior_to_idx, 1, 1,
        )

        np.testing.assert_allclose(c_pp, [0.0])
        np.testing.assert_allclose(c_ii, [0.0])
        assert total == 0.0

    def test_unmapped_node_skipped(self):
        """Cap on unknown node yields zero diags."""
        cap_edges = [('unknown_node', '0', 10.0)]

        c_pp, c_ii, total = self._build_diags(
            cap_edges, {'p1': 0}, {'i1': 0}, 1, 1,
        )

        np.testing.assert_allclose(c_pp, [0.0])
        np.testing.assert_allclose(c_ii, [0.0])
        assert total == 0.0

    def test_empty_cap_edges(self):
        """Empty input returns zero arrays."""
        c_pp, c_ii, total = self._build_diags(
            [], {'p1': 0}, {'i1': 0}, 1, 1,
        )

        np.testing.assert_allclose(c_pp, [0.0])
        np.testing.assert_allclose(c_ii, [0.0])
        assert total == 0.0

    def test_negative_capacitance_skipped(self):
        """Negative capacitance is skipped."""
        cap_edges = [('p1', '0', -5.0)]

        c_pp, c_ii, total = self._build_diags(
            cap_edges, {'p1': 0}, {}, 1, 0,
        )

        np.testing.assert_allclose(c_pp, [0.0])
        assert total == 0.0

    def test_multiple_caps_same_node(self):
        """Two caps on same port node accumulate."""
        port_to_idx = {'p1': 0}
        cap_edges = [
            ('p1', '0', 3.0),
            ('0', 'p1', 7.0),  # Reversed terminals, still p1 grounded
        ]

        c_pp, c_ii, total = self._build_diags(
            cap_edges, port_to_idx, {}, 1, 0,
        )

        np.testing.assert_allclose(c_pp, [10.0])
        np.testing.assert_allclose(total, 10.0)


# ──────────────────────────────────────────────────────────────────────
# 4. Package interface matrix tests
# ──────────────────────────────────────────────────────────────────────


class TestInterfacePackageMatrices:
    """Test build_interface_package_matrices."""

    def _build(self, pkg_edges, pkg_cap_edges, node_to_idx, n,
               dirichlet_nodes):
        from solver.interface_assembly import build_interface_package_matrices
        return build_interface_package_matrices(
            package_edges=pkg_edges,
            package_cap_edges=pkg_cap_edges,
            interface_node_to_idx=node_to_idx,
            n_interface=n,
            dirichlet_nodes=dirichlet_nodes,
        )

    def test_grounded_cap_diagonal_only(self):
        """Package grounded cap adds to C[i,i]."""
        node_to_idx = {'a': 0, 'b': 1}
        pkg_cap_edges = [('a', '0', 5.0)]  # 5 fF grounded

        G, C = self._build([], pkg_cap_edges, node_to_idx, 2, set())

        C_dense = C.toarray()
        np.testing.assert_allclose(C_dense[0, 0], 5.0)
        np.testing.assert_allclose(C_dense[1, 1], 0.0)
        np.testing.assert_allclose(C_dense[0, 1], 0.0)

    def test_coupling_cap_full_stamp(self):
        """Coupling cap between two unknowns: full nodal analysis stamp."""
        node_to_idx = {'a': 0, 'b': 1}
        pkg_cap_edges = [('a', 'b', 10.0)]

        G, C = self._build([], pkg_cap_edges, node_to_idx, 2, set())

        C_dense = C.toarray()
        # Diagonal: +c
        np.testing.assert_allclose(C_dense[0, 0], 10.0)
        np.testing.assert_allclose(C_dense[1, 1], 10.0)
        # Off-diagonal: -c
        np.testing.assert_allclose(C_dense[0, 1], -10.0)
        np.testing.assert_allclose(C_dense[1, 0], -10.0)

    def test_dirichlet_one_terminal(self):
        """Cap with one Dirichlet terminal contributes diagonal-only on unknown."""
        node_to_idx = {'a': 0}
        dirichlet_nodes = {'pad'}
        pkg_cap_edges = [('a', 'pad', 8.0)]

        G, C = self._build([], pkg_cap_edges, node_to_idx, 1,
                           dirichlet_nodes)

        C_dense = C.toarray()
        np.testing.assert_allclose(C_dense[0, 0], 8.0)

    def test_both_dirichlet_skipped(self):
        """Both terminals Dirichlet: no contribution."""
        node_to_idx = {'a': 0}
        dirichlet_nodes = {'pad1', 'pad2'}
        pkg_cap_edges = [('pad1', 'pad2', 100.0)]

        G, C = self._build([], pkg_cap_edges, node_to_idx, 1,
                           dirichlet_nodes)

        C_dense = C.toarray()
        np.testing.assert_allclose(C_dense[0, 0], 0.0)

    def test_symmetry(self):
        """Both G and C matrices are symmetric."""
        node_to_idx = {'a': 0, 'b': 1, 'c': 2}
        pkg_edges = [('a', 'b', 2.0), ('b', 'c', 3.0)]
        pkg_cap_edges = [('a', 'b', 5.0), ('c', '0', 7.0)]

        G, C = self._build(pkg_edges, pkg_cap_edges, node_to_idx, 3, set())

        G_dense = G.toarray()
        C_dense = C.toarray()
        np.testing.assert_allclose(G_dense, G_dense.T, atol=1e-12)
        np.testing.assert_allclose(C_dense, C_dense.T, atol=1e-12)

    def test_empty_inputs(self):
        """Empty edge lists return zero sparse matrices."""
        G, C = self._build([], [], {'a': 0}, 1, set())

        assert G.nnz == 0
        assert C.nnz == 0
        assert G.shape == (1, 1)
        assert C.shape == (1, 1)


# ──────────────────────────────────────────────────────────────────────
# 5. Result type tests
# ──────────────────────────────────────────────────────────────────────


class TestResultTypes:
    """Test distributed result and context dataclasses."""

    def test_smoothed_sources_compatible(self):
        """is_compatible matches when params are within tolerance."""
        from distributed.result import DistributedSmoothedSources

        ss = DistributedSmoothedSources(
            time_step=1e-10, t_start=0.0, t_end=100e-9,
            smoothed=True, n_tiles=4, per_tile_stats={},
        )

        # Same params: compatible
        assert ss.is_compatible(1e-10, 0.0, 100e-9) is True

        # Different time_step: not compatible
        assert ss.is_compatible(2e-10, 0.0, 100e-9) is False

        # Different t_end: not compatible
        assert ss.is_compatible(1e-10, 0.0, 200e-9) is False

    def test_transient_context_dt_property(self):
        """dt_scaled=100.0 (ps) -> dt == 100e-12 (seconds)."""
        from distributed.result import (
            DistributedTransientContext,
            DistributedSolverContext,
        )

        # Minimal dc_context stub
        dc_ctx = DistributedSolverContext(
            interface_lu=lambda x: x,
            interface_nodes=[],
            interface_node_to_idx={},
            rhs_dirichlet_interface=np.zeros(0),
            tile_index_maps={},
        )

        ctx = DistributedTransientContext(
            interface_lu=lambda x: x,
            interface_nodes=[],
            interface_node_to_idx={},
            rhs_dirichlet_interface=np.zeros(0),
            tile_index_maps={},
            dc_context=dc_ctx,
            dt_scaled=100.0,
            integration_method='be',
            has_capacitance=True,
        )

        np.testing.assert_allclose(ctx.dt, 100e-12)

    def test_transient_context_C_coeff_be(self):
        """BE: C_coeff = 1 / dt_scaled."""
        from distributed.result import (
            DistributedTransientContext,
            DistributedSolverContext,
        )

        dc_ctx = DistributedSolverContext(
            interface_lu=lambda x: x,
            interface_nodes=[],
            interface_node_to_idx={},
            rhs_dirichlet_interface=np.zeros(0),
            tile_index_maps={},
        )

        ctx = DistributedTransientContext(
            interface_lu=lambda x: x,
            interface_nodes=[],
            interface_node_to_idx={},
            rhs_dirichlet_interface=np.zeros(0),
            tile_index_maps={},
            dc_context=dc_ctx,
            dt_scaled=100.0,
            integration_method='be',
            has_capacitance=True,
        )

        np.testing.assert_allclose(ctx.C_coeff, 0.01)

    def test_transient_context_C_coeff_trap(self):
        """Trapezoidal: C_coeff = 2 / dt_scaled."""
        from distributed.result import (
            DistributedTransientContext,
            DistributedSolverContext,
        )

        dc_ctx = DistributedSolverContext(
            interface_lu=lambda x: x,
            interface_nodes=[],
            interface_node_to_idx={},
            rhs_dirichlet_interface=np.zeros(0),
            tile_index_maps={},
        )

        ctx = DistributedTransientContext(
            interface_lu=lambda x: x,
            interface_nodes=[],
            interface_node_to_idx={},
            rhs_dirichlet_interface=np.zeros(0),
            tile_index_maps={},
            dc_context=dc_ctx,
            dt_scaled=100.0,
            integration_method='trap',
            has_capacitance=True,
        )

        np.testing.assert_allclose(ctx.C_coeff, 0.02)

    def test_quasi_static_result_as_flat_dedup(self):
        """Two tiles with overlapping boundary node: keeps highest drop."""
        from distributed.result import DistributedQuasiStaticResult

        t_array = np.array([0.0, 1e-9])

        result = DistributedQuasiStaticResult(
            t_array=t_array,
            nominal_voltage=1.0,
        )

        # Manually inject pre-collected peak cache (bypass worker collection)
        result._peak_collected = True
        result._peak_cache = {
            (0, 0): {'shared': (0.05, 0.0), 'a': (0.10, 1e-9)},
            (0, 1): {'shared': (0.08, 1e-9), 'b': (0.03, 0.0)},
        }

        flat = result.as_flat()

        # 'shared' appears in both tiles; should keep higher drop (0.08)
        assert flat['shared'] == (0.08, 1e-9)
        assert flat['a'] == (0.10, 1e-9)
        assert flat['b'] == (0.03, 0.0)

    def test_quasi_static_result_dump_load_roundtrip(self, tmp_path):
        """dump() then load() preserves data."""
        from distributed.result import DistributedQuasiStaticResult

        t_array = np.linspace(0, 10e-9, 5)
        max_drops = np.array([0.01, 0.02, 0.03, 0.02, 0.01])
        total_I = np.array([1.0, 2.0, 3.0, 2.0, 1.0])

        result = DistributedQuasiStaticResult(
            t_array=t_array,
            nominal_voltage=0.9,
            net_name='VDD',
            peak_ir_drop=0.03,
            peak_ir_drop_time=float(t_array[2]),
            max_ir_drop_per_time=max_drops,
            total_current_per_time=total_I,
        )
        # Pre-collect peaks so dump doesn't try to contact workers
        result._peak_collected = True
        result._peak_cache = {(0, 0): {'n1': (0.03, float(t_array[2]))}}

        pkl = tmp_path / "result.pkl"
        result.dump(str(pkl))

        loaded = DistributedQuasiStaticResult.load(str(pkl))

        np.testing.assert_array_equal(loaded.t_array, t_array)
        assert loaded.nominal_voltage == 0.9
        assert loaded.net_name == 'VDD'
        np.testing.assert_allclose(loaded.peak_ir_drop, 0.03)
        np.testing.assert_array_equal(loaded.max_ir_drop_per_time, max_drops)
        np.testing.assert_array_equal(loaded.total_current_per_time, total_I)
        assert loaded._model is None
        # Peaks accessible from cache
        flat = loaded.as_flat()
        assert 'n1' in flat

    def test_quasi_static_result_collect_peaks_no_model(self):
        """Raises RuntimeError when model is None."""
        from distributed.result import DistributedQuasiStaticResult

        result = DistributedQuasiStaticResult(
            t_array=np.array([0.0]),
            nominal_voltage=1.0,
            _model=None,
        )

        with pytest.raises(RuntimeError, match="Peak data not available"):
            result.as_flat()

    def test_transient_result_inherits_dump_load(self, tmp_path):
        """DistributedTransientResult round-trips with correct type."""
        from distributed.result import DistributedTransientResult

        t_array = np.linspace(0, 5e-9, 3)
        result = DistributedTransientResult(
            t_array=t_array,
            nominal_voltage=0.85,
            integration_method='trap',
            has_capacitance=True,
        )
        result._peak_collected = True
        result._peak_cache = {}

        pkl = tmp_path / "transient.pkl"
        result.dump(str(pkl))

        loaded = DistributedTransientResult.load(str(pkl))

        assert isinstance(loaded, DistributedTransientResult)
        assert loaded.integration_method == 'trap'
        assert loaded.has_capacitance is True
        np.testing.assert_array_equal(loaded.t_array, t_array)
        assert loaded.nominal_voltage == 0.85


# ──────────────────────────────────────────────────────────────────────
# 6. Peak tracking tests
# ──────────────────────────────────────────────────────────────────────


def _make_worker_with_tile(edges, port_nodes, currents=None, cap_edges=None):
    """Create a TileWorker set up with a synthetic TileData (no file I/O).

    Args:
        edges: List of (u, v, conductance_mS) resistive edges.
        port_nodes: Set of port (interface) node names.
        currents: Optional dict {node: mA}.
        cap_edges: Optional list of (u, v, c_fF) capacitive edges.

    Returns:
        Configured TileWorker instance.
    """
    from distributed.tile_worker import TileWorker, TileData

    all_nodes = set()
    for u, v, _ in edges:
        if u != '0':
            all_nodes.add(u)
        if v != '0':
            all_nodes.add(v)

    td = TileData(
        tile_id=(0, 0),
        resistive_edges=list(edges),
        all_nodes=all_nodes,
        boundary_nodes=port_nodes & all_nodes,
        current_injections=dict(currents or {}),
        capacitive_edges=list(cap_edges or []),
    )

    worker = TileWorker()
    worker.setup_from_tile_data(td, interface_nodes=port_nodes)
    worker.factor_and_compute_schur()
    return worker


class TestPeakTracking:
    """Test peak tracking mixin methods."""

    def test_peak_tracking_basic(self):
        """Feed 3 steps of voltages; verify max per node."""
        worker = _make_worker_with_tile(
            edges=[('a', 'b', 2.0), ('b', '0', 1.0)],
            port_nodes={'a'},
        )

        worker.init_peak_tracking(vdd=1.0)

        # Step 1: moderate drop
        worker.update_peak_stats({'a': 0.95, 'b': 0.90}, t=0.0)
        # Step 2: worst drop
        worker.update_peak_stats({'a': 0.80, 'b': 0.70}, t=1e-9)
        # Step 3: recovery
        worker.update_peak_stats({'a': 0.90, 'b': 0.85}, t=2e-9)

        stats = worker.get_peak_stats()

        # Node 'a': worst drop = 1.0 - 0.80 = 0.20
        np.testing.assert_allclose(stats['a'][0], 0.20, atol=1e-12)
        # Node 'b': worst drop = 1.0 - 0.70 = 0.30
        np.testing.assert_allclose(stats['b'][0], 0.30, atol=1e-12)

    def test_peak_tracking_time_of_max(self):
        """Verify the timestamp of peak is correct."""
        worker = _make_worker_with_tile(
            edges=[('a', 'b', 2.0), ('b', '0', 1.0)],
            port_nodes={'a'},
        )

        worker.init_peak_tracking(vdd=1.0)

        worker.update_peak_stats({'a': 0.95}, t=0.0)
        worker.update_peak_stats({'a': 0.80}, t=5e-9)
        worker.update_peak_stats({'a': 0.90}, t=10e-9)

        stats = worker.get_peak_stats()

        # Peak at t=5e-9 where drop = 0.20
        np.testing.assert_allclose(stats['a'][1], 5e-9)

    def test_peak_tracking_vdd_validation(self):
        """init_peak_tracking(vdd=0) raises ValueError."""
        worker = _make_worker_with_tile(
            edges=[('a', 'b', 2.0), ('b', '0', 1.0)],
            port_nodes={'a'},
        )

        with pytest.raises(ValueError, match="vdd must be positive"):
            worker.init_peak_tracking(vdd=0)

    def test_tracked_waveforms(self):
        """With tracked nodes, get_tracked_waveforms records all steps."""
        worker = _make_worker_with_tile(
            edges=[('a', 'b', 2.0), ('b', '0', 1.0)],
            port_nodes={'a'},
        )

        worker.init_peak_tracking(track_nodes=['a', 'b'], vdd=1.0)

        voltages_seq = [
            {'a': 0.95, 'b': 0.90},
            {'a': 0.80, 'b': 0.70},
            {'a': 0.90, 'b': 0.85},
        ]
        for i, v in enumerate(voltages_seq):
            worker.update_peak_stats(v, t=float(i) * 1e-9)

        wf = worker.get_tracked_waveforms()

        assert 'a' in wf
        assert 'b' in wf
        assert wf['a'] == [0.95, 0.80, 0.90]
        assert wf['b'] == [0.90, 0.70, 0.85]


# ──────────────────────────────────────────────────────────────────────
# 7. Cached RHS recovery (quasi-static bug fix)
# ──────────────────────────────────────────────────────────────────────


class TestCachedRHSRecovery:
    """Unit tests for the cached interior RHS recovery path.

    Verifies that recover_and_update_peaks() uses the cached RHS from
    evaluate_and_get_reduced_rhs() rather than static DC currents.
    """

    def test_cache_set_and_consumed(self):
        """evaluate_and_get_reduced_rhs sets cache; recover consumes it."""
        from unittest.mock import MagicMock
        from analysis.vectorized_sources import VectorizedCurrentSources
        from parser.current_sources import CurrentSource

        worker = _make_worker_with_tile(
            edges=[('a', 'b', 2.0), ('b', '0', 1.0)],
            port_nodes={'a'},
            currents={'b': 0.5},
        )

        # Build a simple VCS with one DC source
        n_ports = worker._block_system.n_ports
        n_interior = worker._block_system.n_interior
        n_nodes = n_ports + n_interior
        node_to_idx = dict(worker._block_system.port_to_idx)
        for node, idx in worker._block_system.interior_to_idx.items():
            node_to_idx[node] = idx + n_ports

        src = CurrentSource(name='i1', node1='b', node2='0', dc_value=1.0)
        worker._vec_sources = VectorizedCurrentSources.from_current_sources(
            {'i1': src}, node_to_idx, n_nodes,
        )
        worker._active_sources = worker._vec_sources

        # Before evaluate: cache should be None
        assert worker._last_qs_rhs_i is None

        # Evaluate: cache should be set
        g, total, _stats = worker.evaluate_and_get_reduced_rhs(t=0.0)
        assert worker._last_qs_rhs_i is not None
        assert worker._last_qs_rhs_i.shape == (n_interior,)

        # Recover with peak tracking: cache should be consumed
        worker.init_peak_tracking(vdd=1.0)
        drop = worker.recover_and_update_peaks({'a': 0.95}, t=0.0)
        assert worker._last_qs_rhs_i is None  # consumed

        # Subsequent call falls back to static DC path (no error)
        drop2 = worker.recover_and_update_peaks({'a': 0.95}, t=1e-9)
        assert drop2 >= 0

    def test_cached_recovery_matches_evaluate(self):
        """Interior voltages from cached RHS are consistent with the
        reduced RHS sent to the coordinator."""
        from analysis.vectorized_sources import VectorizedCurrentSources
        from parser.current_sources import CurrentSource

        # 2-interior-node tile: a(port) -- b(int) -- c(int) -- 0
        worker = _make_worker_with_tile(
            edges=[('a', 'b', 1.0), ('b', 'c', 1.0), ('c', '0', 1.0)],
            port_nodes={'a'},
            currents={'b': 0.3, 'c': 0.2},
        )

        n_ports = worker._block_system.n_ports
        n_interior = worker._block_system.n_interior
        n_nodes = n_ports + n_interior
        node_to_idx = dict(worker._block_system.port_to_idx)
        for node, idx in worker._block_system.interior_to_idx.items():
            node_to_idx[node] = idx + n_ports

        # Create VCS with different currents than static (0.5 mA vs 0.3/0.2)
        src_b = CurrentSource(name='ib', node1='b', node2='0', dc_value=0.5)
        src_c = CurrentSource(name='ic', node1='c', node2='0', dc_value=0.1)
        worker._vec_sources = VectorizedCurrentSources.from_current_sources(
            {'ib': src_b, 'ic': src_c}, node_to_idx, n_nodes,
        )
        worker._active_sources = worker._vec_sources

        # Solve via evaluate + manual coordinator solve + recover
        g_vcs, _, _stats = worker.evaluate_and_get_reduced_rhs(t=0.0)

        # Also solve via static DC path for comparison
        g_dc, _dc_stats = worker.get_reduced_rhs()

        # VCS currents differ from static, so reduced RHS should differ
        assert not np.allclose(g_vcs, g_dc, atol=1e-10), \
            "VCS and DC reduced RHS should differ (different currents)"

        # Recover with cached path (VCS currents)
        worker.init_peak_tracking(vdd=1.0)
        worker.evaluate_and_get_reduced_rhs(t=0.0)  # re-populate cache
        worker.recover_and_update_peaks({'a': 0.9}, t=0.0)

        # Get the voltages from peak stats
        peaks_vcs = worker.get_peak_stats()

        # Now solve with static DC path
        worker._last_qs_rhs_i = None  # force DC fallback
        worker._peak_per_node = {}  # reset peaks
        worker.recover_and_update_peaks({'a': 0.9}, t=1e-9)
        peaks_dc = worker.get_peak_stats()

        # The two recoveries should produce different interior voltages
        # because VCS injects 0.5+0.1=0.6 mA while static injects 0.3+0.2=0.5 mA
        common = set(peaks_vcs) & set(peaks_dc)
        interior_common = {n for n in common if n != 'a'}
        assert len(interior_common) > 0
        diffs = [abs(peaks_vcs[n][0] - peaks_dc[n][0]) for n in interior_common]
        assert max(diffs) > 1e-6, \
            "Cached and DC recovery should produce different interior voltages"


# ──────────────────────────────────────────────────────────────────────
# 8. Island detection with caps
# ──────────────────────────────────────────────────────────────────────


class TestIslandDetectionWithCaps:
    """Test that cap-only paths do not prevent floating island removal."""

    def test_cap_only_path_is_floating(self, tmp_path):
        """A node connected to pads only via capacitive edges is floating.

        Graph:
            pad --[cap]-- iso --[cap]-- 0  (no resistive path)
            main_a --[R]-- main_b --[R]-- 0  (main component with resistive path)

        The 'iso' node has no resistive path to any other node, so
        the island detection in TileWorker should remove it.
        """
        from distributed.tile_worker import TileWorker

        ckt = tmp_path / "tile_0_0.ckt"
        ckt.write_text(
            "* Main component: resistive chain\n"
            "R1 main_a *main_b 1000\n"
            "R2 main_b 0 500\n"
            "* Isolated node connected only by caps\n"
            "C1 iso 0 10e-15\n"
            "C2 iso *main_b 5e-15\n"
        )

        worker = TileWorker()
        result = worker.setup(
            {
                'tile_id': [0, 0],
                'ckt_path': str(ckt),
                'nd_path': None,
                'instance_path': None,
                'net_filter': None,
            },
            interface_nodes={'main_b'},
        )

        # 'iso' has no resistive edges -> not in all_nodes from R-line parsing
        # (C lines add to all_nodes but not to resistive_edges adjacency)
        # So it should either never appear as a proper graph node or be
        # detected as floating and removed.
        assert 'iso' not in worker._block_system.port_to_idx
        assert 'iso' not in worker._block_system.interior_to_idx

        # Main component should be intact
        assert 'main_a' in worker._tile_data.all_nodes
        assert 'main_b' in worker._tile_data.all_nodes


# ──────────────────────────────────────────────────────────────────────
# 8. Transient Dirichlet RHS tests (B1 blocker fix)
# ──────────────────────────────────────────────────────────────────────


def _build_two_tile_distributed_model(package_cap_edges=None):
    """Build a minimal 2-tile DistributedPowerGridModel for unit tests.

    Topology:
        Tile A: a1 --[1mS]-- shared --[2mS]-- pad
        Tile B: shared --[3mS]-- b1 --[1mS]-- 0 (ground)

    Interface nodes: {shared}
    Pad node: {pad}  (Dirichlet at 1.0 V)
    Package edges: [(pad, shared, 10.0)]  (package resistor 10 mS)

    Args:
        package_cap_edges: Optional list of (u, v, c_fF) package cap edges.

    Returns:
        DistributedPowerGridModel ready for prepare_transient().
    """
    from distributed.backend import LocalBackend
    from distributed.model import DistributedPowerGridModel
    from distributed.parser import PackageData, PowerGridMetaData, TileConfig
    from distributed.tile_worker import TileWorker, TileData

    # Define tiles
    tile_a_data = TileData(
        tile_id=(0, 0),
        resistive_edges=[('a1', 'shared', 1.0), ('shared', 'pad', 2.0)],
        all_nodes={'a1', 'shared', 'pad'},
        boundary_nodes={'shared', 'pad'},
        current_injections={'a1': 0.5},
        capacitive_edges=[('a1', '0', 10.0)],  # 10 fF grounded
    )
    tile_b_data = TileData(
        tile_id=(0, 1),
        resistive_edges=[('shared', 'b1', 3.0), ('b1', '0', 1.0)],
        all_nodes={'shared', 'b1'},
        boundary_nodes={'shared'},
        current_injections={'b1': 0.3},
        capacitive_edges=[('b1', '0', 5.0)],  # 5 fF grounded
    )

    # Interface = boundary nodes shared across tiles + die attachment
    interface_nodes = {'shared', 'pad'}

    # Create workers via local backend
    be = LocalBackend()
    be.initialize()

    worker_a = TileWorker()
    worker_a.setup_from_tile_data(tile_a_data, interface_nodes)

    worker_b = TileWorker()
    worker_b.setup_from_tile_data(tile_b_data, interface_nodes)

    workers = [worker_a, worker_b]

    # Package data
    pkg_data = PackageData(
        vsrc_dict={'V1': {'node+': 'pad', 'node-': '0', 'net': 'VDD', 'value': 1.0}},
        package_edges=[('pad', 'shared', 10.0)],  # 10 mS package resistor
        pad_nodes={'pad'},
        tap_nodes=set(),
        die_attachment_nodes=set(),
        vdd=1.0,
        net_name='VDD',
        package_cap_edges=list(package_cap_edges or []),
    )

    # Tile configs (minimal -- ckt_path not used for unit tests)
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

    tile_boundary_nodes = {
        (0, 0): ['pad', 'shared'],
        (0, 1): ['shared'],
    }
    tile_interior_counts = {
        (0, 0): worker_a.n_interior,
        (0, 1): worker_b.n_interior,
    }

    return DistributedPowerGridModel(
        backend=be,
        workers=workers,
        interface_nodes=interface_nodes,
        tile_boundary_nodes=tile_boundary_nodes,
        tile_interior_counts=tile_interior_counts,
        package_data=pkg_data,
        metadata=metadata,
    )


class TestTransientDirichletRHS:
    """Tests for the transient Dirichlet RHS fix (B1 blocker).

    The B1 bug was that prepare_transient() used rhs_dirichlet_A (which
    includes C_coeff * C_ud cap contributions from package caps) instead
    of rhs_dirichlet_G (G-only) in the transient time loop. The fix adds
    a separate rhs_dirichlet_G field to DistributedTransientContext.
    """

    def test_rhs_dirichlet_G_differs_from_A_when_caps_present(self):
        """With package cap edges to pad, rhs_dirichlet_G != rhs_dirichlet_A.

        Package cap edge (pad -> shared, 100 fF) means the A-based assembly
        includes C_coeff * C_ud contribution, but G-only does not.
        """
        from distributed.solver import DistributedDDMSolver

        # Package cap from pad (Dirichlet) to interface unknown
        model = _build_two_tile_distributed_model(
            package_cap_edges=[('pad', 'shared', 100.0)],  # 100 fF
        )
        solver = DistributedDDMSolver(model)
        ctx = solver.prepare_transient(dt=1e-9, method='be')

        # rhs_dirichlet_interface is the A-based (G + C_coeff*C) version
        rhs_A = ctx.rhs_dirichlet_interface
        # rhs_dirichlet_G is the G-only version
        rhs_G = ctx.rhs_dirichlet_G

        assert rhs_G is not None, "rhs_dirichlet_G should be populated"
        assert rhs_A.shape == rhs_G.shape

        # They must differ because the cap edge adds C_coeff * C_ud * Vdd
        assert not np.allclose(rhs_A, rhs_G), (
            "rhs_dirichlet_A and rhs_dirichlet_G should differ when "
            "package cap edges connect unknowns to Dirichlet nodes"
        )

        # The A-based RHS should have a larger magnitude (cap term adds
        # positive contribution since -C_coeff * C_ud * Vdd is negative
        # but Vdd > 0, so the effect depends on sign convention).
        # Just verify they differ; the direction depends on assembly details.
        diff = np.abs(rhs_A - rhs_G)
        assert np.max(diff) > 1e-6, (
            f"Difference too small: max_diff={np.max(diff):.2e}"
        )

    def test_rhs_dirichlet_G_equals_A_when_no_caps(self):
        """Without package cap edges, rhs_dirichlet_G == rhs_dirichlet_A.

        When there are no package cap edges, the A-based assembly only
        uses resistive package edges, same as G-only assembly.
        """
        from distributed.solver import DistributedDDMSolver

        model = _build_two_tile_distributed_model(
            package_cap_edges=[],  # No package caps
        )
        solver = DistributedDDMSolver(model)
        ctx = solver.prepare_transient(dt=1e-9, method='be')

        rhs_A = ctx.rhs_dirichlet_interface
        rhs_G = ctx.rhs_dirichlet_G

        assert rhs_G is not None
        np.testing.assert_allclose(
            rhs_G, rhs_A, atol=1e-12,
            err_msg="Without package caps, G-only and A-based Dirichlet "
                    "RHS should be identical",
        )

    def test_rhs_dirichlet_G_populated_and_nonzero(self):
        """Smoke test: rhs_dirichlet_G is populated with nonzero values.

        Note: exact match with DC context not expected because tile Schur
        complements are A-based (include cap terms) while DC uses G-only.
        We verify: same shape and both nonzero.
        """
        from distributed.solver import DistributedDDMSolver

        model = _build_two_tile_distributed_model(
            package_cap_edges=[('pad', 'shared', 100.0)],
        )
        solver = DistributedDDMSolver(model)
        ctx = solver.prepare_transient(dt=1e-9, method='be')

        rhs_G = ctx.rhs_dirichlet_G
        rhs_dc = ctx.dc_context.rhs_dirichlet_interface

        assert rhs_G is not None
        assert rhs_dc is not None
        assert rhs_G.shape == rhs_dc.shape, (
            f"Shape mismatch: rhs_G={rhs_G.shape}, rhs_dc={rhs_dc.shape}"
        )
        # Both should be nonzero since we have Dirichlet nodes with V=1.0
        assert np.any(np.abs(rhs_G) > 1e-12), "rhs_dirichlet_G should be nonzero"
        assert np.any(np.abs(rhs_dc) > 1e-12), "DC rhs_dirichlet should be nonzero"

    def test_rhs_dirichlet_G_trapezoidal(self):
        """Trapezoidal integration also produces correct rhs_dirichlet_G."""
        from distributed.solver import DistributedDDMSolver

        model = _build_two_tile_distributed_model(
            package_cap_edges=[('pad', 'shared', 50.0)],
        )
        solver = DistributedDDMSolver(model)
        ctx = solver.prepare_transient(dt=1e-9, method='trap')

        rhs_A = ctx.rhs_dirichlet_interface
        rhs_G = ctx.rhs_dirichlet_G

        assert rhs_G is not None
        assert rhs_A.shape == rhs_G.shape
        assert ctx.integration_method == 'trap'
        # Trapezoidal has C_coeff = 2/dt_scaled, so cap contribution is larger
        assert not np.allclose(rhs_A, rhs_G), (
            "Trapezoidal: rhs_dirichlet_A and rhs_dirichlet_G should differ "
            "when package caps are present"
        )


# ──────────────────────────────────────────────────────────────────────
# 9. Time-domain worker stats tests
# ──────────────────────────────────────────────────────────────────────


class TestTimeDomainWorkerStats:
    """Verify that time-domain TileWorker methods return stats dicts
    with the expected keys, as added in Tasks 5/6."""

    def test_factor_transient_returns_stats(self):
        """factor_transient_system returns a 4-tuple with expected stats keys."""
        worker = _make_worker_with_tile(
            edges=[
                ('a', 'b', 2.0),   # 2 mS
                ('b', 'c', 1.0),   # 1 mS
                ('c', '0', 0.5),   # 0.5 mS to ground
            ],
            port_nodes={'a'},
            currents={'c': 0.1},
            cap_edges=[
                ('b', '0', 8.0),   # 8 fF grounded on interior
                ('a', '0', 3.0),   # 3 fF grounded on port
            ],
        )

        dt_scaled = 100.0  # 100 ps
        result = worker.factor_transient_system(dt_scaled, method='be')

        # Must be a 4-tuple
        assert len(result) == 4
        S_A, port_list, total_cap, stats = result

        assert isinstance(S_A, np.ndarray)
        assert isinstance(port_list, list)
        assert isinstance(total_cap, float)
        assert isinstance(stats, dict)

        # Total cap should be 8 + 3 = 11 fF
        np.testing.assert_allclose(total_cap, 11.0, rtol=1e-6)

        # Check all expected stats keys
        expected_keys = {
            'factor_interior_s',
            'compute_schur_s',
            'total_cap_fF',
            'A_ii_nnz',
            'A_pp_nnz',
            'schur_mem_bytes',
            'schur_chunk_size',
            'factorization_backend',
            'factorization_backend_info',
            'n_ports',
            'n_interior',
            'c_ii_cap_nodes',
            'c_pp_cap_nodes',
            'C_coeff',
        }
        for key in expected_keys:
            assert key in stats, f"Missing stats key: {key}"

        # Sanity: timing values are non-negative floats
        assert stats['factor_interior_s'] >= 0.0
        assert stats['compute_schur_s'] >= 0.0

        # Sanity: capacitance stats match topology
        assert stats['total_cap_fF'] == total_cap
        assert stats['n_ports'] == 1
        assert stats['n_interior'] == 2   # b, c
        assert stats['c_ii_cap_nodes'] >= 1  # at least b has cap
        assert stats['c_pp_cap_nodes'] >= 1  # a has cap

        # C_coeff for BE: 1 / dt_scaled
        np.testing.assert_allclose(stats['C_coeff'], 1.0 / dt_scaled)

        # Schur complement is (n_ports x n_ports)
        assert S_A.shape == (1, 1)

    def test_evaluate_and_get_reduced_rhs_returns_stats(self):
        """evaluate_and_get_reduced_rhs returns a 3-tuple with stats (static fallback)."""
        worker = _make_worker_with_tile(
            edges=[
                ('a', 'b', 2.0),
                ('b', '0', 1.0),
            ],
            port_nodes={'a'},
            currents={'b': 0.5},   # 0.5 mA static current
        )

        # No VCS loaded -> static fallback path (_active_sources is None)
        result = worker.evaluate_and_get_reduced_rhs(t=0.0)

        # Must be a 3-tuple
        assert len(result) == 3
        g, total_current, stats = result

        assert isinstance(g, np.ndarray)
        assert isinstance(total_current, float)
        assert isinstance(stats, dict)

        # Check expected stats keys
        expected_keys = {'eval_time_s', 'n_active_sources'}
        for key in expected_keys:
            assert key in stats, f"Missing stats key: {key}"

        # Sanity: timing is non-negative
        assert stats['eval_time_s'] >= 0.0

        # Static path: 1 active source (b has 0.5 mA != 0)
        assert stats['n_active_sources'] == 1

        # Total current should be the sum of static injections
        np.testing.assert_allclose(total_current, 0.5)

    def test_evaluate_and_get_reduced_rhs_with_vcs_returns_stats(self):
        """evaluate_and_get_reduced_rhs returns stats when VCS is loaded."""
        from analysis.vectorized_sources import VectorizedCurrentSources
        from parser.current_sources import CurrentSource

        worker = _make_worker_with_tile(
            edges=[
                ('a', 'b', 2.0),
                ('b', '0', 1.0),
            ],
            port_nodes={'a'},
            currents={'b': 0.5},
        )

        # Build a VCS with one DC source
        n_ports = worker._block_system.n_ports
        n_interior = worker._block_system.n_interior
        n_nodes = n_ports + n_interior
        node_to_idx = dict(worker._block_system.port_to_idx)
        for node, idx in worker._block_system.interior_to_idx.items():
            node_to_idx[node] = idx + n_ports

        src = CurrentSource(name='i1', node1='b', node2='0', dc_value=1.0)
        worker._vec_sources = VectorizedCurrentSources.from_current_sources(
            {'i1': src}, node_to_idx, n_nodes,
        )
        worker._active_sources = worker._vec_sources

        result = worker.evaluate_and_get_reduced_rhs(t=0.0)

        assert len(result) == 3
        g, total_current, stats = result

        expected_keys = {'eval_time_s', 'n_active_sources'}
        for key in expected_keys:
            assert key in stats, f"Missing stats key: {key}"

        assert isinstance(g, np.ndarray)
        assert isinstance(total_current, float)
        assert isinstance(stats, dict)
        assert stats['eval_time_s'] >= 0.0
        # VCS path: one DC source -> at least 1 active source
        assert stats['n_active_sources'] >= 1
        np.testing.assert_allclose(total_current, 1.0, rtol=1e-6)

    def test_get_transient_reduced_rhs_returns_stats(self):
        """get_transient_reduced_rhs returns a 3-tuple with stats."""
        worker = _make_worker_with_tile(
            edges=[
                ('a', 'b', 2.0),
                ('b', 'c', 1.0),
                ('c', '0', 0.5),
            ],
            port_nodes={'a'},
            currents={'c': 0.2},
            cap_edges=[
                ('b', '0', 5.0),
                ('c', '0', 3.0),
            ],
        )

        # Factor the transient system first (required before get_transient_reduced_rhs)
        dt_scaled = 100.0  # 100 ps
        worker.factor_transient_system(dt_scaled, method='be')

        # Need previous-step port voltages
        boundary_v_old = {'a': 1.0}

        result = worker.get_transient_reduced_rhs(t=0.0, boundary_v_old=boundary_v_old)

        # Must be a 3-tuple
        assert len(result) == 3
        g, total_current, stats = result

        assert isinstance(g, np.ndarray)
        assert isinstance(total_current, float)
        assert isinstance(stats, dict)

        # Check expected stats keys
        expected_keys = {'rhs_time_s', 'total_current', 'rhs_norm'}
        for key in expected_keys:
            assert key in stats, f"Missing stats key: {key}"

        # Sanity: timing is non-negative
        assert stats['rhs_time_s'] >= 0.0

        # total_current and rhs_norm should be numeric
        assert isinstance(stats['total_current'], float)
        assert isinstance(stats['rhs_norm'], float)
        assert stats['rhs_norm'] >= 0.0

        # total_current in stats should match the returned value
        np.testing.assert_allclose(stats['total_current'], total_current)

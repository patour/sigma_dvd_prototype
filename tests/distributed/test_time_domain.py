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
        from distributed.result import DistributedTransientContext

        ctx = DistributedTransientContext(
            interface_lu=lambda x: x,
            interface_nodes=[],
            interface_node_to_idx={},
            rhs_dirichlet_interface=np.zeros(0),
            tile_index_maps={},
            dt_scaled=100.0,
            integration_method='be',
            has_capacitance=True,
        )

        np.testing.assert_allclose(ctx.dt, 100e-12)

    def test_transient_context_C_coeff_be(self):
        """BE: C_coeff = 1 / dt_scaled."""
        from distributed.result import DistributedTransientContext

        ctx = DistributedTransientContext(
            interface_lu=lambda x: x,
            interface_nodes=[],
            interface_node_to_idx={},
            rhs_dirichlet_interface=np.zeros(0),
            tile_index_maps={},
            dt_scaled=100.0,
            integration_method='be',
            has_capacitance=True,
        )

        np.testing.assert_allclose(ctx.C_coeff, 0.01)

    def test_transient_context_C_coeff_trap(self):
        """Trapezoidal: C_coeff = 2 / dt_scaled."""
        from distributed.result import DistributedTransientContext

        ctx = DistributedTransientContext(
            interface_lu=lambda x: x,
            interface_nodes=[],
            interface_node_to_idx={},
            rhs_dirichlet_interface=np.zeros(0),
            tile_index_maps={},
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
        dc_ctx = solver.prepare()
        ctx = solver.prepare_transient(dt=1e-9, method='be')

        rhs_G = ctx.rhs_dirichlet_G
        rhs_dc = dc_ctx.rhs_dirichlet_interface

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


# ──────────────────────────────────────────────────────────────────────
# 10. Factorization lifecycle tests
# ──────────────────────────────────────────────────────────────────────


class TestFactorizationLifecycle:
    """Tests for clear_dc_factorization and clear_transient_factorization."""

    def test_clear_dc_factorization_preserves_matrices(self):
        """After factor + clear_dc, G matrices survive but LU is gone."""
        worker = _make_worker_with_tile(
            edges=[
                ('a', 'b', 2.0),
                ('b', 'c', 1.0),
                ('c', '0', 0.5),
            ],
            port_nodes={'a'},
            currents={'c': 0.1},
        )

        # Verify LU exists after setup (factor_and_compute_schur was called)
        bs = worker._block_system
        assert bs.lu_ii is not None, "LU should exist after factor"

        worker.clear_dc_factorization()

        # LU and factor_adapter should be cleared
        assert bs.lu_ii is None, "lu_ii should be None after clear"
        assert bs.factor_adapter is None, "factor_adapter should be None after clear"

        # G matrices should still be present and valid
        assert bs.G_ii is not None
        assert bs.G_pp is not None
        assert bs.G_pi is not None
        assert bs.G_ip is not None
        assert bs.G_ii.shape[0] == bs.n_interior
        assert bs.G_pp.shape[0] == bs.n_ports

    def test_clear_dc_factorization_returns_stats(self):
        """Returned dict has correct had_lu and n_interior values."""
        worker = _make_worker_with_tile(
            edges=[
                ('a', 'b', 2.0),
                ('b', '0', 1.0),
            ],
            port_nodes={'a'},
            currents={'b': 0.5},
        )

        stats = worker.clear_dc_factorization()

        assert stats['had_lu'] is True
        assert stats['n_interior'] == 1  # only 'b' is interior

    def test_clear_dc_factorization_idempotent(self):
        """Calling clear_dc twice should not raise; second returns had_lu=False."""
        worker = _make_worker_with_tile(
            edges=[
                ('a', 'b', 2.0),
                ('b', '0', 1.0),
            ],
            port_nodes={'a'},
        )

        stats1 = worker.clear_dc_factorization()
        assert stats1['had_lu'] is True

        stats2 = worker.clear_dc_factorization()
        assert stats2['had_lu'] is False
        assert stats2['n_interior'] == stats1['n_interior']

    def test_clear_transient_factorization_removes_system(self):
        """After factor_transient + clear, _transient_block_system is None."""
        worker = _make_worker_with_tile(
            edges=[
                ('a', 'b', 2.0),
                ('b', '0', 1.0),
            ],
            port_nodes={'a'},
            cap_edges=[('b', '0', 5.0)],
        )

        dt_scaled = 100.0  # 100 ps
        worker.factor_transient_system(dt_scaled, method='be')
        assert worker._transient_block_system is not None

        stats = worker.clear_transient_factorization()

        assert stats['had_transient_system'] is True
        assert worker._transient_block_system is None

    def test_clear_transient_factorization_preserves_dc(self):
        """Clearing transient leaves DC _block_system.lu_ii intact."""
        worker = _make_worker_with_tile(
            edges=[
                ('a', 'b', 2.0),
                ('b', '0', 1.0),
            ],
            port_nodes={'a'},
            cap_edges=[('b', '0', 5.0)],
        )

        # Both DC and transient systems exist
        dt_scaled = 100.0
        worker.factor_transient_system(dt_scaled, method='be')
        assert worker._block_system.lu_ii is not None
        assert worker._transient_block_system is not None

        worker.clear_transient_factorization()

        # DC system should be untouched
        assert worker._block_system.lu_ii is not None
        assert worker._block_system.factor_adapter is not None
        # Transient system should be gone
        assert worker._transient_block_system is None

    def test_clear_transient_factorization_clears_stepping_state(self):
        """After clear_transient, _last_f_i and _v_interior_old should be None."""
        worker = _make_worker_with_tile(
            edges=[
                ('a', 'b', 2.0),
                ('b', '0', 1.0),
            ],
            port_nodes={'a'},
            cap_edges=[('b', '0', 5.0)],
        )

        dt_scaled = 100.0
        worker.factor_transient_system(dt_scaled, method='be')

        # Run one transient RHS to populate _last_f_i
        worker.get_transient_reduced_rhs(t=0.0, boundary_v_old={'a': 1.0})
        assert worker._last_f_i is not None

        # Set initial voltages so _v_interior_old is populated
        worker.set_initial_voltages({'a': 1.0, 'b': 0.9})
        assert worker._v_interior_old is not None

        worker.clear_transient_factorization()

        assert worker._last_f_i is None
        assert worker._v_interior_old is None


# ──────────────────────────────────────────────────────────────────────
# 11. Context lifecycle tests (factor / release / topology reuse)
# ──────────────────────────────────────────────────────────────────────


class TestContextLifecycle:
    """Test factor/release lifecycle on active context classes."""

    def test_dc_context_factor_without_model_raises(self):
        """DistributedSolverContext.factor() raises if model is None."""
        from distributed.result import DistributedSolverContext

        ctx = DistributedSolverContext()
        with pytest.raises(RuntimeError, match="Cannot factor without a model"):
            ctx.factor()

    def test_transient_context_factor_without_model_raises(self):
        """DistributedTransientContext.factor() raises if model is None."""
        from distributed.result import DistributedTransientContext

        ctx = DistributedTransientContext(dt=1e-10, method='be')
        with pytest.raises(RuntimeError, match="Cannot factor without a model"):
            ctx.factor()

    def test_dc_context_backward_compat_is_factored(self):
        """Context built with explicit interface_lu is marked as factored."""
        from distributed.result import DistributedSolverContext

        ctx = DistributedSolverContext(
            interface_lu=lambda x: x,
            interface_nodes=['a'],
            interface_node_to_idx={'a': 0},
            rhs_dirichlet_interface=np.zeros(1),
            tile_index_maps={},
        )
        assert ctx.is_factored is True

    def test_transient_context_backward_compat_is_factored(self):
        """Transient context built with explicit interface_lu is marked as factored."""
        from distributed.result import DistributedTransientContext

        ctx = DistributedTransientContext(
            interface_lu=lambda x: x,
            interface_nodes=['a'],
            interface_node_to_idx={'a': 0},
            rhs_dirichlet_interface=np.zeros(1),
            tile_index_maps={},
            dt_scaled=100.0,
            integration_method='be',
            has_capacitance=False,
        )
        assert ctx.is_factored is True

    def test_dc_context_release_clears_state(self):
        """release() on DC context clears interface_lu and is_factored."""
        from distributed.result import DistributedSolverContext

        ctx = DistributedSolverContext(
            interface_lu=lambda x: x,
            interface_nodes=['a'],
            interface_node_to_idx={'a': 0},
            rhs_dirichlet_interface=np.zeros(1),
            tile_index_maps={},
        )
        assert ctx.is_factored is True
        ctx.release()
        assert ctx.is_factored is False
        assert ctx.interface_lu is None

    def test_transient_context_release_clears_state(self):
        """release() on transient context clears interface_lu and is_factored."""
        from distributed.result import DistributedTransientContext

        ctx = DistributedTransientContext(
            interface_lu=lambda x: x,
            interface_nodes=['a'],
            interface_node_to_idx={'a': 0},
            rhs_dirichlet_interface=np.zeros(1),
            tile_index_maps={},
            dt_scaled=100.0,
            integration_method='be',
            has_capacitance=False,
        )
        assert ctx.is_factored is True
        ctx.release()
        assert ctx.is_factored is False
        assert ctx.interface_lu is None
        assert ctx.rhs_dirichlet_A is None
        assert ctx.C_package_uu is None

    def test_dc_context_topology_preserved_after_release(self):
        """release() on DC context preserves topology."""
        from distributed.result import (
            DistributedSolverContext,
            DistributedTopologyContext,
        )

        topo = DistributedTopologyContext(
            interface_nodes=['a'],
            interface_node_to_idx={'a': 0},
            tile_index_maps={},
            rhs_dirichlet_G=np.zeros(1),
            G_package_uu=None,
        )
        ctx = DistributedSolverContext(
            topology=topo,
            interface_lu=lambda x: x,
            interface_nodes=['a'],
            interface_node_to_idx={'a': 0},
            rhs_dirichlet_interface=np.zeros(1),
            tile_index_maps={},
        )
        ctx.release()
        assert ctx.topology is topo

    def test_topology_reuse_across_prepare_calls(self):
        """prepare() then prepare_transient() share the same topology."""
        model = _build_two_tile_distributed_model(package_cap_edges=[])
        from distributed.solver import DistributedDDMSolver

        solver = DistributedDDMSolver(model)
        dc_ctx = solver.prepare()
        assert dc_ctx.topology is not None

        trans_ctx = solver.prepare_transient(dt=1e-10, method='be')
        # Topology should be the exact same object (cached on solver)
        assert trans_ctx.topology is dc_ctx.topology


class TestContextSaveLoad:
    """Tests for context checkpoint save / load / refactor round-trips."""

    def test_dc_context_save_load_round_trip(self, tmp_path):
        """Save a DC context, load it, refactor, and verify identical solve.

        Verifies that topology, S_global, and timings survive the
        pickle round-trip, and that refactor() rebuilds the LU so that
        the reloaded context produces the same interface solution as the
        original.
        """
        from distributed.solver import DistributedDDMSolver

        model = _build_two_tile_distributed_model(package_cap_edges=[])
        solver = DistributedDDMSolver(model)

        # Original prepare (factor)
        ctx_orig = solver.prepare()
        assert ctx_orig.is_factored is True

        # Save
        save_path = str(tmp_path / 'dc_context.pkl')
        returned_path = ctx_orig.save(path=save_path)
        assert returned_path == save_path
        assert os.path.isfile(save_path)

        # Load
        from distributed.result import DistributedSolverContext
        ctx_loaded = DistributedSolverContext.load(model, save_path)
        assert ctx_loaded.is_factored is False
        assert ctx_loaded.topology is not None

        # Verify topology matches
        assert ctx_loaded.interface_nodes == ctx_orig.interface_nodes
        assert (
            ctx_loaded.interface_node_to_idx == ctx_orig.interface_node_to_idx
        )
        for tid in ctx_orig.tile_index_maps:
            np.testing.assert_array_equal(
                ctx_loaded.tile_index_maps[tid],
                ctx_orig.tile_index_maps[tid],
            )

        # Verify S_global round-trips correctly
        assert ctx_loaded._S_global is not None
        np.testing.assert_array_equal(
            ctx_loaded._S_global.toarray(),
            ctx_orig._S_global.toarray(),
        )

        # Verify rhs_dirichlet round-trips via topology
        np.testing.assert_array_equal(
            ctx_loaded.rhs_dirichlet_interface,
            ctx_orig.rhs_dirichlet_interface,
        )

        # Refactor from saved S_global and verify interface solve matches
        ctx_loaded.refactor()
        assert ctx_loaded.is_factored is True

        # Solve the same RHS with both contexts and compare
        rhs = ctx_orig.rhs_dirichlet_interface.copy()
        v_orig = ctx_orig.interface_lu(rhs)
        v_loaded = ctx_loaded.interface_lu(rhs)
        np.testing.assert_allclose(v_loaded, v_orig, atol=1e-12)

    def test_transient_context_save_load_round_trip(self, tmp_path):
        """Save a transient context, load it, refactor, and verify fields."""
        from distributed.solver import DistributedDDMSolver

        pkg_caps = [('pad', 'shared', 50.0)]
        model = _build_two_tile_distributed_model(package_cap_edges=pkg_caps)
        solver = DistributedDDMSolver(model)

        # Prepare transient
        trans_ctx = solver.prepare_transient(dt=1e-10, method='be')
        assert trans_ctx.is_factored is True

        # Save
        save_path = str(tmp_path / 'transient_context.pkl')
        returned_path = trans_ctx.save(path=save_path)
        assert returned_path == save_path
        assert os.path.isfile(save_path)

        # Load
        from distributed.result import DistributedTransientContext
        ctx_loaded = DistributedTransientContext.load(model, save_path)
        assert ctx_loaded.is_factored is False

        # Verify integration params
        np.testing.assert_allclose(
            ctx_loaded.dt_scaled, trans_ctx.dt_scaled, rtol=1e-12,
        )
        assert ctx_loaded.integration_method == trans_ctx.integration_method
        assert ctx_loaded.has_capacitance == trans_ctx.has_capacitance

        # Verify topology
        assert ctx_loaded.interface_nodes == trans_ctx.interface_nodes

        # Verify transient-specific matrices
        if trans_ctx.rhs_dirichlet_A is not None:
            np.testing.assert_array_equal(
                ctx_loaded.rhs_dirichlet_A, trans_ctx.rhs_dirichlet_A,
            )
        if trans_ctx.rhs_dirichlet_G is not None:
            np.testing.assert_array_equal(
                ctx_loaded.rhs_dirichlet_G, trans_ctx.rhs_dirichlet_G,
            )
        if trans_ctx.C_package_uu is not None:
            np.testing.assert_array_equal(
                ctx_loaded.C_package_uu.toarray(),
                trans_ctx.C_package_uu.toarray(),
            )
        if trans_ctx.G_package_uu is not None:
            np.testing.assert_array_equal(
                ctx_loaded.G_package_uu.toarray(),
                trans_ctx.G_package_uu.toarray(),
            )

        # Refactor and verify usable
        ctx_loaded.refactor()
        assert ctx_loaded.is_factored is True

    def test_dc_context_save_default_path(self):
        """save() with path=None uses the model-derived default path."""
        from distributed.result import DistributedSolverContext

        model = _build_two_tile_distributed_model(package_cap_edges=[])

        # The model's tile_configs have empty ckt_path (''), so the
        # default path logic should fall through to './checkpoint/...'
        ctx = DistributedSolverContext(model=model)
        default_path = ctx._default_checkpoint_path('dc_context.pkl')
        assert default_path.endswith('dc_context.pkl')
        assert 'checkpoint' in default_path

    def test_load_wrong_type_raises(self, tmp_path):
        """Loading a DC checkpoint as transient raises ValueError."""
        from distributed.result import (
            DistributedSolverContext,
            DistributedTransientContext,
        )

        model = _build_two_tile_distributed_model(package_cap_edges=[])
        solver_module = __import__('distributed.solver', fromlist=['DistributedDDMSolver'])
        DistributedDDMSolver = solver_module.DistributedDDMSolver
        solver = DistributedDDMSolver(model)

        # Prepare and save a DC context
        dc_ctx = solver.prepare()
        save_path = str(tmp_path / 'dc_context.pkl')
        dc_ctx.save(path=save_path)

        # Try to load as transient -- should raise ValueError
        with pytest.raises(ValueError, match="Expected DistributedTransientContext"):
            DistributedTransientContext.load(model, save_path)

        # Also verify loading DC as DC works fine
        loaded = DistributedSolverContext.load(model, save_path)
        assert loaded.topology is not None

    def test_refactor_without_s_global_raises(self):
        """refactor() without S_global raises RuntimeError."""
        from distributed.result import (
            DistributedSolverContext,
            DistributedTransientContext,
        )

        # DC context without S_global
        dc_ctx = DistributedSolverContext()
        assert dc_ctx._S_global is None
        with pytest.raises(RuntimeError, match="Cannot refactor without S_global"):
            dc_ctx.refactor()

        # Transient context without S_global
        trans_ctx = DistributedTransientContext(dt=1e-10, method='be')
        assert trans_ctx._S_global is None
        with pytest.raises(RuntimeError, match="Cannot refactor without S_global"):
            trans_ctx.refactor()

    def test_save_after_release_raises(self, tmp_path):
        """save() after release() raises RuntimeError (S_global is None)."""
        from distributed.result import DistributedSolverContext

        model = _build_two_tile_distributed_model(package_cap_edges=[])
        from distributed.solver import DistributedDDMSolver

        solver = DistributedDDMSolver(model)
        ctx = solver.prepare()
        ctx.release()

        with pytest.raises(RuntimeError, match="Cannot save"):
            ctx.save(str(tmp_path / 'should_fail.pkl'))


# ──────────────────────────────────────────────────────────────────────
# 14. recover_and_set_initial_voltages tests
# ──────────────────────────────────────────────────────────────────────


class TestRecoverAndSetInitialVoltages:
    """Test that recover_and_set_initial_voltages matches get+set."""

    def test_matches_separate_get_and_set(self):
        """recover_and_set should produce same _v_interior_old as get+set."""
        model = _build_two_tile_distributed_model(package_cap_edges=[])
        try:
            # Factor interior LU on each worker so recovery works
            for worker in model.workers:
                worker._block_system.factor_interior()

            # Construct boundary voltage dicts directly.  The model has
            # interface_nodes = {shared, pad} with pad at Vdd=1.0.
            # Use slightly non-trivial voltages to exercise the math.
            vdd = model.vdd
            bv_tile_a = {'shared': 0.95, 'pad': vdd}
            bv_tile_b = {'shared': 0.95}
            bv_list = [bv_tile_a, bv_tile_b]

            for i, worker in enumerate(model.workers):
                bv = bv_list[i]

                # Method 1: separate get + set (original two-step path)
                voltages, _stats = worker.get_interior_voltages(bv)
                worker.set_initial_voltages(voltages)
                v_old_separate = worker._v_interior_old.copy()

                # Reset state
                worker._v_interior_old = None

                # Method 2: combined recover + set (optimized path)
                worker.recover_and_set_initial_voltages(bv)
                v_old_combined = worker._v_interior_old.copy()

                np.testing.assert_allclose(
                    v_old_separate, v_old_combined,
                    atol=1e-12,
                    err_msg=f"Worker {i}: recover_and_set_initial_voltages "
                            f"produced different _v_interior_old",
                )
        finally:
            model.shutdown()

    def test_returns_stats_dict(self):
        """recover_and_set_initial_voltages returns a stats dict."""
        model = _build_two_tile_distributed_model(package_cap_edges=[])
        try:
            for worker in model.workers:
                worker._block_system.factor_interior()

            bv = {'shared': 0.95, 'pad': model.vdd}
            stats = model.workers[0].recover_and_set_initial_voltages(bv)

            assert isinstance(stats, dict)
            assert 'n_interior' in stats
            assert 'n_ports' in stats
            assert stats['n_interior'] > 0
        finally:
            model.shutdown()


# ──────────────────────────────────────────────────────────────────────
# 15. Current node mask tests
# ──────────────────────────────────────────────────────────────────────


def _make_worker_with_coordinate_nodes():
    """Create a TileWorker with PDN-style coordinate node names.

    Topology (single tile):
        1000_2000_M0  (interior, x=1000, y=2000) --[1mS]-- 3000_4000_M0 (port)
        1000_2000_M0  --[1mS]-- 0 (ground)
        5000_6000_M0  (interior, x=5000, y=6000) --[2mS]-- 3000_4000_M0 (port)
        5000_6000_M0  --[1mS]-- 0 (ground)

    Port: {3000_4000_M0}
    Interior: {1000_2000_M0, 5000_6000_M0}

    Current injections: 1000_2000_M0 -> 0.5 mA, 5000_6000_M0 -> 0.3 mA

    Returns:
        Configured TileWorker instance with coordinate-parseable node names.
    """
    return _make_worker_with_tile(
        edges=[
            ('1000_2000_M0', '3000_4000_M0', 1.0),
            ('1000_2000_M0', '0', 1.0),
            ('5000_6000_M0', '3000_4000_M0', 2.0),
            ('5000_6000_M0', '0', 1.0),
        ],
        port_nodes={'3000_4000_M0'},
        currents={'1000_2000_M0': 0.5, '5000_6000_M0': 0.3},
    )


class TestCurrentNodeMask:
    """Tests for set_current_node_mask and build_node_mask_for_window."""

    def test_build_mask_inside_window(self):
        """Mask selects nodes inside a spatial window; outside nodes are 0."""
        worker = _make_worker_with_coordinate_nodes()

        # Window covers (0..2000, 0..3000) => includes 1000_2000_M0 only
        mask = worker.build_node_mask_for_window(
            x_min=0, x_max=2000, y_min=0, y_max=3000, inside=True,
        )

        bs = worker._block_system
        n_ports = bs.n_ports
        # Node 1000_2000_M0 is interior, should be 1.0
        idx_1000 = bs.interior_to_idx['1000_2000_M0'] + n_ports
        assert mask[idx_1000] == 1.0

        # Node 5000_6000_M0 is interior, outside the window, should be 0.0
        idx_5000 = bs.interior_to_idx['5000_6000_M0'] + n_ports
        assert mask[idx_5000] == 0.0

        # Port node 3000_4000_M0: x=3000 > x_max=2000, outside window
        idx_port = bs.port_to_idx['3000_4000_M0']
        assert mask[idx_port] == 0.0

    def test_build_mask_outside_window(self):
        """With inside=False, mask selects nodes OUTSIDE the window."""
        worker = _make_worker_with_coordinate_nodes()

        # Window covers (0..2000, 0..3000) => 1000_2000_M0 inside
        mask = worker.build_node_mask_for_window(
            x_min=0, x_max=2000, y_min=0, y_max=3000, inside=False,
        )

        bs = worker._block_system
        n_ports = bs.n_ports

        # 1000_2000_M0 is inside window => masked OUT => 0.0
        idx_1000 = bs.interior_to_idx['1000_2000_M0'] + n_ports
        assert mask[idx_1000] == 0.0

        # 5000_6000_M0 is outside window => selected => 1.0
        idx_5000 = bs.interior_to_idx['5000_6000_M0'] + n_ports
        assert mask[idx_5000] == 1.0

        # 3000_4000_M0 is outside window => selected => 1.0
        idx_port = bs.port_to_idx['3000_4000_M0']
        assert mask[idx_port] == 1.0

    def test_mask_shape(self):
        """Returned mask has shape (n_ports + n_interior,)."""
        worker = _make_worker_with_coordinate_nodes()

        mask = worker.build_node_mask_for_window(
            x_min=0, x_max=10000, y_min=0, y_max=10000, inside=True,
        )

        bs = worker._block_system
        expected_size = bs.n_ports + bs.n_interior
        assert mask.shape == (expected_size,)
        assert mask.dtype == np.float64

    def test_set_and_clear_mask(self):
        """set_current_node_mask stores the mask; None clears it."""
        worker = _make_worker_with_tile(
            edges=[('a', 'b', 2.0), ('b', '0', 1.0)],
            port_nodes={'a'},
        )

        bs = worker._block_system
        n_total = bs.n_ports + bs.n_interior
        mask = np.ones(n_total, dtype=np.float64)

        # Set mask
        worker.set_current_node_mask(mask)
        assert worker._current_node_mask is not None
        np.testing.assert_array_equal(worker._current_node_mask, mask)

        # Clear mask
        worker.set_current_node_mask(None)
        assert worker._current_node_mask is None

    def test_mask_zeros_currents_in_qs_path(self):
        """All-zeros mask zeroes out all current contributions in evaluate_and_get_reduced_rhs.

        With mask=0 everywhere, the reduced RHS should only contain the
        Dirichlet contribution (no current from VCS). Compare with a call
        without mask to verify they differ.
        """
        from analysis.vectorized_sources import VectorizedCurrentSources
        from parser.current_sources import CurrentSource

        worker = _make_worker_with_tile(
            edges=[('a', 'b', 2.0), ('b', '0', 1.0)],
            port_nodes={'a'},
            currents={'b': 0.5},
        )

        # Build VCS with a DC source on interior node 'b'
        bs = worker._block_system
        n_ports = bs.n_ports
        n_interior = bs.n_interior
        n_nodes = n_ports + n_interior
        node_to_idx = dict(bs.port_to_idx)
        for node, idx in bs.interior_to_idx.items():
            node_to_idx[node] = idx + n_ports

        src = CurrentSource(name='i1', node1='b', node2='0', dc_value=1.0)
        worker._vec_sources = VectorizedCurrentSources.from_current_sources(
            {'i1': src}, node_to_idx, n_nodes,
        )
        worker._active_sources = worker._vec_sources

        # Evaluate WITHOUT mask -> should have current contributions
        g_no_mask, total_no_mask, _ = worker.evaluate_and_get_reduced_rhs(t=0.0)

        # Set all-zero mask -> all currents zeroed
        zero_mask = np.zeros(n_nodes, dtype=np.float64)
        worker.set_current_node_mask(zero_mask)

        g_with_mask, total_with_mask, _ = worker.evaluate_and_get_reduced_rhs(t=0.0)

        # Zero mask should produce zero total current
        np.testing.assert_allclose(total_with_mask, 0.0, atol=1e-15)

        # The no-mask path should have nonzero total current
        assert abs(total_no_mask) > 0.1, (
            f"Expected nonzero total current without mask, got {total_no_mask}"
        )

        # The reduced RHS values must differ since current contributions change
        assert not np.allclose(g_no_mask, g_with_mask, atol=1e-10), (
            "Masked and unmasked reduced RHS should differ when VCS is active"
        )

    def test_none_mask_is_noop(self):
        """Setting mask to None produces the same result as never setting one."""
        from analysis.vectorized_sources import VectorizedCurrentSources
        from parser.current_sources import CurrentSource

        worker = _make_worker_with_tile(
            edges=[('a', 'b', 2.0), ('b', '0', 1.0)],
            port_nodes={'a'},
            currents={'b': 0.5},
        )

        bs = worker._block_system
        n_ports = bs.n_ports
        n_interior = bs.n_interior
        n_nodes = n_ports + n_interior
        node_to_idx = dict(bs.port_to_idx)
        for node, idx in bs.interior_to_idx.items():
            node_to_idx[node] = idx + n_ports

        src = CurrentSource(name='i1', node1='b', node2='0', dc_value=1.0)
        worker._vec_sources = VectorizedCurrentSources.from_current_sources(
            {'i1': src}, node_to_idx, n_nodes,
        )
        worker._active_sources = worker._vec_sources

        # Evaluate with no mask ever set (default None)
        g_default, total_default, _ = worker.evaluate_and_get_reduced_rhs(t=0.0)

        # Explicitly set mask to None, evaluate again
        worker.set_current_node_mask(None)
        g_none, total_none, _ = worker.evaluate_and_get_reduced_rhs(t=0.0)

        np.testing.assert_allclose(g_none, g_default, atol=1e-15)
        np.testing.assert_allclose(total_none, total_default, atol=1e-15)

    def test_build_and_set_window_mask_sets_mask_and_returns_count(self):
        """build_and_set_window_mask stores the mask and returns active count."""
        worker = _make_worker_with_coordinate_nodes()

        # Window covers (0..2000, 0..3000) => only 1000_2000_M0 inside
        count = worker.build_and_set_window_mask(
            x_min=0, x_max=2000, y_min=0, y_max=3000, inside=True,
        )

        # Mask must be stored on the worker
        assert worker._current_node_mask is not None

        # The returned count must match the number of nonzero mask entries
        expected_count = int(np.sum(worker._current_node_mask > 0))
        assert count == expected_count

        # Only 1000_2000_M0 is inside the window (1 node active)
        assert count == 1

    def test_build_and_set_window_mask_matches_manual_build(self):
        """build_and_set_window_mask produces the same mask as calling
        build_node_mask_for_window + set_current_node_mask separately."""
        worker = _make_worker_with_coordinate_nodes()

        window = dict(x_min=0, x_max=4000, y_min=0, y_max=5000, inside=True)

        # Build the mask independently (without setting it)
        expected_mask = worker.build_node_mask_for_window(**window)

        # Now use the combined method
        count = worker.build_and_set_window_mask(**window)

        # The stored mask must be identical to the independently built one
        np.testing.assert_array_equal(worker._current_node_mask, expected_mask)

        # Count must agree with the mask
        assert count == int(np.sum(expected_mask > 0))

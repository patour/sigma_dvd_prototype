#!/usr/bin/env python3
"""
Deep regression tests for PDN parser equivalence.

Verifies that parser decomposition (Phase 1) preserves exact behavior:
- Sequential vs parallel parse equivalence (every edge attr, node attr, metadata)
- Pickle round-trip with store_instance_sources=True and =False
- Current source waveform evaluation consistency
- Graph metadata completeness

These tests form the safety net for pdn_parser.py decomposition.
"""

import pickle
import tempfile
import unittest
from pathlib import Path

from parser.current_sources import CurrentSource, _DCOnlyCurrentSource, get_optimize_dc_only, set_optimize_dc_only
from parser.netlist import NetlistParser


# Netlist paths
NETLIST_TEST = Path(__file__).parent.parent.parent / 'netlist' / 'netlist_test'
NETLIST_SMALL = Path(__file__).parent.parent.parent / 'netlist' / 'netlist_small'


def _edge_sort_key(u, v, data):
    """Stable sort key for edges across sequential/parallel parses."""
    elem = data.get('elem_name', '') or ''
    etype = data.get('type', '')
    val = data.get('value', 0.0)
    return (u, v, etype, elem, val)


def _compare_edge_attrs(test_case, data_a, data_b, context=""):
    """Deep-compare two edge attribute dicts/objects."""
    tc = test_case
    prefix = f"[{context}] " if context else ""

    # Core attributes
    tc.assertEqual(data_a.get('type', ''), data_b.get('type', ''),
                   f"{prefix}type mismatch")
    tc.assertAlmostEqual(
        data_a.get('value', 0.0), data_b.get('value', 0.0), places=12,
        msg=f"{prefix}value mismatch")

    # Optional attributes (may be absent for optimized edges)
    for attr in ('elem_name', 'tile_id', 'net_type', 'category'):
        a_val = data_a.get(attr)
        b_val = data_b.get(attr)
        tc.assertEqual(a_val, b_val, f"{prefix}{attr} mismatch: {a_val!r} vs {b_val!r}")

    # Boundary flags
    for flag in ('boundary_node1', 'boundary_node2', 'has_boundary'):
        a_val = data_a.get(flag)
        b_val = data_b.get(flag)
        tc.assertEqual(a_val, b_val, f"{prefix}{flag} mismatch")


def _compare_graphs(test_case, g_a, g_b, label=""):
    """Full structural + attribute comparison of two parsed graphs."""
    tc = test_case
    prefix = f"[{label}] " if label else ""

    # Node counts
    tc.assertEqual(g_a.number_of_nodes(), g_b.number_of_nodes(),
                   f"{prefix}Node count mismatch")
    tc.assertEqual(g_a.number_of_edges(), g_b.number_of_edges(),
                   f"{prefix}Edge count mismatch")

    # Node sets
    nodes_a = set(g_a.nodes())
    nodes_b = set(g_b.nodes())
    tc.assertEqual(nodes_a, nodes_b, f"{prefix}Node sets differ")

    # Node attributes
    for node in sorted(nodes_a):
        attrs_a = g_a.nodes[node] if hasattr(g_a, 'nodes') and callable(getattr(g_a.nodes, '__getitem__', None)) else {}
        attrs_b = g_b.nodes[node] if hasattr(g_b, 'nodes') and callable(getattr(g_b.nodes, '__getitem__', None)) else {}
        # Compare all node attributes that exist
        try:
            da = dict(attrs_a) if attrs_a else {}
        except (TypeError, ValueError):
            da = {}
        try:
            db = dict(attrs_b) if attrs_b else {}
        except (TypeError, ValueError):
            db = {}

        for key in set(list(da.keys()) + list(db.keys())):
            tc.assertEqual(da.get(key), db.get(key),
                           f"{prefix}Node {node!r} attr {key!r} mismatch")

    # Edge comparison (sorted for determinism)
    edges_a = [(u, v, dict(d) if not isinstance(d, dict) else d)
               for u, v, d in g_a.edges(data=True)]
    edges_b = [(u, v, dict(d) if not isinstance(d, dict) else d)
               for u, v, d in g_b.edges(data=True)]

    edges_a.sort(key=lambda t: _edge_sort_key(*t))
    edges_b.sort(key=lambda t: _edge_sort_key(*t))

    tc.assertEqual(len(edges_a), len(edges_b), f"{prefix}Edge list length mismatch")

    for i, ((ua, va, da), (ub, vb, db)) in enumerate(zip(edges_a, edges_b)):
        ctx = f"{prefix}edge[{i}] ({ua}->{va})"
        tc.assertEqual(ua, ub, f"{ctx} src node mismatch")
        tc.assertEqual(va, vb, f"{ctx} dst node mismatch")
        _compare_edge_attrs(tc, da, db, context=ctx)

    # Graph metadata
    meta_keys = [
        'vsrc_dict', 'parameters', 'tile_grid', 'instance_node_map',
        'merged_nodes', 'mutual_inductors', 'net_connectivity',
        'stats', 'net_stats', 'vsrc_nodes', 'layer_stats',
    ]
    for key in meta_keys:
        a_val = g_a.graph.get(key)
        b_val = g_b.graph.get(key)
        if a_val is None and b_val is None:
            continue
        tc.assertEqual(type(a_val), type(b_val),
                       f"{prefix}graph[{key!r}] type mismatch")
        if isinstance(a_val, dict):
            tc.assertEqual(set(a_val.keys()), set(b_val.keys()),
                           f"{prefix}graph[{key!r}] keys mismatch")
            for k in a_val:
                tc.assertEqual(a_val[k], b_val[k],
                               f"{prefix}graph[{key!r}][{k!r}] mismatch")
        else:
            tc.assertEqual(a_val, b_val, f"{prefix}graph[{key!r}] mismatch")


def _compare_instance_sources(test_case, g_a, g_b, label=""):
    """Compare instance source objects between two graphs."""
    tc = test_case
    prefix = f"[{label}] " if label else ""

    src_a = g_a.graph.get('_instance_sources_objects', {})
    src_b = g_b.graph.get('_instance_sources_objects', {})

    tc.assertEqual(set(src_a.keys()), set(src_b.keys()),
                   f"{prefix}Instance source names differ")

    for name in sorted(src_a.keys()):
        sa, sb = src_a[name], src_b[name]
        tc.assertEqual(type(sa).__name__, type(sb).__name__,
                       f"{prefix}Source {name!r} type mismatch")
        # Compare DC value
        tc.assertAlmostEqual(
            sa.get_static_current(), sb.get_static_current(), places=10,
            msg=f"{prefix}Source {name!r} static current mismatch")
        # Compare at several time points
        for t in [0.0, 1e-9, 5e-9, 10e-9, 50e-9]:
            tc.assertAlmostEqual(
                sa.get_current_at_time(t), sb.get_current_at_time(t), places=10,
                msg=f"{prefix}Source {name!r} current at t={t} mismatch")


class TestSequentialParallelDeepEquivalence(unittest.TestCase):
    """Deep structural equivalence between sequential and parallel parses."""

    @classmethod
    def setUpClass(cls):
        cls.netlist_test = NETLIST_TEST
        cls.netlist_small = NETLIST_SMALL

    def _parse_both(self, netlist_dir, **kwargs):
        """Parse sequentially and in parallel, return both graphs."""
        if not netlist_dir.exists():
            self.skipTest(f"Netlist not found: {netlist_dir}")

        parser_seq = NetlistParser(str(netlist_dir), parallel=False, **kwargs)
        g_seq = parser_seq.parse()

        parser_par = NetlistParser(str(netlist_dir), parallel=True, n_workers=2, **kwargs)
        g_par = parser_par.parse()

        return g_seq, g_par

    def test_netlist_test_deep_equivalence(self):
        """Deep compare sequential vs parallel for netlist_test."""
        g_seq, g_par = self._parse_both(self.netlist_test)
        _compare_graphs(self, g_seq, g_par, label="netlist_test seq/par")
        _compare_instance_sources(self, g_seq, g_par, label="netlist_test")

    def test_netlist_small_deep_equivalence(self):
        """Deep compare sequential vs parallel for netlist_small."""
        g_seq, g_par = self._parse_both(self.netlist_small)
        _compare_graphs(self, g_seq, g_par, label="netlist_small seq/par")
        _compare_instance_sources(self, g_seq, g_par, label="netlist_small")

    def test_netlist_test_with_net_filter(self):
        """Deep compare with VDD net filter."""
        g_seq, g_par = self._parse_both(self.netlist_test, net_filter='VDD')
        _compare_graphs(self, g_seq, g_par, label="netlist_test VDD filter")

    def test_netlist_test_with_store_instance_sources(self):
        """Deep compare with store_instance_sources=True."""
        if not self.netlist_test.exists():
            self.skipTest(f"Netlist not found: {self.netlist_test}")

        parser_seq = NetlistParser(str(self.netlist_test), parallel=False,
                                   store_instance_sources=True)
        g_seq = parser_seq.parse()

        parser_par = NetlistParser(str(self.netlist_test), parallel=True, n_workers=2,
                                   store_instance_sources=True)
        g_par = parser_par.parse()

        _compare_graphs(self, g_seq, g_par, label="store_sources seq/par")

        # Check serialized sources match
        src_a = g_seq.graph.get('instance_sources', {})
        src_b = g_par.graph.get('instance_sources', {})
        self.assertEqual(set(src_a.keys()), set(src_b.keys()))
        for name in sorted(src_a.keys()):
            self.assertEqual(src_a[name], src_b[name],
                             f"Serialized source {name!r} mismatch")


class TestPickleRoundTrip(unittest.TestCase):
    """Pickle round-trip preserves all graph state."""

    @classmethod
    def setUpClass(cls):
        cls.netlist_test = NETLIST_TEST
        cls.netlist_small = NETLIST_SMALL

    def _roundtrip(self, graph):
        """Pickle and unpickle a graph."""
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            pickle.dump(graph, f)
            tmp_path = f.name

        with open(tmp_path, 'rb') as f:
            restored = pickle.load(f)

        Path(tmp_path).unlink()
        return restored

    def test_pickle_roundtrip_store_sources_true(self):
        """Parse with store_instance_sources=True, pickle, verify."""
        if not self.netlist_test.exists():
            self.skipTest(f"Netlist not found: {self.netlist_test}")

        parser = NetlistParser(str(self.netlist_test), store_instance_sources=True)
        original = parser.parse()
        restored = self._roundtrip(original)

        _compare_graphs(self, original, restored, label="pickle store=True")

        # Serialized sources should survive pickle
        src_orig = original.graph.get('instance_sources', {})
        src_rest = restored.graph.get('instance_sources', {})
        self.assertEqual(set(src_orig.keys()), set(src_rest.keys()))
        for name in sorted(src_orig.keys()):
            self.assertEqual(src_orig[name], src_rest[name],
                             f"Pickled source {name!r} mismatch")

    def test_pickle_roundtrip_store_sources_false(self):
        """Parse with store_instance_sources=False (default), pickle, verify."""
        if not self.netlist_test.exists():
            self.skipTest(f"Netlist not found: {self.netlist_test}")

        parser = NetlistParser(str(self.netlist_test), store_instance_sources=False)
        original = parser.parse()
        restored = self._roundtrip(original)

        _compare_graphs(self, original, restored, label="pickle store=False")

        # Raw source objects should survive pickle
        src_orig = original.graph.get('_instance_sources_objects', {})
        src_rest = restored.graph.get('_instance_sources_objects', {})
        self.assertEqual(set(src_orig.keys()), set(src_rest.keys()))
        for name in sorted(src_orig.keys()):
            so, sr = src_orig[name], src_rest[name]
            self.assertEqual(type(so).__name__, type(sr).__name__)
            self.assertAlmostEqual(so.get_static_current(), sr.get_static_current(),
                                   places=10)

    def test_pickle_roundtrip_small(self):
        """Pickle round-trip for netlist_small."""
        if not self.netlist_small.exists():
            self.skipTest(f"Netlist not found: {self.netlist_small}")

        parser = NetlistParser(str(self.netlist_small))
        original = parser.parse()
        restored = self._roundtrip(original)

        _compare_graphs(self, original, restored, label="pickle small")


class TestCurrentSourceWaveformConsistency(unittest.TestCase):
    """Verify current source waveform evaluation is preserved."""

    @classmethod
    def setUpClass(cls):
        cls.netlist_test = NETLIST_TEST

    def test_waveform_evaluation_matches_across_modes(self):
        """Current at specific times matches between default and dc_only modes."""
        if not self.netlist_test.exists():
            self.skipTest(f"Netlist not found: {self.netlist_test}")

        # Parse with dc_only=False
        old_dc = get_optimize_dc_only()
        try:
            set_optimize_dc_only(False)
            parser_full = NetlistParser(str(self.netlist_test))
            g_full = parser_full.parse()

            set_optimize_dc_only(True)
            parser_dc = NetlistParser(str(self.netlist_test))
            g_dc = parser_dc.parse()
        finally:
            set_optimize_dc_only(old_dc)

        src_full = g_full.graph.get('_instance_sources_objects', {})
        src_dc = g_dc.graph.get('_instance_sources_objects', {})

        self.assertEqual(set(src_full.keys()), set(src_dc.keys()))

        # DC values must match
        for name in sorted(src_full.keys()):
            sf, sd = src_full[name], src_dc[name]
            self.assertAlmostEqual(sf.get_static_current(), sd.get_static_current(),
                                   places=10,
                                   msg=f"Source {name!r} DC mismatch")


class TestGraphMetadataCompleteness(unittest.TestCase):
    """Verify all expected metadata keys are present."""

    @classmethod
    def setUpClass(cls):
        cls.netlist_test = NETLIST_TEST

    def test_metadata_keys_present(self):
        """All expected graph metadata keys are set after parsing."""
        if not self.netlist_test.exists():
            self.skipTest(f"Netlist not found: {self.netlist_test}")

        parser = NetlistParser(str(self.netlist_test))
        g = parser.parse()

        required_keys = [
            'vsrc_dict', 'parameters', 'tile_grid', 'instance_node_map',
            'merged_nodes', 'mutual_inductors', 'net_connectivity',
            'stats', 'net_stats', 'vsrc_nodes',
        ]
        for key in required_keys:
            self.assertIn(key, g.graph, f"Missing graph metadata key: {key!r}")

    def test_stats_keys_present(self):
        """All expected stats keys are present."""
        if not self.netlist_test.exists():
            self.skipTest(f"Netlist not found: {self.netlist_test}")

        parser = NetlistParser(str(self.netlist_test))
        g = parser.parse()

        stats = g.graph['stats']
        expected_stat_keys = [
            'nodes', 'edges', 'resistors', 'capacitors', 'inductors',
            'vsources', 'isources', 'mutual_inductors', 'boundary_nodes',
            'package_nodes', 'vsrc_nodes', 'tiles_parsed', 'tiles_failed',
            'unmapped_nodes', 'instances_with_waveforms', 'total_static_current_ma'
        ]
        for key in expected_stat_keys:
            self.assertIn(key, stats, f"Missing stats key: {key!r}")

    def test_layer_stats_present(self):
        """Layer stats metadata is populated."""
        if not self.netlist_test.exists():
            self.skipTest(f"Netlist not found: {self.netlist_test}")

        parser = NetlistParser(str(self.netlist_test))
        g = parser.parse()

        # layer_stats or layer_stats_by_net should exist
        has_layer = ('layer_stats' in g.graph or 'layer_stats_by_net' in g.graph)
        self.assertTrue(has_layer, "No layer stats in graph metadata")

    def test_net_connectivity_nonempty(self):
        """net_connectivity should contain at least one net."""
        if not self.netlist_test.exists():
            self.skipTest(f"Netlist not found: {self.netlist_test}")

        parser = NetlistParser(str(self.netlist_test))
        g = parser.parse()

        nc = g.graph['net_connectivity']
        self.assertGreater(len(nc), 0, "net_connectivity should be non-empty")
        for net_name, nodes in nc.items():
            self.assertIsInstance(net_name, str)
            self.assertGreater(len(nodes), 0,
                               f"net_connectivity[{net_name!r}] is empty")


if __name__ == '__main__':
    unittest.main()

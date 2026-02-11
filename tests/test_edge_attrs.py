#!/usr/bin/env python3
"""Tests for memory-optimized edge attribute classes."""

import pickle
import sys
import unittest

from pdn.edge_attrs import (
    BaseElementEdge,
    ResistorEdge,
    ResistorEdgeWithName,
    CapacitorEdge,
    InductorEdge,
    VoltageSourceEdge,
    CurrentSourceEdge,
    create_element_edge,
    edge_dict_to_object,
    edge_object_to_dict,
    _get_net_type_index,
    _get_net_type_from_index,
    _pack_tile_net,
    _unpack_tile_id,
    _unpack_net_type_idx,
    reset_net_type_table,
)


class TestNetTypeIndexing(unittest.TestCase):
    """Test net type string to index conversion."""

    def setUp(self):
        reset_net_type_table()

    def tearDown(self):
        reset_net_type_table()

    def test_none_returns_zero(self):
        """None net type returns index 0."""
        self.assertEqual(_get_net_type_index(None), 0)
        self.assertIsNone(_get_net_type_from_index(0))

    def test_first_net_type_returns_one(self):
        """First net type returns index 1."""
        idx = _get_net_type_index('VDD')
        self.assertEqual(idx, 1)
        self.assertEqual(_get_net_type_from_index(idx), 'VDD')

    def test_same_net_type_returns_same_index(self):
        """Same net type always returns same index."""
        idx1 = _get_net_type_index('VDD')
        idx2 = _get_net_type_index('VDD')
        self.assertEqual(idx1, idx2)

    def test_different_net_types_get_different_indices(self):
        """Different net types get different indices."""
        idx_vdd = _get_net_type_index('VDD')
        idx_vss = _get_net_type_index('VSS')
        self.assertNotEqual(idx_vdd, idx_vss)

    def test_max_net_types_limit(self):
        """Exceeding max net types raises error."""
        # Use up indices 1-15 (index 0 is reserved for None)
        for i in range(15):
            _get_net_type_index(f'NET{i}')

        # 16th net type should raise
        with self.assertRaises(ValueError):
            _get_net_type_index('NET_TOO_MANY')


class TestTileNetPacking(unittest.TestCase):
    """Test tile_id and net_type packing into int32."""

    def setUp(self):
        reset_net_type_table()

    def tearDown(self):
        reset_net_type_table()

    def test_pack_unpack_tile_id(self):
        """Pack and unpack tile_id correctly."""
        tile_id = (5, 10)
        net_idx = _get_net_type_index('VDD')
        packed = _pack_tile_net(tile_id, net_idx)

        self.assertEqual(_unpack_tile_id(packed), tile_id)

    def test_pack_unpack_none_tile_id(self):
        """Pack and unpack None tile_id correctly."""
        tile_id = None
        net_idx = _get_net_type_index('VSS')
        packed = _pack_tile_net(tile_id, net_idx)

        self.assertIsNone(_unpack_tile_id(packed))

    def test_pack_unpack_net_type_idx(self):
        """Pack and unpack net_type_idx correctly."""
        tile_id = (100, 200)
        net_idx = _get_net_type_index('VDD')
        packed = _pack_tile_net(tile_id, net_idx)

        self.assertEqual(_unpack_net_type_idx(packed), net_idx)

    def test_large_tile_coordinates(self):
        """Large tile coordinates up to 16382 are handled correctly."""
        # 14-bit range is 0-16383, but 16383 is reserved for None marker
        # Maximum valid tile coordinate is 16382
        tile_id = (16382, 16382)
        net_idx = 5
        packed = _pack_tile_net(tile_id, net_idx)

        self.assertEqual(_unpack_tile_id(packed), tile_id)

    def test_tile_coordinates_at_marker_become_none(self):
        """Tile coordinates at marker value (16383, 16383) become None.

        This is a known limitation: tile (16383, 16383) is indistinguishable
        from None due to packing format. Real PDNs rarely exceed 1000 tiles.
        """
        # (16383, 16383) is the None marker
        tile_id = (16383, 16383)
        net_idx = 3
        packed = _pack_tile_net(tile_id, net_idx)

        # Will be interpreted as None
        self.assertIsNone(_unpack_tile_id(packed))


class TestResistorEdge(unittest.TestCase):
    """Test ResistorEdge (without elem_name)."""

    def setUp(self):
        reset_net_type_table()

    def tearDown(self):
        reset_net_type_table()

    def test_create_resistor_edge(self):
        """Create a die resistor edge."""
        edge = create_element_edge('R', 0.0125, tile_id=(0, 0), net_type='VDD')

        self.assertIsInstance(edge, ResistorEdge)
        self.assertEqual(edge.type, 'R')
        self.assertEqual(edge.value, 0.0125)
        self.assertEqual(edge.tile_id, (0, 0))
        self.assertEqual(edge.net_type, 'VDD')

    def test_dict_like_get(self):
        """Dict-like get() method works."""
        edge = create_element_edge('R', 0.5, tile_id=(1, 2), net_type='VSS')

        self.assertEqual(edge.get('type'), 'R')
        self.assertEqual(edge.get('value'), 0.5)
        self.assertEqual(edge.get('tile_id'), (1, 2))
        self.assertEqual(edge.get('net_type'), 'VSS')
        self.assertIsNone(edge.get('elem_name'))  # Not stored
        self.assertEqual(edge.get('elem_name', ''), '')  # Default works
        self.assertEqual(edge.get('nonexistent', 'default'), 'default')

    def test_dict_like_indexing(self):
        """Dict-like indexing works."""
        edge = create_element_edge('R', 0.5, tile_id=(1, 2), net_type='VDD')

        self.assertEqual(edge['type'], 'R')
        self.assertEqual(edge['value'], 0.5)

        with self.assertRaises(KeyError):
            _ = edge['elem_name']  # Not present in ResistorEdge

    def test_dict_like_contains(self):
        """Dict-like 'in' operator works."""
        edge = create_element_edge('R', 0.5, tile_id=(1, 2), net_type='VDD')

        self.assertIn('type', edge)
        self.assertIn('value', edge)
        self.assertIn('tile_id', edge)
        self.assertIn('net_type', edge)
        self.assertNotIn('elem_name', edge)  # Not present
        self.assertNotIn('nonexistent', edge)

    def test_keys_items_values(self):
        """Dict-like iteration methods work."""
        edge = create_element_edge('R', 0.5, tile_id=(1, 2), net_type='VDD')

        keys = edge.keys()
        self.assertEqual(set(keys), {'type', 'value', 'tile_id', 'net_type'})

        items = dict(edge.items())
        self.assertEqual(items['type'], 'R')
        self.assertEqual(items['value'], 0.5)

    def test_pickle_roundtrip(self):
        """Pickle and unpickle preserves values."""
        edge = create_element_edge('R', 0.0125, tile_id=(5, 10), net_type='VDD')

        pickled = pickle.dumps(edge)
        restored = pickle.loads(pickled)

        self.assertEqual(restored.type, 'R')
        self.assertEqual(restored.value, 0.0125)
        self.assertEqual(restored.tile_id, (5, 10))
        self.assertEqual(restored.net_type, 'VDD')


class TestResistorEdgeWithName(unittest.TestCase):
    """Test ResistorEdgeWithName (with elem_name for package resistors)."""

    def setUp(self):
        reset_net_type_table()

    def tearDown(self):
        reset_net_type_table()

    def test_create_resistor_edge_with_name(self):
        """Create a package resistor edge with elem_name."""
        edge = create_element_edge(
            'R', 0.0, elem_name='rs', tile_id=(0, 0),
            net_type='VDD', needs_elem_name=True
        )

        self.assertIsInstance(edge, ResistorEdgeWithName)
        self.assertEqual(edge.type, 'R')
        self.assertEqual(edge.value, 0.0)
        self.assertEqual(edge.elem_name, 'rs')
        self.assertEqual(edge.tile_id, (0, 0))
        self.assertEqual(edge.net_type, 'VDD')

    def test_elem_name_accessible(self):
        """elem_name is accessible via dict-like interface."""
        edge = create_element_edge(
            'R', 0.0, elem_name='rs', tile_id=(0, 0),
            net_type='VDD', needs_elem_name=True
        )

        self.assertEqual(edge.get('elem_name'), 'rs')
        self.assertEqual(edge['elem_name'], 'rs')
        self.assertIn('elem_name', edge)

    def test_pickle_roundtrip_with_name(self):
        """Pickle preserves elem_name."""
        edge = create_element_edge(
            'R', 0.0, elem_name='rs', tile_id=(5, 10),
            net_type='VDD', needs_elem_name=True
        )

        pickled = pickle.dumps(edge)
        restored = pickle.loads(pickled)

        self.assertEqual(restored.elem_name, 'rs')
        self.assertEqual(restored.tile_id, (5, 10))
        self.assertEqual(restored.net_type, 'VDD')


class TestCapacitorEdge(unittest.TestCase):
    """Test CapacitorEdge."""

    def setUp(self):
        reset_net_type_table()

    def tearDown(self):
        reset_net_type_table()

    def test_create_capacitor_edge(self):
        """Create a capacitor edge."""
        edge = create_element_edge('C', 100.0, tile_id=(0, 0), net_type='VDD')

        self.assertIsInstance(edge, CapacitorEdge)
        self.assertEqual(edge.type, 'C')
        self.assertEqual(edge.value, 100.0)  # fF
        self.assertEqual(edge.tile_id, (0, 0))
        self.assertEqual(edge.net_type, 'VDD')

    def test_pickle_roundtrip(self):
        """Pickle and unpickle preserves values."""
        edge = create_element_edge('C', 500.0, tile_id=(3, 4), net_type='VSS')

        pickled = pickle.dumps(edge)
        restored = pickle.loads(pickled)

        self.assertEqual(restored.type, 'C')
        self.assertEqual(restored.value, 500.0)
        self.assertEqual(restored.tile_id, (3, 4))


class TestInductorEdge(unittest.TestCase):
    """Test InductorEdge."""

    def setUp(self):
        reset_net_type_table()

    def tearDown(self):
        reset_net_type_table()

    def test_create_inductor_edge(self):
        """Create an inductor edge."""
        edge = create_element_edge('L', 0.1, tile_id=(0, 0), net_type='VDD')

        self.assertIsInstance(edge, InductorEdge)
        self.assertEqual(edge.type, 'L')
        self.assertEqual(edge.value, 0.1)  # nH

    def test_pickle_roundtrip(self):
        """Pickle and unpickle preserves values."""
        edge = create_element_edge('L', 0.5, tile_id=(2, 3), net_type='VDD')

        pickled = pickle.dumps(edge)
        restored = pickle.loads(pickled)

        self.assertEqual(restored.type, 'L')
        self.assertEqual(restored.value, 0.5)


class TestVoltageSourceEdge(unittest.TestCase):
    """Test VoltageSourceEdge."""

    def setUp(self):
        reset_net_type_table()

    def tearDown(self):
        reset_net_type_table()

    def test_create_voltage_source_edge(self):
        """Create a voltage source edge."""
        edge = create_element_edge('V', 0.75, elem_name='v_vdd', tile_id=None, net_type='VDD')

        self.assertIsInstance(edge, VoltageSourceEdge)
        self.assertEqual(edge.type, 'V')
        self.assertEqual(edge.value, 0.75)
        self.assertEqual(edge.elem_name, 'v_vdd')
        self.assertIsNone(edge.tile_id)
        self.assertEqual(edge.net_type, 'VDD')

    def test_elem_name_always_stored(self):
        """Voltage source always has elem_name."""
        edge = create_element_edge('V', 0.75, elem_name='vdd_src', net_type='VDD')

        self.assertEqual(edge.get('elem_name'), 'vdd_src')
        self.assertIn('elem_name', edge)

    def test_pickle_roundtrip(self):
        """Pickle and unpickle preserves values."""
        edge = create_element_edge('V', 1.0, elem_name='v_main', net_type='VDD')

        pickled = pickle.dumps(edge)
        restored = pickle.loads(pickled)

        self.assertEqual(restored.elem_name, 'v_main')
        self.assertEqual(restored.value, 1.0)


class TestCurrentSourceEdge(unittest.TestCase):
    """Test CurrentSourceEdge."""

    def setUp(self):
        reset_net_type_table()

    def tearDown(self):
        reset_net_type_table()

    def test_create_current_source_edge(self):
        """Create a current source edge."""
        edge = create_element_edge('I', 5.0, tile_id=(0, 0), net_type='VDD')

        self.assertIsInstance(edge, CurrentSourceEdge)
        self.assertEqual(edge.type, 'I')
        self.assertEqual(edge.value, 5.0)  # mA

    def test_pickle_roundtrip(self):
        """Pickle and unpickle preserves values."""
        edge = create_element_edge('I', 10.0, tile_id=(1, 1), net_type='VDD')

        pickled = pickle.dumps(edge)
        restored = pickle.loads(pickled)

        self.assertEqual(restored.type, 'I')
        self.assertEqual(restored.value, 10.0)


class TestEdgeDictConversion(unittest.TestCase):
    """Test conversion between edge dict and edge objects."""

    def setUp(self):
        reset_net_type_table()

    def tearDown(self):
        reset_net_type_table()

    def test_dict_to_object(self):
        """Convert dict to edge object."""
        edge_dict = {
            'type': 'R',
            'value': 0.5,
            'elem_name': 'r1',
            'tile_id': (0, 0),
            'net_type': 'VDD'
        }

        # Without needs_elem_name, creates ResistorEdge (no elem_name stored)
        edge = edge_dict_to_object(edge_dict)
        self.assertIsInstance(edge, ResistorEdge)
        self.assertEqual(edge.value, 0.5)
        self.assertIsNone(edge.get('elem_name'))

        # With needs_elem_name, creates ResistorEdgeWithName
        edge_with_name = edge_dict_to_object(edge_dict, needs_elem_name=True)
        self.assertIsInstance(edge_with_name, ResistorEdgeWithName)
        self.assertEqual(edge_with_name.elem_name, 'r1')

    def test_object_to_dict(self):
        """Convert edge object to dict."""
        edge = create_element_edge(
            'R', 0.5, elem_name='rs', tile_id=(0, 0),
            net_type='VDD', needs_elem_name=True
        )

        edge_dict = edge_object_to_dict(edge)

        self.assertEqual(edge_dict['type'], 'R')
        self.assertEqual(edge_dict['value'], 0.5)
        self.assertEqual(edge_dict['elem_name'], 'rs')
        self.assertEqual(edge_dict['tile_id'], (0, 0))
        self.assertEqual(edge_dict['net_type'], 'VDD')

    def test_roundtrip_via_dict(self):
        """Convert object -> dict -> object preserves data."""
        original = create_element_edge(
            'V', 0.75, elem_name='vdd_src', tile_id=(5, 10), net_type='VDD'
        )

        edge_dict = edge_object_to_dict(original)
        restored = edge_dict_to_object(edge_dict, needs_elem_name=True)

        self.assertEqual(restored.type, original.type)
        self.assertEqual(restored.value, original.value)
        self.assertEqual(restored.tile_id, original.tile_id)
        self.assertEqual(restored.net_type, original.net_type)


class TestMemorySavings(unittest.TestCase):
    """Test that edge objects use less memory than dicts."""

    def setUp(self):
        reset_net_type_table()

    def tearDown(self):
        reset_net_type_table()

    def test_resistor_edge_smaller_than_dict(self):
        """ResistorEdge uses less memory than equivalent dict."""
        # Create equivalent dict
        edge_dict = {
            'type': 'R',
            'value': 0.0125,
            'tile_id': (0, 0),
            'net_type': 'VDD'
        }

        # Create edge object
        edge_obj = create_element_edge('R', 0.0125, tile_id=(0, 0), net_type='VDD')

        # Get sizes
        dict_size = sys.getsizeof(edge_dict)
        obj_size = sys.getsizeof(edge_obj)

        # Object should be significantly smaller (dict overhead is ~200+ bytes)
        # Note: This doesn't count the string values in the dict, which add more
        self.assertLess(obj_size, dict_size)

    def test_no_dict_attribute(self):
        """Edge objects don't have __dict__ (slots optimization)."""
        edge = create_element_edge('R', 0.5, tile_id=(0, 0), net_type='VDD')

        # Slotted classes don't have __dict__
        self.assertFalse(hasattr(edge, '__dict__'))


class TestFactoryFunctionEdgeCases(unittest.TestCase):
    """Test edge cases in create_element_edge factory."""

    def setUp(self):
        reset_net_type_table()

    def tearDown(self):
        reset_net_type_table()

    def test_unknown_element_type_raises(self):
        """Unknown element type raises ValueError."""
        with self.assertRaises(ValueError):
            create_element_edge('X', 1.0)

    def test_voltage_source_without_elem_name(self):
        """Voltage source without elem_name gets empty string."""
        edge = create_element_edge('V', 0.75, net_type='VDD')

        self.assertEqual(edge.elem_name, '')

    def test_resistor_needs_elem_name_false(self):
        """Resistor with needs_elem_name=False creates ResistorEdge."""
        edge = create_element_edge(
            'R', 0.5, elem_name='r1', tile_id=(0, 0),
            net_type='VDD', needs_elem_name=False
        )

        self.assertIsInstance(edge, ResistorEdge)
        self.assertIsNone(edge.get('elem_name'))

    def test_none_net_type(self):
        """None net_type is handled correctly."""
        edge = create_element_edge('R', 0.5, tile_id=(0, 0), net_type=None)

        self.assertIsNone(edge.net_type)

    def test_none_tile_id(self):
        """None tile_id is handled correctly."""
        edge = create_element_edge('R', 0.5, tile_id=None, net_type='VDD')

        self.assertIsNone(edge.tile_id)

    def test_precomputed_net_type_idx(self):
        """Pre-computed net_type_idx overrides net_type."""
        # First, register a net type
        idx = _get_net_type_index('VDD')

        # Create edge with pre-computed index
        edge = create_element_edge('R', 0.5, net_type='IGNORED', net_type_idx=idx)

        # Should use the pre-computed index, not 'IGNORED'
        self.assertEqual(edge.net_type, 'VDD')


if __name__ == '__main__':
    unittest.main()

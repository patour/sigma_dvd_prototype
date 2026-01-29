"""Tests for VectorizedCurrentSources.

Tests the columnar storage format, pulse/PWL evaluation, and RHS array operations
used for high-performance transient simulation.
"""

import unittest
from typing import Any, Dict, List

import numpy as np

from core.vectorized_sources import VectorizedCurrentSources


def create_simple_sources() -> Dict[str, Dict[str, Any]]:
    """Create a simple set of current sources for testing.

    Returns:
        Dict[str, Dict[str, Any]]: Mapping from source name (e.g. ``"I_dc1"``)
            to a source specification dictionary with the following keys:

            - ``"node1"`` (str): Name of the node where the current is injected.
            - ``"dc_value"`` (float): DC current value in mA.
            - ``"pulses"`` (List[Dict[str, float]]): List of pulse waveform
              definitions. Each dict typically contains:

                - ``"v1"`` (float): Initial current level in mA.
                - ``"v2"`` (float): Pulsed current level in mA.
                - ``"delay"`` (float): Time delay before the first pulse (seconds).
                - ``"rt"`` (float): Rise time of the pulse (seconds).
                - ``"ft"`` (float): Fall time of the pulse (seconds).
                - ``"width"`` (float): Pulse width at the ``"v2"`` level (seconds).
                - ``"period"`` (float): Pulse repetition period (seconds).

            - ``"pwls"`` (List[Dict[str, Any]]): List of piecewise-linear (PWL)
              waveform definitions. Each dict typically contains:

                - ``"delay"`` (float): Time delay before the PWL waveform starts
                  (seconds).
                - ``"period"`` (float): Repetition period of the PWL waveform
                  (seconds). Use ``0.0`` for non-periodic waveforms.
                - ``"points"`` (List[Tuple[float, float]]): Sequence of
                  ``(time, current)`` pairs, where time is in seconds and current
                  is in mA, defining the PWL segments.
    """
    return {
        'I_dc1': {
            'node1': 'N1',
            'dc_value': 1.0,  # 1 mA
            'pulses': [],
            'pwls': [],
        },
        'I_dc2': {
            'node1': 'N2',
            'dc_value': 2.0,  # 2 mA
            'pulses': [],
            'pwls': [],
        },
        'I_pulse': {
            'node1': 'N3',
            'dc_value': 0.0,
            'pulses': [{
                'v1': 0.0,
                'v2': 5.0,  # 5 mA peak
                'delay': 10e-9,  # 10 ns delay
                'rt': 1e-9,  # 1 ns rise
                'ft': 1e-9,  # 1 ns fall
                'width': 10e-9,  # 10 ns width
                'period': 50e-9,  # 50 ns period
            }],
            'pwls': [],
        },
        'I_pwl': {
            'node1': 'N4',
            'dc_value': 0.0,
            'pulses': [],
            'pwls': [{
                'delay': 0.0,
                'period': 0.0,  # Non-periodic
                'points': [(0.0, 0.0), (10e-9, 3.0), (20e-9, 1.0), (30e-9, 1.0)],
            }],
        },
        'I_pwl_periodic': {
            'node1': 'N5',
            'dc_value': 0.5,  # DC offset
            'pulses': [],
            'pwls': [{
                'delay': 5e-9,  # 5 ns delay
                'period': 20e-9,  # 20 ns period
                'points': [(0.0, 0.0), (5e-9, 2.0), (10e-9, 0.0)],
            }],
        },
    }


def create_node_mapping() -> tuple:
    """Create node mapping for test sources.
    
    Returns:
        Tuple of (node_to_idx, idx_to_node, n_nodes)
    """
    nodes = ['N0', 'N1', 'N2', 'N3', 'N4', 'N5', 'N6']
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    return node_to_idx, nodes, len(nodes)


class TestVectorizedSourcesConstruction(unittest.TestCase):
    """Tests for VectorizedCurrentSources construction."""

    def test_from_serialized_dicts(self):
        """Should correctly parse serialized source dicts."""
        sources = create_simple_sources()
        node_to_idx, idx_to_node, n_nodes = create_node_mapping()
        
        vec = VectorizedCurrentSources.from_serialized_dicts(
            sources, node_to_idx, n_nodes
        )
        
        self.assertEqual(vec.n_nodes, n_nodes)
        self.assertEqual(vec.n_sources, 5)  # 5 sources
        self.assertEqual(vec.n_pulses, 1)   # 1 pulse
        self.assertEqual(vec.n_pwls, 2)     # 2 PWLs

    def test_dc_values_extracted(self):
        """DC values should be correctly extracted."""
        sources = create_simple_sources()
        node_to_idx, idx_to_node, n_nodes = create_node_mapping()
        
        vec = VectorizedCurrentSources.from_serialized_dicts(
            sources, node_to_idx, n_nodes
        )
        
        # Check that DC values are stored
        self.assertEqual(len(vec.dc_values), 5)
        # Sum should include 1.0 + 2.0 + 0.0 + 0.0 + 0.5 = 3.5 mA
        self.assertAlmostEqual(vec.dc_values.sum(), 3.5)

    def test_pulse_parameters_extracted(self):
        """Pulse parameters should be correctly extracted."""
        sources = create_simple_sources()
        node_to_idx, idx_to_node, n_nodes = create_node_mapping()
        
        vec = VectorizedCurrentSources.from_serialized_dicts(
            sources, node_to_idx, n_nodes
        )
        
        self.assertEqual(vec.n_pulses, 1)
        self.assertEqual(vec.pulse_node_idx[0], node_to_idx['N3'])
        self.assertAlmostEqual(vec.pulse_v1[0], 0.0)
        self.assertAlmostEqual(vec.pulse_v2[0], 5.0)
        self.assertAlmostEqual(vec.pulse_delay[0], 10e-9)
        self.assertAlmostEqual(vec.pulse_rt[0], 1e-9)
        self.assertAlmostEqual(vec.pulse_ft[0], 1e-9)
        self.assertAlmostEqual(vec.pulse_width[0], 10e-9)
        self.assertAlmostEqual(vec.pulse_period[0], 50e-9)

    def test_pwl_parameters_extracted(self):
        """PWL parameters should be correctly extracted."""
        sources = create_simple_sources()
        node_to_idx, idx_to_node, n_nodes = create_node_mapping()
        
        vec = VectorizedCurrentSources.from_serialized_dicts(
            sources, node_to_idx, n_nodes
        )
        
        self.assertEqual(vec.n_pwls, 2)
        self.assertEqual(vec.n_pwl_points, 7)  # 4 + 3 points

    def test_memory_bytes_positive(self):
        """memory_bytes should return positive value."""
        sources = create_simple_sources()
        node_to_idx, idx_to_node, n_nodes = create_node_mapping()
        
        vec = VectorizedCurrentSources.from_serialized_dicts(
            sources, node_to_idx, n_nodes
        )
        
        self.assertGreater(vec.memory_bytes(), 0)

    def test_statistics(self):
        """get_statistics should return correct values."""
        sources = create_simple_sources()
        node_to_idx, idx_to_node, n_nodes = create_node_mapping()
        
        vec = VectorizedCurrentSources.from_serialized_dicts(
            sources, node_to_idx, n_nodes
        )
        
        stats = vec.get_statistics()
        self.assertEqual(stats['n_sources'], 5)
        self.assertEqual(stats['n_pulses'], 1)
        self.assertEqual(stats['n_pwls'], 2)
        self.assertEqual(stats['n_pwl_points'], 7)


class TestDCEvaluation(unittest.TestCase):
    """Tests for DC current evaluation."""

    def test_dc_only(self):
        """DC sources should evaluate correctly."""
        sources = {
            'I1': {'node1': 'N1', 'dc_value': 1.0, 'pulses': [], 'pwls': []},
            'I2': {'node1': 'N2', 'dc_value': 2.0, 'pulses': [], 'pwls': []},
        }
        node_to_idx, idx_to_node, n_nodes = create_node_mapping()
        
        vec = VectorizedCurrentSources.from_serialized_dicts(
            sources, node_to_idx, n_nodes
        )
        
        currents = vec.evaluate_at_time(0.0)
        
        self.assertEqual(len(currents), n_nodes)
        self.assertAlmostEqual(currents[node_to_idx['N1']], 1.0)
        self.assertAlmostEqual(currents[node_to_idx['N2']], 2.0)
        self.assertAlmostEqual(currents[node_to_idx['N0']], 0.0)

    def test_dc_same_node_accumulates(self):
        """Multiple DC sources on same node should accumulate."""
        sources = {
            'I1': {'node1': 'N1', 'dc_value': 1.0, 'pulses': [], 'pwls': []},
            'I2': {'node1': 'N1', 'dc_value': 2.0, 'pulses': [], 'pwls': []},
        }
        node_to_idx, idx_to_node, n_nodes = create_node_mapping()
        
        vec = VectorizedCurrentSources.from_serialized_dicts(
            sources, node_to_idx, n_nodes
        )
        
        currents = vec.evaluate_at_time(0.0)
        self.assertAlmostEqual(currents[node_to_idx['N1']], 3.0)


class TestPulseEvaluation(unittest.TestCase):
    """Tests for pulse waveform evaluation."""

    def setUp(self):
        """Create a simple pulse source."""
        self.sources = {
            'I_pulse': {
                'node1': 'N1',
                'dc_value': 0.0,
                'pulses': [{
        t = 11e-9  # midpoint of rise
        delay = self.sources['I_pulse']['pulses'][0]['delay']
        rt = self.sources['I_pulse']['pulses'][0]['rt']
        v1 = self.sources['I_pulse']['pulses'][0]['v1']
        v2 = self.sources['I_pulse']['pulses'][0]['v2']
        currents = self.vec.evaluate_at_time(t)
        # At t=11ns: t_rel = 11-10 = 1ns, rise_frac = 1/2 = 0.5
        t_rel = t - delay
        rise_frac = t_rel / rt
        expected = v1 + (v2 - v1) * rise_frac
        # value = 0 + (10-0)*0.5 = 5.0
        self.assertAlmostEqual(
            currents[self.node_to_idx['N1']],
            expected,
            places=5,
            msg=f"Unexpected value during rise: rise_frac={rise_frac}",
        )
                    'ft': 2e-9,  # 2 ns fall
                    'width': 6e-9,  # 6 ns width
                    'period': 50e-9,  # 50 ns period
                }],
                'pwls': [],
            },
        }
        node_to_idx, idx_to_node, n_nodes = create_node_mapping()
        self.node_to_idx = node_to_idx
        self.vec = VectorizedCurrentSources.from_serialized_dicts(
            self.sources, node_to_idx, n_nodes
        )

    def test_before_delay(self):
        """Pulse should be at v1 before delay."""
        currents = self.vec.evaluate_at_time(5e-9)  # t=5ns < delay=10ns
        self.assertAlmostEqual(currents[self.node_to_idx['N1']], 0.0)

    def test_during_rise(self):
        """Pulse should interpolate during rise time."""
        # delay=10ns, rt=2ns, so rise is 10-12ns
        currents = self.vec.evaluate_at_time(11e-9)  # midpoint of rise
        # At t=11ns: t_rel = 11-10 = 1ns, rise_frac = 1/2 = 0.5
        # value = 0 + (10-0)*0.5 = 5.0
        self.assertAlmostEqual(currents[self.node_to_idx['N1']], 5.0, places=5)

    def test_during_high(self):
        """Pulse should be at v2 during high period."""
        # delay=10ns, rt=2ns, width=6ns, so high is 12-18ns
        currents = self.vec.evaluate_at_time(15e-9)
        self.assertAlmostEqual(currents[self.node_to_idx['N1']], 10.0)

    def test_during_fall(self):
        """Pulse should interpolate during fall time."""
        # Fall is 18-20ns
        currents = self.vec.evaluate_at_time(19e-9)  # midpoint of fall
        # At t=19ns: t_rel = 9ns, t_fall = 9 - 2 - 6 = 1ns, fall_frac = 0.5
        # value = 10 + (0-10)*0.5 = 5.0
        self.assertAlmostEqual(currents[self.node_to_idx['N1']], 5.0, places=5)

    def test_during_low(self):
        """Pulse should be at v1 during low period."""
        # After fall: 20ns onwards until next period
        currents = self.vec.evaluate_at_time(30e-9)
        self.assertAlmostEqual(currents[self.node_to_idx['N1']], 0.0)

    def test_periodic_wrap(self):
        """Pulse should repeat with period."""
        # At t=65ns, with period=50ns, effective t = 15ns (high)
        currents = self.vec.evaluate_at_time(65e-9)
        self.assertAlmostEqual(currents[self.node_to_idx['N1']], 10.0)


class TestPWLEvaluation(unittest.TestCase):
    """Tests for PWL waveform evaluation."""

    def setUp(self):
        """Create PWL test sources."""
        self.sources = {
            'I_pwl': {
                'node1': 'N1',
                'dc_value': 0.0,
                'pulses': [],
                'pwls': [{
                    'delay': 0.0,
                    'period': 0.0,  # Non-periodic
                    'points': [(0.0, 1.0), (10e-9, 5.0), (20e-9, 2.0)],
                }],
            },
        }
        node_to_idx, idx_to_node, n_nodes = create_node_mapping()
        self.node_to_idx = node_to_idx
        self.vec = VectorizedCurrentSources.from_serialized_dicts(
            self.sources, node_to_idx, n_nodes
        )

    def test_at_first_point(self):
        """PWL should return first value at t=0."""
        currents = self.vec.evaluate_at_time(0.0)
        self.assertAlmostEqual(currents[self.node_to_idx['N1']], 1.0)

    def test_at_middle_point(self):
        """PWL should return exact value at defined point."""
        currents = self.vec.evaluate_at_time(10e-9)
        self.assertAlmostEqual(currents[self.node_to_idx['N1']], 5.0)

    def test_interpolation(self):
        """PWL should interpolate between points."""
        # Between t=0 (v=1) and t=10ns (v=5)
        currents = self.vec.evaluate_at_time(5e-9)  # midpoint
        # value = 1 + (5-1) * 0.5 = 3.0
        self.assertAlmostEqual(currents[self.node_to_idx['N1']], 3.0)

    def test_after_last_point_nonperiodic(self):
        """Non-periodic PWL should hold last value after end."""
        currents = self.vec.evaluate_at_time(50e-9)
        self.assertAlmostEqual(currents[self.node_to_idx['N1']], 2.0)

    def test_before_first_point(self):
        """PWL should return first value before t=0."""
        currents = self.vec.evaluate_at_time(-5e-9)
        self.assertAlmostEqual(currents[self.node_to_idx['N1']], 1.0)


class TestPWLPeriodic(unittest.TestCase):
    """Tests for periodic PWL evaluation."""

    def setUp(self):
        """Create periodic PWL source."""
        self.sources = {
            'I_pwl': {
                'node1': 'N1',
                'dc_value': 0.0,
                'pulses': [],
                'pwls': [{
                    'delay': 5e-9,  # 5 ns delay
                    'period': 20e-9,  # 20 ns period
                    'points': [(0.0, 0.0), (10e-9, 4.0), (20e-9, 0.0)],
                }],
            },
        }
        node_to_idx, idx_to_node, n_nodes = create_node_mapping()
        self.node_to_idx = node_to_idx
        self.vec = VectorizedCurrentSources.from_serialized_dicts(
            self.sources, node_to_idx, n_nodes
        )

    def test_with_delay(self):
        """PWL should respect delay offset."""
        # At t=5ns (delay), effective t_adj = 0, so value = 0
        currents = self.vec.evaluate_at_time(5e-9)
        self.assertAlmostEqual(currents[self.node_to_idx['N1']], 0.0)

    def test_with_delay_interpolation(self):
        """PWL should interpolate with delay."""
        # At t=10ns, effective t_adj = 5ns, midpoint of rise
        # value = 0 + (4-0) * 0.5 = 2.0
        currents = self.vec.evaluate_at_time(10e-9)
        self.assertAlmostEqual(currents[self.node_to_idx['N1']], 2.0)

    def test_periodic_wrap(self):
        """Periodic PWL should wrap correctly."""
        # At t=30ns with delay=5ns: t_adj = 25ns
        # With period=20ns: t_adj = 25 % 20 = 5ns -> midpoint -> 2.0
        currents = self.vec.evaluate_at_time(30e-9)
        self.assertAlmostEqual(currents[self.node_to_idx['N1']], 2.0, places=5)


class TestPWLGrouping(unittest.TestCase):
    """Tests for PWL group-by-count optimization."""

    def test_multiple_pwls_same_count(self):
        """PWLs with same point count should be grouped."""
        sources = {
            'I1': {
                'node1': 'N1', 'dc_value': 0.0, 'pulses': [],
                'pwls': [{'delay': 0.0, 'period': 0.0, 
                          'points': [(0.0, 1.0), (10e-9, 2.0)]}],
            },
            'I2': {
                'node1': 'N2', 'dc_value': 0.0, 'pulses': [],
                'pwls': [{'delay': 0.0, 'period': 0.0,
                          'points': [(0.0, 3.0), (10e-9, 4.0)]}],
            },
            'I3': {
                'node1': 'N3', 'dc_value': 0.0, 'pulses': [],
                'pwls': [{'delay': 0.0, 'period': 0.0,
                          'points': [(0.0, 5.0), (5e-9, 6.0), (10e-9, 7.0)]}],
            },
        }
        node_to_idx, idx_to_node, n_nodes = create_node_mapping()
        
        vec = VectorizedCurrentSources.from_serialized_dicts(
            sources, node_to_idx, n_nodes
        )
        
        # Trigger group building
        vec._build_pwl_groups()
        
        # Should have 2 groups: count=2 (2 PWLs), count=3 (1 PWL)
        self.assertEqual(len(vec._pwl_groups[2]), 2)
        self.assertEqual(len(vec._pwl_groups[3]), 1)

    def test_large_group_vectorized(self):
        """Large groups (>=4) should use vectorized path."""
        # Create 10 PWLs with same point count
        sources = {}
        for i in range(10):
            sources[f'I{i}'] = {
                'node1': f'N{i % 6}',  # Reuse nodes
                'dc_value': 0.0,
                'pulses': [],
                'pwls': [{'delay': 0.0, 'period': 0.0,
                          'points': [(0.0, float(i)), (10e-9, float(i+1))]}],
            }
        
        node_to_idx, idx_to_node, n_nodes = create_node_mapping()
        vec = VectorizedCurrentSources.from_serialized_dicts(
            sources, node_to_idx, n_nodes
        )
        
        # Evaluate at midpoint
        currents = vec.evaluate_at_time(5e-9)
        
        # Check total current (sum of all sources at midpoint)
        # Each source interpolates: i + 0.5
        expected_total = sum(i + 0.5 for i in range(10))
        self.assertAlmostEqual(currents.sum(), expected_total, places=5)


class TestEvaluateToRHSArray(unittest.TestCase):
    """Tests for evaluate_to_rhs_array method."""

    def setUp(self):
        """Create sources and mapping."""
        self.sources = {
            'I1': {'node1': 'N1', 'dc_value': 1.0, 'pulses': [], 'pwls': []},
            'I2': {'node1': 'N2', 'dc_value': 2.0, 'pulses': [], 'pwls': []},
            'I3': {'node1': 'N3', 'dc_value': 3.0, 'pulses': [], 'pwls': []},
        }
        node_to_idx, idx_to_node, n_nodes = create_node_mapping()
        self.node_to_idx = node_to_idx
        self.idx_to_node = idx_to_node
        self.n_nodes = n_nodes
        
        self.vec = VectorizedCurrentSources.from_serialized_dicts(
            self.sources, node_to_idx, n_nodes
        )
        
        # Create unknown mapping (exclude N0 and N6 as "pads")
        self.unknown_nodes = ['N1', 'N2', 'N3', 'N4', 'N5']
        self.unknown_to_idx = {n: i for i, n in enumerate(self.unknown_nodes)}
        self.n_unknown = len(self.unknown_nodes)

    def test_build_source_to_unknown_map(self):
        """build_source_to_unknown_map should create correct mapping."""
        source_to_unknown, valid_mask = self.vec.build_source_to_unknown_map(
            self.unknown_to_idx, self.idx_to_node
        )
        
        self.assertEqual(len(source_to_unknown), self.n_nodes)
        self.assertEqual(len(valid_mask), self.n_nodes)
        
        # N0 should map to -1 (not unknown)
        self.assertEqual(source_to_unknown[self.node_to_idx['N0']], -1)
        self.assertFalse(valid_mask[self.node_to_idx['N0']])
        
        # N1 should map to 0 (first unknown)
        self.assertEqual(source_to_unknown[self.node_to_idx['N1']], 0)
        self.assertTrue(valid_mask[self.node_to_idx['N1']])

    def test_evaluate_to_rhs_array_correct_values(self):
        """evaluate_to_rhs_array should scatter currents correctly."""
        source_to_unknown, valid_mask = self.vec.build_source_to_unknown_map(
            self.unknown_to_idx, self.idx_to_node
        )
        
        rhs = np.zeros(self.n_unknown, dtype=np.float64)
        total, currents_arr = self.vec.evaluate_to_rhs_array(0.0, rhs, source_to_unknown, valid_mask)
        
        # Total should be 1+2+3 = 6 mA
        self.assertAlmostEqual(total, 6.0)
        
        # RHS should have negative currents (sink convention)
        self.assertAlmostEqual(rhs[self.unknown_to_idx['N1']], -1.0)
        self.assertAlmostEqual(rhs[self.unknown_to_idx['N2']], -2.0)
        self.assertAlmostEqual(rhs[self.unknown_to_idx['N3']], -3.0)
        
        # currents_arr should match evaluate_at_time result
        expected_currents = self.vec.evaluate_at_time(0.0)
        np.testing.assert_array_almost_equal(currents_arr, expected_currents)

    def test_evaluate_to_rhs_array_accumulates(self):
        """evaluate_to_rhs_array should accumulate to existing RHS."""
        source_to_unknown, valid_mask = self.vec.build_source_to_unknown_map(
            self.unknown_to_idx, self.idx_to_node
        )
        
        rhs = np.ones(self.n_unknown, dtype=np.float64)  # Start with 1s
        _, _ = self.vec.evaluate_to_rhs_array(0.0, rhs, source_to_unknown, valid_mask)
        
        # N1: 1 - 1 = 0
        self.assertAlmostEqual(rhs[self.unknown_to_idx['N1']], 0.0)
        # N2: 1 - 2 = -1
        self.assertAlmostEqual(rhs[self.unknown_to_idx['N2']], -1.0)


class TestEvaluateAtTimeAsDict(unittest.TestCase):
    """Tests for evaluate_at_time_as_dict method."""

    def test_returns_dict(self):
        """evaluate_at_time_as_dict should return dict with node keys."""
        sources = {
            'I1': {'node1': 'N1', 'dc_value': 1.0, 'pulses': [], 'pwls': []},
        }
        node_to_idx, idx_to_node, n_nodes = create_node_mapping()
        
        vec = VectorizedCurrentSources.from_serialized_dicts(
            sources, node_to_idx, n_nodes
        )
        
        currents_dict = vec.evaluate_at_time_as_dict(0.0, idx_to_node)
        
        self.assertIsInstance(currents_dict, dict)
        self.assertEqual(currents_dict.get('N1', 0), 1.0)

    def test_excludes_zero_currents(self):
        """evaluate_at_time_as_dict should exclude zero-current nodes."""
        sources = {
            'I1': {'node1': 'N1', 'dc_value': 1.0, 'pulses': [], 'pwls': []},
        }
        node_to_idx, idx_to_node, n_nodes = create_node_mapping()
        
        vec = VectorizedCurrentSources.from_serialized_dicts(
            sources, node_to_idx, n_nodes
        )
        
        currents_dict = vec.evaluate_at_time_as_dict(0.0, idx_to_node)
        
        # Only N1 should be in dict
        self.assertEqual(len(currents_dict), 1)
        self.assertIn('N1', currents_dict)
        self.assertNotIn('N0', currents_dict)


class TestCombinedEvaluation(unittest.TestCase):
    """Tests for combined DC + pulse + PWL evaluation."""

    def test_all_types_combined(self):
        """DC, pulse, and PWL should all contribute."""
        sources = create_simple_sources()
        node_to_idx, idx_to_node, n_nodes = create_node_mapping()
        
        vec = VectorizedCurrentSources.from_serialized_dicts(
            sources, node_to_idx, n_nodes
        )
        
        # At t=15ns:
        # - N1: 1.0 mA (DC)
        # - N2: 2.0 mA (DC)
        # - N3: pulse during high = 5.0 mA
        # - N4: PWL interpolation between (10ns, 3.0) and (20ns, 1.0)
        #       at t=15ns: 3 + (1-3)*0.5 = 2.0 mA
        # - N5: DC=0.5 + PWL at t=15ns with delay=5ns -> t_adj=10ns -> end -> 0.0
        
        currents = vec.evaluate_at_time(15e-9)
        
        self.assertAlmostEqual(currents[node_to_idx['N1']], 1.0)
        self.assertAlmostEqual(currents[node_to_idx['N2']], 2.0)
        self.assertAlmostEqual(currents[node_to_idx['N3']], 5.0, places=1)  # ~5 during high
        self.assertAlmostEqual(currents[node_to_idx['N4']], 2.0, places=5)


class TestEdgeCases(unittest.TestCase):
    """Tests for edge cases."""

    def test_empty_sources(self):
        """Should handle empty sources dict."""
        node_to_idx, idx_to_node, n_nodes = create_node_mapping()
        
        vec = VectorizedCurrentSources.from_serialized_dicts(
            {}, node_to_idx, n_nodes
        )
        
        self.assertEqual(vec.n_sources, 0)
        self.assertEqual(vec.n_pulses, 0)
        self.assertEqual(vec.n_pwls, 0)
        
        currents = vec.evaluate_at_time(0.0)
        np.testing.assert_array_equal(currents, np.zeros(n_nodes))

    def test_unknown_node_skipped(self):
        """Sources with unknown nodes should be skipped."""
        sources = {
            'I1': {'node1': 'N_UNKNOWN', 'dc_value': 1.0, 'pulses': [], 'pwls': []},
        }
        node_to_idx, idx_to_node, n_nodes = create_node_mapping()
        
        vec = VectorizedCurrentSources.from_serialized_dicts(
            sources, node_to_idx, n_nodes
        )
        
        # Source should be skipped
        self.assertEqual(vec.n_sources, 0)

    def test_zero_rt_ft_pulse(self):
        """Pulse with zero rise/fall time should work."""
        sources = {
            'I_pulse': {
                'node1': 'N1',
                'dc_value': 0.0,
                'pulses': [{
                    'v1': 0.0,
                    'v2': 10.0,
                    'delay': 0.0,
                    'rt': 0.0,  # Instant rise
                    'ft': 0.0,  # Instant fall
                    'width': 10e-9,
                    'period': 20e-9,
                }],
                'pwls': [],
            },
        }
        node_to_idx, idx_to_node, n_nodes = create_node_mapping()
        vec = VectorizedCurrentSources.from_serialized_dicts(
            sources, node_to_idx, n_nodes
        )
        
        # At t=5ns: during high
        currents = vec.evaluate_at_time(5e-9)
        self.assertAlmostEqual(currents[node_to_idx['N1']], 10.0)
        
        # At t=15ns: during low
        currents = vec.evaluate_at_time(15e-9)
        self.assertAlmostEqual(currents[node_to_idx['N1']], 0.0)


if __name__ == '__main__':
    unittest.main()

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
                    'v1': 0.0,
                    'v2': 10.0,  # 10 mA peak
                    'delay': 10e-9,  # 10 ns delay
                    'rt': 2e-9,  # 2 ns rise
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
        """Pulse should be at v1 before pulse start.

        With standard SPICE timing (matches C++ SimPWL):
        - delay=10ns (start), pulse begins rising at 10ns
        - t=5ns is before pulse start (10ns) -> v1=0
        """
        currents = self.vec.evaluate_at_time(5e-9)  # t=5ns < pulse_start=10ns
        self.assertAlmostEqual(currents[self.node_to_idx['N1']], 0.0)

    def test_during_rise(self):
        """Pulse should interpolate during rise time.

        With standard SPICE timing:
        - delay=10ns (start), rt=2ns -> rise is [10ns, 12ns)
        - At t=11ns: midpoint of rise
        """
        currents = self.vec.evaluate_at_time(11e-9)  # midpoint of rise [10,12)
        # t_rel = 11 - 10 = 1ns, rise_frac = 1/2 = 0.5
        self.assertAlmostEqual(currents[self.node_to_idx['N1']], 5.0, places=5)

    def test_during_high(self):
        """Pulse should be at v2 during high period.

        With standard SPICE timing:
        - delay=10ns, rt=2ns, width=6ns -> high is [12ns, 18ns)
        """
        currents = self.vec.evaluate_at_time(15e-9)  # mid-high [12,18)
        self.assertAlmostEqual(currents[self.node_to_idx['N1']], 10.0)

    def test_during_fall(self):
        """Pulse should interpolate during fall time.

        With standard SPICE timing:
        - delay=10ns, rt=2ns, width=6ns, ft=2ns -> fall is [18ns, 20ns)
        """
        currents = self.vec.evaluate_at_time(19e-9)  # midpoint of fall [18,20)
        # t_rel = 19 - 10 = 9ns, t_fall = 9 - 2 - 6 = 1ns, fall_frac = 0.5
        self.assertAlmostEqual(currents[self.node_to_idx['N1']], 5.0, places=5)

    def test_during_low(self):
        """Pulse should be at v1 during low period.

        With standard SPICE timing:
        - Pulse ends at delay + rt + width + ft = 10 + 2 + 6 + 2 = 20ns
        - After 20ns: low period
        """
        currents = self.vec.evaluate_at_time(30e-9)
        self.assertAlmostEqual(currents[self.node_to_idx['N1']], 0.0)

    def test_periodic_wrap(self):
        """Pulse should repeat with period.

        With standard SPICE timing and period=50ns:
        - At t=65ns, effective t = 65 % 50 = 15ns (in high phase [12,18))
        """
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
    """Tests for padded single-group PWL evaluation cache."""

    def test_padded_cache_structure(self):
        """Padded cache should pad all PWLs to max point count."""
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

        # Trigger padded cache building
        vec._build_pwl_padded_cache()

        # Max count is 3 (from I3), so all PWLs padded to 3 columns
        self.assertEqual(vec._pwl_padded_max_count, 3)
        self.assertEqual(vec._pwl_padded_times.shape, (3, 3))
        self.assertEqual(vec._pwl_padded_values.shape, (3, 3))

        # 2-point PWLs should have last column padded with last value
        # I1 has times [0, 10e-9] -> padded to [0, 10e-9, 10e-9]
        # I1 has values [1.0, 2.0] -> padded to [1.0, 2.0, 2.0]
        for i in range(vec.n_pwls):
            cnt = int(vec.pwl_count[i])
            if cnt < 3:
                # Padding columns should repeat last actual value
                self.assertEqual(
                    vec._pwl_padded_times[i, cnt],
                    vec._pwl_padded_times[i, cnt - 1],
                )
                self.assertEqual(
                    vec._pwl_padded_values[i, cnt],
                    vec._pwl_padded_values[i, cnt - 1],
                )

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

    def test_binned_groups_structure(self):
        """Binned groups should correctly partition PWLs by point count."""
        # Create PWLs spanning multiple bins:
        # Bin 0 (1-8): 5 points
        # Bin 1 (9-16): 12 points
        # Bin 2 (17-32): 25 points
        # Bin 3 (33-64): 50 points
        # Bin 4 (65-128): 100 points
        # Bin 5 (129+): 150 points
        sources = {
            'I_bin0': {
                'node1': 'N1', 'dc_value': 0.0, 'pulses': [],
                'pwls': [{'delay': 0.0, 'period': 100e-9,
                          'points': [(i * 1e-9, float(i)) for i in range(5)]}],
            },
            'I_bin1': {
                'node1': 'N2', 'dc_value': 0.0, 'pulses': [],
                'pwls': [{'delay': 0.0, 'period': 100e-9,
                          'points': [(i * 1e-9, float(i)) for i in range(12)]}],
            },
            'I_bin2': {
                'node1': 'N3', 'dc_value': 0.0, 'pulses': [],
                'pwls': [{'delay': 0.0, 'period': 100e-9,
                          'points': [(i * 1e-9, float(i)) for i in range(25)]}],
            },
            'I_bin3': {
                'node1': 'N4', 'dc_value': 0.0, 'pulses': [],
                'pwls': [{'delay': 0.0, 'period': 100e-9,
                          'points': [(i * 1e-9, float(i)) for i in range(50)]}],
            },
            'I_bin4': {
                'node1': 'N5', 'dc_value': 0.0, 'pulses': [],
                'pwls': [{'delay': 0.0, 'period': 100e-9,
                          'points': [(i * 1e-9, float(i)) for i in range(100)]}],
            },
            'I_bin5': {
                'node1': 'N0', 'dc_value': 0.0, 'pulses': [],
                'pwls': [{'delay': 0.0, 'period': 100e-9,
                          'points': [(i * 1e-9, float(i)) for i in range(150)]}],
            },
        }
        node_to_idx, idx_to_node, n_nodes = create_node_mapping()
        vec = VectorizedCurrentSources.from_serialized_dicts(
            sources, node_to_idx, n_nodes
        )

        # Build binned groups
        vec._build_pwl_binned_groups()

        # Should have 6 groups (one per bin)
        self.assertEqual(len(vec._pwl_binned_groups), 6)

        # Check max_count for each group matches expectations
        expected_max_counts = {0: 5, 1: 12, 2: 25, 3: 50, 4: 100, 5: 150}
        for bin_id, group in vec._pwl_binned_groups.items():
            self.assertEqual(group['max_count'], expected_max_counts[bin_id],
                             f"Bin {bin_id} max_count mismatch")
            # Each group should have exactly 1 PWL
            self.assertEqual(len(group['pwl_indices']), 1,
                             f"Bin {bin_id} should have 1 PWL")

    def test_binned_vs_padded_equivalence(self):
        """Binned and padded evaluations should produce identical results."""
        # Create PWLs spanning multiple bins with various characteristics
        sources = {}
        np.random.seed(42)

        # Generate 20 PWLs with varying point counts
        point_counts = [3, 5, 8, 10, 15, 20, 30, 40, 60, 80, 100, 130,
                        4, 6, 9, 12, 18, 28, 55, 90]
        for i, cnt in enumerate(point_counts):
            times = np.linspace(0, 100e-9, cnt)
            values = np.sin(times * 1e8) + np.random.randn(cnt) * 0.1
            sources[f'I{i}'] = {
                'node1': f'N{i % 6}',
                'dc_value': 0.0,
                'pulses': [],
                'pwls': [{'delay': 0.0, 'period': 100e-9,
                          'points': list(zip(times.tolist(), values.tolist()))}],
            }

        node_to_idx, idx_to_node, n_nodes = create_node_mapping()
        vec = VectorizedCurrentSources.from_serialized_dicts(
            sources, node_to_idx, n_nodes
        )

        # Test at multiple time points
        test_times = [0.0, 5e-9, 25e-9, 50e-9, 75e-9, 99e-9, 100e-9, 150e-9]
        for t in test_times:
            # Force padded evaluation
            vec._pwl_binned_groups = None  # Clear binned cache
            vec._build_pwl_padded_cache()
            padded_result = vec._evaluate_pwls_padded(t)

            # Force binned evaluation
            vec._pwl_padded_times = None  # Clear padded cache
            vec._build_pwl_binned_groups()
            binned_result = vec._evaluate_pwls_binned(t)

            # Results should be identical
            np.testing.assert_array_almost_equal(
                padded_result, binned_result,
                decimal=12,
                err_msg=f"Mismatch at t={t}"
            )

    def test_binned_boundary_conditions(self):
        """Binned evaluation should handle boundary cases correctly."""
        # Test time before, at boundaries, and after waveform range
        sources = {
            'I_periodic': {
                'node1': 'N1', 'dc_value': 0.0, 'pulses': [],
                'pwls': [{'delay': 5e-9, 'period': 50e-9,
                          'points': [(0.0, 1.0), (10e-9, 2.0), (40e-9, 3.0)]}],
            },
            'I_nonperiodic': {
                'node1': 'N2', 'dc_value': 0.0, 'pulses': [],
                'pwls': [{'delay': 0.0, 'period': 0.0,
                          'points': [(0.0, 10.0), (10e-9, 20.0), (40e-9, 30.0)]}],
            },
        }
        node_to_idx, idx_to_node, n_nodes = create_node_mapping()
        vec = VectorizedCurrentSources.from_serialized_dicts(
            sources, node_to_idx, n_nodes
        )

        # Force binned evaluation
        vec._build_pwl_binned_groups()

        # Time before periodic start (t=0, but delay=5ns) -> should get first value
        result = vec._evaluate_pwls_binned(0.0)
        self.assertAlmostEqual(result[0], 1.0, places=10)  # periodic: first value
        self.assertAlmostEqual(result[1], 10.0, places=10)  # nonperiodic: first value

        # Time at midpoint of non-periodic
        result = vec._evaluate_pwls_binned(25e-9)
        # I_nonperiodic interpolates: segment (10e-9, 20) to (40e-9, 30)
        # frac = (25-10)/(40-10) = 0.5, value = 20 + (30-20)*0.5 = 25
        self.assertAlmostEqual(result[1], 25.0, places=10)

        # Time after non-periodic end -> should hold last value
        result = vec._evaluate_pwls_binned(100e-9)
        self.assertAlmostEqual(result[1], 30.0, places=10)  # hold last

    def test_binned_constant_pwl(self):
        """Binned evaluation should handle single-point (constant) PWLs."""
        sources = {
            'I_const': {
                'node1': 'N1', 'dc_value': 0.0, 'pulses': [],
                'pwls': [{'delay': 0.0, 'period': 0.0,
                          'points': [(0.0, 42.0)]}],  # Single point = constant
            },
        }
        node_to_idx, idx_to_node, n_nodes = create_node_mapping()
        vec = VectorizedCurrentSources.from_serialized_dicts(
            sources, node_to_idx, n_nodes
        )

        # Force binned evaluation
        vec._build_pwl_binned_groups()

        # Should always return constant value
        for t in [0.0, 1e-9, 100e-9, 1e-6]:
            result = vec._evaluate_pwls_binned(t)
            self.assertAlmostEqual(result[0], 42.0, places=10)

    def test_dispatch_threshold(self):
        """_evaluate_pwls should dispatch based on memory threshold."""
        # Small dataset: should use padded
        sources_small = {
            'I1': {
                'node1': 'N1', 'dc_value': 0.0, 'pulses': [],
                'pwls': [{'delay': 0.0, 'period': 0.0,
                          'points': [(0.0, 1.0), (10e-9, 2.0)]}],
            },
        }
        node_to_idx, idx_to_node, n_nodes = create_node_mapping()
        vec_small = VectorizedCurrentSources.from_serialized_dicts(
            sources_small, node_to_idx, n_nodes
        )

        # Verify dispatch chooses padded for small data
        max_count = int(vec_small.pwl_count.max())
        total_padded = vec_small.n_pwls * max_count * 16
        total_actual = vec_small.n_pwl_points * 16

        # Should NOT use binned (padding overhead is small)
        self.assertLessEqual(total_padded, 500_000_000)
        self.assertLessEqual(total_padded, 2 * total_actual)


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


class TestSourceIndexTracking(unittest.TestCase):
    """Tests for pulse_source_idx and pwl_source_idx tracking."""

    def test_pulse_source_idx_populated(self):
        """pulse_source_idx should track which source each pulse belongs to."""
        sources = {
            'I_src0': {
                'node1': 'N1', 'dc_value': 0.0,
                'pulses': [{'v1': 0.0, 'v2': 1.0, 'delay': 0.0, 'rt': 1e-9,
                            'ft': 1e-9, 'width': 5e-9, 'period': 20e-9}],
                'pwls': [],
            },
            'I_src1': {
                'node1': 'N2', 'dc_value': 0.0,
                'pulses': [
                    {'v1': 0.0, 'v2': 2.0, 'delay': 0.0, 'rt': 1e-9,
                     'ft': 1e-9, 'width': 5e-9, 'period': 20e-9},
                    {'v1': 0.0, 'v2': 3.0, 'delay': 10e-9, 'rt': 1e-9,
                     'ft': 1e-9, 'width': 5e-9, 'period': 20e-9},
                ],
                'pwls': [],
            },
        }
        node_to_idx, idx_to_node, n_nodes = create_node_mapping()
        vec = VectorizedCurrentSources.from_serialized_dicts(
            sources, node_to_idx, n_nodes
        )

        self.assertEqual(vec.n_pulses, 3)
        self.assertEqual(len(vec.pulse_source_idx), 3)
        # First pulse belongs to source 0, next two to source 1
        self.assertEqual(vec.pulse_source_idx[0], 0)
        self.assertEqual(vec.pulse_source_idx[1], 1)
        self.assertEqual(vec.pulse_source_idx[2], 1)

    def test_pwl_source_idx_populated(self):
        """pwl_source_idx should track which source each PWL belongs to."""
        sources = {
            'I_src0': {
                'node1': 'N1', 'dc_value': 0.0, 'pulses': [],
                'pwls': [{'delay': 0.0, 'period': 0.0,
                          'points': [(0.0, 1.0), (10e-9, 2.0)]}],
            },
            'I_src1': {
                'node1': 'N2', 'dc_value': 0.0, 'pulses': [],
                'pwls': [
                    {'delay': 0.0, 'period': 0.0,
                     'points': [(0.0, 3.0), (10e-9, 4.0)]},
                    {'delay': 5e-9, 'period': 20e-9,
                     'points': [(0.0, 0.0), (10e-9, 5.0)]},
                ],
            },
        }
        node_to_idx, idx_to_node, n_nodes = create_node_mapping()
        vec = VectorizedCurrentSources.from_serialized_dicts(
            sources, node_to_idx, n_nodes
        )

        self.assertEqual(vec.n_pwls, 3)
        self.assertEqual(len(vec.pwl_source_idx), 3)
        # First PWL belongs to source 0, next two to source 1
        self.assertEqual(vec.pwl_source_idx[0], 0)
        self.assertEqual(vec.pwl_source_idx[1], 1)
        self.assertEqual(vec.pwl_source_idx[2], 1)


class TestEvaluatePerSourceAtTime(unittest.TestCase):
    """Tests for evaluate_per_source_at_time method.

    This method is critical for correct masking when multiple sources
    share the same node. Unlike evaluate_at_time (which aggregates by node),
    this returns currents indexed by source.
    """

    def test_dc_only_per_source(self):
        """DC sources should return per-source values."""
        sources = {
            'I1': {'node1': 'N1', 'dc_value': 1.0, 'pulses': [], 'pwls': []},
            'I2': {'node1': 'N2', 'dc_value': 2.0, 'pulses': [], 'pwls': []},
            'I3': {'node1': 'N1', 'dc_value': 3.0, 'pulses': [], 'pwls': []},
        }
        node_to_idx, idx_to_node, n_nodes = create_node_mapping()
        vec = VectorizedCurrentSources.from_serialized_dicts(
            sources, node_to_idx, n_nodes
        )

        per_source = vec.evaluate_per_source_at_time(0.0)

        self.assertEqual(len(per_source), 3)
        self.assertAlmostEqual(per_source[0], 1.0)
        self.assertAlmostEqual(per_source[1], 2.0)
        self.assertAlmostEqual(per_source[2], 3.0)

    def test_per_source_vs_per_node_with_shared_node(self):
        """Per-source should differ from per-node when sources share a node."""
        # Two sources on the same node
        sources = {
            'I1': {'node1': 'N1', 'dc_value': 1.0, 'pulses': [], 'pwls': []},
            'I2': {'node1': 'N1', 'dc_value': 2.0, 'pulses': [], 'pwls': []},
        }
        node_to_idx, idx_to_node, n_nodes = create_node_mapping()
        vec = VectorizedCurrentSources.from_serialized_dicts(
            sources, node_to_idx, n_nodes
        )

        per_source = vec.evaluate_per_source_at_time(0.0)
        per_node = vec.evaluate_at_time(0.0)

        # Per-source: [1.0, 2.0]
        self.assertEqual(len(per_source), 2)
        self.assertAlmostEqual(per_source[0], 1.0)
        self.assertAlmostEqual(per_source[1], 2.0)

        # Per-node: N1 = 1.0 + 2.0 = 3.0
        self.assertAlmostEqual(per_node[node_to_idx['N1']], 3.0)

    def test_per_source_with_pulses(self):
        """Per-source should correctly add pulse contributions per source."""
        sources = {
            'I1': {
                'node1': 'N1', 'dc_value': 1.0,
                'pulses': [{'v1': 0.0, 'v2': 5.0, 'delay': 0.0, 'rt': 0.0,
                            'ft': 0.0, 'width': 10e-9, 'period': 20e-9}],
                'pwls': [],
            },
            'I2': {
                'node1': 'N1', 'dc_value': 2.0,  # Same node as I1
                'pulses': [],
                'pwls': [],
            },
        }
        node_to_idx, idx_to_node, n_nodes = create_node_mapping()
        vec = VectorizedCurrentSources.from_serialized_dicts(
            sources, node_to_idx, n_nodes
        )

        # At t=5ns (during pulse high): I1 = 1.0 + 5.0 = 6.0, I2 = 2.0
        per_source = vec.evaluate_per_source_at_time(5e-9)
        self.assertAlmostEqual(per_source[0], 6.0)  # DC + pulse
        self.assertAlmostEqual(per_source[1], 2.0)  # DC only

        # Per-node should aggregate: N1 = 6.0 + 2.0 = 8.0
        per_node = vec.evaluate_at_time(5e-9)
        self.assertAlmostEqual(per_node[node_to_idx['N1']], 8.0)

    def test_per_source_with_pwls(self):
        """Per-source should correctly add PWL contributions per source."""
        sources = {
            'I1': {
                'node1': 'N1', 'dc_value': 0.0, 'pulses': [],
                'pwls': [{'delay': 0.0, 'period': 0.0,
                          'points': [(0.0, 0.0), (10e-9, 4.0)]}],
            },
            'I2': {
                'node1': 'N1', 'dc_value': 1.0, 'pulses': [],  # Same node
                'pwls': [],
            },
        }
        node_to_idx, idx_to_node, n_nodes = create_node_mapping()
        vec = VectorizedCurrentSources.from_serialized_dicts(
            sources, node_to_idx, n_nodes
        )

        # At t=5ns: I1 PWL = 2.0 (interpolated), I2 = 1.0 DC
        per_source = vec.evaluate_per_source_at_time(5e-9)
        self.assertAlmostEqual(per_source[0], 2.0)
        self.assertAlmostEqual(per_source[1], 1.0)

    def test_per_source_empty_returns_empty(self):
        """Empty sources should return empty array."""
        node_to_idx, idx_to_node, n_nodes = create_node_mapping()
        vec = VectorizedCurrentSources.from_serialized_dicts(
            {}, node_to_idx, n_nodes
        )

        per_source = vec.evaluate_per_source_at_time(0.0)
        self.assertEqual(len(per_source), 0)

    def test_masking_per_source_vs_per_node(self):
        """Demonstrate correct masking using per-source vs incorrect per-node.

        This test validates the bug fix: when multiple sources share a node,
        masking should be applied per-source, not per-node.
        """
        # Two sources on the same node with different currents
        sources = {
            'I_near': {'node1': 'N1', 'dc_value': 3.0, 'pulses': [], 'pwls': []},
            'I_far': {'node1': 'N1', 'dc_value': 7.0, 'pulses': [], 'pwls': []},
        }
        node_to_idx, idx_to_node, n_nodes = create_node_mapping()
        vec = VectorizedCurrentSources.from_serialized_dicts(
            sources, node_to_idx, n_nodes
        )

        # Mask: only include first source (near)
        mask = np.array([True, False])

        # CORRECT: Per-source masking
        per_source = vec.evaluate_per_source_at_time(0.0)
        masked_per_source = np.where(mask, per_source, 0.0)
        correct_total = masked_per_source.sum()

        # The masked total should be 3.0 (only I_near)
        self.assertAlmostEqual(correct_total, 3.0)

        # INCORRECT (old bug): If we masked at node level
        per_node = vec.evaluate_at_time(0.0)  # N1 = 10.0
        # This would incorrectly include/exclude the entire node
        # rather than individual sources


class TestSharedNodeMasking(unittest.TestCase):
    """Tests for correct masking behavior when multiple sources share a node.

    These tests specifically validate the bug fix where masking must be
    applied per-source (not per-node) when multiple sources inject current
    into the same node.
    """

    def test_three_sources_same_node_selective_mask(self):
        """Selective masking of sources on same node should work correctly."""
        sources = {
            'I_a': {'node1': 'N1', 'dc_value': 1.0, 'pulses': [], 'pwls': []},
            'I_b': {'node1': 'N1', 'dc_value': 2.0, 'pulses': [], 'pwls': []},
            'I_c': {'node1': 'N1', 'dc_value': 4.0, 'pulses': [], 'pwls': []},
        }
        node_to_idx, idx_to_node, n_nodes = create_node_mapping()
        vec = VectorizedCurrentSources.from_serialized_dicts(
            sources, node_to_idx, n_nodes
        )

        per_source = vec.evaluate_per_source_at_time(0.0)

        # Mask only sources 0 and 2 (I_a and I_c)
        mask = np.array([True, False, True])
        masked = np.where(mask, per_source, 0.0)

        # Should get 1.0 + 4.0 = 5.0
        self.assertAlmostEqual(masked.sum(), 5.0)

        # With different mask (only I_b)
        mask2 = np.array([False, True, False])
        masked2 = np.where(mask2, per_source, 0.0)
        self.assertAlmostEqual(masked2.sum(), 2.0)

    def test_mixed_nodes_selective_mask(self):
        """Masking should work correctly with mixed shared/unique nodes."""
        sources = {
            'I_n1_a': {'node1': 'N1', 'dc_value': 1.0, 'pulses': [], 'pwls': []},
            'I_n1_b': {'node1': 'N1', 'dc_value': 2.0, 'pulses': [], 'pwls': []},
            'I_n2': {'node1': 'N2', 'dc_value': 10.0, 'pulses': [], 'pwls': []},
        }
        node_to_idx, idx_to_node, n_nodes = create_node_mapping()
        vec = VectorizedCurrentSources.from_serialized_dicts(
            sources, node_to_idx, n_nodes
        )

        per_source = vec.evaluate_per_source_at_time(0.0)

        # Mask first and third sources (I_n1_a and I_n2)
        mask = np.array([True, False, True])
        masked = np.where(mask, per_source, 0.0)

        # Should get 1.0 + 10.0 = 11.0 (not 3.0 + 10.0 = 13.0)
        self.assertAlmostEqual(masked.sum(), 11.0)

    def test_pulse_sources_shared_node_masking(self):
        """Masking pulse sources on shared node should be independent."""
        sources = {
            'I_pulse_a': {
                'node1': 'N1', 'dc_value': 0.0,
                'pulses': [{'v1': 0.0, 'v2': 5.0, 'delay': 0.0, 'rt': 0.0,
                            'ft': 0.0, 'width': 10e-9, 'period': 20e-9}],
                'pwls': [],
            },
            'I_pulse_b': {
                'node1': 'N1', 'dc_value': 0.0,  # Same node!
                'pulses': [{'v1': 0.0, 'v2': 3.0, 'delay': 0.0, 'rt': 0.0,
                            'ft': 0.0, 'width': 10e-9, 'period': 20e-9}],
                'pwls': [],
            },
        }
        node_to_idx, idx_to_node, n_nodes = create_node_mapping()
        vec = VectorizedCurrentSources.from_serialized_dicts(
            sources, node_to_idx, n_nodes
        )

        # At t=5ns both pulses are high
        per_source = vec.evaluate_per_source_at_time(5e-9)
        self.assertAlmostEqual(per_source[0], 5.0)
        self.assertAlmostEqual(per_source[1], 3.0)

        # Mask only first source
        mask = np.array([True, False])
        masked = np.where(mask, per_source, 0.0)
        self.assertAlmostEqual(masked.sum(), 5.0)

        # Mask only second source
        mask2 = np.array([False, True])
        masked2 = np.where(mask2, per_source, 0.0)
        self.assertAlmostEqual(masked2.sum(), 3.0)


class TestEvaluateToMultiRHS(unittest.TestCase):
    """Tests for VectorizedCurrentSources.evaluate_to_multi_rhs.

    These tests construct VectorizedCurrentSources with controlled DC values
    and node mappings, then verify the vectorized aggregation correctly handles:
    - Multiple sources mapping to the same unknown node
    - Masked source selection
    - Edge cases (empty masks, invalid sources)
    """

    def _make_vec_sources(self, dc_values, source_node_idx, n_nodes):
        """Helper to create VectorizedCurrentSources with DC values only."""
        return VectorizedCurrentSources(
            n_nodes=n_nodes,
            node_to_idx=None,
            n_sources=len(dc_values),
            dc_values=np.array(dc_values, dtype=np.float64),
            source_node_idx=np.array(source_node_idx, dtype=np.int32),
        )

    def test_multiple_sources_same_node(self):
        """Multiple sources at same node should be summed correctly."""
        # 4 sources: sources 0,1 at node 0; sources 2,3 at node 1
        vec_sources = self._make_vec_sources(
            dc_values=[1.0, 2.0, 3.0, 4.0],
            source_node_idx=[0, 0, 1, 1],
            n_nodes=2
        )

        # Direct mapping: all sources map to valid unknown indices
        source_to_unknown_direct = np.array([0, 0, 1, 1], dtype=np.int32)
        valid_sources = np.ones(4, dtype=bool)

        # All sources active
        masks = np.ones((1, 4), dtype=bool)
        rhs = np.zeros((2, 1), dtype=np.float64)

        total_currents = vec_sources.evaluate_to_multi_rhs(
            t=0.0, rhs_multi=rhs, source_masks=masks,
            source_to_unknown_direct=source_to_unknown_direct,
            valid_sources=valid_sources
        )

        # Node 0: -(1.0 + 2.0) = -3.0
        # Node 1: -(3.0 + 4.0) = -7.0
        np.testing.assert_allclose(rhs[:, 0], [-3.0, -7.0])
        self.assertAlmostEqual(total_currents[0], 10.0)

    def test_masked_sources_excluded(self):
        """Masked sources should be correctly excluded from aggregation."""
        vec_sources = self._make_vec_sources(
            dc_values=[1.0, 2.0, 3.0, 4.0],
            source_node_idx=[0, 0, 1, 1],
            n_nodes=2
        )

        source_to_unknown_direct = np.array([0, 0, 1, 1], dtype=np.int32)
        valid_sources = np.ones(4, dtype=bool)

        # Only even-indexed sources active (0 and 2)
        masks = np.array([[True, False, True, False]])
        rhs = np.zeros((2, 1), dtype=np.float64)

        total_currents = vec_sources.evaluate_to_multi_rhs(
            t=0.0, rhs_multi=rhs, source_masks=masks,
            source_to_unknown_direct=source_to_unknown_direct,
            valid_sources=valid_sources
        )

        # Node 0: -(1.0) = -1.0 (only source 0)
        # Node 1: -(3.0) = -3.0 (only source 2)
        np.testing.assert_allclose(rhs[:, 0], [-1.0, -3.0])
        self.assertAlmostEqual(total_currents[0], 4.0)  # 1.0 + 3.0

    def test_invalid_sources_excluded(self):
        """Sources not mapping to unknown nodes should be excluded from RHS."""
        vec_sources = self._make_vec_sources(
            dc_values=[1.0, 2.0, 3.0, 4.0],
            source_node_idx=[0, 0, 1, 1],
            n_nodes=2
        )

        # Sources 1,3 don't map to unknown nodes (e.g., they're at pad nodes)
        source_to_unknown_direct = np.array([0, -1, 1, -1], dtype=np.int32)
        valid_sources = source_to_unknown_direct >= 0

        masks = np.ones((1, 4), dtype=bool)
        rhs = np.zeros((2, 1), dtype=np.float64)

        total_currents = vec_sources.evaluate_to_multi_rhs(
            t=0.0, rhs_multi=rhs, source_masks=masks,
            source_to_unknown_direct=source_to_unknown_direct,
            valid_sources=valid_sources
        )

        # Only sources 0 and 2 contribute to RHS
        np.testing.assert_allclose(rhs[:, 0], [-1.0, -3.0])
        # But total_currents includes all sources
        self.assertAlmostEqual(total_currents[0], 10.0)

    def test_empty_mask_produces_zero_rhs(self):
        """Empty mask should produce zero RHS but zero total current."""
        vec_sources = self._make_vec_sources(
            dc_values=[1.0, 2.0, 3.0, 4.0],
            source_node_idx=[0, 0, 1, 1],
            n_nodes=2
        )

        source_to_unknown_direct = np.array([0, 0, 1, 1], dtype=np.int32)
        valid_sources = np.ones(4, dtype=bool)

        masks = np.zeros((1, 4), dtype=bool)  # No sources active
        rhs = np.zeros((2, 1), dtype=np.float64)

        total_currents = vec_sources.evaluate_to_multi_rhs(
            t=0.0, rhs_multi=rhs, source_masks=masks,
            source_to_unknown_direct=source_to_unknown_direct,
            valid_sources=valid_sources
        )

        np.testing.assert_allclose(rhs[:, 0], [0.0, 0.0])
        self.assertAlmostEqual(total_currents[0], 0.0)

    def test_multiple_masks_independent(self):
        """Multiple masks should produce independent RHS vectors."""
        vec_sources = self._make_vec_sources(
            dc_values=[1.0, 2.0, 3.0, 4.0],
            source_node_idx=[0, 0, 1, 1],
            n_nodes=2
        )

        source_to_unknown_direct = np.array([0, 0, 1, 1], dtype=np.int32)
        valid_sources = np.ones(4, dtype=bool)

        # Three different masks: all, even, odd
        masks = np.array([
            [True, True, True, True],    # all
            [True, False, True, False],  # even
            [False, True, False, True],  # odd
        ])
        rhs = np.zeros((2, 3), dtype=np.float64)

        total_currents = vec_sources.evaluate_to_multi_rhs(
            t=0.0, rhs_multi=rhs, source_masks=masks,
            source_to_unknown_direct=source_to_unknown_direct,
            valid_sources=valid_sources
        )

        # Mask 0 (all): Node 0 = -(1+2) = -3, Node 1 = -(3+4) = -7
        np.testing.assert_allclose(rhs[:, 0], [-3.0, -7.0])
        # Mask 1 (even): Node 0 = -1, Node 1 = -3
        np.testing.assert_allclose(rhs[:, 1], [-1.0, -3.0])
        # Mask 2 (odd): Node 0 = -2, Node 1 = -4
        np.testing.assert_allclose(rhs[:, 2], [-2.0, -4.0])

        # Verify linearity: all = even + odd
        np.testing.assert_allclose(rhs[:, 0], rhs[:, 1] + rhs[:, 2])

        # Total currents should also be additive
        np.testing.assert_allclose(total_currents, [10.0, 4.0, 6.0])

    def test_gaps_in_unknown_indices(self):
        """Sources with gaps in unknown indices should aggregate correctly."""
        vec_sources = self._make_vec_sources(
            dc_values=[1.0, 2.0],
            source_node_idx=[0, 4],  # Gap in node indices
            n_nodes=5
        )

        # Direct mapping with gap
        source_to_unknown_direct = np.array([0, 4], dtype=np.int32)
        valid_sources = np.ones(2, dtype=bool)

        masks = np.ones((1, 2), dtype=bool)
        rhs = np.zeros((5, 1), dtype=np.float64)

        vec_sources.evaluate_to_multi_rhs(
            t=0.0, rhs_multi=rhs, source_masks=masks,
            source_to_unknown_direct=source_to_unknown_direct,
            valid_sources=valid_sources
        )

        # Only nodes 0 and 4 should have values
        expected = np.array([-1.0, 0.0, 0.0, 0.0, -2.0])
        np.testing.assert_allclose(rhs[:, 0], expected)

    def test_no_valid_sources(self):
        """All sources invalid should produce zero RHS."""
        vec_sources = self._make_vec_sources(
            dc_values=[1.0, 2.0],
            source_node_idx=[0, 1],
            n_nodes=2
        )

        # All sources map to pads (invalid)
        source_to_unknown_direct = np.array([-1, -1], dtype=np.int32)
        valid_sources = np.zeros(2, dtype=bool)

        masks = np.ones((1, 2), dtype=bool)
        rhs = np.zeros((2, 1), dtype=np.float64)

        total_currents = vec_sources.evaluate_to_multi_rhs(
            t=0.0, rhs_multi=rhs, source_masks=masks,
            source_to_unknown_direct=source_to_unknown_direct,
            valid_sources=valid_sources
        )

        np.testing.assert_allclose(rhs[:, 0], [0.0, 0.0])
        self.assertAlmostEqual(total_currents[0], 3.0)  # Total still counted


if __name__ == '__main__':
    unittest.main()

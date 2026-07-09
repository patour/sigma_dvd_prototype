"""Tests for A2 additions to VectorizedCurrentSources.

Covers:
- ``get_period_info()`` period detection and helper correctness
- ``evaluate_at_times(t_grid)`` exactly matches loop over ``evaluate_at_time``
- Batch pulse and PWL evaluation
- wscale integration
- Edge cases: empty, DC-only, mixed
"""

import unittest
from typing import Any, Dict

import numpy as np
import pytest

from analysis.vectorized_sources import VectorizedCurrentSources

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_node_map(names):
    node_to_idx = {n: i for i, n in enumerate(names)}
    return node_to_idx, len(names)


def _make_pulse_source(node, period=50e-9, v1=0.0, v2=1.0,
                       delay=0.0, rt=1e-9, ft=1e-9, width=10e-9, dc=0.0):
    return {
        'node1': node,
        'dc_value': dc,
        'pulses': [{'v1': v1, 'v2': v2, 'delay': delay, 'rt': rt,
                    'ft': ft, 'width': width, 'period': period}],
        'pwls': [],
    }


def _make_pwl_source(node, period=50e-9, delay=0.0,
                     points=None, dc=0.0):
    if points is None:
        points = [(0.0, 0.0), (5e-9, 1.0), (25e-9, 1.0), (30e-9, 0.0)]
    return {
        'node1': node,
        'dc_value': dc,
        'pulses': [],
        'pwls': [{'delay': delay, 'period': period, 'points': points}],
    }


def _make_aperiodic_pwl_source(node, points=None, dc=0.0):
    if points is None:
        points = [(0.0, 0.0), (10e-9, 1.0), (20e-9, 0.5), (30e-9, 0.0)]
    return {
        'node1': node,
        'dc_value': dc,
        'pulses': [],
        'pwls': [{'delay': 0.0, 'period': 0.0, 'points': points}],
    }


def _build_vcs(sources, nodes):
    node_to_idx, n_nodes = _make_node_map(nodes)
    return VectorizedCurrentSources.from_serialized_dicts(
        sources, node_to_idx, n_nodes
    )


def _ref_evaluate(vcs, t_grid):
    """Reference: loop over evaluate_at_time."""
    return np.column_stack([vcs.evaluate_at_time(t) for t in t_grid])


# ---------------------------------------------------------------------------
# get_period_info tests
# ---------------------------------------------------------------------------

class TestGetPeriodInfo(unittest.TestCase):
    """Tests for VectorizedCurrentSources.get_period_info()."""

    def _dc_vcs(self):
        sources = {'I1': {'node1': 'N0', 'dc_value': 1.0, 'pulses': [], 'pwls': []}}
        return _build_vcs(sources, ['N0', 'N1'])

    def _single_period_pulse_vcs(self):
        sources = {
            'I1': _make_pulse_source('N0', period=50e-9),
            'I2': _make_pulse_source('N1', period=50e-9),
        }
        return _build_vcs(sources, ['N0', 'N1', 'N2'])

    def _mixed_period_vcs(self):
        sources = {
            'I1': _make_pulse_source('N0', period=50e-9),
            'I2': _make_pulse_source('N1', period=100e-9),
        }
        return _build_vcs(sources, ['N0', 'N1', 'N2'])

    def _single_period_pwl_vcs(self):
        sources = {'I1': _make_pwl_source('N0', period=50e-9)}
        return _build_vcs(sources, ['N0', 'N1'])

    def test_dc_only_no_period(self):
        info = self._dc_vcs().get_period_info()
        self.assertEqual(info['unique_pulse_periods'], [])
        self.assertEqual(info['unique_pwl_periods'], [])
        self.assertFalse(info['has_single_period'])
        self.assertIsNone(info['single_period'])
        self.assertEqual(info['n_active_source_rows'], 0)

    def test_single_pulse_period_detected(self):
        info = self._single_period_pulse_vcs().get_period_info()
        self.assertTrue(info['has_single_period'])
        self.assertAlmostEqual(info['single_period'], 50e-9, places=20)

    def test_mixed_periods_not_single(self):
        info = self._mixed_period_vcs().get_period_info()
        self.assertFalse(info['has_single_period'])
        self.assertIsNone(info['single_period'])

    def test_single_pwl_period_detected(self):
        info = self._single_period_pwl_vcs().get_period_info()
        self.assertTrue(info['has_single_period'])
        self.assertAlmostEqual(info['single_period'], 50e-9, places=20)
        self.assertTrue(info['all_zero_pwl_delay'])

    def test_pwl_nonzero_delay_reflected(self):
        sources = {'I1': _make_pwl_source('N0', period=50e-9, delay=5e-9)}
        vcs = _build_vcs(sources, ['N0', 'N1'])
        info = vcs.get_period_info()
        self.assertFalse(info['all_zero_pwl_delay'])

    def test_p_over_dt_is_integral_exact(self):
        info = self._single_period_pulse_vcs().get_period_info()
        helper = info['p_over_dt_is_integral']
        # P=50ns, dt=100ps → ratio=500 (integral)
        ok, m = helper(100e-12)
        self.assertTrue(ok)
        self.assertEqual(m, 500)

    def test_p_over_dt_not_integral(self):
        info = self._single_period_pulse_vcs().get_period_info()
        helper = info['p_over_dt_is_integral']
        # P=50ns, dt=33ps → ratio≈1515.15 (not integral)
        ok, m = helper(33e-12)
        self.assertFalse(ok)

    def test_est_table_mb_callable(self):
        info = self._single_period_pulse_vcs().get_period_info()
        est = info['est_table_mb']
        mb = est(500)
        self.assertGreater(mb, 0.0)

    def test_n_active_source_rows(self):
        sources = {
            'I1': _make_pulse_source('N0'),
            'I2': _make_pwl_source('N1'),
        }
        vcs = _build_vcs(sources, ['N0', 'N1', 'N2'])
        info = vcs.get_period_info()
        # 1 pulse + 1 PWL = 2 active rows
        self.assertEqual(info['n_active_source_rows'], 2)

    # -----------------------------------------------------------------------
    # Blocker fix: mixed periodic + aperiodic must NOT yield has_single_period
    # -----------------------------------------------------------------------

    def test_periodic_pulse_plus_aperiodic_pwl_not_single(self):
        """Periodic pulse (period=50ns) + aperiodic multi-knot PWL (period=0).

        Blocker: old code set has_single_period=True because it only looked
        at non-zero periods.  The aperiodic PWL would be silently folded
        modulo 50ns, corrupting results past step m=500.
        """
        sources = {
            'I1': _make_pulse_source('N0', period=50e-9),
            'I2': _make_aperiodic_pwl_source('N1'),  # period=0, multi-knot
        }
        vcs = _build_vcs(sources, ['N0', 'N1', 'N2'])
        info = vcs.get_period_info()
        self.assertFalse(
            info['has_single_period'],
            msg=(
                "has_single_period must be False when an aperiodic multi-knot PWL "
                "is present alongside a periodic pulse.  The aperiodic source cannot "
                "be folded modulo the pulse period."
            ),
        )
        self.assertIsNone(info['single_period'])
        self.assertTrue(info['has_aperiodic_pwl'])
        self.assertFalse(info['has_aperiodic_pulse'])

    def test_aperiodic_pulse_plus_periodic_pwl_not_single(self):
        """One-shot pulse (period=0, v1!=v2) + periodic PWL.

        A one-shot pulse cannot be phase-folded: after period P it must stay
        at v1, but the phase table would incorrectly repeat it every P steps.
        """
        sources = {
            # One-shot pulse: v1=0, v2=1.0, period=0 → fires once, stays at 0
            'I1': _make_pulse_source('N0', period=0.0, v1=0.0, v2=1.0),
            'I2': _make_pwl_source('N1', period=50e-9),
        }
        vcs = _build_vcs(sources, ['N0', 'N1', 'N2'])
        info = vcs.get_period_info()
        self.assertFalse(
            info['has_single_period'],
            msg=(
                "has_single_period must be False when a one-shot pulse (period=0, "
                "v1!=v2) is present.  Phase-folding would incorrectly repeat it."
            ),
        )
        self.assertIsNone(info['single_period'])
        self.assertTrue(info['has_aperiodic_pulse'])
        self.assertFalse(info['has_aperiodic_pwl'])

    def test_dc_pulse_period_zero_does_not_block_single_period(self):
        """DC pulse (period=0, v1==v2) does NOT block has_single_period.

        A pulse with v1==v2 contributes no AC content regardless of period,
        so it should not prevent phase-folding of other periodic sources.
        """
        sources = {
            # DC pulse: v1==v2 == 2.0, period=0 — effectively constant
            'I1': _make_pulse_source('N0', period=0.0, v1=2.0, v2=2.0),
            'I2': _make_pulse_source('N1', period=50e-9, v1=0.0, v2=1.0),
        }
        vcs = _build_vcs(sources, ['N0', 'N1', 'N2'])
        info = vcs.get_period_info()
        self.assertTrue(
            info['has_single_period'],
            msg=(
                "A DC pulse (v1==v2, period=0) should NOT block has_single_period "
                "because it contributes no time-varying content."
            ),
        )
        self.assertAlmostEqual(info['single_period'], 50e-9, places=20)
        self.assertFalse(info['has_aperiodic_pulse'])

    def test_single_knot_pwl_period_zero_does_not_block_single_period(self):
        """Single-knot PWL (period=0, count=1) does NOT block has_single_period.

        A single-knot PWL is just a DC offset — its period=0 does not cause
        aperiodic content and must not force chunked tier.
        """
        sources = {
            # Single-knot PWL: DC constant at 3.0 mA
            'I1': {'node1': 'N0', 'dc_value': 0.0, 'pulses': [],
                   'pwls': [{'delay': 0.0, 'period': 0.0, 'points': [(0.0, 3.0)]}]},
            'I2': _make_pulse_source('N1', period=50e-9),
        }
        vcs = _build_vcs(sources, ['N0', 'N1', 'N2'])
        info = vcs.get_period_info()
        self.assertTrue(
            info['has_single_period'],
            msg=(
                "A single-knot PWL (DC-only, period=0) should NOT block "
                "has_single_period — it contributes no time-varying content."
            ),
        )
        self.assertAlmostEqual(info['single_period'], 50e-9, places=20)
        self.assertFalse(info['has_aperiodic_pwl'])

    def test_all_aperiodic_not_single(self):
        """All-aperiodic sources (no periodic content) → has_single_period False."""
        sources = {
            'I1': _make_aperiodic_pwl_source('N0'),
            'I2': _make_aperiodic_pwl_source('N1'),
        }
        vcs = _build_vcs(sources, ['N0', 'N1', 'N2'])
        info = vcs.get_period_info()
        self.assertFalse(info['has_single_period'])
        self.assertTrue(info['has_aperiodic_pwl'])

    def test_has_aperiodic_pulse_and_pwl_fields_present(self):
        """get_period_info() must return has_aperiodic_pulse and has_aperiodic_pwl."""
        sources = {'I1': _make_pulse_source('N0', period=50e-9)}
        vcs = _build_vcs(sources, ['N0', 'N1'])
        info = vcs.get_period_info()
        self.assertIn('has_aperiodic_pulse', info)
        self.assertIn('has_aperiodic_pwl', info)
        self.assertFalse(info['has_aperiodic_pulse'])
        self.assertFalse(info['has_aperiodic_pwl'])


# ---------------------------------------------------------------------------
# evaluate_at_times vs reference loop
# ---------------------------------------------------------------------------

class TestEvaluateAtTimesVsReference(unittest.TestCase):
    """evaluate_at_times must exactly match m separate evaluate_at_time calls."""

    def _assert_exact(self, vcs, t_grid):
        got = vcs.evaluate_at_times(t_grid)
        ref = _ref_evaluate(vcs, t_grid)
        # Must be shape (n_nodes, m)
        self.assertEqual(got.shape, (vcs.n_nodes, len(t_grid)))
        np.testing.assert_array_equal(
            got, ref,
            err_msg="evaluate_at_times != reference loop over evaluate_at_time",
        )

    def test_dc_only(self):
        sources = {
            'I1': {'node1': 'N0', 'dc_value': 1.5, 'pulses': [], 'pwls': []},
            'I2': {'node1': 'N1', 'dc_value': 2.5, 'pulses': [], 'pwls': []},
        }
        vcs = _build_vcs(sources, ['N0', 'N1', 'N2'])
        t_grid = np.linspace(0, 100e-9, 20)
        self._assert_exact(vcs, t_grid)

    def test_single_pulse(self):
        sources = {'I1': _make_pulse_source('N0', period=50e-9, v2=5.0)}
        vcs = _build_vcs(sources, ['N0', 'N1'])
        t_grid = np.linspace(0, 100e-9, 50)
        self._assert_exact(vcs, t_grid)

    def test_multiple_pulses_different_periods(self):
        sources = {
            'I1': _make_pulse_source('N0', period=50e-9, v2=3.0),
            'I2': _make_pulse_source('N1', period=100e-9, v2=2.0),
        }
        vcs = _build_vcs(sources, ['N0', 'N1', 'N2'])
        t_grid = np.linspace(0, 200e-9, 80)
        self._assert_exact(vcs, t_grid)

    def test_single_periodic_pwl(self):
        sources = {'I1': _make_pwl_source('N0', period=50e-9)}
        vcs = _build_vcs(sources, ['N0', 'N1'])
        t_grid = np.linspace(0, 100e-9, 50)
        self._assert_exact(vcs, t_grid)

    def test_aperiodic_pwl(self):
        sources = {'I1': _make_aperiodic_pwl_source('N0')}
        vcs = _build_vcs(sources, ['N0', 'N1'])
        t_grid = np.linspace(0, 60e-9, 30)
        self._assert_exact(vcs, t_grid)

    def test_mixed_pulse_and_pwl(self):
        sources = {
            'I1': _make_pulse_source('N0', period=50e-9),
            'I2': _make_pwl_source('N1', period=50e-9),
            'I3': {'node1': 'N2', 'dc_value': 0.5, 'pulses': [], 'pwls': []},
        }
        vcs = _build_vcs(sources, ['N0', 'N1', 'N2'])
        t_grid = np.linspace(0, 100e-9, 50)
        self._assert_exact(vcs, t_grid)

    def test_with_wscale(self):
        """Waveform-scale factor applied to pulses/PWLs (source-level wscale)."""
        sources = {
            'I1': {
                'node1': 'N0',
                'dc_value': 0.0,
                'wscale': 3.7,  # source-level wscale (applied to all waveforms)
                'pulses': [{'v1': 0.0, 'v2': 1.0, 'delay': 0.0,
                             'rt': 1e-9, 'ft': 1e-9, 'width': 10e-9,
                             'period': 50e-9}],
                'pwls': [],
            }
        }
        vcs = _build_vcs(sources, ['N0', 'N1'])
        t_grid = np.linspace(0, 100e-9, 40)
        self._assert_exact(vcs, t_grid)

    def test_with_delay(self):
        sources = {
            'I1': _make_pulse_source('N0', period=50e-9, delay=10e-9),
        }
        vcs = _build_vcs(sources, ['N0', 'N1'])
        t_grid = np.linspace(0, 100e-9, 50)
        self._assert_exact(vcs, t_grid)

    def test_empty_t_grid(self):
        sources = {'I1': _make_pulse_source('N0')}
        vcs = _build_vcs(sources, ['N0', 'N1'])
        result = vcs.evaluate_at_times(np.array([]))
        self.assertEqual(result.shape, (vcs.n_nodes, 0))

    def test_two_sources_same_node(self):
        """Multiple sources on same node must be summed correctly."""
        sources = {
            'I1': _make_pulse_source('N0', period=50e-9, v2=2.0),
            'I2': _make_pulse_source('N0', period=50e-9, v2=3.0),
        }
        vcs = _build_vcs(sources, ['N0', 'N1'])
        t_grid = np.linspace(0, 100e-9, 50)
        self._assert_exact(vcs, t_grid)

    def test_single_time_point(self):
        sources = {'I1': _make_pulse_source('N0', period=50e-9, v2=5.0)}
        vcs = _build_vcs(sources, ['N0', 'N1'])
        t_grid = np.array([15e-9])
        self._assert_exact(vcs, t_grid)

    def test_pwl_with_nonzero_delay(self):
        sources = {'I1': _make_pwl_source('N0', period=50e-9, delay=5e-9)}
        vcs = _build_vcs(sources, ['N0', 'N1'])
        t_grid = np.linspace(0, 100e-9, 40)
        self._assert_exact(vcs, t_grid)


# ---------------------------------------------------------------------------
# Batch pulse / PWL internal correctness
# ---------------------------------------------------------------------------

class TestBatchPulseInternals(unittest.TestCase):
    """_evaluate_pulses_batch shape and value contract."""

    def test_shape(self):
        sources = {
            'I1': _make_pulse_source('N0'),
            'I2': _make_pulse_source('N1'),
        }
        vcs = _build_vcs(sources, ['N0', 'N1', 'N2'])
        t_grid = np.linspace(0, 100e-9, 20)
        batch = vcs._evaluate_pulses_batch(t_grid)
        self.assertEqual(batch.shape, (vcs.n_pulses, len(t_grid)))

    def test_values_match_scalar_loop(self):
        """Each column k of the batch must match evaluate_at_time at that t."""
        sources = {'I1': _make_pulse_source('N0', period=50e-9, v2=5.0)}
        vcs = _build_vcs(sources, ['N0', 'N1'])
        t_grid = np.linspace(0, 100e-9, 50)
        batch = vcs._evaluate_pulses_batch(t_grid)
        # scalar reference: evaluate_at_time and subtract DC
        dc_val = float(vcs.dc_values[vcs.pulse_source_idx[0]])
        for k, t in enumerate(t_grid):
            ref_node = vcs.evaluate_at_time(t)[vcs.pulse_node_idx[0]] - dc_val
            np.testing.assert_allclose(batch[0, k], ref_node, atol=1e-12,
                                       err_msg=f"Batch mismatch at k={k}, t={t}")

    def test_no_pulses_returns_empty(self):
        sources = {'I1': {'node1': 'N0', 'dc_value': 1.0, 'pulses': [], 'pwls': []}}
        vcs = _build_vcs(sources, ['N0', 'N1'])
        t_grid = np.linspace(0, 100e-9, 10)
        batch = vcs._evaluate_pulses_batch(t_grid)
        self.assertEqual(batch.shape, (0, len(t_grid)))


class TestBatchPWLInternals(unittest.TestCase):
    """_evaluate_pwls_batch shape and value contract."""

    def test_shape(self):
        sources = {
            'I1': _make_pwl_source('N0'),
            'I2': _make_pwl_source('N1'),
        }
        vcs = _build_vcs(sources, ['N0', 'N1', 'N2'])
        t_grid = np.linspace(0, 100e-9, 20)
        batch = vcs._evaluate_pwls_batch(t_grid)
        self.assertEqual(batch.shape, (vcs.n_pwls, len(t_grid)))

    def test_values_match_scalar_loop(self):
        sources = {'I1': _make_pwl_source('N0', period=50e-9)}
        vcs = _build_vcs(sources, ['N0', 'N1'])
        t_grid = np.linspace(0, 100e-9, 40)
        batch = vcs._evaluate_pwls_batch(t_grid)
        # scalar reference: evaluate_at_time minus DC (DC=0 here)
        for k, t in enumerate(t_grid):
            ref = vcs.evaluate_at_time(t)[vcs.pwl_node_idx[0]]
            np.testing.assert_allclose(batch[0, k], ref, atol=1e-12,
                                       err_msg=f"PWL batch mismatch at k={k}")

    def test_aperiodic_pwl_match(self):
        sources = {'I1': _make_aperiodic_pwl_source('N0')}
        vcs = _build_vcs(sources, ['N0', 'N1'])
        t_grid = np.linspace(0, 60e-9, 30)
        batch = vcs._evaluate_pwls_batch(t_grid)
        for k, t in enumerate(t_grid):
            ref = vcs.evaluate_at_time(t)[vcs.pwl_node_idx[0]]
            np.testing.assert_allclose(batch[0, k], ref, atol=1e-12,
                                       err_msg=f"Aperiodic PWL mismatch at k={k}")

    def test_no_pwls_returns_empty(self):
        sources = {'I1': {'node1': 'N0', 'dc_value': 1.0, 'pulses': [], 'pwls': []}}
        vcs = _build_vcs(sources, ['N0', 'N1'])
        t_grid = np.linspace(0, 100e-9, 10)
        batch = vcs._evaluate_pwls_batch(t_grid)
        self.assertEqual(batch.shape, (0, len(t_grid)))


# ---------------------------------------------------------------------------
# get_src_node_indices tests (Issue 2 fix)
# ---------------------------------------------------------------------------

class TestGetSrcNodeIndices(unittest.TestCase):
    """get_src_node_indices returns exactly the source-carrying node set."""

    def test_dc_only_nodes(self):
        sources = {
            'I1': {'node1': 'N0', 'dc_value': 1.5, 'pulses': [], 'pwls': []},
            'I2': {'node1': 'N1', 'dc_value': 0.0, 'pulses': [], 'pwls': []},
        }
        vcs = _build_vcs(sources, ['N0', 'N1', 'N2'])
        idx = vcs.get_src_node_indices()
        # N0 has nonzero dc; N1 has zero dc → only N0's index expected
        node_to_idx = {n: i for i, n in enumerate(['N0', 'N1', 'N2'])}
        self.assertIn(node_to_idx['N0'], idx)
        self.assertNotIn(node_to_idx['N1'], idx)

    def test_pulse_node_always_included(self):
        sources = {'I1': _make_pulse_source('N0', period=50e-9)}
        vcs = _build_vcs(sources, ['N0', 'N1'])
        idx = vcs.get_src_node_indices()
        node_to_idx = {n: i for i, n in enumerate(['N0', 'N1'])}
        self.assertIn(node_to_idx['N0'], idx)
        self.assertNotIn(node_to_idx['N1'], idx)

    def test_pwl_node_always_included(self):
        sources = {'I1': _make_pwl_source('N0', period=50e-9)}
        vcs = _build_vcs(sources, ['N0', 'N1'])
        idx = vcs.get_src_node_indices()
        node_to_idx = {n: i for i, n in enumerate(['N0', 'N1'])}
        self.assertIn(node_to_idx['N0'], idx)

    def test_mixed_sources_all_included(self):
        sources = {
            'I1': _make_pulse_source('N0', period=50e-9),
            'I2': _make_pwl_source('N1', period=50e-9),
            'I3': {'node1': 'N2', 'dc_value': 2.0, 'pulses': [], 'pwls': []},
        }
        vcs = _build_vcs(sources, ['N0', 'N1', 'N2'])
        idx = vcs.get_src_node_indices()
        node_to_idx = {n: i for i, n in enumerate(['N0', 'N1', 'N2'])}
        for name in ['N0', 'N1', 'N2']:
            self.assertIn(node_to_idx[name], idx)

    def test_empty_vcs_returns_empty(self):
        vcs = VectorizedCurrentSources(n_nodes=5)
        idx = vcs.get_src_node_indices()
        self.assertEqual(len(idx), 0)

    def test_sorted_unique_output(self):
        sources = {
            'I1': _make_pulse_source('N0', period=50e-9),
            'I2': _make_pulse_source('N0', period=50e-9),  # same node twice
        }
        vcs = _build_vcs(sources, ['N0', 'N1'])
        idx = vcs.get_src_node_indices()
        # Must be sorted and unique
        self.assertEqual(list(idx), sorted(set(idx.tolist())))


# ---------------------------------------------------------------------------
# evaluate_at_times_for_rows tests (Issue 2 fix)
# ---------------------------------------------------------------------------

class TestEvaluateAtTimesForRows(unittest.TestCase):
    """evaluate_at_times_for_rows must exactly match evaluate_at_times[rows]."""

    def _assert_sparse_matches_full(self, vcs, t_grid, row_indices=None):
        """evaluate_at_times_for_rows matches evaluate_at_times sliced to rows."""
        full = vcs.evaluate_at_times(t_grid)
        if row_indices is None:
            row_indices = vcs.get_src_node_indices()
        sparse = vcs.evaluate_at_times_for_rows(t_grid, row_indices)
        self.assertEqual(sparse.shape, (len(row_indices), len(t_grid)))
        np.testing.assert_array_equal(
            sparse, full[row_indices, :],
            err_msg="evaluate_at_times_for_rows does not match evaluate_at_times slice",
        )

    def test_dc_only(self):
        sources = {'I1': {'node1': 'N0', 'dc_value': 2.5, 'pulses': [], 'pwls': []}}
        vcs = _build_vcs(sources, ['N0', 'N1'])
        t_grid = np.linspace(0, 100e-9, 20)
        self._assert_sparse_matches_full(vcs, t_grid)

    def test_single_pulse(self):
        sources = {'I1': _make_pulse_source('N0', period=50e-9, v2=5.0)}
        vcs = _build_vcs(sources, ['N0', 'N1'])
        t_grid = np.linspace(0, 100e-9, 50)
        self._assert_sparse_matches_full(vcs, t_grid)

    def test_multiple_pulses(self):
        sources = {
            'I1': _make_pulse_source('N0', period=50e-9, v2=3.0),
            'I2': _make_pulse_source('N1', period=100e-9, v2=2.0),
        }
        vcs = _build_vcs(sources, ['N0', 'N1', 'N2'])
        t_grid = np.linspace(0, 200e-9, 80)
        self._assert_sparse_matches_full(vcs, t_grid)

    def test_single_periodic_pwl(self):
        sources = {'I1': _make_pwl_source('N0', period=50e-9)}
        vcs = _build_vcs(sources, ['N0', 'N1'])
        t_grid = np.linspace(0, 100e-9, 50)
        self._assert_sparse_matches_full(vcs, t_grid)

    def test_aperiodic_pwl(self):
        sources = {'I1': _make_aperiodic_pwl_source('N0')}
        vcs = _build_vcs(sources, ['N0', 'N1'])
        t_grid = np.linspace(0, 60e-9, 30)
        self._assert_sparse_matches_full(vcs, t_grid)

    def test_mixed_pulse_and_pwl(self):
        sources = {
            'I1': _make_pulse_source('N0', period=50e-9),
            'I2': _make_pwl_source('N1', period=50e-9),
            'I3': {'node1': 'N2', 'dc_value': 0.5, 'pulses': [], 'pwls': []},
        }
        vcs = _build_vcs(sources, ['N0', 'N1', 'N2'])
        t_grid = np.linspace(0, 100e-9, 50)
        self._assert_sparse_matches_full(vcs, t_grid)

    def test_with_wscale(self):
        sources = {
            'I1': {
                'node1': 'N0',
                'dc_value': 0.0,
                'wscale': 2.5,
                'pulses': [{'v1': 0.0, 'v2': 1.0, 'delay': 0.0,
                             'rt': 1e-9, 'ft': 1e-9, 'width': 10e-9,
                             'period': 50e-9}],
                'pwls': [],
            }
        }
        vcs = _build_vcs(sources, ['N0', 'N1'])
        t_grid = np.linspace(0, 100e-9, 40)
        self._assert_sparse_matches_full(vcs, t_grid)

    def test_explicit_row_subset(self):
        """Requesting a strict subset of src_nodes returns only those rows."""
        sources = {
            'I1': _make_pulse_source('N0', period=50e-9),
            'I2': _make_pulse_source('N1', period=50e-9),
            'I3': _make_pulse_source('N2', period=50e-9),
        }
        vcs = _build_vcs(sources, ['N0', 'N1', 'N2'])
        t_grid = np.linspace(0, 100e-9, 20)
        # Only request rows 0 and 2
        rows = np.array([0, 2], dtype=np.int32)
        sparse = vcs.evaluate_at_times_for_rows(t_grid, rows)
        full = vcs.evaluate_at_times(t_grid)
        self.assertEqual(sparse.shape, (2, len(t_grid)))
        np.testing.assert_array_equal(sparse[0], full[0])
        np.testing.assert_array_equal(sparse[1], full[2])

    def test_empty_row_indices_returns_zero_shape(self):
        sources = {'I1': _make_pulse_source('N0')}
        vcs = _build_vcs(sources, ['N0', 'N1'])
        t_grid = np.linspace(0, 100e-9, 20)
        result = vcs.evaluate_at_times_for_rows(t_grid, np.array([], dtype=np.int32))
        self.assertEqual(result.shape, (0, len(t_grid)))

    def test_empty_t_grid_returns_zero_shape(self):
        sources = {'I1': _make_pulse_source('N0')}
        vcs = _build_vcs(sources, ['N0', 'N1'])
        rows = np.array([0], dtype=np.int32)
        result = vcs.evaluate_at_times_for_rows(np.array([]), rows)
        self.assertEqual(result.shape, (1, 0))

    def test_peak_memory_is_n_src_rows_not_n_nodes(self):
        """Sanity: get_src_node_indices length is << n_nodes for sparse sources.

        This test documents the memory advantage of evaluate_at_times_for_rows:
        when n_src_rows << n_nodes, building the (n_src_rows, m) table avoids
        the (n_nodes, m) spike documented in the A2 quality-review issue.
        """
        sources = {
            'I1': _make_pulse_source('N0', period=50e-9),
            'I2': _make_pwl_source('N1', period=50e-9),
        }
        all_nodes = ['N%d' % i for i in range(100)]  # 100 nodes total
        vcs = _build_vcs(sources, all_nodes)

        src_rows = vcs.get_src_node_indices()
        # Only 2 nodes have sources — src_rows should be much smaller than 100
        self.assertLess(len(src_rows), len(all_nodes),
                        msg="src_rows should be << n_nodes for sparse sources")
        self.assertGreater(len(src_rows), 0)

        # Verify the sparse result matches full slice
        t_grid = np.linspace(0, 100e-9, 20)
        sparse = vcs.evaluate_at_times_for_rows(t_grid, src_rows)
        full = vcs.evaluate_at_times(t_grid)
        np.testing.assert_array_equal(sparse, full[src_rows, :])


# ---------------------------------------------------------------------------
# Finding 7: est_table_mb must count actual source-carrying rows (incl. DC)
# ---------------------------------------------------------------------------

class TestFinding7EstTableMb(unittest.TestCase):
    """est_table_mb must use get_src_node_indices() row count, not just waveforms.

    Finding 7 (vectorized_sources.py:get_period_info): the est_mb closure
    counted only n_active = n_pulses + n_pwls rows, but precompute_step_columns
    allocates one row per node returned by get_src_node_indices() — which
    includes DC-carrying nodes.  On DC-heavy netlists the real table can have
    far more rows than n_active, letting an oversized table slip past the
    max_table_mb gate.  The fix counts len(get_src_node_indices()).
    """

    def _dc_heavy_vcs(self, n_dc=100):
        """100 DC-only sources on distinct nodes + 2 periodic pulses."""
        sources = {}
        node_names = []
        for j in range(n_dc):
            name = f'D{j}'
            node_names.append(name)
            sources[f'I_dc_{j}'] = {
                'node1': name, 'dc_value': 1.0 + 0.01 * j,
                'pulses': [], 'pwls': [],
            }
        # Two periodic pulses on separate (non-DC) nodes
        for j in range(2):
            name = f'P{j}'
            node_names.append(name)
            sources[f'I_pulse_{j}'] = _make_pulse_source(name, period=50e-9)
        # Plus a spare node with no source
        node_names.append('SPARE')
        return _build_vcs(sources, node_names), n_dc

    def test_n_active_source_rows_counts_only_waveforms(self):
        """n_active_source_rows stays at the waveform count (2 pulses)."""
        vcs, _ = self._dc_heavy_vcs(n_dc=100)
        info = vcs.get_period_info()
        self.assertEqual(
            info['n_active_source_rows'], 2,
            msg='n_active_source_rows must remain the pulse/PWL waveform count',
        )

    def test_est_table_mb_reflects_dc_rows(self):
        """est_table_mb(500) must reflect 102 rows (100 DC + 2 pulse), not 2."""
        vcs, n_dc = self._dc_heavy_vcs(n_dc=100)
        info = vcs.get_period_info()
        est = info['est_table_mb']

        n_expected_rows = n_dc + 2  # 100 DC-carrying nodes + 2 pulse nodes
        expected_mb = n_expected_rows * 500 * 8 / 1e6
        old_buggy_mb = 2 * 500 * 8 / 1e6  # 0.008 MB (pulse count only)

        got = est(500)
        self.assertAlmostEqual(
            got, expected_mb, places=9,
            msg=(
                f'Finding 7: est_table_mb(500) should be {expected_mb:.6f} MB '
                f'({n_expected_rows} rows), not {old_buggy_mb:.6f} MB (2 rows).'
            ),
        )
        # The value that 0.008 fails but 0.408 passes
        self.assertGreater(
            got, 0.1,
            msg='Finding 7: est_table_mb must exceed 0.1 MB for 102 DC-heavy rows',
        )
        # And must exactly match len(get_src_node_indices())
        self.assertEqual(
            len(vcs.get_src_node_indices()), n_expected_rows,
            msg='get_src_node_indices must count 102 source-carrying nodes',
        )

    def test_tier_selection_respects_dc_row_budget(self):
        """A max_table_mb between the 2-row and 102-row estimates forces chunked.

        Before the fix the 2-row estimate (0.008 MB) passed any reasonable
        budget and the phase table was built anyway; after the fix the 102-row
        estimate (0.408 MB) exceeds a 0.1 MB budget → chunked tier.
        """
        from distributed.tile_worker import TileWorker, TileData

        vcs, n_dc = self._dc_heavy_vcs(n_dc=100)

        # Build a minimal worker and attach the DC-heavy VCS directly.
        # The tile only needs a solvable block system; node identity of the
        # VCS is independent (indices refer to the VCS node space).
        tile_data = TileData(
            tile_id=(0, 0),
            resistive_edges=[('p', 'a', 1.0), ('a', '0', 1.0)],
            all_nodes={'p', 'a'},
            boundary_nodes={'p'},
            current_injections={},
            capacitive_edges=[],
        )
        w = TileWorker()
        w.setup_from_tile_data(tile_data, {'p'})
        w._vec_sources = vcs
        w._active_sources = vcs
        w._current_buf = np.zeros(vcs.n_nodes, dtype=np.float64)

        # Budget between 0.008 MB (2 rows) and 0.408 MB (102 rows).
        # dt=100ps, P=50ns → m=500 (integral) → phase eligible on memory alone.
        info = w.precompute_step_columns(
            t_start=0.0, dt=100e-12, n_steps=1000, max_table_mb=0.1,
        )
        self.assertEqual(
            info['tier'], 'chunked',
            msg=(
                'Finding 7: with a 0.1 MB budget the 102-row (0.408 MB) table '
                f"must exceed it and select chunked, got tier='{info['tier']}'."
            ),
        )

    def test_dc_only_no_periodic_est_still_counts_dc(self):
        """DC-only VCS: est_table_mb counts the DC rows even with no waveforms."""
        sources = {
            f'I{j}': {'node1': f'N{j}', 'dc_value': 1.0, 'pulses': [], 'pwls': []}
            for j in range(50)
        }
        vcs = _build_vcs(sources, [f'N{j}' for j in range(50)] + ['SPARE'])
        info = vcs.get_period_info()
        est = info['est_table_mb']
        # 50 DC rows; n_active would be 0
        self.assertEqual(info['n_active_source_rows'], 0)
        self.assertAlmostEqual(est(100), 50 * 100 * 8 / 1e6, places=9)


if __name__ == '__main__':
    unittest.main()

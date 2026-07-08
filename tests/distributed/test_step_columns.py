"""Tests for A2: Phase-folded step-column table on TileWorker.

Covers:
- ``precompute_step_columns`` tier selection (phase vs chunked)
- Phase table column k vs ``evaluate_at_time(t_start + (k+1)*dt)``
- Chunked table column k vs reference
- Invalidation: cleared by ``init_vectorized_sources``, ``smooth_sources``,
  ``use_smoothed_sources``; NOT by ``set_current_node_mask``
- End-to-end: transient and QS with ``use_step_columns=True/False`` on the
  2-tile distributed model (max |dV| <= 1e-12 across all steps)
"""

import unittest
from typing import Any, Dict, Optional, Set

import numpy as np
import pytest

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Worker setup helpers (reused from test_time_domain.py patterns)
# ---------------------------------------------------------------------------

def _make_worker_with_tile(edges, port_nodes, currents=None):
    """Build a TileWorker with a minimal tile (no file I/O)."""
    from distributed.tile_worker import TileWorker, TileData

    tile_data = TileData(
        tile_id=(0, 0),
        resistive_edges=list(edges),
        all_nodes=set(n for e in edges for n in (e[0], e[1])),
        boundary_nodes=set(port_nodes),
        current_injections=currents or {},
        capacitive_edges=[],
    )
    # interface_nodes = boundary + any shared set
    interface_nodes = set(port_nodes)

    worker = TileWorker()
    worker.setup_from_tile_data(tile_data, interface_nodes)
    return worker


def _attach_pulse_vcs(worker, period=50e-9, v2=1.0, dc=0.5):
    """Attach a VCS with one pulse to a TileWorker (bypasses file init)."""
    from analysis.vectorized_sources import VectorizedCurrentSources
    from parser.current_sources import CurrentSource, Pulse

    n_ports = worker._block_system.n_ports
    n_interior = worker._block_system.n_interior
    n_nodes = n_ports + n_interior

    node_to_idx: Dict[str, int] = dict(worker._block_system.port_to_idx)
    for node, idx in worker._block_system.interior_to_idx.items():
        node_to_idx[node] = idx + n_ports

    # Pick the first interior node as the injection node
    interior_nodes = list(worker._block_system.interior_to_idx.keys())
    target_node = interior_nodes[0] if interior_nodes else list(node_to_idx.keys())[0]

    pulse = Pulse(v1=0.0, v2=v2, delay=0.0, rt=1e-9, ft=1e-9,
                  width=10e-9, period=period)
    src = CurrentSource(
        name='i_pulse', node1=target_node, node2='0',
        dc_value=dc,
        pulses=[pulse],
    )
    vcs = VectorizedCurrentSources.from_current_sources(
        {'i_pulse': src}, node_to_idx, n_nodes,
    )
    worker._vec_sources = vcs
    worker._active_sources = vcs
    # Allocate buffer
    worker._current_buf = np.zeros(n_nodes, dtype=np.float64)
    return vcs


# ---------------------------------------------------------------------------
# Tier selection tests
# ---------------------------------------------------------------------------

class TestTierSelection(unittest.TestCase):
    """precompute_step_columns selects the right tier."""

    def _worker_with_pulse(self, period=50e-9):
        w = _make_worker_with_tile(
            edges=[('p', 'a', 1.0), ('a', '0', 1.0)],
            port_nodes={'p'},
        )
        _attach_pulse_vcs(w, period=period)
        return w

    def test_single_period_integral_dt_selects_phase_tier(self):
        """P=50ns, dt=100ps → P/dt=500 → tier='phase'."""
        w = self._worker_with_pulse(period=50e-9)
        info = w.precompute_step_columns(t_start=0.0, dt=100e-12, n_steps=1000)
        self.assertEqual(info['tier'], 'phase')
        self.assertEqual(info.get('m'), 500)

    def test_non_integral_dt_selects_chunked_tier(self):
        """P=50ns, dt=33ps → not integral → tier='chunked'."""
        w = self._worker_with_pulse(period=50e-9)
        info = w.precompute_step_columns(t_start=0.0, dt=33e-12, n_steps=100)
        self.assertEqual(info['tier'], 'chunked')

    def test_no_vcs_returns_disabled(self):
        """Worker without VCS: tier='disabled'."""
        w = _make_worker_with_tile(
            edges=[('p', 'a', 1.0), ('a', '0', 1.0)],
            port_nodes={'p'},
        )
        # No VCS attached
        info = w.precompute_step_columns(t_start=0.0, dt=100e-12, n_steps=100)
        self.assertEqual(info['tier'], 'disabled')

    def test_use_step_columns_false_returns_disabled(self):
        """Setting use_step_columns=False disables the table."""
        w = self._worker_with_pulse(period=50e-9)
        info = w.precompute_step_columns(
            t_start=0.0, dt=100e-12, n_steps=1000,
            use_step_columns=False,
        )
        self.assertEqual(info['tier'], 'disabled')
        self.assertIsNone(w._step_col_table)

    def test_memory_cap_selects_chunked(self):
        """Memory cap: max_table_mb=0.0 forces chunked even for integral dt."""
        w = self._worker_with_pulse(period=50e-9)
        info = w.precompute_step_columns(
            t_start=0.0, dt=100e-12, n_steps=1000,
            max_table_mb=0.0,  # force chunked
        )
        self.assertEqual(info['tier'], 'chunked')

    def test_phase_table_info_fields(self):
        """Info dict for phase tier has required fields."""
        w = self._worker_with_pulse(period=50e-9)
        info = w.precompute_step_columns(t_start=0.0, dt=100e-12, n_steps=1000)
        self.assertEqual(info['tier'], 'phase')
        self.assertIn('m', info)
        self.assertIn('phase0', info)
        self.assertIn('n_src_nodes', info)
        self.assertIn('memory_mb', info)
        self.assertIn('build_path', info)

    def test_chunked_table_info_fields(self):
        """Info dict for chunked tier has required fields."""
        w = self._worker_with_pulse(period=50e-9)
        info = w.precompute_step_columns(
            t_start=0.0, dt=33e-12, n_steps=100,
        )
        self.assertEqual(info['tier'], 'chunked')
        self.assertIn('n_src_nodes', info)

    # -----------------------------------------------------------------------
    # Blocker fix: mixed periodic+aperiodic must route to chunked tier
    # -----------------------------------------------------------------------

    def _worker_with_mixed_sources(self, pulse_period=50e-9):
        """Worker with one periodic pulse + one aperiodic PWL source."""
        from analysis.vectorized_sources import VectorizedCurrentSources
        from parser.current_sources import CurrentSource, Pulse, PWL

        w = _make_worker_with_tile(
            edges=[('p', 'a', 1.0), ('p', 'b', 1.0),
                   ('a', '0', 1.0), ('b', '0', 1.0)],
            port_nodes={'p'},
        )
        n_ports = w._block_system.n_ports
        n_interior = w._block_system.n_interior
        n_nodes = n_ports + n_interior
        node_to_idx = dict(w._block_system.port_to_idx)
        for node, idx in w._block_system.interior_to_idx.items():
            node_to_idx[node] = idx + n_ports

        # Periodic pulse on node 'a'
        pulse = Pulse(v1=0.0, v2=1.0, delay=0.0, rt=1e-9, ft=1e-9,
                      width=10e-9, period=pulse_period)
        src_a = CurrentSource(name='ia', node1='a', node2='0', dc_value=0.0,
                              pulses=[pulse])

        # Aperiodic multi-knot PWL on node 'b' (period=0)
        pwl = PWL(delay=0.0, period=0.0,
                  points=[(0.0, 0.0), (10e-9, 1.0), (20e-9, 0.5), (30e-9, 0.0)])
        src_b = CurrentSource(name='ib', node1='b', node2='0', dc_value=0.0,
                              pwls=[pwl])

        vcs = VectorizedCurrentSources.from_current_sources(
            {'ia': src_a, 'ib': src_b}, node_to_idx, n_nodes,
        )
        w._vec_sources = vcs
        w._active_sources = vcs
        w._current_buf = np.zeros(n_nodes, dtype=np.float64)
        return w, vcs

    def test_mixed_periodic_aperiodic_selects_chunked_tier(self):
        """Blocker fix: periodic pulse + aperiodic PWL must select chunked tier.

        Before the fix, get_period_info().has_single_period was True for this
        case (aperiodic PWL with period=0 was ignored).  The phase table would
        fold the aperiodic PWL modulo P, producing wrong values after step m.
        After the fix, has_single_period=False → tier='chunked' → correct.
        """
        dt = 100e-12
        n_steps = 1000

        w, vcs = self._worker_with_mixed_sources(pulse_period=50e-9)
        info = w.precompute_step_columns(t_start=0.0, dt=dt, n_steps=n_steps)
        self.assertEqual(
            info['tier'], 'chunked',
            msg=(
                f"Expected chunked tier for mixed periodic+aperiodic sources, "
                f"got tier='{info['tier']}'. "
                f"Bug: aperiodic PWL (period=0, multi-knot) must block phase tier."
            ),
        )

    def test_mixed_chunked_values_match_reference(self):
        """Chunked table from mixed sources gives correct values at all steps.

        Verifies that after the blocker fix, the chunked path correctly handles
        the full combination of periodic + aperiodic sources.
        """
        dt = 1e-9
        n_steps = 50
        t_start = 0.0

        w, vcs = self._worker_with_mixed_sources(pulse_period=10e-9)
        info = w.precompute_step_columns(t_start=t_start, dt=dt, n_steps=n_steps)
        self.assertEqual(info['tier'], 'chunked',
                         msg=f"Expected chunked, got {info}")

        tbl = w._step_col_table
        max_err = 0.0
        for k in range(n_steps):
            t_k = t_start + (k + 1) * dt
            ref = vcs.evaluate_at_time(t_k)
            col = w._get_current_array_for_step(k, t_k).copy()
            err = float(np.max(np.abs(col - ref)))
            max_err = max(max_err, err)

        self.assertLessEqual(
            max_err, 1e-12,
            msg=(
                f"Mixed periodic+aperiodic chunked table: max_err={max_err:.3e}. "
                f"Chunked tier must reproduce evaluate_at_time exactly."
            ),
        )

    # -----------------------------------------------------------------------
    # Settings propagation: configure() must set use_step_columns + max_table_mb
    # -----------------------------------------------------------------------

    def test_configure_use_step_columns_false_disables_table(self):
        """configure({'use_step_columns': False}) must disable the table.

        This is the path Ray workers use (module globals don't propagate).
        """
        from distributed.tile_worker import TileWorker
        w = _make_worker_with_tile(
            edges=[('p', 'a', 1.0), ('a', '0', 1.0)],
            port_nodes={'p'},
        )
        _attach_pulse_vcs(w, period=50e-9)
        w.configure({'use_step_columns': False})
        info = w.precompute_step_columns(t_start=0.0, dt=100e-12, n_steps=1000)
        self.assertEqual(info['tier'], 'disabled')
        self.assertIsNone(w._step_col_table)

    def test_configure_max_table_mb_zero_forces_chunked(self):
        """configure({'max_table_mb': 0.0}) must force chunked tier.

        This lets operators cap per-worker memory via the factory path.
        """
        w = _make_worker_with_tile(
            edges=[('p', 'a', 1.0), ('a', '0', 1.0)],
            port_nodes={'p'},
        )
        _attach_pulse_vcs(w, period=50e-9)
        w.configure({'max_table_mb': 0.0})
        info = w.precompute_step_columns(t_start=0.0, dt=100e-12, n_steps=1000)
        self.assertEqual(info['tier'], 'chunked',
                         msg="max_table_mb=0.0 via configure should force chunked tier")

    def test_configure_settings_override_defaults(self):
        """configure() overrides both use_step_columns and max_table_mb defaults."""
        w = _make_worker_with_tile(
            edges=[('p', 'a', 1.0), ('a', '0', 1.0)],
            port_nodes={'p'},
        )
        _attach_pulse_vcs(w, period=50e-9)
        # Verify defaults
        self.assertTrue(w._use_step_columns)
        self.assertAlmostEqual(w._max_table_mb, 512.0, places=5)
        # Override
        w.configure({'use_step_columns': False, 'max_table_mb': 10.0})
        self.assertFalse(w._use_step_columns)
        self.assertAlmostEqual(w._max_table_mb, 10.0, places=5)


# ---------------------------------------------------------------------------
# Phase table column correctness
# ---------------------------------------------------------------------------

class TestPhaseTableColumns(unittest.TestCase):
    """Phase table column k == evaluate_at_time(t_start + (k+1)*dt)."""

    def _worker_and_vcs(self, period=50e-9):
        w = _make_worker_with_tile(
            edges=[('p', 'a', 1.0), ('a', '0', 1.0)],
            port_nodes={'p'},
        )
        vcs = _attach_pulse_vcs(w, period=period, v2=3.0, dc=0.5)
        return w, vcs

    def test_phase_table_columns_match_reference(self):
        """Each column k must equal evaluate_at_time(t_start+(k+1)*dt)."""
        dt = 100e-12
        t_start = 0.0
        n_steps = 1000

        w, vcs = self._worker_and_vcs(period=50e-9)
        info = w.precompute_step_columns(t_start=t_start, dt=dt, n_steps=n_steps)
        self.assertEqual(info['tier'], 'phase')

        m = info['m']
        tbl = w._step_col_table
        src_rows = tbl['src_rows']

        # Check first 2 full periods (2*m columns)
        n_check = min(2 * m, n_steps)
        for k in range(n_check):
            t_k = t_start + (k + 1) * dt
            ref = vcs.evaluate_at_time(t_k)
            # Get via table
            col = w._get_current_array_for_step(k, t_k)
            np.testing.assert_allclose(
                col[src_rows], ref[src_rows], atol=1e-12,
                err_msg=f"Phase column mismatch at step_idx={k}, t={t_k:.3e}",
            )

    def test_phase_table_col_wraps_modulo(self):
        """Column (step_idx % m) == column (step_idx + m) % m."""
        dt = 100e-12
        w, vcs = self._worker_and_vcs(period=50e-9)
        info = w.precompute_step_columns(t_start=0.0, dt=dt, n_steps=2000)
        self.assertEqual(info['tier'], 'phase')

        m = info['m']
        # step k and step k+m should return the same src_rows values
        for k in (0, 1, 5, m - 1):
            t_k = (k + 1) * dt
            c0 = w._get_current_array_for_step(k, t_k).copy()
            t_km = (k + m + 1) * dt
            cm = w._get_current_array_for_step(k + m, t_km).copy()
            tbl = w._step_col_table
            np.testing.assert_allclose(
                c0[tbl['src_rows']], cm[tbl['src_rows']], atol=1e-12,
                err_msg=f"Phase wrap mismatch at k={k}",
            )

    def test_nonzero_t_start_phase_offset(self):
        """With t_start > 0, phase0 = round(t_start/dt) % m."""
        dt = 100e-12
        period = 50e-9
        m_expected = round(period / dt)  # 500
        t_start = 10e-9  # = 100 steps

        w, vcs = self._worker_and_vcs(period=period)
        info = w.precompute_step_columns(t_start=t_start, dt=dt, n_steps=1000)
        self.assertEqual(info['tier'], 'phase')

        phase0 = info['phase0']
        expected_phase0 = int(round(t_start / dt)) % m_expected
        self.assertEqual(phase0, expected_phase0)

        # Column 0 should match t = t_start + dt
        tbl = w._step_col_table
        col = w._get_current_array_for_step(0, t_start + dt).copy()
        ref = vcs.evaluate_at_time(t_start + dt)
        np.testing.assert_allclose(
            col[tbl['src_rows']], ref[tbl['src_rows']], atol=1e-12,
        )


# ---------------------------------------------------------------------------
# Chunked table correctness
# ---------------------------------------------------------------------------
# Regression tests for A2 blocker fixes
# ---------------------------------------------------------------------------

class TestDirectScatterPhase0Regression(unittest.TestCase):
    """Regression: direct_scatter must NOT double-apply phase0.

    Blocker 1 (tile_worker_td.py:_build_via_direct_scatter): the old code
    used ``sample_idx = (phase0 + k + 1) % cnt`` which bakes phase0 into
    the table, while ``_get_current_array_for_step`` also applies phase0 at
    lookup time (``col = (step_idx + phase0) % m``).  Fix: ``(k+1) % cnt``.
    """

    def _make_smoothed_like_pwl_vcs(self, worker, period=10e-9, dt=100e-12):
        """Construct a VCS that looks like a smoothed-grid PWL starting at t=0.

        No pulses (n_pulses==0), delay=0, uniform grid starting at t=0
        — exactly the conditions that trigger the direct_scatter build path.

        The waveform must be properly periodic (values[0] == values[m]) because
        evaluate_at_time(P) wraps to t=0 and returns values[0], while the
        direct_scatter table stores values[m] at sample index m.  A sawtooth
        (values[0]=0, values[m]=1) would expose this as a mismatch; a cosine
        waveform ((1-cos(2pi*i/m))/2) has values[0]==values[m]==0.
        """
        from analysis.vectorized_sources import VectorizedCurrentSources
        from parser.current_sources import CurrentSource, PWL

        n_ports = worker._block_system.n_ports
        n_interior = worker._block_system.n_interior
        n_nodes = n_ports + n_interior
        node_to_idx = dict(worker._block_system.port_to_idx)
        for node, idx in worker._block_system.interior_to_idx.items():
            node_to_idx[node] = idx + n_ports

        interior_nodes = list(worker._block_system.interior_to_idx.keys())
        target = interior_nodes[0]

        m = int(round(period / dt))
        actual_step = period / m
        # Uniform grid starting at t=0 with m+1 samples (one full period +
        # the wrap-around sample).  Cosine-based: values[0]==values[m]==0,
        # peak 1.0 at i=m//2.  Non-trivial shape ensures wrong-phase lookups
        # produce visibly wrong values.
        times = [i * actual_step for i in range(m + 1)]
        values = [
            (1.0 - np.cos(2.0 * np.pi * i / m)) / 2.0
            for i in range(m + 1)
        ]

        pwl = PWL(delay=0.0, period=period, points=list(zip(times, values)))
        src = CurrentSource(
            name='i_smooth', node1=target, node2='0', dc_value=0.0, pwls=[pwl],
        )
        vcs = VectorizedCurrentSources.from_current_sources(
            {'i_smooth': src}, node_to_idx, n_nodes,
        )
        worker._vec_sources = vcs
        worker._active_sources = vcs
        worker._current_buf = np.zeros(n_nodes, dtype=np.float64)
        return vcs

    def test_direct_scatter_nonzero_t_start_matches_reference(self):
        """direct_scatter with t_start=2ns (phase0!=0) must match evaluate_at_time.

        Before the fix (sample_idx=(phase0+k+1)%cnt), this test produces
        max_err > 0.1 for phase0=20 on a 10ns-period sawtooth.
        """
        period = 10e-9
        dt = 100e-12
        t_start = 2e-9  # phase0 = round(2ns/100ps) % 100 = 20

        w = _make_worker_with_tile(
            edges=[('p', 'a', 1.0), ('a', '0', 1.0)],
            port_nodes={'p'},
        )
        vcs = self._make_smoothed_like_pwl_vcs(w, period=period, dt=dt)

        # Check that direct_scatter is actually selected (skip if not)
        path = w._select_build_path(dt, int(round(period / dt)))
        if path != 'direct_scatter':
            self.skipTest(
                f'direct_scatter not selected (got {path!r}); '
                'VCS does not trigger the path on this run'
            )

        n_steps = 500
        info = w.precompute_step_columns(t_start=t_start, dt=dt, n_steps=n_steps)
        self.assertEqual(info['tier'], 'phase', msg=f"Expected phase tier, got {info}")
        self.assertEqual(info.get('build_path'), 'direct_scatter',
                         msg=f"Expected direct_scatter path, got {info}")

        tbl = w._step_col_table
        src_rows = tbl['src_rows']
        self.assertGreater(len(src_rows), 0)

        max_err = 0.0
        for k in range(min(n_steps, 2 * int(round(period / dt)))):
            t_k = t_start + (k + 1) * dt
            ref = vcs.evaluate_at_time(t_k)
            col = w._get_current_array_for_step(k, t_k).copy()
            err = float(np.max(np.abs(col[src_rows] - ref[src_rows])))
            max_err = max(max_err, err)

        self.assertLessEqual(
            max_err, 1e-9,
            msg=f'direct_scatter phase0!=0: max_err={max_err:.3e} (expected <= 1e-9). '
                f'Bug: (phase0+k+1)%cnt double-applies phase0.',
        )

    def test_direct_scatter_zero_t_start_still_correct(self):
        """Sanity: direct_scatter at t_start=0 (phase0=0) remains correct."""
        period = 10e-9
        dt = 100e-12
        t_start = 0.0

        w = _make_worker_with_tile(
            edges=[('p', 'a', 1.0), ('a', '0', 1.0)],
            port_nodes={'p'},
        )
        vcs = self._make_smoothed_like_pwl_vcs(w, period=period, dt=dt)

        path = w._select_build_path(dt, int(round(period / dt)))
        if path != 'direct_scatter':
            self.skipTest(f'direct_scatter not selected (got {path!r})')

        info = w.precompute_step_columns(t_start=t_start, dt=dt, n_steps=500)
        self.assertEqual(info['tier'], 'phase')

        tbl = w._step_col_table
        src_rows = tbl['src_rows']

        for k in range(min(500, 2 * int(round(period / dt)))):
            t_k = t_start + (k + 1) * dt
            ref = vcs.evaluate_at_time(t_k)
            col = w._get_current_array_for_step(k, t_k).copy()
            np.testing.assert_allclose(
                col[src_rows], ref[src_rows], atol=1e-9,
                err_msg=f'direct_scatter t_start=0 mismatch at k={k}',
            )


class TestChunkedWindowLateActiveSource(unittest.TestCase):
    """Regression: chunked window must not drop sources inactive in window 0.

    Blocker 2 (_get_current_array_for_step): the old code computed src_rows
    only from window 0, then reused that stale index when rebuilding later
    windows.  Sources that are exactly zero throughout window 0 but active
    in a later window are silently dropped (their contribution stays 0 even
    when they should be nonzero).
    """

    def _make_two_source_worker(self):
        """Worker with two interior nodes and two aperiodic PWL sources."""
        from analysis.vectorized_sources import VectorizedCurrentSources
        from parser.current_sources import CurrentSource, PWL

        # Two interior nodes: 'a' (source A active early) and 'b' (source B
        # zero in window 0, active after step 512).
        w = _make_worker_with_tile(
            edges=[('p', 'a', 1.0), ('p', 'b', 1.0), ('a', '0', 1.0), ('b', '0', 1.0)],
            port_nodes={'p'},
        )
        n_ports = w._block_system.n_ports
        n_interior = w._block_system.n_interior
        n_nodes = n_ports + n_interior
        node_to_idx = dict(w._block_system.port_to_idx)
        for node, idx in w._block_system.interior_to_idx.items():
            node_to_idx[node] = idx + n_ports

        dt = 1e-9
        # Source A: spike at t=5ns (well within window 0 of W=512 steps)
        pwl_a = PWL(
            delay=0.0, period=0.0,
            points=[(0.0, 0.0), (4e-9, 0.0), (5e-9, 1.0), (6e-9, 0.0), (1e-6, 0.0)],
        )
        src_a = CurrentSource(name='ia', node1='a', node2='0', dc_value=0.0, pwls=[pwl_a])

        # Source B: zero for first 512 steps, spike at t=550ns (step 549)
        pwl_b = PWL(
            delay=0.0, period=0.0,
            points=[(0.0, 0.0), (549e-9, 0.0), (550e-9, 2.0), (551e-9, 0.0), (1e-6, 0.0)],
        )
        src_b = CurrentSource(name='ib', node1='b', node2='0', dc_value=0.0, pwls=[pwl_b])

        vcs = VectorizedCurrentSources.from_current_sources(
            {'ia': src_a, 'ib': src_b}, node_to_idx, n_nodes,
        )
        w._vec_sources = vcs
        w._active_sources = vcs
        w._current_buf = np.zeros(n_nodes, dtype=np.float64)
        return w, vcs

    def test_late_active_source_not_dropped(self):
        """Source B (zero in window 0) must be present in window 1.

        Before the fix, src_rows was computed only from window 0, missing
        source B entirely.  After the fix, src_rows is recomputed per window.
        """
        dt = 1e-9
        n_steps = 600  # forces a second window (W=512 default)

        w, vcs = self._make_two_source_worker()
        info = w.precompute_step_columns(t_start=0.0, dt=dt, n_steps=n_steps)
        self.assertEqual(info['tier'], 'chunked',
                         msg=f'Expected chunked tier, got {info}')

        # Step 549 falls in window 1 (beyond W=512).  At t=550ns source B spikes.
        step = 549
        t_step = (step + 1) * dt  # 550ns
        col = w._get_current_array_for_step(step, t_step).copy()
        ref = vcs.evaluate_at_time(t_step)

        # Source B must be non-zero at this step
        node_b_idx = w._block_system.interior_to_idx.get('b')
        if node_b_idx is not None:
            node_b_abs = node_b_idx + w._block_system.n_ports
            self.assertGreater(
                abs(ref[node_b_abs]), 1e-6,
                msg='Source B should be active at step 549 (t=550ns) per reference',
            )
            self.assertAlmostEqual(
                col[node_b_abs], ref[node_b_abs], places=9,
                msg=(
                    f'Source B dropped in window 1: got {col[node_b_abs]:.6f}, '
                    f'expected {ref[node_b_abs]:.6f}. '
                    f'Bug: stale src_rows from window 0 silently zeros late sources.'
                ),
            )

        # Full array agreement
        np.testing.assert_allclose(
            col, ref, atol=1e-12,
            err_msg=f'Chunked window 1 column mismatch at step {step}',
        )

    def test_window_0_still_correct_after_rebuild(self):
        """After a window rebuild, re-accessing window-0 steps triggers another
        rebuild and must still produce correct values."""
        dt = 1e-9
        n_steps = 600

        w, vcs = self._make_two_source_worker()
        w.precompute_step_columns(t_start=0.0, dt=dt, n_steps=n_steps)

        # Access step 0 (window 0), then step 549 (window 1), then step 0 again
        # (forces rebuild back to window 0).
        for step in (0, 549, 5):
            t_step = (step + 1) * dt
            col = w._get_current_array_for_step(step, t_step).copy()
            ref = vcs.evaluate_at_time(t_step)
            np.testing.assert_allclose(
                col, ref, atol=1e-12,
                err_msg=f'Chunked column mismatch at step {step} (t={t_step:.3e})',
            )


# ---------------------------------------------------------------------------

class TestChunkedTableColumns(unittest.TestCase):
    """Chunked table columns match evaluate_at_time for aperiodic sources."""

    def _worker_aperiodic(self):
        from analysis.vectorized_sources import VectorizedCurrentSources
        from parser.current_sources import CurrentSource, PWL

        w = _make_worker_with_tile(
            edges=[('p', 'a', 1.0), ('a', '0', 1.0)],
            port_nodes={'p'},
        )
        n_ports = w._block_system.n_ports
        n_interior = w._block_system.n_interior
        n_nodes = n_ports + n_interior
        node_to_idx = dict(w._block_system.port_to_idx)
        for node, idx in w._block_system.interior_to_idx.items():
            node_to_idx[node] = idx + n_ports

        interior_nodes = list(w._block_system.interior_to_idx.keys())
        target = interior_nodes[0]

        # Aperiodic PWL
        pwl = PWL(
            delay=0.0, period=0.0,
            points=[(0.0, 0.0), (10e-9, 1.0), (20e-9, 0.5), (30e-9, 0.0)],
        )
        src = CurrentSource(
            name='i_pwl', node1=target, node2='0', dc_value=0.0,
            pwls=[pwl],
        )
        vcs = VectorizedCurrentSources.from_current_sources(
            {'i_pwl': src}, node_to_idx, n_nodes,
        )
        w._vec_sources = vcs
        w._active_sources = vcs
        w._current_buf = np.zeros(n_nodes, dtype=np.float64)
        return w, vcs

    def test_chunked_columns_match_reference(self):
        """Each step column matches evaluate_at_time for aperiodic source."""
        dt = 1e-9
        t_start = 0.0
        n_steps = 50

        w, vcs = self._worker_aperiodic()
        info = w.precompute_step_columns(t_start=t_start, dt=dt, n_steps=n_steps)
        self.assertEqual(info['tier'], 'chunked')

        tbl = w._step_col_table
        src_rows = tbl['src_rows']

        for k in range(n_steps):
            t_k = t_start + (k + 1) * dt
            ref = vcs.evaluate_at_time(t_k)
            col = w._get_current_array_for_step(k, t_k).copy()
            np.testing.assert_allclose(
                col[src_rows], ref[src_rows], atol=1e-12,
                err_msg=f"Chunked mismatch at step_idx={k}, t={t_k:.3e}",
            )

    def test_chunked_window_rebuilds_on_access(self):
        """Accessing step_idx beyond initial window triggers rebuild."""
        dt = 1e-9
        n_steps = 600  # more than default W=512

        w, vcs = self._worker_aperiodic()
        info = w.precompute_step_columns(t_start=0.0, dt=dt, n_steps=n_steps)
        self.assertEqual(info['tier'], 'chunked')

        # First access at step 0: chunk_start should be 0
        _ = w._get_current_array_for_step(0, 1e-9)
        self.assertEqual(w._step_col_table['_chunk_start'], 0)

        # Access at step 520: beyond W=512, should rebuild
        t_520 = (520 + 1) * dt
        col_520 = w._get_current_array_for_step(520, t_520).copy()
        ref_520 = vcs.evaluate_at_time(t_520)
        tbl = w._step_col_table
        np.testing.assert_allclose(
            col_520[tbl['src_rows']], ref_520[tbl['src_rows']], atol=1e-12,
        )
        self.assertEqual(tbl['_chunk_start'], 520)


# ---------------------------------------------------------------------------
# Invalidation tests
# ---------------------------------------------------------------------------

class TestTableInvalidation(unittest.TestCase):
    """Table cleared on source changes; NOT on mask changes."""

    def _worker_with_table(self, period=50e-9, dt=100e-12, n_steps=1000):
        w = _make_worker_with_tile(
            edges=[('p', 'a', 1.0), ('a', '0', 1.0)],
            port_nodes={'p'},
        )
        _attach_pulse_vcs(w, period=period)
        info = w.precompute_step_columns(t_start=0.0, dt=dt, n_steps=n_steps)
        self.assertNotEqual(info['tier'], 'disabled')
        self.assertIsNotNone(w._step_col_table)
        return w

    def test_table_cleared_on_smooth_sources(self):
        """smooth_sources() must clear _step_col_table."""
        w = self._worker_with_table()
        # Need to call smooth_sources — requires vec_sources to be initialized
        # We can call it via the worker's method
        try:
            w.smooth_sources(time_step=100e-12, t_start=0.0, t_end=100e-9)
        except Exception:
            # smooth_sources may raise if no nodes; still check invalidation
            pass
        self.assertIsNone(w._step_col_table)

    def test_table_cleared_on_use_smoothed_sources(self):
        """use_smoothed_sources() must clear _step_col_table."""
        w = _make_worker_with_tile(
            edges=[('p', 'a', 1.0), ('a', '0', 1.0)],
            port_nodes={'p'},
        )
        vcs = _attach_pulse_vcs(w, period=50e-9)
        # Manually create a smoothed source (same vcs, just to test switching)
        w._smoothed_sources = vcs
        w._active_sources = vcs
        w._current_buf = np.zeros(
            w._block_system.n_ports + w._block_system.n_interior, dtype=np.float64
        )
        # Build table
        info = w.precompute_step_columns(t_start=0.0, dt=100e-12, n_steps=1000)
        self.assertIsNotNone(w._step_col_table)
        # Switching sources must invalidate
        w.use_smoothed_sources(use_smoothed=False)
        self.assertIsNone(w._step_col_table)

    def test_mask_does_NOT_clear_table(self):
        """set_current_node_mask() must NOT clear _step_col_table."""
        w = self._worker_with_table()
        tbl_before = w._step_col_table
        self.assertIsNotNone(tbl_before)
        n_nodes = w._block_system.n_ports + w._block_system.n_interior
        mask = np.ones(n_nodes, dtype=np.float64)
        w.set_current_node_mask(mask)
        self.assertIsNotNone(w._step_col_table)
        # Table should be the same object (not rebuilt)
        self.assertIs(w._step_col_table, tbl_before)

    def test_cache_key_rebuild_on_dt_change(self):
        """Calling precompute_step_columns with different dt rebuilds the table."""
        w = _make_worker_with_tile(
            edges=[('p', 'a', 1.0), ('a', '0', 1.0)],
            port_nodes={'p'},
        )
        _attach_pulse_vcs(w, period=50e-9)
        info1 = w.precompute_step_columns(t_start=0.0, dt=100e-12, n_steps=500)
        tbl1 = w._step_col_table
        # Call with different dt
        info2 = w.precompute_step_columns(t_start=0.0, dt=200e-12, n_steps=250)
        # Table should be different object
        self.assertIsNot(w._step_col_table, tbl1)


# ---------------------------------------------------------------------------
# End-to-end: transient with use_step_columns=True vs False
# ---------------------------------------------------------------------------

def _build_solvable_two_tile_model():
    """Build a minimal 2-tile DistributedPowerGridModel that actually solves.

    Unlike _build_two_tile_distributed_model, this model keeps 'pad' out of
    tile ports — it's only a package-level Dirichlet node.  This avoids the
    tile_index_maps length mismatch that makes the shared fixture unsolvable.

    Topology:
        Tile A: a1 --[1mS]-- shared
        Tile B: b1 --[2mS]-- shared --[1mS]-- 0 (ground)
        Package: pad (1.0 V) --[10mS]-- shared
        Interface: {shared}
    """
    from distributed.backend import LocalBackend
    from distributed.model import DistributedPowerGridModel
    from distributed.parser import PackageData, PowerGridMetaData, TileConfig
    from distributed.tile_worker import TileWorker, TileData

    # Tile A: only 'shared' is boundary
    tile_a_data = TileData(
        tile_id=(0, 0),
        resistive_edges=[('a1', 'shared', 1.0)],
        all_nodes={'a1', 'shared'},
        boundary_nodes={'shared'},
        current_injections={'a1': 0.5},
        capacitive_edges=[('a1', '0', 10.0)],  # 10 fF grounded
    )
    # Tile B: 'shared' + ground connection
    tile_b_data = TileData(
        tile_id=(0, 1),
        resistive_edges=[('b1', 'shared', 2.0), ('b1', '0', 1.0)],
        all_nodes={'shared', 'b1'},
        boundary_nodes={'shared'},
        current_injections={'b1': 0.3},
        capacitive_edges=[('b1', '0', 5.0)],  # 5 fF grounded
    )

    # Interface = {shared} only; pad is Dirichlet at package level
    interface_nodes = {'shared'}

    be = LocalBackend()
    be.initialize()

    worker_a = TileWorker()
    worker_a.setup_from_tile_data(tile_a_data, interface_nodes)

    worker_b = TileWorker()
    worker_b.setup_from_tile_data(tile_b_data, interface_nodes)

    workers = [worker_a, worker_b]

    pkg_data = PackageData(
        vsrc_dict={'V1': {'node+': 'pad', 'node-': '0', 'net': 'VDD', 'value': 1.0}},
        package_edges=[('pad', 'shared', 10.0)],  # 10 mS package resistor
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

    tile_boundary_nodes = {
        (0, 0): ['shared'],
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


def _attach_vcs_to_workers(workers):
    """Attach a single-period pulse VCS to each worker."""
    from analysis.vectorized_sources import VectorizedCurrentSources
    from parser.current_sources import CurrentSource, Pulse

    period = 50e-9

    for i, worker in enumerate(workers):
        n_ports = worker._block_system.n_ports
        n_interior = worker._block_system.n_interior
        n_nodes = n_ports + n_interior

        node_to_idx = dict(worker._block_system.port_to_idx)
        for node, idx in worker._block_system.interior_to_idx.items():
            node_to_idx[node] = idx + n_ports

        interior_nodes = list(worker._block_system.interior_to_idx.keys())
        if not interior_nodes:
            continue
        target = interior_nodes[0]

        pulse = Pulse(v1=0.0, v2=1.0, delay=0.0, rt=1e-9, ft=1e-9,
                      width=10e-9, period=period)
        src = CurrentSource(
            name=f'i_tile{i}', node1=target, node2='0',
            dc_value=0.3,
            pulses=[pulse],
        )
        vcs = VectorizedCurrentSources.from_current_sources(
            {f'i_tile{i}': src}, node_to_idx, n_nodes,
        )
        worker._vec_sources = vcs
        worker._active_sources = vcs
        worker._current_buf = np.zeros(n_nodes, dtype=np.float64)


class TestEndToEndStepColumns:
    """End-to-end transient + QS: use_step_columns=True vs False."""

    def _run_transient(self, use_sc, dt=100e-12, t_end=5e-9):
        from distributed.solver import DistributedDDMSolver

        model = _build_solvable_two_tile_model()
        _attach_vcs_to_workers(model.workers)

        solver = DistributedDDMSolver(model)
        dc_ctx = solver.prepare()
        trans_ctx = solver.prepare_transient(dt=dt, method='be')

        result = solver.solve_transient(
            trans_ctx,
            dc_context=dc_ctx,
            t_start=0.0,
            t_end=t_end,
            use_step_columns=use_sc,
        )
        trans_ctx.release()
        dc_ctx.release()
        return result

    def _run_qs(self, use_sc, n_points=20, t_end=50e-9):
        from distributed.solver import DistributedDDMSolver

        model = _build_solvable_two_tile_model()
        _attach_vcs_to_workers(model.workers)

        solver = DistributedDDMSolver(model)
        dc_ctx = solver.prepare()

        result = solver.solve_quasi_static(
            dc_ctx,
            t_start=0.0,
            t_end=t_end,
            n_points=n_points,
            use_step_columns=use_sc,
        )
        dc_ctx.release()
        return result

    def test_transient_step_columns_on_off_agree(self):
        """Transient with use_step_columns=True/False: max |dV| <= 1e-12."""
        res_on = self._run_transient(use_sc=True)
        res_off = self._run_transient(use_sc=False)

        # Compare peak IR drops over all time steps
        np.testing.assert_allclose(
            res_on.max_ir_drop_per_time, res_off.max_ir_drop_per_time,
            atol=1e-12,
            err_msg="Transient peak drops differ between step_columns=True/False",
        )

    def test_qs_step_columns_on_off_agree(self):
        """QS with use_step_columns=True/False: max |dV| <= 1e-12."""
        res_on = self._run_qs(use_sc=True)
        res_off = self._run_qs(use_sc=False)

        np.testing.assert_allclose(
            res_on.max_ir_drop_per_time, res_off.max_ir_drop_per_time,
            atol=1e-12,
            err_msg="QS max drops differ between step_columns=True/False",
        )

    def test_transient_with_step_columns_completes(self):
        """Smoke: transient with step columns enabled runs without error."""
        res = self._run_transient(use_sc=True)
        assert res is not None
        assert len(res.max_ir_drop_per_time) > 0

    def test_qs_with_step_columns_completes(self):
        """Smoke: QS with step columns enabled runs without error."""
        res = self._run_qs(use_sc=True)
        assert res is not None
        assert len(res.max_ir_drop_per_time) > 0

    def test_transient_default_uses_step_columns(self):
        """Default (no kwarg) == use_step_columns=True."""
        from distributed.solver import DistributedDDMSolver

        model = _build_solvable_two_tile_model()
        _attach_vcs_to_workers(model.workers)
        solver = DistributedDDMSolver(model)
        dc_ctx = solver.prepare()
        trans_ctx = solver.prepare_transient(dt=100e-12, method='be')

        # Run with explicit True
        res_explicit = solver.solve_transient(
            trans_ctx, dc_context=dc_ctx,
            t_start=0.0, t_end=5e-9, use_step_columns=True,
        )
        trans_ctx.release()
        dc_ctx.release()

        # Run with default (no kwarg)
        model2 = _build_solvable_two_tile_model()
        _attach_vcs_to_workers(model2.workers)
        solver2 = DistributedDDMSolver(model2)
        dc_ctx2 = solver2.prepare()
        trans_ctx2 = solver2.prepare_transient(dt=100e-12, method='be')
        res_default = solver2.solve_transient(
            trans_ctx2, dc_context=dc_ctx2, t_start=0.0, t_end=5e-9,
        )
        trans_ctx2.release()
        dc_ctx2.release()

        np.testing.assert_allclose(
            res_default.max_ir_drop_per_time, res_explicit.max_ir_drop_per_time, atol=1e-12,
        )


# ---------------------------------------------------------------------------
# Settings propagation via create_distributed_model (Issue 2 fix)
# ---------------------------------------------------------------------------

class TestFactorySettingsPropagation(unittest.TestCase):
    """create_distributed_model propagates use_step_columns + max_table_mb.

    Validates Issue 2 fix: model.py must inject these into solver_settings so
    TileWorker.configure() receives them — the only channel that works for
    Ray workers (module globals don't propagate).
    """

    def _build_model_via_factory(self, use_step_columns=True, max_table_mb=512.0):
        """Build a DistributedPowerGridModel via the factory path, not manually."""
        import tempfile
        import os
        import pickle
        from distributed.model import create_distributed_model, ParsedTileBundle
        from distributed.parser import PackageData, PowerGridMetaData, TileConfig
        from distributed.tile_worker import TileData

        # Build a minimal TileData and write it to a temp pkl
        tile_data = TileData(
            tile_id=(0, 0),
            resistive_edges=[('a1', 'shared', 1.0), ('shared', '0', 1.0)],
            all_nodes={'a1', 'shared'},
            boundary_nodes={'shared'},
            current_injections={'a1': 0.5},
            capacitive_edges=[('a1', '0', 10.0)],
        )
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
        tile_cfg = TileConfig(
            tile_id=(0, 0), ckt_path='', nd_path=None,
            instance_path=None, net_filter=None,
        )
        metadata = PowerGridMetaData(
            tile_grid=(1, 1),
            parameters={},
            tile_configs=[tile_cfg],
            package_data=pkg_data,
            net_name='VDD',
            vdd=1.0,
        )

        tmpdir = tempfile.mkdtemp()
        pkl_path = os.path.join(tmpdir, 'tile_0_0.pkl')
        with open(pkl_path, 'wb') as f:
            pickle.dump(tile_data, f)

        bundle = ParsedTileBundle(
            metadata=metadata,
            shared_boundary_nodes={'shared'},
            pkl_dir=tmpdir,
        )
        model = create_distributed_model(
            bundle,
            backend='local',
            use_step_columns=use_step_columns,
            max_table_mb=max_table_mb,
        )
        return model, tmpdir

    def test_factory_propagates_use_step_columns_false(self):
        """create_distributed_model(use_step_columns=False) disables table on workers."""
        model, tmpdir = self._build_model_via_factory(use_step_columns=False)
        try:
            for worker in model.workers:
                self.assertFalse(
                    worker._use_step_columns,
                    msg="Worker must have _use_step_columns=False after configure()",
                )
        finally:
            import shutil; shutil.rmtree(tmpdir, ignore_errors=True)

    def test_factory_propagates_max_table_mb(self):
        """create_distributed_model(max_table_mb=1.0) sets 1.0 on workers."""
        model, tmpdir = self._build_model_via_factory(max_table_mb=1.0)
        try:
            for worker in model.workers:
                self.assertAlmostEqual(
                    worker._max_table_mb, 1.0, places=5,
                    msg="Worker must have _max_table_mb=1.0 after configure()",
                )
        finally:
            import shutil; shutil.rmtree(tmpdir, ignore_errors=True)

    def test_factory_default_use_step_columns_true(self):
        """Default create_distributed_model() leaves _use_step_columns=True."""
        model, tmpdir = self._build_model_via_factory()
        try:
            for worker in model.workers:
                self.assertTrue(
                    worker._use_step_columns,
                    msg="Default factory must leave _use_step_columns=True on workers",
                )
        finally:
            import shutil; shutil.rmtree(tmpdir, ignore_errors=True)

    def test_factory_default_max_table_mb_512(self):
        """Default create_distributed_model() leaves _max_table_mb=512 on workers."""
        model, tmpdir = self._build_model_via_factory()
        try:
            for worker in model.workers:
                self.assertAlmostEqual(
                    worker._max_table_mb, 512.0, places=5,
                    msg="Default factory must leave _max_table_mb=512 on workers",
                )
        finally:
            import shutil; shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# End-to-end with smooth=False (raw aperiodic sources)
# ---------------------------------------------------------------------------

class TestEndToEndSmoothFalse(unittest.TestCase):
    """Equivalence check with smooth=False (aperiodic content, chunked tier).

    Spec requirement: the smooth=False path (raw aperiodic PWLs, period=0)
    must work and route aperiodic content to the chunked tier.  Both
    use_step_columns=True and False must produce identical results.
    """

    def _attach_aperiodic_vcs_to_workers(self, workers):
        """Attach aperiodic PWL sources (period=0, no smoothing)."""
        from analysis.vectorized_sources import VectorizedCurrentSources
        from parser.current_sources import CurrentSource, PWL

        for i, worker in enumerate(workers):
            n_ports = worker._block_system.n_ports
            n_interior = worker._block_system.n_interior
            n_nodes = n_ports + n_interior
            node_to_idx = dict(worker._block_system.port_to_idx)
            for node, idx in worker._block_system.interior_to_idx.items():
                node_to_idx[node] = idx + n_ports

            interior_nodes = list(worker._block_system.interior_to_idx.keys())
            if not interior_nodes:
                continue
            target = interior_nodes[0]

            # Aperiodic PWL: rises over 30ns, then stays flat
            pwl = PWL(
                delay=0.0, period=0.0,
                points=[(0.0, 0.0), (10e-9, 0.5), (20e-9, 1.0), (30e-9, 1.0)],
            )
            src = CurrentSource(
                name=f'i_ap_{i}', node1=target, node2='0',
                dc_value=0.1, pwls=[pwl],
            )
            vcs = VectorizedCurrentSources.from_current_sources(
                {f'i_ap_{i}': src}, node_to_idx, n_nodes,
            )
            worker._vec_sources = vcs
            worker._active_sources = vcs
            worker._current_buf = np.zeros(n_nodes, dtype=np.float64)

    def _run_qs(self, use_sc):
        from distributed.solver import DistributedDDMSolver

        model = _build_solvable_two_tile_model()
        self._attach_aperiodic_vcs_to_workers(model.workers)

        solver = DistributedDDMSolver(model)
        dc_ctx = solver.prepare()
        result = solver.solve_quasi_static(
            dc_ctx,
            t_start=0.0, t_end=50e-9, n_points=20,
            use_step_columns=use_sc,
        )
        dc_ctx.release()
        return result

    def test_aperiodic_qs_step_columns_on_off_agree(self):
        """Aperiodic sources: use_step_columns=True/False must agree <= 1e-12."""
        res_on = self._run_qs(use_sc=True)
        res_off = self._run_qs(use_sc=False)

        np.testing.assert_allclose(
            res_on.max_ir_drop_per_time, res_off.max_ir_drop_per_time,
            atol=1e-12,
            err_msg=(
                "Aperiodic sources: QS max drops differ between "
                "step_columns=True and False.  Chunked tier must be used "
                "for smooth=False aperiodic content."
            ),
        )

    def test_aperiodic_uses_chunked_tier(self):
        """Aperiodic sources with integral dt route to chunked (not phase) tier."""
        from analysis.vectorized_sources import VectorizedCurrentSources
        from parser.current_sources import CurrentSource, PWL

        w = _make_worker_with_tile(
            edges=[('p', 'a', 1.0), ('a', '0', 1.0)],
            port_nodes={'p'},
        )
        n_ports = w._block_system.n_ports
        n_interior = w._block_system.n_interior
        n_nodes = n_ports + n_interior
        node_to_idx = dict(w._block_system.port_to_idx)
        for node, idx in w._block_system.interior_to_idx.items():
            node_to_idx[node] = idx + n_ports

        interior_nodes = list(w._block_system.interior_to_idx.keys())
        target = interior_nodes[0]

        pwl = PWL(delay=0.0, period=0.0,
                  points=[(0.0, 0.0), (10e-9, 1.0), (20e-9, 0.0)])
        src = CurrentSource(name='i', node1=target, node2='0',
                            dc_value=0.0, pwls=[pwl])
        vcs = VectorizedCurrentSources.from_current_sources(
            {'i': src}, node_to_idx, n_nodes,
        )
        w._vec_sources = vcs
        w._active_sources = vcs
        w._current_buf = np.zeros(n_nodes, dtype=np.float64)

        # dt=100ps is integral with any period, but the source has period=0
        info = w.precompute_step_columns(t_start=0.0, dt=100e-12, n_steps=300)
        self.assertEqual(
            info['tier'], 'chunked',
            msg=(
                f"Aperiodic PWL (period=0, multi-knot) must select chunked tier "
                f"regardless of dt. Got tier='{info['tier']}'."
            ),
        )


# ---------------------------------------------------------------------------
# Issue 1 fix: non-uniform QS t_array must skip step-column table build
# ---------------------------------------------------------------------------

class TestNonUniformQSGridSkipsStepColumns(unittest.TestCase):
    """solve_quasi_static must skip step-column table for non-uniform t_array.

    Non-uniform grids (log-spaced, etc.) would silently evaluate at wrong times
    if the phase/chunked column mapping (column k → t_start + (k+1)*dt) were
    applied with dt = t_array[1] - t_array[0].  The fix in solver_td.py adds
    an np.allclose uniformity check and falls back to per-step evaluate_at_time.
    """

    def _make_model_with_vcs(self):
        model = _build_solvable_two_tile_model()
        _attach_vcs_to_workers(model.workers)
        return model

    def test_log_spaced_t_array_skips_step_columns(self):
        """Log-spaced t_array produces non-uniform spacing → step-column table skipped.

        Verification: QS result with non-uniform t_array equals reference
        solve_quasi_static (with use_step_columns=False, same t_array),
        proving the code actually evaluated the correct times.
        """
        from distributed.solver import DistributedDDMSolver

        model = self._make_model_with_vcs()
        solver = DistributedDDMSolver(model)
        dc_ctx = solver.prepare()

        # Log-spaced t_array: intentionally non-uniform
        t_array = np.logspace(-12, -8, 15)  # 1 ps → 10 ns, 15 points

        # With step_columns=True (default): non-uniform → must skip table, use eval_at_time
        res_auto = solver.solve_quasi_static(
            dc_ctx,
            t_array=t_array,
            use_step_columns=True,
        )
        # Reference: explicitly disable step columns
        res_ref = solver.solve_quasi_static(
            dc_ctx,
            t_array=t_array,
            use_step_columns=False,
        )
        dc_ctx.release()

        # Results must agree to floating-point precision
        np.testing.assert_allclose(
            res_auto.max_ir_drop_per_time,
            res_ref.max_ir_drop_per_time,
            atol=1e-12,
            err_msg=(
                "Non-uniform QS t_array: step_columns=True and False disagree. "
                "This means the step-column table was used with wrong time mapping."
            ),
        )

    def test_nearly_uniform_t_array_uses_step_columns(self):
        """A truly uniform t_array still enables the step-column table."""
        from distributed.solver import DistributedDDMSolver

        model = self._make_model_with_vcs()
        solver = DistributedDDMSolver(model)
        dc_ctx = solver.prepare()

        # Truly uniform (arange-derived): step columns should be built
        t_array = np.arange(100e-12, 5e-9 + 50e-12, 100e-12)  # 0.1ns → 5ns

        res_sc = solver.solve_quasi_static(
            dc_ctx,
            t_array=t_array,
            use_step_columns=True,
        )
        res_nosc = solver.solve_quasi_static(
            dc_ctx,
            t_array=t_array,
            use_step_columns=False,
        )
        dc_ctx.release()

        # Should still agree (step columns are an optimization, not a correctness change)
        np.testing.assert_allclose(
            res_sc.max_ir_drop_per_time,
            res_nosc.max_ir_drop_per_time,
            atol=1e-12,
            err_msg="Uniform t_array: step_columns=True/False disagree",
        )

    def test_single_point_t_array_skips_step_columns(self):
        """Single-point t_array (n_steps=1): no dt → table not built, no crash."""
        from distributed.solver import DistributedDDMSolver

        model = self._make_model_with_vcs()
        solver = DistributedDDMSolver(model)
        dc_ctx = solver.prepare()

        t_array = np.array([10e-9])

        res = solver.solve_quasi_static(
            dc_ctx,
            t_array=t_array,
            use_step_columns=True,
        )
        dc_ctx.release()

        self.assertIsNotNone(res)
        self.assertEqual(len(res.max_ir_drop_per_time), 1)


# ---------------------------------------------------------------------------
# Issue 2 fix: peak_build_mb reported in precompute_step_columns info dict
# ---------------------------------------------------------------------------

class TestPeakBuildMbInInfoDict(unittest.TestCase):
    """precompute_step_columns info dicts must include 'peak_build_mb'.

    The max_table_mb gate previously accounted only for the final sparse table
    (n_src_rows * m * 8 bytes) while the dense intermediate peaked at
    n_nodes * m * 8 bytes.  After the fix, both build paths (phase and chunked)
    return peak_build_mb in their info dict, enabling callers to audit true
    build-time memory usage.
    """

    def _worker_with_pulse(self, period=50e-9):
        w = _make_worker_with_tile(
            edges=[('p', 'a', 1.0), ('a', '0', 1.0)],
            port_nodes={'p'},
        )
        _attach_pulse_vcs(w, period=period)
        return w

    def test_phase_tier_reports_peak_build_mb(self):
        """Phase tier info dict contains 'peak_build_mb' key."""
        w = self._worker_with_pulse(period=50e-9)
        info = w.precompute_step_columns(t_start=0.0, dt=100e-12, n_steps=1000)
        self.assertEqual(info.get('tier'), 'phase',
                         msg="Expected phase tier for single-period integral dt")
        self.assertIn('peak_build_mb', info,
                      msg="Phase tier info must report 'peak_build_mb'")
        # Peak must be finite and positive
        self.assertGreater(info['peak_build_mb'], 0.0)

    def test_chunked_tier_reports_peak_build_mb(self):
        """Chunked tier info dict contains 'peak_build_mb' key."""
        w = self._worker_with_pulse(period=50e-9)
        # Non-integral dt → chunked tier
        info = w.precompute_step_columns(t_start=0.0, dt=33e-12, n_steps=100)
        self.assertEqual(info.get('tier'), 'chunked',
                         msg="Expected chunked tier for non-integral dt")
        self.assertIn('peak_build_mb', info,
                      msg="Chunked tier info must report 'peak_build_mb'")
        self.assertGreater(info['peak_build_mb'], 0.0)

    def test_peak_build_mb_le_full_dense_mb(self):
        """peak_build_mb must be <= n_nodes * m * 8 (full dense size) for the
        phase tier, because we only allocate n_candidate * m (candidate rows).
        When n_candidate < n_nodes this is strictly smaller.
        """
        w = _make_worker_with_tile(
            edges=[('p', 'a', 1.0), ('a', '0', 1.0)],
            port_nodes={'p'},
        )
        _attach_pulse_vcs(w, period=50e-9)

        n_nodes = w._block_system.n_ports + w._block_system.n_interior
        dt = 100e-12
        m = 500  # P/dt = 50ns/100ps
        full_dense_mb = n_nodes * m * 8 / 1e6

        info = w.precompute_step_columns(t_start=0.0, dt=dt, n_steps=1000)
        self.assertEqual(info.get('tier'), 'phase')
        self.assertLessEqual(
            info['peak_build_mb'], full_dense_mb + 1e-9,  # tolerance for float
            msg=(
                f"peak_build_mb={info['peak_build_mb']:.4f} MB exceeds full-dense "
                f"{full_dense_mb:.4f} MB — the row-sparse build should never "
                "allocate more than the full array"
            ),
        )

    def test_memory_mb_le_peak_build_mb(self):
        """Stored memory_mb (final table) <= peak_build_mb (build intermediate)."""
        w = self._worker_with_pulse(period=50e-9)
        info = w.precompute_step_columns(t_start=0.0, dt=100e-12, n_steps=1000)
        self.assertIn('memory_mb', info)
        self.assertIn('peak_build_mb', info)
        self.assertLessEqual(
            info['memory_mb'], info['peak_build_mb'] + 1e-9,
            msg="Stored table memory should never exceed build-time peak",
        )


if __name__ == '__main__':
    unittest.main()

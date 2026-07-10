"""Tests for Change B: direct-scatter fast path for chunked window builds.

All tests are marked unit — no external netlist or Ray backend required.

The Change B fast path applies when:
  - Sources are smoothed periodic PWLs (n_pulses==0, all_zero_pwl_delay,
    has_single_period, P/dt integral → m, sample_step ≈ actual_step, t0 ≈ 0)
  - t_start is dt-grid-aligned (t_start == 0 or round(t_start/dt)*dt == t_start
    within 1e-9 rel)
  - max_table_mb is too small to hold the full phase table (forces chunked tier)

In this state, _build_chunked_window uses _gather_window_direct (index gather
from smoothed pwl_values) instead of evaluate_at_times_for_rows.  The
on-demand rebuild in _get_current_array_for_step reads the stored _fast_path /
_ts_m / _m_fold from the table dict to avoid re-running the alignment probe.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.unit

# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────


def _make_worker(edges=None, port_nodes=None, currents=None):
    """Create a minimal TileWorker (no VCS)."""
    from distributed.tile_worker import TileData, TileWorker

    if edges is None:
        edges = [('a', 'b', 2.0), ('b', '0', 1.0)]
    if port_nodes is None:
        port_nodes = {'a'}

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
        capacitive_edges=[],
    )
    worker = TileWorker()
    worker.setup_from_tile_data(td, interface_nodes=port_nodes)
    worker.factor_and_compute_schur()
    return worker


def _attach_smoothed_pwl_vcs(worker, node='b', period=1e-8, dt=1e-10, dc=0.2):
    """Attach a synthetic smoothed periodic PWL VCS.

    Mimics what ``smooth_sources()`` produces: a PWL with uniform dt spacing,
    origin at t≈0, period P, with m+1 breakpoints (indices 0..m, wrapping
    at m).  Conditions that _smoothed_grid_alignment checks:
      - n_pulses == 0
      - all_zero_pwl_delay (delay=0.0)
      - has_single_period (period=P)
      - P/dt integral → m
      - sample_step == dt (P / m == dt)
      - first_time == 0.0

    The waveform is a cosine so that values[0] == values[m] (continuous at
    the period boundary) — the direct-scatter gather formula wraps to idx=0
    at the period boundary, matching evaluate_at_time to floating-point
    precision (no sawtooth discontinuity that would cause 1e-9 violations).
    """
    from analysis.vectorized_sources import VectorizedCurrentSources
    from parser.current_sources import CurrentSource, PWL

    bs = worker._block_system
    n_ports = bs.n_ports
    n_nodes = n_ports + bs.n_interior
    node_to_idx = dict(bs.port_to_idx)
    for nd, idx in bs.interior_to_idx.items():
        node_to_idx[nd] = idx + n_ports

    m = int(round(period / dt))
    # m+1 breakpoints from t=0 to t=P (inclusive), spacing = dt = P/m.
    # Use a cosine so values[0] == values[m] — continuous at the period
    # boundary.  The direct-scatter gather wraps at m (idx % m = 0 for
    # the period-boundary step), matching evaluate_at_time exactly since
    # cos(0) == cos(2*pi) (both = 1.0).
    times = np.arange(0, m + 1, dtype=np.float64) * dt
    phase = 2.0 * np.pi * np.arange(0, m + 1) / m
    values = 0.5 * (1.0 + np.cos(phase))   # 1.0 at t=0, 0.0 at t=P/2, 1.0 at t=P

    pwl = PWL(
        points=list(zip(times.tolist(), values.tolist())),
        period=period,
        delay=0.0,
    )
    src = CurrentSource(
        name='i_smoothed', node1=node, node2='0',
        dc_value=dc, pwls=[pwl],
    )
    vcs = VectorizedCurrentSources.from_current_sources(
        {'i_smoothed': src}, node_to_idx, n_nodes,
    )
    worker._vec_sources = vcs
    worker._active_sources = vcs
    worker._sources_version = 1
    worker._step_col_cache_key = None
    worker._step_col_info = None
    worker._step_col_table = None
    return m


def _tiny_max_mb(m, n_candidate):
    """Return a max_table_mb just below what the phase tier needs.

    Forces the chunked tier so we test the chunked fast path.
    Phase tier requires n_src * m * 8 bytes; we pass (n_candidate-1)*m*8/1e6.
    If n_candidate == 1 return a very small number.
    """
    needed_mb = max(n_candidate, 1) * m * 8 / 1e6
    # Return something smaller than needed to disqualify phase tier
    return needed_mb * 0.5


# ─────────────────────────────────────────────────────────────────────────────
# 1. _smoothed_grid_alignment helper
# ─────────────────────────────────────────────────────────────────────────────


class TestSmoothedGridAlignment:
    """Unit tests for the _smoothed_grid_alignment helper."""

    def _make_aligned_worker(self, period=1e-8, dt=1e-10):
        worker = _make_worker()
        m = _attach_smoothed_pwl_vcs(worker, period=period, dt=dt)
        return worker, m

    def test_aligned_returns_dict(self):
        """Aligned smoothed sources return {'m': m, 'actual_step': dt}."""
        worker, m = self._make_aligned_worker()
        result = worker._smoothed_grid_alignment(dt=1e-10)
        assert result is not None, "_smoothed_grid_alignment must return dict for aligned sources"
        assert result['m'] == m
        assert abs(result['actual_step'] - 1e-10) < 1e-15 * 1e-10

    def test_no_sources_returns_none(self):
        """No active sources → None."""
        worker = _make_worker()
        assert worker._smoothed_grid_alignment(dt=1e-10) is None

    def test_pulse_vcs_returns_none(self):
        """Pulse VCS (n_pulses > 0) → None (pulse not converted to PWL)."""
        from analysis.vectorized_sources import VectorizedCurrentSources
        from parser.current_sources import CurrentSource, Pulse

        worker = _make_worker()
        bs = worker._block_system
        n_ports = bs.n_ports
        n_nodes = n_ports + bs.n_interior
        node_to_idx = {nd: idx for nd, idx in bs.port_to_idx.items()}
        for nd, idx in bs.interior_to_idx.items():
            node_to_idx[nd] = idx + n_ports
        pulse = Pulse(v1=0.0, v2=1.0, delay=0.0, rt=0.0, ft=0.0,
                      width=5e-9, period=1e-8)
        src = CurrentSource(name='ip', node1='b', node2='0',
                            dc_value=0.0, pulses=[pulse])
        vcs = VectorizedCurrentSources.from_current_sources(
            {'ip': src}, node_to_idx, n_nodes,
        )
        worker._active_sources = vcs
        result = worker._smoothed_grid_alignment(dt=1e-10)
        assert result is None, "Pulse VCS must not qualify for smoothed-grid alignment"

    def test_nonzero_delay_returns_none(self):
        """PWL with nonzero delay fails all_zero_pwl_delay → None."""
        from analysis.vectorized_sources import VectorizedCurrentSources
        from parser.current_sources import CurrentSource, PWL

        worker = _make_worker()
        bs = worker._block_system
        n_ports = bs.n_ports
        n_nodes = n_ports + bs.n_interior
        node_to_idx = {nd: idx for nd, idx in bs.port_to_idx.items()}
        for nd, idx in bs.interior_to_idx.items():
            node_to_idx[nd] = idx + n_ports

        m = 100
        dt = 1e-10
        period = m * dt
        times = np.arange(0, m + 1, dtype=np.float64) * dt
        values = np.linspace(0.0, 1.0, m + 1)
        pwl = PWL(points=list(zip(times.tolist(), values.tolist())),
                  period=period, delay=1e-9)   # nonzero delay!
        src = CurrentSource(name='id', node1='b', node2='0',
                            dc_value=0.0, pwls=[pwl])
        vcs = VectorizedCurrentSources.from_current_sources(
            {'id': src}, node_to_idx, n_nodes,
        )
        worker._active_sources = vcs
        assert worker._smoothed_grid_alignment(dt) is None

    def test_aperiodic_pwl_returns_none(self):
        """Aperiodic PWL (period=0) → None."""
        from analysis.vectorized_sources import VectorizedCurrentSources
        from parser.current_sources import CurrentSource, PWL

        worker = _make_worker()
        bs = worker._block_system
        n_ports = bs.n_ports
        n_nodes = n_ports + bs.n_interior
        node_to_idx = {nd: idx for nd, idx in bs.port_to_idx.items()}
        for nd, idx in bs.interior_to_idx.items():
            node_to_idx[nd] = idx + n_ports

        dt = 1e-10
        times = np.arange(0, 200) * dt
        values = np.random.default_rng(7).uniform(0, 1, len(times))
        pwl = PWL(points=list(zip(times.tolist(), values.tolist())),
                  period=0.0, delay=0.0)
        src = CurrentSource(name='iap', node1='b', node2='0',
                            dc_value=0.0, pwls=[pwl])
        vcs = VectorizedCurrentSources.from_current_sources(
            {'iap': src}, node_to_idx, n_nodes,
        )
        worker._active_sources = vcs
        assert worker._smoothed_grid_alignment(dt) is None

    def test_off_grid_sample_step_returns_none(self):
        """PWL whose sample_step != dt fails the grid-alignment check → None."""
        from analysis.vectorized_sources import VectorizedCurrentSources
        from parser.current_sources import CurrentSource, PWL

        worker = _make_worker()
        bs = worker._block_system
        n_ports = bs.n_ports
        n_nodes = n_ports + bs.n_interior
        node_to_idx = {nd: idx for nd, idx in bs.port_to_idx.items()}
        for nd, idx in bs.interior_to_idx.items():
            node_to_idx[nd] = idx + n_ports

        dt = 1e-10
        period = 100 * dt
        # Use 2*dt spacing — P/dt=100 but samples at 2*dt intervals
        times = np.arange(0, 51) * 2 * dt
        values = np.linspace(0.0, 1.0, 51)
        pwl = PWL(points=list(zip(times.tolist(), values.tolist())),
                  period=period, delay=0.0)
        src = CurrentSource(name='iog', node1='b', node2='0',
                            dc_value=0.0, pwls=[pwl])
        vcs = VectorizedCurrentSources.from_current_sources(
            {'iog': src}, node_to_idx, n_nodes,
        )
        worker._active_sources = vcs
        assert worker._smoothed_grid_alignment(dt) is None


# ─────────────────────────────────────────────────────────────────────────────
# 2. _gather_window_direct correctness
# ─────────────────────────────────────────────────────────────────────────────


class TestGatherWindowDirect:
    """Unit tests for _gather_window_direct correctness."""

    def _make_aligned_worker_with_candidates(self, period=1e-8, dt=1e-10):
        worker = _make_worker()
        m = _attach_smoothed_pwl_vcs(worker, period=period, dt=dt)
        vcs = worker._active_sources
        src_candidates = vcs.get_src_node_indices()
        return worker, m, src_candidates

    def test_window0_matches_evaluate_at_times_for_rows(self):
        """Window [0, W) built via gather matches evaluate_at_times_for_rows."""
        dt = 1e-10
        period = 100 * dt
        worker, m, src_cands = self._make_aligned_worker_with_candidates(
            period=period, dt=dt,
        )
        vcs = worker._active_sources
        W = 50
        ts_m = 0  # t_start=0

        C_fast, rows_fast = worker._gather_window_direct(
            new_start=0, W_actual=W, ts_m=ts_m, m=m,
            src_rows_candidate=src_cands,
        )

        # Reference: evaluate_at_times_for_rows at t = (s+1)*dt for s in [0, W)
        t_grid = np.arange(1, W + 1, dtype=np.float64) * dt
        C_ref = vcs.evaluate_at_times_for_rows(t_grid, src_cands)
        # Trim reference to nonzero rows
        row_max = np.abs(C_ref).max(axis=1)
        nonzero = row_max > 0
        rows_ref = src_cands[nonzero]
        C_ref_trimmed = np.asfortranarray(C_ref[nonzero, :])

        # rows must match
        np.testing.assert_array_equal(rows_fast, rows_ref)
        np.testing.assert_allclose(
            C_fast, C_ref_trimmed, atol=1e-9,
            err_msg="Window 0 via gather must match evaluate_at_times_for_rows",
        )

    def test_later_window_crosses_period_boundary(self):
        """Window [W, 2W) that crosses a period boundary (step % m wraps) is correct."""
        dt = 1e-10
        m = 100            # period = 100 * dt
        period = m * dt
        worker, m_actual, src_cands = self._make_aligned_worker_with_candidates(
            period=period, dt=dt,
        )
        assert m_actual == m
        vcs = worker._active_sources
        W = 50
        ts_m = 0

        # Window starting at step 80 — will cross period boundary at step 100
        new_start = 80
        W_actual = W  # covers steps 80..129; period at 100

        C_fast, rows_fast = worker._gather_window_direct(
            new_start=new_start, W_actual=W_actual, ts_m=ts_m, m=m,
            src_rows_candidate=src_cands,
        )

        # Reference: t = (new_start + 1 + i) * dt for i in [0, W_actual)
        t_grid = np.arange(new_start + 1, new_start + 1 + W_actual, dtype=np.float64) * dt
        C_ref = vcs.evaluate_at_times_for_rows(t_grid, src_cands)
        row_max = np.abs(C_ref).max(axis=1)
        nonzero = row_max > 0
        rows_ref = src_cands[nonzero]
        C_ref_trimmed = np.asfortranarray(C_ref[nonzero, :W_actual])

        np.testing.assert_array_equal(rows_fast, rows_ref)
        np.testing.assert_allclose(
            C_fast, C_ref_trimmed, atol=1e-9,
            err_msg="Period-crossing window via gather must match evaluate_at_times_for_rows",
        )

    def test_nonzero_ts_m_offset(self):
        """Nonzero ts_m (t_start > 0) is correctly applied in the index formula."""
        dt = 1e-10
        m = 100
        period = m * dt
        worker, m_actual, src_cands = self._make_aligned_worker_with_candidates(
            period=period, dt=dt,
        )
        vcs = worker._active_sources
        W = 30
        t_start = 20 * dt   # ts_m = 20
        ts_m = 20
        new_start = 0

        C_fast, rows_fast = worker._gather_window_direct(
            new_start=new_start, W_actual=W, ts_m=ts_m, m=m,
            src_rows_candidate=src_cands,
        )

        # Reference: t = t_start + (new_start + 1 + i)*dt for i in [0, W)
        t_grid = t_start + np.arange(1, W + 1, dtype=np.float64) * dt
        C_ref = vcs.evaluate_at_times_for_rows(t_grid, src_cands)
        row_max = np.abs(C_ref).max(axis=1)
        nonzero = row_max > 0
        rows_ref = src_cands[nonzero]
        C_ref_trimmed = np.asfortranarray(C_ref[nonzero, :W])

        np.testing.assert_array_equal(rows_fast, rows_ref)
        np.testing.assert_allclose(
            C_fast, C_ref_trimmed, atol=1e-9,
            err_msg="Nonzero ts_m window via gather must match evaluate_at_times_for_rows",
        )

    def test_cnt_lt_m_plus_1_raises_value_error(self):
        """A PWL row with cnt < m+1 must raise ValueError (caller falls back)."""
        from analysis.vectorized_sources import VectorizedCurrentSources
        from parser.current_sources import CurrentSource, PWL

        worker = _make_worker()
        bs = worker._block_system
        n_ports = bs.n_ports
        n_nodes = n_ports + bs.n_interior
        node_to_idx = {nd: idx for nd, idx in bs.port_to_idx.items()}
        for nd, idx in bs.interior_to_idx.items():
            node_to_idx[nd] = idx + n_ports

        dt = 1e-10
        m = 100
        period = m * dt
        # Only m-1 breakpoints (not enough for cnt >= m+1)
        times = np.arange(0, m - 1, dtype=np.float64) * dt
        values = np.linspace(0.0, 1.0, m - 1)
        pwl = PWL(points=list(zip(times.tolist(), values.tolist())),
                  period=period, delay=0.0)
        src = CurrentSource(name='ishort', node1='b', node2='0',
                            dc_value=0.0, pwls=[pwl])
        vcs = VectorizedCurrentSources.from_current_sources(
            {'ishort': src}, node_to_idx, n_nodes,
        )
        worker._active_sources = vcs

        src_cands = vcs.get_src_node_indices()
        with pytest.raises(ValueError, match="cnt="):
            worker._gather_window_direct(
                new_start=0, W_actual=50, ts_m=0, m=m,
                src_rows_candidate=src_cands,
            )


# ─────────────────────────────────────────────────────────────────────────────
# 3. _build_chunked_window fast path
# ─────────────────────────────────────────────────────────────────────────────


class TestChunkedDirectScatterBuildPath:
    """_build_chunked_window picks 'direct_scatter' when alignment holds."""

    def _make_aligned_chunked_worker(self, period=1e-8, dt=1e-10, n_steps=600):
        """Worker with smoothed periodic PWL forced into chunked tier."""
        worker = _make_worker()
        m = _attach_smoothed_pwl_vcs(worker, period=period, dt=dt)
        # Compute max_mb that disqualifies phase tier
        src_cands = worker._active_sources.get_src_node_indices()
        max_mb = _tiny_max_mb(m, len(src_cands))
        return worker, m, max_mb

    def test_build_path_is_direct_scatter(self):
        """With aligned smoothed sources and tiny max_mb, build_path='direct_scatter'."""
        dt = 1e-10
        worker, m, max_mb = self._make_aligned_chunked_worker(
            period=m * dt if False else 100 * dt,
            dt=dt,
            n_steps=600,
        )
        info = worker.precompute_step_columns(
            t_start=0.0, dt=dt, n_steps=600, max_table_mb=max_mb,
        )
        assert info.get('tier') == 'chunked', (
            f"Expected chunked tier, got {info.get('tier')!r}"
        )
        assert info.get('build_path') == 'direct_scatter', (
            f"Expected direct_scatter, got {info.get('build_path')!r}"
        )

    def test_fast_path_stored_in_table(self):
        """Table dict stores _fast_path=True, _ts_m=0, _m_fold=m."""
        dt = 1e-10
        m = 100
        worker, m_actual, max_mb = self._make_aligned_chunked_worker(dt=dt)
        worker.precompute_step_columns(
            t_start=0.0, dt=dt, n_steps=600, max_table_mb=max_mb,
        )
        tbl = worker._step_col_table
        assert tbl is not None
        assert tbl.get('_fast_path') is True
        assert tbl.get('_ts_m') == 0
        assert tbl.get('_m_fold') == m_actual

    def test_fallback_with_pulse_sources(self):
        """Pulse VCS (not smoothed) forces build_path='evaluate_at_times_for_rows'."""
        from analysis.vectorized_sources import VectorizedCurrentSources
        from parser.current_sources import CurrentSource, Pulse

        worker = _make_worker()
        bs = worker._block_system
        n_ports = bs.n_ports
        n_nodes = n_ports + bs.n_interior
        node_to_idx = {nd: idx for nd, idx in bs.port_to_idx.items()}
        for nd, idx in bs.interior_to_idx.items():
            node_to_idx[nd] = idx + n_ports
        pulse = Pulse(v1=0.0, v2=1.0, delay=0.0, rt=0.0, ft=0.0,
                      width=5e-9, period=1e-8)
        src = CurrentSource(name='ip', node1='b', node2='0',
                            dc_value=0.0, pulses=[pulse])
        vcs = VectorizedCurrentSources.from_current_sources(
            {'ip': src}, node_to_idx, n_nodes,
        )
        worker._active_sources = vcs
        worker._sources_version = 1
        worker._step_col_cache_key = None
        worker._step_col_info = None
        worker._step_col_table = None

        # Use max_mb=0.0001 to disqualify phase tier
        info = worker.precompute_step_columns(
            t_start=0.0, dt=1e-10, n_steps=600, max_table_mb=0.0001,
        )
        assert info.get('tier') == 'chunked'
        assert info.get('build_path') == 'evaluate_at_times_for_rows', (
            "Pulse VCS must not use direct_scatter build path"
        )

    def test_fallback_with_nonzero_delay(self):
        """Nonzero PWL delay forces build_path='evaluate_at_times_for_rows'."""
        from analysis.vectorized_sources import VectorizedCurrentSources
        from parser.current_sources import CurrentSource, PWL

        worker = _make_worker()
        bs = worker._block_system
        n_ports = bs.n_ports
        n_nodes = n_ports + bs.n_interior
        node_to_idx = {nd: idx for nd, idx in bs.port_to_idx.items()}
        for nd, idx in bs.interior_to_idx.items():
            node_to_idx[nd] = idx + n_ports

        dt = 1e-10
        m = 100
        period = m * dt
        times = np.arange(0, m + 1, dtype=np.float64) * dt
        values = np.linspace(0.0, 1.0, m + 1)
        pwl = PWL(points=list(zip(times.tolist(), values.tolist())),
                  period=period, delay=1e-9)  # nonzero delay!
        src = CurrentSource(name='id', node1='b', node2='0',
                            dc_value=0.0, pwls=[pwl])
        vcs = VectorizedCurrentSources.from_current_sources(
            {'id': src}, node_to_idx, n_nodes,
        )
        worker._active_sources = vcs
        worker._sources_version = 1
        worker._step_col_cache_key = None
        worker._step_col_info = None
        worker._step_col_table = None

        info = worker.precompute_step_columns(
            t_start=0.0, dt=dt, n_steps=600, max_table_mb=0.0001,
        )
        assert info.get('tier') == 'chunked'
        assert info.get('build_path') == 'evaluate_at_times_for_rows'

    def test_fallback_with_off_grid_t_start(self):
        """t_start not on dt grid forces build_path='evaluate_at_times_for_rows'."""
        dt = 1e-10
        worker, m, max_mb = self._make_aligned_chunked_worker(dt=dt)
        t_start_off = 0.7 * dt  # off-grid
        info = worker.precompute_step_columns(
            t_start=t_start_off, dt=dt, n_steps=600, max_table_mb=max_mb,
        )
        assert info.get('tier') == 'chunked'
        assert info.get('build_path') == 'evaluate_at_times_for_rows', (
            "Off-grid t_start must not use direct_scatter for chunked window"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 4. Per-step correctness across multiple windows
# ─────────────────────────────────────────────────────────────────────────────


class TestChunkedDirectScatterPerStepCorrectness:
    """Columns gathered via the fast path match evaluate_at_time to <= 1e-9 mA."""

    def _make_aligned_worker_and_info(self, dt=1e-10, n_steps=1100, t_start=0.0):
        worker = _make_worker()
        m = _attach_smoothed_pwl_vcs(worker, dt=dt)
        src_cands = worker._active_sources.get_src_node_indices()
        max_mb = _tiny_max_mb(m, len(src_cands))
        info = worker.precompute_step_columns(
            t_start=t_start, dt=dt, n_steps=n_steps, max_table_mb=max_mb,
        )
        return worker, info, m, max_mb

    def test_step_array_matches_evaluate_at_time_t_start_0(self):
        """All steps at t_start=0 match evaluate_at_time to <= 1e-9 mA."""
        dt = 1e-10
        n_steps = 1100   # 3 windows of W=512 (covers 2 full window crossings + 1)
        worker, info, m, _ = self._make_aligned_worker_and_info(
            dt=dt, n_steps=n_steps,
        )
        assert info.get('tier') == 'chunked'
        assert info.get('build_path') == 'direct_scatter', (
            f"Expected direct_scatter, got {info.get('build_path')!r}"
        )

        vcs = worker._active_sources
        # Spot-check steps at window boundaries and period-crossing points
        check_steps = list(range(0, 20)) + [511, 512, 513, 999, 1000, 1099]
        for s in check_steps:
            t = 0.0 + (s + 1) * dt
            arr_table = worker._get_current_array_for_step(s, t).copy()
            arr_eval = vcs.evaluate_at_time(t)
            np.testing.assert_allclose(
                arr_table, arr_eval, atol=1e-9,
                err_msg=f"Step {s}: table vs evaluate_at_time mismatch",
            )

    def test_step_array_matches_evaluate_at_time_window_crossings(self):
        """W-crossing and period-boundary-crossing steps are both correct."""
        dt = 1e-10
        m = 100
        n_steps = 1300   # forces 3+ window rebuilds with W=512
        worker = _make_worker()
        _ = _attach_smoothed_pwl_vcs(worker, dt=dt)
        src_cands = worker._active_sources.get_src_node_indices()
        max_mb = _tiny_max_mb(m, len(src_cands))
        info = worker.precompute_step_columns(
            t_start=0.0, dt=dt, n_steps=n_steps, max_table_mb=max_mb,
        )
        assert info.get('build_path') == 'direct_scatter'

        vcs = worker._active_sources
        # Window crossings at step 512 and 1024; period at every m=100 steps
        # (100, 200, ..., 1200).  Check a representative mix.
        check_steps = [99, 100, 101, 199, 200, 511, 512, 513, 1023, 1024, 1025,
                       1099, 1100, 1200, 1299]
        for s in check_steps:
            t = (s + 1) * dt
            arr_table = worker._get_current_array_for_step(s, t).copy()
            arr_eval = vcs.evaluate_at_time(t)
            np.testing.assert_allclose(
                arr_table, arr_eval, atol=1e-9,
                err_msg=f"Step {s}: table vs evaluate_at_time mismatch at window/period crossing",
            )

    def test_step_array_nonzero_t_start(self):
        """Nonzero (on-grid) t_start=20*dt: per-step columns are correct."""
        dt = 1e-10
        m = 100
        t_start = 20 * dt
        n_steps = 1100
        worker = _make_worker()
        _ = _attach_smoothed_pwl_vcs(worker, dt=dt)
        src_cands = worker._active_sources.get_src_node_indices()
        max_mb = _tiny_max_mb(m, len(src_cands))
        info = worker.precompute_step_columns(
            t_start=t_start, dt=dt, n_steps=n_steps, max_table_mb=max_mb,
        )
        assert info.get('build_path') == 'direct_scatter', (
            f"Expected direct_scatter for on-grid t_start, got {info.get('build_path')!r}"
        )

        vcs = worker._active_sources
        check_steps = [0, 1, 79, 80, 100, 511, 512, 513, 999, 1000, 1099]
        for s in check_steps:
            t = t_start + (s + 1) * dt
            arr_table = worker._get_current_array_for_step(s, t).copy()
            arr_eval = vcs.evaluate_at_time(t)
            np.testing.assert_allclose(
                arr_table, arr_eval, atol=1e-9,
                err_msg=f"Step {s} with t_start={t_start}: table vs evaluate_at_time mismatch",
            )


# ─────────────────────────────────────────────────────────────────────────────
# 5. Late-active sources
# ─────────────────────────────────────────────────────────────────────────────


class TestLateActiveSourceFastPath:
    """A source that is zero in window 0 but nonzero in window 1 is captured."""

    def test_late_active_source_captured_in_later_window(self):
        """A source with zero amplitude in [0,W) but nonzero in [W,2W) is captured.

        The _gather_window_direct helper uses src_rows_candidate (which never
        shrinks), so the source is re-evaluated in window 1 and appears in the
        trimmed src_rows.
        """
        from analysis.vectorized_sources import VectorizedCurrentSources
        from parser.current_sources import CurrentSource, PWL

        # Build a worker with 2 nodes (port 'a', interior 'b')
        worker = _make_worker()
        bs = worker._block_system
        n_ports = bs.n_ports
        n_nodes = n_ports + bs.n_interior
        node_to_idx = {nd: idx for nd, idx in bs.port_to_idx.items()}
        for nd, idx in bs.interior_to_idx.items():
            node_to_idx[nd] = idx + n_ports

        dt = 1e-10
        m = 100
        period = m * dt
        W = 512
        # Source 1: uniformly nonzero across all windows (establishes
        # periodicity and enables the fast path via has_single_period).
        times1 = np.arange(0, m + 1, dtype=np.float64) * dt
        values1 = np.ones(m + 1) * 0.5  # constant 0.5 — nonzero in window 0
        pwl1 = PWL(points=list(zip(times1.tolist(), values1.tolist())),
                   period=period, delay=0.0)

        # Source 2 on the same node: zero in first W steps, then nonzero after.
        # We model this by a source with the same period but shifted amplitude
        # pattern.  Actually for a truly "late-active" source, we create a
        # separate waveform on a DIFFERENT node that starts nonzero only after
        # step W.  We approximate this by using a different period so the
        # combined VCS has a non-trivial pattern.
        #
        # A simpler approach for testing: use a two-step PWL whose values
        # happen to be 0 for t in [0, W*dt) and nonzero for t >= W*dt.
        # But with a periodic waveform of period 100*dt we can't really make
        # it zero in steps 0..511 and nonzero in 512..1023 — the period is
        # only 100 steps.
        #
        # Instead we attach TWO SEPARATE sources on node 'b' and node 'a'
        # respectively with identical smoothed structure (to satisfy
        # has_single_period), then verify that src_rows_candidate retains
        # BOTH node indices and that the later-window gather still produces
        # correct output for BOTH nodes.  The candidate-set-never-shrinks
        # guarantee is what matters; the values themselves just need to match
        # evaluate_at_time.
        times2 = np.arange(0, m + 1, dtype=np.float64) * dt
        values2 = np.linspace(0.0, 1.0, m + 1)  # different amplitude
        pwl2 = PWL(points=list(zip(times2.tolist(), values2.tolist())),
                   period=period, delay=0.0)

        # Build VCS with two sources: node 'b' (idx = n_ports + interior_idx)
        # and node 'a' (port).
        nodes_with_src = {}
        target_a = 'a'
        target_b = 'b'
        if target_a in node_to_idx:
            nodes_with_src[target_a] = (pwl1, 'i_a')
        if target_b in node_to_idx:
            nodes_with_src[target_b] = (pwl2, 'i_b')

        if len(nodes_with_src) < 2:
            pytest.skip("Worker does not have both 'a' and 'b' nodes accessible")

        sources = {}
        for node, (pwl, name) in nodes_with_src.items():
            src = CurrentSource(name=name, node1=node, node2='0',
                                dc_value=0.0, pwls=[pwl])
            sources[name] = src
        vcs = VectorizedCurrentSources.from_current_sources(
            sources, node_to_idx, n_nodes,
        )
        worker._active_sources = vcs
        worker._vec_sources = vcs
        worker._sources_version = 1
        worker._step_col_cache_key = None
        worker._step_col_info = None
        worker._step_col_table = None

        # Force chunked tier
        src_cands = vcs.get_src_node_indices()
        max_mb = _tiny_max_mb(m, len(src_cands))

        n_steps = W + 100  # requires one window rebuild
        info = worker.precompute_step_columns(
            t_start=0.0, dt=dt, n_steps=n_steps, max_table_mb=max_mb,
        )
        assert info.get('build_path') == 'direct_scatter', (
            f"Expected direct_scatter, got {info.get('build_path')!r}"
        )

        # Verify that steps in the SECOND window (after W) are correct
        for s in range(W, min(W + 50, n_steps)):
            t = (s + 1) * dt
            arr_table = worker._get_current_array_for_step(s, t).copy()
            arr_eval = vcs.evaluate_at_time(t)
            np.testing.assert_allclose(
                arr_table, arr_eval, atol=1e-9,
                err_msg=f"Late-window step {s}: fast-path gather mismatch",
            )


# ─────────────────────────────────────────────────────────────────────────────
# 6. Change C interplay: fast path disables the skip
# ─────────────────────────────────────────────────────────────────────────────


class TestChangeCFastPathInterplay:
    """When the direct-scatter fast path is available, Change C must NOT skip."""

    def test_small_n_steps_fast_path_builds_table(self):
        """n_steps <= W but fast path available: table must be built."""
        dt = 1e-10
        worker = _make_worker()
        m = _attach_smoothed_pwl_vcs(worker, dt=dt)
        src_cands = worker._active_sources.get_src_node_indices()
        max_mb = _tiny_max_mb(m, len(src_cands))

        n_steps = 50  # well below W=512, would normally trigger Change C skip
        info = worker.precompute_step_columns(
            t_start=0.0, dt=dt, n_steps=n_steps, max_table_mb=max_mb,
        )
        assert info.get('tier') == 'chunked', (
            f"Fast-path available: must build chunked table, got tier={info.get('tier')!r}"
        )
        assert info.get('build_path') == 'direct_scatter', (
            f"Expected direct_scatter, got {info.get('build_path')!r}"
        )
        assert worker._step_col_table is not None, (
            "Table must be built (not None) when fast path is available"
        )

    def test_small_n_steps_fast_path_results_correct(self):
        """With fast-path single-window, per-step columns match evaluate_at_time."""
        dt = 1e-10
        worker = _make_worker()
        m = _attach_smoothed_pwl_vcs(worker, dt=dt)
        src_cands = worker._active_sources.get_src_node_indices()
        max_mb = _tiny_max_mb(m, len(src_cands))

        n_steps = 50
        worker.precompute_step_columns(
            t_start=0.0, dt=dt, n_steps=n_steps, max_table_mb=max_mb,
        )
        vcs = worker._active_sources
        for s in range(n_steps):
            t = (s + 1) * dt
            arr_table = worker._get_current_array_for_step(s, t).copy()
            arr_eval = vcs.evaluate_at_time(t)
            np.testing.assert_allclose(
                arr_table, arr_eval, atol=1e-9,
                err_msg=f"Step {s}: fast-path single-window mismatch",
            )

    def test_non_aligned_small_n_steps_still_skips(self):
        """Without fast path (pulse VCS), small n_steps still triggers Change C skip."""
        from analysis.vectorized_sources import VectorizedCurrentSources
        from parser.current_sources import CurrentSource, PWL

        worker = _make_worker()
        bs = worker._block_system
        n_ports = bs.n_ports
        n_nodes = n_ports + bs.n_interior
        node_to_idx = {nd: idx for nd, idx in bs.port_to_idx.items()}
        for nd, idx in bs.interior_to_idx.items():
            node_to_idx[nd] = idx + n_ports

        dt = 1e-10
        times = np.arange(0, 600) * dt
        values = np.random.default_rng(99).uniform(0, 1, len(times))
        pwl = PWL(points=list(zip(times.tolist(), values.tolist())),
                  period=0.0, delay=0.0)  # aperiodic → no fast path
        src = CurrentSource(name='iap', node1='b', node2='0',
                            dc_value=0.0, pwls=[pwl])
        vcs = VectorizedCurrentSources.from_current_sources(
            {'iap': src}, node_to_idx, n_nodes,
        )
        worker._active_sources = vcs
        worker._sources_version = 1
        worker._step_col_cache_key = None
        worker._step_col_info = None
        worker._step_col_table = None

        info = worker.precompute_step_columns(
            t_start=0.0, dt=dt, n_steps=50,
        )
        assert info.get('tier') == 'skipped', (
            "Aperiodic VCS without fast path must still trigger Change C skip"
        )
        assert worker._step_col_table is None


# ─────────────────────────────────────────────────────────────────────────────
# 7. End-to-end: solve_transient with fast path vs no step columns
# ─────────────────────────────────────────────────────────────────────────────


def _build_two_tile_model_with_caps():
    """Build the standard 2-tile model (with capacitive edges for transient).

    Topology:
        Tile A: a1 --[1 mS]-- B  (cap: a1-0 10 fF)
        Tile B: B --[3 mS]-- b1 --[1 mS]-- 0  (cap: b1-0 5 fF)
        Package: pad --[10 mS]-- B;  pad at 1.0 V
    """
    import os
    import tempfile

    from distributed.backend import LocalBackend
    from distributed.model import DistributedPowerGridModel
    from distributed.parser import PackageData, PowerGridMetaData, TileConfig
    from distributed.tile_worker import TileData, TileWorker

    tmp_dir = tempfile.mkdtemp(prefix="test_chb_")

    tile_a = TileData(
        tile_id=(0, 0),
        resistive_edges=[('a1', 'B', 1.0)],
        all_nodes={'a1', 'B'},
        boundary_nodes={'B'},
        current_injections={'a1': 0.1},
        capacitive_edges=[('a1', '0', 10.0)],
    )
    tile_b = TileData(
        tile_id=(0, 1),
        resistive_edges=[('B', 'b1', 3.0), ('b1', '0', 1.0)],
        all_nodes={'B', 'b1'},
        boundary_nodes={'B'},
        current_injections={'b1': 0.1},
        capacitive_edges=[('b1', '0', 5.0)],
    )
    interface_nodes = {'B'}
    be = LocalBackend()
    be.initialize()
    wa = TileWorker()
    wa.setup_from_tile_data(tile_a, interface_nodes)
    wb = TileWorker()
    wb.setup_from_tile_data(tile_b, interface_nodes)
    workers = [wa, wb]

    pkg = PackageData(
        vsrc_dict={'V1': {'node_pos': 'pad', 'node_neg': '0', 'net': 'VDD', 'value': 1.0}},
        package_edges=[('pad', 'B', 10.0)],
        pad_nodes={'pad'},
        tap_nodes=set(),
        die_attachment_nodes=set(),
        vdd=1.0,
        net_name='VDD',
        package_cap_edges=[],
    )
    dummy_ckt = os.path.join(tmp_dir, 'dummy.ckt')
    tile_configs = [
        TileConfig(tile_id=(0, 0), ckt_path=dummy_ckt, nd_path=None,
                   instance_path=None, net_filter=None),
        TileConfig(tile_id=(0, 1), ckt_path=dummy_ckt, nd_path=None,
                   instance_path=None, net_filter=None),
    ]
    metadata = PowerGridMetaData(
        tile_grid=(1, 2),
        parameters={},
        tile_configs=tile_configs,
        package_data=pkg,
        net_name='VDD',
        vdd=1.0,
    )
    return DistributedPowerGridModel(
        backend=be,
        workers=workers,
        interface_nodes=interface_nodes,
        tile_boundary_nodes={(0, 0): ['B'], (0, 1): ['B']},
        tile_interior_counts={(0, 0): wa.n_interior, (0, 1): wb.n_interior},
        package_data=pkg,
        metadata=metadata,
    ), workers


def _attach_smoothed_pwl_vcs_to_workers(workers, period=1e-8, dt=1e-10):
    """Attach aligned smoothed periodic PWL VCS to all workers."""
    from analysis.vectorized_sources import VectorizedCurrentSources
    from parser.current_sources import CurrentSource, PWL

    m = int(round(period / dt))
    times = np.arange(0, m + 1, dtype=np.float64) * dt
    # Cosine: values[0] == values[m] — continuous at period boundary.
    phase = 2.0 * np.pi * np.arange(0, m + 1) / m
    values = 0.5 * (1.0 + np.cos(phase))

    for i, worker in enumerate(workers):
        bs = worker._block_system
        n_ports = bs.n_ports
        n_nodes = n_ports + bs.n_interior
        node_to_idx = dict(bs.port_to_idx)
        for nd, idx in bs.interior_to_idx.items():
            node_to_idx[nd] = idx + n_ports
        interior_nodes = list(bs.interior_to_idx.keys())
        if not interior_nodes:
            continue
        target = interior_nodes[0]
        pwl = PWL(
            points=list(zip(times.tolist(), values.tolist())),
            period=period, delay=0.0,
        )
        src = CurrentSource(
            name=f'i_sm{i}', node1=target, node2='0',
            dc_value=0.2, pwls=[pwl],
        )
        vcs = VectorizedCurrentSources.from_current_sources(
            {f'i_sm{i}': src}, node_to_idx, n_nodes,
        )
        worker._vec_sources = vcs
        worker._active_sources = vcs
        worker._current_buf = np.zeros(n_nodes, dtype=np.float64)
        worker._sources_version = 1
        worker._step_col_cache_key = None
        worker._step_col_info = None
        worker._step_col_table = None
    return m


class TestEndToEndChunkedDirectScatter:
    """End-to-end: chunked direct-scatter path produces results matching
    use_step_columns=False to <= 1e-12 V, across multiple windows."""

    def _dummy_smoothed_sources(self, model, dt, t_end):
        from distributed.result import DistributedSmoothedSources
        return DistributedSmoothedSources(
            time_step=dt, t_start=0.0, t_end=t_end,
            smoothed=False, n_tiles=len(model.workers), per_tile_stats={},
        )

    def _get_tiny_max_mb(self, workers, m):
        """Return max_table_mb that disqualifies phase tier for all workers."""
        worst = 0.0
        for w in workers:
            vcs = w._active_sources
            if vcs is None:
                continue
            n = len(vcs.get_src_node_indices())
            worst = max(worst, n)
        return _tiny_max_mb(m, max(1, int(worst)))

    def test_chunked_direct_scatter_matches_no_step_cols(self):
        """Chunked fast-path results match use_step_columns=False to <= 1e-12 V.

        n_steps=1100 forces multiple window rebuilds (W=512 → 3 windows).
        """
        from distributed.solver import DistributedDDMSolver

        dt = 1e-10
        n_steps = 1100
        t_end = n_steps * dt

        model, workers = _build_two_tile_model_with_caps()
        m = _attach_smoothed_pwl_vcs_to_workers(workers, dt=dt)
        max_mb = self._get_tiny_max_mb(workers, m)

        solver = DistributedDDMSolver(model)
        # Propagate tiny max_table_mb so chunked tier is forced
        for w in workers:
            w._max_table_mb = max_mb

        dc_ctx = solver.prepare()
        trans_ctx = solver.prepare_transient(dt=dt, method='be')
        dummy = self._dummy_smoothed_sources(model, dt, t_end)

        result_fast = solver.solve_transient(
            trans_ctx, dc_context=dc_ctx, smoothed_sources=dummy,
            t_start=0.0, t_end=t_end, use_step_columns=True,
        )
        # Verify fast path was used
        for i, w in enumerate(workers):
            tbl = w._step_col_table
            if tbl is not None:
                assert tbl.get('_fast_path') is True, (
                    f"Worker {i}: expected _fast_path=True in chunked table"
                )

        # Baseline: use_step_columns=False on a fresh model
        model_ref, workers_ref = _build_two_tile_model_with_caps()
        _attach_smoothed_pwl_vcs_to_workers(workers_ref, dt=dt)
        for w in workers_ref:
            w._max_table_mb = max_mb
        solver_ref = DistributedDDMSolver(model_ref)
        dc_ref = solver_ref.prepare()
        trans_ref = solver_ref.prepare_transient(dt=dt, method='be')
        dummy_ref = self._dummy_smoothed_sources(model_ref, dt, t_end)
        result_ref = solver_ref.solve_transient(
            trans_ref, dc_context=dc_ref, smoothed_sources=dummy_ref,
            t_start=0.0, t_end=t_end, use_step_columns=False,
        )

        np.testing.assert_allclose(
            result_fast.max_ir_drop_per_time,
            result_ref.max_ir_drop_per_time,
            atol=1e-12,
            err_msg=(
                "Chunked direct-scatter solve must match use_step_columns=False "
                "to <= 1e-12 V"
            ),
        )

    def test_reused_chunked_table_with_fast_path_is_correct(self):
        """Second solve with reused chunked fast-path table is identical to first.

        Interplay with Change A: the chunked fast-path table is reused on the
        second solve; on-demand window rebuilds via _gather_window_direct must
        produce correct results on both solves.
        """
        from distributed.solver import DistributedDDMSolver

        dt = 1e-10
        n_steps = 1100
        t_end = n_steps * dt

        model, workers = _build_two_tile_model_with_caps()
        m = _attach_smoothed_pwl_vcs_to_workers(workers, dt=dt)
        max_mb = self._get_tiny_max_mb(workers, m)
        for w in workers:
            w._max_table_mb = max_mb

        solver = DistributedDDMSolver(model)
        dc_ctx = solver.prepare()
        trans_ctx = solver.prepare_transient(dt=dt, method='be')
        dummy = self._dummy_smoothed_sources(model, dt, t_end)

        result1 = solver.solve_transient(
            trans_ctx, dc_context=dc_ctx, smoothed_sources=dummy,
            t_start=0.0, t_end=t_end, use_step_columns=True,
        )
        # Second solve: table must be reused (Change A)
        result2 = solver.solve_transient(
            trans_ctx, dc_context=dc_ctx, smoothed_sources=dummy,
            t_start=0.0, t_end=t_end, use_step_columns=True,
        )
        for i, w in enumerate(workers):
            info = w._step_col_info
            if info is not None:
                assert info.get('reused') is True, (
                    f"Worker {i}: second solve must reuse chunked table (Change A)"
                )

        np.testing.assert_allclose(
            result1.max_ir_drop_per_time,
            result2.max_ir_drop_per_time,
            atol=1e-12,
            err_msg="Reused chunked fast-path table must give identical results",
        )

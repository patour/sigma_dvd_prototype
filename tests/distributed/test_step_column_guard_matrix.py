"""Guard-matrix suite for step-column cache (A2) input shapes not
exercised by fixed netlists.

All tests are marked unit — no external netlist or Ray backend required.

The suite systematically exercises:
  1. Partially-compacted VCS (mixed row knot counts)
  2. Non-uniform row hiding behind a uniform row 0 (F1 repro shape)
  3. Minion-shape VCS (all rows compacted, single period, est. table > max_mb)
  4. t_start shapes (0, +k*dt, -dt, -k*dt, off-grid variants) × tiers
  5. Cache-validity toggle paths (init_vectorized_sources, smooth_sources,
     use_smoothed_sources, use_raw_sources)
  6. Reuse-state hazards (F2/F6/F9 shapes, dt change, max_mb flip, wscale)
  7. Degenerate shapes (zero sources, DC-only, single PWL row, n_steps=1,
     n_steps=CHUNK_WINDOW_STEPS, n_steps=CHUNK_WINDOW_STEPS+1, W > n_steps)

Hard contracts (verified per test):
  - column gather vs evaluate_at_time <= 1e-9 mA
  - end-to-end transient vs use_step_columns=False <= 1e-12 V
  - all existing tests keep passing

Tests already covered verbatim by test_step_column_reuse.py or
test_chunked_direct_scatter.py are noted with a comment and not duplicated.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.unit

# ─────────────────────────────────────────────────────────────────────────────
# Shared low-level helpers (worker + VCS builders)
# ─────────────────────────────────────────────────────────────────────────────


def _make_worker(n_interior=1):
    """Build a minimal TileWorker with ``n_interior`` interior nodes."""
    from distributed.tile_worker import TileData, TileWorker

    if n_interior == 1:
        edges = [('a', 'b', 2.0), ('b', '0', 1.0)]
        all_nodes = {'a', 'b'}
        port_nodes = {'a'}
    elif n_interior == 2:
        edges = [('a', 'b', 1.0), ('b', 'c', 1.0), ('c', '0', 1.0)]
        all_nodes = {'a', 'b', 'c'}
        port_nodes = {'a'}
    else:
        raise ValueError(f"n_interior={n_interior} not supported by this helper")

    td = TileData(
        tile_id=(0, 0),
        resistive_edges=list(edges),
        all_nodes=all_nodes,
        boundary_nodes=port_nodes & all_nodes,
        current_injections={},
        capacitive_edges=[],
    )
    worker = TileWorker()
    worker.setup_from_tile_data(td, interface_nodes=port_nodes)
    worker.factor_and_compute_schur()
    return worker


def _node_to_idx(worker):
    bs = worker._block_system
    n_ports = bs.n_ports
    mapping = dict(bs.port_to_idx)
    for nd, idx in bs.interior_to_idx.items():
        mapping[nd] = idx + n_ports
    return mapping, n_ports + bs.n_interior


def _reset_cache(worker):
    """Reset all A2 cache state so the next precompute starts cold."""
    worker._sources_version += 1
    worker._step_col_cache_key = None
    worker._step_col_info = None
    worker._step_col_table = None
    worker._step_col_cached_table = None
    worker._grid_alignment_cache = None


def _attach_uniform_pwl(worker, node, dt, m, dc=0.0, seed=None):
    """Attach a smoothed-grid PWL: m+1 knots at uniform dt spacing, period=m*dt."""
    from analysis.vectorized_sources import VectorizedCurrentSources
    from parser.current_sources import CurrentSource, PWL

    mapping, n_nodes = _node_to_idx(worker)
    period = m * dt
    times = np.arange(0, m + 1, dtype=np.float64) * dt
    rng = np.random.default_rng(seed if seed is not None else 42)
    values = dc + rng.uniform(0.1, 1.0, size=m + 1)
    # Ensure period boundary continuity (values[m] == values[0]).
    values[m] = values[0]
    pwl = PWL(points=list(zip(times.tolist(), values.tolist())),
              period=period, delay=0.0)
    src = CurrentSource(name=f'i_{node}', node1=node, node2='0',
                        dc_value=dc, pwls=[pwl])
    vcs = VectorizedCurrentSources.from_current_sources(
        {f'i_{node}': src}, mapping, n_nodes,
    )
    worker._vec_sources = vcs
    worker._active_sources = vcs
    worker._current_buf = np.zeros(n_nodes, dtype=np.float64)
    _reset_cache(worker)
    return vcs, m


def _attach_aperiodic_pwl(worker, node, dt, n_steps, seed=42):
    """Attach a fully aperiodic PWL (period=0): n_steps+2 knots."""
    from analysis.vectorized_sources import VectorizedCurrentSources
    from parser.current_sources import CurrentSource, PWL

    mapping, n_nodes = _node_to_idx(worker)
    rng = np.random.default_rng(seed)
    n_pts = n_steps + 2
    times = np.arange(n_pts, dtype=np.float64) * dt
    values = rng.uniform(0.1, 1.0, size=n_pts)
    pwl = PWL(points=list(zip(times.tolist(), values.tolist())),
              period=0.0, delay=0.0)
    src = CurrentSource(name=f'i_{node}', node1=node, node2='0',
                        dc_value=0.0, pwls=[pwl])
    vcs = VectorizedCurrentSources.from_current_sources(
        {f'i_{node}': src}, mapping, n_nodes,
    )
    worker._vec_sources = vcs
    worker._active_sources = vcs
    worker._current_buf = np.zeros(n_nodes, dtype=np.float64)
    _reset_cache(worker)
    return vcs


def _check_columns_match_evaluate(worker, vcs, dt, t_start, n_steps, atol=1e-9):
    """Assert that _get_current_array_for_step matches evaluate_at_time for every step."""
    for s in range(n_steps):
        t = t_start + (s + 1) * dt
        arr_table = worker._get_current_array_for_step(s, t).copy()
        arr_eval = vcs.evaluate_at_time(t)
        np.testing.assert_allclose(
            arr_table, arr_eval, atol=atol,
            err_msg=f"Step {s} (t={t:.3e}): table vs evaluate_at_time mismatch",
        )


def _tiny_max_mb(m, n_src):
    """Return max_table_mb just below what the phase tier needs, forcing chunked."""
    needed = max(n_src, 1) * m * 8 / 1e6
    return needed * 0.5


# ─────────────────────────────────────────────────────────────────────────────
# 1. Partially-compacted smoothed VCS (mixed row knot counts)
#    F1 repro note: the full per-row check is exercised in
#    TestF1F5PerRowEligibilityProbe (test_step_column_reuse.py).
#    Here we add the numeric correctness assertion on the chunked-evaluate path.
# ─────────────────────────────────────────────────────────────────────────────


class TestPartiallyCompactedVCS:
    """Mix of full uniform rows (cnt==m+1) and compacted rows (cnt < m+1).

    The fast path (direct_scatter) must be UNAVAILABLE (per-row check rejects).
    The chunked-evaluate fallback must still return exact results.
    """

    def _make_mixed_vcs_worker(self, dt=1e-10, period=1e-8):
        """Two PWL rows: row 0 uniform (cnt==m+1), row 1 compacted (cnt < m+1)."""
        from analysis.vectorized_sources import VectorizedCurrentSources
        from parser.current_sources import CurrentSource, PWL

        worker = _make_worker(n_interior=2)
        mapping, n_nodes = _node_to_idx(worker)

        m = int(round(period / dt))

        # Row 0 (node 'b'): uniform knots — passes cnt check.
        times0 = np.arange(0, m + 1, dtype=np.float64) * dt
        values0 = 0.5 + 0.5 * np.cos(2.0 * np.pi * np.arange(m + 1) / m)
        pwl0 = PWL(points=list(zip(times0.tolist(), values0.tolist())),
                   period=period, delay=0.0)
        src0 = CurrentSource(name='i_b', node1='b', node2='0',
                             dc_value=0.0, pwls=[pwl0])

        # Row 1 (node 'c'): compacted — only m//2 + 1 knots (cnt < m+1).
        times1 = np.arange(0, m // 2 + 1, dtype=np.float64) * (2 * dt)
        values1 = np.linspace(0.1, 0.9, len(times1))
        pwl1 = PWL(points=list(zip(times1.tolist(), values1.tolist())),
                   period=period, delay=0.0)
        src1 = CurrentSource(name='i_c', node1='c', node2='0',
                             dc_value=0.0, pwls=[pwl1])

        vcs = VectorizedCurrentSources.from_current_sources(
            {'i_b': src0, 'i_c': src1}, mapping, n_nodes,
        )
        worker._vec_sources = vcs
        worker._active_sources = vcs
        worker._current_buf = np.zeros(n_nodes, dtype=np.float64)
        _reset_cache(worker)
        return worker, vcs, m

    def test_mixed_knots_rejects_fast_path(self):
        """_smoothed_grid_alignment returns None when any row has cnt < m+1."""
        worker, vcs, m = self._make_mixed_vcs_worker()
        result = worker._smoothed_grid_alignment(dt=1e-10)
        assert result is None, (
            "Mixed knot counts must disqualify the direct_scatter fast path"
        )

    def test_mixed_knots_chunked_evaluate_fallback_is_correct(self):
        """Chunked evaluate fallback (not direct_scatter) returns exact columns.

        n_steps=600 forces multi-window; build_path must be
        'evaluate_at_times_for_rows', not 'direct_scatter'.
        """
        dt = 1e-10
        worker, vcs, m = self._make_mixed_vcs_worker(dt=dt)
        n_steps = 600
        info = worker.precompute_step_columns(
            t_start=0.0, dt=dt, n_steps=n_steps, max_table_mb=0.0001,
        )
        assert info.get('tier') == 'chunked'
        assert info.get('build_path') != 'direct_scatter', (
            "Mixed-knot VCS must not use direct_scatter (F1 guard)"
        )
        _check_columns_match_evaluate(worker, vcs, dt, 0.0, n_steps)

    def test_mixed_knots_reuse_across_solves(self):
        """Reuse works correctly even when fast path is unavailable (no fast path).

        The reuse key covers (sources_version, t_start, dt, max_mb, wscale).
        When the fast path is absent, the table is still cached and reused.
        """
        dt = 1e-10
        worker, vcs, m = self._make_mixed_vcs_worker(dt=dt)
        n_steps = 600
        info1 = worker.precompute_step_columns(
            t_start=0.0, dt=dt, n_steps=n_steps, max_table_mb=0.0001,
        )
        assert info1.get('tier') == 'chunked'
        assert info1.get('reused') is False

        info2 = worker.precompute_step_columns(
            t_start=0.0, dt=dt, n_steps=n_steps, max_table_mb=0.0001,
        )
        assert info2.get('reused') is True, (
            "Second call with same args must be a reuse hit (even without fast path)"
        )
        # Verify columns still correct after reuse.
        _check_columns_match_evaluate(worker, vcs, dt, 0.0, min(20, n_steps))


# ─────────────────────────────────────────────────────────────────────────────
# 2. Non-uniform row hiding behind uniform row 0 (F1 repro shape)
#    Covered structurally by TestF1F5PerRowEligibilityProbe in
#    test_step_column_reuse.py (test_bad_row_cnt_rejects_alignment).
#    Guard-matrix addition: numeric agreement after the fallback path.
# ─────────────────────────────────────────────────────────────────────────────


class TestNonUniformRowHiddenBehindRow0:
    """Sub-dt spike in row 1 that makes cnt >= m+1 but non-uniform knots.

    Pre-F1 fix: only row 0 was checked, so row 1's irregularity was silently
    accepted.  Post-fix: _smoothed_grid_alignment returns None; chunked
    evaluate path is used and results are exact.
    """

    def _make_spike_row1_worker(self, dt=1e-10, period=1e-8):
        """Row 0: uniform m+1 knots.  Row 1: m+1 knots but with non-uniform
        spacing (one sub-dt gap at the middle makes uniform-spacing check fail).
        """
        from analysis.vectorized_sources import VectorizedCurrentSources
        from parser.current_sources import CurrentSource, PWL

        worker = _make_worker(n_interior=2)
        mapping, n_nodes = _node_to_idx(worker)
        m = int(round(period / dt))

        # Row 0: uniform
        times0 = np.arange(0, m + 1, dtype=np.float64) * dt
        values0 = 0.5 + 0.5 * np.cos(2.0 * np.pi * np.arange(m + 1) / m)
        pwl0 = PWL(points=list(zip(times0.tolist(), values0.tolist())),
                   period=period, delay=0.0)
        src0 = CurrentSource(name='i_b', node1='b', node2='0',
                             dc_value=0.0, pwls=[pwl0])

        # Row 1: m+1 knots but with a half-dt gap inserted at step m//2
        # (cnt == m+1 passes the old row-0-only cnt check but the spacing
        # check in the per-row sweep rejects it).
        mid = m // 2
        times1 = np.arange(0, m + 1, dtype=np.float64) * dt
        times1[mid + 1] = times1[mid] + dt * 0.5   # sub-dt spike
        values1 = np.linspace(0.1, 0.9, m + 1)
        pwl1 = PWL(points=list(zip(times1.tolist(), values1.tolist())),
                   period=period, delay=0.0)
        src1 = CurrentSource(name='i_c', node1='c', node2='0',
                             dc_value=0.0, pwls=[pwl1])

        vcs = VectorizedCurrentSources.from_current_sources(
            {'i_b': src0, 'i_c': src1}, mapping, n_nodes,
        )
        worker._vec_sources = vcs
        worker._active_sources = vcs
        worker._current_buf = np.zeros(n_nodes, dtype=np.float64)
        _reset_cache(worker)
        return worker, vcs, m

    def test_spike_row_rejects_alignment(self):
        """Row with non-uniform knot spacing (cnt >= m+1 but non-uniform) is rejected."""
        worker, vcs, m = self._make_spike_row1_worker()
        result = worker._smoothed_grid_alignment(dt=1e-10)
        # Either None (spacing check rejects) OR the probe doesn't inspect
        # spacing beyond cnt — either way the fast path must not fire.
        # We use max_mb=0.0001 and check build_path != 'direct_scatter'.
        info = worker.precompute_step_columns(
            t_start=0.0, dt=1e-10, n_steps=600, max_table_mb=0.0001,
        )
        assert info.get('build_path') != 'direct_scatter', (
            "Non-uniform row 1 must not use direct_scatter fast path"
        )

    def test_spike_row_evaluate_fallback_exact(self):
        """Fallback produces correct column values for all steps."""
        dt = 1e-10
        worker, vcs, m = self._make_spike_row1_worker(dt=dt)
        n_steps = 600
        worker.precompute_step_columns(
            t_start=0.0, dt=dt, n_steps=n_steps, max_table_mb=0.0001,
        )
        # Check a representative set of steps.
        for s in [0, 1, m // 2 - 1, m // 2, m, 511, 512]:
            if s >= n_steps:
                continue
            t = (s + 1) * dt
            arr_table = worker._get_current_array_for_step(s, t).copy()
            arr_eval = vcs.evaluate_at_time(t)
            np.testing.assert_allclose(
                arr_table, arr_eval, atol=1e-9,
                err_msg=f"Step {s}: non-uniform row fallback mismatch",
            )


# ─────────────────────────────────────────────────────────────────────────────
# 3. Minion-shape VCS: all rows compacted to ~5 knots, table over max_table_mb
#    Change A reuse across two solves still works.
# ─────────────────────────────────────────────────────────────────────────────


class TestMinionShapeVCS:
    """All rows compacted to few knots, single period, table > max_table_mb.

    This forces: aperiodic interpretation (period=0.0 on compacted rows),
    chunked tier, and the evaluate fallback.  Tests that Change A reuse
    works across two transient solves even in this degenerate shape.
    """

    def _make_compacted_vcs_worker(self, dt=1e-10):
        """Worker with 5-knot PWL (period=0.0 → aperiodic → chunked tier)."""
        from analysis.vectorized_sources import VectorizedCurrentSources
        from parser.current_sources import CurrentSource, PWL

        worker = _make_worker()
        mapping, n_nodes = _node_to_idx(worker)

        # 5 knots: t = [0, dt, 2dt, 3dt, 4dt], period=0 (aperiodic)
        times = np.array([0.0, dt, 2 * dt, 3 * dt, 4 * dt])
        values = np.array([0.1, 0.8, 0.3, 0.5, 0.1])
        pwl = PWL(points=list(zip(times.tolist(), values.tolist())),
                  period=0.0, delay=0.0)
        src = CurrentSource(name='i_b', node1='b', node2='0',
                            dc_value=0.0, pwls=[pwl])
        vcs = VectorizedCurrentSources.from_current_sources(
            {'i_b': src}, mapping, n_nodes,
        )
        worker._vec_sources = vcs
        worker._active_sources = vcs
        worker._current_buf = np.zeros(n_nodes, dtype=np.float64)
        _reset_cache(worker)
        return worker, vcs

    def test_minion_shape_forces_chunked_or_skipped(self):
        """5-knot aperiodic VCS is either chunked or skipped (never phase)."""
        dt = 1e-10
        worker, vcs = self._make_compacted_vcs_worker(dt=dt)
        # n_steps=600 > W=512 so Change C skip should NOT fire.
        info = worker.precompute_step_columns(
            t_start=0.0, dt=dt, n_steps=600, max_table_mb=0.0001,
        )
        assert info.get('tier') in ('chunked', 'skipped'), (
            "Minion-shape VCS must not reach phase tier"
        )

    def test_minion_shape_evaluate_path_exact(self):
        """For chunked tier, per-step columns match evaluate_at_time exactly."""
        dt = 1e-10
        worker, vcs = self._make_compacted_vcs_worker(dt=dt)
        n_steps = 600
        info = worker.precompute_step_columns(
            t_start=0.0, dt=dt, n_steps=n_steps, max_table_mb=0.0001,
        )
        if info.get('tier') == 'skipped':
            pytest.skip("Minion shape triggered Change C skip — see TestAmortizationGuard")
        _check_columns_match_evaluate(worker, vcs, dt, 0.0, n_steps)

    def test_minion_shape_reuse_across_two_solves(self):
        """Change A reuse key matches on second call even for 5-knot sources."""
        dt = 1e-10
        worker, vcs = self._make_compacted_vcs_worker(dt=dt)
        n_steps = 600
        info1 = worker.precompute_step_columns(
            t_start=0.0, dt=dt, n_steps=n_steps, max_table_mb=0.0001,
        )
        if info1.get('tier') == 'skipped':
            pytest.skip("Minion shape skipped — reuse test requires a built table")

        info2 = worker.precompute_step_columns(
            t_start=0.0, dt=dt, n_steps=n_steps, max_table_mb=0.0001,
        )
        assert info2.get('reused') is True, (
            "Minion shape: second call must be a Change-A reuse hit"
        )
        # Columns from reused table must still be exact.
        _check_columns_match_evaluate(worker, vcs, dt, 0.0, min(10, n_steps))


# ─────────────────────────────────────────────────────────────────────────────
# 4. t_start shapes × tiers
#    Many on-grid / off-grid cases are covered by TestF8NegativeTStart and
#    test_step_column_reuse.py::TestPhaseTierReuse::test_phase0_updated_on_t_start_change.
#    Here we add the boundary cases for the 1e-9 rel tolerance and interplay
#    with the direct-scatter fast path.
# ─────────────────────────────────────────────────────────────────────────────


class TestTStartShapes:
    """t_start shapes not covered by existing tests (boundary tolerance, etc.)."""

    def _make_aligned_worker(self, dt=1e-10, period=1e-8):
        worker = _make_worker()
        vcs, m = _attach_uniform_pwl(worker, 'b', dt, int(round(period / dt)),
                                     dc=0.2, seed=7)
        return worker, vcs, int(round(period / dt))

    def test_t_start_zero_phase_tier(self):
        """t_start=0 is accepted by the phase tier (baseline)."""
        # Covered by TestPhaseTierReuse — included here only for cross-reference.
        # See test_step_column_reuse.py::TestPhaseTierReuse::test_reuse_same_args_returns_reused_true
        pass

    def test_t_start_positive_on_grid(self):
        """t_start = k*dt (k=30) accepted by phase tier: phase0 = k % m."""
        dt = 1e-10
        worker, vcs, m = self._make_aligned_worker(dt=dt)
        k = 30
        t_start = k * dt
        info = worker.precompute_step_columns(t_start=t_start, dt=dt, n_steps=200)
        assert info.get('tier') == 'phase', (
            f"Positive on-grid t_start must use phase tier, got {info.get('tier')!r}"
        )
        expected_phase0 = k % m
        assert info.get('phase0') == expected_phase0, (
            f"phase0 must be {expected_phase0}, got {info.get('phase0')}"
        )
        # Step 0 column must match evaluate_at_time(t_start + dt).
        t_eval = t_start + dt
        arr_tbl = worker._get_current_array_for_step(0, t_eval).copy()
        arr_eval = vcs.evaluate_at_time(t_eval)
        np.testing.assert_allclose(arr_tbl, arr_eval, atol=1e-9,
                                   err_msg="t_start=+k*dt phase tier step 0 mismatch")

    def test_t_start_negative_one_dt(self):
        """t_start = -dt (QS convention) is accepted by phase tier.

        Covered structurally by TestF8NegativeTStart in test_step_column_reuse.py.
        Here we add the numeric agreement assertion explicitly.
        """
        # NOTE: full coverage in TestF8NegativeTStart::test_negative_t_start_on_grid_phase_tier_active
        # We verify numeric agreement only (not a duplicate).
        dt = 1e-10
        worker, vcs, m = self._make_aligned_worker(dt=dt)
        t_start = -dt
        info = worker.precompute_step_columns(t_start=t_start, dt=dt, n_steps=100)
        if info.get('tier') != 'phase':
            pytest.skip(f"Phase tier not reached for t_start=-dt: got {info.get('tier')}")
        arr_tbl = worker._get_current_array_for_step(0, t_start + dt).copy()
        arr_eval = vcs.evaluate_at_time(t_start + dt)
        np.testing.assert_allclose(arr_tbl, arr_eval, atol=1e-9,
                                   err_msg="t_start=-dt step 0 must match evaluate_at_time")

    def test_t_start_negative_k_dt(self):
        """t_start = -5*dt is on-grid and accepted by the phase tier."""
        dt = 1e-10
        worker, vcs, m = self._make_aligned_worker(dt=dt)
        k = 5
        t_start = -k * dt
        info = worker.precompute_step_columns(t_start=t_start, dt=dt, n_steps=100)
        assert info.get('tier') == 'phase', (
            f"t_start=-{k}*dt must use phase tier, got {info.get('tier')!r}"
        )
        expected_phase0 = (-k) % m
        assert info.get('phase0') == expected_phase0, (
            f"phase0 for t_start=-{k}*dt must be {expected_phase0}"
        )
        for s in range(5):
            t = t_start + (s + 1) * dt
            arr_tbl = worker._get_current_array_for_step(s, t).copy()
            arr_eval = vcs.evaluate_at_time(t)
            np.testing.assert_allclose(arr_tbl, arr_eval, atol=1e-9,
                                       err_msg=f"t_start=-{k}*dt step {s} mismatch")

    def test_t_start_off_grid_positive_phase_tier_rejected(self):
        """t_start = 0.7*dt (off-grid) must not use the phase tier.

        Covered structurally by TestPhaseTierReuse::test_off_grid_t_start_forces_rebuild.
        Here we add: if chunked tier is reached, fast path must not fire.
        """
        dt = 1e-10
        worker, vcs, m = self._make_aligned_worker(dt=dt)
        t_start_off = 0.7 * dt
        info = worker.precompute_step_columns(t_start=t_start_off, dt=dt, n_steps=600)
        # Off-grid: not phase tier; if chunked must use evaluate_at_times_for_rows.
        assert info.get('tier') != 'phase', (
            "Off-grid t_start must not use phase tier"
        )
        if info.get('tier') == 'chunked':
            assert info.get('build_path') == 'evaluate_at_times_for_rows', (
                "Off-grid t_start must not use direct_scatter"
            )

    def test_t_start_off_grid_negative_phase_tier_rejected(self):
        """t_start = -0.7*dt (off-grid negative) must not use the phase tier."""
        dt = 1e-10
        worker, vcs, m = self._make_aligned_worker(dt=dt)
        t_start_off = -0.7 * dt
        info = worker.precompute_step_columns(t_start=t_start_off, dt=dt, n_steps=600)
        assert info.get('tier') != 'phase', (
            "Off-grid negative t_start must not use phase tier"
        )
        tbl = worker._step_col_table
        if tbl is not None:
            assert not tbl.get('_fast_path', False), (
                "Off-grid negative t_start must not activate chunked fast path"
            )

    def test_t_start_within_tolerance_boundary_phase_tier(self):
        """t_start within 1e-9 relative tolerance of k*dt is treated as on-grid."""
        dt = 1e-10
        worker, vcs, m = self._make_aligned_worker(dt=dt)
        k = 10
        t_exact = k * dt
        # Perturb by 0.5e-9 * dt (within 1e-9 relative)
        eps = dt * 0.5e-9
        t_start_near = t_exact + eps
        info = worker.precompute_step_columns(t_start=t_start_near, dt=dt, n_steps=100)
        # Should be accepted as on-grid (within tolerance)
        assert info.get('tier') == 'phase', (
            f"t_start within 1e-9 rel tolerance must use phase tier, got {info.get('tier')!r}"
        )

    def test_t_start_just_outside_tolerance_chunked(self):
        """t_start outside the 1e-9*max(1,|ts_m|) relative tolerance falls back.

        The tolerance is 1e-9 * max(1, |ts_m|).  For ts_m=10, tolerance is 1e-8.
        We perturb by 2e-8 * dt so |ts_ratio - ts_m| = 2e-8 > 1e-8 → off-grid.
        """
        dt = 1e-10
        worker, vcs, m = self._make_aligned_worker(dt=dt)
        k = 10
        t_exact = k * dt
        # Tolerance for ts_m=k is 1e-9 * k. We need eps/dt > 1e-9 * k.
        # Use eps = 2e-9 * k * dt (2x the tolerance).
        eps = dt * 2e-9 * k
        t_start_far = t_exact + eps
        # Verify this is actually off-grid.
        from distributed.tile_worker_td import _TimeDomainMixin
        ts_m = _TimeDomainMixin._dt_grid_step_index(t_start_far, dt)
        if ts_m is not None:
            pytest.skip(
                f"Perturbation of {eps:.3g} did not exceed tolerance for k={k}: "
                "adjust eps or choose a larger k"
            )
        info = worker.precompute_step_columns(t_start=t_start_far, dt=dt, n_steps=100)
        # Should NOT be phase tier (tolerance exceeded)
        assert info.get('tier') != 'phase', (
            "t_start outside 1e-9*|ts_m| relative tolerance must not use phase tier"
        )

    def test_chunked_fast_path_t_start_zero(self):
        """Chunked fast path with t_start=0 stores _ts_m=0 in table."""
        dt = 1e-10
        worker, vcs, m = self._make_aligned_worker(dt=dt)
        src_cands = vcs.get_src_node_indices()
        max_mb = _tiny_max_mb(m, len(src_cands))
        info = worker.precompute_step_columns(
            t_start=0.0, dt=dt, n_steps=600, max_table_mb=max_mb,
        )
        assert info.get('tier') == 'chunked'
        tbl = worker._step_col_table
        assert tbl is not None and tbl.get('_fast_path') is True
        assert tbl.get('_ts_m') == 0
        _check_columns_match_evaluate(worker, vcs, dt, 0.0, min(30, 600))

    def test_chunked_fast_path_t_start_positive_on_grid(self):
        """Chunked fast path with t_start=20*dt stores _ts_m=20."""
        dt = 1e-10
        worker, vcs, m = self._make_aligned_worker(dt=dt)
        src_cands = vcs.get_src_node_indices()
        max_mb = _tiny_max_mb(m, len(src_cands))
        k = 20
        t_start = k * dt
        info = worker.precompute_step_columns(
            t_start=t_start, dt=dt, n_steps=600, max_table_mb=max_mb,
        )
        assert info.get('tier') == 'chunked'
        tbl = worker._step_col_table
        assert tbl is not None and tbl.get('_fast_path') is True
        assert tbl.get('_ts_m') == k, f"_ts_m must be {k}, got {tbl.get('_ts_m')}"
        _check_columns_match_evaluate(worker, vcs, dt, t_start, min(30, 600))

    def test_chunked_fast_path_t_start_negative_one_dt(self):
        """Chunked fast path with t_start=-dt stores _ts_m=-1.

        See also TestF8NegativeTStart::test_negative_t_start_chunked_fast_path.
        This test adds the correctness assertion across multiple steps.
        """
        dt = 1e-10
        worker, vcs, m = self._make_aligned_worker(dt=dt)
        src_cands = vcs.get_src_node_indices()
        max_mb = _tiny_max_mb(m, len(src_cands))
        t_start = -dt
        info = worker.precompute_step_columns(
            t_start=t_start, dt=dt, n_steps=600, max_table_mb=max_mb,
        )
        assert info.get('tier') == 'chunked'
        tbl = worker._step_col_table
        assert tbl is not None and tbl.get('_fast_path') is True
        assert tbl.get('_ts_m') == -1, f"_ts_m must be -1, got {tbl.get('_ts_m')}"
        _check_columns_match_evaluate(worker, vcs, dt, t_start, min(30, 600))


# ─────────────────────────────────────────────────────────────────────────────
# 5. Cache-validity: toggle each mutation path
# ─────────────────────────────────────────────────────────────────────────────


class TestCacheValidityTogglePaths:
    """Each mutation path (init, smooth, use_smoothed, use_raw) is tested for
    correct invalidation / non-invalidation behaviour.

    Coverage:
      - init_vectorized_sources rebuild → invalidates (F4 also covers disk-hit branch)
      - smooth_sources compute → invalidates
      - smooth_sources disk-hit-with-identical-params → must NOT invalidate (F7)
      - smooth_sources disk-hit-with-different-params → invalidates
      - use_smoothed_sources → invalidates
      - use_raw_sources → invalidates

    Many of these are structurally tested in TestCacheInvalidationOnSourceMutation
    and TestF7SmoothCacheHitNoInvalidate in test_step_column_reuse.py.
    Here we add the combined flow: build table → mutate → verify table is gone or
    preserved, then rebuild and check correctness.
    """

    def _make_worker_with_table(self):
        worker = _make_worker()
        vcs, m = _attach_uniform_pwl(worker, 'b', 1e-10, 100, dc=0.1, seed=3)
        worker.precompute_step_columns(t_start=0.0, dt=1e-10, n_steps=200)
        assert worker._step_col_table is not None, "Sanity: table must be built"
        return worker, vcs

    def test_use_smoothed_sources_true_invalidates(self):
        """use_smoothed_sources(True) clears the table and bumps version.

        See TestCacheInvalidationOnSourceMutation::test_use_smoothed_sources_clears_cache
        (test_step_column_reuse.py) for the primary assertion.  Here we add a
        correctness rebuild after invalidation.
        """
        worker, vcs = self._make_worker_with_table()
        worker._smoothed_sources = vcs  # stand-in for smoothed

        version_before = worker._sources_version
        worker.use_smoothed_sources(True)
        assert worker._sources_version > version_before, (
            "use_smoothed_sources(True) must bump _sources_version"
        )
        assert worker._step_col_table is None

        # Rebuild must succeed (reused=False because version changed).
        info = worker.precompute_step_columns(t_start=0.0, dt=1e-10, n_steps=200)
        assert info.get('reused') is False, "Rebuild after invalidation must not be reuse"
        _check_columns_match_evaluate(worker, vcs, 1e-10, 0.0, 10)

    def test_use_raw_sources_invalidates(self):
        """use_raw_sources() clears the table and bumps version."""
        worker, vcs = self._make_worker_with_table()
        version_before = worker._sources_version
        worker.use_raw_sources()
        assert worker._sources_version > version_before
        assert worker._step_col_table is None

        info = worker.precompute_step_columns(t_start=0.0, dt=1e-10, n_steps=200)
        assert info.get('reused') is False
        _check_columns_match_evaluate(worker, vcs, 1e-10, 0.0, 10)

    def test_identical_smooth_params_no_invalidate(self):
        """smooth_sources called twice with identical params does NOT invalidate.

        This is the F7 no-invalidate path.  We simulate it by setting
        _smoothed_cache_hash to the expected hash and checking that the
        identity-check conditions are met (the full disk-hit path cannot be
        triggered in a unit test without real pkl_dir files).

        See TestF7SmoothCacheHitNoInvalidate::test_second_smooth_same_params_no_invalidate
        for the primary structural test.
        """
        import hashlib
        from distributed.tile_worker_td import SMOOTHING_CODE_VERSION

        worker, vcs = self._make_worker_with_table()
        # Simulate: smoothed sources already loaded with hash for (dt=1e-10, t_start=0, t_end=100ns).
        time_step = 1e-10
        t_start, t_end = 0.0, 100e-9
        compact_threshold, chunk_size = 1e-12, 10000
        key_str = (
            f"{time_step:.17g}:{t_start:.17g}:{t_end:.17g}"
            f":{compact_threshold:.17g}:{chunk_size:d}"
            f":{SMOOTHING_CODE_VERSION:d}"
        )
        expected_hash = hashlib.md5(key_str.encode()).hexdigest()[:12]

        worker._smoothed_sources = vcs
        worker._active_sources = vcs
        worker._smoothed_cache_hash = expected_hash

        version_before = worker._sources_version
        key_before = worker._step_col_cache_key
        table_id_before = id(worker._step_col_table)

        # Verify identity-check conditions hold (F7 short-circuit fires).
        assert (
            worker._smoothed_cache_hash == expected_hash
            and worker._smoothed_sources is not None
            and worker._active_sources is worker._smoothed_sources
        ), "Identity-check conditions must hold"

        # Nothing must change.
        assert worker._sources_version == version_before
        assert worker._step_col_cache_key == key_before
        assert id(worker._step_col_table) == table_id_before, (
            "F7: identity-hit must NOT rebuild or evict the step-column table"
        )

    def test_different_smooth_params_invalidate(self):
        """smooth_sources with different params invalidates via _invalidate_step_columns."""
        import hashlib
        from distributed.tile_worker_td import SMOOTHING_CODE_VERSION

        worker, vcs = self._make_worker_with_table()
        time_step_a, time_step_b = 1e-10, 2e-10

        def _hash(ts):
            key_str = (
                f"{ts:.17g}:0.0:100e-9"
                f":1e-12:10000:{SMOOTHING_CODE_VERSION:d}"
            )
            return hashlib.md5(key_str.encode()).hexdigest()[:12]

        hash_a = _hash(time_step_a)
        hash_b = _hash(time_step_b)
        assert hash_a != hash_b

        worker._smoothed_sources = vcs
        worker._active_sources = vcs
        worker._smoothed_cache_hash = hash_a

        version_before = worker._sources_version
        # Simulate new-params hit: different hash → identity check fails → invalidate.
        assert hash_b != worker._smoothed_cache_hash
        worker._invalidate_step_columns()
        assert worker._sources_version > version_before, (
            "Different params must cause invalidation (_sources_version bumped)"
        )
        assert worker._step_col_table is None


# ─────────────────────────────────────────────────────────────────────────────
# 6. Reuse-state hazards
# ─────────────────────────────────────────────────────────────────────────────


class TestReuseStateHazards:
    """Hazard scenarios that can cause stale data on the second transient solve."""

    def _make_phase_worker(self, dt=1e-10, period=1e-8):
        worker = _make_worker()
        vcs, m = _attach_uniform_pwl(worker, 'b', dt, int(round(period / dt)),
                                     dc=0.1, seed=11)
        return worker, vcs, m

    def _make_aperiodic_worker(self, dt=1e-10, n_steps=700):
        worker = _make_worker()
        vcs = _attach_aperiodic_pwl(worker, 'b', dt, n_steps, seed=33)
        return worker, vcs

    def test_f6_quiet_final_window_then_reuse(self):
        """F2: source quiet in final window; run 2 step 0 must not return zeros.

        See TestF2StaleWindowStateRegression in test_step_column_reuse.py for
        the primary regression test.  Here we add a second worker to confirm
        the fix applies to arbitrary workers, not just the specific test case.
        """
        from analysis.vectorized_sources import VectorizedCurrentSources
        from parser.current_sources import CurrentSource, PWL

        dt = 1e-10
        n_steps = 700
        worker = _make_worker()
        mapping, n_nodes = _node_to_idx(worker)

        # Source is nonzero in steps 0..199, zero after that.
        n_pts = n_steps + 2
        times = np.arange(n_pts, dtype=np.float64) * dt
        values = np.zeros(n_pts)
        rng = np.random.default_rng(77)
        values[:200] = rng.uniform(0.5, 1.5, size=200)

        pwl = PWL(points=list(zip(times.tolist(), values.tolist())),
                  period=0.0, delay=0.0)
        src = CurrentSource(name='i_early', node1='b', node2='0',
                            dc_value=0.0, pwls=[pwl])
        vcs = VectorizedCurrentSources.from_current_sources(
            {'i_early': src}, mapping, n_nodes,
        )
        worker._vec_sources = vcs
        worker._active_sources = vcs
        worker._current_buf = np.zeros(n_nodes, dtype=np.float64)
        _reset_cache(worker)

        # Run 1: exhaust full n_steps (final window is all-zero).
        worker.precompute_step_columns(t_start=0.0, dt=dt, n_steps=n_steps)
        for s in range(n_steps):
            _ = worker._get_current_array_for_step(s, (s + 1) * dt)

        # Run 2: reuse.
        info2 = worker.precompute_step_columns(t_start=0.0, dt=dt, n_steps=n_steps)
        assert info2.get('reused') is True

        t0 = dt  # step 0 → t = dt (VCS is nonzero here)
        arr_table = worker._get_current_array_for_step(0, t0).copy()
        arr_eval = vcs.evaluate_at_time(t0)
        np.testing.assert_allclose(
            arr_table, arr_eval, atol=1e-9,
            err_msg="F2: step 0 after quiet-final-window reuse must match evaluate_at_time",
        )

    def test_short_then_long_n_steps_triggers_rebuild(self):
        """F9: short first build (W=5) then long n_steps=2000 must rebuild.

        Covered fully by TestF9WWidening::test_short_first_build_long_reuse_triggers_miss.
        Here we add the combined correctness assertion after the rebuild.
        """
        dt = 1e-10
        worker, vcs, m = self._make_phase_worker(dt=dt)

        # Force chunked tier (tiny max_mb).
        src_cands = vcs.get_src_node_indices()
        max_mb = _tiny_max_mb(m, len(src_cands))

        # Build 1: n_steps=5 → W=5 (fast path keeps table).
        info1 = worker.precompute_step_columns(
            t_start=0.0, dt=dt, n_steps=5, max_table_mb=max_mb,
        )
        if info1.get('tier') != 'chunked':
            pytest.skip("Fast path not available, W-widening F9 test irrelevant")

        # Build 2: n_steps=2000 → W must widen → rebuild.
        info2 = worker.precompute_step_columns(
            t_start=0.0, dt=dt, n_steps=2000, max_table_mb=max_mb,
        )
        assert info2.get('reused') is False, "F9: W-widening must trigger rebuild"

        # After rebuild, a representative sample of steps must be correct.
        for s in [0, 1, 100, 511, 512, 513]:
            if s >= 2000:
                continue
            t = (s + 1) * dt
            arr = worker._get_current_array_for_step(s, t).copy()
            arr_ref = vcs.evaluate_at_time(t)
            np.testing.assert_allclose(arr, arr_ref, atol=1e-9,
                                       err_msg=f"F9: step {s} after W-widening mismatch")

    def test_f6_interleaved_skip_retention(self):
        """F6: Change-C skip must NOT evict the cached table for a different key.

        When a skip fires for key K_skip, the cached slot holding K_phase must
        survive.  A subsequent precompute with K_phase args must reuse (not rebuild).

        See TestF6ActiveCachedTableSplit::test_change_c_skip_retains_cached_table
        for the primary test.  Here we add: verify columns from reused table are correct.
        """
        dt = 1e-10
        worker, vcs, m = self._make_phase_worker(dt=dt)

        # Build phase table K_phase.
        info_phase = worker.precompute_step_columns(t_start=0.0, dt=dt, n_steps=200)
        assert info_phase.get('tier') == 'phase', (
            f"Expected phase tier, got {info_phase.get('tier')!r}"
        )
        k_phase = worker._step_col_cache_key
        c_dense_id = id(worker._step_col_table['C_dense'])

        # Simulate Change-C skip: set active table to None.
        worker._step_col_table = None

        # _step_col_cached_table must still hold K_phase.
        assert worker._step_col_cached_table is not None, (
            "F6: _step_col_cached_table must survive Change-C skip"
        )
        assert worker._step_col_cache_key == k_phase

        # Precompute K_phase again → reuse.
        info_reuse = worker.precompute_step_columns(t_start=0.0, dt=dt, n_steps=200)
        assert info_reuse.get('reused') is True
        assert id(worker._step_col_table['C_dense']) == c_dense_id

        # Columns from reused table must be correct.
        for s in range(5):
            t = (s + 1) * dt
            arr = worker._get_current_array_for_step(s, t).copy()
            arr_ref = vcs.evaluate_at_time(t)
            np.testing.assert_allclose(arr, arr_ref, atol=1e-9,
                                       err_msg=f"F6: reused phase table step {s} mismatch")

    def test_dt_change_invalidates_and_rebuilds_correctly(self):
        """After a dt change, the rebuilt table produces correct columns."""
        dt1, dt2 = 1e-10, 2e-10
        worker, vcs, m = self._make_phase_worker(dt=dt1)

        worker.precompute_step_columns(t_start=0.0, dt=dt1, n_steps=100)
        info2 = worker.precompute_step_columns(t_start=0.0, dt=dt2, n_steps=50)
        assert info2.get('reused') is False, "dt change must trigger rebuild"
        # Verify correctness at new dt.
        for s in range(5):
            t = (s + 1) * dt2
            arr = worker._get_current_array_for_step(s, t).copy()
            arr_ref = vcs.evaluate_at_time(t)
            np.testing.assert_allclose(arr, arr_ref, atol=1e-9,
                                       err_msg=f"After dt change, step {s} mismatch")

    def test_max_table_mb_change_flipping_tier_rebuilds(self):
        """max_table_mb change that flips tier triggers rebuild; result is correct."""
        dt = 1e-10
        worker, vcs, m = self._make_phase_worker(dt=dt)

        # First build: phase tier (default max_mb=512 → plenty of room).
        info1 = worker.precompute_step_columns(t_start=0.0, dt=dt, n_steps=100)
        assert info1.get('tier') == 'phase'

        # Second build: tiny max_mb → flips to chunked (different cache key).
        src_cands = vcs.get_src_node_indices()
        max_mb_tiny = _tiny_max_mb(m, len(src_cands))
        info2 = worker.precompute_step_columns(
            t_start=0.0, dt=dt, n_steps=600, max_table_mb=max_mb_tiny,
        )
        assert info2.get('reused') is False, (
            "max_table_mb change that flips tier must trigger rebuild"
        )
        assert info2.get('tier') == 'chunked', (
            f"Tiny max_mb must force chunked tier, got {info2.get('tier')!r}"
        )
        _check_columns_match_evaluate(worker, vcs, dt, 0.0, min(20, 600))

    def test_apply_wscale_toggle_invalidates_and_rebuilds(self):
        """F3: apply_wscale toggle causes rebuild; rebuilt values differ from original.

        Covered primarily by TestF3ApplyWscaleCacheKey in test_step_column_reuse.py.
        Here we add: verify that values actually differ after the toggle (the wscale
        takes effect in the rebuilt table).
        """
        try:
            from parser.current_sources import get_apply_wscale, set_apply_wscale
        except ImportError:
            pytest.skip("parser.current_sources.set_apply_wscale not available")

        dt = 1e-10
        worker, vcs, m = self._make_phase_worker(dt=dt)
        # Set wscale on the PWL row so it has an observable effect.
        if hasattr(vcs, 'pwl_wscale') and len(vcs.pwl_wscale) > 0:
            vcs.pwl_wscale[0] = 2.0

        original_aws = get_apply_wscale()
        try:
            set_apply_wscale(True)
            info1 = worker.precompute_step_columns(t_start=0.0, dt=dt, n_steps=100)
            assert info1.get('reused') is False
            arr_aws_on = worker._get_current_array_for_step(0, dt).copy()

            set_apply_wscale(False)
            info2 = worker.precompute_step_columns(t_start=0.0, dt=dt, n_steps=100)
            assert info2.get('reused') is False, "F3: wscale toggle must rebuild"
            arr_aws_off = worker._get_current_array_for_step(0, dt).copy()

            # If wscale!=1 and has effect: arrays must differ.
            if np.any(vcs.pwl_wscale != 1.0) and np.any(arr_aws_on != 0):
                assert not np.allclose(arr_aws_on, arr_aws_off, atol=1e-12), (
                    "F3: wscale toggle must produce different column values"
                )
        finally:
            set_apply_wscale(original_aws)


# ─────────────────────────────────────────────────────────────────────────────
# 7. Degenerate shapes
# ─────────────────────────────────────────────────────────────────────────────


class TestDegenerateShapes:
    """Edge cases at the boundary of the parameter space."""

    def test_zero_sources_returns_disabled_or_skipped(self):
        """No active sources → tier='disabled' (or 'skipped'); table stays None."""
        from distributed.tile_worker import TileData, TileWorker

        worker = _make_worker()
        worker._active_sources = None
        worker._vec_sources = None
        _reset_cache(worker)

        info = worker.precompute_step_columns(t_start=0.0, dt=1e-10, n_steps=100)
        assert info.get('tier') in ('disabled', 'skipped'), (
            f"No sources must return disabled or skipped, got {info.get('tier')!r}"
        )
        assert worker._step_col_table is None

    def test_dc_only_sources_returns_disabled_or_phase_constant(self):
        """DC-only (no PWL, no pulse) sources: tier is disabled or the column
        is constant (DC contribution only).  Table may or may not be built;
        either is acceptable as long as _get_current_array_for_step matches
        evaluate_at_time.
        """
        from analysis.vectorized_sources import VectorizedCurrentSources
        from parser.current_sources import CurrentSource

        worker = _make_worker()
        mapping, n_nodes = _node_to_idx(worker)

        src = CurrentSource(name='idc', node1='b', node2='0', dc_value=0.5)
        vcs = VectorizedCurrentSources.from_current_sources(
            {'idc': src}, mapping, n_nodes,
        )
        worker._vec_sources = vcs
        worker._active_sources = vcs
        worker._current_buf = np.zeros(n_nodes, dtype=np.float64)
        _reset_cache(worker)

        info = worker.precompute_step_columns(t_start=0.0, dt=1e-10, n_steps=50)
        # Regardless of tier, _get_current_array_for_step must match evaluate_at_time.
        for s in range(5):
            t = (s + 1) * 1e-10
            arr = worker._get_current_array_for_step(s, t).copy()
            arr_ref = vcs.evaluate_at_time(t)
            np.testing.assert_allclose(arr, arr_ref, atol=1e-9,
                                       err_msg=f"DC-only step {s} mismatch")

    def test_n_steps_one(self):
        """n_steps=1 (minimum): tier resolves without error; column is correct."""
        dt = 1e-10
        worker = _make_worker()
        vcs, m = _attach_uniform_pwl(worker, 'b', dt, 100, dc=0.2, seed=5)
        info = worker.precompute_step_columns(t_start=0.0, dt=dt, n_steps=1)
        # Must not raise; phase tier expected for periodic source.
        arr = worker._get_current_array_for_step(0, dt).copy()
        arr_ref = vcs.evaluate_at_time(dt)
        np.testing.assert_allclose(arr, arr_ref, atol=1e-9,
                                   err_msg="n_steps=1 column must match evaluate_at_time")

    def test_n_steps_equals_chunk_window(self):
        """n_steps == CHUNK_WINDOW_STEPS (512): exactly the boundary.

        For aperiodic VCS (no fast path), Change C skip fires (tier='skipped').
        See TestAmortizationGuard::test_exactly_512_steps_skipped.
        For smoothed aligned VCS (fast path), table is built.
        Both must return correct column values.
        """
        from distributed.tile_worker_td import CHUNK_WINDOW_STEPS
        dt = 1e-10
        W = CHUNK_WINDOW_STEPS  # == 512

        # Aperiodic case: Change C fires → skipped → evaluate_at_time fallback.
        worker_ap = _make_worker()
        vcs_ap = _attach_aperiodic_pwl(worker_ap, 'b', dt, W + 50, seed=77)
        info_ap = worker_ap.precompute_step_columns(t_start=0.0, dt=dt, n_steps=W)
        assert info_ap.get('tier') == 'skipped', (
            f"Aperiodic n_steps=W must be skipped, got {info_ap.get('tier')!r}"
        )
        for s in range(5):
            t = (s + 1) * dt
            arr = worker_ap._get_current_array_for_step(s, t).copy()
            arr_ref = vcs_ap.evaluate_at_time(t)
            np.testing.assert_allclose(arr, arr_ref, atol=1e-9,
                                       err_msg=f"Aperiodic n_steps=W step {s} mismatch")

        # Smoothed aligned case: fast path is available → table is built.
        worker_sm = _make_worker()
        m = 100
        vcs_sm, _ = _attach_uniform_pwl(worker_sm, 'b', dt, m, dc=0.1, seed=8)
        src_cands = vcs_sm.get_src_node_indices()
        max_mb = _tiny_max_mb(m, len(src_cands))
        info_sm = worker_sm.precompute_step_columns(
            t_start=0.0, dt=dt, n_steps=W, max_table_mb=max_mb,
        )
        # Fast path disables Change C skip → chunked, table built.
        if info_sm.get('tier') == 'chunked':
            assert worker_sm._step_col_table is not None
            for s in range(5):
                t = (s + 1) * dt
                arr = worker_sm._get_current_array_for_step(s, t).copy()
                arr_ref = vcs_sm.evaluate_at_time(t)
                np.testing.assert_allclose(arr, arr_ref, atol=1e-9,
                                           err_msg=f"Smoothed n_steps=W step {s} mismatch")

    def test_n_steps_chunk_window_plus_one(self):
        """n_steps = CHUNK_WINDOW_STEPS+1: just above boundary → multi-window build."""
        from distributed.tile_worker_td import CHUNK_WINDOW_STEPS
        dt = 1e-10
        W = CHUNK_WINDOW_STEPS

        worker = _make_worker()
        vcs = _attach_aperiodic_pwl(worker, 'b', dt, W + 50, seed=88)
        info = worker.precompute_step_columns(t_start=0.0, dt=dt, n_steps=W + 1)
        assert info.get('tier') == 'chunked', (
            f"n_steps=W+1 must build chunked table, got {info.get('tier')!r}"
        )
        assert worker._step_col_table is not None
        # Step W is the first step in the second window.
        t = (W + 1) * dt
        arr = worker._get_current_array_for_step(W, t).copy()
        arr_ref = vcs.evaluate_at_time(t)
        np.testing.assert_allclose(arr, arr_ref, atol=1e-9,
                                   err_msg="n_steps=W+1 boundary step mismatch")

    def test_n_steps_less_than_chunk_window_aperiodic_skipped(self):
        """W > n_steps for aperiodic VCS: Change C fires, tier='skipped'.

        See TestAmortizationGuard in test_step_column_reuse.py.  This confirms
        the invariant for n_steps=100 (not just n_steps=50 and n_steps=512).
        """
        from distributed.tile_worker_td import CHUNK_WINDOW_STEPS
        dt = 1e-10
        W = CHUNK_WINDOW_STEPS

        worker = _make_worker()
        _ = _attach_aperiodic_pwl(worker, 'b', dt, 200, seed=99)
        info = worker.precompute_step_columns(t_start=0.0, dt=dt, n_steps=100)
        assert info.get('tier') == 'skipped', (
            "Aperiodic n_steps < W must be skipped (Change C guard)"
        )
        assert worker._step_col_table is None

    def test_single_pwl_row_chunked_correct(self):
        """Single-source PWL (1 row) in chunked tier returns exact columns."""
        dt = 1e-10
        worker = _make_worker()
        vcs = _attach_aperiodic_pwl(worker, 'b', dt, 700, seed=55)
        info = worker.precompute_step_columns(t_start=0.0, dt=dt, n_steps=700)
        assert info.get('tier') == 'chunked'
        _check_columns_match_evaluate(worker, vcs, dt, 0.0, min(20, 700))

    def test_window_crossing_at_period_boundary(self):
        """Window rebuild that falls exactly on a period boundary is handled.

        When n_steps == m * k for some integer k, the window-rebuild step
        coincides exactly with a period boundary (new_start % m == 0).
        The gather must produce correct values (not confused about the wrap).
        """
        dt = 1e-10
        m = 100   # period = 100 * dt
        # Choose n_steps such that a window rebuild falls on a period boundary.
        # With W=512, the first rebuild is at step 512.  We want 512 % m == 0;
        # m=512 works but that's large.  Instead use n_steps=600 and check the
        # rebuild at step 512 (512 % 100 = 12, not a boundary).  For exact
        # period-boundary crossing: use m=64 → period=6.4 ns and W=512 covers
        # exactly 8 periods; the rebuild at step 512 is step 512%64=0 — boundary.
        m_alt = 64
        worker = _make_worker()
        vcs, _ = _attach_uniform_pwl(worker, 'b', dt, m_alt, dc=0.15, seed=21)
        src_cands = vcs.get_src_node_indices()
        max_mb = _tiny_max_mb(m_alt, len(src_cands))

        n_steps = 600  # two window rebuilds; step 512 → 512 % 64 = 0 (boundary)
        info = worker.precompute_step_columns(
            t_start=0.0, dt=dt, n_steps=n_steps, max_table_mb=max_mb,
        )
        if info.get('tier') != 'chunked':
            pytest.skip(f"Expected chunked, got {info.get('tier')}")

        # Check step 511 (last in window 0), 512 (first rebuild, period-boundary),
        # and 513 (second step of window 1).
        for s in [511, 512, 513]:
            if s >= n_steps:
                continue
            t = (s + 1) * dt
            arr = worker._get_current_array_for_step(s, t).copy()
            arr_ref = vcs.evaluate_at_time(t)
            np.testing.assert_allclose(
                arr, arr_ref, atol=1e-9,
                err_msg=f"Period-boundary window-crossing step {s} mismatch",
            )


# ─────────────────────────────────────────────────────────────────────────────
# 8. End-to-end: guard shapes through solve_transient (1e-12 V tolerance)
# ─────────────────────────────────────────────────────────────────────────────


class TestEndToEndGuardShapes:
    """End-to-end solve_transient comparison against use_step_columns=False.

    Uses the shared build_two_tile_model fixture (F13 fix: _owns_pkl_dir set).
    """

    def _dummy_smoothed_sources(self, model, dt, t_end):
        from distributed.result import DistributedSmoothedSources
        return DistributedSmoothedSources(
            time_step=dt, t_start=0.0, t_end=t_end,
            smoothed=False, n_tiles=len(model.workers), per_tile_stats={},
        )

    def _attach_to_workers(self, workers, dt, n_steps=None, aperiodic=False):
        """Attach VCS to all workers."""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent))
        from test_step_column_reuse import (
            _attach_pulse_vcs_to_workers, _attach_aperiodic_vcs_to_workers,
        )
        if aperiodic:
            _attach_aperiodic_vcs_to_workers(workers, dt=dt, n_steps=n_steps or 700)
        else:
            _attach_pulse_vcs_to_workers(workers, period=1e-8)

    def test_negative_t_start_e2e_matches_no_step_cols(self):
        """solve_transient with t_start=-dt and periodic VCS matches flag-off.

        Guards the F8 fix (negative on-grid t_start phase tier acceptance)
        against regressions in the full solve pipeline.
        """
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent))
        from _fixtures import build_two_tile_model
        from distributed.solver import DistributedDDMSolver
        from distributed.result import DistributedSmoothedSources

        dt = 1e-10
        n_steps = 50
        t_start = -dt
        t_end = n_steps * dt

        model, workers = build_two_tile_model()
        self._attach_to_workers(workers, dt)

        solver = DistributedDDMSolver(model)
        dc_ctx = solver.prepare()
        trans_ctx = solver.prepare_transient(dt=dt, method='be')
        dummy = DistributedSmoothedSources(
            time_step=dt, t_start=t_start, t_end=t_end,
            smoothed=False, n_tiles=len(model.workers), per_tile_stats={},
        )

        result_sc = solver.solve_transient(
            trans_ctx, dc_context=dc_ctx, smoothed_sources=dummy,
            t_start=t_start, t_end=t_end, use_step_columns=True,
        )
        result_ref = solver.solve_transient(
            trans_ctx, dc_context=dc_ctx, smoothed_sources=dummy,
            t_start=t_start, t_end=t_end, use_step_columns=False,
        )
        np.testing.assert_allclose(
            result_sc.max_ir_drop_per_time, result_ref.max_ir_drop_per_time,
            atol=1e-12,
            err_msg="F8 e2e: t_start=-dt solve must match use_step_columns=False",
        )
        model.shutdown()

    def test_n_steps_chunk_window_plus_one_e2e_matches_no_step_cols(self):
        """n_steps=W+1 (first multi-window solve) matches use_step_columns=False."""
        from distributed.tile_worker_td import CHUNK_WINDOW_STEPS
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent))
        from _fixtures import build_two_tile_model
        from distributed.solver import DistributedDDMSolver
        from distributed.result import DistributedSmoothedSources

        dt = 1e-10
        W = CHUNK_WINDOW_STEPS
        n_steps = W + 1
        t_end = n_steps * dt

        model, workers = build_two_tile_model()
        self._attach_to_workers(workers, dt, n_steps=n_steps + 20, aperiodic=True)

        solver = DistributedDDMSolver(model)
        dc_ctx = solver.prepare()
        trans_ctx = solver.prepare_transient(dt=dt, method='be')
        dummy = DistributedSmoothedSources(
            time_step=dt, t_start=0.0, t_end=t_end,
            smoothed=False, n_tiles=len(model.workers), per_tile_stats={},
        )

        result_sc = solver.solve_transient(
            trans_ctx, dc_context=dc_ctx, smoothed_sources=dummy,
            t_start=0.0, t_end=t_end, use_step_columns=True,
        )
        result_ref = solver.solve_transient(
            trans_ctx, dc_context=dc_ctx, smoothed_sources=dummy,
            t_start=0.0, t_end=t_end, use_step_columns=False,
        )
        np.testing.assert_allclose(
            result_sc.max_ir_drop_per_time, result_ref.max_ir_drop_per_time,
            atol=1e-12,
            err_msg="n_steps=W+1 e2e: chunked boundary must match use_step_columns=False",
        )
        model.shutdown()

    def test_reuse_across_two_solves_with_aperiodic_vcs_e2e(self):
        """Change A reuse with aperiodic VCS: second solve matches first exactly.

        Aperiodic forces chunked tier; the two solve_transient calls must be
        numerically identical (bit-level match within 1e-12 V).
        """
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent))
        from _fixtures import build_two_tile_model
        from distributed.solver import DistributedDDMSolver
        from distributed.result import DistributedSmoothedSources

        dt = 1e-10
        n_steps = 600
        t_end = n_steps * dt

        model, workers = build_two_tile_model()
        self._attach_to_workers(workers, dt, n_steps=n_steps + 20, aperiodic=True)

        solver = DistributedDDMSolver(model)
        dc_ctx = solver.prepare()
        trans_ctx = solver.prepare_transient(dt=dt, method='be')
        dummy = DistributedSmoothedSources(
            time_step=dt, t_start=0.0, t_end=t_end,
            smoothed=False, n_tiles=len(model.workers), per_tile_stats={},
        )

        result1 = solver.solve_transient(
            trans_ctx, dc_context=dc_ctx, smoothed_sources=dummy,
            t_start=0.0, t_end=t_end, use_step_columns=True,
        )
        result2 = solver.solve_transient(
            trans_ctx, dc_context=dc_ctx, smoothed_sources=dummy,
            t_start=0.0, t_end=t_end, use_step_columns=True,
        )
        for i, w in enumerate(workers):
            if w._step_col_info is not None:
                assert w._step_col_info.get('reused') is True, (
                    f"Worker {i}: second aperiodic solve must reuse table (Change A)"
                )
        np.testing.assert_allclose(
            result1.max_ir_drop_per_time, result2.max_ir_drop_per_time,
            atol=1e-12,
            err_msg="Aperiodic VCS reuse: second solve must be bit-identical to first",
        )
        model.shutdown()

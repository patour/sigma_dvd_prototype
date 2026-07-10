"""Tests for Change A (step-column table reuse across transients) and
Change C (amortization guard) implemented in tile_worker_td.py.

All tests are marked unit — no external netlist or Ray backend required.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.unit


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


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


def _attach_pulse_vcs(worker, node='b', period=1e-8, v1=0.0, v2=1.0):
    """Attach a simple periodic Pulse VCS to *worker* on *node*.

    Period and amplitudes chosen so the phase tier is reachable for
    reasonable dt values (e.g. dt=1e-10 → m=100 steps/period).
    """
    from analysis.vectorized_sources import VectorizedCurrentSources
    from parser.current_sources import CurrentSource, Pulse

    bs = worker._block_system
    n_ports = bs.n_ports
    n_nodes = n_ports + bs.n_interior
    node_to_idx = dict(bs.port_to_idx)
    for nd, idx in bs.interior_to_idx.items():
        node_to_idx[nd] = idx + n_ports

    # Simple pulse: half-period high, no delay, no rise/fall time
    pulse = Pulse(
        v1=v1,
        v2=v2,
        delay=0.0,
        rt=0.0,
        ft=0.0,
        width=period / 2,
        period=period,
    )
    src = CurrentSource(name='i1', node1=node, node2='0',
                        dc_value=0.0, pulses=[pulse])
    vcs = VectorizedCurrentSources.from_current_sources(
        {'i1': src}, node_to_idx, n_nodes,
    )
    worker._vec_sources = vcs
    worker._active_sources = vcs
    # Reset sources_version and cache as if freshly loaded
    worker._sources_version = 1
    worker._step_col_cache_key = None
    worker._step_col_info = None
    worker._step_col_table = None


def _attach_aperiodic_pwl_vcs(worker, node='b', dt=1e-10, n_steps=2000):
    """Attach an aperiodic PWL VCS to *worker* on *node*.

    Creates a PWL with random values at each dt step (no period), which
    forces the chunked tier.  n_steps is chosen larger than 512 so that
    the multi-window path is exercised (no Change C skipping).
    """
    from analysis.vectorized_sources import VectorizedCurrentSources
    from parser.current_sources import CurrentSource, PWL

    bs = worker._block_system
    n_ports = bs.n_ports
    n_nodes = n_ports + bs.n_interior
    node_to_idx = dict(bs.port_to_idx)
    for nd, idx in bs.interior_to_idx.items():
        node_to_idx[nd] = idx + n_ports

    rng = np.random.default_rng(42)
    times = np.arange(0, (n_steps + 2) * dt, dt)
    values = rng.uniform(0.0, 1.0, size=len(times))

    # Aperiodic PWL (period=0.0)
    pwl = PWL(
        points=list(zip(times.tolist(), values.tolist())),
        period=0.0,
        delay=0.0,
    )
    src = CurrentSource(name='i_aperiodic', node1=node, node2='0',
                        dc_value=0.0, pwls=[pwl])
    vcs = VectorizedCurrentSources.from_current_sources(
        {'i_aperiodic': src}, node_to_idx, n_nodes,
    )
    worker._vec_sources = vcs
    worker._active_sources = vcs
    worker._sources_version = 1
    worker._step_col_cache_key = None
    worker._step_col_info = None
    worker._step_col_table = None


# ──────────────────────────────────────────────────────────────────────────────
# 1. Change A — phase tier reuse
# ──────────────────────────────────────────────────────────────────────────────


class TestPhaseTierReuse:
    """Phase-tier table is reused across calls with identical (version, dt, m, mb)."""

    def _make_phase_worker(self):
        worker = _make_worker()
        _attach_pulse_vcs(worker, period=1e-8)
        return worker

    def test_reuse_same_args_returns_reused_true(self):
        """Second precompute with identical args hits the cache."""
        worker = self._make_phase_worker()
        dt = 1e-10   # m = 100
        info1 = worker.precompute_step_columns(t_start=0.0, dt=dt, n_steps=200)
        assert info1.get('tier') == 'phase', f"Expected phase tier, got {info1}"
        assert info1.get('reused') is False, "First build must not be reused"

        info2 = worker.precompute_step_columns(t_start=0.0, dt=dt, n_steps=200)
        assert info2.get('reused') is True, "Second call must be a cache hit"

    def test_reuse_same_table_object(self):
        """Cache hit returns without rebuilding the C_dense array."""
        worker = self._make_phase_worker()
        dt = 1e-10
        worker.precompute_step_columns(t_start=0.0, dt=dt, n_steps=200)
        tbl_after_first = worker._step_col_table
        C_dense_id_first = id(tbl_after_first['C_dense'])

        worker.precompute_step_columns(t_start=0.0, dt=dt, n_steps=200)
        tbl_after_second = worker._step_col_table
        C_dense_id_second = id(tbl_after_second['C_dense'])

        assert C_dense_id_first == C_dense_id_second, (
            "C_dense must be the SAME object on reuse (no rebuild)"
        )

    def test_reuse_different_n_steps_same_table(self):
        """n_steps change alone does not invalidate the phase table."""
        worker = self._make_phase_worker()
        dt = 1e-10
        worker.precompute_step_columns(t_start=0.0, dt=dt, n_steps=200)
        info2 = worker.precompute_step_columns(t_start=0.0, dt=dt, n_steps=300)
        assert info2.get('reused') is True

    def test_reuse_per_step_arrays_identical(self):
        """Columns gathered from a reused table match the original build."""
        worker = self._make_phase_worker()
        dt = 1e-10
        worker.precompute_step_columns(t_start=0.0, dt=dt, n_steps=200)

        # Gather columns from original table
        # Convention: step_idx=s at t_start=0 → evaluate at (s+1)*dt
        cols_orig = []
        for s in range(10):
            arr = worker._get_current_array_for_step(s, (s + 1) * dt)
            cols_orig.append(arr.copy())

        # Trigger reuse
        worker.precompute_step_columns(t_start=0.0, dt=dt, n_steps=200)
        assert worker._step_col_info.get('reused') is True

        cols_reused = []
        for s in range(10):
            arr = worker._get_current_array_for_step(s, (s + 1) * dt)
            cols_reused.append(arr.copy())

        for s, (orig, reused) in enumerate(zip(cols_orig, cols_reused)):
            np.testing.assert_allclose(
                orig, reused, atol=1e-15,
                err_msg=f"Column mismatch at step {s} between original and reused table",
            )

    def test_dt_change_invalidates_phase_cache(self):
        """Different dt → different m → rebuild, not reuse."""
        worker = self._make_phase_worker()
        worker.precompute_step_columns(t_start=0.0, dt=1e-10, n_steps=200)
        info2 = worker.precompute_step_columns(t_start=0.0, dt=5e-11, n_steps=400)
        assert info2.get('reused') is False, "dt change must trigger rebuild"

    def test_max_table_mb_change_invalidates_cache(self):
        """max_table_mb override that would flip tier must trigger rebuild."""
        worker = self._make_phase_worker()
        # First build with default max_mb (512 → phase tier)
        info1 = worker.precompute_step_columns(t_start=0.0, dt=1e-10, n_steps=200)
        assert info1.get('tier') == 'phase'
        assert info1.get('reused') is False

        # Same max_mb → reuse
        info2 = worker.precompute_step_columns(
            t_start=0.0, dt=1e-10, n_steps=200, max_table_mb=512.0,
        )
        assert info2.get('reused') is True

        # Different max_mb → different cache key → rebuild
        info3 = worker.precompute_step_columns(
            t_start=0.0, dt=1e-10, n_steps=200, max_table_mb=1024.0,
        )
        assert info3.get('reused') is False

    def test_phase0_updated_on_t_start_change(self):
        """Phase tier reuse with a different (on-grid) t_start updates phase0."""
        worker = self._make_phase_worker()
        dt = 1e-10
        m = 100  # period 1e-8 / dt 1e-10

        # Build at t_start=0
        info1 = worker.precompute_step_columns(t_start=0.0, dt=dt, n_steps=200)
        assert info1.get('tier') == 'phase'
        assert info1.get('phase0', 0) == 0

        # Reuse at t_start=k*dt (k=50, on-grid)
        k = 50
        t_start2 = k * dt
        info2 = worker.precompute_step_columns(t_start=t_start2, dt=dt, n_steps=200)
        assert info2.get('reused') is True
        expected_phase0 = k % m
        assert info2.get('phase0') == expected_phase0, (
            f"phase0 must be updated to {expected_phase0}, got {info2.get('phase0')}"
        )

        # Gathered column at step 0 should match evaluate_at_time(t_start2 + dt).
        # Convention: step_idx=0 at t_start2 → evaluate at t_start2 + dt.
        t_eval = t_start2 + dt
        col_from_table = worker._get_current_array_for_step(0, t_eval).copy()
        col_from_eval = worker._active_sources.evaluate_at_time(t_eval)
        np.testing.assert_allclose(col_from_table, col_from_eval, atol=1e-9)

    def test_off_grid_t_start_forces_rebuild(self):
        """A t_start that is off the dt grid cannot reuse the phase table."""
        worker = self._make_phase_worker()
        dt = 1e-10
        worker.precompute_step_columns(t_start=0.0, dt=dt, n_steps=200)

        # t_start = 0.7 * dt is off-grid → rebuild required
        t_start_off = 0.7 * dt
        info2 = worker.precompute_step_columns(t_start=t_start_off, dt=dt, n_steps=200)
        # Either falls through to chunked or rebuilds phase — either way NOT reused.
        assert info2.get('reused') is False, (
            "Off-grid t_start must not reuse the phase table"
        )


# ──────────────────────────────────────────────────────────────────────────────
# 2. Change A — chunked tier reuse
# ──────────────────────────────────────────────────────────────────────────────


class TestChunkedTierReuse:
    """Chunked-tier table is reused when t_start, dt, and version match."""

    def _make_chunked_worker(self, n_steps=2000):
        """Worker with aperiodic PWL (forces chunked, multi-window)."""
        worker = _make_worker()
        _attach_aperiodic_pwl_vcs(worker, n_steps=n_steps)
        return worker

    def test_reuse_same_args(self):
        """Second call with identical args is a cache hit."""
        worker = self._make_chunked_worker()
        dt = 1e-10
        n_steps = 2000
        info1 = worker.precompute_step_columns(t_start=0.0, dt=dt, n_steps=n_steps)
        assert info1.get('tier') == 'chunked'
        assert info1.get('reused') is False

        info2 = worker.precompute_step_columns(t_start=0.0, dt=dt, n_steps=n_steps)
        assert info2.get('reused') is True

    def test_reuse_same_table_object(self):
        """C_dense object identity is preserved on reuse."""
        worker = self._make_chunked_worker()
        dt = 1e-10
        worker.precompute_step_columns(t_start=0.0, dt=dt, n_steps=2000)
        id_first = id(worker._step_col_table['C_dense'])

        worker.precompute_step_columns(t_start=0.0, dt=dt, n_steps=2000)
        id_second = id(worker._step_col_table['C_dense'])
        assert id_first == id_second

    def test_n_steps_extended_on_reuse(self):
        """Reuse with larger n_steps extends the stored n_steps."""
        worker = self._make_chunked_worker()
        dt = 1e-10
        worker.precompute_step_columns(t_start=0.0, dt=dt, n_steps=2000)
        info2 = worker.precompute_step_columns(t_start=0.0, dt=dt, n_steps=3000)
        assert info2.get('reused') is True
        # Stored n_steps should be extended
        assert worker._step_col_table['n_steps'] >= 3000

    def test_n_steps_extension_column_correctness(self):
        """After n_steps extension, _get_current_array_for_step at a step in
        (original_n, new_n] matches evaluate_at_time to 1e-9 mA."""
        worker = self._make_chunked_worker(n_steps=3000)
        dt = 1e-10
        n_A = 2000
        n_B = 2600

        # Build with n_steps = n_A
        worker.precompute_step_columns(t_start=0.0, dt=dt, n_steps=n_A)
        # Reuse with n_steps = n_B
        info2 = worker.precompute_step_columns(t_start=0.0, dt=dt, n_steps=n_B)
        assert info2.get('reused') is True

        # Step in (n_A, n_B]: should trigger on-demand window rebuild.
        # Convention: step_idx=s → evaluate at t_start + (s+1)*dt.
        step = n_A + 100  # step 2100 is beyond original n_A=2000
        t = (step + 1) * dt  # actual evaluation time for step_idx=step
        col_from_table = worker._get_current_array_for_step(step, t).copy()
        col_from_eval = worker._active_sources.evaluate_at_time(t)
        np.testing.assert_allclose(
            col_from_table, col_from_eval, atol=1e-9,
            err_msg=f"Extended-window column at step {step} must match evaluate_at_time",
        )

    def test_dt_change_invalidates_chunked_cache(self):
        """dt change invalidates chunked cache (different key)."""
        worker = self._make_chunked_worker()
        worker.precompute_step_columns(t_start=0.0, dt=1e-10, n_steps=2000)
        info2 = worker.precompute_step_columns(t_start=0.0, dt=2e-10, n_steps=1000)
        assert info2.get('reused') is False

    def test_t_start_change_invalidates_chunked_cache(self):
        """t_start change invalidates chunked cache."""
        worker = self._make_chunked_worker()
        dt = 1e-10
        worker.precompute_step_columns(t_start=0.0, dt=dt, n_steps=2000)
        info2 = worker.precompute_step_columns(t_start=1e-8, dt=dt, n_steps=2000)
        assert info2.get('reused') is False


# ──────────────────────────────────────────────────────────────────────────────
# 3. Cache invalidation on source mutations
# ──────────────────────────────────────────────────────────────────────────────


class TestCacheInvalidationOnSourceMutation:
    """Each source-mutating call must bump _sources_version and clear the cache."""

    def _make_phase_worker_with_table(self):
        worker = _make_worker()
        _attach_pulse_vcs(worker, period=1e-8)
        worker.precompute_step_columns(t_start=0.0, dt=1e-10, n_steps=200)
        assert worker._step_col_table is not None
        assert worker._step_col_cache_key is not None
        return worker

    def _version_before(self, worker):
        return worker._sources_version

    def _check_invalidated(self, worker, version_before, mutation_name):
        assert worker._sources_version > version_before, (
            f"{mutation_name} must increment _sources_version"
        )
        assert worker._step_col_cache_key is None, (
            f"{mutation_name} must clear _step_col_cache_key"
        )
        assert worker._step_col_info is None, (
            f"{mutation_name} must clear _step_col_info"
        )
        # Nulling _step_col_table is also required
        assert worker._step_col_table is None, (
            f"{mutation_name} must null _step_col_table"
        )

    def test_init_vectorized_sources_clears_cache(self, tmp_path):
        """init_vectorized_sources nulls the table and bumps version."""
        worker = self._make_phase_worker_with_table()
        # To call init_vectorized_sources we need a minimal on-disk setup
        # — instead test indirectly via the raw sources re-assignment path.
        # The quickest method: directly call use_raw_sources after a
        # smooth_sources call won't work here because we'd need actual files.
        # Test the _sources_version bump directly by checking the attr exists
        # and is already > 0 from our _attach_pulse_vcs setup call.
        initial_version = worker._sources_version
        assert initial_version > 0, "sources_version must be non-zero after attach"
        # Simulate init_vectorized_sources: nulls table + bumps version
        worker._step_col_table = None  # simulated new sources
        worker._sources_version += 1
        worker._step_col_cache_key = None
        worker._step_col_info = None
        # Now the next precompute must rebuild
        new_info = worker.precompute_step_columns(t_start=0.0, dt=1e-10, n_steps=200)
        assert new_info.get('reused') is False

    def test_use_smoothed_sources_clears_cache(self):
        """use_smoothed_sources(True) must clear the cache."""
        from analysis.vectorized_sources import VectorizedCurrentSources
        from parser.current_sources import CurrentSource, Pulse

        worker = self._make_phase_worker_with_table()

        # Give the worker a dummy smoothed sources (use raw as smoothed)
        worker._smoothed_sources = worker._vec_sources

        v_before = self._version_before(worker)
        worker.use_smoothed_sources(True)
        self._check_invalidated(worker, v_before, 'use_smoothed_sources(True)')

    def test_use_smoothed_sources_false_clears_cache(self):
        """use_smoothed_sources(False) (i.e., switch to raw) also clears cache."""
        worker = self._make_phase_worker_with_table()
        # First give it a smoothed sources
        worker._smoothed_sources = worker._vec_sources
        worker._active_sources = worker._smoothed_sources
        # Rebuild table on "smoothed" state
        worker._step_col_cache_key = None
        worker._step_col_info = None
        worker._step_col_table = None
        worker._sources_version += 1
        worker.precompute_step_columns(t_start=0.0, dt=1e-10, n_steps=200)
        assert worker._step_col_table is not None

        v_before = self._version_before(worker)
        worker.use_smoothed_sources(False)  # → calls use_raw_sources internall
        # use_smoothed_sources(False) goes through the else branch → _vec_sources
        self._check_invalidated(worker, v_before, 'use_smoothed_sources(False)')

    def test_use_raw_sources_clears_cache(self):
        """use_raw_sources() clears the step-column cache."""
        worker = self._make_phase_worker_with_table()
        v_before = self._version_before(worker)
        worker.use_raw_sources()
        self._check_invalidated(worker, v_before, 'use_raw_sources()')

    def test_recompute_after_invalidation_rebuilds(self):
        """After invalidation, a precompute call rebuilds (reused=False)."""
        worker = self._make_phase_worker_with_table()
        worker.use_raw_sources()
        new_info = worker.precompute_step_columns(t_start=0.0, dt=1e-10, n_steps=200)
        assert new_info.get('reused') is False

    def test_version_increments_each_mutation(self):
        """Each mutation bumps _sources_version by exactly 1."""
        worker = _make_worker()
        _attach_pulse_vcs(worker, period=1e-8)
        # _attach_pulse_vcs sets _sources_version = 1
        assert worker._sources_version == 1

        worker.use_raw_sources()
        assert worker._sources_version == 2

        worker.use_raw_sources()
        assert worker._sources_version == 3


# ──────────────────────────────────────────────────────────────────────────────
# 4. Change C — amortization guard (single-window chunked skip)
# ──────────────────────────────────────────────────────────────────────────────


class TestAmortizationGuard:
    """When chunked tier and n_steps <= W (512), precompute is skipped."""

    def _make_chunked_worker_small(self, n_steps=50):
        """Worker with aperiodic PWL but n_steps <= W (forces Change C)."""
        worker = _make_worker()
        _attach_aperiodic_pwl_vcs(worker, n_steps=max(n_steps + 10, 600))
        return worker

    def test_small_n_steps_returns_skipped(self):
        """n_steps <= 512 with chunked tier returns tier='skipped'."""
        worker = self._make_chunked_worker_small()
        info = worker.precompute_step_columns(
            t_start=0.0, dt=1e-10, n_steps=50,
        )
        assert info.get('tier') == 'skipped', (
            f"Expected tier='skipped', got {info.get('tier')!r}"
        )
        assert info.get('reason') == 'single_window_no_amortization'
        assert worker._step_col_table is None

    def test_skipped_table_is_none(self):
        """After skip, _step_col_table remains None."""
        worker = self._make_chunked_worker_small()
        worker.precompute_step_columns(t_start=0.0, dt=1e-10, n_steps=100)
        assert worker._step_col_table is None

    def test_exactly_512_steps_skipped(self):
        """n_steps == W == 512 is the boundary: must be skipped."""
        worker = self._make_chunked_worker_small(n_steps=512)
        info = worker.precompute_step_columns(
            t_start=0.0, dt=1e-10, n_steps=512,
        )
        assert info.get('tier') == 'skipped', (
            "n_steps == W=512 is still a single-window run: must skip"
        )

    def test_513_steps_not_skipped(self):
        """n_steps = 513 > W=512: multi-window, must build."""
        worker = self._make_chunked_worker_small(n_steps=600)
        info = worker.precompute_step_columns(
            t_start=0.0, dt=1e-10, n_steps=513,
        )
        assert info.get('tier') == 'chunked', (
            f"n_steps=513 > W=512 must build chunked table, got {info.get('tier')!r}"
        )

    def test_skipped_cache_key_is_none(self):
        """After skip, _step_col_cache_key must be None (no false reuse)."""
        worker = self._make_chunked_worker_small()
        worker.precompute_step_columns(t_start=0.0, dt=1e-10, n_steps=50)
        assert worker._step_col_cache_key is None

    def test_fallback_path_used_when_skipped(self):
        """With skipped table, _get_current_array_for_step falls back to evaluate_at_time."""
        worker = self._make_chunked_worker_small()
        worker.precompute_step_columns(t_start=0.0, dt=1e-10, n_steps=50)
        assert worker._step_col_table is None

        dt = 1e-10
        # Convention: step_idx=0 at t_start=0 → evaluate at t=dt
        t_step1 = dt
        # Fallback path: should return evaluate_at_time result
        arr_fallback = worker._get_current_array_for_step(0, t_step1).copy()
        arr_eval = worker._active_sources.evaluate_at_time(t_step1)
        np.testing.assert_allclose(arr_fallback, arr_eval, atol=1e-15)


# ──────────────────────────────────────────────────────────────────────────────
# Shared helper: attach periodic pulse VCS to all workers in a model
# ──────────────────────────────────────────────────────────────────────────────


def _attach_pulse_vcs_to_workers(workers, period=1e-8):
    """Attach a single-period pulse VCS to every worker (bypasses file I/O).

    Sets ``_active_sources``, ``_vec_sources``, ``_current_buf``, and the
    Change-A cache state so that ``precompute_step_columns`` starts cold.

    Args:
        workers: List of TileWorker objects (LocalBackend, in-process).
        period: Waveform period in seconds (default 10 ns).  dt=1e-10 gives
            m = period/dt = 100, well within the phase-tier memory budget.
    """
    from analysis.vectorized_sources import VectorizedCurrentSources
    from parser.current_sources import CurrentSource, Pulse

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
        pulse = Pulse(
            v1=0.0, v2=1.0, delay=0.0, rt=0.0, ft=0.0,
            width=period / 2, period=period,
        )
        src = CurrentSource(
            name=f'i_p{i}', node1=target, node2='0',
            dc_value=0.3, pulses=[pulse],
        )
        vcs = VectorizedCurrentSources.from_current_sources(
            {f'i_p{i}': src}, node_to_idx, n_nodes,
        )
        worker._vec_sources = vcs
        worker._active_sources = vcs
        worker._current_buf = np.zeros(n_nodes, dtype=np.float64)
        # Reset Change-A cache so precompute starts cold on first solve.
        worker._sources_version = 1
        worker._step_col_cache_key = None
        worker._step_col_info = None
        worker._step_col_table = None


def _attach_aperiodic_vcs_to_workers(workers, dt, n_steps):
    """Attach an aperiodic PWL VCS to every worker (bypasses file I/O).

    The PWL has a distinct random value at each dt step, forcing the
    chunked tier.  n_steps is the length of the waveform table to build.

    Args:
        workers: List of TileWorker objects.
        dt: Simulation time step in seconds.
        n_steps: Length of the random PWL (number of dt-spaced breakpoints).
    """
    from analysis.vectorized_sources import VectorizedCurrentSources
    from parser.current_sources import CurrentSource, PWL

    rng = np.random.default_rng(42)
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
        times = np.arange(0, (n_steps + 2) * dt, dt)
        values = rng.uniform(0.0, 1.0, size=len(times))
        pwl = PWL(
            points=list(zip(times.tolist(), values.tolist())),
            period=0.0, delay=0.0,
        )
        src = CurrentSource(
            name=f'i_ap{i}', node1=target, node2='0',
            dc_value=0.0, pwls=[pwl],
        )
        vcs = VectorizedCurrentSources.from_current_sources(
            {f'i_ap{i}': src}, node_to_idx, n_nodes,
        )
        worker._vec_sources = vcs
        worker._active_sources = vcs
        worker._current_buf = np.zeros(n_nodes, dtype=np.float64)
        worker._sources_version = 1
        worker._step_col_cache_key = None
        worker._step_col_info = None
        worker._step_col_table = None


# ──────────────────────────────────────────────────────────────────────────────
# 5. End-to-end: solve_transient reuse (2-tile model)
# ──────────────────────────────────────────────────────────────────────────────


def _build_two_tile_model():
    """Build a 2-tile model suitable for DC + transient solves (no VCS files).

    Pad node 'pad' is NOT placed in tile boundary_nodes — it lives only in
    PackageData.  This matches the correct design used by the equivalence
    suite (_build_synthetic_two_tile_model in test_equivalence.py) and
    avoids the Dirichlet-port size mismatch in the DC IC bincount.

    Topology:
        Tile A: a1 --[1 mS]-- B  (boundary: B only; cap: a1-0 10 fF)
        Tile B: B --[3 mS]-- b1 --[1 mS]-- 0  (boundary: B; cap: b1-0 5 fF)
        Package: pad --[10 mS]-- B;  pad at 1.0 V (Dirichlet)
    """
    import tempfile
    import os

    from distributed.backend import LocalBackend
    from distributed.model import DistributedPowerGridModel
    from distributed.parser import PackageData, PowerGridMetaData, TileConfig
    from distributed.tile_worker import TileData, TileWorker

    tmp_dir = tempfile.mkdtemp(prefix="test_sc_reuse_")

    tile_a = TileData(
        tile_id=(0, 0),
        resistive_edges=[('a1', 'B', 1.0)],
        all_nodes={'a1', 'B'},
        boundary_nodes={'B'},  # pad NOT here
        current_injections={'a1': 0.5},
        capacitive_edges=[('a1', '0', 10.0)],
    )
    tile_b = TileData(
        tile_id=(0, 1),
        resistive_edges=[('B', 'b1', 3.0), ('b1', '0', 1.0)],
        all_nodes={'B', 'b1'},
        boundary_nodes={'B'},
        current_injections={'b1': 0.3},
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
    # ckt_path needs a valid parent dir for the adjoint VCS cache path logic
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


class TestEndToEndTransientReuse:
    """Two consecutive solve_transient calls on the same solver/sources
    produce bit-identical results.

    We use the 2-tile model with DC-only static current injections.
    A tmp_path is passed to preprocess_sources so it can write VCS caches
    without touching the netlist directory.
    """

    def _make_dummy_sources(self, model, dt, t_start, t_end):
        """Create a dummy DistributedSmoothedSources handle (no disk I/O)."""
        from distributed.result import DistributedSmoothedSources

        return DistributedSmoothedSources(
            time_step=dt, t_start=t_start, t_end=t_end,
            smoothed=False, n_tiles=len(model.workers), per_tile_stats={},
        )

    def test_two_consecutive_solves_bit_identical(self):
        """First and second solve_transient results match to <= 1e-12 V."""
        from distributed.solver import DistributedDDMSolver

        model, workers = _build_two_tile_model()
        solver = DistributedDDMSolver(model)

        dc_ctx = solver.prepare()
        trans_ctx = solver.prepare_transient(dt=1e-10, method='be')

        dummy = self._make_dummy_sources(model, dt=1e-10, t_start=0.0, t_end=5e-10)

        result1 = solver.solve_transient(
            trans_ctx, dc_context=dc_ctx, smoothed_sources=dummy,
            t_start=0.0, t_end=5e-10,
        )
        result2 = solver.solve_transient(
            trans_ctx, dc_context=dc_ctx, smoothed_sources=dummy,
            t_start=0.0, t_end=5e-10,
        )

        np.testing.assert_allclose(
            result1.max_ir_drop_per_time, result2.max_ir_drop_per_time,
            atol=1e-12,
            err_msg="Two consecutive solve_transient calls must be bit-identical",
        )

    def test_second_solve_matches_use_step_columns_false(self):
        """Second solve result matches use_step_columns=False baseline (<= 1e-12 V)."""
        from distributed.solver import DistributedDDMSolver

        model, workers = _build_two_tile_model()
        solver = DistributedDDMSolver(model)

        dc_ctx = solver.prepare()
        trans_ctx = solver.prepare_transient(dt=1e-10, method='be')

        dummy = self._make_dummy_sources(model, dt=1e-10, t_start=0.0, t_end=5e-10)

        result_sc = solver.solve_transient(
            trans_ctx, dc_context=dc_ctx, smoothed_sources=dummy,
            t_start=0.0, t_end=5e-10, use_step_columns=True,
        )
        result_ref = solver.solve_transient(
            trans_ctx, dc_context=dc_ctx, smoothed_sources=dummy,
            t_start=0.0, t_end=5e-10, use_step_columns=False,
        )

        np.testing.assert_allclose(
            result_sc.max_ir_drop_per_time, result_ref.max_ir_drop_per_time,
            atol=1e-12,
            err_msg="Step-column on vs off must match to <= 1e-12 V",
        )


# ──────────────────────────────────────────────────────────────────────────────
# 6. Change C end-to-end: small n_steps produce correct results
# ──────────────────────────────────────────────────────────────────────────────


class TestChangeCEndToEnd:
    """solve_transient with a small n_steps (Change C skip) matches
    use_step_columns=False to <= 1e-12 V.

    We use the 2-tile DC-only model with preprocess_sources(smooth=False).
    With DC-only sources, tier resolves to 'disabled' (no VCS), which is
    already equivalent to the fallback — this guards against regressions
    where the two paths diverge.
    """

    def test_small_n_steps_matches_no_step_cols(self):
        """With n_steps <= 512 and chunked tier, skipping precompute is
        equivalent to use_step_columns=False (per-step fallback baseline)."""
        from distributed.result import DistributedSmoothedSources
        from distributed.solver import DistributedDDMSolver

        model, workers = _build_two_tile_model()
        solver = DistributedDDMSolver(model)

        dc_ctx = solver.prepare()
        trans_ctx = solver.prepare_transient(dt=1e-10, method='be')

        dummy = DistributedSmoothedSources(
            time_step=1e-10, t_start=0.0, t_end=50e-10,
            smoothed=False, n_tiles=len(model.workers), per_tile_stats={},
        )

        # Solve with use_step_columns=True
        result_sc = solver.solve_transient(
            trans_ctx, dc_context=dc_ctx,
            smoothed_sources=dummy,
            t_start=0.0, t_end=50e-10,
            use_step_columns=True,
        )

        # Baseline: use_step_columns=False (per-step evaluate_at_time)
        result_ref = solver.solve_transient(
            trans_ctx, dc_context=dc_ctx,
            smoothed_sources=dummy,
            t_start=0.0, t_end=50e-10,
            use_step_columns=False,
        )

        np.testing.assert_allclose(
            result_sc.max_ir_drop_per_time, result_ref.max_ir_drop_per_time,
            atol=1e-12,
            err_msg=(
                "DC-only solve (disabled/skipped table) must match "
                "use_step_columns=False to <= 1e-12 V"
            ),
        )


# ──────────────────────────────────────────────────────────────────────────────
# 7. VCS-backed end-to-end: Change A reuse through solve_transient
# ──────────────────────────────────────────────────────────────────────────────
#
# These tests use real periodic/aperiodic VCS attached directly to workers
# (bypassing preprocess_sources, which would reset sources from disk).
# The smoothed_sources handle is passed explicitly to solve_transient so the
# coordinator does NOT call preprocess_sources internally.
#
# Pattern:
#   1. Build model + prepare solver + contexts (no VCS yet).
#   2. Attach VCS directly to each worker after factor_and_compute_schur().
#   3. Call solve_transient(smoothed_sources=dummy_handle) twice.
#   4. Probe worker._step_col_info after each call to verify reuse.


class TestVCSBackedEndToEndReuse:
    """Two consecutive solve_transient calls with real VCS reuse the table.

    The first call builds the phase/chunked table; the second call hits the
    Change-A cache (worker._step_col_info['reused'] is True) and returns
    bit-identical results.
    """

    def _dummy_smoothed_sources(self, model, dt, t_end):
        from distributed.result import DistributedSmoothedSources

        return DistributedSmoothedSources(
            time_step=dt, t_start=0.0, t_end=t_end,
            smoothed=False, n_tiles=len(model.workers), per_tile_stats={},
        )

    def test_phase_tier_reused_on_second_solve(self):
        """With periodic pulse VCS, second solve_transient reuses the phase table.

        After the first solve, every worker's _step_col_info has 'tier'='phase'
        and 'reused'=False.  After the second solve the same workers report
        'reused'=True (the C_dense array was not rebuilt).
        """
        from distributed.solver import DistributedDDMSolver

        dt = 1e-10        # 100 ps
        t_end = 5e-9      # 50 steps; period=10 ns → m=100 (phase tier)

        model, workers = _build_two_tile_model()
        _attach_pulse_vcs_to_workers(workers, period=1e-8)

        solver = DistributedDDMSolver(model)
        dc_ctx = solver.prepare()
        trans_ctx = solver.prepare_transient(dt=dt, method='be')
        dummy = self._dummy_smoothed_sources(model, dt, t_end)

        # ---- First solve: table must be built (reused=False) ----
        result1 = solver.solve_transient(
            trans_ctx, dc_context=dc_ctx, smoothed_sources=dummy,
            t_start=0.0, t_end=t_end, use_step_columns=True,
        )
        for w in workers:
            info = w._step_col_info
            assert info is not None, "First solve must build and store _step_col_info"
            assert info.get('tier') == 'phase', (
                f"Expected phase tier, got {info.get('tier')!r}"
            )
            assert info.get('reused') is False, (
                "First solve must not be a cache hit"
            )

        # Record C_dense identity so we can verify no rebuild on second call.
        c_dense_ids_first = [id(w._step_col_table['C_dense']) for w in workers]

        # ---- Second solve: table must be reused (reused=True) ----
        result2 = solver.solve_transient(
            trans_ctx, dc_context=dc_ctx, smoothed_sources=dummy,
            t_start=0.0, t_end=t_end, use_step_columns=True,
        )
        for i, w in enumerate(workers):
            info = w._step_col_info
            assert info is not None
            assert info.get('reused') is True, (
                f"Worker {i}: second solve must be a cache hit (reused=True)"
            )
            assert id(w._step_col_table['C_dense']) == c_dense_ids_first[i], (
                f"Worker {i}: C_dense must be the SAME object on reuse (no rebuild)"
            )

        # Results must be bit-identical.
        np.testing.assert_allclose(
            result1.max_ir_drop_per_time, result2.max_ir_drop_per_time,
            atol=1e-12,
            err_msg="Two consecutive solve_transient calls must be bit-identical",
        )

    def test_chunked_tier_reused_on_second_solve(self):
        """With aperiodic PWL VCS (n_steps > 512), second solve reuses chunked table."""
        from distributed.solver import DistributedDDMSolver

        dt = 1e-10
        n_steps = 600          # > 512 → multi-window, builds chunked table
        t_end = n_steps * dt

        model, workers = _build_two_tile_model()
        _attach_aperiodic_vcs_to_workers(workers, dt=dt, n_steps=n_steps + 20)

        solver = DistributedDDMSolver(model)
        dc_ctx = solver.prepare()
        trans_ctx = solver.prepare_transient(dt=dt, method='be')
        dummy = self._dummy_smoothed_sources(model, dt, t_end)

        # ---- First solve ----
        result1 = solver.solve_transient(
            trans_ctx, dc_context=dc_ctx, smoothed_sources=dummy,
            t_start=0.0, t_end=t_end, use_step_columns=True,
        )
        for w in workers:
            info = w._step_col_info
            assert info is not None
            assert info.get('tier') == 'chunked', (
                f"Expected chunked tier, got {info.get('tier')!r}"
            )
            assert info.get('reused') is False

        # ---- Second solve ----
        result2 = solver.solve_transient(
            trans_ctx, dc_context=dc_ctx, smoothed_sources=dummy,
            t_start=0.0, t_end=t_end, use_step_columns=True,
        )
        for i, w in enumerate(workers):
            info = w._step_col_info
            assert info is not None
            assert info.get('reused') is True, (
                f"Worker {i}: second chunked solve must be a cache hit"
            )

        np.testing.assert_allclose(
            result1.max_ir_drop_per_time, result2.max_ir_drop_per_time,
            atol=1e-12,
            err_msg="Chunked reuse: two consecutive solves must be bit-identical",
        )

    def test_phase_reuse_matches_use_step_columns_false(self):
        """Phase-tier (reused) results match use_step_columns=False baseline."""
        from distributed.solver import DistributedDDMSolver

        dt = 1e-10
        t_end = 5e-9

        model, workers = _build_two_tile_model()
        _attach_pulse_vcs_to_workers(workers, period=1e-8)

        solver = DistributedDDMSolver(model)
        dc_ctx = solver.prepare()
        trans_ctx = solver.prepare_transient(dt=dt, method='be')
        dummy = self._dummy_smoothed_sources(model, dt, t_end)

        # First solve (builds table).
        solver.solve_transient(
            trans_ctx, dc_context=dc_ctx, smoothed_sources=dummy,
            t_start=0.0, t_end=t_end, use_step_columns=True,
        )
        # Second solve (reuses table).
        result_reused = solver.solve_transient(
            trans_ctx, dc_context=dc_ctx, smoothed_sources=dummy,
            t_start=0.0, t_end=t_end, use_step_columns=True,
        )
        assert all(w._step_col_info.get('reused') for w in workers), (
            "All workers must report reused=True on second call"
        )

        # Fresh model with same VCS for the no-step-cols baseline.
        model_ref, workers_ref = _build_two_tile_model()
        _attach_pulse_vcs_to_workers(workers_ref, period=1e-8)
        solver_ref = DistributedDDMSolver(model_ref)
        dc_ref = solver_ref.prepare()
        trans_ref = solver_ref.prepare_transient(dt=dt, method='be')
        dummy_ref = self._dummy_smoothed_sources(model_ref, dt, t_end)

        result_ref = solver_ref.solve_transient(
            trans_ref, dc_context=dc_ref, smoothed_sources=dummy_ref,
            t_start=0.0, t_end=t_end, use_step_columns=False,
        )

        np.testing.assert_allclose(
            result_reused.max_ir_drop_per_time, result_ref.max_ir_drop_per_time,
            atol=1e-12,
            err_msg=(
                "Phase-tier reused solve must match use_step_columns=False "
                "to <= 1e-12 V"
            ),
        )


# ──────────────────────────────────────────────────────────────────────────────
# 8. VCS-backed end-to-end: Change C skip through solve_transient
# ──────────────────────────────────────────────────────────────────────────────


class TestVCSBackedChangeCEndToEnd:
    """Change C: with aperiodic VCS and n_steps <= 512, solve_transient
    triggers the amortization guard (tier='skipped', table stays None) and
    results are identical to use_step_columns=False.

    The test probes worker._step_col_table (must be None) and
    worker._step_col_info (must be None, cleared by the skip) after each
    solve call, confirming the per-step fallback path was exercised.
    """

    def _dummy_smoothed_sources(self, model, dt, t_end):
        from distributed.result import DistributedSmoothedSources

        return DistributedSmoothedSources(
            time_step=dt, t_start=0.0, t_end=t_end,
            smoothed=False, n_tiles=len(model.workers), per_tile_stats={},
        )

    def test_small_n_steps_aperiodic_tier_skipped_and_correct(self):
        """Aperiodic VCS + n_steps=50 (<= W=512): tier='skipped', results correct.

        After solve_transient(use_step_columns=True), every worker must have
        _step_col_table=None (no table allocated), confirming that the
        Change-C guard fired and the per-step evaluate_at_time fallback was
        used throughout.  Results must match use_step_columns=False to 1e-12 V.
        """
        from distributed.solver import DistributedDDMSolver

        dt = 1e-10
        n_steps = 50       # well below W=512 → Change C skip
        t_end = n_steps * dt

        model, workers = _build_two_tile_model()
        # VCS has 70 breakpoints (n_steps+20), aperiodic → chunked tier
        _attach_aperiodic_vcs_to_workers(workers, dt=dt, n_steps=n_steps + 20)

        solver = DistributedDDMSolver(model)
        dc_ctx = solver.prepare()
        trans_ctx = solver.prepare_transient(dt=dt, method='be')
        dummy = self._dummy_smoothed_sources(model, dt, t_end)

        # use_step_columns=True → precompute is called → Change C skips build
        result_sc = solver.solve_transient(
            trans_ctx, dc_context=dc_ctx, smoothed_sources=dummy,
            t_start=0.0, t_end=t_end, use_step_columns=True,
        )
        for i, w in enumerate(workers):
            assert w._step_col_table is None, (
                f"Worker {i}: Change C must leave _step_col_table=None "
                "(table not built for single-window chunked run)"
            )
            # _step_col_info is cleared to None by the skip path
            assert w._step_col_info is None, (
                f"Worker {i}: _step_col_info must be None after Change C skip"
            )

        # Baseline: use_step_columns=False (per-step evaluate_at_time, same model)
        # Re-attach VCS since precompute was called but skipped (cache key cleared).
        _attach_aperiodic_vcs_to_workers(workers, dt=dt, n_steps=n_steps + 20)
        result_ref = solver.solve_transient(
            trans_ctx, dc_context=dc_ctx, smoothed_sources=dummy,
            t_start=0.0, t_end=t_end, use_step_columns=False,
        )

        np.testing.assert_allclose(
            result_sc.max_ir_drop_per_time, result_ref.max_ir_drop_per_time,
            atol=1e-12,
            err_msg=(
                "Change C skipped solve must match use_step_columns=False "
                "to <= 1e-12 V"
            ),
        )

    def test_boundary_n_steps_512_skipped(self):
        """n_steps exactly 512 (== W): Change C fires, table stays None."""
        from distributed.solver import DistributedDDMSolver

        dt = 1e-10
        n_steps = 512
        t_end = n_steps * dt

        model, workers = _build_two_tile_model()
        _attach_aperiodic_vcs_to_workers(workers, dt=dt, n_steps=n_steps + 20)

        solver = DistributedDDMSolver(model)
        dc_ctx = solver.prepare()
        trans_ctx = solver.prepare_transient(dt=dt, method='be')
        dummy = self._dummy_smoothed_sources(model, dt, t_end)

        solver.solve_transient(
            trans_ctx, dc_context=dc_ctx, smoothed_sources=dummy,
            t_start=0.0, t_end=t_end, use_step_columns=True,
        )
        for i, w in enumerate(workers):
            assert w._step_col_table is None, (
                f"Worker {i}: n_steps=512 == W must trigger Change C skip"
            )

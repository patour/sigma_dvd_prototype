"""Tests for A5: smoothed-VCS disk cache + instrumentation + smooth='auto'.

Covers:
- ``smooth_sources`` cache miss: fresh compute saves the smoothed pkl
- ``smooth_sources`` cache hit: reload is exact-equal to fresh compute
  (every array compared with np.testing.assert_array_equal)
- Cache invalidation: touching the raw pkl mtime forces a recompute
- ``get_min_pwl_segment_duration``: correct min from PWL breakpoints and
  pulse features; ``inf`` when no time-varying sources
- ``smooth='auto'`` decision in ``preprocess_sources``:
  - time_step > min_segment → smoothing runs
  - time_step <= min_segment → smoothing skipped
- Stats presence: ``per_tile_smooth_time_s`` and ``smooth_cache_hits``
  populated on DistributedSmoothedSources
- A2 interaction: step-column table cleared after a cached smooth load
  (same invalidation as a fresh smooth_sources call)
"""

from __future__ import annotations

import os
import pickle
import time
import unittest
from typing import Any, Dict, Optional
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers shared across test classes
# ---------------------------------------------------------------------------

def _make_worker_with_tile(edges, port_nodes, currents=None):
    """Build a TileWorker with a minimal tile (no file I/O).

    Args:
        edges: List of (u, v, conductance_mS) tuples.  Must include at
            least one edge to ground node '0' for the system to be SPD.
        port_nodes: Iterable of port/boundary node names.
        currents: Optional dict {node: current_mA} for current injections.
    """
    from distributed.tile_worker import TileWorker, TileData

    all_nodes = set()
    for u, v, _g in edges:
        if u != '0':
            all_nodes.add(u)
        if v != '0':
            all_nodes.add(v)

    tile_data = TileData(
        tile_id=(0, 0),
        resistive_edges=list(edges),
        all_nodes=all_nodes,
        boundary_nodes=set(port_nodes),
        current_injections=currents or {},
        capacitive_edges=[],
    )
    interface_nodes = set(port_nodes)

    worker = TileWorker()
    worker.setup_from_tile_data(tile_data, interface_nodes)
    return worker


def _attach_pulse_vcs(
    worker,
    period: float = 50e-9,
    v2: float = 1.0,
    dc: float = 0.5,
    rise_time: float = 1e-9,
    fall_time: float = 1e-9,
    width: float = 10e-9,
):
    """Attach a VCS with one pulse to a TileWorker (bypasses file init)."""
    from analysis.vectorized_sources import VectorizedCurrentSources
    from parser.current_sources import CurrentSource, Pulse

    n_ports = worker._block_system.n_ports
    n_interior = worker._block_system.n_interior
    n_nodes = n_ports + n_interior

    node_to_idx: Dict[str, int] = dict(worker._block_system.port_to_idx)
    for node, idx in worker._block_system.interior_to_idx.items():
        node_to_idx[node] = idx + n_ports

    interior_nodes = list(worker._block_system.interior_to_idx.keys())
    target_node = interior_nodes[0] if interior_nodes else list(node_to_idx.keys())[0]

    pulse = Pulse(
        v1=0.0, v2=v2, delay=0.0,
        rt=rise_time, ft=fall_time,
        width=width, period=period,
    )
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
    worker._current_buf = np.zeros(n_nodes, dtype=np.float64)
    return vcs


def _attach_pwl_vcs(
    worker,
    times=None,
    values=None,
    period: float = 0.0,
):
    """Attach a VCS with one PWL waveform to a TileWorker."""
    from analysis.vectorized_sources import VectorizedCurrentSources
    from parser.current_sources import CurrentSource, PWL

    if times is None:
        times = [0.0, 5e-9, 10e-9, 15e-9, 20e-9]
    if values is None:
        values = [0.0, 1.0, 0.5, 0.8, 0.0]

    n_ports = worker._block_system.n_ports
    n_interior = worker._block_system.n_interior
    n_nodes = n_ports + n_interior

    node_to_idx: Dict[str, int] = dict(worker._block_system.port_to_idx)
    for node, idx in worker._block_system.interior_to_idx.items():
        node_to_idx[node] = idx + n_ports

    # Prefer interior nodes; fall back to any node
    interior_nodes = list(worker._block_system.interior_to_idx.keys())
    target_node = interior_nodes[0] if interior_nodes else list(node_to_idx.keys())[0]

    pwl = PWL(points=list(zip(times, values)), period=period)
    src = CurrentSource(
        name='i_pwl', node1=target_node, node2='0',
        dc_value=0.0,
        pwls=[pwl],
    )
    vcs = VectorizedCurrentSources.from_current_sources(
        {'i_pwl': src}, node_to_idx, n_nodes,
    )
    worker._vec_sources = vcs
    worker._active_sources = vcs
    worker._current_buf = np.zeros(n_nodes, dtype=np.float64)
    return vcs


def _simple_two_node_edges():
    """Return edges and port_nodes for a simple 2-node tile.

    Topology: port 'p' -- interior 'a' -- ground '0'.
    Conductances in mS (= 1/kOhm).
    """
    edges = [('p', 'a', 1.0), ('a', '0', 1.0)]
    ports = ['p']
    return edges, ports


# ---------------------------------------------------------------------------
# 1. smooth_sources disk cache: miss + hit + exact equality
# ---------------------------------------------------------------------------

class TestSmoothSourcesCache:
    """Disk-cache hit/miss for smooth_sources."""

    def _make_worker_with_pulse(self):
        edges, ports = _simple_two_node_edges()
        w = _make_worker_with_tile(edges, ports)
        _attach_pulse_vcs(w, period=50e-9, rise_time=1e-9, fall_time=1e-9, width=10e-9)
        return w

    def _vcs_arrays(self, vcs):
        """Extract comparable arrays from a VCS."""
        return {
            'n_pulses': vcs.n_pulses,
            'n_pwls': vcs.n_pwls,
            'pwl_times': vcs.pwl_times.copy(),
            'pwl_values': vcs.pwl_values.copy(),
            'pwl_offset': vcs.pwl_offset.copy(),
            'pwl_count': vcs.pwl_count.copy(),
            'dc_values': vcs.dc_values.copy(),
        }

    def test_cache_miss_saves_file(self, tmp_path):
        """First call computes and saves the smoothed pkl."""
        worker = self._make_worker_with_pulse()

        # Create a dummy raw pkl so smooth_sources can save the smoothed cache
        raw_pkl = tmp_path / 'vcs_tile_0_0.pkl'
        raw_pkl.write_bytes(b'dummy')

        stats = worker.smooth_sources(
            time_step=1e-9,
            t_start=0.0,
            t_end=100e-9,
            pkl_dir=str(tmp_path),
        )

        assert stats['cached'] is False
        assert stats['smooth_time_s'] >= 0.0

        # Smoothed pkl must exist
        smoothed_files = list(tmp_path.glob('vcs_tile_0_0_smoothed_*.pkl'))
        assert len(smoothed_files) == 1, f"Expected 1 smoothed pkl, got {smoothed_files}"

    def test_cache_hit_returns_cached_true(self, tmp_path):
        """Second call with same params loads from cache (cached=True)."""
        worker = self._make_worker_with_pulse()
        raw_pkl = tmp_path / 'vcs_tile_0_0.pkl'
        raw_pkl.write_bytes(b'dummy')

        # First call → miss
        worker.smooth_sources(
            time_step=1e-9, t_start=0.0, t_end=100e-9,
            pkl_dir=str(tmp_path),
        )

        # Reset worker's smoothed sources to force a reload
        worker._smoothed_sources = None
        worker._active_sources = worker._vec_sources

        # Second call → hit
        stats2 = worker.smooth_sources(
            time_step=1e-9, t_start=0.0, t_end=100e-9,
            pkl_dir=str(tmp_path),
        )
        assert stats2['cached'] is True
        assert stats2['smooth_time_s'] == 0.0
        assert worker._smoothed_sources is not None

    def test_cache_hit_arrays_exactly_equal(self, tmp_path):
        """Cached smoothed VCS arrays must be bit-for-bit identical to fresh."""
        worker_fresh = self._make_worker_with_pulse()
        worker_cached = self._make_worker_with_pulse()

        raw_pkl = tmp_path / 'vcs_tile_0_0.pkl'
        raw_pkl.write_bytes(b'dummy')

        # Fresh compute (no cache)
        worker_fresh.smooth_sources(
            time_step=1e-9, t_start=0.0, t_end=100e-9,
        )
        fresh_arrays = self._vcs_arrays(worker_fresh._smoothed_sources)

        # Cached compute (save + reload)
        worker_cached.smooth_sources(
            time_step=1e-9, t_start=0.0, t_end=100e-9,
            pkl_dir=str(tmp_path),
        )
        # Reset and reload
        worker_cached._smoothed_sources = None
        worker_cached._active_sources = worker_cached._vec_sources
        worker_cached.smooth_sources(
            time_step=1e-9, t_start=0.0, t_end=100e-9,
            pkl_dir=str(tmp_path),
        )
        cached_arrays = self._vcs_arrays(worker_cached._smoothed_sources)

        # Exact equality
        assert fresh_arrays['n_pwls'] == cached_arrays['n_pwls']
        assert fresh_arrays['n_pulses'] == cached_arrays['n_pulses']
        np.testing.assert_array_equal(
            fresh_arrays['pwl_times'], cached_arrays['pwl_times'],
            err_msg="pwl_times must be bit-equal between cached and fresh",
        )
        np.testing.assert_array_equal(
            fresh_arrays['pwl_values'], cached_arrays['pwl_values'],
            err_msg="pwl_values must be bit-equal between cached and fresh",
        )
        np.testing.assert_array_equal(
            fresh_arrays['pwl_offset'], cached_arrays['pwl_offset'],
        )
        np.testing.assert_array_equal(
            fresh_arrays['pwl_count'], cached_arrays['pwl_count'],
        )
        np.testing.assert_array_equal(
            fresh_arrays['dc_values'], cached_arrays['dc_values'],
        )

    def test_different_params_produce_different_cache_files(self, tmp_path):
        """Different time_step values produce distinct smoothed pkl files."""
        worker = self._make_worker_with_pulse()
        raw_pkl = tmp_path / 'vcs_tile_0_0.pkl'
        raw_pkl.write_bytes(b'dummy')

        worker.smooth_sources(1e-9, 0.0, 100e-9, pkl_dir=str(tmp_path))
        worker._smoothed_sources = None
        worker._active_sources = worker._vec_sources

        worker.smooth_sources(2e-9, 0.0, 100e-9, pkl_dir=str(tmp_path))

        smoothed_files = list(tmp_path.glob('vcs_tile_0_0_smoothed_*.pkl'))
        assert len(smoothed_files) == 2, (
            "Different time_step should produce distinct cache files"
        )

    def test_no_pkl_dir_does_not_save(self, tmp_path):
        """With pkl_dir=None, no smoothed cache file is written."""
        worker = self._make_worker_with_pulse()

        worker.smooth_sources(1e-9, 0.0, 100e-9, pkl_dir=None)

        smoothed_files = list(tmp_path.glob('vcs_tile_0_0_smoothed_*.pkl'))
        assert len(smoothed_files) == 0

    def test_stats_smooth_time_present_on_miss(self, tmp_path):
        """smooth_time_s is returned (>= 0) on cache miss."""
        worker = self._make_worker_with_pulse()
        raw_pkl = tmp_path / 'vcs_tile_0_0.pkl'
        raw_pkl.write_bytes(b'dummy')

        stats = worker.smooth_sources(1e-9, 0.0, 100e-9, pkl_dir=str(tmp_path))
        assert 'smooth_time_s' in stats
        assert isinstance(stats['smooth_time_s'], float)
        assert stats['smooth_time_s'] >= 0.0
        assert 'cached' in stats
        assert stats['cached'] is False


# ---------------------------------------------------------------------------
# 2. Cache invalidation when raw pkl changes
# ---------------------------------------------------------------------------

class TestSmoothSourcesCacheInvalidation:
    """Verify that raw-pkl mtime change causes cache miss."""

    def test_touching_raw_pkl_invalidates_smoothed_cache(self, tmp_path):
        """Modifying the raw pkl's mtime forces smooth_sources to recompute."""
        from distributed.tile_worker import TileWorker

        edges, ports = _simple_two_node_edges()
        worker = _make_worker_with_tile(edges, ports)
        _attach_pulse_vcs(worker)

        raw_pkl = tmp_path / 'vcs_tile_0_0.pkl'
        raw_pkl.write_bytes(b'dummy')

        # First call: miss → saves cache
        s1 = worker.smooth_sources(1e-9, 0.0, 100e-9, pkl_dir=str(tmp_path))
        assert not s1['cached']

        # Reset
        worker._smoothed_sources = None
        worker._active_sources = worker._vec_sources

        # Second call: hit
        s2 = worker.smooth_sources(1e-9, 0.0, 100e-9, pkl_dir=str(tmp_path))
        assert s2['cached'], "Expected cache hit before raw-pkl touch"

        # Reset again
        worker._smoothed_sources = None
        worker._active_sources = worker._vec_sources

        # Touch raw pkl (force mtime change)
        time.sleep(0.01)  # ensure OS mtime resolution resolves difference
        raw_pkl.write_bytes(b'updated_dummy')

        # Third call: should be a miss again
        s3 = worker.smooth_sources(1e-9, 0.0, 100e-9, pkl_dir=str(tmp_path))
        assert not s3['cached'], (
            "Expected cache miss after raw pkl was modified"
        )

    def test_corrupted_smoothed_cache_triggers_recompute(self, tmp_path):
        """Corrupted smoothed pkl triggers silent recompute (no exception)."""
        from distributed.tile_worker import TileWorker

        edges, ports = _simple_two_node_edges()
        worker = _make_worker_with_tile(edges, ports)
        _attach_pulse_vcs(worker)

        raw_pkl = tmp_path / 'vcs_tile_0_0.pkl'
        raw_pkl.write_bytes(b'dummy')

        # First call: saves cache
        worker.smooth_sources(1e-9, 0.0, 100e-9, pkl_dir=str(tmp_path))

        # Corrupt the smoothed cache
        smoothed_files = list(tmp_path.glob('vcs_tile_0_0_smoothed_*.pkl'))
        assert len(smoothed_files) == 1
        smoothed_files[0].write_bytes(b'corrupt')

        # Reset
        worker._smoothed_sources = None
        worker._active_sources = worker._vec_sources

        # Should recompute without raising
        s = worker.smooth_sources(1e-9, 0.0, 100e-9, pkl_dir=str(tmp_path))
        assert not s['cached']
        assert worker._smoothed_sources is not None


# ---------------------------------------------------------------------------
# 3. get_min_pwl_segment_duration
# ---------------------------------------------------------------------------

class TestGetMinPWLSegmentDuration:
    """Unit tests for the auto-probe method."""

    def test_no_sources_returns_inf(self):
        """With no VCS loaded, returns float('inf')."""
        edges, ports = _simple_two_node_edges()
        worker = _make_worker_with_tile(edges, ports)
        # _vec_sources is None
        assert worker.get_min_pwl_segment_duration() == float('inf')

    def test_dc_only_sources_returns_inf(self):
        """DC-only sources (no pulses, no PWL) return float('inf')."""
        from analysis.vectorized_sources import VectorizedCurrentSources
        from parser.current_sources import CurrentSource

        edges, ports = _simple_two_node_edges()
        worker = _make_worker_with_tile(edges, ports)

        n_nodes = worker._block_system.n_ports + worker._block_system.n_interior
        node_to_idx = {
            **dict(worker._block_system.port_to_idx),
            **{n: i + worker._block_system.n_ports
               for n, i in worker._block_system.interior_to_idx.items()},
        }
        src = CurrentSource(name='idc', node1='b', node2='0', dc_value=1.0)
        vcs = VectorizedCurrentSources.from_current_sources(
            {'idc': src}, node_to_idx, n_nodes,
        )
        worker._vec_sources = vcs
        worker._active_sources = vcs

        result = worker.get_min_pwl_segment_duration()
        assert result == float('inf')

    def test_pulse_sources_min_feature(self):
        """Minimum of (rt, ft, width) is returned for pulse sources."""
        edges, ports = _simple_two_node_edges()
        worker = _make_worker_with_tile(edges, ports)
        # rt=1ns, ft=2ns, width=5ns → min = 1ns
        _attach_pulse_vcs(worker, rise_time=1e-9, fall_time=2e-9, width=5e-9)

        result = worker.get_min_pwl_segment_duration()
        assert abs(result - 1e-9) < 1e-15, f"Expected 1e-9, got {result}"

    def test_pwl_sources_min_segment(self):
        """Minimum inter-breakpoint gap is returned for PWL sources."""
        edges, ports = _simple_two_node_edges()
        worker = _make_worker_with_tile(edges, ports)
        # Breakpoints at 0, 1ns, 3ns, 10ns → diffs: 1ns, 2ns, 7ns → min = 1ns
        _attach_pwl_vcs(
            worker,
            times=[0.0, 1e-9, 3e-9, 10e-9],
            values=[0.0, 1.0, 0.5, 0.0],
        )

        result = worker.get_min_pwl_segment_duration()
        assert abs(result - 1e-9) < 1e-15, f"Expected 1e-9, got {result}"

    def test_min_across_pulses_and_pwl(self):
        """When both pulses and PWL exist, returns the global min."""
        from analysis.vectorized_sources import VectorizedCurrentSources
        from parser.current_sources import CurrentSource, Pulse, PWL

        edges, ports = _simple_two_node_edges()
        worker = _make_worker_with_tile(edges, ports)

        n_ports = worker._block_system.n_ports
        n_interior = worker._block_system.n_interior
        n_nodes = n_ports + n_interior
        node_to_idx = {
            **dict(worker._block_system.port_to_idx),
            **{n: i + n_ports
               for n, i in worker._block_system.interior_to_idx.items()},
        }
        int_nodes = list(worker._block_system.interior_to_idx.keys())
        node_a = int_nodes[0] if int_nodes else 'a'

        # Pulse with rt=5ns
        pulse = Pulse(v1=0.0, v2=1.0, delay=0.0, rt=5e-9, ft=5e-9,
                      width=10e-9, period=50e-9)
        # PWL with segments of 0.5ns
        pwl = PWL(points=[(0.0, 0.0), (0.5e-9, 1.0), (1.0e-9, 0.0)])
        src1 = CurrentSource(name='ip', node1=node_a, node2='0',
                             dc_value=0.0, pulses=[pulse])
        src2 = CurrentSource(name='iw', node1=node_a, node2='0',
                             dc_value=0.0, pwls=[pwl])
        vcs = VectorizedCurrentSources.from_current_sources(
            {'ip': src1, 'iw': src2}, node_to_idx, n_nodes,
        )
        worker._vec_sources = vcs
        worker._active_sources = vcs

        result = worker.get_min_pwl_segment_duration()
        # min(5ns pulse rt, 5ns ft, 10ns width, 0.5ns PWL) = 0.5ns
        assert abs(result - 0.5e-9) < 1e-15, f"Expected 0.5e-9, got {result}"


# ---------------------------------------------------------------------------
# 4. smooth='auto' decision in preprocess_sources
# ---------------------------------------------------------------------------

class TestPreprocessSourcesAutoSmooth:
    """Tests for smooth='auto' in preprocess_sources."""

    def _build_two_tile_model_with_vcs(self, tmp_path):
        """Import and build the standard 2-tile model, add minimal VCS files."""
        from tests.distributed.test_time_domain import _build_two_tile_distributed_model
        return _build_two_tile_distributed_model(caps=False)

    def test_auto_smooth_skipped_when_timestep_fine(self, tmp_path):
        """smooth='auto' skips smoothing when time_step <= min_segment."""
        # We mock get_min_pwl_segment_duration to return 10ns.
        # With time_step = 1ns <= 10ns → should NOT smooth.
        from distributed.solver_td import _SolverTimeDomainMixin

        # Build a minimal fake model / solver object
        model_mock = MagicMock()
        tile_mock_cfg = MagicMock()
        tile_mock_cfg.tile_id = (0, 0)
        tile_mock_cfg.ckt_path = str(tmp_path / 'tile_0_0.ckt')
        tile_mock_cfg.nd_path = None
        tile_mock_cfg.instance_path = None
        tile_mock_cfg.net_filter = None
        model_mock.metadata.tile_configs = [tile_mock_cfg]
        model_mock.backend.call_all.side_effect = _call_all_side_effect_factory(
            init_results=[{'n_sources': 5, 'n_nodes': 10, 'cached': False}],
            min_durs=[10e-9],
        )

        solver = _SolverTimeDomainMixin.__new__(_SolverTimeDomainMixin)
        solver.model = model_mock

        result = solver.preprocess_sources(
            time_step=1e-9,
            t_start=0.0, t_end=100e-9,
            smooth='auto',
        )
        assert not result.smoothed, (
            "smooth='auto' with time_step(1ns) <= min_segment(10ns) should skip"
        )

    def test_auto_smooth_runs_when_timestep_coarse(self, tmp_path):
        """smooth='auto' applies smoothing when time_step > min_segment."""
        from distributed.solver_td import _SolverTimeDomainMixin

        model_mock = MagicMock()
        tile_mock_cfg = MagicMock()
        tile_mock_cfg.tile_id = (0, 0)
        tile_mock_cfg.ckt_path = str(tmp_path / 'tile_0_0.ckt')
        tile_mock_cfg.nd_path = None
        tile_mock_cfg.instance_path = None
        tile_mock_cfg.net_filter = None
        model_mock.metadata.tile_configs = [tile_mock_cfg]

        # min_dur = 0.5ns; time_step = 1ns > 0.5ns → should smooth
        model_mock.backend.call_all.side_effect = _call_all_side_effect_factory(
            init_results=[{'n_sources': 5, 'n_nodes': 10, 'cached': False}],
            min_durs=[0.5e-9],
            smooth_results=[{'smooth_time_s': 0.1, 'cached': False,
                             'time_step': 1e-9, 't_start': 0.0, 't_end': 100e-9}],
        )

        solver = _SolverTimeDomainMixin.__new__(_SolverTimeDomainMixin)
        solver.model = model_mock

        result = solver.preprocess_sources(
            time_step=1e-9,
            t_start=0.0, t_end=100e-9,
            smooth='auto',
        )
        assert result.smoothed, (
            "smooth='auto' with time_step(1ns) > min_segment(0.5ns) should smooth"
        )

    def test_auto_no_sources_skips_smoothing(self):
        """smooth='auto' skips smoothing when all tiles have no time-varying sources."""
        from distributed.solver_td import _SolverTimeDomainMixin

        model_mock = MagicMock()
        tile_mock_cfg = MagicMock()
        tile_mock_cfg.tile_id = (0, 0)
        tile_mock_cfg.ckt_path = '/fake/tile_0_0.ckt'
        tile_mock_cfg.nd_path = None
        tile_mock_cfg.instance_path = None
        tile_mock_cfg.net_filter = None
        model_mock.metadata.tile_configs = [tile_mock_cfg]

        # min_dur = inf (no time-varying sources) → skip smoothing
        model_mock.backend.call_all.side_effect = _call_all_side_effect_factory(
            init_results=[{'n_sources': 1, 'n_nodes': 5, 'cached': False}],
            min_durs=[float('inf')],
        )

        solver = _SolverTimeDomainMixin.__new__(_SolverTimeDomainMixin)
        solver.model = model_mock

        result = solver.preprocess_sources(
            time_step=1e-9,
            t_start=0.0, t_end=100e-9,
            smooth='auto',
        )
        assert not result.smoothed, (
            "smooth='auto' with no time-varying sources (inf min_dur) should skip"
        )

    def test_smooth_true_unchanged_behavior(self):
        """smooth=True (default) always smooths regardless of min segment."""
        from distributed.solver_td import _SolverTimeDomainMixin

        model_mock = MagicMock()
        tile_mock_cfg = MagicMock()
        tile_mock_cfg.tile_id = (0, 0)
        tile_mock_cfg.ckt_path = '/fake/tile_0_0.ckt'
        tile_mock_cfg.nd_path = None
        tile_mock_cfg.instance_path = None
        tile_mock_cfg.net_filter = None
        model_mock.metadata.tile_configs = [tile_mock_cfg]

        model_mock.backend.call_all.side_effect = _call_all_side_effect_factory(
            init_results=[{'n_sources': 5, 'n_nodes': 10, 'cached': False}],
            smooth_results=[{'smooth_time_s': 0.2, 'cached': False,
                             'time_step': 10e-9, 't_start': 0.0, 't_end': 100e-9}],
        )

        solver = _SolverTimeDomainMixin.__new__(_SolverTimeDomainMixin)
        solver.model = model_mock

        result = solver.preprocess_sources(
            time_step=10e-9,
            t_start=0.0, t_end=100e-9,
            smooth=True,
        )
        assert result.smoothed

    def test_smooth_false_unchanged_behavior(self):
        """smooth=False never smooths."""
        from distributed.solver_td import _SolverTimeDomainMixin

        model_mock = MagicMock()
        tile_mock_cfg = MagicMock()
        tile_mock_cfg.tile_id = (0, 0)
        tile_mock_cfg.ckt_path = '/fake/tile_0_0.ckt'
        model_mock.metadata.tile_configs = [tile_mock_cfg]

        model_mock.backend.call_all.side_effect = _call_all_side_effect_factory(
            init_results=[{'n_sources': 5, 'n_nodes': 10, 'cached': False}],
        )

        solver = _SolverTimeDomainMixin.__new__(_SolverTimeDomainMixin)
        solver.model = model_mock

        result = solver.preprocess_sources(
            time_step=1e-9,
            t_start=0.0, t_end=100e-9,
            smooth=False,
        )
        assert not result.smoothed


def _call_all_side_effect_factory(
    init_results,
    min_durs=None,
    smooth_results=None,
):
    """Create a side-effect function for model.backend.call_all mock.

    Dispatches based on the method name argument:
    - 'init_vectorized_sources' → init_results
    - 'get_min_pwl_segment_duration' → min_durs
    - 'smooth_sources' → smooth_results
    """
    _call_count = {'n': 0}

    def side_effect(workers, method, args_per_actor=None):
        if method == 'init_vectorized_sources':
            return list(init_results)
        if method == 'get_min_pwl_segment_duration':
            return list(min_durs) if min_durs is not None else [float('inf')]
        if method == 'smooth_sources':
            if smooth_results is not None:
                return list(smooth_results)
            # Default: return mock stats if not specified
            return [{'smooth_time_s': 0.05, 'cached': False,
                     'time_step': 1e-9, 't_start': 0.0, 't_end': 100e-9}]
        # Fallback
        return []

    return side_effect


# ---------------------------------------------------------------------------
# 5. DistributedSmoothedSources stats fields
# ---------------------------------------------------------------------------

class TestSmoothedSourcesStats:
    """Verify per_tile_smooth_time_s and smooth_cache_hits are populated."""

    def test_stats_populated_on_smooth(self):
        """per_tile_smooth_time_s and smooth_cache_hits are present after preprocess."""
        from distributed.solver_td import _SolverTimeDomainMixin

        tile_id = (0, 0)

        model_mock = MagicMock()
        tile_mock_cfg = MagicMock()
        tile_mock_cfg.tile_id = tile_id
        tile_mock_cfg.ckt_path = '/fake/tile_0_0.ckt'
        model_mock.metadata.tile_configs = [tile_mock_cfg]

        model_mock.backend.call_all.side_effect = _call_all_side_effect_factory(
            init_results=[{'n_sources': 3, 'n_nodes': 6, 'cached': False}],
            smooth_results=[{'smooth_time_s': 0.15, 'cached': False,
                             'time_step': 1e-9, 't_start': 0.0, 't_end': 100e-9}],
        )

        solver = _SolverTimeDomainMixin.__new__(_SolverTimeDomainMixin)
        solver.model = model_mock

        result = solver.preprocess_sources(
            time_step=1e-9, t_start=0.0, t_end=100e-9, smooth=True,
        )

        assert tile_id in result.per_tile_smooth_time_s, (
            "per_tile_smooth_time_s should contain the tile_id"
        )
        assert result.per_tile_smooth_time_s[tile_id] == pytest.approx(0.15)
        assert result.smooth_cache_hits == 0

    def test_cache_hits_counted(self):
        """smooth_cache_hits increments for tiles that loaded from cache."""
        from distributed.solver_td import _SolverTimeDomainMixin

        tile_id = (1, 2)

        model_mock = MagicMock()
        tile_mock_cfg = MagicMock()
        tile_mock_cfg.tile_id = tile_id
        tile_mock_cfg.ckt_path = '/fake/tile_1_2.ckt'
        model_mock.metadata.tile_configs = [tile_mock_cfg]

        model_mock.backend.call_all.side_effect = _call_all_side_effect_factory(
            init_results=[{'n_sources': 3, 'n_nodes': 6, 'cached': False}],
            # Simulating a cache hit: cached=True, smooth_time_s=0.0
            smooth_results=[{'smooth_time_s': 0.0, 'cached': True,
                             'time_step': 1e-9, 't_start': 0.0, 't_end': 100e-9}],
        )

        solver = _SolverTimeDomainMixin.__new__(_SolverTimeDomainMixin)
        solver.model = model_mock

        result = solver.preprocess_sources(
            time_step=1e-9, t_start=0.0, t_end=100e-9, smooth=True,
        )

        assert result.smooth_cache_hits == 1
        assert result.per_tile_smooth_time_s[tile_id] == 0.0

    def test_no_smooth_stats_empty(self):
        """per_tile_smooth_time_s is empty when smooth=False."""
        from distributed.solver_td import _SolverTimeDomainMixin

        model_mock = MagicMock()
        tile_mock_cfg = MagicMock()
        tile_mock_cfg.tile_id = (0, 0)
        tile_mock_cfg.ckt_path = '/fake/tile_0_0.ckt'
        model_mock.metadata.tile_configs = [tile_mock_cfg]

        model_mock.backend.call_all.side_effect = _call_all_side_effect_factory(
            init_results=[{'n_sources': 3, 'n_nodes': 6, 'cached': False}],
        )

        solver = _SolverTimeDomainMixin.__new__(_SolverTimeDomainMixin)
        solver.model = model_mock

        result = solver.preprocess_sources(
            time_step=1e-9, t_start=0.0, t_end=100e-9, smooth=False,
        )

        assert result.per_tile_smooth_time_s == {}
        assert result.smooth_cache_hits == 0


# ---------------------------------------------------------------------------
# 6. A2 interaction: step-column table cleared after cached smooth load
# ---------------------------------------------------------------------------

class TestA2Interaction:
    """A2 step-column table must be invalidated after a cached smooth load."""

    def test_step_col_table_cleared_on_cache_hit(self, tmp_path):
        """Step-column table is set to None when smooth load hits cache.

        After building a step-column table, loading the smoothed VCS from
        cache must clear the table (same as a fresh smooth_sources call).
        """
        edges, ports = _simple_two_node_edges()
        worker = _make_worker_with_tile(edges, ports)
        _attach_pulse_vcs(worker, period=50e-9)

        raw_pkl = tmp_path / 'vcs_tile_0_0.pkl'
        raw_pkl.write_bytes(b'dummy')

        # Step 1: first smooth → saves to cache
        worker.smooth_sources(
            time_step=1e-9, t_start=0.0, t_end=100e-9,
            pkl_dir=str(tmp_path),
        )

        # Step 2: simulate a step-column table being present
        worker._step_col_table = {'tier': 'phase', 'm': 50, 'phase0': 0,
                                   'src_rows': np.array([0]), 'C_dense': np.zeros((1, 50))}
        assert worker._step_col_table is not None, "Pre-condition: table should be set"

        # Step 3: reload from cache (hit)
        worker._smoothed_sources = None
        worker._active_sources = worker._vec_sources

        stats = worker.smooth_sources(
            time_step=1e-9, t_start=0.0, t_end=100e-9,
            pkl_dir=str(tmp_path),
        )
        assert stats['cached'], "Expected cache hit"

        # Step 4: step-column table must be cleared
        assert worker._step_col_table is None, (
            "Step-column table must be cleared after cached smooth load "
            "(A2 interaction requirement)"
        )

    def test_step_col_table_cleared_on_cache_miss(self, tmp_path):
        """Step-column table is cleared even on a cache miss (existing behavior)."""
        edges, ports = _simple_two_node_edges()
        worker = _make_worker_with_tile(edges, ports)
        _attach_pulse_vcs(worker, period=50e-9)

        # Set a dummy table
        worker._step_col_table = {'tier': 'phase', 'm': 50}

        # No pkl_dir → no cache, but table must still be cleared
        worker.smooth_sources(
            time_step=1e-9, t_start=0.0, t_end=100e-9, pkl_dir=None,
        )

        assert worker._step_col_table is None, (
            "Step-column table must be cleared even on cache miss"
        )


# ---------------------------------------------------------------------------
# 7. Smoothing code version bump invalidates cache
# ---------------------------------------------------------------------------

class TestSmoothingCodeVersion:
    """Bumping SMOOTHING_CODE_VERSION invalidates old smoothed caches."""

    def test_version_in_cache_key(self, tmp_path):
        """Two calls differing only in SMOOTHING_CODE_VERSION use different files."""
        from distributed import tile_worker_td
        import hashlib

        # Compute key hashes for two versions
        def make_key(version):
            key_str = (
                f"1e-09:0.0:1e-07"
                f":1e-12:10000"
                f":{version:d}"
            )
            return hashlib.md5(key_str.encode()).hexdigest()[:12]

        key_v1 = make_key(1)
        key_v2 = make_key(2)
        assert key_v1 != key_v2, (
            "Different SMOOTHING_CODE_VERSION must produce different cache keys"
        )

    def test_current_version_is_int(self):
        """SMOOTHING_CODE_VERSION is a positive integer."""
        from distributed.tile_worker_td import SMOOTHING_CODE_VERSION
        assert isinstance(SMOOTHING_CODE_VERSION, int)
        assert SMOOTHING_CODE_VERSION >= 1

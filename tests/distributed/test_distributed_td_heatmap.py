"""Tests for time-domain distributed heatmap pipeline.

Covers TileWorker.prebin_peak_data, TileWorker.get_top_k_peaks,
and the plot_distributed_td_heatmaps orchestrator.
"""

import os
import tempfile
import unittest
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock

import numpy as np
import pytest

from distributed.heatmap import (
    GlobalBinSpec,
    LayerBinSpec,
    plot_distributed_td_heatmaps,
)

pytestmark = pytest.mark.unit


# ──────────────────────────────────────────────────────────────────────
# Helpers (duplicated from test_distributed_heatmap.py for independence)
# ──────────────────────────────────────────────────────────────────────

def _make_single_layer_metadata(
    layer: str = 'M1',
    bbox: Tuple[float, float, float, float] = (0.0, 100.0, 0.0, 200.0),
    n_nodes: int = 100,
    stripe_coords_h: Optional[List[float]] = None,
    stripe_coords_v: Optional[List[float]] = None,
    edge_orientation: Tuple[int, int, int] = (50, 10, 5),
) -> Dict[str, Dict]:
    """Build a single-tile, single-layer metadata dict."""
    return {
        layer: {
            'bbox': bbox,
            'n_nodes': n_nodes,
            'stripe_coords_h': stripe_coords_h if stripe_coords_h is not None else [10.0, 20.0, 30.0],
            'stripe_coords_v': stripe_coords_v if stripe_coords_v is not None else [5.0, 15.0, 25.0],
            'edge_orientation': edge_orientation,
        }
    }


@dataclass
class FakeTileConfig:
    tile_id: Tuple[int, int]


def _make_mock_model(
    tile_configs: List[Any],
    tile_boundary_nodes: Dict[Tuple[int, int], List[str]],
) -> MagicMock:
    """Build a mock DistributedPowerGridModel."""
    model = MagicMock()
    model.metadata.tile_configs = tile_configs
    model.tile_boundary_nodes = tile_boundary_nodes
    return model


def _make_worker_with_peaks(peaks, vdd=1.0):
    """Create a TileWorker with peak data set up via init + update."""
    from distributed.tile_worker import TileWorker
    from distributed.tile_parsing import TileData

    # Create edges connecting all peak nodes to a shared boundary node
    all_nodes = set(peaks.keys()) | {'shared'}
    resistive_edges = [(node, 'shared', 1.0) for node in peaks]

    td = TileData(
        tile_id=(0, 0),
        resistive_edges=resistive_edges,
        all_nodes=all_nodes,
        boundary_nodes={'shared'},
        current_injections={},
        capacitive_edges=[],
    )
    worker = TileWorker()
    worker.setup_from_tile_data(td, interface_nodes={'shared'})
    worker.init_peak_tracking(vdd=vdd)
    if peaks:
        worker.update_peak_stats(peaks, t=0.0)
    return worker


# ──────────────────────────────────────────────────────────────────────
# Tests: TileWorker.prebin_peak_data
# ──────────────────────────────────────────────────────────────────────

class TestPrebinPeakData(unittest.TestCase):
    """Tests for TileWorker.prebin_peak_data() (time-domain heatmap path)."""

    def test_returns_empty_when_inactive(self):
        """Worker without init_peak_tracking returns empty dict."""
        from distributed.tile_worker import TileWorker
        from distributed.tile_parsing import TileData

        td = TileData(
            tile_id=(0, 0),
            resistive_edges=[('100_200_M1', 'port', 1.0)],
            all_nodes={'100_200_M1', 'port'},
            boundary_nodes={'port'},
            current_injections={},
            capacitive_edges=[],
        )
        worker = TileWorker()
        worker.setup_from_tile_data(td, interface_nodes={'port'})
        # Do NOT call init_peak_tracking
        spec = GlobalBinSpec(
            layers={'M1': LayerBinSpec('H', [(0.0, 400.0)],
                                       [np.linspace(0, 200, 3)])},
            nominal_voltage=1.0, aggregation='max',
        )
        result = worker.prebin_peak_data(spec)
        self.assertEqual(result, {})

    def test_converts_drops_to_correct_bins(self):
        """Peak drops are converted to mV IR-drop in the correct bins."""
        voltages = {'100_200_M1': 0.9, '300_200_M1': 0.8}  # drops 0.1, 0.2
        worker = _make_worker_with_peaks(voltages, vdd=1.0)

        spec = GlobalBinSpec(
            layers={'M1': LayerBinSpec(
                orientation='H',
                stripe_edges=[(150.0, 250.0)],
                bin_edges_list=[np.array([0.0, 200.0, 400.0])],  # 2 bins
            )},
            nominal_voltage=1.0, aggregation='max',
        )
        result = worker.prebin_peak_data(spec)
        self.assertIn('M1', result)
        s0 = result['M1'][0]
        self.assertIsNotNone(s0)
        # Node 100_200_M1: X=100 -> bin 0, drop=0.1 -> 100 mV
        self.assertAlmostEqual(s0[0], 100.0, places=1)
        # Node 300_200_M1: X=300 -> bin 1, drop=0.2 -> 200 mV
        self.assertAlmostEqual(s0[1], 200.0, places=1)

    def test_empty_peaks_returns_empty(self):
        """Worker with peak tracking active but no peaks returns empty."""
        worker = _make_worker_with_peaks({}, vdd=1.0)
        spec = GlobalBinSpec(
            layers={'M1': LayerBinSpec('H', [(0.0, 400.0)],
                                       [np.linspace(0, 200, 3)])},
            nominal_voltage=1.0, aggregation='max',
        )
        result = worker.prebin_peak_data(spec)
        self.assertEqual(result, {})


# ──────────────────────────────────────────────────────────────────────
# Tests: TileWorker.get_top_k_peaks
# ──────────────────────────────────────────────────────────────────────

class TestGetTopKPeaks(unittest.TestCase):
    """Tests for TileWorker.get_top_k_peaks()."""

    def test_returns_sorted_descending(self):
        """Top-K nodes should be sorted by drop descending."""
        # Drops: 100_200_M1 = 0.1V, 300_200_M1 = 0.3V, 500_200_M1 = 0.2V
        voltages = {'100_200_M1': 0.9, '300_200_M1': 0.7, '500_200_M1': 0.8}
        worker = _make_worker_with_peaks(voltages, vdd=1.0)
        top = worker.get_top_k_peaks(k=3)
        self.assertEqual(len(top), 3)
        # Worst first
        self.assertEqual(top[0][0], '300_200_M1')
        self.assertAlmostEqual(top[0][1], 0.3)
        self.assertEqual(top[1][0], '500_200_M1')
        self.assertAlmostEqual(top[1][1], 0.2)
        self.assertEqual(top[2][0], '100_200_M1')
        self.assertAlmostEqual(top[2][1], 0.1)

    def test_limits_results_to_k(self):
        """Should return at most k entries."""
        voltages = {'100_200_M1': 0.9, '300_200_M1': 0.7, '500_200_M1': 0.8}
        worker = _make_worker_with_peaks(voltages, vdd=1.0)
        top = worker.get_top_k_peaks(k=1)
        self.assertEqual(len(top), 1)
        self.assertEqual(top[0][0], '300_200_M1')

    def test_k_larger_than_nodes_returns_all(self):
        """K larger than node count returns all nodes."""
        voltages = {'100_200_M1': 0.9, '300_200_M1': 0.7}
        worker = _make_worker_with_peaks(voltages, vdd=1.0)
        top = worker.get_top_k_peaks(k=100)
        self.assertEqual(len(top), 2)

    def test_empty_peaks_returns_empty(self):
        """Worker with no peaks returns empty list."""
        from distributed.tile_worker import TileWorker
        worker = TileWorker()
        self.assertEqual(worker.get_top_k_peaks(k=10), [])

    def test_tuple_format(self):
        """Each entry is (node, drop, time)."""
        voltages = {'100_200_M1': 0.9}  # drop=0.1 at t=0
        worker = _make_worker_with_peaks(voltages, vdd=1.0)
        top = worker.get_top_k_peaks(k=1)
        node, drop, t = top[0]
        self.assertEqual(node, '100_200_M1')
        self.assertAlmostEqual(drop, 0.1)
        self.assertAlmostEqual(t, 0.0)


# ──────────────────────────────────────────────────────────────────────
# Tests: plot_distributed_td_heatmaps orchestrator
# ──────────────────────────────────────────────────────────────────────

class TestPlotDistributedTdHeatmaps(unittest.TestCase):
    """Smoke tests for plot_distributed_td_heatmaps orchestrator."""

    def test_smoke_creates_pngs(self):
        """TD heatmap pipeline with mock workers should produce PNGs."""
        meta = _make_single_layer_metadata(
            layer='M1', bbox=(0.0, 100.0, 0.0, 20.0),
            stripe_coords_h=[10.0], edge_orientation=(20, 2, 0),
        )
        prebin_data = {'M1': [np.array([10.0, 20.0, 30.0, 40.0, 50.0])]}

        tile_configs = [FakeTileConfig((0, 0)), FakeTileConfig((0, 1))]
        model = _make_mock_model(
            tile_configs=tile_configs,
            tile_boundary_nodes={(0, 0): [], (0, 1): []},
        )
        model.workers = [MagicMock(), MagicMock()]
        model.backend.call_all.side_effect = [
            [meta, meta],
            [prebin_data, prebin_data],
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            plot_distributed_td_heatmaps(
                model=model, nominal_voltage=1.0,
                output_dir=tmpdir, stripe_bin_size=5,
            )
            pngs = [f for f in os.listdir(tmpdir) if f.endswith('.png')]
            self.assertGreater(len(pngs), 0, "Expected at least one PNG")
            for png_name in pngs:
                self.assertGreater(
                    os.path.getsize(os.path.join(tmpdir, png_name)), 0,
                )

    def test_all_empty_prebins_produces_no_pngs(self):
        """All-empty prebins should warn and produce no PNGs."""
        meta = _make_single_layer_metadata(
            layer='M1', bbox=(0.0, 100.0, 0.0, 20.0),
            stripe_coords_h=[10.0], edge_orientation=(20, 2, 0),
        )
        tile_configs = [FakeTileConfig((0, 0))]
        model = _make_mock_model(
            tile_configs=tile_configs,
            tile_boundary_nodes={(0, 0): []},
        )
        model.workers = [MagicMock()]
        model.backend.call_all.side_effect = [
            [meta],
            [{}],   # empty prebin
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            plot_distributed_td_heatmaps(
                model=model, nominal_voltage=1.0,
                output_dir=tmpdir,
            )
            pngs = [f for f in os.listdir(tmpdir) if f.endswith('.png')]
            self.assertEqual(len(pngs), 0, "No PNGs expected for empty prebins")


if __name__ == '__main__':
    unittest.main()

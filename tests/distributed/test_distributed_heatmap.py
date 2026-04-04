"""Tests for distributed pre-binning heatmap pipeline.

Covers build_global_bin_spec, prebin_tile, merge_tile_prebins,
compute_boundary_ownership, plot_distributed_heatmaps, and edge cases.
"""

import os
import tempfile
import unittest
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock

import numpy as np

from distributed.heatmap import (
    GlobalBinSpec,
    LayerBinSpec,
    build_global_bin_spec,
    compute_boundary_ownership,
    merge_tile_prebins,
    plot_distributed_heatmaps,
    prebin_tile,
    _merge_stripe_coords,
    _compute_stripe_boundaries,
)

import pytest

pytestmark = pytest.mark.unit


# ──────────────────────────────────────────────────────────────────────
# Helpers for building test data
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


def _make_mock_model(
    tile_configs: List[Any],
    tile_boundary_nodes: Dict[Tuple[int, int], List[str]],
) -> MagicMock:
    """Build a mock DistributedPowerGridModel."""
    model = MagicMock()
    model.metadata.tile_configs = tile_configs
    model.tile_boundary_nodes = tile_boundary_nodes
    return model


@dataclass
class FakeTileConfig:
    tile_id: Tuple[int, int]


# ──────────────────────────────────────────────────────────────────────
# Tests: _merge_stripe_coords
# ──────────────────────────────────────────────────────────────────────

class TestMergeStripeCoords(unittest.TestCase):
    """Tests for the stripe coordinate merging helper."""

    def test_empty(self):
        self.assertEqual(_merge_stripe_coords([]), [])

    def test_single(self):
        self.assertEqual(_merge_stripe_coords([5.0]), [5.0])

    def test_no_merging_needed(self):
        result = _merge_stripe_coords([10.0, 20.0, 30.0])
        self.assertEqual(result, [10.0, 20.0, 30.0])

    def test_duplicates_merged(self):
        result = _merge_stripe_coords([10.0, 10.0, 20.0, 20.0, 30.0])
        self.assertEqual(result, [10.0, 20.0, 30.0])

    def test_near_duplicates_merged(self):
        """Coordinates within tolerance should be merged.

        Tolerance = median_spacing * 0.001. With spacing ~10 between
        real stripes, tol = 0.01. Offsets of 0.005 are within tolerance.
        Need the near-duplicates to be within tol of the previous merged
        representative, so the median spacing must dominate.
        """
        # Coords: 10.0, 20.0, 20.005, 30.0
        # Sorted unique diffs: [10.0, 0.005, 9.995], median = 9.995
        # tol = 9.995 * 0.001 ~ 0.01 > 0.005 -> 20.005 merges with 20.0
        result = _merge_stripe_coords([10.0, 20.0, 20.005, 30.0])
        self.assertEqual(len(result), 3)
        self.assertAlmostEqual(result[0], 10.0)
        self.assertAlmostEqual(result[1], 20.0)
        self.assertAlmostEqual(result[2], 30.0)


# ──────────────────────────────────────────────────────────────────────
# Tests: _compute_stripe_boundaries
# ──────────────────────────────────────────────────────────────────────

class TestComputeStripeBoundaries(unittest.TestCase):

    def test_empty(self):
        self.assertEqual(_compute_stripe_boundaries([]), [])

    def test_single_coord(self):
        edges = _compute_stripe_boundaries([50.0])
        self.assertEqual(len(edges), 1)
        self.assertAlmostEqual(edges[0][0], 49.0)
        self.assertAlmostEqual(edges[0][1], 51.0)

    def test_two_coords(self):
        edges = _compute_stripe_boundaries([10.0, 20.0])
        self.assertEqual(len(edges), 2)
        # First stripe: start = 10 - 5 = 5, end = 15
        self.assertAlmostEqual(edges[0][0], 5.0)
        self.assertAlmostEqual(edges[0][1], 15.0)
        # Second stripe: start = 15, end = 20 + 5 = 25
        self.assertAlmostEqual(edges[1][0], 15.0)
        self.assertAlmostEqual(edges[1][1], 25.0)

    def test_three_uniform_coords(self):
        edges = _compute_stripe_boundaries([10.0, 20.0, 30.0])
        self.assertEqual(len(edges), 3)
        # Boundaries are midpoints
        self.assertAlmostEqual(edges[0][1], 15.0)
        self.assertAlmostEqual(edges[1][0], 15.0)
        self.assertAlmostEqual(edges[1][1], 25.0)
        self.assertAlmostEqual(edges[2][0], 25.0)


# ──────────────────────────────────────────────────────────────────────
# Tests: build_global_bin_spec
# ──────────────────────────────────────────────────────────────────────

class TestBuildGlobalBinSpec(unittest.TestCase):
    """Tests for build_global_bin_spec."""

    def test_single_tile_single_layer(self):
        """Single tile, single layer should produce valid spec."""
        meta = _make_single_layer_metadata(
            layer='M1',
            bbox=(0.0, 100.0, 0.0, 200.0),
            edge_orientation=(50, 10, 5),  # predominantly H
            stripe_coords_h=[10.0, 20.0, 30.0],
        )
        spec = build_global_bin_spec(
            tile_metadatas=[meta],
            nominal_voltage=0.8,
            aggregation='max',
            stripe_bin_size=20,
        )
        self.assertIn('M1', spec.layers)
        layer = spec.layers['M1']
        self.assertEqual(layer.orientation, 'H')
        self.assertEqual(len(layer.stripe_edges), 3)
        self.assertEqual(len(layer.bin_edges_list), 3)
        # stripe_bin_size=20 (physical), x_range=100 -> int(100/20)=5 bins -> 6 edges
        for be in layer.bin_edges_list:
            self.assertEqual(len(be), 6)
        self.assertAlmostEqual(spec.nominal_voltage, 0.8)
        self.assertEqual(spec.aggregation, 'max')

    def test_multi_tile_bbox_union(self):
        """Two tiles with disjoint bboxes should produce union bbox."""
        meta1 = _make_single_layer_metadata(
            bbox=(0.0, 50.0, 0.0, 100.0),
            stripe_coords_h=[10.0, 20.0],
            edge_orientation=(30, 5, 0),
        )
        meta2 = _make_single_layer_metadata(
            bbox=(50.0, 100.0, 0.0, 100.0),
            stripe_coords_h=[10.0, 30.0],
            edge_orientation=(20, 5, 0),
        )
        spec = build_global_bin_spec(
            tile_metadatas=[meta1, meta2],
            nominal_voltage=1.0,
            stripe_bin_size=10,
        )
        layer = spec.layers['M1']
        # Bin edges should span full [0, 100] range along X (parallel for H)
        for be in layer.bin_edges_list:
            self.assertAlmostEqual(be[0], 0.0)
            self.assertAlmostEqual(be[-1], 100.0)

    def test_multi_layer(self):
        """Metadata with two layers should produce specs for both."""
        meta = {
            'M1': {
                'bbox': (0.0, 100.0, 0.0, 200.0),
                'n_nodes': 50,
                'stripe_coords_h': [10.0, 20.0],
                'stripe_coords_v': [5.0, 15.0],
                'edge_orientation': (40, 5, 0),  # H
            },
            'M2': {
                'bbox': (0.0, 100.0, 0.0, 200.0),
                'n_nodes': 50,
                'stripe_coords_h': [10.0, 20.0],
                'stripe_coords_v': [5.0, 15.0],
                'edge_orientation': (5, 40, 0),  # V
            },
        }
        spec = build_global_bin_spec(
            tile_metadatas=[meta],
            nominal_voltage=0.8,
        )
        self.assertEqual(spec.layers['M1'].orientation, 'H')
        self.assertEqual(spec.layers['M2'].orientation, 'V')

    def test_vertical_orientation_uses_v_coords(self):
        """V-oriented layer should use stripe_coords_v."""
        meta = _make_single_layer_metadata(
            edge_orientation=(5, 50, 0),  # predominantly V
            stripe_coords_v=[100.0, 200.0, 300.0],
        )
        spec = build_global_bin_spec(
            tile_metadatas=[meta],
            nominal_voltage=1.0,
            stripe_bin_size=10,
        )
        layer = spec.layers['M1']
        self.assertEqual(layer.orientation, 'V')
        self.assertEqual(len(layer.stripe_edges), 3)

    def test_stripe_consolidation(self):
        """Many stripe coords should be consolidated when exceeding max_stripes."""
        coords = list(range(100))
        meta = _make_single_layer_metadata(
            stripe_coords_h=[float(c) for c in coords],
            edge_orientation=(50, 10, 0),
        )
        spec = build_global_bin_spec(
            tile_metadatas=[meta],
            nominal_voltage=1.0,
            max_stripes=10,
        )
        layer = spec.layers['M1']
        self.assertLessEqual(len(layer.stripe_edges), 10)

    def test_default_stripe_bin_size(self):
        """Default stripe_bin_size (auto) should target ~500 bins."""
        meta = _make_single_layer_metadata(stripe_coords_h=[10.0])
        spec = build_global_bin_spec(
            tile_metadatas=[meta],
            nominal_voltage=1.0,
        )
        # auto -> 500 bins -> 501 edges
        self.assertEqual(len(spec.layers['M1'].bin_edges_list[0]), 501)

    def test_stripe_bin_size_physical(self):
        """stripe_bin_size as physical size should compute correct bin count."""
        # bbox x_range = 100.0, stripe_bin_size = 20 -> int(100/20) = 5 bins
        meta = _make_single_layer_metadata(stripe_coords_h=[10.0])
        spec = build_global_bin_spec(
            tile_metadatas=[meta],
            nominal_voltage=1.0,
            stripe_bin_size=20,
        )
        # 5 bins -> 6 edges
        self.assertEqual(len(spec.layers['M1'].bin_edges_list[0]), 6)


# ──────────────────────────────────────────────────────────────────────
# Tests: prebin_tile
# ──────────────────────────────────────────────────────────────────────

class TestPrebinTile(unittest.TestCase):
    """Tests for prebin_tile."""

    def _make_simple_bin_spec(self, orientation='H', n_bins=5):
        """Build a simple GlobalBinSpec for testing."""
        if orientation == 'H':
            # Stripes along Y, bins along X [0, 100]
            stripe_edges = [(5.0, 15.0), (15.0, 25.0)]
            bin_edges = np.linspace(0.0, 100.0, n_bins + 1)
        else:
            # Stripes along X, bins along Y [0, 200]
            stripe_edges = [(5.0, 15.0), (15.0, 25.0)]
            bin_edges = np.linspace(0.0, 200.0, n_bins + 1)

        return GlobalBinSpec(
            layers={
                'M1': LayerBinSpec(
                    orientation=orientation,
                    stripe_edges=stripe_edges,
                    bin_edges_list=[bin_edges.copy(), bin_edges.copy()],
                ),
            },
            nominal_voltage=1.0,
            aggregation='max',
        )

    def test_basic_max_binning(self):
        """Nodes at known positions should land in expected bins."""
        bin_spec = self._make_simple_bin_spec(orientation='H', n_bins=5)
        # Bins along X: [0, 20, 40, 60, 80, 100]
        # Stripe 0: Y in [5, 15], Stripe 1: Y in [15, 25]
        voltages = {
            '10_10_M1': 0.99,   # X=10 -> bin 0, Y=10 -> stripe 0
            '50_10_M1': 0.98,   # X=50 -> bin 2, Y=10 -> stripe 0
            '50_20_M1': 0.97,   # X=50 -> bin 2, Y=20 -> stripe 1
        }
        result = prebin_tile(voltages, bin_spec)
        self.assertIn('M1', result)
        self.assertEqual(len(result['M1']), 2)

        # Stripe 0 should have data
        s0 = result['M1'][0]
        self.assertIsNotNone(s0)
        self.assertEqual(len(s0), 5)

        # IR-drop = (1.0 - 0.99) * 1000 = 10.0 mV at bin 0
        self.assertAlmostEqual(s0[0], 10.0, places=2)
        # IR-drop = (1.0 - 0.98) * 1000 = 20.0 mV at bin 2
        self.assertAlmostEqual(s0[2], 20.0, places=2)

    def test_sum_aggregation(self):
        """Sum aggregation should add values in same bin."""
        bin_spec = GlobalBinSpec(
            layers={
                'M1': LayerBinSpec(
                    orientation='H',
                    stripe_edges=[(0.0, 100.0)],
                    bin_edges_list=[np.linspace(0.0, 100.0, 3)],  # 2 bins: [0,50,100]
                ),
            },
            nominal_voltage=1.0,
            aggregation='sum',
        )
        # Two nodes in the same bin
        voltages = {
            '10_50_M1': 0.5,   # bin 0, value = 0.5 (raw for sum)
            '20_50_M1': 0.3,   # bin 0, value = 0.3
        }
        result = prebin_tile(voltages, bin_spec)
        s0 = result['M1'][0]
        self.assertIsNotNone(s0)
        self.assertAlmostEqual(s0[0], 0.8, places=5)

    def test_max_takes_maximum(self):
        """Max aggregation should take the max of nodes in the same bin."""
        bin_spec = GlobalBinSpec(
            layers={
                'M1': LayerBinSpec(
                    orientation='H',
                    stripe_edges=[(0.0, 100.0)],
                    bin_edges_list=[np.linspace(0.0, 100.0, 3)],  # 2 bins
                ),
            },
            nominal_voltage=1.0,
            aggregation='max',
        )
        # Two nodes in same bin with different voltages
        voltages = {
            '10_50_M1': 0.99,   # IR-drop = 10 mV
            '20_50_M1': 0.97,   # IR-drop = 30 mV (worse)
        }
        result = prebin_tile(voltages, bin_spec)
        s0 = result['M1'][0]
        self.assertIsNotNone(s0)
        # Should take max IR-drop = 30 mV
        self.assertAlmostEqual(s0[0], 30.0, places=2)

    def test_empty_tile(self):
        """Tile with no parseable nodes should produce all-None stripes."""
        bin_spec = self._make_simple_bin_spec()
        voltages = {
            'VDD_vsrc': 1.0,
            '0': 0.0,
        }
        result = prebin_tile(voltages, bin_spec)
        self.assertIn('M1', result)
        for partial in result['M1']:
            self.assertIsNone(partial)

    def test_boundary_exclusion(self):
        """Excluded boundary nodes should be skipped."""
        bin_spec = GlobalBinSpec(
            layers={
                'M1': LayerBinSpec(
                    orientation='H',
                    stripe_edges=[(0.0, 100.0)],
                    bin_edges_list=[np.linspace(0.0, 100.0, 3)],
                ),
            },
            nominal_voltage=1.0,
            aggregation='sum',
        )
        voltages = {
            '10_50_M1': 0.5,   # should be excluded
            '60_50_M1': 0.3,   # should be counted -> bin 1
        }
        result = prebin_tile(
            voltages, bin_spec, boundary_exclude={'10_50_M1'}
        )
        s0 = result['M1'][0]
        self.assertIsNotNone(s0)
        self.assertTrue(np.isnan(s0[0]))                  # excluded -> NaN
        self.assertAlmostEqual(s0[1], 0.3, places=5)   # counted

    def test_sum_empty_bins_are_nan(self):
        """Sum aggregation: untouched bins should be NaN, not zero."""
        bin_spec = GlobalBinSpec(
            layers={
                'M1': LayerBinSpec(
                    orientation='H',
                    stripe_edges=[(0.0, 100.0)],
                    bin_edges_list=[np.linspace(0.0, 100.0, 6)],  # 5 bins
                ),
            },
            nominal_voltage=1.0,
            aggregation='sum',
        )
        # Single node in bin 2 only (x=50 -> bin 2 of [0,20,40,60,80,100])
        voltages = {'50_50_M1': 0.7}
        result = prebin_tile(voltages, bin_spec)
        s0 = result['M1'][0]
        self.assertIsNotNone(s0)
        # Only bin 2 should have data; others should be NaN
        self.assertTrue(np.isnan(s0[0]))
        self.assertTrue(np.isnan(s0[1]))
        self.assertAlmostEqual(s0[2], 0.7, places=5)
        self.assertTrue(np.isnan(s0[3]))
        self.assertTrue(np.isnan(s0[4]))

    def test_single_node_tile(self):
        """Single-node tile should produce data in exactly one bin."""
        bin_spec = self._make_simple_bin_spec(orientation='H', n_bins=5)
        voltages = {'50_10_M1': 0.95}  # stripe 0, bin 2
        result = prebin_tile(voltages, bin_spec)
        s0 = result['M1'][0]
        self.assertIsNotNone(s0)
        # Exactly one non-NaN bin
        non_nan = np.count_nonzero(~np.isnan(s0))
        self.assertEqual(non_nan, 1)
        # IR-drop = 50 mV
        self.assertAlmostEqual(np.nanmax(s0), 50.0, places=2)

    def test_nodes_outside_all_stripes(self):
        """Nodes outside all stripe ranges should produce None partials."""
        bin_spec = GlobalBinSpec(
            layers={
                'M1': LayerBinSpec(
                    orientation='H',
                    stripe_edges=[(100.0, 200.0)],  # Y range far away
                    bin_edges_list=[np.linspace(0.0, 100.0, 6)],
                ),
            },
            nominal_voltage=1.0,
            aggregation='max',
        )
        # Node at Y=10, which is outside [100, 200]
        voltages = {'50_10_M1': 0.95}
        result = prebin_tile(voltages, bin_spec)
        self.assertIsNone(result['M1'][0])


# ──────────────────────────────────────────────────────────────────────
# Tests: merge_tile_prebins
# ──────────────────────────────────────────────────────────────────────

class TestMergeTilePrebins(unittest.TestCase):
    """Tests for merge_tile_prebins."""

    def _make_spec(self, n_bins=5, aggregation='max'):
        return GlobalBinSpec(
            layers={
                'M1': LayerBinSpec(
                    orientation='H',
                    stripe_edges=[(0.0, 20.0)],
                    bin_edges_list=[np.linspace(0.0, 100.0, n_bins + 1)],
                ),
            },
            nominal_voltage=1.0,
            aggregation=aggregation,
        )

    def test_single_tile_passthrough(self):
        """Single tile prebin should pass through unchanged."""
        spec = self._make_spec(n_bins=3)
        prebin = {
            'M1': [np.array([1.0, 2.0, 3.0])],
        }
        merged = merge_tile_prebins([prebin], spec)
        np.testing.assert_array_almost_equal(merged['M1'][0][3], np.array([1.0, 2.0, 3.0]))

    def test_max_merge_two_tiles(self):
        """Max merge of two tiles should take element-wise max."""
        spec = self._make_spec(n_bins=3, aggregation='max')
        prebin1 = {'M1': [np.array([1.0, 5.0, np.nan])]}
        prebin2 = {'M1': [np.array([3.0, 2.0, 4.0])]}
        merged = merge_tile_prebins([prebin1, prebin2], spec)
        bin_values = merged['M1'][0][3]
        self.assertAlmostEqual(bin_values[0], 3.0)
        self.assertAlmostEqual(bin_values[1], 5.0)
        self.assertAlmostEqual(bin_values[2], 4.0)

    def test_sum_merge_two_tiles(self):
        """Sum merge of two tiles should add element-wise."""
        spec = self._make_spec(n_bins=3, aggregation='sum')
        prebin1 = {'M1': [np.array([1.0, 0.0, 3.0])]}
        prebin2 = {'M1': [np.array([2.0, 4.0, 0.0])]}
        merged = merge_tile_prebins([prebin1, prebin2], spec)
        bin_values = merged['M1'][0][3]
        np.testing.assert_array_almost_equal(bin_values, [3.0, 4.0, 3.0])

    def test_none_partials_ignored(self):
        """None partials from empty tiles should not affect merge."""
        spec = self._make_spec(n_bins=3, aggregation='max')
        prebin1 = {'M1': [np.array([10.0, 20.0, 30.0])]}
        prebin2 = {'M1': [None]}  # empty tile
        merged = merge_tile_prebins([prebin1, prebin2], spec)
        bin_values = merged['M1'][0][3]
        np.testing.assert_array_almost_equal(bin_values, [10.0, 20.0, 30.0])

    def test_all_none_returns_nan_for_max(self):
        """All-None stripe should produce NaN for max aggregation."""
        spec = self._make_spec(n_bins=3, aggregation='max')
        prebin1 = {'M1': [None]}
        prebin2 = {'M1': [None]}
        merged = merge_tile_prebins([prebin1, prebin2], spec)
        bin_values = merged['M1'][0][3]
        self.assertTrue(np.all(np.isnan(bin_values)))

    def test_all_none_returns_nan_for_sum(self):
        """All-None stripe should produce NaN for sum aggregation."""
        spec = self._make_spec(n_bins=3, aggregation='sum')
        prebin1 = {'M1': [None]}
        prebin2 = {'M1': [None]}
        merged = merge_tile_prebins([prebin1, prebin2], spec)
        bin_values = merged['M1'][0][3]
        self.assertTrue(np.all(np.isnan(bin_values)))

    def test_sum_merge_preserves_nan_for_all_nan_bins(self):
        """Sum merge: bins where ALL tiles have NaN should remain NaN."""
        spec = self._make_spec(n_bins=3, aggregation='sum')
        # Tile 1: bin 0 has data, bins 1-2 are NaN (no nodes)
        prebin1 = {'M1': [np.array([1.0, np.nan, np.nan])]}
        # Tile 2: bins 0-1 have data, bin 2 is NaN
        prebin2 = {'M1': [np.array([2.0, 4.0, np.nan])]}
        merged = merge_tile_prebins([prebin1, prebin2], spec)
        bin_values = merged['M1'][0][3]
        self.assertAlmostEqual(bin_values[0], 3.0)   # 1.0 + 2.0
        self.assertAlmostEqual(bin_values[1], 4.0)   # NaN + 4.0
        self.assertTrue(np.isnan(bin_values[2]))       # NaN + NaN -> NaN


# ──────────────────────────────────────────────────────────────────────
# Tests: compute_boundary_ownership
# ──────────────────────────────────────────────────────────────────────

class TestComputeBoundaryOwnership(unittest.TestCase):
    """Tests for compute_boundary_ownership."""

    def test_no_shared_boundaries(self):
        """Tiles with disjoint boundary nodes should have empty exclusion sets."""
        model = _make_mock_model(
            tile_configs=[FakeTileConfig((0, 0)), FakeTileConfig((0, 1))],
            tile_boundary_nodes={
                (0, 0): ['a', 'b'],
                (0, 1): ['c', 'd'],
            },
        )
        exclusions = compute_boundary_ownership(model)
        self.assertEqual(exclusions[0], set())
        self.assertEqual(exclusions[1], set())

    def test_shared_boundary_owned_by_smallest_idx(self):
        """Shared boundary node should be excluded from higher-index tile."""
        model = _make_mock_model(
            tile_configs=[FakeTileConfig((0, 0)), FakeTileConfig((0, 1))],
            tile_boundary_nodes={
                (0, 0): ['shared', 'a'],
                (0, 1): ['shared', 'b'],
            },
        )
        exclusions = compute_boundary_ownership(model)
        # Tile 0 owns 'shared', so tile 1 should exclude it
        self.assertNotIn('shared', exclusions[0])
        self.assertIn('shared', exclusions[1])

    def test_three_tiles_shared_node(self):
        """Node shared by 3 tiles: only tile with smallest idx keeps it."""
        model = _make_mock_model(
            tile_configs=[
                FakeTileConfig((0, 0)),
                FakeTileConfig((0, 1)),
                FakeTileConfig((1, 0)),
            ],
            tile_boundary_nodes={
                (0, 0): ['n1'],
                (0, 1): ['n1'],
                (1, 0): ['n1'],
            },
        )
        exclusions = compute_boundary_ownership(model)
        self.assertNotIn('n1', exclusions[0])  # owner
        self.assertIn('n1', exclusions[1])
        self.assertIn('n1', exclusions[2])


# ──────────────────────────────────────────────────────────────────────
# Integration test: end-to-end prebinning pipeline
# ──────────────────────────────────────────────────────────────────────

class TestEndToEndPrebinning(unittest.TestCase):
    """Integration test: metadata -> bin_spec -> prebin -> merge."""

    def test_two_tile_horizontal_merge(self):
        """Two tiles with H-oriented layer should merge correctly."""
        # Tile 1 covers X=[0,50], Tile 2 covers X=[50,100]
        # Both have a stripe at Y=10
        meta1 = _make_single_layer_metadata(
            bbox=(0.0, 50.0, 0.0, 20.0),
            stripe_coords_h=[10.0],
            edge_orientation=(20, 2, 0),
        )
        meta2 = _make_single_layer_metadata(
            bbox=(50.0, 100.0, 0.0, 20.0),
            stripe_coords_h=[10.0],
            edge_orientation=(20, 2, 0),
        )

        spec = build_global_bin_spec(
            tile_metadatas=[meta1, meta2],
            nominal_voltage=1.0,
            aggregation='max',
            stripe_bin_size=5,
        )

        # Tile 1 nodes: X in [0,50], Y=10
        voltages1 = {
            '10_10_M1': 0.99,   # IR-drop = 10 mV
            '30_10_M1': 0.98,   # IR-drop = 20 mV
        }
        # Tile 2 nodes: X in [50,100], Y=10
        voltages2 = {
            '70_10_M1': 0.97,   # IR-drop = 30 mV
            '90_10_M1': 0.96,   # IR-drop = 40 mV
        }

        pb1 = prebin_tile(voltages1, spec)
        pb2 = prebin_tile(voltages2, spec)

        merged = merge_tile_prebins([pb1, pb2], spec)

        self.assertIn('M1', merged)
        self.assertEqual(len(merged['M1']), 1)  # 1 stripe

        # The merged bin_values should have the max IR-drop from either tile
        s_start, s_end, bin_edges, bin_values = merged['M1'][0]
        # At least some bins should have non-NaN values
        non_nan = bin_values[~np.isnan(bin_values)]
        self.assertGreater(len(non_nan), 0)
        # Max IR-drop should be 40 mV
        self.assertAlmostEqual(np.nanmax(bin_values), 40.0, places=1)

    def test_vertical_layer_binning(self):
        """V-oriented layer should bin along Y axis correctly."""
        meta = _make_single_layer_metadata(
            bbox=(0.0, 20.0, 0.0, 100.0),
            stripe_coords_v=[10.0],
            edge_orientation=(2, 20, 0),  # V
        )
        spec = build_global_bin_spec(
            tile_metadatas=[meta],
            nominal_voltage=1.0,
            aggregation='max',
            stripe_bin_size=5,
        )
        self.assertEqual(spec.layers['M1'].orientation, 'V')

        voltages = {
            '10_20_M1': 0.98,   # Y=20 -> some bin along Y
            '10_80_M1': 0.95,   # Y=80 -> different bin along Y
        }
        pb = prebin_tile(voltages, spec)
        self.assertIn('M1', pb)

        merged = merge_tile_prebins([pb], spec)
        _, _, bin_edges, bin_values = merged['M1'][0]
        # Bin edges should span Y range [0, 100]
        self.assertAlmostEqual(bin_edges[0], 0.0)
        self.assertAlmostEqual(bin_edges[-1], 100.0)
        # Max IR-drop = (1.0 - 0.95) * 1000 = 50 mV
        self.assertAlmostEqual(np.nanmax(bin_values), 50.0, places=1)


# ──────────────────────────────────────────────────────────────────────
# Tests: aggregation parameter validation (M6)
# ──────────────────────────────────────────────────────────────────────

class TestAggregationValidation(unittest.TestCase):
    """Tests for aggregation parameter validation."""

    def test_invalid_aggregation_in_global_bin_spec(self):
        """GlobalBinSpec should reject invalid aggregation values."""
        with self.assertRaises(ValueError):
            GlobalBinSpec(layers={}, nominal_voltage=1.0, aggregation='mean')

    def test_invalid_aggregation_in_build_global_bin_spec(self):
        """build_global_bin_spec should reject invalid aggregation values."""
        meta = _make_single_layer_metadata()
        with self.assertRaises(ValueError):
            build_global_bin_spec(
                tile_metadatas=[meta],
                nominal_voltage=1.0,
                aggregation='avg',
            )

    def test_valid_aggregation_max(self):
        """'max' should be accepted without error."""
        spec = GlobalBinSpec(layers={}, nominal_voltage=1.0, aggregation='max')
        self.assertEqual(spec.aggregation, 'max')

    def test_valid_aggregation_sum(self):
        """'sum' should be accepted without error."""
        spec = GlobalBinSpec(layers={}, nominal_voltage=1.0, aggregation='sum')
        self.assertEqual(spec.aggregation, 'sum')


# ──────────────────────────────────────────────────────────────────────
# Tests: out-of-range node filtering in prebin_tile (H1)
# ──────────────────────────────────────────────────────────────────────

class TestPrebinTileOutOfRange(unittest.TestCase):
    """Tests verifying out-of-range nodes are dropped, not clamped."""

    def test_out_of_range_nodes_not_clamped_to_edge_bins(self):
        """Nodes outside bin range should not corrupt edge bin values.

        Previously, np.clip forced out-of-range indices into edge bins.
        Now they are filtered out entirely.
        """
        # Bin edges [10, 30, 50]: 2 bins, covering X in [10, 50]
        bin_spec = GlobalBinSpec(
            layers={
                'M1': LayerBinSpec(
                    orientation='H',
                    stripe_edges=[(0.0, 100.0)],
                    bin_edges_list=[np.array([10.0, 30.0, 50.0])],
                ),
            },
            nominal_voltage=1.0,
            aggregation='max',
        )
        # Node at X=5 is BELOW bin range, node at X=60 is ABOVE bin range.
        # Node at X=20 is in bin 0.  The out-of-range nodes should NOT
        # appear in bin 0 or bin 1.
        voltages = {
            '5_50_M1': 0.90,    # X=5 -> out of range (below)
            '20_50_M1': 0.95,   # X=20 -> bin 0, IR-drop = 50 mV
            '60_50_M1': 0.80,   # X=60 -> out of range (above)
        }
        result = prebin_tile(voltages, bin_spec)
        s0 = result['M1'][0]
        self.assertIsNotNone(s0)
        # Bin 0 should have ONLY the X=20 node (IR-drop = 50 mV)
        self.assertAlmostEqual(s0[0], 50.0, places=2)
        # Bin 1 should be empty (NaN), not corrupted by X=60 node
        self.assertTrue(np.isnan(s0[1]))

    def test_all_out_of_range_produces_none(self):
        """If all nodes are out of range, the stripe partial should be None."""
        bin_spec = GlobalBinSpec(
            layers={
                'M1': LayerBinSpec(
                    orientation='H',
                    stripe_edges=[(0.0, 100.0)],
                    bin_edges_list=[np.array([10.0, 30.0, 50.0])],
                ),
            },
            nominal_voltage=1.0,
            aggregation='max',
        )
        voltages = {
            '5_50_M1': 0.90,    # below range
            '60_50_M1': 0.80,   # above range
        }
        result = prebin_tile(voltages, bin_spec)
        # All nodes out of range -> stripe partial is None
        self.assertIsNone(result['M1'][0])


# ──────────────────────────────────────────────────────────────────────
# Smoke test: plot_distributed_heatmaps (H3)
# ──────────────────────────────────────────────────────────────────────

class TestPlotDistributedHeatmaps(unittest.TestCase):
    """Smoke test for the full plot_distributed_heatmaps orchestrator."""

    def test_smoke_two_tiles_creates_png(self):
        """A minimal 2-tile setup should produce at least one PNG."""
        # Build tile metadata (returned by get_layer_metadata)
        meta_tile0 = _make_single_layer_metadata(
            layer='M1',
            bbox=(0.0, 50.0, 0.0, 20.0),
            stripe_coords_h=[10.0],
            edge_orientation=(20, 2, 0),
        )
        meta_tile1 = _make_single_layer_metadata(
            layer='M1',
            bbox=(50.0, 100.0, 0.0, 20.0),
            stripe_coords_h=[10.0],
            edge_orientation=(20, 2, 0),
        )

        # Build tile voltage dicts (returned from result.tile_results)
        voltages_tile0 = {
            '10_10_M1': 0.99,
            '30_10_M1': 0.98,
        }
        voltages_tile1 = {
            '70_10_M1': 0.97,
            '90_10_M1': 0.96,
        }

        # Mock tile result objects
        tile_result0 = MagicMock()
        tile_result0.voltages = voltages_tile0
        tile_result1 = MagicMock()
        tile_result1.voltages = voltages_tile1

        # Mock the DistributedSolveResult
        result = MagicMock()
        result.nominal_voltage = 1.0
        result.net_name = 'VDD'
        result.tile_results = {
            (0, 0): tile_result0,
            (0, 1): tile_result1,
        }

        # Mock the DistributedPowerGridModel
        tile_configs = [FakeTileConfig((0, 0)), FakeTileConfig((0, 1))]
        model = _make_mock_model(
            tile_configs=tile_configs,
            tile_boundary_nodes={(0, 0): [], (0, 1): []},
        )

        # Wire backend.call_all to return per-tile metadata
        model.backend.call_all.return_value = [meta_tile0, meta_tile1]

        # Wire backend.map_func to actually call prebin_tile
        def fake_map_func(func, args_list):
            return [func(*args) for args in args_list]
        model.backend.map_func.side_effect = fake_map_func

        with tempfile.TemporaryDirectory() as tmpdir:
            plot_distributed_heatmaps(
                result=result,
                model=model,
                output_dir=tmpdir,
                stripe_bin_size=5,
            )

            # Verify at least one PNG was created
            pngs = [f for f in os.listdir(tmpdir) if f.endswith('.png')]
            self.assertGreater(len(pngs), 0, "Expected at least one PNG file")

            # Verify the file has nonzero size
            for png_name in pngs:
                png_path = os.path.join(tmpdir, png_name)
                self.assertGreater(
                    os.path.getsize(png_path), 0,
                    f"PNG {png_name} should have nonzero size",
                )


if __name__ == '__main__':
    unittest.main()

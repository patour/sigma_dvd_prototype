#!/usr/bin/env python3
"""Unit tests for stripe_heatmap module."""

import os
import tempfile
import unittest

import numpy as np

from visualization.stripe_heatmap import (
    parse_node_info,
    extract_node_data_vectorized,
    detect_orientation_from_coords,
    group_nodes_into_stripes,
    consolidate_stripes,
    aggregate_stripe_bins,
    aggregate_fast_imshow,
    plot_stripe_heatmap,
    render_from_prebinned_stripe_data,
)

import pytest

pytestmark = pytest.mark.unit


class TestNodeParsing(unittest.TestCase):
    """Tests for node name parsing utilities."""

    def test_parse_node_info_valid(self):
        """Test parsing valid node names."""
        x, y, layer = parse_node_info("1000_2000_M1")
        self.assertEqual(x, 1000.0)
        self.assertEqual(y, 2000.0)
        self.assertEqual(layer, "M1")

    def test_parse_node_info_float_coords(self):
        """Test parsing node names with float coordinates."""
        x, y, layer = parse_node_info("1000.5_2000.75_M2")
        self.assertEqual(x, 1000.5)
        self.assertEqual(y, 2000.75)
        self.assertEqual(layer, "M2")

    def test_parse_node_info_invalid(self):
        """Test parsing invalid node names."""
        x, y, layer = parse_node_info("VDD_vsrc")
        self.assertIsNone(x)
        self.assertIsNone(y)
        self.assertIsNone(layer)

    def test_parse_node_info_short(self):
        """Test parsing short node names."""
        x, y, layer = parse_node_info("100_200")
        self.assertIsNone(x)
        self.assertIsNone(y)
        self.assertIsNone(layer)


class TestDataExtraction(unittest.TestCase):
    """Tests for vectorized data extraction."""

    def test_extract_node_data_vectorized(self):
        """Test extracting node data from dict."""
        node_values = {
            "1000_2000_M1": 0.01,
            "1500_2500_M1": 0.02,
            "VDD_vsrc": 1.0,  # Invalid - should be skipped
            "2000_3000_M2": 0.015,
        }

        nodes, xs, ys, layers, values = extract_node_data_vectorized(node_values)

        self.assertEqual(len(nodes), 3)
        self.assertEqual(len(xs), 3)
        self.assertEqual(len(ys), 3)
        self.assertEqual(len(layers), 3)
        self.assertEqual(len(values), 3)

        # Check values are preserved
        self.assertIn(0.01, values)
        self.assertIn(0.02, values)
        self.assertIn(0.015, values)


class TestOrientationDetection(unittest.TestCase):
    """Tests for orientation detection."""

    def test_detect_horizontal(self):
        """Test detecting horizontal routing."""
        # Many X positions, few Y positions -> horizontal
        xs = np.array([100, 200, 300, 400, 500, 600, 700, 800])
        ys = np.array([100, 100, 100, 100, 100, 100, 100, 100])
        orientation = detect_orientation_from_coords(xs, ys)
        self.assertEqual(orientation, "H")

    def test_detect_vertical(self):
        """Test detecting vertical routing."""
        # Few X positions, many Y positions -> vertical
        xs = np.array([100, 100, 100, 100, 100, 100, 100, 100])
        ys = np.array([100, 200, 300, 400, 500, 600, 700, 800])
        orientation = detect_orientation_from_coords(xs, ys)
        self.assertEqual(orientation, "V")

    def test_detect_mixed(self):
        """Test detecting mixed routing."""
        # Similar variance in both directions -> mixed
        xs = np.array([100, 200, 300, 400])
        ys = np.array([100, 200, 300, 400])
        orientation = detect_orientation_from_coords(xs, ys)
        self.assertEqual(orientation, "MIXED")

    def test_single_point(self):
        """Test single point returns MIXED."""
        xs = np.array([100])
        ys = np.array([200])
        orientation = detect_orientation_from_coords(xs, ys)
        self.assertEqual(orientation, "MIXED")

    def test_moderate_aspect_ratio_horizontal(self):
        """Test moderate aspect ratio (2x-3x) classifies as horizontal.

        With the lowered threshold (2.0 from 3.0), layers with moderate
        aspect ratios should now be correctly classified instead of MIXED.
        """
        # 5 unique X values, 2 unique Y values -> ~2.5x ratio
        xs = np.array([100, 200, 300, 400, 500, 100, 200, 300, 400, 500])
        ys = np.array([100, 100, 100, 100, 100, 200, 200, 200, 200, 200])
        orientation = detect_orientation_from_coords(xs, ys)
        self.assertEqual(orientation, "H")

    def test_moderate_aspect_ratio_vertical(self):
        """Test moderate aspect ratio (2x-3x) classifies as vertical."""
        # 2 unique X values, 5 unique Y values -> ~2.5x ratio
        xs = np.array([100, 100, 100, 100, 100, 200, 200, 200, 200, 200])
        ys = np.array([100, 200, 300, 400, 500, 100, 200, 300, 400, 500])
        orientation = detect_orientation_from_coords(xs, ys)
        self.assertEqual(orientation, "V")

    def test_custom_threshold_higher(self):
        """Test with higher threshold (3.0) - should return MIXED for 2.5x ratio."""
        # 5 unique X values, 2 unique Y values -> ~2.5x ratio
        xs = np.array([100, 200, 300, 400, 500, 100, 200, 300, 400, 500])
        ys = np.array([100, 100, 100, 100, 100, 200, 200, 200, 200, 200])
        # With threshold=3.0, 2.5x ratio should be MIXED
        orientation = detect_orientation_from_coords(xs, ys, threshold=3.0)
        self.assertEqual(orientation, "MIXED")

    def test_custom_threshold_lower(self):
        """Test with lower threshold (1.5) - should return H for 1.8x ratio."""
        # 9 unique X values, 5 unique Y values -> ~1.8x ratio
        xs = np.array([100, 200, 300, 400, 500, 600, 700, 800, 900] * 5)
        ys = np.array([100] * 9 + [200] * 9 + [300] * 9 + [400] * 9 + [500] * 9)
        # With threshold=1.5, 1.8x ratio should be H
        orientation = detect_orientation_from_coords(xs, ys, threshold=1.5)
        self.assertEqual(orientation, "H")

    def test_range_fallback_horizontal(self):
        """Test range ratio fallback for horizontal case."""
        # Same unique counts but very wide X range
        xs = np.array([0, 10000])  # Wide X range
        ys = np.array([0, 100])    # Narrow Y range
        # Unique counts are equal (2 each), but range ratio is 100:1
        # Should fall back to range-based detection
        orientation = detect_orientation_from_coords(xs, ys)
        self.assertEqual(orientation, "H")

    def test_range_fallback_vertical(self):
        """Test range ratio fallback for vertical case."""
        # Same unique counts but very tall Y range
        xs = np.array([0, 100])    # Narrow X range
        ys = np.array([0, 10000])  # Wide Y range
        orientation = detect_orientation_from_coords(xs, ys)
        self.assertEqual(orientation, "V")


class TestStripeGrouping(unittest.TestCase):
    """Tests for stripe grouping."""

    def test_group_horizontal(self):
        """Test grouping into horizontal stripes."""
        xs = np.array([100, 200, 300, 100, 200, 300])
        ys = np.array([100, 100, 100, 200, 200, 200])
        values = np.array([1, 2, 3, 4, 5, 6])

        stripes = group_nodes_into_stripes(xs, ys, values, "H")

        self.assertEqual(len(stripes), 2)  # Two Y values: 100 and 200
        self.assertIn(100, stripes)
        self.assertIn(200, stripes)

        # Each stripe has 3 points
        self.assertEqual(len(stripes[100][0]), 3)
        self.assertEqual(len(stripes[200][0]), 3)

    def test_group_vertical(self):
        """Test grouping into vertical stripes."""
        xs = np.array([100, 100, 100, 200, 200, 200])
        ys = np.array([100, 200, 300, 100, 200, 300])
        values = np.array([1, 2, 3, 4, 5, 6])

        stripes = group_nodes_into_stripes(xs, ys, values, "V")

        self.assertEqual(len(stripes), 2)  # Two X values: 100 and 200
        self.assertIn(100, stripes)
        self.assertIn(200, stripes)


class TestStripeConsolidation(unittest.TestCase):
    """Tests for stripe consolidation."""

    def test_no_consolidation_needed(self):
        """Test when stripe count is below threshold."""
        stripes = {
            100: (np.array([100, 200]), np.array([100, 100]), np.array([1, 2])),
            200: (np.array([100, 200]), np.array([200, 200]), np.array([3, 4])),
        }

        consolidated = consolidate_stripes(stripes, max_stripes=10)

        self.assertEqual(len(consolidated), 2)

    def test_consolidation_needed(self):
        """Test when stripe count exceeds threshold."""
        # Create 10 stripes
        stripes = {}
        for i in range(10):
            y = i * 100
            stripes[y] = (
                np.array([100, 200]),
                np.array([y, y]),
                np.array([i, i + 1]),
            )

        consolidated = consolidate_stripes(stripes, max_stripes=3)

        # Should consolidate to ~3 groups
        self.assertLessEqual(len(consolidated), 4)


class TestBinAggregation(unittest.TestCase):
    """Tests for stripe bin aggregation."""

    def test_aggregate_max(self):
        """Test max aggregation within stripe."""
        xs = np.array([100, 150, 200, 250, 300])
        ys = np.array([100, 100, 100, 100, 100])
        values = np.array([1, 5, 2, 8, 3])

        bins, bin_values = aggregate_stripe_bins(
            xs, ys, values, "H", bin_size=100, aggregation="max"
        )

        # Should have bins covering 100-300
        self.assertTrue(len(bins) > 1)

        # Max value in the data should appear in bin_values
        valid_values = bin_values[~np.isnan(bin_values)]
        self.assertGreater(len(valid_values), 0)

    def test_aggregate_sum(self):
        """Test sum aggregation within stripe."""
        # Use coordinates that fall clearly within bins
        xs = np.array([110, 150, 210, 250, 310])
        ys = np.array([100, 100, 100, 100, 100])
        values = np.array([1, 1, 1, 1, 1])

        bins, bin_values = aggregate_stripe_bins(
            xs, ys, values, "H", bin_size=100, aggregation="sum"
        )

        # Sum of all values should be 5
        total = sum(v for v in bin_values if v > 0)
        self.assertEqual(total, 5)

    def test_target_bins_parameter(self):
        """Test that target_bins parameter affects bin size for large stripes."""
        # Create a large stripe (>1000 nodes)
        n_nodes = 2000
        xs = np.linspace(0, 10000, n_nodes)
        ys = np.full(n_nodes, 100)
        values = np.random.uniform(0, 1, n_nodes)

        # With target_bins=100, should get ~100 bins
        bins_100, _ = aggregate_stripe_bins(
            xs, ys, values, "H", bin_size=None, aggregation="max", target_bins=100
        )

        # With target_bins=500, should get ~500 bins
        bins_500, _ = aggregate_stripe_bins(
            xs, ys, values, "H", bin_size=None, aggregation="max", target_bins=500
        )

        # More target_bins should result in more actual bins
        self.assertGreater(len(bins_500), len(bins_100))

    def test_target_bins_small_stripe(self):
        """Test that target_bins doesn't affect small stripes (<1000 nodes)."""
        # Create a small stripe
        n_nodes = 100
        xs = np.linspace(0, 1000, n_nodes)
        ys = np.full(n_nodes, 100)
        values = np.random.uniform(0, 1, n_nodes)

        # For small stripes, target_bins should be ignored (uses sqrt formula)
        bins_100, _ = aggregate_stripe_bins(
            xs, ys, values, "H", bin_size=None, aggregation="max", target_bins=100
        )

        bins_500, _ = aggregate_stripe_bins(
            xs, ys, values, "H", bin_size=None, aggregation="max", target_bins=500
        )

        # For small stripes, bin count should be similar regardless of target_bins
        # (within a factor of 2 since both use sqrt formula)
        self.assertEqual(len(bins_100), len(bins_500))


class TestPlotting(unittest.TestCase):
    """Tests for heatmap plotting."""

    def test_plot_stripe_heatmap_basic(self):
        """Test basic heatmap generation."""
        # Create test data
        node_values = {}
        for x in range(0, 1000, 100):
            for y in range(0, 100, 10):  # Horizontal layer
                node_values[f"{x}_{y}_M1"] = np.random.uniform(0.001, 0.01)

        with tempfile.TemporaryDirectory() as tmpdir:
            plot_stripe_heatmap(
                node_values=node_values,
                layers=["M1"],
                plot_dir=tmpdir,
                show=False,
                verbose=False,
            )

            # Check output file exists
            expected_file = os.path.join(tmpdir, "peak_ir-drop_heatmap_M1.png")
            self.assertTrue(os.path.exists(expected_file))

    def test_plot_with_markers_and_windows(self):
        """Test heatmap with markers and windows."""
        # Create test data
        node_values = {}
        for x in range(0, 500, 50):
            for y in range(0, 50, 10):
                node_values[f"{x}_{y}_M1"] = 0.005

        markers = [(250, 25, "M1")]
        windows = [(200, 300, 0, 50, "M1")]

        with tempfile.TemporaryDirectory() as tmpdir:
            plot_stripe_heatmap(
                node_values=node_values,
                layers=["M1"],
                plot_dir=tmpdir,
                markers=markers,
                windows=windows,
                show=False,
                verbose=False,
            )

            expected_file = os.path.join(tmpdir, "peak_ir-drop_heatmap_M1.png")
            self.assertTrue(os.path.exists(expected_file))

    def test_plot_empty_data(self):
        """Test handling of empty data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Should not raise an exception
            plot_stripe_heatmap(
                node_values={},
                plot_dir=tmpdir,
                show=False,
                verbose=False,
            )

    def test_plot_imshow_mode_forced(self):
        """Test heatmap generation using imshow mode (low threshold)."""
        # Create test data
        node_values = {}
        for x in range(0, 500, 50):
            for y in range(0, 100, 10):
                node_values[f"{x}_{y}_M1"] = np.random.uniform(0.001, 0.01)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Force imshow mode by setting threshold to 0
            plot_stripe_heatmap(
                node_values=node_values,
                layers=["M1"],
                plot_dir=tmpdir,
                show=False,
                verbose=False,
                use_imshow_threshold=0,  # Force imshow mode
            )

            expected_file = os.path.join(tmpdir, "peak_ir-drop_heatmap_M1.png")
            self.assertTrue(os.path.exists(expected_file))

    def test_plot_mixed_orientation_uses_imshow(self):
        """Test that MIXED orientation falls back to imshow."""
        # Create data with equal X and Y variance (MIXED orientation)
        node_values = {}
        for x in range(0, 500, 50):
            for y in range(0, 500, 50):
                node_values[f"{x}_{y}_M1"] = np.random.uniform(0.001, 0.01)

        with tempfile.TemporaryDirectory() as tmpdir:
            plot_stripe_heatmap(
                node_values=node_values,
                layers=["M1"],
                plot_dir=tmpdir,
                show=False,
                verbose=False,
            )

            expected_file = os.path.join(tmpdir, "peak_ir-drop_heatmap_M1.png")
            self.assertTrue(os.path.exists(expected_file))

    def test_plot_with_custom_orientation_threshold(self):
        """Test heatmap with custom orientation_threshold parameter."""
        # Create data with moderate aspect ratio (~2.5x)
        node_values = {}
        for x in range(0, 500, 100):  # 5 unique X
            for y in range(0, 200, 100):  # 2 unique Y
                node_values[f"{x}_{y}_M1"] = np.random.uniform(0.001, 0.01)

        with tempfile.TemporaryDirectory() as tmpdir:
            # With threshold=2.0, should detect as horizontal
            plot_stripe_heatmap(
                node_values=node_values,
                layers=["M1"],
                plot_dir=tmpdir,
                orientation_threshold=2.0,
                show=False,
                verbose=False,
            )

            expected_file = os.path.join(tmpdir, "peak_ir-drop_heatmap_M1.png")
            self.assertTrue(os.path.exists(expected_file))

    def test_plot_with_custom_target_bins(self):
        """Test heatmap with custom target_bins_per_stripe parameter."""
        # Create horizontal layer data
        node_values = {}
        for x in range(0, 1000, 10):  # 100 unique X
            for y in range(0, 100, 50):  # 2 unique Y
                node_values[f"{x}_{y}_M1"] = np.random.uniform(0.001, 0.01)

        with tempfile.TemporaryDirectory() as tmpdir:
            plot_stripe_heatmap(
                node_values=node_values,
                layers=["M1"],
                plot_dir=tmpdir,
                target_bins_per_stripe=100,
                show=False,
                verbose=False,
            )

            expected_file = os.path.join(tmpdir, "peak_ir-drop_heatmap_M1.png")
            self.assertTrue(os.path.exists(expected_file))

    def test_plot_with_high_max_stripes(self):
        """Test heatmap with high max_stripes (new default 500)."""
        # Create horizontal layer with many stripes
        node_values = {}
        for x in range(0, 1000, 10):
            for y in range(0, 1000, 10):  # 100 unique Y values
                node_values[f"{x}_{y}_M1"] = np.random.uniform(0.001, 0.01)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Use max_stripes=500 (new default)
            plot_stripe_heatmap(
                node_values=node_values,
                layers=["M1"],
                plot_dir=tmpdir,
                max_stripes=500,
                show=False,
                verbose=False,
            )

            expected_file = os.path.join(tmpdir, "peak_ir-drop_heatmap_M1.png")
            self.assertTrue(os.path.exists(expected_file))


class TestFastImshow(unittest.TestCase):
    """Tests for fast imshow aggregation."""

    def test_aggregate_fast_imshow_basic(self):
        """Test basic imshow aggregation."""
        xs = np.array([100, 200, 300, 400, 500])
        ys = np.array([100, 200, 300, 400, 500])
        values = np.array([1, 2, 3, 4, 5])

        x_edges, y_edges, grid = aggregate_fast_imshow(
            xs, ys, values, "MIXED", max_bins=10, aggregation="max"
        )

        # Check output shapes
        self.assertEqual(len(x_edges), grid.shape[1] + 1)
        self.assertEqual(len(y_edges), grid.shape[0] + 1)

        # Check that grid has some non-NaN values
        valid_count = np.sum(~np.isnan(grid))
        self.assertGreater(valid_count, 0)

    def test_aggregate_fast_imshow_horizontal(self):
        """Test imshow aggregation with horizontal orientation."""
        # Create horizontal data (many X, few Y)
        xs = np.array([100, 200, 300, 400, 500, 600, 700, 800])
        ys = np.array([100, 100, 100, 100, 100, 100, 100, 100])
        values = np.array([1, 2, 3, 4, 5, 6, 7, 8])

        x_edges, y_edges, grid = aggregate_fast_imshow(
            xs, ys, values, "H", max_bins=100, aggregation="max"
        )

        # For horizontal, should have more X bins than Y bins
        n_x_bins = len(x_edges) - 1
        n_y_bins = len(y_edges) - 1
        self.assertGreaterEqual(n_x_bins, n_y_bins)

    def test_aggregate_fast_imshow_vertical(self):
        """Test imshow aggregation with vertical orientation."""
        # Create vertical data (few X, many Y)
        xs = np.array([100, 100, 100, 100, 100, 100, 100, 100])
        ys = np.array([100, 200, 300, 400, 500, 600, 700, 800])
        values = np.array([1, 2, 3, 4, 5, 6, 7, 8])

        x_edges, y_edges, grid = aggregate_fast_imshow(
            xs, ys, values, "V", max_bins=100, aggregation="max"
        )

        # For vertical, should have more Y bins than X bins
        n_x_bins = len(x_edges) - 1
        n_y_bins = len(y_edges) - 1
        self.assertGreaterEqual(n_y_bins, n_x_bins)

    def test_aggregate_fast_imshow_sum(self):
        """Test imshow aggregation with sum statistic."""
        xs = np.array([100, 100, 200, 200])
        ys = np.array([100, 100, 200, 200])
        values = np.array([1, 1, 1, 1])

        x_edges, y_edges, grid = aggregate_fast_imshow(
            xs, ys, values, "MIXED", max_bins=2, aggregation="sum"
        )

        # Total sum should be 4
        total = np.nansum(grid)
        self.assertEqual(total, 4)

    def test_aggregate_fast_imshow_min(self):
        """Test imshow aggregation with min statistic."""
        xs = np.array([100, 150, 200, 250])
        ys = np.array([100, 100, 100, 100])
        values = np.array([5, 2, 8, 3])

        x_edges, y_edges, grid = aggregate_fast_imshow(
            xs, ys, values, "H", max_bins=100, aggregation="min"
        )

        # Min value should be 2
        min_val = np.nanmin(grid)
        self.assertEqual(min_val, 2)


class TestRenderFromPrebinnedStripeData(unittest.TestCase):
    """Tests for render_from_prebinned_stripe_data."""

    def _make_stripe_data(self, n_stripes=2, n_bins=5, with_nans=False):
        """Build simple synthetic stripe data for testing.

        Returns stripe_data list suitable for render_from_prebinned_stripe_data.
        """
        stripe_data = []
        for si in range(n_stripes):
            s_start = float(si * 100)
            s_end = float((si + 1) * 100)
            bin_edges = np.linspace(0.0, 500.0, n_bins + 1)
            bin_values = np.linspace(10.0 + si * 5, 50.0 + si * 5, n_bins)
            if with_nans:
                # Set a couple of bins to NaN to simulate empty bins
                bin_values[1] = np.nan
                bin_values[3] = np.nan
            stripe_data.append((s_start, s_end, bin_edges, bin_values))
        return stripe_data

    def test_horizontal_creates_png(self):
        """H orientation should produce a non-empty PNG."""
        stripe_data = self._make_stripe_data()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'test_h.png')
            render_from_prebinned_stripe_data(
                stripe_data=stripe_data,
                orientation='H',
                output_path=output_path,
                layer='M1',
            )
            self.assertTrue(os.path.exists(output_path))
            self.assertGreater(os.path.getsize(output_path), 0)

    def test_vertical_creates_png(self):
        """V orientation should produce a non-empty PNG."""
        stripe_data = self._make_stripe_data()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'test_v.png')
            render_from_prebinned_stripe_data(
                stripe_data=stripe_data,
                orientation='V',
                output_path=output_path,
                layer='M2',
            )
            self.assertTrue(os.path.exists(output_path))
            self.assertGreater(os.path.getsize(output_path), 0)

    def test_nan_bins_do_not_crash(self):
        """Stripes with NaN bins should render without error."""
        stripe_data = self._make_stripe_data(with_nans=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'test_nans.png')
            render_from_prebinned_stripe_data(
                stripe_data=stripe_data,
                orientation='H',
                output_path=output_path,
                layer='M1',
            )
            self.assertTrue(os.path.exists(output_path))
            self.assertGreater(os.path.getsize(output_path), 0)

    def test_all_nan_produces_no_file(self):
        """If every bin is NaN, no PNG should be produced."""
        stripe_data = [
            (0.0, 100.0, np.linspace(0.0, 500.0, 6), np.full(5, np.nan)),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'test_all_nan.png')
            render_from_prebinned_stripe_data(
                stripe_data=stripe_data,
                orientation='H',
                output_path=output_path,
                layer='M1',
            )
            # Function returns early when all values are NaN
            self.assertFalse(os.path.exists(output_path))

    def test_custom_vmin_vmax(self):
        """Explicit vmin/vmax should be honored (no crash)."""
        stripe_data = self._make_stripe_data()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'test_vrange.png')
            render_from_prebinned_stripe_data(
                stripe_data=stripe_data,
                orientation='H',
                output_path=output_path,
                layer='M1',
                vmin=0.0,
                vmax=100.0,
            )
            self.assertTrue(os.path.exists(output_path))
            self.assertGreater(os.path.getsize(output_path), 0)

    def test_n_nodes_in_title(self):
        """n_nodes > 0 should appear in the plot title (smoke test)."""
        stripe_data = self._make_stripe_data(n_stripes=1, n_bins=3)
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'test_nodes.png')
            render_from_prebinned_stripe_data(
                stripe_data=stripe_data,
                orientation='V',
                output_path=output_path,
                layer='M3',
                n_nodes=42000,
                title_prefix='Current',
                value_label='Current (mA)',
            )
            self.assertTrue(os.path.exists(output_path))
            self.assertGreater(os.path.getsize(output_path), 0)

    def test_custom_cmap_and_labels(self):
        """Custom cmap, title_prefix, value_label should not crash."""
        stripe_data = self._make_stripe_data(n_stripes=1, n_bins=4)
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'test_custom.png')
            render_from_prebinned_stripe_data(
                stripe_data=stripe_data,
                orientation='H',
                output_path=output_path,
                layer='M5',
                cmap='viridis',
                title_prefix='Voltage',
                value_label='Voltage (V)',
            )
            self.assertTrue(os.path.exists(output_path))

    def test_empty_stripe_data_produces_no_file(self):
        """Empty stripe_data list should produce no PNG."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'test_empty.png')
            render_from_prebinned_stripe_data(
                stripe_data=[],
                orientation='H',
                output_path=output_path,
                layer='M1',
            )
            self.assertFalse(os.path.exists(output_path))

    def test_single_bin_stripe(self):
        """Single bin per stripe should render without error."""
        stripe_data = [
            (0.0, 100.0, np.array([0.0, 500.0]), np.array([42.0])),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'test_single_bin.png')
            render_from_prebinned_stripe_data(
                stripe_data=stripe_data,
                orientation='H',
                output_path=output_path,
                layer='M1',
            )
            self.assertTrue(os.path.exists(output_path))
            self.assertGreater(os.path.getsize(output_path), 0)

    def test_output_dir_created_automatically(self):
        """Output directory should be created if it does not exist."""
        stripe_data = self._make_stripe_data(n_stripes=1, n_bins=3)
        with tempfile.TemporaryDirectory() as tmpdir:
            nested_path = os.path.join(tmpdir, 'sub', 'dir', 'test.png')
            render_from_prebinned_stripe_data(
                stripe_data=stripe_data,
                orientation='H',
                output_path=nested_path,
                layer='M1',
            )
            self.assertTrue(os.path.exists(nested_path))


if __name__ == "__main__":
    unittest.main()

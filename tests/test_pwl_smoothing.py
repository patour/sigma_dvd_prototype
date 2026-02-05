"""Unit tests for PWL waveform smoothing."""

import unittest
import math
import numpy as np

from core.pwl_smoothing import (
    triangular_window,
    analytical_triangle_pwl_integral,
    smooth_pwl_points,
    compact_pwl,
    pulse_to_pwl_points,
    PWLSmoother,
    SmoothedWaveformCache,
    SmoothingConfig,
)


class TestTriangularWindow(unittest.TestCase):
    """Tests for triangular_window function."""

    def test_peak_at_center(self):
        """Triangle should have peak of 1.0 at center."""
        self.assertAlmostEqual(triangular_window(5.0, 5.0, 1.0), 1.0)
        self.assertAlmostEqual(triangular_window(0.0, 0.0, 2.0), 1.0)

    def test_zero_at_edges(self):
        """Triangle should be zero at edges."""
        h = 1.0
        center = 5.0
        self.assertAlmostEqual(triangular_window(center - h, center, h), 0.0)
        self.assertAlmostEqual(triangular_window(center + h, center, h), 0.0)

    def test_zero_outside_window(self):
        """Triangle should be zero outside window."""
        h = 1.0
        center = 5.0
        self.assertAlmostEqual(triangular_window(center - h - 0.1, center, h), 0.0)
        self.assertAlmostEqual(triangular_window(center + h + 0.1, center, h), 0.0)

    def test_linear_interpolation(self):
        """Triangle should interpolate linearly from edges to peak."""
        h = 2.0
        center = 4.0
        # Left half: at t = center - h/2, should be 0.5
        self.assertAlmostEqual(triangular_window(center - h / 2, center, h), 0.5)
        # Right half: at t = center + h/2, should be 0.5
        self.assertAlmostEqual(triangular_window(center + h / 2, center, h), 0.5)

    def test_zero_half_width(self):
        """Zero half_width should give delta-like behavior."""
        self.assertAlmostEqual(triangular_window(0.0, 0.0, 0.0), 1.0)
        self.assertAlmostEqual(triangular_window(0.001, 0.0, 0.0), 0.0)


class TestAnalyticalIntegral(unittest.TestCase):
    """Tests for analytical_triangle_pwl_integral function."""

    def test_constant_segment_full_overlap(self):
        """Integral of triangle * constant should equal constant * triangle_area."""
        # Triangle centered at t=2 with half_width=1, constant PWL segment v=3.0 from t=0 to t=4
        # Triangle area = half_width = 1.0
        # Expected: 3.0 * 1.0 = 3.0
        result = analytical_triangle_pwl_integral(2.0, 1.0, 0.0, 3.0, 4.0, 3.0)
        self.assertAlmostEqual(result, 3.0, places=10)

    def test_constant_segment_partial_overlap(self):
        """Partial overlap should give proportionally smaller result."""
        # Triangle centered at t=2 with half_width=1 -> window [1, 3]
        # PWL segment constant=2.0 from t=0 to t=2 -> only covers [1, 2]
        # This is half of the left triangle (area = 0.5 * 1.0 = 0.5)
        # Expected: 2.0 * 0.5 = 1.0
        result = analytical_triangle_pwl_integral(2.0, 1.0, 0.0, 2.0, 2.0, 2.0)
        self.assertAlmostEqual(result, 1.0, places=10)

    def test_no_overlap(self):
        """No overlap should give zero."""
        # Triangle at t=5, segment from t=0 to t=1
        result = analytical_triangle_pwl_integral(5.0, 1.0, 0.0, 1.0, 1.0, 1.0)
        self.assertAlmostEqual(result, 0.0, places=10)

    def test_linear_segment_full_overlap(self):
        """Linear segment across full triangle should integrate correctly."""
        # Use numerical integration to verify
        from scipy import integrate

        t_out = 5.0
        h = 1.0
        t1, v1 = 4.0, 0.0
        t2, v2 = 6.0, 2.0

        # Analytical result
        analytical = analytical_triangle_pwl_integral(t_out, h, t1, v1, t2, v2)

        # Numerical integration
        def integrand(t):
            # Triangle
            if t < t_out - h or t > t_out + h:
                return 0.0
            tri = 1.0 - abs(t - t_out) / h
            # PWL
            pwl = v1 + (v2 - v1) * (t - t1) / (t2 - t1)
            return tri * pwl

        numerical, _ = integrate.quad(integrand, t1, t2)

        self.assertAlmostEqual(analytical, numerical, places=8)

    def test_symmetric_linear_segment(self):
        """Linear segment symmetric around triangle center should integrate to line at center."""
        # Triangle at t=0, half_width=1, segment from -1 to 1 with values -1 to 1
        # By symmetry, this should equal 0 * triangle_area = 0
        result = analytical_triangle_pwl_integral(0.0, 1.0, -1.0, -1.0, 1.0, 1.0)
        self.assertAlmostEqual(result, 0.0, places=10)


class TestSmoothPWLPoints(unittest.TestCase):
    """Tests for smooth_pwl_points function."""

    def test_constant_pwl_unchanged(self):
        """Constant PWL should remain constant after smoothing (interior points)."""
        points = [(0.0, 5.0), (10.0, 5.0)]
        smoothed = smooth_pwl_points(points, period=0, time_step=1.0, t_start=0, t_end=10)

        # Interior values should be 5.0 (edge points affected by boundary)
        for t, v in smoothed:
            if 1.0 <= t <= 9.0:  # Skip edge points
                self.assertAlmostEqual(v, 5.0, places=8)

    def test_dc_preservation(self):
        """Smoothing should preserve DC (average) value for periodic signals."""
        # Square wave: 0 for t in [0, 5), 1 for t in [5, 10)
        points = [(0.0, 0.0), (5.0, 0.0), (5.0, 1.0), (10.0, 1.0)]
        period = 10.0

        smoothed = smooth_pwl_points(points, period=period, time_step=0.5, t_start=0, t_end=10)

        # Original DC = (0 * 5 + 1 * 5) / 10 = 0.5
        original_dc = 0.5

        # Smoothed DC (trapezoidal integration)
        smoothed_sum = 0.0
        for i in range(len(smoothed) - 1):
            t1, v1 = smoothed[i]
            t2, v2 = smoothed[i + 1]
            smoothed_sum += (v1 + v2) / 2 * (t2 - t1)
        smoothed_dc = smoothed_sum / (smoothed[-1][0] - smoothed[0][0])

        self.assertAlmostEqual(smoothed_dc, original_dc, places=2)

    def test_smoothing_reduces_high_frequency(self):
        """Smoothed signal should have reduced high-frequency content."""
        # Fast oscillation
        points = [(i * 0.1, math.sin(i * 10 * math.pi)) for i in range(101)]

        smoothed = smooth_pwl_points(points, period=0, time_step=0.5, t_start=0, t_end=10)

        # Smoothed signal should have much smaller amplitude
        max_original = max(abs(v) for _, v in points)
        max_smoothed = max(abs(v) for _, v in smoothed)

        self.assertLess(max_smoothed, max_original * 0.5)


class TestCompactPWL(unittest.TestCase):
    """Tests for compact_pwl function."""

    def test_keeps_first_and_last(self):
        """Compaction should always keep first and last points."""
        points = [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]
        compacted = compact_pwl(points)

        self.assertEqual(compacted[0], points[0])
        self.assertEqual(compacted[-1], points[-1])

    def test_removes_collinear_points(self):
        """Collinear interior points should be removed."""
        # All points on line y = x
        points = [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]
        compacted = compact_pwl(points)

        # Should reduce to just first and last
        self.assertEqual(len(compacted), 2)
        self.assertEqual(compacted[0], (0.0, 0.0))
        self.assertEqual(compacted[1], (3.0, 3.0))

    def test_preserves_slope_changes(self):
        """Points where slope changes should be preserved."""
        # Triangle: 0 -> 1 -> 0
        points = [(0.0, 0.0), (1.0, 1.0), (2.0, 0.0)]
        compacted = compact_pwl(points)

        # All points should be preserved
        self.assertEqual(len(compacted), 3)

    def test_removes_flat_region_interior(self):
        """Interior points in flat regions should be removed."""
        # Flat segment with extra points
        points = [(0.0, 5.0), (1.0, 5.0), (2.0, 5.0), (3.0, 5.0)]
        compacted = compact_pwl(points)

        # Should reduce to just first and last
        self.assertEqual(len(compacted), 2)

    def test_short_list_unchanged(self):
        """Lists with <= 2 points should be unchanged."""
        points1 = [(0.0, 0.0)]
        points2 = [(0.0, 0.0), (1.0, 1.0)]

        self.assertEqual(compact_pwl(points1), points1)
        self.assertEqual(compact_pwl(points2), points2)


class TestPulseToPWL(unittest.TestCase):
    """Tests for pulse_to_pwl_points function."""

    def test_basic_pulse_timing(self):
        """Pulse timing should match SPICE convention."""
        points = pulse_to_pwl_points(
            v1=0.0,
            v2=1.0,
            delay=1.0,
            rt=0.1,
            ft=0.1,
            width=2.0,
            period=5.0,
            n_periods=1,
        )

        # Extract key times and values
        times = [p[0] for p in points]
        values = [p[1] for p in points]

        # Check timing: start at v1, rise at delay, high at delay+rt, fall at delay+rt+width
        self.assertIn(0.0, times)  # Start
        self.assertIn(1.0, times)  # Delay
        self.assertIn(1.1, times)  # End of rise (delay + rt)
        self.assertIn(3.1, times)  # End of high (delay + rt + width)
        self.assertIn(3.2, times)  # End of fall (delay + rt + width + ft)

    def test_pulse_dc_value(self):
        """Pulse DC value should match duty cycle."""
        # Pulse high for 50% of period
        points = pulse_to_pwl_points(
            v1=0.0,
            v2=1.0,
            delay=0.0,
            rt=0.0,  # Instantaneous rise
            ft=0.0,  # Instantaneous fall
            width=5.0,  # High for 5 units
            period=10.0,  # Period is 10 units
            n_periods=1,
        )

        # Compute DC by integration
        total_area = 0.0
        for i in range(len(points) - 1):
            t1, v1 = points[i]
            t2, v2 = points[i + 1]
            total_area += (v1 + v2) / 2 * (t2 - t1)

        period = points[-1][0] - points[0][0]
        dc = total_area / period

        # Should be approximately 0.5
        self.assertAlmostEqual(dc, 0.5, places=2)

    def test_multi_period(self):
        """Multi-period pulse should repeat correctly."""
        points = pulse_to_pwl_points(
            v1=0.0,
            v2=1.0,
            delay=0.5,
            rt=0.1,
            ft=0.1,
            width=0.5,
            period=2.0,
            n_periods=3,
        )

        # Should span 3 periods
        self.assertGreaterEqual(points[-1][0], 6.0)


class TestPWLSmoother(unittest.TestCase):
    """Tests for PWLSmoother class."""

    def test_disabled_smoother_passthrough(self):
        """Disabled smoother should return input unchanged."""
        from pdn.pdn_parser import PWL

        smoother = PWLSmoother(time_step=0.1, enabled=False)
        pwl = PWL(points=[(0, 0), (1, 1), (2, 0)], period=2.0)

        smoothed = smoother.smooth_pwl(pwl, t_start=0, t_end=2)

        # Should have same points (identity transform when disabled)
        self.assertEqual(smoothed.points, pwl.points)

    def test_smooth_pulse(self):
        """Smoother should convert pulse to smoothed PWL."""
        from pdn.pdn_parser import Pulse

        smoother = PWLSmoother(time_step=0.1)
        pulse = Pulse(v1=0, v2=1, delay=0.1, rt=0.1, ft=0.1, width=0.5, period=1.0)

        smoothed = smoother.smooth_pulse(pulse, t_start=0, t_end=1)

        # Result should be a PWL object
        self.assertIsNotNone(smoothed.points)
        self.assertGreater(len(smoothed.points), 0)

    def test_statistics_tracking(self):
        """Smoother should track statistics."""
        smoother = PWLSmoother(time_step=0.1)

        # Need to call create_smoothed_cache to populate stats
        # For now, just verify get_statistics returns a dict
        stats = smoother.get_statistics()
        self.assertIsInstance(stats, dict)


class TestSmoothedWaveformCache(unittest.TestCase):
    """Tests for SmoothedWaveformCache class."""

    def test_is_compatible_exact_match(self):
        """Cache with exact parameters should be compatible."""
        cache = SmoothedWaveformCache(
            time_step=0.1,
            t_start=0.0,
            t_end=100.0,
            compact_threshold=1e-12,
        )

        self.assertTrue(cache.is_compatible(0.1, 0.0, 100.0))

    def test_is_compatible_subset_range(self):
        """Cache covering larger range should be compatible with subset."""
        cache = SmoothedWaveformCache(
            time_step=0.1,
            t_start=0.0,
            t_end=100.0,
            compact_threshold=1e-12,
        )

        # Subset range should be compatible
        self.assertTrue(cache.is_compatible(0.1, 10.0, 90.0))

    def test_is_compatible_different_step(self):
        """Cache with different time step should not be compatible."""
        cache = SmoothedWaveformCache(
            time_step=0.1,
            t_start=0.0,
            t_end=100.0,
            compact_threshold=1e-12,
        )

        self.assertFalse(cache.is_compatible(0.2, 0.0, 100.0))

    def test_is_compatible_larger_range_needed(self):
        """Cache should not be compatible if larger range is needed."""
        cache = SmoothedWaveformCache(
            time_step=0.1,
            t_start=10.0,
            t_end=90.0,
            compact_threshold=1e-12,
        )

        # Request range extends beyond cache
        self.assertFalse(cache.is_compatible(0.1, 0.0, 100.0))


class TestPeriodicWraparound(unittest.TestCase):
    """Tests for periodic boundary handling."""

    def test_periodic_smooth_near_boundary(self):
        """Smoothing near period boundary should handle wraparound."""
        # Sawtooth wave that wraps at boundary
        points = [(0.0, 0.0), (5.0, 1.0), (10.0, 0.0)]
        period = 10.0

        # Smooth near the period boundary
        smoothed = smooth_pwl_points(points, period=period, time_step=2.0, t_start=0, t_end=10)

        # Values near t=0 and t=10 should be similar (periodic)
        v_near_start = smoothed[0][1] if smoothed else 0
        v_near_end = smoothed[-1][1] if smoothed else 0

        # Should be approximately equal due to periodicity
        self.assertAlmostEqual(v_near_start, v_near_end, places=1)


class TestIntegrationVectorizedSources(unittest.TestCase):
    """Integration tests for VectorizedCurrentSources smoothing."""

    def test_smoothed_cache_reuse(self):
        """Applying same cache twice should produce identical results."""
        from core.vectorized_sources import VectorizedCurrentSources

        # Create mock vectorized sources with some PWLs
        sources = VectorizedCurrentSources(
            n_nodes=10,
            node_to_idx={f'n{i}': i for i in range(10)},
            n_sources=2,
            dc_values=np.array([1.0, 2.0]),
            source_node_idx=np.array([0, 1], dtype=np.int32),
            n_pulses=0,
            pulse_node_idx=np.array([], dtype=np.int32),
            pulse_source_idx=np.array([], dtype=np.int32),
            pulse_v1=np.array([], dtype=np.float64),
            pulse_v2=np.array([], dtype=np.float64),
            pulse_delay=np.array([], dtype=np.float64),
            pulse_rt=np.array([], dtype=np.float64),
            pulse_ft=np.array([], dtype=np.float64),
            pulse_width=np.array([], dtype=np.float64),
            pulse_period=np.array([], dtype=np.float64),
            n_pwls=2,
            n_pwl_points=4,
            pwl_node_idx=np.array([0, 1], dtype=np.int32),
            pwl_source_idx=np.array([0, 1], dtype=np.int32),
            pwl_period=np.array([10.0, 10.0]),
            pwl_delay=np.array([0.0, 0.0]),
            pwl_offset=np.array([0, 2], dtype=np.int32),
            pwl_count=np.array([2, 2], dtype=np.int32),
            pwl_times=np.array([0.0, 10.0, 0.0, 10.0]),
            pwl_values=np.array([0.0, 1.0, 1.0, 0.0]),
        )

        smoother = PWLSmoother(time_step=1.0, compact_threshold=1e-12)

        # Create cache
        cache = smoother.create_smoothed_cache(sources, t_start=0, t_end=10)

        # Apply cache twice
        smoothed1 = smoother.apply_cache_to_sources(sources, cache)
        smoothed2 = smoother.apply_cache_to_sources(sources, cache)

        # Results should be identical
        np.testing.assert_array_equal(smoothed1.pwl_times, smoothed2.pwl_times)
        np.testing.assert_array_equal(smoothed1.pwl_values, smoothed2.pwl_values)

    def test_pulse_smoothing_in_vectorized(self):
        """Pulses should be converted to PWL and smoothed."""
        from core.vectorized_sources import VectorizedCurrentSources

        # Create sources with pulses
        sources = VectorizedCurrentSources(
            n_nodes=5,
            node_to_idx={f'n{i}': i for i in range(5)},
            n_sources=1,
            dc_values=np.array([0.0]),
            source_node_idx=np.array([0], dtype=np.int32),
            n_pulses=1,
            pulse_node_idx=np.array([0], dtype=np.int32),
            pulse_source_idx=np.array([0], dtype=np.int32),
            pulse_v1=np.array([0.0]),
            pulse_v2=np.array([1.0]),
            pulse_delay=np.array([0.0]),
            pulse_rt=np.array([0.1]),
            pulse_ft=np.array([0.1]),
            pulse_width=np.array([0.5]),
            pulse_period=np.array([2.0]),
            n_pwls=0,
            n_pwl_points=0,
            pwl_node_idx=np.array([], dtype=np.int32),
            pwl_source_idx=np.array([], dtype=np.int32),
            pwl_period=np.array([]),
            pwl_delay=np.array([]),
            pwl_offset=np.array([], dtype=np.int32),
            pwl_count=np.array([], dtype=np.int32),
            pwl_times=np.array([]),
            pwl_values=np.array([]),
        )

        smoother = PWLSmoother(time_step=0.1, compact_threshold=1e-12)
        cache = smoother.create_smoothed_cache(sources, t_start=0, t_end=2)

        # Pulses should be converted to PWLs
        self.assertEqual(cache.original_n_pulses, 1)
        self.assertEqual(cache.original_n_pwls, 0)
        self.assertGreater(cache.n_pwls, 0)  # Should have converted pulse to PWL

        # Apply cache - smoothed sources should have no pulses
        smoothed = smoother.apply_cache_to_sources(sources, cache)
        self.assertEqual(smoothed.n_pulses, 0)
        self.assertGreater(smoothed.n_pwls, 0)

    def test_create_smoothed_copy_convenience(self):
        """VectorizedCurrentSources.create_smoothed_copy should work."""
        from core.vectorized_sources import VectorizedCurrentSources

        # Create sources with PWL
        sources = VectorizedCurrentSources(
            n_nodes=5,
            node_to_idx={f'n{i}': i for i in range(5)},
            n_sources=1,
            dc_values=np.array([1.0]),
            source_node_idx=np.array([0], dtype=np.int32),
            n_pulses=0,
            pulse_node_idx=np.array([], dtype=np.int32),
            pulse_source_idx=np.array([], dtype=np.int32),
            pulse_v1=np.array([]),
            pulse_v2=np.array([]),
            pulse_delay=np.array([]),
            pulse_rt=np.array([]),
            pulse_ft=np.array([]),
            pulse_width=np.array([]),
            pulse_period=np.array([]),
            n_pwls=1,
            n_pwl_points=3,
            pwl_node_idx=np.array([0], dtype=np.int32),
            pwl_source_idx=np.array([0], dtype=np.int32),
            pwl_period=np.array([10.0]),
            pwl_delay=np.array([0.0]),
            pwl_offset=np.array([0], dtype=np.int32),
            pwl_count=np.array([3], dtype=np.int32),
            pwl_times=np.array([0.0, 5.0, 10.0]),
            pwl_values=np.array([0.0, 1.0, 0.0]),
        )

        # Use convenience method
        smoothed = sources.create_smoothed_copy(
            time_step=1.0,
            t_start=0,
            t_end=10,
        )

        # Should have PWLs
        self.assertGreater(smoothed.n_pwls, 0)
        self.assertGreater(smoothed.n_pwl_points, 0)

        # DC values should be preserved
        np.testing.assert_array_equal(smoothed.dc_values, sources.dc_values)


if __name__ == '__main__':
    unittest.main()

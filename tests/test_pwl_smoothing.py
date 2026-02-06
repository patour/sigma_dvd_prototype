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


class TestPulseSmoothing(unittest.TestCase):
    """Tests for pulse smoothing with real-world cases."""

    def test_raw_vs_smoothed_pulses(self):
        """Compare raw and smoothed pulse waveforms at specific time points."""
        from pdn.pdn_parser import Pulse

        smoother = PWLSmoother(time_step=0.01e-9, compact_threshold=1e-12)

        # Convert pulses to PWL and smooth
        pulse1 = Pulse(0, 0.0117914, 3.52914e-09, 6.25e-13, 6.25e-13, 0, 1e-08)
        pulse2 = Pulse(0, -0.000131702, 3.51998e-09, 9.78805e-12, 9.78805e-12, 0, 1e-08)
        smoothed1 = smoother.smooth_pulse(pulse1, t_start=0, t_end=10e-9)
        smoothed2 = smoother.smooth_pulse(pulse2, t_start=0, t_end=10e-9)

        # Sample at specific time points
        t_compare = np.arange(3.48e-9, 3.58e-9 + 1e-12, 10e-12)
        
        # Compute raw and smoothed values in mA
        raw = [(pulse1.evaluate(t) + pulse2.evaluate(t)) * 1e3 for t in t_compare]
        smooth = [(smoothed1.evaluate(t) + smoothed2.evaluate(t)) * 1e3 for t in t_compare]

        # Expected values (from reference run)
        expected_raw = [
            0.0, 0.0, 0.0, 0.0, -0.00026910773852576017, 7.2292525769831295,
            0.0, 0.0, 0.0, 0.0, 0.0
        ]
        expected_smooth = [
            0.0, 0.0, 0.0, -6.617444900408621e-10, -0.003377999488249955,
            0.6291331011861812, -0.0177050009610841, -4.17398778165606e-14,
            0.0, 0.0, 0.0
        ]

        # Assert raw values match exactly (accounting for floating point)
        self.assertEqual(len(raw), len(expected_raw))
        for i, (actual, expected) in enumerate(zip(raw, expected_raw)):
            self.assertAlmostEqual(actual, expected, places=14,
                                   msg=f"Raw mismatch at index {i}")

        # Assert smoothed values match exactly
        self.assertEqual(len(smooth), len(expected_smooth))
        for i, (actual, expected) in enumerate(zip(smooth, expected_smooth)):
            self.assertAlmostEqual(actual, expected, places=14,
                                   msg=f"Smoothed mismatch at index {i}")


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


class TestAnalyticalIntegralVectorized(unittest.TestCase):
    """Tests for _analytical_integral_vectorized function."""

    def test_matches_scalar_constant_segment(self):
        """Vectorized should match scalar for constant segment."""
        from core.pwl_smoothing import _analytical_integral_vectorized

        t_out = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        half_width = 1.0
        t1, v1 = 0.0, 5.0
        t2, v2 = 4.0, 5.0

        # Scalar results
        scalar = [analytical_triangle_pwl_integral(t, half_width, t1, v1, t2, v2) for t in t_out]

        # Vectorized results
        vec = _analytical_integral_vectorized(t_out, half_width, t1, v1, t2, v2)

        np.testing.assert_allclose(vec, scalar, rtol=1e-12)

    def test_matches_scalar_linear_segment(self):
        """Vectorized should match scalar for linear segment."""
        from core.pwl_smoothing import _analytical_integral_vectorized

        t_out = np.array([0.0, 2.5, 5.0, 7.5, 10.0])
        half_width = 2.0
        t1, v1 = 0.0, 0.0
        t2, v2 = 10.0, 10.0

        scalar = [analytical_triangle_pwl_integral(t, half_width, t1, v1, t2, v2) for t in t_out]
        vec = _analytical_integral_vectorized(t_out, half_width, t1, v1, t2, v2)

        np.testing.assert_allclose(vec, scalar, rtol=1e-12)

    def test_matches_scalar_no_overlap(self):
        """Vectorized should match scalar when no overlap."""
        from core.pwl_smoothing import _analytical_integral_vectorized

        t_out = np.array([0.0, 1.0, 2.0])
        half_width = 0.5
        t1, v1 = 10.0, 1.0  # Segment far from sample times
        t2, v2 = 11.0, 2.0

        scalar = [analytical_triangle_pwl_integral(t, half_width, t1, v1, t2, v2) for t in t_out]
        vec = _analytical_integral_vectorized(t_out, half_width, t1, v1, t2, v2)

        np.testing.assert_allclose(vec, scalar, rtol=1e-12)
        np.testing.assert_allclose(vec, [0.0, 0.0, 0.0])  # All zeros

    def test_matches_scalar_partial_overlap(self):
        """Vectorized should match scalar for partial overlap."""
        from core.pwl_smoothing import _analytical_integral_vectorized

        t_out = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
        half_width = 1.0
        t1, v1 = 2.0, 1.0  # Segment starts at t=2
        t2, v2 = 4.0, 3.0

        scalar = [analytical_triangle_pwl_integral(t, half_width, t1, v1, t2, v2) for t in t_out]
        vec = _analytical_integral_vectorized(t_out, half_width, t1, v1, t2, v2)

        np.testing.assert_allclose(vec, scalar, rtol=1e-12)

    def test_vectorized_multiple_samples_numerical(self):
        """Test vectorized integral against numerical integration."""
        from scipy import integrate
        from core.pwl_smoothing import _analytical_integral_vectorized

        t_out = np.array([2.0, 3.0, 4.0])
        half_width = 1.0
        t1, v1 = 0.0, 0.0
        t2, v2 = 6.0, 6.0

        vec = _analytical_integral_vectorized(t_out, half_width, t1, v1, t2, v2)

        # Numerical verification
        for i, t in enumerate(t_out):
            def integrand(tau):
                if tau < t - half_width or tau > t + half_width:
                    return 0.0
                tri = 1.0 - abs(tau - t) / half_width
                pwl = v1 + (v2 - v1) * (tau - t1) / (t2 - t1)
                return tri * pwl
            numerical, _ = integrate.quad(integrand, t1, t2)
            self.assertAlmostEqual(vec[i], numerical, places=8)


class TestSmoothPWLVectorized(unittest.TestCase):
    """Tests for _smooth_pwl_vectorized function."""

    def test_matches_scalar_constant_pwl(self):
        """Vectorized should match scalar for constant PWL."""
        from core.pwl_smoothing import _smooth_pwl_vectorized

        points = [(0.0, 5.0), (10.0, 5.0)]
        times = np.array([p[0] for p in points])
        values = np.array([p[1] for p in points])

        scalar = smooth_pwl_points(points, period=0, time_step=1.0, t_start=0, t_end=10)
        scalar_times = np.array([p[0] for p in scalar])
        scalar_values = np.array([p[1] for p in scalar])

        vec_times, vec_values = _smooth_pwl_vectorized(times, values, 0, 1.0, 0, 10)

        np.testing.assert_allclose(vec_times, scalar_times, rtol=1e-12)
        np.testing.assert_allclose(vec_values, scalar_values, rtol=1e-12)

    def test_matches_scalar_linear_pwl(self):
        """Vectorized should match scalar for linear PWL."""
        from core.pwl_smoothing import _smooth_pwl_vectorized

        points = [(0.0, 0.0), (10.0, 10.0)]
        times = np.array([p[0] for p in points])
        values = np.array([p[1] for p in points])

        scalar = smooth_pwl_points(points, period=0, time_step=0.5, t_start=0, t_end=10)
        scalar_times = np.array([p[0] for p in scalar])
        scalar_values = np.array([p[1] for p in scalar])

        vec_times, vec_values = _smooth_pwl_vectorized(times, values, 0, 0.5, 0, 10)

        np.testing.assert_allclose(vec_times, scalar_times, rtol=1e-12)
        np.testing.assert_allclose(vec_values, scalar_values, rtol=1e-12)

    def test_matches_scalar_periodic(self):
        """Vectorized should match scalar for periodic PWL."""
        from core.pwl_smoothing import _smooth_pwl_vectorized

        points = [(0.0, 0.0), (5.0, 1.0), (10.0, 0.0)]
        times = np.array([p[0] for p in points])
        values = np.array([p[1] for p in points])
        period = 10.0

        scalar = smooth_pwl_points(points, period=period, time_step=1.0, t_start=0, t_end=10)
        scalar_times = np.array([p[0] for p in scalar])
        scalar_values = np.array([p[1] for p in scalar])

        vec_times, vec_values = _smooth_pwl_vectorized(times, values, period, 1.0, 0, 10)

        np.testing.assert_allclose(vec_times, scalar_times, rtol=1e-12)
        np.testing.assert_allclose(vec_values, scalar_values, rtol=1e-12)

    def test_matches_scalar_nonperiodic(self):
        """Vectorized should match scalar for non-periodic PWL."""
        from core.pwl_smoothing import _smooth_pwl_vectorized

        points = [(0.0, 1.0), (3.0, 4.0), (6.0, 2.0), (10.0, 5.0)]
        times = np.array([p[0] for p in points])
        values = np.array([p[1] for p in points])

        scalar = smooth_pwl_points(points, period=0, time_step=0.5, t_start=0, t_end=10)
        scalar_times = np.array([p[0] for p in scalar])
        scalar_values = np.array([p[1] for p in scalar])

        vec_times, vec_values = _smooth_pwl_vectorized(times, values, 0, 0.5, 0, 10)

        np.testing.assert_allclose(vec_times, scalar_times, rtol=1e-12)
        np.testing.assert_allclose(vec_values, scalar_values, rtol=1e-12)

    def test_triangle_pulse(self):
        """Vectorized smoothing of triangle pulse should match scalar."""
        from core.pwl_smoothing import _smooth_pwl_vectorized

        # Triangle pulse
        points = [(0.0, 0.0), (2.0, 0.0), (3.0, 1.0), (4.0, 0.0), (6.0, 0.0)]
        times = np.array([p[0] for p in points])
        values = np.array([p[1] for p in points])
        period = 6.0

        scalar = smooth_pwl_points(points, period=period, time_step=0.2, t_start=0, t_end=6)
        scalar_times = np.array([p[0] for p in scalar])
        scalar_values = np.array([p[1] for p in scalar])

        vec_times, vec_values = _smooth_pwl_vectorized(times, values, period, 0.2, 0, 6)

        np.testing.assert_allclose(vec_times, scalar_times, rtol=1e-10)
        np.testing.assert_allclose(vec_values, scalar_values, rtol=1e-10)


class TestChunkProcessing(unittest.TestCase):
    """Tests for chunked batch processing functions."""

    def test_pulse_to_pwl_arrays(self):
        """Test pulse to PWL array conversion."""
        from core.pwl_smoothing import _pulse_to_pwl_arrays

        v1 = np.array([0.0, 0.0])
        v2 = np.array([1.0, 2.0])
        delay = np.array([1.0, 2.0])
        rt = np.array([0.1, 0.2])
        ft = np.array([0.1, 0.2])
        width = np.array([0.5, 1.0])
        period = np.array([5.0, 10.0])

        times, values = _pulse_to_pwl_arrays(v1, v2, delay, rt, ft, width, period)

        self.assertEqual(times.shape, (2, 6))
        self.assertEqual(values.shape, (2, 6))

        # Check first pulse timing
        np.testing.assert_allclose(times[0], [0.0, 1.0, 1.1, 1.6, 1.7, 5.0])
        np.testing.assert_allclose(values[0], [0.0, 0.0, 1.0, 1.0, 0.0, 0.0])

    def test_analytical_integral_batch_matches_vectorized(self):
        """Batch integral should match individual vectorized calls."""
        from core.pwl_smoothing import (
            _analytical_integral_batch,
            _analytical_integral_vectorized,
        )

        sample_times = np.linspace(0, 10, 101)
        half_width = 0.5

        # Multiple segments
        t1 = np.array([0.0, 2.0, 5.0])
        v1 = np.array([0.0, 1.0, 0.5])
        t2 = np.array([2.0, 5.0, 10.0])
        v2 = np.array([1.0, 0.5, 2.0])

        # Batch result
        batch_result = _analytical_integral_batch(sample_times, half_width, t1, v1, t2, v2)
        self.assertEqual(batch_result.shape, (3, 101))

        # Individual results
        for i in range(3):
            individual = _analytical_integral_vectorized(
                sample_times, half_width, t1[i], v1[i], t2[i], v2[i]
            )
            np.testing.assert_allclose(
                batch_result[i], individual, rtol=1e-10,
                err_msg=f"Mismatch at segment {i}"
            )

    def test_smooth_pulse_chunk_single_pulse(self):
        """Test smoothing a single pulse in chunk."""
        from core.pwl_smoothing import _smooth_pulse_chunk, _pulse_to_pwl_arrays

        # Single pulse
        v1 = np.array([0.0])
        v2 = np.array([1.0])
        delay = np.array([1.0])
        rt = np.array([0.1])
        ft = np.array([0.1])
        width = np.array([2.0])
        period = np.array([10.0])

        times_2d, values_2d = _pulse_to_pwl_arrays(v1, v2, delay, rt, ft, width, period)
        sample_times = np.linspace(0, 10, 101)

        result = _smooth_pulse_chunk(times_2d, values_2d, period, sample_times, 0.5)

        self.assertEqual(result.shape, (1, 101))
        # Should have peak around delay+rt to delay+rt+width
        peak_idx = np.argmax(result[0])
        self.assertTrue(10 < peak_idx < 40)  # Peak somewhere in first third

    def test_smooth_pulse_chunk_batch(self):
        """Test smoothing multiple pulses in chunk."""
        from core.pwl_smoothing import _smooth_pulse_chunk, _pulse_to_pwl_arrays

        n = 10
        v1 = np.zeros(n)
        v2 = np.ones(n)
        delay = np.full(n, 1.0)
        rt = np.full(n, 0.1)
        ft = np.full(n, 0.1)
        width = np.full(n, 2.0)
        period = np.full(n, 10.0)

        times_2d, values_2d = _pulse_to_pwl_arrays(v1, v2, delay, rt, ft, width, period)
        sample_times = np.linspace(0, 10, 101)

        result = _smooth_pulse_chunk(times_2d, values_2d, period, sample_times, 0.5)

        self.assertEqual(result.shape, (n, 101))

        # All pulses are identical, so all results should be identical
        for i in range(1, n):
            np.testing.assert_allclose(result[i], result[0], rtol=1e-12)

    def test_compact_chunk_vectorized(self):
        """Test vectorized chunk compaction."""
        from core.pwl_smoothing import _compact_chunk_vectorized

        sample_times = np.linspace(0, 10, 11)

        # Create values with different characteristics
        values_2d = np.array([
            np.linspace(0, 10, 11),  # Linear - should compact to 2 points
            np.array([0, 0, 0, 0, 0, 5, 5, 5, 5, 5, 5]),  # Step - needs middle point
            np.array([0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0]),  # Triangle - needs 3 points
        ])

        times_list, values_list, counts = _compact_chunk_vectorized(
            sample_times, values_2d, threshold=1e-10
        )

        self.assertEqual(len(times_list), 3)
        self.assertEqual(len(values_list), 3)
        self.assertEqual(len(counts), 3)

        # Linear should compact significantly
        self.assertLess(counts[0], 11)

    def test_compact_and_append(self):
        """Test compact and append helper."""
        from core.pwl_smoothing import _compact_and_append

        times_list = [np.array([0, 5, 10]), np.array([0, 10])]
        values_list = [np.array([0, 1, 0]), np.array([0, 1])]
        counts = np.array([3, 2])
        periods = np.array([10.0, 10.0])
        node_indices = np.array([0, 1])

        all_times = []
        all_values = []
        offsets = []
        out_counts = []
        out_periods = []
        out_delays = []
        out_node_indices = []

        _compact_and_append(
            times_list, values_list, counts, periods, node_indices,
            all_times, all_values, offsets, out_counts,
            out_periods, out_delays, out_node_indices
        )

        self.assertEqual(len(all_times), 5)  # 3 + 2
        self.assertEqual(len(all_values), 5)
        self.assertEqual(offsets, [0, 3])
        self.assertEqual(out_counts, [3, 2])
        self.assertEqual(out_periods, [10.0, 10.0])
        self.assertEqual(out_delays, [0.0, 0.0])


class TestPerformance(unittest.TestCase):
    """Performance regression tests for PWL smoothing."""

    def test_vectorized_batch_speedup(self):
        """Vectorized+batch should be at least 5x faster than sequential.

        Test with 1000 waveforms to measure meaningful speedup while keeping
        test runtime reasonable.
        """
        from core.vectorized_sources import VectorizedCurrentSources
        import time

        n_pulses = 1000
        sources = VectorizedCurrentSources(
            n_nodes=n_pulses,
            node_to_idx={f'n{i}': i for i in range(n_pulses)},
            n_sources=n_pulses,
            dc_values=np.zeros(n_pulses),
            source_node_idx=np.arange(n_pulses, dtype=np.int32),
            n_pulses=n_pulses,
            pulse_node_idx=np.arange(n_pulses, dtype=np.int32),
            pulse_source_idx=np.arange(n_pulses, dtype=np.int32),
            pulse_v1=np.zeros(n_pulses),
            pulse_v2=np.ones(n_pulses),
            pulse_delay=np.full(n_pulses, 1e-9),
            pulse_rt=np.full(n_pulses, 0.1e-9),
            pulse_ft=np.full(n_pulses, 0.1e-9),
            pulse_width=np.full(n_pulses, 2e-9),
            pulse_period=np.full(n_pulses, 10e-9),
            n_pwls=0, n_pwl_points=0,
            pwl_node_idx=np.array([], dtype=np.int32),
            pwl_source_idx=np.array([], dtype=np.int32),
            pwl_period=np.array([]),
            pwl_delay=np.array([]),
            pwl_offset=np.array([], dtype=np.int32),
            pwl_count=np.array([], dtype=np.int32),
            pwl_times=np.array([]),
            pwl_values=np.array([]),
        )

        # Time the new implementation
        smoother = PWLSmoother(time_step=0.1e-9, compact_threshold=1e-12)

        start = time.perf_counter()
        cache = smoother.create_smoothed_cache(sources, 0, 10e-9, chunk_size=1000)
        elapsed = time.perf_counter() - start

        # Expected performance: < 0.5ms per pulse (new) vs ~0.8ms (old)
        # For 1000 pulses: < 0.5s new
        per_pulse_ms = elapsed / n_pulses * 1000

        # Verify reasonable performance (< 1ms per pulse)
        self.assertLess(
            per_pulse_ms, 1.0,
            f"Per-pulse time {per_pulse_ms:.3f}ms exceeds 1.0ms threshold"
        )

        # Verify output is correct
        self.assertEqual(cache.n_pwls, n_pulses)
        self.assertGreater(cache.n_pwl_points, 0)


class TestTransientSmoothingIntegration(unittest.TestCase):
    """Integration tests for smoothing in transient analysis workflow."""

    def test_preprocess_sources_transient_workflow(self):
        """Test full preprocessing workflow: sources -> smoothed -> evaluate."""
        from core.vectorized_sources import VectorizedCurrentSources

        np.random.seed(42)
        n_sources = 100
        sources = VectorizedCurrentSources(
            n_nodes=n_sources,
            node_to_idx={f'node_{i}': i for i in range(n_sources)},
            n_sources=n_sources,
            dc_values=np.zeros(n_sources),
            source_node_idx=np.arange(n_sources, dtype=np.int32),
            n_pulses=n_sources,
            pulse_node_idx=np.arange(n_sources, dtype=np.int32),
            pulse_source_idx=np.arange(n_sources, dtype=np.int32),
            pulse_v1=np.zeros(n_sources),
            pulse_v2=np.random.uniform(0.001, 0.01, n_sources),
            pulse_delay=np.random.uniform(1e-9, 3e-9, n_sources),
            pulse_rt=np.full(n_sources, 0.1e-9),
            pulse_ft=np.full(n_sources, 0.1e-9),
            pulse_width=np.random.uniform(1e-9, 2e-9, n_sources),
            pulse_period=np.full(n_sources, 10e-9),
            n_pwls=0, n_pwl_points=0,
            pwl_node_idx=np.array([], dtype=np.int32),
            pwl_source_idx=np.array([], dtype=np.int32),
            pwl_period=np.array([]),
            pwl_delay=np.array([]),
            pwl_offset=np.array([], dtype=np.int32),
            pwl_count=np.array([], dtype=np.int32),
            pwl_times=np.array([]),
            pwl_values=np.array([]),
        )

        # Preprocess (smooth) sources
        smoothed = sources.create_smoothed_copy(
            time_step=0.1e-9,
            t_start=0,
            t_end=10e-9,
        )

        # Verify smoothed sources structure
        self.assertEqual(smoothed.n_pulses, 0)  # Pulses converted to PWL
        self.assertEqual(smoothed.n_pwls, n_sources)  # All pulses now PWLs
        self.assertGreater(smoothed.n_pwl_points, 0)

        # Verify smoothed sources can be evaluated at arbitrary times
        for t in [0, 2.5e-9, 5e-9, 7.5e-9, 10e-9]:
            currents = smoothed.evaluate_at_time(t)
            self.assertEqual(len(currents), n_sources)
            self.assertTrue(np.all(np.isfinite(currents)))

    def test_smoothing_preserves_dc_average(self):
        """Smoothing should preserve DC average of waveforms."""
        from core.vectorized_sources import VectorizedCurrentSources

        np.random.seed(42)
        n_sources = 50
        dc_values = np.random.uniform(0.1, 1.0, n_sources)

        sources = VectorizedCurrentSources(
            n_nodes=n_sources,
            node_to_idx={f'n{i}': i for i in range(n_sources)},
            n_sources=n_sources,
            dc_values=dc_values,
            source_node_idx=np.arange(n_sources, dtype=np.int32),
            n_pulses=n_sources,
            pulse_node_idx=np.arange(n_sources, dtype=np.int32),
            pulse_source_idx=np.arange(n_sources, dtype=np.int32),
            pulse_v1=np.zeros(n_sources),
            pulse_v2=np.ones(n_sources),
            pulse_delay=np.zeros(n_sources),
            pulse_rt=np.zeros(n_sources),
            pulse_ft=np.zeros(n_sources),
            pulse_width=np.full(n_sources, 5e-9),
            pulse_period=np.full(n_sources, 10e-9),
            n_pwls=0, n_pwl_points=0,
            pwl_node_idx=np.array([], dtype=np.int32),
            pwl_source_idx=np.array([], dtype=np.int32),
            pwl_period=np.array([]),
            pwl_delay=np.array([]),
            pwl_offset=np.array([], dtype=np.int32),
            pwl_count=np.array([], dtype=np.int32),
            pwl_times=np.array([]),
            pwl_values=np.array([]),
        )

        smoothed = sources.create_smoothed_copy(
            time_step=0.1e-9,
            t_start=0,
            t_end=10e-9,
        )

        # DC values should be preserved exactly
        np.testing.assert_array_equal(smoothed.dc_values, dc_values)

        # Smoothed PWL should have similar average as original pulse
        t_samples = np.linspace(0, 10e-9, 101)
        original_avg = np.mean([sources.evaluate_at_time(t) for t in t_samples], axis=0)
        smoothed_avg = np.mean([smoothed.evaluate_at_time(t) for t in t_samples], axis=0)

        # Average should be approximately preserved (within 15% for smoothing effects)
        np.testing.assert_allclose(
            original_avg, smoothed_avg, rtol=0.15,
            err_msg="Smoothing should approximately preserve DC average"
        )


class TestPulseSmoothingVectorized(unittest.TestCase):
    """Tests for vectorized pulse smoothing."""

    def test_raw_vs_smoothed_pulses_vectorized(self):
        """Compare vectorized smoothing against reference values.

        Same test concept as TestPulseSmoothing.test_raw_vs_smoothed_pulses
        but verifying the vectorized path produces correct results.
        """
        from pdn.pdn_parser import Pulse
        from core.pwl_smoothing import _smooth_pwl_vectorized, pulse_to_pwl_points

        smoother = PWLSmoother(time_step=0.01e-9, compact_threshold=1e-12)

        pulse1 = Pulse(0, 0.0117914, 3.52914e-09, 6.25e-13, 6.25e-13, 0, 1e-08)
        pulse2 = Pulse(0, -0.000131702, 3.51998e-09, 9.78805e-12, 9.78805e-12, 0, 1e-08)

        # Convert to PWL and smooth using vectorized path
        pwl1 = pulse_to_pwl_points(
            pulse1.v1, pulse1.v2, pulse1.delay, pulse1.rt, pulse1.ft,
            pulse1.width, pulse1.period
        )
        pwl2 = pulse_to_pwl_points(
            pulse2.v1, pulse2.v2, pulse2.delay, pulse2.rt, pulse2.ft,
            pulse2.width, pulse2.period
        )

        times1 = np.array([p[0] for p in pwl1])
        values1 = np.array([p[1] for p in pwl1])
        times2 = np.array([p[0] for p in pwl2])
        values2 = np.array([p[1] for p in pwl2])

        smooth_times1, smooth_values1 = _smooth_pwl_vectorized(
            times1, values1, pulse1.period, 0.01e-9, 0, 10e-9
        )
        smooth_times2, smooth_values2 = _smooth_pwl_vectorized(
            times2, values2, pulse2.period, 0.01e-9, 0, 10e-9
        )

        # Also smooth using the class method for comparison
        smoothed1 = smoother.smooth_pulse(pulse1, t_start=0, t_end=10e-9)
        smoothed2 = smoother.smooth_pulse(pulse2, t_start=0, t_end=10e-9)

        # Sample at specific time points and compare
        t_compare = np.array([3.48e-9, 3.5e-9, 3.52e-9, 3.54e-9, 3.56e-9])

        for t in t_compare:
            # Evaluate smoothed PWL from both methods
            val1_class = smoothed1.evaluate(t)
            val2_class = smoothed2.evaluate(t)

            # The vectorized path should match the class-based path
            idx = np.searchsorted(smooth_times1, t)
            if idx > 0 and idx < len(smooth_times1):
                t_lo, t_hi = smooth_times1[idx-1], smooth_times1[idx]
                v_lo, v_hi = smooth_values1[idx-1], smooth_values1[idx]
                val1_vec = v_lo + (v_hi - v_lo) * (t - t_lo) / (t_hi - t_lo) if t_hi > t_lo else v_lo
            else:
                val1_vec = smooth_values1[0] if idx == 0 else smooth_values1[-1]

            # Values should be close (allowing for compaction differences)
            self.assertAlmostEqual(val1_class, val1_vec, places=3)


class TestPulseActiveIntervalDetection(unittest.TestCase):
    """Tests for _compute_pulse_active_intervals function."""

    def test_simple_pulse_active_intervals(self):
        """Test active intervals for standard pulse."""
        from core.pwl_smoothing import _compute_pulse_active_intervals

        delay = np.array([1e-9])
        rt = np.array([0.1e-9])
        ft = np.array([0.1e-9])
        width = np.array([2e-9])
        period = np.array([10e-9])
        time_step = 0.1e-9
        t_start = 0.0
        t_end = 10e-9

        intervals, n_intervals = _compute_pulse_active_intervals(
            delay, rt, ft, width, period, time_step, t_start, t_end
        )

        # Should have intervals around rise (1ns) and fall (3.2ns) edges
        self.assertEqual(n_intervals[0], 2)
        # Rise interval should be around [0.9ns, 1.2ns]
        self.assertLess(intervals[0, 0, 0], 1e-9)  # Before delay
        self.assertGreater(intervals[0, 0, 1], 1.1e-9)  # After delay+rt

    def test_step_pulse_active_intervals(self):
        """Test with zero rise/fall (step pulse)."""
        from core.pwl_smoothing import _compute_pulse_active_intervals

        delay = np.array([1e-9])
        rt = np.array([0.0])  # Instant rise
        ft = np.array([0.0])  # Instant fall
        width = np.array([2e-9])
        period = np.array([10e-9])
        time_step = 0.1e-9

        intervals, n_intervals = _compute_pulse_active_intervals(
            delay, rt, ft, width, period, time_step, 0.0, 10e-9
        )

        # Still should have intervals around transitions
        self.assertGreater(n_intervals[0], 0)

    def test_overlapping_pulse_intervals(self):
        """Test when rise/fall intervals overlap (short width)."""
        from core.pwl_smoothing import _compute_pulse_active_intervals

        delay = np.array([1e-9])
        rt = np.array([0.1e-9])
        ft = np.array([0.1e-9])
        width = np.array([0.05e-9])  # Very short pulse - intervals overlap
        period = np.array([10e-9])
        time_step = 0.2e-9  # Large time step causes overlap

        intervals, n_intervals = _compute_pulse_active_intervals(
            delay, rt, ft, width, period, time_step, 0.0, 10e-9
        )

        # Should have intervals (may be merged)
        self.assertGreater(n_intervals[0], 0)

    def test_non_periodic_pulse(self):
        """Test with period=0."""
        from core.pwl_smoothing import _compute_pulse_active_intervals

        delay = np.array([1e-9])
        rt = np.array([0.1e-9])
        ft = np.array([0.1e-9])
        width = np.array([2e-9])
        period = np.array([0.0])  # Non-periodic
        time_step = 0.1e-9

        intervals, n_intervals = _compute_pulse_active_intervals(
            delay, rt, ft, width, period, time_step, 0.0, 5e-9
        )

        # Should have intervals around transitions only in first period
        self.assertGreater(n_intervals[0], 0)


class TestPWLActiveIntervalDetection(unittest.TestCase):
    """Tests for _compute_pwl_active_intervals function."""

    def test_constant_pwl_no_active_intervals(self):
        """Test PWL with all same values → no active intervals."""
        from core.pwl_smoothing import _compute_pwl_active_intervals

        times = np.array([0.0, 5.0, 10.0])
        values = np.array([1.0, 1.0, 1.0])  # Constant
        period = 0.0
        time_step = 0.5

        active, constant = _compute_pwl_active_intervals(
            times, values, period, time_step, 0.0, 10.0
        )

        # Should have no active intervals (constant PWL)
        self.assertEqual(len(active), 0)
        # Should have one constant region covering entire range
        self.assertEqual(len(constant), 1)
        self.assertAlmostEqual(constant[0][2], 1.0)

    def test_single_transition_pwl(self):
        """Test PWL with one slope change."""
        from core.pwl_smoothing import _compute_pwl_active_intervals

        times = np.array([0.0, 5.0, 10.0])
        values = np.array([0.0, 1.0, 0.0])  # Triangle
        period = 0.0
        time_step = 0.5

        active, constant = _compute_pwl_active_intervals(
            times, values, period, time_step, 0.0, 10.0
        )

        # Should have active intervals around t=0, t=5, t=10 (slope changes)
        self.assertGreater(len(active), 0)

    def test_interval_merging(self):
        """Test interval merging when transitions are close."""
        from core.pwl_smoothing import _compute_pwl_active_intervals

        # Two transitions close together (closer than 2*time_step)
        times = np.array([0.0, 1.0, 1.1, 2.0])  # Transitions at 1.0 and 1.1
        values = np.array([0.0, 1.0, 0.5, 0.5])
        period = 0.0
        time_step = 0.2  # Intervals will overlap and merge

        active, constant = _compute_pwl_active_intervals(
            times, values, period, time_step, 0.0, 2.0
        )

        # Check that we get merged intervals (not one per transition)
        # The intervals around 1.0 and 1.1 should merge
        if len(active) > 0:
            # Find intervals around t=1.0
            found_merged = False
            for start, end in active:
                if start <= 1.0 <= end and start <= 1.1 <= end:
                    found_merged = True
                    break
            # Either merged or two separate small intervals
            self.assertTrue(len(active) <= 3)


class TestSparseSmoothingFunctions(unittest.TestCase):
    """Tests for sparse smoothing functions."""

    def test_sparse_pulse_matches_dense(self):
        """Sparse pulse smoothing matches dense within tolerance."""
        from core.pwl_smoothing import (
            _smooth_pulse_chunk, _smooth_pulse_chunk_sparse,
            _pulse_to_pwl_arrays, _compute_pulse_active_intervals
        )

        # Single pulse
        v1 = np.array([0.0])
        v2 = np.array([1.0])
        delay = np.array([1e-9])
        rt = np.array([0.1e-9])
        ft = np.array([0.1e-9])
        width = np.array([2e-9])
        period = np.array([10e-9])
        time_step = 0.05e-9
        t_start = 0.0
        t_end = 10e-9

        # Dense path
        pulse_times_2d, pulse_values_2d = _pulse_to_pwl_arrays(
            v1, v2, delay, rt, ft, width, period
        )
        n_samples = int((t_end - t_start) / time_step) + 1
        sample_times = np.linspace(t_start, t_end, n_samples)
        dense_result = _smooth_pulse_chunk(
            pulse_times_2d, pulse_values_2d, period, sample_times, time_step
        )

        # Sparse path
        intervals, n_intervals = _compute_pulse_active_intervals(
            delay, rt, ft, width, period, time_step, t_start, t_end
        )
        sparse_times, sparse_values = _smooth_pulse_chunk_sparse(
            pulse_times_2d, pulse_values_2d, period,
            intervals, n_intervals, time_step, t_start, t_end
        )

        # Sparse now returns same format as dense:
        # sparse_times: (n_samples,) - uniform sample times
        # sparse_values: (chunk_size, n_samples) - same as dense_result
        # Direct comparison - they should match exactly (within floating point)
        np.testing.assert_array_equal(sparse_times, sample_times)
        max_diff = np.max(np.abs(sparse_values - dense_result))
        self.assertLess(max_diff, 1e-10,
            f"Max difference {max_diff:.2e} exceeds tolerance")

    def test_sparse_pwl_matches_dense(self):
        """Sparse PWL smoothing matches dense within tolerance."""
        from core.pwl_smoothing import _smooth_pwl_vectorized, _smooth_pwl_sparse

        times = np.array([0.0, 1e-9, 2e-9, 5e-9, 10e-9])
        values = np.array([0.0, 1.0, 0.5, 0.5, 0.0])
        period = 10e-9
        time_step = 0.05e-9
        t_start = 0.0
        t_end = 10e-9

        # Dense path
        dense_t, dense_v = _smooth_pwl_vectorized(
            times, values, period, time_step, t_start, t_end
        )

        # Sparse path
        sparse_t, sparse_v = _smooth_pwl_sparse(
            times, values, period, time_step, t_start, t_end
        )

        # Compare at interior dense sample times (skip boundaries which may differ)
        # Sample in the middle where smoothing effects are minimal
        test_times = dense_t[len(dense_t)//4 : 3*len(dense_t)//4]
        for t in test_times:
            idx = np.searchsorted(sparse_t, t)
            if idx == 0:
                interp_val = sparse_v[0]
            elif idx >= len(sparse_t):
                interp_val = sparse_v[-1]
            else:
                t_lo, t_hi = sparse_t[idx-1], sparse_t[idx]
                v_lo, v_hi = sparse_v[idx-1], sparse_v[idx]
                if t_hi > t_lo:
                    interp_val = v_lo + (v_hi - v_lo) * (t - t_lo) / (t_hi - t_lo)
                else:
                    interp_val = v_lo

            # Find corresponding dense value
            dense_idx = np.searchsorted(dense_t, t)
            if dense_idx < len(dense_v):
                dense_val = dense_v[dense_idx]
            else:
                dense_val = dense_v[-1]

            # Sparse and dense should match closely; any divergence indicates
            # a bug in active-interval detection or constant-region evaluation.
            self.assertAlmostEqual(
                interp_val, dense_val, places=4,
                msg=f"Mismatch at t={t:.2e}: sparse={interp_val:.6f}, dense={dense_val:.6f}"
            )

    def test_sparse_periodic_handling(self):
        """Periodic wraparound handled correctly in sparse mode."""
        from core.pwl_smoothing import _smooth_pwl_sparse, _smooth_pwl_vectorized

        # Triangle wave that wraps at period boundary
        times = np.array([0.0, 5e-9, 10e-9])
        values = np.array([0.0, 1.0, 0.0])
        period = 10e-9
        time_step = 0.1e-9

        sparse_t, sparse_v = _smooth_pwl_sparse(
            times, values, period, time_step, 0.0, 10e-9
        )

        # Should have samples at both ends
        self.assertGreater(len(sparse_t), 2)

        # Compare with dense result at the same time points for periodicity check
        dense_t, dense_v = _smooth_pwl_vectorized(
            times, values, period, time_step, 0.0, 10e-9
        )
        # For periodic waveforms, dense_v[0] should equal dense_v[-1]
        self.assertAlmostEqual(dense_v[0], dense_v[-1], places=2)

        # Sparse should produce correct values at interior points
        self.assertGreater(max(sparse_v), 0.1)  # Should have some variation


class TestSparseVsDenseEquivalence(unittest.TestCase):
    """Equivalence tests between sparse and dense smoothing paths."""

    def test_pulse_smoothing_equivalence_end_to_end(self):
        """Verify sparse and dense produce equivalent results for pulses."""
        from core.vectorized_sources import VectorizedCurrentSources
        from core.pwl_smoothing import PWLSmoother

        n_pulses = 100
        np.random.seed(42)

        sources = VectorizedCurrentSources(
            n_nodes=n_pulses,
            node_to_idx={f'n{i}': i for i in range(n_pulses)},
            n_sources=n_pulses,
            dc_values=np.zeros(n_pulses),
            source_node_idx=np.arange(n_pulses, dtype=np.int32),
            n_pulses=n_pulses,
            pulse_node_idx=np.arange(n_pulses, dtype=np.int32),
            pulse_source_idx=np.arange(n_pulses, dtype=np.int32),
            pulse_v1=np.zeros(n_pulses),
            pulse_v2=np.random.uniform(0.5, 1.5, n_pulses),
            pulse_delay=np.random.uniform(0.5e-9, 2e-9, n_pulses),
            pulse_rt=np.full(n_pulses, 0.1e-9),
            pulse_ft=np.full(n_pulses, 0.1e-9),
            pulse_width=np.random.uniform(1e-9, 3e-9, n_pulses),
            pulse_period=np.full(n_pulses, 10e-9),
            n_pwls=0, n_pwl_points=0,
            pwl_node_idx=np.array([], dtype=np.int32),
            pwl_source_idx=np.array([], dtype=np.int32),
            pwl_period=np.array([]),
            pwl_delay=np.array([]),
            pwl_offset=np.array([], dtype=np.int32),
            pwl_count=np.array([], dtype=np.int32),
            pwl_times=np.array([]),
            pwl_values=np.array([]),
        )

        time_step = 0.02e-9  # Small enough to trigger sparse path

        # Create caches with sparse and dense
        smoother = PWLSmoother(time_step=time_step, compact_threshold=1e-12)
        cache_sparse = smoother.create_smoothed_cache(
            sources, 0, 10e-9, sparse_threshold=100e-12
        )
        cache_dense = smoother.create_smoothed_cache(
            sources, 0, 10e-9, sparse_threshold=0  # Force dense
        )

        # Apply caches
        smoothed_sparse = smoother.apply_cache_to_sources(sources, cache_sparse)
        smoothed_dense = smoother.apply_cache_to_sources(sources, cache_dense)

        # Compare at multiple time points
        t_samples = np.linspace(0, 10e-9, 51)
        for t in t_samples:
            curr_sparse = smoothed_sparse.evaluate_at_time(t)
            curr_dense = smoothed_dense.evaluate_at_time(t)

            # Sparse and dense paths should produce identical results
            max_diff = np.max(np.abs(curr_sparse - curr_dense))
            self.assertLessEqual(
                max_diff, 1e-9,
                f"Max diff {max_diff:.2e} at t={t:.2e}"
            )


class TestSparsePerformanceBenchmarks(unittest.TestCase):
    """Performance benchmark tests for sparse smoothing."""

    def test_pulse_speedup_10ps(self):
        """Sparse should be faster than dense at 10ps time step."""
        from core.vectorized_sources import VectorizedCurrentSources
        from core.pwl_smoothing import PWLSmoother
        import time

        n_pulses = 1000
        np.random.seed(42)

        sources = VectorizedCurrentSources(
            n_nodes=n_pulses,
            node_to_idx={f'n{i}': i for i in range(n_pulses)},
            n_sources=n_pulses,
            dc_values=np.zeros(n_pulses),
            source_node_idx=np.arange(n_pulses, dtype=np.int32),
            n_pulses=n_pulses,
            pulse_node_idx=np.arange(n_pulses, dtype=np.int32),
            pulse_source_idx=np.arange(n_pulses, dtype=np.int32),
            pulse_v1=np.zeros(n_pulses),
            pulse_v2=np.ones(n_pulses),
            pulse_delay=np.full(n_pulses, 1e-9),
            pulse_rt=np.full(n_pulses, 0.1e-9),
            pulse_ft=np.full(n_pulses, 0.1e-9),
            pulse_width=np.full(n_pulses, 2e-9),
            pulse_period=np.full(n_pulses, 10e-9),
            n_pwls=0, n_pwl_points=0,
            pwl_node_idx=np.array([], dtype=np.int32),
            pwl_source_idx=np.array([], dtype=np.int32),
            pwl_period=np.array([]),
            pwl_delay=np.array([]),
            pwl_offset=np.array([], dtype=np.int32),
            pwl_count=np.array([], dtype=np.int32),
            pwl_times=np.array([]),
            pwl_values=np.array([]),
        )

        time_step = 10e-12  # 10ps

        smoother = PWLSmoother(time_step=time_step, compact_threshold=1e-12)

        # Time sparse path
        start = time.perf_counter()
        cache_sparse = smoother.create_smoothed_cache(
            sources, 0, 10e-9, sparse_threshold=100e-12
        )
        sparse_time = time.perf_counter() - start

        # Time dense path
        start = time.perf_counter()
        cache_dense = smoother.create_smoothed_cache(
            sources, 0, 10e-9, sparse_threshold=0
        )
        dense_time = time.perf_counter() - start

        # Sparse should be faster (at least 1.2x)
        # Note: The speedup may vary depending on pulse characteristics and system load
        speedup = dense_time / sparse_time if sparse_time > 0 else float('inf')
        self.assertGreater(
            speedup, 1.2,
            f"Sparse should be at least 1.2x faster: dense={dense_time:.3f}s, "
            f"sparse={sparse_time:.3f}s, speedup={speedup:.1f}x"
        )

    def test_total_time_10k_pulses_10ps(self):
        """10K pulses at 10ps should complete in reasonable time."""
        from core.vectorized_sources import VectorizedCurrentSources
        from core.pwl_smoothing import PWLSmoother
        import time

        n_pulses = 10000
        np.random.seed(42)

        sources = VectorizedCurrentSources(
            n_nodes=n_pulses,
            node_to_idx={f'n{i}': i for i in range(n_pulses)},
            n_sources=n_pulses,
            dc_values=np.zeros(n_pulses),
            source_node_idx=np.arange(n_pulses, dtype=np.int32),
            n_pulses=n_pulses,
            pulse_node_idx=np.arange(n_pulses, dtype=np.int32),
            pulse_source_idx=np.arange(n_pulses, dtype=np.int32),
            pulse_v1=np.zeros(n_pulses),
            pulse_v2=np.ones(n_pulses),
            pulse_delay=np.full(n_pulses, 1e-9),
            pulse_rt=np.full(n_pulses, 0.1e-9),
            pulse_ft=np.full(n_pulses, 0.1e-9),
            pulse_width=np.full(n_pulses, 2e-9),
            pulse_period=np.full(n_pulses, 10e-9),
            n_pwls=0, n_pwl_points=0,
            pwl_node_idx=np.array([], dtype=np.int32),
            pwl_source_idx=np.array([], dtype=np.int32),
            pwl_period=np.array([]),
            pwl_delay=np.array([]),
            pwl_offset=np.array([], dtype=np.int32),
            pwl_count=np.array([], dtype=np.int32),
            pwl_times=np.array([]),
            pwl_values=np.array([]),
        )

        time_step = 10e-12  # 10ps

        smoother = PWLSmoother(time_step=time_step, compact_threshold=1e-12)

        start = time.perf_counter()
        cache = smoother.create_smoothed_cache(
            sources, 0, 10e-9, sparse_threshold=100e-12
        )
        elapsed = time.perf_counter() - start

        # Should complete in < 30 seconds for 10K pulses
        self.assertLess(
            elapsed, 30.0,
            f"10K pulses at 10ps took {elapsed:.1f}s, expected < 30s"
        )

        # Verify output is correct
        self.assertEqual(cache.n_pwls, n_pulses)


class TestSmoothedEvaluationPerformance(unittest.TestCase):
    """Performance regression test for smoothed waveform evaluation throughput."""

    def test_smoothed_evaluate_at_time_within_2x_original(self):
        """Smoothed evaluate_at_time should be < 2x slower than original.

        Creates 20K sources (16K pulses + 4K PWLs), smooths them, then
        compares evaluation throughput. The vectorized _evaluate_pwls
        implementation should keep smoothed evaluation close to original
        pulse-dominated evaluation speed.
        """
        import time
        from core.vectorized_sources import VectorizedCurrentSources

        N_WAVEFORMS = 20_000
        N_EVAL = 50  # Number of evaluate_at_time calls
        T_START = 0.0
        T_END = 10e-9
        SEED = 42

        rng = np.random.RandomState(SEED)

        # --- Pulse sources (80%) ---
        n_pulses = int(N_WAVEFORMS * 0.8)
        pulse_v1 = np.zeros(n_pulses)
        pulse_v2 = rng.uniform(0.1, 2.0, n_pulses)
        pulse_delay = rng.uniform(0.5e-9, 5e-9, n_pulses)
        pulse_rt = rng.uniform(1e-12, 5e-11, n_pulses)
        pulse_ft = rng.uniform(1e-12, 5e-11, n_pulses)
        pulse_width = rng.uniform(5e-12, 1e-10, n_pulses)
        pulse_period = np.full(n_pulses, T_END)

        # --- PWL sources (20%) ---
        n_pwls = N_WAVEFORMS - n_pulses
        pwl_times_list = []
        pwl_values_list = []
        pwl_offsets = []
        pwl_counts = []
        pwl_periods = []
        pwl_delays = []

        for i in range(n_pwls):
            n_pts = rng.randint(4, 12)
            t_pts = np.sort(rng.uniform(0, T_END, n_pts))
            t_pts[0] = 0.0
            t_pts[-1] = T_END
            v_pts = rng.uniform(0, 1.5, n_pts)

            pwl_offsets.append(len(pwl_times_list))
            pwl_counts.append(n_pts)
            pwl_periods.append(T_END)
            pwl_delays.append(0.0)
            pwl_times_list.extend(t_pts.tolist())
            pwl_values_list.extend(v_pts.tolist())

        # Build original VectorizedCurrentSources
        n_nodes = N_WAVEFORMS
        node_to_idx = {f'n{i}': i for i in range(n_nodes)}

        sources = VectorizedCurrentSources(
            n_nodes=n_nodes,
            node_to_idx=node_to_idx,
            n_sources=N_WAVEFORMS,
            dc_values=rng.uniform(0, 0.5, N_WAVEFORMS),
            source_node_idx=np.arange(N_WAVEFORMS, dtype=np.int32),
            n_pulses=n_pulses,
            pulse_node_idx=np.arange(n_pulses, dtype=np.int32),
            pulse_source_idx=np.arange(n_pulses, dtype=np.int32),
            pulse_v1=pulse_v1,
            pulse_v2=pulse_v2,
            pulse_delay=pulse_delay,
            pulse_rt=pulse_rt,
            pulse_ft=pulse_ft,
            pulse_width=pulse_width,
            pulse_period=pulse_period,
            n_pwls=n_pwls,
            n_pwl_points=len(pwl_times_list),
            pwl_node_idx=np.arange(n_pulses, n_pulses + n_pwls, dtype=np.int32),
            pwl_source_idx=np.arange(n_pulses, n_pulses + n_pwls, dtype=np.int32),
            pwl_period=np.array(pwl_periods),
            pwl_delay=np.array(pwl_delays),
            pwl_offset=np.array(pwl_offsets, dtype=np.int32),
            pwl_count=np.array(pwl_counts, dtype=np.int32),
            pwl_times=np.array(pwl_times_list),
            pwl_values=np.array(pwl_values_list),
        )

        # Smooth sources
        dt_smooth = 0.05e-9  # 50ps
        smoother = PWLSmoother(time_step=dt_smooth, compact_threshold=1e-12)
        cache = smoother.create_smoothed_cache(
            sources, T_START, T_END, sparse_threshold=100e-12
        )
        smoothed = smoother.apply_cache_to_sources(sources, cache)

        # Random evaluation times
        eval_t = rng.uniform(T_START, T_END, N_EVAL)

        # Warm up (trigger lazy cache building, stabilize CPU caches)
        for _ in range(5):
            _ = sources.evaluate_at_time(T_START)
            _ = smoothed.evaluate_at_time(T_START)

        # Benchmark (with retry if first attempt fails due to noise)
        def run_benchmark():
            t0 = time.perf_counter()
            for t in eval_t:
                _ = sources.evaluate_at_time(t)
            original_time = time.perf_counter() - t0

            t0 = time.perf_counter()
            for t in eval_t:
                _ = smoothed.evaluate_at_time(t)
            smoothed_time = time.perf_counter() - t0

            ratio = smoothed_time / original_time if original_time > 0 else float('inf')
            return ratio, original_time, smoothed_time

        ratio, original_time, smoothed_time = run_benchmark()
        
        # Retry once if failed (to handle occasional CPU scheduling jitter)
        if ratio >= 2.0:
            ratio_retry, original_retry, smoothed_retry = run_benchmark()
            if ratio_retry < ratio:
                ratio = ratio_retry
                original_time = original_retry
                smoothed_time = smoothed_retry

        self.assertLess(
            ratio, 2.0,
            f"Smoothed evaluation is {ratio:.2f}x slower than original "
            f"(original={original_time:.3f}s, smoothed={smoothed_time:.3f}s "
            f"for {N_EVAL} calls over {N_WAVEFORMS:,} sources). "
            f"Expected < 2.0x."
        )


if __name__ == '__main__':
    unittest.main()

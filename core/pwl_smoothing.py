"""PWL waveform smoothing using analytical triangular low-pass filter.

This module provides preprocessing of PWL and Pulse current waveforms before
transient/dynamic IR-drop analysis.

The triangular filter has:
- Window width = 2 * time_step
- Exact analytical integration (no numerical approximation)
- Compaction to remove redundant points after filtering
- Proper handling of periodic waveforms

Example usage:
    from core import PWLSmoother
    from pdn.pdn_parser import PWL, Pulse

    smoother = PWLSmoother(time_step=0.1e-9, compact_threshold=1e-12)

    # Smooth individual PWL
    pwl = PWL(points=[(0, 0), (1e-9, 1), (2e-9, 0)], period=10e-9)
    smoothed_pwl = smoother.smooth_pwl(pwl, t_start=0, t_end=10e-9)

    # Convert pulse to PWL and smooth
    pulse = Pulse(v1=0, v2=1, delay=1e-9, rt=0.1e-9, ft=0.1e-9, width=2e-9, period=10e-9)
    smoothed_from_pulse = smoother.smooth_pulse(pulse, t_start=0, t_end=10e-9)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from pdn.pdn_parser import PWL, Pulse
    from .vectorized_sources import VectorizedCurrentSources


# =============================================================================
# Core Smoothing Functions
# =============================================================================


def triangular_window(t: float, t_center: float, half_width: float) -> float:
    """Evaluate triangular window function centered at t_center.

    The triangle has:
    - Peak of 1.0 at t_center
    - Zero at t_center +/- half_width
    - Zero outside [t_center - half_width, t_center + half_width]

    Args:
        t: Time point to evaluate
        t_center: Center of triangle
        half_width: Half-width (distance from center to zero)

    Returns:
        Window value in [0, 1]
    """
    if half_width <= 0:
        return 1.0 if t == t_center else 0.0

    dist = abs(t - t_center)
    if dist >= half_width:
        return 0.0
    return 1.0 - dist / half_width


def _integrate_linear_product(
    a: float, b: float, c: float, d: float, t_lo: float, t_hi: float
) -> float:
    """Integrate (a + b*t) * (c + d*t) from t_lo to t_hi.

    Expands to: ac + (ad + bc)*t + bd*t^2
    Integral:   ac*t + (ad + bc)*t^2/2 + bd*t^3/3

    Returns:
        Definite integral value
    """
    if t_hi <= t_lo:
        return 0.0

    # Coefficients for polynomial ac + (ad+bc)*t + bd*t^2
    c0 = a * c
    c1 = a * d + b * c
    c2 = b * d

    # Evaluate at bounds: c0*t + c1*t^2/2 + c2*t^3/3
    def eval_integral(t: float) -> float:
        return c0 * t + c1 * t * t / 2.0 + c2 * t * t * t / 3.0

    return eval_integral(t_hi) - eval_integral(t_lo)


def analytical_triangle_pwl_integral(
    t_out: float,
    half_width: float,
    t1: float,
    v1: float,
    t2: float,
    v2: float,
) -> float:
    """Analytically integrate triangle(t) * pwl_segment(t) dt.

    The triangular window is centered at t_out with half_width.
    The PWL segment is linear from (t1, v1) to (t2, v2).

    The integral is computed exactly using closed-form expressions for
    the product of linear functions.

    Args:
        t_out: Center of triangular window (output sample time)
        half_width: Half-width of triangle (typically = time_step)
        t1, v1: Start of PWL segment
        t2, v2: End of PWL segment

    Returns:
        Integral value (contribution to smoothed output)
    """
    if half_width <= 0 or t2 <= t1:
        return 0.0

    # Window bounds
    w_lo = t_out - half_width
    w_hi = t_out + half_width

    # Find overlap of segment [t1, t2] with window [w_lo, w_hi]
    seg_lo = max(t1, w_lo)
    seg_hi = min(t2, w_hi)

    if seg_lo >= seg_hi:
        return 0.0  # No overlap

    # PWL segment: f(t) = v1 + m * (t - t1) = (v1 - m*t1) + m*t
    m = (v2 - v1) / (t2 - t1)
    pwl_a = v1 - m * t1  # Intercept
    pwl_b = m  # Slope

    result = 0.0

    # Left half of triangle: t in [w_lo, t_out]
    # Triangle: w(t) = (t - w_lo) / half_width = (-w_lo/h) + (1/h)*t
    left_lo = max(seg_lo, w_lo)
    left_hi = min(seg_hi, t_out)
    if left_lo < left_hi:
        tri_a = -w_lo / half_width
        tri_b = 1.0 / half_width
        result += _integrate_linear_product(tri_a, tri_b, pwl_a, pwl_b, left_lo, left_hi)

    # Right half of triangle: t in [t_out, w_hi]
    # Triangle: w(t) = (w_hi - t) / half_width = (w_hi/h) + (-1/h)*t
    right_lo = max(seg_lo, t_out)
    right_hi = min(seg_hi, w_hi)
    if right_lo < right_hi:
        tri_a = w_hi / half_width
        tri_b = -1.0 / half_width
        result += _integrate_linear_product(tri_a, tri_b, pwl_a, pwl_b, right_lo, right_hi)

    return result


# =============================================================================
# Vectorized Smoothing Functions
# =============================================================================


def _integrate_linear_product_vectorized(
    a: float,
    b: float,
    c: float,
    d: float,
    t_lo: np.ndarray,
    t_hi: np.ndarray,
) -> np.ndarray:
    """Vectorized integration of (a + b*t) * (c + d*t) from t_lo to t_hi.

    Args:
        a, b: Coefficients of first linear function (scalars)
        c, d: Coefficients of second linear function (scalars)
        t_lo: Lower bounds (n_samples,)
        t_hi: Upper bounds (n_samples,)

    Returns:
        Definite integral values (n_samples,)
    """
    # Mask for valid intervals
    valid = t_hi > t_lo

    # Coefficients for polynomial ac + (ad+bc)*t + bd*t^2
    c0 = a * c
    c1 = a * d + b * c
    c2 = b * d

    # Evaluate antiderivative: c0*t + c1*t^2/2 + c2*t^3/3
    result = np.zeros_like(t_lo)

    if np.any(valid):
        t_lo_v = t_lo[valid]
        t_hi_v = t_hi[valid]

        F_hi = c0 * t_hi_v + c1 * t_hi_v * t_hi_v / 2.0 + c2 * t_hi_v**3 / 3.0
        F_lo = c0 * t_lo_v + c1 * t_lo_v * t_lo_v / 2.0 + c2 * t_lo_v**3 / 3.0
        result[valid] = F_hi - F_lo

    return result


def _analytical_integral_vectorized(
    t_out: np.ndarray,
    half_width: float,
    t1: float,
    v1: float,
    t2: float,
    v2: float,
) -> np.ndarray:
    """Vectorized analytical integral of triangle(t) * pwl_segment(t).

    Computes the integral for multiple output sample times at once.
    Fully vectorized with no Python loops.

    Args:
        t_out: Center of triangular windows (n_samples,)
        half_width: Half-width of triangle (typically = time_step)
        t1, v1: Start of PWL segment
        t2, v2: End of PWL segment

    Returns:
        Integral values (n_samples,) - contribution to smoothed output
    """
    n_samples = len(t_out)
    result = np.zeros(n_samples, dtype=np.float64)

    if half_width <= 0 or t2 <= t1:
        return result

    # Window bounds for all samples
    w_lo = t_out - half_width
    w_hi = t_out + half_width

    # Find overlap of segment [t1, t2] with each window [w_lo, w_hi]
    seg_lo = np.maximum(t1, w_lo)
    seg_hi = np.minimum(t2, w_hi)

    # Mask for samples with overlap
    has_overlap = seg_lo < seg_hi
    if not np.any(has_overlap):
        return result

    # PWL segment: f(t) = v1 + m * (t - t1) = (v1 - m*t1) + m*t
    m = (v2 - v1) / (t2 - t1)
    pwl_a = v1 - m * t1  # Intercept
    pwl_b = m  # Slope

    # Process left half of triangle: t in [w_lo, t_out]
    # Triangle: w(t) = (t - w_lo) / half_width = (-w_lo/h) + (1/h)*t
    left_lo = np.maximum(seg_lo, w_lo)
    left_hi = np.minimum(seg_hi, t_out)
    left_valid = (left_lo < left_hi) & has_overlap

    if np.any(left_valid):
        # Triangle coefficients: tri_a = -w_lo/h, tri_b = 1/h
        # The integral is: ∫(tri_a + tri_b*t) * (pwl_a + pwl_b*t) dt
        # = ∫(tri_a*pwl_a) + (tri_a*pwl_b + tri_b*pwl_a)*t + (tri_b*pwl_b)*t^2 dt
        #
        # With tri_a = -w_lo/h varying per sample, we expand:
        # c0[i] = tri_a[i] * pwl_a = (-w_lo[i]/h) * pwl_a
        # c1[i] = tri_a[i] * pwl_b + tri_b * pwl_a = (-w_lo[i]/h) * pwl_b + (1/h) * pwl_a
        # c2 = tri_b * pwl_b = (1/h) * pwl_b  (constant)

        tri_b_left = 1.0 / half_width
        c2 = tri_b_left * pwl_b  # Constant for all samples

        # Vectorized computation for all valid left samples
        w_lo_v = w_lo[left_valid]
        tri_a_v = -w_lo_v / half_width
        c0_v = tri_a_v * pwl_a
        c1_v = tri_a_v * pwl_b + tri_b_left * pwl_a

        t_lo_v = left_lo[left_valid]
        t_hi_v = left_hi[left_valid]

        # Antiderivative: F(t) = c0*t + c1*t^2/2 + c2*t^3/3
        F_hi = c0_v * t_hi_v + c1_v * t_hi_v**2 / 2.0 + c2 * t_hi_v**3 / 3.0
        F_lo = c0_v * t_lo_v + c1_v * t_lo_v**2 / 2.0 + c2 * t_lo_v**3 / 3.0

        result[left_valid] += F_hi - F_lo

    # Process right half of triangle: t in [t_out, w_hi]
    # Triangle: w(t) = (w_hi - t) / half_width = (w_hi/h) + (-1/h)*t
    right_lo = np.maximum(seg_lo, t_out)
    right_hi = np.minimum(seg_hi, w_hi)
    right_valid = (right_lo < right_hi) & has_overlap

    if np.any(right_valid):
        # Triangle coefficients: tri_a = w_hi/h, tri_b = -1/h
        tri_b_right = -1.0 / half_width
        c2 = tri_b_right * pwl_b  # Constant for all samples

        w_hi_v = w_hi[right_valid]
        tri_a_v = w_hi_v / half_width
        c0_v = tri_a_v * pwl_a
        c1_v = tri_a_v * pwl_b + tri_b_right * pwl_a

        t_lo_v = right_lo[right_valid]
        t_hi_v = right_hi[right_valid]

        # Antiderivative: F(t) = c0*t + c1*t^2/2 + c2*t^3/3
        F_hi = c0_v * t_hi_v + c1_v * t_hi_v**2 / 2.0 + c2 * t_hi_v**3 / 3.0
        F_lo = c0_v * t_lo_v + c1_v * t_lo_v**2 / 2.0 + c2 * t_lo_v**3 / 3.0

        result[right_valid] += F_hi - F_lo

    return result


def _smooth_pwl_vectorized(
    times: np.ndarray,
    values: np.ndarray,
    period: float,
    time_step: float,
    t_start: float,
    t_end: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized smoothing for a single PWL waveform.

    Applies triangular low-pass filter using vectorized operations.

    Args:
        times: PWL time points (n_points,)
        values: PWL values (n_points,)
        period: PWL period (0 = non-periodic)
        time_step: Simulation time step (window = 2 * time_step)
        t_start: Start of output time range
        t_end: End of output time range

    Returns:
        Tuple of (smoothed_times, smoothed_values) arrays
    """
    if len(times) < 2 or time_step <= 0:
        return times.copy(), values.copy()

    half_width = time_step

    # Generate output sample times
    # For periodic waveforms, adjust to ensure integer number of periods
    if period > 0 and t_end - t_start >= period:
        steps_per_period = max(1, int(round(period / time_step)))
        actual_step = period / steps_per_period
    else:
        actual_step = time_step

    n_samples = int((t_end - t_start) / actual_step) + 1
    sample_times = np.linspace(t_start, t_end, n_samples)

    # Initialize output
    weighted_sum = np.zeros(n_samples, dtype=np.float64)

    # Process each segment
    n_points = len(times)
    for i in range(n_points - 1):
        t1, v1 = times[i], values[i]
        t2, v2 = times[i + 1], values[i + 1]

        # Direct contribution from this segment
        weighted_sum += _analytical_integral_vectorized(
            sample_times, half_width, t1, v1, t2, v2
        )

    # Handle periodic wraparound
    if period > 0:
        for i in range(n_points - 1):
            t1, v1 = times[i], values[i]
            t2, v2 = times[i + 1], values[i + 1]

            # Previous period contribution
            weighted_sum += _analytical_integral_vectorized(
                sample_times, half_width, t1 - period, v1, t2 - period, v2
            )
            # Next period contribution
            weighted_sum += _analytical_integral_vectorized(
                sample_times, half_width, t1 + period, v1, t2 + period, v2
            )

    # Normalize by triangle area (= half_width)
    smoothed_values = weighted_sum / half_width

    return sample_times, smoothed_values


# =============================================================================
# Chunked Batch Processing Functions
# =============================================================================


def _pulse_to_pwl_arrays(
    v1: np.ndarray,
    v2: np.ndarray,
    delay: np.ndarray,
    rt: np.ndarray,
    ft: np.ndarray,
    width: np.ndarray,
    period: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert batch of pulses to PWL arrays.

    Args:
        v1, v2, delay, rt, ft, width, period: Arrays (n_pulses,) of pulse parameters

    Returns:
        Tuple of (times_2d, values_2d) with shape (n_pulses, max_points)
        Padded with last value where needed.
    """
    n = len(v1)

    # Standard pulse has up to 7 points:
    # (0, v1), (delay, v1), (delay+rt, v2), (delay+rt+width, v2),
    # (delay+rt+width+ft, v1), (period, v1)
    # But we use a simplified 6-point representation that captures key transitions
    max_points = 6

    times = np.zeros((n, max_points), dtype=np.float64)
    values = np.zeros((n, max_points), dtype=np.float64)

    # Point 0: Start at v1
    times[:, 0] = 0.0
    values[:, 0] = v1

    # Point 1: End of delay (still at v1)
    times[:, 1] = delay
    values[:, 1] = v1

    # Point 2: End of rise (at v2)
    times[:, 2] = delay + rt
    values[:, 2] = v2

    # Point 3: End of high (still at v2)
    times[:, 3] = delay + rt + width
    values[:, 3] = v2

    # Point 4: End of fall (back to v1)
    times[:, 4] = delay + rt + width + ft
    values[:, 4] = v1

    # Point 5: End of period (at v1)
    # Use period if > 0, else use point 4 time
    times[:, 5] = np.where(period > 0, period, times[:, 4])
    values[:, 5] = v1

    return times, values


def _analytical_integral_batch(
    t_out: np.ndarray,
    half_width: float,
    t1: np.ndarray,
    v1: np.ndarray,
    t2: np.ndarray,
    v2: np.ndarray,
) -> np.ndarray:
    """Batch analytical integral for multiple segments at once.

    Computes integral for multiple segments, each evaluated at same sample times.

    Args:
        t_out: Sample times (n_samples,)
        half_width: Filter half-width
        t1, v1: Segment start times/values (n_segments,)
        t2, v2: Segment end times/values (n_segments,)

    Returns:
        Integrals (n_segments, n_samples)
    """
    n_samples = len(t_out)
    n_segments = len(t1)

    if half_width <= 0:
        return np.zeros((n_segments, n_samples), dtype=np.float64)

    # Broadcast to (n_segments, n_samples)
    t_out_2d = t_out[np.newaxis, :]  # (1, n_samples)
    t1_2d = t1[:, np.newaxis]  # (n_segments, 1)
    t2_2d = t2[:, np.newaxis]
    v1_2d = v1[:, np.newaxis]
    v2_2d = v2[:, np.newaxis]

    # Window bounds (n_segments, n_samples)
    w_lo = t_out_2d - half_width
    w_hi = t_out_2d + half_width

    # Segment validity mask
    valid_seg = (t2_2d > t1_2d)  # (n_segments, 1)

    # Find overlap
    seg_lo = np.maximum(t1_2d, w_lo)  # (n_segments, n_samples)
    seg_hi = np.minimum(t2_2d, w_hi)

    # Overlap mask
    has_overlap = (seg_lo < seg_hi) & valid_seg

    # PWL slope and intercept
    dt_seg = t2_2d - t1_2d
    dt_seg = np.where(dt_seg > 0, dt_seg, 1.0)  # Avoid division by zero
    m = (v2_2d - v1_2d) / dt_seg
    pwl_a = v1_2d - m * t1_2d
    pwl_b = m

    result = np.zeros((n_segments, n_samples), dtype=np.float64)

    # Left half of triangle: t in [w_lo, t_out]
    left_lo = np.maximum(seg_lo, w_lo)
    left_hi = np.minimum(seg_hi, t_out_2d)
    left_valid = (left_lo < left_hi) & has_overlap

    if np.any(left_valid):
        tri_b_left = 1.0 / half_width
        tri_a_left = -w_lo / half_width

        c0 = tri_a_left * pwl_a
        c1 = tri_a_left * pwl_b + tri_b_left * pwl_a
        c2 = tri_b_left * pwl_b

        F_hi = c0 * left_hi + c1 * left_hi**2 / 2.0 + c2 * left_hi**3 / 3.0
        F_lo = c0 * left_lo + c1 * left_lo**2 / 2.0 + c2 * left_lo**3 / 3.0

        result = np.where(left_valid, result + (F_hi - F_lo), result)

    # Right half of triangle: t in [t_out, w_hi]
    right_lo = np.maximum(seg_lo, t_out_2d)
    right_hi = np.minimum(seg_hi, w_hi)
    right_valid = (right_lo < right_hi) & has_overlap

    if np.any(right_valid):
        tri_b_right = -1.0 / half_width
        tri_a_right = w_hi / half_width

        c0 = tri_a_right * pwl_a
        c1 = tri_a_right * pwl_b + tri_b_right * pwl_a
        c2 = tri_b_right * pwl_b

        F_hi = c0 * right_hi + c1 * right_hi**2 / 2.0 + c2 * right_hi**3 / 3.0
        F_lo = c0 * right_lo + c1 * right_lo**2 / 2.0 + c2 * right_lo**3 / 3.0

        result = np.where(right_valid, result + (F_hi - F_lo), result)

    return result


def _smooth_pulse_chunk(
    pulse_times_2d: np.ndarray,
    pulse_values_2d: np.ndarray,
    periods: np.ndarray,
    sample_times: np.ndarray,
    time_step: float,
) -> np.ndarray:
    """Smooth a chunk of pulses using fully vectorized operations.

    Args:
        pulse_times_2d: PWL times (chunk_size, n_pwl_points)
        pulse_values_2d: PWL values (chunk_size, n_pwl_points)
        periods: Period for each pulse (chunk_size,)
        sample_times: Output sample times (n_samples,)
        time_step: Filter half-width

    Returns:
        Smoothed values (chunk_size, n_samples)
    """
    chunk_size = pulse_times_2d.shape[0]
    n_pwl_points = pulse_times_2d.shape[1]
    n_samples = len(sample_times)

    # Output array
    result = np.zeros((chunk_size, n_samples), dtype=np.float64)

    half_width = time_step

    # Process all segments for all waveforms using batch operations
    # Loop over segment index j (typically 5-6 segments), but vectorize over chunk_size
    for j in range(n_pwl_points - 1):
        # Extract segment j from all waveforms
        t1 = pulse_times_2d[:, j]  # (chunk_size,)
        v1 = pulse_values_2d[:, j]
        t2 = pulse_times_2d[:, j + 1]
        v2 = pulse_values_2d[:, j + 1]

        # Compute integrals for all waveforms at once
        # Shape: (chunk_size, n_samples)
        contrib = _analytical_integral_batch(sample_times, half_width, t1, v1, t2, v2)
        result += contrib

    # Handle periodic wraparound - check which waveforms need it
    periodic_mask = periods > 0
    if np.any(periodic_mask):
        periods_2d = periods[:, np.newaxis]  # (chunk_size, 1)

        for j in range(n_pwl_points - 1):
            t1 = pulse_times_2d[:, j]
            v1 = pulse_values_2d[:, j]
            t2 = pulse_times_2d[:, j + 1]
            v2 = pulse_values_2d[:, j + 1]

            # Previous period
            contrib_prev = _analytical_integral_batch(
                sample_times, half_width,
                t1 - periods, v1, t2 - periods, v2
            )
            result = np.where(periodic_mask[:, np.newaxis], result + contrib_prev, result)

            # Next period
            contrib_next = _analytical_integral_batch(
                sample_times, half_width,
                t1 + periods, v1, t2 + periods, v2
            )
            result = np.where(periodic_mask[:, np.newaxis], result + contrib_next, result)

    # Normalize
    result /= half_width

    return result


def _compact_chunk_vectorized(
    sample_times: np.ndarray,
    values_2d: np.ndarray,
    threshold: float = 1e-12,
) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray]:
    """Vectorized compaction for a chunk of smoothed waveforms.

    Args:
        sample_times: Shared sample times (n_samples,)
        values_2d: Smoothed values (chunk_size, n_samples)
        threshold: Slope change threshold for compaction

    Returns:
        Tuple of (times_list, values_list, counts) where each list has
        chunk_size elements of variable-length arrays.
    """
    chunk_size, n_samples = values_2d.shape

    if n_samples <= 2:
        # Nothing to compact
        times_list = [sample_times.copy() for _ in range(chunk_size)]
        values_list = [values_2d[i].copy() for i in range(chunk_size)]
        counts = np.full(chunk_size, n_samples, dtype=np.int32)
        return times_list, values_list, counts

    # Compute time differences (same for all waveforms)
    dt = np.diff(sample_times)

    # Compute value differences and slopes for each waveform
    dv = np.diff(values_2d, axis=1)  # (chunk_size, n_samples-1)

    # Compute slopes, avoiding division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        slopes = np.where(dt > 0, dv / dt, 0.0)  # (chunk_size, n_samples-1)

    # Compute slope changes at interior points
    slope_changes = np.abs(np.diff(slopes, axis=1))  # (chunk_size, n_samples-2)

    # Compute relative threshold
    slope_magnitude = np.abs(slopes[:, :-1]) + np.abs(slopes[:, 1:])
    with np.errstate(divide='ignore', invalid='ignore'):
        relative_changes = np.where(
            slope_magnitude > 0,
            slope_changes / slope_magnitude,
            slope_changes
        )

    # Keep mask: True for points to keep
    # Always keep first and last, keep interior if slope changes significantly
    keep_interior = (relative_changes > threshold) | (slope_changes > threshold)

    times_list = []
    values_list = []
    counts = np.zeros(chunk_size, dtype=np.int32)

    for i in range(chunk_size):
        # Build keep mask for this waveform
        keep = np.ones(n_samples, dtype=bool)
        keep[1:-1] = keep_interior[i]

        # Extract kept points
        kept_times = sample_times[keep]
        kept_values = values_2d[i, keep]

        times_list.append(kept_times)
        values_list.append(kept_values)
        counts[i] = len(kept_times)

    return times_list, values_list, counts


def _compact_and_append(
    times_list: List[np.ndarray],
    values_list: List[np.ndarray],
    counts: np.ndarray,
    periods: np.ndarray,
    node_indices: np.ndarray,
    all_times: List[float],
    all_values: List[float],
    offsets: List[int],
    out_counts: List[int],
    out_periods: List[float],
    out_delays: List[float],
    out_node_indices: List[int],
) -> None:
    """Append compacted chunk results to output lists.

    Args:
        times_list: List of compacted time arrays
        values_list: List of compacted value arrays
        counts: Number of points in each compacted waveform
        periods: Period for each waveform
        node_indices: Node index for each waveform
        all_times: Output times list (modified in-place)
        all_values: Output values list (modified in-place)
        offsets: Output offsets list (modified in-place)
        out_counts: Output counts list (modified in-place)
        out_periods: Output periods list (modified in-place)
        out_delays: Output delays list (modified in-place)
        out_node_indices: Output node indices list (modified in-place)
    """
    for i in range(len(times_list)):
        offsets.append(len(all_times))
        out_counts.append(int(counts[i]))
        out_periods.append(float(periods[i]))
        out_delays.append(0.0)  # Delay absorbed into points
        out_node_indices.append(int(node_indices[i]))

        all_times.extend(times_list[i].tolist())
        all_values.extend(values_list[i].tolist())


def smooth_pwl_points(
    points: List[Tuple[float, float]],
    period: float,
    time_step: float,
    t_start: float,
    t_end: float,
) -> List[Tuple[float, float]]:
    """Apply analytical triangular low-pass filter to PWL waveform.

    For each output sample time t (at intervals of time_step):
    - Define triangular window centered at t with half_width = time_step
    - Sum analytical integrals of triangle * pwl_segment for overlapping segments
    - Normalize by triangle area (= half_width)

    Args:
        points: Original PWL (time, value) pairs (must be sorted by time)
        period: PWL period (0 = non-periodic)
        time_step: Simulation time step (filter window = 2 * time_step)
        t_start: Start of output time range
        t_end: End of output time range

    Returns:
        Smoothed PWL points at uniform time_step intervals
    """
    if not points or len(points) < 2 or time_step <= 0:
        return list(points) if points else []

    half_width = time_step
    output_points: List[Tuple[float, float]] = []

    # Determine output sample times
    # Ensure we include both t_start and t_end
    n_samples = max(2, int((t_end - t_start) / time_step) + 1)

    # For periodic waveforms, adjust to ensure integer number of periods
    if period > 0 and t_end - t_start >= period:
        # Number of steps per period
        steps_per_period = max(1, int(round(period / time_step)))
        actual_step = period / steps_per_period
    else:
        actual_step = time_step

    # Generate output sample times
    sample_times = []
    t = t_start
    while t <= t_end + actual_step * 0.5:  # Include t_end with tolerance
        sample_times.append(t)
        t += actual_step

    # For each output sample, compute filtered value
    for t_out in sample_times:
        weighted_sum = 0.0

        # Process each segment in the original PWL
        for i in range(len(points) - 1):
            t1, v1 = points[i]
            t2, v2 = points[i + 1]

            # Direct contribution from this segment
            weighted_sum += analytical_triangle_pwl_integral(
                t_out, half_width, t1, v1, t2, v2
            )

        # Handle periodic wraparound
        if period > 0:
            # Window extends into previous/next periods
            # Shift points by +/- period and check for contributions

            # Previous period contributions
            for i in range(len(points) - 1):
                t1, v1 = points[i]
                t2, v2 = points[i + 1]
                # Shift segment to previous period
                weighted_sum += analytical_triangle_pwl_integral(
                    t_out, half_width, t1 - period, v1, t2 - period, v2
                )
                # Shift segment to next period
                weighted_sum += analytical_triangle_pwl_integral(
                    t_out, half_width, t1 + period, v1, t2 + period, v2
                )

        # Normalize by triangle area (area of unit-height triangle = half_width)
        if half_width > 0:
            smoothed_value = weighted_sum / half_width
        else:
            # Fallback: just interpolate at t_out
            smoothed_value = _interpolate_pwl(points, t_out, period)

        output_points.append((t_out, smoothed_value))

    return output_points


def _interpolate_pwl(
    points: List[Tuple[float, float]], t: float, period: float
) -> float:
    """Linearly interpolate PWL at time t."""
    if not points:
        return 0.0

    # Handle periodic wrapping
    if period > 0:
        t = t % period
        if t < 0:
            t += period

    # Before first point
    if t <= points[0][0]:
        return points[0][1]

    # After last point
    if t >= points[-1][0]:
        if period > 0:
            return points[0][1]  # Wrap to start
        return points[-1][1]

    # Binary search for interval
    lo, hi = 0, len(points) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if points[mid][0] <= t:
            lo = mid
        else:
            hi = mid

    t1, v1 = points[lo]
    t2, v2 = points[hi]

    if t2 == t1:
        return v1
    return v1 + (v2 - v1) * (t - t1) / (t2 - t1)


def compact_pwl(
    points: List[Tuple[float, float]],
    threshold: float = 1e-12,
) -> List[Tuple[float, float]]:
    """Remove redundant collinear/flat-region points from PWL.

    Reference: C++ compaction step in analytical_LP_filter

    Algorithm:
    1. Always keep first and last points
    2. For each interior point, compute slopes before and after
    3. If |slope_after - slope_before| < threshold, point is collinear -> remove

    Args:
        points: PWL (time, value) pairs
        threshold: Slope change threshold for removal

    Returns:
        Compacted PWL with redundant points removed
    """
    if len(points) <= 2:
        return list(points)

    result = [points[0]]  # Always keep first point

    for i in range(1, len(points) - 1):
        t_prev, v_prev = result[-1]
        t_curr, v_curr = points[i]
        t_next, v_next = points[i + 1]

        # Compute slopes
        dt1 = t_curr - t_prev
        dt2 = t_next - t_curr

        if dt1 > 0:
            slope_before = (v_curr - v_prev) / dt1
        else:
            slope_before = 0.0

        if dt2 > 0:
            slope_after = (v_next - v_curr) / dt2
        else:
            slope_after = 0.0

        # Check if point is needed (slope changes significantly)
        slope_change = abs(slope_after - slope_before)

        # Use relative threshold for better numerical stability
        slope_magnitude = abs(slope_before) + abs(slope_after)
        if slope_magnitude > 0:
            relative_change = slope_change / slope_magnitude
        else:
            relative_change = slope_change

        # Keep point if slope change exceeds threshold
        if relative_change > threshold or slope_change > threshold:
            result.append(points[i])

    result.append(points[-1])  # Always keep last point
    return result


# =============================================================================
# Pulse-to-PWL Conversion
# =============================================================================


def pulse_to_pwl_points(
    v1: float,
    v2: float,
    delay: float,
    rt: float,
    ft: float,
    width: float,
    period: float,
    n_periods: int = 1,
) -> List[Tuple[float, float]]:
    """Convert Pulse parameters to PWL points.

    Pulse timing:
    t=0:                      v1 (initial)
    t=delay:                  v1 (start of rise)
    t=delay+rt:               v2 (end of rise, start of high)
    t=delay+rt+width:         v2 (end of high, start of fall)
    t=delay+rt+width+ft:      v1 (end of fall)
    t=period:                 v1 (wrap)

    Args:
        v1: Initial/low value
        v2: Pulsed/high value
        delay: Delay before pulse starts
        rt: Rise time
        ft: Fall time
        width: Pulse width (high duration)
        period: Pulse period (0 = non-periodic)
        n_periods: Number of periods to generate

    Returns:
        List of (time, value) tuples representing the pulse as PWL
    """
    points: List[Tuple[float, float]] = []

    # Handle edge cases
    if period <= 0:
        n_periods = 1

    # Use tiny offset for step transitions to avoid same-time points
    # Must be larger than floating point precision (1e-15 causes issues with 5.0 + eps)
    eps = 1e-12

    for p in range(n_periods):
        base = p * period if period > 0 else 0.0

        # Start of period at v1
        if p == 0:
            points.append((0.0, v1))

        # Before rise: at v1
        t_rise_start = base + delay
        if delay > 0 and (not points or points[-1][0] < t_rise_start - eps):
            points.append((t_rise_start, v1))

        # After rise: at v2
        t_rise_end = base + delay + rt
        if rt > 0:
            # Gradual rise
            if not points or abs(points[-1][0] - t_rise_start) > eps:
                points.append((t_rise_start, v1))
            points.append((t_rise_end, v2))
        else:
            # Instantaneous rise - use tiny offset to create step
            if points and abs(points[-1][0] - t_rise_start) < eps:
                # Already have a point at this time, just update to transition
                pass
            points.append((t_rise_start + eps, v2))

        # End of high (before fall): at v2
        t_fall_start = base + delay + rt + width
        if width > 0 and t_fall_start > t_rise_end + eps:
            points.append((t_fall_start, v2))

        # After fall: at v1
        t_fall_end = base + delay + rt + width + ft
        if ft > 0:
            # Gradual fall
            if not points or abs(points[-1][0] - t_fall_start) > eps:
                points.append((t_fall_start, v2))
            points.append((t_fall_end, v1))
        else:
            # Instantaneous fall - use tiny offset to create step
            points.append((t_fall_start + eps, v1))

    # Add final point at period boundary if periodic
    if period > 0 and n_periods > 0:
        t_final = n_periods * period
        if not points or points[-1][0] < t_final - eps:
            points.append((t_final, v1))

    # Sort and remove exact duplicates
    points.sort(key=lambda p: p[0])
    cleaned: List[Tuple[float, float]] = []
    for t, v in points:
        if not cleaned or t > cleaned[-1][0] + eps / 2:
            cleaned.append((t, v))

    return cleaned


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class SmoothingConfig:
    """Configuration for PWL smoothing.

    Attributes:
        time_step: Simulation time step (window_width = 2 * time_step)
        compact_threshold: Slope change threshold for compaction (default: 1e-12)
        enabled: If False, smoothing is skipped (returns original PWL)
    """

    time_step: float
    compact_threshold: float = 1e-12
    enabled: bool = True


@dataclass
class SmoothedWaveformCache:
    """Cache of pre-smoothed waveforms for reuse across analyses.

    This cache stores smoothed PWL data in the same packed format as
    VectorizedCurrentSources, allowing efficient reuse across multiple
    transient/dynamic analyses with the same time parameters.

    Attributes:
        time_step: Time step used for smoothing
        t_start: Start time of smoothed range
        t_end: End time of smoothed range
        compact_threshold: Compaction threshold used

        pwl_times: Packed array of all PWL time points
        pwl_values: Packed array of all PWL values
        pwl_offset: Starting index for each PWL in packed arrays
        pwl_count: Number of points for each PWL
        pwl_period: Period for each PWL (0 = non-periodic)
        pwl_delay: Delay for each PWL (always 0 after smoothing)
        pwl_node_idx: Node index for each PWL

        original_n_pulses: Number of pulses converted to PWL
        original_n_pwls: Number of original PWLs
    """

    time_step: float
    t_start: float
    t_end: float
    compact_threshold: float

    # Packed PWL data
    pwl_times: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    pwl_values: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=np.float64)
    )
    pwl_offset: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int32))
    pwl_count: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int32))
    pwl_period: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=np.float64)
    )
    pwl_delay: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    pwl_node_idx: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=np.int32)
    )

    # Statistics
    original_n_pulses: int = 0
    original_n_pwls: int = 0
    n_pwls: int = 0
    n_pwl_points: int = 0

    def is_compatible(
        self, time_step: float, t_start: float, t_end: float, tol: float = 1e-12
    ) -> bool:
        """Check if cache can be reused for given parameters.

        Args:
            time_step: Simulation time step
            t_start: Simulation start time
            t_end: Simulation end time
            tol: Tolerance for floating point comparison

        Returns:
            True if cache is compatible with the given parameters
        """
        return (
            abs(self.time_step - time_step) < tol
            and self.t_start <= t_start + tol
            and self.t_end >= t_end - tol
        )


# =============================================================================
# PWLSmoother Class
# =============================================================================


class PWLSmoother:
    """PWL waveform smoother using analytical triangular low-pass filter.

    This class provides methods to smooth individual PWL objects, Pulse objects,
    or entire VectorizedCurrentSources structures before transient analysis.

    Example usage:
        from core import PWLSmoother, TransientIRDropSolver

        # Create smoother with time step
        smoother = PWLSmoother(time_step=0.1e-9, compact_threshold=1e-12)

        # Create smoothed cache from vectorized sources
        cache = smoother.create_smoothed_cache(vec_sources, t_start=0, t_end=100e-9)

        # Apply cache to get smoothed VectorizedCurrentSources
        smoothed_sources = smoother.apply_cache_to_sources(vec_sources, cache)
    """

    def __init__(
        self,
        time_step: float,
        compact_threshold: float = 1e-12,
        enabled: bool = True,
    ):
        """Initialize smoother.

        Args:
            time_step: Simulation time step (window = 2 * time_step)
            compact_threshold: Slope change threshold for compaction
            enabled: If False, smooth_* methods return input unchanged
        """
        self.config = SmoothingConfig(
            time_step=time_step,
            compact_threshold=compact_threshold,
            enabled=enabled,
        )
        self._stats: Dict[str, Any] = {}

    def smooth_pwl(
        self,
        pwl: "PWL",
        t_start: Optional[float] = None,
        t_end: Optional[float] = None,
    ) -> "PWL":
        """Smooth a single PWL object.

        Args:
            pwl: PWL object from pdn.pdn_parser
            t_start: Start time for output (default: 0 or first point)
            t_end: End time for output (default: period or last point)

        Returns:
            New PWL with smoothed points
        """
        from pdn.pdn_parser import PWL as PWLClass

        if not self.config.enabled:
            return pwl

        if not pwl.points:
            return PWLClass(points=[], period=pwl.period, delay=pwl.delay)

        # Determine time range
        if t_start is None:
            t_start = pwl.delay if pwl.delay > 0 else 0.0
        if t_end is None:
            if pwl.period > 0:
                t_end = t_start + pwl.period
            else:
                t_end = pwl.points[-1][0] if pwl.points else t_start

        # Smooth and compact
        smoothed_points = smooth_pwl_points(
            pwl.points,
            pwl.period,
            self.config.time_step,
            t_start,
            t_end,
        )
        compacted_points = compact_pwl(smoothed_points, self.config.compact_threshold)

        return PWLClass(
            points=compacted_points,
            period=pwl.period,
            delay=0.0,  # Delay is absorbed into the smoothed points
        )

    def smooth_pulse(
        self,
        pulse: "Pulse",
        t_start: Optional[float] = None,
        t_end: Optional[float] = None,
    ) -> "PWL":
        """Convert pulse to PWL and smooth.

        Args:
            pulse: Pulse object from pdn.pdn_parser
            t_start: Start time for output
            t_end: End time for output

        Returns:
            New PWL with smoothed points (converted from pulse)
        """
        from pdn.pdn_parser import PWL as PWLClass

        if not self.config.enabled:
            # Return unsmoothed PWL conversion
            points = pulse_to_pwl_points(
                pulse.v1,
                pulse.v2,
                pulse.delay,
                pulse.rt,
                pulse.ft,
                pulse.width,
                pulse.period,
                n_periods=1,
            )
            return PWLClass(points=points, period=pulse.period, delay=0.0)

        # Determine time range
        if t_start is None:
            t_start = 0.0
        if t_end is None:
            t_end = pulse.period if pulse.period > 0 else pulse.delay + pulse.rt + pulse.width + pulse.ft

        # Convert pulse to PWL points
        pwl_points = pulse_to_pwl_points(
            pulse.v1,
            pulse.v2,
            pulse.delay,
            pulse.rt,
            pulse.ft,
            pulse.width,
            pulse.period,
            n_periods=1,
        )

        # Smooth and compact
        smoothed_points = smooth_pwl_points(
            pwl_points,
            pulse.period,
            self.config.time_step,
            t_start,
            t_end,
        )
        compacted_points = compact_pwl(smoothed_points, self.config.compact_threshold)

        return PWLClass(
            points=compacted_points,
            period=pulse.period,
            delay=0.0,
        )

    def create_smoothed_cache(
        self,
        vec_sources: "VectorizedCurrentSources",
        t_start: float,
        t_end: float,
        chunk_size: int = 10000,
    ) -> SmoothedWaveformCache:
        """Create reusable cache from VectorizedCurrentSources.

        This method:
        1. Converts all pulses to PWL
        2. Smooths all PWL waveforms using chunked batch processing
        3. Packs results into cache for reuse

        Uses chunked batch processing to control memory usage while
        maintaining high performance through vectorization.

        Args:
            vec_sources: VectorizedCurrentSources instance
            t_start: Simulation start time
            t_end: Simulation end time
            chunk_size: Number of waveforms to process per chunk (default 10000).
                       Controls memory/speed tradeoff. Larger = faster but more memory.

        Returns:
            SmoothedWaveformCache with all smoothed waveforms
        """
        if not self.config.enabled:
            # Return cache with original data (no smoothing)
            return SmoothedWaveformCache(
                time_step=self.config.time_step,
                t_start=t_start,
                t_end=t_end,
                compact_threshold=self.config.compact_threshold,
                pwl_times=vec_sources.pwl_times.copy(),
                pwl_values=vec_sources.pwl_values.copy(),
                pwl_offset=vec_sources.pwl_offset.copy(),
                pwl_count=vec_sources.pwl_count.copy(),
                pwl_period=vec_sources.pwl_period.copy(),
                pwl_delay=vec_sources.pwl_delay.copy(),
                pwl_node_idx=vec_sources.pwl_node_idx.copy(),
                original_n_pulses=vec_sources.n_pulses,
                original_n_pwls=vec_sources.n_pwls,
                n_pwls=vec_sources.n_pwls,
                n_pwl_points=vec_sources.n_pwl_points,
            )

        # Pre-compute shared sample times
        time_step = self.config.time_step
        n_samples = int((t_end - t_start) / time_step) + 1
        sample_times = np.linspace(t_start, t_end, n_samples)

        # Collect smoothed PWL data
        all_times: List[float] = []
        all_values: List[float] = []
        offsets: List[int] = []
        counts: List[int] = []
        periods: List[float] = []
        delays: List[float] = []
        node_indices: List[int] = []

        # Process original PWLs using vectorized smoothing
        for i in range(vec_sources.n_pwls):
            offset = int(vec_sources.pwl_offset[i])
            count = int(vec_sources.pwl_count[i])
            period = float(vec_sources.pwl_period[i])
            delay = float(vec_sources.pwl_delay[i])
            node_idx = int(vec_sources.pwl_node_idx[i])

            # Extract original points as arrays
            pwl_times = vec_sources.pwl_times[offset : offset + count].copy()
            pwl_values = vec_sources.pwl_values[offset : offset + count].copy()

            # Adjust for delay
            if delay > 0:
                pwl_times = pwl_times + delay

            # Smooth using vectorized function
            smoothed_times, smoothed_values = _smooth_pwl_vectorized(
                pwl_times, pwl_values, period, time_step, t_start, t_end
            )

            # Compact
            compacted = compact_pwl(
                list(zip(smoothed_times, smoothed_values)),
                self.config.compact_threshold
            )

            # Store results
            offsets.append(len(all_times))
            counts.append(len(compacted))
            periods.append(period)
            delays.append(0.0)  # Delay absorbed into points
            node_indices.append(node_idx)

            for t, v in compacted:
                all_times.append(t)
                all_values.append(v)

        # Process pulses in chunks using batch processing
        n_pulses = vec_sources.n_pulses
        if n_pulses > 0:
            for chunk_start in range(0, n_pulses, chunk_size):
                chunk_end = min(chunk_start + chunk_size, n_pulses)
                chunk_n = chunk_end - chunk_start

                # Extract chunk data
                chunk_v1 = vec_sources.pulse_v1[chunk_start:chunk_end]
                chunk_v2 = vec_sources.pulse_v2[chunk_start:chunk_end]
                chunk_delay = vec_sources.pulse_delay[chunk_start:chunk_end]
                chunk_rt = vec_sources.pulse_rt[chunk_start:chunk_end]
                chunk_ft = vec_sources.pulse_ft[chunk_start:chunk_end]
                chunk_width = vec_sources.pulse_width[chunk_start:chunk_end]
                chunk_period = vec_sources.pulse_period[chunk_start:chunk_end]
                chunk_node_idx = vec_sources.pulse_node_idx[chunk_start:chunk_end]

                # Convert pulses to PWL arrays
                pulse_times_2d, pulse_values_2d = _pulse_to_pwl_arrays(
                    chunk_v1, chunk_v2, chunk_delay, chunk_rt,
                    chunk_ft, chunk_width, chunk_period
                )

                # Smooth chunk
                smoothed_values_2d = _smooth_pulse_chunk(
                    pulse_times_2d, pulse_values_2d, chunk_period,
                    sample_times, self.config.time_step
                )

                # Compact chunk
                times_list, values_list, chunk_counts = _compact_chunk_vectorized(
                    sample_times, smoothed_values_2d, self.config.compact_threshold
                )

                # Append to output
                _compact_and_append(
                    times_list, values_list, chunk_counts,
                    chunk_period, chunk_node_idx,
                    all_times, all_values, offsets, counts,
                    periods, delays, node_indices
                )

        # Update stats
        self._stats = {
            "original_pwls": vec_sources.n_pwls,
            "original_pulses": vec_sources.n_pulses,
            "total_smoothed": len(offsets),
            "original_points": vec_sources.n_pwl_points,
            "smoothed_points": len(all_times),
        }

        return SmoothedWaveformCache(
            time_step=self.config.time_step,
            t_start=t_start,
            t_end=t_end,
            compact_threshold=self.config.compact_threshold,
            pwl_times=np.array(all_times, dtype=np.float64),
            pwl_values=np.array(all_values, dtype=np.float64),
            pwl_offset=np.array(offsets, dtype=np.int32),
            pwl_count=np.array(counts, dtype=np.int32),
            pwl_period=np.array(periods, dtype=np.float64),
            pwl_delay=np.array(delays, dtype=np.float64),
            pwl_node_idx=np.array(node_indices, dtype=np.int32),
            original_n_pulses=vec_sources.n_pulses,
            original_n_pwls=vec_sources.n_pwls,
            n_pwls=len(offsets),
            n_pwl_points=len(all_times),
        )

    def apply_cache_to_sources(
        self,
        vec_sources: "VectorizedCurrentSources",
        cache: SmoothedWaveformCache,
    ) -> "VectorizedCurrentSources":
        """Apply cached smoothed waveforms to sources.

        Creates a new VectorizedCurrentSources with:
        - Pulses zeroed out (their contribution moved to PWL)
        - PWL arrays replaced with smoothed versions from cache

        Args:
            vec_sources: Original VectorizedCurrentSources
            cache: SmoothedWaveformCache from create_smoothed_cache

        Returns:
            New VectorizedCurrentSources with smoothed waveforms
        """
        from .vectorized_sources import VectorizedCurrentSources

        # Create new instance with same DC values but smoothed PWL/pulses
        smoothed = VectorizedCurrentSources(
            n_nodes=vec_sources.n_nodes,
            node_to_idx=vec_sources.node_to_idx,
            n_sources=vec_sources.n_sources,
            dc_values=vec_sources.dc_values.copy(),
            source_node_idx=vec_sources.source_node_idx.copy(),
            # Zero out pulses (their contribution is now in PWL)
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
            # Use smoothed PWL data from cache
            n_pwls=cache.n_pwls,
            n_pwl_points=cache.n_pwl_points,
            pwl_node_idx=cache.pwl_node_idx.copy(),
            pwl_source_idx=np.zeros(cache.n_pwls, dtype=np.int32),  # Not needed for evaluation
            pwl_period=cache.pwl_period.copy(),
            pwl_delay=cache.pwl_delay.copy(),
            pwl_offset=cache.pwl_offset.copy(),
            pwl_count=cache.pwl_count.copy(),
            pwl_times=cache.pwl_times.copy(),
            pwl_values=cache.pwl_values.copy(),
        )

        # Clear any cached data that depends on PWL structure
        smoothed._pwl_groups = None

        return smoothed

    def get_statistics(self) -> Dict[str, Any]:
        """Return smoothing statistics from last operation.

        Returns:
            Dict with keys: original_pwls, original_pulses, total_smoothed,
                          original_points, smoothed_points
        """
        return dict(self._stats)

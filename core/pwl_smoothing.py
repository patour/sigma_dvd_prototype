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
# Sparse Sampling Functions (Optimized for small time_step)
# =============================================================================


def _compute_pulse_active_intervals(
    delay: np.ndarray,
    rt: np.ndarray,
    ft: np.ndarray,
    width: np.ndarray,
    period: np.ndarray,
    time_step: float,
    t_start: float,
    t_end: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute active intervals for each pulse where smoothed output varies.

    Active intervals are regions within ±time_step of transitions:
    - Rising edge: [delay - time_step, delay + rt + time_step]
    - Falling edge: [delay+rt+width - time_step, delay+rt+width+ft + time_step]

    For periodic pulses, this also considers transitions from adjacent periods
    that fall within the [t_start, t_end] window.

    Args:
        delay: Pulse delays (n_pulses,)
        rt: Rise times (n_pulses,)
        ft: Fall times (n_pulses,)
        width: Pulse widths (n_pulses,)
        period: Pulse periods (n_pulses,), 0 = non-periodic
        time_step: Simulation time step
        t_start: Start of output time range
        t_end: End of output time range

    Returns:
        intervals: (n_pulses, max_intervals, 2) array of [start, end] pairs.
                   Invalid intervals are marked with start=end=0.
        n_intervals: (n_pulses,) count of valid intervals per pulse
    """
    n_pulses = len(delay)
    h = time_step  # Half-width of triangle filter

    # Defensive check: this function only handles ±1 period offset from the
    # base transition times.  If the simulation window spans more than 2 periods,
    # transitions in intermediate periods would be missed and those samples
    # would receive un-smoothed PWL values instead of correctly smoothed ones.
    periodic_mask = period > 0
    if np.any(periodic_mask):
        max_periods_spanned = np.max(
            (t_end - t_start) / period[periodic_mask]
        )
        if max_periods_spanned > 2.0 + 1e-9:
            raise ValueError(
                f"Sparse active-interval detection only supports simulation windows "
                f"up to 2 periods wide, but window spans {max_periods_spanned:.1f} "
                f"periods. Use dense smoothing (sparse_threshold=0) for longer windows."
            )

    # Maximum possible intervals per pulse:
    # - 2 transitions per period (rise and fall)
    # - Up to 3 periods worth of intervals (current, previous wrap, next wrap)
    # Allocate for worst case: 6 intervals per pulse
    max_intervals = 6
    intervals = np.zeros((n_pulses, max_intervals, 2), dtype=np.float64)
    n_intervals = np.zeros(n_pulses, dtype=np.int32)

    # Compute rise edge intervals for base period
    rise_start = delay - h
    rise_end = delay + rt + h

    # Compute fall edge intervals for base period
    fall_start = delay + rt + width - h
    fall_end = delay + rt + width + ft + h

    for i in range(n_pulses):
        idx = 0
        p = period[i]

        # Function to add interval if it overlaps [t_start, t_end]
        def add_interval(start: float, end: float) -> int:
            nonlocal idx
            # Clip to [t_start, t_end]
            clipped_start = max(start, t_start)
            clipped_end = min(end, t_end)
            if clipped_start < clipped_end and idx < max_intervals:
                intervals[i, idx, 0] = clipped_start
                intervals[i, idx, 1] = clipped_end
                idx += 1
            return idx

        # Add base period intervals
        add_interval(rise_start[i], rise_end[i])
        add_interval(fall_start[i], fall_end[i])

        # Handle periodic wraparound
        if p > 0:
            # How many periods fit in [t_start, t_end]?
            # Check previous period contribution (for samples near t_start)
            if rise_start[i] + p <= t_end:
                add_interval(rise_start[i] + p, rise_end[i] + p)
            if fall_start[i] + p <= t_end:
                add_interval(fall_start[i] + p, fall_end[i] + p)

            # Check next period contribution (for samples near t_end)
            # This handles the case where t_out is near t_end and the filter
            # window extends into the next period
            if rise_start[i] - p >= t_start - h:
                add_interval(rise_start[i] - p, rise_end[i] - p)
            if fall_start[i] - p >= t_start - h:
                add_interval(fall_start[i] - p, fall_end[i] - p)

        n_intervals[i] = idx

    return intervals, n_intervals


def _compute_pwl_active_intervals(
    times: np.ndarray,
    values: np.ndarray,
    period: float,
    time_step: float,
    t_start: float,
    t_end: float,
) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float, float]]]:
    """Compute active intervals for PWL where smoothed output varies.

    Active intervals are regions within ±time_step of any point where
    the slope changes (i.e., at each PWL vertex). Between these intervals,
    the smoothed output is constant (equal to the local PWL value).

    Args:
        times: PWL time points (n_points,) sorted
        values: PWL values (n_points,)
        period: PWL period (0 = non-periodic)
        time_step: Simulation time step
        t_start: Start of output time range
        t_end: End of output time range

    Returns:
        active_intervals: List of (start, end) tuples where output varies
        constant_regions: List of (start, end, value) tuples for constant regions
    """
    if len(times) < 2:
        # Single point or empty - entire range is constant
        val = values[0] if len(values) > 0 else 0.0
        return [], [(t_start, t_end, val)]

    h = time_step

    n_points = len(times)

    # Check if all values are the same (constant PWL)
    val_range = np.max(values) - np.min(values)
    if val_range < 1e-15:
        # Constant PWL - no active intervals
        return [], [(t_start, t_end, values[0])]

    # Compute slopes between consecutive points
    slopes = np.zeros(n_points - 1)
    for i in range(n_points - 1):
        dt = times[i + 1] - times[i]
        if dt > 0:
            slopes[i] = (values[i + 1] - values[i]) / dt
        else:
            slopes[i] = 0.0

    # Find slope change points
    transition_times: List[float] = []

    # For non-periodic: add first point if slope is non-zero
    if period <= 0 and len(slopes) > 0 and abs(slopes[0]) > 1e-15:
        transition_times.append(times[0])

    # Interior points where slope changes
    for i in range(1, n_points - 1):
        if abs(slopes[i] - slopes[i - 1]) > 1e-15:
            transition_times.append(times[i])

    # For non-periodic: add last point if last slope is non-zero
    if period <= 0 and len(slopes) > 0 and abs(slopes[-1]) > 1e-15:
        transition_times.append(times[-1])

    # For periodic: check slope changes at wraparound
    if period > 0 and n_points > 1:
        dt_wrap = period - times[-1] + times[0]
        if dt_wrap > 0:
            slope_wrap = (values[0] - values[-1]) / dt_wrap
            # Check if slope changes at times[0]
            if abs(slopes[0] - slope_wrap) > 1e-15:
                transition_times.append(times[0])
            # Check if slope changes at times[-1]
            if abs(slope_wrap - slopes[-1]) > 1e-15:
                transition_times.append(times[-1])
        else:
            # dt_wrap <= 0: times[-1] coincides with times[0]+period (no gap
            # segment), but the slope may still change at the boundary.
            if abs(slopes[0] - slopes[-1]) > 1e-15:
                transition_times.append(times[0])
                transition_times.append(times[-1])

    # If no transitions found, entire range is constant
    if not transition_times:
        t_mid = (t_start + t_end) / 2
        val = _interpolate_pwl_vectorized(times, values, period, t_mid)
        return [], [(t_start, t_end, val)]

    # Create active intervals around each transition point
    active_intervals: List[Tuple[float, float]] = []

    for t_trans in transition_times:
        int_start = max(t_trans - h, t_start)
        int_end = min(t_trans + h, t_end)
        if int_start < int_end:
            active_intervals.append((int_start, int_end))

    # Handle periodic contributions from adjacent periods
    if period > 0:
        for t_trans in transition_times:
            # Previous period
            t_prev = t_trans - period
            int_start = max(t_prev - h, t_start)
            int_end = min(t_prev + h, t_end)
            if int_start < int_end:
                active_intervals.append((int_start, int_end))

            # Next period
            t_next = t_trans + period
            int_start = max(t_next - h, t_start)
            int_end = min(t_next + h, t_end)
            if int_start < int_end:
                active_intervals.append((int_start, int_end))

    # Sort and merge overlapping intervals
    if active_intervals:
        active_intervals.sort(key=lambda x: x[0])
        merged: List[Tuple[float, float]] = []
        current_start, current_end = active_intervals[0]
        for start, end in active_intervals[1:]:
            if start <= current_end:
                current_end = max(current_end, end)
            else:
                merged.append((current_start, current_end))
                current_start, current_end = start, end
        merged.append((current_start, current_end))
        active_intervals = merged

    # Compute constant regions (gaps between active intervals)
    constant_regions: List[Tuple[float, float, float]] = []

    if not active_intervals:
        t_mid = (t_start + t_end) / 2
        val = _interpolate_pwl_vectorized(times, values, period, t_mid)
        constant_regions.append((t_start, t_end, val))
    else:
        # Before first active interval
        if active_intervals[0][0] > t_start:
            t_mid = (t_start + active_intervals[0][0]) / 2
            val = _interpolate_pwl_vectorized(times, values, period, t_mid)
            constant_regions.append((t_start, active_intervals[0][0], val))

        # Between active intervals
        for i in range(len(active_intervals) - 1):
            gap_start = active_intervals[i][1]
            gap_end = active_intervals[i + 1][0]
            if gap_end > gap_start:
                t_mid = (gap_start + gap_end) / 2
                val = _interpolate_pwl_vectorized(times, values, period, t_mid)
                constant_regions.append((gap_start, gap_end, val))

        # After last active interval
        if active_intervals[-1][1] < t_end:
            t_mid = (active_intervals[-1][1] + t_end) / 2
            val = _interpolate_pwl_vectorized(times, values, period, t_mid)
            constant_regions.append((active_intervals[-1][1], t_end, val))

    return active_intervals, constant_regions


def _interpolate_pwl_vectorized(
    times: np.ndarray,
    values: np.ndarray,
    period: float,
    t: float,
) -> float:
    """Interpolate PWL at time t (vectorized-friendly version)."""
    if len(times) == 0:
        return 0.0

    # Handle periodic wrapping
    if period > 0:
        t = t % period
        if t < 0:
            t += period

    # Before first point
    if t <= times[0]:
        return values[0]

    # After last point
    if t >= times[-1]:
        if period > 0:
            return values[0]  # Wrap to start
        return values[-1]

    # Binary search for interval
    idx = np.searchsorted(times, t, side='right') - 1
    idx = max(0, min(idx, len(times) - 2))

    t1, v1 = times[idx], values[idx]
    t2, v2 = times[idx + 1], values[idx + 1]

    if t2 == t1:
        return v1
    return v1 + (v2 - v1) * (t - t1) / (t2 - t1)


def _smooth_pwl_sparse(
    times: np.ndarray,
    values: np.ndarray,
    period: float,
    time_step: float,
    t_start: float,
    t_end: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Smooth PWL using sparse sampling within active intervals.

    This function computes the smoothed PWL by:
    1. Identifying active intervals (where the output varies)
    2. Sampling densely only within active intervals
    3. Computing smoothed boundary values for constant regions

    This provides significant speedup when time_step is small relative to
    the waveform period, as most of the output is constant between transitions.

    Args:
        times: PWL time points (n_points,) sorted
        values: PWL values (n_points,)
        period: PWL period (0 = non-periodic)
        time_step: Simulation time step
        t_start: Start of output time range
        t_end: End of output time range

    Returns:
        Tuple of (smoothed_times, smoothed_values) arrays
    """
    if len(times) < 2 or time_step <= 0:
        return times.copy(), values.copy()

    # Get active intervals
    active_intervals, constant_regions = _compute_pwl_active_intervals(
        times, values, period, time_step, t_start, t_end
    )

    half_width = time_step
    n_points = len(times)

    def compute_smoothed_value(t_out: float) -> float:
        """Compute smoothed value at a single time point."""
        weighted_sum = 0.0
        for i in range(n_points - 1):
            t1, v1_seg = times[i], values[i]
            t2, v2_seg = times[i + 1], values[i + 1]
            weighted_sum += analytical_triangle_pwl_integral(
                t_out, half_width, t1, v1_seg, t2, v2_seg
            )

        if period > 0:
            for i in range(n_points - 1):
                t1, v1_seg = times[i], values[i]
                t2, v2_seg = times[i + 1], values[i + 1]
                weighted_sum += analytical_triangle_pwl_integral(
                    t_out, half_width, t1 - period, v1_seg, t2 - period, v2_seg
                )
                weighted_sum += analytical_triangle_pwl_integral(
                    t_out, half_width, t1 + period, v1_seg, t2 + period, v2_seg
                )

        return weighted_sum / half_width if half_width > 0 else 0.0

    # If no active intervals, compute value at a few key points
    if not active_intervals:
        v_start = compute_smoothed_value(t_start)
        v_end = compute_smoothed_value(t_end)
        return np.array([t_start, t_end]), np.array([v_start, v_end])

    # Build sparse output
    out_times: List[float] = []
    out_values: List[float] = []

    # Always start with t_start
    current_t = t_start

    for i, (a_start, a_end) in enumerate(active_intervals):
        # Add points for constant region before this active interval
        if current_t < a_start:
            # Compute smoothed value at boundaries of constant region
            if not out_times or out_times[-1] < current_t:
                out_times.append(current_t)
                out_values.append(compute_smoothed_value(current_t))
            out_times.append(a_start)
            out_values.append(compute_smoothed_value(a_start))

        # Dense samples within active interval
        # Use round() instead of int() to avoid floating-point truncation
        # (e.g. int(1.9999998) = 1 instead of the correct 2)
        n_samples = max(2, round((a_end - a_start) / time_step) + 1)
        sample_times = np.linspace(a_start, a_end, n_samples)

        # Compute smoothed values using vectorized function
        weighted_sum = np.zeros(n_samples, dtype=np.float64)
        for j in range(n_points - 1):
            t1, v1_seg = times[j], values[j]
            t2, v2_seg = times[j + 1], values[j + 1]
            weighted_sum += _analytical_integral_vectorized(
                sample_times, half_width, t1, v1_seg, t2, v2_seg
            )

        if period > 0:
            for j in range(n_points - 1):
                t1, v1_seg = times[j], values[j]
                t2, v2_seg = times[j + 1], values[j + 1]
                weighted_sum += _analytical_integral_vectorized(
                    sample_times, half_width, t1 - period, v1_seg, t2 - period, v2_seg
                )
                weighted_sum += _analytical_integral_vectorized(
                    sample_times, half_width, t1 + period, v1_seg, t2 + period, v2_seg
                )

        smoothed = weighted_sum / half_width

        # Add to output (handle overlap with previous points)
        start_idx = 0
        if out_times and abs(out_times[-1] - sample_times[0]) < 1e-15:
            out_values[-1] = smoothed[0]
            start_idx = 1

        out_times.extend(sample_times[start_idx:].tolist())
        out_values.extend(smoothed[start_idx:].tolist())

        current_t = a_end

    # Handle constant region after last active interval
    if current_t < t_end:
        if not out_times or out_times[-1] < current_t:
            out_times.append(current_t)
            out_values.append(compute_smoothed_value(current_t))
        out_times.append(t_end)
        out_values.append(compute_smoothed_value(t_end))

    # Ensure we have at least 2 points
    if len(out_times) < 2:
        v_start = compute_smoothed_value(t_start)
        v_end = compute_smoothed_value(t_end)
        return np.array([t_start, t_end]), np.array([v_start, v_end])

    return np.array(out_times, dtype=np.float64), np.array(out_values, dtype=np.float64)


def _smooth_pulse_chunk_sparse(
    pulse_times_2d: np.ndarray,
    pulse_values_2d: np.ndarray,
    periods: np.ndarray,
    intervals: np.ndarray,
    n_intervals: np.ndarray,
    time_step: float,
    t_start: float,
    t_end: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Smooth pulses using sparse computation within active intervals only.

    This function produces the SAME output format as _smooth_pulse_chunk
    (uniform sample times), but only computes smoothed values within active
    intervals. Values outside active intervals are set to the PWL value at
    that time (which equals the smoothed value when far from transitions).

    This ensures perfect numerical equivalence with the dense path while
    achieving speedup by skipping computation outside active intervals.

    Args:
        pulse_times_2d: PWL times (chunk_size, n_pwl_points)
        pulse_values_2d: PWL values (chunk_size, n_pwl_points)
        periods: Period for each pulse (chunk_size,)
        intervals: Pre-computed active intervals (chunk_size, max_intervals, 2)
        n_intervals: Number of valid intervals per pulse (chunk_size,)
        time_step: Filter half-width
        t_start: Start of output time range
        t_end: End of output time range

    Returns:
        sample_times: Uniform sample times (n_samples,) - same as dense path
        result: Smoothed values (chunk_size, n_samples) - same format as dense path
    """
    chunk_size = pulse_times_2d.shape[0]
    n_pwl_points = pulse_times_2d.shape[1]
    half_width = time_step

    # Generate the same uniform sample times as the dense path
    n_samples = int((t_end - t_start) / time_step) + 1
    sample_times = np.linspace(t_start, t_end, n_samples)

    # Initialize result using vectorized constant region assignment
    # For pulses, there are only 2 values: v1 (low) and v2 (high)
    # The PWL structure is: [0, delay, delay+rt, delay+rt+width, delay+rt+width+ft, period]
    #                       [v1,   v1,      v2,              v2,                v1,    v1]
    # So the "high" region is [delay+rt, delay+rt+width]
    result = _init_pulse_values_vectorized(
        sample_times, pulse_times_2d, pulse_values_2d, periods
    )

    # Process active samples for all pulses together
    _compute_active_samples_vectorized(
        result, sample_times, pulse_times_2d, pulse_values_2d,
        periods, intervals, n_intervals, half_width
    )

    return sample_times, result


def _init_pulse_values_vectorized(
    sample_times: np.ndarray,
    pulse_times_2d: np.ndarray,
    pulse_values_2d: np.ndarray,
    periods: np.ndarray,
) -> np.ndarray:
    """Initialize result array with PWL values using vectorized operations.

    For pulse waveforms, the structure is:
    - times: [0, delay, delay+rt, delay+rt+width, delay+rt+width+ft, period]
    - values: [v1, v1, v2, v2, v1, v1]

    Outside transitions, samples are either v1 or v2 depending on time.

    Args:
        sample_times: Uniform sample times (n_samples,)
        pulse_times_2d: PWL times (chunk_size, 6)
        pulse_values_2d: PWL values (chunk_size, 6)
        periods: Period for each pulse (chunk_size,)

    Returns:
        result: Initialized values (chunk_size, n_samples)
    """
    chunk_size = pulse_times_2d.shape[0]
    n_samples = len(sample_times)

    # Extract v1 (baseline) and v2 (high) for each pulse
    v1 = pulse_values_2d[:, 0]  # (chunk_size,)
    v2 = pulse_values_2d[:, 2]  # (chunk_size,)

    # Initialize all to v1 (baseline value)
    result = np.outer(v1, np.ones(n_samples))  # (chunk_size, n_samples)

    # Get high region boundaries: [delay+rt, delay+rt+width]
    # From pulse_times_2d: index 2 = delay+rt, index 3 = delay+rt+width
    high_start = pulse_times_2d[:, 2]  # (chunk_size,)
    high_end = pulse_times_2d[:, 3]    # (chunk_size,)

    # For each pulse, set high region samples to v2
    # Use broadcasting: sample_times is (n_samples,), high_start/end are (chunk_size,)
    sample_times_2d = sample_times[np.newaxis, :]  # (1, n_samples)
    high_start_2d = high_start[:, np.newaxis]       # (chunk_size, 1)
    high_end_2d = high_end[:, np.newaxis]           # (chunk_size, 1)

    # Handle periodic case: need to check wrapped time
    periods_2d = periods[:, np.newaxis]  # (chunk_size, 1)
    is_periodic = periods > 0  # (chunk_size,)

    # For non-periodic pulses, use sample times directly
    # For periodic pulses, wrap sample times to [0, period)
    effective_times = np.where(
        is_periodic[:, np.newaxis] & (sample_times_2d >= periods_2d),
        sample_times_2d % periods_2d,
        sample_times_2d
    )

    # Create mask for high region
    in_high_region = (effective_times >= high_start_2d) & (effective_times < high_end_2d)

    # Set high region values
    result = np.where(in_high_region, v2[:, np.newaxis], result)

    return result


def _compute_active_samples_vectorized(
    result: np.ndarray,
    sample_times: np.ndarray,
    pulse_times_2d: np.ndarray,
    pulse_values_2d: np.ndarray,
    periods: np.ndarray,
    intervals: np.ndarray,
    n_intervals: np.ndarray,
    half_width: float,
) -> None:
    """Compute smoothed values at active sample positions.

    Modifies result in-place to replace PWL values with smoothed values
    at sample positions within active intervals.

    Uses fully vectorized batch processing: all active (pulse, sample) pairs
    are computed together to maximize numpy efficiency.

    **Memory note:** The intermediate 3D boolean array has shape
    ``(chunk_size, n_samples, max_intervals)``.  At chunk_size=10000,
    n_samples=1001 (10ns/10ps), max_intervals=6 this is ~60 MB.
    For very small dt (e.g. 1ps -> n_samples=10001) this can reach ~600 MB.
    A warning is emitted when estimated usage exceeds 100 MB.

    Args:
        result: Output array (chunk_size, n_samples) - modified in-place
        sample_times: Uniform sample times (n_samples,)
        pulse_times_2d: PWL times (chunk_size, n_pwl_points)
        pulse_values_2d: PWL values (chunk_size, n_pwl_points)
        periods: Period for each pulse (chunk_size,)
        intervals: Pre-computed active intervals (chunk_size, max_intervals, 2)
        n_intervals: Number of valid intervals per pulse (chunk_size,)
        half_width: Filter half-width (= time_step)
    """
    import warnings

    chunk_size = pulse_times_2d.shape[0]
    n_pwl_points = pulse_times_2d.shape[1]
    n_samples = len(sample_times)

    # Build active mask for all pulses using vectorized interval checks
    # intervals shape: (chunk_size, max_intervals, 2)
    max_intervals = intervals.shape[1]

    # Estimate peak memory for the 3D boolean broadcast array
    # Shape: (chunk_size, n_samples, max_intervals), dtype=bool (1 byte each)
    estimated_bytes = chunk_size * n_samples * max_intervals
    estimated_mb = estimated_bytes / (1024 * 1024)
    if estimated_mb > 100:
        warnings.warn(
            f"_compute_active_samples_vectorized: intermediate 3D array "
            f"({chunk_size} x {n_samples} x {max_intervals}) will use "
            f"~{estimated_mb:.0f} MB. Consider reducing chunk_size or "
            f"increasing time_step to reduce memory usage.",
            ResourceWarning,
            stacklevel=2,
        )

    # Expand sample_times for broadcasting: (1, n_samples, 1)
    sample_times_3d = sample_times[np.newaxis, :, np.newaxis]

    # Expand intervals: (chunk_size, 1, max_intervals)
    int_starts = intervals[:, :, 0][:, np.newaxis, :]  # (chunk_size, 1, max_intervals)
    int_ends = intervals[:, :, 1][:, np.newaxis, :]

    # Create validity mask for intervals: (chunk_size, max_intervals)
    interval_valid = np.arange(max_intervals)[np.newaxis, :] < n_intervals[:, np.newaxis]
    interval_valid_3d = interval_valid[:, np.newaxis, :]  # (chunk_size, 1, max_intervals)

    # Check if each (pulse, sample) is in any valid interval
    # Shape: (chunk_size, n_samples, max_intervals)
    in_interval = (
        (sample_times_3d >= int_starts - 1e-15) &
        (sample_times_3d <= int_ends + 1e-15) &
        interval_valid_3d
    )

    # Reduce over intervals: (chunk_size, n_samples)
    all_active_masks = np.any(in_interval, axis=2)

    # Find all active (pulse_idx, sample_idx) pairs
    pulse_indices, sample_indices = np.where(all_active_masks)

    if len(pulse_indices) == 0:
        return  # No active samples to compute

    # Get the sample times for active pairs
    active_sample_times = sample_times[sample_indices]  # (n_active,)

    # Get pulse parameters for active pairs
    active_periods = periods[pulse_indices]  # (n_active,)

    # Compute weighted sums for all active pairs at once
    # We need to sum contributions from all PWL segments for each active pair
    weighted_sums = np.zeros(len(pulse_indices), dtype=np.float64)

    # Process each segment index (typically 5 segments for pulses)
    for k in range(n_pwl_points - 1):
        # Get segment endpoints for each active pair's pulse
        t1 = pulse_times_2d[pulse_indices, k]    # (n_active,)
        v1 = pulse_values_2d[pulse_indices, k]
        t2 = pulse_times_2d[pulse_indices, k + 1]
        v2 = pulse_values_2d[pulse_indices, k + 1]

        # Compute contribution from this segment (vectorized over all active pairs)
        valid_seg = t2 > t1
        if np.any(valid_seg):
            contrib = _analytical_integral_batch_flat(
                active_sample_times, half_width, t1, v1, t2, v2
            )
            weighted_sums += contrib

    # Handle periodic wraparound (only for pulses with period > 0)
    periodic_mask = active_periods > 0
    if np.any(periodic_mask):
        for k in range(n_pwl_points - 1):
            t1 = pulse_times_2d[pulse_indices, k]
            v1 = pulse_values_2d[pulse_indices, k]
            t2 = pulse_times_2d[pulse_indices, k + 1]
            v2 = pulse_values_2d[pulse_indices, k + 1]

            valid_seg = (t2 > t1) & periodic_mask
            if np.any(valid_seg):
                # Previous period
                contrib_prev = _analytical_integral_batch_flat(
                    active_sample_times, half_width,
                    t1 - active_periods, v1, t2 - active_periods, v2
                )
                weighted_sums = np.where(periodic_mask, weighted_sums + contrib_prev, weighted_sums)

                # Next period
                contrib_next = _analytical_integral_batch_flat(
                    active_sample_times, half_width,
                    t1 + active_periods, v1, t2 + active_periods, v2
                )
                weighted_sums = np.where(periodic_mask, weighted_sums + contrib_next, weighted_sums)

    # Store results
    result[pulse_indices, sample_indices] = weighted_sums / half_width


def _analytical_integral_batch_flat(
    t_out: np.ndarray,
    half_width: float,
    t1: np.ndarray,
    v1: np.ndarray,
    t2: np.ndarray,
    v2: np.ndarray,
) -> np.ndarray:
    """Batch analytical integral for flat arrays (one segment per sample).

    Unlike _analytical_integral_batch which computes all samples for each segment,
    this computes one sample per segment where each element has its own (t1, v1, t2, v2).

    Args:
        t_out: Sample times (n,)
        half_width: Filter half-width
        t1, v1: Segment start times/values (n,)
        t2, v2: Segment end times/values (n,)

    Returns:
        Integrals (n,) - one per (t_out, segment) pair
    """
    n = len(t_out)
    result = np.zeros(n, dtype=np.float64)

    if half_width <= 0:
        return result

    # Window bounds
    w_lo = t_out - half_width
    w_hi = t_out + half_width

    # Segment validity
    valid_seg = t2 > t1

    # Find overlap
    seg_lo = np.maximum(t1, w_lo)
    seg_hi = np.minimum(t2, w_hi)
    has_overlap = (seg_lo < seg_hi) & valid_seg

    if not np.any(has_overlap):
        return result

    # PWL slope and intercept (avoid division by zero for invalid segments)
    dt_seg = np.where(valid_seg, t2 - t1, 1.0)
    m = (v2 - v1) / dt_seg
    pwl_a = v1 - m * t1
    pwl_b = m

    # Left half of triangle: t in [w_lo, t_out]
    left_lo = np.maximum(seg_lo, w_lo)
    left_hi = np.minimum(seg_hi, t_out)
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
    right_lo = np.maximum(seg_lo, t_out)
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
        sparse_threshold: float = 100e-12,
    ) -> SmoothedWaveformCache:
        """Create reusable cache from VectorizedCurrentSources.

        This method:
        1. Converts all pulses to PWL
        2. Smooths all PWL waveforms using chunked batch processing
        3. Packs results into cache for reuse

        Uses chunked batch processing to control memory usage while
        maintaining high performance through vectorization.

        For small time_step values (< sparse_threshold), uses sparse sampling
        to achieve 10x+ speedup by only computing samples within active
        intervals where the smoothed output varies.

        Args:
            vec_sources: VectorizedCurrentSources instance
            t_start: Simulation start time
            t_end: Simulation end time
            chunk_size: Number of waveforms to process per chunk (default 10000).
                       Controls memory/speed tradeoff. Larger = faster but more memory.
            sparse_threshold: Use sparse sampling when time_step < this value (default 100ps).
                             Set to 0 to always use dense sampling.

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

        time_step = self.config.time_step
        use_sparse = time_step < sparse_threshold

        # Collect smoothed PWL data using pre-allocated lists for efficiency
        all_times: List[float] = []
        all_values: List[float] = []
        offsets: List[int] = []
        counts: List[int] = []
        periods: List[float] = []
        delays: List[float] = []
        node_indices: List[int] = []

        # Process original PWLs
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

            # Choose sparse or dense smoothing
            if use_sparse:
                smoothed_times, smoothed_values = _smooth_pwl_sparse(
                    pwl_times, pwl_values, period, time_step, t_start, t_end
                )
            else:
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

        # Process pulses in chunks
        n_pulses = vec_sources.n_pulses
        if n_pulses > 0:
            if use_sparse:
                # Sparse path: use active interval detection
                self._process_pulses_sparse(
                    vec_sources, t_start, t_end, chunk_size,
                    all_times, all_values, offsets, counts, periods, delays, node_indices
                )
            else:
                # Dense path: use existing batch processing
                n_samples = int((t_end - t_start) / time_step) + 1
                sample_times = np.linspace(t_start, t_end, n_samples)

                for chunk_start in range(0, n_pulses, chunk_size):
                    chunk_end = min(chunk_start + chunk_size, n_pulses)

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
            "use_sparse": use_sparse,
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

    def _process_pulses_sparse(
        self,
        vec_sources: "VectorizedCurrentSources",
        t_start: float,
        t_end: float,
        chunk_size: int,
        all_times: List[float],
        all_values: List[float],
        offsets: List[int],
        counts: List[int],
        periods: List[float],
        delays: List[float],
        node_indices: List[int],
    ) -> None:
        """Process pulses using sparse sampling (internal method).

        Uses active interval detection to only compute samples where the
        smoothed output varies, achieving significant speedup for small time_step.
        """
        time_step = self.config.time_step
        n_pulses = vec_sources.n_pulses

        for chunk_start in range(0, n_pulses, chunk_size):
            chunk_end = min(chunk_start + chunk_size, n_pulses)

            # Extract chunk data
            chunk_delay = vec_sources.pulse_delay[chunk_start:chunk_end]
            chunk_rt = vec_sources.pulse_rt[chunk_start:chunk_end]
            chunk_ft = vec_sources.pulse_ft[chunk_start:chunk_end]
            chunk_width = vec_sources.pulse_width[chunk_start:chunk_end]
            chunk_period = vec_sources.pulse_period[chunk_start:chunk_end]
            chunk_node_idx = vec_sources.pulse_node_idx[chunk_start:chunk_end]
            chunk_v1 = vec_sources.pulse_v1[chunk_start:chunk_end]
            chunk_v2 = vec_sources.pulse_v2[chunk_start:chunk_end]

            # Compute active intervals for all pulses in chunk
            intervals, n_intervals = _compute_pulse_active_intervals(
                chunk_delay, chunk_rt, chunk_ft, chunk_width, chunk_period,
                time_step, t_start, t_end
            )

            # Convert pulses to PWL arrays
            pulse_times_2d, pulse_values_2d = _pulse_to_pwl_arrays(
                chunk_v1, chunk_v2, chunk_delay, chunk_rt,
                chunk_ft, chunk_width, chunk_period
            )

            # Smooth using sparse computation (same output format as dense)
            sample_times, smoothed_values_2d = _smooth_pulse_chunk_sparse(
                pulse_times_2d, pulse_values_2d, chunk_period,
                intervals, n_intervals, time_step, t_start, t_end
            )

            # Compact chunk (same as dense path)
            times_list, values_list, chunk_counts = _compact_chunk_vectorized(
                sample_times, smoothed_values_2d, self.config.compact_threshold
            )

            # Append to output (same as dense path)
            _compact_and_append(
                times_list, values_list, chunk_counts,
                chunk_period, chunk_node_idx,
                all_times, all_values, offsets, counts,
                periods, delays, node_indices
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
        smoothed._pwl_padded_times = None

        return smoothed

    def get_statistics(self) -> Dict[str, Any]:
        """Return smoothing statistics from last operation.

        Returns:
            Dict with keys: original_pwls, original_pulses, total_smoothed,
                          original_points, smoothed_points
        """
        return dict(self._stats)

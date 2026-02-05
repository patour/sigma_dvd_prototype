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


def smooth_pwl_points(
    points: List[Tuple[float, float]],
    period: float,
    time_step: float,
    t_start: float,
    t_end: float,
) -> List[Tuple[float, float]]:
    """Apply analytical triangular low-pass filter to PWL waveform.

    Reference: C++ Filter::analytical_LP_filter

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
    ) -> SmoothedWaveformCache:
        """Create reusable cache from VectorizedCurrentSources.

        This method:
        1. Converts all pulses to PWL
        2. Smooths all PWL waveforms
        3. Packs results into cache for reuse

        Args:
            vec_sources: VectorizedCurrentSources instance
            t_start: Simulation start time
            t_end: Simulation end time

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

        # Collect smoothed PWL data
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

            # Extract original points
            times = vec_sources.pwl_times[offset : offset + count]
            values = vec_sources.pwl_values[offset : offset + count]
            points = list(zip(times, values))

            # Adjust for delay
            if delay > 0:
                points = [(t + delay, v) for t, v in points]

            # Smooth and compact
            smoothed = smooth_pwl_points(
                points, period, self.config.time_step, t_start, t_end
            )
            compacted = compact_pwl(smoothed, self.config.compact_threshold)

            # Store results
            offsets.append(len(all_times))
            counts.append(len(compacted))
            periods.append(period)
            delays.append(0.0)  # Delay absorbed into points
            node_indices.append(node_idx)

            for t, v in compacted:
                all_times.append(t)
                all_values.append(v)

        # Process pulses (convert to PWL and smooth)
        for i in range(vec_sources.n_pulses):
            node_idx = int(vec_sources.pulse_node_idx[i])
            v1 = float(vec_sources.pulse_v1[i])
            v2 = float(vec_sources.pulse_v2[i])
            delay = float(vec_sources.pulse_delay[i])
            rt = float(vec_sources.pulse_rt[i])
            ft = float(vec_sources.pulse_ft[i])
            width = float(vec_sources.pulse_width[i])
            period = float(vec_sources.pulse_period[i])

            # Convert to PWL points
            pwl_points = pulse_to_pwl_points(v1, v2, delay, rt, ft, width, period)

            # Smooth and compact
            smoothed = smooth_pwl_points(
                pwl_points, period, self.config.time_step, t_start, t_end
            )
            compacted = compact_pwl(smoothed, self.config.compact_threshold)

            # Store results
            offsets.append(len(all_times))
            counts.append(len(compacted))
            periods.append(period)
            delays.append(0.0)
            node_indices.append(node_idx)

            for t, v in compacted:
                all_times.append(t)
                all_values.append(v)

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

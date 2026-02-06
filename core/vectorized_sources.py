"""Vectorized current source storage and evaluation for large-scale simulations.

This module provides memory-efficient storage and fast evaluation of millions
of current sources using columnar (struct-of-arrays) layout and numpy vectorization.

Memory reduction: ~8-10x compared to object-based storage
Runtime speedup: ~30-50x compared to sequential evaluation

Example usage:
    from core import VectorizedCurrentSources, DynamicIRDropSolver

    # Automatic usage in DynamicIRDropSolver for large source counts
    solver = DynamicIRDropSolver(model, graph, vectorize_threshold=10000)

    # Manual usage
    vec_sources = VectorizedCurrentSources.from_graph(
        graph, model.edge_cache.node_to_idx, model.edge_cache.n_nodes
    )
    currents = vec_sources.evaluate_at_time(5e-9)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class VectorizedCurrentSources:
    """Memory-efficient storage for millions of current sources.

    Uses columnar (struct-of-arrays) layout instead of object-based storage.
    All evaluation is vectorized using numpy for O(N) performance with low
    constant factors.

    Attributes:
        n_nodes: Number of nodes in the grid
        node_to_idx: Reference to model's node->index mapping (not owned)

        dc_values: DC current values for each source (mA)
        source_node_idx: Node index for each source

        pulse_*: Columnar arrays for all pulse waveforms
        pulse_source_idx: Source index for each pulse (for masking)
        pwl_*: Packed arrays for all PWL waveforms
        pwl_source_idx: Source index for each PWL (for masking)
    """

    # Node mapping (reference to model's mapping, not owned)
    n_nodes: int = 0
    node_to_idx: Optional[Dict[Any, int]] = None

    # DC currents (n_sources,)
    n_sources: int = 0
    dc_values: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    source_node_idx: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int32))

    # Pulse data - columnar storage (n_pulses,)
    n_pulses: int = 0
    pulse_node_idx: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int32))
    pulse_source_idx: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int32))
    pulse_v1: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    pulse_v2: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    pulse_delay: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    pulse_rt: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    pulse_ft: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    pulse_width: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    pulse_period: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))

    # PWL data - packed storage
    n_pwls: int = 0
    n_pwl_points: int = 0
    pwl_node_idx: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int32))
    pwl_source_idx: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int32))
    pwl_period: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    pwl_delay: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    pwl_offset: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int32))
    pwl_count: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int32))
    pwl_times: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    pwl_values: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))

    @classmethod
    def from_current_sources(
        cls,
        sources: Dict[str, Any],
        node_to_idx: Dict[Any, int],
        n_nodes: int,
    ) -> 'VectorizedCurrentSources':
        """Convert dict of CurrentSource objects to vectorized format.

        Args:
            sources: Dict mapping source name to CurrentSource object
            node_to_idx: Model's node->index mapping (from model.edge_cache.node_to_idx)
            n_nodes: Total number of nodes (from model.edge_cache.n_nodes)

        Returns:
            VectorizedCurrentSources instance with columnar storage
        """
        obj = cls(n_nodes=n_nodes, node_to_idx=node_to_idx)

        # Collect all data into lists first
        dc_list: List[float] = []
        source_node_list: List[int] = []

        pulse_data: Dict[str, List] = {
            'node_idx': [], 'source_idx': [], 'v1': [], 'v2': [], 'delay': [],
            'rt': [], 'ft': [], 'width': [], 'period': []
        }

        pwl_meta: Dict[str, List] = {
            'node_idx': [], 'source_idx': [], 'period': [], 'delay': [], 'offset': [], 'count': []
        }
        pwl_times_list: List[float] = []
        pwl_values_list: List[float] = []

        source_idx = 0
        for name, src in sources.items():
            node = src.node1
            if node not in node_to_idx:
                continue
            node_idx = node_to_idx[node]

            dc_list.append(src.dc_value)
            source_node_list.append(node_idx)

            # Collect pulses (track source index for masking)
            for pulse in src.pulses:
                pulse_data['node_idx'].append(node_idx)
                pulse_data['source_idx'].append(source_idx)
                pulse_data['v1'].append(pulse.v1)
                pulse_data['v2'].append(pulse.v2)
                pulse_data['delay'].append(pulse.delay)
                pulse_data['rt'].append(pulse.rt)
                pulse_data['ft'].append(pulse.ft)
                pulse_data['width'].append(pulse.width)
                pulse_data['period'].append(pulse.period)

            # Collect PWLs (packed format, track source index for masking)
            for pwl in src.pwls:
                pwl_meta['node_idx'].append(node_idx)
                pwl_meta['source_idx'].append(source_idx)
                pwl_meta['period'].append(pwl.period)
                pwl_meta['delay'].append(pwl.delay)
                pwl_meta['offset'].append(len(pwl_times_list))
                pwl_meta['count'].append(len(pwl.points))
                for t, v in pwl.points:
                    pwl_times_list.append(t)
                    pwl_values_list.append(v)

            source_idx += 1

        # Convert to numpy arrays
        obj.n_sources = len(dc_list)
        obj.dc_values = np.array(dc_list, dtype=np.float64)
        obj.source_node_idx = np.array(source_node_list, dtype=np.int32)

        obj.n_pulses = len(pulse_data['node_idx'])
        obj.pulse_node_idx = np.array(pulse_data['node_idx'], dtype=np.int32)
        obj.pulse_source_idx = np.array(pulse_data['source_idx'], dtype=np.int32)
        obj.pulse_v1 = np.array(pulse_data['v1'], dtype=np.float64)
        obj.pulse_v2 = np.array(pulse_data['v2'], dtype=np.float64)
        obj.pulse_delay = np.array(pulse_data['delay'], dtype=np.float64)
        obj.pulse_rt = np.array(pulse_data['rt'], dtype=np.float64)
        obj.pulse_ft = np.array(pulse_data['ft'], dtype=np.float64)
        obj.pulse_width = np.array(pulse_data['width'], dtype=np.float64)
        obj.pulse_period = np.array(pulse_data['period'], dtype=np.float64)

        obj.n_pwls = len(pwl_meta['node_idx'])
        obj.n_pwl_points = len(pwl_times_list)
        obj.pwl_node_idx = np.array(pwl_meta['node_idx'], dtype=np.int32)
        obj.pwl_source_idx = np.array(pwl_meta['source_idx'], dtype=np.int32)
        obj.pwl_period = np.array(pwl_meta['period'], dtype=np.float64)
        obj.pwl_delay = np.array(pwl_meta['delay'], dtype=np.float64)
        obj.pwl_offset = np.array(pwl_meta['offset'], dtype=np.int32)
        obj.pwl_count = np.array(pwl_meta['count'], dtype=np.int32)
        obj.pwl_times = np.array(pwl_times_list, dtype=np.float64)
        obj.pwl_values = np.array(pwl_values_list, dtype=np.float64)

        return obj

    @classmethod
    def from_graph(
        cls,
        graph: Any,
        node_to_idx: Dict[Any, int],
        n_nodes: int,
    ) -> 'VectorizedCurrentSources':
        """Create from graph metadata (instance_sources).

        Args:
            graph: Parsed graph with 'instance_sources' in metadata
            node_to_idx: Model's node->index mapping
            n_nodes: Total number of nodes

        Returns:
            VectorizedCurrentSources instance
        """
        # Import here to avoid circular imports
        try:
            from pdn.pdn_parser import CurrentSource as CS
        except ImportError:
            return cls(n_nodes=n_nodes, node_to_idx=node_to_idx)

        # Extract instance_sources from graph metadata
        instance_sources: Dict = {}
        if hasattr(graph, 'graph') and isinstance(graph.graph, dict):
            instance_sources = graph.graph.get('instance_sources', {})
        elif hasattr(graph, '_attrs'):
            instance_sources = graph._attrs.get('instance_sources', {})

        if not instance_sources:
            return cls(n_nodes=n_nodes, node_to_idx=node_to_idx)

        # Use direct parsing (no intermediate objects) for efficiency
        return cls.from_serialized_dicts(instance_sources, node_to_idx, n_nodes)

    @classmethod
    def from_serialized_dicts(
        cls,
        instance_sources: Dict[str, Dict],
        node_to_idx: Dict[Any, int],
        n_nodes: int,
    ) -> 'VectorizedCurrentSources':
        """Create directly from serialized dict format (no intermediate objects).

        This is the most memory-efficient method - parses directly into numpy
        arrays without creating CurrentSource, Pulse, or PWL objects.

        Args:
            instance_sources: Dict mapping name -> serialized CurrentSource dict
            node_to_idx: Model's node->index mapping
            n_nodes: Total number of nodes

        Returns:
            VectorizedCurrentSources instance
        """
        obj = cls(n_nodes=n_nodes, node_to_idx=node_to_idx)

        if not instance_sources:
            return obj

        # Pre-allocate lists for collecting data
        dc_list: List[float] = []
        source_node_list: List[int] = []

        pulse_data: Dict[str, List] = {
            'node_idx': [], 'source_idx': [], 'v1': [], 'v2': [], 'delay': [],
            'rt': [], 'ft': [], 'width': [], 'period': []
        }

        pwl_meta: Dict[str, List] = {
            'node_idx': [], 'source_idx': [], 'period': [], 'delay': [], 'offset': [], 'count': []
        }
        pwl_times_list: List[float] = []
        pwl_values_list: List[float] = []

        # Parse directly from serialized format
        source_idx = 0
        for name, data in instance_sources.items():
            node1 = data.get('node1', '')
            if node1 not in node_to_idx:
                continue
            node_idx = node_to_idx[node1]

            # DC value (already in mA from parser)
            dc_list.append(data.get('dc_value', 0.0))
            source_node_list.append(node_idx)

            # Parse pulses directly from dict (track source index for masking)
            for pulse_dict in data.get('pulses', []):
                pulse_data['node_idx'].append(node_idx)
                pulse_data['source_idx'].append(source_idx)
                pulse_data['v1'].append(pulse_dict.get('v1', 0.0))
                pulse_data['v2'].append(pulse_dict.get('v2', 0.0))
                pulse_data['delay'].append(pulse_dict.get('delay', 0.0))
                pulse_data['rt'].append(pulse_dict.get('rt', 0.0))
                pulse_data['ft'].append(pulse_dict.get('ft', 0.0))
                pulse_data['width'].append(pulse_dict.get('width', 0.0))
                pulse_data['period'].append(pulse_dict.get('period', 0.0))

            # Parse PWLs directly from dict (packed format, track source index for masking)
            for pwl_dict in data.get('pwls', []):
                points = pwl_dict.get('points', [])
                pwl_meta['node_idx'].append(node_idx)
                pwl_meta['source_idx'].append(source_idx)
                pwl_meta['period'].append(pwl_dict.get('period', 0.0))
                pwl_meta['delay'].append(pwl_dict.get('delay', 0.0))
                pwl_meta['offset'].append(len(pwl_times_list))
                pwl_meta['count'].append(len(points))
                for t, v in points:
                    pwl_times_list.append(t)
                    pwl_values_list.append(v)

            source_idx += 1

        # Convert to numpy arrays
        obj.n_sources = len(dc_list)
        obj.dc_values = np.array(dc_list, dtype=np.float64)
        obj.source_node_idx = np.array(source_node_list, dtype=np.int32)

        obj.n_pulses = len(pulse_data['node_idx'])
        obj.pulse_node_idx = np.array(pulse_data['node_idx'], dtype=np.int32)
        obj.pulse_source_idx = np.array(pulse_data['source_idx'], dtype=np.int32)
        obj.pulse_v1 = np.array(pulse_data['v1'], dtype=np.float64)
        obj.pulse_v2 = np.array(pulse_data['v2'], dtype=np.float64)
        obj.pulse_delay = np.array(pulse_data['delay'], dtype=np.float64)
        obj.pulse_rt = np.array(pulse_data['rt'], dtype=np.float64)
        obj.pulse_ft = np.array(pulse_data['ft'], dtype=np.float64)
        obj.pulse_width = np.array(pulse_data['width'], dtype=np.float64)
        obj.pulse_period = np.array(pulse_data['period'], dtype=np.float64)

        obj.n_pwls = len(pwl_meta['node_idx'])
        obj.n_pwl_points = len(pwl_times_list)
        obj.pwl_node_idx = np.array(pwl_meta['node_idx'], dtype=np.int32)
        obj.pwl_source_idx = np.array(pwl_meta['source_idx'], dtype=np.int32)
        obj.pwl_period = np.array(pwl_meta['period'], dtype=np.float64)
        obj.pwl_delay = np.array(pwl_meta['delay'], dtype=np.float64)
        obj.pwl_offset = np.array(pwl_meta['offset'], dtype=np.int32)
        obj.pwl_count = np.array(pwl_meta['count'], dtype=np.int32)
        obj.pwl_times = np.array(pwl_times_list, dtype=np.float64)
        obj.pwl_values = np.array(pwl_values_list, dtype=np.float64)

        return obj

    def memory_bytes(self) -> int:
        """Total memory usage in bytes."""
        arrays = [
            self.dc_values, self.source_node_idx,
            self.pulse_node_idx, self.pulse_source_idx, self.pulse_v1, self.pulse_v2,
            self.pulse_delay, self.pulse_rt, self.pulse_ft,
            self.pulse_width, self.pulse_period,
            self.pwl_node_idx, self.pwl_source_idx, self.pwl_period, self.pwl_delay,
            self.pwl_offset, self.pwl_count, self.pwl_times, self.pwl_values
        ]
        return sum(arr.nbytes for arr in arrays)

    def evaluate_at_time(self, t: float) -> np.ndarray:
        """Evaluate all current sources at time t.

        Args:
            t: Time in seconds

        Returns:
            np.ndarray of shape (n_nodes,) with total current at each node (mA)
        """
        currents = np.zeros(self.n_nodes, dtype=np.float64)

        # Add DC values
        if self.n_sources > 0:
            np.add.at(currents, self.source_node_idx, self.dc_values)

        # Evaluate and accumulate pulses
        if self.n_pulses > 0:
            pulse_values = self._evaluate_pulses(t)
            np.add.at(currents, self.pulse_node_idx, pulse_values)

        # Evaluate and accumulate PWLs
        if self.n_pwls > 0:
            pwl_values = self._evaluate_pwls(t)
            np.add.at(currents, self.pwl_node_idx, pwl_values)

        return currents

    def evaluate_per_source_at_time(self, t: float) -> np.ndarray:
        """Evaluate per-source currents at time t (for masking operations).

        Unlike evaluate_at_time which aggregates currents by node, this returns
        the current for each source individually. This is required for
        solve_transient_multi_rhs where masks are applied per-source.

        Args:
            t: Time in seconds

        Returns:
            np.ndarray of shape (n_sources,) with current for each source (mA).
            Includes DC value plus any pulse/PWL contributions for that source.
        """
        if self.n_sources == 0:
            return np.array([], dtype=np.float64)

        # Start with DC values (one per source)
        per_source = self.dc_values.copy()

        # Add pulse contributions (accumulate to the source each pulse belongs to)
        if self.n_pulses > 0:
            pulse_values = self._evaluate_pulses(t)
            np.add.at(per_source, self.pulse_source_idx, pulse_values)

        # Add PWL contributions (accumulate to the source each PWL belongs to)
        if self.n_pwls > 0:
            pwl_values = self._evaluate_pwls(t)
            np.add.at(per_source, self.pwl_source_idx, pwl_values)

        return per_source

    def evaluate_at_time_as_dict(
        self,
        t: float,
        idx_to_node: List[Any],
    ) -> Dict[Any, float]:
        """Evaluate and return as dict with node keys.

        Args:
            t: Time in seconds
            idx_to_node: Model's index->node mapping (from model.edge_cache.idx_to_node)

        Returns:
            Dict mapping node -> current (mA), only non-zero entries
        """
        currents = self.evaluate_at_time(t)
        return {
            idx_to_node[i]: currents[i]
            for i in range(self.n_nodes)
            if currents[i] != 0
        }

    def _evaluate_pulses(self, t: float) -> np.ndarray:
        """Vectorized pulse evaluation (standard SPICE timing).

        Pulse timing interpretation (matches C++ SimPWL):
        - delay = START time (when signal begins rising from v1)
        - rt = rise time from v1 to v2
        - width = duration at v2 after rise completes
        - ft = fall time from v2 back to v1

        Timeline: [delay] --rise--> [delay+rt] --high--> [delay+rt+width] --fall--> [delay+rt+width+ft]

        Args:
            t: Time in seconds

        Returns:
            Array of pulse values (n_pulses,)
        """
        n = self.n_pulses
        result = np.empty(n, dtype=np.float64)

        period = self.pulse_period
        v1, v2 = self.pulse_v1, self.pulse_v2
        delay = self.pulse_delay
        rt, ft, width = self.pulse_rt, self.pulse_ft, self.pulse_width

        # Pulse starts at delay (standard SPICE timing)
        pulse_start = delay

        # Handle periodicity: t_arr = t % period for periodic signals
        t_arr = np.where(period > 0, t % period, t)
        t_rel = t_arr - pulse_start

        # Wrap negative t_rel for periodic signals
        periodic_neg = (period > 0) & (t_rel < 0)
        t_rel = np.where(periodic_neg, t_rel + period, t_rel)

        # Phase masks (mutually exclusive)
        m_before = t_rel < 0
        m_rise = (~m_before) & (t_rel < rt)
        m_high = (~m_before) & (~m_rise) & (t_rel < rt + width)
        m_fall = (~m_before) & (~m_rise) & (~m_high) & (t_rel < rt + width + ft)
        m_low = (~m_before) & (~m_rise) & (~m_high) & (~m_fall)

        # Compute values for each phase
        result[m_before] = v1[m_before]

        # Rise phase: linear interpolation
        with np.errstate(divide='ignore', invalid='ignore'):
            rise_frac = np.where(rt > 0, t_rel / rt, 1.0)
        result[m_rise] = v1[m_rise] + (v2[m_rise] - v1[m_rise]) * rise_frac[m_rise]

        # High phase
        result[m_high] = v2[m_high]

        # Fall phase: linear interpolation
        t_fall = t_rel - rt - width
        with np.errstate(divide='ignore', invalid='ignore'):
            fall_frac = np.where(ft > 0, t_fall / ft, 1.0)
        result[m_fall] = v2[m_fall] + (v1[m_fall] - v2[m_fall]) * fall_frac[m_fall]

        # Low phase
        result[m_low] = v1[m_low]

        return result

    def _evaluate_pwls(self, t: float) -> np.ndarray:
        """Evaluate all PWLs at time t using vectorized processing.

        Uses padded 2D arrays with per-row searchsorted to find segment indices.
        The per-row approach avoids floating-point precision issues that occur
        with flat searchsorted when row offsets become large (> ~500K rows).

        Args:
            t: Time in seconds

        Returns:
            Array of PWL values (n_pwls,)
        """
        if self.n_pwls == 0:
            return np.zeros(0, dtype=np.float64)

        # Build padded cache on first call
        if not hasattr(self, '_pwl_padded_times') or self._pwl_padded_times is None:
            self._build_pwl_padded()

        n = self.n_pwls
        max_count = self._pwl_padded_max_count
        times_2d = self._pwl_padded_times
        values_2d = self._pwl_padded_values
        row_idx = self._pwl_padded_row_idx
        first_times = self._pwl_padded_first_times
        last_times = self._pwl_padded_last_times
        first_values = self._pwl_padded_first_values
        last_values = self._pwl_padded_last_values

        # Adjust time for delay and periodicity (vectorized, with fast paths)
        if self._pwl_all_zero_delay and self._pwl_single_period > 0:
            # Common smoothed case: no delay, uniform period
            t_mod = t % self._pwl_single_period
            t_adj = np.full(n, t_mod, dtype=np.float64)
        else:
            t_adj = np.full(n, t, dtype=np.float64) - self.pwl_delay
            periodic_mask = self._pwl_has_period & (t_adj >= 0)
            t_adj[periodic_mask] = t_adj[periodic_mask] % self.pwl_period[periodic_mask]

        if max_count == 1:
            return first_values.copy()

        # Clamp time to valid range for safe interpolation
        t_clamped = np.clip(t_adj, first_times, last_times)

        # Per-row searchsorted using broadcasting
        # times_2d: (n, max_count), t_clamped: (n,)
        # Compare each row's times against its corresponding t_clamped value
        # seg_idx[i] = largest j such that times_2d[i, j] <= t_clamped[i]
        t_clamped_2d = t_clamped[:, np.newaxis]  # (n, 1)
        # Count how many time points are <= t_clamped for each row
        seg_idx = np.sum(times_2d <= t_clamped_2d, axis=1) - 1
        seg_idx = np.clip(seg_idx, 0, max_count - 2)

        # Gather segment endpoints using advanced indexing
        t1 = times_2d[row_idx, seg_idx]
        t2 = times_2d[row_idx, seg_idx + 1]
        v1 = values_2d[row_idx, seg_idx]
        v2 = values_2d[row_idx, seg_idx + 1]

        # Vectorized linear interpolation
        dt = t2 - t1
        safe_dt = np.where(dt > 0, dt, 1.0)
        frac = np.where(dt > 0, (t_clamped - t1) / safe_dt, 0.0)
        values = v1 + (v2 - v1) * frac

        # Override boundary cases
        before_mask = t_adj <= first_times
        after_mask = t_adj >= last_times

        values[before_mask] = first_values[before_mask]

        # After last point: periodic → wrap to first value;
        # non-periodic → hold last value
        after_periodic = after_mask & self._pwl_has_period
        after_nonperiodic = after_mask & self._pwl_no_period
        values[after_periodic] = first_values[after_periodic]
        values[after_nonperiodic] = last_values[after_nonperiodic]

        return values

    def _build_pwl_padded(self) -> None:
        """Build padded 2D arrays for vectorized PWL evaluation.

        Pads all PWLs to max point count by repeating the last time/value.
        This creates a single (n_pwls, max_count) array that enables
        vectorized evaluation with no per-PWL Python loop.

        Padding correctness: repeated last-time segments have dt=0, so
        interpolation returns last_value via the np.where(dt > 0, ..., 0.0)
        guard. Boundary overrides then handle periodic wrap.

        Memory: O(n_pwls * max_count * 16 bytes) for times + values.
        """
        n = self.n_pwls
        max_count = int(self.pwl_count.max()) if n > 0 else 0

        # Build padded 2D arrays
        times_padded = np.empty((n, max_count), dtype=np.float64)
        values_padded = np.empty((n, max_count), dtype=np.float64)

        for i in range(n):
            off = int(self.pwl_offset[i])
            cnt = int(self.pwl_count[i])
            times_padded[i, :cnt] = self.pwl_times[off:off + cnt]
            times_padded[i, cnt:] = self.pwl_times[off + cnt - 1]
            values_padded[i, :cnt] = self.pwl_values[off:off + cnt]
            values_padded[i, cnt:] = self.pwl_values[off + cnt - 1]

        self._pwl_padded_times = times_padded
        self._pwl_padded_values = values_padded
        self._pwl_padded_max_count = max_count

        # Pre-cache per-call constants to avoid repeated allocation
        self._pwl_padded_row_idx = np.arange(n, dtype=np.intp)
        self._pwl_padded_first_times = times_padded[:, 0].copy() if max_count > 0 else np.zeros(n)
        self._pwl_padded_last_times = times_padded[:, -1].copy() if max_count > 0 else np.zeros(n)
        self._pwl_padded_first_values = values_padded[:, 0].copy() if max_count > 0 else np.zeros(n)
        self._pwl_padded_last_values = values_padded[:, -1].copy() if max_count > 0 else np.zeros(n)

        # Cache period/delay analysis for fast-path dispatch
        self._pwl_has_period = self.pwl_period > 0
        self._pwl_no_period = ~self._pwl_has_period
        self._pwl_all_zero_delay = bool(np.all(self.pwl_delay == 0))
        periods_unique = np.unique(self.pwl_period)
        if len(periods_unique) == 1 and periods_unique[0] > 0:
            self._pwl_single_period = float(periods_unique[0])
        else:
            self._pwl_single_period = 0.0  # 0 = not uniform

    def get_statistics(self) -> Dict[str, Any]:
        """Return statistics about the vectorized sources."""
        return {
            'n_sources': self.n_sources,
            'n_nodes': self.n_nodes,
            'n_pulses': self.n_pulses,
            'n_pwls': self.n_pwls,
            'n_pwl_points': self.n_pwl_points,
            'memory_bytes': self.memory_bytes(),
            'memory_mb': self.memory_bytes() / 1e6,
        }

    def evaluate_to_rhs_array(
        self,
        t: float,
        rhs: np.ndarray,
        source_to_unknown: np.ndarray,
        valid_mask: np.ndarray,
    ) -> Tuple[float, np.ndarray]:
        """Evaluate currents and write directly to RHS array.

        This is the most efficient method for time-stepping loops - avoids
        creating intermediate dicts and uses vectorized scatter operations.

        Args:
            t: Time in seconds
            rhs: Pre-allocated RHS array (n_unknown,) to write to. Will be modified in-place.
            source_to_unknown: Mapping from source node indices to unknown indices.
                               Shape (n_nodes,), value -1 indicates node is not unknown.
            valid_mask: Boolean mask where source_to_unknown >= 0. Shape (n_nodes,).

        Returns:
            Tuple of (total_current, currents_array) where:
            - total_current: Sum of all source currents for statistics
            - currents_array: Full currents array (n_nodes,) for reuse (e.g., peak tracking)
        """
        # Evaluate all sources to get currents array
        currents = self.evaluate_at_time(t)

        # Compute total current for statistics
        total_current = currents.sum()

        # Get valid (non-zero, unknown) currents
        # Filter to nodes that are in the unknown set
        valid_currents = currents[valid_mask]
        valid_unknown_idx = source_to_unknown[valid_mask]

        # Scatter-add negative currents to RHS (sink current convention: I_u = -current,
        # see "Current Sign Convention" in the core solver documentation)
        # Use np.add.at for accumulation at duplicate indices
        if len(valid_currents) > 0:
            np.add.at(rhs, valid_unknown_idx, -valid_currents)

        return total_current, currents

    def build_source_to_unknown_map(
        self,
        unknown_to_idx: Dict[Any, int],
        idx_to_node: List[Any],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build mapping from source node indices to unknown indices.

        This should be called once at solver initialization to create the
        mapping arrays used by evaluate_to_rhs_array().

        Args:
            unknown_to_idx: Dict mapping unknown node -> index in reduced system
            idx_to_node: List mapping node index -> node (from edge_cache.idx_to_node)

        Returns:
            Tuple of (source_to_unknown, valid_mask) where:
            - source_to_unknown: int32 array (n_nodes,), -1 for non-unknown nodes
            - valid_mask: bool array (n_nodes,), True where source_to_unknown >= 0
        """
        source_to_unknown = np.full(self.n_nodes, -1, dtype=np.int32)

        for node_idx in range(self.n_nodes):
            node = idx_to_node[node_idx]
            if node in unknown_to_idx:
                source_to_unknown[node_idx] = unknown_to_idx[node]

        valid_mask = source_to_unknown >= 0

        return source_to_unknown, valid_mask

    def build_source_to_unknown_direct(
        self,
        source_to_unknown: np.ndarray,
        valid_mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build direct source-to-unknown mapping for multi-RHS operations.

        Unlike build_source_to_unknown_map which maps node indices to unknown indices,
        this maps source indices directly to unknown indices, enabling efficient
        vectorized aggregation in solve_transient_multi_rhs.

        Args:
            source_to_unknown: Node-level mapping from build_source_to_unknown_map
            valid_mask: Node-level validity mask from build_source_to_unknown_map

        Returns:
            Tuple of (source_to_unknown_direct, valid_sources) where:
            - source_to_unknown_direct: int32 array (n_sources,), -1 if source's node is not unknown
            - valid_sources: bool array (n_sources,), True where source maps to a valid unknown node
        """
        # Map each source directly to its unknown node index
        source_node_indices = self.source_node_idx
        source_to_unknown_direct = source_to_unknown[source_node_indices]
        valid_sources = valid_mask[source_node_indices]

        return source_to_unknown_direct, valid_sources

    def build_source_to_unknown_direct_from_dict(
        self,
        unknown_to_idx: Dict[Any, int],
        idx_to_node: List[Any],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build direct source-to-unknown mapping without intermediate node mapping.

        This is the efficient method for solve_transient_multi_rhs - builds the
        source -> unknown mapping directly without creating the intermediate
        node -> unknown mapping.

        Args:
            unknown_to_idx: Dict mapping unknown node -> index in reduced system
            idx_to_node: List mapping node index -> node (from edge_cache.idx_to_node)

        Returns:
            Tuple of (source_to_unknown_direct, valid_sources) where:
            - source_to_unknown_direct: int32 array (n_sources,), -1 if source's node is not unknown
            - valid_sources: bool array (n_sources,), True where source maps to a valid unknown node
        """
        n_sources = self.n_sources
        source_to_unknown_direct = np.full(n_sources, -1, dtype=np.int32)

        for src_idx in range(n_sources):
            node_idx = self.source_node_idx[src_idx]
            node = idx_to_node[node_idx]
            if node in unknown_to_idx:
                source_to_unknown_direct[src_idx] = unknown_to_idx[node]

        valid_sources = source_to_unknown_direct >= 0

        return source_to_unknown_direct, valid_sources

    def evaluate_to_multi_rhs(
        self,
        t: float,
        rhs_multi: np.ndarray,
        source_masks: np.ndarray,
        source_to_unknown_direct: np.ndarray,
        valid_sources: np.ndarray,
    ) -> np.ndarray:
        """Evaluate currents for multiple RHS vectors with source masks.

        This is the efficient vectorized method for solve_transient_multi_rhs.
        Evaluates per-source currents once, then aggregates to each RHS using
        vectorized operations.

        Args:
            t: Time in seconds
            rhs_multi: Pre-allocated RHS array (n_unknown, n_masks) to write to.
                       Will be modified in-place.
            source_masks: Boolean mask (n_masks, n_sources) indicating which
                          sources are active for each mask.
            source_to_unknown_direct: Direct source->unknown mapping from
                                      build_source_to_unknown_direct.
            valid_sources: Boolean mask from build_source_to_unknown_direct.

        Returns:
            total_currents: Array (n_masks,) with total current for each mask.
        """
        n_masks = source_masks.shape[0]
        n_unknown = rhs_multi.shape[0]

        # Evaluate per-source currents once
        per_source_currents = self.evaluate_per_source_at_time(t)

        # Compute total currents per mask (fully vectorized)
        # source_masks: (n_masks, n_sources), per_source_currents: (n_sources,)
        total_currents = (source_masks * per_source_currents).sum(axis=1)

        # Pre-filter to valid sources (those that map to unknown nodes)
        valid_idx = np.where(valid_sources)[0]
        if len(valid_idx) == 0:
            return total_currents

        valid_unknown_idx = source_to_unknown_direct[valid_idx]
        valid_currents = per_source_currents[valid_idx]

        # Get masks for valid sources only: (n_masks, n_valid_sources)
        valid_masks = source_masks[:, valid_idx]

        # Compute masked currents for all masks: (n_masks, n_valid_sources)
        masked_currents_all = valid_masks * valid_currents  # broadcasts

        # Aggregate using bincount for each mask (more efficient than add.at in loop)
        # bincount is O(n_valid) per mask, avoiding repeated indexing
        for m in range(n_masks):
            masked_currents = masked_currents_all[m]
            # Use bincount for fast aggregation
            if masked_currents.any():
                aggregated = np.bincount(
                    valid_unknown_idx,
                    weights=-masked_currents,
                    minlength=n_unknown
                )
                rhs_multi[:, m] += aggregated

        return total_currents

    def create_smoothed_copy(
        self,
        time_step: float,
        t_start: float,
        t_end: float,
        compact_threshold: float = 1e-12,
        chunk_size: int = 10000,
    ) -> 'VectorizedCurrentSources':
        """Create copy with smoothed waveforms (pulses converted to PWL).

        Applies analytical triangular low-pass filtering to all PWL and Pulse
        waveforms, then compacts redundant points. Pulses are converted to PWL
        format before smoothing.

        Uses chunked batch processing for high performance with controllable
        memory usage.

        Args:
            time_step: Simulation time step (filter window = 2 * time_step)
            t_start: Simulation start time
            t_end: Simulation end time
            compact_threshold: Slope change threshold for compaction
            chunk_size: Number of waveforms to process per chunk (default 10000).
                       Controls memory/speed tradeoff. Larger = faster but more memory.

        Returns:
            New VectorizedCurrentSources with smoothed waveforms
        """
        from .pwl_smoothing import PWLSmoother

        smoother = PWLSmoother(
            time_step=time_step,
            compact_threshold=compact_threshold,
        )
        cache = smoother.create_smoothed_cache(self, t_start, t_end, chunk_size=chunk_size)
        return smoother.apply_cache_to_sources(self, cache)

    @classmethod
    def from_smoothed_cache(
        cls,
        cache: 'SmoothedWaveformCache',
        original: 'VectorizedCurrentSources',
    ) -> 'VectorizedCurrentSources':
        """Create from smoothed cache, preserving DC values and node mappings.

        Args:
            cache: SmoothedWaveformCache from PWLSmoother.create_smoothed_cache
            original: Original VectorizedCurrentSources to copy DC values from

        Returns:
            New VectorizedCurrentSources with smoothed waveforms
        """
        from .pwl_smoothing import SmoothedWaveformCache

        return cls(
            n_nodes=original.n_nodes,
            node_to_idx=original.node_to_idx,
            n_sources=original.n_sources,
            dc_values=original.dc_values.copy(),
            source_node_idx=original.source_node_idx.copy(),
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
            pwl_source_idx=np.zeros(cache.n_pwls, dtype=np.int32),
            pwl_period=cache.pwl_period.copy(),
            pwl_delay=cache.pwl_delay.copy(),
            pwl_offset=cache.pwl_offset.copy(),
            pwl_count=cache.pwl_count.copy(),
            pwl_times=cache.pwl_times.copy(),
            pwl_values=cache.pwl_values.copy(),
        )

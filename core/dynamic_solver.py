"""Dynamic IR-drop solver for time-varying currents.

Provides quasi-static analysis via batch DC solves at discrete time points.
Memory-efficient: Only stores waveforms for tracked nodes, not all nodes.
"""

from __future__ import annotations

import time as time_module
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .unified_model import UnifiedPowerGridModel, GridSource
from .unified_solver import UnifiedIRDropSolver
from .solver_results import FlatSolverContext, HierarchicalSolverContext
from .vectorized_sources import VectorizedCurrentSources

# Try to import CurrentSource for type checking
try:
    from pdn.pdn_parser import CurrentSource
except ImportError:
    CurrentSource = None  # type: ignore


@dataclass
class QuasiStaticResult:
    """Result of quasi-static dynamic analysis.

    Memory-efficient: Only stores waveforms for tracked nodes, not all nodes.
    Peak statistics computed on-the-fly during solve.
    """
    t_array: np.ndarray                           # Time points
    peak_ir_drop: float                           # Max IR-drop across all nodes/times
    peak_ir_drop_time: float                      # Time of peak IR-drop
    peak_ir_drop_node: Any                        # Node with peak IR-drop
    worst_nodes: List[Tuple[Any, float, float]]   # (node, max_drop, time) top N
    nominal_voltage: float                        # Vdd
    timings: Dict[str, float]                     # Timing breakdown

    # Memory-efficient waveform storage (only tracked nodes)
    tracked_waveforms: Dict[Any, np.ndarray]      # node -> voltage array over time
    tracked_ir_drop: Dict[Any, np.ndarray]        # node -> ir_drop array over time

    # Summary statistics per time point (lightweight)
    max_ir_drop_per_time: np.ndarray              # max IR-drop at each time
    total_current_per_time: np.ndarray            # total load current at each time
    total_vsrc_current_per_time: np.ndarray       # total current through pads (voltage sources)

    # Spatial peak tracking (for heatmap plotting)
    peak_ir_drop_per_node: Dict[Any, float]       # max IR-drop each node ever reached
    peak_current_per_node: Dict[Any, float]       # max current at each node (for current heatmap)

    def get_voltage_waveform(self, node: Any) -> np.ndarray:
        """Get voltage waveform for a tracked node.

        Args:
            node: Node identifier

        Returns:
            Array of voltages at each time point.

        Raises:
            KeyError: If node is not in tracked nodes.
        """
        if node not in self.tracked_waveforms:
            raise KeyError(f"Node {node} not in tracked nodes. Use track_nodes parameter.")
        return self.tracked_waveforms[node]

    def get_ir_drop_waveform(self, node: Any) -> np.ndarray:
        """Get IR-drop waveform for a tracked node.

        Args:
            node: Node identifier

        Returns:
            Array of IR-drops at each time point.

        Raises:
            KeyError: If node is not in tracked nodes.
        """
        if node not in self.tracked_ir_drop:
            raise KeyError(f"Node {node} not in tracked nodes. Use track_nodes parameter.")
        return self.tracked_ir_drop[node]


class DynamicIRDropSolver:
    """Dynamic IR-drop solver for time-varying currents.

    Supports quasi-static analysis (batch DC solves at discrete time points).

    Example usage:
        from pdn.pdn_parser import NetlistParser
        from core import create_model_from_pdn, DynamicIRDropSolver

        parser = NetlistParser('./netlist_dir')
        graph = parser.parse()
        model = create_model_from_pdn(graph, 'VDD')

        solver = DynamicIRDropSolver(model, graph)
        result = solver.solve_quasi_static(
            t_start=0, t_end=100e-9, n_points=101
        )

        print(f"Peak IR-drop: {result.peak_ir_drop*1000:.2f} mV")
    """

    def __init__(
        self,
        model: UnifiedPowerGridModel,
        graph: Any = None,
        vectorize_threshold: int = 10000,
        clear_graph_metadata: bool = False,
    ):
        """Initialize dynamic solver.

        Args:
            model: UnifiedPowerGridModel instance
            graph: Original parsed graph with 'instance_sources' metadata.
                   If None, will try to use model.graph.
            vectorize_threshold: Use vectorized evaluation when source count
                                 exceeds this threshold. Set to 0 to always use
                                 vectorized, or -1 to never use it. Default 10000.
            clear_graph_metadata: If True and vectorized mode is used, clear the
                                  serialized instance_sources from graph metadata
                                  after vectorization to save memory. Default False.
        """
        self.model = model
        self._graph = graph if graph is not None else model.graph
        self._solver = UnifiedIRDropSolver(model)
        self._current_sources: Dict[str, Any] = {}  # name -> CurrentSource
        self._node_to_sources: Dict[Any, List[str]] = {}  # node -> [source_names]
        self._vectorize_threshold = vectorize_threshold
        self._clear_graph_metadata = clear_graph_metadata

        # Vectorized sources (initialized after loading current sources)
        self._vec_sources: Optional[VectorizedCurrentSources] = None

        # Track data source format
        self._has_raw_objects = False  # True if graph has raw CurrentSource objects

        # Get instance_sources from graph metadata (raw objects or serialized dicts)
        self._instance_sources_raw = self._get_instance_sources_data()

        # Initialize current sources - either vectorized or object-based
        self._init_current_sources()

        # Pre-compute pad-adjacent edges for efficient vsrc current calculation
        self._pad_edges = self._build_pad_edges()

    def _get_instance_sources_data(self) -> Dict[str, Any]:
        """Get instance_sources from graph metadata.

        Checks for raw CurrentSource objects first ('_instance_sources_objects'),
        then falls back to serialized dicts ('instance_sources').

        Returns:
            Dict mapping name -> CurrentSource object or serialized dict
        """
        graph_obj = self._graph
        graph_dict = None
        if hasattr(graph_obj, 'graph') and isinstance(graph_obj.graph, dict):
            graph_dict = graph_obj.graph
        elif hasattr(graph_obj, '_attrs'):
            graph_dict = graph_obj._attrs

        if graph_dict is None:
            return {}

        # Check for raw CurrentSource objects first (memory-efficient storage)
        if '_instance_sources_objects' in graph_dict:
            self._has_raw_objects = True
            return graph_dict.get('_instance_sources_objects', {})

        # Fall back to serialized dict format (backward compat)
        return graph_dict.get('instance_sources', {})

    def _init_current_sources(self) -> None:
        """Initialize current sources - vectorized or object-based.

        For large source counts (>= threshold), converts to vectorized format.
        Handles both raw CurrentSource objects and serialized dicts.
        """
        n_sources = len(self._instance_sources_raw)
        threshold = self._vectorize_threshold

        # Determine if we should use vectorized mode
        use_vectorized = (
            threshold >= 0 and
            (threshold == 0 or n_sources >= threshold)
        )

        if use_vectorized and n_sources > 0:
            edge_cache = self.model.edge_cache

            if self._has_raw_objects:
                # Convert directly from CurrentSource objects
                self._vec_sources = VectorizedCurrentSources.from_current_sources(
                    self._instance_sources_raw,
                    edge_cache.node_to_idx,
                    edge_cache.n_nodes,
                )
            else:
                # Parse from serialized dicts
                self._vec_sources = VectorizedCurrentSources.from_serialized_dicts(
                    self._instance_sources_raw,
                    edge_cache.node_to_idx,
                    edge_cache.n_nodes,
                )
            # Don't need object-based storage when using vectorized
            self._current_sources = {}
            self._node_to_sources = {}

            # Clear graph metadata to save memory (optional)
            if self._clear_graph_metadata:
                self._clear_instance_sources_from_graph()
        else:
            # Fall back to object-based storage for sequential evaluation
            self._load_current_sources_as_objects()

        # Clear local reference to raw data (no longer needed)
        self._instance_sources_raw = {}

    def _clear_instance_sources_from_graph(self) -> None:
        """Clear instance_sources from graph metadata to save memory."""
        graph_obj = self._graph
        graph_dict = None
        if hasattr(graph_obj, 'graph') and isinstance(graph_obj.graph, dict):
            graph_dict = graph_obj.graph
        elif hasattr(graph_obj, '_attrs'):
            graph_dict = graph_obj._attrs

        if graph_dict is not None:
            # Clear both possible keys
            graph_dict.pop('instance_sources', None)
            graph_dict.pop('_instance_sources_objects', None)

    def _load_current_sources_as_objects(self) -> None:
        """Load current sources as CurrentSource objects (sequential mode).

        Handles both raw CurrentSource objects and serialized dicts.
        """
        if self._has_raw_objects:
            # Already have CurrentSource objects - use directly
            for name, src in self._instance_sources_raw.items():
                self._current_sources[name] = src
                node1 = src.node1
                if node1 and node1 != '0':
                    if node1 not in self._node_to_sources:
                        self._node_to_sources[node1] = []
                    self._node_to_sources[node1].append(name)
        else:
            # Reconstruct from serialized dicts
            try:
                from pdn.pdn_parser import CurrentSource as CS
            except ImportError:
                return

            for name, data in self._instance_sources_raw.items():
                src = CS.from_dict(data)
                self._current_sources[name] = src

                node1 = src.node1
                if node1 and node1 != '0':
                    if node1 not in self._node_to_sources:
                        self._node_to_sources[node1] = []
                    self._node_to_sources[node1].append(name)

    def _evaluate_currents_at_time(self, t: float) -> Dict[Any, float]:
        """Evaluate all current sources at a given time.

        Args:
            t: Time in seconds

        Returns:
            Dict mapping node -> current (positive = sink, in mA for PDN)
        """
        # Use vectorized evaluation if available (much faster for large source counts)
        if self._vec_sources is not None:
            edge_cache = self.model.edge_cache
            return self._vec_sources.evaluate_at_time_as_dict(t, edge_cache.idx_to_node)

        # Use object-based sequential evaluation
        if self._current_sources:
            current_injections: Dict[Any, float] = {}
            valid_nodes = self.model.valid_nodes

            for name, src in self._current_sources.items():
                current_ma = src.get_current_at_time(t)
                if current_ma == 0:
                    continue

                node = src.node1
                if node and node in valid_nodes and node != '0':
                    current_injections[node] = current_injections.get(node, 0.0) + current_ma

            return current_injections

        # Fall back to static current sources from model
        return self.model.extract_current_sources()

        return current_injections

    def _build_pad_edges(self) -> List[Tuple[Any, Any, float]]:
        """Pre-compute edges adjacent to pads for efficient vsrc current calculation.

        Returns:
            List of (other_node, conductance, sign) tuples where:
            - other_node: The non-pad endpoint
            - conductance: Edge conductance (1/R)
            - sign: +1 if current flows pad->other, -1 if other->pad
        """
        pad_set = set(self.model.pad_nodes)
        GMAX = 1e5
        SHORT_THRESHOLD = 1e-6
        edges: List[Tuple[Any, Any, float]] = []

        for u, v, edge_info in self.model._iter_resistive_edges():
            R = edge_info.resistance
            if R is None:
                continue

            # Compute conductance
            if R <= 0 or R < SHORT_THRESHOLD:
                g = GMAX
            else:
                g = 1.0 / R

            u_is_pad = u in pad_set
            v_is_pad = v in pad_set

            # Only store edges where exactly one end is a pad
            if u_is_pad and not v_is_pad and v != '0':
                # Current from pad u to non-pad v: I = g * (Vdd - V_v)
                edges.append((v, g, 1.0))
            elif v_is_pad and not u_is_pad and u != '0':
                # Current from pad v to non-pad u: I = g * (Vdd - V_u)
                edges.append((u, g, 1.0))

        return edges

    def _compute_vsrc_current(
        self,
        voltages: Dict[Any, float],
    ) -> float:
        """Compute total current flowing through voltage sources (pads).

        Uses pre-computed pad-adjacent edges for O(E_pad) instead of O(E).

        Args:
            voltages: Node voltages from solve

        Returns:
            Total current in mA (for PDN) or A (for synthetic)
        """
        total = 0.0
        vdd = self.model.vdd

        for other_node, g, sign in self._pad_edges:
            v_other = voltages.get(other_node, vdd)
            total += sign * g * (vdd - v_other)

        return total

    def solve_quasi_static(
        self,
        t_start: float = 0.0,
        t_end: float = 100e-9,
        n_points: int = 101,
        t_array: Optional[np.ndarray] = None,
        method: str = 'flat',
        partition_layer: Optional[Union[str, int]] = None,
        top_k: int = 5,
        weighting: str = 'shortest_path',
        verbose: bool = False,
        n_worst_nodes: int = 10,
        track_nodes: Optional[List[Any]] = None,
        use_legacy: bool = False,
    ) -> QuasiStaticResult:
        """Quasi-static analysis via batch DC solves.

        Memory-efficient: Statistics computed on-the-fly. Full waveforms only
        stored for nodes in track_nodes list. If track_nodes is None, only
        stores waveforms for worst_nodes discovered during solve.

        Args:
            t_start: Start time in seconds
            t_end: End time in seconds
            n_points: Number of time points (ignored if t_array provided)
            t_array: Optional explicit time array (overrides t_start/t_end/n_points)
            method: Solving method - 'flat' or 'hierarchical'
            partition_layer: Layer for hierarchical partition (required if method='hierarchical')
            top_k: Number of nearest ports per load for hierarchical solving
            weighting: Weighting method for hierarchical - 'shortest_path' or 'effective'
            verbose: Print progress
            n_worst_nodes: Track N worst-case nodes
            track_nodes: Nodes to store full waveforms for (None = worst nodes only)
            use_legacy: If True, use legacy dict-based code path (for validation).
                        If False (default), use optimized array-based operations.

        Returns:
            QuasiStaticResult with peak statistics and optional waveforms.
        """
        if use_legacy:
            return self._solve_quasi_static_legacy(
                t_start, t_end, n_points, t_array, method, partition_layer,
                top_k, weighting, verbose, n_worst_nodes, track_nodes
            )
        return self._solve_quasi_static_array(
            t_start, t_end, n_points, t_array, method, partition_layer,
            top_k, weighting, verbose, n_worst_nodes, track_nodes
        )

    def _solve_quasi_static_array(
        self,
        t_start: float,
        t_end: float,
        n_points: int,
        t_array: Optional[np.ndarray],
        method: str,
        partition_layer: Optional[Union[str, int]],
        top_k: int,
        weighting: str,
        verbose: bool,
        n_worst_nodes: int,
        track_nodes: Optional[List[Any]],
    ) -> QuasiStaticResult:
        """Optimized array-based quasi-static solve.

        Uses numpy arrays instead of dicts for peak tracking and statistics.
        """
        timings: Dict[str, float] = {}
        t0_total = time_module.perf_counter()

        # Build time array
        if t_array is None:
            t_array = np.linspace(t_start, t_end, n_points)
        n_steps = len(t_array)

        vdd = self.model.vdd

        # Prepare solver context
        t0_prep = time_module.perf_counter()
        if method == 'flat':
            context = self._solver.prepare_flat()
        elif method == 'hierarchical':
            if partition_layer is None:
                raise ValueError("partition_layer required for hierarchical method")
            context = self._solver.prepare_hierarchical(
                partition_layer=partition_layer,
                top_k=top_k,
                weighting=weighting,
            )
        else:
            raise ValueError(f"Unknown method: {method}. Use 'flat' or 'hierarchical'.")
        timings['prepare'] = time_module.perf_counter() - t0_prep

        # Get unknown nodes from context for array-based operations
        if method == 'flat':
            reduced = context.reduced_system
            unknown_nodes = reduced.unknown_nodes
            unknown_to_idx = reduced.index_unknown
        else:
            # For hierarchical, we'll fall back to dict-based peak tracking
            unknown_nodes = list(self.model.valid_nodes - set(self.model.pad_nodes) - {'0'})
            unknown_to_idx = {n: i for i, n in enumerate(unknown_nodes)}

        n_unknown = len(unknown_nodes)

        # Pre-allocate summary arrays (per time step)
        max_ir_drop_per_time = np.zeros(n_steps, dtype=np.float64)
        total_current_per_time = np.zeros(n_steps, dtype=np.float64)
        total_vsrc_current_per_time = np.zeros(n_steps, dtype=np.float64)

        # Pre-allocate spatial peak arrays (per node)
        peak_ir_drop_arr = np.full(n_unknown, -np.inf, dtype=np.float64)
        peak_time_arr = np.zeros(n_unknown, dtype=np.float64)
        peak_current_arr = np.zeros(n_unknown, dtype=np.float64)

        # Global peak tracking
        global_peak_drop = 0.0
        global_peak_time = t_array[0]
        global_peak_idx = 0

        # Build tracked node index mapping
        track_set = set(track_nodes) if track_nodes else set()
        tracked_indices: List[int] = []
        tracked_nodes_list: List[Any] = []
        for node in track_set:
            if node in unknown_to_idx:
                tracked_indices.append(unknown_to_idx[node])
                tracked_nodes_list.append(node)

        # Pre-allocate tracked waveforms array
        n_tracked = len(tracked_indices)
        tracked_voltages_arr = np.zeros((n_tracked, n_steps), dtype=np.float64) if n_tracked > 0 else None
        tracked_idx_arr = np.array(tracked_indices, dtype=np.int32) if n_tracked > 0 else None

        # Time-stepping loop
        t0_solve = time_module.perf_counter()
        time_evaluate_currents = 0.0
        time_dc_solve = 0.0
        time_statistics = 0.0
        time_vsrc_current = 0.0
        time_spatial_peaks = 0.0
        time_tracked_waveforms = 0.0

        for i, t in enumerate(t_array):
            if verbose and i % max(1, n_steps // 10) == 0:
                print(f"  Time step {i}/{n_steps} (t={t*1e9:.2f} ns)")

            # Evaluate currents at this time
            t0_eval = time_module.perf_counter()
            currents = self._evaluate_currents_at_time(t)
            total_current = sum(currents.values())
            total_current_per_time[i] = total_current
            time_evaluate_currents += time_module.perf_counter() - t0_eval

            # Solve DC
            t0_dc = time_module.perf_counter()
            if method == 'flat':
                result = self._solver.solve_prepared(currents, context)
            else:
                result = self._solver.solve_hierarchical_prepared(currents, context)

            voltages = result.voltages
            ir_drop = result.ir_drop
            time_dc_solve += time_module.perf_counter() - t0_dc

            # Compute max IR-drop at this time (array-based)
            t0_stats = time_module.perf_counter()
            if ir_drop:
                # Build IR-drop array from result dict
                ir_drop_arr = np.zeros(n_unknown, dtype=np.float64)
                for node, drop in ir_drop.items():
                    if node in unknown_to_idx:
                        ir_drop_arr[unknown_to_idx[node]] = drop

                max_drop = ir_drop_arr.max()
                max_drop_idx = ir_drop_arr.argmax()
                max_ir_drop_per_time[i] = max_drop

                if max_drop > global_peak_drop:
                    global_peak_drop = max_drop
                    global_peak_time = t
                    global_peak_idx = max_drop_idx
            else:
                max_drop = 0.0
                ir_drop_arr = np.zeros(n_unknown, dtype=np.float64)
                max_ir_drop_per_time[i] = 0.0
            time_statistics += time_module.perf_counter() - t0_stats

            # Compute vsrc current
            t0_vsrc = time_module.perf_counter()
            total_vsrc_current_per_time[i] = self._compute_vsrc_current(voltages)
            time_vsrc_current += time_module.perf_counter() - t0_vsrc

            # Update spatial peak tracking (vectorized)
            t0_spatial = time_module.perf_counter()
            update_mask = ir_drop_arr > peak_ir_drop_arr
            peak_ir_drop_arr[update_mask] = ir_drop_arr[update_mask]
            peak_time_arr[update_mask] = t

            # Update peak current per node
            for node, curr in currents.items():
                if node in unknown_to_idx:
                    idx = unknown_to_idx[node]
                    if abs(curr) > abs(peak_current_arr[idx]):
                        peak_current_arr[idx] = curr
            time_spatial_peaks += time_module.perf_counter() - t0_spatial

            # Store tracked waveforms (vectorized gather)
            t0_track = time_module.perf_counter()
            if tracked_voltages_arr is not None:
                for local_idx, node in enumerate(tracked_nodes_list):
                    tracked_voltages_arr[local_idx, i] = voltages.get(node, vdd)
            time_tracked_waveforms += time_module.perf_counter() - t0_track

        timings['time_stepping'] = time_module.perf_counter() - t0_solve
        timings['solve'] = time_dc_solve
        timings['evaluate_currents'] = time_evaluate_currents
        timings['statistics'] = time_statistics
        timings['vsrc_current'] = time_vsrc_current
        timings['spatial_peaks'] = time_spatial_peaks
        timings['tracked_waveforms'] = time_tracked_waveforms

        # Convert arrays to dicts for result construction
        t0_convert = time_module.perf_counter()

        # Global peak node
        global_peak_node = unknown_nodes[global_peak_idx] if n_unknown > 0 else None

        # Build node_max_drops for worst nodes
        node_max_drops: Dict[Any, Tuple[float, float]] = {}
        for idx, node in enumerate(unknown_nodes):
            if peak_ir_drop_arr[idx] > -np.inf:
                node_max_drops[node] = (peak_ir_drop_arr[idx], peak_time_arr[idx])

        worst_nodes_list = sorted(
            node_max_drops.items(),
            key=lambda x: x[1][0],
            reverse=True
        )[:n_worst_nodes]
        worst_nodes = [(node, drop, time_val) for node, (drop, time_val) in worst_nodes_list]

        # Build peak_ir_drop_per_node dict
        peak_ir_drop_per_node: Dict[Any, float] = {}
        for idx, node in enumerate(unknown_nodes):
            peak_ir_drop_per_node[node] = peak_ir_drop_arr[idx]
        for node in self.model.pad_nodes:
            peak_ir_drop_per_node[node] = 0.0

        # Build peak_current_per_node dict
        peak_current_per_node: Dict[Any, float] = {}
        for idx, node in enumerate(unknown_nodes):
            if peak_current_arr[idx] != 0:
                peak_current_per_node[node] = peak_current_arr[idx]

        # Build tracked waveform dicts
        tracked_waveforms: Dict[Any, np.ndarray] = {}
        tracked_ir_drop: Dict[Any, np.ndarray] = {}
        if tracked_voltages_arr is not None:
            for local_idx, node in enumerate(tracked_nodes_list):
                tracked_waveforms[node] = tracked_voltages_arr[local_idx, :].copy()
                tracked_ir_drop[node] = vdd - tracked_waveforms[node]

        timings['convert_results'] = time_module.perf_counter() - t0_convert
        timings['total'] = time_module.perf_counter() - t0_total

        return QuasiStaticResult(
            t_array=t_array,
            peak_ir_drop=global_peak_drop,
            peak_ir_drop_time=global_peak_time,
            peak_ir_drop_node=global_peak_node,
            worst_nodes=worst_nodes,
            nominal_voltage=vdd,
            timings=timings,
            tracked_waveforms=tracked_waveforms,
            tracked_ir_drop=tracked_ir_drop,
            max_ir_drop_per_time=max_ir_drop_per_time,
            total_current_per_time=total_current_per_time,
            total_vsrc_current_per_time=total_vsrc_current_per_time,
            peak_ir_drop_per_node=peak_ir_drop_per_node,
            peak_current_per_node=peak_current_per_node,
        )

    def _solve_quasi_static_legacy(
        self,
        t_start: float,
        t_end: float,
        n_points: int,
        t_array: Optional[np.ndarray],
        method: str,
        partition_layer: Optional[Union[str, int]],
        top_k: int,
        weighting: str,
        verbose: bool,
        n_worst_nodes: int,
        track_nodes: Optional[List[Any]],
    ) -> QuasiStaticResult:
        """Legacy dict-based quasi-static solve (for validation)."""
        timings: Dict[str, float] = {}
        t0_total = time_module.perf_counter()

        # Build time array
        if t_array is None:
            t_array = np.linspace(t_start, t_end, n_points)
        n_steps = len(t_array)

        # Pre-allocate summary arrays
        max_ir_drop_per_time = np.zeros(n_steps)
        total_current_per_time = np.zeros(n_steps)
        total_vsrc_current_per_time = np.zeros(n_steps)

        # Track spatial peaks for heatmaps
        peak_ir_drop_per_node: Dict[Any, float] = {}
        peak_current_per_node: Dict[Any, float] = {}

        # Track worst N nodes with timestamps: node -> (max_drop, time_of_max)
        node_max_drops: Dict[Any, Tuple[float, float]] = {}

        # Global peak tracking
        global_peak_drop = 0.0
        global_peak_time = t_array[0]
        global_peak_node: Any = None

        # Prepare track set (we'll add worst nodes at the end)
        track_set = set(track_nodes) if track_nodes else set()

        # Temporary storage for waveforms (will be trimmed to track_set + worst nodes)
        temp_voltages: Dict[Any, List[float]] = {n: [] for n in track_set}

        # Prepare solver context for efficiency
        t0_prep = time_module.perf_counter()
        if method == 'flat':
            context = self._solver.prepare_flat()
        elif method == 'hierarchical':
            if partition_layer is None:
                raise ValueError("partition_layer required for hierarchical method")
            context = self._solver.prepare_hierarchical(
                partition_layer=partition_layer,
                top_k=top_k,
                weighting=weighting,
            )
        else:
            raise ValueError(f"Unknown method: {method}. Use 'flat' or 'hierarchical'.")
        timings['prepare'] = time_module.perf_counter() - t0_prep

        # Time-stepping loop
        t0_solve = time_module.perf_counter()
        vdd = self.model.vdd
        
        # Timing accumulators for per-step operations
        time_evaluate_currents = 0.0
        time_dc_solve = 0.0
        time_statistics = 0.0
        time_vsrc_current = 0.0
        time_spatial_peaks = 0.0
        time_tracked_waveforms = 0.0

        for i, t in enumerate(t_array):
            if verbose and i % max(1, n_steps // 10) == 0:
                print(f"  Time step {i}/{n_steps} (t={t*1e9:.2f} ns)")

            # Evaluate currents at this time
            t0_eval = time_module.perf_counter()
            currents = self._evaluate_currents_at_time(t)
            total_current = sum(currents.values())
            total_current_per_time[i] = total_current
            time_evaluate_currents += time_module.perf_counter() - t0_eval

            # Solve DC
            t0_dc = time_module.perf_counter()
            if method == 'flat':
                result = self._solver.solve_prepared(currents, context)
            else:
                result = self._solver.solve_hierarchical_prepared(currents, context)

            voltages = result.voltages
            ir_drop = result.ir_drop
            time_dc_solve += time_module.perf_counter() - t0_dc

            # Compute max IR-drop at this time
            t0_stats = time_module.perf_counter()
            if ir_drop:
                max_drop = max(ir_drop.values())
                max_drop_node = max(ir_drop, key=ir_drop.get)
            else:
                max_drop = 0.0
                max_drop_node = None

            max_ir_drop_per_time[i] = max_drop

            # Update global peak
            if max_drop > global_peak_drop:
                global_peak_drop = max_drop
                global_peak_time = t
                global_peak_node = max_drop_node
            time_statistics += time_module.perf_counter() - t0_stats

            # Compute vsrc current
            t0_vsrc = time_module.perf_counter()
            total_vsrc_current_per_time[i] = self._compute_vsrc_current(voltages)
            time_vsrc_current += time_module.perf_counter() - t0_vsrc

            # Update spatial peak tracking (per node)
            t0_spatial = time_module.perf_counter()
            for node, drop in ir_drop.items():
                if node not in peak_ir_drop_per_node or drop > peak_ir_drop_per_node[node]:
                    peak_ir_drop_per_node[node] = drop
                if node not in node_max_drops or drop > node_max_drops[node][0]:
                    node_max_drops[node] = (drop, t)

            # Update peak current per node
            for node, curr in currents.items():
                if node not in peak_current_per_node or abs(curr) > abs(peak_current_per_node[node]):
                    peak_current_per_node[node] = curr
            time_spatial_peaks += time_module.perf_counter() - t0_spatial

            # Store waveforms for tracked nodes
            t0_track = time_module.perf_counter()
            for node in track_set:
                if node in voltages:
                    temp_voltages[node].append(voltages[node])
                else:
                    temp_voltages[node].append(vdd)  # Default to Vdd if not found
            time_tracked_waveforms += time_module.perf_counter() - t0_track

        timings['time_stepping'] = time_module.perf_counter() - t0_solve
        timings['solve'] = time_dc_solve
        timings['evaluate_currents'] = time_evaluate_currents
        timings['statistics'] = time_statistics
        timings['vsrc_current'] = time_vsrc_current
        timings['spatial_peaks'] = time_spatial_peaks
        timings['tracked_waveforms'] = time_tracked_waveforms

        # Determine worst nodes
        worst_nodes_list = sorted(
            node_max_drops.items(),
            key=lambda x: x[1][0],
            reverse=True
        )[:n_worst_nodes]
        worst_nodes = [(node, drop, time_val) for node, (drop, time_val) in worst_nodes_list]

        # Build final waveform storage
        tracked_waveforms: Dict[Any, np.ndarray] = {}
        tracked_ir_drop: Dict[Any, np.ndarray] = {}

        for node in track_set:
            if node in temp_voltages and len(temp_voltages[node]) == n_steps:
                tracked_waveforms[node] = np.array(temp_voltages[node])
                tracked_ir_drop[node] = vdd - tracked_waveforms[node]

        timings['total'] = time_module.perf_counter() - t0_total

        return QuasiStaticResult(
            t_array=t_array,
            peak_ir_drop=global_peak_drop,
            peak_ir_drop_time=global_peak_time,
            peak_ir_drop_node=global_peak_node,
            worst_nodes=worst_nodes,
            nominal_voltage=vdd,
            timings=timings,
            tracked_waveforms=tracked_waveforms,
            tracked_ir_drop=tracked_ir_drop,
            max_ir_drop_per_time=max_ir_drop_per_time,
            total_current_per_time=total_current_per_time,
            total_vsrc_current_per_time=total_vsrc_current_per_time,
            peak_ir_drop_per_node=peak_ir_drop_per_node,
            peak_current_per_node=peak_current_per_node,
        )

    @property
    def has_dynamic_sources(self) -> bool:
        """Check if any current sources have time-varying waveforms."""
        for src in self._current_sources.values():
            if hasattr(src, 'has_waveform_data') and src.has_waveform_data():
                return True
        return False

    @property
    def num_current_sources(self) -> int:
        """Get number of current sources."""
        return len(self._current_sources)

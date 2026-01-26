"""Transient IR-drop solver with RC support.

Provides time-domain simulation incorporating capacitance for accurate
decoupling effects. Uses Backward Euler or Trapezoidal time integration.

Inductance support is deferred to a later phase.
"""

from __future__ import annotations

import time as time_module
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .unified_model import UnifiedPowerGridModel, GridSource
from .edge_adapter import ElementType

# Try to import CurrentSource for type checking
try:
    from pdn.pdn_parser import CurrentSource
except ImportError:
    CurrentSource = None  # type: ignore


class IntegrationMethod(Enum):
    """Time integration methods for transient analysis."""
    BACKWARD_EULER = 'be'   # First-order implicit (stable, damped)
    TRAPEZOIDAL = 'trap'    # Second-order implicit (more accurate)


@dataclass
class TransientResult:
    """Result of transient analysis.

    Memory-efficient: Only stores waveforms for tracked nodes.
    Statistics computed on-the-fly during time stepping.
    """
    t_array: np.ndarray                           # Time points
    peak_ir_drop: float                           # Max IR-drop across all nodes/times
    peak_ir_drop_time: float                      # Time of peak IR-drop
    peak_ir_drop_node: Any                        # Node with peak IR-drop
    worst_nodes: List[Tuple[Any, float, float]]   # (node, max_drop, time) top N
    nominal_voltage: float                        # Vdd
    integration_method: IntegrationMethod         # Method used
    timings: Dict[str, float]                     # Timing breakdown

    # Memory-efficient waveform storage (only tracked nodes)
    tracked_waveforms: Dict[Any, np.ndarray]      # node -> voltage array over time
    tracked_ir_drop: Dict[Any, np.ndarray]        # node -> ir_drop array over time

    # Summary statistics per time point (lightweight)
    max_ir_drop_per_time: np.ndarray              # max IR-drop at each time
    total_current_per_time: np.ndarray            # total load current at each time
    total_vsrc_current_per_time: np.ndarray       # total current through pads

    # Spatial peak tracking (for heatmap plotting)
    peak_ir_drop_per_node: Dict[Any, float]       # max IR-drop each node ever reached
    peak_current_per_node: Dict[Any, float]       # max current at each node

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


@dataclass
class RCSystem:
    """RC system matrices for transient analysis.

    Contains the conductance (G) and capacitance (C) matrices for the
    reduced system (after eliminating pad nodes via Schur complement).
    """
    G_full: sp.csr_matrix          # Full conductance matrix (all nodes except ground)
    C_full: sp.csr_matrix          # Full capacitance matrix (all nodes except ground)
    G_uu: sp.csr_matrix            # Reduced G (unknown nodes only)
    G_up: sp.csr_matrix            # G cross-terms (unknown to pad)
    C_uu: sp.csr_matrix            # Reduced C (unknown nodes only)
    node_order: List[Any]          # Node ordering (all except ground)
    node_to_idx: Dict[Any, int]    # Node to index mapping
    unknown_nodes: List[Any]       # Unknown (non-pad) nodes
    unknown_to_idx: Dict[Any, int] # Unknown node to index mapping
    pad_nodes: List[Any]           # Pad (Dirichlet) nodes
    n_nodes: int                   # Total nodes (excluding ground)
    n_unknown: int                 # Number of unknown nodes


class TransientIRDropSolver:
    """Transient solver with RC support.

    Solves time-domain IR-drop using implicit time integration methods.
    Capacitance provides smoothing/decoupling effects on voltage response.

    Example usage:
        from core import TransientIRDropSolver, IntegrationMethod

        solver = TransientIRDropSolver(model, graph)
        result = solver.solve_transient(
            t_start=0, t_end=100e-9, dt=0.1e-9,
            method=IntegrationMethod.BACKWARD_EULER
        )

        print(f"Peak IR-drop: {result.peak_ir_drop*1000:.2f} mV")
    """

    def __init__(self, model: UnifiedPowerGridModel, graph: Any = None):
        """Initialize transient solver.

        Args:
            model: UnifiedPowerGridModel instance
            graph: Original parsed graph with 'instance_sources' metadata.
                   If None, will try to use model.graph.
        """
        self.model = model
        self._graph = graph if graph is not None else model.graph
        self._current_sources: Dict[str, Any] = {}
        self._node_to_sources: Dict[Any, List[str]] = {}

        # RC system (lazy initialization)
        self._rc_system: Optional[RCSystem] = None

        # Load current source data
        self._load_current_sources()

    def _load_current_sources(self) -> None:
        """Load current source data from graph metadata."""
        try:
            from pdn.pdn_parser import CurrentSource as CS
        except ImportError:
            return

        graph_obj = self._graph
        instance_sources = {}

        if hasattr(graph_obj, 'graph') and isinstance(graph_obj.graph, dict):
            instance_sources = graph_obj.graph.get('instance_sources', {})
        elif hasattr(graph_obj, '_attrs'):
            instance_sources = graph_obj._attrs.get('instance_sources', {})

        for name, data in instance_sources.items():
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
        if not self._current_sources:
            return self.model.extract_current_sources()

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

    def _iter_capacitive_edges(self):
        """Iterate over capacitive edges in the graph.

        Yields:
            Tuples of (u, v, edge_info) for each capacitive edge.
        """
        from .rx_graph import RustworkxMultiDiGraphWrapper

        if isinstance(self.model.graph, RustworkxMultiDiGraphWrapper):
            # PDN MultiDiGraph: iterate over all edges, filter C type
            for u, v, k, data in self.model.graph.edges(keys=True, data=True):
                if data.get('type') == 'C':
                    edge_info = self.model._edge_extractor.get_info(data)
                    yield u, v, edge_info
        # Synthetic grids don't have capacitors

    def _build_rc_system(self) -> RCSystem:
        """Build G and C matrices from model graph.

        Returns:
            RCSystem with conductance and capacitance matrices.
        """
        # Get nodes (excluding ground '0')
        valid_nodes = self.model.valid_nodes
        nodes = [n for n in valid_nodes if n != '0']
        node_to_idx = {n: i for i, n in enumerate(nodes)}
        n_nodes = len(nodes)

        pad_set = set(self.model.pad_nodes)
        unknown_nodes = [n for n in nodes if n not in pad_set]
        pad_nodes = [n for n in nodes if n in pad_set]
        unknown_to_idx = {n: i for i, n in enumerate(unknown_nodes)}
        n_unknown = len(unknown_nodes)
        n_pads = len(pad_nodes)

        # Constants for handling shorts
        GMAX = 1e5
        SHORT_THRESHOLD = 1e-6

        # Build full G matrix (same as _build_conductance_matrix)
        g_data, g_rows, g_cols = [], [], []
        g_diag = np.zeros(n_nodes, dtype=float)

        for u, v, edge_info in self.model._iter_resistive_edges():
            R = edge_info.resistance
            if R is None:
                continue

            if R <= 0 or R < SHORT_THRESHOLD:
                g = GMAX
            else:
                g = 1.0 / R

            u_is_ground = (u == '0')
            v_is_ground = (v == '0')

            if u_is_ground and v_is_ground:
                continue
            elif u_is_ground:
                if v in node_to_idx:
                    g_diag[node_to_idx[v]] += g
            elif v_is_ground:
                if u in node_to_idx:
                    g_diag[node_to_idx[u]] += g
            else:
                if u not in node_to_idx or v not in node_to_idx:
                    continue
                iu, iv = node_to_idx[u], node_to_idx[v]
                g_rows.extend([iu, iv])
                g_cols.extend([iv, iu])
                g_data.extend([-g, -g])
                g_diag[iu] += g
                g_diag[iv] += g

        # Add diagonal entries
        for i in range(n_nodes):
            g_rows.append(i)
            g_cols.append(i)
            g_data.append(g_diag[i])

        G_full = sp.csr_matrix((g_data, (g_rows, g_cols)), shape=(n_nodes, n_nodes))

        # Build full C matrix
        c_data, c_rows, c_cols = [], [], []
        c_diag = np.zeros(n_nodes, dtype=float)

        for u, v, edge_info in self._iter_capacitive_edges():
            C_val = edge_info.capacitance
            if C_val is None or C_val <= 0:
                continue

            u_is_ground = (u == '0')
            v_is_ground = (v == '0')

            if u_is_ground and v_is_ground:
                continue
            elif u_is_ground:
                # Capacitor from ground to v: C to ground
                if v in node_to_idx:
                    c_diag[node_to_idx[v]] += C_val
            elif v_is_ground:
                # Capacitor from u to ground: C to ground
                if u in node_to_idx:
                    c_diag[node_to_idx[u]] += C_val
            else:
                # Floating capacitor between two non-ground nodes
                if u not in node_to_idx or v not in node_to_idx:
                    continue
                iu, iv = node_to_idx[u], node_to_idx[v]
                c_rows.extend([iu, iv])
                c_cols.extend([iv, iu])
                c_data.extend([-C_val, -C_val])
                c_diag[iu] += C_val
                c_diag[iv] += C_val

        # Add diagonal entries
        for i in range(n_nodes):
            if c_diag[i] > 0:  # Only add if there's capacitance
                c_rows.append(i)
                c_cols.append(i)
                c_data.append(c_diag[i])

        C_full = sp.csr_matrix((c_data, (c_rows, c_cols)), shape=(n_nodes, n_nodes))

        # Build reduced matrices (unknown nodes only)
        # Partition into unknown and pad indices
        u_indices = np.array([node_to_idx[n] for n in unknown_nodes], dtype=int)
        p_indices = np.array([node_to_idx[n] for n in pad_nodes], dtype=int)

        if n_unknown > 0 and n_pads > 0:
            G_uu = G_full[np.ix_(u_indices, u_indices)]
            G_up = G_full[np.ix_(u_indices, p_indices)]
            C_uu = C_full[np.ix_(u_indices, u_indices)]
        elif n_unknown > 0:
            G_uu = G_full[np.ix_(u_indices, u_indices)]
            G_up = sp.csr_matrix((n_unknown, 0))
            C_uu = C_full[np.ix_(u_indices, u_indices)]
        else:
            G_uu = sp.csr_matrix((0, 0))
            G_up = sp.csr_matrix((0, n_pads))
            C_uu = sp.csr_matrix((0, 0))

        return RCSystem(
            G_full=G_full,
            C_full=C_full,
            G_uu=G_uu,
            G_up=G_up,
            C_uu=C_uu,
            node_order=nodes,
            node_to_idx=node_to_idx,
            unknown_nodes=unknown_nodes,
            unknown_to_idx=unknown_to_idx,
            pad_nodes=pad_nodes,
            n_nodes=n_nodes,
            n_unknown=n_unknown,
        )

    def _ensure_rc_system(self) -> RCSystem:
        """Get or build RC system."""
        if self._rc_system is None:
            self._rc_system = self._build_rc_system()
        return self._rc_system

    def _compute_vsrc_current(self, V_dict: Dict[Any, float]) -> float:
        """Compute total current flowing through voltage sources (pads).

        Args:
            V_dict: Node voltages

        Returns:
            Total current through pads (mA for PDN, A for synthetic)
        """
        total = 0.0
        pad_set = set(self.model.pad_nodes)
        vdd = self.model.vdd

        GMAX = 1e5
        SHORT_THRESHOLD = 1e-6

        for u, v, edge_info in self.model._iter_resistive_edges():
            R = edge_info.resistance
            if R is None:
                continue

            if R <= 0 or R < SHORT_THRESHOLD:
                g = GMAX
            else:
                g = 1.0 / R

            u_is_pad = u in pad_set
            v_is_pad = v in pad_set

            if u_is_pad and not v_is_pad and v != '0':
                v_u = vdd
                v_v = V_dict.get(v, vdd)
                total += g * (v_u - v_v)
            elif v_is_pad and not u_is_pad and u != '0':
                v_v = vdd
                v_u = V_dict.get(u, vdd)
                total += g * (v_v - v_u)

        return total

    def solve_transient(
        self,
        t_start: float = 0.0,
        t_end: float = 100e-9,
        dt: float = 0.1e-9,
        method: IntegrationMethod = IntegrationMethod.BACKWARD_EULER,
        verbose: bool = False,
        n_worst_nodes: int = 10,
        track_nodes: Optional[List[Any]] = None,
    ) -> TransientResult:
        """Run transient RC simulation.

        Memory-efficient: Statistics computed on-the-fly. Full waveforms only
        stored for nodes in track_nodes list.

        Args:
            t_start: Start time in seconds
            t_end: End time in seconds
            dt: Time step in seconds
            method: Integration method (BACKWARD_EULER or TRAPEZOIDAL)
            verbose: Print progress
            n_worst_nodes: Track N worst-case nodes
            track_nodes: Nodes to store full waveforms for (None = worst nodes only)

        Returns:
            TransientResult with voltages, IR-drop, and timing.
        """
        timings: Dict[str, float] = {}
        t0_total = time_module.perf_counter()

        # Build time array
        t_array = np.arange(t_start, t_end + dt * 0.5, dt)
        n_steps = len(t_array)

        # Build RC system
        t0_build = time_module.perf_counter()
        rc = self._ensure_rc_system()
        timings['build_rc'] = time_module.perf_counter() - t0_build

        vdd = self.model.vdd
        n_unknown = rc.n_unknown

        if n_unknown == 0:
            # All nodes are pads - no transient needed
            return self._create_trivial_result(t_array, vdd, method, timings)

        # Pre-allocate summary arrays
        max_ir_drop_per_time = np.zeros(n_steps)
        total_current_per_time = np.zeros(n_steps)
        total_vsrc_current_per_time = np.zeros(n_steps)

        # Track spatial peaks
        peak_ir_drop_per_node: Dict[Any, float] = {}
        peak_current_per_node: Dict[Any, float] = {}
        node_max_drops: Dict[Any, Tuple[float, float]] = {}

        # Global peak tracking
        global_peak_drop = 0.0
        global_peak_time = t_array[0]
        global_peak_node: Any = None

        # Track waveforms for selected nodes
        track_set = set(track_nodes) if track_nodes else set()
        temp_voltages: Dict[Any, List[float]] = {n: [] for n in track_set}

        # Build system matrix: A = G + C/dt (for BE) or A = G + 2C/dt (for Trap)
        t0_factor = time_module.perf_counter()
        if method == IntegrationMethod.BACKWARD_EULER:
            # BE: (G + C/dt) * V(t+dt) = I(t+dt) + (C/dt) * V(t) - G_up * V_p
            A = rc.G_uu + rc.C_uu / dt
            C_coeff = 1.0 / dt
        else:
            # Trapezoidal: (G + 2C/dt) * V(t+dt) = 2*(I_avg) + (2C/dt - G) * V(t) - 2*G_up * V_p
            A = rc.G_uu + 2.0 * rc.C_uu / dt
            C_coeff = 2.0 / dt

        # Factor the matrix once
        lu = spla.factorized(A.tocsc())
        timings['factor'] = time_module.perf_counter() - t0_factor

        # Pad voltage contribution (constant)
        V_p = np.full(len(rc.pad_nodes), vdd, dtype=float)
        if rc.G_up.shape[1] > 0:
            G_up_Vp = rc.G_up @ V_p
        else:
            G_up_Vp = np.zeros(n_unknown)

        # Initial condition: all unknown nodes at Vdd (steady state with no current)
        V_u = np.full(n_unknown, vdd, dtype=float)

        # Time stepping
        t0_solve = time_module.perf_counter()

        for i, t in enumerate(t_array):
            if verbose and i % max(1, n_steps // 10) == 0:
                print(f"  Time step {i}/{n_steps} (t={t*1e9:.2f} ns)")

            # Evaluate currents
            currents = self._evaluate_currents_at_time(t)
            total_current = sum(currents.values())
            total_current_per_time[i] = total_current

            # Build current vector
            I_u = np.zeros(n_unknown, dtype=float)
            for node, curr in currents.items():
                if node in rc.unknown_to_idx:
                    # Sink current is positive input, nodal equation uses negative
                    I_u[rc.unknown_to_idx[node]] += -float(curr)

            if i == 0:
                # For first step, just compute steady state with initial condition
                # (or we could do DC solve, but we start from Vdd)
                pass
            else:
                # Time step
                if method == IntegrationMethod.BACKWARD_EULER:
                    # rhs = I(t+dt) + (C/dt) * V(t) - G_up * V_p
                    rhs = I_u + C_coeff * (rc.C_uu @ V_u) - G_up_Vp
                else:
                    # Trapezoidal: uses average of current
                    # rhs = I(t+dt) + (2C/dt - G) * V(t) - G_up * V_p
                    rhs = I_u + C_coeff * (rc.C_uu @ V_u) - (rc.G_uu @ V_u) - G_up_Vp

                V_u = lu(rhs)

            # Build voltage dict and compute IR-drop
            V_dict: Dict[Any, float] = {}
            ir_drop_dict: Dict[Any, float] = {}

            for n in rc.pad_nodes:
                V_dict[n] = vdd
                ir_drop_dict[n] = 0.0

            for idx, n in enumerate(rc.unknown_nodes):
                v = float(V_u[idx])
                V_dict[n] = v
                ir_drop_dict[n] = vdd - v

            # Compute statistics
            if ir_drop_dict:
                max_drop = max(ir_drop_dict.values())
                max_drop_node = max(ir_drop_dict, key=ir_drop_dict.get)
            else:
                max_drop = 0.0
                max_drop_node = None

            max_ir_drop_per_time[i] = max_drop

            if max_drop > global_peak_drop:
                global_peak_drop = max_drop
                global_peak_time = t
                global_peak_node = max_drop_node

            # Compute vsrc current
            total_vsrc_current_per_time[i] = self._compute_vsrc_current(V_dict)

            # Update spatial peaks
            for node, drop in ir_drop_dict.items():
                if node not in peak_ir_drop_per_node or drop > peak_ir_drop_per_node[node]:
                    peak_ir_drop_per_node[node] = drop
                if node not in node_max_drops or drop > node_max_drops[node][0]:
                    node_max_drops[node] = (drop, t)

            for node, curr in currents.items():
                if node not in peak_current_per_node or abs(curr) > abs(peak_current_per_node[node]):
                    peak_current_per_node[node] = curr

            # Store tracked waveforms
            for node in track_set:
                if node in V_dict:
                    temp_voltages[node].append(V_dict[node])
                else:
                    temp_voltages[node].append(vdd)

        timings['solve'] = time_module.perf_counter() - t0_solve

        # Determine worst nodes
        worst_nodes_list = sorted(
            node_max_drops.items(),
            key=lambda x: x[1][0],
            reverse=True
        )[:n_worst_nodes]
        worst_nodes = [(node, drop, time) for node, (drop, time) in worst_nodes_list]

        # Build final waveform storage
        tracked_waveforms: Dict[Any, np.ndarray] = {}
        tracked_ir_drop: Dict[Any, np.ndarray] = {}

        for node in track_set:
            if node in temp_voltages and len(temp_voltages[node]) == n_steps:
                tracked_waveforms[node] = np.array(temp_voltages[node])
                tracked_ir_drop[node] = vdd - tracked_waveforms[node]

        timings['total'] = time_module.perf_counter() - t0_total

        return TransientResult(
            t_array=t_array,
            peak_ir_drop=global_peak_drop,
            peak_ir_drop_time=global_peak_time,
            peak_ir_drop_node=global_peak_node,
            worst_nodes=worst_nodes,
            nominal_voltage=vdd,
            integration_method=method,
            timings=timings,
            tracked_waveforms=tracked_waveforms,
            tracked_ir_drop=tracked_ir_drop,
            max_ir_drop_per_time=max_ir_drop_per_time,
            total_current_per_time=total_current_per_time,
            total_vsrc_current_per_time=total_vsrc_current_per_time,
            peak_ir_drop_per_node=peak_ir_drop_per_node,
            peak_current_per_node=peak_current_per_node,
        )

    def _create_trivial_result(
        self,
        t_array: np.ndarray,
        vdd: float,
        method: IntegrationMethod,
        timings: Dict[str, float],
    ) -> TransientResult:
        """Create result when all nodes are pads (trivial case)."""
        n_steps = len(t_array)
        return TransientResult(
            t_array=t_array,
            peak_ir_drop=0.0,
            peak_ir_drop_time=t_array[0],
            peak_ir_drop_node=None,
            worst_nodes=[],
            nominal_voltage=vdd,
            integration_method=method,
            timings=timings,
            tracked_waveforms={},
            tracked_ir_drop={},
            max_ir_drop_per_time=np.zeros(n_steps),
            total_current_per_time=np.zeros(n_steps),
            total_vsrc_current_per_time=np.zeros(n_steps),
            peak_ir_drop_per_node={},
            peak_current_per_node={},
        )

    @property
    def has_capacitance(self) -> bool:
        """Check if the model has any capacitive elements."""
        rc = self._ensure_rc_system()
        return rc.C_full.nnz > 0

    @property
    def total_capacitance(self) -> float:
        """Get total capacitance in the model (sum of diagonal)."""
        rc = self._ensure_rc_system()
        return float(rc.C_full.diagonal().sum()) / 2.0  # Divide by 2 for nodal stamping

    @property
    def has_dynamic_sources(self) -> bool:
        """Check if any current sources have time-varying waveforms."""
        for src in self._current_sources.values():
            if hasattr(src, 'has_waveform_data') and src.has_waveform_data():
                return True
        return False

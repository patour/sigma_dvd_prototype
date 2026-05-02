"""Adjoint sensitivity solver for dynamic IR-drop attribution.

Implements the adjoint method to compute which aggressor current sources
contribute most to the dynamic IR-drop at a victim node at observation time T.

The adjoint method captures true dynamic attribution by propagating sensitivities
backward through the RC network's memory. For an RC system C·dV/dt + G·V = I(t),
the adjoint variable λ(t) at each node captures how a current perturbation at
time t affects the final IR-drop at time T.

Key insight: Convolving λ with each aggressor's current waveform gives the
true attribution.

Mathematical Formulation:
    Forward Problem (Backward Euler):
        A · V_{n+1} = I_{n+1} + B · V_n - G_up · V_pad
        where A = G_uu + C_uu/dt, B = C_uu/dt

    Adjoint Problem (Backward in Time):
        Terminal condition: A · λ_{N-1} = e_victim  →  λ_{N-1} = A^{-1} · e_victim
        Recursion: A^T · λ_n = B^T · λ_{n+1}  (for symmetric A: A^T = A)

    Contribution Computation (Discrete Adjoint):
        contribution_i = Σ_k λ_k[node_i] · I_i(t_k)  # no dt factor for discrete adjoint

Example usage:
    from parser.netlist import NetlistParser
    from model.factory import create_model_from_pdn
    from analysis.transient_solver import TransientIRDropSolver
    from analysis.adjoint_sensitivity import AdjointSensitivitySolver

    parser = NetlistParser('./netlist/netlist_small')
    graph = parser.parse()
    model = create_model_from_pdn(graph, 'VDD')

    trans = TransientIRDropSolver(model, graph)
    result = trans.solve_transient(t_start=0, t_end=100e-9, dt=1e-9)

    victim = result.peak_ir_drop_node
    T = result.peak_ir_drop_time

    adjoint = AdjointSensitivitySolver.from_transient_solver(trans)
    attribution = adjoint.analyze_victim(
        victim_node=victim, observation_time=T,
        memory_window=20, dt=1e-9, top_k=10,
    )

    print(f"Victim: {attribution.victim_node}")
    print(f"Total IR-drop: {attribution.ir_drop_at_T:.2f} mV")
    for i, agg in enumerate(attribution.top_aggressors, 1):
        print(f"  {i}. {agg.node}: {agg.contribution_mV:.3f} mV ({agg.contribution_pct:.1f}%)")
"""

from __future__ import annotations

import time as time_module
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import scipy.sparse as sp

from model.unified_model import UnifiedPowerGridModel
from solver.unified_solver import _factor_conductance_matrix
from .transient_solver import TransientIRDropSolver, RCSystem, IntegrationMethod, TransientResult
from .vectorized_sources import VectorizedCurrentSources



@dataclass
class AggressorContribution:
    """Contribution of an aggressor node to the victim's IR-drop.

    Attributes:
        node: Aggressor node name (string for PDN, NodeID for synthetic)
        contribution_mV: Contribution to IR-drop in millivolts.
            - In 'zero' mode: total contribution from I(t)
            - In 'dc' mode: incremental contribution from ΔI = I(t) - I_DC
        contribution_pct: Percentage of total IR-drop at victim
        source_names: List of current source instance names connected to this node
        current_waveform: Optional current waveform I(t) over the memory window
        static_contribution_mV: Static (DC) contribution in millivolts.
            Only populated in 'dc' mode. This is the contribution from I_DC.
            Total contribution = contribution_mV + static_contribution_mV
    """
    node: Any
    contribution_mV: float
    contribution_pct: float
    source_names: List[str]
    current_waveform: Optional[np.ndarray] = None
    static_contribution_mV: Optional[float] = None


@dataclass
class AdjointAttribution:
    """Result of adjoint sensitivity analysis.

    Attributes:
        victim_node: The victim node being analyzed
        observation_time: Time T at which IR-drop is observed (seconds)
        ir_drop_at_T: Total IR-drop at victim at time T (mV), i.e., VDD - V_T
        memory_window: Tuple (t_start, t_end) in seconds
        t_array: Time points in the memory window
        spatial_window: Optional (x_min, x_max, y_min, y_max) spatial filter
        n_candidate_sources: Number of sources considered within spatial window
        self_contribution_mV: Victim's own current contribution (mV)
        self_contribution_pct: Percentage of total from victim's own switching
        victim_current_waveform: Victim's I(t) if it has current sources
        top_aggressors: List of top K remote aggressor contributions
        total_attributed_ir_drop: Sum of all contributions (mV)
        attribution_efficiency: Ratio total_attributed / ir_drop_at_T (~1.0)
        timings: Dict with timing breakdown
        initial_condition: Initial condition used ('zero' or 'dc')
        dc_ir_drop_mV: DC baseline IR-drop (mV), only populated when using 'dc'.
                       Incremental IR-drop = ir_drop_at_T - dc_ir_drop_mV
        all_node_contributions: Optional raw per-node contributions in mV for
            downstream spatial partitioning. Keys are node names, values are
            the total contribution of all sources attached to that node.
            Does not include the victim's self-contribution (which is in
            self_contribution_mV). Only populated by the distributed solver.
    """
    victim_node: Any
    observation_time: float
    ir_drop_at_T: float
    memory_window: Tuple[float, float]
    t_array: np.ndarray
    spatial_window: Optional[Tuple[float, float, float, float]]
    n_candidate_sources: int
    self_contribution_mV: float
    self_contribution_pct: float
    victim_current_waveform: Optional[np.ndarray]
    top_aggressors: List[AggressorContribution]
    total_attributed_ir_drop: float
    attribution_efficiency: float
    timings: Dict[str, float]
    initial_condition: str = 'zero'
    dc_ir_drop_mV: Optional[float] = None
    all_node_contributions: Optional[Dict[str, float]] = None


@dataclass
class AdjointSolverContext:
    """Pre-computed context for adjoint solving.

    Stores factorized matrices and mappings for efficient batch adjoint solves.
    """
    rc_system: RCSystem
    dt: float
    dt_scaled: float
    A: sp.csr_matrix                   # G_uu + C_uu/dt (BE) or G_uu + 2*C_uu/dt (TR)
    lu_A: Callable                     # Cached LU factorization of A
    B: sp.csr_matrix                   # C_uu/dt (BE) or 2*C_uu/dt (TR)
    vdd: float
    unknown_nodes: List[Any]
    unknown_to_idx: Dict[Any, int]
    n_unknown: int
    method: IntegrationMethod = IntegrationMethod.BACKWARD_EULER


class AdjointSensitivitySolver:
    """Adjoint sensitivity solver for dynamic IR-drop attribution.

    Computes which aggressor current sources contribute most to the IR-drop
    at a victim node at a specified observation time, accounting for the
    RC network's dynamic memory.

    The adjoint method propagates sensitivities backward through time,
    capturing how current perturbations at different times affect the
    final voltage at the victim.

    Thread-safety: instances are NOT safe for concurrent use across threads.
    ``analyze_victim`` / ``analyze_victim_static`` temporarily swap
    ``self._vec_sources`` when ``smoothed_sources`` is supplied, so
    overlapping calls on the same instance can race.
    """

    def __init__(
        self,
        model: UnifiedPowerGridModel,
        graph: Any = None,
        vectorize_threshold: int = 10000,
    ):
        """Initialize adjoint solver.

        Args:
            model: UnifiedPowerGridModel instance
            graph: Original parsed graph with current source metadata.
                   If None, will try to use model.graph.
            vectorize_threshold: Use vectorized evaluation when source count
                                 exceeds this threshold. Default 10000.

        Note:
            To control wscale application, use parser.current_sources.set_apply_wscale()
            before calling solver methods. This is thread-safe.
        """
        self.model = model
        self._graph = graph if graph is not None else model.graph
        self._vectorize_threshold = vectorize_threshold

        # RC system (lazy initialization)
        self._rc_system: Optional[RCSystem] = None

        # Cached LU factorization (shared from transient solver to avoid redundant factorization)
        self._cached_lu: Optional[Any] = None
        self._cached_lu_dt: Optional[float] = None
        self._cached_lu_method: Optional[IntegrationMethod] = None

        # Vectorized sources
        self._vec_sources: Optional[VectorizedCurrentSources] = None

        # Source-to-node mapping: source_name -> node
        self._source_to_node: Dict[str, Any] = {}

        # Node-to-sources mapping: node -> [source_names]
        self._node_to_sources: Dict[Any, List[str]] = {}

        # Initialize from graph metadata
        self._init_sources()

    @classmethod
    def from_transient_solver(
        cls,
        transient_solver: TransientIRDropSolver,
    ) -> 'AdjointSensitivitySolver':
        """Create adjoint solver from an existing transient solver.

        This is the recommended way to create an AdjointSensitivitySolver,
        as it reuses the already-built RC system, vectorized sources, and
        cached LU factorization (saving ~200-500 MB).

        Args:
            transient_solver: TransientIRDropSolver instance

        Returns:
            AdjointSensitivitySolver with shared data structures
        """
        solver = cls.__new__(cls)
        solver.model = transient_solver.model
        solver._graph = transient_solver._graph
        solver._vectorize_threshold = transient_solver._vectorize_threshold

        # Ensure RC system is built (this also builds vectorized sources)
        transient_solver._ensure_rc_system()

        # Share RC system and vectorized sources
        solver._rc_system = transient_solver._rc_system
        solver._vec_sources = transient_solver._vec_sources

        # Share cached LU factorization (if available) to avoid redundant factorization
        # The adjoint uses the same system matrix A = G_uu + C_uu/dt as the transient
        solver._cached_lu = getattr(transient_solver, '_cached_lu', None)
        solver._cached_lu_dt = getattr(transient_solver, '_cached_lu_dt', None)
        solver._cached_lu_method = getattr(transient_solver, '_cached_lu_method', None)

        # Build source-to-node mappings
        solver._source_to_node = {}
        solver._node_to_sources = {}
        solver._build_source_mappings(transient_solver)

        return solver

    def _init_sources(self) -> None:
        """Initialize current sources and build mappings."""
        # Get instance_sources from graph metadata
        graph_obj = self._graph
        graph_dict = None
        if hasattr(graph_obj, 'graph') and isinstance(graph_obj.graph, dict):
            graph_dict = graph_obj.graph
        elif hasattr(graph_obj, '_attrs'):
            graph_dict = graph_obj._attrs

        if graph_dict is None:
            return

        # Check for raw CurrentSource objects or serialized dicts
        raw_sources = graph_dict.get('_instance_sources_objects', {})
        has_raw = bool(raw_sources)
        if not raw_sources:
            raw_sources = graph_dict.get('instance_sources', {})

        if not raw_sources:
            return

        n_sources = len(raw_sources)

        # Always build source-to-node mappings (needed for attribution)
        if has_raw:
            for name, src in raw_sources.items():
                node = src.node1
                if node and node != '0':
                    self._source_to_node[name] = node
                    if node not in self._node_to_sources:
                        self._node_to_sources[node] = []
                    self._node_to_sources[node].append(name)
        else:
            for name, data in raw_sources.items():
                node = data.get('node1', '')
                if node and node != '0':
                    self._source_to_node[name] = node
                    if node not in self._node_to_sources:
                        self._node_to_sources[node] = []
                    self._node_to_sources[node].append(name)

        # Build vectorized sources if threshold is met
        threshold = self._vectorize_threshold
        use_vectorized = threshold >= 0 and (threshold == 0 or n_sources >= threshold)

        if use_vectorized and n_sources > 0:
            edge_cache = self.model.edge_cache

            if has_raw:
                self._vec_sources = VectorizedCurrentSources.from_current_sources(
                    raw_sources,
                    edge_cache.node_to_idx,
                    edge_cache.n_nodes,
                )
            else:
                self._vec_sources = VectorizedCurrentSources.from_serialized_dicts(
                    raw_sources,
                    edge_cache.node_to_idx,
                    edge_cache.n_nodes,
                )

    def _build_source_mappings(self, transient_solver: TransientIRDropSolver) -> None:
        """Build source-to-node mappings from transient solver's data."""
        # Get instance_sources from graph metadata
        graph_obj = transient_solver._graph
        graph_dict = None
        if hasattr(graph_obj, 'graph') and isinstance(graph_obj.graph, dict):
            graph_dict = graph_obj.graph
        elif hasattr(graph_obj, '_attrs'):
            graph_dict = graph_obj._attrs

        if graph_dict is None:
            return

        # Check for raw CurrentSource objects or serialized dicts
        raw_sources = graph_dict.get('_instance_sources_objects', {})
        has_raw = bool(raw_sources)
        if not raw_sources:
            raw_sources = graph_dict.get('instance_sources', {})

        if not raw_sources:
            return

        if has_raw:
            for name, src in raw_sources.items():
                node = src.node1
                if node and node != '0':
                    self._source_to_node[name] = node
                    if node not in self._node_to_sources:
                        self._node_to_sources[node] = []
                    self._node_to_sources[node].append(name)
        else:
            for name, data in raw_sources.items():
                node = data.get('node1', '')
                if node and node != '0':
                    self._source_to_node[name] = node
                    if node not in self._node_to_sources:
                        self._node_to_sources[node] = []
                    self._node_to_sources[node].append(name)

    def _ensure_rc_system(self) -> RCSystem:
        """Get or build RC system."""
        if self._rc_system is None:
            # Build RC system using the same method as TransientIRDropSolver
            temp_solver = TransientIRDropSolver(self.model, self._graph)
            self._rc_system = temp_solver._ensure_rc_system()
        return self._rc_system

    def prepare(
        self,
        dt: float,
        method: IntegrationMethod = IntegrationMethod.BACKWARD_EULER,
    ) -> AdjointSolverContext:
        """Prepare solver context for a given time step.

        Pre-computes and caches the LU factorization for efficient
        repeated adjoint solves at the same time step.

        Args:
            dt: Time step in seconds
            method: Integration method (default: BACKWARD_EULER)

        Returns:
            AdjointSolverContext with cached factorization

        Note:
            For PDN matrices, G and C are symmetric → A^T = A,
            so the same LU factorization can be reused for the adjoint.
            If created via from_transient_solver() and parameters match,
            the cached factorization is reused (~200-500 MB savings).
        """
        rc = self._ensure_rc_system()

        # Scale dt for PDN unit consistency (s -> ps)
        dt_scaled = dt * 1e12

        if method == IntegrationMethod.BACKWARD_EULER:
            A = rc.G_uu + rc.C_uu / dt_scaled
            B = rc.C_uu / dt_scaled
        else:
            # Trapezoidal
            A = rc.G_uu + 2.0 * rc.C_uu / dt_scaled
            B = 2.0 * rc.C_uu / dt_scaled

        # Reuse cached factorization if parameters match (saves ~200-500 MB)
        # The adjoint uses the same system matrix A = G_uu + C_uu/dt as the transient
        if (self._cached_lu is not None and
            self._cached_lu_dt == dt and
            self._cached_lu_method == method):
            lu_A = self._cached_lu
        else:
            # Factor A fresh (symmetric for PDN, so A^T = A)
            lu_A = _factor_conductance_matrix(A)

        return AdjointSolverContext(
            rc_system=rc,
            dt=dt,
            dt_scaled=dt_scaled,
            A=A,
            lu_A=lu_A,
            B=B,
            vdd=self.model.vdd,
            unknown_nodes=rc.unknown_nodes,
            unknown_to_idx=rc.unknown_to_idx,
            n_unknown=rc.n_unknown,
            method=method,
        )

    def analyze_victim_static(
        self,
        victim_node: Any,
        observation_time: float,
        top_k: int = 10,
        spatial_window: Optional[Tuple[float, float, float, float]] = None,
        window_margin: float = 0.0,
        include_waveforms: bool = False,
        initial_condition: str = 'zero',
        smoothed_sources: Optional[VectorizedCurrentSources] = None,
    ) -> AdjointAttribution:
        """Static sensitivity analysis for stiff RC systems.

        Uses the steady-state sensitivity (∂V/∂I = G^-1) to compute
        contributions. This is appropriate when the RC time constant
        is much smaller than the time scale of interest.

        For stiff systems (τ << dt), the dynamic adjoint method gives
        near-zero contributions because the RC memory decays instantly.
        This static method directly computes how each source affects
        the victim's voltage at steady state.

        Args:
            victim_node: Node at which to measure IR-drop
            observation_time: Time T at which to evaluate currents
            top_k: Number of top aggressors to return
            spatial_window: Optional (x_min, x_max, y_min, y_max) filter
            window_margin: Auto-create window with this margin (um)
            include_waveforms: If True, include single-point currents
            initial_condition: Initial condition for attribution:
                - 'zero': Assume V=VDD at start, attribute total IR-drop (default)
                - 'dc': Start from DC operating point, attribute incremental IR-drop
                        (IR-drop above the DC baseline from static currents)
            smoothed_sources: Pre-smoothed VectorizedCurrentSources to evaluate
                currents against. **Must match the handle passed to
                ``TransientIRDropSolver.solve_transient(smoothed_sources=...)``
                used to compute the forward solve being attributed.** Adjoint
                duality (``e_v^T V = lambda^T I``) requires the same ``I`` in
                both forward and backward sweeps; passing a different (or
                None) handle here breaks contribution-sums-to-IR-drop.

        Returns:
            AdjointAttribution with static sensitivity results
        """
        # Temporarily swap in smoothed sources if provided, mirroring the
        # solve_transient(smoothed_sources=...) pattern. Restored in finally
        # so other code paths (forward eval, repeated analyses) see the
        # unchanged raw VCS.
        original_sources = self._vec_sources
        if smoothed_sources is not None:
            self._vec_sources = smoothed_sources
        try:
            return self._analyze_victim_static_impl(
                victim_node=victim_node,
                observation_time=observation_time,
                top_k=top_k,
                spatial_window=spatial_window,
                window_margin=window_margin,
                include_waveforms=include_waveforms,
                initial_condition=initial_condition,
            )
        finally:
            self._vec_sources = original_sources

    def _analyze_victim_static_impl(
        self,
        victim_node: Any,
        observation_time: float,
        top_k: int,
        spatial_window: Optional[Tuple[float, float, float, float]],
        window_margin: float,
        include_waveforms: bool,
        initial_condition: str,
    ) -> AdjointAttribution:
        """Implementation of analyze_victim_static after source swap."""
        timings: Dict[str, float] = {}
        t0_total = time_module.perf_counter()

        rc = self._ensure_rc_system()

        if victim_node not in rc.unknown_to_idx:
            raise ValueError(f"Victim node {victim_node} not in unknown nodes")

        victim_idx = rc.unknown_to_idx[victim_node]
        vdd = self.model.vdd

        # Spatial filtering
        t0_filter = time_module.perf_counter()
        if spatial_window is None and window_margin > 0:
            x_v, y_v = self._get_node_coordinates(victim_node)
            if x_v is not None:
                spatial_window = (
                    x_v - window_margin,
                    x_v + window_margin,
                    y_v - window_margin,
                    y_v + window_margin,
                )

        if spatial_window is not None:
            candidate_sources = self._filter_sources_by_window(
                spatial_window, rc.unknown_to_idx
            )
        else:
            candidate_sources = {
                name: (node, rc.unknown_to_idx[node])
                for name, node in self._source_to_node.items()
                if node in rc.unknown_to_idx
            }
        timings['filter'] = time_module.perf_counter() - t0_filter

        n_candidates = len(candidate_sources)

        # Compute static sensitivity: solve G * λ = e_victim
        # λ = G^-1 * e_victim gives sensitivity of V to current at each node
        t0_sensitivity = time_module.perf_counter()
        lu_G = _factor_conductance_matrix(rc.G_uu)
        e_victim = np.zeros(rc.n_unknown, dtype=np.float64)
        e_victim[victim_idx] = 1.0
        sensitivity = lu_G.solve(e_victim)  # kOhm (V/mA)
        timings['sensitivity'] = time_module.perf_counter() - t0_sensitivity

        # Compute IR-drop at observation time
        t0_forward = time_module.perf_counter()
        n_unknown = rc.n_unknown
        I_u = np.zeros(n_unknown, dtype=np.float64)

        if self._vec_sources is not None:
            currents = self._vec_sources.evaluate_at_time(observation_time)
            for node_idx in range(self._vec_sources.n_nodes):
                if currents[node_idx] != 0:
                    node = self.model.edge_cache.idx_to_node[node_idx]
                    if node in rc.unknown_to_idx:
                        unknown_idx = rc.unknown_to_idx[node]
                        I_u[unknown_idx] -= currents[node_idx]
        else:
            # Use raw source evaluation
            all_sources = {
                name: (node, rc.unknown_to_idx[node])
                for name, node in self._source_to_node.items()
                if node in rc.unknown_to_idx
            }
            source_currents = self._evaluate_currents_from_raw_sources(observation_time, all_sources)
            for src_name, (node, unknown_idx) in all_sources.items():
                current = source_currents.get(src_name, 0.0)
                I_u[unknown_idx] -= current

        V_p = np.full(len(rc.pad_nodes), vdd, dtype=float)
        if rc.G_up.shape[1] > 0:
            G_up_Vp = rc.G_up @ V_p
        else:
            G_up_Vp = np.zeros(n_unknown)

        rhs = I_u - G_up_Vp
        V_u = lu_G.solve(rhs)
        total_ir_drop_at_T = (vdd - V_u[victim_idx]) * 1000.0  # mV
        timings['forward_eval'] = time_module.perf_counter() - t0_forward

        # For 'dc' initial condition: compute DC baseline and use incremental currents
        dc_ir_drop_mV: Optional[float] = None
        dc_currents: Optional[Dict[str, float]] = None

        if initial_condition == 'dc':
            t0_dc = time_module.perf_counter()
            # Compute DC currents (static component at t=0, which represents DC)
            dc_currents = self._evaluate_candidate_currents(0.0, candidate_sources)

            # Compute DC IR-drop at victim
            I_u_dc = np.zeros(n_unknown, dtype=np.float64)
            if self._vec_sources is not None:
                currents_dc = self._vec_sources.evaluate_at_time(0.0)
                for node_idx in range(self._vec_sources.n_nodes):
                    if currents_dc[node_idx] != 0:
                        node = self.model.edge_cache.idx_to_node[node_idx]
                        if node in rc.unknown_to_idx:
                            unknown_idx = rc.unknown_to_idx[node]
                            I_u_dc[unknown_idx] -= currents_dc[node_idx]
            else:
                all_sources_dc = {
                    name: (node, rc.unknown_to_idx[node])
                    for name, node in self._source_to_node.items()
                    if node in rc.unknown_to_idx
                }
                dc_source_currents = self._evaluate_currents_from_raw_sources(0.0, all_sources_dc)
                for src_name, (node, unknown_idx) in all_sources_dc.items():
                    current = dc_source_currents.get(src_name, 0.0)
                    I_u_dc[unknown_idx] -= current

            rhs_dc = I_u_dc - G_up_Vp
            V_u_dc = lu_G.solve(rhs_dc)
            dc_ir_drop_mV = (vdd - V_u_dc[victim_idx]) * 1000.0  # mV
            timings['dc_eval'] = time_module.perf_counter() - t0_dc

        # ir_drop_at_T is always the total IR-drop
        ir_drop_at_T = total_ir_drop_at_T

        # For attribution efficiency, use incremental IR-drop in 'dc' mode
        if initial_condition == 'dc' and dc_ir_drop_mV is not None:
            ir_drop_for_efficiency = total_ir_drop_at_T - dc_ir_drop_mV
        else:
            ir_drop_for_efficiency = ir_drop_at_T

        # Compute contributions using static sensitivity
        # contribution_i = sensitivity[node_i] * I_i
        # For 'dc' mode, use incremental current: ΔI_i = I_i(T) - I_i(DC)
        t0_contrib = time_module.perf_counter()
        contributions: Dict[str, float] = {}
        static_contributions: Optional[Dict[str, float]] = None
        if initial_condition == 'dc':
            static_contributions = {}
        node_currents: Dict[Any, np.ndarray] = {}

        source_currents = self._evaluate_candidate_currents(observation_time, candidate_sources)
        for src_name, (node, node_idx) in candidate_sources.items():
            current = source_currents.get(src_name, 0.0)

            # For 'dc' mode, use incremental current
            if initial_condition == 'dc' and dc_currents is not None:
                dc_current = dc_currents.get(src_name, 0.0)
                delta_current = current - dc_current

                # Also compute static contribution
                static_contributions[src_name] = sensitivity[node_idx] * dc_current * 1000.0
            else:
                delta_current = current

            # sensitivity is in kOhm (V/mA), current is in mA
            # contribution = sensitivity * current = kOhm * mA = V
            # Convert to mV
            contributions[src_name] = sensitivity[node_idx] * delta_current * 1000.0

            if include_waveforms:
                if node not in node_currents:
                    node_currents[node] = np.array([0.0])
                node_currents[node][0] += delta_current

        timings['contributions'] = time_module.perf_counter() - t0_contrib

        # Build results
        t0_build = time_module.perf_counter()

        # Self-contribution
        self_contribution_mV = 0.0
        victim_current_waveform = None
        self_source_names = self._node_to_sources.get(victim_node, [])

        for src_name in self_source_names:
            if src_name in contributions:
                self_contribution_mV += contributions[src_name]
                del contributions[src_name]

        if victim_node in node_currents and include_waveforms:
            victim_current_waveform = node_currents[victim_node]

        # Use ir_drop_for_efficiency for percentage calculations
        # (incremental for 'dc' mode, total for 'zero' mode)
        self_contribution_pct = (
            100.0 * self_contribution_mV / ir_drop_for_efficiency
            if ir_drop_for_efficiency > 0 else 0.0
        )

        # Build top-K remote aggressors
        top_aggressors = self._build_top_k_results(
            contributions=contributions,
            node_currents=node_currents,
            top_k=top_k,
            total_ir_drop=ir_drop_for_efficiency,
            include_waveforms=include_waveforms,
            static_contributions=static_contributions,
        )

        # Total attributed
        total_attributed = self_contribution_mV + sum(
            agg.contribution_mV for agg in top_aggressors
        )
        for src_name, contrib in contributions.items():
            node = self._source_to_node.get(src_name)
            if node and node not in {agg.node for agg in top_aggressors}:
                total_attributed += contrib

        attribution_efficiency = (
            total_attributed / ir_drop_for_efficiency if ir_drop_for_efficiency > 0 else 0.0
        )

        timings['build_results'] = time_module.perf_counter() - t0_build
        timings['total'] = time_module.perf_counter() - t0_total

        return AdjointAttribution(
            victim_node=victim_node,
            observation_time=observation_time,
            ir_drop_at_T=ir_drop_at_T,
            memory_window=(observation_time, observation_time),
            t_array=np.array([observation_time]),
            spatial_window=spatial_window,
            n_candidate_sources=n_candidates,
            self_contribution_mV=self_contribution_mV,
            self_contribution_pct=self_contribution_pct,
            victim_current_waveform=victim_current_waveform,
            top_aggressors=top_aggressors,
            total_attributed_ir_drop=total_attributed,
            attribution_efficiency=attribution_efficiency,
            timings=timings,
            initial_condition=initial_condition,
            dc_ir_drop_mV=dc_ir_drop_mV,
        )

    def analyze_victim(
        self,
        victim_node: Any,
        observation_time: float,
        memory_window: int = 20,
        dt: float = 1e-9,
        top_k: int = 10,
        spatial_window: Optional[Tuple[float, float, float, float]] = None,
        window_margin: float = 0.0,
        include_waveforms: bool = True,
        context: Optional[AdjointSolverContext] = None,
        method: IntegrationMethod = IntegrationMethod.BACKWARD_EULER,
        use_static: bool = False,
        initial_condition: str = 'zero',
        smoothed_sources: Optional[VectorizedCurrentSources] = None,
        transient_result: Optional[TransientResult] = None,
    ) -> AdjointAttribution:
        """Analyze aggressor contributions to IR-drop at victim node.

        Uses the adjoint method to compute how each current source
        contributes to the IR-drop at the victim at observation time T.

        Args:
            victim_node: Node at which to measure IR-drop
            observation_time: Time T at which to observe IR-drop (seconds)
            memory_window: Number of time steps to look back from T
            dt: Time step in seconds
            top_k: Number of top aggressors to return
            spatial_window: Optional (x_min, x_max, y_min, y_max) to limit
                           candidate sources. If None, considers all sources.
            window_margin: If > 0 and spatial_window is None, auto-creates
                          a window around victim with this margin (um)
            include_waveforms: If True, include current waveforms in results
            context: Optional pre-computed AdjointSolverContext (for batch use)
            method: Integration method (default: BACKWARD_EULER)
            use_static: If True, use static sensitivity (for stiff RC systems).
                       Recommended when RC time constant << dt.
            initial_condition: Initial condition for attribution:
                - 'zero': Assume V=VDD at start, attribute total IR-drop (default)
                - 'dc': Start from DC operating point, attribute incremental IR-drop
                        (IR-drop above the DC baseline from static currents)
            smoothed_sources: Pre-smoothed VectorizedCurrentSources to evaluate
                currents against. **Must match the handle passed to
                ``TransientIRDropSolver.solve_transient(smoothed_sources=...)``
                used to compute the forward solve being attributed.** Adjoint
                duality (``e_v^T V = lambda^T I``) requires the same ``I`` in
                both forward and backward sweeps; passing a different (or
                None) handle here breaks contribution-sums-to-IR-drop.
            transient_result: Optional pre-computed TransientResult. When
                provided and ``victim_node`` is in
                ``transient_result.tracked_ir_drop``, ``ir_drop_at_T`` is
                looked up from the tracked waveform at the timestep nearest
                ``observation_time`` instead of running a full forward
                transient sweep -- eliminating the dominant per-victim cost
                in production loops. **Consistency contract**: the
                TransientResult MUST come from a ``solve_transient(...)``
                call that used the same ``dt``, the same integration
                ``method``, the same ``smoothed_sources`` handle (if any),
                and the DC operating-point initial condition at t=0 (the
                default of both ``solve_transient`` and the fallback
                forward sweep). The victim must have been listed in
                ``track_nodes`` of the forward solve. Violating any of
                these silently yields an ``ir_drop_at_T`` inconsistent
                with the contributions sum and corrupts
                ``attribution_efficiency``. If the victim was not tracked,
                the lookup returns ``None`` internally and the fallback
                forward sweep runs.

        Returns:
            AdjointAttribution with victim analysis and top aggressors

        Raises:
            ValueError: If victim_node is not in the unknown nodes, or
                ``transient_result.t_array`` step does not match
                ``context.dt``, or ``observation_time`` falls outside the
                ``transient_result.t_array`` range by more than half a
                step.
        """
        # For stiff systems, use static sensitivity
        if use_static:
            return self.analyze_victim_static(
                victim_node=victim_node,
                observation_time=observation_time,
                top_k=top_k,
                spatial_window=spatial_window,
                window_margin=window_margin,
                include_waveforms=include_waveforms,
                initial_condition=initial_condition,
                smoothed_sources=smoothed_sources,
            )

        # Temporarily swap in smoothed sources if provided, mirroring the
        # solve_transient(smoothed_sources=...) pattern. Restored in finally
        # so other code paths (forward eval, repeated analyses) see the
        # unchanged raw VCS.
        original_sources = self._vec_sources
        if smoothed_sources is not None:
            self._vec_sources = smoothed_sources
        try:
            return self._analyze_victim_dynamic_impl(
                victim_node=victim_node,
                observation_time=observation_time,
                memory_window=memory_window,
                dt=dt,
                top_k=top_k,
                spatial_window=spatial_window,
                window_margin=window_margin,
                include_waveforms=include_waveforms,
                context=context,
                method=method,
                initial_condition=initial_condition,
                transient_result=transient_result,
            )
        finally:
            self._vec_sources = original_sources

    def _analyze_victim_dynamic_impl(
        self,
        victim_node: Any,
        observation_time: float,
        memory_window: int,
        dt: float,
        top_k: int,
        spatial_window: Optional[Tuple[float, float, float, float]],
        window_margin: float,
        include_waveforms: bool,
        context: Optional[AdjointSolverContext],
        method: IntegrationMethod,
        initial_condition: str,
        transient_result: Optional[TransientResult] = None,
    ) -> AdjointAttribution:
        """Implementation of analyze_victim (dynamic) after source swap."""
        timings: Dict[str, float] = {}
        t0_total = time_module.perf_counter()

        # Build or reuse context
        t0_prep = time_module.perf_counter()
        if context is None:
            context = self.prepare(dt, method)
        elif context.method != method:
            raise ValueError(
                f"context.method={context.method} disagrees with analyze_victim "
                f"method={method}; the context's lu_A was factored under the "
                f"context's method, so applying a different rhs formula yields "
                f"silently wrong results. Re-run prepare(dt, method) or pass "
                f"the matching method= to analyze_victim()."
            )
        timings['prepare'] = time_module.perf_counter() - t0_prep

        rc = context.rc_system

        # Validate victim node
        if victim_node not in context.unknown_to_idx:
            raise ValueError(f"Victim node {victim_node} not in unknown nodes")

        victim_idx = context.unknown_to_idx[victim_node]

        # Build time array for memory window
        t_end = observation_time
        t_start = observation_time - (memory_window - 1) * dt
        t_array = np.arange(t_start, t_end + dt * 0.5, dt)
        L = len(t_array)

        # Apply spatial filtering if specified
        t0_filter = time_module.perf_counter()
        if spatial_window is None and window_margin > 0:
            x_v, y_v = self._get_node_coordinates(victim_node)
            if x_v is not None:
                spatial_window = (
                    x_v - window_margin,
                    x_v + window_margin,
                    y_v - window_margin,
                    y_v + window_margin,
                )

        if spatial_window is not None:
            candidate_sources = self._filter_sources_by_window(
                spatial_window, context.unknown_to_idx
            )
        else:
            # All sources with valid (unknown) nodes
            candidate_sources = {
                name: (node, context.unknown_to_idx[node])
                for name, node in self._source_to_node.items()
                if node in context.unknown_to_idx
            }
        timings['filter'] = time_module.perf_counter() - t0_filter

        n_candidates = len(candidate_sources)

        # For 'dc' initial condition: compute DC baseline currents
        dc_currents: Optional[Dict[str, float]] = None
        dc_ir_drop_mV: Optional[float] = None
        total_ir_drop_at_T: Optional[float] = None

        if initial_condition == 'dc':
            t0_dc = time_module.perf_counter()
            # Compute DC currents (at t=0, which represents the static/DC state)
            dc_currents = self._evaluate_candidate_currents(0.0, candidate_sources)
            # Compute DC IR-drop at victim
            dc_ir_drop_mV = self._evaluate_ir_drop_at_time(victim_node, 0.0, context)
            timings['dc_eval'] = time_module.perf_counter() - t0_dc

        # Solve adjoint and accumulate contributions
        t0_adjoint = time_module.perf_counter()
        contributions, node_currents, static_contributions = self._solve_adjoint_and_accumulate(
            victim_idx=victim_idx,
            t_array=t_array,
            context=context,
            candidate_sources=candidate_sources,
            include_waveforms=include_waveforms,
            dc_currents=dc_currents,
        )
        timings['adjoint_solve'] = time_module.perf_counter() - t0_adjoint

        # Compute IR-drop at T via a forward TRANSIENT sweep (BE/TR), not a
        # quasi-static solve. By the discrete adjoint duality
        # ``e_v^T V_L = sum_k lambda_k^T I_k`` (zero IC), the contribution
        # sum equals the actual transient V at observation_time -- NOT the
        # quasi-static value at I(t=T). Reporting the QSS here would make
        # ir_drop_at_T match the static-adjoint number even though dynamic
        # contributions sum to the transient value, breaking
        # attribution_efficiency. The forward sweep reuses the cached lu_A
        # and B from the adjoint context (same matrices the backward sweep
        # used).
        t0_forward = time_module.perf_counter()
        ir_drop_at_T: Optional[float] = None
        if transient_result is not None:
            ir_drop_at_T = self._lookup_transient_ir_drop_at_time(
                victim_node, observation_time, context, transient_result,
            )
        if ir_drop_at_T is None:
            ir_drop_at_T = self._evaluate_transient_ir_drop_at_time(
                victim_node, observation_time, context,
            )
        timings['forward_eval'] = time_module.perf_counter() - t0_forward

        # ir_drop_at_T is always the total IR-drop
        # For attribution efficiency, use incremental IR-drop in 'dc' mode
        if initial_condition == 'dc' and dc_ir_drop_mV is not None:
            ir_drop_for_efficiency = ir_drop_at_T - dc_ir_drop_mV
        else:
            ir_drop_for_efficiency = ir_drop_at_T

        # Build results
        t0_build = time_module.perf_counter()

        # Separate self-contribution (victim's own current sources)
        self_contribution_mV = 0.0
        victim_current_waveform = None
        self_source_names = self._node_to_sources.get(victim_node, [])

        for src_name in self_source_names:
            if src_name in contributions:
                self_contribution_mV += contributions[src_name]
                del contributions[src_name]  # Remove from remote contributions

        if victim_node in node_currents and include_waveforms:
            victim_current_waveform = node_currents[victim_node]

        # Use ir_drop_for_efficiency for percentage calculations
        self_contribution_pct = (
            100.0 * self_contribution_mV / ir_drop_for_efficiency
            if ir_drop_for_efficiency > 0 else 0.0
        )

        # Build top-K remote aggressors
        top_aggressors = self._build_top_k_results(
            contributions=contributions,
            node_currents=node_currents,
            top_k=top_k,
            total_ir_drop=ir_drop_for_efficiency,
            include_waveforms=include_waveforms,
            static_contributions=static_contributions,
        )

        # Compute total attributed IR-drop
        total_attributed = self_contribution_mV + sum(
            agg.contribution_mV for agg in top_aggressors
        )

        # Add contribution from sources not in top-K
        for src_name, contrib in contributions.items():
            node = self._source_to_node.get(src_name)
            if node and node not in {agg.node for agg in top_aggressors}:
                total_attributed += contrib

        attribution_efficiency = (
            total_attributed / ir_drop_for_efficiency if ir_drop_for_efficiency > 0 else 0.0
        )

        timings['build_results'] = time_module.perf_counter() - t0_build
        timings['total'] = time_module.perf_counter() - t0_total

        return AdjointAttribution(
            victim_node=victim_node,
            observation_time=observation_time,
            ir_drop_at_T=ir_drop_at_T,
            memory_window=(t_start, t_end),
            t_array=t_array,
            spatial_window=spatial_window,
            n_candidate_sources=n_candidates,
            self_contribution_mV=self_contribution_mV,
            self_contribution_pct=self_contribution_pct,
            victim_current_waveform=victim_current_waveform,
            top_aggressors=top_aggressors,
            total_attributed_ir_drop=total_attributed,
            attribution_efficiency=attribution_efficiency,
            timings=timings,
            initial_condition=initial_condition,
            dc_ir_drop_mV=dc_ir_drop_mV,
        )

    def _get_node_coordinates(
        self,
        node: Any,
    ) -> Tuple[Optional[float], Optional[float]]:
        """Extract (x, y) coordinates from a node name.

        Args:
            node: Node name (e.g., '1500_2000_M1')

        Returns:
            Tuple (x, y) or (None, None) if parsing fails
        """
        if not isinstance(node, str):
            return None, None

        parts = node.split('_')
        if len(parts) >= 2:
            try:
                x = float(parts[0])
                y = float(parts[1])
                return x, y
            except ValueError:
                pass

        return None, None

    def _filter_sources_by_window(
        self,
        window: Tuple[float, float, float, float],
        unknown_to_idx: Dict[Any, int],
    ) -> Dict[str, Tuple[Any, int]]:
        """Filter sources to those within spatial window.

        Args:
            window: (x_min, x_max, y_min, y_max) in um
            unknown_to_idx: Mapping from unknown nodes to indices

        Returns:
            Dict mapping source_name -> (node, unknown_idx)
        """
        x_min, x_max, y_min, y_max = window
        candidates: Dict[str, Tuple[Any, int]] = {}

        for src_name, node in self._source_to_node.items():
            if node not in unknown_to_idx:
                continue

            x, y = self._get_node_coordinates(node)
            if x is None:
                continue

            if x_min <= x <= x_max and y_min <= y <= y_max:
                candidates[src_name] = (node, unknown_to_idx[node])

        return candidates

    def _solve_adjoint_and_accumulate(
        self,
        victim_idx: int,
        t_array: np.ndarray,
        context: AdjointSolverContext,
        candidate_sources: Dict[str, Tuple[Any, int]],
        include_waveforms: bool,
        dc_currents: Optional[Dict[str, float]] = None,
    ) -> Tuple[Dict[str, float], Dict[Any, np.ndarray], Optional[Dict[str, float]]]:
        """Solve adjoint backward and accumulate contributions on-the-fly.

        Memory-efficient: Only keeps λ_current and λ_next (O(N) memory).

        For the discrete Backward Euler adjoint:
            Terminal condition: A^T · λ_{L-1} = e_victim  →  λ_{L-1} = A^{-1} · e_victim
            Recursion: A^T · λ_k = B^T · λ_{k+1}  →  λ_k = A^{-1} · B · λ_{k+1}

        The contribution of source i at time step k is simply:
            contribution_i += λ_k[node_i] · I_i(t_k)

        No dt factor is needed because the discrete adjoint already accounts
        for time discretization in the matrices A and B.

        Args:
            victim_idx: Index of victim in unknown nodes
            t_array: Time points in memory window
            context: AdjointSolverContext with cached factorization
            candidate_sources: Dict of source_name -> (node, unknown_idx)
            include_waveforms: Whether to store current waveforms
            dc_currents: Optional dict of DC currents for each source.
                        If provided, uses incremental currents (I(t) - I_DC)
                        for contribution calculation.

        Returns:
            Tuple of:
            - contributions: Dict[source_name, contribution_value in mV]
            - node_currents: Dict[node, current_waveform] if include_waveforms
            - static_contributions: Dict[source_name, static_contribution in mV]
                                   Only returned when dc_currents is provided.
        """
        L = len(t_array)
        n_unknown = context.n_unknown

        # Contribution accumulators (per source) - in V, converted to mV at end
        contributions: Dict[str, float] = {name: 0.0 for name in candidate_sources}

        # Static contribution accumulators (only used when dc_currents provided)
        static_contributions: Optional[Dict[str, float]] = None
        if dc_currents is not None:
            static_contributions = {name: 0.0 for name in candidate_sources}

        # Per-node current waveforms for reporting (aggregate by node)
        candidate_nodes = set(node for node, idx in candidate_sources.values())
        node_currents: Dict[Any, np.ndarray] = {}
        if include_waveforms:
            node_currents = {node: np.zeros(L) for node in candidate_nodes}

        # Only need two λ vectors (O(N) instead of O(L×N))
        lambda_next = np.zeros(n_unknown, dtype=np.float64)
        lambda_current = np.zeros(n_unknown, dtype=np.float64)

        # Terminal condition: A · λ_{L-1} = e_victim  (for symmetric A: A^T = A)
        # Solving: λ_{L-1} = A^{-1} · e_victim
        # This gives the correct steady-state sensitivity for stiff systems.
        e_victim = np.zeros(n_unknown, dtype=np.float64)
        e_victim[victim_idx] = 1.0
        lambda_next = context.lu_A.solve(e_victim)

        # Backward sweep from k = L-1 down to 0
        for k in range(L - 1, -1, -1):
            if k < L - 1:
                # Solve A^T · λ_k = B^T · λ_{k+1}
                # For symmetric A (PDN case): A^T = A, so reuse lu_A
                # B^T = B for symmetric B
                rhs = context.B @ lambda_next
                lambda_current = context.lu_A.solve(rhs)
            else:
                # k = L-1: use terminal condition (already computed)
                lambda_current = lambda_next.copy()

            # Evaluate currents at this time and accumulate contributions
            t = t_array[k]
            source_currents = self._evaluate_candidate_currents(t, candidate_sources)

            for src_name, (node, node_idx) in candidate_sources.items():
                current = source_currents.get(src_name, 0.0)

                # For 'dc' mode, use incremental current: ΔI = I(t) - I_DC
                if dc_currents is not None:
                    dc_current = dc_currents.get(src_name, 0.0)
                    delta_current = current - dc_current

                    # Also accumulate static contribution: λ · I_DC
                    static_contributions[src_name] += lambda_current[node_idx] * dc_current
                else:
                    delta_current = current

                # Discrete adjoint contribution (no dt factor):
                # λ has units of kOhm (from A^{-1} where A is in mS)
                # current has units of mA
                # λ · current has units: kOhm · mA = V
                contributions[src_name] += lambda_current[node_idx] * delta_current

                # Store current waveform for reporting (use incremental for consistency)
                if include_waveforms and node in node_currents:
                    node_currents[node][k] += delta_current

            # Swap for next iteration
            lambda_next, lambda_current = lambda_current, lambda_next

        # Convert contributions from V to mV
        for src_name in contributions:
            contributions[src_name] *= 1000.0

        # Convert static contributions from V to mV
        if static_contributions is not None:
            for src_name in static_contributions:
                static_contributions[src_name] *= 1000.0

        return contributions, node_currents, static_contributions

    def _evaluate_candidate_currents(
        self,
        t: float,
        candidate_sources: Dict[str, Tuple[Any, int]],
    ) -> Dict[str, float]:
        """Evaluate currents for candidate sources at time t.

        Args:
            t: Time in seconds
            candidate_sources: Dict of source_name -> (node, unknown_idx)

        Returns:
            Dict mapping source_name -> current in mA
        """
        if self._vec_sources is not None:
            # Use vectorized evaluation
            per_source_currents = self._vec_sources.evaluate_per_source_at_time(t)

            # Map source indices to names
            result: Dict[str, float] = {}

            # Build source name to index mapping if not cached
            if not hasattr(self, '_source_name_to_idx'):
                self._source_name_to_idx: Dict[str, int] = {}
                # Get source names from the mapping
                graph_obj = self._graph
                graph_dict = None
                if hasattr(graph_obj, 'graph') and isinstance(graph_obj.graph, dict):
                    graph_dict = graph_obj.graph
                elif hasattr(graph_obj, '_attrs'):
                    graph_dict = graph_obj._attrs

                if graph_dict:
                    raw_sources = graph_dict.get('_instance_sources_objects', {})
                    if not raw_sources:
                        raw_sources = graph_dict.get('instance_sources', {})

                    for idx, name in enumerate(raw_sources.keys()):
                        self._source_name_to_idx[name] = idx

            # Look up currents for candidate sources
            for src_name in candidate_sources:
                if src_name in self._source_name_to_idx:
                    idx = self._source_name_to_idx[src_name]
                    if idx < len(per_source_currents):
                        result[src_name] = per_source_currents[idx]

            return result

        # Fall back to object-based evaluation from raw CurrentSource objects
        return self._evaluate_currents_from_raw_sources(t, candidate_sources)

    def _evaluate_currents_from_raw_sources(
        self,
        t: float,
        candidate_sources: Dict[str, Tuple[Any, int]],
    ) -> Dict[str, float]:
        """Evaluate currents from raw CurrentSource objects.

        Fallback when vectorized sources are not available.

        Args:
            t: Time in seconds
            candidate_sources: Dict of source_name -> (node, unknown_idx)

        Returns:
            Dict mapping source_name -> current in mA
        """
        # Get raw sources from graph
        graph_obj = self._graph
        graph_dict = None
        if hasattr(graph_obj, 'graph') and isinstance(graph_obj.graph, dict):
            graph_dict = graph_obj.graph
        elif hasattr(graph_obj, '_attrs'):
            graph_dict = graph_obj._attrs

        if graph_dict is None:
            return {}

        raw_sources = graph_dict.get('_instance_sources_objects', {})
        has_raw = bool(raw_sources)
        if not raw_sources:
            raw_sources = graph_dict.get('instance_sources', {})

        if not raw_sources:
            return {}

        result: Dict[str, float] = {}

        # wscale is controlled by the thread-safe ContextVar (get_apply_wscale)
        for src_name in candidate_sources:
            if src_name not in raw_sources:
                continue

            src = raw_sources[src_name]

            if has_raw:
                # Raw CurrentSource object
                if hasattr(src, 'get_current_at_time'):
                    result[src_name] = src.get_current_at_time(t)
            else:
                # Serialized dict - need to evaluate manually
                dc = src.get('dc_value', 0.0)
                static = src.get('static_value', 0.0)
                # For serialized dicts, we just use DC + static (no waveform eval)
                result[src_name] = dc + static

        return result

    def _scatter_raw_sources_into(
        self,
        t: float,
        I_u: np.ndarray,
        context: AdjointSolverContext,
    ) -> None:
        """Raw-source scatter (legacy fallback when ``_vec_sources is None``).

        Adds ``-I_j(t)`` for each raw source j attached to an unknown node into
        the pre-allocated ``I_u`` in place. The vectorized fast path uses
        ``VectorizedCurrentSources.evaluate_to_rhs_array`` directly with cached
        ``(source_to_unknown, valid_mask)`` arrays — see the transient solver
        for the same pattern.
        """
        all_sources = {
            name: (node, context.unknown_to_idx[node])
            for name, node in self._source_to_node.items()
            if node in context.unknown_to_idx
        }
        source_currents = self._evaluate_currents_from_raw_sources(
            t, all_sources,
        )
        for src_name, (_node, unknown_idx) in all_sources.items():
            I_u[unknown_idx] -= source_currents.get(src_name, 0.0)

    def _evaluate_ir_drop_at_time(
        self,
        victim_node: Any,
        t: float,
        context: AdjointSolverContext,
    ) -> float:
        """Evaluate IR-drop at victim node at time t.

        Uses a single forward solve to get the IR-drop.

        Args:
            victim_node: Node to evaluate
            t: Time in seconds
            context: AdjointSolverContext

        Returns:
            IR-drop in mV
        """
        rc = context.rc_system
        vdd = context.vdd

        # Build RHS from current sources
        n_unknown = context.n_unknown
        I_u = np.zeros(n_unknown, dtype=np.float64)

        if self._vec_sources is not None:
            source_to_unknown, valid_mask = self._vec_sources.build_source_to_unknown_map(
                context.unknown_to_idx, self.model.edge_cache.idx_to_node,
            )
            self._vec_sources.evaluate_to_rhs_array(
                t, I_u, source_to_unknown, valid_mask,
            )
        else:
            self._scatter_raw_sources_into(t, I_u, context)

        # Pad contribution
        V_p = np.full(len(rc.pad_nodes), vdd, dtype=float)
        if rc.G_up.shape[1] > 0:
            G_up_Vp = rc.G_up @ V_p
        else:
            G_up_Vp = np.zeros(n_unknown)

        # Solve for steady state (quasi-static at time t)
        # For steady state: G_uu * V = I - G_up * V_p
        lu_G = _factor_conductance_matrix(rc.G_uu)
        rhs = I_u - G_up_Vp
        V_u = lu_G.solve(rhs)

        victim_idx = context.unknown_to_idx[victim_node]
        ir_drop = vdd - V_u[victim_idx]

        # Convert to mV
        return ir_drop * 1000.0

    def _evaluate_transient_ir_drop_at_time(
        self,
        victim_node: Any,
        t_target: float,
        context: AdjointSolverContext,
    ) -> float:
        """IR-drop at the victim from a forward transient sweep at t=t_target.

        Runs a BE or TR integration from t=0 (DC operating point as initial
        condition, matching ``TransientIRDropSolver.solve_transient``) to
        ``t_target`` using the cached ``context.lu_A`` and ``context.B``.
        The integration method is taken from ``context.method`` so the rhs
        formula and the factorization in ``context.lu_A`` cannot disagree.
        The result is the actual transient ``V(victim, t_target)`` -- the
        quantity the dynamic adjoint contribution sum is dual to.

        Differs from ``_evaluate_ir_drop_at_time`` (which is quasi-static)
        and is the right thing to report from the dynamic adjoint path.

        Args:
            victim_node: Node to evaluate.
            t_target: Time in seconds.
            context: AdjointSolverContext with cached lu_A, B, and method.

        Returns:
            IR-drop at victim at t_target in mV.
        """
        method = context.method
        rc = context.rc_system
        vdd = context.vdd
        dt = context.dt
        n_unknown = context.n_unknown

        if victim_node not in context.unknown_to_idx:
            raise ValueError(
                f"Victim {victim_node} not in unknown nodes"
            )
        victim_idx = context.unknown_to_idx[victim_node]

        # Pad voltage contribution (constant in time)
        V_p = np.full(len(rc.pad_nodes), vdd, dtype=float)
        if rc.G_up.shape[1] > 0:
            G_up_Vp = rc.G_up @ V_p
        else:
            G_up_Vp = np.zeros(n_unknown)

        # Build the source -> unknown mapping once for the vectorized fast
        # path; the per-step loop then becomes a single in-place scatter via
        # evaluate_to_rhs_array (matches transient_solver.py:838-841).
        if self._vec_sources is not None:
            source_to_unknown, valid_mask = self._vec_sources.build_source_to_unknown_map(
                context.unknown_to_idx, self.model.edge_cache.idx_to_node,
            )
        else:
            source_to_unknown = None
            valid_mask = None

        I_u = np.zeros(n_unknown, dtype=np.float64)

        def _fill_I_u(t: float) -> None:
            I_u.fill(0.0)
            if self._vec_sources is not None:
                self._vec_sources.evaluate_to_rhs_array(
                    t, I_u, source_to_unknown, valid_mask,
                )
            else:
                self._scatter_raw_sources_into(t, I_u, context)

        # Initial condition: DC operating point at t=0, same as
        # TransientIRDropSolver.solve_transient (transient_solver.py:806-808)
        _fill_I_u(0.0)
        lu_G = _factor_conductance_matrix(rc.G_uu)
        V_u = lu_G.solve(I_u - G_up_Vp)
        del lu_G  # Free DC factorization promptly

        # If observation_time is at or before t=0, return the DC IR-drop
        if t_target <= 0.0:
            return float((vdd - V_u[victim_idx]) * 1000.0)

        # Forward integrate to t_target
        n_steps = max(1, int(round(t_target / dt)))
        for k in range(1, n_steps + 1):
            t = k * dt
            _fill_I_u(t)
            if method == IntegrationMethod.BACKWARD_EULER:
                # BE RHS: I + (C/dt) V_old - G_up V_p
                rhs = I_u + context.B @ V_u - G_up_Vp
            else:
                # Trapezoidal RHS: 2I + (2C/dt) V_old - G V_old - 2 G_up V_p
                rhs = (
                    2.0 * I_u
                    + context.B @ V_u
                    - rc.G_uu @ V_u
                    - 2.0 * G_up_Vp
                )
            V_u = context.lu_A.solve(rhs)

        return float((vdd - V_u[victim_idx]) * 1000.0)

    def _lookup_transient_ir_drop_at_time(
        self,
        victim_node: Any,
        t_target: float,
        context: AdjointSolverContext,
        transient_result: TransientResult,
    ) -> Optional[float]:
        """Look up IR-drop at victim from a pre-computed transient result.

        Returns IR-drop in mV at the timestep nearest ``t_target``, or
        ``None`` if the victim was not in ``track_nodes`` of the forward
        solve (signals "fall back to forward sweep"). Nearest-step (not
        interpolation) matches the fallback's ``round(t_target/dt)``
        rounding, so a swap-in is behavior-preserving when both paths
        would land on the same step.

        Returns:
            IR-drop in mV at the nearest timestep, or ``None`` if the
            victim was not tracked.

        Raises:
            ValueError: If ``transient_result.t_array`` step does not
                match ``context.dt`` to within 1e-15 s, or ``t_target``
                is outside ``t_array`` by more than half a step.
        """
        if victim_node not in transient_result.tracked_ir_drop:
            return None

        t_array = transient_result.t_array
        if len(t_array) >= 2:
            dt_result = float(t_array[1] - t_array[0])
        else:
            dt_result = context.dt

        if dt_result <= 0:
            raise ValueError(
                f"transient_result has non-positive t_array step "
                f"(dt_result={dt_result}); cannot validate against context.dt"
            )
        if abs(dt_result - context.dt) > 1e-15:
            raise ValueError(
                f"transient_result dt={dt_result} does not match "
                f"context.dt={context.dt}"
            )

        t0 = float(t_array[0])
        tL = float(t_array[-1])
        if t_target < t0 - 0.5 * dt_result or t_target > tL + 0.5 * dt_result:
            raise ValueError(
                f"observation_time {t_target} outside transient_result range "
                f"[{t0}, {tL}] (dt={dt_result})"
            )

        # t_array is uniformly spaced (enforced by the dt match check above),
        # so the nearest index is O(1) arithmetic; the clamp is defensive
        # against floating-point on the half-step boundary.
        k = int(round((t_target - t0) / dt_result))
        k = max(0, min(len(t_array) - 1, k))
        # tracked_ir_drop[node] = (vdd - V_node) in volts; convert to mV
        return float(transient_result.tracked_ir_drop[victim_node][k] * 1000.0)

    def _build_top_k_results(
        self,
        contributions: Dict[str, float],
        node_currents: Dict[Any, np.ndarray],
        top_k: int,
        total_ir_drop: float,
        include_waveforms: bool,
        static_contributions: Optional[Dict[str, float]] = None,
    ) -> List[AggressorContribution]:
        """Build top-K aggressor results with waveforms.

        Groups contributions by node (multiple sources may connect to same node).

        Args:
            contributions: Dict of source_name -> contribution in mV
            node_currents: Dict of node -> current waveform
            top_k: Number of top aggressors to return
            total_ir_drop: Total IR-drop for percentage calculation
            include_waveforms: Whether to include waveforms
            static_contributions: Optional dict of source_name -> static contribution
                                 in mV. Only provided in 'dc' mode.

        Returns:
            List of AggressorContribution sorted by |contribution|
        """
        # Group contributions by node
        node_contributions: Dict[Any, float] = {}
        node_static_contributions: Dict[Any, float] = {}
        node_sources: Dict[Any, List[str]] = {}

        for src_name, contrib in contributions.items():
            node = self._source_to_node.get(src_name)
            if node is None:
                continue

            node_contributions[node] = node_contributions.get(node, 0.0) + contrib

            # Also aggregate static contributions if available
            if static_contributions is not None:
                static_contrib = static_contributions.get(src_name, 0.0)
                node_static_contributions[node] = (
                    node_static_contributions.get(node, 0.0) + static_contrib
                )

            if node not in node_sources:
                node_sources[node] = []
            node_sources[node].append(src_name)

        # Sort by |contribution|, take top K
        sorted_nodes = sorted(
            node_contributions.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:top_k]

        results: List[AggressorContribution] = []
        for node, contrib_mV in sorted_nodes:
            waveform = None
            if include_waveforms and node in node_currents:
                waveform = node_currents[node].copy()

            contribution_pct = (
                100.0 * contrib_mV / total_ir_drop if total_ir_drop > 0 else 0.0
            )

            # Get static contribution if available
            static_mV = None
            if static_contributions is not None:
                static_mV = node_static_contributions.get(node, 0.0)

            results.append(AggressorContribution(
                node=node,
                contribution_mV=contrib_mV,
                contribution_pct=contribution_pct,
                source_names=node_sources.get(node, []),
                current_waveform=waveform,
                static_contribution_mV=static_mV,
            ))

        return results

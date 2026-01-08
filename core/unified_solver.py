"""Unified IR-drop solver supporting both flat and hierarchical solving.

This module provides the UnifiedIRDropSolver class that works with
UnifiedPowerGridModel for both synthetic grids and PDN netlists.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np

from .unified_model import UnifiedPowerGridModel, UnifiedReducedSystem, LayerID


@dataclass
class UnifiedSolveResult:
    """Result of a unified IR-drop solve.

    Attributes:
        voltages: Node -> voltage mapping
        ir_drop: Node -> IR-drop mapping (vdd - voltage)
        nominal_voltage: Reference Vdd used for IR-drop calculation
        net_name: Optional net name for multi-net support
        metadata: Additional metadata from the solve
    """
    voltages: Dict[Any, float]
    ir_drop: Dict[Any, float]
    nominal_voltage: float
    net_name: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UnifiedHierarchicalResult:
    """Result of hierarchical IR-drop solve with sub-grid details.

    Attributes:
        voltages: Complete node -> voltage mapping (merged from top and bottom)
        ir_drop: Complete node -> IR-drop mapping
        partition_layer: Layer used for decomposition
        top_grid_voltages: Voltages for top-grid nodes
        bottom_grid_voltages: Voltages for bottom-grid nodes
        port_nodes: Set of port nodes at partition layer
        port_voltages: Port node -> voltage mapping
        port_currents: Port node -> aggregated current
        aggregation_map: Load node -> list of (port, weight, current_contribution)
    """
    voltages: Dict[Any, float]
    ir_drop: Dict[Any, float]
    partition_layer: LayerID
    top_grid_voltages: Dict[Any, float]
    bottom_grid_voltages: Dict[Any, float]
    port_nodes: Set[Any]
    port_voltages: Dict[Any, float]
    port_currents: Dict[Any, float]
    aggregation_map: Dict[Any, List[Tuple[Any, float, float]]] = field(default_factory=dict)


class UnifiedIRDropSolver:
    """Unified IR-drop solver for power grids.

    Supports:
    - Flat (direct) solving
    - Hierarchical solving with top/bottom grid decomposition
    - Both synthetic and PDN sources via UnifiedPowerGridModel

    Example usage:
        model = create_model_from_synthetic(G, pads, vdd=1.0)
        solver = UnifiedIRDropSolver(model)

        # Flat solve
        result = solver.solve(load_currents)

        # Hierarchical solve
        hier_result = solver.solve_hierarchical(load_currents, partition_layer=2)
    """

    def __init__(self, model: UnifiedPowerGridModel):
        """Initialize solver with a unified model.

        Args:
            model: UnifiedPowerGridModel instance
        """
        self.model = model

    @property
    def vdd(self) -> float:
        """Get nominal voltage."""
        return self.model.vdd

    def solve(
        self,
        current_injections: Dict[Any, float],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> UnifiedSolveResult:
        """Solve for voltages using flat (direct) method.

        Args:
            current_injections: Node -> current (positive = sink)
            metadata: Optional metadata to include in result

        Returns:
            UnifiedSolveResult with voltages and IR-drop.
        """
        voltages = self.model.solve_voltages(current_injections)
        ir_drop = self.model.ir_drop(voltages)

        return UnifiedSolveResult(
            voltages=voltages,
            ir_drop=ir_drop,
            nominal_voltage=self.model.vdd,
            net_name=self.model.net_name,
            metadata=metadata or {},
        )

    def solve_batch(
        self,
        stimuli: List[Dict[Any, float]],
        metadatas: Optional[List[Dict]] = None,
    ) -> List[UnifiedSolveResult]:
        """Solve for multiple stimuli (reuses LU factorization).

        Args:
            stimuli: List of current injection dicts
            metadatas: Optional list of metadata dicts

        Returns:
            List of UnifiedSolveResult, one per stimulus.
        """
        metadatas = metadatas or [{}] * len(stimuli)
        results = []

        for currents, meta in zip(stimuli, metadatas):
            result = self.solve(currents, metadata=meta)
            results.append(result)

        return results

    def summarize(self, result: UnifiedSolveResult) -> Dict[str, float]:
        """Compute summary statistics from a solve result.

        Args:
            result: UnifiedSolveResult

        Returns:
            Dict with min_voltage, max_voltage, max_drop, avg_drop.
        """
        voltages = list(result.voltages.values())
        drops = list(result.ir_drop.values())

        return {
            'min_voltage': min(voltages),
            'max_voltage': max(voltages),
            'max_drop': max(drops),
            'avg_drop': sum(drops) / len(drops) if drops else 0.0,
        }

    def solve_hierarchical(
        self,
        current_injections: Dict[Any, float],
        partition_layer: LayerID,
        top_k: int = 5,
        weighting: str = "effective",
    ) -> UnifiedHierarchicalResult:
        """Solve using hierarchical decomposition.

        Decomposes the grid at partition_layer into:
        - Top-grid: layers >= partition_layer (contains pads)
        - Bottom-grid: layers < partition_layer (contains loads)
        - Ports: nodes at partition_layer connecting to bottom-grid

        Steps:
        1. Aggregate bottom-grid currents to ports (using top-k weighting)
        2. Solve top-grid with aggregated port currents
        3. Solve bottom-grid with port voltages as Dirichlet BCs

        Args:
            current_injections: Node -> current (positive = sink)
            partition_layer: Layer to partition at
            top_k: Number of nearest ports for current aggregation
            weighting: "effective" (effective resistance) or "shortest_path"

        Returns:
            UnifiedHierarchicalResult with complete voltages and decomposition info.
        """
        # Decompose the grid
        top_nodes, bottom_nodes, port_nodes, via_edges = self.model._decompose_at_layer(partition_layer)

        # Get currents in bottom-grid only
        bottom_currents = {n: c for n, c in current_injections.items() if n in bottom_nodes}

        # Step 1: Aggregate currents to ports
        port_currents, aggregation_map = self._aggregate_currents_to_ports(
            current_injections=bottom_currents,
            bottom_grid_nodes=bottom_nodes,
            port_nodes=port_nodes,
            top_k=top_k,
            weighting=weighting,
        )

        # Step 2: Solve top-grid with port currents
        top_grid_with_ports = top_nodes | port_nodes
        top_system = self.model._build_subgrid_system(
            subgrid_nodes=top_grid_with_ports,
            dirichlet_nodes=set(self.model.pad_nodes),
            dirichlet_voltage=self.model.vdd,
        )

        if top_system is None:
            raise ValueError("Failed to build top-grid system")

        top_voltages = self.model._solve_subgrid(
            reduced_system=top_system,
            current_injections=port_currents,
        )

        # Extract port voltages
        port_voltages = {p: top_voltages[p] for p in port_nodes if p in top_voltages}

        # Step 3: Solve bottom-grid with port voltages as Dirichlet BCs
        bottom_grid_with_ports = bottom_nodes | port_nodes
        bottom_system = self.model._build_subgrid_system(
            subgrid_nodes=bottom_grid_with_ports,
            dirichlet_nodes=port_nodes,
            dirichlet_voltage=self.model.vdd,
        )

        if bottom_system is None:
            # No bottom-grid to solve (all loads at ports?)
            bottom_voltages = port_voltages.copy()
        else:
            bottom_voltages = self.model._solve_subgrid(
                reduced_system=bottom_system,
                current_injections=bottom_currents,
                dirichlet_voltages=port_voltages,
            )

        # Merge voltages
        all_voltages = {}
        all_voltages.update(top_voltages)
        all_voltages.update(bottom_voltages)

        # Compute IR-drop
        ir_drop = self.model.ir_drop(all_voltages)

        return UnifiedHierarchicalResult(
            voltages=all_voltages,
            ir_drop=ir_drop,
            partition_layer=partition_layer,
            top_grid_voltages=top_voltages,
            bottom_grid_voltages=bottom_voltages,
            port_nodes=port_nodes,
            port_voltages=port_voltages,
            port_currents=port_currents,
            aggregation_map=aggregation_map,
        )

    def _aggregate_currents_to_ports(
        self,
        current_injections: Dict[Any, float],
        bottom_grid_nodes: Set[Any],
        port_nodes: Set[Any],
        top_k: int = 5,
        weighting: str = "effective",
    ) -> Tuple[Dict[Any, float], Dict[Any, List[Tuple[Any, float, float]]]]:
        """Aggregate bottom-grid currents to port nodes.

        For each load node with current, distributes current to top-k nearest
        ports weighted by inverse resistance.

        Args:
            current_injections: Node -> current for bottom-grid nodes
            bottom_grid_nodes: Set of nodes in bottom-grid
            port_nodes: Set of port nodes at partition boundary
            top_k: Number of nearest ports per load
            weighting: "effective" or "shortest_path"

        Returns:
            (port_currents, aggregation_map)
            - port_currents: Port -> aggregated current
            - aggregation_map: Load -> [(port, weight, current_contrib), ...]
        """
        port_currents = {p: 0.0 for p in port_nodes}
        aggregation_map = {}

        if not port_nodes or not current_injections:
            return port_currents, aggregation_map

        port_list = list(port_nodes)

        # Build subgraph for resistance computation
        subgraph_nodes = bottom_grid_nodes | port_nodes
        subgraph = self.model.graph.subgraph(subgraph_nodes)

        for load_node, load_current in current_injections.items():
            if load_current == 0:
                continue

            # Compute resistance to each port
            resistances = {}
            for port in port_list:
                if weighting == "shortest_path":
                    R = self._shortest_path_resistance(subgraph, load_node, port)
                else:
                    # Effective resistance (approximated by shortest path for efficiency)
                    R = self._shortest_path_resistance(subgraph, load_node, port)

                if R is not None and R > 0:
                    resistances[port] = R

            if not resistances:
                # No path to any port - distribute equally
                weight = 1.0 / len(port_list)
                contributions = []
                for port in port_list:
                    contrib = load_current * weight
                    port_currents[port] += contrib
                    contributions.append((port, weight, contrib))
                aggregation_map[load_node] = contributions
                continue

            # Sort by resistance and take top-k
            sorted_ports = sorted(resistances.items(), key=lambda x: x[1])
            selected = sorted_ports[:top_k] if top_k < len(sorted_ports) else sorted_ports

            # Compute weights (inverse resistance)
            inv_R = [(p, 1.0 / R) for p, R in selected]
            total_inv_R = sum(w for _, w in inv_R)

            # Distribute current
            contributions = []
            for port, inv_r in inv_R:
                weight = inv_r / total_inv_R
                contrib = load_current * weight
                port_currents[port] += contrib
                contributions.append((port, weight, contrib))

            aggregation_map[load_node] = contributions

        return port_currents, aggregation_map

    def _shortest_path_resistance(
        self,
        subgraph: nx.Graph,
        source: Any,
        target: Any,
    ) -> Optional[float]:
        """Compute shortest path resistance between two nodes.

        Uses Dijkstra with edge resistance as weight.

        Args:
            subgraph: Graph to search in
            source: Source node
            target: Target node

        Returns:
            Total resistance along shortest path, or None if no path.
        """
        if source == target:
            return 0.0

        if source not in subgraph or target not in subgraph:
            return None

        # Build weight function
        def get_weight(u, v, data):
            if isinstance(subgraph, nx.MultiDiGraph):
                # For MultiDiGraph, data is already the edge dict
                R = data.get('value', 0.0) if data.get('type') == 'R' else float('inf')
                if self.model.resistance_unit_kohm:
                    R = R * 1e3
                return R if R > 0 else float('inf')
            else:
                R = data.get('resistance', 0.0)
                return R if R > 0 else float('inf')

        try:
            # Use networkx shortest path with custom weight
            length = nx.dijkstra_path_length(subgraph, source, target, weight=get_weight)
            return length if length < float('inf') else None
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

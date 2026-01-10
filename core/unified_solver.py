"""Unified IR-drop solver supporting both flat and hierarchical solving.

This module provides the UnifiedIRDropSolver class that works with
UnifiedPowerGridModel for both synthetic grids and PDN netlists.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import heapq
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

        Statistics exclude pad nodes to report only free (non-vsrc) node behavior,
        consistent with PDNSolver reporting conventions.

        Args:
            result: UnifiedSolveResult

        Returns:
            Dict with nominal_voltage, min_voltage, max_voltage, max_drop, avg_drop.
        """
        # Exclude pad nodes from statistics (they are fixed at nominal voltage)
        pad_set = set(self.model.pad_nodes)
        free_voltages = [v for n, v in result.voltages.items() if n not in pad_set]
        free_drops = [d for n, d in result.ir_drop.items() if n not in pad_set]

        return {
            'nominal_voltage': result.nominal_voltage,
            'min_voltage': min(free_voltages) if free_voltages else result.nominal_voltage,
            'max_voltage': max(free_voltages) if free_voltages else result.nominal_voltage,
            'max_drop': max(free_drops) if free_drops else 0.0,
            'avg_drop': sum(free_drops) / len(free_drops) if free_drops else 0.0,
        }

    def solve_hierarchical(
        self,
        current_injections: Dict[Any, float],
        partition_layer: LayerID,
        top_k: int = 5,
        weighting: str = "effective",
        rmax: Optional[float] = None,
    ) -> UnifiedHierarchicalResult:
        """Solve using hierarchical decomposition.

        Decomposes the grid at partition_layer into:
        - Top-grid: layers >= partition_layer (contains pads)
        - Bottom-grid: layers < partition_layer (contains loads)
        - Ports: nodes at partition_layer connecting to bottom-grid

        Steps:
        1. Aggregate bottom-grid currents to ports (using top-k weighting)
        2. Solve top-grid with pad voltages as Dirichlet BC and port injections
        3. Solve bottom-grid with port voltages as Dirichlet BCs

        Args:
            current_injections: Node -> current (positive = sink)
            partition_layer: Layer to partition at
            top_k: Number of nearest ports for current aggregation
            weighting: "effective" (effective resistance) or "shortest_path"
            rmax: Maximum resistance distance for shortest_path weighting.
                  Paths beyond this distance are ignored. None means no limit.
                  Only applies when weighting="shortest_path".

        Returns:
            UnifiedHierarchicalResult with complete voltages and decomposition info.

        Raises:
            ValueError: If partition layer has no ports, or if load nodes are
                electrically disconnected from ports. The error message will
                suggest alternative partition layers if disconnection is detected.
        """
        # Step 0: Decompose the grid
        top_nodes, bottom_nodes, port_nodes, via_edges = self.model._decompose_at_layer(partition_layer)

        if not port_nodes:
            raise ValueError(
                f"No ports found at partition layer {partition_layer}. "
                "Check that vias connect this layer to the layer below."
            )

        # Step 0.5: Validate load-to-port connectivity before expensive computation
        bottom_load_nodes = {
            n for n, c in current_injections.items()
            if n in bottom_nodes and n not in port_nodes and c != 0
        }
        if bottom_load_nodes:
            disconnected_loads = self._find_disconnected_loads(
                bottom_grid_nodes=bottom_nodes,
                port_nodes=port_nodes,
                load_nodes=bottom_load_nodes,
            )
            if disconnected_loads:
                # Find alternative partition layers that might work better
                suggestions = self._suggest_partition_layers(
                    disconnected_loads=disconnected_loads,
                    current_partition=partition_layer,
                )
                raise ValueError(
                    f"{len(disconnected_loads)} load node(s) are electrically disconnected from "
                    f"ports at partition layer {partition_layer}. "
                    f"Example disconnected load: {next(iter(disconnected_loads))}. "
                    f"This typically means the partition layer is too high. "
                    f"{suggestions}"
                )

        # Step 1: Aggregate bottom-grid currents onto ports
        port_currents, aggregation_map = self._aggregate_currents_to_ports(
            current_injections=current_injections,
            bottom_grid_nodes=bottom_nodes,
            port_nodes=port_nodes,
            top_k=top_k,
            weighting=weighting,
            rmax=rmax,
        )

        # Step 2: Solve top-grid with pads as Dirichlet BC
        # Top-grid includes: top_nodes (layers >= partition_layer)
        # Dirichlet nodes: pads (on top layer)
        pad_set = set(self.model.pad_nodes)
        top_grid_pads = pad_set & top_nodes

        if not top_grid_pads:
            raise ValueError(
                f"No pad nodes found in top-grid (layers >= {partition_layer}). "
                "Pads should be on the top-most layer."
            )

        # Build and solve top-grid system
        top_system = self.model._build_subgrid_system(
            subgrid_nodes=top_nodes,
            dirichlet_nodes=top_grid_pads,
            dirichlet_voltage=self.model.vdd,
        )

        if top_system is None:
            raise ValueError("Failed to build top-grid system")

        # Current injections for top-grid: loads in top-grid + aggregated port currents
        # Note: Any loads that happen to be in top-grid are handled directly
        top_grid_currents = {n: c for n, c in current_injections.items() if n in top_nodes}
        # Add aggregated port currents
        for port, curr in port_currents.items():
            top_grid_currents[port] = top_grid_currents.get(port, 0.0) + curr

        top_voltages = self.model._solve_subgrid(
            reduced_system=top_system,
            current_injections=top_grid_currents,
            dirichlet_voltages=None,  # Use vdd for pads
        )

        # Extract port voltages for bottom-grid BC
        port_voltages = {p: top_voltages[p] for p in port_nodes}

        # Step 3: Solve bottom-grid with port voltages as Dirichlet BC
        # Bottom-grid includes: bottom_nodes (layers < partition_layer)
        # We need to include the ports as Dirichlet nodes but they're in top-grid
        # Solution: Include ports in the bottom subgrid for the solve
        bottom_subgrid = bottom_nodes | port_nodes

        bottom_system = self.model._build_subgrid_system(
            subgrid_nodes=bottom_subgrid,
            dirichlet_nodes=port_nodes,
            dirichlet_voltage=self.model.vdd,  # Will be overridden by dirichlet_voltages
        )

        if bottom_system is None:
            raise ValueError("Failed to build bottom-grid system")

        # Current injections for bottom-grid: original currents in bottom-grid
        bottom_grid_currents = {
            n: c for n, c in current_injections.items()
            if n in bottom_nodes
        }

        bottom_voltages = self.model._solve_subgrid(
            reduced_system=bottom_system,
            current_injections=bottom_grid_currents,
            dirichlet_voltages=port_voltages,
        )

        # Merge voltages: top-grid + bottom-grid (excluding duplicate ports)
        all_voltages = {}
        all_voltages.update(top_voltages)
        # Add bottom-grid nodes (excluding ports which are already in top_voltages)
        for n, v in bottom_voltages.items():
            if n not in port_nodes:
                all_voltages[n] = v

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

    def _find_disconnected_loads(
        self,
        bottom_grid_nodes: Set[Any],
        port_nodes: Set[Any],
        load_nodes: Set[Any],
    ) -> Set[Any]:
        """Find load nodes that are electrically disconnected from ports.

        Uses connected components analysis on R-type edges to identify loads
        that have no resistive path to any port node.

        Args:
            bottom_grid_nodes: Set of nodes in bottom-grid
            port_nodes: Set of port nodes at partition boundary
            load_nodes: Set of load nodes to check

        Returns:
            Set of load nodes that are disconnected from all ports.
        """
        # Build R-type subgraph for bottom-grid + ports
        subgraph_nodes = bottom_grid_nodes | port_nodes
        
        # Use Union-Find for efficient component detection
        parent: Dict[Any, Any] = {}
        
        def find(x: Any) -> Any:
            if x not in parent:
                parent[x] = x
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x: Any, y: Any) -> None:
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
        
        # Process edges using EdgeInfoExtractor to handle both synthetic and PDN formats
        edge_extractor = self.model._edge_extractor
        for u, v, d in self.model.graph.edges(subgraph_nodes, data=True):
            if u not in subgraph_nodes or v not in subgraph_nodes:
                continue
            edge_info = edge_extractor.get_info(d)
            if not edge_info.is_resistive:
                continue
            R = edge_info.resistance if edge_info.resistance else 0.0
            if R > 0 and R < float('inf'):
                union(u, v)
                union(u, v)
        
        # Find components containing ports
        port_components = {find(p) for p in port_nodes if p in parent}
        
        # Find loads not in any port-containing component
        disconnected = set()
        for load in load_nodes:
            if load not in parent:
                # Node has no R-type edges at all
                disconnected.add(load)
            elif find(load) not in port_components:
                disconnected.add(load)
        
        return disconnected

    def _suggest_partition_layers(
        self,
        disconnected_loads: Set[Any],
        current_partition: LayerID,
    ) -> str:
        """Suggest alternative partition layers that might include disconnected loads.

        Args:
            disconnected_loads: Set of load nodes disconnected from current partition
            current_partition: The partition layer that failed

        Returns:
            String with suggestions for alternative partition layers.
        """
        # Get layers present in disconnected loads
        load_layers: Dict[LayerID, int] = {}
        for load in list(disconnected_loads)[:1000]:  # Sample for efficiency
            info = self.model.get_node_info(load)
            if info.layer is not None:
                load_layers[info.layer] = load_layers.get(info.layer, 0) + 1
        
        if not load_layers:
            return "Could not determine layers of disconnected loads."
        
        # Find the highest layer with disconnected loads
        all_layers = sorted(self.model.get_all_layers())
        pad_set = set(self.model.pad_nodes)
        
        # Try lower partition layers and check:
        # 1. Has ports
        # 2. Has pads in top-grid
        suggestions = []
        
        for layer in reversed(all_layers):
            if layer >= current_partition:
                continue
            
            try:
                top_nodes, bottom_nodes, ports, _ = self.model._decompose_at_layer(layer)
            except ValueError:
                # Skip invalid partition layers (e.g., bottom layer)
                continue
            if not ports:
                continue
            
            # Check if pads are in top-grid
            pads_in_top = pad_set & top_nodes
            if not pads_in_top:
                continue
            
            suggestions.append(f"layer {layer} ({len(ports)} ports, {len(pads_in_top)} pads in top-grid)")
            if len(suggestions) >= 3:
                break
        
        if suggestions:
            return f"Try partitioning at a lower layer: {', '.join(suggestions)}."
        return "Try using a lower partition layer closer to the load layers."

    def _aggregate_currents_to_ports(
        self,
        current_injections: Dict[Any, float],
        bottom_grid_nodes: Set[Any],
        port_nodes: Set[Any],
        top_k: int = 5,
        weighting: str = "effective",
        rmax: Optional[float] = None,
    ) -> Tuple[Dict[Any, float], Dict[Any, List[Tuple[Any, float, float]]]]:
        """Aggregate bottom-grid currents to port nodes.

        For each load node with current in bottom-grid, distributes current to
        top-k nearest ports weighted by inverse resistance.

        Args:
            current_injections: Node -> current (can include nodes outside bottom-grid)
            bottom_grid_nodes: Set of nodes in bottom-grid
            port_nodes: Set of port nodes at partition boundary
            top_k: Number of nearest ports per load
            weighting: "effective" or "shortest_path"
            rmax: Maximum resistance distance for shortest_path (None = no limit)

        Returns:
            (port_currents, aggregation_map)
            - port_currents: Port -> aggregated current
            - aggregation_map: Load -> [(port, weight, current_contrib), ...]
        """
        port_currents = {p: 0.0 for p in port_nodes}
        aggregation_map = {}

        if not port_nodes or not current_injections:
            return port_currents, aggregation_map

        # Filter currents to only those in bottom-grid (excluding ports)
        bottom_currents = {
            n: c for n, c in current_injections.items()
            if n in bottom_grid_nodes and n not in port_nodes
        }

        if not bottom_currents:
            return port_currents, aggregation_map

        port_list = list(port_nodes)
        subgraph_nodes = bottom_grid_nodes | port_nodes

        # Precompute multi-source shortest-path distances once for all loads
        shortest_path_cache: Dict[Any, List[Tuple[Any, float]]] = {}
        if weighting == "shortest_path":
            shortest_path_cache = self._multi_source_port_distances(
                subgrid_nodes=subgraph_nodes,
                port_nodes=port_list,
                top_k=top_k,
                rmax=rmax,
            )

        for load_node, load_current in bottom_currents.items():
            if load_current == 0:
                continue

            # Compute resistance to each port
            if weighting == "effective":
                resistances = self._compute_effective_resistance_in_subgrid(
                    subgrid_nodes=subgraph_nodes,
                    source_node=load_node,
                    target_nodes=port_list,
                    dirichlet_nodes=port_nodes,
                )
            else:
                port_distances = shortest_path_cache.get(load_node, [])
                resistances = {p: d for p, d in port_distances if d is not None}

            valid_resistances = {
                p: r for p, r in resistances.items()
                if r is not None and r < float('inf') and r > 0
            }

            if not valid_resistances:
                raise ValueError(
                    f"No valid resistance paths found from load node {load_node} to any port. "
                    f"This indicates the load is electrically isolated from the port layer. "
                    f"Check grid connectivity."
                )

            sorted_ports = sorted(valid_resistances.items(), key=lambda x: x[1])
            selected = sorted_ports[:top_k] if top_k < len(sorted_ports) else sorted_ports

            inv_R = [(p, 1.0 / R) for p, R in selected]
            total_inv_R = sum(w for _, w in inv_R)
            if total_inv_R == 0:
                raise ValueError(
                    f"Total inverse resistance is zero for load node {load_node}. "
                    f"Selected ports: {[p for p, _ in selected]}. "
                    f"This should not happen with valid_resistances already filtered."
                )

            contributions = []
            for port, inv_r in inv_R:
                weight = inv_r / total_inv_R
                contrib = load_current * weight
                port_currents[port] += contrib
                contributions.append((port, weight, contrib))

            aggregation_map[load_node] = contributions

        return port_currents, aggregation_map

    def _build_resistive_csr(
        self,
        subgrid_nodes: Set[Any],
    ) -> Tuple[List[Any], Dict[Any, int], np.ndarray, np.ndarray, np.ndarray]:
        """Build CSR adjacency with resistance weights for a node set."""
        if not subgrid_nodes:
            return [], {}, np.array([], dtype=np.int64), np.array([], dtype=np.int32), np.array([], dtype=float)

        node_list = list(subgrid_nodes)
        index = {n: i for i, n in enumerate(node_list)}

        edge_min = {}
        graph = self.model.graph
        node_set = set(subgrid_nodes)

        for u, v, edge_data in graph.edges(subgrid_nodes, data=True):
            if u not in node_set or v not in node_set:
                continue

            if isinstance(graph, nx.MultiDiGraph):
                if edge_data.get('type') != 'R':
                    continue
                R = float(edge_data.get('value', 0.0))
                if self.model.resistance_unit_kohm:
                    R *= 1e3
            else:
                R = float(edge_data.get('resistance', 0.0))

            if R <= 0.0 or not np.isfinite(R):
                continue

            iu, iv = index[u], index[v]
            if iu == iv:
                continue

            key = (iu, iv) if iu < iv else (iv, iu)
            prev = edge_min.get(key)
            if prev is None or R < prev:
                edge_min[key] = R

        n = len(node_list)
        adjacency: List[List[Tuple[int, float]]] = [[] for _ in range(n)]
        for (iu, iv), R in edge_min.items():
            adjacency[iu].append((iv, R))
            adjacency[iv].append((iu, R))

        indptr = np.zeros(n + 1, dtype=np.int64)
        indices: List[int] = []
        data: List[float] = []

        for i, nbrs in enumerate(adjacency):
            for j, w in nbrs:
                indices.append(j)
                data.append(w)
            indptr[i + 1] = len(indices)

        return (
            node_list,
            index,
            indptr,
            np.array(indices, dtype=np.int32),
            np.array(data, dtype=float),
        )

    def _multi_source_port_distances(
        self,
        subgrid_nodes: Set[Any],
        port_nodes: List[Any],
        top_k: int,
        rmax: Optional[float] = None,
    ) -> Dict[Any, List[Tuple[Any, float]]]:
        """Compute up to top_k nearest ports for every node via multi-source Dijkstra.

        Uses an optimized label-setting algorithm that tracks settled (port, node)
        pairs to avoid redundant heap operations. For large graphs (1M+ nodes,
        10K+ ports), this avoids heap explosion by pruning aggressively.

        Args:
            subgrid_nodes: Set of nodes to consider
            port_nodes: List of port nodes (sources for Dijkstra)
            top_k: Number of nearest ports to track per node
            rmax: Maximum resistance distance. Paths beyond this are pruned.
                  None means no limit (default behavior).

        Complexity: O((N + E) * K * log(N * K)) where N=nodes, E=edges, K=top_k
                    With rmax, complexity can be significantly reduced as
                    propagation stops at the distance boundary.
        """
        node_list, index, indptr, indices, weights = self._build_resistive_csr(subgrid_nodes)
        if not node_list or not port_nodes:
            return {}

        top_k = max(1, top_k)
        n = len(node_list)

        # For each node, store dict {port_id: best_distance} for O(1) lookup
        best_dist: List[Dict[int, float]] = [{} for _ in range(n)]
        # Max-heap (negated distances) to track k-th best distance in O(1)
        # Format: List of max-heaps where heap[i] contains (-dist, port_id)
        kth_heaps: List[List[Tuple[float, int]]] = [[] for _ in range(n)]
        # Track settled (port_id, node_idx) pairs to skip redundant processing
        settled: Set[Tuple[int, int]] = set()

        # Priority queue: (distance, port_id, node_idx)
        heap: List[Tuple[float, int, int]] = []
        for port_id, port in enumerate(port_nodes):
            if port not in index:
                continue
            node_idx = index[port]
            heapq.heappush(heap, (0.0, port_id, node_idx))
            best_dist[node_idx][port_id] = 0.0
            heapq.heappush(kth_heaps[node_idx], (0.0, port_id))  # max-heap uses negated dist, but 0 is same

        while heap:
            dist, port_id, node_idx = heapq.heappop(heap)

            # Early termination: skip if distance exceeds rmax
            if rmax is not None and dist > rmax:
                continue

            # Skip if this (port, node) pair was already settled
            key = (port_id, node_idx)
            if key in settled:
                continue

            # Skip if we've found a better path since this was queued
            node_best = best_dist[node_idx]
            if port_id in node_best and dist > node_best[port_id]:
                continue

            # Mark as settled - this is the optimal distance for this (port, node) pair
            settled.add(key)

            # Propagate to neighbors
            start, end = indptr[node_idx], indptr[node_idx + 1]
            for offset in range(start, end):
                nbr = int(indices[offset])
                w = float(weights[offset])
                if w <= 0.0 or not np.isfinite(w):
                    continue

                new_dist = dist + w
                nbr_best = best_dist[nbr]
                nbr_kth_heap = kth_heaps[nbr]

                # Skip if already settled for this port
                if (port_id, nbr) in settled:
                    continue

                # Pruning: if neighbor already has top_k labels and new_dist
                # is worse than the k-th best, skip (O(1) check using max-heap)
                if len(nbr_kth_heap) >= top_k:
                    # Max-heap root has the largest (worst) of top-k distances
                    kth_best = -nbr_kth_heap[0][0]  # negate to get actual distance
                    if new_dist >= kth_best:
                        continue

                # Skip if new distance exceeds rmax
                if rmax is not None and new_dist > rmax:
                    continue

                # Check if this improves existing distance for this port
                if port_id in nbr_best:
                    if new_dist >= nbr_best[port_id]:
                        continue
                    # Update existing - need to rebuild heap (rare case)
                    nbr_best[port_id] = new_dist
                else:
                    # New port for this node
                    nbr_best[port_id] = new_dist

                # Update the k-th best heap
                if len(nbr_kth_heap) < top_k:
                    heapq.heappush(nbr_kth_heap, (-new_dist, port_id))
                elif new_dist < -nbr_kth_heap[0][0]:
                    # Replace the worst (largest distance) in top-k
                    heapq.heapreplace(nbr_kth_heap, (-new_dist, port_id))

                heapq.heappush(heap, (new_dist, port_id, nbr))

        # Build result: for each node, return top-k (port, distance) pairs
        result: Dict[Any, List[Tuple[Any, float]]] = {}
        for idx in range(n):
            node_best = best_dist[idx]
            if not node_best:
                continue
            # Sort by distance and take top_k
            sorted_labels = sorted(node_best.items(), key=lambda x: x[1])[:top_k]
            node = node_list[idx]
            result[node] = [(port_nodes[pid], d) for pid, d in sorted_labels]

        return result

    def _compute_effective_resistance_in_subgrid(
        self,
        subgrid_nodes: Set[Any],
        source_node: Any,
        target_nodes: List[Any],
        dirichlet_nodes: Set[Any],
    ) -> Dict[Any, float]:
        """Compute effective resistance from source to targets within a subgrid.

        Properly computes electrical effective resistance considering all parallel
        paths, with specified nodes as Dirichlet (fixed voltage) boundaries.

        Args:
            subgrid_nodes: Set of nodes forming the subgrid
            source_node: Source node for resistance computation
            target_nodes: List of target nodes (typically ports)
            dirichlet_nodes: Nodes with fixed voltage (ports)

        Returns:
            Dict mapping target -> effective resistance (Ohms)
        """
        import scipy.sparse as sp
        import scipy.sparse.linalg as spla

        # Build subgraph
        subgraph = self.model.graph.subgraph(subgrid_nodes)
        nodes = list(subgraph.nodes())

        if source_node not in nodes:
            return {t: float('inf') for t in target_nodes}

        index = {n: i for i, n in enumerate(nodes)}
        n_nodes = len(nodes)

        # Build sparse conductance matrix
        data, rows, cols = [], [], []
        diag = np.zeros(n_nodes, dtype=float)

        for u, v, edge_data in subgraph.edges(data=True):
            # Get resistance based on graph type
            if isinstance(self.model.graph, nx.MultiDiGraph):
                # PDN: R type edges
                if edge_data.get('type') == 'R':
                    R = float(edge_data.get('value', 0.0))
                    if self.model.resistance_unit_kohm:
                        R = R * 1e3
                else:
                    continue
            else:
                # Synthetic: all edges have resistance attribute
                R = float(edge_data.get('resistance', 0.0))

            if R <= 0.0:
                continue

            g = 1.0 / R
            iu, iv = index[u], index[v]
            rows.extend([iu, iv])
            cols.extend([iv, iu])
            data.extend([-g, -g])
            diag[iu] += g
            diag[iv] += g

        for i in range(n_nodes):
            rows.append(i)
            cols.append(i)
            data.append(diag[i])

        G_mat = sp.csr_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))

        # Separate Dirichlet (fixed) nodes from unknowns
        dirichlet_in_subgrid = dirichlet_nodes & set(nodes)
        unknown_nodes = [n for n in nodes if n not in dirichlet_in_subgrid]

        if source_node in dirichlet_in_subgrid:
            return {t: (0.0 if t == source_node else float('inf')) for t in target_nodes}

        if not unknown_nodes:
            return {t: float('inf') for t in target_nodes}

        index_unknown = {n: i for i, n in enumerate(unknown_nodes)}
        u_idx = [index[n] for n in unknown_nodes]
        G_uu = G_mat[np.ix_(u_idx, u_idx)].tocsr()

        try:
            lu = spla.factorized(G_uu.tocsc())
        except Exception:
            return {t: float('inf') for t in target_nodes}

        if source_node not in index_unknown:
            return {t: float('inf') for t in target_nodes}

        source_idx = index_unknown[source_node]
        e_source = np.zeros(len(unknown_nodes))
        e_source[source_idx] = 1.0
        x_source = lu(e_source)

        results = {}

        # Get coupling matrix G_ud for computing resistance to Dirichlet nodes
        dirichlet_list = list(dirichlet_in_subgrid)
        d_idx = [index[d] for d in dirichlet_list]
        if d_idx:
            G_ud = G_mat[np.ix_(u_idx, d_idx)].tocsr()
        else:
            G_ud = sp.csr_matrix((len(u_idx), 0))

        for target in target_nodes:
            if target not in dirichlet_in_subgrid:
                # Target is unknown node
                if target in index_unknown:
                    target_idx = index_unknown[target]
                    e_target = np.zeros(len(unknown_nodes))
                    e_target[target_idx] = 1.0
                    x_target = lu(e_target)
                    # R_eff = x_s[s] + x_t[t] - 2*x_s[t]
                    r_eff = (x_source[source_idx] + x_target[target_idx]
                             - x_source[target_idx] - x_target[source_idx])
                    results[target] = max(0.0, r_eff)
                else:
                    results[target] = float('inf')
            else:
                # Target is a Dirichlet node (port)
                if G_ud.shape[1] > 0:
                    target_d_idx = None
                    for i, d in enumerate(dirichlet_list):
                        if d == target:
                            target_d_idx = i
                            break

                    if target_d_idx is not None:
                        target_col = G_ud[:, target_d_idx].toarray().flatten()
                        current_to_target = np.dot(target_col, x_source)

                        if abs(current_to_target) > 1e-10:
                            r_eff = x_source[source_idx] / abs(current_to_target)
                            results[target] = max(0.0, r_eff)
                        else:
                            results[target] = float('inf')
                    else:
                        results[target] = float('inf')
                else:
                    results[target] = max(0.0, x_source[source_idx])

        return results

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

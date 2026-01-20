"""Current aggregation utilities for hierarchical IR-drop solving.

This module provides the CurrentAggregator class for distributing load currents
to port nodes using shortest-path or effective resistance weighting.
"""

from __future__ import annotations

import heapq
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .unified_model import UnifiedPowerGridModel
from .rx_graph import RustworkxGraphWrapper, RustworkxMultiDiGraphWrapper
from .rx_algorithms import dijkstra_path_length, NetworkXNoPath, NodeNotFound


class CurrentAggregator:
    """Aggregates load currents to ports using shortest-path or effective resistance.

    This class handles the distribution of bottom-grid currents to port nodes
    during hierarchical solving. It supports two weighting schemes:
    - shortest_path: Uses cumulative resistance along shortest path
    - effective: Uses electrical effective resistance (more accurate but slower)
    """

    def __init__(self, model: UnifiedPowerGridModel):
        """Initialize aggregator with a unified model.

        Args:
            model: UnifiedPowerGridModel instance
        """
        self.model = model

    def aggregate_currents_to_ports(
        self,
        current_injections: Dict[Any, float],
        bottom_grid_nodes: Set[Any],
        port_nodes: Set[Any],
        top_k: int = 5,
        weighting: str = "shortest_path",
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
        aggregation_map: Dict[Any, List[Tuple[Any, float, float]]] = {}

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
            shortest_path_cache = self.multi_source_port_distances(
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
                resistances = self.compute_effective_resistance_in_subgrid(
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

    def build_resistive_csr(
        self,
        subgrid_nodes: Set[Any],
    ) -> Tuple[List[Any], Dict[Any, int], np.ndarray, np.ndarray, np.ndarray]:
        """Build CSR adjacency with resistance weights for a node set.

        Args:
            subgrid_nodes: Set of nodes to include in the CSR graph

        Returns:
            (node_list, index, indptr, indices, data)
            - node_list: List of nodes in order
            - index: Node -> index mapping
            - indptr: CSR row pointers
            - indices: CSR column indices
            - data: CSR data (resistance weights)
        """
        if not subgrid_nodes:
            return [], {}, np.array([], dtype=np.int64), np.array([], dtype=np.int32), np.array([], dtype=float)

        node_list = list(subgrid_nodes)
        index = {n: i for i, n in enumerate(node_list)}

        edge_min: Dict[Tuple[int, int], float] = {}
        graph = self.model.graph
        node_set = set(subgrid_nodes)

        for u, v, edge_data in graph.edges(subgrid_nodes, data=True):
            if u not in node_set or v not in node_set:
                continue

            if isinstance(graph, RustworkxMultiDiGraphWrapper):
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

    def multi_source_port_distances(
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

        Returns:
            Dict mapping node -> list of (port, distance) tuples, sorted by distance

        Complexity: O((N + E) * K * log(N * K)) where N=nodes, E=edges, K=top_k
                    With rmax, complexity can be significantly reduced as
                    propagation stops at the distance boundary.
        """
        node_list, index, indptr, indices, weights = self.build_resistive_csr(subgrid_nodes)
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

    def compute_effective_resistance_in_subgrid(
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
            if isinstance(self.model.graph, RustworkxMultiDiGraphWrapper):
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

        results: Dict[Any, float] = {}

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

    def shortest_path_resistance(
        self,
        subgraph: "RustworkxGraphWrapper | RustworkxMultiDiGraphWrapper",
        source: Any,
        target: Any,
    ) -> Optional[float]:
        """Compute shortest path resistance between two nodes.

        Uses Dijkstra with edge resistance as weight.

        Args:
            subgraph: Graph wrapper to search in
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
        is_pdn = isinstance(subgraph, RustworkxMultiDiGraphWrapper)

        def get_weight(u: Any, v: Any, data: Dict[str, Any]) -> float:
            if is_pdn:
                # For MultiDiGraph, data is already the edge dict
                R = data.get('value', 0.0) if data.get('type') == 'R' else float('inf')
                if self.model.resistance_unit_kohm:
                    R = R * 1e3
                return R if R > 0 else float('inf')
            else:
                R = data.get('resistance', 0.0)
                return R if R > 0 else float('inf')

        try:
            # Use rustworkx-based shortest path with custom weight
            length = dijkstra_path_length(subgraph, source, target, weight_fn=get_weight)
            return length if length < float('inf') else None
        except (NetworkXNoPath, NodeNotFound):
            return None

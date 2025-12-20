"""Power grid model utilities for static IR-drop analysis.

Builds a sparse conductance matrix from the graph produced by
`generate_power_grid.generate_power_grid`.

Terminology:
  - Pads: voltage source nodes assumed fixed at Vdd (default 1.0V)
  - Loads: current sink nodes (positive current draws from the grid)

We construct the nodal equation: G * V = I, where
  G: nodal conductance matrix (symmetric positive definite for connected grids)
  I: net current injection vector (pads removed from unknown set)

Pads are treated as Dirichlet boundary conditions (fixed voltage). We perform
Schur reduction to form the reduced system on unknown nodes U:
  (G_UU) * V_U = I_U - G_UP * V_P
where P are pad nodes.

The reduced matrix is cached for reuse across multiple stimulus solves.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple, Set, Optional

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import networkx as nx


@dataclass
class HierarchicalSolveResult:
    """Result of hierarchical IR-drop solve with sub-grid details.

    Attributes:
        voltages: Complete node -> voltage mapping (merged from top and bottom)
        ir_drop: Complete node -> IR-drop mapping
        partition_layer: The layer index M_p used for decomposition
        top_grid_voltages: Voltages for nodes in top-grid (layers M_p to M_T)
        bottom_grid_voltages: Voltages for nodes in bottom-grid (layers M_0 to M_p-1)
        port_nodes: Set of port nodes at M_p (interface between grids)
        port_voltages: Port node -> voltage mapping
        port_currents: Port node -> aggregated current injection
        aggregation_map: Load node -> list of (port, weight, current_contribution)
    """
    voltages: Dict
    ir_drop: Dict
    partition_layer: int
    top_grid_voltages: Dict
    bottom_grid_voltages: Dict
    port_nodes: Set
    port_voltages: Dict
    port_currents: Dict
    aggregation_map: Dict = field(default_factory=dict)


@dataclass
class ReducedSystem:
    """Holds factorization and mapping for solving nodal voltages.

    Attributes:
        node_order: list of all nodes in matrix order
        unknown_nodes: list of nodes solved for (non-pad)
        pad_nodes: list of pad nodes (Dirichlet)
        G_uu: sparse conductance submatrix for unknowns
        G_up: sparse coupling between unknowns and pads
        lu: factorization object (from spla.factorized) for fast solves
        pad_voltage: float voltage applied at pad nodes
        index_of: dict mapping node -> index in full ordering
        index_unknown: dict mapping node -> index in unknown ordering
    """

    node_order: List
    unknown_nodes: List
    pad_nodes: List
    G_uu: sp.csr_matrix
    G_up: sp.csr_matrix
    lu: callable
    pad_voltage: float
    index_of: Dict
    index_unknown: Dict


class PowerGridModel:
    """Wraps the graph and builds sparse matrices for IR-drop solving."""

    def __init__(self, G: nx.Graph, pad_nodes: Sequence, vdd: float = 1.0):
        self.G = G
        self.pad_nodes = list(pad_nodes)
        self.vdd = float(vdd)
        # Pre-build reduced system
        self._reduced = self._build_reduced_system()

    def _build_conductance_matrix(self) -> Tuple[sp.csr_matrix, List]:
        """Return (G_matrix, node_order).

        Each resistor edge (u,v) with resistance R contributes conductance g=1/R.
        We build the Laplacian-like nodal conductance matrix.
        """
        nodes = list(self.G.nodes())
        index = {n: i for i, n in enumerate(nodes)}
        data = []
        rows = []
        cols = []
        # Accumulate diagonal and off-diagonal entries
        diag = np.zeros(len(nodes), dtype=float)
        for u, v, d in self.G.edges(data=True):
            R = float(d.get("resistance", 0.0))
            if R <= 0.0:
                continue
            g = 1.0 / R
            iu = index[u]; iv = index[v]
            # Off-diagonal
            rows.append(iu); cols.append(iv); data.append(-g)
            rows.append(iv); cols.append(iu); data.append(-g)
            # Diagonal accumulation
            diag[iu] += g
            diag[iv] += g
        # Insert diagonal entries
        for n, i in index.items():
            rows.append(i); cols.append(i); data.append(diag[i])
        G_mat = sp.csr_matrix((data, (rows, cols)), shape=(len(nodes), len(nodes)))
        return G_mat, nodes

    def _build_reduced_system(self) -> ReducedSystem:
        G_mat, nodes = self._build_conductance_matrix()
        pad_set = set(self.pad_nodes)
        unknown_nodes = [n for n in nodes if n not in pad_set]
        pad_nodes = [n for n in nodes if n in pad_set]
        index = {n: i for i, n in enumerate(nodes)}
        index_unknown = {n: i for i, n in enumerate(unknown_nodes)}
        # Extract submatrices
        # Unknown rows/cols
        u_idx = [index[n] for n in unknown_nodes]
        p_idx = [index[n] for n in pad_nodes]
        G_uu = G_mat[u_idx][:, u_idx].tocsr()
        G_up = G_mat[u_idx][:, p_idx].tocsr()
        # Factorize G_uu for repeated solves
        # Use a robust factorization routine; for SPD we can use spla.factorized
        lu = spla.factorized(G_uu)  # returns a callable solve(rhs)
        return ReducedSystem(
            node_order=nodes,
            unknown_nodes=unknown_nodes,
            pad_nodes=pad_nodes,
            G_uu=G_uu,
            G_up=G_up,
            lu=lu,
            pad_voltage=self.vdd,
            index_of=index,
            index_unknown=index_unknown,
        )

    @property
    def reduced(self) -> ReducedSystem:
        return self._reduced

    def solve_voltages(self, current_injections: Dict, assume_missing_zero: bool = True) -> Dict:
        """Solve for nodal voltages given current injections at load nodes.

        current_injections: mapping node -> current (Amps). Convention: positive
            current means drawing current from the grid (sink). The nodal equation
            G * V = I uses net current injection (sources positive). Therefore we
            flip sign: I_node = -I_load for loads.
        Returns dict node->voltage.
        """
        rs = self._reduced
        # Build RHS for unknown nodes
        I_u = np.zeros(len(rs.unknown_nodes), dtype=float)
        for n, cur in current_injections.items():
            if n in rs.index_unknown:
                I_u[rs.index_unknown[n]] += -float(cur)  # sink becomes negative injection
        # Adjust for pad voltages: RHS' = I_u - G_up * V_p
        V_p = np.full(len(rs.pad_nodes), rs.pad_voltage, dtype=float)
        rhs = I_u - rs.G_up @ V_p
        V_u = rs.lu(rhs)  # solve
        # Assemble full voltage dict
        voltages = {}
        for n in rs.pad_nodes:
            voltages[n] = rs.pad_voltage
        for i, n in enumerate(rs.unknown_nodes):
            voltages[n] = float(V_u[i])
        return voltages

    def solve_batch(self, currents_list: Sequence[Dict]) -> List[Dict]:
        """Solve multiple stimuli efficiently using shared factorization.

        currents_list: list of dict node->current (sink positive)
        Returns list of voltages dicts.
        """
        rs = self._reduced
        V_p = np.full(len(rs.pad_nodes), rs.pad_voltage, dtype=float)
        base = -rs.G_up @ V_p  # constant term
        solutions: List[Dict] = []
        for cur_map in currents_list:
            I_u = np.zeros(len(rs.unknown_nodes), dtype=float)
            for n, cur in cur_map.items():
                if n in rs.index_unknown:
                    I_u[rs.index_unknown[n]] += -float(cur)
            rhs = I_u + base
            V_u = rs.lu(rhs)
            voltages = {n: rs.pad_voltage for n in rs.pad_nodes}
            for i, n in enumerate(rs.unknown_nodes):
                voltages[n] = float(V_u[i])
            solutions.append(voltages)
        return solutions

    @staticmethod
    def ir_drop(voltages: Dict, vdd: float) -> Dict:
        """Return IR-drop per node: vdd - V_node."""
        return {n: vdd - v for n, v in voltages.items()}

    # =========================================================================
    # Hierarchical Solver Methods
    # =========================================================================

    def _decompose_at_layer(
        self, partition_layer: int
    ) -> Tuple[Set, Set, Set, List[Tuple]]:
        """Decompose the PDN at partition layer M_p into top-grid and bottom-grid.

        Given partition layer M_p:
          - top_grid: nodes in layers [M_p, M_p+1, ..., M_T] (includes partition layer)
          - bottom_grid: nodes in layers [M_0, M_1, ..., M_p-1]
          - ports: nodes in M_p that connect to M_p-1 via resistors (interface)
          - via_edges: edges connecting M_p to M_p-1 (the interface vias)

        Args:
            partition_layer: Layer index M_p to partition at (must be > 0)

        Returns:
            Tuple of (top_grid_nodes, bottom_grid_nodes, port_nodes, via_edges)

        Raises:
            ValueError: If partition_layer is invalid
        """
        # Collect all layers present in the graph
        all_layers = set()
        for node in self.G.nodes():
            layer = getattr(node, 'layer', self.G.nodes[node].get('layer', 0))
            all_layers.add(layer)

        max_layer = max(all_layers)
        min_layer = min(all_layers)

        if partition_layer <= min_layer:
            raise ValueError(
                f"partition_layer {partition_layer} must be > min layer {min_layer}"
            )
        if partition_layer > max_layer:
            raise ValueError(
                f"partition_layer {partition_layer} must be <= max layer {max_layer}"
            )

        # Categorize nodes by layer
        top_grid_nodes: Set = set()
        bottom_grid_nodes: Set = set()

        for node in self.G.nodes():
            layer = getattr(node, 'layer', self.G.nodes[node].get('layer', 0))
            if layer >= partition_layer:
                top_grid_nodes.add(node)
            else:
                bottom_grid_nodes.add(node)

        # Find via edges connecting M_p to M_p-1 and identify port nodes
        port_nodes: Set = set()
        via_edges: List[Tuple] = []

        for u, v, data in self.G.edges(data=True):
            u_layer = getattr(u, 'layer', self.G.nodes[u].get('layer', 0))
            v_layer = getattr(v, 'layer', self.G.nodes[v].get('layer', 0))

            # Check if this edge connects M_p and M_p-1
            if (u_layer == partition_layer and v_layer == partition_layer - 1):
                via_edges.append((u, v, data))
                port_nodes.add(u)  # u is in M_p
            elif (v_layer == partition_layer and u_layer == partition_layer - 1):
                via_edges.append((u, v, data))
                port_nodes.add(v)  # v is in M_p

        return top_grid_nodes, bottom_grid_nodes, port_nodes, via_edges

    def _compute_effective_resistance_in_subgrid(
        self,
        subgrid_nodes: Set,
        source_node,
        target_nodes: List,
        dirichlet_nodes: Set,
    ) -> Dict:
        """Compute effective resistance from source to each target within a subgrid.

        Uses the subgrid conductance matrix to compute R_eff. Dirichlet nodes
        (like ports) are treated as fixed voltage boundaries.

        Args:
            subgrid_nodes: Set of nodes forming the subgrid
            source_node: The source node for R_eff computation
            target_nodes: List of target nodes (typically ports)
            dirichlet_nodes: Nodes with fixed voltage (ports for bottom-grid)

        Returns:
            Dict mapping target_node -> R_eff from source to target
        """
        # Build subgraph and conductance matrix for the subgrid
        subgraph = self.G.subgraph(subgrid_nodes).copy()

        # Build conductance matrix for subgrid
        nodes = list(subgraph.nodes())
        if source_node not in nodes:
            return {t: float('inf') for t in target_nodes}

        index = {n: i for i, n in enumerate(nodes)}
        n_nodes = len(nodes)

        # Build sparse conductance matrix
        data, rows, cols = [], [], []
        diag = np.zeros(n_nodes, dtype=float)

        for u, v, d in subgraph.edges(data=True):
            R = float(d.get("resistance", 0.0))
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
            # Source is a Dirichlet node (port), R_eff = 0 to itself
            return {t: (0.0 if t == source_node else float('inf')) for t in target_nodes}

        if not unknown_nodes:
            return {t: float('inf') for t in target_nodes}

        index_unknown = {n: i for i, n in enumerate(unknown_nodes)}
        u_idx = [index[n] for n in unknown_nodes]
        G_uu = G_mat[np.ix_(u_idx, u_idx)].tocsr()

        # Factorize for solving
        try:
            lu = spla.factorized(G_uu)
        except Exception:
            return {t: float('inf') for t in target_nodes}

        # For R_eff(source, target), we need G_uu^(-1)[source, source]
        # and G_uu^(-1)[target, target] and G_uu^(-1)[source, target]
        # We solve G_uu * x = e_source to get column of G_uu^(-1)

        if source_node not in index_unknown:
            return {t: float('inf') for t in target_nodes}

        source_idx = index_unknown[source_node]
        e_source = np.zeros(len(unknown_nodes))
        e_source[source_idx] = 1.0
        x_source = lu(e_source)  # G_uu^(-1) * e_source

        results = {}
        for target in target_nodes:
            if target not in dirichlet_in_subgrid:
                # Target is also unknown - use standard R_eff formula
                if target in index_unknown:
                    target_idx = index_unknown[target]
                    e_target = np.zeros(len(unknown_nodes))
                    e_target[target_idx] = 1.0
                    x_target = lu(e_target)
                    # R_eff = x_source[source] + x_target[target] - x_source[target] - x_target[source]
                    r_eff = (x_source[source_idx] + x_target[target_idx]
                             - x_source[target_idx] - x_target[source_idx])
                    results[target] = max(0.0, r_eff)
                else:
                    results[target] = float('inf')
            else:
                # Target is a Dirichlet node (port)
                # R_eff to a Dirichlet node = G_uu^(-1)[source, source] for the reduced system
                # where the Dirichlet node acts as ground
                results[target] = max(0.0, x_source[source_idx])

        return results

    def _compute_shortest_path_resistance_in_subgrid(
        self,
        subgrid_nodes: Set,
        source_node,
        target_nodes: List,
    ) -> Dict:
        """Compute least resistive path from source to each target within a subgrid.

        Uses Dijkstra's algorithm with resistance as edge weights to find the
        minimum total resistance path from source to each target.

        Args:
            subgrid_nodes: Set of nodes forming the subgrid
            source_node: The source node for path computation
            target_nodes: List of target nodes (typically ports)

        Returns:
            Dict mapping target_node -> resistance of least resistive path
        """
        # Build subgraph
        subgraph = self.G.subgraph(subgrid_nodes)

        if source_node not in subgraph:
            return {t: float('inf') for t in target_nodes}

        # Use networkx's single_source_dijkstra with resistance as weight
        try:
            # Get shortest path lengths (by resistance) from source to all reachable nodes
            distances = nx.single_source_dijkstra_path_length(
                subgraph, source_node, weight='resistance'
            )
        except nx.NetworkXError:
            return {t: float('inf') for t in target_nodes}

        # Extract distances to target nodes
        results = {}
        for target in target_nodes:
            if target in distances:
                results[target] = distances[target]
            else:
                results[target] = float('inf')

        return results

    def _aggregate_currents_to_ports(
        self,
        current_injections: Dict,
        bottom_grid_nodes: Set,
        port_nodes: Set,
        top_k: int = 2,
        weighting: str = "effective",
    ) -> Tuple[Dict, Dict]:
        """Aggregate bottom-grid currents onto interface ports.

        Maps each bottom current source to the top-k nearest ports by resistance,
        weighted by inverse resistance. The resistance metric can be either
        effective resistance or shortest path resistance.

        Args:
            current_injections: Dict of node -> current (sink positive)
            bottom_grid_nodes: Set of nodes in bottom-grid
            port_nodes: Set of port nodes (interface between grids)
            top_k: Number of nearest ports to distribute current to (default 2)
            weighting: Resistance metric for weighting, one of:
                - "effective": Use effective electrical resistance (default)
                - "shortest_path": Use resistance of least resistive path

        Returns:
            Tuple of:
                - port_currents: Dict of port -> aggregated current
                - aggregation_map: Dict of load -> list of (port, weight, current_contrib)

        Raises:
            ValueError: If weighting is not a valid option
        """
        if weighting not in ("effective", "shortest_path"):
            raise ValueError(
                f"weighting must be 'effective' or 'shortest_path', got '{weighting}'"
            )

        port_currents: Dict = {p: 0.0 for p in port_nodes}
        aggregation_map: Dict = {}

        # Filter currents to only those in bottom-grid
        bottom_currents = {
            n: c for n, c in current_injections.items()
            if n in bottom_grid_nodes and n not in port_nodes
        }

        if not bottom_currents or not port_nodes:
            return port_currents, aggregation_map

        port_list = list(port_nodes)

        # Include ports in the subgrid for resistance computation
        subgrid = bottom_grid_nodes | port_nodes

        for load_node, current in bottom_currents.items():
            # Compute resistance from this load to each port
            if weighting == "effective":
                resistance_map = self._compute_effective_resistance_in_subgrid(
                    subgrid_nodes=subgrid,
                    source_node=load_node,
                    target_nodes=port_list,
                    dirichlet_nodes=port_nodes,
                )
            else:  # shortest_path
                resistance_map = self._compute_shortest_path_resistance_in_subgrid(
                    subgrid_nodes=subgrid,
                    source_node=load_node,
                    target_nodes=port_list,
                )

            # Sort ports by resistance and select top-k
            port_resistance_pairs = [(p, resistance_map.get(p, float('inf'))) for p in port_list]
            port_resistance_pairs.sort(key=lambda x: x[1])

            # Filter out infinite resistances and take top-k
            valid_pairs = [(p, r) for p, r in port_resistance_pairs if r < float('inf') and r > 0]

            if not valid_pairs:
                # Fallback: distribute uniformly to all ports
                weight = 1.0 / len(port_list)
                contrib = current * weight
                aggregation_map[load_node] = [(p, weight, contrib) for p in port_list]
                for p in port_list:
                    port_currents[p] += contrib
                continue

            # Take top-k
            selected = valid_pairs[:top_k]

            # Compute weights as inverse resistance (normalized)
            inv_resistances = [1.0 / r for _, r in selected]
            total_inv = sum(inv_resistances)

            contributions = []
            for (port, resistance), inv_r in zip(selected, inv_resistances):
                weight = inv_r / total_inv
                contrib = current * weight
                port_currents[port] += contrib
                contributions.append((port, weight, contrib))

            aggregation_map[load_node] = contributions

        return port_currents, aggregation_map

    def _build_subgrid_system(
        self,
        subgrid_nodes: Set,
        dirichlet_nodes: Set,
        dirichlet_voltage: float,
    ) -> Optional[ReducedSystem]:
        """Build a ReducedSystem for a subset of nodes with Dirichlet BCs.

        Creates a conductance matrix and factorization for the subgrid, treating
        dirichlet_nodes as fixed voltage boundaries.

        Args:
            subgrid_nodes: Set of nodes forming the subgrid
            dirichlet_nodes: Nodes with fixed voltage (e.g., pads or ports)
            dirichlet_voltage: Voltage applied at Dirichlet nodes

        Returns:
            ReducedSystem for the subgrid, or None if empty/invalid
        """
        if not subgrid_nodes:
            return None

        # Build subgraph
        subgraph = self.G.subgraph(subgrid_nodes)
        nodes = list(subgraph.nodes())

        if not nodes:
            return None

        index = {n: i for i, n in enumerate(nodes)}
        n_nodes = len(nodes)

        # Build sparse conductance matrix
        data, rows, cols = [], [], []
        diag = np.zeros(n_nodes, dtype=float)

        for u, v, d in subgraph.edges(data=True):
            R = float(d.get("resistance", 0.0))
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

        # Separate Dirichlet nodes from unknowns
        dirichlet_in_subgrid = list(dirichlet_nodes & set(nodes))
        unknown_nodes = [n for n in nodes if n not in dirichlet_nodes]

        if not unknown_nodes:
            # All nodes are Dirichlet - nothing to solve
            return None

        index_unknown = {n: i for i, n in enumerate(unknown_nodes)}
        u_idx = [index[n] for n in unknown_nodes]
        d_idx = [index[n] for n in dirichlet_in_subgrid]

        G_uu = G_mat[np.ix_(u_idx, u_idx)].tocsr()
        G_ud = G_mat[np.ix_(u_idx, d_idx)].tocsr() if d_idx else sp.csr_matrix((len(u_idx), 0))

        # Factorize
        try:
            lu = spla.factorized(G_uu)
        except Exception as e:
            raise ValueError(f"Failed to factorize subgrid system: {e}")

        return ReducedSystem(
            node_order=nodes,
            unknown_nodes=unknown_nodes,
            pad_nodes=dirichlet_in_subgrid,
            G_uu=G_uu,
            G_up=G_ud,
            lu=lu,
            pad_voltage=dirichlet_voltage,
            index_of=index,
            index_unknown=index_unknown,
        )

    def _solve_subgrid(
        self,
        reduced_system: ReducedSystem,
        current_injections: Dict,
        dirichlet_voltages: Optional[Dict] = None,
    ) -> Dict:
        """Solve nodal voltages for a subgrid system.

        Args:
            reduced_system: ReducedSystem for the subgrid
            current_injections: Dict of node -> current (sink positive)
            dirichlet_voltages: Optional dict of dirichlet_node -> voltage
                               If None, uses reduced_system.pad_voltage for all

        Returns:
            Dict of node -> voltage for all nodes in the subgrid
        """
        rs = reduced_system

        # Build RHS for unknown nodes
        I_u = np.zeros(len(rs.unknown_nodes), dtype=float)
        for n, cur in current_injections.items():
            if n in rs.index_unknown:
                I_u[rs.index_unknown[n]] += -float(cur)  # sink becomes negative injection

        # Build Dirichlet voltage vector
        if dirichlet_voltages is not None:
            V_d = np.array([dirichlet_voltages.get(n, rs.pad_voltage) for n in rs.pad_nodes])
        else:
            V_d = np.full(len(rs.pad_nodes), rs.pad_voltage, dtype=float)

        # Adjust RHS for Dirichlet conditions: RHS' = I_u - G_ud * V_d
        if rs.G_up.shape[1] > 0:
            rhs = I_u - rs.G_up @ V_d
        else:
            rhs = I_u

        # Solve
        V_u = rs.lu(rhs)

        # Assemble full voltage dict
        voltages = {}
        for i, n in enumerate(rs.pad_nodes):
            voltages[n] = V_d[i] if dirichlet_voltages else rs.pad_voltage
        for i, n in enumerate(rs.unknown_nodes):
            voltages[n] = float(V_u[i])

        return voltages

    def solve_hierarchical(
        self,
        current_injections: Dict,
        partition_layer: int,
        top_k: int = 2,
        weighting: str = "effective",
    ) -> HierarchicalSolveResult:
        """Solve IR-drop using hierarchical decomposition at partition layer.

        Decomposes the PDN into top-grid (layers M_p to M_T) and bottom-grid
        (layers M_0 to M_p-1), solves them sequentially:

        1. Aggregate bottom-grid currents onto interface ports using top-k
           nearest ports weighted by inverse resistance
        2. Solve top-grid with pad voltages as Dirichlet BC and port injections
        3. Solve bottom-grid with port voltages (from step 2) as Dirichlet BC

        Args:
            current_injections: Dict of node -> current (sink positive, Amps)
            partition_layer: Layer index M_p to partition at (must be > 0)
            top_k: Number of nearest ports for current aggregation (default 2)
            weighting: Resistance metric for port weighting, one of:
                - "effective": Use effective electrical resistance (default)
                - "shortest_path": Use resistance of least resistive path

        Returns:
            HierarchicalSolveResult with complete and sub-grid voltages

        Raises:
            ValueError: If partition_layer is invalid or decomposition fails
        """
        # Step 0: Decompose the PDN
        top_grid_nodes, bottom_grid_nodes, port_nodes, via_edges = \
            self._decompose_at_layer(partition_layer)

        if not port_nodes:
            raise ValueError(
                f"No ports found at partition layer {partition_layer}. "
                "Check that vias connect this layer to the layer below."
            )

        # Step 1: Aggregate bottom-grid currents onto ports
        port_currents, aggregation_map = self._aggregate_currents_to_ports(
            current_injections=current_injections,
            bottom_grid_nodes=bottom_grid_nodes,
            port_nodes=port_nodes,
            top_k=top_k,
            weighting=weighting,
        )

        # Step 2: Solve top-grid with pads as Dirichlet BC
        # Top-grid includes: top_grid_nodes (layers M_p to M_T)
        # Dirichlet nodes: pads (on top layer)
        pad_set = set(self.pad_nodes)
        top_grid_pads = pad_set & top_grid_nodes

        if not top_grid_pads:
            raise ValueError(
                f"No pad nodes found in top-grid (layers >= {partition_layer}). "
                "Pads should be on the top-most layer."
            )

        # Build and solve top-grid system
        top_system = self._build_subgrid_system(
            subgrid_nodes=top_grid_nodes,
            dirichlet_nodes=top_grid_pads,
            dirichlet_voltage=self.vdd,
        )

        if top_system is None:
            raise ValueError("Failed to build top-grid system")

        # Current injections for top-grid: only port currents (aggregated from bottom)
        # Note: Any loads that happen to be in top-grid are handled directly
        top_grid_currents = {n: c for n, c in current_injections.items() if n in top_grid_nodes}
        # Add aggregated port currents
        for port, curr in port_currents.items():
            top_grid_currents[port] = top_grid_currents.get(port, 0.0) + curr

        top_grid_voltages = self._solve_subgrid(
            reduced_system=top_system,
            current_injections=top_grid_currents,
            dirichlet_voltages=None,  # Use vdd for pads
        )

        # Extract port voltages for bottom-grid BC
        port_voltages = {p: top_grid_voltages[p] for p in port_nodes}

        # Step 3: Solve bottom-grid with port voltages as Dirichlet BC
        # Bottom-grid includes: bottom_grid_nodes (layers M_0 to M_p-1)
        # We need to include the ports as Dirichlet nodes but they're in top-grid
        # Solution: Include ports in the bottom subgrid for the solve
        bottom_subgrid = bottom_grid_nodes | port_nodes

        bottom_system = self._build_subgrid_system(
            subgrid_nodes=bottom_subgrid,
            dirichlet_nodes=port_nodes,
            dirichlet_voltage=self.vdd,  # Will be overridden by dirichlet_voltages
        )

        if bottom_system is None:
            raise ValueError("Failed to build bottom-grid system")

        # Current injections for bottom-grid: original currents in bottom-grid
        bottom_grid_currents = {
            n: c for n, c in current_injections.items()
            if n in bottom_grid_nodes
        }

        bottom_grid_voltages = self._solve_subgrid(
            reduced_system=bottom_system,
            current_injections=bottom_grid_currents,
            dirichlet_voltages=port_voltages,
        )

        # Merge voltages: top-grid + bottom-grid (excluding duplicate ports)
        all_voltages = {}
        all_voltages.update(top_grid_voltages)
        # Add bottom-grid nodes (excluding ports which are already in top_grid_voltages)
        for n, v in bottom_grid_voltages.items():
            if n not in port_nodes:
                all_voltages[n] = v

        # Compute IR-drop
        ir_drop = self.ir_drop(all_voltages, self.vdd)

        return HierarchicalSolveResult(
            voltages=all_voltages,
            ir_drop=ir_drop,
            partition_layer=partition_layer,
            top_grid_voltages=top_grid_voltages,
            bottom_grid_voltages=bottom_grid_voltages,
            port_nodes=port_nodes,
            port_voltages=port_voltages,
            port_currents=port_currents,
            aggregation_map=aggregation_map,
        )

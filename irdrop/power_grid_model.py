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

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import networkx as nx


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

"""Block-partitioned matrices and core Schur complement data structures.

Extracted from solver/coupled_system.py.  Contains:
  - BlockMatrixSystem dataclass
  - build_block_system_from_edges
  - extract_block_matrices
  - compute_reduced_rhs
  - recover_bottom_voltages
  - build_grounded_capacitance_diags
  - Regularization resistance setters/getters
  - Memory-formatting helpers

RULE: this module imports ONLY numpy / scipy / stdlib / optional sksparse and
      other pgmath sub-modules.  It must never import from solver/, distributed/,
      analysis/, or model/.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import scipy.sparse as sp

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Partial Cholesky regularization / chunked path defaults
# ---------------------------------------------------------------------------
_PARTIAL_FACTOR_REG_OHMS: float = 1e8  # regularization resistance in Ohms
_CHUNKED_MAX_MEMORY_GB: float = 4.0    # memory budget for Z_chunk in chunked path


def set_partial_factor_reg_resistance(value: float) -> None:
    """Set the regularization resistance (Ohms) for the partial Cholesky path.

    A small grounding conductance ``1/R`` (in mS) is added to port diagonals
    to ensure the full tile matrix is SPD.  Subtracted from S after extraction.

    Args:
        value: Resistance in Ohms (default 1e8 = 100 MΩ).
    """
    if value <= 0:
        raise ValueError(
            f"Regularization resistance must be positive, got {value}"
        )
    global _PARTIAL_FACTOR_REG_OHMS
    _PARTIAL_FACTOR_REG_OHMS = value


def get_partial_factor_reg_resistance() -> float:
    """Return the current regularization resistance in Ohms."""
    return _PARTIAL_FACTOR_REG_OHMS


def _sparse_mem_bytes(M: sp.spmatrix) -> int:
    """Memory footprint of a scipy sparse CSR/CSC matrix."""
    return M.data.nbytes + M.indices.nbytes + M.indptr.nbytes


def _format_bytes(n: int) -> str:
    """Human-readable byte size."""
    for unit in ('B', 'KB', 'MB', 'GB'):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


# ---------------------------------------------------------------------------
# BlockMatrixSystem
# ---------------------------------------------------------------------------

@dataclass
class BlockMatrixSystem:
    """Block-partitioned conductance matrix for a grid region.

    Holds the block matrices from partitioning nodes into:
    - port nodes (boundary nodes at partition layer)
    - interior nodes (non-boundary nodes)

    The full conductance matrix has the structure:
        G = [[G_pp, G_pi],
             [G_ip, G_ii]]

    where p=port, i=interior.
    """

    G_pp: sp.csr_matrix
    G_pi: sp.csr_matrix
    G_ip: sp.csr_matrix
    G_ii: sp.csr_matrix
    port_nodes: List[Any]
    interior_nodes: List[Any]
    port_to_idx: Dict[Any, int]
    interior_to_idx: Dict[Any, int]
    lu_ii: Optional[Callable[[np.ndarray], np.ndarray]] = None
    factor_adapter: Optional[Any] = field(default=None, repr=False)

    @property
    def n_ports(self) -> int:
        """Number of port nodes."""
        return len(self.port_nodes)

    @property
    def n_interior(self) -> int:
        """Number of interior nodes."""
        return len(self.interior_nodes)

    def factor_interior(self, verbose: bool = False) -> None:
        """Pre-compute factorization of G_ii for fast solves.

        Uses cholmod if available, otherwise falls back to splu.
        Stores the full SparseFactorAdapter in ``self.factor_adapter``
        and the solve callable in ``self.lu_ii`` for backward compatibility.
        """
        if self.n_interior > 0:
            from pgmath.factor import _factor_conductance_matrix
            factor = _factor_conductance_matrix(self.G_ii, verbose=verbose)
            self.factor_adapter = factor
            self.lu_ii = factor.solve
        else:
            self.lu_ii = lambda x: np.array([])

    def memory_bytes(self) -> int:
        """Total memory footprint of the four block matrices (CSR storage)."""
        total = 0
        for M in (self.G_pp, self.G_pi, self.G_ip, self.G_ii):
            if sp.issparse(M):
                total += _sparse_mem_bytes(M)
        return total

    def stats(self) -> Dict[str, Any]:
        """Summary statistics for this block system."""
        return {
            'n_ports': self.n_ports,
            'n_interior': self.n_interior,
            'G_ii_nnz': self.G_ii.nnz if sp.issparse(self.G_ii) else 0,
            'G_pp_nnz': self.G_pp.nnz if sp.issparse(self.G_pp) else 0,
            'G_pi_nnz': self.G_pi.nnz if sp.issparse(self.G_pi) else 0,
            'G_ip_nnz': self.G_ip.nnz if sp.issparse(self.G_ip) else 0,
            'mem_bytes': self.memory_bytes(),
        }

    def solve_interior(self, b: np.ndarray) -> np.ndarray:
        """Solve G_ii * x = b using cached LU factorization."""
        if self.lu_ii is None:
            raise ValueError(
                "LU factorization not computed. Call factor_interior() first."
            )
        return self.lu_ii(b)


# ---------------------------------------------------------------------------
# extract_block_matrices (model-based builder, used by flat coupled solver)
# ---------------------------------------------------------------------------

def extract_block_matrices(
    model: Any,
    grid_nodes: Set[Any],
    dirichlet_nodes: Set[Any],
    port_nodes: Set[Any],
    dirichlet_voltage: float,
    exclude_port_to_port: bool = False,
) -> Tuple['BlockMatrixSystem', np.ndarray]:
    """Extract block matrices from a grid region of a UnifiedPowerGridModel.

    Builds the conductance matrix for the subgrid and partitions it into
    blocks based on port nodes vs interior nodes.  Dirichlet nodes (pads)
    are eliminated via their contribution to the RHS.

    Args:
        model: UnifiedPowerGridModel containing the graph.
        grid_nodes: Set of all nodes in this grid region.
        dirichlet_nodes: Set of nodes with fixed voltage (e.g., pads).
        port_nodes: Set of port nodes (boundary for coupled solve).
        dirichlet_voltage: Voltage applied at Dirichlet nodes.
        exclude_port_to_port: If True, exclude edges where both endpoints
            are ports (use for bottom-grid extraction).

    Returns:
        Tuple of (BlockMatrixSystem, rhs_dirichlet).
    """
    if not grid_nodes:
        raise ValueError("grid_nodes is empty")

    port_set = port_nodes & grid_nodes
    dirichlet_set = dirichlet_nodes & grid_nodes
    interior_nodes = grid_nodes - port_set - dirichlet_set

    port_list = sorted(port_set, key=str)
    interior_list = sorted(interior_nodes, key=str)
    dirichlet_list = sorted(dirichlet_set, key=str)

    all_nodes = port_list + interior_list + dirichlet_list
    node_to_idx = {n: i for i, n in enumerate(all_nodes)}

    n_ports = len(port_list)
    n_interior = len(interior_list)
    n_dirichlet = len(dirichlet_list)
    n_unknown = n_ports + n_interior

    if n_unknown == 0:
        raise ValueError("No unknown nodes (all nodes are Dirichlet)")

    grid_node_set = set(all_nodes)
    data, rows, cols = [], [], []
    diag = np.zeros(len(all_nodes), dtype=np.float64)

    GMAX = 1e5
    SHORT_THRESHOLD = 1e-6

    for u, v, edge_info in model._iter_resistive_edges():
        if u not in grid_node_set or v not in grid_node_set:
            continue
        if exclude_port_to_port and u in port_set and v in port_set:
            continue

        R = edge_info.resistance
        if R is None:
            continue

        g = GMAX if (R <= 0 or R < SHORT_THRESHOLD) else 1.0 / R

        iu, iv = node_to_idx[u], node_to_idx[v]
        rows.extend([iu, iv])
        cols.extend([iv, iu])
        data.extend([-g, -g])
        diag[iu] += g
        diag[iv] += g

    for i in range(len(all_nodes)):
        rows.append(i)
        cols.append(i)
        data.append(diag[i])

    G_full = sp.csr_matrix(
        (data, (rows, cols)), shape=(len(all_nodes), len(all_nodes))
    )

    p_idx = np.arange(n_ports)
    i_idx = np.arange(n_ports, n_ports + n_interior)
    d_idx = np.arange(n_unknown, len(all_nodes))
    u_idx = np.arange(n_unknown)

    G_pp = G_full[np.ix_(p_idx, p_idx)].tocsr() if n_ports > 0 else sp.csr_matrix((0, 0))
    G_pi = G_full[np.ix_(p_idx, i_idx)].tocsr() if n_ports > 0 and n_interior > 0 else sp.csr_matrix((n_ports, 0))
    G_ip = G_full[np.ix_(i_idx, p_idx)].tocsr() if n_interior > 0 and n_ports > 0 else sp.csr_matrix((0, n_ports))
    G_ii = G_full[np.ix_(i_idx, i_idx)].tocsr() if n_interior > 0 else sp.csr_matrix((0, 0))

    if n_dirichlet > 0:
        G_ud = G_full[np.ix_(u_idx, d_idx)].tocsr()
        V_d = np.full(n_dirichlet, dirichlet_voltage, dtype=np.float64)
        rhs_dirichlet = -(G_ud @ V_d)
    else:
        rhs_dirichlet = np.zeros(n_unknown, dtype=np.float64)

    port_to_idx = {n: i for i, n in enumerate(port_list)}
    interior_to_idx = {n: i for i, n in enumerate(interior_list)}

    block_system = BlockMatrixSystem(
        G_pp=G_pp,
        G_pi=G_pi,
        G_ip=G_ip,
        G_ii=G_ii,
        port_nodes=port_list,
        interior_nodes=interior_list,
        port_to_idx=port_to_idx,
        interior_to_idx=interior_to_idx,
        lu_ii=None,
    )

    return block_system, rhs_dirichlet


# ---------------------------------------------------------------------------
# build_block_system_from_edges (edge-based builder, used by distributed tile workers)
# ---------------------------------------------------------------------------

def build_block_system_from_edges(
    edges: Any,
    port_nodes: Set[str],
    dirichlet_nodes: Optional[Set[str]] = None,
    dirichlet_voltage: float = 0.0,
    ground_node: str = '0',
) -> Tuple['BlockMatrixSystem', np.ndarray]:
    """Build BlockMatrixSystem from raw (u, v, conductance) edge data.

    Generalized variant of extract_block_matrices() that accepts edges directly
    instead of requiring a UnifiedPowerGridModel.  Same constants (GMAX,
    SHORT_THRESHOLD), same ground handling (diagonal-only), same Dirichlet
    elimination (-G_ud @ V_d), same port/interior ordering.

    All non-port, non-Dirichlet, non-ground nodes are classified as interior.

    Args:
        edges: Iterable of (u, v, conductance) tuples. Conductance is g = 1/R.
        port_nodes: Set of boundary/port node names.
        dirichlet_nodes: Set of nodes with fixed voltage. Defaults to empty.
        dirichlet_voltage: Voltage applied at Dirichlet nodes (default 0.0).
        ground_node: Name of ground node (default '0').

    Returns:
        Tuple of (BlockMatrixSystem, rhs_dirichlet).
    """
    if dirichlet_nodes is None:
        dirichlet_nodes = set()

    GMAX = 1e5
    SHORT_THRESHOLD = 1e-6

    all_nodes_set: Set[str] = set()
    edge_list = []
    for u, v, g in edges:
        all_nodes_set.add(u)
        all_nodes_set.add(v)
        edge_list.append((u, v, g))

    all_nodes_set.discard(ground_node)

    port_set = port_nodes & all_nodes_set
    dirichlet_set = dirichlet_nodes & all_nodes_set
    interior_nodes = all_nodes_set - port_set - dirichlet_set

    port_list = sorted(port_set)
    interior_list = sorted(interior_nodes)
    dirichlet_list = sorted(dirichlet_set)

    all_nodes = port_list + interior_list + dirichlet_list
    node_to_idx = {n: i for i, n in enumerate(all_nodes)}

    n_ports = len(port_list)
    n_interior = len(interior_list)
    n_dirichlet = len(dirichlet_list)
    n_unknown = n_ports + n_interior

    if n_unknown == 0:
        block_system = BlockMatrixSystem(
            G_pp=sp.csr_matrix((0, 0)),
            G_pi=sp.csr_matrix((0, 0)),
            G_ip=sp.csr_matrix((0, 0)),
            G_ii=sp.csr_matrix((0, 0)),
            port_nodes=[],
            interior_nodes=[],
            port_to_idx={},
            interior_to_idx={},
            lu_ii=None,
        )
        return block_system, np.zeros(0, dtype=np.float64)

    data, rows, cols = [], [], []
    diag = np.zeros(len(all_nodes), dtype=np.float64)

    for u, v, g in edge_list:
        if g <= 0 or g > GMAX:
            g = min(max(g, 1.0 / GMAX), GMAX)
        if 1.0 / g < SHORT_THRESHOLD:
            g = GMAX

        if u == ground_node:
            if v in node_to_idx:
                diag[node_to_idx[v]] += g
            continue
        if v == ground_node:
            if u in node_to_idx:
                diag[node_to_idx[u]] += g
            continue

        if u not in node_to_idx or v not in node_to_idx:
            continue

        iu, iv = node_to_idx[u], node_to_idx[v]
        rows.extend([iu, iv])
        cols.extend([iv, iu])
        data.extend([-g, -g])
        diag[iu] += g
        diag[iv] += g

    for i in range(len(all_nodes)):
        rows.append(i)
        cols.append(i)
        data.append(diag[i])

    n_total = len(all_nodes)
    G_full = sp.csr_matrix((data, (rows, cols)), shape=(n_total, n_total))

    p_idx = np.arange(n_ports)
    i_idx = np.arange(n_ports, n_ports + n_interior)
    d_idx = np.arange(n_unknown, n_total)
    u_idx = np.arange(n_unknown)

    G_pp = G_full[np.ix_(p_idx, p_idx)].tocsr() if n_ports > 0 else sp.csr_matrix((0, 0))
    G_pi = G_full[np.ix_(p_idx, i_idx)].tocsr() if n_ports > 0 and n_interior > 0 else sp.csr_matrix((n_ports, 0))
    G_ip = G_full[np.ix_(i_idx, p_idx)].tocsr() if n_interior > 0 and n_ports > 0 else sp.csr_matrix((0, n_ports))
    G_ii = G_full[np.ix_(i_idx, i_idx)].tocsr() if n_interior > 0 else sp.csr_matrix((0, 0))

    if n_dirichlet > 0:
        G_ud = G_full[np.ix_(u_idx, d_idx)].tocsr()
        V_d = np.full(n_dirichlet, dirichlet_voltage, dtype=np.float64)
        rhs_dirichlet = -(G_ud @ V_d)
    else:
        rhs_dirichlet = np.zeros(n_unknown, dtype=np.float64)

    port_to_idx_map = {n: i for i, n in enumerate(port_list)}
    interior_to_idx_map = {n: i for i, n in enumerate(interior_list)}

    block_system = BlockMatrixSystem(
        G_pp=G_pp,
        G_pi=G_pi,
        G_ip=G_ip,
        G_ii=G_ii,
        port_nodes=port_list,
        interior_nodes=interior_list,
        port_to_idx=port_to_idx_map,
        interior_to_idx=interior_to_idx_map,
        lu_ii=None,
    )

    return block_system, rhs_dirichlet


# ---------------------------------------------------------------------------
# compute_reduced_rhs
# ---------------------------------------------------------------------------

def compute_reduced_rhs(
    bottom_blocks: 'BlockMatrixSystem',
    current_injections: Dict[Any, float],
    rhs_dirichlet_bottom: np.ndarray,
) -> np.ndarray:
    """Compute reduced RHS r^B for the coupled system.

    The reduced RHS at ports is:
        r^B = (i_p + rhs_dirichlet_p) - G^B_pi * inv(G^B_ii) * (i_i + rhs_dirichlet_i)
    """
    n_ports = bottom_blocks.n_ports
    n_interior = bottom_blocks.n_interior

    i_p = np.zeros(n_ports, dtype=np.float64)
    i_i = np.zeros(n_interior, dtype=np.float64)

    for node, current in current_injections.items():
        if node in bottom_blocks.port_to_idx:
            i_p[bottom_blocks.port_to_idx[node]] -= current
        elif node in bottom_blocks.interior_to_idx:
            i_i[bottom_blocks.interior_to_idx[node]] -= current

    rhs_p = i_p + rhs_dirichlet_bottom[:n_ports]
    rhs_i = i_i + rhs_dirichlet_bottom[n_ports:n_ports + n_interior]

    if n_interior > 0 and bottom_blocks.lu_ii is not None:
        v_i = bottom_blocks.lu_ii(rhs_i)
        r_B = rhs_p - bottom_blocks.G_pi @ v_i
    else:
        r_B = rhs_p

    return r_B


# ---------------------------------------------------------------------------
# recover_bottom_voltages
# ---------------------------------------------------------------------------

def recover_bottom_voltages(
    bottom_blocks: 'BlockMatrixSystem',
    port_voltages: np.ndarray,
    current_injections: Dict[Any, float],
    rhs_dirichlet_bottom: np.ndarray,
) -> Dict[Any, float]:
    """Recover bottom-grid interior voltages from port voltages.

    Once port voltages are known, interior voltages are recovered via:
        v_i = inv(G^B_ii) * (rhs_i - G^B_ip * v_p)
    """
    n_ports = bottom_blocks.n_ports
    n_interior = bottom_blocks.n_interior

    voltages: Dict[Any, float] = {}

    if n_interior == 0 or bottom_blocks.lu_ii is None:
        return voltages

    i_i = np.zeros(n_interior, dtype=np.float64)
    for node, current in current_injections.items():
        if node in bottom_blocks.interior_to_idx:
            i_i[bottom_blocks.interior_to_idx[node]] -= current

    rhs_i = i_i + rhs_dirichlet_bottom[n_ports:n_ports + n_interior]
    coupling = bottom_blocks.G_ip @ port_voltages
    v_i = bottom_blocks.lu_ii(rhs_i - coupling)

    for i, node in enumerate(bottom_blocks.interior_nodes):
        voltages[node] = float(v_i[i])

    return voltages


# ---------------------------------------------------------------------------
# build_grounded_capacitance_diags
# ---------------------------------------------------------------------------

def build_grounded_capacitance_diags(
    cap_edges: List[Tuple[str, str, float]],
    port_to_idx: Dict[str, int],
    interior_to_idx: Dict[str, int],
    n_ports: int,
    n_interior: int,
    ground_node: str = '0',
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Build diagonal C vectors for grounded-only capacitors.

    Since all tile/instance caps are grounded (one terminal is '0'),
    the C matrix is purely diagonal -- no off-diagonal coupling.

    Args:
        cap_edges: List of (u, v, capacitance_fF) tuples, one terminal = ground.
        port_to_idx: Dict mapping port node name -> index.
        interior_to_idx: Dict mapping interior node name -> index.
        n_ports: Length of the port vector.
        n_interior: Length of the interior vector.
        ground_node: Name of the ground node (default '0').

    Returns:
        Tuple of (c_pp_diag, c_ii_diag, total_capacitance_fF).
    """
    c_pp_diag = np.zeros(n_ports, dtype=np.float64)
    c_ii_diag = np.zeros(n_interior, dtype=np.float64)
    total_cap = 0.0

    for u, v, c_fF in cap_edges:
        if c_fF <= 0:
            continue

        if u == ground_node and v == ground_node:
            continue
        if u == ground_node:
            node = v
        elif v == ground_node:
            node = u
        else:
            continue  # non-grounded cap; skip

        idx = port_to_idx.get(node)
        if idx is not None:
            c_pp_diag[idx] += c_fF
            total_cap += c_fF
            continue

        idx = interior_to_idx.get(node)
        if idx is not None:
            c_ii_diag[idx] += c_fF
            total_cap += c_fF

    return c_pp_diag, c_ii_diag, total_cap

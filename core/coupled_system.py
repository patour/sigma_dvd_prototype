"""Block-partitioned matrices and operators for coupled hierarchical solving.

This module provides classes for matrix-free Schur complement operations
used by the coupled hierarchical solver. The key insight is that we can
solve the coupled top-grid + bottom-grid system iteratively without
explicitly forming the Schur complement matrix.

Mathematical Formulation:
------------------------
Given a block-partitioned conductance matrix where:
- t: top-grid interior nodes (above partition layer, excluding ports)
- b: bottom-grid interior nodes (below partition layer)
- p: port nodes (at partition layer)

Top-grid (after eliminating pads):
    G^T = [[G^T_pp, G^T_pt],
           [G^T_tp, G^T_tt]]

Bottom-grid:
    G^B = [[G^B_pp, G^B_pb],
           [G^B_bp, G^B_bb]]

The Schur complement of the bottom-grid interior onto ports is:
    S^B = G^B_pp - G^B_pb * inv(G^B_bb) * G^B_bp

The coupled system at ports becomes:
    (G^T_pp + S^B) * v_p + G^T_pt * v_t = r^B + g_p
    G^T_tp * v_p + G^T_tt * v_t = b_t

where r^B = i_p - G^B_pb * inv(G^B_bb) * i_b is the reduced RHS from bottom-grid.

Key Classes:
- BlockMatrixSystem: Holds block matrices and LU factorization
- SchurComplementOperator: Matrix-free S^B * x operation
- CoupledSystemOperator: Full coupled system A * x operation
- BlockDiagonalPreconditioner: Efficient block diagonal approximation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .unified_model import UnifiedPowerGridModel


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

    Attributes:
        G_pp: Sparse port-port conductance (n_ports x n_ports)
        G_pi: Sparse port-interior conductance (n_ports x n_interior)
        G_ip: Sparse interior-port conductance (n_interior x n_ports)
        G_ii: Sparse interior-interior conductance (n_interior x n_interior)
        port_nodes: Ordered list of port nodes
        interior_nodes: Ordered list of interior nodes
        port_to_idx: Dict mapping port node -> index in port_nodes
        interior_to_idx: Dict mapping interior node -> index in interior_nodes
        lu_ii: LU factorization of G_ii (callable), or None if not factored
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
        
        Args:
            verbose: If True, print which backend is being used
        """
        if self.n_interior > 0:
            from .unified_solver import _factor_conductance_matrix
            factor = _factor_conductance_matrix(self.G_ii, verbose=verbose)
            self.lu_ii = factor.solve
        else:
            self.lu_ii = lambda x: np.array([])

    def solve_interior(self, b: np.ndarray) -> np.ndarray:
        """Solve G_ii * x = b using cached LU factorization.

        Args:
            b: Right-hand side vector of shape (n_interior,)

        Returns:
            Solution x of shape (n_interior,)

        Raises:
            ValueError: If LU factorization not computed (call factor_interior first)
        """
        if self.lu_ii is None:
            raise ValueError("LU factorization not computed. Call factor_interior() first.")
        return self.lu_ii(b)


def extract_block_matrices(
    model: UnifiedPowerGridModel,
    grid_nodes: Set[Any],
    dirichlet_nodes: Set[Any],
    port_nodes: Set[Any],
    dirichlet_voltage: float,
    exclude_port_to_port: bool = False,
) -> Tuple[BlockMatrixSystem, np.ndarray]:
    """Extract block matrices from a grid region.

    Builds the conductance matrix for the subgrid and partitions it into
    blocks based on port nodes vs interior nodes. Dirichlet nodes (pads)
    are eliminated via their contribution to the RHS.

    The returned system has structure:
        [[G_pp, G_pi],   [v_p]   [rhs_p]
         [G_ip, G_ii]] * [v_i] = [rhs_i]

    where the RHS incorporates contributions from Dirichlet boundary conditions.

    Args:
        model: UnifiedPowerGridModel containing the graph
        grid_nodes: Set of all nodes in this grid region (including ports and Dirichlet)
        dirichlet_nodes: Set of nodes with fixed voltage (e.g., pads)
        port_nodes: Set of port nodes (boundary for coupled solve)
        dirichlet_voltage: Voltage applied at Dirichlet nodes
        exclude_port_to_port: If True, exclude edges where both endpoints are ports.
            Use this for bottom-grid extraction to avoid double-counting lateral
            port connections that are already in the top-grid.

    Returns:
        Tuple of:
        - BlockMatrixSystem with extracted block matrices
        - rhs_dirichlet: Array of shape (n_ports + n_interior,) containing
          the contribution from Dirichlet nodes (-G_ud * V_d) in the same
          ordering as [port_nodes, interior_nodes]

    Raises:
        ValueError: If grid_nodes is empty or contains no edges
    """
    if not grid_nodes:
        raise ValueError("grid_nodes is empty")

    # Ensure port_nodes are within grid_nodes
    port_set = port_nodes & grid_nodes
    dirichlet_set = dirichlet_nodes & grid_nodes

    # Interior nodes: in grid but not port and not Dirichlet
    interior_nodes = grid_nodes - port_set - dirichlet_set

    # Order nodes: ports first, then interior, then Dirichlet
    port_list = sorted(port_set, key=str)
    interior_list = sorted(interior_nodes, key=str)
    dirichlet_list = sorted(dirichlet_set, key=str)

    all_nodes = port_list + interior_list + dirichlet_list
    node_to_idx = {n: i for i, n in enumerate(all_nodes)}

    n_ports = len(port_list)
    n_interior = len(interior_list)
    n_dirichlet = len(dirichlet_list)
    n_unknown = n_ports + n_interior  # unknowns = ports + interior (not Dirichlet)

    if n_unknown == 0:
        raise ValueError("No unknown nodes (all nodes are Dirichlet)")

    # Build conductance matrix for subgrid
    grid_node_set = set(all_nodes)
    data, rows, cols = [], [], []
    diag = np.zeros(len(all_nodes), dtype=np.float64)

    GMAX = 1e5
    SHORT_THRESHOLD = 1e-6

    for u, v, edge_info in model._iter_resistive_edges():
        if u not in grid_node_set or v not in grid_node_set:
            continue

        # Optionally exclude port-to-port edges
        if exclude_port_to_port and u in port_set and v in port_set:
            continue

        R = edge_info.resistance
        if R is None:
            continue

        # Handle zero/short resistance like the flat solver
        if R <= 0 or R < SHORT_THRESHOLD:
            g = GMAX
        else:
            g = 1.0 / R

        iu, iv = node_to_idx[u], node_to_idx[v]

        rows.extend([iu, iv])
        cols.extend([iv, iu])
        data.extend([-g, -g])

        diag[iu] += g
        diag[iv] += g

    # Add diagonal entries
    for i in range(len(all_nodes)):
        rows.append(i)
        cols.append(i)
        data.append(diag[i])

    G_full = sp.csr_matrix(
        (data, (rows, cols)), shape=(len(all_nodes), len(all_nodes))
    )

    # Extract block submatrices
    # Indexing: [0:n_ports] = ports, [n_ports:n_ports+n_interior] = interior,
    #           [n_unknown:] = Dirichlet
    p_idx = np.arange(n_ports)
    i_idx = np.arange(n_ports, n_ports + n_interior)
    d_idx = np.arange(n_unknown, len(all_nodes))

    # Unknown blocks (ports + interior)
    u_idx = np.arange(n_unknown)

    G_pp = G_full[np.ix_(p_idx, p_idx)].tocsr() if n_ports > 0 else sp.csr_matrix((0, 0))
    G_pi = G_full[np.ix_(p_idx, i_idx)].tocsr() if n_ports > 0 and n_interior > 0 else sp.csr_matrix((n_ports, 0))
    G_ip = G_full[np.ix_(i_idx, p_idx)].tocsr() if n_interior > 0 and n_ports > 0 else sp.csr_matrix((0, n_ports))
    G_ii = G_full[np.ix_(i_idx, i_idx)].tocsr() if n_interior > 0 else sp.csr_matrix((0, 0))

    # Compute RHS contribution from Dirichlet nodes
    # rhs_dirichlet = -G_ud * V_d where G_ud is unknown-to-Dirichlet coupling
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


class SchurComplementOperator(spla.LinearOperator):
    """Matrix-free operator for Schur complement S = G_pp - G_pi * inv(G_ii) * G_ip.

    This operator computes S * x without explicitly forming the dense Schur
    complement matrix. Each matvec requires one LU solve with G_ii.

    The Schur complement represents the effective conductance at port nodes
    after eliminating interior nodes.

    Args:
        G_pp: Port-port conductance matrix
        G_pi: Port-interior conductance matrix
        G_ip: Interior-port conductance matrix
        lu_ii: LU factorization callable for G_ii

    Example:
        >>> blocks = BlockMatrixSystem(...)
        >>> blocks.factor_interior()
        >>> S = SchurComplementOperator(blocks.G_pp, blocks.G_pi, blocks.G_ip, blocks.lu_ii)
        >>> y = S @ x  # Computes Schur complement application
    """

    def __init__(
        self,
        G_pp: sp.csr_matrix,
        G_pi: sp.csr_matrix,
        G_ip: sp.csr_matrix,
        lu_ii: Callable[[np.ndarray], np.ndarray],
    ):
        self.G_pp = G_pp
        self.G_pi = G_pi
        self.G_ip = G_ip
        self.lu_ii = lu_ii

        n_ports = G_pp.shape[0]
        super().__init__(dtype=np.float64, shape=(n_ports, n_ports))

    def _matvec(self, x: np.ndarray) -> np.ndarray:
        """Compute S * x = G_pp * x - G_pi * inv(G_ii) * G_ip * x."""
        # y1 = G_ip * x
        y1 = self.G_ip @ x

        # y2 = inv(G_ii) * y1
        if len(y1) > 0:
            y2 = self.lu_ii(y1)
        else:
            y2 = np.array([])

        # y3 = G_pi * y2
        y3 = self.G_pi @ y2

        # result = G_pp * x - y3
        return self.G_pp @ x - y3

    def _rmatvec(self, x: np.ndarray) -> np.ndarray:
        """Compute S^T * x (for symmetric matrices, same as matvec)."""
        # For conductance matrices, G is symmetric, so S is also symmetric
        return self._matvec(x)


class CoupledSystemOperator(spla.LinearOperator):
    """Matrix-free operator for the coupled top-grid + Schur complement system.

    The coupled system has the form:
        A = [[G^T_pp + S^B, G^T_pt],
             [G^T_tp,       G^T_tt]]

    where S^B is the Schur complement of the bottom-grid interior onto ports.

    Args:
        top_blocks: BlockMatrixSystem for top-grid (with pads eliminated)
        schur_B: SchurComplementOperator for bottom-grid

    Example:
        >>> top_blocks = extract_block_matrices(model, top_nodes, pad_nodes, port_nodes, vdd)
        >>> bottom_blocks = extract_block_matrices(model, bottom_nodes, set(), port_nodes, vdd)
        >>> bottom_blocks.factor_interior()
        >>> schur_B = SchurComplementOperator(...)
        >>> A = CoupledSystemOperator(top_blocks, schur_B)
        >>> solution, info = gmres(A, rhs)
    """

    def __init__(
        self,
        top_blocks: BlockMatrixSystem,
        schur_B: SchurComplementOperator,
    ):
        self.top_blocks = top_blocks
        self.schur_B = schur_B

        self.n_ports = top_blocks.n_ports
        self.n_top_interior = top_blocks.n_interior
        self.n_total = self.n_ports + self.n_top_interior

        super().__init__(dtype=np.float64, shape=(self.n_total, self.n_total))

    def _matvec(self, x: np.ndarray) -> np.ndarray:
        """Compute A * x for the coupled system."""
        x_p = x[: self.n_ports]
        x_t = x[self.n_ports :]

        # Port equations: (G^T_pp + S^B) * x_p + G^T_pt * x_t
        y_p = self.top_blocks.G_pp @ x_p
        y_p += self.schur_B @ x_p
        if self.n_top_interior > 0:
            y_p += self.top_blocks.G_pi @ x_t

        # Top interior equations: G^T_tp * x_p + G^T_tt * x_t
        if self.n_top_interior > 0:
            y_t = self.top_blocks.G_ip @ x_p + self.top_blocks.G_ii @ x_t
        else:
            y_t = np.array([])

        return np.concatenate([y_p, y_t])


class BlockDiagonalPreconditioner(spla.LinearOperator):
    """Block diagonal preconditioner for the coupled system.

    Approximates the coupled system with:
        M = [[G^T_pp + diag(S^B), 0    ],
             [0,                  G^T_tt]]

    where diag(S^B) is approximated by the diagonal of G^B_pp (a reasonable
    upper bound since S^B = G^B_pp - non-negative term).

    This preconditioner is cheap to apply (just diagonal scaling for ports,
    LU solve for top interior) and provides reasonable convergence acceleration.

    Args:
        top_blocks: BlockMatrixSystem for top-grid
        bottom_G_pp_diag: Diagonal of bottom-grid G_pp (approximation to diag(S^B))
    """

    def __init__(
        self,
        top_blocks: BlockMatrixSystem,
        bottom_G_pp_diag: np.ndarray,
    ):
        self.top_blocks = top_blocks
        self.n_ports = top_blocks.n_ports
        self.n_top_interior = top_blocks.n_interior
        self.n_total = self.n_ports + self.n_top_interior

        # Port block: diagonal of G^T_pp + diag(G^B_pp)
        top_pp_diag = np.array(top_blocks.G_pp.diagonal()).flatten()
        self.port_diag = top_pp_diag + bottom_G_pp_diag

        # Avoid division by zero
        self.port_diag = np.maximum(self.port_diag, 1e-12)

        # Top interior block: LU of G^T_tt (reuse from top_blocks if available)
        if self.n_top_interior > 0:
            if top_blocks.lu_ii is not None:
                self.lu_tt = top_blocks.lu_ii
            else:
                self.lu_tt = spla.factorized(top_blocks.G_ii.tocsc())
        else:
            self.lu_tt = None

        super().__init__(dtype=np.float64, shape=(self.n_total, self.n_total))

    def _matvec(self, x: np.ndarray) -> np.ndarray:
        """Apply M^{-1} * x."""
        x_p = x[: self.n_ports]
        x_t = x[self.n_ports :]

        # Port block: diagonal solve
        y_p = x_p / self.port_diag

        # Top interior block: LU solve
        if self.n_top_interior > 0 and self.lu_tt is not None:
            y_t = self.lu_tt(x_t)
        else:
            y_t = np.array([])

        return np.concatenate([y_p, y_t])


class ILUPreconditioner(spla.LinearOperator):
    """ILU-based preconditioner for harder problems.

    Uses incomplete LU factorization of an approximation to the coupled system
    matrix. More expensive to construct but may provide better convergence
    for ill-conditioned systems.

    Args:
        top_blocks: BlockMatrixSystem for top-grid
        bottom_G_pp: Bottom-grid G_pp matrix (used as Schur complement approximation)
        drop_tol: Drop tolerance for ILU factorization (default 1e-4)
        fill_factor: Fill factor for ILU factorization (default 10)
    """

    def __init__(
        self,
        top_blocks: BlockMatrixSystem,
        bottom_G_pp: sp.csr_matrix,
        drop_tol: float = 1e-4,
        fill_factor: int = 10,
    ):
        self.n_ports = top_blocks.n_ports
        self.n_top_interior = top_blocks.n_interior
        self.n_total = self.n_ports + self.n_top_interior

        # Build approximate coupled matrix
        # A_approx = [[G^T_pp + G^B_pp, G^T_pt],
        #             [G^T_tp,          G^T_tt]]
        # (Using G^B_pp as approximation to Schur complement S^B)

        if self.n_top_interior > 0:
            # Top-left block: G^T_pp + G^B_pp
            A_pp = top_blocks.G_pp + bottom_G_pp

            # Build full approximate matrix
            A_approx = sp.bmat(
                [
                    [A_pp, top_blocks.G_pi],
                    [top_blocks.G_ip, top_blocks.G_ii],
                ],
                format="csc",
            )
        else:
            # No top interior nodes, just ports
            A_approx = (top_blocks.G_pp + bottom_G_pp).tocsc()

        # Compute ILU factorization
        try:
            self.ilu = spla.spilu(A_approx, drop_tol=drop_tol, fill_factor=fill_factor)
        except RuntimeError:
            # Fall back to less aggressive ILU if factorization fails
            self.ilu = spla.spilu(A_approx, drop_tol=drop_tol * 10, fill_factor=fill_factor // 2)

        super().__init__(dtype=np.float64, shape=(self.n_total, self.n_total))

    def _matvec(self, x: np.ndarray) -> np.ndarray:
        """Apply M^{-1} * x using ILU factorization."""
        return self.ilu.solve(x)


# Try to import pyamg for AMG preconditioner (optional dependency)
try:
    import pyamg
    HAS_PYAMG = True
except ImportError:
    HAS_PYAMG = False
    pyamg = None


class AMGPreconditioner(spla.LinearOperator):
    """Algebraic Multigrid (AMG) preconditioner for the coupled system.

    AMG is near-optimal for graph Laplacians (like conductance matrices):
    - O(n) complexity per iteration
    - Mesh-independent convergence (iteration count stays ~10-20 regardless of size)
    - Excellent for large-scale problems (10M+ nodes)

    Uses pyamg's smoothed aggregation solver. Requires pyamg to be installed.

    Args:
        top_blocks: BlockMatrixSystem for top-grid
        bottom_G_pp: Bottom-grid G_pp matrix (used as Schur complement approximation)
        strength: Strength of connection threshold for AMG (default 'symmetric')
        max_coarse: Maximum coarse grid size (default 500)

    Raises:
        ImportError: If pyamg is not installed

    Example:
        >>> precond = AMGPreconditioner(top_blocks, bottom_blocks.G_pp)
        >>> x, info = scipy.sparse.linalg.cg(A, b, M=precond)
    """

    def __init__(
        self,
        top_blocks: BlockMatrixSystem,
        bottom_G_pp: sp.csr_matrix,
        strength: str = 'symmetric',
        max_coarse: int = 500,
    ):
        if not HAS_PYAMG:
            raise ImportError(
                "pyamg is required for AMG preconditioner. "
                "Install it with: pip install pyamg"
            )

        self.n_ports = top_blocks.n_ports
        self.n_top_interior = top_blocks.n_interior
        self.n_total = self.n_ports + self.n_top_interior

        # Build approximate coupled matrix (same as ILU)
        # A_approx = [[G^T_pp + G^B_pp, G^T_pt],
        #             [G^T_tp,          G^T_tt]]
        if self.n_top_interior > 0:
            A_pp = top_blocks.G_pp + bottom_G_pp
            A_approx = sp.bmat(
                [
                    [A_pp, top_blocks.G_pi],
                    [top_blocks.G_ip, top_blocks.G_ii],
                ],
                format="csr",
            )
        else:
            A_approx = (top_blocks.G_pp + bottom_G_pp).tocsr()

        # Build AMG hierarchy using smoothed aggregation
        # This is a one-time cost that gets amortized over many solves
        self.ml = pyamg.smoothed_aggregation_solver(
            A_approx,
            strength=strength,
            max_coarse=max_coarse,
            symmetry='hermitian',  # SPD system
        )

        # Store the preconditioner (M^{-1} application)
        self._M = self.ml.aspreconditioner(cycle='V')

        super().__init__(dtype=np.float64, shape=(self.n_total, self.n_total))

    def _matvec(self, x: np.ndarray) -> np.ndarray:
        """Apply M^{-1} * x using AMG V-cycle."""
        return self._M @ x

    @property
    def levels(self) -> int:
        """Number of levels in AMG hierarchy."""
        return len(self.ml.levels)

    @property
    def operator_complexity(self) -> float:
        """Operator complexity (nnz ratio across levels)."""
        return self.ml.operator_complexity()

    @property
    def grid_complexity(self) -> float:
        """Grid complexity (size ratio across levels)."""
        return self.ml.grid_complexity()


def compute_reduced_rhs(
    bottom_blocks: BlockMatrixSystem,
    current_injections: Dict[Any, float],
    rhs_dirichlet_bottom: np.ndarray,
) -> np.ndarray:
    """Compute reduced RHS r^B for the coupled system.

    The reduced RHS at ports is:
        r^B = (i_p + rhs_dirichlet_p) - G^B_pi * inv(G^B_ii) * (i_i + rhs_dirichlet_i)

    where i_p, i_i are current injections at port and interior nodes respectively,
    and rhs_dirichlet accounts for any Dirichlet boundary contributions.

    Args:
        bottom_blocks: BlockMatrixSystem for bottom-grid
        current_injections: Dict mapping node -> current (positive = sink)
        rhs_dirichlet_bottom: RHS contribution from Dirichlet nodes in bottom-grid

    Returns:
        Reduced RHS vector of shape (n_ports,)
    """
    n_ports = bottom_blocks.n_ports
    n_interior = bottom_blocks.n_interior

    # Build current vectors (negated for nodal equation)
    i_p = np.zeros(n_ports, dtype=np.float64)
    i_i = np.zeros(n_interior, dtype=np.float64)

    for node, current in current_injections.items():
        if node in bottom_blocks.port_to_idx:
            i_p[bottom_blocks.port_to_idx[node]] -= current  # Negate for nodal eqn
        elif node in bottom_blocks.interior_to_idx:
            i_i[bottom_blocks.interior_to_idx[node]] -= current  # Negate for nodal eqn

    # Add Dirichlet contributions
    rhs_p = i_p + rhs_dirichlet_bottom[:n_ports]
    rhs_i = i_i + rhs_dirichlet_bottom[n_ports : n_ports + n_interior]

    # Compute reduced RHS: r^B = rhs_p - G^B_pi * inv(G^B_ii) * rhs_i
    if n_interior > 0 and bottom_blocks.lu_ii is not None:
        v_i = bottom_blocks.lu_ii(rhs_i)
        r_B = rhs_p - bottom_blocks.G_pi @ v_i
    else:
        r_B = rhs_p

    return r_B


def recover_bottom_voltages(
    bottom_blocks: BlockMatrixSystem,
    port_voltages: np.ndarray,
    current_injections: Dict[Any, float],
    rhs_dirichlet_bottom: np.ndarray,
) -> Dict[Any, float]:
    """Recover bottom-grid interior voltages from port voltages.

    Once port voltages are known, interior voltages are recovered via:
        v_i = inv(G^B_ii) * (rhs_i - G^B_ip * v_p)

    Args:
        bottom_blocks: BlockMatrixSystem for bottom-grid
        port_voltages: Array of port voltages in port_nodes order
        current_injections: Dict mapping node -> current
        rhs_dirichlet_bottom: RHS contribution from Dirichlet nodes

    Returns:
        Dict mapping bottom-grid interior node -> voltage
    """
    n_ports = bottom_blocks.n_ports
    n_interior = bottom_blocks.n_interior

    voltages: Dict[Any, float] = {}

    if n_interior == 0 or bottom_blocks.lu_ii is None:
        return voltages

    # Build interior current vector (negated)
    i_i = np.zeros(n_interior, dtype=np.float64)
    for node, current in current_injections.items():
        if node in bottom_blocks.interior_to_idx:
            i_i[bottom_blocks.interior_to_idx[node]] -= current

    # Add Dirichlet contribution to interior RHS
    rhs_i = i_i + rhs_dirichlet_bottom[n_ports : n_ports + n_interior]

    # Compute: v_i = inv(G_ii) * (rhs_i - G_ip * v_p)
    coupling = bottom_blocks.G_ip @ port_voltages
    v_i = bottom_blocks.lu_ii(rhs_i - coupling)

    # Map back to node names
    for i, node in enumerate(bottom_blocks.interior_nodes):
        voltages[node] = float(v_i[i])

    return voltages

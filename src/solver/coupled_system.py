"""Block-partitioned matrices and core Schur complement math.

This module provides the core data structures and mathematical functions for
coupled hierarchical solving:

- BlockMatrixSystem: Holds block matrices and LU factorization
- extract_block_matrices: Build block system from a UnifiedPowerGridModel
- build_block_system_from_edges: Build block system from raw edge data
- compute_reduced_rhs / recover_bottom_voltages: Schur complement RHS math
- compute_explicit_schur: Dense Schur complement computation

Operators (SchurComplementOperator, CoupledSystemOperator, preconditioners)
are in coupled_operators.py. Interface assembly and island detection are in
interface_assembly.py. Both are re-exported here for backward compatibility.

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
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import scipy.sparse as sp

from model.unified_model import UnifiedPowerGridModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Partial Cholesky factorization threshold
# ---------------------------------------------------------------------------
# When the number of port nodes exceeds this threshold, compute_explicit_schur()
# uses a single full-matrix Cholesky factorization (ports last) and extracts
# S = L22 @ L22.T instead of the chunked multi-RHS solve path.  Set to 0 to
# always use the partial path (when CHOLMOD is available), or to a very large
# value to disable it.
_PARTIAL_FACTOR_THRESHOLD: int = 500
_PARTIAL_FACTOR_REG_OHMS: float = 1e8  # regularization resistance in Ohms


def set_partial_factor_threshold(value: int) -> None:
    """Set the minimum number of ports for the partial Cholesky path.

    Args:
        value: Threshold (0 = always use partial when CHOLMOD available,
               large value = effectively disable).
    """
    global _PARTIAL_FACTOR_THRESHOLD
    _PARTIAL_FACTOR_THRESHOLD = value


def get_partial_factor_threshold() -> int:
    """Return the current partial Cholesky threshold."""
    return _PARTIAL_FACTOR_THRESHOLD


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


def should_use_partial_factor(block_system: 'BlockMatrixSystem') -> bool:
    """Return True if the partial Cholesky path should be used for *block_system*.

    Checks: CHOLMOD available, port count >= threshold, interior non-empty,
    and the user hasn't forced the splu backend.
    """
    from .unified_solver import HAS_CHOLMOD, get_use_cholmod
    return (
        HAS_CHOLMOD
        and block_system.n_ports >= _PARTIAL_FACTOR_THRESHOLD
        and block_system.n_interior > 0
        and get_use_cholmod() is not False
    )


def _sparse_mem_bytes(M) -> int:
    """Memory footprint of a scipy sparse CSR/CSC matrix."""
    return M.data.nbytes + M.indices.nbytes + M.indptr.nbytes


def _format_bytes(n: int) -> str:
    """Human-readable byte size."""
    for unit in ('B', 'KB', 'MB', 'GB'):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


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

        Args:
            verbose: If True, print which backend is being used
        """
        if self.n_interior > 0:
            from .unified_solver import _factor_conductance_matrix
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


def _ensure_dense(x):
    """Convert sparse matrix to dense array if needed."""
    return x.toarray() if sp.issparse(x) else x


def _make_partial_interior_solve(
    cholmod_factor,
    P_ii: np.ndarray,
    n_interior: int,
    n_ports: int,
) -> Callable[[np.ndarray], np.ndarray]:
    """Create an interior solve closure using the full-matrix Cholesky factor.

    The closure implements the solve_L / solve_Lt truncation trick: given a
    Cholesky factor L of the full permuted matrix ``G_perm = L @ L.T`` with
    interior nodes ordered first and ports last, an interior solve
    ``G_ii^{-1} @ b`` is obtained by:

    1. Permute b to AMD order, pad with port zeros.
    2. Forward-solve: ``y = L^{-1} @ b_pad``.
    3. Zero the port portion of y.
    4. Backward-solve: ``x = L^{-T} @ y``.
    5. Un-permute the interior portion of x.

    Args:
        cholmod_factor: CHOLMOD factor of the full permuted matrix.
        P_ii: Fill-reducing permutation of interior indices (length n_interior).
        n_interior: Number of interior nodes.
        n_ports: Number of port nodes.

    Returns:
        Callable that maps ``b`` (1-D or 2-D, interior-sized) to
        ``G_ii^{-1} @ b`` in original interior ordering.
    """
    inv_P_ii = np.argsort(P_ii)  # inverse permutation (pre-computed once)
    n_full = n_interior + n_ports

    def solve(b):
        if b.ndim == 1:
            b_pad = np.zeros(n_full, dtype=np.float64)
            b_pad[:n_interior] = b[P_ii]
            y = cholmod_factor.solve_L(b_pad, use_LDLt_decomposition=False)
            y[n_interior:] = 0.0
            x = cholmod_factor.solve_Lt(y, use_LDLt_decomposition=False)
            return x[:n_interior][inv_P_ii]
        else:
            # Multi-RHS: handle sparse input (e.g. from G_ip chunks)
            if sp.issparse(b):
                b = b.toarray()
            n_rhs = b.shape[1]
            b_pad = np.zeros((n_full, n_rhs), dtype=np.float64)
            b_pad[:n_interior] = b[P_ii]
            y = cholmod_factor.solve_L(b_pad, use_LDLt_decomposition=False)
            y[n_interior:] = 0.0
            x = cholmod_factor.solve_Lt(y, use_LDLt_decomposition=False)
            return x[:n_interior][inv_P_ii]

    return solve


def _compute_schur_partial(
    block_system: BlockMatrixSystem,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Compute Schur complement via partial Cholesky of the full block matrix.

    Factors ``G_full = [[G_ii, G_ip], [G_pi, G_pp]]`` with interior nodes in a
    fill-reducing (AMD/METIS) order and ports appended last.  The Schur
    complement is then ``S = L22 @ L22.T`` where L22 is the lower-right
    ``n_ports x n_ports`` block of the Cholesky factor L.

    This replaces the O(n_ports) column-solve chunked approach with a single
    larger factorization + sparse slicing, which is faster when n_ports is large.

    As a side-effect, sets ``block_system.lu_ii`` to a closure that performs
    interior solves via the solve_L/solve_Lt truncation trick, and
    ``block_system.factor_adapter`` to a ``SparseFactorAdapter`` wrapping the
    full factor.

    Args:
        block_system: BlockMatrixSystem (lu_ii need NOT be pre-set).

    Returns:
        Tuple of (S, stats) where S is the dense Schur complement and stats
        contains timing and metadata.
    """
    from sksparse.cholmod import cholesky as cholmod_cholesky, analyze as cholmod_analyze
    from .unified_solver import (
        SparseFactorAdapter,
        get_cholmod_mode, get_cholmod_ordering, get_cholmod_use_long,
    )

    n_i = block_system.n_interior
    n_p = block_system.n_ports
    ordering = get_cholmod_ordering()
    mode = get_cholmod_mode()
    use_long = get_cholmod_use_long()

    logger.debug(
        "_compute_schur_partial: n_interior=%d, n_ports=%d, ordering=%s, mode=%s",
        n_i, n_p, ordering, mode,
    )

    # --- Phase 1: Analyze G_ii to obtain fill-reducing permutation -----------
    t0 = time.perf_counter()
    G_ii_csc = block_system.G_ii.tocsc()
    symbolic = cholmod_analyze(
        G_ii_csc, mode=mode, ordering_method=ordering, use_long=use_long,
    )
    P_ii = symbolic.P()
    t_analyze = time.perf_counter() - t0

    # --- Phase 2: Build full permuted matrix and factor ----------------------
    t0 = time.perf_counter()
    G_full = sp.bmat(
        [[block_system.G_ii, block_system.G_ip],
         [block_system.G_pi, block_system.G_pp]],
        format='csc',
    )
    full_perm = np.concatenate([P_ii, n_i + np.arange(n_p)])
    G_perm = G_full[np.ix_(full_perm, full_perm)].tocsc()

    # Regularize: add a small grounding conductance to port diagonals.
    # Tiles without direct ground connections have a PSD (singular)
    # Laplacian; this makes it SPD for Cholesky.  Subtracted from S after.
    R_TO_KOHM = 1e-3  # Ohm -> kOhm
    r_kohm = _PARTIAL_FACTOR_REG_OHMS * R_TO_KOHM
    g_reg = 1.0 / r_kohm  # conductance in mS (= 1/kOhm)
    diag = G_perm.diagonal()
    diag[n_i:] += g_reg
    G_perm.setdiag(diag)

    # Factor with natural ordering (our permutation is already applied)
    factor = cholmod_cholesky(
        G_perm, mode=mode, ordering_method='natural', use_long=use_long,
    )
    t_factor = time.perf_counter() - t0

    # --- Phase 3: Extract L22 and compute S ----------------------------------
    t0 = time.perf_counter()
    L = factor.L()  # sparse CSC, LL^T form (D absorbed)
    L22 = L[n_i:, n_i:].toarray()  # dense n_p x n_p
    S = L22 @ L22.T
    # Subtract port regularization: S_reg = S_true + g_reg * I
    np.fill_diagonal(S, S.diagonal() - g_reg)
    t_extract = time.perf_counter() - t0

    # --- Side-effects: set lu_ii and factor_adapter on block_system -----------
    block_system.lu_ii = _make_partial_interior_solve(factor, P_ii, n_i, n_p)
    use_long_str = 'auto' if use_long is None else ('64-bit' if use_long else '32-bit')
    info_str = (
        f"cholmod(partial, mode={mode}, ordering={ordering}, "
        f"natural, idx={use_long_str})"
    )
    # NOTE: factor_adapter wraps the FULL (n_i + n_p) Cholesky factor, not
    # the G_ii factor.  It is stored here ONLY for backend metadata
    # (.backend, .backend_info).  Do NOT call factor_adapter.solve() --
    # use lu_ii() for interior solves instead.
    block_system.factor_adapter = SparseFactorAdapter(factor, 'cholmod', info_str)

    stats = {
        'path': 'partial_cholesky',
        'ordering': ordering,
        'mode': mode,
        'chunk_size': 0,       # compat keys for callers that read these
        'n_chunks': 0,
        'schur_mem_bytes': S.nbytes,
        'full_factor_nnz': L.nnz,
        'L22_dense_shape': L22.shape,
        'analyze_s': t_analyze,
        'factor_s': t_factor,
        'extract_s': t_extract,
    }
    logger.debug(
        "_compute_schur_partial: S %dx%d (%s), L nnz=%d, "
        "analyze %.3fs, factor %.3fs, extract %.3fs",
        n_p, n_p, _format_bytes(S.nbytes), L.nnz,
        t_analyze, t_factor, t_extract,
    )
    return S, stats


def compute_explicit_schur(
    block_system: BlockMatrixSystem,
    max_memory_gb: float = 4.0,
    use_partial_factor: Optional[bool] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Compute explicit Schur complement S = G_pp - G_pi * inv(G_ii) * G_ip.

    Two computation paths are available:

    **Partial Cholesky** (``use_partial_factor=True``): Factors the full block
    matrix ``[[G_ii, G_ip], [G_pi, G_pp]]`` with a fill-reducing interior
    ordering and ports last, then extracts S = L22 @ L22.T.  Does NOT require
    ``lu_ii`` to be pre-set (factorization is done internally).  Requires
    CHOLMOD.  Preferred when n_ports is large (>= threshold).

    **Chunked multi-RHS** (``use_partial_factor=False``): Solves
    ``G_ii^{-1} @ G_ip`` column-by-column in chunks.  Requires ``lu_ii`` to
    be set (call ``factor_interior()`` first).  Works with both CHOLMOD and
    splu backends.

    When ``use_partial_factor`` is None (default), the path is auto-selected
    based on CHOLMOD availability, port count, and user configuration.

    Args:
        block_system: BlockMatrixSystem. For the chunked path, ``lu_ii`` must
            be set. For the partial path, ``lu_ii`` is set as a side-effect.
        max_memory_gb: Memory budget for Z_chunk in the chunked path
            (default 4.0 GB).
        use_partial_factor: If True, force the partial Cholesky path. If
            False, force the chunked path. If None (default), auto-detect.

    Returns:
        Tuple of (S, stats) where S is the dense Schur complement matrix of
        shape (n_ports, n_ports) and stats is a dict with computation metadata.
        Both paths include ``chunk_size``, ``n_chunks``, and
        ``schur_mem_bytes`` keys for backward compatibility.

    Raises:
        ValueError: If the chunked path is selected but ``lu_ii`` is not set.
    """
    n_ports = block_system.n_ports
    n_interior = block_system.n_interior

    if n_interior == 0:
        # No interior nodes: Schur complement is just G_pp
        S = block_system.G_pp.toarray()
        stats = {
            'path': 'trivial',
            'chunk_size': 0,
            'n_chunks': 0,
            'schur_mem_bytes': S.nbytes,
        }
        return S, stats

    # --- Auto-detect path when use_partial_factor is None --------------------
    from .unified_solver import HAS_CHOLMOD
    if use_partial_factor is None:
        use_partial = should_use_partial_factor(block_system)
    else:
        use_partial = use_partial_factor

    # --- Partial Cholesky path -----------------------------------------------
    if use_partial:
        if not HAS_CHOLMOD:
            raise ImportError(
                "Partial Cholesky factorization requires sksparse.cholmod. "
                "Install with: pip install scikit-sparse"
            )
        return _compute_schur_partial(block_system)

    # --- Chunked multi-RHS path (existing) -----------------------------------
    if block_system.lu_ii is None:
        raise ValueError("Interior not factored. Call factor_interior() first.")

    # Compute optimal chunk size balancing memory, CHOLMOD limits, and BLAS efficiency
    INT_MAX = 2**31 - 1
    # Assume ~50% fill for Z_chunk (sparse RHS -> partially dense solution)
    bytes_per_col = n_interior * 8 * 0.5
    memory_chunk = max(1, int(max_memory_gb * 1e9 / bytes_per_col))
    index_chunk = max(1, INT_MAX // max(n_interior, 1))
    # Cap at 256 for BLAS efficiency (diminishing returns beyond), floor at 32
    chunk_size = min(memory_chunk, index_chunk, n_ports, 256)
    chunk_size = max(chunk_size, min(32, n_ports))

    n_chunks = (n_ports + chunk_size - 1) // chunk_size

    logger.debug(
        "compute_explicit_schur: n_ports=%d, n_interior=%d, chunk_size=%d, n_chunks=%d",
        n_ports, n_interior, chunk_size, n_chunks,
    )

    # S starts as G_pp (dense copy)
    S = block_system.G_pp.toarray()

    # Chunked sparse solve (CHOLMOD accepts sparse RHS)
    # When chunk_size >= n_ports, loop runs once processing all columns
    G_ip_csc = block_system.G_ip.tocsc()
    for start in range(0, n_ports, chunk_size):
        end = min(start + chunk_size, n_ports)
        G_ip_chunk = G_ip_csc[:, start:end]  # Sparse slice (no toarray!)
        Z_chunk = block_system.lu_ii(G_ip_chunk)
        S[:, start:end] -= _ensure_dense(block_system.G_pi @ Z_chunk)

    stats = {
        'path': 'chunked',
        'chunk_size': chunk_size,
        'n_chunks': n_chunks,
        'schur_mem_bytes': S.nbytes,
    }
    logger.debug(
        "compute_explicit_schur: Schur %dx%d, memory %s",
        n_ports, n_ports, _format_bytes(S.nbytes),
    )
    return S, stats


def build_block_system_from_edges(
    edges,
    port_nodes: Set[str],
    dirichlet_nodes: Optional[Set[str]] = None,
    dirichlet_voltage: float = 0.0,
    ground_node: str = '0',
) -> Tuple[BlockMatrixSystem, np.ndarray]:
    """Build BlockMatrixSystem from raw (u, v, conductance) edge data.

    Generalized variant of extract_block_matrices() that accepts edges directly
    instead of requiring a UnifiedPowerGridModel. Same constants (GMAX,
    SHORT_THRESHOLD), same ground handling (diagonal-only), same Dirichlet
    elimination (-G_ud @ V_d), same port/interior ordering.

    All non-port, non-Dirichlet, non-ground nodes are classified as interior.

    Args:
        edges: Iterable of (u, v, conductance) tuples. Conductance is g = 1/R.
        port_nodes: Set of boundary/port node names
        dirichlet_nodes: Set of nodes with fixed voltage (e.g., pads). Defaults to empty.
        dirichlet_voltage: Voltage applied at Dirichlet nodes (default 0.0)
        ground_node: Name of ground node (default '0'). Edges touching ground
            add to diagonal only.

    Returns:
        Tuple of:
        - BlockMatrixSystem with extracted block matrices
        - rhs_dirichlet: Array of shape (n_ports + n_interior,) containing
          Dirichlet contributions in [port_nodes, interior_nodes] ordering
    """
    if dirichlet_nodes is None:
        dirichlet_nodes = set()

    GMAX = 1e5
    SHORT_THRESHOLD = 1e-6

    # Collect all nodes from edges
    all_nodes_set: Set[str] = set()
    edge_list = []
    for u, v, g in edges:
        all_nodes_set.add(u)
        all_nodes_set.add(v)
        edge_list.append((u, v, g))

    # Remove ground from classification (handled via diagonal-only)
    all_nodes_set.discard(ground_node)

    # Classify nodes
    port_set = port_nodes & all_nodes_set
    dirichlet_set = dirichlet_nodes & all_nodes_set
    interior_nodes = all_nodes_set - port_set - dirichlet_set

    # Order nodes: ports first, then interior, then Dirichlet
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
        # Edge case: all nodes are Dirichlet (or no nodes)
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

    # Build conductance matrix via COO
    data, rows, cols = [], [], []
    diag = np.zeros(len(all_nodes), dtype=np.float64)

    for u, v, g in edge_list:
        # Clamp conductance
        if g <= 0 or g > GMAX:
            g = min(max(g, 1.0 / GMAX), GMAX)
        if 1.0 / g < SHORT_THRESHOLD:
            g = GMAX

        # Ground handling: only add to diagonal of the non-ground node
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

    # Add diagonal entries
    for i in range(len(all_nodes)):
        rows.append(i)
        cols.append(i)
        data.append(diag[i])

    n_total = len(all_nodes)
    G_full = sp.csr_matrix((data, (rows, cols)), shape=(n_total, n_total))

    # Extract block submatrices
    p_idx = np.arange(n_ports)
    i_idx = np.arange(n_ports, n_ports + n_interior)
    d_idx = np.arange(n_unknown, n_total)
    u_idx = np.arange(n_unknown)

    G_pp = G_full[np.ix_(p_idx, p_idx)].tocsr() if n_ports > 0 else sp.csr_matrix((0, 0))
    G_pi = G_full[np.ix_(p_idx, i_idx)].tocsr() if n_ports > 0 and n_interior > 0 else sp.csr_matrix((n_ports, 0))
    G_ip = G_full[np.ix_(i_idx, p_idx)].tocsr() if n_interior > 0 and n_ports > 0 else sp.csr_matrix((0, n_ports))
    G_ii = G_full[np.ix_(i_idx, i_idx)].tocsr() if n_interior > 0 else sp.csr_matrix((0, 0))

    # Compute RHS contribution from Dirichlet nodes: rhs = -G_ud * V_d
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
    C_ip = C_pi = 0.

    For each capacitive edge ``(u, v, c_fF)`` the non-ground terminal is
    located in either ``port_to_idx`` or ``interior_to_idx`` and the
    capacitance is accumulated on the corresponding diagonal vector.
    Edges whose non-ground node appears in neither mapping (e.g. removed
    island nodes) are silently skipped.

    Args:
        cap_edges: List of ``(u, v, capacitance_fF)`` tuples where one
            terminal is the ground node.
        port_to_idx: Dict mapping port node name -> index in port vector.
        interior_to_idx: Dict mapping interior node name -> index in
            interior vector.
        n_ports: Length of the port vector (used for array allocation).
        n_interior: Length of the interior vector (used for array allocation).
        ground_node: Name of the ground node (default ``'0'``).

    Returns:
        Tuple of ``(c_pp_diag, c_ii_diag, total_capacitance_fF)`` where

        - ``c_pp_diag``: shape ``(n_ports,)`` -- per-port capacitance in fF
        - ``c_ii_diag``: shape ``(n_interior,)`` -- per-interior capacitance in fF
        - ``total_capacitance_fF``: sum of all accumulated capacitances
    """
    c_pp_diag = np.zeros(n_ports, dtype=np.float64)
    c_ii_diag = np.zeros(n_interior, dtype=np.float64)
    total_cap = 0.0

    for u, v, c_fF in cap_edges:
        if c_fF <= 0:
            continue

        # Identify the non-ground node
        if u == ground_node and v == ground_node:
            continue
        if u == ground_node:
            node = v
        elif v == ground_node:
            node = u
        else:
            # Both terminals non-ground -- not a grounded cap; skip.
            continue

        idx = port_to_idx.get(node)
        if idx is not None:
            c_pp_diag[idx] += c_fF
            total_cap += c_fF
            continue

        idx = interior_to_idx.get(node)
        if idx is not None:
            c_ii_diag[idx] += c_fF
            total_cap += c_fF
        # else: node not in either mapping (removed island node) -- skip.

    return c_pp_diag, c_ii_diag, total_cap


# ---------------------------------------------------------------------------
# Re-exports from split-out modules for backward compatibility.
# All existing ``from solver.coupled_system import X`` continue to work.
# ---------------------------------------------------------------------------

from .coupled_operators import (  # noqa: E402, F401
    SchurComplementOperator,
    CoupledSystemOperator,
    BlockDiagonalPreconditioner,
    ILUPreconditioner,
    AMGPreconditioner,
    HAS_PYAMG,
)

from .interface_assembly import (  # noqa: E402, F401
    assemble_schur_complement_system,
    build_interface_package_matrices,
    find_interface_islands,
    apply_island_penalty,
    detect_interface_islands,
)

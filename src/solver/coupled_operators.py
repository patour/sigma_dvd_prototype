"""Iterative solve operators and preconditioners for coupled hierarchical solving.

Contains matrix-free operators for the Schur complement and the full coupled
system, plus block-diagonal, ILU, and AMG preconditioners.

Split from coupled_system.py for maintainability. All public names are
re-exported from coupled_system so existing imports keep working.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

if TYPE_CHECKING:
    from .coupled_system import BlockMatrixSystem


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

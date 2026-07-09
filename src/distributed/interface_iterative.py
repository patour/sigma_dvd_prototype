"""Iterative interface solve for large-scale DDM.

Provides InterfaceCGSolver, a wrapper around scipy.sparse.linalg.cg that
replaces the direct CHOLMOD/SuperLU factorization of the global Schur
complement S_global.  Two coordinator-side matvec modes are supported:

  'assembled'  -- matvec on the assembled sparse S_global (removes the
                  dominant ~100-300 GB CHOLMOD factor; keeps ~50 GB S_global).
  'tilewise'   -- matvec = sum_i P_i^T (S_i @ (P_i @ x)) using the per-tile
                  dense S_i blocks + tile_index_maps.  Avoids global assembly
                  entirely (removes both the ~50 GB S_global and its factor).

Worker-side RPC matvec (S_i lives on the worker) is out of scope -- noted as
future work for B4.

Preconditioners (pluggable, via ``preconditioner`` keyword):
  'block_jacobi' (default) -- block-Jacobi from the owner's principal S_i
                              submatrix for each interface node.  Ownership:
                              first tile whose tile_index_map includes the node.

                              Memory note: each owned block is a k_i x k_i
                              dense Cholesky factor.  Total factor memory scales
                              as Sum(k_i^2) ≈ n^2/T for T balanced tiles.  At
                              n=1M interface nodes with T=1000 tiles this is
                              ~8 GB for factors alone — the same coordinator-
                              memory class CG is designed to avoid.  For
                              netlist_sampled (n~2K, max block 478) the factor
                              is ~5.9 MB (negligible).

                              When the estimated factor memory exceeds
                              ``BLOCK_JACOBI_MAX_FACTOR_BYTES`` (default 4 GB),
                              _build_block_jacobi() automatically falls back to
                              'jacobi' (diagonal) and logs a WARNING.  For
                              1M-interface systems with few tiles, prefer
                              'jacobi' explicitly or reduce tile size via B1
                              splitting so T is large enough that individual
                              block sizes are manageable.
  'jacobi'                  -- diagonal of S_global (assembled) or the summed
                              tile S_i diagonals (tilewise).  Cheap, scales
                              to any interface size; weaker than block_jacobi
                              but does not use O(n^2/T) memory.
  'none'                    -- identity (no preconditioning).
  'amg'                     -- algebraic multigrid via pyamg (lazy import,
                              skipped gracefully when pyamg is not installed).

Save / load semantics for CG mode
-----------------------------------
CG mode has no factor to save (that is the whole point).  On save():
  - S_global is stored as usual (used by assembled mode; can also be used to
    refactor into direct mode later).
  - The ``interface_solver`` setting is stored so that load() + refactor()
    reconstructs a CG solve callable, not an LU factor.
On load() + refactor() in CG mode:
  - ``refactor`` rebuilds a CG callable from S_global (for assembled mode) or
    from S_global + tile_schur_complements (for tilewise mode).
  - Per-tile S_i blocks are NOT saved (too large; always recomputed).
  - Simplest correct behaviour: after load(), callers must call factor()
    for tilewise mode (which re-runs factor_and_compute_schur on workers),
    or refactor() for assembled mode (which rebuilds the CG op from S_global).

Adjoint note
------------
analyze_adjoint* uses the interface factor.  CG mode is acceptable for DC
adjoint (the linear system has the same coefficient matrix).  However, the
existing adjoint implementation passes the *factor callable* ctx._interface_lu
directly for the forward solve and expects a linear-system solve.  CG provides
exactly that interface (same call signature), so adjoint DC works out of the
box for CG assembled mode.  Tilewise adjoint requires per-tile S_i blocks that
the adjoint code never re-assembles -- for v1, force direct for adjoint by
raising a clear error when interface_solver='cg' with tilewise matvec is used
and an adjoint call is attempted.  Document here; the adjoint code checks
ctx._interface_solver_mode and falls back gracefully.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Threshold for auto-select: use CG when n_interface >= this value.
# Overridable via monkeypatch in tests.
# ---------------------------------------------------------------------------
AUTO_CG_N_INTERFACE_THRESHOLD: int = 200_000

# ---------------------------------------------------------------------------
# Sentinel: factor-memory budget (bytes) for auto-select.
# CHOLMOD supernodal factor is roughly 5-10x S_global memory.
# At 50 GB S_global (138K interface) CHOLMOD factor = 100-300 GB.
# We use a 32 GB budget as the auto cutover (conservative for coordinator).
# ---------------------------------------------------------------------------
AUTO_CG_FACTOR_MEMORY_BUDGET_BYTES: int = 32 * 1024 ** 3  # 32 GB

# ---------------------------------------------------------------------------
# Block-Jacobi preconditioner memory budget.
#
# Each owned block is a k_i x k_i dense float64 Cholesky factor.  Total
# factor memory scales as Sum(k_i^2) = n^2/T for T balanced tiles.
# At n=1M interface nodes with T=1000 tiles: ~8 GB just for factors.
# With fewer or larger tiles the cost grows quickly into the coordinator-
# memory class that CG is designed to avoid.
#
# When the estimated block-Jacobi factor memory exceeds this threshold,
# _build_block_jacobi() falls back to 'jacobi' (diagonal) automatically
# and logs a WARNING.  Override via monkeypatch in tests.
#
# Default 4 GB is conservative: keeps the preconditioner build out of the
# multi-GB danger zone while still allowing block_jacobi on all realistic
# netlist_sampled / BRCM-class systems where n_interface is 2-140K.
# (Measured 5.9 MB on netlist_sampled 2015-interface max-block-478 system.)
# ---------------------------------------------------------------------------
BLOCK_JACOBI_MAX_FACTOR_BYTES: int = 4 * 1024 ** 3  # 4 GB


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class InterfaceCGSolver:
    """Iterative CG solver for the global Schur complement interface system.

    Wraps scipy.sparse.linalg.cg with a LinearOperator.  The returned
    callable has the same signature as the direct LU callable:
        v_gamma = solver(rhs)

    Parameters
    ----------
    n_interface : int
        Size of the interface system (number of unknowns).
    matvec_mode : str
        'assembled' or 'tilewise'.
    S_global : sp.csc_matrix, optional
        Assembled sparse Schur matrix.  Required for mode='assembled'.
        Also used for 'jacobi' preconditioner in tilewise mode.
    tile_schur_complements : dict, optional
        {tile_id: S_i (dense np.ndarray)}.  Required for mode='tilewise'.
    tile_index_maps : dict, optional
        {tile_id: np.ndarray(int32)} mapping local port index -> global
        interface index.  Required for mode='tilewise'.
    preconditioner : str
        'block_jacobi' (default), 'jacobi', 'none', 'amg'.
    rtol : float
        Relative tolerance for CG convergence (default 1e-10).
    maxiter : int or None
        Maximum CG iterations.  None = 3 * n_interface.
    x0 : np.ndarray or None
        Initial guess for the next solve.  Warm-start support.
    stats_dict : dict or None
        If provided, iteration counts and timing are written here.
        Updated in-place on each solve call.
    """

    def __init__(
        self,
        n_interface: int,
        matvec_mode: str = 'assembled',
        S_global: Optional[sp.spmatrix] = None,
        tile_schur_complements: Optional[Dict[Any, np.ndarray]] = None,
        tile_index_maps: Optional[Dict[Any, np.ndarray]] = None,
        S_extra: Optional[sp.spmatrix] = None,
        preconditioner: str = 'block_jacobi',
        rtol: float = 1e-12,
        atol: float = 1e-14,
        maxiter: Optional[int] = None,
        x0: Optional[np.ndarray] = None,
        stats_dict: Optional[Dict[str, Any]] = None,
        strict: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        S_extra : sp.spmatrix, optional
            Additional sparse contribution to the matvec, added on top of the
            per-tile Schur sum.  For ``matvec_mode='tilewise'`` this carries the
            package-edge contribution (G_pkg_uu) that is NOT included in the
            per-tile Schur complements.  Ignored in assembled mode (where S_global
            already contains all contributions).
        atol : float
            Absolute tolerance for CG convergence.  The CG stopping criterion is
            ``||r|| <= max(rtol * ||b||, atol)``.  The default (1e-14) prevents
            CG from burning all maxiter iterations when the RHS is near-zero
            (e.g. early transient steps with no active sources), where
            ``rtol * ||b|| -> 0`` would otherwise make convergence unreachable.
            Set to 0 to recover the pure rtol behaviour.
        strict : bool
            If True (default), raise ``RuntimeError`` when CG does not converge
            within maxiter iterations.  The error message includes the relative
            residual so it appears clearly in logs.  Set to False to demote to a
            warning (not recommended for production; useful only for unit-testing
            the non-convergence path).
        """
        if matvec_mode not in ('assembled', 'tilewise'):
            raise ValueError(
                f"matvec_mode must be 'assembled' or 'tilewise', got {matvec_mode!r}"
            )
        if matvec_mode == 'assembled' and S_global is None:
            raise ValueError("matvec_mode='assembled' requires S_global")
        if matvec_mode == 'tilewise' and (
            tile_schur_complements is None or tile_index_maps is None
        ):
            raise ValueError(
                "matvec_mode='tilewise' requires tile_schur_complements "
                "and tile_index_maps"
            )
        if preconditioner not in ('block_jacobi', 'jacobi', 'none', 'amg'):
            raise ValueError(
                f"preconditioner must be one of 'block_jacobi', 'jacobi', "
                f"'none', 'amg'; got {preconditioner!r}"
            )

        self.n = n_interface
        self.matvec_mode = matvec_mode
        self.S_global = S_global
        self.tile_schur_complements = tile_schur_complements
        self.tile_index_maps = tile_index_maps
        self.S_extra = S_extra  # package-edge contribution (tilewise mode)
        self.preconditioner = preconditioner
        self.rtol = rtol
        self.atol = atol
        self.strict = strict
        self.maxiter = maxiter if maxiter is not None else max(3 * n_interface, 100)
        self._x0: Optional[np.ndarray] = x0
        self._stats: Dict[str, Any] = stats_dict if stats_dict is not None else {}

        # Build operator + preconditioner
        self._linear_op: spla.LinearOperator = self._build_linear_op()
        self._M: Optional[spla.LinearOperator] = self._build_preconditioner()

        # Track cumulative iteration stats
        self._total_iters: int = 0
        self._total_solves: int = 0

    # ------------------------------------------------------------------
    # LinearOperator / preconditioner construction
    # ------------------------------------------------------------------

    def _build_linear_op(self) -> spla.LinearOperator:
        """Build the matvec LinearOperator for S_global."""
        n = self.n

        if self.matvec_mode == 'assembled':
            # Convert to CSR for efficient matvec
            S = self.S_global.tocsr()

            def _matvec(x: np.ndarray) -> np.ndarray:
                return S @ x

        else:  # 'tilewise'
            # Precompute list of (idx_map, S_i) for fast per-step iteration.
            # idx_map entries may repeat across tiles (shared interface nodes),
            # so scatter-add is required.  We use np.bincount (one call per
            # global index per tile) rather than np.add.at because np.add.at
            # is ~10-30x slower for this scatter pattern (same bottleneck A1
            # replaced in solver_td.py with bincount for the transient RHS).
            _tiles: List[Tuple[np.ndarray, np.ndarray]] = []
            for tid, S_i in self.tile_schur_complements.items():
                idx = self.tile_index_maps[tid]
                _tiles.append((idx, np.asarray(S_i, dtype=np.float64)))
            _n = n
            # Package-edge contribution (sparse, small).  Not included in per-tile S_i.
            _S_extra = (
                self.S_extra.tocsr() if self.S_extra is not None else None
            )

            def _matvec(x: np.ndarray) -> np.ndarray:  # type: ignore[misc]
                # Start with package-edge contribution
                if _S_extra is not None:
                    result = (_S_extra @ x).astype(np.float64)
                else:
                    result = np.zeros(_n, dtype=np.float64)
                for idx, S_i in _tiles:
                    x_local = x[idx]
                    y_local = S_i @ x_local
                    # Scatter-add via bincount: ~10-30x faster than np.add.at
                    # for dense index maps with repeated global indices.
                    # np.bincount requires non-negative integer indices (guaranteed
                    # by int32 tile_index_maps built from interface_node_to_idx).
                    result += np.bincount(idx, weights=y_local, minlength=_n)
                return result

        return spla.LinearOperator(
            shape=(n, n), matvec=_matvec, dtype=np.float64
        )

    def _build_preconditioner(self) -> Optional[spla.LinearOperator]:
        """Build preconditioner LinearOperator."""
        n = self.n

        if self.preconditioner == 'none':
            return None

        if self.preconditioner == 'jacobi':
            # Extract diagonal from S_global (assembled) or sum tile diagonals
            if self.S_global is not None:
                diag = np.array(self.S_global.diagonal(), dtype=np.float64)
            else:
                diag = np.zeros(n, dtype=np.float64)
                for tid, S_i in self.tile_schur_complements.items():
                    idx = self.tile_index_maps[tid]
                    np.add.at(diag, idx, np.diag(S_i))
            diag = np.where(diag > 0, diag, 1.0)  # guard against zero diagonal

            def _msolve(x: np.ndarray) -> np.ndarray:
                return x / diag

            return spla.LinearOperator(shape=(n, n), matvec=_msolve, dtype=np.float64)

        if self.preconditioner == 'block_jacobi':
            return self._build_block_jacobi()

        if self.preconditioner == 'amg':
            return self._build_amg()

        return None

    def _build_block_jacobi(self) -> Optional[spla.LinearOperator]:
        """Block-Jacobi preconditioner.

        Two implementation paths:
        1. S_global available (assembled mode or tilewise with S_global passed):
           uses the actual principal submatrices of S_global for each tile's
           owned nodes.  These are the TRUE block-diagonal blocks of the global
           system.  Much better preconditioner than path 2.
        2. S_global not available (tilewise-only): falls back to per-tile
           Schur complement submatrices.  These are NOT the diagonal blocks of
           S_global (they miss cross-tile couplings and package edges), so they
           are a weaker approximation.

        Ownership assignment: each interface node is assigned to the FIRST
        tile (in iteration order of tile_index_maps keys) whose tile_index_map
        includes it.  The owner contributes the block.

        Each per-owner submatrix is factored via scipy.linalg.cho_factor (dense
        Cholesky) for numerically robust small-block solves.  Falls back to
        pinv on singular blocks (1e-12 diagonal regularization applied first).

        Memory guard: when the estimated factor memory across all blocks exceeds
        ``BLOCK_JACOBI_MAX_FACTOR_BYTES`` (default 4 GB), falls back to the
        'jacobi' (diagonal) preconditioner automatically and logs a WARNING.
        This prevents the block-Jacobi from re-introducing the coordinator-
        memory pressure that CG mode is designed to remove.  Scaling analysis:
        for T balanced tiles with n total interface nodes, total factor memory
        is Sum(k_i^2) * 8 bytes ≈ n^2/T * 8 bytes.  At n=1M, T=1000 this is
        ~8 GB; with fewer/larger tiles it grows rapidly.  Use 'jacobi' or
        increase T (via B1 splitting) when targeting 1M+ interface nodes.

        Returns None only on a complete build failure; triggers logged warning.
        """
        n = self.n
        import scipy.linalg as la

        # ---- 1. Assign ownership (first-seen tile for each global index) ----
        owner_tile: Dict[int, Any] = {}
        if self.tile_index_maps:
            for tid, idx in self.tile_index_maps.items():
                for g in idx:
                    gi = int(g)
                    if gi not in owner_tile:
                        owner_tile[gi] = tid
        elif self.tile_schur_complements is None:
            # Neither tile_index_maps nor tile_schur_complements: cannot build
            return None

        # ---- 2. Collect owned global indices per tile ----
        tile_owned: Dict[Any, List[int]] = {}
        for g, tid in owner_tile.items():
            tile_owned.setdefault(tid, []).append(g)

        # ---- 2b. Pre-flight memory estimate ----
        # Sum(k_i^2) * 8 bytes = total dense Cholesky factor memory.
        # Each k_i is the number of nodes owned by tile i.  Use the
        # owned-count dict (already built) to estimate without allocating.
        est_factor_bytes = sum(
            len(owned) * len(owned) * 8 for owned in tile_owned.values()
        )
        if est_factor_bytes > BLOCK_JACOBI_MAX_FACTOR_BYTES:
            max_block = max(len(v) for v in tile_owned.values()) if tile_owned else 0
            logger.warning(
                "Block-Jacobi: estimated factor memory %.1f GB exceeds budget "
                "%.1f GB (n_interface=%d, T=%d tiles, max_block=%d). "
                "Scaling: Sum(k_i^2)*8 bytes ~ n^2/T*8; at n=1M, T=1000 this "
                "is ~8 GB. Falling back to 'jacobi' (diagonal) preconditioner. "
                "To keep block_jacobi, reduce interface size via B1 tile splitting "
                "or set BLOCK_JACOBI_MAX_FACTOR_BYTES higher.",
                est_factor_bytes / 1024 ** 3,
                BLOCK_JACOBI_MAX_FACTOR_BYTES / 1024 ** 3,
                n,
                len(tile_owned),
                max_block,
            )
            # Fall back to diagonal preconditioner (cheap, always safe)
            return self._build_jacobi_fallback()

        logger.debug(
            "Block-Jacobi: estimated factor memory %.1f MB "
            "(n=%d, T=%d tiles, max_block=%d).",
            est_factor_bytes / 1024 ** 2,
            n,
            len(tile_owned),
            max(len(v) for v in tile_owned.values()) if tile_owned else 0,
        )

        # ---- 3. Extract submatrices and build Cholesky factors ----
        _block_factors: List[Tuple[np.ndarray, Any, bool]] = []

        # Path 1: use actual diagonal blocks of S_global (preferred)
        _use_s_global = self.S_global is not None
        if _use_s_global:
            S_csr = self.S_global.tocsr()

        for tile_or_node_group, owned_global in tile_owned.items():
            owned_global_arr = np.array(sorted(owned_global), dtype=np.int32)

            if _use_s_global:
                # Extract true diagonal block from S_global
                sub = S_csr[np.ix_(owned_global_arr, owned_global_arr)].toarray()
            elif (self.tile_schur_complements is not None
                  and self.tile_index_maps is not None):
                # Fallback: use per-tile Schur block (less accurate)
                tid = tile_or_node_group
                S_i = np.asarray(self.tile_schur_complements[tid], dtype=np.float64)
                idx_full = self.tile_index_maps[tid]
                global_to_local: Dict[int, int] = {
                    int(g): loc for loc, g in enumerate(idx_full)
                }
                owned_local = np.array(
                    [global_to_local[g] for g in owned_global_arr], dtype=np.int32
                )
                sub = S_i[np.ix_(owned_local, owned_local)].copy()
            else:
                continue

            # Regularize to ensure SPD
            sub += 1e-12 * np.eye(sub.shape[0])

            try:
                cho = la.cho_factor(sub, lower=False, check_finite=False)
                _block_factors.append((owned_global_arr, cho, False))
            except la.LinAlgError:
                logger.warning(
                    "Block-Jacobi: owned block size %d is singular; "
                    "using pinv fallback.",
                    sub.shape[0],
                )
                try:
                    pinv = np.linalg.pinv(sub)
                    _block_factors.append((owned_global_arr, pinv, True))
                except Exception:
                    logger.warning(
                        "Block-Jacobi: pinv also failed for block size %d; "
                        "skipping.",
                        sub.shape[0],
                    )

        if not _block_factors:
            logger.warning(
                "Block-Jacobi: all blocks failed; falling back to no preconditioner."
            )
            return None

        def _bj_solve(x: np.ndarray) -> np.ndarray:
            result = x.copy()  # identity for any unowned nodes
            for global_idx, factor, is_pinv in _block_factors:
                x_sub = x[global_idx]
                if is_pinv:
                    y_sub = factor @ x_sub
                else:
                    y_sub = la.cho_solve(factor, x_sub, check_finite=False)
                result[global_idx] = y_sub
            return result

        return spla.LinearOperator(
            shape=(n, n), matvec=_bj_solve, dtype=np.float64
        )

    def _build_jacobi_fallback(self) -> Optional[spla.LinearOperator]:
        """Build a diagonal (Jacobi) preconditioner as a fallback.

        Used when block_jacobi memory exceeds the budget threshold, or as an
        explicit alternative. Cheap and scalable to any interface size.
        """
        n = self.n
        if self.S_global is not None:
            diag = np.array(self.S_global.diagonal(), dtype=np.float64)
        elif self.tile_schur_complements is not None and self.tile_index_maps is not None:
            diag = np.zeros(n, dtype=np.float64)
            for tid, S_i in self.tile_schur_complements.items():
                idx = self.tile_index_maps[tid]
                # Scatter-add diagonal entries using bincount (fast path)
                diag += np.bincount(idx, weights=np.diag(S_i), minlength=n)
        else:
            return None  # cannot build any preconditioner

        diag = np.where(diag > 0, diag, 1.0)

        def _diag_solve(x: np.ndarray) -> np.ndarray:
            return x / diag

        return spla.LinearOperator(shape=(n, n), matvec=_diag_solve, dtype=np.float64)

    def _build_amg(self) -> Optional[spla.LinearOperator]:
        """AMG preconditioner via pyamg (optional dependency).

        Requires the assembled S_global (CSR format).  Returns None if
        pyamg is not installed or S_global is not available.
        """
        try:
            import pyamg  # type: ignore[import]
        except ImportError:
            logger.info(
                "pyamg not installed; falling back to block_jacobi preconditioner."
            )
            return self._build_block_jacobi()

        if self.S_global is None:
            logger.warning(
                "AMG preconditioner requires S_global (assembled mode); "
                "falling back to block_jacobi."
            )
            return self._build_block_jacobi()

        try:
            ml = pyamg.smoothed_aggregation_solver(self.S_global.tocsr())
            M_op = ml.aspreconditioner(cycle='V')
            return M_op
        except Exception as exc:
            logger.warning(
                "AMG setup failed (%s); falling back to block_jacobi.", exc
            )
            return self._build_block_jacobi()

    # ------------------------------------------------------------------
    # Solve (callable interface matching direct LU)
    # ------------------------------------------------------------------

    def __call__(self, rhs: np.ndarray) -> np.ndarray:
        """Solve S_global @ x = rhs via CG.

        Args:
            rhs: Right-hand-side vector, shape (n_interface,).

        Returns:
            Solution vector x, shape (n_interface,).

        Raises:
            RuntimeError: If CG fails to converge within maxiter.
        """
        t0 = time.perf_counter()

        x0 = self._x0

        iters: List[int] = [0]

        def _callback(xk: np.ndarray) -> None:
            iters[0] += 1

        result, info = spla.cg(
            self._linear_op,
            rhs,
            x0=x0,
            rtol=self.rtol,
            atol=self.atol,
            maxiter=self.maxiter,
            M=self._M,
            callback=_callback,
        )

        elapsed = time.perf_counter() - t0
        n_iter = iters[0]

        if info < 0:
            raise RuntimeError(
                f"InterfaceCGSolver: CG failed with illegal input (info={info})."
            )

        # Accumulate stats before any possible raise so callers that catch the
        # RuntimeError can still inspect stats['cg_failed'] / 'total_cg_failures'.
        self._total_iters += n_iter
        self._total_solves += 1
        self._stats['last_cg_iters'] = n_iter
        self._stats['last_cg_time_s'] = elapsed
        self._stats['last_cg_info'] = info
        self._stats.setdefault('total_cg_iters', 0)
        self._stats['total_cg_iters'] = self._total_iters
        self._stats.setdefault('total_cg_solves', 0)
        self._stats['total_cg_solves'] = self._total_solves

        _failed = info > 0
        self._stats['cg_failed'] = _failed
        self._stats.setdefault('total_cg_failures', 0)

        if info > 0:
            # Compute actual relative residual so the error message is informative.
            r_norm = float(np.linalg.norm(rhs - self._linear_op.matvec(result)))
            b_norm = float(np.linalg.norm(rhs))
            rel_res = r_norm / b_norm if b_norm > 0 else r_norm
            self._stats['last_cg_rel_residual'] = rel_res
            self._stats['total_cg_failures'] += 1

            if self.strict:
                raise RuntimeError(
                    f"InterfaceCGSolver: CG did not converge after {n_iter} "
                    f"iterations (rtol={self.rtol:.2e}, atol={self.atol:.2e}, "
                    f"rel_residual={rel_res:.3e}). "
                    f"Increase maxiter, relax tolerances, or improve the "
                    f"preconditioner. Set strict=False to demote to a warning."
                )
            else:
                logger.warning(
                    "InterfaceCGSolver: CG did not converge after %d iterations "
                    "(rtol=%.2e, rel_residual=%.3e). "
                    "Using best iterate (strict=False).",
                    n_iter, self.rtol, rel_res,
                )

        # Warm-start: update x0 for next call
        self._x0 = result.copy()

        logger.debug(
            "InterfaceCGSolver: %d iters, info=%d, %.4fs (warm=%s)",
            n_iter, info, elapsed, x0 is not None,
        )

        return result

    # ------------------------------------------------------------------
    # Warm-start control
    # ------------------------------------------------------------------

    def reset_warm_start(self) -> None:
        """Clear the warm-start initial guess (force cold start next call)."""
        self._x0 = None

    def set_x0(self, x0: Optional[np.ndarray]) -> None:
        """Explicitly set the warm-start guess for the next solve call."""
        self._x0 = x0 if x0 is None else np.asarray(x0, dtype=np.float64).copy()

    # ------------------------------------------------------------------
    # Stats access
    # ------------------------------------------------------------------

    @property
    def stats(self) -> Dict[str, Any]:
        """Read-only view of accumulated stats (same dict as stats_dict)."""
        return self._stats

    @property
    def total_iterations(self) -> int:
        """Total CG iterations across all calls."""
        return self._total_iters

    @property
    def total_solves(self) -> int:
        """Total solve calls."""
        return self._total_solves


# ---------------------------------------------------------------------------
# Auto-select helper (used by result_factorization)
# ---------------------------------------------------------------------------


def auto_select_interface_solver(
    n_interface: int,
    S_global: Optional[sp.spmatrix] = None,
    n_interface_threshold: int = AUTO_CG_N_INTERFACE_THRESHOLD,
    factor_memory_budget_bytes: int = AUTO_CG_FACTOR_MEMORY_BUDGET_BYTES,
) -> str:
    """Decide whether to use 'direct' or 'cg' for the interface solve.

    Rules:
      1. If n_interface < n_interface_threshold AND estimated factor memory
         fits within factor_memory_budget_bytes: return 'direct'.
      2. Otherwise: return 'cg'.

    The factor memory estimate uses S_global.nnz * 8 * factor_fill_ratio
    (empirical fill ratio 5 for supernodal CHOLMOD on PDN-class matrices).

    Args:
        n_interface: Number of interface unknowns.
        S_global: Assembled Schur matrix (optional; used for memory estimate).
        n_interface_threshold: Cutover size (default 200_000).
        factor_memory_budget_bytes: Memory budget in bytes (default 32 GB).

    Returns:
        'direct' or 'cg'.
    """
    if n_interface < n_interface_threshold:
        if S_global is not None:
            # Rough factor memory: nnz * bytes_per_double * fill_ratio
            fill_ratio = 5.0
            est_bytes = S_global.nnz * 8 * fill_ratio
            if est_bytes > factor_memory_budget_bytes:
                logger.info(
                    "auto_select: n_interface=%d < threshold but estimated "
                    "factor memory %.1f GB > budget %.1f GB; using CG.",
                    n_interface,
                    est_bytes / 1024 ** 3,
                    factor_memory_budget_bytes / 1024 ** 3,
                )
                return 'cg'
        return 'direct'

    logger.info(
        "auto_select: n_interface=%d >= threshold %d; using CG.",
        n_interface, n_interface_threshold,
    )
    return 'cg'


# ---------------------------------------------------------------------------
# Factory: build an interface_lu callable from CG or direct
# ---------------------------------------------------------------------------


def build_interface_solver(
    S_global: sp.spmatrix,
    interface_solver: str = 'auto',
    tile_schur_complements: Optional[Dict[Any, np.ndarray]] = None,
    tile_index_maps: Optional[Dict[Any, np.ndarray]] = None,
    matvec_mode: str = 'assembled',
    preconditioner: str = 'block_jacobi',
    rtol: float = 1e-12,
    coordinator_solver_config: Optional[Any] = None,
    verbose: bool = False,
    cg_stats_dict: Optional[Dict[str, Any]] = None,
) -> Tuple[Callable, str, Optional['InterfaceCGSolver']]:
    """Build an interface solve callable (direct LU or CG).

    This is the single entry point called by _factor_dc_context and
    _factor_transient_context after S_global is assembled.

    Args:
        S_global: Assembled sparse Schur matrix (CSR or CSC).
        interface_solver: 'direct', 'cg', or 'auto'.
        tile_schur_complements: Per-tile dense Schur blocks (for tilewise).
        tile_index_maps: Per-tile index maps (for tilewise).
        matvec_mode: 'assembled' or 'tilewise' (only used when cg selected).
        preconditioner: 'block_jacobi', 'jacobi', 'none', or 'amg'.
        rtol: CG convergence tolerance.
        coordinator_solver_config: SolverBackendConfig (for direct path).
        verbose: Log the resolved backend.
        cg_stats_dict: Mutable dict for CG statistics.

    Returns:
        (solve_callable, resolved_mode, cg_solver_or_none)
        resolved_mode is 'direct' or 'cg'.
        cg_solver_or_none is the InterfaceCGSolver when cg is used, else None.
    """
    n = S_global.shape[0]

    # Resolve 'auto'
    if interface_solver == 'auto':
        resolved = auto_select_interface_solver(n, S_global)
    else:
        resolved = interface_solver

    if resolved == 'direct':
        from pgmath.factor import _factor_conductance_matrix
        t0 = time.perf_counter()
        result = _factor_conductance_matrix(
            S_global, verbose=verbose, config=coordinator_solver_config,
        )
        elapsed = time.perf_counter() - t0
        if verbose:
            logger.info(
                "Interface direct factor: %s, %.3fs, n=%d",
                result.backend_info, elapsed, n,
            )
        return result.solve, 'direct', None

    # CG path
    if cg_stats_dict is None:
        cg_stats_dict = {}

    # For tilewise mode: require tile_schur_complements + tile_index_maps
    if matvec_mode == 'tilewise' and (
        tile_schur_complements is None or tile_index_maps is None
    ):
        logger.warning(
            "build_interface_solver: matvec_mode='tilewise' requested but "
            "tile_schur_complements or tile_index_maps not provided; "
            "falling back to 'assembled' mode."
        )
        matvec_mode = 'assembled'

    t0 = time.perf_counter()
    cg_solver = InterfaceCGSolver(
        n_interface=n,
        matvec_mode=matvec_mode,
        S_global=S_global,
        tile_schur_complements=tile_schur_complements,
        tile_index_maps=tile_index_maps,
        preconditioner=preconditioner,
        rtol=rtol,
        stats_dict=cg_stats_dict,
    )
    elapsed = time.perf_counter() - t0
    if verbose:
        logger.info(
            "Interface CG solver built: mode=%s, precond=%s, rtol=%.2e, n=%d, "
            "build_time=%.3fs",
            matvec_mode, preconditioner, rtol, n, elapsed,
        )

    return cg_solver, 'cg', cg_solver

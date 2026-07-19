"""Two-level coarse-space additive preconditioner for the interface CG solve.

Stage 3 (interface_solve_acceleration_plan.md / docs Sec 7.8): Stage 2 measured
that cold block-Jacobi CG STAGNATES at the mi200k_v2 split regime (64 tiles,
167,659 interface unknowns) -- rel-res 0.32 -> 0.27 over iterations 200 -> 1000
at the DC solve. Root cause: ``x . M^-1 x ~ 1e15`` vs ``x . Ax ~ 1e9`` on random
vectors -- the cho-factored block-Jacobi ownership blocks have genuine
near-null eigendirections (~1e-10 relative), so ``kappa(M^-1 S) >~ 1e6``.
Block-Jacobi intrinsically collapses at this granularity.

This module builds an ADDITIVE two-level correction on top of block-Jacobi::

    M^-1 = M_bj^-1 + Z S_c^+ Z^T

where ``Z`` (n x T') is a coarse-space basis (partition-of-unity columns, one
per tile, plus GenEO-lite eigenvector columns capturing each block's
near-null directions), ``S_c = Z^T S Z`` is the small (T' x T') coarse Schur
operator, and ``S_c^+`` is its eigh-based PSD pseudo-inverse (structural rank
deficiency -- e.g. the alternating-sign checkerboard combination of
partition-of-unity columns on an even-multiplicity grid tiling -- is
EXPECTED, not an error).

Contents
--------
``build_partition_of_unity``
    Nicolaides partition-of-unity coarse-space columns: one per tile
    (weighted 1/multiplicity on shared boundary nodes), plus one indicator
    column for interface unknowns owned by NO tile (package/die/tap nodes).
    Island-penalized rows are zeroed (the 1e5 mS penalty diagonal would
    otherwise dominate S_c and neuter the coarse correction).

``geneo_lowest_eigenpairs``
    GenEO-lite enrichment: the ``k`` lowest eigenpairs of a single
    block-Jacobi ownership block, filtered to eigenvalue <=
    ``tol * lambda_max(block)``.  Reuses the block's existing Cholesky
    factor via ``eigsh(..., sigma=0, OPinv=...)`` shift-invert (no
    refactor) for large blocks; falls back to dense ``eigh`` for small
    blocks or when a precomputed full spectrum is already available (the
    ``_spd_safe_pseudo_solve_factor`` indefinite-block fallback path in
    ``interface_iterative.py`` already eigendecomposes the block -- that
    spectrum is passed in via ``precomputed`` instead of being recomputed).

``CoarseSpace``
    Holds ``Z`` (sparse), the eigh factorization of ``S_c^+``, and column
    bookkeeping; ``apply(r)`` computes ``Z S_c^+ Z^T r`` via three small
    matvecs (never materializes ``S_c^+`` as a dense operator on ``n``).
    Never persisted across ``save()``/``release()`` -- rebuilt on every
    ``factor()``/``refactor()`` exactly like the per-tile dense Schur blocks
    it is derived from.

``build_coarse_space``
    Top-level builder: assembles ``Z`` (PoU + GenEO columns), computes
    ``S_c = Z^T (S Z)`` via the caller-supplied ``matmat`` (the SAME
    tilewise/assembled matvec the CG solve itself uses, so ``S_extra`` --
    package edges + island penalties -- is automatically included), and
    factors ``S_c^+`` via eigh with rank truncation.  If the full (PoU +
    GenEO) column count T' exceeds ``max_cols``, degrades to the PoU-only
    prefix of ``Z`` (WARNING, GenEO columns dropped) rather than disabling
    the coarse space outright -- PoU-only is a strictly smaller space
    (``n_tiles + 1`` columns vs. up to ``geneo_k`` extra per tile) so it can
    still fit comfortably where PoU+GenEO overflows.  Returns ``None``
    (logging a WARNING) only when the coarse space cannot be built at all
    (no tile_index_maps, PoU-only T' ALSO exceeds ``max_cols``, or ``S_c``
    has no positive spectrum) -- callers degrade to the plain block-Jacobi
    preconditioner in that case.

Island rows in GenEO columns
-----------------------------
VERIFIED (do not re-derive without checking this): block-Jacobi ownership
assignment (``_build_block_jacobi`` in ``interface_iterative.py``) is built
from ``tile_index_maps``, which do NOT exclude island nodes -- an
interface-island node is still a real physical node on some tile's port
list; it is handled by a 1e5 mS penalty stamped into ``S_extra``'s diagonal
via ``apply_island_penalty`` / ``_build_s_extra_direct``, NOT by removal from
the interface unknown set or any tile's port map. Consequently an island
node CAN be the "owner" of (or contribute a row to) a block-Jacobi block, so
a block's GenEO eigenvectors CAN have nonzero components on island rows.
``build_coarse_space`` zeros those rows on every GenEO column it scatters,
exactly like ``build_partition_of_unity`` zeros them on the PoU columns --
otherwise the penalty-dominated island row would leak spurious weight into
the coarse space and corrupt ``S_c``.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults (also the model.settings / CLI defaults -- see cli.py
# _IFACE_SETTING_DEFAULTS and result_factorization.py
# _read_interface_cg_settings).
# ---------------------------------------------------------------------------
DEFAULT_GENEO_K: int = 4
DEFAULT_GENEO_TOL: float = 1e-6
DEFAULT_EPS_RANK: float = 1e-12
DEFAULT_MAX_COLS: int = 4096
# Finding 5 (round 1) / Finding 3 (round 2): byte-based guard on the dense
# (n x T') fp64 allocations build_coarse_space's matmat call chain holds
# concurrently -- Z_dense + SZ (the two arrays materialized directly in
# build_coarse_space) PLUS the union of per-thread compact buffers
# `_tilewise_matmat` allocates internally while computing SZ (~one more
# n*T'*8 held alongside Z_dense and the in-progress SZ result) -- true peak
# is ~3*n*T'*8, not 2*n*T'*8.  Distinct from DEFAULT_MAX_COLS (a
# column-count cap that does not scale with n).  8 GB matches
# resolve_block_jacobi_max_bytes's 'auto' cap (both are coordinator-resident
# dense-array budgets of the same order); see
# interface_iterative.resolve_coarse_max_bytes for the host-aware 'auto'
# resolution (min(8 GB, 0.1 * total RAM) via psutil).
DEFAULT_MAX_BYTES: int = 8 * 1024 ** 3

# Below this block size, GenEO eigendecomposition uses dense np.linalg.eigh
# directly rather than ARPACK shift-invert (eigsh has diminishing returns --
# and a k < N requirement -- at small N; dense eigh is both simpler and
# usually faster below a few hundred rows).
GENEO_EIGSH_SMALL_BLOCK: int = 500

# Finding 7: fp32 tilewise matvec floor for the S_c rank-truncation
# threshold.  S_c = Z^T (S Z) is computed through the SAME matmat the CG
# solve itself uses (build_coarse_space's ``matmat`` arg) -- when that
# matmat runs in float32 (interface_matvec_dtype='float32'), S_c entries
# carry ~1e-7-relative fp32 GEMM noise.  The fp64 default (eps_rank=1e-12)
# would then treat noise-level eigenvalues as "structural rank deficiency"
# and invert them, injecting amplified (~1e7x) error into the coarse
# correction.  Mirrors interface_iterative.FP32_MATVEC_MIN_RTOL's role for
# CG's own rtol.
FP32_COARSE_EPS_RANK_FLOOR: float = 1e-6


# ---------------------------------------------------------------------------
# 1. Partition-of-unity coarse-space columns
# ---------------------------------------------------------------------------


def build_partition_of_unity(
    tile_index_maps: Dict[Any, np.ndarray],
    n: int,
    island_idx: Optional[np.ndarray] = None,
    unowned_idx: Optional[np.ndarray] = None,
) -> Tuple[sp.csr_matrix, List[Any], int]:
    """Build the Nicolaides partition-of-unity coarse-space basis ``Z``.

    Column ``j`` (one per tile, in ``tile_index_maps`` iteration order):
    ``Z[g, j] = 1/m[g]`` for every global interface index ``g`` in
    ``tile_index_maps[tile_j]``, where ``m = bincount(concat(tile_index_maps
    .values()))`` is each node's tile-membership multiplicity (shared
    boundary nodes appear in >1 tile's port map). An optional extra column
    (label ``'__unowned__'``) is appended as a plain indicator (weight 1) over
    ``unowned_idx`` -- interface unknowns present in NO tile's port map
    (package/die-attachment/tap unknowns that block-Jacobi treats as
    identity) -- so the coarse space still "sees" them.

    Args:
        tile_index_maps: ``{tile_id: np.ndarray(int)}`` -- global interface
            indices each tile's (pad-filtered) port list touches.
        n: Interface system size.
        island_idx: Global indices of interface-island nodes (penalized via
            ``S_extra``'s 1e5 mS diagonal).  Zeroed out of every column post-
            hoc so the penalty does not leak into the coarse space.
        unowned_idx: Global indices with no tile-map membership at all.

    Returns:
        ``(Z, col_labels, n_dropped)`` -- ``Z`` is CSR ``(n, T')`` with all-
        zero columns (fully-islanded/Dirichlet tiles) already dropped;
        ``col_labels`` names each surviving column (tile id, or
        ``'__unowned__'``); ``n_dropped`` is the number of columns removed.
    """
    tids = list(tile_index_maps.keys())
    if not tids and (unowned_idx is None or len(unowned_idx) == 0):
        return sp.csr_matrix((n, 0)), [], 0

    all_idx = (
        np.concatenate([
            np.asarray(tile_index_maps[t], dtype=np.int64) for t in tids
        ])
        if tids else np.empty(0, dtype=np.int64)
    )
    m = np.bincount(all_idx, minlength=n).astype(np.float64) if all_idx.size else np.zeros(n)
    m_safe = np.where(m > 0, m, 1.0)

    rows: List[np.ndarray] = []
    cols: List[np.ndarray] = []
    data: List[np.ndarray] = []
    col_labels: List[Any] = []

    for j, tid in enumerate(tids):
        idx = np.asarray(tile_index_maps[tid], dtype=np.int64)
        col_labels.append(tid)
        if idx.size == 0:
            continue
        rows.append(idx)
        cols.append(np.full(idx.shape, j, dtype=np.int64))
        data.append(1.0 / m_safe[idx])

    n_cols = len(tids)
    if unowned_idx is not None and len(unowned_idx) > 0:
        u_idx = np.asarray(unowned_idx, dtype=np.int64)
        rows.append(u_idx)
        cols.append(np.full(u_idx.shape, n_cols, dtype=np.int64))
        data.append(np.ones(u_idx.shape, dtype=np.float64))
        col_labels.append('__unowned__')
        n_cols += 1

    if rows:
        rows_cat = np.concatenate(rows)
        cols_cat = np.concatenate(cols)
        data_cat = np.concatenate(data)
    else:
        rows_cat = np.empty(0, dtype=np.int64)
        cols_cat = np.empty(0, dtype=np.int64)
        data_cat = np.empty(0, dtype=np.float64)

    Z = sp.csr_matrix((data_cat, (rows_cat, cols_cat)), shape=(n, n_cols))

    if island_idx is not None and len(island_idx) > 0 and Z.nnz > 0:
        isl_mask = np.zeros(n, dtype=bool)
        isl_mask[np.asarray(island_idx, dtype=np.int64)] = True
        Zcoo = Z.tocoo()
        keep = ~isl_mask[Zcoo.row]
        Z = sp.csr_matrix(
            (Zcoo.data[keep], (Zcoo.row[keep], Zcoo.col[keep])), shape=Z.shape,
        )

    # Drop all-zero columns (fully-islanded or all-Dirichlet tile: every row
    # it would have contributed got zeroed above, or its index array was
    # empty to begin with).  Finding 14: convert to CSC once and reuse it
    # for both the indptr-diff nnz check and the column slice below,
    # instead of two separate O(nnz) tocsc() conversions.
    Zc = Z.tocsc()
    col_nnz = np.diff(Zc.indptr)
    keep_cols = col_nnz > 0
    n_dropped = int((~keep_cols).sum())
    if n_dropped:
        dropped_labels = [col_labels[j] for j in range(len(col_labels)) if not keep_cols[j]]
        logger.info(
            "build_partition_of_unity: dropping %d all-zero column(s) "
            "(fully-islanded/Dirichlet tiles): %s",
            n_dropped, dropped_labels,
        )
        Z = Zc[:, keep_cols].tocsr()
        col_labels = [lbl for j, lbl in enumerate(col_labels) if keep_cols[j]]

    return Z, col_labels, n_dropped


# ---------------------------------------------------------------------------
# 2. GenEO-lite enrichment
# ---------------------------------------------------------------------------


def geneo_lowest_eigenpairs(
    sub: np.ndarray,
    cho: Optional[Any] = None,
    k: int = DEFAULT_GENEO_K,
    tol: float = DEFAULT_GENEO_TOL,
    small_block_threshold: int = GENEO_EIGSH_SMALL_BLOCK,
    precomputed: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    island_local_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Up to ``k`` lowest eigenpairs of ``sub`` with eigenvalue <=
    ``tol * lambda_max(sub)``.

    Args:
        sub: Dense symmetric block (a block-Jacobi ownership block).
        cho: The block's existing ``scipy.linalg.cho_factor`` result, reused
            for ARPACK shift-invert (``sigma=0``) -- NEVER refactored here.
            Required (and used) only when ``precomputed`` is ``None`` and the
            block is >= ``small_block_threshold``.
        k: Requested number of eigenpairs (capped internally at ``n - 1`` for
            the ARPACK path, or ``n`` for dense eigh).
        tol: Relative eigenvalue threshold (fraction of ``lambda_max``).
        small_block_threshold: Blocks smaller than this always use dense
            ``eigh`` (ARPACK has a hard ``k < N`` requirement and little
            benefit at small N).
        precomputed: ``(w_full, V_full)`` -- an already-computed full
            eigendecomposition of ``sub`` (e.g. from the indefinite-block
            ``_spd_safe_pseudo_solve_factor`` fallback in
            ``interface_iterative.py``) to reuse instead of recomputing.
        island_local_mask: Finding 3 -- boolean array of length ``n`` (True =
            island-penalized row/col, in THIS block's local ordering).  When
            any entry is True, the eigen-analysis (BOTH the lambda_max
            estimate and the k-lowest eigenpairs) runs on the PRINCIPAL
            SUBMATRIX restricted to the non-island rows/cols, not on the
            full (penalty-inflated) ``sub``: island rows carry a ~1e5 mS
            penalty diagonal (see the ownership-block assembly in
            ``interface_iterative.py``) that would otherwise dominate
            ``lambda_max`` by ~3 orders of magnitude, making the
            ``tol * lambda_max`` near-null threshold far too permissive and
            letting ordinary (non-near-null) eigenpairs of the PHYSICAL
            spectrum through as spurious "near-null" GenEO columns.  The
            returned ``V_k`` is scattered back to the full ``n``-row space
            with island rows implicitly zero (also re-zeroed defensively by
            ``build_coarse_space``'s own island-row zeroing on the Z
            scatter). ``cho`` (factored for the FULL, non-restricted
            ``sub``) and ``precomputed`` (an eigendecomposition of the FULL
            ``sub``) cannot be reused for the restricted submatrix -- but
            (Finding 4, round 2) that does NOT mean this path must fall back
            to dense ``eigh``: it cho-factors the RESTRICTED submatrix
            itself (a fresh, cheap O(k^3/3) factorization -- far cheaper
            than the O(k^3) dense eigendecomposition it replaces) and
            recurses with THAT factor, so a large island-touching block
            still takes the ARPACK shift-invert path exactly like the
            unmasked flow. Only falls back to dense ``eigh`` (via the
            recursive call's own existing fallback machinery) when the
            restricted cho-factorization itself raises ``LinAlgError``
            (the restricted submatrix is indefinite/singular) or ARPACK
            fails to converge with no usable partial result.

    Returns:
        ``(V_k, w_k)`` -- ``V_k`` has shape ``(n, j)`` with ``0 <= j <= k``
        (``j = 0`` -> no columns pass the threshold, or ``k <= 0``/``n == 0``);
        ``w_k`` holds the corresponding eigenvalues, ascending.
    """
    n = sub.shape[0]
    if k <= 0 or n == 0:
        return np.zeros((n, 0)), np.zeros(0)

    if island_local_mask is not None and np.any(island_local_mask):
        keep = ~np.asarray(island_local_mask, dtype=bool)
        n_keep = int(np.count_nonzero(keep))
        if n_keep == 0:
            return np.zeros((n, 0)), np.zeros(0)
        sub_restricted = sub[np.ix_(keep, keep)]
        # Finding 4: cho-factor the RESTRICTED submatrix ONCE (cheap --
        # O(k^3/3), not the O(k^3) dense eigh it lets us skip) so the
        # recursive call below can take the same ARPACK shift-invert path
        # the unmasked flow uses, instead of unconditionally recursing into
        # dense eigh.  A LinAlgError here (restricted submatrix is
        # indefinite/singular -- can happen since restricting to non-island
        # rows/cols is not guaranteed to preserve SPD-ness) passes cho=None
        # through, which the recursive call's own `cho is None` branch
        # already treats as "use dense eigh" -- exactly the documented
        # fallback.
        cho_restricted: Optional[Any] = None
        if n_keep >= small_block_threshold:
            try:
                sub_restricted_sym = 0.5 * (sub_restricted + sub_restricted.T)
                cho_restricted = la.cho_factor(sub_restricted_sym, check_finite=False)
            except la.LinAlgError:
                cho_restricted = None
        V_restricted, w_k = geneo_lowest_eigenpairs(
            sub_restricted, cho=cho_restricted, k=k, tol=tol,
            small_block_threshold=small_block_threshold, precomputed=None,
        )
        V_k = np.zeros((n, V_restricted.shape[1]))
        V_k[keep, :] = V_restricted
        return V_k, w_k

    w: Optional[np.ndarray] = None
    V: Optional[np.ndarray] = None

    if precomputed is not None:
        w, V = precomputed
    elif cho is None or n < small_block_threshold or k >= n:
        sub_sym = 0.5 * (sub + sub.T)
        w, V = np.linalg.eigh(sub_sym)
    else:
        # Finding 1: tracked OUTSIDE the try block (initialized None) so the
        # ArpackNoConvergence handler below can tell "the k=1 LM lambda_max
        # estimate already succeeded, use its TRUE value" apart from "the
        # exception hit before lambda_max was ever computed" -- see that
        # branch's comment for why using max(w_partial) instead is wrong.
        lam_max_true: Optional[float] = None
        try:
            w_max, _ = spla.eigsh(sub, k=1, which='LM')
            lam_max_true = float(w_max[0])
            OPinv = spla.LinearOperator(
                (n, n),
                matvec=lambda v: la.cho_solve(cho, v, check_finite=False),
                dtype=np.float64,
            )
            w_lo, V_lo = spla.eigsh(
                sub, k=min(k, n - 1), sigma=0.0, which='LM', OPinv=OPinv,
            )
            return _select_lowest(V_lo, w_lo, lam_max_true, k, tol)
        except spla.ArpackNoConvergence as exc:
            w_partial = np.asarray(exc.eigenvalues) if exc.eigenvalues is not None else np.empty(0)
            V_partial = (
                np.asarray(exc.eigenvectors) if exc.eigenvectors is not None
                else np.empty((n, 0))
            )
            if w_partial.size == 0 or lam_max_true is None:
                logger.warning(
                    "geneo_lowest_eigenpairs: ARPACK shift-invert produced no "
                    "usable partial result on a %dx%d block; falling back to "
                    "dense eigh.", n, n,
                )
                sub_sym = 0.5 * (sub + sub.T)
                w, V = np.linalg.eigh(sub_sym)
            else:
                logger.info(
                    "geneo_lowest_eigenpairs: ARPACK shift-invert did not "
                    "fully converge on a %dx%d block; using %d partial "
                    "eigenpair(s).", n, n, w_partial.size,
                )
                # Finding 1: threshold against the block's TRUE lambda_max
                # (the k=1 LM eigsh above, already computed and in scope) --
                # NOT max(w_partial).  w_partial holds the LOWEST
                # eigenvalues by construction (shift-invert sigma=0), so
                # max(w_partial) is itself tiny; thresholding a near-null
                # partial set against its own tiny max silently rejects
                # every one of them (tol * lam_max_partial ~ tol * near-null
                # ~= 0), defeating GenEO on exactly the ill-conditioned
                # blocks it targets.
                return _select_lowest(V_partial, w_partial, lam_max_true, k, tol)
        except Exception as exc:
            logger.warning(
                "geneo_lowest_eigenpairs: shift-invert eigsh failed on a "
                "%dx%d block (%s: %s); falling back to dense eigh.",
                n, n, type(exc).__name__, exc,
            )
            sub_sym = 0.5 * (sub + sub.T)
            w, V = np.linalg.eigh(sub_sym)

    lam_max = float(w[-1]) if w.size else 0.0
    return _select_lowest(V, w, lam_max, k, tol)


def _select_lowest(
    V: np.ndarray, w: np.ndarray, lam_max: float, k: int, tol: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Pick the <= k lowest entries of ``w`` (with matching columns of
    ``V``) whose value is <= ``tol * lam_max``."""
    n = V.shape[0]
    if lam_max <= 0 or w.size == 0:
        return np.zeros((n, 0)), np.zeros(0)
    thresh = tol * lam_max
    order = np.argsort(w)
    sel = order[w[order] <= thresh][:k]
    return np.ascontiguousarray(V[:, sel]), np.ascontiguousarray(w[sel])


# ---------------------------------------------------------------------------
# 3/4/5. Coarse operator: build, factor (eigh pseudo-inverse), apply
# ---------------------------------------------------------------------------


class CoarseSpace:
    """Additive coarse-space correction ``Z S_c^+ Z^T`` (never persisted).

    Built fresh on every ``factor()``/``refactor()`` -- like the per-tile
    dense Schur blocks it derives from, it is not part of any save()/load()
    checkpoint.

    Attributes:
        Z: Sparse ``(n, T')`` coarse-space basis (PoU + GenEO columns, all-
            zero columns already dropped).
        V_c: Dense ``(rank, rank)`` eigenvectors of the truncated ``S_c``
            eigendecomposition (``rank <= T'``).
        inv_lambda_c: ``(rank,)`` reciprocal eigenvalues (the pseudo-inverse
            diagonal in the ``V_c`` basis).
        n_pou_cols / n_geneo_cols / n_dropped_cols: Column-count bookkeeping
            for logging/introspection.
        rank / cond_estimate: Diagnostics from the eigh rank-truncation step.
    """

    __slots__ = (
        'Z', 'V_c', 'inv_lambda_c', 'n_pou_cols', 'n_geneo_cols',
        'n_dropped_cols', 'rank', 'cond_estimate', 'col_labels',
    )

    def __init__(
        self,
        Z: sp.csr_matrix,
        V_c: np.ndarray,
        inv_lambda_c: np.ndarray,
        n_pou_cols: int,
        n_geneo_cols: int,
        n_dropped_cols: int,
        rank: int,
        cond_estimate: float,
        col_labels: List[Any],
    ) -> None:
        self.Z = Z
        self.V_c = V_c
        self.inv_lambda_c = inv_lambda_c
        self.n_pou_cols = n_pou_cols
        self.n_geneo_cols = n_geneo_cols
        self.n_dropped_cols = n_dropped_cols
        self.rank = rank
        self.cond_estimate = cond_estimate
        self.col_labels = col_labels

    @property
    def n_cols(self) -> int:
        """T' -- total coarse-space column count (post drop)."""
        return self.Z.shape[1]

    def apply(self, r: np.ndarray) -> np.ndarray:
        """``Z @ (V_c @ (inv_lambda_c * (V_c.T @ (Z.T @ r))))``.

        Accepts either a 1-D vector or a ``(n, k)`` batch (matmat-style);
        returns the same shape.  ``S_c^+`` is never materialized as a dense
        ``T' x T'`` array beyond ``V_c``/``inv_lambda_c`` (already tiny), and
        is never broadcast up to an ``n x n`` operator.
        """
        squeeze = r.ndim == 1
        Rr = r.reshape(-1, 1) if squeeze else r
        w = np.asarray(self.Z.T @ Rr, dtype=np.float64)      # (T', k)
        w2 = self.V_c.T @ w                                   # (rank, k)
        w2 *= self.inv_lambda_c[:, None]
        y = self.V_c @ w2                                     # (rank -> T', k)
        out = np.asarray(self.Z @ y, dtype=np.float64)        # (n, k)
        return out[:, 0] if squeeze else out

    def describe(self) -> str:
        return (
            f"geneo k={self.n_geneo_cols}, T'={self.n_cols}, rank={self.rank}"
        )


def _degrade_to_pou_if_over_budget(
    Z: sp.csr_matrix,
    col_labels: List[Any],
    n_pou_cols: int,
    n_geneo_cols: int,
    T_prime: int,
    *,
    metric_fn: Callable[[int], float],
    limit: float,
    setting_name: str,
    metric_desc: str,
    fmt: Callable[[float], str],
) -> Tuple[sp.csr_matrix, List[Any], int, int, bool]:
    """Finding 12: shared GenEO-then-disable degradation ladder for BOTH
    the ``max_cols`` (column-count) and ``max_bytes`` (dense-allocation)
    guards in :func:`build_coarse_space` -- previously two near-identical
    copy-pasted blocks with four near-identical warning texts.

    ``metric_fn(T_prime)`` computes the guarded quantity from a (possibly
    already-degraded) column count -- ``lambda t: t`` for the column cap,
    ``lambda t: 3 * n * t * 8`` for the byte cap (Finding 3) -- so the same
    two-rung logic (WARNING + fall back to the PoU-only prefix of ``Z``
    when only GenEO enrichment pushed the metric over budget; WARNING +
    disable outright when even the PoU-only rung doesn't fit) is written
    once instead of twice.

    No-op (state passed through unchanged, ``disabled=False``) when the
    metric at the CURRENT ``T_prime`` already fits ``limit``.

    Sequential two-rung behaviour (matches the original per-cap blocks
    exactly, so both existing max_cols and max_bytes tests -- written
    against the pre-unification message sequence -- keep passing
    unmodified): if GenEO columns are present, ALWAYS attempt the
    fall-back-to-PoU-only rung first (WARNING) and re-check; only emit the
    second "PoU-only still exceeds" WARNING (and report ``disabled=True``)
    if the metric is STILL over budget after that truncation (or there
    were no GenEO columns to drop in the first place).

    Returns ``(Z, col_labels, n_geneo_cols, T_prime, disabled)`` --
    ``disabled=True`` tells the caller to return ``None`` from
    ``build_coarse_space`` (even the PoU-only rung exceeds ``limit``).
    """
    current = metric_fn(T_prime)
    if current <= limit:
        return Z, col_labels, n_geneo_cols, T_prime, False

    if n_geneo_cols > 0:
        # Fall back to the PoU-only rung (spec: refuse GenEO, not the whole
        # coarse space) -- Z's first n_pou_cols columns are exactly the PoU
        # prefix (GenEO columns were hstack'd on afterward), so truncating
        # here is equivalent to never having built the GenEO columns.
        logger.warning(
            "build_coarse_space: T'=%d (pou=%d, geneo=%d) exceeds "
            "interface_coarse_%s=%s (%s %s); falling back to PoU-only "
            "coarse space (dropping all %d GenEO column(s) for this "
            "solve). Reduce interface_coarse_geneo_k or raise "
            "interface_coarse_%s to re-enable GenEO enrichment.",
            T_prime, n_pou_cols, n_geneo_cols, setting_name, fmt(limit),
            metric_desc, fmt(current), n_geneo_cols, setting_name,
        )
        Z = Z[:, :n_pou_cols].tocsr()
        col_labels = col_labels[:n_pou_cols]
        n_geneo_cols = 0
        T_prime = n_pou_cols
        current = metric_fn(T_prime)

    if current > limit:
        # Even the PoU-only prefix doesn't fit -- there is no smaller rung
        # to fall back to, so disable the coarse space entirely.
        logger.warning(
            "build_coarse_space: PoU-only %s (%s) still exceeds "
            "interface_coarse_%s=%s; coarse space disabled (degrading to "
            "plain block-Jacobi). Raise interface_coarse_%s or use "
            "fewer/larger tiles.",
            metric_desc, fmt(current), setting_name, fmt(limit), setting_name,
        )
        return Z, col_labels, n_geneo_cols, T_prime, True

    return Z, col_labels, n_geneo_cols, T_prime, False


def build_coarse_space(
    matmat: Callable[[np.ndarray], np.ndarray],
    tile_index_maps: Dict[Any, np.ndarray],
    n: int,
    island_idx: Optional[np.ndarray] = None,
    geneo_pairs: Optional[List[Tuple[np.ndarray, np.ndarray, np.ndarray]]] = None,
    max_cols: int = DEFAULT_MAX_COLS,
    eps_rank: float = DEFAULT_EPS_RANK,
    max_bytes: int = DEFAULT_MAX_BYTES,
    matvec_dtype: Any = np.float64,
) -> Optional[CoarseSpace]:
    """Build the full two-level coarse space: PoU + GenEO columns, ``S_c``,
    and its eigh-based PSD pseudo-inverse factorization.

    Args:
        matmat: The SAME linear operator matmat the CG solve itself uses
            (``InterfaceCGSolver._linear_op.matmat`` -- tilewise or
            assembled, already including ``S_extra``).  Called once with a
            dense ``(n, T')`` array.
        tile_index_maps: Per-tile global interface index arrays (ownership
            source for both the PoU columns and the GenEO block scatter).
        n: Interface system size.
        island_idx: Global indices of interface-island nodes (see module
            docstring -- zeroed on every column, PoU and GenEO alike).
        geneo_pairs: ``[(global_idx, V_k, w_k), ...]`` -- one entry per
            block-Jacobi ownership block that contributed GenEO columns
            (built by ``InterfaceCGSolver._build_block_jacobi``).  ``None``
            or ``[]`` disables GenEO enrichment (PoU-only coarse space).
        max_cols: Hard cap on T' (post-PoU, post-GenEO, pre-rank-truncation)
            -- exceeding it falls back to the PoU-only prefix of ``Z``
            (WARNING, GenEO columns dropped for this solve); only when even
            the PoU-only column count exceeds ``max_cols`` is the coarse
            space disabled entirely (WARNING; caller degrades to plain
            block-Jacobi).
        eps_rank: ``S_c`` eigenvalues <= ``eps_rank * lambda_max(S_c)`` are
            treated as structural rank deficiency (e.g. the checkerboard
            null space of an even-multiplicity PoU basis) and dropped from
            the pseudo-inverse, not "small but real".  Finding 7: when
            ``matvec_dtype`` is float32, floored at
            ``FP32_COARSE_EPS_RANK_FLOOR`` (1e-6) regardless of the passed
            value -- fp32 GEMM noise in ``S_c`` sits around 1e-7-relative,
            so a tighter (fp64-scale) threshold would invert noise instead
            of dropping it.
        max_bytes: Finding 5 (round 1) / Finding 3 (round 2): byte-based
            guard on the dense ``(n, T')`` fp64 allocations this call chain
            holds concurrently -- ``Z_dense`` + ``SZ`` (``n * T' * 8`` bytes
            each) PLUS the per-thread compact-buffer union the threaded
            ``matmat`` allocates while computing ``SZ`` (~one more
            ``n * T' * 8``) -- true peak ~``3 * n * T' * 8`` bytes --
            distinct from ``max_cols`` (a column-count cap that does not
            scale with ``n``).  Exceeding it
            first drops GenEO columns (falls back to the PoU-only rung,
            re-checks), then disables the coarse space entirely if PoU-only
            alone still exceeds the budget -- same two-rung degradation as
            the ``max_cols`` cap above, WARNING at each step.
        matvec_dtype: The linear op's matvec storage dtype (Finding 7) --
            ``InterfaceCGSolver.matvec_dtype`` in 'tilewise' mode, float64
            otherwise (see the ``eps_rank`` note above).

    Returns:
        A :class:`CoarseSpace` (possibly PoU-only if the GenEO-enriched
        space overflowed ``max_cols``), or ``None`` if no coarse space can
        be built at all (logged as a WARNING in every ``None`` case).
    """
    if not tile_index_maps:
        logger.warning(
            "build_coarse_space: no tile_index_maps available; coarse "
            "space disabled (degrading to plain block-Jacobi)."
        )
        return None

    # Finding 7: vectorized equivalent of `set(range(n)) - owned_or_mapped`
    # -- the same tile-membership multiplicity build_partition_of_unity
    # computes internally (bincount over the concatenated index arrays);
    # unowned nodes are exactly those with multiplicity 0.  The prior
    # Python-object `set()` union + `int(g) for g in ...` generator was
    # O(n * multiplicity) pure-Python work -- at the large-n CG regime
    # two_level is the resolved default for (n_interface >= 200,000, see
    # auto_select_interface_solver), this ran on every prepare()/
    # prepare_transient()/tilewise refactor() before the coarse build even
    # started.
    all_idx = (
        np.concatenate([
            np.asarray(idx, dtype=np.int64) for idx in tile_index_maps.values()
        ]) if tile_index_maps else np.empty(0, dtype=np.int64)
    )
    multiplicity = np.bincount(all_idx, minlength=n) if all_idx.size else np.zeros(n, dtype=np.int64)
    unowned = np.flatnonzero(multiplicity == 0).astype(np.int64)

    Z, col_labels, n_dropped = build_partition_of_unity(
        tile_index_maps, n, island_idx=island_idx, unowned_idx=unowned,
    )
    n_pou_cols = Z.shape[1]

    # --- GenEO columns: scatter each block's eigenvectors as extra Z
    # columns (disjoint support per block -- S_c stays small/cheap). ---
    geneo_rows: List[np.ndarray] = []
    geneo_cols: List[np.ndarray] = []
    geneo_data: List[np.ndarray] = []
    n_geneo_cols = 0
    isl_mask = None
    if island_idx is not None and len(island_idx):
        isl_mask = np.zeros(n, dtype=bool)
        isl_mask[np.asarray(island_idx, dtype=np.int64)] = True

    if geneo_pairs:
        # Finding 15: 0-based column indices into a (n, n_geneo_cols)-shape
        # Z_geneo -- NOT offset by n_pou_cols into a (n, n_pou_cols +
        # n_geneo_cols)-shape matrix whose first n_pou_cols columns are
        # then immediately sliced off below.  Same result, no dead columns,
        # no offset bookkeeping to keep in sync with n_pou_cols (which
        # shifts whenever build_partition_of_unity drops an all-zero
        # column).
        for global_idx, V_k, _w_k in geneo_pairs:
            if V_k is None or V_k.shape[1] == 0:
                continue
            gidx = np.asarray(global_idx, dtype=np.int64)
            for c in range(V_k.shape[1]):
                col = V_k[:, c]
                if isl_mask is not None:
                    row_mask = isl_mask[gidx]
                    if row_mask.any():
                        col = col.copy()
                        col[row_mask] = 0.0
                        if not np.any(col):
                            continue
                geneo_rows.append(gidx)
                geneo_cols.append(
                    np.full(gidx.shape, n_geneo_cols, dtype=np.int64)
                )
                geneo_data.append(col)
                n_geneo_cols += 1
        col_labels = col_labels + [f'geneo_{i}' for i in range(n_geneo_cols)]

    if n_geneo_cols:
        rows_cat = np.concatenate(geneo_rows)
        cols_cat = np.concatenate(geneo_cols)
        data_cat = np.concatenate(geneo_data)
        Z_geneo = sp.csr_matrix(
            (data_cat, (rows_cat, cols_cat)),
            shape=(n, n_geneo_cols),
        )
        Z = sp.csr_matrix(sp.hstack([Z, Z_geneo], format='csr'))

    T_prime = Z.shape[1]
    if T_prime == 0:
        logger.warning(
            "build_coarse_space: T'=0 (no surviving PoU/GenEO columns); "
            "coarse space disabled."
        )
        return None

    # Finding 12: both the max_cols (column-count) and max_bytes
    # (dense-allocation) caps below share the SAME two-rung degradation
    # ladder (drop GenEO -> re-check -> disable outright) via
    # _degrade_to_pou_if_over_budget -- see its docstring.
    Z, col_labels, n_geneo_cols, T_prime, _disabled = _degrade_to_pou_if_over_budget(
        Z, col_labels, n_pou_cols, n_geneo_cols, T_prime,
        metric_fn=lambda t: t,
        limit=max_cols,
        setting_name='max_cols',
        metric_desc="column count T'",
        fmt=lambda v: str(int(v)),
    )
    if _disabled:
        return None

    # Finding 5 (round 1) / Finding 3 (round 2): byte-based guard on the
    # dense (n, T') fp64 allocations held concurrently by this call chain
    # (Z_dense + SZ + the threaded matmat's per-thread compact-buffer
    # union -- see DEFAULT_MAX_BYTES's comment) -- max_cols alone does not
    # scale with n, so a large-n/modest-T' system (the "never-assemble"/
    # streaming regime this feature must not regress -- see the module
    # docstring) can still blow the coordinator memory bound even when
    # comfortably under max_cols.
    Z, col_labels, n_geneo_cols, T_prime, _disabled = _degrade_to_pou_if_over_budget(
        Z, col_labels, n_pou_cols, n_geneo_cols, T_prime,
        metric_fn=lambda t: 3 * n * t * 8,
        limit=max_bytes,
        setting_name='max_bytes',
        metric_desc='dense coarse-build allocation',
        fmt=lambda v: f"{v / 1024 ** 2:.1f} MB",
    )
    if _disabled:
        return None

    z_mem_mb = n * T_prime * 8 / 1024 ** 2
    logger.info(
        "build_coarse_space: n=%d, T'=%d (pou=%d, geneo=%d, dropped=%d), "
        "Z_dense ~%.1f MB",
        n, T_prime, n_pou_cols, n_geneo_cols, n_dropped, z_mem_mb,
    )

    Z_dense = np.asarray(Z.toarray(), dtype=np.float64)
    SZ = np.asarray(matmat(Z_dense), dtype=np.float64)
    S_c = Z_dense.T @ SZ
    S_c = 0.5 * (S_c + S_c.T)

    w, V = np.linalg.eigh(S_c)
    lam_max = float(w[-1]) if w.size else 0.0
    if lam_max <= 0:
        logger.warning(
            "build_coarse_space: S_c (%dx%d) has no positive spectrum; "
            "coarse space disabled.", T_prime, T_prime,
        )
        return None

    # Finding 7: floor eps_rank at FP32_COARSE_EPS_RANK_FLOOR when the
    # matmat that produced S_c ran in float32 -- see this function's
    # docstring and the module-level constant's comment.
    effective_eps_rank = eps_rank
    if (
        np.dtype(matvec_dtype) == np.dtype(np.float32)
        and effective_eps_rank < FP32_COARSE_EPS_RANK_FLOOR
    ):
        logger.info(
            "build_coarse_space: matvec_dtype=float32 -- flooring eps_rank "
            "%.1e -> %.1e (S_c carries ~1e-7-relative fp32 GEMM noise; "
            "inverting noise-level eigenvalues would inject amplified "
            "error into the coarse correction).",
            eps_rank, FP32_COARSE_EPS_RANK_FLOOR,
        )
        effective_eps_rank = FP32_COARSE_EPS_RANK_FLOOR

    eps = effective_eps_rank * lam_max
    keep = w > eps
    rank = int(np.sum(keep))
    if rank == 0:
        logger.warning(
            "build_coarse_space: S_c (%dx%d) rank 0 after eps_rank=%.1e "
            "truncation; coarse space disabled.",
            T_prime, T_prime, effective_eps_rank,
        )
        return None

    lam_min_kept = float(w[keep][0])
    cond_estimate = lam_max / max(lam_min_kept, 1e-300)
    logger.info(
        "build_coarse_space: S_c %dx%d, rank=%d (eps_rank=%.1e), "
        "cond~%.3e (%d structurally-deficient direction(s) dropped)",
        T_prime, T_prime, rank, effective_eps_rank, cond_estimate, T_prime - rank,
    )

    V_c = np.ascontiguousarray(V[:, keep])
    inv_lambda_c = 1.0 / w[keep]

    return CoarseSpace(
        Z=Z, V_c=V_c, inv_lambda_c=inv_lambda_c,
        n_pou_cols=n_pou_cols, n_geneo_cols=n_geneo_cols,
        n_dropped_cols=n_dropped, rank=rank, cond_estimate=cond_estimate,
        col_labels=col_labels,
    )

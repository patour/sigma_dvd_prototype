"""Stage 3 tests (interface_solve_acceleration_plan.md / docs Sec 7.8): the
two-level coarse-space additive preconditioner (``interface_coarse.py``) and
its wiring into ``InterfaceCGSolver`` / ``build_interface_solver``.

Fixture note -- why a NEW synthetic generator instead of the existing
``_make_synthetic_tiles`` (test_interface_iterative_stage2.py)
--------------------------------------------------------------------------
``_make_synthetic_tiles`` builds tiles from independent random SPD blocks
with randomly-overlapping index windows; it is deliberately only used
upstream for MATVEC/threading equivalence (never for an actual CG solve).
Assembling it densely reveals why: with n=6000/30 tiles the resulting global
matrix has ~half ZERO rows (index ranges no tile's window reaches) --
singular, not just ill-conditioned -- so a real CG solve on it does not
converge to anything meaningful. It is reused here only where the tests
already deliberately avoid running CG to convergence (thread-count
equivalence).

For every test that needs an actual, meaningful CG SOLVE (the "strict
iteration reduction" and "additive M SPD" requirements), this file uses
``_chain_tiles``: a 1D resistor-chain domain decomposition (T tiles, each a
physical chain segment of ``m_interior`` resistors + weak grounding, reduced
to its 2 end-port Schur complement via real block elimination -- i.e. an
actual small DDM interface system, not a hand-waved random matrix). This is
the textbook scenario two-level Schwarz/coarse-space correction exists for:
plain block-Jacobi's iteration count grows ~linearly with the number of
subdomains (each BJ sweep only propagates information one subdomain at a
time), while a coarse space with even one degree of freedom per subdomain
(the partition-of-unity space alone, before any GenEO enrichment) restores
~constant iteration count regardless of chain length -- verified below to
be robust and NOT an artifact of a specific random seed (the chain has no
randomness at all beyond the RHS).
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import scipy.sparse as sp

pytestmark = pytest.mark.unit

sys.path.insert(0, str(Path(__file__).parent))
from test_time_domain import _build_two_tile_distributed_model  # noqa: E402
from test_interface_iterative_stage2 import _make_synthetic_tiles  # noqa: E402

from distributed import interface_coarse as ic  # noqa: E402
from distributed.interface_iterative import (  # noqa: E402
    InterfaceCGSolver,
    build_interface_solver,
    resolve_preconditioner,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _tile_schur_1d_chain(m_interior: int, r: float, g: float) -> np.ndarray:
    """Schur complement (2x2, over the two end ports) of a physical chain
    segment: port_L -R- n0 -R- n1 -...- n_{m-1} -R- port_R, each interior
    node additionally grounded through conductance ``g``.  Built via a real
    block elimination (Gpp - Gpi @ Gii^-1 @ Gip), not a hand-picked matrix.
    """
    n_local = m_interior + 2  # 0 = port_L, 1..m = interior, m+1 = port_R

    def stamp(G: np.ndarray, a: int, b: int, cond: float) -> None:
        G[a, a] += cond
        G[b, b] += cond
        G[a, b] -= cond
        G[b, a] -= cond

    G = np.zeros((n_local, n_local))
    chain = [0] + list(range(1, m_interior + 1)) + [m_interior + 1]
    for i in range(len(chain) - 1):
        stamp(G, chain[i], chain[i + 1], 1.0 / r)
    for i in range(1, m_interior + 1):
        G[i, i] += g

    port_idx = [0, n_local - 1]
    int_idx = list(range(1, n_local - 1))
    Gpp = G[np.ix_(port_idx, port_idx)]
    Gpi = G[np.ix_(port_idx, int_idx)]
    Gii = G[np.ix_(int_idx, int_idx)]
    return Gpp - Gpi @ np.linalg.solve(Gii, Gpi.T)


def _chain_tiles(n_tiles: int, m_interior: int = 15, r: float = 1.0, g: float = 1e-4):
    """T tiles in a 1D chain; port node ``t`` is shared by tiles ``t-1`` and
    ``t`` (adjacent tiles), so ownership/PoU multiplicity is exactly 2
    everywhere except the two chain ends (multiplicity 1). Returns
    ``(tile_schur, tile_idx, n)``."""
    tile_schur: dict = {}
    tile_idx: dict = {}
    for t in range(n_tiles):
        tile_schur[t] = _tile_schur_1d_chain(m_interior, r, g)
        tile_idx[t] = np.array([t, t + 1], dtype=np.int32)
    return tile_schur, tile_idx, n_tiles + 1


def _near_null_spd_block(n: int, rng: np.random.Generator,
                          small: float = 1e-9, high: float = 3.0) -> np.ndarray:
    """SPD n x n matrix with one genuine near-null eigendirection
    (eigenvalue ratio ``small / high`` ~ 3e-10, well under the default
    ``interface_coarse_geneo_tol=1e-6``).  Mirrors the Stage 2 measured
    root cause ("cho-factored ownership blocks have genuine near-null
    eigendirections (~1e-10 relative)", see this module's docstring / the
    plan doc Sec 7.8) so GenEO enrichment actually has something to pick
    up, instead of the well-conditioned ``A @ A.T / len + 3*eye`` blocks
    used elsewhere in this file (min eigenvalue O(1), nowhere near the
    tol*lambda_max threshold -- those blocks yield zero GenEO columns by
    construction and are fine for tests that don't care about GenEO).
    """
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    eigs = np.full(n, high)
    eigs[0] = small
    return (Q * eigs) @ Q.T


def _checkerboard_4tile():
    """2x2 four-tile checkerboard: nodes 0,1,2,3 arranged in a cycle, each
    touched by exactly 2 (adjacent) tiles -- the textbook partition-of-unity
    rank-deficiency fixture (spec edge case 4): the alternating-sign
    combination (+1,-1,+1,-1) across tiles maps to the EXACT zero vector
    under Z (verified in TestBuildPartitionOfUnity), so rank(Z) = T'-1 = 3
    and therefore rank(S_c) <= 3 too."""
    n = 4
    tile_idx = {
        0: np.array([0, 1], dtype=np.int32),
        1: np.array([1, 2], dtype=np.int32),
        2: np.array([2, 3], dtype=np.int32),
        3: np.array([3, 0], dtype=np.int32),
    }
    tile_schur = {
        t: np.array([[1.0, -0.4], [-0.4, 1.0]]) for t in tile_idx
    }
    S = np.zeros((n, n))
    for tid, idx in tile_idx.items():
        S[np.ix_(idx, idx)] += tile_schur[tid]
    return tile_schur, tile_idx, n, S


# ---------------------------------------------------------------------------
# 1. build_partition_of_unity
# ---------------------------------------------------------------------------


class TestBuildPartitionOfUnity:

    def test_basic_weights_and_multiplicity(self):
        tile_index_maps = {
            'A': np.array([0, 1, 2], dtype=np.int32),
            'B': np.array([2, 3, 4], dtype=np.int32),
        }
        n = 5
        Z, labels, n_dropped = ic.build_partition_of_unity(tile_index_maps, n)
        assert n_dropped == 0
        assert labels == ['A', 'B']
        Zd = Z.toarray()
        # node 2 has multiplicity 2 -> weight 1/2 in both columns.
        assert Zd[2, 0] == pytest.approx(0.5)
        assert Zd[2, 1] == pytest.approx(0.5)
        # node 0 (only tile A) has weight 1 in column A, 0 elsewhere.
        assert Zd[0, 0] == pytest.approx(1.0)
        assert Zd[0, 1] == 0.0

    def test_island_rows_zeroed(self):
        """Edge case 1: island-penalized rows are zeroed in every column."""
        tile_index_maps = {
            'A': np.array([0, 1, 2], dtype=np.int32),
            'B': np.array([2, 3, 4], dtype=np.int32),
        }
        n = 5
        island_idx = np.array([2], dtype=np.int64)
        Z, _, _ = ic.build_partition_of_unity(
            tile_index_maps, n, island_idx=island_idx,
        )
        Zd = Z.toarray()
        assert np.all(Zd[2, :] == 0.0)
        # non-island rows unaffected.
        assert Zd[0, 0] == pytest.approx(1.0)
        assert Zd[4, 1] == pytest.approx(1.0)

    def test_unowned_indicator_column(self):
        """Edge case 2: an index in NO tile's map gets a dedicated
        indicator column (weight 1), so package/die/tap unknowns remain
        visible to the coarse space."""
        tile_index_maps = {
            'A': np.array([0, 1, 2], dtype=np.int32),
            'B': np.array([3, 4], dtype=np.int32),
        }
        n = 6  # node 5 is unowned (e.g. a tap node)
        unowned_idx = np.array([5], dtype=np.int64)
        Z, labels, _ = ic.build_partition_of_unity(
            tile_index_maps, n, unowned_idx=unowned_idx,
        )
        assert labels[-1] == '__unowned__'
        Zd = Z.toarray()
        assert Zd[5, -1] == pytest.approx(1.0)
        assert np.all(Zd[5, :-1] == 0.0)
        assert np.all(Zd[:5, -1] == 0.0)

    def test_all_zero_column_dropped_and_logged(self, caplog):
        """Edge case 3: a tile whose entire owned range is island-zeroed
        contributes an all-zero column, which must be dropped (and logged),
        not passed through as a dead coarse-space dimension."""
        tile_index_maps = {
            'A': np.array([0, 1], dtype=np.int32),
            'B': np.array([2, 3], dtype=np.int32),
        }
        n = 4
        island_idx = np.array([0, 1], dtype=np.int64)  # ALL of tile A
        with caplog.at_level(logging.INFO):
            Z, labels, n_dropped = ic.build_partition_of_unity(
                tile_index_maps, n, island_idx=island_idx,
            )
        assert n_dropped == 1
        assert labels == ['B']
        assert Z.shape == (4, 1)
        assert any('dropping' in rec.message for rec in caplog.records)

    def test_empty_inputs_return_zero_columns(self):
        Z, labels, n_dropped = ic.build_partition_of_unity({}, 5)
        assert Z.shape == (5, 0)
        assert labels == []
        assert n_dropped == 0

    def test_checkerboard_alternating_combination_is_null(self):
        """The algebraic fact behind edge case 4: on the 2x2 checkerboard,
        Z @ (+1,-1,+1,-1) is EXACTLY zero (not just S_c singular -- Z itself
        is rank-deficient)."""
        _, tile_idx, n, _ = _checkerboard_4tile()
        Z, _, _ = ic.build_partition_of_unity(tile_idx, n)
        assert np.linalg.matrix_rank(Z.toarray()) == 3
        c = np.array([1.0, -1.0, 1.0, -1.0])
        np.testing.assert_allclose(Z @ c, np.zeros(n), atol=1e-14)


# ---------------------------------------------------------------------------
# 2. geneo_lowest_eigenpairs
# ---------------------------------------------------------------------------


class TestGeneoLowestEigenpairs:

    def test_small_block_eigh_path_selects_lowest(self):
        rng = np.random.default_rng(0)
        V, _ = np.linalg.qr(rng.standard_normal((6, 6)))
        w_true = np.array([1e-8, 1e-7, 0.5, 1.0, 2.0, 10.0])
        sub = (V * w_true) @ V.T
        sub = 0.5 * (sub + sub.T)
        V_k, w_k = ic.geneo_lowest_eigenpairs(sub, k=4, tol=1e-5)
        # tol=1e-5 * lambda_max(10.0) = 1e-4 -- only the two ~1e-8/1e-7
        # eigenvalues qualify.
        assert V_k.shape == (6, 2)
        np.testing.assert_allclose(sorted(w_k), [1e-8, 1e-7], atol=1e-12)

    def test_k_zero_or_n_zero_returns_empty(self):
        sub = np.eye(4)
        V_k, w_k = ic.geneo_lowest_eigenpairs(sub, k=0)
        assert V_k.shape == (4, 0)
        assert w_k.shape == (0,)
        V_k2, w_k2 = ic.geneo_lowest_eigenpairs(np.zeros((0, 0)), k=4)
        assert V_k2.shape == (0, 0)

    def test_no_eigenvalue_passes_tol_returns_empty(self):
        rng = np.random.default_rng(1)
        A = rng.standard_normal((5, 5))
        sub = A @ A.T + 5 * np.eye(5)  # well-conditioned, no near-null modes
        V_k, w_k = ic.geneo_lowest_eigenpairs(sub, k=4, tol=1e-10)
        assert V_k.shape[1] == 0

    def test_large_block_eigsh_shift_invert_matches_dense_eigh(self):
        """Blocks >= small_block_threshold use ARPACK shift-invert reusing
        an EXISTING cho factor; verify it agrees with the dense reference."""
        import scipy.linalg as la

        n = 40
        rng = np.random.default_rng(2)
        V, _ = np.linalg.qr(rng.standard_normal((n, n)))
        w_true = np.concatenate([[1e-9, 5e-9], rng.uniform(1, 10, n - 2)])
        sub = (V * w_true) @ V.T
        sub = 0.5 * (sub + sub.T)
        cho = la.cho_factor(sub, check_finite=False)

        V_k, w_k = ic.geneo_lowest_eigenpairs(
            sub, cho=cho, k=3, tol=1e-6, small_block_threshold=10,
        )
        V_ref, w_ref = ic.geneo_lowest_eigenpairs(sub, k=3, tol=1e-6)
        assert V_k.shape[1] == V_ref.shape[1] == 2
        np.testing.assert_allclose(
            sorted(w_k), sorted(w_ref), rtol=1e-6, atol=1e-12,
        )

    def test_precomputed_spectrum_reused_no_recompute(self):
        """GenEO reuse path (edge case 7): passing precomputed=(w, V) must
        not call np.linalg.eigh again."""
        rng = np.random.default_rng(3)
        A = rng.standard_normal((5, 5))
        sub = 0.5 * (A + A.T)
        w, V = np.linalg.eigh(sub)
        with mock.patch('numpy.linalg.eigh') as mocked_eigh:
            V_k, w_k = ic.geneo_lowest_eigenpairs(
                sub, k=2, tol=1.0, precomputed=(w, V),
            )
            mocked_eigh.assert_not_called()
        assert V_k.shape[1] == 2  # tol=1.0 keeps everything <= lambda_max

    def test_arpack_partial_convergence_thresholds_against_true_lambda_max(
        self, monkeypatch,
    ):
        """Finding 1 regression: the ArpackNoConvergence partial-result
        branch must threshold against the block's TRUE lambda_max (the k=1
        LM eigsh call already computed and in scope) -- NOT
        max(w_partial) (the largest of the partial LOWEST eigenvalues,
        tiny by construction).  Simulates ARPACK hitting maxiter on the
        shift-invert (lowest-eigenpair) call with a genuine near-null
        partial result, while the preceding k=1 LM lambda_max estimate
        itself succeeds normally and is NOT near-null.

        Pre-fix: thresh = tol * max(w_partial) = 1e-6 * 5e-9 = 5e-15,
        which every one of the (~1e-9-scale) partial eigenpairs fails,
        silently discarding GenEO enrichment on exactly the near-null
        block the feature exists to enrich.
        """
        import scipy.linalg as la

        n = 4
        sub = np.diag([100.0, 50.0, 20.0, 10.0]).astype(np.float64)
        cho = la.cho_factor(sub, check_finite=False)

        true_lam_max = 100.0
        partial_w = np.array([1e-9, 2e-9, 5e-9])
        partial_V = np.eye(n)[:, :3]
        calls = {'n': 0}

        def fake_eigsh(A, k=None, which=None, sigma=None, OPinv=None):
            calls['n'] += 1
            if calls['n'] == 1:
                # The k=1 LM lambda_max estimate -- succeeds normally.
                return np.array([true_lam_max]), np.zeros((n, 1))
            # The shift-invert lowest-eigenpair call -- simulate ARPACK
            # hitting maxiter with a genuine near-null PARTIAL result.
            raise ic.spla.ArpackNoConvergence(
                "ARPACK did not converge", partial_w, partial_V,
            )

        monkeypatch.setattr(ic.spla, 'eigsh', fake_eigsh)

        V_k, w_k = ic.geneo_lowest_eigenpairs(
            sub, cho=cho, k=3, tol=1e-6, small_block_threshold=1,
        )
        assert w_k.shape[0] == 3, (
            f"expected all 3 near-null partial eigenpairs to be kept "
            f"against the TRUE lambda_max={true_lam_max} threshold "
            f"(tol*lambda_max=1e-4 -- all partial eigenvalues <= 5e-9 "
            f"qualify); got {w_k.shape[0]} kept (old/buggy code thresholds "
            f"against max(w_partial)=5e-9, giving thresh=5e-15, which "
            f"rejects every partial eigenpair)"
        )
        np.testing.assert_allclose(sorted(w_k), sorted(partial_w))

    def test_island_local_mask_restricts_to_physical_spectrum(self):
        """Finding 3 regression: a block-local island penalty (~1e5 mS
        diagonal) must not inflate the GenEO near-null threshold -- the
        eigen-analysis (both the lambda_max estimate and the k-lowest
        selection) must run on the RESTRICTED (non-island) principal
        submatrix, not the full penalty-inflated block.

        Without ``island_local_mask``, the block's own eigh gives
        lambda_max=1e5 (dominated by the isolated penalty entry), so
        thresh = tol*1e5 = 0.1 lets the 0.01/0.05 "healthy" (not
        near-null) eigenvalues through as spurious GenEO columns. With the
        mask, lambda_max is the PHYSICAL 5.0, thresh = tol*5.0 = 5e-6, and
        only the genuine near-null 1e-9 eigenpair qualifies.
        """
        rng = np.random.default_rng(31)
        Q, _ = np.linalg.qr(rng.standard_normal((5, 5)))
        eigs = np.array([1e-9, 0.01, 0.05, 2.0, 5.0])
        sub5 = (Q * eigs) @ Q.T
        sub5 = 0.5 * (sub5 + sub5.T)

        sub = np.zeros((6, 6))
        sub[:5, :5] = sub5
        sub[5, 5] = 1e5  # island penalty diagonal, no off-diag coupling --
                          # matches apply_island_penalty's diagonal-only stamp.
        island_local_mask = np.array(
            [False, False, False, False, False, True],
        )

        V_k, w_k = ic.geneo_lowest_eigenpairs(
            sub, k=4, tol=1e-6, island_local_mask=island_local_mask,
        )
        assert w_k.shape[0] == 1, (
            f"expected only the genuine near-null eigenpair (1e-9) against "
            f"the PHYSICAL lambda_max=5.0 (thresh=5e-6); got "
            f"{w_k.shape[0]} kept"
        )
        np.testing.assert_allclose(w_k[0], 1e-9, atol=1e-12, rtol=1e-3)
        # Island row is exactly zero in the returned (scattered-back)
        # eigenvector.
        assert np.all(V_k[5, :] == 0.0)

        # Sanity/contrast: WITHOUT the mask (the pre-fix call-site
        # behaviour), the island-inflated lambda_max=1e5 lets the
        # 0.01/0.05 "healthy" eigenvalues through as spurious near-null
        # columns too.
        V_bad, w_bad = ic.geneo_lowest_eigenpairs(sub, k=4, tol=1e-6)
        assert w_bad.shape[0] > 1, (
            "expected the unmasked call to demonstrate the spurious-column "
            "behaviour the island_local_mask fixes at the call site"
        )

    def test_island_local_mask_all_island_returns_empty(self):
        mask = np.array([True, True, True])
        sub = np.diag([1e5, 1e5, 1e5]).astype(np.float64)
        V_k, w_k = ic.geneo_lowest_eigenpairs(sub, k=4, tol=1e-6, island_local_mask=mask)
        assert V_k.shape == (3, 0)
        assert w_k.shape == (0,)

    def test_island_local_mask_large_block_uses_eigsh_not_eigh(self):
        """Finding 4 (round 2) regression: a large (>= small_block_
        threshold) island-touching block must take the ARPACK shift-invert
        path on its RESTRICTED (non-island) submatrix -- cho-factored
        fresh there (cheap O(k^3/3), see the module docstring's Finding 4
        note) -- not unconditionally recurse into a full O(k^3) dense
        ``eigh``. The round-1 island_local_mask fix always passed
        ``cho=None`` into the recursive call, which forced dense eigh
        regardless of block size -- the exact O(n^3) regression this
        finding fixes. Verified by spying: ``np.linalg.eigh`` must NOT be
        called and ``scipy.sparse.linalg.eigsh`` MUST be called, and the
        result must match dense eigh computed directly on the same
        restricted submatrix (correctness, not just "took a fast path").
        """
        n_phys = 40
        rng = np.random.default_rng(41)
        V, _ = np.linalg.qr(rng.standard_normal((n_phys, n_phys)))
        w_true = np.concatenate([[1e-9, 5e-9], rng.uniform(1, 10, n_phys - 2)])
        sub_phys = (V * w_true) @ V.T
        sub_phys = 0.5 * (sub_phys + sub_phys.T)

        n = n_phys + 1
        sub = np.zeros((n, n))
        sub[:n_phys, :n_phys] = sub_phys
        sub[n_phys, n_phys] = 1e5  # island penalty diagonal, no off-diag
        island_local_mask = np.zeros(n, dtype=bool)
        island_local_mask[n_phys] = True

        # Dense-eigh reference: geneo_lowest_eigenpairs on the bare
        # restricted submatrix, well under its own small_block_threshold
        # default (500) so IT takes the dense path -- the ground truth the
        # masked/large-block call above must reproduce.
        V_ref, w_ref = ic.geneo_lowest_eigenpairs(sub_phys, k=3, tol=1e-6)

        _real_eigh = np.linalg.eigh
        _real_eigsh = ic.spla.eigsh
        with mock.patch(
            'numpy.linalg.eigh', wraps=_real_eigh,
        ) as mocked_eigh, mock.patch.object(
            ic.spla, 'eigsh', wraps=_real_eigsh,
        ) as mocked_eigsh:
            V_k, w_k = ic.geneo_lowest_eigenpairs(
                sub, k=3, tol=1e-6, small_block_threshold=10,
                island_local_mask=island_local_mask,
            )
            mocked_eigh.assert_not_called()
            assert mocked_eigsh.call_count >= 1, (
                "expected the large island-touching block to take the "
                "ARPACK shift-invert path (on its restricted, freshly "
                "cho-factored submatrix), not dense eigh"
            )

        assert w_k.shape[0] == w_ref.shape[0] == 2
        np.testing.assert_allclose(
            sorted(w_k), sorted(w_ref), rtol=1e-6, atol=1e-12,
        )
        # Island row is exactly zero in the returned (scattered-back)
        # eigenvector.
        assert np.all(V_k[n_phys, :] == 0.0)


# ---------------------------------------------------------------------------
# 3. build_coarse_space / CoarseSpace
# ---------------------------------------------------------------------------


class TestBuildCoarseSpace:

    def test_no_tile_index_maps_returns_none(self, caplog):
        with caplog.at_level(logging.WARNING):
            result = ic.build_coarse_space(lambda X: X, {}, n=5)
        assert result is None
        assert any('no tile_index_maps' in r.message for r in caplog.records)

    def test_max_cols_cap_disables_coarse_space(self, caplog):
        """max_cols=5 here is below even the PoU-only column count (20
        tiles -> n_pou_cols=20), so there is no smaller rung to fall back
        to and the coarse space is disabled entirely."""
        tile_schur, tile_idx, n = _chain_tiles(20)
        S = np.zeros((n, n))
        for tid, idx in tile_idx.items():
            S[np.ix_(idx, idx)] += tile_schur[tid]
        with caplog.at_level(logging.WARNING):
            result = ic.build_coarse_space(
                lambda X: S @ X, tile_idx, n=n, max_cols=5,
            )
        assert result is None
        assert any("exceeds" in r.message for r in caplog.records)

    def test_max_cols_cap_falls_back_to_pou_only_when_geneo_overflows(self, caplog):
        """Regression (spec-compliance finding): when GenEO enrichment
        pushes T' = pou + geneo over ``max_cols`` but the PoU-only column
        count alone still fits, ``build_coarse_space`` must fall back to
        the PoU-only rung (spec: 'refuse (fall back to PoU-only, WARNING)')
        instead of disabling the coarse space outright. Before the fix this
        returned ``None`` whenever T' > max_cols regardless of whether the
        PoU-only prefix alone would have fit."""
        tile_schur, tile_idx, n = _chain_tiles(5)
        n_pou_cols = 5  # one column per tile; _chain_tiles has no unowned nodes
        S = np.zeros((n, n))
        for tid, idx in tile_idx.items():
            S[np.ix_(idx, idx)] += tile_schur[tid]

        # Fabricate GenEO pairs (disjoint per-tile support, as the real
        # scatter does) that alone would push T' well past a cap that still
        # comfortably fits the PoU-only prefix.
        rng = np.random.default_rng(0)
        geneo_pairs = []
        for tid, idx in tile_idx.items():
            V = rng.standard_normal((len(idx), 3))
            w = np.array([1e-8, 1e-7, 1e-6])
            geneo_pairs.append((idx, V, w))
        # Full T' = 5 (pou) + 5*3 (geneo) = 20; cap at exactly n_pou_cols so
        # PoU-only fits (5 <= 5) but PoU+GenEO does not (20 > 5).
        with caplog.at_level(logging.WARNING):
            coarse = ic.build_coarse_space(
                lambda X: S @ X, tile_idx, n=n, geneo_pairs=geneo_pairs,
                max_cols=n_pou_cols,
            )
        assert coarse is not None, (
            "must fall back to PoU-only, not disable the coarse space "
            "outright, when the PoU-only prefix alone fits max_cols"
        )
        assert coarse.n_geneo_cols == 0
        assert coarse.n_cols == n_pou_cols
        assert any(
            'falling back to pou-only' in r.message.lower()
            for r in caplog.records
        )

        # The surviving Z must be EXACTLY the PoU-only basis (GenEO columns
        # dropped, not just zeroed/hidden), and the resulting coarse
        # correction must match a from-scratch PoU-only build bit-for-bit.
        Z_pou_only, _labels, _dropped = ic.build_partition_of_unity(
            tile_idx, n,
        )
        np.testing.assert_array_equal(
            coarse.Z.toarray(), Z_pou_only.toarray(),
        )

    def test_zT_S_Z_equals_Sc_two_tile_fixture(self):
        """Mathematical invariant: Z^T S Z == S_c (incl. S_extra) on the
        standard two-tile fixture, tolerance ~1e-10 relative."""
        from distributed.solver import DistributedDDMSolver

        model = _build_two_tile_distributed_model(package_cap_edges=[])
        model.settings.update({
            'interface_solver': 'cg',
            'interface_matvec_mode': 'tilewise',
            'interface_preconditioner': 'two_level',
            'interface_cg_rtol': 1e-12,
            'interface_cg_atol': 1e-16,
        })
        solver = DistributedDDMSolver(model)
        ctx = solver.prepare()
        try:
            cg = ctx._cg_solver
            assert cg is not None
            coarse = cg._coarse
            assert coarse is not None

            Z_dense = coarse.Z.toarray()
            SZ = cg._linear_op.matmat(Z_dense)
            Sc_direct = Z_dense.T @ SZ
            Sc_direct = 0.5 * (Sc_direct + Sc_direct.T)

            # Reconstruct S_c from the stored eigenfactorization on the
            # KEPT (rank-truncated) subspace only -- compare on that
            # subspace, which is what the pseudo-inverse actually uses.
            Sc_from_factor = (
                coarse.V_c @ np.diag(1.0 / coarse.inv_lambda_c) @ coarse.V_c.T
            )
            proj = coarse.V_c @ coarse.V_c.T
            Sc_direct_proj = proj @ Sc_direct @ proj
            rel = (
                np.max(np.abs(Sc_from_factor - Sc_direct_proj))
                / max(1e-300, np.max(np.abs(Sc_direct_proj)))
            )
            assert rel <= 1e-10, f"Z^T S Z vs reconstructed S_c rel diff {rel:.3e}"

            # Spec cross-check: the tilewise-derived S_c (via
            # cg._linear_op.matmat, i.e. sum_i P_i^T S_i P_i + S_extra --
            # the SAME per-tile Schur blocks build_coarse_space consumed)
            # must also agree with Z^T S_global Z, where S_global is the
            # INDEPENDENTLY assembled global Schur complement built by
            # assemble_schur_complement_system() in result_factorization.py
            # (a wholly separate code path: direct sparse assembly from
            # tile port maps + package edges + island penalties, not a
            # sum-of-tilewise-matvecs). Comparing Sc_direct only against
            # its own eigen-reconstruction (above) is a self-consistency
            # check; this closes the gap by cross-validating against the
            # assembled operator, per the spec's literal requirement.
            S_global = ctx._S_global
            assert S_global is not None, (
                "fixture is expected to assemble S_global on the default "
                "(non streaming, non never-assemble) DC path -- required "
                "for this cross-check to be meaningful"
            )
            Sc_from_assembled = Z_dense.T @ (S_global @ Z_dense)
            Sc_from_assembled = 0.5 * (Sc_from_assembled + Sc_from_assembled.T)
            rel_assembled = (
                np.max(np.abs(Sc_direct - Sc_from_assembled))
                / max(1e-300, np.max(np.abs(Sc_from_assembled)))
            )
            assert rel_assembled <= 1e-10, (
                "tilewise Z^T S Z vs assembled-S_global Z^T S Z cross-check "
                f"rel diff {rel_assembled:.3e}"
            )
        finally:
            ctx.release()
            model.shutdown()

    def test_checkerboard_rank_deficiency_handled(self, caplog):
        """Edge case 4: rank(S_c) = T'-1 on the 2x2 checkerboard; eigh
        pseudo-inverse handles it (logged), CG converges, M stays SPD."""
        tile_schur, tile_idx, n, S = _checkerboard_4tile()
        with caplog.at_level(logging.INFO):
            coarse = ic.build_coarse_space(lambda X: S @ X, tile_idx, n=n)
        assert coarse is not None
        assert coarse.n_cols == 4
        assert coarse.rank == 3
        assert any('rank=3' in r.message for r in caplog.records)

        # Probe: M (coarse-only here) stays PSD on random vectors.
        rng = np.random.default_rng(9)
        for _ in range(20):
            x = rng.standard_normal(n)
            y = coarse.apply(x)
            assert x @ y >= -1e-12

    def test_apply_matches_dense_pseudo_inverse(self):
        tile_schur, tile_idx, n = _chain_tiles(6)
        S = np.zeros((n, n))
        for tid, idx in tile_idx.items():
            S[np.ix_(idx, idx)] += tile_schur[tid]
        coarse = ic.build_coarse_space(lambda X: S @ X, tile_idx, n=n)
        assert coarse is not None

        Zd = coarse.Z.toarray()
        Sc = Zd.T @ S @ Zd
        Sc = 0.5 * (Sc + Sc.T)
        Sc_pinv = np.linalg.pinv(Sc, rcond=1e-10)

        rng = np.random.default_rng(11)
        x = rng.standard_normal(n)
        y_ref = Zd @ (Sc_pinv @ (Zd.T @ x))
        y = coarse.apply(x)
        np.testing.assert_allclose(y, y_ref, rtol=1e-8, atol=1e-10)

    # -----------------------------------------------------------------
    # Finding 5: byte-based dense-allocation guard
    # -----------------------------------------------------------------

    def test_max_bytes_tiny_budget_drops_geneo_then_disables(self, caplog):
        """Finding 5 (round 1) / Finding 3 (round 2): a byte-based budget
        (distinct from max_cols) guards the dense (n, T') fp64 allocations
        held concurrently (Z_dense + SZ + the threaded matmat's per-thread
        compact-buffer union -- true peak ~3*n*T'*8). A tiny budget that
        fits neither PoU+GenEO nor PoU-only must first drop GenEO
        (WARNING), then disable the coarse space entirely (WARNING)."""
        tile_schur, tile_idx, n = _chain_tiles(20)
        S = np.zeros((n, n))
        for tid, idx in tile_idx.items():
            S[np.ix_(idx, idx)] += tile_schur[tid]
        rng = np.random.default_rng(0)
        geneo_pairs = []
        for tid, idx in tile_idx.items():
            V = rng.standard_normal((len(idx), 2))
            w = np.array([1e-8, 1e-7])
            geneo_pairs.append((idx, V, w))

        # n=21, T'(pou+geneo) = 21 + 20*2 = 61 -> 3*21*61*8 ~ 30.8 KB.
        # A budget of a few hundred bytes fits NEITHER rung.
        with caplog.at_level(logging.WARNING):
            result = ic.build_coarse_space(
                lambda X: S @ X, tile_idx, n=n, geneo_pairs=geneo_pairs,
                max_bytes=200,
            )
        assert result is None
        messages = [r.message.lower() for r in caplog.records]
        assert any('dropping' in m and 'geneo' in m for m in messages)
        assert any('pou-only' in m and 'still exceeds' in m for m in messages)

    def test_max_bytes_budget_fits_pou_only_falls_back(self, caplog):
        """A budget too small for PoU+GenEO but large enough for the
        PoU-only rung must fall back (not disable outright)."""
        tile_schur, tile_idx, n = _chain_tiles(20)
        n_pou_cols = 20
        S = np.zeros((n, n))
        for tid, idx in tile_idx.items():
            S[np.ix_(idx, idx)] += tile_schur[tid]
        rng = np.random.default_rng(1)
        geneo_pairs = []
        for tid, idx in tile_idx.items():
            V = rng.standard_normal((len(idx), 2))
            w = np.array([1e-8, 1e-7])
            geneo_pairs.append((idx, V, w))

        # Finding 3 (round 2): true peak is 3*n*T'*8 (Z_dense + SZ + the
        # threaded matmat's per-thread compact-buffer union), not 2*n*T'*8.
        pou_only_bytes = 3 * n * n_pou_cols * 8
        with caplog.at_level(logging.WARNING):
            coarse = ic.build_coarse_space(
                lambda X: S @ X, tile_idx, n=n, geneo_pairs=geneo_pairs,
                max_bytes=int(pou_only_bytes * 1.5),
            )
        assert coarse is not None
        assert coarse.n_geneo_cols == 0
        assert coarse.n_cols == n_pou_cols
        assert any(
            'dropping' in r.message.lower() and 'geneo' in r.message.lower()
            for r in caplog.records
        )

    def test_max_bytes_default_generous_enough_for_small_fixtures(self):
        """The default DEFAULT_MAX_BYTES (8 GB) must not interfere with
        the small fixtures used throughout this test module."""
        tile_schur, tile_idx, n = _chain_tiles(6)
        S = np.zeros((n, n))
        for tid, idx in tile_idx.items():
            S[np.ix_(idx, idx)] += tile_schur[tid]
        coarse = ic.build_coarse_space(lambda X: S @ X, tile_idx, n=n)
        assert coarse is not None

    # -----------------------------------------------------------------
    # Finding 7: fp32 matvec_dtype floors the eps_rank truncation
    # -----------------------------------------------------------------

    def test_fp32_matvec_dtype_floors_eps_rank(self, caplog):
        """Finding 7 regression: when the linear op's matvec dtype is
        float32, the S_c rank-truncation eps must be floored at
        FP32_COARSE_EPS_RANK_FLOOR (1e-6) regardless of the (fp64-scale)
        eps_rank passed in -- S_c entries carry ~1e-7-relative fp32 GEMM
        noise, so a tighter threshold would invert noise as if it were a
        genuine near-null direction.

        Uses a trivial Z=identity fixture (one singleton tile per node, no
        islands/unowned) so S_c == S exactly and its eigenvalues are fully
        controlled: [1e-9, 1e-7, 0.1, 1.0]. With the fp64 default
        (eps_rank=1e-12), eps=1e-12*1.0=1e-12 keeps all 4. With
        matvec_dtype=float32, the floored eps=1e-6*1.0=1e-6 drops the
        1e-9 AND 1e-7 eigenvalues, keeping only 2.
        """
        n = 4
        tile_idx = {i: np.array([i], dtype=np.int32) for i in range(n)}
        S = np.diag([1e-9, 1e-7, 0.1, 1.0]).astype(np.float64)

        coarse_fp64 = ic.build_coarse_space(
            lambda X: S @ X, tile_idx, n=n, matvec_dtype=np.float64,
        )
        assert coarse_fp64 is not None
        assert coarse_fp64.rank == 4, (
            "fp64 (default eps_rank=1e-12) must keep ALL 4 eigenvalues -- "
            "fp64 behaviour must stay identical to pre-Finding-7 code"
        )

        with caplog.at_level(logging.INFO):
            coarse_fp32 = ic.build_coarse_space(
                lambda X: S @ X, tile_idx, n=n, matvec_dtype=np.float32,
            )
        assert coarse_fp32 is not None
        assert coarse_fp32.rank == 2, (
            f"fp32 matvec_dtype must floor eps_rank at "
            f"FP32_COARSE_EPS_RANK_FLOOR=1e-6, dropping the two "
            f"noise-level (1e-9, 1e-7) eigenvalues; got "
            f"rank={coarse_fp32.rank}"
        )
        assert any('flooring eps_rank' in r.message for r in caplog.records)

    def test_fp32_flag_is_dtype_object_not_just_string(self):
        """matvec_dtype accepts a numpy dtype object (as InterfaceCGSolver
        passes it), not only a string."""
        n = 3
        tile_idx = {i: np.array([i], dtype=np.int32) for i in range(n)}
        S = np.diag([1e-9, 0.5, 1.0]).astype(np.float64)
        coarse = ic.build_coarse_space(
            lambda X: S @ X, tile_idx, n=n,
            matvec_dtype=np.dtype(np.float32),
        )
        assert coarse is not None
        assert coarse.rank == 2


# ---------------------------------------------------------------------------
# 4. resolve_preconditioner
# ---------------------------------------------------------------------------


class TestResolvePreconditioner:

    def test_auto_cg_tilewise_resolves_two_level(self):
        assert resolve_preconditioner('auto', 'cg', 'tilewise') == 'two_level'
        assert resolve_preconditioner(None, 'cg', 'tilewise') == 'two_level'

    def test_auto_cg_assembled_resolves_block_jacobi(self):
        assert resolve_preconditioner('auto', 'cg', 'assembled') == 'block_jacobi'

    def test_explicit_value_passthrough(self):
        for val in ('block_jacobi', 'jacobi', 'none', 'amg', 'two_level'):
            assert resolve_preconditioner(val, 'cg', 'tilewise') == val
            assert resolve_preconditioner(val, 'cg', 'assembled') == val

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            resolve_preconditioner('bogus', 'cg', 'tilewise')

    def test_explicit_values_normalized_strip_lower(self):
        """Finding 12 regression: the pre-fix code normalized (strip+lower)
        ONLY the 'auto' sentinel, so an equivalently-sloppy explicit value
        (e.g. a trailing space from a quoted YAML scalar, or mismatched
        case) raised ValueError instead of resolving -- an inconsistent-
        coercion trap, since ' AUTO ' WAS silently accepted. All string
        values must be normalized the same way."""
        assert resolve_preconditioner('Two_Level ', 'cg', 'tilewise') == 'two_level'
        assert resolve_preconditioner(' BLOCK_JACOBI', 'cg', 'tilewise') == 'block_jacobi'
        assert resolve_preconditioner('  auto  ', 'cg', 'tilewise') == 'two_level'
        assert resolve_preconditioner('NONE', 'cg', 'tilewise') == 'none'
        # Still rejects genuine garbage after normalization.
        with pytest.raises(ValueError):
            resolve_preconditioner(' Bogus ', 'cg', 'tilewise')


# ---------------------------------------------------------------------------
# 5. InterfaceCGSolver('two_level') integration
# ---------------------------------------------------------------------------


class TestTwoLevelPreconditioner:

    @pytest.mark.parametrize('n_tiles', [2, 30])
    def test_strict_iteration_reduction_vs_block_jacobi(self, n_tiles):
        """Mathematical invariant: two_level < block_jacobi CG iterations
        on the same solve, same converged solution (rtol-consistent).

        At n_tiles=2 (spec's literal "two-tile" case) plain block-Jacobi is
        already close to exact for a 1D chain (2-subdomain DDM is not yet
        in the O(T)-growth regime), so equality (<=) is accepted there;
        n_tiles=30 is the load-bearing STRICT (<) assertion, matching the
        spec's "randomized ~30-tile synthetic" requirement -- see this
        file's module docstring for why a physically-real chain fixture is
        used here instead of ``_make_synthetic_tiles``.
        """
        tile_schur, tile_idx, n = _chain_tiles(n_tiles)
        rng = np.random.default_rng(555)
        b = rng.standard_normal(n)

        S = np.zeros((n, n))
        for tid, idx in tile_idx.items():
            S[np.ix_(idx, idx)] += tile_schur[tid]
        x_direct = np.linalg.solve(S, b)

        iters = {}
        for precond in ('block_jacobi', 'two_level'):
            solver = InterfaceCGSolver(
                n_interface=n, matvec_mode='tilewise',
                tile_schur_complements={k: v.copy() for k, v in tile_schur.items()},
                tile_index_maps=tile_idx,
                preconditioner=precond, rtol=1e-10, atol=1e-16,
                matvec_threads=1, strict=True, maxiter=5000,
                interface_coarse_geneo_k=4,
            )
            try:
                x = solver(b)
                iters[precond] = solver.stats['last_cg_iters']
                np.testing.assert_allclose(x, x_direct, rtol=1e-6, atol=1e-8)
            finally:
                solver.close()

        assert iters['two_level'] <= iters['block_jacobi'], iters
        if n_tiles >= 30:
            assert iters['two_level'] < iters['block_jacobi'], iters

    def test_iteration_count_grows_for_block_jacobi_stays_flat_for_two_level(self):
        """The O(T) vs O(1) domain-decomposition signature: as the chain
        lengthens, block_jacobi's iteration count grows with n_tiles while
        two_level's stays essentially flat."""
        results = {}
        for n_tiles in (15, 60, 150):
            tile_schur, tile_idx, n = _chain_tiles(n_tiles)
            rng = np.random.default_rng(42)
            b = rng.standard_normal(n)
            row = {}
            for precond in ('block_jacobi', 'two_level'):
                solver = InterfaceCGSolver(
                    n_interface=n, matvec_mode='tilewise',
                    tile_schur_complements={k: v.copy() for k, v in tile_schur.items()},
                    tile_index_maps=tile_idx,
                    preconditioner=precond, rtol=1e-10, atol=1e-16,
                    matvec_threads=1, strict=True, maxiter=5000,
                )
                try:
                    solver(b)
                    row[precond] = solver.stats['last_cg_iters']
                finally:
                    solver.close()
            results[n_tiles] = row

        bj_growth = results[150]['block_jacobi'] - results[15]['block_jacobi']
        tl_growth = results[150]['two_level'] - results[15]['two_level']
        assert bj_growth > 5 * max(tl_growth, 1), results

    def test_additive_M_spd_random_vectors(self):
        """Additive M SPD: random-vector x.Mx > 0 with islands + unowned +
        GenEO all active simultaneously.

        Regression note: the well-conditioned ``A @ A.T / len + 3*eye``
        blocks used elsewhere in this file have min eigenvalue O(1), far
        above ``tol * lambda_max`` (default tol=1e-6) -- with ONLY those
        blocks this test silently exercises GenEO in name only
        (``n_geneo_cols == 0``) despite requesting
        ``interface_coarse_geneo_k=4`` and claiming "GenEO all active" in
        the docstring above. Tile 0's block is replaced with
        ``_near_null_spd_block`` (a genuine near-null eigendirection, as
        measured in Stage 2 -- see this module's docstring) specifically so
        at least one GenEO column is actually built and exercised by the
        SPD probe below; the ``n_geneo_cols > 0`` assert would fail without
        that block.
        """
        n = 40
        tile_idx = {
            0: np.arange(0, 12, dtype=np.int32),
            1: np.arange(10, 22, dtype=np.int32),
            2: np.arange(20, 32, dtype=np.int32),
            3: np.arange(30, 38, dtype=np.int32),  # leaves 38, 39 unowned
        }
        rng = np.random.default_rng(21)
        tile_schur = {}
        for tid, idx in tile_idx.items():
            A = rng.standard_normal((len(idx), len(idx)))
            tile_schur[tid] = A @ A.T / len(idx) + 3.0 * np.eye(len(idx))
        # Tile 0 is entirely first-seen-owned (no overlap with an earlier
        # tile), so its block-Jacobi ownership block equals this full
        # matrix -- give it a near-null mode so GenEO has something to pick
        # up (see docstring above).
        tile_schur[0] = _near_null_spd_block(len(tile_idx[0]), rng)
        island_idx = np.array([15], dtype=np.int64)  # inside tile 1's block
        extra_diag = np.zeros(n)
        extra_diag[island_idx] = 1e5
        S_extra = sp.diags(extra_diag).tocsr()

        solver = InterfaceCGSolver(
            n_interface=n, matvec_mode='tilewise',
            tile_schur_complements={k: v.copy() for k, v in tile_schur.items()},
            tile_index_maps=tile_idx, S_extra=S_extra,
            preconditioner='two_level', rtol=1e-8, atol=1e-14,
            matvec_threads=1, island_idx=island_idx,
            interface_coarse_geneo_k=4,
        )
        try:
            assert solver._coarse is not None
            # GenEO must actually be ACTIVE (not just requested): without
            # tile 0's near-null block this is 0 and the SPD probe below
            # would exercise only islands + unowned, not GenEO.
            assert solver._coarse.n_geneo_cols > 0, (
                "GenEO enrichment produced no columns -- fixture no longer "
                "has a near-null eigendirection to enrich"
            )
            rng2 = np.random.default_rng(22)
            for _ in range(50):
                x = rng2.standard_normal(n)
                Mx = solver._M.matvec(x)
                assert x @ Mx > 0, "two_level M must stay SPD"
            # island row zeroed in Z.
            assert np.all(solver._coarse.Z.toarray()[15, :] == 0.0)
            # unowned column present.
            assert '__unowned__' in solver._coarse.col_labels
        finally:
            solver.close()

    def test_island_penalty_block_does_not_produce_spurious_geneo_columns(self):
        """Finding 3 regression (call-site level, ownership block containing
        a penalized island row): a block-Jacobi ownership block whose row
        set includes an island node (penalized via S_extra's 1e5 mS
        diagonal, not part of the tile's own physical Schur block) must
        thread the block-local island mask through to
        interface_coarse.geneo_lowest_eigenpairs so the near-null selection
        runs against the PHYSICAL spectrum, not the penalty-inflated one --
        no spurious GenEO columns from healthy (not near-null) eigenvalues,
        and the genuine near-null pair still enriches.
        """
        n = 6
        idx = np.arange(n, dtype=np.int32)
        tile_idx = {'A': idx}

        # Physical (pre-penalty) tile Schur block: healthy eigenspectrum
        # [1e-9, 0.01, 0.05, 2.0, 5.0] on the non-island 5x5 submatrix,
        # decoupled (zero off-diagonal) from the island row -- matches a
        # real isolated/floating island node.
        rng = np.random.default_rng(31)
        Q, _ = np.linalg.qr(rng.standard_normal((5, 5)))
        eigs = np.array([1e-9, 0.01, 0.05, 2.0, 5.0])
        sub5 = (Q * eigs) @ Q.T
        sub5 = 0.5 * (sub5 + sub5.T)
        tile_schur_block = np.zeros((n, n))
        tile_schur_block[:5, :5] = sub5
        tile_idx_maps_schur = {'A': tile_schur_block}

        island_idx = np.array([5], dtype=np.int64)
        extra_diag = np.zeros(n)
        extra_diag[5] = 1e5  # S_extra's island penalty diagonal
        S_extra = sp.diags(extra_diag).tocsr()

        solver = InterfaceCGSolver(
            n_interface=n, matvec_mode='tilewise',
            tile_schur_complements=tile_idx_maps_schur, tile_index_maps=tile_idx,
            S_extra=S_extra, preconditioner='two_level', matvec_threads=1,
            island_idx=island_idx, interface_coarse_geneo_k=4,
            interface_coarse_geneo_tol=1e-6,
        )
        try:
            assert len(solver._geneo_pairs) == 1
            _gidx, _V_k, w_k = solver._geneo_pairs[0]
            assert w_k.shape[0] == 1, (
                f"expected only the genuine near-null (1e-9) eigenpair to "
                f"enrich against the PHYSICAL lambda_max=5.0 threshold "
                f"(1e-6*5.0=5e-6); got {w_k.shape[0]} columns -- old/buggy "
                f"code thresholds against the island-inflated lambda_max="
                f"1e5 (thresh=0.1), letting the 0.01/0.05 healthy "
                f"eigenvalues through as spurious near-null columns"
            )
            np.testing.assert_allclose(w_k[0], 1e-9, atol=1e-12, rtol=1e-3)
        finally:
            solver.close()

    @pytest.mark.parametrize('n_threads', [1, 2, 8])
    def test_thread_count_invariance(self, n_threads):
        tile_schur, tile_idx, n = _chain_tiles(24)
        rng = np.random.default_rng(77)
        b = rng.standard_normal(n)
        solver = InterfaceCGSolver(
            n_interface=n, matvec_mode='tilewise',
            tile_schur_complements={k: v.copy() for k, v in tile_schur.items()},
            tile_index_maps=tile_idx,
            preconditioner='two_level', rtol=1e-10, atol=1e-16,
            matvec_threads=n_threads, strict=True, maxiter=5000,
        )
        try:
            x = solver(b)
            assert solver._coarse is not None
        finally:
            solver.close()
        # Reference: serial.
        solver_ref = InterfaceCGSolver(
            n_interface=n, matvec_mode='tilewise',
            tile_schur_complements={k: v.copy() for k, v in tile_schur.items()},
            tile_index_maps=tile_idx,
            preconditioner='two_level', rtol=1e-10, atol=1e-16,
            matvec_threads=1, strict=True, maxiter=5000,
        )
        try:
            x_ref = solver_ref(b)
        finally:
            solver_ref.close()
        np.testing.assert_allclose(x, x_ref, rtol=1e-8, atol=1e-10)

    def test_warm_start_composes_reproducible_iterations(self):
        """Edge case 5: within one solver, Z/S_c are built ONCE (fixed);
        repeated cold-started solves give reproducible iteration counts."""
        tile_schur, tile_idx, n = _chain_tiles(20)
        rng = np.random.default_rng(3)
        b = rng.standard_normal(n)
        solver = InterfaceCGSolver(
            n_interface=n, matvec_mode='tilewise',
            tile_schur_complements={k: v.copy() for k, v in tile_schur.items()},
            tile_index_maps=tile_idx,
            preconditioner='two_level', rtol=1e-10, atol=1e-16,
            matvec_threads=1, strict=True, maxiter=5000,
        )
        try:
            coarse_before = solver._coarse
            solver(b)
            iters1 = solver.stats['last_cg_iters']

            solver.reset_warm_start()
            solver(b)
            iters2 = solver.stats['last_cg_iters']

            solver.reset_warm_start()
            solver(b)
            iters3 = solver.stats['last_cg_iters']

            assert iters1 == iters2 == iters3
            # Coarse space is built once, never rebuilt across solves.
            assert solver._coarse is coarse_before
        finally:
            solver.close()

    def test_fp32_storage_path_converges(self):
        """Edge case 6: fp32 tilewise storage + two_level -- the coarse
        build's S @ Z matmat must accumulate fp64 (it reuses the existing
        matmat, which already does); assert convergence + accuracy at the
        documented fp32 tolerance."""
        tile_schur, tile_idx, n = _chain_tiles(30)
        S = np.zeros((n, n))
        for tid, idx in tile_idx.items():
            S[np.ix_(idx, idx)] += tile_schur[tid]
        rng = np.random.default_rng(6)
        b = rng.standard_normal(n)
        x_direct = np.linalg.solve(S, b)

        solver = InterfaceCGSolver(
            n_interface=n, matvec_mode='tilewise',
            tile_schur_complements={k: v.copy() for k, v in tile_schur.items()},
            tile_index_maps=tile_idx,
            preconditioner='two_level', rtol=1e-6, atol=1e-12,
            matvec_threads=1, strict=True, maxiter=5000,
            matvec_dtype='float32', interface_coarse_geneo_k=4,
        )
        try:
            assert solver._coarse is not None
            x = solver(b)
            rel = np.max(np.abs(x - x_direct)) / max(1e-300, np.max(np.abs(x_direct)))
            assert rel < 1e-3, f"fp32 two_level rel err {rel:.3e}"
        finally:
            solver.close()

    def test_geneo_reuse_path_no_recompute(self):
        """Edge case 7: a block that hits the indefinite-block eigh
        fallback contributes its already-computed lowest eigenvectors to
        GenEO WITHOUT a second eigendecomposition of that block."""
        n = 3
        idx = np.array([0, 1, 2], dtype=np.int32)
        rng = np.random.default_rng(4)
        V, _ = np.linalg.qr(rng.standard_normal((3, 3)))
        S_i = V @ np.diag([-5.0, -1.0, 2.0]) @ V.T
        S_i = 0.5 * (S_i + S_i.T)
        tile_schur = {'A': S_i}
        tile_idx = {'A': idx}

        real_geneo = ic.geneo_lowest_eigenpairs
        calls = []

        def spy_geneo(sub, *args, **kwargs):
            calls.append(kwargs.get('precomputed'))
            return real_geneo(sub, *args, **kwargs)

        with mock.patch(
            'distributed.interface_iterative.interface_coarse.geneo_lowest_eigenpairs',
            side_effect=spy_geneo,
        ):
            solver = InterfaceCGSolver(
                n_interface=n, matvec_mode='tilewise',
                tile_schur_complements=tile_schur, tile_index_maps=tile_idx,
                preconditioner='two_level', matvec_threads=1,
                interface_coarse_geneo_k=4, interface_coarse_geneo_tol=1.0,
            )
        try:
            kinds = [kind for _idx, kind, _payload in solver._bj_block_factors]
            assert kinds == ['eigh']
            # Exactly one GenEO call for this block, and it was given the
            # PRECOMPUTED spectrum (reuse, not a second eigendecomposition).
            assert len(calls) == 1
            assert calls[0] is not None
            w_reused, V_reused = calls[0]
            assert w_reused.shape == (3,) and V_reused.shape == (3, 3)
            assert len(solver._geneo_pairs) == 1
            assert solver._geneo_pairs[0][1].shape[1] > 0
        finally:
            solver.close()

    def test_degrades_to_block_jacobi_with_warning_when_coarse_build_fails(self, caplog):
        tile_schur, tile_idx, n = _chain_tiles(10)
        with caplog.at_level(logging.WARNING):
            solver = InterfaceCGSolver(
                n_interface=n, matvec_mode='tilewise',
                tile_schur_complements={k: v.copy() for k, v in tile_schur.items()},
                tile_index_maps=tile_idx,
                preconditioner='two_level', matvec_threads=1,
                interface_coarse_max_cols=1,  # forces T' cap failure
            )
        try:
            assert solver._coarse is None
            assert solver.preconditioner == 'block_jacobi'
            assert solver.requested_preconditioner == 'two_level'
            assert any('degrading' in r.message.lower() for r in caplog.records)
            assert solver._M is not None
        finally:
            solver.close()

    def test_geneo_overflow_stays_two_level_with_pou_only_via_solver(self, caplog):
        """Regression (spec-compliance finding), end-to-end through
        InterfaceCGSolver: when GenEO enrichment alone pushes T' over
        interface_coarse_max_cols but the PoU-only column count fits, the
        solver must keep ``preconditioner == 'two_level'`` (PoU-only coarse
        space active) rather than degrading all the way to block_jacobi --
        the exact stagnating preconditioner Stage 3 exists to avoid at the
        high-tile-count regime. ``interface_coarse_geneo_tol=1.0`` forces
        every eigenpair of every block to pass the GenEO threshold (deter-
        ministic overflow, independent of this fixture's actual spectrum)."""
        tile_schur, tile_idx, n = _chain_tiles(12)
        n_pou_cols = 12  # one column per tile; _chain_tiles has no unowned nodes

        with caplog.at_level(logging.WARNING):
            solver = InterfaceCGSolver(
                n_interface=n, matvec_mode='tilewise',
                tile_schur_complements={k: v.copy() for k, v in tile_schur.items()},
                tile_index_maps=tile_idx,
                preconditioner='two_level', matvec_threads=1,
                interface_coarse_geneo_k=4, interface_coarse_geneo_tol=1.0,
                interface_coarse_max_cols=n_pou_cols,
                strict=True, rtol=1e-10, atol=1e-16, maxiter=5000,
            )
        try:
            assert solver._geneo_pairs, "fixture must contribute GenEO pairs for this test to be meaningful"
            assert solver.preconditioner == 'two_level', (
                "must NOT degrade to block_jacobi when PoU-only fits max_cols"
            )
            assert solver._coarse is not None
            assert solver._coarse.n_geneo_cols == 0
            assert solver._coarse.n_cols == n_pou_cols
            assert any(
                'falling back to pou-only' in r.message.lower()
                for r in caplog.records
            )
            # The (degraded, PoU-only) two_level preconditioner must still be
            # usable end-to-end: SPD probe + a real CG solve converges.
            rng = np.random.default_rng(17)
            for _ in range(10):
                x = rng.standard_normal(n)
                assert x @ solver._M.matvec(x) > 0
            b = rng.standard_normal(n)
            x_cg = solver(b)  # strict=True -> raises if it fails to converge
            S = np.zeros((n, n))
            for tid, idx in tile_idx.items():
                S[np.ix_(idx, idx)] += tile_schur[tid]
            x_direct = np.linalg.solve(S, b)
            np.testing.assert_allclose(x_cg, x_direct, rtol=1e-6, atol=1e-8)
        finally:
            solver.close()

    def test_bj_memory_downgrade_still_adds_coarse_term(self, monkeypatch):
        """If block_jacobi itself downgrades to jacobi (memory budget),
        two_level must still layer the coarse term on top."""
        import distributed.interface_iterative as ii

        tile_schur, tile_idx, n = _chain_tiles(20)
        monkeypatch.setattr(ii, 'BLOCK_JACOBI_MAX_FACTOR_BYTES', 1)
        solver = InterfaceCGSolver(
            n_interface=n, matvec_mode='tilewise',
            tile_schur_complements={k: v.copy() for k, v in tile_schur.items()},
            tile_index_maps=tile_idx,
            preconditioner='two_level', matvec_threads=1,
        )
        try:
            assert solver._bj_downgraded is True
            assert solver._coarse is not None
            assert solver.preconditioner == 'two_level'
            assert 'jacobi' in solver.preconditioner_label
        finally:
            solver.close()

    def test_bj_memory_downgrade_warning_names_two_level_and_geneo_skip(
        self, monkeypatch, caplog,
    ):
        """Finding 5 (round 2) regression: when a preconditioner='two_level'
        request trips the block-Jacobi memory-budget downgrade guard, the
        WARNING must name the ACTUAL requested preconditioner
        ('two_level'), not a hardcoded 'block_jacobi' (self.requested_
        preconditioner is 'two_level' at that point -- _build_block_jacobi
        also serves two_level requests, dispatched from _build_
        preconditioner's two_level branch) -- and it must additionally say
        GenEO enrichment is skipped for this factor (no blocks get
        cho-factored on the _build_jacobi_fallback path, so the eventual
        coarse space -- if built at all -- is PoU-only, not PoU+GenEO).
        """
        import logging
        import distributed.interface_iterative as ii

        tile_schur, tile_idx, n = _chain_tiles(20)
        monkeypatch.setattr(ii, 'BLOCK_JACOBI_MAX_FACTOR_BYTES', 1)
        with caplog.at_level(logging.WARNING):
            solver = InterfaceCGSolver(
                n_interface=n, matvec_mode='tilewise',
                tile_schur_complements={k: v.copy() for k, v in tile_schur.items()},
                tile_index_maps=tile_idx,
                preconditioner='two_level', matvec_threads=1,
            )
        try:
            downgrade_records = [
                r for r in caplog.records
                if 'estimated factor memory' in r.message
            ]
            assert downgrade_records, (
                f"expected the memory-budget downgrade WARNING; got: "
                f"{[r.message for r in caplog.records]}"
            )
            joined = ' '.join(r.message for r in downgrade_records)
            assert "'two_level' -> 'jacobi'" in joined, (
                f"WARNING must name the ACTUAL requested preconditioner "
                f"('two_level'), not a hardcoded 'block_jacobi'; got: "
                f"{joined!r}"
            )
            assert "'block_jacobi' -> 'jacobi'" not in joined, (
                f"WARNING must NOT falsely claim 'block_jacobi' was "
                f"requested when the actual request was 'two_level'; got: "
                f"{joined!r}"
            )
            assert 'geneo' in joined.lower() and (
                'skip' in joined.lower()
            ), (
                f"WARNING must say GenEO enrichment is skipped for this "
                f"factor; got: {joined!r}"
            )
        finally:
            solver.close()

    def test_bj_memory_downgrade_warning_plain_block_jacobi_unaffected(
        self, monkeypatch, caplog,
    ):
        """Sibling no-regression check: an EXPLICIT preconditioner=
        'block_jacobi' request (not two_level) must still see its own name
        in the downgrade WARNING, with no spurious GenEO-skip note (GenEO
        is a two_level-only concept)."""
        import logging
        import distributed.interface_iterative as ii

        tile_schur, tile_idx, n = _chain_tiles(20)
        monkeypatch.setattr(ii, 'BLOCK_JACOBI_MAX_FACTOR_BYTES', 1)
        with caplog.at_level(logging.WARNING):
            solver = InterfaceCGSolver(
                n_interface=n, matvec_mode='tilewise',
                tile_schur_complements={k: v.copy() for k, v in tile_schur.items()},
                tile_index_maps=tile_idx,
                preconditioner='block_jacobi', matvec_threads=1,
            )
        try:
            downgrade_records = [
                r for r in caplog.records
                if 'estimated factor memory' in r.message
            ]
            assert downgrade_records
            joined = ' '.join(r.message for r in downgrade_records)
            assert "'block_jacobi' -> 'jacobi'" in joined
            assert 'geneo' not in joined.lower()
        finally:
            solver.close()

    def test_preconditioner_label_format(self):
        tile_schur, tile_idx, n = _chain_tiles(20)
        solver = InterfaceCGSolver(
            n_interface=n, matvec_mode='tilewise',
            tile_schur_complements={k: v.copy() for k, v in tile_schur.items()},
            tile_index_maps=tile_idx,
            preconditioner='two_level', matvec_threads=1,
        )
        try:
            label = solver.preconditioner_label
            assert label.startswith('two_level(')
            assert "T'=" in label
            assert 'rank=' in label
        finally:
            solver.close()

    # -----------------------------------------------------------------
    # Finding 6: build_coarse_space call must never raise out of __init__
    # -----------------------------------------------------------------

    def test_build_coarse_space_exception_degrades_to_base_not_raises(
        self, monkeypatch, caplog,
    ):
        """Finding 6 regression: _augment_with_coarse_space documents
        'never raises', but the pre-fix code called
        interface_coarse.build_coarse_space with NO try/except -- any
        exception it raises (e.g. MemoryError/LinAlgError from the dense
        Z.toarray()/eigh(S_c) calls) propagated straight out of
        InterfaceCGSolver.__init__, aborting prepare()/factor() entirely
        instead of degrading to the base preconditioner. Monkeypatches
        build_coarse_space to raise unconditionally and asserts
        construction still succeeds with a working base preconditioner and
        a WARNING naming the failure."""
        import distributed.interface_iterative as ii_mod

        tile_schur, tile_idx, n = _chain_tiles(10)

        def _boom(*args, **kwargs):
            raise MemoryError("simulated coarse-build OOM")

        monkeypatch.setattr(ii_mod.interface_coarse, 'build_coarse_space', _boom)

        with caplog.at_level(logging.WARNING):
            solver = InterfaceCGSolver(
                n_interface=n, matvec_mode='tilewise',
                tile_schur_complements={k: v.copy() for k, v in tile_schur.items()},
                tile_index_maps=tile_idx,
                preconditioner='two_level', matvec_threads=1,
                strict=True, rtol=1e-10, atol=1e-16, maxiter=5000,
            )
        try:
            assert solver._coarse is None
            assert solver.preconditioner == 'block_jacobi'
            assert solver.requested_preconditioner == 'two_level'
            assert solver._M is not None, (
                "must degrade to a WORKING base preconditioner, not crash "
                "or leave M=None"
            )
            assert any(
                'memoryerror' in r.message.lower() for r in caplog.records
            ), "expected a WARNING naming the exception type"
            # And the degraded solver must still actually solve correctly.
            S = np.zeros((n, n))
            for tid, idx in tile_idx.items():
                S[np.ix_(idx, idx)] += tile_schur[tid]
            rng = np.random.default_rng(2)
            b = rng.standard_normal(n)
            x = solver(b)
            x_direct = np.linalg.solve(S, b)
            np.testing.assert_allclose(x, x_direct, rtol=1e-6, atol=1e-8)
        finally:
            solver.close()

    # -----------------------------------------------------------------
    # Finding 8: degrade/label logic must derive the ACTUAL base component
    # -----------------------------------------------------------------

    def test_all_blocks_failed_and_coarse_failed_reports_none_not_block_jacobi(self):
        """Finding 8 regression: when the base block-Jacobi builder returns
        None OUTRIGHT (no tile_index_maps AND no tile_schur_complements --
        e.g. 'assembled' matvec mode with neither provided) and the coarse
        build also fails (empty tile_index_maps), the degrade path must
        report preconditioner='none' (CG genuinely unpreconditioned), NOT
        the pre-fix 'block_jacobi' (which would falsely claim an active
        preconditioner in backend_info/logs while CG runs raw)."""
        n = 5
        S = sp.eye(n, format='csr') * 2.0 - sp.eye(n, k=1) * 0.5 - sp.eye(n, k=-1) * 0.5
        S = S.tocsr()
        solver = InterfaceCGSolver(
            n_interface=n, matvec_mode='assembled', S_global=S,
            tile_schur_complements=None, tile_index_maps=None,
            preconditioner='two_level', matvec_threads=1,
        )
        try:
            assert solver._M is None, (
                "fixture must produce a genuinely unpreconditioned solver "
                "for this test to be meaningful"
            )
            assert solver._coarse is None
            assert solver.preconditioner == 'none', (
                f"expected 'none' (base builder returned None outright), "
                f"got {solver.preconditioner!r} (pre-fix: falsely reported "
                f"'block_jacobi' whenever _bj_downgraded was False, even "
                f"when the base was actually None)"
            )
            assert solver.preconditioner_label == 'none'
        finally:
            solver.close()


# ---------------------------------------------------------------------------
# 6. build_interface_solver wiring
# ---------------------------------------------------------------------------


class TestBuildInterfaceSolverTwoLevel:

    def test_default_auto_promotes_two_level_for_tilewise(self):
        n = 500
        tile_schur, tile_idx = _make_synthetic_tiles(n, 10, 20, 60, seed=11)
        _, resolved, cg_solver = build_interface_solver(
            S_global=None, interface_solver='cg',
            tile_schur_complements=tile_schur, tile_index_maps=tile_idx,
            matvec_mode='tilewise', preconditioner='auto',
            n_interface=n, matvec_threads=1,
        )
        assert resolved == 'cg'
        assert cg_solver.preconditioner == 'two_level'
        cg_solver.close()

    def test_island_idx_threaded_through(self):
        tile_schur, tile_idx, n = _chain_tiles(15)
        island_idx = np.array([3], dtype=np.int64)
        _, _, cg_solver = build_interface_solver(
            S_global=None, interface_solver='cg',
            tile_schur_complements={k: v.copy() for k, v in tile_schur.items()},
            tile_index_maps=tile_idx,
            matvec_mode='tilewise', preconditioner='two_level',
            n_interface=n, matvec_threads=1, island_idx=island_idx,
        )
        try:
            assert cg_solver._coarse is not None
            assert np.all(cg_solver._coarse.Z.toarray()[3, :] == 0.0)
        finally:
            cg_solver.close()


# ---------------------------------------------------------------------------
# 7. S15 forced-CG equivalence matrix extension (1e-12 agreement with direct)
# ---------------------------------------------------------------------------


class TestTwoLevelForcedCGEquivalence:

    @staticmethod
    def _build_model():
        return _build_two_tile_distributed_model(package_cap_edges=[])

    def _run(self, settings):
        from distributed.solver import DistributedDDMSolver

        model = self._build_model()
        model.settings.update(settings)
        solver = DistributedDDMSolver(model)
        ctx = solver.prepare()
        try:
            return solver.solve_dc(ctx).flatten()
        finally:
            ctx.release()
            model.shutdown()

    def test_two_level_matches_direct_at_pinned_rtol(self):
        v_direct = self._run({
            'interface_solver': 'direct', 'interface_matvec_mode': 'assembled',
        })
        v_two_level = self._run({
            'interface_solver': 'cg',
            'interface_matvec_mode': 'tilewise',
            'interface_preconditioner': 'two_level',
            'interface_cg_rtol': 1e-12,
            'interface_cg_atol': 1e-16,
        })
        common = set(v_direct) & set(v_two_level)
        assert common
        for node in common:
            diff = abs(v_direct[node] - v_two_level[node])
            assert diff <= 1e-8, (
                f"node {node}: direct={v_direct[node]!r} vs "
                f"two_level={v_two_level[node]!r} diff={diff:.3e}"
            )

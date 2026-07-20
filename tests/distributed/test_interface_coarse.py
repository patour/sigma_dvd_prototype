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
from distributed import interface_iterative as ii  # noqa: E402
from distributed.interface_iterative import (  # noqa: E402
    InterfaceCGSolver,
    build_interface_solver,
    resolve_preconditioner,
    _deflated_pcg,
    _is_breakdown,
    _BREAKDOWN_EPS,
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


def _ill_conditioned_jacobi_fixture(n=200, n_tiles=5, block=4, cond_log=9.0,
                                     seed=11):
    """A-DEF2 work package: a deliberately ILL-CONDITIONED dense SPD system
    (condition number ~exp(cond_log)) with a SMALL, non-degenerate coarse
    space (T'=n_tiles, disjoint 4-row ownership blocks -- unlike the chain
    fixture, T'/n here is a realistic ~10%, not near-100%) and NO tile-
    Schur structure at all (S is a full dense random-rotation matrix, not
    block-diagonal) -- forces the whole ``S`` through ``S_extra`` (tile
    Schur blocks are all-zero placeholders) so the ownership-block
    EXTRACTION in ``_form_owned_block`` still recovers the TRUE physical
    sub-blocks of ``S`` correctly (S4-style: sub = 0 (tile) + S_extra's
    slice = S's own slice).

    Needed because chain fixtures have T' close to n (near-degenerate --
    the coarse space alone captures almost the whole solution in ONE CG
    iteration with the corrected A-DEF2 algorithm, see interface_iterative
    module docstring's "A-DEF2" section) -- useless for testing anything
    that needs MANY iterations (progress_every logging, reprojection
    drift). This fixture, combined with a forced jacobi (not block_jacobi)
    base via a tiny ``block_jacobi_max_bytes``, needs ~600 iterations at
    rtol=1e-9 (measured, seed=11/cond_log=9.0/n_tiles=5/block=4) -- long
    enough to exercise both progress_every and many reprojection events.

    Returns ``(tile_schur, tile_idx, S_extra, S, n)``.
    """
    rng = np.random.default_rng(seed)
    Qm, _ = np.linalg.qr(rng.standard_normal((n, n)))
    eigs = np.exp(np.linspace(0.0, cond_log, n))
    S = (Qm * eigs) @ Qm.T
    S = 0.5 * (S + S.T)
    tile_idx = {
        t: np.arange(t * block, (t + 1) * block, dtype=np.int32)
        for t in range(n_tiles)
    }
    tile_schur = {t: np.zeros((block, block)) for t in range(n_tiles)}
    S_extra = sp.csr_matrix(S)
    return tile_schur, tile_idx, S_extra, S, n


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

    def test_bj_memory_downgrade_warning_names_two_level_and_geneo_decoupled(
        self, monkeypatch, caplog,
    ):
        """Finding 5 (round 2) regression, UPDATED for the A-DEF2 work
        package's Deliverable 1 (decoupled GenEO): when a
        preconditioner='two_level' request trips the block-Jacobi memory-
        budget downgrade guard, the WARNING must name the ACTUAL requested
        preconditioner ('two_level'), not a hardcoded 'block_jacobi'
        (self.requested_preconditioner is 'two_level' at that point --
        _build_block_jacobi also serves two_level requests, dispatched from
        _build_preconditioner's two_level branch) -- and it must now say
        GenEO enrichment runs via the DECOUPLED one-block-at-a-time pass
        (NOT "skipped" -- pre-Deliverable-1 behaviour, superseded: no block
        got cho-factored on the _build_jacobi_fallback path, so the
        eventual coarse space was always PoU-only; now
        _extract_geneo_decoupled runs regardless, so it stays PoU+GenEO).
        See test_bj_memory_downgrade_geneo_k_zero_still_says_disabled for
        the (still-accurate) "disabled" wording when GenEO itself is off.
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
                # Measurement-driven flip (2026-07-20) changed
                # interface_coarse.DEFAULT_GENEO_K to 0 -- this test
                # specifically exercises the "GenEO enabled" decoupled-pass
                # wording, so it must now request GenEO explicitly rather
                # than rely on the (no-longer-enriching) default.  See
                # test_bj_memory_downgrade_geneo_k_zero_still_says_disabled
                # for the geneo_k=0 sibling.
                interface_coarse_geneo_k=4,
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
            assert 'geneo' in joined.lower() and 'decoupled' in joined.lower(), (
                f"WARNING must say GenEO enrichment runs via the decoupled "
                f"pass (Deliverable 1); got: {joined!r}"
            )
            assert 'skip' not in joined.lower(), (
                f"WARNING must NOT claim GenEO is skipped -- Deliverable 1 "
                f"runs it via the decoupled pass; got: {joined!r}"
            )
        finally:
            solver.close()

    def test_bj_memory_downgrade_geneo_k_zero_still_says_disabled(
        self, monkeypatch, caplog,
    ):
        """Sibling to the decoupled-GenEO regression above: with
        interface_coarse_geneo_k=0 (GenEO explicitly disabled), the
        downgrade WARNING must still say the coarse space will be
        PoU-only -- _extract_geneo_decoupled must NOT run (self._want_geneo
        is False), so no cho-factor-per-block cost is paid when the caller
        asked for PoU-only in the first place."""
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
                interface_coarse_geneo_k=0,
            )
        try:
            assert solver._geneo_pairs == []
            downgrade_records = [
                r for r in caplog.records
                if 'estimated factor memory' in r.message
            ]
            joined = ' '.join(r.message for r in downgrade_records)
            assert 'disabled' in joined.lower() and 'pou-only' in joined.lower()
            assert 'decoupled' not in joined.lower()
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
        # Measurement-driven flip (2026-07-20): interface_coarse.
        # DEFAULT_APPLY_MODE is now 'deflated', so the default-constructed
        # label carries the '[deflated]' tag (see preconditioner_label's
        # docstring) -- pin the NEW default's format here.
        tile_schur, tile_idx, n = _chain_tiles(20)
        solver = InterfaceCGSolver(
            n_interface=n, matvec_mode='tilewise',
            tile_schur_complements={k: v.copy() for k, v in tile_schur.items()},
            tile_index_maps=tile_idx,
            preconditioner='two_level', matvec_threads=1,
        )
        try:
            label = solver.preconditioner_label
            assert label.startswith('two_level[deflated](')
            assert "T'=" in label
            assert 'rank=' in label
        finally:
            solver.close()

    def test_preconditioner_label_format_additive(self):
        """Explicit apply_mode='additive' keeps the byte-identical pre-flip
        (untagged) label format -- see preconditioner_label's docstring."""
        tile_schur, tile_idx, n = _chain_tiles(20)
        solver = InterfaceCGSolver(
            n_interface=n, matvec_mode='tilewise',
            tile_schur_complements={k: v.copy() for k, v in tile_schur.items()},
            tile_index_maps=tile_idx,
            preconditioner='two_level', matvec_threads=1,
            interface_coarse_apply_mode='additive',
        )
        try:
            label = solver.preconditioner_label
            assert label.startswith('two_level(')
            assert not label.startswith('two_level[')
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

    def test_two_level_adef2_matches_direct_dc_assembled(self):
        """A-DEF2 work package: adef2 row for the DC (assembled matvec)
        forced-CG equivalence matrix."""
        v_direct = self._run({
            'interface_solver': 'direct', 'interface_matvec_mode': 'assembled',
        })
        v_adef2 = self._run({
            'interface_solver': 'cg',
            'interface_matvec_mode': 'tilewise',
            'interface_preconditioner': 'two_level',
            'interface_coarse_apply_mode': 'deflated',
            'interface_cg_rtol': 1e-12,
            'interface_cg_atol': 1e-16,
        })
        common = set(v_direct) & set(v_adef2)
        assert common
        for node in common:
            diff = abs(v_direct[node] - v_adef2[node])
            assert diff <= 1e-8, (
                f"node {node}: direct={v_direct[node]!r} vs "
                f"adef2={v_adef2[node]!r} diff={diff:.3e}"
            )

    def test_two_level_adef2_matches_direct_dc_never_assemble(self):
        """A-DEF2 work package: adef2 row for the DC never-assemble
        (interface_drop_s_global) forced-CG equivalence matrix."""
        from distributed.solver import DistributedDDMSolver

        v_direct = self._run({
            'interface_solver': 'direct', 'interface_matvec_mode': 'assembled',
        })

        model = self._build_model()
        model.island_detection_mode = 'summaries'
        model.component_summaries = []
        model.settings.update({
            'interface_solver': 'cg',
            'interface_matvec_mode': 'tilewise',
            'interface_preconditioner': 'two_level',
            'interface_coarse_apply_mode': 'deflated',
            'interface_cg_rtol': 1e-12,
            'interface_cg_atol': 1e-16,
            'interface_drop_s_global': True,
        })
        solver = DistributedDDMSolver(model)
        ctx = solver.prepare()
        try:
            assert ctx._S_global is None, "sanity: never-assemble must have run"
            v_adef2 = solver.solve_dc(ctx).flatten()
        finally:
            ctx.release()
            model.shutdown()

        common = set(v_direct) & set(v_adef2)
        assert common
        for node in common:
            diff = abs(v_direct[node] - v_adef2[node])
            assert diff <= 1e-8, (
                f"node {node}: direct={v_direct[node]!r} vs "
                f"adef2(never-assemble)={v_adef2[node]!r} diff={diff:.3e}"
            )


class TestADef2TransientForcedCGEquivalence:
    """A-DEF2 work package: extend the forced-CG equivalence matrix with
    TRANSIENT rows (assembled + never-assemble), mirroring
    TestS15ForcedCGEquivalenceMatrix.test_transient_tilewise_never_assemble_matches_direct
    in test_interface_iterative_stage2.py but for apply_mode='deflated'."""

    def _run(self, tmp_path, interface_solver, matvec_mode, drop_s_global,
              summaries, apply_mode):
        from distributed.solver import DistributedDDMSolver

        model = _build_two_tile_distributed_model(
            package_cap_edges=[('pad', 'shared', 50.0)],
        )
        tag = f'{interface_solver}_{drop_s_global}_{apply_mode}'
        for tc in model.metadata.tile_configs:
            tc.ckt_path = str(tmp_path / tag / 'dummy.ckt')
        if summaries:
            model.island_detection_mode = 'summaries'
            model.component_summaries = []
        settings = {
            'interface_solver': interface_solver,
            'interface_matvec_mode': matvec_mode,
            'interface_cg_rtol': 1e-12,
            'interface_cg_atol': 1e-16,
        }
        if apply_mode is not None:
            settings['interface_preconditioner'] = 'two_level'
            settings['interface_coarse_apply_mode'] = apply_mode
        if drop_s_global:
            settings['interface_drop_s_global'] = True
        model.settings.update(settings)
        solver = DistributedDDMSolver(model)
        dc_ctx = solver.prepare()
        trans_ctx = solver.prepare_transient(dt=1e-10, method='be')
        try:
            smoothed = solver.preprocess_sources(
                time_step=1e-10, t_start=0.0, t_end=2e-9, smooth=False,
                pkl_dir=str(tmp_path / tag),
            )
            result = solver.solve_transient(
                trans_ctx, dc_context=dc_ctx, t_end=2e-9,
                smoothed_sources=smoothed,
            )
            return result.as_flat(), trans_ctx._S_global is None
        finally:
            trans_ctx.release()
            dc_ctx.release()
            model.shutdown()

    def _assert_matches(self, v_direct, v_test, label):
        common = set(v_direct) & set(v_test)
        assert common, "fixture produced no comparable nodes"
        for node in common:
            drop_d, _t_d = v_direct[node]
            drop_t, _t_t = v_test[node]
            assert abs(drop_d - drop_t) < 1e-8, (
                f"[{label}] node {node}: direct drop={drop_d!r} vs "
                f"{label} drop={drop_t!r}"
            )

    def test_transient_adef2_assembled_matches_direct(self, tmp_path):
        v_direct, _ = self._run(
            tmp_path, 'direct', 'assembled', False, summaries=False,
            apply_mode=None,
        )
        v_adef2, was_never = self._run(
            tmp_path, 'cg', 'tilewise', False, summaries=False,
            apply_mode='deflated',
        )
        assert not was_never
        self._assert_matches(v_direct, v_adef2, 'adef2-assembled')

    def test_transient_adef2_never_assemble_matches_direct(self, tmp_path):
        v_direct, _ = self._run(
            tmp_path, 'direct', 'assembled', False, summaries=False,
            apply_mode=None,
        )
        v_adef2, was_never = self._run(
            tmp_path, 'cg', 'tilewise', True, summaries=True,
            apply_mode='deflated',
        )
        assert was_never, "sanity: never-assemble path must have run"
        self._assert_matches(v_direct, v_adef2, 'adef2-never-assemble')


# ---------------------------------------------------------------------------
# 8. A-DEF2 work package: apply_with_SQ / retained SZ, hand-rolled deflated
#    PCG, decoupled GenEO, warm-start extrapolation.
# ---------------------------------------------------------------------------


class TestSolveScPinv:
    """CoarseSpace._solve_Sc_pinv (Finding 8, round-1 code review): the
    method documents accepting either a 1-D ``(T',)`` or 2-D ``(T', k)``
    input. A direct 1-D call used to silently return a ``(T', rank)``
    matrix instead of the documented ``(T',)`` vector -- a numpy
    broadcasting bug (``(rank,) * (rank, 1)`` promotes to ``(rank,
    rank)``) -- see ``interface_coarse.py``'s docstring for the full
    root-cause explanation. All current callers (apply/apply_with_SQ,
    plus this file's test-local ``_apply_QS`` helper -- see Finding 10,
    round 2) already reshape to 2-D before calling this, so this class
    tests the method directly, per its own documented 1-D contract."""

    @staticmethod
    def _build(n_tiles=10):
        tile_schur, tile_idx, n = _chain_tiles(n_tiles)
        S = np.zeros((n, n))
        for tid, idx in tile_idx.items():
            S[np.ix_(idx, idx)] += tile_schur[tid]
        coarse = ic.build_coarse_space(lambda X: S @ X, tile_idx, n=n)
        assert coarse is not None
        return coarse

    @staticmethod
    def _dense_sc_pinv(coarse):
        # S_c^+ restricted to the kept (V_c, inv_lambda_c) eigenpairs --
        # the exact quantity _solve_Sc_pinv computes without ever
        # materializing it as a dense (T', T') array.
        return (coarse.V_c * coarse.inv_lambda_c) @ coarse.V_c.T

    def test_1d_input_returns_1d_matching_dense_reference(self):
        coarse = self._build()
        T_prime = coarse.n_cols
        Sc_pinv_dense = self._dense_sc_pinv(coarse)

        rng = np.random.default_rng(7)
        w = rng.standard_normal(T_prime)
        out = coarse._solve_Sc_pinv(w)
        assert out.shape == (T_prime,), (
            f"expected a 1-D (T',)={T_prime} output for a 1-D input, got "
            f"shape {out.shape}"
        )
        ref = Sc_pinv_dense @ w
        np.testing.assert_allclose(out, ref, rtol=1e-10, atol=1e-12)

    def test_2d_input_returns_2d_matching_dense_reference(self):
        coarse = self._build()
        T_prime = coarse.n_cols
        Sc_pinv_dense = self._dense_sc_pinv(coarse)

        rng = np.random.default_rng(8)
        W = rng.standard_normal((T_prime, 3))
        out = coarse._solve_Sc_pinv(W)
        assert out.shape == (T_prime, 3)
        ref = Sc_pinv_dense @ W
        np.testing.assert_allclose(out, ref, rtol=1e-10, atol=1e-12)

    def test_1d_columns_match_2d_batch(self):
        """The 1-D fix must stay consistent with the 2-D path: each
        column of a batch call must equal the corresponding 1-D call."""
        coarse = self._build()
        T_prime = coarse.n_cols
        rng = np.random.default_rng(9)
        W = rng.standard_normal((T_prime, 4))
        batch = coarse._solve_Sc_pinv(W)
        for j in range(4):
            col = coarse._solve_Sc_pinv(W[:, j])
            np.testing.assert_allclose(col, batch[:, j], rtol=1e-12, atol=1e-14)


class TestApplyWithSQ:
    """CoarseSpace.apply_with_SQ (Q r, S Q r) -- correctness vs a dense
    reference, retain_sz plumbing, and the "SZ not retained" guard."""

    def test_matches_dense_reference(self):
        tile_schur, tile_idx, n = _chain_tiles(12)
        S = np.zeros((n, n))
        for tid, idx in tile_idx.items():
            S[np.ix_(idx, idx)] += tile_schur[tid]
        coarse = ic.build_coarse_space(
            lambda X: S @ X, tile_idx, n=n, retain_sz=True,
        )
        assert coarse is not None
        assert coarse.SZ is not None

        Zd = coarse.Z.toarray()
        Sc = Zd.T @ S @ Zd
        Sc = 0.5 * (Sc + Sc.T)
        Sc_pinv = np.linalg.pinv(Sc, rcond=1e-10)

        rng = np.random.default_rng(41)
        x = rng.standard_normal(n)
        Qx_ref = Zd @ (Sc_pinv @ (Zd.T @ x))
        SQx_ref = S @ Qx_ref

        Qx, SQx = coarse.apply_with_SQ(x)
        np.testing.assert_allclose(Qx, Qx_ref, rtol=1e-8, atol=1e-10)
        np.testing.assert_allclose(SQx, SQx_ref, rtol=1e-8, atol=1e-9)

        # Qx must also equal the existing apply() -- same operator, just
        # returning the extra S-applied piece too.
        np.testing.assert_allclose(Qx, coarse.apply(x), rtol=1e-12, atol=1e-14)

    def test_batch_matches_columnwise(self):
        tile_schur, tile_idx, n = _chain_tiles(8)
        S = np.zeros((n, n))
        for tid, idx in tile_idx.items():
            S[np.ix_(idx, idx)] += tile_schur[tid]
        coarse = ic.build_coarse_space(
            lambda X: S @ X, tile_idx, n=n, retain_sz=True,
        )
        assert coarse is not None
        rng = np.random.default_rng(5)
        X = rng.standard_normal((n, 4))
        Q_batch, SQ_batch = coarse.apply_with_SQ(X)
        for j in range(4):
            Qj, SQj = coarse.apply_with_SQ(X[:, j])
            np.testing.assert_allclose(Q_batch[:, j], Qj, rtol=1e-10, atol=1e-12)
            np.testing.assert_allclose(SQ_batch[:, j], SQj, rtol=1e-10, atol=1e-12)

    def test_retain_sz_false_leaves_sz_none(self):
        tile_schur, tile_idx, n = _chain_tiles(6)
        S = np.zeros((n, n))
        for tid, idx in tile_idx.items():
            S[np.ix_(idx, idx)] += tile_schur[tid]
        coarse = ic.build_coarse_space(
            lambda X: S @ X, tile_idx, n=n, retain_sz=False,
        )
        assert coarse is not None
        assert coarse.SZ is None

    def test_apply_with_sq_raises_when_sz_not_retained(self):
        tile_schur, tile_idx, n = _chain_tiles(6)
        S = np.zeros((n, n))
        for tid, idx in tile_idx.items():
            S[np.ix_(idx, idx)] += tile_schur[tid]
        coarse = ic.build_coarse_space(lambda X: S @ X, tile_idx, n=n)
        assert coarse is not None
        assert coarse.SZ is None
        with pytest.raises(ValueError, match='retain_sz=True'):
            coarse.apply_with_SQ(np.zeros(n))


class TestADef2SolutionEquivalence:
    """Equivalence of solutions: adef2 vs additive vs direct at pinned
    rtol=1e-12 on the two-tile-scale fixture (n_tiles=2) and the ~30-tile
    chain fixture -- same solution <= 1e-10 V."""

    @pytest.mark.parametrize('n_tiles', [2, 30])
    def test_adef2_additive_direct_agree(self, n_tiles):
        tile_schur, tile_idx, n = _chain_tiles(n_tiles)
        rng = np.random.default_rng(555)
        b = rng.standard_normal(n)
        S = np.zeros((n, n))
        for tid, idx in tile_idx.items():
            S[np.ix_(idx, idx)] += tile_schur[tid]
        x_direct = np.linalg.solve(S, b)

        solutions = {}
        for mode in ('additive', 'deflated'):
            solver = InterfaceCGSolver(
                n_interface=n, matvec_mode='tilewise',
                tile_schur_complements={k: v.copy() for k, v in tile_schur.items()},
                tile_index_maps=tile_idx,
                preconditioner='two_level', rtol=1e-12, atol=1e-16,
                matvec_threads=1, strict=True, maxiter=5000,
                interface_coarse_apply_mode=mode,
            )
            try:
                solutions[mode] = solver(b)
            finally:
                solver.close()

        for mode, x in solutions.items():
            err = np.max(np.abs(x - x_direct))
            assert err <= 1e-10, f"[{mode}, n_tiles={n_tiles}] err={err:.3e}"
        err_between = np.max(np.abs(solutions['additive'] - solutions['deflated']))
        assert err_between <= 1e-10, (
            f"[n_tiles={n_tiles}] additive vs adef2 disagree by {err_between:.3e}"
        )


class TestADef2IterationSuperiority:
    """Iteration superiority: adef2 iters <= additive iters at n_tiles in
    {30, 60}, strictly fewer at 60 (also strictly fewer at 30, in practice
    -- see the docstring below for the measured numbers).

    Measured (seed 555, rtol=1e-10, atol=1e-16, matvec_threads=1, jacobi+
    PoU base -- chain fixture has no near-null blocks so n_geneo_cols=0
    throughout, isolating the additive-vs-adef2 APPLY MODE comparison from
    GenEO): n_tiles=30: additive=26, adef2=1. n_tiles=60: additive=27,
    adef2=1. The chain fixture's coarse-space column count T' is close to
    n (one PoU column per tile, n = n_tiles + 1) -- a near-degenerate
    regime where the corrected A-DEF2 algorithm's projected matvec removes
    essentially the ENTIRE error in one CG step (see interface_iterative's
    module docstring, "A-DEF2" section, and _deflated_pcg's docstring for
    why this fixture family is a poor STRESS test for iteration counts,
    even though it remains a perfectly good EQUIVALENCE/correctness check
    -- see TestADef2SolutionEquivalence). On realistic-ratio fixtures
    (T'/n ~ 10%, e.g. TestReprojectionDrift's ill-conditioned fixture or
    the netlist_multi_tile real-PDN gate script) the margin is a more
    modest but still real few-percent iteration reduction (measured 127 ->
    125 on netlist_multi_tile's DC solve). This matches §7.9's finding
    that the additive form's warm gain is modest (coarse and fine spaces
    stay coupled) while A-DEF2 removes range(Z) from the iteration
    exactly.

    **Round-3 spec-compliance review caveat**: this superiority is regime-
    dependent, not universal -- see
    :class:`TestADef2IterationRegimeDependence` for a realistic-``T'/n``-
    ratio, PoU-only, ill-conditioned fixture where DEF needs MORE
    iterations than additive. Do not read this class as establishing the
    work package's warm-iteration goal in general; that is the
    coordinator's mi200k measurement to make (task #29).
    """

    @pytest.mark.parametrize('n_tiles', [30, 60])
    def test_adef2_beats_additive(self, n_tiles):
        tile_schur, tile_idx, n = _chain_tiles(n_tiles)
        rng = np.random.default_rng(555)
        b = rng.standard_normal(n)

        iters = {}
        for mode in ('additive', 'deflated'):
            solver = InterfaceCGSolver(
                n_interface=n, matvec_mode='tilewise',
                tile_schur_complements={k: v.copy() for k, v in tile_schur.items()},
                tile_index_maps=tile_idx,
                preconditioner='two_level', rtol=1e-10, atol=1e-16,
                matvec_threads=1, strict=True, maxiter=5000,
                interface_coarse_apply_mode=mode,
            )
            try:
                solver(b)
                iters[mode] = solver.stats['last_cg_iters']
            finally:
                solver.close()

        assert iters['deflated'] <= iters['additive'], iters
        if n_tiles == 60:
            assert iters['deflated'] < iters['additive'], iters


class TestADef2IterationRegimeDependence:
    """Spec-compliance review round 3 (minor finding): the chain fixture
    ``TestADef2IterationSuperiority`` uses has T' close to n (near-
    degenerate -- one PoU column per tile, n = n_tiles + 1), where the
    projected matvec captures almost the entire solution in a single CG
    step regardless of formula details. That is a real correctness/
    equivalence stress case (see ``TestADef2SolutionEquivalence``) but it is
    NOT evidence that ``apply_mode='deflated'`` (the shipped DEF algorithm
    -- see ``interface_iterative.py``'s "A-DEF2 work package" docstring
    section) is generally faster than the additive two-level form at
    realistic ``T'/n`` ratios.

    This test pins the OPPOSITE finding on the repo's own realistic-ratio
    fixture (``_ill_conditioned_jacobi_fixture``: T'=n_tiles=5, n=200, T'/n
    = 2.5%, PoU-only since block_jacobi_max_bytes=1 forces jacobi and this
    fixture has no engineered near-null structure for GenEO to find): DEF
    needs MORE iterations than additive there, at two condition numbers,
    confirming the benefit is regime-dependent (a diagonal-jacobi base with
    a near-degenerate coarse space favors DEF; a small PoU-only coarse
    space on an ill-conditioned system does not). This does NOT contradict
    ``TestADef2IterationSuperiority`` -- both are accurate measurements of
    their respective fixtures -- it exists so nobody reads the chain-
    fixture superiority test as establishing the work package's stated
    warm-iteration objective (<=10 iters/step) in general; that measurement
    is explicitly deferred to the coordinator's mi200k head-to-head (task
    #29), not any unit fixture in this file.
    """

    @pytest.mark.parametrize('cond_log', [9.0, 12.0])
    def test_adef2_needs_more_iterations_than_additive_on_realistic_ratio_fixture(
        self, cond_log,
    ):
        tile_schur, tile_idx, S_extra, S, n = _ill_conditioned_jacobi_fixture(
            cond_log=cond_log,
        )
        rng = np.random.default_rng(11)
        b = rng.standard_normal(n)

        iters = {}
        for mode in ('additive', 'deflated'):
            solver = InterfaceCGSolver(
                n_interface=n, matvec_mode='tilewise',
                tile_schur_complements=tile_schur, tile_index_maps=tile_idx,
                S_extra=S_extra,
                preconditioner='two_level', rtol=1e-9, atol=1e-14,
                matvec_threads=1, strict=False, maxiter=5000,
                block_jacobi_max_bytes=1,
                interface_coarse_apply_mode=mode,
            )
            try:
                solver(b)
                iters[mode] = solver.stats['last_cg_iters']
            finally:
                solver.close()

        # Documented, not merely asserted: if a future coarse-space/apply
        # change flips this relationship, that is itself signal worth
        # re-examining (the work package's superiority claim would then
        # generalize beyond the chain fixture, which would be good news --
        # update this test's docstring rather than deleting it).
        assert iters['deflated'] > iters['additive'], (
            f"expected DEF/adef2 ({iters['deflated']} iters) to need MORE "
            f"iterations than additive ({iters['additive']} iters) on this "
            f"realistic-T'/n-ratio, PoU-only, ill-conditioned fixture "
            f"(cond_log={cond_log}) -- if this now passes with adef2 <= "
            f"additive, the 'regime-dependent, not universally superior' "
            f"finding no longer reproduces; see this class's docstring"
        )


class TestDecoupledGenEOImprovesIterations:
    """Decoupled GenEO (Deliverable 1): with a jacobi base (forced via a
    tiny interface_block_jacobi_max_bytes) and an engineered near-null-
    eigendirection fixture, two_level now reports n_geneo_cols > 0 -- and
    iterations improve vs the PoU-only (geneo_k=0) coarse space.

    Fixture: 6 DISJOINT (non-overlapping) tiles of 10 nodes each, each an
    independent _near_null_spd_block (one genuine near-null mode per tile,
    eigenvalue ratio 1e-6). Disjoint tiles avoid the ownership-submatrix-
    slicing pitfall a chain/overlapping fixture has (restricting a near-
    null block's eigenvector to a PARTIAL row/col set does not generally
    preserve near-nullity) -- with disjoint tiles, "owned" == "the whole
    tile block", so the injected near-null structure survives into the
    block-Jacobi ownership block exactly.

    Measured (seed 1 for blocks, seed 3 for b): PoU-only (geneo_k=0) 58
    iters; PoU+GenEO (geneo_k=4, n_geneo_cols=6 -- all 6 near-null modes
    captured) 13 iters.
    """

    @staticmethod
    def _disjoint_near_null_tiles(n_tiles, block=10, seed=1, small=1e-6):
        rng = np.random.default_rng(seed)
        tile_idx = {}
        tile_schur = {}
        for t in range(n_tiles):
            idx = np.arange(t * block, (t + 1) * block, dtype=np.int32)
            tile_idx[t] = idx
            tile_schur[t] = _near_null_spd_block(block, rng, small=small, high=3.0)
        n = n_tiles * block
        return tile_schur, tile_idx, n

    def test_geneo_active_and_iterations_improve(self):
        tile_schur, tile_idx, n = self._disjoint_near_null_tiles(6)
        rng = np.random.default_rng(3)
        b = rng.standard_normal(n)
        S = np.zeros((n, n))
        for tid, idx in tile_idx.items():
            S[np.ix_(idx, idx)] += tile_schur[tid]
        x_direct = np.linalg.solve(S, b)

        results = {}
        for geneo_k in (0, 4):
            solver = InterfaceCGSolver(
                n_interface=n, matvec_mode='tilewise',
                tile_schur_complements={k: v.copy() for k, v in tile_schur.items()},
                tile_index_maps=tile_idx,
                preconditioner='two_level', rtol=1e-10, atol=1e-16,
                matvec_threads=1, strict=True, maxiter=5000,
                block_jacobi_max_bytes=1,  # force jacobi downgrade
                interface_coarse_geneo_k=geneo_k,
                # This test isolates the GenEO-vs-PoU-only iteration-count
                # comparison (measured numbers in the class docstring) --
                # pin apply_mode='additive' explicitly so the 2026-07-20
                # DEFAULT_APPLY_MODE flip to 'deflated' doesn't also change
                # what's being measured here.
                interface_coarse_apply_mode='additive',
            )
            try:
                assert solver._bj_downgraded is True, (
                    "fixture must trip the byte-budget downgrade for this "
                    "test to exercise the decoupled path"
                )
                x = solver(b)
                rel = (
                    np.max(np.abs(x - x_direct))
                    / max(1e-300, np.max(np.abs(x_direct)))
                )
                results[geneo_k] = {
                    'n_geneo_cols': solver._coarse.n_geneo_cols,
                    'iters': solver.stats['last_cg_iters'],
                    'rel_err': rel,
                }
            finally:
                solver.close()

        assert results[0]['n_geneo_cols'] == 0, "PoU-only baseline must have no GenEO"
        assert results[4]['n_geneo_cols'] == 6, (
            f"expected all 6 disjoint near-null blocks to enrich; got "
            f"{results[4]['n_geneo_cols']}"
        )
        for k, r in results.items():
            assert r['rel_err'] < 1e-6, f"[geneo_k={k}] rel_err={r['rel_err']:.3e}"
        assert results[4]['iters'] < results[0]['iters'], results

    def test_completion_log_says_pou_only_when_no_block_contributes(self, caplog):
        """Finding 12 (round-1 code review) regression: the decoupled
        pass's completion INFO log must condition its "stays PoU+GenEO"
        claim on ``n_geneo_cols > 0`` -- when every block's GenEO call
        finds nothing to enrich (well-conditioned blocks, no near-null
        spectrum below tol), the log must say PoU-only, not falsely claim
        GenEO is active. Uses the well-conditioned chain fixture (no
        engineered near-null structure -- see _near_null_spd_block's
        docstring contrasting the two) so GenEO genuinely contributes
        zero columns even though geneo_k > 0 and the decoupled pass DOES
        run (unlike test_bj_memory_downgrade_geneo_k_zero_still_says_
        disabled, which uses geneo_k=0 so the pass never runs at all)."""
        tile_schur, tile_idx, n = _chain_tiles(6)
        with caplog.at_level(logging.INFO):
            solver = InterfaceCGSolver(
                n_interface=n, matvec_mode='tilewise',
                tile_schur_complements={k: v.copy() for k, v in tile_schur.items()},
                tile_index_maps=tile_idx,
                preconditioner='two_level', matvec_threads=1,
                block_jacobi_max_bytes=1,  # force jacobi downgrade
                # Measurement-driven flip (2026-07-20) changed the default
                # geneo_k to 0, which would skip the decoupled pass
                # entirely -- this test specifically needs the pass to RUN
                # and contribute zero columns, so request GenEO explicitly.
                interface_coarse_geneo_k=4,
            )
        try:
            assert solver._bj_downgraded is True
            assert solver._geneo_pairs == [], (
                "test precondition: the well-conditioned chain fixture "
                "must yield zero GenEO columns"
            )
            decoupled_records = [
                r for r in caplog.records
                if 'Decoupled GenEO:' in r.message
                and 'ownership block(s) contributed' in r.message
            ]
            assert decoupled_records, (
                f"expected the decoupled-pass completion INFO log; got: "
                f"{[r.message for r in caplog.records]}"
            )
            joined = ' '.join(r.message for r in decoupled_records)
            assert 'stays pou-only' in joined.lower(), (
                f"expected PoU-only wording when 0 blocks contributed; "
                f"got: {joined!r}"
            )
            assert 'stays pou+geneo' not in joined.lower(), (
                f"must NOT claim 'stays PoU+GenEO' when nothing "
                f"contributed; got: {joined!r}"
            )
        finally:
            solver.close()

    def test_completion_log_says_pou_plus_geneo_when_a_block_contributes(
        self, caplog,
    ):
        """Sibling positive check: when at least one block DOES
        contribute GenEO columns, the log must say PoU+GenEO (unchanged
        wording from before Finding 12)."""
        tile_schur, tile_idx, n = self._disjoint_near_null_tiles(6)
        with caplog.at_level(logging.INFO):
            solver = InterfaceCGSolver(
                n_interface=n, matvec_mode='tilewise',
                tile_schur_complements={k: v.copy() for k, v in tile_schur.items()},
                tile_index_maps=tile_idx,
                preconditioner='two_level', rtol=1e-10, atol=1e-16,
                matvec_threads=1, strict=True, maxiter=5000,
                block_jacobi_max_bytes=1,  # force jacobi downgrade
                interface_coarse_geneo_k=4,
            )
        try:
            assert solver._bj_downgraded is True
            assert len(solver._geneo_pairs) > 0
            decoupled_records = [
                r for r in caplog.records
                if 'Decoupled GenEO:' in r.message
                and 'ownership block(s) contributed' in r.message
            ]
            assert decoupled_records
            joined = ' '.join(r.message for r in decoupled_records)
            assert 'stays pou+geneo' in joined.lower(), (
                f"expected PoU+GenEO wording when >0 blocks contributed; "
                f"got: {joined!r}"
            )
        finally:
            solver.close()

    def test_block_jacobi_base_path_unchanged_no_double_eigsolve(self):
        """Negative-style check (spec-required): when the base IS
        block_jacobi (budget holds -- today's behaviour), the retained-
        factor loop's own GenEO call is the ONLY eigensolve per block; the
        decoupled path (_extract_geneo_decoupled) must not ALSO run and
        double-count. Extends test_geneo_reuse_path_no_recompute's spy
        pattern in TestGeneoLowestEigenpairs."""
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
            calls.append(1)
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
                # Budget generous -- block_jacobi must NOT downgrade here.
            )
        try:
            assert solver._bj_downgraded is False
            assert len(calls) == 1, (
                f"expected exactly ONE eigensolve for this single block "
                f"(retained-factor loop only, decoupled path must not "
                f"also run); got {len(calls)}"
            )
        finally:
            solver.close()


class TestGeneoDecoupledMemoryGuard:
    """Finding 1 (round-1 code review) regression: the decoupled GenEO
    pass (InterfaceCGSolver._extract_geneo_decoupled, runs on the
    block_jacobi -> jacobi memory-downgrade path) must respect the same
    memory contract the downgrade guarantees -- a per-block byte cap that
    skips oversized blocks WITHOUT ever forming them, and a
    MemoryError-tolerant guard around block formation + factoring that
    skips the block instead of aborting prepare()/__init__."""

    def test_memory_error_during_block_formation_is_tolerated(self, caplog):
        """Negative-test evidence: reverting the MemoryError guard around
        _form_owned_block's call inside _extract_geneo_decoupled makes
        this FAIL with an uncaught MemoryError propagating out of
        InterfaceCGSolver.__init__ instead of completing with a WARNING
        and a PoU-only (n_geneo_cols == 0) coarse space."""
        tile_schur, tile_idx, n = _chain_tiles(6)

        with mock.patch.object(
            InterfaceCGSolver, '_form_owned_block',
            side_effect=MemoryError("simulated OOM"),
        ):
            with caplog.at_level(logging.WARNING):
                solver = InterfaceCGSolver(
                    n_interface=n, matvec_mode='tilewise',
                    tile_schur_complements={
                        k: v.copy() for k, v in tile_schur.items()
                    },
                    tile_index_maps=tile_idx,
                    preconditioner='two_level', matvec_threads=1,
                    block_jacobi_max_bytes=1,  # force jacobi downgrade
                    # Measurement-driven flip (2026-07-20) changed the
                    # default geneo_k to 0, which would skip the decoupled
                    # pass (and thus _form_owned_block) entirely -- this
                    # test needs the pass to actually run into the
                    # MemoryError guard, so request GenEO explicitly.
                    interface_coarse_geneo_k=4,
                )
        try:
            assert solver._bj_downgraded is True
            assert solver._coarse is not None, (
                "the PoU-only coarse space must still be built (base "
                "downgrade is unrelated to the failed GenEO enrichment)"
            )
            assert solver._geneo_pairs == [], (
                "every block's formation raised MemoryError -- GenEO "
                "must have contributed nothing (PoU-only enrichment)"
            )
            assert solver._coarse.n_geneo_cols == 0
            assert any(
                'memoryerror' in r.message.lower() for r in caplog.records
            ), [r.message for r in caplog.records]
        finally:
            solver.close()

    def test_over_cap_block_skipped_without_formation(self, caplog):
        """spy on _form_owned_block: a single ownership block whose
        estimated dense formation+cho/eigh-fallback peak (5*k^2*8 bytes --
        Finding 7, round 2: covers the eigh-fallback path's peak, not just
        the cheaper cho_factor-succeeds 2x) exceeds the per-block cap must
        be skipped WITHOUT ever calling _form_owned_block."""
        k = 2200  # 5*k^2*8 ~= 193.6 MB > the 64 MiB per-block cap floor
        idx = np.arange(k, dtype=np.int32)
        # Content is never read (formation must be skipped entirely) --
        # identity is cheap to allocate and numerically harmless even if
        # this guard regressed and the block WAS formed/factored.
        tile_schur = {'A': np.eye(k) * 5.0}
        tile_idx = {'A': idx}

        calls = []

        def spy_form(*args, **kwargs):
            calls.append(1)
            return None  # never expected to be reached

        with mock.patch.object(
            InterfaceCGSolver, '_form_owned_block', side_effect=spy_form,
        ):
            with caplog.at_level(logging.WARNING):
                solver = InterfaceCGSolver(
                    n_interface=k, matvec_mode='tilewise',
                    tile_schur_complements=tile_schur, tile_index_maps=tile_idx,
                    preconditioner='two_level', matvec_threads=1,
                    block_jacobi_max_bytes=1,  # force jacobi downgrade
                    # Measurement-driven flip (2026-07-20) changed the
                    # default geneo_k to 0, which would skip the decoupled
                    # pass entirely before it ever gets to the per-block
                    # byte-cap check this test exercises -- request GenEO
                    # explicitly.
                    interface_coarse_geneo_k=4,
                )
        try:
            assert solver._bj_downgraded is True
            assert calls == [], (
                f"_form_owned_block must NOT be called for an over-cap "
                f"block (k={k}); got {len(calls)} call(s)"
            )
            assert solver._geneo_pairs == []
            assert any(
                'skipping ownership block' in r.message.lower()
                for r in caplog.records
            ), [r.message for r in caplog.records]
        finally:
            solver.close()


class TestBaseBlockJacobiMemoryErrorPropagates:
    """Round-2 code review finding 2 (regression in round-1's Finding 1
    fix): the MemoryError-tolerant guard belongs ONLY to the DECOUPLED
    GenEO pass (:class:`TestGeneoDecoupledMemoryGuard` above,
    ``_extract_geneo_decoupled``) -- that pass is an optional enrichment
    step, and skip-on-OOM is the documented, intentional degrade for it.
    The shared ``_form_owned_block``/``_cho_or_eigh_with_geneo`` helper
    refactor also (incorrectly) wrapped ``_build_block_jacobi``'s
    PRE-EXISTING retained-factor loop -- which builds the BASE
    preconditioner itself, not an enrichment -- in the identical
    ``except MemoryError: continue`` guard, silently converting a
    coordinator-under-memory-pressure crash (which used to abort
    prepare()/factor() immediately, the correct fail-fast signal to raise
    the budget or retile) into "skip this block, use identity
    preconditioning for its nodes" -- masking the real OOM behind a slow
    CG stall or strict-mode non-convergence error, potentially hours into
    a run. Base-preconditioner block formation/factoring must stay
    fail-fast, exactly as it was before Finding 1 (round 1).

    Negative-test evidence (see the round-2 fix agent's final report for
    the transcript): temporarily re-wrapping either call site below in
    ``except MemoryError: continue`` (restoring the round-1 regression)
    makes BOTH tests in this class FAIL (``pytest.raises(MemoryError)``
    catches nothing -- ``InterfaceCGSolver.__init__`` returns normally
    instead)."""

    def test_memory_error_during_base_block_formation_propagates(self):
        tile_schur, tile_idx, n = _chain_tiles(6)
        with mock.patch.object(
            InterfaceCGSolver, '_form_owned_block',
            side_effect=MemoryError("simulated OOM"),
        ):
            with pytest.raises(MemoryError):
                InterfaceCGSolver(
                    n_interface=n, matvec_mode='tilewise',
                    tile_schur_complements={
                        k: v.copy() for k, v in tile_schur.items()
                    },
                    tile_index_maps=tile_idx,
                    preconditioner='block_jacobi', matvec_threads=1,
                    # Deliberately NOT forcing the jacobi memory-downgrade
                    # (contrast TestGeneoDecoupledMemoryGuard, which does)
                    # -- this fixture is small enough that the retained-
                    # factor loop under test actually runs.
                )

    def test_memory_error_during_base_block_factoring_propagates(self):
        tile_schur, tile_idx, n = _chain_tiles(6)
        with mock.patch.object(
            InterfaceCGSolver, '_cho_or_eigh_with_geneo',
            side_effect=MemoryError("simulated OOM"),
        ):
            with pytest.raises(MemoryError):
                InterfaceCGSolver(
                    n_interface=n, matvec_mode='tilewise',
                    tile_schur_complements={
                        k: v.copy() for k, v in tile_schur.items()
                    },
                    tile_index_maps=tile_idx,
                    preconditioner='block_jacobi', matvec_threads=1,
                )


class TestHandRolledPCGSemanticsParity:
    """Hand-rolled PCG semantics parity: same fixture solved by the scipy
    path (additive) and the hand-rolled loop (adef2, a REAL Z, not zero
    columns) -- stats keys present, warm start works, strict-mode
    RuntimeError fires at tiny maxiter with cg_failed stats set,
    progress_every emits, reset_warm_start clears, rtol/atol acceptance
    matches the documented criterion (a case straddling atol)."""

    @staticmethod
    def _solver(mode, **kw):
        tile_schur, tile_idx, n = _chain_tiles(20)
        defaults = dict(
            n_interface=n, matvec_mode='tilewise',
            tile_schur_complements={k: v.copy() for k, v in tile_schur.items()},
            tile_index_maps=tile_idx,
            preconditioner='two_level', matvec_threads=1,
            interface_coarse_apply_mode=mode,
        )
        defaults.update(kw)
        return InterfaceCGSolver(**defaults), n

    def test_stats_keys_present_both_modes(self):
        for mode in ('additive', 'deflated'):
            solver, n = self._solver(mode, rtol=1e-10, atol=1e-16, maxiter=2000)
            try:
                rng = np.random.default_rng(1)
                solver(rng.standard_normal(n))
                for key in (
                    'last_cg_iters', 'last_cg_time_s', 'last_cg_info',
                    'total_cg_iters', 'total_cg_solves', 'cg_failed',
                    'total_cg_failures',
                ):
                    assert key in solver.stats, f"[{mode}] missing stats key {key!r}"
                assert solver.stats['cg_failed'] is False
            finally:
                solver.close()

    def test_warm_start_second_identical_solve_not_more_iters(self):
        solver, n = self._solver('deflated', rtol=1e-10, atol=1e-16, maxiter=5000)
        try:
            rng = np.random.default_rng(2)
            b = rng.standard_normal(n)
            solver(b)
            iters1 = solver.stats['last_cg_iters']
            solver(b)  # warm-started from the same solution -> should converge fast
            iters2 = solver.stats['last_cg_iters']
            assert iters2 <= iters1, (iters1, iters2)
        finally:
            solver.close()

    def test_strict_mode_raises_with_cg_failed_stats_at_tiny_maxiter(self):
        # Round-2 spec-compliance review fix: the chain fixture's coarse
        # space captures almost the WHOLE solution in a single adef2
        # iteration (T' close to n -- see TestADef2IterationSuperiority's
        # docstring, "n_tiles=X: adef2=1"), so maxiter=1 is not actually
        # insufficient budget for THAT fixture once the maxiter-th
        # iteration's convergence is correctly checked (the finding-3 fix
        # below) -- it genuinely converges in 1 iteration. Use the
        # deliberately long ill-conditioned fixture (needs ~600+ iterations
        # at this rtol, per its own docstring) so maxiter=1 is a real
        # insufficient-budget case regardless of that fix.
        tile_schur, tile_idx, S_extra, S, n = _ill_conditioned_jacobi_fixture()
        solver = InterfaceCGSolver(
            n_interface=n, matvec_mode='tilewise',
            tile_schur_complements=tile_schur, tile_index_maps=tile_idx,
            S_extra=S_extra,
            preconditioner='two_level', rtol=1e-14, atol=1e-18,
            matvec_threads=1, strict=True, maxiter=1,
            block_jacobi_max_bytes=1,  # force jacobi -- more iterations
            interface_coarse_apply_mode='deflated',
        )
        try:
            rng = np.random.default_rng(3)
            b = rng.standard_normal(n)
            with pytest.raises(RuntimeError, match='did not converge'):
                solver(b)
            assert solver.stats['cg_failed'] is True
            assert solver.stats['total_cg_failures'] == 1
            assert 'last_cg_rel_residual' in solver.stats
            assert solver.stats['last_cg_iters'] <= 1
        finally:
            solver.close()

    def test_progress_every_emits_log(self, caplog):
        # Chain fixtures converge in ~1 iteration under the corrected
        # A-DEF2 algorithm (T' close to n -- see the module docstring's
        # "A-DEF2" section) -- too short to ever reach progress_every.
        # Use the deliberately-long ill-conditioned fixture instead.
        tile_schur, tile_idx, S_extra, S, n = _ill_conditioned_jacobi_fixture()
        rng = np.random.default_rng(4)
        b = rng.standard_normal(n)
        solver = InterfaceCGSolver(
            n_interface=n, matvec_mode='tilewise',
            tile_schur_complements=tile_schur, tile_index_maps=tile_idx,
            S_extra=S_extra,
            preconditioner='two_level', rtol=1e-9, atol=1e-14,
            matvec_threads=1, strict=True, maxiter=3000,
            block_jacobi_max_bytes=1,  # force jacobi -- more iterations
            interface_coarse_apply_mode='deflated',
        )
        solver.progress_every = 20
        try:
            with caplog.at_level(logging.INFO):
                solver(b)
            assert solver.stats['last_cg_iters'] > 20, (
                "fixture must run enough iterations to reach progress_every"
            )
            assert any(
                'InterfaceCG progress' in r.message for r in caplog.records
            ), "expected at least one progress log line"
        finally:
            solver.close()

    def test_reset_warm_start_clears_x0(self):
        solver, n = self._solver('deflated', rtol=1e-10, atol=1e-16, maxiter=5000)
        try:
            rng = np.random.default_rng(5)
            b = rng.standard_normal(n)
            solver(b)
            assert solver._x0 is not None
            solver.reset_warm_start()
            assert solver._x0 is None
        finally:
            solver.close()

    def test_rtol_atol_acceptance_matches_documented_criterion(self):
        """Construct a case straddling atol: rtol effectively irrelevant
        (tiny rtol), atol set exactly at the true solution's residual norm
        scale so convergence is governed by atol alone -- both apply modes
        must accept/reject identically per ||r|| <= max(rtol*||b||, atol)."""
        tile_schur, tile_idx, n = _chain_tiles(10)
        rng = np.random.default_rng(6)
        b = rng.standard_normal(n)
        S = np.zeros((n, n))
        for tid, idx in tile_idx.items():
            S[np.ix_(idx, idx)] += tile_schur[tid]
        x_direct = np.linalg.solve(S, b)

        for mode in ('additive', 'deflated'):
            solver = InterfaceCGSolver(
                n_interface=n, matvec_mode='tilewise',
                tile_schur_complements={k: v.copy() for k, v in tile_schur.items()},
                tile_index_maps=tile_idx,
                preconditioner='two_level', rtol=1e-300, atol=1e-6,
                matvec_threads=1, strict=True, maxiter=5000,
                interface_coarse_apply_mode=mode,
            )
            try:
                x = solver(b)
                r_true = np.linalg.norm(b - S @ x)
                assert r_true <= 1e-6 * 10, (
                    f"[{mode}] converged residual {r_true:.3e} grossly "
                    f"exceeds the atol=1e-6 acceptance bound"
                )
            finally:
                solver.close()


class TestHistoryHygieneRunsBeforeStrictRaise:
    """Round-2 code review finding 8 (PLAUSIBLE, CONFIRMED by code
    inspection): the failed-solve warm-start/extrapolation history
    hygiene (Finding 2, round 1 -- clear ``_x_hist_prev``/
    ``_x_hist_prev2``, reseed ``_x0`` from the best iterate) used to run
    AFTER the ``strict`` mode ``raise RuntimeError`` in
    ``InterfaceCGSolver.__call__``. Since ``raise`` immediately exits the
    function, with ``strict=True`` (the production default) that
    hygiene NEVER ran on a failed solve: a caller that catches the
    RuntimeError and retries (the documented recovery path short of
    ``strict=False``) would re-solve from the SAME pre-failure
    ``_x0``/history that just failed, and a LATER converged solve could
    extrapolate ``2*x_prev - x_prev2`` across the failed step using
    stale pre-failure history -- exactly what the hygiene exists to
    prevent.

    Fix: reorder so the hygiene runs unconditionally (right after the
    stats/rel-residual bookkeeping) BEFORE the ``if self.strict: raise
    ...`` check -- the raise still fires with the identical message/stats
    either way.
    """

    def test_strict_true_failure_clears_history_before_raise(self):
        tile_schur, tile_idx, S_extra, S, n = _ill_conditioned_jacobi_fixture()
        solver = InterfaceCGSolver(
            n_interface=n, matvec_mode='tilewise',
            tile_schur_complements=tile_schur, tile_index_maps=tile_idx,
            S_extra=S_extra,
            preconditioner='two_level', rtol=1e-14, atol=1e-18,
            matvec_threads=1, strict=True, maxiter=1,
            block_jacobi_max_bytes=1,  # force jacobi -- more iterations
            interface_coarse_apply_mode='deflated',
        )
        try:
            rng = np.random.default_rng(3)
            b = rng.standard_normal(n)

            # Pre-seed history so we can positively verify it gets
            # CLEARED (not merely "never populated") -- distinguishes
            # "hygiene ran" from "hygiene was a no-op because there was
            # nothing to clear".
            solver._x_hist_prev = np.ones(n)
            solver._x_hist_prev2 = np.full(n, 2.0)
            solver._x0 = np.full(n, 3.0)

            with pytest.raises(RuntimeError, match='did not converge'):
                solver(b)

            assert solver.stats['cg_failed'] is True
            assert solver._x_hist_prev is None, (
                "warm-start extrapolation history's `_x_hist_prev` "
                "survived a strict=True failure -- the hygiene that "
                "clears it did not run before the raise"
            )
            assert solver._x_hist_prev2 is None, (
                "warm-start extrapolation history's `_x_hist_prev2` "
                "survived a strict=True failure -- the hygiene that "
                "clears it did not run before the raise"
            )
            # _x0 is reseeded from the best (failed) iterate -- no longer
            # the pre-solve sentinel value 3.0 in every component.
            assert not np.array_equal(solver._x0, np.full(n, 3.0)), (
                "_x0 was not reseeded from the best iterate before the "
                "raise -- the hygiene that does this did not run"
            )
        finally:
            solver.close()


class TestDeflatedPCGRecoverySkippedWithoutProgressLogging:
    """Code-quality review round-1 (perf finding): ``_deflated_pcg`` used to
    call ``coarse.apply`` (two dense O(n*T') GEMVs) inside ``_recover_x()``
    once per completed iteration solely to feed the ``callback`` -- wasted
    work whenever ``InterfaceCGSolver.progress_every`` is at its default (0,
    disabled), since ``_callback`` only ever reads its ``xk`` argument
    inside the ``if _progress_every and iters[0] % _progress_every == 0``
    guard. Verify the fix: with ``progress_every`` left at its default,
    ``coarse.apply`` is called O(1) times per solve (only at the final
    accept), not once per CG iteration -- on a fixture that runs enough
    iterations (from ``_ill_conditioned_jacobi_fixture``, forced jacobi
    base) that O(iters) vs O(1) is unambiguous."""

    def test_recovery_not_called_per_iteration_when_progress_disabled(self):
        tile_schur, tile_idx, S_extra, S, n = _ill_conditioned_jacobi_fixture()
        rng = np.random.default_rng(4)
        b = rng.standard_normal(n)
        solver = InterfaceCGSolver(
            n_interface=n, matvec_mode='tilewise',
            tile_schur_complements=tile_schur, tile_index_maps=tile_idx,
            S_extra=S_extra,
            preconditioner='two_level', rtol=1e-9, atol=1e-14,
            matvec_threads=1, strict=True, maxiter=3000,
            block_jacobi_max_bytes=1,  # force jacobi -- more iterations
            interface_coarse_apply_mode='deflated',
        )
        try:
            coarse = solver._coarse
            assert coarse is not None and coarse.SZ is not None
            # CoarseSpace uses __slots__, so the instance's bound method
            # can't be shadowed directly -- patch the unbound method on the
            # class (affects every CoarseSpace instance, but this test owns
            # its own solver/coarse space, so that's fine).
            apply_calls = [0]
            orig_apply = ic.CoarseSpace.apply

            def counting_apply(self, r):
                apply_calls[0] += 1
                return orig_apply(self, r)

            with mock.patch.object(ic.CoarseSpace, 'apply', counting_apply):
                x = solver(b)

            iters = solver.stats['last_cg_iters']
            assert iters > 20, (
                "fixture must run enough iterations for O(iters) vs O(1) "
                f"to be unambiguous; only got {iters}"
            )
            assert apply_calls[0] <= 3, (
                f"coarse.apply was called {apply_calls[0]} times over "
                f"{iters} CG iterations with progress logging disabled -- "
                f"expected O(1) (only at the final accept), not O(iters); "
                f"the per-iteration recovery-for-the-callback is being "
                f"computed and discarded again"
            )
            # The fix must not change the actual solution.
            true_r = float(np.linalg.norm(b - S @ x))
            assert true_r <= max(1e-9, 1e-9 * float(np.linalg.norm(b))), (
                f"solution changed by the recovery-skip fix: true residual "
                f"{true_r:.3e}"
            )
        finally:
            solver.close()


class TestNegativeProgressEveryDisabled:
    """Finding 3 (round-1 code review) regression: ``progress_every <= 0``
    must be treated as DISABLED in BOTH ``InterfaceCGSolver._callback``'s
    logging gate and ``_deflated_pcg``'s own recovery gate -- a negative
    value is truthy in Python (the old ``if _progress_every:`` gate), and
    ``k % -m == 0`` whenever ``k`` is a multiple of ``m``, so the old
    callback gate entered its logging branch on a negative
    ``progress_every`` while ``_deflated_pcg``'s gate (``progress_every >
    0``) still delivered ``xk=None`` for every iteration --
    ``self._linear_op.matvec(None)`` then raised ``TypeError``, aborting
    the whole deflated solve. Negative-test evidence: reverting
    ``_callback``'s gate from ``_progress_every > 0`` back to the plain
    truthiness check ``_progress_every`` reproduces the crash on the
    deflated path (the additive/scipy path was never affected -- scipy's
    ``cg`` always passes a real ``xk``)."""

    def test_negative_progress_every_deflated_neither_logs_nor_crashes(self, caplog):
        tile_schur, tile_idx, S_extra, S, n = _ill_conditioned_jacobi_fixture()
        rng = np.random.default_rng(4)
        b = rng.standard_normal(n)
        solver = InterfaceCGSolver(
            n_interface=n, matvec_mode='tilewise',
            tile_schur_complements=tile_schur, tile_index_maps=tile_idx,
            S_extra=S_extra,
            preconditioner='two_level', rtol=1e-9, atol=1e-14,
            matvec_threads=1, strict=True, maxiter=3000,
            block_jacobi_max_bytes=1,  # force jacobi -- more iterations
            interface_coarse_apply_mode='deflated',
        )
        solver.progress_every = -200
        try:
            assert solver._coarse is not None and solver._coarse.SZ is not None
            with caplog.at_level(logging.INFO):
                x = solver(b)  # must NOT raise TypeError
            assert solver.stats['last_cg_iters'] > 20, (
                "fixture must run enough iterations for the negative-"
                "modulus branch to actually be exercised"
            )
            assert not any(
                'InterfaceCG progress' in r.message for r in caplog.records
            ), "progress_every <= 0 must disable logging, not merely avoid a crash"
            true_r = float(np.linalg.norm(b - S @ x))
            assert true_r <= max(1e-9, 1e-9 * float(np.linalg.norm(b))), (
                f"solution incorrect: true residual {true_r:.3e}"
            )
        finally:
            solver.close()

    def test_negative_progress_every_additive_scipy_path_neither_logs_nor_crashes(
        self, caplog,
    ):
        """Sibling no-regression check: the scipy/additive path was never
        deflated-crash-affected (scipy always passes a real ``xk``), but
        must still honor "<= 0 disables logging"."""
        tile_schur, tile_idx, S_extra, S, n = _ill_conditioned_jacobi_fixture()
        rng = np.random.default_rng(4)
        b = rng.standard_normal(n)
        solver = InterfaceCGSolver(
            n_interface=n, matvec_mode='tilewise',
            tile_schur_complements=tile_schur, tile_index_maps=tile_idx,
            S_extra=S_extra,
            preconditioner='two_level', rtol=1e-9, atol=1e-14,
            matvec_threads=1, strict=True, maxiter=3000,
            block_jacobi_max_bytes=1,
            interface_coarse_apply_mode='additive',
        )
        solver.progress_every = -200
        try:
            with caplog.at_level(logging.INFO):
                x = solver(b)  # must not raise
            assert not any(
                'InterfaceCG progress' in r.message for r in caplog.records
            )
            true_r = float(np.linalg.norm(b - S @ x))
            assert true_r <= max(1e-9, 1e-9 * float(np.linalg.norm(b)))
        finally:
            solver.close()


class TestApplyModeLabeling:
    """A-DEF2 work package -- ratified naming (coordinator selection: DEF
    ships as ``apply_mode='deflated'``; see the module docstring's "A-DEF2
    work package" section for the full head-to-head record against true
    A-DEF2). An earlier revision of this module shipped this same DEF
    algorithm under the setting value ``'adef2'``, which required runtime
    self-disclosure (a module constant ``ADEF2_ACTUAL_ALGORITHM``, a
    ``[adef2:def1]`` label tag, a one-time WARNING) because the name
    claimed an algorithm that wasn't actually running. Renaming the setting
    to ``'deflated'`` removes the mismatch, so that disclosure machinery no
    longer exists -- these tests pin the SIMPLER resulting contract: the
    label carries a plain ``[deflated]`` tag (only when the deflated apply
    genuinely took effect -- SZ retained), ``stats['apply_algorithm']`` is
    the plain string ``'deflated'``/``'additive'``, and there is no
    disclosure WARNING to log (nothing to disclose)."""

    @staticmethod
    def _solver(mode, **kw):
        tile_schur, tile_idx, n = _chain_tiles(20)
        defaults = dict(
            n_interface=n, matvec_mode='tilewise',
            tile_schur_complements={k: v.copy() for k, v in tile_schur.items()},
            tile_index_maps=tile_idx,
            preconditioner='two_level', matvec_threads=1,
            interface_coarse_apply_mode=mode,
        )
        defaults.update(kw)
        return InterfaceCGSolver(**defaults), n

    def test_deflated_label_carries_plain_tag(self):
        solver, n = self._solver('deflated', rtol=1e-10, atol=1e-16, maxiter=2000)
        try:
            assert solver._coarse is not None and solver._coarse.SZ is not None, (
                "fixture must actually retain SZ for this labeling check "
                "to be meaningful"
            )
            label = solver.preconditioner_label
            assert '[deflated]' in label, (
                f"label must carry the plain [deflated] tag: {label!r}"
            )
            # No 'adef2' token anywhere -- the setting/label must never
            # claim the literal A-DEF2 formula ran (it didn't; DEF did).
            assert 'adef2' not in label.lower(), (
                f"label must not mention 'adef2' at all: {label!r}"
            )
        finally:
            solver.close()

    def test_additive_label_unchanged_byte_identical_format(self):
        """Negative control: the additive path's label format must NOT
        gain any tag (spec's 'byte-identical when apply_mode=additive'
        requirement) -- only 'deflated' gets one."""
        solver, n = self._solver('additive', rtol=1e-10, atol=1e-16, maxiter=2000)
        try:
            label = solver.preconditioner_label
            assert '[' not in label, (
                f"additive label must carry no bracketed tag at all: {label!r}"
            )
        finally:
            solver.close()

    def test_stats_apply_algorithm_is_deflated_when_deflated_dispatches(self):
        solver, n = self._solver('deflated', rtol=1e-10, atol=1e-16, maxiter=2000)
        try:
            rng = np.random.default_rng(11)
            solver(rng.standard_normal(n))
            assert solver.stats['apply_algorithm'] == 'deflated'
        finally:
            solver.close()

    def test_stats_apply_algorithm_is_additive_for_additive_mode(self):
        solver, n = self._solver('additive', rtol=1e-10, atol=1e-16, maxiter=2000)
        try:
            rng = np.random.default_rng(12)
            solver(rng.standard_normal(n))
            assert solver.stats['apply_algorithm'] == 'additive'
        finally:
            solver.close()

    def test_no_deviation_warning_logged_for_either_mode(self, caplog):
        """Negative-style check: with the honest 'deflated' name, there is
        nothing to self-disclose -- neither mode should log a
        requested-vs-actual-algorithm mismatch WARNING (the old
        '...algorithm actually dispatched...' message is gone entirely,
        not merely silenced)."""
        for mode in ('additive', 'deflated'):
            solver, n = self._solver(mode, rtol=1e-10, atol=1e-16, maxiter=2000)
            try:
                rng = np.random.default_rng(13)
                b = rng.standard_normal(n)
                with caplog.at_level(logging.WARNING):
                    solver(b)
                    solver(b)
                assert not any(
                    "algorithm actually dispatched" in r.message
                    for r in caplog.records
                ), f"[{mode}] unexpected disclosure-style WARNING logged"
            finally:
                solver.close()


def _literal_spec_adef2_pcg(matvec, base_apply, coarse, b, x0, rtol, atol,
                             maxiter, callback=None):
    """Independent re-implementation of the work-package spec's LITERAL
    in-line A-DEF2 formula (Design section 2), for regression-evidence
    purposes ONLY -- never used by production code (see
    ``_deflated_pcg``'s docstring in ``interface_iterative.py`` for why the
    shipped algorithm -- ``apply_mode='deflated'`` -- is the "DEF" taxonomy
    member instead):

        x0' = Q b + (I - Q S) x0
        M^-1_ADEF2 r = M_base^-1 (r - S Q r) + Q r

    with a PLAIN (unprojected) matvec ``S p`` and a standard 3-term PCG
    recurrence -- exactly as literally written in the spec, with no
    projected-matvec trick. Written independently from
    ``interface_iterative._deflated_pcg`` (not copy-pasted) so this test
    class is a genuine, from-scratch re-verification of the round-1
    finding, not a repetition of the implementation's own claim.
    """
    b = np.asarray(b, dtype=np.float64)
    n = b.shape[0]
    bnrm2 = float(np.linalg.norm(b))
    atol_eff = max(float(atol), float(rtol) * bnrm2)
    if bnrm2 == 0.0:
        return np.zeros(n, dtype=np.float64), 0

    Qb, _ = coarse.apply_with_SQ(b)
    if x0 is None:
        x = Qb.copy()
    else:
        x0 = np.asarray(x0, dtype=np.float64)
        Q_Sx0, _ = coarse.apply_with_SQ(matvec(x0))
        x = Qb + (x0 - Q_Sx0)  # Q b + (I - Q S) x0

    r = b - matvec(x)
    rho_prev = None
    p = None
    for _ in range(maxiter):
        rnorm = float(np.linalg.norm(r))
        if rnorm <= atol_eff:
            return x, 0
        Qr, SQr = coarse.apply_with_SQ(r)
        z = base_apply(r - SQr) + Qr  # M_base^-1 (r - S Q r) + Q r
        rho_cur = float(np.dot(r, z))
        if rho_prev is not None and rho_prev != 0.0:
            p = z + (rho_cur / rho_prev) * p
        else:
            p = z.copy()
        Sp = matvec(p)  # PLAIN, unprojected matvec
        pSp = float(np.dot(p, Sp))
        if pSp == 0.0:
            return x, (0 if rnorm <= atol_eff else maxiter)
        alpha = rho_cur / pSp
        x = x + alpha * p
        r = r - alpha * Sp
        rho_prev = rho_cur
        if callback is not None:
            callback(x)
    return x, maxiter


class TestDeflatedPcgReprojectEveryDefault:
    """Regression: ``_deflated_pcg(reproject_every=None)`` must resolve
    dynamically to ``interface_coarse.DEFAULT_DEFLATED_REPROJECT_EVERY``.

    The 'adef2' -> 'deflated' rename left the ``None`` branch referencing
    the deleted ``DEFAULT_ADEF2_REPROJECT_EVERY`` name -- latent because
    every production/test call site passed an explicit int, so the branch
    only crashed for callers relying on the documented default (final
    confirmation-review finding). This pins both the default resolution
    and its dynamic (call-time, monkeypatchable) read.
    """

    def _solve(self, reproject_every):
        tile_schur, tile_idx, n = _chain_tiles(10)
        S = np.zeros((n, n))
        for tid, idx in tile_idx.items():
            S[np.ix_(idx, idx)] += tile_schur[tid]
        coarse = ic.build_coarse_space(
            lambda X: S @ X, tile_idx, n=n, retain_sz=True,
        )
        assert coarse is not None and coarse.SZ is not None
        d = np.diag(S).copy()
        rng = np.random.default_rng(3)
        b = rng.standard_normal(n)
        x, info = ii._deflated_pcg(
            lambda v: S @ v, lambda r: r / d, coarse, b, None,
            rtol=1e-10, atol=1e-14, maxiter=2000,
            reproject_every=reproject_every,
        )
        assert info == 0
        assert np.linalg.norm(b - S @ x) <= 1e-10 * np.linalg.norm(b)
        return x

    @pytest.mark.unit
    def test_none_resolves_to_module_default(self):
        # Pre-fix: AttributeError (module has no DEFAULT_ADEF2_REPROJECT_EVERY).
        self._solve(reproject_every=None)

    @pytest.mark.unit
    def test_none_default_is_read_dynamically(self, monkeypatch):
        # Finding-9 pattern: the default must be read at call time, so a
        # monkeypatch of the module constant takes effect for None callers.
        monkeypatch.setattr(ic, 'DEFAULT_DEFLATED_REPROJECT_EVERY', 3)
        self._solve(reproject_every=None)
        monkeypatch.delattr(ic, 'DEFAULT_DEFLATED_REPROJECT_EVERY')
        with pytest.raises(AttributeError):
            self._solve(reproject_every=None)


class TestLiteralADef2FormulaIndependentlyReverified:
    """Spec-compliance review round 2 (major finding): the reviewer noted
    the implementation's claim that the spec's literal A-DEF2 formula
    stalls on real fixtures "was NOT independently re-verified" by the
    reviewer, and asked for a coordinator decision between reverting to
    the literal formula or keeping the DEF substitute. This test class IS
    that independent re-verification (round-2 fix pass): a from-scratch
    re-implementation of the literal formula (``_literal_spec_adef2_pcg``
    above, not a copy of ``_deflated_pcg``'s internals) is run against the
    SAME fixtures/coarse-space machinery the shipped DEF algorithm uses,
    confirming the literal formula fails to converge while DEF/the shipped
    ``_deflated_pcg`` succeeds on the identical problem.

    This pins the round-1/round-2 design decision as a regression check:
    if the literal formula's convergence behaviour on these fixtures ever
    changes (e.g. a future coarse-space refactor), this test's failure is
    the signal to re-open the "should apply_mode='deflated' dispatch the
    literal formula instead of DEF" question -- it is not merely
    documentation, it is verified evidence.
    """

    def test_literal_formula_stalls_on_ill_conditioned_jacobi_fixture(self):
        tile_schur, tile_idx, S_extra, S, n = _ill_conditioned_jacobi_fixture()
        rng = np.random.default_rng(11)
        b = rng.standard_normal(n)
        rtol, atol = 1e-9, 1e-14

        solver = InterfaceCGSolver(
            n_interface=n, matvec_mode='tilewise',
            tile_schur_complements=tile_schur, tile_index_maps=tile_idx,
            S_extra=S_extra,
            preconditioner='two_level', rtol=rtol, atol=atol,
            matvec_threads=1, strict=False, maxiter=3000,
            block_jacobi_max_bytes=1,  # force jacobi base
            interface_coarse_apply_mode='deflated',
        )
        matvec = solver._linear_op.matvec
        base_apply = solver._M_base_apply
        coarse = solver._coarse
        assert coarse.SZ is not None
        try:
            # Shipped algorithm (DEF, projected matvec): converges.
            x_shipped, info_shipped = _deflated_pcg(
                matvec, base_apply, coarse, b, None, rtol=rtol, atol=atol,
                maxiter=3000, reproject_every=50,
            )
            assert info_shipped == 0, (
                f"sanity check failed: shipped DEF algorithm itself did "
                f"not converge (info={info_shipped}) on this fixture"
            )
            true_r_shipped = float(np.linalg.norm(b - matvec(x_shipped)))
            assert true_r_shipped <= rtol * float(np.linalg.norm(b))

            # Literal spec formula (unprojected matvec, x0'/M^-1_ADEF2 as
            # written): independently re-verified to STALL -- never gets
            # anywhere close to rtol within the same maxiter budget.
            x_literal, info_literal = _literal_spec_adef2_pcg(
                matvec, base_apply, coarse, b, None, rtol=rtol, atol=atol,
                maxiter=3000,
            )
            true_r_literal = float(np.linalg.norm(b - matvec(x_literal)))
            rel_r_literal = true_r_literal / float(np.linalg.norm(b))
            assert info_literal != 0, (
                "literal A-DEF2 formula unexpectedly converged (info=0) -- "
                "the round-1/round-2 'formula stalls' finding no longer "
                "reproduces; re-open the apply_mode='deflated' dispatch "
                "decision (see this class's docstring)"
            )
            # Generous bound (the shipped algorithm reaches ~1e-9 relative;
            # the literal formula must be at least several orders of
            # magnitude short of that after the SAME iteration budget --
            # not merely "didn't quite converge").
            assert rel_r_literal > 1e-3, (
                f"literal A-DEF2 formula's true relative residual "
                f"({rel_r_literal:.3e}) is surprisingly close to "
                f"converged -- re-examine whether it truly stalls"
            )
        finally:
            solver.close()

    def test_literal_formula_stalls_on_chain_fixture(self):
        """Negative-style check on the OTHER fixture family: even the
        near-degenerate chain fixture (where DEF converges in ~1
        iteration, see TestADef2IterationSuperiority) does not rescue the
        literal formula -- it stalls flat rather than merely converging
        slower, confirming this is not fixture-selection bias."""
        tile_schur, tile_idx, n = _chain_tiles(60)
        rng = np.random.default_rng(555)
        b = rng.standard_normal(n)
        rtol, atol = 1e-10, 1e-16

        solver = InterfaceCGSolver(
            n_interface=n, matvec_mode='tilewise',
            tile_schur_complements={k: v.copy() for k, v in tile_schur.items()},
            tile_index_maps=tile_idx,
            preconditioner='two_level', rtol=rtol, atol=atol,
            matvec_threads=1, strict=False, maxiter=2000,
            interface_coarse_apply_mode='deflated',
        )
        matvec = solver._linear_op.matvec
        base_apply = solver._M_base_apply
        coarse = solver._coarse
        assert coarse.SZ is not None
        try:
            x_shipped, info_shipped = _deflated_pcg(
                matvec, base_apply, coarse, b, None, rtol=rtol, atol=atol,
                maxiter=2000, reproject_every=50,
            )
            assert info_shipped == 0

            x_literal, info_literal = _literal_spec_adef2_pcg(
                matvec, base_apply, coarse, b, None, rtol=rtol, atol=atol,
                maxiter=2000,
            )
            assert info_literal != 0, (
                "literal A-DEF2 formula unexpectedly converged on the "
                "chain fixture too -- re-open the apply_mode='deflated' "
                "dispatch decision"
            )
            true_r_literal = float(np.linalg.norm(b - matvec(x_literal)))
            rel_r_literal = true_r_literal / float(np.linalg.norm(b))
            assert rel_r_literal > 1e-2, (
                f"literal formula's relative residual ({rel_r_literal:.3e}) "
                f"is closer to converged than the documented flat stall"
            )
        finally:
            solver.close()


def _transpose_corrected_adef2_pcg(matvec, base_apply, coarse, b, x0, rtol,
                                    atol, maxiter):
    """Spec-compliance review round 3: a THIRD independently-derived
    candidate, checking whether the spec's literal formula's stall (see
    ``_literal_spec_adef2_pcg`` above) was merely a projector-transpose
    slip rather than a fundamental limitation of the "fold Q into the
    preconditioner, keep a plain matvec" family.

    The published Tang/Nabben/Vuik/Erlangga taxonomy defines ``P = I - S
    Q`` and its transpose ``P^T = I - Q S`` (NOT equal in general, since
    ``S`` and ``Q`` do not commute), and places A-DEF2's preconditioner as
    ``M_AD2^-1 = M_base^-1 P^T + Q`` -- ``P^T`` (``I - Q S``, Q applied
    to ``S r`` FIRST) ahead of ``M_base^-1``. The work-package spec's
    Design section 2 instead wrote ``M_base^-1 (r - S Q r) + Q r`` --
    ``P`` (``I - S Q``, S applied to ``Q r`` first), not ``P^T``. This
    function swaps in the literature-correct ``P^T`` ordering (``Q(S r)``
    via ``coarse.apply(matvec(r))``, vs. the spec's ``S(Q r)`` via
    ``coarse.apply_with_SQ``) to test whether that transpose distinction
    -- not merely stalling vs. diverging -- explains the round-1/round-2
    rejection, still inside a standard 3-term PCG recurrence with a plain
    (unprojected) matvec, exactly as literal A-DEF1/A-DEF2 formulas are
    conventionally written.

    Written independently of ``_deflated_pcg`` and ``_literal_spec_adef2_
    pcg`` (fix-agent round-3 verification, not a copy of either).
    """
    b = np.asarray(b, dtype=np.float64)
    n = b.shape[0]
    bnrm2 = float(np.linalg.norm(b))
    atol_eff = max(float(atol), float(rtol) * bnrm2)
    if bnrm2 == 0.0:
        return np.zeros(n, dtype=np.float64), 0

    Qb = coarse.apply(b)
    if x0 is None:
        x = Qb.copy()
    else:
        x0 = np.asarray(x0, dtype=np.float64)
        QSx0 = coarse.apply(matvec(x0))  # Q (S x0) -- P^T = I - QS
        x = Qb + (x0 - QSx0)

    r = b - matvec(x)
    rho_prev = None
    p = None
    for _ in range(maxiter):
        rnorm = float(np.linalg.norm(r))
        if rnorm <= atol_eff:
            return x, 0
        Qr = coarse.apply(r)
        QSr = coarse.apply(matvec(r))  # Q (S r) -- P^T applied to r
        z = base_apply(r - QSr) + Qr   # M_base^-1 (r - Q S r) + Q r
        rho_cur = float(np.dot(r, z))
        if rho_prev is not None and rho_prev != 0.0:
            p = z + (rho_cur / rho_prev) * p
        else:
            p = z.copy()
        Sp = matvec(p)  # PLAIN, unprojected matvec
        pSp = float(np.dot(p, Sp))
        if pSp == 0.0:
            return x, (0 if rnorm <= atol_eff else maxiter)
        alpha = rho_cur / pSp
        x = x + alpha * p
        r = r - alpha * Sp
        rho_prev = rho_cur
    return x, maxiter


class TestTransposeCorrectedADef2AlsoFails:
    """Spec-compliance review round 3 (major finding follow-up): the
    reviewer's finding was that the shipped ``apply_mode='deflated'`` ships
    DEF instead of the spec's literal formula, and asked the coordinator to
    ratify that substitution. Before accepting round-1/round-2's
    "literal formula stalls" conclusion at face value a third time, this
    fix-agent pass tried one more independently-motivated hypothesis: that
    the spec's formula got the deflation-operator TRANSPOSE backwards
    (``P = I - SQ`` where the published taxonomy's A-DEF2 preconditioner
    uses ``P^T = I - QS``   -- see ``_transpose_corrected_adef2_pcg``'s
    docstring), and that fixing the transpose might recover a genuinely
    convergent literal-A-DEF2-family algorithm.

    It does not. On the realistic-ratio ill-conditioned fixture the
    transpose-corrected formula does not merely fail to converge within
    budget -- it DIVERGES (true relative residual grows past 1, i.e. worse
    than the ``x=0`` starting guess would give for a normalized RHS), which
    is a strictly worse failure mode than either of round-1/round-2's two
    rejected candidates. This is a THIRD independent line of evidence (on
    top of the two already in ``TestLiteralADef2FormulaIndependentlyReverified``)
    that folding ``Q`` into the preconditioner while keeping a plain
    (unprojected) matvec is not a viable "swap the formula into a vanilla
    PCG loop" substitution for this problem class, regardless of which
    operator-transpose convention is used -- consistent with this module's
    "A-DEF2" docstring's root-cause argument (the resulting preconditioner
    is not self-adjoint in a way that preserves PCG's conjugacy
    guarantees). Strengthens, rather than reopens, the round-2 decision to
    keep DEF as ``apply_mode='deflated'``'s shipped algorithm -- ratification
    of that decision is still the coordinator's call (task #29), not this
    fix pass's, since it is a disclosed design decision rather than a code
    defect.
    """

    def test_transpose_corrected_formula_diverges_on_ill_conditioned_fixture(self):
        tile_schur, tile_idx, S_extra, S, n = _ill_conditioned_jacobi_fixture()
        rng = np.random.default_rng(11)
        b = rng.standard_normal(n)
        rtol, atol = 1e-9, 1e-14

        solver = InterfaceCGSolver(
            n_interface=n, matvec_mode='tilewise',
            tile_schur_complements=tile_schur, tile_index_maps=tile_idx,
            S_extra=S_extra,
            preconditioner='two_level', rtol=rtol, atol=atol,
            matvec_threads=1, strict=False, maxiter=3000,
            block_jacobi_max_bytes=1,  # force jacobi base
            interface_coarse_apply_mode='deflated',
        )
        matvec = solver._linear_op.matvec
        base_apply = solver._M_base_apply
        coarse = solver._coarse
        assert coarse.SZ is not None
        try:
            x_shipped, info_shipped = _deflated_pcg(
                matvec, base_apply, coarse, b, None, rtol=rtol, atol=atol,
                maxiter=3000, reproject_every=50,
            )
            assert info_shipped == 0, (
                f"sanity check failed: shipped DEF algorithm itself did "
                f"not converge (info={info_shipped}) on this fixture"
            )

            x_t, info_t = _transpose_corrected_adef2_pcg(
                matvec, base_apply, coarse, b, None, rtol=rtol, atol=atol,
                maxiter=3000,
            )
            rel_r_t = (
                float(np.linalg.norm(b - matvec(x_t)))
                / float(np.linalg.norm(b))
            )
            assert info_t != 0, (
                "transpose-corrected A-DEF2 formula unexpectedly converged "
                "-- re-examine whether the transpose hypothesis changes "
                "the round-1/round-2 conclusion"
            )
            # Strictly worse than "merely didn't converge": true relative
            # residual exceeds 1 (worse than the zero vector for this
            # normalized-b fixture) -- genuine divergence, not a slow
            # crawl toward the answer.
            assert rel_r_t > 1.0, (
                f"transpose-corrected formula's relative residual "
                f"({rel_r_t:.3e}) did not diverge as measured during "
                f"round-3 verification -- re-check this test's evidence"
            )
        finally:
            solver.close()


# ---------------------------------------------------------------------------
# TRUE A-DEF2 (coordinator ruling, A-DEF2 work package ratification pass):
# a fourth independently-implemented candidate, distinct from all three
# above. The coordinator's ruling on the round-2/3/4 history: the spec's
# original in-line formula (``_literal_spec_adef2_pcg``) was a
# TRANSCRIPTION ERROR -- "apply P to r before M_base^-1 with an un-projected
# matvec" is actually **A-DEF1** in the Tang/Nabben/Vuik/Erlangga taxonomy,
# known to be non-robust with inexact/pseudo-inverse coarse solves (matches
# the observed stall). TRUE A-DEF2 is a genuinely different combination
# from EITHER rejected candidate above (``_literal_spec_adef2_pcg`` applies
# ``P``/``(I - SQ)`` to ``r`` BEFORE ``M_base^-1``; ``_transpose_corrected_
# adef2_pcg`` applies ``P^T``/``(I - QS)`` to ``r`` BEFORE ``M_base^-1`` --
# BOTH pre-multiply the INPUT to ``M_base^-1``). True A-DEF2 instead applies
# ``P^T`` to the OUTPUT of ``M_base^-1``:
#
#     z = (I - QS) M_base^-1 r + Q r  =  P^T M_base^-1 r + Q r,  P = I - S Q
#
# inside a STANDARD (unprojected-matvec) PCG recurrence, with the MANDATORY
# starting vector ``x0' = Q b + (I - QS) x0`` (``x0=0`` -> ``x0' = Q b``).
# Implemented and measured by the ratification-resolution agent (this pass)
# per the coordinator's explicit instruction; selected BY DATA against the
# shipped DEF algorithm (``apply_mode='deflated'``) on:
#   - the ``netlist_multi_tile`` gate script (both scenarios, rtol=1e-8):
#     true A-DEF2 TIES DEF exactly in the production-representative
#     'jacobi-forced' scenario (16.75 == 16.75 warm iters/step -- consistent
#     with the DEF1/A-DEF2 mathematical-equivalence theorem for a projected
#     x0 and a diagonal base), but REGRESSES in 'natural' (98.10 vs DEF's
#     83.00 and additive's 74.65 warm iters/step -- fails the required
#     ``<= additive * 1.05`` bar);
#   - this file's ``_ill_conditioned_jacobi_fixture`` (realistic T'/n ratio,
#     PoU-only): true A-DEF2 FAILS TO CONVERGE outright (hits maxiter) at
#     cond_log >= 9, while DEF and additive both converge (592 / 577 iters
#     at cond_log=9; 2029 / 1848 iters at cond_log=12).
# DEF wins on this data (see ``interface_iterative.py``'s "A-DEF2 work
# package" docstring section for the full record) -- true A-DEF2 is NOT
# shipped as a selectable ``interface_coarse_apply_mode``. Kept here (not in
# ``interface_iterative.py``) for the regression coverage its mandatory
# ``x0`` projection deserves (see ``TestTrueADef2X0ProjectionRegression``
# below) -- exactly the same "measured, rejected, kept in the test file"
# treatment as ``_literal_spec_adef2_pcg``/``_transpose_corrected_adef2_pcg``
# above.  ``Q S v`` is computed via the ``_apply_QS`` helper below (the
# retained ``SZ`` -- ``S`` symmetric => ``Z^T S = (S Z)^T``), never a fresh
# full ``S`` matvec, exactly like DEF's own ``apply_with_SQ``.
# ---------------------------------------------------------------------------


def _apply_QS(coarse, v: np.ndarray) -> np.ndarray:
    r"""True A-DEF2's ``Q S v`` product: ``Q S v = Z S_c^+ (S Z)^T v = Z
    S_c^+ Z^T S v`` -- computed via the RETAINED ``SZ`` (``S`` is
    symmetric, so ``Z^T S = (S Z)^T``), never a fresh full ``S`` matvec.

    Round-2 code review finding 10: moved here from
    ``interface_coarse.CoarseSpace.apply_QS`` -- this function (via
    :func:`_true_adef2_pcg`, below) was always its only caller, so
    production carried ~45 lines of otherwise-dead API surface for a
    single rejected-algorithm test helper. Shares the same ``(T', k)``-dim
    pseudo-inverse solve helper (``coarse._solve_Sc_pinv``) as
    ``CoarseSpace.apply``/``apply_with_SQ``; the only difference is the
    vector fed into it is ``(S Z)^T v`` (a GEMV against the retained
    ``SZ``) instead of ``Z^T r``.

    Accepts either a 1-D vector or a ``(n, k)`` batch; returns the same
    shape.

    Raises:
        ValueError: If ``coarse.SZ`` was not retained (build with
            ``retain_sz=True``) -- same precondition as
            ``CoarseSpace.apply_with_SQ``.
    """
    if coarse.SZ is None:
        raise ValueError(
            "_apply_QS requires SZ retained (build with retain_sz=True); "
            "this CoarseSpace was built with SZ dropped (additive apply "
            "mode, or a byte-budget SZ-drop degrade)."
        )
    v = np.asarray(v, dtype=np.float64)
    squeeze = v.ndim == 1
    Vv = v.reshape(-1, 1) if squeeze else v
    w = np.asarray(coarse.SZ.T @ Vv, dtype=np.float64)        # (T', k)
    y = coarse._solve_Sc_pinv(w)
    out = np.asarray(coarse.Z @ y, dtype=np.float64)          # (n, k)
    return out[:, 0] if squeeze else out


def _true_adef2_pcg(matvec, base_apply, coarse, b, x0, rtol, atol, maxiter,
                     reproject_every=None, callback=None,
                     _debug_skip_x0_projection=False):
    r"""True A-DEF2 preconditioned CG on ``S x = b`` (see the module-level
    comment immediately above for the formula, why it differs from the
    other three candidates in this file, and the measured selection
    record).

    Standard (unprojected-matvec) PCG: the preconditioner apply is
    ``z = P^T M_base^-1 r + Q r`` computed as ``base_apply(r) -
    _apply_QS(coarse, base_apply(r)) + coarse.apply(r)``. Because the matvec
    is unprojected, the CG iterate ``y`` IS the returned solution ``x``
    directly -- no separate recovery step (unlike DEF's ``_recover_x``,
    which exists only because DEF's matvec is projected) and no ``Z^T r_k =
    0`` invariant maintained by construction. Re-projection here is
    therefore ordinary CG hygiene (recompute ``r = b - S @ y`` fresh every
    ``reproject_every`` iterations, ``<= 0`` disables), not a deflation-
    specific correctness requirement.

    **Mandatory starting vector** -- this is "the crux" the coordinator's
    ruling calls out: ``y0 = Q b + (I - QS) x0 = coarse.apply(b) + x0 -
    _apply_QS(coarse, x0)`` for a warm ``x0``; ``x0=None`` reduces to
    ``coarse.apply(b)`` (``Q b``). See ``_debug_skip_x0_projection`` below.

    Args:
        matvec: The ORIGINAL system's matvec (``S @ x``) -- PLAIN/unprojected.
        base_apply: ``M_base^-1`` alone (``InterfaceCGSolver._M_base_apply``).
        coarse: A ``CoarseSpace`` with ``SZ`` retained (required for
            :func:`_apply_QS`).
        b: Right-hand side, shape ``(n,)``.
        x0: Initial guess (projected per the mandatory formula above), or
            ``None`` for a cold start.
        rtol, atol: ``||r|| <= max(rtol * ||b||, atol)``.
        maxiter: Maximum CG iterations.
        reproject_every: Fresh-residual-recompute interval; ``None``
            resolves to ``ic.DEFAULT_DEFLATED_REPROJECT_EVERY``.
        callback: Called as ``callback(xk)`` once per completed iteration
            with the real, up-to-date iterate (no recovery to gate, unlike
            DEF's loop).
        _debug_skip_x0_projection: TEST/DEMONSTRATION-ONLY. When True,
            bypasses the mandatory projection and seeds ``y0 = x0`` raw --
            exists solely so ``TestTrueADef2X0ProjectionRegression`` can
            reproduce, as a permanent regression test, the exact blowup the
            coordinator's ruling names as the crux of the x0-projection
            requirement. Never set True outside a test.

    Returns:
        ``(x, info)`` -- ``info=0`` on convergence, else ``maxiter``.
    """
    if reproject_every is None:
        reproject_every = ic.DEFAULT_DEFLATED_REPROJECT_EVERY
    b = np.asarray(b, dtype=np.float64)
    n = b.shape[0]
    bnrm2 = float(np.linalg.norm(b))
    atol_eff = max(float(atol), float(rtol) * bnrm2)
    if bnrm2 == 0.0:
        return np.zeros(n, dtype=np.float64), 0

    def _precond_apply(r):
        Mr = base_apply(r)
        return Mr - _apply_QS(coarse, Mr) + coarse.apply(r)

    if x0 is None:
        y = coarse.apply(b)
    else:
        x0_arr = np.asarray(x0, dtype=np.float64)
        if _debug_skip_x0_projection:
            y = x0_arr.copy()
        else:
            y = coarse.apply(b) + x0_arr - _apply_QS(coarse, x0_arr)

    r = b - matvec(y)

    def _try_accept(rnorm_tracked):
        true_rnorm = float(np.linalg.norm(b - matvec(y)))
        if true_rnorm <= atol_eff:
            return y.copy(), 0
        return None

    rho_prev = None
    p = None
    for iteration in range(maxiter):
        rnorm = float(np.linalg.norm(r))
        if rnorm <= atol_eff:
            accepted = _try_accept(rnorm)
            if accepted is not None:
                return accepted

        if reproject_every > 0 and iteration > 0 and iteration % reproject_every == 0:
            r = b - matvec(y)

        z = _precond_apply(r)
        rho_cur = float(np.dot(r, z))
        if iteration > 0 and rho_prev != 0.0:
            beta = rho_cur / rho_prev
            p = z + beta * p
        else:
            p = z.copy()

        Sp = matvec(p)
        pw = float(np.dot(p, Sp))
        if pw == 0.0:
            p = z.copy()
            Sp = matvec(p)
            pw = float(np.dot(p, Sp))
            if pw == 0.0:
                if rnorm <= atol_eff:
                    accepted = _try_accept(rnorm)
                    if accepted is not None:
                        return accepted
                return y.copy(), maxiter
        alpha = rho_cur / pw
        y = y + alpha * p
        r = r - alpha * Sp
        rho_prev = rho_cur

        if callback is not None:
            callback(y)

    final_rnorm = float(np.linalg.norm(r))
    if final_rnorm <= atol_eff:
        accepted = _try_accept(final_rnorm)
        if accepted is not None:
            return accepted
    return y.copy(), maxiter


class TestTrueADef2SelectionRecord:
    """Ruling item 2 (accuracy verification): true A-DEF2 reproduces the
    direct solution at pinned rtol=1e-12 on the two-tile and chain fixtures
    -- the implementation itself is CORRECT, it was rejected on iteration/
    robustness grounds (see the module-level comment above and
    ``interface_iterative.py``'s "A-DEF2 work package" docstring section),
    not because it computes a wrong answer."""

    @pytest.mark.parametrize('n_tiles', [2, 30])
    def test_true_adef2_matches_direct(self, n_tiles):
        tile_schur, tile_idx, n = _chain_tiles(n_tiles)
        rng = np.random.default_rng(555)
        b = rng.standard_normal(n)
        S = np.zeros((n, n))
        for tid, idx in tile_idx.items():
            S[np.ix_(idx, idx)] += tile_schur[tid]
        x_direct = np.linalg.solve(S, b)

        solver = InterfaceCGSolver(
            n_interface=n, matvec_mode='tilewise',
            tile_schur_complements={k: v.copy() for k, v in tile_schur.items()},
            tile_index_maps=tile_idx,
            preconditioner='two_level', rtol=1e-12, atol=1e-16,
            matvec_threads=1, strict=True, maxiter=5000,
            interface_coarse_apply_mode='deflated',  # only to retain SZ
        )
        try:
            coarse = solver._coarse
            assert coarse.SZ is not None
            x, info = _true_adef2_pcg(
                solver._linear_op.matvec, solver._M_base_apply, coarse,
                b, None, rtol=1e-12, atol=1e-16, maxiter=5000,
            )
            assert info == 0
            err = np.max(np.abs(x - x_direct))
            assert err <= 1e-10, f"[n_tiles={n_tiles}] err={err:.3e}"
        finally:
            solver.close()

    def test_true_adef2_fails_to_converge_on_realistic_ratio_fixture(self):
        """Negative-style check reinforcing the selection: DEF and additive
        both converge on this fixture (see
        ``TestADef2IterationRegimeDependence``); true A-DEF2 does not --
        pinned as a regression check on the selection record itself."""
        tile_schur, tile_idx, S_extra, S, n = _ill_conditioned_jacobi_fixture(
            cond_log=12.0,
        )
        rng = np.random.default_rng(11)
        b = rng.standard_normal(n)
        solver = InterfaceCGSolver(
            n_interface=n, matvec_mode='tilewise',
            tile_schur_complements=tile_schur, tile_index_maps=tile_idx,
            S_extra=S_extra,
            preconditioner='two_level', rtol=1e-9, atol=1e-14,
            matvec_threads=1, strict=False, maxiter=2100,
            block_jacobi_max_bytes=1,
            interface_coarse_apply_mode='deflated',
        )
        try:
            coarse = solver._coarse
            assert coarse.SZ is not None
            _, info_def = _deflated_pcg(
                solver._linear_op.matvec, solver._M_base_apply, coarse,
                b, None, rtol=1e-9, atol=1e-14, maxiter=2100, reproject_every=50,
            )
            assert info_def == 0, "sanity: DEF must converge on this fixture"
            _, info_true = _true_adef2_pcg(
                solver._linear_op.matvec, solver._M_base_apply, coarse,
                b, None, rtol=1e-9, atol=1e-14, maxiter=2100,
            )
            assert info_true != 0, (
                "true A-DEF2 unexpectedly converged where DEF's advantage "
                "was measured -- re-open the selection decision"
            )
        finally:
            solver.close()


class TestTrueADef2X0ProjectionRegression:
    """Ruling item 5 (the crux): with true A-DEF2, the warm ``x0`` MUST be
    projected (``x0' = Q b + (I - QS) x0``) on every solve. Measured
    evidence (``_ill_conditioned_jacobi_fixture(cond_log=7.0)``, seed 11,
    a 5%-perturbed second RHS mimicking a transient step):

        cold solve (system 1):            276 iters, converged
        warm solve, PROJECTED x0:          236 iters, converged  (<= cold)
        warm solve, x0 projection SKIPPED: 5000 iters, DID NOT CONVERGE

    Skipping the projection does not merely cost a few extra iterations --
    it breaks convergence outright within the same iteration budget that
    the properly-projected warm start clears in fewer iterations than the
    cold solve. This is the true-A-DEF2-specific manifestation of "the
    crux"; contrast DEF (the SHIPPED algorithm, ``apply_mode='deflated'``),
    whose own warm-start pathology (measured 74.65 -> 83.00 warm iters/step
    on the real ``netlist_multi_tile`` PDN, additive vs. DEF, 'natural'
    scenario -- see ``scripts/benchmark/microbench/run_adef2_multi_tile_
    gate.py``) is real but does NOT reproduce on any synthetic fixture
    tried here (chain: DEF converges in ~1 iteration regardless of warm/
    cold, near-degenerate T'~n; this file's ill-conditioned fixture: DEF's
    warm solve was measured FASTER than cold, 471 vs 589 iters at
    cond_log=9.0) -- DEF's own warm-start defect needs the real PDN's
    coarse-space/RHS-delta geometry to manifest (see the gate script's
    extensively-documented round-2 root-cause section), unlike true
    A-DEF2's x0-projection requirement, which is a direct algebraic
    necessity reproducible on a plain synthetic fixture.
    """

    def test_projected_warm_start_converges_at_or_below_cold_iters(self):
        tile_schur, tile_idx, S_extra, S, n = _ill_conditioned_jacobi_fixture(
            cond_log=7.0,
        )
        rng = np.random.default_rng(11)
        b1 = rng.standard_normal(n)
        b2 = b1 + 0.05 * rng.standard_normal(n)

        solver = InterfaceCGSolver(
            n_interface=n, matvec_mode='tilewise',
            tile_schur_complements=tile_schur, tile_index_maps=tile_idx,
            S_extra=S_extra,
            preconditioner='two_level', rtol=1e-9, atol=1e-14,
            matvec_threads=1, strict=False, maxiter=5000,
            block_jacobi_max_bytes=1,
            interface_coarse_apply_mode='deflated',
        )
        try:
            coarse = solver._coarse
            matvec = solver._linear_op.matvec
            base_apply = solver._M_base_apply
            assert coarse.SZ is not None

            def _count(**kw):
                cnt = [0]
                x, info = _true_adef2_pcg(
                    matvec, base_apply, coarse, callback=lambda _x: cnt.__setitem__(0, cnt[0] + 1),
                    **kw,
                )
                return x, info, cnt[0]

            x1, info1, n1 = _count(
                b=b1, x0=None, rtol=1e-9, atol=1e-14, maxiter=5000,
            )
            assert info1 == 0, "sanity: cold solve1 must converge"

            # Negative-test evidence (Rules section): skip the mandatory
            # projection first, on the SAME warm x0/RHS, and show the
            # blowup, before demonstrating the fix.
            x2_naive, info2_naive, n2_naive = _count(
                b=b2, x0=x1, rtol=1e-9, atol=1e-14, maxiter=5000,
                _debug_skip_x0_projection=True,
            )
            assert info2_naive != 0, (
                f"expected the UNPROJECTED warm start to fail to converge "
                f"within maxiter (measured: hits maxiter=5000) -- got "
                f"info={info2_naive} after {n2_naive} iters; the x0-"
                f"projection regression this test exists to pin no longer "
                f"reproduces, re-examine _true_adef2_pcg"
            )

            # Restore: the properly-projected warm start converges in FEWER
            # iterations than the cold solve (measured 236 <= 276).
            x2_proj, info2_proj, n2_proj = _count(
                b=b2, x0=x1, rtol=1e-9, atol=1e-14, maxiter=5000,
            )
            assert info2_proj == 0, (
                f"projected warm start must converge; info={info2_proj}"
            )
            assert n2_proj <= n1, (
                f"projected warm solve ({n2_proj} iters) must take <= the "
                f"cold first solve ({n1} iters) -- measured 236 <= 276"
            )
        finally:
            solver.close()


class TestReprojectionDrift:
    """Re-projection drift: a long solve (many CG iterations) asserts
    ||Z^T r||/||r|| stays bounded (small) with reprojection on, and the
    final true residual meets rtol.

    Under the corrected A-DEF2 algorithm (matvec itself projected -- see
    the module docstring's "A-DEF2" section), ``Z^T r_k = 0`` is an
    ALGEBRAIC IDENTITY at every iterate (``Z^T (P v) = 0`` for any ``v``),
    not an empirical hope -- so ||Z^T r||/||r|| is expected to be tiny
    essentially throughout, bounded only by floating-point roundoff in the
    ``Z^T(...)`` computation itself, and re-projection is genuine hygiene
    (verified to make no difference to whether this fixture converges --
    reproject_every=0 converges too, just tests that specifically).

    Fixture: the ill-conditioned dense-SPD fixture (chain fixtures converge
    in ~1 iteration under this algorithm -- too short to exercise
    reprojection at all).
    """

    def test_drift_bounded_and_true_residual_meets_rtol(self):
        tile_schur, tile_idx, S_extra, S, n = _ill_conditioned_jacobi_fixture()
        rng = np.random.default_rng(11)
        b = rng.standard_normal(n)
        rtol, atol = 1e-9, 1e-14

        solver = InterfaceCGSolver(
            n_interface=n, matvec_mode='tilewise',
            tile_schur_complements=tile_schur, tile_index_maps=tile_idx,
            S_extra=S_extra,
            preconditioner='two_level', rtol=rtol, atol=atol,
            matvec_threads=1, strict=True, maxiter=3000,
            block_jacobi_max_bytes=1,  # force jacobi base -- more iterations
            interface_coarse_apply_mode='deflated',
            interface_deflated_reproject_every=5,
        )
        matvec = solver._linear_op.matvec
        base_apply = solver._M_base_apply
        coarse = solver._coarse
        assert coarse.SZ is not None

        abs_ztr = []
        rns = []

        def track(xk):
            r = b - matvec(xk)
            rns.append(float(np.linalg.norm(r)))
            abs_ztr.append(float(np.linalg.norm(coarse.Z.T @ r)))

        x, info = _deflated_pcg(
            matvec, base_apply, coarse, b, None, rtol=rtol, atol=atol,
            maxiter=3000, reproject_every=5, callback=track,
        )
        solver.close()

        assert info == 0, f"did not converge (info={info})"
        assert len(abs_ztr) >= 100, (
            "fixture must run enough iterations to exercise reprojection "
            f"many times; only got {len(abs_ztr)}"
        )
        # Z^T(b - S xk) = 0 is an algebraic identity for the RECOVERED xk
        # too (b - S x_k = P(b - S y_k), see the module docstring) --
        # checked in ABSOLUTE terms (not the ratio to ||r||, which becomes
        # a noisy floating-point artifact once ||r|| itself shrinks toward
        # the convergence floor near the end of the run -- the ABSOLUTE
        # ||Z^T r|| is what "no growing drift" actually means, and stays
        # near machine-precision-scale throughout, well below r's own
        # starting scale (||r_0|| ~ O(1) for this fixture's normalized b).
        assert max(abs_ztr) < 1e-8, (
            f"||Z^T r|| (absolute) grew far beyond floating-point roundoff "
            f"despite reprojection: max={max(abs_ztr):.3e}"
        )

        true_r = float(np.linalg.norm(b - matvec(x)))
        target = rtol * float(np.linalg.norm(b))
        assert true_r <= target, (
            f"final true residual {true_r:.3e} exceeds the rtol target "
            f"{target:.3e}"
        )

    def test_reprojection_disabled_still_converges(self):
        """reprojection is hygiene, not correctness, under the corrected
        algorithm -- reproject_every=0 (disabled) must still converge
        (possibly needing a different iteration count) on the same
        fixture."""
        tile_schur, tile_idx, S_extra, S, n = _ill_conditioned_jacobi_fixture()
        rng = np.random.default_rng(11)
        b = rng.standard_normal(n)

        solver = InterfaceCGSolver(
            n_interface=n, matvec_mode='tilewise',
            tile_schur_complements=tile_schur, tile_index_maps=tile_idx,
            S_extra=S_extra,
            preconditioner='two_level', rtol=1e-9, atol=1e-14,
            matvec_threads=1, strict=True, maxiter=3000,
            block_jacobi_max_bytes=1,
            interface_coarse_apply_mode='deflated',
            interface_deflated_reproject_every=0,
        )
        try:
            x = solver(b)
            true_r = np.linalg.norm(b - solver._linear_op.matvec(x))
            assert true_r <= 1e-9 * np.linalg.norm(b)
        finally:
            solver.close()


class _DriftedSyCoarseStub:
    """Duck-typed ``CoarseSpace`` for :class:`TestSyDriftRepairedAtAccept`:
    ``Q = 0`` for the PROJECTED matvec (``apply_with_SQ`` -> ``(0, 0)``, so
    ``w = Sp`` and the CG recurrence is plain, self-consistent
    unpreconditioned-by-Q CG driven entirely by whatever ``matvec``
    implements), but ``apply(v) = v`` exactly (used ONLY by
    ``_recover_x``/``_try_accept``, giving ``x_candidate = y + (b - Sy)``
    -- i.e. ``Q = I`` for RECOVERY purposes specifically). Combined with a
    TRUE operator of the identity (see ``_PoisonedEarlyMatvecStub``),
    ``x_candidate`` recovers to EXACTLY ``b`` whenever the ``Sy`` fed into
    it correctly reflects ``S @ y`` (``x = y + b - S y = b`` when ``S =
    I``) -- and to something ELSE, provably wrong by exactly the drift
    amount, whenever a STALE/drifted ``Sy`` is used instead. This directly
    exposes ``_try_accept``'s choice of ``Sy`` source to the test."""

    SZ = np.zeros((1, 1))

    def apply_with_SQ(self, v):
        v = np.asarray(v, dtype=np.float64)
        return np.zeros_like(v), np.zeros_like(v)

    def apply(self, v):
        return np.asarray(v, dtype=np.float64).copy()


class _PoisonedEarlyMatvecStub:
    """``matvec`` = identity, EXCEPT the first ``n_poison`` calls silently
    ADD a fixed ``poison`` vector to the output. Deterministic, CALL-
    COUNT-based (not argument-based): the per-iteration ``Sp = matvec(p)``
    calls that happen while ``self.calls <= n_poison`` are corrupted, so
    the incrementally-tracked ``Sy = Sy + alpha*Sp`` (built entirely from
    those early, poisoned ``Sp`` values) permanently drifts away from the
    TRUE ``S @ y`` (``S`` = identity) by a fixed, nonzero amount -- this
    is a controlled STAND-IN for the real mechanism (floating-point
    accumulation error over many iterations of a real matvec), engineered
    to be large and immediate rather than requiring a very long/ill-
    conditioned real solve to become numerically visible (see
    ``TestReprojectionDrift``/``TestTrueADef2SelectionRecord`` for the
    real-fixture-scale version, where drift is genuine but tiny). Calls
    made AFTER the poison window (e.g. ``_try_accept``'s own fresh
    ``matvec(y)`` call, made once enough iterations have passed) see the
    TRUE, un-poisoned identity -- resolving the drift the STALE
    incrementally-tracked ``Sy`` still carries."""

    def __init__(self, n_poison, poison):
        self.calls = 0
        self.n_poison = n_poison
        self.poison = poison

    def __call__(self, v):
        v = np.asarray(v, dtype=np.float64)
        self.calls += 1
        if self.calls <= self.n_poison:
            return v.copy() + self.poison
        return v.copy()


class TestSyDriftRepairedAtAccept:
    """Round-3 code review finding 4 (PLAUSIBLE, CONFIRMED): without a
    fresh ``Sy`` refresh at acceptance, ``_try_accept`` recovers
    ``x_candidate`` from the SAME stale, incrementally-tracked ``Sy`` its
    own fresh true-residual check is supposed to validate against -- if
    ``Sy`` has drifted from the true ``S @ y`` by more than the tolerance
    margin, the fresh check disagrees with the tracked residual FOREVER
    (both derive from the same drifted quantity), burning the entire
    ``maxiter`` budget on a solve that should have accepted almost
    immediately. The fix (scoped to ``_try_accept`` itself -- see its
    docstring for why NOT the periodic re-projection block, which was
    tried first and found to slow convergence on a real ill-conditioned
    fixture) recomputes ``Sy_fresh = matvec(y)`` before recovering
    ``x_candidate``, repairing the drift at exactly the point it matters.

    This test forces drift via :class:`_PoisonedEarlyMatvecStub`/
    :class:`_DriftedSyCoarseStub` (see their docstrings) rather than
    relying on real floating-point accumulation over a very long solve.

    Negative-test evidence: temporarily reverting ``_try_accept`` to
    recover ``x_candidate`` from the stale incrementally-tracked ``Sy``
    (``x_candidate = y + coarse.apply(b - Sy)``, dropping the
    ``Sy_fresh = matvec(y)`` line) makes
    :meth:`test_acceptance_recovers_via_fresh_sy_instead_of_burning_maxiter`
    FAIL (``info`` becomes ``maxiter`` instead of ``0``, and the returned
    ``x`` is off from ``b`` by exactly the engineered drift amount) --
    verified directly while implementing this fix (see this session's
    final report for the revert/restore transcript).
    """

    def test_acceptance_recovers_via_fresh_sy_instead_of_burning_maxiter(self):
        n = 5
        b = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        # Large enough that the resulting x_candidate error (~alpha *
        # ||poison||, alpha close to 1) comfortably exceeds atol_eff below
        # -- forcing a REAL accept/reject decision, not a marginal one.
        poison = np.array([0.3, -0.2, 0.1, 0.05, -0.15])
        matvec = _PoisonedEarlyMatvecStub(n_poison=1, poison=poison)
        coarse = _DriftedSyCoarseStub()
        base_apply = lambda v: np.asarray(v, dtype=np.float64).copy()

        x, info = _deflated_pcg(
            matvec, base_apply, coarse, b, None, rtol=1e-10, atol=1e-12,
            maxiter=50, reproject_every=0,
        )
        assert info == 0, (
            f"expected the fresh-Sy repair to accept well within maxiter=50 "
            f"-- got info={info} (maxiter reached means the stale-Sy "
            f"acceptance failure was NOT repaired)"
        )
        np.testing.assert_allclose(x, b, atol=1e-8), (
            f"expected the recovered x to be the mathematically exact "
            f"identity-system solution (b) once Sy drift is repaired; "
            f"got x={x!r}"
        )


class _AlwaysDegenerateProjectedMatvecStub:
    """Duck-typed ``CoarseSpace`` stand-in for directly testing
    ``_deflated_pcg``'s double-breakdown control flow (Finding 7,
    round-1 code review) -- ``apply_with_SQ``'s second return value
    (``S Q v``) is defined to equal its own input ``v`` for ANY ``v``,
    making the projected matvec ``w = Sp - SQp`` EXACTLY zero on every
    call regardless of the search direction. Combined with an identity
    ``matvec``/``base_apply``, this deterministically forces BOTH the
    first ``pw`` breakdown attempt AND the ``p=z.copy()`` retry to also
    land on ``pw == 0`` -- the "genuinely stuck" double-breakdown exit --
    with no fragile dependence on floating-point rounding landing near
    zero (unlike a real ill-conditioned SPD fixture)."""

    SZ = np.zeros((1, 1))  # truthy "SZ retained" sentinel; never read directly

    def apply_with_SQ(self, v):
        v = np.asarray(v, dtype=np.float64)
        return np.zeros_like(v), v.copy()

    def apply(self, v):
        return np.zeros_like(np.asarray(v, dtype=np.float64))


class TestDoubleBreakdownInfoContract:
    """Finding 7 (round-1 code review): the double-breakdown (``pw == 0``
    twice) early exit in ``_deflated_pcg`` must report the ACTUAL number
    of iterations performed (scipy contract: ``info`` = iterations
    performed on failure) when that count is >= 1, not a hardcoded
    ``maxiter`` -- the round-1-pre-fix code always claimed the iteration
    BUDGET was exhausted, even when the true cause (a degenerate search
    direction) struck on literally the first iteration, contradicting the
    solver's own ``last_cg_iters``/strict-mode error message.

    Round-2 code review finding 1 (CRITICAL regression in the round-1
    fix, corrected here): reporting the LITERAL ``iteration`` value
    collides with scipy's ``info == 0`` SUCCESS code whenever the
    breakdown strikes on the very first iteration (``iteration == 0``,
    exactly what :class:`_AlwaysDegenerateProjectedMatvecStub` forces
    deterministically) -- an UNCONVERGED solve was reported as converged,
    silently corrupting downstream DC/QS/transient results (see
    :class:`TestDoubleBreakdownRegressionInfoZero` below for the
    end-to-end ``InterfaceCGSolver`` regression coverage: strict-mode
    RuntimeError, ``stats['cg_failed']``, and warm-start history hygiene).
    The contract is now: report ``max(iteration, 1)`` -- the actual
    iteration count when it is already >= 1 (preserving the round-1 fix's
    intent), floored at 1 (scipy's smallest FAILURE code) when it would
    otherwise be 0, since "0 iterations performed but NOT converged" has
    no representation in scipy's info-is-iteration-count-on-failure
    contract that isn't also its success code."""

    def test_double_breakdown_on_first_iteration_reports_info_one_not_zero(self):
        b = np.array([1.0, 2.0, 3.0, 4.0])
        matvec = lambda v: np.asarray(v, dtype=np.float64).copy()
        base_apply = lambda v: np.asarray(v, dtype=np.float64).copy()
        coarse = _AlwaysDegenerateProjectedMatvecStub()

        maxiter = 7
        x, info = _deflated_pcg(
            matvec, base_apply, coarse, b, None, rtol=1e-30, atol=1e-30,
            maxiter=maxiter, reproject_every=0,
        )
        # The stub forces the double-breakdown exit on the very FIRST
        # iteration (0 complete iterations performed before it). The
        # literal iteration count (0) collides with scipy's success code,
        # so info must be floored at 1 -- NOT 0 (would be silently
        # reported as converged) and NOT maxiter=7 (would falsely claim
        # the iteration budget was exhausted).
        assert info == 1, (
            f"expected info == 1 (the smallest value that both reflects "
            f"'0 iterations actually performed' and still signals FAILURE "
            f"per scipy's info>0-means-unconverged contract), got "
            f"info={info} (maxiter was {maxiter})"
        )
        assert np.all(np.isfinite(x))

    def test_double_breakdown_reports_actual_iteration_count_when_nonzero(self):
        """When the double-breakdown strikes AFTER at least one completed
        iteration, info must still report that ACTUAL count (the
        ``max(iteration, 1)`` floor is a no-op once ``iteration >= 1``) --
        the round-1 regression this class also guards against (a
        hardcoded ``maxiter`` return) is independent of the round-2
        iteration==0 collision fixed above. Uses a stub identical to
        :class:`_AlwaysDegenerateProjectedMatvecStub` except the
        projected matvec only degenerates from a chosen CALL onward, so
        the first few iterations proceed as ordinary (non-preconditioned,
        diagonal-``S``) CG before the breakdown strikes.

        Round-3 code review finding 7 (CONFIRMED off-by-one, fixed here):
        ``_deflated_pcg`` makes ONE ``apply_with_SQ`` call in its SETUP
        (``_, SQr0 = coarse.apply_with_SQ(b - Sy)``, projecting the
        initial residual) BEFORE the loop's first iteration -- so with
        ``degenerate_from=2`` the call sequence was: #1 setup (normal,
        ``self._calls=1 <= 2``), #2 iteration 0's ``Sp`` (normal,
        ``self._calls=2 <= 2``), #3 iteration 1's ``Sp`` (DEGENERATE,
        ``self._calls=3 > 2``) -- the breakdown struck at ``iteration ==
        1``, not ``2`` as the docstring claimed, and (with this fixture's
        ``atol=0.0``, so the exit's ``rnorm <= atol_eff`` last-chance
        accept never fires) ``info`` came out ``max(1, 1) == 1`` --
        exactly the SAME value the ``max(iteration, 1)`` floor produces
        for a GENUINE ``iteration == 0`` breakdown (the OTHER test in this
        class). The loose ``0 < info < maxiter`` assertion passed either
        way, so this test's stated purpose -- verifying the floor is a
        no-op once the real iteration count is already >= 1 -- was
        unverified: a regression that hardcoded ``return 1`` or otherwise
        lost the actual iteration count would have passed unnoticed.

        Fix: ``degenerate_from=3`` (accounting for the setup call) so the
        first 2 PROJECTED-MATVEC calls -- #2 (iteration 0) and #3
        (iteration 1) -- behave normally and the breakdown first strikes
        on call #4 (iteration 2's first attempt); assert ``info == 2``
        exactly instead of the loose range check."""

        class _DelayedDegenerateStub:
            SZ = np.zeros((1, 1))

            def __init__(self, degenerate_from: int):
                self._degenerate_from = degenerate_from
                self._calls = 0

            def apply_with_SQ(self, v):
                v = np.asarray(v, dtype=np.float64)
                # Q = 0 (no real deflation) until `_calls` reaches the
                # trigger point, matching _AlwaysDegenerateProjectedMatvecStub
                # exactly from then on (SQr = r -- collapses the projected
                # matvec S p - S Q(S p) to identically zero).
                self._calls += 1
                if self._calls > self._degenerate_from:
                    return np.zeros_like(v), v.copy()
                return np.zeros_like(v), np.zeros_like(v)

            def apply(self, v):
                return np.zeros_like(np.asarray(v, dtype=np.float64))

        n = 3
        S = np.diag([1.0, 2.0, 5.0])
        matvec = lambda v: S @ np.asarray(v, dtype=np.float64)
        base_apply = lambda v: np.asarray(v, dtype=np.float64).copy()
        b = np.array([1.0, 1.0, 1.0])
        maxiter = 20
        # Round-3 fix (finding 7): degenerate_from=3, NOT 2 -- accounts
        # for _deflated_pcg's SETUP apply_with_SQ call (projecting the
        # initial residual, before the loop's first iteration), which the
        # pre-fix value of 2 missed. Call sequence: #1 setup (normal),
        # #2 iteration 0's Sp (normal), #3 iteration 1's Sp (normal --
        # self._calls=3 <= 3), #4 iteration 2's Sp (DEGENERATE --
        # self._calls=4 > 3) -- so the first 2 PROJECTED-MATVEC calls
        # (#2, #3 -- iterations 0-1) behave normally (Q=0, plain diagonal
        # CG) and the breakdown deterministically first strikes on call
        # #4 (iteration 2's first attempt), well before this diagonal
        # system would otherwise converge in n=3 exact CG steps.
        coarse = _DelayedDegenerateStub(degenerate_from=3)
        x, info = _deflated_pcg(
            matvec, base_apply, coarse, b, None, rtol=1e-300, atol=0.0,
            maxiter=maxiter, reproject_every=0,
        )
        # Exact assertion (not the pre-fix loose `0 < info < maxiter`,
        # which could not distinguish the actual iteration count from the
        # max(iteration, 1) floor value -- see this test's docstring):
        # the double-breakdown exit reports max(iteration, 1) with
        # iteration == 2 (the retry -- p was NOT already z at iteration 2,
        # since beta != 0 -- also degenerates, on call #5), so info must
        # be exactly 2.
        assert info == 2, (
            f"expected the double-breakdown exit to report the ACTUAL "
            f"iteration count (2) -- the max(iteration, 1) floor is a "
            f"no-op here since iteration >= 1 -- got info={info} "
            f"(maxiter={maxiter})"
        )
        assert np.all(np.isfinite(x))


class TestDoubleBreakdownRegressionInfoZero:
    """Round-2 code review finding 1 (CRITICAL): end-to-end regression
    coverage, through the REAL ``InterfaceCGSolver.__call__`` (not the
    bare ``_deflated_pcg`` function), that a first-iteration double
    breakdown is reported as a FAILURE, not scipy's success code.

    Negative-test evidence (see the round-2 fix agent's final report for
    the full revert/restore transcript): reverting
    ``return _recover_x(), max(iteration, 1)`` back to
    ``return _recover_x(), iteration`` in ``_deflated_pcg`` makes EVERY
    assertion in :meth:`test_stub_scenario_reports_failure_not_success`
    FAIL -- ``info`` becomes ``0``, no ``RuntimeError`` is raised despite
    ``strict=True``, ``stats['cg_failed']`` is ``False``, and the
    ``x=[0,0,0,0]`` non-solution gets pushed into the warm-start history.
    """

    def test_stub_scenario_reports_failure_not_success(self):
        tile_schur, tile_idx, n = _chain_tiles(4)
        solver = InterfaceCGSolver(
            n_interface=n, matvec_mode='tilewise',
            tile_schur_complements={k: v.copy() for k, v in tile_schur.items()},
            tile_index_maps=tile_idx,
            preconditioner='jacobi', matvec_threads=1, strict=True,
            rtol=1e-30, atol=1e-30, maxiter=7,
            interface_coarse_apply_mode='deflated',
        )
        try:
            # Force the deflated hand-rolled loop with the SAME
            # deterministic double-breakdown stub _deflated_pcg's own
            # unit tests use, wired directly into a real
            # InterfaceCGSolver -- 'jacobi' base (cheap, no real coarse
            # build attempted) + a duck-typed CoarseSpace substituted
            # post-construction so `_use_deflated` in __call__ (checks
            # apply_mode == 'deflated' and self._coarse is not None and
            # self._coarse.SZ is not None) dispatches to _deflated_pcg.
            solver._coarse = _AlwaysDegenerateProjectedMatvecStub()
            solver._linear_op.matvec = lambda v: np.asarray(v, dtype=np.float64).copy()
            solver._M_base_apply = lambda v: np.asarray(v, dtype=np.float64).copy()

            # Pre-seed warm-start history so we can positively verify it
            # gets CLEARED (not merely "never populated") on this failure
            # -- distinguishes "hygiene ran" from "hygiene was a no-op".
            solver._x_hist_prev = np.array([9.0, 9.0, 9.0, 9.0])
            solver._x_hist_prev2 = np.array([8.0, 8.0, 8.0, 8.0])

            b = np.array([1.0, 2.0, 3.0, 4.0])
            with pytest.raises(RuntimeError, match='did not converge'):
                solver(b)

            assert solver.stats['last_cg_info'] > 0, (
                f"expected info > 0 (FAILURE) for this deterministically-"
                f"stuck-on-iteration-0 scenario, got "
                f"{solver.stats['last_cg_info']} -- info==0 here is "
                f"exactly the false-success regression this test guards "
                f"against (the double-breakdown exit collided with "
                f"scipy's success code on iteration==0)"
            )
            assert solver.stats['cg_failed'] is True
            assert solver.stats['total_cg_failures'] >= 1

            # Warm-start/extrapolation history hygiene (Finding 2, round 1
            # -- and Finding 8, round 2, which moved this to run BEFORE
            # the strict-mode raise): a failed solve must NOT be pushed
            # into the two-point extrapolation history, and any
            # PRE-EXISTING history must be cleared, not left stale.
            assert solver._x_hist_prev is None, (
                "warm-start extrapolation history was not cleared on a "
                "failed solve"
            )
            assert solver._x_hist_prev2 is None, (
                "warm-start extrapolation history was not cleared on a "
                "failed solve"
            )
            # _x0 is still seeded with the plain best iterate (never
            # worse than pre-extrapolation behaviour) -- just not via
            # push_solution_history's extrapolating path.
            assert solver._x0 is not None
        finally:
            solver.close()


class TestReprojectEveryNegativeDisables:
    """Spec-compliance review round 3 (minor finding): the docstring,
    ``InterfaceCGSolver`` param doc, and CLI help all say
    ``interface_deflated_reproject_every <= 0`` disables reprojection, but the
    old guard (``if reproject_every and iteration > 0 and iteration %
    reproject_every == 0``) only treated exactly ``0`` as falsy -- a
    NEGATIVE value is truthy in Python, and ``iteration % (-1) == 0`` for
    EVERY integer ``iteration`` (Python's ``%`` follows the divisor's
    sign), so ``reproject_every=-1`` reprojected on EVERY iteration instead
    of disabling -- the opposite of "disable". This is a spy-based
    regression test (not a solution-value check, since reprojection is
    numerically harmless -- see :class:`TestReprojectionDrift` -- so a
    solution-only test cannot distinguish "disabled" from "fires every
    iteration"): it counts calls to ``CoarseSpace.apply_with_SQ`` and
    asserts ``reproject_every=-1`` costs exactly as many as the
    known-good-disable value ``reproject_every=0``, not more.
    """

    @staticmethod
    def _count_apply_with_sq_calls(coarse, matvec, base_apply, b,
                                    reproject_every, maxiter):
        # CoarseSpace uses __slots__ (no per-instance apply_with_SQ
        # override possible) -- patch the CLASS method instead, via a
        # wrapper that still dispatches to the original unbound function.
        count = [0]
        orig = ic.CoarseSpace.apply_with_SQ

        def counting_apply_with_sq(self, r):
            count[0] += 1
            return orig(self, r)

        with mock.patch.object(
            ic.CoarseSpace, 'apply_with_SQ', counting_apply_with_sq,
        ):
            _deflated_pcg(
                matvec, base_apply, coarse, b, None, rtol=1e-300,
                atol=0.0,  # never converges -- exhausts the full maxiter
                maxiter=maxiter, reproject_every=reproject_every,
            )
        return count[0]

    def test_negative_reproject_every_costs_same_as_disabled(self):
        tile_schur, tile_idx, S_extra, S, n = _ill_conditioned_jacobi_fixture()
        rng = np.random.default_rng(11)
        b = rng.standard_normal(n)
        maxiter = 8

        solver = InterfaceCGSolver(
            n_interface=n, matvec_mode='tilewise',
            tile_schur_complements=tile_schur, tile_index_maps=tile_idx,
            S_extra=S_extra,
            preconditioner='two_level', rtol=1e-9, atol=1e-14,
            matvec_threads=1, strict=False, maxiter=maxiter,
            block_jacobi_max_bytes=1,
            interface_coarse_apply_mode='deflated',
        )
        matvec = solver._linear_op.matvec
        base_apply = solver._M_base_apply
        coarse = solver._coarse
        try:
            n_disabled = self._count_apply_with_sq_calls(
                coarse, matvec, base_apply, b, reproject_every=0,
                maxiter=maxiter,
            )
            n_negative = self._count_apply_with_sq_calls(
                coarse, matvec, base_apply, b, reproject_every=-1,
                maxiter=maxiter,
            )
            # Pre-fix this was 1 (init) + maxiter (per-iteration SQp) +
            # (maxiter - 1) (reprojection firing on every iteration > 0,
            # since x % -1 == 0 always) = 16 for maxiter=8, vs 9 disabled --
            # a clear, deterministic discriminator, not a fuzzy timing
            # measurement.
            assert n_negative == n_disabled, (
                f"reproject_every=-1 made {n_negative} apply_with_SQ calls "
                f"vs {n_disabled} for the documented-equivalent "
                f"reproject_every=0 -- a negative value must disable "
                f"reprojection exactly like 0, not fire extra reprojections"
            )
            # Also confirm a genuinely positive interval still reprojects
            # (guards against a degenerate fix that disables reprojection
            # unconditionally).
            n_positive = self._count_apply_with_sq_calls(
                coarse, matvec, base_apply, b, reproject_every=2,
                maxiter=maxiter,
            )
            assert n_positive > n_disabled
        finally:
            solver.close()


class TestMaxiterBoundaryConvergence:
    """Spec-compliance review round 2 (minor finding): the ``for`` loop in
    :func:`_deflated_pcg` used to check convergence only at the TOP of each
    iteration (before that iteration's update), so a solve that first meets
    the tolerance exactly on the FINAL (``maxiter``-th) update fell through
    to ``return _recover_x(), maxiter`` -- reported as non-convergence
    (``info=maxiter``, strict mode raises) despite the residual actually
    meeting the bar. scipy.sparse.linalg.cg checks after every update
    including the last. Constructed by first running with a generous
    ``maxiter`` to discover the EXACT iteration count a fixture converges
    at, then re-running with ``maxiter`` pinned to that exact count."""

    @staticmethod
    def _iters_to_converge(matvec, base_apply, coarse, b, rtol, atol,
                            generous_maxiter=5000, reproject_every=0):
        count = [0]

        def cb(_x):
            count[0] += 1

        x, info = _deflated_pcg(
            matvec, base_apply, coarse, b, None, rtol=rtol, atol=atol,
            maxiter=generous_maxiter, reproject_every=reproject_every,
            callback=cb,
        )
        assert info == 0, f"fixture did not converge at all (info={info})"
        return count[0]

    def test_convergence_on_the_maxiter_th_iteration_reports_info_zero(self):
        tile_schur, tile_idx, S_extra, S, n = _ill_conditioned_jacobi_fixture(
            n=80, n_tiles=5, block=4, cond_log=5.0, seed=31,
        )
        rng = np.random.default_rng(31)
        b = rng.standard_normal(n)
        rtol, atol = 1e-8, 1e-14

        solver = InterfaceCGSolver(
            n_interface=n, matvec_mode='tilewise',
            tile_schur_complements=tile_schur, tile_index_maps=tile_idx,
            S_extra=S_extra,
            preconditioner='two_level', rtol=rtol, atol=atol,
            matvec_threads=1, strict=False, maxiter=5000,
            block_jacobi_max_bytes=1,
            interface_coarse_apply_mode='deflated',
        )
        matvec = solver._linear_op.matvec
        base_apply = solver._M_base_apply
        coarse = solver._coarse
        assert coarse.SZ is not None
        try:
            iters_needed = self._iters_to_converge(
                matvec, base_apply, coarse, b, rtol, atol,
            )
            assert iters_needed > 1, (
                "fixture converges too fast to exercise the maxiter "
                "boundary meaningfully"
            )

            # Pin maxiter to EXACTLY the iteration count the solve needs --
            # convergence now happens on the very last allowed iteration.
            x, info = _deflated_pcg(
                matvec, base_apply, coarse, b, None, rtol=rtol, atol=atol,
                maxiter=iters_needed, reproject_every=0,
            )
            assert info == 0, (
                f"solve converges in exactly {iters_needed} iterations but "
                f"maxiter={iters_needed} reported info={info} (expected 0 "
                f"-- the maxiter-th update's residual was never checked)"
            )
            true_r = float(np.linalg.norm(b - matvec(x)))
            assert true_r <= rtol * float(np.linalg.norm(b)), (
                f"reported converged (info=0) but true residual "
                f"{true_r:.3e} exceeds the rtol target"
            )

            # One iteration short of that must still fail to converge
            # (sanity check that iters_needed is truly the exact boundary,
            # not an off-by-one artifact of this test).
            x_short, info_short = _deflated_pcg(
                matvec, base_apply, coarse, b, None, rtol=rtol, atol=atol,
                maxiter=iters_needed - 1, reproject_every=0,
            )
            assert info_short != 0, (
                "maxiter one less than the true boundary unexpectedly "
                "converged -- iters_needed is not the exact boundary this "
                "test assumes"
            )
        finally:
            solver.close()


class TestFreshTrueResidualAcceptanceGate:
    """Spec-compliance review round 2 (minor finding): Design section 2
    numerical-hygiene item (b) requires the CONVERGED-path acceptance check
    to use the ORIGINAL system's residual (a fresh ``b - S @ x`` matvec),
    not the internally-tracked deflated recurrence ``r`` alone -- exactly
    like the strict-mode FAILURE path in ``InterfaceCGSolver.__call__``
    already does. Verified by wrapping ``matvec`` with a call counter: the
    total call count at acceptance must be exactly ``iters_needed + 1`` (one
    extra matvec beyond the per-iteration ``S @ p`` calls, fired once at
    the point convergence is first detected) -- the pre-fix code accepted
    on the tracked residual alone, with no extra matvec at acceptance
    (``iters_needed`` calls, not ``iters_needed + 1``)."""

    def test_acceptance_performs_one_extra_fresh_matvec(self):
        tile_schur, tile_idx, S_extra, S, n = _ill_conditioned_jacobi_fixture(
            n=80, n_tiles=5, block=4, cond_log=5.0, seed=41,
        )
        rng = np.random.default_rng(41)
        b = rng.standard_normal(n)
        rtol, atol = 1e-8, 1e-14

        solver = InterfaceCGSolver(
            n_interface=n, matvec_mode='tilewise',
            tile_schur_complements=tile_schur, tile_index_maps=tile_idx,
            S_extra=S_extra,
            preconditioner='two_level', rtol=rtol, atol=atol,
            matvec_threads=1, strict=False, maxiter=5000,
            block_jacobi_max_bytes=1,
            interface_coarse_apply_mode='deflated',
        )
        matvec = solver._linear_op.matvec
        base_apply = solver._M_base_apply
        coarse = solver._coarse
        assert coarse.SZ is not None
        try:
            # Cold start (x0=None) -> Sy0 is NOT computed via matvec (y0 is
            # exactly zero), so with reproject_every=0 every counted matvec
            # call is either one per CG iteration (S @ p) or one of the TWO
            # fresh acceptance-gate calls (round-3 code review finding 4:
            # _try_accept now also recomputes a fresh Sy = matvec(y) before
            # recovering x_candidate, on top of the pre-existing fresh
            # true-residual matvec on x_candidate itself -- see
            # _try_accept's docstring for why the refresh is scoped there
            # rather than into the shared recurrence state).
            call_count = [0]

            def counting_matvec(v):
                call_count[0] += 1
                return matvec(v)

            iters = [0]

            def cb(_x):
                iters[0] += 1

            x, info = _deflated_pcg(
                counting_matvec, base_apply, coarse, b, None, rtol=rtol,
                atol=atol, maxiter=5000, reproject_every=0, callback=cb,
            )
            assert info == 0
            assert call_count[0] == iters[0] + 2, (
                f"expected exactly two extra fresh matvecs at acceptance "
                f"(Sy = matvec(y) refresh, then the true-residual check on "
                f"x_candidate -- round-3 finding 4) beyond the {iters[0]} "
                f"per-iteration S@p calls; got {call_count[0]} total "
                f"matvec calls -- the converged path is not gated by a "
                f"fresh true-residual check"
            )
        finally:
            solver.close()

    def test_matvec_count_bounded_end_to_end(self):
        """Finding 9 (round-1 code review) regression: the fresh true-
        residual check must run ONLY when the TRACKED residual meets
        tolerance (debounced on the TRANSITION into that state, not
        re-run unconditionally on every iteration the tracked residual
        merely continues to sit at/below the bar) -- through the full
        ``InterfaceCGSolver.__call__`` -> ``_deflated_pcg`` path (not
        bypassing to the module-level function directly, unlike the
        sibling test above), on the same ill-conditioned fixture that
        needs many iterations. Total matvec count for a solve converging
        in ``k`` CG iterations must stay at ``k`` + a small constant, not
        grow with ``k`` (the per-iteration-unconditional-check bug would
        cost roughly 2x per iteration for as long as the tracked residual
        drifted below tolerance without confirming)."""
        tile_schur, tile_idx, S_extra, S, n = _ill_conditioned_jacobi_fixture()
        rng = np.random.default_rng(4)
        b = rng.standard_normal(n)
        solver = InterfaceCGSolver(
            n_interface=n, matvec_mode='tilewise',
            tile_schur_complements=tile_schur, tile_index_maps=tile_idx,
            S_extra=S_extra,
            preconditioner='two_level', rtol=1e-9, atol=1e-14,
            matvec_threads=1, strict=True, maxiter=3000,
            block_jacobi_max_bytes=1,  # force jacobi -- more iterations
            interface_coarse_apply_mode='deflated',
        )
        try:
            assert solver._coarse is not None and solver._coarse.SZ is not None
            orig_matvec = solver._linear_op.matvec
            calls = [0]

            def counting_matvec(v):
                calls[0] += 1
                return orig_matvec(v)

            solver._linear_op.matvec = counting_matvec
            x = solver(b)
            iters = solver.stats['last_cg_iters']
            assert iters > 20, (
                "fixture must run enough iterations for O(iters) vs O(1) "
                f"acceptance-check overhead to be unambiguous; only got "
                f"{iters}"
            )
            # One matvec per CG iteration (Sp = matvec(p)) plus a small
            # constant for acceptance-check overhead -- NOT O(iters),
            # which is what re-running the fresh check on every iteration
            # the tracked residual merely continues to sit below tolerance
            # (without a debounce) would cost.
            assert calls[0] <= iters + 5, (
                f"matvec called {calls[0]} times over {iters} CG "
                f"iterations -- expected <= iters + 5 (O(1) acceptance-"
                f"check overhead, not O(iters))"
            )
            true_r = float(np.linalg.norm(b - S @ x))
            assert true_r <= max(1e-9, 1e-9 * float(np.linalg.norm(b))), (
                f"solution incorrect: true residual {true_r:.3e}"
            )
        finally:
            solver.close()


class _BiasedZeroQStub:
    """Duck-typed ``CoarseSpace`` stand-in for :class:`TestDebounceReArm`:
    ``Q = 0`` always (``apply_with_SQ`` returns zeros -- no real
    deflection, so the tracked recurrence residual ``r`` is the plain
    (unpreconditioned-by-Q) CG residual, entirely independent of this
    stub's own bookkeeping), but ``apply`` -- which only ``_recover_x`` /
    ``_try_accept`` ever call -- returns a fixed bias instead of the
    mathematically-correct zero for the first ``n_disagree`` calls, then
    the correct zero from then on. This deterministically simulates
    "the tracked recurrence already meets tolerance, but the fresh
    true-residual check on the recovered candidate disagrees" -- the
    drift scenario the debounce/re-arm policy exists to bound -- without
    depending on hard-to-control real ``Sy``-drift mechanics."""

    SZ = np.zeros((1, 1))  # truthy "SZ retained" sentinel; never read directly

    def __init__(self, n_disagree: int, bias: float):
        self._n_disagree = n_disagree
        self._bias = bias
        self.calls = 0

    def apply_with_SQ(self, v):
        v = np.asarray(v, dtype=np.float64)
        return np.zeros_like(v), np.zeros_like(v)

    def apply(self, v):
        v = np.asarray(v, dtype=np.float64)
        self.calls += 1
        if self.calls <= self._n_disagree:
            return np.zeros_like(v) + self._bias
        return np.zeros_like(v)


class TestDebounceReArm:
    """Round-2 code review finding 3 (regression in round-1 Finding 9's
    debounce, exercised by :class:`TestFreshTrueResidualAcceptanceGate`
    above): a PURE edge-trigger ("only re-attempt the fresh true-residual
    check on a NEW transition into tracked-residual-below-tolerance")
    never RE-ARMS once it has fired once and disagreed. If the tracked
    residual then stays at/below tolerance for the rest of the solve (the
    common case -- Sy-drift is small and roughly monotonic, so once
    tracked and true residuals are close they tend to stay close), no
    SECOND fresh check is ever attempted: the loop burns every remaining
    iteration up to ``maxiter`` (or the single unconditional post-loop
    check) even though a fresh check a few iterations later would very
    likely have succeeded -- a ~20-iteration warm transient step turning
    into an ``O(n_interface)``-iteration stall or a strict-mode
    RuntimeError.

    Fix: re-arm the fresh check every ``_rearm_period`` iterations
    (``reproject_every`` when reprojection is enabled, else the module
    constant ``DEFLATED_DEBOUNCE_REARM_FALLBACK_ITERS`` = 10) while the
    tracked residual continues to sit at/below tolerance without being
    accepted -- bounding the "stuck disagreeing" cost to O(1) extra
    matvecs per re-arm period, not the full maxiter budget.

    Fixture design note: a REAL ill-conditioned CG solve's tracked
    residual can legitimately oscillate back above tolerance and re-cross
    (the Euclidean-norm residual is not monotonic in exact CG, only the
    A-norm error is) -- that natural re-crossing ALSO re-arms a pure
    edge-trigger, which would make a real-fixture-based test pass
    regardless of whether the re-arm fix is present (verified empirically
    while developing this test). Uses a small SYNTHETIC diagonal SPD
    system instead (well-separated, well-conditioned-enough eigenvalues,
    checked to decay smoothly/monotonically well short of the breakdown-
    prone near-machine-precision regime) specifically so the tracked
    residual crosses below tolerance ONCE and stays there monotonically
    for the rest of the solve -- isolating the re-arm mechanism as the
    ONLY possible source of a second fresh-check attempt.

    Negative-test evidence: temporarily reverting the acceptance gate to
    the pure edge-trigger (``if _is_below_tol and not _was_below_tol:``,
    dropping the ``or _iters_since_accept_attempt >= _rearm_period``
    re-arm clause) makes
    :meth:`test_persistent_disagreement_reaccepts_within_rearm_period`
    FAIL (``info`` becomes ``maxiter`` instead of ``0``, and
    ``proxy.calls`` stops at 3 instead of reaching 4) -- see the round-2
    fix agent's final report for the transcript.
    """

    def test_persistent_disagreement_reaccepts_within_rearm_period(self):
        n = 40
        rng = np.random.default_rng(1)
        # Well-separated, moderately ill-conditioned (cond ~ 200) diagonal
        # SPD system -- chosen (see class docstring) so unpreconditioned
        # CG's residual decays smoothly over ~20-40 iterations without
        # hitting the breakdown-prone near-machine-precision tail.
        d = np.linspace(1.0, 200.0, n)
        matvec = lambda v: np.asarray(v, dtype=np.float64).copy()  # S = I
        base_apply = lambda v: np.asarray(v, dtype=np.float64) / d
        b = rng.standard_normal(n)

        # 3 deliberately-wrong fresh-check attempts before the bias
        # clears. A pure edge-trigger gets exactly ONE mid-loop attempt
        # (at the first, and -- by this fixture's construction -- ONLY,
        # crossing) plus the single unconditional post-loop check: 2
        # attempts total, both within the N_DISAGREE=3 bias window, so it
        # reports non-convergence at maxiter. The re-arm fix instead
        # keeps retrying every `reproject_every` iterations while stuck
        # below tolerance, reaching (and succeeding on) the 4th, unbiased
        # attempt well before maxiter.
        N_DISAGREE = 3
        coarse = _BiasedZeroQStub(N_DISAGREE, bias=0.01)
        maxiter = 45
        x, info = _deflated_pcg(
            matvec, base_apply, coarse, b, None, rtol=0.0, atol=1e-4,
            maxiter=maxiter, reproject_every=5,
        )

        assert coarse.calls > N_DISAGREE, (
            f"expected the debounce to RE-ARM and retry the fresh check "
            f"past the {N_DISAGREE} deliberately-disagreeing attempts "
            f"(got only {coarse.calls} total attempts) -- a pure "
            f"edge-trigger that never re-arms tops out at 2 attempts here "
            f"(the initial crossing plus the single unconditional "
            f"post-loop check) and never tries again in between"
        )
        assert info == 0, (
            f"expected the solve to CONVERGE (info == 0) once the bias "
            f"clears on the {N_DISAGREE + 1}-th fresh-check attempt, well "
            f"before maxiter={maxiter} -- got info={info}. A pure "
            f"edge-trigger (never re-arms) reports info == maxiter here: "
            f"the tracked residual crosses below tolerance exactly once "
            f"and stays there for the rest of this fixture's solve, so no "
            f"second mid-loop attempt is ever made, and even the single "
            f"unconditional post-loop check is still within the "
            f"{N_DISAGREE}-call bias window"
        )
        true_r = float(np.linalg.norm(b - matvec(x)))
        assert true_r <= 1e-6, f"solution incorrect: true residual {true_r:.3e}"


class TestUnthrottledReArmOnOscillatingResidual:
    """Round-3 code review finding 6 (PLAUSIBLE, CONFIRMED): the round-2
    re-arm condition was ``not _was_below_tol or _iters_since_accept_
    attempt >= _rearm_period`` -- an EDGE trigger (``not _was_below_tol``)
    ORed with the period throttle, not a PURE period throttle. CG's
    Euclidean residual is not monotone, so on an ill-conditioned solve the
    tracked residual can cross above and back below ``atol_eff`` every
    couple of iterations; EVERY such re-crossing fires the edge-trigger
    REGARDLESS of how recently the last attempt ran, defeating the O(1)-
    extra-matvecs-per-period bound the debounce exists to guarantee.

    Fix: gate PURELY on ``_iters_since_accept_attempt >= _rearm_period``
    (this module's finding-6 fix) -- the counter now increments every
    iteration REGARDLESS of above/below-tolerance state, so consecutive
    attempts are always at least ``_rearm_period`` iterations apart, with
    no edge-trigger bypass.

    Fixture: reuses :class:`TestDebounceReArm`'s own precedent for why a
    hand-picked SYNTHETIC oscillating trajectory (rather than a real
    small dense system) would be needed for a CLEAN bound-violation demo
    -- but this finding specifically needs GENUINE non-monotone
    oscillation (the opposite of that class's monotone-by-design
    fixture), so this test instead uses a REAL, deliberately very
    ill-conditioned (cond ~1e18) unpreconditioned small dense SPD system
    (plain CG, ``Q=0`` coarse stub -- no deflection), whose Euclidean
    residual trajectory is VERIFIED (see the module-level scan performed
    while developing this test) to cross a chosen ``atol`` threshold many
    times (~40 crossings over 400 iterations at the pinned seed/threshold
    below) -- genuine CG behavior, not an artificially injected value.
    ``coarse.apply`` is permanently biased (never the correct answer), so
    every actual ``_try_accept()`` attempt fails and the loop runs the
    full ``maxiter`` -- isolating the test to HOW OFTEN accept is
    attempted, not whether it ever succeeds.

    Negative-test evidence: temporarily reverting the throttle to the
    round-2 edge-trigger-OR-period condition (``if _is_below_tol and
    (not _was_below_tol or _iters_since_accept_attempt >=
    _rearm_period):``, restoring the ``_was_below_tol`` bookkeeping)
    makes :meth:`test_matvec_count_bounded_under_oscillation` FAIL
    (measured: 64 attempts / 526 matvec calls instead of <= 55 attempts
    -- verified directly while implementing this fix; see this session's
    final report for the revert/restore transcript).
    """

    @staticmethod
    def _ill_conditioned_dense_fixture():
        rng = np.random.default_rng(3)
        n = 25
        # Extreme eigenvalue spread (~1e12) -- verified (module-level
        # scan) to make plain unpreconditioned CG's Euclidean residual
        # cross a threshold around 3.2e-5 roughly 40 times over 400
        # iterations, well before settling near machine precision.
        eigs = np.concatenate(
            [[1e-6], np.geomspace(1e-3, 1.0, n - 2), [1e6]]
        )
        Qm, _ = np.linalg.qr(rng.standard_normal((n, n)))
        S = Qm @ np.diag(eigs) @ Qm.T
        S = (S + S.T) / 2.0
        b = rng.standard_normal(n)
        return S, b

    class _ZeroQAlwaysWrongStub:
        """Q=0 for the projected matvec (plain, self-consistent CG on
        whatever ``matvec`` implements), but ``apply`` always returns a
        provably-wrong constant -- every ``_try_accept()`` call that
        actually runs is GUARANTEED to reject, so the loop always runs
        the full ``maxiter`` and the ONLY thing under test is how many
        times ``apply`` (hence ``_try_accept``) gets called."""

        SZ = np.zeros((1, 1))

        def __init__(self):
            self.apply_calls = 0

        def apply_with_SQ(self, v):
            v = np.asarray(v, dtype=np.float64)
            return np.zeros_like(v), np.zeros_like(v)

        def apply(self, v):
            v = np.asarray(v, dtype=np.float64)
            self.apply_calls += 1
            return np.zeros_like(v) + 1000.0

    def test_matvec_count_bounded_under_oscillation(self):
        S, b = self._ill_conditioned_dense_fixture()
        matvec = lambda v: S @ np.asarray(v, dtype=np.float64)
        base_apply = lambda v: np.asarray(v, dtype=np.float64).copy()
        coarse = self._ZeroQAlwaysWrongStub()

        matvec_calls = [0]

        def counting_matvec(v):
            matvec_calls[0] += 1
            return matvec(v)

        maxiter = 400
        rearm_period = 5
        x, info = _deflated_pcg(
            counting_matvec, base_apply, coarse, b, None, rtol=0.0,
            atol=3.2e-5, maxiter=maxiter, reproject_every=rearm_period,
        )
        assert info == maxiter, (
            "test precondition: the permanently-biased apply() must "
            "make every attempt fail, so the loop runs the full maxiter "
            f"-- got info={info}"
        )
        # Measured with the fix: 50 attempts / 498 matvec calls. Pinned
        # bound (55) sits comfortably above the measured value (numerical
        # noise margin) but well below the pre-fix regressed count (64) --
        # see this class's docstring for the revert/restore evidence.
        assert coarse.apply_calls <= 55, (
            f"expected the pure-period throttle to keep the number of "
            f"fresh-accept-check attempts close to maxiter/rearm_period "
            f"({maxiter}/{rearm_period}={maxiter // rearm_period}) despite "
            f"the tracked residual oscillating across atol_eff many times "
            f"-- got {coarse.apply_calls} attempts (pre-fix edge-trigger "
            f"regression measured 64 on this identical fixture), "
            f"{matvec_calls[0]} total matvec calls"
        )


class TestStaleRnormAtBreakdownExit:
    """Round-2 code review finding 4 (PLAUSIBLE, CONFIRMED by code
    inspection): the double-breakdown "genuinely stuck" exit gates its
    last-chance acceptance on ``rnorm`` -- computed at the TOP of the
    iteration, BEFORE the re-projection block (which may replace ``r``
    later in the SAME iteration) runs. If re-projection lands on a
    breakdown iteration and reveals that the (freshly recomputed) ``r``
    now meets ``atol_eff`` -- even though the STALE pre-reprojection
    value did not -- the pre-fix exit compared against the stale value,
    skipped ``_try_accept()`` entirely, and reported failure (``info !=
    0``, a strict-mode RuntimeError) for a solve whose true residual (per
    the fresh check any ``_try_accept()`` call would have performed) was
    already converged.

    Fix: recompute ``rnorm`` immediately after the re-projection block
    updates ``r``, so any later use within the same iteration (the
    double-breakdown exit's ``if rnorm <= atol_eff:`` gate) sees the
    CURRENT residual, not the value from before re-projection ran.

    Verifying the scenario is real: between top-of-loop and the exit,
    the ONLY code that can replace ``r`` is the re-projection block
    (nothing else touches ``r`` before the exit check) -- so a
    re-projection landing on an iteration whose ``pw`` breakdown-guard
    ALSO triggers is a real, reachable interleaving, not a hypothetical
    one; this test forces exactly that interleaving deterministically.

    Negative-test evidence: temporarily removing the
    ``rnorm = float(np.linalg.norm(r))`` line that follows the
    re-projection block's ``r = (b - Sy) - _SQr_rp`` (reverting to the
    stale top-of-loop value) makes
    :meth:`test_reprojection_reveals_convergence_on_breakdown_iteration`
    FAIL (``info`` becomes non-zero -- the exit denies the last-chance
    accept it should have granted) -- see the round-2 fix agent's final
    report for the transcript.
    """

    class _StaleRnormStub:
        """Duck-typed ``CoarseSpace``, call-count-sequenced to force: (1)
        several ordinary (non-degenerate, Q=0) iterations: (2) on a
        chosen re-projection call, an ARTIFICIAL "reveal" that snaps the
        reprojected ``r`` down to a tiny value regardless of the actual
        ``b - Sy`` passed in (simulating re-projection correcting drift
        that the incremental recurrence had not); (3) from that call
        onward, every projected-matvec call degenerates (``SQv = v``
        exactly, so ``w = Sp - SQp = 0``) -- guaranteeing the
        double-breakdown exit fires on the SAME iteration the reveal
        happened, exercising exactly the interleaving under test.

        ``apply`` always returns its input unchanged: with ``matvec`` =
        identity (``S = I``) in this test, ``_recover_x() = y +
        apply(b - S y) = y + (b - y) = b`` EXACTLY, independent of
        ``y`` -- so ``_try_accept()`` trivially succeeds THE MOMENT it
        is actually invoked. This isolates the test to "was
        ``_try_accept`` called at all with the right ``rnorm``", not
        "did the recovered iterate happen to be numerically close
        enough" -- the property under test is purely about the exit's
        control flow, not CG numerics.
        """

        SZ = np.zeros((1, 1))

        def __init__(self, reveal_at_call: int, tiny: np.ndarray):
            self.calls = 0
            self._reveal_at_call = reveal_at_call
            self._tiny = tiny

        def apply_with_SQ(self, v):
            v = np.asarray(v, dtype=np.float64)
            self.calls += 1
            if self.calls < self._reveal_at_call:
                return np.zeros_like(v), np.zeros_like(v)
            if self.calls == self._reveal_at_call:
                # The chosen re-projection call: snap r down to `tiny`
                # regardless of the actual (b - Sy) passed in.
                return np.zeros_like(v), v - self._tiny
            # Every call from here on degenerates the projected matvec.
            return np.zeros_like(v), v.copy()

        def apply(self, v):
            return np.asarray(v, dtype=np.float64).copy()

    def test_reprojection_reveals_convergence_on_breakdown_iteration(self):
        n = 12
        rng = np.random.default_rng(0)
        # Well-separated diagonal preconditioner (matvec itself is
        # identity, S = I) so unpreconditioned-by-Q PCG needs several
        # iterations to converge -- gives the top-of-loop check room to
        # see a genuinely NOT-yet-converged tracked residual before the
        # engineered re-projection reveal fires.
        d = np.array([1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0, 34.0, 55.0,
                      89.0, 144.0, 233.0])
        matvec = lambda v: np.asarray(v, dtype=np.float64).copy()
        base_apply = lambda v: np.asarray(v, dtype=np.float64) / d
        b = rng.standard_normal(n)

        # Call sequence with reproject_every=1 (reprojects every
        # iteration once iteration > 0): #1 initial r0 setup, #2 iter0's
        # Sp, #3 iter1's reprojection, #4 iter1's Sp, #5 iter2's
        # reprojection, #6 iter2's Sp, #7 iter3's reprojection (chosen
        # reveal point -- tracked r at iter3's TOP-of-loop check is still
        # far above atol_eff=1e-6 on this fixture, so the ordinary
        # top-of-loop debounce check does NOT fire early), #8/#9 iter3's
        # Sp attempts (both degenerate -- double breakdown).
        tiny = np.full(n, 1e-16)
        coarse = self._StaleRnormStub(reveal_at_call=7, tiny=tiny)
        x, info = _deflated_pcg(
            matvec, base_apply, coarse, b, None, rtol=0.0, atol=1e-6,
            maxiter=20, reproject_every=1,
        )
        assert info == 0, (
            f"expected the double-breakdown exit to ACCEPT using the "
            f"freshly-reprojected residual (info == 0) -- got info="
            f"{info}. A stale-rnorm exit compares against the "
            f"PRE-reprojection value (still above atol_eff at this "
            f"iteration on this fixture) and denies the last-chance "
            f"accept even though the reprojected residual -- and any "
            f"fresh _try_accept() check -- would confirm convergence"
        )
        assert np.array_equal(x, b), (
            "expected the trivially-always-correct _recover_x() result "
            "(see _StaleRnormStub.apply's docstring) -- got a different "
            "x, meaning _try_accept() was never actually invoked"
        )


class TestBreakdownGuardMagnitudeAware:
    """Finding 10 (round-1 code review, PLAUSIBLE verdict): _deflated_pcg's
    CG-breakdown guards (``rho_prev``, ``pw``) must be magnitude/sign-aware
    (``value <= eps * scale``), not exact float equality to 0.0 -- a PSD
    operator's dot products land NEAR, not AT, zero on ill-conditioned/
    penalty-heavy systems (occasionally slightly NEGATIVE from FP noise),
    and exact-equality guards let alpha/beta divide by a tiny-but-nonzero
    value instead of triggering the restart path. Direct unit test of the
    module-level ``_is_breakdown`` guard (the branch itself), per the
    review's own fallback instruction ("unit-test the guard branch
    directly") -- a naturally-occurring degenerate CG iterate is not
    cheaply constructible as an end-to-end fixture."""

    def test_exact_zero_is_breakdown(self):
        assert _is_breakdown(0.0, 1.0) is True

    def test_small_negative_is_breakdown(self):
        """The exact-equality pre-fix guard (``value == 0.0``) would treat
        this as NOT a breakdown, letting a caller divide by a tiny
        negative value -- a PSD dot product going slightly negative is
        FP noise, not a legitimate operator response."""
        assert _is_breakdown(-1e-18, 1.0) is True

    def test_tiny_positive_below_noise_floor_is_breakdown(self):
        scale = 1.0
        below_floor = 0.5 * _BREAKDOWN_EPS * scale
        assert _is_breakdown(below_floor, scale) is True

    def test_genuinely_positive_value_is_not_breakdown(self):
        assert _is_breakdown(1.0, 1.0) is False
        # Comfortably above the noise floor at a realistic problem scale.
        assert _is_breakdown(1e-6, 1.0) is False

    def test_threshold_scales_with_vector_norms(self):
        """The SAME raw value must flip from "not breakdown" to
        "breakdown" as the natural scale (norm(a)*norm(b)) grows -- an
        exact-equality guard is blind to this scale entirely."""
        value = 1e-10
        assert _is_breakdown(value, 1.0) is False
        assert _is_breakdown(value, 1e10) is True

    def test_deflated_pcg_survives_near_degenerate_search_direction(self):
        """Near-degenerate (not exactly singular) fixture: an SPD system
        with one eigenvalue many orders of magnitude below the rest, so
        rounding noise in a poorly-aligned coarse space can push a CG dot
        product to a tiny (possibly slightly negative) value without ever
        being EXACTLY 0.0 in floating point -- the magnitude-aware guard
        must still restart cleanly and converge, not silently diverge via
        a near-zero division."""
        rng = np.random.default_rng(13)
        n = 12
        Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
        eigs = np.concatenate([[1e-6], np.full(n - 1, 1.0)])
        S = (Q * eigs) @ Q.T
        S = 0.5 * (S + S.T)

        # A trivial 1-column coarse space (T'=1) whose column is NEARLY
        # (not exactly) the near-null eigenvector of S -- deliberately
        # perturbed so P = I - S Q is a genuine oblique projector (not a
        # no-op), while still deflating enough of the bad eigendirection
        # that a plain-identity base preconditioner can converge on the
        # well-conditioned remainder. The perturbation is exactly what
        # produces near-degenerate (not exactly zero) CG dot products.
        z_col = Q[:, 0] + 1e-9 * rng.standard_normal(n)
        z_col /= np.linalg.norm(z_col)
        Z = z_col.reshape(-1, 1)
        SZ = S @ Z
        Sc = Z.T @ SZ
        w, V = np.linalg.eigh(Sc)
        inv_lambda_c = 1.0 / np.where(w > 0, w, 1e-300)

        coarse = ic.CoarseSpace(
            Z=sp.csr_matrix(Z), V_c=V, inv_lambda_c=inv_lambda_c,
            n_pou_cols=1, n_geneo_cols=0, n_dropped_cols=0, rank=1,
            cond_estimate=1.0, col_labels=['col0'], SZ=SZ,
        )

        b = rng.standard_normal(n)
        x_direct = np.linalg.solve(S, b)
        # b has a real component along the near-null eigendirection, so
        # x_direct's norm is ~1e6 (amplified by the ~1e-6 eigenvalue) --
        # compare RELATIVE error, not an absolute tolerance sized for an
        # O(1) solution.

        x, info = _deflated_pcg(
            lambda v: S @ v, lambda v: v, coarse, b, None,
            rtol=1e-8, atol=1e-12, maxiter=500, reproject_every=10,
        )
        assert info == 0, f"must converge (not stall/diverge), info={info}"
        assert np.all(np.isfinite(x)), "breakdown must not inject NaN/Inf"
        rel_err = np.linalg.norm(x - x_direct) / np.linalg.norm(x_direct)
        assert rel_err < 1e-6, f"relative error too large: {rel_err:.3e}"


class TestIslandADef2:
    """Island + adef2: penalized-island fixture, adef2 solution matches
    direct, island rows land at ~Vdd."""

    def test_island_row_pinned_and_matches_direct(self):
        tile_schur, tile_idx, n = _chain_tiles(15)
        vdd = 1.0
        island_local = 5  # an interior chain port, arbitrary
        island_idx = np.array([island_local], dtype=np.int64)
        penalty = 1e5
        extra_diag = np.zeros(n)
        extra_diag[island_local] = penalty
        S_extra = sp.diags(extra_diag).tocsr()

        S = np.zeros((n, n))
        for tid, idx in tile_idx.items():
            S[np.ix_(idx, idx)] += tile_schur[tid]
        S += np.diag(extra_diag)

        rng = np.random.default_rng(8)
        b = rng.standard_normal(n)
        # Mirror apply_island_penalty's RHS convention: penalty * vdd added
        # at the penalized row so the island settles near vdd.
        b[island_local] += penalty * vdd
        x_direct = np.linalg.solve(S, b)
        assert abs(x_direct[island_local] - vdd) < 1e-3, (
            "sanity: fixture's island row must actually settle near vdd"
        )

        solver = InterfaceCGSolver(
            n_interface=n, matvec_mode='tilewise',
            tile_schur_complements={k: v.copy() for k, v in tile_schur.items()},
            tile_index_maps=tile_idx, S_extra=S_extra,
            preconditioner='two_level', rtol=1e-12, atol=1e-18,
            matvec_threads=1, strict=True, maxiter=5000,
            island_idx=island_idx, interface_coarse_apply_mode='deflated',
        )
        try:
            assert solver._coarse is not None
            assert solver._coarse.SZ is not None
            assert np.all(solver._coarse.Z.toarray()[island_local, :] == 0.0), (
                "island row must be zeroed out of Z (existing PoU/GenEO "
                "invariant) even under apply_mode='deflated'"
            )
            x = solver(b)
            err = np.max(np.abs(x - x_direct))
            # Finding-5-style ~1e5 mS island penalty amplifies the physical
            # condition number (production ISLAND_PENALTY_CONDUCTANCE is
            # ALSO 1e5, see pgmath.schur) -- a CG residual meeting rtol=
            # 1e-12 still amplifies to ~1e-7-1e-6 solution-level error
            # through S's conditioning (verified: measured ~4.4e-7 here);
            # this is inherent to the fixture/production penalty scale, not
            # an adef2-specific weakness -- 1e-5 leaves a comfortable margin
            # above the measured value while still being a tight bound.
            assert err <= 1e-5, f"adef2 vs direct err={err:.3e}"
            assert abs(x[island_local] - vdd) < 1e-3, (
                f"island row {x[island_local]!r} not pinned near vdd={vdd}"
            )
        finally:
            solver.close()


class TestFp32ADef2:
    """fp32 + adef2: converges, accuracy bound as the existing fp32
    additive test (test_fp32_storage_path_converges)."""

    def test_fp32_adef2_converges_at_documented_tolerance(self):
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
            interface_coarse_apply_mode='deflated',
        )
        try:
            assert solver._coarse is not None
            assert solver._coarse.SZ is not None, (
                "SZ must be retained in fp64 regardless of matvec_dtype "
                "(see build_coarse_space's fp64-accumulating matmat note)"
            )
            assert solver._coarse.SZ.dtype == np.float64
            x = solver(b)
            rel = np.max(np.abs(x - x_direct)) / max(1e-300, np.max(np.abs(x_direct)))
            assert rel < 1e-3, f"fp32 adef2 rel err {rel:.3e}"
        finally:
            solver.close()


class TestFp32DeflatedRtolGuard:
    """Round-3 code review finding 1 (CONFIRMED -- option (a) chosen: raise
    the minimum permitted rtol for deflated+fp32 rather than a bounded-
    disagreement/WARNING escape hatch; see FP32_MATVEC_MIN_RTOL_DEFLATED's
    module-level docstring in interface_iterative.py for the full
    justification): with matvec_dtype='float32' and apply_mode='deflated',
    EVERY acceptance (not just a failure-branch diagnostic, unlike the
    additive/scipy path) is gated on a fresh true residual computed
    through the same fp32 tilewise matvec whose documented floor is
    ~1e-7 relative. At the plain FP32_MATVEC_MIN_RTOL floor (1e-7) that
    gate can sit persistently at the fp32 noise floor and never confirm
    convergence, even though CG's own tracked residual has genuinely
    converged -- turning a converging deflated solve into a guaranteed
    strict-mode RuntimeError after burning the whole maxiter budget. The
    guard now enforces a one-decade-looser floor
    (FP32_MATVEC_MIN_RTOL_DEFLATED = 1e-6) specifically when
    apply_mode='deflated'; the additive/plain path's floor
    (FP32_MATVEC_MIN_RTOL = 1e-7) is UNCHANGED (no regression to the
    existing behaviour for every other mode).

    Negative-test evidence: temporarily reverting the guard in
    InterfaceCGSolver.__init__ to always compare against
    FP32_MATVEC_MIN_RTOL (dropping the ``self._apply_mode == 'deflated'``
    branch that selects FP32_MATVEC_MIN_RTOL_DEFLATED instead) makes
    :meth:`test_rtol_1e7_deflated_fp32_is_now_rejected` FAIL (construction
    succeeds silently at rtol=1e-7 instead of raising ValueError) -- this
    was verified directly while implementing the fix (see this session's
    final report)."""

    @staticmethod
    def _tiny_tilewise_kwargs():
        tile_schur, tile_idx, n = _chain_tiles(4)
        return dict(
            n_interface=n, matvec_mode='tilewise',
            tile_schur_complements={k: v.copy() for k, v in tile_schur.items()},
            tile_index_maps=tile_idx,
            preconditioner='two_level', matvec_threads=1,
        )

    def test_rtol_1e7_additive_fp32_still_permitted(self):
        """Additive path's floor is UNCHANGED at 1e-7 -- no regression."""
        solver = InterfaceCGSolver(
            rtol=1e-7, atol=0.0, maxiter=10,
            matvec_dtype='float32', interface_coarse_apply_mode='additive',
            **self._tiny_tilewise_kwargs(),
        )
        solver.close()

    def test_rtol_1e7_deflated_fp32_is_now_rejected(self):
        """deflated + fp32 at rtol == FP32_MATVEC_MIN_RTOL (1e-7, the
        plain floor) must now be REJECTED -- this exact combination is
        what made deflated solves unreachable at that rtol (finding 1)."""
        with pytest.raises(ValueError, match=r'requires rtol'):
            InterfaceCGSolver(
                rtol=1e-7, atol=0.0, maxiter=10,
                matvec_dtype='float32',
                interface_coarse_apply_mode='deflated',
                **self._tiny_tilewise_kwargs(),
            )

    def test_rtol_1e6_deflated_fp32_permitted(self):
        """One decade looser (1e-6 == FP32_MATVEC_MIN_RTOL_DEFLATED) is
        accepted -- the guard is a floor, not a blanket rejection of
        deflated+fp32."""
        solver = InterfaceCGSolver(
            rtol=1e-6, atol=0.0, maxiter=10,
            matvec_dtype='float32', interface_coarse_apply_mode='deflated',
            **self._tiny_tilewise_kwargs(),
        )
        solver.close()

    def test_strict_dtype_rtol_false_overrides_deflated_floor_too(self):
        """strict_dtype_rtol=False still bypasses the guard entirely for
        deflated mode, same override contract as the plain floor."""
        solver = InterfaceCGSolver(
            rtol=1e-7, atol=0.0, maxiter=10,
            matvec_dtype='float32', interface_coarse_apply_mode='deflated',
            strict_dtype_rtol=False,
            **self._tiny_tilewise_kwargs(),
        )
        solver.close()

    def test_deflated_fp32_at_permitted_rtol_actually_converges(self):
        """End-to-end: at the new floor (1e-6), a deflated+fp32 solve
        actually converges (not just passes construction) -- the whole
        point of raising the floor is that the fresh-residual acceptance
        gate becomes reliably attainable, not merely permitted. Same
        fixture/settings as TestFp32ADef2's own converging test (proven
        stable there) -- this test's purpose is the rtol-floor guard, not
        re-proving fp32+deflated accuracy from scratch."""
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
            matvec_dtype='float32', interface_coarse_apply_mode='deflated',
            interface_coarse_geneo_k=4,
        )
        try:
            assert solver._coarse is not None and solver._coarse.SZ is not None
            x = solver(b)
            assert solver.stats['last_cg_info'] == 0, (
                "sanity: this must actually converge (info == 0), not "
                "just avoid raising"
            )
            rel = np.max(np.abs(x - x_direct)) / max(1e-300, np.max(np.abs(x_direct)))
            assert rel < 1e-3, f"deflated fp32 rel err {rel:.3e}"
        finally:
            solver.close()


class TestWarmStartExtrapolation:
    """Extrapolation: transient-like sequence of slowly (linearly) varying
    RHS: extrapolation on <= iters off; exactness unchanged (results
    identical to rtol-consistent tolerance).

    Measured (seed 555, chain(30), 10 linearly-drifting RHS steps):
    extrapolation off total iters=235 ([26,24,23,23,23,24,23,23,23,23]);
    on total=58 ([26,24,1,1,0,2,0,2,0,2]) -- from step 3 onward (two prior
    solves recorded) the linear trend is captured almost exactly.
    """

    @staticmethod
    def _run(extrap, n_steps=10):
        tile_schur, tile_idx, n = _chain_tiles(30)
        rng = np.random.default_rng(555)
        b0 = rng.standard_normal(n)
        direction = rng.standard_normal(n) * 0.05
        solver = InterfaceCGSolver(
            n_interface=n, matvec_mode='tilewise',
            tile_schur_complements={k: v.copy() for k, v in tile_schur.items()},
            tile_index_maps=tile_idx,
            preconditioner='two_level', rtol=1e-10, atol=1e-16,
            matvec_threads=1, strict=True, maxiter=5000,
            warm_start_extrapolation=extrap,
        )
        iters = []
        xs = []
        try:
            for step in range(n_steps):
                b = b0 + step * direction
                x = solver(b)
                iters.append(solver.stats['last_cg_iters'])
                xs.append(x.copy())
        finally:
            solver.close()
        return iters, xs

    def test_extrapolation_reduces_total_iterations(self):
        iters_off, xs_off = self._run(False)
        iters_on, xs_on = self._run(True)
        assert sum(iters_on) < sum(iters_off), (iters_on, iters_off)
        # From the third solve onward (two solves recorded), per-step iters
        # with extrapolation must not exceed the no-extrapolation baseline.
        for step in range(2, len(iters_off)):
            assert iters_on[step] <= iters_off[step], (
                step, iters_on, iters_off,
            )

    def test_exactness_unchanged(self):
        _, xs_off = self._run(False)
        _, xs_on = self._run(True)
        for step, (xo, xn) in enumerate(zip(xs_off, xs_on)):
            np.testing.assert_allclose(
                xo, xn, atol=1e-6, rtol=1e-6,
                err_msg=f"step {step}: extrapolation changed the converged solution",
            )

    def test_push_solution_history_falls_back_to_x_prev_for_first_two_steps(self):
        """Unit-level check of push_solution_history's documented fallback
        (first two solves -> plain x_prev seed, not extrapolated)."""
        tile_schur, tile_idx, n = _chain_tiles(10)
        solver = InterfaceCGSolver(
            n_interface=n, matvec_mode='tilewise',
            tile_schur_complements={k: v.copy() for k, v in tile_schur.items()},
            tile_index_maps=tile_idx,
            preconditioner='two_level', matvec_threads=1,
            warm_start_extrapolation=True,
        )
        try:
            x1 = np.ones(n)
            solver.push_solution_history(x1)
            np.testing.assert_allclose(solver._x0, x1)  # falls back to x_prev

            x2 = 2.0 * np.ones(n)
            solver.push_solution_history(x2)
            expected = 2.0 * x2 - x1  # now extrapolates
            np.testing.assert_allclose(solver._x0, expected)
        finally:
            solver.close()

    def test_non_converged_solve_not_pushed_into_extrapolation_history(self):
        """Finding 2 (round-1 code review) regression: a non-converged
        solve (info != 0, strict=False) must NOT be pushed into the
        extrapolation history -- pushing it would let the NEXT step's
        seed ``2*x_bad - x_prev2`` amplify the error, compounding step
        over step on a transient run that keeps hitting maxiter. The
        history must instead be CLEARED on a failed solve (so the next
        step seeds from the plain best iterate, at worst), and the
        per-step iteration count must stay pinned at ``maxiter`` -- not
        grow -- across repeated failed solves on a slowly-varying RHS
        (mirrors ``_run``'s drifting-RHS pattern above).

        Negative-test evidence: reverting Finding 2's fix (calling
        ``push_solution_history(result)`` unconditionally, regardless of
        ``info``) makes this FAIL -- ``_x_hist_prev``/``_x_hist_prev2``
        end up populated with non-converged iterates after a failed
        solve, and the per-step iters compound upward across steps
        instead of staying pinned at ``maxiter=2``.
        """
        tile_schur, tile_idx, n = _chain_tiles(30)
        rng = np.random.default_rng(555)
        b0 = rng.standard_normal(n)
        direction = rng.standard_normal(n) * 0.05
        solver = InterfaceCGSolver(
            n_interface=n, matvec_mode='tilewise',
            tile_schur_complements={k: v.copy() for k, v in tile_schur.items()},
            tile_index_maps=tile_idx,
            preconditioner='two_level', rtol=1e-14, atol=1e-16,
            matvec_threads=1, strict=False, maxiter=2,
            warm_start_extrapolation=True,
            # This test relies on maxiter=2 NOT being enough to converge
            # this fixture (test precondition below) -- the deflated apply
            # mode's coarse-only initial residual converges this PoU-only
            # chain fixture too well for that precondition to hold under
            # the 2026-07-20 DEFAULT_APPLY_MODE flip, so pin the original
            # 'additive' behavior explicitly (unrelated to what this test
            # actually exercises: extrapolation-history clearing on
            # failure).
            interface_coarse_apply_mode='additive',
        )
        try:
            iters = []
            for step in range(6):
                b = b0 + step * direction
                x = solver(b)
                assert solver.stats['cg_failed'] is True, (
                    "maxiter=2 must not be enough to converge this "
                    "fixture at rtol=1e-14 -- test precondition"
                )
                iters.append(solver.stats['last_cg_iters'])
                # History must be cleared on every failed solve -- the
                # next step's seed is the plain best iterate, never an
                # extrapolated (potentially amplified) one.
                assert solver._x_hist_prev is None, step
                assert solver._x_hist_prev2 is None, step
                np.testing.assert_allclose(solver._x0, x)
            # Iteration count must stay pinned at maxiter every step --
            # no compounding growth from an amplified warm start.
            assert iters == [2] * 6, iters
        finally:
            solver.close()

    def test_reset_warm_start_clears_extrapolation_history(self):
        tile_schur, tile_idx, n = _chain_tiles(10)
        solver = InterfaceCGSolver(
            n_interface=n, matvec_mode='tilewise',
            tile_schur_complements={k: v.copy() for k, v in tile_schur.items()},
            tile_index_maps=tile_idx,
            preconditioner='two_level', matvec_threads=1,
            warm_start_extrapolation=True,
        )
        try:
            solver.push_solution_history(np.ones(n))
            solver.push_solution_history(2.0 * np.ones(n))
            assert solver._x_hist_prev is not None
            assert solver._x_hist_prev2 is not None
            solver.reset_warm_start()
            assert solver._x0 is None
            assert solver._x_hist_prev is None
            assert solver._x_hist_prev2 is None
        finally:
            solver.close()

    def test_zero_rhs_solve_not_pushed_into_history(self):
        """Round-3 code review finding 3 (CONFIRMED): the ``bnrm2 == 0``
        early exit (both the scipy path and ``_deflated_pcg`` special-case
        an exactly-zero RHS to ``x=0, info=0`` WITHOUT running a single
        iteration) is not a genuine converged solution of a "family" the
        two-point extrapolation history should track. Pushing it in
        seeded the NEXT solve with ``2*0 - x_prev == -x_prev`` -- a seed
        reliably WORSE than a cold start.

        Fix: ``__call__`` now skips ``push_solution_history`` entirely
        when the RHS norm is exactly ``0.0``, leaving ``_x0``/
        ``_x_hist_prev``/``_x_hist_prev2`` exactly as they were before the
        zero-RHS call -- the next solve warm-starts from whatever state
        preceded the zero-RHS solve, unaffected by it.

        Sequence under test: ``solve(b)``, ``solve(0)``, ``solve(b')`` --
        the third solve's seed must be the plain ``x1`` (the first
        solve's result), never ``-x1``.

        Negative-test evidence: temporarily reverting the fix (dropping
        the ``_bnrm2 != 0.0`` guard so ``if info == 0:
        self.push_solution_history(result)`` runs unconditionally) makes
        this FAIL -- the third solve's seed becomes ``-x1`` instead of
        ``x1`` -- verified directly while implementing this fix (see this
        session's final report for the revert/restore transcript).
        """
        tile_schur, tile_idx, n = _chain_tiles(10)
        solver = InterfaceCGSolver(
            n_interface=n, matvec_mode='tilewise',
            tile_schur_complements={k: v.copy() for k, v in tile_schur.items()},
            tile_index_maps=tile_idx,
            preconditioner='two_level', rtol=1e-10, atol=1e-16,
            matvec_threads=1, strict=True, maxiter=5000,
            warm_start_extrapolation=True,
        )
        try:
            rng = np.random.default_rng(77)
            b = rng.standard_normal(n)
            x1 = solver(b)
            assert solver._x_hist_prev is not None
            assert solver._x_hist_prev2 is None  # only one solve so far
            np.testing.assert_allclose(solver._x0, x1)

            x_zero = solver(np.zeros(n))
            np.testing.assert_array_equal(x_zero, np.zeros(n))

            # History/seed must be COMPLETELY UNAFFECTED by the zero-RHS
            # solve -- unchanged from right after the first solve.
            assert solver._x_hist_prev2 is None
            np.testing.assert_allclose(solver._x_hist_prev, x1)
            np.testing.assert_allclose(solver._x0, x1)

            x0_before_third = solver._x0.copy()
            b2 = b + rng.standard_normal(n) * 0.01
            solver(b2)

            # The seed used for the third solve must be the PLAIN previous
            # solution (x1) -- never "extrapolated across the zero-RHS
            # solve" into -2*0 - x1 == -x1.
            np.testing.assert_allclose(x0_before_third, x1, atol=1e-8)
            bad_seed = -x1
            assert not np.allclose(x0_before_third, bad_seed), (
                "the zero-RHS solve corrupted the next solve's warm-start "
                "seed into -x_prev"
            )
        finally:
            solver.close()


class TestByteGuardADef2:
    """Byte-guard accounting: retained-SZ counted (findings 3+4, round-1
    code review, fixed the accounting -- see build_coarse_space's
    ``retain_sz``/``max_bytes`` docstrings). SZ is the SAME array the
    build's own ~3*n*T'*8 transient peak already forms (regardless of
    retain_sz), so retaining it is FREE relative to that peak -- there is
    no separate "4x deflated total" to guard against. The max_bytes
    degradation ladder (GenEO -> PoU-only -> disable) now runs FIRST to
    settle the FINAL T', and the SZ-retention rung is evaluated on that
    final T' (3x, not 4x) -- so whenever the ladder leaves the coarse
    space alive, SZ retention survives too; the only way retain_sz=True
    ends up with ``coarse.SZ is None`` is if the coarse space itself is
    disabled outright (in which case ``coarse`` is ``None`` and the
    question is moot)."""

    def test_budget_between_3x_and_4x_keeps_sz_retained(self, caplog):
        """Finding 4 (round-1 code review) regression: a budget that fits
        the plain 3x build peak but not the pre-fix (buggy) 4x formula
        must NOT drop SZ retention -- the pre-fix code incorrectly treated
        this budget as insufficient (double-counting SZ); the corrected
        3x-only accounting shows it was sufficient all along."""
        tile_schur, tile_idx, n = _chain_tiles(20)
        S = np.zeros((n, n))
        for tid, idx in tile_idx.items():
            S[np.ix_(idx, idx)] += tile_schur[tid]
        T_prime = 20  # n_pou_cols, no geneo requested
        budget_3x = 3 * n * T_prime * 8
        budget_4x = 4 * n * T_prime * 8
        budget = int((budget_3x + budget_4x) / 2)  # fits 3x, not the old 4x
        assert budget > budget_3x

        with caplog.at_level(logging.WARNING):
            coarse = ic.build_coarse_space(
                lambda X: S @ X, tile_idx, n=n, max_bytes=budget,
                retain_sz=True,
            )
        assert coarse is not None
        assert coarse.SZ is not None, (
            "SZ retention must SURVIVE -- budget fits the TRUE 3x cost "
            "(the pre-fix 4x-formula bug would have dropped it here)"
        )
        assert coarse.n_cols == T_prime, "no GenEO to drop; T' unaffected"
        assert not any(
            'dropping sz retention' in r.message.lower() for r in caplog.records
        ), [r.message for r in caplog.records]

    def test_budget_between_additive_and_old_adef2_full_keeps_everything(self, caplog):
        """Finding 4 (round-1 code review) regression, GenEO variant: a
        budget between the plain-3x-with-GenEO total and the pre-fix
        (buggy) 4x total must keep BOTH the GenEO columns AND SZ retention
        -- under the corrected 3x-only accounting, this budget was never
        actually insufficient."""
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
        n_pou_cols = 20
        T_prime = n_pou_cols + 20 * 2  # 60

        budget_old_adef2_full = 4 * n * T_prime * 8
        budget_additive_full = 3 * n * T_prime * 8
        budget = int((budget_additive_full + budget_old_adef2_full) / 2)
        assert budget < budget_old_adef2_full and budget > budget_additive_full

        with caplog.at_level(logging.WARNING):
            coarse = ic.build_coarse_space(
                lambda X: S @ X, tile_idx, n=n, geneo_pairs=geneo_pairs,
                max_bytes=budget, retain_sz=True,
            )
        assert coarse is not None
        assert coarse.n_geneo_cols == 40, (
            f"GenEO columns must survive; got n_geneo_cols={coarse.n_geneo_cols}"
        )
        assert coarse.SZ is not None, (
            "SZ retention must also survive -- budget exceeds the TRUE 3x "
            "cost at this T' (the pre-fix 4x-formula bug would have "
            "dropped it here even though GenEO itself was never at risk)"
        )
        assert not any(
            'dropping sz retention' in r.message.lower() for r in caplog.records
        ), [r.message for r in caplog.records]

    def test_reviewer_repro_stale_t_prime_sz_survives_after_geneo_drop(self):
        """Finding 3 (round-1 code review) regression -- reviewer's exact
        repro: n=100, T'_full=28 (T'_pou=4 + 24 GenEO columns),
        max_bytes=50000.

        The pre-fix code evaluated SZ retention against the STALE
        pre-degradation T'=28 (4*100*28*8=89600 > 50000 -> dropped SZ)
        even though the max_bytes ladder was about to drop all 24 GenEO
        columns anyway, landing on T'=4 where retaining SZ easily fits
        (3*100*4*8=12800 <= 50000). Post-fix: the ladder runs FIRST
        (settles on T'=4), THEN the SZ rung is evaluated on that FINAL
        T' -- SZ must survive.
        """
        n = 100
        n_tiles = 4
        block = n // n_tiles  # 25
        tile_idx = {
            t: np.arange(t * block, (t + 1) * block, dtype=np.int32)
            for t in range(n_tiles)
        }
        rng = np.random.default_rng(2)
        geneo_pairs = []
        for t, idx in tile_idx.items():
            V = rng.standard_normal((len(idx), 6))
            w = np.full(6, 1e-8)
            geneo_pairs.append((idx, V, w))
        # Disjoint tiles covering all n nodes -> n_pou_cols = 4 (one column
        # per tile); T'_full = 4 + 4*6 = 28; T'_pou = 4.
        S = np.eye(n) * 3.0  # SPD stand-in; only the byte-guard/degradation
        # bookkeeping is exercised here, not solve accuracy.
        max_bytes = 50000
        assert 3 * n * 28 * 8 > max_bytes  # T'_full doesn't fit
        assert 3 * n * 4 * 8 <= max_bytes  # T'_pou (with SZ) does

        coarse = ic.build_coarse_space(
            lambda X: S @ X, tile_idx, n=n, geneo_pairs=geneo_pairs,
            max_bytes=max_bytes, retain_sz=True,
        )
        assert coarse is not None
        assert coarse.n_pou_cols == 4
        assert coarse.n_geneo_cols == 0, (
            "budget forces the GenEO drop (3*100*28*8=67200 > 50000); "
            f"got n_geneo_cols={coarse.n_geneo_cols}"
        )
        assert coarse.n_cols == 4
        assert coarse.SZ is not None, (
            "SZ retention must SURVIVE at the final (post-GenEO-drop) "
            "T'=4 -- the pre-fix code incorrectly evaluated this against "
            "the stale T'=28 and dropped it"
        )

    def test_extreme_budget_disables_coarse_space_entirely(self, caplog):
        """Post-fix (findings 3+4), SZ retention and the coarse space's
        existence are no longer independently droppable for a byte-budget
        reason: since retaining SZ costs nothing beyond the shared 3x
        build peak, a budget too small for that peak disables the coarse
        space OUTRIGHT (no PoU-only, no SZ, nothing) rather than degrading
        to "PoU-only, SZ dropped" -- there is no smaller rung left once
        even the bare PoU-only 3x cost does not fit."""
        tile_schur, tile_idx, n = _chain_tiles(20)
        S = np.zeros((n, n))
        for tid, idx in tile_idx.items():
            S[np.ix_(idx, idx)] += tile_schur[tid]
        n_pou_cols = 20
        budget = int(3 * n * n_pou_cols * 8 * 0.5)  # under even bare PoU-only

        with caplog.at_level(logging.WARNING):
            coarse = ic.build_coarse_space(
                lambda X: S @ X, tile_idx, n=n, max_bytes=budget,
                retain_sz=True,
            )
        assert coarse is None
        assert any(
            'coarse space disabled' in r.message.lower() for r in caplog.records
        ), [r.message for r in caplog.records]

    def test_generous_budget_keeps_sz_retained(self):
        tile_schur, tile_idx, n = _chain_tiles(10)
        S = np.zeros((n, n))
        for tid, idx in tile_idx.items():
            S[np.ix_(idx, idx)] += tile_schur[tid]
        coarse = ic.build_coarse_space(
            lambda X: S @ X, tile_idx, n=n, retain_sz=True,
        )
        assert coarse is not None
        assert coarse.SZ is not None

    def test_interface_cg_solver_keeps_deflated_label_between_3x_and_4x(self):
        """End-to-end through InterfaceCGSolver: findings 3+4 fix -- a
        budget between the TRUE 3x cost and the pre-fix (buggy) 4x
        threshold must keep the deflated apply active (SZ retained), not
        silently downgrade to additive."""
        tile_schur, tile_idx, n = _chain_tiles(20)
        T_prime = 20
        budget = int(3.5 * n * T_prime * 8)  # between 3x and the old 4x
        solver = InterfaceCGSolver(
            n_interface=n, matvec_mode='tilewise',
            tile_schur_complements={k: v.copy() for k, v in tile_schur.items()},
            tile_index_maps=tile_idx,
            preconditioner='two_level', matvec_threads=1,
            interface_coarse_apply_mode='deflated',
            interface_coarse_max_bytes=budget,
        )
        try:
            assert solver._coarse is not None
            assert solver._coarse.SZ is not None, (
                "the pre-fix 4x-formula bug would have dropped SZ here; "
                "the corrected 3x accounting shows this budget suffices"
            )
            assert 'deflated' in solver.preconditioner_label, (
                f"label must reflect the deflated apply actually being "
                f"active: {solver.preconditioner_label!r}"
            )
            rng = np.random.default_rng(9)
            b = rng.standard_normal(n)
            x = solver(b)
            S = np.zeros((n, n))
            for tid, idx in tile_idx.items():
                S[np.ix_(idx, idx)] += tile_schur[tid]
            x_direct = np.linalg.solve(S, b)
            np.testing.assert_allclose(x, x_direct, rtol=1e-6, atol=1e-8)
        finally:
            solver.close()


class TestADef2Lifecycle:
    """Lifecycle: release drops SZ; refactor rebuilds it; save/load
    unaffected (coarse state, like the rest of Stage 3, is never
    persisted)."""

    @staticmethod
    def _build_model():
        return _build_two_tile_distributed_model(package_cap_edges=[])

    def test_release_drops_sz_refactor_rebuilds(self, tmp_path):
        """Mirrors TestRefactorRebuildsTilewise.
        test_refactor_after_release_rebuilds_tilewise's save->release->
        load->refactor pattern (workers stay attached to the same live
        model, so refactor() re-gathers tilewise rather than downgrading
        to 'assembled')."""
        from distributed.solver import DistributedDDMSolver
        from distributed.result import DistributedSolverContext

        model = self._build_model()
        model.settings.update({
            'interface_solver': 'cg',
            'interface_matvec_mode': 'tilewise',
            'interface_preconditioner': 'two_level',
            'interface_coarse_apply_mode': 'deflated',
        })
        solver = DistributedDDMSolver(model)
        ctx = solver.prepare()
        try:
            cg_solver_1 = ctx._cg_solver
            assert cg_solver_1 is not None
            assert cg_solver_1._coarse is not None
            # Two-tile fixture: T'=2 (one PoU column per tile), n=1 --
            # 3*n*T'*8 is tiny, well under the default 8 GB budget, so SZ
            # retention should have succeeded. (3x, not 4x -- SZ is the
            # SAME array the build's own transient peak already forms;
            # round-1 findings 3+4 corrected the accounting from a
            # double-counted 4x -- see build_coarse_space's `retain_sz`
            # docstring and TestByteGuardADef2's class docstring.)
            assert cg_solver_1._coarse.SZ is not None

            path = ctx.save(str(tmp_path / 'dc_ctx.pkl'))
            ctx.release()
            assert ctx._cg_solver is None, (
                "release() must drop the CG solver (and with it the "
                "coarse space + its retained SZ)"
            )

            ctx = DistributedSolverContext.load(model, path)
            ctx.refactor()
            cg_solver_2 = ctx._cg_solver
            assert cg_solver_2 is not None
            assert cg_solver_2 is not cg_solver_1
            assert cg_solver_2._coarse is not None
            assert cg_solver_2._coarse.SZ is not None, (
                "refactor() must rebuild the coarse space WITH SZ retained "
                "again (apply_mode='deflated' is still in model.settings)"
            )

            result = solver.solve_dc(ctx)
            assert result.flatten()  # sanity: solve actually runs
        finally:
            ctx.release()
            model.shutdown()

    def test_save_load_unaffected_by_adef2(self, tmp_path):
        """save()/load() persist S_global (as always) -- adef2's SZ is
        never part of that payload; after load()+refactor(), a fresh
        coarse space (with SZ retained again) is built exactly like a
        from-scratch factor(), and the reloaded context still solves.

        Note: this test deliberately does NOT compare solve_dc's full
        per-node voltage dict before vs. after the release()/load()/
        refactor() cycle -- verified (with plain 'direct'/'assembled'
        settings, no two_level/adef2 involved at all) that interior-node
        recovery genuinely differs across that cycle on this fixture,
        a PRE-EXISTING lifecycle characteristic unrelated to this work
        package (out of scope -- see spec's "Out of scope: ... TD/DC
        factor paths"). Matches the precedent in test_interface_iterative_
        stage2.py's test_refactor_after_release_rebuilds_tilewise, which
        only asserts the reloaded solve *succeeds* (``assert result.
        flatten()``), not that per-node values are unchanged.
        """
        from distributed.solver import DistributedDDMSolver
        from distributed.result import DistributedSolverContext

        model = self._build_model()
        model.settings.update({
            'interface_solver': 'cg',
            'interface_matvec_mode': 'assembled',
            'interface_preconditioner': 'two_level',
            'interface_coarse_apply_mode': 'deflated',
        })
        solver = DistributedDDMSolver(model)
        ctx = solver.prepare()
        save_path = str(tmp_path / 'dc_ctx.pkl')
        try:
            result_before = solver.solve_dc(ctx)
            assert result_before.flatten()
            ctx.save(save_path)
        finally:
            ctx.release()

        loaded = DistributedSolverContext.load(model, save_path)
        loaded.refactor()
        try:
            assert loaded._cg_solver is not None
            assert loaded._cg_solver._coarse is not None
            assert loaded._cg_solver._coarse.SZ is not None, (
                "refactor() must rebuild the coarse space WITH SZ retained "
                "-- adef2's SZ is never part of the save() payload, so "
                "this confirms it was genuinely rebuilt, not resurrected "
                "from the pickle"
            )
            result_after = solver.solve_dc(loaded)
            assert result_after.flatten()  # sanity: reloaded solve succeeds
        finally:
            loaded.release()
            model.shutdown()

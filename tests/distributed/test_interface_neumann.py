"""NN/BDD work package: weighted Neumann-Neumann fine-space base.

Covers ``InterfaceCGSolver._build_neumann`` / ``_nn_apply_*`` (Candidate 1
of ``docs/interface_precond_sota_research.md``): the ``M^-1 = sum_i R_i^T
D_i S~_i^+ D_i R_i`` base, standalone (``preconditioner='neumann'``) and as
the ``'two_level'`` base (``two_level_base='neumann'``), including SPD-ness,
partition-of-unity weighting, the byte-budget/assembled-mode/all-zero-block
degrade ladder, island exclusion, eigclip fallback on singular blocks,
tile-count iteration scaling vs the jacobi base, and the two-tile
distributed-model forced-CG equivalence row.
"""

import logging

import numpy as np
import pytest
import scipy.sparse as sp

from distributed.interface_iterative import (
    InterfaceCGSolver,
    NEUMANN_MAX_FACTOR_BYTES,
    resolve_neumann_max_bytes,
    resolve_preconditioner,
)

from .test_interface_coarse import _ill_conditioned_jacobi_fixture
from .test_time_domain import _build_two_tile_distributed_model

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _two_tile_overlap_fixture(n=12, k1=8, k2=7, seed=0):
    """Two overlapping dense SPD tile blocks: S = R1^T S1 R1 + R2^T S2 R2.

    Nodes [k1-overlap, k1) are shared by both tiles (overlap = k1+k2-n),
    the textbook NN setting.  Returns (tile_schur, tile_idx, S, n).
    """
    rng = np.random.default_rng(seed)
    A1 = rng.standard_normal((k1, k1))
    S1 = A1 @ A1.T + k1 * np.eye(k1)
    A2 = rng.standard_normal((k2, k2))
    S2 = A2 @ A2.T + k2 * np.eye(k2)
    i1 = np.arange(0, k1, dtype=np.int64)
    i2 = np.arange(n - k2, n, dtype=np.int64)
    S = np.zeros((n, n))
    S[np.ix_(i1, i1)] += S1
    S[np.ix_(i2, i2)] += S2
    return {'t1': S1, 't2': S2}, {'t1': i1, 't2': i2}, S, n


def _chain_fixture(n_tiles, tile_n=20, g_chain=1.0, g_ground=1e-3, seed=3):
    """A 1-D chain of overlapping resistor-ladder tiles (PDN-like).

    Each tile is a (tile_n x tile_n) tridiagonal conductance ladder with a
    WEAK ground leak (g_ground << g_chain), overlapping its neighbor by one
    node -- condition number grows with the chain length, the kappa ~ 1/H^2
    regime a one-level base degrades in.  Returns
    (tile_schur, tile_idx, S, n).
    """
    rng = np.random.default_rng(seed)
    tile_schur, tile_idx = {}, {}
    n = n_tiles * (tile_n - 1) + 1
    S = np.zeros((n, n))
    for t in range(n_tiles):
        start = t * (tile_n - 1)
        idx = np.arange(start, start + tile_n, dtype=np.int64)
        B = np.zeros((tile_n, tile_n))
        for j in range(tile_n - 1):
            g = g_chain * (1.0 + 0.1 * rng.random())
            B[j, j] += g
            B[j + 1, j + 1] += g
            B[j, j + 1] -= g
            B[j + 1, j] -= g
        B += g_ground * np.eye(tile_n)
        tile_schur[t] = B
        tile_idx[t] = idx
        S[np.ix_(idx, idx)] += B
    return tile_schur, tile_idx, S, n


def _solve_and_check(cg, S, b, tol):
    x = cg(b)
    err = np.abs(x - np.linalg.solve(S, b)).max()
    assert err <= tol, f"max err {err:.3e} > {tol:.1e}"
    return x


# ---------------------------------------------------------------------------
# 1. Standalone 'neumann' base
# ---------------------------------------------------------------------------


class TestNeumannStandalone:

    def test_matches_direct(self):
        tile_schur, tile_idx, S, n = _two_tile_overlap_fixture()
        rng = np.random.default_rng(1)
        b = rng.standard_normal(n)
        cg = InterfaceCGSolver(
            n_interface=n, matvec_mode='tilewise',
            tile_schur_complements=tile_schur, tile_index_maps=tile_idx,
            preconditioner='neumann', rtol=1e-12, matvec_threads=2,
        )
        _solve_and_check(cg, S, b, 1e-9)
        assert cg.preconditioner_label == 'neumann'

    def test_apply_is_symmetric_spd(self):
        tile_schur, tile_idx, S, n = _two_tile_overlap_fixture()
        cg = InterfaceCGSolver(
            n_interface=n, matvec_mode='tilewise',
            tile_schur_complements=tile_schur, tile_index_maps=tile_idx,
            preconditioner='neumann', rtol=1e-10,
        )
        M = np.column_stack([cg._M.matvec(e) for e in np.eye(n)])
        scale = np.abs(M).max()
        assert np.abs(M - M.T).max() <= 1e-12 * scale
        w = np.linalg.eigvalsh(0.5 * (M + M.T))
        assert w.min() > 0.0

    def test_serial_and_threaded_apply_agree(self):
        tile_schur, tile_idx, S, n = _two_tile_overlap_fixture()
        cg = InterfaceCGSolver(
            n_interface=n, matvec_mode='tilewise',
            tile_schur_complements=tile_schur, tile_index_maps=tile_idx,
            preconditioner='neumann', rtol=1e-10, matvec_threads=4,
        )
        rng = np.random.default_rng(2)
        x = rng.standard_normal(n)
        np.testing.assert_allclose(
            cg._nn_apply_threaded(x), cg._nn_apply_serial(x),
            rtol=0, atol=1e-13,
        )

    def test_stiffness_equals_multiplicity_on_identical_blocks(self):
        """When every tile stamps the SAME diagonal at a shared node, the
        stiffness weights reduce to 1/multiplicity exactly."""
        B = np.array([[2.0, -0.5], [-0.5, 2.0]])
        tile_schur = {0: B.copy(), 1: B.copy()}
        tile_idx = {0: np.array([0, 1]), 1: np.array([1, 2])}
        n = 3
        rng = np.random.default_rng(4)
        x = rng.standard_normal(n)
        applies = []
        for weight in ('stiffness', 'multiplicity'):
            cg = InterfaceCGSolver(
                n_interface=n, matvec_mode='tilewise',
                tile_schur_complements={k: v.copy() for k, v in tile_schur.items()},
                tile_index_maps=tile_idx,
                preconditioner='neumann', neumann_weight=weight, rtol=1e-10,
            )
            applies.append(cg._M.matvec(x))
        np.testing.assert_allclose(applies[0], applies[1], rtol=0, atol=1e-14)

    def test_uncovered_node_gets_diagonal_complement(self):
        """A node present only in S_extra (tap/package-style) must get the
        exact-jacobi response, keeping M SPD on all of R^n."""
        tile_schur, tile_idx, S, n_cov = _two_tile_overlap_fixture()
        n = n_cov + 1
        g_tap = 4.0
        S_extra = sp.csr_matrix(
            ([g_tap], ([n_cov], [n_cov])), shape=(n, n),
        )
        cg = InterfaceCGSolver(
            n_interface=n, matvec_mode='tilewise',
            tile_schur_complements=tile_schur, tile_index_maps=tile_idx,
            S_extra=S_extra, preconditioner='neumann', rtol=1e-12,
        )
        e = np.zeros(n)
        e[n_cov] = 1.0
        y = cg._M.matvec(e)
        assert y[n_cov] == pytest.approx(1.0 / g_tap)
        S_full = np.zeros((n, n))
        S_full[:n_cov, :n_cov] = S
        S_full[n_cov, n_cov] = g_tap
        b = np.random.default_rng(5).standard_normal(n)
        _solve_and_check(cg, S_full, b, 1e-9)

    def test_singular_block_falls_back_to_eigclip(self, caplog):
        """A floating tile (pure Laplacian, no ground leak) is PSD-singular;
        the builder must survive via the eigclip pseudo-inverse and the
        solve must still converge (the neighbor tile grounds the system)."""
        g = 1.0
        B_sing = np.array([[g, -g], [-g, g]])  # exact kernel: constants
        A = np.random.default_rng(6).standard_normal((2, 2))
        B_spd = A @ A.T + 2 * np.eye(2)
        tile_schur = {0: B_sing, 1: B_spd}
        tile_idx = {0: np.array([0, 1]), 1: np.array([1, 2])}
        n = 3
        S = np.zeros((n, n))
        for t, idx in tile_idx.items():
            S[np.ix_(idx, idx)] += tile_schur[t]
        with caplog.at_level(logging.INFO, logger='distributed.interface_iterative'):
            cg = InterfaceCGSolver(
                n_interface=n, matvec_mode='tilewise',
                tile_schur_complements=tile_schur, tile_index_maps=tile_idx,
                preconditioner='neumann', rtol=1e-12,
            )
        assert any('eigclip' in r.message for r in caplog.records)
        assert cg.preconditioner_label == 'neumann'
        b = np.random.default_rng(7).standard_normal(n)
        _solve_and_check(cg, S, b, 1e-8)

    def test_tikhonov_reg_uses_cholesky_on_singular_block(self, caplog):
        """neumann_reg > 0 must lift a singular block onto the fast
        Cholesky path (no eigclip fallback) while converging to the
        UNREGULARIZED system's solution (reg perturbs M only)."""
        g = 1.0
        B_sing = np.array([[g, -g], [-g, g]])  # exact kernel: constants
        A = np.random.default_rng(14).standard_normal((2, 2))
        B_spd = A @ A.T + 2 * np.eye(2)
        tile_schur = {0: B_sing, 1: B_spd}
        tile_idx = {0: np.array([0, 1]), 1: np.array([1, 2])}
        n = 3
        S = np.zeros((n, n))
        for t, idx in tile_idx.items():
            S[np.ix_(idx, idx)] += tile_schur[t]
        with caplog.at_level(logging.INFO,
                             logger='distributed.interface_iterative'):
            cg = InterfaceCGSolver(
                n_interface=n, matvec_mode='tilewise',
                tile_schur_complements=tile_schur, tile_index_maps=tile_idx,
                preconditioner='neumann', neumann_reg=1e-6, rtol=1e-12,
            )
        rec = [r.getMessage() for r in caplog.records
               if 'Neumann base:' in r.getMessage()]
        assert rec and '0 via eigclip' in rec[-1], rec
        assert 'reg=1e-06' in rec[-1], rec
        b = np.random.default_rng(15).standard_normal(n)
        _solve_and_check(cg, S, b, 1e-8)

    def test_island_nodes_sliced_out_and_served_by_complement(self):
        """Island nodes (S_extra 1e5 penalty) must be excluded from every
        NN block and served by the diagonal complement instead."""
        tile_schur, tile_idx, S, n = _two_tile_overlap_fixture()
        island_g = 1e5
        island = np.array([6], dtype=np.int64)  # inside the tile overlap
        S_extra = sp.csr_matrix(
            ([island_g], ([6], [6])), shape=(n, n),
        )
        cg = InterfaceCGSolver(
            n_interface=n, matvec_mode='tilewise',
            tile_schur_complements=tile_schur, tile_index_maps=tile_idx,
            S_extra=S_extra, island_idx=island,
            preconditioner='neumann', rtol=1e-12,
        )
        for idx, _B in cg._nn_tiles:
            assert 6 not in idx
        assert cg._nn_comp_scale[6] > 0.0
        # complement uses the FULL diagonal (tile sum + penalty)
        expected = 1.0 / (S[6, 6] + island_g)
        assert cg._nn_comp_scale[6] == pytest.approx(expected)
        S_full = S.copy()
        S_full[6, 6] += island_g
        b = np.random.default_rng(8).standard_normal(n)
        _solve_and_check(cg, S_full, b, 1e-8)


# ---------------------------------------------------------------------------
# 2. Degrade ladder
# ---------------------------------------------------------------------------


class TestNeumannDegradeLadder:

    def test_byte_budget_degrades_to_jacobi(self, caplog):
        tile_schur, tile_idx, S, n = _two_tile_overlap_fixture()
        with caplog.at_level(logging.WARNING):
            cg = InterfaceCGSolver(
                n_interface=n, matvec_mode='tilewise',
                tile_schur_complements=tile_schur, tile_index_maps=tile_idx,
                preconditioner='neumann', neumann_max_bytes=1, rtol=1e-12,
            )
        assert cg.preconditioner == 'jacobi'
        assert cg.requested_preconditioner == 'neumann'
        assert any('interface_neumann_max_bytes' in r.message
                   for r in caplog.records)
        b = np.random.default_rng(9).standard_normal(n)
        _solve_and_check(cg, S, b, 1e-9)

    def test_two_level_neumann_budget_downgrade_keeps_coarse(self):
        """A budget downgrade under 'two_level' must still get the coarse
        space (jacobi+PoU), mirroring the BJ downgrade composition."""
        tile_schur, tile_idx, S, n = _two_tile_overlap_fixture()
        cg = InterfaceCGSolver(
            n_interface=n, matvec_mode='tilewise',
            tile_schur_complements=tile_schur, tile_index_maps=tile_idx,
            preconditioner='two_level', two_level_base='neumann',
            neumann_max_bytes=1, rtol=1e-12,
        )
        assert cg._coarse is not None
        assert 'jacobi' in cg.preconditioner_label
        b = np.random.default_rng(10).standard_normal(n)
        _solve_and_check(cg, S, b, 1e-9)

    def test_all_zero_blocks_degrade_and_still_converge(self):
        """The realistic-T'/n-ratio ill-conditioned fixture has ALL-ZERO
        tile Schur placeholders (everything routed through S_extra) -- the
        NN builder must skip every block, degrade to jacobi, and the
        two_level[deflated] solve must still converge (the fixture that
        maxiter-killed true A-DEF2)."""
        tile_schur, tile_idx, S_extra, S, n = _ill_conditioned_jacobi_fixture()
        cg = InterfaceCGSolver(
            n_interface=n, matvec_mode='tilewise',
            tile_schur_complements=tile_schur, tile_index_maps=tile_idx,
            S_extra=S_extra,
            preconditioner='two_level', two_level_base='neumann',
            interface_coarse_apply_mode='deflated',
            rtol=1e-8, maxiter=5000,
        )
        assert 'jacobi' in cg.preconditioner_label
        b = np.random.default_rng(11).standard_normal(n)
        _solve_and_check(cg, S, b, 1e-4)

    def test_assembled_mode_degrades_to_jacobi(self):
        tile_schur, tile_idx, S, n = _two_tile_overlap_fixture()
        cg = InterfaceCGSolver(
            n_interface=n, matvec_mode='assembled',
            S_global=sp.csr_matrix(S),
            preconditioner='neumann', rtol=1e-12,
        )
        assert cg.preconditioner == 'jacobi'
        b = np.random.default_rng(12).standard_normal(n)
        _solve_and_check(cg, S, b, 1e-9)

    def test_invalid_knob_values_raise(self):
        tile_schur, tile_idx, S, n = _two_tile_overlap_fixture()
        kw = dict(
            n_interface=n, matvec_mode='tilewise',
            tile_schur_complements=tile_schur, tile_index_maps=tile_idx,
        )
        with pytest.raises(ValueError, match='two_level_base'):
            InterfaceCGSolver(preconditioner='two_level',
                              two_level_base='bogus', **kw)
        with pytest.raises(ValueError, match='neumann_weight'):
            InterfaceCGSolver(preconditioner='neumann',
                              neumann_weight='bogus', **kw)

    def test_resolvers(self):
        assert resolve_neumann_max_bytes(123) == 123
        assert resolve_neumann_max_bytes('auto') <= NEUMANN_MAX_FACTOR_BYTES
        assert resolve_preconditioner('neumann', 'cg', 'tilewise') == 'neumann'
        # 'auto' still resolves to two_level (base selection is a separate knob)
        assert resolve_preconditioner('auto', 'cg', 'tilewise') == 'two_level'


# ---------------------------------------------------------------------------
# 3. Two-level composition + tile-count scaling
# ---------------------------------------------------------------------------


class TestNeumannTwoLevel:

    @pytest.mark.parametrize('apply_mode', ['deflated', 'additive'])
    def test_two_level_neumann_matches_direct(self, apply_mode):
        tile_schur, tile_idx, S, n = _chain_fixture(n_tiles=6)
        cg = InterfaceCGSolver(
            n_interface=n, matvec_mode='tilewise',
            tile_schur_complements=tile_schur, tile_index_maps=tile_idx,
            preconditioner='two_level', two_level_base='neumann',
            interface_coarse_apply_mode=apply_mode,
            rtol=1e-12, matvec_threads=2,
        )
        assert 'neumann' in cg.preconditioner_label
        b = np.random.default_rng(13).standard_normal(n)
        _solve_and_check(cg, S, b, 1e-7)

    def test_iterations_flat_in_tile_count_and_beats_jacobi(self):
        """The BDD selling point: NN+PoU iteration counts stay ~flat as the
        chain lengthens, and beat jacobi+PoU at every size (the chain's
        weak-ground conditioning worsens with length)."""
        iters = {'neumann': [], 'jacobi-base': []}
        for n_tiles in (5, 15, 30):
            tile_schur, tile_idx, S, n = _chain_fixture(n_tiles=n_tiles)
            b = np.ones(n)
            for label, base in (('neumann', 'neumann'), ('jacobi-base', 'jacobi')):
                cg = InterfaceCGSolver(
                    n_interface=n, matvec_mode='tilewise',
                    tile_schur_complements={k: v.copy() for k, v in tile_schur.items()},
                    tile_index_maps=tile_idx,
                    preconditioner='two_level', two_level_base=base,
                    rtol=1e-10, maxiter=2000,
                )
                cg(b)
                iters[label].append(cg.stats['last_cg_iters'])
        nn = iters['neumann']
        jc = iters['jacobi-base']
        # measured at pinning time: nn = [5, 10, 14] vs jc = [85, 103, 114]
        # -- a 6-17x iteration cut, growing only log-like in chain length.
        assert all(a * 4 <= b for a, b in zip(nn, jc)), (nn, jc)
        # near-flat-ness: growing the chain 6x must stay under 3x iterations
        # (observed 2.8x; the jacobi base needs 85+ iterations at EVERY size)
        assert nn[-1] <= 3 * nn[0], nn


# ---------------------------------------------------------------------------
# 4. Distributed two-tile model forced-CG equivalence row
# ---------------------------------------------------------------------------


class TestNeumannForcedCGEquivalence:

    def _run(self, settings):
        from distributed.solver import DistributedDDMSolver

        model = _build_two_tile_distributed_model(package_cap_edges=[])
        model.settings.update(settings)
        solver = DistributedDDMSolver(model)
        ctx = solver.prepare()
        try:
            return solver.solve_dc(ctx).flatten()
        finally:
            ctx.release()
            model.shutdown()

    def test_two_level_neumann_matches_direct_dc(self):
        v_direct = self._run({
            'interface_solver': 'direct', 'interface_matvec_mode': 'assembled',
        })
        v_nn = self._run({
            'interface_solver': 'cg',
            'interface_matvec_mode': 'tilewise',
            'interface_preconditioner': 'two_level',
            'interface_two_level_base': 'neumann',
            'interface_cg_rtol': 1e-12,
            'interface_cg_atol': 1e-16,
        })
        common = set(v_direct) & set(v_nn)
        assert common
        for node in common:
            diff = abs(v_direct[node] - v_nn[node])
            assert diff <= 1e-8, (
                f"node {node}: direct={v_direct[node]!r} vs "
                f"neumann={v_nn[node]!r} diff={diff:.3e}"
            )

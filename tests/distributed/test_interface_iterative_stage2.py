"""Stage 2 tests (interface_solve_acceleration_plan.md): threaded tilewise
matvec, threaded block-Jacobi apply, SPD-safe fallback, fp32 critical path,
matvec_mode='auto', pool lifecycle, refactor-rebuilds-tilewise,
interface_drop_s_global, zero-port regression, D1 solve_dc-level regression,
settings propagation.

Threaded-vs-serial matvec correctness (two-tile + randomized ~30-tile
synthetic) and thread-count invariance (1/2/8) are the explicit validation
gates from the plan; most other classes exercise the individual items
(1/1-manifestation-2/2/3/4/5/6/7/8/9/10/11) directly against
``InterfaceCGSolver`` (fast, no Ray/model machinery needed) or against
``pgmath``/``result_factorization`` helpers.
"""
from __future__ import annotations

import sys
import threading
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.unit

sys.path.insert(0, str(Path(__file__).parent))
from test_time_domain import _build_two_tile_distributed_model  # noqa: E402


def _random_spd(n, seed):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n))
    return A @ A.T + n * np.eye(n)


def _make_synthetic_tiles(n_interface, n_tiles, ports_lo, ports_hi, seed=0):
    """Synthetic tilewise interface system: random SPD blocks with
    overlapping global index sets (mimicking shared boundary nodes across
    neighboring tiles in a spatial partition)."""
    rng = np.random.default_rng(seed)
    tile_schur = {}
    tile_idx = {}
    for t in range(n_tiles):
        n_p = int(rng.integers(ports_lo, ports_hi + 1))
        center = int(rng.integers(0, n_interface))
        spread = max(1, int(n_interface / n_tiles * 2.5))
        idx = (center + rng.integers(-spread, spread, size=n_p)) % n_interface
        idx = np.unique(idx).astype(np.int32)
        tile_schur[t] = _random_spd(len(idx), seed=1000 + t)
        tile_idx[t] = idx
    return tile_schur, tile_idx


# ---------------------------------------------------------------------------
# Threaded tilewise matvec: correctness + thread-count invariance (gate 4)
# ---------------------------------------------------------------------------


class TestThreadedMatvecEquivalence:

    @pytest.mark.parametrize('n_threads', [1, 2, 8])
    def test_threaded_matches_serial_two_tile(self, n_threads):
        from distributed.interface_iterative import InterfaceCGSolver

        n = 6
        idx1 = np.array([0, 1, 2], dtype=np.int32)
        idx2 = np.array([2, 3, 4, 5], dtype=np.int32)
        tile_schur = {'A': _random_spd(3, 1), 'B': _random_spd(4, 2)}
        tile_idx = {'A': idx1, 'B': idx2}

        solver = InterfaceCGSolver(
            n_interface=n, matvec_mode='tilewise',
            tile_schur_complements=tile_schur, tile_index_maps=tile_idx,
            preconditioner='none', matvec_threads=n_threads,
        )
        try:
            rng = np.random.default_rng(42)
            x = rng.standard_normal(n)
            y = solver._linear_op.matvec(x)

            y_ref = np.zeros(n)
            for tid, S_i in tile_schur.items():
                idx = tile_idx[tid]
                y_ref[idx] += S_i @ x[idx]

            np.testing.assert_allclose(y, y_ref, rtol=1e-13, atol=1e-13)
        finally:
            solver.close()

    @pytest.mark.parametrize('n_threads', [1, 2, 8])
    def test_threaded_matches_serial_synthetic_30_tile(self, n_threads):
        """Gate: threaded-vs-serial matvec <=1e-13 rel on a randomized
        ~30-tile synthetic, thread-count invariance (1/2/8)."""
        from distributed.interface_iterative import InterfaceCGSolver

        n = 6000
        tile_schur, tile_idx = _make_synthetic_tiles(n, 30, 50, 250, seed=7)

        solver_serial = InterfaceCGSolver(
            n_interface=n, matvec_mode='tilewise',
            tile_schur_complements=tile_schur, tile_index_maps=tile_idx,
            preconditioner='none', matvec_threads=1,
        )
        solver_threaded = InterfaceCGSolver(
            n_interface=n, matvec_mode='tilewise',
            tile_schur_complements=tile_schur, tile_index_maps=tile_idx,
            preconditioner='none', matvec_threads=n_threads,
        )
        try:
            rng = np.random.default_rng(1)
            x = rng.standard_normal(n)
            y_serial = solver_serial._linear_op.matvec(x)
            y_threaded = solver_threaded._linear_op.matvec(x)
            rel = (
                np.max(np.abs(y_threaded - y_serial))
                / max(1e-300, float(np.max(np.abs(y_serial))))
            )
            assert rel <= 1e-13, f"threaded (n_threads={n_threads}) vs serial rel err {rel:.3e}"
        finally:
            solver_serial.close()
            solver_threaded.close()

    def test_threaded_matvec_with_s_extra(self):
        """S_extra (package-edge term) must be included identically regardless
        of thread count."""
        from distributed.interface_iterative import InterfaceCGSolver
        import scipy.sparse as sp

        n = 500
        tile_schur, tile_idx = _make_synthetic_tiles(n, 10, 20, 60, seed=11)
        rng = np.random.default_rng(5)
        extra_diag = rng.uniform(0.1, 1.0, size=n)
        S_extra = sp.diags(extra_diag).tocsr()

        solvers = {}
        for nt in (1, 8):
            solvers[nt] = InterfaceCGSolver(
                n_interface=n, matvec_mode='tilewise',
                tile_schur_complements=tile_schur, tile_index_maps=tile_idx,
                S_extra=S_extra, preconditioner='none', matvec_threads=nt,
            )
        try:
            x = rng.standard_normal(n)
            y1 = solvers[1]._linear_op.matvec(x)
            y8 = solvers[8]._linear_op.matvec(x)
            np.testing.assert_allclose(y1, y8, rtol=1e-13, atol=1e-13)

            y_ref = S_extra @ x
            for tid, S_i in tile_schur.items():
                idx = tile_idx[tid]
                y_ref[idx] += S_i @ x[idx]
            np.testing.assert_allclose(y1, y_ref, rtol=1e-12, atol=1e-12)
        finally:
            for s in solvers.values():
                s.close()


# ---------------------------------------------------------------------------
# _matmat vs column-by-column matvecs (item 6)
# ---------------------------------------------------------------------------


class TestMatmatVsColumnMatvecs:

    @pytest.mark.parametrize('n_threads', [1, 4])
    def test_matmat_matches_column_by_column(self, n_threads):
        from distributed.interface_iterative import InterfaceCGSolver

        n = 1500
        tile_schur, tile_idx = _make_synthetic_tiles(n, 12, 30, 120, seed=3)
        solver = InterfaceCGSolver(
            n_interface=n, matvec_mode='tilewise',
            tile_schur_complements=tile_schur, tile_index_maps=tile_idx,
            preconditioner='none', matvec_threads=n_threads,
        )
        try:
            rng = np.random.default_rng(9)
            k = 5
            X = rng.standard_normal((n, k))
            Y_matmat = solver._linear_op.matmat(X)
            Y_cols = np.column_stack(
                [solver._linear_op.matvec(X[:, j]) for j in range(k)]
            )
            np.testing.assert_allclose(Y_matmat, Y_cols, rtol=1e-12, atol=1e-12)
        finally:
            solver.close()


# ---------------------------------------------------------------------------
# Threaded block-Jacobi apply (item 5)
# ---------------------------------------------------------------------------


class TestThreadedBlockJacobi:

    @pytest.mark.parametrize('n_threads', [1, 8])
    def test_threaded_bj_matches_serial(self, n_threads):
        from distributed.interface_iterative import InterfaceCGSolver

        n = 4000
        tile_schur, tile_idx = _make_synthetic_tiles(n, 24, 40, 150, seed=13)

        solver_serial = InterfaceCGSolver(
            n_interface=n, matvec_mode='tilewise',
            tile_schur_complements=tile_schur, tile_index_maps=tile_idx,
            preconditioner='block_jacobi', matvec_threads=1,
        )
        solver_threaded = InterfaceCGSolver(
            n_interface=n, matvec_mode='tilewise',
            tile_schur_complements=tile_schur, tile_index_maps=tile_idx,
            preconditioner='block_jacobi', matvec_threads=n_threads,
        )
        try:
            assert solver_serial._M is not None
            assert solver_threaded._M is not None
            rng = np.random.default_rng(2)
            x = rng.standard_normal(n)
            z_serial = solver_serial._M.matvec(x)
            z_threaded = solver_threaded._M.matvec(x)
            np.testing.assert_allclose(z_serial, z_threaded, rtol=1e-13, atol=1e-13)
        finally:
            solver_serial.close()
            solver_threaded.close()

    def test_threaded_bj_matches_serial_ill_conditioned_block(self):
        """Finding 9 (PLAUSIBLE) extension: on an ill-conditioned block
        (cond ~1e10 -- the module docstring's own motivating regime), the
        materialized dense inverse Sinv must be symmetrized in place, and
        threaded vs serial apply must still agree tightly. A non-symmetric
        Sinv would weaken PCG's SPD-preconditioner assumption; this
        asserts both the stored block's exact symmetry and apply-output
        equivalence."""
        from distributed.interface_iterative import InterfaceCGSolver

        rng = np.random.default_rng(41)
        n = 50
        V, _ = np.linalg.qr(rng.standard_normal((n, n)))
        # cond ~1e10: eigenvalues log-spaced from 1e-5 to 1e5.
        w = np.logspace(-5, 5, n)
        S_i = (V * w) @ V.T
        S_i = 0.5 * (S_i + S_i.T)
        tile_schur = {0: S_i}
        tile_idx = {0: np.arange(n, dtype=np.int32)}

        solver_serial = InterfaceCGSolver(
            n_interface=n, matvec_mode='tilewise',
            tile_schur_complements={0: S_i.copy()}, tile_index_maps=tile_idx,
            preconditioner='block_jacobi', matvec_threads=1,
        )
        solver_threaded = InterfaceCGSolver(
            n_interface=n, matvec_mode='tilewise',
            tile_schur_complements={0: S_i.copy()}, tile_index_maps=tile_idx,
            preconditioner='block_jacobi', matvec_threads=8,
        )
        try:
            # The stored materialized inverse must be EXACTLY symmetric
            # (0.5*(Sinv+Sinv.T) is bit-exact symmetric by construction).
            Sinv = solver_serial._bj_inv_blocks[0]
            np.testing.assert_array_equal(Sinv, Sinv.T)

            rng2 = np.random.default_rng(42)
            x = rng2.standard_normal(n)
            z_serial = solver_serial._M.matvec(x)
            z_threaded = solver_threaded._M.matvec(x)
            np.testing.assert_allclose(
                z_serial, z_threaded, rtol=1e-12, atol=1e-12,
            )
        finally:
            solver_serial.close()
            solver_threaded.close()


# ---------------------------------------------------------------------------
# SPD-safe fallback (item 7)
# ---------------------------------------------------------------------------


class TestSPDSafeFallback:

    def test_eigh_fallback_produces_psd_map(self):
        """Constructed indefinite block: the eigh-based fallback must yield a
        strictly positive-definite pseudo-inverse map (unlike raw pinv, which
        can retain negative eigenvalues from an indefinite input)."""
        from distributed.interface_iterative import _spd_safe_pseudo_solve_factor

        rng = np.random.default_rng(0)
        V, _ = np.linalg.qr(rng.standard_normal((5, 5)))
        w_true = np.array([-2.0, -0.5, 0.1, 3.0, 10.0])  # indefinite
        A = V @ np.diag(w_true) @ V.T
        A = 0.5 * (A + A.T)

        factor = _spd_safe_pseudo_solve_factor(A, eps_rel=1e-10)
        assert factor is not None
        Vf, inv_w = factor
        assert np.all(inv_w > 0), "clipped inverse eigenvalues must be strictly positive"

        M = Vf @ np.diag(inv_w) @ Vf.T
        eigvals = np.linalg.eigvalsh(M)
        assert np.all(eigvals > 0), "SPD-safe fallback must produce a PSD (in fact SPD) map"

    def test_degenerate_all_nonpositive_block_returns_none(self):
        from distributed.interface_iterative import _spd_safe_pseudo_solve_factor

        A = -np.eye(3)  # negative definite: no positive spectrum at all
        assert _spd_safe_pseudo_solve_factor(A) is None

    def test_block_jacobi_engages_eigh_fallback_for_indefinite_owned_block(self, caplog):
        """A single-tile fixture whose own Schur block is indefinite forces
        cho_factor to fail; verify the eigh fallback engages (not pinv) and
        the preconditioner still builds successfully."""
        import logging
        from distributed.interface_iterative import InterfaceCGSolver

        n = 3
        idx = np.array([0, 1, 2], dtype=np.int32)
        rng = np.random.default_rng(4)
        V, _ = np.linalg.qr(rng.standard_normal((3, 3)))
        S_i = V @ np.diag([-5.0, -1.0, 2.0]) @ V.T
        S_i = 0.5 * (S_i + S_i.T)
        tile_schur = {'A': S_i}
        tile_idx = {'A': idx}

        with caplog.at_level(logging.WARNING):
            solver = InterfaceCGSolver(
                n_interface=n, matvec_mode='tilewise',
                tile_schur_complements=tile_schur, tile_index_maps=tile_idx,
                preconditioner='block_jacobi', matvec_threads=1,
            )
        try:
            assert solver._M is not None
            assert any('eigh-based' in rec.message for rec in caplog.records), (
                "expected the SPD-safe eigh fallback WARNING, not pinv"
            )
            # kind stored for the (only) owned block must be 'eigh'.
            kinds = [kind for _idx, kind, _payload in solver._bj_block_factors]
            assert kinds == ['eigh']
        finally:
            solver.close()


# ---------------------------------------------------------------------------
# fp32 critical path (item 10)
# ---------------------------------------------------------------------------


class TestFp32CriticalPath:

    def test_fp32_matches_fp64_within_expected_floor(self):
        from distributed.interface_iterative import InterfaceCGSolver

        n = 3000
        tile_schur, tile_idx = _make_synthetic_tiles(n, 20, 40, 150, seed=21)
        # Finding 3: InterfaceCGSolver now frees fp64 originals in place by
        # mutating the caller's tile_schur_complements dict -- give each
        # solver its OWN dict (shallow copy; per-tile arrays are still only
        # referenced once each) so solver32's in-place conversion cannot
        # leak into solver64's blocks. Production never shares one dict
        # across two InterfaceCGSolver instances (each factor()/refactor()
        # call builds a fresh dict), so this sharing is test-only.
        tile_schur_64 = dict(tile_schur)
        tile_schur_32 = {tid: np.array(S, copy=True) for tid, S in tile_schur.items()}

        solver64 = InterfaceCGSolver(
            n_interface=n, matvec_mode='tilewise',
            tile_schur_complements=tile_schur_64, tile_index_maps=tile_idx,
            preconditioner='none', matvec_threads=4, matvec_dtype='float64',
        )
        solver32 = InterfaceCGSolver(
            n_interface=n, matvec_mode='tilewise',
            tile_schur_complements=tile_schur_32, tile_index_maps=tile_idx,
            preconditioner='none', matvec_threads=4, matvec_dtype='float32',
            rtol=1e-7,
        )
        try:
            rng = np.random.default_rng(6)
            x = rng.standard_normal(n)
            y64 = solver64._linear_op.matvec(x)
            y32 = solver32._linear_op.matvec(x)
            rel = (
                np.max(np.abs(y32 - y64))
                / max(1e-300, float(np.max(np.abs(y64))))
            )
            # fp32 storage should be close to fp64 (single-precision floor,
            # not wildly off); a generous bound avoids host-dependent flakiness.
            assert rel < 1e-4, f"fp32 vs fp64 matvec rel err too large: {rel:.3e}"
            assert solver32.matvec_dtype == np.dtype(np.float32)

            # Finding 3: fp32 storage must not retain the fp64 originals
            # alongside the fp32 copies (that would be strictly worse than
            # fp64-only). solver64's dict is untouched (dtype == float64
            # is a no-op cast); solver32's dict entries must all have been
            # converted+freed in place during construction.
            for S_i in solver64.tile_schur_complements.values():
                assert np.asarray(S_i).dtype == np.float64
            for S_i in solver32.tile_schur_complements.values():
                assert np.asarray(S_i).dtype == np.float32, (
                    "fp64 original not freed: tile_schur_complements still "
                    "holds a float64 array after fp32 conversion"
                )
        finally:
            solver64.close()
            solver32.close()

    def test_fp32_requires_rtol_floor_by_default(self):
        from distributed.interface_iterative import InterfaceCGSolver

        n = 100
        tile_schur, tile_idx = _make_synthetic_tiles(n, 4, 10, 40, seed=23)
        with pytest.raises(ValueError, match='matvec_dtype'):
            InterfaceCGSolver(
                n_interface=n, matvec_mode='tilewise',
                tile_schur_complements=tile_schur, tile_index_maps=tile_idx,
                preconditioner='none', matvec_dtype='float32', rtol=1e-10,
            )

    def test_fp32_rtol_floor_overridable(self):
        from distributed.interface_iterative import InterfaceCGSolver

        n = 100
        tile_schur, tile_idx = _make_synthetic_tiles(n, 4, 10, 40, seed=23)
        solver = InterfaceCGSolver(
            n_interface=n, matvec_mode='tilewise',
            tile_schur_complements=tile_schur, tile_index_maps=tile_idx,
            preconditioner='none', matvec_dtype='float32', rtol=1e-10,
            strict_dtype_rtol=False,
        )
        solver.close()


# ---------------------------------------------------------------------------
# matvec_mode='auto' resolution (item 8)
# ---------------------------------------------------------------------------


class TestMatvecModeAuto:

    def test_auto_resolves_tilewise_when_blocks_available(self):
        from distributed.interface_iterative import resolve_matvec_mode
        assert resolve_matvec_mode('auto', True) == 'tilewise'
        assert resolve_matvec_mode(None, True) == 'tilewise'

    def test_auto_resolves_assembled_when_blocks_unavailable(self):
        from distributed.interface_iterative import resolve_matvec_mode
        assert resolve_matvec_mode('auto', False) == 'assembled'

    def test_explicit_values_pass_through(self):
        from distributed.interface_iterative import resolve_matvec_mode
        assert resolve_matvec_mode('assembled', True) == 'assembled'
        assert resolve_matvec_mode('tilewise', False) == 'tilewise'

    def test_invalid_value_raises(self):
        from distributed.interface_iterative import resolve_matvec_mode
        with pytest.raises(ValueError):
            resolve_matvec_mode('bogus', True)

    def test_build_interface_solver_default_is_auto(self):
        """Item 8: build_interface_solver's own matvec_mode default resolves
        to tilewise when tile blocks are supplied, without the caller
        passing matvec_mode explicitly."""
        from distributed.interface_iterative import build_interface_solver

        n = 30
        tile_schur, tile_idx = _make_synthetic_tiles(n, 4, 6, 12, seed=31)
        solve, mode, cg_solver = build_interface_solver(
            S_global=None, interface_solver='cg',
            tile_schur_complements=tile_schur, tile_index_maps=tile_idx,
            n_interface=n, preconditioner='none',
        )
        try:
            assert mode == 'cg'
            assert cg_solver.matvec_mode == 'tilewise'
        finally:
            cg_solver.close()


# ---------------------------------------------------------------------------
# matvec_threads resolution (item 4)
# ---------------------------------------------------------------------------


class TestMatvecThreadsResolution:

    def test_auto_caps_at_8_not_32(self):
        from distributed.interface_iterative import (
            resolve_matvec_threads, DEFAULT_MATVEC_THREADS_CAP,
        )
        assert DEFAULT_MATVEC_THREADS_CAP == 8
        # Plenty of tiles/CPUs available on any CI host -- 'auto' must not
        # exceed the cap even when both cpu_count and n_tiles are large.
        assert resolve_matvec_threads('auto', n_tiles=1000) <= 8

    def test_auto_bounded_by_n_tiles(self):
        from distributed.interface_iterative import resolve_matvec_threads
        assert resolve_matvec_threads('auto', n_tiles=1) == 1
        assert resolve_matvec_threads('auto', n_tiles=3) <= 3

    def test_explicit_value_not_capped(self):
        from distributed.interface_iterative import resolve_matvec_threads
        assert resolve_matvec_threads(16, n_tiles=2) == 16

    def test_explicit_string_coerced(self):
        from distributed.interface_iterative import resolve_matvec_threads
        assert resolve_matvec_threads('4', n_tiles=10) == 4

    def test_invalid_raises(self):
        from distributed.interface_iterative import resolve_matvec_threads
        with pytest.raises(ValueError):
            resolve_matvec_threads(0, n_tiles=10)
        with pytest.raises(ValueError):
            resolve_matvec_threads(True, n_tiles=10)


# ---------------------------------------------------------------------------
# Thread-pool lifecycle
# ---------------------------------------------------------------------------


class TestPoolLifecycle:

    def test_close_shuts_down_pool(self):
        from distributed.interface_iterative import InterfaceCGSolver

        n = 200
        tile_schur, tile_idx = _make_synthetic_tiles(n, 6, 10, 40, seed=41)
        solver = InterfaceCGSolver(
            n_interface=n, matvec_mode='tilewise',
            tile_schur_complements=tile_schur, tile_index_maps=tile_idx,
            preconditioner='none', matvec_threads=4,
        )
        rng = np.random.default_rng(1)
        x = rng.standard_normal(n)
        solver._linear_op.matvec(x)  # forces pool creation
        assert solver._pool is not None
        solver.close()
        assert solver._pool is None
        solver.close()  # idempotent

    def test_repeated_build_and_close_no_thread_leak(self):
        from distributed.interface_iterative import InterfaceCGSolver

        n = 200
        tile_schur, tile_idx = _make_synthetic_tiles(n, 6, 10, 40, seed=43)
        baseline = threading.active_count()
        rng = np.random.default_rng(2)
        x = rng.standard_normal(n)
        for _ in range(5):
            solver = InterfaceCGSolver(
                n_interface=n, matvec_mode='tilewise',
                tile_schur_complements=tile_schur, tile_index_maps=tile_idx,
                preconditioner='block_jacobi', matvec_threads=4,
            )
            solver._linear_op.matvec(x)
            solver._M.matvec(x)
            solver.close()
        assert threading.active_count() == baseline, (
            "repeated build/close cycles must not accumulate live threads"
        )


# ---------------------------------------------------------------------------
# Zero-port regression (item 11)
# ---------------------------------------------------------------------------


class TestZeroPortRegression:

    def test_build_block_system_zero_ports_correct_shapes(self):
        from pgmath.block_system import build_block_system_from_edges

        edges = [('a', 'b', 1.0), ('b', 'c', 1.0), ('c', '0', 1.0)]
        bs, rhs = build_block_system_from_edges(
            edges, port_nodes=set(), dirichlet_nodes=None,
        )
        assert bs.n_ports == 0
        assert bs.n_interior == 3
        assert bs.G_pi.shape == (0, 3)
        assert bs.G_ip.shape == (3, 0)

    def test_compute_explicit_schur_zero_ports_no_crash(self):
        from pgmath.block_system import build_block_system_from_edges
        from pgmath.schur import compute_explicit_schur

        edges = [('a', 'b', 1.0), ('b', 'c', 1.0), ('c', '0', 1.0)]
        bs, rhs = build_block_system_from_edges(
            edges, port_nodes=set(), dirichlet_nodes=None,
        )
        S, stats = compute_explicit_schur(bs)
        assert S.shape == (0, 0)
        assert bs.lu_ii is not None
        # Interior solve must still work (used by recovery).
        x = bs.lu_ii(np.ones(3))
        assert x.shape == (3,)

    def test_zero_interior_still_trivial(self):
        """Companion case (n_interior == 0): must remain unaffected."""
        from pgmath.block_system import build_block_system_from_edges
        from pgmath.schur import compute_explicit_schur

        edges = [('a', 'b', 1.0)]
        bs, rhs = build_block_system_from_edges(
            edges, port_nodes={'a', 'b'}, dirichlet_nodes=None,
        )
        assert bs.n_interior == 0
        S, stats = compute_explicit_schur(bs)
        assert stats['path'] == 'trivial'
        assert S.shape == (2, 2)


# ---------------------------------------------------------------------------
# D1 (item 1) -- full solve_dc()-level regression on the pad-on-tile-port
# fixture.  TestForcedCGEquivalence (test_interface_iterative.py) already
# covers the tilewise-MATVEC-level fix; this class exercises the SECOND D1
# manifestation (solve_dc's RHS scatter) end-to-end through solve_dc().
# ---------------------------------------------------------------------------


class TestD1SolveDcRegression:

    @staticmethod
    def _build_model():
        return _build_two_tile_distributed_model(package_cap_edges=[])

    def test_solve_dc_cg_tilewise_matches_direct_on_pad_on_port_fixture(self):
        from distributed.solver import DistributedDDMSolver

        def _run(interface_solver, matvec_mode):
            model = self._build_model()
            model.settings.update({
                'interface_solver': interface_solver,
                'interface_matvec_mode': matvec_mode,
                'interface_preconditioner': 'block_jacobi',
                'interface_cg_rtol': 1e-12,
            })
            solver = DistributedDDMSolver(model)
            ctx = solver.prepare()
            try:
                result = solver.solve_dc(ctx)
                flat = result.flatten()
            finally:
                ctx.release()
                model.shutdown()
            return flat

        v_direct = _run('direct', 'assembled')
        v_cg_assembled = _run('cg', 'assembled')
        v_cg_tilewise = _run('cg', 'tilewise')

        common = set(v_direct) & set(v_cg_tilewise)
        assert common, "fixture produced no comparable nodes"
        for node in common:
            assert abs(v_direct[node] - v_cg_tilewise[node]) < 1e-8, (
                f"node {node}: direct={v_direct[node]!r} vs "
                f"cg/tilewise={v_cg_tilewise[node]!r}"
            )
            assert abs(v_direct[node] - v_cg_assembled[node]) < 1e-8, (
                f"node {node}: direct={v_direct[node]!r} vs "
                f"cg/assembled={v_cg_assembled[node]!r}"
            )

    def test_solve_dc_streaming_assembly_matches_bulk_on_pad_on_port_fixture(self):
        """Finding 4: the streaming DC assembly path (_stream_assemble_schur)
        must populate tile_kept_port_pos for a tile-resident pad-on-port
        fixture (this class's standard D1 fixture -- tile (0,0)'s port list
        is ['pad', 'shared']), not leave it empty.  Before this fix,
        solve_dc's D1 reduced-RHS scatter (solver.py) would raise
        ``AssertionError: ... no usable tile_kept_port_pos entry is
        available to recover`` whenever streaming_assembly=True hit this
        exact scenario -- streaming DC with a tile-resident pad on a port.
        """
        from distributed.solver import DistributedDDMSolver

        model_bulk = self._build_model()
        model_bulk.settings.update({'streaming_assembly': False})
        solver_bulk = DistributedDDMSolver(model_bulk)
        ctx_bulk = solver_bulk.prepare()
        try:
            v_bulk = solver_bulk.solve_dc(ctx_bulk).flatten()
        finally:
            ctx_bulk.release()
            model_bulk.shutdown()

        model_stream = self._build_model()
        model_stream.settings.update({'streaming_assembly': True})
        solver_stream = DistributedDDMSolver(model_stream)
        ctx_stream = solver_stream.prepare()
        try:
            # Finding 4: must be populated (not empty) so the D1 RHS scatter
            # can recover tile (0,0)'s tile-resident 'pad' port.
            assert ctx_stream.tile_kept_port_pos, (
                "streaming DC factor must populate tile_kept_port_pos"
            )
            kept_00 = ctx_stream.tile_kept_port_pos.get((0, 0))
            assert kept_00 is not None and len(kept_00) == 1, (
                f"tile (0,0) port list is ['pad', 'shared']; only 'shared' "
                f"(position 1) should be kept, got {kept_00}"
            )
            assert list(kept_00) == [1]

            v_stream = solver_stream.solve_dc(ctx_stream).flatten()
        finally:
            ctx_stream.release()
            model_stream.shutdown()

        common = set(v_bulk) & set(v_stream)
        assert common, "fixture produced no comparable nodes"
        for node in common:
            assert abs(v_bulk[node] - v_stream[node]) < 1e-8, (
                f"node {node}: bulk={v_bulk[node]!r} vs streaming={v_stream[node]!r}"
            )


# ---------------------------------------------------------------------------
# S2/S13: D1 pad-on-port fix extended to solve_quasi_static, solve_transient,
# and analyze_adjoint_static (originally only solve_dc was fixed).
# ---------------------------------------------------------------------------


class TestD1QsTransientAdjointRegression:
    """Regression test for the S2 finding: the D1 kept-position RHS filter
    existed only in solve_dc; solve_quasi_static/solve_transient/
    analyze_adjoint_static scatter sites still paired full-port-length g_i
    with pad-filtered tile_index_maps, crashing (or silently corrupting) on
    the exact pad-on-tile-port fixture the D1 campaign targets.
    """

    @staticmethod
    def _build_model(tmp_path):
        # A real (existing) ckt_path parent dir is required: preprocess_
        # sources()/_ensure_adjoint_source_mappings() derive the VCS pkl_dir
        # cache location from os.path.dirname(tile_configs[0].ckt_path), and
        # the standard fixture's ckt_path='' resolves to '' (os.makedirs('')
        # raises FileNotFoundError).
        model = _build_two_tile_distributed_model(package_cap_edges=[])
        for tc in model.metadata.tile_configs:
            tc.ckt_path = str(tmp_path / 'dummy.ckt')
        return model

    def test_solve_quasi_static_cg_tilewise_matches_direct_on_pad_on_port_fixture(
        self, tmp_path,
    ):
        from distributed.solver import DistributedDDMSolver

        def _run(interface_solver, matvec_mode):
            model = self._build_model(tmp_path)
            model.settings.update({
                'interface_solver': interface_solver,
                'interface_matvec_mode': matvec_mode,
                'interface_preconditioner': 'block_jacobi',
                'interface_cg_rtol': 1e-12,
            })
            solver = DistributedDDMSolver(model)
            ctx = solver.prepare()
            try:
                # Pre-fix, this raised ValueError/AssertionError from
                # np.bincount ("weights and list don't have the same
                # length") on tile (0, 0)'s pad-on-port scatter.
                result = solver.solve_quasi_static(
                    ctx, t_start=0.0, t_end=2e-9, n_points=5,
                )
                flat = result.as_flat()
            finally:
                ctx.release()
                model.shutdown()
            return flat

        v_direct = _run('direct', 'assembled')
        v_cg_tilewise = _run('cg', 'tilewise')

        common = set(v_direct) & set(v_cg_tilewise)
        assert common, "fixture produced no comparable nodes"
        for node in common:
            drop_d, _t_d = v_direct[node]
            drop_c, _t_c = v_cg_tilewise[node]
            assert abs(drop_d - drop_c) < 1e-8, (
                f"node {node}: direct drop={drop_d!r} vs cg/tilewise drop={drop_c!r}"
            )

    def test_solve_transient_cg_tilewise_matches_direct_on_pad_on_port_fixture(
        self, tmp_path,
    ):
        from distributed.solver import DistributedDDMSolver

        def _run(interface_solver, matvec_mode):
            model = self._build_model(tmp_path)
            model.settings.update({
                'interface_solver': interface_solver,
                'interface_matvec_mode': matvec_mode,
                'interface_preconditioner': 'block_jacobi',
                'interface_cg_rtol': 1e-12,
            })
            solver = DistributedDDMSolver(model)
            dc_ctx = solver.prepare()
            trans_ctx = solver.prepare_transient(dt=1e-10, method='be')
            try:
                smoothed = solver.preprocess_sources(
                    time_step=1e-10, t_start=0.0, t_end=2e-9, smooth=False,
                )
                # Pre-fix, this raised from np.bincount on tile (0, 0)'s
                # pad-on-port scatter (the same D1 scenario as solve_dc).
                result = solver.solve_transient(
                    trans_ctx, dc_context=dc_ctx, t_end=2e-9,
                    smoothed_sources=smoothed,
                )
                flat = result.as_flat()
            finally:
                trans_ctx.release()
                dc_ctx.release()
                model.shutdown()
            return flat

        v_direct = _run('direct', 'assembled')
        v_cg_tilewise = _run('cg', 'tilewise')

        common = set(v_direct) & set(v_cg_tilewise)
        assert common, "fixture produced no comparable nodes"
        for node in common:
            drop_d, _t_d = v_direct[node]
            drop_c, _t_c = v_cg_tilewise[node]
            assert abs(drop_d - drop_c) < 1e-8, (
                f"node {node}: direct drop={drop_d!r} vs cg/tilewise drop={drop_c!r}"
            )

    def test_analyze_adjoint_static_cg_tilewise_matches_direct_on_pad_on_port_fixture(
        self, tmp_path,
    ):
        from distributed.solver import DistributedDDMSolver

        def _run(interface_solver, matvec_mode):
            model = self._build_model(tmp_path)
            model.settings.update({
                'interface_solver': interface_solver,
                'interface_matvec_mode': matvec_mode,
                'interface_preconditioner': 'block_jacobi',
                'interface_cg_rtol': 1e-12,
            })
            solver = DistributedDDMSolver(model)
            dc_ctx = solver.prepare()
            try:
                # Pre-fix, this raised from np.add.at shape mismatch on
                # tile (0, 0)'s pad-on-port scatter.
                attribution = solver.analyze_adjoint_static(
                    victim_nodes='b1', observation_time=1e-9,
                    dc_context=dc_ctx,
                )
            finally:
                dc_ctx.release()
                model.shutdown()
            return attribution.ir_drop_at_T

        ir_drop_direct = _run('direct', 'assembled')
        ir_drop_cg_tilewise = _run('cg', 'tilewise')

        assert abs(ir_drop_direct - ir_drop_cg_tilewise) < 1e-5, (
            f"direct ir_drop_at_T={ir_drop_direct!r} vs "
            f"cg/tilewise ir_drop_at_T={ir_drop_cg_tilewise!r}"
        )


# ---------------------------------------------------------------------------
# refactor() rebuilds tilewise instead of downgrading (item 9)
# ---------------------------------------------------------------------------


class TestRefactorRebuildsTilewise:

    @staticmethod
    def _build_model():
        return _build_two_tile_distributed_model(package_cap_edges=[])

    def test_refactor_after_release_rebuilds_tilewise(self, tmp_path):
        from distributed.solver import DistributedDDMSolver
        from distributed.result import DistributedSolverContext

        model = self._build_model()
        model.settings.update({
            'interface_solver': 'cg',
            'interface_matvec_mode': 'tilewise',
            'interface_preconditioner': 'block_jacobi',
            'interface_cg_rtol': 1e-12,
        })
        solver = DistributedDDMSolver(model)
        try:
            ctx = solver.prepare()
            assert ctx._cg_solver.matvec_mode == 'tilewise'

            path = ctx.save(str(tmp_path / 'dc_ctx.pkl'))
            ctx.release()  # clears coordinator LU + worker LU (G matrices preserved)

            # Workers are still attached (same model) -- load() restores
            # S_global (needed for the top-of-function guard) and topology;
            # refactor() must then re-gather S_i via streaming and rebuild
            # tilewise, not downgrade to 'assembled'.
            ctx = DistributedSolverContext.load(model, path)
            ctx.refactor()
            assert ctx._cg_solver is not None
            assert ctx._cg_solver.matvec_mode == 'tilewise', (
                "refactor() with attached workers must rebuild tilewise, "
                "not silently downgrade to 'assembled'"
            )

            result = solver.solve_dc(ctx)
            assert result.flatten()  # sanity: solve actually runs
        finally:
            ctx.release()
            model.shutdown()

    def test_refactor_without_workers_falls_back_to_assembled_with_warning(
        self, tmp_path, caplog,
    ):
        import logging
        from distributed.result import DistributedSolverContext

        model = self._build_model()
        model.settings.update({
            'interface_solver': 'cg',
            'interface_matvec_mode': 'tilewise',
            'interface_preconditioner': 'block_jacobi',
            'interface_cg_rtol': 1e-12,
        })
        from distributed.solver import DistributedDDMSolver
        solver = DistributedDDMSolver(model)
        ctx = solver.prepare()
        path = ctx.save(str(tmp_path / 'dc_ctx2.pkl'))
        ctx.release()
        model.shutdown()  # workers fully torn down

        # Load into a model-less context stand-in: DistributedSolverContext.load
        # requires a live model, so simulate "no workers attached" by loading
        # with a model whose .workers is empty.
        class _EmptyWorkersModel:
            workers = []
            settings = {
                'interface_solver': 'cg',
                'interface_matvec_mode': 'tilewise',
                'interface_preconditioner': 'block_jacobi',
                'interface_cg_rtol': 1e-12,
            }
            coordinator_solver_config = None

        ctx2 = DistributedSolverContext.load(_EmptyWorkersModel(), path)
        with caplog.at_level(logging.WARNING):
            ctx2.refactor()
        assert ctx2._cg_solver.matvec_mode == 'assembled'
        assert any(
            'no workers are attached' in rec.message for rec in caplog.records
        )

    def test_refactor_auto_preconditioner_degrades_to_jacobi_not_unpreconditioned(
        self, tmp_path, caplog,
    ):
        """Regression (spec-compliance review round 1, finding on
        result_factorization.py:2650/3623): Stage 3 flipped the
        ``interface_preconditioner`` DEFAULT from the literal
        ``'block_jacobi'`` to ``'auto'``, but the degrade-to-jacobi guard in
        both refactor paths keyed off the literal ``'block_jacobi'``/
        ``'two_level'`` names only. With the (now-default) 'auto' setting
        and ``tile_index_maps`` genuinely unavailable on ``ctx.topology``
        (the pre-B2-checkpoint case the guard's own comment names), the
        guard did NOT fire; 'auto' then resolved to 'block_jacobi' inside
        ``build_interface_solver``, and ``_build_block_jacobi()`` returned
        None (no ``tile_index_maps``, no ``tile_schur_complements``) --
        silently running UNPRECONDITIONED CG instead of the intended
        diagonal 'jacobi' fallback.

        Note: merely detaching workers (as the sibling test above does) is
        NOT sufficient to reach this branch -- ``ctx.topology.
        tile_index_maps`` survives save()/load() independently of worker
        attachment, so 'block_jacobi'/'two_level' still build a WORKING
        preconditioner from ``S_global`` + the restored ``tile_index_maps``
        in that case (this is intended: see ``_build_block_jacobi``'s
        docstring path 1). This test additionally clears
        ``ctx.topology.tile_index_maps`` after load() to simulate a
        checkpoint that genuinely predates/lacks the field -- the guard's
        actual `not _tile_index_maps` precondition.

        This test leaves ``interface_preconditioner`` UNSET (exercising the
        Stage 3 default) where the sibling test above sets it explicitly to
        'block_jacobi'.
        """
        import logging
        from distributed.result import DistributedSolverContext

        model = self._build_model()
        model.settings.update({
            'interface_solver': 'cg',
            'interface_matvec_mode': 'tilewise',
            # Deliberately NOT setting interface_preconditioner: exercise
            # the Stage 3 default ('auto'), not an explicit setting.
            'interface_cg_rtol': 1e-12,
        })
        from distributed.solver import DistributedDDMSolver
        solver = DistributedDDMSolver(model)
        ctx = solver.prepare()
        # Finding 10 (round 2): try/finally around save/release/shutdown,
        # matching the TD twin (test_refactor_transient_auto_
        # preconditioner_degrades_to_jacobi, below) -- straight-line code
        # here would leak the factored context (workers, thread pool) if
        # ctx.save() ever raised.
        try:
            path = ctx.save(str(tmp_path / 'dc_ctx3.pkl'))
        finally:
            ctx.release()
            model.shutdown()  # workers fully torn down

        class _EmptyWorkersModel:
            workers = []
            settings = {
                'interface_solver': 'cg',
                'interface_matvec_mode': 'tilewise',
                'interface_cg_rtol': 1e-12,
            }
            coordinator_solver_config = None

        ctx2 = DistributedSolverContext.load(_EmptyWorkersModel(), path)
        # Simulate a checkpoint that predates/lacks tile_index_maps (the
        # guard's documented trigger case) -- with workers detached too, so
        # the tilewise re-gather branch (which would otherwise refresh
        # _tile_index_maps from live workers) cannot run.
        ctx2.topology.tile_index_maps = {}
        with caplog.at_level(logging.WARNING):
            ctx2.refactor()
        assert ctx2._cg_solver.matvec_mode == 'assembled'
        # The degrade-to-jacobi guard must fire for the 'auto' default too.
        assert ctx2._cg_solver.preconditioner == 'jacobi', (
            f"expected the degrade-to-jacobi guard to fire for the 'auto' "
            f"default preconditioner when tile_index_maps is unavailable; "
            f"got preconditioner={ctx2._cg_solver.preconditioner!r} "
            f"(pre-fix: stayed 'auto' -> resolved to 'block_jacobi' inside "
            f"build_interface_solver -> _build_block_jacobi() returned "
            f"None)"
        )
        # ...and it must be a WORKING preconditioner (M is not None), not
        # silently-unpreconditioned CG.
        assert ctx2._cg_solver._M is not None, (
            "degrade-to-jacobi guard bypassed: CG is running with M=None "
            "(unpreconditioned) instead of the diagonal 'jacobi' fallback"
        )
        assert any(
            'jacobi' in rec.message.lower() for rec in caplog.records
        )

    def test_mixed_case_two_level_setting_still_hits_degrade_guard(
        self, tmp_path, caplog,
    ):
        """Finding 1 (round 2) regression: ``interface_preconditioner=
        'Two_Level'`` (mixed case -- a valid-but-sloppy YAML scalar that
        ``resolve_preconditioner``'s own normalization already tolerates at
        factor() time) must ALSO be recognized by the refactor degrade
        guards in this module, which (pre-fix) compared the RAW settings
        string against lowercase literals
        (``_preconditioner in ('block_jacobi', 'two_level', 'auto')``).
        Without normalizing at settings-READ time (Finding 1's fix, in
        ``_read_interface_cg_settings``), the mixed-case string bypasses
        that tile_index_maps-unavailable guard, silently building NO
        preconditioner (M=None, unpreconditioned CG) instead of degrading
        to 'jacobi' -- the exact silent regression the guard exists to
        prevent, just reached via a case-variant spelling instead of
        'auto'.
        """
        import logging
        from distributed.result import DistributedSolverContext

        model = self._build_model()
        model.settings.update({
            'interface_solver': 'cg',
            'interface_matvec_mode': 'tilewise',
            'interface_preconditioner': 'Two_Level',  # mixed case
            'interface_cg_rtol': 1e-12,
        })
        from distributed.solver import DistributedDDMSolver
        solver = DistributedDDMSolver(model)
        ctx = solver.prepare()
        try:
            path = ctx.save(str(tmp_path / 'dc_ctx_f1_mixed_case.pkl'))
        finally:
            ctx.release()
            model.shutdown()  # workers fully torn down

        class _EmptyWorkersModel:
            workers = []
            settings = {
                'interface_solver': 'cg',
                'interface_matvec_mode': 'tilewise',
                'interface_preconditioner': 'Two_Level',
                'interface_cg_rtol': 1e-12,
            }
            coordinator_solver_config = None

        ctx2 = DistributedSolverContext.load(_EmptyWorkersModel(), path)
        # Simulate a checkpoint that predates/lacks tile_index_maps, with
        # workers detached so the tilewise re-gather branch cannot refresh
        # it from live workers -- the degrade guard's documented trigger.
        ctx2.topology.tile_index_maps = {}
        with caplog.at_level(logging.WARNING):
            ctx2.refactor()
        assert ctx2._cg_solver.matvec_mode == 'assembled'
        assert ctx2._cg_solver.preconditioner == 'jacobi', (
            f"expected the mixed-case 'Two_Level' setting to still hit the "
            f"degrade-to-jacobi guard; got preconditioner="
            f"{ctx2._cg_solver.preconditioner!r} (pre-fix: the raw "
            f"'Two_Level' string bypassed the lowercase-literal guard, "
            f"leaving CG unpreconditioned)"
        )
        assert ctx2._cg_solver._M is not None, (
            "degrade-to-jacobi guard bypassed: CG is running with M=None "
            "(unpreconditioned) for the mixed-case setting"
        )
        assert any(
            'jacobi' in rec.message.lower() for rec in caplog.records
        )

    def test_refactor_transient_auto_preconditioner_degrades_to_jacobi(
        self, tmp_path, caplog,
    ):
        """Transient twin of
        ``test_refactor_auto_preconditioner_degrades_to_jacobi_not_
        unpreconditioned`` -- the reviewer finding flagged BOTH refactor
        guards (result_factorization.py:2650 DC and :3623 TD) as sharing
        the identical bypass, since they are separate copy-pasted `if`
        statements rather than shared code. Covers the TD-specific guard
        independently so a future edit that fixes one copy but not the
        other is caught.
        """
        import logging
        from distributed.result import DistributedTransientContext

        model = _build_two_tile_distributed_model(package_cap_edges=[])
        model.settings.update({
            'interface_solver': 'cg',
            'interface_matvec_mode': 'tilewise',
            # Deliberately NOT setting interface_preconditioner: exercise
            # the Stage 3 default ('auto'), not an explicit setting.
            'interface_cg_rtol': 1e-12,
        })
        from distributed.solver import DistributedDDMSolver
        solver = DistributedDDMSolver(model)
        dc_ctx = solver.prepare()
        trans_ctx = solver.prepare_transient(dt=1e-10, method='be')
        try:
            td_path = trans_ctx.save(str(tmp_path / 'td_ctx_auto.pkl'))
        finally:
            trans_ctx.release()
            dc_ctx.release()
            model.shutdown()  # workers fully torn down

        class _EmptyWorkersModel:
            workers = []
            settings = {
                'interface_solver': 'cg',
                'interface_matvec_mode': 'tilewise',
                'interface_cg_rtol': 1e-12,
            }
            coordinator_solver_config = None

        trans_ctx2 = DistributedTransientContext.load(_EmptyWorkersModel(), td_path)
        # Simulate a checkpoint that predates/lacks tile_index_maps, with
        # workers detached so the tilewise re-gather branch cannot refresh
        # it from live workers -- see the DC sibling test's docstring.
        trans_ctx2.topology.tile_index_maps = {}
        with caplog.at_level(logging.WARNING):
            trans_ctx2.refactor()
        assert trans_ctx2._cg_solver.matvec_mode == 'assembled'
        assert trans_ctx2._cg_solver.preconditioner == 'jacobi', (
            f"expected the degrade-to-jacobi guard to fire for the 'auto' "
            f"default preconditioner (TD refactor path) when "
            f"tile_index_maps is unavailable; got "
            f"preconditioner={trans_ctx2._cg_solver.preconditioner!r}"
        )
        assert trans_ctx2._cg_solver._M is not None, (
            "TD refactor: degrade-to-jacobi guard bypassed -- CG is "
            "running with M=None (unpreconditioned)"
        )
        assert any(
            'jacobi' in rec.message.lower() for rec in caplog.records
        )


class TestFinding4RefactorTwoLevelLossWarning:
    """Finding 4 (round 1) / Finding 6 (round 2) regression: refactor() can
    silently lose the two_level coarse-space correction with NO warning
    naming that fact -- distinct from the sibling ``tile_index_maps``-
    unavailable guard (which fires on a DIFFERENT failure mode and is
    already tested above). Covers BOTH refactor paths (DC and TD), per the
    finding's guidance.

    Unlike the ``TestRefactorRebuildsTilewise`` siblings above, the first
    two (warning) fixtures below deliberately do NOT clear
    ``ctx.topology.tile_index_maps`` after load() -- tile_index_maps stays
    available, so the resolved preconditioner is a WORKING 'block_jacobi'
    (not a 'jacobi' downgrade); the only symptom is the silently-lost
    coarse space this finding's fix makes loud.

    Finding 6 (round 2) corrected the warning's semantics: the round-1
    guard fired whenever the resolved matvec mode was not 'tilewise',
    which is a FALSE premise -- an explicit 'two_level' preconditioner
    setting rebuilds a working coarse space in 'assembled' matvec mode
    too (``_augment_with_coarse_space``'s ``build_coarse_space`` call uses
    ``self._linear_op.matmat`` regardless of matvec mode). The corrected
    condition compares what the preconditioner setting WOULD resolve to at
    the INTENDED matvec mode against what it ACTUALLY resolves to at the
    matvec mode this refactor ended up with -- warn only when that is a
    genuine regression. The two "no-warning" tests below cover the cases
    the round-1 guard got wrong (spurious firing); the two "warns" tests
    above cover the genuine-loss case (the workers-not-attached 'auto'
    case still warns -- unchanged from round 1).
    """

    def test_dc_refactor_without_workers_warns_coarse_space_lost(
        self, tmp_path, caplog,
    ):
        import logging
        from distributed.result import DistributedSolverContext

        model = _build_two_tile_distributed_model(package_cap_edges=[])
        model.settings.update({
            'interface_solver': 'cg',
            'interface_matvec_mode': 'tilewise',
            # Deliberately NOT setting interface_preconditioner: exercise
            # the Stage 3 default ('auto'), which resolves to 'two_level'
            # while workers are attached (the original factor() below).
            'interface_cg_rtol': 1e-12,
        })
        from distributed.solver import DistributedDDMSolver
        solver = DistributedDDMSolver(model)
        ctx = solver.prepare()
        assert ctx._cg_solver.preconditioner == 'two_level', (
            "fixture must resolve to two_level under the original factor() "
            "for this test (workers attached, tilewise matvec) to be "
            "meaningful"
        )
        # Finding 10 (round 2): try/finally around save/release/shutdown,
        # matching the TD twin (test_td_refactor_without_workers_warns_
        # coarse_space_lost, below).
        try:
            path = ctx.save(str(tmp_path / 'dc_ctx_f4.pkl'))
        finally:
            ctx.release()
            model.shutdown()  # workers fully torn down

        class _EmptyWorkersModel:
            workers = []
            settings = {
                'interface_solver': 'cg',
                'interface_matvec_mode': 'tilewise',
                'interface_cg_rtol': 1e-12,
            }
            coordinator_solver_config = None

        ctx2 = DistributedSolverContext.load(_EmptyWorkersModel(), path)
        # tile_index_maps is NOT cleared -- it survives save()/load()
        # independently of worker attachment, so block_jacobi builds a
        # WORKING preconditioner; only the coarse space is silently lost.
        with caplog.at_level(logging.WARNING):
            ctx2.refactor()
        assert ctx2._cg_solver.matvec_mode == 'assembled'
        assert ctx2._cg_solver.preconditioner == 'block_jacobi', (
            f"expected a working block_jacobi base (tile_index_maps IS "
            f"available), got {ctx2._cg_solver.preconditioner!r}"
        )
        assert any(
            'two_level' in rec.message.lower()
            and 'coarse space is lost' in rec.message.lower()
            for rec in caplog.records
        ), (
            "expected a WARNING naming the silently-lost two_level coarse "
            f"space; got records: {[r.message for r in caplog.records]}"
        )

    def test_td_refactor_without_workers_warns_coarse_space_lost(
        self, tmp_path, caplog,
    ):
        import logging
        from distributed.result import DistributedTransientContext

        model = _build_two_tile_distributed_model(package_cap_edges=[])
        model.settings.update({
            'interface_solver': 'cg',
            'interface_matvec_mode': 'tilewise',
            'interface_cg_rtol': 1e-12,
        })
        from distributed.solver import DistributedDDMSolver
        solver = DistributedDDMSolver(model)
        dc_ctx = solver.prepare()
        trans_ctx = solver.prepare_transient(dt=1e-10, method='be')
        assert trans_ctx._cg_solver.preconditioner == 'two_level'
        try:
            td_path = trans_ctx.save(str(tmp_path / 'td_ctx_f4.pkl'))
        finally:
            trans_ctx.release()
            dc_ctx.release()
            model.shutdown()

        class _EmptyWorkersModel:
            workers = []
            settings = {
                'interface_solver': 'cg',
                'interface_matvec_mode': 'tilewise',
                'interface_cg_rtol': 1e-12,
            }
            coordinator_solver_config = None

        trans_ctx2 = DistributedTransientContext.load(_EmptyWorkersModel(), td_path)
        with caplog.at_level(logging.WARNING):
            trans_ctx2.refactor()
        assert trans_ctx2._cg_solver.matvec_mode == 'assembled'
        assert trans_ctx2._cg_solver.preconditioner == 'block_jacobi'
        assert any(
            'two_level' in rec.message.lower()
            and 'coarse space is lost' in rec.message.lower()
            for rec in caplog.records
        ), (
            "expected a WARNING naming the silently-lost two_level coarse "
            f"space (TD refactor path); got records: "
            f"{[r.message for r in caplog.records]}"
        )

    def test_explicit_two_level_with_assembled_matvec_rebuilds_no_warning(
        self, tmp_path, caplog,
    ):
        """Finding 6 (round 2) no-warning case: an EXPLICIT
        interface_preconditioner='two_level' setting with
        interface_matvec_mode='assembled' rebuilds a WORKING coarse space
        on refactor() too -- preconditioner='two_level' does not require
        'tilewise' matvec (_augment_with_coarse_space's build_coarse_space
        call uses self._linear_op.matmat regardless of mode, and
        resolve_preconditioner() honours an explicit 'two_level' setting
        verbatim in both modes). No workers need be attached for this
        (matvec_mode is explicitly 'assembled', never auto-resolved away
        from 'tilewise'), so nothing is EVER lost here -- must NOT emit the
        coarse-space-lost warning the round-1 guard spuriously fired here.
        """
        import logging
        from distributed.result import DistributedSolverContext

        model = _build_two_tile_distributed_model(package_cap_edges=[])
        model.settings.update({
            'interface_solver': 'cg',
            'interface_matvec_mode': 'assembled',
            'interface_preconditioner': 'two_level',
            'interface_cg_rtol': 1e-12,
        })
        from distributed.solver import DistributedDDMSolver
        solver = DistributedDDMSolver(model)
        ctx = solver.prepare()
        assert ctx._cg_solver.preconditioner == 'two_level', (
            "fixture must resolve to two_level under the original factor() "
            "(explicit setting, assembled matvec) for this test to be "
            "meaningful"
        )
        try:
            path = ctx.save(str(tmp_path / 'dc_ctx_f6_no_warn1.pkl'))
        finally:
            ctx.release()
            model.shutdown()  # workers fully torn down; irrelevant here --
            # matvec_mode is explicitly 'assembled', never auto-resolved.

        class _EmptyWorkersModel:
            workers = []
            settings = {
                'interface_solver': 'cg',
                'interface_matvec_mode': 'assembled',
                'interface_preconditioner': 'two_level',
                'interface_cg_rtol': 1e-12,
            }
            coordinator_solver_config = None

        ctx2 = DistributedSolverContext.load(_EmptyWorkersModel(), path)
        with caplog.at_level(logging.WARNING):
            ctx2.refactor()
        assert ctx2._cg_solver.matvec_mode == 'assembled'
        assert ctx2._cg_solver.preconditioner == 'two_level', (
            "explicit two_level must rebuild successfully in assembled "
            f"matvec mode; got {ctx2._cg_solver.preconditioner!r}"
        )
        assert ctx2._cg_solver._coarse is not None
        assert not any(
            'coarse space is lost' in rec.message.lower()
            for rec in caplog.records
        ), (
            "must NOT warn 'coarse space is lost' -- explicit two_level "
            "rebuilds a working coarse space in assembled matvec mode too; "
            f"got records: {[r.message for r in caplog.records]}"
        )

    def test_auto_preconditioner_with_explicit_assembled_matvec_no_warning(
        self, tmp_path, caplog,
    ):
        """Finding 6 (round 2) no-warning case: the default 'auto'
        preconditioner with an EXPLICITLY 'assembled' interface_matvec_mode
        resolves to plain 'block_jacobi' at the ORIGINAL factor() already
        (never 'two_level' to begin with, since matvec never resolves to
        'tilewise' -- resolve_preconditioner('auto', 'cg', 'assembled') ==
        'block_jacobi') -- refactor() resolves the SAME way. Nothing is
        lost by this refactor (there was never a coarse space to lose), so
        it must NOT warn -- the round-1 guard fired here spuriously
        (matvec_mode != 'tilewise' was its whole condition).
        """
        import logging
        from distributed.result import DistributedSolverContext

        model = _build_two_tile_distributed_model(package_cap_edges=[])
        model.settings.update({
            'interface_solver': 'cg',
            'interface_matvec_mode': 'assembled',
            # Deliberately NOT setting interface_preconditioner: exercise
            # the Stage 3 'auto' default, which resolves to 'block_jacobi'
            # here (matvec never resolves to 'tilewise').
            'interface_cg_rtol': 1e-12,
        })
        from distributed.solver import DistributedDDMSolver
        solver = DistributedDDMSolver(model)
        ctx = solver.prepare()
        assert ctx._cg_solver.preconditioner == 'block_jacobi', (
            "fixture must resolve to block_jacobi (not two_level) under "
            "the original factor() for this test to be meaningful -- got "
            f"{ctx._cg_solver.preconditioner!r}"
        )
        try:
            path = ctx.save(str(tmp_path / 'dc_ctx_f6_no_warn2.pkl'))
        finally:
            ctx.release()
            model.shutdown()

        class _EmptyWorkersModel:
            workers = []
            settings = {
                'interface_solver': 'cg',
                'interface_matvec_mode': 'assembled',
                'interface_cg_rtol': 1e-12,
            }
            coordinator_solver_config = None

        ctx2 = DistributedSolverContext.load(_EmptyWorkersModel(), path)
        with caplog.at_level(logging.WARNING):
            ctx2.refactor()
        assert ctx2._cg_solver.matvec_mode == 'assembled'
        assert ctx2._cg_solver.preconditioner == 'block_jacobi'
        assert not any(
            'coarse space is lost' in rec.message.lower()
            for rec in caplog.records
        ), (
            "must NOT warn 'coarse space is lost' -- this refactor resolves "
            "to the SAME block_jacobi the original factor() already used; "
            "nothing was ever two_level here; got records: "
            f"{[r.message for r in caplog.records]}"
        )


# ---------------------------------------------------------------------------
# interface_drop_s_global (item 3)
# ---------------------------------------------------------------------------


class TestInterfaceDropSGlobal:

    @staticmethod
    def _build_summaries_model(tmp_path):
        """Build a small distributed model with island_detection_mode
        resolved to 'summaries' (drop_s_global's precondition), via the
        component-summaries integration test's model-building pattern."""
        model = _build_two_tile_distributed_model(package_cap_edges=[])
        # Force summaries mode with trivial (no-op) per-tile summaries: no
        # islands in this fixture, so an empty summaries list is sufficient
        # for detect_interface_islands_from_summaries to run cleanly.
        model.island_detection_mode = 'summaries'
        model.component_summaries = []
        return model

    def test_never_assembles_s_global_and_matches_direct(self, tmp_path):
        from distributed.solver import DistributedDDMSolver

        model_direct = self._build_summaries_model(tmp_path)
        model_direct.island_detection_mode = 'schur_bfs'  # irrelevant for direct
        solver_direct = DistributedDDMSolver(model_direct)
        ctx_direct = solver_direct.prepare()
        try:
            result_direct = solver_direct.solve_dc(ctx_direct).flatten()
        finally:
            ctx_direct.release()
            model_direct.shutdown()

        model = self._build_summaries_model(tmp_path)
        model.settings.update({
            'interface_solver': 'cg',
            'interface_matvec_mode': 'tilewise',
            'interface_preconditioner': 'block_jacobi',
            'interface_cg_rtol': 1e-12,
            'interface_drop_s_global': True,
        })
        solver = DistributedDDMSolver(model)
        ctx = solver.prepare()
        try:
            assert ctx._S_global is None, (
                "interface_drop_s_global=True must never assemble S_global"
            )
            assert getattr(ctx, '_s_global_dropped', False) is True
            assert ctx._cg_solver.matvec_mode == 'tilewise'

            result = solver.solve_dc(ctx).flatten()
            common = set(result) & set(result_direct)
            assert common
            for node in common:
                assert abs(result[node] - result_direct[node]) < 1e-8

            with pytest.raises(RuntimeError, match='never assemble'):
                ctx.save(str(tmp_path / 'should_fail.pkl'))
        finally:
            ctx.release()
            model.shutdown()

    def test_island_penalty_path_matches_direct_oracle(self, tmp_path):
        """Finding 11: every other test in this class uses
        component_summaries=[] on a fixture with NO islands (empty
        island_nodes), so the never-assemble path's own hand-rolled
        island-penalty stamping -- the direct
        ``rhs_dirichlet[idx] += penalty_conductance * model.vdd`` at
        _factor_dc_context_no_s_global's RHS step, and the island diagonal
        folded into S_extra via _build_s_extra_direct -- was never
        exercised with a genuinely non-empty island set. Use the 'iso'
        island fixture (shared with TestS1RefactorModeCorrectIslands) to
        close that gap: a defect in either penalty stamp (e.g. the S4-style
        preconditioner omissions found elsewhere in this review round, or a
        future edit that changes the penalty conductance in pgmath but not
        here) would silently produce a wrong voltage on 'iso' with no test
        able to catch it."""
        from distributed.solver import DistributedDDMSolver

        model_direct = _build_three_tile_island_model(cap_c_fF=0.0)
        model_direct.island_detection_mode = 'schur_bfs'  # irrelevant for direct
        solver_direct = DistributedDDMSolver(model_direct)
        ctx_direct = solver_direct.prepare()
        try:
            assert ctx_direct.topology.island_nodes == {'iso'}, (
                "fixture sanity: 'iso' has no resistive path to a pad"
            )
            result_direct = solver_direct.solve_dc(ctx_direct).flatten()
        finally:
            ctx_direct.release()
            model_direct.shutdown()

        model = _build_three_tile_island_model(cap_c_fF=0.0)
        model.island_detection_mode = 'summaries'
        model.component_summaries = []  # 'iso' is an unreached singleton
        model.settings.update({
            'interface_solver': 'cg',
            'interface_matvec_mode': 'tilewise',
            'interface_preconditioner': 'block_jacobi',
            'interface_cg_rtol': 1e-10,
            'interface_drop_s_global': True,
        })
        solver = DistributedDDMSolver(model)
        ctx = solver.prepare()
        try:
            assert ctx._S_global is None, (
                "interface_drop_s_global=True must never assemble S_global"
            )
            assert getattr(ctx, '_s_global_dropped', False) is True

            result = solver.solve_dc(ctx).flatten()
            common = set(result) & set(result_direct)
            assert common, "fixture produced no comparable nodes"
            assert 'iso' in common, "the fixture's island node must be comparable"
            for node in common:
                assert abs(result[node] - result_direct[node]) < 1e-6, (
                    f"node {node}: never-assemble={result[node]!r} vs "
                    f"direct oracle={result_direct[node]!r}"
                )
            # 'iso' specifically must be pinned near Vdd by the island
            # penalty (not left at ~0, an unforced diagonal-only system --
            # the classic symptom of a missing/mis-scaled RHS or diagonal
            # penalty term).
            assert abs(result['iso'] - model.vdd) < 1e-3, (
                f"island node 'iso'={result['iso']!r}, expected ~"
                f"{model.vdd!r} V (pinned by the island penalty)."
            )
        finally:
            ctx.release()
            model.shutdown()

    def test_fp32_frees_fp64_originals_and_block_jacobi_still_correct(self, tmp_path):
        """Finding 3: with interface_drop_s_global=True (S_global is None),
        block_jacobi's Path 2 fallback (_build_block_jacobi reading
        self.tile_schur_complements in fp64) is the exact configuration
        where the fp64-read-before-fp32-free ordering matters.  Verifies:

        1. The preconditioner is built correctly (matches the fp64-only
           oracle's result within CG tolerance) -- proving __init__'s
           reordered build (preconditioner before linear op) reads the
           fp64 blocks before they are converted/freed.
        2. After construction, ``cg_solver.tile_schur_complements`` holds
           only fp32 arrays -- proving the fp64 originals were actually
           freed in place, not retained alongside the fp32 copies.
        """
        from distributed.solver import DistributedDDMSolver

        model_direct = self._build_summaries_model(tmp_path)
        model_direct.island_detection_mode = 'schur_bfs'
        solver_direct = DistributedDDMSolver(model_direct)
        ctx_direct = solver_direct.prepare()
        try:
            result_direct = solver_direct.solve_dc(ctx_direct).flatten()
        finally:
            ctx_direct.release()
            model_direct.shutdown()

        model = self._build_summaries_model(tmp_path)
        model.settings.update({
            'interface_solver': 'cg',
            'interface_matvec_mode': 'tilewise',
            'interface_preconditioner': 'block_jacobi',
            'interface_cg_rtol': 1e-6,
            'interface_drop_s_global': True,
            'interface_matvec_dtype': 'float32',
        })
        solver = DistributedDDMSolver(model)
        ctx = solver.prepare()
        try:
            assert ctx._S_global is None
            cg_solver = ctx._cg_solver
            assert cg_solver.matvec_mode == 'tilewise'
            assert cg_solver.matvec_dtype == np.dtype(np.float32)
            assert cg_solver._M is not None, (
                "block_jacobi preconditioner must still be built (from fp64 "
                "blocks, before they are converted/freed)"
            )
            assert cg_solver.tile_schur_complements, "sanity: tiles present"
            for S_i in cg_solver.tile_schur_complements.values():
                assert np.asarray(S_i).dtype == np.float32, (
                    "fp64 originals must be freed -- tile_schur_complements "
                    "should hold only fp32 arrays after construction"
                )

            result = solver.solve_dc(ctx).flatten()
            common = set(result) & set(result_direct)
            assert common
            for node in common:
                assert abs(result[node] - result_direct[node]) < 1e-4, (
                    "fp32 tilewise block_jacobi result diverges from the "
                    "fp64 direct oracle beyond the fp32 matvec floor"
                )
        finally:
            ctx.release()
            model.shutdown()

    def test_falls_back_with_warning_when_preconditions_unmet(self, tmp_path, caplog):
        import logging
        from distributed.solver import DistributedDDMSolver

        model = self._build_summaries_model(tmp_path)
        model.settings.update({
            'interface_solver': 'auto',  # precondition requires explicit 'cg'
            'interface_drop_s_global': True,
        })
        solver = DistributedDDMSolver(model)
        with caplog.at_level(logging.WARNING):
            ctx = solver.prepare()
        try:
            assert ctx._S_global is not None, (
                "must fall back to the normal assembling path"
            )
            assert any(
                'preconditions are not met' in rec.message for rec in caplog.records
            )
        finally:
            ctx.release()
            model.shutdown()

    def test_transient_warns_that_drop_s_global_is_dc_only(self, tmp_path, caplog):
        """Finding 2: interface_drop_s_global=True is DC-only in Stage 2 --
        prepare_transient() must never silently ignore it; it must assemble
        S_global normally AND emit a clear WARNING explaining the scope
        reduction."""
        import logging
        from distributed.solver import DistributedDDMSolver

        model = self._build_summaries_model(tmp_path)
        model.settings.update({
            'interface_solver': 'cg',
            'interface_matvec_mode': 'tilewise',
            'interface_preconditioner': 'block_jacobi',
            'interface_cg_rtol': 1e-12,
            'interface_drop_s_global': True,
        })
        solver = DistributedDDMSolver(model)
        with caplog.at_level(logging.WARNING):
            trans_ctx = solver.prepare_transient(dt=1e-10, method='be')
        try:
            assert trans_ctx._S_global is not None, (
                "transient factor must assemble S_global normally even "
                "when interface_drop_s_global=True (DC-only in Stage 2)"
            )
            assert any(
                'interface_drop_s_global is DC-only' in rec.message
                for rec in caplog.records
            ), "expected a clear WARNING that transient never-assemble is not implemented"
        finally:
            trans_ctx.release()
            model.shutdown()


# ---------------------------------------------------------------------------
# Settings propagation (matvec_threads, interface_matvec_dtype,
# interface_drop_s_global reach the InterfaceCGSolver instance)
# ---------------------------------------------------------------------------


class TestStage2SettingsPropagation:

    @staticmethod
    def _build_model():
        return _build_two_tile_distributed_model(package_cap_edges=[])

    def test_matvec_threads_setting_reaches_cg_solver(self):
        from distributed.solver import DistributedDDMSolver

        model = self._build_model()
        model.settings.update({
            'interface_solver': 'cg',
            'interface_matvec_mode': 'tilewise',
            'matvec_threads': 3,
        })
        solver = DistributedDDMSolver(model)
        ctx = solver.prepare()
        try:
            assert ctx._cg_solver.matvec_threads == 3
        finally:
            ctx.release()
            model.shutdown()

    def test_interface_matvec_dtype_setting_reaches_cg_solver(self):
        from distributed.solver import DistributedDDMSolver

        model = self._build_model()
        model.settings.update({
            'interface_solver': 'cg',
            'interface_matvec_mode': 'tilewise',
            'interface_matvec_dtype': 'float32',
            'interface_cg_rtol': 1e-6,
        })
        solver = DistributedDDMSolver(model)
        ctx = solver.prepare()
        try:
            assert ctx._cg_solver.matvec_dtype == np.dtype(np.float32)
        finally:
            ctx.release()
            model.shutdown()

    def test_default_matvec_mode_is_auto_tilewise(self):
        """Item 8: with no interface_matvec_mode set at all, cg resolves to
        tilewise (tile blocks are available in the default non-streaming path)."""
        from distributed.solver import DistributedDDMSolver

        model = self._build_model()
        model.settings.update({'interface_solver': 'cg'})
        solver = DistributedDDMSolver(model)
        ctx = solver.prepare()
        try:
            assert ctx._cg_solver.matvec_mode == 'tilewise'
        finally:
            ctx.release()
            model.shutdown()


# ---------------------------------------------------------------------------
# S1: refactor() tilewise re-gather must use the CONTEXT's own mode-correct
# island set, not the shared topology.removed_interface_nodes field (which
# is set once, by whichever mode -- DC or TD -- first creates the topology,
# and never updated after that).
# ---------------------------------------------------------------------------


def _build_three_tile_island_model(cap_c_fF):
    """Two-tile fixture (see ``_build_two_tile_distributed_model``) plus a
    THIRD tile holding a lone interface node ``'iso'`` with NO resistive
    path to any pad -- only a package CAPACITOR edge to ``'shared'`` (which
    itself resistively reaches ``'pad'``).

    DC-mode island detection (extra_edges = resistive package_edges only):
    ``'iso'`` has zero resistive connectivity to a Dirichlet node -> island.

    TD-mode island detection (extra_edges = combined_edges = resistive +
    C_coeff-weighted package cap edges): the cap edge ``('iso', 'shared',
    C_coeff * cap_c_fF)`` bridges ``'iso'`` into the live (pad-reaching)
    component -> NOT an island.

    This is the model-level analogue of
    ``test_island_summaries.TestOracleEquivalenceVsSchurBFS.
    test_cap_coupled_only_component_dc_vs_td``, used here to exercise the
    ORCHESTRATION (S1) rather than the underlying island-detection algorithm
    (already covered there).
    """
    from distributed.backend import LocalBackend
    from distributed.model import DistributedPowerGridModel
    from distributed.parser import PackageData, PowerGridMetaData, TileConfig
    from distributed.tile_worker import TileWorker, TileData

    tile_a_data = TileData(
        tile_id=(0, 0),
        resistive_edges=[('a1', 'shared', 1.0), ('shared', 'pad', 2.0)],
        all_nodes={'a1', 'shared', 'pad'},
        boundary_nodes={'shared', 'pad'},
        current_injections={'a1': 0.5},
        capacitive_edges=[('a1', '0', 10.0)],
    )
    tile_b_data = TileData(
        tile_id=(0, 1),
        resistive_edges=[('shared', 'b1', 3.0), ('b1', '0', 1.0)],
        all_nodes={'shared', 'b1'},
        boundary_nodes={'shared'},
        current_injections={'b1': 0.3},
        capacitive_edges=[('b1', '0', 5.0)],
    )
    tile_c_data = TileData(
        tile_id=(0, 2),
        resistive_edges=[('iso', '0', 1.0)],  # grounded, no path to any pad
        all_nodes={'iso'},
        boundary_nodes={'iso'},
        current_injections={},
        capacitive_edges=[],
    )

    interface_nodes = {'shared', 'pad', 'iso'}

    be = LocalBackend()
    be.initialize()

    worker_a = TileWorker()
    worker_a.setup_from_tile_data(tile_a_data, interface_nodes)
    worker_b = TileWorker()
    worker_b.setup_from_tile_data(tile_b_data, interface_nodes)
    worker_c = TileWorker()
    worker_c.setup_from_tile_data(tile_c_data, interface_nodes)

    workers = [worker_a, worker_b, worker_c]

    pkg_data = PackageData(
        vsrc_dict={'V1': {'node+': 'pad', 'node-': '0', 'net': 'VDD', 'value': 1.0}},
        package_edges=[('pad', 'shared', 10.0)],  # resistive only -- no 'iso'
        pad_nodes={'pad'},
        tap_nodes=set(),
        die_attachment_nodes=set(),
        vdd=1.0,
        net_name='VDD',
        package_cap_edges=[('iso', 'shared', cap_c_fF)],
    )

    tile_configs = [
        TileConfig(tile_id=(0, 0), ckt_path='', nd_path=None,
                   instance_path=None, net_filter=None),
        TileConfig(tile_id=(0, 1), ckt_path='', nd_path=None,
                   instance_path=None, net_filter=None),
        TileConfig(tile_id=(0, 2), ckt_path='', nd_path=None,
                   instance_path=None, net_filter=None),
    ]

    metadata = PowerGridMetaData(
        tile_grid=(1, 3),
        parameters={},
        tile_configs=tile_configs,
        package_data=pkg_data,
        net_name='VDD',
        vdd=1.0,
    )

    tile_boundary_nodes = {
        (0, 0): ['pad', 'shared'],
        (0, 1): ['shared'],
        (0, 2): ['iso'],
    }
    tile_interior_counts = {
        (0, 0): worker_a.n_interior,
        (0, 1): worker_b.n_interior,
        (0, 2): worker_c.n_interior,
    }

    return DistributedPowerGridModel(
        backend=be,
        workers=workers,
        interface_nodes=interface_nodes,
        tile_boundary_nodes=tile_boundary_nodes,
        tile_interior_counts=tile_interior_counts,
        package_data=pkg_data,
        metadata=metadata,
    )


class TestS1RefactorModeCorrectIslands:
    """Regression test for the S1 finding: refactor()'s tilewise re-gather
    must stamp S_extra's island penalty from the CONTEXT's own mode-correct
    island set, not ``topology.removed_interface_nodes`` (set once by
    whichever mode -- DC or TD -- first creates the shared topology).
    """

    def test_refactor_td_context_uses_td_islands_not_dc(self, tmp_path):
        from distributed.solver import DistributedDDMSolver
        from distributed.result import DistributedTransientContext

        model = _build_three_tile_island_model(cap_c_fF=20.0)
        model.settings.update({
            'interface_solver': 'cg',
            'interface_matvec_mode': 'tilewise',
            'interface_preconditioner': 'block_jacobi',
            'interface_cg_rtol': 1e-12,
            'interface_cg_atol': 1e-16,
        })
        solver = DistributedDDMSolver(model)

        # prepare() DC FIRST -- this is what creates the shared topology, so
        # topology.removed_interface_nodes (the buggy source) gets stamped
        # with DC's island set ({'iso'}).  Pre-fix, using that shared field
        # for the TD refactor would keep penalizing 'iso' in TD too (wrong:
        # the cap bridge makes it live) while a TD-only island (none exist
        # in this fixture, but the code path is symmetric) would go
        # unpenalized.
        dc_ctx = solver.prepare()
        assert dc_ctx.topology.island_nodes == {'iso'}, (
            "fixture sanity: 'iso' has no resistive path to a pad in DC mode"
        )

        trans_ctx = solver.prepare_transient(dt=1e-10, method='be')
        assert trans_ctx.topology.island_nodes_td == set(), (
            "fixture sanity: the cap bridge makes 'iso' live in TD mode"
        )
        # The shared, first-mode-wins field must still hold DC's value --
        # this is the confusable field the bug used.
        assert trans_ctx.topology.removed_interface_nodes == {'iso'}

        try:
            smoothed = solver.preprocess_sources(
                time_step=1e-10, t_start=0.0, t_end=2e-9, smooth=False,
                pkl_dir=str(tmp_path),
            )
            v_fresh = solver.solve_transient(
                trans_ctx, dc_context=dc_ctx, t_end=2e-9,
                smoothed_sources=smoothed,
            ).as_flat()

            # Save + release + load + refactor the TD context (workers stay
            # attached throughout) -- exercises the tilewise re-gather's
            # island-set selection.  save() must run before release() (which
            # clears S_global); load() restores S_global + topology but
            # leaves the context unfactored until refactor() rebuilds the
            # coordinator solve callable.
            td_path = trans_ctx.save(str(tmp_path / 'td_ctx.pkl'))
            trans_ctx.release()
            trans_ctx = DistributedTransientContext.load(model, td_path)
            trans_ctx.refactor()
            assert trans_ctx._cg_solver is not None
            assert trans_ctx._cg_solver.matvec_mode == 'tilewise'

            iso_idx = trans_ctx.interface_node_to_idx['iso']
            s_extra_diag_iso = trans_ctx._cg_solver.S_extra.tocsr()[iso_idx, iso_idx]
            assert abs(s_extra_diag_iso) < 1.0, (
                f"refactor() penalized 'iso' (S_extra diag={s_extra_diag_iso!r}) "
                f"using the stale DC-mode island set -- it must use TD's own "
                f"island_nodes_td (empty: 'iso' is cap-bridged live in TD)."
            )

            v_refactored = solver.solve_transient(
                trans_ctx, dc_context=dc_ctx, t_end=2e-9,
                smoothed_sources=smoothed,
            ).as_flat()
        finally:
            trans_ctx.release()
            dc_ctx.release()
            model.shutdown()

        # as_flat() maps node -> (max_drop, time_of_max).
        common = set(v_fresh) & set(v_refactored)
        assert common, "fixture produced no comparable nodes"
        assert 'iso' in common, "the fixture's island node must be comparable"
        for node in common:
            drop_fresh, _t_fresh = v_fresh[node]
            drop_refactored, _t_refactored = v_refactored[node]
            assert abs(drop_fresh - drop_refactored) < 1e-8, (
                f"node {node}: fresh-factor drop={drop_fresh!r} vs "
                f"post-refactor drop={drop_refactored!r}"
            )


# ---------------------------------------------------------------------------
# S15: forced-CG equivalence matrix for the interface_matvec_mode default
# flip (assembled -> auto/tilewise).
#
# tests/validation/test_equivalence.py's SOLVER_VARIANTS mechanism cannot
# host this matrix: solver_kw entries there are splatted directly as
# **kwargs into solve_dc/solve_quasi_static/solve_transient/
# analyze_adjoint_static calls (see e.g. TestQSSampledPkl), so only flags
# those functions actually accept (like use_step_columns) fit.
# interface_solver/interface_matvec_mode/interface_matvec_dtype/
# interface_drop_s_global are MODEL SETTINGS (model.settings[...]), not
# solve-call kwargs -- adding them to SOLVER_VARIANTS would raise
# TypeError on every call site that doesn't expect them.  This mirrors the
# Stage 1 report's documented reason CG-mode coverage lives here instead
# of the equivalence suite; see the cross-reference comment placed in
# test_equivalence.py's module docstring.
#
# Matrix: {assembled, tilewise(auto)} x {fp64} x {with S_global,
# never-assemble} vs the direct solver, pinned rtol=1e-12 on the standard
# 2-tile fixture, plus one fp32 case at its documented looser tolerance
# (fp32 tilewise matvec has a ~1e-7 relative residual floor -- see
# FP32_MATVEC_MIN_RTOL in interface_iterative.py).
# ---------------------------------------------------------------------------


class TestS15ForcedCGEquivalenceMatrix:

    @staticmethod
    def _build_model():
        return _build_two_tile_distributed_model(package_cap_edges=[])

    @staticmethod
    def _build_summaries_model():
        """island_detection_mode='summaries' -- required precondition for
        the never-assemble (interface_drop_s_global) path."""
        model = _build_two_tile_distributed_model(package_cap_edges=[])
        model.island_detection_mode = 'summaries'
        model.component_summaries = []
        return model

    def _run(self, build_model, settings):
        from distributed.solver import DistributedDDMSolver

        model = build_model()
        model.settings.update(settings)
        solver = DistributedDDMSolver(model)
        ctx = solver.prepare()
        try:
            return solver.solve_dc(ctx).flatten()
        finally:
            ctx.release()
            model.shutdown()

    def _direct_reference(self):
        return self._run(self._build_model, {
            'interface_solver': 'direct',
            'interface_matvec_mode': 'assembled',
        })

    @pytest.mark.parametrize(
        "case_id,build_model,settings,rtol",
        [
            pytest.param(
                "assembled_fp64_with_s_global", None,
                {
                    'interface_solver': 'cg',
                    'interface_matvec_mode': 'assembled',
                    'interface_cg_rtol': 1e-12,
                    'interface_cg_atol': 1e-16,
                },
                1e-12, id="assembled_fp64_with_s_global",
            ),
            pytest.param(
                "tilewise_fp64_with_s_global", None,
                {
                    'interface_solver': 'cg',
                    'interface_matvec_mode': 'tilewise',
                    'interface_cg_rtol': 1e-12,
                    'interface_cg_atol': 1e-16,
                },
                1e-12, id="tilewise_fp64_with_s_global",
            ),
            pytest.param(
                "auto_fp64_with_s_global_resolves_tilewise", None,
                {
                    'interface_solver': 'cg',
                    'interface_matvec_mode': 'auto',
                    'interface_cg_rtol': 1e-12,
                    'interface_cg_atol': 1e-16,
                },
                1e-12, id="auto_fp64_with_s_global_resolves_tilewise",
            ),
            pytest.param(
                "tilewise_fp64_never_assemble", 'summaries',
                {
                    'interface_solver': 'cg',
                    'interface_matvec_mode': 'tilewise',
                    'interface_cg_rtol': 1e-12,
                    'interface_cg_atol': 1e-16,
                    'interface_drop_s_global': True,
                },
                1e-12, id="tilewise_fp64_never_assemble",
            ),
            # Stage 3: explicit preconditioner='two_level' rows.
            pytest.param(
                "two_level_tilewise_with_s_global", None,
                {
                    'interface_solver': 'cg',
                    'interface_matvec_mode': 'tilewise',
                    'interface_preconditioner': 'two_level',
                    'interface_cg_rtol': 1e-12,
                    'interface_cg_atol': 1e-16,
                },
                1e-12, id="two_level_tilewise_with_s_global",
            ),
            pytest.param(
                "two_level_assembled_with_s_global", None,
                {
                    'interface_solver': 'cg',
                    'interface_matvec_mode': 'assembled',
                    'interface_preconditioner': 'two_level',
                    'interface_cg_rtol': 1e-12,
                    'interface_cg_atol': 1e-16,
                },
                1e-12, id="two_level_assembled_with_s_global",
            ),
            pytest.param(
                "two_level_tilewise_never_assemble", 'summaries',
                {
                    'interface_solver': 'cg',
                    'interface_matvec_mode': 'tilewise',
                    'interface_preconditioner': 'two_level',
                    'interface_cg_rtol': 1e-12,
                    'interface_cg_atol': 1e-16,
                    'interface_drop_s_global': True,
                },
                1e-12, id="two_level_tilewise_never_assemble",
            ),
        ],
    )
    def test_cg_matches_direct_at_pinned_rtol(
        self, case_id, build_model, settings, rtol,
    ):
        v_direct = self._direct_reference()
        builder = (
            self._build_summaries_model if build_model == 'summaries'
            else self._build_model
        )
        v_cg = self._run(builder, settings)

        common = set(v_direct) & set(v_cg)
        assert common, f"[{case_id}] fixture produced no comparable nodes"
        for node in common:
            diff = abs(v_direct[node] - v_cg[node])
            assert diff <= 1e-8, (
                f"[{case_id}] node {node}: direct={v_direct[node]!r} vs "
                f"cg={v_cg[node]!r} diff={diff:.3e} (matvec/solve rtol "
                f"pinned at {rtol:.0e}, but end-to-end voltage tolerance "
                f"is looser -- direct LU vs converged CG residual)"
            )

    def test_cg_tilewise_fp32_matches_direct_at_documented_tolerance(self):
        """fp32 case, at its documented looser tolerance (item: matvec_dtype
        rtol >= FP32_MATVEC_MIN_RTOL, ~1e-7 relative residual floor)."""
        v_direct = self._direct_reference()
        v_cg = self._run(self._build_model, {
            'interface_solver': 'cg',
            'interface_matvec_mode': 'tilewise',
            'interface_matvec_dtype': 'float32',
            'interface_cg_rtol': 1e-6,
            'interface_cg_atol': 1e-12,
        })

        common = set(v_direct) & set(v_cg)
        assert common, "fixture produced no comparable nodes"
        for node in common:
            diff = abs(v_direct[node] - v_cg[node])
            assert diff <= 1e-4, (
                f"node {node}: direct={v_direct[node]!r} vs fp32 cg="
                f"{v_cg[node]!r} diff={diff:.3e} (fp32 tilewise matvec has "
                f"a ~1e-7 relative residual floor; looser tolerance than "
                f"the fp64 cases is expected and documented)"
            )


# ---------------------------------------------------------------------------
# code-quality review round 1: verbose "Interface CG solver" / "Transient
# interface CG solver" INFO logs (result_factorization.py) must print the
# preconditioner that was ACTUALLY built (cg_solver.preconditioner_label),
# not the unresolved 'auto' setting.  Stage 3 flipped the default
# ``interface_preconditioner`` to 'auto' but these two log lines still
# formatted the raw pre-resolution setting -- with the default left
# untouched, a --verbose run logged "precond=auto" even though
# build_interface_solver's ONE resolution point (resolve_preconditioner())
# had already turned it into 'two_level(...)' (or 'block_jacobi'/'jacobi' on
# a degrade).  backend_info (used by solver_stats/tests elsewhere) was
# already correct; only these two bare logger.info() calls were stale.
# ---------------------------------------------------------------------------


class TestInterfaceCgVerboseLogReportsResolvedPreconditioner:
    """Regression: the verbose 'Interface CG solver' / 'Transient interface
    CG solver' log lines must never print the literal unresolved 'auto'
    setting -- they must match the resolved ``preconditioner_label`` (the
    same value already reported in solver_stats['interface']['backend_info'])."""

    @staticmethod
    def _build_model(tmp_path=None):
        model = _build_two_tile_distributed_model(package_cap_edges=[])
        if tmp_path is not None:
            for tc in model.metadata.tile_configs:
                tc.ckt_path = str(tmp_path / 'dummy.ckt')
        model.settings.update({
            'interface_solver': 'cg',
            'interface_matvec_mode': 'tilewise',
            # interface_preconditioner deliberately left UNSET so it takes
            # the Stage 3 default of 'auto' -- the exact condition under
            # which the pre-fix code printed the unresolved literal.
            'interface_cg_rtol': 1e-10,
        })
        return model

    def test_dc_factor_verbose_log_does_not_print_auto(self, caplog):
        import logging
        from distributed.solver import DistributedDDMSolver

        model = self._build_model()
        solver = DistributedDDMSolver(model)
        try:
            with caplog.at_level(
                logging.INFO, logger='distributed.result_factorization',
            ):
                ctx = solver.prepare(verbose=True)
            backend_info = ctx.timings['solver_stats']['interface']['backend_info']
            ctx.release()
        finally:
            model.shutdown()

        records = [
            r for r in caplog.records
            if r.message.startswith('Interface CG solver: mode=')
        ]
        assert records, (
            f"expected an 'Interface CG solver: mode=...' INFO log; got "
            f"{[r.message for r in caplog.records]}"
        )
        msg = records[0].message
        assert 'precond=auto' not in msg, (
            f"verbose DC factor log printed the unresolved 'auto' setting "
            f"instead of the resolved preconditioner: {msg!r}"
        )
        # Must match the SAME resolved label already reported in
        # backend_info (e.g. "CG/tilewise/precond=two_level(...)").
        resolved_suffix = backend_info.split('precond=', 1)[1]
        assert f"precond={resolved_suffix}" in msg, (
            f"log line {msg!r} does not report the resolved preconditioner "
            f"{resolved_suffix!r} from backend_info={backend_info!r}"
        )

    def test_transient_factor_verbose_log_does_not_print_auto(
        self, caplog, tmp_path,
    ):
        import logging
        from distributed.solver import DistributedDDMSolver

        model = self._build_model(tmp_path)
        solver = DistributedDDMSolver(model)
        try:
            with caplog.at_level(
                logging.INFO, logger='distributed.result_factorization',
            ):
                trans_ctx = solver.prepare_transient(
                    dt=1e-10, method='be', verbose=True,
                )
            backend_info = (
                trans_ctx.timings['solver_stats']['interface']['backend_info']
            )
            trans_ctx.release()
        finally:
            model.shutdown()

        records = [
            r for r in caplog.records
            if r.message.startswith('Transient interface CG solver: mode=')
        ]
        assert records, (
            f"expected a 'Transient interface CG solver: mode=...' INFO "
            f"log; got {[r.message for r in caplog.records]}"
        )
        msg = records[0].message
        assert 'precond=auto' not in msg, (
            f"verbose transient factor log printed the unresolved 'auto' "
            f"setting instead of the resolved preconditioner: {msg!r}"
        )
        resolved_suffix = backend_info.split('precond=', 1)[1]
        assert f"precond={resolved_suffix}" in msg, (
            f"log line {msg!r} does not report the resolved preconditioner "
            f"{resolved_suffix!r} from backend_info={backend_info!r}"
        )


# ---------------------------------------------------------------------------
# T1: transient island-penalty RHS regression.
#
# apply_island_penalty() folds penalty*vdd into rhs_dirichlet_A only, but the
# transient time loop (solver_td.py) uses rhs_dirichlet_G (BE: +1x, TR: +2x
# per the documented convention) -- rhs_dirichlet_A is never read during
# transient stepping. Pre-fix, a penalized interface-island node carries the
# 1e5 mS penalty on the A-diagonal but gets NO matching per-step forcing
# term, so it decays from Vdd toward 0 within a few steps and the reported
# IR-drop saturates at ~Vdd. The fix adds a SEPARATE island_penalty_rhs
# vector, added exactly once per step for both BE and TR (never folded into
# rhs_dirichlet_G, which TR already scales by 2 -- folding it in there would
# double-count the penalty under TR).
# ---------------------------------------------------------------------------


class TestT1TransientIslandPenaltyRHS:
    """Regression: a penalized interface-island node must stay pinned near
    Vdd throughout a transient solve (BE and TR, bulk and streaming
    assembly), not decay toward 0 V."""

    @staticmethod
    def _build_model(streaming_assembly):
        # cap_c_fF=0.0 -> the package cap edge is filtered out of
        # combined_edges (only c_fF > 0 edges count), so 'iso' has NO path
        # to any pad in EITHER DC or transient island detection -- a
        # genuine transient-mode island, exercising the buggy RHS path.
        model = _build_three_tile_island_model(cap_c_fF=0.0)
        model.settings.update({'streaming_assembly': streaming_assembly})
        return model

    @pytest.mark.parametrize('streaming_assembly', [False, True])
    @pytest.mark.parametrize('method', ['be', 'trap'])
    def test_island_node_stays_pinned_near_vdd(
        self, method, streaming_assembly, tmp_path,
    ):
        from distributed.solver import DistributedDDMSolver

        model = self._build_model(streaming_assembly)
        solver = DistributedDDMSolver(model)
        vdd = model.vdd

        dc_ctx = solver.prepare()
        # Fixture sanity: 'iso' really is an island in DC mode.
        assert 'iso' in dc_ctx.topology.island_nodes

        trans_ctx = solver.prepare_transient(dt=1e-10, method=method)
        # Fixture sanity: 'iso' really is a TRANSIENT-mode island too (no
        # cap bridge -- cap_c_fF=0.0 above), so the buggy RHS path is
        # actually exercised (a cap-bridged 'iso', as in the S1 fixture's
        # default cap_c_fF=20.0, would NOT be penalized in TD mode at all).
        assert 'iso' in trans_ctx.topology.island_nodes_td

        try:
            smoothed = solver.preprocess_sources(
                time_step=1e-10, t_start=0.0, t_end=2e-9, smooth=False,
                pkl_dir=str(tmp_path),
            )
            result = solver.solve_transient(
                trans_ctx, dc_context=dc_ctx, t_end=2e-9,
                smoothed_sources=smoothed,
                track_nodes=['iso', 'shared', 'b1'],
            )
        finally:
            trans_ctx.release()
            dc_ctx.release()
            model.shutdown()

        iso_wave = result.tracked_waveforms.get('iso')
        assert iso_wave is not None and len(iso_wave) > 1, (
            "expected a multi-step tracked waveform for the island node"
        )
        # Pre-fix: 'iso' decays from Vdd toward 0 within a few steps because
        # the per-step transient RHS never applies the island-penalty
        # forcing term (only the A-diagonal is penalized). Post-fix: 'iso'
        # stays pinned near Vdd at EVERY step -- residual error from the
        # finite 1e5 mS penalty vs. the fixture's local 1 mS
        # ('iso'-to-ground) conductance is O(1/1e5), i.e. ~1e-5 V here.
        for step_idx, v in enumerate(iso_wave):
            assert abs(vdd - v) < 1e-3, (
                f"[method={method}, streaming_assembly={streaming_assembly}] "
                f"step {step_idx}: island node 'iso' drifted to v={v!r} "
                f"(expected ~{vdd!r} V, pinned); the per-step "
                f"island-penalty RHS term is missing or wrong."
            )
        assert iso_wave[-1] > 0.9 * vdd, (
            f"[method={method}, streaming_assembly={streaming_assembly}] "
            f"island node 'iso' decayed to {iso_wave[-1]!r} V by the last "
            f"step (expected ~{vdd!r} V, pinned) -- this is the exact "
            f"pre-fix failure signature (decay toward 0 within a few steps)."
        )

    @pytest.mark.parametrize('method', ['be', 'trap'])
    def test_bulk_and_streaming_agree_including_island_node(
        self, method, tmp_path,
    ):
        """Both bulk and streaming transient assembly share the same T1 fix
        (a single island-detection block feeds both paths in
        _factor_transient_context) -- verify parity AND that 'iso' reports
        ~0 IR-drop (pinned near Vdd) in both, not just one."""
        from distributed.solver import DistributedDDMSolver

        results = {}
        for streaming_assembly in (False, True):
            model = self._build_model(streaming_assembly)
            solver = DistributedDDMSolver(model)
            dc_ctx = solver.prepare()
            trans_ctx = solver.prepare_transient(dt=1e-10, method=method)
            try:
                smoothed = solver.preprocess_sources(
                    time_step=1e-10, t_start=0.0, t_end=2e-9, smooth=False,
                    pkl_dir=str(tmp_path / f'{method}_{streaming_assembly}'),
                )
                result = solver.solve_transient(
                    trans_ctx, dc_context=dc_ctx, t_end=2e-9,
                    smoothed_sources=smoothed,
                )
                results[streaming_assembly] = result.as_flat()
            finally:
                trans_ctx.release()
                dc_ctx.release()
                model.shutdown()

        v_bulk, v_stream = results[False], results[True]
        common = set(v_bulk) & set(v_stream)
        assert common, "fixture produced no comparable nodes"
        assert 'iso' in common, "the fixture's island node must be comparable"
        for node in common:
            drop_bulk, _t_b = v_bulk[node]
            drop_stream, _t_s = v_stream[node]
            assert abs(drop_bulk - drop_stream) < 1e-6, (
                f"[method={method}] node {node}: bulk drop={drop_bulk!r} "
                f"vs streaming drop={drop_stream!r}"
            )

        # 'iso' must show ~0 IR-drop (pinned near Vdd) in BOTH paths. Pre-fix
        # both bulk and streaming shared the SAME (buggy) time loop, so both
        # would show 'iso' collapsing toward ~Vdd drop (~0 V) instead of the
        # correct ~0 drop.
        drop_iso_bulk, _ = v_bulk['iso']
        drop_iso_stream, _ = v_stream['iso']
        assert drop_iso_bulk < 1e-3, (
            f"[method={method}] island node 'iso' shows peak IR-drop="
            f"{drop_iso_bulk!r} in the bulk path (expected ~0, pinned near "
            f"Vdd)."
        )
        assert drop_iso_stream < 1e-3, (
            f"[method={method}] island node 'iso' shows peak IR-drop="
            f"{drop_iso_stream!r} in the streaming path (expected ~0, "
            f"pinned near Vdd)."
        )


# ---------------------------------------------------------------------------
# T2: explicit preconditioner='jacobi' in never-assemble mode must include
# S_extra's diagonal (island 1e5 penalties + package conductances) -- the S4
# fix was applied to the sibling _build_jacobi_fallback (the block_jacobi
# memory-budget-downgrade path) but not to this explicit branch.
# ---------------------------------------------------------------------------


class TestT2ExplicitJacobiDiagonalIncludesSExtra:
    """Direct, deterministic regression: the explicit preconditioner='jacobi'
    LinearOperator's diagonal must equal sum_i diag(S_i) + diag(S_extra) in
    never-assemble mode (S_global=None) -- NOT just sum_i diag(S_i). This is
    the exact S4 fix applied to the sibling ``_build_jacobi_fallback`` but
    missed on this explicit branch. A direct diagonal check (rather than an
    end-to-end CG convergence check) is used because exact CG converges in
    at most n_interface iterations regardless of preconditioner quality, so
    a small fixture's end-to-end convergence cannot reliably distinguish
    "correctly scaled" from "mis-scaled by 5 orders of magnitude"."""

    def test_diagonal_includes_s_extra_in_never_assemble_mode(self):
        from distributed.interface_iterative import InterfaceCGSolver
        import scipy.sparse as sp

        # Two tiny tiles: 'a' covers global indices {0, 1}; 'b' covers
        # global index {2} (the "island" row).
        n = 3
        tile_schur = {
            'a': np.array([[2.0, 0.0], [0.0, 3.0]]),
            'b': np.array([[4.0]]),
        }
        tile_idx = {
            'a': np.array([0, 1], dtype=np.int32),
            'b': np.array([2], dtype=np.int32),
        }
        penalty = 1e5
        S_extra = sp.coo_matrix(
            ([penalty], ([2], [2])), shape=(n, n),
        ).tocsr()

        cg_solver = InterfaceCGSolver(
            n_interface=n,
            matvec_mode='tilewise',
            S_global=None,  # never-assemble mode
            tile_schur_complements=tile_schur,
            tile_index_maps=tile_idx,
            S_extra=S_extra,
            preconditioner='jacobi',
        )

        M = cg_solver._M
        assert M is not None, "jacobi preconditioner must be built"

        # M^{-1} applied to e_2 (the island row) should be ~1/(4 + 1e5),
        # NOT 1/4 -- pre-fix, the explicit 'jacobi' branch omits S_extra's
        # diagonal entirely, mis-scaling this row by ~5 orders of magnitude.
        e2 = np.zeros(n)
        e2[2] = 1.0
        m_e2 = M.matvec(e2)
        expected_diag = 4.0 + penalty
        assert abs(1.0 / m_e2[2] - expected_diag) < 1e-6, (
            f"preconditioner diagonal at the island row = "
            f"{1.0 / m_e2[2]!r}, expected {expected_diag!r} (tile diag 4.0 "
            f"+ S_extra's 1e5 mS island penalty) -- S_extra's diagonal is "
            f"missing from the explicit preconditioner='jacobi' branch."
        )

        # Non-penalized rows are unaffected by S_extra (all zero there).
        e0 = np.zeros(n)
        e0[0] = 1.0
        assert abs(M.matvec(e0)[0] - 1.0 / 2.0) < 1e-12
        e1 = np.zeros(n)
        e1[1] = 1.0
        assert abs(M.matvec(e1)[1] - 1.0 / 3.0) < 1e-12


class TestT2ExplicitJacobiNeverAssembleIncludesSExtra:
    """End-to-end sanity check (never-assemble + explicit jacobi on an
    islanded fixture): solve must converge comparably to block_jacobi.
    Complements the deterministic diagonal check above -- the
    diagonal omitted S_extra's island-penalty term, mis-scaling the
    preconditioner by ~5 orders of magnitude on the islanded row and
    destroying CG convergence (or, with strict=False, silently returning a
    degraded iterate)."""

    @staticmethod
    def _build_model(tmp_path):
        """Never-assemble precondition (summaries island detection) + a
        genuine island ('iso', no resistive OR cap path to any pad --
        cap_c_fF=0.0)."""
        model = _build_three_tile_island_model(cap_c_fF=0.0)
        model.island_detection_mode = 'summaries'
        model.component_summaries = []
        return model

    def test_explicit_jacobi_converges_comparably_to_block_jacobi(self, tmp_path):
        from distributed.solver import DistributedDDMSolver

        # Direct-mode (schur_bfs, assembled S_global) oracle.
        model_direct = self._build_model(tmp_path)
        model_direct.island_detection_mode = 'schur_bfs'  # irrelevant for direct
        solver_direct = DistributedDDMSolver(model_direct)
        ctx_direct = solver_direct.prepare()
        try:
            result_direct = solver_direct.solve_dc(ctx_direct).flatten()
        finally:
            ctx_direct.release()
            model_direct.shutdown()

        common_settings = {
            'interface_solver': 'cg',
            'interface_matvec_mode': 'tilewise',
            'interface_drop_s_global': True,
            'interface_cg_rtol': 1e-10,
            'interface_cg_atol': 1e-14,
            'interface_cg_strict': True,
        }

        results = {}
        for precond in ('jacobi', 'block_jacobi'):
            model = self._build_model(tmp_path)
            model.settings.update(common_settings)
            model.settings['interface_preconditioner'] = precond
            solver = DistributedDDMSolver(model)
            ctx = solver.prepare()
            try:
                assert ctx._S_global is None, (
                    "interface_drop_s_global=True must never assemble S_global"
                )
                assert ctx._cg_solver.preconditioner == precond
                # Pre-fix, the explicit 'jacobi' branch's mis-scaled
                # diagonal (missing S_extra's 1e5 mS island penalty) makes
                # CG fail to converge within the resolved maxiter, raising
                # RuntimeError under interface_cg_strict=True (the default
                # used here). This call must NOT raise.
                results[precond] = solver.solve_dc(ctx).flatten()
            finally:
                ctx.release()
                model.shutdown()

        v_jacobi, v_bj = results['jacobi'], results['block_jacobi']
        common = set(v_jacobi) & set(v_bj) & set(result_direct)
        assert common, "fixture produced no comparable nodes"
        assert 'iso' in common, "the fixture's island node must be comparable"
        for node in common:
            # jacobi vs block_jacobi: both are preconditioners for the SAME
            # underlying linear system -- converged CG must agree closely
            # regardless of which preconditioner was used.
            assert abs(v_jacobi[node] - v_bj[node]) < 1e-6, (
                f"node {node}: explicit jacobi={v_jacobi[node]!r} vs "
                f"block_jacobi={v_bj[node]!r} -- diverge beyond CG "
                f"convergence tolerance (mis-scaled preconditioner "
                f"symptom)."
            )
            # jacobi vs the direct/schur_bfs oracle.
            assert abs(v_jacobi[node] - result_direct[node]) < 1e-6, (
                f"node {node}: explicit jacobi={v_jacobi[node]!r} vs "
                f"direct oracle={result_direct[node]!r}"
            )


# ---------------------------------------------------------------------------
# T3: refactor()'s tilewise re-gather must raise loudly (not silently drop a
# port's row/column) when a LIVE worker's port list contains a node absent
# from the LOADED checkpoint's STALE interface_node_to_idx ordering, unless
# that node is a genuine Dirichlet pad.
# ---------------------------------------------------------------------------


class TestT3RefactorStaleOrderingRaisesLoudly:
    """Regression: a model/checkpoint topology mismatch (e.g. re-parsing
    with a different retile between save() and load()+refactor()) must
    raise a clear ValueError naming the tile and node, not silently
    reclassify the drifted port as a Dirichlet pad and drop its row/column
    from the tilewise interface operator."""

    def test_drifted_live_port_raises_on_refactor(self, tmp_path):
        from test_interface_iterative import _build_dc_model
        from distributed.solver import DistributedDDMSolver

        model = _build_dc_model(
            interface_solver='cg',
            interface_matvec_mode='tilewise',
            interface_preconditioner='jacobi',
            interface_cg_rtol=1e-10,
        )
        solver = DistributedDDMSolver(model)
        ctx = solver.prepare()
        assert ctx._cg_solver is not None
        assert ctx._cg_solver.matvec_mode == 'tilewise'

        path = str(tmp_path / 'dc_ctx_drift.pkl')
        ctx.save(path)
        ctx.release()

        ctx2 = type(ctx).load(model, path)

        # Simulate topology drift: tamper tile (0, 0)'s worker so its LIVE
        # port list (as returned by factor_and_compute_schur(), the RPC
        # refactor()'s re-gather calls) includes a node that was NEVER part
        # of the saved/stale interface_node_to_idx ordering, and is not a
        # Dirichlet pad either -- e.g. a retile that changed which nodes
        # land on tile (0, 0)'s boundary between save() and load().
        worker0 = model.workers[0]
        assert model.metadata.tile_configs[0].tile_id == (0, 0)
        orig_method = worker0.factor_and_compute_schur

        def _tampered_factor_and_compute_schur():
            S, port_nodes, stats = orig_method()
            S_arr = np.asarray(S, dtype=np.float64)
            n = S_arr.shape[0]
            S_new = np.zeros((n + 1, n + 1), dtype=np.float64)
            S_new[:n, :n] = S_arr
            S_new[n, n] = 1.0
            port_nodes_new = list(port_nodes) + ['drifted_phantom_node']
            return S_new, port_nodes_new, stats

        worker0.factor_and_compute_schur = _tampered_factor_and_compute_schur

        try:
            with pytest.raises(ValueError, match='drifted_phantom_node'):
                ctx2.refactor()
        finally:
            worker0.factor_and_compute_schur = orig_method
            ctx2.release()
            model.shutdown()

    def test_genuine_dirichlet_pad_on_port_does_not_raise(self, tmp_path):
        """Sanity: a REAL Dirichlet pad directly on a tile's port list (the
        D1 scenario -- legitimately dropped, not a topology mismatch) must
        NOT trigger the new loud error."""
        from distributed.solver import DistributedDDMSolver

        model = _build_two_tile_distributed_model(
            package_cap_edges=[('pad', 'shared', 50.0)],  # D1 fixture
        )
        model.settings.update({
            'interface_solver': 'cg',
            'interface_matvec_mode': 'tilewise',
            'interface_preconditioner': 'jacobi',
            'interface_cg_rtol': 1e-10,
        })
        solver = DistributedDDMSolver(model)
        ctx = solver.prepare()
        assert ctx._cg_solver.matvec_mode == 'tilewise'

        path = str(tmp_path / 'dc_ctx_pad_on_port.pkl')
        ctx.save(path)
        ctx.release()

        ctx2 = type(ctx).load(model, path)
        try:
            ctx2.refactor()  # must NOT raise
            assert ctx2._cg_solver is not None
            assert ctx2._cg_solver.matvec_mode == 'tilewise'
        finally:
            ctx2.release()
            model.shutdown()


# ---------------------------------------------------------------------------
# T4: filter_kept_rhs's len(g_i) == len(idx_map) fast path must still
# validate against port_count_map -- a drifted tile whose g_i length only
# COINCIDENTALLY matches idx_map's length must raise, not silently pass
# through unfiltered and scatter to the wrong interface rows.
# ---------------------------------------------------------------------------


class TestT4FilterKeptRhsFastPathValidation:

    def test_fast_path_raises_on_coincidental_length_match_with_drifted_port_count(
        self,
    ):
        """A tile recorded at factor() time with FULL port count 3 (one
        dropped Dirichlet/pad port, kept_pos length 2) that has since
        drifted so a fresh worker call returns a 2-length g_i must NOT take
        the (len(g_i) == len(idx_map)) fast path silently -- 2 == 2 is a
        coincidence, not proof the tile still has zero dropped ports."""
        from distributed.interface_iterative import filter_kept_rhs

        tid = (0, 0)
        g_i = np.array([10.0, 20.0])
        idx_map = np.array([3, 7], dtype=np.int32)
        kept_pos_map = {tid: np.array([0, 2], dtype=np.int64)}
        port_count_map = {tid: 3}  # recorded FULL port count != len(g_i)

        with pytest.raises(ValueError, match=r'tile \(0, 0\)'):
            filter_kept_rhs(
                g_i, tid, idx_map, kept_pos_map, port_count_map,
                caller='test_caller',
            )

    def test_fast_path_passes_when_port_count_is_consistent(self):
        """A genuinely no-dropped-port tile (port_count == len(idx_map))
        must still take the fast path and return g_i unfiltered."""
        from distributed.interface_iterative import filter_kept_rhs

        tid = (0, 0)
        g_i = np.array([10.0, 20.0])
        idx_map = np.array([3, 7], dtype=np.int32)
        kept_pos_map = {tid: np.array([0, 1], dtype=np.int64)}
        port_count_map = {tid: 2}  # consistent: no ports dropped

        result = filter_kept_rhs(
            g_i, tid, idx_map, kept_pos_map, port_count_map,
            caller='test_caller',
        )
        np.testing.assert_array_equal(result, g_i)

    def test_fast_path_gracefully_degrades_without_port_count_map(self):
        """Pre-Stage-2 contexts (no port_count_map, or an entry missing for
        this tile) must still take the fast path without raising."""
        from distributed.interface_iterative import filter_kept_rhs

        tid = (0, 0)
        g_i = np.array([10.0, 20.0])
        idx_map = np.array([3, 7], dtype=np.int32)

        result = filter_kept_rhs(
            g_i, tid, idx_map, kept_pos_map={}, port_count_map=None,
            caller='test_caller',
        )
        np.testing.assert_array_equal(result, g_i)

        # port_count_map present but with no entry for THIS tile.
        result2 = filter_kept_rhs(
            g_i, tid, idx_map, kept_pos_map={}, port_count_map={(9, 9): 5},
            caller='test_caller',
        )
        np.testing.assert_array_equal(result2, g_i)


# ---------------------------------------------------------------------------
# Finding 6: factor() re-run on an already-factored context must close() the
# OUTGOING InterfaceCGSolver's persistent thread pool before replacing it --
# the S12 lifecycle-parity fix covered release() and both refactor() sites,
# but not the three factor() sites.
# ---------------------------------------------------------------------------


class TestFinding6FactorClosesOutgoingCgSolver:

    def test_dc_repeated_factor_closes_outgoing_cg_solver_pool(self, tmp_path):
        from test_interface_iterative import _build_dc_model
        from distributed.solver import DistributedDDMSolver

        model = _build_dc_model(
            interface_solver='cg',
            interface_preconditioner='block_jacobi',
            interface_cg_rtol=1e-10,
            matvec_threads=2,
        )
        solver = DistributedDDMSolver(model)
        ctx = solver.prepare()
        try:
            first_solver = ctx._cg_solver
            assert first_solver is not None
            # The pool is built lazily on first use (module docstring) --
            # force it via a direct solve (__call__).
            first_solver(np.ones(first_solver.n))
            assert first_solver._pool is not None, (
                "sanity: matvec_threads=2 + block_jacobi must build a "
                "persistent thread pool to begin with"
            )

            # Re-run factor() on the SAME context without release() --
            # exactly the finding 6 scenario (e.g. changing CG tolerances
            # between calls).
            ctx.factor()
            second_solver = ctx._cg_solver
            assert second_solver is not None
            assert second_solver is not first_solver, (
                "sanity: factor() must build a fresh InterfaceCGSolver"
            )
            assert first_solver._pool is None, (
                "the OUTGOING CG solver's thread pool must be closed "
                "(pool set to None) when factor() replaces ctx._cg_solver "
                "on an already-factored context -- otherwise its threads "
                "(and, in never-assemble mode, its retained per-tile Schur "
                "blocks) leak until an eventual cyclic-GC pass."
            )
        finally:
            ctx.release()
            model.shutdown()

    def test_transient_repeated_factor_closes_outgoing_cg_solver_pool(self):
        from distributed.solver import DistributedDDMSolver

        model = _build_two_tile_distributed_model(package_cap_edges=[])
        model.settings.update({
            'interface_solver': 'cg',
            'interface_preconditioner': 'block_jacobi',
            'interface_cg_rtol': 1e-10,
            'matvec_threads': 2,
        })
        solver = DistributedDDMSolver(model)
        trans_ctx = solver.prepare_transient(dt=1e-10, method='be')
        try:
            first_solver = trans_ctx._cg_solver
            assert first_solver is not None
            # The pool is built lazily on first use -- force it via a
            # direct solve (__call__) rather than the full solve_transient
            # pipeline (which needs a DC context + preprocessed sources,
            # out of scope for this lifecycle-only test).
            first_solver(np.ones(first_solver.n))
            assert first_solver._pool is not None, (
                "sanity: matvec_threads=2 + block_jacobi must build a "
                "persistent thread pool to begin with"
            )

            trans_ctx.factor()
            second_solver = trans_ctx._cg_solver
            assert second_solver is not None
            assert second_solver is not first_solver
            assert first_solver._pool is None, (
                "the OUTGOING transient CG solver's thread pool must be "
                "closed when factor() replaces ctx._cg_solver on an "
                "already-factored context."
            )
        finally:
            trans_ctx.release()
            model.shutdown()

    def test_never_assemble_repeated_factor_closes_outgoing_cg_solver_pool(
        self, tmp_path,
    ):
        """Third factor() site: _factor_dc_context_no_s_global (item 3,
        interface_drop_s_global=True)."""
        from distributed.solver import DistributedDDMSolver

        model = _build_two_tile_distributed_model(package_cap_edges=[])
        model.island_detection_mode = 'summaries'
        model.component_summaries = []
        model.settings.update({
            'interface_solver': 'cg',
            'interface_matvec_mode': 'tilewise',
            'interface_preconditioner': 'block_jacobi',
            'interface_cg_rtol': 1e-10,
            'interface_drop_s_global': True,
            'matvec_threads': 2,
        })
        solver = DistributedDDMSolver(model)
        ctx = solver.prepare()
        try:
            assert ctx._S_global is None, "sanity: never-assemble path"
            first_solver = ctx._cg_solver
            assert first_solver is not None
            # The pool is built lazily on first use -- force it via a
            # direct solve (__call__).
            first_solver(np.ones(first_solver.n))
            assert first_solver._pool is not None, (
                "sanity: matvec_threads=2 + block_jacobi must build a "
                "persistent thread pool to begin with"
            )

            ctx.factor()
            second_solver = ctx._cg_solver
            assert second_solver is not None
            assert second_solver is not first_solver
            assert first_solver._pool is None, (
                "the OUTGOING never-assemble CG solver's thread pool must "
                "be closed when factor() replaces ctx._cg_solver on an "
                "already-factored context -- this mode ALSO retains a "
                "full tile_schur_complements dict (tens of GB at BRCM "
                "scale), so leaking it here is the highest-stakes instance "
                "of finding 6."
            )
        finally:
            ctx.release()
            model.shutdown()


# ---------------------------------------------------------------------------
# Stage 3 code-review finding (double-append guard): the GenEO-lite
# eigensolve used to run INSIDE the try block whose `except la.LinAlgError`
# implements the cho_factor-failed SPD-safe fallback, AFTER the 'cho' entry
# was already appended to _block_factors for this block. A LinAlgError
# raised by GenEO's own dense eigh calls (interface_coarse.py's ARPACK-
# ArpackNoConvergence/exception fallbacks) landed in that same except
# branch, which appended a SECOND ('eigh') entry for the SAME
# owned_global_arr -- breaking the block-ownership partition
# _bj_perm/_bj_offsets rely on (double gather/apply, double-write scatter).
# ---------------------------------------------------------------------------


class TestGenEODoubleAppendGuard:

    def test_geneo_linalgerror_does_not_double_append_block(self, monkeypatch):
        """A LinAlgError raised inside the GenEO-lite eigensolve (reusing
        the just-built cho factor) must give exactly ONE _block_factors
        entry for the block ('cho', unchanged) -- not two. Also verifies
        the resulting block-Jacobi apply is still numerically correct
        (single, unambiguous inverse per node)."""
        import scipy.linalg as la
        import distributed.interface_iterative as ii_mod
        from distributed.interface_iterative import InterfaceCGSolver

        n = 4
        diag_vals = np.array([2.0, 3.0, 4.0, 5.0])
        block = np.diag(diag_vals)
        tile_schur = {'a': block}
        tile_idx = {'a': np.arange(n, dtype=np.int32)}

        def _raise_linalg_error(sub, cho=None, k=4, tol=1e-6,
                                 small_block_threshold=500,
                                 precomputed=None, island_local_mask=None):
            raise la.LinAlgError("simulated GenEO eigensolve failure")

        monkeypatch.setattr(
            ii_mod.interface_coarse, 'geneo_lowest_eigenpairs',
            _raise_linalg_error,
        )

        cg_solver = InterfaceCGSolver(
            n_interface=n, matvec_mode='tilewise',
            tile_schur_complements=tile_schur, tile_index_maps=tile_idx,
            preconditioner='two_level', matvec_threads=1,
            interface_coarse_geneo_k=4,
        )
        try:
            # Exactly one entry per block -- not two ('cho' + spurious
            # 'eigh' double-append for the same owned_global_arr).
            assert len(cg_solver._bj_block_factors) == 1, (
                f"expected exactly 1 block-Jacobi factor entry (single "
                f"'cho' block), got {len(cg_solver._bj_block_factors)} -- "
                f"a GenEO LinAlgError double-appended the same block"
            )
            kinds = [kind for (_, kind, _) in cg_solver._bj_block_factors]
            assert kinds == ['cho'], (
                f"expected the block to remain 'cho' (GenEO failure must "
                f"not touch the cho block's own factor entry); got "
                f"kinds={kinds!r}"
            )
            # No duplicated global indices in the ownership permutation.
            assert len(cg_solver._bj_perm) == n
            assert len(set(cg_solver._bj_perm.tolist())) == n, (
                "block-ownership partition must have no duplicate indices "
                "(a double-append would duplicate this tile's global "
                "indices in _bj_perm)"
            )
            # The base block-Jacobi component's materialized inverse must
            # be numerically correct (single, unambiguous inverse per
            # node) -- diag(1/diag_vals) -- checked directly on the
            # materialized block rather than through cg_solver._M
            # (which, for preconditioner='two_level', is the ADDITIVE
            # base+coarse combination, not the base alone).
            np.testing.assert_allclose(
                cg_solver._bj_inv_blocks[0], np.diag(1.0 / diag_vals),
                rtol=1e-10, atol=1e-12,
            )
        finally:
            cg_solver.close()


# ---------------------------------------------------------------------------
# Finding 7: S11 narrowed the block-Jacobi eigh-based fallback's error path
# to `except la.LinAlgError`, but np.linalg.eigh (and its 0.5*(sub+sub.T)
# allocation) can raise other exceptions (MemoryError, ValueError on
# NaN/Inf-contaminated input) -- the pre-Stage-2 code's own comment claims
# "must be just as resilient" (skip the block, don't crash prepare()).
# ---------------------------------------------------------------------------


class TestFinding7BlockJacobiEighFallbackCatchAll:

    def test_non_linalgerror_from_eigh_fallback_does_not_crash_construction(
        self, monkeypatch,
    ):
        import distributed.interface_iterative as ii_mod
        from distributed.interface_iterative import InterfaceCGSolver

        n = 4
        # Tile 'a' is well-conditioned SPD. Tile 'b' is symmetric
        # INDEFINITE (eigenvalues +3, -1) -- cho_factor raises LinAlgError,
        # entering the eigh-based fallback branch.
        tile_schur = {
            'a': np.array([[2.0, 0.0], [0.0, 3.0]]),
            'b': np.array([[1.0, 2.0], [2.0, 1.0]]),
        }
        tile_idx = {
            'a': np.array([0, 1], dtype=np.int32),
            'b': np.array([2, 3], dtype=np.int32),
        }

        def _raise_non_linalg_error(sub, eps_rel=1e-10):
            raise ValueError("simulated non-LinAlgError failure inside eigh")

        # `_build_block_jacobi` calls `_spd_safe_pseudo_solve_factor_ex`
        # directly on its eigh-fallback path (Stage 3 GenEO-lite plumbing
        # needs the raw spectrum `w`, see that function's docstring) -- NOT
        # the 2-tuple `_spd_safe_pseudo_solve_factor` wrapper, which is only
        # a thin, unused-by-production delegator kept for its own direct
        # unit test (`TestSPDSafeFallback`). Patching the wrapper here would
        # leave production's `_ex` call untouched and this test would pass
        # vacuously without ever reaching the `except Exception` catch-all
        # under test.
        monkeypatch.setattr(
            ii_mod, '_spd_safe_pseudo_solve_factor_ex', _raise_non_linalg_error,
        )

        # Pre-fix: `except la.LinAlgError` does not catch ValueError, so
        # this ValueError propagates out of _build_block_jacobi and aborts
        # __init__ entirely (the whole prepare()/factor() call crashes).
        # Post-fix: caught, logged, and the block is skipped (identity
        # fallback for its nodes) -- construction must succeed.
        cg_solver = InterfaceCGSolver(
            n_interface=n,
            matvec_mode='tilewise',
            S_global=None,
            tile_schur_complements=tile_schur,
            tile_index_maps=tile_idx,
            preconditioner='block_jacobi',
        )
        assert cg_solver is not None
        assert cg_solver._M is not None, (
            "tile 'a' block still builds a valid preconditioner even "
            "though tile 'b's block failed and was skipped"
        )
        # Directly verify the catch-all actually fired for tile 'b': only
        # tile 'a's well-conditioned block should have factored ('cho');
        # tile 'b' must be ABSENT (skipped), not present as 'eigh' (which
        # would mean the simulated failure never reached the eigh-fallback
        # path at all -- i.e. the monkeypatch missed). This is the
        # assertion that fails vacuously (kinds == ['cho', 'eigh']) if the
        # monkeypatch target regresses back to the unused 2-tuple wrapper.
        kinds = [kind for (_, kind, _) in cg_solver._bj_block_factors]
        assert kinds == ['cho'], (
            f"expected tile 'b' to be skipped by the except-Exception "
            f"catch-all (leaving only tile 'a's 'cho' block), got kinds="
            f"{kinds!r} -- the simulated eigh-fallback failure did not "
            f"reach _build_block_jacobi's eigh-fallback path"
        )


# ---------------------------------------------------------------------------
# Finding 8: a YAML '.inf' budget/thread-count/maxiter setting (PyYAML
# parses '.inf' to float('inf')) must raise the intended friendly
# ValueError, not an unhandled OverflowError.
# ---------------------------------------------------------------------------


class TestFinding8OverflowErrorOnInfSettings:

    def test_memory_budget_inf_raises_value_error_not_overflow_error(self):
        from distributed.interface_iterative import _resolve_memory_budget_bytes

        with pytest.raises(ValueError, match='Invalid memory-budget setting'):
            _resolve_memory_budget_bytes(
                float('inf'), auto_cap_bytes=1024, auto_fraction=0.1,
            )

    def test_factor_memory_budget_inf_raises_value_error(self):
        from distributed.interface_iterative import resolve_factor_memory_budget_bytes

        with pytest.raises(ValueError, match='Invalid memory-budget setting'):
            resolve_factor_memory_budget_bytes(float('inf'))

    def test_block_jacobi_max_bytes_inf_raises_value_error(self):
        from distributed.interface_iterative import resolve_block_jacobi_max_bytes

        with pytest.raises(ValueError, match='Invalid memory-budget setting'):
            resolve_block_jacobi_max_bytes(float('inf'))

    def test_matvec_threads_inf_raises_value_error_not_overflow_error(self):
        from distributed.interface_iterative import resolve_matvec_threads

        with pytest.raises(ValueError, match='Invalid matvec_threads setting'):
            resolve_matvec_threads(float('inf'), n_tiles=4)

    def test_coerce_int_inf_raises_value_error_not_overflow_error(self):
        from distributed.result_factorization import _coerce_int

        with pytest.raises(ValueError, match='Invalid integer setting'):
            _coerce_int(float('inf'), 'interface_cg_maxiter')

    def test_yaml_string_dot_inf_also_raises_value_error(self):
        """PyYAML parses the bare token '.inf' to float('inf') -- exercise
        the exact string form a user would write in solver.yaml."""
        import yaml
        from distributed.interface_iterative import resolve_matvec_threads

        parsed = yaml.safe_load('matvec_threads: .inf')['matvec_threads']
        assert parsed == float('inf'), "sanity: PyYAML parses '.inf' to float('inf')"

        with pytest.raises(ValueError, match='Invalid matvec_threads setting'):
            resolve_matvec_threads(parsed, n_tiles=4)


# ---------------------------------------------------------------------------
# Finding 9: solve_transient's dc_context initial-condition RHS gather must
# label its filter_kept_rhs drift error as coming from solve_transient (the
# function it's actually in), not solve_quasi_static (copy-paste label from
# the sibling fix).
# ---------------------------------------------------------------------------


class TestFinding9SolveTransientICCallerLabel:

    def test_dc_ic_drift_error_labeled_solve_transient_not_solve_quasi_static(
        self, tmp_path,
    ):
        from distributed.solver import DistributedDDMSolver

        # D1 pad-on-port fixture: tile (0, 0) has a pad directly on its own
        # port list, so its dc_ctx IC gather takes filter_kept_rhs's SLOW
        # (port_count-validated) path, not the fast path.
        model = _build_two_tile_distributed_model(
            package_cap_edges=[('pad', 'shared', 50.0)],
        )
        for tc in model.metadata.tile_configs:
            tc.ckt_path = str(tmp_path / 'dummy.ckt')
        model.settings.update({
            'interface_solver': 'cg',
            'interface_matvec_mode': 'tilewise',
            'interface_preconditioner': 'block_jacobi',
            'interface_cg_rtol': 1e-10,
        })
        solver = DistributedDDMSolver(model)
        dc_ctx = solver.prepare()
        trans_ctx = solver.prepare_transient(dt=1e-10, method='be')
        try:
            # Simulate topology drift: tamper dc_ctx's recorded FULL port
            # count for tile (0, 0) so it no longer matches what
            # tile_kept_port_pos was actually built from -- triggers the
            # S13 drift ValueError specifically inside solve_transient's
            # dc_context IC-gather block (NOT the per-step transient
            # scatter, which uses trans_ctx's own -- untouched -- maps).
            tid = (0, 0)
            assert tid in dc_ctx._tile_port_count, "fixture sanity"
            dc_ctx._tile_port_count[tid] += 1

            smoothed = solver.preprocess_sources(
                time_step=1e-10, t_start=0.0, t_end=2e-9, smooth=False,
            )
            with pytest.raises(ValueError) as exc_info:
                solver.solve_transient(
                    trans_ctx, dc_context=dc_ctx, t_end=2e-9,
                    smoothed_sources=smoothed,
                )
            msg = str(exc_info.value)
            assert 'solve_transient(dc_initial_condition)' in msg, (
                f"error must be labeled with the ACTUAL calling function "
                f"(solve_transient), not the sibling solve_quasi_static; "
                f"got: {msg!r}"
            )
            assert 'solve_quasi_static' not in msg, (
                f"must not misattribute this solve_transient-only IC path "
                f"to solve_quasi_static; got: {msg!r}"
            )
        finally:
            trans_ctx.release()
            dc_ctx.release()
            model.shutdown()


# ---------------------------------------------------------------------------
# Finding 15: an explicit matvec_threads > n_tiles (or > n_owned_blocks for
# block_jacobi) must not create permanently-empty LPT partition bins that
# get dispatched through the thread pool on every matvec/apply call.
# ---------------------------------------------------------------------------


class TestFinding15MatvecThreadsCappedByPartitionSize:

    def test_tilewise_partition_capped_by_n_tiles(self):
        """matvec_threads=32 on a 6-tile model must produce a 6-bin (not
        32-bin) LPT partition -- the excess would otherwise be pure
        dispatch overhead (26 empty work items through the pool) on every
        CG matvec, the exact regime the module's own Stage 0 note flags as
        inverting scaling."""
        from distributed.interface_iterative import InterfaceCGSolver

        n_interface = 12
        tile_schur, tile_idx = _make_synthetic_tiles(
            n_interface, n_tiles=6, ports_lo=2, ports_hi=4, seed=99,
        )
        cg_solver = InterfaceCGSolver(
            n_interface=n_interface,
            matvec_mode='tilewise',
            S_global=None,
            tile_schur_complements=tile_schur,
            tile_index_maps=tile_idx,
            preconditioner='none',
            matvec_threads=32,
        )
        assert cg_solver.matvec_threads == 32, (
            "sanity: the resolved setting itself is honoured as-is (not "
            "capped) -- only the partition/dispatch bin count is capped"
        )
        assert cg_solver._mv_n_bins <= 6, (
            f"LPT partition bin count ({cg_solver._mv_n_bins}) must be "
            f"capped at the number of tiles (6), not matvec_threads (32)"
        )
        assert len(cg_solver._mv_partition) == cg_solver._mv_n_bins

    def test_block_jacobi_partition_capped_by_n_owned_blocks(self):
        """Same cap for the block-Jacobi apply's LPT partition, sized by
        the number of OWNED blocks (one per tile here), not
        matvec_threads."""
        from distributed.interface_iterative import InterfaceCGSolver

        n_interface = 12
        tile_schur, tile_idx = _make_synthetic_tiles(
            n_interface, n_tiles=6, ports_lo=2, ports_hi=4, seed=99,
        )
        cg_solver = InterfaceCGSolver(
            n_interface=n_interface,
            matvec_mode='tilewise',
            S_global=None,
            tile_schur_complements=tile_schur,
            tile_index_maps=tile_idx,
            preconditioner='block_jacobi',
            matvec_threads=32,
        )
        assert cg_solver._bj_n_bins <= 6, (
            f"block-Jacobi LPT partition bin count ({cg_solver._bj_n_bins}) "
            f"must be capped at the number of owned blocks (<=6), not "
            f"matvec_threads (32)"
        )
        assert len(cg_solver._bj_partition) == cg_solver._bj_n_bins

    def test_capped_partition_still_produces_correct_result(self):
        """The cap is a pure dispatch-efficiency fix -- results with
        matvec_threads > n_tiles must be identical to a properly-sized
        thread count."""
        from distributed.interface_iterative import InterfaceCGSolver

        n_interface = 12
        tile_schur, tile_idx = _make_synthetic_tiles(
            n_interface, n_tiles=6, ports_lo=2, ports_hi=4, seed=99,
        )
        rng = np.random.default_rng(7)
        x = rng.standard_normal(n_interface)

        results = {}
        for n_threads in (2, 32):
            cg_solver = InterfaceCGSolver(
                n_interface=n_interface,
                matvec_mode='tilewise',
                S_global=None,
                tile_schur_complements={k: v.copy() for k, v in tile_schur.items()},
                tile_index_maps=tile_idx,
                preconditioner='none',
                matvec_threads=n_threads,
            )
            results[n_threads] = cg_solver._linear_op.matvec(x)

        np.testing.assert_allclose(results[2], results[32], rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------------------
# Round-2 code review fixes: Finding 2 (bare float() coerces YAML bools),
# Finding 9 (import-time DEFAULT_* snapshots defeat monkeypatching).
# ---------------------------------------------------------------------------


class TestFinding2CoerceFloatRejectsBool:
    """Finding 2 (round 2): interface_coarse_geneo_tol/interface_coarse_
    eps_rank were read via a bare ``float(...)`` call in
    ``_read_interface_cg_settings`` -- unlike every sibling coercer in that
    function (``_coerce_int`` for geneo_k/max_cols, ``_coerce_bool`` for
    cg_strict/strict_dtype_rtol) -- so a YAML-1.1 bool
    (``interface_coarse_geneo_tol: yes`` -> Python ``True``) silently
    coerced to ``1.0`` instead of raising (``float(True) == 1.0``)."""

    def test_coerce_float_rejects_bool(self):
        from distributed.result_factorization import _coerce_float

        with pytest.raises(ValueError, match='Invalid float setting'):
            _coerce_float(True, 'interface_coarse_geneo_tol')
        with pytest.raises(ValueError, match='Invalid float setting'):
            _coerce_float(False, 'interface_coarse_eps_rank')

    def test_coerce_float_accepts_real_floats_and_numeric_strings(self):
        from distributed.result_factorization import _coerce_float

        assert _coerce_float(1e-6, 'x') == 1e-6
        assert _coerce_float('1e-6', 'x') == 1e-6
        assert _coerce_float(0, 'x') == 0.0

    def test_coerce_float_inf_raises_value_error(self):
        """Same OverflowError/TypeError/ValueError-catching discipline as
        _coerce_int -- though float('inf') itself is a valid float (unlike
        int(float('inf'))), a non-numeric string must still raise a clean
        ValueError, not a raw TypeError/ValueError from float()."""
        from distributed.result_factorization import _coerce_float

        with pytest.raises(ValueError, match='Invalid float setting'):
            _coerce_float('not-a-number', 'interface_coarse_geneo_tol')
        with pytest.raises(ValueError, match='Invalid float setting'):
            _coerce_float(None, 'interface_coarse_eps_rank')

    def test_geneo_tol_bool_setting_raises_via_read_interface_cg_settings(self):
        """End-to-end: a bool interface_coarse_geneo_tol/eps_rank setting
        (the exact YAML-1.1 'yes'/'no' pitfall) must raise loudly out of
        _read_interface_cg_settings, not silently resolve to 1.0/0.0."""
        from distributed.result_factorization import _read_interface_cg_settings

        class _Model:
            settings = {'interface_coarse_geneo_tol': True}

        with pytest.raises(ValueError, match='Invalid float setting'):
            _read_interface_cg_settings(_Model())

        class _Model2:
            settings = {'interface_coarse_eps_rank': False}

        with pytest.raises(ValueError, match='Invalid float setting'):
            _read_interface_cg_settings(_Model2())


class TestFinding9DynamicCoarseDefaults:
    """Finding 9 (round 2): interface_coarse.py's DEFAULT_GENEO_K/
    DEFAULT_GENEO_TOL/DEFAULT_EPS_RANK/DEFAULT_MAX_COLS/DEFAULT_MAX_BYTES
    must be read DYNAMICALLY (a live attribute lookup at call time) by
    InterfaceCGSolver.__init__/build_interface_solver, not bound as
    def-time (module-import-time) default parameter values -- the latter
    silently defeats monkeypatch.setattr(interface_coarse, 'DEFAULT_*', ...)
    despite a since-removed module constant's comment claiming the
    opposite."""

    def test_monkeypatched_default_geneo_k_reaches_constructor(self, monkeypatch):
        import distributed.interface_coarse as interface_coarse
        from distributed.interface_iterative import InterfaceCGSolver

        tile_schur = {
            0: np.array([[2.0, -1.0], [-1.0, 2.0]]),
            1: np.array([[2.0, -1.0], [-1.0, 2.0]]),
        }
        tile_idx = {
            0: np.array([0, 1], dtype=np.int32),
            1: np.array([1, 2], dtype=np.int32),
        }
        n = 3

        monkeypatch.setattr(interface_coarse, 'DEFAULT_GENEO_K', 0)
        solver = InterfaceCGSolver(
            n_interface=n, matvec_mode='tilewise',
            tile_schur_complements={k: v.copy() for k, v in tile_schur.items()},
            tile_index_maps=tile_idx,
            preconditioner='two_level', matvec_threads=1,
        )
        try:
            assert solver._geneo_k == 0, (
                "InterfaceCGSolver.__init__ did not pick up the "
                "monkeypatched interface_coarse.DEFAULT_GENEO_K -- its "
                "constructor default is bound at def/import time instead "
                "of being resolved dynamically"
            )
            # k=0 means GenEO is disabled -- PoU-only coarse space, no
            # geneo_k mismatch anywhere downstream.
            assert solver._want_geneo is False
        finally:
            solver.close()

    def test_monkeypatched_default_max_bytes_reaches_coarse_build(
        self, monkeypatch,
    ):
        """A tiny monkeypatched DEFAULT_MAX_BYTES must actually constrain
        the coarse build when interface_coarse_max_bytes is NOT passed
        explicitly to the constructor (the direct-constructor, no-
        model.settings path this finding's failure scenario describes)."""
        import distributed.interface_coarse as interface_coarse
        from distributed.interface_iterative import InterfaceCGSolver

        tile_schur = {
            i: np.array([[2.0, -1.0], [-1.0, 2.0]]) for i in range(10)
        }
        tile_idx = {
            i: np.array([i, i + 1], dtype=np.int32) for i in range(10)
        }
        n = 11

        monkeypatch.setattr(interface_coarse, 'DEFAULT_MAX_BYTES', 1)
        solver = InterfaceCGSolver(
            n_interface=n, matvec_mode='tilewise',
            tile_schur_complements={k: v.copy() for k, v in tile_schur.items()},
            tile_index_maps=tile_idx,
            preconditioner='two_level', matvec_threads=1,
        )
        try:
            assert solver._coarse is None, (
                "expected the monkeypatched 1-byte DEFAULT_MAX_BYTES to "
                "disable the coarse space entirely (degrading to the base "
                "preconditioner) -- if _coarse is not None, the "
                "constructor did not pick up the monkeypatch"
            )
            assert solver.preconditioner != 'two_level'
        finally:
            solver.close()

    def test_monkeypatched_default_reaches_build_interface_solver(
        self, monkeypatch,
    ):
        """The same dynamic-default requirement applies one layer up, at
        build_interface_solver -- it must forward a None sentinel (not its
        own def-time-bound copy of interface_coarse.DEFAULT_GENEO_K) so
        InterfaceCGSolver's own dynamic resolution is not shadowed."""
        import distributed.interface_coarse as interface_coarse
        from distributed.interface_iterative import build_interface_solver

        tile_schur = {
            0: np.array([[2.0, -1.0], [-1.0, 2.0]]),
            1: np.array([[2.0, -1.0], [-1.0, 2.0]]),
        }
        tile_idx = {
            0: np.array([0, 1], dtype=np.int32),
            1: np.array([1, 2], dtype=np.int32),
        }
        n = 3

        monkeypatch.setattr(interface_coarse, 'DEFAULT_GENEO_K', 0)
        _, _, cg_solver = build_interface_solver(
            S_global=None,
            interface_solver='cg',
            matvec_mode='tilewise',
            tile_schur_complements={k: v.copy() for k, v in tile_schur.items()},
            tile_index_maps=tile_idx,
            preconditioner='two_level',
            matvec_threads=1,
            n_interface=n,
        )
        try:
            assert cg_solver._geneo_k == 0, (
                "build_interface_solver did not forward a None sentinel "
                "for interface_coarse_geneo_k -- InterfaceCGSolver's "
                "dynamic default resolution was shadowed by a def-time-"
                "bound value here instead"
            )
        finally:
            cg_solver.close()

"""Tests for B2: iterative interface solve (InterfaceCGSolver).

Covers:
  - CG vs direct on two-tile fixture and netlist_sampled pkl (DC: |dV| <= 1e-8)
  - Both matvec modes ('assembled', 'tilewise') agree with each other and direct
  - Preconditioner effectiveness: block_jacobi < none in iteration count
  - Warm start reduces iterations across consecutive transient steps
  - Auto-select boundaries (monkeypatch thresholds)
  - Default on netlist_sampled resolves to 'direct' (small interface system)
  - CG vs flat-solver comparison (DC |dV| <= 1e-8)

Two fixtures are used:
  - _build_dc_model(): built via ParsedTileBundle/create_distributed_model;
    compatible with solver.solve_dc() (used for all DC tests).
  - _build_two_tile_distributed_model(): used for transient tests;
    designed for prepare_transient() / solve_transient() tests.
"""

from __future__ import annotations

import os
import sys
import tempfile
import pickle
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest
import scipy.sparse as sp

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# Transient fixture from test_time_domain (prepare_transient-compatible only)
# ---------------------------------------------------------------------------

sys.path.insert(0, str(Path(__file__).parent))
from test_time_domain import _build_two_tile_distributed_model


# ---------------------------------------------------------------------------
# DC fixture: uses create_distributed_model (solve_dc-compatible)
# ---------------------------------------------------------------------------


def _build_dc_model(**extra_settings):
    """Build a minimal two-tile distributed model compatible with solve_dc().

    Uses ParsedTileBundle + create_distributed_model so the tile workers are
    set up correctly (boundary_nodes only contains the shared interface node,
    not Dirichlet/pad nodes).

    Layout:
        Tile (0,0): n1 --[1mS]-- B --[2mS]-- n2 --[1mS to gnd]
                    n2 has grounded cap 10fF, 0.5 mA load
        Tile (0,1): B --[3mS]-- n3 --[2mS to gnd]
                    n3 has grounded cap 20fF, 0.3 mA load
        Package: pad_v at Vdd=1.0V connected to B via 10mS

    Interface: {B}
    Pad (Dirichlet): {pad_v}
    """
    from distributed.tile_worker import TileData
    from distributed.parser import PackageData, PowerGridMetaData, TileConfig
    from distributed.model import ParsedTileBundle, create_distributed_model

    tile_a = TileData(
        tile_id=(0, 0),
        resistive_edges=[
            ('n1', 'B', 1.0),
            ('B', 'n2', 2.0),
            ('n2', '0', 1.0),
        ],
        all_nodes={'n1', 'B', 'n2'},
        boundary_nodes={'B'},
        current_injections={'n2': 0.5},
        capacitive_edges=[('n2', '0', 10.0)],
    )
    tile_b = TileData(
        tile_id=(0, 1),
        resistive_edges=[
            ('B', 'n3', 3.0),
            ('n3', '0', 2.0),
        ],
        all_nodes={'B', 'n3'},
        boundary_nodes={'B'},
        current_injections={'n3': 0.3},
        capacitive_edges=[('n3', '0', 20.0)],
    )

    pkg = PackageData(
        vsrc_dict={'V1': {'node_pos': 'pad_v', 'node_neg': '0',
                          'net': 'VDD', 'value': 1.0}},
        package_edges=[('pad_v', 'B', 10.0)],
        pad_nodes={'pad_v'},
        tap_nodes=set(),
        die_attachment_nodes=set(),
        vdd=1.0,
        net_name='VDD',
        package_cap_edges=[('pad_v', '0', 5.0)],
    )

    tile_configs = [
        TileConfig(tile_id=(0, 0), ckt_path='/dev/null',
                   nd_path=None, instance_path=None, net_filter=None),
        TileConfig(tile_id=(0, 1), ckt_path='/dev/null',
                   nd_path=None, instance_path=None, net_filter=None),
    ]
    metadata = PowerGridMetaData(
        tile_grid=(1, 2),
        parameters={'VDD': '1.0'},
        tile_configs=tile_configs,
        package_data=pkg,
        net_name='VDD',
        vdd=1.0,
    )

    tmpdir = tempfile.mkdtemp()
    for td in [tile_a, tile_b]:
        x, y = td.tile_id
        with open(os.path.join(tmpdir, f'tile_{x}_{y}.pkl'), 'wb') as f:
            pickle.dump(td, f)
    with open(os.path.join(tmpdir, 'metadata.pkl'), 'wb') as f:
        pickle.dump({
            'metadata': metadata,
            'boundary_nodes': {'B'},
        }, f)

    bundle = ParsedTileBundle(
        metadata=metadata,
        shared_boundary_nodes={'B'},
        pkl_dir=tmpdir,
    )
    model = create_distributed_model(bundle, backend='local')
    model._owns_pkl_dir = tmpdir
    model.settings.update(extra_settings)
    return model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_fixtures_dir():
    """Return the path to test fixture netlists (netlist_sampled/distributed_pkl)."""
    tests_dir = Path(__file__).parent.parent
    return tests_dir.parent / 'netlist' / 'netlist_sampled' / 'distributed_pkl'


def _sampled_pkl_available():
    d = _get_fixtures_dir()
    return d.exists() and (d / 'metadata.pkl').exists()


SAMPLED_PKL_AVAILABLE = _sampled_pkl_available()


# ──────────────────────────────────────────────────────────────────────
# 1. Unit tests for InterfaceCGSolver itself
# ──────────────────────────────────────────────────────────────────────


class TestInterfaceCGSolverBasics:
    """Unit tests for InterfaceCGSolver: LinearOperator, preconditioner, warm start."""

    def _make_spd_matrix(self, n, seed=42):
        """Build a random SPD matrix of size n."""
        rng = np.random.default_rng(seed)
        A = rng.standard_normal((n, n))
        S = A @ A.T + n * np.eye(n)  # guarantee SPD
        return sp.csr_matrix(S)

    def test_assembled_mode_solves_correctly(self):
        """Assembled matvec CG matches direct solve."""
        from distributed.interface_iterative import InterfaceCGSolver

        n = 20
        S = self._make_spd_matrix(n)
        rhs = np.random.default_rng(0).standard_normal(n)

        cg = InterfaceCGSolver(
            n_interface=n,
            matvec_mode='assembled',
            S_global=S,
            preconditioner='none',
            rtol=1e-12,
        )
        x_cg = cg(rhs)

        from scipy.sparse.linalg import spsolve
        x_direct = spsolve(S, rhs)

        np.testing.assert_allclose(x_cg, x_direct, atol=1e-8)

    def test_tilewise_mode_solves_correctly(self):
        """Tilewise matvec CG matches direct solve on block-diagonal system."""
        from distributed.interface_iterative import InterfaceCGSolver

        # Build a block-diagonal 2-tile system (4 interface nodes, 2 tiles of 2 each)
        n = 4
        rng = np.random.default_rng(7)
        blocks = []
        tile_schur = {}
        tile_idx = {}
        for i in range(2):
            A = rng.standard_normal((2, 2))
            S_i = A @ A.T + 4 * np.eye(2)
            blocks.append(S_i)
            tile_schur[(i, 0)] = S_i
            tile_idx[(i, 0)] = np.array([2 * i, 2 * i + 1], dtype=np.int32)

        S_block = sp.block_diag(blocks).tocsr()
        rhs = rng.standard_normal(n)

        cg = InterfaceCGSolver(
            n_interface=n,
            matvec_mode='tilewise',
            S_global=None,
            tile_schur_complements=tile_schur,
            tile_index_maps=tile_idx,
            preconditioner='none',
            rtol=1e-12,
        )
        x_cg = cg(rhs)

        from scipy.sparse.linalg import spsolve
        x_direct = spsolve(S_block, rhs)

        np.testing.assert_allclose(x_cg, x_direct, atol=1e-8)

    def test_both_modes_agree_on_block_diagonal_system(self):
        """Assembled and tilewise matvec produce the same solution on block-diagonal system."""
        from distributed.interface_iterative import InterfaceCGSolver

        n = 6
        rng = np.random.default_rng(7)
        blocks = []
        tile_schur = {}
        tile_idx = {}
        for i in range(3):
            A = rng.standard_normal((2, 2))
            S_i = A @ A.T + 4 * np.eye(2)
            blocks.append(S_i)
            tile_schur[(i, 0)] = S_i
            tile_idx[(i, 0)] = np.array([2 * i, 2 * i + 1], dtype=np.int32)

        S_global = sp.block_diag(blocks).tocsr()
        rhs = rng.standard_normal(n)

        cg_asm = InterfaceCGSolver(
            n_interface=n, matvec_mode='assembled',
            S_global=S_global, preconditioner='none', rtol=1e-12,
        )
        cg_tiled = InterfaceCGSolver(
            n_interface=n, matvec_mode='tilewise',
            S_global=None, tile_schur_complements=tile_schur,
            tile_index_maps=tile_idx, preconditioner='none', rtol=1e-12,
        )

        x_asm = cg_asm(rhs)
        x_tiled = cg_tiled(rhs)

        np.testing.assert_allclose(x_asm, x_tiled, atol=1e-8,
                                   err_msg="Assembled vs tilewise should agree")

    def test_warm_start_reduces_iterations_second_solve(self):
        """Warm start (x0 from previous solve) does not increase iterations on similar RHS."""
        from distributed.interface_iterative import InterfaceCGSolver

        n = 30
        S = self._make_spd_matrix(n, seed=99)
        rhs1 = np.random.default_rng(10).standard_normal(n)
        rhs2 = rhs1 + 0.01 * np.random.default_rng(11).standard_normal(n)

        stats_cold: Dict[str, Any] = {}
        cg_cold = InterfaceCGSolver(
            n_interface=n, matvec_mode='assembled',
            S_global=S, preconditioner='none', rtol=1e-12, stats_dict=stats_cold,
        )
        cg_cold(rhs1)      # First solve; warms x0
        cg_cold.reset_warm_start()  # Force cold for second
        cg_cold(rhs2)
        iters_cold = stats_cold['last_cg_iters']

        stats_warm: Dict[str, Any] = {}
        cg_warm = InterfaceCGSolver(
            n_interface=n, matvec_mode='assembled',
            S_global=S, preconditioner='none', rtol=1e-12, stats_dict=stats_warm,
        )
        cg_warm(rhs1)      # First solve; x0 is now the solution of rhs1
        cg_warm(rhs2)      # Second solve WITH warm start (x0 close to solution of rhs2)
        iters_warm = stats_warm['last_cg_iters']

        # Warm start should not increase iterations compared to cold restart
        assert iters_warm <= iters_cold, (
            f"Warm start should not increase iterations: "
            f"warm={iters_warm} vs cold={iters_cold}"
        )

    def test_block_jacobi_strictly_better_than_none(self):
        """Block-Jacobi preconditioner uses STRICTLY FEWER iterations than no preconditioner.

        When a single tile owns ALL interface nodes, the block-Jacobi block is
        the exact inverse of S (the block IS S restricted to all n nodes), so
        CG with block_jacobi converges in exactly 1 iteration — strictly less
        than the unpreconditioned count on a random well-conditioned SPD matrix.

        This is the unit-test complement to
        TestNetlistSampledCG.test_block_jacobi_fewer_iters_than_none_on_sampled,
        which asserts the same strict inequality on the real netlist_sampled
        system.
        """
        from distributed.interface_iterative import InterfaceCGSolver

        n = 10
        S = self._make_spd_matrix(n, seed=55)
        rhs = np.random.default_rng(20).standard_normal(n)
        S_arr = S.toarray()
        # Single tile owns all n nodes → block_jacobi block = exact inverse of S
        # → CG converges in 1 iteration.
        tile_schur = {(0, 0): S_arr}
        tile_idx = {(0, 0): np.arange(n, dtype=np.int32)}

        stats_bj: Dict[str, Any] = {}
        stats_none: Dict[str, Any] = {}

        cg_bj = InterfaceCGSolver(
            n_interface=n, matvec_mode='assembled',
            S_global=S, tile_schur_complements=tile_schur,
            tile_index_maps=tile_idx,
            preconditioner='block_jacobi', rtol=1e-10, stats_dict=stats_bj,
        )
        cg_none = InterfaceCGSolver(
            n_interface=n, matvec_mode='assembled',
            S_global=S, preconditioner='none', rtol=1e-10, stats_dict=stats_none,
        )

        cg_bj(rhs)
        cg_none(rhs)

        iters_bj = stats_bj['last_cg_iters']
        iters_none = stats_none['last_cg_iters']

        assert iters_bj < iters_none, (
            f"block_jacobi ({iters_bj} iters) should be STRICTLY LESS than "
            f"none ({iters_none} iters). When a single tile owns all interface "
            f"nodes the block-Jacobi block is the exact S inverse, so CG "
            f"converges in 1 iteration."
        )

    def test_stats_accumulate_across_calls(self):
        """Stats dict accumulates total_cg_iters and total_cg_solves."""
        from distributed.interface_iterative import InterfaceCGSolver

        n = 10
        S = self._make_spd_matrix(n)
        rhs = np.random.default_rng(30).standard_normal(n)
        stats: Dict[str, Any] = {}

        cg = InterfaceCGSolver(
            n_interface=n, matvec_mode='assembled',
            S_global=S, preconditioner='none', rtol=1e-10, stats_dict=stats,
        )
        cg(rhs)
        cg(rhs + 0.001)
        cg(rhs + 0.002)

        assert stats['total_cg_solves'] == 3
        assert stats['total_cg_iters'] > 0
        assert stats['total_cg_iters'] == cg.total_iterations

    def test_non_convergence_raises_runtime_error(self):
        """CG non-convergence must raise RuntimeError (strict=True default).

        Blocker fix: previously __call__ only emitted a warning and returned
        the best (possibly garbage) iterate.  Now it raises with the rel residual
        in the message and sets stats['cg_failed']=True.

        Construction: use a well-conditioned SPD matrix but cap maxiter=1 so
        CG cannot converge on a non-trivial RHS.
        """
        from distributed.interface_iterative import InterfaceCGSolver

        n = 10
        S = self._make_spd_matrix(n, seed=7)
        rhs = np.random.default_rng(42).standard_normal(n)
        stats: Dict[str, Any] = {}

        cg = InterfaceCGSolver(
            n_interface=n,
            matvec_mode='assembled',
            S_global=S,
            preconditioner='none',
            rtol=1e-14,  # very tight — won't converge in 1 iter
            maxiter=1,   # force non-convergence
            stats_dict=stats,
            strict=True,  # default; explicit for clarity
        )

        with pytest.raises(RuntimeError, match="did not converge"):
            cg(rhs)

        # Stats must record the failure flag and failure count
        assert stats.get('cg_failed') is True, (
            "stats['cg_failed'] must be True after non-convergence"
        )
        assert stats.get('total_cg_failures', 0) >= 1, (
            "stats['total_cg_failures'] must be incremented"
        )
        # The error must include rel_residual in the stats
        assert 'last_cg_rel_residual' in stats, (
            "stats must include 'last_cg_rel_residual' after non-convergence"
        )

    def test_non_convergence_strict_false_warns_not_raises(self):
        """CG non-convergence with strict=False emits warning, returns iterate, sets stats.

        This is the non-strict path for callers that want to recover gracefully.
        """
        from distributed.interface_iterative import InterfaceCGSolver
        import warnings

        n = 10
        S = self._make_spd_matrix(n, seed=7)
        rhs = np.random.default_rng(42).standard_normal(n)
        stats: Dict[str, Any] = {}

        cg = InterfaceCGSolver(
            n_interface=n,
            matvec_mode='assembled',
            S_global=S,
            preconditioner='none',
            rtol=1e-14,
            maxiter=1,   # force non-convergence
            stats_dict=stats,
            strict=False,  # demote to warning
        )

        # Must not raise; must return an array
        result = cg(rhs)
        assert isinstance(result, np.ndarray), "Expected ndarray result"
        assert result.shape == (n,), "Result shape must match n_interface"
        assert stats.get('cg_failed') is True
        assert stats.get('total_cg_failures', 0) >= 1

    def test_atol_prevents_stall_on_near_zero_rhs(self):
        """atol prevents CG from burning maxiter on a near-zero RHS.

        When the RHS is very small (||b|| ~ 1e-15), a pure rtol-only
        criterion requires ||r|| <= rtol * ||b|| ~ 1e-25, which is
        unreachable in finite precision.  With atol=1e-14 CG declares
        convergence as soon as ||r|| <= 1e-14 regardless of ||b||.
        """
        from distributed.interface_iterative import InterfaceCGSolver

        n = 20
        S = self._make_spd_matrix(n, seed=3)
        # Near-zero RHS simulating an early transient step with no active sources
        rhs = np.ones(n) * 1e-15
        stats: Dict[str, Any] = {}

        # With atol=1e-14 (default): CG should converge quickly
        cg_with_atol = InterfaceCGSolver(
            n_interface=n,
            matvec_mode='assembled',
            S_global=S,
            preconditioner='none',
            rtol=1e-10,
            atol=1e-14,
            maxiter=500,
            stats_dict=stats,
        )
        result = cg_with_atol(rhs)
        assert isinstance(result, np.ndarray)
        assert stats.get('cg_failed', False) is False, (
            f"CG with atol=1e-14 should converge on near-zero RHS "
            f"(got rel_res={stats.get('last_cg_rel_residual', 'N/A')})"
        )
        # Iterations should be much less than maxiter
        assert stats['last_cg_iters'] < 500, (
            "With atol=1e-14, CG should converge well before maxiter on near-zero RHS"
        )

    def test_auto_select_direct_for_small_n(self):
        """auto_select returns 'direct' for small interface systems."""
        from distributed.interface_iterative import auto_select_interface_solver

        n = 100
        S = sp.eye(n, format='csr')
        resolved = auto_select_interface_solver(n, S)
        assert resolved == 'direct', f"Expected 'direct' for n={n}, got {resolved!r}"

    def test_auto_select_cg_for_large_n(self):
        """auto_select returns 'cg' when n_interface >= threshold (explicit param)."""
        from distributed.interface_iterative import auto_select_interface_solver
        # Pass threshold explicitly (default arg is captured at definition time)
        resolved = auto_select_interface_solver(
            n_interface=10,
            S_global=sp.eye(10),
            n_interface_threshold=5,  # Force tiny threshold
        )
        assert resolved == 'cg', f"Expected 'cg' for n=10 > threshold=5, got {resolved!r}"

    def test_auto_select_default_unchanged_on_small_system(self):
        """Default auto threshold: n=100 still resolves to 'direct' (regression guard)."""
        from distributed.interface_iterative import auto_select_interface_solver
        n = 100
        resolved = auto_select_interface_solver(n)
        assert resolved == 'direct'

    # ------------------------------------------------------------------
    # Fix: tilewise matvec uses bincount scatter (not np.add.at)
    # ------------------------------------------------------------------

    def test_tilewise_matvec_bincount_correctness(self):
        """Tilewise matvec scatter via bincount gives same result as direct dense matvec.

        This is the regression guard for the np.add.at -> np.bincount fix.
        Build a 4-node system where two tiles share an interface node (node 0):
          tile A covers nodes [0, 1] with a 2x2 S_A block.
          tile B covers nodes [0, 2] with a 2x2 S_B block.
        The assembled S_global is 3x3.  The tilewise matvec must scatter-add
        S_A @ x[[0,1]] and S_B @ x[[0,2]] into the result at the correct
        (possibly repeated) global indices — exactly the bincount scatter path.
        """
        from distributed.interface_iterative import InterfaceCGSolver
        import scipy.linalg as la

        rng = np.random.default_rng(123)

        # Build per-tile dense Schur blocks (2x2 each)
        def _spd2(seed):
            A = rng.standard_normal((2, 2))
            return A @ A.T + 4 * np.eye(2)

        S_A = _spd2(1)
        S_B = _spd2(2)

        # Interface nodes: 0, 1, 2 (3 global nodes)
        # tile A → local [0,1] map to global [0, 1]
        # tile B → local [0,1] map to global [0, 2]
        n = 3
        tile_schur = {'A': S_A, 'B': S_B}
        tile_idx = {
            'A': np.array([0, 1], dtype=np.int32),
            'B': np.array([0, 2], dtype=np.int32),
        }

        # Assemble reference S_global manually
        S_ref = np.zeros((n, n))
        for tid in ['A', 'B']:
            idx = tile_idx[tid]
            Si = tile_schur[tid]
            for li, gi in enumerate(idx):
                for lj, gj in enumerate(idx):
                    S_ref[gi, gj] += Si[li, lj]
        S_ref_sp = sp.csr_matrix(S_ref)

        x = rng.standard_normal(n)

        # Direct matvec on assembled S_ref
        y_ref = S_ref @ x

        # Tilewise CG matvec (exercises bincount scatter path)
        cg_tiled = InterfaceCGSolver(
            n_interface=n,
            matvec_mode='tilewise',
            tile_schur_complements=tile_schur,
            tile_index_maps=tile_idx,
            preconditioner='none',
            rtol=1e-12,
        )
        y_tiled = cg_tiled._linear_op.matvec(x)

        np.testing.assert_allclose(
            y_tiled, y_ref, rtol=1e-12,
            err_msg="Tilewise bincount scatter matvec does not match assembled reference",
        )

    # ------------------------------------------------------------------
    # Fix: block_jacobi memory guard falls back to jacobi for large systems
    # ------------------------------------------------------------------

    def test_block_jacobi_falls_back_to_jacobi_when_memory_exceeded(self):
        """block_jacobi falls back to diagonal preconditioner when factor memory exceeds budget.

        Construction: 4-node system (all owned by one tile -> block = 4x4 dense)
        with BLOCK_JACOBI_MAX_FACTOR_BYTES monkeypatched to 0 bytes so the
        estimate always exceeds the budget.  The returned preconditioner must
        be a LinearOperator (the diagonal fallback), not None.  CG with the
        fallback must still converge to the correct answer.
        """
        import distributed.interface_iterative as ii_mod
        from distributed.interface_iterative import InterfaceCGSolver

        n = 4
        S = self._make_spd_matrix(n, seed=22)
        rhs = np.random.default_rng(5).standard_normal(n)

        # Monkeypatch budget to 0 to force the fallback unconditionally
        orig_budget = ii_mod.BLOCK_JACOBI_MAX_FACTOR_BYTES
        try:
            ii_mod.BLOCK_JACOBI_MAX_FACTOR_BYTES = 0
            cg = InterfaceCGSolver(
                n_interface=n,
                matvec_mode='assembled',
                S_global=S,
                tile_schur_complements={(0, 0): S.toarray()},
                tile_index_maps={(0, 0): np.arange(n, dtype=np.int32)},
                preconditioner='block_jacobi',
                rtol=1e-12,
            )
        finally:
            ii_mod.BLOCK_JACOBI_MAX_FACTOR_BYTES = orig_budget

        # The preconditioner must not be None (diagonal fallback should be built)
        assert cg._M is not None, (
            "block_jacobi memory-exceeded fallback should return a non-None "
            "diagonal preconditioner, not None."
        )

        # Finding 11: the downgrade must be reflected in cg.preconditioner
        # (every downstream report -- backend_info, verbose logs -- reads
        # this attribute) while the originally-requested value is preserved
        # separately so callers/logs can still say what the user asked for.
        assert cg.preconditioner == 'jacobi', (
            f"Downgraded solver must report preconditioner='jacobi' (actually "
            f"in use), not the stale requested value; got {cg.preconditioner!r}"
        )
        assert cg.requested_preconditioner == 'block_jacobi', (
            f"requested_preconditioner must retain the ORIGINALLY-requested "
            f"value even after downgrade; got {cg.requested_preconditioner!r}"
        )

        # CG should still converge correctly
        x_cg = cg(rhs)
        from scipy.sparse.linalg import spsolve
        x_direct = spsolve(S, rhs)
        np.testing.assert_allclose(x_cg, x_direct, atol=1e-8,
                                   err_msg="CG with fallback diagonal preconditioner must converge")

    def test_block_jacobi_memory_estimate_logged(self, caplog):
        """block_jacobi logs a WARNING (not error) when memory budget is exceeded."""
        import logging
        import distributed.interface_iterative as ii_mod
        from distributed.interface_iterative import InterfaceCGSolver

        n = 4
        S = self._make_spd_matrix(n, seed=22)

        orig_budget = ii_mod.BLOCK_JACOBI_MAX_FACTOR_BYTES
        try:
            ii_mod.BLOCK_JACOBI_MAX_FACTOR_BYTES = 0
            with caplog.at_level(logging.WARNING, logger='distributed.interface_iterative'):
                InterfaceCGSolver(
                    n_interface=n,
                    matvec_mode='assembled',
                    S_global=S,
                    tile_schur_complements={(0, 0): S.toarray()},
                    tile_index_maps={(0, 0): np.arange(n, dtype=np.int32)},
                    preconditioner='block_jacobi',
                    rtol=1e-12,
                )
        finally:
            ii_mod.BLOCK_JACOBI_MAX_FACTOR_BYTES = orig_budget

        # A WARNING mentioning the memory fallback must appear in the log
        relevant = [
            r for r in caplog.records
            if r.levelno == logging.WARNING and 'Block-Jacobi' in r.message
        ]
        assert relevant, (
            "Expected a WARNING log about block_jacobi memory budget being exceeded; "
            f"got records: {[r.message for r in caplog.records]}"
        )

    def test_preflight_estimate_includes_materialization_peak(self):
        """Finding 10 (round 1, PLAUSIBLE) / Finding 8 (round 2): the
        pre-flight memory guard (est_factor_bytes = Sum(k_i^2)*8) modeled
        only the cho-factor storage, but the Stage 3 materialization loop
        transiently holds FOUR k x k arrays at once while symmetrizing the
        block being processed (the not-yet-replaced cho factor, the fresh
        cho_solve inverse, the `Sinv + Sinv.T` temporary, and the `0.5 *
        (...)` result) -- three arrays beyond the counted cho factor, not
        two -- a peak the pre-Stage-3 apply-time cho_solve design never had.

        Single k=100 block: Sum(k^2)*8 = 80,000 bytes (old estimate).
        New estimate adds 3*k_max^2*8 = 240,000 bytes -> 320,000 total.
        A budget of 100,000 sits strictly between the old and new totals --
        old code would NOT downgrade (80,000 <= 100,000); the fixed code
        MUST downgrade (320,000 > 100,000).
        """
        from distributed.interface_iterative import InterfaceCGSolver

        n = 100
        S = self._make_spd_matrix(n, seed=17)

        cg = InterfaceCGSolver(
            n_interface=n,
            matvec_mode='assembled',
            S_global=S,
            tile_schur_complements={(0, 0): S.toarray()},
            tile_index_maps={(0, 0): np.arange(n, dtype=np.int32)},
            preconditioner='block_jacobi',
            rtol=1e-10,
            block_jacobi_max_bytes=100_000,
        )
        try:
            assert cg.preconditioner == 'jacobi', (
                f"expected the pre-flight guard (now including the "
                f"materialization peak) to downgrade block_jacobi -> "
                f"jacobi at this budget; got preconditioner="
                f"{cg.preconditioner!r} (pre-fix: the old Sum(k_i^2)*8-only "
                f"estimate of 80,000 bytes sits under the 100,000 budget, "
                f"so no downgrade would fire)"
            )
        finally:
            cg.close()


# ──────────────────────────────────────────────────────────────────────
# 2. CG vs direct on two-tile fixture (DC and transient)
# ──────────────────────────────────────────────────────────────────────


class TestCGVsDirectTwoTile:
    """CG and direct produce matching results on the standard two-tile fixture."""

    def _run_dc_with_solver(self, interface_solver, matvec_mode='assembled'):
        """Run a DC solve using _build_dc_model with given interface_solver."""
        from distributed.solver import DistributedDDMSolver

        model = _build_dc_model(
            interface_solver=interface_solver,
            interface_matvec_mode=matvec_mode,
            interface_cg_rtol=1e-12,
        )
        solver = DistributedDDMSolver(model)
        ctx = solver.prepare()
        result = solver.solve_dc(ctx)
        ctx.release()
        return result.flatten()

    def test_cg_assembled_dc_agrees_with_direct(self):
        """CG assembled mode DC matches direct to 1e-8."""
        v_direct = self._run_dc_with_solver('direct')
        v_cg = self._run_dc_with_solver('cg', matvec_mode='assembled')
        common = set(v_direct) & set(v_cg)
        max_diff = max(abs(v_direct[k] - v_cg[k]) for k in common)
        assert max_diff <= 1e-8, f"CG assembled vs direct DC max |dV| = {max_diff:.3e} > 1e-8"

    def test_cg_tilewise_dc_agrees_with_direct(self):
        """CG tilewise mode DC matches direct to 1e-8."""
        v_direct = self._run_dc_with_solver('direct')
        v_cg = self._run_dc_with_solver('cg', matvec_mode='tilewise')
        common = set(v_direct) & set(v_cg)
        max_diff = max(abs(v_direct[k] - v_cg[k]) for k in common)
        assert max_diff <= 1e-8, f"CG tilewise vs direct DC max |dV| = {max_diff:.3e} > 1e-8"

    def test_assembled_and_tilewise_agree(self):
        """Assembled and tilewise modes agree within 1e-8."""
        v_asm = self._run_dc_with_solver('cg', matvec_mode='assembled')
        v_tiled = self._run_dc_with_solver('cg', matvec_mode='tilewise')
        common = set(v_asm) & set(v_tiled)
        max_diff = max(abs(v_asm[k] - v_tiled[k]) for k in common)
        assert max_diff <= 1e-8, f"Assembled vs tilewise max |dV| = {max_diff:.3e} > 1e-8"

    def test_cg_context_has_cg_solver_attribute(self):
        """Context prepared with CG has a non-None _cg_solver."""
        from distributed.solver import DistributedDDMSolver

        model = _build_dc_model(interface_solver='cg')
        solver = DistributedDDMSolver(model)
        ctx = solver.prepare()
        cg_solver = getattr(ctx, '_cg_solver', None)
        assert cg_solver is not None, "CG mode should set ctx._cg_solver"
        assert ctx._interface_solver_mode == 'cg'
        ctx.release()

    def test_direct_context_has_no_cg_solver(self):
        """Context prepared with direct mode has _cg_solver=None."""
        from distributed.solver import DistributedDDMSolver

        model = _build_dc_model(interface_solver='direct')
        solver = DistributedDDMSolver(model)
        ctx = solver.prepare()
        assert ctx._interface_solver_mode == 'direct'
        assert getattr(ctx, '_cg_solver', None) is None
        ctx.release()


# ──────────────────────────────────────────────────────────────────────
# 3. Auto-select + monkeypatching
# ──────────────────────────────────────────────────────────────────────


class TestAutoSelectBoundaries:
    """Auto-select resolves correctly; monkeypatch thresholds work."""

    def test_auto_resolves_to_direct_on_small_model(self):
        """Auto mode resolves to 'direct' on small two-tile fixture (n_interface = 1)."""
        from distributed.solver import DistributedDDMSolver

        model = _build_dc_model(interface_solver='auto')
        solver = DistributedDDMSolver(model)
        ctx = solver.prepare()
        mode = getattr(ctx, '_interface_solver_mode', None)
        assert mode == 'direct', (
            f"Expected 'direct' for small two-tile interface, got {mode!r}"
        )
        ctx.release()

    def test_monkeypatch_forces_cg(self):
        """Patching interface_iterative.auto_select_interface_solver forces auto->cg.

        The lazy import inside _factor_dc_context does `from .interface_iterative
        import auto_select_interface_solver` on every call, so patching the
        module attribute propagates to the next call.
        """
        from distributed.solver import DistributedDDMSolver
        import distributed.interface_iterative as ii_mod

        orig_fn = ii_mod.auto_select_interface_solver

        def _always_cg(n, S=None, **kwargs):
            return 'cg'

        try:
            ii_mod.auto_select_interface_solver = _always_cg
            model = _build_dc_model(interface_solver='auto')
            solver = DistributedDDMSolver(model)
            ctx = solver.prepare()
            mode = getattr(ctx, '_interface_solver_mode', None)
            assert mode == 'cg', (
                f"With forced auto='cg', expected 'cg', got {mode!r}"
            )
            ctx.release()
        finally:
            ii_mod.auto_select_interface_solver = orig_fn

    def test_explicit_direct_stays_direct(self):
        """Explicit 'direct' setting always uses direct."""
        from distributed.solver import DistributedDDMSolver

        model = _build_dc_model(interface_solver='direct')
        solver = DistributedDDMSolver(model)
        ctx = solver.prepare()
        assert ctx._interface_solver_mode == 'direct'
        ctx.release()

    def test_explicit_cg_stays_cg(self):
        """Explicit 'cg' setting always uses CG."""
        from distributed.solver import DistributedDDMSolver

        model = _build_dc_model(interface_solver='cg')
        solver = DistributedDDMSolver(model)
        ctx = solver.prepare()
        assert ctx._interface_solver_mode == 'cg'
        assert getattr(ctx, '_cg_solver', None) is not None
        ctx.release()


# ──────────────────────────────────────────────────────────────────────
# 4. Preconditioner effectiveness
# ──────────────────────────────────────────────────────────────────────


class TestPreconditionerEffectiveness:
    """Block-Jacobi not worse than no preconditioner on two-tile fixture."""

    def _get_cg_solver_from_model(self, preconditioner):
        """Prepare a CG context and return ctx."""
        from distributed.solver import DistributedDDMSolver

        model = _build_dc_model(
            interface_solver='cg',
            interface_preconditioner=preconditioner,
            interface_cg_rtol=1e-10,
        )
        solver = DistributedDDMSolver(model)
        ctx = solver.prepare()
        return ctx

    def test_block_jacobi_not_worse_than_none_on_fixture(self):
        """Block-Jacobi iterations <= no-preconditioner on minimal fixture."""
        ctx_bj = self._get_cg_solver_from_model('block_jacobi')
        ctx_none = self._get_cg_solver_from_model('none')

        cg_bj = getattr(ctx_bj, '_cg_solver', None)
        cg_none = getattr(ctx_none, '_cg_solver', None)

        assert cg_bj is not None and cg_none is not None, (
            "Both CG contexts should have _cg_solver populated"
        )

        rhs = ctx_bj.rhs_dirichlet_interface.copy()

        stats_bj: Dict[str, Any] = {}
        stats_none: Dict[str, Any] = {}
        cg_bj._stats = stats_bj
        cg_none._stats = stats_none
        cg_bj.reset_warm_start()
        cg_none.reset_warm_start()

        cg_bj(rhs)
        cg_none(rhs)

        iters_bj = stats_bj['last_cg_iters']
        iters_none = stats_none['last_cg_iters']

        assert iters_bj <= iters_none, (
            f"block_jacobi ({iters_bj} iters) should not be worse than "
            f"none ({iters_none} iters)"
        )
        ctx_bj.release()
        ctx_none.release()


# ──────────────────────────────────────────────────────────────────────
# 5. Warm start across transient steps (uses transient fixture)
# ──────────────────────────────────────────────────────────────────────


class TestTransientWarmStart:
    """CG warm start does not increase total iterations over consecutive solves.

    Tests directly against InterfaceCGSolver with a sequence of similar RHS
    vectors (simulating successive time steps of a slowly-varying interface
    system), which avoids needing a full transient-capable fixture.
    """

    def _make_spd_matrix(self, n, seed=42):
        rng = np.random.default_rng(seed)
        A = rng.standard_normal((n, n))
        return sp.csr_matrix(A @ A.T + n * np.eye(n))

    def _count_iters_sequence(self, n_steps, use_warm_start, seed=0):
        """Simulate n_steps consecutive CG solves with slowly-varying RHS.

        Returns the total CG iterations across all steps.
        """
        from distributed.interface_iterative import InterfaceCGSolver

        n = 20
        S = self._make_spd_matrix(n, seed=42)
        rng = np.random.default_rng(seed)

        # Base RHS + small per-step perturbation (simulating slow time evolution)
        rhs_base = rng.standard_normal(n)
        perturbations = [0.02 * rng.standard_normal(n) for _ in range(n_steps)]

        stats: Dict[str, Any] = {}
        cg = InterfaceCGSolver(
            n_interface=n, matvec_mode='assembled',
            S_global=S, preconditioner='none', rtol=1e-12, stats_dict=stats,
        )

        total_iters = 0
        for step in range(n_steps):
            rhs = rhs_base + perturbations[step]
            if not use_warm_start:
                cg.reset_warm_start()
            cg(rhs)
            total_iters += stats['last_cg_iters']

        return total_iters

    def test_warm_start_strictly_reduces_total_iterations(self):
        """Total CG iterations with warm start STRICTLY LESS than without, over 10 steps.

        Uses a synthetic RHS sequence (slowly-varying, seed-fixed) to verify the
        warm-start mechanism at unit-test speed.  The complementary
        TestNetlistSampledCG.test_warm_start_strictly_less_over_transient_steps
        drives the actual solve_transient loop on the real netlist_sampled system.

        Verified analytically: the fixed seed=0 sequence with n=20 and 0.02
        perturbation gives warm=171, cold=180 — strict inequality with ample
        margin (9 iters).  Checked robust across seeds 0-9.
        """
        n_steps = 10
        iters_warm = self._count_iters_sequence(n_steps, use_warm_start=True)
        iters_cold = self._count_iters_sequence(n_steps, use_warm_start=False)

        assert iters_warm < iters_cold, (
            f"Warm start should use STRICTLY FEWER total iterations than cold: "
            f"warm={iters_warm} vs cold={iters_cold} over {n_steps} steps"
        )

    def test_warm_start_second_step_does_not_regress(self):
        """Warm start on step 2 (starting from step 1 solution) does not regress."""
        from distributed.interface_iterative import InterfaceCGSolver

        n = 20
        S = self._make_spd_matrix(n)
        rhs1 = np.random.default_rng(10).standard_normal(n)
        rhs2 = rhs1 + 0.05 * np.random.default_rng(11).standard_normal(n)

        stats_cold: Dict[str, Any] = {}
        cg_cold = InterfaceCGSolver(n_interface=n, matvec_mode='assembled',
                                    S_global=S, preconditioner='none', rtol=1e-12,
                                    stats_dict=stats_cold)
        cg_cold(rhs1)
        cg_cold.reset_warm_start()
        cg_cold(rhs2)
        iters_cold = stats_cold['last_cg_iters']

        stats_warm: Dict[str, Any] = {}
        cg_warm = InterfaceCGSolver(n_interface=n, matvec_mode='assembled',
                                    S_global=S, preconditioner='none', rtol=1e-12,
                                    stats_dict=stats_warm)
        cg_warm(rhs1)    # x0 is now solution of rhs1
        cg_warm(rhs2)    # warm start from rhs1 solution
        iters_warm = stats_warm['last_cg_iters']

        # Step 2 with warm start should not be worse than cold start
        assert iters_warm <= iters_cold, (
            f"Warm-start step 2 ({iters_warm} iters) should not exceed "
            f"cold ({iters_cold} iters)"
        )


# ──────────────────────────────────────────────────────────────────────
# 6. Save / load / refactor in CG mode
# ──────────────────────────────────────────────────────────────────────


class TestCGSaveLoadRefactor:
    """CG mode persists through save/load/refactor cycles."""

    def test_save_load_refactor_direct_mode(self, tmp_path):
        """Direct mode save/load/refactor: existing behavior unchanged.

        After load() + refactor() (coordinator LU), workers also need their
        tile factors rebuilt (factor()) before the next solve.
        """
        from distributed.solver import DistributedDDMSolver

        model = _build_dc_model(interface_solver='direct')
        solver = DistributedDDMSolver(model)
        ctx = solver.prepare()
        v_before = solver.solve_dc(ctx).flatten()

        path = str(tmp_path / 'dc_ctx.pkl')
        ctx.save(path)
        ctx.release()  # clears coordinator LU + worker tile factors

        ctx2 = type(ctx).load(model, path)
        assert not ctx2.is_factored
        # Rebuild coordinator LU from saved S_global
        ctx2.refactor()
        assert ctx2.is_factored
        assert getattr(ctx2, '_interface_solver_mode', 'direct') == 'direct'
        # Rebuild worker tile factors (cleared by release())
        model.backend.call_all(model.workers, 'factor_and_compute_schur')

        v_after = solver.solve_dc(ctx2).flatten()
        ctx2.release()

        common = set(v_before) & set(v_after)
        assert common, "No shared nodes to compare"
        for k in common:
            assert abs(v_before[k] - v_after[k]) <= 1e-10, (
                f"Node {k}: before={v_before[k]:.8f}, after={v_after[k]:.8f}"
            )

    def test_save_load_refactor_cg_mode_assembled(self, tmp_path):
        """CG assembled mode: load+refactor reconstructs a CG callable.

        For CG mode, refactor() rebuilds the CG LinearOperator from S_global.
        Workers still need their tile factors rebuilt separately.
        """
        from distributed.solver import DistributedDDMSolver

        # Pin rtol=1e-12 (Stage 1b): this test asserts tight (<=1e-8 V)
        # before/after agreement, so it must not depend on the production
        # default (1e-8) drifting in the future.
        model = _build_dc_model(interface_solver='cg', interface_cg_rtol=1e-12)
        solver = DistributedDDMSolver(model)
        ctx = solver.prepare()
        v_before = solver.solve_dc(ctx).flatten()

        path = str(tmp_path / 'dc_ctx_cg.pkl')
        ctx.save(path)
        ctx.release()

        ctx2 = type(ctx).load(model, path)
        assert not ctx2.is_factored
        assert getattr(ctx2, '_interface_solver_mode', None) == 'cg'
        ctx2.refactor()  # For CG assembled, refactor reconstructs CG from S_global
        assert ctx2.is_factored
        assert getattr(ctx2, '_interface_solver_mode', None) == 'cg'
        # Rebuild worker tile factors
        model.backend.call_all(model.workers, 'factor_and_compute_schur')

        v_after = solver.solve_dc(ctx2).flatten()
        ctx2.release()

        common = set(v_before) & set(v_after)
        assert common, "No shared nodes to compare"
        for k in common:
            assert abs(v_before[k] - v_after[k]) <= 1e-8, (
                f"Node {k}: before={v_before[k]:.8f}, after={v_after[k]:.8f}"
            )

    def test_save_load_refactor_cg_block_jacobi_rebuilds_preconditioner(
        self, tmp_path,
    ):
        """Finding 1 regression test: refactor() after load() must rebuild
        the block_jacobi preconditioner, not silently degrade to
        unpreconditioned CG.

        Before the fix, _refactor_dc_context() called build_interface_solver
        WITHOUT tile_index_maps, so InterfaceCGSolver._build_block_jacobi()
        had neither tile_index_maps nor tile_schur_complements to assign
        ownership from and returned None (no preconditioner at all) --
        silently, since matvec_mode='assembled' still solves correctly, just
        slower / with more iterations (or a strict=True convergence failure
        on a larger interface).  This test asserts the rebuilt CG solver's
        preconditioner operator (`_M`) is present after refactor(), and that
        the solve still matches the pre-save answer.
        """
        from distributed.solver import DistributedDDMSolver

        model = _build_dc_model(
            interface_solver='cg', interface_cg_rtol=1e-12,
            interface_preconditioner='block_jacobi',
        )
        solver = DistributedDDMSolver(model)
        ctx = solver.prepare()
        assert ctx._cg_solver is not None
        assert ctx._cg_solver._M is not None, (
            "Sanity check: the initial factor() build must have a non-None "
            "preconditioner operator (block_jacobi) to begin with."
        )
        v_before = solver.solve_dc(ctx).flatten()

        path = str(tmp_path / 'dc_ctx_cg_bj.pkl')
        ctx.save(path)
        ctx.release()

        ctx2 = type(ctx).load(model, path)
        ctx2.refactor()
        model.backend.call_all(model.workers, 'factor_and_compute_schur')

        assert ctx2._cg_solver is not None
        assert ctx2._cg_solver.preconditioner == 'block_jacobi', (
            "refactor() must not have silently downgraded the requested "
            f"preconditioner; got {ctx2._cg_solver.preconditioner!r}"
        )
        assert ctx2._cg_solver._M is not None, (
            "refactor() must rebuild the block_jacobi preconditioner "
            "operator (tile_index_maps must reach build_interface_solver), "
            "not silently fall back to unpreconditioned CG."
        )

        v_after = solver.solve_dc(ctx2).flatten()
        ctx2.release()

        common = set(v_before) & set(v_after)
        assert common, "No shared nodes to compare"
        for k in common:
            assert abs(v_before[k] - v_after[k]) <= 1e-8, (
                f"Node {k}: before={v_before[k]:.8f}, after={v_after[k]:.8f}"
            )

    def test_save_load_refactor_cg_tilewise_fp32_still_works(self, tmp_path):
        """Finding 3 regression: refactor() re-gathers tile_schur_complements
        fresh from workers via _gather_kept_tile_schur_streaming (a brand
        new dict each call) rather than reading any retained state on the
        InterfaceCGSolver instance, so freeing the fp64 originals in place
        inside one InterfaceCGSolver's __init__ (Finding 3's fix) cannot
        leave a later refactor() with stale/missing fp64 data. This
        exercises that re-gather path end-to-end with
        interface_matvec_dtype='float32'.
        """
        from distributed.solver import DistributedDDMSolver

        model = _build_dc_model(
            interface_solver='cg', interface_cg_rtol=1e-6,
            interface_preconditioner='block_jacobi',
            interface_matvec_mode='tilewise',
            interface_matvec_dtype='float32',
        )
        solver = DistributedDDMSolver(model)
        ctx = solver.prepare()
        assert ctx._cg_solver is not None
        assert ctx._cg_solver.matvec_dtype == np.dtype(np.float32)
        v_before = solver.solve_dc(ctx).flatten()

        path = str(tmp_path / 'dc_ctx_cg_bj_fp32.pkl')
        ctx.save(path)
        ctx.release()

        ctx2 = type(ctx).load(model, path)
        ctx2.refactor()
        model.backend.call_all(model.workers, 'factor_and_compute_schur')

        assert ctx2._cg_solver is not None
        assert ctx2._cg_solver.matvec_dtype == np.dtype(np.float32)
        assert ctx2._cg_solver._M is not None, (
            "refactor() must still rebuild the block_jacobi preconditioner "
            "with fp32 tilewise storage"
        )

        v_after = solver.solve_dc(ctx2).flatten()
        ctx2.release()

        common = set(v_before) & set(v_after)
        assert common, "No shared nodes to compare"
        for k in common:
            assert abs(v_before[k] - v_after[k]) <= 1e-6, (
                f"Node {k}: before={v_before[k]:.8f}, after={v_after[k]:.8f}"
            )

    def test_save_load_refactor_transient_cg_block_jacobi_rebuilds_preconditioner(
        self, tmp_path,
    ):
        """Finding 1 regression test (transient counterpart): the same
        missing-tile_index_maps bug existed in _refactor_transient_context.
        Lighter than the DC version above -- checks the rebuilt
        preconditioner operator directly rather than a full solve_transient
        round-trip (which needs sources/dc_context plumbing out of scope
        here); the DC test above already proves the solve-level equivalence
        for the shared build_interface_solver() factory both paths route
        through.
        """
        model = _build_two_tile_distributed_model(package_cap_edges=[])
        model.settings.update({
            'interface_solver': 'cg',
            'interface_preconditioner': 'block_jacobi',
            'interface_cg_rtol': 1e-12,
        })
        from distributed.solver import DistributedDDMSolver

        solver = DistributedDDMSolver(model)
        trans_ctx = solver.prepare_transient(dt=1e-10, method='be')
        assert trans_ctx._cg_solver is not None
        assert trans_ctx._cg_solver._M is not None, (
            "Sanity check: the initial factor() build must have a non-None "
            "preconditioner operator (block_jacobi) to begin with."
        )

        path = str(tmp_path / 'trans_ctx_cg_bj.pkl')
        trans_ctx.save(path)
        trans_ctx.release()

        trans_ctx2 = type(trans_ctx).load(model, path)
        # refactor() only rebuilds the coordinator-side interface solve
        # callable from the saved S_global; it does not require worker tile
        # factors to be rebuilt first (that is a separate concern for an
        # actual solve_transient() call, out of scope for this test).
        trans_ctx2.refactor()

        assert trans_ctx2._cg_solver is not None
        assert trans_ctx2._cg_solver.preconditioner == 'block_jacobi', (
            "refactor() must not have silently downgraded the requested "
            f"preconditioner; got {trans_ctx2._cg_solver.preconditioner!r}"
        )
        assert trans_ctx2._cg_solver._M is not None, (
            "refactor() must rebuild the block_jacobi preconditioner "
            "operator (tile_index_maps must reach build_interface_solver), "
            "not silently fall back to unpreconditioned CG."
        )
        trans_ctx2.release()


# ──────────────────────────────────────────────────────────────────────
# 7. CG vs flat oracle (DC |dV| <= 1e-8)
# ──────────────────────────────────────────────────────────────────────


class TestCGVsFlatOracle:
    """CG distributed DC matches flat solver oracle to 1e-8."""

    def test_cg_dc_vs_flat_oracle(self):
        """CG DC on two-tile fixture matches a direct flat solve to 1e-8."""
        import scipy.sparse.linalg as spla
        from distributed.solver import DistributedDDMSolver

        # ----------------------------------------------------------------
        # Flat oracle: hand-build the conductance matrix for the _build_dc_model
        # topology and solve directly.
        #
        # Nodes (unknowns): n1, B, n2, n3    (pad_v at Vdd=1.0V is eliminated)
        # Resistive connections:
        #   n1 -- B:     1 mS
        #   B -- n2:     2 mS
        #   n2 -- 0:     1 mS (ground)
        #   B -- n3:     3 mS
        #   n3 -- 0:     2 mS (ground)
        #   pad_v -- B: 10 mS (package; Dirichlet pad_v = 1.0 V)
        # Currents: n2 draws 0.5 mA, n3 draws 0.3 mA
        # ----------------------------------------------------------------
        nodes = ['n1', 'B', 'n2', 'n3']
        idx = {n: i for i, n in enumerate(nodes)}
        n = len(nodes)
        G = np.zeros((n, n))

        def add_edge(u, v, g):
            if u in idx and v in idx:
                G[idx[u], idx[u]] += g
                G[idx[v], idx[v]] += g
                G[idx[u], idx[v]] -= g
                G[idx[v], idx[u]] -= g
            elif u in idx:
                G[idx[u], idx[u]] += g
            elif v in idx:
                G[idx[v], idx[v]] += g

        add_edge('n1', 'B', 1.0)
        add_edge('B', 'n2', 2.0)
        add_edge('n2', '0', 1.0)   # n2 to ground
        add_edge('B', 'n3', 3.0)
        add_edge('n3', '0', 2.0)   # n3 to ground
        # Package: pad_v eliminated; contributes to B diagonal + RHS
        B_pad_cond = 10.0
        G[idx['B'], idx['B']] += B_pad_cond

        rhs = np.zeros(n)
        # Dirichlet contribution: G_ip * vdd (vdd = 1.0 V)
        rhs[idx['B']] += B_pad_cond * 1.0
        # Current sources (positive = sink → negative RHS in nodal equations)
        rhs[idx['n2']] -= 0.5
        rhs[idx['n3']] -= 0.3

        G_sp = sp.csr_matrix(G)
        v_flat = spla.spsolve(G_sp, rhs)
        v_flat_dict = {name: v_flat[i] for name, i in idx.items()}
        v_flat_dict['pad_v'] = 1.0

        # ----------------------------------------------------------------
        # CG distributed solve
        # ----------------------------------------------------------------
        model = _build_dc_model(interface_solver='cg', interface_cg_rtol=1e-12)
        solver = DistributedDDMSolver(model)
        ctx = solver.prepare()
        res = solver.solve_dc(ctx)
        ctx.release()
        v_cg = res.flatten()

        # Compare shared nodes
        for node in ['n1', 'n2', 'n3', 'B']:
            if node in v_cg:
                diff = abs(v_cg[node] - v_flat_dict[node])
                assert diff <= 1e-8, (
                    f"Node {node}: CG={v_cg[node]:.8f}, flat={v_flat_dict[node]:.8f}, "
                    f"diff={diff:.3e} > 1e-8"
                )


# ──────────────────────────────────────────────────────────────────────
# 8. Netlist sampled: auto resolves to 'direct'; CG DC end-to-end
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(
    not SAMPLED_PKL_AVAILABLE,
    reason="netlist_sampled/distributed_pkl not available"
)
class TestNetlistSampledCG:
    """Tests on the netlist_sampled benchmark."""

    def _load_model(self, interface_solver='auto', **extra_settings):
        from distributed.model import create_distributed_model, load_distributed_partitions
        pkl_dir = str(_get_fixtures_dir())
        bundle = load_distributed_partitions(pkl_dir)
        model = create_distributed_model(bundle, backend='local')
        model.settings['interface_solver'] = interface_solver
        model.settings.update(extra_settings)
        return model

    def test_auto_resolves_to_direct_on_sampled(self):
        """Auto mode resolves to 'direct' on netlist_sampled (n_interface ~2-4K)."""
        from distributed.solver import DistributedDDMSolver

        model = self._load_model(interface_solver='auto')
        solver = DistributedDDMSolver(model)
        ctx = solver.prepare()
        mode = getattr(ctx, '_interface_solver_mode', None)
        ctx.release()
        model.shutdown()

        assert mode == 'direct', (
            f"Auto should resolve to 'direct' for netlist_sampled "
            f"(small interface), got {mode!r}"
        )

    def test_cg_dc_agrees_with_direct_on_sampled(self):
        """CG DC on netlist_sampled matches direct to 1e-8."""
        from distributed.solver import DistributedDDMSolver
        import time as _time

        def _run_dc(iface_solver, **extra):
            m = self._load_model(iface_solver, **extra)
            s = DistributedDDMSolver(m)
            t0 = _time.perf_counter()
            ctx = s.prepare()
            res = s.solve_dc(ctx)
            elapsed = _time.perf_counter() - t0
            mode = getattr(ctx, '_interface_solver_mode', 'unknown')
            cg_solver = getattr(ctx, '_cg_solver', None)
            n_iters = cg_solver.total_iterations if cg_solver else 0
            ctx.release()
            m.shutdown()
            return res.flatten(), elapsed, mode, n_iters

        v_direct, t_direct, mode_direct, _ = _run_dc('direct')
        v_cg, t_cg, mode_cg, n_iters_cg = _run_dc(
            'cg',
            interface_matvec_mode='assembled',
            interface_preconditioner='block_jacobi',
            interface_cg_rtol=1e-12,  # tighter rtol for 1e-8 voltage tolerance
        )

        assert mode_direct == 'direct'
        assert mode_cg == 'cg'

        common = set(v_direct) & set(v_cg)
        assert common, "No shared nodes between direct and CG results"

        max_diff = max(abs(v_direct[n] - v_cg[n]) for n in common)
        assert max_diff <= 1e-8, (
            f"CG vs direct on netlist_sampled: max |dV| = {max_diff:.3e} > 1e-8"
        )

        print(
            f"\nNetlist sampled CG DC:"
            f"\n  direct:  {t_direct:.3f}s"
            f"\n  CG:      {t_cg:.3f}s, {n_iters_cg} total iters"
            f"\n  max |dV| direct vs CG: {max_diff:.3e}"
        )

    def test_cg_transient_agrees_with_direct_on_sampled(self):
        """Finding 10 (b): transient counterpart of
        test_cg_dc_agrees_with_direct_on_sampled -- forced-CG (block_jacobi,
        pinned rtol=1e-12) vs direct, comparing per-node PEAK IR-drop over a
        short transient run, at the same tight-rtol exactness tolerance
        (1e-8 V) used by the DC test above.

        This is the "equivalent coverage in tests/distributed" called for by
        finding 10(b) in lieu of a forced-CG entry in
        tests/validation/test_equivalence.py's SOLVER_VARIANTS: that list is
        threaded as **solver_kw directly into solve_quasi_static()/
        solve_transient() (e.g. use_step_columns), but interface_solver is a
        model.settings key that must be set BEFORE the module-scoped
        sampled_setup fixture's single shared dc_ctx = dist_solver.prepare()
        call -- which every parametrized variant in that suite reuses.
        Threading a CG variant through SOLVER_VARIANTS would either require
        rebuilding a whole separate model per test (duplicating most of the
        ~100-line fixture) or re-parametrizing the module-scoped fixture
        itself (affecting every non-CG test in the module for a flag that
        only this coverage cares about) -- structurally out of place there.
        The self-contained _load_model()/_run_transient() helpers in this
        class already build a fresh model per call, which is exactly what a
        CG variant needs, so the equivalent guard lives here instead.
        """
        from distributed.solver import DistributedDDMSolver

        N_STEPS = 5
        DT = 1e-10
        T_END = N_STEPS * DT

        def _run_transient(iface_solver, **extra):
            m = self._load_model(iface_solver, **extra)
            s = DistributedDDMSolver(m)
            dc_ctx = s.prepare()
            trans_ctx = s.prepare_transient(dt=DT, method='be')
            sources = s.preprocess_sources(time_step=DT, t_end=T_END, smooth=False)
            result = s.solve_transient(
                trans_ctx, dc_context=dc_ctx,
                smoothed_sources=sources, t_end=T_END,
            )
            peaks = result.as_flat()
            mode = getattr(trans_ctx, '_interface_solver_mode', 'unknown')
            dc_ctx.release()
            trans_ctx.release()
            m.shutdown()
            return peaks, mode

        peaks_direct, mode_direct = _run_transient('direct')
        peaks_cg, mode_cg = _run_transient(
            'cg',
            interface_matvec_mode='assembled',
            interface_preconditioner='block_jacobi',
            interface_cg_rtol=1e-12,  # tight rtol -> tight exactness tolerance
        )

        assert mode_direct == 'direct'
        assert mode_cg == 'cg'

        common = set(peaks_direct) & set(peaks_cg)
        assert common, "No shared nodes between direct and CG transient results"

        max_diff = max(
            abs(peaks_direct[n][0] - peaks_cg[n][0]) for n in common
        )
        assert max_diff <= 1e-8, (
            f"CG (rtol=1e-12) vs direct transient peak IR-drop on "
            f"netlist_sampled: max |d(peak_drop)| = {max_diff:.3e} > 1e-8 V"
        )

    def test_rtol_1e8_dc_error_below_20uV(self):
        """Stage 1b: the production DEFAULT rtol (1e-8, not pinned) keeps CG
        DC voltage error bounded vs the direct reference on netlist_sampled.

        This directly guards the accuracy of the new default against
        regressions in the CG/preconditioner path -- unlike
        test_cg_dc_agrees_with_direct_on_sampled (which pins rtol=1e-12 for a
        tighter 1e-8 V bound), this test leaves interface_cg_rtol unset so it
        exercises exactly the value a real user gets with no override.

        Threshold note: docs Sec 7.7 (Stage 0 sweep) reports 166 nV max error
        at rtol=1e-8 on the *BRCM-class 107-tile proxy* (190,867-node
        interface, ~105 mV signal) -- that netlist is not available on this
        dev host (see CLAUDE.md benchmark snapshot). netlist_sampled here is
        a much smaller fixture (n_interface=2015) with its own conditioning;
        empirically it measures ~7.9 uV max|dV| at the default rtol (see
        module docstring point 1 / test_cg_dc_agrees_with_direct_on_sampled
        for the tight-rtol=1e-12 comparison). The 20 uV bound below is this
        fixture's own empirical figure plus >2x headroom -- it is a
        regression gate for *this* proxy, not a re-assertion of the BRCM
        166 nV number, which has no standing test on this host.
        """
        from distributed.solver import DistributedDDMSolver

        # Matches TestStage1DefaultRtol's pinned expectation for the
        # production default (Stage 0 sweep: 166 nV max error on the BRCM
        # proxy, docs Sec 7.7; see threshold note above for netlist_sampled).
        expected_default_rtol = 1e-8

        def _run_dc(iface_solver, **extra):
            m = self._load_model(iface_solver, **extra)
            s = DistributedDDMSolver(m)
            ctx = s.prepare()
            res = s.solve_dc(ctx)
            resolved_rtol = ctx._cg_solver.rtol if ctx._cg_solver else None
            ctx.release()
            m.shutdown()
            return res.flatten(), resolved_rtol

        v_direct, _ = _run_dc('direct')
        # No interface_cg_rtol override -- exercises the production default.
        v_cg, resolved_rtol = _run_dc(
            'cg',
            interface_matvec_mode='assembled',
            interface_preconditioner='block_jacobi',
        )

        assert resolved_rtol == expected_default_rtol, (
            f"Expected the resolved default rtol to be {expected_default_rtol:.1e}, "
            f"got {resolved_rtol!r} -- default may have drifted, invalidating "
            f"this accuracy guard."
        )

        common = set(v_direct) & set(v_cg)
        assert common, "No shared nodes between direct and CG results"

        max_diff = max(abs(v_direct[n] - v_cg[n]) for n in common)
        assert max_diff <= 2e-5, (
            f"CG (default rtol={resolved_rtol:.1e}) vs direct on "
            f"netlist_sampled: max |dV| = {max_diff:.3e} V > 2e-5 V (20 uV) -- "
            f"the production default rtol=1e-8 is no longer accurate enough "
            f"(empirical baseline on this fixture is ~7.9 uV)."
        )

    def test_rtol_1e8_transient_error_below_27uV(self):
        """Finding 10 (test-coverage gap): a transient-mode counterpart to
        test_rtol_1e8_dc_error_below_20uV above.  The DC test guards the CG
        default rtol against the DC voltage error only; per-step CG error at
        the production default feeds forward through the warm-start x0, the
        interior recovery, and the RC history terms (C*v_old on next step's
        RHS), so it can compound differently over a transient run than a
        single DC solve does.  This test runs a short (10-step, BE) transient
        on netlist_sampled with the interface_cg_rtol setting left UNSET
        (exercising exactly the production default, like the DC test) and
        compares the per-node PEAK IR-drop (max over the whole transient,
        via DistributedTransientResult.as_flat()) against a direct-solver
        reference over the same window.

        Measured baseline (this fixture, 10 steps @ dt=100ps, BE,
        n_interface=2015): max|d(peak_drop)| ~= 10.7 uV, mean ~= 0.18 uV.
        The 27 uV bound below is ~2.5x that measured baseline -- a
        regression gate for *this* proxy (netlist_sampled), analogous to
        the DC test's 20 uV bound derived from its own ~7.9 uV baseline.
        """
        from distributed.solver import DistributedDDMSolver

        N_STEPS = 10
        DT = 1e-10  # 100 ps per step
        T_END = N_STEPS * DT

        def _run_transient(iface_solver, **extra):
            m = self._load_model(iface_solver, **extra)
            s = DistributedDDMSolver(m)
            dc_ctx = s.prepare()
            trans_ctx = s.prepare_transient(dt=DT, method='be')
            sources = s.preprocess_sources(time_step=DT, t_end=T_END, smooth=False)
            result = s.solve_transient(
                trans_ctx, dc_context=dc_ctx,
                smoothed_sources=sources, t_end=T_END,
            )
            peaks = result.as_flat()
            resolved_rtol = trans_ctx._cg_solver.rtol if trans_ctx._cg_solver else None
            dc_ctx.release()
            trans_ctx.release()
            m.shutdown()
            return peaks, resolved_rtol

        peaks_direct, _ = _run_transient('direct')
        # No interface_cg_rtol override -- exercises the production default.
        peaks_cg, resolved_rtol = _run_transient(
            'cg',
            interface_matvec_mode='assembled',
            interface_preconditioner='block_jacobi',
        )

        assert resolved_rtol == 1e-8, (
            f"Expected the resolved default rtol to be 1e-8, got "
            f"{resolved_rtol!r} -- default may have drifted, invalidating "
            f"this accuracy guard."
        )

        common = set(peaks_direct) & set(peaks_cg)
        assert common, "No shared nodes between direct and CG transient results"

        max_diff = max(
            abs(peaks_direct[n][0] - peaks_cg[n][0]) for n in common
        )
        assert max_diff <= 2.7e-5, (
            f"CG (default rtol={resolved_rtol:.1e}) vs direct transient peak "
            f"IR-drop on netlist_sampled ({N_STEPS} steps): "
            f"max |d(peak_drop)| = {max_diff:.3e} V > 2.7e-5 V (27 uV) -- "
            f"the production default rtol=1e-8 is no longer accurate enough "
            f"in transient mode (empirical baseline on this fixture is "
            f"~10.7 uV)."
        )

    def test_block_jacobi_fewer_iters_than_none_on_sampled(self):
        """Block-Jacobi < no-preconditioner in iteration count on netlist_sampled."""
        from distributed.solver import DistributedDDMSolver

        def _run_cg(preconditioner):
            m = self._load_model('cg',
                                 interface_preconditioner=preconditioner,
                                 interface_cg_rtol=1e-10)
            s = DistributedDDMSolver(m)
            ctx = s.prepare()
            s.solve_dc(ctx)
            cg = getattr(ctx, '_cg_solver', None)
            n_iters = cg.total_iterations if cg else 0
            ctx.release()
            m.shutdown()
            return n_iters

        iters_bj = _run_cg('block_jacobi')
        iters_none = _run_cg('none')

        print(
            f"\nNetlist sampled preconditioner: "
            f"block_jacobi={iters_bj} vs none={iters_none} iterations"
        )

        assert iters_bj < iters_none, (
            f"block_jacobi ({iters_bj} iters) should be strictly less than "
            f"none ({iters_none} iters) on netlist_sampled"
        )

    def test_warm_start_strictly_less_over_transient_steps(self):
        """Warm-start CG uses STRICTLY FEWER total iterations than cold over >=10 transient steps.

        Drives the actual solver.solve_transient() time loop (not synthetic CG
        calls) to verify that the automatic warm-start from the previous step's
        v_gamma materially reduces iteration counts on the netlist_sampled system.

        The cold-start baseline is realised by monkey-patching _interface_lu on
        the transient context to reset x0 before each call.  This does NOT
        modify InterfaceCGSolver at the class level, so other tests are unaffected.

        Assertion: total CG iters (warm) < total CG iters (cold) over N_STEPS steps,
        with N_STEPS >= 10 as required by the B2 spec.
        """
        from distributed.solver import DistributedDDMSolver

        N_STEPS = 10
        DT = 1e-10  # 100 ps per step — small enough for fast unit-test loop
        T_END = N_STEPS * DT

        def _run_transient(use_warm_start: bool) -> int:
            """Return total CG iterations for N_STEPS transient steps.

            When use_warm_start=False, _interface_lu is wrapped to reset the
            CG x0 guess before every step (simulating cold restart).
            """
            m = self._load_model(
                'cg',
                interface_preconditioner='none',
                interface_cg_rtol=1e-10,
            )
            s = DistributedDDMSolver(m)
            dc_ctx = s.prepare()
            trans_ctx = s.prepare_transient(dt=DT, method='be')
            # Preprocess sources without smoothing so it is fast
            sources = s.preprocess_sources(time_step=DT, t_end=T_END, smooth=False)

            if not use_warm_start:
                # Wrap _interface_lu with an x0-resetting shim at instance level.
                # Preserves warm-start accumulation inside the CG solver for the
                # stats counter, but forces each step to start from x0=None.
                cg = trans_ctx._cg_solver
                _orig_lu = trans_ctx._interface_lu  # same object as cg

                def _cold_lu(rhs):
                    cg.reset_warm_start()
                    return _orig_lu(rhs)

                trans_ctx._interface_lu = _cold_lu

            # Run transient; use dc_context for initial condition (standard path)
            s.solve_transient(
                trans_ctx,
                dc_context=dc_ctx,
                smoothed_sources=sources,
                t_end=T_END,
            )

            # Read total CG iterations from the solver regardless of wrapping
            cg_solver = trans_ctx._cg_solver
            total_iters = cg_solver.total_iterations if cg_solver is not None else 0

            dc_ctx.release()
            trans_ctx.release()
            m.shutdown()
            return total_iters

        iters_warm = _run_transient(use_warm_start=True)
        iters_cold = _run_transient(use_warm_start=False)

        print(
            f"\nNetlist sampled transient warm-start ({N_STEPS} steps):"
            f"\n  warm: {iters_warm} total CG iters"
            f"\n  cold: {iters_cold} total CG iters"
            f"\n  ratio cold/warm: {iters_cold/max(iters_warm,1):.2f}x"
        )

        assert iters_warm < iters_cold, (
            f"Warm-start CG should use STRICTLY FEWER iterations than cold "
            f"over {N_STEPS} transient steps on netlist_sampled: "
            f"warm={iters_warm} vs cold={iters_cold}"
        )


# ──────────────────────────────────────────────────────────────────────
# 9. Stage 1b: production default rtol is 1e-8
# ──────────────────────────────────────────────────────────────────────


class TestStage1DefaultRtol:
    """Stage 1b: the production default rtol is now 1e-8 (Stage 0 sweep:

    166 nV max error vs direct on the BRCM-class proxy; 1e-7 fails at
    1.66 uV -- see interface_solve_acceleration_plan.md / docs Sec 7.7).
    These are settings-level checks -- no distributed solve is required.
    """

    def test_constructor_default_rtol_is_1e8(self):
        """InterfaceCGSolver's own constructor default (no solve needed)."""
        from distributed.interface_iterative import InterfaceCGSolver

        cg = InterfaceCGSolver(
            n_interface=5, matvec_mode='assembled',
            S_global=sp.eye(5, format='csr'), preconditioner='none',
        )
        assert cg.rtol == 1e-8

    def test_factory_default_rtol_is_1e8(self):
        """build_interface_solver()'s default rtol (no solve needed)."""
        from distributed.interface_iterative import build_interface_solver

        _solve_callable, mode, cg_solver = build_interface_solver(
            S_global=sp.eye(5, format='csr'), interface_solver='cg',
        )
        assert mode == 'cg'
        assert cg_solver.rtol == 1e-8

    def test_resolved_rtol_via_prepare_is_1e8(self):
        """End-to-end: solver.prepare() with no explicit rtol resolves to 1e-8."""
        from distributed.solver import DistributedDDMSolver

        model = _build_dc_model(interface_solver='cg')  # no explicit rtol
        solver = DistributedDDMSolver(model)
        ctx = solver.prepare()
        assert ctx._cg_solver.rtol == 1e-8
        ctx.release()


# ──────────────────────────────────────────────────────────────────────
# 10. Stage 1c: host-aware memory budgets (psutil)
# ──────────────────────────────────────────────────────────────────────


class TestHostAwareMemoryBudgets:
    """Stage 1c: interface_factor_memory_budget / interface_block_jacobi_max_bytes.

    'auto' resolves to min(cap, fraction * total_RAM) via psutil; explicit
    byte counts are used directly.  Module-level constants remain as
    fallbacks for callers that don't resolve a setting at all.
    """

    def test_resolve_factor_memory_budget_auto_uses_psutil(self):
        import psutil

        from distributed.interface_iterative import (
            resolve_factor_memory_budget_bytes,
        )

        total_ram = psutil.virtual_memory().total
        expected = min(32 * 1024 ** 3, int(0.4 * total_ram))
        assert resolve_factor_memory_budget_bytes('auto') == expected
        assert resolve_factor_memory_budget_bytes(None) == expected

    def test_resolve_factor_memory_budget_explicit_bytes(self):
        from distributed.interface_iterative import (
            resolve_factor_memory_budget_bytes,
        )

        assert resolve_factor_memory_budget_bytes(12345) == 12345
        assert resolve_factor_memory_budget_bytes('12345') == 12345

    def test_resolve_block_jacobi_max_bytes_auto_uses_psutil(self):
        """Finding 9: 'auto' = max(legacy 4 GB floor, min(8 GB, 0.1*RAM)) --
        never below the legacy fixed constant, even though on this (large
        RAM) test host min(8GB, 0.1*RAM) already exceeds the floor."""
        import psutil

        import distributed.interface_iterative as ii_mod
        from distributed.interface_iterative import (
            resolve_block_jacobi_max_bytes,
        )

        total_ram = psutil.virtual_memory().total
        expected = max(
            ii_mod.BLOCK_JACOBI_MAX_FACTOR_BYTES,
            min(8 * 1024 ** 3, int(0.1 * total_ram)),
        )
        assert resolve_block_jacobi_max_bytes('auto') == expected
        assert resolve_block_jacobi_max_bytes(None) == expected

    def test_resolve_block_jacobi_max_bytes_auto_never_below_legacy_floor(self):
        """Finding 9 regression test: on a host with little RAM (mocked),
        'auto' must still resolve to at least the legacy 4 GB constant, not
        the smaller min(8 GB, 0.1*RAM) value -- this is the exact scenario
        the finding describes (<40 GB hosts silently downgrading
        block_jacobi where the pre-Stage-1c fixed 4 GB constant did not)."""
        import distributed.interface_iterative as ii_mod
        from distributed.interface_iterative import (
            resolve_block_jacobi_max_bytes,
        )

        # 16 GB host: 0.1*RAM = 1.6 GB, well below the 4 GB legacy floor.
        orig = ii_mod._get_total_ram_bytes
        try:
            ii_mod._get_total_ram_bytes = lambda: 16 * 1024 ** 3
            resolved = resolve_block_jacobi_max_bytes('auto')
        finally:
            ii_mod._get_total_ram_bytes = orig

        assert resolved == ii_mod.BLOCK_JACOBI_MAX_FACTOR_BYTES == 4 * 1024 ** 3

    def test_resolve_block_jacobi_max_bytes_explicit_bytes(self):
        from distributed.interface_iterative import (
            resolve_block_jacobi_max_bytes,
        )

        assert resolve_block_jacobi_max_bytes(999) == 999

    def test_resolve_memory_budget_invalid_setting_raises(self):
        from distributed.interface_iterative import (
            resolve_factor_memory_budget_bytes,
        )

        with pytest.raises(ValueError):
            resolve_factor_memory_budget_bytes('not-a-number')

    def test_resolve_memory_budget_rejects_bool(self):
        """Finding 7/8: bool must be rejected explicitly BEFORE int
        coercion -- bool is an int subclass, so int(float(True)) would
        otherwise silently resolve to 1 byte instead of raising (the
        YAML-1.1 'interface_factor_memory_budget: yes' pitfall)."""
        from distributed.interface_iterative import (
            resolve_factor_memory_budget_bytes,
            resolve_block_jacobi_max_bytes,
        )

        with pytest.raises(ValueError):
            resolve_factor_memory_budget_bytes(True)
        with pytest.raises(ValueError):
            resolve_factor_memory_budget_bytes(False)
        with pytest.raises(ValueError):
            resolve_block_jacobi_max_bytes(True)
        with pytest.raises(ValueError):
            resolve_block_jacobi_max_bytes(False)

    def test_resolve_memory_budget_accepts_exponent_numeric_strings(self):
        """Finding 7/8: numeric strings with exponent notation (as PyYAML
        1.1 delivers for bare-exponent floats like '2e9') are accepted via
        int(float(setting)), matching the already-defensive rtol/atol
        coercion, instead of raising on the visually-plausible '2e9' while
        silently accepting the dotted '2.0e9' equivalent."""
        from distributed.interface_iterative import (
            resolve_factor_memory_budget_bytes,
            resolve_block_jacobi_max_bytes,
        )

        assert resolve_factor_memory_budget_bytes('2e9') == 2_000_000_000
        assert resolve_factor_memory_budget_bytes('2.5e9') == 2_500_000_000
        assert resolve_block_jacobi_max_bytes('2e9') == 2_000_000_000
        assert resolve_block_jacobi_max_bytes('2.5e9') == 2_500_000_000

    def test_block_jacobi_warning_names_setting_and_iteration_consequence(
        self, caplog,
    ):
        """The block_jacobi->jacobi downgrade WARNING names the setting and
        the iteration-count consequence (loud, not silent/generic)."""
        import logging

        from distributed.interface_iterative import InterfaceCGSolver

        n = 4
        rng = np.random.default_rng(22)
        A = rng.standard_normal((n, n))
        S = sp.csr_matrix(A @ A.T + n * np.eye(n))

        with caplog.at_level(
            logging.WARNING, logger='distributed.interface_iterative',
        ):
            InterfaceCGSolver(
                n_interface=n, matvec_mode='assembled', S_global=S,
                tile_schur_complements={(0, 0): S.toarray()},
                tile_index_maps={(0, 0): np.arange(n, dtype=np.int32)},
                preconditioner='block_jacobi', rtol=1e-12,
                block_jacobi_max_bytes=0,  # force the downgrade
            )

        joined = ' '.join(
            r.message for r in caplog.records if r.levelno == logging.WARNING
        )
        assert 'interface_block_jacobi_max_bytes' in joined, (
            f"WARNING must name the setting; got: {joined!r}"
        )
        assert 'iteration' in joined.lower(), (
            f"WARNING must describe the iteration-count consequence; "
            f"got: {joined!r}"
        )


# ──────────────────────────────────────────────────────────────────────
# 11. Stage 1 settings propagation: coordinator-side settings reach the
#     object that consumes them.
# ──────────────────────────────────────────────────────────────────────


class TestStage1SettingsPropagation:
    """One test per new Stage 1 setting, proving it reaches the consuming
    object.  Per CLAUDE.md: Ray module-globals do NOT propagate to workers;
    these settings are all consumed coordinator-side (InterfaceCGSolver /
    auto_select_interface_solver), so a direct construction + attribute
    assertion is sufficient proof (no Ray worker round-trip needed).
    """

    def test_interface_cg_atol_reaches_cg_solver(self):
        from distributed.solver import DistributedDDMSolver

        model = _build_dc_model(interface_solver='cg', interface_cg_atol=1e-9)
        solver = DistributedDDMSolver(model)
        ctx = solver.prepare()
        assert ctx._cg_solver.atol == 1e-9
        ctx.release()

    def test_interface_cg_maxiter_reaches_cg_solver(self):
        from distributed.solver import DistributedDDMSolver

        model = _build_dc_model(interface_solver='cg', interface_cg_maxiter=7)
        solver = DistributedDDMSolver(model)
        ctx = solver.prepare()
        assert ctx._cg_solver.maxiter == 7
        ctx.release()

    def test_interface_cg_strict_reaches_cg_solver(self):
        from distributed.solver import DistributedDDMSolver

        model = _build_dc_model(interface_solver='cg', interface_cg_strict=False)
        solver = DistributedDDMSolver(model)
        ctx = solver.prepare()
        assert ctx._cg_solver.strict is False
        ctx.release()

    def test_interface_factor_memory_budget_reaches_auto_select(self):
        """A tiny explicit byte budget forces auto -> 'cg' even though
        n_interface=1 is far below the count threshold, proving the setting
        reached auto_select_interface_solver's decision."""
        from distributed.solver import DistributedDDMSolver

        model = _build_dc_model(
            interface_solver='auto', interface_factor_memory_budget=1,
        )
        solver = DistributedDDMSolver(model)
        ctx = solver.prepare()
        assert ctx._interface_solver_mode == 'cg'
        ctx.release()

    def test_interface_block_jacobi_max_bytes_reaches_cg_solver(self):
        from distributed.solver import DistributedDDMSolver

        model = _build_dc_model(
            interface_solver='cg', interface_block_jacobi_max_bytes=0,
        )
        solver = DistributedDDMSolver(model)
        ctx = solver.prepare()
        assert ctx._cg_solver.block_jacobi_max_bytes == 0
        ctx.release()

    def test_transient_settings_reach_trans_ctx_cg_solver(self):
        """Same settings, but through prepare_transient() rather than
        prepare() -- the transient InterfaceCGSolver construction
        (result_factorization.py) is a separate code path from the DC one
        above, so it needs its own propagation proof rather than relying on
        "it looks code-identical by inspection"."""
        from distributed.solver import DistributedDDMSolver

        model = _build_two_tile_distributed_model(package_cap_edges=[])
        model.settings.update({
            'interface_solver': 'cg',
            'interface_cg_atol': 1e-9,
            'interface_cg_maxiter': 7,
            'interface_cg_strict': False,
            'interface_block_jacobi_max_bytes': 0,
        })
        solver = DistributedDDMSolver(model)
        trans_ctx = solver.prepare_transient(dt=1e-10, method='be')

        assert trans_ctx._cg_solver.atol == 1e-9
        assert trans_ctx._cg_solver.maxiter == 7
        assert trans_ctx._cg_solver.strict is False
        assert trans_ctx._cg_solver.block_jacobi_max_bytes == 0

        trans_ctx.release()


# ──────────────────────────────────────────────────────────────────────
# 12. Gate-coverage fix: forced-CG equivalence on the pad-on-port fixture
# ──────────────────────────────────────────────────────────────────────


class TestForcedCGEquivalence:
    """Stage 1 'Gate-coverage fix': the standing gates exercise small
    *direct* solves, not the CG target path.  This class forces
    interface_solver='cg' (block_jacobi, pinned rtol=1e-12) against the
    direct solver on ``_build_two_tile_distributed_model`` -- a fixture
    where tile A's port list includes a Dirichlet pad ('pad') directly,
    the exact D1 scenario (interface_solve_acceleration_plan.md).
    """

    @pytest.mark.parametrize('matvec_mode', ['assembled', 'tilewise'])
    def test_cg_dc_matches_direct_on_pad_on_port_fixture(self, matvec_mode):
        """Interface-solve level: CG (block_jacobi, pinned rtol=1e-12) vs
        direct on the assembled S_global from the pad-on-tile-port fixture.

        Compares at the ``ctx._cg_solver`` / ``ctx._S_global`` level (not via
        ``solve_dc()``) because ``solve_dc()``'s coordinator-side reduced-RHS
        assembly (``get_reduced_rhs`` vs the filtered ``tile_index_maps``) has
        its own separate, pre-existing length-mismatch limitation for tiles
        with a Dirichlet pad directly on their port list -- unrelated to CG
        vs direct and out of scope for this stage.  Comparing the interface
        solve directly isolates exactly the D1 matvec mechanism instead.
        """
        import scipy.sparse.linalg as spla

        from distributed.solver import DistributedDDMSolver

        def _build_ctx(interface_solver):
            model = _build_two_tile_distributed_model(package_cap_edges=[])
            model.settings.update({
                'interface_solver': interface_solver,
                'interface_matvec_mode': matvec_mode,
                'interface_preconditioner': 'block_jacobi',
                'interface_cg_rtol': 1e-12,
            })
            solver = DistributedDDMSolver(model)
            try:
                ctx = solver.prepare()
            except Exception:
                # Defensive: shut the model down before propagating so it
                # never leaks even if construction raises for an unrelated
                # reason and the caller's ctx variable is never assigned.
                model.shutdown()
                raise
            return ctx, model

        # Finding 15 (defensive, retained post-D1-fix): without try/finally,
        # a raise anywhere in this block would skip release()/shutdown() for
        # whichever context(s) DID build successfully, leaking factored tile
        # workers and coordinator factors for the rest of the pytest session.
        ctx_direct = model_direct = None
        ctx_cg = model_cg = None
        try:
            ctx_direct, model_direct = _build_ctx('direct')
            ctx_cg, model_cg = _build_ctx('cg')

            n = len(ctx_direct.interface_nodes)
            rng = np.random.default_rng(0)
            rhs = rng.standard_normal(n)

            x_direct = spla.spsolve(ctx_direct._S_global.tocsr(), rhs)
            x_cg = ctx_cg._cg_solver(rhs)
        finally:
            if ctx_direct is not None:
                ctx_direct.release()
            if ctx_cg is not None:
                ctx_cg.release()
            if model_direct is not None:
                model_direct.shutdown()
            if model_cg is not None:
                model_cg.shutdown()

        np.testing.assert_allclose(
            x_cg, x_direct, atol=1e-8,
            err_msg=f"CG ({matvec_mode}) vs direct interface solve on "
                    f"pad-on-port fixture",
        )

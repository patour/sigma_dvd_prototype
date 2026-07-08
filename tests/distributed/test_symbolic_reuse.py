"""Unit tests for A4: symbolic-reuse + assembly-pattern-reuse optimisations.

Tests:
  1. Symbolic reuse: DC → transient.  S_A from numeric-only path must match
     full-analyse path to ≤1e-12 relative tolerance.
  2. Symbolic reuse: transient → DC (reverse order, same guarantee).
  3. Repeated-prepare cycle (DC → trans → DC2): each step uses the cached
     symbolic; results match independent fresh solves.
  4. Fallback on pattern mismatch: injecting a different G_ii structure
     triggers a fresh analyse and falls back gracefully.
  5. DC lu_ii preserved: after transient uses the cache, DC lu_ii closure
     still returns correct results.
  6. Assembly pattern reuse: DC assemble caches local_to_global; transient
     reuse produces S_global with identical indptr/indices/data.
  7. Assembly cache invalidated correctly on topology change.
  8. Numpy-concat assembly matches result of a chain-of-tiles scenario.
  9. End-to-end prepare + prepare_transient: workers have _symbolic_ii set;
     both contexts factored; DC solve correct.
  10. End-to-end transient-first, then DC prepare: same guarantees.
  11. Reuse vs no-reuse produces identical DC voltages.
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import pytest
import scipy.sparse as sp

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_cholmod_available() -> bool:
    try:
        from sksparse.cholmod import cholesky  # noqa: F401
        return True
    except ImportError:
        return False


def _build_simple_block_system(g_interior=5.0):
    """Return a BlockMatrixSystem with 1 interior and 1 port node."""
    from pgmath.block_system import BlockMatrixSystem
    G_ii = sp.csc_matrix(np.array([[g_interior + 1.0]]))
    G_pp = sp.csc_matrix(np.array([[3.0]]))
    # G_pi is (n_p × n_i) = (1 × 1)
    G_pi = sp.csc_matrix(np.array([[-1.0]]))
    G_ip = G_pi.T.tocsc()
    return BlockMatrixSystem(
        G_pp=G_pp, G_pi=G_pi, G_ip=G_ip, G_ii=G_ii,
        port_nodes=['p0'], interior_nodes=['i0'],
        port_to_idx={'p0': 0}, interior_to_idx={'i0': 0},
        lu_ii=None,
    )


def _build_transient_block_system(bs, c_coeff=0.1, c_ii_val=2.0, c_pp_val=1.0):
    """Return A = G + C_coeff * diag(c) block system (same sparsity as G)."""
    from pgmath.block_system import BlockMatrixSystem
    A_ii = bs.G_ii + sp.diags(c_coeff * np.array([c_ii_val]), format='csc')
    A_pp = bs.G_pp + sp.diags(c_coeff * np.array([c_pp_val]), format='csc')
    return BlockMatrixSystem(
        G_pp=A_pp, G_pi=bs.G_pi, G_ip=bs.G_ip, G_ii=A_ii,
        port_nodes=list(bs.port_nodes), interior_nodes=list(bs.interior_nodes),
        port_to_idx=dict(bs.port_to_idx), interior_to_idx=dict(bs.interior_to_idx),
        lu_ii=None,
    )


def _build_full_two_tile_model():
    """Build the two-tile model used in test_distributed_solver.py.

    This fixture has pad_v as a pure package node (not a tile boundary),
    so solve_dc() works correctly.
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
        package_cap_edges=[],
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

    import pickle as _pickle

    tmpdir = tempfile.mkdtemp()
    for td in [tile_a, tile_b]:
        x, y = td.tile_id
        with open(os.path.join(tmpdir, f'tile_{x}_{y}.pkl'), 'wb') as f:
            _pickle.dump(td, f)
    with open(os.path.join(tmpdir, 'metadata.pkl'), 'wb') as f:
        _pickle.dump({
            'metadata': metadata,
            'boundary_nodes': {'B'},
        }, f)

    bundle = ParsedTileBundle(
        metadata=metadata,
        shared_boundary_nodes={'B'},
        pkl_dir=tmpdir,
    )
    model = create_distributed_model(bundle, backend='local')
    model._owns_pkl_dir = tmpdir  # so tests can clean up if needed
    return model


# ---------------------------------------------------------------------------
# 1. Symbolic reuse — DC → transient
# ---------------------------------------------------------------------------

class TestSymbolicReuseCholesky:
    """Symbolic-reuse path tests. Skipped when CHOLMOD is unavailable."""

    @pytest.fixture(autouse=True)
    def skip_if_no_cholmod(self):
        if not _is_cholmod_available():
            pytest.skip("CHOLMOD not available — symbolic-reuse path inactive")

    def test_dc_to_transient_s_equal(self):
        """S from symbolic-reuse path == S from full-analyse path (≤1e-12 rel)."""
        from pgmath.schur import compute_explicit_schur

        bs_dc = _build_simple_block_system()
        bs_trans = _build_transient_block_system(bs_dc, c_coeff=0.1)

        # Full-analyse reference for transient
        bs_trans_ref = _build_transient_block_system(_build_simple_block_system(), c_coeff=0.1)
        S_ref, _ = compute_explicit_schur(bs_trans_ref, symbolic_cache=None)

        # DC factor — populates cache
        S_dc, stats_dc = compute_explicit_schur(bs_dc, symbolic_cache=None)
        assert '_new_symbolic_cache' in stats_dc
        cache = stats_dc['_new_symbolic_cache']

        # Transient factor using cache
        S_trans, stats_trans = compute_explicit_schur(bs_trans, symbolic_cache=cache)

        assert stats_trans.get('symbolic_reuse') is True
        assert '_new_symbolic_cache' not in stats_trans  # no new cache on hit

        rel_diff = np.max(np.abs(S_trans - S_ref)) / (np.max(np.abs(S_ref)) + 1e-30)
        assert rel_diff <= 1e-12, f"Rel diff S_A symbolic vs full: {rel_diff:.2e}"

    def test_transient_to_dc_s_equal(self):
        """Reverse order: transient first, then DC reuses the cached symbolic."""
        from pgmath.schur import compute_explicit_schur

        bs_dc_base = _build_simple_block_system()
        bs_trans = _build_transient_block_system(bs_dc_base, c_coeff=0.05)

        # Reference for DC (no cache)
        S_dc_ref, _ = compute_explicit_schur(_build_simple_block_system(), symbolic_cache=None)

        # Transient first — populates cache
        _, stats_trans = compute_explicit_schur(bs_trans, symbolic_cache=None)
        assert '_new_symbolic_cache' in stats_trans
        cache = stats_trans['_new_symbolic_cache']

        # DC using cache (same G_ii pattern as base of transient A_ii)
        bs_dc2 = _build_simple_block_system()
        S_dc2, stats_dc2 = compute_explicit_schur(bs_dc2, symbolic_cache=cache)

        assert stats_dc2.get('symbolic_reuse') is True
        rel_diff = np.max(np.abs(S_dc2 - S_dc_ref)) / (np.max(np.abs(S_dc_ref)) + 1e-30)
        assert rel_diff <= 1e-12, f"Rel diff S_dc symbolic vs full: {rel_diff:.2e}"

    def test_repeated_prepare_cycle(self):
        """DC → trans → DC2: each step reuses symbolic; DC2 matches DC reference."""
        from pgmath.schur import compute_explicit_schur

        bs1 = _build_simple_block_system()
        bs_trans = _build_transient_block_system(bs1, c_coeff=0.08)
        bs2 = _build_simple_block_system()

        # Reference DC (no cache)
        S_ref, _ = compute_explicit_schur(_build_simple_block_system())

        # Step 1: DC
        _, stats1 = compute_explicit_schur(bs1)
        assert '_new_symbolic_cache' in stats1
        cache = stats1['_new_symbolic_cache']

        # Step 2: transient reuses cache
        _, stats2 = compute_explicit_schur(bs_trans, symbolic_cache=cache)
        assert stats2.get('symbolic_reuse') is True

        # Step 3: DC again, reuses updated cache
        S_dc2, stats3 = compute_explicit_schur(bs2, symbolic_cache=cache)
        assert stats3.get('symbolic_reuse') is True

        rel_diff = np.max(np.abs(S_dc2 - S_ref)) / (np.max(np.abs(S_ref)) + 1e-30)
        assert rel_diff <= 1e-12, f"DC2 rel diff after cycle: {rel_diff:.2e}"

    def test_fallback_on_pattern_mismatch(self):
        """Mismatched G_ii structure triggers full-analyse fallback (no error)."""
        from pgmath.schur import compute_explicit_schur
        from pgmath.block_system import BlockMatrixSystem

        bs1 = _build_simple_block_system()
        _, stats1 = compute_explicit_schur(bs1)
        cache = stats1['_new_symbolic_cache']

        # Build a DIFFERENT block system: 2 interior nodes, 1 port.
        # G_pi must be (n_p × n_i) = (1 × 2); G_ip = G_pi.T is (2 × 1).
        G_ii2 = sp.csc_matrix(np.array([[6.0, -0.5], [-0.5, 4.0]]))  # 2×2
        G_pp2 = sp.csc_matrix(np.array([[3.0]]))                       # 1×1
        G_pi2 = sp.csc_matrix(np.array([[-1.0, -0.5]]))               # 1×2
        G_ip2 = G_pi2.T.tocsc()                                        # 2×1
        bs2 = BlockMatrixSystem(
            G_pp=G_pp2, G_pi=G_pi2, G_ip=G_ip2, G_ii=G_ii2,
            port_nodes=['p0'], interior_nodes=['i0', 'i1'],
            port_to_idx={'p0': 0}, interior_to_idx={'i0': 0, 'i1': 1},
            lu_ii=None,
        )

        # Should NOT raise; should fall back to full analyse
        S2, stats2 = compute_explicit_schur(bs2, symbolic_cache=cache)

        assert stats2.get('symbolic_reuse') is not True
        assert '_new_symbolic_cache' in stats2  # new cache built from fresh analyse
        assert S2.shape == (1, 1)

        # Verify result is numerically correct (compare to no-cache call)
        bs2_ref = BlockMatrixSystem(
            G_pp=G_pp2.copy(), G_pi=G_pi2.copy(), G_ip=G_ip2.copy(), G_ii=G_ii2.copy(),
            port_nodes=['p0'], interior_nodes=['i0', 'i1'],
            port_to_idx={'p0': 0}, interior_to_idx={'i0': 0, 'i1': 1},
            lu_ii=None,
        )
        S2_ref, _ = compute_explicit_schur(bs2_ref)
        np.testing.assert_allclose(S2, S2_ref, rtol=1e-12)

    def test_dc_lu_preserved_after_transient_factor(self):
        """DC lu_ii closure still solves correctly after transient uses the cache."""
        from pgmath.schur import compute_explicit_schur

        bs_dc = _build_simple_block_system()
        bs_trans = _build_transient_block_system(bs_dc, c_coeff=0.2)

        # DC factor
        _, stats_dc = compute_explicit_schur(bs_dc)
        cache = stats_dc['_new_symbolic_cache']
        lu_dc = bs_dc.lu_ii  # closure over DC factor

        # Transient factor uses cache (in-place refactor of a COPY)
        _, _ = compute_explicit_schur(bs_trans, symbolic_cache=cache)

        # DC lu_ii should still solve G_ii x = b correctly
        b = np.array([1.0])
        x_lu = lu_dc(b)
        G_ii_dense = bs_dc.G_ii.toarray()
        x_ref = np.linalg.solve(G_ii_dense, b)
        np.testing.assert_allclose(
            x_lu, x_ref, rtol=1e-12,
            err_msg="DC lu_ii corrupted by transient symbolic reuse",
        )


# ---------------------------------------------------------------------------
# 2. Assembly pattern reuse
# ---------------------------------------------------------------------------

class TestAssemblyPatternReuse:
    """Test that the assembly_cache avoids redundant node-index map builds."""

    def test_assembly_cache_produces_identical_s_global(self):
        """Second assemble call with cache gives identical S_global to fresh call."""
        from pgmath.schur import assemble_schur_complement_system

        # Two tiles sharing node 'shared'
        S_dc_a = np.array([[3.0, -1.0], [-1.0, 2.0]])
        S_dc_b = np.array([[1.5]])

        tile_schur_dc = {(0, 0): S_dc_a, (0, 1): S_dc_b}
        tile_ports = {(0, 0): ['shared', 'pad'], (0, 1): ['shared']}

        extra = [('pad', 'shared', 0.5)]
        dirichlet = {'pad'}

        # First call (DC) — populates cache
        cache = {}
        S1, rhs1, nodes1, idx1 = assemble_schur_complement_system(
            tile_schur_dc, tile_ports,
            extra_edges=extra, dirichlet_nodes=dirichlet,
            dirichlet_voltage=1.0, assembly_cache=cache,
        )
        assert 'tile_local_to_global' in cache
        assert 'full_node_to_idx' in cache

        # Second call (transient) with different S_i values but same topology
        S_td_a = S_dc_a + 0.05 * np.eye(2)
        S_td_b = S_dc_b + 0.02 * np.eye(1)
        tile_schur_td = {(0, 0): S_td_a, (0, 1): S_td_b}

        S2, rhs2, nodes2, idx2 = assemble_schur_complement_system(
            tile_schur_td, tile_ports,
            extra_edges=extra, dirichlet_nodes=dirichlet,
            dirichlet_voltage=1.0, assembly_cache=cache,
        )

        # Reference: fresh call without cache
        S2_ref, rhs2_ref, _, _ = assemble_schur_complement_system(
            tile_schur_td, tile_ports,
            extra_edges=extra, dirichlet_nodes=dirichlet,
            dirichlet_voltage=1.0, assembly_cache=None,
        )

        # Node ordering must be the same
        assert nodes1 == nodes2
        assert idx1 == idx2

        # S_global must be numerically identical
        diff = (S2 - S2_ref)
        if diff.nnz > 0:
            assert np.max(np.abs(diff.data)) < 1e-14
        np.testing.assert_allclose(rhs2, rhs2_ref, rtol=1e-14)

    def test_assembly_cache_invalidated_on_shape_mismatch(self):
        """Shape mismatch (different n_unknown) causes fall-back, no error."""
        from pgmath.schur import assemble_schur_complement_system

        S1 = np.array([[2.0]])
        tile_schur1 = {(0, 0): S1}
        tile_ports1 = {(0, 0): ['a']}

        cache = {}
        assemble_schur_complement_system(
            tile_schur1, tile_ports1, assembly_cache=cache,
        )
        assert 'n_unknown' in cache

        # Second call with MORE nodes — should fall back and re-cache
        S2 = np.array([[2.0, -0.5], [-0.5, 3.0]])
        tile_schur2 = {(0, 0): S2}
        tile_ports2 = {(0, 0): ['a', 'b']}

        S_out, _, nodes_out, _ = assemble_schur_complement_system(
            tile_schur2, tile_ports2, assembly_cache=cache,
        )
        S_ref, _, _, _ = assemble_schur_complement_system(
            tile_schur2, tile_ports2, assembly_cache=None,
        )
        diff = S_out - S_ref
        max_diff = np.max(np.abs(diff.data)) if diff.nnz > 0 else 0.0
        assert max_diff < 1e-14

    def test_l2g_not_reused_when_node_map_rebuilt(self):
        """l2g must NOT be reused when full_node_to_idx was rebuilt (A4 bug fix).

        Scenario: first assemble with N resistive interface nodes; second
        assemble with N+k nodes because extra_edges introduce a new node
        (mirrors the transient-prepare two-assemble sequence when
        package_cap_edges add nodes absent from the resistive interface graph).

        Before the fix: _cached_l2g was read unconditionally from assembly_cache
        whenever assembly_cache was truthy.  When n_unknown changed (_use_cached_idx=False)
        the second call rebuilt full_node_to_idx with a different sorted ordering
        but still reused the stale per-tile l2g arrays (indexing into the OLD
        ordering) -> silently wrong S_global.

        After the fix: l2g is only reused when _use_cached_idx is True.
        """
        from pgmath.schur import assemble_schur_complement_system

        # Call 1 (DC): tile A with ports ['x', 'y'] — sorted order x=0, y=1
        S_dc = np.array([[3.0, -1.0], [-1.0, 2.0]])
        tile_schur_1 = {(0, 0): S_dc}
        tile_ports_1 = {(0, 0): ['x', 'y']}

        cache = {}
        assemble_schur_complement_system(tile_schur_1, tile_ports_1, assembly_cache=cache)
        assert cache.get('n_unknown') == 2
        # Cache: l2g[(0,0)] = [0, 1] in ordering x=0, y=1

        # Call 2 (transient): SAME tile A, PLUS extra_edge introducing 'cap_n'
        # -> all_interface_nodes = {'cap_n','x','y'} -> n_unknown = 3 != 2
        # -> _use_cached_idx = False -> full_node_to_idx rebuilt
        # -> fresh sorted order: ['cap_n','x','y'] -> cap_n=0, x=1, y=2
        # Bug: old code read l2g[(0,0)]=[0,1] (x=0,y=1 OLD) and used it against
        #      new ordering (x=1,y=2) -> S_dc was scattered to wrong positions.
        S_td = np.array([[3.5, -1.0], [-1.0, 2.5]])
        tile_schur_2 = {(0, 0): S_td}
        tile_ports_2 = {(0, 0): ['x', 'y']}
        extra = [('cap_n', '0', 1.0)]  # new node not present in call 1

        S_cached, rhs_cached, nodes_cached, _ = assemble_schur_complement_system(
            tile_schur_2, tile_ports_2,
            extra_edges=extra,
            assembly_cache=cache,
        )

        # Reference: fresh call without any cache
        S_ref, rhs_ref, nodes_ref, _ = assemble_schur_complement_system(
            tile_schur_2, tile_ports_2,
            extra_edges=extra,
            assembly_cache=None,
        )

        assert nodes_cached == nodes_ref, (
            f"Node ordering mismatch with cache: {nodes_cached} vs ref {nodes_ref}"
        )
        diff = S_cached - S_ref
        max_diff = np.max(np.abs(diff.data)) if diff.nnz > 0 else 0.0
        assert max_diff < 1e-14, (
            f"S_global mismatch (stale l2g reuse bug): max|diff|={max_diff:.2e}. "
            f"Tile Schur data was scattered to wrong global indices."
        )
        np.testing.assert_allclose(rhs_cached, rhs_ref, atol=1e-14)

    def test_numpy_concat_vs_list_extend(self):
        """Numpy-concat assembly matches a chain-of-tiles scenario."""
        from pgmath.schur import assemble_schur_complement_system

        n_tiles = 10
        all_nodes = [f"n{i}" for i in range(n_tiles + 1)]
        tile_schur = {}
        tile_ports = {}
        for i in range(n_tiles):
            ports = [all_nodes[i], all_nodes[i + 1]]
            S = np.array([[5.0, -1.0], [-1.0, 4.0]])
            tile_schur[(i, 0)] = S
            tile_ports[(i, 0)] = ports

        dirichlet = {'n0'}

        S_global, rhs, nodes, idx = assemble_schur_complement_system(
            tile_schur, tile_ports,
            dirichlet_nodes=dirichlet, dirichlet_voltage=1.0,
            assembly_cache={},
        )
        S_ref, rhs_ref, _, _ = assemble_schur_complement_system(
            tile_schur, tile_ports,
            dirichlet_nodes=dirichlet, dirichlet_voltage=1.0,
            assembly_cache=None,
        )
        diff = S_global - S_ref
        max_diff = np.max(np.abs(diff.data)) if diff.nnz > 0 else 0.0
        assert max_diff < 1e-14
        np.testing.assert_allclose(rhs, rhs_ref, rtol=1e-14)


# ---------------------------------------------------------------------------
# 3. End-to-end: prepare + prepare_transient on two-tile model
# ---------------------------------------------------------------------------

class TestEndToEndSymbolicReuse:
    """Verify A4 changes work end-to-end with a proper distributed model."""

    @pytest.fixture(autouse=True)
    def skip_if_no_cholmod(self):
        if not _is_cholmod_available():
            pytest.skip("CHOLMOD not available")

    def test_dc_prepare_and_transient_prepare_numerics(self):
        """DC prepare then transient prepare; both contexts factored, solve correct."""
        from distributed.solver import DistributedDDMSolver

        model = _build_full_two_tile_model()
        try:
            solver = DistributedDDMSolver(model)

            dc_ctx = solver.prepare()
            trans_ctx = solver.prepare_transient(dt=1e-10, method='be')

            # Workers should have symbolic_ii populated after DC prepare
            for w in model.workers:
                assert w._symbolic_ii is not None, \
                    "Worker._symbolic_ii should be set after DC prepare"

            assert dc_ctx.is_factored
            assert trans_ctx.is_factored

            # DC solve correctness
            result = solver.solve_dc(dc_ctx)
            v_all = result.flatten()
            for node, v in v_all.items():
                if node == '0':
                    continue
                assert 0.0 <= v <= 1.01, \
                    f"DC voltage {v:.4f} out of range for {node}"

            trans_ctx.release()
            dc_ctx.release()
        finally:
            model.shutdown()

    def test_transient_prepare_first_then_dc(self):
        """Transient prepare runs first; DC prepare reuses the worker caches."""
        from distributed.solver import DistributedDDMSolver

        model = _build_full_two_tile_model()
        try:
            solver = DistributedDDMSolver(model)

            trans_ctx = solver.prepare_transient(dt=1e-10, method='be')
            assert trans_ctx.is_factored

            # After transient, workers have _symbolic_ii from transient factor
            for w in model.workers:
                assert w._symbolic_ii is not None

            dc_ctx = solver.prepare()
            assert dc_ctx.is_factored

            result = solver.solve_dc(dc_ctx)
            v_all = result.flatten()
            for node, v in v_all.items():
                if node == '0':
                    continue
                assert 0.0 <= v <= 1.01

            trans_ctx.release()
            dc_ctx.release()
        finally:
            model.shutdown()

    def test_reuse_vs_no_reuse_dc_voltages_match(self):
        """Disable symbolic reuse on a second model; DC voltages must match."""
        from distributed.solver import DistributedDDMSolver

        model_a = _build_full_two_tile_model()
        model_b = _build_full_two_tile_model()
        try:
            # Disable symbolic reuse on model_b
            for w in model_b.workers:
                w._use_symbolic_reuse = False

            solver_a = DistributedDDMSolver(model_a)
            dc_a = solver_a.prepare()
            solver_a.prepare_transient(dt=1e-10, method='be')
            result_a = solver_a.solve_dc(dc_a)

            solver_b = DistributedDDMSolver(model_b)
            dc_b = solver_b.prepare()
            solver_b.prepare_transient(dt=1e-10, method='be')
            result_b = solver_b.solve_dc(dc_b)

            va = result_a.flatten()
            vb = result_b.flatten()

            for node in va:
                if node == '0':
                    continue
                diff = abs(va[node] - vb.get(node, 0.0))
                assert diff <= 1e-12, \
                    f"DC voltage mismatch reuse vs no-reuse at {node}: {diff:.2e}"
        finally:
            model_a.shutdown()
            model_b.shutdown()

    def test_dc_prepare_populates_assembly_cache_on_topology(self):
        """DC prepare must populate _assembly_cache on the topology so that
        prepare_transient's PRIMARY assemble gets a cache hit (not a rebuild).

        This is the A4 blocker: previously, DC prepare ran with _asm_cache=None
        (because ctx.topology was None during DC's assemble call), so the
        transient prepare rebuilt full_node_to_idx and all local_to_global
        arrays from scratch.  After the fix the cache is pre-populated by DC
        and the transient's combined-edges assemble hits it directly.
        """
        from distributed.solver import DistributedDDMSolver

        model = _build_full_two_tile_model()
        try:
            solver = DistributedDDMSolver(model)

            dc_ctx = solver.prepare()

            # After DC prepare, topology._assembly_cache must be populated
            # (not None and not empty).
            topo = dc_ctx.topology
            assert topo is not None, "Topology must exist after DC prepare"
            assert getattr(topo, '_assembly_cache', None) is not None, \
                "_assembly_cache must not be None after DC prepare"
            cache = topo._assembly_cache
            assert 'tile_local_to_global' in cache, \
                "tile_local_to_global must be in cache after DC prepare"
            assert 'full_node_to_idx' in cache, \
                "full_node_to_idx must be in cache after DC prepare"
            assert 'n_unknown' in cache, \
                "n_unknown must be in cache after DC prepare"

            n_unknown_dc = cache['n_unknown']
            n_full_dc = cache.get('n_full')

            # Now call prepare_transient — it must HIT the cache on its primary
            # assemble call (combined_edges).  Verify by checking that the cache
            # n_unknown/n_full values are UNCHANGED (same topology, so the quick
            # equality check in assemble_schur_complement_system must pass).
            trans_ctx = solver.prepare_transient(dt=1e-10, method='be')
            assert trans_ctx.is_factored

            # n_unknown must still match (cache hit → same node ordering)
            assert cache.get('n_unknown') == n_unknown_dc, \
                "n_unknown changed after prepare_transient — cache was NOT reused"
            assert cache.get('n_full') == n_full_dc, \
                "n_full changed after prepare_transient — cache was NOT reused"

            trans_ctx.release()
            dc_ctx.release()
        finally:
            model.shutdown()

    def test_transient_first_populates_assembly_cache_on_topology(self):
        """Transient-first prepare also populates _assembly_cache on the topology."""
        from distributed.solver import DistributedDDMSolver

        model = _build_full_two_tile_model()
        try:
            solver = DistributedDDMSolver(model)

            trans_ctx = solver.prepare_transient(dt=1e-10, method='be')

            topo = trans_ctx.topology
            assert topo is not None
            assert getattr(topo, '_assembly_cache', None) is not None, \
                "_assembly_cache must not be None after transient-first prepare"
            cache = topo._assembly_cache
            assert 'tile_local_to_global' in cache
            assert 'full_node_to_idx' in cache
            assert 'n_unknown' in cache

            n_unknown_td = cache['n_unknown']

            # DC prepare should also reuse the cache
            dc_ctx = solver.prepare()
            assert dc_ctx.is_factored
            assert cache.get('n_unknown') == n_unknown_td, \
                "n_unknown changed after DC prepare (transient-first) — cache not reused"

            trans_ctx.release()
            dc_ctx.release()
        finally:
            model.shutdown()

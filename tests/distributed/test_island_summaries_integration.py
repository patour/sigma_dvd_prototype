"""Stage 1e integration tests (real netlist data).

Covers the plan items that need a real multi-tile netlist or a full DC
solve pipeline to validate:

  (i)   End-to-end parity: pre-clean-on (summaries fast path) vs
        forced-legacy (Schur-BFS) on netlist_test -- identical per-tile
        boundary_nodes, BlockMatrixSystem dimensions, S_global (exact),
        island sets, and DC/transient solutions.
  (iv)  Tile-resident-pad fixture: the documented BFS divergence, validated
        against an independent nodal-analysis oracle (see class docstring
        for why this substitutes for the full UnifiedIRDropSolver pipeline),
        plus the rescue WARNING.
  (vii) netlist_test DC + TD: identical island sets, summaries vs BFS.

Unit-level coverage (engineered fixtures, _pre_clean_tile_data internals,
the union-find oracle-equivalence matrix, and the trust-assertion /
fallback-matrix resolution) lives in test_island_summaries.py.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from tests.fixtures import NETLIST_TEST

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# (i) + (vii): end-to-end parity, netlist_test
# ---------------------------------------------------------------------------

class TestEndToEndParitySummariesVsSchurBFS:
    """Parse netlist_test once; build one model per island_detection mode
    from the SAME bundle and assert every downstream artifact matches.
    """

    def _parse_bundle(self, tmp_path_str):
        from distributed.parser import DistributedNetlistParser
        parser = DistributedNetlistParser(NETLIST_TEST, net_filter='VDD')
        _, bundle = parser.parse_and_dump(tmp_path_str, backend='local')
        return bundle

    def _build_model(self, bundle, island_detection):
        from distributed.model import create_distributed_model
        return create_distributed_model(
            bundle, backend='local', island_detection=island_detection,
        )

    def test_bundle_has_stage1e_summaries(self, tmp_path):
        """Sanity: parse_and_dump now unconditionally produces summaries."""
        bundle = self._parse_bundle(str(tmp_path))
        assert bundle.component_summaries is not None
        assert bundle.parser_interface_set is not None
        assert bundle.connectivity_summary_version is not None
        assert len(bundle.component_summaries) > 0

    def test_resolved_modes_are_as_requested(self, tmp_path):
        bundle = self._parse_bundle(str(tmp_path))
        model_fast = self._build_model(bundle, 'auto')
        model_legacy = self._build_model(bundle, 'schur_bfs')
        try:
            assert model_fast.island_detection_mode == 'summaries', (
                "Expected the fast path to engage on a freshly Stage-1e-parsed "
                "netlist_test bundle -- if this fails, the trust assertion is "
                "failing and defeats the entire parity comparison below."
            )
            assert model_legacy.island_detection_mode == 'schur_bfs'
        finally:
            model_fast.shutdown()
            model_legacy.shutdown()

    def test_per_tile_boundary_and_dims_identical(self, tmp_path):
        bundle = self._parse_bundle(str(tmp_path))
        model_fast = self._build_model(bundle, 'auto')
        model_legacy = self._build_model(bundle, 'schur_bfs')
        try:
            assert model_fast.tile_boundary_nodes == model_legacy.tile_boundary_nodes
            assert model_fast.tile_interior_counts == model_legacy.tile_interior_counts
        finally:
            model_fast.shutdown()
            model_legacy.shutdown()

    def test_dc_s_global_and_islands_identical(self, tmp_path):
        from distributed.solver import DistributedDDMSolver

        bundle = self._parse_bundle(str(tmp_path))
        model_fast = self._build_model(bundle, 'auto')
        model_legacy = self._build_model(bundle, 'schur_bfs')
        try:
            solver_fast = DistributedDDMSolver(model_fast)
            solver_legacy = DistributedDDMSolver(model_legacy)
            ctx_fast = solver_fast.prepare()
            ctx_legacy = solver_legacy.prepare()
            try:
                assert ctx_fast._removed_interface_nodes == ctx_legacy._removed_interface_nodes, (
                    "DC island sets diverge between the summaries union-find "
                    "and the legacy Schur-BFS"
                )
                S_fast = ctx_fast._S_global
                S_legacy = ctx_legacy._S_global
                assert S_fast.shape == S_legacy.shape
                # Same interface-node ORDER is not guaranteed (assembly order
                # can differ) -- compare via each side's own index map so the
                # comparison is node-identity-based, not position-based.
                nodes_fast = ctx_fast._interface_nodes
                nodes_legacy = ctx_legacy._interface_nodes
                assert set(nodes_fast) == set(nodes_legacy)
                idx_fast = ctx_fast._interface_node_to_idx
                idx_legacy = ctx_legacy._interface_node_to_idx
                perm = np.array([idx_legacy[n] for n in nodes_fast])
                S_legacy_reordered = S_legacy.tocsr()[perm][:, perm]
                diff = (S_fast.tocsr() - S_legacy_reordered)
                max_abs_diff = np.abs(diff.data).max() if diff.nnz else 0.0
                assert max_abs_diff == 0.0, (
                    f"S_global differs between summaries and legacy paths: "
                    f"max|diff|={max_abs_diff:.3e}"
                )

                v_fast = solver_fast.solve_dc(ctx_fast).flatten()
                v_legacy = solver_legacy.solve_dc(ctx_legacy).flatten()
                common = set(v_fast.keys()) & set(v_legacy.keys())
                assert len(common) > 0
                max_dv = max(abs(v_fast[n] - v_legacy[n]) for n in common)
                assert max_dv <= 1e-9, f"DC max|dV|={max_dv:.3e} V exceeds 1e-9 V"
            finally:
                ctx_fast.release()
                ctx_legacy.release()
        finally:
            model_fast.shutdown()
            model_legacy.shutdown()

    def test_transient_islands_and_solution_identical(self, tmp_path):
        from distributed.solver import DistributedDDMSolver

        bundle = self._parse_bundle(str(tmp_path))
        model_fast = self._build_model(bundle, 'auto')
        model_legacy = self._build_model(bundle, 'schur_bfs')
        try:
            solver_fast = DistributedDDMSolver(model_fast)
            solver_legacy = DistributedDDMSolver(model_legacy)
            dc_ctx_fast = solver_fast.prepare()
            dc_ctx_legacy = solver_legacy.prepare()
            trans_ctx_fast = solver_fast.prepare_transient(dt=100e-12, method='BE')
            trans_ctx_legacy = solver_legacy.prepare_transient(dt=100e-12, method='BE')
            try:
                assert (
                    trans_ctx_fast.topology.island_nodes_td
                    == trans_ctx_legacy.topology.island_nodes_td
                ), "Transient island sets diverge between summaries and legacy BFS"

                solver_fast.preprocess_sources(
                    time_step=100e-12, t_end=1e-9, smooth=False,
                )
                solver_legacy.preprocess_sources(
                    time_step=100e-12, t_end=1e-9, smooth=False,
                )
                r_fast = solver_fast.solve_transient(
                    trans_ctx_fast, dc_context=dc_ctx_fast,
                ).as_flat()
                r_legacy = solver_legacy.solve_transient(
                    trans_ctx_legacy, dc_context=dc_ctx_legacy,
                ).as_flat()
                common = set(r_fast.keys()) & set(r_legacy.keys())
                assert len(common) > 0
                max_dv = max(abs(r_fast[n][0] - r_legacy[n][0]) for n in common)
                assert max_dv <= 1e-8, f"Transient max|dV|={max_dv:.3e} V exceeds 1e-8 V"
            finally:
                dc_ctx_fast.release()
                dc_ctx_legacy.release()
                trans_ctx_fast.release()
                trans_ctx_legacy.release()
        finally:
            model_fast.shutdown()
            model_legacy.shutdown()


# ---------------------------------------------------------------------------
# (iv): tile-resident-pad fixture validated against a flat nodal-analysis
# oracle, plus the rescue WARNING
# ---------------------------------------------------------------------------

class TestTileResidentPadFlatOracle:
    """Validates the documented BFS divergence end-to-end against an
    independent ground truth.

    Design note (scope decision, documented in the implementation report):
    rather than routing this synthetic fixture through the full
    NetlistParser -> UnifiedPowerGridModel -> UnifiedIRDropSolver flat
    pipeline (which expects real .ckt-format PDN files and net-graph
    construction machinery this hand-built fixture does not produce), the
    "flat oracle" here is a direct nodal-analysis solve (G @ V = I with
    Dirichlet elimination) of the IDENTICAL resistor network using
    ``numpy.linalg.solve`` -- an independent, exact computation of the same
    physics (Kirchhoff's current law), not a re-derivation of the DDM
    machinery under test.  The comparison is exact to floating-point
    precision because DDM is algebraically exact for any partition.
    """

    def _build_model(self, island_detection, tmp_path):
        """Single-tile, single-COMPONENT model: a stub chain whose only
        interface node ('cutnode') hangs off a tile-resident Dirichlet pad
        ('padnode') -- the tile-resident-pad divergence scenario.

        Deliberately a single resistive component (no separate island-prone
        branch) so tile-level removal (_remove_floating_islands, which this
        hand-built TileData is NOT exempted from -- pre_cleaned_full defaults
        False) is a structural no-op (``len(components) <= 1``) regardless of
        island_detection mode, isolating the COORDINATOR-level divergence
        this test targets (the tile-level removal path is separately covered
        by test_island_summaries.py's TestPreCleanComponentSummaries).
        """
        import pickle
        from distributed.model import ParsedTileBundle, create_distributed_model
        from distributed.parser import PackageData, PowerGridMetaData, TileConfig
        from distributed.tile_parsing import TileData

        # stub0..stub3 chain ending in cutnode, which is Dirichlet-adjacent
        # to the tile-resident pad 'padnode' via a resistor.  cutnode is the
        # ONLY interface node in this (sole) component.
        stub_chain = ['stub0', 'stub1', 'stub2', 'stub3']
        stub_edges = [(stub_chain[i], stub_chain[i + 1], 1.0) for i in range(3)]
        stub_edges.append((stub_chain[0], 'cutnode', 1.0))
        stub_edges.append(('cutnode', 'padnode', 2.0))
        stub_current = 0.03

        all_nodes = set(stub_chain) | {'cutnode', 'padnode'}
        currents = {stub_chain[0]: stub_current}

        td = TileData(
            tile_id=(0, 0),
            resistive_edges=stub_edges,
            all_nodes=all_nodes,
            boundary_nodes=set(),
            current_injections=currents,
            capacitive_edges=[],
        )

        tile_dir = tmp_path / island_detection
        tile_dir.mkdir()
        with open(tile_dir / 'tile_0_0.pkl', 'wb') as f:
            pickle.dump(td, f, protocol=pickle.HIGHEST_PROTOCOL)

        vdd = 1.0
        pkg_data = PackageData(
            vsrc_dict={
                'V_padnode': {'node_pos': 'padnode', 'node_neg': '0', 'net': 'VDD', 'value': vdd},
            },
            package_edges=[],
            pad_nodes={'padnode'},
            tap_nodes=set(),
            die_attachment_nodes={'padnode'},
            vdd=vdd,
            net_name='VDD',
            package_cap_edges=[],
        )
        # 'cutnode' must be a genuine tile PORT (interface unknown) for it to
        # appear in S_global/interface_node_to_idx at all -- it is declared
        # here as a raw shared-boundary node (bypassing the real 2+-tile
        # discovery machinery, which this single-tile synthetic bundle does
        # not exercise).
        shared_boundary_nodes = {'cutnode'}
        metadata = PowerGridMetaData(
            tile_grid=(1, 1), parameters={},
            tile_configs=[TileConfig(
                tile_id=(0, 0), ckt_path='', nd_path=None,
                instance_path=None, net_filter=None,
            )],
            package_data=pkg_data, net_name='VDD', vdd=vdd,
        )
        with open(tile_dir / 'metadata.pkl', 'wb') as f:
            pickle.dump(
                {'metadata': metadata, 'boundary_nodes': shared_boundary_nodes},
                f, protocol=pickle.HIGHEST_PROTOCOL,
            )

        component_summaries = [
            {
                # F10 structural-completeness note: this component's own
                # 'padnode' node is ALSO a worker-reported port here (dirichlet
                # elimination happens at the coordinator, not per-tile, so
                # _build_block_system's port_nodes_local includes it too --
                # it's in die_attachment_nodes for this fixture) -- so it must
                # be listed as a candidate here too, exactly as the real
                # _pre_clean_tile_data would (port_nodes includes die_attachment
                # candidates, and 'padnode' is one for this fixture's
                # PackageData).  Omitting it trips the F10 runtime completeness
                # check and forces the legacy fallback.
                'candidates': frozenset({'cutnode', 'padnode'}),
                'n_nodes': len(stub_chain) + 2, 'has_pad': True,
                'tile_id': (0, 0),
            },
        ]
        bundle = ParsedTileBundle(
            metadata=metadata,
            shared_boundary_nodes=shared_boundary_nodes,
            pkl_dir=str(tile_dir),
            connectivity_summary_version=_connectivity_summary_version(),
            # == shared_boundary_nodes | die_attachment_nodes, matching what
            # create_distributed_model derives: {'cutnode', 'padnode'}.
            parser_interface_set=shared_boundary_nodes | {'padnode'},
            component_summaries=component_summaries,
        )
        model = create_distributed_model(
            bundle, backend='local', island_detection=island_detection,
        )
        return model, td, stub_chain, stub_current

    def _flat_oracle_cutnode_voltage(self, stub_chain, stub_current, vdd=1.0):
        """Direct nodal-analysis (G V = I, Dirichlet-eliminated) solve of the
        stub branch alone: stub0..stub3 chain, stub0 also -> cutnode -> padnode.

        Unknowns: stub0, stub1, stub2, stub3, cutnode (padnode is Dirichlet=vdd).
        """
        # Node order: stub0, stub1, stub2, stub3, cutnode
        nodes = list(stub_chain) + ['cutnode']
        n = len(nodes)
        idx = {name: i for i, name in enumerate(nodes)}
        G = np.zeros((n, n))
        I = np.zeros(n)

        def stamp(u, v, g):
            if u in idx:
                G[idx[u], idx[u]] += g
            if v in idx:
                G[idx[v], idx[v]] += g
            if u in idx and v in idx:
                G[idx[u], idx[v]] -= g
                G[idx[v], idx[u]] -= g

        for i in range(3):
            stamp(stub_chain[i], stub_chain[i + 1], 1.0)
        stamp(stub_chain[0], 'cutnode', 1.0)
        # cutnode -- padnode (Dirichlet at vdd): contributes to cutnode's
        # diagonal and to the RHS (I += g*vdd), padnode itself unknown-free.
        # NOTE: TileData.resistive_edges values are CONDUCTANCE (mS)
        # directly, not resistance -- the cutnode-padnode edge weight (2.0)
        # IS g_pad, matching _build_model's `('cutnode', 'padnode', 2.0)`.
        g_pad = 2.0
        G[idx['cutnode'], idx['cutnode']] += g_pad
        I[idx['cutnode']] += g_pad * vdd

        I[idx[stub_chain[0]]] -= stub_current  # sink = positive current draw

        V = np.linalg.solve(G, I)
        return dict(zip(nodes, V))

    @staticmethod
    def _solve_interface_voltages(model, ctx):
        """Manual, minimal interface-level DC solve: r = sum_i P_i^T g_i + rhs_dirichlet.

        NOT a call to ``DistributedDDMSolver.solve_dc`` -- this fixture (a
        single tile whose OWN port list mixes a Dirichlet pad ('padnode')
        with an ordinary unknown ('cutnode')) trips a genuine PRE-EXISTING
        bug discovered while writing this test: ``solve_dc``'s RHS scatter
        (``solver.py`` ``global_rhs`` assembly) pairs the per-tile reduced
        RHS ``g_i`` (length == n_ports, i.e. it still includes pad ports)
        against ``ctx.tile_index_maps[tid]`` (length == n_ports MINUS pads,
        per the existing ``if n in interface_node_to_idx`` filter also
        described in the plan's D1 finding for the tilewise CG matvec) ->
        an ``AssertionError`` on the length mismatch for ANY tile with a pad
        among its own ports.  Fixing that RHS-assembly bug is out of scope
        for Stage 1e (D1 is explicitly Stage 2 scope) and orthogonal to
        island detection, so this helper reproduces the CORRECT computation
        (filtering ``g_i`` to non-pad ports before the scatter-add, exactly
        matching what ``tile_index_maps`` already assumes) to validate the
        Stage 1e island-detection machinery (S_global assembly + island
        detection + interface solve) independently of that unrelated defect.
        """
        tid = model.metadata.tile_configs[0].tile_id
        worker = model.workers[0]
        g_i, _ = worker.get_reduced_rhs()
        port_nodes = worker._block_system.port_nodes
        idx_map = ctx.tile_index_maps[tid]
        kept = [i for i, n in enumerate(port_nodes) if n in ctx.interface_node_to_idx]
        assert len(kept) == len(idx_map)
        g_i_filtered = g_i[kept]

        n_interface = len(ctx.interface_nodes)
        global_rhs = np.zeros(n_interface, dtype=np.float64)
        np.add.at(global_rhs, idx_map, g_i_filtered)
        global_rhs = global_rhs + ctx.rhs_dirichlet_interface

        v_gamma = ctx.interface_lu(global_rhs)
        return {n: float(v_gamma[i]) for n, i in ctx.interface_node_to_idx.items()}

    def test_schur_bfs_wrongly_pins_cutnode_near_vdd(self, tmp_path):
        """Legacy path: the BFS cannot see padnode's adjacency (sliced into
        rhs_dirichlet) -> penalizes cutnode's whole component to ~Vdd."""
        from distributed.solver import DistributedDDMSolver

        model, td, stub_chain, stub_current = self._build_model('schur_bfs', tmp_path)
        try:
            solver = DistributedDDMSolver(model)
            ctx = solver.prepare()
            try:
                assert 'cutnode' in ctx._removed_interface_nodes, (
                    "Fixture setup error: expected the legacy BFS to wrongly "
                    "island 'cutnode' -- if this fails, the fixture no longer "
                    "reproduces the divergence scenario"
                )
                v = self._solve_interface_voltages(model, ctx)
                oracle = self._flat_oracle_cutnode_voltage(stub_chain, stub_current)
                # Penalty-pinned: far from the true (lower, current-loaded) voltage.
                assert abs(v['cutnode'] - oracle['cutnode']) > 1e-3
            finally:
                ctx.release()
        finally:
            model.shutdown()

    def test_summaries_path_matches_flat_oracle_and_warns(self, tmp_path, caplog):
        """Fast path: has_pad rescues cutnode's component; DC voltage matches
        the independent nodal-analysis oracle to solver precision, and the
        rescue WARNING is emitted."""
        from distributed.solver import DistributedDDMSolver

        model, td, stub_chain, stub_current = self._build_model('summaries', tmp_path)
        try:
            assert model.island_detection_mode == 'summaries'
            solver = DistributedDDMSolver(model)
            with caplog.at_level(logging.WARNING, logger='pgmath.schur'):
                ctx = solver.prepare()
            try:
                assert 'cutnode' not in ctx._removed_interface_nodes, (
                    "has_pad should have rescued 'cutnode' from island penalty"
                )
                v = self._solve_interface_voltages(model, ctx)
                oracle = self._flat_oracle_cutnode_voltage(stub_chain, stub_current)
                assert abs(v['cutnode'] - oracle['cutnode']) <= 1e-6, (
                    f"summaries-path DC voltage {v['cutnode']:.6f} V does not "
                    f"match the flat nodal-analysis oracle {oracle['cutnode']:.6f} V"
                )
            finally:
                ctx.release()
            assert any('rescued' in rec.message for rec in caplog.records), (
                "Expected the rescue WARNING to be emitted during prepare()"
            )
        finally:
            model.shutdown()


def _connectivity_summary_version():
    from distributed.parser import CONNECTIVITY_SUMMARY_VERSION
    return CONNECTIVITY_SUMMARY_VERSION


# ---------------------------------------------------------------------------
# Finding F1 (THE critical finding): parser.py's final shared_boundary_nodes/
# parser_interface_set must be derived from the PRE-clean (step-0 scan)
# boundary declarations, not from post-pre-clean (possibly shrunk) tile
# boundary sets.
# ---------------------------------------------------------------------------

class TestF1CrossTileKeptVsRemovedRegression:
    """Reproduces the exact cross-tile scenario from the F1 finding:

    A floating (no-ground, no-pad) component's interface candidates
    (``b1``..``b5``, 5 nodes, globally '*'-declared in BOTH tiles) are kept
    together as ONE component in tile B (>=5 local candidates -> kept by
    threshold) but appear in tile A only as 5 SEPARATE 1-candidate stub
    components (<5 each -> all removed by tile A's own pre-clean pass).

    Pre-fix: tile A's post-pre-clean boundary_nodes no longer contains
    b1..b5 (removed along with their stub components), so
    compute_shared_boundary_nodes (fed post-pre-clean sets) sees b1..b5 in
    only ONE tile (B) and drops them from the persisted interface set
    entirely -- demoting tile B's kept 5-node component to a portless,
    singular-G_ii interior island under summaries-mode trust.

    Post-fix: b1..b5 remain part of the persisted global interface set
    (derived from the PRE-clean scan, where they legitimately appear '*'-
    declared in 2+ tiles), exactly like the legacy path (which never ran
    pre-clean on whole tiles at all, so its interface-node set was always
    scan-based).  Both tiles solve without crashing, and (since neither side
    has any path to a pad) b1..b5 end up correctly penalty-pinned as a
    global island by BOTH the union-find and the Schur-BFS -- identically.
    """

    def _write_netlist(self, tmp_path):
        netlist_dir = tmp_path / 'netlist'
        netlist_dir.mkdir()

        # Tile A (0,0): 5 SEPARATE 1-node stubs s1..s5, each hanging off its
        # own boundary candidate b1..b5 (1 candidate each -- all removed by
        # tile A's whole-tile pre-clean, threshold=5).  Plus an unrelated
        # "main" component (m0..m3) tied to a die-attachment pad node,
        # sized larger than any individual stub so it -- not a stub -- is
        # tile A's "largest" component.
        tile_a = (
            "R_stub1 s1 *b1 100\n"
            "R_stub2 s2 *b2 100\n"
            "R_stub3 s3 *b3 100\n"
            "R_stub4 s4 *b4 100\n"
            "R_stub5 s5 *b5 100\n"
            "R_m0 m0 m1 100\n"
            "R_m1 m1 m2 100\n"
            "R_m2 m2 m3 100\n"
            "R_mpad m0 1000_100_M1 100\n"
            "I_m0 m0 0 1e-3\n"
        )
        (netlist_dir / 'tile_0_0.ckt').write_text(tile_a)

        # Tile B (1,0): b1..b5 chained TOGETHER into ONE 5-candidate
        # component (no ground, no pad -- genuinely floating/dead, kept by
        # the >=5-candidate threshold rule).  Plus an unrelated, LARGER
        # "n" chain (7 nodes incl. its own die-attachment pad) so the
        # victim component is kept via the THRESHOLD path, not by
        # accidentally being tile B's largest component.
        tile_b = (
            "R_v1 *b1 *b2 100\n"
            "R_v2 *b2 *b3 100\n"
            "R_v3 *b3 *b4 100\n"
            "R_v4 *b4 *b5 100\n"
            "R_n0 n0 n1 100\n"
            "R_n1 n1 n2 100\n"
            "R_n2 n2 n3 100\n"
            "R_n3 n3 n4 100\n"
            "R_n4 n4 n5 100\n"
            "R_npad n0 2000_100_M1 100\n"
            "I_n0 n0 0 1e-3\n"
        )
        (netlist_dir / 'tile_1_0.ckt').write_text(tile_b)

        (netlist_dir / 'package.ckt').write_text(
            "v_VDD VDD_vsrc 0 VDD\n"
            "r VDD_vsrc VDD_tap0 0.001\n"
            "r VDD_tap0 1000_100_M1 0.001\n"
            "r VDD_vsrc VDD_tap1 0.001\n"
            "r VDD_tap1 2000_100_M1 0.001\n"
        )
        (netlist_dir / 'pg_net_voltage').write_text("VDD 1.0\n")
        (netlist_dir / 'ckt.sp').write_text(
            ".partition_info 2 1\n"
            ".parameter VDD 1.0\n"
            ".include ./tile_0_0.ckt\n"
            ".include ./tile_1_0.ckt\n"
            ".include ./package.ckt\n"
        )
        return netlist_dir

    def _parse_bundle(self, tmp_path_str, netlist_dir):
        from distributed.parser import DistributedNetlistParser
        parser = DistributedNetlistParser(str(netlist_dir), net_filter=None)
        _, bundle = parser.parse_and_dump(tmp_path_str, backend='local')
        return bundle

    def test_kept_victim_candidates_survive_in_persisted_interface_set(self, tmp_path):
        """Core F1 assertion: b1..b5 remain in the persisted interface set
        even though tile A's local pre-clean removed all 5 of them from ITS
        OWN (post-clean) boundary_nodes."""
        netlist_dir = self._write_netlist(tmp_path)
        bundle = self._parse_bundle(str(tmp_path / 'pkl'), netlist_dir)

        victim = {'b1', 'b2', 'b3', 'b4', 'b5'}
        assert victim <= bundle.shared_boundary_nodes, (
            f"Expected b1..b5 in the persisted shared_boundary_nodes; "
            f"got {bundle.shared_boundary_nodes}"
        )
        assert victim <= bundle.parser_interface_set

        # Tile B's kept 5-node victim component summary must list all 5 as
        # candidates (proves it wasn't silently dropped from the summary).
        b_summaries = [
            s for s in bundle.component_summaries
            if s.get('tile_id') == (1, 0) and s['candidates'] == frozenset(victim)
        ]
        assert len(b_summaries) == 1, (
            f"Expected exactly one tile-B summary with candidates == "
            f"{victim}; summaries: {bundle.component_summaries}"
        )

        # Tile A must have genuinely removed all 5 stub components (they
        # cannot appear as candidates in ANY tile-A summary).
        a_candidate_union = {
            n for s in bundle.component_summaries
            if s.get('tile_id') == (0, 0) for n in s['candidates']
        }
        assert not (a_candidate_union & victim)

    def test_summaries_mode_matches_forced_legacy_no_crash_identical(self, tmp_path):
        """The critical regression: summaries mode must not crash (singular
        G_ii) and must match forced-legacy exactly -- island sets AND
        voltages (DDM exactness)."""
        from distributed.model import create_distributed_model
        from distributed.solver import DistributedDDMSolver

        netlist_dir = self._write_netlist(tmp_path)
        bundle = self._parse_bundle(str(tmp_path / 'pkl'), netlist_dir)

        model_fast = create_distributed_model(
            bundle, backend='local', island_detection='auto',
        )
        model_legacy = create_distributed_model(
            bundle, backend='local', island_detection='schur_bfs',
        )
        try:
            assert model_fast.island_detection_mode == 'summaries', (
                "Expected the fast path to engage -- if this fails, the "
                "trust assertion is failing and defeats this regression test"
            )
            assert model_legacy.island_detection_mode == 'schur_bfs'

            solver_fast = DistributedDDMSolver(model_fast)
            solver_legacy = DistributedDDMSolver(model_legacy)
            # No crash (singular G_ii) is itself part of the assertion --
            # prepare()/solve_dc() raising would fail this test.
            ctx_fast = solver_fast.prepare()
            ctx_legacy = solver_legacy.prepare()
            try:
                assert ctx_fast._removed_interface_nodes == ctx_legacy._removed_interface_nodes, (
                    "Island sets diverge between summaries union-find and "
                    "legacy Schur-BFS"
                )
                victim = {'b1', 'b2', 'b3', 'b4', 'b5'}
                assert victim <= ctx_fast._removed_interface_nodes, (
                    "Expected the padless victim component to be correctly "
                    "penalty-pinned as an island in BOTH modes"
                )

                v_fast = solver_fast.solve_dc(ctx_fast).flatten()
                v_legacy = solver_legacy.solve_dc(ctx_legacy).flatten()
                common = set(v_fast.keys()) & set(v_legacy.keys())
                assert len(common) > 0
                max_dv = max(abs(v_fast[n] - v_legacy[n]) for n in common)
                assert max_dv == 0.0, (
                    f"DC max|dV|={max_dv:.3e} V between summaries and legacy "
                    f"-- expected exact (DDM is algebraically exact)"
                )
                # Penalty-pinned victim nodes solve to ~Vdd on both sides.
                for n in victim:
                    if n in v_fast:
                        assert abs(v_fast[n] - 1.0) < 1e-6
            finally:
                ctx_fast.release()
                ctx_legacy.release()
        finally:
            model_fast.shutdown()
            model_legacy.shutdown()


# ---------------------------------------------------------------------------
# Findings F8/F9: lost floating-node diagnostics under summaries-mode trust
# ---------------------------------------------------------------------------

class TestFloatingNodesReportEqualitySummariesVsLegacy:
    """A genuinely-floating (0 interface candidates) component is removed by
    the UNIVERSAL parse-time pre-clean regardless of island_detection mode
    (the pkl on disk is already clean by the time either model loads it).
    Pre-fix, the removed node NAMES were never persisted (only ephemeral
    parse-time counts survived), so TileWorker.get_floating_nodes_data()
    reported ZERO removed nodes in BOTH modes -- a silent, universal
    regression the moment Stage 1e's parse-time pre-clean started running
    (not merely an 'auto' vs 'schur_bfs' divergence).  Post-fix, both modes
    correctly recover {'float1', 'float2'} via TileData.removed_floating_nodes.
    """

    def _write_netlist(self, tmp_path):
        netlist_dir = tmp_path / 'netlist'
        netlist_dir.mkdir()

        tile = (
            "R_m0 m0 m1 100\n"
            "R_m1 m1 m2 100\n"
            "R_mpad m0 1000_100_M1 100\n"
            "I_m0 m0 0 1e-3\n"
            # Genuinely floating: no ground, no pad, no boundary marker,
            # zero interface candidates -- removed regardless of threshold.
            "R_float float1 float2 100\n"
        )
        (netlist_dir / 'tile_0_0.ckt').write_text(tile)
        (netlist_dir / 'package.ckt').write_text(
            "v_VDD VDD_vsrc 0 VDD\n"
            "r VDD_vsrc VDD_tap0 0.001\n"
            "r VDD_tap0 1000_100_M1 0.001\n"
        )
        (netlist_dir / 'pg_net_voltage').write_text("VDD 1.0\n")
        (netlist_dir / 'ckt.sp').write_text(
            ".partition_info 1 1\n"
            ".parameter VDD 1.0\n"
            ".include ./tile_0_0.ckt\n"
            ".include ./package.ckt\n"
        )
        return netlist_dir

    def _parse_bundle(self, tmp_path_str, netlist_dir):
        from distributed.parser import DistributedNetlistParser
        parser = DistributedNetlistParser(str(netlist_dir), net_filter=None)
        _, bundle = parser.parse_and_dump(tmp_path_str, backend='local')
        return bundle

    def test_removed_component_persisted_on_tile_data(self, tmp_path):
        """Sanity: the parse-time pre-clean actually removed float1/float2
        from the pkl, and TileData.removed_floating_nodes records them."""
        import pickle

        netlist_dir = self._write_netlist(tmp_path)
        pkl_dir = tmp_path / 'pkl'
        bundle = self._parse_bundle(str(pkl_dir), netlist_dir)

        with open(pkl_dir / 'tile_0_0.pkl', 'rb') as f:
            td = pickle.load(f)
        assert 'float1' not in td.all_nodes
        assert 'float2' not in td.all_nodes
        assert td.removed_floating_nodes == {'float1', 'float2'}
        assert td.n_floating_components_removed == 1

    def test_floating_nodes_report_identical_and_correct_both_modes(self, tmp_path):
        from distributed.model import create_distributed_model
        from distributed.solver import DistributedDDMSolver
        from reports.floating_nodes import collect_floating_nodes_distributed

        netlist_dir = self._write_netlist(tmp_path)
        bundle = self._parse_bundle(str(tmp_path / 'pkl'), netlist_dir)

        model_fast = create_distributed_model(
            bundle, backend='local', island_detection='auto',
        )
        model_legacy = create_distributed_model(
            bundle, backend='local', island_detection='schur_bfs',
        )
        try:
            assert model_fast.island_detection_mode == 'summaries'
            assert model_legacy.island_detection_mode == 'schur_bfs'

            solver_fast = DistributedDDMSolver(model_fast)
            solver_legacy = DistributedDDMSolver(model_legacy)
            ctx_fast = solver_fast.prepare()
            ctx_legacy = solver_legacy.prepare()
            try:
                data_fast = collect_floating_nodes_distributed(
                    model_fast, ctx_fast, model_fast.workers, net_name='VDD',
                )
                data_legacy = collect_floating_nodes_distributed(
                    model_legacy, ctx_legacy, model_legacy.workers, net_name='VDD',
                )
                expected = {'float1', 'float2'}
                assert data_fast.dropped_connectivity == expected, (
                    f"summaries-mode floating-nodes report missing the "
                    f"parse-time-removed component: {data_fast.dropped_connectivity}"
                )
                assert data_legacy.dropped_connectivity == expected
                assert data_fast.dropped_connectivity == data_legacy.dropped_connectivity

                # model.island_stats (F8/F9 + R2): summaries mode must
                # recover the parse-time removal count via
                # TileData.n_floating_components_removed.  Finding R2: the
                # forced-legacy worker's FRESH _remove_floating_islands call
                # finds nothing left to remove (the pkl already arrived
                # clean -- Stage 1e's parse-time pre-clean runs UNIVERSALLY,
                # regardless of which island_detection mode a LATER model
                # creation requests), so schur_bfs mode must ALSO surface the
                # parse-time count (added on top of its own fresh-removal
                # count in tile_worker._build_block_system) rather than
                # silently reporting 0 -- both modes must report IDENTICAL
                # totals for the same bundle.
                assert model_fast.island_stats[(0, 0)]['islands_removed'] == 1
                assert model_legacy.island_stats[(0, 0)]['islands_removed'] == 1, (
                    "R2 regression: forced-legacy (schur_bfs) mode must also "
                    "surface the parse-time removal count, not just fresh "
                    "worker-time removal (which finds nothing left to remove "
                    "on an already-pre-cleaned bundle)"
                )
                assert (
                    model_fast.island_stats[(0, 0)]['islands_removed']
                    == model_legacy.island_stats[(0, 0)]['islands_removed']
                ), "island_stats must be IDENTICAL between summaries and schur_bfs modes"
            finally:
                ctx_fast.release()
                ctx_legacy.release()
        finally:
            model_fast.shutdown()
            model_legacy.shutdown()


# ---------------------------------------------------------------------------
# Finding R1: parse-end consistency re-check (split-side demotion)
# ---------------------------------------------------------------------------

class TestSplitSideDemotionConsistencyCheck:
    """Reproduces the R1 failure scenario end-to-end via the REAL
    ``parse_and_dump`` pipeline (real B1 splitting -- no monkeypatched
    ``split_tile``):

    Node 'X' is '*'-declared in exactly two tiles: whole tile W and
    oversized tile A (which B1-splits).  A also contains a tiny fragment
    {X, fy} whose sole interface candidate is X (count=1 < the whole-tile
    threshold=5) -- A's OWN whole-tile pre-clean (parse_and_dump_tile, which
    runs on EVERY tile before any split decision) removes this fragment
    BEFORE A is ever split, so X is absent from A's post-clean
    boundary_nodes.  Since A is subsequently split (3-tuple sub-tile ids),
    parser.py's step-3b raw-boundary substitution is EXEMPT for A's
    contribution (only whole/2-tuple tiles get the pre-clean raw-scan
    substitution) -- so X's declaring-tile count drops to 1 (only W) in the
    FINAL shared_boundary_nodes, and X is demoted out of the final
    interface set entirely.

    Meanwhile W's component C = {hub, X, G1, G2, G3, G4} was kept at W's
    OWN whole-tile pre-clean because X was one of exactly 5 candidates
    (X, G1..G4; G1..G4 are separately '*'-declared a second time in tile B,
    making them globally shared too) reaching the keep threshold.  Once X is
    demoted, C's candidate overlap with the FINAL interface set drops to 4
    (G1..G4 only) -- below the threshold that justified keeping C.  Neither
    the F10 structural-completeness check nor the
    interface_nodes == parser_interface_set trust assertion observes this
    (both see the ORIGINAL candidate list, not that it has silently shrunk
    against the final interface set) -- only the R1 consistency check does.

    Without the R1 check, 'auto' mode would trust the summaries: W's tile
    keeps {hub, X, G1..G4} intact (worker-side removal skipped under trust),
    with X folded into W's INTERIOR (no longer a port).  'schur_bfs' mode's
    FRESH worker-time removal, however, evaluates C's candidates against the
    FINAL interface set (G1..G4 only, count=4 < threshold=5) and REMOVES the
    whole component C (hub, X, G1..G4) from W entirely -- a genuine
    topology divergence between the two modes.  This test is designed to
    FAIL (via the island_detection_mode / island_stats / DC-parity
    assertions below) if the R1 consistency check is removed.
    """

    N_CHAIN = 20
    MAX_INTERIOR = 8

    def _write_netlist(self, tmp_path):
        netlist_dir = tmp_path / 'netlist'
        netlist_dir.mkdir()

        # Tile W (0,0): 'main' (w0..w6, 7 nodes, no candidates -> always the
        # tile's largest, unconditionally kept) + component C (hub + 5
        # candidates X,G1..G4, 6 nodes) kept via threshold=5 -- X is one of
        # the 5 candidates that pushed C over the keep threshold.  W's own
        # interior count (8: 7 from main + 1 hub) stays <= MAX_INTERIOR so W
        # remains a WHOLE (2-tuple) tile -- required for the step-3b raw-
        # boundary substitution to apply to it.
        # 'main' also anchors to its OWN globally-shared port (W2, declared
        # '*' a second time in tile B below) so that W always has >= 1 port
        # regardless of whether component C survives -- keeps this fixture
        # scoped to the R1 consistency-check gap, not a separate zero-port-
        # tile edge case in the Schur-complement assembly.
        tile_w = (
            "R_w0 w0 w1 100\n"
            "R_w1 w1 w2 100\n"
            "R_w2 w2 w3 100\n"
            "R_w3 w3 w4 100\n"
            "R_w4 w4 w5 100\n"
            "R_w5 w5 w6 100\n"
            "R_w6b w6 *W2 10\n"
            "I_w0 w0 0 1e-3\n"
            "R_c0 hub *X 10\n"
            "R_c1 hub *G1 10\n"
            "R_c2 hub *G2 10\n"
            "R_c3 hub *G3 10\n"
            "R_c4 hub *G4 10\n"
        )
        (netlist_dir / 'tile_0_0.ckt').write_text(tile_w)

        # Tile B (0,1): G1..G4 declared '*' a SECOND time -- makes them
        # globally shared (2+ raw declarations).  Also re-declares 'W2' (see
        # tile W above) so it too is globally shared.  Zero interior (every
        # resistor endpoint is '*'-marked); unrelated to W/A's physics
        # otherwise.
        tile_b = (
            "R_g1 *G1 *G2 1\n"
            "R_g2 *G2 *G3 1\n"
            "R_g3 *G3 *G4 1\n"
            "R_w2anchor *W2 *G1 1\n"
        )
        (netlist_dir / 'tile_0_1.ckt').write_text(tile_b)

        # Tile A (1,0): a long coordinate-named chain (20 nodes, no '*'
        # markers, tied to ground at one end) that B1-splits under
        # MAX_INTERIOR, PLUS a tiny fragment {X, fy} whose sole interface
        # candidate is X.  'X' is declared '*' in exactly A and W (2 raw
        # declarations -> globally shared at the step-0 scan), but the
        # fragment's single candidate (count=1) is below the whole-tile
        # threshold=5, so A's OWN whole-tile pre-clean (which runs on EVERY
        # tile, including ones that will later split, BEFORE any split is
        # decided) removes {X, fy} before A is ever split.
        chain_nodes = [f'{100 * (i + 1)}_100_M1' for i in range(self.N_CHAIN)]
        lines = [
            f"R_a{i} {chain_nodes[i]} {chain_nodes[i + 1]} 100\n"
            for i in range(self.N_CHAIN - 1)
        ]
        lines.append(f"R_gnd {chain_nodes[0]} 0 0.001\n")
        lines.append("R_f *X fy 100\n")
        (netlist_dir / 'tile_1_0.ckt').write_text(''.join(lines))

        (netlist_dir / 'package.ckt').write_text("v_VDD VDD_vsrc 0 VDD\n")
        (netlist_dir / 'pg_net_voltage').write_text("VDD 1.0\n")
        (netlist_dir / 'ckt.sp').write_text(
            ".partition_info 2 2\n"
            ".parameter VDD 1.0\n"
            ".include ./tile_0_0.ckt\n"
            ".include ./tile_0_1.ckt\n"
            ".include ./tile_1_0.ckt\n"
            ".include ./package.ckt\n"
        )
        return netlist_dir

    def _parse_bundle(self, tmp_path, netlist_dir):
        from distributed.parser import DistributedNetlistParser

        parser = DistributedNetlistParser(str(netlist_dir), net_filter=None)
        _, bundle = parser.parse_and_dump(
            str(tmp_path / 'pkl'), backend='local', max_interior=self.MAX_INTERIOR,
        )
        return bundle

    def _assert_fixture_reproduces_scenario(self, bundle):
        """Sanity checks that the fixture actually exercises the R1 gap
        (not merely a differently-broken setup) -- if these fail, the
        fixture itself needs adjustment, not the production code."""
        # Tile A really did split (3-tuple sub-tile ids present).
        split_ids = [tid for tid in bundle.metadata.tile_configs if len(tid.tile_id) == 3]
        assert split_ids, (
            "Fixture setup error: tile A (1,0) did not split -- "
            "increase N_CHAIN or lower MAX_INTERIOR"
        )
        # X was genuinely demoted out of the final interface set.
        assert 'X' not in bundle.shared_boundary_nodes, (
            "Fixture setup error: 'X' unexpectedly survived in the final "
            "shared_boundary_nodes -- the split-side demotion did not occur"
        )
        # G1..G4 remain genuinely shared (W's component C still has 4
        # candidates -- one short of the threshold-5 decision that kept it).
        assert {'G1', 'G2', 'G3', 'G4'} <= bundle.shared_boundary_nodes

    def test_consistency_check_drops_summaries_with_info_log(self, tmp_path, caplog):
        """(a) summaries are dropped at parse (version None) with an INFO
        log naming the offending tile/component."""
        import logging as _logging

        netlist_dir = self._write_netlist(tmp_path)
        with caplog.at_level(_logging.INFO, logger='distributed.parser'):
            bundle = self._parse_bundle(tmp_path, netlist_dir)

        self._assert_fixture_reproduces_scenario(bundle)

        assert bundle.connectivity_summary_version is None, (
            "R1 regression: the parse-end consistency check should have "
            "dropped connectivity_summary_version to None for this bundle "
            "(W's component C no longer satisfies its own keep decision "
            "against the final interface set)"
        )
        assert any(
            'consistency check' in rec.message and '(0, 0)' in rec.message
            for rec in caplog.records
        ), (
            "Expected an INFO log naming tile (0, 0) from the parse-end "
            "consistency check"
        )

    def test_model_resolves_to_legacy(self, tmp_path):
        """(b) model creation resolves to the legacy schur_bfs path even
        when 'auto' is requested, because the dropped
        connectivity_summary_version fails _resolve_island_detection's
        version check."""
        from distributed.model import create_distributed_model

        netlist_dir = self._write_netlist(tmp_path)
        bundle = self._parse_bundle(tmp_path, netlist_dir)
        self._assert_fixture_reproduces_scenario(bundle)

        model = create_distributed_model(bundle, backend='local', island_detection='auto')
        try:
            assert model.island_detection_mode == 'schur_bfs', (
                "R1 regression: 'auto' must resolve to schur_bfs on a "
                "bundle whose consistency check failed -- if this is "
                "'summaries' instead, the R1 check was not wired into "
                "connectivity_summary_version"
            )
        finally:
            model.shutdown()

    def test_results_identical_to_forced_schur_bfs_no_crash(self, tmp_path):
        """(c) DC results from 'auto' are identical to forced 'schur_bfs' on
        the SAME bundle, and neither crashes.  This is the assertion
        designed to FAIL if the R1 consistency check is removed: without
        it, 'auto' would trust the (unsound) summaries and diverge from
        'schur_bfs' -- W's component C would survive intact (with X folded
        to interior) under trust, vs. being entirely removed by
        schur_bfs's fresh worker-time removal (C's candidates against the
        FINAL interface set = {G1..G4}, count=4 < threshold=5)."""
        from distributed.model import create_distributed_model
        from distributed.solver import DistributedDDMSolver

        netlist_dir = self._write_netlist(tmp_path)
        bundle = self._parse_bundle(tmp_path, netlist_dir)
        self._assert_fixture_reproduces_scenario(bundle)

        model_auto = create_distributed_model(bundle, backend='local', island_detection='auto')
        model_legacy = create_distributed_model(
            bundle, backend='local', island_detection='schur_bfs',
        )
        try:
            assert model_auto.island_detection_mode == 'schur_bfs'
            assert model_legacy.island_detection_mode == 'schur_bfs'

            solver_auto = DistributedDDMSolver(model_auto)
            solver_legacy = DistributedDDMSolver(model_legacy)
            # No crash (e.g. singular G_ii from a portless component) is
            # itself part of the assertion.
            ctx_auto = solver_auto.prepare()
            ctx_legacy = solver_legacy.prepare()
            try:
                v_auto = solver_auto.solve_dc(ctx_auto).flatten()
                v_legacy = solver_legacy.solve_dc(ctx_legacy).flatten()
                common = set(v_auto.keys()) & set(v_legacy.keys())
                assert len(common) > 0
                max_dv = max(abs(v_auto[n] - v_legacy[n]) for n in common)
                assert max_dv == 0.0, (
                    f"DC max|dV|={max_dv:.3e} V between 'auto' and "
                    f"'schur_bfs' -- expected exact identity (both should "
                    f"resolve to the SAME legacy code path once the R1 "
                    f"check drops trust for this bundle)"
                )
                # 'hub' must be consistently absent (schur_bfs removed
                # component C) or consistently present in BOTH -- never
                # split between the two modes.
                assert ('hub' in v_auto) == ('hub' in v_legacy)
            finally:
                ctx_auto.release()
                ctx_legacy.release()
        finally:
            model_auto.shutdown()
            model_legacy.shutdown()

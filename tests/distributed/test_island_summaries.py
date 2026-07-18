"""Stage 1e: parse-time island pre-clean + connectivity summaries + coordinator
union-find island detection.

Covers plan items (ii), (iii), (v), (vi), and the core mechanism of (iv)
(rescue + WARNING).  End-to-end parity (i), the full netlist_test DC/TD
identical-island-set assertion (vii), and the flat-oracle-validated
tile-resident-pad scenario (iv, full) live in
``test_island_summaries_integration.py`` (marked ``integration``).

  1. _pre_clean_tile_data component-summary output (candidates/n_nodes/has_pad)
  2. detect_interface_islands_from_summaries oracle equivalence vs
     find_interface_islands on engineered fixtures (item ii):
       - pad-less boundary component
       - cap-coupled-only component (DC island, TD-with-package-cap not)
       - package-resistor-bridged component
       - multi-component single tile (structural zero)
       - removal-created island (unreached singleton == BFS zero-row island)
  3. Tile-resident-pad divergence: summary rescues a node the BFS would
     wrongly island; rescue WARNING is emitted (item iv, core mechanism)
  4. Sub-tile threshold-1-at-parse fixture: a component connected only
     through a cut-interface node survives the split-path pre-clean (item iii)
  5. Trust-assertion / fallback-matrix resolution (model._resolve_island_detection)
     (items v, vi)
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pytest
import scipy.sparse as sp

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_laplacian(nodes: List[str], edges: List[Tuple[str, str, float]]) -> sp.csr_matrix:
    """Build a sparse Laplacian (off-diag = -g, diag = row sum) for *nodes*."""
    idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)
    data, rows, cols = [], [], []
    diag = np.zeros(n, dtype=np.float64)
    for u, v, g in edges:
        i, j = idx[u], idx[v]
        rows.extend([i, j])
        cols.extend([j, i])
        data.extend([-g, -g])
        diag[i] += g
        diag[j] += g
    for k in range(n):
        rows.append(k)
        cols.append(k)
        data.append(diag[k])
    return sp.csr_matrix((data, (rows, cols)), shape=(n, n))


def _run_oracle_and_summaries(
    nodes: List[str],
    s_edges: List[Tuple[str, str, float]],
    component_summaries: List[Dict],
    pad_nodes: Set[str],
    extra_edges,
) -> Tuple[Set[str], Set[str]]:
    """Return (bfs_islands, summary_islands) for the same logical system.

    *s_edges* populate S_global's off-diagonal (same-tile-component
    adjacency); *component_summaries* is the Stage 1e parse-time summary
    list representing the identical grouping.  Both detectors additionally
    see *extra_edges* (package-level) and *pad_nodes* identically --
    find_interface_islands also treats extra_edges as iface-iface adjacency
    (redundant with S_global in production; sufficient on its own here).
    """
    from pgmath.schur import detect_interface_islands_from_summaries, find_interface_islands

    idx = {n: i for i, n in enumerate(nodes)}
    S = _make_laplacian(nodes, s_edges)

    bfs_islands = find_interface_islands(S, nodes, idx, pad_nodes, extra_edges)
    summary_islands = detect_interface_islands_from_summaries(
        component_summaries, idx, pad_nodes, extra_edges,
    )
    return bfs_islands, summary_islands


# ──────────────────────────────────────────────────────────────────────
# 1. _pre_clean_tile_data component-summary output
# ──────────────────────────────────────────────────────────────────────

class TestPreCleanComponentSummaries:
    """_pre_clean_tile_data's new (n_removed, component_summaries) return."""

    def test_kept_component_summary_fields(self):
        from distributed.retile import _pre_clean_tile_data
        from distributed.tile_parsing import TileData

        td = TileData(
            tile_id=(0, 0),
            resistive_edges=[('n0', 'n1', 1.0), ('n0', 'shared', 5.0), ('n1', '0', 0.5)],
            all_nodes={'n0', 'n1', 'shared'},
            boundary_nodes={'shared'},
            current_injections={'n0': 0.1},
            capacitive_edges=[],
        )
        n_removed, summaries = _pre_clean_tile_data(td, {'shared'})

        assert n_removed == 0
        assert len(summaries) == 1
        comp = summaries[0]
        assert comp['candidates'] == frozenset({'shared'})
        assert comp['n_nodes'] == 3
        assert comp['has_pad'] is False

    def test_removed_component_has_no_summary(self):
        """Removed (floating) components do not appear in component_summaries."""
        from distributed.retile import _pre_clean_tile_data
        from distributed.tile_parsing import TileData

        td = TileData(
            tile_id=(0, 0),
            resistive_edges=[
                ('n0', 'n1', 1.0), ('n0', 'shared', 5.0), ('n1', '0', 0.5),
                ('island_a', 'island_b', 1.0),
            ],
            all_nodes={'n0', 'n1', 'shared', 'island_a', 'island_b'},
            boundary_nodes={'shared'},
            current_injections={},
            capacitive_edges=[],
        )
        n_removed, summaries = _pre_clean_tile_data(td, {'shared'})

        assert n_removed == 1
        # Only the surviving {n0, n1, shared} component is summarized.
        assert len(summaries) == 1
        assert summaries[0]['candidates'] == frozenset({'shared'})

    def test_has_pad_flag_true_for_tile_resident_pad(self):
        from distributed.retile import _pre_clean_tile_data
        from distributed.tile_parsing import TileData

        td = TileData(
            tile_id=(0, 0),
            resistive_edges=[('stub', 'padnode', 2.0)],
            all_nodes={'stub', 'padnode'},
            boundary_nodes=set(),
            current_injections={'stub': 0.05},
            capacitive_edges=[],
        )
        n_removed, summaries = _pre_clean_tile_data(
            td, port_nodes={'padnode'}, pad_nodes={'padnode'},
        )
        assert n_removed == 0
        assert len(summaries) == 1
        assert summaries[0]['has_pad'] is True
        assert summaries[0]['candidates'] == frozenset({'padnode'})

    def test_pre_cleaned_full_flag_set_whole_tile_does_not_set_pre_cleaned(self):
        """pre_cleaned_full=True regardless of threshold, distinct from pre_cleaned.

        Finding F3: a whole-tile-style call (mark_split_pre_cleaned=False,
        the default -- e.g. parse_and_dump_tile's universal pre-clean pass)
        must NOT set tile_data.pre_cleaned=True; only pre_cleaned_full.
        Setting pre_cleaned=True unconditionally would flip the legacy
        fallback's MIN_INTERFACE_NODES_KEEP threshold from 5 to 1 for every
        never-split tile.
        """
        from distributed.retile import _pre_clean_tile_data
        from distributed.tile_parsing import TileData

        td = TileData(
            tile_id=(0, 0),
            resistive_edges=[('n0', 'shared', 1.0)],
            all_nodes={'n0', 'shared'},
            boundary_nodes={'shared'},
            current_injections={},
            capacitive_edges=[],
        )
        assert td.pre_cleaned_full is False
        assert td.pre_cleaned is False
        _pre_clean_tile_data(td, {'shared'}, min_interface_keep=1)
        assert td.pre_cleaned_full is True
        assert td.pre_cleaned is False

    def test_mark_split_pre_cleaned_sets_pre_cleaned(self):
        """Finding F3: the genuine sub-tile pass (mark_split_pre_cleaned=True,
        as used by _apply_tile_splits) DOES set tile_data.pre_cleaned=True,
        selecting the legacy fallback's threshold=1 for this sub-tile."""
        from distributed.retile import _pre_clean_tile_data
        from distributed.tile_parsing import TileData

        td = TileData(
            tile_id=(0, 0, 0),
            resistive_edges=[('n0', 'shared', 1.0)],
            all_nodes={'n0', 'shared'},
            boundary_nodes={'shared'},
            current_injections={},
            capacitive_edges=[],
        )
        assert td.pre_cleaned is False
        _pre_clean_tile_data(
            td, {'shared'}, min_interface_keep=1, mark_split_pre_cleaned=True,
        )
        assert td.pre_cleaned_full is True
        assert td.pre_cleaned is True

    def test_multi_component_single_tile_separate_summaries(self):
        """Two disconnected components in one tile -> two separate summary dicts."""
        from distributed.retile import _pre_clean_tile_data
        from distributed.tile_parsing import TileData

        td = TileData(
            tile_id=(0, 0),
            resistive_edges=[
                ('e1', 'e1b', 1.0), ('e1b', 'pad1', 10.0),  # comp 1: reaches pad1
                ('e2', 'e2b', 1.0), ('e2b', 'e2c', 1.0),    # comp 2: no pad
            ],
            all_nodes={'e1', 'e1b', 'pad1', 'e2', 'e2b', 'e2c'},
            boundary_nodes={'e1', 'pad1', 'e2'},
            current_injections={},
            capacitive_edges=[],
        )
        n_removed, summaries = _pre_clean_tile_data(
            td, port_nodes={'e1', 'pad1', 'e2'}, min_interface_keep=1,
        )
        assert n_removed == 0
        assert len(summaries) == 2
        candidate_sets = {s['candidates'] for s in summaries}
        assert frozenset({'e1', 'pad1'}) in candidate_sets
        assert frozenset({'e2'}) in candidate_sets

    def test_is_largest_and_keep_threshold_fields(self):
        """Finding R1: each summary carries 'is_largest' (the unconditionally
        -kept largest component never depended on the threshold) and
        'keep_threshold' (the min_interface_keep value this pass used) --
        needed by tile_parsing.verify_component_keep_decisions."""
        from distributed.retile import _pre_clean_tile_data
        from distributed.tile_parsing import TileData

        td = TileData(
            tile_id=(0, 0),
            resistive_edges=[
                # 'main' component: 3 nodes, no candidates -- unconditionally
                # kept as the largest.
                ('m0', 'm1', 1.0), ('m1', 'm2', 1.0),
                # 'small' component: 2 nodes, 1 candidate -- kept via the
                # threshold (min_interface_keep=1 here).
                ('s0', 'cand', 1.0),
            ],
            all_nodes={'m0', 'm1', 'm2', 's0', 'cand'},
            boundary_nodes={'cand'},
            current_injections={},
            capacitive_edges=[],
        )
        n_removed, summaries = _pre_clean_tile_data(
            td, port_nodes={'cand'}, min_interface_keep=1,
        )
        assert n_removed == 0
        by_candidates = {s['candidates']: s for s in summaries}
        small = by_candidates[frozenset({'cand'})]
        assert small['is_largest'] is False
        assert small['keep_threshold'] == 1
        # 'main' has no candidates at all -> no summary is emitted for it
        # (summaries only cover components with >=1 candidate), but the
        # 'small' summary above is sufficient to prove the fields exist and
        # are correctly threaded through.


# ──────────────────────────────────────────────────────────────────────
# 1b. R1: parse-end consistency re-check (verify_component_keep_decisions)
# ──────────────────────────────────────────────────────────────────────

class TestVerifyComponentKeepDecisions:
    """tile_parsing.verify_component_keep_decisions: the standalone,
    directly-testable core of the R1 parse-end consistency re-check.  A
    non-largest component's keep decision (>= keep_threshold candidates AT
    THE PASS THAT DECIDED IT) must still hold against the FINAL interface
    set; the largest component is exempt (its keep never depended on the
    threshold)."""

    def test_all_satisfied_ok(self):
        from distributed.tile_parsing import verify_component_keep_decisions

        summaries = [
            {'candidates': frozenset({'a', 'b', 'c'}), 'is_largest': False, 'keep_threshold': 2},
            {'candidates': frozenset({'x'}), 'is_largest': True, 'keep_threshold': 5},
        ]
        ok, violations = verify_component_keep_decisions(summaries, {'a', 'b', 'c', 'x'})
        assert ok is True
        assert violations == []

    def test_demoted_candidate_below_threshold_fails(self):
        """Core R1 scenario: a component kept because it had 5 candidates at
        parse time now has only 4 against the final interface set (one
        candidate was demoted -- e.g. dropped from the final
        shared_boundary_nodes by a split neighbor's parse-time clean)."""
        from distributed.tile_parsing import verify_component_keep_decisions

        summaries = [{
            'candidates': frozenset({'x', 'g1', 'g2', 'g3', 'g4'}),
            'is_largest': False,
            'keep_threshold': 5,
            'tile_id': (0, 0),
            'n_nodes': 6,
        }]
        # 'x' has been demoted -- absent from the final interface set.
        final_interface_set = {'g1', 'g2', 'g3', 'g4'}
        ok, violations = verify_component_keep_decisions(summaries, final_interface_set)
        assert ok is False
        assert len(violations) == 1
        assert violations[0]['tile_id'] == (0, 0)

    def test_largest_component_exempt_even_if_all_candidates_demoted(self):
        """The unconditionally-kept largest component must NOT be flagged,
        even if every one of its listed candidates is later demoted --
        proves the check is unsound (over-triggers) without the is_largest
        exemption, which would make this test FAIL if the exemption were
        removed."""
        from distributed.tile_parsing import verify_component_keep_decisions

        summaries = [{
            'candidates': frozenset({'x'}),
            'is_largest': True,
            'keep_threshold': 5,
        }]
        ok, violations = verify_component_keep_decisions(summaries, set())
        assert ok is True
        assert violations == []

    def test_missing_keep_threshold_degrades_safely(self):
        """A summary predating this field (shouldn't happen post-version-2,
        but must not crash) is skipped, not flagged."""
        from distributed.tile_parsing import verify_component_keep_decisions

        summaries = [{'candidates': frozenset({'x'}), 'is_largest': False}]
        ok, violations = verify_component_keep_decisions(summaries, set())
        assert ok is True
        assert violations == []


# ──────────────────────────────────────────────────────────────────────
# 2. detect_interface_islands_from_summaries oracle equivalence (item ii)
# ──────────────────────────────────────────────────────────────────────

class TestOracleEquivalenceVsSchurBFS:
    """Engineered fixtures where the union-find must match find_interface_islands."""

    def test_padless_boundary_component(self):
        """Component with interface candidates but no path to any pad -> island."""
        nodes = ['A', 'B']
        summaries = [{'candidates': frozenset({'A', 'B'}), 'n_nodes': 2, 'has_pad': False}]
        bfs, summ = _run_oracle_and_summaries(
            nodes, s_edges=[('A', 'B', 1.0)],
            component_summaries=summaries, pad_nodes=set(), extra_edges=None,
        )
        assert bfs == {'A', 'B'}
        assert summ == bfs

    def test_cap_coupled_only_component_dc_vs_td(self):
        """DC (resistive-only extra_edges): A is an island.

        TD (extra_edges includes the weighted package-cap edge A-B): the
        cap bridge makes A live through B's pad connection -> no islands.
        Same component_summaries serve both modes (tile-local components are
        cap-independent); only extra_edges differs, mirroring how the real
        DC/TD callers pass package_edges vs combined_edges.
        """
        nodes = ['A', 'B']
        summaries = [
            {'candidates': frozenset({'A'}), 'n_nodes': 1, 'has_pad': False},
            {'candidates': frozenset({'B'}), 'n_nodes': 1, 'has_pad': False},
        ]
        pad_nodes = {'PAD1'}
        dc_extra = [('B', 'PAD1', 10.0)]
        td_extra = [('B', 'PAD1', 10.0), ('A', 'B', 0.02)]  # C_coeff*c_fF > 0

        bfs_dc, summ_dc = _run_oracle_and_summaries(
            nodes, s_edges=[], component_summaries=summaries,
            pad_nodes=pad_nodes, extra_edges=dc_extra,
        )
        assert bfs_dc == {'A'}
        assert summ_dc == bfs_dc

        bfs_td, summ_td = _run_oracle_and_summaries(
            nodes, s_edges=[], component_summaries=summaries,
            pad_nodes=pad_nodes, extra_edges=td_extra,
        )
        assert bfs_td == set()
        assert summ_td == bfs_td

    def test_package_resistor_bridged_component(self):
        """Two separate tile-local components bridged by a package resistor.

        comp1={C1a, C1b} (no pad); comp2={C2a} reaches PAD1 via a package
        edge; a SEPARATE package resistor bridges C1a-C2a.  Both C1* and C2a
        end up live; the unrelated standalone D remains an island.
        """
        nodes = ['C1a', 'C1b', 'C2a', 'D']
        summaries = [
            {'candidates': frozenset({'C1a', 'C1b'}), 'n_nodes': 2, 'has_pad': False},
            {'candidates': frozenset({'C2a'}), 'n_nodes': 1, 'has_pad': False},
            {'candidates': frozenset({'D'}), 'n_nodes': 1, 'has_pad': False},
        ]
        pad_nodes = {'PAD1'}
        extra_edges = [('C1a', 'C2a', 5.0), ('C2a', 'PAD1', 3.0)]

        bfs, summ = _run_oracle_and_summaries(
            nodes, s_edges=[('C1a', 'C1b', 1.0)],
            component_summaries=summaries, pad_nodes=pad_nodes, extra_edges=extra_edges,
        )
        assert bfs == {'D'}
        assert summ == bfs

    def test_multi_component_single_tile_structural_zero(self):
        """Same tile, two components with NO union between them (structural zero).

        E1 reaches PAD2; E2 does not -> only E2 is an island, and critically
        E1/E2 are never unioned together despite originating from the same
        tile (S_i[E1,E2] == 0.0 exactly, the BFS's value-aware skip).
        """
        nodes = ['E1', 'E2']
        summaries = [
            {'candidates': frozenset({'E1'}), 'n_nodes': 1, 'has_pad': False},
            {'candidates': frozenset({'E2'}), 'n_nodes': 1, 'has_pad': False},
        ]
        pad_nodes = {'PAD2'}
        extra_edges = [('E1', 'PAD2', 4.0)]

        bfs, summ = _run_oracle_and_summaries(
            nodes, s_edges=[],  # no S_i[E1,E2] coupling -- different components
            component_summaries=summaries, pad_nodes=pad_nodes, extra_edges=extra_edges,
        )
        assert bfs == {'E2'}
        assert summ == bfs

    def test_removal_created_island_unreached_singleton(self):
        """X, Y absent from ANY summary (removed in both adjacent tiles) but
        still appear in interface_node_to_idx via a package edge to each
        other (neither reaches a pad) -> both are islands, matching the
        BFS's zero-row-island case for an unreached node.
        """
        nodes = ['X', 'Y']
        summaries: List[Dict] = []  # neither X nor Y summarized anywhere
        extra_edges = [('X', 'Y', 2.0)]

        bfs, summ = _run_oracle_and_summaries(
            nodes, s_edges=[], component_summaries=summaries,
            pad_nodes=set(), extra_edges=extra_edges,
        )
        assert bfs == {'X', 'Y'}
        assert summ == bfs

    def test_completely_unreferenced_node_is_island(self):
        """A node in interface_node_to_idx with zero summaries and zero
        extra_edges is still judged (singleton, unreached -> island)."""
        from pgmath.schur import detect_interface_islands_from_summaries

        idx = {'Z': 0}
        islands = detect_interface_islands_from_summaries(
            component_summaries=[], interface_node_to_idx=idx,
            pad_nodes=set(), extra_edges=None,
        )
        assert islands == {'Z'}


# ──────────────────────────────────────────────────────────────────────
# 3. Tile-resident-pad rescue divergence (item iv, core mechanism)
# ──────────────────────────────────────────────────────────────────────

class TestTileResidentPadRescue:
    """The one deliberate divergence from the BFS: has_pad rescues a
    component the value-aware S_global BFS cannot see any pad adjacency
    for (the pad's row/col was sliced into rhs_dirichlet during Schur
    elimination, so it never appears as an S_global off-diagonal nonzero)."""

    def test_bfs_wrongly_islands_tile_resident_pad_component(self):
        from pgmath.schur import find_interface_islands

        # T is the only interface node in this component; the tile-resident
        # pad itself is NOT part of interface_node_to_idx (Dirichlet nodes
        # are excluded from the unknown/interface set by construction, just
        # like model.py's real pipeline) -- so S_global has zero information
        # about T's pad reachability at all.
        S = _make_laplacian(['T'], [])
        islands = find_interface_islands(S, ['T'], {'T': 0}, pad_nodes=set(), extra_edges=None)
        assert islands == {'T'}, "BFS should (wrongly) island a tile-resident-pad component"

    def test_summary_rescues_tile_resident_pad_component(self, caplog):
        from pgmath.schur import detect_interface_islands_from_summaries

        summaries = [{'candidates': frozenset({'T'}), 'n_nodes': 2, 'has_pad': True}]
        with caplog.at_level(logging.WARNING, logger='pgmath.schur'):
            islands = detect_interface_islands_from_summaries(
                summaries, {'T': 0}, pad_nodes=set(), extra_edges=None,
            )
        assert islands == set(), "has_pad must rescue T from being penalized"
        assert any('rescued' in rec.message for rec in caplog.records), (
            "Expected a WARNING documenting the rescue"
        )

    def test_no_rescue_warning_when_liveness_comes_from_package_edge(self, caplog):
        """A component live via an ordinary package pad-edge (not has_pad)
        must NOT trigger the rescue warning -- it's not a divergence."""
        from pgmath.schur import detect_interface_islands_from_summaries

        summaries = [{'candidates': frozenset({'U'}), 'n_nodes': 1, 'has_pad': False}]
        extra_edges = [('U', 'PAD9', 1.0)]
        with caplog.at_level(logging.WARNING, logger='pgmath.schur'):
            islands = detect_interface_islands_from_summaries(
                summaries, {'U': 0}, pad_nodes={'PAD9'}, extra_edges=extra_edges,
            )
        assert islands == set()
        assert not any('rescued' in rec.message for rec in caplog.records)

    def test_rescue_and_package_liveness_together_no_warning(self, caplog):
        """A component reached BOTH via has_pad AND a package edge is not a
        'the BFS would have missed it' case -- no rescue warning."""
        from pgmath.schur import detect_interface_islands_from_summaries

        summaries = [{'candidates': frozenset({'V'}), 'n_nodes': 1, 'has_pad': True}]
        extra_edges = [('V', 'PAD9', 1.0)]
        with caplog.at_level(logging.WARNING, logger='pgmath.schur'):
            islands = detect_interface_islands_from_summaries(
                summaries, {'V': 0}, pad_nodes={'PAD9'}, extra_edges=extra_edges,
            )
        assert islands == set()
        assert not any('rescued' in rec.message for rec in caplog.records), (
            "Expected NO rescue warning when liveness comes from BOTH "
            "has_pad AND a package edge (not a 'BFS would have missed it' case)"
        )


# ──────────────────────────────────────────────────────────────────────
# 4. Sub-tile threshold-1-at-parse fixture (item iii)
# ──────────────────────────────────────────────────────────────────────

class TestSubTileThresholdOneAtParse:
    """A component connected only through a cut-interface node must survive
    the SPLIT-PATH parse-time pre-clean (_apply_tile_splits' sub-tile pass),
    not just the worker-side legacy removal this used to rely on.

    ``split_tile`` itself (geometric bisection) is monkeypatched to return a
    deterministic, hand-crafted 2-way split -- exercising the REAL bisection
    heuristics is already covered by ``test_retile.py``; this test isolates
    ``_apply_tile_splits``'s NEW sub-tile threshold=1 pre-clean wiring
    (port candidates = ``sub.boundary_nodes | die_attachment_nodes``; the
    parent boundary is always a subset of each sub-tile's boundary, so no
    separate cut-node delta is needed) deterministically.
    """

    def _write_metadata_and_tile(self, tmp_path: Path, tile_data):
        import pickle as _pickle
        from distributed.parser import PackageData, PowerGridMetaData, TileConfig

        tile_str = '_'.join(str(c) for c in tile_data.tile_id)
        with open(tmp_path / f'tile_{tile_str}.pkl', 'wb') as f:
            _pickle.dump(tile_data, f, protocol=_pickle.HIGHEST_PROTOCOL)

        pkg_data = PackageData(
            vsrc_dict={}, package_edges=[], pad_nodes=set(),
            tap_nodes=set(), die_attachment_nodes=set(),
            vdd=1.0, net_name='VDD', package_cap_edges=[],
        )
        metadata = PowerGridMetaData(
            tile_grid=(1, 1), parameters={},
            tile_configs=[TileConfig(
                tile_id=tile_data.tile_id, ckt_path='', nd_path=None,
                instance_path=None, net_filter=None,
            )],
            package_data=pkg_data, net_name='VDD', vdd=1.0,
        )
        return metadata

    def test_component_reachable_only_via_cut_node_survives_split(self, tmp_path, monkeypatch):
        """RIGHT sub-tile has a satellite component (2 nodes) whose sole
        interface candidate is the NEW cut node ``cutnode`` (1 < threshold=5,
        so a threshold=5 pass would wrongly drop it) alongside an unrelated
        larger (5-node) component that is always kept as the tile's largest.
        The sub-tile pass must use threshold=1 and keep the satellite.
        """
        from distributed.parser import DistributedNetlistParser
        import distributed.retile as retile_mod
        from distributed.tile_parsing import TileData

        parent_td = TileData(
            tile_id=(0, 0),
            resistive_edges=[('mainL0', 'shared', 5.0)],
            all_nodes={'mainL0', 'shared'},
            boundary_nodes={'shared'},
            current_injections={'mainL0': 0.1},
            capacitive_edges=[],
            pre_cleaned_full=True,  # already went through the universal step-1 pass
        )
        metadata = self._write_metadata_and_tile(tmp_path, parent_td)

        # LEFT sub-tile: main body + the cut edge (per _build_halves convention,
        # cut edges live in the LEFT sub-tile).
        left_td = TileData(
            tile_id=(0, 0, 0),
            resistive_edges=[('mainL0', 'shared', 5.0), ('mainL0', 'cutnode', 1.0)],
            all_nodes={'mainL0', 'shared', 'cutnode'},
            boundary_nodes={'shared', 'cutnode'},
            current_injections={'mainL0': 0.1},
            capacitive_edges=[],
            pre_cleaned=True,
        )
        # RIGHT sub-tile: an unrelated larger (always-largest) component
        # mainR0..mainR4, PLUS the satellite {satnode, cutnode} whose only
        # interface candidate is cutnode (n_interface=1).
        mainR_nodes = [f'mainR{i}' for i in range(5)]
        right_edges = [(mainR_nodes[i], mainR_nodes[i + 1], 1.0) for i in range(4)]
        right_edges.append((mainR_nodes[0], '0', 0.5))
        right_edges.append(('satnode', 'cutnode', 2.0))
        right_td = TileData(
            tile_id=(0, 0, 1),
            resistive_edges=right_edges,
            all_nodes=set(mainR_nodes) | {'satnode', 'cutnode'},
            boundary_nodes={'cutnode'},
            current_injections={'satnode': 0.02, **{n: 0.01 for n in mainR_nodes}},
            capacitive_edges=[],
            pre_cleaned=True,
        )

        monkeypatch.setattr(retile_mod, 'split_tile', lambda td, max_interior: [left_td, right_td])

        # Negative control: at threshold=5 (the whole-tile default), the
        # satellite {satnode, cutnode} component WOULD be wrongly dropped --
        # proves the fixture actually exercises the threshold distinction.
        import copy
        control_right = copy.deepcopy(right_td)
        n_removed_ctrl, _ = retile_mod._pre_clean_tile_data(
            control_right, port_nodes={'cutnode'}, min_interface_keep=5,
        )
        assert n_removed_ctrl == 1
        assert 'satnode' not in control_right.all_nodes, (
            "Fixture setup error: threshold=5 should drop the satellite component"
        )

        worker_results = [{
            'tile_id': (0, 0),
            'boundary_nodes': {'shared'},
            'die_attachment_net_map': {},
            'n_nodes': len(parent_td.all_nodes),
            'n_edges': len(parent_td.resistive_edges),
            'n_currents': len(parent_td.current_injections),
            'n_cap_edges': 0,
            'islands_removed_at_parse': 0,
            'component_summaries': [{
                'candidates': frozenset({'shared'}),
                'n_nodes': len(parent_td.all_nodes), 'has_pad': False,
            }],
        }]

        parser = DistributedNetlistParser.__new__(DistributedNetlistParser)
        new_results, new_metadata, _parent_diag = parser._apply_tile_splits(
            # max_interior=0 so the 1-interior-node parent (mainL0) always
            # exceeds threshold and the (mocked) split branch is taken.
            worker_results, metadata, tmp_path, max_interior=0, pickle_mod=pickle,
            shared_bnd=set(),
        )

        assert len(new_results) == 2, "Expected the (mocked) tile to split into 2 sub-tiles"

        right_result = next(r for r in new_results if r['tile_id'] == (0, 0, 1))
        with open(tmp_path / 'tile_0_0_1.pkl', 'rb') as f:
            right_sub_td = pickle.load(f)

        assert right_sub_td.pre_cleaned_full is True
        assert 'satnode' in right_sub_td.all_nodes, (
            "Satellite component (reachable only via the cut-interface node "
            "'cutnode') was wrongly dropped by the sub-tile pre-clean pass "
            "(threshold=1 was not applied)"
        )
        assert right_result['islands_removed_at_parse'] == 0

        candidate_sets = {c['candidates'] for c in right_result['component_summaries']}
        assert frozenset({'cutnode'}) in candidate_sets, (
            f"Expected the satellite's summary to list 'cutnode' as its sole "
            f"interface candidate; got {candidate_sets}"
        )

    def test_cross_declared_node_survives_sub_tile_pass(self, tmp_path, monkeypatch):
        """Finding F5: a node 'crossnode' present UNMARKED in the sub-tile's
        all_nodes (no '*' prefix here -- e.g. an ordinary resistor endpoint)
        but '*'-declared as shared boundary by 2+ OTHER tiles entirely (so
        crossnode ∈ the global shared_bnd scan set) must still count as an
        interface candidate at the sub-tile pass -- exactly as it would at
        the whole-tile pass (which uses the same global shared_bnd set).

        Without the `sub.all_nodes & shared_bnd` term, crossnode is invisible
        to the sub-tile port-candidate computation (it's in neither
        sub.boundary_nodes nor die_attachment_nodes), so a satellite
        component reachable ONLY via crossnode is wrongly removed even
        though it was legitimately kept at the (correct, global-shared_bnd-
        aware) parent-level whole-tile pass.
        """
        from distributed.parser import DistributedNetlistParser
        import distributed.retile as retile_mod
        from distributed.tile_parsing import TileData

        parent_td = TileData(
            tile_id=(0, 0),
            resistive_edges=[('mainL0', 'shared', 5.0)],
            all_nodes={'mainL0', 'shared'},
            boundary_nodes={'shared'},
            current_injections={'mainL0': 0.1},
            capacitive_edges=[],
            pre_cleaned_full=True,
        )
        metadata = self._write_metadata_and_tile(tmp_path, parent_td)

        left_td = TileData(
            tile_id=(0, 0, 0),
            resistive_edges=[('mainL0', 'shared', 5.0), ('mainL0', 'cutnode', 1.0)],
            all_nodes={'mainL0', 'shared', 'cutnode'},
            boundary_nodes={'shared', 'cutnode'},
            current_injections={'mainL0': 0.1},
            capacitive_edges=[],
            pre_cleaned=True,
        )
        # RIGHT sub-tile: main body (always-largest, with its OWN genuine
        # cut-boundary node 'cutnode2' so the pre-clean pass runs a real
        # threshold check rather than short-circuiting on an empty port
        # set) PLUS a satellite {satnode, crossnode} whose only interface
        # candidate is 'crossnode' -- NOT marked '*' here (not in
        # boundary_nodes at all), globally shared only because 2 OTHER
        # tiles (not part of this fixture) declared it.
        mainR_nodes = [f'mainR{i}' for i in range(5)]
        right_edges = [(mainR_nodes[i], mainR_nodes[i + 1], 1.0) for i in range(4)]
        right_edges.append((mainR_nodes[0], '0', 0.5))
        right_edges.append((mainR_nodes[0], 'cutnode2', 1.0))
        right_edges.append(('satnode', 'crossnode', 2.0))
        right_td = TileData(
            tile_id=(0, 0, 1),
            resistive_edges=right_edges,
            all_nodes=set(mainR_nodes) | {'satnode', 'crossnode', 'cutnode2'},
            boundary_nodes={'cutnode2'},  # crossnode is deliberately NOT '*'-marked here
            current_injections={'satnode': 0.02, **{n: 0.01 for n in mainR_nodes}},
            capacitive_edges=[],
            pre_cleaned=True,
        )

        monkeypatch.setattr(retile_mod, 'split_tile', lambda td, max_interior: [left_td, right_td])

        # Negative control: without the global shared_bnd term, the
        # satellite is wrongly dropped -- proves the fixture actually
        # exercises the F5 gap.
        import copy
        control_right = copy.deepcopy(right_td)
        n_removed_ctrl, _ = retile_mod._pre_clean_tile_data(
            control_right, port_nodes=control_right.boundary_nodes, min_interface_keep=1,
        )
        assert n_removed_ctrl == 1
        assert 'satnode' not in control_right.all_nodes, (
            "Fixture setup error: without crossnode as a candidate, the "
            "satellite component should be dropped"
        )

        worker_results = [{
            'tile_id': (0, 0),
            'boundary_nodes': {'shared'},
            'die_attachment_net_map': {},
            'n_nodes': len(parent_td.all_nodes),
            'n_edges': len(parent_td.resistive_edges),
            'n_currents': len(parent_td.current_injections),
            'n_cap_edges': 0,
            'islands_removed_at_parse': 0,
            'component_summaries': [{
                'candidates': frozenset({'shared'}),
                'n_nodes': len(parent_td.all_nodes), 'has_pad': False,
                'tile_id': (0, 0),
            }],
        }]

        # 'crossnode' is globally shared (declared '*' by 2+ OTHER tiles,
        # not modeled in this fixture) -- the step-0 scan set passed through.
        shared_bnd = {'shared', 'crossnode'}

        parser = DistributedNetlistParser.__new__(DistributedNetlistParser)
        new_results, new_metadata, _parent_diag = parser._apply_tile_splits(
            worker_results, metadata, tmp_path, max_interior=0, pickle_mod=pickle,
            shared_bnd=shared_bnd,
        )

        assert len(new_results) == 2

        right_result = next(r for r in new_results if r['tile_id'] == (0, 0, 1))
        with open(tmp_path / 'tile_0_0_1.pkl', 'rb') as f:
            right_sub_td = pickle.load(f)

        assert 'satnode' in right_sub_td.all_nodes, (
            "Satellite component (reachable only via the globally-shared but "
            "locally-unmarked 'crossnode') was wrongly dropped by the "
            "sub-tile pre-clean pass -- the F5 bug"
        )
        assert right_result['islands_removed_at_parse'] == 0

        candidate_sets = {c['candidates'] for c in right_result['component_summaries']}
        assert frozenset({'crossnode'}) in candidate_sets, (
            f"Expected the satellite's summary to list 'crossnode' as its "
            f"sole interface candidate; got {candidate_sets}"
        )


# ──────────────────────────────────────────────────────────────────────
# 5. Trust assertion / fallback matrix (items v, vi)
# ──────────────────────────────────────────────────────────────────────

class TestResolveIslandDetection:
    """model._resolve_island_detection: the all-new-or-all-legacy decision."""

    def _bundle(self, **overrides):
        from distributed.model import ParsedTileBundle
        from distributed.parser import CONNECTIVITY_SUMMARY_VERSION, PackageData, PowerGridMetaData, TileConfig

        pkg = PackageData(
            vsrc_dict={}, package_edges=[], pad_nodes=set(),
            tap_nodes=set(), die_attachment_nodes=set(),
            vdd=1.0, net_name='VDD',
        )
        metadata = PowerGridMetaData(
            tile_grid=(1, 1), parameters={}, tile_configs=[], package_data=pkg,
            net_name='VDD', vdd=1.0,
        )
        defaults = dict(
            metadata=metadata,
            shared_boundary_nodes={'shared'},
            pkl_dir='/tmp/irrelevant',
            connectivity_summary_version=CONNECTIVITY_SUMMARY_VERSION,
            parser_interface_set={'shared'},
            component_summaries=[{'candidates': frozenset({'shared'}), 'n_nodes': 1, 'has_pad': False}],
        )
        defaults.update(overrides)
        return ParsedTileBundle(**defaults)

    def test_matching_interface_nodes_trusts_summaries(self):
        from distributed.model import _resolve_island_detection

        bundle = self._bundle()
        trust, mode = _resolve_island_detection(bundle, {'shared'}, 'auto')
        assert trust is True
        assert mode == 'summaries'

    def test_explicit_schur_bfs_forces_legacy(self):
        from distributed.model import _resolve_island_detection

        bundle = self._bundle()
        trust, mode = _resolve_island_detection(bundle, {'shared'}, 'schur_bfs')
        assert trust is False
        assert mode == 'schur_bfs'

    def test_missing_summaries_falls_back_with_info(self, caplog):
        from distributed.model import _resolve_island_detection

        bundle = self._bundle(component_summaries=None, parser_interface_set=None,
                               connectivity_summary_version=None)
        with caplog.at_level(logging.INFO, logger='distributed.model'):
            trust, mode = _resolve_island_detection(bundle, {'shared'}, 'auto')
        assert trust is False
        assert mode == 'schur_bfs'
        assert any(r.levelno == logging.INFO for r in caplog.records)
        assert not any(r.levelno >= logging.WARNING for r in caplog.records)

    def test_version_mismatch_falls_back_with_info(self, caplog):
        from distributed.model import _resolve_island_detection
        from distributed.parser import CONNECTIVITY_SUMMARY_VERSION

        bundle = self._bundle(connectivity_summary_version=CONNECTIVITY_SUMMARY_VERSION + 999)
        with caplog.at_level(logging.INFO, logger='distributed.model'):
            trust, mode = _resolve_island_detection(bundle, {'shared'}, 'auto')
        assert trust is False
        assert mode == 'schur_bfs'
        assert not any(r.levelno >= logging.WARNING for r in caplog.records)

    def test_tampered_parser_interface_set_warns_and_falls_back(self, caplog):
        """Item (v): tamper parser_interface_set -> WARNING + full legacy."""
        from distributed.model import _resolve_island_detection

        bundle = self._bundle(parser_interface_set={'shared', 'someone_edited_this'})
        with caplog.at_level(logging.WARNING, logger='distributed.model'):
            trust, mode = _resolve_island_detection(bundle, {'shared'}, 'auto')
        assert trust is False
        assert mode == 'schur_bfs'
        assert any(r.levelno == logging.WARNING for r in caplog.records)

    def test_summaries_setting_same_as_auto_when_available(self):
        from distributed.model import _resolve_island_detection

        bundle = self._bundle()
        trust, mode = _resolve_island_detection(bundle, {'shared'}, 'summaries')
        assert trust is True
        assert mode == 'summaries'

    def test_unknown_setting_raises(self):
        from distributed.model import _resolve_island_detection

        bundle = self._bundle()
        with pytest.raises(ValueError):
            _resolve_island_detection(bundle, {'shared'}, 'not_a_real_mode')


# ──────────────────────────────────────────────────────────────────────
# 6. Determinism (finding F4)
# ──────────────────────────────────────────────────────────────────────

class TestDeterministicComponentDecomposition:
    """_decompose_and_remove_floating must not depend on set-iteration
    order / PYTHONHASHSEED: the outer BFS-seed loop iterates
    sorted(all_nodes), and the largest-component tie-break uses
    key=(len(c), min(c))."""

    def test_largest_tie_break_deterministic_across_insertion_orders(self):
        """Two components tied at size 2; `max(..., key=(len(c), min(c)))`
        picks the component with the LEXICOGRAPHICALLY LARGEST min-node as
        'largest' (kept unconditionally) -- so {'bbb','yyy'} (min='bbb')
        always wins the tie over {'aaa','zzz'} (min='aaa', 'aaa' < 'bbb').
        The loser ({'aaa','zzz'}, 0 interface candidates) must ALWAYS be
        removed, regardless of the order `all_nodes` happens to iterate in."""
        import random
        from distributed.tile_parsing import TileData, _decompose_and_remove_floating

        base_nodes = ['aaa', 'zzz', 'bbb', 'yyy']
        edges = [('aaa', 'zzz', 1.0), ('bbb', 'yyy', 1.0)]

        results = []
        for _ in range(10):
            order = list(base_nodes)
            random.shuffle(order)
            td = TileData(
                tile_id=(0, 0),
                resistive_edges=list(edges),
                all_nodes=set(order),
                boundary_nodes=set(),
                current_injections={},
                capacitive_edges=[],
            )
            _components, _largest, _kept_iface, removed, n_removed = _decompose_and_remove_floating(
                td, port_nodes_local=set(), min_interface_keep=5,
            )
            results.append((frozenset(removed), n_removed))

        assert len(set(results)) == 1, (
            f"Non-deterministic removal across insertion orders: {results}"
        )
        removed_final, n_removed_final = results[0]
        assert n_removed_final == 1
        assert removed_final == frozenset({'aaa', 'zzz'}), (
            "Expected {'aaa','zzz'} (min-node 'aaa' < 'bbb') to always lose "
            "the size tie and be removed"
        )

    def test_component_enumeration_order_independent_of_all_nodes_order(self):
        """The SET of removed nodes for a larger, more realistic topology
        must be identical regardless of the insertion order of `all_nodes`."""
        import random
        from distributed.tile_parsing import TileData, _decompose_and_remove_floating

        # main: 6-node chain to ground (always the largest, unconditionally
        # kept).  Three separate floating stub pairs, none reaching a port.
        main_nodes = [f'm{i}' for i in range(6)]
        edges = [(main_nodes[i], main_nodes[i + 1], 1.0) for i in range(5)]
        edges.append((main_nodes[0], '0', 0.5))
        stub_pairs = [('p0', 'q0'), ('p1', 'q1'), ('p2', 'q2')]
        for u, v in stub_pairs:
            edges.append((u, v, 1.0))
        all_nodes_base = set(main_nodes) | {n for pair in stub_pairs for n in pair}

        results = []
        for _ in range(10):
            order = list(all_nodes_base)
            random.shuffle(order)
            td = TileData(
                tile_id=(0, 0),
                resistive_edges=list(edges),
                all_nodes=set(order),
                boundary_nodes=set(),
                current_injections={},
                capacitive_edges=[],
            )
            _components, _largest, _kept_iface, removed, n_removed = _decompose_and_remove_floating(
                td, port_nodes_local=set(), min_interface_keep=5,
            )
            results.append((frozenset(removed), n_removed))

        assert len(set(results)) == 1, (
            f"Non-deterministic removal across insertion orders: {results}"
        )
        removed_final, n_removed_final = results[0]
        assert n_removed_final == 3
        assert removed_final == {n for pair in stub_pairs for n in pair}


class TestDeterministicSummaryAggregation:
    """parser._sort_component_summaries: final aggregation order must not
    depend on the order summaries arrive in from per-tile worker results."""

    def test_sort_is_stable_and_order_independent(self):
        from distributed.parser import _sort_component_summaries
        import random

        summaries = [
            {'candidates': frozenset({'zz'}), 'n_nodes': 1, 'has_pad': False, 'tile_id': (0, 0)},
            {'candidates': frozenset({'aa'}), 'n_nodes': 1, 'has_pad': False, 'tile_id': (0, 0)},
            {'candidates': frozenset({'mm'}), 'n_nodes': 1, 'has_pad': False, 'tile_id': (1, 0)},
            {'candidates': frozenset({'bb', 'cc'}), 'n_nodes': 2, 'has_pad': False, 'tile_id': (0, 1)},
        ]

        results = []
        for _ in range(10):
            shuffled = list(summaries)
            random.shuffle(shuffled)
            sorted_out = _sort_component_summaries(shuffled)
            key = tuple(
                (s['tile_id'], tuple(sorted(s['candidates'])))
                for s in sorted_out
            )
            results.append(key)

        assert len(set(results)) == 1, (
            f"Non-deterministic aggregation order across shuffles: {results}"
        )
        # Expected order: (0,0)/'aa', (0,0)/'zz', (0,1)/'bb', (1,0)/'mm'
        expected_tile_order = [(0, 0), (0, 0), (0, 1), (1, 0)]
        assert [k[0] for k in results[0]] == expected_tile_order


# ──────────────────────────────────────────────────────────────────────
# 7. F10: structural-completeness runtime guarantee
# ──────────────────────────────────────────────────────────────────────

class TestStructuralCompletenessCheck:
    """model._verify_summary_structural_completeness: cheap runtime guard
    that (worker-reported ports ∩ finalized interface) ⊆ that tile's
    summarized interface-candidates."""

    def test_complete_summaries_pass(self):
        from distributed.model import _verify_summary_structural_completeness

        summaries = [
            {'candidates': frozenset({'a', 'b'}), 'n_nodes': 2, 'has_pad': False, 'tile_id': (0, 0)},
        ]
        interface_nodes = {'a', 'b'}
        tile_boundary_nodes = {(0, 0): ['a', 'b']}
        assert _verify_summary_structural_completeness(
            summaries, interface_nodes, tile_boundary_nodes,
        ) is True

    def test_tampered_summary_missing_candidate_fails(self, caplog):
        """Drop 'b' from the tile's only summary -- the worker still reports
        it as a port -- must fail and log a WARNING."""
        from distributed.model import _verify_summary_structural_completeness

        summaries = [
            {'candidates': frozenset({'a'}), 'n_nodes': 2, 'has_pad': False, 'tile_id': (0, 0)},
        ]
        interface_nodes = {'a', 'b'}
        tile_boundary_nodes = {(0, 0): ['a', 'b']}
        with caplog.at_level(logging.WARNING, logger='distributed.model'):
            result = _verify_summary_structural_completeness(
                summaries, interface_nodes, tile_boundary_nodes,
            )
        assert result is False
        assert any(
            'structural-completeness check FAILED' in rec.message
            for rec in caplog.records
        )

    def test_none_summaries_short_circuits_true(self):
        from distributed.model import _verify_summary_structural_completeness

        assert _verify_summary_structural_completeness(
            None, {'a'}, {(0, 0): ['a']},
        ) is True

    def test_model_creation_falls_back_to_legacy_on_tampered_summary(self, tmp_path):
        """End-to-end: create_distributed_model() with a deliberately
        tampered (candidate-dropping) summary must transparently fall back
        to island_detection_mode='schur_bfs' -- not crash, not silently
        trust a structurally-incomplete summary."""
        import pickle
        from distributed.model import ParsedTileBundle, create_distributed_model
        from distributed.parser import PackageData, PowerGridMetaData, TileConfig
        from distributed.tile_parsing import TileData

        td = TileData(
            tile_id=(0, 0),
            resistive_edges=[('a', 'b', 1.0), ('a', '0', 0.5)],
            all_nodes={'a', 'b'},
            boundary_nodes={'b'},
            current_injections={'a': 0.1},
            capacitive_edges=[],
            pre_cleaned_full=True,
        )
        tile_dir = tmp_path
        with open(tile_dir / 'tile_0_0.pkl', 'wb') as f:
            pickle.dump(td, f, protocol=pickle.HIGHEST_PROTOCOL)

        pkg_data = PackageData(
            vsrc_dict={'V1': {'node_pos': 'b', 'node_neg': '0', 'net': 'VDD', 'value': 1.0}},
            package_edges=[], pad_nodes={'b'}, tap_nodes=set(),
            die_attachment_nodes=set(), vdd=1.0, net_name='VDD',
        )
        metadata = PowerGridMetaData(
            tile_grid=(1, 1), parameters={},
            tile_configs=[TileConfig(
                tile_id=(0, 0), ckt_path='', nd_path=None,
                instance_path=None, net_filter=None,
            )],
            package_data=pkg_data, net_name='VDD', vdd=1.0,
        )
        with open(tile_dir / 'metadata.pkl', 'wb') as f:
            pickle.dump(
                {'metadata': metadata, 'boundary_nodes': {'b'}},
                f, protocol=pickle.HIGHEST_PROTOCOL,
            )

        # Tampered: 'b' is dropped from the (only) summary's candidates, even
        # though the worker will report it as a port ('b' is in
        # shared_boundary_nodes / interface_nodes).
        component_summaries = [
            {'candidates': frozenset(), 'n_nodes': 2, 'has_pad': True, 'tile_id': (0, 0)},
        ]
        bundle = ParsedTileBundle(
            metadata=metadata,
            shared_boundary_nodes={'b'},
            pkl_dir=str(tile_dir),
            connectivity_summary_version=_stage1e_connectivity_summary_version(),
            parser_interface_set={'b'},
            component_summaries=component_summaries,
        )

        model = create_distributed_model(bundle, backend='local', island_detection='auto')
        try:
            assert model.island_detection_mode == 'schur_bfs', (
                "Expected the structural-completeness check to force the "
                "legacy fallback on a tampered summary"
            )
        finally:
            model.shutdown()


def _stage1e_connectivity_summary_version():
    from distributed.parser import CONNECTIVITY_SUMMARY_VERSION
    return CONNECTIVITY_SUMMARY_VERSION


# ──────────────────────────────────────────────────────────────────────
# 8. F11: shared candidate sets shipped via backend.put() for Ray
# ──────────────────────────────────────────────────────────────────────

class TestBackendPutForRayShipping:
    """backend.put(): LocalBackend is a passthrough (no object store);
    RayBackend actually stores + returns a dereferenceable handle."""

    def test_local_backend_put_is_passthrough(self):
        from distributed.backend import LocalBackend

        be = LocalBackend()
        be.initialize()
        obj = {'a', 'b', 'c'}
        assert be.put(obj) is obj

    def test_ray_backend_put_roundtrips(self):
        try:
            import ray
        except ImportError:
            pytest.skip("ray not installed")
        from distributed.backend import RayBackend

        be = RayBackend()
        be.initialize()
        obj = {'x', 'y', 'z'}
        ref = be.put(obj)
        assert ray.get(ref) == obj

    def test_parse_and_dump_ray_matches_local(self, tmp_path):
        """End-to-end: parse_and_dump(backend='ray') -- which now ships
        shared_bnd/pad_candidates via be.put() (finding F11) -- must produce
        a bundle identical to backend='local' (proves the ray.put wiring
        changes transport only, not parse semantics/results)."""
        try:
            import ray  # noqa: F401
        except ImportError:
            pytest.skip("ray not installed")
        from distributed.parser import DistributedNetlistParser

        netlist_dir = tmp_path / 'netlist'
        netlist_dir.mkdir()
        (netlist_dir / 'tile_0_0.ckt').write_text(
            "R_m0 m0 m1 100\n"
            "R_mpad m0 1000_100_M1 100\n"
            "I_m0 m0 0 1e-3\n"
        )
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

        parser_local = DistributedNetlistParser(str(netlist_dir), net_filter=None)
        _, bundle_local = parser_local.parse_and_dump(
            str(tmp_path / 'pkl_local'), backend='local',
        )
        parser_ray = DistributedNetlistParser(str(netlist_dir), net_filter=None)
        _, bundle_ray = parser_ray.parse_and_dump(
            str(tmp_path / 'pkl_ray'), backend='ray',
        )

        assert bundle_ray.shared_boundary_nodes == bundle_local.shared_boundary_nodes
        assert bundle_ray.parser_interface_set == bundle_local.parser_interface_set
        assert bundle_ray.connectivity_summary_version == bundle_local.connectivity_summary_version
        local_candidates = [
            (s.get('tile_id'), s['candidates']) for s in bundle_local.component_summaries
        ]
        ray_candidates = [
            (s.get('tile_id'), s['candidates']) for s in bundle_ray.component_summaries
        ]
        assert ray_candidates == local_candidates


# ──────────────────────────────────────────────────────────────────────
# 9. F15: _detect_islands_dispatch model=None handling
# ──────────────────────────────────────────────────────────────────────

class TestDetectIslandsDispatchNoneModel:
    """model=None must raise a clear, immediate error (not a confusing
    AttributeError deep inside whichever branch runs next)."""

    def test_none_model_raises_value_error(self):
        import numpy as np
        import scipy.sparse as sp
        from distributed.result_factorization import _detect_islands_dispatch

        S = sp.csr_matrix((2, 2))
        rhs = np.zeros(2)
        with pytest.raises(ValueError, match='model'):
            _detect_islands_dispatch(
                None, S, rhs, ['a', 'b'], {'a': 0, 'b': 1}, extra_edges=None,
            )


# ──────────────────────────────────────────────────────────────────────
# 10. R6: _detect_islands_dispatch forbidden mixed state
# ──────────────────────────────────────────────────────────────────────

class _FakeModelMixedState:
    """Minimal stand-in for a hand-constructed/corrupted
    DistributedPowerGridModel exercising the forbidden mixed state."""

    def __init__(self, island_detection_mode, component_summaries):
        self.island_detection_mode = island_detection_mode
        self.component_summaries = component_summaries
        self.pad_nodes = set()
        self.vdd = 1.0


class TestDetectIslandsDispatchMixedState:
    """island_detection_mode == 'summaries' with component_summaries is None
    is a FORBIDDEN mixed state -- model._resolve_island_detection is the
    only legitimate place these two fields are set (always together), so
    this combination can only arise from a corrupted/hand-constructed model.
    Silently falling back to the legacy Schur-BFS would lose the
    tile-resident-pad rescue and produce silently wrong voltages."""

    def test_summaries_mode_with_none_summaries_raises(self):
        import numpy as np
        import scipy.sparse as sp
        from distributed.result_factorization import _detect_islands_dispatch

        model = _FakeModelMixedState('summaries', None)
        S = sp.csr_matrix((2, 2))
        rhs = np.zeros(2)
        with pytest.raises(ValueError, match='mixed state'):
            _detect_islands_dispatch(
                model, S, rhs, ['a', 'b'], {'a': 0, 'b': 1}, extra_edges=None,
            )

    def test_summaries_mode_with_summaries_does_not_raise(self):
        """Sanity: the legitimate 'summaries' state (both fields set
        together) must NOT trip the mixed-state guard."""
        import numpy as np
        import scipy.sparse as sp
        from distributed.result_factorization import _detect_islands_dispatch

        model = _FakeModelMixedState(
            'summaries',
            [{'candidates': frozenset({'a'}), 'n_nodes': 1, 'has_pad': True}],
        )
        S = sp.csr_matrix((2, 2))
        rhs = np.zeros(2)
        # Should not raise -- exercises the union-find path instead.
        _detect_islands_dispatch(
            model, S, rhs, ['a', 'b'], {'a': 0, 'b': 1}, extra_edges=None,
        )

    def test_schur_bfs_mode_with_none_summaries_does_not_raise(self):
        """Sanity: 'schur_bfs' mode with component_summaries=None is the
        LEGITIMATE degrade point (not the forbidden mixed state)."""
        import numpy as np
        import scipy.sparse as sp
        from distributed.result_factorization import _detect_islands_dispatch

        model = _FakeModelMixedState('schur_bfs', None)
        S = sp.csr_matrix((2, 2))
        rhs = np.zeros(2)
        # Should not raise -- exercises the legacy Schur-BFS path instead.
        _detect_islands_dispatch(
            model, S, rhs, ['a', 'b'], {'a': 0, 'b': 1}, extra_edges=None,
        )


# ──────────────────────────────────────────────────────────────────────
# 11. R5: PackedTileWorkerActor.call_worker defensive ObjectRef deref
# ──────────────────────────────────────────────────────────────────────

class TestPackedWorkerCallDefensiveDeref:
    """VirtualWorkerHandle dispatch routes packed-worker calls via
    ``call_worker.remote(local_idx, method, args)`` -- Ray's automatic
    ObjectRef resolution only applies to TOP-LEVEL positional arguments of
    a ``.remote()`` call, not to refs nested inside the *args* container.
    ``PackedTileWorkerActor.call_worker`` must defensively deref before
    dispatch so a ``ComputeBackend.put()`` handle works uniformly whether or
    not ``tiles_per_worker`` packing is active."""

    def test_local_backend_put_passthrough_needs_no_deref(self):
        """LocalBackend.put() is a no-op passthrough -- _deref_packed_args
        must be a safe no-op on a plain (non-ObjectRef) value too."""
        from distributed.backend import _deref_packed_args

        assert _deref_packed_args(None) is None
        assert _deref_packed_args((1, 2)) == (1, 2)
        assert _deref_packed_args([1, 2]) == [1, 2]
        assert _deref_packed_args({'a': 1}) == {'a': 1}
        assert _deref_packed_args('scalar') == 'scalar'

    def test_ray_packed_call_worker_derefs_put_handle(self):
        """_deref_packed_args resolves a REAL ray.ObjectRef nested inside an
        args tuple -- the exact shape VirtualWorkerHandle dispatch produces
        (call_worker.remote(local_idx, method, args), where args is itself a
        container).  Uses the same RayBackend().initialize() pattern as the
        other Ray tests in this module (idempotent -- reuses any already-
        running cluster; no local_mode, which is unsupported/unmaintained)."""
        try:
            import ray  # noqa: F401
        except ImportError:
            pytest.skip("ray not installed")
        from distributed.backend import RayBackend, _deref_packed_args

        be = RayBackend()
        be.initialize()
        handle = be.put({'shared', 'candidate', 'set'})

        resolved = _deref_packed_args((handle, 'other_arg'))
        assert resolved == ({'shared', 'candidate', 'set'}, 'other_arg'), (
            "Expected the nested ObjectRef to be resolved to its "
            "underlying value, not left as a raw ObjectRef"
        )
        # Also cover the dict-args ('**kwargs') and bare-scalar conventions.
        resolved_dict = _deref_packed_args({'settings': handle})
        assert resolved_dict == {'settings': {'shared', 'candidate', 'set'}}
        resolved_scalar = _deref_packed_args(handle)
        assert resolved_scalar == {'shared', 'candidate', 'set'}

    def test_ray_packed_call_all_end_to_end_derefs_put_handle(self):
        """End-to-end: a put() handle nested inside a packed call_all's args
        tuple reaches TileWorker.configure() as the real settings dict, not
        a raw ray.ObjectRef -- reproduces the exact failure scenario from
        finding R5 (a future call site shipping a shared value via put() to
        a packed/tiles_per_worker>1 Ray worker).  Without the fix,
        TileWorker.configure() would receive a raw ray.ObjectRef and
        `'pre_cleaned_full_trusted' in settings` would raise TypeError
        (ObjectRef does not support containment checks), surfaced by
        ray.get() as a RayTaskError."""
        try:
            import ray  # noqa: F401
        except ImportError:
            pytest.skip("ray not installed")
        from distributed.backend import PackedTileWorkerActor, RayBackend, VirtualWorkerHandle

        be = RayBackend()
        be.initialize()
        RemotePacked = be._ray.remote(PackedTileWorkerActor)
        actor = RemotePacked.remote(1)
        handle = VirtualWorkerHandle(actor, 0)

        settings_handle = be.put({'pre_cleaned_full_trusted': True})
        # Must not raise.
        be.call_all([handle], 'configure', [(settings_handle,)])

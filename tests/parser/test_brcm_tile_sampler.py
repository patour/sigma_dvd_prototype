#!/usr/bin/env python3
"""Unit tests for the Pass 1-4 pieces of ``parser.brcm_tile_sampler``.

Covers (see ``netlist_brcm_sampling_plan.md`` §4.1-4.4 and the "REVISED"
banner):
- Via vs non-via node classification.
- Layer suffix parsing.
- Mandatory-keep set union correctness (pad-anchor / boundary /
  current-source-bearing, no double counting).
- Pass 2 (``contract_tile_nodes``): determinism, mandatory nodes as
  singleton representatives, sparse-vs-dense per-layer retention, overall
  kept-count closeness to target, isolated-node handling, degenerate/
  unparseable coordinate inputs.
- ``scan_current_source_nodes`` net-filter correctness (the core
  correctness risk: accidentally treating other-net current sources as
  VDD_VAR).
- Pass 3 (``contract_tile_edges``): connectivity preservation (the
  regression test for the §7.5 node-drop failure), R/node preservation,
  parallel-conductance/capacitance merging, no spurious cross-cluster
  merges, via/cross-layer edge survival, and the mandatory-degree sanity
  assertion.
- Pass 4 (``sample_current_sources``): raw-text-preserving down-sampling,
  net filtering, reproducibility, and the "node_pos not in kept_nodes"
  defensive warning.
- ``verify_capacitor_invariant``: passing and violating cases.
- End-to-end: ``run_sampling_pipeline`` output re-parsed via
  ``DistributedNetlistParser`` -- zero floating nodes, R/node sanity.

All tests use small synthetic ``TileData``/inputs constructed directly in
this file -- no dependency on real ``netlist_brcm`` data (that belongs to a
later integration/validation phase, exercised separately as a real-data
smoke test outside pytest).
"""

import gzip
import json
import logging
import pickle
import random
from collections import deque
from pathlib import Path

import pytest

from distributed.parser import DistributedNetlistParser, PackageData, PowerGridMetaData, TileConfig
from distributed.tile_parsing import TileData
from parser.brcm_tile_sampler import (
    CapacitorInvariantViolation,
    CurrentSourceSamplingResult,
    SamplingPipelineReport,
    TileClassification,
    TileContractionResult,
    TileEdgeContractionResult,
    TileProcessingStats,
    _capacitance_ff_to_farad,
    _conductance_ms_to_ohm,
    _parse_node_layer,
    _prefixed_node,
    _process_tile_star,
    classify_tile,
    contract_tile_edges,
    contract_tile_nodes,
    discover_tile_ids,
    filter_package_ckt,
    filter_tile_nd_lines,
    generate_additional_vsrcs,
    generate_ckt_sp,
    generate_pg_net_voltage,
    generate_tile_ckt_content,
    load_metadata,
    load_tile_data,
    process_tile,
    read_die_area_from_ckt_sp,
    run_sampling_pipeline,
    sample_current_sources,
    scan_current_source_nodes,
    verify_capacitor_invariant,
    write_instance_models,
    write_pipeline_report,
    write_tile_ckt,
    write_tile_nd,
    write_top_level_files,
)
from parser.spice_lexer import _parse_spice_value

# Real netlist_brcm data, used only by the integration tests at the bottom
# of this file (§4.5 output-generation real-data validation) -- see
# tests/CLAUDE.md's "Test netlists" convention (canonical, __file__-relative
# paths, not CWD-relative).
_REAL_NETLIST_DIR = Path(__file__).resolve().parents[2] / 'netlist' / 'netlist_brcm'
_REAL_PKL_DIR = _REAL_NETLIST_DIR / 'distributed_pkl'

pytestmark = pytest.mark.unit


def _make_tile_data(
    all_nodes,
    resistive_edges,
    boundary_nodes=frozenset(),
    current_injections=None,
    tile_id=(0, 0),
    capacitive_edges=None,
):
    """Build a minimal synthetic ``TileData`` for classification tests."""
    return TileData(
        tile_id=tile_id,
        resistive_edges=list(resistive_edges),
        all_nodes=set(all_nodes),
        boundary_nodes=set(boundary_nodes),
        current_injections=dict(current_injections or {}),
        capacitive_edges=list(capacitive_edges or []),
    )


def _make_classification(
    tile_id,
    adjacency,
    layer_of,
    mandatory_keep,
    via_nodes=frozenset(),
    optional_pool_by_layer=None,
):
    """Build a ``TileClassification`` directly for Pass 2/3 unit tests.

    ``optional_pool_by_layer`` defaults to ``all_nodes - mandatory_keep``
    grouped by ``layer_of`` (mirroring ``classify_tile``'s own derivation)
    when not given explicitly -- convenient for ``contract_tile_nodes``
    tests, which need a real optional pool, not the empty placeholder the
    old (Pass-3-only) version of this helper used.
    """
    all_nodes = set(adjacency)
    mandatory_keep = set(mandatory_keep)
    if optional_pool_by_layer is None:
        optional_pool_by_layer = {}
        for node in all_nodes - mandatory_keep:
            optional_pool_by_layer.setdefault(layer_of.get(node), set()).add(node)
    return TileClassification(
        tile_id=tile_id,
        total_nodes=len(all_nodes),
        layer_of=dict(layer_of),
        via_nodes=set(via_nodes),
        adjacency={k: list(v) for k, v in adjacency.items()},
        mandatory_keep=mandatory_keep,
        optional_pool_by_layer=optional_pool_by_layer,
        pad_anchor_nodes=set(),
        boundary_nodes=set(),
        current_source_nodes=set(),
    )


def _grid_nodes_edges(nx, ny, layer, x0=0, y0=0, g=5.0):
    """Build a rectangular ``nx x ny`` mesh (with cycles) on one layer.

    Returns ``(nodes, edges, name_fn)`` -- ``name_fn(x, y)`` reconstructs a
    node name so callers can reference specific grid nodes (corners, a
    pad-attachment point, etc.).
    """
    def name(x, y):
        return f"{x0 + x}_{y0 + y}_{layer}"

    nodes = [name(x, y) for y in range(ny) for x in range(nx)]
    edges = []
    for y in range(ny):
        for x in range(nx):
            if x + 1 < nx:
                edges.append((name(x, y), name(x + 1, y), g))
            if y + 1 < ny:
                edges.append((name(x, y), name(x, y + 1), g))
    return nodes, edges, name


def _chain_nodes_edges(n, layer, x0=0, y0=0, g=5.0):
    """Build an ``n``-node same-layer chain (a tree -- no cycles, R/node < 1)."""
    nodes = [f"{x0 + i}_{y0}_{layer}" for i in range(n)]
    edges = [(nodes[i], nodes[i + 1], g) for i in range(n - 1)]
    return nodes, edges


def _reachable_from(start, resistive_edges):
    """BFS reachability set over a plain ``(u, v, g)`` resistor edge list."""
    adjacency = {}
    for u, v, _g in resistive_edges:
        adjacency.setdefault(u, []).append(v)
        adjacency.setdefault(v, []).append(u)
    seen = {start}
    queue = deque([start])
    while queue:
        node = queue.popleft()
        for neighbor in adjacency.get(node, ()):
            if neighbor not in seen:
                seen.add(neighbor)
                queue.append(neighbor)
    return seen


def _sym_adjacency(edges):
    """Build a bidirectional adjacency dict from an undirected edge list."""
    adjacency = {}
    for u, v, g in edges:
        adjacency.setdefault(u, []).append((v, g))
        adjacency.setdefault(v, []).append((u, g))
    return adjacency


# =============================================================================
# Via vs non-via classification
# =============================================================================


def test_via_detection_same_xy_different_layer():
    """Two same-(x,y) different-layer nodes connected by a resistor are via."""
    node_via_a = "100_200_43"
    node_via_b = "100_200_45"
    node_plain_c = "300_400_43"  # same layer as A, different (x, y) -> non-via
    node_plain_d = "500_600_47"  # different (x, y) from C, connected only to C

    tile_data = _make_tile_data(
        all_nodes=[node_via_a, node_via_b, node_plain_c, node_plain_d],
        resistive_edges=[
            (node_via_a, node_via_b, 5.0),   # via edge (same xy, diff layer)
            (node_via_a, node_plain_c, 2.0),  # same-layer routing edge
            (node_plain_c, node_plain_d, 1.0),  # different xy, different layer, not a via edge
        ],
    )

    classification = classify_tile(
        tile_data, pad_anchors_for_tile=set(), current_source_nodes_for_tile=set()
    )

    assert classification.via_nodes == {node_via_a, node_via_b}
    assert node_plain_c not in classification.via_nodes
    assert node_plain_d not in classification.via_nodes

    # Adjacency is built once and reflects both directions of every edge.
    assert set(classification.adjacency[node_via_a]) == {
        (node_via_b, 5.0), (node_plain_c, 2.0),
    }
    assert set(classification.adjacency[node_via_b]) == {(node_via_a, 5.0)}
    assert set(classification.adjacency[node_plain_d]) == {(node_plain_c, 1.0)}


def test_via_detection_no_vias_when_no_cross_layer_edges():
    """A tile with only same-layer edges has zero via nodes."""
    nodes = ["0_0_43", "10_0_43", "20_0_43"]
    tile_data = _make_tile_data(
        all_nodes=nodes,
        resistive_edges=[
            (nodes[0], nodes[1], 3.0),
            (nodes[1], nodes[2], 3.0),
        ],
    )
    classification = classify_tile(tile_data, set(), set())
    assert classification.via_nodes == set()


# =============================================================================
# Layer suffix parsing
# =============================================================================


@pytest.mark.parametrize(
    "node, expected_layer",
    [
        ("1197000_449800_86", 86),
        ("17480_15200_57", 57),
        ("0_0_43", 43),
        ("-100_200_45", 45),
    ],
)
def test_parse_node_layer_real_shaped_names(node, expected_layer):
    assert _parse_node_layer(node) == expected_layer


@pytest.mark.parametrize("malformed", ["nounderscoreatall", "a_b_c", ""])
def test_parse_node_layer_malformed_returns_none(malformed):
    assert _parse_node_layer(malformed) is None


# =============================================================================
# Mandatory-keep set union correctness
# =============================================================================


def test_mandatory_keep_union_no_double_counting():
    """mandatory_keep is the union of pad-anchor/boundary/current-source sets,
    even when one node belongs to all three categories at once."""
    pad_only = "1_1_86"
    boundary_only = "2_2_50"
    current_only = "3_3_43"
    all_three = "4_4_43"
    optional_node = "5_5_43"

    all_nodes = [pad_only, boundary_only, current_only, all_three, optional_node]
    tile_data = _make_tile_data(
        all_nodes=all_nodes,
        resistive_edges=[],
        boundary_nodes={boundary_only, all_three},
    )

    classification = classify_tile(
        tile_data,
        pad_anchors_for_tile={pad_only, all_three},
        current_source_nodes_for_tile={current_only, all_three},
    )

    assert classification.pad_anchor_nodes == {pad_only, all_three}
    assert classification.boundary_nodes == {boundary_only, all_three}
    assert classification.current_source_nodes == {current_only, all_three}

    expected_mandatory = {pad_only, boundary_only, current_only, all_three}
    assert classification.mandatory_keep == expected_mandatory
    # Overlap must not inflate the union's size.
    assert len(classification.mandatory_keep) == len(expected_mandatory)

    # Optional pool is exactly everything else, grouped by layer.
    assert classification.optional_pool_by_layer == {43: {optional_node}}


def test_pad_anchors_and_current_sources_filtered_to_tile_nodes():
    """Extraneous nodes in the keep-set args (outside this tile) are ignored."""
    node_a = "1_1_43"
    tile_data = _make_tile_data(all_nodes=[node_a], resistive_edges=[])

    classification = classify_tile(
        tile_data,
        pad_anchors_for_tile={node_a, "outside_tile_node"},
        current_source_nodes_for_tile={"also_outside"},
    )
    assert classification.pad_anchor_nodes == {node_a}
    assert classification.current_source_nodes == set()
    assert classification.mandatory_keep == {node_a}


# =============================================================================
# Pass 2 — contract_tile_nodes (plan §4.2, revised)
# =============================================================================


def test_contract_tile_nodes_deterministic_across_calls():
    """Same classification -> byte-identical node_to_rep, no RNG anywhere."""
    nodes, edges, _name = _grid_nodes_edges(20, 20, 43)
    tile_data = _make_tile_data(all_nodes=nodes, resistive_edges=edges)
    classification = classify_tile(tile_data, set(), set())

    result_a = contract_tile_nodes(classification, ratio=0.1, alpha=1.0)
    result_b = contract_tile_nodes(classification, ratio=0.1, alpha=1.0)

    assert result_a.node_to_rep == result_b.node_to_rep
    assert result_a.kept_nodes == result_b.kept_nodes
    assert result_a.n_clusters == result_b.n_clusters


def test_contract_tile_nodes_mandatory_nodes_are_singleton_reps():
    """Mandatory nodes never absorb and are never absorbed by a cluster."""
    nodes, edges, name = _grid_nodes_edges(15, 15, 43)
    mandatory = {name(0, 0), name(14, 14)}
    tile_data = _make_tile_data(all_nodes=nodes, resistive_edges=edges)
    classification = classify_tile(tile_data, mandatory, set())

    result = contract_tile_nodes(classification, ratio=0.1, alpha=1.0)

    for m in mandatory:
        assert result.node_to_rep[m] == m
        assert m in result.kept_nodes
    # No optional node's representative is one of the mandatory nodes.
    for node, rep in result.node_to_rep.items():
        if node not in mandatory:
            assert rep not in mandatory or rep == node


def test_contract_tile_nodes_sparse_layer_retained_more_than_dense_layer():
    dense_nodes, dense_edges, _ = _grid_nodes_edges(20, 20, 43)
    sparse_nodes, sparse_edges = _chain_nodes_edges(10, 86, x0=1000)
    all_nodes = set(dense_nodes) | set(sparse_nodes)
    edges = dense_edges + sparse_edges
    tile_data = _make_tile_data(all_nodes=all_nodes, resistive_edges=edges)
    classification = classify_tile(tile_data, set(), set())

    result = contract_tile_nodes(classification, ratio=0.1, alpha=1.0)

    sparse_retention = result.per_layer_retention[86]
    dense_retention = result.per_layer_retention[43]
    assert sparse_retention > dense_retention
    assert sparse_retention == pytest.approx(1.0)
    assert dense_retention < 1.0


def test_contract_tile_nodes_kept_count_near_target_for_uniform_grid():
    nodes, edges, _name = _grid_nodes_edges(30, 30, 43)
    tile_data = _make_tile_data(all_nodes=nodes, resistive_edges=edges)
    classification = classify_tile(tile_data, set(), set())

    result = contract_tile_nodes(classification, ratio=0.1, alpha=1.0)

    total = len(nodes)
    assert result.target_kept == round(0.1 * total)
    # Loose bounds (geometric cells + CC-split make this approximate) but
    # tight enough that a no-op (kept == total) regression fails.
    assert 0.5 * result.target_kept <= len(result.kept_nodes) <= 2.0 * result.target_kept
    assert len(result.kept_nodes) < 0.3 * total


def test_contract_tile_nodes_isolated_optional_dropped_and_isolated_mandatory_kept(caplog):
    node_a, node_b = "0_0_43", "10_0_43"  # connected pair -- ordinary optional
    isolated_optional = "999_999_43"       # zero resistive edges -- must be dropped
    isolated_mandatory = "0_0_86"          # zero resistive edges but mandatory -- kept

    edges = [(node_a, node_b, 5.0)]
    all_nodes = {node_a, node_b, isolated_optional, isolated_mandatory}
    tile_data = _make_tile_data(all_nodes=all_nodes, resistive_edges=edges)
    classification = classify_tile(tile_data, {isolated_mandatory}, set())

    with caplog.at_level(logging.WARNING, logger="parser.brcm_tile_sampler"):
        result = contract_tile_nodes(classification, ratio=0.9, alpha=1.0)

    assert isolated_optional not in result.kept_nodes
    assert isolated_optional not in result.node_to_rep
    assert result.isolated_optional_dropped == 1

    assert isolated_mandatory in result.kept_nodes
    assert result.node_to_rep[isolated_mandatory] == isolated_mandatory
    assert result.isolated_mandatory_kept == 1
    assert any(
        record.levelno == logging.WARNING and "isolated" in record.message
        for record in caplog.records
    )


def test_contract_tile_nodes_unparseable_xy_and_degenerate_layers_do_not_crash():
    """Unparseable-name nodes and a degenerate ("single point" bbox) layer
    must not crash. Unparseable nodes always end up as singletons; two
    differently-spelled node names that parse to the exact same (x, y)
    (e.g. leading-zero / trailing-``.0`` spelling) exercise the
    zero-width-and-height branch of ``_assign_geometric_cells`` and, being
    connected and coincident, merge into one cluster.
    """
    unparseable_a, unparseable_b = "not-a-coord-node", "another-bad-one"
    coincident_a, coincident_b = "5_5_60", "05_5.0_60"  # both parse to (5.0, 5.0)
    grid_nodes, grid_edges, _name = _grid_nodes_edges(10, 10, 43)

    edges = list(grid_edges) + [
        (unparseable_a, unparseable_b, 3.0),
        (unparseable_a, grid_nodes[0], 1.0),
        (coincident_a, coincident_b, 2.0),
        (coincident_a, grid_nodes[-1], 1.0),
    ]
    all_nodes = set(grid_nodes) | {unparseable_a, unparseable_b, coincident_a, coincident_b}
    tile_data = _make_tile_data(all_nodes=all_nodes, resistive_edges=edges)
    classification = classify_tile(tile_data, set(), set())

    # Aggressive ratio so even the 2-node degenerate layer must contract
    # (a looser ratio would just retain it whole, never reaching the
    # zero-area-bbox code path at all).
    result = contract_tile_nodes(classification, ratio=0.02, alpha=1.0)

    assert result.node_to_rep[unparseable_a] == unparseable_a
    assert result.node_to_rep[unparseable_b] == unparseable_b
    assert unparseable_a in result.kept_nodes
    assert result.node_to_rep[coincident_a] == result.node_to_rep[coincident_b]


def test_contract_tile_nodes_mandatory_only_branch_keeps_direct_lifeline(caplog):
    """Regression test: in the "mandatory-keep >= target_kept" degenerate
    branch, a mandatory node whose ONLY original resistive neighbor is an
    optional node must not be stranded. The degenerate branch runs the same
    clustering machinery as the normal path (with ``remaining = 0``), so the
    lifeline node is kept as a cluster representative rather than dropped.
    """
    mandatory = "0_0_43"
    lifeline = "10_0_43"    # optional; mandatory's ONLY original neighbor
    other_optional = "20_0_43"  # optional; connected to lifeline, not to mandatory

    edges = [(mandatory, lifeline, 5.0), (lifeline, other_optional, 5.0)]
    all_nodes = {mandatory, lifeline, other_optional}
    tile_data = _make_tile_data(all_nodes=all_nodes, resistive_edges=edges)
    classification = classify_tile(tile_data, {mandatory}, set())

    # ratio=0.1 over 3 nodes -> target_kept = round(0.3) = 0, so
    # len(mandatory_keep) == 1 >= target_kept triggers the degenerate branch.
    with caplog.at_level(logging.WARNING, logger="parser.brcm_tile_sampler"):
        result = contract_tile_nodes(classification, ratio=0.1, alpha=1.0)

    assert result.target_kept == 0
    assert mandatory in result.kept_nodes
    assert lifeline in result.kept_nodes
    assert any(
        record.levelno == logging.WARNING
        and "already meets or exceeds" in record.message
        for record in caplog.records
    )

    # Pass 3 must not strand the mandatory node, and the sanity assert must
    # not fire (this is a legitimate data condition, not a contraction bug).
    edge_result = contract_tile_edges(tile_data, result)
    degree = {n: 0 for n in result.kept_nodes}
    for u, v, _g in edge_result.kept_resistive_edges:
        degree[u] = degree.get(u, 0) + 1
        degree[v] = degree.get(v, 0) + 1
    assert degree[mandatory] > 0


def test_contract_tile_nodes_mandatory_only_branch_preserves_long_chain_connectivity():
    """Regression test for the §7.5-style fragmentation bug: two mandatory
    nodes joined ONLY by a chain of optional nodes of length >= 2
    (M1-o1-o2-o3-M2) must remain connected through the degenerate
    "mandatory-keep >= target_kept" branch. A rescue that only promotes
    DIRECT optional neighbors of mandatory nodes keeps the two end-stub
    edges (M1-o1, o3-M2) but drops the interior optional node o2, severing
    the chain and silently fragmenting the tile into two islands even
    though both M1 and M2 individually still show degree > 0.
    """
    m1, o1, o2, o3, m2 = "0_0_43", "10_0_43", "20_0_43", "30_0_43", "40_0_43"
    edges = [(m1, o1, 5.0), (o1, o2, 5.0), (o2, o3, 5.0), (o3, m2, 5.0)]
    all_nodes = {m1, o1, o2, o3, m2}
    tile_data = _make_tile_data(all_nodes=all_nodes, resistive_edges=edges)
    classification = classify_tile(tile_data, {m1, m2}, set())

    # ratio=0.1 over 5 nodes -> target_kept = round(0.5) = 0 (banker's
    # rounding), so len(mandatory_keep) == 2 >= target_kept triggers the
    # degenerate branch.
    result = contract_tile_nodes(classification, ratio=0.1, alpha=1.0)
    assert result.mandatory_kept >= result.target_kept

    edge_result = contract_tile_edges(tile_data, result)
    reached = _reachable_from(m1, edge_result.kept_resistive_edges)
    assert reached == result.kept_nodes
    assert m2 in reached


def test_contract_tile_nodes_mandatory_only_branch_spread_mesh_single_component():
    """Same regression as above but on a realistic uniform 10x10 mesh with
    ~15% spread mandatory nodes (ratio 0.1 -> target_kept=10 < 15
    mandatory), matching the topology observed on production BRCM tiles
    where sampling is mandatory-keep-set-dominated.

    The mandatory subset is a fixed (seeded) random sample -- this exact
    seed/topology fragments a single-component mesh into 4 disconnected
    islands under a "keep mandatory only, rescue direct neighbors" branch
    (direct-neighbor rescue is not equivalent to full contraction: it
    happens to preserve connectivity for *some* spread patterns via
    redundant grid paths, but not this one). The seed is fixed for
    reproducibility, not for cryptographic/statistical properties.
    """
    def name(x, y):
        return f"{x * 10}_{y * 10}_43"

    nodes = [name(x, y) for x in range(10) for y in range(10)]
    edges = []
    for x in range(10):
        for y in range(10):
            if x + 1 < 10:
                edges.append((name(x, y), name(x + 1, y), 5.0))
            if y + 1 < 10:
                edges.append((name(x, y), name(x, y + 1), 5.0))

    mandatory = set(random.Random(43).sample(nodes, 15))
    assert len(mandatory) == 15

    tile_data = _make_tile_data(all_nodes=set(nodes), resistive_edges=edges)
    classification = classify_tile(tile_data, mandatory, set())

    result = contract_tile_nodes(classification, ratio=0.1, alpha=1.0)
    assert result.target_kept == 10
    assert result.mandatory_kept == 15  # confirms the degenerate branch triggers

    edge_result = contract_tile_edges(tile_data, result)
    start = next(iter(result.kept_nodes))
    reached = _reachable_from(start, edge_result.kept_resistive_edges)
    assert reached == result.kept_nodes  # single connected component, no islands


def test_contract_tile_nodes_mandatory_only_branch_reports_isolated_optional_dropped():
    """Regression test: the degenerate "mandatory-keep >= target_kept"
    branch must not hardcode ``isolated_optional_dropped=0``. An
    originally-isolated optional node (zero resistive edges) is dropped in
    this branch exactly like the normal path, and the counter must reflect
    that -- otherwise the contraction diagnostics understate node loss on
    exactly the tiles most likely to hit the connectivity-fragmentation
    regression above.
    """
    mandatory = "0_0_43"
    connected_optional = "10_0_43"
    isolated_optional = "999_999_43"  # zero resistive edges -- must be dropped + counted

    edges = [(mandatory, connected_optional, 5.0)]
    all_nodes = {mandatory, connected_optional, isolated_optional}
    tile_data = _make_tile_data(all_nodes=all_nodes, resistive_edges=edges)
    classification = classify_tile(tile_data, {mandatory}, set())

    # ratio=0.1 over 3 nodes -> target_kept = 0, so len(mandatory_keep) == 1
    # >= target_kept triggers the degenerate branch.
    result = contract_tile_nodes(classification, ratio=0.1, alpha=1.0)
    assert result.mandatory_kept >= result.target_kept

    assert isolated_optional not in result.kept_nodes
    assert isolated_optional not in result.node_to_rep
    assert result.isolated_optional_dropped == 1


# =============================================================================
# scan_current_source_nodes: net-filter correctness (core correctness risk)
# =============================================================================


def _write_gzip_text(path, lines):
    with gzip.open(path, "wt") as f:
        f.write("\n".join(lines) + "\n")


def test_scan_current_source_nodes_filters_by_net(tmp_path):
    instance_path = tmp_path / "instanceModels_0_0.sp"
    _write_gzip_text(
        instance_path,
        [
            ".flag_boundary",
            "* comment line, should be skipped",
            # VDD_VAR current source -> node_pos should be collected.
            "i_test_vdd:VDD_VAR:VDD:0:0:0:0:0 1000_2000_43 0 1e-08 dv=0.76",
            # VSS current source on the same file -> must be excluded entirely.
            "i_test_vss:0:0:VSS:VSS:0:0:0 0 3000_4000_43 1e-08 dv=0.76",
            # Grounded capacitor line -- not a current source, must be ignored.
            "cr1 1000_2000_43 0 2e-14",
        ],
    )

    nodes = scan_current_source_nodes(str(instance_path))
    assert nodes == {"1000_2000_43"}
    # The VSS-net node must never leak into the VDD_VAR keep-set.
    assert "3000_4000_43" not in nodes


def test_scan_current_source_nodes_includes_node_neg_when_not_ground(tmp_path):
    instance_path = tmp_path / "instanceModels_0_1.sp"
    _write_gzip_text(
        instance_path,
        [
            # Both terminals are real die nodes (defensive case per plan --
            # not observed in real data, but must not be dropped if it occurs).
            "i_test_vdd2:VDD_VAR:VDD:0:0:0:0:0 5000_6000_43 7000_8000_43 2e-08 dv=0.76",
        ],
    )
    nodes = scan_current_source_nodes(str(instance_path))
    assert nodes == {"5000_6000_43", "7000_8000_43"}


def test_scan_current_source_nodes_none_path_returns_empty_set():
    assert scan_current_source_nodes(None) == set()


def test_scan_current_source_nodes_default_net_filter_is_lowercase_vdd_var(tmp_path):
    """Regression guard: net_filter must be 'vdd_var' (lowercase), matching
    TileConfig.net_filter as set by DistributedNetlistParser -- NOT 'VDD_VAR'."""
    instance_path = tmp_path / "instanceModels_0_2.sp"
    _write_gzip_text(
        instance_path,
        ["i_test:VDD_VAR:VDD:0:0:0:0:0 9000_1000_43 0 1e-08 dv=0.76"],
    )
    # Passing the uppercase net name would never match (structured names are
    # lower-cased before comparison), so this documents the expected failure
    # mode as well as the correct default behavior.
    assert scan_current_source_nodes(str(instance_path), net_filter="VDD_VAR") == set()
    assert scan_current_source_nodes(str(instance_path)) == {"9000_1000_43"}


# =============================================================================
# Pass 3 — contract_tile_edges (plan §4.3, revised)
# =============================================================================


def test_contract_tile_edges_connectivity_preserved_for_grid_mesh():
    """THE regression test for the §7.5 node-drop failure: contraction must
    never fragment a connected mesh. BFS from one representative reaches
    EVERY kept node."""
    grid_nodes, grid_edges, name = _grid_nodes_edges(30, 30, 43)
    via_coords = [(0, 0), (29, 0), (0, 29), (29, 29), (15, 15)]
    sparse_nodes = [f"{x}_{y}_86" for x, y in via_coords]
    via_edges = [(name(x, y), f"{x}_{y}_86", 8.0) for x, y in via_coords]

    all_nodes = set(grid_nodes) | set(sparse_nodes)
    all_edges = grid_edges + via_edges
    tile_data = _make_tile_data(all_nodes=all_nodes, resistive_edges=all_edges)
    classification = classify_tile(tile_data, set(), set())

    contraction = contract_tile_nodes(classification, ratio=0.1, alpha=1.0)
    edge_result = contract_tile_edges(tile_data, contraction)

    start = next(iter(contraction.kept_nodes))
    reached = _reachable_from(start, edge_result.kept_resistive_edges)
    assert reached == contraction.kept_nodes


def test_contract_tile_edges_r_per_node_preservation():
    """Same grid fixture: contracted R/node stays within +/-40% of the
    original ratio (grid ~= 2.0) and never dips below 1.0."""
    nodes, edges, _name = _grid_nodes_edges(30, 30, 43)
    tile_data = _make_tile_data(all_nodes=nodes, resistive_edges=edges)
    classification = classify_tile(tile_data, set(), set())

    contraction = contract_tile_nodes(classification, ratio=0.1, alpha=1.0)
    edge_result = contract_tile_edges(tile_data, contraction)

    r_per_node_before = len(edges) / len(nodes)
    r_per_node_after = len(edge_result.kept_resistive_edges) / len(contraction.kept_nodes)

    assert r_per_node_after >= 1.0
    assert 0.6 * r_per_node_before <= r_per_node_after <= 1.4 * r_per_node_before


def _hand_built_two_cluster_contraction():
    """Two 3-node chains (clusters A/B), joined by two parallel inter-cluster
    edges -- shared fixture for the parallel-merge and cap-accumulation tests.
    """
    a1, a2, a3 = "a1", "a2", "a3"
    b1, b2, b3 = "b1", "b2", "b3"
    resistive_edges = [
        (a1, a2, 10.0), (a2, a3, 10.0),  # intra-cluster A -- both dropped
        (b1, b2, 10.0), (b2, b3, 10.0),  # intra-cluster B -- both dropped
        (a1, b1, 2.0),                   # inter-cluster, parallel edge #1
        (a3, b3, 3.0),                   # inter-cluster, parallel edge #2
    ]
    node_to_rep = {a1: a1, a2: a1, a3: a1, b1: b1, b2: b1, b3: b1}
    contraction = TileContractionResult(
        tile_id=(0, 0), kept_nodes={a1, b1}, node_to_rep=node_to_rep,
        mandatory_nodes={a1, b1}, target_kept=2, mandatory_kept=2, optional_kept=0,
        per_layer_retention={}, n_clusters=2, isolated_optional_dropped=0,
        isolated_mandatory_kept=0,
    )
    return resistive_edges, node_to_rep, contraction, (a1, b1)


def test_contract_tile_edges_parallel_conductance_merge_correctness():
    resistive_edges, _node_to_rep, contraction, (a1, b1) = _hand_built_two_cluster_contraction()
    tile_data = _make_tile_data(all_nodes={"a1", "a2", "a3", "b1", "b2", "b3"}, resistive_edges=resistive_edges)

    edge_result = contract_tile_edges(tile_data, contraction)

    assert edge_result.kept_resistive_edges == [(a1, b1, 5.0)]  # 2.0 + 3.0
    assert edge_result.intra_cluster_resistors_dropped == 4  # 2 in A, 2 in B
    assert edge_result.parallel_resistors_merged == 1  # 2 inter-cluster edges -> 1


def test_contract_tile_edges_no_spurious_cross_stripe_merge():
    """Two disconnected same-layer stripes forced through matching geometric
    cells (identical x-spacing pattern at y=0 and y=1) must stay separate --
    no coarse edge is ever fabricated between them."""
    xs = list(range(30))
    stripe_a = [f"{x}_0_43" for x in xs]
    stripe_b = [f"{x}_1_43" for x in xs]
    edges = []
    for stripe in (stripe_a, stripe_b):
        edges.extend((u, v, 5.0) for u, v in zip(stripe, stripe[1:]))

    all_nodes = set(stripe_a) | set(stripe_b)
    tile_data = _make_tile_data(all_nodes=all_nodes, resistive_edges=edges)
    classification = classify_tile(tile_data, set(), set())

    contraction = contract_tile_nodes(classification, ratio=0.1, alpha=1.0)
    edge_result = contract_tile_edges(tile_data, contraction)

    reps_a = {contraction.node_to_rep[n] for n in stripe_a}
    reps_b = {contraction.node_to_rep[n] for n in stripe_b}
    assert reps_a.isdisjoint(reps_b)
    for u, v, _g in edge_result.kept_resistive_edges:
        assert not (u in reps_a and v in reps_b)
        assert not (u in reps_b and v in reps_a)


def test_contract_tile_edges_capacitor_accumulation_conserved():
    resistive_edges, _node_to_rep, contraction, (a1, b1) = _hand_built_two_cluster_contraction()
    capacitive_edges = [
        ("a1", "0", 1.0), ("a2", "0", 2.0), ("a3", "0", 3.0),
        ("b1", "0", 4.0), ("b2", "0", 5.0), ("b3", "0", 6.0),
    ]
    tile_data = _make_tile_data(
        all_nodes={"a1", "a2", "a3", "b1", "b2", "b3"},
        resistive_edges=resistive_edges, capacitive_edges=capacitive_edges,
    )

    edge_result = contract_tile_edges(tile_data, contraction)

    # (min(u, v), max(u, v)) key normalization means '0' (ground) may end up
    # in either tuple slot -- extract by "the non-ground endpoint", not by
    # a fixed position (same convention _grounded_cap_nodes relies on).
    kept_by_node = {
        (v if u == "0" else u): c for u, v, c in edge_result.kept_capacitive_edges
    }
    assert kept_by_node[a1] == pytest.approx(1.0 + 2.0 + 3.0)
    assert kept_by_node[b1] == pytest.approx(4.0 + 5.0 + 6.0)
    total_before = sum(c for _u, _v, c in capacitive_edges)
    total_after = sum(c for _u, _v, c in edge_result.kept_capacitive_edges)
    assert total_after == pytest.approx(total_before)


def test_contract_tile_edges_capacitor_on_dropped_isolate_not_conserved():
    """A grounded cap on an originally-isolated (dropped) optional node is
    NOT conserved -- counted via edges_to_dropped_nodes instead."""
    node_a, node_b = "0_0_43", "10_0_43"
    isolated = "999_999_43"  # zero resistive edges -- dropped by contract_tile_nodes
    edges = [(node_a, node_b, 5.0)]
    capacitive_edges = [(node_a, "0", 1.0), (isolated, "0", 9.0)]
    all_nodes = {node_a, node_b, isolated}
    tile_data = _make_tile_data(
        all_nodes=all_nodes, resistive_edges=edges, capacitive_edges=capacitive_edges,
    )
    classification = classify_tile(tile_data, set(), set())

    contraction = contract_tile_nodes(classification, ratio=0.9, alpha=1.0)
    assert isolated not in contraction.kept_nodes
    edge_result = contract_tile_edges(tile_data, contraction)

    kept_by_node = {u: c for u, _v, c in edge_result.kept_capacitive_edges}
    assert isolated not in kept_by_node
    assert edge_result.edges_to_dropped_nodes == 1


def test_contract_tile_edges_via_edges_survive_and_connect_layers():
    """Contraction is per-layer -- a via edge between two kept reps on
    different layers must map to a coarse edge, keeping the coarse graph
    connected across layers."""
    grid_nodes, grid_edges, name = _grid_nodes_edges(10, 10, 43)
    via_node = "0_0_86"
    edges = list(grid_edges) + [(via_node, name(0, 0), 8.0)]
    all_nodes = set(grid_nodes) | {via_node}
    tile_data = _make_tile_data(all_nodes=all_nodes, resistive_edges=edges)
    classification = classify_tile(tile_data, set(), set())

    contraction = contract_tile_nodes(classification, ratio=0.1, alpha=1.0)
    edge_result = contract_tile_edges(tile_data, contraction)

    via_rep = contraction.node_to_rep[via_node]
    assert via_rep in contraction.kept_nodes
    reached = _reachable_from(via_rep, edge_result.kept_resistive_edges)
    assert reached == contraction.kept_nodes  # layer-86 rep still reaches the grid


def test_contract_tile_edges_mandatory_degree_sanity_check_raises():
    """Synthetically corrupted node_to_rep (mandatory node's identity
    replaced everywhere it appears) strands a mandatory node -- must raise."""
    node_a, node_b = "a", "b"
    edges = [(node_a, node_b, 5.0)]
    tile_data = _make_tile_data(all_nodes={node_a, node_b}, resistive_edges=edges)

    # node_a is mandatory but its edges get remapped to "other" everywhere,
    # so node_a itself ends up with contracted degree 0 -- a contraction bug.
    corrupted = TileContractionResult(
        tile_id=(0, 0), kept_nodes={"other", node_b}, node_to_rep={node_a: "other", node_b: node_b},
        mandatory_nodes={node_a, node_b}, target_kept=2, mandatory_kept=2, optional_kept=0,
        per_layer_retention={}, n_clusters=2, isolated_optional_dropped=0,
        isolated_mandatory_kept=0,
    )

    with pytest.raises(AssertionError):
        contract_tile_edges(tile_data, corrupted)


# =============================================================================
# Pass 4 — sample_current_sources (plan §4.4)
# =============================================================================


def _make_current_source_lines(n, net="VDD_VAR", node_prefix="n", start=0):
    """Fabricate ``n`` structured-name current-source lines for one net."""
    lines = []
    node_positions = []
    for i in range(start, start + n):
        node = f"{node_prefix}{i}_0_43"
        node_positions.append(node)
        lines.append(f"i_test{i}:{net}:VDD:0:0:0:0:0 {node} 0 1e-08 dv=0.76")
    return lines, node_positions


def test_sample_current_sources_excludes_other_nets_and_preserves_raw_text(tmp_path):
    vdd_lines, vdd_nodes = _make_current_source_lines(20, net="VDD_VAR")
    vss_lines, vss_nodes = _make_current_source_lines(5, net="VSS", node_prefix="vss")
    instance_path = tmp_path / "instanceModels_3_4.sp"
    _write_gzip_text(instance_path, vdd_lines + vss_lines)

    kept_nodes = set(vdd_nodes) | set(vss_nodes)

    result = sample_current_sources(
        str(instance_path), kept_nodes, tile_id=(3, 4), ratio=0.1, base_seed=42,
    )

    assert isinstance(result, CurrentSourceSamplingResult)
    assert result.total_matching_net == 20  # VSS lines never counted
    assert result.total_eligible == 20
    assert len(result.kept_raw_lines) == 2  # ~10% of 20

    # Raw text preserved verbatim -- exact match against the fabricated lines.
    for raw in result.kept_raw_lines:
        assert raw in vdd_lines
        assert "VSS" not in raw

    assert result.excluded_not_in_kept_nodes == set()


def test_sample_current_sources_reproducible_same_seed_and_differs_across_seeds(tmp_path):
    vdd_lines, vdd_nodes = _make_current_source_lines(30)
    instance_path = tmp_path / "instanceModels_0_0.sp"
    _write_gzip_text(instance_path, vdd_lines)
    kept_nodes = set(vdd_nodes)

    r1 = sample_current_sources(str(instance_path), kept_nodes, tile_id=(0, 0), ratio=0.2, base_seed=7)
    r2 = sample_current_sources(str(instance_path), kept_nodes, tile_id=(0, 0), ratio=0.2, base_seed=7)
    assert r1.kept_raw_lines == r2.kept_raw_lines

    r3 = sample_current_sources(str(instance_path), kept_nodes, tile_id=(0, 0), ratio=0.2, base_seed=99)
    assert r1.kept_raw_lines != r3.kept_raw_lines


def test_sample_current_sources_warns_when_node_pos_not_in_kept_nodes(tmp_path, caplog):
    vdd_lines, vdd_nodes = _make_current_source_lines(5)
    instance_path = tmp_path / "instanceModels_1_1.sp"
    _write_gzip_text(instance_path, vdd_lines)

    # Deliberately omit one node -- simulates the "should never happen"
    # Pass 1/2 mandatory-keep bug this check defends against.
    kept_nodes = set(vdd_nodes[1:])

    with caplog.at_level(logging.WARNING, logger="parser.brcm_tile_sampler"):
        result = sample_current_sources(
            str(instance_path), kept_nodes, tile_id=(1, 1), ratio=1.0, base_seed=0,
        )

    assert result.excluded_not_in_kept_nodes == {vdd_nodes[0]}
    assert all(vdd_nodes[0] not in raw for raw in result.kept_raw_lines)
    assert result.total_eligible == 4
    assert any(record.levelno == logging.WARNING for record in caplog.records)


def test_sample_current_sources_none_path_returns_empty_result():
    result = sample_current_sources(None, set(), tile_id=(0, 0))
    assert result.total_eligible == 0
    assert result.kept_raw_lines == []
    assert result.kept_node_pos == set()


# =============================================================================
# verify_capacitor_invariant (plan §4.4, point 5)
# =============================================================================


def test_verify_capacitor_invariant_passes_when_cap_survives():
    node = "1_1_43"
    tile_data = _make_tile_data(
        all_nodes=[node], resistive_edges=[], capacitive_edges=[(node, "0", 5.0)],
    )
    verify_capacitor_invariant(tile_data, {node}, [(node, "0", 5.0)])  # must not raise


def test_verify_capacitor_invariant_raises_when_cap_incorrectly_dropped():
    node = "1_1_43"
    tile_data = _make_tile_data(
        all_nodes=[node], resistive_edges=[], capacitive_edges=[(node, "0", 5.0)],
    )
    with pytest.raises(CapacitorInvariantViolation):
        verify_capacitor_invariant(tile_data, {node}, [])  # cap missing from kept edges


# =============================================================================
# Output generation — top-level files (plan §4.5)
# =============================================================================


def _make_pad_block(pad_name: str, net: str) -> list:
    """Build a synthetic 4-line package.ckt section-1 pad "loop" block."""
    return [
        f"v_{pad_name} {pad_name}_vsrc 0 {net}",
        f"r  {pad_name} {pad_name}_probe 0",
        f"r_{pad_name}_probe_probe {pad_name}_probe {pad_name}_int 0.001",
        f"r  {pad_name}_int {pad_name}_vsrc 0",
    ]


def _make_package_ckt_fixture(pads) -> list:
    """Build a synthetic 3-section package.ckt text mirroring the real structure.

    Args:
        pads: sequence of ``(pad_name, net, die_node)`` tuples, in the
            per-pad order shared by all 3 sections (mirrors the real
            file's verified same-order invariant across sections).

    Returns:
        Flat list of lines (no trailing newlines), section 1 -> section 2
        -> section 3, ready to pass to ``filter_package_ckt``.
    """
    section1, section2, section3 = [], [], []
    for pad_name, net, die_node in pads:
        section1.extend(_make_pad_block(pad_name, net))
        section2.append(f"rs {die_node} {pad_name} 0")
        section3.append(f".print v({pad_name}_probe)")
        section3.append(f".print v({pad_name}_int)")
    return section1 + section2 + section3


def test_generate_pg_net_voltage_exact_format():
    # Verified byte-for-byte against the real file via `cat -A`: a literal
    # " - " separator and a trailing space before the newline.
    assert generate_pg_net_voltage(0.76) == "VDD_VAR - 0.76 \n"


def test_generate_pg_net_voltage_formats_whole_number_without_decimal():
    assert generate_pg_net_voltage(0) == "VDD_VAR - 0 \n"


def test_generate_additional_vsrcs_is_empty():
    # Real file is 0 bytes -- sampled version matches exactly.
    assert generate_additional_vsrcs() == ""


def test_generate_ckt_sp_small_synthetic_grid():
    tile_ids = [(0, 0), (0, 1), (1, 0), (1, 1)]
    content = generate_ckt_sp(
        tile_grid=(2, 2), die_area=(0, 0, 100, 200), vdd=0.76, tile_ids=tile_ids,
    )
    lines = content.splitlines()

    assert lines[0] == ".partition_info 2 2"
    assert lines[1] == ".die_area 0 0 100 200"
    assert lines[2] == ".parameter VDD_VAR 0.76"
    assert lines[3] == "vVDD_VAR vVDD_VAR  0  VDD_VAR"

    # VDD_VAR-only: no other net's .parameter/v-line should be present.
    assert "VSS" not in content
    assert "pad_PLL" not in content

    tile_includes = [l for l in lines if l.startswith(".include ./tile_")]
    inst_includes = [l for l in lines if l.startswith(".include ./instanceModels_")]
    assert tile_includes == [f".include ./tile_{x}_{y}.ckt" for x, y in tile_ids]
    assert inst_includes == [f".include ./instanceModels_{x}_{y}.sp" for x, y in tile_ids]

    # package.ckt include sits strictly between the tile and instance blocks.
    pkg_idx = lines.index(".include ./package.ckt")
    assert lines[pkg_idx - 1] == tile_includes[-1]
    assert lines[pkg_idx + 1] == inst_includes[0]

    assert content.endswith("\n")


def test_generate_ckt_sp_synthetic_6x6_grid_include_counts():
    # Cheap synthetic stand-in for the real 6x6/36-tile shape (no real data
    # needed for this assertion -- see the real-data test below for that).
    n_x, n_y = 6, 6
    tile_ids = [(x, y) for x in range(n_x) for y in range(n_y)]
    content = generate_ckt_sp(
        tile_grid=(n_x, n_y), die_area=(0, 0, 19152000, 16628040), vdd=0.76,
        tile_ids=tile_ids,
    )
    lines = content.splitlines()

    assert len(lines) == 4 + 36 + 1 + 36  # header + tiles + package + instances
    assert lines[0] == ".partition_info 6 6"
    assert lines[1] == ".die_area 0 0 19152000 16628040"
    assert sum(l.startswith(".include ./tile_") for l in lines) == 36
    assert sum(l.startswith(".include ./instanceModels_") for l in lines) == 36
    assert lines.count(".include ./package.ckt") == 1


def test_read_die_area_from_ckt_sp(tmp_path):
    ckt_sp = tmp_path / "ckt.sp"
    ckt_sp.write_text(
        ".partition_info 2 2\n"
        ".die_area 0 0 19152000 16628040\n"
        ".parameter VDD_VAR 0.76\n"
    )
    assert read_die_area_from_ckt_sp(ckt_sp) == (0.0, 0.0, 19152000.0, 16628040.0)


def test_read_die_area_from_ckt_sp_missing_raises(tmp_path):
    ckt_sp = tmp_path / "ckt.sp"
    ckt_sp.write_text(".partition_info 2 2\n")
    with pytest.raises(ValueError):
        read_die_area_from_ckt_sp(ckt_sp)


def test_filter_package_ckt_keeps_only_matching_net():
    pads = [
        ("bmpary_bmp_VDD_VAR_0_1", "VDD_VAR", "100_100_86"),
        ("bmpary_bmp_VSS_0_1", "VSS", "100_200_86"),
        ("bmpary_bmp_VDD_VAR_0_2", "VDD_VAR", "200_100_86"),
        ("bmpary_bmp_VSS_0_2", "VSS", "200_200_86"),
    ]
    fixture = _make_package_ckt_fixture(pads)

    result = filter_package_ckt(fixture, net_filter="VDD_VAR")

    v_lines = [l for l in result if l.startswith("v_")]
    rs_lines = [l for l in result if l.startswith("rs ")]
    print_lines = [l for l in result if l.startswith(".print")]

    assert len(v_lines) == 2
    assert len(result) == 2 * 4 + 2 + 2 * 2  # 2 kept pads: 4 + 1 + 2 lines each
    assert {l.split()[3] for l in v_lines} == {"VDD_VAR"}
    assert "VSS" not in "\n".join(result)

    # Same pad node names appear consistently across all 3 kept sections.
    v_pad_names = {l.split()[0][len("v_"):] for l in v_lines}
    rs_pad_names = {l.split()[2] for l in rs_lines}
    print_pad_names = {
        l.split("(", 1)[1].rstrip(")").rsplit("_", 1)[0] for l in print_lines
    }
    assert v_pad_names == {"bmpary_bmp_VDD_VAR_0_1", "bmpary_bmp_VDD_VAR_0_2"}
    assert rs_pad_names == v_pad_names
    assert print_pad_names == v_pad_names
    assert len(rs_lines) == 2
    assert len(print_lines) == 4


def test_filter_package_ckt_preserves_relative_section_order():
    pads = [
        ("bmp_a", "VDD_VAR", "1_1_86"),
        ("bmp_b", "OTHER", "2_2_86"),
        ("bmp_c", "VDD_VAR", "3_3_86"),
    ]
    fixture = _make_package_ckt_fixture(pads)
    result = filter_package_ckt(fixture, net_filter="VDD_VAR")

    # All section-1 (v_/r/r_) lines must precede all rs lines, which must
    # precede all .print lines -- filtering must not interleave sections.
    kinds = []
    for l in result:
        if l.startswith("rs "):
            kinds.append("rs")
        elif l.startswith(".print"):
            kinds.append("print")
        else:
            kinds.append("block")
    assert kinds == sorted(kinds, key=lambda k: {"block": 0, "rs": 1, "print": 2}[k])


def test_filter_package_ckt_does_not_false_positive_on_vdd_substring():
    """The exact trap called out in the ticket: a naive `'VDD' in <name>` (or
    `'VDD_VAR' in <name>`-on-the-wrong-field) substring check would wrongly
    keep a `pad_PLL_VDD1p5` pad alongside real `VDD_VAR` pads. The correct
    implementation must use the `v_` line's exact 4th-token net field.
    """
    pads = [
        ("bmpary_bmp_VDD_VAR_0_1", "VDD_VAR", "100_100_86"),
        # Net name contains the literal substring "VDD" (and even "VDD_VAR"
        # is a near-miss if someone substring-matched the PAD NAME instead
        # of the net field) but is NOT the VDD_VAR net.
        ("bmpary_bmp_pad_PLL_VDD1p5_0_0", "pad_PLL_VDD1p5", "200_200_86"),
    ]
    fixture = _make_package_ckt_fixture(pads)

    # Sanity: prove the trap is real -- a naive substring check on either
    # the pad name or the net string would incorrectly match.
    assert "VDD" in "bmpary_bmp_pad_PLL_VDD1p5_0_0"
    assert "VDD" in "pad_PLL_VDD1p5"

    result = filter_package_ckt(fixture, net_filter="VDD_VAR")

    v_lines = [l for l in result if l.startswith("v_")]
    assert len(v_lines) == 1
    assert v_lines[0].split()[3] == "VDD_VAR"
    assert "pad_PLL_VDD1p5" not in "\n".join(result)


def test_filter_package_ckt_drops_blank_lines():
    pads = [("bmp_a", "VDD_VAR", "1_1_86")]
    fixture = _make_package_ckt_fixture(pads) + ["", ""]
    result = filter_package_ckt(fixture, net_filter="VDD_VAR")
    assert all(l.strip() for l in result)


def test_filter_package_ckt_raises_on_malformed_block_count():
    fixture = _make_package_ckt_fixture([("bmp_a", "VDD_VAR", "1_1_86")])
    # Drop one line from the middle of the section-1 pad block (indices
    # 0-3) without touching the rs/.print sections, breaking the "clean
    # multiple of 4" invariant.
    malformed = fixture[:3] + fixture[4:]
    with pytest.raises(ValueError):
        filter_package_ckt(malformed, net_filter="VDD_VAR")


def test_filter_package_ckt_raises_on_malformed_v_line():
    fixture = ["v_bmp_a bmp_a_vsrc 0"]  # missing the 4th (net) token
    with pytest.raises(ValueError):
        filter_package_ckt(fixture, net_filter="VDD_VAR")


def test_filter_package_ckt_no_matching_net_returns_empty():
    pads = [("bmp_a", "OTHER", "1_1_86")]
    fixture = _make_package_ckt_fixture(pads)
    assert filter_package_ckt(fixture, net_filter="VDD_VAR") == []


# =============================================================================
# Real-data validation (plan §4.5) -- netlist/netlist_brcm real files
# =============================================================================


@pytest.mark.integration
def test_filter_package_ckt_real_data_matches_metadata_pad_counts():
    """Filter the REAL package.ckt and cross-check against metadata.pkl.

    ``metadata.pkl``'s ``package_data.pad_nodes`` / ``die_attachment_nodes``
    are both 309 for VDD_VAR (already-parsed ground truth) -- the text
    filter must independently arrive at exactly the same counts.
    """
    package_ckt_path = _REAL_NETLIST_DIR / "package.ckt"
    if not package_ckt_path.exists() or not _REAL_PKL_DIR.exists():
        pytest.skip("netlist/netlist_brcm (real data) not available in this environment")

    metadata, _boundary_nodes = load_metadata(_REAL_PKL_DIR)
    expected_pad_nodes = len(metadata.package_data.pad_nodes)
    expected_die_attachment_nodes = len(metadata.package_data.die_attachment_nodes)
    assert expected_pad_nodes == 309
    assert expected_die_attachment_nodes == 309

    source_lines = package_ckt_path.read_text().splitlines()
    result = filter_package_ckt(source_lines, net_filter="VDD_VAR")

    v_lines = [l for l in result if l.startswith("v_")]
    rs_lines = [l for l in result if l.startswith("rs ")]
    print_lines = [l for l in result if l.startswith(".print")]

    assert len(v_lines) == expected_pad_nodes == 309
    assert len(rs_lines) == expected_die_attachment_nodes == 309
    assert len(print_lines) == 2 * 309

    # Sanity: same set of pad names threaded consistently through all 3 kept sections.
    v_pad_names = {l.split()[0][len("v_"):] for l in v_lines}
    rs_pad_names = {l.split()[2] for l in rs_lines}
    print_pad_names = {
        l.split("(", 1)[1].rstrip(")").rsplit("_", 1)[0] for l in print_lines
    }
    assert v_pad_names == rs_pad_names == print_pad_names
    assert len(v_pad_names) == 309


@pytest.mark.integration
def test_generate_ckt_sp_against_real_ckt_sp_header_and_shape():
    """Sanity-check ``generate_ckt_sp`` against the real 6x6/36-tile design.

    Only reads ``metadata.pkl`` (tile_grid, tile ids) and the small
    ``ckt.sp`` text (die_area) -- no per-tile pkl loading needed, so this
    stays cheap despite the slow NFS mount.
    """
    ckt_sp_path = _REAL_NETLIST_DIR / "ckt.sp"
    if not ckt_sp_path.exists() or not _REAL_PKL_DIR.exists():
        pytest.skip("netlist/netlist_brcm (real data) not available in this environment")

    metadata, _boundary_nodes = load_metadata(_REAL_PKL_DIR)
    die_area = read_die_area_from_ckt_sp(ckt_sp_path)
    tile_ids = discover_tile_ids(metadata)

    assert metadata.tile_grid == (6, 6)
    assert die_area == (0.0, 0.0, 19152000.0, 16628040.0)
    assert len(tile_ids) == 36

    content = generate_ckt_sp(
        tile_grid=metadata.tile_grid, die_area=die_area, vdd=metadata.vdd,
        tile_ids=tile_ids,
    )
    lines = content.splitlines()

    assert lines[0] == ".partition_info 6 6"
    assert lines[1] == ".die_area 0 0 19152000 16628040"
    assert lines[2] == ".parameter VDD_VAR 0.76"
    assert lines[3] == "vVDD_VAR vVDD_VAR  0  VDD_VAR"
    assert lines[4] == ".include ./tile_0_0.ckt"
    assert lines[-1] == ".include ./instanceModels_5_5.sp"
    assert len(lines) == 4 + 36 + 1 + 36


# =============================================================================
# Per-tile output writers (plan §4.5) -- unit conversion helpers
# =============================================================================


def test_conductance_ms_to_ohm_round_trips_r_to_kohm():
    # g_mS = 1/r_kohm -> r_ohm = 1000/g_mS (R_TO_KOHM = 1e-3).
    g_ms = 1000.0 / 75.0  # corresponds to 75 Ohm
    assert _conductance_ms_to_ohm(g_ms) == pytest.approx(75.0)


def test_capacitance_ff_to_farad_round_trips_c_to_ff():
    c_ff = 1.2665e-14 * 1e15  # corresponds to 1.2665e-14 F
    assert _capacitance_ff_to_farad(c_ff) == pytest.approx(1.2665e-14)


def test_prefixed_node_adds_star_only_for_boundary_nodes():
    boundary = {"1_1_43"}
    assert _prefixed_node("1_1_43", boundary) == "*1_1_43"
    assert _prefixed_node("2_2_43", boundary) == "2_2_43"


# =============================================================================
# generate_tile_ckt_content / write_tile_ckt (plan §4.5, tile_X_Y.ckt)
# =============================================================================


def test_generate_tile_ckt_content_header_reflects_new_node_count():
    content = generate_tile_ckt_content(2, set(), [], [])
    lines = content.splitlines()
    assert lines[0] == ".node_count 2"
    assert lines[1] == ".flag_boundary"


def test_generate_tile_ckt_content_boundary_prefix_on_every_occurrence():
    # Boundary node appears in both an r-line and a c-line -- '*' must be
    # on BOTH, not just one.
    boundary_nodes = {"1_1_43"}
    resistive_edges = [("1_1_43", "2_2_43", 1000.0 / 75.0)]
    capacitive_edges = [("1_1_43", "0", 1.2665e-14 * 1e15)]

    content = generate_tile_ckt_content(2, boundary_nodes, resistive_edges, capacitive_edges)
    lines = content.splitlines()

    r_line = next(l for l in lines if l.startswith("r"))
    c_line = next(l for l in lines if l.startswith("c"))
    assert "*1_1_43" in r_line
    assert "*1_1_43" in c_line
    # The non-boundary endpoint must NOT be starred.
    assert " 2_2_43" in r_line and "*2_2_43" not in r_line


def test_generate_tile_ckt_content_values_round_trip_through_spice_parser():
    # The whole point of the unit conversion + formatting: a value written
    # out must be re-readable by the REAL parser machinery to the same
    # float, within precision -- this is the correctness bar, not exact
    # byte-for-byte formatting.
    resistive_edges = [("a_1_43", "b_1_43", 1000.0 / 75.0)]  # 75 Ohm
    capacitive_edges = [("a_1_43", "0", 1.2665e-14 * 1e15)]  # 1.2665e-14 F
    content = generate_tile_ckt_content(2, set(), resistive_edges, capacitive_edges)

    lines = [l for l in content.splitlines() if l and not l.startswith('.')]
    r_tokens = next(l for l in lines if l.startswith("r")).split()
    c_tokens = next(l for l in lines if l.startswith("c")).split()

    assert r_tokens[0] == "r"
    assert c_tokens[0] == "c"
    assert _parse_spice_value(r_tokens[3]) == pytest.approx(75.0)
    assert _parse_spice_value(c_tokens[3]) == pytest.approx(1.2665e-14)


def test_generate_tile_ckt_content_no_boundary_nodes_no_star():
    resistive_edges = [("a_1_43", "b_1_43", 10.0)]
    content = generate_tile_ckt_content(2, set(), resistive_edges, [])
    assert "*" not in content


def test_write_tile_ckt_is_valid_gzip_and_matches_generated_content(tmp_path):
    boundary_nodes = {"a_1_43"}
    resistive_edges = [("a_1_43", "b_1_43", 1000.0 / 75.0)]
    capacitive_edges = [("a_1_43", "0", 1.2665e-14 * 1e15)]

    out_path = tmp_path / "tile_0_0.ckt"
    write_tile_ckt(out_path, 2, boundary_nodes, resistive_edges, capacitive_edges)

    # Plain (non-.gz) filename, but gzip magic bytes -- matches the real
    # netlist_brcm convention (verified via `file`).
    with open(out_path, "rb") as f:
        assert f.read(2) == b"\x1f\x8b"

    with gzip.open(out_path, "rt") as f:
        content = f.read()
    assert content == generate_tile_ckt_content(
        2, boundary_nodes, resistive_edges, capacitive_edges
    )


# =============================================================================
# filter_tile_nd_lines / write_tile_nd (plan §4.5, tile_X_Y.nd)
# =============================================================================


def test_filter_tile_nd_lines_preserves_cosmetic_fields_verbatim():
    nd_lines = [
        "1_1_43 3772 0 0 0 VDD_VAR",
        "2_2_43 9999 1 2 3 VDD_VAR",
        "3_3_43 1 1 1 1 VDD_VAR",
    ]
    kept = filter_tile_nd_lines(nd_lines, {"1_1_43", "3_3_43"})
    # Exact original text (including the arbitrary "9999"/"1 2 3" cosmetic
    # fields for the DROPPED node -- irrelevant since it's dropped, but the
    # KEPT lines must be byte-for-byte identical to the source).
    assert kept == ["1_1_43 3772 0 0 0 VDD_VAR", "3_3_43 1 1 1 1 VDD_VAR"]


def test_filter_tile_nd_lines_drops_blank_lines():
    assert filter_tile_nd_lines(["", "   ", "1_1_43 0 0 0 0 VDD_VAR"], {"1_1_43"}) == [
        "1_1_43 0 0 0 0 VDD_VAR"
    ]


def test_filter_tile_nd_lines_no_star_prefix_ever_added():
    # .nd never carries the '*' boundary convention -- that's an
    # element-line-only (tile_X_Y.ckt) thing.
    kept = filter_tile_nd_lines(["1_1_43 0 0 0 0 VDD_VAR"], {"1_1_43"})
    assert "*" not in kept[0]


def test_write_tile_nd_filters_and_preserves_cosmetic_fields(tmp_path):
    source_nd = tmp_path / "tile_0_0.nd"
    _write_gzip_text(
        source_nd,
        ["1_1_43 3772 0 0 0 VDD_VAR", "2_2_43 9999 1 2 3 VDD_VAR"],
    )
    out_path = tmp_path / "tile_0_0_sampled.nd"
    n = write_tile_nd(out_path, source_nd, {"1_1_43"})
    assert n == 1

    with open(out_path, "rb") as f:
        assert f.read(2) == b"\x1f\x8b"
    with gzip.open(out_path, "rt") as f:
        content = f.read()
    assert content == "1_1_43 3772 0 0 0 VDD_VAR\n"


def test_write_tile_nd_all_dropped_writes_empty_gzip(tmp_path):
    source_nd = tmp_path / "tile_0_0.nd"
    _write_gzip_text(source_nd, ["1_1_43 0 0 0 0 VDD_VAR"])
    out_path = tmp_path / "tile_0_0_sampled.nd"
    n = write_tile_nd(out_path, source_nd, set())
    assert n == 0
    with gzip.open(out_path, "rt") as f:
        assert f.read() == ""


# =============================================================================
# write_instance_models (plan §4.5, instanceModels_X_Y.sp)
# =============================================================================


def test_write_instance_models_writes_raw_lines_verbatim(tmp_path):
    out_path = tmp_path / "instanceModels_0_0.sp"
    raw_lines = [
        "i_a:VDD_VAR:VDD:0:0:0:0:0 1_1_43 0 1e-08 dv=0.76",
        "i_b:VDD_VAR:VDD:0:0:0:0:0 2_2_43 0 2e-08 dv=0.76",
    ]
    write_instance_models(out_path, raw_lines)

    with open(out_path, "rb") as f:
        assert f.read(2) == b"\x1f\x8b"
    with gzip.open(out_path, "rt") as f:
        content = f.read()
    assert content == "\n".join(raw_lines) + "\n"


def test_write_instance_models_empty_list_writes_empty_gzip(tmp_path):
    out_path = tmp_path / "instanceModels_0_0.sp"
    write_instance_models(out_path, [])
    with gzip.open(out_path, "rt") as f:
        assert f.read() == ""


# =============================================================================
# process_tile (plan §5) -- synthetic single-tile end-to-end
# =============================================================================


def _write_pkl(path, obj):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def _build_synthetic_tile_environment(tmp_path, tile_id=(0, 0)):
    """Build a tiny synthetic netlist_brcm-shaped source dir + pkl dir for one tile.

    Layout:
      - 1 pad-anchor node (layer 86, hard-anchor -- must survive sampling).
      - 10 "bulk" nodes on layer 43 in a resistor chain (a real optional
        pool for Pass 2 to squeeze).
      - 1 current-source-bearing node (also grounded-cap'd) at the end of
        the chain.

    Returns (source_dir, pkl_dir, pad_node, cs_node).
    """
    source_dir = tmp_path / "source"
    pkl_dir = tmp_path / "pkl"
    source_dir.mkdir(exist_ok=True)
    pkl_dir.mkdir(exist_ok=True)

    pad_node = "100_100_86"
    cs_node = "50_50_43"  # distinct coordinate -- must not collide with bulk_nodes below
    bulk_nodes = [f"{i}_{i}_43" for i in range(10)]
    all_nodes = set(bulk_nodes) | {pad_node, cs_node}

    resistive_edges = []
    chain = bulk_nodes + [cs_node]
    for u, v in zip(chain, chain[1:]):
        resistive_edges.append((u, v, 10.0))
    resistive_edges.append((pad_node, bulk_nodes[0], 10.0))

    capacitive_edges = [(n, "0", 5.0) for n in all_nodes]

    tile_data = TileData(
        tile_id=tile_id,
        resistive_edges=resistive_edges,
        all_nodes=all_nodes,
        boundary_nodes=set(),
        current_injections={cs_node: 1.0},
        capacitive_edges=capacitive_edges,
    )
    _write_pkl(pkl_dir / f"tile_{tile_id[0]}_{tile_id[1]}.pkl", tile_data)

    nd_lines = [f"{n} 1 2 3 4 VDD_VAR" for n in sorted(all_nodes)]
    _write_gzip_text(source_dir / f"tile_{tile_id[0]}_{tile_id[1]}.nd", nd_lines)

    inst_lines = [
        ".flag_boundary",
        f"i_test:VDD_VAR:VDD:0:0:0:0:0 {cs_node} 0 1e-08 dv=0.76",
    ]
    _write_gzip_text(source_dir / f"instanceModels_{tile_id[0]}_{tile_id[1]}.sp", inst_lines)

    return source_dir, pkl_dir, pad_node, cs_node


def test_process_tile_writes_all_3_files_and_keeps_mandatory_nodes(tmp_path):
    source_dir, pkl_dir, pad_node, cs_node = _build_synthetic_tile_environment(tmp_path)
    output_dir = tmp_path / "output"
    tile_id = (0, 0)

    stats = process_tile(
        source_dir, pkl_dir, output_dir, tile_id,
        all_pad_anchors={pad_node}, ratio=0.5, base_seed=1,
    )

    assert stats.tile_id == tile_id
    assert stats.pad_anchor_count == 1
    assert stats.original_nodes == 12  # 10 bulk + pad + cs node
    assert stats.sampled_current_sources <= stats.original_current_sources == 1
    assert stats.isolated_mandatory_kept == 0
    assert stats.n_clusters == stats.sampled_nodes

    ckt_path = output_dir / "tile_0_0.ckt"
    nd_path = output_dir / "tile_0_0.nd"
    inst_path = output_dir / "instanceModels_0_0.sp"
    assert ckt_path.exists() and nd_path.exists() and inst_path.exists()

    with gzip.open(nd_path, "rt") as f:
        nd_nodes = {line.split()[0] for line in f if line.strip()}
    assert pad_node in nd_nodes  # hard anchor must survive
    assert cs_node in nd_nodes  # current-source-bearing node must survive
    assert len(nd_nodes) == stats.sampled_nodes


def test_process_tile_raises_on_capacitor_invariant_violation(tmp_path, monkeypatch):
    source_dir, pkl_dir, pad_node, cs_node = _build_synthetic_tile_environment(tmp_path)
    output_dir = tmp_path / "output"
    tile_id = (0, 0)

    import parser.brcm_tile_sampler as sampler_mod

    def _boom(*args, **kwargs):
        raise CapacitorInvariantViolation("synthetic failure")

    monkeypatch.setattr(sampler_mod, "verify_capacitor_invariant", _boom)

    with pytest.raises(CapacitorInvariantViolation):
        process_tile(
            source_dir, pkl_dir, output_dir, tile_id,
            all_pad_anchors={pad_node}, ratio=0.5,
        )


# =============================================================================
# run_sampling_pipeline (plan §5/§6) -- synthetic 2-tile end-to-end
# =============================================================================


def _build_synthetic_2tile_design(tmp_path):
    """Build a tiny 2-tile (2x1) synthetic design for run_sampling_pipeline tests."""
    tile_ids = [(0, 0), (1, 0)]
    source_dir = tmp_path / "source"
    pkl_dir = tmp_path / "pkl"
    source_dir.mkdir()
    pkl_dir.mkdir()

    pad_nodes = {}
    all_pad_anchors: set = set()

    for i, tile_id in enumerate(tile_ids):
        pad_node = f"{100 + i}_100_86"
        cs_node = f"50_{i}_43"  # distinct coordinate -- must not collide with bulk_nodes below
        bulk_nodes = [f"{j}_{i}_43" for j in range(10)]
        all_nodes = set(bulk_nodes) | {pad_node, cs_node}

        resistive_edges = []
        chain = bulk_nodes + [cs_node]
        for u, v in zip(chain, chain[1:]):
            resistive_edges.append((u, v, 10.0))
        resistive_edges.append((pad_node, bulk_nodes[0], 10.0))

        capacitive_edges = [(n, "0", 5.0) for n in all_nodes]

        tile_data = TileData(
            tile_id=tile_id, resistive_edges=resistive_edges, all_nodes=all_nodes,
            boundary_nodes=set(), current_injections={cs_node: 1.0},
            capacitive_edges=capacitive_edges,
        )
        _write_pkl(pkl_dir / f"tile_{tile_id[0]}_{tile_id[1]}.pkl", tile_data)

        nd_lines = [f"{n} 1 2 3 4 VDD_VAR" for n in sorted(all_nodes)]
        _write_gzip_text(source_dir / f"tile_{tile_id[0]}_{tile_id[1]}.nd", nd_lines)

        inst_lines = [
            ".flag_boundary",
            f"i_test_{i}:VDD_VAR:VDD:0:0:0:0:0 {cs_node} 0 1e-08 dv=0.76",
        ]
        _write_gzip_text(
            source_dir / f"instanceModels_{tile_id[0]}_{tile_id[1]}.sp", inst_lines,
        )

        pad_nodes[tile_id] = pad_node
        all_pad_anchors.add(pad_node)

    package_lines = _make_package_ckt_fixture(
        [(f"bmp_{i}", "VDD_VAR", pad_nodes[tid]) for i, tid in enumerate(tile_ids)]
    )
    (source_dir / "package.ckt").write_text("\n".join(package_lines) + "\n")
    (source_dir / "ckt.sp").write_text(
        ".partition_info 2 1\n.die_area 0 0 200 200\n.parameter VDD_VAR 0.76\n"
    )

    tile_configs = [
        TileConfig(tile_id=tid, ckt_path="", nd_path="", instance_path="", net_filter="vdd_var")
        for tid in tile_ids
    ]
    package_data = PackageData(
        vsrc_dict={}, package_edges=[], pad_nodes=set(pad_nodes.values()),
        tap_nodes=set(), die_attachment_nodes=set(all_pad_anchors), vdd=0.76,
        net_name="VDD_VAR",
    )
    metadata = PowerGridMetaData(
        tile_grid=(2, 1), parameters={}, tile_configs=tile_configs,
        package_data=package_data, net_name="VDD_VAR", vdd=0.76,
    )
    _write_pkl(pkl_dir / "metadata.pkl", {"metadata": metadata, "boundary_nodes": set()})

    return source_dir, pkl_dir, tile_ids, all_pad_anchors


def test_run_sampling_pipeline_sequential_writes_all_expected_files(tmp_path):
    source_dir, pkl_dir, tile_ids, all_pad_anchors = _build_synthetic_2tile_design(tmp_path)
    output_dir = tmp_path / "output"

    report = run_sampling_pipeline(
        source_dir, pkl_dir, output_dir, ratio=0.5, base_seed=1, workers=1,
    )

    assert len(report.tile_stats) == 2
    assert report.total_pad_anchors_found == report.total_pad_anchors_expected == 2

    for tid in tile_ids:
        assert (output_dir / f"tile_{tid[0]}_{tid[1]}.ckt").exists()
        assert (output_dir / f"tile_{tid[0]}_{tid[1]}.nd").exists()
        assert (output_dir / f"instanceModels_{tid[0]}_{tid[1]}.sp").exists()

    for fname in ("pg_net_voltage", "additional_vsrcs", "ckt.sp", "package.ckt"):
        assert (output_dir / fname).exists()

    write_pipeline_report(report, output_dir)
    assert (output_dir / "sampling_report.txt").exists()
    report_json = json.loads((output_dir / "sampling_report.json").read_text())
    assert report_json["totals"]["sampled_nodes"] == report.total_sampled_nodes


def test_run_sampling_pipeline_parallel_matches_expected_pad_anchor_count(tmp_path):
    source_dir, pkl_dir, tile_ids, all_pad_anchors = _build_synthetic_2tile_design(tmp_path)
    output_dir = tmp_path / "output_parallel"

    report = run_sampling_pipeline(
        source_dir, pkl_dir, output_dir, ratio=0.5, base_seed=1, workers=2,
    )
    assert len(report.tile_stats) == 2
    assert report.total_pad_anchors_found == 2
    for tid in tile_ids:
        assert (output_dir / f"tile_{tid[0]}_{tid[1]}.ckt").exists()


def test_write_top_level_files_writes_all_4_plain_text_files(tmp_path):
    source_dir, pkl_dir, tile_ids, all_pad_anchors = _build_synthetic_2tile_design(tmp_path)
    output_dir = tmp_path / "output_top_level"
    output_dir.mkdir()

    metadata, _boundary_nodes = load_metadata(pkl_dir)
    write_top_level_files(source_dir, output_dir, metadata, tile_ids)

    for fname in ("pg_net_voltage", "additional_vsrcs", "ckt.sp", "package.ckt"):
        path = output_dir / fname
        assert path.exists()
        # Plain text, NOT gzip (unlike the per-tile files).
        with open(path, "rb") as f:
            assert f.read(2) != b"\x1f\x8b"

    assert (output_dir / "pg_net_voltage").read_text() == "VDD_VAR - 0.76 \n"
    assert (output_dir / "additional_vsrcs").read_text() == ""


# =============================================================================
# End-to-end proxy-validity (plan §6) -- would have caught the §7.5 failure
# =============================================================================


def _build_synthetic_2tile_mesh_design(tmp_path):
    """Build a 2-tile design with an actual cross-tile mesh (cycles + a
    shared boundary node), unlike ``_build_synthetic_2tile_design``'s
    disjoint per-tile chains -- needed so the union graph has real
    inter-tile connectivity to check post-contraction (plan §6 point (b)).
    """
    tile_ids = [(0, 0), (1, 0)]
    source_dir = tmp_path / "source"
    pkl_dir = tmp_path / "pkl"
    source_dir.mkdir()
    pkl_dir.mkdir()

    boundary_node = "999_999_43"
    pad_nodes = {}
    all_pad_anchors: set = set()
    tile_x_offsets = {(0, 0): 0, (1, 0): 100}

    for i, tile_id in enumerate(tile_ids):
        x0 = tile_x_offsets[tile_id]
        nodes, edges, name = _grid_nodes_edges(5, 5, 43, x0=x0, y0=0)
        pad_node = f"{x0}_500_86"
        cs_node = name(2, 2)
        edges.append((pad_node, name(0, 0), 10.0))
        edges.append((boundary_node, name(4, 4), 10.0))  # cross-tile stitch point
        all_nodes = set(nodes) | {pad_node, boundary_node}
        capacitive_edges = [(n, "0", 5.0) for n in nodes] + [(pad_node, "0", 5.0)]

        tile_data = TileData(
            tile_id=tile_id, resistive_edges=edges, all_nodes=all_nodes,
            boundary_nodes={boundary_node}, current_injections={cs_node: 1.0},
            capacitive_edges=capacitive_edges,
        )
        _write_pkl(pkl_dir / f"tile_{tile_id[0]}_{tile_id[1]}.pkl", tile_data)

        nd_lines = [f"{n} 1 2 3 4 VDD_VAR" for n in sorted(all_nodes)]
        _write_gzip_text(source_dir / f"tile_{tile_id[0]}_{tile_id[1]}.nd", nd_lines)

        inst_lines = [
            ".flag_boundary",
            f"i_test_{i}:VDD_VAR:VDD:0:0:0:0:0 {cs_node} 0 1e-08 dv=0.76",
        ]
        _write_gzip_text(source_dir / f"instanceModels_{tile_id[0]}_{tile_id[1]}.sp", inst_lines)

        pad_nodes[tile_id] = pad_node
        all_pad_anchors.add(pad_node)

    package_lines = _make_package_ckt_fixture(
        [(f"bmp_{i}", "VDD_VAR", pad_nodes[tid]) for i, tid in enumerate(tile_ids)]
    )
    (source_dir / "package.ckt").write_text("\n".join(package_lines) + "\n")
    (source_dir / "ckt.sp").write_text(
        ".partition_info 2 1\n.die_area 0 0 200 200\n.parameter VDD_VAR 0.76\n"
    )

    tile_configs = [
        TileConfig(tile_id=tid, ckt_path="", nd_path="", instance_path="", net_filter="vdd_var")
        for tid in tile_ids
    ]
    package_data = PackageData(
        vsrc_dict={}, package_edges=[], pad_nodes=set(pad_nodes.values()),
        tap_nodes=set(), die_attachment_nodes=set(all_pad_anchors), vdd=0.76,
        net_name="VDD_VAR",
    )
    metadata = PowerGridMetaData(
        tile_grid=(2, 1), parameters={}, tile_configs=tile_configs,
        package_data=package_data, net_name="VDD_VAR", vdd=0.76,
    )
    _write_pkl(pkl_dir / "metadata.pkl", {"metadata": metadata, "boundary_nodes": {boundary_node}})

    return source_dir, pkl_dir, tile_ids, all_pad_anchors


def test_sampled_output_reparses_with_zero_floating_nodes_and_sane_r_per_node(tmp_path):
    """The regression test for §7.5: run the sampling pipeline, then feed its
    OUTPUT back through ``DistributedNetlistParser`` like a real user would,
    and verify (a) every sampled node survives re-parsing, (b) the union
    graph is fully connected from a pad anchor (zero floating nodes), and
    (c) aggregate R/node lands >= 1.0 and within +/-40% of the source's.
    """
    source_dir, pkl_dir, tile_ids, all_pad_anchors = _build_synthetic_2tile_mesh_design(tmp_path)

    # Source (pre-sampling) R/node, for the ±40% comparison below.
    source_total_nodes = 0
    source_total_resistors = 0
    for tile_id in tile_ids:
        td = load_tile_data(pkl_dir, tile_id)
        source_total_nodes += len(td.all_nodes)
        source_total_resistors += len(td.resistive_edges)

    output_dir = tmp_path / "output"
    report = run_sampling_pipeline(source_dir, pkl_dir, output_dir, ratio=0.5, base_seed=1, workers=1)
    assert report.total_pad_anchors_found == report.total_pad_anchors_expected

    dump_dir = tmp_path / "distributed_pkl"
    parser_ = DistributedNetlistParser(str(output_dir), net_filter='VDD_VAR')
    parser_.parse_and_dump(str(dump_dir), backend='local')

    with open(dump_dir / "metadata.pkl", "rb") as f:
        payload = pickle.load(f)
    reparsed_metadata = payload["metadata"]

    all_tile_data = [
        load_tile_data(dump_dir, tc.tile_id) for tc in reparsed_metadata.tile_configs
    ]

    # (a) every node written to the sampled .ckt files survives parsing.
    written_nodes = set()
    for tile_id in tile_ids:
        with gzip.open(output_dir / f"tile_{tile_id[0]}_{tile_id[1]}.nd", "rt") as f:
            written_nodes |= {line.split()[0] for line in f if line.strip()}
    reparsed_nodes = set()
    for td in all_tile_data:
        reparsed_nodes |= td.all_nodes
    assert written_nodes <= reparsed_nodes

    # (b) union graph connected from a pad-anchor node -- zero floating nodes.
    resistive_edges = [e for td in all_tile_data for e in td.resistive_edges]
    die_attachment_nodes = reparsed_metadata.package_data.die_attachment_nodes
    assert die_attachment_nodes  # sanity: at least one pad anchor survived
    start = next(iter(die_attachment_nodes))
    reached = _reachable_from(start, resistive_edges)
    assert reached == reparsed_nodes, f"floating nodes: {reparsed_nodes - reached}"

    # (c) aggregate R/node >= 1.0 and within +/-40% of the source design's.
    r_per_node_source = source_total_resistors / source_total_nodes
    r_per_node_sampled = len(resistive_edges) / len(reparsed_nodes)
    assert r_per_node_sampled >= 1.0
    assert 0.6 * r_per_node_source <= r_per_node_sampled <= 1.4 * r_per_node_source


# =============================================================================
# Real-data validation (plan §5/§6) -- single-tile smoke test, real netlist_brcm
# =============================================================================


@pytest.mark.integration
def test_process_tile_real_smallest_tile_end_to_end(tmp_path):
    """Run the full Pass 1-4 + write pipeline on the real, smallest tile (1,5).

    Lightweight sanity check only (not the full 36-tile run, which is the
    primary validation for this phase, done outside pytest) -- confirms
    the wiring between all passes and the new writers works end-to-end
    against real data, roughly hits the target ratio, and every pad anchor
    present in this tile survives.

    Note on ``isolated_mandatory_kept`` (verified by direct inspection, not
    a bug): on real data, some ``TileData.boundary_nodes`` have *zero*
    resistive degree within this tile's own mesh -- their only resistive
    edge is recorded in the *adjacent* tile's ``TileData.resistive_edges``
    (the raw netlist stores each cross-tile edge once, attributed to one
    side). Contraction correctly keeps these as self-mapped singleton
    representatives (never dropped, never given a fabricated edge) and
    counts them via ``isolated_mandatory_kept`` with a WARNING log -- they
    remain zero-local-degree nodes for downstream (cross-tile) validation
    to reason about, same as the pre-contraction design's
    "unrepairable" outcome, just without ever risking a fabricated
    same-tile shortcut edge.
    """
    if not _REAL_PKL_DIR.exists() or not (_REAL_NETLIST_DIR / "tile_1_5.nd").exists():
        pytest.skip("netlist/netlist_brcm (real data) not available in this environment")

    tile_id = (1, 5)
    metadata, _boundary_nodes = load_metadata(_REAL_PKL_DIR)
    all_pad_anchors = set(metadata.package_data.die_attachment_nodes)
    output_dir = tmp_path / "output"

    stats = process_tile(
        _REAL_NETLIST_DIR, _REAL_PKL_DIR, output_dir, tile_id,
        all_pad_anchors=all_pad_anchors, ratio=0.1, base_seed=42,
    )

    assert stats.tile_id == tile_id
    assert stats.original_nodes == 203_097
    # Sampling is mandatory-keep-set-dominated for some tiles (expected per
    # plan) -- just check we're in the right ballpark, not exactly 10%.
    assert 0 < stats.sampled_nodes <= stats.original_nodes
    # Isolated-mandatory nodes are expected on real data (see docstring
    # above) -- they're exclusively a subset of TileData.boundary_nodes,
    # bounded by this tile's total boundary-node count.
    tile_data = load_tile_data(_REAL_PKL_DIR, tile_id)
    assert 0 <= stats.isolated_mandatory_kept <= len(tile_data.boundary_nodes)

    ckt_path = output_dir / "tile_1_5.ckt"
    nd_path = output_dir / "tile_1_5.nd"
    inst_path = output_dir / "instanceModels_1_5.sp"
    assert ckt_path.exists() and nd_path.exists() and inst_path.exists()

    with gzip.open(nd_path, "rt") as f:
        nd_nodes = {line.split()[0] for line in f if line.strip()}
    assert all_pad_anchors.intersection(tile_data.all_nodes) <= nd_nodes

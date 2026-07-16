"""Phase 1-4 tooling for the netlist_brcm sampling pipeline.

Design doc: ``netlist_brcm_sampling_plan.md`` (repo root), see the
"REVISED" banner + §4.2/§4.3 for the algorithm this module implements, and
``docs/brcm_distributed_runtime_optimization.md`` §7.5 for why the
original node-drop + BFS-repair design was replaced.

- **Phase 1**: reading the existing ``distributed_pkl/`` output of
  `netlist/netlist_brcm` (a real 36-tile (6x6) production PDN netlist,
  already parsed via ``sigma-dvd parse ./netlist/netlist_brcm --net
  VDD_VAR``) and verifying the pad-anchor bookkeeping needed by later
  phases.
- **Pass 1** (plan §4.1): per-tile node classification (layer, boundary,
  pad-anchor, current-source-bearing) into a mandatory-keep set and an
  optional pool grouped by layer.
- **Pass 2** (plan §4.2, revised): ``contract_tile_nodes`` -- deterministic
  geometric contraction/coarsening of the optional pool. Nodes are grouped
  into small per-layer geometric cells, connected-component split within
  each cell (never merges disconnected mesh fragments), and each resulting
  cluster collapses to one representative node. Unlike the original
  node-drop design, contraction never discards a connected node without
  folding its edges into a surviving representative -- connectivity is
  preserved BY CONSTRUCTION (a quotient of a connected graph is connected).
- **Pass 3** (plan §4.3, revised): ``contract_tile_edges`` -- remaps every
  resistor/capacitor edge through Pass 2's node-to-representative map,
  merging parallel edges (conductances/capacitances add) and dropping
  intra-cluster edges. There is NO repair phase: contraction makes BFS
  connectivity repair unnecessary, replaced by a cheap mandatory-node
  degree sanity assertion (fail loud on a genuine contraction bug, not a
  data condition).
- **Pass 4** (plan §4.4, unchanged): current-source down-sampling from the
  raw ``instanceModels_X_Y.sp`` text (preserving original line text
  verbatim, unlike ``scan_current_source_nodes``'s node-only pre-scan),
  plus the capacitor-follows-current-source invariant check. ``base_seed``
  now only affects this pass -- Passes 2/3 are fully deterministic (no RNG).

This module also provides the full pipeline orchestration/CLI
(``process_tile``, ``run_sampling_pipeline``, ``main``) that stitches all
of the above -- plus the output-file writers below -- into a real
``netlist/netlist_brcm_sampled/`` directory:

- Loaders for ``metadata.pkl`` / ``tile_X_Y.pkl`` (thin wrappers, no
  redefinition of ``PowerGridMetaData`` / ``TileData``).
- Tile-id discovery from the parsed metadata.
- A pad-anchor accounting pass: for each tile, intersect its node set with
  the design-wide set of 309 ``die_attachment_node`` hard anchors (the
  real coordinate nodes each of the 309 pads is zero-ohm-anchored to) and
  verify every anchor is accounted for exactly once across all tiles.
- ``scan_current_source_nodes``: a raw-text pre-scan of a tile's
  ``instanceModels_X_Y.sp`` for VDD_VAR current-source-bearing nodes
  (deliberately NOT using ``TileData.current_injections``, which is a
  flattened DC-only scalar with no instance-name/waveform fidelity).
- ``classify_tile``: per-tile node classification (layer/boundary/
  pad-anchor/current-source-bearing, mandatory-keep set, optional pool by
  layer), building a plain-dict adjacency structure reused by
  ``contract_tile_nodes``'s connected-component split.
- ``sample_current_sources`` / ``verify_capacitor_invariant``: Pass 4
  current-source down-sampling (raw-text-preserving) and the
  capacitor-follows-current-source correctness invariant.
- **Output generation** (plan §4.5): ``generate_pg_net_voltage``,
  ``generate_additional_vsrcs``, ``generate_ckt_sp`` (+
  ``read_die_area_from_ckt_sp``), ``filter_package_ckt`` (top-level,
  tile-count-independent files), and the per-tile writers
  (``write_tile_ckt``, ``write_tile_nd``, ``write_instance_models``).
"""

from __future__ import annotations

import argparse
import contextlib
import gzip
import io
import json
import logging
import math
import multiprocessing
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
import pickle
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from distributed.parser import PackageData, PowerGridMetaData, TileConfig
from distributed.tile_parsing import (
    C_TO_FF,
    R_TO_KOHM,
    TileData,
    _is_gzip_file,
    _iter_instance_sources,
    _load_nd_file,
    _parse_node_xy,
)

logger = logging.getLogger(__name__)

DEFAULT_PKL_DIR = Path('netlist/netlist_brcm/distributed_pkl')


def load_metadata(pkl_dir: Path) -> Tuple[PowerGridMetaData, Set[str]]:
    """Load ``metadata.pkl`` from a distributed_pkl directory.

    Args:
        pkl_dir: Directory containing ``metadata.pkl`` (as produced by
            ``DistributedNetlistParser.parse_and_dump`` / ``sigma-dvd parse``).

    Returns:
        ``(metadata, boundary_nodes)`` — the parsed ``PowerGridMetaData``
        and the global set of shared (2+ tile) boundary node names.
    """
    meta_path = Path(pkl_dir) / 'metadata.pkl'
    logger.debug("Loading metadata from %s", meta_path)
    with open(meta_path, 'rb') as f:
        payload = pickle.load(f)
    metadata: PowerGridMetaData = payload['metadata']
    boundary_nodes: Set[str] = payload['boundary_nodes']
    logger.info(
        "Loaded metadata: net=%s vdd=%.3g tile_grid=%s tiles=%d boundary_nodes=%d",
        metadata.net_name, metadata.vdd, metadata.tile_grid,
        len(metadata.tile_configs), len(boundary_nodes),
    )
    return metadata, boundary_nodes


def load_tile_data(pkl_dir: Path, tile_id: Tuple[int, int]) -> TileData:
    """Load a single tile's ``TileData`` from ``tile_X_Y.pkl``.

    Args:
        pkl_dir: Directory containing the per-tile pkl files.
        tile_id: ``(x, y)`` tile coordinate, matching the ``tile_X_Y.pkl``
            filename convention used by ``DistributedNetlistParser``.

    Returns:
        The unpickled ``TileData`` object (unlike ``metadata.pkl``, tile
        pkls unpickle directly to the dataclass, not a wrapper dict).
    """
    tile_path = Path(pkl_dir) / f"tile_{tile_id[0]}_{tile_id[1]}.pkl"
    logger.debug("Loading tile data from %s", tile_path)
    with open(tile_path, 'rb') as f:
        tile_data: TileData = pickle.load(f)
    return tile_data


def discover_tile_ids(metadata: PowerGridMetaData) -> List[Tuple[int, int]]:
    """Return the sorted list of tile ids present in *metadata*."""
    tile_configs: List[TileConfig] = metadata.tile_configs
    return sorted(tc.tile_id for tc in tile_configs)


def _pad_anchors_in_tile(tile_data: TileData, all_anchors: Set[str]) -> Set[str]:
    """Return the subset of *all_anchors* (design-wide pad-anchor nodes) in *tile_data*.

    Trivial set-intersection helper, extracted so it has one home shared by
    ``compute_pad_anchor_accounting`` (Phase 1 accounting loop) and
    ``classify_tile`` (Pass 1 node classification) — both need exactly this
    "which of the 309 die_attachment_nodes live in this tile" computation.
    """
    return tile_data.all_nodes & all_anchors


def compute_pad_anchor_accounting(
    pkl_dir: Path, metadata: PowerGridMetaData
) -> Dict[Tuple[int, int], Set[str]]:
    """Compute each tile's pad-anchor (``die_attachment_nodes``) keep-set.

    For each tile, loads its ``TileData`` and intersects ``all_nodes``
    with the design-wide ``package_data.die_attachment_nodes`` set (309
    hard-anchor nodes that each pad's zero-ohm ``rs`` line lands on).
    There is no per-tile geometric bounds field to consult instead — tiles
    are not guaranteed to be uniform rectangles at the node level, so
    set-intersection against the loaded mesh is the only reliable way to
    classify which tile a die_attachment_node belongs to (and is exactly
    what ``classify_tile``'s mandatory-keep pass needs as its
    ``pad_anchors_for_tile`` argument).

    Verifies (logging at INFO/WARNING, does not raise) that:
      (a) the union of all per-tile sets covers every die_attachment_node
          exactly (matches ``len(die_attachment_nodes)``, expected 309),
      (b) no die_attachment_node is claimed by more than one tile,
      (c) no die_attachment_node is missing from every tile (would mean an
          unreachable/orphaned pad — a serious problem).

    Returns:
        Mapping of tile_id -> set of die_attachment_nodes found in that
        tile. Returned regardless of verification outcome so later phases
        can use it directly as each tile's mandatory pad-anchor keep-set.
    """
    package_data: PackageData = metadata.package_data
    all_anchors = package_data.die_attachment_nodes
    tile_ids = discover_tile_ids(metadata)

    per_tile_anchors: Dict[Tuple[int, int], Set[str]] = {}
    anchor_to_tiles: Dict[str, List[Tuple[int, int]]] = {}

    for tile_id in tile_ids:
        tile_data = load_tile_data(pkl_dir, tile_id)
        anchors_in_tile = _pad_anchors_in_tile(tile_data, all_anchors)
        per_tile_anchors[tile_id] = anchors_in_tile
        for node in anchors_in_tile:
            anchor_to_tiles.setdefault(node, []).append(tile_id)
        logger.info(
            "Tile %s: %d nodes, %d pad anchors", tile_id,
            len(tile_data.all_nodes), len(anchors_in_tile),
        )

    union_anchors = set(anchor_to_tiles)
    duplicated = {n: t for n, t in anchor_to_tiles.items() if len(t) > 1}
    missing = all_anchors - union_anchors

    if len(union_anchors) == len(all_anchors) and not duplicated and not missing:
        logger.info(
            "Pad-anchor accounting OK: all %d die_attachment_nodes accounted "
            "for exactly once across %d tiles", len(all_anchors), len(tile_ids),
        )
    else:
        logger.warning(
            "Pad-anchor accounting mismatch: expected %d die_attachment_nodes, "
            "union covers %d", len(all_anchors), len(union_anchors),
        )

    if duplicated:
        logger.warning(
            "%d die_attachment_nodes claimed by more than one tile: %s",
            len(duplicated), duplicated,
        )
    if missing:
        logger.warning(
            "%d die_attachment_nodes missing from every tile (orphaned pad "
            "anchor): %s", len(missing), sorted(missing),
        )

    return per_tile_anchors


@dataclass
class PadAnchorSummary:
    """Human-readable rollup of a pad-anchor accounting pass."""

    total_anchors: int
    total_tiles: int
    per_tile_counts: Dict[Tuple[int, int], int]

    @property
    def min_count(self) -> int:
        return min(self.per_tile_counts.values()) if self.per_tile_counts else 0

    @property
    def max_count(self) -> int:
        return max(self.per_tile_counts.values()) if self.per_tile_counts else 0

    @property
    def total_count(self) -> int:
        return sum(self.per_tile_counts.values())


def _summarize(
    metadata: PowerGridMetaData, per_tile_anchors: Dict[Tuple[int, int], Set[str]]
) -> PadAnchorSummary:
    return PadAnchorSummary(
        total_anchors=len(metadata.package_data.die_attachment_nodes),
        total_tiles=len(per_tile_anchors),
        per_tile_counts={tid: len(nodes) for tid, nodes in per_tile_anchors.items()},
    )


def _print_summary(summary: PadAnchorSummary) -> None:
    print(f"Total die_attachment_nodes (design-wide): {summary.total_anchors}")
    print(f"Total tiles: {summary.total_tiles}")
    print("Per-tile pad-anchor counts:")
    for tile_id in sorted(summary.per_tile_counts):
        count = summary.per_tile_counts[tile_id]
        flag = "  <-- ZERO anchors" if count == 0 else ""
        print(f"  {tile_id}: {count}{flag}")
    print(
        f"min={summary.min_count} max={summary.max_count} "
        f"total={summary.total_count} (expected {summary.total_anchors})"
    )


# =============================================================================
# Pass 1 — per-tile node classification (plan §4.1)
# =============================================================================


def _parse_node_layer(node: str) -> Optional[int]:
    """Extract the integer layer suffix from a PDN node name ``<x>_<y>_<layer>``.

    Companion to ``distributed.tile_parsing._parse_node_xy`` (which extracts
    ``(x, y)`` but deliberately ignores the layer suffix). Verified against
    real ``netlist_brcm`` data (tiles ``(1, 5)`` and ``(2, 3)``, the
    smallest and largest tiles): every node name splits into exactly 3
    ``'_'``-separated parts and the last part is always integer-parseable
    (e.g. ``'1197000_449800_86'`` -> layer ``86``; layers observed ranged
    43-86).

    Returns ``None`` for malformed names (defensive only — not expected to
    trigger on real data; a node with an unparseable layer is simply
    grouped under the ``None`` "layer" key by ``classify_tile`` rather than
    raising).
    """
    parts = str(node).split('_')
    if len(parts) < 2:
        return None
    try:
        return int(parts[-1])
    except ValueError:
        return None


def scan_current_source_nodes(
    instance_path: Optional[str],
    nd_path: Optional[str] = None,
    net_filter: str = 'vdd_var',
) -> Set[str]:
    """Pre-scan a tile's raw ``instanceModels_X_Y.sp`` for current-source-bearing nodes.

    Per the plan (§4.1/§4.4), current-source classification must come from
    the RAW text, not ``TileData.current_injections`` (a flattened,
    per-node DC-only scalar with no instance-name/waveform fidelity — fine
    for solving, not enough for later down-sampling/regeneration).

    Reuses the production ``distributed.tile_parsing._iter_instance_sources``
    helper, which already handles gzip detection, comment/dot-line
    skipping, the fast structured-instance-name net filter (falling back to
    slow ``.nd``-based filtering for unstructured names), and
    ``parser.current_sources._prepare_instance_source`` parsing — so
    filtering semantics exactly match what ``sigma-dvd parse --net
    VDD_VAR`` uses, and there's no hand-rolled substring-matching net
    filter to get subtly wrong.

    Verified against real data (``instanceModels_1_5.sp`` and
    ``instanceModels_2_3.sp``, the smallest and largest tiles):

    - Instance names are structured
      (``i_<inst>:<net1>:<pin1>:<net2>:<pin2>:<tx>:<ty>[:extra]``, >= 7
      colon-delimited fields), so the fast filter path is used.
    - The net filter comparison is against a *lowercase* net name — matches
      ``TileConfig.net_filter`` as set by ``DistributedNetlistParser``
      (``net_filter.lower()``), confirmed directly from
      ``distributed_pkl/metadata.pkl``: ``tile_configs[0].net_filter ==
      'vdd_var'``. Hence this function's default is ``'vdd_var'``, not
      ``'VDD_VAR'``.
    - Every VDD_VAR current-source line's negative terminal (``node_neg``)
      observed in both tiles was the literal ground node ``'0'``. This
      function still defensively adds ``node_neg`` to the result when it is
      *not* ``'0'``, in case some other tile/design has a real die node
      there (per the ticket's explicit ask — not proven impossible, just
      not observed).

    Args:
        instance_path: Path to ``instanceModels_X_Y.sp`` (gzip or plain
            text, either extension). Returns an empty set when ``None``
            (mirrors ``_iter_instance_sources``'s own ``None`` handling —
            a tile with no instance-models file simply has no
            current-source-bearing nodes).
        nd_path: Optional ``.nd`` path, consulted only as a fallback when
            instance names are not structured (see
            ``_iter_instance_sources``).
        net_filter: Lowercase net name to match (default ``'vdd_var'`` —
            the only net this whole sampling pipeline operates on).

    Returns:
        Set of node names (boundary ``*`` prefix already stripped by
        ``_prepare_instance_source``) with >= 1 matching current source
        attached at either terminal.
    """
    nodes: Set[str] = set()
    for prepared in _iter_instance_sources(instance_path, net_filter, nd_path):
        if prepared.node_pos != '0':
            nodes.add(prepared.node_pos)
        if prepared.node_neg != '0':
            nodes.add(prepared.node_neg)
    return nodes


@dataclass
class TileClassification:
    """Per-tile node classification (Pass 1, plan §4.1).

    ``adjacency`` is built once here specifically so Pass 2
    (``contract_tile_nodes``, which restricts cell-splitting BFS/union-find
    to this same adjacency dict) and the mandatory-degree sanity check in
    Pass 3 (``contract_tile_edges``) can reuse it directly rather than
    re-scanning ``resistive_edges``.

    The pad-anchor / boundary / current-source subsets are kept alongside
    the ``mandatory_keep`` union (rather than only exposing the union)
    because later phases need to distinguish *why* a node is mandatory —
    e.g. the "capacitor-follows-current-source" invariant (plan §4.4) only
    cares about ``current_source_nodes`` specifically, not the full
    mandatory-keep set.
    """

    tile_id: Tuple[int, int]
    total_nodes: int
    layer_of: Dict[str, Optional[int]]
    adjacency: Dict[str, List[Tuple[str, float]]]  # node -> [(neighbor, conductance_mS), ...]
    mandatory_keep: Set[str]
    optional_pool_by_layer: Dict[Optional[int], Set[str]]
    pad_anchor_nodes: Set[str]
    boundary_nodes: Set[str]
    current_source_nodes: Set[str]


def classify_tile(
    tile_data: TileData,
    pad_anchors_for_tile: Set[str],
    current_source_nodes_for_tile: Set[str],
) -> TileClassification:
    """Classify every node in *tile_data* (Pass 1, plan §4.1).

    Builds a plain-dict adjacency structure from ``resistive_edges`` (kept
    on the returned ``TileClassification`` for reuse by Pass 2's
    connected-component split and Pass 3's mandatory-degree sanity check,
    not a throwaway local) and each node's layer. (Via/non-via
    classification was removed together with the BFS repair machinery it
    fed -- layer-stratified contraction never consults via-ness.)

    Args:
        tile_data: Loaded ``TileData`` for one tile.
        pad_anchors_for_tile: This tile's pad-anchor (``die_attachment_node``)
            keep-set, e.g. from ``_pad_anchors_in_tile(tile_data, all_anchors)``
            or ``compute_pad_anchor_accounting(...)[tile_id]``.
        current_source_nodes_for_tile: This tile's current-source-bearing
            node set, from ``scan_current_source_nodes(...)`` run against
            the tile's raw ``instanceModels_X_Y.sp`` (NOT
            ``tile_data.current_injections`` — see module docstring).

        Both keep-set arguments are intersected with ``tile_data.all_nodes``
        defensively, so callers may pass design-wide/whole-file sets
        without pre-filtering them to this tile.

    Returns:
        A ``TileClassification`` with layer/mandatory/optional-pool
        breakdowns.
    """
    all_nodes = tile_data.all_nodes

    pad_anchor_nodes = pad_anchors_for_tile & all_nodes
    boundary_nodes = tile_data.boundary_nodes & all_nodes
    current_source_nodes = current_source_nodes_for_tile & all_nodes

    layer_of: Dict[str, Optional[int]] = {node: _parse_node_layer(node) for node in all_nodes}

    adjacency: Dict[str, List[Tuple[str, float]]] = {node: [] for node in all_nodes}

    for u, v, g in tile_data.resistive_edges:
        # Both endpoints of every resistive edge are always members of
        # all_nodes (see _parse_tile_ckt), but setdefault is used
        # defensively rather than assuming that invariant unconditionally.
        adjacency.setdefault(u, []).append((v, g))
        adjacency.setdefault(v, []).append((u, g))

    mandatory_keep = pad_anchor_nodes | boundary_nodes | current_source_nodes

    optional_pool_by_layer: Dict[Optional[int], Set[str]] = {}
    for node in all_nodes - mandatory_keep:
        optional_pool_by_layer.setdefault(layer_of[node], set()).add(node)

    logger.info(
        "Tile %s classified: %d nodes, mandatory_keep=%d "
        "(pad_anchor=%d boundary=%d current_source=%d), optional_pool=%d "
        "nodes across %d layers",
        tile_data.tile_id, len(all_nodes), len(mandatory_keep),
        len(pad_anchor_nodes), len(boundary_nodes), len(current_source_nodes),
        len(all_nodes) - len(mandatory_keep), len(optional_pool_by_layer),
    )

    return TileClassification(
        tile_id=tile_data.tile_id,
        total_nodes=len(all_nodes),
        layer_of=layer_of,
        adjacency=adjacency,
        mandatory_keep=mandatory_keep,
        optional_pool_by_layer=optional_pool_by_layer,
        pad_anchor_nodes=pad_anchor_nodes,
        boundary_nodes=boundary_nodes,
        current_source_nodes=current_source_nodes,
    )


# =============================================================================
# Pass 2 — geometric contraction/coarsening (plan §4.2, revised)
# =============================================================================
#
# Replaces the original node-drop layer-stratified sampling (see the §7.5
# failure analysis in the module docstring). Instead of independently
# keep/drop-ing each optional node (which starves an edge of survival to
# P ~= retention^2 for optional-optional pairs), every optional node is
# assigned to a small connected cluster and the cluster collapses to one
# representative -- no node is ever discarded without its edges being
# folded into a surviving representative, so connectivity is preserved BY
# CONSTRUCTION. Fully deterministic: no ``random`` use anywhere below.


GROUND_NODE = '0'

_SINGLETON_CELL_EPS = 1e-9  # bbox-area floor to avoid a division by zero

# Mirror of distributed.tile_parsing._parse_tile_ckt's SHORT_THRESHOLD
# (function-local there, not importable): raw resistances below this many
# kOhm are clamped to the GMAX sentinel on (re)parse.
_REPARSE_SHORT_THRESHOLD_KOHM = 1e-6


@dataclass
class TileContractionResult:
    """Result of Pass 2 geometric contraction for one tile.

    ``node_to_rep`` is the primary output consumed by Pass 3
    (``contract_tile_edges``): every original node that was not dropped as
    an isolated optional node maps to its cluster's representative
    (mandatory and singleton-representative nodes map to themselves).
    ``mandatory_nodes`` is carried through so Pass 3's degree sanity check
    doesn't need a second ``classify_tile`` call.
    """

    tile_id: Tuple[int, int]
    kept_nodes: Set[str]
    node_to_rep: Dict[str, str]
    mandatory_nodes: Set[str]
    # Mandatory nodes with >= 1 non-self, non-ground resistive neighbor --
    # the exact scope of Pass 3's fail-loud degree sanity check.
    mandatory_connected_nodes: Set[str]
    target_kept: int
    mandatory_kept: int
    optional_kept: int
    # ACHIEVED per-layer retention (clusters / active optional nodes), which
    # can differ from the solved budget target on stripe-shaped layers.
    per_layer_retention: Dict[Optional[int], float]
    n_clusters: int
    isolated_optional_dropped: int
    isolated_mandatory_kept: int


def _achieved_optional_sum(
    optional_sizes: Dict[Any, int], weights: Dict[Any, float], scale: float
) -> float:
    """``sum_L( min(1.0, w_L * scale) * n_L )`` — monotonic non-decreasing in *scale*.

    Summed in sorted layer order: ``optional_sizes`` inherits dict/set
    iteration order (PYTHONHASHSEED-dependent across processes), and the
    bisected scale feeds cell assignment where a last-ulp difference can
    flip a node across a cell boundary — the float sum order must be
    canonical for the cross-process byte-identical-output contract.
    """
    return sum(
        min(1.0, weights[layer] * scale) * optional_sizes[layer]
        for layer in sorted(optional_sizes, key=lambda k: -1 if k is None else k)
    )


def _solve_per_layer_retention(
    optional_sizes: Dict[Optional[int], int], remaining: int, alpha: float,
) -> Dict[Optional[int], float]:
    """Solve for per-layer retention fractions via the plan §4.2 weight/bisection scheme.

    Identical math to the original ``sample_tile_nodes``: weight
    ``w_L = n_L ** (-alpha)``, single scale factor ``s`` via exponential
    search + bisection so ``sum_L( min(1.0, w_L * s) * n_L ) ~= remaining``.
    """
    total_optional = sum(optional_sizes.values())
    if total_optional == 0 or remaining >= total_optional:
        return {layer: 1.0 for layer in optional_sizes}

    weights = {layer: n ** (-alpha) for layer, n in optional_sizes.items()}

    lo, hi = 0.0, 1.0
    # Exponential search for an upper bound at which the achieved sum
    # reaches `remaining`. Guaranteed to terminate: remaining <
    # total_optional was just checked above, and
    # _achieved_optional_sum(scale) -> total_optional monotonically as
    # scale -> inf. The 1e12 cap is just a safety net against pathological
    # inputs (e.g. all-zero weights), not expected to be hit.
    while (
        _achieved_optional_sum(optional_sizes, weights, hi) < remaining
        and hi < 1e12
    ):
        hi *= 2.0

    mid = hi
    for _ in range(100):
        mid = (lo + hi) / 2.0
        achieved = _achieved_optional_sum(optional_sizes, weights, mid)
        if abs(achieved - remaining) <= 1.0:
            break
        if achieved < remaining:
            lo = mid
        else:
            hi = mid

    return {layer: min(1.0, weights[layer] * mid) for layer in optional_sizes}


def _assign_geometric_cells(
    nodes: Set[str], xy_of: Dict[str, Tuple[Optional[float], Optional[float]]], retention: float,
) -> Dict[str, Tuple]:
    """Assign each node a hashable cell id for the CC-split step (plan §4.2, point 2).

    Unparseable-xy nodes get a per-node-unique cell key so they always end
    up as singleton clusters after the CC split (never grouped with any
    other node, parseable or not) -- this directly implements "each becomes
    a singleton cluster" without a separate code path.

    Cell edge length targets an expected cluster size of ``1 / retention``:
    ``h = sqrt(bbox_area / (n_parseable * retention))``. Degenerate bboxes
    (zero width and/or height -- collinear or coincident nodes) fall back
    to a 1-D or single-cell assignment along whichever axis actually varies.
    """
    cell_of: Dict[str, Tuple] = {}
    parseable = []
    for n in nodes:
        x, y = xy_of[n]
        if x is None or y is None:
            cell_of[n] = ('unparseable', n)
        else:
            parseable.append(n)

    if not parseable:
        return cell_of

    xs = [xy_of[n][0] for n in parseable]
    ys = [xy_of[n][1] for n in parseable]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    width, height = x1 - x0, y1 - y0
    n_parseable = len(parseable)
    denom = max(n_parseable * retention, 1.0)

    if width > 0 and height > 0:
        h = math.sqrt(max(width * height, _SINGLETON_CELL_EPS) / denom)
        for n in parseable:
            x, y = xy_of[n]
            cell_of[n] = (math.floor((x - x0) / h), math.floor((y - y0) / h))
    elif width > 0 or height > 0:
        axis, extent, origin = (0, width, x0) if width > 0 else (1, height, y0)
        cell_len = extent / denom
        for n in parseable:
            cell_of[n] = (math.floor((xy_of[n][axis] - origin) / cell_len),)
    else:
        # Single point (all parseable nodes share (x, y)) -- one shared cell.
        for n in parseable:
            cell_of[n] = (0,)

    return cell_of


def _connected_component_split(
    nodes: Set[str], cell_of: Dict[str, Tuple], adjacency: Dict[str, List[Tuple[str, float]]],
) -> List[List[str]]:
    """Union-find over *nodes*, unioning only same-cell neighbors (plan §4.2, point 3).

    Restricting unions to ``cell_of[u] == cell_of[v]`` (in addition to both
    endpoints being members of *nodes*) is what prevents two disconnected
    mesh fragments that happen to land in the same geometric cell from
    being spuriously merged.
    """
    parent = {n: n for n in nodes}

    def find(n: str) -> str:
        while parent[n] != n:
            parent[n] = parent[parent[n]]
            n = parent[n]
        return n

    for u in nodes:
        cell_u = cell_of[u]
        for v, _g in adjacency.get(u, ()):
            if v in nodes and cell_of[v] == cell_u:
                ru, rv = find(u), find(v)
                if ru != rv:
                    parent[ru] = rv

    clusters: Dict[str, List[str]] = {}
    for n in nodes:
        clusters.setdefault(find(n), []).append(n)
    return list(clusters.values())


_RETENTION_FIT_MAX_ITERS = 5
_RETENTION_FIT_TOLERANCE = 1.2  # accept achieved/target cluster count within [1/1.2, 1.2]


def _cluster_layer_to_target(
    active_nodes: Set[str],
    xy_of: Dict[str, Tuple[Optional[float], Optional[float]]],
    adjacency: Dict[str, List[Tuple[str, float]]],
    r_L: float,
) -> List[List[str]]:
    """Cluster one layer's active optional nodes to ~``round(r_L * n)`` clusters.

    A single geometric pass systematically overshoots on stripe-shaped PDN
    layers: clusters = cells x within-cell connected components, and
    parallel stripes that only interconnect through OTHER layers split
    every h-by-h cell into one cluster per stripe segment, so achieved
    retention lands near sqrt(r_L) instead of r_L (~3x too many nodes at
    r_L = 0.1). The achieved count is monotone in the cell-count parameter
    (clusters ~ cells^gamma, gamma in (0, 1]), so a multiplicative
    fixed-point update on the retention parameter (error exponent
    ``1 - gamma`` per iteration) converges in a few steps for both the
    stripe (gamma ~= 0.5) and 2-D-mesh (gamma ~= 1) regimes.
    """
    target = max(1, round(r_L * len(active_nodes)))
    r_eff = r_L
    best: Optional[List[List[str]]] = None
    prev_achieved: Optional[int] = None
    for _ in range(_RETENTION_FIT_MAX_ITERS):
        cell_of = _assign_geometric_cells(active_nodes, xy_of, r_eff)
        clusters = _connected_component_split(active_nodes, cell_of, adjacency)
        achieved = len(clusters)
        if best is None or abs(achieved - target) < abs(len(best) - target):
            best = clusters
        err = achieved / target
        if 1.0 / _RETENTION_FIT_TOLERANCE <= err <= _RETENTION_FIT_TOLERANCE:
            break
        if achieved == prev_achieved:
            # Floor/ceiling reached (e.g. target below the layer's intrinsic
            # connected-component count) -- further rescaling cannot help.
            break
        prev_achieved = achieved
        r_eff = min(1.0, r_eff / err)
    return best if best is not None else []


def _pick_cluster_representative(
    members: List[str], xy_of: Dict[str, Tuple[Optional[float], Optional[float]]],
) -> str:
    """Deterministic representative: nearest to the coordinate centroid, tie-broken by name.

    Falls back to the lexicographically smallest name when no member has a
    parseable coordinate (plan §4.2, point 4) -- not expected to trigger
    given ``_assign_geometric_cells``'s singleton treatment of unparseable
    nodes, but a cluster is a plain list here, not guaranteed coordinate-full.
    """
    if len(members) == 1:
        return members[0]

    # Sorted so the float centroid sum has a canonical order: cluster member
    # lists inherit set-iteration order (PYTHONHASHSEED-dependent across
    # processes), and float addition is only order-independent when the
    # coordinates happen to be integer-valued.
    coord_members = sorted(m for m in members if xy_of.get(m, (None, None))[0] is not None)
    if not coord_members:
        return min(members)

    cx = sum(xy_of[m][0] for m in coord_members) / len(coord_members)
    cy = sum(xy_of[m][1] for m in coord_members) / len(coord_members)

    def _key(m: str) -> Tuple[float, str]:
        x, y = xy_of.get(m, (None, None))
        if x is None:
            return (float('inf'), m)
        return ((x - cx) ** 2 + (y - cy) ** 2, m)

    return min(members, key=_key)


def contract_tile_nodes(
    classification: TileClassification,
    ratio: float = 0.1,
    alpha: float = 1.0,
) -> TileContractionResult:
    """Geometric contraction of the optional pool for one tile (Pass 2, plan §4.2).

    1. Per-layer retention targets ``r_L`` -- identical budget math to the
       original sampling design (see ``_solve_per_layer_retention``):
       ``target_kept = round(ratio * total_nodes)``. If the mandatory-keep
       set alone already meets/exceeds ``target_kept``, a warning is logged
       (this tile's reduction will be worse than the target ratio), but the
       optional pool still runs through the SAME clustering machinery below
       with ``remaining = 0`` -- NOT a bare "drop every optional node"
       special case. A hand-picked rescue of only the DIRECT resistive
       neighbors of mandatory nodes is not sufficient: two mandatory nodes
       joined solely by a chain of optional nodes of length >= 2
       (M1-o1-o2-o3-M2) would keep their end-stub edges but lose the
       interior edge, silently fragmenting the tile into disconnected
       islands even though every individual mandatory node still shows
       degree > 0. Feeding ``remaining = 0`` into the normal per-layer
       clustering keeps every non-isolated optional node as a member of
       SOME cluster (never simply dropped), so the "quotient of a connected
       graph is connected" guarantee below applies uniformly, degenerate
       branch included.
       ``r_L`` is "fraction of layer L's optional nodes that remain as
       cluster representatives", i.e. expected cluster size on layer L is
       ``~= 1 / r_L``.
    2. Nodes with zero resistive edges in ``classification.adjacency``
       ("originally isolated") are handled specially, never clustered:
       optional isolates are dropped entirely (counted, not repaired --
       the production parser would island-drop them anyway); mandatory
       isolates are kept as self-mapped singletons (counted + one WARNING
       if any -- never given a fabricated edge).
    3. Every other optional node is assigned a geometric cell
       (``_assign_geometric_cells``, skipped -- singleton per node -- when
       ``r_L >= 1.0``) then connected-component split within its
       (layer, cell) bucket (``_connected_component_split``) so two
       disconnected mesh fragments sharing a cell never merge.
    4. Each cluster collapses to one representative
       (``_pick_cluster_representative``). Mandatory nodes are never part
       of this process -- always singleton representatives, never absorbed
       and never absorbing.

    Args:
        classification: Output of ``classify_tile`` for one tile.
        ratio: Overall target retention ratio (default 0.1, i.e. ~10x
            reduction).
        alpha: Layer-weight exponent (default 1.0). ``alpha = 0`` degenerates
            to flat/uniform-across-layers retention.

    Returns:
        A ``TileContractionResult`` (see above).
    """
    tile_id = classification.tile_id
    total_nodes = classification.total_nodes
    mandatory_keep = classification.mandatory_keep
    optional_pool_by_layer = classification.optional_pool_by_layer
    adjacency = classification.adjacency

    def _degree(n: str) -> int:
        return len(adjacency.get(n, ()))

    isolated_mandatory = {n for n in mandatory_keep if _degree(n) == 0}
    if isolated_mandatory:
        logger.warning(
            "Tile %s: %d mandatory-keep node(s) have zero resistive edges "
            "(isolated) -- kept as singleton representatives per plan, no "
            "fabricated edge added: %s%s",
            tile_id, len(isolated_mandatory), sorted(isolated_mandatory)[:10],
            "..." if len(isolated_mandatory) > 10 else "",
        )

    target_kept = round(ratio * total_nodes)
    node_to_rep: Dict[str, str] = {n: n for n in mandatory_keep}

    if len(mandatory_keep) >= target_kept:
        logger.warning(
            "Tile %s: mandatory-keep set (%d nodes: pad_anchor=%d "
            "boundary=%d current_source=%d) already meets or exceeds the "
            "target kept-node count (%d = ratio %.3g of %d original nodes) "
            "-- this tile's reduction will be less than the target ratio. "
            "The optional pool is still contracted (never bulk-dropped) so "
            "connectivity between mandatory nodes joined only by optional "
            "chains is preserved.",
            tile_id, len(mandatory_keep), len(classification.pad_anchor_nodes),
            len(classification.boundary_nodes), len(classification.current_source_nodes),
            target_kept, ratio, total_nodes,
        )
        # Do NOT special-case "keep mandatory only, rescue direct optional
        # neighbors": that drops every non-rescued optional node outright,
        # and two mandatory nodes connected solely through a chain of
        # optional nodes of length >= 2 would lose their interior edge and
        # fragment into disconnected islands (the exact §7.5 failure mode)
        # even though each mandatory endpoint still shows degree > 0.
        # Falling through to the normal clustering path below with
        # remaining = 0 keeps every non-isolated optional node in some
        # cluster (never dropped), so connectivity is preserved by
        # construction here too.

    remaining = max(0, target_kept - len(mandatory_keep))
    optional_sizes = {layer: len(pool) for layer, pool in optional_pool_by_layer.items()}
    retention = _solve_per_layer_retention(optional_sizes, remaining, alpha)

    isolated_optional_dropped = 0
    n_clusters_optional = 0
    achieved_retention: Dict[Optional[int], float] = {}

    for layer, pool in optional_pool_by_layer.items():
        r_L = retention.get(layer, 0.0)

        active_nodes = set()
        for n in pool:
            if _degree(n) == 0:
                isolated_optional_dropped += 1
            else:
                active_nodes.add(n)

        if not active_nodes:
            achieved_retention[layer] = 0.0
            continue

        if r_L >= 1.0:
            # Every active node is its own cluster -- skip the geometric
            # assignment entirely (cheap path, no coordinate parsing needed).
            for n in active_nodes:
                node_to_rep[n] = n
            n_clusters_optional += len(active_nodes)
            achieved_retention[layer] = 1.0
            continue

        xy_of = {n: _parse_node_xy(n) for n in active_nodes}
        clusters = _cluster_layer_to_target(active_nodes, xy_of, adjacency, r_L)

        for members in clusters:
            rep = _pick_cluster_representative(members, xy_of)
            for m in members:
                node_to_rep[m] = rep
        n_clusters_optional += len(clusters)
        achieved_retention[layer] = len(clusters) / len(active_nodes)

    # kept_nodes is derived, not maintained in parallel across the branches
    # above -- every representative (and every self-mapped mandatory node)
    # appears as a node_to_rep value exactly once.
    kept_nodes = set(node_to_rep.values())
    n_clusters = len(mandatory_keep) + n_clusters_optional

    # Mandatory nodes with at least one real (non-self, non-ground) resistive
    # neighbor -- Pass 3's degree sanity check covers exactly these (a
    # self-loop-only or ground-only mandatory node legitimately contracts to
    # zero effective degree and is NOT a contraction bug).
    mandatory_connected = {
        n for n in mandatory_keep
        if any(v != n and v != GROUND_NODE for v, _g in adjacency.get(n, ()))
    }

    logger.info(
        "Tile %s: contracted optional pool into %d clusters (target=%d "
        "mandatory=%d isolated_optional_dropped=%d isolated_mandatory_kept=%d) "
        "-> kept=%d (%.3g%% of %d original nodes)",
        tile_id, n_clusters_optional, target_kept, len(mandatory_keep),
        isolated_optional_dropped, len(isolated_mandatory), len(kept_nodes),
        100.0 * len(kept_nodes) / total_nodes if total_nodes else 0.0,
        total_nodes,
    )

    return TileContractionResult(
        tile_id=tile_id,
        kept_nodes=kept_nodes,
        node_to_rep=node_to_rep,
        mandatory_nodes=set(mandatory_keep),
        mandatory_connected_nodes=mandatory_connected,
        target_kept=target_kept,
        mandatory_kept=len(mandatory_keep),
        optional_kept=n_clusters_optional,
        per_layer_retention=achieved_retention,
        n_clusters=n_clusters,
        isolated_optional_dropped=isolated_optional_dropped,
        isolated_mandatory_kept=len(isolated_mandatory),
    )


# =============================================================================
# Pass 3 — remap edges through the contraction (plan §4.3, revised)
# =============================================================================
#
# Replaces the original filter + BFS/via-chain connectivity repair (see the
# §7.5 failure analysis in the module docstring). There is no repair phase
# here: contraction already guarantees every kept node keeps a path to the
# rest of the tile (a quotient of a connected graph is connected), so Pass 3
# only needs to remap and merge edges through Pass 2's node_to_rep map.


@dataclass
class TileEdgeContractionResult:
    """Result of Pass 3 edge remapping for one tile.

    ``kept_resistive_edges``/``kept_capacitive_edges`` use the same
    ``(u, v, value)`` tuple shape as the original filter output, so the
    per-tile output writers are unchanged.
    """

    tile_id: Tuple[int, int]
    kept_resistive_edges: List[Tuple[str, str, float]]
    kept_capacitive_edges: List[Tuple[str, str, float]]
    intra_cluster_resistors_dropped: int
    parallel_resistors_merged: int
    intra_cluster_caps_dropped: int
    parallel_caps_merged: int
    # Split counters: resistors routing to a dropped node indicate a Pass 2
    # isolation bug (isolated nodes have zero R edges by definition) and are
    # ERROR-logged; caps routing to a dropped node are EXPECTED (cap-only
    # nodes are resistively isolated yet carry grounded caps) and quantify
    # the fF NOT conserved by contraction.
    resistors_to_dropped_nodes: int
    caps_to_dropped_nodes: int
    # Kept optional representatives removed post-merge because their
    # effective (non-ground) contracted degree was zero -- optional-only
    # islands swallowed by one cell, or ground-only stubs -- which would
    # otherwise reparse as floating nodes.
    islanded_optional_reps_dropped: int
    islanded_edges_dropped: int


def _remap_and_merge_edges(
    edges: Sequence[Tuple[str, str, float]], rep_of,
) -> Tuple[List[Tuple[str, str, float]], int, int, int]:
    """Remap+merge one edge list through *rep_of* (shared by R and C passes).

    Returns ``(kept_edges, n_intra_cluster_dropped, n_parallel_merged,
    n_edges_to_dropped_nodes)``. ``rep_of`` is a plain callable
    (``node_to_rep.get`` with ground special-cased) rather than a dict, so
    the same helper serves both resistors and capacitors without exposing
    ``GROUND_NODE`` handling twice.
    """
    merged: Dict[Tuple[str, str], float] = {}
    intra_cluster_dropped = 0
    edges_to_dropped_nodes = 0
    n_mapped = 0

    for u, v, value in edges:
        ru, rv = rep_of(u), rep_of(v)
        if ru is None or rv is None:
            # Impossible by construction (dropped nodes have no edges --
            # they were isolated), guarded defensively per plan §4.3.
            edges_to_dropped_nodes += 1
            continue
        n_mapped += 1
        if ru == rv:
            intra_cluster_dropped += 1
            continue
        key = (ru, rv) if ru < rv else (rv, ru)
        merged[key] = merged.get(key, 0.0) + value

    kept_edges = sorted((a, b, value) for (a, b), value in merged.items())
    parallel_merged = n_mapped - intra_cluster_dropped - len(merged)
    return kept_edges, intra_cluster_dropped, parallel_merged, edges_to_dropped_nodes


def contract_tile_edges(
    tile_data: TileData, contraction: TileContractionResult,
) -> TileEdgeContractionResult:
    """Pass 3: remap R/C edges through Pass 2's contraction map (plan §4.3).

    1. ``rep(n) = node_to_rep.get(n)``, with ``rep('0') = '0'``.
    2. Resistors: intra-cluster edges (``ru == rv``) are dropped; the rest
       accumulate parallel conductance keyed by ``(min(ru,rv), max(ru,rv))``
       so independently-ordered mappings of the same physical edge always
       merge into one entry.
    3. Capacitors: same remap/merge, general case handled (not just
       grounded) though tile capacitors are always grounded per root
       ``CLAUDE.md`` -- grounded caps of absorbed cluster members
       therefore accumulate onto their representative, so total tile
       capacitance is preserved exactly (minus caps on dropped isolates,
       counted separately).
    4. A cheap sanity check: every node in ``mandatory_connected_nodes``
       (mandatory with >= 1 non-self, non-ground resistive neighbor,
       precomputed in Pass 2) must have effective (non-ground) degree > 0
       in the contracted resistor list. This is NOT a repair -- it's a hard
       fail-loud assertion, since contraction guarantees it by construction
       (a mandatory node never absorbs its neighbor, so the remapped edge
       always survives as inter-cluster); a violation means a bug in
       Pass 2, not a data condition. Self-loop-only / ground-only mandatory
       nodes are exempt by definition of ``mandatory_connected_nodes``.
    5. Kept OPTIONAL representatives with zero effective contracted degree
       (optional-only islands fully swallowed by one cell; ground-only
       stubs) are removed together with their remaining ground/cap edges --
       they would otherwise be written to ``.nd``/``.node_count`` yet
       reparse as floating nodes. This step MUTATES
       ``contraction.kept_nodes`` / ``contraction.node_to_rep`` in place so
       downstream writers (``write_tile_nd`` etc.) see the final node set.

    Args:
        tile_data: Loaded ``TileData`` for this tile (original, pre-contraction
            edge lists).
        contraction: Output of ``contract_tile_nodes`` for this tile.
            ``kept_nodes``/``node_to_rep`` may be shrunk in place (step 5).

    Returns:
        A ``TileEdgeContractionResult`` (see above).

    Raises:
        AssertionError: if a ``mandatory_connected_nodes`` member ends up
            with zero effective degree after contraction -- a contraction bug.
    """
    tile_id = contraction.tile_id
    node_to_rep = contraction.node_to_rep

    def rep_of(n: str) -> Optional[str]:
        return GROUND_NODE if n == GROUND_NODE else node_to_rep.get(n)

    (
        kept_resistive_edges, intra_cluster_resistors_dropped,
        parallel_resistors_merged, resistors_to_dropped_nodes,
    ) = _remap_and_merge_edges(tile_data.resistive_edges, rep_of)

    (
        kept_capacitive_edges, intra_cluster_caps_dropped,
        parallel_caps_merged, caps_to_dropped_nodes,
    ) = _remap_and_merge_edges(tile_data.capacitive_edges, rep_of)

    if resistors_to_dropped_nodes:
        # Dropped (isolated) nodes have zero resistive edges by definition,
        # so a resistor routing to a dropped rep means Pass 2's isolation
        # logic mis-dropped a connected node: the output mesh is missing
        # real conductance. (Caps routing to dropped cap-only nodes are the
        # expected case and are reported, not errored.)
        logger.error(
            "Tile %s: %d resistor(s) reference a dropped node -- Pass 2 "
            "isolation bug, output mesh is missing conductance",
            tile_id, resistors_to_dropped_nodes,
        )

    # Effective (non-ground) contracted degree: ground resistors do not count
    # as connectivity for island detection at reparse (root CLAUDE.md).
    effective_degree: Dict[str, int] = {}
    for u, v, _g in kept_resistive_edges:
        if u != GROUND_NODE and v != GROUND_NODE:
            effective_degree[u] = effective_degree.get(u, 0) + 1
            effective_degree[v] = effective_degree.get(v, 0) + 1

    stranded = sorted(
        n for n in contraction.mandatory_connected_nodes
        if effective_degree.get(n, 0) == 0
    )
    if stranded:
        raise AssertionError(
            f"Tile {tile_id}: contraction stranded {len(stranded)} mandatory "
            f"node(s) that had >= 1 non-self, non-ground resistive neighbor "
            f"before contraction but zero effective degree after (contraction "
            f"bug, not a data condition): {stranded[:20]}"
        )

    # A kept OPTIONAL rep with zero effective degree would reparse as a
    # floating node: either an optional-only resistive island fully swallowed
    # by one cell (every incident edge intra-cluster) or a ground-only stub.
    # Such reps carry no edges to any other kept rep (an inter-rep edge would
    # give both endpoints degree >= 1), so removing them and their remaining
    # ground/cap edges cannot strand anything else.
    islanded_reps = {
        n for n in contraction.kept_nodes
        if n not in contraction.mandatory_nodes and effective_degree.get(n, 0) == 0
    }
    islanded_edges_dropped = 0
    if islanded_reps:
        n_r, n_c = len(kept_resistive_edges), len(kept_capacitive_edges)
        kept_resistive_edges = [
            e for e in kept_resistive_edges
            if e[0] not in islanded_reps and e[1] not in islanded_reps
        ]
        kept_capacitive_edges = [
            e for e in kept_capacitive_edges
            if e[0] not in islanded_reps and e[1] not in islanded_reps
        ]
        islanded_edges_dropped = (
            (n_r - len(kept_resistive_edges)) + (n_c - len(kept_capacitive_edges))
        )
        contraction.kept_nodes -= islanded_reps
        for member, rep in list(node_to_rep.items()):
            if rep in islanded_reps:
                del node_to_rep[member]
        logger.warning(
            "Tile %s: dropped %d optional representative(s) with zero "
            "effective (non-ground) contracted degree plus %d incident "
            "edge(s) -- these would reparse as floating nodes",
            tile_id, len(islanded_reps), islanded_edges_dropped,
        )

    logger.info(
        "Tile %s Pass 3: resistors %d -> %d (intra_cluster_dropped=%d "
        "parallel_merged=%d), capacitors %d -> %d (intra_cluster_dropped=%d "
        "parallel_merged=%d), islanded_reps_dropped=%d",
        tile_id, len(tile_data.resistive_edges), len(kept_resistive_edges),
        intra_cluster_resistors_dropped, parallel_resistors_merged,
        len(tile_data.capacitive_edges), len(kept_capacitive_edges),
        intra_cluster_caps_dropped, parallel_caps_merged, len(islanded_reps),
    )

    return TileEdgeContractionResult(
        tile_id=tile_id,
        kept_resistive_edges=kept_resistive_edges,
        kept_capacitive_edges=kept_capacitive_edges,
        intra_cluster_resistors_dropped=intra_cluster_resistors_dropped,
        parallel_resistors_merged=parallel_resistors_merged,
        intra_cluster_caps_dropped=intra_cluster_caps_dropped,
        parallel_caps_merged=parallel_caps_merged,
        resistors_to_dropped_nodes=resistors_to_dropped_nodes,
        caps_to_dropped_nodes=caps_to_dropped_nodes,
        islanded_optional_reps_dropped=len(islanded_reps),
        islanded_edges_dropped=islanded_edges_dropped,
    )


# =============================================================================
# Pass 4 — current sources (plan §4.4)
# =============================================================================


@dataclass
class CurrentSourceSamplingResult:
    """Result of Pass 4 current-source down-sampling for one tile.

    ``kept_raw_lines`` preserves the ORIGINAL raw line text verbatim (no
    ``pulse(...)``/``pwl(...)``/``static_value=``/``wscale=``
    reconstruction) -- unlike ``scan_current_source_nodes``, which only
    extracts node membership and discards the raw text entirely.
    """

    tile_id: Tuple[int, int]
    total_eligible: int
    kept_raw_lines: List[str]
    kept_node_pos: Set[str]
    excluded_not_in_kept_nodes: Set[str]
    total_matching_net: int


def _tile_current_source_seed(base_seed: int, tile_id: Tuple[int, int]) -> int:
    """Deterministic per-tile RNG seed for Pass 4 current-source down-sampling.

    Built only from plain ints/tuples-of-ints (no string elements), so
    ``hash(...)`` is stable across separate Python processes regardless of
    ``PYTHONHASHSEED`` string-hash randomization -- required for
    reproducibility across the embarrassingly-parallel per-tile runs the
    plan calls for (§5). This is the only RNG seed left in the pipeline --
    Passes 2/3 (contraction) are fully deterministic, no RNG at all.
    """
    return hash((base_seed, tile_id, -2)) & 0xFFFFFFFF


def _iter_instance_sources_with_raw_text(
    instance_path: Optional[str],
    net_filter: Optional[str],
    nd_path: Optional[str] = None,
):
    """Like ``distributed.tile_parsing._iter_instance_sources`` but also
    yields the original raw line text alongside each parsed ``PreparedSource``.

    ``_iter_instance_sources`` itself discards the raw line (it only needs
    the parsed ``node_pos``/``node_neg``/current values for solving), so
    Pass 4 -- which must write kept lines back out verbatim, with zero
    serializer round-trip risk for ``pulse(...)``/``pwl(...)``/
    ``static_value=``/``wscale=`` syntax -- needs its own loop. This
    mirrors ``_iter_instance_sources``'s structure line-for-line but reuses
    every one of its filtering/parsing helpers (``_has_structured_instance_names``,
    ``_fast_instance_net_filter``, ``_check_net_filter``,
    ``_prepare_instance_source``, ``_load_nd_file``) rather than
    reimplementing any of their logic.

    Yields:
        ``(raw_line, prepared)`` tuples, where ``raw_line`` is the
        original line with only the trailing newline/carriage-return
        stripped (otherwise byte-for-byte as read) and ``prepared`` is the
        ``PreparedSource`` that ``_prepare_instance_source`` returned for
        the corresponding stripped line.
    """
    if instance_path is None:
        return

    import gzip

    from distributed.tile_parsing import _is_gzip_file
    from parser.current_sources import _prepare_instance_source
    from parser.spice_lexer import (
        _check_net_filter,
        _fast_instance_net_filter,
        _has_structured_instance_names,
    )

    use_fast_filter = (
        net_filter is not None
        and _has_structured_instance_names(instance_path)
    )

    if net_filter and not use_fast_filter:
        node_net_map_lower = _load_nd_file(nd_path)
    else:
        node_net_map_lower = {}

    is_gzip = instance_path.endswith('.gz') or _is_gzip_file(instance_path)
    open_fn = gzip.open if is_gzip else open
    mode = 'rt' if is_gzip else 'r'

    with open_fn(instance_path, mode) as f:
        for raw_line in f:
            raw_line = raw_line.rstrip('\r\n')
            stripped = raw_line.strip()
            if not stripped or stripped.startswith('*') or stripped.startswith('.'):
                continue

            if use_fast_filter and not _fast_instance_net_filter(stripped, net_filter):
                continue

            prepared = _prepare_instance_source(stripped)
            if prepared is None:
                continue

            if net_filter and not use_fast_filter:
                if not _check_net_filter(
                    prepared.node_pos, prepared.node_neg,
                    node_net_map_lower, net_filter,
                ):
                    continue

            yield raw_line, prepared


def sample_current_sources(
    instance_path: Optional[str],
    kept_nodes: Set[str],
    tile_id: Tuple[int, int],
    nd_path: Optional[str] = None,
    net_filter: str = 'vdd_var',
    ratio: float = 0.1,
    base_seed: int = 0,
) -> CurrentSourceSamplingResult:
    """Pass 4: down-sample VDD_VAR current-source lines to ~``ratio`` (plan §4.4).

    Reuses the same fast structured-name / ``.nd``-fallback net filtering
    as ``scan_current_source_nodes``/``_iter_instance_sources`` (via
    ``_iter_instance_sources_with_raw_text``, which shares every one of
    the same underlying filter/parse helpers), but additionally preserves
    the original raw line text for each matching line, since kept lines
    must be written back out verbatim -- no reconstruction of
    ``pulse(...)``/``pwl(...)``/``static_value=``/``wscale=`` syntax,
    avoiding any serializer round-trip risk.

    Eligibility: a VDD_VAR current-source line is eligible for
    down-sampling iff its ``node_pos`` is in ``kept_nodes``. Since every
    current-source-bearing node was already force-kept as mandatory in
    Pass 1/2 (``classify_tile``'s ``current_source_nodes`` category feeds
    ``mandatory_keep``, always a subset of Pass 2's ``kept_nodes``), this
    should hold for *every* matching line -- any line whose ``node_pos``
    is NOT in ``kept_nodes`` is logged at WARNING (indicating a Pass 1/2
    bug elsewhere, not a Pass 4 concern) and excluded, since that node
    isn't present in the sampled tile output at all.

    Down-sampling uses a seeded ``random.Random`` (deterministic per-tile
    seeding, see ``_tile_current_source_seed``) to pick
    ``round(ratio * total_eligible)`` of the eligible *lines*, index-sampled
    from the file's natural (already-deterministic) iteration order -- no
    ``set``-iteration-order reproducibility risk, since eligible lines are
    collected into a plain list, not a set. Kept lines are returned in
    original file order, with their raw text unmodified.

    Args:
        instance_path: Path to ``instanceModels_X_Y.sp`` (gzip or plain
            text). Returns an all-empty result when ``None``.
        kept_nodes: This tile's final Pass-2/Pass-3 kept-node set.
        tile_id: This tile's ``(x, y)`` id. Not present in
            ``scan_current_source_nodes``'s signature, but required here
            for deterministic per-tile seeding (see
            ``_tile_current_source_seed``) and for log messages.
        nd_path: Optional ``.nd`` path (fallback net filtering only).
        net_filter: Lowercase net name to match (default ``'vdd_var'``).
        ratio: Target down-sampling ratio (default 0.1, i.e. keep ~10% of
            eligible current-source lines).
        base_seed: Base seed for the deterministic per-tile RNG.

    Returns:
        A ``CurrentSourceSamplingResult`` (see above).
    """
    eligible_lines: List[str] = []
    eligible_node_pos: List[str] = []
    excluded_not_in_kept_nodes: Set[str] = set()
    total_matching_net = 0

    for raw_line, prepared in _iter_instance_sources_with_raw_text(
        instance_path, net_filter, nd_path,
    ):
        total_matching_net += 1
        # Mirror scan_current_source_nodes' terminal handling: when the
        # positive terminal is ground (reversed-terminal line), the die node
        # that Pass 1/2 mandatory-kept is node_neg -- eligibility must check
        # the same terminal or the line is silently lost.
        anchor = (
            prepared.node_pos if prepared.node_pos != GROUND_NODE
            else prepared.node_neg
        )
        if anchor in kept_nodes:
            eligible_lines.append(raw_line)
            eligible_node_pos.append(anchor)
        else:
            excluded_not_in_kept_nodes.add(anchor)

    if excluded_not_in_kept_nodes:
        logger.warning(
            "Tile %s: %d VDD_VAR current-source line(s) reference node_pos "
            "NOT in kept_nodes (%s%s) -- excluded from output, but this "
            "indicates a Pass 1/2 mandatory-keep bug (current-source-bearing "
            "nodes should always be force-kept)",
            tile_id, len(excluded_not_in_kept_nodes),
            sorted(excluded_not_in_kept_nodes)[:10],
            "..." if len(excluded_not_in_kept_nodes) > 10 else "",
        )

    total_eligible = len(eligible_lines)
    k = min(total_eligible, round(ratio * total_eligible))

    rng = random.Random(_tile_current_source_seed(base_seed, tile_id))
    kept_indices = sorted(rng.sample(range(total_eligible), k)) if k > 0 else []

    kept_raw_lines = [eligible_lines[i] for i in kept_indices]
    kept_node_pos = {eligible_node_pos[i] for i in kept_indices}

    logger.info(
        "Tile %s Pass 4: %d VDD_VAR current-source line(s) matched net filter, "
        "%d eligible (node_pos in kept_nodes), kept %d (~%.3g%% of eligible, "
        "target ratio %.3g)",
        tile_id, total_matching_net, total_eligible, len(kept_raw_lines),
        100.0 * len(kept_raw_lines) / total_eligible if total_eligible else 0.0,
        ratio,
    )

    return CurrentSourceSamplingResult(
        tile_id=tile_id,
        total_eligible=total_eligible,
        kept_raw_lines=kept_raw_lines,
        kept_node_pos=kept_node_pos,
        excluded_not_in_kept_nodes=excluded_not_in_kept_nodes,
        total_matching_net=total_matching_net,
    )


class CapacitorInvariantViolation(AssertionError):
    """Raised when the capacitor-follows-current-source invariant fails.

    Subclasses ``AssertionError`` (not a bare ``Exception``) since this
    documents a "must hold by construction" correctness contract (plan
    §4.4 point 5) -- a violation means a real bug in Pass 3's capacitor
    filtering or Pass 1/2's mandatory-keep logic, not a recoverable/
    expected runtime condition, so it must fail loudly rather than
    warn-and-continue.
    """


def _grounded_cap_nodes(edges: Sequence[Tuple[str, str, float]]) -> Set[str]:
    """Return the set of non-ground endpoints among grounded cap edges."""
    nodes: Set[str] = set()
    for u, v, _c in edges:
        if u == GROUND_NODE and v != GROUND_NODE:
            nodes.add(v)
        elif v == GROUND_NODE and u != GROUND_NODE:
            nodes.add(u)
    return nodes


def verify_capacitor_invariant(
    tile_data: TileData,
    current_source_nodes: Set[str],
    kept_capacitive_edges: List[Tuple[str, str, float]],
) -> None:
    """Verify the capacitor-follows-current-source invariant (plan §4.4.5).

    For every node in *current_source_nodes* (the full Pass-1 pre-scanned
    set -- this invariant is about the *node* having its grounded
    capacitor, independent of which specific current-source *lines*
    survived Pass 4's down-sampling, since down-sampling only removes some
    lines at an already-mandatory-kept node, never the node itself) that
    originally had >= 1 grounded capacitor edge in
    ``tile_data.capacitive_edges``, asserts that at least one grounded cap
    edge for that same node is present in *kept_capacitive_edges*.

    Args:
        tile_data: Original (pre-sampling) ``TileData`` for this tile.
        current_source_nodes: This tile's current-source-bearing node set
            (e.g. ``classification.current_source_nodes`` from Pass 1 /
            ``scan_current_source_nodes``'s pre-scan).
        kept_capacitive_edges: This tile's Pass-3 output
            (``TileEdgeContractionResult.kept_capacitive_edges``).

    Raises:
        CapacitorInvariantViolation: if any current-source-bearing node
            that originally had a grounded cap lost it during Pass 3
            contraction. Per the plan this must hold by construction, so a
            violation indicates a real bug elsewhere -- this function
            does not warn-and-continue.
    """
    originally_capped = _grounded_cap_nodes(tile_data.capacitive_edges)
    kept_capped = _grounded_cap_nodes(kept_capacitive_edges)

    violations = (current_source_nodes & originally_capped) - kept_capped

    if violations:
        raise CapacitorInvariantViolation(
            f"Capacitor-follows-current-source invariant violated for tile "
            f"{tile_data.tile_id}: {len(violations)} current-source-bearing "
            f"node(s) originally had a grounded capacitor but it did not "
            f"survive Pass 3 filtering: {sorted(violations)[:20]}"
            + ("..." if len(violations) > 20 else "")
        )


# =============================================================================
# Output generation — top-level files (plan §4.5)
# =============================================================================
#
# Covers ONLY the 4 top-level, tile-count-independent files: `pg_net_voltage`,
# `additional_vsrcs`, `ckt.sp`, and `package.ckt`. Per-tile output writers
# (`generate_tile_ckt_content`/`write_tile_ckt`/`write_tile_nd`, gzip
# `write_instance_models`) follow below, and the full pipeline orchestration/
# CLI that stitches Phases 1-4 + this section together into a real
# `netlist/netlist_brcm_sampled/` directory (`process_tile`,
# `run_sampling_pipeline`, `main`) follows after that -- see the plan doc's
# §4.5 per-tile bullets for the file-format details.


def _format_number(value: float) -> str:
    """Render a numeric value the way the source SPICE text does.

    Integral values are rendered as plain integers (``19152000``, not
    ``19152000.0`` and NOT ``%g``'s ``'1.9152e+07'`` -- die_area coordinates
    are up to 8 digits and ``%g``'s default 6-significant-digit precision
    silently pushes them into scientific notation, verified). Non-integral
    values (e.g. the ``0.76`` VDD_VAR voltage) use plain ``str()``, which
    already round-trips SPICE-style decimals like ``0.76``/``1.5`` without
    exponents for any realistic PDN parameter magnitude.
    """
    if float(value).is_integer():
        return str(int(value))
    return str(value)


def _resolve_net_display_name(source_dir: Path, net_name: str) -> str:
    """Resolve the net's DISPLAY casing from the source ``pg_net_voltage``.

    ``metadata.net_name`` is the verbatim ``--net`` argument from parse time
    (possibly lowercase); output files must declare the net as the SOURCE
    design spells it, or reparsing the sampled netlist fails to find its
    voltage (``_extract_vdd`` matches the file token, not a lowercased
    form). Falls back to *net_name* as-is when the source file is missing
    or has no case-insensitive match.
    """
    pg_path = Path(source_dir) / 'pg_net_voltage'
    if pg_path.exists():
        for line in pg_path.read_text().splitlines():
            tokens = line.split()
            if tokens and tokens[0].lower() == net_name.lower():
                return tokens[0]
    return net_name


def generate_pg_net_voltage(vdd: float, net_name: str = 'VDD_VAR') -> str:
    """Generate the sampled ``pg_net_voltage`` file content (VDD_VAR-only).

    Matches the REAL file's exact format, verified byte-for-byte against
    ``netlist/netlist_brcm/pg_net_voltage`` via ``cat -A``:  each line is
    ``<net_name> - <voltage> `` -- a literal ``" - "`` separator and a
    trailing space before the newline -- NOT the plan doc's paraphrased
    "single ``VDD_VAR 0.76``" description. The real file has 4 lines
    (VDD_VAR/VSS/pad_PLL_VDD1p5/pad_PLL_VSS); the sampled netlist is
    VDD_VAR-only, so this always emits exactly one line.

    Args:
        vdd: The net's voltage (``metadata.vdd``, e.g. ``0.76``).
        net_name: Display net name as it appears in the SOURCE design's
            files (default ``'VDD_VAR'`` for backward compatibility --
            callers should pass ``_resolve_net_display_name(...)``).

    Returns:
        Full file content, including the trailing newline.
    """
    return f"{net_name} - {_format_number(vdd)} \n"


def generate_additional_vsrcs() -> str:
    """Generate the sampled ``additional_vsrcs`` file content.

    The real file is 0 bytes (verified via ``wc -c``) -- there is nothing
    net-specific to filter, so the sampled version is identically empty.
    """
    return ""


def read_die_area_from_ckt_sp(ckt_sp_path: Path) -> Tuple[float, float, float, float]:
    """Read the ``.die_area x0 y0 x1 y1`` line from a source ``ckt.sp``.

    Neither ``PowerGridMetaData`` nor ``TileConfig`` carries a die_area
    field (verified against ``distributed.parser``: ``_parse_main_netlist``
    reads ``.die_area`` while scanning ``ckt.sp`` but never stores it on the
    ``PowerGridMetaData`` it returns), so ``generate_ckt_sp`` cannot get it
    from ``distributed_pkl/metadata.pkl`` -- it must be read from the real
    ``netlist/netlist_brcm/ckt.sp`` directly, once, up front.

    Args:
        ckt_sp_path: Path to a (real, unsampled) ``ckt.sp`` file.

    Returns:
        ``(x0, y0, x1, y1)`` as floats.

    Raises:
        ValueError: if no ``.die_area`` line is found.
    """
    with open(ckt_sp_path, 'r') as f:
        for line in f:
            tokens = line.split()
            if tokens and tokens[0] == '.die_area':
                if len(tokens) < 5:
                    raise ValueError(f"Malformed .die_area line: {line!r}")
                x0, y0, x1, y1 = (float(t) for t in tokens[1:5])
                return (x0, y0, x1, y1)
    raise ValueError(f"No .die_area line found in {ckt_sp_path}")


def generate_ckt_sp(
    tile_grid: Tuple[int, int],
    die_area: Tuple[float, float, float, float],
    vdd: float,
    tile_ids: Sequence[Tuple[int, int]],
    net_name: str = 'VDD_VAR',
) -> str:
    """Generate the sampled top-level ``ckt.sp`` content (VDD_VAR-only).

    Reproduces the real file's structure, verified against
    ``netlist/netlist_brcm/ckt.sp``: ``.partition_info``/``.die_area``
    (both UNCHANGED -- same 36-tile grid, same absolute coordinates, no
    spatial cropping per the plan), then a single VDD_VAR
    ``.parameter``/``v...`` pair (the other 3 nets' lines -- VSS,
    pad_PLL_VDD1p5, pad_PLL_VSS -- are dropped entirely, since this whole
    pipeline is VDD_VAR-only), then one ``.include ./tile_X_Y.ckt`` per
    *tile_ids* entry, then ``.include ./package.ckt``, then one
    ``.include ./instanceModels_X_Y.sp`` per *tile_ids* entry (same order
    as the tile includes).

    This file **is** read by ``DistributedNetlistParser`` (for
    ``.partition_info`` + ``.parameter``/vdd extraction), but tile/instance
    file discovery itself is by filename pattern on disk
    (``distributed.parser._discover_tiles``), not by parsing these
    ``.include`` lines -- so their order doesn't affect correctness, only
    diffability against the source. Callers should still pass *tile_ids*
    in the same (X outer, Y inner) order as the source file (e.g.
    ``discover_tile_ids(metadata)``, which already returns sorted tuples)
    for that diffability.

    Args:
        tile_grid: ``(n_x, n_y)`` tile grid shape, e.g. ``(6, 6)``. Pass
            ``metadata.tile_grid`` directly (this field IS carried on
            ``PowerGridMetaData``, unlike ``die_area``).
        die_area: ``(x0, y0, x1, y1)`` absolute die-area bounds -- see
            ``read_die_area_from_ckt_sp``.
        vdd: The net's voltage (``metadata.vdd``).
        tile_ids: Tile ids to ``.include``, in output order (used for both
            the ``tile_X_Y.ckt`` and ``instanceModels_X_Y.sp`` blocks).

    Returns:
        Full file content, including the trailing newline.
    """
    n_x, n_y = tile_grid
    lines: List[str] = [
        f".partition_info {n_x} {n_y}",
        ".die_area " + " ".join(_format_number(v) for v in die_area),
        f".parameter {net_name} {_format_number(vdd)}",
        f"v{net_name} v{net_name}  0  {net_name}",
    ]
    lines.extend(f".include ./tile_{x}_{y}.ckt" for x, y in tile_ids)
    lines.append(".include ./package.ckt")
    lines.extend(f".include ./instanceModels_{x}_{y}.sp" for x, y in tile_ids)
    return "\n".join(lines) + "\n"


def _pkg_first_token(line: str) -> Optional[str]:
    """Return the first whitespace-separated token of *line*, or ``None`` if blank."""
    tokens = line.split()
    return tokens[0] if tokens else None


def _print_line_pad_name(line: str) -> Optional[str]:
    """Extract the base pad name from a ``package.ckt`` section-3 ``.print`` line.

    Matches ``.print v(<pad>_probe)`` / ``.print v(<pad>_int)`` and returns
    ``<pad>``. Returns ``None`` if *line* doesn't have that shape
    (defensive: real ``package.ckt`` section 3 is uniformly this shape, but
    callers should not crash on an unexpected line -- it's simply excluded
    from the kept-pad-name membership test).
    """
    if '(' not in line or ')' not in line:
        return None
    inner = line.split('(', 1)[1].rsplit(')', 1)[0]
    for suffix in ('_probe', '_int'):
        if inner.endswith(suffix):
            return inner[: -len(suffix)]
    return None


def filter_package_ckt(source_lines: List[str], net_filter: str = 'VDD_VAR') -> List[str]:
    """Filter a ``package.ckt``'s lines down to a single net's pads.

    The real ``package.ckt`` (``netlist/netlist_brcm/package.ckt``,
    verified) is organized into exactly 3 contiguous sections -- NOT
    per-pad interleaved blocks, as the plan doc's paraphrase suggested:

      1. Pad "loop" blocks -- 4 lines per pad (``v_<pad> <pad>_vsrc 0
         <NET>``, ``r  <pad> <pad>_probe 0``, ``r_<pad>_probe_probe
         <pad>_probe <pad>_int 0.001``, ``r  <pad>_int <pad>_vsrc 0``),
         all 4 nets combined (VDD_VAR, VSS, pad_PLL_VDD1p5, pad_PLL_VSS).
         628 pads -> 2512 lines in the real file (verified).
      2. ``rs <die_coord_node> <pad_node> 0`` anchor lines, one per pad,
         in the SAME per-pad order as section 1 (verified via a direct
         per-pad-name comparison across both sections on the real file).
         628 lines in the real file (lines 2513-3140).
      3. ``.print v(<pad>_probe)`` / ``.print v(<pad>_int)`` lines, two per
         pad, same per-pad order again (verified). 1256 lines in the real
         file (lines 3141-4396), followed by 2 trailing blank lines
         (total file length 4398).

    Section boundaries are detected structurally (first line whose first
    token is ``'rs'`` starts section 2; first line whose first token is
    ``'.print'`` starts section 3) rather than via hardcoded line counts,
    so this works on both the real 4398-line file and small synthetic
    fixtures.

    Net identification uses the exact 4th token of each block's ``v_``
    line (the literal net name, e.g. ``VDD_VAR``) -- NOT a substring match
    on the pad name, since e.g. ``pad_PLL_VDD1p5`` also contains ``VDD`` as
    a substring and would be wrongly kept by a naive ``'VDD' in padname``
    check. The pad node name itself (for section 2/3 membership) is taken
    from the ``v_`` line's first token with its ``'v_'`` prefix stripped
    (e.g. ``'v_bmpary_bmp_VDD_VAR_0_1'`` -> ``'bmpary_bmp_VDD_VAR_0_1'``),
    which is the exact bare pad node name used verbatim in the ``rs``
    line's 3rd token and (with a ``_probe``/``_int`` suffix) in the
    ``.print`` lines.

    Args:
        source_lines: package.ckt content, one element per line, WITHOUT
            trailing newlines (e.g. ``Path(...).read_text().splitlines()``).
            Blank lines anywhere in the input are dropped.
        net_filter: Exact net name to keep (default ``'VDD_VAR'``).

    Returns:
        The filtered lines (same "no trailing newline per element"
        convention as the input), covering all 3 sections for the kept net
        only, in section 1 -> section 2 -> section 3 order. Join with
        ``'\\n'`` (plus a final trailing newline) to write out the file.

    Raises:
        ValueError: if section 1 isn't a clean multiple of 4 lines, if a
            ``.print`` line is found before the first ``rs`` line, or a
            block's first line doesn't look like a ``v_<pad> ... <NET>``
            header -- all indicate malformed/unexpected input, a real
            correctness bug rather than a case to silently skip.
    """
    lines = [l.rstrip('\n') for l in source_lines if l.strip()]

    sec2_start = next(
        (i for i, l in enumerate(lines) if _pkg_first_token(l) == 'rs'), len(lines)
    )
    sec3_start = next(
        (i for i, l in enumerate(lines) if _pkg_first_token(l) == '.print'), len(lines)
    )
    if sec3_start < sec2_start:
        raise ValueError(
            "package.ckt structure violated: found a '.print' line before "
            "the first 'rs' line"
        )

    section1 = lines[:sec2_start]
    section2 = lines[sec2_start:sec3_start]
    section3 = lines[sec3_start:]

    if len(section1) % 4 != 0:
        raise ValueError(
            f"package.ckt section 1 (pad loop blocks) has {len(section1)} "
            f"lines, not a clean multiple of 4 -- cannot chunk into per-pad blocks"
        )

    kept_pad_names: Set[str] = set()
    kept_section1: List[str] = []
    for i in range(0, len(section1), 4):
        block = section1[i:i + 4]
        v_tokens = block[0].split()
        if len(v_tokens) < 4 or not v_tokens[0].startswith('v_'):
            raise ValueError(
                "Unexpected package.ckt pad-block header line (expected "
                f"'v_<pad> <pad>_vsrc 0 <NET>'): {block[0]!r}"
            )
        net = v_tokens[3]
        # Case-insensitive: metadata.net_name is the verbatim --net argument
        # (a lowercase `sigma-dvd parse --net vdd_var` works everywhere else
        # in the pipeline), while package.ckt net tokens are upper-case.
        if net.lower() != net_filter.lower():
            continue
        pad_name = v_tokens[0][len('v_'):]
        kept_pad_names.add(pad_name)
        kept_section1.extend(block)

    kept_section2 = [
        l for l in section2
        if len(l.split()) >= 3 and l.split()[2] in kept_pad_names
    ]

    kept_section3 = [
        l for l in section3 if _print_line_pad_name(l) in kept_pad_names
    ]

    logger.info(
        "filter_package_ckt(net_filter=%s): kept %d/%d pads -> %d section-1 "
        "lines, %d rs anchors, %d .print lines",
        net_filter, len(kept_pad_names), len(section1) // 4,
        len(kept_section1), len(kept_section2), len(kept_section3),
    )

    return kept_section1 + kept_section2 + kept_section3


# =============================================================================
# Output generation — per-tile files (plan §4.5)
# =============================================================================
#
# Covers the 3 per-tile output files (`tile_X_Y.ckt`, `tile_X_Y.nd`,
# `instanceModels_X_Y.sp`), all gzip despite plain (non-`.gz`) filenames --
# same convention as the real source files (verified via `file`/magic-byte
# detection, see module docstring / memory notes). Pairs with the top-level
# file generators above to fully cover plan §4.5.


def _read_text_lines_auto(path: Path) -> List[str]:
    """Read every line of a gzip-or-plain text file, verbatim (newline stripped only).

    Real per-tile files are gzip content despite a plain (non-``.gz``)
    extension, so this uses magic-byte detection
    (``distributed.tile_parsing._is_gzip_file``), not an extension check --
    same convention as ``_load_nd_file``/``_iter_instance_sources_with_raw_text``.
    """
    path_str = str(path)
    is_gzip = path_str.endswith('.gz') or _is_gzip_file(path_str)
    open_fn = gzip.open if is_gzip else open
    mode = 'rt' if is_gzip else 'r'
    with open_fn(path_str, mode) as f:
        return [line.rstrip('\r\n') for line in f]


def _conductance_ms_to_ohm(g_mS: float) -> float:
    """Convert ``TileData``'s native conductance (mS) back to raw resistance (Ohms).

    Inverse of ``distributed.tile_parsing._parse_tile_ckt``'s forward
    conversion (``r_kohm = r_value_ohm * R_TO_KOHM``, then
    ``g_mS = 1 / r_kohm``): ``r_kohm = 1 / g_mS``, then
    ``r_ohm = r_kohm / R_TO_KOHM`` (equivalently ``1000.0 / g_mS``, since
    ``R_TO_KOHM == 1e-3`` -- reusing the shared constant rather than
    hardcoding the ``1000.0`` magic number).

    Note (near-short edges): the forward conversion clamps any raw
    resistance below ``SHORT_THRESHOLD`` (1 mOhm, incl. literal 0 Ohm
    shorts) to a fixed ``GMAX = 1e5`` mS sentinel -- the TRUE original
    sub-mOhm value is already permanently lost at that point, before this
    sampling pipeline ever runs (an accepted consequence of using the pkl,
    not the raw text, as the mesh data source -- see the plan's Pass
    1 design decision). This function's inverse therefore reconstructs
    ``GMAX`` back to a fixed ``0.01`` Ohm (``1000.0 / 1e5``) for every such
    edge, not the unrecoverable true original value -- but this IS the
    electrically-correct round-trip for simulation purposes: 0.01 Ohm
    (10 mOhm) is above ``SHORT_THRESHOLD``, so re-parsing the sampled
    output reconstructs exactly ``GMAX`` again, matching the same
    regularized conductance the original ``distributed_pkl`` (and any
    solve against it) already used.
    """
    # Parallel-merged near-shorts (e.g. k GMAX via edges between two reps,
    # g = k*1e5 mS) can invert to r_kohm below the reparse SHORT_THRESHOLD
    # (1e-6 kOhm in distributed.tile_parsing), which would re-clamp the edge
    # back to a single GMAX and silently lose a factor-k of conductance.
    # Clamp the written resistance AT the threshold instead: the reparse
    # then reconstructs min(g_mS, 1e6) exactly (bounded, conservative loss
    # only for g > 1e6 mS, i.e. sub-microohm merges).
    r_kohm = max(1.0 / g_mS, _REPARSE_SHORT_THRESHOLD_KOHM)
    return r_kohm / R_TO_KOHM


def _capacitance_ff_to_farad(c_fF: float) -> float:
    """Convert ``TileData``'s native capacitance (fF) back to raw Farads.

    Inverse of ``distributed.tile_parsing._parse_tile_ckt``'s forward
    conversion (``c_fF = c_value_farad * C_TO_FF``): ``c_farad = c_fF /
    C_TO_FF`` (equivalently ``c_fF * 1e-15``, since ``C_TO_FF == 1e15``).
    """
    return c_fF / C_TO_FF


def _format_spice_float(value: float) -> str:
    """Render a raw Ohm/Farad value for a ``.ckt`` element line.

    Python 3's ``str()``/``repr()`` for floats already produces the
    shortest decimal string that round-trips to the exact same float (and
    naturally mixes plain-decimal and scientific notation depending on
    magnitude, matching the real file's style, e.g. ``0.013312`` /
    ``1.2665e-14``) -- no custom formatting needed, verified by round-trip
    through ``parser.spice_lexer._parse_spice_value``.
    """
    return str(value)


def _prefixed_node(node: str, boundary_nodes: Set[str]) -> str:
    """Add the literal ``*`` boundary-node prefix iff *node* is a boundary node.

    Applied independently at each occurrence of a node token in an
    element line (a boundary node gets ``*`` on EVERY line where it
    appears, not just some -- verified against real data: both the ``c``
    and ``r`` line for the same boundary node carry the prefix).
    """
    return f"*{node}" if node in boundary_nodes else node


def _format_element_line(
    kind: str, n1: str, n2: str, value: float, boundary_nodes: Set[str],
) -> str:
    """Format one nameless ``r``/``c`` element line (plan §4.5's ``tile_X_Y.ckt``).

    Matches the real file's nameless-element, double-space-separated
    convention (``r  <node1>  <node2>  <value>`` / ``c  <node1>  <node2>
    <value>``) -- verified this doesn't affect parser correctness (all real
    tokenizers use ``str.split()``, whitespace-count-insensitive), but kept
    for diffability against the source.
    """
    return (
        f"{kind}  {_prefixed_node(n1, boundary_nodes)}  "
        f"{_prefixed_node(n2, boundary_nodes)}  {_format_spice_float(value)}"
    )


def generate_tile_ckt_content(
    node_count: int,
    boundary_nodes: Set[str],
    resistive_edges: Sequence[Tuple[str, str, float]],
    capacitive_edges: Sequence[Tuple[str, str, float]],
) -> str:
    """Generate the sampled ``tile_X_Y.ckt`` content (plan §4.5).

    Header is ``.node_count <node_count>`` (reflecting the NEW sampled
    node count, not the original) then ``.flag_boundary`` (both no-ops to
    the parser, regenerated only for diffability -- see module docstring),
    followed by one nameless ``r``/``c`` line per surviving edge, unit
    conversion back to raw Ohms/Farads via ``_conductance_ms_to_ohm``/
    ``_capacitance_ff_to_farad``, with the ``*`` boundary prefix re-added
    per plan §4.5.

    Args:
        node_count: The new (post-contraction) kept-node count for this
            tile, i.e. ``len(TileContractionResult.kept_nodes)``.
        boundary_nodes: This tile's boundary-node set (e.g.
            ``TileClassification.boundary_nodes`` /
            ``TileData.boundary_nodes``) -- a node gets the ``*`` prefix
            on every element line it appears on iff it's in this set.
        resistive_edges: Final (post Pass-3 contraction) resistor edges,
            conductance in mS -- typically
            ``TileEdgeContractionResult.kept_resistive_edges``.
        capacitive_edges: Final (post Pass-3 contraction) grounded
            capacitor edges, in fF -- typically
            ``TileEdgeContractionResult.kept_capacitive_edges``.

    Returns:
        Full file content (including trailing newline), ready to be
        gzip-written via ``write_tile_ckt``.
    """
    lines: List[str] = [f".node_count {node_count}", ".flag_boundary"]
    for u, v, g_ms in resistive_edges:
        r_ohm = _conductance_ms_to_ohm(g_ms)
        lines.append(_format_element_line('r', u, v, r_ohm, boundary_nodes))
    for u, v, c_ff in capacitive_edges:
        c_farad = _capacitance_ff_to_farad(c_ff)
        lines.append(_format_element_line('c', u, v, c_farad, boundary_nodes))
    return "\n".join(lines) + "\n"


@contextlib.contextmanager
def _open_gzip_text_deterministic(output_path: Path):
    """gzip text writer with ``mtime=0`` and no embedded filename.

    ``gzip.open(path, 'wt')`` stamps the current wall clock into the gzip
    header's MTIME field, so identical content yields byte-different files
    across runs -- breaking the pipeline's rerun-and-checksum
    reproducibility contract (plan §5). Report files (wall clock inside)
    are exempt from that contract; per-tile outputs are not.
    """
    with open(output_path, 'wb') as raw:
        with gzip.GzipFile(fileobj=raw, mode='wb', filename='', mtime=0) as gz:
            with io.TextIOWrapper(gz, encoding='utf-8') as text:
                yield text


def write_tile_ckt(
    output_path: Path,
    node_count: int,
    boundary_nodes: Set[str],
    resistive_edges: Sequence[Tuple[str, str, float]],
    capacitive_edges: Sequence[Tuple[str, str, float]],
) -> None:
    """Write a gzip ``tile_X_Y.ckt`` (plain filename, gzip content -- plan §4.5)."""
    content = generate_tile_ckt_content(node_count, boundary_nodes, resistive_edges, capacitive_edges)
    with _open_gzip_text_deterministic(output_path) as f:
        f.write(content)


def filter_tile_nd_lines(nd_lines: Sequence[str], kept_nodes: Set[str]) -> List[str]:
    """Filter raw ``.nd`` lines down to ``kept_nodes`` (plan §4.5).

    The parser (``parser.netlist._load_node_net_map``) only consumes
    ``tokens[0]`` (node name) and ``tokens[5]`` (net name) of each 6-token
    ``.nd`` line -- the 4 middle "cosmetic" fields are carried through
    VERBATIM (this function does no reformatting at all, just a
    keep/drop decision on ``tokens[0]``), matching the plan's explicit
    "copied verbatim from the original raw .nd file" instruction.

    Args:
        nd_lines: Raw ``.nd`` lines (newline already stripped, e.g. from
            ``_read_text_lines_auto``), one per original node.
        kept_nodes: This tile's final kept-node set.

    Returns:
        The subset of *nd_lines* whose first whitespace-separated token
        is in *kept_nodes*, in original order, unmodified.
    """
    kept: List[str] = []
    for line in nd_lines:
        if not line.strip():
            continue
        first_token = line.split(None, 1)[0]
        if first_token in kept_nodes:
            kept.append(line)
    return kept


def write_tile_nd(output_path: Path, source_nd_path: Path, kept_nodes: Set[str]) -> int:
    """Filter+write a gzip ``tile_X_Y.nd`` from the original raw ``.nd`` file (plan §4.5).

    Args:
        output_path: Destination ``tile_X_Y.nd`` path (written gzip, plain
            filename).
        source_nd_path: Original (real, unsampled) ``tile_X_Y.nd`` path to
            read and filter (gzip or plain, magic-byte detected).
        kept_nodes: This tile's final kept-node set.

    Returns:
        Number of lines written (for logging/diagnostics).
    """
    source_lines = _read_text_lines_auto(source_nd_path)
    kept_lines = filter_tile_nd_lines(source_lines, kept_nodes)
    content = "\n".join(kept_lines) + ("\n" if kept_lines else "")
    with _open_gzip_text_deterministic(output_path) as f:
        f.write(content)
    return len(kept_lines)


def write_instance_models(output_path: Path, kept_raw_lines: Sequence[str]) -> None:
    """Write a gzip ``instanceModels_X_Y.sp`` from Pass 4's kept raw lines (plan §4.5).

    Just ``CurrentSourceSamplingResult.kept_raw_lines`` joined with
    newlines and gzip-written -- no header/footer lines are added (the
    real file's leading ``.flag_boundary`` line is a no-op that
    ``_iter_instance_sources_with_raw_text`` already skips on read, so
    omitting it on write is harmless and matches the plan's explicit
    "nothing else needs to be in this file" instruction).

    Args:
        output_path: Destination ``instanceModels_X_Y.sp`` path (written
            gzip, plain filename).
        kept_raw_lines: ``CurrentSourceSamplingResult.kept_raw_lines`` for
            this tile.
    """
    content = "\n".join(kept_raw_lines) + ("\n" if kept_raw_lines else "")
    with _open_gzip_text_deterministic(output_path) as f:
        f.write(content)


# =============================================================================
# Full per-tile pipeline + orchestration (plan §5)
# =============================================================================


@dataclass
class TileProcessingStats:
    """Original vs. sampled counts + contraction diagnostics for one processed tile.

    Everything a final aggregate report (plan §6) needs, without forcing
    callers to re-derive it from the individual Pass 1-4 result objects
    (which ``process_tile`` doesn't return directly, to keep its return
    type simple for orchestration/``multiprocessing`` use).
    """

    tile_id: Tuple[int, int]
    original_nodes: int
    sampled_nodes: int
    original_resistors: int
    sampled_resistors: int
    original_capacitors: int
    sampled_capacitors: int
    original_current_sources: int
    sampled_current_sources: int
    pad_anchor_count: int
    n_clusters: int
    isolated_optional_dropped: int
    isolated_mandatory_kept: int
    islanded_optional_reps_dropped: int
    intra_cluster_resistors_dropped: int
    parallel_resistors_merged: int

    @property
    def node_ratio(self) -> float:
        return self.sampled_nodes / self.original_nodes if self.original_nodes else 0.0


def process_tile(
    source_dir: Path,
    pkl_dir: Path,
    output_dir: Path,
    tile_id: Tuple[int, int],
    all_pad_anchors: Set[str],
    ratio: float = 0.1,
    alpha: float = 1.0,
    base_seed: int = 42,
    net_filter: str = 'vdd_var',
) -> TileProcessingStats:
    """Run Pass 1-4 + per-tile output writing for a single tile (plan §5).

    Steps (see module docstring for the individual pass functions):
      1. Load ``TileData`` from ``pkl_dir`` and this tile's pad-anchor
         subset (``_pad_anchors_in_tile``).
      2. Pre-scan current-source-bearing nodes from the raw
         ``instanceModels_X_Y.sp`` (``scan_current_source_nodes``).
      3. Classify (``classify_tile``), contract nodes
         (``contract_tile_nodes``), remap edges (``contract_tile_edges``).
      4. Down-sample current sources (``sample_current_sources`` --
         re-reads the same raw instanceModels file a second time; a known,
         accepted minor inefficiency, not restructured here since Pass 4's
         API is already reviewed/tested).
      5. Verify the capacitor-follows-current-source invariant
         (``verify_capacitor_invariant`` -- lets ``CapacitorInvariantViolation``
         propagate uncaught; this is a hard pipeline failure per the plan,
         never swallowed).
      6. Write ``tile_X_Y.ckt``/``.nd`` and ``instanceModels_X_Y.sp`` into
         *output_dir*.

    Args:
        source_dir: Directory with the original raw per-tile text files
            (``tile_X_Y.nd``, ``instanceModels_X_Y.sp`` -- read-only
            source, e.g. ``netlist/netlist_brcm``).
        pkl_dir: Directory with ``tile_X_Y.pkl`` (``distributed_pkl``, the
            already-parsed mesh topology -- see ``load_tile_data``).
        output_dir: Destination directory for this tile's 3 sampled output
            files (created if missing).
        tile_id: ``(x, y)`` tile coordinate.
        all_pad_anchors: Design-wide pad-anchor (``die_attachment_nodes``)
            set, e.g. ``metadata.package_data.die_attachment_nodes`` --
            intersected internally with this tile's node set.
        ratio: Target retention ratio (default 0.1, ~10x reduction), used
            for both node contraction (Pass 2) and current-source
            down-sampling (Pass 4).
        alpha: Layer-weight exponent for Pass 2 (default 1.0).
        base_seed: Base seed for the deterministic per-tile RNG -- affects
            ONLY Pass 4's current-source down-sampling; Passes 2/3
            (contraction) have no RNG and are unaffected by this value.

    Returns:
        A ``TileProcessingStats`` summarizing this tile's run.

    Raises:
        CapacitorInvariantViolation: propagated uncaught from
            ``verify_capacitor_invariant`` -- a hard failure, not
            swallowed or downgraded to a warning.
        AssertionError: propagated uncaught from ``contract_tile_edges``'s
            mandatory-degree sanity check -- a contraction bug, not a data
            condition.
    """
    tile_x, tile_y = tile_id
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    instance_path = str(Path(source_dir) / f"instanceModels_{tile_x}_{tile_y}.sp")
    source_nd_path = Path(source_dir) / f"tile_{tile_x}_{tile_y}.nd"
    nd_path = str(source_nd_path)

    tile_data = load_tile_data(pkl_dir, tile_id)
    pad_anchors_for_tile = _pad_anchors_in_tile(tile_data, all_pad_anchors)
    current_source_nodes = scan_current_source_nodes(
        instance_path, nd_path, net_filter=net_filter,
    )

    classification = classify_tile(tile_data, pad_anchors_for_tile, current_source_nodes)
    contraction = contract_tile_nodes(classification, ratio=ratio, alpha=alpha)
    edge_result = contract_tile_edges(tile_data, contraction)

    cs_result = sample_current_sources(
        instance_path, contraction.kept_nodes, tile_id,
        nd_path=nd_path, net_filter=net_filter, ratio=ratio, base_seed=base_seed,
    )

    verify_capacitor_invariant(
        tile_data, classification.current_source_nodes, edge_result.kept_capacitive_edges,
    )

    ckt_out = output_dir / f"tile_{tile_x}_{tile_y}.ckt"
    nd_out = output_dir / f"tile_{tile_x}_{tile_y}.nd"
    instance_out = output_dir / f"instanceModels_{tile_x}_{tile_y}.sp"

    write_tile_ckt(
        ckt_out, len(contraction.kept_nodes), classification.boundary_nodes,
        edge_result.kept_resistive_edges, edge_result.kept_capacitive_edges,
    )
    n_nd_lines = write_tile_nd(nd_out, source_nd_path, contraction.kept_nodes)
    write_instance_models(instance_out, cs_result.kept_raw_lines)

    logger.info(
        "Tile %s: wrote %s (%d R + %d C), %s (%d lines), %s (%d current sources)",
        tile_id, ckt_out.name, len(edge_result.kept_resistive_edges),
        len(edge_result.kept_capacitive_edges), nd_out.name, n_nd_lines,
        instance_out.name, len(cs_result.kept_raw_lines),
    )

    return TileProcessingStats(
        tile_id=tile_id,
        original_nodes=classification.total_nodes,
        sampled_nodes=len(contraction.kept_nodes),
        original_resistors=len(tile_data.resistive_edges),
        sampled_resistors=len(edge_result.kept_resistive_edges),
        original_capacitors=len(tile_data.capacitive_edges),
        sampled_capacitors=len(edge_result.kept_capacitive_edges),
        # total_matching_net (not total_eligible) is the true pre-sampling
        # count of all VDD_VAR current-source lines in the raw file --
        # total_eligible already excludes the rare lines whose node_pos
        # isn't in kept_nodes (see sample_current_sources's docstring), so
        # using it here would slightly understate the true "original" count.
        original_current_sources=cs_result.total_matching_net,
        sampled_current_sources=len(cs_result.kept_raw_lines),
        pad_anchor_count=len(pad_anchors_for_tile),
        n_clusters=contraction.n_clusters,
        isolated_optional_dropped=contraction.isolated_optional_dropped,
        islanded_optional_reps_dropped=edge_result.islanded_optional_reps_dropped,
        isolated_mandatory_kept=contraction.isolated_mandatory_kept,
        intra_cluster_resistors_dropped=edge_result.intra_cluster_resistors_dropped,
        parallel_resistors_merged=edge_result.parallel_resistors_merged,
    )


def _process_tile_star(args: Tuple) -> TileProcessingStats:
    """``multiprocessing.Pool.map``-friendly unpacking wrapper for ``process_tile``.

    ``Pool.starmap`` would work directly, but a plain ``map`` + single
    tuple argument plays more predictably with process-pool error
    propagation/pickling across Python versions, so orchestration uses
    this thin wrapper instead.
    """
    return process_tile(*args)


def _init_worker_logging(level: int) -> None:
    """``multiprocessing.Pool`` initializer: configure logging in each worker.

    Worker processes are fresh Python interpreters -- the ``logging.
    basicConfig`` call in ``main()``/the driver process does NOT propagate
    to them (no shared handlers), so without this, every ``logger.info``/
    ``logger.warning`` call made inside ``process_tile`` (and everything it
    calls: ``classify_tile``, ``contract_tile_nodes``, ``contract_tile_edges``,
    etc.) would be silently dropped rather than just unformatted -- Python's
    logging module only auto-prints WARNING+ via a bare "lastResort" handler
    when NO handler is configured at all, and even that is easy to miss
    across many concurrent workers. Matches the same format/datefmt as
    ``main()`` for consistent, greppable log lines regardless of
    ``workers=1`` vs. ``workers>1``.
    """
    logging.basicConfig(
        level=level,
        format='%(asctime)s %(name)s %(levelname)s: %(message)s',
        datefmt='%H:%M:%S',
    )


def write_top_level_files(
    source_dir: Path,
    output_dir: Path,
    metadata: PowerGridMetaData,
    tile_ids: Sequence[Tuple[int, int]],
) -> None:
    """Write the 4 top-level, tile-count-independent output files (plan §4.5).

    Wires together the already-implemented top-level generators
    (``generate_pg_net_voltage``, ``generate_additional_vsrcs``,
    ``generate_ckt_sp``, ``filter_package_ckt``) -- all plain TEXT (NOT
    gzip, verified via ``file`` against the real files), unlike the
    per-tile outputs.

    Args:
        source_dir: Directory with the original ``ckt.sp``/``package.ckt``
            (read for ``.die_area`` and pad-loop text respectively).
        output_dir: Destination directory (created if missing).
        metadata: Loaded ``PowerGridMetaData`` (``vdd``, ``tile_grid``,
            ``net_name``).
        tile_ids: Tile ids in output order, e.g. ``discover_tile_ids(metadata)``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_dir = Path(source_dir)

    net_display = _resolve_net_display_name(source_dir, metadata.net_name)
    (output_dir / 'pg_net_voltage').write_text(
        generate_pg_net_voltage(metadata.vdd, net_name=net_display)
    )
    (output_dir / 'additional_vsrcs').write_text(generate_additional_vsrcs())

    die_area = read_die_area_from_ckt_sp(source_dir / 'ckt.sp')
    ckt_sp_content = generate_ckt_sp(
        tile_grid=metadata.tile_grid, die_area=die_area, vdd=metadata.vdd,
        tile_ids=tile_ids, net_name=net_display,
    )
    (output_dir / 'ckt.sp').write_text(ckt_sp_content)

    package_source_lines = (source_dir / 'package.ckt').read_text().splitlines()
    filtered_package_lines = filter_package_ckt(package_source_lines, net_filter=metadata.net_name)
    (output_dir / 'package.ckt').write_text("\n".join(filtered_package_lines) + "\n")

    logger.info(
        "Wrote top-level files to %s: pg_net_voltage, additional_vsrcs, ckt.sp "
        "(%d tile includes), package.ckt (%d lines)",
        output_dir, len(tile_ids), len(filtered_package_lines),
    )


@dataclass
class SamplingPipelineReport:
    """Aggregate Pass 1-4 + output-write report across all processed tiles (plan §6)."""

    tile_stats: List[TileProcessingStats]
    total_pad_anchors_expected: int
    wall_clock_seconds: float

    def _sum(self, attr: str) -> int:
        return sum(getattr(t, attr) for t in self.tile_stats)

    @property
    def total_original_nodes(self) -> int:
        return self._sum('original_nodes')

    @property
    def total_sampled_nodes(self) -> int:
        return self._sum('sampled_nodes')

    @property
    def total_original_resistors(self) -> int:
        return self._sum('original_resistors')

    @property
    def total_sampled_resistors(self) -> int:
        return self._sum('sampled_resistors')

    @property
    def total_original_capacitors(self) -> int:
        return self._sum('original_capacitors')

    @property
    def total_sampled_capacitors(self) -> int:
        return self._sum('sampled_capacitors')

    @property
    def total_original_current_sources(self) -> int:
        return self._sum('original_current_sources')

    @property
    def total_sampled_current_sources(self) -> int:
        return self._sum('sampled_current_sources')

    @property
    def total_pad_anchors_found(self) -> int:
        return self._sum('pad_anchor_count')

    @property
    def total_n_clusters(self) -> int:
        return self._sum('n_clusters')

    @property
    def total_isolated_optional_dropped(self) -> int:
        return self._sum('isolated_optional_dropped')

    @property
    def total_isolated_mandatory_kept(self) -> int:
        return self._sum('isolated_mandatory_kept')

    @property
    def total_islanded_optional_reps_dropped(self) -> int:
        return self._sum('islanded_optional_reps_dropped')

    @property
    def overall_node_reduction_ratio(self) -> float:
        sampled = self.total_sampled_nodes
        return self.total_original_nodes / sampled if sampled else float('inf')

    @property
    def r_per_node_before(self) -> float:
        """Resistors-per-node in the ORIGINAL (pre-contraction) mesh.

        THE acceptance metric this redesign exists for (§7.5): the
        node-drop design let this collapse from ~1.87 to ~0.72
        (< 1.0 guarantees fragmentation). Contraction should keep
        ``r_per_node_after`` within ~+/-30% of ``r_per_node_before``.
        """
        return (
            self.total_original_resistors / self.total_original_nodes
            if self.total_original_nodes else 0.0
        )

    @property
    def r_per_node_after(self) -> float:
        return (
            self.total_sampled_resistors / self.total_sampled_nodes
            if self.total_sampled_nodes else 0.0
        )

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serializable rollup (for ``sampling_report.json``)."""
        return {
            'total_pad_anchors_expected': self.total_pad_anchors_expected,
            'total_pad_anchors_found': self.total_pad_anchors_found,
            'wall_clock_seconds': self.wall_clock_seconds,
            'totals': {
                'original_nodes': self.total_original_nodes,
                'sampled_nodes': self.total_sampled_nodes,
                'original_resistors': self.total_original_resistors,
                'sampled_resistors': self.total_sampled_resistors,
                'original_capacitors': self.total_original_capacitors,
                'sampled_capacitors': self.total_sampled_capacitors,
                'original_current_sources': self.total_original_current_sources,
                'sampled_current_sources': self.total_sampled_current_sources,
                'overall_node_reduction_ratio': self.overall_node_reduction_ratio,
                'r_per_node_before': self.r_per_node_before,
                'r_per_node_after': self.r_per_node_after,
                'n_clusters': self.total_n_clusters,
                'isolated_optional_dropped': self.total_isolated_optional_dropped,
                'isolated_mandatory_kept': self.total_isolated_mandatory_kept,
                'islanded_optional_reps_dropped': self.total_islanded_optional_reps_dropped,
            },
            'per_tile': [
                {
                    'tile_id': list(t.tile_id),
                    'original_nodes': t.original_nodes,
                    'sampled_nodes': t.sampled_nodes,
                    'original_resistors': t.original_resistors,
                    'sampled_resistors': t.sampled_resistors,
                    'original_capacitors': t.original_capacitors,
                    'sampled_capacitors': t.sampled_capacitors,
                    'original_current_sources': t.original_current_sources,
                    'sampled_current_sources': t.sampled_current_sources,
                    'pad_anchor_count': t.pad_anchor_count,
                    'n_clusters': t.n_clusters,
                    'isolated_optional_dropped': t.isolated_optional_dropped,
                    'isolated_mandatory_kept': t.isolated_mandatory_kept,
                    'islanded_optional_reps_dropped': t.islanded_optional_reps_dropped,
                    'intra_cluster_resistors_dropped': t.intra_cluster_resistors_dropped,
                    'parallel_resistors_merged': t.parallel_resistors_merged,
                }
                for t in sorted(self.tile_stats, key=lambda t: t.tile_id)
            ],
        }

    def format_text(self) -> str:
        """Human-readable rollup, printed to stdout and persisted to ``sampling_report.txt``."""
        lines = [
            "=" * 78,
            "netlist_brcm sampling pipeline -- final report",
            "=" * 78,
            f"Tiles processed: {len(self.tile_stats)}",
            f"Wall clock: {self.wall_clock_seconds:.1f}s",
            "",
            f"Pad anchors: {self.total_pad_anchors_found}/{self.total_pad_anchors_expected} "
            f"found across all tiles"
            + (
                "  <-- MISMATCH" if self.total_pad_anchors_found != self.total_pad_anchors_expected
                else " (OK)"
            ),
            "",
            f"Nodes:            {self.total_original_nodes:>12,} -> {self.total_sampled_nodes:>12,} "
            f"({self.overall_node_reduction_ratio:.2f}x reduction)",
            f"Resistors:        {self.total_original_resistors:>12,} -> {self.total_sampled_resistors:>12,}",
            f"Capacitors:       {self.total_original_capacitors:>12,} -> {self.total_sampled_capacitors:>12,}",
            f"Current sources:  {self.total_original_current_sources:>12,} -> {self.total_sampled_current_sources:>12,}",
            "",
            f"R/node:           {self.r_per_node_before:>12.3f} -> {self.r_per_node_after:>12.3f}"
            + (
                "  <-- BELOW 1.0 (fragmentation risk)" if self.r_per_node_after < 1.0 else ""
            ),
            "",
            f"Contraction diagnostics: clusters={self.total_n_clusters} "
            f"isolated_optional_dropped={self.total_isolated_optional_dropped} "
            f"isolated_mandatory_kept={self.total_isolated_mandatory_kept} "
            f"islanded_reps_dropped={self.total_islanded_optional_reps_dropped}",
            "",
            "Per-tile breakdown:",
            f"{'tile':>10} {'orig_nodes':>12} {'samp_nodes':>12} {'ratio':>7} "
            f"{'pad':>5} {'clusters':>8} {'iso_drop':>8} {'iso_mand':>8}",
        ]
        for t in sorted(self.tile_stats, key=lambda t: t.tile_id):
            ratio = t.original_nodes / t.sampled_nodes if t.sampled_nodes else float('inf')
            lines.append(
                f"{str(t.tile_id):>10} {t.original_nodes:>12,} {t.sampled_nodes:>12,} "
                f"{ratio:>6.2f}x {t.pad_anchor_count:>5} {t.n_clusters:>8} "
                f"{t.isolated_optional_dropped:>8} {t.isolated_mandatory_kept:>8}"
            )
        lines.append("=" * 78)
        return "\n".join(lines) + "\n"


def write_pipeline_report(report: SamplingPipelineReport, output_dir: Path) -> None:
    """Persist *report* as both ``sampling_report.txt`` and ``sampling_report.json``."""
    output_dir = Path(output_dir)
    (output_dir / 'sampling_report.txt').write_text(report.format_text())
    (output_dir / 'sampling_report.json').write_text(json.dumps(report.to_dict(), indent=2))


def run_sampling_pipeline(
    source_dir: Path,
    pkl_dir: Path,
    output_dir: Path,
    ratio: float = 0.1,
    alpha: float = 1.0,
    base_seed: int = 42,
    workers: int = 1,
    log_level: int = logging.INFO,
) -> SamplingPipelineReport:
    """Orchestrate the full Pass 1-4 + output-write pipeline for all tiles (plan §5).

    1. Loads ``metadata.pkl`` once (for ``package_data.die_attachment_nodes``
       -- the global pad-anchor set -- and ``tile_grid``).
    2. Discovers all tile ids (``discover_tile_ids``).
    3. Runs ``process_tile`` for every tile -- sequential when
       ``workers <= 1`` (simplest, easiest to debug: a
       ``CapacitorInvariantViolation`` or any other exception surfaces
       immediately with a clean traceback), or a
       ``multiprocessing.Pool`` when ``workers > 1`` (the plan explicitly
       calls per-tile processing "embarrassingly parallel" -- each tile is
       fully self-contained; a pool trades a little error-message clarity
       for significantly lower wall-clock on this NFS-backed, many-core
       environment. Exceptions raised in a worker (e.g.
       ``CapacitorInvariantViolation``) are still re-raised in the parent
       by ``multiprocessing.Pool.imap``, so this remains a hard pipeline
       failure either way).
    4. Writes the 4 top-level files (``write_top_level_files``).
    5. Aggregates all tiles' ``TileProcessingStats`` into a
       ``SamplingPipelineReport``.

    Args:
        source_dir: Original raw ``netlist_brcm``-shaped directory (READ
            ONLY -- e.g. a symlink into shared external storage).
        pkl_dir: ``distributed_pkl`` directory for *source_dir* (already
            parsed via ``sigma-dvd parse --net VDD_VAR``).
        output_dir: Destination directory for the sampled netlist
            (created if missing -- must NOT be *source_dir* itself).
        ratio: Target retention ratio (default 0.1).
        alpha: Layer-weight exponent for Pass 2 (default 1.0).
        base_seed: Base seed for the deterministic per-tile RNG -- affects
            ONLY Pass 4's current-source down-sampling (Pass 2/3 node
            contraction has no RNG at all).
        workers: Number of worker processes. ``1`` (default) runs fully
            sequential; ``> 1`` uses a ``multiprocessing.Pool`` of that
            size (capped at the tile count).
        log_level: Logging level propagated to worker processes via
            ``_init_worker_logging`` when ``workers > 1`` (workers get
            their own fresh, unconfigured logger otherwise -- see that
            function's docstring). Ignored when ``workers <= 1`` (the
            driver process's own ``logging.basicConfig``, e.g. from
            ``main()``, already applies).

    Returns:
        A ``SamplingPipelineReport`` with per-tile and aggregate stats.
    """
    start = time.monotonic()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata, _boundary_nodes = load_metadata(pkl_dir)
    all_pad_anchors: Set[str] = set(metadata.package_data.die_attachment_nodes)
    tile_ids = discover_tile_ids(metadata)

    logger.info(
        "Starting sampling pipeline: %d tiles, ratio=%.3g, alpha=%.3g, "
        "base_seed=%d, workers=%d -> %s",
        len(tile_ids), ratio, alpha, base_seed, workers, output_dir,
    )

    # Single source of truth for the net across ALL pipeline stages: the
    # metadata's parse-time net name, lowercased for the instance-source
    # filters (matching TileConfig.net_filter's own lowercasing) -- never a
    # hardcoded 'vdd_var'.
    net_filter = metadata.net_name.lower()

    task_args = [
        (source_dir, pkl_dir, output_dir, tile_id, all_pad_anchors, ratio, alpha,
         base_seed, net_filter)
        for tile_id in tile_ids
    ]

    if workers <= 1:
        tile_stats = [process_tile(*args) for args in task_args]
    else:
        n_workers = min(workers, len(tile_ids))
        with multiprocessing.Pool(
            processes=n_workers, initializer=_init_worker_logging, initargs=(log_level,),
        ) as pool:
            tile_stats = list(pool.imap(_process_tile_star, task_args))

    write_top_level_files(source_dir, output_dir, metadata, tile_ids)

    wall_clock = time.monotonic() - start
    report = SamplingPipelineReport(
        tile_stats=tile_stats,
        total_pad_anchors_expected=len(all_pad_anchors),
        wall_clock_seconds=wall_clock,
    )

    logger.info(
        "Sampling pipeline complete in %.1fs: %d tiles, %s",
        wall_clock, len(tile_stats),
        f"{report.total_original_nodes:,} -> {report.total_sampled_nodes:,} nodes "
        f"({report.overall_node_reduction_ratio:.2f}x)",
    )

    return report


DEFAULT_OUTPUT_DIR = Path('netlist/netlist_brcm_sampled')


def main() -> None:
    """CLI entry point.

    Default mode (no ``--sample``): Phase 1 pad-anchor accounting/
    verification pass only -- unchanged from the original CLI behavior.

    ``--sample`` mode: run the full Pass 1-4 + output-write pipeline
    (``run_sampling_pipeline``) across all tiles, writing a complete
    ``netlist/netlist_brcm_sampled/``-shaped directory and a
    ``sampling_report.txt``/``.json`` summary.
    """
    parser_ = argparse.ArgumentParser(
        description=(
            "netlist_brcm sampling pipeline. Default mode loads an existing "
            "distributed_pkl directory and verifies pad-anchor (die_attachment_node) "
            "accounting across tiles. --sample runs the full Pass 1-4 sampling "
            "pipeline and writes a new netlist/netlist_brcm_sampled/-shaped "
            "directory. See netlist_brcm_sampling_plan.md."
        )
    )
    parser_.add_argument(
        '--pkl-dir', type=Path, default=DEFAULT_PKL_DIR,
        help="Directory with metadata.pkl and tile_X_Y.pkl (default: %(default)s)",
    )
    parser_.add_argument('--verbose', action='store_true', help="Enable debug logging")
    parser_.add_argument(
        '--sample', action='store_true',
        help="Run the full Pass 1-4 sampling pipeline (writes a sampled netlist "
             "directory) instead of the default pad-anchor-accounting-only check.",
    )
    parser_.add_argument(
        '--source-dir', type=Path, default=None,
        help="Original raw netlist_brcm-shaped directory (read-only source of "
             "tile_X_Y.nd / instanceModels_X_Y.sp / ckt.sp / package.ckt). "
             "Required with --sample.",
    )
    parser_.add_argument(
        '--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR,
        help="Destination directory for the sampled netlist (--sample only; "
             "default: %(default)s). Must not be --source-dir.",
    )
    parser_.add_argument(
        '--ratio', type=float, default=0.1,
        help="Target retention ratio for node/current-source sampling (--sample "
             "only; default: %(default)s, i.e. ~10x reduction).",
    )
    parser_.add_argument(
        '--alpha', type=float, default=1.0,
        help="Layer-weight exponent for Pass 2 geometric contraction "
             "(--sample only; default: %(default)s).",
    )
    parser_.add_argument(
        '--seed', type=int, default=42,
        help="Base seed for the deterministic per-tile RNG (--sample only; "
             "default: %(default)s). Affects ONLY Pass 4's current-source "
             "down-sampling -- Pass 2/3 node contraction has no RNG.",
    )
    parser_.add_argument(
        '--workers', type=int, default=1,
        help="Number of worker processes for --sample (default: %(default)s, "
             "fully sequential). Values > 1 use a multiprocessing.Pool across "
             "tiles (embarrassingly parallel per-tile work, per plan §5).",
    )
    args = parser_.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s %(name)s %(levelname)s: %(message)s',
        datefmt='%H:%M:%S',
    )

    if args.sample:
        if args.source_dir is None:
            parser_.error("--source-dir is required with --sample")
        if Path(args.output_dir).resolve() == Path(args.source_dir).resolve():
            parser_.error("--output-dir must not be the same as --source-dir")
        if not (0.0 < args.ratio <= 1.0):
            parser_.error(f"--ratio must be in (0, 1], got {args.ratio}")
        if args.workers < 1:
            parser_.error(f"--workers must be >= 1, got {args.workers}")

        report = run_sampling_pipeline(
            source_dir=args.source_dir,
            pkl_dir=args.pkl_dir,
            output_dir=args.output_dir,
            ratio=args.ratio,
            alpha=args.alpha,
            base_seed=args.seed,
            workers=args.workers,
            log_level=log_level,
        )
        print(report.format_text())
        write_pipeline_report(report, args.output_dir)
        return

    metadata, _boundary_nodes = load_metadata(args.pkl_dir)
    per_tile_anchors = compute_pad_anchor_accounting(args.pkl_dir, metadata)
    summary = _summarize(metadata, per_tile_anchors)
    _print_summary(summary)


if __name__ == "__main__":
    main()

"""Tile splitting for load-balanced DDM via recursive geometric bisection.

B1: After _parse_tile_ckt, tiles with n_interior > max_interior are
recursively bisected by node coordinates, producing sub-tile TileData
with new interface nodes at cut edges.

Exactness claim
---------------
DDM is algebraically exact for any partition: the global solution is
mathematically identical regardless of how tiles are split.  In
floating-point arithmetic, floating-point noise grows with the number
of interface (cut) nodes created by splitting:

- One-level bisection (e.g., 15K → 7.5K, max_interior ≈ 8000):
  max|dV| ≲ 2e-14 V (DC, QS, and transient — machine precision).
- Aggressive bisection (e.g., 15K → 4K, max_interior ≈ 3900):
  max|dV| ≲ 6e-8 V (BE) / 6e-6 V (TR) on a 135K-node benchmark.

Splitting is purely a load-balance transform; it does not change the
physics.  The typical FP noise is orders of magnitude below the
integration-method truncation error (BE/TR ~Δt or ~Δt²).

Sub-tile ID scheme
------------------
Parent tile_id ``(x, y)`` yields sub-tiles with 3-tuple IDs ``(x, y, k)``
where k = 0, 1, 2, ... is the sequential leaf index after all bisections.
Filenames and VCS caches derive from tile_id via :func:`_tile_id_str`
(e.g. ``'tile_0_1.pkl'`` → ``'tile_0_1_2.pkl'``).

Coupling-cap-on-cut handling
-----------------------------
Grounded caps (one terminal ``'0'``) cannot cross a spatial cut (ground is
not a spatial node); they always stay with their non-ground node.
Non-grounded coupling caps crossing a candidate cut are detected; the
algorithm tries alternative split points to avoid them.  If no valid split
point exists the tile is returned **unsplit** with a warning rather than
silently corrupting the physics.

Current-injection ownership
----------------------------
A node's current injection goes to exactly the sub-tile that owns the node:

- Nodes in the *left partition* (by coordinate split) → left sub-tile.
- Nodes in the *right partition* → right sub-tile.
- Original boundary nodes (already interface nodes) → left sub-tile as a
  deterministic tiebreaker.

Cut edges (crossing the partition) are placed in the *left* sub-tile.
Right sub-tile sees the cut neighbour as a new boundary (port) node but
does not carry the cut edge — its Schur complement naturally captures all
right-side connections of that node.

Termination guarantee
---------------------
If no valid bisection point exists for a tile (all candidates cut through
coupling caps or produce 0-interior partitions), :func:`split_tile` returns
the tile unchanged and the caller logs a warning via ``n_tiles_over_max``.
Callers (e.g. :func:`parser._apply_tile_splits`) must aggregate and surface
this count so oversized tiles are not silently ignored.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from .tile_parsing import TileData, _parse_node_xy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Parent-level island detection (pre-cleaning before split)
# ---------------------------------------------------------------------------

def _pre_clean_tile_data(
    tile_data: TileData,
    port_nodes: Set[str],
    min_interface_keep: int = 5,
    pad_nodes: Optional[Set[str]] = None,
    mark_split_pre_cleaned: bool = False,
) -> Tuple[int, List[Dict[str, Any]]]:
    """Remove genuinely floating components from *tile_data* in-place.

    Always sets ``tile_data.pre_cleaned_full = True`` (Stage 1e: signals that
    a COMPLETE removal pass, at whatever threshold/port-set this call used,
    has already run on this exact tile -- see
    ``tile_worker._build_block_system``).

    ``tile_data.pre_cleaned`` (the *legacy* split-path flag that selects
    threshold=1 vs threshold=5 in ``TileWorker._remove_floating_islands``'s
    fallback path) is set ONLY when *mark_split_pre_cleaned* is True (Stage
    1e finding F3).  Whole-tile parse-time cleans (the universal pre-clean
    that now runs on every tile, not just tiles about to be split) must NOT
    set it -- doing so unconditionally flips the legacy-fallback worker
    removal threshold from 5 to 1 for every never-split tile, corrupting the
    ``island_detection='schur_bfs'`` / trust-assertion-failure fallback path.
    Only the genuine post-split sub-tile pass (``_apply_tile_splits``, called
    with ``min_interface_keep=1``) passes ``mark_split_pre_cleaned=True``.

    Args:
        tile_data: TileData to clean in-place.
        port_nodes: Global interface-node candidates visible to this tile
            (``shared_boundary_nodes ∪ die_attachment_nodes``, intersected
            with ``tile_data.all_nodes`` below).
        min_interface_keep: Minimum interface-node count for a non-largest
            component to be kept (matches ``TileWorker.MIN_INTERFACE_NODES_KEEP``
            for whole tiles at parse time; callers pass 1 for sub-tiles,
            matching ``TileWorker.MIN_INTERFACE_NODES_KEEP_PRE_CLEANED``).
        pad_nodes: Global Dirichlet (voltage-source) node names.  Used only to
            compute the ``has_pad`` flag on each kept-component summary
            (Stage 1e connectivity summaries) -- never affects removal.
        mark_split_pre_cleaned: If True, also set ``tile_data.pre_cleaned =
            True`` (the legacy split-path fallback-threshold flag).  Pass
            True ONLY from the post-split sub-tile pass.

    Returns:
        ``(n_removed, component_summaries)``.  ``n_removed`` is the number of
        floating components removed.  ``component_summaries`` is a list of
        dicts, one per **kept** component with non-empty interface-candidate
        overlap: ``{'candidates': frozenset[str], 'n_nodes': int,
        'has_pad': bool, 'tile_id': tuple, 'is_largest': bool,
        'keep_threshold': int}``.  Removed components and components with
        zero interface-candidate overlap are omitted (they cannot
        participate in the coordinator-side interface union-find either
        way).  ``is_largest``/``keep_threshold`` (finding R1) let the parser's
        parse-end consistency re-check (``tile_parsing.
        verify_component_keep_decisions``) distinguish the unconditionally-
        kept largest component from a component kept only because it met
        *min_interface_keep* candidates AT THIS PASS -- a decision that a
        later, coordinator-side interface-set recompute can invalidate.
    """
    from .tile_parsing import _decompose_and_remove_floating

    port_nodes_local: Set[str] = port_nodes & tile_data.all_nodes
    pad_nodes_local: Set[str] = (pad_nodes & tile_data.all_nodes) if pad_nodes else set()

    def _summarize(comp: Set[str], is_largest_comp: bool) -> Optional[Dict[str, Any]]:
        candidates = comp & port_nodes_local
        if not candidates:
            return None
        return {
            'candidates': frozenset(candidates),
            'n_nodes': len(comp),
            'has_pad': bool(comp & pad_nodes_local),
            'tile_id': tile_data.tile_id,
            'is_largest': is_largest_comp,
            'keep_threshold': min_interface_keep,
        }

    if not port_nodes_local:
        if mark_split_pre_cleaned:
            tile_data.pre_cleaned = True
        tile_data.pre_cleaned_full = True
        return 0, []

    components, largest, _kept_iface, removed_nodes, n_removed = _decompose_and_remove_floating(
        tile_data, port_nodes_local, min_interface_keep,
    )

    component_summaries: List[Dict[str, Any]] = []
    for comp in components:
        if comp & removed_nodes:
            continue  # this whole component was removed (removal is atomic per component)
        summary = _summarize(comp, comp is largest)
        if summary is not None:
            component_summaries.append(summary)

    if removed_nodes:
        logger.info(
            "Tile %s: pre_clean removed %d floating component(s) (%d nodes total)",
            tile_data.tile_id, n_removed, len(removed_nodes),
        )

    if mark_split_pre_cleaned:
        tile_data.pre_cleaned = True
    tile_data.pre_cleaned_full = True
    return n_removed, component_summaries


# ---------------------------------------------------------------------------
# Public helper: filename slug from any-length tile_id
# ---------------------------------------------------------------------------

def _tile_id_str(tile_id: tuple) -> str:
    """Convert tile_id (any-length tuple) to a ``'_'``-joined string.

    Used for deterministic filename generation that works for both original
    2-tuple tile IDs and sub-tile 3-tuple IDs.

    Examples::

        _tile_id_str((0, 1))    # -> '0_1'
        _tile_id_str((0, 1, 2)) # -> '0_1_2'
    """
    return '_'.join(str(c) for c in tile_id)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def split_tile(
    tile_data: TileData,
    max_interior: int,
    alpha: float = 0.5,
) -> List[TileData]:
    """Split a parsed tile into sub-tiles by recursive geometric bisection.

    DDM is algebraically exact for any partition; splitting is purely a
    load-balance transform.  In floating-point arithmetic, noise grows with
    the number of interface (cut) nodes created — typically ≤ 2e-14 V for
    one-level bisections, up to ~60 nV (BE) / ~6 µV (TR) for very aggressive
    four-level splits.

    The balance metric weights both factor cost (interior node count) and
    RHS cost (current-source count) via the *alpha* parameter:
    ``weight(node) = 1 + alpha * (1 if node has a current source else 0)``.

    Args:
        tile_data: Parsed tile to potentially split.  The object is not
            modified; new TileData instances are created.
        max_interior: Maximum interior node count per sub-tile.  Tiles at
            or below this threshold are returned as-is.
        alpha: Balance weight for current-source nodes relative to
            non-source interior nodes (default 0.5).

    Returns:
        ``[tile_data]`` if no split is needed or a valid cut could not be
        found; otherwise a list of 2+ TileData sub-tiles with 3-tuple IDs
        ``(parent_x, parent_y, k)``.
    """
    interior = tile_data.all_nodes - tile_data.boundary_nodes
    if len(interior) <= max_interior:
        return [tile_data]

    leaves = _split_recursive(tile_data, max_interior, alpha, _depth=0)
    if len(leaves) == 1:
        return [tile_data]  # kept original object

    # Assign sequential 3-tuple IDs derived from parent
    px, py = tile_data.tile_id[0], tile_data.tile_id[1]
    # Propagate pre_cleaned flag: if the parent was pre-cleaned before splitting,
    # sub-tiles inherit the flag so TileWorker uses threshold=1 for island detection.
    parent_pre_cleaned = tile_data.pre_cleaned
    result: List[TileData] = []
    for k, leaf in enumerate(leaves):
        result.append(TileData(
            tile_id=(px, py, k),
            resistive_edges=leaf.resistive_edges,
            all_nodes=leaf.all_nodes,
            boundary_nodes=leaf.boundary_nodes,
            current_injections=leaf.current_injections,
            capacitive_edges=leaf.capacitive_edges,
            pre_cleaned=parent_pre_cleaned,
        ))
    return result


# ---------------------------------------------------------------------------
# Recursive bisection
# ---------------------------------------------------------------------------

def _split_recursive(
    tile_data: TileData,
    max_interior: int,
    alpha: float,
    _depth: int,
) -> List[TileData]:
    """Recursively bisect until all leaves have ≤ max_interior nodes."""
    interior = tile_data.all_nodes - tile_data.boundary_nodes
    if len(interior) <= max_interior or _depth >= 20:
        return [tile_data]

    halves = _bisect_once(tile_data, alpha)
    if halves is None:
        # No valid cut found — return tile unchanged
        logger.warning(
            "Tile %s: cannot find a valid bisection point "
            "(n_interior=%d > max_interior=%d); tile returned unsplit",
            tile_data.tile_id, len(interior), max_interior,
        )
        return [tile_data]

    left, right = halves
    return (
        _split_recursive(left, max_interior, alpha, _depth + 1)
        + _split_recursive(right, max_interior, alpha, _depth + 1)
    )


# ---------------------------------------------------------------------------
# Single bisection step — helpers
# ---------------------------------------------------------------------------

def _try_axis_split(
    tile_data: TileData,
    node_coords: Dict[str, Tuple[float, float]],
    coupling_cap_pairs: List[Tuple[str, str]],
    has_current: Set[str],
    split_axis: int,
    alpha: float,
    adj: Optional[Dict[str, Set[str]]] = None,
) -> Optional[Tuple[TileData, TileData]]:
    """Attempt bisection candidates along *split_axis*.

    Tries the weighted-median candidate and nearby offsets / quartile points.
    Rejects splits that:
    * are degenerate (one side empty), or
    * cut through coupling caps, or
    * yield a sub-tile with 0 interior nodes after :func:`_build_halves`.

    Args:
        tile_data: Tile to split.
        node_coords: Coordinate dict for interior nodes with parseable coords.
        coupling_cap_pairs: Non-grounded cap pairs (forbidden to cross cut).
        has_current: Set of interior nodes that carry a current injection.
        split_axis: 0=x, 1=y.
        alpha: Balance weight for current-source nodes.
        adj: Precomputed resistive adjacency (excluding ground).  When
            provided, used for a cheap 0-interior pre-check before calling
            the full :func:`_build_halves`.  Significant speedup on large
            tiles (millions of interior nodes) where the full build is O(E).

    Returns ``(left_tile, right_tile)`` on the first valid cut, ``None``
    if every candidate fails.
    """
    # Sort by (split_axis_coord, node_name) for full determinism
    by_coord = sorted(
        node_coords.items(),
        key=lambda kv: (kv[1][split_axis], kv[0]),
    )

    n = len(by_coord)
    if n < 2:
        return None

    weights = [
        (1.0 + alpha if name in has_current else 1.0)
        for name, _ in by_coord
    ]
    total_weight = float(sum(weights))
    target = total_weight / 2.0

    cum = 0.0
    median_idx = 0
    for i, w in enumerate(weights):
        cum += w
        if cum >= target:
            median_idx = i
            break

    def _cut_val(idx: int) -> float:
        """Mid-point between by_coord[idx] and by_coord[idx+1]."""
        c_left = by_coord[idx][1][split_axis]
        c_right = by_coord[min(idx + 1, n - 1)][1][split_axis]
        if idx >= n - 1:
            return c_left + 1.0  # beyond all nodes
        return (c_left + c_right) / 2.0

    # ── Build candidate split indices ─────────────────────────────────────
    # Priority order:
    #  1. Weighted-median and nearby offsets (best balance)
    #  2. Quartile points (gross asymmetry fallback)
    #  3. Transition-point sweep: every index where the sorted coordinate
    #     VALUE changes.  This replaces the old O(n) exhaustive sweep with
    #     an O(distinct_values) scan — for large tiles with many nodes
    #     sharing the same coordinate (e.g., via-stacks in upper metals),
    #     the number of distinct values is far smaller than n-1.
    #     For tiles with n ≤ _SMALL_TILE_THRESHOLD we fall back to the full
    #     exhaustive sweep for reliability on tiny, highly-clustered tiles.
    _SMALL_TILE_THRESHOLD = 1000

    seen: Set[int] = set()
    candidates: List[int] = []

    def _add(idx: int) -> None:
        if 0 <= idx < n - 1 and idx not in seen:
            candidates.append(idx)
            seen.add(idx)

    _add(median_idx)
    for delta in (-1, 1, -2, 2, -5, 5, -10, 10):
        _add(median_idx + delta)
    for frac in (0.25, 0.75, 0.1, 0.9):
        _add(int(frac * (n - 1)))

    if n <= _SMALL_TILE_THRESHOLD:
        # Small tiles: exhaustive sweep (original behaviour, fast enough)
        for idx in range(n - 1):
            _add(idx)
    else:
        # Large tiles: only add candidates at distinct coordinate transitions.
        # A cut between two nodes with identical coordinate values produces
        # the same partition as the transition immediately preceding them,
        # so we skip redundant positions.
        for idx in range(n - 1):
            if by_coord[idx][1][split_axis] != by_coord[idx + 1][1][split_axis]:
                _add(idx)

    for split_idx in candidates:
        if split_idx < 0 or split_idx >= n - 1:
            continue
        cut_v = _cut_val(split_idx)

        left_set: Set[str] = set()
        right_set: Set[str] = set()
        for node, coord in by_coord:
            if coord[split_axis] <= cut_v:
                left_set.add(node)
            else:
                right_set.add(node)

        if not left_set or not right_set:
            continue  # degenerate split

        # Check coupling-cap crossing (cheap: O(coupling_caps) per candidate)
        cap_cut = any(
            (u in left_set) != (v in left_set)
            for u, v in coupling_cap_pairs
            if u in left_set or u in right_set or v in left_set or v in right_set
        )
        if cap_cut:
            logger.debug(
                "Tile %s: axis=%d split_idx=%d cuts through a coupling cap; skipping",
                tile_data.tile_id, split_axis, split_idx,
            )
            continue

        # Cheap 0-interior pre-check using precomputed adjacency.
        # A right-side node is a cut_right_guest (becomes boundary, not
        # interior) iff it has at least one left-side neighbour.  The right
        # sub-tile has 0 interior nodes iff every right-set node has a left
        # neighbour.  We can check this cheaply from adj without calling
        # the full O(E) _build_halves.
        #
        # Safety: this pre-check only considers COORDINATED interior nodes
        # (those in right_set = node_coords interior with x,y).  If there
        # are uncoordinated interior nodes, they might still provide right
        # interior even when all coordinated right nodes are cut_right_guests.
        # We therefore skip the pre-check when uncoordinated nodes exist so
        # we never false-skip a valid candidate.
        if adj is not None:
            orig_boundary = tile_data.boundary_nodes
            all_coordinated = set(node_coords.keys())
            has_uncoordinated = bool(
                (tile_data.all_nodes - orig_boundary - all_coordinated - {'0'})
            )
            if not has_uncoordinated:
                right_has_interior = any(
                    not any(
                        nb in left_set
                        for nb in adj.get(node, ())
                        if nb not in orig_boundary
                    )
                    for node in right_set
                )
                if not right_has_interior:
                    logger.debug(
                        "Tile %s: axis=%d split_idx=%d pre-check: 0-interior right; skipping",
                        tile_data.tile_id, split_axis, split_idx,
                    )
                    continue

        # Build sub-tiles and check for 0-interior (degenerate partition).
        # A 0-interior sub-tile arises when every coordinated right-side node
        # is adjacent to a left-side node (all become cut_right_guests →
        # right interior is empty).  Skip and try the next candidate.
        left_tile, right_tile = _build_halves(
            tile_data, left_set, right_set, node_coords, split_axis
        )
        left_n_interior = len(left_tile.all_nodes - left_tile.boundary_nodes)
        right_n_interior = len(right_tile.all_nodes - right_tile.boundary_nodes)
        if left_n_interior == 0 or right_n_interior == 0:
            logger.debug(
                "Tile %s: axis=%d split_idx=%d yields 0-interior sub-tile "
                "(left_n=%d, right_n=%d); skipping",
                tile_data.tile_id, split_axis, split_idx,
                left_n_interior, right_n_interior,
            )
            continue

        return left_tile, right_tile

    return None


# ---------------------------------------------------------------------------
# Single bisection step
# ---------------------------------------------------------------------------

def _bisect_once(
    tile_data: TileData,
    alpha: float,
) -> Optional[Tuple[TileData, TileData]]:
    """Bisect tile_data into two halves using geometric coordinate splitting.

    Tries the longer bounding-box axis first (primary), then the perpendicular
    axis (secondary) if the primary fails.  Returns ``(left_tile, right_tile)``
    or ``None`` if no valid cut exists on either axis.

    The halves retain the parent tile_id (callers assign final IDs).

    Perpendicular-axis fallback
    ---------------------------
    Coarse-metal / stripe-dominated PDN tiles often have many interior nodes
    sharing the same coordinate value on the primary axis.  In that case all
    median-vicinity candidates yield an empty partition on that axis, and the
    algorithm retries the perpendicular axis before giving up.

    Adjacency precomputation
    ------------------------
    The resistive adjacency (excluding ground) is built ONCE here and passed
    to both :func:`_try_axis_split` calls.  This allows the cheap 0-interior
    pre-check to avoid the O(E) :func:`_build_halves` call for obviously-
    degenerate partitions, and keeps the adjacency build cost at O(E) total
    rather than O(candidates × E).
    """
    orig_boundary = tile_data.boundary_nodes
    interior = tile_data.all_nodes - orig_boundary

    # ── 1. Parse spatial coordinates ──────────────────────────────────────
    node_coords: Dict[str, Tuple[float, float]] = {}
    for node in interior:
        if node == '0':
            continue
        x, y = _parse_node_xy(node)
        if x is not None:
            node_coords[node] = (x, y)

    if not node_coords:
        logger.debug(
            "Tile %s: no parseable coordinates in interior nodes; cannot bisect",
            tile_data.tile_id,
        )
        return None

    # ── 2. Determine primary and secondary axes ────────────────────────────
    xs = [c[0] for c in node_coords.values()]
    ys = [c[1] for c in node_coords.values()]
    x_range = max(xs) - min(xs)
    y_range = max(ys) - min(ys)
    primary_axis = 0 if x_range >= y_range else 1
    alt_axis = 1 - primary_axis
    alt_range = y_range if primary_axis == 0 else x_range

    # ── 3. Coupling-cap detection ──────────────────────────────────────────
    # Coupling caps (both terminals non-ground) that cross a cut are forbidden
    # in the current tile-cap model.  Detect which node pairs are coupled.
    coupling_cap_pairs: List[Tuple[str, str]] = [
        (u, v)
        for u, v, _c in tile_data.capacitive_edges
        if u != '0' and v != '0'
    ]

    # ── 4. Common weight set (axis-independent) ────────────────────────────
    has_current: Set[str] = set(tile_data.current_injections.keys())

    # ── 5. Precompute resistive adjacency (excluding ground) ───────────────
    # Built once; shared across both axis attempts so the cheap 0-interior
    # pre-check in _try_axis_split does not rebuild it per-candidate.
    adj: Dict[str, Set[str]] = {}
    for u, v, _g in tile_data.resistive_edges:
        if u != '0' and v != '0':
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)

    # ── 6. Try primary axis ────────────────────────────────────────────────
    result = _try_axis_split(
        tile_data, node_coords, coupling_cap_pairs, has_current,
        primary_axis, alpha, adj=adj,
    )
    if result is not None:
        return result

    # ── 7. Perpendicular-axis fallback ─────────────────────────────────────
    # Only attempt when the perpendicular axis has actual coordinate spread;
    # if both ranges are 0 the tile cannot be bisected geometrically.
    if alt_range > 0:
        logger.debug(
            "Tile %s: primary axis %d failed (x_range=%.1f, y_range=%.1f); "
            "retrying on perpendicular axis %d",
            tile_data.tile_id, primary_axis, x_range, y_range, alt_axis,
        )
        result = _try_axis_split(
            tile_data, node_coords, coupling_cap_pairs, has_current,
            alt_axis, alpha, adj=adj,
        )
        if result is not None:
            return result

    logger.warning(
        "Tile %s: all candidate split points on both axes failed "
        "(coupling caps or 0-interior); tile returned unsplit",
        tile_data.tile_id,
    )
    return None


# ---------------------------------------------------------------------------
# Build two half-TileData from a partition of coordinated interior nodes
# ---------------------------------------------------------------------------

def _build_halves(
    tile_data: TileData,
    left_coordinated: Set[str],
    right_coordinated: Set[str],
    node_coords: Dict[str, Tuple[float, float]],
    split_axis: int,
) -> Tuple[TileData, TileData]:
    """Construct left and right TileData from a bisection.

    Args:
        tile_data: Original tile.
        left_coordinated: Coordinated interior nodes assigned to left.
        right_coordinated: Coordinated interior nodes assigned to right.
        node_coords: Coordinate dict (all coordinated interior nodes).
        split_axis: 0=x, 1=y (for uncoordinated-node fallback heuristic).

    Returns:
        ``(left_tile, right_tile)`` both with the parent's tile_id.
    """
    orig_boundary = tile_data.boundary_nodes

    # ── Assign uncoordinated interior nodes ───────────────────────────────
    uncoordinated = (
        tile_data.all_nodes
        - orig_boundary
        - left_coordinated
        - right_coordinated
        - {'0'}
    )

    left_interior = set(left_coordinated)
    right_interior = set(right_coordinated)

    if uncoordinated:
        _assign_uncoordinated(
            uncoordinated,
            left_interior,
            right_interior,
            tile_data.resistive_edges,
        )

    # ── Edge partition ─────────────────────────────────────────────────────
    # An edge (u, v) goes to LEFT if:
    #   u ∈ left_interior  OR  (u, v both in orig_boundary)
    # An edge (u, v) goes to RIGHT if:
    #   v ∈ right_interior  OR  u ∈ right_interior
    #   AND  neither u nor v is in left_interior
    left_edges = []
    right_edges = []

    for u, v, g in tile_data.resistive_edges:
        u_gnd = (u == '0')
        v_gnd = (v == '0')
        u_left = u in left_interior
        v_left = v in left_interior
        u_right = u in right_interior
        v_right = v in right_interior
        u_bnd = u in orig_boundary
        v_bnd = v in orig_boundary

        # ── Ground edges: (node, '0') or ('0', node) ──────────────────────
        # Assign by ownership of the non-ground node.
        # Ownership: left_interior/orig_boundary → left; right_interior → right.
        if u_gnd or v_gnd:
            non_gnd = v if u_gnd else u
            ng_left = (non_gnd in left_interior) or (non_gnd in orig_boundary)
            if ng_left:
                left_edges.append((u, v, g))
            else:
                right_edges.append((u, v, g))
            continue

        # Cut edges: one endpoint each side → LEFT (left hosts the cut edge)
        if (u_left and v_right) or (u_right and v_left):
            left_edges.append((u, v, g))
            continue

        # Both in left or left+boundary → LEFT
        if (u_left or u_bnd) and (v_left or v_bnd):
            # Avoid bnd-bnd edges appearing in both sub-tiles when both
            # bnd nodes also appear in right tile.  Assign to LEFT only.
            left_edges.append((u, v, g))
            continue

        # Both in right or right+boundary (no left_interior involved) → RIGHT
        if (u_right or u_bnd) and (v_right or v_bnd) and not u_left and not v_left:
            right_edges.append((u, v, g))
            continue

        # Remaining: unmapped node — skip (shouldn't occur in well-formed tiles)

    # ── Identify cut endpoints ─────────────────────────────────────────────
    # Cut nodes on left side that now have a right-side neighbour
    cut_left_endpoints: Set[str] = set()
    # Right-side nodes that appear as guests in left tile (from cut edges)
    cut_right_guests: Set[str] = set()

    for u, v, g in tile_data.resistive_edges:
        if (u in left_interior and v in right_interior):
            cut_left_endpoints.add(u)
            cut_right_guests.add(v)
        elif (u in right_interior and v in left_interior):
            cut_left_endpoints.add(v)
            cut_right_guests.add(u)

    # ── all_nodes ──────────────────────────────────────────────────────────
    left_all = left_interior | cut_right_guests | orig_boundary
    right_all = right_interior | orig_boundary

    # ── boundary_nodes ─────────────────────────────────────────────────────
    # DDM correctness: cut_left_endpoints (left-interior nodes adjacent to cut)
    # remain INTERIOR in the left tile — they are eliminated in S_left, and their
    # coupling to cut_right_guests flows through G_ip/G_pi into the Schur complement.
    # Only cut_right_guests (which appear in BOTH sub-tiles) become new interface nodes.
    # compute_shared_boundary_nodes will find them shared (in 2+ tile boundary sets).
    left_bnd = cut_right_guests | orig_boundary
    right_bnd = cut_right_guests | orig_boundary  # cut_right_guests promoted to boundary

    # ── current_injections ────────────────────────────────────────────────
    # Left: left_interior nodes (their home).
    # Right: right_interior nodes (their home; includes cut_right_guests).
    # orig_boundary nodes: determine which sub-tile has their edges and
    # assign the current injection there to avoid silent drops.  If a node
    # has edges in both sub-tiles (or neither), left wins as a tiebreaker.
    orig_bnd_in_left: Set[str] = set()
    orig_bnd_in_right: Set[str] = set()
    for eu, ev, _eg in left_edges:
        if eu in orig_boundary:
            orig_bnd_in_left.add(eu)
        if ev in orig_boundary:
            orig_bnd_in_left.add(ev)
    for eu, ev, _eg in right_edges:
        if eu in orig_boundary:
            orig_bnd_in_right.add(eu)
        if ev in orig_boundary:
            orig_bnd_in_right.add(ev)

    left_currents = {
        n: v
        for n, v in tile_data.current_injections.items()
        if n in left_interior
        or (n in orig_boundary
            and (n in orig_bnd_in_left or n not in orig_bnd_in_right))
    }
    right_currents = {
        n: v
        for n, v in tile_data.current_injections.items()
        if n in right_interior
        or (n in orig_boundary
            and n in orig_bnd_in_right and n not in orig_bnd_in_left)
    }

    # ── capacitive_edges ─────────────────────────────────────────────────
    # Pass orig_bnd_in_left/right so _split_caps can mirror the tiebreaker
    # used by current_injections: orig_boundary nodes whose R edges live
    # ONLY in the right sub-tile must have their caps go to right as well,
    # otherwise the cap lands in the left tile_data.capacitive_edges but the
    # node is absent from the left BlockMatrixSystem → cap silently dropped.
    left_caps = _split_caps(
        tile_data.capacitive_edges,
        left_interior,
        right_interior,
        orig_boundary,
        assign_to_left=True,
        orig_bnd_in_left=orig_bnd_in_left,
        orig_bnd_in_right=orig_bnd_in_right,
    )
    right_caps = _split_caps(
        tile_data.capacitive_edges,
        left_interior,
        right_interior,
        orig_boundary,
        assign_to_left=False,
        orig_bnd_in_left=orig_bnd_in_left,
        orig_bnd_in_right=orig_bnd_in_right,
    )

    # Keep only edges whose both endpoints are in the respective all_nodes set
    left_edges = [
        (u, v, g) for u, v, g in left_edges
        if u in left_all or u == '0'
        if v in left_all or v == '0'
    ]
    right_edges = [
        (u, v, g) for u, v, g in right_edges
        if u in right_all or u == '0'
        if v in right_all or v == '0'
    ]

    left_tile = TileData(
        tile_id=tile_data.tile_id,  # callers reassign
        resistive_edges=left_edges,
        all_nodes=left_all,
        boundary_nodes=left_bnd,
        current_injections=left_currents,
        capacitive_edges=left_caps,
        pre_cleaned=tile_data.pre_cleaned,  # propagate parent flag
    )
    right_tile = TileData(
        tile_id=tile_data.tile_id,  # callers reassign
        resistive_edges=right_edges,
        all_nodes=right_all,
        boundary_nodes=right_bnd,
        current_injections=right_currents,
        capacitive_edges=right_caps,
        pre_cleaned=tile_data.pre_cleaned,  # propagate parent flag
    )
    return left_tile, right_tile


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _assign_uncoordinated(
    uncoordinated: Set[str],
    left_interior: Set[str],
    right_interior: Set[str],
    resistive_edges: list,
) -> None:
    """BFS-assign uncoordinated interior nodes to left or right partition.

    Modifies *left_interior* and *right_interior* in-place.

    Strategy:
    1. Build adjacency from resistive_edges (ground edges excluded).
    2. Sort uncoordinated nodes deterministically.
    3. For each uncoordinated node, look at how many coordinated neighbours
       are in left vs right.  Assign to the majority.  Ties go to left.
    4. Iterate until stable (cascading assignments possible).
    5. Orphans (no coordinated neighbours found) → left.
    """
    adj: Dict[str, Set[str]] = {}
    for u, v, _g in resistive_edges:
        if u != '0' and v != '0':
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)

    remaining = set(uncoordinated)
    changed = True
    while changed and remaining:
        changed = False
        for node in sorted(remaining):  # sorted = deterministic
            left_nb = sum(1 for nb in adj.get(node, []) if nb in left_interior)
            right_nb = sum(1 for nb in adj.get(node, []) if nb in right_interior)
            if left_nb == 0 and right_nb == 0:
                continue  # no resolved neighbours yet
            if left_nb >= right_nb:
                left_interior.add(node)
            else:
                right_interior.add(node)
            remaining.discard(node)
            changed = True

    # Orphans (no neighbours in either partition) → left
    for node in sorted(remaining):
        left_interior.add(node)


def _split_caps(
    cap_edges: list,
    left_interior: Set[str],
    right_interior: Set[str],
    orig_boundary: Set[str],
    assign_to_left: bool,
    orig_bnd_in_left: Optional[Set[str]] = None,
    orig_bnd_in_right: Optional[Set[str]] = None,
) -> list:
    """Assign capacitive edges to one sub-tile.

    Ownership rule:
    - Grounded cap ``(node, '0', c)`` → sub-tile that owns *node*.
    - Coupling cap ``(u, v, c)`` → sub-tile that owns the majority endpoint.
    - orig_boundary nodes → left sub-tile by default (tiebreaker), BUT if
      ``orig_bnd_in_left``/``orig_bnd_in_right`` are provided, the node
      follows the sub-tile whose resistive edges contain it.  A node that
      appears only in right sub-tile resistive edges must have its cap in
      the right sub-tile; otherwise the cap ends up in left tile_data but
      the node is absent from the left BlockMatrixSystem → cap silently
      dropped from the transient solver.

    Args:
        assign_to_left: If True, return caps assigned to left; else right.
        orig_bnd_in_left: Set of orig_boundary nodes that appear in left
            sub-tile resistive edges (computed in ``_build_halves``).
        orig_bnd_in_right: Set of orig_boundary nodes that appear in right
            sub-tile resistive edges.
    """
    result = []
    for u, v, c in cap_edges:
        if u == '0' or v == '0':
            # Grounded cap: ownership by non-ground node
            non_gnd = v if u == '0' else u
            if non_gnd in left_interior:
                is_left = True
            elif non_gnd in orig_boundary:
                if orig_bnd_in_left is not None and orig_bnd_in_right is not None:
                    # Mirror current_injections tiebreaker: left wins unless the
                    # node is exclusively in right sub-tile resistive edges.
                    is_left = (
                        non_gnd in orig_bnd_in_left
                        or non_gnd not in orig_bnd_in_right
                    )
                else:
                    is_left = True  # legacy: orig_boundary → left
            else:
                is_left = False  # right_interior
            if is_left == assign_to_left:
                result.append((u, v, c))
        else:
            # Coupling cap: should not cross cut (caller ensured this);
            # assign by whether u is left-owned.
            is_left = (u in left_interior) or (u in orig_boundary and v not in right_interior)
            if is_left == assign_to_left:
                result.append((u, v, c))
    return result

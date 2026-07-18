"""Tile data structures and stateless parsing functions.

Contains TileData, unit conversion constants, and all stateless parsing
helpers for tile .ckt / .nd / instanceModels files.

Split from tile_worker.py for maintainability. All public names are
re-exported from tile_worker so existing imports keep working.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple



def _parse_node_xy(node: str) -> Tuple[Optional[float], Optional[float]]:
    """Extract ``(x, y)`` from a PDN node name ``<x>_<y>_<layer>``.

    PDN node names encode coordinates as ``'1000_2000_M1'``.  This helper
    splits on ``'_'`` and attempts to convert the first two parts to floats.

    Returns ``(None, None)`` when parsing fails.
    """
    parts = str(node).split('_')
    if len(parts) >= 2:
        try:
            return float(parts[0]), float(parts[1])
        except ValueError:
            pass
    return None, None

# Unit conversions matching pdn_parser.py
R_TO_KOHM = 1e-3  # Ohm to kOhm
C_TO_FF = 1e15  # Farad to femtoFarad
I_TO_MA = 1e3  # Ampere to mA


@dataclass
class TileData:
    """Parsed tile data. Serializable for Ray."""

    tile_id: Tuple[int, int]
    resistive_edges: List[Tuple[str, str, float]]  # (u, v, conductance_mS)
    all_nodes: Set[str]
    boundary_nodes: Set[str]
    current_injections: Dict[str, float]  # node -> current in mA (positive = sink)
    capacitive_edges: List[Tuple[str, str, float]] = field(default_factory=list)  # (u, v, C_fF)
    pre_cleaned: bool = False  # True if island detection ran at parent level before splitting
    # Stage 1e: True if _pre_clean_tile_data() has already run a COMPLETE
    # island-removal pass on this exact tile using the CURRENT global port
    # set semantics (threshold=5 for whole tiles, threshold=1 for post-split
    # sub-tiles).  Distinct from `pre_cleaned` (which only selects the
    # sub-tile threshold at *worker* removal time) -- when this flag is True
    # AND the model-creation trust assertion holds, TileWorker._build_block_system
    # skips _remove_floating_islands entirely (see tile_worker.py).  Old pickled
    # bundles predating this field simply lack it in __dict__ after unpickling;
    # always access via getattr(tile_data, 'pre_cleaned_full', False), never
    # tile_data.pre_cleaned_full directly.
    pre_cleaned_full: bool = False
    # --- Stage 1e findings F8/F9: persisted removal diagnostics -------------
    # Populated by _decompose_and_remove_floating() (below), called from BOTH
    # retile._pre_clean_tile_data (parse-time) and TileWorker._remove_floating_
    # islands (legacy worker-time fallback removal) -- whichever ran most
    # recently.  Unioned/accumulated (never overwritten) across repeat calls
    # so a parent-level whole-tile pass followed by a sub-tile pass both
    # contribute.  Tile-local (persisted in this tile's own .pkl, NOT
    # metadata.pkl) so the floating-nodes report and island_stats can recover
    # exactly what legacy _remove_floating_islands would have reported, even
    # under Stage 1e summaries-mode trust (worker-side removal skipped).
    # Old pickled bundles predating these fields simply lack them in __dict__
    # after unpickling; always access via getattr(tile_data, <name>, <default>).
    #
    # Finding R9: named ``*_at_parse`` originally, but these fields are also
    # mutated at WORKER (model-creation) time by the legacy-fallback removal
    # path (TileWorker._remove_floating_islands, via the shared
    # _decompose_and_remove_floating core) -- "_at_parse" misstates what the
    # field holds once a legacy-fallback model has run.  Renamed to neutral
    # names that describe the CONTENT, not which pass populated it.  Pkl
    # compat: pre-Stage-1e bundles never had these fields at all (default via
    # getattr(..., <default>) everywhere); Stage 1e bundles are dev-only and
    # already version-gated by CONNECTIVITY_SUMMARY_VERSION (bumped to 2 by
    # finding R1 in the same round), so no migration path is needed.
    removed_floating_nodes: Set[str] = field(default_factory=set)
    n_floating_components_removed: int = 0
    kept_nonlargest_iface_cached: Set[str] = field(default_factory=set)


def compute_interface_nodes(
    shared_boundary_nodes: Set[str],
    die_attachment_net_map: Optional[Dict[str, str]],
    die_attachment_nodes: Set[str],
) -> Set[str]:
    """Single-sourced interface-node formula (Stage 1e finding F13).

    ``shared_boundary_nodes | die_attachment_net_map.keys() | die_attachment_nodes``.
    Used by BOTH ``DistributedNetlistParser.parse_and_dump`` (to compute the
    persisted ``parser_interface_set``) and
    ``model._create_distributed_model_from_bundle`` (to compute the
    model-creation-time ``interface_nodes``) so the two can never drift apart
    and spuriously fail the Stage 1e trust assertion.  Lives in this leaf
    module (no imports of ``parser.py``/``model.py``) so both call sites can
    import it without triggering the ``parser -> model`` circular-import
    restriction.
    """
    return (
        set(shared_boundary_nodes)
        | set((die_attachment_net_map or {}).keys())
        | set(die_attachment_nodes)
    )


def _decompose_and_remove_floating(
    tile_data: 'TileData',
    port_nodes_local: Set[str],
    min_interface_keep: int,
) -> Tuple[List[Set[str]], Optional[Set[str]], Set[str], Set[str], int]:
    """Ground-excluding BFS decomposition + largest/threshold-keep removal.

    Stage 1e finding F14: shared core mutating *tile_data* in place, used by
    BOTH ``retile._pre_clean_tile_data`` (parse-time pre-clean) and
    ``TileWorker._remove_floating_islands`` (legacy worker-time fallback
    removal) so parse-time and worker-time removal decisions are
    bit-identical BY CONSTRUCTION rather than by parallel maintenance.

    Stage 1e finding F4 (determinism): connected-component enumeration must
    not depend on ``set`` iteration order (PYTHONHASHSEED).  The outer BFS
    seed loop iterates ``sorted(tile_data.all_nodes)``, and the
    largest-component tie-break uses ``key=(len(c), min(c))`` -- mirrors the
    canonicalization convention from commit d0b6071 (sort before any
    order-sensitive reduction over set-derived collections).

    *port_nodes_local* must already be intersected with ``tile_data.all_nodes``
    by the caller (both current callers do this before invoking).

    Returns:
        ``(components, largest, kept_nonlargest_iface, removed_nodes, n_removed)``.
        ``components`` is the FULL decomposition (all components, largest
        included) so callers that need per-KEPT-component summaries (parse-
        time pre-clean) can classify them (``comp & removed_nodes`` is empty
        iff *comp* was kept, since removal is atomic per whole component).
        ``largest`` is the SAME set object as the matching entry of
        ``components`` (identity-comparable via ``comp is largest``), or
        ``None`` when *tile_data* has no nodes at all.  Finding R1: callers
        that build per-component summaries need to tag which component was
        kept UNCONDITIONALLY (the largest) vs. kept because it met
        *min_interface_keep* -- only the latter's keep decision can be
        invalidated by a later, coordinator-side interface-set recompute.
        ``kept_nonlargest_iface``/``removed_nodes``/``n_removed`` mirror the
        legacy ``TileWorker._remove_floating_islands`` return contract.

    Side effect (Stage 1e F8/F9, R9): on removal,
    ``tile_data.removed_floating_nodes``,
    ``tile_data.n_floating_components_removed``, and
    ``tile_data.kept_nonlargest_iface_cached`` are updated -- UNIONED with
    (not overwriting) any prior value, so a parent-level whole-tile pass
    followed by a sub-tile pass both contribute to the persisted diagnostic.
    """
    adj: Dict[str, Set[str]] = {}
    for u, v, _g in tile_data.resistive_edges:
        if u == '0' or v == '0':
            continue
        adj.setdefault(u, set()).add(v)
        adj.setdefault(v, set()).add(u)

    visited: Set[str] = set()
    components: List[Set[str]] = []
    for start in sorted(tile_data.all_nodes):
        if start in visited or start == '0':
            continue
        comp: Set[str] = set()
        queue = [start]
        while queue:
            node = queue.pop()
            if node in visited:
                continue
            visited.add(node)
            comp.add(node)
            for nb in adj.get(node, ()):
                if nb not in visited:
                    queue.append(nb)
        components.append(comp)

    if len(components) <= 1:
        largest = components[0] if components else None
        return components, largest, set(), set(), 0

    largest = max(components, key=lambda c: (len(c), min(c)))
    removed_nodes: Set[str] = set()
    kept_nonlargest_iface: Set[str] = set()
    n_removed = 0

    for comp in components:
        if comp is largest:
            continue
        n_interface = len(comp & port_nodes_local)
        if n_interface >= min_interface_keep:
            kept_nonlargest_iface.update(comp & port_nodes_local)
            continue
        removed_nodes.update(comp)
        n_removed += 1

    if removed_nodes:
        tile_data.all_nodes -= removed_nodes
        tile_data.boundary_nodes -= removed_nodes
        tile_data.resistive_edges = [
            (u, v, g) for u, v, g in tile_data.resistive_edges
            if u not in removed_nodes and v not in removed_nodes
        ]
        tile_data.capacitive_edges = [
            (u, v, c) for u, v, c in tile_data.capacitive_edges
            if u not in removed_nodes and v not in removed_nodes
        ]
        for node in removed_nodes:
            tile_data.current_injections.pop(node, None)

    if removed_nodes or n_removed:
        tile_data.removed_floating_nodes = (
            set(getattr(tile_data, 'removed_floating_nodes', None) or ()) | removed_nodes
        )
        tile_data.n_floating_components_removed = (
            getattr(tile_data, 'n_floating_components_removed', 0) + n_removed
        )
    if kept_nonlargest_iface:
        tile_data.kept_nonlargest_iface_cached = (
            set(getattr(tile_data, 'kept_nonlargest_iface_cached', None) or ())
            | kept_nonlargest_iface
        )

    return components, largest, kept_nonlargest_iface, removed_nodes, n_removed


def verify_component_keep_decisions(
    component_summaries: List[Dict[str, Any]],
    final_interface_set: Set[str],
) -> Tuple[bool, List[Dict[str, Any]]]:
    """Stage 1e finding R1: parse-end consistency re-check.

    A non-largest component is kept only because ``len(candidates) >=
    keep_threshold`` at the pass that decided it (``_pre_clean_tile_data``,
    threshold=5 whole-tile / threshold=1 sub-tile).  ``candidates`` is scoped
    against the port-candidate set AVAILABLE AT THAT PASS (the step-0 raw
    ``*``-declaration scan for whole tiles; ``sub.boundary_nodes``-derived
    candidates for sub-tiles) -- NOT against the FINAL, post-split
    ``shared_boundary_nodes`` the parser computes afterward (step 3b).  A
    raw-shared candidate can be silently DEMOTED between those two points
    (dropped to a single declaring tile) when a SPLIT neighbor's own
    parse-time pre-clean removed the candidate's fragment there before the
    split ever ran -- see the parser.py step-3b comment.  When that happens,
    the component's keep decision no longer holds against the interface set
    the coordinator will actually use, and BOTH the F10 structural-
    completeness check and the ``interface_nodes == parser_interface_set``
    trust assertion pass right through it (neither observes a *demotion* of
    a candidate the summary still lists) -- so this must be checked
    explicitly, once, at the end of parsing.

    The unconditionally-kept LARGEST component of each tile (``is_largest``)
    never depended on a threshold and is exempt.

    Args:
        component_summaries: The full, aggregated (all tiles, post-split)
            list of kept-component summary dicts.
        final_interface_set: The parser's own final interface-node set
            (``parser_interface_set`` -- ``compute_interface_nodes`` applied
            to the FINAL, post-step-3b ``shared_boundary_nodes``).

    Returns:
        ``(ok, violations)``.  ``ok`` is False iff any non-largest kept
        component's candidate overlap with *final_interface_set* has fallen
        below the threshold that justified keeping it.  ``violations`` is
        the (possibly empty) list of offending summary dicts, for logging.
    """
    violations: List[Dict[str, Any]] = []
    for summary in component_summaries:
        if summary.get('is_largest'):
            continue
        threshold = summary.get('keep_threshold')
        if threshold is None:
            # Summary predates this field (shouldn't happen once bundles are
            # version-gated on CONNECTIVITY_SUMMARY_VERSION >= 2) -- degrade
            # safely (nothing to check) rather than raise on an unexpected
            # shape.
            continue
        candidates = summary.get('candidates') or ()
        n_final = len(set(candidates) & final_interface_set)
        if n_final < threshold:
            violations.append(summary)
    return (not violations), violations


def _is_gzip_file(path: str) -> bool:
    """Check if file is gzip-compressed by magic bytes."""
    with open(path, 'rb') as f:
        return f.read(2) == b'\x1f\x8b'


def _load_nd_file(nd_path: str) -> Dict[str, str]:
    """Load .nd file for node-net mapping (lowercase net names).

    Returns:
        node -> lowercase net name mapping
    """
    import gzip

    node_net_map_lower: Dict[str, str] = {}

    if nd_path is None:
        return node_net_map_lower

    is_gzip = nd_path.endswith('.gz') or _is_gzip_file(nd_path)
    open_fn = gzip.open if is_gzip else open
    mode = 'rt' if is_gzip else 'r'

    with open_fn(nd_path, mode) as f:
        for line in f:
            parts = line.strip().split()
            # Format: node_name x y layer tile_id net_name
            if len(parts) >= 6:
                node_net_map_lower[parts[0]] = parts[5].lower()

    return node_net_map_lower


def _parse_tile_ckt(
    ckt_path: str,
    nd_path: Optional[str],
    net_filter: Optional[str],
    tile_id: Tuple[int, int],
) -> TileData:
    """Parse a tile .ckt file to extract resistive edges and current injections.

    Args:
        ckt_path: Path to tile .ckt file
        nd_path: Path to tile .nd file (for net filtering)
        net_filter: Optional lowercase net name to filter by
        tile_id: Tile identifier

    Returns:
        TileData with parsed edges and nodes
    """
    import gzip
    from parser.spice_lexer import _parse_spice_value

    node_net_map_lower = _load_nd_file(nd_path)

    resistive_edges: List[Tuple[str, str, float]] = []
    capacitive_edges: List[Tuple[str, str, float]] = []
    current_injections: Dict[str, float] = {}
    all_nodes: Set[str] = set()
    boundary_nodes: Set[str] = set()

    GMAX = 1e5
    SHORT_THRESHOLD = 1e-6

    is_gzip = ckt_path.endswith('.gz') or _is_gzip_file(ckt_path)
    open_fn = gzip.open if is_gzip else open
    mode = 'rt' if is_gzip else 'r'

    with open_fn(ckt_path, mode) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('*'):
                continue

            tokens = line.split()
            if len(tokens) < 4:
                continue

            first = tokens[0].lower()

            # Resistors
            if first[0] == 'r':
                if first == 'r':
                    # Unnamed: r node1 node2 value
                    node1, node2, value_token = tokens[1], tokens[2], tokens[3]
                else:
                    # Named: R_name node1 node2 value
                    node1, node2, value_token = tokens[1], tokens[2], tokens[3]

                # Handle boundary markers
                is_bnd1 = node1.startswith('*')
                is_bnd2 = node2.startswith('*')
                if is_bnd1:
                    node1 = node1[1:]
                    boundary_nodes.add(node1)
                if is_bnd2:
                    node2 = node2[1:]
                    boundary_nodes.add(node2)

                # Net filter
                if net_filter is not None:
                    n1_net = node_net_map_lower.get(node1)
                    n2_net = node_net_map_lower.get(node2)
                    if n1_net != net_filter and n2_net != net_filter:
                        continue

                try:
                    r_value = _parse_spice_value(value_token)
                except ValueError:
                    continue

                # Convert to kOhm then conductance (mS)
                r_kohm = r_value * R_TO_KOHM
                if r_kohm <= 0 or r_kohm < SHORT_THRESHOLD:
                    g = GMAX
                else:
                    g = 1.0 / r_kohm

                resistive_edges.append((node1, node2, g))
                all_nodes.add(node1)
                all_nodes.add(node2)

            # Current sources (for DC value extraction)
            elif first[0] == 'i':
                if first == 'i':
                    continue  # Unnamed current source without enough info
                # Named: I_name node+ node- dc_value ...
                node_pos = tokens[1]
                node_neg = tokens[2]

                if node_pos.startswith('*'):
                    node_pos = node_pos[1:]
                if node_neg.startswith('*'):
                    node_neg = node_neg[1:]

                # Net filter
                if net_filter is not None:
                    n1_net = node_net_map_lower.get(node_pos)
                    n2_net = node_net_map_lower.get(node_neg)
                    if n1_net != net_filter and n2_net != net_filter:
                        continue

                try:
                    dc_value = _parse_spice_value(tokens[3])
                    dc_ma = dc_value * I_TO_MA  # Convert A to mA
                except (ValueError, IndexError):
                    continue

                # Current source from node_pos to node_neg (pos = net node, neg = ground)
                if node_pos != '0':
                    current_injections[node_pos] = current_injections.get(node_pos, 0.0) + dc_ma

            # Capacitors
            elif first[0] == 'c':
                if first == 'c':
                    # Unnamed: c node1 node2 value
                    node1, node2, value_token = tokens[1], tokens[2], tokens[3]
                else:
                    # Named: C_name node1 node2 value
                    node1, node2, value_token = tokens[1], tokens[2], tokens[3]

                # Handle boundary markers
                is_bnd1 = node1.startswith('*')
                is_bnd2 = node2.startswith('*')
                if is_bnd1:
                    node1 = node1[1:]
                    boundary_nodes.add(node1)
                if is_bnd2:
                    node2 = node2[1:]
                    boundary_nodes.add(node2)

                # Net filter
                if net_filter is not None:
                    n1_net = node_net_map_lower.get(node1)
                    n2_net = node_net_map_lower.get(node2)
                    if n1_net != net_filter and n2_net != net_filter:
                        continue

                try:
                    c_value = _parse_spice_value(value_token)
                except ValueError:
                    continue

                c_fF = c_value * C_TO_FF
                capacitive_edges.append((node1, node2, c_fF))
                if node1 != '0':
                    all_nodes.add(node1)
                if node2 != '0':
                    all_nodes.add(node2)

    return TileData(
        tile_id=tile_id,
        resistive_edges=resistive_edges,
        all_nodes=all_nodes,
        boundary_nodes=boundary_nodes,
        current_injections=current_injections,
        capacitive_edges=capacitive_edges,
    )


def _iter_instance_sources(
    instance_path: Optional[str],
    net_filter: Optional[str],
    nd_path: Optional[str] = None,
):
    """Yield PreparedSource objects from an instanceModels file.

    Encapsulates all shared I/O and filtering logic: gzip detection,
    comment/blank/dot-prefix line skipping, fast structured-name filter,
    ``_prepare_instance_source()`` parsing, and slow .nd-based filter
    fallback.

    Args:
        instance_path: Path to instanceModels*.sp file.  Yields nothing
            when ``None``.
        net_filter: Optional lowercase net name to filter by.
        nd_path: Path to .nd file for net filtering (used only when
            *net_filter* is set and instance names are not structured).

    Yields:
        ``PreparedSource`` namedtuples that passed all filters.
    """
    if instance_path is None:
        return

    import gzip
    from parser.current_sources import _prepare_instance_source
    from parser.spice_lexer import (
        _check_net_filter,
        _fast_instance_net_filter,
        _has_structured_instance_names,
    )

    # Detect structured names for fast net filtering
    use_fast_filter = (
        net_filter is not None
        and _has_structured_instance_names(instance_path)
    )

    # Only load .nd file when net_filter is set AND names are not structured
    if net_filter and not use_fast_filter:
        node_net_map_lower = _load_nd_file(nd_path)
    else:
        node_net_map_lower = {}

    is_gzip = instance_path.endswith('.gz') or _is_gzip_file(instance_path)
    open_fn = gzip.open if is_gzip else open
    mode = 'rt' if is_gzip else 'r'

    with open_fn(instance_path, mode) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('*') or line.startswith('.'):
                continue

            # Fast path: reject non-matching nets before expensive parsing
            if use_fast_filter and not _fast_instance_net_filter(line, net_filter):
                continue

            prepared = _prepare_instance_source(line)
            if prepared is None:
                continue

            # Slow path fallback: .nd-based filtering for unstructured names
            if net_filter and not use_fast_filter:
                if not _check_net_filter(
                    prepared.node_pos, prepared.node_neg,
                    node_net_map_lower, net_filter,
                ):
                    continue

            yield prepared


def _parse_instance_models(
    instance_path: str,
    net_filter: Optional[str],
    nd_path: Optional[str] = None,
) -> Dict[str, float]:
    """Parse instanceModels file for current source DC values.

    Uses shared ``_iter_instance_sources()`` for parsing + A-to-mA conversion,
    matching the flat parser's handling exactly.

    Args:
        instance_path: Path to instanceModels*.sp file
        net_filter: Optional lowercase net name to filter by
        nd_path: Path to .nd file for net filtering (required when net_filter is set)

    Returns:
        Dict mapping node -> current in mA (positive = sink)
    """
    current_injections: Dict[str, float] = {}

    for prepared in _iter_instance_sources(instance_path, net_filter, nd_path):
        # Inject at positive terminal only: instance model current sources
        # are always node+ -> ground ('0'), so the negative terminal is
        # eliminated from the nodal system and needs no entry.
        if prepared.node_pos != '0':
            current_injections[prepared.node_pos] = (
                current_injections.get(prepared.node_pos, 0.0) + prepared.static_current_ma
            )

    return current_injections


def _iter_instance_capacitors(
    instance_path: Optional[str],
    net_filter: Optional[str],
    nd_path: Optional[str] = None,
):
    """Yield ``(node, capacitance_fF)`` tuples from an instanceModels file.

    Mirrors the I/O and filtering pattern of :func:`_iter_instance_sources`
    but for grounded capacitor lines (``c...`` prefix).  Each capacitor is
    assumed to have one terminal connected to ground (``'0'``); the ground
    terminal is stripped and only the non-ground node is yielded.

    Args:
        instance_path: Path to instanceModels*.sp file.  Yields nothing
            when ``None``.
        net_filter: Optional lowercase net name to filter by.
        nd_path: Path to .nd file for net filtering (used only when
            *net_filter* is set).

    Yields:
        ``(node, c_fF)`` tuples that passed all filters.
    """
    if instance_path is None:
        return

    import gzip
    from parser.spice_lexer import _parse_spice_value, _check_net_filter

    # Load .nd file when net_filter is set
    if net_filter:
        node_net_map_lower = _load_nd_file(nd_path)
    else:
        node_net_map_lower = {}

    is_gzip = instance_path.endswith('.gz') or _is_gzip_file(instance_path)
    open_fn = gzip.open if is_gzip else open
    mode = 'rt' if is_gzip else 'r'

    with open_fn(instance_path, mode) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('*') or line.startswith('.'):
                continue

            tokens = line.split()
            if len(tokens) < 4:
                continue

            first = tokens[0].lower()
            if first[0] != 'c':
                continue

            node1 = tokens[1]
            node2 = tokens[2]

            # Determine non-ground node (grounded cap: one terminal is '0')
            if node1 == '0' and node2 == '0':
                continue  # Both ground — skip
            if node1 != '0' and node2 != '0':
                continue  # Coupling cap in instanceModels — skip (not grounded)

            non_ground = node1 if node1 != '0' else node2

            # Net filter: check that the non-ground terminal belongs to the target net
            if net_filter:
                if not _check_net_filter(
                    node1, node2, node_net_map_lower, net_filter,
                ):
                    continue

            try:
                c_value = _parse_spice_value(tokens[3])
            except (ValueError, IndexError):
                continue

            c_fF = c_value * C_TO_FF
            yield (non_ground, c_fF)


def _parse_instance_capacitors(
    instance_path: Optional[str],
    net_filter: Optional[str],
    nd_path: Optional[str] = None,
) -> List[Tuple[str, str, float]]:
    """Parse instanceModels file for grounded capacitor values.

    Accumulates per-node capacitances and returns a list of
    ``(node, '0', total_c_fF)`` tuples suitable for merging into
    ``TileData.capacitive_edges``.

    Args:
        instance_path: Path to instanceModels*.sp file
        net_filter: Optional lowercase net name to filter by
        nd_path: Path to .nd file for net filtering

    Returns:
        List of ``(node, '0', total_c_fF)`` tuples.
    """
    per_node: Dict[str, float] = {}

    for node, c_fF in _iter_instance_capacitors(instance_path, net_filter, nd_path):
        per_node[node] = per_node.get(node, 0.0) + c_fF

    return [(node, '0', total) for node, total in per_node.items()]


def parse_tile_with_instances(
    ckt_path: str,
    nd_path: Optional[str],
    net_filter: Optional[str],
    tile_id: Tuple[int, int],
    instance_path: Optional[str] = None,
) -> TileData:
    """Parse a tile .ckt file and merge instance model current injections.

    Combines _parse_tile_ckt() + _parse_instance_models() + merge into a
    single call, eliminating duplication between TileWorker.setup() and
    DistributedNetlistParser.parse_and_dump().

    Args:
        ckt_path: Path to tile .ckt file
        nd_path: Path to tile .nd file (for net filtering)
        net_filter: Optional lowercase net name to filter by
        tile_id: Tile identifier tuple (x, y)
        instance_path: Optional path to instanceModels*.sp file

    Returns:
        TileData with parsed edges, nodes, and merged current injections
    """
    tile_data = _parse_tile_ckt(ckt_path, nd_path, net_filter, tile_id)

    if instance_path:
        inst_currents = _parse_instance_models(instance_path, net_filter, nd_path)
        for node, current in inst_currents.items():
            if node in tile_data.all_nodes:
                tile_data.current_injections[node] = (
                    tile_data.current_injections.get(node, 0.0) + current
                )

        inst_caps = _parse_instance_capacitors(instance_path, net_filter, nd_path)
        tile_data.capacitive_edges.extend(
            (node, gnd, c) for node, gnd, c in inst_caps
            if node in tile_data.all_nodes
        )

    return tile_data


def parse_and_dump_tile(
    ckt_path, nd_path, net_filter, tile_id, instance_path,
    output_dir, die_attachment_candidates=None, net_name=None,
    shared_boundary_candidates=None, pad_node_candidates=None,
):
    """Parse tile, pre-clean in memory, dump TileData to .pkl, return metadata.

    Stage 1e: when *shared_boundary_candidates* is provided (the hoisted
    global ``*``-prefixed boundary scan, computed once before the parse pass
    -- see ``DistributedNetlistParser.parse_and_dump``), this function runs
    the SAME island-removal pass that used to be deferred to
    ``TileWorker._remove_floating_islands`` at model-creation time, in memory,
    before the single pkl write.  Port-set semantics are bit-identical to the
    worker's threshold-5 removal: ``port_nodes = (all_nodes ∩ shared_bnd) ∪
    (all_nodes ∩ die_nodes)``.  The kept-component decomposition is returned
    as ``component_summaries`` so the coordinator can persist it for the A7
    union-find island detector (``pgmath.schur.detect_interface_islands_from_summaries``)
    and skip the O(S.nnz) Schur-BFS entirely at prepare() time.

    When *shared_boundary_candidates* is ``None`` (legacy callers, e.g. the
    ``_adapt_legacy_args`` no-pkl shim), no pre-clean runs and the returned
    dict has no ``component_summaries`` key -- callers must treat a missing
    key the same as an empty list.
    """
    import pickle
    from pathlib import Path
    tile_data = parse_tile_with_instances(ckt_path, nd_path, net_filter, tile_id, instance_path)

    die_found = {}
    if die_attachment_candidates and net_name:
        die_found = {n: net_name for n in die_attachment_candidates if n in tile_data.all_nodes}

    result: Dict[str, Any] = {}
    if shared_boundary_candidates is not None:
        from .retile import _pre_clean_tile_data
        # Finding R11: intersect each (potentially huge, multi-hundred-
        # thousand-entry) global candidate set with this tile's own (small)
        # all_nodes BEFORE unioning them, instead of unioning the two global
        # sets first (a full copy+union of both, repeated once per parse
        # task -- ~100x redundant on a BRCM-scale parse) only for
        # _pre_clean_tile_data to immediately re-intersect the union with
        # all_nodes anyway.  Set intersection is O(min(|a|, |b|)), so this is
        # O(|tile|) work per task instead of O(|global sets|).
        all_nodes = tile_data.all_nodes
        port_nodes = all_nodes & shared_boundary_candidates
        if die_attachment_candidates:
            port_nodes = port_nodes | (all_nodes & die_attachment_candidates)
        n_removed, component_summaries = _pre_clean_tile_data(
            tile_data, port_nodes, min_interface_keep=5,
            pad_nodes=pad_node_candidates,
        )
        result['islands_removed_at_parse'] = n_removed
        result['component_summaries'] = component_summaries

    tile_str = '_'.join(str(c) for c in tile_id)
    pkl_path = Path(output_dir) / f'tile_{tile_str}.pkl'
    with open(pkl_path, 'wb') as f:
        pickle.dump(tile_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    result.update({
        'tile_id': tile_id,
        'boundary_nodes': tile_data.boundary_nodes,
        'die_attachment_net_map': die_found,
        'n_nodes': len(tile_data.all_nodes),
        'n_edges': len(tile_data.resistive_edges),
        'n_currents': len(tile_data.current_injections),
        'n_cap_edges': len(tile_data.capacitive_edges),
    })
    return result

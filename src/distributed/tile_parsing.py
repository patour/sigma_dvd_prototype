"""Tile data structures and stateless parsing functions.

Contains TileData, unit conversion constants, and all stateless parsing
helpers for tile .ckt / .nd / instanceModels files.

Split from tile_worker.py for maintainability. All public names are
re-exported from tile_worker so existing imports keep working.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple



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
):
    """Parse tile, dump TileData to .pkl, return lightweight metadata."""
    import pickle
    from pathlib import Path
    tile_data = parse_tile_with_instances(ckt_path, nd_path, net_filter, tile_id, instance_path)
    tile_str = '_'.join(str(c) for c in tile_id)
    pkl_path = Path(output_dir) / f'tile_{tile_str}.pkl'
    with open(pkl_path, 'wb') as f:
        pickle.dump(tile_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    die_found = {}
    if die_attachment_candidates and net_name:
        die_found = {n: net_name for n in die_attachment_candidates if n in tile_data.all_nodes}
    return {
        'tile_id': tile_id,
        'boundary_nodes': tile_data.boundary_nodes,
        'die_attachment_net_map': die_found,
        'n_nodes': len(tile_data.all_nodes),
        'n_edges': len(tile_data.resistive_edges),
        'n_currents': len(tile_data.current_injections),
        'n_cap_edges': len(tile_data.capacitive_edges),
    }

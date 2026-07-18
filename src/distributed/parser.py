"""Distributed PDN netlist parser.

Parses metadata from a tile-based PDN netlist directory without building
a monolithic graph. Provides per-tile file paths and package data for
distributed model creation.
"""

from __future__ import annotations

import gzip
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


def _parse_spice_value(value_str: str) -> float:
    """Parse a numeric value with optional SPICE unit suffix.

    Thin wrapper that lazily delegates to parser.spice_lexer._parse_spice_value
    to avoid circular imports (parser -> solver -> distributed).
    """
    from parser.spice_lexer import _parse_spice_value as _psv
    return _psv(value_str)

# Regex for tile and instance model files
_RE_TILE_FILE = re.compile(r'tile_(\d+)_(\d+)\.(?:ckt|sp)(?:\.gz)?$')
_RE_INST_FILE = re.compile(r'instanceModels_(\d+)_(\d+)\.(?:sp|ckt)(?:\.gz)?$')
_RE_BOUNDARY_NODE = re.compile(r'^\*(\S+)')

# Unit conversions (matching pdn_parser.py)
R_TO_KOHM = 1e-3  # Ohm to kOhm
C_TO_FF = 1e15  # Farad to femtoFarad

# Stage 1e: bump whenever the connectivity-summary computation (component
# decomposition semantics, port-set rule, or has_pad definition) changes.
# `model.py._resolve_island_detection` compares this against
# ``ParsedTileBundle.connectivity_summary_version``; a mismatch (older pkl
# bundle, or a code change post-parse) forces the full legacy fallback
# (worker-side removal + coordinator Schur-BFS) rather than trusting stale
# summaries.
#
# Bumped 1 -> 2 for finding R1 (second-round review): component summaries
# now carry 'is_largest'/'keep_threshold' fields (schema change) and the
# parser runs a new parse-end consistency re-check
# (tile_parsing.verify_component_keep_decisions) that can drop trust in a
# bundle a v1-era parser would have trusted.  Also covers finding R9's
# TileData field renames (removed_at_parse -> removed_floating_nodes, etc.)
# -- pre-Stage-1e bundles never had these fields at all, and Stage 1e
# bundles are dev-only, so no migration path is needed; a version bump is
# sufficient to force any stale pre-R1 pkl bundle onto the legacy fallback.
CONNECTIVITY_SUMMARY_VERSION = 2


@dataclass
class TileConfig:
    """Configuration for a single tile."""

    tile_id: Tuple[int, int]
    ckt_path: str
    nd_path: Optional[str]
    instance_path: Optional[str]
    net_filter: Optional[str]


@dataclass
class PackageData:
    """Package elements held by coordinator."""

    vsrc_dict: Dict[str, Dict]  # voltage source name -> {node+, node-, net, value}
    package_edges: List[Tuple[str, str, float]]  # (u, v, conductance) in mS
    pad_nodes: Set[str]  # Dirichlet nodes (vsrc positive terminal)
    tap_nodes: Set[str]  # Package tap unknowns
    die_attachment_nodes: Set[str]  # M13 die nodes promoted to interface
    vdd: float
    net_name: str
    die_attachment_net_map: Dict[str, str] = field(default_factory=dict)  # node → net (worker-validated)
    package_cap_edges: List[Tuple[str, str, float]] = field(default_factory=list)  # (u, v, C_fF)


@dataclass
class PowerGridMetaData:
    """All metadata needed to create a DistributedPowerGridModel.

    Net-specific: net_name, vdd, package_data are for the filtered net.
    """

    tile_grid: Tuple[int, int]
    parameters: Dict[str, str]
    tile_configs: List[TileConfig]
    package_data: PackageData
    net_name: str
    vdd: float


def _is_gzip_file(path: str) -> bool:
    """Check if file is gzip-compressed by magic bytes (0x1f 0x8b)."""
    with open(path, 'rb') as f:
        return f.read(2) == b'\x1f\x8b'


def _open_file(path: str):
    """Open a file, auto-detecting gzip compression by magic bytes."""
    if path.endswith('.gz') or _is_gzip_file(path):
        return gzip.open(path, 'rt')
    return open(path, 'r')


def _is_die_coordinate_node(node: str) -> bool:
    """Detect X_Y_* die coordinate pattern (first two _-delimited parts are digits)."""
    parts = node.split('_')
    return len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit()


def _uf_find(parent: dict, node: str) -> str:
    """Union-Find: find root with iterative path compression."""
    if node not in parent:
        parent[node] = node
        return node
    # Walk to root
    root = node
    while parent[root] != root:
        root = parent[root]
    # Path compression
    while parent[node] != root:
        parent[node], node = root, parent[node]
    return root


def _uf_union(parent: dict, uf_net: dict, node1: str, node2: str) -> None:
    """Union-Find: union two nodes, preferring root with known net type."""
    root1 = _uf_find(parent, node1)
    root2 = _uf_find(parent, node2)
    if root1 == root2:
        return
    net1 = uf_net.get(root1)
    net2 = uf_net.get(root2)
    if net1:
        parent[root2] = root1
        uf_net[root1] = net1
    elif net2:
        parent[root1] = root2
        uf_net[root2] = net2
    else:
        parent[root2] = root1


def compute_shared_boundary_nodes(per_tile_boundaries):
    """Return nodes appearing in 2+ tile boundary sets."""
    from collections import Counter
    tile_count = Counter()
    for boundary_set in per_tile_boundaries:
        tile_count.update(boundary_set)
    return {node for node, count in tile_count.items() if count >= 2}


def _sort_component_summaries(
    component_summaries: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Stage 1e finding F4: deterministic ordering for the aggregated
    per-tile connectivity summaries.

    Sorts by tile id, then by the summary's smallest candidate node name.
    Independent of upstream per-tile set-iteration order / PYTHONHASHSEED
    (the underlying BFS decomposition that produces each tile's summaries is
    already made deterministic in
    ``tile_parsing._decompose_and_remove_floating``; this is a cheap,
    explicit belt-and-suspenders stabilization of the FINAL flattened list
    order across tiles).  ``candidates`` is always non-empty in a real
    summary (``retile._summarize`` only emits a summary when it is).
    """
    return sorted(
        component_summaries,
        key=lambda s: (s.get('tile_id', ()), min(s['candidates'])),
    )


class DistributedNetlistParser:
    """Parser for distributed tile-based PDN netlists.

    Reads the top-level ckt.sp to discover tile structure, parameters,
    and package data. Does NOT build a monolithic graph - instead returns
    PowerGridMetaData with per-tile file paths for distributed processing.
    """

    def __init__(self, netlist_dir: str, net_filter: Optional[str] = None):
        self.netlist_dir = Path(netlist_dir)
        self.net_filter = net_filter
        self._net_filter_lower = net_filter.lower() if net_filter else None

    def parse_metadata(self) -> PowerGridMetaData:
        """Parse netlist directory to extract metadata without building graph.

        Returns:
            PowerGridMetaData with tile configs, package data, parameters
        """
        # 1. Parse ckt.sp for structure
        main_file = self.netlist_dir / 'ckt.sp'
        if not main_file.exists():
            raise FileNotFoundError(f"Main netlist file not found: {main_file}")

        parameters, tile_grid = self._parse_main_netlist(main_file)

        # 2. Discover tile files
        tile_configs = self._discover_tiles(tile_grid)

        # 3. Determine net name and VDD
        net_name = self.net_filter or self._infer_net_name(parameters)
        vdd = self._extract_vdd(parameters, net_name)

        # 4. Parse package model
        package_data = self._parse_package(net_name, vdd)

        # 5. Set net_filter on all tile configs
        for tc in tile_configs:
            tc.net_filter = self._net_filter_lower

        return PowerGridMetaData(
            tile_grid=tile_grid,
            parameters=parameters,
            tile_configs=tile_configs,
            package_data=package_data,
            net_name=net_name,
            vdd=vdd,
        )

    def _parse_main_netlist(self, path: Path) -> Tuple[Dict[str, str], Tuple[int, int]]:
        """Parse main netlist for parameters and tile grid info."""
        parameters: Dict[str, str] = {}
        tile_grid = (1, 1)

        # Also check pg_net_voltage
        pg_net_voltage = self.netlist_dir / 'pg_net_voltage'
        if pg_net_voltage.exists():
            with open(pg_net_voltage, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('*') or line.startswith('#'):
                        continue
                    parts = line.split()
                    if len(parts) >= 2:
                        parameters[parts[0].upper()] = parts[1]

        with _open_file(str(path)) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('*'):
                    continue

                tokens = line.split()
                cmd = tokens[0].lower()

                if cmd == '.partition_info' and len(tokens) >= 3:
                    tile_grid = (int(tokens[1]), int(tokens[2]))
                elif cmd in ('.parameter', '.param'):
                    if len(tokens) == 3 and '=' not in tokens[1]:
                        parameters[tokens[1].upper()] = tokens[2]
                    else:
                        for token in tokens[1:]:
                            if '=' in token:
                                name, value = token.split('=', 1)
                                parameters[name.strip().upper()] = value.strip()

        return parameters, tile_grid

    def _discover_tiles(self, tile_grid: Tuple[int, int]) -> List[TileConfig]:
        """Discover tile files in the netlist directory."""
        n_x, n_y = tile_grid
        tile_configs = []

        for x in range(n_x):
            for y in range(n_y):
                ckt_path = self.netlist_dir / f'tile_{x}_{y}.ckt'
                if not ckt_path.exists():
                    ckt_gz = self.netlist_dir / f'tile_{x}_{y}.ckt.gz'
                    if ckt_gz.exists():
                        ckt_path = ckt_gz
                    else:
                        continue

                nd_path = self.netlist_dir / f'tile_{x}_{y}.nd'
                nd_str = str(nd_path) if nd_path.exists() else None

                inst_path = None
                for ext in ('.sp', '.sp.gz'):
                    ip = self.netlist_dir / f'instanceModels_{x}_{y}{ext}'
                    if ip.exists():
                        inst_path = str(ip)
                        break

                tile_configs.append(TileConfig(
                    tile_id=(x, y),
                    ckt_path=str(ckt_path),
                    nd_path=nd_str,
                    instance_path=inst_path,
                    net_filter=None,
                ))

        return tile_configs

    def _infer_net_name(self, parameters: Dict[str, str]) -> str:
        """Infer the net name from parameters (first non-VSS net)."""
        for name in parameters:
            if 'VSS' not in name.upper():
                return name
        if parameters:
            return next(iter(parameters))
        raise ValueError("Cannot infer net name from parameters")

    def _extract_vdd(self, parameters: Dict[str, str], net_name: str) -> float:
        """Extract VDD from parameters."""
        key = net_name.upper()
        if key in parameters:
            return float(parameters[key])
        raise ValueError(
            f"Could not determine voltage for net '{net_name}'. "
            f"Available parameters: {list(parameters.keys())}"
        )

    def _parse_package(self, net_name: str, vdd: float,
                       die_net_map: dict = None) -> PackageData:
        """Parse package.ckt for voltage sources and bump connections.

        Uses union-find to classify nodes by net, replacing brittle
        name-based heuristics. Each voltage source seeds its positive
        terminal with the declared net (4th token). Resistor edges
        propagate net labels through the union-find structure. After
        parsing, nodes are filtered to the target *net_name*.

        Parameters
        ----------
        net_name : str
            Target power net (e.g. ``'VDD_XLV'``, ``'VDD_VAR'``).
        vdd : float
            Supply voltage for this net.
        die_net_map : dict, optional
            Mapping of ``{node: net}`` from worker tiles. If provided,
            seeds the union-find with die-side net labels after parsing.
        """
        pkg_path = self.netlist_dir / 'package.ckt'
        if not pkg_path.exists():
            pkg_gz = self.netlist_dir / 'package.ckt.gz'
            if not pkg_gz.exists():
                return PackageData(
                    vsrc_dict={}, package_edges=[], pad_nodes=set(),
                    tap_nodes=set(), die_attachment_nodes=set(),
                    vdd=vdd, net_name=net_name,
                )
            pkg_path = pkg_gz

        # ------------------------------------------------------------------
        # Phase 1: Parse ALL elements without net filtering
        # ------------------------------------------------------------------
        parent: Dict[str, str] = {}   # union-find parent
        uf_net: Dict[str, str] = {}   # root -> net label

        # Collected raw elements (unfiltered)
        vsrc_list: List[Tuple[str, str, str, str]] = []   # (name, node_pos, node_neg, vsrc_net)
        resistor_list: List[Tuple[str, str, float]] = []   # (node1, node2, g)
        capacitor_list: List[Tuple[str, str, float]] = []  # (node1, node2, c_fF)
        all_resistor_nodes: Set[str] = set()               # every node seen in any resistor

        with _open_file(str(pkg_path)) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('*'):
                    continue

                tokens = line.split()
                first = tokens[0].lower()

                # Voltage source: v_VDD_XLV VDD_XLV_vsrc 0 VDD_XLV
                #             or: V_VDD VDD_vrm 0 0.75  (numeric 4th token)
                if first.startswith('v') and len(tokens) >= 4:
                    name = tokens[0]
                    node_pos = tokens[1]
                    node_neg = tokens[2]
                    raw_net = tokens[3]
                    try:
                        float(raw_net)
                        # Numeric voltage value — infer net from element name
                        vsrc_net = name.split('_', 1)[1] if '_' in name else name[1:]
                    except ValueError:
                        vsrc_net = raw_net
                    vsrc_list.append((name, node_pos, node_neg, vsrc_net))
                    # Seed union-find: positive terminal belongs to declared net
                    _uf_find(parent, node_pos)
                    uf_net[_uf_find(parent, node_pos)] = vsrc_net

                # Resistor (r, rs, R_name, ...): r node1 node2 value
                elif first.startswith('r') and len(tokens) >= 4:
                    node1, node2 = tokens[1], tokens[2]
                    try:
                        r_value = _parse_spice_value(tokens[3])
                    except (ValueError, IndexError):
                        continue

                    # Convert to kOhm then conductance (mS)
                    r_kohm = r_value * R_TO_KOHM
                    if r_kohm <= 0 or r_kohm < 1e-6:
                        g = 1e5  # GMAX for zero-ohm shorts
                    else:
                        g = 1.0 / r_kohm

                    resistor_list.append((node1, node2, g))
                    all_resistor_nodes.add(node1)
                    all_resistor_nodes.add(node2)

                    # Union the two nodes (skip ground '0')
                    if node1 != '0' and node2 != '0':
                        _uf_union(parent, uf_net, node1, node2)

                # Inductor: short circuit for DC analysis
                elif first.startswith('l') and len(tokens) >= 4:
                    node1, node2 = tokens[1], tokens[2]
                    resistor_list.append((node1, node2, 1e5))  # GMAX
                    all_resistor_nodes.add(node1)
                    all_resistor_nodes.add(node2)
                    if node1 != '0' and node2 != '0':
                        _uf_union(parent, uf_net, node1, node2)

                # Capacitor: open for DC, but union for net label propagation
                elif first.startswith('c') and len(tokens) >= 4:
                    node1, node2 = tokens[1], tokens[2]
                    if node1 != '0' and node2 != '0':
                        _uf_union(parent, uf_net, node1, node2)
                    try:
                        c_value = _parse_spice_value(tokens[3])
                    except (ValueError, IndexError):
                        continue
                    capacitor_list.append((node1, node2, c_value * C_TO_FF))

        # Seed union-find with external die_net_map (worker-validated)
        if die_net_map:
            for node, net in die_net_map.items():
                if node in parent:
                    root = _uf_find(parent, node)
                    if root not in uf_net:
                        uf_net[root] = net

        # ------------------------------------------------------------------
        # Phase 2: Filter by target net + classify
        # ------------------------------------------------------------------
        net_upper = net_name.upper()

        vsrc_dict: Dict[str, Dict] = {}
        pad_nodes: Set[str] = set()

        for name, node_pos, node_neg, vsrc_net in vsrc_list:
            root = _uf_find(parent, node_pos)
            root_net = uf_net.get(root, '')
            if root_net.upper() == net_upper:
                vsrc_dict[name] = {
                    'node_pos': node_pos,
                    'node_neg': node_neg,
                    'net': vsrc_net,
                    'value': vdd,
                }
                pad_nodes.add(node_pos)

        package_edges: List[Tuple[str, str, float]] = []
        filtered_nodes: Set[str] = set()

        for node1, node2, g in resistor_list:
            # Check if EITHER non-ground node's root net matches target
            match = False
            for node in (node1, node2):
                if node == '0':
                    continue
                root = _uf_find(parent, node)
                if uf_net.get(root, '').upper() == net_upper:
                    match = True
                    break
            if match:
                package_edges.append((node1, node2, g))
                filtered_nodes.add(node1)
                filtered_nodes.add(node2)

        # Filter capacitors by net (package caps CAN be coupling / non-grounded)
        package_cap_edges: List[Tuple[str, str, float]] = []
        for node1, node2, c_fF in capacitor_list:
            match = False
            for node in (node1, node2):
                if node == '0':
                    continue
                root = _uf_find(parent, node)
                if uf_net.get(root, '').upper() == net_upper:
                    match = True
                    break
            if match:
                package_cap_edges.append((node1, node2, c_fF))

        # die_attachment_nodes: ALL nodes from ALL resistors with die coordinate pattern
        die_attachment_nodes: Set[str] = {
            node for node in all_resistor_nodes
            if _is_die_coordinate_node(node)
        }

        # tap_nodes: net-filtered nodes that are NOT die coordinates,
        # NOT pad nodes, NOT ground, and NOT vsrc nodes
        tap_nodes: Set[str] = set()
        for node in filtered_nodes:
            if (node != '0'
                    and not _is_die_coordinate_node(node)
                    and node not in pad_nodes
                    and 'vsrc' not in node.lower()):
                tap_nodes.add(node)

        return PackageData(
            vsrc_dict=vsrc_dict,
            package_edges=package_edges,
            pad_nodes=pad_nodes,
            tap_nodes=tap_nodes,
            die_attachment_nodes=die_attachment_nodes,
            vdd=vdd,
            net_name=net_name,
            package_cap_edges=package_cap_edges,
        )

    def parse_and_dump(
        self,
        output_dir: str,
        backend: str = 'local',
        max_interior: Optional[int] = None,
    ):
        """Parse netlist and dump per-tile TileData + metadata as .pkl files.

        Creates output_dir (if needed) with:
          - tile_X_Y.pkl  for each tile (pickled TileData, pre-cleaned -- see
            Stage 1e below)
          - metadata.pkl  with ``{'metadata': PowerGridMetaData,
            'boundary_nodes': Set[str], 'connectivity_summary_version': int,
            'parser_interface_set': Set[str], 'component_summaries':
            List[Dict]}``.  The last three keys are Stage 1e additions;
            ``load_distributed_partitions`` degrades gracefully (``None``)
            when loading an older bundle written before this field existed.

        Workers parse AND dump their own tiles in parallel. The coordinator
        collects only lightweight per-tile boundary/die-attachment metadata,
        merges them, and writes ``metadata.pkl``.

        Stage 1e (parse-time island pre-clean + connectivity summaries):
        each parse task pre-cleans its own tile in memory before writing the
        pkl (single pass, no reload), using the SAME port-set/threshold
        semantics as the island removal that would otherwise run at every
        ``create_distributed_model()`` call.  This both removes truly-floating
        components once (at parse time, amortized across every future model
        creation from this bundle) and produces per-kept-component
        connectivity summaries that let the coordinator's ``prepare()`` skip
        the O(S.nnz) Schur-BFS island detection in favour of a cheap
        union-find (see ``pgmath.schur.detect_interface_islands_from_summaries``
        and ``model._resolve_island_detection``).

        Args:
            output_dir: Directory to write .pkl files into
            backend: Compute backend for tile parsing ('local' or 'ray')
            max_interior: If set, tiles with more than this many interior
                nodes are recursively bisected and replaced by sub-tile pkls
                (B1 balanced retiling).  ``None`` disables splitting (default).

        Returns:
            Tuple of (Path to output_dir, ParsedTileBundle).
        """
        import pickle
        from .backend import LocalBackend, RayBackend
        from .tile_worker import parse_and_dump_tile

        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        # 1. Parse coordinator-side metadata (tile configs + package data)
        metadata = self.parse_metadata()

        # 2. Workers parse + dump tiles in parallel
        be = RayBackend() if backend == 'ray' else LocalBackend()
        be.initialize()

        die_candidates = metadata.package_data.die_attachment_nodes
        pad_candidates = metadata.package_data.pad_nodes
        net_name = metadata.net_name

        # Stage 1e step 0: hoist the global shared-boundary port set AHEAD of
        # the parse pass via a fast, parallel streaming scan of just the
        # '*'-prefixed boundary declarations (no graph build).  Each parse
        # map task below pre-cleans its own tile in memory (single pkl write,
        # no reload) using port_nodes = (all_nodes ∩ shared_bnd) ∪
        # (all_nodes ∩ die_candidates), promoting the island-removal work
        # that used to be deferred to TileWorker._remove_floating_islands at
        # every model creation into a one-time parse-time pass.
        _scan_args = [(tc.ckt_path,) for tc in metadata.tile_configs]
        _per_tile_raw_boundaries = be.map_func(
            DistributedNetlistParser._scan_tile_boundary_nodes, _scan_args,
        )
        shared_bnd = compute_shared_boundary_nodes(_per_tile_raw_boundaries)

        # Finding F1: keep the PRE-clean (raw scan) boundary set for every
        # whole tile, keyed by its (still 2-tuple) tile_id.  Needed below
        # (step 3b) to compute the FINAL shared_boundary_nodes from the
        # pre-clean declarations for tiles that stay whole, matching legacy
        # semantics (legacy never pre-cleaned whole tiles at parse time, so
        # their contribution to the persisted global interface set was always
        # the raw '*'-declaration scan, never shrunk by a threshold removal
        # decision made independently -- and possibly differently -- by each
        # neighbor tile).  Captured from the ORIGINAL (pre-split)
        # metadata.tile_configs, before _apply_tile_splits mutates it below.
        _raw_boundary_by_tile: Dict[tuple, Set[str]] = {
            tc.tile_id: raw
            for tc, raw in zip(metadata.tile_configs, _per_tile_raw_boundaries)
        }

        # Finding F11: ship the (potentially large) shared-boundary and pad
        # candidate sets to parse map tasks via a single backend.put() call
        # when the backend has a real shared object store (Ray) instead of
        # embedding them by value into every one of the N per-tile args
        # tuples -- LocalBackend's put() is a no-op passthrough (its map_func
        # never dereferences anything), so this is safe for both backends.
        _shared_bnd_arg = be.put(shared_bnd)
        _pad_candidates_arg = be.put(pad_candidates)

        args_list = [
            (
                tc.ckt_path, tc.nd_path, tc.net_filter, tc.tile_id,
                tc.instance_path, str(out_path), die_candidates, net_name,
                _shared_bnd_arg, _pad_candidates_arg,
            )
            for tc in metadata.tile_configs
        ]
        worker_results = be.map_func(parse_and_dump_tile, args_list)

        # 3. Merge per-tile die_attachment_net_map from UNSPLIT worker results.
        #
        # ORDERING RATIONALE (die-node ordering fix):
        # _apply_tile_splits reads metadata.package_data.die_attachment_nodes to
        # build parent_port_nodes for pre_clean.  On the common path the initial
        # _parse_package already populates die_attachment_nodes via a pattern scan
        # of all_resistor_nodes.  However, when that initial parse yields empty
        # pad_nodes (fallback path), die_attachment_nodes is only populated after
        # the worker-validated die_net_map is merged and package is re-parsed.
        # If _apply_tile_splits ran first (old ordering), die_attachment_nodes would
        # be empty at split time, causing pre_clean to drop cap-stub components that
        # are connected only through die-attachment nodes — producing up to 173 µV
        # (BE) / 509 µV (TR) transient divergence.
        #
        # Fix: resolve die_attachment_nodes BEFORE calling _apply_tile_splits.
        # The die_attachment_net_map is collected from the UNSPLIT worker results
        # (original tiles know which die nodes they contain; sub-tiles inherit
        # subsets of that map in _apply_tile_splits itself).
        merged_die_map: Dict[str, str] = {}
        for r in worker_results:
            merged_die_map.update(r.get('die_attachment_net_map', {}))

        # Fallback: re-parse package with die_net_map if initial parse found no pads
        #
        # Finding F2: whole-tile connectivity summaries (computed above, in
        # parse_and_dump_tile) were built against `pad_candidates` as it was
        # captured BEFORE this point -- the (possibly empty) pre-fallback pad
        # set.  If this fallback fires, every whole tile's `has_pad` flags are
        # therefore wrong (computed against an empty pad set), while any
        # sub-tile produced by _apply_tile_splits below sees the CORRECT
        # (post-fallback) pad set, because that call happens after this
        # block.  Rather than silently shipping an internally-inconsistent
        # bundle, mark it: connectivity_summary_version is set to a sentinel
        # that never matches CONNECTIVITY_SUMMARY_VERSION (below), forcing
        # model creation to always take the full legacy fallback path
        # (worker-side removal + coordinator Schur-BFS) for this bundle,
        # end-to-end, regardless of island_detection='auto'/'summaries'.
        _pads_fallback_fired = False
        if not metadata.package_data.pad_nodes and merged_die_map:
            logger.info(
                "No pad nodes from initial package parse; "
                "re-parsing with worker-validated die_net_map (%d entries)",
                len(merged_die_map),
            )
            metadata.package_data = self._parse_package(
                metadata.net_name, metadata.vdd, die_net_map=merged_die_map,
            )
            _pads_fallback_fired = True

        # Set die_attachment_net_map and narrow die_attachment_nodes once.
        # After this point die_attachment_nodes is fully resolved and safe to use
        # in _apply_tile_splits → _pre_clean_tile_data.
        if merged_die_map:
            metadata.package_data.die_attachment_net_map = merged_die_map
            metadata.package_data.die_attachment_nodes = set(merged_die_map.keys())

        # 2b. Optional: split oversized tiles (B1 balanced retiling).
        # Runs AFTER die_attachment_nodes is resolved so the sub-tile pass
        # inside _apply_tile_splits receives the full die_attachment_nodes set
        # (its own boundary-node port candidates come from each sub-tile's
        # own already-complete sub.boundary_nodes -- see that method).
        parent_removal_diagnostics: Dict[tuple, Dict[str, Any]] = {}
        if max_interior is not None:
            worker_results, metadata, parent_removal_diagnostics = self._apply_tile_splits(
                worker_results, metadata, out_path, max_interior, pickle,
                shared_bnd,
            )

        # 3b. Recompute the FINAL shared_boundary_nodes.
        #
        # Finding F1: whole (never-split) tiles must contribute their
        # PRE-clean boundary declarations (`_raw_boundary_by_tile`, the
        # step-0 scan state), NOT their post-pre-clean (possibly shrunk)
        # `tile_data.boundary_nodes`.  Legacy never ran island removal on
        # whole tiles at parse time, so a node kept as a port in one tile
        # (>=5 local interface candidates) but removed from EVERY neighbor
        # tile's independent local pre-clean pass still counted as globally
        # "shared" (declared '*' in 2+ tiles at raw-scan time) in legacy, and
        # remained a genuine port wherever it survived -- the coordinator
        # union-find (or the legacy Schur-BFS) then judges liveness/islanding
        # correctly.  Using the post-pre-clean set instead silently demotes
        # such a surviving port to an interior-only node (it is no longer
        # "shared" once every neighbor's local view of it disappears),
        # turning a legitimately-kept component into an unelidable,
        # portless, singular G_ii under summaries-mode trust.
        #
        # Split (sub-)tiles are exempt from this substitution: their
        # post-split boundary sets (r['boundary_nodes']) trace back through
        # _build_halves to the PARENT tile as it was pre-cleaned at parse
        # time -- this mirrors the pre-Stage-1e split path, which explicitly
        # pre-cleaned the parent BEFORE splitting for exactly this reason.
        # Sub-tiles are identified by their 3-tuple tile_id (parent (x, y) ->
        # sub-tile (x, y, k)); a tile that failed to split, or didn't need
        # to, keeps its original 2-tuple id and is treated as "whole".
        per_tile_boundaries = [
            _raw_boundary_by_tile[r['tile_id']]
            if len(r['tile_id']) == 2 and r['tile_id'] in _raw_boundary_by_tile
            else r['boundary_nodes']
            for r in worker_results
        ]
        shared_boundary_nodes = compute_shared_boundary_nodes(per_tile_boundaries)

        if not metadata.package_data.pad_nodes:
            logger.warning(
                "No pad (voltage source) nodes found for net '%s' after package parse. "
                "The solver will likely fail with a singular matrix. "
                "Check that package.ckt contains voltage sources for this net.",
                metadata.net_name,
            )

        # Log stats
        all_boundary_count = len(set().union(*per_tile_boundaries)) if per_tile_boundaries else 0
        n_single = all_boundary_count - len(shared_boundary_nodes)
        logger.info(
            f"Boundary node filtering: {all_boundary_count} total, "
            f"{len(shared_boundary_nodes)} shared (2+ tiles), "
            f"{n_single} single-tile-only filtered"
        )

        for r in worker_results:
            tile_str = '_'.join(str(c) for c in r['tile_id'])
            logger.info(
                f"Tile ({tile_str}): {r['n_nodes']} nodes, "
                f"{r['n_edges']} edges, "
                f"{r['n_currents']} current sources -> tile_{tile_str}.pkl"
            )

        # Stage 1e: aggregate per-tile connectivity summaries (one component
        # decomposition traversal already paid for by pre-clean above; no
        # extra pass here).  Sub-tiles produced by _apply_tile_splits already
        # REPLACE their parent's entry in worker_results (the parent's own
        # dict never survives when a split occurs), so this flatten naturally
        # picks up sub-tile summaries instead of stale parent ones.
        # parser_interface_set is the coordinator-authoritative interface set
        # the union-find (pgmath.schur.detect_interface_islands_from_summaries)
        # is scoped against; create_distributed_model() asserts equality
        # against its own independently-derived interface_nodes before
        # trusting these summaries (Stage 1e trust assertion).
        component_summaries: List[Dict[str, Any]] = []
        for r in worker_results:
            component_summaries.extend(r.get('component_summaries', ()))

        # Finding F4: aggregate in a deterministic order (see
        # _sort_component_summaries docstring).
        component_summaries = _sort_component_summaries(component_summaries)

        # Finding F13: single-sourced interface-node formula (also used by
        # model._create_distributed_model_from_bundle) -- must mirror it
        # exactly or the trust assertion spuriously fails and silently
        # disables the summaries path.
        from .tile_parsing import compute_interface_nodes, verify_component_keep_decisions
        die_net_map = getattr(
            metadata.package_data, 'die_attachment_net_map', {}) or {}
        parser_interface_set: Set[str] = compute_interface_nodes(
            shared_boundary_nodes, die_net_map,
            metadata.package_data.die_attachment_nodes,
        )

        # Finding R1: parse-end consistency re-check (all-or-nothing
        # philosophy).  A non-largest component's keep decision was made
        # against the port-candidate set available AT THAT PASS (raw
        # shared_bnd for whole tiles; sub.boundary_nodes-derived candidates
        # for sub-tiles) -- not against the FINAL shared_boundary_nodes just
        # recomputed above (step 3b).  A raw-shared candidate can be
        # silently demoted (dropped to a single declaring tile) when a
        # SPLIT neighbor's own parse-time pre-clean removed the candidate's
        # fragment there before the split ever ran (see the step-3b comment
        # above) -- leaving a SIBLING tile's already-kept component
        # under-ported relative to its own keep decision.  Neither the F10
        # structural-completeness check (model.py, worker-ports-subset-of-
        # candidates) nor the interface_nodes == parser_interface_set trust
        # assertion observes this: both see the ORIGINAL (still-listed)
        # candidate set, not that the FINAL interface set has shrunk out
        # from under it.  Re-evaluate every kept component's decision here,
        # once, against the FINAL interface set.
        _keep_ok, _keep_violations = verify_component_keep_decisions(
            component_summaries, parser_interface_set,
        )
        if not _keep_ok:
            for _v in _keep_violations[:10]:
                _cand = _v.get('candidates') or ()
                _n_final = len(set(_cand) & parser_interface_set)
                logger.info(
                    "Stage 1e parse-end consistency check: tile %s kept a "
                    "non-largest component (n_nodes=%s) whose keep decision "
                    "required >=%s interface candidate(s) but only %s "
                    "survive against the FINAL interface set (had %s at "
                    "parse time) -- a sibling tile's split-side pre-clean "
                    "likely demoted a raw-shared candidate. Dropping "
                    "connectivity_summary_version so this bundle always "
                    "falls back to the legacy Schur-BFS island detection "
                    "path end-to-end.",
                    _v.get('tile_id'), _v.get('n_nodes'), _v.get('keep_threshold'),
                    _n_final, len(_cand),
                )
            if len(_keep_violations) > 10:
                logger.info(
                    "Stage 1e parse-end consistency check: %d additional "
                    "violation(s) not shown.", len(_keep_violations) - 10,
                )

        # Finding F2: if the pads fallback re-parse fired above, the
        # whole-tile summaries (has_pad flags) are internally inconsistent
        # with the sub-tile summaries -- drop trust in the WHOLE bundle by
        # persisting a connectivity_summary_version that can never match
        # CONNECTIVITY_SUMMARY_VERSION, forcing model creation onto the full
        # legacy (worker-removal + Schur-BFS) path end-to-end.
        _persisted_summary_version: Optional[int] = CONNECTIVITY_SUMMARY_VERSION
        if not _keep_ok:
            _persisted_summary_version = None
        if _pads_fallback_fired:
            logger.info(
                "Stage 1e: pads fallback re-parse fired for net '%s' (initial "
                "package parse found no pad nodes; re-parsed using "
                "worker-validated die_net_map). Whole-tile connectivity "
                "summaries were computed against the pre-fallback (empty) pad "
                "set and are therefore untrustworthy for has_pad-based island "
                "rescue -- dropping connectivity_summary_version so this "
                "bundle always falls back to the legacy Schur-BFS island "
                "detection path end-to-end.",
                metadata.net_name,
            )
            _persisted_summary_version = None

        # 4. Dump metadata + shared boundary nodes + Stage 1e connectivity summaries
        meta_pkl_path = out_path / 'metadata.pkl'
        with open(meta_pkl_path, 'wb') as f:
            pickle.dump(
                {
                    'metadata': metadata,
                    'boundary_nodes': shared_boundary_nodes,
                    'connectivity_summary_version': _persisted_summary_version,
                    'parser_interface_set': parser_interface_set,
                    'component_summaries': component_summaries,
                    'parent_removal_diagnostics': parent_removal_diagnostics,
                },
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

        logger.info(
            f"Metadata: {len(metadata.tile_configs)} tiles, "
            f"{len(shared_boundary_nodes)} shared boundary nodes, "
            f"{len(component_summaries)} connectivity summaries -> {meta_pkl_path.name}"
        )

        # 5. Build and return ParsedTileBundle (lazy import to avoid circular)
        from .model import ParsedTileBundle  # noqa: circular-safe (function-level)

        bundle = ParsedTileBundle(
            metadata=metadata,
            shared_boundary_nodes=shared_boundary_nodes,
            pkl_dir=str(out_path),
            connectivity_summary_version=_persisted_summary_version,
            parser_interface_set=parser_interface_set,
            component_summaries=component_summaries,
            parent_removal_diagnostics=parent_removal_diagnostics,
        )

        return out_path, bundle

    # ------------------------------------------------------------------
    # B1: tile-splitting helper
    # ------------------------------------------------------------------

    def _apply_tile_splits(
        self,
        worker_results: List[Dict],
        metadata: 'PowerGridMetaData',
        out_path: 'Path',
        max_interior: int,
        pickle_mod,
        shared_bnd: Set[str],
    ):
        """Split oversized tiles and update worker_results + metadata in-place.

        For each tile whose interior node count > max_interior:
        1. Loads the already-written pkl from *out_path*.  Stage 1e: the pkl
           was pre-cleaned in memory by ``parse_and_dump_tile`` at threshold=5
           BEFORE it was ever written, so the parent-level pre-clean pass this
           function used to run here is now redundant -- the loaded tile
           already carries ``pre_cleaned_full=True`` and a clean topology.
        2. Calls :func:`retile.split_tile` recursively.
        3. Writes sub-tile pkls, deletes the original.
        4. Runs the sub-tile threshold=1 pre-clean pass on EACH sub-tile
           (port candidates = ``sub.boundary_nodes ∪ die_attachment_nodes ∪
           (sub.all_nodes ∩ shared_bnd)`` -- see the inline comment at the
           call site for why the sub-tile's own already-complete boundary set
           ALONE is not sufficient, finding F5), replacing the parent's
           connectivity summary with the sub-tiles' summaries.
        5. Replaces the tile's entry in worker_results and metadata.tile_configs.

        Sub-tiles carry ``TileData.pre_cleaned=True`` (set explicitly by the
        sub-tile ``_pre_clean_tile_data(..., mark_split_pre_cleaned=True)``
        call below -- finding F3; NOT by the whole-tile pass in
        ``parse_and_dump_tile``, which must leave it False), which causes
        :meth:`TileWorker._remove_floating_islands` to use a threshold of 1
        (instead of 5) in the legacy fallback path so that nodes legitimately
        connected through cut interface nodes are never wrongly discarded.

        Args:
            worker_results: List returned by ``be.map_func``.  Modified in-place.
            metadata: PowerGridMetaData with tile_configs list. Modified in-place.
            out_path: Directory where pkls live.
            max_interior: Threshold for splitting.
            pickle_mod: The ``pickle`` module (passed in to avoid re-import).
            shared_bnd: The step-0 global ``*``-declaration scan set (finding
                F5).  Required so a node '*'-declared only in OTHER tiles --
                but present unmarked in this tile's ``all_nodes`` (e.g. via an
                un-prefixed resistor endpoint) -- is still considered an
                interface-candidate at the sub-tile pass, matching both the
                whole-tile pass (which uses the same global set) and the
                legacy worker-time removal (which uses the finalized global
                interface set).  Finding R10: REQUIRED (no ``None`` default)
                -- production has exactly one call site (``parse_and_dump``,
                which always passes it); a ``None`` default would silently
                revert any future/test caller that omits it to the pre-F5
                (buggy) candidate set with no warning.  Pass ``set()``
                explicitly if a caller genuinely has no global candidates.

        Returns:
            ``(new_worker_results, metadata, parent_removal_diagnostics)``
            tuple.  ``parent_removal_diagnostics`` (finding R4) maps a
            SPLIT parent's original ``(x, y)`` tile id to
            ``{'islands_removed': int, 'removed_nodes': Set[str]}`` for
            parents whose whole-tile pre-clean (at ``parse_and_dump_tile``,
            before this method ever saw them) removed component(s) --
            reported by the coordinator under a synthetic parent-id
            ``island_stats`` entry (``model._merge_parent_removal_diagnostics``)
            instead of being misattributed to whichever sub-tile happens to
            be first.
        """
        from .retile import split_tile, _pre_clean_tile_data

        pad_nodes = metadata.package_data.pad_nodes
        _shared_bnd = shared_bnd

        # Build lookup: tile_id → TileConfig
        tc_by_id = {tc.tile_id: tc for tc in metadata.tile_configs}

        new_results: List[Dict] = []
        new_tile_configs: List[TileConfig] = []
        # Track tiles that remain over max_interior after all split attempts
        # (coupling caps on every candidate, identical coordinates, etc.).
        n_tiles_over_max: int = 0
        # Finding R4: parse-time WHOLE-TILE removal diagnostics, keyed by the
        # PARENT tile's own (x, y) id -- see the docstring Returns section.
        parent_removal_diagnostics: Dict[tuple, Dict[str, Any]] = {}

        for r in worker_results:
            tid = r['tile_id']
            n_interior = r['n_nodes'] - len(r['boundary_nodes'])

            if n_interior <= max_interior:
                new_results.append(r)
                new_tile_configs.append(tc_by_id[tid])
                continue

            # Need to split this tile
            tile_str = '_'.join(str(c) for c in tid)
            orig_pkl = out_path / f'tile_{tile_str}.pkl'

            try:
                with open(orig_pkl, 'rb') as f:
                    tile_data = pickle_mod.load(f)
            except Exception as exc:
                logger.warning(
                    "Tile %s: cannot load pkl for splitting (%s); keeping unsplit "
                    "(n_interior=%d > max_interior=%d)",
                    tile_str, exc, n_interior, max_interior,
                )
                new_results.append(r)
                new_tile_configs.append(tc_by_id[tid])
                n_tiles_over_max += 1
                continue

            # Stage 1e: the parent-level pre-clean that used to run HERE is now
            # redundant -- parse_and_dump_tile() already pre-cleaned this exact
            # tile (threshold=5, port_nodes = pre-split shared_bnd ∪
            # die_attachment_nodes) in memory before writing the pkl we just
            # loaded.  `tile_data.pre_cleaned_full` is True and its topology is
            # already clean w.r.t. the pre-split interface set.
            die_attachment_nodes = metadata.package_data.die_attachment_nodes

            sub_tiles = split_tile(tile_data, max_interior)

            if len(sub_tiles) == 1:
                # Split refused (e.g. coupling caps block all candidates)
                new_results.append(r)
                new_tile_configs.append(tc_by_id[tid])
                n_tiles_over_max += 1
                continue

            # Delete original pkl; write sub-tile pkls
            orig_pkl.unlink(missing_ok=True)
            tc = tc_by_id[tid]
            parent_die_map = r.get('die_attachment_net_map', {})

            # Findings F8/F9 + R4: the loaded PARENT tile_data already
            # carries its own removal diagnostics from the whole-tile
            # pre-clean pass that ran in parse_and_dump_tile (before this pkl
            # was ever written).  Those nodes no longer exist in
            # `tile_data.all_nodes` and cannot be assigned to either sub-tile
            # by split_tile(), so their NAMES would otherwise be silently
            # lost once the parent object is discarded below -- forward them
            # (union, never overwrite) onto the FIRST sub-tile (transport
            # only) so the floating-nodes report can still recover them.
            # _pre_clean_tile_data's sub-tile pass below UNIONS (never
            # overwrites) this field, so seeding here is safe regardless of
            # call order.
            #
            # R4 fix: the parent's removal COUNT must NOT be forwarded onto
            # sub_tiles[0] -- that misattributes the WHOLE parent tile's
            # pre-clean to whichever sub-tile happens to be first, even when
            # the removed components physically lay in a DIFFERENT
            # sub-tile's region (finding: island_stats[(x,y,0)] reporting the
            # parent's count while the sub-tile that actually contained the
            # removed components reports 0).  Record it instead in
            # parent_removal_diagnostics, keyed by the PARENT's own (x, y)
            # id, so the coordinator can report it under a synthetic
            # parent-id island_stats entry -- sub-tiles report ONLY their own
            # sub-clean removals.
            _parent_removed = getattr(tile_data, 'removed_floating_nodes', None) or set()
            _parent_n_removed = getattr(tile_data, 'n_floating_components_removed', 0)
            if _parent_removed:
                sub_tiles[0].removed_floating_nodes = (
                    set(getattr(sub_tiles[0], 'removed_floating_nodes', None) or ())
                    | _parent_removed
                )
            if _parent_n_removed:
                parent_removal_diagnostics[tid] = {
                    'islands_removed': _parent_n_removed,
                    'removed_nodes': set(_parent_removed),
                }

            for sub in sub_tiles:
                sub_str = '_'.join(str(c) for c in sub.tile_id)

                # Stage 1e sub-tile pass: threshold=1, port candidates =
                # sub.boundary_nodes | die_attachment_nodes | (sub.all_nodes
                # & shared_bnd).
                #
                # sub.boundary_nodes is a CORRECT but NOT COMPLETE port-
                # candidate set here -- not just the newly-introduced cut
                # nodes, and not just pre_split_shared_bnd.  _build_halves
                # unconditionally propagates the ENTIRE parent boundary set
                # into BOTH children (`left_bnd = right_bnd = cut_right_guests
                # | orig_boundary`), so ANY parent-boundary node -- even one
                # that was NOT pre-split-shared (declared '*' in only ONE
                # original tile's .ckt, e.g. a coordinate with no partner tile
                # in this particular netlist) -- ends up duplicated into >=2
                # sub-tiles by the split itself and therefore SHOWS UP as a
                # genuine shared (multi-tile) interface node once
                # compute_shared_boundary_nodes re-aggregates post-split
                # (parse_and_dump step "3b").  A first version of this pass
                # used pre_split_shared_bnd | cut_nodes instead and silently
                # dropped exactly these split-induced-shared nodes from every
                # component summary (0 summaries referenced them, despite
                # them being genuine S_global rows) -- the union-find then
                # correctly treated them as unreached singletons and wrongly
                # islanded them (625 nodes on netlist_sampled at
                # max_interior=8000).  root_parent_boundary ⊆ sub.boundary_nodes
                # always holds (by induction over _build_halves recursion), so
                # sub.boundary_nodes ∪ die_attachment_nodes subsumes the
                # pre_split_shared_bnd case with no extra union needed.
                #
                # Finding F5 (remaining gap): a node X that is present in
                # sub.all_nodes UNMARKED (e.g. an ordinary resistor endpoint
                # with no '*' prefix in THIS original tile) but '*'-declared
                # by 2+ OTHER tiles entirely is globally shared (X ∈
                # shared_bnd) and WAS correctly considered a port candidate at
                # the parent-level whole-tile pass (which uses the global
                # shared_bnd set) -- but X is not in sub.boundary_nodes (never
                # '*'-marked here) and is not necessarily a cut node either, so
                # it is silently dropped as a candidate at this sub-tile pass
                # without the explicit `& shared_bnd` term, causing a
                # legitimately-connected fragment to be removed by the
                # threshold-1 check.  Adding `sub.all_nodes & shared_bnd`
                # mirrors the whole-tile pass and the legacy worker-time
                # removal (both scoped against the global interface set, not
                # just this tile's own boundary declarations).
                #
                # Sets sub.pre_cleaned_full=True AND sub.pre_cleaned=True
                # (mark_split_pre_cleaned=True -- finding F3: this IS the
                # genuine post-split sub-tile pass, so the legacy fallback
                # threshold=1 selection is correct here), and replaces the
                # parent's (now stale) connectivity summary with this
                # sub-tile's own.
                sub_port_nodes = (
                    sub.boundary_nodes
                    | die_attachment_nodes
                    | (sub.all_nodes & _shared_bnd)
                )
                n_removed_sub, sub_summaries = _pre_clean_tile_data(
                    sub, sub_port_nodes, min_interface_keep=1,
                    pad_nodes=pad_nodes, mark_split_pre_cleaned=True,
                )

                sub_pkl = out_path / f'tile_{sub_str}.pkl'
                with open(sub_pkl, 'wb') as f:
                    pickle_mod.dump(sub, f, protocol=pickle_mod.HIGHEST_PROTOCOL)

                # Die attachment nodes: only those that ended up in this sub-tile
                sub_die_map = {
                    n: v for n, v in parent_die_map.items()
                    if n in sub.all_nodes
                }

                sub_interior = len(sub.all_nodes) - len(sub.boundary_nodes)
                # Check if the sub-tile is still over max_interior (recursive
                # bisection may have exhausted all valid cut points for its
                # own branch before reaching the threshold).
                if sub_interior > max_interior:
                    n_tiles_over_max += 1
                logger.info(
                    "Tile %s → sub-tile %s: %d nodes (%d interior), "
                    "%d edges, %d current sources -> tile_%s.pkl "
                    "(sub-clean removed %d component(s))",
                    tile_str, sub_str,
                    len(sub.all_nodes), sub_interior,
                    len(sub.resistive_edges),
                    len(sub.current_injections),
                    sub_str, n_removed_sub,
                )

                new_results.append({
                    'tile_id': sub.tile_id,
                    'boundary_nodes': sub.boundary_nodes,
                    'die_attachment_net_map': sub_die_map,
                    'n_nodes': len(sub.all_nodes),
                    'n_edges': len(sub.resistive_edges),
                    'n_currents': len(sub.current_injections),
                    'n_cap_edges': len(sub.capacitive_edges),
                    'islands_removed_at_parse': n_removed_sub,
                    'component_summaries': sub_summaries,
                })
                # Sub-tile TileConfig re-uses parent ckt/nd/instance paths.
                # VCS init loads from instance_path but filters to sub-tile nodes.
                new_tile_configs.append(TileConfig(
                    tile_id=sub.tile_id,
                    ckt_path=tc.ckt_path,
                    nd_path=tc.nd_path,
                    instance_path=tc.instance_path,
                    net_filter=tc.net_filter,
                ))

        # Major 4: Surface oversized-tile count loudly so it is never silent.
        if n_tiles_over_max > 0:
            logger.warning(
                "_apply_tile_splits: %d tile(s) still have n_interior > max_interior=%d "
                "after all split attempts. These tiles remain oversized and will take "
                "longer to factor. Possible causes: (a) all split candidates cross a "
                "coupling cap, (b) all interior nodes share identical (x, y) coordinates "
                "(name-based bisection not yet implemented). To suppress this warning, "
                "increase max_interior or remove coupling caps between distant nodes.",
                n_tiles_over_max, max_interior,
            )
        else:
            logger.info(
                "_apply_tile_splits: all %d tile(s) split successfully "
                "(0 tiles over max_interior=%d remaining).",
                len([r for r in new_results if r['n_nodes'] - len(r['boundary_nodes']) > 0]),
                max_interior,
            )

        metadata.tile_configs = new_tile_configs
        return new_results, metadata, parent_removal_diagnostics

    def collect_boundary_nodes(self, tile_configs: List[TileConfig]) -> Set[str]:
        """Pre-scan all tile .ckt files to collect *all* ``*``-prefixed boundary nodes.

        .. deprecated::
            Use :meth:`collect_shared_boundary_nodes` instead, which filters
            out nodes that appear in only one tile (no cross-tile coupling).

        This is a fast pass that only looks for the ``*`` prefix marker,
        without full element parsing.
        """
        import warnings
        warnings.warn(
            "collect_boundary_nodes() is deprecated. Use collect_shared_boundary_nodes() "
            "which filters out single-tile-only nodes.",
            DeprecationWarning,
            stacklevel=2,
        )
        boundary_nodes: Set[str] = set()
        for tc in tile_configs:
            boundary_nodes.update(self._scan_tile_boundary_nodes(tc.ckt_path))
        return boundary_nodes

    def collect_shared_boundary_nodes(self, tile_configs: List[TileConfig]) -> Set[str]:
        """Pre-scan all tile .ckt files to collect shared boundary nodes.

        Only returns ``*``-prefixed nodes that appear in 2 or more tiles.
        Nodes appearing in a single tile have no cross-tile coupling and are
        demoted to interior nodes, reducing per-tile Schur complement cost.

        Parameters
        ----------
        tile_configs : List[TileConfig]
            Tile configurations with ``ckt_path`` for each tile.

        Returns
        -------
        Set[str]
            Boundary node names (``*`` prefix stripped) appearing in 2+ tiles.
        """
        per_tile_boundaries = [
            self._scan_tile_boundary_nodes(tc.ckt_path)
            for tc in tile_configs
        ]

        shared = compute_shared_boundary_nodes(per_tile_boundaries)

        # Compute stats for logging
        from collections import Counter
        tile_count: Counter = Counter()
        for boundary_set in per_tile_boundaries:
            tile_count.update(boundary_set)
        all_boundary = set(tile_count.keys())
        n_single = len(all_boundary) - len(shared)

        logger.info(
            f"Boundary node filtering: {len(all_boundary)} total, "
            f"{len(shared)} shared (2+ tiles), {n_single} single-tile-only filtered"
        )

        return shared

    @staticmethod
    def _scan_tile_boundary_nodes(ckt_path: str) -> Set[str]:
        """Scan a single tile .ckt file for ``*``-prefixed boundary nodes.

        Parameters
        ----------
        ckt_path : str
            Path to the tile ``.ckt`` (or ``.ckt.gz``) file.

        Returns
        -------
        Set[str]
            Boundary node names with ``*`` prefix stripped.
        """
        boundary_nodes: Set[str] = set()
        with _open_file(ckt_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('*'):
                    continue
                tokens = line.split()
                if len(tokens) < 4:
                    continue
                first = tokens[0].lower()
                if first[0] in ('r', 'c'):  # Match _parse_tile_ckt: R and C lines
                    for t in tokens[1:3]:
                        if t.startswith('*'):
                            boundary_nodes.add(t[1:])  # Strip *
        return boundary_nodes

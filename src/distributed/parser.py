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
from typing import Dict, List, Optional, Set, Tuple

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
          - tile_X_Y.pkl  for each tile (pickled TileData)
          - metadata.pkl  with ``{'metadata': PowerGridMetaData,
            'boundary_nodes': Set[str]}``

        Workers parse AND dump their own tiles in parallel. The coordinator
        collects only lightweight per-tile boundary/die-attachment metadata,
        merges them, and writes ``metadata.pkl``.

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
        net_name = metadata.net_name

        args_list = [
            (
                tc.ckt_path, tc.nd_path, tc.net_filter, tc.tile_id,
                tc.instance_path, str(out_path), die_candidates, net_name,
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
        if not metadata.package_data.pad_nodes and merged_die_map:
            logger.info(
                "No pad nodes from initial package parse; "
                "re-parsing with worker-validated die_net_map (%d entries)",
                len(merged_die_map),
            )
            metadata.package_data = self._parse_package(
                metadata.net_name, metadata.vdd, die_net_map=merged_die_map,
            )

        # Set die_attachment_net_map and narrow die_attachment_nodes once.
        # After this point die_attachment_nodes is fully resolved and safe to use
        # in _apply_tile_splits → _pre_clean_tile_data.
        if merged_die_map:
            metadata.package_data.die_attachment_net_map = merged_die_map
            metadata.package_data.die_attachment_nodes = set(merged_die_map.keys())

        # 2b. Optional: split oversized tiles (B1 balanced retiling).
        # Runs AFTER die_attachment_nodes is resolved so _pre_clean_tile_data
        # receives the full port_nodes set (pre_split_shared_bnd | die_attachment_nodes).
        if max_interior is not None:
            worker_results, metadata = self._apply_tile_splits(
                worker_results, metadata, out_path, max_interior, pickle,
            )

        # 3b. Recompute per-tile boundary nodes from POST-split worker_results.
        # (Pre-split results were used above for die-map collection; post-split
        # results reflect any new sub-tile boundary nodes introduced by the cut.)
        per_tile_boundaries = [r['boundary_nodes'] for r in worker_results]
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

        # 4. Dump metadata + shared boundary nodes
        meta_pkl_path = out_path / 'metadata.pkl'
        with open(meta_pkl_path, 'wb') as f:
            pickle.dump(
                {'metadata': metadata, 'boundary_nodes': shared_boundary_nodes},
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

        logger.info(
            f"Metadata: {len(metadata.tile_configs)} tiles, "
            f"{len(shared_boundary_nodes)} shared boundary nodes -> {meta_pkl_path.name}"
        )

        # 5. Build and return ParsedTileBundle (lazy import to avoid circular)
        from .model import ParsedTileBundle  # noqa: circular-safe (function-level)

        bundle = ParsedTileBundle(
            metadata=metadata,
            shared_boundary_nodes=shared_boundary_nodes,
            pkl_dir=str(out_path),
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
    ):
        """Split oversized tiles and update worker_results + metadata in-place.

        For each tile whose interior node count > max_interior:
        1. Loads the already-written pkl from *out_path*.
        2. Pre-cleans the parent tile: runs island detection using the parent's
           global interface nodes (boundary_nodes shared across tiles in the
           pre-split partition).  This removes genuinely floating nodes before
           splitting so sub-tiles never need aggressive island removal.
        3. Calls :func:`retile.split_tile` recursively.
        4. Writes sub-tile pkls, deletes the original.
        5. Replaces the tile's entry in worker_results and metadata.tile_configs.

        Sub-tiles carry ``TileData.pre_cleaned=True``, which causes
        :meth:`TileWorker._remove_floating_islands` to use a threshold of 1
        (instead of 5) so that nodes legitimately connected through cut
        interface nodes are never wrongly discarded.

        Args:
            worker_results: List returned by ``be.map_func``.  Modified in-place.
            metadata: PowerGridMetaData with tile_configs list. Modified in-place.
            out_path: Directory where pkls live.
            max_interior: Threshold for splitting.
            pickle_mod: The ``pickle`` module (passed in to avoid re-import).

        Returns:
            ``(new_worker_results, metadata)`` tuple.
        """
        from .retile import split_tile, _pre_clean_tile_data

        # Build lookup: tile_id → TileConfig
        tc_by_id = {tc.tile_id: tc for tc in metadata.tile_configs}

        # Compute the pre-split shared boundary nodes for parent-level island
        # detection.  These are the nodes shared between tiles in the CURRENT
        # (unsplit) partition — they serve as port_nodes when running island
        # detection on a parent tile before it is bisected.
        pre_split_shared_bnd = compute_shared_boundary_nodes(
            [r['boundary_nodes'] for r in worker_results]
        )

        new_results: List[Dict] = []
        new_tile_configs: List[TileConfig] = []
        # Track tiles that remain over max_interior after all split attempts
        # (coupling caps on every candidate, identical coordinates, etc.).
        n_tiles_over_max: int = 0

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

            # --- B1 blocker fix: parent-level island detection ----------------
            # Run island detection on the PARENT tile before splitting.  Port
            # nodes must EXACTLY MATCH what create_distributed_model passes to
            # TileWorker._remove_floating_islands for the unsplit path:
            #   interface_nodes = shared_boundary_nodes | die_attachment_nodes
            # (same threshold=5 used by TileWorker.MIN_INTERFACE_NODES_KEEP).
            #
            # CRITICAL: the original bug used only
            #   tile_data.boundary_nodes & pre_split_shared_bnd
            # which omitted die_attachment_nodes.  Die nodes are in
            # tile_data.all_nodes but NOT in tile_data.boundary_nodes (they are
            # not marked '*' in the .ckt file), so intersecting with
            # boundary_nodes was silently dropping them.  This caused 434 nodes
            # to be removed from the split path that the unsplit path kept,
            # changing ~1710 fF of grounded capacitance and producing 173 µV
            # (BE) / 509 µV (TR) transient divergence.
            #
            # After this call tile_data.pre_cleaned=True.  That flag propagates
            # to all sub-tiles via split_tile/_build_halves, signalling
            # TileWorker to use threshold=1 rather than 5 when sub-tile island
            # detection runs.  A cut may create small components (< 5 port nodes)
            # that are still legitimate — they were connected in the parent and
            # remain connected through at least one cut-interface port node.
            die_attachment_nodes = metadata.package_data.die_attachment_nodes
            parent_port_nodes = pre_split_shared_bnd | die_attachment_nodes
            # _pre_clean_tile_data further intersects with tile_data.all_nodes
            _pre_clean_tile_data(tile_data, parent_port_nodes)

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

            for sub in sub_tiles:
                sub_str = '_'.join(str(c) for c in sub.tile_id)
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
                    "%d edges, %d current sources -> tile_%s.pkl",
                    tile_str, sub_str,
                    len(sub.all_nodes), sub_interior,
                    len(sub.resistive_edges),
                    len(sub.current_injections),
                    sub_str,
                )

                new_results.append({
                    'tile_id': sub.tile_id,
                    'boundary_nodes': sub.boundary_nodes,
                    'die_attachment_net_map': sub_die_map,
                    'n_nodes': len(sub.all_nodes),
                    'n_edges': len(sub.resistive_edges),
                    'n_currents': len(sub.current_injections),
                    'n_cap_edges': len(sub.capacitive_edges),
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
        return new_results, metadata

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

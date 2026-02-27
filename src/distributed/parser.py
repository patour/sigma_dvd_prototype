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


def _open_file(path: str):
    """Open a file, auto-detecting gzip compression."""
    if path.endswith('.gz'):
        return gzip.open(path, 'rt')
    return open(path, 'r')


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

    def _parse_package(self, net_name: str, vdd: float) -> PackageData:
        """Parse package.ckt for voltage sources and bump connections."""
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

        vsrc_dict: Dict[str, Dict] = {}
        package_edges: List[Tuple[str, str, float]] = []
        pad_nodes: Set[str] = set()
        tap_nodes: Set[str] = set()
        die_attachment_nodes: Set[str] = set()

        net_upper = net_name.upper()
        net_lower = net_name.lower()

        with _open_file(str(pkg_path)) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('*'):
                    continue

                tokens = line.split()
                first = tokens[0].lower()

                # Voltage source: v_VDD_XLV VDD_XLV_vsrc 0 VDD_XLV
                if first.startswith('v') and len(tokens) >= 4:
                    name = tokens[0]
                    node_pos = tokens[1]
                    node_neg = tokens[2]
                    # Check if this vsrc belongs to our net
                    # The 4th token is the net name reference
                    vsrc_net = tokens[3] if len(tokens) > 3 else ''
                    if vsrc_net.upper() == net_upper or net_lower in name.lower():
                        vsrc_dict[name] = {
                            'node_pos': node_pos,
                            'node_neg': node_neg,
                            'net': vsrc_net,
                            'value': vdd,
                        }
                        pad_nodes.add(node_pos)

                # Resistor: r VDD_XLV_vsrc VDD_XLV_tap_00000 0.001
                elif first.startswith('r') and len(tokens) >= 4:
                    if first[0] == 'r' and len(first) == 1:
                        # Unnamed: r node1 node2 value
                        node1, node2 = tokens[1], tokens[2]
                        try:
                            r_value = _parse_spice_value(tokens[3])
                        except (ValueError, IndexError):
                            continue
                    else:
                        # Named: R_name node1 node2 value
                        node1, node2 = tokens[1], tokens[2]
                        try:
                            r_value = _parse_spice_value(tokens[3])
                        except (ValueError, IndexError):
                            continue

                    # Convert to kOhm then conductance (mS)
                    r_kohm = r_value * R_TO_KOHM
                    if r_kohm <= 0 or r_kohm < 1e-6:
                        g = 1e5  # GMAX
                    else:
                        g = 1.0 / r_kohm

                    # Check if this edge belongs to our net
                    is_net_node1 = net_lower in node1.lower()
                    is_net_node2 = net_lower in node2.lower()
                    if is_net_node1 or is_net_node2:
                        package_edges.append((node1, node2, g))

                        # Classify nodes
                        if 'tap' in node1.lower():
                            tap_nodes.add(node1)
                        if 'tap' in node2.lower():
                            tap_nodes.add(node2)

                        # Die attachment nodes (M13 nodes connected to package)
                        for node in (node1, node2):
                            if '_M' in node and node not in pad_nodes and 'tap' not in node.lower() and 'vsrc' not in node.lower():
                                die_attachment_nodes.add(node)

        return PackageData(
            vsrc_dict=vsrc_dict,
            package_edges=package_edges,
            pad_nodes=pad_nodes,
            tap_nodes=tap_nodes,
            die_attachment_nodes=die_attachment_nodes,
            vdd=vdd,
            net_name=net_name,
        )

    def collect_boundary_nodes(self, tile_configs: List[TileConfig]) -> Set[str]:
        """Pre-scan all tile .ckt files to collect *-prefixed boundary nodes.

        This is a fast pass that only looks for the * prefix marker, without
        full element parsing.
        """
        boundary_nodes: Set[str] = set()

        for tc in tile_configs:
            with _open_file(tc.ckt_path) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('*'):
                        continue
                    # Look for *-prefixed nodes in element lines
                    tokens = line.split()
                    if len(tokens) < 4:
                        continue
                    # Element lines: type/name node1 node2 value ...
                    first = tokens[0].lower()
                    if first[0] in ('r', 'c', 'l', 'i', 'v'):
                        # Check node tokens for * prefix
                        for t in tokens[1:3]:
                            if t.startswith('*'):
                                boundary_nodes.add(t[1:])  # Strip *

        return boundary_nodes

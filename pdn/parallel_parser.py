#!/usr/bin/env python3
"""
Parallel PDN Netlist Parser Components.

This module provides components for parallel tile parsing using multiprocessing:
- TileParseResult, InstanceParseResult: Pickle-serializable result data classes
- MMapSpiceLineReader: Memory-mapped file reader with gzip fallback
- Worker functions for tile and instance model parsing

The parallel parsing strategy:
1. Main netlist parsing (sequential)
2. Tile parsing (PARALLEL using multiprocessing.Pool)
3. Instance model parsing (PARALLEL)
4. Merge phase (parallel aggregation, bulk graph insertion)
5. Post-processing (sequential)

Target: ~6-8x speedup for ~100 tiles with 8 workers.
"""

import gzip
import logging
import mmap
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Import parsing helpers from pdn_parser
from pdn.pdn_parser import (
    R_TO_KOHM, C_TO_FF, L_TO_NH, I_TO_MA,
    _parse_spice_value, _parse_pulse, _parse_pwl,
    Pulse, PWL, CurrentSource, InstanceInfo
)

# Pre-interned strings for memory optimization in worker processes
# Each worker process needs its own interned strings
import sys
_ELEM_R = sys.intern('R')
_ELEM_C = sys.intern('C')
_ELEM_L = sys.intern('L')
_ELEM_V = sys.intern('V')
_ELEM_I = sys.intern('I')

# Interned edge attribute keys
_KEY_NET_TYPE = sys.intern('net_type')
_KEY_HAS_BOUNDARY = sys.intern('has_boundary')
_KEY_BOUNDARY_NODE1 = sys.intern('boundary_node1')
_KEY_BOUNDARY_NODE2 = sys.intern('boundary_node2')
_KEY_DC = sys.intern('dc')
_KEY_NLCAP_MODEL = sys.intern('nlcap_model')


logger = logging.getLogger(__name__)


# =============================================================================
# Result Data Classes (Pickle-Serializable)
# =============================================================================

@dataclass(slots=True)
class TileParseResult:
    """
    Pickle-serializable result from parsing a single tile.

    Contains only primitive types (no graph objects) for efficient
    serialization across process boundaries.
    """
    tile_id: Tuple[int, int]

    # Edges: List of (node1, node2, elem_type, value, name, attrs_dict)
    edges: List[Tuple[str, str, str, float, str, Dict[str, Any]]] = field(default_factory=list)

    # Nodes: {node_name: {attr: value}}
    nodes: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Node-to-net mapping from .nd file
    node_net_map: Dict[str, str] = field(default_factory=dict)

    # Per-tile statistics (element counts)
    stats: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

    # Boundary nodes found in this tile
    boundary_nodes: List[str] = field(default_factory=list)

    # Warnings/errors encountered
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


@dataclass(slots=True)
class InstanceParseResult:
    """
    Pickle-serializable result from parsing instance models.

    Contains serialized current source data (as dicts) for efficient
    transfer between processes.
    """
    tile_id: Tuple[int, int]

    # Current sources: {name: CurrentSource.to_dict()}
    current_sources: Dict[str, Dict] = field(default_factory=dict)

    # Instance node mapping: {name: [node+, node-]}
    instance_node_map: Dict[str, List[str]] = field(default_factory=dict)

    # Statistics
    stats: Dict[str, Any] = field(default_factory=lambda: {
        'total': 0,
        'with_waveforms': 0,
        'total_static_current_ma': 0.0,
        'wscale_values': []
    })

    warnings: List[str] = field(default_factory=list)


# =============================================================================
# Memory-Mapped File Reader
# =============================================================================

class MMapSpiceLineReader:
    """
    Memory-mapped SPICE file reader with chunk support.

    Falls back to streaming gzip for compressed files since
    gzip doesn't support memory mapping.

    Usage:
        with MMapSpiceLineReader('tile_0_0.ckt') as reader:
            while True:
                chunk = reader.read_chunk()
                if not chunk:
                    break
                for line in chunk:
                    process(line)
    """

    def __init__(self, filepath: str, chunk_size: int = 10000):
        """
        Initialize reader.

        Args:
            filepath: Path to SPICE file
            chunk_size: Number of logical lines per chunk
        """
        self.filepath = filepath
        self.chunk_size = chunk_size
        self.mmap_obj: Optional[mmap.mmap] = None
        self.fd = None
        self.file_handle = None
        self.is_gzipped = False
        self.position = 0
        self.line_number = 0
        self._pending_line: Optional[str] = None
        self._eof = False

    def __enter__(self) -> 'MMapSpiceLineReader':
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def open(self):
        """Open file with automatic gzip detection."""
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"File not found: {self.filepath}")

        # Check for gzip magic bytes (0x1f8b)
        with open(self.filepath, 'rb') as f:
            magic = f.read(2)
            self.is_gzipped = (magic == b'\x1f\x8b')

        if self.is_gzipped:
            # Gzip doesn't support mmap - use streaming
            self.file_handle = gzip.open(self.filepath, 'rt', encoding='utf-8', errors='ignore')
        else:
            # Memory-map plain text file
            self.fd = open(self.filepath, 'rb')
            file_size = os.fstat(self.fd.fileno()).st_size
            if file_size > 0:
                self.mmap_obj = mmap.mmap(self.fd.fileno(), 0, access=mmap.ACCESS_READ)
            else:
                # Empty file
                self._eof = True

    def close(self):
        """Close file handles."""
        if self.mmap_obj:
            self.mmap_obj.close()
            self.mmap_obj = None
        if self.fd:
            self.fd.close()
            self.fd = None
        if self.file_handle:
            self.file_handle.close()
            self.file_handle = None

    def _read_line_mmap(self) -> Optional[str]:
        """Read a single raw line from memory-mapped file."""
        if self._eof or self.mmap_obj is None:
            return None

        # Find next newline
        start = self.position
        newline_pos = self.mmap_obj.find(b'\n', start)

        if newline_pos == -1:
            # No more newlines - read rest of file
            if start < len(self.mmap_obj):
                line = self.mmap_obj[start:].decode('utf-8', errors='ignore')
                self.position = len(self.mmap_obj)
                self._eof = True
                return line.rstrip('\r\n')
            else:
                self._eof = True
                return None

        line = self.mmap_obj[start:newline_pos].decode('utf-8', errors='ignore')
        self.position = newline_pos + 1
        return line.rstrip('\r\n')

    def _read_line_gzip(self) -> Optional[str]:
        """Read a single raw line from gzip file."""
        if self._eof or self.file_handle is None:
            return None

        line = self.file_handle.readline()
        if not line:
            self._eof = True
            return None
        return line.rstrip('\r\n')

    def _read_raw_line(self) -> Optional[str]:
        """Read raw line from either mmap or gzip."""
        if self.is_gzipped:
            return self._read_line_gzip()
        else:
            return self._read_line_mmap()

    def read_line(self) -> Optional[str]:
        """
        Read a logical line from SPICE file, handling:
        - Line continuation (lines starting with '+')
        - Comment removal (lines starting with '*')
        - Multiple whitespace normalization

        Returns None at EOF.
        """
        lines = []

        while True:
            # Check if we have a pending line from previous call
            if self._pending_line is not None:
                line = self._pending_line
                self._pending_line = None
            else:
                line = self._read_raw_line()
                if line is None:
                    break
                self.line_number += 1
                line = line.strip()

            # Skip empty lines
            if not line:
                continue

            # Skip comment lines (but not inline comments)
            if line.startswith('*'):
                continue

            # Check for continuation
            if line.startswith('+'):
                if lines:
                    # Append continuation (remove leading +)
                    lines.append(line[1:].strip())
                    continue
                else:
                    # Continuation without previous line - skip
                    logger.warning(f"Line {self.line_number}: Continuation '+' without previous line")
                    continue

            # If we have accumulated lines, this is a new statement
            # Save current line for next call
            if lines:
                self._pending_line = line
                break

            # Start new logical line
            lines.append(line)

        if not lines:
            return None

        # Join all parts, normalize whitespace
        logical_line = ' '.join(lines)

        # Remove inline comments (careful with * at start of token)
        parts = logical_line.split()
        cleaned_parts = []
        for i, part in enumerate(parts):
            if part.startswith('*') and i > 0:
                # Check if it's a boundary marker (alphanumeric after *)
                if len(part) > 1 and (part[1].isalnum() or part[1] == '_'):
                    cleaned_parts.append(part)  # Boundary marker
                else:
                    break  # Comment
            elif part == '*':
                break  # Standalone comment marker
            else:
                cleaned_parts.append(part)

        logical_line = ' '.join(cleaned_parts)
        return logical_line if logical_line else self.read_line()

    def read_chunk(self) -> List[str]:
        """
        Read a chunk of logical lines.

        Returns:
            List of logical lines (up to chunk_size), empty list at EOF.
        """
        lines = []
        for _ in range(self.chunk_size):
            line = self.read_line()
            if line is None:
                break
            lines.append(line)
        return lines


# =============================================================================
# Node-Net Mapping Loader
# =============================================================================

def _load_nd_file_worker(nd_filepath: str) -> Tuple[Dict[str, str], Dict[str, str], List[str]]:
    """
    Load node-to-net mapping from .nd file.

    Format: <node_name> <val1> <val2> <val3> <val4> <net_name>

    Args:
        nd_filepath: Path to .nd file

    Returns:
        Tuple of (node_net_map, node_net_map_lower, warnings)
    """
    node_map = {}
    node_map_lower = {}
    warnings = []

    try:
        with MMapSpiceLineReader(nd_filepath) as reader:
            while True:
                line = reader.read_line()
                if line is None:
                    break

                tokens = line.split()
                if len(tokens) >= 6:
                    node_name = tokens[0]
                    net_name = tokens[5]
                    node_map[node_name] = net_name
                    node_map_lower[node_name] = net_name.lower()
                elif len(tokens) > 0:
                    warnings.append(f"Invalid .nd line (expected 6 tokens, got {len(tokens)}): {line}")
    except FileNotFoundError:
        raise FileNotFoundError(f"Required .nd file not found: {nd_filepath}")
    except Exception as e:
        raise RuntimeError(f"Error reading .nd file {nd_filepath}: {e}")

    return node_map, node_map_lower, warnings


# =============================================================================
# Element Parsing Helpers
# =============================================================================

def _parse_element_line(
    line: str,
    node_net_map: Dict[str, str],
    node_net_map_lower: Dict[str, str],
    net_filter: Optional[str],
    tile_id: Tuple[int, int]
) -> Optional[Tuple[str, str, str, float, str, Dict[str, Any]]]:
    """
    Parse a single element line and return edge tuple.

    Args:
        line: SPICE element line
        node_net_map: Node-to-net mapping
        node_net_map_lower: Lowercase node-to-net mapping
        net_filter: Optional net filter (lowercase)
        tile_id: Current tile ID

    Returns:
        Tuple of (node1, node2, elem_type, value, name, attrs) or None if filtered
    """
    tokens = line.split()
    if not tokens:
        return None

    first_token = tokens[0]
    first_char = first_token[0].upper()

    # Skip non-element lines
    if first_char not in ('R', 'C', 'L', 'V', 'I') and first_token.lower() not in ('r', 'c', 'l'):
        return None

    # Parse element based on type
    try:
        if first_char == 'R' or first_token.lower() == 'r':
            return _parse_resistor_worker(tokens, node_net_map, node_net_map_lower, net_filter, tile_id)
        elif first_char == 'C' or first_token.lower() == 'c':
            return _parse_capacitor_worker(tokens, node_net_map, node_net_map_lower, net_filter, tile_id)
        elif first_char == 'L' or first_token.lower() == 'l':
            return _parse_inductor_worker(tokens, node_net_map, node_net_map_lower, net_filter, tile_id)
        elif first_char == 'V':
            return _parse_vsource_worker(tokens, node_net_map, node_net_map_lower, net_filter, tile_id)
        elif first_char == 'I':
            return _parse_isource_worker(tokens, line, node_net_map, node_net_map_lower, net_filter, tile_id)
    except Exception as e:
        logger.warning(f"Error parsing element: {line[:50]}...: {e}")
        return None

    return None


def _check_net_filter(
    node1: str,
    node2: str,
    node_net_map_lower: Dict[str, str],
    net_filter: Optional[str]
) -> bool:
    """
    Check if element passes net filter.

    Returns True if element should be included, False if filtered out.
    """
    if net_filter is None:
        return True

    node1_net = node_net_map_lower.get(node1)
    node2_net = node_net_map_lower.get(node2)

    return node1_net == net_filter or node2_net == net_filter


def _handle_boundary_node(node: str) -> Tuple[str, bool]:
    """Handle boundary node marker (*prefix)."""
    if node.startswith('*'):
        return node[1:], True
    return node, False


def _parse_resistor_worker(
    tokens: List[str],
    node_net_map: Dict[str, str],
    node_net_map_lower: Dict[str, str],
    net_filter: Optional[str],
    tile_id: Tuple[int, int]
) -> Optional[Tuple[str, str, str, float, str, Dict[str, Any]]]:
    """Parse resistor element."""
    first_token = tokens[0]

    if first_token.lower() == 'r':
        # Format: r <node1> <node2> <value>
        if len(tokens) < 4:
            return None
        name = f"r_{tokens[1]}_{tokens[2]}_{tile_id[0]}_{tile_id[1]}"
        node1, node2, value_token = tokens[1], tokens[2], tokens[3]
    else:
        # Format: R<name> <node1> <node2> <value>
        if len(tokens) < 4:
            return None
        name = tokens[0]
        node1, node2, value_token = tokens[1], tokens[2], tokens[3]

    node1, is_boundary1 = _handle_boundary_node(node1)
    node2, is_boundary2 = _handle_boundary_node(node2)

    if not _check_net_filter(node1, node2, node_net_map_lower, net_filter):
        return None

    value = _parse_spice_value(value_token) * R_TO_KOHM

    net_type = node_net_map.get(node1) or node_net_map.get(node2)
    attrs = {_KEY_NET_TYPE: net_type}
    if is_boundary1 or is_boundary2:
        attrs[_KEY_HAS_BOUNDARY] = True
        # Track which specific nodes are boundary nodes
        attrs[_KEY_BOUNDARY_NODE1] = is_boundary1
        attrs[_KEY_BOUNDARY_NODE2] = is_boundary2

    return (node1, node2, _ELEM_R, value, name, attrs)


def _parse_capacitor_worker(
    tokens: List[str],
    node_net_map: Dict[str, str],
    node_net_map_lower: Dict[str, str],
    net_filter: Optional[str],
    tile_id: Tuple[int, int]
) -> Optional[Tuple[str, str, str, float, str, Dict[str, Any]]]:
    """Parse capacitor element."""
    first_token = tokens[0]

    if first_token.lower() == 'c':
        # Format: c <node1> <node2> <value> [model]
        if len(tokens) < 4:
            return None
        name = f"c_{tokens[1]}_{tokens[2]}_{tile_id[0]}_{tile_id[1]}"
        node1, node2, value_token = tokens[1], tokens[2], tokens[3]
        model_idx = 4
    else:
        # Format: C<name> <node1> <node2> <value> [model]
        if len(tokens) < 4:
            return None
        name = tokens[0]
        node1, node2, value_token = tokens[1], tokens[2], tokens[3]
        model_idx = 4

    node1, is_boundary1 = _handle_boundary_node(node1)
    node2, is_boundary2 = _handle_boundary_node(node2)

    if not _check_net_filter(node1, node2, node_net_map_lower, net_filter):
        return None

    value = _parse_spice_value(value_token) * C_TO_FF

    net_type = node_net_map.get(node1) or node_net_map.get(node2)
    attrs = {_KEY_NET_TYPE: net_type}
    if is_boundary1 or is_boundary2:
        attrs[_KEY_HAS_BOUNDARY] = True
        attrs[_KEY_BOUNDARY_NODE1] = is_boundary1
        attrs[_KEY_BOUNDARY_NODE2] = is_boundary2

    # Check for nonlinear cap model
    if len(tokens) > model_idx and not tokens[model_idx].startswith('*'):
        attrs[_KEY_NLCAP_MODEL] = tokens[model_idx]

    return (node1, node2, _ELEM_C, value, name, attrs)


def _parse_inductor_worker(
    tokens: List[str],
    node_net_map: Dict[str, str],
    node_net_map_lower: Dict[str, str],
    net_filter: Optional[str],
    tile_id: Tuple[int, int]
) -> Optional[Tuple[str, str, str, float, str, Dict[str, Any]]]:
    """Parse inductor element."""
    first_token = tokens[0]

    if first_token.lower() == 'l':
        # Format: l <node1> <node2> <value>
        if len(tokens) < 4:
            return None
        name = f"l_{tokens[1]}_{tokens[2]}_{tile_id[0]}_{tile_id[1]}"
        node1, node2, value_token = tokens[1], tokens[2], tokens[3]
    else:
        # Format: L<name> <node1> <node2> <value>
        if len(tokens) < 4:
            return None
        name = tokens[0]
        node1, node2, value_token = tokens[1], tokens[2], tokens[3]

    node1, is_boundary1 = _handle_boundary_node(node1)
    node2, is_boundary2 = _handle_boundary_node(node2)

    if not _check_net_filter(node1, node2, node_net_map_lower, net_filter):
        return None

    value = _parse_spice_value(value_token) * L_TO_NH

    net_type = node_net_map.get(node1) or node_net_map.get(node2)
    attrs = {_KEY_NET_TYPE: net_type}
    if is_boundary1 or is_boundary2:
        attrs[_KEY_HAS_BOUNDARY] = True
        attrs[_KEY_BOUNDARY_NODE1] = is_boundary1
        attrs[_KEY_BOUNDARY_NODE2] = is_boundary2

    return (node1, node2, _ELEM_L, value, name, attrs)


def _parse_vsource_worker(
    tokens: List[str],
    node_net_map: Dict[str, str],
    node_net_map_lower: Dict[str, str],
    net_filter: Optional[str],
    tile_id: Tuple[int, int]
) -> Optional[Tuple[str, str, str, float, str, Dict[str, Any]]]:
    """Parse voltage source element."""
    if len(tokens) < 4:
        return None

    name = tokens[0]
    node_pos, is_boundary1 = _handle_boundary_node(tokens[1])
    node_neg, is_boundary2 = _handle_boundary_node(tokens[2])

    # Voltage sources pass net filter check (filtered later)
    try:
        dc_value = _parse_spice_value(tokens[3])
    except ValueError:
        return None

    net_type = node_net_map.get(node_pos) or node_net_map.get(node_neg)
    attrs = {_KEY_NET_TYPE: net_type, _KEY_DC: dc_value}
    if is_boundary1 or is_boundary2:
        attrs[_KEY_HAS_BOUNDARY] = True
        attrs[_KEY_BOUNDARY_NODE1] = is_boundary1
        attrs[_KEY_BOUNDARY_NODE2] = is_boundary2

    # Parse additional parameters
    i = 4
    while i < len(tokens):
        token = tokens[i].upper()
        if token == 'AC' and i + 1 < len(tokens):
            try:
                attrs['ac'] = float(tokens[i + 1])
            except ValueError:
                pass
            i += 2
        elif token == 'PORTID' and i + 1 < len(tokens):
            port_str = tokens[i + 1]
            if '=' in port_str:
                attrs['portid'] = port_str.split('=')[1]
            i += 2
        else:
            i += 1

    return (node_pos, node_neg, _ELEM_V, dc_value, name, attrs)


def _parse_isource_worker(
    tokens: List[str],
    line: str,
    node_net_map: Dict[str, str],
    node_net_map_lower: Dict[str, str],
    net_filter: Optional[str],
    tile_id: Tuple[int, int]
) -> Optional[Tuple[str, str, str, float, str, Dict[str, Any]]]:
    """Parse current source element (simplified for tile parsing)."""
    if len(tokens) < 4:
        return None

    name = tokens[0]
    node_pos, is_boundary1 = _handle_boundary_node(tokens[1])
    node_neg, is_boundary2 = _handle_boundary_node(tokens[2])

    if not _check_net_filter(node_pos, node_neg, node_net_map_lower, net_filter):
        return None

    # Parse DC value
    try:
        if tokens[3].lower() == 'dc' or tokens[3].lower().startswith('dc.'):
            dc_value = _parse_spice_value(tokens[4]) if len(tokens) > 4 else 0.0
        else:
            dc_value = _parse_spice_value(tokens[3])
    except (ValueError, IndexError):
        dc_value = 0.0

    dc_value_ma = dc_value * I_TO_MA

    net_type = node_net_map.get(node_pos) or node_net_map.get(node_neg)
    attrs = {_KEY_NET_TYPE: net_type, _KEY_DC: dc_value_ma}
    if is_boundary1 or is_boundary2:
        attrs[_KEY_HAS_BOUNDARY] = True
        attrs[_KEY_BOUNDARY_NODE1] = is_boundary1
        attrs[_KEY_BOUNDARY_NODE2] = is_boundary2

    return (node_pos, node_neg, _ELEM_I, dc_value_ma, name, attrs)


# =============================================================================
# Tile Worker Function
# =============================================================================

def _parse_tile_worker(args: Tuple) -> TileParseResult:
    """
    Parse single tile file. Runs in worker process.

    Args tuple: (tile_x, tile_y, ckt_path, nd_path, net_filter, chunk_size, config)

    Returns:
        TileParseResult with parsed edges, nodes, and statistics
    """
    tile_x, tile_y, ckt_path, nd_path, net_filter, chunk_size, config = args
    tile_id = (tile_x, tile_y)

    result = TileParseResult(tile_id=tile_id)
    result.stats = defaultdict(int)

    try:
        # 1. Load .nd file for node-net mapping
        node_net_map, node_net_map_lower, nd_warnings = _load_nd_file_worker(nd_path)
        result.node_net_map = node_net_map
        result.warnings.extend(nd_warnings)

        # 2. Parse .ckt file
        with MMapSpiceLineReader(ckt_path, chunk_size) as reader:
            while True:
                chunk = reader.read_chunk()
                if not chunk:
                    break

                for line in chunk:
                    edge = _parse_element_line(
                        line, node_net_map, node_net_map_lower, net_filter, tile_id
                    )
                    if edge:
                        node1, node2, elem_type, value, name, attrs = edge
                        result.edges.append(edge)
                        result.stats[elem_type] += 1

                        # Track nodes
                        if node1 not in result.nodes:
                            result.nodes[node1] = {'tile_id': tile_id}
                        if node2 not in result.nodes:
                            result.nodes[node2] = {'tile_id': tile_id}

                        # Track boundary nodes - attrs tracks which specific nodes had '*' prefix
                        if attrs.get('has_boundary'):
                            if attrs.get('boundary_node1'):
                                result.boundary_nodes.append(node1)
                            if attrs.get('boundary_node2'):
                                result.boundary_nodes.append(node2)

    except Exception as e:
        result.errors.append(f"Error parsing tile {tile_id}: {str(e)}")

    return result


# =============================================================================
# Instance Models Worker Function
# =============================================================================

def _parse_instance_worker(args: Tuple) -> InstanceParseResult:
    """
    Parse instance models file. Runs in worker process.

    Args tuple: (tile_x, tile_y, filepath, net_filter, chunk_size, config)

    Returns:
        InstanceParseResult with parsed current sources
    """
    tile_x, tile_y, filepath, net_filter, chunk_size, config = args
    tile_id = (tile_x, tile_y)

    result = InstanceParseResult(tile_id=tile_id)

    try:
        with MMapSpiceLineReader(filepath, chunk_size) as reader:
            while True:
                chunk = reader.read_chunk()
                if not chunk:
                    break

                for line in chunk:
                    if not line or line[0].lower() != 'i':
                        continue

                    # Parse current source using existing parser
                    from pdn.pdn_parser import _parse_current_source_line
                    isrc = _parse_current_source_line(line)
                    if isrc is None:
                        continue

                    # Handle boundary nodes
                    node_pos = isrc.node1
                    node_neg = isrc.node2
                    if node_pos.startswith('*'):
                        node_pos = node_pos[1:]
                    if node_neg.startswith('*'):
                        node_neg = node_neg[1:]

                    isrc.node1 = node_pos
                    isrc.node2 = node_neg

                    # Convert values to mA
                    isrc.dc_value *= I_TO_MA
                    if isrc.static_value is not None:
                        isrc.static_value *= I_TO_MA
                    for pulse in isrc.pulses:
                        pulse.v1 *= I_TO_MA
                        pulse.v2 *= I_TO_MA
                    for pwl in isrc.pwls:
                        pwl.points = [(t, v * I_TO_MA) for t, v in pwl.points]

                    # Store serialized current source
                    result.current_sources[isrc.name] = isrc.to_dict()
                    result.instance_node_map[isrc.name] = [node_pos, node_neg]

                    # Update statistics
                    result.stats['total'] += 1
                    if isrc.has_waveform_data():
                        result.stats['with_waveforms'] += 1
                        result.stats['wscale_values'].append(isrc.wscale)
                    result.stats['total_static_current_ma'] += abs(isrc.get_static_current())

    except Exception as e:
        result.warnings.append(f"Error parsing instance models {tile_id}: {str(e)}")

    return result


# =============================================================================
# Worker Argument Builders
# =============================================================================

def _build_tile_worker_args(
    tile_queue: List[Tuple[int, int, str]],
    netlist_dir: Path,
    net_filter: Optional[str],
    chunk_size: int,
    config: Dict[str, Any]
) -> List[Tuple]:
    """
    Build argument tuples for tile workers.

    Args:
        tile_queue: List of (x, y, ckt_filepath) tuples
        netlist_dir: Base netlist directory
        net_filter: Optional net filter (lowercase)
        chunk_size: Lines per chunk
        config: Additional configuration

    Returns:
        List of argument tuples for _parse_tile_worker
    """
    worker_args = []

    for x, y, ckt_path in tile_queue:
        # Find corresponding .nd file
        ckt_path = Path(ckt_path)
        nd_path = ckt_path.parent / f"tile_{x}_{y}.nd"
        if not nd_path.exists():
            nd_path = ckt_path.parent / f"tile_{x}_{y}.nd.gz"

        worker_args.append((
            x, y,
            str(ckt_path),
            str(nd_path),
            net_filter,
            chunk_size,
            config
        ))

    return worker_args


def _build_instance_worker_args(
    instance_queue: List[Tuple[int, int, str]],
    net_filter: Optional[str],
    chunk_size: int,
    config: Dict[str, Any]
) -> List[Tuple]:
    """
    Build argument tuples for instance model workers.

    Args:
        instance_queue: List of (x, y, filepath) tuples
        net_filter: Optional net filter (lowercase)
        chunk_size: Lines per chunk
        config: Additional configuration

    Returns:
        List of argument tuples for _parse_instance_worker
    """
    worker_args = []

    for x, y, filepath in instance_queue:
        worker_args.append((
            x, y,
            filepath,
            net_filter,
            chunk_size,
            config
        ))

    return worker_args

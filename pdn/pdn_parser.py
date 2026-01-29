#!/usr/bin/env python3
"""
PDN Netlist Parser - Converts SPICE-like Power Delivery Network netlists to NetworkX graphs.

This parser reads hierarchical, tile-based PDN netlists (as used by the mpower simulator)
and constructs a NetworkX MultiDiGraph representation suitable for power grid analysis.

Features:
- Automatic gzip detection (files with/without .gz extension)
- Tile-based parallel netlist parsing with progress tracking
- Subcircuit flattening with hierarchical naming
- Instance-to-node mapping for current source tracking
- Sanity validation (short detection, floating nodes, merged nodes)
- Package model support with node marking
- FSDB waveform metadata extraction

Usage Examples:
    # Basic parsing
    parser = NetlistParser('/path/to/netlist/dir')
    graph = parser.parse()
    
    # With validation
    parser = NetlistParser('/path/to/netlist/dir', validate=True)
    graph = parser.parse()
    
    # Query the graph
    # Find all resistors
    resistors = [(u, v, d) for u, v, d in graph.edges(data=True) if d['type'] == 'R']
    
    # Find current sources connected to specific node
    node = 'vdd_1000_2000'
    isrcs = [(u, v, d) for u, v, d in graph.edges(data=True) 
             if d['type'] == 'I' and (u == node or v == node)]
    
    # Get instance to node mapping
    inst_map = graph.graph['instance_node_map']
    nodes = inst_map['i_cpu_core:inst1:vdd:0']  # Returns list of node names
    
    # Filter package nodes
    pkg_nodes = [n for n, d in graph.nodes(data=True) if d.get('is_package', False)]
    
    # Trace resistive path
    import networkx as nx
    path = nx.shortest_path(graph, 'vrm_node', 'cell_node')

Author: Based on mpower C++ parser implementation
Date: December 9, 2025
"""

import bisect
import os
import sys
import gzip
import re
import pickle
import logging
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Set, TextIO, Union
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict

# Add project root to path for imports when running as script
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from core.rx_graph import RustworkxMultiDiGraphWrapper
from core.rx_algorithms import contract_nodes, node_connected_component

try:
    from tqdm import tqdm
except ImportError:
    print("WARNING: tqdm not found. Install for progress bars: pip install tqdm")

try:
    from pdn_plotter import PDNPlotter
except ImportError:
    print("WARNING: pdn_plotter not found. Plotting features will be disabled.")
    # Fallback tqdm
    class tqdm:
        def __init__(self, iterable=None, desc=None, total=None, **kwargs):
            self.iterable = iterable
            self.n = 0
            self.total = total
        def __iter__(self):
            return iter(self.iterable) if self.iterable else iter([])
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def update(self, n=1):
            self.n += n
        def set_description(self, desc):
            pass


# Constants matching C++ parser
GMAX = 1e5  # Maximum conductance (from parser.cc line 1119)
SHORT_THRESHOLD = 1e-6  # Resistance threshold for shorts (KOhm)
INVALID_STATIC_CURRENT = -999.0

# Unit conversions (matching C++ parser)
R_TO_KOHM = 1e-3  # Ohm to KOhm
C_TO_FF = 1e15   # Farad to fF
L_TO_NH = 1e9    # Henry to nH
I_TO_MA = 1e3    # Ampere to mA


@dataclass
class ParseStats:
    """Statistics for parsed netlist"""
    nodes_before_cleanup: int = 0
    nodes_after_cleanup: int = 0
    elements_total: int = 0
    elements_removed: int = 0
    resistors: int = 0
    capacitors: int = 0
    inductors: int = 0
    vsources: int = 0
    isources: int = 0
    mutual_inductors: int = 0
    shorted_elements: int = 0
    floating_nodes: int = 0
    boundary_nodes: int = 0
    package_nodes: int = 0
    vsrc_nodes: int = 0
    tiles_parsed: int = 0
    tiles_failed: int = 0
    layer_stats: Dict[str, Dict] = field(default_factory=dict)
    net_stats: Dict[str, Dict] = field(default_factory=dict)
    unmapped_nodes: int = 0
    # Instance current statistics
    instances_with_waveforms: int = 0
    total_static_current_ma: float = 0.0


# =============================================================================
# Instance Current Source Data Structures
# =============================================================================
# These classes support parsing and evaluation of time-domain current waveforms
# from instanceModels*.sp files. All current values are stored in mA.

@dataclass
class InstanceInfo:
    """
    Parsed instance name information.
    
    Instance name format:
      i_<instance_name>:<vdd_net>:<vdd_pin>:<vss_net>:<vss_pin>:<tile_x>:<tile_y>[:<extra>]
    
    Example:
      i_U123/cell:VDD_XLV:VDD:0:0:5:3:0
    """
    full_name: str
    instance_name: str
    vdd_net: Optional[str] = None
    vdd_pin: Optional[str] = None
    vss_net: Optional[str] = None
    vss_pin: Optional[str] = None
    tile_x: int = 0
    tile_y: int = 0

    @classmethod
    def parse(cls, name: str, delimiter: str = ':') -> 'InstanceInfo':
        """Parse instance name to extract net and location info."""
        info = cls(full_name=name, instance_name=name)
        
        # Remove 'i_' or 'I_' prefix for parsing
        parse_name = name
        if parse_name.lower().startswith('i_'):
            parse_name = parse_name[2:]
        
        parts = parse_name.split(delimiter)
        
        if len(parts) >= 1:
            info.instance_name = parts[0]
        if len(parts) >= 2:
            info.vdd_net = parts[1] if parts[1] != '0' else None
        if len(parts) >= 3:
            info.vdd_pin = parts[2] if parts[2] != '0' else None
        if len(parts) >= 4:
            info.vss_net = parts[3] if parts[3] != '0' else None
        if len(parts) >= 5:
            info.vss_pin = parts[4] if parts[4] != '0' else None
        if len(parts) >= 6:
            try:
                info.tile_x = int(parts[5])
            except ValueError:
                pass
        if len(parts) >= 7:
            try:
                info.tile_y = int(parts[6])
            except ValueError:
                pass
        
        return info
    
    def to_dict(self) -> Dict:
        """Serialize to dictionary for JSON-compatible storage."""
        return {
            'full_name': self.full_name,
            'instance_name': self.instance_name,
            'vdd_net': self.vdd_net,
            'vdd_pin': self.vdd_pin,
            'vss_net': self.vss_net,
            'vss_pin': self.vss_pin,
            'tile_x': self.tile_x,
            'tile_y': self.tile_y
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'InstanceInfo':
        """Reconstruct from dictionary."""
        return cls(
            full_name=d.get('full_name', ''),
            instance_name=d.get('instance_name', ''),
            vdd_net=d.get('vdd_net'),
            vdd_pin=d.get('vdd_pin'),
            vss_net=d.get('vss_net'),
            vss_pin=d.get('vss_pin'),
            tile_x=d.get('tile_x', 0),
            tile_y=d.get('tile_y', 0)
        )


@dataclass
class Pulse:
    """
    Pulse waveform definition.
    
    All values in base units (Amperes for current). Convert to mA when storing.
    """
    v1: float = 0.0       # Initial value
    v2: float = 0.0       # Pulsed value
    delay: float = 0.0    # Delay time (s)
    rt: float = 0.0       # Rise time (s)
    ft: float = 0.0       # Fall time (s)
    width: float = 0.0    # Pulse width (s)
    period: float = 0.0   # Period (s), 0 = non-periodic

    def evaluate(self, time: float) -> float:
        """Evaluate pulse value at given time."""
        if self.period <= 0:
            t = time
        else:
            t = time % self.period

        t_rel = t - self.delay
        if t_rel < 0:
            if self.period > 0:
                t_rel += self.period
            else:
                return self.v1

        if t_rel < 0:
            return self.v1
        elif t_rel < self.rt:
            if self.rt > 0:
                return self.v1 + (self.v2 - self.v1) * (t_rel / self.rt)
            return self.v2
        elif t_rel < self.rt + self.width:
            return self.v2
        elif t_rel < self.rt + self.width + self.ft:
            if self.ft > 0:
                t_fall = t_rel - self.rt - self.width
                return self.v2 + (self.v1 - self.v2) * (t_fall / self.ft)
            return self.v1
        else:
            return self.v1

    def get_dc(self) -> float:
        """Calculate average DC value of pulse over one period."""
        if self.period <= 0:
            return 0.0
        rise_area = 0.5 * self.rt * (self.v1 + self.v2)
        high_area = self.width * self.v2
        fall_area = 0.5 * self.ft * (self.v1 + self.v2)
        low_time = self.period - self.rt - self.width - self.ft
        low_area = max(0, low_time) * self.v1
        total_area = rise_area + high_area + fall_area + low_area
        return total_area / self.period
    
    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            'v1': self.v1, 'v2': self.v2, 'delay': self.delay,
            'rt': self.rt, 'ft': self.ft, 'width': self.width, 'period': self.period
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'Pulse':
        """Reconstruct from dictionary."""
        return cls(
            v1=d.get('v1', 0.0), v2=d.get('v2', 0.0), delay=d.get('delay', 0.0),
            rt=d.get('rt', 0.0), ft=d.get('ft', 0.0), width=d.get('width', 0.0),
            period=d.get('period', 0.0)
        )


@dataclass
class PWL:
    """
    Piece-wise linear waveform.

    points: List of (time, value) tuples.
    """
    points: List[Tuple[float, float]] = field(default_factory=list)
    period: float = 0.0
    delay: float = 0.0
    _times_cache: Optional[Tuple[float, ...]] = field(default=None, init=False, repr=False)

    def _get_times(self) -> Tuple[float, ...]:
        """Get cached tuple of time values for binary search."""
        if self._times_cache is None:
            self._times_cache = tuple(p[0] for p in self.points)
        return self._times_cache

    def evaluate(self, time: float) -> float:
        """Evaluate PWL value at given time using binary search (O(log N))."""
        if not self.points:
            return 0.0

        t = time - self.delay
        if self.period > 0 and t >= 0:
            t = t % self.period

        if t <= self.points[0][0]:
            return self.points[0][1]

        if t >= self.points[-1][0]:
            if self.period > 0:
                return self.points[0][1]
            return self.points[-1][1]

        # Binary search for the interval containing t
        times = self._get_times()
        i = bisect.bisect_right(times, t) - 1

        t1, v1 = self.points[i]
        t2, v2 = self.points[i + 1]
        if t2 == t1:
            return v1
        return v1 + (v2 - v1) * (t - t1) / (t2 - t1)

    def get_dc(self) -> float:
        """Calculate average DC value of PWL over one period."""
        if not self.points or len(self.points) < 2:
            if self.points:
                return self.points[0][1]
            return 0.0

        if self.period <= 0:
            return 0.0

        total_area = 0.0
        for i in range(len(self.points) - 1):
            t1, v1 = self.points[i]
            t2, v2 = self.points[i + 1]
            total_area += 0.5 * (v1 + v2) * (t2 - t1)

        if self.period > self.points[-1][0]:
            t_last, v_last = self.points[-1]
            t_first, v_first = self.points[0]
            remaining = self.period - t_last + t_first
            total_area += 0.5 * (v_last + v_first) * remaining

        return total_area / self.period
    
    def __reduce__(self):
        """Custom pickle support - only serialize init fields, not _times_cache."""
        return (PWL, (self.points, self.period, self.delay))

    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {'points': self.points, 'period': self.period, 'delay': self.delay}

    @classmethod
    def from_dict(cls, d: Dict) -> 'PWL':
        """Reconstruct from dictionary."""
        return cls(
            points=[(p[0], p[1]) for p in d.get('points', [])],
            period=d.get('period', 0.0),
            delay=d.get('delay', 0.0)
        )


@dataclass
class CurrentSource:
    """
    Current source instance with full waveform data.
    
    All current values are stored in mA for consistency with the PDN parser.
    Use get_static_current() for DC analysis, get_current_at_time(t) for transient.
    """
    name: str
    node1: str
    node2: str
    dc_value: float = 0.0                           # DC component (mA)
    static_value: Optional[float] = None            # Static value override (mA)
    pulses: List[Pulse] = field(default_factory=list)
    pwls: List[PWL] = field(default_factory=list)
    info: Optional[InstanceInfo] = None

    def has_waveform_data(self) -> bool:
        """Check if instance has any dynamic waveform data (Pulse or PWL)."""
        return len(self.pulses) > 0 or len(self.pwls) > 0

    def has_current_data(self) -> bool:
        """Check if instance has any current data."""
        return (self.dc_value != 0.0 or
                self.static_value is not None or
                self.has_waveform_data())

    def get_static_current(self) -> float:
        """Get static/DC current value (mA)."""
        if self.static_value is not None:
            return self.dc_value + self.static_value
        total = self.dc_value
        for pulse in self.pulses:
            total += pulse.get_dc()
        for pwl in self.pwls:
            total += pwl.get_dc()
        return total

    def get_current_at_time(self, time: float) -> float:
        """Get current value at specified time (mA)."""
        total = self.dc_value
        for pulse in self.pulses:
            total += pulse.evaluate(time)
        for pwl in self.pwls:
            total += pwl.evaluate(time)
        return total
    
    def to_dict(self) -> Dict:
        """Serialize to dictionary for JSON-compatible storage."""
        return {
            'name': self.name,
            'node1': self.node1,
            'node2': self.node2,
            'dc_value': self.dc_value,
            'static_value': self.static_value,
            'pulses': [p.to_dict() for p in self.pulses],
            'pwls': [p.to_dict() for p in self.pwls],
            'info': self.info.to_dict() if self.info else None
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'CurrentSource':
        """Reconstruct from dictionary."""
        return cls(
            name=d.get('name', ''),
            node1=d.get('node1', ''),
            node2=d.get('node2', ''),
            dc_value=d.get('dc_value', 0.0),
            static_value=d.get('static_value'),
            pulses=[Pulse.from_dict(p) for p in d.get('pulses', [])],
            pwls=[PWL.from_dict(p) for p in d.get('pwls', [])],
            info=InstanceInfo.from_dict(d['info']) if d.get('info') else None
        )


# =============================================================================
# Instance Current Source Parsing Helpers
# =============================================================================

def _parse_spice_value(value_str: str) -> float:
    """
    Parse a numeric value with optional SPICE unit suffix.
    
    Supports: f (femto), p (pico), n (nano), u (micro), m (milli),
              k (kilo), meg (mega), g (giga), t (tera)
    """
    value_str = value_str.strip().lower()
    multipliers = {
        'f': 1e-15, 'p': 1e-12, 'n': 1e-9, 'u': 1e-6,
        'm': 1e-3, 'k': 1e3, 'meg': 1e6, 'g': 1e9, 't': 1e12
    }

    match = re.match(r'^([+-]?[\d.]+(?:e[+-]?\d+)?)\s*(\w*)$', value_str)
    if match:
        num = float(match.group(1))
        unit = match.group(2)
        if unit in multipliers:
            return num * multipliers[unit]
        elif unit.startswith('meg'):
            return num * 1e6
        return num

    return float(value_str)


def _parse_pulse(pulse_str: str) -> Pulse:
    """Parse pulse definition: pulse(v1, v2, delay, rt, ft, width, period)"""
    match = re.search(r'pulse\s*\(\s*([^)]+)\)', pulse_str, re.IGNORECASE)
    if not match:
        return Pulse()

    values = re.split(r'[,\s]+', match.group(1).strip())
    values = [v for v in values if v]

    pulse = Pulse()
    if len(values) >= 1:
        pulse.v1 = _parse_spice_value(values[0])
    if len(values) >= 2:
        pulse.v2 = _parse_spice_value(values[1])
    if len(values) >= 3:
        pulse.delay = _parse_spice_value(values[2])
    if len(values) >= 4:
        pulse.rt = _parse_spice_value(values[3])
    if len(values) >= 5:
        pulse.ft = _parse_spice_value(values[4])
    if len(values) >= 6:
        pulse.width = _parse_spice_value(values[5])
    if len(values) >= 7:
        pulse.period = _parse_spice_value(values[6])

    return pulse


def _parse_pwl(pwl_str: str) -> PWL:
    """Parse PWL definition: pwl(t1 v1 t2 v2 ...)"""
    match = re.search(r'pwl\s*\(\s*([^)]+)\)', pwl_str, re.IGNORECASE)
    if not match:
        return PWL()

    values = re.split(r'[,\s]+', match.group(1).strip())
    values = [v for v in values if v]

    pwl = PWL()
    for i in range(0, len(values) - 1, 2):
        t = _parse_spice_value(values[i])
        v = _parse_spice_value(values[i + 1])
        pwl.points.append((t, v))

    pwl.points.sort(key=lambda x: x[0])
    return pwl


def _parse_current_source_line(line: str) -> Optional[CurrentSource]:
    """
    Parse a current source line and return a CurrentSource object.
    
    Handles complex formats including:
    - DC values with optional 'dc' prefix
    - static_value= parameter
    - pulse(...) waveforms
    - pwl(...) waveforms with pwl_period= and pwl_delay=
    - sp= (source period) parameter
    
    Note: Values are returned in base SPICE units (Amperes). Caller must convert to mA.
    """
    line = line.strip()
    if not line or not line[0].lower() == 'i':
        return None

    # Tokenize preserving parenthesized expressions
    tokens = []
    current = ""
    paren_depth = 0

    for char in line:
        if char == '(':
            paren_depth += 1
            current += char
        elif char == ')':
            paren_depth -= 1
            current += char
        elif char in ' \t' and paren_depth == 0:
            if current:
                tokens.append(current)
                current = ""
        else:
            current += char

    if current:
        tokens.append(current)

    if len(tokens) < 4:
        return None

    isrc = CurrentSource(
        name=tokens[0],
        node1=tokens[1],
        node2=tokens[2]
    )
    
    # Parse instance name to extract net information
    isrc.info = InstanceInfo.parse(tokens[0])

    # Parse DC value (might be prefixed with 'dc')
    idx = 3
    if tokens[idx].lower() == 'dc' or tokens[idx].lower().startswith('dc.'):
        idx += 1
        if idx < len(tokens):
            try:
                isrc.dc_value = _parse_spice_value(tokens[idx])
                idx += 1
            except ValueError:
                pass
    else:
        try:
            isrc.dc_value = _parse_spice_value(tokens[idx])
            idx += 1
        except ValueError:
            pass

    # Parse remaining parameters
    i = idx
    while i < len(tokens):
        token = tokens[i]
        token_lower = token.lower()

        if token_lower.startswith('pulse'):
            isrc.pulses.append(_parse_pulse(token))
        elif token_lower.startswith('pwl') and not token_lower.startswith('pwl_'):
            pwl = _parse_pwl(token)
            isrc.pwls.append(pwl)
        elif token_lower.startswith('pwl_period='):
            period = _parse_spice_value(token.split('=')[1])
            for pwl in isrc.pwls:
                if pwl.period == 0:
                    pwl.period = period
        elif token_lower.startswith('pwl_delay='):
            delay = _parse_spice_value(token.split('=')[1])
            for pwl in isrc.pwls:
                if pwl.delay == 0:
                    pwl.delay = delay
        elif token_lower.startswith('static_value='):
            isrc.static_value = _parse_spice_value(token.split('=')[1])
        elif token_lower.startswith('sp='):
            # Source period - apply to pulses
            period = _parse_spice_value(token.split('=')[1])
            for pulse in isrc.pulses:
                if pulse.period == 0:
                    pulse.period = period

        i += 1

    return isrc


class SpiceLineReader:
    """
    Handles SPICE file reading with automatic gzip detection, line continuation,
    and comment handling.
    """
    
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.file_handle: Optional[TextIO] = None
        self.line_number = 0
        self.is_gzipped = False
        self._pending_line: Optional[str] = None  # Buffer for line read but not yet processed
        
    def __enter__(self):
        self.open()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        
    def open(self):
        """Open file with automatic gzip detection using magic number"""
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"File not found: {self.filepath}")
            
        # Check for gzip magic number (0x1f8b)
        with open(self.filepath, 'rb') as f:
            magic = f.read(2)
            self.is_gzipped = (magic == b'\x1f\x8b')
        
        # Open with appropriate handler
        if self.is_gzipped:
            self.file_handle = gzip.open(self.filepath, 'rt', encoding='utf-8', errors='ignore')
        else:
            self.file_handle = open(self.filepath, 'r', encoding='utf-8', errors='ignore')
            
        logging.debug(f"Opened {'gzipped' if self.is_gzipped else 'plain'} file: {self.filepath}")
        
    def close(self):
        """Close file handle"""
        if self.file_handle:
            self.file_handle.close()
            self.file_handle = None
            
    def read_line(self) -> Optional[str]:
        """
        Read a logical line from SPICE file, handling:
        - Line continuation (lines starting with '+')
        - Comment removal (lines starting with '*')
        - Multiple whitespace normalization
        
        Returns None at EOF
        """
        if not self.file_handle:
            return None
            
        lines = []
        
        while True:
            # Check if we have a pending line from previous call
            if self._pending_line is not None:
                line = self._pending_line
                self._pending_line = None
            else:
                raw_line = self.file_handle.readline()
                if not raw_line:  # EOF
                    break
                    
                self.line_number += 1
                line = raw_line.strip()
            
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
                    logging.warning(f"Line {self.line_number}: Continuation '+' without previous line")
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
        
        # Remove inline comments (but be careful - * at start of a token in middle of line might be boundary marker)
        # Only remove comment if * is preceded by whitespace
        parts = logical_line.split()
        cleaned_parts = []
        for i, part in enumerate(parts):
            if part.startswith('*') and i > 0:
                # This might be a boundary marker like *node_name, not a comment
                # Check if it looks like a node name (contains alphanumeric/underscore after *)
                if len(part) > 1 and (part[1].isalnum() or part[1] == '_'):
                    cleaned_parts.append(part)  # It's a boundary marker
                else:
                    # It's a comment, stop here
                    break
            elif part == '*':
                # Standalone *, treat as start of comment
                break
            else:
                cleaned_parts.append(part)
        
        logical_line = ' '.join(cleaned_parts)
        
        return logical_line if logical_line else self.read_line()  # Recurse if empty after cleanup


class GraphBuilder:
    """
    Builds and manages the rustworkx MultiDiGraph representation of the PDN.
    """

    def __init__(self, validate: bool = False, strict: bool = False, net_filter: Optional[str] = None,
                 store_instance_sources: bool = False):
        self.graph = RustworkxMultiDiGraphWrapper()
        self.validate = validate
        self.strict = strict
        self.net_filter = net_filter.lower() if net_filter else None  # Store lowercase for case-insensitive comparison
        self.store_instance_sources = store_instance_sources
        # Union-Find structure for package/main netlist connectivity
        self.uf_parent: Dict[str, str] = {}  # Union-Find parent pointers for package nodes
        self.uf_net: Dict[str, str] = {}  # Net type for each union-find root
        self.package_edges: List[Tuple[str, str]] = []  # Deferred package edges for union-find
        self.stats = ParseStats()
        
        # Metadata dictionaries
        self.vsrc_dict: Dict[str, Dict] = {}
        self.parameters: Dict[str, str] = {}
        self.instance_node_map: Dict[str, List[str]] = {}  # Backward compat: name -> [node+, node-]
        self.instance_sources: Dict[str, CurrentSource] = {}  # Full current source data
        self.merged_nodes: List[Tuple[str, str, int]] = []
        self.mutual_inductors: Dict[str, Tuple[str, str, float]] = {}
        self.node_net_map: Dict[str, str] = {}  # Die node name -> net name from .nd files
        self.node_net_map_lower: Dict[str, str] = {}  # Die node name -> lowercase net name for filtering
        
        # Parsing context
        self.current_tile_id: Optional[Tuple[int, int]] = None
        self.current_file_type: str = 'die'  # 'die', 'package', or 'instance'
        self.tile_grid: Optional[Tuple[int, int]] = None  # (N, M) from .partition_info
        
        # Node tracking
        self.boundary_nodes: Set[str] = set()
        self.node_attributes: Dict[str, Dict] = defaultdict(dict)

        # Edge index tracking for efficient filtering (avoid full graph iteration)
        self.package_edge_indices: List[int] = []  # Edge indices from package.ckt
        self.vsrc_edge_indices: List[int] = []     # Voltage source edge indices

        self.logger = logging.getLogger(__name__)
        
    def _uf_find(self, node: str) -> str:
        """Union-Find: find root with path compression"""
        if node not in self.uf_parent:
            self.uf_parent[node] = node  # Initialize
            return node
        
        # Path compression
        if self.uf_parent[node] != node:
            self.uf_parent[node] = self._uf_find(self.uf_parent[node])
        return self.uf_parent[node]
    
    def _uf_union(self, node1: str, node2: str) -> None:
        """Union-Find: union two nodes, propagating net type from die nodes"""
        root1 = self._uf_find(node1)
        root2 = self._uf_find(node2)
        
        if root1 == root2:
            return  # Already in same set
        
        # Get net types (from .nd file for die nodes, or from existing union)
        # Use original case net names for union-find
        net1 = self.node_net_map.get(node1)  # Die node from .nd file
        net2 = self.node_net_map.get(node2)  # Die node from .nd file
        root1_net = net1 or self.uf_net.get(root1)  # Existing net from root
        root2_net = net2 or self.uf_net.get(root2)  # Existing net from root
        
        # Union: prefer root with net type from die node
        if root1_net:
            self.uf_parent[root2] = root1
            self.uf_net[root1] = root1_net
        elif root2_net:
            self.uf_parent[root1] = root2
            self.uf_net[root2] = root2_net
        else:
            # Neither has net type yet, arbitrary union
            self.uf_parent[root2] = root1
    
    def _get_node_net(self, node: str) -> Optional[str]:
        """Get effective net type for a node (from .nd file or union-find)"""
        # First check if it's a die node with explicit mapping
        net = self.node_net_map.get(node)
        if net:
            return net
        
        # Check union-find for package/unmapped nodes
        root = self._uf_find(node)
        return self.uf_net.get(root)
    
    def add_node(self, name: str, **attrs):
        """Add node with attributes, merging with existing if present"""
        if name not in self.graph:
            # Node '0' is special: never package, always unmapped, excluded from statistics
            if name == '0':
                is_package_node = False
            else:
                # Determine if this is a package node (in package file but not in die node map)
                is_package_node = (self.current_file_type == 'package' and 
                                 name not in self.node_net_map)
            
            # Set default attributes
            node_attrs = {
                'name': name,
                'x': None,
                'y': None,
                'layer': None,
                'is_boundary': name in self.boundary_nodes,
                'is_package': is_package_node,
                'is_vsrc_node': False,
                'net_type': None,
                'voltage': None,
                'tile_id': self.current_tile_id
            }
            node_attrs.update(attrs)
            self.graph.add_node(name, **node_attrs)
            
            # Track unmapped nodes (not in .nd file, not package)
            # Node '0' is global ground and always counts as unmapped (but excluded from statistics)
            if name == '0':
                pass  # Node '0' is unmapped but not counted in statistics
            elif (name not in self.node_net_map and 
                  not is_package_node and 
                  self.current_file_type != 'package'):
                self.stats.unmapped_nodes += 1
            
            # Extract coordinates from node name patterns
            self._extract_coordinates(name)
        else:
            # Update existing node attributes
            self.graph.nodes_dict[name].update(attrs)
            
    def _extract_coordinates(self, node_name: str):
        """
        Extract x, y, layer coordinates from node names.
        Patterns: X_Y_LAYER (3D) or X_Y (2D)
        Layer is stored as string to support both numeric and named layers (e.g., 'M1', 'AP')
        """
        # Try 3D pattern first: X_Y_LAYER
        match = re.search(r'(\d+)_(\d+)_(.+?)(?:_|$)', node_name)
        if match:
            x, y, layer = int(match.group(1)), int(match.group(2)), match.group(3)
            coords = {'x': x, 'y': y, 'layer': layer}
            # Also update graph if node exists
            if node_name in self.graph:
                self.graph.nodes_dict[node_name].update(coords)
            return coords
        
        # Fallback to 2D pattern: X_Y
        match = re.search(r'(\d+)_(\d+)', node_name)
        if match:
            x, y = int(match.group(1)), int(match.group(2))
            coords = {'x': x, 'y': y}
            # Also update graph if node exists
            if node_name in self.graph:
                self.graph.nodes_dict[node_name].update(coords)
            return coords
        
        return {}
    
    def _infer_net_type(self, node_name: str) -> Optional[str]:
        """
        Infer power net type from node name.
        For die nodes: use exact mapping from .nd files
        For package nodes or unmapped nodes: return None (will be categorized separately)
        Node '0' is global ground, not a net type.
        """
        # Only use exact mapping from .nd file (die nodes)
        if node_name in self.node_net_map:
            return self.node_net_map[node_name]
        
        # No pattern matching - if not in .nd file, it's unmapped or package
        # Node '0' is global ground, not mapped to any net
        return None
            
    def add_element(self, elem_type: str, node1: str, node2: str, 
                   value: float, name: str, **attrs) -> bool:
        """Add circuit element as edge between two nodes.
        
        Returns:
            True if element was added, False if filtered out by net_filter.
        """
        # For package/main netlist elements (not in tiles), defer union-find processing
        # Tile elements have die nodes with explicit net types from .nd files
        if self.current_file_type in ['package', 'die'] and self.current_tile_id is None:
            # Defer package connectivity (will process after all parsing)
            # Skip node '0' to prevent cross-net contamination
            if node1 != '0' and node2 != '0':
                self.package_edges.append((node1, node2))
        
        # Get net type from .nd file (die nodes) or union-find (package nodes)
        node1_net = self._get_node_net(node1)
        node2_net = self._get_node_net(node2)
        net_type = node1_net or node2_net
        
        # Apply net filter if active (case-insensitive)
        # Exception: Don't filter voltage sources or package elements during parsing
        # These will be filtered post-processing based on connectivity to filtered net
        if self.net_filter is not None and elem_type != 'V' and self.current_file_type != 'package':
            # Get lowercase net names for comparison
            node1_net_lower = self.node_net_map_lower.get(node1) or (node1_net.lower() if node1_net else None)
            node2_net_lower = self.node_net_map_lower.get(node2) or (node2_net.lower() if node2_net else None)
            # Include element if either node belongs to filtered net
            if node1_net_lower != self.net_filter and node2_net_lower != self.net_filter:
                return False  # Skip this element
        
        # Ensure nodes exist
        self.add_node(node1)
        self.add_node(node2)
        
        # Create edge attributes
        edge_attrs = {
            'type': elem_type,
            'value': value,
            'elem_name': name,
            'tile_id': self.current_tile_id,
            'net_type': net_type
        }
        edge_attrs.update(attrs)
        
        # Add edge (MultiDiGraph allows multiple edges between same nodes)
        edge_idx = self.graph.add_edge(node1, node2, **edge_attrs)

        # Track edge indices for efficient post-processing filtering
        if elem_type == 'V':
            self.vsrc_edge_indices.append(edge_idx)
        if self.current_file_type == 'package':
            self.package_edge_indices.append(edge_idx)

        # Update global statistics
        self.stats.elements_total += 1
        if elem_type == 'R':
            self.stats.resistors += 1
        elif elem_type == 'C':
            self.stats.capacitors += 1
        elif elem_type == 'L':
            self.stats.inductors += 1
        elif elem_type == 'V':
            self.stats.vsources += 1
        elif elem_type == 'I':
            self.stats.isources += 1
        
        # Update per-net statistics (separate die, package, and unmapped)
        if net_type:
            # Determine category: package nodes are in package file and not in die node map
            node1_is_package = (node1 not in self.node_net_map and 
                              self.current_file_type == 'package')
            node2_is_package = (node2 not in self.node_net_map and 
                              self.current_file_type == 'package')
            
            # Element is package if either node is a package node
            is_package_elem = node1_is_package or node2_is_package
            category = 'package' if is_package_elem else 'die'
            
            if net_type not in self.stats.net_stats:
                self.stats.net_stats[net_type] = {
                    'die': {
                        'nodes': set(),
                        'resistors': 0,
                        'capacitors': 0,
                        'inductors': 0,
                        'vsources': 0,
                        'isources': 0,
                        'isources_with_waveforms': 0,
                        'total_resistance': 0.0,
                        'total_capacitance': 0.0,
                        'total_inductance': 0.0,
                        'total_current': 0.0
                    },
                    'package': {
                        'nodes': set(),
                        'resistors': 0,
                        'capacitors': 0,
                        'inductors': 0,
                        'vsources': 0,
                        'isources': 0,
                        'isources_with_waveforms': 0,
                        'total_resistance': 0.0,
                        'total_capacitance': 0.0,
                        'total_inductance': 0.0,
                        'total_current': 0.0
                    },
                    'unmapped': {
                        'nodes': set(),
                        'resistors': 0,
                        'capacitors': 0,
                        'inductors': 0,
                        'vsources': 0,
                        'isources': 0,
                        'isources_with_waveforms': 0,
                        'total_resistance': 0.0,
                        'total_capacitance': 0.0,
                        'total_inductance': 0.0,
                        'total_current': 0.0
                    }
                }
            
            net_stat = self.stats.net_stats[net_type][category]
            # Exclude node '0' from statistics
            if node1 != '0':
                net_stat['nodes'].add(node1)
            if node2 != '0':
                net_stat['nodes'].add(node2)
            
            if elem_type == 'R':
                net_stat['resistors'] += 1
                net_stat['total_resistance'] += value
            elif elem_type == 'C':
                net_stat['capacitors'] += 1
                net_stat['total_capacitance'] += value
            elif elem_type == 'L':
                net_stat['inductors'] += 1
                net_stat['total_inductance'] += value
            elif elem_type == 'V':
                net_stat['vsources'] += 1
            elif elem_type == 'I':
                net_stat['isources'] += 1
                net_stat['total_current'] += abs(value)
        else:
            # No net_type means unmapped element
            if 'unmapped' not in self.stats.net_stats:
                self.stats.net_stats['unmapped'] = {
                    'die': {
                        'nodes': set(),
                        'resistors': 0,
                        'capacitors': 0,
                        'inductors': 0,
                        'vsources': 0,
                        'isources': 0,
                        'isources_with_waveforms': 0,
                        'total_resistance': 0.0,
                        'total_capacitance': 0.0,
                        'total_inductance': 0.0,
                        'total_current': 0.0
                    },
                    'package': {
                        'nodes': set(),
                        'resistors': 0,
                        'capacitors': 0,
                        'inductors': 0,
                        'vsources': 0,
                        'isources': 0,
                        'isources_with_waveforms': 0,
                        'total_resistance': 0.0,
                        'total_capacitance': 0.0,
                        'total_inductance': 0.0,
                        'total_current': 0.0
                    },
                    'unmapped': {
                        'nodes': set(),
                        'resistors': 0,
                        'capacitors': 0,
                        'inductors': 0,
                        'vsources': 0,
                        'isources': 0,
                        'isources_with_waveforms': 0,
                        'total_resistance': 0.0,
                        'total_capacitance': 0.0,
                        'total_inductance': 0.0,
                        'total_current': 0.0
                    }
                }
            
            # Determine if unmapped element is in package or die
            is_package_elem = self.current_file_type == 'package'
            category = 'package' if is_package_elem else 'unmapped'
            
            net_stat = self.stats.net_stats['unmapped'][category]
            # Exclude node '0' from statistics
            if node1 != '0':
                net_stat['nodes'].add(node1)
            if node2 != '0':
                net_stat['nodes'].add(node2)
            
            if elem_type == 'R':
                net_stat['resistors'] += 1
                net_stat['total_resistance'] += value
            elif elem_type == 'C':
                net_stat['capacitors'] += 1
                net_stat['total_capacitance'] += value
            elif elem_type == 'L':
                net_stat['inductors'] += 1
                net_stat['total_inductance'] += value
            elif elem_type == 'V':
                net_stat['vsources'] += 1
            elif elem_type == 'I':
                net_stat['isources'] += 1
                net_stat['total_current'] += abs(value)
        
        return True
            
    def add_grounded_element(self, elem_type: str, node: str, value: float, 
                            name: str, **attrs) -> bool:
        """Add element connected to ground (node '0').
        
        Returns:
            True if element was added, False if filtered out by net_filter.
        """
        return self.add_element(elem_type, node, '0', value, name, **attrs)
        
    def mark_boundary_node(self, name: str):
        """Mark node as boundary node (needs stitching)"""
        self.boundary_nodes.add(name)
        if name in self.graph:
            self.graph.nodes_dict[name]['is_boundary'] = True
        self.stats.boundary_nodes += 1
        
    def stitch_nodes(self, name1: str, name2: str):
        """
        Stitch two nodes together (merge them).
        This is used for boundary nodes across tiles.
        """
        if name1 not in self.graph or name2 not in self.graph:
            self.logger.warning(f"Cannot stitch nodes {name1} and {name2}: one or both not found")
            return

        # Merge name2 into name1
        try:
            contract_nodes(self.graph, name1, name2, self_loops=False)
            self.merged_nodes.append((name2, name1, 0))  # 0 = stitch merge type
            self.logger.debug(f"Stitched nodes: {name2} -> {name1}")
        except Exception as e:
            self.logger.error(f"Error stitching nodes {name1} and {name2}: {e}")
            
    def validate_node_uniqueness(self):
        """Check for node name collisions and report detailed errors"""
        # This is mostly already handled by the wrapper, but we can add custom checks
        node_sources = defaultdict(list)

        for node in self.graph.nodes():
            tile_id = self.graph.nodes_dict[node].get('tile_id')
            is_package = self.graph.nodes_dict[node].get('is_package')
            source = f"tile_{tile_id}" if tile_id else ('package' if is_package else 'main')
            node_sources[node].append(source)
        
        # Check for actual duplicates (shouldn't happen with NetworkX, but check our tracking)
        duplicates = {node: sources for node, sources in node_sources.items() if len(sources) > 1}
        
        if duplicates:
            error_msg = "Node name collisions detected:\n"
            for node, sources in duplicates.items():
                error_msg += f"  Node '{node}' appears in: {', '.join(sources)}\n"
            
            if self.strict:
                raise ValueError(error_msg)
            else:
                self.logger.warning(error_msg)
                
    def finalize(self):
        """Finalize graph and add metadata"""
        self.stats.nodes_after_cleanup = self.graph.number_of_nodes()
        
        # Convert net_stats node sets to counts (separate die, package, and unmapped)
        net_stats_serializable = {}
        for net, categories in self.stats.net_stats.items():
            net_stats_serializable[net] = {}
            for category in ['die', 'package', 'unmapped']:
                if category in categories:
                    stats = categories[category]
                    net_stats_serializable[net][category] = {
                        'nodes': len(stats['nodes']),
                        'resistors': stats['resistors'],
                        'capacitors': stats['capacitors'],
                        'inductors': stats['inductors'],
                        'vsources': stats['vsources'],
                        'isources': stats['isources'],
                        'total_resistance_kohm': stats['total_resistance'],
                        'total_capacitance_ff': stats['total_capacitance'],
                        'total_inductance_nh': stats['total_inductance'],
                        'total_current_ma': stats['total_current']
                    }
        
        # Build net_connectivity from union-find results and die node mappings
        net_connectivity = defaultdict(list)
        
        # Add die nodes from .nd files
        for node, net in self.node_net_map.items():
            if node in self.graph:
                net_connectivity[net].append(node)
        
        # Add package nodes from union-find
        for node in self.graph.nodes():
            if node not in self.node_net_map and node != '0':
                net = self._get_node_net(node)
                if net:
                    net_connectivity[net].append(node)
        
        # Add metadata to graph (vsrc_nodes and layer_stats already added by compute methods)
        self.graph.graph['vsrc_dict'] = self.vsrc_dict
        self.graph.graph['parameters'] = self.parameters
        self.graph.graph['tile_grid'] = self.tile_grid
        self.graph.graph['instance_node_map'] = self.instance_node_map

        # Store instance_sources - either serialized (for pickle compatibility) or raw objects (for memory efficiency)
        if self.store_instance_sources:
            # Serialize for storage (backward compat, pickle-safe)
            instance_sources_serialized = {
                name: src.to_dict() for name, src in self.instance_sources.items()
            }
            self.graph.graph['instance_sources'] = instance_sources_serialized
        else:
            # Store raw CurrentSource objects directly (memory efficient, not pickle-safe)
            # Solvers can access via '_instance_sources_objects' key
            self.graph.graph['_instance_sources_objects'] = self.instance_sources
        self.graph.graph['merged_nodes'] = self.merged_nodes
        self.graph.graph['mutual_inductors'] = self.mutual_inductors
        self.graph.graph['net_connectivity'] = dict(net_connectivity)
        self.graph.graph['stats'] = {
            'nodes': self.stats.nodes_after_cleanup,
            'edges': self.graph.number_of_edges(),
            'resistors': self.stats.resistors,
            'capacitors': self.stats.capacitors,
            'inductors': self.stats.inductors,
            'vsources': self.stats.vsources,
            'isources': self.stats.isources,
            'mutual_inductors': self.stats.mutual_inductors,
            'boundary_nodes': self.stats.boundary_nodes,
            'package_nodes': self.stats.package_nodes,
            'vsrc_nodes': self.stats.vsrc_nodes,
            'tiles_parsed': self.stats.tiles_parsed,
            'tiles_failed': self.stats.tiles_failed,
            'unmapped_nodes': self.stats.unmapped_nodes,
            'instances_with_waveforms': self.stats.instances_with_waveforms,
            'total_static_current_ma': self.stats.total_static_current_ma
        }
        self.graph.graph['net_stats'] = net_stats_serializable
        
        self.logger.info(f"Graph finalized: {self.stats.nodes_after_cleanup} nodes, "
                        f"{self.graph.number_of_edges()} edges")


class NetlistParser:
    """
    Main parser for PDN netlists. Orchestrates the parsing process including
    tile-based parsing, subcircuit expansion, and validation.
    """
    
    def __init__(self, netlist_dir: str, validate: bool = False, strict: bool = False,
                 net_filter: Optional[str] = None, verbose: bool = False,
                 vsrc_resistor_pattern: str = 'rs', vsrc_depth_limit: int = 3,
                 store_instance_sources: bool = False):
        """
        Initialize PDN netlist parser.

        Args:
            netlist_dir: Path to directory containing netlist files
            validate: Enable validation checks during parsing
            strict: Raise errors on validation failures (vs warnings)
            net_filter: Only parse elements for this net (e.g., 'VDD')
            verbose: Enable debug logging
            vsrc_resistor_pattern: Pattern for identifying voltage source resistors
            vsrc_depth_limit: Max depth for voltage source node traversal
            store_instance_sources: If True, serialize instance_sources to graph metadata
                                   (needed for pickling). If False (default), store raw
                                   CurrentSource objects for memory efficiency (~60% savings
                                   for large netlists with 1M+ sources).
        """
        self.netlist_dir = Path(netlist_dir)
        self.validate = validate
        self.strict = strict
        self.net_filter = net_filter
        self.vsrc_resistor_pattern = vsrc_resistor_pattern
        self.vsrc_depth_limit = vsrc_depth_limit
        self.store_instance_sources = store_instance_sources

        # Setup logging
        log_level = logging.DEBUG if verbose else logging.INFO
        logging.basicConfig(level=log_level,
                          format='%(levelname)s: %(message)s')
        self.logger = logging.getLogger(__name__)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = self.netlist_dir / f'pdn_parser_{timestamp}.log'
        if not any(isinstance(h, logging.FileHandler) for h in self.logger.handlers):
            file_handler = logging.FileHandler(log_file, mode='a')
            file_handler.setLevel(log_level)
            file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
            self.logger.addHandler(file_handler)
            self.logger.info(f"Logging to file: {log_file}")

        # Initialize graph builder
        self.builder = GraphBuilder(validate=validate, strict=strict, net_filter=net_filter,
                                    store_instance_sources=store_instance_sources)
        
        # Parsing state
        self.subcircuits: Dict[str, Dict] = {}  # name -> {pins: [...], body: [...]}
        self.tile_queue: List[Tuple[int, int, str]] = []  # (x, y, filepath)
        self.instance_queue: List[Tuple[int, int, str]] = []  # (x, y, filepath)
        self.include_stack: List[str] = []  # Track nested includes
        
        # Check for main netlist file
        self.main_netlist = self.netlist_dir / 'ckt.sp'
        if not self.main_netlist.exists():
            raise FileNotFoundError(f"Main netlist file not found: {self.main_netlist}")
            
    def parse(self) -> RustworkxMultiDiGraphWrapper:
        """
        Main parsing entry point. Returns populated rustworkx graph wrapper.
        """
        self.logger.info(f"Parsing PDN netlist from: {self.netlist_dir}")
        timings: Dict[str, float] = {}
        parse_start = time.perf_counter()
        
        try:
            # Parse main netlist file
            t0 = time.perf_counter()
            self._parse_file(str(self.main_netlist), is_main=True)
            timings["parse_main"] = time.perf_counter() - t0
            
            # Parse tiles if present
            if self.tile_queue:
                t0 = time.perf_counter()
                self._parse_tiles()
                timings["parse_tiles"] = time.perf_counter() - t0
                
            # Parse instance models
            if self.instance_queue:
                t0 = time.perf_counter()
                self._parse_instance_models()
                timings["parse_instance_models"] = time.perf_counter() - t0
            
            # Propagate net connectivity through package elements
            # This handles cases where package elements were parsed before die elements
            t0 = time.perf_counter()
            self._propagate_net_connectivity()
            timings["propagate_net_connectivity"] = time.perf_counter() - t0
            
            # Update per-net statistics for package elements now that net types are known
            t0 = time.perf_counter()
            self._update_package_statistics()
            timings["update_package_statistics"] = time.perf_counter() - t0
            
            # Filter voltage sources based on connectivity to filtered net
            t0 = time.perf_counter()
            self._filter_voltage_sources_by_net()
            timings["filter_voltage_sources_by_net"] = time.perf_counter() - t0
            
            # Identify voltage source nodes (must be before validation for floating node check)
            t0 = time.perf_counter()
            self._identify_vsrc_nodes()
            timings["identify_vsrc_nodes"] = time.perf_counter() - t0
            
            # Compute layer statistics
            t0 = time.perf_counter()
            self._compute_layer_stats()
            timings["compute_layer_stats"] = time.perf_counter() - t0
                
            # Perform validation if requested (after vsrc node identification)
            if self.validate:
                t0 = time.perf_counter()
                self._perform_validation()
                timings["perform_validation"] = time.perf_counter() - t0
                
            # Validate node uniqueness
            t0 = time.perf_counter()
            self.builder.validate_node_uniqueness()
            timings["validate_node_uniqueness"] = time.perf_counter() - t0
            
            # Finalize graph
            t0 = time.perf_counter()
            self.builder.finalize()
            timings["finalize"] = time.perf_counter() - t0
            
            # Print statistics
            t0 = time.perf_counter()
            self._print_statistics()
            timings["print_statistics"] = time.perf_counter() - t0

            timings["total_parse"] = time.perf_counter() - parse_start

            self.logger.info("Parse timing breakdown (s):")
            for key in sorted(timings.keys()):
                self.logger.info(f"  {key}: {timings[key]:.4f}")
            
            return self.builder.graph
            
        except Exception as e:
            self.logger.error(f"Parsing failed: {e}")
            if self.strict:
                raise
            return self.builder.graph
            
    def _parse_file(self, filepath: str, is_main: bool = False):
        """Parse a single SPICE file"""
        self.logger.debug(f"Parsing file: {filepath}")
        self.include_stack.append(filepath)
        
        try:
            with SpiceLineReader(filepath) as reader:
                while True:
                    line = reader.read_line()
                    if line is None:
                        break
                        
                    # Process line
                    try:
                        self._process_line(line, filepath)
                    except Exception as e:
                        msg = f"Error parsing line {reader.line_number} in {filepath}: {e}\n  Line: {line}"
                        if self.strict:
                            raise RuntimeError(msg) from e
                        else:
                            self.logger.warning(msg)
                            
        finally:
            self.include_stack.pop()
            
    def _process_line(self, line: str, source_file: str):
        """Process a single logical line from netlist"""
        if not line:
            return
            
        # Get first token (case-insensitive)
        tokens = line.split()
        if not tokens:
            return
            
        first_token = tokens[0]
        first_char = first_token[0].upper()
        
        # Dot commands
        if first_char == '.':
            self._process_dot_command(line, source_file)
        # Circuit elements - check for lowercase single char (simplified format) or standard format
        elif first_token == 'r':  # Exactly 'r' for simplified format
            self._parse_resistor(line)
        elif first_char == 'R' or (first_token.startswith('r') and len(first_token) > 1):
            # R<name> or r<name> format
            self._parse_resistor(line)
        elif first_token == 'c':  # Exactly 'c' for simplified format
            self._parse_capacitor(line)
        elif first_char == 'C' or (first_token.startswith('c') and len(first_token) > 1):
            # C<name> or c<name> format
            self._parse_capacitor(line)
        elif first_token == 'l':  # Exactly 'l' for simplified format
            self._parse_inductor(line)
        elif first_char == 'L' or (first_token.startswith('l') and len(first_token) > 1):
            # L<name> or l<name> format
            self._parse_inductor(line)
        elif first_char == 'K':
            self._parse_mutual_inductor(line)
        elif first_char == 'V':
            self._parse_vsource(line)
        elif first_char == 'I':
            self._parse_isource(line)
        elif first_char in ['E', 'F', 'G', 'H']:
            self._parse_controlled_source(line, first_char)
        elif first_char == 'X':
            self._parse_subcircuit_instance(line)
        elif first_char == 'M':
            self.logger.debug(f"Skipping transistor: {tokens[0]}")
        elif first_char == 'P':
            self.logger.debug(f"Skipping power gate: {tokens[0]}")
        else:
            self.logger.debug(f"Unknown element type: {first_char} in line: {line[:50]}")
            
    def _process_dot_command(self, line: str, source_file: str):
        """Process dot commands (.include, .subckt, .parameter, etc.)"""
        tokens = line.split()
        cmd = tokens[0].lower()
        
        if cmd == '.partition_info' and len(tokens) >= 3:
            # .partition_info N M
            n, m = int(tokens[1]), int(tokens[2])
            self.builder.tile_grid = (n, m)
            self.logger.info(f"Detected tile grid: {n} x {m}")
            
        elif cmd == '.include' and len(tokens) >= 2:
            include_file = tokens[1]
            self._process_include(include_file, source_file)
            
        elif cmd == '.parameter' or cmd == '.param':
            # .parameter name=value or .parameter name value
            if len(tokens) == 3 and '=' not in tokens[1]:
                # Space-separated format: .parameter VDD 0.75
                name, value = tokens[1], tokens[2]
                self.builder.parameters[name.strip().upper()] = value.strip()
            else:
                # Equals-separated format: .parameter VDD=0.75
                for token in tokens[1:]:
                    if '=' in token:
                        name, value = token.split('=', 1)
                        self.builder.parameters[name.strip().upper()] = value.strip()
                    
        elif cmd == '.subckt':
            self._parse_subcircuit_definition(line)
            
        elif cmd == '.ends':
            pass  # Handled in _parse_subcircuit_definition
            
        elif cmd == '.flag_boundary':
            # Mark following nodes as boundary nodes (handled during parsing)
            pass
            
        elif cmd in ['.print', '.tran', '.ac', '.dc', '.die_area', '.model']:
            # Analysis commands - store but don't process
            self.logger.debug(f"Skipping command: {cmd}")
            
    def _process_include(self, include_path: str, source_file: str):
        """Process .include directive"""
        # Resolve path relative to source file directory
        source_dir = Path(source_file).parent
        full_path = source_dir / include_path
        
        if not full_path.exists():
            # Try relative to netlist_dir
            full_path = self.netlist_dir / include_path
            
        if not full_path.exists():
            msg = f"Include file not found: {include_path}"
            if self.strict:
                raise FileNotFoundError(msg)
            else:
                self.logger.warning(msg)
                return
                
        # Check for tile or instance model files
        filename = full_path.name
        
        # Pattern: tile_X_Y.ckt or tile_X_Y.sp
        tile_match = re.match(r'tile_(\d+)_(\d+)\.(ckt|sp)', filename)
        if tile_match:
            x, y = int(tile_match.group(1)), int(tile_match.group(2))
            self.tile_queue.append((x, y, str(full_path)))
            self.logger.debug(f"Queued tile {x}_{y}: {full_path}")
            return
            
        # Pattern: instanceModels_X_Y.sp
        inst_match = re.match(r'instanceModels_(\d+)_(\d+)\.sp', filename)
        if inst_match:
            x, y = int(inst_match.group(1)), int(inst_match.group(2))
            self.instance_queue.append((x, y, str(full_path)))
            self.logger.debug(f"Queued instance models {x}_{y}: {full_path}")
            return
            
        # Check for package files
        if 'package' in filename.lower():
            old_file_type = self.builder.current_file_type
            self.builder.current_file_type = 'package'
            self.builder.stats.package_nodes += 1
            self._parse_file(str(full_path))
            self.builder.current_file_type = old_file_type
        else:
            # Regular include
            self._parse_file(str(full_path))
            
    def _load_node_net_map(self, nd_filepath: str) -> Dict[str, str]:
        """
        Load node-to-net mapping from .nd file.
        Format: <node_name> <val1> <val2> <val3> <val4> <net_name>
        """
        node_map = {}
        
        try:
            with SpiceLineReader(nd_filepath) as reader:
                while True:
                    line = reader.read_line()
                    if line is None:
                        break
                    
                    tokens = line.split()
                    if len(tokens) >= 6:
                        node_name = tokens[0]
                        net_name = tokens[5]
                        node_map[node_name] = net_name
                        # Also store lowercase version for case-insensitive filtering
                        self.builder.node_net_map_lower[node_name] = net_name.lower()
                    elif len(tokens) > 0:
                        self.logger.warning(f"Invalid .nd line (expected 6 tokens, got {len(tokens)}): {line}")
        except FileNotFoundError:
            raise FileNotFoundError(f"Required .nd file not found: {nd_filepath}")
        except Exception as e:
            raise RuntimeError(f"Error reading .nd file {nd_filepath}: {e}")
        
        return node_map
    
    def _propagate_net_connectivity(self):
        """
        Process deferred package edges using union-find to efficiently trace connectivity.
        Die nodes from .nd files serve as anchors with explicit net types.
        This also assigns net types to all package nodes based on their connectivity to die nodes.
        """
        if not self.builder.package_edges:
            return
        
        self.logger.info(f"Processing {len(self.builder.package_edges)} package edges with union-find...")
        
        # Process all package edges with union-find
        for node1, node2 in self.builder.package_edges:
            self.builder._uf_union(node1, node2)
        
        # After union-find, assign net types to all package nodes based on their root
        for node in self.builder.graph.nodes():
            if node not in self.builder.node_net_map and node != '0':
                # This is a package node - get its net type from union-find
                net_type = self.builder._get_node_net(node)
                if net_type:
                    # Update node attributes and lowercase map for filtering
                    self.builder.graph.nodes_dict[node]['net_type'] = net_type
                    self.builder.node_net_map_lower[node] = net_type.lower()
        
        self.logger.debug(f"Union-find processing complete - net types propagated to package nodes")
    
    def _update_package_statistics(self):
        """
        Update per-net statistics for package elements after net types have been propagated.
        This is called after _propagate_net_connectivity assigns net types to package nodes.
        """
        self.logger.debug("Updating package element statistics...")
        
        # Iterate through all edges and update statistics for package elements
        for u, v, d in self.builder.graph.edges(data=True):
            elem_type = d.get('type')
            value = d.get('value', 0.0)
            
            # Check if this is a package element
            u_is_die = u in self.builder.node_net_map
            v_is_die = v in self.builder.node_net_map
            u_is_ground = u == '0'
            v_is_ground = v == '0'
            u_is_package = not u_is_die and not u_is_ground
            v_is_package = not v_is_die and not v_is_ground
            
            # Determine if this is a package element:
            # 1. Voltage sources: at least one package node (can connect to ground or die)
            # 2. Other elements (R,C,L): at least one package node
            if elem_type == 'V':
                # Package voltage source: at least one terminal is package node
                if not (u_is_package or v_is_package):
                    continue
            else:
                # For R,C,L: include if at least one node is a package node
                # This includes die-to-package connections (like "rs" resistors)
                if not (u_is_package or v_is_package):
                    continue
            
            # This is a package element - get its net type from the propagated data
            net_type = d.get('net_type')
            if not net_type:
                # Try to get from node attributes (set by union-find)
                u_net = self.builder.graph.nodes_dict[u].get('net_type') if u != '0' else None
                v_net = self.builder.graph.nodes_dict[v].get('net_type') if v != '0' else None
                net_type = u_net or v_net
            
            if net_type:
                # Initialize net stats if needed
                if net_type not in self.builder.stats.net_stats:
                    self.builder.stats.net_stats[net_type] = {
                        'die': {
                            'nodes': set(),
                            'resistors': 0, 'capacitors': 0, 'inductors': 0,
                            'vsources': 0, 'isources': 0,
                            'total_resistance': 0.0, 'total_capacitance': 0.0,
                            'total_inductance': 0.0, 'total_current': 0.0
                        },
                        'package': {
                            'nodes': set(),
                            'resistors': 0, 'capacitors': 0, 'inductors': 0,
                            'vsources': 0, 'isources': 0,
                            'total_resistance': 0.0, 'total_capacitance': 0.0,
                            'total_inductance': 0.0, 'total_current': 0.0
                        }
                    }
                
                # Add to package statistics (only package nodes, not die nodes)
                net_stat = self.builder.stats.net_stats[net_type]['package']
                if u != '0' and u not in self.builder.node_net_map:
                    net_stat['nodes'].add(u)
                if v != '0' and v not in self.builder.node_net_map:
                    net_stat['nodes'].add(v)
                
                if elem_type == 'R':
                    net_stat['resistors'] += 1
                    net_stat['total_resistance'] += value
                elif elem_type == 'C':
                    net_stat['capacitors'] += 1
                    net_stat['total_capacitance'] += value
                elif elem_type == 'L':
                    net_stat['inductors'] += 1
                    net_stat['total_inductance'] += value
                elif elem_type == 'V':
                    net_stat['vsources'] += 1
                elif elem_type == 'I':
                    net_stat['isources'] += 1
                    net_stat['total_current'] += abs(value)
        
        # Print package node names in verbose mode
        if self.logger.isEnabledFor(logging.DEBUG):
            for net_type, net_categories in self.builder.stats.net_stats.items():
                if net_type == 'unmapped':
                    continue
                pkg_nodes = net_categories.get('package', {}).get('nodes', set())
                if pkg_nodes:
                    self.logger.debug(f"Package nodes for net {net_type}: {sorted(pkg_nodes)}")
        
        self.logger.debug("Package element statistics updated")
    
    def _filter_voltage_sources_by_net(self):
        """
        Remove voltage sources and package elements that don't connect to the filtered net.
        This is called after _propagate_net_connectivity has assigned net types to package nodes.

        Strategy: Use tracked edge indices from parsing to avoid iterating over all edges.
        Only voltage sources and package elements (from package.ckt) need to be checked.
        """
        if self.builder.net_filter is None:
            return

        self.logger.info(f"Filtering voltage sources and package elements by net '{self.builder.net_filter}'...")
        self.logger.debug(f"Checking {len(self.builder.vsrc_edge_indices)} vsrc edges, "
                         f"{len(self.builder.package_edge_indices)} package edges")

        # Access internal rustworkx graph for efficient index-based operations
        rx_graph = self.builder.graph._graph
        idx_to_node = self.builder.graph._idx_to_node

        # Collect edge indices to remove and statistics updates
        edges_to_remove = []
        vsrc_remove_count = 0
        pkg_remove_count = 0
        stats_updates = {'R': 0, 'C': 0, 'L': 0, 'V': 0}
        per_net_updates = []  # [(net_type, elem_type, value, u, v), ...]

        # Check voltage source edges (only iterate over tracked vsrc indices)
        for edge_idx in self.builder.vsrc_edge_indices:
            try:
                u_idx, v_idx = rx_graph.get_edge_endpoints_by_index(edge_idx)
                u = idx_to_node[u_idx]
                v = idx_to_node[v_idx]
            except Exception:
                # Edge may have been removed by earlier processing
                continue

            # Get positive terminal (non-ground node)
            vsrc_pos_node = u if u != '0' else v
            if vsrc_pos_node == '0':
                # Both terminals grounded - remove
                edges_to_remove.append(edge_idx)
                vsrc_remove_count += 1
                stats_updates['V'] += 1
                continue

            # Check if positive terminal has the filtered net type
            node_net_lower = self.builder.node_net_map_lower.get(vsrc_pos_node)
            if node_net_lower != self.builder.net_filter:
                edges_to_remove.append(edge_idx)
                vsrc_remove_count += 1
                stats_updates['V'] += 1

        # Check package edges (only iterate over tracked package indices)
        # Skip edges already marked for removal (vsrc edges that are also in package)
        edges_to_remove_set = set(edges_to_remove)

        for edge_idx in self.builder.package_edge_indices:
            if edge_idx in edges_to_remove_set:
                continue  # Already marked for removal as vsrc

            try:
                u_idx, v_idx = rx_graph.get_edge_endpoints_by_index(edge_idx)
                u = idx_to_node[u_idx]
                v = idx_to_node[v_idx]
                edge_data = rx_graph.get_edge_data_by_index(edge_idx)
            except Exception:
                continue

            # Package edges: both nodes are not in die node map
            if u not in self.builder.node_net_map and u != '0' and \
               v not in self.builder.node_net_map and v != '0':
                u_net_lower = self.builder.node_net_map_lower.get(u)
                v_net_lower = self.builder.node_net_map_lower.get(v)

                # Remove if neither node has the filtered net type
                if u_net_lower != self.builder.net_filter and v_net_lower != self.builder.net_filter:
                    edges_to_remove.append(edge_idx)
                    pkg_remove_count += 1

                    elem_type = edge_data.get('type') if edge_data else None
                    if elem_type and elem_type in stats_updates:
                        stats_updates[elem_type] += 1

                    # Track per-net statistics update
                    net_type = edge_data.get('net_type') if edge_data else None
                    value = edge_data.get('value', 0.0) if edge_data else 0.0
                    if net_type:
                        per_net_updates.append((net_type, elem_type, value, u, v))

        # Batch remove edges (sort descending to avoid index invalidation issues)
        for edge_idx in sorted(edges_to_remove, reverse=True):
            try:
                rx_graph.remove_edge_from_index(edge_idx)
            except Exception:
                pass  # Edge already removed

        # Update global statistics
        self.builder.stats.vsources -= stats_updates['V']
        self.builder.stats.resistors -= stats_updates['R']
        self.builder.stats.capacitors -= stats_updates['C']
        self.builder.stats.inductors -= stats_updates['L']
        self.builder.stats.elements_total -= len(edges_to_remove)

        # Update per-net statistics
        for net_type, elem_type, value, u, v in per_net_updates:
            if net_type in self.builder.stats.net_stats:
                net_stat = self.builder.stats.net_stats[net_type].get('package')
                if net_stat:
                    net_stat['nodes'].discard(u)
                    net_stat['nodes'].discard(v)
                    if elem_type == 'R':
                        net_stat['resistors'] -= 1
                        net_stat['total_resistance'] -= value
                    elif elem_type == 'C':
                        net_stat['capacitors'] -= 1
                        net_stat['total_capacitance'] -= value
                    elif elem_type == 'L':
                        net_stat['inductors'] -= 1
                        net_stat['total_inductance'] -= value

        # Remove isolated package nodes (only check package nodes, not all nodes)
        # Build set of package nodes from the tracked edges for efficiency
        package_nodes_to_check = set()
        for edge_idx in self.builder.package_edge_indices:
            try:
                u_idx, v_idx = rx_graph.get_edge_endpoints_by_index(edge_idx)
                u = idx_to_node.get(u_idx)
                v = idx_to_node.get(v_idx)
                if u and u not in self.builder.node_net_map and u != '0':
                    package_nodes_to_check.add(u)
                if v and v not in self.builder.node_net_map and v != '0':
                    package_nodes_to_check.add(v)
            except Exception:
                pass

        package_nodes_to_remove = []
        for node in package_nodes_to_check:
            if node in self.builder.graph and self.builder.graph.degree(node) == 0:
                package_nodes_to_remove.append(node)

        for node in package_nodes_to_remove:
            self.builder.graph.remove_node(node)

        self.logger.info(f"Filtered out {vsrc_remove_count} voltage sources, "
                        f"{pkg_remove_count} package edges, {len(package_nodes_to_remove)} package nodes")
    
    def _parse_tiles(self):
        """Parse all queued tile files with progress bar"""
        if not self.tile_queue:
            return
            
        self.logger.info(f"Parsing {len(self.tile_queue)} tile files...")
        
        with tqdm(total=len(self.tile_queue), desc="Parsing tiles") as pbar:
            for x, y, filepath in self.tile_queue:
                pbar.set_description(f"Parsing tile {x}_{y}")
                
                try:
                    self.builder.current_tile_id = (x, y)
                    
                    # Load corresponding .nd file for node-to-net mapping
                    nd_filepath = Path(filepath).parent / f"tile_{x}_{y}.nd"
                    if not nd_filepath.exists():
                        # Try with .gz extension
                        nd_filepath = Path(filepath).parent / f"tile_{x}_{y}.nd.gz"
                    
                    self.logger.debug(f"Loading node map from {nd_filepath}")
                    tile_node_map = self._load_node_net_map(str(nd_filepath))
                    self.builder.node_net_map.update(tile_node_map)
                    self.logger.debug(f"Loaded {len(tile_node_map)} node mappings for tile {x}_{y}")
                    
                    # Now parse the tile netlist
                    self._parse_file(filepath)
                    self.builder.stats.tiles_parsed += 1
                except Exception as e:
                    self.builder.stats.tiles_failed += 1
                    msg = f"Failed to parse tile {x}_{y}: {e}"
                    if self.strict:
                        raise RuntimeError(msg) from e
                    else:
                        self.logger.warning(msg)
                finally:
                    pbar.update(1)
                    
        self.builder.current_tile_id = None
        
    def _parse_instance_models(self):
        """Parse instance model files (current sources)"""
        if not self.instance_queue:
            return
            
        self.logger.info(f"Parsing {len(self.instance_queue)} instance model files...")
        
        old_file_type = self.builder.current_file_type
        self.builder.current_file_type = 'instance'
        
        with tqdm(total=len(self.instance_queue), desc="Parsing instance models") as pbar:
            for x, y, filepath in self.instance_queue:
                pbar.set_description(f"Parsing instances {x}_{y}")
                
                try:
                    self.builder.current_tile_id = (x, y)
                    self._parse_file(filepath)
                except Exception as e:
                    msg = f"Failed to parse instance models {x}_{y}: {e}"
                    if self.strict:
                        raise RuntimeError(msg) from e
                    else:
                        self.logger.warning(msg)
                finally:
                    pbar.update(1)
                    
        self.builder.current_file_type = old_file_type
        self.builder.current_tile_id = None
        
    def _parse_resistor(self, line: str):
        """Parse resistor: R<name> <node1> <node2> <value> OR r <node1> <node2> <value>"""
        tokens = line.split()
        if len(tokens) < 3:
            self.logger.warning(f"Invalid resistor line: {line}")
            return
        
        # Check format: 'r <node1> <node2> <value>' (no name) or 'R<name> <node1> <node2> <value>'
        first_token = tokens[0]
        if first_token.lower() == 'r':
            # Format: r <node1> <node2> <value>
            if len(tokens) < 4:
                self.logger.warning(f"Invalid resistor line: {line}")
                return
            # Generate unique name
            name = f"r_{tokens[1]}_{tokens[2]}_{id(line)}"
            node1 = tokens[1]
            node2 = tokens[2]
            value_token = tokens[3]
        else:
            # Format: R<name> <node1> <node2> <value>
            if len(tokens) < 4:
                self.logger.warning(f"Invalid resistor line: {line}")
                return
            name = tokens[0]
            node1 = tokens[1]
            node2 = tokens[2]
            value_token = tokens[3]
        
        # Handle boundary nodes (marked with *)
        node1_is_boundary = node1.startswith('*')
        node2_is_boundary = node2.startswith('*')
        
        if node1_is_boundary:
            node1 = node1[1:]  # Remove *
            self.builder.mark_boundary_node(node1)
        if node2_is_boundary:
            node2 = node2[1:]
            self.builder.mark_boundary_node(node2)
            
        try:
            value = self._parse_value(value_token)
            # Convert to KOhm
            value_kohm = value * R_TO_KOHM
            self.builder.add_element('R', node1, node2, value_kohm, name)
        except ValueError as e:
            self.logger.warning(f"Error parsing resistor value in line: {line}: {e}")
            
    def _parse_capacitor(self, line: str):
        """Parse capacitor: C<name> <node1> <node2> <value> [model] OR c <node1> <node2> <value> [model]"""
        tokens = line.split()
        if len(tokens) < 3:
            self.logger.warning(f"Invalid capacitor line: {line}")
            return
        
        # Check format: 'c <node1> <node2> <value>' (no name) or 'C<name> <node1> <node2> <value>'
        first_token = tokens[0]
        if first_token.lower() == 'c':
            # Format: c <node1> <node2> <value> [model]
            if len(tokens) < 4:
                self.logger.warning(f"Invalid capacitor line: {line}")
                return
            # Generate unique name
            name = f"c_{tokens[1]}_{tokens[2]}_{id(line)}"
            node1 = tokens[1]
            node2 = tokens[2]
            value_token = tokens[3]
            model_idx = 4
        else:
            # Format: C<name> <node1> <node2> <value> [model]
            if len(tokens) < 4:
                self.logger.warning(f"Invalid capacitor line: {line}")
                return
            name = tokens[0]
            node1 = tokens[1]
            node2 = tokens[2]
            value_token = tokens[3]
            model_idx = 4
        
        # Handle boundary nodes
        if node1.startswith('*'):
            node1 = node1[1:]
            self.builder.mark_boundary_node(node1)
        if node2.startswith('*'):
            node2 = node2[1:]
            self.builder.mark_boundary_node(node2)
            
        try:
            value = self._parse_value(value_token)
            # Convert to fF
            value_ff = value * C_TO_FF
            
            # Check for nonlinear cap model
            attrs = {}
            if len(tokens) > model_idx and not tokens[model_idx].startswith('*'):
                attrs['nlcap_model'] = tokens[model_idx]
                
            self.builder.add_element('C', node1, node2, value_ff, name, **attrs)
        except ValueError as e:
            self.logger.warning(f"Error parsing capacitor value in line: {line}: {e}")
            
    def _parse_inductor(self, line: str):
        """Parse inductor: L<name> <node1> <node2> <value> OR l <node1> <node2> <value>"""
        tokens = line.split()
        if len(tokens) < 3:
            self.logger.warning(f"Invalid inductor line: {line}")
            return
        
        # Check format: 'l <node1> <node2> <value>' (no name) or 'L<name> <node1> <node2> <value>'
        first_token = tokens[0]
        if first_token.lower() == 'l':
            # Format: l <node1> <node2> <value>
            if len(tokens) < 4:
                self.logger.warning(f"Invalid inductor line: {line}")
                return
            # Generate unique name
            name = f"l_{tokens[1]}_{tokens[2]}_{id(line)}"
            node1 = tokens[1]
            node2 = tokens[2]
            value_token = tokens[3]
        else:
            # Format: L<name> <node1> <node2> <value>
            if len(tokens) < 4:
                self.logger.warning(f"Invalid inductor line: {line}")
                return
            name = tokens[0]
            node1 = tokens[1]
            node2 = tokens[2]
            value_token = tokens[3]
        
        # Handle boundary nodes
        if node1.startswith('*'):
            node1 = node1[1:]
            self.builder.mark_boundary_node(node1)
        if node2.startswith('*'):
            node2 = node2[1:]
            self.builder.mark_boundary_node(node2)
            
        try:
            value = self._parse_value(value_token)
            # Convert to nH
            value_nh = value * L_TO_NH
            self.builder.add_element('L', node1, node2, value_nh, name)
        except ValueError as e:
            self.logger.warning(f"Error parsing inductor value in line: {line}: {e}")
            
    def _parse_mutual_inductor(self, line: str):
        """Parse mutual inductor: K<name> L<name1> L<name2> <coupling>"""
        tokens = line.split()
        if len(tokens) < 4:
            self.logger.warning(f"Invalid mutual inductor line: {line}")
            return
            
        name = tokens[0]
        l1_name = tokens[1]
        l2_name = tokens[2]
        
        try:
            coupling = float(tokens[3])
            self.builder.mutual_inductors[name] = (l1_name, l2_name, coupling)
            self.builder.stats.mutual_inductors += 1
            self.logger.debug(f"Added mutual inductor {name}: {l1_name} <-> {l2_name}, k={coupling}")
        except ValueError as e:
            self.logger.warning(f"Error parsing mutual inductor in line: {line}: {e}")
            
    def _parse_vsource(self, line: str):
        """Parse voltage source: V<name> <node+> <node-> <dc_value> [AC ...] [PWL ...]"""
        tokens = line.split()
        if len(tokens) < 4:
            self.logger.warning(f"Invalid voltage source line: {line}")
            return
            
        name = tokens[0]
        node_pos = tokens[1]
        node_neg = tokens[2]
        
        # Handle boundary nodes
        if node_pos.startswith('*'):
            node_pos = node_pos[1:]
            self.builder.mark_boundary_node(node_pos)
        if node_neg.startswith('*'):
            node_neg = node_neg[1:]
            self.builder.mark_boundary_node(node_neg)
            
        try:
            dc_value = self._parse_value(tokens[3])
            
            # Parse additional parameters (AC, PWL, etc.)
            attrs = {'dc': dc_value}
            i = 4
            while i < len(tokens):
                token = tokens[i].upper()
                if token == 'AC' and i + 1 < len(tokens):
                    attrs['ac'] = float(tokens[i + 1])
                    i += 2
                elif token == 'PORTID' and i + 1 < len(tokens):
                    port_str = tokens[i + 1]
                    if '=' in port_str:
                        attrs['portid'] = port_str.split('=')[1]
                    i += 2
                elif token.startswith('PWL'):
                    # PWL(...) - extract but don't parse fully
                    attrs['pwl'] = line[line.upper().find('PWL'):]
                    break
                else:
                    i += 1
                    
            self.builder.add_element('V', node_pos, node_neg, dc_value, name, **attrs)
            self.builder.vsrc_dict[name] = attrs
            
        except ValueError as e:
            self.logger.warning(f"Error parsing voltage source in line: {line}: {e}")
            
    def _parse_isource(self, line: str):
        """
        Parse current source with full waveform support.
        
        Handles:
        - I<name> <node+> <node-> <dc_value> [static_value=...] [pulse(...)] [pwl(...)] [fsdb ...]
        
        All current values are converted to mA and stored in CurrentSource objects
        for both static DC analysis and time-domain evaluation.
        """
        # Use the enhanced parser to extract all waveform data
        isrc = _parse_current_source_line(line)
        if isrc is None:
            self.logger.warning(f"Invalid current source line: {line}")
            return
        
        name = isrc.name
        node_pos = isrc.node1
        node_neg = isrc.node2
        
        # Handle boundary nodes (marked with *)
        if node_pos.startswith('*'):
            node_pos = node_pos[1:]
            self.builder.mark_boundary_node(node_pos)
        if node_neg.startswith('*'):
            node_neg = node_neg[1:]
            self.builder.mark_boundary_node(node_neg)
        
        # Update node names in CurrentSource after boundary handling
        isrc.node1 = node_pos
        isrc.node2 = node_neg
        
        # Convert all current values from Amperes to mA
        isrc.dc_value = isrc.dc_value * I_TO_MA
        if isrc.static_value is not None:
            isrc.static_value = isrc.static_value * I_TO_MA
        
        # Convert pulse waveform values to mA
        for pulse in isrc.pulses:
            pulse.v1 *= I_TO_MA
            pulse.v2 *= I_TO_MA
        
        # Convert PWL waveform values to mA
        for pwl in isrc.pwls:
            pwl.points = [(t, v * I_TO_MA) for t, v in pwl.points]
        
        # Calculate static current value (mA) for DC analysis
        static_current_ma = isrc.get_static_current()
        
        # Build edge attributes for graph element
        attrs = {
            'dc': static_current_ma,
            'has_waveform': isrc.has_waveform_data()
        }
        
        # Parse FSDB reference if present (legacy support)
        tokens = line.split()
        for i, token in enumerate(tokens):
            if token.lower() == 'fsdb' and i + 1 < len(tokens):
                attrs['fsdb_path'] = tokens[i + 1]
                if i + 2 < len(tokens):
                    try:
                        attrs['fsdb_coeff'] = float(tokens[i + 2])
                    except ValueError:
                        pass
                if i + 3 < len(tokens):
                    try:
                        attrs['fsdb_shift'] = float(tokens[i + 3])
                    except ValueError:
                        pass
                break
        
        # Extract coordinates from instance name (e.g., i_cell:1000_2000:vdd)
        coord_match = re.search(r':(\d+)_(\d+):', name)
        if coord_match:
            attrs['inst_x'] = int(coord_match.group(1))
            attrs['inst_y'] = int(coord_match.group(2))
        
        # Add element to graph
        added = self.builder.add_element('I', node_pos, node_neg, static_current_ma, name, **attrs)
        
        # Only store if element was actually added (not filtered by net_filter)
        if added:
            # Backward compatibility: instance_node_map
            self.builder.instance_node_map[name] = [node_pos, node_neg]
            
            # Store full CurrentSource for time-domain analysis
            self.builder.instance_sources[name] = isrc
            
            # Update statistics
            if isrc.has_waveform_data():
                self.builder.stats.instances_with_waveforms += 1
                
                # Update per-net waveform statistics
                net_type = self.builder.node_net_map.get(node_pos) or self.builder.node_net_map.get(node_neg)
                if net_type and net_type in self.builder.stats.net_stats:
                    # Current sources from instanceModels are always 'die' category
                    if 'die' in self.builder.stats.net_stats[net_type]:
                        self.builder.stats.net_stats[net_type]['die']['isources_with_waveforms'] += 1
                        
            self.builder.stats.total_static_current_ma += abs(static_current_ma)
            
    def _parse_controlled_source(self, line: str, source_type: str):
        """Parse controlled sources (E/F/G/H)"""
        # E: VCVS, F: CCCS, G: VCCS, H: CCVS
        tokens = line.split()
        if len(tokens) < 6:
            self.logger.warning(f"Invalid controlled source line: {line}")
            return
            
        name = tokens[0]
        out_pos = tokens[1]
        out_neg = tokens[2]
        
        # Handle boundary nodes
        for i in [1, 2, 3, 4]:
            if i < len(tokens) and tokens[i].startswith('*'):
                tokens[i] = tokens[i][1:]
                self.builder.mark_boundary_node(tokens[i])
        
        attrs = {
            'source_type': source_type,
            'ctrl_pos': tokens[3],
            'ctrl_neg': tokens[4]
        }
        
        try:
            gain = float(tokens[5])
            # Store as special edge with type indicating controlled source
            self.builder.add_element(source_type, out_pos, out_neg, gain, name, **attrs)
        except ValueError as e:
            self.logger.warning(f"Error parsing controlled source in line: {line}: {e}")
            
    def _parse_subcircuit_definition(self, line: str):
        """Parse .subckt definition (store for later expansion)"""
        tokens = line.split()
        if len(tokens) < 3:
            return
            
        subckt_name = tokens[1]
        pins = tokens[2:]
        
        # Read body until .ends
        body_lines = []
        # Note: This is simplified - in full implementation, would need to handle
        # nested subcircuits properly
        
        self.subcircuits[subckt_name] = {
            'pins': pins,
            'body': body_lines
        }
        self.logger.debug(f"Defined subcircuit: {subckt_name} with pins: {pins}")
        
    def _parse_subcircuit_instance(self, line: str):
        """Parse subcircuit instance and expand it (flatten hierarchy)"""
        tokens = line.split()
        if len(tokens) < 3:
            self.logger.warning(f"Invalid subcircuit instance: {line}")
            return
            
        inst_name = tokens[0]
        subckt_name = tokens[-1]
        node_list = tokens[1:-1]
        
        if subckt_name not in self.subcircuits:
            self.logger.warning(f"Subcircuit {subckt_name} not defined for instance {inst_name}")
            return
            
        # Flatten: expand subcircuit with hierarchical naming
        subckt = self.subcircuits[subckt_name]
        pins = subckt['pins']
        
        if len(node_list) != len(pins):
            self.logger.warning(f"Pin count mismatch for instance {inst_name}")
            return
            
        # Create pin mapping
        pin_map = dict(zip(pins, node_list))
        
        # Process subcircuit body (simplified - would need full implementation)
        # For now, just log the expansion
        self.logger.debug(f"Expanding subcircuit instance {inst_name} of type {subckt_name}")
        
    def _parse_value(self, value_str: str) -> float:
        """Parse SPICE value with suffix (K, M, G, etc.)"""
        value_str = value_str.strip()
        
        # Check if this is a parameter reference
        # Parameters are case-insensitive in SPICE
        param_key = value_str.upper()
        if param_key in self.builder.parameters:
            value_str = self.builder.parameters[param_key]
        # Also check case-preserving lookup
        elif value_str in self.builder.parameters:
            value_str = self.builder.parameters[value_str]
        
        value_str = value_str.strip().upper()
        
        # SPICE suffixes - order matters! Check longer suffixes first (MEG before M)
        suffixes = [
            ('MEG', 1e6), ('T', 1e12), ('G', 1e9), ('X', 1e6), ('K', 1e3),
            ('M', 1e-3), ('U', 1e-6), ('N', 1e-9), ('P', 1e-12), ('F', 1e-15)
        ]
        
        # Check for suffix
        for suffix, multiplier in suffixes:
            if value_str.endswith(suffix):
                base = value_str[:-len(suffix)]
                return float(base) * multiplier
                
        # No suffix
        return float(value_str)
        
    def _perform_validation(self):
        """Perform sanity checks on the parsed netlist"""
        self.logger.info("Performing netlist validation...")
        
        self.builder.stats.nodes_before_cleanup = self.builder.graph.number_of_nodes()
        
        # Check for shorted resistors
        self._check_shorts()
        
        # Check for floating nodes
        self._check_floating_nodes()
        
        # Check for grounded nodes
        self._check_grounded_nodes()
        
    def _check_shorts(self):
        """Detect and report shorted resistors.

        Optimized to use rustworkx directly instead of wrapper iteration.
        """
        rx_graph = self.builder.graph._graph
        idx_to_node = self.builder.graph._idx_to_node
        shorts = []

        for edge_idx in rx_graph.edge_indices():
            data = rx_graph.get_edge_data_by_index(edge_idx)
            if data and data.get('type') == 'R':
                value = data.get('value', float('inf'))
                if value < SHORT_THRESHOLD:
                    u_idx, v_idx = rx_graph.get_edge_endpoints_by_index(edge_idx)
                    shorts.append((idx_to_node[u_idx], idx_to_node[v_idx],
                                  data.get('elem_name'), value))
                    self.builder.stats.shorted_elements += 1

        if shorts:
            msg = f"Found {len(shorts)} shorted resistors:\n"
            for u, v, name, value in shorts[:10]:  # Show first 10
                msg += f"  {name}: {u} <-> {v} = {value:.2e} KOhm\n"
            if len(shorts) > 10:
                msg += f"  ... and {len(shorts) - 10} more\n"
                
            if self.strict:
                raise ValueError(msg)
            else:
                self.logger.warning(msg)
                
    def _check_floating_nodes(self):
        """Detect nodes not connected to any voltage source.

        Optimized to:
        1. Use tracked vsrc_edge_indices instead of iterating all edges
        2. Use directed graph directly - node_connected_component already
           handles weak connectivity for directed graphs via
           rx.weakly_connected_components(), so to_undirected() is unnecessary.
        """
        rx_graph = self.builder.graph._graph
        idx_to_node = self.builder.graph._idx_to_node

        # Find all nodes connected to voltage sources using tracked indices
        grounded_nodes = set()
        for edge_idx in self.builder.vsrc_edge_indices:
            try:
                u_idx, v_idx = rx_graph.get_edge_endpoints_by_index(edge_idx)
                grounded_nodes.add(idx_to_node[u_idx])
                grounded_nodes.add(idx_to_node[v_idx])
            except Exception:
                # Edge may have been removed by earlier processing
                pass

        # BFS from grounded nodes using weak connectivity on directed graph
        connected_nodes = set()
        for node in grounded_nodes:
            if node not in connected_nodes:
                component = node_connected_component(self.builder.graph, node)
                connected_nodes.update(component)

        # Find floating nodes
        all_nodes = set(self.builder.graph.nodes())
        floating = all_nodes - connected_nodes

        self.builder.stats.floating_nodes = len(floating)

        if floating:
            msg = f"Found {len(floating)} floating nodes (not connected to voltage source)"
            if len(floating) <= 10:
                msg += f": {', '.join(list(floating)[:10])}"
            else:
                msg += f". First 10: {', '.join(list(floating)[:10])}"

            self.logger.warning(msg)
            
    def _check_grounded_nodes(self):
        """Check for nodes directly grounded via shorts"""
        # This would be more complex - simplified version
        ground_node = '0'
        if ground_node in self.builder.graph:
            neighbors = list(self.builder.graph.neighbors(ground_node))
            self.logger.debug(f"Ground node '0' has {len(neighbors)} direct connections")
    
    def _identify_vsrc_nodes(self):
        """
        Identify nodes connected to voltage sources via zero-valued resistors.
        These are typically package nodes connected to ideal voltage sources through
        resistors named 'rs' (or custom pattern) with value == 0.0.
        
        The identification propagates through zero-resistance paths up to a 
        configurable depth limit.
        """
        self.logger.info("Identifying voltage source nodes...")
        
        vsrc_nodes = set()
        
        # Step 1: Find all nodes directly connected to voltage sources
        for u, v, data in self.builder.graph.edges(data=True):
            if data.get('type') == 'V':
                vsrc_nodes.add(u)
                vsrc_nodes.add(v)
        
        # Step 2: Find zero-valued resistors matching pattern
        zero_resistors = []
        for u, v, key, data in self.builder.graph.edges(keys=True, data=True):
            if data.get('type') == 'R':
                elem_name = data.get('elem_name', '')
                value = data.get('value', float('inf'))
                # Check for exact pattern match and zero resistance
                if elem_name == self.vsrc_resistor_pattern and value == 0.0:
                    zero_resistors.append((u, v))
        
        # Step 3: Propagate through zero-resistance paths using BFS with depth limit
        from collections import deque
        
        visited = set(vsrc_nodes)
        queue = deque([(node, 0) for node in vsrc_nodes])
        
        while queue:
            node, depth = queue.popleft()
            
            if depth >= self.vsrc_depth_limit:
                continue
            
            # Check all zero-resistance connections from this node
            for u, v in zero_resistors:
                neighbor = None
                if u == node and v not in visited:
                    neighbor = v
                elif v == node and u not in visited:
                    neighbor = u
                
                if neighbor:
                    visited.add(neighbor)
                    vsrc_nodes.add(neighbor)
                    queue.append((neighbor, depth + 1))
        
        # Step 4: Mark nodes in graph
        for node in vsrc_nodes:
            if node in self.builder.graph:
                self.builder.graph.nodes_dict[node]['is_vsrc_node'] = True
        
        # Update statistics
        self.builder.stats.vsrc_nodes = len(vsrc_nodes)
        
        # Store in graph metadata
        self.builder.graph.graph['vsrc_nodes'] = vsrc_nodes
        
        self.logger.info(f"Identified {len(vsrc_nodes)} voltage source nodes")
        if zero_resistors:
            self.logger.debug(f"Found {len(zero_resistors)} zero-valued '{self.vsrc_resistor_pattern}' resistors")
    
    def _compute_layer_stats(self):
        """
        Compute per-layer and per-net statistics for nodes and elements.
        Aggregates counts by layer identifier and net type.
        """
        self.logger.info("Computing layer statistics...")
        
        # Structure: layer_stats_by_net[net][layer] = stats
        layer_stats_by_net = defaultdict(lambda: defaultdict(lambda: {
            'nodes': 0,
            'vsrc_nodes': 0,
            'resistors': 0,
            'capacitors': 0,
            'inductors': 0,
            'vsources': 0,
            'isources': 0
        }))
        
        # Count nodes per layer per net
        for node, data in self.builder.graph.nodes(data=True):
            net_type = data.get('net_type') or self.builder._get_node_net(node)
            
            if net_type:
                # Determine if this is a package node
                is_package_node = node not in self.builder.node_net_map and node != '0'
                
                if is_package_node:
                    layer = 'package'
                else:
                    layer = data.get('layer')
                
                layer_stats_by_net[net_type][layer]['nodes'] += 1
                if data.get('is_vsrc_node', False):
                    layer_stats_by_net[net_type][layer]['vsrc_nodes'] += 1
        
        # Count elements per layer per net
        # Count both intra-layer and inter-layer elements
        for u, v, data in self.builder.graph.edges(data=True):
            elem_type = data.get('type')
            net_type = data.get('net_type') or self.builder._get_node_net(u)
            
            # Determine layer for this element
            u_is_die = u in self.builder.node_net_map
            v_is_die = v in self.builder.node_net_map
            u_is_package = not u_is_die and u != '0'
            v_is_package = not v_is_die and v != '0'
            
            # Determine layer based on node types
            if u_is_package or v_is_package:
                # At least one node is package - count as package layer
                layer = 'package'
            else:
                # Both are die nodes or ground, get their layers
                # Treat node '0' (ground) as having no layer - use the other node's layer
                u_layer = self.builder.graph.nodes_dict[u].get('layer') if u in self.builder.graph and u != '0' else None
                v_layer = self.builder.graph.nodes_dict[v].get('layer') if v in self.builder.graph and v != '0' else None
                
                # Handle ground node cases
                if u == '0':
                    layer = v_layer
                elif v == '0':
                    layer = u_layer
                elif u_layer == v_layer:
                    # Same layer (intra-layer)
                    layer = u_layer
                elif u_layer and v_layer:
                    # Inter-layer connection - create combined layer name
                    # Sort to ensure consistency (e.g., "19-21" not "21-19")
                    layers_sorted = sorted([u_layer, v_layer], key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else x))
                    layer = f"{layers_sorted[0]}-{layers_sorted[1]}"
                else:
                    # One or both layers unknown
                    layer = u_layer or v_layer
            
            if net_type:
                if elem_type == 'R':
                    layer_stats_by_net[net_type][layer]['resistors'] += 1
                elif elem_type == 'C':
                    layer_stats_by_net[net_type][layer]['capacitors'] += 1
                elif elem_type == 'L':
                    layer_stats_by_net[net_type][layer]['inductors'] += 1
                elif elem_type == 'V':
                    layer_stats_by_net[net_type][layer]['vsources'] += 1
                elif elem_type == 'I':
                    layer_stats_by_net[net_type][layer]['isources'] += 1
        
        # Convert to regular dict and store
        self.builder.stats.layer_stats_by_net = {net: dict(layers) for net, layers in layer_stats_by_net.items()}
        self.builder.graph.graph['layer_stats_by_net'] = self.builder.stats.layer_stats_by_net
        
        # Also compute global layer stats for backwards compatibility
        layer_stats = defaultdict(lambda: {
            'nodes': 0, 'vsrc_nodes': 0, 'resistors': 0,
            'capacitors': 0, 'inductors': 0, 'vsources': 0, 'isources': 0
        })
        for net_layers in layer_stats_by_net.values():
            for layer, stats in net_layers.items():
                for key in stats:
                    layer_stats[layer][key] += stats[key]
        self.builder.stats.layer_stats = dict(layer_stats)
        self.builder.graph.graph['layer_stats'] = dict(layer_stats)
        
        # Log summary
        total_layers = set()
        for net_layers in layer_stats_by_net.values():
            total_layers.update(k for k in net_layers.keys() if k is not None and k != 'package')
        if total_layers:
            self.logger.info(f"Found {len(total_layers)} layers")
            for net, net_layers in sorted(layer_stats_by_net.items()):
                for layer, stats in sorted(net_layers.items(), key=lambda x: (x[0] is None, x[0] == 'package', x[0])):
                    if layer is not None and layer != 'package':
                        self.logger.debug(f"  Net {net}, Layer {layer}: {stats['nodes']} nodes, "
                                        f"{stats['resistors']} resistors, {stats['capacitors']} capacitors")
            
    def _print_statistics(self):
        """Print parsing statistics"""
        stats = self.builder.stats
        
        # Per-net statistics
        if stats.net_stats:
            print("\n" + "-"*70)
            print("Per-Net Statistics:")
            print("-"*70)
            
            # Sort nets by name, exclude 'unmapped'
            # If net filter is active, only show that net
            if self.net_filter:
                sorted_nets = [net for net in stats.net_stats.keys() if net != 'unmapped' and net.lower() == self.net_filter.lower()]
            else:
                sorted_nets = sorted([net for net in stats.net_stats.keys() if net != 'unmapped'])
            
            for net in sorted_nets:
                net_categories = stats.net_stats[net]
                print(f"\n  Net: {net}")
                
                for category in ['die', 'package']:
                    if category not in net_categories:
                        continue
                    
                    net_stat = net_categories[category]
                    
                    # Skip if no elements in this category
                    total_elems = (net_stat['resistors'] + net_stat['capacitors'] + 
                                 net_stat['inductors'] + net_stat['vsources'] + net_stat['isources'])
                    if total_elems == 0:
                        continue
                    
                    print(f"    [{category.upper()}]")
                    print(f"      Nodes: {len(net_stat['nodes']):,}")
                    print(f"      Resistors: {net_stat['resistors']:,}")
                    if net_stat['resistors'] > 0:
                        avg_r = net_stat['total_resistance'] / net_stat['resistors']
                        print(f"        Total: {net_stat['total_resistance']:.3f} KOhm")
                        print(f"        Average: {avg_r:.6f} KOhm")
                    
                    print(f"      Capacitors: {net_stat['capacitors']:,}")
                    if net_stat['capacitors'] > 0:
                        avg_c = net_stat['total_capacitance'] / net_stat['capacitors']
                        print(f"        Total: {net_stat['total_capacitance']:.3f} fF")
                        print(f"        Average: {avg_c:.6f} fF")
                    
                    if net_stat['inductors'] > 0:
                        print(f"      Inductors: {net_stat['inductors']:,}")
                        avg_l = net_stat['total_inductance'] / net_stat['inductors']
                        print(f"        Total: {net_stat['total_inductance']:.3f} nH")
                        print(f"        Average: {avg_l:.6f} nH")
                    
                    if net_stat['vsources'] > 0:
                        print(f"      Voltage Sources: {net_stat['vsources']:,}")
                    
                    if net_stat['isources'] > 0:
                        print(f"      Current Sources: {net_stat['isources']:,}")
                        waveform_count = net_stat.get('isources_with_waveforms', 0)
                        if waveform_count > 0:
                            print(f"        With Waveforms: {waveform_count:,}")
                        avg_i = net_stat['total_current'] / net_stat['isources']
                        print(f"        Total Current: {net_stat['total_current']:.3f} mA")
                        print(f"        Average: {avg_i:.6f} mA")
        
        print("\n" + "="*70 + "\n")
        stats = self.builder.stats
        
        self.logger.info("=" * 60)
        self.logger.info("Netlist Parsing Statistics")
        self.logger.info("=" * 60)
        self.logger.info(f"Nodes: {stats.nodes_after_cleanup}")
        if self.validate and stats.nodes_before_cleanup > 0:
            self.logger.info(f"  (before cleanup: {stats.nodes_before_cleanup})")
        if stats.vsrc_nodes > 0:
            self.logger.info(f"  Voltage Source Nodes: {stats.vsrc_nodes}")
        self.logger.info(f"Elements: {stats.elements_total}")
        self.logger.info(f"  Resistors: {stats.resistors}")
        self.logger.info(f"  Capacitors: {stats.capacitors}")
        self.logger.info(f"  Inductors: {stats.inductors}")
        self.logger.info(f"  Voltage Sources: {stats.vsources}")
        self.logger.info(f"  Current Sources: {stats.isources}")
        self.logger.info(f"  Mutual Inductors: {stats.mutual_inductors}")
        if self.builder.tile_grid:
            self.logger.info(f"Tile Grid: {self.builder.tile_grid[0]} x {self.builder.tile_grid[1]}")
            self.logger.info(f"  Tiles Parsed: {stats.tiles_parsed}")
            if stats.tiles_failed > 0:
                self.logger.info(f"  Tiles Failed: {stats.tiles_failed}")
        self.logger.info(f"Boundary Nodes: {stats.boundary_nodes}")
        if stats.package_nodes > 0:
            self.logger.info(f"Package Nodes: {stats.package_nodes}")
        if self.validate:
            self.logger.info(f"Shorted Elements: {stats.shorted_elements}")
            self.logger.info(f"Floating Nodes: {stats.floating_nodes}")
        self.logger.info(f"Instance-Node Mappings: {len(self.builder.instance_node_map)}")
        
        # Print layer statistics by net
        if hasattr(stats, 'layer_stats_by_net') and stats.layer_stats_by_net:
            # Filter nets if net_filter is active
            if self.net_filter:
                nets_to_show = [net for net in stats.layer_stats_by_net.keys() if net.lower() == self.net_filter.lower()]
            else:
                nets_to_show = sorted(stats.layer_stats_by_net.keys())
            
            for net in nets_to_show:
                net_layers = stats.layer_stats_by_net[net]
                
                # Separate single layers and inter-layer connections
                single_layers = []
                inter_layers = []
                for k in net_layers.keys():
                    if k is not None and k != 'package':
                        if '-' in str(k):
                            inter_layers.append(k)
                        else:
                            single_layers.append(k)
                
                if single_layers or inter_layers or 'package' in net_layers:
                    self.logger.info("=" * 60)
                    self.logger.info(f"Layer Statistics for Net: {net} ({len(single_layers)} layers)")
                    self.logger.info("=" * 60)
                    
                    # Print header
                    self.logger.info(f"{'Layer':<15} {'Nodes':>8} {'Vsrc':>6} {'Res':>8} {'Cap':>8} {'Ind':>6} {'Isrc':>6}")
                    self.logger.info("-" * 60)
                    
                    # Sort: numeric layers first (as ints), then alphabetic layers
                    def layer_sort_key(x):
                        if isinstance(x, str) and x.isdigit():
                            return (0, int(x))
                        else:
                            return (1, str(x))
                    
                    # Initialize totals
                    total_stats = {
                        'nodes': 0, 'vsrc_nodes': 0, 'resistors': 0,
                        'capacitors': 0, 'inductors': 0, 'vsources': 0, 'isources': 0
                    }
                    
                    # Print single-layer stats (sorted by layer name)
                    for layer in sorted(single_layers, key=layer_sort_key):
                        layer_stat = net_layers[layer]
                        self.logger.info(
                            f"{str(layer):<15} "
                            f"{layer_stat['nodes']:>8} "
                            f"{layer_stat['vsrc_nodes']:>6} "
                            f"{layer_stat['resistors']:>8} "
                            f"{layer_stat['capacitors']:>8} "
                            f"{layer_stat['inductors']:>6} "
                            f"{layer_stat['isources']:>6}"
                        )
                        # Accumulate totals
                        for key in total_stats:
                            total_stats[key] += layer_stat[key]
                    
                    # Print inter-layer stats (sorted)
                    for layer in sorted(inter_layers):
                        layer_stat = net_layers[layer]
                        self.logger.info(
                            f"{str(layer):<15} "
                            f"{layer_stat['nodes']:>8} "
                            f"{layer_stat['vsrc_nodes']:>6} "
                            f"{layer_stat['resistors']:>8} "
                            f"{layer_stat['capacitors']:>8} "
                            f"{layer_stat['inductors']:>6} "
                            f"{layer_stat['isources']:>6}"
                        )
                        # Accumulate totals
                        for key in total_stats:
                            total_stats[key] += layer_stat[key]
                    
                    # Print Package row if exists
                    if 'package' in net_layers:
                        layer_stat = net_layers['package']
                        self.logger.info(
                            f"{'Package':<15} "
                            f"{layer_stat['nodes']:>8} "
                            f"{layer_stat['vsrc_nodes']:>6} "
                            f"{layer_stat['resistors']:>8} "
                            f"{layer_stat['capacitors']:>8} "
                            f"{layer_stat['inductors']:>6} "
                            f"{layer_stat['isources']:>6}"
                        )
                        # Add package to totals
                        for key in total_stats:
                            total_stats[key] += layer_stat[key]
                    
                    # Print Total row
                    self.logger.info("-" * 60)
                    self.logger.info(
                        f"{'Total':<15} "
                        f"{total_stats['nodes']:>8} "
                        f"{total_stats['vsrc_nodes']:>6} "
                        f"{total_stats['resistors']:>8} "
                        f"{total_stats['capacitors']:>8} "
                        f"{total_stats['inductors']:>6} "
                        f"{total_stats['isources']:>6}"
                    )
        
        self.logger.info("=" * 60)


def main():
    """Command-line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Parse PDN netlist and convert to NetworkX graph',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic parsing
  python pdn_parser.py --netlist-dir /path/to/netlist
  
  # With validation and output
  python pdn_parser.py --netlist-dir /path/to/netlist --validate --output pdn.graphml
  
  # Strict mode (fail on errors)
  python pdn_parser.py --netlist-dir /path/to/netlist --validate --strict
  
  # Filter specific net
  python pdn_parser.py --netlist-dir /path/to/netlist --net vdd
  
  # With memory profiling
  python pdn_parser.py --netlist-dir /path/to/netlist --profile-memory
        """
    )
    
    parser.add_argument('--netlist-dir', type=str, default='.',
                       help='Directory containing ckt.sp (default: current directory)')
    parser.add_argument('--output', '-o', type=str,
                       help='Output file (.graphml or .pkl)')
    parser.add_argument('--validate', action='store_true',
                       help='Enable sanity checks (shorts, floating nodes, etc.)')
    parser.add_argument('--strict', action='store_true',
                       help='Treat warnings as errors (fail on validation issues)')
    parser.add_argument('--net', type=str,
                       help='Filter specific power net (e.g., vdd, vcc)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output (debug logging)')
    parser.add_argument('--profile-memory', action='store_true',
                       help='Profile memory usage (requires memory_profiler)')
    parser.add_argument('--vsrc-resistor-pattern', type=str, default='rs',
                       help='Resistor name pattern for voltage source identification (default: rs)')
    parser.add_argument('--vsrc-depth-limit', type=int, default=3,
                       help='Depth limit for voltage source node propagation (default: 3)')
    
    # Plotting options
    parser.add_argument('--plot-layer', type=str,
                       help='Plot specific layer (e.g., "5", "M1")')
    parser.add_argument('--plot-all-layers', action='store_true',
                       help='Plot all layers in a single figure')
    parser.add_argument('--plot-output', type=str,
                       help='Output file for plot (default: show plot)')
    parser.add_argument('--plot-bin-size', type=int,
                       help='Bin size for grid aggregation (auto-calculated if not specified)')
    parser.add_argument('--plot-statistic', type=str, default='node_count',
                       choices=['node_count', 'avg_voltage', 'total_capacitance', 'avg_resistance'],
                       help='Statistic to display in plot (default: node_count)')
    
    args = parser.parse_args()
    
    # Memory profiling
    if args.profile_memory:
        try:
            from memory_profiler import profile
            # Wrap parse function
            netlist_parser = NetlistParser(
                args.netlist_dir,
                validate=args.validate,
                strict=args.strict,
                net_filter=args.net,
                verbose=args.verbose,
                vsrc_resistor_pattern=args.vsrc_resistor_pattern,
                vsrc_depth_limit=args.vsrc_depth_limit
            )
            profiled_parse = profile(netlist_parser.parse)
            graph = profiled_parse()
        except ImportError:
            print("ERROR: memory_profiler not installed. Install with: pip install memory_profiler")
            sys.exit(1)
    else:
        # Normal parsing
        netlist_parser = NetlistParser(
            args.netlist_dir,
            validate=args.validate,
            strict=args.strict,
            net_filter=args.net,
            verbose=args.verbose,
            vsrc_resistor_pattern=args.vsrc_resistor_pattern,
            vsrc_depth_limit=args.vsrc_depth_limit
        )
        graph = netlist_parser.parse()
    
    # Save output
    if args.output:
        output_path = Path(args.output)

        if output_path.suffix == '.graphml':
            print(f"ERROR: GraphML export not supported with rustworkx backend")
            print("Use .pkl format instead")
            sys.exit(1)
        elif output_path.suffix == '.pkl':
            # Register pickle handlers to fix __main__ module references
            # When run as __main__, dataclasses get __module__ = '__main__'
            if __name__ == '__main__':
                import copyreg
                from dataclasses import fields
                import pdn.pdn_parser as target_module

                def make_reducer(cls_name):
                    """Create reducer that uses class from pdn.pdn_parser module."""
                    def reducer(obj):
                        # Get the correct class from the module
                        correct_cls = getattr(target_module, cls_name)
                        # Only include fields with init=True (exclude _times_cache etc.)
                        field_values = tuple(
                            getattr(obj, f.name) for f in fields(obj) if f.init
                        )
                        # Return (class, args_tuple) for reconstruction
                        return (correct_cls, field_values)
                    return reducer

                for cls_name in ('InstanceInfo', 'Pulse', 'PWL', 'CurrentSource'):
                    cls = globals()[cls_name]
                    copyreg.pickle(cls, make_reducer(cls_name))

            with open(output_path, 'wb') as f:
                pickle.dump(graph, f)
            print(f"Graph saved to: {output_path}")
        else:
            print(f"ERROR: Unsupported output format: {output_path.suffix}")
            print("Supported formats: .pkl")
            sys.exit(1)
    
    # Print summary
    print(f"\nGraph has {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
    print(f"Instance-node mappings: {len(graph.graph.get('instance_node_map', {}))}")
    
    # Plotting using PDNPlotter
    if args.plot_layer or args.plot_all_layers:
        try:
            # Get net connectivity for PDNPlotter
            net_connectivity = graph.graph.get('net_connectivity', {})
            
            # Create plotter
            plotter = PDNPlotter(graph, net_connectivity, logging.getLogger(__name__))
            
            # Set up output path and filename
            if args.plot_output:
                output_path = Path(args.plot_output)
                if output_path.suffix:  # It's a file
                    output_dir = output_path.parent
                    output_filename = output_path.name
                else:  # It's a directory
                    output_dir = output_path
                    output_filename = None
            else:
                output_dir = Path('./results')
                output_filename = None
            
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Determine net name for plotting
            net_name = args.net if args.net else 'PDN'
            
            if args.plot_layer:
                print(f"\nPlotting layer {args.plot_layer}...")
                # Parse layer string (comma-separated list)
                layers = [l.strip() for l in args.plot_layer.split(',')]
                
                # Generate heatmaps for specified layers
                plotter.generate_layer_heatmaps(
                    net_name=net_name,
                    output_path=output_dir,
                    plot_layers=layers,
                    plot_bin_size=args.plot_bin_size,
                    anisotropic_bins=True,  # Use anisotropic bins by default
                    output_filename=output_filename
                )
                
            elif args.plot_all_layers:
                print(f"\nPlotting all layers...")
                
                # Generate heatmaps for all layers (None = all layers)
                plotter.generate_layer_heatmaps(
                    net_name=net_name,
                    output_path=output_dir,
                    plot_layers=None,  # None = all layers
                    plot_bin_size=args.plot_bin_size,
                    anisotropic_bins=True,
                    output_filename=output_filename
                )
                    
        except NameError:
            print("ERROR: PDNPlotter not available. Install pdn_plotter.py in the same directory.")
        except Exception as e:
            print(f"ERROR during plotting: {e}")
            import traceback
            traceback.print_exc()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

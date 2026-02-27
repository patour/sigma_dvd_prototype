"""
SPICE lexical helpers: constants, regex, value parsing, file reading, net filters.

This module contains standalone utilities with no project dependencies beyond stdlib.
"""

import gzip
import logging
import os
import re
import sys
from contextvars import ContextVar
from typing import Dict, Optional, TextIO


# =============================================================================
# Constants matching C++ parser
# =============================================================================

GMAX = 1e5  # Maximum conductance (from parser.cc line 1119)
SHORT_THRESHOLD = 1e-6  # Resistance threshold for shorts (KOhm)
INVALID_STATIC_CURRENT = -999.0

# Unit conversions (matching C++ parser)
R_TO_KOHM = 1e-3  # Ohm to KOhm
C_TO_FF = 1e15   # Farad to fF
L_TO_NH = 1e9    # Henry to nH
I_TO_MA = 1e3    # Ampere to mA


# =============================================================================
# Pre-interned strings for memory optimization
# =============================================================================

_ELEM_R = sys.intern('R')
_ELEM_C = sys.intern('C')
_ELEM_L = sys.intern('L')
_ELEM_V = sys.intern('V')
_ELEM_I = sys.intern('I')
_ELEM_TYPES = {'R': _ELEM_R, 'C': _ELEM_C, 'L': _ELEM_L, 'V': _ELEM_V, 'I': _ELEM_I}

# Interned edge attribute keys
_KEY_TYPE = sys.intern('type')
_KEY_VALUE = sys.intern('value')
_KEY_ELEM_NAME = sys.intern('elem_name')
_KEY_TILE_ID = sys.intern('tile_id')
_KEY_NET_TYPE = sys.intern('net_type')

# Interned category strings for statistics
_CAT_DIE = sys.intern('die')
_CAT_PACKAGE = sys.intern('package')
_CAT_UNMAPPED = sys.intern('unmapped')


# =============================================================================
# Pre-compiled Regex Patterns for Parsing Performance
# =============================================================================

_RE_SPICE_VALUE = re.compile(r'^([+-]?[\d.]+(?:e[+-]?\d+)?)\s*(\w*)$', re.IGNORECASE)
_RE_PULSE = re.compile(r'pulse\s*\(\s*([^)]+)\)', re.IGNORECASE)
_RE_PWL = re.compile(r'pwl\s*\(\s*([^)]+)\)', re.IGNORECASE)
_RE_COORD_EXTRACT = re.compile(r':(\d+)_(\d+):')
_RE_TILE_FILE = re.compile(r'tile_(\d+)_(\d+)\.(ckt|sp)')
_RE_INST_FILE = re.compile(r'instanceModels_(\d+)_(\d+)\.sp')


# =============================================================================
# SPICE Value Parsing
# =============================================================================

# SPICE unit multipliers (hoisted to module level for performance)
_SPICE_MULTIPLIERS: Dict[str, float] = {
    'f': 1e-15, 'p': 1e-12, 'n': 1e-9, 'u': 1e-6,
    'm': 1e-3, 'k': 1e3, 'meg': 1e6, 'g': 1e9, 't': 1e12
}


def _parse_spice_value(value_str: str) -> float:
    """
    Parse a numeric value with optional SPICE unit suffix.

    Supports: f (femto), p (pico), n (nano), u (micro), m (milli),
              k (kilo), meg (mega), g (giga), t (tera)
    """
    value_str = value_str.strip().lower()

    match = _RE_SPICE_VALUE.match(value_str)
    if match:
        num = float(match.group(1))
        unit = match.group(2)
        if unit in _SPICE_MULTIPLIERS:
            return num * _SPICE_MULTIPLIERS[unit]
        elif unit.startswith('meg'):
            return num * 1e6
        return num

    return float(value_str)


# =============================================================================
# Net Filter Helpers
# =============================================================================

def _check_net_filter(
    node1: str,
    node2: str,
    node_net_map_lower: Dict[str, str],
    net_filter: Optional[str],
) -> bool:
    """Check if element passes net filter.

    Returns True if element should be included, False if filtered out.
    """
    if net_filter is None:
        return True

    node1_net = node_net_map_lower.get(node1)
    node2_net = node_net_map_lower.get(node2)

    return node1_net == net_filter or node2_net == net_filter


def _fast_instance_net_filter(line: str, net_filter: str) -> bool:
    """Fast net filter using structured instance name embedded net info.

    Instance name format:
        i_<inst>:<vdd_net>:<vdd_pin>:<vss_net>:<vss_pin>:<tile_x>:<tile_y>[:<extra>]

    Extracts vdd_net (field 1) and vss_net (field 3) from the first token and
    compares against net_filter (lowercase). Avoids expensive full-line parsing
    for non-matching lines.

    Args:
        line: Raw SPICE line (must start with 'i'/'I')
        net_filter: Lowercase net name to match

    Returns:
        True if line matches net_filter (should be parsed), False to skip.
    """
    # Extract first token (instance name) without splitting entire line
    idx = line.find(' ')
    tab = line.find('\t')
    if tab >= 0 and (idx < 0 or tab < idx):
        idx = tab
    if idx < 0:
        return True  # Malformed, let full parser handle

    name = line[:idx]

    # Split name on ':' — skip 'i_' prefix
    start = 2 if (len(name) > 2 and name[1] == '_') else 0
    parts = name[start:].split(':')

    # Structured format has >= 7 fields: inst, vdd_net, vdd_pin, vss_net, vss_pin, tx, ty
    if len(parts) < 7:
        return True  # Not structured, include (caller falls back to .nd filtering)

    return parts[1].lower() == net_filter or parts[3].lower() == net_filter


def _has_structured_instance_names(filepath: str) -> bool:
    """Detect whether instanceModels file uses structured instance names.

    Reads lines until the first current source line (starts with 'i'/'I'),
    then checks if its name has >= 7 colon-delimited fields.

    Returns False for empty files or files with unstructured names.
    """
    open_fn = gzip.open if filepath.endswith('.gz') else open
    mode = 'rt' if filepath.endswith('.gz') else 'r'

    try:
        with open_fn(filepath, mode) as f:
            for line in f:
                line = line.strip()
                if not line or line[0] == '*' or line[0] == '.':
                    continue
                if line[0].lower() != 'i':
                    continue

                # Found first current source line — check name format
                idx = line.find(' ')
                tab = line.find('\t')
                if tab >= 0 and (idx < 0 or tab < idx):
                    idx = tab
                if idx < 0:
                    return False

                name = line[:idx]
                start = 2 if (len(name) > 2 and name[1] == '_') else 0
                parts = name[start:].split(':')
                return len(parts) >= 7
    except (OSError, IOError):
        return False

    return False


# =============================================================================
# SPICE File Reader
# =============================================================================

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

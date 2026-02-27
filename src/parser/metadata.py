"""
PDN graph metadata: node attributes, parse statistics, net type tables.

This module contains data structures for PDN node metadata and parse-time
statistics. These are standalone types with no project dependencies.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# =============================================================================
# Net Type Lookup Tables
# =============================================================================
# Module-level net type lookup tables for memory-efficient storage.
# Instead of storing net_type string per node (~50 bytes), we store a small
# int index (~8 bytes).

_NET_TYPE_TABLE: List[str] = []  # Index -> net name (e.g., ['VDD', 'VSS'])
_NET_TYPE_INDEX: Dict[str, int] = {}  # Net name -> index


def _get_net_type_index(net_type: Optional[str]) -> int:
    """Get or create index for net_type. Returns -1 for None."""
    if net_type is None:
        return -1
    if net_type not in _NET_TYPE_INDEX:
        _NET_TYPE_INDEX[net_type] = len(_NET_TYPE_TABLE)
        _NET_TYPE_TABLE.append(net_type)
    return _NET_TYPE_INDEX[net_type]


def _reset_net_type_tables() -> None:
    """Reset net type lookup tables (for testing)."""
    global _NET_TYPE_TABLE, _NET_TYPE_INDEX
    _NET_TYPE_TABLE = []
    _NET_TYPE_INDEX = {}


# =============================================================================
# PDN Node Attributes
# =============================================================================

# Bit flag constants (module-level for slots=True compatibility)
_FLAG_BOUNDARY: int = 1
_FLAG_PACKAGE: int = 2
_FLAG_VSRC: int = 4
_FLAG_DIE: int = 8   # Die node - can extract x, y, layer from name


@dataclass(slots=True)
class PDNNodeAttrs:
    """Ultra-compact node attributes for PDN netlist nodes.

    Memory optimization: x, y, layer are NOT stored - computed on-demand from name.
    Die nodes follow pattern X_Y_LAYER (e.g., '1000_2000_M1').

    Typical memory savings vs dict: ~57% (471 -> 199 bytes/node)

    Attributes:
        name: Node name (required for all nodes)
        _net_type_idx: Index into _NET_TYPE_TABLE (-1 = None)
        tile_id: Tile location as (x, y) tuple
        flags: Bit flags for boolean properties
        voltage: Voltage value (set during solve only)
    """
    name: str                          # Required - stored for all nodes
    _net_type_idx: int = -1            # Index into _NET_TYPE_TABLE (-1 = None)
    tile_id: Optional[Tuple[int, int]] = None
    flags: int = 0                     # Bit flags for booleans
    voltage: Optional[float] = None    # Set during solve only

    # Expose flag constants as class attributes for external access
    FLAG_BOUNDARY = _FLAG_BOUNDARY
    FLAG_PACKAGE = _FLAG_PACKAGE
    FLAG_VSRC = _FLAG_VSRC
    FLAG_DIE = _FLAG_DIE

    @property
    def net_type(self) -> Optional[str]:
        """Get net_type string from lookup table."""
        if self._net_type_idx < 0:
            return None
        return _NET_TYPE_TABLE[self._net_type_idx]

    @net_type.setter
    def net_type(self, value: Optional[str]) -> None:
        """Set net_type via lookup table index."""
        self._net_type_idx = _get_net_type_index(value)

    @property
    def is_boundary(self) -> bool:
        """Check if node is a boundary node."""
        return (self.flags & _FLAG_BOUNDARY) != 0

    @is_boundary.setter
    def is_boundary(self, value: bool) -> None:
        """Set boundary flag."""
        if value:
            self.flags |= _FLAG_BOUNDARY
        else:
            self.flags &= ~_FLAG_BOUNDARY

    @property
    def is_package(self) -> bool:
        """Check if node is a package node."""
        return (self.flags & _FLAG_PACKAGE) != 0

    @is_package.setter
    def is_package(self, value: bool) -> None:
        """Set package flag."""
        if value:
            self.flags |= _FLAG_PACKAGE
        else:
            self.flags &= ~_FLAG_PACKAGE

    @property
    def is_vsrc_node(self) -> bool:
        """Check if node is a voltage source node."""
        return (self.flags & _FLAG_VSRC) != 0

    @is_vsrc_node.setter
    def is_vsrc_node(self, value: bool) -> None:
        """Set voltage source node flag."""
        if value:
            self.flags |= _FLAG_VSRC
        else:
            self.flags &= ~_FLAG_VSRC

    @property
    def is_die(self) -> bool:
        """Check if node is a die node (x, y, layer extractable from name)."""
        return (self.flags & _FLAG_DIE) != 0

    @is_die.setter
    def is_die(self, value: bool) -> None:
        """Set die node flag."""
        if value:
            self.flags |= _FLAG_DIE
        else:
            self.flags &= ~_FLAG_DIE

    # Computed properties - extract from name on-demand (no storage cost)
    @property
    def x(self) -> Optional[int]:
        """Extract x coordinate from name for die nodes."""
        if self.flags & _FLAG_DIE:
            parts = self.name.split('_')
            if len(parts) >= 2:
                try:
                    return int(parts[0])
                except ValueError:
                    pass
        return None

    @property
    def y(self) -> Optional[int]:
        """Extract y coordinate from name for die nodes."""
        if self.flags & _FLAG_DIE:
            parts = self.name.split('_')
            if len(parts) >= 2:
                try:
                    return int(parts[1])
                except ValueError:
                    pass
        return None

    @property
    def layer(self) -> Optional[str]:
        """Extract layer from name for die nodes."""
        if self.flags & _FLAG_DIE:
            parts = self.name.split('_')
            if len(parts) >= 3:
                return parts[2]
        return None

    def get(self, key: str, default: any = None) -> any:
        """Dict-like get() for backward compatibility with code using dict access."""
        return getattr(self, key, default)

    def __contains__(self, key: str) -> bool:
        """Dict-like 'in' check for backward compatibility."""
        return hasattr(self, key)

    def __getitem__(self, key: str) -> any:
        """Dict-like [] access for backward compatibility."""
        return getattr(self, key)

    def __setitem__(self, key: str, value: any) -> None:
        """Dict-like [] assignment for backward compatibility.

        Note: Read-only properties (x, y, layer, name, is_die) raise AttributeError.
        """
        _readonly = {'x', 'y', 'layer', 'name', 'is_die'}
        if key in _readonly:
            raise AttributeError(f"'{key}' is a read-only computed property")
        setattr(self, key, value)

    def update(self, attrs: Dict[str, any]) -> None:
        """Dict-like update() for backward compatibility.

        Note: Read-only properties (x, y, layer, name, is_die) are silently ignored.
        """
        _readonly = {'x', 'y', 'layer', 'name', 'is_die'}
        for k, v in attrs.items():
            if k not in _readonly and hasattr(self, k) and not k.startswith('_'):
                setattr(self, k, v)

    def __reduce__(self):
        """Custom pickle support - ensures class is resolved from parser.metadata module."""
        return (PDNNodeAttrs, (self.name, self._net_type_idx, self.tile_id,
                               self.flags, self.voltage))


# =============================================================================
# Parse Statistics
# =============================================================================

@dataclass(slots=True)
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
    layer_stats_by_net: Dict[str, Dict] = field(default_factory=dict)
    unmapped_nodes: int = 0
    # Instance current statistics
    instances_with_waveforms: int = 0
    total_static_current_ma: float = 0.0

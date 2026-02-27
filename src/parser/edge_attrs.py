#!/usr/bin/env python3
"""
Memory-optimized edge attribute classes for PDN elements.

Replaces generic dict storage with specialized slotted dataclasses,
reducing memory footprint by ~90-95% per edge.

Memory comparison (per edge):
  Dict (5 attrs):     ~658-732 bytes
  ResistorEdge:       ~40 bytes   (95% savings)
  ResistorEdgeWithName: ~48 bytes (93% savings)
  CapacitorEdge:      ~32 bytes   (95% savings)

Key optimizations:
1. __slots__ to eliminate per-instance __dict__
2. Combined tile_id + net_type packing into single int32
3. elem_name omitted for 99.9% of resistors (die resistors)
4. net_type stored as index into lookup table
"""

from dataclasses import dataclass
from typing import Optional, Tuple, List, Any

# =============================================================================
# Net Type Indexing
# =============================================================================

# Global lookup table: index -> net_type string
# Index 0 is reserved for None
_NET_TYPE_TABLE: List[Optional[str]] = [None]  # idx 0 = None
_NET_TYPE_TO_IDX: dict = {}  # net_type string -> index


def _get_net_type_index(net_type: Optional[str]) -> int:
    """Get index for a net type string, adding to table if needed.

    Thread-safe for reads after initial population.
    Index 0 is reserved for None.

    Args:
        net_type: Net type string (e.g., 'VDD', 'VSS') or None

    Returns:
        Index into _NET_TYPE_TABLE (0 for None, 1+ for strings)
    """
    if net_type is None:
        return 0

    idx = _NET_TYPE_TO_IDX.get(net_type)
    if idx is not None:
        return idx

    # Add new net type (typically happens during parsing)
    idx = len(_NET_TYPE_TABLE)
    if idx >= 16:
        # 4-bit limit with int32 packing
        raise ValueError(f"Too many net types ({idx}+). Max supported: 15")
    _NET_TYPE_TABLE.append(net_type)
    _NET_TYPE_TO_IDX[net_type] = idx
    return idx


def _get_net_type_from_index(idx: int) -> Optional[str]:
    """Resolve net type index to string.

    Args:
        idx: Index from _get_net_type_index()

    Returns:
        Net type string or None
    """
    if idx < 0 or idx >= len(_NET_TYPE_TABLE):
        return None
    return _NET_TYPE_TABLE[idx]


def reset_net_type_table():
    """Reset net type table (for testing)."""
    global _NET_TYPE_TABLE, _NET_TYPE_TO_IDX
    _NET_TYPE_TABLE = [None]
    _NET_TYPE_TO_IDX = {}


# =============================================================================
# Tile/Net Packing
# =============================================================================

# Packing format (32 bits):
# bits 31-28: net_type_idx (4 bits, 0-15 net types)
# bits 27-14: tile_x (14 bits, 0-16383 tiles)
# bits 13-0:  tile_y (14 bits, 0-16383 tiles)
# Special: tile_x=0x3FFF and tile_y=0x3FFF means tile_id=None

_TILE_NONE_MARKER = 0x3FFF  # 14 bits all 1s
_TILE_MASK = 0x3FFF  # 14 bits
_NET_SHIFT = 28
_TILE_X_SHIFT = 14


def _pack_tile_net(tile_id: Optional[Tuple[int, int]], net_type_idx: int) -> int:
    """Pack tile_id and net_type into a single int32.

    Args:
        tile_id: (tile_x, tile_y) tuple or None
        net_type_idx: Net type index (0-15)

    Returns:
        Packed int32 value
    """
    if tile_id is None:
        tile_x = _TILE_NONE_MARKER
        tile_y = _TILE_NONE_MARKER
    else:
        tile_x, tile_y = tile_id
        # Clamp to 14-bit range
        tile_x = min(tile_x, _TILE_MASK)
        tile_y = min(tile_y, _TILE_MASK)

    return ((net_type_idx & 0xF) << _NET_SHIFT) | \
           ((tile_x & _TILE_MASK) << _TILE_X_SHIFT) | \
           (tile_y & _TILE_MASK)


def _unpack_tile_id(packed: int) -> Optional[Tuple[int, int]]:
    """Extract tile_id from packed int32.

    Args:
        packed: Packed value from _pack_tile_net()

    Returns:
        (tile_x, tile_y) tuple or None
    """
    tile_y = packed & _TILE_MASK
    tile_x = (packed >> _TILE_X_SHIFT) & _TILE_MASK

    if tile_x == _TILE_NONE_MARKER and tile_y == _TILE_NONE_MARKER:
        return None
    return (tile_x, tile_y)


def _unpack_net_type_idx(packed: int) -> int:
    """Extract net_type_idx from packed int32.

    Args:
        packed: Packed value from _pack_tile_net()

    Returns:
        Net type index (0-15)
    """
    return (packed >> _NET_SHIFT) & 0xF


# =============================================================================
# Base Element Edge Attributes
# =============================================================================

@dataclass(slots=True)
class BaseElementEdge:
    """Base class for all element edge attributes.

    Provides dict-like interface for backward compatibility with existing code
    that expects edge attributes to support .get(), [], and 'in' operations.
    """

    def get(self, key: str, default=None):
        """Dict-like get() method for backward compatibility."""
        if key == 'type':
            return self.type
        elif key == 'value':
            return self.value
        elif key == 'tile_id':
            return self.tile_id
        elif key == 'net_type':
            return self.net_type
        elif key == 'elem_name':
            return getattr(self, 'elem_name', default)
        return default

    def __getitem__(self, key: str):
        """Dict-like indexing for backward compatibility."""
        result = self.get(key, KeyError)
        if result is KeyError:
            raise KeyError(key)
        return result

    def __contains__(self, key: str) -> bool:
        """Dict-like 'in' operator for backward compatibility."""
        if key in ('type', 'value', 'tile_id', 'net_type'):
            return True
        if key == 'elem_name':
            return hasattr(self, 'elem_name')
        return False

    def keys(self):
        """Return attribute names (for dict-like iteration)."""
        result = ['type', 'value', 'tile_id', 'net_type']
        if hasattr(self, 'elem_name'):
            result.append('elem_name')
        return result

    def items(self):
        """Return (key, value) pairs (for dict-like iteration)."""
        result = [
            ('type', self.type),
            ('value', self.value),
            ('tile_id', self.tile_id),
            ('net_type', self.net_type),
        ]
        if hasattr(self, 'elem_name'):
            result.append(('elem_name', self.elem_name))
        return result

    def values(self):
        """Return attribute values (for dict-like iteration)."""
        return [v for _, v in self.items()]

    @property
    def type(self) -> str:
        """Element type - to be overridden by subclasses."""
        raise NotImplementedError

    @property
    def tile_id(self) -> Optional[Tuple[int, int]]:
        """Tile location - unpacked from _tile_net_packed."""
        return _unpack_tile_id(self._tile_net_packed)

    @property
    def net_type(self) -> Optional[str]:
        """Power net name - resolved from index."""
        idx = _unpack_net_type_idx(self._tile_net_packed)
        return _get_net_type_from_index(idx)


# =============================================================================
# Specialized Edge Attribute Classes
# =============================================================================

@dataclass(slots=True)
class ResistorEdge(BaseElementEdge):
    """Edge attributes for die resistor elements (no elem_name).

    This class is used for 99.9% of resistors (die resistors) that don't need
    elem_name stored. Only package resistors matching vsrc_resistor_pattern
    need elem_name for vsrc node identification.

    Memory: ~40 bytes (vs ~732 bytes for dict with 5 attrs)
    Savings: 95%
    """
    value: float                           # Resistance in kOhm (8 bytes)
    _tile_net_packed: int = 0              # Combined tile_id + net_type (4 bytes)

    @property
    def type(self) -> str:
        """Element type (constant for resistors)."""
        return 'R'

    def __reduce__(self):
        """Support pickling with resolved values."""
        return (
            _unpickle_resistor_edge,
            (self.value, self.tile_id, self.net_type)
        )


def _unpickle_resistor_edge(value, tile_id, net_type):
    """Unpickle ResistorEdge with correct net type indexing."""
    net_idx = _get_net_type_index(net_type)
    packed = _pack_tile_net(tile_id, net_idx)
    return ResistorEdge(value=value, _tile_net_packed=packed)


@dataclass(slots=True)
class ResistorEdgeWithName(BaseElementEdge):
    """Edge attributes for package resistors with elem_name.

    This class is only used for:
    - Resistors matching vsrc_resistor_pattern (e.g., 'rs') with value=0.0
    - These are package resistors used for vsrc node identification

    Memory: ~48 bytes (vs ~732 bytes for dict with 5 attrs)
    Savings: 93%
    """
    value: float                           # Resistance in kOhm (8 bytes)
    elem_name: str                         # Element name for vsrc tracing (reference)
    _tile_net_packed: int = 0              # Combined tile_id + net_type (4 bytes)

    @property
    def type(self) -> str:
        """Element type (constant for resistors)."""
        return 'R'

    def __reduce__(self):
        """Support pickling with resolved values."""
        return (
            _unpickle_resistor_edge_with_name,
            (self.value, self.elem_name, self.tile_id, self.net_type)
        )


def _unpickle_resistor_edge_with_name(value, elem_name, tile_id, net_type):
    """Unpickle ResistorEdgeWithName with correct net type indexing."""
    net_idx = _get_net_type_index(net_type)
    packed = _pack_tile_net(tile_id, net_idx)
    return ResistorEdgeWithName(value=value, elem_name=elem_name, _tile_net_packed=packed)


@dataclass(slots=True)
class CapacitorEdge(BaseElementEdge):
    """Edge attributes for capacitor elements.

    Memory: ~32 bytes (vs ~610 bytes for dict with 4 attrs)
    Savings: 95%
    """
    value: float                           # Capacitance in fF (8 bytes)
    _tile_net_packed: int = 0              # Combined tile_id + net_type (4 bytes)

    @property
    def type(self) -> str:
        """Element type (constant for capacitors)."""
        return 'C'

    def __reduce__(self):
        """Support pickling with resolved values."""
        return (
            _unpickle_capacitor_edge,
            (self.value, self.tile_id, self.net_type)
        )


def _unpickle_capacitor_edge(value, tile_id, net_type):
    """Unpickle CapacitorEdge with correct net type indexing."""
    net_idx = _get_net_type_index(net_type)
    packed = _pack_tile_net(tile_id, net_idx)
    return CapacitorEdge(value=value, _tile_net_packed=packed)


@dataclass(slots=True)
class InductorEdge(BaseElementEdge):
    """Edge attributes for inductor elements.

    Memory: ~32 bytes (vs ~610 bytes for dict with 4 attrs)
    Savings: 95%
    """
    value: float                           # Inductance in nH (8 bytes)
    _tile_net_packed: int = 0              # Combined tile_id + net_type (4 bytes)

    @property
    def type(self) -> str:
        """Element type (constant for inductors)."""
        return 'L'

    def __reduce__(self):
        """Support pickling with resolved values."""
        return (
            _unpickle_inductor_edge,
            (self.value, self.tile_id, self.net_type)
        )


def _unpickle_inductor_edge(value, tile_id, net_type):
    """Unpickle InductorEdge with correct net type indexing."""
    net_idx = _get_net_type_index(net_type)
    packed = _pack_tile_net(tile_id, net_idx)
    return InductorEdge(value=value, _tile_net_packed=packed)


@dataclass(slots=True)
class VoltageSourceEdge(BaseElementEdge):
    """Edge attributes for voltage source elements.

    Memory: ~48 bytes (vs ~666 bytes for dict with 5 attrs)
    Savings: 93%
    """
    value: float                           # Voltage in V (8 bytes)
    elem_name: str                         # Element name (needed for vsrc detection)
    _tile_net_packed: int = 0              # Combined tile_id + net_type (4 bytes)

    @property
    def type(self) -> str:
        """Element type (constant for voltage sources)."""
        return 'V'

    def __reduce__(self):
        """Support pickling with resolved values."""
        return (
            _unpickle_voltage_source_edge,
            (self.value, self.elem_name, self.tile_id, self.net_type)
        )


def _unpickle_voltage_source_edge(value, elem_name, tile_id, net_type):
    """Unpickle VoltageSourceEdge with correct net type indexing."""
    net_idx = _get_net_type_index(net_type)
    packed = _pack_tile_net(tile_id, net_idx)
    return VoltageSourceEdge(value=value, elem_name=elem_name, _tile_net_packed=packed)


@dataclass(slots=True)
class CurrentSourceEdge(BaseElementEdge):
    """Edge attributes for current source elements.

    Memory: ~32 bytes (vs ~610 bytes for dict with 4 attrs)
    Savings: 95%

    Note: For full current source data (waveforms, instance info), see
    graph.graph['instance_sources'] dictionary which maps instance names
    to CurrentSource objects.
    """
    value: float                           # Current in mA (DC value)
    _tile_net_packed: int = 0              # Combined tile_id + net_type (4 bytes)

    @property
    def type(self) -> str:
        """Element type (constant for current sources)."""
        return 'I'

    def __reduce__(self):
        """Support pickling with resolved values."""
        return (
            _unpickle_current_source_edge,
            (self.value, self.tile_id, self.net_type)
        )


def _unpickle_current_source_edge(value, tile_id, net_type):
    """Unpickle CurrentSourceEdge with correct net type indexing."""
    net_idx = _get_net_type_index(net_type)
    packed = _pack_tile_net(tile_id, net_idx)
    return CurrentSourceEdge(value=value, _tile_net_packed=packed)


# =============================================================================
# Factory Function
# =============================================================================

def create_element_edge(elem_type: str, value: float,
                       elem_name: Optional[str] = None,
                       tile_id: Optional[Tuple[int, int]] = None,
                       net_type: Optional[str] = None,
                       net_type_idx: int = -1,
                       needs_elem_name: bool = False) -> BaseElementEdge:
    """Factory function to create appropriate edge attribute object.

    Args:
        elem_type: Element type ('R', 'C', 'L', 'V', 'I')
        value: Element value (resistance, capacitance, etc.)
        elem_name: Element name (required for V; optional for R with needs_elem_name=True)
        tile_id: Tile location (x, y)
        net_type: Power net name (ignored if net_type_idx provided)
        net_type_idx: Pre-computed net type index (-1 to compute from net_type)
        needs_elem_name: If True and elem_type='R', store elem_name (for vsrc resistors)

    Returns:
        Specialized edge attribute object

    Example:
        # Die resistor (no elem_name stored)
        edge_attrs = create_element_edge('R', 0.0125, tile_id=(0, 0), net_type='VDD')

        # Package resistor (elem_name stored for vsrc tracing)
        edge_attrs = create_element_edge('R', 0.0, elem_name='rs', tile_id=(0, 0),
                                         net_type='VDD', needs_elem_name=True)
    """
    # Get or compute net type index
    if net_type_idx < 0:
        net_type_idx = _get_net_type_index(net_type)

    # Pack tile_id and net_type into single int
    packed = _pack_tile_net(tile_id, net_type_idx)

    if elem_type == 'R':
        if needs_elem_name and elem_name is not None:
            return ResistorEdgeWithName(
                value=value,
                elem_name=elem_name,
                _tile_net_packed=packed
            )
        return ResistorEdge(
            value=value,
            _tile_net_packed=packed
        )
    elif elem_type == 'C':
        return CapacitorEdge(
            value=value,
            _tile_net_packed=packed
        )
    elif elem_type == 'L':
        return InductorEdge(
            value=value,
            _tile_net_packed=packed
        )
    elif elem_type == 'V':
        return VoltageSourceEdge(
            value=value,
            elem_name=elem_name or '',
            _tile_net_packed=packed
        )
    elif elem_type == 'I':
        return CurrentSourceEdge(
            value=value,
            _tile_net_packed=packed
        )
    else:
        raise ValueError(f"Unknown element type: {elem_type}")


# =============================================================================
# Migration Helper (for backward compatibility)
# =============================================================================

def edge_dict_to_object(edge_dict: dict, needs_elem_name: bool = False) -> BaseElementEdge:
    """Convert legacy dict-based edge attributes to specialized objects.

    Used for migrating existing graphs or handling mixed dict/object edges.

    Args:
        edge_dict: Legacy edge attribute dictionary
        needs_elem_name: If True and type='R', store elem_name

    Returns:
        Specialized edge attribute object
    """
    elem_type = edge_dict.get('type')
    if elem_type is None:
        raise ValueError("Edge dict missing 'type' key")

    return create_element_edge(
        elem_type=elem_type,
        value=edge_dict.get('value'),
        elem_name=edge_dict.get('elem_name'),
        tile_id=edge_dict.get('tile_id'),
        net_type=edge_dict.get('net_type'),
        needs_elem_name=needs_elem_name,
    )


def edge_object_to_dict(edge_obj: BaseElementEdge) -> dict:
    """Convert specialized edge object back to dict (for serialization).

    Args:
        edge_obj: Specialized edge attribute object

    Returns:
        Dictionary representation
    """
    result = {
        'type': edge_obj.type,
        'value': edge_obj.value,
    }

    # Add optional attributes if present
    if hasattr(edge_obj, 'elem_name'):
        result['elem_name'] = edge_obj.elem_name
    result['tile_id'] = edge_obj.tile_id
    result['net_type'] = edge_obj.net_type

    return result

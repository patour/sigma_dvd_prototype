"""Edge adapter for unified power grid model.

Extracts unified edge information from various edge representations:
- Synthetic grids: Resistance-only edges with 'resistance' attribute
- PDN netlists: Multi-type edges (R, C, L, V, I) with 'type' and 'value' attributes
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class ElementType(Enum):
    """Circuit element types supported in power grids."""
    RESISTOR = 'R'
    CAPACITOR = 'C'
    INDUCTOR = 'L'
    VOLTAGE_SOURCE = 'V'
    CURRENT_SOURCE = 'I'
    # Controlled sources (for future use)
    VCVS = 'E'  # Voltage-controlled voltage source
    VCCS = 'G'  # Voltage-controlled current source
    CCCS = 'F'  # Current-controlled current source
    CCVS = 'H'  # Current-controlled voltage source


@dataclass
class UnifiedEdgeInfo:
    """Unified edge information for power grid elements.

    Attributes:
        element_type: Type of circuit element
        resistance: Resistance in Ohms (for resistors)
        capacitance: Capacitance in Farads (for capacitors)
        inductance: Inductance in Henrys (for inductors)
        voltage: Voltage in Volts (for voltage sources)
        current: Current in Amps (for current sources)
        name: Element name/identifier
        net_type: Power net type (e.g., 'VDD', 'VSS')
        original_data: Original edge attribute dictionary
    """
    element_type: ElementType
    resistance: Optional[float] = None
    capacitance: Optional[float] = None
    inductance: Optional[float] = None
    voltage: Optional[float] = None
    current: Optional[float] = None
    name: Optional[str] = None
    net_type: Optional[str] = None
    original_data: Dict[str, Any] = field(default_factory=dict)

    @property
    def conductance(self) -> Optional[float]:
        """Return conductance (1/R) if resistance is defined and non-zero."""
        if self.resistance is not None and self.resistance > 0:
            return 1.0 / self.resistance
        return None

    @property
    def is_resistive(self) -> bool:
        """Check if this is a resistive element."""
        return self.element_type == ElementType.RESISTOR

    @property
    def is_source(self) -> bool:
        """Check if this is a source element (V or I)."""
        return self.element_type in (ElementType.VOLTAGE_SOURCE, ElementType.CURRENT_SOURCE)


class EdgeInfoExtractor:
    """Extracts unified edge info from various graph edge representations.

    Supports:
    - Synthetic grids: Resistance-only edges with 'resistance' attribute
    - PDN netlists: Multi-type edges with 'type' and 'value' attributes

    Handles unit conversions between different representations.
    """

    # Element type mapping from string to enum
    TYPE_MAP = {
        'R': ElementType.RESISTOR,
        'C': ElementType.CAPACITOR,
        'L': ElementType.INDUCTOR,
        'V': ElementType.VOLTAGE_SOURCE,
        'I': ElementType.CURRENT_SOURCE,
        'E': ElementType.VCVS,
        'G': ElementType.VCCS,
        'F': ElementType.CCCS,
        'H': ElementType.CCVS,
    }

    def __init__(
        self,
        is_pdn: bool = False,
        resistance_unit_kohm: bool = False,
        capacitance_unit_ff: bool = False,
        inductance_unit_nh: bool = False,
        current_unit_ma: bool = False,
    ):
        """Initialize edge extractor.

        Args:
            is_pdn: True if source is PDN netlist (has 'type' attribute)
            resistance_unit_kohm: True if resistance values are in kOhms (converts to Ohms)
            capacitance_unit_ff: True if capacitance values are in fF (converts to F)
            inductance_unit_nh: True if inductance values are in nH (converts to H)
            current_unit_ma: True if current values are in mA (converts to A)
        """
        self.is_pdn = is_pdn
        self.resistance_unit_kohm = resistance_unit_kohm
        self.capacitance_unit_ff = capacitance_unit_ff
        self.inductance_unit_nh = inductance_unit_nh
        self.current_unit_ma = current_unit_ma

    def get_info(self, edge_data: Dict[str, Any]) -> UnifiedEdgeInfo:
        """Extract unified edge info from edge attribute dict.

        Args:
            edge_data: Edge attributes from graph.edges(data=True)

        Returns:
            UnifiedEdgeInfo with normalized values.
        """
        if self.is_pdn:
            return self._extract_pdn_edge(edge_data)
        else:
            return self._extract_synthetic_edge(edge_data)

    def _extract_synthetic_edge(self, data: Dict) -> UnifiedEdgeInfo:
        """Extract from synthetic grid edge (resistance-only).

        Synthetic edges have 'resistance' attribute directly.
        """
        resistance = data.get('resistance', 0.0)
        if resistance is not None:
            resistance = float(resistance)

        return UnifiedEdgeInfo(
            element_type=ElementType.RESISTOR,
            resistance=resistance,
            name=data.get('kind'),
            original_data=data,
        )

    def _extract_pdn_edge(self, data: Dict) -> UnifiedEdgeInfo:
        """Extract from PDN netlist edge (multi-type).

        PDN edges have 'type' attribute indicating element type,
        and 'value' attribute with the element value.
        """
        elem_type_str = data.get('type', 'R')
        value = data.get('value', 0.0)
        if value is not None:
            value = float(value)

        # Map string type to enum
        elem_type = self.TYPE_MAP.get(elem_type_str, ElementType.RESISTOR)

        info = UnifiedEdgeInfo(
            element_type=elem_type,
            name=data.get('elem_name'),
            net_type=data.get('net_type'),
            original_data=data,
        )

        # Assign value to appropriate field based on type, with unit conversion
        if elem_type == ElementType.RESISTOR:
            # PDN uses kOhms by default
            if self.resistance_unit_kohm and value is not None:
                info.resistance = value * 1e3  # kOhm -> Ohm
            else:
                info.resistance = value

        elif elem_type == ElementType.CAPACITOR:
            # PDN uses fF by default
            if self.capacitance_unit_ff and value is not None:
                info.capacitance = value * 1e-15  # fF -> F
            else:
                info.capacitance = value

        elif elem_type == ElementType.INDUCTOR:
            # PDN uses nH by default
            if self.inductance_unit_nh and value is not None:
                info.inductance = value * 1e-9  # nH -> H
            else:
                info.inductance = value

        elif elem_type == ElementType.VOLTAGE_SOURCE:
            info.voltage = value  # Volts

        elif elem_type == ElementType.CURRENT_SOURCE:
            # PDN uses mA by default
            if self.current_unit_ma and value is not None:
                info.current = value * 1e-3  # mA -> A
            else:
                info.current = value

        return info

    def is_resistive_edge(self, edge_data: Dict[str, Any]) -> bool:
        """Check if edge is a resistive element.

        Args:
            edge_data: Edge attributes

        Returns:
            True if edge is a resistor.
        """
        if self.is_pdn:
            return edge_data.get('type', 'R') == 'R'
        else:
            # Synthetic edges are always resistive
            return 'resistance' in edge_data

    def get_resistance(self, edge_data: Dict[str, Any]) -> Optional[float]:
        """Get resistance value from edge data (with unit conversion).

        Args:
            edge_data: Edge attributes

        Returns:
            Resistance in Ohms, or None if not a resistor.
        """
        if self.is_pdn:
            if edge_data.get('type') != 'R':
                return None
            value = edge_data.get('value', 0.0)
            if self.resistance_unit_kohm and value is not None:
                return float(value) * 1e3
            return float(value) if value is not None else None
        else:
            return float(edge_data.get('resistance', 0.0))

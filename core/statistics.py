"""Unified statistics for power grid analysis.

Provides netlist statistics computation for both synthetic and PDN grids.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from .unified_model import UnifiedPowerGridModel, LayerID
from .edge_adapter import ElementType


@dataclass
class LayerStats:
    """Statistics for a single layer.

    Attributes:
        layer: Layer identifier
        node_count: Number of nodes at this layer
        resistor_count: Number of resistive edges
        total_resistance: Sum of resistance values (Ohms)
        capacitor_count: Number of capacitors (PDN only)
        total_capacitance: Sum of capacitance values (Farads)
        inductor_count: Number of inductors (PDN only)
        total_inductance: Sum of inductance values (Henrys)
        current_source_count: Number of current sources
        total_current: Sum of current source values (Amps)
    """
    layer: LayerID
    node_count: int = 0
    resistor_count: int = 0
    total_resistance: float = 0.0
    capacitor_count: int = 0
    total_capacitance: float = 0.0
    inductor_count: int = 0
    total_inductance: float = 0.0
    current_source_count: int = 0
    total_current: float = 0.0


@dataclass
class GridStats:
    """Overall grid statistics.

    Attributes:
        total_nodes: Total number of nodes
        total_edges: Total number of edges
        pad_nodes: Number of voltage source nodes
        load_nodes: Number of load/current sink nodes
        resistor_count: Total resistors
        total_resistance: Sum of all resistance values
        capacitor_count: Total capacitors
        total_capacitance: Sum of all capacitance values
        inductor_count: Total inductors
        total_inductance: Sum of all inductance values
        current_source_count: Total current sources
        total_current: Sum of all current source values
        layers: List of layer identifiers
        layer_stats: Per-layer statistics
        net_name: Optional net name (for multi-net PDN)
    """
    total_nodes: int = 0
    total_edges: int = 0
    pad_nodes: int = 0
    load_nodes: int = 0
    resistor_count: int = 0
    total_resistance: float = 0.0
    capacitor_count: int = 0
    total_capacitance: float = 0.0
    inductor_count: int = 0
    total_inductance: float = 0.0
    current_source_count: int = 0
    total_current: float = 0.0
    layers: List[LayerID] = field(default_factory=list)
    layer_stats: Dict[LayerID, LayerStats] = field(default_factory=dict)
    net_name: Optional[str] = None


class UnifiedStatistics:
    """Compute statistics for unified power grid models.

    Example:
        model = create_model_from_synthetic(G, pads, vdd=1.0)
        stats_calc = UnifiedStatistics(model)
        stats = stats_calc.compute()

        print(f"Nodes: {stats.total_nodes}")
        print(f"Resistors: {stats.resistor_count}, Total R: {stats.total_resistance}")
    """

    def __init__(
        self,
        model: UnifiedPowerGridModel,
        load_nodes: Optional[Dict[Any, float]] = None,
    ):
        """Initialize statistics calculator.

        Args:
            model: UnifiedPowerGridModel instance
            load_nodes: Optional dict of load nodes and their currents
        """
        self.model = model
        self.load_nodes = load_nodes or {}

    def compute(self) -> GridStats:
        """Compute all statistics for the grid.

        Returns:
            GridStats object with all statistics
        """
        stats = GridStats(net_name=self.model.net_name)

        # Basic counts
        stats.total_nodes = self.model.graph.number_of_nodes()
        stats.total_edges = self.model.graph.number_of_edges()
        stats.pad_nodes = len(self.model.pad_nodes)
        stats.load_nodes = len(self.load_nodes)

        # Get layers
        stats.layers = self.model.get_all_layers()

        # Initialize per-layer stats
        for layer in stats.layers:
            stats.layer_stats[layer] = LayerStats(layer=layer)

        # Count nodes per layer
        for node in self.model.graph.nodes():
            info = self.model.get_node_info(node)
            if info.layer in stats.layer_stats:
                stats.layer_stats[info.layer].node_count += 1

        # Process edges
        for u, v, edge_info in self.model._iter_resistive_edges():
            # Get layer from either endpoint
            u_info = self.model.get_node_info(u)
            layer = u_info.layer

            # Global stats
            stats.resistor_count += 1
            if edge_info.resistance:
                stats.total_resistance += edge_info.resistance

            # Per-layer stats
            if layer in stats.layer_stats:
                stats.layer_stats[layer].resistor_count += 1
                if edge_info.resistance:
                    stats.layer_stats[layer].total_resistance += edge_info.resistance

        # Process non-resistive edges (PDN only)
        if hasattr(self.model.graph, 'edges'):
            import networkx as nx
            if isinstance(self.model.graph, nx.MultiDiGraph):
                for u, v, k, data in self.model.graph.edges(keys=True, data=True):
                    edge_info = self.model._edge_extractor.get_info(data)

                    u_info = self.model.get_node_info(u)
                    layer = u_info.layer

                    if edge_info.element_type == ElementType.CAPACITOR:
                        stats.capacitor_count += 1
                        if edge_info.capacitance:
                            stats.total_capacitance += edge_info.capacitance
                        if layer in stats.layer_stats:
                            stats.layer_stats[layer].capacitor_count += 1
                            if edge_info.capacitance:
                                stats.layer_stats[layer].total_capacitance += edge_info.capacitance

                    elif edge_info.element_type == ElementType.INDUCTOR:
                        stats.inductor_count += 1
                        if edge_info.inductance:
                            stats.total_inductance += edge_info.inductance
                        if layer in stats.layer_stats:
                            stats.layer_stats[layer].inductor_count += 1
                            if edge_info.inductance:
                                stats.layer_stats[layer].total_inductance += edge_info.inductance

                    elif edge_info.element_type == ElementType.CURRENT_SOURCE:
                        stats.current_source_count += 1
                        if edge_info.current:
                            stats.total_current += edge_info.current
                        if layer in stats.layer_stats:
                            stats.layer_stats[layer].current_source_count += 1
                            if edge_info.current:
                                stats.layer_stats[layer].total_current += edge_info.current

        return stats

    def format_summary(self, stats: Optional[GridStats] = None) -> str:
        """Format statistics as a summary string.

        Args:
            stats: GridStats object (computed if None)

        Returns:
            Formatted summary string
        """
        if stats is None:
            stats = self.compute()

        lines = []
        lines.append("=" * 60)
        if stats.net_name:
            lines.append(f"Grid Statistics - Net: {stats.net_name}")
        else:
            lines.append("Grid Statistics")
        lines.append("=" * 60)

        lines.append(f"Nodes: {stats.total_nodes}")
        lines.append(f"  Pad nodes (Vdd): {stats.pad_nodes}")
        lines.append(f"  Load nodes: {stats.load_nodes}")

        lines.append(f"\nEdges: {stats.total_edges}")
        lines.append(f"  Resistors: {stats.resistor_count}")
        if stats.total_resistance > 0:
            avg_r = stats.total_resistance / stats.resistor_count if stats.resistor_count > 0 else 0
            lines.append(f"    Total: {stats.total_resistance:.6f} Ohms")
            lines.append(f"    Average: {avg_r:.6f} Ohms")

        if stats.capacitor_count > 0:
            avg_c = stats.total_capacitance / stats.capacitor_count
            lines.append(f"  Capacitors: {stats.capacitor_count}")
            lines.append(f"    Total: {stats.total_capacitance * 1e15:.3f} fF")
            lines.append(f"    Average: {avg_c * 1e15:.3f} fF")

        if stats.inductor_count > 0:
            avg_l = stats.total_inductance / stats.inductor_count
            lines.append(f"  Inductors: {stats.inductor_count}")
            lines.append(f"    Total: {stats.total_inductance * 1e9:.3f} nH")
            lines.append(f"    Average: {avg_l * 1e9:.3f} nH")

        if stats.current_source_count > 0:
            lines.append(f"  Current Sources: {stats.current_source_count}")
            lines.append(f"    Total: {stats.total_current * 1e3:.3f} mA")

        if stats.layers:
            lines.append(f"\nLayers: {len(stats.layers)}")
            for layer in stats.layers:
                ls = stats.layer_stats.get(layer)
                if ls:
                    lines.append(f"  Layer {layer}: {ls.node_count} nodes, {ls.resistor_count} resistors")

        lines.append("=" * 60)

        return "\n".join(lines)

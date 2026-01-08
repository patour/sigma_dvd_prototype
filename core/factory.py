"""Factory functions for creating unified power grid models.

Provides convenient functions to create UnifiedPowerGridModel instances
from different sources (synthetic grids, PDN netlists).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union

import networkx as nx

from .unified_model import UnifiedPowerGridModel, GridSource


def create_model_from_synthetic(
    graph: nx.Graph,
    pad_nodes: Sequence[Any],
    vdd: float = 1.0,
) -> UnifiedPowerGridModel:
    """Create unified model from synthetic power grid.

    Args:
        graph: Graph from generate_power_grid()
        pad_nodes: List of pad NodeID objects (voltage sources)
        vdd: Supply voltage

    Returns:
        UnifiedPowerGridModel configured for synthetic source.

    Example:
        G, loads, pads = generate_power_grid(K=3, N0=12, ...)
        model = create_model_from_synthetic(G, pads, vdd=1.0)
    """
    return UnifiedPowerGridModel(
        graph=graph,
        pad_nodes=list(pad_nodes),
        vdd=vdd,
        source=GridSource.SYNTHETIC,
        resistance_unit_kohm=False,  # Synthetic uses Ohms
    )


def create_model_from_pdn(
    graph: nx.MultiDiGraph,
    net_name: str,
    vdd: Optional[float] = None,
    vsrc_nodes: Optional[Sequence[Any]] = None,
) -> UnifiedPowerGridModel:
    """Create unified model from PDN netlist for a specific net.

    Args:
        graph: Graph from NetlistParser.parse()
        net_name: Power net to model (e.g., 'VDD', 'VSS')
        vdd: Supply voltage (auto-detected from graph if None)
        vsrc_nodes: Voltage source nodes (auto-detected if None)

    Returns:
        UnifiedPowerGridModel configured for PDN source.

    Raises:
        ValueError: If net_name not found in graph.

    Example:
        parser = NetlistParser('./netlist_dir')
        graph = parser.parse()
        model = create_model_from_pdn(graph, 'VDD', vdd=1.0)
    """
    # Get net connectivity
    net_connectivity = graph.graph.get('net_connectivity', {})

    # Case-insensitive net lookup
    net_key = None
    for key in net_connectivity.keys():
        if key.lower() == net_name.lower():
            net_key = key
            break

    if net_key is None:
        available = list(net_connectivity.keys())
        raise ValueError(f"Net '{net_name}' not found. Available: {available}")

    net_nodes = set(net_connectivity[net_key])

    # Extract subgraph for this net
    net_subgraph = graph.subgraph(net_nodes).copy()

    # Find voltage source nodes
    if vsrc_nodes is not None:
        pad_nodes = list(vsrc_nodes)
    else:
        # Auto-detect from graph metadata
        all_vsrc_nodes = graph.graph.get('vsrc_nodes', set())
        pad_nodes = [n for n in net_nodes if n in all_vsrc_nodes]

        # If no vsrc_nodes found, look for V-type edges
        if not pad_nodes:
            for u, v, data in graph.edges(data=True):
                if data.get('type') == 'V' and (u in net_nodes or v in net_nodes):
                    if u in net_nodes and u != '0':
                        pad_nodes.append(u)
                    if v in net_nodes and v != '0':
                        pad_nodes.append(v)
            pad_nodes = list(set(pad_nodes))

    # Determine VDD
    if vdd is None:
        # Try to get from parameters
        params = graph.graph.get('parameters', {})
        # Look for net-specific voltage parameter
        vdd_key = net_name.upper()
        if vdd_key in params:
            vdd = float(params[vdd_key])
        else:
            # Default based on net name
            if 'vdd' in net_name.lower():
                vdd = 1.0
            elif 'vss' in net_name.lower():
                vdd = 0.0
            else:
                vdd = 1.0

    return UnifiedPowerGridModel(
        graph=net_subgraph,
        pad_nodes=pad_nodes,
        vdd=vdd,
        source=GridSource.PDN_NETLIST,
        net_name=net_name,
        resistance_unit_kohm=True,  # PDN uses kOhms
    )


def create_multi_net_models(
    graph: nx.MultiDiGraph,
    net_filter: Optional[List[str]] = None,
    vdd_map: Optional[Dict[str, float]] = None,
) -> Dict[str, UnifiedPowerGridModel]:
    """Create unified models for all (or filtered) nets in PDN.

    Args:
        graph: Graph from NetlistParser.parse()
        net_filter: Optional list of net names to include (case-insensitive)
        vdd_map: Optional dict mapping net_name -> vdd voltage

    Returns:
        Dict mapping net_name -> UnifiedPowerGridModel

    Example:
        parser = NetlistParser('./netlist_dir')
        graph = parser.parse()
        models = create_multi_net_models(graph, net_filter=['VDD', 'VSS'])

        for net_name, model in models.items():
            voltages = model.solve_voltages(currents)
    """
    net_connectivity = graph.graph.get('net_connectivity', {})

    # Determine which nets to process
    if net_filter is not None:
        # Case-insensitive filter
        filter_lower = {n.lower() for n in net_filter}
        nets_to_process = [n for n in net_connectivity.keys() if n.lower() in filter_lower]
    else:
        nets_to_process = list(net_connectivity.keys())

    models = {}
    vdd_map = vdd_map or {}

    for net_name in nets_to_process:
        try:
            vdd = vdd_map.get(net_name)
            models[net_name] = create_model_from_pdn(graph, net_name, vdd=vdd)
        except ValueError as e:
            # Skip nets that can't be modeled (e.g., no nodes)
            import warnings
            warnings.warn(f"Skipping net '{net_name}': {e}")

    return models


def create_model_from_graph(
    graph: Union[nx.Graph, nx.MultiDiGraph],
    pad_nodes: Sequence[Any],
    vdd: float = 1.0,
    auto_detect_source: bool = True,
) -> UnifiedPowerGridModel:
    """Create unified model with automatic source detection.

    Args:
        graph: NetworkX graph
        pad_nodes: Voltage source nodes
        vdd: Supply voltage
        auto_detect_source: If True, detect source type from graph structure

    Returns:
        UnifiedPowerGridModel with appropriate configuration.
    """
    if auto_detect_source:
        # Detect source based on graph type and edge attributes
        if isinstance(graph, nx.MultiDiGraph):
            source = GridSource.PDN_NETLIST
            resistance_unit_kohm = True
        else:
            # Check if edges have 'type' attribute (PDN) or 'resistance' (synthetic)
            sample_edge = next(iter(graph.edges(data=True)), None)
            if sample_edge and 'type' in sample_edge[2]:
                source = GridSource.PDN_NETLIST
                resistance_unit_kohm = True
            else:
                source = GridSource.SYNTHETIC
                resistance_unit_kohm = False
    else:
        source = GridSource.SYNTHETIC
        resistance_unit_kohm = False

    return UnifiedPowerGridModel(
        graph=graph,
        pad_nodes=list(pad_nodes),
        vdd=vdd,
        source=source,
        resistance_unit_kohm=resistance_unit_kohm,
    )

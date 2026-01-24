"""Factory functions for creating unified power grid models.

Provides convenient functions to create UnifiedPowerGridModel instances
from different sources (synthetic grids, PDN netlists).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union

from .unified_model import UnifiedPowerGridModel, GridSource
from .rx_graph import RustworkxGraphWrapper, RustworkxMultiDiGraphWrapper


def create_model_from_synthetic(
    graph: RustworkxGraphWrapper,
    pad_nodes: Sequence[Any],
    vdd: float = 1.0,
    lazy_factor: bool = True,
) -> UnifiedPowerGridModel:
    """Create unified model from synthetic power grid.

    Args:
        graph: Graph from generate_power_grid()
        pad_nodes: List of pad NodeID objects (voltage sources)
        vdd: Supply voltage
        lazy_factor: If True (default), defer LU factorization until first flat solve.
                    Set to False for backward compatibility if you need eager factorization.

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
        lazy_factor=lazy_factor,
    )


def create_model_from_pdn(
    graph: RustworkxMultiDiGraphWrapper,
    net_name: str,
    vsrc_nodes: Optional[Sequence[Any]] = None,
    lazy_factor: bool = True,
) -> UnifiedPowerGridModel:
    """Create unified model from PDN netlist for a specific net.

    The nominal voltage is automatically extracted from the parsed graph,
    either from the 'parameters' dict (populated from pg_net_voltage file)
    or from voltage source edges in the graph.

    Args:
        graph: Graph from NetlistParser.parse()
        net_name: Power net to model (e.g., 'VDD', 'VSS')
        vsrc_nodes: Voltage source nodes (auto-detected if None)
        lazy_factor: If True (default), defer LU factorization until first flat solve.
                    This significantly improves model creation time when using
                    hierarchical solvers. Set to False for backward compatibility.

    Returns:
        UnifiedPowerGridModel configured for PDN source.

    Raises:
        ValueError: If net_name not found in graph or nominal voltage cannot be determined.

    Example:
        parser = NetlistParser('./netlist_dir')
        graph = parser.parse()
        model = create_model_from_pdn(graph, 'VDD')
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

    # Include ground node '0' to preserve current source edges
    # Current sources connect net nodes to ground for load modeling
    nodes_for_subgraph = net_nodes.copy()
    if '0' in graph:
        nodes_for_subgraph.add('0')

    # Extract subgraph for this net (including ground)
    # Note: subgraph() already creates an independent copy, no need for .copy()
    net_subgraph = graph.subgraph(nodes_for_subgraph)

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

    # Extract nominal voltage from graph (required)
    vdd = None
    
    # Try to get from parameters (parsed from pg_net_voltage file)
    params = graph.graph.get('parameters', {})
    vdd_key = net_name.upper()
    if vdd_key in params:
        try:
            vdd = float(params[vdd_key])
        except (ValueError, TypeError):
            pass
    
    # Fallback: check voltage source edges for this net
    if vdd is None:
        for u, v, data in graph.edges(data=True):
            if data.get('type') == 'V':
                edge_net = data.get('net', '').upper()
                if edge_net == vdd_key or u in net_nodes or v in net_nodes:
                    vsrc_voltage = data.get('value')
                    if vsrc_voltage is not None and vsrc_voltage > 0:
                        vdd = float(vsrc_voltage)
                        break
    
    # Special case for ground nets
    if vdd is None and 'vss' in net_name.lower():
        vdd = 0.0
    
    # Error if voltage not found
    if vdd is None:
        raise ValueError(
            f"Could not determine nominal voltage for net '{net_name}'. "
            f"Ensure pg_net_voltage file contains '{vdd_key} <voltage>' or "
            f"voltage source edges exist in the netlist."
        )

    return UnifiedPowerGridModel(
        graph=net_subgraph,
        pad_nodes=pad_nodes,
        vdd=vdd,
        source=GridSource.PDN_NETLIST,
        net_name=net_name,
        resistance_unit_kohm=False,  # Keep PDN native kOhm/mA units (self-consistent mS conductance)
        lazy_factor=lazy_factor,
    )


def create_multi_net_models(
    graph: RustworkxMultiDiGraphWrapper,
    net_filter: Optional[List[str]] = None,
    lazy_factor: bool = True,
) -> Dict[str, UnifiedPowerGridModel]:
    """Create unified models for all (or filtered) nets in PDN.

    Nominal voltages are automatically extracted from the graph for each net.

    Args:
        graph: Graph from NetlistParser.parse()
        net_filter: Optional list of net names to include (case-insensitive)
        lazy_factor: If True (default), defer LU factorization until first flat solve.

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

    for net_name in nets_to_process:
        try:
            models[net_name] = create_model_from_pdn(graph, net_name, lazy_factor=lazy_factor)
        except ValueError as e:
            # Skip nets that can't be modeled (e.g., no nodes, no voltage spec)
            import warnings
            warnings.warn(f"Skipping net '{net_name}': {e}")

    return models


def create_model_from_graph(
    graph: Union[RustworkxGraphWrapper, RustworkxMultiDiGraphWrapper],
    pad_nodes: Sequence[Any],
    vdd: float = 1.0,
    auto_detect_source: bool = True,
    lazy_factor: bool = True,
) -> UnifiedPowerGridModel:
    """Create unified model with automatic source detection.

    Args:
        graph: Rustworkx graph wrapper
        pad_nodes: Voltage source nodes
        vdd: Supply voltage
        auto_detect_source: If True, detect source type from graph structure
        lazy_factor: If True (default), defer LU factorization until first flat solve.

    Returns:
        UnifiedPowerGridModel with appropriate configuration.
    """
    if auto_detect_source:
        # Detect source based on graph type and edge attributes
        if isinstance(graph, RustworkxMultiDiGraphWrapper):
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
        lazy_factor=lazy_factor,
    )

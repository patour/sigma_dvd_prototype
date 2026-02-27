"""Graph type detection and conversion utilities.

This module provides functions to detect graph types and convert between
NetworkX and Rustworkx graph representations for PDN analysis.

Supports conversion of pickled NetworkX graphs (legacy format) to
RustworkxMultiDiGraphWrapper for use with the current solver infrastructure.
"""

from __future__ import annotations

from typing import Any, Union

import networkx as nx

from .rx_graph import RustworkxMultiDiGraphWrapper


def is_networkx_graph(graph: Any) -> bool:
    """Check if graph is a NetworkX graph type.

    Args:
        graph: Graph object to check

    Returns:
        True if graph is a NetworkX Graph, DiGraph, MultiGraph, or MultiDiGraph
    """
    return isinstance(graph, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph))


def is_rustworkx_graph(graph: Any) -> bool:
    """Check if graph is a Rustworkx wrapper type.

    Args:
        graph: Graph object to check

    Returns:
        True if graph is a RustworkxMultiDiGraphWrapper
    """
    return isinstance(graph, RustworkxMultiDiGraphWrapper)


def detect_graph_type(graph: Any) -> str:
    """Detect the type of graph.

    Args:
        graph: Graph object to check

    Returns:
        String identifier: 'networkx', 'rustworkx', or 'unknown'
    """
    if is_rustworkx_graph(graph):
        return 'rustworkx'
    elif is_networkx_graph(graph):
        return 'networkx'
    else:
        return 'unknown'


def convert_networkx_to_rustworkx(
    nx_graph: nx.MultiDiGraph,
    verbose: bool = False,
) -> RustworkxMultiDiGraphWrapper:
    """Convert a NetworkX MultiDiGraph to RustworkxMultiDiGraphWrapper.

    Preserves all node attributes, edge attributes, and graph metadata.

    Args:
        nx_graph: NetworkX MultiDiGraph to convert
        verbose: If True, print conversion progress

    Returns:
        RustworkxMultiDiGraphWrapper with same structure and attributes

    Raises:
        TypeError: If input is not a NetworkX MultiDiGraph
    """
    if not isinstance(nx_graph, nx.MultiDiGraph):
        raise TypeError(
            f"Expected networkx.MultiDiGraph, got {type(nx_graph).__name__}. "
            f"Only MultiDiGraph conversion is currently supported."
        )

    rx_graph = RustworkxMultiDiGraphWrapper()

    # Copy graph-level metadata
    rx_graph.graph.update(nx_graph.graph)

    # Copy nodes with attributes
    node_count = nx_graph.number_of_nodes()
    if verbose:
        print(f"Converting {node_count} nodes...")

    for node, attrs in nx_graph.nodes(data=True):
        rx_graph.add_node(node, **attrs)

    # Copy edges with attributes
    edge_count = nx_graph.number_of_edges()
    if verbose:
        print(f"Converting {edge_count} edges...")

    for u, v, key, attrs in nx_graph.edges(keys=True, data=True):
        rx_graph.add_edge(u, v, **attrs)

    if verbose:
        print(f"Conversion complete: {rx_graph.number_of_nodes()} nodes, "
              f"{rx_graph.number_of_edges()} edges")

    return rx_graph


def ensure_rustworkx_graph(
    graph: Any,
    verbose: bool = False,
) -> RustworkxMultiDiGraphWrapper:
    """Ensure graph is a RustworkxMultiDiGraphWrapper, converting if needed.

    This is the main entry point for handling graphs that may be either
    NetworkX or Rustworkx format (e.g., loaded from pickle).

    Args:
        graph: Graph object (NetworkX or Rustworkx)
        verbose: If True, print conversion info

    Returns:
        RustworkxMultiDiGraphWrapper (original if already Rustworkx, converted if NetworkX)

    Raises:
        TypeError: If graph type is not supported
    """
    graph_type = detect_graph_type(graph)

    if graph_type == 'rustworkx':
        if verbose:
            print("Graph is already RustworkxMultiDiGraphWrapper")
        return graph

    elif graph_type == 'networkx':
        if verbose:
            print(f"Converting {type(graph).__name__} to RustworkxMultiDiGraphWrapper...")
        return convert_networkx_to_rustworkx(graph, verbose=verbose)

    else:
        raise TypeError(
            f"Unsupported graph type: {type(graph).__name__}. "
            f"Expected networkx.MultiDiGraph or RustworkxMultiDiGraphWrapper."
        )

"""Custom graph algorithms for rustworkx wrappers.

This module provides implementations of NetworkX algorithms that don't have
direct rustworkx equivalents. These are designed to work with the wrapper
classes in rx_graph.py.

Functions:
- contract_nodes: Merge two nodes into one (nx.contracted_nodes equivalent)
- node_connected_component: Find connected component containing a node
- connected_components: Find all connected components
"""

from __future__ import annotations

from typing import Any, Set, List, TYPE_CHECKING

import rustworkx as rx

if TYPE_CHECKING:
    from .rx_graph import RustworkxGraphWrapper, RustworkxMultiDiGraphWrapper


def contract_nodes(
    wrapper: "RustworkxGraphWrapper | RustworkxMultiDiGraphWrapper",
    keep_node: Any,
    remove_node: Any,
    self_loops: bool = False,
) -> "RustworkxGraphWrapper | RustworkxMultiDiGraphWrapper":
    """Contract remove_node into keep_node (NetworkX equivalent).

    All edges incident to remove_node are redirected to keep_node.
    The remove_node is then deleted from the graph.

    This is the rustworkx equivalent of:
        nx.contracted_nodes(G, keep_node, remove_node, self_loops=False)

    Args:
        wrapper: Graph wrapper (undirected or directed)
        keep_node: Node to keep (edges redirected here)
        remove_node: Node to remove (merged into keep_node)
        self_loops: If True, create self-loops when contracting adjacent nodes.
                   If False (default), edges between keep and remove are dropped.

    Returns:
        The modified wrapper (same object, modified in-place)

    Raises:
        KeyError: If either node is not in the graph

    Example:
        # Contract node 'b' into node 'a'
        # Before: a -- b -- c
        # After:  a -- c (b is removed, its edge to c redirected to a)
        contract_nodes(g, 'a', 'b')
    """
    if keep_node not in wrapper._node_to_idx:
        raise KeyError(f"Keep node {keep_node} not in graph")
    if remove_node not in wrapper._node_to_idx:
        raise KeyError(f"Remove node {remove_node} not in graph")

    keep_idx = wrapper._node_to_idx[keep_node]
    remove_idx = wrapper._node_to_idx[remove_node]

    # Collect edges to redirect
    edges_to_add: List[tuple] = []

    # Check if this is a directed graph (use isinstance, not hasattr)
    is_directed = isinstance(wrapper._graph, rx.PyDiGraph)

    if is_directed:
        # Handle outgoing edges from remove_node
        # Use out_edge_indices to get edge indices
        for edge_idx in list(wrapper._graph.out_edge_indices(remove_idx)):
            u_idx, v_idx = wrapper._graph.get_edge_endpoints_by_index(edge_idx)
            target_idx = v_idx

            # Skip if this would create self-loop and we don't want them
            if target_idx == keep_idx and not self_loops:
                continue

            edge_data = wrapper._graph.get_edge_data_by_index(edge_idx)
            target_node = wrapper._idx_to_node[target_idx]
            edges_to_add.append((keep_node, target_node, edge_data))

        # Handle incoming edges to remove_node
        # Use in_edge_indices to get edge indices
        for edge_idx in list(wrapper._graph.in_edge_indices(remove_idx)):
            u_idx, v_idx = wrapper._graph.get_edge_endpoints_by_index(edge_idx)
            source_idx = u_idx

            # Skip if this would create self-loop and we don't want them
            if source_idx == keep_idx and not self_loops:
                continue

            edge_data = wrapper._graph.get_edge_data_by_index(edge_idx)
            source_node = wrapper._idx_to_node[source_idx]
            edges_to_add.append((source_node, keep_node, edge_data))
    else:
        # Undirected graph - use incident_edges
        for edge_idx in list(wrapper._graph.incident_edges(remove_idx)):
            u_idx, v_idx = wrapper._graph.get_edge_endpoints_by_index(edge_idx)
            neighbor_idx = v_idx if u_idx == remove_idx else u_idx

            # Skip if this would create self-loop and we don't want them
            if neighbor_idx == keep_idx and not self_loops:
                continue

            edge_data = wrapper._graph.get_edge_data_by_index(edge_idx)
            neighbor_node = wrapper._idx_to_node[neighbor_idx]
            edges_to_add.append((keep_node, neighbor_node, edge_data))

    # Merge node attributes (keep_node attrs take precedence)
    if remove_node in wrapper._node_attrs:
        keep_attrs = wrapper._node_attrs.get(keep_node, {})
        remove_attrs = wrapper._node_attrs[remove_node]
        merged = {**remove_attrs, **keep_attrs}
        wrapper._node_attrs[keep_node] = merged

    # Remove the node (this removes all its edges automatically)
    wrapper.remove_node(remove_node)

    # Add redirected edges
    for edge_tuple in edges_to_add:
        u, v, data = edge_tuple
        # For undirected graphs, avoid duplicate edges
        if not is_directed and wrapper.has_edge(u, v):
            continue
        wrapper.add_edge(u, v, **(data or {}))

    return wrapper


def node_connected_component(
    wrapper: "RustworkxGraphWrapper | RustworkxMultiDiGraphWrapper",
    node: Any,
) -> Set[Any]:
    """Return the set of nodes in the connected component containing node.

    This is the rustworkx equivalent of:
        nx.node_connected_component(G, node)

    For directed graphs, uses weak connectivity (ignores edge direction).

    Args:
        wrapper: Graph wrapper
        node: Node to find component for

    Returns:
        Set of node keys in the same connected component

    Raises:
        KeyError: If node is not in the graph

    Example:
        # Graph: a -- b -- c    d -- e (two components)
        component = node_connected_component(g, 'b')
        # Returns: {'a', 'b', 'c'}
    """
    if node not in wrapper._node_to_idx:
        raise KeyError(f"Node {node} not in graph")

    node_idx = wrapper._node_to_idx[node]

    # Get all connected components (use weak connectivity for directed graphs)
    is_directed = isinstance(wrapper._graph, rx.PyDiGraph)
    if is_directed:
        components = rx.weakly_connected_components(wrapper._graph)
    else:
        components = rx.connected_components(wrapper._graph)

    # Find component containing our node
    for component_indices in components:
        if node_idx in component_indices:
            return {wrapper._idx_to_node[idx] for idx in component_indices}

    # Shouldn't reach here, but return singleton if we do
    return {node}


def connected_components(
    wrapper: "RustworkxGraphWrapper | RustworkxMultiDiGraphWrapper",
) -> List[Set[Any]]:
    """Return list of sets of nodes, one set per connected component.

    This is the rustworkx equivalent of:
        list(nx.connected_components(G))

    For directed graphs, uses weak connectivity (ignores edge direction).

    Args:
        wrapper: Graph wrapper

    Returns:
        List of sets, where each set contains node keys in one component

    Example:
        # Graph: a -- b -- c    d -- e (two components)
        components = connected_components(g)
        # Returns: [{'a', 'b', 'c'}, {'d', 'e'}]
    """
    # Use weak connectivity for directed graphs
    is_directed = isinstance(wrapper._graph, rx.PyDiGraph)
    if is_directed:
        idx_components = rx.weakly_connected_components(wrapper._graph)
    else:
        idx_components = rx.connected_components(wrapper._graph)

    return [
        {wrapper._idx_to_node[idx] for idx in comp}
        for comp in idx_components
    ]


def weakly_connected_components(
    wrapper: "RustworkxMultiDiGraphWrapper",
) -> List[Set[Any]]:
    """Return weakly connected components of a directed graph.

    This is the rustworkx equivalent of:
        list(nx.weakly_connected_components(G))

    Args:
        wrapper: Directed graph wrapper

    Returns:
        List of sets, where each set contains node keys in one component
    """
    # rx.connected_components already treats directed graphs as undirected
    return connected_components(wrapper)


def is_connected(
    wrapper: "RustworkxGraphWrapper | RustworkxMultiDiGraphWrapper",
) -> bool:
    """Check if graph is connected.

    This is the rustworkx equivalent of:
        nx.is_connected(G)

    For directed graphs, checks weak connectivity.

    Args:
        wrapper: Graph wrapper

    Returns:
        True if graph is connected, False otherwise
    """
    if wrapper.number_of_nodes() == 0:
        return True  # Empty graph is considered connected

    # Use weak connectivity for directed graphs
    is_directed = isinstance(wrapper._graph, rx.PyDiGraph)
    if is_directed:
        return rx.is_weakly_connected(wrapper._graph)
    else:
        return rx.is_connected(wrapper._graph)


def number_connected_components(
    wrapper: "RustworkxGraphWrapper | RustworkxMultiDiGraphWrapper",
) -> int:
    """Return number of connected components.

    For directed graphs, counts weakly connected components.

    Args:
        wrapper: Graph wrapper

    Returns:
        Number of connected components
    """
    # Use weak connectivity for directed graphs
    is_directed = isinstance(wrapper._graph, rx.PyDiGraph)
    if is_directed:
        return rx.number_weakly_connected_components(wrapper._graph)
    else:
        return rx.number_connected_components(wrapper._graph)


class NetworkXNoPath(Exception):
    """Exception raised when no path exists between nodes."""
    pass


class NodeNotFound(Exception):
    """Exception raised when a node is not found in the graph."""
    pass


def dijkstra_path_length(
    wrapper: "RustworkxGraphWrapper | RustworkxMultiDiGraphWrapper",
    source: Any,
    target: Any,
    weight_fn: callable = None,
) -> float:
    """Compute shortest path length using Dijkstra's algorithm.

    This is the rustworkx equivalent of:
        nx.dijkstra_path_length(G, source, target, weight=weight_fn)

    Args:
        wrapper: Graph wrapper
        source: Source node
        target: Target node
        weight_fn: Optional callable(u, v, data) -> float to compute edge weight.
                   If None, uses edge weight of 1.0 for all edges.

    Returns:
        Shortest path length (sum of edge weights)

    Raises:
        NodeNotFound: If source or target is not in the graph
        NetworkXNoPath: If no path exists between source and target
    """
    if source not in wrapper._node_to_idx:
        raise NodeNotFound(f"Source node {source} not in graph")
    if target not in wrapper._node_to_idx:
        raise NodeNotFound(f"Target node {target} not in graph")

    if source == target:
        return 0.0

    source_idx = wrapper._node_to_idx[source]
    target_idx = wrapper._node_to_idx[target]

    # Build weight function that works with rustworkx edge indices
    if weight_fn is not None:
        # weight_fn takes (u, v, data) but rustworkx gives edge_data
        # We need to convert rustworkx format to NetworkX format
        def rx_weight_fn(edge_data):
            # For the rustworkx call, we get edge data directly
            # We can't easily get u, v here, so approximate using a simpler approach
            if edge_data is None:
                return 1.0
            # The caller's weight_fn expects (u, v, data) but we only have data
            # This is a limitation - use a wrapper that only uses data
            return weight_fn(None, None, edge_data)
    else:
        def rx_weight_fn(edge_data):
            return 1.0

    try:
        # Use rustworkx dijkstra
        distances = rx.dijkstra_shortest_path_lengths(
            wrapper._graph, source_idx, rx_weight_fn
        )

        if target_idx not in distances:
            raise NetworkXNoPath(f"No path between {source} and {target}")

        return distances[target_idx]
    except Exception as e:
        if "No path" in str(e) or isinstance(e, NetworkXNoPath):
            raise NetworkXNoPath(f"No path between {source} and {target}")
        raise


def has_path(
    wrapper: "RustworkxGraphWrapper | RustworkxMultiDiGraphWrapper",
    source: Any,
    target: Any,
) -> bool:
    """Check if a path exists between source and target.

    This is the rustworkx equivalent of:
        nx.has_path(G, source, target)

    For directed graphs, checks for a directed path.

    Args:
        wrapper: Graph wrapper
        source: Source node
        target: Target node

    Returns:
        True if a path exists, False otherwise

    Raises:
        KeyError: If source or target is not in the graph
    """
    if source not in wrapper._node_to_idx:
        raise KeyError(f"Source node {source} not in graph")
    if target not in wrapper._node_to_idx:
        raise KeyError(f"Target node {target} not in graph")

    source_idx = wrapper._node_to_idx[source]
    target_idx = wrapper._node_to_idx[target]

    # Use dijkstra to check path existence (returns empty dict if no path)
    is_directed = isinstance(wrapper._graph, rx.PyDiGraph)

    if is_directed:
        # For directed graphs, use has_path which checks directed paths
        try:
            # Try to find any path using BFS
            path = rx.dijkstra_shortest_paths(
                wrapper._graph, source_idx, target_idx,
                weight_fn=lambda _: 1.0
            )
            return target_idx in path
        except Exception:
            return False
    else:
        # For undirected graphs, check if they're in the same component
        return node_connected_component(wrapper, source) == node_connected_component(wrapper, target) or \
               target in node_connected_component(wrapper, source)

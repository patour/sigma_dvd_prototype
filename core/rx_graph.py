"""Rustworkx graph wrappers providing NetworkX-compatible API.

This module provides wrapper classes that expose a NetworkX-like interface
over rustworkx PyGraph and PyDiGraph for improved performance.

Key differences from NetworkX:
- rustworkx uses integer indices for nodes, not arbitrary keys
- These wrappers maintain bidirectional mappings (key <-> index)
- Graph metadata stored in separate dict (replaces nx graph.graph)
"""

from __future__ import annotations

from typing import (
    Any, Dict, Iterator, List, Optional, Set, Tuple, Union,
    Iterable, TypeVar, Generic,
)

import rustworkx as rx

NodeT = TypeVar('NodeT')


class RustworkxGraphWrapper:
    """NetworkX-compatible wrapper for rustworkx PyGraph (undirected).

    Provides familiar NetworkX API while using rustworkx for performance.
    Maintains bidirectional mapping between application node keys and
    rustworkx integer indices.

    Example:
        g = RustworkxGraphWrapper()
        g.add_node('a', x=100, y=200)
        g.add_edge('a', 'b', resistance=1.5)
        for u, v, data in g.edges(data=True):
            print(f"{u} -- {v}: {data}")
    """

    def __init__(self):
        """Initialize empty undirected graph."""
        self._graph: rx.PyGraph = rx.PyGraph()
        self._node_to_idx: Dict[Any, int] = {}
        self._idx_to_node: Dict[int, Any] = {}
        self._node_attrs: Dict[Any, Dict[str, Any]] = {}
        self._metadata: Dict[str, Any] = {}

    # =========================================================================
    # Graph Metadata (replaces nx graph.graph dict)
    # =========================================================================

    @property
    def graph(self) -> Dict[str, Any]:
        """Access graph-level metadata dict (NetworkX graph.graph equivalent)."""
        return self._metadata

    # =========================================================================
    # Node Operations
    # =========================================================================

    def add_node(self, node: Any, **attrs) -> int:
        """Add a node with optional attributes.

        If node already exists, updates its attributes.

        Args:
            node: Node key (any hashable)
            **attrs: Node attributes

        Returns:
            rustworkx index of the node
        """
        if node in self._node_to_idx:
            # Update existing node attrs
            self._node_attrs[node].update(attrs)
            return self._node_to_idx[node]

        # Add new node
        idx = self._graph.add_node(node)  # Store key as payload
        self._node_to_idx[node] = idx
        self._idx_to_node[idx] = node
        self._node_attrs[node] = dict(attrs)
        return idx

    def remove_node(self, node: Any) -> None:
        """Remove a node and all its incident edges.

        Args:
            node: Node key to remove

        Raises:
            KeyError: If node not in graph
        """
        if node not in self._node_to_idx:
            raise KeyError(f"Node {node} not in graph")

        idx = self._node_to_idx[node]
        self._graph.remove_node(idx)

        # Clean up mappings
        del self._node_to_idx[node]
        del self._idx_to_node[idx]
        del self._node_attrs[node]

    def has_node(self, node: Any) -> bool:
        """Check if node exists in graph."""
        return node in self._node_to_idx

    def __contains__(self, node: Any) -> bool:
        """Support 'node in graph' syntax."""
        return node in self._node_to_idx

    def nodes(self, data: bool = False) -> Iterator:
        """Iterate over nodes, optionally with data.

        Args:
            data: If True, yield (node, attr_dict) tuples

        Yields:
            node or (node, attr_dict) depending on data parameter
        """
        if data:
            for node, attrs in self._node_attrs.items():
                yield (node, attrs)
        else:
            yield from self._node_attrs.keys()

    @property
    def nodes_dict(self) -> Dict[Any, Dict[str, Any]]:
        """Direct access to node attributes dict.

        Replaces NetworkX graph.nodes[n] access pattern.
        Use: wrapper.nodes_dict[node]['attr']
        """
        return self._node_attrs

    def number_of_nodes(self) -> int:
        """Return number of nodes."""
        return self._graph.num_nodes()

    def __len__(self) -> int:
        """Return number of nodes (len(graph))."""
        return self._graph.num_nodes()

    # =========================================================================
    # Edge Operations
    # =========================================================================

    def add_edge(self, u: Any, v: Any, **attrs) -> int:
        """Add an edge between u and v with optional attributes.

        Nodes are automatically added if they don't exist.

        Args:
            u: Source node key
            v: Target node key
            **attrs: Edge attributes

        Returns:
            rustworkx edge index
        """
        # Ensure nodes exist
        if u not in self._node_to_idx:
            self.add_node(u)
        if v not in self._node_to_idx:
            self.add_node(v)

        u_idx = self._node_to_idx[u]
        v_idx = self._node_to_idx[v]

        return self._graph.add_edge(u_idx, v_idx, attrs)

    def remove_edge(self, u: Any, v: Any) -> None:
        """Remove edge between u and v.

        Args:
            u: Source node key
            v: Target node key

        Raises:
            KeyError: If edge does not exist
        """
        u_idx = self._node_to_idx.get(u)
        v_idx = self._node_to_idx.get(v)

        if u_idx is None or v_idx is None:
            raise KeyError(f"Edge ({u}, {v}) not in graph")

        if not self._graph.has_edge(u_idx, v_idx):
            raise KeyError(f"Edge ({u}, {v}) not in graph")

        self._graph.remove_edge(u_idx, v_idx)

    def has_edge(self, u: Any, v: Any) -> bool:
        """Check if edge exists between u and v."""
        u_idx = self._node_to_idx.get(u)
        v_idx = self._node_to_idx.get(v)

        if u_idx is None or v_idx is None:
            return False

        return self._graph.has_edge(u_idx, v_idx)

    def edges(
        self,
        nbunch: Optional[Any] = None,
        data: bool = False,
        keys: bool = False,
    ) -> Iterator:
        """Iterate over edges, optionally with data.

        Args:
            nbunch: Node or iterable of nodes to get edges for (None = all)
            data: If True, include edge attributes
            keys: If True, include edge keys (for API compatibility)

        Yields:
            Tuples of (u, v), (u, v, data), (u, v, key), or (u, v, key, data)
        """
        if nbunch is not None:
            # Edges for specific node(s)
            if not isinstance(nbunch, (set, list, tuple)):
                nbunch = [nbunch]

            seen_edges: Set[Tuple[int, int]] = set()
            for node in nbunch:
                idx = self._node_to_idx.get(node)
                if idx is None:
                    continue

                for edge_idx in self._graph.incident_edges(idx):
                    endpoints = self._graph.get_edge_endpoints_by_index(edge_idx)
                    if endpoints in seen_edges:
                        continue
                    # Normalize edge order for undirected
                    u_idx, v_idx = endpoints
                    edge_key = (min(u_idx, v_idx), max(u_idx, v_idx))
                    if edge_key in seen_edges:
                        continue
                    seen_edges.add(edge_key)

                    u = self._idx_to_node[u_idx]
                    v = self._idx_to_node[v_idx]
                    edge_data = self._graph.get_edge_data_by_index(edge_idx)

                    yield self._format_edge(u, v, edge_idx, edge_data, data, keys)
        else:
            # All edges
            for edge_idx in self._graph.edge_indices():
                u_idx, v_idx = self._graph.get_edge_endpoints_by_index(edge_idx)
                u = self._idx_to_node[u_idx]
                v = self._idx_to_node[v_idx]
                edge_data = self._graph.get_edge_data_by_index(edge_idx)

                yield self._format_edge(u, v, edge_idx, edge_data, data, keys)

    def _format_edge(
        self, u: Any, v: Any, key: int, edge_data: Dict, data: bool, keys: bool
    ) -> Tuple:
        """Format edge tuple based on data/keys flags."""
        if data and keys:
            return (u, v, key, edge_data)
        elif data:
            return (u, v, edge_data)
        elif keys:
            return (u, v, key)
        else:
            return (u, v)

    def get_edge_data(self, u: Any, v: Any, default: Any = None) -> Optional[Dict]:
        """Get edge data for edge (u, v).

        Args:
            u: Source node
            v: Target node
            default: Value to return if edge doesn't exist

        Returns:
            Edge attribute dict or default
        """
        u_idx = self._node_to_idx.get(u)
        v_idx = self._node_to_idx.get(v)

        if u_idx is None or v_idx is None:
            return default

        try:
            return self._graph.get_edge_data(u_idx, v_idx)
        except rx.NoEdgeBetweenNodes:
            return default

    def number_of_edges(self) -> int:
        """Return number of edges."""
        return self._graph.num_edges()

    # =========================================================================
    # Neighbor/Adjacency Operations
    # =========================================================================

    def neighbors(self, node: Any) -> Iterator[Any]:
        """Get neighbors of a node.

        Args:
            node: Node key

        Yields:
            Neighbor node keys
        """
        idx = self._node_to_idx.get(node)
        if idx is None:
            return

        for neighbor_idx in self._graph.neighbors(idx):
            yield self._idx_to_node[neighbor_idx]

    def degree(self, node: Any) -> int:
        """Get degree of a node.

        Args:
            node: Node key

        Returns:
            Number of edges incident to node
        """
        idx = self._node_to_idx.get(node)
        if idx is None:
            return 0
        return self._graph.degree(idx)

    def adj(self, node: Any) -> Dict[Any, Dict]:
        """Get adjacency dict for a node.

        Returns dict mapping neighbor -> edge_data.

        Args:
            node: Node key

        Returns:
            Dict of neighbor -> edge attributes
        """
        result = {}
        idx = self._node_to_idx.get(node)
        if idx is None:
            return result

        for edge_idx in self._graph.incident_edges(idx):
            u_idx, v_idx = self._graph.get_edge_endpoints_by_index(edge_idx)
            neighbor_idx = v_idx if u_idx == idx else u_idx
            neighbor = self._idx_to_node[neighbor_idx]
            edge_data = self._graph.get_edge_data_by_index(edge_idx)
            result[neighbor] = edge_data

        return result

    def __getitem__(self, node: Any) -> Dict[Any, Dict]:
        """Support graph[node] -> adjacency dict."""
        return self.adj(node)

    # =========================================================================
    # Subgraph Operations
    # =========================================================================

    def subgraph(self, nodes: Iterable[Any]) -> 'RustworkxGraphWrapper':
        """Return a subgraph containing only the specified nodes.

        Creates a new independent graph (not a view).
        Uses rustworkx native subgraph for O(n+m) performance where n, m are
        the subgraph node/edge counts, rather than O(N+M) for full graph.

        Args:
            nodes: Iterable of node keys to include

        Returns:
            New RustworkxGraphWrapper with only specified nodes and edges between them
        """
        nodes_set = set(nodes)

        # Get rustworkx indices for requested nodes
        node_indices = [self._node_to_idx[n] for n in nodes_set if n in self._node_to_idx]

        if not node_indices:
            # Return empty graph
            sub = RustworkxGraphWrapper()
            sub._metadata = dict(self._metadata)
            return sub

        # Use rustworkx native subgraph (Rust implementation - much faster)
        rx_sub, node_map = self._graph.subgraph_with_nodemap(node_indices, preserve_attrs=True)

        # Build new wrapper with correct mappings
        sub = RustworkxGraphWrapper.__new__(RustworkxGraphWrapper)
        sub._graph = rx_sub
        sub._metadata = dict(self._metadata)
        sub._node_to_idx = {}
        sub._idx_to_node = {}
        sub._node_attrs = {}

        # Rebuild node mappings using node_map (new_idx -> old_idx)
        for new_idx in range(rx_sub.num_nodes()):
            old_idx = node_map[new_idx]
            node_key = self._idx_to_node[old_idx]
            sub._node_to_idx[node_key] = new_idx
            sub._idx_to_node[new_idx] = node_key
            if node_key in self._node_attrs:
                sub._node_attrs[node_key] = self._node_attrs[node_key]

        return sub

    def copy(self) -> 'RustworkxGraphWrapper':
        """Return a deep copy of the graph."""
        return self.subgraph(self._node_to_idx.keys())

    # =========================================================================
    # Direct Access to Underlying Graph
    # =========================================================================

    @property
    def rx_graph(self) -> rx.PyGraph:
        """Direct access to underlying rustworkx PyGraph.

        Use with caution - prefer wrapper methods for API consistency.
        """
        return self._graph

    def get_index(self, node: Any) -> Optional[int]:
        """Get rustworkx index for a node key."""
        return self._node_to_idx.get(node)

    def get_node(self, idx: int) -> Optional[Any]:
        """Get node key for a rustworkx index."""
        return self._idx_to_node.get(idx)

    def is_directed(self) -> bool:
        """Return False (this is an undirected graph)."""
        return False


class RustworkxMultiDiGraphWrapper:
    """NetworkX-compatible wrapper for rustworkx PyDiGraph (directed multigraph).

    Provides familiar NetworkX MultiDiGraph API while using rustworkx for performance.
    Supports multiple parallel edges between the same node pair.

    Example:
        g = RustworkxMultiDiGraphWrapper()
        g.add_edge('a', 'b', type='R', value=1.0)
        g.add_edge('a', 'b', type='C', value=100.0)  # Parallel edge
        for u, v, key, data in g.edges(keys=True, data=True):
            print(f"{u} -> {v} [{key}]: {data}")
    """

    def __init__(self):
        """Initialize empty directed multigraph."""
        self._graph: rx.PyDiGraph = rx.PyDiGraph(multigraph=True)
        self._node_to_idx: Dict[Any, int] = {}
        self._idx_to_node: Dict[int, Any] = {}
        self._node_attrs: Dict[Any, Dict[str, Any]] = {}
        self._metadata: Dict[str, Any] = {}

    # =========================================================================
    # Graph Metadata
    # =========================================================================

    @property
    def graph(self) -> Dict[str, Any]:
        """Access graph-level metadata dict."""
        return self._metadata

    # =========================================================================
    # Node Operations
    # =========================================================================

    def add_node(self, node: Any, **attrs) -> int:
        """Add a node with optional attributes."""
        if node in self._node_to_idx:
            self._node_attrs[node].update(attrs)
            return self._node_to_idx[node]

        idx = self._graph.add_node(node)
        self._node_to_idx[node] = idx
        self._idx_to_node[idx] = node
        self._node_attrs[node] = dict(attrs)
        return idx

    def remove_node(self, node: Any) -> None:
        """Remove a node and all its incident edges."""
        if node not in self._node_to_idx:
            raise KeyError(f"Node {node} not in graph")

        idx = self._node_to_idx[node]
        self._graph.remove_node(idx)

        del self._node_to_idx[node]
        del self._idx_to_node[idx]
        del self._node_attrs[node]

    def remove_nodes_from(self, nodes: Iterable[Any]) -> None:
        """Remove multiple nodes and all their incident edges.

        Args:
            nodes: Iterable of node keys to remove

        Note:
            Nodes not in the graph are silently skipped.
        """
        for node in nodes:
            if node in self._node_to_idx:
                self.remove_node(node)

    def has_node(self, node: Any) -> bool:
        """Check if node exists."""
        return node in self._node_to_idx

    def __contains__(self, node: Any) -> bool:
        """Support 'node in graph' syntax."""
        return node in self._node_to_idx

    def nodes(self, data: bool = False) -> Iterator:
        """Iterate over nodes, optionally with data."""
        if data:
            for node, attrs in self._node_attrs.items():
                yield (node, attrs)
        else:
            yield from self._node_attrs.keys()

    @property
    def nodes_dict(self) -> Dict[Any, Dict[str, Any]]:
        """Direct access to node attributes dict."""
        return self._node_attrs

    def number_of_nodes(self) -> int:
        """Return number of nodes."""
        return self._graph.num_nodes()

    def __len__(self) -> int:
        """Return number of nodes."""
        return self._graph.num_nodes()

    # =========================================================================
    # Edge Operations
    # =========================================================================

    def add_edge(self, u: Any, v: Any, **attrs) -> int:
        """Add a directed edge from u to v.

        Multiple edges between same nodes are allowed (multigraph).

        Args:
            u: Source node
            v: Target node
            **attrs: Edge attributes

        Returns:
            Edge index
        """
        if u not in self._node_to_idx:
            self.add_node(u)
        if v not in self._node_to_idx:
            self.add_node(v)

        u_idx = self._node_to_idx[u]
        v_idx = self._node_to_idx[v]

        return self._graph.add_edge(u_idx, v_idx, attrs)

    def remove_edge(self, u: Any, v: Any, key: Optional[int] = None) -> None:
        """Remove edge from u to v.

        Args:
            u: Source node
            v: Target node
            key: Optional edge key. If None, removes first edge found.
        """
        u_idx = self._node_to_idx.get(u)
        v_idx = self._node_to_idx.get(v)

        if u_idx is None or v_idx is None:
            raise KeyError(f"Edge ({u}, {v}) not in graph")

        if key is not None:
            # Remove specific edge by index
            self._graph.remove_edge_from_index(key)
        else:
            # Remove first edge between u and v
            if not self._graph.has_edge(u_idx, v_idx):
                raise KeyError(f"Edge ({u}, {v}) not in graph")
            self._graph.remove_edge(u_idx, v_idx)

    def has_edge(self, u: Any, v: Any) -> bool:
        """Check if directed edge exists from u to v."""
        u_idx = self._node_to_idx.get(u)
        v_idx = self._node_to_idx.get(v)

        if u_idx is None or v_idx is None:
            return False

        return self._graph.has_edge(u_idx, v_idx)

    def edges(
        self,
        nbunch: Optional[Any] = None,
        data: bool = False,
        keys: bool = False,
    ) -> Iterator:
        """Iterate over edges.

        Args:
            nbunch: Node(s) to get edges for (outgoing). None = all edges.
            data: Include edge data
            keys: Include edge keys

        Yields:
            Edge tuples
        """
        if nbunch is not None:
            if not isinstance(nbunch, (set, list, tuple)):
                nbunch = [nbunch]

            for node in nbunch:
                idx = self._node_to_idx.get(node)
                if idx is None:
                    continue

                # Use out_edge_indices to get edge indices
                for edge_idx in self._graph.out_edge_indices(idx):
                    u_idx, v_idx = self._graph.get_edge_endpoints_by_index(edge_idx)
                    u = self._idx_to_node[u_idx]
                    v = self._idx_to_node[v_idx]
                    edge_data = self._graph.get_edge_data_by_index(edge_idx)

                    yield self._format_edge(u, v, edge_idx, edge_data, data, keys)
        else:
            for edge_idx in self._graph.edge_indices():
                u_idx, v_idx = self._graph.get_edge_endpoints_by_index(edge_idx)
                u = self._idx_to_node[u_idx]
                v = self._idx_to_node[v_idx]
                edge_data = self._graph.get_edge_data_by_index(edge_idx)

                yield self._format_edge(u, v, edge_idx, edge_data, data, keys)

    def _format_edge(
        self, u: Any, v: Any, key: int, edge_data: Dict, data: bool, keys: bool
    ) -> Tuple:
        """Format edge tuple based on data/keys flags."""
        if data and keys:
            return (u, v, key, edge_data)
        elif data:
            return (u, v, edge_data)
        elif keys:
            return (u, v, key)
        else:
            return (u, v)

    def get_edge_data(self, u: Any, v: Any, key: Optional[int] = None, default: Any = None) -> Optional[Dict]:
        """Get edge data between u and v.

        For MultiDiGraph compatibility, returns dict of {edge_key: edge_data}
        when key is None (all edges), or just edge_data when key is specified.

        Args:
            u: Source node
            v: Target node
            key: Optional specific edge key. If None, returns all edges.
            default: Value to return if no edges exist

        Returns:
            If key is None: {edge_key: edge_data_dict} for all edges
            If key is specified: edge_data_dict for that edge, or default
        """
        u_idx = self._node_to_idx.get(u)
        v_idx = self._node_to_idx.get(v)

        if u_idx is None or v_idx is None:
            return default

        # Collect all edges between u and v
        result: Dict[int, Dict] = {}
        for edge_idx in self._graph.edge_indices():
            eu_idx, ev_idx = self._graph.get_edge_endpoints_by_index(edge_idx)
            if eu_idx == u_idx and ev_idx == v_idx:
                edge_data = self._graph.get_edge_data_by_index(edge_idx)
                result[edge_idx] = edge_data

        if not result:
            return default

        if key is not None:
            # Return specific edge data or default
            return result.get(key, default)

        # Return dict of all edges (NetworkX MultiDiGraph behavior)
        return result

    def get_all_edge_data(self, u: Any, v: Any) -> List[Dict]:
        """Get data for all edges from u to v (multi-edge support)."""
        u_idx = self._node_to_idx.get(u)
        v_idx = self._node_to_idx.get(v)

        if u_idx is None or v_idx is None:
            return []

        return self._graph.get_all_edge_data(u_idx, v_idx)

    def number_of_edges(self) -> int:
        """Return total number of edges."""
        return self._graph.num_edges()

    # =========================================================================
    # Directed Graph Operations
    # =========================================================================

    def in_edges(
        self, node: Any, data: bool = False, keys: bool = False
    ) -> Iterator:
        """Iterate over incoming edges to a node.

        Args:
            node: Target node
            data: Include edge data
            keys: Include edge keys

        Yields:
            (u, node) or (u, node, data) or (u, node, key, data)
        """
        idx = self._node_to_idx.get(node)
        if idx is None:
            return

        # in_edges returns WeightedEdgeList of (source_idx, target_idx, data) tuples
        # Use in_edge_indices to get edge indices for key support
        in_edge_indices = list(self._graph.in_edge_indices(idx))
        for edge_idx in in_edge_indices:
            u_idx, v_idx = self._graph.get_edge_endpoints_by_index(edge_idx)
            u = self._idx_to_node[u_idx]
            v = self._idx_to_node[v_idx]
            edge_data = self._graph.get_edge_data_by_index(edge_idx)

            yield self._format_edge(u, v, edge_idx, edge_data, data, keys)

    def out_edges(
        self, node: Any, data: bool = False, keys: bool = False
    ) -> Iterator:
        """Iterate over outgoing edges from a node.

        Args:
            node: Source node
            data: Include edge data
            keys: Include edge keys

        Yields:
            (node, v) or (node, v, data) or (node, v, key, data)
        """
        idx = self._node_to_idx.get(node)
        if idx is None:
            return

        # out_edges returns WeightedEdgeList, use out_edge_indices for indices
        out_edge_indices = list(self._graph.out_edge_indices(idx))
        for edge_idx in out_edge_indices:
            u_idx, v_idx = self._graph.get_edge_endpoints_by_index(edge_idx)
            u = self._idx_to_node[u_idx]
            v = self._idx_to_node[v_idx]
            edge_data = self._graph.get_edge_data_by_index(edge_idx)

            yield self._format_edge(u, v, edge_idx, edge_data, data, keys)

    def predecessors(self, node: Any) -> Iterator[Any]:
        """Get predecessor nodes (nodes with edges into this node)."""
        idx = self._node_to_idx.get(node)
        if idx is None:
            return

        # Use predecessor_indices to get indices, not payloads
        for pred_idx in self._graph.predecessor_indices(idx):
            yield self._idx_to_node[pred_idx]

    def successors(self, node: Any) -> Iterator[Any]:
        """Get successor nodes (nodes reachable via outgoing edges)."""
        idx = self._node_to_idx.get(node)
        if idx is None:
            return

        # Use successor_indices to get indices, not payloads
        for succ_idx in self._graph.successor_indices(idx):
            yield self._idx_to_node[succ_idx]

    def neighbors(self, node: Any) -> Iterator[Any]:
        """Get neighbors (successors for directed graph)."""
        return self.successors(node)

    def in_degree(self, node: Any) -> int:
        """Get number of incoming edges."""
        idx = self._node_to_idx.get(node)
        if idx is None:
            return 0
        return self._graph.in_degree(idx)

    def out_degree(self, node: Any) -> int:
        """Get number of outgoing edges."""
        idx = self._node_to_idx.get(node)
        if idx is None:
            return 0
        return self._graph.out_degree(idx)

    def degree(self, node: Any) -> int:
        """Get total degree (in + out)."""
        return self.in_degree(node) + self.out_degree(node)

    # =========================================================================
    # Adjacency Access
    # =========================================================================

    def __getitem__(self, node: Any) -> Dict[Any, Dict[int, Dict]]:
        """Get adjacency dict for a node.

        Returns nested dict: {neighbor: {edge_key: edge_data}}
        This matches NetworkX MultiDiGraph structure.
        """
        result: Dict[Any, Dict[int, Dict]] = {}
        idx = self._node_to_idx.get(node)
        if idx is None:
            return result

        # Use out_edge_indices to get edge indices
        for edge_idx in self._graph.out_edge_indices(idx):
            _, v_idx = self._graph.get_edge_endpoints_by_index(edge_idx)
            neighbor = self._idx_to_node[v_idx]
            edge_data = self._graph.get_edge_data_by_index(edge_idx)

            if neighbor not in result:
                result[neighbor] = {}
            result[neighbor][edge_idx] = edge_data

        return result

    # =========================================================================
    # Subgraph Operations
    # =========================================================================

    def subgraph(self, nodes: Iterable[Any]) -> 'RustworkxMultiDiGraphWrapper':
        """Return a subgraph containing only the specified nodes.

        Creates a new independent graph (not a view).
        Uses rustworkx native subgraph for O(n+m) performance where n, m are
        the subgraph node/edge counts, rather than O(N+M) for full graph.

        Args:
            nodes: Iterable of node keys to include

        Returns:
            New RustworkxMultiDiGraphWrapper with only specified nodes and edges between them
        """
        nodes_set = set(nodes)

        # Get rustworkx indices for requested nodes
        node_indices = [self._node_to_idx[n] for n in nodes_set if n in self._node_to_idx]

        if not node_indices:
            # Return empty graph
            sub = RustworkxMultiDiGraphWrapper()
            sub._metadata = dict(self._metadata)
            return sub

        # Use rustworkx native subgraph (Rust implementation - much faster)
        rx_sub, node_map = self._graph.subgraph_with_nodemap(node_indices, preserve_attrs=True)

        # Build new wrapper with correct mappings
        sub = RustworkxMultiDiGraphWrapper.__new__(RustworkxMultiDiGraphWrapper)
        sub._graph = rx_sub
        sub._metadata = dict(self._metadata)
        sub._node_to_idx = {}
        sub._idx_to_node = {}
        sub._node_attrs = {}

        # Rebuild node mappings using node_map (new_idx -> old_idx)
        for new_idx in range(rx_sub.num_nodes()):
            old_idx = node_map[new_idx]
            node_key = self._idx_to_node[old_idx]
            sub._node_to_idx[node_key] = new_idx
            sub._idx_to_node[new_idx] = node_key
            if node_key in self._node_attrs:
                sub._node_attrs[node_key] = self._node_attrs[node_key]

        return sub

    def copy(self) -> 'RustworkxMultiDiGraphWrapper':
        """Return a deep copy of the graph."""
        return self.subgraph(self._node_to_idx.keys())

    def to_undirected(self) -> RustworkxGraphWrapper:
        """Convert to undirected graph (for connectivity analysis)."""
        undirected = RustworkxGraphWrapper()
        undirected._metadata = dict(self._metadata)

        for node, attrs in self._node_attrs.items():
            undirected.add_node(node, **attrs)

        for edge_idx in self._graph.edge_indices():
            u_idx, v_idx = self._graph.get_edge_endpoints_by_index(edge_idx)
            u = self._idx_to_node[u_idx]
            v = self._idx_to_node[v_idx]
            edge_data = self._graph.get_edge_data_by_index(edge_idx)

            if not undirected.has_edge(u, v):
                undirected.add_edge(u, v, **(edge_data or {}))

        return undirected

    # =========================================================================
    # Direct Access to Underlying Graph
    # =========================================================================

    @property
    def rx_graph(self) -> rx.PyDiGraph:
        """Direct access to underlying rustworkx PyDiGraph."""
        return self._graph

    def get_index(self, node: Any) -> Optional[int]:
        """Get rustworkx index for a node key."""
        return self._node_to_idx.get(node)

    def get_node(self, idx: int) -> Optional[Any]:
        """Get node key for a rustworkx index."""
        return self._idx_to_node.get(idx)

    def is_directed(self) -> bool:
        """Return True (this is a directed graph)."""
        return True

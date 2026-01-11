"""Unified power grid model supporting both synthetic and PDN sources.

This module provides a unified interface for power grid analysis that works
with both synthetic grids (from generate_power_grid.py) and PDN netlists
(from pdn/pdn_parser.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import networkx as nx
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .node_adapter import NodeInfoExtractor, UnifiedNodeInfo, LayerID
from .edge_adapter import EdgeInfoExtractor, ElementType


class GridSource(Enum):
    """Source type for power grid data."""
    SYNTHETIC = 'synthetic'
    PDN_NETLIST = 'pdn'


@dataclass
class EdgeArrayCache:
    """Pre-computed edge arrays for fast subgrid building.
    
    Stores all resistive edges in NumPy arrays for O(E) vectorized
    filtering instead of O(E) Python iteration per subgrid.
    
    Attributes:
        edge_u_idx: Source node indices (int32)
        edge_v_idx: Target node indices (int32)
        edge_g: Conductance values (float64)
        node_to_idx: Dict mapping node -> global index
        idx_to_node: List for reverse lookup (index -> node)
        n_nodes: Total number of nodes (excluding ground)
        n_edges: Total number of resistive edges
    """
    edge_u_idx: np.ndarray  # shape (n_edges,), dtype=int32
    edge_v_idx: np.ndarray  # shape (n_edges,), dtype=int32
    edge_g: np.ndarray      # shape (n_edges,), dtype=float64
    node_to_idx: Dict[Any, int]
    idx_to_node: List[Any]
    n_nodes: int
    n_edges: int


@dataclass
class UnifiedReducedSystem:
    """Reduced system for solving nodal voltages.

    Holds the Schur-reduced conductance matrix and LU factorization
    for efficient solving of the nodal equation G*V = I.

    Attributes:
        node_order: List of all nodes in matrix order
        unknown_nodes: List of nodes to solve for (non-pad)
        pad_nodes: List of Dirichlet boundary nodes (voltage sources)
        G_uu: Sparse conductance submatrix for unknowns
        G_up: Sparse coupling between unknowns and pads
        lu: LU factorization callable for fast solves
        pad_voltage: Voltage applied at pad nodes
        index_of: Dict mapping node -> index in full ordering
        index_unknown: Dict mapping node -> index in unknown ordering
        net_name: Optional net name for multi-net support
    """
    node_order: List[Any]
    unknown_nodes: List[Any]
    pad_nodes: List[Any]
    G_uu: sp.csr_matrix
    G_up: sp.csr_matrix
    lu: callable
    pad_voltage: float
    index_of: Dict[Any, int]
    index_unknown: Dict[Any, int]
    net_name: Optional[str] = None


class UnifiedPowerGridModel:
    """Unified power grid model supporting both synthetic and PDN sources.

    Key features:
    - Adapts to both NodeID-based and string-based node representations
    - Supports R-only (synthetic) and RLC (PDN) edge types
    - Multi-net support via net_name parameter
    - Layer-based decomposition for hierarchical solving
    - Preserves original graph structure (no modification)

    Example usage:
        # From synthetic grid
        model = UnifiedPowerGridModel(G, pads, vdd=1.0, source=GridSource.SYNTHETIC)

        # From PDN netlist
        model = UnifiedPowerGridModel(graph, vsrc_nodes, vdd=1.0, source=GridSource.PDN_NETLIST)

        # Solve
        voltages = model.solve_voltages(current_injections)
    """

    def __init__(
        self,
        graph: Union[nx.Graph, nx.MultiDiGraph],
        pad_nodes: Sequence[Any],
        vdd: float = 1.0,
        source: GridSource = GridSource.SYNTHETIC,
        net_name: Optional[str] = None,
        resistance_unit_kohm: bool = False,
    ):
        """Initialize unified power grid model.

        Args:
            graph: NetworkX graph (Graph or MultiDiGraph)
            pad_nodes: Sequence of voltage source nodes (Dirichlet BCs)
            vdd: Nominal supply voltage
            source: Grid source type (SYNTHETIC or PDN_NETLIST)
            net_name: For multi-net PDN, which net this model represents
            resistance_unit_kohm: True if resistance values are in kOhms
        """
        self.graph = graph
        self.pad_nodes = list(pad_nodes)
        self.vdd = float(vdd)
        self.source = source
        self.net_name = net_name
        self.resistance_unit_kohm = resistance_unit_kohm

        # Initialize adapters
        self._node_extractor = NodeInfoExtractor(graph)
        self._edge_extractor = EdgeInfoExtractor(
            is_pdn=(source == GridSource.PDN_NETLIST),
            resistance_unit_kohm=resistance_unit_kohm,
            capacitance_unit_ff=(source == GridSource.PDN_NETLIST),
            inductance_unit_nh=(source == GridSource.PDN_NETLIST),
            current_unit_ma=(source == GridSource.PDN_NETLIST),
        )

        # Track removed islands for diagnostics
        self._removed_island_nodes: Set[Any] = set()
        self._island_stats: Dict[str, Any] = {}

        # Cached edge arrays for fast subgrid building (lazy init)
        self._edge_cache: Optional[EdgeArrayCache] = None

        # Build reduced system
        self._reduced: Optional[UnifiedReducedSystem] = None
        self._build_reduced_system()

    @property
    def reduced(self) -> UnifiedReducedSystem:
        """Get the reduced system for solving."""
        if self._reduced is None:
            self._build_reduced_system()
        return self._reduced

    @property
    def G(self) -> Union[nx.Graph, nx.MultiDiGraph]:
        """Alias for graph (backward compatibility with PowerGridModel)."""
        return self.graph

    def get_node_info(self, node: Any) -> UnifiedNodeInfo:
        """Get unified node information.

        Args:
            node: Node identifier

        Returns:
            UnifiedNodeInfo with coordinates and layer.
        """
        return self._node_extractor.get_info(node)

    def get_node_layer(self, node: Any) -> Optional[LayerID]:
        """Get layer for a node.

        Args:
            node: Node identifier

        Returns:
            Layer identifier (int or str) or None.
        """
        return self._node_extractor.get_layer(node)

    def get_node_xy(self, node: Any) -> Optional[Tuple[float, float]]:
        """Get (x, y) coordinates for a node.

        Args:
            node: Node identifier

        Returns:
            (x, y) tuple or None if unavailable.
        """
        return self._node_extractor.get_xy(node)

    def _detect_and_remove_islands(self, nodes: List[Any]) -> List[Any]:
        """Detect disconnected islands and remove those not connected to voltage sources.

        Islands without voltage sources are "floating" and cannot be solved.
        This method identifies such islands and removes their nodes.

        Args:
            nodes: List of all nodes to consider

        Returns:
            List of nodes with floating island nodes removed
        """
        import warnings
        
        pad_set = set(self.pad_nodes)
        node_set = set(nodes)
        
        # Build adjacency for resistive network only
        # Create a simple undirected graph for connectivity analysis
        resistive_graph = nx.Graph()
        resistive_graph.add_nodes_from(nodes)
        
        for u, v, edge_info in self._iter_resistive_edges():
            if u in node_set and v in node_set:
                resistive_graph.add_edge(u, v)
        
        # Find connected components
        components = list(nx.connected_components(resistive_graph))
        
        if len(components) == 1:
            # Single component - no islands
            self._island_stats = {
                'num_components': 1,
                'islands_removed': 0,
                'nodes_removed': 0,
            }
            return nodes
        
        # Identify which components have voltage sources
        valid_nodes = set()
        islands_removed = 0
        nodes_removed = 0
        
        for component in components:
            has_vsrc = bool(component & pad_set)
            
            if has_vsrc:
                # Keep this component
                valid_nodes.update(component)
            else:
                # Floating island - remove
                islands_removed += 1
                nodes_removed += len(component)
                self._removed_island_nodes.update(component)
        
        if islands_removed > 0:
            warnings.warn(
                f"Removed {islands_removed} floating island(s) with {nodes_removed} nodes "
                f"(not connected to any voltage source). "
                f"Total components: {len(components)}, kept: {len(components) - islands_removed}"
            )
        
        self._island_stats = {
            'num_components': len(components),
            'islands_removed': islands_removed,
            'nodes_removed': nodes_removed,
        }
        
        return [n for n in nodes if n in valid_nodes]

    def _iter_resistive_edges(self):
        """Iterate over resistive edges, handling both Graph and MultiDiGraph.

        Yields:
            Tuples of (u, v, edge_info) for each resistive edge.
        """
        if isinstance(self.graph, nx.MultiDiGraph):
            # PDN MultiDiGraph: iterate over all edges, filter R type
            for u, v, k, data in self.graph.edges(keys=True, data=True):
                if self._edge_extractor.is_resistive_edge(data):
                    edge_info = self._edge_extractor.get_info(data)
                    yield u, v, edge_info
        else:
            # Synthetic Graph: all edges are resistive
            for u, v, data in self.graph.edges(data=True):
                edge_info = self._edge_extractor.get_info(data)
                if edge_info.is_resistive:
                    yield u, v, edge_info

    def _ensure_edge_cache(self) -> EdgeArrayCache:
        """Build or return cached edge arrays for fast subgrid building.
        
        Extracts all resistive edges into NumPy arrays once, enabling O(E) 
        vectorized filtering per tile instead of O(E) Python iteration.
        
        Returns:
            EdgeArrayCache with pre-computed edge arrays.
        """
        if self._edge_cache is not None:
            return self._edge_cache
        
        # Collect all nodes (excluding ground '0')
        all_nodes = [n for n in self.graph.nodes() if n != '0']
        node_to_idx = {n: i for i, n in enumerate(all_nodes)}
        n_nodes = len(all_nodes)
        
        # Pre-allocate lists for edges (will convert to numpy)
        edges_u = []
        edges_v = []
        edges_g = []
        
        GMAX = 1e5
        SHORT_THRESHOLD = 1e-6
        
        for u, v, edge_info in self._iter_resistive_edges():
            R = edge_info.resistance
            if R is None:
                continue
            
            # Skip edges involving ground (handled separately in subgrid)
            if u == '0' or v == '0':
                continue
            
            if u not in node_to_idx or v not in node_to_idx:
                continue
            
            g = GMAX if R <= SHORT_THRESHOLD else 1.0 / R
            
            edges_u.append(node_to_idx[u])
            edges_v.append(node_to_idx[v])
            edges_g.append(g)
        
        self._edge_cache = EdgeArrayCache(
            edge_u_idx=np.array(edges_u, dtype=np.int32),
            edge_v_idx=np.array(edges_v, dtype=np.int32),
            edge_g=np.array(edges_g, dtype=np.float64),
            node_to_idx=node_to_idx,
            idx_to_node=all_nodes,
            n_nodes=n_nodes,
            n_edges=len(edges_u),
        )
        return self._edge_cache

    def _build_subgrid_system_fast(
        self,
        subgrid_nodes: Set[Any],
        dirichlet_nodes: Set[Any],
        dirichlet_voltage: float,
        node_in_subgrid: Optional[np.ndarray] = None,
    ) -> Optional[UnifiedReducedSystem]:
        """Build reduced system for a subgrid using vectorized operations.
        
        This is a faster alternative to _build_subgrid_system that uses
        pre-computed edge arrays and NumPy boolean masking for O(E) filtering
        instead of Python iteration.
        
        Args:
            subgrid_nodes: Set of nodes in the subgrid (including Dirichlet nodes)
            dirichlet_nodes: Set of nodes with fixed voltage (boundary conditions)
            dirichlet_voltage: Default voltage for Dirichlet nodes
            node_in_subgrid: Optional pre-computed boolean array of shape (n_global_nodes,)
                             where True indicates node is in subgrid. If None, computed
                             from subgrid_nodes. Pass this for repeated calls with same
                             subgrid to avoid recomputation.
        
        Returns:
            UnifiedReducedSystem for the subgrid, or None if empty.
        """
        if not subgrid_nodes:
            return None
        
        cache = self._ensure_edge_cache()
        
        # Build boolean mask for nodes in subgrid if not provided
        if node_in_subgrid is None:
            node_in_subgrid = np.zeros(cache.n_nodes, dtype=bool)
            for node in subgrid_nodes:
                idx = cache.node_to_idx.get(node)
                if idx is not None:
                    node_in_subgrid[idx] = True
        
        return self._build_subgrid_system_fast_internal(
            subgrid_nodes=subgrid_nodes,
            dirichlet_nodes=dirichlet_nodes,
            dirichlet_voltage=dirichlet_voltage,
            node_in_subgrid=node_in_subgrid,
            cache=cache,
            skip_lu=False,
        )

    def build_node_membership_mask(self, nodes: Set[Any]) -> np.ndarray:
        """Build a boolean mask array for node membership.
        
        Useful for building multiple subgrid systems from overlapping tiles.
        Pre-compute the mask once per tile, then pass to _build_subgrid_system_fast.
        
        Args:
            nodes: Set of nodes to include
            
        Returns:
            Boolean array of shape (n_global_nodes,) where True = node in set
        """
        cache = self._ensure_edge_cache()
        mask = np.zeros(cache.n_nodes, dtype=bool)
        for node in nodes:
            idx = cache.node_to_idx.get(node)
            if idx is not None:
                mask[idx] = True
        return mask

    def build_tile_systems_batch(
        self,
        tiles: List[Tuple[Set[Any], Set[Any], float]],
        skip_lu: bool = False,
    ) -> List[Optional[UnifiedReducedSystem]]:
        """Build reduced systems for multiple tiles efficiently.
        
        This method is optimized for building many tile systems at once:
        1. Pre-computes edge arrays once (shared across all tiles)
        2. Uses vectorized NumPy operations for edge filtering
        3. Optionally skips LU factorization for tiles that will be further processed
        
        Args:
            tiles: List of (subgrid_nodes, dirichlet_nodes, dirichlet_voltage) tuples
            skip_lu: If True, skip LU factorization (set lu=None). Useful when tiles
                     will be merged or further processed before solving.
        
        Returns:
            List of UnifiedReducedSystem (or None for empty tiles), same order as input.
        
        Example:
            # Define tiles
            tiles = [
                (tile1_nodes, tile1_boundary, 1.0),
                (tile2_nodes, tile2_boundary, 1.0),
                ...
            ]
            systems = model.build_tile_systems_batch(tiles)
        """
        # Ensure cache is built once
        cache = self._ensure_edge_cache()
        
        results = []
        for subgrid_nodes, dirichlet_nodes, dirichlet_voltage in tiles:
            if not subgrid_nodes:
                results.append(None)
                continue
            
            # Build node mask
            node_in_subgrid = np.zeros(cache.n_nodes, dtype=bool)
            for node in subgrid_nodes:
                idx = cache.node_to_idx.get(node)
                if idx is not None:
                    node_in_subgrid[idx] = True
            
            # Use fast builder
            system = self._build_subgrid_system_fast_internal(
                subgrid_nodes=subgrid_nodes,
                dirichlet_nodes=dirichlet_nodes,
                dirichlet_voltage=dirichlet_voltage,
                node_in_subgrid=node_in_subgrid,
                cache=cache,
                skip_lu=skip_lu,
            )
            results.append(system)
        
        return results

    def _build_subgrid_system_fast_internal(
        self,
        subgrid_nodes: Set[Any],
        dirichlet_nodes: Set[Any],
        dirichlet_voltage: float,
        node_in_subgrid: np.ndarray,
        cache: EdgeArrayCache,
        skip_lu: bool = False,
    ) -> Optional[UnifiedReducedSystem]:
        """Internal fast builder with pre-computed cache and mask.
        
        This is the core vectorized implementation used by both
        _build_subgrid_system_fast and build_tile_systems_batch.
        """
        # Filter edges: both endpoints must be in subgrid
        edge_mask = node_in_subgrid[cache.edge_u_idx] & node_in_subgrid[cache.edge_v_idx]
        
        if not np.any(edge_mask):
            return None
        
        # Extract filtered edges
        filtered_u = cache.edge_u_idx[edge_mask]
        filtered_v = cache.edge_v_idx[edge_mask]
        filtered_g = cache.edge_g[edge_mask]
        n_filtered_edges = len(filtered_g)
        
        # Build local node indexing for subgrid
        global_indices_in_subgrid = np.unique(np.concatenate([filtered_u, filtered_v]))
        n_local_nodes = len(global_indices_in_subgrid)
        
        if n_local_nodes == 0:
            return None
        
        # Map global index -> local index
        global_to_local = np.full(cache.n_nodes, -1, dtype=np.int32)
        global_to_local[global_indices_in_subgrid] = np.arange(n_local_nodes, dtype=np.int32)
        
        # Convert edge indices to local
        local_u = global_to_local[filtered_u]
        local_v = global_to_local[filtered_v]
        
        # Build COO data for conductance matrix
        n_offdiag = 2 * n_filtered_edges
        offdiag_rows = np.empty(n_offdiag, dtype=np.int32)
        offdiag_cols = np.empty(n_offdiag, dtype=np.int32)
        offdiag_data = np.empty(n_offdiag, dtype=np.float64)
        
        offdiag_rows[0::2] = local_u
        offdiag_rows[1::2] = local_v
        offdiag_cols[0::2] = local_v
        offdiag_cols[1::2] = local_u
        offdiag_data[0::2] = -filtered_g
        offdiag_data[1::2] = -filtered_g
        
        # Diagonal: accumulate via bincount
        diag_contrib_u = np.bincount(local_u, weights=filtered_g, minlength=n_local_nodes)
        diag_contrib_v = np.bincount(local_v, weights=filtered_g, minlength=n_local_nodes)
        diag_values = diag_contrib_u + diag_contrib_v
        
        diag_rows = np.arange(n_local_nodes, dtype=np.int32)
        diag_cols = np.arange(n_local_nodes, dtype=np.int32)
        
        # Combine
        all_rows = np.concatenate([offdiag_rows, diag_rows])
        all_cols = np.concatenate([offdiag_cols, diag_cols])
        all_data = np.concatenate([offdiag_data, diag_values])
        
        G_mat = sp.csr_matrix(
            (all_data, (all_rows, all_cols)),
            shape=(n_local_nodes, n_local_nodes)
        )
        
        # Build node lists
        nodes = [cache.idx_to_node[gi] for gi in global_indices_in_subgrid]
        index = {n: i for i, n in enumerate(nodes)}
        
        # Partition into unknown and Dirichlet
        unknown_nodes = [n for n in nodes if n not in dirichlet_nodes]
        dirichlet_ordered = [n for n in nodes if n in dirichlet_nodes]
        
        if not unknown_nodes:
            return None
        
        index_unknown = {n: i for i, n in enumerate(unknown_nodes)}
        
        # Extract submatrices
        u_idx = np.array([index[n] for n in unknown_nodes], dtype=np.int32)
        d_idx = np.array([index[n] for n in dirichlet_ordered], dtype=np.int32) if dirichlet_ordered else np.array([], dtype=np.int32)
        
        G_uu = G_mat[u_idx][:, u_idx].tocsr()
        G_ud = G_mat[u_idx][:, d_idx].tocsr() if len(d_idx) > 0 else sp.csr_matrix((len(u_idx), 0))
        
        # LU factorization (optional)
        if skip_lu:
            lu = None
        else:
            lu = spla.factorized(G_uu.tocsc())
        
        return UnifiedReducedSystem(
            node_order=nodes,
            unknown_nodes=unknown_nodes,
            pad_nodes=dirichlet_ordered,
            G_uu=G_uu,
            G_up=G_ud,
            lu=lu,
            pad_voltage=dirichlet_voltage,
            index_of=index,
            index_unknown=index_unknown,
            net_name=self.net_name,
        )

    @property
    def edge_cache(self) -> EdgeArrayCache:
        """Access the edge array cache (builds if not already cached)."""
        return self._ensure_edge_cache()

    def _build_conductance_matrix(self) -> Tuple[sp.csr_matrix, List[Any]]:
        """Build conductance matrix from resistive edges.

        The ground node '0' is excluded from the matrix - edges to ground
        contribute to the diagonal of the connected node (grounding conductance).

        Returns:
            (G_matrix, node_order) where G_matrix is the nodal conductance matrix.
        """
        # Exclude ground node '0' from the matrix
        nodes = [n for n in self.graph.nodes() if n != '0']
        index = {n: i for i, n in enumerate(nodes)}
        n_nodes = len(nodes)

        data, rows, cols = [], [], []
        diag = np.zeros(n_nodes, dtype=float)

        # Constants for handling short resistances (matching PDNSolver)
        GMAX = 1e5  # Maximum conductance for shorts (mS for PDN, S for synthetic)
        SHORT_THRESHOLD = 1e-6  # Resistance threshold for shorts (kOhm for PDN)

        for u, v, edge_info in self._iter_resistive_edges():
            R = edge_info.resistance
            if R is None:
                continue

            # Handle zero/short resistance like PDNSolver
            if R <= 0 or R < SHORT_THRESHOLD:
                g = GMAX
            else:
                g = 1.0 / R
            
            # Check if either node is ground
            u_is_ground = (u == '0')
            v_is_ground = (v == '0')
            
            if u_is_ground and v_is_ground:
                # Edge between ground nodes - skip
                continue
            elif u_is_ground:
                # Edge from ground to v: adds grounding conductance to v
                if v in index:
                    diag[index[v]] += g
            elif v_is_ground:
                # Edge from u to ground: adds grounding conductance to u
                if u in index:
                    diag[index[u]] += g
            else:
                # Normal edge between two non-ground nodes
                if u not in index or v not in index:
                    continue
                iu, iv = index[u], index[v]

                # Off-diagonal entries (symmetric)
                rows.extend([iu, iv])
                cols.extend([iv, iu])
                data.extend([-g, -g])

                # Diagonal accumulation
                diag[iu] += g
                diag[iv] += g

        # Add diagonal entries
        for i in range(n_nodes):
            rows.append(i)
            cols.append(i)
            data.append(diag[i])

        G_mat = sp.csr_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))
        return G_mat, nodes

    def _build_reduced_system(self) -> None:
        """Build reduced system using Schur complement.

        Eliminates pad nodes (fixed voltage) to form reduced system:
        G_uu * V_u = I_u - G_up * V_p
        
        Also detects and removes floating islands (disconnected components
        not connected to any voltage source).
        """
        G_mat, all_nodes = self._build_conductance_matrix()
        
        # Detect and remove floating islands
        nodes = self._detect_and_remove_islands(all_nodes)
        
        pad_set = set(self.pad_nodes)

        unknown_nodes = [n for n in nodes if n not in pad_set]
        pad_nodes_ordered = [n for n in nodes if n in pad_set]

        # Build index for valid (non-island) nodes only
        index = {n: i for i, n in enumerate(nodes)}
        index_unknown = {n: i for i, n in enumerate(unknown_nodes)}

        # Re-extract submatrices for valid nodes only
        # We need to rebuild the matrix with only valid nodes
        if self._removed_island_nodes:
            # Rebuild conductance matrix for valid nodes only (excluding ground '0')
            n_nodes = len(nodes)
            data, rows, cols = [], [], []
            diag = np.zeros(n_nodes, dtype=float)
            node_set = set(nodes)
            
            # Constants for handling short resistances (matching PDNSolver)
            GMAX = 1e5  # Maximum conductance for shorts
            SHORT_THRESHOLD = 1e-6  # Resistance threshold for shorts (kOhm)

            for u, v, edge_info in self._iter_resistive_edges():
                R = edge_info.resistance
                if R is None:
                    continue

                # Handle zero/short resistance like PDNSolver
                if R <= 0 or R < SHORT_THRESHOLD:
                    g = GMAX
                else:
                    g = 1.0 / R
                
                # Handle ground node
                u_is_ground = (u == '0')
                v_is_ground = (v == '0')
                
                if u_is_ground and v_is_ground:
                    continue
                elif u_is_ground:
                    if v in node_set and v in index:
                        diag[index[v]] += g
                elif v_is_ground:
                    if u in node_set and u in index:
                        diag[index[u]] += g
                else:
                    if u not in node_set or v not in node_set:
                        continue
                    iu, iv = index[u], index[v]

                    rows.extend([iu, iv])
                    cols.extend([iv, iu])
                    data.extend([-g, -g])

                    diag[iu] += g
                    diag[iv] += g

            for i in range(n_nodes):
                rows.append(i)
                cols.append(i)
                data.append(diag[i])

            G_mat = sp.csr_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))

        # Extract submatrices
        u_idx = [index[n] for n in unknown_nodes]
        p_idx = [index[n] for n in pad_nodes_ordered]

        if len(u_idx) == 0:
            # No unknowns - all nodes are pads
            G_uu = sp.csr_matrix((0, 0))
            G_up = sp.csr_matrix((0, len(p_idx)))
            lu = lambda x: np.array([])
        else:
            G_uu = G_mat[np.ix_(u_idx, u_idx)].tocsr()
            G_up = G_mat[np.ix_(u_idx, p_idx)].tocsr() if p_idx else sp.csr_matrix((len(u_idx), 0))

            # Factorize for fast solves
            lu = spla.factorized(G_uu.tocsc())

        self._reduced = UnifiedReducedSystem(
            node_order=nodes,
            unknown_nodes=unknown_nodes,
            pad_nodes=pad_nodes_ordered,
            G_uu=G_uu,
            G_up=G_up,
            lu=lu,
            pad_voltage=self.vdd,
            index_of=index,
            index_unknown=index_unknown,
            net_name=self.net_name,
        )

    @property
    def island_stats(self) -> Dict[str, Any]:
        """Get statistics about detected islands.
        
        Returns:
            Dict with keys: num_components, islands_removed, nodes_removed
        """
        return self._island_stats.copy()

    @property
    def removed_island_nodes(self) -> Set[Any]:
        """Get set of nodes removed due to being in floating islands."""
        return self._removed_island_nodes.copy()

    def extract_current_sources(self) -> Dict[Any, float]:
        """Extract current source injections from PDN graph.
        
        For PDN netlists, current sources are stored as edges with type='I'.
        Current flows from positive terminal (u) to negative terminal (v).
        For power delivery, current sources represent load current sinking
        from the power net to ground ('0').
        
        Returns:
            Dict mapping node -> current (positive = sink from power net)
            Units match the source: mA for PDN sources (native kOhm/mA/mS unit system).
        """
        if self.source != GridSource.PDN_NETLIST:
            return {}
        
        if not isinstance(self.graph, nx.MultiDiGraph):
            return {}
        
        current_injections = {}
        valid_nodes = set(self.reduced.node_order)  # Only nodes not in floating islands
        
        for u, v, data in self.graph.edges(data=True):
            if data.get('type') != 'I':
                continue
            
            current_ma = data.get('value', 0.0)
            if current_ma == 0:
                continue
            
            # Current flows from u to v
            # For power delivery: current source from net node (u) to ground (v='0')
            # This represents load current sinking from the power net
            if u in valid_nodes and u != '0':
                # Current is pulled OUT of the power net at u
                current_injections[u] = current_injections.get(u, 0.0) + current_ma
            elif v in valid_nodes and v != '0':
                # If edge is reversed (ground to net node), current enters at v
                # This is less common but handle it for completeness
                current_injections[v] = current_injections.get(v, 0.0) + current_ma
        
        return current_injections

    def solve_voltages(self, current_injections: Dict[Any, float]) -> Dict[Any, float]:
        """Solve for nodal voltages given current injections.

        Args:
            current_injections: Dict mapping node -> current (positive = sink)

        Returns:
            Dict mapping node -> voltage
        """
        rs = self.reduced

        if len(rs.unknown_nodes) == 0:
            # All nodes are pads
            return {n: rs.pad_voltage for n in rs.pad_nodes}

        # Build RHS: I_u (current injections at unknown nodes)
        I_u = np.zeros(len(rs.unknown_nodes), dtype=float)
        for n, cur in current_injections.items():
            if n in rs.index_unknown:
                # Sink current is positive input, but nodal equation uses negative injection
                I_u[rs.index_unknown[n]] += -float(cur)

        # Pad voltage contribution: -G_up * V_p
        V_p = np.full(len(rs.pad_nodes), rs.pad_voltage, dtype=float)
        if rs.G_up.shape[1] > 0:
            rhs = I_u - rs.G_up @ V_p
        else:
            rhs = I_u

        # Solve: V_u = lu(rhs)
        V_u = rs.lu(rhs)

        # Assemble result
        voltages = {}
        for n in rs.pad_nodes:
            voltages[n] = rs.pad_voltage
        for i, n in enumerate(rs.unknown_nodes):
            voltages[n] = float(V_u[i])

        return voltages

    def ir_drop(self, voltages: Dict[Any, float], vdd: Optional[float] = None) -> Dict[Any, float]:
        """Compute IR-drop from voltages.

        Args:
            voltages: Dict mapping node -> voltage
            vdd: Reference voltage (defaults to self.vdd)

        Returns:
            Dict mapping node -> IR-drop (vdd - voltage)
        """
        if vdd is None:
            vdd = self.vdd
        return {n: vdd - v for n, v in voltages.items()}

    # Layer-based decomposition methods

    def get_nodes_at_layer(self, layer: LayerID) -> Set[Any]:
        """Get all nodes at a specific layer.

        Args:
            layer: Layer identifier (int or str)

        Returns:
            Set of nodes at the specified layer.
        """
        result = set()
        for node in self.graph.nodes():
            info = self.get_node_info(node)
            node_layer = info.layer

            # Direct match
            if node_layer == layer:
                result.add(node)
            # String/int comparison
            elif str(node_layer) == str(layer):
                result.add(node)
            # Numeric comparison for mixed types
            elif isinstance(layer, int) and info.layer_numeric == layer:
                result.add(node)

        return result

    def get_all_layers(self) -> List[LayerID]:
        """Get all unique layers in the grid, sorted.

        Returns:
            List of layer identifiers, sorted numerically if possible.
        """
        layers = set()
        for node in self.graph.nodes():
            info = self.get_node_info(node)
            if info.layer is not None:
                layers.add(info.layer)

        # Sort: try numeric first, then string
        def sort_key(x):
            if isinstance(x, int):
                return (0, x, '')
            if isinstance(x, str) and x.isdigit():
                return (0, int(x), '')
            return (1, 0, str(x))

        return sorted(layers, key=sort_key)

    def _decompose_at_layer(
        self,
        partition_layer: LayerID,
    ) -> Tuple[Set[Any], Set[Any], Set[Any], List[Tuple[Any, Any, Any]]]:
        """Decompose grid at partition layer into top/bottom grids.

        Handles package nodes (layer=None) by including them in the top-grid
        if they are connected to the die via resistive paths. This preserves
        the path from die nodes through package nodes to voltage sources.

        Args:
            partition_layer: Layer index/name to partition at

        Returns:
            (top_grid_nodes, bottom_grid_nodes, port_nodes, via_edges)
            - top_grid_nodes: Nodes at layers >= partition_layer, plus pad nodes
              and package nodes connected to top-grid via R-edges
            - bottom_grid_nodes: Nodes at layers < partition_layer
            - port_nodes: Nodes at partition_layer connected to layer below
            - via_edges: Edges connecting partition_layer to layer below
        """
        all_layers = self.get_all_layers()

        # Find partition layer index
        partition_idx = None
        for i, layer in enumerate(all_layers):
            if layer == partition_layer or str(layer) == str(partition_layer):
                partition_idx = i
                break

        if partition_idx is None:
            raise ValueError(f"Partition layer {partition_layer} not found in grid")
        if partition_idx == 0:
            raise ValueError(f"Cannot partition at bottom layer (layer={partition_layer})")

        # Categorize nodes by layer
        top_layers = set(all_layers[partition_idx:])
        bottom_layers = set(all_layers[:partition_idx])
        partition_layer_below = all_layers[partition_idx - 1]

        top_grid_nodes = set()
        bottom_grid_nodes = set()
        package_nodes = set()  # Nodes with layer=None

        # Get nodes to exclude (floating islands that were removed during model construction)
        excluded_nodes = self._removed_island_nodes

        # First pass: categorize die nodes (with layer info)
        for node in self.graph.nodes():
            # Skip floating island nodes
            if node in excluded_nodes:
                continue
                
            info = self.get_node_info(node)
            node_layer = info.layer

            if node_layer is None:
                # Track package nodes for later processing
                package_nodes.add(node)
                continue

            # Check membership
            in_top = node_layer in top_layers or str(node_layer) in [str(l) for l in top_layers]
            in_bottom = node_layer in bottom_layers or str(node_layer) in [str(l) for l in bottom_layers]

            if in_top:
                top_grid_nodes.add(node)
            elif in_bottom:
                bottom_grid_nodes.add(node)

        # Second pass: assign package nodes (layer=None) to top-grid if connected to top-grid
        # This includes pad nodes and intermediate package nodes (tap, probe, int, vsrc)
        # Use BFS from top-grid nodes to find connected package nodes via R-type edges
        pad_set = set(self.pad_nodes)
        
        # Always include pad nodes in top-grid (they are voltage sources)
        for pad in pad_set:
            if pad in package_nodes:
                top_grid_nodes.add(pad)
        
        # BFS to find package nodes connected to top-grid via R-edges
        # NOTE: For directed graphs (MultiDiGraph), we must check BOTH outgoing
        # and incoming edges since via edges may be directed (e.g., tap→M5 only)
        from collections import deque
        visited_pkg = set()
        queue = deque()
        
        is_directed = isinstance(self.graph, nx.MultiDiGraph)
        
        # Helper to get all R-type neighbors (both directions for directed graphs)
        def get_r_neighbors(node):
            neighbors = set()
            # Outgoing edges
            for u, v, d in self.graph.edges(node, data=True):
                if d.get('type') == 'R':
                    neighbors.add(v)
            # Incoming edges (for directed graphs)
            if is_directed:
                for u, v, d in self.graph.in_edges(node, data=True):
                    if d.get('type') == 'R':
                        neighbors.add(u)
            return neighbors
        
        # Start from top-grid die nodes that have R-edges to package nodes
        # Use list() to avoid "Set changed size during iteration" error
        for node in list(top_grid_nodes):
            if node in package_nodes:
                continue  # Skip package nodes already added
            for neighbor in get_r_neighbors(node):
                if neighbor in package_nodes and neighbor not in visited_pkg:
                    visited_pkg.add(neighbor)
                    queue.append(neighbor)
                    top_grid_nodes.add(neighbor)
        
        # Continue BFS within package nodes
        while queue:
            pkg_node = queue.popleft()
            for neighbor in get_r_neighbors(pkg_node):
                if neighbor in package_nodes and neighbor not in visited_pkg:
                    visited_pkg.add(neighbor)
                    queue.append(neighbor)
                    top_grid_nodes.add(neighbor)

        # Find port nodes and via edges
        # Port nodes are top-grid nodes that have R-edges to bottom-grid nodes
        # This handles cases where edges skip layers (e.g., layer 28 -> layer 32)
        port_nodes = set()
        via_edges = []

        for u, v, edge_info in self._iter_resistive_edges():
            u_in_top = u in top_grid_nodes
            v_in_top = v in top_grid_nodes
            u_in_bottom = u in bottom_grid_nodes
            v_in_bottom = v in bottom_grid_nodes

            # Edge connects top-grid to bottom-grid
            if u_in_top and v_in_bottom:
                via_edges.append((u, v, edge_info))
                port_nodes.add(u)
            elif v_in_top and u_in_bottom:
                via_edges.append((u, v, edge_info))
                port_nodes.add(v)

        return top_grid_nodes, bottom_grid_nodes, port_nodes, via_edges

    def _build_subgrid_system(
        self,
        subgrid_nodes: Set[Any],
        dirichlet_nodes: Set[Any],
        dirichlet_voltage: float,
    ) -> Optional[UnifiedReducedSystem]:
        """Build a reduced system for a subgrid with Dirichlet boundary nodes.

        Args:
            subgrid_nodes: Set of nodes in the subgrid (including Dirichlet nodes)
            dirichlet_nodes: Set of nodes with fixed voltage (boundary conditions)
            dirichlet_voltage: Default voltage for Dirichlet nodes

        Returns:
            UnifiedReducedSystem for the subgrid, or None if empty.
        """
        if not subgrid_nodes:
            return None

        # Filter edges to subgrid
        subgrid_edges = []
        for u, v, edge_info in self._iter_resistive_edges():
            if u in subgrid_nodes and v in subgrid_nodes:
                subgrid_edges.append((u, v, edge_info))

        if not subgrid_edges:
            return None

        # Build conductance matrix for subgrid
        nodes = list(subgrid_nodes)
        index = {n: i for i, n in enumerate(nodes)}
        n_nodes = len(nodes)

        data, rows, cols = [], [], []
        diag = np.zeros(n_nodes, dtype=float)

        # Constants for handling short resistances (matching PDNSolver)
        GMAX = 1e5  # Maximum conductance for shorts (mS for PDN, S for synthetic)
        SHORT_THRESHOLD = 1e-6  # Resistance threshold for shorts (kOhm for PDN)

        for u, v, edge_info in subgrid_edges:
            R = edge_info.resistance
            if R is None:
                continue

            # Handle short resistances (R=0 or very small R)
            if R <= SHORT_THRESHOLD:
                g = GMAX
            else:
                g = 1.0 / R

            iu, iv = index[u], index[v]

            rows.extend([iu, iv])
            cols.extend([iv, iu])
            data.extend([-g, -g])

            diag[iu] += g
            diag[iv] += g

        for i in range(n_nodes):
            rows.append(i)
            cols.append(i)
            data.append(diag[i])

        G_mat = sp.csr_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))

        # Partition into unknown and Dirichlet
        unknown_nodes = [n for n in nodes if n not in dirichlet_nodes]
        dirichlet_ordered = [n for n in nodes if n in dirichlet_nodes]

        if not unknown_nodes:
            return None

        index_unknown = {n: i for i, n in enumerate(unknown_nodes)}

        u_idx = [index[n] for n in unknown_nodes]
        d_idx = [index[n] for n in dirichlet_ordered]

        G_uu = G_mat[np.ix_(u_idx, u_idx)].tocsr()
        G_ud = G_mat[np.ix_(u_idx, d_idx)].tocsr() if d_idx else sp.csr_matrix((len(u_idx), 0))

        lu = spla.factorized(G_uu.tocsc())

        return UnifiedReducedSystem(
            node_order=nodes,
            unknown_nodes=unknown_nodes,
            pad_nodes=dirichlet_ordered,
            G_uu=G_uu,
            G_up=G_ud,
            lu=lu,
            pad_voltage=dirichlet_voltage,
            index_of=index,
            index_unknown=index_unknown,
            net_name=self.net_name,
        )

    def _solve_subgrid(
        self,
        reduced_system: UnifiedReducedSystem,
        current_injections: Dict[Any, float],
        dirichlet_voltages: Optional[Dict[Any, float]] = None,
    ) -> Dict[Any, float]:
        """Solve a subgrid system with given currents and boundary voltages.

        Args:
            reduced_system: UnifiedReducedSystem for the subgrid
            current_injections: Current injections at nodes (positive = sink)
            dirichlet_voltages: Optional custom voltages for Dirichlet nodes

        Returns:
            Dict mapping node -> voltage for all subgrid nodes
        """
        rs = reduced_system

        if len(rs.unknown_nodes) == 0:
            return {n: rs.pad_voltage for n in rs.pad_nodes}

        # Build current vector
        I_u = np.zeros(len(rs.unknown_nodes), dtype=float)
        for n, cur in current_injections.items():
            if n in rs.index_unknown:
                I_u[rs.index_unknown[n]] += -float(cur)

        # Get Dirichlet voltages
        if dirichlet_voltages is not None:
            V_d = np.array([dirichlet_voltages.get(n, rs.pad_voltage) for n in rs.pad_nodes])
        else:
            V_d = np.full(len(rs.pad_nodes), rs.pad_voltage)

        # Solve
        if rs.G_up.shape[1] > 0:
            rhs = I_u - rs.G_up @ V_d
        else:
            rhs = I_u

        V_u = rs.lu(rhs)

        # Assemble result
        voltages = {}
        for i, n in enumerate(rs.pad_nodes):
            voltages[n] = float(V_d[i])
        for i, n in enumerate(rs.unknown_nodes):
            voltages[n] = float(V_u[i])

        return voltages

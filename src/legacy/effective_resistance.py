"""Effective resistance computation for power grid networks.

Computes the effective resistance R_eff between node pairs in the power grid.
For a resistive network with conductance matrix G, the effective resistance
between nodes u and v is:

    R_eff(u,v) = (e_u - e_v)^T * G^(-1) * (e_u - e_v)

where e_u, e_v are unit basis vectors. This simplifies to:
    R_eff(u,v) = G^(-1)[u,u] + G^(-1)[v,v] - 2*G^(-1)[u,v]

For computing effective resistance to ground (voltage sources), we use the
reduced system where pads are eliminated:
    R_eff(u, pads) = (G_uu^(-1))[u,u]

This module provides efficient batch computation by solving multiple sparse
linear systems simultaneously.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .power_grid_model import PowerGridModel, ReducedSystem


class EffectiveResistanceCalculator:
    """Computes effective resistance between node pairs in a power grid.
    
    Uses the reduced conductance matrix from PowerGridModel to efficiently
    compute effective resistances in batch mode.
    """

    def __init__(self, model: PowerGridModel):
        """Initialize calculator with a power grid model.
        
        Args:
            model: PowerGridModel instance with pre-computed reduced system
        """
        self.model = model
        self.reduced: ReducedSystem = model.reduced
        
    def compute_batch(
        self, 
        pairs: Union[Sequence[Tuple], np.ndarray]
    ) -> np.ndarray:
        """Compute effective resistance for a batch of node pairs.
        
        Args:
            pairs: List or array of tuples (u, v) where:
                - u is a NodeID (source node)
                - v is a NodeID or None:
                    - If v is None: compute R_eff from u to ground (pads)
                    - If v is NodeID: compute R_eff between u and v
                    
        Returns:
            numpy array of effective resistances, same length as pairs,
            with R_eff[k] corresponding to pairs[k]
            
        Raises:
            ValueError: if a node is not in the unknown set (e.g., pad node)
            
        Example:
            >>> calc = EffectiveResistanceCalculator(model)
            >>> pairs = [(node1, node2), (node3, None), (node4, node5)]
            >>> reff = calc.compute_batch(pairs)
        """
        if len(pairs) == 0:
            return np.array([])
            
        # Separate ground and node-to-node pairs
        ground_pairs = []
        node_pairs = []
        pair_types = []  # Track which type each original pair is
        
        for u, v in pairs:
            if v is None:
                ground_pairs.append(u)
                pair_types.append('ground')
            else:
                node_pairs.append((u, v))
                pair_types.append('node')
        
        # Compute ground resistances
        ground_results = {}
        if ground_pairs:
            ground_results = self._compute_ground_batch(ground_pairs)
            
        # Compute node-to-node resistances
        node_results = {}
        if node_pairs:
            node_results = self._compute_node_to_node_batch(node_pairs)
            
        # Assemble results in original order
        results = np.zeros(len(pairs))
        ground_idx = 0
        node_idx = 0
        
        for i, pair_type in enumerate(pair_types):
            if pair_type == 'ground':
                u = ground_pairs[ground_idx]
                results[i] = ground_results[u]
                ground_idx += 1
            else:
                u, v = node_pairs[node_idx]
                results[i] = node_results[(u, v)]
                node_idx += 1
                
        return results
    
    def _compute_ground_batch(self, nodes: List) -> dict:
        """Compute R_eff from each node to ground (pads).
        
        For node u: R_eff(u, ground) = (G_uu^(-1))[u,u]
        
        Args:
            nodes: List of NodeID objects
            
        Returns:
            dict mapping node -> R_eff to ground
        """
        rs = self.reduced
        results = {}
        
        # Build right-hand sides: unit vectors for each node
        n_unknown = len(rs.unknown_nodes)
        rhs_matrix = np.zeros((n_unknown, len(nodes)))
        
        node_indices = []
        for i, node in enumerate(nodes):
            if node not in rs.index_unknown:
                if node in set(rs.pad_nodes):
                    # Pad nodes have zero resistance to ground
                    results[node] = 0.0
                    continue
                else:
                    raise ValueError(f"Node {node} not found in grid")
            idx = rs.index_unknown[node]
            node_indices.append((i, idx, node))
            rhs_matrix[idx, i] = 1.0
        
        if node_indices:
            # Solve G_uu * X = RHS for all columns at once
            # Each column of X gives G_uu^(-1) * e_i
            solutions = np.zeros((n_unknown, len(node_indices)))
            for i in range(len(node_indices)):
                solutions[:, i] = rs.lu(rhs_matrix[:, node_indices[i][0]])
            
            # Extract diagonal elements
            for i, (_, idx, node) in enumerate(node_indices):
                results[node] = solutions[idx, i]
                
        return results
    
    def _compute_node_to_node_batch(self, pairs: List[Tuple]) -> dict:
        """Compute R_eff between pairs of nodes.
        
        For nodes u, v: R_eff(u,v) = G^(-1)[u,u] + G^(-1)[v,v] - 2*G^(-1)[u,v]
        
        To compute this efficiently, we solve:
            G_uu * X_u = e_u for each unique u
            G_uu * X_v = e_v for each unique v
        Then: R_eff(u,v) = X_u[u] + X_v[v] - X_u[v] - X_v[u]
        
        Args:
            pairs: List of (u, v) tuples where both are NodeID
            
        Returns:
            dict mapping (u, v) -> R_eff
        """
        rs = self.reduced
        results = {}
        
        # Collect unique nodes and validate
        unique_nodes = set()
        for u, v in pairs:
            unique_nodes.add(u)
            unique_nodes.add(v)
        
        # Check for pad nodes
        pad_set = set(rs.pad_nodes)
        for node in unique_nodes:
            if node in pad_set:
                raise ValueError(
                    f"Cannot compute effective resistance for pad node {node}. "
                    "Pad nodes are fixed voltage sources."
                )
            if node not in rs.index_unknown:
                raise ValueError(f"Node {node} not found in unknown set")
        
        # Build RHS matrix: one column per unique node
        unique_list = list(unique_nodes)
        n_unknown = len(rs.unknown_nodes)
        rhs_matrix = np.zeros((n_unknown, len(unique_list)))
        
        node_to_col = {}
        for i, node in enumerate(unique_list):
            idx = rs.index_unknown[node]
            rhs_matrix[idx, i] = 1.0
            node_to_col[node] = i
        
        # Solve all systems: G_uu * X = RHS
        # X[:, i] = G_uu^(-1) * e_i
        X = np.zeros((n_unknown, len(unique_list)))
        for i in range(len(unique_list)):
            X[:, i] = rs.lu(rhs_matrix[:, i])
        
        # Compute R_eff for each pair
        for u, v in pairs:
            col_u = node_to_col[u]
            col_v = node_to_col[v]
            idx_u = rs.index_unknown[u]
            idx_v = rs.index_unknown[v]
            
            # R_eff(u,v) = X_u[u] + X_v[v] - X_u[v] - X_v[u]
            # Equivalently: = X[idx_u, col_u] + X[idx_v, col_v] - X[idx_v, col_u] - X[idx_u, col_v]
            reff = (X[idx_u, col_u] + X[idx_v, col_v] - 
                    X[idx_v, col_u] - X[idx_u, col_v])
            results[(u, v)] = reff
            
        return results
    
    def compute_single(self, u, v=None) -> float:
        """Compute effective resistance for a single pair (convenience method).
        
        Args:
            u: NodeID of source node
            v: NodeID of target node, or None for resistance to ground
            
        Returns:
            Effective resistance as a float
        """
        result = self.compute_batch([(u, v)])
        return float(result[0])

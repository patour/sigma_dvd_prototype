"""Regional voltage solver for partitioned power grids.

Computes DC voltage at a subset of load nodes S within a partition/region R
using effective resistance matrices and boundary/separator nodes A.

The solver uses the following algorithm:
1. Build resistance matrices K_A, K_SA, K_FA, K_S from effective resistances
2. Compute boundary voltages b_A due to far loads F
3. Solve for boundary currents j_A 
4. Compute voltage at S from far loads (v_far) and near loads (v_near)
5. Return total voltage v = v_near + v_far

This approach allows efficient voltage computation within a region by treating
loads outside the region as equivalent boundary currents.
"""

from __future__ import annotations

from typing import Dict, List, Set, Optional
import numpy as np
import scipy.linalg as la

from .effective_resistance import EffectiveResistanceCalculator
from .power_grid_model import PowerGridModel


class RegionalVoltageSolver:
    """Solves for DC voltage at nodes within a partitioned region.
    
    Uses effective resistance matrices and boundary conditions to compute
    voltages efficiently for a subset of load nodes.
    """
    
    def __init__(
        self, 
        calc: EffectiveResistanceCalculator,
        model: PowerGridModel
    ):
        """Initialize the regional voltage solver.
        
        Args:
            calc: EffectiveResistanceCalculator for computing R_eff
            model: PowerGridModel containing the grid structure
        """
        self.calc = calc
        self.model = model
        
    def compute_voltages(
        self,
        S: Set,
        R: Set,
        A: Set,
        I_S: Dict,
        I_F: Dict
    ) -> Dict[any, float]:
        """Compute DC voltage at nodes in subset S.
        
        Args:
            S: Set of target load nodes (subset of region R)
            R: Set of all nodes in the region containing S
            A: Set of boundary/separator nodes (ports) for region R
            I_S: Dict mapping nodes in S to their current values (near loads)
            I_F: Dict mapping nodes NOT in S to their current values (far loads)
            
        Returns:
            Dictionary mapping each node in S to its computed voltage
            
        Raises:
            ValueError: If sets are inconsistent or empty
        """
        # Validate inputs
        if not S:
            raise ValueError("Subset S cannot be empty")
        if not A:
            raise ValueError("Boundary set A cannot be empty")
        if not S.issubset(R):
            raise ValueError("Subset S must be contained in region R")
        
        # Convert sets to sorted lists for consistent indexing
        A_list = sorted(list(A), key=lambda n: (n.layer, n.idx))
        S_list = sorted(list(S), key=lambda n: (n.layer, n.idx))
        F_list = sorted(list(I_F.keys()), key=lambda n: (n.layer, n.idx))
        
        # Step 1: Build resistance matrices
        K_A = self._build_K_A(A_list)
        K_SA = self._build_K_SA(S_list, A_list)
        K_FA = self._build_K_FA(F_list, A_list)
        K_S = self._build_K_S(S_list)
        
        # Step 2: Compute b_A - voltage at boundary due to far loads
        I_F_vec = np.array([I_F[node] for node in F_list])
        b_A = K_FA @ I_F_vec
        
        # Step 3: Solve K_A @ j_A = b_A for boundary currents
        # Use Cholesky factorization for symmetric positive definite K_A
        try:
            L = la.cholesky(K_A, lower=True)
            j_A = la.cho_solve((L, True), b_A)
        except la.LinAlgError:
            # Fallback to LU if Cholesky fails (shouldn't happen for valid grids)
            j_A = la.solve(K_A, b_A)
        
        # Step 4: Compute voltage DROP at S due to far loads
        v_far_drop = {}
        for u in S_list:
            v_far_drop[u] = np.dot(K_SA[u], j_A)
        
        # Step 5: Compute voltage DROP at S due to near loads  
        I_S_vec = np.array([I_S.get(node, 0.0) for node in S_list])
        v_near_drop = {}
        for i, u in enumerate(S_list):
            v_near_drop[u] = np.dot(K_S[i], I_S_vec)
        
        # Step 6: Compute final voltage
        # The K matrices compute IR drops (R*I), so voltage = Vdd - IR_drop
        vdd = self.model.vdd
        v_total = {}
        for u in S_list:
            total_drop = v_near_drop[u] + v_far_drop[u]
            v_total[u] = vdd - total_drop
        
        return v_total
    
    def _build_K_A(self, A_list: List) -> np.ndarray:
        """Build the K_A resistance matrix for boundary nodes.
        
        K_A[p,q] = 0.5 * (R_eff(p,0) + R_eff(q,0) - R_eff(p,q))
        
        Args:
            A_list: Sorted list of boundary nodes
            
        Returns:
            |A| x |A| numpy array
        """
        n = len(A_list)
        K_A = np.zeros((n, n))
        
        # Compute R_eff to ground for all nodes in A
        ground_pairs = [(node, None) for node in A_list]
        R_ground = self.calc.compute_batch(ground_pairs)
        
        # Compute R_eff between all pairs in A
        for i, p in enumerate(A_list):
            for j, q in enumerate(A_list):
                if i == j:
                    # Diagonal: R_eff(p,p) = 0
                    K_A[i, j] = R_ground[i]
                else:
                    # Off-diagonal: compute R_eff(p,q)
                    R_pq = self.calc.compute_batch([(p, q)])[0]
                    K_A[i, j] = 0.5 * (R_ground[i] + R_ground[j] - R_pq)
        
        return K_A
    
    def _build_K_SA(self, S_list: List, A_list: List) -> Dict[any, np.ndarray]:
        """Build the K_SA resistance matrix mapping S to A.
        
        K_SA[u,p] = 0.5 * (R_eff(u,0) + R_eff(p,0) - R_eff(u,p))
        
        Args:
            S_list: Sorted list of nodes in S
            A_list: Sorted list of boundary nodes
            
        Returns:
            Dictionary mapping each node u in S to a |A|-length array
        """
        K_SA = {}
        
        # Compute R_eff to ground for all nodes in S and A
        S_ground_pairs = [(node, None) for node in S_list]
        A_ground_pairs = [(node, None) for node in A_list]
        
        R_S_ground = self.calc.compute_batch(S_ground_pairs)
        R_A_ground = self.calc.compute_batch(A_ground_pairs)
        
        # For each node u in S, compute vector to all nodes in A
        for i, u in enumerate(S_list):
            K_SA[u] = np.zeros(len(A_list))
            for j, p in enumerate(A_list):
                # Compute R_eff(u,p)
                R_up = self.calc.compute_batch([(u, p)])[0]
                K_SA[u][j] = 0.5 * (R_S_ground[i] + R_A_ground[j] - R_up)
        
        return K_SA
    
    def _build_K_FA(self, F_list: List, A_list: List) -> np.ndarray:
        """Build the K_FA resistance matrix mapping F to A.
        
        K_FA[p,u] = 0.5 * (R_eff(p,0) + R_eff(u,0) - R_eff(p,u))
        
        Args:
            F_list: Sorted list of far load nodes
            A_list: Sorted list of boundary nodes
            
        Returns:
            |A| x |F| numpy array
        """
        n_A = len(A_list)
        n_F = len(F_list)
        K_FA = np.zeros((n_A, n_F))
        
        # Compute R_eff to ground for all nodes in A and F
        A_ground_pairs = [(node, None) for node in A_list]
        F_ground_pairs = [(node, None) for node in F_list]
        
        R_A_ground = self.calc.compute_batch(A_ground_pairs)
        R_F_ground = self.calc.compute_batch(F_ground_pairs)
        
        # Compute K_FA[p,u] for each p in A and u in F
        for i, p in enumerate(A_list):
            for j, u in enumerate(F_list):
                # Compute R_eff(p,u)
                R_pu = self.calc.compute_batch([(p, u)])[0]
                K_FA[i, j] = 0.5 * (R_A_ground[i] + R_F_ground[j] - R_pu)
        
        return K_FA
    
    def _build_K_S(self, S_list: List) -> np.ndarray:
        """Build the K_S resistance matrix for nodes in S.
        
        K_S[p,q] = 0.5 * (R_eff(p,0) + R_eff(q,0) - R_eff(p,q))
        
        Args:
            S_list: Sorted list of nodes in S
            
        Returns:
            |S| x |S| numpy array
        """
        n = len(S_list)
        K_S = np.zeros((n, n))
        
        # Compute R_eff to ground for all nodes in S
        ground_pairs = [(node, None) for node in S_list]
        R_ground = self.calc.compute_batch(ground_pairs)
        
        # Compute R_eff between all pairs in S
        for i, p in enumerate(S_list):
            for j, q in enumerate(S_list):
                if i == j:
                    # Diagonal: R_eff(p,p) = 0
                    K_S[i, j] = R_ground[i]
                else:
                    # Off-diagonal: compute R_eff(p,q)
                    R_pq = self.calc.compute_batch([(p, q)])[0]
                    K_S[i, j] = 0.5 * (R_ground[i] + R_ground[j] - R_pq)
        
        return K_S

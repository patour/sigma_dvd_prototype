"""Regional IR-drop solver for partitioned power grids.

Computes IR-drop at a subset of load nodes S within a partition/region R
using effective resistance matrices and boundary/separator nodes A.

The solver uses the following algorithm:
1. Build resistance matrices K_A, K_SA, K_FA, K_SR from effective resistances
2. Compute boundary IR-drops b_A due to far loads F (loads outside region R)
3. Solve for boundary currents j_A 
4. Compute IR-drop at S from far loads (drop_far) and near loads (drop_near)
5. Return total IR-drop = drop_near + drop_far

Near loads (I_R) are all loads within region R, while far loads (I_F) are
loads outside region R. This allows efficient IR-drop computation by treating
external loads as equivalent boundary currents.
"""

from __future__ import annotations

from typing import Dict, List, Set, Optional
import numpy as np
import scipy.linalg as la

from .effective_resistance import EffectiveResistanceCalculator
from .power_grid_model import PowerGridModel


class RegionalIRDropSolver:
    """Solves for IR-drop at nodes within a partitioned region.
    
    Uses effective resistance matrices and boundary conditions to compute
    IR-drops efficiently for a subset of load nodes.
    """
    
    def __init__(
        self, 
        calc: EffectiveResistanceCalculator
    ):
        """Initialize the regional IR-drop solver.
        
        Args:
            calc: EffectiveResistanceCalculator for computing R_eff
        """
        self.calc = calc
        
    def compute_ir_drops(
        self,
        S: Set,
        R: Set,
        A: Set,
        I_R: Dict,
        I_F: Dict
    ) -> Dict[any, float]:
        """Compute IR-drop at nodes in subset S.
        
        Args:
            S: Set of target load nodes (subset of region R)
            R: Set of all nodes in the region containing S
            A: Set of boundary/separator nodes (ports) for region R
            I_R: Dict mapping load nodes IN region R to their current values (near loads)
            I_F: Dict mapping load nodes NOT in region R to their current values (far loads)
            
        Returns:
            Dictionary mapping each node in S to its computed IR-drop
            
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
        
        # Extract load nodes in region R (near loads)
        R_loads = set(I_R.keys())
        if not R_loads.issubset(R):
            raise ValueError("All near load nodes in I_R must be within region R")
        
        # Convert sets to sorted lists for consistent indexing
        A_list = sorted(list(A), key=lambda n: (n.layer, n.idx))
        S_list = sorted(list(S), key=lambda n: (n.layer, n.idx))
        R_list = sorted(list(R_loads), key=lambda n: (n.layer, n.idx))
        F_list = sorted(list(I_F.keys()), key=lambda n: (n.layer, n.idx))
        
        # Step 1: Build resistance matrices
        K_A = self._build_K_A(A_list)
        K_SA = self._build_K_SA(S_list, A_list)
        K_FA = self._build_K_FA(F_list, A_list)
        K_SR = self._build_K_SR(S_list, R_list)
        
        # Step 2: Compute b_A - IR-drop at boundary due to far loads
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
        
        # Step 4: Compute IR-drop at S due to far loads
        drop_far = {}
        for u in S_list:
            drop_far[u] = np.dot(K_SA[u], j_A)
        
        # Step 5: Compute IR-drop at S due to near loads (all loads in R)
        I_R_vec = np.array([I_R.get(node, 0.0) for node in R_list])
        drop_near = {}
        for i, u in enumerate(S_list):
            drop_near[u] = np.dot(K_SR[u], I_R_vec)
        
        # Step 6: Compute final IR-drop
        # The K matrices compute IR drops (R*I)
        ir_drops = {}
        for u in S_list:
            ir_drops[u] = drop_near[u] + drop_far[u]
        
        return ir_drops
    
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
    
    def _build_K_SR(self, S_list: List, R_list: List) -> Dict[any, np.ndarray]:
        """Build the K_SR resistance matrix mapping S to all load nodes in R.
        
        K_SR[u,v] = 0.5 * (R_eff(u,0) + R_eff(v,0) - R_eff(u,v))
        
        Args:
            S_list: Sorted list of nodes in S (targets)
            R_list: Sorted list of load nodes in region R (near loads)
            
        Returns:
            Dictionary mapping each node u in S to a |R|-length array
        """
        K_SR = {}
        
        # Compute R_eff to ground for all nodes in S and R
        S_ground_pairs = [(node, None) for node in S_list]
        R_ground_pairs = [(node, None) for node in R_list]
        
        R_S_ground = self.calc.compute_batch(S_ground_pairs)
        R_R_ground = self.calc.compute_batch(R_ground_pairs)
        
        # For each node u in S, compute resistance vector to all load nodes in R
        for i, u in enumerate(S_list):
            K_SR[u] = np.zeros(len(R_list))
            for j, v in enumerate(R_list):
                if u == v:
                    # Diagonal: R_eff(u,u) = 0, so K_SR[u,u] = R_eff(u,0)
                    K_SR[u][j] = R_S_ground[i]
                else:
                    # Off-diagonal: compute R_eff(u,v)
                    R_uv = self.calc.compute_batch([(u, v)])[0]
                    K_SR[u][j] = 0.5 * (R_S_ground[i] + R_R_ground[j] - R_uv)
        
        return K_SR

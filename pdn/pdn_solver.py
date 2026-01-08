#!/usr/bin/env python3
"""
PDN IR-Drop/Ground-Bounce Solver - DC voltage analysis for power delivery networks.

This solver computes static voltage drop across power delivery networks by solving
the DC resistive network equations. It handles multi-million node graphs efficiently
using sparse matrix techniques.

Features:
- Automatic floating node/island detection and removal
- Multi-net support with independent solves
- Direct sparse solve (default) or iterative CG solver
- Layer-wise voltage heatmap visualization
- Top-K worst IR-drop reporting with instance names

Usage Examples:
    # Basic IR-drop solve
    from pdn_parser import NetlistParser
    from pdn_solver import PDNSolver
    
    parser = NetlistParser('./netlist_data')
    graph = parser.parse()
    
    solver = PDNSolver(graph)
    solver.solve()
    solver.generate_reports(output_dir='./results')
    
    # With iterative CG solver
    solver = PDNSolver(graph, solver='cg', tolerance=1e-6)
    solver.solve()
    
    # Custom top-K reporting
    solver.generate_reports(top_k=50, output_dir='./results')

Author: Based on mpower power grid analysis
Date: December 12, 2025
"""

import sys
import os
import logging
import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict

try:
    import networkx as nx
except ImportError:
    print("ERROR: NetworkX is required. Install with: pip install networkx")
    sys.exit(1)

try:
    import numpy as np
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
except ImportError:
    print("ERROR: NumPy and SciPy are required. Install with: pip install numpy scipy")
    sys.exit(1)

# Import constants from pdn_parser
try:
    from pdn_parser import GMAX, SHORT_THRESHOLD
except ImportError:
    print("ERROR: pdn_parser module required. Make sure pdn_parser.py is in the same directory.")
    sys.exit(1)

# Import plotter module
try:
    from pdn_plotter import PDNPlotter
except ImportError:
    print("ERROR: pdn_plotter module required. Make sure pdn_plotter.py is in the same directory.")
    sys.exit(1)


@dataclass
class IslandStats:
    """Statistics for disconnected islands"""
    island_id: int
    num_nodes: int
    num_resistors: int
    num_isources: int
    num_vsources: int
    has_voltage_source: bool
    nodes: List[str] = field(default_factory=list)


@dataclass
class NetSolveStats:
    """Statistics for a single net solve"""
    net_name: str
    num_nodes: int
    num_free_nodes: int
    num_vsrc_nodes: int
    num_ground_connections: int
    num_resistors: int
    num_isources: int
    total_current_injection: float
    islands_removed: int
    nodes_removed: int
    resistors_removed: int
    isources_removed: int
    solve_time: float
    solver_iterations: int
    solver_residual: float
    nominal_voltage: float
    max_voltage: float
    min_voltage: float
    avg_voltage: float
    max_drop: float
    avg_drop: float


@dataclass
class SolveResults:
    """Complete solve results"""
    net_stats: Dict[str, NetSolveStats] = field(default_factory=dict)
    island_warnings: List[IslandStats] = field(default_factory=list)
    total_solve_time: float = 0.0


class PDNSolver:
    """
    DC IR-drop and ground-bounce solver for power delivery networks.
    
    Solves the linear system G*V = I where:
    - G is the conductance matrix (from resistors)
    - V is the vector of node voltages (unknowns)
    - I is the current injection vector (from current sources)
    
    Voltage sources are treated as boundary conditions with fixed voltages.
    """
    
    def __init__(self, graph: nx.MultiDiGraph, solver: str = 'direct',
                 tolerance: float = 1e-6, max_iterations: int = 10000,
                 verbose: bool = False, net_filter: Optional[str] = None,
                 anisotropic_bins: bool = True, bin_aspect_ratio: int = 50,
                 layer_orientations: Optional[Dict[str, str]] = None):
        """
        Initialize PDN solver.
        
        Args:
            graph: Parsed PDN graph from NetlistParser
            solver: Solver type ('direct', 'cg', 'bicgstab')
            tolerance: Convergence tolerance for iterative solvers
            max_iterations: Maximum iterations for iterative solvers
            verbose: Enable verbose logging
            net_filter: Solve only specific net (e.g., 'VDD')
            anisotropic_bins: Enable orientation-aware anisotropic binning for heatmaps (default: True)
            bin_aspect_ratio: Aspect ratio for anisotropic bins (default: 50)
            layer_orientations: Manual layer orientation override dict {layer_id: 'H'|'V'|'SQUARE'}
        """
        self.graph = graph
        self.solver_type = solver.lower()
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.net_filter = net_filter
        self.anisotropic_bins = anisotropic_bins
        self.bin_aspect_ratio = bin_aspect_ratio
        self.layer_orientations = layer_orientations or {}
        
        # Setup logging
        log_level = logging.DEBUG if verbose else logging.INFO
        logging.basicConfig(level=log_level, format='%(levelname)s: %(message)s')
        self.logger = logging.getLogger(__name__)
        
        # Extract metadata
        self.net_connectivity = graph.graph.get('net_connectivity', {})
        self.vsrc_dict = graph.graph.get('vsrc_dict', {})
        self.parameters = graph.graph.get('parameters', {})
        self.instance_node_map = graph.graph.get('instance_node_map', {})
        self.vsrc_nodes_global = graph.graph.get('vsrc_nodes', set())
        
        # Results storage
        self.results = SolveResults()
        
        # Validate solver type
        if self.solver_type not in ['direct', 'cg', 'bicgstab']:
            raise ValueError(f"Invalid solver type: {solver}. Choose 'direct', 'cg', or 'bicgstab'")
        
        self.logger.info(f"Initialized PDN solver with {self.solver_type} solver")
        
    def solve(self) -> SolveResults:
        """
        Solve IR-drop for all nets in the graph.
        
        Returns:
            SolveResults object with statistics and results
        """
        start_time = time.time()
        
        self.logger.info("=" * 70)
        self.logger.info("Starting PDN IR-Drop/Ground-Bounce Analysis")
        self.logger.info("=" * 70)
        
        # Get list of nets to solve
        all_nets = list(self.net_connectivity.keys())
        
        if not all_nets:
            self.logger.warning("No nets found in graph. Nothing to solve.")
            return self.results
        
        # Filter nets if specific net was requested
        if self.net_filter:
            # Case-insensitive matching
            net_filter_lower = self.net_filter.lower()
            nets_to_solve = [net for net in all_nets if net.lower() == net_filter_lower]
            
            if not nets_to_solve:
                self.logger.warning(f"Requested net '{self.net_filter}' not found in graph.")
                self.logger.info(f"Available nets: {', '.join(all_nets)}")
                return self.results
            
            self.logger.info(f"Filtering to requested net: {nets_to_solve[0]}")
        else:
            # Solve all nets
            nets_to_solve = all_nets
        
        self.logger.info(f"Solving {len(nets_to_solve)} net(s): {', '.join(nets_to_solve)}")
        
        # Solve each net independently
        for net_name in nets_to_solve:
            self.logger.info(f"\n{'='*70}")
            self.logger.info(f"Solving net: {net_name}")
            self.logger.info(f"{'='*70}")
            
            try:
                net_stats = self._solve_net(net_name)
                self.results.net_stats[net_name] = net_stats
                
                self.logger.info(f"✓ Net {net_name} solved successfully")
                self._print_net_stats(net_stats)
                
            except Exception as e:
                self.logger.error(f"✗ Failed to solve net {net_name}: {e}")
                import traceback
                traceback.print_exc()
        
        self.results.total_solve_time = time.time() - start_time
        
        self.logger.info(f"\n{'='*70}")
        self.logger.info(f"All nets solved in {self.results.total_solve_time:.2f} seconds")
        self.logger.info(f"{'='*70}")
        
        return self.results
    
    def _solve_net(self, net_name: str) -> NetSolveStats:
        """Solve IR-drop for a single net"""
        net_start_time = time.time()
        
        # Get nodes for this net
        net_nodes_list = self.net_connectivity.get(net_name, [])
        net_nodes_set = set(net_nodes_list)
        
        self.logger.info(f"Net has {len(net_nodes_set)} nodes")
        
        # Step 1: Extract net subgraph (only resistive elements)
        net_graph = self._extract_net_subgraph(net_name, net_nodes_set)
        
        # Step 2: Detect and remove floating islands
        island_stats = self._detect_and_remove_islands(net_graph, net_name)
        
        # Step 3: Identify voltage source nodes and their voltages
        vsrc_nodes, vsrc_voltages = self._identify_voltage_sources(net_graph, net_name)
        
        # Step 4: Build node index (free nodes only)
        all_nodes = list(net_graph.nodes())
        free_nodes = [n for n in all_nodes if n != '0' and n not in vsrc_nodes]
        
        if not free_nodes:
            self.logger.warning(f"No free nodes to solve for net {net_name}")
            return self._create_empty_stats(net_name, net_start_time)
        
        node_to_idx = {node: i for i, node in enumerate(free_nodes)}
        n_free = len(free_nodes)
        
        self.logger.info(f"Building system: {n_free} free nodes, {len(vsrc_nodes)} vsrc nodes")
        
        # Step 5: Build conductance matrix and current vector
        G, I, g_stats = self._build_system_matrices(net_graph, free_nodes, node_to_idx,
                                                      vsrc_nodes, vsrc_voltages)
        
        # Step 6: Solve linear system
        V_free, solve_time, iterations, residual = self._solve_linear_system(G, I, n_free)
        
        # Step 7: Store voltages in graph
        self._store_voltages(free_nodes, V_free, vsrc_nodes, vsrc_voltages)
        
        # Step 8: Compute statistics
        nominal_voltage = self._get_nominal_voltage(net_name, vsrc_voltages)
        stats = self._compute_net_statistics(
            net_name, net_graph, free_nodes, vsrc_nodes, g_stats,
            island_stats, V_free, nominal_voltage, solve_time, iterations, residual
        )
        
        stats.solve_time = time.time() - net_start_time
        
        return stats
    
    def _extract_net_subgraph(self, net_name: str, net_nodes: Set[str]) -> nx.MultiDiGraph:
        """Extract subgraph containing only nodes and resistors for this net"""
        # Create subgraph with net nodes
        net_graph = self.graph.subgraph(net_nodes).copy()
        
        # Remove non-resistor edges (we only need resistive network for DC solve)
        edges_to_remove = []
        for u, v, k, d in net_graph.edges(keys=True, data=True):
            if d.get('type') != 'R':
                edges_to_remove.append((u, v, k))
        
        for u, v, k in edges_to_remove:
            net_graph.remove_edge(u, v, k)
        
        self.logger.debug(f"Net subgraph: {net_graph.number_of_nodes()} nodes, "
                         f"{net_graph.number_of_edges()} resistors")
        
        return net_graph
    
    def _detect_and_remove_islands(self, net_graph: nx.MultiDiGraph, net_name: str) -> Dict:
        """
        Detect disconnected islands, remove floating islands, warn about islands with current sources.
        """
        # Convert to undirected for connectivity analysis
        undirected = net_graph.to_undirected()
        
        # Find connected components
        components = list(nx.connected_components(undirected))
        
        if len(components) == 1:
            self.logger.debug("Net is fully connected (1 component)")
            return {'islands_removed': 0, 'nodes_removed': 0, 'resistors_removed': 0, 
                   'isources_removed': 0}
        
        self.logger.warning(f"Net has {len(components)} disconnected islands")
        
        # Analyze each island
        island_info = []
        for i, component in enumerate(components):
            component_graph = net_graph.subgraph(component)
            
            # Check if island has voltage source
            has_vsrc = any(n in self.vsrc_nodes_global for n in component)
            has_vsrc_edge = any(d.get('type') == 'V' 
                              for u, v, d in self.graph.edges(component, data=True))
            has_voltage_source = has_vsrc or has_vsrc_edge
            
            # Count current sources in original graph (not just resistive subgraph)
            num_isources = sum(1 for u, v, d in self.graph.edges(component, data=True)
                             if d.get('type') == 'I')
            
            num_resistors = component_graph.number_of_edges()
            
            island = IslandStats(
                island_id=i,
                num_nodes=len(component),
                num_resistors=num_resistors,
                num_isources=num_isources,
                num_vsources=1 if has_voltage_source else 0,
                has_voltage_source=has_voltage_source,
                nodes=list(component)[:10]  # Store first 10 nodes for debugging
            )
            island_info.append(island)
        
        # Find islands to remove (no voltage source)
        islands_to_remove = [island for island in island_info if not island.has_voltage_source]
        islands_to_keep = [island for island in island_info if island.has_voltage_source]
        
        # Warn about islands with current sources
        for island in islands_to_remove:
            if island.num_isources > 0:
                self.logger.warning(
                    f"  Island {island.island_id}: {island.num_nodes} nodes, "
                    f"{island.num_resistors} resistors, {island.num_isources} current sources - "
                    f"NO VOLTAGE SOURCE (will be removed)"
                )
                self.results.island_warnings.append(island)
            else:
                self.logger.debug(
                    f"  Island {island.island_id}: {island.num_nodes} nodes, "
                    f"{island.num_resistors} resistors - floating (will be removed)"
                )
        
        for island in islands_to_keep:
            self.logger.info(
                f"  Island {island.island_id}: {island.num_nodes} nodes, "
                f"{island.num_resistors} resistors, {island.num_isources} current sources - "
                f"has voltage source (kept)"
            )
        
        # Remove floating islands from net_graph
        nodes_to_remove = set()
        for island in islands_to_remove:
            # Get all nodes from this island's component
            for comp_idx, component in enumerate(components):
                if comp_idx == island.island_id:
                    nodes_to_remove.update(component)
                    break
        
        total_nodes_removed = sum(island.num_nodes for island in islands_to_remove)
        total_resistors_removed = sum(island.num_resistors for island in islands_to_remove)
        total_isources_removed = sum(island.num_isources for island in islands_to_remove)
        
        net_graph.remove_nodes_from(nodes_to_remove)
        
        self.logger.info(f"Removed {len(islands_to_remove)} floating islands: "
                        f"{total_nodes_removed} nodes, {total_resistors_removed} resistors, "
                        f"{total_isources_removed} current sources")
        
        return {
            'islands_removed': len(islands_to_remove),
            'nodes_removed': total_nodes_removed,
            'resistors_removed': total_resistors_removed,
            'isources_removed': total_isources_removed
        }
    
    def _identify_voltage_sources(self, net_graph: nx.MultiDiGraph, net_name: str) -> Tuple[Set[str], Dict[str, float]]:
        """Identify voltage source nodes and their voltage values"""
        vsrc_nodes = set()
        vsrc_voltages = {}
        
        # Find voltage sources in original graph connected to net nodes
        net_nodes = set(net_graph.nodes())
        
        for u, v, d in self.graph.edges(data=True):
            if d.get('type') == 'V':
                voltage = d.get('value', 0.0)
                
                # Check if positive terminal is in net
                if u in net_nodes and u != '0':
                    vsrc_nodes.add(u)
                    vsrc_voltages[u] = voltage
                
                # Check if negative terminal is in net (typically ground)
                if v in net_nodes and v != '0':
                    vsrc_nodes.add(v)
                    vsrc_voltages[v] = 0.0  # Negative terminal at 0V relative to positive
        
        # Also check pre-identified vsrc nodes from parser
        for node in net_nodes:
            if node in self.vsrc_nodes_global and node not in vsrc_voltages:
                # Try to find voltage from nearby V source
                voltage = self._find_voltage_for_node(node)
                if voltage is not None:
                    vsrc_nodes.add(node)
                    vsrc_voltages[node] = voltage
        
        self.logger.info(f"Found {len(vsrc_nodes)} voltage source nodes")
        if self.logger.isEnabledFor(logging.DEBUG) and vsrc_voltages:
            for node, v in list(vsrc_voltages.items())[:5]:
                self.logger.debug(f"  {node}: {v:.6f} V")
        
        return vsrc_nodes, vsrc_voltages
    
    def _find_voltage_for_node(self, node: str) -> Optional[float]:
        """Find voltage value for a vsrc node by traversing nearby V sources"""
        # Check adjacent voltage sources
        for u, v, d in self.graph.edges(node, data=True):
            if d.get('type') == 'V':
                return d.get('value', 0.0)
        
        # Check incoming voltage sources
        for u, v, d in self.graph.in_edges(node, data=True):
            if d.get('type') == 'V':
                return d.get('value', 0.0)
        
        # Try to get from parameters
        net_type = self.graph.nodes[node].get('net_type', '').upper()
        if net_type in self.parameters:
            try:
                return float(self.parameters[net_type])
            except:
                pass
        
        return None
    
    def _build_system_matrices(self, net_graph: nx.MultiDiGraph, free_nodes: List[str],
                                node_to_idx: Dict[str, int], vsrc_nodes: Set[str],
                                vsrc_voltages: Dict[str, float]) -> Tuple[sp.csr_matrix, np.ndarray, Dict]:
        """
        Build conductance matrix G and current injection vector I.
        
        G*V = I where V are the unknown free node voltages.
        """
        n_free = len(free_nodes)
        
        # Use LIL format for efficient construction
        G = sp.lil_matrix((n_free, n_free))
        I = np.zeros(n_free)
        
        # Statistics
        num_resistors = 0
        num_isources = 0
        num_ground_connections = 0
        total_current = 0.0
        
        # Process resistors
        for u, v, k, d in net_graph.edges(keys=True, data=True):
            if d.get('type') != 'R':
                continue
            
            resistance_kohm = d.get('value', 1e-12)
            
            # Handle shorts
            if resistance_kohm < SHORT_THRESHOLD:
                conductance = GMAX  # milli-siemens
            else:
                conductance = 1.0 / resistance_kohm  # milli-siemens
            
            num_resistors += 1
            
            # Stamp conductance matrix
            u_is_free = u in node_to_idx
            v_is_free = v in node_to_idx
            u_is_vsrc = u in vsrc_nodes
            v_is_vsrc = v in vsrc_nodes
            u_is_ground = (u == '0')
            v_is_ground = (v == '0')
            
            if u_is_free and v_is_free:
                # Both nodes are free
                i = node_to_idx[u]
                j = node_to_idx[v]
                G[i, i] += conductance
                G[j, j] += conductance
                G[i, j] -= conductance
                G[j, i] -= conductance
                
            elif u_is_free and (v_is_ground or v_is_vsrc):
                # u is free, v is ground or vsrc
                i = node_to_idx[u]
                G[i, i] += conductance
                
                if v_is_ground:
                    num_ground_connections += 1
                elif v_is_vsrc:
                    # Move voltage source contribution to RHS
                    v_voltage = vsrc_voltages.get(v, 0.0)
                    I[i] += conductance * v_voltage
                    
            elif v_is_free and (u_is_ground or u_is_vsrc):
                # v is free, u is ground or vsrc
                j = node_to_idx[v]
                G[j, j] += conductance
                
                if u_is_ground:
                    num_ground_connections += 1
                elif u_is_vsrc:
                    # Move voltage source contribution to RHS
                    u_voltage = vsrc_voltages.get(u, 0.0)
                    I[j] += conductance * u_voltage
        
        # Process current sources from original graph
        for u, v, d in self.graph.edges(data=True):
            if d.get('type') != 'I':
                continue
            
            current_ma = d.get('value', 0.0)
            
            # Current flows from u to v (u is positive terminal)
            # For power delivery: current sources represent load current (sinking from power net)
            # So we need to flip the sign: current is pulled OUT of the power net
            u_is_free = u in node_to_idx
            v_is_free = v in node_to_idx
            
            if u_is_free:
                i = node_to_idx[u]
                I[i] -= current_ma  # Sink current from power net
                total_current += current_ma
                num_isources += 1
                
            if v_is_free:
                j = node_to_idx[v]
                I[j] += current_ma  # Return current
        
        # Convert to CSR for efficient solving
        G = G.tocsr()
        
        self.logger.debug(f"System built: G is {n_free}x{n_free}, nnz={G.nnz}")
        self.logger.debug(f"  {num_resistors} resistors, {num_isources} current sources")
        self.logger.debug(f"  Total current injection: {total_current:.6f} mA")
        
        stats = {
            'num_resistors': num_resistors,
            'num_isources': num_isources,
            'num_ground_connections': num_ground_connections,
            'total_current': total_current
        }
        
        return G, I, stats
    
    def _solve_linear_system(self, G: sp.csr_matrix, I: np.ndarray, n: int) -> Tuple[np.ndarray, float, int, float]:
        """
        Solve G*V = I using specified solver.
        
        Returns:
            V: Solution vector
            solve_time: Time taken to solve
            iterations: Number of iterations (0 for direct)
            residual: Final residual norm
        """
        self.logger.info(f"Solving {n}x{n} system with {self.solver_type} solver...")
        
        start_time = time.time()
        iterations = 0
        residual = 0.0
        
        try:
            if self.solver_type == 'direct':
                # Direct solve using UMFPACK
                V = spla.spsolve(G, I)
                residual = np.linalg.norm(G @ V - I) / np.linalg.norm(I) if np.linalg.norm(I) > 0 else 0.0
                
            elif self.solver_type == 'cg':
                # Conjugate Gradient (for symmetric positive definite)
                # Check if matrix is symmetric
                is_symmetric = np.allclose((G - G.T).data, 0.0) if G.nnz > 0 else True
                
                if not is_symmetric:
                    self.logger.warning("Matrix is not symmetric, CG may not converge. Consider using 'bicgstab'.")
                
                # Build preconditioner (incomplete LU)
                try:
                    M = spla.spilu(G.tocsc(), drop_tol=1e-4, fill_factor=10)
                    M_op = spla.LinearOperator(G.shape, M.solve)
                except:
                    self.logger.warning("ILU preconditioner failed, using no preconditioner")
                    M_op = None
                
                # Solve with CG
                V, info = spla.cg(G, I, tol=self.tolerance, maxiter=self.max_iterations, M=M_op)
                
                if info > 0:
                    self.logger.warning(f"CG did not converge after {info} iterations")
                    iterations = info
                elif info == 0:
                    iterations = -1  # Converged but don't know exact iterations
                else:
                    self.logger.error(f"CG failed with error code {info}")
                    raise RuntimeError(f"CG solver failed: {info}")
                
                residual = np.linalg.norm(G @ V - I) / np.linalg.norm(I) if np.linalg.norm(I) > 0 else 0.0
                
            elif self.solver_type == 'bicgstab':
                # BiConjugate Gradient Stabilized (for unsymmetric)
                # Build preconditioner
                try:
                    M = spla.spilu(G.tocsc(), drop_tol=1e-4, fill_factor=10)
                    M_op = spla.LinearOperator(G.shape, M.solve)
                except:
                    self.logger.warning("ILU preconditioner failed, using no preconditioner")
                    M_op = None
                
                # Solve with BiCGSTAB
                V, info = spla.bicgstab(G, I, tol=self.tolerance, maxiter=self.max_iterations, M=M_op)
                
                if info > 0:
                    self.logger.warning(f"BiCGSTAB did not converge after {info} iterations")
                    iterations = info
                elif info == 0:
                    iterations = -1
                else:
                    self.logger.error(f"BiCGSTAB failed with error code {info}")
                    raise RuntimeError(f"BiCGSTAB solver failed: {info}")
                
                residual = np.linalg.norm(G @ V - I) / np.linalg.norm(I) if np.linalg.norm(I) > 0 else 0.0
            
            solve_time = time.time() - start_time
            
            self.logger.info(f"System solved in {solve_time:.3f} seconds")
            if iterations > 0:
                self.logger.info(f"  Iterations: {iterations}, Residual: {residual:.2e}")
            else:
                self.logger.debug(f"  Residual: {residual:.2e}")
            
            return V, solve_time, iterations, residual
            
        except Exception as e:
            self.logger.error(f"Solver failed: {e}")
            raise
    
    def _store_voltages(self, free_nodes: List[str], V_free: np.ndarray,
                       vsrc_nodes: Set[str], vsrc_voltages: Dict[str, float]):
        """Store computed voltages in graph node attributes"""
        # Store free node voltages
        for i, node in enumerate(free_nodes):
            self.graph.nodes[node]['voltage'] = float(V_free[i])
        
        # Store vsrc node voltages
        for node, voltage in vsrc_voltages.items():
            self.graph.nodes[node]['voltage'] = voltage
        
        # Ground node is always 0V
        if '0' in self.graph:
            self.graph.nodes['0']['voltage'] = 0.0
    
    def _get_nominal_voltage(self, net_name: str, vsrc_voltages: Dict[str, float]) -> float:
        """Get nominal voltage for this net"""
        # Try to get from vsrc voltages
        if vsrc_voltages:
            return max(vsrc_voltages.values())
        
        # Try to get from parameters
        net_upper = net_name.upper()
        if net_upper in self.parameters:
            try:
                return float(self.parameters[net_upper])
            except:
                pass
        
        # Default
        return 1.0
    
    def _compute_net_statistics(self, net_name: str, net_graph: nx.MultiDiGraph,
                                free_nodes: List[str], vsrc_nodes: Set[str],
                                g_stats: Dict, island_stats: Dict,
                                V_free: np.ndarray, nominal_voltage: float,
                                solve_time: float, iterations: int, residual: float) -> NetSolveStats:
        """Compute statistics for solved net"""
        
        # Voltage statistics
        voltages = list(V_free)
        max_voltage = float(np.max(V_free)) if len(V_free) > 0 else 0.0
        min_voltage = float(np.min(V_free)) if len(V_free) > 0 else 0.0
        avg_voltage = float(np.mean(V_free)) if len(V_free) > 0 else 0.0
        
        # IR-drop statistics
        drops = [abs(nominal_voltage - v) for v in voltages]
        max_drop = max(drops) if drops else 0.0
        avg_drop = sum(drops) / len(drops) if drops else 0.0
        
        stats = NetSolveStats(
            net_name=net_name,
            num_nodes=net_graph.number_of_nodes(),
            num_free_nodes=len(free_nodes),
            num_vsrc_nodes=len(vsrc_nodes),
            num_ground_connections=g_stats['num_ground_connections'],
            num_resistors=g_stats['num_resistors'],
            num_isources=g_stats['num_isources'],
            total_current_injection=g_stats['total_current'],
            islands_removed=island_stats['islands_removed'],
            nodes_removed=island_stats['nodes_removed'],
            resistors_removed=island_stats['resistors_removed'],
            isources_removed=island_stats['isources_removed'],
            solve_time=solve_time,
            solver_iterations=iterations,
            solver_residual=residual,
            nominal_voltage=nominal_voltage,
            max_voltage=max_voltage,
            min_voltage=min_voltage,
            avg_voltage=avg_voltage,
            max_drop=max_drop,
            avg_drop=avg_drop
        )
        
        return stats
    
    def _create_empty_stats(self, net_name: str, start_time: float) -> NetSolveStats:
        """Create empty statistics for nets that couldn't be solved"""
        return NetSolveStats(
            net_name=net_name,
            num_nodes=0, num_free_nodes=0, num_vsrc_nodes=0,
            num_ground_connections=0, num_resistors=0, num_isources=0,
            total_current_injection=0.0,
            islands_removed=0, nodes_removed=0, resistors_removed=0, isources_removed=0,
            solve_time=time.time() - start_time,
            solver_iterations=0, solver_residual=0.0,
            nominal_voltage=0.0, max_voltage=0.0, min_voltage=0.0,
            avg_voltage=0.0, max_drop=0.0, avg_drop=0.0
        )
    
    def _print_net_stats(self, stats: NetSolveStats):
        """Print statistics for a solved net"""
        self.logger.info(f"\nNet Statistics:")
        self.logger.info(f"  Nodes: {stats.num_nodes} total, {stats.num_free_nodes} free, "
                        f"{stats.num_vsrc_nodes} voltage source")
        self.logger.info(f"  Elements: {stats.num_resistors} resistors, {stats.num_isources} current sources")
        self.logger.info(f"  Current injection: {stats.total_current_injection:.3f} mA")
        
        if stats.islands_removed > 0:
            self.logger.info(f"  Removed {stats.islands_removed} floating islands: "
                           f"{stats.nodes_removed} nodes, {stats.resistors_removed} resistors, "
                           f"{stats.isources_removed} current sources")
        
        self.logger.info(f"\nVoltage Results:")
        self.logger.info(f"  Nominal: {stats.nominal_voltage:.6f} V")
        self.logger.info(f"  Range: [{stats.min_voltage:.6f}, {stats.max_voltage:.6f}] V")
        self.logger.info(f"  Average: {stats.avg_voltage:.6f} V")
        self.logger.info(f"\nIR-Drop Results:")
        self.logger.info(f"  Maximum drop: {stats.max_drop*1000:.3f} mV ({stats.max_drop/stats.nominal_voltage*100:.2f}%)")
        self.logger.info(f"  Average drop: {stats.avg_drop*1000:.3f} mV ({stats.avg_drop/stats.nominal_voltage*100:.2f}%)")
    
    def generate_reports(self, output_dir: str = '.', top_k: int = 100,
                        plot_layers: Optional[List[str]] = None,
                        plot_bin_size: Optional[int] = None,
                        anisotropic_bins: Optional[bool] = None,
                        bin_aspect_ratio: Optional[int] = None,
                        stripe_mode: bool = False,
                        max_stripes: int = 50,
                        stripe_bin_size: Optional[int] = None):
        """
        Generate visualization and reports.
        
        Args:
            output_dir: Directory to save outputs
            top_k: Number of worst nodes to report
            plot_layers: List of layer IDs to generate heatmaps for. None = all layers
            plot_bin_size: Bin size for heatmap grid aggregation. None = auto-calculate
            anisotropic_bins: Enable anisotropic binning. None = use solver default
            bin_aspect_ratio: Aspect ratio for anisotropic bins. None = use solver default
            stripe_mode: Enable stripe-based plotting mode (default: False)
            max_stripes: Maximum number of stripes before consolidation (default: 50)
            stripe_bin_size: Bin size for within-stripe aggregation. None = auto-calculate
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Use provided parameters or fall back to solver defaults
        use_anisotropic = anisotropic_bins if anisotropic_bins is not None else self.anisotropic_bins
        use_aspect_ratio = bin_aspect_ratio if bin_aspect_ratio is not None else self.bin_aspect_ratio
        
        self.logger.info(f"\n{'='*70}")
        self.logger.info("Generating Reports and Visualizations")
        if stripe_mode:
            self.logger.info(f"Using stripe-based plotting mode (max_stripes={max_stripes})")
        elif use_anisotropic:
            self.logger.info(f"Using anisotropic binning with aspect ratio {use_aspect_ratio}:1")
        self.logger.info(f"{'='*70}")
        
        # Create plotter instance
        plotter = PDNPlotter(self.graph, self.net_connectivity, self.logger)
        
        # Generate reports for each net
        for net_name, stats in self.results.net_stats.items():
            self.logger.info(f"\nGenerating reports for net: {net_name}")
            
            # Top-K worst IR-drop report
            self._generate_topk_report(net_name, stats, output_path, top_k)
            
            if stripe_mode:
                # Stripe-based heatmaps
                plotter.generate_stripe_heatmaps(net_name, output_path, plot_layers,
                                                max_stripes, stripe_bin_size, 
                                                is_current=False, layer_orientations=self.layer_orientations)
                plotter.generate_stripe_heatmaps(net_name, output_path, plot_layers,
                                                max_stripes, stripe_bin_size,
                                                is_current=True, layer_orientations=self.layer_orientations)
            else:
                # Traditional 2D grid heatmaps
                plotter.generate_layer_heatmaps(net_name, output_path, plot_layers, plot_bin_size,
                                               use_anisotropic, use_aspect_ratio, self.layer_orientations)
                plotter.generate_current_heatmaps(net_name, output_path, plot_layers, plot_bin_size,
                                                 use_anisotropic, use_aspect_ratio, self.layer_orientations)
    
    def _generate_topk_report(self, net_name: str, stats: NetSolveStats,
                              output_path: Path, top_k: int):
        """Generate top-K worst IR-drop report"""
        # Get all nodes with voltages for this net
        net_nodes = self.net_connectivity.get(net_name, [])
        
        # Build reverse instance map once (much faster than nested loop)
        node_to_instance = {}
        for inst, nodes in self.instance_node_map.items():
            for node in nodes:
                node_to_instance[node] = inst
        
        node_data = []
        for node in net_nodes:
            if node == '0':
                continue
            
            node_attrs = self.graph.nodes[node]
            voltage = node_attrs.get('voltage')
            if voltage is None:
                continue
            
            drop = abs(stats.nominal_voltage - voltage)
            drop_pct = (drop / stats.nominal_voltage * 100) if stats.nominal_voltage > 0 else 0.0
            
            node_data.append({
                'node': node,
                'voltage': voltage,
                'drop': drop,
                'drop_pct': drop_pct,
                'layer': node_attrs.get('layer', 'N/A'),
                'x': node_attrs.get('x', 'N/A'),
                'y': node_attrs.get('y', 'N/A'),
                'instance': node_to_instance.get(node, 'N/A')
            })
        
        # Sort by drop (worst first)
        node_data.sort(key=lambda x: x['drop'], reverse=True)
        
        # Write to file
        report_file = output_path / f'topk_irdrop_{net_name}.txt'
        with open(report_file, 'w') as f:
            f.write(f"Top-{top_k} Worst IR-Drop Report\n")
            f.write(f"Net: {net_name}\n")
            f.write(f"Nominal Voltage: {stats.nominal_voltage:.6f} V\n")
            f.write(f"{'='*120}\n")
            f.write(f"{'Rank':<6} {'Node':<30} {'Layer':<8} {'X':<10} {'Y':<10} "
                   f"{'Voltage(V)':<12} {'Drop(mV)':<12} {'Drop(%)':<10} {'Instance':<30}\n")
            f.write(f"{'='*120}\n")
            
            for i, data in enumerate(node_data[:top_k], 1):
                f.write(f"{i:<6} {data['node']:<30} {str(data['layer']):<8} "
                       f"{str(data['x']):<10} {str(data['y']):<10} "
                       f"{data['voltage']:<12.6f} {data['drop']*1000:<12.3f} "
                       f"{data['drop_pct']:<10.2f} {data['instance']:<30}\n")
        
        self.logger.info(f"  Saved top-K report: {report_file}")
        
        # Also print to console
        self.logger.info(f"\n  Top-10 Worst IR-Drop Nodes for {net_name}:")
        self.logger.info(f"  {'Rank':<6} {'Node':<30} {'Layer':<8} {'Drop(mV)':<12} {'Drop(%)':<10}")
        self.logger.info(f"  {'-'*66}")
        for i, data in enumerate(node_data[:10], 1):
            self.logger.info(f"  {i:<6} {data['node']:<30} {str(data['layer']):<8} "
                           f"{data['drop']*1000:<12.3f} {data['drop_pct']:<10.2f}")


def main():
    """Command-line interface for PDN solver"""
    parser = argparse.ArgumentParser(
        description='PDN IR-Drop/Ground-Bounce Solver',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic solve from parsed graph
  python pdn_solver.py --input pdn.pkl --output ./results
  
  # With iterative CG solver
  python pdn_solver.py --input pdn.pkl --solver cg --output ./results
  
  # Custom top-K reporting
  python pdn_solver.py --input pdn.pkl --top-k 50 --output ./results
  
  # Parse and solve in one go
  python pdn_solver.py --netlist-dir ./netlist_data --output ./results
        """
    )
    
    parser.add_argument('--input', '-i', type=str,
                       help='Input graph file (.pkl or .graphml)')
    parser.add_argument('--netlist-dir', type=str,
                       help='Parse netlist from directory (alternative to --input)')
    parser.add_argument('--output', '-o', type=str, default='./results',
                       help='Output directory for results (default: ./results)')
    parser.add_argument('--solver', type=str, default='direct',
                       choices=['direct', 'cg', 'bicgstab'],
                       help='Solver type (default: direct)')
    parser.add_argument('--tolerance', type=float, default=1e-6,
                       help='Convergence tolerance for iterative solvers (default: 1e-6)')
    parser.add_argument('--max-iterations', type=int, default=10000,
                       help='Maximum iterations for iterative solvers (default: 10000)')
    parser.add_argument('--top-k', type=int, default=100,
                       help='Number of worst nodes to report (default: 100)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    parser.add_argument('--net', type=str,
                       help='Solve only specific net (default: all nets)')
    parser.add_argument('--plot-layers', type=str,
                       help='Generate heatmaps only for specified layers (comma-separated, e.g., "19,21,23"). Default: all layers')
    parser.add_argument('--plot-bin-size', type=int,
                       help='Override base bin size (perpendicular dimension). Works with anisotropic binning.')
    parser.add_argument('--no-anisotropic-bins', dest='anisotropic_bins', action='store_false', default=True,
                       help='Disable orientation-aware anisotropic binning (enabled by default)')
    parser.add_argument('--bin-aspect-ratio', type=int, default=50,
                       help='Aspect ratio for anisotropic bins (default: 50, e.g., 50:1 for horizontal layers)')
    parser.add_argument('--layer-orientations', type=str,
                       help='Manual layer orientation override (e.g., "M1:H,M2:V,VIA:SQUARE"). Values: H=horizontal, V=vertical, SQUARE=isotropic')
    parser.add_argument('--stripe-mode', action='store_true',
                       help='Enable stripe-based heatmap plotting mode')
    parser.add_argument('--max-stripes', type=int, default=50,
                       help='Maximum number of stripes before grouping (default: 50, stripe mode only)')
    parser.add_argument('--stripe-bin-size', type=int,
                       help='Physical bin size for within-stripe aggregation in coordinate units (default: auto-calculate, stripe mode only)')
    
    args = parser.parse_args()
    
    # Load or parse graph
    if args.input:
        print(f"Loading graph from {args.input}...")
        
        if args.input.endswith('.pkl'):
            import pickle
            with open(args.input, 'rb') as f:
                graph = pickle.load(f)
        elif args.input.endswith('.graphml'):
            graph = nx.read_graphml(args.input)
        else:
            print(f"ERROR: Unsupported input format. Use .pkl or .graphml")
            return 1
            
    elif args.netlist_dir:
        print(f"Parsing netlist from {args.netlist_dir}...")
        from pdn_parser import NetlistParser
        
        parser_obj = NetlistParser(args.netlist_dir, 
                             net_filter=args.net,
                             verbose=args.verbose)
        graph = parser_obj.parse()
        
    else:
        print("ERROR: Either --input or --netlist-dir must be specified")
        parser.print_help()
        return 1
    
    # Parse layer orientations if provided
    layer_orientations = {}
    if args.layer_orientations:
        for pair in args.layer_orientations.split(','):
            if ':' in pair:
                layer, orient = pair.split(':', 1)
                layer_orientations[layer.strip()] = orient.strip().upper()
    
    # Create solver
    solver = PDNSolver(graph, 
                      solver=args.solver,
                      tolerance=args.tolerance,
                      max_iterations=args.max_iterations,
                      verbose=args.verbose,
                      net_filter=args.net,
                      anisotropic_bins=args.anisotropic_bins,
                      bin_aspect_ratio=args.bin_aspect_ratio,
                      layer_orientations=layer_orientations)
    
    # Solve
    results = solver.solve()
    
    # Generate reports
    plot_layers = getattr(args, 'plot_layers', None)
    if plot_layers:
        plot_layers = plot_layers.split(',')
    plot_bin_size = getattr(args, 'plot_bin_size', None)
    stripe_mode = getattr(args, 'stripe_mode', False)
    max_stripes = getattr(args, 'max_stripes', 50)
    stripe_bin_size = getattr(args, 'stripe_bin_size', None)
    
    solver.generate_reports(output_dir=args.output, top_k=args.top_k, 
                           plot_layers=plot_layers,
                           plot_bin_size=plot_bin_size,
                           stripe_mode=stripe_mode,
                           max_stripes=max_stripes,
                           stripe_bin_size=stripe_bin_size)
    
    print(f"\n{'='*70}")
    print("Analysis complete!")
    print(f"Results saved to: {args.output}")
    print(f"{'='*70}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

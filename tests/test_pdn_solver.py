#!/usr/bin/env python3
"""
Unit tests for PDN Solver (pdn_solver.py)

Tests cover:
- Graph loading and validation
- Island detection and removal
- Voltage source identification
- System matrix construction
- Linear system solving (direct and iterative)
- Voltage storage and retrieval
- Statistics computation
- Multi-net solving
- Report generation
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless testing

import unittest
import sys
import tempfile
import numpy as np
from pathlib import Path

# Add pdn directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'pdn'))

from pdn_parser import NetlistParser
from pdn_solver import PDNSolver, IslandStats, NetSolveStats, SolveResults


class TestPDNSolverBasic(unittest.TestCase):
    """Basic PDN solver tests with test netlist"""
    
    @classmethod
    def setUpClass(cls):
        """Parse test netlist and create solver once for all tests"""
        test_netlist_dir = Path(__file__).parent.parent / 'pdn' / 'netlist_test'
        if not test_netlist_dir.exists():
            raise unittest.SkipTest("Test netlist not found")
        
        parser = NetlistParser(str(test_netlist_dir))
        cls.graph = parser.parse()
        cls.solver = PDNSolver(cls.graph, verbose=False)
    
    def test_solver_initialization(self):
        """Test solver initialization"""
        self.assertIsNotNone(self.solver)
        self.assertEqual(self.solver.solver_type, 'direct')
        self.assertIsNotNone(self.solver.graph)
        self.assertIsNotNone(self.solver.net_connectivity)
    
    def test_solver_type_validation(self):
        """Test solver type validation"""
        with self.assertRaises(ValueError):
            PDNSolver(self.graph, solver='invalid_solver')
    
    def test_metadata_extraction(self):
        """Test metadata extraction from graph"""
        self.assertIsNotNone(self.solver.net_connectivity)
        self.assertIsNotNone(self.solver.vsrc_dict)
        self.assertIsNotNone(self.solver.parameters)
        self.assertIsNotNone(self.solver.instance_node_map)


class TestNetSubgraphExtraction(unittest.TestCase):
    """Test net subgraph extraction"""
    
    @classmethod
    def setUpClass(cls):
        test_netlist_dir = Path(__file__).parent.parent / 'pdn' / 'netlist_test'
        if not test_netlist_dir.exists():
            raise unittest.SkipTest("Test netlist not found")
        
        parser = NetlistParser(str(test_netlist_dir))
        cls.graph = parser.parse()
        cls.solver = PDNSolver(cls.graph, verbose=False)
    
    def test_extract_net_subgraph(self):
        """Test extraction of net subgraph"""
        net_nodes = set(self.solver.net_connectivity.get('VDD', []))
        self.assertGreater(len(net_nodes), 0)
        
        net_graph = self.solver._extract_net_subgraph('VDD', net_nodes)
        
        # Subgraph should have nodes
        self.assertGreater(net_graph.number_of_nodes(), 0)
        
        # Subgraph should only have resistors
        for u, v, d in net_graph.edges(data=True):
            self.assertEqual(d['type'], 'R')


class TestIslandDetection(unittest.TestCase):
    """Test floating island detection and removal"""

    @classmethod
    def setUpClass(cls):
        test_netlist_dir = Path(__file__).parent.parent / 'pdn' / 'netlist_test'
        if not test_netlist_dir.exists():
            raise unittest.SkipTest("Test netlist not found")

        parser = NetlistParser(str(test_netlist_dir))
        cls.graph = parser.parse()
        cls.solver = PDNSolver(cls.graph, verbose=False)

    def test_no_islands_in_test_netlist(self):
        """Test that test netlist has no floating islands"""
        net_nodes = set(self.solver.net_connectivity.get('VDD', []))
        net_graph = self.solver._extract_net_subgraph('VDD', net_nodes)

        island_stats = self.solver._detect_and_remove_islands(net_graph, 'VDD')

        # Test netlist should have no floating islands
        self.assertEqual(island_stats['islands_removed'], 0)
        self.assertEqual(island_stats['nodes_removed'], 0)


class TestIslandDetectionSynthetic(unittest.TestCase):
    """Test island detection with synthetic graphs containing disconnected components"""

    def setUp(self):
        """Create a synthetic graph with disconnected components"""
        from core.rx_graph import RustworkxMultiDiGraphWrapper

        # Create main graph
        self.graph = RustworkxMultiDiGraphWrapper()

        # Component 1: Connected to voltage source (3 nodes)
        self.graph.add_node('n1', x=0, y=0, layer='M1')
        self.graph.add_node('n2', x=1000, y=0, layer='M1')
        self.graph.add_node('n3', x=2000, y=0, layer='M1')
        self.graph.add_edge('n1', 'n2', type='R', value=0.01)
        self.graph.add_edge('n2', 'n3', type='R', value=0.01)

        # Component 2: Floating island (2 nodes, no vsrc)
        self.graph.add_node('f1', x=5000, y=0, layer='M1')
        self.graph.add_node('f2', x=6000, y=0, layer='M1')
        self.graph.add_edge('f1', 'f2', type='R', value=0.01)

        # Component 3: Another floating island (2 nodes)
        self.graph.add_node('f3', x=10000, y=0, layer='M1')
        self.graph.add_node('f4', x=11000, y=0, layer='M1')
        self.graph.add_edge('f3', 'f4', type='R', value=0.01)

        # Add voltage source connecting n1 to ground
        self.graph.add_node('0')  # Ground
        self.graph.add_node('vsrc_node')
        self.graph.add_edge('vsrc_node', '0', type='V', value=1.0)
        self.graph.add_edge('vsrc_node', 'n1', type='R', value=0.001)

        # Add current sources to component 2 (makes it a warning case)
        self.graph.add_edge('f1', '0', type='I', value=1.0)

        # Setup metadata
        self.graph.graph['net_connectivity'] = {
            'VDD': ['n1', 'n2', 'n3', 'f1', 'f2', 'f3', 'f4', 'vsrc_node']
        }
        self.graph.graph['vsrc_dict'] = {'vsrc_node': {'voltage': 1.0}}
        self.graph.graph['vsrc_nodes'] = {'vsrc_node'}
        self.graph.graph['parameters'] = {}
        self.graph.graph['instance_node_map'] = {}

        # Create solver
        self.solver = PDNSolver(self.graph, verbose=False)

    def test_detects_floating_islands(self):
        """Test that floating islands are detected"""
        net_nodes = set(self.solver.net_connectivity.get('VDD', []))
        net_graph = self.solver._extract_net_subgraph('VDD', net_nodes)

        # Before removal, should have 3 components
        from core.rx_algorithms import connected_components
        components_before = connected_components(net_graph)
        self.assertEqual(len(components_before), 3)

        island_stats = self.solver._detect_and_remove_islands(net_graph, 'VDD')

        # Should remove 2 floating islands
        self.assertEqual(island_stats['islands_removed'], 2)

    def test_removes_correct_node_count(self):
        """Test that correct number of nodes are removed"""
        net_nodes = set(self.solver.net_connectivity.get('VDD', []))
        net_graph = self.solver._extract_net_subgraph('VDD', net_nodes)

        island_stats = self.solver._detect_and_remove_islands(net_graph, 'VDD')

        # f1, f2, f3, f4 = 4 nodes removed
        self.assertEqual(island_stats['nodes_removed'], 4)

    def test_removes_correct_resistor_count(self):
        """Test that correct number of resistors are removed"""
        net_nodes = set(self.solver.net_connectivity.get('VDD', []))
        net_graph = self.solver._extract_net_subgraph('VDD', net_nodes)

        island_stats = self.solver._detect_and_remove_islands(net_graph, 'VDD')

        # 2 resistors (f1-f2, f3-f4) should be removed
        self.assertEqual(island_stats['resistors_removed'], 2)

    def test_tracks_current_sources_in_removed_islands(self):
        """Test that current sources in removed islands are tracked"""
        net_nodes = set(self.solver.net_connectivity.get('VDD', []))
        net_graph = self.solver._extract_net_subgraph('VDD', net_nodes)

        island_stats = self.solver._detect_and_remove_islands(net_graph, 'VDD')

        # 1 current source in component 2 should be tracked
        self.assertEqual(island_stats['isources_removed'], 1)

    def test_keeps_component_with_vsrc(self):
        """Test that component with voltage source is kept"""
        net_nodes = set(self.solver.net_connectivity.get('VDD', []))
        net_graph = self.solver._extract_net_subgraph('VDD', net_nodes)

        nodes_before = set(net_graph.nodes())
        self.assertIn('n1', nodes_before)
        self.assertIn('n2', nodes_before)
        self.assertIn('n3', nodes_before)

        self.solver._detect_and_remove_islands(net_graph, 'VDD')

        nodes_after = set(net_graph.nodes())
        # n1, n2, n3 should still be present (connected to vsrc)
        self.assertIn('n1', nodes_after)
        self.assertIn('n2', nodes_after)
        self.assertIn('n3', nodes_after)

    def test_removes_floating_nodes(self):
        """Test that floating nodes are actually removed from graph"""
        net_nodes = set(self.solver.net_connectivity.get('VDD', []))
        net_graph = self.solver._extract_net_subgraph('VDD', net_nodes)

        nodes_before = set(net_graph.nodes())
        self.assertIn('f1', nodes_before)
        self.assertIn('f2', nodes_before)

        self.solver._detect_and_remove_islands(net_graph, 'VDD')

        nodes_after = set(net_graph.nodes())
        # f1, f2, f3, f4 should be removed
        self.assertNotIn('f1', nodes_after)
        self.assertNotIn('f2', nodes_after)
        self.assertNotIn('f3', nodes_after)
        self.assertNotIn('f4', nodes_after)

    def test_single_component_no_removal(self):
        """Test that fully connected graph has no islands removed"""
        from core.rx_graph import RustworkxMultiDiGraphWrapper

        # Create fully connected graph
        graph = RustworkxMultiDiGraphWrapper()
        graph.add_node('a', x=0, y=0, layer='M1')
        graph.add_node('b', x=1000, y=0, layer='M1')
        graph.add_node('c', x=2000, y=0, layer='M1')
        graph.add_edge('a', 'b', type='R', value=0.01)
        graph.add_edge('b', 'c', type='R', value=0.01)

        # Add voltage source
        graph.add_node('0')
        graph.add_node('vsrc')
        graph.add_edge('vsrc', '0', type='V', value=1.0)
        graph.add_edge('vsrc', 'a', type='R', value=0.001)

        graph.graph['net_connectivity'] = {'VDD': ['a', 'b', 'c', 'vsrc']}
        graph.graph['vsrc_dict'] = {}
        graph.graph['vsrc_nodes'] = {'vsrc'}
        graph.graph['parameters'] = {}
        graph.graph['instance_node_map'] = {}

        solver = PDNSolver(graph, verbose=False)
        net_nodes = set(solver.net_connectivity.get('VDD', []))
        net_graph = solver._extract_net_subgraph('VDD', net_nodes)

        island_stats = solver._detect_and_remove_islands(net_graph, 'VDD')

        self.assertEqual(island_stats['islands_removed'], 0)
        self.assertEqual(island_stats['nodes_removed'], 0)


class TestIslandDetectionOptimization(unittest.TestCase):
    """Test that optimized island detection produces same results as reference"""

    @classmethod
    def setUpClass(cls):
        """Parse netlist_small for comparison tests"""
        test_netlist_dir = Path(__file__).parent.parent / 'pdn' / 'netlist_small'
        if not test_netlist_dir.exists():
            raise unittest.SkipTest("netlist_small not found")

        parser = NetlistParser(str(test_netlist_dir), validate=False)
        cls.graph = parser.parse()

    def test_optimized_matches_reference(self):
        """Test optimized island detection (no to_undirected) matches reference"""
        from core.rx_algorithms import connected_components

        # Extract net subgraph
        net_nodes = set(self.graph.graph.get('net_connectivity', {}).get('VDD_XLV', []))
        net_graph = self.graph.subgraph(net_nodes).copy()

        # Remove non-resistor edges
        edges_to_remove = []
        for u, v, k, d in net_graph.edges(keys=True, data=True):
            if d.get('type') != 'R':
                edges_to_remove.append((u, v, k))
        for u, v, k in edges_to_remove:
            net_graph.remove_edge(u, v, k)

        # Reference: to_undirected + connected_components
        undirected = net_graph.to_undirected()
        ref_components = connected_components(undirected)

        # Optimized: direct connected_components on directed graph
        opt_components = connected_components(net_graph)

        # Compare as sets of frozensets
        ref_sets = set(frozenset(c) for c in ref_components)
        opt_sets = set(frozenset(c) for c in opt_components)

        self.assertEqual(ref_sets, opt_sets)
        self.assertEqual(len(ref_components), len(opt_components))


class TestVoltageSourceIdentification(unittest.TestCase):
    """Test voltage source node identification"""
    
    @classmethod
    def setUpClass(cls):
        test_netlist_dir = Path(__file__).parent.parent / 'pdn' / 'netlist_test'
        if not test_netlist_dir.exists():
            raise unittest.SkipTest("Test netlist not found")
        
        parser = NetlistParser(str(test_netlist_dir))
        cls.graph = parser.parse()
        cls.solver = PDNSolver(cls.graph, verbose=False)
    
    def test_identify_voltage_sources(self):
        """Test voltage source identification"""
        net_nodes = set(self.solver.net_connectivity.get('VDD', []))
        net_graph = self.solver._extract_net_subgraph('VDD', net_nodes)
        
        vsrc_nodes, vsrc_voltages = self.solver._identify_voltage_sources(net_graph, 'VDD')
        
        # Should find voltage source nodes
        self.assertGreater(len(vsrc_nodes), 0)
        self.assertGreater(len(vsrc_voltages), 0)
        
        # Voltages should be reasonable (0.75V for VDD)
        for node, voltage in vsrc_voltages.items():
            self.assertGreater(voltage, 0.0)
            self.assertLess(voltage, 2.0)


class TestSystemMatrixConstruction(unittest.TestCase):
    """Test conductance matrix and current vector construction"""
    
    @classmethod
    def setUpClass(cls):
        test_netlist_dir = Path(__file__).parent.parent / 'pdn' / 'netlist_test'
        if not test_netlist_dir.exists():
            raise unittest.SkipTest("Test netlist not found")
        
        parser = NetlistParser(str(test_netlist_dir))
        cls.graph = parser.parse()
        cls.solver = PDNSolver(cls.graph, verbose=False)
    
    def test_build_system_matrices(self):
        """Test system matrix construction"""
        net_nodes = set(self.solver.net_connectivity.get('VDD', []))
        net_graph = self.solver._extract_net_subgraph('VDD', net_nodes)
        
        # Remove islands
        self.solver._detect_and_remove_islands(net_graph, 'VDD')
        
        # Identify voltage sources
        vsrc_nodes, vsrc_voltages = self.solver._identify_voltage_sources(net_graph, 'VDD')
        
        # Build node index
        all_nodes = list(net_graph.nodes())
        free_nodes = [n for n in all_nodes if n != '0' and n not in vsrc_nodes]
        node_to_idx = {node: i for i, node in enumerate(free_nodes)}
        
        # Build matrices
        G, I, stats = self.solver._build_system_matrices(
            net_graph, free_nodes, node_to_idx, vsrc_nodes, vsrc_voltages
        )
        
        # Check matrix dimensions
        n = len(free_nodes)
        self.assertEqual(G.shape, (n, n))
        self.assertEqual(I.shape, (n,))
        
        # Check that G is square and sparse
        self.assertEqual(G.shape[0], G.shape[1])
        
        # Check statistics
        self.assertIn('num_ground_connections', stats)
        self.assertIn('num_resistors', stats)


class TestLinearSystemSolving(unittest.TestCase):
    """Test linear system solving"""
    
    @classmethod
    def setUpClass(cls):
        test_netlist_dir = Path(__file__).parent.parent / 'pdn' / 'netlist_test'
        if not test_netlist_dir.exists():
            raise unittest.SkipTest("Test netlist not found")
        
        parser = NetlistParser(str(test_netlist_dir))
        cls.graph = parser.parse()
    
    def test_direct_solver(self):
        """Test direct sparse solver"""
        solver = PDNSolver(self.graph, solver='direct', verbose=False)
        results = solver.solve()
        
        # Check that solve completed
        self.assertIsNotNone(results)
        self.assertGreater(len(results.net_stats), 0)
    
    def test_iterative_cg_solver(self):
        """Test iterative CG solver"""
        solver = PDNSolver(self.graph, solver='cg', tolerance=1e-6, verbose=False)
        results = solver.solve()
        
        # Check that solve completed
        self.assertIsNotNone(results)
        self.assertGreater(len(results.net_stats), 0)
    
    def test_iterative_bicgstab_solver(self):
        """Test iterative BiCGSTAB solver"""
        solver = PDNSolver(self.graph, solver='bicgstab', tolerance=1e-6, verbose=False)
        results = solver.solve()
        
        # Check that solve completed
        self.assertIsNotNone(results)
        self.assertGreater(len(results.net_stats), 0)


class TestVoltageSolution(unittest.TestCase):
    """Test voltage solution and storage"""
    
    @classmethod
    def setUpClass(cls):
        test_netlist_dir = Path(__file__).parent.parent / 'pdn' / 'netlist_test'
        if not test_netlist_dir.exists():
            raise unittest.SkipTest("Test netlist not found")
        
        parser = NetlistParser(str(test_netlist_dir))
        cls.graph = parser.parse()
        cls.solver = PDNSolver(cls.graph, verbose=False)
        cls.results = cls.solver.solve()
    
    def test_voltages_stored(self):
        """Test that voltages are stored in graph"""
        # Check that some nodes have voltage attribute
        nodes_with_voltage = [n for n, d in self.solver.graph.nodes(data=True) 
                             if 'voltage' in d]
        self.assertGreater(len(nodes_with_voltage), 0)
    
    def test_voltage_values_reasonable(self):
        """Test that voltage values are reasonable"""
        # VDD is 0.75V, voltages should be close to that
        for node, data in self.solver.graph.nodes(data=True):
            if 'voltage' in data:
                voltage = data['voltage']
                # Skip None or zero voltages (e.g., unused package nodes, ground)
                if voltage is not None and voltage > 0:
                    self.assertGreater(voltage, 0.65)  # Min 0.65V (allow for package drop)
                    self.assertLess(voltage, 0.76)     # Max 0.76V
    
    def test_ir_drop_values(self):
        """Test that IR-drop values are computed"""
        vdd_stats = self.results.net_stats.get('VDD')
        if vdd_stats:
            # IR-drop should be positive and small
            self.assertGreater(vdd_stats.max_drop, 0.0)
            self.assertLess(vdd_stats.max_drop, 0.01)  # Less than 10mV drop
            
            # Min voltage should be less than nominal
            self.assertLess(vdd_stats.min_voltage, vdd_stats.nominal_voltage)


class TestStatisticsComputation(unittest.TestCase):
    """Test statistics computation"""
    
    @classmethod
    def setUpClass(cls):
        test_netlist_dir = Path(__file__).parent.parent / 'pdn' / 'netlist_test'
        if not test_netlist_dir.exists():
            raise unittest.SkipTest("Test netlist not found")
        
        parser = NetlistParser(str(test_netlist_dir))
        cls.graph = parser.parse()
        cls.solver = PDNSolver(cls.graph, verbose=False)
        cls.results = cls.solver.solve()
    
    def test_solve_results_structure(self):
        """Test solve results structure"""
        self.assertIsInstance(self.results, SolveResults)
        self.assertIsInstance(self.results.net_stats, dict)
        self.assertGreater(self.results.total_solve_time, 0.0)
    
    def test_net_stats_structure(self):
        """Test net statistics structure"""
        vdd_stats = self.results.net_stats.get('VDD')
        self.assertIsNotNone(vdd_stats)
        self.assertIsInstance(vdd_stats, NetSolveStats)
        
        # Check required fields
        self.assertEqual(vdd_stats.net_name, 'VDD')
        self.assertGreater(vdd_stats.num_nodes, 0)
        self.assertGreater(vdd_stats.num_free_nodes, 0)
        self.assertGreater(vdd_stats.num_resistors, 0)
        self.assertGreaterEqual(vdd_stats.solve_time, 0.0)
    
    def test_voltage_statistics(self):
        """Test voltage statistics"""
        vdd_stats = self.results.net_stats.get('VDD')
        if vdd_stats:
            # Nominal voltage should be 0.75V
            self.assertAlmostEqual(vdd_stats.nominal_voltage, 0.75, places=2)
            
            # Min/max/avg should be reasonable
            self.assertGreater(vdd_stats.min_voltage, 0.0)
            self.assertLessEqual(vdd_stats.max_voltage, vdd_stats.nominal_voltage)
            self.assertGreater(vdd_stats.avg_voltage, 0.0)
            
            # Ordering: min <= avg <= max
            self.assertLessEqual(vdd_stats.min_voltage, vdd_stats.avg_voltage)
            self.assertLessEqual(vdd_stats.avg_voltage, vdd_stats.max_voltage)
    
    def test_current_injection_statistics(self):
        """Test current injection statistics"""
        vdd_stats = self.results.net_stats.get('VDD')
        if vdd_stats:
            # Test netlist has 17 current sources totaling 38mA
            self.assertGreater(vdd_stats.total_current_injection, 0.0)
            self.assertAlmostEqual(vdd_stats.total_current_injection, 38.0, delta=1.0)


class TestMultiNetSolving(unittest.TestCase):
    """Test solving multiple nets"""
    
    def test_single_net_filter(self):
        """Test solving with net filter"""
        test_netlist_dir = Path(__file__).parent.parent / 'pdn' / 'netlist_test'
        if not test_netlist_dir.exists():
            self.skipTest("Test netlist not found")
        
        parser = NetlistParser(str(test_netlist_dir))
        graph = parser.parse()
        
        solver = PDNSolver(graph, net_filter='VDD', verbose=False)
        results = solver.solve()
        
        # Should only solve VDD net
        self.assertEqual(len(results.net_stats), 1)
        self.assertIn('VDD', results.net_stats)


class TestReportGeneration(unittest.TestCase):
    """Test report generation"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_generate_reports(self):
        """Test report generation"""
        test_netlist_dir = Path(__file__).parent.parent / 'pdn' / 'netlist_test'
        if not test_netlist_dir.exists():
            self.skipTest("Test netlist not found")
        
        parser = NetlistParser(str(test_netlist_dir))
        graph = parser.parse()
        
        solver = PDNSolver(graph, verbose=False)
        results = solver.solve()
        
        # Generate reports
        solver.generate_reports(output_dir=self.temp_dir, top_k=10)
        
        # Check that report files were created
        report_file = Path(self.temp_dir) / 'topk_irdrop_VDD.txt'
        self.assertTrue(report_file.exists())
        
        # Check report content
        with open(report_file, 'r') as f:
            content = f.read()
            self.assertIn('Top-10 Worst IR-Drop Report', content)
            self.assertIn('VDD', content)
    
    def test_heatmap_generation(self):
        """Test heatmap generation"""
        test_netlist_dir = Path(__file__).parent.parent / 'pdn' / 'netlist_test'
        if not test_netlist_dir.exists():
            self.skipTest("Test netlist not found")
        
        parser = NetlistParser(str(test_netlist_dir))
        graph = parser.parse()
        
        solver = PDNSolver(graph, verbose=False)
        results = solver.solve()
        
        # Generate reports with heatmaps
        solver.generate_reports(output_dir=self.temp_dir, top_k=10)
        
        # Check that heatmap files were created (per-layer files)
        output_path = Path(self.temp_dir)
        voltage_heatmaps = list(output_path.glob('*_heatmap_VDD_layer_*.png'))
        current_heatmap = output_path / 'current_heatmap_VDD.png'
        
        self.assertGreater(len(voltage_heatmaps), 0, "Should create at least one voltage/irdrop heatmap")
        self.assertTrue(current_heatmap.exists())


class TestSolverEdgeCases(unittest.TestCase):
    """Test edge cases and error handling"""
    
    def test_empty_net(self):
        """Test handling of empty net"""
        test_netlist_dir = Path(__file__).parent.parent / 'pdn' / 'netlist_test'
        if not test_netlist_dir.exists():
            self.skipTest("Test netlist not found")
        
        parser = NetlistParser(str(test_netlist_dir))
        graph = parser.parse()
        
        # Try to solve non-existent net
        solver = PDNSolver(graph, net_filter='NONEXISTENT', verbose=False)
        results = solver.solve()
        
        # Should handle gracefully
        self.assertEqual(len(results.net_stats), 0)
    
    def test_solve_convergence(self):
        """Test that solver converges"""
        test_netlist_dir = Path(__file__).parent.parent / 'pdn' / 'netlist_test'
        if not test_netlist_dir.exists():
            self.skipTest("Test netlist not found")
        
        parser = NetlistParser(str(test_netlist_dir))
        graph = parser.parse()
        
        solver = PDNSolver(graph, solver='direct', verbose=False)
        results = solver.solve()
        
        # Check convergence
        vdd_stats = results.net_stats.get('VDD')
        if vdd_stats:
            # Direct solver should have 0 iterations
            self.assertEqual(vdd_stats.solver_iterations, 0)
            # Residual should be very small
            self.assertLess(vdd_stats.solver_residual, 1e-10)


class TestNominalVoltage(unittest.TestCase):
    """Test nominal voltage determination"""
    
    @classmethod
    def setUpClass(cls):
        test_netlist_dir = Path(__file__).parent.parent / 'pdn' / 'netlist_test'
        if not test_netlist_dir.exists():
            raise unittest.SkipTest("Test netlist not found")
        
        parser = NetlistParser(str(test_netlist_dir))
        cls.graph = parser.parse()
        cls.solver = PDNSolver(cls.graph, verbose=False)
    
    def test_get_nominal_voltage(self):
        """Test nominal voltage extraction"""
        vsrc_voltages = {'node1': 0.75, 'node2': 0.75}
        nominal = self.solver._get_nominal_voltage('VDD', vsrc_voltages)
        
        self.assertAlmostEqual(nominal, 0.75, places=2)


if __name__ == '__main__':
    unittest.main()

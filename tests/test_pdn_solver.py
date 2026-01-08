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
        
        # Check that heatmap files were created
        voltage_heatmap = Path(self.temp_dir) / 'voltage_heatmap_VDD.png'
        current_heatmap = Path(self.temp_dir) / 'current_heatmap_VDD.png'
        
        self.assertTrue(voltage_heatmap.exists())
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

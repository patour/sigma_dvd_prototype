#!/usr/bin/env python3
"""
Unit tests for PDN Solver (pdn_solver.py)

Tests cover:
- Solver initialization and metadata extraction
- End-to-end solving using UnifiedIRDropSolver backend
- PDNSolveResult statistics validation
- Solver backend configuration (cholmod/splu)
- Timing and memory profiling
- Config file loading
- Multi-net solving
- Report generation
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless testing

import unittest
import sys
import tempfile
import json
import numpy as np
from pathlib import Path

# Add pdn directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'pdn'))

from pdn_parser import NetlistParser
from pdn_solver import (
    PDNSolver, PDNSolveResult, SolveResults, TimingStats, MemoryStats,
    load_config, merge_config_with_args,
)


class TestPDNSolverBasic(unittest.TestCase):
    """Basic PDN solver tests with netlist_small"""

    @classmethod
    def setUpClass(cls):
        """Parse netlist_small and create solver once for all tests"""
        cls.netlist_dir = Path(__file__).parent.parent / 'pdn' / 'netlist_small'
        if not cls.netlist_dir.exists():
            raise unittest.SkipTest("netlist_small not found")

        parser = NetlistParser(str(cls.netlist_dir))
        cls.graph = parser.parse()
        cls.solver = PDNSolver(cls.graph, verbose=False)

    def test_solver_initialization(self):
        """Test solver initialization"""
        self.assertIsNotNone(self.solver)
        self.assertIsNotNone(self.solver.graph)
        self.assertIsNotNone(self.solver.net_connectivity)

    def test_metadata_extraction(self):
        """Test metadata extraction from graph"""
        self.assertIsNotNone(self.solver.net_connectivity)
        self.assertIsNotNone(self.solver.vsrc_dict)
        self.assertIsNotNone(self.solver.parameters)
        self.assertIsNotNone(self.solver.instance_node_map)


class TestPDNSolveResult(unittest.TestCase):
    """Test PDNSolveResult data class"""

    @classmethod
    def setUpClass(cls):
        netlist_dir = Path(__file__).parent.parent / 'pdn' / 'netlist_small'
        if not netlist_dir.exists():
            raise unittest.SkipTest("netlist_small not found")

        parser = NetlistParser(str(netlist_dir))
        cls.graph = parser.parse()
        cls.solver = PDNSolver(cls.graph, verbose=True)
        cls.results = cls.solver.solve()

    def test_result_structure(self):
        """Test that results have correct structure"""
        self.assertIsInstance(self.results, SolveResults)
        self.assertIsInstance(self.results.net_results, dict)
        self.assertGreater(len(self.results.net_results), 0)

    def test_pdn_solve_result_fields(self):
        """Test PDNSolveResult has all expected fields"""
        # Get first net result
        net_name = list(self.results.net_results.keys())[0]
        result = self.results.net_results[net_name]

        self.assertIsInstance(result, PDNSolveResult)

        # Check inherited fields from UnifiedSolveResult
        self.assertIsInstance(result.voltages, dict)
        self.assertIsInstance(result.ir_drop, dict)
        self.assertGreater(result.nominal_voltage, 0)
        self.assertEqual(result.net_name, net_name)

        # Check PDN-specific fields
        self.assertGreater(result.num_nodes, 0)
        self.assertGreater(result.num_free_nodes, 0)
        self.assertGreaterEqual(result.num_vsrc_nodes, 0)
        self.assertGreaterEqual(result.num_isources, 0)
        self.assertGreaterEqual(result.total_current_ma, 0)

        # Check voltage statistics
        self.assertGreater(result.max_voltage, 0)
        self.assertGreater(result.min_voltage, 0)
        self.assertGreater(result.avg_voltage, 0)
        self.assertGreaterEqual(result.max_ir_drop, 0)
        self.assertGreaterEqual(result.avg_ir_drop, 0)

    def test_timing_stats(self):
        """Test TimingStats in result"""
        net_name = list(self.results.net_results.keys())[0]
        result = self.results.net_results[net_name]

        self.assertIsInstance(result.timings, TimingStats)
        self.assertGreater(result.timings.model_creation, 0)
        self.assertGreater(result.timings.solve, 0)
        self.assertGreater(result.timings.total, 0)

    def test_backend_field(self):
        """Test backend field is set"""
        net_name = list(self.results.net_results.keys())[0]
        result = self.results.net_results[net_name]

        self.assertIn(result.backend, ['cholmod', 'splu'])


class TestIslandDetection(unittest.TestCase):
    """Test island detection with synthetic graphs"""

    def setUp(self):
        """Create a synthetic graph with disconnected components"""
        from core.rx_graph import RustworkxMultiDiGraphWrapper

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
        self.graph.graph['parameters'] = {'VDD': '1.0'}
        self.graph.graph['instance_node_map'] = {}

    def test_floating_islands_detected_via_solve(self):
        """Test that floating islands are detected and removed during solve."""
        solver = PDNSolver(self.graph, verbose=False)
        results = solver.solve()

        result = results.net_results.get('VDD')
        self.assertIsNotNone(result)

        # UnifiedPowerGridModel detects and removes 2 floating islands
        self.assertEqual(result.islands_removed, 2)
        self.assertEqual(result.nodes_removed, 4)  # f1, f2, f3, f4

    def test_valid_voltages_computed(self):
        """Test that valid voltages are computed for connected component."""
        solver = PDNSolver(self.graph, verbose=False)
        results = solver.solve()

        # n1, n2, n3 should have voltages stored
        self.assertIn('n1', self.graph.nodes_dict)
        voltage = self.graph.nodes_dict.get('n1', {}).get('voltage')
        self.assertIsNotNone(voltage)
        self.assertGreater(voltage, 0.0)


class TestSolverBackendConfiguration(unittest.TestCase):
    """Test solver backend configuration options"""

    @classmethod
    def setUpClass(cls):
        netlist_dir = Path(__file__).parent.parent / 'pdn' / 'netlist_small'
        if not netlist_dir.exists():
            raise unittest.SkipTest("netlist_small not found")

        parser = NetlistParser(str(netlist_dir))
        cls.graph = parser.parse()

    def test_default_backend(self):
        """Test default backend selection"""
        solver = PDNSolver(self.graph, verbose=False)
        results = solver.solve()

        net_name = list(results.net_results.keys())[0]
        result = results.net_results[net_name]

        # Backend should be either cholmod or splu
        self.assertIn(result.backend, ['cholmod', 'splu'])

    def test_force_splu_backend(self):
        """Test forcing splu backend"""
        solver = PDNSolver(self.graph, verbose=False, use_cholmod=False)
        results = solver.solve()

        net_name = list(results.net_results.keys())[0]
        result = results.net_results[net_name]

        self.assertEqual(result.backend, 'splu')

    def test_cholmod_mode_option(self):
        """Test cholmod mode configuration"""
        # This should not raise even if cholmod is not available
        try:
            solver = PDNSolver(
                self.graph,
                verbose=False,
                use_cholmod=True,
                cholmod_mode='supernodal'
            )
            results = solver.solve()
            # If we got here, cholmod is available
            net_name = list(results.net_results.keys())[0]
            result = results.net_results[net_name]
            self.assertEqual(result.backend, 'cholmod')
        except ImportError:
            # Cholmod not available, that's fine
            self.skipTest("Cholmod not available")


class TestTimingAndMemoryProfiling(unittest.TestCase):
    """Test timing and memory profiling features"""

    @classmethod
    def setUpClass(cls):
        netlist_dir = Path(__file__).parent.parent / 'pdn' / 'netlist_small'
        if not netlist_dir.exists():
            raise unittest.SkipTest("netlist_small not found")

        parser = NetlistParser(str(netlist_dir))
        cls.graph = parser.parse()

    def test_verbose_timing(self):
        """Test that verbose mode produces timing information"""
        solver = PDNSolver(self.graph, verbose=True)
        results = solver.solve()

        net_name = list(results.net_results.keys())[0]
        result = results.net_results[net_name]

        # All timing fields should be populated
        self.assertGreater(result.timings.model_creation, 0)
        self.assertGreaterEqual(result.timings.current_extraction, 0)
        self.assertGreater(result.timings.solve, 0)
        self.assertGreater(result.timings.total, 0)

    def test_memory_profiling(self):
        """Test memory profiling feature"""
        solver = PDNSolver(self.graph, verbose=True, profile_memory=True)
        results = solver.solve()

        # Total memory should be tracked
        self.assertGreater(results.total_memory_peak_mb, 0)

        net_name = list(results.net_results.keys())[0]
        result = results.net_results[net_name]

        # Per-net memory should be tracked
        self.assertGreater(result.memory.peak_mb, 0)


class TestConfigFileSupport(unittest.TestCase):
    """Test config file loading and merging"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_load_json_config(self):
        """Test loading JSON config file"""
        config_path = Path(self.temp_dir) / 'config.json'
        config_data = {
            'verbose': True,
            'top_k': 50,
            'use_cholmod': True,
            'cholmod_mode': 'supernodal',
        }

        with open(config_path, 'w') as f:
            json.dump(config_data, f)

        loaded = load_config(str(config_path))

        self.assertEqual(loaded['verbose'], True)
        self.assertEqual(loaded['top_k'], 50)
        self.assertEqual(loaded['use_cholmod'], True)
        self.assertEqual(loaded['cholmod_mode'], 'supernodal')

    def test_merge_config_with_args(self):
        """Test merging config with CLI args"""
        import argparse

        config = {
            'verbose': True,
            'top_k': 50,
            'output': './custom_output',
        }

        # Create args namespace with defaults
        args = argparse.Namespace(
            verbose=False,  # Default
            top_k=100,  # Default
            output='./results',  # Default
            net=None,
        )

        merged = merge_config_with_args(config, args)

        # Config values should override defaults
        self.assertEqual(merged.verbose, True)
        self.assertEqual(merged.top_k, 50)
        self.assertEqual(merged.output, './custom_output')

    def test_config_file_not_found(self):
        """Test error handling for missing config file"""
        with self.assertRaises(FileNotFoundError):
            load_config('/nonexistent/config.yaml')

    def test_unsupported_config_format(self):
        """Test error handling for unsupported format"""
        config_path = Path(self.temp_dir) / 'config.txt'
        config_path.touch()

        with self.assertRaises(ValueError):
            load_config(str(config_path))


class TestVoltageSolution(unittest.TestCase):
    """Test voltage solution and storage"""

    @classmethod
    def setUpClass(cls):
        netlist_dir = Path(__file__).parent.parent / 'pdn' / 'netlist_small'
        if not netlist_dir.exists():
            raise unittest.SkipTest("netlist_small not found")

        parser = NetlistParser(str(netlist_dir))
        cls.graph = parser.parse()
        cls.solver = PDNSolver(cls.graph, verbose=False)
        cls.results = cls.solver.solve()

    def test_voltages_stored(self):
        """Test that voltages are stored in graph"""
        nodes_with_voltage = [n for n, d in self.solver.graph.nodes(data=True)
                             if 'voltage' in d]
        self.assertGreater(len(nodes_with_voltage), 0)

    def test_voltage_values_reasonable(self):
        """Test that voltage values are reasonable"""
        net_name = list(self.results.net_results.keys())[0]
        result = self.results.net_results[net_name]

        # All voltages should be positive and close to nominal
        for node, voltage in result.voltages.items():
            if voltage > 0:  # Skip ground nodes
                self.assertGreater(voltage, 0)
                self.assertLess(voltage, result.nominal_voltage * 1.01)

    def test_ir_drop_values(self):
        """Test that IR-drop values are computed"""
        net_name = list(self.results.net_results.keys())[0]
        result = self.results.net_results[net_name]

        # IR-drop should be positive and small
        self.assertGreater(result.max_ir_drop, 0.0)
        self.assertLess(result.max_ir_drop, 0.1)  # Less than 100mV drop

        # Min voltage should be less than nominal
        self.assertLess(result.min_voltage, result.nominal_voltage)


class TestStatisticsComputation(unittest.TestCase):
    """Test statistics computation"""

    @classmethod
    def setUpClass(cls):
        netlist_dir = Path(__file__).parent.parent / 'pdn' / 'netlist_small'
        if not netlist_dir.exists():
            raise unittest.SkipTest("netlist_small not found")

        parser = NetlistParser(str(netlist_dir))
        cls.graph = parser.parse()
        cls.solver = PDNSolver(cls.graph, verbose=False)
        cls.results = cls.solver.solve()

    def test_solve_results_structure(self):
        """Test solve results structure"""
        self.assertIsInstance(self.results, SolveResults)
        self.assertIsInstance(self.results.net_results, dict)
        self.assertGreater(self.results.total_solve_time, 0.0)

    def test_net_result_structure(self):
        """Test net result structure"""
        net_name = list(self.results.net_results.keys())[0]
        result = self.results.net_results[net_name]

        self.assertIsNotNone(result)
        self.assertIsInstance(result, PDNSolveResult)

        # Check required fields
        self.assertEqual(result.net_name, net_name)
        self.assertGreater(result.num_nodes, 0)
        self.assertGreater(result.num_free_nodes, 0)
        self.assertGreaterEqual(result.timings.total, 0.0)

    def test_voltage_statistics(self):
        """Test voltage statistics"""
        net_name = list(self.results.net_results.keys())[0]
        result = self.results.net_results[net_name]

        # Min/max/avg should be reasonable
        self.assertGreater(result.min_voltage, 0.0)
        self.assertLessEqual(result.max_voltage, result.nominal_voltage)
        self.assertGreater(result.avg_voltage, 0.0)

        # Ordering: min <= avg <= max
        self.assertLessEqual(result.min_voltage, result.avg_voltage)
        self.assertLessEqual(result.avg_voltage, result.max_voltage)


class TestMultiNetSolving(unittest.TestCase):
    """Test solving multiple nets"""

    def test_single_net_filter(self):
        """Test solving with net filter"""
        netlist_dir = Path(__file__).parent.parent / 'pdn' / 'netlist_small'
        if not netlist_dir.exists():
            self.skipTest("netlist_small not found")

        parser = NetlistParser(str(netlist_dir))
        graph = parser.parse()

        # Get first net name
        nets = list(graph.graph.get('net_connectivity', {}).keys())
        if not nets:
            self.skipTest("No nets found")

        net_name = nets[0]
        solver = PDNSolver(graph, net_filter=net_name, verbose=False)
        results = solver.solve()

        # Should only solve filtered net
        self.assertEqual(len(results.net_results), 1)
        self.assertIn(net_name, results.net_results)


class TestSolverEdgeCases(unittest.TestCase):
    """Test edge cases and error handling"""

    def test_empty_net(self):
        """Test handling of empty net"""
        netlist_dir = Path(__file__).parent.parent / 'pdn' / 'netlist_small'
        if not netlist_dir.exists():
            self.skipTest("netlist_small not found")

        parser = NetlistParser(str(netlist_dir))
        graph = parser.parse()

        # Try to solve non-existent net
        solver = PDNSolver(graph, net_filter='NONEXISTENT', verbose=False)
        results = solver.solve()

        # Should handle gracefully
        self.assertEqual(len(results.net_results), 0)

    def test_solve_convergence(self):
        """Test that solver converges"""
        netlist_dir = Path(__file__).parent.parent / 'pdn' / 'netlist_small'
        if not netlist_dir.exists():
            self.skipTest("netlist_small not found")

        parser = NetlistParser(str(netlist_dir))
        graph = parser.parse()

        solver = PDNSolver(graph, verbose=False)
        results = solver.solve()

        # Check convergence
        net_name = list(results.net_results.keys())[0]
        result = results.net_results[net_name]

        # Should have valid voltages
        self.assertGreater(len(result.voltages), 0)
        self.assertGreater(result.num_nodes, 0)


class TestNetlistSmallValidation(unittest.TestCase):
    """Validation tests using netlist_small reference values"""

    @classmethod
    def setUpClass(cls):
        netlist_dir = Path(__file__).parent.parent / 'pdn' / 'netlist_small'
        if not netlist_dir.exists():
            raise unittest.SkipTest("netlist_small not found")

        parser = NetlistParser(str(netlist_dir))
        cls.graph = parser.parse()
        cls.solver = PDNSolver(cls.graph, verbose=False)
        cls.results = cls.solver.solve()

    def test_expected_net_count(self):
        """Test expected number of nets"""
        # netlist_small should have VDD_XLV net
        self.assertIn('VDD_XLV', self.results.net_results)

    def test_vdd_xlv_voltage(self):
        """Test VDD_XLV nominal voltage"""
        result = self.results.net_results.get('VDD_XLV')
        if result:
            # Check nominal voltage is reasonable
            self.assertGreater(result.nominal_voltage, 0.5)
            self.assertLess(result.nominal_voltage, 2.0)

    def test_vdd_xlv_node_count(self):
        """Test VDD_XLV has expected node count"""
        result = self.results.net_results.get('VDD_XLV')
        if result:
            # netlist_small should have thousands of nodes
            self.assertGreater(result.num_nodes, 1000)

    def test_vdd_xlv_current_sources(self):
        """Test VDD_XLV has current sources"""
        result = self.results.net_results.get('VDD_XLV')
        if result:
            # netlist_small should have many current sources
            self.assertGreater(result.num_isources, 100)
            self.assertGreater(result.total_current_ma, 0)


if __name__ == '__main__':
    unittest.main()

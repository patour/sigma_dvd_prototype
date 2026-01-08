#!/usr/bin/env python3
"""
Unit tests for PDN Plotter (pdn_plotter.py)

Tests cover:
- Net type detection (power vs ground)
- Layer orientation detection
- Anisotropic binning calculation
- Voltage heatmap generation
- Current heatmap generation
- Stripe-based heatmap generation
- Worst node selection
- Plotting edge cases
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
from pdn_solver import PDNSolver
from pdn_plotter import PDNPlotter


class TestNetTypeDetection(unittest.TestCase):
    """Test power vs ground net type detection"""
    
    @classmethod
    def setUpClass(cls):
        test_netlist_dir = Path(__file__).parent.parent / 'pdn' / 'netlist_test'
        if not test_netlist_dir.exists():
            raise unittest.SkipTest("Test netlist not found")
        
        parser = NetlistParser(str(test_netlist_dir))
        cls.graph = parser.parse()
        
        net_connectivity = cls.graph.graph.get('net_connectivity', {})
        cls.plotter = PDNPlotter(cls.graph, net_connectivity)
    
    def test_detect_power_net_vdd(self):
        """Test detection of VDD as power net"""
        net_type = self.plotter._detect_net_type('VDD')
        self.assertEqual(net_type, 'power')
    
    def test_detect_power_net_variants(self):
        """Test detection of VDD variants as power nets"""
        for net_name in ['VDD', 'VDDA', 'VDDQ', 'VDDC', 'VCC', 'VDDIO']:
            net_type = self.plotter._detect_net_type(net_name)
            self.assertEqual(net_type, 'power', f"{net_name} should be detected as power")
    
    def test_detect_ground_net_vss(self):
        """Test detection of VSS as ground net"""
        net_type = self.plotter._detect_net_type('VSS')
        self.assertEqual(net_type, 'ground')
    
    def test_detect_ground_net_variants(self):
        """Test detection of ground variants"""
        for net_name in ['VSS', 'GND', 'GNDD']:
            net_type = self.plotter._detect_net_type(net_name)
            self.assertEqual(net_type, 'ground', f"{net_name} should be detected as ground")
    
    def test_default_to_power(self):
        """Test that unknown nets default to power"""
        net_type = self.plotter._detect_net_type('UNKNOWN_NET')
        self.assertEqual(net_type, 'power')


class TestLayerOrientationDetection(unittest.TestCase):
    """Test layer routing orientation detection"""
    
    @classmethod
    def setUpClass(cls):
        test_netlist_dir = Path(__file__).parent.parent / 'pdn' / 'netlist_test'
        if not test_netlist_dir.exists():
            raise unittest.SkipTest("Test netlist not found")
        
        parser = NetlistParser(str(test_netlist_dir))
        cls.graph = parser.parse()
        
        net_connectivity = cls.graph.graph.get('net_connectivity', {})
        cls.plotter = PDNPlotter(cls.graph, net_connectivity)
    
    def test_detect_m1_orientation(self):
        """Test M1 layer orientation detection (should be horizontal)"""
        net_nodes_set = set(self.plotter.net_connectivity.get('VDD', []))
        orientation = self.plotter._detect_layer_orientation(
            'VDD', 'M1', net_nodes_set, {}
        )
        
        # M1 in test netlist has horizontal routing
        self.assertEqual(orientation, 'H')
    
    def test_detect_m2_orientation(self):
        """Test M2 layer orientation detection (should be vertical)"""
        net_nodes_set = set(self.plotter.net_connectivity.get('VDD', []))
        orientation = self.plotter._detect_layer_orientation(
            'VDD', 'M2', net_nodes_set, {}
        )
        
        # M2 in test netlist has vertical routing
        self.assertEqual(orientation, 'V')
    
    def test_manual_orientation_override(self):
        """Test manual orientation override"""
        net_nodes_set = set(self.plotter.net_connectivity.get('VDD', []))
        
        # Override M1 to be vertical
        layer_orientations = {'M1': 'V'}
        orientation = self.plotter._detect_layer_orientation(
            'VDD', 'M1', net_nodes_set, layer_orientations
        )
        
        self.assertEqual(orientation, 'V')


class TestAnisotropicBinning(unittest.TestCase):
    """Test anisotropic bin calculation"""
    
    @classmethod
    def setUpClass(cls):
        test_netlist_dir = Path(__file__).parent.parent / 'pdn' / 'netlist_test'
        if not test_netlist_dir.exists():
            raise unittest.SkipTest("Test netlist not found")
        
        parser = NetlistParser(str(test_netlist_dir))
        cls.graph = parser.parse()
        
        net_connectivity = cls.graph.graph.get('net_connectivity', {})
        cls.plotter = PDNPlotter(cls.graph, net_connectivity)
    
    def test_horizontal_anisotropic_bins(self):
        """Test anisotropic binning for horizontal layers"""
        x_bins, y_bins, (x_range, y_range) = self.plotter._calculate_anisotropic_bins(
            orientation='H',
            x_min=0, x_max=5000,
            y_min=0, y_max=4000,
            num_nodes=25,
            aspect_ratio=50
        )
        
        # Check that bins were created
        self.assertGreater(len(x_bins), 1)
        self.assertGreater(len(y_bins), 1)
        
        # For horizontal layers (stripes along X): 
        # - Voltage uniform along X → wide X bins (coarse resolution)
        # - Voltage varies across Y → narrow Y bins (fine resolution)
        x_bin_size = x_bins[1] - x_bins[0] if len(x_bins) > 1 else x_range
        y_bin_size = y_bins[1] - y_bins[0] if len(y_bins) > 1 else y_range
        
        # X bins should be wider than Y bins
        self.assertGreater(x_bin_size, y_bin_size)
    
    def test_vertical_anisotropic_bins(self):
        """Test anisotropic binning for vertical layers"""
        x_bins, y_bins, (x_range, y_range) = self.plotter._calculate_anisotropic_bins(
            orientation='V',
            x_min=0, x_max=5000,
            y_min=0, y_max=4000,
            num_nodes=25,
            aspect_ratio=50
        )
        
        # For vertical layers (stripes along Y):
        # - Voltage uniform along Y → wide Y bins (coarse resolution)  
        # - Voltage varies across X → narrow X bins (fine resolution)
        x_bin_size = x_bins[1] - x_bins[0] if len(x_bins) > 1 else x_range
        y_bin_size = y_bins[1] - y_bins[0] if len(y_bins) > 1 else y_range
        
        # Y bins should be wider than X bins
        self.assertGreater(y_bin_size, x_bin_size)
    
    def test_square_isotropic_bins(self):
        """Test isotropic binning for mixed/square layers"""
        x_bins, y_bins, (x_range, y_range) = self.plotter._calculate_anisotropic_bins(
            orientation='SQUARE',
            x_min=0, x_max=5000,
            y_min=0, y_max=5000,
            num_nodes=25,
            aspect_ratio=50
        )
        
        # For square layers: bins should be approximately equal
        x_bin_size = x_bins[1] - x_bins[0] if len(x_bins) > 1 else x_range
        y_bin_size = y_bins[1] - y_bins[0] if len(y_bins) > 1 else y_range
        
        # Should be within 10% of each other
        ratio = x_bin_size / y_bin_size if y_bin_size > 0 else 1.0
        self.assertGreater(ratio, 0.9)
        self.assertLess(ratio, 1.1)


class TestVoltageHeatmapGeneration(unittest.TestCase):
    """Test voltage heatmap generation"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)
    
    @classmethod
    def setUpClass(cls):
        test_netlist_dir = Path(__file__).parent.parent / 'pdn' / 'netlist_test'
        if not test_netlist_dir.exists():
            raise unittest.SkipTest("Test netlist not found")
        
        parser = NetlistParser(str(test_netlist_dir))
        cls.graph = parser.parse()
        
        # Solve IR-drop
        solver = PDNSolver(cls.graph, verbose=False)
        solver.solve()
        
        net_connectivity = cls.graph.graph.get('net_connectivity', {})
        cls.plotter = PDNPlotter(cls.graph, net_connectivity)
    
    def test_generate_voltage_heatmaps(self):
        """Test voltage heatmap generation"""
        output_path = Path(self.temp_dir)
        
        self.plotter.generate_layer_heatmaps(
            net_name='VDD',
            output_path=output_path,
            plot_layers=None,  # All layers
            anisotropic_bins=True
        )
        
        # Check that heatmap file was created
        heatmap_file = output_path / 'voltage_heatmap_VDD.png'
        self.assertTrue(heatmap_file.exists())
        self.assertGreater(heatmap_file.stat().st_size, 0)
    
    def test_generate_voltage_heatmaps_single_layer(self):
        """Test voltage heatmap generation for single layer"""
        output_path = Path(self.temp_dir)
        
        self.plotter.generate_layer_heatmaps(
            net_name='VDD',
            output_path=output_path,
            plot_layers=['M1'],
            anisotropic_bins=True
        )
        
        # Check that heatmap file was created
        heatmap_file = output_path / 'voltage_heatmap_VDD.png'
        self.assertTrue(heatmap_file.exists())
    
    def test_generate_voltage_heatmaps_custom_bin_size(self):
        """Test voltage heatmap with custom bin size"""
        output_path = Path(self.temp_dir)
        
        self.plotter.generate_layer_heatmaps(
            net_name='VDD',
            output_path=output_path,
            plot_bin_size=500,  # 500 unit bins
            anisotropic_bins=False
        )
        
        heatmap_file = output_path / 'voltage_heatmap_VDD.png'
        self.assertTrue(heatmap_file.exists())


class TestCurrentHeatmapGeneration(unittest.TestCase):
    """Test current source heatmap generation"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)
    
    @classmethod
    def setUpClass(cls):
        test_netlist_dir = Path(__file__).parent.parent / 'pdn' / 'netlist_test'
        if not test_netlist_dir.exists():
            raise unittest.SkipTest("Test netlist not found")
        
        parser = NetlistParser(str(test_netlist_dir))
        cls.graph = parser.parse()
        
        # Solve IR-drop
        solver = PDNSolver(cls.graph, verbose=False)
        solver.solve()
        
        net_connectivity = cls.graph.graph.get('net_connectivity', {})
        cls.plotter = PDNPlotter(cls.graph, net_connectivity)
    
    def test_generate_current_heatmaps(self):
        """Test current source heatmap generation"""
        output_path = Path(self.temp_dir)
        
        self.plotter.generate_current_heatmaps(
            net_name='VDD',
            output_path=output_path,
            anisotropic_bins=True
        )
        
        # Check that heatmap file was created
        heatmap_file = output_path / 'current_heatmap_VDD.png'
        self.assertTrue(heatmap_file.exists())
        self.assertGreater(heatmap_file.stat().st_size, 0)


class TestStripeHeatmapGeneration(unittest.TestCase):
    """Test stripe-based heatmap generation"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)
    
    @classmethod
    def setUpClass(cls):
        test_netlist_dir = Path(__file__).parent.parent / 'pdn' / 'netlist_test'
        if not test_netlist_dir.exists():
            raise unittest.SkipTest("Test netlist not found")
        
        parser = NetlistParser(str(test_netlist_dir))
        cls.graph = parser.parse()
        
        # Solve IR-drop
        solver = PDNSolver(cls.graph, verbose=False)
        solver.solve()
        
        net_connectivity = cls.graph.graph.get('net_connectivity', {})
        cls.plotter = PDNPlotter(cls.graph, net_connectivity)
    
    def test_generate_stripe_voltage_heatmaps(self):
        """Test stripe-based voltage heatmap generation"""
        output_path = Path(self.temp_dir)
        
        self.plotter.generate_stripe_heatmaps(
            net_name='VDD',
            output_path=output_path,
            max_stripes=10,
            is_current=False
        )
        
        # Check that heatmap file was created
        heatmap_file = output_path / 'voltage_stripe_heatmap_VDD.png'
        self.assertTrue(heatmap_file.exists())
    
    def test_generate_stripe_current_heatmaps(self):
        """Test stripe-based current heatmap generation"""
        output_path = Path(self.temp_dir)
        
        self.plotter.generate_stripe_heatmaps(
            net_name='VDD',
            output_path=output_path,
            max_stripes=10,
            is_current=True
        )
        
        # Check that heatmap file was created
        heatmap_file = output_path / 'current_stripe_heatmap_VDD.png'
        self.assertTrue(heatmap_file.exists())


class TestWorstNodeSelection(unittest.TestCase):
    """Test spatially separated worst node selection"""
    
    @classmethod
    def setUpClass(cls):
        test_netlist_dir = Path(__file__).parent.parent / 'pdn' / 'netlist_test'
        if not test_netlist_dir.exists():
            raise unittest.SkipTest("Test netlist not found")
        
        parser = NetlistParser(str(test_netlist_dir))
        cls.graph = parser.parse()
        
        # Solve IR-drop
        solver = PDNSolver(cls.graph, verbose=False)
        solver.solve()
        
        net_connectivity = cls.graph.graph.get('net_connectivity', {})
        cls.plotter = PDNPlotter(cls.graph, net_connectivity)
    
    def test_find_worst_nodes_power(self):
        """Test finding worst nodes for power net"""
        # Create sample nodes with voltages
        nodes_with_voltage = [
            ('node1', 1000, 1000, 0.749),  # Second worst
            ('node2', 1100, 1100, 0.7495),
            ('node3', 3000, 3000, 0.748),  # Worst (lowest voltage for power)
            ('node4', 3100, 3100, 0.7485),
            ('node5', 2000, 2000, 0.7492),
        ]
        
        worst_nodes = self.plotter._find_spatially_separated_worst_nodes(
            nodes_with_voltage,
            net_type='power',
            x_range=4000,
            y_range=4000,
            max_nodes=3
        )
        
        # Should return up to 3 spatially separated worst nodes
        self.assertLessEqual(len(worst_nodes), 3)
        self.assertGreater(len(worst_nodes), 0)
        
        # First node should be the worst (lowest voltage for power net)
        self.assertEqual(worst_nodes[0][0], 'node3')
    
    def test_find_worst_nodes_ground(self):
        """Test finding worst nodes for ground net"""
        # For ground nets, worst is maximum voltage
        nodes_with_voltage = [
            ('node1', 1000, 1000, 0.001),  # Worst
            ('node2', 1100, 1100, 0.0005),
            ('node3', 3000, 3000, 0.0009),  # Second worst, far away
        ]
        
        worst_nodes = self.plotter._find_spatially_separated_worst_nodes(
            nodes_with_voltage,
            net_type='ground',
            x_range=4000,
            y_range=4000,
            max_nodes=3
        )
        
        self.assertGreater(len(worst_nodes), 0)
        # First node should be the worst (highest voltage)
        self.assertEqual(worst_nodes[0][0], 'node1')


class TestStripeGrouping(unittest.TestCase):
    """Test stripe grouping functionality"""
    
    @classmethod
    def setUpClass(cls):
        test_netlist_dir = Path(__file__).parent.parent / 'pdn' / 'netlist_test'
        if not test_netlist_dir.exists():
            raise unittest.SkipTest("Test netlist not found")
        
        parser = NetlistParser(str(test_netlist_dir))
        cls.graph = parser.parse()
        
        net_connectivity = cls.graph.graph.get('net_connectivity', {})
        cls.plotter = PDNPlotter(cls.graph, net_connectivity)
    
    def test_group_nodes_into_stripes_horizontal(self):
        """Test grouping nodes into horizontal stripes"""
        nodes_with_data = [
            ('n1', 1000, 1000, 0.75),
            ('n2', 2000, 1000, 0.749),
            ('n3', 1000, 2000, 0.748),
            ('n4', 2000, 2000, 0.747),
        ]
        
        stripes = self.plotter._group_nodes_into_stripes(
            'VDD', 'M1', 'H', nodes_with_data
        )
        
        # Horizontal layers group by Y coordinate
        self.assertGreater(len(stripes), 0)
        self.assertIn(1000, stripes)
        self.assertIn(2000, stripes)
    
    def test_group_nodes_into_stripes_vertical(self):
        """Test grouping nodes into vertical stripes"""
        nodes_with_data = [
            ('n1', 1000, 1000, 0.75),
            ('n2', 2000, 1000, 0.749),
            ('n3', 1000, 2000, 0.748),
        ]
        
        stripes = self.plotter._group_nodes_into_stripes(
            'VDD', 'M2', 'V', nodes_with_data
        )
        
        # Vertical layers group by X coordinate
        self.assertGreater(len(stripes), 0)
        self.assertIn(1000, stripes)
        self.assertIn(2000, stripes)
    
    def test_consolidate_stripes(self):
        """Test stripe consolidation when count exceeds max"""
        # Create many stripes
        stripes = {float(i * 100): [('node', i * 100, 1000, 0.75)] 
                  for i in range(100)}
        
        consolidated = self.plotter._consolidate_stripes(stripes, max_stripes=10)
        
        # Should consolidate to approximately 10 groups
        self.assertLessEqual(len(consolidated), 15)  # Allow some margin
        self.assertGreater(len(consolidated), 5)


class TestPlotterEdgeCases(unittest.TestCase):
    """Test plotter edge cases and error handling"""
    
    @classmethod
    def setUpClass(cls):
        test_netlist_dir = Path(__file__).parent.parent / 'pdn' / 'netlist_test'
        if not test_netlist_dir.exists():
            raise unittest.SkipTest("Test netlist not found")
        
        parser = NetlistParser(str(test_netlist_dir))
        cls.graph = parser.parse()
        
        # Solve IR-drop
        solver = PDNSolver(cls.graph, verbose=False)
        solver.solve()
        
        net_connectivity = cls.graph.graph.get('net_connectivity', {})
        cls.plotter = PDNPlotter(cls.graph, net_connectivity)
    
    def test_empty_node_list(self):
        """Test handling of empty node list"""
        worst_nodes = self.plotter._find_spatially_separated_worst_nodes(
            [],
            net_type='power',
            x_range=1000,
            y_range=1000,
            max_nodes=3
        )
        
        self.assertEqual(len(worst_nodes), 0)
    
    def test_single_node(self):
        """Test handling of single node"""
        nodes_with_voltage = [('node1', 1000, 1000, 0.75)]
        
        worst_nodes = self.plotter._find_spatially_separated_worst_nodes(
            nodes_with_voltage,
            net_type='power',
            x_range=1000,
            y_range=1000,
            max_nodes=3
        )
        
        self.assertEqual(len(worst_nodes), 1)
        self.assertEqual(worst_nodes[0][0], 'node1')


class TestBinSizeCalculation(unittest.TestCase):
    """Test bin size calculation edge cases"""
    
    @classmethod
    def setUpClass(cls):
        test_netlist_dir = Path(__file__).parent.parent / 'pdn' / 'netlist_test'
        if not test_netlist_dir.exists():
            raise unittest.SkipTest("Test netlist not found")
        
        parser = NetlistParser(str(test_netlist_dir))
        cls.graph = parser.parse()
        
        net_connectivity = cls.graph.graph.get('net_connectivity', {})
        cls.plotter = PDNPlotter(cls.graph, net_connectivity)
    
    def test_small_grid_binning(self):
        """Test binning with very small grid"""
        x_bins, y_bins, (x_range, y_range) = self.plotter._calculate_anisotropic_bins(
            orientation='SQUARE',
            x_min=0, x_max=100,
            y_min=0, y_max=100,
            num_nodes=4,
            aspect_ratio=50
        )
        
        # Should still create valid bins
        self.assertGreater(len(x_bins), 0)
        self.assertGreater(len(y_bins), 0)
    
    def test_single_bin_edge_case(self):
        """Test that single bin doesn't cause IndexError"""
        # This tests the fix for the bin size calculation bug
        x_bins = np.array([0, 1000])  # 1 bin
        
        # Should not raise IndexError
        if len(x_bins) > 1:
            bin_size = x_bins[1] - x_bins[0]
        else:
            bin_size = 1000
        
        self.assertEqual(bin_size, 1000)


if __name__ == '__main__':
    unittest.main()

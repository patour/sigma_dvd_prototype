#!/usr/bin/env python3
"""
Unit tests for PDN Parser (pdn_parser.py)

Tests cover:
- SPICE line reading (gzip detection, line continuation, comments)
- Element parsing (resistors, capacitors, inductors, voltage sources, current sources)
- Node coordinate extraction
- Node-net mapping from .nd files
- Tile parsing
- Package parsing with union-find
- Net filtering
- Validation (shorts, floating nodes)
- Graph structure and metadata
"""

import unittest
import sys
import os
import tempfile
import gzip
from pathlib import Path

# Add pdn directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'pdn'))

from pdn_parser import (
    NetlistParser, SpiceLineReader, GraphBuilder,
    R_TO_KOHM, C_TO_FF, L_TO_NH, I_TO_MA, SHORT_THRESHOLD
)


class TestSpiceLineReader(unittest.TestCase):
    """Test SPICE file reading with gzip detection and line continuation"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_plain_text_file(self):
        """Test reading plain text SPICE file"""
        filepath = Path(self.temp_dir) / "test.sp"
        with open(filepath, 'w') as f:
            f.write("R1 n1 n2 100\n")
            f.write("C1 n3 n4 1e-12\n")
        
        with SpiceLineReader(str(filepath)) as reader:
            self.assertFalse(reader.is_gzipped)
            line1 = reader.read_line()
            self.assertEqual(line1, "R1 n1 n2 100")
            line2 = reader.read_line()
            self.assertEqual(line2, "C1 n3 n4 1e-12")
    
    def test_gzipped_file(self):
        """Test automatic gzip detection and reading"""
        filepath = Path(self.temp_dir) / "test.sp.gz"
        with gzip.open(filepath, 'wt') as f:
            f.write("R1 n1 n2 100\n")
            f.write("C1 n3 n4 1e-12\n")
        
        with SpiceLineReader(str(filepath)) as reader:
            self.assertTrue(reader.is_gzipped)
            line1 = reader.read_line()
            self.assertEqual(line1, "R1 n1 n2 100")
    
    def test_line_continuation(self):
        """Test line continuation with + prefix"""
        filepath = Path(self.temp_dir) / "test.sp"
        with open(filepath, 'w') as f:
            f.write("R1 n1 n2\n")
            f.write("+ 100\n")
        
        with SpiceLineReader(str(filepath)) as reader:
            line = reader.read_line()
            self.assertEqual(line, "R1 n1 n2 100")
    
    def test_comment_removal(self):
        """Test comment line removal"""
        filepath = Path(self.temp_dir) / "test.sp"
        with open(filepath, 'w') as f:
            f.write("* This is a comment\n")
            f.write("R1 n1 n2 100\n")
            f.write("* Another comment\n")
        
        with SpiceLineReader(str(filepath)) as reader:
            line = reader.read_line()
            self.assertEqual(line, "R1 n1 n2 100")
            self.assertIsNone(reader.read_line())
    
    def test_whitespace_normalization(self):
        """Test multiple whitespace normalization"""
        filepath = Path(self.temp_dir) / "test.sp"
        with open(filepath, 'w') as f:
            f.write("R1    n1   n2    100\n")
        
        with SpiceLineReader(str(filepath)) as reader:
            line = reader.read_line()
            self.assertEqual(line, "R1 n1 n2 100")


class TestGraphBuilder(unittest.TestCase):
    """Test graph building and element parsing"""
    
    def setUp(self):
        self.builder = GraphBuilder()
    
    def test_add_node(self):
        """Test node addition with attributes"""
        self.builder.add_node("n1", x=1000, y=2000, layer="M1")
        self.assertIn("n1", self.builder.graph.nodes)
        self.assertEqual(self.builder.graph.nodes["n1"]["x"], 1000)
        self.assertEqual(self.builder.graph.nodes["n1"]["y"], 2000)
        self.assertEqual(self.builder.graph.nodes["n1"]["layer"], "M1")
    
    def test_coordinate_extraction(self):
        """Test coordinate extraction from node names"""
        # 3D pattern: X_Y_LAYER
        coords = self.builder._extract_coordinates("1000_2000_M1")
        self.assertEqual(coords, {'x': 1000, 'y': 2000, 'layer': 'M1'})
        
        # 2D pattern: X_Y
        coords = self.builder._extract_coordinates("1000_2000")
        self.assertEqual(coords, {'x': 1000, 'y': 2000})
        
        # Numeric layer
        coords = self.builder._extract_coordinates("1000_2000_5")
        self.assertEqual(coords, {'x': 1000, 'y': 2000, 'layer': '5'})
    
    def test_add_resistor(self):
        """Test resistor element addition"""
        self.builder.add_element('R', 'n1', 'n2', 0.1, 'R1')
        
        self.assertIn('n1', self.builder.graph.nodes)
        self.assertIn('n2', self.builder.graph.nodes)
        self.assertEqual(self.builder.stats.resistors, 1)
        
        edges = list(self.builder.graph.edges(data=True))
        self.assertEqual(len(edges), 1)
        u, v, data = edges[0]
        self.assertEqual(data['type'], 'R')
        self.assertEqual(data['value'], 0.1)
        self.assertEqual(data['elem_name'], 'R1')
    
    def test_add_capacitor(self):
        """Test capacitor element addition"""
        self.builder.add_element('C', 'n1', '0', 1e12, 'C1')
        self.assertEqual(self.builder.stats.capacitors, 1)
    
    def test_add_voltage_source(self):
        """Test voltage source addition"""
        self.builder.add_element('V', 'vdd', '0', 0.75, 'vVDD')
        self.assertEqual(self.builder.stats.vsources, 1)
    
    def test_add_current_source(self):
        """Test current source addition"""
        self.builder.add_element('I', 'n1', '0', 5.0, 'I1')
        self.assertEqual(self.builder.stats.isources, 1)
    
    def test_boundary_node_marking(self):
        """Test boundary node marking"""
        self.builder.mark_boundary_node("n_boundary")
        self.assertIn("n_boundary", self.builder.boundary_nodes)
        self.assertEqual(self.builder.stats.boundary_nodes, 1)
    
    def test_union_find(self):
        """Test union-find for package node connectivity"""
        self.builder._uf_union("pkg1", "pkg2")
        root1 = self.builder._uf_find("pkg1")
        root2 = self.builder._uf_find("pkg2")
        self.assertEqual(root1, root2)


class TestNetlistParser(unittest.TestCase):
    """Test full netlist parsing with test netlist"""
    
    @classmethod
    def setUpClass(cls):
        """Parse test netlist once for all tests"""
        test_netlist_dir = Path(__file__).parent.parent / 'pdn' / 'netlist_test'
        if not test_netlist_dir.exists():
            raise unittest.SkipTest("Test netlist not found")
        
        cls.parser = NetlistParser(str(test_netlist_dir), validate=True)
        cls.graph = cls.parser.parse()
    
    def test_graph_creation(self):
        """Test that graph was created successfully"""
        self.assertIsNotNone(self.graph)
        self.assertGreater(self.graph.number_of_nodes(), 0)
        self.assertGreater(self.graph.number_of_edges(), 0)
    
    def test_node_count(self):
        """Test expected node count from test netlist"""
        # Test netlist has: 25 M1 nodes + 25 M2 nodes + package nodes
        self.assertGreaterEqual(self.graph.number_of_nodes(), 50)
    
    def test_resistor_parsing(self):
        """Test resistor parsing from tiles"""
        resistors = [(u,v,d) for u, v, d in self.graph.edges(data=True) if d['type'] == 'R']
        self.assertGreater(len(resistors), 40)  # M1 + M2 + via resistors
    
    def test_capacitor_parsing(self):
        """Test capacitor parsing"""
        capacitors = [(u,v,d) for u, v, d in self.graph.edges(data=True) if d['type'] == 'C']
        self.assertGreater(len(capacitors), 25)  # Decoupling caps
    
    def test_voltage_source_parsing(self):
        """Test voltage source parsing from package"""
        vsources = [(u,v,d) for u, v, d in self.graph.edges(data=True) if d['type'] == 'V']
        self.assertGreater(len(vsources), 0)
    
    def test_current_source_parsing(self):
        """Test current source parsing from instanceModels"""
        isources = [(u,v,d) for u, v, d in self.graph.edges(data=True) if d['type'] == 'I']
        self.assertEqual(len(isources), 17)  # Test netlist has 17 current sources
    
    def test_node_attributes(self):
        """Test node coordinate and layer attributes"""
        # Check a known node from test netlist
        node = '1000_1000_M1'
        if node in self.graph.nodes:
            attrs = self.graph.nodes[node]
            self.assertEqual(attrs.get('x'), 1000)
            self.assertEqual(attrs.get('y'), 1000)
            self.assertEqual(attrs.get('layer'), 'M1')
    
    def test_net_connectivity(self):
        """Test net connectivity metadata"""
        net_connectivity = self.graph.graph.get('net_connectivity', {})
        self.assertIn('VDD', net_connectivity)
        self.assertGreater(len(net_connectivity['VDD']), 0)
    
    def test_instance_node_mapping(self):
        """Test instance-to-node mapping for current sources"""
        inst_map = self.graph.graph.get('instance_node_map', {})
        self.assertGreater(len(inst_map), 0)
        
        # Check that instances have node lists
        for inst_name, nodes in inst_map.items():
            self.assertIsInstance(nodes, list)
            self.assertGreater(len(nodes), 0)
    
    def test_vsrc_dict(self):
        """Test voltage source dictionary"""
        vsrc_dict = self.graph.graph.get('vsrc_dict', {})
        self.assertGreater(len(vsrc_dict), 0)
    
    def test_parameters(self):
        """Test parameter extraction"""
        params = self.graph.graph.get('parameters', {})
        self.assertIn('VDD', params)
        self.assertEqual(params['VDD'], '0.75')
    
    def test_layer_statistics(self):
        """Test per-layer statistics"""
        layer_stats = self.graph.graph.get('layer_stats_by_net', {})
        self.assertIn('VDD', layer_stats)
        
        vdd_layers = layer_stats['VDD']
        self.assertIn('M1', vdd_layers)
        self.assertIn('M2', vdd_layers)
        
        # Check that M1 layer has nodes and resistors
        m1_stats = vdd_layers['M1']
        self.assertGreater(m1_stats['nodes'], 0)
        self.assertGreater(m1_stats['resistors'], 0)
    
    def test_tile_grid(self):
        """Test tile grid metadata"""
        tile_grid = self.graph.graph.get('tile_grid')
        self.assertEqual(tile_grid, (1, 1))
    
    def test_validation_shorts(self):
        """Test short detection in validation"""
        # Test netlist has zero-resistance rs resistors (intentional)
        shorts = []
        for u, v, k, d in self.graph.edges(keys=True, data=True):
            if d.get('type') == 'R' and d.get('value', float('inf')) < SHORT_THRESHOLD:
                shorts.append(d.get('elem_name'))
        
        # Package has 4 zero-resistance bump connections
        self.assertGreaterEqual(len(shorts), 4)


class TestNetFiltering(unittest.TestCase):
    """Test net filtering functionality"""
    
    def test_net_filter_vdd(self):
        """Test parsing with VDD net filter"""
        test_netlist_dir = Path(__file__).parent.parent / 'pdn' / 'netlist_test'
        if not test_netlist_dir.exists():
            self.skipTest("Test netlist not found")
        
        parser = NetlistParser(str(test_netlist_dir), net_filter='VDD')
        graph = parser.parse()
        
        # All nodes should be VDD net
        net_conn = graph.graph.get('net_connectivity', {})
        self.assertIn('VDD', net_conn)
        
        # No other nets should have nodes
        for net_name, nodes in net_conn.items():
            if net_name != 'VDD':
                self.assertEqual(len(nodes), 0, f"Net {net_name} should have no nodes with VDD filter")


class TestValueParsing(unittest.TestCase):
    """Test SPICE value parsing with suffixes"""
    
    def setUp(self):
        from pathlib import Path
        test_netlist_dir = Path(__file__).parent.parent / 'pdn' / 'netlist_test'
        self.parser = NetlistParser(str(test_netlist_dir))
    
    def test_parse_value_basic(self):
        """Test basic numeric value parsing"""
        self.assertEqual(self.parser._parse_value("100"), 100.0)
        self.assertEqual(self.parser._parse_value("1.5"), 1.5)
    
    def test_parse_value_with_suffixes(self):
        """Test value parsing with SPICE suffixes"""
        self.assertEqual(self.parser._parse_value("1K"), 1000.0)
        self.assertEqual(self.parser._parse_value("1M"), 0.001)
        self.assertEqual(self.parser._parse_value("1U"), 1e-6)
        self.assertEqual(self.parser._parse_value("1N"), 1e-9)
        self.assertEqual(self.parser._parse_value("1P"), 1e-12)
        self.assertEqual(self.parser._parse_value("1F"), 1e-15)
        self.assertEqual(self.parser._parse_value("1MEG"), 1e6)
        self.assertEqual(self.parser._parse_value("1G"), 1e9)
    
    def test_parse_value_with_parameter(self):
        """Test value parsing with parameter reference"""
        self.parser.builder.parameters['VDD'] = '0.75'
        self.assertEqual(self.parser._parse_value("VDD"), 0.75)


class TestNodeNetMapping(unittest.TestCase):
    """Test node-to-net mapping from .nd files"""
    
    def test_nd_file_loading(self):
        """Test loading node-net map from .nd file"""
        test_netlist_dir = Path(__file__).parent.parent / 'pdn' / 'netlist_test'
        nd_file = test_netlist_dir / 'tile_0_0.nd'
        
        if not nd_file.exists():
            self.skipTest("Test .nd file not found")
        
        parser = NetlistParser(str(test_netlist_dir))
        node_map = parser._load_node_net_map(str(nd_file))
        
        # Check that nodes are mapped to VDD
        self.assertGreater(len(node_map), 0)
        
        # All test netlist nodes should map to VDD
        for node, net in node_map.items():
            self.assertEqual(net, 'VDD')


if __name__ == '__main__':
    unittest.main()

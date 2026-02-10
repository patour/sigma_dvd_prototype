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
    R_TO_KOHM, C_TO_FF, L_TO_NH, I_TO_MA, SHORT_THRESHOLD,
    # Waveform parsing classes and functions
    InstanceInfo, Pulse, PWL, CurrentSource, _DCOnlyCurrentSource,
    _parse_spice_value, _parse_pulse, _parse_pwl, _parse_current_source_line,
    # Wscale control functions (thread-safe via ContextVar)
    get_apply_wscale, set_apply_wscale,
    # DC-only optimization control functions
    get_optimize_dc_only, set_optimize_dc_only,
    # Node attributes class
    PDNNodeAttrs
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
        """Test node addition with attributes.

        Note: x, y, layer are computed from node name (X_Y_LAYER pattern) and cannot be set directly.
        """
        # Test die node with X_Y_LAYER pattern - coordinates extracted from name
        self.builder.add_node("1000_2000_M1")
        self.assertIn("1000_2000_M1", self.builder.graph)
        attrs = self.builder.graph.nodes_dict["1000_2000_M1"]
        self.assertEqual(attrs.x, 1000)
        self.assertEqual(attrs.y, 2000)
        self.assertEqual(attrs.layer, "M1")
        self.assertTrue(attrs.is_die)

        # Test package node (no coordinate extraction)
        self.builder.current_file_type = 'package'
        self.builder.add_node("VDD_vsrc")
        self.assertIn("VDD_vsrc", self.builder.graph)
        pkg_attrs = self.builder.graph.nodes_dict["VDD_vsrc"]
        self.assertIsNone(pkg_attrs.x)
        self.assertIsNone(pkg_attrs.y)
        self.assertIsNone(pkg_attrs.layer)
        self.assertTrue(pkg_attrs.is_package)
    
    def test_coordinate_extraction(self):
        """Test coordinate extraction from node names via PDNNodeAttrs properties.

        Coordinates are now computed on-demand from node name, not stored.
        """
        from pdn.pdn_parser import PDNNodeAttrs

        # 3D pattern: X_Y_LAYER
        attrs = PDNNodeAttrs(name="1000_2000_M1", flags=PDNNodeAttrs.FLAG_DIE)
        self.assertEqual(attrs.x, 1000)
        self.assertEqual(attrs.y, 2000)
        self.assertEqual(attrs.layer, 'M1')

        # 2D pattern: X_Y (no layer)
        attrs_2d = PDNNodeAttrs(name="1000_2000", flags=PDNNodeAttrs.FLAG_DIE)
        self.assertEqual(attrs_2d.x, 1000)
        self.assertEqual(attrs_2d.y, 2000)
        self.assertIsNone(attrs_2d.layer)

        # Numeric layer
        attrs_num = PDNNodeAttrs(name="1000_2000_5", flags=PDNNodeAttrs.FLAG_DIE)
        self.assertEqual(attrs_num.x, 1000)
        self.assertEqual(attrs_num.y, 2000)
        self.assertEqual(attrs_num.layer, '5')

        # Package node (not die) - no coordinate extraction
        pkg_attrs = PDNNodeAttrs(name="VDD_vsrc", flags=PDNNodeAttrs.FLAG_PACKAGE)
        self.assertIsNone(pkg_attrs.x)
        self.assertIsNone(pkg_attrs.y)
        self.assertIsNone(pkg_attrs.layer)
    
    def test_add_resistor(self):
        """Test resistor element addition"""
        self.builder.add_element('R', 'n1', 'n2', 0.1, 'R1')

        self.assertIn('n1', self.builder.graph)
        self.assertIn('n2', self.builder.graph)
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
    
    def test_net_filter_excludes_other_net_elements(self):
        """Test that net_filter excludes elements from other nets"""
        # Create builder with VDD filter
        builder = GraphBuilder(net_filter='VDD')
        
        # Set up node-to-net mapping for both VDD and VSS nodes
        builder.node_net_map = {
            'vdd_node1': 'VDD',
            'vdd_node2': 'VDD',
            'vss_node1': 'VSS',
            'vss_node2': 'VSS',
        }
        builder.node_net_map_lower = {k: v.lower() for k, v in builder.node_net_map.items()}
        
        # Set file type to 'instance' (not 'package') so filtering applies
        builder.current_file_type = 'instance'
        
        # Add VDD resistor - should be added
        added_vdd = builder.add_element('R', 'vdd_node1', 'vdd_node2', 0.1, 'R_VDD')
        self.assertTrue(added_vdd, "VDD element should be added")
        
        # Add VSS resistor - should be filtered out
        added_vss = builder.add_element('R', 'vss_node1', 'vss_node2', 0.1, 'R_VSS')
        self.assertFalse(added_vss, "VSS element should be filtered out")
        
        # Verify only VDD resistor is in graph
        self.assertEqual(builder.stats.resistors, 1)
    
    def test_net_filter_current_source_filtering(self):
        """Test that net_filter filters current sources and instance_sources consistently"""
        # Create builder with VDD filter
        builder = GraphBuilder(net_filter='VDD')
        
        # Set up node-to-net mapping
        builder.node_net_map = {
            'vdd_node': 'VDD',
            'vss_node': 'VSS',
        }
        builder.node_net_map_lower = {k: v.lower() for k, v in builder.node_net_map.items()}
        builder.current_file_type = 'instance'
        
        # Add VDD current source - should be added
        added_vdd = builder.add_element('I', 'vdd_node', '0', 1.0, 'I_VDD')
        self.assertTrue(added_vdd, "VDD current source should be added")
        
        # Add VSS current source - should be filtered out
        added_vss = builder.add_element('I', 'vss_node', '0', 1.0, 'I_VSS')
        self.assertFalse(added_vss, "VSS current source should be filtered out")
        
        # Only VDD current source should be counted
        self.assertEqual(builder.stats.isources, 1)


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
        if node in self.graph:
            attrs = self.graph.nodes_dict[node]
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
    
    def test_net_filter_instance_sources_match_edges(self):
        """Test that instance_sources count matches I-type edges when net_filter is active"""
        test_netlist_dir = Path(__file__).parent.parent / 'pdn' / 'netlist_test'
        if not test_netlist_dir.exists():
            self.skipTest("Test netlist not found")
        
        parser = NetlistParser(str(test_netlist_dir), net_filter='VDD')
        graph = parser.parse()
        
        # Get instance_sources (raw objects since store_instance_sources=False by default)
        raw_sources = graph.graph.get('_instance_sources_objects', {})
        
        # Count I-type edges
        isource_edges = [(u, v, d) for u, v, d in graph.edges(data=True) 
                        if d.get('type') == 'I']
        
        # instance_sources should match I-type edge count
        self.assertEqual(len(raw_sources), len(isource_edges),
                        f"instance_sources count ({len(raw_sources)}) should match "
                        f"I-type edge count ({len(isource_edges)}) when net_filter is active")
    
    def test_net_filter_instance_node_map_match_edges(self):
        """Test that instance_node_map count matches I-type edges when net_filter is active"""
        test_netlist_dir = Path(__file__).parent.parent / 'pdn' / 'netlist_test'
        if not test_netlist_dir.exists():
            self.skipTest("Test netlist not found")
        
        parser = NetlistParser(str(test_netlist_dir), net_filter='VDD')
        graph = parser.parse()
        
        # Get instance_node_map
        inst_node_map = graph.graph.get('instance_node_map', {})
        
        # Count I-type edges
        isource_edges = [(u, v, d) for u, v, d in graph.edges(data=True) 
                        if d.get('type') == 'I']
        
        # instance_node_map should match I-type edge count
        self.assertEqual(len(inst_node_map), len(isource_edges),
                        f"instance_node_map count ({len(inst_node_map)}) should match "
                        f"I-type edge count ({len(isource_edges)}) when net_filter is active")
    
    def test_net_filter_instance_sources_serialized_match_edges(self):
        """Test that serialized instance_sources matches I-type edges when net_filter is active"""
        test_netlist_dir = Path(__file__).parent.parent / 'pdn' / 'netlist_test'
        if not test_netlist_dir.exists():
            self.skipTest("Test netlist not found")
        
        parser = NetlistParser(str(test_netlist_dir), net_filter='VDD', 
                              store_instance_sources=True)
        graph = parser.parse()
        
        # Get serialized instance_sources
        inst_sources = graph.graph.get('instance_sources', {})
        
        # Count I-type edges
        isource_edges = [(u, v, d) for u, v, d in graph.edges(data=True) 
                        if d.get('type') == 'I']
        
        # instance_sources should match I-type edge count
        self.assertEqual(len(inst_sources), len(isource_edges),
                        f"serialized instance_sources count ({len(inst_sources)}) should match "
                        f"I-type edge count ({len(isource_edges)}) when net_filter is active")


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


class TestValidation(unittest.TestCase):
    """Test validation methods: _check_shorts and _check_floating_nodes"""

    def setUp(self):
        """Create a temporary directory for test netlists"""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary directory"""
        import shutil
        shutil.rmtree(self.temp_dir)

    def _create_minimal_netlist(self, tile_content, nd_content, package_content="",
                                 additional_vsrcs="", pg_net_voltage="VDD 1.0"):
        """Helper to create a minimal netlist directory structure"""
        netlist_dir = Path(self.temp_dir) / "test_netlist"
        netlist_dir.mkdir()

        # ckt.sp - main file
        ckt_sp = f""".partition_info 1 1
.include tile_0_0.ckt
.include package.ckt
"""
        (netlist_dir / "ckt.sp").write_text(ckt_sp)

        # tile_0_0.ckt
        (netlist_dir / "tile_0_0.ckt").write_text(tile_content)

        # tile_0_0.nd
        (netlist_dir / "tile_0_0.nd").write_text(nd_content)

        # package.ckt
        (netlist_dir / "package.ckt").write_text(package_content)

        # pg_net_voltage
        (netlist_dir / "pg_net_voltage").write_text(pg_net_voltage)

        # additional_vsrcs
        (netlist_dir / "additional_vsrcs").write_text(additional_vsrcs)

        return str(netlist_dir)

    def test_check_shorts_detects_zero_resistance(self):
        """Test that _check_shorts detects resistors below SHORT_THRESHOLD"""
        tile_content = """
R1 1000_1000_M1 2000_1000_M1 0.01
R2 2000_1000_M1 3000_1000_M1 0.0000001
R3 3000_1000_M1 4000_1000_M1 0.01
"""
        nd_content = """1000 1000 M1 1000_1000_M1 VDD
2000 1000 M1 2000_1000_M1 VDD
3000 1000 M1 3000_1000_M1 VDD
4000 1000 M1 4000_1000_M1 VDD
"""
        package_content = "VVDD VDD_vsrc 0 1.0\nRs VDD_vsrc 1000_1000_M1 0.0"

        netlist_dir = self._create_minimal_netlist(tile_content, nd_content, package_content)

        parser = NetlistParser(netlist_dir, validate=True)
        graph = parser.parse()

        # Should detect 2 shorts: R2 (0.0000001) and Rs (0.0)
        self.assertEqual(parser.builder.stats.shorted_elements, 2)

    def test_check_shorts_no_shorts(self):
        """Test that _check_shorts reports zero when no shorts exist"""
        tile_content = """
R1 1000_1000_M1 2000_1000_M1 0.01
R2 2000_1000_M1 3000_1000_M1 0.01
"""
        nd_content = """1000 1000 M1 1000_1000_M1 VDD
2000 1000 M1 2000_1000_M1 VDD
3000 1000 M1 3000_1000_M1 VDD
"""
        package_content = "VVDD VDD_vsrc 0 1.0\nRs VDD_vsrc 1000_1000_M1 0.001"

        netlist_dir = self._create_minimal_netlist(tile_content, nd_content, package_content)

        parser = NetlistParser(netlist_dir, validate=True)
        graph = parser.parse()

        self.assertEqual(parser.builder.stats.shorted_elements, 0)

    def test_check_floating_nodes_detects_disconnected(self):
        """Test that _check_floating_nodes detects nodes not connected to voltage source"""
        # Create two disconnected components - one with vsrc, one without
        tile_content = """
R1 1000_1000_M1 2000_1000_M1 0.01
R2 5000_5000_M1 6000_5000_M1 0.01
"""
        nd_content = """1000 1000 M1 1000_1000_M1 VDD
2000 1000 M1 2000_1000_M1 VDD
5000 5000 M1 5000_5000_M1 VDD
6000 5000 M1 6000_5000_M1 VDD
"""
        # Voltage source only connects to the first component
        package_content = "VVDD VDD_vsrc 0 1.0\nRs VDD_vsrc 1000_1000_M1 0.001"

        netlist_dir = self._create_minimal_netlist(tile_content, nd_content, package_content)

        parser = NetlistParser(netlist_dir, validate=True)
        graph = parser.parse()

        # Second component (5000_5000_M1, 6000_5000_M1) should be floating
        self.assertEqual(parser.builder.stats.floating_nodes, 2)

    def test_check_floating_nodes_all_connected(self):
        """Test that _check_floating_nodes reports zero when all nodes connected"""
        tile_content = """
R1 1000_1000_M1 2000_1000_M1 0.01
R2 2000_1000_M1 3000_1000_M1 0.01
"""
        nd_content = """1000 1000 M1 1000_1000_M1 VDD
2000 1000 M1 2000_1000_M1 VDD
3000 1000 M1 3000_1000_M1 VDD
"""
        package_content = "VVDD VDD_vsrc 0 1.0\nRs VDD_vsrc 1000_1000_M1 0.001"

        netlist_dir = self._create_minimal_netlist(tile_content, nd_content, package_content)

        parser = NetlistParser(netlist_dir, validate=True)
        graph = parser.parse()

        self.assertEqual(parser.builder.stats.floating_nodes, 0)

    def test_validation_uses_tracked_vsrc_indices(self):
        """Test that validation uses tracked vsrc_edge_indices correctly"""
        tile_content = """
R1 1000_1000_M1 2000_1000_M1 0.01
"""
        nd_content = """1000 1000 M1 1000_1000_M1 VDD
2000 1000 M1 2000_1000_M1 VDD
"""
        package_content = "VVDD VDD_vsrc 0 1.0\nRs VDD_vsrc 1000_1000_M1 0.001"

        netlist_dir = self._create_minimal_netlist(tile_content, nd_content, package_content)

        parser = NetlistParser(netlist_dir, validate=True)
        graph = parser.parse()

        # Verify vsrc_edge_indices were tracked
        self.assertEqual(len(parser.builder.vsrc_edge_indices), 1)

        # Verify the tracked index points to a voltage source
        rx_graph = parser.builder.graph._graph
        edge_data = rx_graph.get_edge_data_by_index(parser.builder.vsrc_edge_indices[0])
        self.assertEqual(edge_data['type'], 'V')

    def test_validation_disabled(self):
        """Test that validation is skipped when validate=False"""
        # Create netlist with intentional issues
        tile_content = """
R1 1000_1000_M1 2000_1000_M1 0.0000001
R2 5000_5000_M1 6000_5000_M1 0.01
"""
        nd_content = """1000 1000 M1 1000_1000_M1 VDD
2000 1000 M1 2000_1000_M1 VDD
5000 5000 M1 5000_5000_M1 VDD
6000 5000 M1 6000_5000_M1 VDD
"""
        package_content = "VVDD VDD_vsrc 0 1.0\nRs VDD_vsrc 1000_1000_M1 0.001"

        netlist_dir = self._create_minimal_netlist(tile_content, nd_content, package_content)

        parser = NetlistParser(netlist_dir, validate=False)
        graph = parser.parse()

        # With validate=False, these should remain at 0 (not computed)
        self.assertEqual(parser.builder.stats.shorted_elements, 0)
        self.assertEqual(parser.builder.stats.floating_nodes, 0)

    def test_validation_with_multiple_voltage_sources(self):
        """Test floating node detection with multiple voltage sources"""
        tile_content = """
R1 1000_1000_M1 2000_1000_M1 0.01
R2 3000_3000_M1 4000_3000_M1 0.01
R3 5000_5000_M1 6000_5000_M1 0.01
"""
        nd_content = """1000 1000 M1 1000_1000_M1 VDD
2000 1000 M1 2000_1000_M1 VDD
3000 3000 M1 3000_3000_M1 VDD
4000 3000 M1 4000_3000_M1 VDD
5000 5000 M1 5000_5000_M1 VDD
6000 5000 M1 6000_5000_M1 VDD
"""
        # Two voltage sources connecting to first two components
        package_content = """VVDD1 VDD_vsrc1 0 1.0
Rs1 VDD_vsrc1 1000_1000_M1 0.001
VVDD2 VDD_vsrc2 0 1.0
Rs2 VDD_vsrc2 3000_3000_M1 0.001
"""

        netlist_dir = self._create_minimal_netlist(tile_content, nd_content, package_content)

        parser = NetlistParser(netlist_dir, validate=True)
        graph = parser.parse()

        # Third component (5000_5000_M1, 6000_5000_M1) should be floating
        self.assertEqual(parser.builder.stats.floating_nodes, 2)

        # Should track 2 voltage sources
        self.assertEqual(len(parser.builder.vsrc_edge_indices), 2)


class TestValidationOptimization(unittest.TestCase):
    """Test that optimized validation produces same results as reference implementation"""

    @classmethod
    def setUpClass(cls):
        """Parse netlist_small for comparison tests"""
        test_netlist_dir = Path(__file__).parent.parent / 'pdn' / 'netlist_small'
        if not test_netlist_dir.exists():
            raise unittest.SkipTest("netlist_small not found")

        cls.parser = NetlistParser(str(test_netlist_dir), validate=False)
        cls.graph = cls.parser.parse()

    def test_optimized_shorts_matches_reference(self):
        """Test optimized _check_shorts produces same results as reference"""
        from core.rx_algorithms import node_connected_component

        # Reference implementation (iterate all edges via wrapper)
        def reference_check_shorts(graph):
            shorts = []
            for u, v, key, data in graph.edges(keys=True, data=True):
                if data.get('type') == 'R':
                    value = data.get('value', float('inf'))
                    if value < SHORT_THRESHOLD:
                        shorts.append((u, v, data.get('elem_name'), value))
            return shorts

        # Optimized implementation (direct rustworkx access)
        def optimized_check_shorts(graph):
            rx_graph = graph._graph
            idx_to_node = graph._idx_to_node
            shorts = []
            for edge_idx in rx_graph.edge_indices():
                data = rx_graph.get_edge_data_by_index(edge_idx)
                if data and data.get('type') == 'R':
                    value = data.get('value', float('inf'))
                    if value < SHORT_THRESHOLD:
                        u_idx, v_idx = rx_graph.get_edge_endpoints_by_index(edge_idx)
                        shorts.append((idx_to_node[u_idx], idx_to_node[v_idx],
                                      data.get('elem_name'), value))
            return shorts

        ref_shorts = reference_check_shorts(self.graph)
        opt_shorts = optimized_check_shorts(self.graph)

        # Compare as sets of (sorted endpoints, name, value)
        ref_set = set((tuple(sorted([u, v])), n, v) for u, v, n, v in ref_shorts)
        opt_set = set((tuple(sorted([u, v])), n, v) for u, v, n, v in opt_shorts)

        self.assertEqual(ref_set, opt_set)

    def test_optimized_floating_matches_reference(self):
        """Test optimized _check_floating_nodes produces same results as reference"""
        from core.rx_algorithms import node_connected_component

        # Reference implementation (iterate all edges, use to_undirected)
        def reference_check_floating(graph):
            grounded_nodes = set()
            for u, v, data in graph.edges(data=True):
                if data.get('type') == 'V':
                    grounded_nodes.add(u)
                    grounded_nodes.add(v)

            connected_nodes = set()
            undirected_graph = graph.to_undirected()
            for node in grounded_nodes:
                if node not in connected_nodes:
                    component = node_connected_component(undirected_graph, node)
                    connected_nodes.update(component)

            all_nodes = set(graph.nodes())
            return all_nodes - connected_nodes

        # Optimized implementation (tracked indices, direct weak connectivity)
        def optimized_check_floating(graph, vsrc_edge_indices):
            rx_graph = graph._graph
            idx_to_node = graph._idx_to_node

            grounded_nodes = set()
            for edge_idx in vsrc_edge_indices:
                try:
                    u_idx, v_idx = rx_graph.get_edge_endpoints_by_index(edge_idx)
                    grounded_nodes.add(idx_to_node[u_idx])
                    grounded_nodes.add(idx_to_node[v_idx])
                except Exception:
                    pass

            connected_nodes = set()
            for node in grounded_nodes:
                if node not in connected_nodes:
                    component = node_connected_component(graph, node)
                    connected_nodes.update(component)

            all_nodes = set(graph.nodes())
            return all_nodes - connected_nodes

        ref_floating = reference_check_floating(self.graph)
        opt_floating = optimized_check_floating(self.graph, self.parser.builder.vsrc_edge_indices)

        self.assertEqual(ref_floating, opt_floating)


# =============================================================================
# Instance Current Source Waveform Parsing Tests
# =============================================================================

class TestInstanceInfo(unittest.TestCase):
    """Test InstanceInfo dataclass parsing and serialization"""
    
    def test_parse_full_format(self):
        """Test parsing full instance name format"""
        # Format: i_<instance_name>:<vdd_net>:<vdd_pin>:<vss_net>:<vss_pin>:<tile_x>:<tile_y>
        info = InstanceInfo.parse("i_U123/cell:VDD_XLV:VDD:VSS:0:5:3")
        
        self.assertEqual(info.full_name, "i_U123/cell:VDD_XLV:VDD:VSS:0:5:3")
        self.assertEqual(info.instance_name, "U123/cell")
        self.assertEqual(info.vdd_net, "VDD_XLV")
        self.assertEqual(info.vdd_pin, "VDD")
        self.assertEqual(info.vss_net, "VSS")
        self.assertIsNone(info.vss_pin)  # '0' is converted to None
        self.assertEqual(info.tile_x, 5)
        self.assertEqual(info.tile_y, 3)
    
    def test_parse_minimal_format(self):
        """Test parsing minimal instance name"""
        info = InstanceInfo.parse("i_simple_inst")
        
        self.assertEqual(info.full_name, "i_simple_inst")
        self.assertEqual(info.instance_name, "simple_inst")
        self.assertIsNone(info.vdd_net)
        self.assertIsNone(info.vdd_pin)
    
    def test_parse_without_i_prefix(self):
        """Test parsing name without i_ prefix"""
        info = InstanceInfo.parse("plain_name:VDD:pin")
        
        self.assertEqual(info.full_name, "plain_name:VDD:pin")
        self.assertEqual(info.instance_name, "plain_name")
        self.assertEqual(info.vdd_net, "VDD")
    
    def test_serialization_roundtrip(self):
        """Test to_dict/from_dict roundtrip"""
        # Note: instance_name is now a computed property derived from full_name
        original = InstanceInfo(
            full_name="i_test:VDD:vdd:VSS::2:4",
            vdd_net="VDD",
            vdd_pin="vdd",
            vss_net="VSS",
            vss_pin=None,
            tile_x=2,
            tile_y=4
        )

        # Verify computed property works
        self.assertEqual(original.instance_name, "test")

        d = original.to_dict()
        restored = InstanceInfo.from_dict(d)

        self.assertEqual(original.full_name, restored.full_name)
        self.assertEqual(original.instance_name, restored.instance_name)  # Computed property
        self.assertEqual(original.vdd_net, restored.vdd_net)
        self.assertEqual(original.tile_x, restored.tile_x)
        self.assertEqual(original.tile_y, restored.tile_y)


class TestPulseWaveform(unittest.TestCase):
    """Test Pulse waveform dataclass"""
    
    def test_evaluate_before_delay(self):
        """Test pulse evaluation before delay time"""
        pulse = Pulse(v1=0.0, v2=1.0, delay=1e-9, rt=0.1e-9, ft=0.1e-9, 
                     width=0.5e-9, period=2e-9)
        
        # Before delay, should return v1
        self.assertEqual(pulse.evaluate(0.5e-9), 0.0)
    
    def test_evaluate_at_high(self):
        """Test pulse evaluation during high period"""
        pulse = Pulse(v1=0.0, v2=1.0, delay=0.0, rt=0.1e-9, ft=0.1e-9, 
                     width=0.5e-9, period=2e-9)
        
        # During high (after rise, before fall)
        self.assertEqual(pulse.evaluate(0.3e-9), 1.0)
    
    def test_evaluate_during_rise(self):
        """Test pulse evaluation during rise time.

        With delay=start time interpretation (standard SPICE):
        - delay=1.0, rt=1.0 means rise phase is [1.0, 2.0)
        - At t=1.5, we're halfway through rise
        """
        pulse = Pulse(v1=0.0, v2=1.0, delay=1.0, rt=1.0, ft=1.0,
                     width=2.0, period=10.0)

        # Halfway through rise (t=1.5 is in [delay, delay+rt) = [1, 2))
        value = pulse.evaluate(1.5)
        self.assertAlmostEqual(value, 0.5, places=5)

    def test_evaluate_during_fall(self):
        """Test pulse evaluation during fall time.

        With delay=start time interpretation (standard SPICE):
        - delay=1.0, rt=1.0, width=2.0, ft=1.0
        - Rise: [1, 2), High: [2, 4), Fall: [4, 5)
        - At t=4.5, we're halfway through fall
        """
        pulse = Pulse(v1=0.0, v2=1.0, delay=1.0, rt=1.0, ft=1.0,
                     width=2.0, period=10.0)

        # Halfway through fall (t=4.5 is in [delay+rt+width, delay+rt+width+ft) = [4, 5))
        value = pulse.evaluate(4.5)
        self.assertAlmostEqual(value, 0.5, places=5)
    
    def test_get_dc(self):
        """Test DC average calculation"""
        # 50% duty cycle square wave
        pulse = Pulse(v1=0.0, v2=1.0, delay=0.0, rt=0.0, ft=0.0, 
                     width=5.0, period=10.0)
        
        dc = pulse.get_dc()
        self.assertAlmostEqual(dc, 0.5, places=5)
    
    def test_get_dc_non_periodic(self):
        """Test DC average returns 0 for non-periodic pulse"""
        pulse = Pulse(v1=0.0, v2=1.0, delay=0.0, period=0.0)
        self.assertEqual(pulse.get_dc(), 0.0)
    
    def test_serialization_roundtrip(self):
        """Test to_dict/from_dict roundtrip"""
        original = Pulse(v1=0.1, v2=2.0, delay=1e-9, rt=0.5e-9, 
                        ft=0.6e-9, width=3e-9, period=10e-9)
        
        d = original.to_dict()
        restored = Pulse.from_dict(d)
        
        self.assertEqual(original.v1, restored.v1)
        self.assertEqual(original.v2, restored.v2)
        self.assertEqual(original.delay, restored.delay)
        self.assertEqual(original.rt, restored.rt)
        self.assertEqual(original.period, restored.period)

    def test_evaluate_periodic_basic_waveform(self):
        """Test basic periodic pulse evaluation through all phases.

        With delay=start time interpretation (standard SPICE):
        - delay=1ns (start), rt=1ns, width=2ns, ft=1ns, period=10ns
        - Rise: [1, 2ns), High: [2, 4ns), Fall: [4, 5ns), Low: [5, 11ns)
        """
        pulse = Pulse(v1=0.0, v2=1.0, delay=1e-9, rt=1e-9, ft=1e-9,
                      width=2e-9, period=10e-9)

        # Phase timeline with delay=1ns as start time
        self.assertAlmostEqual(pulse.evaluate(1.5e-9), 0.5, places=5)  # Rise [1, 2)
        self.assertEqual(pulse.evaluate(3e-9), 1.0)                     # High [2, 4)
        self.assertAlmostEqual(pulse.evaluate(4.5e-9), 0.5, places=5)  # Fall [4, 5)
        self.assertEqual(pulse.evaluate(6e-9), 0.0)                     # Low [5, 11)

        # Verify periodicity
        self.assertAlmostEqual(pulse.evaluate(11.5e-9), 0.5, places=5)  # Rise in 2nd period

    def test_evaluate_periodic_with_delay(self):
        """Test periodic pulse with delay (start time) offset.

        With delay=start time interpretation (standard SPICE):
        - delay=5ns (start), rt=1ns, width=2ns, ft=1ns, period=20ns
        - Rise: [5, 6ns), High: [6, 8ns), Fall: [8, 9ns), Low: [9, 25ns)
        """
        pulse = Pulse(v1=0.0, v2=1.0, delay=5e-9, rt=1e-9, ft=1e-9,
                      width=2e-9, period=20e-9)

        # Before pulse (0-5ns): should be v1
        self.assertEqual(pulse.evaluate(3e-9), 0.0)

        # During high (6-8ns): should be v2
        self.assertEqual(pulse.evaluate(7e-9), 1.0)

        # After pulse (9-20ns): should be v1
        self.assertEqual(pulse.evaluate(12e-9), 0.0)

        # Next period: t=23ns equivalent to t=3ns (before pulse)
        self.assertEqual(pulse.evaluate(23e-9), 0.0)
        # t=27ns equivalent to t=7ns (during high)
        self.assertEqual(pulse.evaluate(27e-9), 1.0)

    def test_evaluate_periodic_delay_exceeds_period(self):
        """Test edge case where delay >= period (should normalize correctly)."""
        # delay=15ns, period=10ns -> effective delay = 5ns
        pulse = Pulse(v1=0.0, v2=1.0, delay=15e-9, rt=1e-9, ft=1e-9,
                      width=2e-9, period=10e-9)

        # At t=3ns: before effective delay (5ns), should be v1
        self.assertEqual(pulse.evaluate(3e-9), 0.0)

        # At t=6ns: during high phase, should be v2
        self.assertEqual(pulse.evaluate(6e-9), 1.0)

    def test_evaluate_periodic_pulse_wraps_around_period(self):
        """Test pulse that extends beyond period boundary.

        With delay=start time interpretation (standard SPICE):
        - delay=8ns (start), rt=1ns, width=2ns, ft=1ns, period=10ns
        - Rise: [8, 9ns), High: [9, 11ns) wraps to [9,10)+[0,1), Fall: [1, 2ns), Low: [2, 8ns)
        """
        pulse = Pulse(v1=0.0, v2=1.0, delay=8e-9, rt=1e-9, ft=1e-9,
                      width=2e-9, period=10e-9)

        # At t=8.5ns (rise phase): should be rising
        self.assertAlmostEqual(pulse.evaluate(8.5e-9), 0.5, places=5)

        # At t=0ns (in high phase after wrap): should be v2
        self.assertAlmostEqual(pulse.evaluate(0e-9), 1.0, places=5)

        # At t=5ns (in low region): should be v1
        self.assertAlmostEqual(pulse.evaluate(5e-9), 0.0, places=5)

    def test_evaluate_non_periodic_before_delay(self):
        """Test non-periodic pulse returns v1 before pulse start.

        With delay=start time interpretation (standard SPICE):
        - delay=5ns (start), rt=1ns means pulse starts at 5ns
        - Rise: [5, 6ns), High: [6, 8ns), Fall: [8, 9ns)
        """
        pulse = Pulse(v1=0.0, v2=1.0, delay=5e-9, rt=1e-9, ft=1e-9,
                      width=2e-9, period=0.0)  # Non-periodic

        self.assertEqual(pulse.evaluate(4e-9), 0.0)  # Before pulse start (5ns)
        self.assertEqual(pulse.evaluate(7e-9), 1.0)  # During high [6, 8ns)

    def test_evaluate_matches_cpp_simpwl_conversion(self):
        """Verify Python evaluate matches expected pulse behavior.

        With delay=start time interpretation (standard SPICE):
        - delay=2ns (start), rt=1ns, width=3ns, ft=1ns, period=10ns
        - Rise: [2, 3ns), High: [3, 6ns), Fall: [6, 7ns), Low: [7, 12ns) wraps to [7,10)+[0,2)
        """
        pulse = Pulse(v1=0.0, v2=10.0, delay=2e-9, rt=1e-9, ft=1e-9,
                      width=3e-9, period=10e-9)

        # Timeline: Rise[2,3), High[3,6), Fall[6,7), Low[7,10)+[0,2)
        test_cases = [
            (0e-9, 0.0),    # Before pulse (in low region, wrapped)
            (2e-9, 0.0),    # At pulse start (start of rise)
            (2.5e-9, 5.0),  # Mid-rise
            (4e-9, 10.0),   # High phase
            (6.5e-9, 5.0),  # Mid-fall
            (8e-9, 0.0),    # After pulse (low region)
        ]

        for t, expected in test_cases:
            actual = pulse.evaluate(t)
            self.assertAlmostEqual(actual, expected, places=3,
                                   msg=f"At t={t*1e9:.1f}ns: expected {expected}, got {actual}")


class TestPWLWaveform(unittest.TestCase):
    """Test PWL (piece-wise linear) waveform dataclass"""
    
    def test_evaluate_before_first_point(self):
        """Test PWL evaluation before first time point"""
        pwl = PWL(points=[(1.0, 0.5), (2.0, 1.0), (3.0, 0.0)])
        
        # Before first point, return first value
        self.assertEqual(pwl.evaluate(0.5), 0.5)
    
    def test_evaluate_after_last_point(self):
        """Test PWL evaluation after last time point (non-periodic)"""
        pwl = PWL(points=[(1.0, 0.5), (2.0, 1.0), (3.0, 0.0)], period=0.0)
        
        # After last point, return last value
        self.assertEqual(pwl.evaluate(5.0), 0.0)
    
    def test_evaluate_at_point(self):
        """Test PWL evaluation at exact time point"""
        pwl = PWL(points=[(1.0, 0.5), (2.0, 1.0), (3.0, 0.0)])
        
        self.assertEqual(pwl.evaluate(2.0), 1.0)
    
    def test_evaluate_interpolation(self):
        """Test PWL linear interpolation between points"""
        pwl = PWL(points=[(0.0, 0.0), (2.0, 2.0)])
        
        # Midpoint should interpolate to 1.0
        self.assertAlmostEqual(pwl.evaluate(1.0), 1.0, places=5)
    
    def test_evaluate_with_delay(self):
        """Test PWL evaluation with delay offset"""
        pwl = PWL(points=[(0.0, 0.0), (1.0, 1.0)], delay=5.0)
        
        # At t=5, effective time is 0, so value is 0
        self.assertEqual(pwl.evaluate(5.0), 0.0)
        # At t=5.5, effective time is 0.5, interpolate to 0.5
        self.assertAlmostEqual(pwl.evaluate(5.5), 0.5, places=5)
    
    def test_evaluate_periodic(self):
        """Test PWL evaluation with periodic wrapping"""
        pwl = PWL(points=[(0.0, 0.0), (1.0, 1.0)], period=2.0)
        
        # At t=3.5, wraps to t=1.5, which is after last point
        # With period, returns first value (0.0)
        value = pwl.evaluate(3.5)
        self.assertEqual(value, 0.0)
    
    def test_get_dc(self):
        """Test DC average calculation for triangular wave"""
        # Triangle from 0 to 2 back to 0
        pwl = PWL(points=[(0.0, 0.0), (1.0, 2.0), (2.0, 0.0)], period=2.0)
        
        dc = pwl.get_dc()
        # Average of triangle is 1.0
        self.assertAlmostEqual(dc, 1.0, places=5)
    
    def test_get_dc_non_periodic(self):
        """Test DC average returns 0 for non-periodic PWL"""
        pwl = PWL(points=[(0.0, 0.0), (1.0, 1.0)], period=0.0)
        self.assertEqual(pwl.get_dc(), 0.0)
    
    def test_empty_pwl(self):
        """Test empty PWL returns 0"""
        pwl = PWL(points=[])
        self.assertEqual(pwl.evaluate(1.0), 0.0)
        self.assertEqual(pwl.get_dc(), 0.0)
    
    def test_serialization_roundtrip(self):
        """Test to_dict/from_dict roundtrip"""
        original = PWL(points=[(0.0, 0.1), (1.0, 0.5), (2.0, 0.2)], 
                      period=3.0, delay=0.5)
        
        d = original.to_dict()
        restored = PWL.from_dict(d)
        
        self.assertEqual(original.points, restored.points)
        self.assertEqual(original.period, restored.period)
        self.assertEqual(original.delay, restored.delay)


class TestCurrentSource(unittest.TestCase):
    """Test CurrentSource dataclass"""
    
    def test_has_waveform_data_with_pulse(self):
        """Test waveform detection with pulse"""
        cs = CurrentSource(name="I1", node1="n1", node2="n2")
        self.assertFalse(cs.has_waveform_data())
        
        cs.pulses.append(Pulse())
        self.assertTrue(cs.has_waveform_data())
    
    def test_has_waveform_data_with_pwl(self):
        """Test waveform detection with PWL"""
        cs = CurrentSource(name="I1", node1="n1", node2="n2")
        cs.pwls.append(PWL(points=[(0, 0), (1, 1)]))
        self.assertTrue(cs.has_waveform_data())
    
    def test_has_current_data(self):
        """Test current data detection"""
        cs = CurrentSource(name="I1", node1="n1", node2="n2")
        self.assertFalse(cs.has_current_data())
        
        cs.dc_value = 1.0
        self.assertTrue(cs.has_current_data())
    
    def test_get_static_current_dc_only(self):
        """Test static current with DC value only"""
        cs = CurrentSource(name="I1", node1="n1", node2="n2", dc_value=5.0)
        self.assertEqual(cs.get_static_current(), 5.0)
    
    def test_get_static_current_with_static_value(self):
        """Test static current with static_value override"""
        cs = CurrentSource(name="I1", node1="n1", node2="n2", 
                          dc_value=2.0, static_value=3.0)
        # dc_value + static_value
        self.assertEqual(cs.get_static_current(), 5.0)
    
    def test_get_static_current_with_pulse(self):
        """Test static current includes pulse DC average"""
        cs = CurrentSource(name="I1", node1="n1", node2="n2", dc_value=1.0)
        # 50% duty cycle pulse with amplitude 2.0 -> DC = 1.0
        cs.pulses.append(Pulse(v1=0.0, v2=2.0, width=5.0, period=10.0))
        
        # dc_value + pulse_dc = 1.0 + 1.0 = 2.0
        self.assertAlmostEqual(cs.get_static_current(), 2.0, places=5)
    
    def test_get_current_at_time(self):
        """Test time-domain current evaluation"""
        cs = CurrentSource(name="I1", node1="n1", node2="n2", dc_value=1.0)
        cs.pulses.append(Pulse(v1=0.0, v2=2.0, delay=0.0, rt=0.0, ft=0.0, 
                              width=5.0, period=10.0))
        
        # During high: dc + pulse_high = 1.0 + 2.0 = 3.0
        self.assertEqual(cs.get_current_at_time(2.5), 3.0)
        
        # During low: dc + pulse_low = 1.0 + 0.0 = 1.0
        self.assertEqual(cs.get_current_at_time(7.5), 1.0)
    
    def test_serialization_roundtrip(self):
        """Test to_dict/from_dict roundtrip"""
        original = CurrentSource(
            name="I_test",
            node1="vdd_node",
            node2="0",
            dc_value=1.5,
            static_value=2.0,
            pulses=[Pulse(v1=0, v2=1, period=10)],
            pwls=[PWL(points=[(0, 0), (1, 1)])],
            info=InstanceInfo(full_name="i_test")  # instance_name is computed from full_name
        )
        
        d = original.to_dict()
        restored = CurrentSource.from_dict(d)
        
        self.assertEqual(original.name, restored.name)
        self.assertEqual(original.node1, restored.node1)
        self.assertEqual(original.node2, restored.node2)
        self.assertEqual(original.dc_value, restored.dc_value)
        self.assertEqual(original.static_value, restored.static_value)
        self.assertEqual(len(original.pulses), len(restored.pulses))
        self.assertEqual(len(original.pwls), len(restored.pwls))
        self.assertEqual(original.info.full_name, restored.info.full_name)

    def test_wscale_default(self):
        """Test default wscale value is 1.0"""
        cs = CurrentSource(name="I1", node1="n1", node2="n2", dc_value=1.0)
        self.assertEqual(cs.wscale, 1.0)

    def test_wscale_affects_get_current_at_time(self):
        """Test wscale scales pulse/PWL values in get_current_at_time"""
        cs = CurrentSource(name="I1", node1="n1", node2="n2", dc_value=1.0, wscale=0.5)
        cs.pulses.append(Pulse(v1=0.0, v2=2.0, delay=0.0, rt=0.0, ft=0.0,
                               width=5.0, period=10.0))

        # During high: dc + pulse_high * wscale = 1.0 + 2.0 * 0.5 = 2.0
        self.assertAlmostEqual(cs.get_current_at_time(2.5), 2.0, places=5)

        # During low: dc + pulse_low * wscale = 1.0 + 0.0 * 0.5 = 1.0
        self.assertAlmostEqual(cs.get_current_at_time(7.5), 1.0, places=5)

    def test_wscale_not_applied_to_dc(self):
        """Test wscale does NOT affect DC value (matches C++ behavior)"""
        cs = CurrentSource(name="I1", node1="n1", node2="n2", dc_value=2.0, wscale=0.5)
        # No waveforms, just DC - should be unscaled
        self.assertAlmostEqual(cs.get_current_at_time(0.0), 2.0, places=5)

    def test_wscale_affects_get_static_current(self):
        """Test wscale scales pulse/PWL DC averages in get_static_current"""
        cs = CurrentSource(name="I1", node1="n1", node2="n2", dc_value=1.0, wscale=0.5)
        # 50% duty cycle pulse with amplitude 2.0 -> DC avg = 1.0
        cs.pulses.append(Pulse(v1=0.0, v2=2.0, width=5.0, period=10.0))

        # dc_value + pulse_dc * wscale = 1.0 + 1.0 * 0.5 = 1.5
        self.assertAlmostEqual(cs.get_static_current(), 1.5, places=5)

    def test_wscale_static_value_override(self):
        """Test static_value override ignores wscale (as expected)"""
        cs = CurrentSource(name="I1", node1="n1", node2="n2",
                          dc_value=1.0, static_value=2.0, wscale=0.5)
        cs.pulses.append(Pulse(v1=0.0, v2=4.0, width=5.0, period=10.0))

        # With static_value set, wscale is ignored: dc + static = 1.0 + 2.0 = 3.0
        self.assertAlmostEqual(cs.get_static_current(), 3.0, places=5)

    def test_wscale_serialization_roundtrip(self):
        """Test wscale is preserved through to_dict/from_dict"""
        original = CurrentSource(
            name="I_test",
            node1="vdd_node",
            node2="0",
            dc_value=1.5,
            wscale=0.25
        )

        d = original.to_dict()
        self.assertEqual(d['wscale'], 0.25)

        restored = CurrentSource.from_dict(d)
        self.assertEqual(restored.wscale, 0.25)

    def test_wscale_default_in_from_dict(self):
        """Test from_dict defaults wscale to 1.0 for backward compat"""
        d = {'name': 'I1', 'node1': 'n1', 'node2': 'n2', 'dc_value': 1.0}
        # No wscale key in dict
        restored = CurrentSource.from_dict(d)
        self.assertEqual(restored.wscale, 1.0)

    def test_global_apply_wscale_flag(self):
        """Test set_apply_wscale() controls wscale application (thread-safe via ContextVar)"""
        cs = CurrentSource(name="I1", node1="n1", node2="n2", dc_value=1.0, wscale=0.5)
        cs.pulses.append(Pulse(v1=0.0, v2=4.0, delay=0.0, rt=0.0, ft=0.0,
                               width=5.0, period=10.0))

        # Save original state
        original_state = get_apply_wscale()

        try:
            # With apply_wscale=True (default)
            set_apply_wscale(True)
            # dc + pulse * wscale = 1.0 + 4.0 * 0.5 = 3.0
            self.assertAlmostEqual(cs.get_current_at_time(2.5), 3.0, places=5)

            # With apply_wscale=False
            set_apply_wscale(False)
            # dc + pulse (no wscale) = 1.0 + 4.0 = 5.0
            self.assertAlmostEqual(cs.get_current_at_time(2.5), 5.0, places=5)
        finally:
            # Restore original state
            set_apply_wscale(original_state)


class TestSpiceValueParsing(unittest.TestCase):
    """Test _parse_spice_value function"""
    
    def test_basic_numbers(self):
        """Test basic numeric parsing"""
        self.assertEqual(_parse_spice_value("100"), 100.0)
        self.assertEqual(_parse_spice_value("1.5"), 1.5)
        self.assertEqual(_parse_spice_value("-2.5"), -2.5)
        self.assertEqual(_parse_spice_value("+3.0"), 3.0)
    
    def test_scientific_notation(self):
        """Test scientific notation parsing"""
        self.assertEqual(_parse_spice_value("1e-3"), 1e-3)
        self.assertEqual(_parse_spice_value("2.5E+6"), 2.5e6)
        self.assertEqual(_parse_spice_value("1e9"), 1e9)
    
    def test_unit_suffixes(self):
        """Test SPICE unit suffix parsing"""
        self.assertEqual(_parse_spice_value("1f"), 1e-15)
        self.assertEqual(_parse_spice_value("1p"), 1e-12)
        self.assertEqual(_parse_spice_value("1n"), 1e-9)
        self.assertEqual(_parse_spice_value("1u"), 1e-6)
        self.assertEqual(_parse_spice_value("1m"), 1e-3)
        self.assertEqual(_parse_spice_value("1k"), 1e3)
        self.assertEqual(_parse_spice_value("1meg"), 1e6)
        self.assertEqual(_parse_spice_value("1g"), 1e9)
        self.assertEqual(_parse_spice_value("1t"), 1e12)
    
    def test_combined_values(self):
        """Test combined number and suffix"""
        self.assertAlmostEqual(_parse_spice_value("2.5m"), 2.5e-3, places=10)
        self.assertAlmostEqual(_parse_spice_value("100n"), 100e-9, places=15)
        self.assertAlmostEqual(_parse_spice_value("4.7k"), 4700.0, places=5)


class TestPulseParsing(unittest.TestCase):
    """Test _parse_pulse function"""
    
    def test_parse_full_pulse(self):
        """Test parsing complete pulse definition"""
        line = "pulse(0 1m 10n 1n 2n 5n 20n)"
        pulse = _parse_pulse(line)
        
        self.assertEqual(pulse.v1, 0.0)
        self.assertAlmostEqual(pulse.v2, 1e-3, places=10)
        self.assertAlmostEqual(pulse.delay, 10e-9, places=15)
        self.assertAlmostEqual(pulse.rt, 1e-9, places=15)
        self.assertAlmostEqual(pulse.ft, 2e-9, places=15)
        self.assertAlmostEqual(pulse.width, 5e-9, places=15)
        self.assertAlmostEqual(pulse.period, 20e-9, places=15)
    
    def test_parse_partial_pulse(self):
        """Test parsing pulse with fewer parameters"""
        line = "pulse(0 1)"
        pulse = _parse_pulse(line)
        
        self.assertEqual(pulse.v1, 0.0)
        self.assertEqual(pulse.v2, 1.0)
        self.assertEqual(pulse.delay, 0.0)
        self.assertEqual(pulse.period, 0.0)
    
    def test_parse_pulse_with_commas(self):
        """Test parsing pulse with comma separators"""
        line = "PULSE(0, 1e-3, 0, 1n, 1n, 5n, 10n)"
        pulse = _parse_pulse(line)
        
        self.assertEqual(pulse.v1, 0.0)
        self.assertAlmostEqual(pulse.v2, 1e-3, places=10)
    
    def test_invalid_pulse_returns_empty(self):
        """Test that invalid pulse string returns empty Pulse"""
        pulse = _parse_pulse("not a pulse")
        self.assertEqual(pulse.v1, 0.0)
        self.assertEqual(pulse.v2, 0.0)


class TestPWLParsing(unittest.TestCase):
    """Test _parse_pwl function"""
    
    def test_parse_pwl_basic(self):
        """Test parsing basic PWL definition"""
        line = "pwl(0 0 1n 1m 2n 0)"
        pwl = _parse_pwl(line)
        
        self.assertEqual(len(pwl.points), 3)
        self.assertEqual(pwl.points[0], (0.0, 0.0))
        self.assertAlmostEqual(pwl.points[1][0], 1e-9, places=15)
        self.assertAlmostEqual(pwl.points[1][1], 1e-3, places=10)
    
    def test_parse_pwl_with_commas(self):
        """Test parsing PWL with comma separators"""
        line = "PWL(0, 0, 1e-9, 0.001, 2e-9, 0)"
        pwl = _parse_pwl(line)
        
        self.assertEqual(len(pwl.points), 3)
    
    def test_parse_pwl_sorted(self):
        """Test that PWL points are sorted by time"""
        line = "pwl(2 2 0 0 1 1)"
        pwl = _parse_pwl(line)
        
        # Points should be sorted by time
        self.assertEqual(pwl.points[0][0], 0.0)
        self.assertEqual(pwl.points[1][0], 1.0)
        self.assertEqual(pwl.points[2][0], 2.0)
    
    def test_invalid_pwl_returns_empty(self):
        """Test that invalid PWL string returns empty PWL"""
        pwl = _parse_pwl("not a pwl")
        self.assertEqual(len(pwl.points), 0)


class TestCurrentSourceLineParsing(unittest.TestCase):
    """Test _parse_current_source_line function

    These tests verify full CurrentSource interface, so DC-only optimization
    is disabled to ensure the full dataclass is returned.
    """

    def setUp(self):
        """Disable DC-only optimization to test full CurrentSource behavior"""
        self.original_flag = get_optimize_dc_only()
        set_optimize_dc_only(False)

    def tearDown(self):
        """Restore original optimization flag"""
        set_optimize_dc_only(self.original_flag)

    def test_parse_dc_only(self):
        """Test parsing current source with DC value only"""
        line = "I_inst n1 n2 1e-3"
        cs = _parse_current_source_line(line)
        
        self.assertIsNotNone(cs)
        self.assertEqual(cs.name, "I_inst")
        self.assertEqual(cs.node1, "n1")
        self.assertEqual(cs.node2, "n2")
        self.assertAlmostEqual(cs.dc_value, 1e-3, places=10)
    
    def test_parse_dc_with_prefix(self):
        """Test parsing current source with 'dc' prefix"""
        line = "I_inst n1 n2 dc 2m"
        cs = _parse_current_source_line(line)
        
        self.assertIsNotNone(cs)
        self.assertAlmostEqual(cs.dc_value, 2e-3, places=10)
    
    def test_parse_with_static_value(self):
        """Test parsing current source with static_value parameter"""
        line = "I_inst n1 n2 1m static_value=5m"
        cs = _parse_current_source_line(line)
        
        self.assertIsNotNone(cs)
        self.assertAlmostEqual(cs.dc_value, 1e-3, places=10)
        self.assertAlmostEqual(cs.static_value, 5e-3, places=10)
    
    def test_parse_with_pulse(self):
        """Test parsing current source with pulse waveform"""
        line = "I_inst n1 n2 0 pulse(0 1m 0 1n 1n 5n 10n)"
        cs = _parse_current_source_line(line)
        
        self.assertIsNotNone(cs)
        self.assertEqual(len(cs.pulses), 1)
        self.assertAlmostEqual(cs.pulses[0].v2, 1e-3, places=10)
    
    def test_parse_with_pwl(self):
        """Test parsing current source with PWL waveform"""
        line = "I_inst n1 n2 0 pwl(0 0 1n 1m 2n 0)"
        cs = _parse_current_source_line(line)
        
        self.assertIsNotNone(cs)
        self.assertEqual(len(cs.pwls), 1)
        self.assertEqual(len(cs.pwls[0].points), 3)
    
    def test_parse_with_pwl_parameters(self):
        """Test parsing current source with PWL and period/delay"""
        line = "I_inst n1 n2 0 pwl(0 0 1n 1m) pwl_period=10n pwl_delay=5n"
        cs = _parse_current_source_line(line)
        
        self.assertIsNotNone(cs)
        self.assertEqual(len(cs.pwls), 1)
        self.assertAlmostEqual(cs.pwls[0].period, 10e-9, places=15)
        self.assertAlmostEqual(cs.pwls[0].delay, 5e-9, places=15)
    
    def test_parse_complex_line(self):
        """Test parsing complex current source with multiple waveforms"""
        line = "I_core:cluster1:2000_2000:vdd:0 node1 0 dc 1m static_value=6m pulse(0 0.5m 0 1n 1n 5n 20n)"
        cs = _parse_current_source_line(line)
        
        self.assertIsNotNone(cs)
        self.assertEqual(cs.name, "I_core:cluster1:2000_2000:vdd:0")
        self.assertAlmostEqual(cs.dc_value, 1e-3, places=10)
        self.assertAlmostEqual(cs.static_value, 6e-3, places=10)
        self.assertEqual(len(cs.pulses), 1)
        self.assertIsNotNone(cs.info)
    
    def test_parse_invalid_line_returns_none(self):
        """Test that invalid lines return None"""
        self.assertIsNone(_parse_current_source_line(""))
        self.assertIsNone(_parse_current_source_line("R1 n1 n2 100"))
        self.assertIsNone(_parse_current_source_line("I_short n1"))  # Too few tokens
    
    def test_instance_info_extracted(self):
        """Test that InstanceInfo is extracted from name"""
        line = "i_U123/cell:VDD:vdd:VSS:0:5:3 node1 node2 1m"
        cs = _parse_current_source_line(line)

        self.assertIsNotNone(cs)
        self.assertIsNotNone(cs.info)
        self.assertEqual(cs.info.instance_name, "U123/cell")
        self.assertEqual(cs.info.vdd_net, "VDD")

    def test_parse_wscale_default(self):
        """Test default wscale is 1.0 when not specified"""
        line = "I_inst n1 n2 1e-3 pulse(0 1m 0 1n 1n 5n 10n)"
        cs = _parse_current_source_line(line)

        self.assertIsNotNone(cs)
        self.assertEqual(cs.wscale, 1.0)

    def test_parse_wscale_explicit(self):
        """Test parsing wscale parameter"""
        line = "I_inst n1 n2 1e-3 pulse(0 1m 0 1n 1n 5n 10n) wscale=0.5"
        cs = _parse_current_source_line(line)

        self.assertIsNotNone(cs)
        self.assertEqual(cs.wscale, 0.5)

    def test_parse_wscale_with_spice_suffix(self):
        """Test parsing wscale with SPICE unit suffix"""
        line = "I_inst n1 n2 1e-3 pulse(0 1m 0 1n 1n 5n 10n) wscale=250m"
        cs = _parse_current_source_line(line)

        self.assertIsNotNone(cs)
        self.assertAlmostEqual(cs.wscale, 0.25, places=10)

    def test_parse_wscale_with_pwl(self):
        """Test parsing wscale with PWL waveform"""
        line = "I_inst n1 n2 0 pwl(0 0 1n 1m 2n 0) pwl_period=10n wscale=0.25"
        cs = _parse_current_source_line(line)

        self.assertIsNotNone(cs)
        self.assertEqual(cs.wscale, 0.25)
        self.assertEqual(len(cs.pwls), 1)

    def test_parse_wscale_complex_line(self):
        """Test wscale parsing in complex current source line"""
        line = "I_core:cluster1:2000_2000:vdd:0 node1 0 dc 1m static_value=5m pulse(0 2m 0 0.1n 0.1n 4.9n 10n) wscale=1.0"
        cs = _parse_current_source_line(line)

        self.assertIsNotNone(cs)
        self.assertAlmostEqual(cs.dc_value, 1e-3, places=10)
        self.assertAlmostEqual(cs.static_value, 5e-3, places=10)
        self.assertEqual(len(cs.pulses), 1)
        self.assertEqual(cs.wscale, 1.0)

    def test_skip_zero_current_source(self):
        """Test that current sources with zero dc and no waveforms are correctly parsed.

        These are placeholder entries like filler cells with no actual current draw.
        The parser should return a valid CurrentSource object that has_current_data() == False.
        """
        # Real-world example of a filler cell with no actual current
        line = "i_tapfiller_HDBULT11_TAPDS_0:0:0:VSS:VSS:0:0:0 0 517300_12000_24 0 wstart=0 wstop=0 dv=0.66"
        isrc = _parse_current_source_line(line)

        # Parser should return valid CurrentSource object
        self.assertIsNotNone(isrc)
        self.assertEqual(isrc.dc_value, 0.0)
        self.assertIsNone(isrc.static_value)
        self.assertEqual(len(isrc.pulses), 0)
        self.assertEqual(len(isrc.pwls), 0)

        # has_current_data() should return False for zero-current sources
        self.assertFalse(isrc.has_current_data())

    # =========================================================================
    # PWL and Pulse with space before parenthesis (regression tests)
    # =========================================================================

    def test_parse_pwl_with_space_before_paren(self):
        """Test parsing PWL with space between 'pwl' and '(' - regression test.

        Real netlists may have 'pwl (...)' instead of 'pwl(...)'.
        """
        line = "I_inst n1 n2 1.0 pwl (7.50042e-09, 0, 7.50911e-09, 0.000325075, 7.52287e-09, 0)"
        cs = _parse_current_source_line(line)

        self.assertIsNotNone(cs)
        self.assertEqual(len(cs.pwls), 1)
        self.assertEqual(len(cs.pwls[0].points), 3)
        # Verify points are correctly parsed
        self.assertAlmostEqual(cs.pwls[0].points[0][0], 7.50042e-09, places=15)
        self.assertAlmostEqual(cs.pwls[0].points[1][1], 0.000325075, places=10)

    def test_parse_pwl_with_space_and_static_value(self):
        """Test PWL with space when static_value is also present.

        This triggers the full parser path even with DC-only optimization on.
        """
        line = "I_inst n1 n2 1.0 static_value=2.0 pwl (0, 0, 1e-9, 1e-3, 2e-9, 0)"
        cs = _parse_current_source_line(line)

        self.assertIsNotNone(cs)
        self.assertAlmostEqual(cs.static_value, 2.0, places=10)
        self.assertEqual(len(cs.pwls), 1)
        self.assertEqual(len(cs.pwls[0].points), 3)

    def test_parse_pulse_with_space_before_paren(self):
        """Test parsing pulse with space between 'pulse' and '(' - regression test.

        Real netlists may have 'pulse (...)' instead of 'pulse(...)'.
        """
        line = "I_inst n1 n2 0 pulse (0 1e-3 1e-9 0.1e-9 0.1e-9 2e-9 10e-9)"
        cs = _parse_current_source_line(line)

        self.assertIsNotNone(cs)
        self.assertEqual(len(cs.pulses), 1)
        self.assertAlmostEqual(cs.pulses[0].v1, 0.0, places=10)
        self.assertAlmostEqual(cs.pulses[0].v2, 1e-3, places=10)
        self.assertAlmostEqual(cs.pulses[0].delay, 1e-9, places=15)
        self.assertAlmostEqual(cs.pulses[0].period, 10e-9, places=15)

    def test_parse_pulse_with_space_and_static_value(self):
        """Test pulse with space when static_value is also present."""
        line = "I_inst n1 n2 1.0 static_value=3.0 pulse (0 2e-3 0 1n 1n 5n 20n)"
        cs = _parse_current_source_line(line)

        self.assertIsNotNone(cs)
        self.assertAlmostEqual(cs.static_value, 3.0, places=10)
        self.assertEqual(len(cs.pulses), 1)
        self.assertAlmostEqual(cs.pulses[0].v2, 2e-3, places=10)

    # =========================================================================
    # Negative tests for malformed PWL
    # =========================================================================

    def test_parse_empty_pwl_skipped(self):
        """Test that empty pwl() creates no PWL entries."""
        line = "I_inst n1 n2 1.0 pwl()"
        cs = _parse_current_source_line(line)

        self.assertIsNotNone(cs)
        # Empty PWL should be skipped (0 points)
        self.assertEqual(len(cs.pwls), 0)

    def test_parse_pwl_single_point_skipped(self):
        """Test that pwl with single value (incomplete point) is handled."""
        line = "I_inst n1 n2 1.0 pwl(1e-9)"
        cs = _parse_current_source_line(line)

        self.assertIsNotNone(cs)
        # Single value can't form a point (needs time,value pair)
        self.assertEqual(len(cs.pwls), 0)

    def test_parse_pwl_with_space_empty(self):
        """Test that 'pwl ()' with space and empty content is handled."""
        line = "I_inst n1 n2 1.0 pwl ()"
        cs = _parse_current_source_line(line)

        self.assertIsNotNone(cs)
        self.assertEqual(len(cs.pwls), 0)

    def test_parse_pwl_whitespace_only(self):
        """Test that pwl with only whitespace inside is handled."""
        line = "I_inst n1 n2 1.0 pwl(   )"
        cs = _parse_current_source_line(line)

        self.assertIsNotNone(cs)
        self.assertEqual(len(cs.pwls), 0)

    def test_parse_pwl_single_point_pair(self):
        """Test PWL with single time-value pair (valid but minimal)."""
        line = "I_inst n1 n2 1.0 pwl(1e-9, 1e-3)"
        cs = _parse_current_source_line(line)

        self.assertIsNotNone(cs)
        self.assertEqual(len(cs.pwls), 1)
        self.assertEqual(len(cs.pwls[0].points), 1)
        self.assertAlmostEqual(cs.pwls[0].points[0][0], 1e-9, places=15)
        self.assertAlmostEqual(cs.pwls[0].points[0][1], 1e-3, places=10)

    # =========================================================================
    # Negative tests for malformed pulse
    # =========================================================================

    def test_parse_empty_pulse_handled(self):
        """Test that empty pulse() is handled gracefully."""
        line = "I_inst n1 n2 1.0 pulse()"
        cs = _parse_current_source_line(line)

        self.assertIsNotNone(cs)
        # Empty pulse should create pulse with default values
        self.assertEqual(len(cs.pulses), 1)
        # All values should be 0 (defaults from _parse_spice_value failure)
        self.assertEqual(cs.pulses[0].v1, 0.0)
        self.assertEqual(cs.pulses[0].v2, 0.0)

    def test_parse_pulse_with_space_empty(self):
        """Test that 'pulse ()' with space and empty content is handled."""
        line = "I_inst n1 n2 1.0 pulse ()"
        cs = _parse_current_source_line(line)

        self.assertIsNotNone(cs)
        self.assertEqual(len(cs.pulses), 1)

    def test_parse_pulse_partial_params(self):
        """Test pulse with partial parameters (missing some values)."""
        line = "I_inst n1 n2 1.0 pulse(0 1e-3)"
        cs = _parse_current_source_line(line)

        self.assertIsNotNone(cs)
        self.assertEqual(len(cs.pulses), 1)
        self.assertAlmostEqual(cs.pulses[0].v1, 0.0, places=10)
        self.assertAlmostEqual(cs.pulses[0].v2, 1e-3, places=10)
        # Missing params should be 0
        self.assertEqual(cs.pulses[0].delay, 0.0)
        self.assertEqual(cs.pulses[0].rt, 0.0)

    def test_parse_pulse_whitespace_only(self):
        """Test that pulse with only whitespace inside is handled."""
        line = "I_inst n1 n2 1.0 pulse(   )"
        cs = _parse_current_source_line(line)

        self.assertIsNotNone(cs)
        self.assertEqual(len(cs.pulses), 1)


class TestInstanceSourcesIntegration(unittest.TestCase):
    """Integration tests for instance_sources in parsed graph.

    These tests verify the serialization functionality when store_instance_sources=True.
    """

    @classmethod
    def setUpClass(cls):
        """Parse test netlist once for all tests"""
        test_netlist_dir = Path(__file__).parent.parent / 'pdn' / 'netlist_test'
        if not test_netlist_dir.exists():
            raise unittest.SkipTest("Test netlist not found")

        # Use store_instance_sources=True to test serialization (not default)
        cls.parser = NetlistParser(str(test_netlist_dir), validate=True,
                                   store_instance_sources=True)
        cls.graph = cls.parser.parse()
    
    def test_instance_sources_populated(self):
        """Test that instance_sources metadata is populated"""
        inst_sources = self.graph.graph.get('instance_sources', {})
        self.assertGreater(len(inst_sources), 0)
    
    def test_instance_sources_serialized(self):
        """Test that instance_sources are serialized as dicts"""
        inst_sources = self.graph.graph.get('instance_sources', {})
        
        for name, data in inst_sources.items():
            self.assertIsInstance(data, dict)
            self.assertIn('name', data)
            self.assertIn('dc_value', data)
            self.assertIn('node1', data)
            self.assertIn('node2', data)
    
    def test_instance_sources_reconstructable(self):
        """Test that CurrentSource objects can be reconstructed"""
        inst_sources = self.graph.graph.get('instance_sources', {})
        
        for name, data in inst_sources.items():
            cs = CurrentSource.from_dict(data)
            self.assertEqual(cs.name, name)
            self.assertIsNotNone(cs.node1)
            self.assertIsNotNone(cs.node2)
    
    def test_instance_sources_match_isource_count(self):
        """Test that instance_sources count matches I-type edges"""
        inst_sources = self.graph.graph.get('instance_sources', {})
        isource_edges = [(u, v, d) for u, v, d in self.graph.edges(data=True) 
                        if d.get('type') == 'I']
        
        self.assertEqual(len(inst_sources), len(isource_edges))
    
    def test_waveform_statistics_in_stats(self):
        """Test that waveform statistics are in graph stats"""
        stats = self.graph.graph.get('stats', {})
        
        self.assertIn('instances_with_waveforms', stats)
        self.assertIn('total_static_current_ma', stats)
    
    def test_total_static_current_positive(self):
        """Test that total static current is computed"""
        stats = self.graph.graph.get('stats', {})
        total_current = stats.get('total_static_current_ma', 0.0)
        
        # Test netlist should have positive total current
        self.assertGreater(total_current, 0.0)
    
    def test_backward_compat_instance_node_map(self):
        """Test that instance_node_map is still populated for backward compat"""
        inst_node_map = self.graph.graph.get('instance_node_map', {})
        inst_sources = self.graph.graph.get('instance_sources', {})

        # Both should have same keys
        self.assertEqual(set(inst_node_map.keys()), set(inst_sources.keys()))


class TestInstanceSourcesRawObjects(unittest.TestCase):
    """Test the default behavior: store raw CurrentSource objects (not serialized)."""

    @classmethod
    def setUpClass(cls):
        """Parse test netlist once for all tests"""
        test_netlist_dir = Path(__file__).parent.parent / 'pdn' / 'netlist_test'
        if not test_netlist_dir.exists():
            raise unittest.SkipTest("Test netlist not found")

        # Use default store_instance_sources=False for raw object storage
        cls.parser = NetlistParser(str(test_netlist_dir), validate=True)
        cls.graph = cls.parser.parse()

    def test_raw_objects_stored(self):
        """Test that raw CurrentSource objects are stored under _instance_sources_objects"""
        raw_sources = self.graph.graph.get('_instance_sources_objects', {})
        # Should have raw objects stored
        self.assertGreater(len(raw_sources), 0)
        # Should NOT have serialized version
        self.assertEqual(len(self.graph.graph.get('instance_sources', {})), 0)

    def test_raw_objects_are_current_source(self):
        """Test that raw objects are CurrentSource instances"""
        raw_sources = self.graph.graph.get('_instance_sources_objects', {})
        for name, src in raw_sources.items():
            self.assertIsInstance(src, CurrentSource)
            self.assertEqual(src.name, name)

    def test_raw_objects_match_isource_count(self):
        """Test that raw object count matches I-type edges"""
        raw_sources = self.graph.graph.get('_instance_sources_objects', {})
        isource_edges = [(u, v, d) for u, v, d in self.graph.edges(data=True)
                        if d.get('type') == 'I']
        self.assertEqual(len(raw_sources), len(isource_edges))

    def test_backward_compat_instance_node_map_with_raw(self):
        """Test that instance_node_map matches raw objects"""
        inst_node_map = self.graph.graph.get('instance_node_map', {})
        raw_sources = self.graph.graph.get('_instance_sources_objects', {})
        # Both should have same keys
        self.assertEqual(set(inst_node_map.keys()), set(raw_sources.keys()))


# =============================================================================
# DC-Only Current Source Optimization Tests
# =============================================================================

class TestDCOnlyOptimization(unittest.TestCase):
    """Test the lightweight _DCOnlyCurrentSource optimization"""

    def setUp(self):
        """Save original flag state"""
        self.original_flag = get_optimize_dc_only()

    def tearDown(self):
        """Restore original flag state"""
        set_optimize_dc_only(self.original_flag)

    def test_flag_default_is_true(self):
        """Test that DC-only optimization is enabled by default"""
        # Reset to default by creating fresh context
        set_optimize_dc_only(True)
        self.assertTrue(get_optimize_dc_only())

    def test_flag_toggle_works(self):
        """Test that flag can be toggled"""
        set_optimize_dc_only(False)
        self.assertFalse(get_optimize_dc_only())
        set_optimize_dc_only(True)
        self.assertTrue(get_optimize_dc_only())

    def test_dc_only_line_returns_lightweight(self):
        """Test that DC-only lines return _DCOnlyCurrentSource when flag is on"""
        set_optimize_dc_only(True)
        result = _parse_current_source_line("I_test n1 n2 1.5m")
        self.assertIsInstance(result, _DCOnlyCurrentSource)
        self.assertEqual(result.node1, 'n1')
        self.assertEqual(result.node2, 'n2')
        self.assertAlmostEqual(result.dc_value, 1.5e-3)

    def test_dc_only_with_dc_prefix(self):
        """Test DC-only line with 'dc' prefix"""
        set_optimize_dc_only(True)
        result = _parse_current_source_line("I_test n1 n2 dc 2.0m")
        self.assertIsInstance(result, _DCOnlyCurrentSource)
        self.assertAlmostEqual(result.dc_value, 2.0e-3)

    def test_dc_only_with_extra_params_no_waveform(self):
        """Test DC-only line with extra params (wstart=, dv=) but no waveform"""
        set_optimize_dc_only(True)
        # These extra params are ignored by fast-path since no waveform
        result = _parse_current_source_line("I_test n1 n2 1.0m wscale=0.5 wstart=1e-9")
        self.assertIsInstance(result, _DCOnlyCurrentSource)
        self.assertAlmostEqual(result.dc_value, 1.0e-3)

    def test_line_with_pulse_returns_full_current_source(self):
        """Test that lines with pulse waveform return full CurrentSource"""
        set_optimize_dc_only(True)
        result = _parse_current_source_line("I_test n1 n2 0 pulse(0, 1m, 0, 1n, 1n, 5n, 10n)")
        self.assertIsInstance(result, CurrentSource)
        self.assertEqual(len(result.pulses), 1)

    def test_line_with_pwl_returns_full_current_source(self):
        """Test that lines with PWL waveform return full CurrentSource"""
        set_optimize_dc_only(True)
        result = _parse_current_source_line("I_test n1 n2 0 pwl(0 0 1n 1m 2n 0)")
        self.assertIsInstance(result, CurrentSource)
        self.assertEqual(len(result.pwls), 1)

    def test_line_with_static_value_returns_full_current_source(self):
        """Test that lines with static_value= return full CurrentSource"""
        set_optimize_dc_only(True)
        result = _parse_current_source_line("I_test n1 n2 1.0m static_value=0.5m")
        self.assertIsInstance(result, CurrentSource)
        self.assertAlmostEqual(result.static_value, 0.5e-3)

    def test_flag_off_returns_full_current_source(self):
        """Test that flag off returns full CurrentSource for DC-only lines"""
        set_optimize_dc_only(False)
        result = _parse_current_source_line("I_test n1 n2 1.5m")
        self.assertIsInstance(result, CurrentSource)
        self.assertAlmostEqual(result.dc_value, 1.5e-3)

    def test_get_current_at_time_returns_dc_value(self):
        """Test that get_current_at_time returns dc_value for any time"""
        dc_src = _DCOnlyCurrentSource('n1', 'n2', 1.5)
        self.assertEqual(dc_src.get_current_at_time(0), 1.5)
        self.assertEqual(dc_src.get_current_at_time(1e-9), 1.5)
        self.assertEqual(dc_src.get_current_at_time(100e-9), 1.5)

    def test_get_static_current_returns_dc_value(self):
        """Test that get_static_current returns dc_value"""
        dc_src = _DCOnlyCurrentSource('n1', 'n2', 2.5)
        self.assertEqual(dc_src.get_static_current(), 2.5)

    def test_wscale_property_returns_one(self):
        """Test that wscale property always returns 1.0"""
        dc_src = _DCOnlyCurrentSource('n1', 'n2', 1.0)
        self.assertEqual(dc_src.wscale, 1.0)

    def test_has_waveform_data_returns_false(self):
        """Test that has_waveform_data returns False"""
        dc_src = _DCOnlyCurrentSource('n1', 'n2', 1.0)
        self.assertFalse(dc_src.has_waveform_data())

    def test_has_current_data_true_for_nonzero_dc(self):
        """Test that has_current_data returns True for non-zero dc_value"""
        dc_src = _DCOnlyCurrentSource('n1', 'n2', 1.0)
        self.assertTrue(dc_src.has_current_data())

    def test_has_current_data_false_for_zero_dc(self):
        """Test that has_current_data returns False for zero dc_value"""
        dc_src = _DCOnlyCurrentSource('n1', 'n2', 0.0)
        self.assertFalse(dc_src.has_current_data())

    def test_pulses_property_returns_empty_list(self):
        """Test that pulses property returns empty list"""
        dc_src = _DCOnlyCurrentSource('n1', 'n2', 1.0)
        self.assertEqual(dc_src.pulses, [])

    def test_pwls_property_returns_empty_list(self):
        """Test that pwls property returns empty list"""
        dc_src = _DCOnlyCurrentSource('n1', 'n2', 1.0)
        self.assertEqual(dc_src.pwls, [])

    def test_static_value_property_returns_none(self):
        """Test that static_value property returns None"""
        dc_src = _DCOnlyCurrentSource('n1', 'n2', 1.0)
        self.assertIsNone(dc_src.static_value)

    def test_name_property_returns_from_info(self):
        """Test that name property returns info.full_name if available"""
        # Without info, name is empty
        dc_src = _DCOnlyCurrentSource('n1', 'n2', 1.0)
        self.assertEqual(dc_src.name, '')

        # With info, name comes from full_name
        info = InstanceInfo.parse("I_test:VDD:vdd:VSS:0:1:2")
        dc_src_with_info = _DCOnlyCurrentSource('n1', 'n2', 1.0, info)
        self.assertEqual(dc_src_with_info.name, "I_test:VDD:vdd:VSS:0:1:2")

    def test_to_dict_roundtrip_via_current_source_from_dict(self):
        """Test that to_dict can be roundtripped via CurrentSource.from_dict"""
        dc_src = _DCOnlyCurrentSource('n1', 'n2', 1.5)
        d = dc_src.to_dict()

        # Reconstruct via CurrentSource.from_dict
        full_src = CurrentSource.from_dict(d)

        # Verify values match
        self.assertEqual(full_src.node1, dc_src.node1)
        self.assertEqual(full_src.node2, dc_src.node2)
        self.assertEqual(full_src.dc_value, dc_src.dc_value)
        self.assertEqual(full_src.wscale, dc_src.wscale)
        self.assertEqual(full_src.pulses, dc_src.pulses)
        self.assertEqual(full_src.pwls, dc_src.pwls)
        self.assertEqual(full_src.static_value, dc_src.static_value)

    def test_to_dict_with_instance_info(self):
        """Test to_dict serializes InstanceInfo correctly"""
        info = InstanceInfo.parse("i_test:VDD:vdd:VSS:0:1:2")
        dc_src = _DCOnlyCurrentSource('n1', 'n2', 1.0, info)
        d = dc_src.to_dict()

        self.assertIsNotNone(d['info'])
        self.assertEqual(d['info']['vdd_net'], 'VDD')

    def test_node_mutability(self):
        """Test that node1/node2 are mutable (required by _parse_isource)"""
        dc_src = _DCOnlyCurrentSource('original_n1', 'original_n2', 1.0)

        # Modify nodes (as _parse_isource does for boundary handling)
        dc_src.node1 = 'modified_n1'
        dc_src.node2 = 'modified_n2'

        self.assertEqual(dc_src.node1, 'modified_n1')
        self.assertEqual(dc_src.node2, 'modified_n2')

    def test_repr(self):
        """Test __repr__ provides readable output"""
        dc_src = _DCOnlyCurrentSource('n1', 'n2', 1.5)
        repr_str = repr(dc_src)
        self.assertIn('_DCOnlyCurrentSource', repr_str)
        self.assertIn('n1', repr_str)
        self.assertIn('n2', repr_str)
        self.assertIn('1.5', repr_str)

    def test_instance_info_parsed_from_name(self):
        """Test that instance info is parsed from source name"""
        set_optimize_dc_only(True)
        result = _parse_current_source_line("i_U1/cell:VDD:vdd:VSS:0:5:3 n1 n2 1.0m")
        self.assertIsInstance(result, _DCOnlyCurrentSource)
        self.assertIsNotNone(result.info)
        self.assertEqual(result.info.tile_x, 5)
        self.assertEqual(result.info.tile_y, 3)

    def test_pickle_roundtrip(self):
        """Test that _DCOnlyCurrentSource pickles and unpickles correctly"""
        import pickle

        info = InstanceInfo.parse("i_test:VDD:vdd:VSS:0:1:2")
        dc_src = _DCOnlyCurrentSource('n1', 'n2', 1.5, info)

        # Pickle and unpickle
        data = pickle.dumps(dc_src)
        restored = pickle.loads(data)

        # Verify restored object
        self.assertIsInstance(restored, _DCOnlyCurrentSource)
        self.assertEqual(restored.node1, 'n1')
        self.assertEqual(restored.node2, 'n2')
        self.assertEqual(restored.dc_value, 1.5)
        self.assertIsNotNone(restored.info)
        self.assertEqual(restored.info.tile_x, 1)

    # =========================================================================
    # Tests for space-separated PWL/pulse detection with DC-only optimization ON
    # =========================================================================

    def test_pwl_with_space_detected_with_dc_opt_on(self):
        """Test that 'pwl (...)' with space is detected and returns full CurrentSource.

        Regression test: DC-only fast path used to miss 'pwl (...)' with space.
        """
        set_optimize_dc_only(True)
        result = _parse_current_source_line(
            "I_test n1 n2 1.0 pwl (0, 0, 1e-9, 1e-3, 2e-9, 0)"
        )
        # Should return full CurrentSource, not _DCOnlyCurrentSource
        self.assertIsInstance(result, CurrentSource)
        self.assertEqual(len(result.pwls), 1)
        self.assertEqual(len(result.pwls[0].points), 3)

    def test_pulse_with_space_detected_with_dc_opt_on(self):
        """Test that 'pulse (...)' with space is detected and returns full CurrentSource.

        Regression test: DC-only fast path used to miss 'pulse (...)' with space.
        """
        set_optimize_dc_only(True)
        result = _parse_current_source_line(
            "I_test n1 n2 0 pulse (0 1e-3 1e-9 0.1e-9 0.1e-9 2e-9 10e-9)"
        )
        self.assertIsInstance(result, CurrentSource)
        self.assertEqual(len(result.pulses), 1)
        self.assertAlmostEqual(result.pulses[0].v2, 1e-3, places=10)

    def test_pwl_with_space_and_multiple_spaces(self):
        """Test PWL with multiple spaces between keyword and paren."""
        set_optimize_dc_only(True)
        result = _parse_current_source_line(
            "I_test n1 n2 1.0 pwl   (0, 0, 1e-9, 1e-3)"
        )
        self.assertIsInstance(result, CurrentSource)
        self.assertEqual(len(result.pwls), 1)

    def test_empty_pwl_with_space_returns_dc_only(self):
        """Test that empty 'pwl ()' with space returns DC-only (no valid waveform).

        Empty pwl() has no valid waveform data, so it correctly doesn't trigger
        the full parser path - the regex requires content inside parentheses.
        """
        set_optimize_dc_only(True)
        result = _parse_current_source_line("I_test n1 n2 1.0 pwl ()")
        # Empty pwl() doesn't match the waveform regex, so DC-only path is taken
        self.assertIsInstance(result, _DCOnlyCurrentSource)
        self.assertAlmostEqual(result.dc_value, 1.0, places=10)


class TestPDNNodeAttrsPickle(unittest.TestCase):
    """Test PDNNodeAttrs pickle support"""

    def test_pickle_roundtrip(self):
        """Test that PDNNodeAttrs pickles and unpickles correctly"""
        import pickle
        from pdn_parser import PDNNodeAttrs

        attrs = PDNNodeAttrs(
            name='1000_2000_M1',
            _net_type_idx=0,
            tile_id=(1, 2),
            flags=8,  # FLAG_DIE
            voltage=0.95
        )

        # Pickle and unpickle
        data = pickle.dumps(attrs)
        restored = pickle.loads(data)

        # Verify restored object
        self.assertIsInstance(restored, PDNNodeAttrs)
        self.assertEqual(restored.name, '1000_2000_M1')
        self.assertEqual(restored.tile_id, (1, 2))
        self.assertEqual(restored.flags, 8)
        self.assertEqual(restored.voltage, 0.95)


if __name__ == '__main__':
    unittest.main()

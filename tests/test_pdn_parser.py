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
        self.assertIn("n1", self.builder.graph)
        self.assertEqual(self.builder.graph.nodes_dict["n1"]["x"], 1000)
        self.assertEqual(self.builder.graph.nodes_dict["n1"]["y"], 2000)
        self.assertEqual(self.builder.graph.nodes_dict["n1"]["layer"], "M1")
    
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


if __name__ == '__main__':
    unittest.main()

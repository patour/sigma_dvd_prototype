"""Tests for unified power grid core modules."""

import math
import unittest

import networkx as nx
import numpy as np

from generate_power_grid import generate_power_grid, NodeID
from core.node_adapter import NodeInfoExtractor, UnifiedNodeInfo
from core.edge_adapter import EdgeInfoExtractor, UnifiedEdgeInfo, ElementType
from core.unified_model import UnifiedPowerGridModel, GridSource
from core.factory import create_model_from_synthetic, create_model_from_pdn
from core.unified_solver import UnifiedIRDropSolver


def build_small_grid():
    """Build a small synthetic grid for testing."""
    G, loads, pads = generate_power_grid(
        K=3, N0=8, I_N=80, N_vsrc=3,
        max_stripe_res=1.0, max_via_res=0.1,
        load_current=1.0, seed=3, plot=False
    )
    return G, loads, pads


class TestNodeAdapter(unittest.TestCase):
    """Tests for NodeInfoExtractor."""

    def test_synthetic_node_extraction(self):
        """Test extraction from NodeID nodes."""
        G, loads, pads = build_small_grid()
        extractor = NodeInfoExtractor(G)

        # Test a node
        node = list(G.nodes())[0]
        info = extractor.get_info(node)

        self.assertIsInstance(info, UnifiedNodeInfo)
        self.assertEqual(info.original_node, node)
        self.assertIsNotNone(info.layer)
        self.assertEqual(info.layer, node.layer)

    def test_xy_extraction(self):
        """Test coordinate extraction."""
        G, loads, pads = build_small_grid()
        extractor = NodeInfoExtractor(G)

        for node in list(G.nodes())[:5]:
            info = extractor.get_info(node)
            xy = info.xy

            if xy is not None:
                self.assertEqual(len(xy), 2)
                self.assertIsInstance(xy[0], float)
                self.assertIsInstance(xy[1], float)

    def test_caching(self):
        """Test that results are cached."""
        G, loads, pads = build_small_grid()
        extractor = NodeInfoExtractor(G)

        node = list(G.nodes())[0]
        info1 = extractor.get_info(node)
        info2 = extractor.get_info(node)

        self.assertIs(info1, info2)  # Same object (cached)

    def test_string_node_parsing(self):
        """Test parsing of string node names."""
        G = nx.Graph()
        G.add_node("1000_2000_M1")
        G.add_node("500_750")

        extractor = NodeInfoExtractor(G)

        info1 = extractor.get_info("1000_2000_M1")
        self.assertEqual(info1.x, 1000.0)
        self.assertEqual(info1.y, 2000.0)
        self.assertEqual(info1.layer, "M1")

        info2 = extractor.get_info("500_750")
        self.assertEqual(info2.x, 500.0)
        self.assertEqual(info2.y, 750.0)


class TestEdgeAdapter(unittest.TestCase):
    """Tests for EdgeInfoExtractor."""

    def test_synthetic_edge_extraction(self):
        """Test extraction from synthetic grid edges."""
        extractor = EdgeInfoExtractor(is_pdn=False)

        edge_data = {'resistance': 0.5, 'kind': 'via'}
        info = extractor.get_info(edge_data)

        self.assertEqual(info.element_type, ElementType.RESISTOR)
        self.assertEqual(info.resistance, 0.5)
        self.assertEqual(info.conductance, 2.0)

    def test_pdn_edge_extraction(self):
        """Test extraction from PDN edges with unit conversion."""
        extractor = EdgeInfoExtractor(is_pdn=True, resistance_unit_kohm=True)

        # Resistor in kOhms
        edge_data = {'type': 'R', 'value': 1.0, 'elem_name': 'R1'}
        info = extractor.get_info(edge_data)

        self.assertEqual(info.element_type, ElementType.RESISTOR)
        self.assertEqual(info.resistance, 1000.0)  # Converted to Ohms

    def test_element_types(self):
        """Test different element type extraction."""
        extractor = EdgeInfoExtractor(is_pdn=True)

        # Capacitor
        info = extractor.get_info({'type': 'C', 'value': 100.0})
        self.assertEqual(info.element_type, ElementType.CAPACITOR)
        self.assertEqual(info.capacitance, 100.0)

        # Inductor
        info = extractor.get_info({'type': 'L', 'value': 10.0})
        self.assertEqual(info.element_type, ElementType.INDUCTOR)
        self.assertEqual(info.inductance, 10.0)

        # Voltage source
        info = extractor.get_info({'type': 'V', 'value': 1.0})
        self.assertEqual(info.element_type, ElementType.VOLTAGE_SOURCE)
        self.assertEqual(info.voltage, 1.0)


class TestUnifiedModel(unittest.TestCase):
    """Tests for UnifiedPowerGridModel."""

    def test_model_creation(self):
        """Test model creation from synthetic grid."""
        G, loads, pads = build_small_grid()
        model = UnifiedPowerGridModel(G, pads, vdd=1.0, source=GridSource.SYNTHETIC)

        self.assertEqual(model.vdd, 1.0)
        self.assertEqual(len(model.pad_nodes), len(pads))
        self.assertIsNotNone(model.reduced)

    def test_factory_creation(self):
        """Test model creation via factory function."""
        G, loads, pads = build_small_grid()
        model = create_model_from_synthetic(G, pads, vdd=1.0)

        self.assertEqual(model.source, GridSource.SYNTHETIC)
        self.assertEqual(model.vdd, 1.0)

    def test_pdn_factory_missing_voltage_error(self):
        """Test that create_model_from_pdn raises ValueError when voltage cannot be determined."""
        # Create a minimal PDN-like graph without voltage parameters
        G = nx.MultiDiGraph()
        G.add_node("node1_M1")
        G.add_node("node2_M1")
        G.add_edge("node1_M1", "node2_M1", type='R', value=1.0)
        
        # Set up minimal net connectivity but NO voltage parameters
        G.graph['net_connectivity'] = {'VDD': ['node1_M1', 'node2_M1']}
        G.graph['vsrc_nodes'] = {'node1_M1'}  # Has a vsrc node but no voltage info
        G.graph['parameters'] = {}  # Empty - no voltage specified
        
        # Should raise ValueError because voltage cannot be determined
        with self.assertRaises(ValueError) as ctx:
            create_model_from_pdn(G, 'VDD')
        
        self.assertIn("Could not determine nominal voltage", str(ctx.exception))
        self.assertIn("VDD", str(ctx.exception))

    def test_solve_voltages_no_load(self):
        """Test that zero load gives pad voltage everywhere."""
        G, loads, pads = build_small_grid()
        model = create_model_from_synthetic(G, pads, vdd=1.0)

        voltages = model.solve_voltages({})

        # All pad nodes should be at Vdd
        for pad in pads:
            self.assertAlmostEqual(voltages[pad], 1.0, places=10)

        # All voltages should be within [0, Vdd]
        for v in voltages.values():
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.0 + 1e-10)

    def test_solve_voltages_with_load(self):
        """Test that load causes voltage drop."""
        G, loads, pads = build_small_grid()
        model = create_model_from_synthetic(G, pads, vdd=1.0)

        voltages = model.solve_voltages(loads)

        # Pads should be at Vdd
        for pad in pads:
            self.assertAlmostEqual(voltages[pad], 1.0, places=10)

        # Non-pad nodes should have voltage drop
        min_voltage = min(voltages.values())
        self.assertLess(min_voltage, 1.0)

    def test_get_all_layers(self):
        """Test layer enumeration."""
        G, loads, pads = build_small_grid()
        model = create_model_from_synthetic(G, pads, vdd=1.0)

        layers = model.get_all_layers()

        self.assertEqual(len(layers), 3)  # K=3
        self.assertEqual(layers, [0, 1, 2])

    def test_decompose_at_layer(self):
        """Test grid decomposition."""
        G, loads, pads = build_small_grid()
        model = create_model_from_synthetic(G, pads, vdd=1.0)

        top, bottom, ports, vias = model._decompose_at_layer(1)

        # All nodes should be categorized
        total = len(top) + len(bottom) - len(ports)  # ports counted in top
        self.assertLessEqual(total, G.number_of_nodes())

        # Ports should be at partition layer
        for port in ports:
            self.assertEqual(model.get_node_layer(port), 1)


class TestUnifiedSolver(unittest.TestCase):
    """Tests for UnifiedIRDropSolver."""

    def test_flat_solve(self):
        """Test flat solve produces valid results."""
        G, loads, pads = build_small_grid()
        model = create_model_from_synthetic(G, pads, vdd=1.0)
        solver = UnifiedIRDropSolver(model)

        result = solver.solve(loads)

        self.assertIsNotNone(result.voltages)
        self.assertIsNotNone(result.ir_drop)
        self.assertEqual(result.nominal_voltage, 1.0)

        # Check IR-drop consistency
        for node, drop in result.ir_drop.items():
            self.assertAlmostEqual(drop, 1.0 - result.voltages[node], places=10)

    def test_summarize(self):
        """Test summary statistics."""
        G, loads, pads = build_small_grid()
        model = create_model_from_synthetic(G, pads, vdd=1.0)
        solver = UnifiedIRDropSolver(model)

        result = solver.solve(loads)
        summary = solver.summarize(result)

        self.assertIn('min_voltage', summary)
        self.assertIn('max_voltage', summary)
        self.assertIn('max_drop', summary)
        self.assertIn('avg_drop', summary)

        self.assertLessEqual(summary['min_voltage'], summary['max_voltage'])
        self.assertGreaterEqual(summary['max_drop'], summary['avg_drop'])

    def test_hierarchical_solve(self):
        """Test hierarchical solve produces valid results."""
        G, loads, pads = build_small_grid()
        model = create_model_from_synthetic(G, pads, vdd=1.0)
        solver = UnifiedIRDropSolver(model)

        result = solver.solve_hierarchical(loads, partition_layer=1)

        self.assertIsNotNone(result.voltages)
        self.assertIsNotNone(result.port_nodes)
        self.assertEqual(result.partition_layer, 1)

        # Check port voltages exist
        for port in result.port_nodes:
            self.assertIn(port, result.port_voltages)

    def test_current_conservation(self):
        """Test that current is conserved during aggregation."""
        G, loads, pads = build_small_grid()
        model = create_model_from_synthetic(G, pads, vdd=1.0)
        solver = UnifiedIRDropSolver(model)

        result = solver.solve_hierarchical(loads, partition_layer=1)

        # Total load current
        total_load = sum(loads.values())

        # Total port current
        total_port = sum(result.port_currents.values())

        self.assertAlmostEqual(total_load, total_port, places=10)


class TestBackwardCompatibility(unittest.TestCase):
    """Tests to verify backward compatibility with existing PowerGridModel."""

    def test_results_match_original(self):
        """Test that unified model produces same results as original."""
        from irdrop import PowerGridModel as OriginalPowerGridModel

        G, loads, pads = build_small_grid()

        # Original model
        original = OriginalPowerGridModel(G, pads, vdd=1.0)
        original_voltages = original.solve_voltages(loads)

        # Unified model
        unified = create_model_from_synthetic(G, pads, vdd=1.0)
        unified_voltages = unified.solve_voltages(loads)

        # Compare voltages
        for node in original_voltages:
            self.assertAlmostEqual(
                original_voltages[node],
                unified_voltages[node],
                places=10,
                msg=f"Voltage mismatch at {node}"
            )


if __name__ == '__main__':
    unittest.main()

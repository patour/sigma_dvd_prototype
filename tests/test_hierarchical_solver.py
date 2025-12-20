"""Unit tests for hierarchical IR-drop solver functionality.

Tests the following components:
- _decompose_at_layer: PDN decomposition into top/bottom grids
- _compute_shortest_path_resistance_in_subgrid: Dijkstra-based path resistance
- _aggregate_currents_to_ports: Current aggregation with different weighting methods
- _build_subgrid_system: Subgrid ReducedSystem construction
- _solve_subgrid: Subgrid voltage solving
- solve_hierarchical: Full hierarchical solve integration
"""

import math
import unittest

import numpy as np

from generate_power_grid import generate_power_grid, NodeID
from irdrop import PowerGridModel, HierarchicalSolveResult


def build_small_grid():
    """Build a small test grid with 3 layers."""
    G, loads, pads = generate_power_grid(
        K=3, N0=8, I_N=10, N_vsrc=2,
        max_stripe_res=1.0, max_via_res=0.1,
        load_current=0.1, seed=42, plot=False
    )
    return G, loads, pads


def build_medium_grid():
    """Build a medium test grid with 5 layers."""
    G, loads, pads = generate_power_grid(
        K=5, N0=8, I_N=15, N_vsrc=4,
        max_stripe_res=1.0, max_via_res=0.1,
        load_current=0.1, seed=42, plot=False
    )
    return G, loads, pads


class TestDecomposeAtLayer(unittest.TestCase):
    """Tests for _decompose_at_layer method."""

    def test_basic_decomposition(self):
        """Test basic decomposition produces non-empty sets."""
        G, loads, pads = build_small_grid()
        model = PowerGridModel(G, pads, vdd=1.0)

        top_nodes, bottom_nodes, ports, via_edges = model._decompose_at_layer(1)

        self.assertGreater(len(top_nodes), 0, "Top grid should have nodes")
        self.assertGreater(len(bottom_nodes), 0, "Bottom grid should have nodes")
        self.assertGreater(len(ports), 0, "Should have port nodes")
        self.assertGreater(len(via_edges), 0, "Should have via edges")

    def test_partition_covers_all_nodes(self):
        """Top and bottom grids together should cover all nodes."""
        G, loads, pads = build_small_grid()
        model = PowerGridModel(G, pads, vdd=1.0)

        top_nodes, bottom_nodes, ports, _ = model._decompose_at_layer(1)

        all_nodes = set(G.nodes())
        covered = top_nodes | bottom_nodes

        self.assertEqual(covered, all_nodes, "Decomposition should cover all nodes")

    def test_no_overlap_except_ports(self):
        """Top and bottom grids should not overlap (ports are in top only)."""
        G, loads, pads = build_small_grid()
        model = PowerGridModel(G, pads, vdd=1.0)

        top_nodes, bottom_nodes, ports, _ = model._decompose_at_layer(1)

        overlap = top_nodes & bottom_nodes
        self.assertEqual(len(overlap), 0, "Top and bottom should not overlap")

    def test_ports_in_top_grid(self):
        """Port nodes should be in the top grid."""
        G, loads, pads = build_small_grid()
        model = PowerGridModel(G, pads, vdd=1.0)

        top_nodes, _, ports, _ = model._decompose_at_layer(1)

        self.assertTrue(ports.issubset(top_nodes), "Ports should be in top grid")

    def test_ports_at_partition_layer(self):
        """Port nodes should be at the partition layer."""
        G, loads, pads = build_small_grid()
        model = PowerGridModel(G, pads, vdd=1.0)
        partition_layer = 1

        _, _, ports, _ = model._decompose_at_layer(partition_layer)

        for port in ports:
            layer = getattr(port, 'layer', G.nodes[port].get('layer', 0))
            self.assertEqual(layer, partition_layer, 
                           f"Port {port} should be at layer {partition_layer}")

    def test_invalid_partition_layer_zero(self):
        """Partition at layer 0 should raise ValueError."""
        G, loads, pads = build_small_grid()
        model = PowerGridModel(G, pads, vdd=1.0)

        with self.assertRaises(ValueError):
            model._decompose_at_layer(0)

    def test_invalid_partition_layer_too_high(self):
        """Partition above max layer should raise ValueError."""
        G, loads, pads = build_small_grid()
        model = PowerGridModel(G, pads, vdd=1.0)

        with self.assertRaises(ValueError):
            model._decompose_at_layer(100)

    def test_different_partition_layers(self):
        """Different partition layers should produce different decompositions."""
        G, loads, pads = build_medium_grid()
        model = PowerGridModel(G, pads, vdd=1.0)

        top1, bottom1, ports1, _ = model._decompose_at_layer(1)
        top2, bottom2, ports2, _ = model._decompose_at_layer(2)

        # Higher partition layer means larger top grid
        self.assertGreater(len(top1), len(top2), 
                          "Lower partition should have larger top grid")
        self.assertLess(len(bottom1), len(bottom2),
                       "Lower partition should have smaller bottom grid")


class TestShortestPathResistance(unittest.TestCase):
    """Tests for _compute_shortest_path_resistance_in_subgrid method."""

    def test_basic_shortest_path(self):
        """Test that shortest path resistance is computed."""
        G, loads, pads = build_small_grid()
        model = PowerGridModel(G, pads, vdd=1.0)

        top_nodes, bottom_nodes, ports, _ = model._decompose_at_layer(1)
        subgrid = bottom_nodes | ports

        load_node = list(loads.keys())[0]
        port_list = list(ports)

        resistances = model._compute_shortest_path_resistance_in_subgrid(
            subgrid_nodes=subgrid,
            source_node=load_node,
            target_nodes=port_list,
        )

        self.assertEqual(len(resistances), len(port_list))

    def test_shortest_path_positive(self):
        """Shortest path resistance should be positive for reachable nodes."""
        G, loads, pads = build_small_grid()
        model = PowerGridModel(G, pads, vdd=1.0)

        top_nodes, bottom_nodes, ports, _ = model._decompose_at_layer(1)
        subgrid = bottom_nodes | ports

        load_node = list(loads.keys())[0]
        port_list = list(ports)

        resistances = model._compute_shortest_path_resistance_in_subgrid(
            subgrid_nodes=subgrid,
            source_node=load_node,
            target_nodes=port_list,
        )

        for port, r in resistances.items():
            if r < float('inf'):
                self.assertGreater(r, 0, f"Resistance to {port} should be positive")

    def test_shortest_path_to_self_is_zero(self):
        """Shortest path from a node to itself should be zero."""
        G, loads, pads = build_small_grid()
        model = PowerGridModel(G, pads, vdd=1.0)

        top_nodes, bottom_nodes, ports, _ = model._decompose_at_layer(1)
        subgrid = bottom_nodes | ports

        port_list = list(ports)
        source_port = port_list[0]

        resistances = model._compute_shortest_path_resistance_in_subgrid(
            subgrid_nodes=subgrid,
            source_node=source_port,
            target_nodes=[source_port],
        )

        self.assertAlmostEqual(resistances[source_port], 0.0, places=10)

    def test_unreachable_node_returns_inf(self):
        """Unreachable target should return infinity."""
        G, loads, pads = build_small_grid()
        model = PowerGridModel(G, pads, vdd=1.0)

        # Create a small subgrid that doesn't include some nodes
        some_nodes = set(list(G.nodes())[:5])
        
        # A node not in the subgrid
        outside_node = list(set(G.nodes()) - some_nodes)[0]
        source = list(some_nodes)[0]

        resistances = model._compute_shortest_path_resistance_in_subgrid(
            subgrid_nodes=some_nodes,
            source_node=source,
            target_nodes=[outside_node],
        )

        self.assertEqual(resistances[outside_node], float('inf'))

    def test_source_not_in_subgrid(self):
        """If source is not in subgrid, all targets should return infinity."""
        G, loads, pads = build_small_grid()
        model = PowerGridModel(G, pads, vdd=1.0)

        some_nodes = set(list(G.nodes())[:5])
        outside_node = list(set(G.nodes()) - some_nodes)[0]
        targets = list(some_nodes)[:2]

        resistances = model._compute_shortest_path_resistance_in_subgrid(
            subgrid_nodes=some_nodes,
            source_node=outside_node,
            target_nodes=targets,
        )

        for t in targets:
            self.assertEqual(resistances[t], float('inf'))


class TestAggregateCurrentsToPorts(unittest.TestCase):
    """Tests for _aggregate_currents_to_ports method."""

    def test_basic_aggregation(self):
        """Test that current aggregation produces results."""
        G, loads, pads = build_small_grid()
        model = PowerGridModel(G, pads, vdd=1.0)

        top_nodes, bottom_nodes, ports, _ = model._decompose_at_layer(1)

        port_currents, agg_map = model._aggregate_currents_to_ports(
            current_injections=loads,
            bottom_grid_nodes=bottom_nodes,
            port_nodes=ports,
            top_k=2,
            weighting="effective",
        )

        self.assertEqual(len(port_currents), len(ports))

    def test_current_conservation(self):
        """Total aggregated port current should equal total bottom current."""
        G, loads, pads = build_small_grid()
        model = PowerGridModel(G, pads, vdd=1.0)

        top_nodes, bottom_nodes, ports, _ = model._decompose_at_layer(1)

        port_currents, _ = model._aggregate_currents_to_ports(
            current_injections=loads,
            bottom_grid_nodes=bottom_nodes,
            port_nodes=ports,
            top_k=2,
            weighting="effective",
        )

        # Only loads in bottom grid should be aggregated
        bottom_loads = {n: c for n, c in loads.items() if n in bottom_nodes}
        total_bottom_current = sum(bottom_loads.values())
        total_port_current = sum(port_currents.values())

        self.assertAlmostEqual(total_port_current, total_bottom_current, places=10,
                              msg="Current should be conserved during aggregation")

    def test_aggregation_map_structure(self):
        """Aggregation map should have correct structure."""
        G, loads, pads = build_small_grid()
        model = PowerGridModel(G, pads, vdd=1.0)

        top_nodes, bottom_nodes, ports, _ = model._decompose_at_layer(1)

        _, agg_map = model._aggregate_currents_to_ports(
            current_injections=loads,
            bottom_grid_nodes=bottom_nodes,
            port_nodes=ports,
            top_k=2,
            weighting="effective",
        )

        for load_node, contributions in agg_map.items():
            # Each contribution should be (port, weight, current)
            self.assertTrue(len(contributions) <= 2, "Should use top-k ports")
            
            weights_sum = sum(w for _, w, _ in contributions)
            self.assertAlmostEqual(weights_sum, 1.0, places=10,
                                  msg="Weights should sum to 1")

    def test_top_k_limiting(self):
        """Should only use top_k ports per load."""
        G, loads, pads = build_medium_grid()
        model = PowerGridModel(G, pads, vdd=1.0)

        top_nodes, bottom_nodes, ports, _ = model._decompose_at_layer(2)

        for k in [1, 2, 3]:
            _, agg_map = model._aggregate_currents_to_ports(
                current_injections=loads,
                bottom_grid_nodes=bottom_nodes,
                port_nodes=ports,
                top_k=k,
                weighting="effective",
            )

            for load_node, contributions in agg_map.items():
                self.assertLessEqual(len(contributions), k,
                                    f"Should use at most {k} ports")

    def test_weighting_effective(self):
        """Test effective resistance weighting method."""
        G, loads, pads = build_small_grid()
        model = PowerGridModel(G, pads, vdd=1.0)

        top_nodes, bottom_nodes, ports, _ = model._decompose_at_layer(1)

        port_currents, _ = model._aggregate_currents_to_ports(
            current_injections=loads,
            bottom_grid_nodes=bottom_nodes,
            port_nodes=ports,
            top_k=2,
            weighting="effective",
        )

        # Should complete without error and produce valid currents
        self.assertTrue(all(c >= 0 for c in port_currents.values()))

    def test_weighting_shortest_path(self):
        """Test shortest path resistance weighting method."""
        G, loads, pads = build_small_grid()
        model = PowerGridModel(G, pads, vdd=1.0)

        top_nodes, bottom_nodes, ports, _ = model._decompose_at_layer(1)

        port_currents, _ = model._aggregate_currents_to_ports(
            current_injections=loads,
            bottom_grid_nodes=bottom_nodes,
            port_nodes=ports,
            top_k=2,
            weighting="shortest_path",
        )

        # Should complete without error and produce valid currents
        self.assertTrue(all(c >= 0 for c in port_currents.values()))

    def test_different_weighting_can_differ(self):
        """Different weighting methods may produce different results."""
        G, loads, pads = build_medium_grid()
        model = PowerGridModel(G, pads, vdd=1.0)

        top_nodes, bottom_nodes, ports, _ = model._decompose_at_layer(2)

        _, agg_map_eff = model._aggregate_currents_to_ports(
            current_injections=loads,
            bottom_grid_nodes=bottom_nodes,
            port_nodes=ports,
            top_k=2,
            weighting="effective",
        )

        _, agg_map_sp = model._aggregate_currents_to_ports(
            current_injections=loads,
            bottom_grid_nodes=bottom_nodes,
            port_nodes=ports,
            top_k=2,
            weighting="shortest_path",
        )

        # Both should have entries
        self.assertGreater(len(agg_map_eff), 0)
        self.assertGreater(len(agg_map_sp), 0)

    def test_invalid_weighting_raises(self):
        """Invalid weighting method should raise ValueError."""
        G, loads, pads = build_small_grid()
        model = PowerGridModel(G, pads, vdd=1.0)

        top_nodes, bottom_nodes, ports, _ = model._decompose_at_layer(1)

        with self.assertRaises(ValueError):
            model._aggregate_currents_to_ports(
                current_injections=loads,
                bottom_grid_nodes=bottom_nodes,
                port_nodes=ports,
                top_k=2,
                weighting="invalid_method",
            )

    def test_empty_currents(self):
        """Empty current injections should return zero port currents."""
        G, loads, pads = build_small_grid()
        model = PowerGridModel(G, pads, vdd=1.0)

        top_nodes, bottom_nodes, ports, _ = model._decompose_at_layer(1)

        port_currents, agg_map = model._aggregate_currents_to_ports(
            current_injections={},
            bottom_grid_nodes=bottom_nodes,
            port_nodes=ports,
            top_k=2,
            weighting="effective",
        )

        self.assertTrue(all(c == 0 for c in port_currents.values()))
        self.assertEqual(len(agg_map), 0)


class TestBuildSubgridSystem(unittest.TestCase):
    """Tests for _build_subgrid_system method."""

    def test_basic_subgrid_system(self):
        """Test that subgrid system is built correctly."""
        G, loads, pads = build_small_grid()
        model = PowerGridModel(G, pads, vdd=1.0)

        top_nodes, _, _, _ = model._decompose_at_layer(1)
        pad_set = set(pads) & top_nodes

        system = model._build_subgrid_system(
            subgrid_nodes=top_nodes,
            dirichlet_nodes=pad_set,
            dirichlet_voltage=1.0,
        )

        self.assertIsNotNone(system)
        self.assertGreater(len(system.unknown_nodes), 0)

    def test_empty_subgrid_returns_none(self):
        """Empty subgrid should return None."""
        G, loads, pads = build_small_grid()
        model = PowerGridModel(G, pads, vdd=1.0)

        system = model._build_subgrid_system(
            subgrid_nodes=set(),
            dirichlet_nodes=set(),
            dirichlet_voltage=1.0,
        )

        self.assertIsNone(system)

    def test_all_dirichlet_returns_none(self):
        """If all nodes are Dirichlet, should return None."""
        G, loads, pads = build_small_grid()
        model = PowerGridModel(G, pads, vdd=1.0)

        some_nodes = set(pads[:2])

        system = model._build_subgrid_system(
            subgrid_nodes=some_nodes,
            dirichlet_nodes=some_nodes,
            dirichlet_voltage=1.0,
        )

        self.assertIsNone(system)

    def test_factorization_works(self):
        """The LU factorization should work for solving."""
        G, loads, pads = build_small_grid()
        model = PowerGridModel(G, pads, vdd=1.0)

        top_nodes, _, _, _ = model._decompose_at_layer(1)
        pad_set = set(pads) & top_nodes

        system = model._build_subgrid_system(
            subgrid_nodes=top_nodes,
            dirichlet_nodes=pad_set,
            dirichlet_voltage=1.0,
        )

        # Try to use the factorization
        rhs = np.zeros(len(system.unknown_nodes))
        result = system.lu(rhs)

        self.assertEqual(len(result), len(system.unknown_nodes))


class TestSolveSubgrid(unittest.TestCase):
    """Tests for _solve_subgrid method."""

    def test_basic_subgrid_solve(self):
        """Test basic subgrid voltage solve."""
        G, loads, pads = build_small_grid()
        model = PowerGridModel(G, pads, vdd=1.0)

        top_nodes, _, _, _ = model._decompose_at_layer(1)
        pad_set = set(pads) & top_nodes

        system = model._build_subgrid_system(
            subgrid_nodes=top_nodes,
            dirichlet_nodes=pad_set,
            dirichlet_voltage=1.0,
        )

        voltages = model._solve_subgrid(
            reduced_system=system,
            current_injections={},
            dirichlet_voltages=None,
        )

        self.assertEqual(len(voltages), len(top_nodes))

    def test_dirichlet_nodes_have_correct_voltage(self):
        """Dirichlet nodes should have the specified voltage."""
        G, loads, pads = build_small_grid()
        model = PowerGridModel(G, pads, vdd=1.0)

        top_nodes, _, _, _ = model._decompose_at_layer(1)
        pad_set = set(pads) & top_nodes

        system = model._build_subgrid_system(
            subgrid_nodes=top_nodes,
            dirichlet_nodes=pad_set,
            dirichlet_voltage=1.0,
        )

        voltages = model._solve_subgrid(
            reduced_system=system,
            current_injections={},
            dirichlet_voltages=None,
        )

        for pad in pad_set:
            self.assertAlmostEqual(voltages[pad], 1.0, places=10)

    def test_custom_dirichlet_voltages(self):
        """Custom Dirichlet voltages should be applied."""
        G, loads, pads = build_small_grid()
        model = PowerGridModel(G, pads, vdd=1.0)

        top_nodes, bottom_nodes, ports, _ = model._decompose_at_layer(1)
        bottom_subgrid = bottom_nodes | ports

        system = model._build_subgrid_system(
            subgrid_nodes=bottom_subgrid,
            dirichlet_nodes=ports,
            dirichlet_voltage=1.0,
        )

        # Custom voltages for ports
        custom_voltages = {p: 0.95 for p in ports}

        voltages = model._solve_subgrid(
            reduced_system=system,
            current_injections={},
            dirichlet_voltages=custom_voltages,
        )

        for port in ports:
            self.assertAlmostEqual(voltages[port], 0.95, places=10)

    def test_with_current_injection(self):
        """Current injection should affect voltages."""
        G, loads, pads = build_small_grid()
        model = PowerGridModel(G, pads, vdd=1.0)

        top_nodes, _, _, _ = model._decompose_at_layer(1)
        pad_set = set(pads) & top_nodes

        system = model._build_subgrid_system(
            subgrid_nodes=top_nodes,
            dirichlet_nodes=pad_set,
            dirichlet_voltage=1.0,
        )

        # Solve with no current
        v_no_current = model._solve_subgrid(
            reduced_system=system,
            current_injections={},
        )

        # Find a non-pad node to inject current
        non_pad = [n for n in top_nodes if n not in pad_set][0]

        # Solve with current
        v_with_current = model._solve_subgrid(
            reduced_system=system,
            current_injections={non_pad: 1.0},
        )

        # Voltage at the injection node should be lower
        self.assertLess(v_with_current[non_pad], v_no_current[non_pad])


class TestSolveHierarchical(unittest.TestCase):
    """Integration tests for solve_hierarchical method."""

    def test_basic_hierarchical_solve(self):
        """Test that hierarchical solve completes without error."""
        G, loads, pads = build_small_grid()
        model = PowerGridModel(G, pads, vdd=1.0)

        result = model.solve_hierarchical(
            current_injections=loads,
            partition_layer=1,
            top_k=2,
        )

        self.assertIsInstance(result, HierarchicalSolveResult)

    def test_result_structure(self):
        """Test that result has all expected fields."""
        G, loads, pads = build_small_grid()
        model = PowerGridModel(G, pads, vdd=1.0)

        result = model.solve_hierarchical(loads, partition_layer=1)

        # Check all fields exist and are non-empty
        self.assertGreater(len(result.voltages), 0)
        self.assertGreater(len(result.ir_drop), 0)
        self.assertEqual(result.partition_layer, 1)
        self.assertGreater(len(result.top_grid_voltages), 0)
        self.assertGreater(len(result.bottom_grid_voltages), 0)
        self.assertGreater(len(result.port_nodes), 0)
        self.assertGreater(len(result.port_voltages), 0)

    def test_voltage_range(self):
        """All voltages should be between 0 and Vdd."""
        G, loads, pads = build_small_grid()
        model = PowerGridModel(G, pads, vdd=1.0)

        result = model.solve_hierarchical(loads, partition_layer=1)

        for v in result.voltages.values():
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.0)

    def test_pad_voltages_at_vdd(self):
        """Pad nodes should be at Vdd."""
        G, loads, pads = build_small_grid()
        model = PowerGridModel(G, pads, vdd=1.0)

        result = model.solve_hierarchical(loads, partition_layer=1)

        for pad in pads:
            self.assertAlmostEqual(result.voltages[pad], 1.0, places=10)

    def test_ir_drop_consistent(self):
        """IR-drop should be Vdd - voltage."""
        G, loads, pads = build_small_grid()
        model = PowerGridModel(G, pads, vdd=1.0)

        result = model.solve_hierarchical(loads, partition_layer=1)

        for node, v in result.voltages.items():
            expected_drop = 1.0 - v
            self.assertAlmostEqual(result.ir_drop[node], expected_drop, places=10)

    def test_different_partition_layers(self):
        """Different partition layers should produce valid results."""
        G, loads, pads = build_medium_grid()
        model = PowerGridModel(G, pads, vdd=1.0)

        for partition_layer in [1, 2, 3]:
            result = model.solve_hierarchical(loads, partition_layer=partition_layer)

            self.assertEqual(result.partition_layer, partition_layer)
            self.assertGreater(len(result.voltages), 0)

    def test_different_weighting_methods(self):
        """Both weighting methods should work."""
        G, loads, pads = build_small_grid()
        model = PowerGridModel(G, pads, vdd=1.0)

        result_eff = model.solve_hierarchical(
            loads, partition_layer=1, weighting="effective"
        )
        result_sp = model.solve_hierarchical(
            loads, partition_layer=1, weighting="shortest_path"
        )

        # Both should produce valid results
        self.assertGreater(len(result_eff.voltages), 0)
        self.assertGreater(len(result_sp.voltages), 0)

    def test_comparison_with_flat_solve(self):
        """Hierarchical solve should produce voltages in similar range to flat solve."""
        G, loads, pads = build_small_grid()
        model = PowerGridModel(G, pads, vdd=1.0)

        # Flat solve
        flat_voltages = model.solve_voltages(loads)
        flat_min = min(flat_voltages.values())
        flat_max = max(flat_voltages.values())

        # Hierarchical solve
        hier_result = model.solve_hierarchical(loads, partition_layer=1)
        hier_min = min(hier_result.voltages.values())
        hier_max = max(hier_result.voltages.values())

        # Should be in similar range (not exact due to approximation)
        self.assertAlmostEqual(flat_max, hier_max, places=5,
                              msg="Max voltage should be similar")
        # Min voltage may differ due to approximation, but should be reasonable
        self.assertGreater(hier_min, 0.5, "Min voltage should be reasonable")

    def test_no_currents(self):
        """With no currents, all nodes should be at Vdd."""
        G, loads, pads = build_small_grid()
        model = PowerGridModel(G, pads, vdd=1.0)

        result = model.solve_hierarchical({}, partition_layer=1)

        for v in result.voltages.values():
            self.assertAlmostEqual(v, 1.0, places=10)

    def test_invalid_partition_layer_raises(self):
        """Invalid partition layer should raise ValueError."""
        G, loads, pads = build_small_grid()
        model = PowerGridModel(G, pads, vdd=1.0)

        with self.assertRaises(ValueError):
            model.solve_hierarchical(loads, partition_layer=0)

        with self.assertRaises(ValueError):
            model.solve_hierarchical(loads, partition_layer=100)

    def test_port_voltages_match_top_grid(self):
        """Port voltages should match the corresponding values in top_grid_voltages."""
        G, loads, pads = build_small_grid()
        model = PowerGridModel(G, pads, vdd=1.0)

        result = model.solve_hierarchical(loads, partition_layer=1)

        for port, port_v in result.port_voltages.items():
            top_v = result.top_grid_voltages[port]
            self.assertAlmostEqual(port_v, top_v, places=10)

    def test_top_k_parameter(self):
        """top_k parameter should affect aggregation."""
        G, loads, pads = build_medium_grid()
        model = PowerGridModel(G, pads, vdd=1.0)

        result_k1 = model.solve_hierarchical(loads, partition_layer=2, top_k=1)
        result_k3 = model.solve_hierarchical(loads, partition_layer=2, top_k=3)

        # Check aggregation maps have different structures
        for load in result_k1.aggregation_map:
            self.assertLessEqual(len(result_k1.aggregation_map[load]), 1)

        for load in result_k3.aggregation_map:
            self.assertLessEqual(len(result_k3.aggregation_map[load]), 3)


class TestHierarchicalSolveResultDataclass(unittest.TestCase):
    """Tests for HierarchicalSolveResult dataclass."""

    def test_dataclass_fields(self):
        """Test that all expected fields are accessible."""
        G, loads, pads = build_small_grid()
        model = PowerGridModel(G, pads, vdd=1.0)

        result = model.solve_hierarchical(loads, partition_layer=1)

        # Access all fields
        _ = result.voltages
        _ = result.ir_drop
        _ = result.partition_layer
        _ = result.top_grid_voltages
        _ = result.bottom_grid_voltages
        _ = result.port_nodes
        _ = result.port_voltages
        _ = result.port_currents
        _ = result.aggregation_map

    def test_aggregation_map_default(self):
        """aggregation_map should default to empty dict."""
        from irdrop.power_grid_model import HierarchicalSolveResult

        result = HierarchicalSolveResult(
            voltages={},
            ir_drop={},
            partition_layer=1,
            top_grid_voltages={},
            bottom_grid_voltages={},
            port_nodes=set(),
            port_voltages={},
            port_currents={},
        )

        self.assertEqual(result.aggregation_map, {})


if __name__ == '__main__':
    unittest.main()


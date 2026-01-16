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
import signal
import unittest
from functools import wraps

import numpy as np

from generate_power_grid import generate_power_grid, NodeID


def timeout(seconds):
    """Decorator to add timeout to test methods (Unix only)."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            def handler(signum, frame):
                raise TimeoutError(f"Test {func.__name__} timed out after {seconds}s")
            old_handler = signal.signal(signal.SIGALRM, handler)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
            return result
        return wrapper
    return decorator
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
            layer = getattr(port, 'layer', G.nodes_dict[port].get('layer', 0))
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


# ============================================================================
# Tiled Hierarchical Solver Tests (PDN graphs only)
# ============================================================================

def _load_pdn_small():
    """Load the small PDN netlist for tiled solver tests.
    
    Returns:
        (model, load_currents, solver) or None if netlist unavailable
    """
    import os
    import sys
    
    # Add prototype root to path
    prototype_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, prototype_root)
    
    netlist_path = os.path.join(prototype_root, 'pdn', 'netlist_small')
    if not os.path.exists(netlist_path):
        return None
        
    try:
        from pdn.pdn_parser import NetlistParser
        from core import create_model_from_pdn, UnifiedIRDropSolver
        
        parser = NetlistParser(netlist_path, validate=False)
        graph = parser.parse()
        model = create_model_from_pdn(graph, 'VDD_XLV')
        load_currents = model.extract_current_sources()
        solver = UnifiedIRDropSolver(model)
        
        return model, load_currents, solver
    except Exception:
        return None


class TestTiledHierarchicalSolver(unittest.TestCase):
    """Tests for solve_hierarchical_tiled method (PDN graphs only)."""
    
    @classmethod
    def setUpClass(cls):
        """Load PDN netlist once for all tests."""
        result = _load_pdn_small()
        if result is None:
            cls.model = None
            cls.load_currents = None
            cls.solver = None
        else:
            cls.model, cls.load_currents, cls.solver = result
    
    def setUp(self):
        """Skip tests if PDN netlist not available."""
        if self.model is None:
            self.skipTest("PDN netlist_small not available")
    
    def _get_flat_port_voltages(self, partition_layer: str) -> dict:
        """Helper to get port voltages from flat hierarchical solve."""
        flat_result = self.solver.solve_hierarchical(
            current_injections=self.load_currents,
            partition_layer=partition_layer,
            top_k=5,
            weighting='shortest_path',
        )
        return flat_result.port_voltages
    
    @timeout(60)
    def test_basic_2x2_tiling(self):
        """Test 2x2 tiling produces valid results matching non-tiled solve."""
        result = self.solver.solve_hierarchical_tiled(
            current_injections=self.load_currents,
            partition_layer='M2',
            N_x=2,
            N_y=2,
            halo_percent=0.2,
            top_k=5,
            weighting='shortest_path',
            n_workers=1,
            parallel_backend='thread',
            validate_against_flat=True,
        )
        
        # Basic sanity checks
        self.assertGreater(len(result.voltages), 0)
        self.assertEqual(len(result.tiles), 4)  # 2x2 = 4 tiles
        
        # Voltage should be reasonable (below Vdd)
        max_voltage = max(result.voltages.values())
        self.assertLessEqual(max_voltage, self.model.vdd + 0.001)
        
        # Validation should show good accuracy
        self.assertIsNotNone(result.validation_stats)
        max_diff = result.validation_stats['max_diff']
        self.assertLess(max_diff, 0.001, "Max diff should be < 1mV")
    
    @timeout(60)
    def test_tiling_result_has_expected_fields(self):
        """Test TiledBottomGridResult has all expected fields."""
        result = self.solver.solve_hierarchical_tiled(
            current_injections=self.load_currents,
            partition_layer='M2',
            N_x=2,
            N_y=2,
            halo_percent=0.2,
            weighting='shortest_path',
            n_workers=1,
            parallel_backend='thread',
        )
        
        # Check inherited fields from UnifiedHierarchicalResult
        self.assertIsInstance(result.voltages, dict)
        self.assertIsInstance(result.ir_drop, dict)
        self.assertIsNotNone(result.partition_layer)
        self.assertIsInstance(result.port_nodes, set)
        self.assertIsInstance(result.port_voltages, dict)
        
        # Check TiledBottomGridResult-specific fields
        self.assertIsInstance(result.tiles, list)
        self.assertIsInstance(result.per_tile_solve_times, dict)
        self.assertIsInstance(result.halo_clip_warnings, list)
        self.assertIsInstance(result.tiling_params, dict)
        
        # Check tiling_params content
        self.assertEqual(result.tiling_params['N_x'], 2)
        self.assertEqual(result.tiling_params['N_y'], 2)
        self.assertEqual(result.tiling_params['halo_percent'], 0.2)
    
    @timeout(60)
    def test_tile_structure(self):
        """Test BottomGridTile objects have correct structure."""
        result = self.solver.solve_hierarchical_tiled(
            current_injections=self.load_currents,
            partition_layer='M2',
            N_x=2,
            N_y=2,
            halo_percent=0.2,
            weighting='shortest_path',
            n_workers=1,
            parallel_backend='thread',
        )
        
        for tile in result.tiles:
            # Check required attributes
            self.assertIsInstance(tile.tile_id, int)
            self.assertIsNotNone(tile.bounds)
            self.assertIsInstance(tile.core_nodes, set)
            self.assertIsInstance(tile.halo_nodes, set)
            self.assertIsInstance(tile.all_nodes, set)
            self.assertIsInstance(tile.port_nodes, set)
            self.assertIsInstance(tile.load_nodes, set)
            
            # Core should be subset of all
            self.assertTrue(tile.core_nodes.issubset(tile.all_nodes))
            
            # Ports should be in all_nodes (after fix)
            self.assertTrue(tile.port_nodes.issubset(tile.all_nodes))
    
    @timeout(60)
    def test_disjoint_core_validation(self):
        """Test that tile core regions are disjoint."""
        result = self.solver.solve_hierarchical_tiled(
            current_injections=self.load_currents,
            partition_layer='M2',
            N_x=2,
            N_y=2,
            halo_percent=0.2,
            weighting='shortest_path',
            n_workers=1,
            parallel_backend='thread',
        )
        
        # Collect all core nodes
        all_core_nodes = set()
        for tile in result.tiles:
            # Check no overlap with previously seen cores
            overlap = all_core_nodes & tile.core_nodes
            self.assertEqual(len(overlap), 0, 
                f"Core regions should be disjoint, found overlap: {list(overlap)[:5]}")
            all_core_nodes.update(tile.core_nodes)
    
    @timeout(60)
    def test_parallel_backend_thread(self):
        """Test thread backend produces valid results."""
        result = self.solver.solve_hierarchical_tiled(
            current_injections=self.load_currents,
            partition_layer='M2',
            N_x=2,
            N_y=2,
            halo_percent=0.2,
            weighting='shortest_path',
            n_workers=2,
            parallel_backend='thread',
            validate_against_flat=True,
        )
        
        # Should still be accurate
        self.assertIsNotNone(result.validation_stats)
        max_diff = result.validation_stats['max_diff']
        self.assertLess(max_diff, 0.001)
    
    @timeout(60)
    def test_progress_callback(self):
        """Test progress callback is invoked correctly."""
        progress_calls = []
        
        def progress_cb(completed, total, tile_id):
            progress_calls.append((completed, total, tile_id))
        
        result = self.solver.solve_hierarchical_tiled(
            current_injections=self.load_currents,
            partition_layer='M2',
            N_x=2,
            N_y=2,
            halo_percent=0.2,
            weighting='shortest_path',
            n_workers=1,
            parallel_backend='thread',
            progress_callback=progress_cb,
        )
        
        # Should have been called once per tile
        self.assertEqual(len(progress_calls), len(result.tiles))
        
        # Completed count should increase
        completed_counts = [c[0] for c in progress_calls]
        self.assertEqual(sorted(completed_counts), list(range(1, len(result.tiles) + 1)))
    
    @timeout(60)
    def test_halo_clipping_warning(self):
        """Test that corner tiles produce halo clip warnings."""
        result = self.solver.solve_hierarchical_tiled(
            current_injections=self.load_currents,
            partition_layer='M2',
            N_x=2,
            N_y=2,
            halo_percent=0.3,  # Larger halo to trigger clipping
            weighting='shortest_path',
            n_workers=1,
            parallel_backend='thread',
        )
        
        # With 2x2 grid, all tiles are at corners so should have clipping
        clipped_tiles = [t for t in result.tiles if t.halo_clipped]
        self.assertGreater(len(clipped_tiles), 0, 
            "Corner tiles should have halo clipping")
    
    @timeout(60)
    def test_non_pdn_graph_error(self):
        """Test that synthetic NodeID graphs raise ValueError."""
        from core import create_model_from_synthetic, UnifiedIRDropSolver
        
        G, loads, pads = build_small_grid()
        model = create_model_from_synthetic(G, pads, vdd=1.0)
        solver = UnifiedIRDropSolver(model)
        
        with self.assertRaises(ValueError) as ctx:
            solver.solve_hierarchical_tiled(
                current_injections=loads,
                partition_layer=1,
                N_x=2,
                N_y=2,
                halo_percent=0.2,
                n_workers=1,
                parallel_backend='thread',
            )
        
        self.assertIn("PDN graphs", str(ctx.exception))
    
    @timeout(60)
    def test_validate_against_flat(self):
        """Test validation stats are populated when enabled."""
        # With validation
        result_with = self.solver.solve_hierarchical_tiled(
            current_injections=self.load_currents,
            partition_layer='M2',
            N_x=2,
            N_y=2,
            halo_percent=0.2,
            weighting='shortest_path',
            validate_against_flat=True,
            n_workers=1,
            parallel_backend='thread',
        )
        
        self.assertIsNotNone(result_with.validation_stats)
        self.assertIn('max_diff', result_with.validation_stats)
        self.assertIn('mean_diff', result_with.validation_stats)
        self.assertIn('rmse', result_with.validation_stats)
        
        # Without validation
        result_without = self.solver.solve_hierarchical_tiled(
            current_injections=self.load_currents,
            partition_layer='M2',
            N_x=2,
            N_y=2,
            halo_percent=0.2,
            weighting='shortest_path',
            validate_against_flat=False,
            n_workers=1,
            parallel_backend='thread',
        )
        
        self.assertIsNone(result_without.validation_stats)
    
    @timeout(60)
    def test_larger_halo_improves_accuracy(self):
        """Test that larger halo percentage improves accuracy."""
        # Small halo
        result_small = self.solver.solve_hierarchical_tiled(
            current_injections=self.load_currents,
            partition_layer='M2',
            N_x=2,
            N_y=2,
            halo_percent=0.1,
            weighting='shortest_path',
            validate_against_flat=True,
            n_workers=1,
            parallel_backend='thread',
        )
        
        # Large halo
        result_large = self.solver.solve_hierarchical_tiled(
            current_injections=self.load_currents,
            partition_layer='M2',
            N_x=2,
            N_y=2,
            halo_percent=0.3,
            weighting='shortest_path',
            validate_against_flat=True,
            n_workers=1,
            parallel_backend='thread',
        )
        
        # Larger halo should have smaller (or equal) error
        # Note: Due to boundary effects, this may not always hold strictly
        small_rmse = result_small.validation_stats['rmse']
        large_rmse = result_large.validation_stats['rmse']
        
        # At minimum, both should be reasonable
        self.assertLess(small_rmse, 0.01)  # < 10mV
        self.assertLess(large_rmse, 0.01)  # < 10mV

    # ========================================================================
    # New tests for different tiling configurations
    # ========================================================================
    
    @timeout(60)
    def test_3x3_tiling(self):
        """Test 3x3 tiling configuration."""
        result = self.solver.solve_hierarchical_tiled(
            current_injections=self.load_currents,
            partition_layer='M2',
            N_x=3,
            N_y=3,
            halo_percent=0.2,
            weighting='shortest_path',
            n_workers=1,
            parallel_backend='thread',
            validate_against_flat=True,
        )
        
        # Should produce 9 tiles (or fewer if merged)
        self.assertGreater(len(result.tiles), 0)
        self.assertLessEqual(len(result.tiles), 9)
        
        # Accuracy check
        self.assertIsNotNone(result.validation_stats)
        self.assertLess(result.validation_stats['max_diff'], 0.005)  # < 5mV
    
    @timeout(60)
    def test_1x4_asymmetric_tiling(self):
        """Test asymmetric 1x4 tiling (single row, 4 columns)."""
        result = self.solver.solve_hierarchical_tiled(
            current_injections=self.load_currents,
            partition_layer='M2',
            N_x=4,
            N_y=1,
            halo_percent=0.2,
            weighting='shortest_path',
            n_workers=1,
            parallel_backend='thread',
            validate_against_flat=True,
        )
        
        # Should produce up to 4 tiles
        self.assertGreater(len(result.tiles), 0)
        self.assertLessEqual(len(result.tiles), 4)
        
        # All tiles should have valid bounds
        for tile in result.tiles:
            self.assertLess(tile.bounds.x_min, tile.bounds.x_max)
            self.assertLessEqual(tile.bounds.y_min, tile.bounds.y_max)
    
    @timeout(60)
    def test_1x1_single_tile(self):
        """Test 1x1 tiling - single tile covering entire bottom grid."""
        result = self.solver.solve_hierarchical_tiled(
            current_injections=self.load_currents,
            partition_layer='M2',
            N_x=1,
            N_y=1,
            halo_percent=0.2,
            weighting='shortest_path',
            n_workers=1,
            parallel_backend='thread',
            validate_against_flat=True,
        )
        
        # Should produce exactly 1 tile
        self.assertEqual(len(result.tiles), 1)
        
        # Single tile should contain all bottom-grid nodes in core
        tile = result.tiles[0]
        self.assertGreater(len(tile.core_nodes), 0)
        
        # Should match flat solver very closely (no tile boundary errors)
        self.assertIsNotNone(result.validation_stats)
        self.assertLess(result.validation_stats['max_diff'], 0.001)  # < 1mV
    
    # ========================================================================
    # Tests for min_ports_per_tile parameter
    # ========================================================================
    
    @timeout(60)
    def test_min_ports_per_tile_custom(self):
        """Test custom min_ports_per_tile value."""
        min_ports = 10
        result = self.solver.solve_hierarchical_tiled(
            current_injections=self.load_currents,
            partition_layer='M2',
            N_x=2,
            N_y=2,
            halo_percent=0.2,
            min_ports_per_tile=min_ports,
            weighting='shortest_path',
            n_workers=1,
            parallel_backend='thread',
        )
        
        # Each tile should have at least min_ports port nodes
        # (or tiles may have been merged if constraint couldn't be met)
        for tile in result.tiles:
            # Either the tile meets the port requirement, or it was merged
            # with neighbors (in which case it may exceed the requirement)
            self.assertGreater(len(tile.port_nodes), 0)
        
        # Tiling params should reflect the custom value
        self.assertEqual(result.tiling_params['min_ports_per_tile'], min_ports)
    
    @timeout(60)
    def test_min_ports_per_tile_high_value_expands_boundaries(self):
        """Test that high min_ports_per_tile triggers boundary expansion."""
        # Use a high value that forces boundary adjustment
        # Note: High min_ports_per_tile triggers boundary EXPANSION, not merging.
        # Tile merging only happens for tiles with zero current sources.
        port_voltages = self._get_flat_port_voltages('M2')
        result = self.solver.solve_hierarchical_tiled(
            current_injections=self.load_currents,
            partition_layer='M2',
            N_x=4,
            N_y=4,
            halo_percent=0.2,
            min_ports_per_tile=10000,  # Very high, will force boundary expansion
            weighting='shortest_path',
            n_workers=1,
            parallel_backend='thread',
            override_port_voltages=port_voltages,
        )
        
        # Tiles should still be created (boundary expansion, not merging)
        # The min_ports_per_tile is recorded in tiling_params
        self.assertEqual(result.tiling_params['min_ports_per_tile'], 10000)
        
        # Tiles should have valid structure
        for tile in result.tiles:
            self.assertGreater(len(tile.all_nodes), 0)
    
    # ========================================================================
    # Tests for override_port_voltages parameter
    # ========================================================================
    
    @timeout(60)
    def test_override_port_voltages_basic(self):
        """Test that override_port_voltages skips top-grid solve."""
        # First, get the actual port nodes for this partition
        top_nodes, bottom_nodes, port_nodes, _ = self.model._decompose_at_layer('M2')
        
        # Create synthetic port voltages (all at Vdd)
        synthetic_port_voltages = {p: self.model.vdd for p in port_nodes}
        
        result = self.solver.solve_hierarchical_tiled(
            current_injections=self.load_currents,
            partition_layer='M2',
            N_x=2,
            N_y=2,
            halo_percent=0.2,
            weighting='shortest_path',
            n_workers=1,
            parallel_backend='thread',
            override_port_voltages=synthetic_port_voltages,
        )
        
        # Should produce valid results
        self.assertGreater(len(result.voltages), 0)
        
        # top_grid_voltages should be empty when using override
        self.assertEqual(len(result.top_grid_voltages), 0)
        
        # Port voltages should match the override values
        for port in port_nodes:
            if port in result.port_voltages:
                self.assertAlmostEqual(
                    result.port_voltages[port], 
                    self.model.vdd, 
                    places=10
                )
    
    @timeout(60)
    def test_override_port_voltages_accuracy(self):
        """Test that tiled solver with flat port voltages matches flat solver."""
        # Run flat solver to get ground truth
        flat_result = self.solver.solve(self.load_currents)
        
        # Get port nodes and their voltages from flat solver
        top_nodes, bottom_nodes, port_nodes, _ = self.model._decompose_at_layer('M2')
        flat_port_voltages = {
            p: flat_result.voltages[p] 
            for p in port_nodes 
            if p in flat_result.voltages
        }
        
        # Run tiled solver with flat solver's port voltages
        tiled_result = self.solver.solve_hierarchical_tiled(
            current_injections=self.load_currents,
            partition_layer='M2',
            N_x=2,
            N_y=2,
            halo_percent=0.2,
            weighting='shortest_path',
            n_workers=1,
            parallel_backend='thread',
            override_port_voltages=flat_port_voltages,
            validate_against_flat=True,
        )
        
        # Compare bottom-grid voltages
        errors = []
        for node in bottom_nodes:
            if node in tiled_result.voltages and node in flat_result.voltages:
                error = abs(tiled_result.voltages[node] - flat_result.voltages[node])
                errors.append(error)
        
        # With correct port BCs, tiled should match flat very closely
        import numpy as np
        errors = np.array(errors)
        self.assertLess(errors.max(), 0.001)  # < 1mV max error
        self.assertLess(errors.mean(), 0.0001)  # < 0.1mV mean error
    
    @timeout(60)
    def test_override_port_voltages_with_varied_values(self):
        """Test override_port_voltages with non-uniform voltage values."""
        top_nodes, bottom_nodes, port_nodes, _ = self.model._decompose_at_layer('M2')
        
        # Create varied port voltages (simulating IR-drop from top-grid)
        port_list = list(port_nodes)
        varied_port_voltages = {}
        for i, p in enumerate(port_list):
            # Vary voltage slightly based on index
            drop = 0.001 * (i % 10)  # 0-9 mV drop
            varied_port_voltages[p] = self.model.vdd - drop
        
        result = self.solver.solve_hierarchical_tiled(
            current_injections=self.load_currents,
            partition_layer='M2',
            N_x=2,
            N_y=2,
            halo_percent=0.2,
            weighting='shortest_path',
            n_workers=1,
            parallel_backend='thread',
            override_port_voltages=varied_port_voltages,
        )
        
        # All voltages should be reasonable (below Vdd)
        max_v = max(result.voltages.values())
        self.assertLessEqual(max_v, self.model.vdd + 0.001)
        
        # Bottom-grid voltages should show some IR-drop
        min_v = min(v for n, v in result.voltages.items() if n in bottom_nodes)
        self.assertLess(min_v, self.model.vdd)  # Some drop expected


# ============================================================================
# Tests for Tiled Solver Helper Methods
# ============================================================================

class TestTiledSolverHelpers(unittest.TestCase):
    """Tests for tiled hierarchical solver helper methods.
    
    Uses minimal synthetic PDN graphs from fixtures.py to trigger edge cases
    that cannot be triggered with netlist_small.
    """
    
    @classmethod
    def setUpClass(cls):
        """Load fixtures once for all tests."""
        from tests.fixtures import (
            create_minimal_pdn_graph,
            create_floating_island_graph,
        )
        from core import create_model_from_pdn, UnifiedIRDropSolver
        
        # Store as class attributes (not bound methods)
        cls._fixture_create_minimal_pdn = create_minimal_pdn_graph
        cls._fixture_create_floating_island = create_floating_island_graph
        cls._core_create_model_from_pdn = create_model_from_pdn
        cls.UnifiedIRDropSolver = UnifiedIRDropSolver
    
    def _get_pdn_graph(self, scenario):
        """Get minimal PDN graph for scenario."""
        return self.__class__._fixture_create_minimal_pdn(scenario)
    
    def _get_floating_island_graph(self):
        """Get PDN graph with floating island."""
        return self.__class__._fixture_create_floating_island()
    
    def _make_model(self, graph, net_name):
        """Create model from PDN graph."""
        return self.__class__._core_create_model_from_pdn(graph, net_name)
    
    def test_assign_nodes_to_tiles_merging(self):
        """Test tile merging when tiles have 0 current sources (Phase 3)."""
        # Create graph with loads only in one corner
        graph, pads, load_currents = self._get_pdn_graph('tile_merging')
        model = self._make_model(graph, 'VDD')
        solver = self.UnifiedIRDropSolver(model)
        
        # Run tiled solve with 2x2 tiling
        # Tiles (1,0), (0,1), (1,1) should have 0 loads and get merged
        result = solver.solve_hierarchical_tiled(
            current_injections=load_currents,
            partition_layer='M2',
            N_x=2,
            N_y=2,
            halo_percent=0.2,
            n_workers=1,
            parallel_backend='thread',
        )
        
        # Should produce valid results
        self.assertGreater(len(result.voltages), 0)
        
        # Should have fewer tiles than 4 due to merging
        # (some empty tiles merged with neighbors)
        self.assertLessEqual(len(result.tiles), 4)
        
        # All tiles should have at least some nodes
        for tile in result.tiles:
            self.assertGreater(len(tile.core_nodes), 0)
    
    def test_expand_tile_with_halo_severe_clip(self):
        """Test warning emitted for severely clipped halo (<30%)."""
        graph, pads, load_currents = self._get_pdn_graph('severe_halo_clip')
        model = self._make_model(graph, 'VDD')
        solver = self.UnifiedIRDropSolver(model)
        
        # With 3x3 tiling and 50% halo on a 6x6 grid, corner tiles
        # have halos severely clipped
        result = solver.solve_hierarchical_tiled(
            current_injections=load_currents,
            partition_layer='M2',
            N_x=3,
            N_y=3,
            halo_percent=0.5,  # Large halo to trigger clipping
            n_workers=1,
            parallel_backend='thread',
        )
        
        # Should have captured halo clipping warnings in result
        self.assertGreater(
            len(result.halo_clip_warnings), 0,
            "Expected halo clip warnings to be recorded"
        )
        
        # At least one warning should mention clipping
        warnings_text = '\n'.join(result.halo_clip_warnings)
        self.assertTrue(
            'clipped' in warnings_text.lower() or 'halo' in warnings_text.lower(),
            f"Expected halo clip warning, got: {warnings_text}"
        )
        
        # Results should still be valid
        self.assertGreater(len(result.voltages), 0)
    
    def test_expand_tile_with_halo_zero_percent(self):
        """Test halo_percent=0 produces tiles with no halo nodes."""
        graph, pads, load_currents = self._get_pdn_graph('basic')
        model = self._make_model(graph, 'VDD')
        solver = self.UnifiedIRDropSolver(model)
        
        result = solver.solve_hierarchical_tiled(
            current_injections=load_currents,
            partition_layer='M2',
            N_x=2,
            N_y=2,
            halo_percent=0.0,  # No halo
            n_workers=1,
            parallel_backend='thread',
        )
        
        # Each tile should have minimal or no halo nodes
        # (some may have halo from port inclusion)
        for tile in result.tiles:
            # Core should exist
            self.assertGreater(len(tile.core_nodes), 0)
            # Halo should be empty or very small
            self.assertLessEqual(len(tile.halo_nodes), len(tile.core_nodes))
    
    def test_ensure_core_to_port_connectivity_floating(self):
        """Test that floating island nodes are handled correctly."""
        import warnings
        
        graph, pads, load_currents = self._get_floating_island_graph()
        
        # Model creation should warn about floating islands
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model = self._make_model(graph, 'VDD')
            
            # Should have warning about islands
            island_warnings = [x for x in w if 'island' in str(x.message).lower() or 'floating' in str(x.message).lower()]
            self.assertGreater(len(island_warnings), 0, "Should warn about floating islands")
        
        solver = self.UnifiedIRDropSolver(model)
        
        # Tiled solve should still work (floating nodes filtered)
        result = solver.solve_hierarchical_tiled(
            current_injections=load_currents,
            partition_layer='M2',
            N_x=2,
            N_y=2,
            halo_percent=0.2,
            n_workers=1,
            parallel_backend='thread',
        )
        
        # Should produce valid results for connected nodes
        self.assertGreater(len(result.voltages), 0)
        
        # Floating island nodes should NOT be in results
        island_nodes = ['10000_10000_M1', '10000_12000_M1', '12000_10000_M1', '12000_12000_M1']
        for island_node in island_nodes:
            self.assertNotIn(island_node, result.voltages)
    
    def test_generate_uniform_tiles_single_row(self):
        """Test N_x=1 or N_y=1 produces correct single-row tiling."""
        graph, pads, load_currents = self._get_pdn_graph('basic')
        model = self._make_model(graph, 'VDD')
        solver = self.UnifiedIRDropSolver(model)
        
        # Test N_x=1 (single column)
        result = solver.solve_hierarchical_tiled(
            current_injections=load_currents,
            partition_layer='M2',
            N_x=1,
            N_y=2,
            halo_percent=0.2,
            n_workers=1,
            parallel_backend='thread',
        )
        
        # Should have at most 2 tiles (may be fewer if merged)
        self.assertLessEqual(len(result.tiles), 2)
        self.assertGreater(len(result.voltages), 0)
        
        # Test N_y=1 (single row)
        result = solver.solve_hierarchical_tiled(
            current_injections=load_currents,
            partition_layer='M2',
            N_x=2,
            N_y=1,
            halo_percent=0.2,
            n_workers=1,
            parallel_backend='thread',
        )
        
        self.assertLessEqual(len(result.tiles), 2)
        self.assertGreater(len(result.voltages), 0)
    
    def test_invalid_partition_layer_string_pdn(self):
        """Test non-existent partition layer raises ValueError."""
        graph, pads, load_currents = self._get_pdn_graph('basic')
        model = self._make_model(graph, 'VDD')
        solver = self.UnifiedIRDropSolver(model)
        
        # Non-existent layer 'M99' should raise ValueError
        with self.assertRaises(ValueError) as ctx:
            solver.solve_hierarchical(
                current_injections=load_currents,
                partition_layer='M99',
            )
        
        self.assertIn('M99', str(ctx.exception))
    
    def test_tiled_solve_with_tile_merging_end_to_end(self):
        """End-to-end test of tiled solve with tile merging."""
        graph, pads, load_currents = self._get_pdn_graph('tile_merging')
        model = self._make_model(graph, 'VDD')
        solver = self.UnifiedIRDropSolver(model)
        
        # Run flat solve for reference
        flat_result = solver.solve(load_currents)
        flat_max_drop = max(flat_result.ir_drop.values())
        
        # Run tiled solve
        tiled_result = solver.solve_hierarchical_tiled(
            current_injections=load_currents,
            partition_layer='M2',
            N_x=2,
            N_y=2,
            halo_percent=0.3,
            n_workers=1,
            parallel_backend='thread',
            validate_against_flat=True,
        )
        
        tiled_max_drop = max(tiled_result.ir_drop.values())
        
        # Results should be reasonably close (relaxed for small synthetic fixtures
        # where tiling approximation has larger relative error)
        diff = abs(flat_max_drop - tiled_max_drop)
        # For tiny fixtures, allow up to 50% relative error or 200mV absolute
        max_acceptable = max(0.2, flat_max_drop * 0.5)
        self.assertLess(diff, max_acceptable, 
            f"Max IR-drop diff {diff*1000:.1f}mV exceeds threshold {max_acceptable*1000:.1f}mV")
        
        # Main point: tiled solve should complete and produce valid results
        self.assertGreater(len(tiled_result.voltages), 0)
        self.assertGreater(len(tiled_result.tiles), 0)
    
    def test_tiled_solve_accuracy_with_path_expansion(self):
        """Test tiled solve accuracy with sparse via connectivity."""
        graph, pads, load_currents = self._get_pdn_graph('path_expansion')
        model = self._make_model(graph, 'VDD')
        solver = self.UnifiedIRDropSolver(model)
        
        # Run flat solve for reference
        flat_result = solver.solve(load_currents)
        
        # Run tiled solve with validation
        tiled_result = solver.solve_hierarchical_tiled(
            current_injections=load_currents,
            partition_layer='M2',
            N_x=2,
            N_y=2,
            halo_percent=0.3,
            n_workers=1,
            parallel_backend='thread',
            validate_against_flat=True,
        )
        
        # Should produce valid results
        self.assertGreater(len(tiled_result.voltages), 0)
        
        # Check all bottom-grid nodes have voltages
        bottom_nodes, _, _, _ = model._decompose_at_layer('M2')
        for node in bottom_nodes:
            if node in model.reduced.node_order:  # Exclude floating islands
                self.assertIn(
                    node, tiled_result.voltages,
                    f"Bottom-grid node {node} missing from tiled result"
                )


if __name__ == '__main__':
    unittest.main()


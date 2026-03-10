"""Test floating nodes consistency between flat and distributed modes.

This test verifies that flat and distributed solvers produce identical floating
node reports for the same netlist, ensuring that the two modes identify the same
disconnected islands.

The test requires pre-generated distributed pkl files. To generate them:
    python -m distributed parse ./netlist/netlist_sampled --net VDD_XLV \\
        -o ./netlist/netlist_sampled/distributed_pkl

Tests will skip if the required test data is not available.
"""
import pytest
from pathlib import Path

# Skip if netlist_sampled doesn't exist or distributed_pkl isn't available
NETLIST_DIR = Path('./netlist/netlist_sampled')
DISTRIBUTED_PKL_DIR = NETLIST_DIR / 'distributed_pkl'

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not NETLIST_DIR.exists() or not DISTRIBUTED_PKL_DIR.exists(),
        reason="netlist_sampled test data not available"
    ),
]


class TestFlatVsDistributedConsistency:
    """Verify flat and distributed modes identify same floating nodes."""

    def test_floating_nodes_match(self):
        """Both modes should identify identical floating nodes."""
        # 1. Run flat mode and collect floating nodes
        flat_floating = self._collect_flat_floating_nodes()

        # 2. Run distributed mode and collect floating nodes
        distributed_floating = self._collect_distributed_floating_nodes()

        # 3. Compare sets of floating nodes
        # The union of all distributed floating nodes should equal flat floating nodes
        assert flat_floating == distributed_floating, (
            f"Floating node mismatch!\n"
            f"Flat-only: {flat_floating - distributed_floating}\n"
            f"Distributed-only: {distributed_floating - flat_floating}"
        )

    def test_per_layer_counts_match(self):
        """Per-layer floating counts should match."""
        flat_data = self._collect_flat_data()
        distributed_data = self._collect_distributed_data()

        assert flat_data.layer_floating_counts == distributed_data.layer_floating_counts

    def test_total_counts_match(self):
        """Total floating node counts should match."""
        flat_data = self._collect_flat_data()
        distributed_data = self._collect_distributed_data()

        assert flat_data.total_floating == distributed_data.total_floating

    def test_total_nodes_match(self):
        """Total node counts should match exactly between flat and distributed modes."""
        flat_data = self._collect_flat_data()
        distributed_data = self._collect_distributed_data()

        assert flat_data.total_nodes == distributed_data.total_nodes, (
            f"Total node count mismatch!\n"
            f"Flat: {flat_data.total_nodes:,}\n"
            f"Distributed: {distributed_data.total_nodes:,}\n"
            f"Difference: {flat_data.total_nodes - distributed_data.total_nodes}"
        )

    def test_per_layer_node_sets_match(self):
        """Per-layer floating node sets should match (order may differ)."""
        flat_data = self._collect_flat_data()
        distributed_data = self._collect_distributed_data()

        # Compare all layers present in either dataset
        all_layers = set(flat_data.layer_floating_nodes.keys()) | \
                     set(distributed_data.layer_floating_nodes.keys())

        for layer in all_layers:
            flat_nodes = set(flat_data.layer_floating_nodes.get(layer, []))
            dist_nodes = set(distributed_data.layer_floating_nodes.get(layer, []))

            assert flat_nodes == dist_nodes, (
                f"Layer {layer} floating nodes mismatch!\n"
                f"Flat-only: {flat_nodes - dist_nodes}\n"
                f"Distributed-only: {dist_nodes - flat_nodes}"
            )

    def _collect_flat_floating_nodes(self) -> set:
        """Run flat solver and return set of floating nodes."""
        from parser.netlist import NetlistParser
        from model.factory import create_model_from_pdn

        parser = NetlistParser(str(NETLIST_DIR), validate=False)
        graph = parser.parse()
        model = create_model_from_pdn(graph, 'VDD_XLV')

        return model.removed_island_nodes

    def _collect_flat_data(self):
        """Run flat solver and return FloatingNodesData."""
        from parser.netlist import NetlistParser
        from model.factory import create_model_from_pdn
        from reports.floating_nodes import collect_floating_nodes_flat

        parser = NetlistParser(str(NETLIST_DIR), validate=False)
        graph = parser.parse()
        model = create_model_from_pdn(graph, 'VDD_XLV')

        return collect_floating_nodes_flat(model, net_name='VDD_XLV')

    def _collect_distributed_floating_nodes(self) -> set:
        """Run distributed solver and return set of floating nodes."""
        from distributed.model import create_distributed_model, load_distributed_partitions
        from distributed.solver import DistributedDDMSolver

        bundle = load_distributed_partitions(str(DISTRIBUTED_PKL_DIR))
        model = create_distributed_model(bundle, backend='local')

        try:
            solver = DistributedDDMSolver(model)
            ctx = solver.prepare(verbose=False)

            # Collect all floating nodes from tiles
            tile_data = model.backend.call_all(model.workers, 'get_floating_nodes_data')
            all_floating = set()
            for td in tile_data:
                all_floating.update(td['removed_nodes'])

            # Add interface islands
            if ctx.removed_interface_nodes:
                all_floating.update(ctx.removed_interface_nodes)

            return all_floating
        finally:
            model.shutdown()

    def _collect_distributed_data(self):
        """Run distributed solver and return FloatingNodesData."""
        from distributed.model import create_distributed_model, load_distributed_partitions
        from distributed.solver import DistributedDDMSolver
        from reports.floating_nodes import collect_floating_nodes_distributed

        bundle = load_distributed_partitions(str(DISTRIBUTED_PKL_DIR))
        model = create_distributed_model(bundle, backend='local')

        try:
            solver = DistributedDDMSolver(model)
            ctx = solver.prepare(verbose=False)

            return collect_floating_nodes_distributed(
                model, ctx, model.workers, net_name=model.net_name
            )
        finally:
            model.shutdown()

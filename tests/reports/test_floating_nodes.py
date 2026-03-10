"""Tests for floating nodes report generation."""

import tempfile
from pathlib import Path

import pytest

from reports.floating_nodes import (
    FloatingNodesData,
    _group_nodes_by_layer,
    _sort_layer_key,
    collect_floating_nodes_flat,
    generate_floating_nodes_report,
)

pytestmark = pytest.mark.unit


class TestLayerSorting:
    """Test layer sorting utilities."""

    def test_sort_layer_key_numeric(self):
        """Test sorting of numeric layers.

        Sorting order: all plain digits by numeric value,
        then all M-prefixed layers by numeric value.
        """
        layers = ['M10', 'M2', 'M1', '3', '1', 'M3']
        sorted_layers = sorted(layers, key=_sort_layer_key)
        # Plain digits first: 1, M1, M2, 3, M3, M10
        # Wait, that's not right either. Let me think...
        # With key (0, num, prefix, str):
        # '1' -> (0, 1, '', '')
        # 'M1' -> (0, 1, 'M', '')
        # So '1' < 'M1' because '' < 'M'
        # '3' -> (0, 3, '', '')
        # 'M2' -> (0, 2, 'M', '')
        # So we get interleaved: 1, M1, M2, 3, M3, M10
        assert sorted_layers == ['1', 'M1', 'M2', '3', 'M3', 'M10']

    def test_sort_layer_key_mixed(self):
        """Test sorting of mixed numeric and alphabetic layers."""
        layers = ['package', 'M5', 'VIA23', 'M1', 'AP']
        sorted_layers = sorted(layers, key=_sort_layer_key)
        # Numeric layers first, then alphabetic
        assert sorted_layers[:2] == ['M1', 'M5']
        # Alphabetic layers last, in alphabetic order
        assert set(sorted_layers[2:]) == {'AP', 'VIA23', 'package'}


class TestGroupNodesByLayer:
    """Test node grouping and sorting."""

    def test_group_nodes_by_layer_simple(self):
        """Test basic node grouping."""
        nodes = {
            '1000_2000_M1',
            '1000_2100_M1',
            '1050_2000_M1',
            '2000_3000_M2',
        }
        grouped = _group_nodes_by_layer(nodes)

        assert set(grouped.keys()) == {'M1', 'M2'}
        assert len(grouped['M1']) == 3
        assert len(grouped['M2']) == 1

    def test_group_nodes_sorted_by_xy(self):
        """Test that nodes are sorted by X then Y."""
        nodes = {
            '1050_2000_M1',
            '1000_2100_M1',
            '1000_2000_M1',
        }
        grouped = _group_nodes_by_layer(nodes)

        # Should be sorted by X first (1000, 1000, 1050), then Y
        assert grouped['M1'] == [
            '1000_2000_M1',
            '1000_2100_M1',
            '1050_2000_M1',
        ]

    def test_group_nodes_unparseable_ignored(self):
        """Test that unparseable nodes are skipped."""
        nodes = {
            '1000_2000_M1',
            'invalid_node',
            'VDD_vsrc',
            '0',
        }
        grouped = _group_nodes_by_layer(nodes)

        # Only the valid node should be grouped
        assert list(grouped.keys()) == ['M1']
        assert grouped['M1'] == ['1000_2000_M1']


class TestFloatingNodesData:
    """Test FloatingNodesData dataclass."""

    def test_dataclass_defaults(self):
        """Test default initialization."""
        data = FloatingNodesData()
        assert data.total_floating == 0
        assert data.total_nodes == 0
        assert len(data.layer_floating_counts) == 0
        assert len(data.dropped_connectivity) == 0
        assert data.net_name == ''

    def test_dataclass_with_values(self):
        """Test initialization with values."""
        data = FloatingNodesData(
            layer_floating_counts={'M1': 10, 'M2': 5},
            layer_total_counts={'M1': 100, 'M2': 50},
            total_floating=15,
            total_nodes=150,
            net_name='VDD',
        )
        assert data.total_floating == 15
        assert data.total_nodes == 150
        assert data.layer_floating_counts['M1'] == 10
        assert data.net_name == 'VDD'


class TestCollectFloatingNodesDistributed:
    """Test distributed mode floating node collection."""

    def test_collect_with_error_recovery(self):
        """Test error handling when worker call fails."""
        from unittest.mock import Mock
        import warnings

        # Create mock model and context
        model = Mock()
        context = Mock()
        workers = [Mock(), Mock()]

        # Mock model attributes
        model.tile_interior_counts = {(0, 0): 100, (0, 1): 150}
        model.interface_nodes = {'iface1', 'iface2', 'iface3'}
        model.tile_boundary_nodes = {(0, 0): {'iface1', 'iface2'}, (0, 1): {'iface2', 'iface3'}}
        context.removed_interface_nodes = {'iface1'}

        # Mock package_data
        model.package_data.tap_nodes = set()  # No package tap nodes in this mock
        model.package_data.pad_nodes = set()  # No package pad nodes in this mock
        model.package_data.die_attachment_nodes = set()

        # Mock backend.call_all to raise an exception
        model.backend.call_all.side_effect = RuntimeError("Worker communication failed")

        # Import after mock setup
        from reports.floating_nodes import collect_floating_nodes_distributed

        # Should handle exception gracefully and issue warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            data = collect_floating_nodes_distributed(model, context, workers)

            # Verify warning was issued
            assert len(w) == 1
            assert "Failed to collect floating nodes data from workers" in str(w[0].message)
            assert "may be incomplete" in str(w[0].message)

        # Should still return valid data (with just interface islands)
        assert data.total_floating == 1  # Only interface island
        # total_nodes = interior (250) + interface (3) + dropped_unique (0) = 253
        # dropped_connectivity is empty (worker failed), so dropped_unique is also empty
        assert data.total_nodes == 253

    def test_collect_with_success(self):
        """Test successful collection from workers."""
        from unittest.mock import Mock

        # Create mock model and context
        model = Mock()
        context = Mock()
        workers = [Mock(), Mock()]

        # Mock model attributes
        model.tile_interior_counts = {(0, 0): 100, (0, 1): 150}
        model.interface_nodes = {'1000_2000_M1', '1000_2100_M1', '2000_3000_M2'}
        model.tile_boundary_nodes = {(0, 0): {'1000_2000_M1', '1000_2100_M1'}, (0, 1): {'1000_2100_M1', '2000_3000_M2'}}
        context.removed_interface_nodes = {'1000_2000_M1'}

        # Mock package_data
        model.package_data.tap_nodes = set()  # No package tap nodes in this mock
        model.package_data.pad_nodes = set()  # No package pad nodes in this mock
        model.package_data.die_attachment_nodes = set()

        # Mock successful worker responses
        model.backend.call_all.return_value = [
            {'removed_nodes': {'1000_2100_M1'}},
            {'removed_nodes': {'2000_3000_M2'}},
        ]

        from reports.floating_nodes import collect_floating_nodes_distributed

        data = collect_floating_nodes_distributed(model, context, workers)

        # Should aggregate all floating nodes
        assert data.total_floating == 3  # 1 interface + 2 tile nodes
        # total_nodes = interior (250) + interface (3) + dropped_unique (0) = 253
        # dropped_connectivity = {'1000_2100_M1', '2000_3000_M2'}, both are in
        # interface_nodes, so dropped_unique is empty (no double-counting)
        assert data.total_nodes == 253
        assert len(data.dropped_connectivity) == 2
        assert len(data.disconnected_interface) == 1
        assert data.layer_total_counts == {}  # Empty in distributed mode

    def test_collect_with_mixed_dropped_nodes(self):
        """Test that dropped nodes overlapping interface are not double-counted."""
        from unittest.mock import Mock

        from reports.floating_nodes import collect_floating_nodes_distributed

        model = Mock()
        context = Mock()
        workers = [Mock()]

        # Interface has 3 nodes; tile drops 2 interior + 1 interface node
        model.interface_nodes = {'1000_2000_M1', '1000_2100_M1', '2000_3000_M2'}
        model.tile_interior_counts = {(0, 0): 100}  # POST-removal
        model.package_data.tap_nodes = {'3000_4000_M13'}
        model.package_data.pad_nodes = {'VDD_vsrc'}
        context.removed_interface_nodes = set()  # No global interface islands

        model.backend.call_all.return_value = [
            {'removed_nodes': {'5000_6000_M1', '5000_6100_M1', '2000_3000_M2'}},
        ]

        data = collect_floating_nodes_distributed(model, context, workers)

        # dropped_connectivity = {'5000_6000_M1', '5000_6100_M1', '2000_3000_M2'}
        # dropped_unique = dropped_connectivity - interface_nodes = {'5000_6000_M1', '5000_6100_M1'}
        # total = 100 (interior) + 3 (interface) + 2 (dropped_unique) + 1 (tap) + 1 (pad) = 107
        assert data.total_nodes == 107
        assert data.total_floating == 3  # all three dropped nodes are floating
        assert len(data.dropped_connectivity) == 3


class TestCollectFloatingNodesFlat:
    """Test flat mode floating node collection."""

    def test_collect_with_mock_model(self):
        """Test collection with a mock model."""
        from unittest.mock import Mock, patch

        # Create mock model
        model = Mock()

        # Mock floating and valid nodes
        floating_nodes = {
            '1000_2000_M1',
            '1000_2100_M1',
            '2000_3000_M2',
        }
        valid_nodes = {
            '1000_2200_M1',
            '1100_2000_M1',
            '2000_3100_M2',
            '3000_4000_M3',
        }

        model.removed_island_nodes = floating_nodes
        model.valid_nodes = valid_nodes

        # Patch the isinstance check to accept our mock
        with patch('reports.floating_nodes.isinstance', return_value=True):
            data = collect_floating_nodes_flat(model)

            # Verify totals
            assert data.total_floating == 3
            assert data.total_nodes == 7

            # Verify layer counts
            assert data.layer_floating_counts['M1'] == 2
            assert data.layer_floating_counts['M2'] == 1
            assert 'M3' not in data.layer_floating_counts  # No floating nodes on M3

            # Verify categorization (flat mode)
            assert len(data.dropped_connectivity) == 3
            assert len(data.disconnected_interface) == 0
            assert len(data.connected_to_floating) == 0


class TestGenerateFloatingNodesReport:
    """Test report generation."""

    def test_generate_report_no_floating_nodes(self):
        """Test report generation with no floating nodes."""
        data = FloatingNodesData(
            total_floating=0,
            total_nodes=100,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            generate_floating_nodes_report(data, output_dir, verbose=False)

            # Check report file exists
            report_file = output_dir / 'floating_nodes_report.md'
            assert report_file.exists()

            content = report_file.read_text()
            assert 'No floating nodes detected' in content
            # Should not have net name in header when net_name is empty
            assert '# Floating Nodes Report\n' in content
            assert 'Net:' not in content.split('\n')[0]

    def test_generate_report_with_floating_nodes(self):
        """Test report generation with floating nodes."""
        data = FloatingNodesData(
            layer_floating_counts={'M1': 2, 'M2': 1},
            layer_total_counts={'M1': 10, 'M2': 10, 'M3': 10},
            layer_floating_nodes={
                'M1': ['1000_2000_M1', '1000_2100_M1'],
                'M2': ['2000_3000_M2'],
            },
            dropped_connectivity={'1000_2000_M1', '1000_2100_M1', '2000_3000_M2'},
            total_floating=3,
            total_nodes=30,
            net_name='VDD',
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            generate_floating_nodes_report(data, output_dir, verbose=False)

            # Check report file exists
            report_file = output_dir / 'floating_nodes_report.md'
            assert report_file.exists()

            content = report_file.read_text()

            # Check header contains net name
            assert '# Floating Nodes Report — Net: VDD\n' in content

            # Check summary section
            assert 'Total floating nodes: 3' in content
            assert '10.00%' in content  # 3/30 = 10%
            assert 'Layers affected: 2 of 3' in content

            # Check per-layer breakdown
            assert 'Per-Layer Breakdown' in content
            assert 'M1' in content
            assert 'M2' in content

            # Check category breakdown
            assert 'Category Breakdown' in content
            assert 'Dropped during connectivity tracing' in content

    def test_generate_report_verbose_mode(self):
        """Test verbose mode generates per-layer files."""
        data = FloatingNodesData(
            layer_floating_counts={'M1': 2},
            layer_total_counts={'M1': 10},
            layer_floating_nodes={
                'M1': ['1000_2000_M1', '1000_2100_M1'],
            },
            dropped_connectivity={'1000_2000_M1', '1000_2100_M1'},
            total_floating=2,
            total_nodes=10,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            generate_floating_nodes_report(data, output_dir, verbose=True)

            # Check per-layer file exists (without net name in filename)
            layer_file = output_dir / 'floating_nodes_M1.txt'
            assert layer_file.exists()

            content = layer_file.read_text()
            lines = content.strip().split('\n')
            assert len(lines) == 2
            assert '1000_2000_M1' in lines
            assert '1000_2100_M1' in lines

    def test_generate_report_verbose_mode_with_net_name(self):
        """Test verbose mode generates per-layer files with net name."""
        data = FloatingNodesData(
            layer_floating_counts={'M1': 2, 'M2': 1},
            layer_total_counts={'M1': 10, 'M2': 10},
            layer_floating_nodes={
                'M1': ['1000_2000_M1', '1000_2100_M1'],
                'M2': ['2000_3000_M2'],
            },
            dropped_connectivity={'1000_2000_M1', '1000_2100_M1', '2000_3000_M2'},
            total_floating=3,
            total_nodes=20,
            net_name='VDD_XLV',
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            generate_floating_nodes_report(data, output_dir, verbose=True)

            # Check per-layer files exist with net name prefix
            layer_file_m1 = output_dir / 'floating_nodes_vdd_xlv_M1.txt'
            layer_file_m2 = output_dir / 'floating_nodes_vdd_xlv_M2.txt'
            assert layer_file_m1.exists()
            assert layer_file_m2.exists()

            content_m1 = layer_file_m1.read_text()
            lines_m1 = content_m1.strip().split('\n')
            assert len(lines_m1) == 2
            assert '1000_2000_M1' in lines_m1
            assert '1000_2100_M1' in lines_m1

            content_m2 = layer_file_m2.read_text()
            lines_m2 = content_m2.strip().split('\n')
            assert len(lines_m2) == 1
            assert '2000_3000_M2' in lines_m2

    def test_generate_report_category_filtering(self):
        """Test that only non-zero categories are shown."""
        data = FloatingNodesData(
            layer_floating_counts={'M1': 2},
            layer_total_counts={'M1': 10},
            dropped_connectivity={'1000_2000_M1', '1000_2100_M1'},
            disconnected_interface=set(),  # Empty
            connected_to_floating=set(),  # Empty
            total_floating=2,
            total_nodes=10,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            generate_floating_nodes_report(data, output_dir, verbose=False)

            content = (output_dir / 'floating_nodes_report.md').read_text()

            # Should only show the non-zero category
            assert 'Dropped during connectivity tracing' in content
            assert 'Disconnected interface nodes' not in content
            assert 'Connected to floating interface' not in content

    def test_generate_report_no_layer_totals(self):
        """Test report generation without per-layer totals (distributed mode)."""
        data = FloatingNodesData(
            layer_floating_counts={'M1': 2, 'M2': 1},
            layer_total_counts={},  # Empty in distributed mode
            layer_floating_nodes={
                'M1': ['1000_2000_M1', '1000_2100_M1'],
                'M2': ['2000_3000_M2'],
            },
            dropped_connectivity={'1000_2000_M1', '1000_2100_M1'},
            disconnected_interface={'2000_3000_M2'},
            total_floating=3,
            total_nodes=300,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            generate_floating_nodes_report(data, output_dir, verbose=False)

            content = (output_dir / 'floating_nodes_report.md').read_text()

            # Check summary section doesn't say "of N" for layers
            assert 'Total floating nodes: 3' in content
            assert 'Layers affected: 2\n' in content  # No "of N"

            # Check per-layer breakdown table format (no Total/Percentage columns)
            assert 'Per-Layer Breakdown' in content

            # Find the per-layer table by extracting lines after the header
            lines = content.split('\n')
            per_layer_idx = lines.index('## Per-Layer Breakdown')
            per_layer_section = '\n'.join(lines[per_layer_idx:per_layer_idx+10])

            # Verify table has only 2 columns: Layer and Floating
            assert '| Layer | Floating |' in per_layer_section
            assert '|-------|----------|' in per_layer_section
            # The per-layer section should NOT have "Total" or "Percentage" columns
            # (Category breakdown will have Percentage, but that's different)
            assert '| Total |' not in per_layer_section
            assert '| M1    |        2 |' in content
            assert '| M2    |        1 |' in content

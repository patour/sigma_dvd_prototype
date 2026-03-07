"""Tests for top-K worst IR-drop report generation."""

import os
import shutil
import tempfile

import pytest

from reports.topk_irdrop import generate_topk_report


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _parse_report_file(path):
    """Parse a topk_irdrop report file and return header lines + data rows.

    Returns:
        (header_lines, data_rows) where header_lines is the raw text before
        the second '=' separator and data_rows is a list of whitespace-split
        token lists (one per data row).
    """
    with open(path, 'r') as f:
        lines = f.read().splitlines()

    # Find the two '===' separator lines dynamically (supports extra_header_lines)
    sep_indices = [i for i, line in enumerate(lines) if line.startswith('===')]
    assert len(sep_indices) >= 2, f"Expected 2 separator lines, found {len(sep_indices)}"
    header_lines = lines[:sep_indices[1] + 1]
    data_rows = []
    for line in lines[sep_indices[1] + 1:]:
        stripped = line.strip()
        if stripped:
            data_rows.append(stripped.split())
    return header_lines, data_rows


# ──────────────────────────────────────────────────────────────────────
# Test data
# ──────────────────────────────────────────────────────────────────────

VOLTAGES_BASIC = {
    '1000_2000_M1': 0.990,   # drop = 10mV
    '1000_2100_M1': 0.985,   # drop = 15mV
    '2000_3000_M2': 0.970,   # drop = 30mV (worst)
    '3000_4000_M3': 0.980,   # drop = 20mV
    '4000_5000_M1': 0.995,   # drop = 5mV
    'VDD_vsrc': 1.0,         # pad node
    '0': 0.0,                # ground
}

PAD_NODES = {'VDD_vsrc'}


class TestTopKReport:
    """Tests for generate_topk_report()."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp(prefix='topk_test_')

    def teardown_method(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    # ------------------------------------------------------------------
    # 1. Happy path
    # ------------------------------------------------------------------
    def test_happy_path_output_exists_and_sorted(self):
        """Report file exists, has correct header, data rows sorted by IR-drop descending."""
        generate_topk_report(
            voltages=VOLTAGES_BASIC,
            nominal_voltage=1.0,
            net_name='VDD',
            pad_nodes=PAD_NODES,
            output_dir=self.tmpdir,
            top_k=100,
        )

        report_path = os.path.join(self.tmpdir, 'topk_irdrop_VDD.txt')
        assert os.path.isfile(report_path)

        header, rows = _parse_report_file(report_path)

        # Header checks
        assert 'Top-100' in header[0]
        assert 'VDD' in header[1]
        assert '1.000000' in header[2]

        # 5 non-pad, non-ground nodes
        assert len(rows) == 5

        # Rows sorted by IR-drop descending (column index 6 = Drop(mV))
        drops_mv = [float(r[6]) for r in rows]
        assert drops_mv == sorted(drops_mv, reverse=True)

        # Verify exact ordering: 30mV, 20mV, 15mV, 10mV, 5mV
        assert drops_mv == pytest.approx([30.0, 20.0, 15.0, 10.0, 5.0], abs=0.01)

        # Rank column should be 1..5
        ranks = [int(r[0]) for r in rows]
        assert ranks == [1, 2, 3, 4, 5]

    # ------------------------------------------------------------------
    # 2. Pad and ground exclusion
    # ------------------------------------------------------------------
    def test_pad_and_ground_excluded(self):
        """Pad nodes and ground '0' must not appear in the report."""
        generate_topk_report(
            voltages=VOLTAGES_BASIC,
            nominal_voltage=1.0,
            net_name='VDD',
            pad_nodes=PAD_NODES,
            output_dir=self.tmpdir,
            top_k=100,
        )

        report_path = os.path.join(self.tmpdir, 'topk_irdrop_VDD.txt')
        _, rows = _parse_report_file(report_path)

        all_nodes_in_report = {r[1] for r in rows}
        assert 'VDD_vsrc' not in all_nodes_in_report
        assert '0' not in all_nodes_in_report

    # ------------------------------------------------------------------
    # 3. Instance names
    # ------------------------------------------------------------------
    def test_instance_names_present_and_na(self):
        """Nodes with instance mapping show instance name; others show N/A."""
        node_to_instance = {
            '2000_3000_M2': 'I_inst_worst',
            '1000_2100_M1': 'I_inst_mid',
        }

        generate_topk_report(
            voltages=VOLTAGES_BASIC,
            nominal_voltage=1.0,
            net_name='VDD',
            pad_nodes=PAD_NODES,
            output_dir=self.tmpdir,
            top_k=100,
            node_to_instance=node_to_instance,
        )

        report_path = os.path.join(self.tmpdir, 'topk_irdrop_VDD.txt')
        with open(report_path, 'r') as f:
            content = f.read()

        # Mapped nodes should have their instance name
        assert 'I_inst_worst' in content
        assert 'I_inst_mid' in content

        # Non-mapped nodes should show N/A.  Read data rows and check.
        _, rows = _parse_report_file(report_path)
        # Last column is Instance
        instances = {r[1]: r[8] for r in rows}
        assert instances['2000_3000_M2'] == 'I_inst_worst'
        assert instances['1000_2100_M1'] == 'I_inst_mid'
        assert instances['1000_2000_M1'] == 'N/A'
        assert instances['3000_4000_M3'] == 'N/A'
        assert instances['4000_5000_M1'] == 'N/A'

    # ------------------------------------------------------------------
    # 4. Empty voltages
    # ------------------------------------------------------------------
    def test_empty_voltages_header_only(self):
        """Empty voltages dict should produce a file with headers but no data rows."""
        generate_topk_report(
            voltages={},
            nominal_voltage=1.0,
            net_name='VDD',
            pad_nodes=set(),
            output_dir=self.tmpdir,
            top_k=100,
        )

        report_path = os.path.join(self.tmpdir, 'topk_irdrop_VDD.txt')
        assert os.path.isfile(report_path)

        _, rows = _parse_report_file(report_path)
        assert len(rows) == 0

    # ------------------------------------------------------------------
    # 5. Zero nominal voltage
    # ------------------------------------------------------------------
    def test_zero_nominal_voltage_no_divide_by_zero(self):
        """With nominal_voltage=0, drop_pct should be 0.0 (no ZeroDivisionError)."""
        voltages = {'1000_2000_M1': 0.5, '2000_3000_M2': 0.3}

        generate_topk_report(
            voltages=voltages,
            nominal_voltage=0.0,
            net_name='VSS',
            pad_nodes=set(),
            output_dir=self.tmpdir,
            top_k=100,
        )

        report_path = os.path.join(self.tmpdir, 'topk_irdrop_VSS.txt')
        assert os.path.isfile(report_path)

        _, rows = _parse_report_file(report_path)
        assert len(rows) == 2

        # Drop(%) column (index 7) should be 0.00
        for row in rows:
            assert float(row[7]) == 0.0

    # ------------------------------------------------------------------
    # 6. Top-K limiting
    # ------------------------------------------------------------------
    def test_topk_limits_output_rows(self):
        """Report must contain at most top_k data rows when more nodes are available."""
        # Build 20 nodes
        voltages = {
            f'{1000 + i}_2000_M1': 1.0 - i * 0.001
            for i in range(20)
        }

        generate_topk_report(
            voltages=voltages,
            nominal_voltage=1.0,
            net_name='VDD',
            pad_nodes=set(),
            output_dir=self.tmpdir,
            top_k=5,
        )

        report_path = os.path.join(self.tmpdir, 'topk_irdrop_VDD.txt')
        _, rows = _parse_report_file(report_path)
        assert len(rows) == 5

        # The 5 rows should be the 5 worst (highest index = most drop)
        drops_mv = [float(r[6]) for r in rows]
        assert drops_mv[0] > drops_mv[-1]
        # Worst node: i=19 -> drop=19mV
        assert drops_mv[0] == pytest.approx(19.0, abs=0.01)

    # ------------------------------------------------------------------
    # 7. Unparseable node names
    # ------------------------------------------------------------------
    def test_unparseable_node_names_show_na(self):
        """Nodes that don't match X_Y_LAYER should show N/A for layer/x/y columns."""
        voltages = {
            'weirdnode': 0.95,
            'another_node': 0.90,
            'VDD_bump_pkg': 0.92,
            '1000_2000_M1': 0.98,   # This one IS parseable
        }

        generate_topk_report(
            voltages=voltages,
            nominal_voltage=1.0,
            net_name='VDD',
            pad_nodes=set(),
            output_dir=self.tmpdir,
            top_k=100,
        )

        report_path = os.path.join(self.tmpdir, 'topk_irdrop_VDD.txt')
        _, rows = _parse_report_file(report_path)
        assert len(rows) == 4

        # Build lookup by node name
        by_node = {r[1]: r for r in rows}

        # Parseable node should have real layer/x/y
        assert by_node['1000_2000_M1'][2] == 'M1'      # layer
        assert by_node['1000_2000_M1'][3] == '1000.0'   # x
        assert by_node['1000_2000_M1'][4] == '2000.0'   # y

        # Unparseable nodes should have N/A
        assert by_node['weirdnode'][2] == 'N/A'
        assert by_node['weirdnode'][3] == 'N/A'
        assert by_node['weirdnode'][4] == 'N/A'

        assert by_node['another_node'][2] == 'N/A'

    # ------------------------------------------------------------------
    # Additional edge cases
    # ------------------------------------------------------------------
    def test_output_dir_created_if_missing(self):
        """Output directory is created automatically if it does not exist."""
        nested_dir = os.path.join(self.tmpdir, 'sub', 'dir')
        assert not os.path.exists(nested_dir)

        generate_topk_report(
            voltages={'1000_2000_M1': 0.99},
            nominal_voltage=1.0,
            net_name='VDD',
            pad_nodes=set(),
            output_dir=nested_dir,
            top_k=100,
        )

        report_path = os.path.join(nested_dir, 'topk_irdrop_VDD.txt')
        assert os.path.isfile(report_path)

    def test_voltage_column_values_correct(self):
        """Verify that voltage values in the report match input voltages."""
        voltages = {
            '1000_2000_M1': 0.987654,
            '2000_3000_M2': 0.912345,
        }

        generate_topk_report(
            voltages=voltages,
            nominal_voltage=1.0,
            net_name='VDD',
            pad_nodes=set(),
            output_dir=self.tmpdir,
            top_k=100,
        )

        report_path = os.path.join(self.tmpdir, 'topk_irdrop_VDD.txt')
        _, rows = _parse_report_file(report_path)

        by_node = {r[1]: r for r in rows}
        # Voltage column is index 5
        assert float(by_node['1000_2000_M1'][5]) == pytest.approx(0.987654, abs=1e-6)
        assert float(by_node['2000_3000_M2'][5]) == pytest.approx(0.912345, abs=1e-6)

    def test_drop_percentage_calculation(self):
        """Verify drop percentage is computed correctly: (drop / Vdd) * 100."""
        voltages = {'1000_2000_M1': 0.950}  # drop = 50mV = 5%

        generate_topk_report(
            voltages=voltages,
            nominal_voltage=1.0,
            net_name='VDD',
            pad_nodes=set(),
            output_dir=self.tmpdir,
            top_k=100,
        )

        report_path = os.path.join(self.tmpdir, 'topk_irdrop_VDD.txt')
        _, rows = _parse_report_file(report_path)
        assert len(rows) == 1
        assert float(rows[0][7]) == pytest.approx(5.0, abs=0.01)

    def test_multiple_pad_nodes_excluded(self):
        """Multiple pad nodes should all be excluded from the report."""
        voltages = {
            'VDD_vsrc_1': 1.0,
            'VDD_vsrc_2': 1.0,
            'VDD_bump': 1.0,
            '1000_2000_M1': 0.95,
        }
        pad_nodes = {'VDD_vsrc_1', 'VDD_vsrc_2', 'VDD_bump'}

        generate_topk_report(
            voltages=voltages,
            nominal_voltage=1.0,
            net_name='VDD',
            pad_nodes=pad_nodes,
            output_dir=self.tmpdir,
            top_k=100,
        )

        report_path = os.path.join(self.tmpdir, 'topk_irdrop_VDD.txt')
        _, rows = _parse_report_file(report_path)
        assert len(rows) == 1
        assert rows[0][1] == '1000_2000_M1'

    def test_extra_header_lines(self):
        """Extra header lines appear between 'Nominal Voltage' and first separator."""
        generate_topk_report(
            voltages={'1000_2000_M1': 0.99},
            nominal_voltage=1.0,
            net_name='VDD',
            pad_nodes=set(),
            output_dir=self.tmpdir,
            top_k=100,
            extra_header_lines=['Backend: splu', 'Solve Time: 12.34 ms'],
        )

        report_path = os.path.join(self.tmpdir, 'topk_irdrop_VDD.txt')
        with open(report_path, 'r') as f:
            lines = f.read().splitlines()

        # Extra lines should be at indices 3 and 4 (after Nominal Voltage at 2)
        assert lines[3] == 'Backend: splu'
        assert lines[4] == 'Solve Time: 12.34 ms'
        # First separator should now be at index 5
        assert lines[5].startswith('===')

        # Data rows should still parse correctly via dynamic helper
        _, rows = _parse_report_file(report_path)
        assert len(rows) == 1
        assert rows[0][1] == '1000_2000_M1'

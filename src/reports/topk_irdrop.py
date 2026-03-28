"""Top-K worst IR-drop report generation.

Generates a ranked report of the nodes with the largest IR-drop, sorted
in descending order.  Output format matches the flat solver's
``_generate_topk_report()`` in ``solver/pdn_solver.py`` so that flat and
distributed results are directly comparable.

Only PDN netlists with ``X_Y_LAYER`` node names (e.g. ``1000_2000_M1``)
are supported.  Synthetic grids using ``NodeID`` tuples will produce
``N/A`` for layer/coordinate columns.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Set

from visualization.stripe_heatmap import parse_node_info

logger = logging.getLogger(__name__)


def _fmt_peak_time(peak_time: Optional[float], show: bool) -> str:
    """Format peak time column value. Returns empty string when disabled."""
    if not show:
        return ''
    if peak_time is not None:
        return f"{peak_time * 1e9:<14.3f} "
    return f"{'N/A':<14} "


def generate_topk_report(
    voltages: Dict[str, float],
    nominal_voltage: float,
    net_name: str,
    pad_nodes: Set[str],
    output_dir: str,
    top_k: int = 100,
    node_to_instance: Optional[Dict[str, str]] = None,
    extra_header_lines: Optional[List[str]] = None,
    node_to_peak_time: Optional[Dict[str, float]] = None,
) -> None:
    """Write a top-K worst IR-drop report file and log the top-10 to console.

    Args:
        voltages: Mapping of node name to solved voltage (V).
        nominal_voltage: Nominal supply voltage (Vdd) in volts.
        net_name: Power net name (used in file name and header).
        pad_nodes: Set of pad/voltage-source node names to exclude from the
            ranking (these sit at Vdd by construction).
        output_dir: Directory where ``topk_irdrop_{net_name}.txt`` will be
            written.  Created if it does not exist.
        top_k: Number of worst nodes to include in the file (default 100).
        node_to_instance: Optional mapping of node name to instance name.
            Nodes without a mapping are reported as ``N/A``.
        extra_header_lines: Optional list of additional header lines inserted
            between the "Nominal Voltage" line and the first ``===`` separator
            (e.g. ``['Backend: splu', 'Solve Time: 12.34 ms']``).
        node_to_peak_time: Optional mapping of node name to peak time in
            seconds.  When provided, a ``PeakTime(ns)`` column is added to
            the report.  Nodes without a mapping show ``N/A``.
    """
    if node_to_instance is None:
        node_to_instance = {}

    show_peak_time = node_to_peak_time is not None

    # ------------------------------------------------------------------
    # Build per-node records, excluding pads and ground
    # ------------------------------------------------------------------
    node_data = []
    for node, voltage in voltages.items():
        if node == '0' or node in pad_nodes:
            continue

        drop = abs(nominal_voltage - voltage)
        drop_pct = (drop / nominal_voltage * 100) if nominal_voltage > 0 else 0.0

        x, y, layer = parse_node_info(node)
        peak_time = node_to_peak_time.get(node) if node_to_peak_time else None
        node_data.append({
            'node': node,
            'voltage': voltage,
            'drop': drop,
            'drop_pct': drop_pct,
            'layer': layer if layer is not None else 'N/A',
            'x': x if x is not None else 'N/A',
            'y': y if y is not None else 'N/A',
            'instance': node_to_instance.get(node, 'N/A'),
            'peak_time': peak_time,
        })

    # Descending by IR-drop magnitude
    node_data.sort(key=lambda d: d['drop'], reverse=True)

    # ------------------------------------------------------------------
    # Write report file
    # ------------------------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)
    report_file = os.path.join(output_dir, f'topk_irdrop_{net_name}.txt')

    peak_time_hdr = f"{'PeakTime(ns)':<14} " if show_peak_time else ''
    header_line = (
        f"{'Rank':<6} {'Node':<30} {'Layer':<8} {'X':<10} {'Y':<10} "
        f"{'Voltage(V)':<12} {'Drop(mV)':<12} {'Drop(%)':<10} "
        f"{peak_time_hdr}{'Instance':<30}"
    )
    sep_width = len(header_line)

    with open(report_file, 'w') as f:
        f.write(f"Top-{top_k} Worst IR-Drop Report\n")
        f.write(f"Net: {net_name}\n")
        f.write(f"Nominal Voltage: {nominal_voltage:.6f} V\n")
        if extra_header_lines:
            for hline in extra_header_lines:
                f.write(f"{hline}\n")
        f.write(f"{'=' * sep_width}\n")
        f.write(f"{header_line}\n")
        f.write(f"{'=' * sep_width}\n")

        for i, data in enumerate(node_data[:top_k], 1):
            pt_str = _fmt_peak_time(data['peak_time'], show_peak_time)
            f.write(
                f"{i:<6} {data['node']:<30} {str(data['layer']):<8} "
                f"{str(data['x']):<10} {str(data['y']):<10} "
                f"{data['voltage']:<12.6f} {data['drop'] * 1000:<12.3f} "
                f"{data['drop_pct']:<10.2f} {pt_str}{data['instance']:<30}\n"
            )

    logger.info(f"  Saved top-K report: {report_file}")

    # ------------------------------------------------------------------
    # Console summary (top-10)
    # ------------------------------------------------------------------
    peak_time_con_hdr = f"{'PeakTime(ns)':<14} " if show_peak_time else ''
    con_header = (
        f"{'Rank':<6} {'Node':<30} {'Layer':<8} {'Drop(mV)':<12} {'Drop(%)':<10} "
        f"{peak_time_con_hdr}"
    )
    logger.info(f"\n  Top-10 Worst IR-Drop Nodes for {net_name}:")
    logger.info(f"  {con_header}")
    logger.info(f"  {'-' * len(con_header)}")
    for i, data in enumerate(node_data[:10], 1):
        logger.info(
            f"  {i:<6} {data['node']:<30} {str(data['layer']):<8} "
            f"{data['drop'] * 1000:<12.3f} {data['drop_pct']:<10.2f} "
            f"{_fmt_peak_time(data['peak_time'], show_peak_time)}"
        )

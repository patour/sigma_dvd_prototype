"""Floating nodes report generation for power grid analysis.

This module provides utilities to collect and report on floating nodes (disconnected
islands) detected during IR-drop analysis.

IMPORTANT: This module only supports PDN netlists with X_Y_LAYER node naming format
(e.g., "1000_2000_M1"). Synthetic grids using NodeID(layer, idx) tuples are not
supported and will produce incomplete layer groupings.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set

from visualization.stripe_heatmap import parse_node_info


@dataclass
class FloatingNodesData:
    """Container for floating node information.

    Attributes:
        layer_floating_counts: Count of floating nodes per layer
        layer_total_counts: Total node count per layer
        layer_floating_nodes: List of floating node names per layer, sorted by (X, Y)
        dropped_connectivity: Nodes dropped during connectivity tracing (not connected to pads)
        disconnected_interface: Global interface nodes not connected to pads (distributed only)
        connected_to_floating: Tile nodes connected to floating interface (distributed only)
        total_floating: Total number of floating nodes
        total_nodes: Total number of nodes in the grid
    """
    layer_floating_counts: Dict[str, int] = field(default_factory=dict)
    layer_total_counts: Dict[str, int] = field(default_factory=dict)
    layer_floating_nodes: Dict[str, List[str]] = field(default_factory=dict)

    dropped_connectivity: Set[str] = field(default_factory=set)
    disconnected_interface: Set[str] = field(default_factory=set)
    connected_to_floating: Set[str] = field(default_factory=set)

    total_floating: int = 0
    total_nodes: int = 0
    net_name: str = ''


def _sort_layer_key(layer: str) -> tuple:
    """Generate sort key for layer names.

    Numeric layers (e.g., 'M1', '2') sort before alphabetic layers.
    Numeric layers are sorted by integer value.
    Plain digits sort before 'M' prefix layers at the same numeric level.

    Args:
        layer: Layer name

    Returns:
        Sort key tuple (is_non_numeric, numeric_value, prefix, string_value)
    """
    layer_str = str(layer)

    # Try to extract numeric suffix (e.g., "M1" -> 1, "2" -> 2)
    if layer_str.isdigit():
        return (0, int(layer_str), '', '')

    # Try "M<number>" pattern
    if layer_str.startswith('M') and len(layer_str) > 1 and layer_str[1:].isdigit():
        return (0, int(layer_str[1:]), 'M', '')

    # Non-numeric layers sort last
    return (1, 0, '', layer_str)


def _group_nodes_by_layer(nodes: Set[str]) -> Dict[str, List[str]]:
    """Group nodes by layer and sort each layer's nodes by (X, Y).

    Args:
        nodes: Set of node names

    Returns:
        Dict mapping layer -> sorted list of node names
    """
    # Group nodes by layer with parsed coordinates
    layer_nodes: Dict[str, List[tuple]] = {}

    for node in nodes:
        x, y, layer = parse_node_info(node)
        if layer is not None:
            if layer not in layer_nodes:
                layer_nodes[layer] = []
            # Store (x, y, node_name) for sorting
            layer_nodes[layer].append((x if x is not None else 0,
                                       y if y is not None else 0,
                                       node))

    # Sort each layer's nodes by (x, y) and extract node names
    result = {}
    for layer, node_list in layer_nodes.items():
        # Sort by x first, then y
        sorted_nodes = sorted(node_list, key=lambda item: (item[0], item[1]))
        result[layer] = [node for _, _, node in sorted_nodes]

    return result


def collect_floating_nodes_distributed(
    model: 'DistributedPowerGridModel',
    context: 'DistributedSolverContext',
    workers: List[Any],
    net_name: str = '',
) -> FloatingNodesData:
    """Collect floating nodes information from distributed DDM solve.

    Aggregates floating nodes from:
    - Per-tile island removal (via TileWorker.get_floating_nodes_data())
    - Global interface islands (from context.removed_interface_nodes)

    Args:
        model: The distributed power grid model
        context: Solver context with removed_interface_nodes
        workers: List of TileWorker actors

    Returns:
        FloatingNodesData with aggregated floating node information.
    """
    # 1. Collect per-tile floating nodes
    try:
        tile_floating_data = model.backend.call_all(workers, 'get_floating_nodes_data')
    except Exception as e:
        warnings.warn(
            f"Failed to collect floating nodes data from workers: {e}. "
            f"Floating nodes report may be incomplete."
        )
        tile_floating_data = []  # Continue with empty data

    # 2. Aggregate removed nodes from all tiles
    dropped_connectivity: Set[str] = set()
    for tile_data in tile_floating_data:
        dropped_connectivity.update(tile_data['removed_nodes'])

    # 3. Get global interface islands
    disconnected_interface = context.removed_interface_nodes

    # 4. Combine all floating nodes
    all_floating_nodes = dropped_connectivity | disconnected_interface

    # 5. Group by layer
    layer_grouped = _group_nodes_by_layer(all_floating_nodes)

    # 6. Calculate total node count
    # tile_interior_counts are POST-removal (floating nodes already removed from them),
    # Package tap nodes (internal package nodes) aren't in tiles or interface,
    # so they need to be counted separately
    total_interior = sum(model.tile_interior_counts.values())
    total_interface = len(model.interface_nodes)
    # tap_nodes are internal package nodes (not vsrc pads, not die attachments)
    package_tap_nodes = len(model.package_data.tap_nodes)

    total_nodes = total_interior + total_interface + len(all_floating_nodes) + package_tap_nodes

    # 7. For distributed mode, we don't have accurate per-layer total counts
    # Use empty dict so the report shows counts without percentages
    layer_total_counts: Dict[str, int] = {}

    # Build result
    layer_floating_counts = {layer: len(nodes) for layer, nodes in layer_grouped.items()}

    return FloatingNodesData(
        layer_floating_counts=layer_floating_counts,
        layer_total_counts=layer_total_counts,
        layer_floating_nodes=layer_grouped,
        dropped_connectivity=dropped_connectivity,
        disconnected_interface=disconnected_interface,
        connected_to_floating=set(),  # Not tracked in current implementation
        total_floating=len(all_floating_nodes),
        total_nodes=total_nodes,
        net_name=net_name,
    )


def collect_floating_nodes_flat(model, net_name: str = '') -> FloatingNodesData:
    """Collect floating nodes information from a flat UnifiedPowerGridModel.

    Note: This function requires PDN netlists with X_Y_LAYER node naming format
    (e.g., "1000_2000_M1"). Synthetic grids using NodeID(layer, idx) tuples are
    not supported and will result in incomplete layer groupings. Nodes that fail
    to parse will be silently skipped during layer grouping.

    Args:
        model: UnifiedPowerGridModel instance (PDN netlist only)

    Returns:
        FloatingNodesData with collected information
    """
    from model.unified_model import UnifiedPowerGridModel

    if not isinstance(model, UnifiedPowerGridModel):
        raise TypeError(f"Expected UnifiedPowerGridModel, got {type(model).__name__}")

    # Get floating nodes and total nodes
    floating_nodes = model.removed_island_nodes
    valid_nodes = model.valid_nodes
    total_nodes = len(valid_nodes) + len(floating_nodes)

    # Group floating nodes by layer
    layer_grouped = _group_nodes_by_layer(floating_nodes)

    # Count nodes per layer (need to scan all nodes)
    all_nodes = valid_nodes | floating_nodes
    all_nodes_by_layer = _group_nodes_by_layer(all_nodes)

    # Detect if model appears to have non-PDN nodes (e.g., NodeID tuples)
    # Sample up to 100 nodes to check parse success rate
    sample_size = min(100, len(all_nodes))
    if sample_size > 0:
        import random
        sample_nodes = random.sample(list(all_nodes), sample_size)
        unparseable_count = sum(1 for node in sample_nodes
                               if parse_node_info(node) == (None, None, None))
        unparseable_fraction = unparseable_count / sample_size

        # Warn if >50% of sampled nodes fail to parse
        if unparseable_fraction > 0.5:
            warnings.warn(
                f"Detected {unparseable_fraction:.0%} unparseable nodes in sample. "
                "This module only supports PDN netlists with X_Y_LAYER node naming "
                "(e.g., '1000_2000_M1'). Synthetic grids using NodeID(layer, idx) "
                "tuples are not supported and will produce incomplete layer groupings.",
                UserWarning,
                stacklevel=2
            )

    layer_total_counts = {layer: len(nodes) for layer, nodes in all_nodes_by_layer.items()}
    layer_floating_counts = {layer: len(nodes) for layer, nodes in layer_grouped.items()}

    # In flat mode, all floating nodes are in "dropped_connectivity" category
    return FloatingNodesData(
        layer_floating_counts=layer_floating_counts,
        layer_total_counts=layer_total_counts,
        layer_floating_nodes=layer_grouped,
        dropped_connectivity=floating_nodes.copy(),
        disconnected_interface=set(),
        connected_to_floating=set(),
        total_floating=len(floating_nodes),
        total_nodes=total_nodes,
        net_name=net_name,
    )


def generate_floating_nodes_report(
    data: FloatingNodesData,
    output_dir: Path,
    verbose: bool = False,
) -> None:
    """Generate floating nodes report files.

    Always generates:
        - floating_nodes_report.md: Summary and per-layer statistics

    If verbose=True, also generates:
        - floating_nodes_<layer>.txt: One file per layer with node names (one per line)

    Args:
        data: FloatingNodesData with collected information
        output_dir: Directory to write report files
        verbose: If True, write per-layer node lists
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / 'floating_nodes_report.md'

    with open(report_path, 'w') as f:
        # Header
        if data.net_name:
            f.write(f"# Floating Nodes Report — Net: {data.net_name}\n\n")
        else:
            f.write("# Floating Nodes Report\n\n")

        # Summary section
        f.write("## Summary\n\n")
        if data.total_floating == 0:
            f.write("No floating nodes detected.\n\n")
        else:
            percentage = 100.0 * data.total_floating / data.total_nodes if data.total_nodes > 0 else 0.0
            f.write(f"- Total floating nodes: {data.total_floating:,} ({percentage:.2f}% of {data.total_nodes:,} total)\n")

            layers_affected = len(data.layer_floating_counts)
            if data.layer_total_counts:
                total_layers = len(data.layer_total_counts)
                f.write(f"- Layers affected: {layers_affected} of {total_layers}\n\n")
            else:
                f.write(f"- Layers affected: {layers_affected}\n\n")

        # Per-layer breakdown
        if data.layer_floating_counts:
            f.write("## Per-Layer Breakdown\n\n")

            # Check if we have total counts available
            has_totals = bool(data.layer_total_counts)

            if has_totals:
                f.write("| Layer | Floating | Total | Percentage |\n")
                f.write("|-------|----------|-------|------------|\n")
            else:
                f.write("| Layer | Floating |\n")
                f.write("|-------|----------|\n")

            # Sort layers
            sorted_layers = sorted(data.layer_floating_counts.keys(), key=_sort_layer_key)

            for layer in sorted_layers:
                floating_count = data.layer_floating_counts[layer]
                if has_totals:
                    total_count = data.layer_total_counts.get(layer, floating_count)
                    percentage = 100.0 * floating_count / total_count if total_count > 0 else 0.0
                    f.write(f"| {layer:<5} | {floating_count:>8,} | {total_count:>5,} | {percentage:>9.2f}% |\n")
                else:
                    f.write(f"| {layer:<5} | {floating_count:>8,} |\n")

            f.write("\n")

        # Category breakdown
        category_counts = [
            ('Dropped during connectivity tracing', len(data.dropped_connectivity)),
            ('Disconnected interface nodes', len(data.disconnected_interface)),
            ('Connected to floating interface', len(data.connected_to_floating)),
        ]

        # Only show categories with non-zero counts
        non_zero_categories = [(name, count) for name, count in category_counts if count > 0]

        if non_zero_categories:
            f.write("## Category Breakdown\n\n")
            f.write("| Category | Count | Percentage |\n")
            f.write("|----------|-------|------------|\n")

            for name, count in non_zero_categories:
                percentage = 100.0 * count / data.total_floating if data.total_floating > 0 else 0.0
                f.write(f"| {name:<40} | {count:>5,} | {percentage:>9.1f}% |\n")

            f.write("\n")

    # Write per-layer files in verbose mode
    if verbose and data.layer_floating_nodes:
        sorted_layers = sorted(data.layer_floating_nodes.keys(), key=_sort_layer_key)

        # Prepare net name suffix for filenames
        net_suffix = ''
        if data.net_name:
            net_name_lower = data.net_name.lower().replace(' ', '_')
            net_suffix = f'{net_name_lower}_'

        for layer in sorted_layers:
            nodes = data.layer_floating_nodes[layer]
            layer_file = output_dir / f'floating_nodes_{net_suffix}{layer}.txt'

            with open(layer_file, 'w') as f:
                for node in nodes:
                    f.write(f"{node}\n")

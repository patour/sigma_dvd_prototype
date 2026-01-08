"""Unified grid partitioner for power grid analysis.

Supports:
- Layer-based partitioning (for hierarchical IR-drop solving)
- Spatial partitioning (for load balancing)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np

from .unified_model import UnifiedPowerGridModel, LayerID


@dataclass
class UnifiedPartition:
    """A single partition in the grid.

    Attributes:
        partition_id: Integer identifier for this partition
        interior_nodes: Set of nodes inside this partition
        separator_nodes: Set of boundary/separator nodes
        load_nodes: Set of load nodes in this partition
        layer_range: (min_layer, max_layer) if layer-based partitioning
    """
    partition_id: int
    interior_nodes: Set[Any]
    separator_nodes: Set[Any]
    load_nodes: Set[Any]
    layer_range: Tuple[Optional[LayerID], Optional[LayerID]] = (None, None)


@dataclass
class UnifiedPartitionResult:
    """Result of grid partitioning.

    Attributes:
        partitions: List of UnifiedPartition objects
        separator_nodes: Global set of all separator/boundary nodes
        boundary_edges: List of edges crossing partition boundaries
        load_balance_ratio: Ratio of max/min load counts (lower is better)
        partition_axis: Axis used for partitioning ('layer', 'x', 'y', 'auto')
        metadata: Additional partitioning metadata
    """
    partitions: List[UnifiedPartition]
    separator_nodes: Set[Any]
    boundary_edges: List[Tuple[Any, Any, Dict]]
    load_balance_ratio: float
    partition_axis: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class UnifiedPartitioner:
    """Unified partitioner supporting both synthetic and PDN grids.

    Supports two partitioning strategies:
    1. Layer-based: Partition at metal layer boundaries (for hierarchical solve)
    2. Spatial: Partition by X/Y coordinates (load balancing)

    Example:
        model = create_model_from_synthetic(G, pads, vdd=1.0)
        partitioner = UnifiedPartitioner(model, load_nodes=loads)

        # Layer-based partitioning
        result = partitioner.partition_by_layer(partition_layer=1)

        # Spatial partitioning
        result = partitioner.partition_spatial(num_partitions=4, axis='auto')
    """

    def __init__(
        self,
        model: UnifiedPowerGridModel,
        load_nodes: Optional[Dict[Any, float]] = None,
    ):
        """Initialize partitioner.

        Args:
            model: UnifiedPowerGridModel instance
            load_nodes: Dict mapping load nodes to current values (optional)
        """
        self.model = model
        self.load_nodes = set(load_nodes.keys()) if load_nodes else set()
        self.load_currents = load_nodes or {}

    def partition_by_layer(
        self,
        partition_layer: LayerID,
    ) -> UnifiedPartitionResult:
        """Partition grid at a metal layer boundary.

        Creates two partitions:
        - Top grid: layers >= partition_layer
        - Bottom grid: layers < partition_layer

        Args:
            partition_layer: Layer to partition at

        Returns:
            UnifiedPartitionResult with two partitions
        """
        # Use model's decomposition method
        top_nodes, bottom_nodes, port_nodes, via_edges = self.model._decompose_at_layer(partition_layer)

        # Categorize loads
        top_loads = self.load_nodes & top_nodes
        bottom_loads = self.load_nodes & bottom_nodes

        # Get all layers for range info
        all_layers = self.model.get_all_layers()
        partition_idx = None
        for i, layer in enumerate(all_layers):
            if layer == partition_layer or str(layer) == str(partition_layer):
                partition_idx = i
                break

        top_layer_range = (partition_layer, all_layers[-1] if all_layers else None)
        bottom_layer_range = (all_layers[0] if all_layers else None, partition_layer)

        # Create partitions
        top_partition = UnifiedPartition(
            partition_id=0,
            interior_nodes=top_nodes - port_nodes,
            separator_nodes=port_nodes,
            load_nodes=top_loads,
            layer_range=top_layer_range,
        )

        bottom_partition = UnifiedPartition(
            partition_id=1,
            interior_nodes=bottom_nodes,
            separator_nodes=port_nodes,
            load_nodes=bottom_loads,
            layer_range=bottom_layer_range,
        )

        # Compute balance ratio
        counts = [len(top_loads), len(bottom_loads)]
        if min(counts) > 0:
            balance_ratio = max(counts) / min(counts)
        else:
            balance_ratio = float('inf') if max(counts) > 0 else 1.0

        # Format boundary edges
        boundary_edges = [(u, v, edge_info.original_data) for u, v, edge_info in via_edges]

        return UnifiedPartitionResult(
            partitions=[top_partition, bottom_partition],
            separator_nodes=port_nodes,
            boundary_edges=boundary_edges,
            load_balance_ratio=balance_ratio,
            partition_axis='layer',
            metadata={
                'partition_layer': partition_layer,
                'top_node_count': len(top_nodes),
                'bottom_node_count': len(bottom_nodes),
                'port_count': len(port_nodes),
            },
        )

    def partition_spatial(
        self,
        num_partitions: int,
        axis: str = 'auto',
        balance_tolerance: float = 0.15,
    ) -> UnifiedPartitionResult:
        """Partition grid spatially by coordinates.

        Args:
            num_partitions: Number of partitions to create
            axis: 'x', 'y', or 'auto' (selects axis with larger range)
            balance_tolerance: Maximum allowed load imbalance (0.15 = 15%)

        Returns:
            UnifiedPartitionResult with num_partitions partitions
        """
        if num_partitions < 2:
            raise ValueError("Need at least 2 partitions")

        if len(self.load_nodes) < num_partitions:
            raise ValueError(f"Cannot create {num_partitions} partitions with only {len(self.load_nodes)} load nodes")

        # Collect load node coordinates
        load_coords = []
        for node in self.load_nodes:
            info = self.model.get_node_info(node)
            if info.xy:
                load_coords.append((node, info.xy[0], info.xy[1]))

        if not load_coords:
            raise ValueError("No load nodes with coordinates found")

        # Determine best axis if auto
        if axis == 'auto':
            xs = [c[1] for c in load_coords]
            ys = [c[2] for c in load_coords]
            x_range = max(xs) - min(xs) if xs else 0
            y_range = max(ys) - min(ys) if ys else 0
            axis = 'x' if x_range >= y_range else 'y'

        coord_idx = 1 if axis == 'x' else 2  # Index in (node, x, y) tuple

        # Sort by chosen axis
        load_coords.sort(key=lambda t: t[coord_idx])

        # Compute partition boundaries for balanced loads
        total_loads = len(load_coords)
        base = total_loads // num_partitions
        extras = total_loads % num_partitions
        counts = [base + (1 if i < extras else 0) for i in range(num_partitions)]

        # Assign loads to partitions
        partition_loads: List[Set] = [set() for _ in range(num_partitions)]
        idx = 0
        for p, count in enumerate(counts):
            for _ in range(count):
                partition_loads[p].add(load_coords[idx][0])
                idx += 1

        # Compute partition boundaries (for separator identification)
        boundaries = []
        acc = 0
        for c in counts[:-1]:
            acc += c
            if acc > 0 and acc < len(load_coords):
                boundary_coord = (load_coords[acc-1][coord_idx] + load_coords[acc][coord_idx]) / 2
                boundaries.append(boundary_coord)

        # Assign all nodes to partitions based on coordinate
        partition_nodes: List[Set] = [set() for _ in range(num_partitions)]
        separator_nodes = set()

        for node in self.model.graph.nodes():
            info = self.model.get_node_info(node)
            if info.xy is None:
                continue

            coord = info.xy[0] if axis == 'x' else info.xy[1]

            # Find which partition
            p = 0
            for i, boundary in enumerate(boundaries):
                if coord > boundary:
                    p = i + 1

            # Check if near boundary (potential separator)
            near_boundary = any(abs(coord - b) < 1e-6 for b in boundaries)

            if near_boundary and node not in self.load_nodes:
                separator_nodes.add(node)
            partition_nodes[p].add(node)

        # Build partition objects
        partitions = []
        for p in range(num_partitions):
            partitions.append(UnifiedPartition(
                partition_id=p,
                interior_nodes=partition_nodes[p] - separator_nodes,
                separator_nodes=separator_nodes & partition_nodes[p],
                load_nodes=partition_loads[p],
                layer_range=(None, None),
            ))

        # Balance ratio
        load_counts = [len(p.load_nodes) for p in partitions if len(p.load_nodes) > 0]
        if load_counts:
            balance_ratio = max(load_counts) / min(load_counts) if min(load_counts) > 0 else float('inf')
        else:
            balance_ratio = 1.0

        # Identify boundary edges
        boundary_edges = []
        for u, v, data in self.model.graph.edges(data=True):
            if u in separator_nodes or v in separator_nodes:
                boundary_edges.append((u, v, data))

        return UnifiedPartitionResult(
            partitions=partitions,
            separator_nodes=separator_nodes,
            boundary_edges=boundary_edges,
            load_balance_ratio=balance_ratio,
            partition_axis=axis,
            metadata={
                'boundaries': boundaries,
                'partition_counts': [len(p.interior_nodes) for p in partitions],
                'load_counts': [len(p.load_nodes) for p in partitions],
            },
        )

    def visualize_partitions(
        self,
        result: UnifiedPartitionResult,
        ax=None,
        show: bool = False,
        colors: Optional[List[str]] = None,
    ):
        """Visualize partition result.

        Args:
            result: UnifiedPartitionResult to visualize
            ax: Matplotlib axes (created if None)
            show: Whether to display the plot
            colors: List of colors for partitions

        Returns:
            (fig, ax) matplotlib objects
        """
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        else:
            fig = ax.figure

        # Default colors
        if colors is None:
            colors = plt.cm.tab10.colors[:len(result.partitions)]

        # Draw edges
        for u, v in self.model.graph.edges():
            u_info = self.model.get_node_info(u)
            v_info = self.model.get_node_info(v)

            xy_u = u_info.xy
            xy_v = v_info.xy

            if xy_u is None or xy_v is None:
                continue

            # Check if boundary edge
            is_boundary = u in result.separator_nodes or v in result.separator_nodes
            color = 'green' if is_boundary else 'lightgray'
            alpha = 0.8 if is_boundary else 0.3
            lw = 1.5 if is_boundary else 0.5

            ax.plot([xy_u[0], xy_v[0]], [xy_u[1], xy_v[1]], color=color, alpha=alpha, lw=lw)

        # Draw nodes by partition
        for i, partition in enumerate(result.partitions):
            color = colors[i % len(colors)]

            for node in partition.interior_nodes:
                info = self.model.get_node_info(node)
                if info.xy:
                    marker = 'o' if node in partition.load_nodes else '.'
                    size = 50 if node in partition.load_nodes else 10
                    ax.scatter(info.xy[0], info.xy[1], c=[color], s=size, marker=marker, alpha=0.7)

        # Draw separator nodes
        for node in result.separator_nodes:
            info = self.model.get_node_info(node)
            if info.xy:
                ax.scatter(info.xy[0], info.xy[1], c='green', s=80, marker='s',
                          edgecolors='black', linewidths=1.5, zorder=5)

        # Draw pad nodes
        for pad in self.model.pad_nodes:
            info = self.model.get_node_info(pad)
            if info.xy:
                ax.scatter(info.xy[0], info.xy[1], c='gold', s=100, marker='*',
                          edgecolors='black', linewidths=1, zorder=6)

        ax.set_aspect('equal')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f'Partitions ({result.partition_axis} axis)\n'
                    f'Balance ratio: {result.load_balance_ratio:.2f}')

        # Legend
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D

        legend_elements = []
        for i, partition in enumerate(result.partitions):
            color = colors[i % len(colors)]
            legend_elements.append(Patch(facecolor=color, alpha=0.7,
                                        label=f'Partition {i} ({len(partition.load_nodes)} loads)'))

        legend_elements.extend([
            Line2D([0], [0], marker='s', color='w', markerfacecolor='green',
                  markeredgecolor='black', markersize=10, label=f'Separators ({len(result.separator_nodes)})'),
            Line2D([0], [0], marker='*', color='w', markerfacecolor='gold',
                  markeredgecolor='black', markersize=12, label=f'Pads ({len(self.model.pad_nodes)})'),
        ])

        ax.legend(handles=legend_elements, loc='upper right', fontsize=8)

        plt.tight_layout()

        if show:
            plt.show()

        return fig, ax

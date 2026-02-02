#!/usr/bin/env python3
"""Stripe-based heatmap visualization for PDN analysis.

Provides efficient visualization of node values using orientation-aware
stripe aggregation. Optimized for 10M+ nodes with numpy vectorization.

This module is standalone and does not require PDNPlotter or net_connectivity.
It parses node coordinates directly from node names (X_Y_LAYER format).

Usage:
    from analysis.stripe_heatmap import plot_stripe_heatmap

    # Basic usage with peak IR-drop values
    plot_stripe_heatmap(
        node_values=peak_ir_drop_per_node,  # Dict[node_name, value_in_volts]
        layers=['M1', 'M2'],
        plot_dir='./plots',
        title_prefix='Peak IR-Drop',
        value_label='Peak IR-Drop (mV)',
        value_scale=1000.0,  # V to mV
    )

    # With markers and windows overlay
    plot_stripe_heatmap(
        node_values=peak_ir_drop_per_node,
        layers=['M1'],
        markers=[(1000.0, 2000.0, 'M1')],  # (x, y, layer)
        windows=[(900, 1100, 1900, 2100, 'M1')],  # (x_min, x_max, y_min, y_max, layer)
    )
"""

from __future__ import annotations

import gc
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# =============================================================================
# Node Parsing Utilities
# =============================================================================

def parse_node_info(node_name: str) -> Tuple[Optional[float], Optional[float], Optional[str]]:
    """Parse node name to (x, y, layer).

    Node name format: <x>_<y>_<layer> (e.g., "1000_2000_M1")

    Args:
        node_name: Node name string

    Returns:
        (x, y, layer) tuple, or (None, None, None) if parsing fails.
    """
    parts = str(node_name).split('_')
    if len(parts) >= 3:
        try:
            x = float(parts[0])
            y = float(parts[1])
            layer = parts[2]
            return x, y, layer
        except ValueError:
            pass
    return None, None, None


def extract_node_data_vectorized(
    node_values: Dict[str, float]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract arrays from node_values dict in single pass.

    Optimized for large node counts (10M+).

    Args:
        node_values: Dict mapping node name to value

    Returns:
        (nodes, xs, ys, layers, values) arrays for valid nodes only
    """
    n = len(node_values)
    nodes = np.empty(n, dtype=object)
    xs = np.empty(n, dtype=np.float64)
    ys = np.empty(n, dtype=np.float64)
    layers = np.empty(n, dtype=object)
    values = np.empty(n, dtype=np.float64)
    valid = np.zeros(n, dtype=bool)

    for i, (node, val) in enumerate(node_values.items()):
        x, y, layer = parse_node_info(node)
        if x is not None:
            nodes[i] = node
            xs[i] = x
            ys[i] = y
            layers[i] = layer
            values[i] = val
            valid[i] = True

    return nodes[valid], xs[valid], ys[valid], layers[valid], values[valid]


# =============================================================================
# Orientation Detection
# =============================================================================

def detect_orientation_from_coords(xs: np.ndarray, ys: np.ndarray) -> str:
    """Detect H/V/MIXED orientation from coordinate distribution.

    Uses coordinate variance ratio to determine predominant routing direction.

    Args:
        xs: X coordinates array
        ys: Y coordinates array

    Returns:
        'H' for horizontal, 'V' for vertical, 'MIXED' for no clear orientation
    """
    if len(xs) < 2:
        return 'MIXED'

    # Use unique coordinate counts as a proxy for routing direction
    x_unique = len(np.unique(np.round(xs, 1)))
    y_unique = len(np.unique(np.round(ys, 1)))

    # High X variance with low Y variance = horizontal routing
    if x_unique > y_unique * 3:
        return 'H'
    # High Y variance with low X variance = vertical routing
    elif y_unique > x_unique * 3:
        return 'V'
    return 'MIXED'


# =============================================================================
# Stripe Aggregation
# =============================================================================

def group_nodes_into_stripes(
    xs: np.ndarray,
    ys: np.ndarray,
    values: np.ndarray,
    orientation: str,
) -> Dict[float, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Group nodes into stripes based on orientation.

    For horizontal layers: group by Y coordinate (rows)
    For vertical layers: group by X coordinate (columns)

    Args:
        xs: X coordinates array
        ys: Y coordinates array
        values: Values array
        orientation: 'H' or 'V'

    Returns:
        Dictionary mapping stripe coordinate to (xs, ys, values) tuple
    """
    stripes: Dict[float, Tuple[List[float], List[float], List[float]]] = defaultdict(
        lambda: ([], [], [])
    )

    if orientation == 'H':
        for x, y, v in zip(xs, ys, values):
            stripes[y][0].append(x)
            stripes[y][1].append(y)
            stripes[y][2].append(v)
    else:  # orientation == 'V'
        for x, y, v in zip(xs, ys, values):
            stripes[x][0].append(x)
            stripes[x][1].append(y)
            stripes[x][2].append(v)

    # Convert to numpy arrays
    return {
        coord: (np.array(data[0]), np.array(data[1]), np.array(data[2]))
        for coord, data in stripes.items()
    }


def consolidate_stripes(
    stripes: Dict[float, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    max_stripes: int,
) -> List[Tuple[float, float, np.ndarray, np.ndarray, np.ndarray]]:
    """Consolidate contiguous stripes when count exceeds threshold.

    Groups N consecutive stripes together uniformly to reduce total count.

    Args:
        stripes: Dictionary mapping stripe coordinate to (xs, ys, values)
        max_stripes: Maximum number of stripes to display

    Returns:
        List of (start_coord, end_coord, xs, ys, values) tuples
    """
    sorted_coords = sorted(stripes.keys())
    n_stripes = len(sorted_coords)

    if n_stripes <= max_stripes:
        # No consolidation needed - compute proper boundaries
        result = []
        for i, coord in enumerate(sorted_coords):
            # Calculate stripe boundaries as midpoints between adjacent stripes
            if i == 0:
                if n_stripes > 1:
                    start_coord = coord - (sorted_coords[1] - coord) / 2
                else:
                    start_coord = coord - 1
            else:
                start_coord = (sorted_coords[i - 1] + coord) / 2

            if i == n_stripes - 1:
                if n_stripes > 1:
                    end_coord = coord + (coord - sorted_coords[-2]) / 2
                else:
                    end_coord = coord + 1
            else:
                end_coord = (coord + sorted_coords[i + 1]) / 2

            xs, ys, vals = stripes[coord]
            result.append((start_coord, end_coord, xs, ys, vals))
        return result

    # Calculate grouping factor
    group_size = int(np.ceil(n_stripes / max_stripes))

    consolidated = []
    for i in range(0, n_stripes, group_size):
        group_coords = sorted_coords[i : i + group_size]
        start_coord = group_coords[0]
        end_coord = group_coords[-1]

        # Merge all data from stripes in this group
        merged_xs = []
        merged_ys = []
        merged_vals = []
        for coord in group_coords:
            xs, ys, vals = stripes[coord]
            merged_xs.append(xs)
            merged_ys.append(ys)
            merged_vals.append(vals)

        consolidated.append((
            start_coord,
            end_coord,
            np.concatenate(merged_xs),
            np.concatenate(merged_ys),
            np.concatenate(merged_vals),
        ))

    return consolidated


def aggregate_stripe_bins(
    xs: np.ndarray,
    ys: np.ndarray,
    values: np.ndarray,
    orientation: str,
    bin_size: Optional[int] = None,
    aggregation: str = 'max',
) -> Tuple[np.ndarray, np.ndarray]:
    """Aggregate values within a stripe into bins.

    For horizontal stripes: bin along X direction
    For vertical stripes: bin along Y direction

    Args:
        xs: X coordinates array
        ys: Y coordinates array
        values: Values array
        orientation: 'H' or 'V'
        bin_size: Physical bin size. None = auto-calculate
        aggregation: 'max', 'min', or 'sum'

    Returns:
        (bin_edges, bin_values) arrays
    """
    if orientation == 'H':
        coords = xs
        coord_min, coord_max = xs.min(), xs.max()
    else:
        coords = ys
        coord_min, coord_max = ys.min(), ys.max()

    coord_range = coord_max - coord_min
    if coord_range == 0:
        coord_range = 1.0

    # Calculate bin size
    if bin_size is None:
        # Auto-calculate based on node count
        bin_size = max(1, int(coord_range / max(10, np.sqrt(len(values)))))

    # Cap bin count to avoid memory issues
    max_bins = 10000
    if coord_range / bin_size > max_bins:
        bin_size = int(coord_range / max_bins)

    # Create bins - add small epsilon to ensure max value is included
    eps = bin_size * 0.001
    bins = np.arange(coord_min, coord_max + bin_size + eps, bin_size)
    if len(bins) < 2:
        bins = np.array([coord_min, coord_max + 1])

    # Bin assignment
    indices = np.digitize(coords, bins) - 1
    valid_mask = (indices >= 0) & (indices < len(bins) - 1)

    # Aggregate
    bin_values = np.full(len(bins) - 1, np.nan)

    if aggregation == 'sum':
        bin_values = np.zeros(len(bins) - 1)
        np.add.at(bin_values, indices[valid_mask], values[valid_mask])
    else:
        # max or min aggregation
        for idx in range(len(bins) - 1):
            mask = indices[valid_mask] == idx
            if np.any(mask):
                bin_vals = values[valid_mask][mask]
                if aggregation == 'max':
                    bin_values[idx] = np.max(bin_vals)
                else:  # min
                    bin_values[idx] = np.min(bin_vals)

    return bins, bin_values


# =============================================================================
# Fast Imshow Aggregation (for very large layers)
# =============================================================================

def aggregate_fast_imshow(
    xs: np.ndarray,
    ys: np.ndarray,
    values: np.ndarray,
    orientation: str,
    max_bins: int = 1000,
    aggregation: str = 'max',
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fast aggregation returning 2D grid for imshow.

    Uses scipy binned_statistic_2d for efficient aggregation.

    Args:
        xs: X coordinates array
        ys: Y coordinates array
        values: Values array
        orientation: 'H' or 'V'
        max_bins: Maximum bins per dimension
        aggregation: 'max', 'min', or 'sum' (scipy calls it 'mean' for sum)

    Returns:
        (x_edges, y_edges, grid) for imshow
    """
    try:
        from scipy.stats import binned_statistic_2d
    except ImportError:
        raise ImportError("scipy is required for fast imshow aggregation")

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    # Adjust bin counts based on orientation
    n_nodes = len(values)
    base_bins = max(10, int(np.sqrt(n_nodes)))

    if orientation == 'H':
        n_x = min(max_bins, base_bins * 2)
        n_y = min(50, max(5, base_bins // 4))
    elif orientation == 'V':
        n_x = min(50, max(5, base_bins // 4))
        n_y = min(max_bins, base_bins * 2)
    else:
        n_x = min(max_bins, base_bins)
        n_y = min(max_bins, base_bins)

    # Map aggregation to scipy statistic
    stat_map = {'max': 'max', 'min': 'min', 'sum': 'sum'}
    stat = stat_map.get(aggregation, 'max')

    grid, x_edges, y_edges, _ = binned_statistic_2d(
        xs, ys, values, statistic=stat, bins=[n_x, n_y]
    )

    return x_edges, y_edges, grid.T


# =============================================================================
# Main Plotting Function
# =============================================================================

def plot_stripe_heatmap(
    node_values: Dict[str, float],
    layers: Optional[List[str]] = None,
    plot_dir: str = './plots',
    max_stripes: int = 50,
    stripe_bin_size: Optional[int] = None,
    cmap: str = 'RdYlGn_r',
    title_prefix: str = 'Peak IR-Drop',
    value_label: str = 'Peak IR-Drop (mV)',
    value_scale: float = 1000.0,
    aggregation: str = 'max',
    markers: Optional[List[Tuple[float, float, str]]] = None,
    windows: Optional[List[Tuple[float, float, float, float, str]]] = None,
    show: bool = False,
    use_imshow_threshold: int = 100000,
    verbose: bool = False,
) -> None:
    """Generate stripe-mode heatmaps with optional markers and windows.

    Creates one PNG per layer with stripe-based visualization optimized
    for routing-aware display. For layers with >100K nodes, falls back
    to faster imshow-based rendering.

    Args:
        node_values: Dict mapping node name to value (e.g., peak IR-drop in V)
        layers: Optional list of layers to plot (None = all detected)
        plot_dir: Output directory
        max_stripes: Max stripes before consolidation
        stripe_bin_size: Bin size along stripe (None = auto)
        cmap: Colormap name
        title_prefix: Title prefix for each plot
        value_label: Colorbar label
        value_scale: Scale factor for values (e.g., 1000 for V->mV)
        aggregation: 'max', 'min', or 'sum'
        markers: List of (x, y, layer) for marker overlay
        windows: List of (x_min, x_max, y_min, y_max, layer) for window overlay
        show: Display plots interactively
        use_imshow_threshold: Use imshow for layers with more nodes
        verbose: Print progress
    """
    try:
        import matplotlib
        if not show:
            matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches_mod
        from matplotlib.patches import Rectangle
        from matplotlib.collections import PatchCollection
    except ImportError:
        print("Warning: matplotlib not available, skipping plots")
        return

    os.makedirs(plot_dir, exist_ok=True)

    markers = markers or []
    windows = windows or []

    # Extract all node data in single pass
    if verbose:
        print(f"Extracting data from {len(node_values)} nodes...")

    nodes, xs, ys, layer_arr, values = extract_node_data_vectorized(node_values)

    if len(nodes) == 0:
        print("Warning: No valid node data found")
        return

    # Apply value scaling
    values = values * value_scale

    if verbose:
        print(f"Valid nodes: {len(nodes)}, value range: [{values.min():.3f}, {values.max():.3f}]")

    # Get unique layers
    unique_layers = np.unique(layer_arr)
    if layers is not None:
        unique_layers = [l for l in unique_layers if l in layers]

    if len(unique_layers) == 0:
        print("Warning: No matching layers found")
        return

    # Sort layers
    try:
        unique_layers = sorted(unique_layers, key=lambda x: int(x) if str(x).isdigit() else x)
    except (ValueError, TypeError):
        unique_layers = sorted(unique_layers)

    # Process each layer
    for layer in unique_layers:
        if verbose:
            print(f"Processing layer {layer}...")

        # Extract layer data
        mask = layer_arr == layer
        layer_xs = xs[mask]
        layer_ys = ys[mask]
        layer_vals = values[mask]
        n_nodes = len(layer_xs)

        if n_nodes == 0:
            continue

        # Detect orientation
        orientation = detect_orientation_from_coords(layer_xs, layer_ys)

        if verbose:
            print(f"  Orientation: {orientation}, Nodes: {n_nodes}")

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))

        # Coordinate ranges
        x_min, x_max = layer_xs.min(), layer_xs.max()
        y_min, y_max = layer_ys.min(), layer_ys.max()

        # Choose rendering method based on node count
        if n_nodes > use_imshow_threshold or orientation == 'MIXED':
            # Use fast imshow for large layers or MIXED orientation
            if verbose:
                print(f"  Using imshow mode ({n_nodes} nodes)")

            try:
                x_edges, y_edges, grid = aggregate_fast_imshow(
                    layer_xs, layer_ys, layer_vals, orientation, aggregation=aggregation
                )
                im = ax.imshow(
                    grid,
                    extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
                    origin='lower',
                    aspect='auto',
                    cmap=cmap,
                )
            except ImportError:
                # Fallback to simple scatter if scipy not available
                im = ax.scatter(
                    layer_xs, layer_ys, c=layer_vals, cmap=cmap, s=1, alpha=0.7
                )

        else:
            # Use stripe-based rendering
            if verbose:
                print(f"  Using stripe mode")

            # Group into stripes
            stripes = group_nodes_into_stripes(layer_xs, layer_ys, layer_vals, orientation)
            n_original_stripes = len(stripes)

            # Consolidate if needed
            consolidated = consolidate_stripes(stripes, max_stripes)
            n_display_stripes = len(consolidated)

            if verbose:
                print(f"  Stripes: {n_original_stripes} -> {n_display_stripes}")

            # Build stripe data with bin aggregation
            all_bin_values = []
            stripe_data = []

            for start_coord, end_coord, s_xs, s_ys, s_vals in consolidated:
                if len(s_vals) == 0:
                    continue

                bins, bin_values = aggregate_stripe_bins(
                    s_xs, s_ys, s_vals, orientation, stripe_bin_size, aggregation
                )

                valid_vals = bin_values[~np.isnan(bin_values)]
                if len(valid_vals) > 0:
                    stripe_data.append((start_coord, end_coord, bins, bin_values))
                    all_bin_values.extend(valid_vals.tolist())

            if not all_bin_values:
                plt.close(fig)
                continue

            # Create visualization using PatchCollection
            vmin = min(all_bin_values)
            vmax = max(all_bin_values)
            norm = plt.Normalize(vmin=vmin, vmax=vmax)

            rects = []
            colors = []

            if orientation == 'H':
                for start_coord, end_coord, bins, bin_values in stripe_data:
                    stripe_height = end_coord - start_coord
                    for i, (bin_start, bin_end) in enumerate(zip(bins[:-1], bins[1:])):
                        if not np.isnan(bin_values[i]):
                            bin_width = bin_end - bin_start
                            rect = Rectangle((bin_start, start_coord), bin_width, stripe_height)
                            rects.append(rect)
                            colors.append(bin_values[i])
            else:  # V orientation
                for start_coord, end_coord, bins, bin_values in stripe_data:
                    stripe_width = end_coord - start_coord
                    for i, (bin_start, bin_end) in enumerate(zip(bins[:-1], bins[1:])):
                        if not np.isnan(bin_values[i]):
                            bin_height = bin_end - bin_start
                            rect = Rectangle((start_coord, bin_start), stripe_width, bin_height)
                            rects.append(rect)
                            colors.append(bin_values[i])

            if rects:
                pc = PatchCollection(rects, cmap=cmap, norm=norm, edgecolors='none')
                pc.set_array(np.array(colors))
                ax.add_collection(pc)

                # Create ScalarMappable for colorbar
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                sm.set_array([])
                plt.colorbar(sm, ax=ax, label=value_label)

        # Set axis limits
        margin_x = (x_max - x_min) * 0.02
        margin_y = (y_max - y_min) * 0.02
        ax.set_xlim(x_min - margin_x, x_max + margin_x)
        ax.set_ylim(y_min - margin_y, y_max + margin_y)

        # Add markers for this layer
        for mx, my, ml in markers:
            if ml == layer:
                ax.plot(mx, my, 'x', color='red', markersize=12, markeredgewidth=2.5,
                        zorder=10)

        # Add windows for this layer
        for x0, x1, y0, y1, wl in windows:
            if wl == layer:
                rect = patches_mod.Rectangle(
                    (x0, y0), x1 - x0, y1 - y0,
                    edgecolor='red', facecolor='none',
                    linestyle='--', linewidth=2, zorder=9
                )
                ax.add_patch(rect)

        # Labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f'{title_prefix} - Layer {layer} ({n_nodes} nodes, {orientation})')
        ax.set_aspect('equal', adjustable='box')

        plt.tight_layout()

        # Save
        output_file = os.path.join(plot_dir, f'{title_prefix.lower().replace(" ", "_")}_heatmap_{layer}.png')
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        if show:
            plt.show()
        plt.close(fig)

        if verbose:
            print(f"  Saved: {output_file}")

        # Free memory
        del layer_xs, layer_ys, layer_vals
        gc.collect()

    if verbose:
        print("Done generating heatmaps")

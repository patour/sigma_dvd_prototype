#!/usr/bin/env python3
"""
PDN Plotter - Visualization module for power delivery network analysis.

This module handles all heatmap generation for PDN IR-drop analysis:
- Traditional 2D grid heatmaps (with anisotropic binning support)
- Stripe-based heatmaps (orientation-aware visualization)

Author: Based on mpower power grid analysis
Date: December 13, 2025
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict

try:
    import numpy as np
except ImportError:
    print("ERROR: NumPy is required. Install with: pip install numpy")
    raise

try:
    import networkx as nx
except ImportError:
    print("ERROR: NetworkX is required. Install with: pip install networkx")
    raise


class PDNPlotter:
    """
    Visualization generator for PDN analysis results.
    
    Generates voltage and current heatmaps in multiple modes:
    - 2D grid mode: Traditional rectangular binning with anisotropic support
    - Stripe mode: Orientation-aware stripe visualization with consolidation
    """
    
    def __init__(self, graph: nx.MultiDiGraph, 
                 net_connectivity: Dict[str, List[str]],
                 logger: Optional[logging.Logger] = None):
        """
        Initialize PDN plotter.
        
        Args:
            graph: PDN graph with solved voltages
            net_connectivity: Dictionary mapping net names to node lists
            logger: Optional logger instance
        """
        self.graph = graph
        self.net_connectivity = net_connectivity
        self.logger = logger or logging.getLogger(__name__)
    
    def _detect_net_type(self, net_name: str) -> str:
        """
        Detect if net is power or ground type.
        
        Power nets (VDD, VDDD, VDDQ, etc.): worst-case is minimum voltage
        Ground nets (VSS, GND, GNDD, etc.): worst-case is maximum voltage
        
        Args:
            net_name: Net name to classify
            
        Returns:
            'power' or 'ground'
        """
        net_upper = net_name.upper()
        
        # Check for common ground patterns
        ground_patterns = ['VSS', 'GND', 'GNDD']
        for pattern in ground_patterns:
            if pattern in net_upper:
                return 'ground'
        
        # Check for common power patterns
        power_patterns = ['VDD', 'VCC', 'VDDA', 'VDDQ', 'VDDC', 'VDDIO', 'VPP']
        for pattern in power_patterns:
            if pattern in net_upper:
                return 'power'
        
        # Check node net_type attributes
        net_nodes = self.net_connectivity.get(net_name, [])
        for node in net_nodes[:10]:  # Sample first 10 nodes
            if node == '0':
                continue
            node_data = self.graph.nodes.get(node)
            if node_data:
                net_type = node_data.get('net_type', '').upper()
                if 'VSS' in net_type or 'GND' in net_type:
                    return 'ground'
                if 'VDD' in net_type or 'VCC' in net_type:
                    return 'power'
        
        # Default to power (conservative for IR-drop analysis)
        return 'power'
    
    def _find_spatially_separated_worst_nodes(self, nodes_with_voltage: List[Tuple[str, float, float, float]],
                                             net_type: str, x_range: float, y_range: float,
                                             max_nodes: int = 3) -> List[Tuple[str, float, float, float]]:
        """
        Find top-N worst nodes that are spatially separated.
        
        Args:
            nodes_with_voltage: List of (node_name, x, y, voltage) tuples
            net_type: 'power' or 'ground'
            x_range: Range of X coordinates in layer
            y_range: Range of Y coordinates in layer
            max_nodes: Maximum number of nodes to return (default: 3)
            
        Returns:
            List of (node_name, x, y, voltage) tuples for spatially separated worst nodes
        """
        if not nodes_with_voltage:
            return []
        
        # Sort by worst voltage (min for power, max for ground)
        if net_type == 'power':
            sorted_nodes = sorted(nodes_with_voltage, key=lambda x: x[3])  # Min voltage
        else:
            sorted_nodes = sorted(nodes_with_voltage, key=lambda x: x[3], reverse=True)  # Max voltage
        
        # Minimum separation distance (10% of average dimension)
        min_separation = 0.10 * ((x_range + y_range) / 2)
        
        selected = []
        for node_name, x, y, v in sorted_nodes:
            if len(selected) >= max_nodes:
                break
            
            # Check if spatially separated from already selected nodes
            is_separated = True
            for _, sel_x, sel_y, _ in selected:
                distance = np.sqrt((x - sel_x)**2 + (y - sel_y)**2)
                if distance < min_separation:
                    is_separated = False
                    break
            
            if is_separated:
                selected.append((node_name, x, y, v))
        
        return selected
    
    def _detect_layer_orientation(self, net_name: str, layer_id: str, 
                                   net_nodes_set: Set[str],
                                   layer_orientations: Dict[str, str]) -> str:
        """
        Detect predominant routing orientation of a layer by analyzing resistor edges.
        
        Args:
            net_name: Net name
            layer_id: Layer identifier
            net_nodes_set: Set of nodes in this net
            layer_orientations: Manual orientation overrides
            
        Returns:
            'H' for horizontal, 'V' for vertical, 'MIXED' for no clear orientation
        """
        # Check manual override first
        if layer_id in layer_orientations:
            override = layer_orientations[layer_id].upper()
            if override in ['H', 'V', 'SQUARE', 'MIXED']:
                return override
        
        horizontal_edges = 0
        vertical_edges = 0
        diagonal_edges = 0
        
        # Analyze all resistor edges in this layer
        for node in net_nodes_set:
            if node == '0':
                continue
            
            node_data = self.graph.nodes.get(node)
            if not node_data or node_data.get('layer') != layer_id:
                continue
            
            x1 = node_data.get('x')
            y1 = node_data.get('y')
            if x1 is None or y1 is None:
                continue
            
            # Check edges from this node
            for neighbor in self.graph.neighbors(node):
                if neighbor not in net_nodes_set or neighbor == '0':
                    continue
                
                neighbor_data = self.graph.nodes.get(neighbor)
                if not neighbor_data or neighbor_data.get('layer') != layer_id:
                    continue
                
                # Check if there's a resistor edge
                edge_data = self.graph.get_edge_data(node, neighbor)
                if not edge_data:
                    continue
                
                has_resistor = any(d.get('type') == 'R' for d in edge_data.values())
                if not has_resistor:
                    continue
                
                x2 = neighbor_data.get('x')
                y2 = neighbor_data.get('y')
                if x2 is None or y2 is None:
                    continue
                
                # Calculate direction
                dx = abs(x2 - x1)
                dy = abs(y2 - y1)
                
                if dx == 0 and dy == 0:
                    continue
                
                # Classify edge direction (±15° tolerance)
                if dx > 0 or dy > 0:
                    angle_ratio = dy / (dx + 1e-9)  # Avoid division by zero
                    
                    if angle_ratio < 0.27:  # tan(15°) ≈ 0.27 → horizontal
                        horizontal_edges += 1
                    elif angle_ratio > 3.73:  # tan(75°) ≈ 3.73 → vertical
                        vertical_edges += 1
                    else:
                        diagonal_edges += 1
        
        total_edges = horizontal_edges + vertical_edges + diagonal_edges
        
        if total_edges == 0:
            self.logger.debug(f"    Layer {layer_id}: No edges found, defaulting to MIXED")
            return 'MIXED'
        
        h_ratio = horizontal_edges / total_edges
        v_ratio = vertical_edges / total_edges
        
        self.logger.debug(f"    Layer {layer_id}: H={horizontal_edges} ({h_ratio*100:.1f}%), "
                         f"V={vertical_edges} ({v_ratio*100:.1f}%), "
                         f"Diag={diagonal_edges} ({diagonal_edges/total_edges*100:.1f}%)")
        
        # Determine orientation (70% threshold)
        if h_ratio >= 0.70:
            return 'H'
        elif v_ratio >= 0.70:
            return 'V'
        else:
            return 'MIXED'
    
    def _calculate_anisotropic_bins(self, orientation: str, x_min: float, x_max: float,
                                     y_min: float, y_max: float, num_nodes: int,
                                     aspect_ratio: int, base_size_override: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float]]:
        """
        Calculate anisotropic bin arrays based on layer orientation.
        
        For horizontal layers: thin vertical bins (stripes), wide horizontal bins
        For vertical layers: thin horizontal bins, wide vertical bins
        For mixed/square: isotropic square bins
        
        Args:
            orientation: 'H', 'V', or 'MIXED'/'SQUARE'
            x_min, x_max, y_min, y_max: Coordinate ranges
            num_nodes: Number of nodes for bin size calculation
            aspect_ratio: Ratio between parallel and perpendicular dimensions
            base_size_override: Optional override for base size
            
        Returns:
            (x_bins, y_bins, (x_range, y_range)): Bin arrays and original grid dimensions
        """
        x_range = x_max - x_min
        y_range = y_max - y_min
        
        if orientation == 'H':
            # Horizontal routing: preserve vertical resolution, average horizontally
            if base_size_override is not None:
                base_size = base_size_override
            else:
                base_size = max(1, int(y_range / np.sqrt(num_nodes)))
            bin_height = base_size
            bin_width = base_size * aspect_ratio
            
        elif orientation == 'V':
            # Vertical routing: preserve horizontal resolution, average vertically
            if base_size_override is not None:
                base_size = base_size_override
            else:
                base_size = max(1, int(x_range / np.sqrt(num_nodes)))
            bin_width = base_size
            bin_height = base_size * aspect_ratio
            
        else:  # MIXED or SQUARE
            # Isotropic bins
            if base_size_override is not None:
                base_size = base_size_override
            else:
                avg_range = (x_range + y_range) / 2
                base_size = max(1, int(avg_range / np.sqrt(num_nodes)))
            bin_width = base_size
            bin_height = base_size
        
        # Create bin arrays
        x_bins = np.arange(x_min, x_max + bin_width, bin_width)
        y_bins = np.arange(y_min, y_max + bin_height, bin_height)
        
        return x_bins, y_bins, (x_range, y_range)
    
    def _group_nodes_into_stripes(self, net_name: str, layer_id: str, 
                                   orientation: str, nodes_with_data: List[Tuple]) -> Dict[float, List[Tuple]]:
        """
        Group nodes into stripes based on layer orientation.
        
        For horizontal layers: group by Y coordinate (rows)
        For vertical layers: group by X coordinate (columns)
        
        Args:
            net_name: Net name
            layer_id: Layer identifier
            orientation: 'H' or 'V' 
            nodes_with_data: List of (node, x, y, value) tuples
            
        Returns:
            Dictionary mapping stripe coordinate to list of (node, x, y, value) tuples
        """
        stripes = defaultdict(list)
        
        for node_tuple in nodes_with_data:
            node, x, y, value = node_tuple
            
            if orientation == 'H':
                # Horizontal: group by Y (rows at constant Y)
                stripe_coord = y
            else:  # orientation == 'V'
                # Vertical: group by X (columns at constant X)
                stripe_coord = x
            
            stripes[stripe_coord].append(node_tuple)
        
        return stripes
    
    def _consolidate_stripes(self, stripes: Dict[float, List[Tuple]], 
                            max_stripes: int) -> List[Tuple[float, float, List[Tuple]]]:
        """
        Consolidate contiguous stripes when count exceeds threshold.
        
        Groups N consecutive stripes together uniformly to reduce total count below max_stripes.
        
        Args:
            stripes: Dictionary mapping stripe coordinate to node list
            max_stripes: Maximum number of stripes to display
            
        Returns:
            List of (start_coord, end_coord, nodes_list) tuples for consolidated stripes
        """
        if len(stripes) <= max_stripes:
            # No consolidation needed - but still need proper boundaries for each stripe
            sorted_coords = sorted(stripes.keys())
            result = []
            for i, coord in enumerate(sorted_coords):
                # Calculate stripe boundaries as midpoints between adjacent stripes
                if i == 0:
                    if len(sorted_coords) > 1:
                        start_coord = coord - (sorted_coords[1] - coord) / 2
                    else:
                        start_coord = coord
                else:
                    start_coord = (sorted_coords[i-1] + coord) / 2
                
                if i == len(sorted_coords) - 1:
                    if len(sorted_coords) > 1:
                        end_coord = coord + (coord - sorted_coords[-2]) / 2
                    else:
                        end_coord = coord
                else:
                    end_coord = (coord + sorted_coords[i+1]) / 2
                
                result.append((start_coord, end_coord, stripes[coord]))
            return result
        
        # Calculate grouping factor (how many consecutive stripes to merge)
        group_size = int(np.ceil(len(stripes) / max_stripes))
        
        sorted_coords = sorted(stripes.keys())
        consolidated = []
        
        for i in range(0, len(sorted_coords), group_size):
            group_coords = sorted_coords[i:i+group_size]
            start_coord = group_coords[0]
            end_coord = group_coords[-1]
            
            # Merge all nodes from stripes in this group
            merged_nodes = []
            for coord in group_coords:
                merged_nodes.extend(stripes[coord])
            
            consolidated.append((start_coord, end_coord, merged_nodes))
        
        return consolidated
    
    def _calculate_stripe_bins(self, stripe_nodes: List[Tuple], orientation: str,
                               x_min: float, x_max: float, y_min: float, y_max: float,
                               stripe_bin_size: Optional[int] = None) -> np.ndarray:
        """
        Calculate bins for within-stripe aggregation.
        
        For horizontal stripes: bin along X direction
        For vertical stripes: bin along Y direction
        
        Args:
            stripe_nodes: List of (node, x, y, value) tuples in this stripe
            orientation: 'H' or 'V'
            x_min, x_max, y_min, y_max: Coordinate ranges for this stripe
            stripe_bin_size: Physical bin size in coordinate units. None = auto-calculate
            
        Returns:
            Array of bin edges for the parallel dimension
        """
        num_nodes = len(stripe_nodes)
        
        if orientation == 'H':
            # Horizontal stripe: bin along X
            coord_range = x_max - x_min
            if stripe_bin_size is not None:
                bin_size = stripe_bin_size
            else:
                bin_size = max(1, int(coord_range / np.sqrt(max(num_nodes, 1))))
            
            # Ensure at least 2 bins and reasonable bin count
            if coord_range / bin_size > 100000:
                # Too many bins, cap at 10000
                bin_size = int(coord_range / 10000)
            bins = np.arange(x_min, x_max + bin_size, bin_size)
        else:  # orientation == 'V'
            # Vertical stripe: bin along Y
            coord_range = y_max - y_min
            if stripe_bin_size is not None:
                bin_size = stripe_bin_size
            else:
                bin_size = max(1, int(coord_range / np.sqrt(max(num_nodes, 1))))
            
            # Ensure at least 2 bins and reasonable bin count
            if coord_range / bin_size > 100000:
                # Too many bins, cap at 10000
                bin_size = int(coord_range / 10000)
            bins = np.arange(y_min, y_max + bin_size, bin_size)
        
        return bins
    
    def generate_layer_heatmaps(self, net_name: str, output_path: Path, 
                                plot_layers: Optional[List[str]] = None,
                                plot_bin_size: Optional[int] = None,
                                anisotropic_bins: bool = False,
                                bin_aspect_ratio: int = 50,
                                layer_orientations: Optional[Dict[str, str]] = None,
                                output_filename: Optional[str] = None,
                                nominal_voltage: float = 1.0,
                                show_irdrop: bool = True):
        """
        Generate layer-wise heatmaps as one PNG per layer (2D grid mode).
        
        Args:
            net_name: Net name to generate heatmaps for
            output_path: Directory to save heatmaps
            plot_layers: List of layer IDs to plot. None = all layers
            plot_bin_size: Bin size for grid aggregation. None = auto-calculate
            anisotropic_bins: Enable orientation-aware anisotropic binning
            bin_aspect_ratio: Aspect ratio for anisotropic bins (default: 50)
            layer_orientations: Manual layer orientation overrides
            output_filename: Optional custom filename (deprecated, ignored for per-layer output)
            nominal_voltage: Nominal voltage for IR-drop calculation (default: 1.0 V)
            show_irdrop: If True, show IR-drop in mV (default). If False, show absolute voltage.
        """
        try:
            import matplotlib.pyplot as plt
            from math import ceil, sqrt
            import matplotlib
            matplotlib.set_loglevel('warning')
        except ImportError:
            self.logger.warning("matplotlib not available, skipping heatmaps")
            return
        
        layer_orientations = layer_orientations or {}
        
        # Get layers for this net
        layer_stats = self.graph.graph.get('layer_stats_by_net', {})
        net_layers = layer_stats.get(net_name, {})
        
        if not net_layers:
            self.logger.warning(f"  No layer information for net {net_name}")
            return
        
        # Filter out inter-layer entries and 'package' layer
        layers = [k for k in net_layers.keys() 
                  if k is not None and '-' not in str(k) and str(k).lower() != 'package']
        
        # Further filter by plot_layers if specified
        if plot_layers is not None:
            layers = [k for k in layers if str(k) in plot_layers]
        
        if not layers:
            return
        
        # Sort layers
        try:
            layers_sorted = sorted(layers, key=lambda x: int(x) if str(x).isdigit() else x)
        except:
            layers_sorted = sorted(layers)
        
        n_layers = len(layers_sorted)
        
        # Determine plot type label
        net_type = self._detect_net_type(net_name)
        if show_irdrop:
            plot_type = 'IR-Drop' if net_type == 'power' else 'Ground-Bounce'
            value_unit = 'mV'
            cmap = 'RdYlGn_r'  # Inverted: red=high drop (bad), green=low drop (good)
        else:
            plot_type = 'Voltage'
            value_unit = 'V'
            cmap = 'RdYlGn'  # Normal: red=low voltage, green=high voltage
        
        self.logger.info(f"  Generating {n_layers} layer {plot_type.lower()} heatmaps...")
        
        # Generate one PNG per layer
        for layer_id in layers_sorted:
            # Create single-layer figure
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # Get nodes for this layer and net
            net_nodes = self.net_connectivity.get(net_name, [])
            nodes_with_data = []
            
            for n in net_nodes:
                if n == '0':
                    continue
                
                d = self.graph.nodes[n]
                
                # Skip package nodes (no coordinates)
                if d.get('is_package', False):
                    continue
                
                # Check layer
                if d.get('layer') != layer_id:
                    continue
                
                # Extract data
                x = d.get('x')
                y = d.get('y')
                v = d.get('voltage')
                
                if x is not None and y is not None and v is not None:
                    nodes_with_data.append((n, x, y, v))
            
            if not nodes_with_data:
                plt.close(fig)
                self.logger.debug(f"    Layer {layer_id}: No data, skipping")
                continue
            
            # Unpack data efficiently
            _, xs_list, ys_list, voltages_list = zip(*nodes_with_data)
            xs = np.array(xs_list, dtype=np.float32)
            ys = np.array(ys_list, dtype=np.float32)
            voltages = np.array(voltages_list, dtype=np.float32)
            
            # Convert to IR-drop/ground-bounce in mV if requested
            if show_irdrop:
                if net_type == 'power':
                    # IR-drop for power nets: Vdd - V_node (positive = voltage dropped)
                    values = (nominal_voltage - voltages) * 1000.0  # mV
                else:
                    # Ground-bounce for ground nets: V_node - 0 (positive = voltage rise)
                    values = voltages * 1000.0  # mV
            else:
                values = voltages
            
            # Use provided bin size or auto-calculate
            x_min, x_max = xs.min(), xs.max()
            y_min, y_max = ys.min(), ys.max()
            
            if anisotropic_bins:
                # Anisotropic binning based on layer orientation
                net_nodes_set = set(self.net_connectivity.get(net_name, []))
                orientation = self._detect_layer_orientation(net_name, layer_id, net_nodes_set, layer_orientations)
                x_bins, y_bins, (x_range_val, y_range_val) = self._calculate_anisotropic_bins(
                    orientation, x_min, x_max, y_min, y_max, len(xs), bin_aspect_ratio,
                    base_size_override=plot_bin_size
                )
                # Calculate bin size (handle edge case of single bin)
                x_bin_size = x_bins[1] - x_bins[0] if len(x_bins) > 1 else x_range_val
                y_bin_size = y_bins[1] - y_bins[0] if len(y_bins) > 1 else y_range_val
                self.logger.debug(f"    Layer {layer_id}: orientation={orientation}, "
                                f"bins=({len(x_bins)-1}x{len(y_bins)-1}), "
                                f"bin_size=({x_bin_size:.1f}x{y_bin_size:.1f}), "
                                f"grid=({x_range_val:.0f}x{y_range_val:.0f})")
            elif plot_bin_size is not None:
                # Manual bin size - use square bins
                bin_size = plot_bin_size
                x_bins = np.arange(x_min, x_max + bin_size, bin_size)
                y_bins = np.arange(y_min, y_max + bin_size, bin_size)
                self.logger.debug(f"    Layer {layer_id}: using manual bin_size={bin_size} for {len(xs)} nodes")
            else:
                # Standard isotropic binning
                num_nodes = len(xs)
                avg_range = ((x_max - x_min) + (y_max - y_min)) / 2
                bin_size = max(1, int(avg_range / np.sqrt(num_nodes)))
                x_bins = np.arange(x_min, x_max + bin_size, bin_size)
                y_bins = np.arange(y_min, y_max + bin_size, bin_size)
                self.logger.debug(f"    Layer {layer_id}: using bin_size={bin_size} for {len(xs)} nodes")
            
            # Vectorized binning
            x_indices = np.digitize(xs, x_bins) - 1
            y_indices = np.digitize(ys, y_bins) - 1
            
            # Filter valid indices
            valid_mask = ((x_indices >= 0) & (x_indices < len(x_bins) - 1) & 
                         (y_indices >= 0) & (y_indices < len(y_bins) - 1))
            
            x_indices = x_indices[valid_mask]
            y_indices = y_indices[valid_mask]
            values_valid = values[valid_mask]
            
            # Aggregate using max for IR-drop/ground-bounce (worst case), min/max for voltage
            grid = np.full((len(y_bins) - 1, len(x_bins) - 1), np.nan)
            
            # Track worst-case per bin
            for i, (y_idx, x_idx, val) in enumerate(zip(y_indices, x_indices, values_valid)):
                current_val = grid[y_idx, x_idx]
                if np.isnan(current_val):
                    grid[y_idx, x_idx] = val
                else:
                    if show_irdrop:
                        # For IR-drop/ground-bounce: max is worst case
                        grid[y_idx, x_idx] = max(current_val, val)
                    elif net_type == 'power':
                        grid[y_idx, x_idx] = min(current_val, val)  # Min voltage for power
                    else:
                        grid[y_idx, x_idx] = max(current_val, val)  # Max voltage for ground
            
            # Get value range for this layer
            valid_values = values_valid[~np.isnan(values_valid)]
            if len(valid_values) > 0:
                vmin = valid_values.min()
                vmax = valid_values.max()
            else:
                vmin, vmax = 0, 1
            
            # Plot with appropriate colormap
            im = ax.imshow(grid, cmap=cmap, origin='lower',
                          extent=[x_bins[0], x_bins[-1], y_bins[0], y_bins[-1]],
                          aspect='auto', interpolation='nearest',
                          vmin=vmin, vmax=vmax)
            
            ax.set_title(f'{plot_type} - Net: {net_name} - Layer {layer_id} ({len(nodes_with_data)} nodes)')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            
            # Colorbar with appropriate label
            if show_irdrop:
                cbar_label = f'Max {plot_type} per Bin ({value_unit})'
            else:
                agg_label = 'Min' if net_type == 'power' else 'Max'
                cbar_label = f'{agg_label} Voltage per Bin ({value_unit})'
            plt.colorbar(im, ax=ax, label=cbar_label)
            
            plt.tight_layout()
            
            # Save per-layer file
            if show_irdrop:
                output_file = output_path / f'irdrop_heatmap_{net_name}_layer_{layer_id}.png'
            else:
                output_file = output_path / f'voltage_heatmap_{net_name}_layer_{layer_id}.png'
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            self.logger.info(f"    Saved: {output_file}")
    
    def generate_current_heatmaps(self, net_name: str, output_path: Path,
                                  plot_layers: Optional[List[str]] = None,
                                  plot_bin_size: Optional[int] = None,
                                  anisotropic_bins: bool = False,
                                  bin_aspect_ratio: int = 50,
                                  layer_orientations: Optional[Dict[str, str]] = None):
        """
        Generate current source heatmaps for layers with current sources (2D grid mode).
        
        Args:
            net_name: Net name to generate heatmaps for
            output_path: Directory to save heatmaps
            plot_layers: List of layer IDs to plot. None = all layers with current sources
            plot_bin_size: Bin size for grid aggregation. None = auto-calculate
            anisotropic_bins: Enable orientation-aware anisotropic binning
            bin_aspect_ratio: Aspect ratio for anisotropic bins (default: 50)
            layer_orientations: Manual layer orientation overrides
        """
        try:
            import matplotlib.pyplot as plt
            from math import ceil, sqrt
            import matplotlib
            matplotlib.set_loglevel('warning')
        except ImportError:
            self.logger.warning("matplotlib not available, skipping current heatmaps")
            return
        
        layer_orientations = layer_orientations or {}
        
        # Get net nodes as a set for faster lookup
        net_nodes_set = set(self.net_connectivity.get(net_name, []))
        
        # Find all current sources for this net and group by layer
        layer_currents = defaultdict(list)  # layer -> [(x, y, current_ma)]
        
        # Iterate only over nodes in this net
        for node in net_nodes_set:
            if node == '0':
                continue
            
            node_data = self.graph.nodes[node]
            
            # Skip package nodes (no coordinates)
            if node_data.get('is_package', False):
                continue
            
            # Check for current sources connected to this node
            for neighbor, edge_data in self.graph[node].items():
                # MultiDiGraph can have multiple edges
                for key, d in edge_data.items():
                    if d.get('type') == 'I':
                        layer = node_data.get('layer')
                        x = node_data.get('x')
                        y = node_data.get('y')
                        current_ma = d.get('value', 0)
                        
                        if layer is not None and x is not None and y is not None and current_ma != 0:
                            layer_currents[layer].append((x, y, current_ma))
        
        if not layer_currents:
            self.logger.info(f"  No current sources found for net {net_name}")
            return
        
        # Filter layers
        layers = list(layer_currents.keys())
        if plot_layers is not None:
            layers = [k for k in layers if str(k) in plot_layers]
        
        if not layers:
            return
        
        # Sort layers
        try:
            layers_sorted = sorted(layers, key=lambda x: int(x) if str(x).isdigit() else x)
        except:
            layers_sorted = sorted(layers)
        
        # Calculate subplot grid
        n_layers = len(layers_sorted)
        n_cols = ceil(sqrt(n_layers))
        n_rows = ceil(n_layers / n_cols)
        
        self.logger.info(f"  Generating {n_layers} current heatmaps...")
        
        # Create figure
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        if n_layers == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        # Plot each layer
        for idx, layer_id in enumerate(layers_sorted):
            ax = axes[idx]
            
            currents_data = layer_currents[layer_id]
            
            # Unpack data
            xs_list, ys_list, currents_list = zip(*currents_data)
            xs = np.array(xs_list, dtype=np.float32)
            ys = np.array(ys_list, dtype=np.float32)
            currents = np.array(currents_list, dtype=np.float32)
            
            # Use provided bin size or auto-calculate
            x_min, x_max = xs.min(), xs.max()
            y_min, y_max = ys.min(), ys.max()
            
            if anisotropic_bins:
                # Anisotropic binning based on layer orientation
                orientation = self._detect_layer_orientation(net_name, layer_id, net_nodes_set, layer_orientations)
                x_bins, y_bins, (x_range_val, y_range_val) = self._calculate_anisotropic_bins(
                    orientation, x_min, x_max, y_min, y_max, len(xs), bin_aspect_ratio,
                    base_size_override=plot_bin_size
                )
                # Calculate bin size (handle edge case of single bin)
                x_bin_size = x_bins[1] - x_bins[0] if len(x_bins) > 1 else x_range_val
                y_bin_size = y_bins[1] - y_bins[0] if len(y_bins) > 1 else y_range_val
                self.logger.debug(f"    Layer {layer_id}: orientation={orientation}, "
                                f"bins=({len(x_bins)-1}x{len(y_bins)-1}), "
                                f"bin_size=({x_bin_size:.1f}x{y_bin_size:.1f}), "
                                f"grid=({x_range_val:.0f}x{y_range_val:.0f})")
            elif plot_bin_size is not None:
                # Manual bin size - use square bins
                bin_size = plot_bin_size
                x_bins = np.arange(x_min, x_max + bin_size, bin_size)
                y_bins = np.arange(y_min, y_max + bin_size, bin_size)
                self.logger.debug(f"    Layer {layer_id}: using manual bin_size={bin_size} for {len(xs)} current sources")
            else:
                # Standard isotropic binning
                num_nodes = len(xs)
                avg_range = ((x_max - x_min) + (y_max - y_min)) / 2
                bin_size = max(1, int(avg_range / np.sqrt(num_nodes)))
                x_bins = np.arange(x_min, x_max + bin_size, bin_size)
                y_bins = np.arange(y_min, y_max + bin_size, bin_size)
                self.logger.debug(f"    Layer {layer_id}: using bin_size={bin_size} for {len(xs)} current sources")
            
            # Vectorized binning
            x_indices = np.digitize(xs, x_bins) - 1
            y_indices = np.digitize(ys, y_bins) - 1
            
            # Filter valid indices
            valid_mask = ((x_indices >= 0) & (x_indices < len(x_bins) - 1) & 
                         (y_indices >= 0) & (y_indices < len(y_bins) - 1))
            
            x_indices = x_indices[valid_mask]
            y_indices = y_indices[valid_mask]
            currents_valid = currents[valid_mask]
            
            # Aggregate currents (sum per bin)
            grid = np.zeros((len(y_bins) - 1, len(x_bins) - 1))
            
            np.add.at(grid, (y_indices, x_indices), currents_valid)
            
            # Mask empty bins
            grid = np.where(grid != 0, grid, np.nan)
            
            # Get current range
            valid_currents = currents_valid[~np.isnan(currents_valid)]
            if len(valid_currents) > 0:
                cmin = valid_currents.min()
                cmax = valid_currents.max()
            else:
                cmin, cmax = 0, 1
            
            # Plot with log scale for better visualization
            im = ax.imshow(grid, cmap='hot_r', origin='lower',
                          extent=[x_bins[0], x_bins[-1], y_bins[0], y_bins[-1]],
                          aspect='auto', interpolation='nearest',
                          vmin=cmin, vmax=cmax)
            
            ax.set_title(f'Layer {layer_id} ({len(currents_data)} sources)')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            
            # Colorbar
            plt.colorbar(im, ax=ax, label='Current (mA)')
        
        # Hide unused subplots
        for idx in range(n_layers, len(axes)):
            axes[idx].set_visible(False)
        
        # Overall title
        fig.suptitle(f'PDN Current Source Heatmaps - Net: {net_name}',
                    fontsize=16, y=0.995)
        
        plt.tight_layout()
        
        # Save
        output_file = output_path / f'current_heatmap_{net_name}.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"  Saved current heatmap: {output_file}")
    
    def generate_stripe_heatmaps(self, net_name: str, output_path: Path,
                                 plot_layers: Optional[List[str]] = None,
                                 max_stripes: int = 50,
                                 stripe_bin_size: Optional[int] = None,
                                 is_current: bool = False,
                                 layer_orientations: Optional[Dict[str, str]] = None,
                                 nominal_voltage: float = 1.0,
                                 show_irdrop: bool = True):
        """
        Generate stripe-based heatmaps as one PNG per layer.
        
        Nodes on each stripe are binified along the parallel dimension.
        Contiguous stripes are grouped together if count exceeds max_stripes.
        
        Args:
            net_name: Net name to generate heatmaps for
            output_path: Directory to save heatmaps
            plot_layers: List of layer IDs to plot. None = all layers
            max_stripes: Maximum number of stripes before grouping (default: 50)
            stripe_bin_size: Bin size for within-stripe aggregation. None = auto-calculate
            is_current: True for current heatmaps, False for voltage/IR-drop heatmaps
            layer_orientations: Manual layer orientation overrides
            nominal_voltage: Nominal voltage for IR-drop calculation (default: 1.0 V)
            show_irdrop: If True, show IR-drop in mV (default). If False, show absolute voltage.
        """
        import time
        t_start = time.time()
        enable_timing = self.logger.isEnabledFor(logging.DEBUG)
        
        try:
            import matplotlib.pyplot as plt
            from matplotlib.patches import Rectangle
            from matplotlib.collections import PatchCollection
            from math import ceil, sqrt
            import matplotlib
            matplotlib.set_loglevel('warning')
        except ImportError:
            self.logger.warning("matplotlib not available, skipping stripe heatmaps")
            return
        
        layer_orientations = layer_orientations or {}
        
        # Determine data source based on type
        if is_current:
            # Get current sources grouped by layer
            net_nodes_set = set(self.net_connectivity.get(net_name, []))
            layer_data = defaultdict(list)
            
            for node in net_nodes_set:
                if node == '0':
                    continue
                node_data = self.graph.nodes[node]
                if node_data.get('is_package', False):
                    continue
                
                for neighbor, edge_data in self.graph[node].items():
                    for key, d in edge_data.items():
                        if d.get('type') == 'I':
                            layer = node_data.get('layer')
                            x = node_data.get('x')
                            y = node_data.get('y')
                            current_ma = d.get('value', 0)
                            
                            if layer is not None and x is not None and y is not None and current_ma != 0:
                                layer_data[layer].append((node, x, y, current_ma))
            
            if not layer_data:
                self.logger.info(f"  No current sources found for net {net_name} (stripe mode)")
                return
                
            layers = list(layer_data.keys())
            data_type = "Current"
            cmap = 'hot_r'
            ylabel = 'Current (mA)'
            
        else:
            # Get voltage data grouped by layer
            layer_stats = self.graph.graph.get('layer_stats_by_net', {})
            net_layers = layer_stats.get(net_name, {})
            
            if not net_layers:
                self.logger.warning(f"  No layer information for net {net_name} (stripe mode)")
                return
            
            layers = [k for k in net_layers.keys() 
                     if k is not None and '-' not in str(k) and str(k).lower() != 'package']
            
            net_nodes = self.net_connectivity.get(net_name, [])
            layer_data = defaultdict(list)
            
            for n in net_nodes:
                if n == '0':
                    continue
                d = self.graph.nodes[n]
                if d.get('is_package', False):
                    continue
                
                layer = d.get('layer')
                x = d.get('x')
                y = d.get('y')
                v = d.get('voltage')
                
                if layer is not None and x is not None and y is not None and v is not None:
                    layer_data[layer].append((n, x, y, v))
            
            layers = [k for k in layers if k in layer_data and layer_data[k]]
            
            # Determine data type and colormap based on show_irdrop
            net_type = self._detect_net_type(net_name)
            if show_irdrop:
                data_type = 'IR-Drop' if net_type == 'power' else 'Ground-Bounce'
                cmap = 'RdYlGn_r'  # Inverted: red=high drop (bad), green=low drop (good)
                ylabel = f'{data_type} (mV)'
            else:
                data_type = "Voltage"
                cmap = 'RdYlGn'
                ylabel = 'Voltage (V)'
        
        # Filter by plot_layers
        if plot_layers is not None:
            layers = [k for k in layers if str(k) in plot_layers]
        
        if not layers:
            return
        
        # Sort layers
        try:
            layers_sorted = sorted(layers, key=lambda x: int(x) if str(x).isdigit() else x)
        except:
            layers_sorted = sorted(layers)
        
        n_layers = len(layers_sorted)
        
        if enable_timing:
            t_data = time.time()
            self.logger.debug(f"  [TIMING] Data collection: {t_data - t_start:.2f}s")
        
        self.logger.info(f"  Generating {n_layers} stripe-based {data_type.lower()} heatmaps...")
        
        # Get net nodes set for orientation detection
        net_nodes_set = set(self.net_connectivity.get(net_name, []))
        
        # Detect net type for IR-drop conversion (only needed for voltage mode)
        if not is_current:
            net_type_for_irdrop = self._detect_net_type(net_name)
        else:
            net_type_for_irdrop = 'power'  # Not used for current
        
        # Generate one PNG per layer
        for layer_id in layers_sorted:
            if enable_timing:
                t_layer_start = time.time()
            
            # Create single-layer figure
            fig, ax = plt.subplots(figsize=(10, 8))
            
            nodes_with_data = layer_data[layer_id]
            
            if not nodes_with_data:
                plt.close(fig)
                self.logger.debug(f"    Layer {layer_id}: No data, skipping")
                continue
            
            # Detect orientation
            orientation = self._detect_layer_orientation(net_name, layer_id, net_nodes_set, layer_orientations)
            
            # Fall back to 2D grid for MIXED layers
            if orientation == 'MIXED':
                plt.close(fig)
                self.logger.debug(f"    Layer {layer_id}: MIXED orientation, skipping stripe mode")
                continue
            
            # Group nodes into stripes
            stripes = self._group_nodes_into_stripes(net_name, layer_id, orientation, nodes_with_data)
            
            # Consolidate stripes if needed
            consolidated_stripes = self._consolidate_stripes(stripes, max_stripes)
            
            num_original_stripes = len(stripes)
            num_display_stripes = len(consolidated_stripes)
            
            if enable_timing:
                t_stripe_group = time.time()
                self.logger.debug(f"    Layer {layer_id} [TIMING] Stripe grouping: {t_stripe_group - t_layer_start:.2f}s")
            
            self.logger.debug(f"    Layer {layer_id}: orientation={orientation}, "
                            f"original_stripes={num_original_stripes}, "
                            f"display_stripes={num_display_stripes}")
            
            # Extract all coordinates for range calculation
            all_xs = [x for _, x, _, _ in nodes_with_data]
            all_ys = [y for _, _, y, _ in nodes_with_data]
            layer_x_min, layer_x_max = min(all_xs), max(all_xs)
            layer_y_min, layer_y_max = min(all_ys), max(all_ys)
            
            # Plot stripes - store bin-level data for proper visualization
            stripe_data = []  # List of (start_coord, end_coord, bins, bin_values)
            all_bin_values = []  # For color scale calculation
            
            for start_coord, end_coord, stripe_nodes in consolidated_stripes:
                if not stripe_nodes:
                    continue
                
                # Extract coordinates and values for this stripe
                xs_stripe = np.array([x for _, x, _, _ in stripe_nodes])
                ys_stripe = np.array([y for _, _, y, _ in stripe_nodes])
                values_stripe = np.array([v for _, _, _, v in stripe_nodes])
                
                # Calculate stripe bounds
                stripe_x_min, stripe_x_max = xs_stripe.min(), xs_stripe.max()
                stripe_y_min, stripe_y_max = ys_stripe.min(), ys_stripe.max()
                
                # Calculate bins for parallel dimension
                bins = self._calculate_stripe_bins(stripe_nodes, orientation,
                                                   stripe_x_min, stripe_x_max,
                                                   stripe_y_min, stripe_y_max,
                                                   stripe_bin_size)
                
                # Safety check for excessive bin counts
                if len(bins) > 100000:
                    if enable_timing:
                        self.logger.warning(f"    Layer {layer_id}: Excessive bins ({len(bins)}) for stripe, skipping")
                    continue
                
                # Bin nodes along parallel dimension and aggregate
                if orientation == 'H':
                    # Horizontal stripe: bin along X
                    indices = np.digitize(xs_stripe, bins) - 1
                    valid_mask = (indices >= 0) & (indices < len(bins) - 1)
                    
                    if not is_current:
                        # For voltage/IR-drop: aggregate per bin
                        bin_values = np.full(len(bins) - 1, np.nan)
                        
                        # Aggregate per bin (np.minimum/maximum.at don't work with NaN)
                        for idx in range(len(bins) - 1):
                            mask = indices[valid_mask] == idx
                            if np.any(mask):
                                raw_values = values_stripe[valid_mask][mask]
                                if show_irdrop:
                                    # Convert to IR-drop/ground-bounce in mV
                                    if net_type_for_irdrop == 'power':
                                        converted = (nominal_voltage - raw_values) * 1000.0
                                    else:
                                        converted = raw_values * 1000.0
                                    # Max IR-drop is worst case
                                    bin_values[idx] = np.max(converted)
                                else:
                                    # Voltage mode: min for power, max for ground
                                    if net_type_for_irdrop == 'power':
                                        bin_values[idx] = np.min(raw_values)
                                    else:
                                        bin_values[idx] = np.max(raw_values)
                    else:
                        # For current: sum is correct
                        bin_values = np.zeros(len(bins) - 1)
                        np.add.at(bin_values, indices[valid_mask], values_stripe[valid_mask])
                    
                else:  # orientation == 'V'
                    # Vertical stripe: bin along Y
                    indices = np.digitize(ys_stripe, bins) - 1
                    valid_mask = (indices >= 0) & (indices < len(bins) - 1)
                    
                    if not is_current:
                        # For voltage/IR-drop: aggregate per bin
                        bin_values = np.full(len(bins) - 1, np.nan)
                        
                        # Aggregate per bin (np.minimum/maximum.at don't work with NaN)
                        for idx in range(len(bins) - 1):
                            mask = indices[valid_mask] == idx
                            if np.any(mask):
                                raw_values = values_stripe[valid_mask][mask]
                                if show_irdrop:
                                    # Convert to IR-drop/ground-bounce in mV
                                    if net_type_for_irdrop == 'power':
                                        converted = (nominal_voltage - raw_values) * 1000.0
                                    else:
                                        converted = raw_values * 1000.0
                                    # Max IR-drop is worst case
                                    bin_values[idx] = np.max(converted)
                                else:
                                    # Voltage mode: min for power, max for ground
                                    if net_type_for_irdrop == 'power':
                                        bin_values[idx] = np.min(raw_values)
                                    else:
                                        bin_values[idx] = np.max(raw_values)
                    else:
                        # For current: sum is correct
                        bin_values = np.zeros(len(bins) - 1)
                        np.add.at(bin_values, indices[valid_mask], values_stripe[valid_mask])
                
                # Store stripe data with bin-level detail
                if not np.all(np.isnan(bin_values)):
                    stripe_data.append((start_coord, end_coord, bins, bin_values))
                    all_bin_values.extend([v for v in bin_values if not np.isnan(v)])
            
            if enable_timing:
                t_bin_agg = time.time()
                self.logger.debug(f"    Layer {layer_id} [TIMING] Bin aggregation: {t_bin_agg - t_stripe_group:.2f}s, bins={len(all_bin_values)}")
            
            # Debug: log stripe value range for layer 19
            if not is_current and str(layer_id) == '19' and all_bin_values:
                self.logger.info(f"    Layer 19 bin values: min={min(all_bin_values):.6f}, max={max(all_bin_values):.6f}, count={len(all_bin_values)}")
            
            # Use layer-specific range for better contrast
            layer_vmin = min(all_bin_values) if all_bin_values else 0
            layer_vmax = max(all_bin_values) if all_bin_values else 1
            
            if not is_current:
                if show_irdrop:
                    self.logger.info(f"    Layer {layer_id} {data_type.lower()} range: [{layer_vmin:.3f}, {layer_vmax:.3f}] mV")
                else:
                    self.logger.info(f"    Layer {layer_id} voltage range: [{layer_vmin:.6f}, {layer_vmax:.6f}] V")
            
            # Create visualization using PatchCollection for better performance
            norm = plt.Normalize(vmin=layer_vmin, vmax=layer_vmax)
            cmap_obj = plt.get_cmap(cmap)
            
            patches = []
            colors = []
            
            if orientation == 'H':
                # Horizontal stripes: collect all rectangles and colors
                for start_coord, end_coord, bins, bin_values in stripe_data:
                    stripe_height = end_coord - start_coord
                    for i, (bin_start, bin_end) in enumerate(zip(bins[:-1], bins[1:])):
                        if not np.isnan(bin_values[i]):
                            bin_width = bin_end - bin_start
                            rect = Rectangle((bin_start, start_coord), bin_width, stripe_height)
                            patches.append(rect)
                            colors.append(bin_values[i])
                
                ax.set_xlim(layer_x_min, layer_x_max)
                ax.set_ylim(layer_y_min, layer_y_max)
                ax.set_xlabel('X')
                ax.set_ylabel('Y (Stripe Position)')
                
            else:  # orientation == 'V'
                # Vertical stripes: collect all rectangles and colors
                for start_coord, end_coord, bins, bin_values in stripe_data:
                    stripe_width = end_coord - start_coord
                    for i, (bin_start, bin_end) in enumerate(zip(bins[:-1], bins[1:])):
                        if not np.isnan(bin_values[i]):
                            bin_height = bin_end - bin_start
                            rect = Rectangle((start_coord, bin_start), stripe_width, bin_height)
                            patches.append(rect)
                            colors.append(bin_values[i])
            
            # Add all patches at once using PatchCollection for much better performance
            if patches:
                if enable_timing:
                    t_patches_create = time.time()
                    self.logger.debug(f"    Layer {layer_id} [TIMING] Create patches: {t_patches_create - t_bin_agg:.2f}s, count={len(patches)}")
                
                pc = PatchCollection(patches, cmap=cmap, norm=norm, edgecolors='none')
                pc.set_array(np.array(colors))
                ax.add_collection(pc)
                
                if enable_timing:
                    t_patches_add = time.time()
                    self.logger.debug(f"    Layer {layer_id} [TIMING] Add patches to axes: {t_patches_add - t_patches_create:.2f}s")
                
                ax.set_xlim(layer_x_min, layer_x_max)
                ax.set_ylim(layer_y_min, layer_y_max)
                ax.set_xlabel('X (Stripe Position)')
                ax.set_ylabel('Y')
            
            # Add colorbar with aggregation method label
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            if not is_current:
                if show_irdrop:
                    cbar_label = f'Max {data_type} per Stripe (mV)'
                else:
                    agg_label = 'Min' if net_type_for_irdrop == 'power' else 'Max'
                    cbar_label = f'{agg_label} Voltage per Stripe (V)'
            else:
                cbar_label = f'Total {ylabel} per Stripe'
            plt.colorbar(sm, ax=ax, label=cbar_label)
            
            ax.set_title(f'{data_type} - Net: {net_name} - Layer {layer_id} ({num_original_stripes}→{num_display_stripes} stripes)')
            ax.set_aspect('equal', adjustable='box')
            
            plt.tight_layout()
            
            # Save per-layer file
            if is_current:
                output_file = output_path / f'current_stripe_heatmap_{net_name}_layer_{layer_id}.png'
            elif show_irdrop:
                output_file = output_path / f'irdrop_stripe_heatmap_{net_name}_layer_{layer_id}.png'
            else:
                output_file = output_path / f'voltage_stripe_heatmap_{net_name}_layer_{layer_id}.png'
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            self.logger.info(f"    Saved: {output_file}")
            
            if enable_timing:
                t_layer_end = time.time()
                self.logger.debug(f"    Layer {layer_id} [TIMING] Total layer: {t_layer_end - t_layer_start:.2f}s")
        
        if enable_timing:
            t_total = time.time()
            self.logger.debug(f"  [TIMING] TOTAL: {t_total - t_start:.2f}s")

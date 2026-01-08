"""Unified plotting for power grid analysis.

Provides plotting functions that work with both synthetic and PDN power grids
via the UnifiedPowerGridModel interface.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .unified_model import UnifiedPowerGridModel, LayerID

# Use non-interactive backend by default
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes


class UnifiedPlotter:
    """Unified plotter for power grid visualization.

    Works with any UnifiedPowerGridModel to create:
    - Voltage heatmaps
    - IR-drop heatmaps
    - Current flow maps (edge-based)

    Example:
        model = create_model_from_synthetic(G, pads, vdd=1.0)
        solver = UnifiedIRDropSolver(model)
        result = solver.solve(loads)

        plotter = UnifiedPlotter(model)
        fig, ax = plotter.plot_voltage_heatmap(result.voltages)
    """

    def __init__(self, model: UnifiedPowerGridModel):
        """Initialize plotter with a unified model.

        Args:
            model: UnifiedPowerGridModel instance
        """
        self.model = model

    def _collect_node_data(
        self,
        values: Dict[Any, float],
        layer: Optional[LayerID] = None,
    ) -> Tuple[List[Any], np.ndarray, np.ndarray, np.ndarray]:
        """Collect nodes with coordinates and values for plotting.

        Args:
            values: Node -> value mapping (voltage, IR-drop, etc.)
            layer: Optional layer filter

        Returns:
            (nodes, xs, ys, vals) arrays for plotting
        """
        nodes, xs, ys, vals = [], [], [], []

        for node, value in values.items():
            info = self.model.get_node_info(node)

            # Layer filter
            if layer is not None:
                node_layer = info.layer
                if node_layer != layer and str(node_layer) != str(layer):
                    continue

            xy = info.xy
            if xy is None:
                continue

            nodes.append(node)
            xs.append(xy[0])
            ys.append(xy[1])
            vals.append(value)

        return nodes, np.array(xs), np.array(ys), np.array(vals)

    def plot_voltage_heatmap(
        self,
        voltages: Dict[Any, float],
        layer: Optional[LayerID] = None,
        ax: Optional[Axes] = None,
        cmap: str = 'RdYlGn',
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        title: Optional[str] = None,
        show: bool = False,
        **scatter_kwargs,
    ) -> Tuple[Figure, Axes]:
        """Plot voltage heatmap.

        Args:
            voltages: Node -> voltage mapping
            layer: Optional layer filter
            ax: Matplotlib axes (created if None)
            cmap: Colormap name
            vmin: Minimum value for colormap
            vmax: Maximum value for colormap
            title: Plot title
            show: Whether to display the plot
            **scatter_kwargs: Additional kwargs for scatter plot

        Returns:
            (fig, ax) matplotlib objects
        """
        nodes, xs, ys, vs = self._collect_node_data(voltages, layer)

        if len(xs) == 0:
            raise ValueError("No nodes with coordinates found for plotting")

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        else:
            fig = ax.figure

        # Set defaults
        kwargs = {'s': 40, 'edgecolors': 'k', 'linewidths': 0.3}
        kwargs.update(scatter_kwargs)

        sc = ax.scatter(xs, ys, c=vs, cmap=cmap, vmin=vmin, vmax=vmax, **kwargs)
        ax.set_aspect('equal', adjustable='box')

        # Title
        layer_str = f" (Layer {layer})" if layer is not None else ""
        net_str = f" - {self.model.net_name}" if self.model.net_name else ""
        ax.set_title(title or f"Voltage{layer_str}{net_str}")

        plt.colorbar(sc, ax=ax, label='Voltage (V)')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')

        plt.tight_layout()

        if show:
            plt.show()

        return fig, ax

    def plot_ir_drop_heatmap(
        self,
        voltages: Dict[Any, float],
        vdd: Optional[float] = None,
        layer: Optional[LayerID] = None,
        ax: Optional[Axes] = None,
        cmap: str = 'hot_r',
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        title: Optional[str] = None,
        show: bool = False,
        **scatter_kwargs,
    ) -> Tuple[Figure, Axes]:
        """Plot IR-drop heatmap.

        Args:
            voltages: Node -> voltage mapping
            vdd: Reference voltage (defaults to model.vdd)
            layer: Optional layer filter
            ax: Matplotlib axes (created if None)
            cmap: Colormap name
            vmin: Minimum value for colormap
            vmax: Maximum value for colormap
            title: Plot title
            show: Whether to display the plot
            **scatter_kwargs: Additional kwargs for scatter plot

        Returns:
            (fig, ax) matplotlib objects
        """
        if vdd is None:
            vdd = self.model.vdd

        # Convert voltages to IR-drop
        ir_drop = {n: vdd - v for n, v in voltages.items()}

        nodes, xs, ys, drops = self._collect_node_data(ir_drop, layer)

        if len(xs) == 0:
            raise ValueError("No nodes with coordinates found for plotting")

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        else:
            fig = ax.figure

        # Set defaults
        kwargs = {'s': 40, 'edgecolors': 'k', 'linewidths': 0.3}
        kwargs.update(scatter_kwargs)

        sc = ax.scatter(xs, ys, c=drops, cmap=cmap, vmin=vmin, vmax=vmax, **kwargs)
        ax.set_aspect('equal', adjustable='box')

        # Title
        layer_str = f" (Layer {layer})" if layer is not None else ""
        net_str = f" - {self.model.net_name}" if self.model.net_name else ""
        ax.set_title(title or f"IR-Drop{layer_str}{net_str}")

        plt.colorbar(sc, ax=ax, label='IR-Drop (V)')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')

        plt.tight_layout()

        if show:
            plt.show()

        return fig, ax

    def plot_current_map(
        self,
        voltages: Dict[Any, float],
        layer: Optional[LayerID] = None,
        ax: Optional[Axes] = None,
        cmap: str = 'viridis',
        title: Optional[str] = None,
        show: bool = False,
        **scatter_kwargs,
    ) -> Tuple[Figure, Axes]:
        """Plot current flow map based on edge currents.

        Current is computed as I = (V_u - V_v) / R for each resistive edge.

        Args:
            voltages: Node -> voltage mapping
            layer: Optional layer filter
            ax: Matplotlib axes (created if None)
            cmap: Colormap name
            title: Plot title
            show: Whether to display the plot
            **scatter_kwargs: Additional kwargs for scatter plot

        Returns:
            (fig, ax) matplotlib objects
        """
        # Compute edge currents
        edge_currents = []

        for u, v, edge_info in self.model._iter_resistive_edges():
            R = edge_info.resistance
            if R is None or R <= 0:
                continue

            u_info = self.model.get_node_info(u)
            v_info = self.model.get_node_info(v)

            # Layer filter
            if layer is not None:
                if u_info.layer != layer and v_info.layer != layer:
                    if str(u_info.layer) != str(layer) and str(v_info.layer) != str(layer):
                        continue

            xy_u = u_info.xy
            xy_v = v_info.xy
            if xy_u is None or xy_v is None:
                continue

            V_u = voltages.get(u, self.model.vdd)
            V_v = voltages.get(v, self.model.vdd)
            I = abs(V_u - V_v) / R

            # Midpoint for visualization
            mid_x = (xy_u[0] + xy_v[0]) / 2
            mid_y = (xy_u[1] + xy_v[1]) / 2

            edge_currents.append((mid_x, mid_y, I))

        if not edge_currents:
            raise ValueError("No edges found for current map")

        xs = np.array([e[0] for e in edge_currents])
        ys = np.array([e[1] for e in edge_currents])
        currents = np.array([e[2] for e in edge_currents])

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        else:
            fig = ax.figure

        # Set defaults
        kwargs = {'s': 30, 'edgecolors': 'none'}
        kwargs.update(scatter_kwargs)

        sc = ax.scatter(xs, ys, c=currents, cmap=cmap, **kwargs)
        ax.set_aspect('equal', adjustable='box')

        # Title
        layer_str = f" (Layer {layer})" if layer is not None else ""
        net_str = f" - {self.model.net_name}" if self.model.net_name else ""
        ax.set_title(title or f"Current Flow{layer_str}{net_str}")

        plt.colorbar(sc, ax=ax, label='Current (A)')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')

        plt.tight_layout()

        if show:
            plt.show()

        return fig, ax

    def generate_layer_heatmaps(
        self,
        voltages: Dict[Any, float],
        output_dir: Path,
        heatmap_type: str = 'voltage',
        file_format: str = 'png',
        dpi: int = 150,
    ) -> List[Path]:
        """Generate heatmaps for all layers.

        Args:
            voltages: Node -> voltage mapping
            output_dir: Directory to save files
            heatmap_type: 'voltage' or 'ir_drop'
            file_format: Image format (png, pdf, etc.)
            dpi: Resolution for raster formats

        Returns:
            List of output file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        layers = self.model.get_all_layers()
        output_files = []

        for layer in layers:
            filename = f"{heatmap_type}_layer_{layer}.{file_format}"
            output_path = output_dir / filename

            fig, ax = plt.subplots(figsize=(8, 6))

            try:
                if heatmap_type == 'voltage':
                    self.plot_voltage_heatmap(voltages, layer=layer, ax=ax)
                elif heatmap_type == 'ir_drop':
                    self.plot_ir_drop_heatmap(voltages, layer=layer, ax=ax)
                else:
                    raise ValueError(f"Unknown heatmap type: {heatmap_type}")

                fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
                output_files.append(output_path)
            except ValueError:
                # No nodes at this layer
                pass
            finally:
                plt.close(fig)

        return output_files


# Convenience functions matching the original irdrop/plot.py interface

def plot_voltage_map(
    G,
    voltages: Dict[Any, float],
    layer: Optional[LayerID] = None,
    ax: Optional[Axes] = None,
    show: bool = True,
    **kwargs,
) -> Tuple[Figure, Axes]:
    """Plot voltage heatmap (compatibility function).

    This function provides backward compatibility with the original
    irdrop/plot.py interface.

    Args:
        G: NetworkX graph
        voltages: Node -> voltage mapping
        layer: Optional layer filter
        ax: Matplotlib axes
        show: Whether to display the plot
        **kwargs: Additional kwargs for scatter plot

    Returns:
        (fig, ax) matplotlib objects
    """
    from .unified_model import UnifiedPowerGridModel, GridSource

    # Create a minimal model for plotting
    model = UnifiedPowerGridModel(
        graph=G,
        pad_nodes=[],  # Not needed for plotting
        vdd=1.0,
        source=GridSource.SYNTHETIC,
    )

    plotter = UnifiedPlotter(model)
    return plotter.plot_voltage_heatmap(voltages, layer=layer, ax=ax, show=show, **kwargs)


def plot_ir_drop_map(
    G,
    voltages: Dict[Any, float],
    vdd: float,
    layer: Optional[LayerID] = None,
    ax: Optional[Axes] = None,
    show: bool = True,
    **kwargs,
) -> Tuple[Figure, Axes]:
    """Plot IR-drop heatmap (compatibility function).

    This function provides backward compatibility with the original
    irdrop/plot.py interface.

    Args:
        G: NetworkX graph
        voltages: Node -> voltage mapping
        vdd: Reference voltage (must be scalar float, NOT a list of pads)
        layer: Optional layer filter
        ax: Matplotlib axes
        show: Whether to display the plot
        **kwargs: Additional kwargs for scatter plot

    Returns:
        (fig, ax) matplotlib objects
    """
    from .unified_model import UnifiedPowerGridModel, GridSource

    # Create a minimal model for plotting
    model = UnifiedPowerGridModel(
        graph=G,
        pad_nodes=[],
        vdd=float(vdd),
        source=GridSource.SYNTHETIC,
    )

    plotter = UnifiedPlotter(model)
    return plotter.plot_ir_drop_heatmap(voltages, vdd=vdd, layer=layer, ax=ax, show=show, **kwargs)

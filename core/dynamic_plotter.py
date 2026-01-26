"""Plotting utilities for dynamic IR-drop analysis results.

Provides heatmap and time series plotting for quasi-static and transient
analysis results.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .unified_model import UnifiedPowerGridModel, LayerID
from .dynamic_solver import QuasiStaticResult
from .transient_solver import TransientResult

# Use non-interactive backend by default
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes


class DynamicPlotter:
    """Plotting utilities for dynamic IR-drop analysis results.

    Provides:
    - Peak IR-drop heatmaps (spatial max across time)
    - Peak current heatmaps
    - Time series plots of aggregate metrics

    Example:
        from core import DynamicPlotter

        result = solver.solve_transient(...)

        # Plot peak IR-drop heatmap
        DynamicPlotter.plot_peak_ir_drop_heatmap(
            model, result, layer='M1',
            title='Peak IR-Drop During Transient'
        )

        # Plot time series
        DynamicPlotter.plot_time_series(
            result, metrics=['max_ir_drop', 'vsrc_current']
        )
    """

    @staticmethod
    def _collect_node_data(
        model: UnifiedPowerGridModel,
        values: Dict[Any, float],
        layer: Optional[LayerID] = None,
    ) -> Tuple[List[Any], np.ndarray, np.ndarray, np.ndarray]:
        """Collect nodes with coordinates and values for plotting.

        Args:
            model: UnifiedPowerGridModel instance
            values: Node -> value mapping
            layer: Optional layer filter

        Returns:
            (nodes, xs, ys, vals) arrays for plotting
        """
        nodes, xs, ys, vals = [], [], [], []

        for node, value in values.items():
            info = model.get_node_info(node)

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

    @staticmethod
    def plot_peak_ir_drop_heatmap(
        model: UnifiedPowerGridModel,
        result: Union[QuasiStaticResult, TransientResult],
        layer: Optional[LayerID] = None,
        ax: Optional[Axes] = None,
        cmap: str = 'RdYlGn_r',
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        title: str = "Peak IR-Drop Heatmap",
        show: bool = False,
        save_path: Optional[str] = None,
        unit_scale: float = 1000.0,  # Convert V to mV by default
        unit_label: str = "mV",
        **scatter_kwargs,
    ) -> Tuple[Figure, Axes]:
        """Plot heatmap of peak IR-drop at each node across all time.

        Uses result.peak_ir_drop_per_node which tracks max IR-drop
        each node experienced during simulation.

        Args:
            model: UnifiedPowerGridModel instance
            result: QuasiStaticResult or TransientResult
            layer: Optional layer filter (e.g., 'M1')
            ax: Matplotlib axes (created if None)
            cmap: Colormap name (default 'RdYlGn_r' for red=high drop)
            vmin: Minimum value for colormap
            vmax: Maximum value for colormap
            title: Plot title
            show: Whether to display the plot
            save_path: Path to save figure (None = don't save)
            unit_scale: Scale factor for values (default 1000 for V->mV)
            unit_label: Unit label for colorbar
            **scatter_kwargs: Additional kwargs for scatter plot

        Returns:
            (fig, ax) matplotlib objects
        """
        values = result.peak_ir_drop_per_node
        nodes, xs, ys, vals = DynamicPlotter._collect_node_data(model, values, layer)

        if len(xs) == 0:
            raise ValueError("No nodes with coordinates found for plotting")

        # Scale values
        vals_scaled = vals * unit_scale

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        else:
            fig = ax.figure

        # Set defaults
        kwargs = {'s': 40, 'edgecolors': 'k', 'linewidths': 0.3}
        kwargs.update(scatter_kwargs)

        sc = ax.scatter(xs, ys, c=vals_scaled, cmap=cmap, vmin=vmin, vmax=vmax, **kwargs)
        ax.set_aspect('equal', adjustable='box')

        # Title
        layer_str = f" (Layer {layer})" if layer is not None else ""
        ax.set_title(f"{title}{layer_str}")

        plt.colorbar(sc, ax=ax, label=f'Peak IR-Drop ({unit_label})')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')

        if show:
            plt.show()

        return fig, ax

    @staticmethod
    def plot_peak_current_heatmap(
        model: UnifiedPowerGridModel,
        result: Union[QuasiStaticResult, TransientResult],
        layer: Optional[LayerID] = None,
        ax: Optional[Axes] = None,
        cmap: str = 'hot_r',
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        title: str = "Peak Current Heatmap",
        show: bool = False,
        save_path: Optional[str] = None,
        unit_label: str = "mA",
        **scatter_kwargs,
    ) -> Tuple[Figure, Axes]:
        """Plot heatmap of peak current at each node across all time.

        Uses result.peak_current_per_node which tracks max current
        each node experienced during simulation.

        Args:
            model: UnifiedPowerGridModel instance
            result: QuasiStaticResult or TransientResult
            layer: Optional layer filter (e.g., 'M1')
            ax: Matplotlib axes (created if None)
            cmap: Colormap name (default 'hot_r')
            vmin: Minimum value for colormap
            vmax: Maximum value for colormap
            title: Plot title
            show: Whether to display the plot
            save_path: Path to save figure (None = don't save)
            unit_label: Unit label for colorbar
            **scatter_kwargs: Additional kwargs for scatter plot

        Returns:
            (fig, ax) matplotlib objects
        """
        values = result.peak_current_per_node
        nodes, xs, ys, vals = DynamicPlotter._collect_node_data(model, values, layer)

        if len(xs) == 0:
            raise ValueError("No nodes with coordinates found for plotting")

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        else:
            fig = ax.figure

        # Set defaults
        kwargs = {'s': 40, 'edgecolors': 'k', 'linewidths': 0.3}
        kwargs.update(scatter_kwargs)

        sc = ax.scatter(xs, ys, c=vals, cmap=cmap, vmin=vmin, vmax=vmax, **kwargs)
        ax.set_aspect('equal', adjustable='box')

        # Title
        layer_str = f" (Layer {layer})" if layer is not None else ""
        ax.set_title(f"{title}{layer_str}")

        plt.colorbar(sc, ax=ax, label=f'Peak Current ({unit_label})')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')

        if show:
            plt.show()

        return fig, ax

    @staticmethod
    def plot_time_series(
        result: Union[QuasiStaticResult, TransientResult],
        metrics: Optional[List[str]] = None,
        ax: Optional[Axes] = None,
        title: str = "Dynamic IR-Drop Time Series",
        show: bool = False,
        save_path: Optional[str] = None,
        time_scale: float = 1e9,  # Convert s to ns by default
        time_label: str = "ns",
    ) -> Tuple[Figure, Union[Axes, List[Axes]]]:
        """Plot time series of aggregate metrics.

        Available metrics:
        - 'max_ir_drop': max IR-drop across all nodes per time (mV)
        - 'total_current': total load current per time (mA)
        - 'vsrc_current': total current through voltage sources (mA)

        Args:
            result: QuasiStaticResult or TransientResult
            metrics: List of metrics to plot (default: all available)
            ax: Matplotlib axes (created if None, may create subplots)
            title: Plot title
            show: Whether to display the plot
            save_path: Path to save figure (None = don't save)
            time_scale: Scale factor for time axis (default 1e9 for s->ns)
            time_label: Unit label for time axis

        Returns:
            (fig, axes) matplotlib objects
        """
        # Default metrics
        if metrics is None:
            metrics = ['max_ir_drop', 'total_current', 'vsrc_current']

        # Map metric names to data and labels
        metric_data = {
            'max_ir_drop': (result.max_ir_drop_per_time * 1000, 'Max IR-Drop (mV)', 'tab:red'),
            'total_current': (result.total_current_per_time, 'Total Load Current (mA)', 'tab:blue'),
            'vsrc_current': (result.total_vsrc_current_per_time, 'Vsrc Current (mA)', 'tab:green'),
        }

        # Filter to requested metrics
        plot_metrics = [m for m in metrics if m in metric_data]
        if not plot_metrics:
            raise ValueError(f"No valid metrics found. Available: {list(metric_data.keys())}")

        n_plots = len(plot_metrics)
        t_scaled = result.t_array * time_scale

        if ax is None:
            fig, axes = plt.subplots(n_plots, 1, figsize=(10, 3 * n_plots), sharex=True)
            if n_plots == 1:
                axes = [axes]
        else:
            fig = ax.figure
            axes = [ax]

        for i, metric in enumerate(plot_metrics):
            data, ylabel, color = metric_data[metric]
            ax_i = axes[i] if i < len(axes) else axes[-1]

            ax_i.plot(t_scaled, data, color=color, linewidth=1.5)
            ax_i.set_ylabel(ylabel)
            ax_i.grid(True, alpha=0.3)

            if i == 0:
                ax_i.set_title(title)

        # Set common x label on last axis
        axes[-1].set_xlabel(f'Time ({time_label})')

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')

        if show:
            plt.show()

        return fig, axes if n_plots > 1 else axes[0]

    @staticmethod
    def plot_node_waveforms(
        result: Union[QuasiStaticResult, TransientResult],
        nodes: Optional[List[Any]] = None,
        plot_ir_drop: bool = True,
        ax: Optional[Axes] = None,
        title: str = "Node Voltage Waveforms",
        show: bool = False,
        save_path: Optional[str] = None,
        time_scale: float = 1e9,
        time_label: str = "ns",
    ) -> Tuple[Figure, Axes]:
        """Plot voltage or IR-drop waveforms for tracked nodes.

        Args:
            result: QuasiStaticResult or TransientResult
            nodes: List of nodes to plot (default: all tracked)
            plot_ir_drop: If True plot IR-drop, else plot voltage
            ax: Matplotlib axes (created if None)
            title: Plot title
            show: Whether to display the plot
            save_path: Path to save figure (None = don't save)
            time_scale: Scale factor for time axis
            time_label: Unit label for time axis

        Returns:
            (fig, ax) matplotlib objects
        """
        # Get waveforms
        if plot_ir_drop:
            waveforms = result.tracked_ir_drop
            ylabel = 'IR-Drop (mV)'
            scale = 1000.0
        else:
            waveforms = result.tracked_waveforms
            ylabel = 'Voltage (V)'
            scale = 1.0

        if not waveforms:
            raise ValueError("No tracked waveforms available. Use track_nodes parameter.")

        # Select nodes
        if nodes is None:
            nodes = list(waveforms.keys())
        nodes = [n for n in nodes if n in waveforms]

        if not nodes:
            raise ValueError("None of the requested nodes have waveforms")

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))
        else:
            fig = ax.figure

        t_scaled = result.t_array * time_scale

        for node in nodes:
            data = waveforms[node] * scale
            label = str(node) if len(str(node)) < 30 else str(node)[:27] + '...'
            ax.plot(t_scaled, data, linewidth=1.0, label=label)

        ax.set_xlabel(f'Time ({time_label})')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

        if len(nodes) <= 10:
            ax.legend(loc='best', fontsize='small')

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')

        if show:
            plt.show()

        return fig, ax


# Convenience function aliases
def plot_peak_ir_drop_heatmap(
    model: UnifiedPowerGridModel,
    result: Union[QuasiStaticResult, TransientResult],
    **kwargs,
) -> Tuple[Figure, Axes]:
    """Convenience function for DynamicPlotter.plot_peak_ir_drop_heatmap."""
    return DynamicPlotter.plot_peak_ir_drop_heatmap(model, result, **kwargs)


def plot_peak_current_heatmap(
    model: UnifiedPowerGridModel,
    result: Union[QuasiStaticResult, TransientResult],
    **kwargs,
) -> Tuple[Figure, Axes]:
    """Convenience function for DynamicPlotter.plot_peak_current_heatmap."""
    return DynamicPlotter.plot_peak_current_heatmap(model, result, **kwargs)


def plot_time_series(
    result: Union[QuasiStaticResult, TransientResult],
    **kwargs,
) -> Tuple[Figure, Union[Axes, List[Axes]]]:
    """Convenience function for DynamicPlotter.plot_time_series."""
    return DynamicPlotter.plot_time_series(result, **kwargs)


def plot_node_waveforms(
    result: Union[QuasiStaticResult, TransientResult],
    **kwargs,
) -> Tuple[Figure, Axes]:
    """Convenience function for DynamicPlotter.plot_node_waveforms."""
    return DynamicPlotter.plot_node_waveforms(result, **kwargs)

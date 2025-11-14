"""Plotting utilities for IR-drop analysis.

Provides heatmap-style scatter plots for node voltages or IR-drop values.
Designed to work with the graph produced by `generate_power_grid` where each
node has an `xy` attribute (x,y coordinate).
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")  # Safe for headless environments; caller can override before import
import matplotlib.pyplot as plt
import numpy as np

import networkx as nx


def _collect_layer_nodes(G: nx.Graph, voltages: Dict, layer: int | None) -> Tuple[Sequence, np.ndarray, np.ndarray, np.ndarray]:
    xs = []
    ys = []
    vs = []
    nodes = []
    for n, data in G.nodes(data=True):
        if layer is not None and data.get("layer") != layer:
            continue
        if "xy" not in data:
            continue
        if n not in voltages:
            continue
        x, y = data["xy"]
        xs.append(x); ys.append(y); vs.append(voltages[n]); nodes.append(n)
    return nodes, np.array(xs), np.array(ys), np.array(vs)


def plot_voltage_map(G: nx.Graph, voltages: Dict, layer: int | None = None, cmap: str = "viridis", vmin: float | None = None, vmax: float | None = None, show: bool = True):
    """Scatter plot of node voltages.

    layer: restrict to a single layer; None => all layers.
    Returns (fig, ax).
    """
    nodes, xs, ys, vs = _collect_layer_nodes(G, voltages, layer)
    if xs.size == 0:
        raise ValueError("No nodes to plot (check layer or voltages dictionary)")
    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(xs, ys, c=vs, cmap=cmap, s=35, edgecolors="k", linewidths=0.3, vmin=vmin, vmax=vmax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Voltage Map" + (f" (Layer {layer})" if layer is not None else ""))
    cbar = fig.colorbar(sc, ax=ax, label="Voltage (V)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.tight_layout()
    if show:
        plt.show()
    return fig, ax


def plot_ir_drop_map(G: nx.Graph, voltages: Dict, vdd: float, layer: int | None = None, cmap: str = "inferno", show: bool = True):
    """Scatter plot of IR-drop (vdd - V).

    Parameters
    ----------
    G : nx.Graph
        Power grid graph with node attribute 'xy'.
    voltages : Dict
        Mapping node -> voltage (output of IRDropSolver.solve()).
    vdd : float
        Nominal supply voltage. Must be numeric.
    layer : int | None
        Optional layer filter. If provided must be int.
    cmap : str
        Matplotlib colormap name.
    show : bool
        Whether to display immediately via plt.show().
    """
    if not isinstance(vdd, (int, float)):
        raise TypeError(f"vdd must be a numeric voltage (float), got {type(vdd).__name__}. Did you accidentally pass pad_nodes?")
    if layer is not None and not isinstance(layer, int):
        raise TypeError(f"layer must be int or None, got {type(layer).__name__}")
    drops = {n: vdd - v for n, v in voltages.items()}
    nodes, xs, ys, ds = _collect_layer_nodes(G, drops, layer)
    if xs.size == 0:
        raise ValueError("No nodes to plot (check layer or voltages dictionary)")
    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(xs, ys, c=ds, cmap=cmap, s=40, edgecolors="k", linewidths=0.3)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("IR-Drop Map" + (f" (Layer {layer})" if layer is not None else ""))
    cbar = fig.colorbar(sc, ax=ax, label="IR-Drop (V)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.tight_layout()
    if show:
        plt.show()
    return fig, ax

__all__ = ["plot_voltage_map", "plot_ir_drop_map"]

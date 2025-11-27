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

from matplotlib.collections import LineCollection


def plot_current_map(
    G: nx.Graph,
    voltages: Dict,
    layer: int | None = None,
    cmap: str = "plasma",
    show: bool = True,
    abs_current: bool = True,
    linewidth_scale: float = 3.0,
    include_vias: bool = True,
    min_current: float | None = None,
    loads_current: Dict | None = None,
):
    """Visualize currents through resistive edges.

    Current on edge (u,v): I = (V_u - V_v) / R. Direction implied from higher V to lower V.

    Parameters
    ----------
    G : nx.Graph
        Grid graph with edge attribute 'resistance' and node attribute 'xy'.
    voltages : Dict
        Mapping node -> voltage.
    layer : int | None
        If provided, restrict to stripe edges where both nodes have this layer. Via edges are included only if include_vias=True.
    cmap : str
        Colormap for magnitudes.
    show : bool
        Whether to immediately display.
    abs_current : bool
        Plot absolute value if True; else signed values (positive/negative colors).
    linewidth_scale : float
        Multiplier controlling line width scaling vs max current.
    include_vias : bool
        Whether to include via edges (nodes on different layers).
    min_current : float | None
        If provided, only edges with (|I| if abs_current else I) >= min_current are plotted.
    loads_current : Dict | None
        Optional dictionary mapping nodes to load current values. If provided, load nodes
        are marked with colored dots whose intensity varies with current magnitude.
    """
    segs = []
    vals = []
    for u, v, d in G.edges(data=True):
        R = float(d.get("resistance", 0.0))
        if R <= 0.0:
            continue
        Vu = voltages.get(u); Vv = voltages.get(v)
        if Vu is None or Vv is None:
            continue
        lu = G.nodes[u].get("layer")
        lv = G.nodes[v].get("layer")
        is_via = lu != lv
        if layer is not None:
            if is_via and not include_vias:
                continue
            if not is_via and (lu != layer or lv != layer):
                continue
        if is_via and not include_vias:
            continue
        I = (Vu - Vv) / R
        val = abs(I) if abs_current else I
        if min_current is not None:
            # Compare against magnitude already reflected in val (abs if requested)
            if val < min_current:
                continue
        x1, y1 = G.nodes[u]["xy"]
        x2, y2 = G.nodes[v]["xy"]
        segs.append([(x1, y1), (x2, y2)])
        vals.append(val)
    if not segs:
        raise ValueError("No edges selected for current plotting (check layer/filter).")
    vals_arr = np.array(vals, dtype=float)
    vmax = vals_arr.max() if vals_arr.size else 1.0
    # Line widths proportional to magnitude (avoid zero width)
    lw = 0.4 + linewidth_scale * (vals_arr / vmax if vmax > 0 else vals_arr)
    fig, ax = plt.subplots(figsize=(6, 5))
    lc = LineCollection(segs, array=vals_arr, cmap=cmap, linewidths=lw)
    ax.add_collection(lc)
    ax.set_aspect("equal", adjustable="box")
    ax.autoscale()
    
    # Overlay load currents if provided
    if loads_current is not None:
        load_xs = []
        load_ys = []
        load_currents = []
        for node, current in loads_current.items():
            node_data = G.nodes.get(node)
            if node_data is None or "xy" not in node_data:
                continue
            node_layer = node_data.get("layer")
            if layer is not None and node_layer != layer:
                continue
            x, y = node_data["xy"]
            load_xs.append(x)
            load_ys.append(y)
            load_currents.append(abs(current) if abs_current else current)
        
        if load_xs:
            load_currents_arr = np.array(load_currents)
            sc = ax.scatter(
                load_xs, load_ys, 
                c=load_currents_arr, 
                cmap="Reds", 
                s=80, 
                edgecolors="black", 
                linewidths=1.5,
                alpha=0.8,
                zorder=10,
                label="Load currents"
            )
            # Add separate colorbar for load currents
            cbar_loads = fig.colorbar(sc, ax=ax, label="Load " + ("|I| (A)" if abs_current else "I (A)"), 
                                     pad=0.02, aspect=30, shrink=0.8)
    
    title = "Current Map"
    if layer is not None:
        title += f" (Layer {layer})"
    ax.set_title(title)
    cbar = fig.colorbar(lc, ax=ax, label=("|I| (A)" if abs_current else "I (A)"))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.tight_layout()
    if show:
        plt.show()
    return fig, ax

__all__.append("plot_current_map")

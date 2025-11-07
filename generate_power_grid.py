#!/usr/bin/env python3
"""
Artificial power-grid generator for fast prototyping.

Requirements satisfied:
1) K vertical layers
2) Layer 0 is horizontal with N horizontal stripes, evenly spaced
3) All loads live on layer 0 (uniformly distributed, not at via nodes)
4) Orientation alternates per layer (H,V,H,V,...)
5) Stripe count halves each layer (ceil(N/2^ℓ), minimum 1)
6) Adjacent layers connected by via resistors at intersections
7) No current loads at via nodes
8) Layer-0 stripe resistance = max_stripe_res (total per stripe, across full length)
9) Each upper layer halves its stripe resistance vs the layer below
10) N_vsrc pads on the top-most layer, evenly distributed
11) Via resistors start at max_via_res at layer 0, halved each layer above
"""

from __future__ import annotations
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import networkx as nx
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class NodeID:
    layer: int
    idx: int        # unique running index within layer for deterministic node ids


def generate_power_grid(
    K: int,
    N0: int,
    I_N: int,
    N_vsrc: int,
    max_stripe_res: float,
    max_via_res: float,
    load_current: float = 1.0,
    Lx: float = 1.0,
    Ly: float = 1.0,
    seed: int | None = 42,
    plot: bool = True,
):
    """
    Returns:
      G: nx.Graph with edge attribute 'resistance'
      loads: Dict[node_id, current] (only on layer 0)
      pads:  List[node_id] (top-most layer)
    """
    rng = random.Random(seed)

    # ---------- Helper functions ----------
    def orientation_of(layer: int) -> str:
        # 0: horizontal, then alternate
        return "H" if (layer % 2 == 0) else "V"

    def stripe_count(layer: int) -> int:
        # Half each layer, minimum 1
        return max(1, math.ceil(N0 / (2 ** layer)))

    def stripe_resistance(layer: int) -> float:
        # Halved per ascending layer (total across the stripe)
        return max_stripe_res / (2 ** layer)

    def via_resistance(layer: int) -> float:
        # Via between layer and layer+1; value halves per higher layer
        return max_via_res / (2 ** layer)

    # Even spacing helper
    def positions(count: int, length: float) -> List[float]:
        # Evenly spaced internal lines across the span (avoid exactly 0 and length)
        # place at (i+1)/(count+1) of the span
        return [length * (i + 1) / (count + 1) for i in range(count)]

    # For each layer, we will create nodes at:
    #  - intersections with adjacent layer stripes (via locations)
    #  - stripe endpoints on boundaries (to allow continuous stripes)
    # Later we’ll split segments to insert load “tap” nodes on layer 0 (not at vias).

    G = nx.Graph()

    # Keep bookkeeping of nodes by (layer, x, y) to avoid duplicates at same point
    node_lookup: Dict[Tuple[int, float, float], NodeID] = {}
    node_attrs: Dict[NodeID, Dict] = {}

    # Precompute stripe lines per layer
    layers = []
    for ℓ in range(K):
        ori = orientation_of(ℓ)
        n_stripes = stripe_count(ℓ)
        coords = positions(n_stripes, Ly if ori == "H" else Lx)
        layers.append(dict(layer=ℓ, orientation=ori, n=n_stripes, coords=coords))

    # Build nodes at intersections (vias) between layer ℓ and ℓ+1,
    # and stripe endpoints on boundaries for every stripe on every layer.
    next_idx_in_layer = [0 for _ in range(K)]

    def get_or_create_node(layer: int, x: float, y: float, kind: str, orientation: str):
        key = (layer, round(x, 12), round(y, 12))
        nd = node_lookup.get(key)
        if nd is None:
            nd = NodeID(layer, next_idx_in_layer[layer])
            next_idx_in_layer[layer] += 1
            node_lookup[key] = nd
            node_attrs[nd] = dict(x=x, y=y, kind=set([kind]), layer=layer, orientation=orientation)
            G.add_node(nd)
        else:
            node_attrs[nd]["kind"].add(kind)
        return nd

    # 1) Stripe endpoints on boundaries for every stripe
    for L in layers:
        ℓ, ori, ys_or_xs = L["layer"], L["orientation"], L["coords"]
        if ori == "H":
            # horizontal stripes at y = const; from x=0 to x=Lx
            for y in ys_or_xs:
                # endpoints
                get_or_create_node(ℓ, 0.0, y, "stripe", ori)
                get_or_create_node(ℓ, Lx, y, "stripe", ori)
        else:
            # vertical stripes at x = const; from y=0 to y=Ly
            for x in ys_or_xs:
                get_or_create_node(ℓ, x, 0.0, "stripe", ori)
                get_or_create_node(ℓ, x, Ly, "stripe", ori)

    # 2) Via/intersection nodes between consecutive layers
    for ℓ in range(K - 1):
        L = layers[ℓ]
        U = layers[ℓ + 1]
        # They always alternate orientation
        assert L["orientation"] != U["orientation"]
        if L["orientation"] == "H":  # L horizontal (y fixed), U vertical (x fixed)
            y_list = L["coords"]
            x_list = U["coords"]
        else:                        # L vertical (x fixed), U horizontal (y fixed)
            x_list = L["coords"]
            y_list = U["coords"]

        for x in x_list:
            for y in y_list:
                # create one node on each layer at the same (x,y)
                nL = get_or_create_node(ℓ, x, y, "via", L["orientation"])
                nU = get_or_create_node(ℓ + 1, x, y, "via", U["orientation"])
                # connect via resistor between them
                G.add_edge(nL, nU, resistance=via_resistance(ℓ), kind="via")

    # 3) Connect nodes along each stripe with segment resistors
    # Strategy:
    #   - For each stripe line, collect all nodes on that line (endpoints + via nodes),
    #     sort along the stripe axis, and connect consecutive nodes with equal share
    #     of the stripe's total resistance.
    for L in layers:
        ℓ, ori, coords = L["layer"], L["orientation"], L["coords"]
        total_R = stripe_resistance(ℓ)

        if ori == "H":
            # For each y stripe, collect nodes on that y
            for y in coords:
                # collect all nodes on layer ℓ with that y
                nodes_on_line = [
                    (nd, node_attrs[nd]["x"])
                    for nd in G.nodes
                    if node_attrs[nd]["layer"] == ℓ and math.isclose(node_attrs[nd]["y"], y)
                ]
                # sort by x along the stripe
                nodes_on_line.sort(key=lambda t: t[1])
                segs = max(1, len(nodes_on_line) - 1)
                seg_R = total_R / segs
                for (n1, _), (n2, _) in zip(nodes_on_line, nodes_on_line[1:]):
                    if not G.has_edge(n1, n2):
                        G.add_edge(n1, n2, resistance=seg_R, kind="stripe")
        else:
            # vertical stripes, group by x
            for x in coords:
                nodes_on_line = [
                    (nd, node_attrs[nd]["y"])
                    for nd in G.nodes
                    if node_attrs[nd]["layer"] == ℓ and math.isclose(node_attrs[nd]["x"], x)
                ]
                nodes_on_line.sort(key=lambda t: t[1])
                segs = max(1, len(nodes_on_line) - 1)
                seg_R = total_R / segs
                for (n1, _), (n2, _) in zip(nodes_on_line, nodes_on_line[1:]):
                    if not G.has_edge(n1, n2):
                        G.add_edge(n1, n2, resistance=seg_R, kind="stripe")

    # 4) Place voltage sources (pads) on the top-most layer, evenly distributed
    top = layers[-1]
    ℓT, oriT, coordsT = top["layer"], top["orientation"], top["coords"]

    pad_nodes: List[NodeID] = []
    # We'll place pads at boundary endpoints of stripes, round-robin across stripes.
    endpoints: List[NodeID] = []
    if oriT == "H":
        for y in coordsT:
            # choose right boundary endpoint (x=Lx) for distinct positions
            nd = node_lookup[(ℓT, round(Lx, 12), round(y, 12))]
            endpoints.append(nd)
    else:
        for x in coordsT:
            # choose top boundary endpoint (y=Ly)
            nd = node_lookup[(ℓT, round(x, 12), round(Ly, 12))]
            endpoints.append(nd)
    # Even distribution
    if len(endpoints) == 0:
        # Edge case: only one stripe on top, ensure endpoints list
        # (happens if N0=1 and K small)
        endpoints = [nd for nd in G.nodes if node_attrs[nd]["layer"] == ℓT]

    for i in range(N_vsrc):
        nd = endpoints[i % len(endpoints)]
        node_attrs[nd]["kind"].add("pad")
        pad_nodes.append(nd)

    # 5) Insert layer-0 loads uniformly, NOT at via nodes
    # We will split stripe segments on layer 0 to insert a "load tap" node in the middle.
    L0 = layers[0]
    assert L0["orientation"] == "H"
    load_nodes: Dict[NodeID, float] = {}

    # Collect all layer-0 stripe segments (excluding segments that coincide with via nodes only)
    stripe_segs = []
    for y in L0["coords"]:
        # Gather ordered nodes along this stripe
        nodes_on_line = sorted(
            [nd for nd in G.nodes if node_attrs[nd]["layer"] == 0 and math.isclose(node_attrs[nd]["y"], y)],
            key=lambda nd: node_attrs[nd]["x"],
        )
        # Create segment list between consecutive nodes
        for n1, n2 in zip(nodes_on_line, nodes_on_line[1:]):
            # Avoid tiny zero-length edge
            if node_attrs[n1]["x"] == node_attrs[n2]["x"]:
                continue
            # Edge must exist
            if not G.has_edge(n1, n2):
                continue
            # We will consider this as a candidate to host a load tap
            stripe_segs.append((n1, n2))

    # Sample I_N segments as uniformly as possible across stripes
    if stripe_segs:
        step = max(1, len(stripe_segs) // I_N)
        picks = list(range(0, len(stripe_segs), step))[:I_N]
    else:
        picks = []

    for idx in picks:
        n1, n2 = stripe_segs[idx]
        # Current edge resistance
        R = G.edges[n1, n2]["resistance"]
        # Insert a midpoint node on layer 0 (not a via)
        x1, y1 = node_attrs[n1]["x"], node_attrs[n1]["y"]
        x2, y2 = node_attrs[n2]["x"], node_attrs[n2]["y"]
        mx, my = 0.5 * (x1 + x2), 0.5 * (y1 + y2)

        # Create new node
        mid = NodeID(0, next_idx_in_layer[0])
        next_idx_in_layer[0] += 1
        G.add_node(mid)
        node_attrs[mid] = dict(x=mx, y=my, kind=set(["load", "stripe"]), layer=0, orientation="H")

        # Replace edge n1--n2 by n1--mid and mid--n2 with split resistance
        G.remove_edge(n1, n2)
        G.add_edge(n1, mid, resistance=R / 2.0, kind="stripe")
        G.add_edge(mid, n2, resistance=R / 2.0, kind="stripe")

        # Register load
        load_nodes[mid] = load_current

    # Attach attributes back to networkx graph
    for nd, attrs in node_attrs.items():
        # Convert 'kind' set to a sorted string for readability
        attrs_out = dict(attrs)
        attrs_out["kind"] = ",".join(sorted(list(attrs["kind"])))
        attrs_out["xy"] = (attrs["x"], attrs["y"])
        del attrs_out["x"], attrs_out["y"]
        nx.set_node_attributes(G, {nd: attrs_out})

    # ---------- Optional quick plot ----------
    if plot:
        plt.figure(figsize=(8, 7))
        # Draw edges
        for (u, v, d) in G.edges(data=True):
            x1, y1 = G.nodes[u]["xy"]
            x2, y2 = G.nodes[v]["xy"]
            plt.plot([x1, x2], [y1, y2], lw=0.8, color="#555555", alpha=0.7)

        # Draw nodes by type
        def draw_nodes(filter_fn, color, size, z=3, label=None):
            xs, ys = [], []
            for n, data in G.nodes(data=True):
                if filter_fn(data):
                    x, y = data["xy"]
                    xs.append(x); ys.append(y)
            if xs:
                plt.scatter(xs, ys, s=size, c=color, edgecolors="k", linewidths=0.4, zorder=z, label=label)

        draw_nodes(lambda d: "pad" in d["kind"],   "#FF6B6B", 60, z=5, label="Pads (top)")
        draw_nodes(lambda d: "load" in d["kind"],  "#F8E71C", 50, z=5, label="Loads (L0)")
        draw_nodes(lambda d: "via" in d["kind"],   "#AAAAAA", 18, z=4, label="Vias")
        draw_nodes(lambda d: "stripe" in d["kind"] and "load" not in d["kind"] and "pad" not in d["kind"],
                   "#4A90E2", 10, z=3, label="Stripe nodes")

        plt.gca().set_aspect("equal", adjustable="box")
        plt.xlim(-0.02, Lx + 0.02)
        plt.ylim(-0.02, Ly + 0.02)
        plt.title("Artificial Power Grid")
        plt.legend(loc="upper right", fontsize=8)
        plt.tight_layout()
        plt.show()

    return G, load_nodes, pad_nodes


if __name__ == "__main__":
    # Example usage
    G, loads, pads = generate_power_grid(
        K=5,
        N0=16,
        I_N=20,
        N_vsrc=6,
        max_stripe_res=1.0,   # Ω per stripe in layer 0
        max_via_res=0.1,      # Ω for vias between L0-L1; halves on higher vias
        load_current=1.0,
        seed=7,
        plot=True,
    )
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"Loads placed: {len(loads)} nodes; Pads: {len(pads)}")

"""Distributed pre-binning pipeline for stripe-mode heatmaps.

Generates stripe heatmaps from distributed DDM results without flattening
all tile data into a single dict. Each tile independently digitizes its
nodes into a shared ``GlobalBinSpec``, then the coordinator merges bin
arrays with a simple reduce.

Data flow (2 round-trips to workers):
    1. ``get_layer_metadata()`` per tile -> merge -> ``build_global_bin_spec()``
    2. Per-tile voltage dicts + bin spec -> ``prebin_tile()`` (via ``map_func``)
       -> ``merge_tile_prebins()`` -> render

Usage::

    from distributed.heatmap import plot_distributed_heatmaps

    plot_distributed_heatmaps(result, model, output_dir='./plots')
"""

from __future__ import annotations

import logging
import os
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Data classes
# =============================================================================

@dataclass
class LayerBinSpec:
    """Per-layer stripe binning specification.

    Defines how a single layer's nodes are partitioned into display stripes
    and how values within each stripe are digitized along the parallel axis.

    Attributes
    ----------
    orientation : str
        Routing direction: ``'H'`` (stripes along Y, bins along X),
        ``'V'`` (stripes along X, bins along Y), or ``'MIXED'``.
    stripe_edges : list of (float, float)
        ``(stripe_start, stripe_end)`` for each display stripe in
        the perpendicular coordinate.
    bin_edges_list : list of np.ndarray
        Bin edges per display stripe along the parallel axis.
    n_nodes : int
        Total node count for this layer across all tiles.  Used in
        the plot title.  Defaults to ``0`` (omitted from title).
    """

    orientation: str
    stripe_edges: List[Tuple[float, float]]
    bin_edges_list: List[np.ndarray]
    n_nodes: int = 0


_VALID_AGGREGATIONS = frozenset({'max', 'sum'})

# Default bin resolution for auto-calculated stripe heatmaps.
# Matches the flat solver's target_bins_per_stripe (500) and max_bins (100000).
_TARGET_BINS = 500
_MAX_BINS = 100000


@dataclass
class GlobalBinSpec:
    """Shared binning specification across all tiles.

    Built once from merged tile metadata, then broadcast to every tile
    for independent pre-binning.

    Attributes
    ----------
    layers : dict
        ``layer_name -> LayerBinSpec``
    nominal_voltage : float
        Vdd for IR-drop computation (mV).
    aggregation : str
        ``'max'`` for IR-drop heatmaps, ``'sum'`` for current heatmaps.
    """

    layers: Dict[str, LayerBinSpec]
    nominal_voltage: float
    aggregation: str = 'max'

    def __post_init__(self) -> None:
        if self.aggregation not in _VALID_AGGREGATIONS:
            raise ValueError(
                f"Invalid aggregation '{self.aggregation}'. "
                f"Must be one of {sorted(_VALID_AGGREGATIONS)}."
            )


# =============================================================================
# build_global_bin_spec
# =============================================================================

def _merge_stripe_coords(
    all_coords: List[float],
    tol_frac: float = 0.001,
) -> List[float]:
    """Merge nearby stripe coordinates into representative values.

    Stripe coordinates from different tiles may differ by tiny amounts
    due to floating-point coordinate rounding in the netlist. We merge
    coordinates that are within ``tol_frac`` of the median spacing.

    Parameters
    ----------
    all_coords : list of float
        Sorted union of stripe coordinates from all tiles.
    tol_frac : float
        Fraction of median spacing used as merge tolerance.

    Returns
    -------
    list of float
        Deduplicated, sorted representative stripe coordinates.
    """
    if len(all_coords) <= 1:
        return list(all_coords)

    sorted_c = sorted(set(all_coords))
    if len(sorted_c) <= 1:
        return sorted_c

    # Compute median spacing for tolerance
    diffs = np.diff(sorted_c)
    median_spacing = float(np.median(diffs)) if len(diffs) > 0 else 1.0
    tol = max(median_spacing * tol_frac, 1e-6)

    merged: List[float] = [sorted_c[0]]
    for c in sorted_c[1:]:
        if c - merged[-1] > tol:
            merged.append(c)
        # else: absorbed into previous representative
    return merged


def _compute_stripe_boundaries(
    stripe_coords: List[float],
) -> List[Tuple[float, float]]:
    """Compute (start, end) boundary intervals for display stripes.

    Boundaries are midpoints between adjacent stripe coordinates, with
    extrapolation at the edges.

    Parameters
    ----------
    stripe_coords : list of float
        Sorted representative stripe coordinates.

    Returns
    -------
    list of (float, float)
        ``(start, end)`` for each stripe.
    """
    n = len(stripe_coords)
    if n == 0:
        return []
    if n == 1:
        return [(stripe_coords[0] - 1.0, stripe_coords[0] + 1.0)]

    edges: List[Tuple[float, float]] = []
    for i in range(n):
        if i == 0:
            start = stripe_coords[0] - (stripe_coords[1] - stripe_coords[0]) / 2
        else:
            start = (stripe_coords[i - 1] + stripe_coords[i]) / 2

        if i == n - 1:
            end = stripe_coords[-1] + (stripe_coords[-1] - stripe_coords[-2]) / 2
        else:
            end = (stripe_coords[i] + stripe_coords[i + 1]) / 2

        edges.append((start, end))
    return edges


def build_global_bin_spec(
    tile_metadatas: List[Dict[str, Dict]],
    nominal_voltage: float,
    aggregation: str = 'max',
    max_stripes: int = 2000,
    stripe_bin_size: Optional[int] = None,
) -> GlobalBinSpec:
    """Build a shared ``GlobalBinSpec`` from merged per-tile layer metadata.

    Unions bounding boxes, merges stripe coordinates, determines
    orientation from edge counts, and creates uniform bin edges.

    Parameters
    ----------
    tile_metadatas : list of dict
        Per-tile dicts from ``TileWorker.get_layer_metadata()``.
    stripe_bin_size : int or None
        Physical bin size in coordinate units (matching the flat solver
        semantics).  Converted to a bin count per-layer via
        ``int(parallel_range / stripe_bin_size)``.  ``None`` targets
        ~500 bins per layer automatically.
    """
    if aggregation not in _VALID_AGGREGATIONS:
        raise ValueError(
            f"Invalid aggregation '{aggregation}'. "
            f"Must be one of {sorted(_VALID_AGGREGATIONS)}."
        )

    # Collect per-layer info across all tiles
    layer_bboxes: Dict[str, List[Tuple[float, float, float, float]]] = {}
    layer_stripe_coords_h: Dict[str, List[float]] = {}
    layer_stripe_coords_v: Dict[str, List[float]] = {}
    layer_edge_counts: Dict[str, Tuple[int, int, int]] = {}
    layer_node_counts: Dict[str, int] = {}

    for tile_meta in tile_metadatas:
        for layer, info in tile_meta.items():
            layer_bboxes.setdefault(layer, []).append(info['bbox'])
            layer_stripe_coords_h.setdefault(layer, []).extend(
                info['stripe_coords_h']
            )
            layer_stripe_coords_v.setdefault(layer, []).extend(
                info['stripe_coords_v']
            )
            h, v, d = info['edge_orientation']
            prev_h, prev_v, prev_d = layer_edge_counts.get(layer, (0, 0, 0))
            layer_edge_counts[layer] = (prev_h + h, prev_v + v, prev_d + d)
            layer_node_counts[layer] = (
                layer_node_counts.get(layer, 0) + info.get('n_nodes', 0)
            )

    layers_spec: Dict[str, LayerBinSpec] = {}

    for layer in sorted(layer_bboxes):
        # Union bounding box
        bboxes = layer_bboxes[layer]
        x_min = min(b[0] for b in bboxes)
        x_max = max(b[1] for b in bboxes)
        y_min = min(b[2] for b in bboxes)
        y_max = max(b[3] for b in bboxes)

        # Determine orientation from aggregated edge counts
        h_count, v_count, _d_count = layer_edge_counts.get(layer, (0, 0, 0))
        if h_count >= v_count:
            orientation = 'H'
        else:
            orientation = 'V'

        # Merge stripe coordinates for the chosen orientation
        if orientation == 'H':
            raw_coords = layer_stripe_coords_h.get(layer, [])
        else:
            raw_coords = layer_stripe_coords_v.get(layer, [])

        merged_coords = _merge_stripe_coords(raw_coords)

        # Consolidate if too many stripes
        if len(merged_coords) > max_stripes:
            group_size = int(np.ceil(len(merged_coords) / max_stripes))
            consolidated: List[float] = []
            for i in range(0, len(merged_coords), group_size):
                group = merged_coords[i:i + group_size]
                consolidated.append(float(np.mean(group)))
            merged_coords = consolidated

        # Compute stripe boundaries
        stripe_edges = _compute_stripe_boundaries(merged_coords)

        # Compute bin edges for each stripe along the parallel axis
        if orientation == 'H':
            parallel_min, parallel_max = x_min, x_max
        else:
            parallel_min, parallel_max = y_min, y_max

        parallel_range = parallel_max - parallel_min
        if parallel_range <= 0:
            parallel_range = 1.0

        # Compute number of bins for this layer.
        # stripe_bin_size is a physical size (units); convert to bin count.
        if stripe_bin_size is not None and stripe_bin_size > 0:
            n_bins = max(1, int(parallel_range / stripe_bin_size))
        else:
            n_bins = _TARGET_BINS
        n_bins = min(n_bins, _MAX_BINS)

        # Share a single bin_edges array across all stripes in this layer
        # (they all span the same parallel range with the same bin count).
        shared_edges = np.linspace(parallel_min, parallel_max, n_bins + 1)
        bin_edges_list: List[np.ndarray] = [shared_edges] * len(stripe_edges)

        layers_spec[layer] = LayerBinSpec(
            orientation=orientation,
            stripe_edges=stripe_edges,
            bin_edges_list=bin_edges_list,
            n_nodes=layer_node_counts.get(layer, 0),
        )

    return GlobalBinSpec(
        layers=layers_spec,
        nominal_voltage=nominal_voltage,
        aggregation=aggregation,
    )


# =============================================================================
# prebin_tile (stateless, picklable for map_func)
# =============================================================================

def prebin_tile(
    tile_voltages: Dict[str, float],
    global_bin_spec: GlobalBinSpec,
    boundary_exclude: Optional[Set[str]] = None,
) -> Dict[str, List[Optional[np.ndarray]]]:
    """Pre-bin one tile's nodes into stripe bins.

    Stateless function suitable for ``backend.map_func()``.

    Parameters
    ----------
    tile_voltages : dict
        ``node_name -> voltage`` for this tile (interior + boundary).
    global_bin_spec : GlobalBinSpec
        Shared binning specification.
    boundary_exclude : set of str or None
        Nodes to skip (for sum-mode deduplication of shared boundary nodes).

    Returns
    -------
    dict
        ``layer -> list[Optional[np.ndarray]]`` where each entry is a
        partial bin-values array for one display stripe (``None`` if the
        tile has no nodes in that stripe). For both ``'max'`` and
        ``'sum'``: NaN means no node contributed to the bin.
    """
    from visualization.stripe_heatmap import parse_node_info

    exclude = boundary_exclude or set()
    aggregation = global_bin_spec.aggregation
    vdd = global_bin_spec.nominal_voltage

    # Parse all nodes once into per-layer arrays
    layer_nodes: Dict[str, Tuple[List[float], List[float], List[float]]] = {}

    for node, voltage in tile_voltages.items():
        if node in exclude:
            continue
        x, y, layer = parse_node_info(node)
        if x is None or layer not in global_bin_spec.layers:
            continue

        if aggregation == 'max':
            value = (vdd - voltage) * 1000.0  # IR-drop in mV
        else:
            value = voltage  # raw value (caller provides current in mA)

        bucket = layer_nodes.get(layer)
        if bucket is None:
            bucket = ([], [], [])
            layer_nodes[layer] = bucket
        bucket[0].append(x)
        bucket[1].append(y)
        bucket[2].append(value)

    result: Dict[str, List[Optional[np.ndarray]]] = {}

    for layer, spec in global_bin_spec.layers.items():
        bucket = layer_nodes.get(layer)
        n_stripes = len(spec.stripe_edges)
        stripe_partials: List[Optional[np.ndarray]] = [None] * n_stripes

        if bucket is None:
            result[layer] = stripe_partials
            continue

        xs = np.asarray(bucket[0], dtype=np.float64)
        ys = np.asarray(bucket[1], dtype=np.float64)
        vals = np.asarray(bucket[2], dtype=np.float64)

        # For each display stripe, select nodes and digitize
        for si, (s_start, s_end) in enumerate(spec.stripe_edges):
            bin_edges = spec.bin_edges_list[si]
            n_bins = len(bin_edges) - 1

            # Select nodes whose perpendicular coord falls in stripe
            if spec.orientation == 'H':
                perp = ys
            else:
                perp = xs

            mask = (perp >= s_start) & (perp <= s_end)
            if not np.any(mask):
                continue

            # Parallel coordinates for digitization
            if spec.orientation == 'H':
                parallel = xs[mask]
            else:
                parallel = ys[mask]
            stripe_vals = vals[mask]

            # Digitize: np.digitize returns 1-based bin indices
            bin_idx = np.digitize(parallel, bin_edges) - 1
            # Filter out-of-range nodes instead of clamping to edge bins,
            # which would corrupt values there.  Matches the pattern in
            # visualization/stripe_heatmap.py:aggregate_stripe_bins().
            valid = (bin_idx >= 0) & (bin_idx < n_bins)
            bin_idx = bin_idx[valid]
            stripe_vals = stripe_vals[valid]

            if len(bin_idx) == 0:
                continue

            if aggregation == 'max':
                partial = np.full(n_bins, np.nan, dtype=np.float64)
                # np.fmax.at handles NaN correctly (ignores NaN operands)
                np.fmax.at(partial, bin_idx, stripe_vals)
            else:
                partial = np.full(n_bins, np.nan, dtype=np.float64)
                touched_mask = np.zeros(n_bins, dtype=bool)
                touched_mask[bin_idx] = True
                partial[touched_mask] = 0.0
                np.add.at(partial, bin_idx, stripe_vals)

            stripe_partials[si] = partial

        result[layer] = stripe_partials

    return result


# =============================================================================
# merge_tile_prebins
# =============================================================================

def merge_tile_prebins(
    tile_prebins: List[Dict[str, List[Optional[np.ndarray]]]],
    global_bin_spec: GlobalBinSpec,
) -> Dict[str, List[Tuple[float, float, np.ndarray, np.ndarray]]]:
    """Merge per-tile pre-binned arrays into final stripe data.

    For ``'max'`` aggregation: takes element-wise ``nanmax`` across tiles.
    For ``'sum'`` aggregation: takes element-wise ``nansum`` across tiles,
    preserving ``NaN`` for bins where no tile contributed data.

    Parameters
    ----------
    tile_prebins : list of dict
        Output of ``prebin_tile()`` for each tile.
    global_bin_spec : GlobalBinSpec
        Shared binning specification.

    Returns
    -------
    dict
        ``layer -> [(stripe_start, stripe_end, bin_edges, bin_values), ...]``
        where ``bin_values`` is the final merged array per stripe.
    """
    aggregation = global_bin_spec.aggregation
    result: Dict[str, List[Tuple[float, float, np.ndarray, np.ndarray]]] = {}

    for layer, spec in global_bin_spec.layers.items():
        layer_stripes: List[Tuple[float, float, np.ndarray, np.ndarray]] = []

        for si, (s_start, s_end) in enumerate(spec.stripe_edges):
            bin_edges = spec.bin_edges_list[si]
            n_bins = len(bin_edges) - 1

            # Collect non-None partial arrays for this stripe
            partials: List[np.ndarray] = []
            for tp in tile_prebins:
                layer_data = tp.get(layer)
                if layer_data is None:
                    continue
                if si < len(layer_data) and layer_data[si] is not None:
                    partials.append(layer_data[si])

            if not partials:
                # No tile contributed to this stripe
                merged = np.full(n_bins, np.nan, dtype=np.float64)
            elif len(partials) == 1:
                merged = partials[0].copy()
            else:
                stacked = np.stack(partials, axis=0)
                if aggregation == 'max':
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore', RuntimeWarning)
                        merged = np.nanmax(stacked, axis=0)
                else:
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore', RuntimeWarning)
                        all_nan_mask = np.all(np.isnan(stacked), axis=0)
                        merged = np.nansum(stacked, axis=0)
                        merged[all_nan_mask] = np.nan

            layer_stripes.append((s_start, s_end, bin_edges, merged))

        result[layer] = layer_stripes

    return result


# =============================================================================
# compute_boundary_ownership (sum-mode dedup)
# =============================================================================

def compute_boundary_ownership(
    model: Any,  # DistributedPowerGridModel
) -> Dict[int, Set[str]]:
    """Compute per-tile boundary node exclusion sets for sum-mode dedup.

    A boundary node is *owned* by the tile with the smallest tile index
    (lexicographic order of ``(row, col)`` tuple). All other tiles sharing
    that boundary node must exclude it to avoid double-counting.

    Parameters
    ----------
    model : DistributedPowerGridModel
        The distributed model with ``tile_boundary_nodes``.

    Returns
    -------
    dict
        ``tile_index (int) -> set of boundary node names to EXCLUDE``
        for this tile. Tile indices correspond to the order in
        ``model.metadata.tile_configs``.
    """
    # Map tile_id -> config index
    tile_id_to_idx: Dict[Tuple[int, int], int] = {}
    for i, tc in enumerate(model.metadata.tile_configs):
        tile_id_to_idx[tc.tile_id] = i

    # For each boundary node, find the owning tile (smallest index)
    node_owner: Dict[str, int] = {}
    for tid, bnd_list in model.tile_boundary_nodes.items():
        idx = tile_id_to_idx[tid]
        for node in bnd_list:
            if node not in node_owner or idx < node_owner[node]:
                node_owner[node] = idx

    # Build exclusion sets: for each tile, exclude nodes owned by another tile
    exclusions: Dict[int, Set[str]] = {i: set() for i in range(len(model.metadata.tile_configs))}
    for tid, bnd_list in model.tile_boundary_nodes.items():
        idx = tile_id_to_idx[tid]
        for node in bnd_list:
            if node_owner[node] != idx:
                exclusions[idx].add(node)

    return exclusions


# =============================================================================
# plot_distributed_heatmaps (top-level orchestrator)
# =============================================================================

def plot_distributed_heatmaps(
    result: Any,  # DistributedSolveResult
    model: Any,   # DistributedPowerGridModel
    output_dir: str = './plots',
    plot_layers: Optional[List[str]] = None,
    max_stripes: int = 2000,
    stripe_bin_size: Optional[int] = None,
    is_current: bool = False,
    verbose: bool = False,
) -> None:
    """Generate per-layer stripe heatmaps from distributed DDM DC results.

    Two round-trips: (1) metadata -> ``GlobalBinSpec``,
    (2) per-tile prebinning via ``map_func`` -> merge -> render.
    """
    import time

    os.makedirs(output_dir, exist_ok=True)

    aggregation = 'sum' if is_current else 'max'

    # ── Round trip 1: collect layer metadata ──────────────────────────
    t0 = time.perf_counter()
    tile_metadatas = model.backend.call_all(model.workers, 'get_layer_metadata')
    t_meta = time.perf_counter() - t0

    if verbose:
        logger.info("Collected layer metadata from %d tiles in %.3fs",
                     len(tile_metadatas), t_meta)

    # Build global bin spec
    bin_spec = build_global_bin_spec(
        tile_metadatas=tile_metadatas,
        nominal_voltage=result.nominal_voltage,
        aggregation=aggregation,
        max_stripes=max_stripes,
        stripe_bin_size=stripe_bin_size,
    )

    if verbose:
        for layer, spec in bin_spec.layers.items():
            logger.info("  Layer %s: %s, %d stripes",
                         layer, spec.orientation, len(spec.stripe_edges))

    # ── Round trip 2: pre-bin tiles ──────────────────────────────────
    t0 = time.perf_counter()

    # Build per-tile voltage/current dicts
    if is_current:
        tile_value_dicts = model.backend.call_all(
            model.workers, 'get_current_injections'
        )
    else:
        tile_value_dicts = [
            result.tile_results[tc.tile_id].voltages
            for tc in model.metadata.tile_configs
        ]

    # Compute boundary exclusion for deduplication of shared boundary nodes.
    # For 'sum' aggregation this is essential to avoid double-counting.
    # For 'max' aggregation boundary exclusion is technically unnecessary
    # because max is idempotent (max(v, v) == v), but we always compute it
    # defensively so the pipeline is correct if aggregation mode changes.
    exclusion_map = compute_boundary_ownership(model)

    # Pre-bin each tile
    prebin_args = []
    for i, _tc in enumerate(model.metadata.tile_configs):
        exclude = exclusion_map[i] if exclusion_map is not None else None
        prebin_args.append((tile_value_dicts[i], bin_spec, exclude))

    tile_prebins = model.backend.map_func(prebin_tile, prebin_args)
    t_prebin = time.perf_counter() - t0

    if verbose:
        logger.info("Pre-binned %d tiles in %.3fs", len(tile_prebins), t_prebin)

    # ── Merge and render ─────────────────────────────────────────────
    t0 = time.perf_counter()
    merged = merge_tile_prebins(tile_prebins, bin_spec)
    t_merge = time.perf_counter() - t0

    if verbose:
        logger.info("Merged prebins in %.3fs", t_merge)

    # Filter layers if requested
    if plot_layers is not None:
        merged = {k: v for k, v in merged.items() if k in plot_layers}

    # Render each layer
    _render_merged_heatmaps(
        merged_data=merged,
        bin_spec=bin_spec,
        output_dir=output_dir,
        is_current=is_current,
        net_name=result.net_name,
        verbose=verbose,
    )


def plot_distributed_td_heatmaps(
    model: Any,  # DistributedPowerGridModel
    nominal_voltage: float,
    net_name: Optional[str] = None,
    output_dir: str = './plots',
    plot_layers: Optional[List[str]] = None,
    max_stripes: int = 2000,
    stripe_bin_size: Optional[int] = None,
    title_prefix: str = 'Peak IR-Drop',
    verbose: bool = False,
) -> None:
    """Generate per-layer stripe heatmaps from worker-side peak IR-drop data.

    Like :func:`plot_distributed_heatmaps` but reads peak data directly
    from workers via ``prebin_peak_data()`` — no ``result`` dict needed.
    Only compact pre-binned arrays are transferred to the coordinator.
    """
    import time

    os.makedirs(output_dir, exist_ok=True)

    # -- Round trip 1: collect layer metadata ---------------------------------
    t0 = time.perf_counter()
    tile_metadatas = model.backend.call_all(model.workers, 'get_layer_metadata')
    t_meta = time.perf_counter() - t0

    if verbose:
        logger.info("TD heatmap: collected metadata from %d tiles in %.3fs",
                     len(tile_metadatas), t_meta)

    bin_spec = build_global_bin_spec(
        tile_metadatas=tile_metadatas,
        nominal_voltage=nominal_voltage,
        aggregation='max',
        max_stripes=max_stripes,
        stripe_bin_size=stripe_bin_size,
    )

    if verbose:
        for layer, spec in bin_spec.layers.items():
            logger.info("  Layer %s: %s, %d stripes",
                         layer, spec.orientation, len(spec.stripe_edges))

    # -- Round trip 2: workers prebin peak data locally -----------------------
    t0 = time.perf_counter()

    exclusion_map = compute_boundary_ownership(model)
    prebin_args = [
        (bin_spec, exclusion_map.get(i, set()))
        for i in range(len(model.metadata.tile_configs))
    ]

    tile_prebins = model.backend.call_all(
        model.workers, 'prebin_peak_data', args_per_actor=prebin_args,
    )
    t_prebin = time.perf_counter() - t0

    if verbose:
        logger.info("TD heatmap: pre-binned %d tiles in %.3fs",
                     len(tile_prebins), t_prebin)

    if all(not tp for tp in tile_prebins):
        logger.warning(
            "TD heatmap: all tiles returned empty prebins. "
            "Was peak tracking enabled via init_peak_tracking()?"
        )
        return

    # -- Merge and render -----------------------------------------------------
    t0 = time.perf_counter()
    merged = merge_tile_prebins(tile_prebins, bin_spec)
    t_merge = time.perf_counter() - t0

    if verbose:
        logger.info("TD heatmap: merged prebins in %.3fs", t_merge)

    if plot_layers is not None:
        merged = {k: v for k, v in merged.items() if k in plot_layers}

    _render_merged_heatmaps(
        merged_data=merged,
        bin_spec=bin_spec,
        output_dir=output_dir,
        title_prefix=title_prefix,
        net_name=net_name,
        verbose=verbose,
    )


def _render_merged_heatmaps(
    merged_data: Dict[str, List[Tuple[float, float, np.ndarray, np.ndarray]]],
    bin_spec: GlobalBinSpec,
    output_dir: str,
    is_current: bool = False,
    net_name: Optional[str] = None,
    title_prefix: Optional[str] = None,
    verbose: bool = False,
) -> None:
    """Render merged stripe data to per-layer PNG files.

    When *title_prefix* is ``None``, defaults to ``'IR-Drop'`` or
    ``'Current'`` based on *is_current*.  A custom prefix also sets the
    value label for non-current mode (e.g. ``"Peak IR-Drop (mV)"``).
    """
    from visualization.stripe_heatmap import render_from_prebinned_stripe_data

    if is_current:
        cmap = 'hot_r'
        value_label = 'Current (mA)'
        default_prefix = 'Current'
    else:
        cmap = 'RdYlGn_r'
        default_prefix = 'IR-Drop'

    effective_prefix = title_prefix if title_prefix is not None else default_prefix

    if not is_current:
        value_label = f'{effective_prefix} (mV)'

    if net_name:
        effective_prefix = f'{effective_prefix} ({net_name})'

    for layer, stripes in merged_data.items():
        spec = bin_spec.layers[layer]
        orientation = spec.orientation

        # Skip layers with no data (e.g., current heatmap for layers
        # that have no current sources — all bins will be zero/NaN).
        if is_current:
            has_data = any(
                np.any(bin_vals[~np.isnan(bin_vals)] > 0)
                for _, _, _, bin_vals in stripes
                if len(bin_vals) > 0
            )
            if not has_data:
                if verbose:
                    logger.info("  Skipping %s (no current sources)", layer)
                continue

        # Build output filename matching the previous naming convention
        output_file = os.path.join(
            output_dir,
            f'{effective_prefix.lower().replace(" ", "_").replace("(", "").replace(")", "")}'
            f'_heatmap_{layer}.png',
        )

        render_from_prebinned_stripe_data(
            stripe_data=stripes,
            orientation=orientation,
            output_path=output_file,
            layer=layer,
            cmap=cmap,
            title_prefix=effective_prefix,
            value_label=value_label,
            n_nodes=spec.n_nodes,
        )

        if verbose:
            logger.info("  Saved: %s", output_file)

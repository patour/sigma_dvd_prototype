#!/usr/bin/env python3
"""Dynamic IR-Drop Decomposition Analysis.

Analyzes dynamic IR-drop in a PDN netlist and decomposes the IR-drop at worst-case
instances into contributions from "near" instances (within a local window) and
"far" instances (outside the window). Optionally identifies the top-K aggressors
within each victim's near-window using dynamic adjoint sensitivity analysis,
reporting each aggressor's contribution to the victim IR-drop and distance.

This analysis helps identify whether IR-drop issues are caused by local current
density or by distributed grid resistance effects, and which specific aggressors
contribute most to each victim.

Usage:
    # Via command line arguments (with time units)
    python -m analysis.dynamic_irdrop_decomposition ./netlist/netlist_test \\
        --net VDD \\
        --start-time 0ns \\
        --end-time 100ns \\
        --dt 100ps \\
        --top-k 5 \\
        --window-percent 10 \\
        --integration trap \\
        --output results.json \\
        --plot \\
        --verbose

    # With aggressor analysis (top-10 per victim, dynamic adjoint)
    python -m analysis.dynamic_irdrop_decomposition ./netlist/netlist_test \\
        --net VDD --end-time 100ns --dt 100ps \\
        --aggressor-top-k 10 --adjoint-method dynamic \\
        --plot --output results.json

    # Via config file
    python -m analysis.dynamic_irdrop_decomposition --config config.yaml

Example output:
    ================================================================================
    DYNAMIC IR-DROP DECOMPOSITION ANALYSIS RESULTS
    ================================================================================
    Netlist: ./netlist/netlist_test
    Method: transient (RC)
    Time range: 0.00 ns - 100.00 ns (dt=0.10 ns)
    Window size: 10.0% of design

    TOP-5 WORST INSTANCES WITH NEAR/FAR DECOMPOSITION
    ================================================================================
    Rank  Instance                 Peak(mV)  Near(mV)  Far(mV)  Near%  Far%
    --------------------------------------------------------------------------------
    1     i_cpu_core:VDD:...       12.345    8.234     4.111    66.7%  33.3%
    ...

    TOP AGGRESSORS WITHIN NEAR-WINDOW (per victim)
    ================================================================================
    VICTIM #1: i_cpu_core:VDD:...
      Peak IR-drop: 12.345 mV at t=50.00 ns
      Self contribution: 2.100 mV (17.0%)
      Rank   Node                     Contrib(mV)  %        Distance(um)
      1      1500_2000_M1             3.234        26.2%    150.0
      ...
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import math
import os
import pickle
import re
import shutil
import sys
import time as time_module
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Literal, Mapping, Optional, Set, Tuple

import numpy as np
from visualization.stripe_heatmap import (
    aggregate_fast_imshow,
    build_binned_stripe_data,
    detect_orientation_from_coords,
    detect_orientation_from_edges,
    extract_node_data_vectorized,
    parse_node_info,
    render_binned_stripe_data_to_axes,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

# Optional YAML support
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


# =============================================================================
# Time Unit Parsing
# =============================================================================

def parse_time_value(s: str) -> float:
    """Parse time string with unit suffix to seconds.

    Supports: ps (picoseconds), ns (nanoseconds), us (microseconds),
              ms (milliseconds), s (seconds).

    Args:
        s: Time string like "10ps", "1ns", "100us", "1ms", "1s"
           or plain float (assumed seconds)

    Returns:
        Time value in seconds.

    Examples:
        >>> parse_time_value("10ps")
        1e-11
        >>> parse_time_value("1ns")
        1e-9
        >>> parse_time_value("100us")
        1e-4
        >>> parse_time_value("1e-9")
        1e-9
    """
    units = {'ps': 1e-12, 'ns': 1e-9, 'us': 1e-6, 'ms': 1e-3, 's': 1.0}
    s = str(s).strip().lower()

    for suffix, multiplier in units.items():
        if s.endswith(suffix):
            return float(s[:-len(suffix)]) * multiplier

    # No unit suffix - assume seconds
    return float(s)


def format_time_ns(t_seconds: float) -> str:
    """Format time in nanoseconds for display."""
    return f"{t_seconds * 1e9:.2f}"


# =============================================================================
# Data Classes
# =============================================================================

PEAK_IR_DROP_DATA_FILENAME = 'peak_ir_drop_per_node.json.gz'

@dataclass
class AggressorResult:
    """Top aggressor contribution to victim IR-drop.

    Attributes:
        node: Aggressor node name
        contribution_mV: Contribution to IR-drop in millivolts
        contribution_pct: Percentage of total IR-drop at victim
        source_names: List of current source instance names connected to this node
        distance_um: Euclidean distance to victim in um
    """
    node: str
    contribution_mV: float
    contribution_pct: float
    source_names: List[str]
    distance_um: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            'node': self.node,
            'contribution_mV': self.contribution_mV,
            'contribution_pct': self.contribution_pct,
            'distance_um': self.distance_um,
            'source_names': self.source_names,
        }


@dataclass
class InstanceDecomposition:
    """Decomposition results for a single worst instance.

    Contains time-domain waveforms showing total, near, and far IR-drop
    contributions, along with summary statistics.
    """
    instance_name: str
    node: str
    x: float
    y: float
    window_bounds: Tuple[float, float, float, float]  # (x_min, x_max, y_min, y_max)
    n_near_sources: int
    n_far_sources: int

    # Time arrays
    t_array: np.ndarray
    ir_drop_total: np.ndarray   # Full waveform
    ir_drop_near: np.ndarray    # Near contribution waveform
    ir_drop_far: np.ndarray     # Far contribution waveform

    # Peak statistics
    peak_total_mV: float
    peak_near_mV: float
    peak_far_mV: float
    peak_time_ns: float

    # Average statistics
    avg_total_mV: float
    avg_near_mV: float
    avg_far_mV: float

    # Fraction analysis
    near_fraction_at_peak: float   # % at time of peak
    far_fraction_at_peak: float
    avg_near_fraction: float       # Average % over time
    avg_far_fraction: float

    # Aggressor analysis (optional, populated when aggressor_top_k > 0)
    top_aggressors: List[AggressorResult] = field(default_factory=list)
    self_contribution_mV: float = 0.0
    self_contribution_pct: float = 0.0
    attribution_efficiency: float = 0.0
    near_total_mV: Optional[float] = None  # Total near-window contribution (mV); populated by distributed adjoint only

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary.

        Note: Waveform arrays (t_array, ir_drop_total, ir_drop_near, ir_drop_far)
        are NOT included to keep JSON files compact. Use the dataclass attributes
        directly for plotting or detailed analysis.
        """
        result = {
            'instance_name': self.instance_name,
            'node': self.node,
            'location': [self.x, self.y],
            'window_bounds': list(self.window_bounds),
            'n_near_sources': self.n_near_sources,
            'n_far_sources': self.n_far_sources,
            'peak_ir_drop': {
                'total_mV': self.peak_total_mV,
                'near_mV': self.peak_near_mV,
                'far_mV': self.peak_far_mV,
            },
            'peak_time_ns': self.peak_time_ns,
            'avg_ir_drop': {
                'total_mV': self.avg_total_mV,
                'near_mV': self.avg_near_mV,
                'far_mV': self.avg_far_mV,
            },
            'near_fraction_at_peak_percent': self.near_fraction_at_peak,
            'far_fraction_at_peak_percent': self.far_fraction_at_peak,
            'avg_near_fraction_percent': self.avg_near_fraction,
            'avg_far_fraction_percent': self.avg_far_fraction,
        }
        # Add aggressor data if present
        if self.top_aggressors:
            aggressor_data: Dict[str, Any] = {
                'self_contribution_mV': self.self_contribution_mV,
                'self_contribution_pct': self.self_contribution_pct,
                'attribution_efficiency': self.attribution_efficiency,
                'top_aggressors': [agg.to_dict() for agg in self.top_aggressors],
            }
            if self.near_total_mV is not None:
                aggressor_data['near_total_mV'] = self.near_total_mV
            result['aggressor_analysis'] = aggressor_data
        return result


@dataclass
class DecompositionResult:
    """Complete results of decomposition analysis."""
    netlist_dir: str
    net_name: str
    method: str   # 'transient' or 'quasi_static'
    integration_method: str   # 'trap' or 'be'
    t_start_ns: float
    t_end_ns: float
    dt_ns: float
    window_percent: float
    grid_bounds: Tuple[float, float, float, float]  # (x_min, x_max, y_min, y_max)
    worst_instances: List[InstanceDecomposition]
    timings: Dict[str, float] = field(default_factory=dict)
    peak_ir_drop_per_node: Dict[str, float] = field(default_factory=dict)  # node -> peak IR-drop (V)
    total_current_waveform: np.ndarray = field(default_factory=lambda: np.array([]))  # Design-wide total current (mA)
    t_array: np.ndarray = field(default_factory=lambda: np.array([]))  # Time array for total current plot
    smooth_sources: bool = True  # Whether PWL smoothing was applied to current sources

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary.

        Note: Large arrays are NOT included to keep JSON files compact:
        - peak_ir_drop_per_node: Use attribute directly for heatmaps
        - total_current_waveform, t_array: Use attributes directly for plotting
        - Per-instance waveforms: Use InstanceDecomposition attributes directly
        """
        return {
            'netlist_dir': self.netlist_dir,
            'net_name': self.net_name,
            'method': self.method,
            'integration_method': self.integration_method,
            'smooth_sources': self.smooth_sources,
            't_start_ns': self.t_start_ns,
            't_end_ns': self.t_end_ns,
            'dt_ns': self.dt_ns,
            'window_percent': self.window_percent,
            'grid_bounds': list(self.grid_bounds),
            'worst_instances': [inst.to_dict() for inst in self.worst_instances],
            'timings': self.timings,
            'peak_ir_drop_stats': {
                'n_nodes': len(self.peak_ir_drop_per_node),
                'max_mV': max(self.peak_ir_drop_per_node.values()) * 1000 if self.peak_ir_drop_per_node else 0,
                'min_mV': min(self.peak_ir_drop_per_node.values()) * 1000 if self.peak_ir_drop_per_node else 0,
            } if self.peak_ir_drop_per_node else None,
            'peak_ir_drop_data_file': (
                PEAK_IR_DROP_DATA_FILENAME if self.peak_ir_drop_per_node else None
            ),
            # Note: total_current_waveform and t_array are NOT serialized to keep JSON compact.
            # Use the dataclass attributes directly for plotting.
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def save_json(self, path: str) -> None:
        """Save results to JSON and persist peak IR-drop sidecar when present."""
        with open(path, 'w') as f:
            f.write(self.to_json())
        if self.peak_ir_drop_per_node:
            save_peak_ir_drop_data(
                self.peak_ir_drop_per_node,
                Path(path).with_name(PEAK_IR_DROP_DATA_FILENAME),
            )


def save_peak_ir_drop_data(
    peak_ir_drop_per_node: Mapping[str, float],
    path: str | Path,
) -> str:
    """Persist peak IR-drop data for later post-processing."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out_path, 'wt', encoding='utf-8') as f:
        json.dump(dict(peak_ir_drop_per_node), f, separators=(',', ':'))
    return str(out_path)


def load_peak_ir_drop_data(path: str | Path) -> Dict[str, float]:
    """Load persisted peak IR-drop data from JSON or gzip-compressed JSON."""
    in_path = Path(path)
    open_fn = gzip.open if in_path.suffix == '.gz' else open
    with open_fn(in_path, 'rt', encoding='utf-8') as f:
        data = json.load(f)
    if not isinstance(data, Mapping):
        raise ValueError(f"Peak IR-drop data at {in_path} must be a JSON object")
    return {str(node): float(value) for node, value in data.items()}


def _resolve_distributed_pkl_dir(netlist_dir: str | Path | None) -> Optional[Path]:
    """Return the distributed tile bundle path when one is available."""
    if netlist_dir is None:
        return None

    base_path = Path(netlist_dir)
    candidates = [base_path]
    if base_path.name != 'distributed_pkl':
        candidates.append(base_path / 'distributed_pkl')

    for candidate in candidates:
        if (candidate / 'metadata.pkl').is_file():
            return candidate
    return None


def _infer_layer_orientation_overrides_from_distributed_pkl(
    pkl_dir: str | Path,
    *,
    target_layers: Optional[Set[str]] = None,
) -> Dict[str, str]:
    """Infer layer routing orientation from distributed tile resistive edges."""
    pkl_path = Path(pkl_dir)
    tile_pkls = sorted(pkl_path.glob('tile_*_*.pkl'))
    if not tile_pkls:
        return {}

    normalized_layers = None if target_layers is None else {str(layer) for layer in target_layers}
    node_cache: Dict[str, Tuple[Optional[float], Optional[float], Optional[str]]] = {}
    layer_edge_counts: Dict[str, List[int]] = {}

    for tile_pkl in tile_pkls:
        with open(tile_pkl, 'rb') as f:
            tile_data = pickle.load(f)

        resistive_edges = getattr(tile_data, 'resistive_edges', None)
        if resistive_edges is None:
            raise TypeError(
                f"Distributed tile pickle is missing resistive edge data: {tile_pkl}"
            )

        for u, v, _conductance in resistive_edges:
            u_info = node_cache.get(u)
            if u_info is None:
                u_info = parse_node_info(u)
                node_cache[u] = u_info

            v_info = node_cache.get(v)
            if v_info is None:
                v_info = parse_node_info(v)
                node_cache[v] = v_info

            ux, uy, u_layer = u_info
            vx, vy, v_layer = v_info
            if ux is None or vx is None or u_layer != v_layer:
                continue

            layer = str(u_layer)
            if normalized_layers is not None and layer not in normalized_layers:
                continue

            dx = abs(vx - ux)
            dy = abs(vy - uy)
            if dx == 0 and dy == 0:
                continue

            counts = layer_edge_counts.setdefault(layer, [0, 0, 0])
            if dy == 0:
                counts[0] += 1
            elif dx == 0:
                counts[1] += 1
            else:
                counts[2] += 1

    return {
        layer: ('H' if h_count >= v_count else 'V')
        for layer, (h_count, v_count, _d_count) in layer_edge_counts.items()
    }


def _infer_peak_plot_layers(
    peak_ir_drop_per_node: Mapping[str, float],
    *,
    requested_layers: Optional[List[str]] = None,
) -> Set[str]:
    """Return layers that will be plotted for peak IR-drop heatmaps."""
    if requested_layers is not None:
        return {str(layer) for layer in requested_layers}

    detected_layers: Set[str] = set()
    for node in peak_ir_drop_per_node:
        _x, _y, layer = parse_node_info(str(node))
        if layer is not None:
            detected_layers.add(str(layer))
    return detected_layers


def _load_peak_plot_orientation_overrides(
    result: DecompositionResult | Mapping[str, Any],
    peak_ir_drop_per_node: Mapping[str, float],
    *,
    heatmap_layers: Optional[List[str]] = None,
    verbose: bool = False,
) -> Optional[Dict[str, str]]:
    """Load distributed layer-orientation metadata for decomposition heatmaps."""
    if isinstance(result, DecompositionResult):
        netlist_dir = result.netlist_dir
    else:
        netlist_dir = result.get('netlist_dir')

    pkl_dir = _resolve_distributed_pkl_dir(netlist_dir)
    if pkl_dir is None:
        return None

    target_layers = _infer_peak_plot_layers(
        peak_ir_drop_per_node,
        requested_layers=heatmap_layers,
    )

    try:
        overrides = _infer_layer_orientation_overrides_from_distributed_pkl(
            pkl_dir,
            target_layers=target_layers,
        )
    except (
        OSError,
        EOFError,
        ImportError,
        AttributeError,
        TypeError,
        ValueError,
        pickle.PickleError,
    ) as exc:
        warnings.warn(
            "Failed to infer peak heatmap layer orientations from distributed "
            f"tile data in {pkl_dir}: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        return None

    if verbose and overrides:
        print(f"Using distributed tile orientation metadata from: {pkl_dir}")

    return overrides or None


def load_saved_decomposition_result(path: str | Path) -> Dict[str, Any]:
    """Load a saved decomposition results.json file."""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Saved decomposition result at {path} must be a JSON object")
    return data


def resolve_peak_ir_drop_data_path(
    results_json_path: str | Path,
    saved_result: Mapping[str, Any],
    *,
    peak_ir_drop_data_path: Optional[str | Path] = None,
) -> Path:
    """Resolve the persisted peak IR-drop sidecar for a saved results.json file."""
    results_path = Path(results_json_path).resolve()
    results_dir = results_path.parent

    if peak_ir_drop_data_path is not None:
        candidate = Path(peak_ir_drop_data_path)
        if candidate.is_file():
            return candidate
        raise FileNotFoundError(
            f"Persisted peak IR-drop data file not found: {candidate}"
        )

    candidates: List[Path] = []
    configured_name = saved_result.get('peak_ir_drop_data_file')
    if configured_name:
        candidates.append(results_dir / str(configured_name))
    candidates.append(results_dir / PEAK_IR_DROP_DATA_FILENAME)
    candidates.append(results_dir / 'peak_ir_drop_per_node.json')

    seen: Set[Path] = set()
    deduped_candidates: List[Path] = []
    for candidate in candidates:
        if candidate not in seen:
            seen.add(candidate)
            deduped_candidates.append(candidate)

    for candidate in deduped_candidates:
        if candidate.is_file():
            return candidate

    candidate_list = ', '.join(str(path) for path in deduped_candidates)
    raise FileNotFoundError(
        "Saved decomposition output is missing persisted peak IR-drop data. "
        f"Looked for: {candidate_list}. "
        "results.json only stores peak_ir_drop_stats, so regenerate the saved output "
        "with the updated save_json() path or provide --peak-ir-drop-data."
    )


def _find_legacy_peak_ir_drop_plot_artifacts(
    plot_dir: str | Path,
) -> List[Path]:
    """Return legacy peak IR-drop plot artifacts already present on disk."""
    base_dir = Path(plot_dir)
    artifacts: List[Path] = []
    for pattern in (
        'peak_ir-drop_heatmap_*.png',
        os.path.join('peak_ir_drop_overlays', '*.png'),
    ):
        artifacts.extend(
            path for path in sorted(base_dir.glob(pattern))
            if path.is_file()
        )
    return artifacts


def _count_legacy_peak_ir_drop_plot_artifacts(
    artifacts: List[Path],
) -> Tuple[int, int]:
    """Count recovered legacy heatmap and overlay artifacts."""
    heatmap_count = 0
    overlay_count = 0
    for artifact in artifacts:
        if artifact.parent.name == 'peak_ir_drop_overlays':
            overlay_count += 1
        else:
            heatmap_count += 1
    return heatmap_count, overlay_count


def _format_count_with_label(count: int, singular: str, plural: str) -> str:
    """Return a count with the correct singular/plural label."""
    label = singular if count == 1 else plural
    return f"{count} {label}"


def _remove_stale_peak_ir_drop_outputs(plot_dir: str | Path) -> None:
    """Remove peak IR-drop artifacts that can survive narrower reruns."""
    base_dir = Path(plot_dir)

    def _unlink_or_warn(old_file: Path) -> None:
        try:
            old_file.unlink()
        except OSError as exc:
            warnings.warn(
                f"Failed to remove stale peak IR-drop artifact '{old_file}': {exc}",
                RuntimeWarning,
                stacklevel=2,
            )

    for old_file in base_dir.glob('peak_ir-drop_heatmap_*.png'):
        _unlink_or_warn(old_file)

    overlay_dir = base_dir / 'peak_ir_drop_overlays'
    for old_file in overlay_dir.glob('*.png'):
        _unlink_or_warn(old_file)


def recover_legacy_peak_ir_drop_plot_artifacts(
    results_json_path: str | Path,
    *,
    plot_dir: Optional[str | Path] = None,
    verbose: bool = False,
) -> Optional[str]:
    """Reuse legacy saved peak plot artifacts when no per-node sidecar exists."""
    results_path = Path(results_json_path).resolve()
    legacy_plot_dir = results_path.parent / 'plots'
    artifacts = _find_legacy_peak_ir_drop_plot_artifacts(legacy_plot_dir)
    if not artifacts:
        return None

    target_plot_dir = (
        Path(plot_dir).resolve() if plot_dir is not None else legacy_plot_dir
    )
    target_plot_dir.mkdir(parents=True, exist_ok=True)
    if target_plot_dir != legacy_plot_dir:
        _remove_stale_peak_ir_drop_outputs(target_plot_dir)

    for src in artifacts:
        dst = target_plot_dir / src.relative_to(legacy_plot_dir)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.resolve() != dst.resolve():
            shutil.copy2(src, dst)

    heatmap_count, overlay_count = _count_legacy_peak_ir_drop_plot_artifacts(artifacts)
    if verbose:
        print(
            "Persisted peak IR-drop sidecar not found; "
            f"reusing legacy plot artifacts from: {legacy_plot_dir}"
        )
        print(
            "Recovered legacy peak IR-drop artifacts: "
            f"{len(artifacts)} total "
            f"({_format_count_with_label(heatmap_count, 'heatmap', 'heatmaps')}, "
            f"{_format_count_with_label(overlay_count, 'overlay', 'overlays')}). "
            "Only files already present on disk were reused."
        )

    return str(target_plot_dir)


# =============================================================================
# Overlay Plot Preparation
# =============================================================================


@dataclass(frozen=True)
class OverlayMarkerMetadata:
    """Prepared marker metadata for victim/aggressor overlay plots."""

    node: str
    x: float
    y: float
    layer: str
    marker: str
    edge_color: str
    face_color: Optional[str]
    filled: bool
    face_color_mode: Literal['none', 'fixed', 'strength_mapped'] = 'none'
    color_value: Optional[float] = None
    contribution_mV: Optional[float] = None


@dataclass(frozen=True)
class OverlayWindowMetadata:
    """Prepared near-window metadata for victim overlay plots."""

    x_min: float
    x_max: float
    y_min: float
    y_max: float
    layer: str
    edge_color: str = 'red'


@dataclass(frozen=True)
class VictimOverlayPlotData:
    """Prepared per-victim overlay inputs for live or saved decomposition results."""

    instance_name: str
    victim_node: str
    victim_layer: str
    victim_location: Tuple[float, float]
    zoom_bounds: Tuple[float, float, float, float]
    victim_marker: OverlayMarkerMetadata
    aggressor_markers: List[OverlayMarkerMetadata]
    near_window_box: OverlayWindowMetadata


@dataclass(frozen=True)
class AggressorBarMetadata:
    """Prepared bar metadata for aggressor contribution plots."""

    node: str
    contribution_mV: float
    contribution_pct: float
    color_value: float


@dataclass(frozen=True)
class VictimAggressorBarPlotData:
    """Prepared per-victim aggressor bar-chart inputs."""

    instance_name: str
    victim_node: str
    peak_total_mV: float
    self_contribution_mV: float
    aggressors: List[AggressorBarMetadata]


@dataclass(frozen=True)
class _PreparedPeakIRDropOverlayData:
    """Parsed peak IR-drop arrays reused across overlay renders."""

    nodes: np.ndarray
    xs: np.ndarray
    ys: np.ndarray
    layers: np.ndarray
    values: np.ndarray


def _get_instance_field(
    instance: InstanceDecomposition | Mapping[str, Any],
    field_name: str,
    default: Any = None,
) -> Any:
    """Read an instance field from a live dataclass or saved-result mapping."""
    if isinstance(instance, Mapping):
        return instance.get(field_name, default)
    return getattr(instance, field_name, default)


def _get_top_aggressors(
    instance: InstanceDecomposition | Mapping[str, Any],
) -> List[AggressorResult | Mapping[str, Any]]:
    """Return top aggressors from a live instance or saved JSON instance."""
    if isinstance(instance, Mapping):
        aggressor_analysis = instance.get('aggressor_analysis') or {}
        top_aggressors = aggressor_analysis.get('top_aggressors') or []
        return list(top_aggressors)
    return list(instance.top_aggressors)


def _get_aggressor_field(
    aggressor: AggressorResult | Mapping[str, Any],
    field_name: str,
    default: Any = None,
) -> Any:
    """Read an aggressor field from a live dataclass or saved-result mapping."""
    if isinstance(aggressor, Mapping):
        return aggressor.get(field_name, default)
    return getattr(aggressor, field_name, default)


def _compute_victim_centered_zoom_bounds(
    victim_x: float,
    victim_y: float,
    window_bounds: Tuple[float, float, float, float],
    aggressor_coords: List[Tuple[float, float]],
    padding_ratio: float = 0.10,
) -> Tuple[float, float, float, float]:
    """Compute victim-centered zoom bounds covering window and aggressors."""
    x_min, x_max, y_min, y_max = window_bounds
    x_radius = max(abs(victim_x - x_min), abs(x_max - victim_x), 1.0)
    y_radius = max(abs(victim_y - y_min), abs(y_max - victim_y), 1.0)

    for agg_x, agg_y in aggressor_coords:
        x_radius = max(x_radius, abs(agg_x - victim_x))
        y_radius = max(y_radius, abs(agg_y - victim_y))

    x_radius *= (1.0 + padding_ratio)
    y_radius *= (1.0 + padding_ratio)

    return (
        victim_x - x_radius,
        victim_x + x_radius,
        victim_y - y_radius,
        victim_y + y_radius,
    )


def _compute_relative_strength_values(strengths: List[float]) -> List[float]:
    """Normalize aggressor strengths onto the [0, 1] range."""
    if not strengths:
        return []

    min_strength = min(strengths)
    max_strength = max(strengths)
    strength_span = max_strength - min_strength
    if strength_span == 0.0:
        return [1.0 for _ in strengths]
    return [
        (strength - min_strength) / strength_span
        for strength in strengths
    ]


def _get_instance_peak_total_mV(
    instance: InstanceDecomposition | Mapping[str, Any],
) -> float:
    """Return peak total IR-drop in mV from a live or saved instance."""
    if isinstance(instance, Mapping):
        peak_ir_drop = instance.get('peak_ir_drop') or {}
        if isinstance(peak_ir_drop, Mapping) and 'total_mV' in peak_ir_drop:
            return float(peak_ir_drop.get('total_mV', 0.0))
    return float(_get_instance_field(instance, 'peak_total_mV', 0.0))


def _get_instance_self_contribution_mV(
    instance: InstanceDecomposition | Mapping[str, Any],
) -> float:
    """Return victim self-contribution in mV from a live or saved instance."""
    if isinstance(instance, Mapping):
        aggressor_analysis = instance.get('aggressor_analysis') or {}
        if isinstance(aggressor_analysis, Mapping):
            return float(aggressor_analysis.get('self_contribution_mV', 0.0))
    return float(_get_instance_field(instance, 'self_contribution_mV', 0.0))


def prepare_victim_overlay_plot_data(
    instance: InstanceDecomposition | Mapping[str, Any],
    max_aggressors: int = 10,
) -> Optional[VictimOverlayPlotData]:
    """Prepare per-victim overlay inputs from a live instance or saved result.

    Args:
        instance: InstanceDecomposition or saved-result mapping from results.json.
        max_aggressors: Maximum number of same-layer aggressors to prepare.

    Returns:
        Prepared overlay inputs, or None if victim coordinates/layer cannot be derived.
    """
    victim_node = _get_instance_field(instance, 'node')
    instance_name = _get_instance_field(instance, 'instance_name', victim_node)
    victim_x, victim_y, victim_layer = parse_node_info(victim_node)

    location = _get_instance_field(instance, 'location')
    if location and len(location) >= 2:
        victim_x = float(location[0])
        victim_y = float(location[1])
    else:
        victim_x = _get_instance_field(instance, 'x', victim_x)
        victim_y = _get_instance_field(instance, 'y', victim_y)

    if victim_x is None or victim_y is None or victim_layer is None:
        return None

    window_bounds_raw = _get_instance_field(instance, 'window_bounds')
    if window_bounds_raw is None or len(window_bounds_raw) != 4:
        return None
    window_bounds = tuple(float(v) for v in window_bounds_raw)

    same_layer_aggressors: List[OverlayMarkerMetadata] = []
    strengths: List[float] = []
    max_aggressors = max(0, max_aggressors)

    for aggressor in _get_top_aggressors(instance):
        if len(same_layer_aggressors) >= max_aggressors:
            break
        aggressor_node = _get_aggressor_field(aggressor, 'node')
        agg_x, agg_y, agg_layer = parse_node_info(aggressor_node)
        if agg_x is None or agg_y is None or agg_layer != victim_layer:
            continue

        contribution_mV = float(_get_aggressor_field(aggressor, 'contribution_mV', 0.0))
        strength = abs(contribution_mV)
        strengths.append(strength)
        same_layer_aggressors.append(
            OverlayMarkerMetadata(
                node=aggressor_node,
                x=agg_x,
                y=agg_y,
                layer=agg_layer,
                marker='o',
                edge_color='none',
                face_color=None,
                filled=True,
                face_color_mode='strength_mapped',
                contribution_mV=contribution_mV,
            )
        )

    if strengths:
        color_values = _compute_relative_strength_values(strengths)
        aggressor_markers = []
        for marker, color_value in zip(same_layer_aggressors, color_values):
            aggressor_markers.append(
                OverlayMarkerMetadata(
                    node=marker.node,
                    x=marker.x,
                    y=marker.y,
                    layer=marker.layer,
                    marker=marker.marker,
                    edge_color=marker.edge_color,
                    face_color=None,
                    filled=marker.filled,
                    face_color_mode=marker.face_color_mode,
                    color_value=color_value,
                    contribution_mV=marker.contribution_mV,
                )
            )
    else:
        aggressor_markers = []

    zoom_bounds = _compute_victim_centered_zoom_bounds(
        victim_x,
        victim_y,
        window_bounds,
        [(marker.x, marker.y) for marker in aggressor_markers],
    )

    return VictimOverlayPlotData(
        instance_name=instance_name,
        victim_node=victim_node,
        victim_layer=victim_layer,
        victim_location=(victim_x, victim_y),
        zoom_bounds=zoom_bounds,
        victim_marker=OverlayMarkerMetadata(
            node=victim_node,
            x=victim_x,
            y=victim_y,
            layer=victim_layer,
            marker='x',
            edge_color='red',
            face_color=None,
            filled=False,
        ),
        aggressor_markers=aggressor_markers,
        near_window_box=OverlayWindowMetadata(
            x_min=window_bounds[0],
            x_max=window_bounds[1],
            y_min=window_bounds[2],
            y_max=window_bounds[3],
            layer=victim_layer,
        ),
    )


def prepare_victim_aggressor_bar_plot_data(
    instance: InstanceDecomposition | Mapping[str, Any],
    max_aggressors: int = 10,
) -> Optional[VictimAggressorBarPlotData]:
    """Prepare aggressor contribution bar-chart inputs from live or saved data."""
    victim_node = _get_instance_field(instance, 'node')
    instance_name = _get_instance_field(instance, 'instance_name', victim_node)
    aggressors = sorted(
        _get_top_aggressors(instance),
        key=lambda aggressor: abs(float(_get_aggressor_field(aggressor, 'contribution_mV', 0.0))),
        reverse=True,
    )[:max(0, max_aggressors)]
    if not aggressors:
        return None

    strengths = [
        abs(float(_get_aggressor_field(aggressor, 'contribution_mV', 0.0)))
        for aggressor in aggressors
    ]
    color_values = _compute_relative_strength_values(strengths)
    prepared_aggressors = [
        AggressorBarMetadata(
            node=str(_get_aggressor_field(aggressor, 'node', '')),
            contribution_mV=float(_get_aggressor_field(aggressor, 'contribution_mV', 0.0)),
            contribution_pct=float(_get_aggressor_field(aggressor, 'contribution_pct', 0.0)),
            color_value=color_value,
        )
        for aggressor, color_value in zip(aggressors, color_values)
    ]
    return VictimAggressorBarPlotData(
        instance_name=str(instance_name),
        victim_node=str(victim_node),
        peak_total_mV=_get_instance_peak_total_mV(instance),
        self_contribution_mV=_get_instance_self_contribution_mV(instance),
        aggressors=prepared_aggressors,
    )


def prepare_decomposition_aggressor_bar_plot_data(
    result: DecompositionResult | Mapping[str, Any],
    max_aggressors: int = 10,
) -> List[VictimAggressorBarPlotData]:
    """Prepare bar-chart inputs for each victim in a live or saved result."""
    worst_instances = (
        result.get('worst_instances', [])
        if isinstance(result, Mapping)
        else result.worst_instances
    )
    prepared: List[VictimAggressorBarPlotData] = []
    for instance in worst_instances:
        plot_data = prepare_victim_aggressor_bar_plot_data(
            instance,
            max_aggressors=max_aggressors,
        )
        if plot_data is not None:
            prepared.append(plot_data)
    return prepared


def prepare_decomposition_overlay_plot_data(
    result: DecompositionResult | Mapping[str, Any],
    max_aggressors: int = 10,
) -> List[VictimOverlayPlotData]:
    """Prepare overlay inputs for each victim in a live or saved decomposition result."""
    worst_instances = (
        result.get('worst_instances', [])
        if isinstance(result, Mapping)
        else result.worst_instances
    )
    prepared: List[VictimOverlayPlotData] = []
    for instance in worst_instances:
        overlay_data = prepare_victim_overlay_plot_data(
            instance,
            max_aggressors=max_aggressors,
        )
        if overlay_data is not None:
            prepared.append(overlay_data)
    return prepared


def _coerce_victim_overlay_plot_data(
    overlay_data: VictimOverlayPlotData | InstanceDecomposition | Mapping[str, Any],
    max_aggressors: int = 10,
) -> VictimOverlayPlotData:
    """Normalize live/saved victim input into prepared overlay metadata."""
    if isinstance(overlay_data, VictimOverlayPlotData):
        return overlay_data

    prepared = prepare_victim_overlay_plot_data(
        overlay_data,
        max_aggressors=max_aggressors,
    )
    if prepared is None:
        raise ValueError("Could not prepare overlay plot data for victim")
    return prepared


def _prepare_peak_ir_drop_overlay_data(
    peak_ir_drop_per_node: Mapping[str, float],
) -> _PreparedPeakIRDropOverlayData:
    """Parse peak IR-drop mapping once for overlay plotting."""
    nodes, xs, ys, layers, values = extract_node_data_vectorized(peak_ir_drop_per_node)
    return _PreparedPeakIRDropOverlayData(
        nodes=nodes,
        xs=xs,
        ys=ys,
        layers=layers,
        values=values,
    )


def _extract_peak_ir_drop_overlay_region(
    prepared_peak_ir_drop: _PreparedPeakIRDropOverlayData,
    overlay_data: VictimOverlayPlotData,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract zoom-window peak IR-drop values on the victim layer."""
    layer_mask = prepared_peak_ir_drop.layers == overlay_data.victim_layer
    if not np.any(layer_mask):
        raise ValueError(
            f"No peak IR-drop data found on victim layer {overlay_data.victim_layer}"
        )

    xs = prepared_peak_ir_drop.xs[layer_mask]
    ys = prepared_peak_ir_drop.ys[layer_mask]
    values = prepared_peak_ir_drop.values[layer_mask]

    x_min, x_max, y_min, y_max = overlay_data.zoom_bounds
    zoom_mask = (
        (xs >= x_min) & (xs <= x_max) &
        (ys >= y_min) & (ys <= y_max)
    )
    if not np.any(zoom_mask):
        raise ValueError(
            "No peak IR-drop samples found inside zoom window "
            f"for {overlay_data.instance_name} on layer {overlay_data.victim_layer}"
        )

    xs = xs[zoom_mask]
    ys = ys[zoom_mask]
    values = values[zoom_mask]

    return xs, ys, values


def _resolve_peak_ir_drop_overlay_orientation(
    prepared_peak_ir_drop: _PreparedPeakIRDropOverlayData,
    overlay_data: VictimOverlayPlotData,
    *,
    graph: Optional[Any] = None,
    orientation_overrides: Optional[Mapping[str, str]] = None,
    orientation_threshold: float = 2.0,
) -> str:
    """Resolve overlay layer orientation using the same precedence as peak heatmaps."""
    layer = overlay_data.victim_layer

    orientation_override = None
    if orientation_overrides is not None:
        orientation_override = orientation_overrides.get(layer)
        if orientation_override is None:
            orientation_override = orientation_overrides.get(str(layer))

    if orientation_override is not None:
        if orientation_override not in {'H', 'V', 'MIXED'}:
            raise ValueError(
                f"Invalid orientation override for layer {layer}: {orientation_override!r}"
            )
        return orientation_override

    layer_mask = prepared_peak_ir_drop.layers == layer
    if not np.any(layer_mask):
        raise ValueError(f"No peak IR-drop data found on victim layer {layer}")

    if graph is not None:
        edge_orientation = detect_orientation_from_edges(
            graph,
            prepared_peak_ir_drop.nodes[layer_mask].tolist(),
            layer,
            max_edges=5000,
            threshold=0.70,
        )
        if edge_orientation != 'MIXED':
            return edge_orientation

    return detect_orientation_from_coords(
        prepared_peak_ir_drop.xs[layer_mask],
        prepared_peak_ir_drop.ys[layer_mask],
        orientation_threshold,
    )


def _plot_victim_overlay_peak_ir_drop_heatmap_prepared(
    prepared_peak_ir_drop: _PreparedPeakIRDropOverlayData,
    prepared: VictimOverlayPlotData,
    *,
    ax: Optional['Axes'] = None,
    heatmap_cmap: str = 'RdYlGn_r',
    aggressor_cmap: str = 'plasma',
    title: Optional[str] = None,
    show: bool = False,
    save_path: Optional[str] = None,
    unit_scale: float = 1000.0,
    unit_label: str = 'mV',
    draw_aggressor_colorbar: bool = False,
    graph: Optional[Any] = None,
    orientation_overrides: Optional[Mapping[str, str]] = None,
    max_stripes: int = 500,
    stripe_bin_size: Optional[int] = None,
    target_bins_per_stripe: int = 500,
    aggregation: str = 'max',
    orientation_threshold: float = 2.0,
    **scatter_kwargs: Any,
) -> Tuple['Figure', 'Axes']:
    """Render a zoomed peak IR-drop heatmap from pre-parsed peak arrays."""
    import matplotlib
    if not show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from matplotlib.patches import Rectangle

    xs, ys, values = _extract_peak_ir_drop_overlay_region(
        prepared_peak_ir_drop,
        prepared,
    )
    orientation = _resolve_peak_ir_drop_overlay_orientation(
        prepared_peak_ir_drop,
        prepared,
        graph=graph,
        orientation_overrides=orientation_overrides,
        orientation_threshold=orientation_threshold,
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure

    scaled_values = values * unit_scale
    kwargs = {'s': 40, 'edgecolors': 'k', 'linewidths': 0.3}
    kwargs.update(scatter_kwargs)

    if orientation == 'MIXED':
        try:
            x_edges, y_edges, grid = aggregate_fast_imshow(
                xs, ys, scaled_values, orientation, aggregation=aggregation
            )
            heatmap = ax.imshow(
                grid,
                extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
                origin='lower',
                aspect='auto',
                cmap=heatmap_cmap,
                interpolation='bilinear',
                zorder=1,
            )
        except ImportError:
            heatmap = ax.scatter(
                xs,
                ys,
                c=scaled_values,
                cmap=heatmap_cmap,
                zorder=1,
                **kwargs,
            )
    else:
        stripe_data, vmin, vmax = build_binned_stripe_data(
            xs,
            ys,
            scaled_values,
            orientation,
            max_stripes=max_stripes,
            stripe_bin_size=stripe_bin_size,
            aggregation=aggregation,
            target_bins_per_stripe=target_bins_per_stripe,
        )
        if not stripe_data:
            raise ValueError(
                "No peak IR-drop samples remained after stripe binning for "
                f"{prepared.instance_name} on layer {prepared.victim_layer}"
            )
        heatmap = render_binned_stripe_data_to_axes(
            ax,
            stripe_data,
            orientation,
            cmap=heatmap_cmap,
            vmin=vmin,
            vmax=vmax,
        )
        if heatmap is None:
            raise ValueError(
                "Failed to render stripe-binned peak IR-drop overlay for "
                f"{prepared.instance_name} on layer {prepared.victim_layer}"
            )

    fig.colorbar(heatmap, ax=ax, label=f'Peak IR-Drop ({unit_label})')

    window = prepared.near_window_box
    ax.add_patch(
        Rectangle(
            (window.x_min, window.y_min),
            window.x_max - window.x_min,
            window.y_max - window.y_min,
            edgecolor=window.edge_color,
            facecolor='none',
            linestyle='--',
            linewidth=2.0,
            zorder=3,
        )
    )

    aggressor_markers = prepared.aggressor_markers
    if aggressor_markers:
        ax.scatter(
            [marker.x for marker in aggressor_markers],
            [marker.y for marker in aggressor_markers],
            c=[
                marker.color_value if marker.color_value is not None else 0.0
                for marker in aggressor_markers
            ],
            cmap=aggressor_cmap,
            vmin=0.0,
            vmax=1.0,
            s=110,
            marker='o',
            edgecolors='black',
            linewidths=0.6,
            zorder=4,
        )
        if draw_aggressor_colorbar:
            fig.colorbar(
                ScalarMappable(norm=Normalize(vmin=0.0, vmax=1.0), cmap=aggressor_cmap),
                ax=ax,
                label='Aggressor Strength (relative)',
            )
    ax.plot(
        prepared.victim_marker.x,
        prepared.victim_marker.y,
        marker=prepared.victim_marker.marker,
        color=prepared.victim_marker.edge_color,
        markersize=12,
        markeredgewidth=2.5,
        linestyle='None',
        zorder=5,
    )

    x_min, x_max, y_min, y_max = prepared.zoom_bounds
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(
        title or
        f'Peak IR-Drop Overlay - {prepared.instance_name} (Layer {prepared.victim_layer})'
    )

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()

    return fig, ax


def plot_victim_overlay_peak_ir_drop_heatmap(
    peak_ir_drop_per_node: Mapping[str, float],
    overlay_data: VictimOverlayPlotData | InstanceDecomposition | Mapping[str, Any],
    *,
    max_aggressors: int = 10,
    ax: Optional['Axes'] = None,
    heatmap_cmap: str = 'RdYlGn_r',
    aggressor_cmap: str = 'plasma',
    title: Optional[str] = None,
    show: bool = False,
    save_path: Optional[str] = None,
    unit_scale: float = 1000.0,
    unit_label: str = 'mV',
    draw_aggressor_colorbar: bool = False,
    graph: Optional[Any] = None,
    orientation_overrides: Optional[Mapping[str, str]] = None,
    max_stripes: int = 500,
    stripe_bin_size: Optional[int] = None,
    target_bins_per_stripe: int = 500,
    aggregation: str = 'max',
    orientation_threshold: float = 2.0,
    **scatter_kwargs: Any,
) -> Tuple['Figure', 'Axes']:
    """Render a zoomed peak IR-drop heatmap with victim/aggressor overlays."""
    prepared = _coerce_victim_overlay_plot_data(
        overlay_data,
        max_aggressors=max_aggressors,
    )
    prepared_peak_ir_drop = _prepare_peak_ir_drop_overlay_data(
        peak_ir_drop_per_node,
    )
    return _plot_victim_overlay_peak_ir_drop_heatmap_prepared(
        prepared_peak_ir_drop,
        prepared,
        ax=ax,
        heatmap_cmap=heatmap_cmap,
        aggressor_cmap=aggressor_cmap,
        title=title,
        show=show,
        save_path=save_path,
        unit_scale=unit_scale,
        unit_label=unit_label,
        draw_aggressor_colorbar=draw_aggressor_colorbar,
        graph=graph,
        orientation_overrides=orientation_overrides,
        max_stripes=max_stripes,
        stripe_bin_size=stripe_bin_size,
        target_bins_per_stripe=target_bins_per_stripe,
        aggregation=aggregation,
        orientation_threshold=orientation_threshold,
        **scatter_kwargs,
    )


def _sanitize_overlay_filename_component(value: str) -> str:
    """Return a filesystem-friendly filename component for overlay plot outputs."""
    sanitized = re.sub(r'[^0-9A-Za-z._-]+', '_', value.strip())
    sanitized = sanitized.strip('._')
    return sanitized or 'victim'


def _resolve_overlay_save_path(
    base_save_path: str | Path,
    overlay_data: VictimOverlayPlotData,
    index: int,
) -> str:
    """Derive a unique per-victim save path for bulk overlay rendering."""
    base_path = Path(base_save_path)
    instance_slug = _sanitize_overlay_filename_component(overlay_data.instance_name)
    layer_slug = _sanitize_overlay_filename_component(overlay_data.victim_layer)
    victim_index = index + 1

    if base_path.suffix:
        save_path = base_path.with_name(
            f"{base_path.stem}_{victim_index}_{instance_slug}_{layer_slug}"
            f"{base_path.suffix}"
        )
    else:
        save_path = base_path / (
            f"peak_ir_drop_overlay_{victim_index}_{instance_slug}_{layer_slug}.png"
        )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    return str(save_path)


def plot_victim_aggressor_bar_chart(
    plot_data: VictimAggressorBarPlotData | InstanceDecomposition | Mapping[str, Any],
    *,
    max_aggressors: int = 10,
    ax: Optional['Axes'] = None,
    aggressor_cmap: str = 'plasma',
    title: Optional[str] = None,
    show: bool = False,
    save_path: Optional[str | Path] = None,
) -> Tuple['Figure', 'Axes']:
    """Render a top-aggressor contribution bar chart for one victim."""
    import matplotlib
    if not show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    prepared = (
        plot_data
        if isinstance(plot_data, VictimAggressorBarPlotData)
        else prepare_victim_aggressor_bar_plot_data(
            plot_data,
            max_aggressors=max_aggressors,
        )
    )
    if prepared is None:
        raise ValueError("Could not prepare aggressor bar-plot data for victim")

    if ax is None:
        fig_height = max(4.5, 2.5 + 0.5 * max(1, len(prepared.aggressors)))
        fig, ax = plt.subplots(figsize=(10, fig_height))
    else:
        fig = ax.figure

    aggressors = prepared.aggressors
    y_pos = np.arange(len(aggressors))
    cmap = matplotlib.colormaps.get_cmap(aggressor_cmap)
    bar_colors = cmap([aggressor.color_value for aggressor in aggressors])
    contributions = [aggressor.contribution_mV for aggressor in aggressors]
    labels = [aggressor.node for aggressor in aggressors]

    bars = ax.barh(
        y_pos,
        contributions,
        color=bar_colors,
        edgecolor='black',
        linewidth=0.5,
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel('Contribution (mV)')
    ax.set_ylabel('Aggressor Node')
    ax.grid(axis='x', alpha=0.3)

    x_min = min([0.0, *contributions]) if contributions else 0.0
    x_max = max([0.0, *contributions]) if contributions else 1.0
    if x_min == x_max:
        x_max = x_min + 1.0
    x_span = x_max - x_min
    label_offset = 0.02 * x_span if x_span > 0 else 0.05
    ax.set_xlim(x_min - 0.05 * x_span, x_max + 0.20 * x_span + label_offset)

    label_artists = []
    for bar, aggressor in zip(bars, aggressors):
        contribution = aggressor.contribution_mV
        label_x = contribution + label_offset
        label_artists.append(ax.text(
            label_x,
            bar.get_y() + bar.get_height() / 2.0,
            f'{contribution:.2f} mV | {aggressor.contribution_pct:.1f}%',
            va='center',
            ha='left',
            fontsize=8,
        ))

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    axis_bbox = ax.get_window_extent(renderer=renderer)
    if axis_bbox.width > 0:
        data_per_pixel = (ax.get_xlim()[1] - ax.get_xlim()[0]) / axis_bbox.width
        overflow_right = max(
            (
                max(0.0, text.get_window_extent(renderer=renderer).x1 - axis_bbox.x1)
                for text in label_artists
            ),
            default=0.0,
        )
        if overflow_right > 0.0:
            x0, x1 = ax.get_xlim()
            ax.set_xlim(x0, x1 + overflow_right * data_per_pixel + 4.0 * data_per_pixel)

    ax.set_title(
        title or (
            'Top Aggressor Contributions\n'
            f'{truncate_name(prepared.instance_name)} ({prepared.victim_node}) | '
            f'Peak: {prepared.peak_total_mV:.3f} mV | '
            f'Self: {prepared.self_contribution_mV:.3f} mV'
        )
    )

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()

    return fig, ax


def generate_aggressor_contribution_plots(
    result: DecompositionResult | Mapping[str, Any],
    plot_dir: str | Path,
    *,
    show: bool = False,
    max_aggressors: int = 10,
    verbose: bool = False,
) -> None:
    """Generate per-victim top-aggressor contribution bar charts."""
    import matplotlib.pyplot as plt

    os.makedirs(plot_dir, exist_ok=True)
    for old_file in glob.glob(os.path.join(str(plot_dir), 'aggressors_*.png')):
        try:
            os.remove(old_file)
        except OSError:
            pass

    prepared_plots = prepare_decomposition_aggressor_bar_plot_data(
        result,
        max_aggressors=max_aggressors,
    )
    if verbose and prepared_plots:
        print("Generating aggressor contribution plots...")

    for idx, plot_data in enumerate(prepared_plots):
        fig, _ = plot_victim_aggressor_bar_chart(
            plot_data,
            show=show,
            save_path=Path(plot_dir) / f'aggressors_{idx + 1}.png',
        )
        plt.close(fig)


def plot_decomposition_overlay_peak_ir_drop_heatmaps(
    peak_ir_drop_per_node: Mapping[str, float],
    result: DecompositionResult | Mapping[str, Any],
    *,
    max_aggressors: int = 10,
    save_path: Optional[str | Path] = None,
    on_error: Optional[Callable[[VictimOverlayPlotData, Exception], None]] = None,
    orientation_overrides: Optional[Mapping[str, str]] = None,
    **plot_kwargs: Any,
) -> List[Tuple[VictimOverlayPlotData, 'Figure', 'Axes']]:
    """Render per-victim overlay peak IR-drop heatmaps for a decomposition result."""
    rendered: List[Tuple[VictimOverlayPlotData, 'Figure', 'Axes']] = []
    plot_kwargs.pop('save_path', None)
    prepared_overlays = prepare_decomposition_overlay_plot_data(
        result,
        max_aggressors=max_aggressors,
    )
    if not prepared_overlays:
        return rendered

    prepared_peak_ir_drop = _prepare_peak_ir_drop_overlay_data(peak_ir_drop_per_node)
    for idx, overlay_data in enumerate(prepared_overlays):
        overlay_save_path = (
            _resolve_overlay_save_path(save_path, overlay_data, idx)
            if save_path is not None else None
        )
        try:
            fig, ax = _plot_victim_overlay_peak_ir_drop_heatmap_prepared(
                prepared_peak_ir_drop,
                overlay_data,
                save_path=overlay_save_path,
                orientation_overrides=orientation_overrides,
                **plot_kwargs,
            )
        except Exception as exc:
            if on_error is not None:
                on_error(overlay_data, exc)
            else:
                warnings.warn(
                    "Skipping peak IR-drop overlay render for "
                    f"{overlay_data.instance_name} ({overlay_data.victim_layer}): {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
            continue
        rendered.append((overlay_data, fig, ax))
    return rendered


# =============================================================================
# Solver Backend Configuration
# =============================================================================

def configure_solver_backend(config: Dict[str, Any]) -> None:
    """Apply solver backend settings from config.

    Args:
        config: Dict with optional keys: 'backend', 'ordering', 'mode', 'use_long'
    """
    from solver.unified_solver import (
        set_use_cholmod, set_cholmod_ordering,
        set_cholmod_mode, set_cholmod_use_long
    )

    backend = config.get('backend', 'auto')
    if backend == 'auto':
        set_use_cholmod(None)
    elif backend == 'cholmod':
        set_use_cholmod(True)
    elif backend == 'splu':
        set_use_cholmod(False)

    ordering = config.get('ordering')
    if ordering is not None:
        set_cholmod_ordering(ordering)

    mode = config.get('mode')
    if mode is not None:
        set_cholmod_mode(mode)

    use_long = config.get('use_long')
    if use_long is not None:
        set_cholmod_use_long(use_long)


# =============================================================================
# Location Extraction
# =============================================================================

def parse_node_coordinates(node_name: str) -> Tuple[Optional[float], Optional[float]]:
    """Extract (x, y) coordinates from node name.

    Node name format: <x>_<y>_<layer> (e.g., "1000_2000_M1")

    Args:
        node_name: Node name string

    Returns:
        (x, y) tuple, or (None, None) if parsing fails.
    """
    parts = str(node_name).split('_')
    if len(parts) >= 2:
        try:
            x = float(parts[0])
            y = float(parts[1])
            return (x, y)
        except ValueError:
            pass
    return (None, None)


def extract_instance_locations(
    current_sources: Dict[str, Any],
) -> Tuple[Dict[str, Tuple[float, float]], Tuple[float, float, float, float]]:
    """Extract (x, y) for each current source instance.

    Uses InstanceInfo.tile_x/tile_y if available, otherwise parses from node name.

    Args:
        current_sources: Dict mapping source name to CurrentSource object

    Returns:
        Tuple of:
        - instance_coords: Dict mapping source name to (x, y) tuple
        - grid_bounds: (x_min, x_max, y_min, y_max) bounding box
    """
    instance_coords: Dict[str, Tuple[float, float]] = {}

    for name, src in current_sources.items():
        x, y = None, None

        # Try to get from InstanceInfo (skip if tile_x/tile_y are 0, as that's often uninformative)
        if hasattr(src, 'info') and src.info is not None:
            tile_x = getattr(src.info, 'tile_x', None)
            tile_y = getattr(src.info, 'tile_y', None)
            if tile_x and tile_y:  # Only use if both are non-zero
                x, y = tile_x, tile_y

        # Fall back to parsing node name (X_Y_LAYER format)
        if x is None or y is None:
            node = getattr(src, 'node1', None)
            if node:
                x, y = parse_node_coordinates(node)

        if x is not None and y is not None:
            instance_coords[name] = (float(x), float(y))

    # Compute grid bounds
    if instance_coords:
        xs = [c[0] for c in instance_coords.values()]
        ys = [c[1] for c in instance_coords.values()]
        grid_bounds = (min(xs), max(xs), min(ys), max(ys))
    else:
        grid_bounds = (0.0, 1.0, 0.0, 1.0)

    return instance_coords, grid_bounds


# =============================================================================
# Window and Partitioning Functions
# =============================================================================

def compute_window_for_instance(
    center_x: float,
    center_y: float,
    grid_bounds: Tuple[float, float, float, float],
    window_percent: float,
) -> Tuple[float, float, float, float]:
    """Compute rectangular window centered at instance.

    Args:
        center_x: X coordinate of instance
        center_y: Y coordinate of instance
        grid_bounds: (x_min, x_max, y_min, y_max) of design
        window_percent: Window size as percentage of design dimensions

    Returns:
        (x_min, x_max, y_min, y_max) of window, clipped to grid bounds.
    """
    x_min_g, x_max_g, y_min_g, y_max_g = grid_bounds
    width = (x_max_g - x_min_g) * window_percent / 100.0
    height = (y_max_g - y_min_g) * window_percent / 100.0

    return (
        max(x_min_g, center_x - width / 2),
        min(x_max_g, center_x + width / 2),
        max(y_min_g, center_y - height / 2),
        min(y_max_g, center_y + height / 2),
    )


def windows_intersect(
    w1: Tuple[float, float, float, float],
    w2: Tuple[float, float, float, float],
) -> bool:
    """Check if two windows (x_min, x_max, y_min, y_max) overlap."""
    return not (w1[1] < w2[0] or w2[1] < w1[0] or   # x separation
                w1[3] < w2[2] or w2[3] < w1[2])     # y separation


def partition_sources_by_window(
    current_sources: Dict[str, Any],
    instance_coords: Dict[str, Tuple[float, float]],
    window_bounds: Tuple[float, float, float, float],
) -> Tuple[Set[str], Set[str]]:
    """Split sources into near (inside) and far (outside) window.

    Args:
        current_sources: Dict mapping source name to CurrentSource object
        instance_coords: Dict mapping source name to (x, y) tuple
        window_bounds: (x_min, x_max, y_min, y_max) of window

    Returns:
        Tuple of (near_names, far_names) sets.
    """
    x_min, x_max, y_min, y_max = window_bounds
    near: Set[str] = set()
    far: Set[str] = set()

    for name in current_sources:
        if name not in instance_coords:
            continue
        x, y = instance_coords[name]
        if x_min <= x <= x_max and y_min <= y <= y_max:
            near.add(name)
        else:
            far.add(name)

    return near, far


# =============================================================================
# Worst Instance Selection
# =============================================================================

def find_worst_instances_spatially_separated(
    peak_ir_drop_per_node: Dict[Any, float],
    current_sources: Dict[str, Any],
    instance_coords: Dict[str, Tuple[float, float]],
    grid_bounds: Tuple[float, float, float, float],
    top_k: int,
    window_percent: float,
) -> List[Tuple[str, str, float, float, float]]:
    """Find top-K worst instances with non-overlapping windows.

    Uses greedy selection: picks worst instance, then next worst whose
    window doesn't overlap with any already-selected window, etc.

    Args:
        peak_ir_drop_per_node: Dict mapping node -> peak IR-drop (V)
        current_sources: Dict mapping source name to CurrentSource object
        instance_coords: Dict mapping source name to (x, y) tuple
        grid_bounds: (x_min, x_max, y_min, y_max) of design
        top_k: Number of instances to select
        window_percent: Window size as percentage of design dimensions

    Returns:
        List of (instance_name, node, x, y, peak_ir_drop) tuples.
    """
    # Build mapping from node to instance
    node_to_instance: Dict[str, str] = {}
    for name, src in current_sources.items():
        node = getattr(src, 'node1', None)
        if node:
            node_to_instance[node] = name

    # Build candidates list: (instance, node, x, y, peak_ir_drop)
    candidates: List[Tuple[str, str, float, float, float]] = []
    for node, ir_drop in peak_ir_drop_per_node.items():
        inst = node_to_instance.get(node)
        if inst and inst in instance_coords:
            x, y = instance_coords[inst]
            candidates.append((inst, node, x, y, ir_drop))

    # Sort by IR-drop descending
    candidates.sort(key=lambda c: c[4], reverse=True)

    # Greedily select non-overlapping instances
    selected: List[Tuple[str, str, float, float, float]] = []
    selected_windows: List[Tuple[float, float, float, float]] = []

    for inst, node, x, y, ir_drop in candidates:
        if len(selected) >= top_k:
            break

        window = compute_window_for_instance(x, y, grid_bounds, window_percent)

        # Check for intersection with already-selected windows
        overlaps = any(windows_intersect(window, w) for w in selected_windows)

        if not overlaps:
            selected.append((inst, node, x, y, ir_drop))
            selected_windows.append(window)

    return selected


def resolve_instance_list(
    instances: List[str],
    current_sources: Dict[str, Any],
    instance_coords: Dict[str, Tuple[float, float]],
) -> List[Tuple[str, str, float, float, Optional[float]]]:
    """Resolve instance names or node names to (inst, node, x, y, ir_drop) tuples.

    Args:
        instances: List of instance names or node names
        current_sources: Dict mapping source name to CurrentSource object
        instance_coords: Dict mapping source name to (x, y) tuple

    Returns:
        List of (instance_name, node, x, y, None) tuples.
        IR-drop is None since we skip initial analysis.

    Raises:
        ValueError: If instance/node not found.
    """
    # Build node -> instance mapping
    node_to_instance: Dict[str, str] = {}
    for name, src in current_sources.items():
        node = getattr(src, 'node1', None)
        if node:
            node_to_instance[node] = name

    resolved: List[Tuple[str, str, float, float, Optional[float]]] = []

    for item in instances:
        item = item.strip()
        if not item:
            continue

        if item in current_sources:
            # It's an instance name
            src = current_sources[item]
            node = getattr(src, 'node1', '')
            x, y = instance_coords.get(item, (0.0, 0.0))
        elif item in node_to_instance:
            # It's a node name
            inst = node_to_instance[item]
            node = item
            x, y = instance_coords.get(inst, (0.0, 0.0))
        else:
            raise ValueError(f"Unknown instance or node: {item}")

        resolved.append((item if item in current_sources else inst, node, x, y, None))

    return resolved


# =============================================================================
# Statistics Computation
# =============================================================================

def compute_decomposition_stats(
    inst_name: str,
    node: str,
    x: float,
    y: float,
    window: Tuple[float, float, float, float],
    result_full: Any,
    result_near: Any,
    result_far: Any,
    n_near: int,
    n_far: int,
) -> InstanceDecomposition:
    """Compute decomposition statistics from transient results.

    Args:
        inst_name: Instance name
        node: Node name
        x, y: Instance coordinates
        window: Window bounds
        result_full: TransientResult with all sources
        result_near: TransientResult with near sources only
        result_far: TransientResult with far sources only
        n_near: Number of near sources
        n_far: Number of far sources

    Returns:
        InstanceDecomposition with computed statistics.
    """
    t_array = result_full.t_array

    # Get waveforms for the tracked node
    ir_total = result_full.get_ir_drop_waveform(node)
    ir_near = result_near.get_ir_drop_waveform(node)
    ir_far = result_far.get_ir_drop_waveform(node)

    # Peak statistics
    peak_total = np.max(ir_total)
    peak_near = np.max(ir_near)
    peak_far = np.max(ir_far)
    peak_idx = np.argmax(ir_total)
    peak_time = t_array[peak_idx]

    # Average statistics
    avg_total = np.mean(ir_total)
    avg_near = np.mean(ir_near)
    avg_far = np.mean(ir_far)

    # Fraction at peak
    total_at_peak = ir_total[peak_idx]
    if total_at_peak > 0:
        near_frac_peak = ir_near[peak_idx] / total_at_peak * 100
        far_frac_peak = ir_far[peak_idx] / total_at_peak * 100
    else:
        near_frac_peak = 0.0
        far_frac_peak = 0.0

    # Average fraction over time
    safe_total = np.where(ir_total > 1e-15, ir_total, 1e-15)
    near_frac_arr = ir_near / safe_total * 100
    far_frac_arr = ir_far / safe_total * 100
    avg_near_frac = np.mean(near_frac_arr)
    avg_far_frac = np.mean(far_frac_arr)

    return InstanceDecomposition(
        instance_name=inst_name,
        node=node,
        x=x,
        y=y,
        window_bounds=window,
        n_near_sources=n_near,
        n_far_sources=n_far,
        t_array=t_array,
        ir_drop_total=ir_total,
        ir_drop_near=ir_near,
        ir_drop_far=ir_far,
        peak_total_mV=peak_total * 1000,
        peak_near_mV=peak_near * 1000,
        peak_far_mV=peak_far * 1000,
        peak_time_ns=peak_time * 1e9,
        avg_total_mV=avg_total * 1000,
        avg_near_mV=avg_near * 1000,
        avg_far_mV=avg_far * 1000,
        near_fraction_at_peak=near_frac_peak,
        far_fraction_at_peak=far_frac_peak,
        avg_near_fraction=avg_near_frac,
        avg_far_fraction=avg_far_frac,
    )


# =============================================================================
# Output Formatting
# =============================================================================

class Logger:
    """Simple logger that writes to stdout and optionally to a file."""

    def __init__(self, log_file: Optional[str] = None):
        self.log_file = None
        if log_file:
            self.log_file = open(log_file, 'w')

    def log(self, msg: str = '') -> None:
        """Write message to stdout and log file."""
        print(msg)
        if self.log_file:
            self.log_file.write(msg + '\n')

    def close(self) -> None:
        """Close log file if open."""
        if self.log_file:
            self.log_file.close()
            self.log_file = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def print_results(result: DecompositionResult, logger: Optional[Logger] = None) -> None:
    """Print results to console and optionally to log file."""
    log = logger.log if logger else print

    log("=" * 80)
    log("DYNAMIC IR-DROP DECOMPOSITION ANALYSIS RESULTS")
    log("=" * 80)
    log(f"Netlist: {result.netlist_dir}")
    log(f"Net: {result.net_name}")
    log(f"Method: {result.method} ({result.integration_method.upper()})")
    log(f"Time range: {result.t_start_ns:.2f} ns - {result.t_end_ns:.2f} ns (dt={result.dt_ns:.2f} ns)")
    log(f"Window size: {result.window_percent:.1f}% of design")
    log(f"Grid bounds: ({result.grid_bounds[0]:.0f}, {result.grid_bounds[1]:.0f}) x "
        f"({result.grid_bounds[2]:.0f}, {result.grid_bounds[3]:.0f})")
    log()

    log(f"TOP-{len(result.worst_instances)} WORST INSTANCES WITH NEAR/FAR DECOMPOSITION")
    log("=" * 80)
    log(f"{'Rank':<5} {'Instance':<30} {'Peak(mV)':<10} {'Near(mV)':<10} {'Far(mV)':<10} {'Near%':<7} {'Far%':<7}")
    log("-" * 80)

    for i, inst in enumerate(result.worst_instances, 1):
        inst_short = inst.instance_name[:28] + ".." if len(inst.instance_name) > 30 else inst.instance_name
        log(f"{i:<5} {inst_short:<30} {inst.peak_total_mV:<10.3f} {inst.peak_near_mV:<10.3f} "
            f"{inst.peak_far_mV:<10.3f} {inst.near_fraction_at_peak:<7.1f} {inst.far_fraction_at_peak:<7.1f}")

    log()

    # Summary statistics
    if result.worst_instances:
        avg_near_pct = np.mean([inst.near_fraction_at_peak for inst in result.worst_instances])
        avg_far_pct = np.mean([inst.far_fraction_at_peak for inst in result.worst_instances])
        log(f"Average near contribution: {avg_near_pct:.1f}%")
        log(f"Average far contribution: {avg_far_pct:.1f}%")

    # Print aggressor analysis if available
    has_aggressors = any(inst.top_aggressors for inst in result.worst_instances)
    if has_aggressors:
        log()
        log("=" * 80)
        log("TOP AGGRESSORS WITHIN NEAR-WINDOW (per victim)")
        log("=" * 80)

        for i, inst in enumerate(result.worst_instances, 1):
            if not inst.top_aggressors:
                continue

            log()
            log(f"VICTIM #{i}: {inst.instance_name}")
            log(f"  Peak IR-drop: {inst.peak_total_mV:.3f} mV at t={inst.peak_time_ns:.2f} ns")
            log(f"  Self contribution: {inst.self_contribution_mV:.3f} mV ({inst.self_contribution_pct:.1f}%)")
            log(f"  Attribution efficiency: {inst.attribution_efficiency:.1%}")
            log()
            log(f"  {'Rank':<5} {'Node':<25} {'Contrib(mV)':<12} {'%':<8} {'Distance(um)':<12}")
            log(f"  {'-'*62}")

            for j, agg in enumerate(inst.top_aggressors, 1):
                node_short = agg.node[:23] + ".." if len(agg.node) > 25 else agg.node
                log(f"  {j:<5} {node_short:<25} {agg.contribution_mV:<12.3f} "
                    f"{agg.contribution_pct:<8.1f} {agg.distance_um:<12.1f}")

            # Summary lines after aggressor table
            top_k_total = inst.self_contribution_mV + sum(
                agg.contribution_mV for agg in inst.top_aggressors
            )
            top_k_pct = 100.0 * top_k_total / inst.peak_total_mV if inst.peak_total_mV > 0 else 0.0
            log(f"  Top-{len(inst.top_aggressors)} + self: {top_k_total:.3f} mV ({top_k_pct:.1f}%)")

            if inst.near_total_mV is not None:
                near_pct = 100.0 * inst.near_total_mV / inst.peak_total_mV if inst.peak_total_mV > 0 else 0.0
                log(f"  Near-window total: {inst.near_total_mV:.3f} mV ({near_pct:.1f}%)")

    log()
    log("Timing breakdown:")
    for key, val in result.timings.items():
        log(f"  {key}: {val:.3f} s")


# =============================================================================
# Plotting Functions
# =============================================================================

def truncate_name(name: str, max_len: int = 40) -> str:
    """Truncate name with '...' suffix if too long."""
    if len(name) > max_len:
        return name[:max_len - 3] + '...'
    return name


def generate_peak_ir_drop_plots(
    result: DecompositionResult | Mapping[str, Any],
    peak_ir_drop_per_node: Mapping[str, float],
    plot_dir: str,
    *,
    show: bool = False,
    heatmap_layers: Optional[List[str]] = None,
    max_stripes: int = 500,
    verbose: bool = False,
    graph: Optional[Any] = None,
) -> None:
    """Generate design-wide and per-victim peak IR-drop plots from live or saved data."""
    if not peak_ir_drop_per_node:
        return

    from visualization.stripe_heatmap import plot_stripe_heatmap
    import matplotlib.pyplot as plt

    overlay_plot_dir = os.path.join(plot_dir, 'peak_ir_drop_overlays')
    os.makedirs(plot_dir, exist_ok=True)
    _remove_stale_peak_ir_drop_outputs(plot_dir)
    orientation_overrides = None
    if graph is None:
        orientation_overrides = _load_peak_plot_orientation_overrides(
            result,
            peak_ir_drop_per_node,
            heatmap_layers=heatmap_layers,
            verbose=verbose,
        )

    overlay_data = prepare_decomposition_overlay_plot_data(result, max_aggressors=10)
    markers = [
        (overlay.victim_marker.x, overlay.victim_marker.y, overlay.victim_layer)
        for overlay in overlay_data
    ]
    windows = [
        (
            overlay.near_window_box.x_min,
            overlay.near_window_box.x_max,
            overlay.near_window_box.y_min,
            overlay.near_window_box.y_max,
            overlay.near_window_box.layer,
        )
        for overlay in overlay_data
    ]

    if verbose:
        print("Generating peak IR-drop heatmaps...")

    plot_stripe_heatmap(
        node_values=peak_ir_drop_per_node,
        layers=heatmap_layers,
        plot_dir=plot_dir,
        max_stripes=max_stripes,
        title_prefix='Peak IR-Drop',
        value_label='Peak IR-Drop (mV)',
        value_scale=1000.0,
        markers=markers,
        windows=windows,
        show=show,
        verbose=verbose,
        graph=graph,
        orientation_overrides=orientation_overrides,
    )

    if verbose:
        print("Generating per-victim peak IR-drop overlays...")

    def _warn_overlay_failure(
        overlay: VictimOverlayPlotData,
        exc: Exception,
    ) -> None:
        print(
            "Warning: failed to render peak IR-drop overlay for "
            f"{overlay.instance_name} ({overlay.victim_layer}): {exc}"
        )

    rendered_overlays = plot_decomposition_overlay_peak_ir_drop_heatmaps(
        peak_ir_drop_per_node,
        result,
        save_path=overlay_plot_dir,
        show=show,
        on_error=_warn_overlay_failure,
        graph=graph,
        orientation_overrides=orientation_overrides,
    )
    for _, fig, _ in rendered_overlays:
        plt.close(fig)


def _compute_waveform_zoom_window_ns(
    t_ns: np.ndarray,
    waveform_mV: np.ndarray,
    peak_time_ns: float,
) -> Tuple[float, float]:
    """Return a tighter x-axis window around the local peak activity."""
    t_arr = np.asarray(t_ns, dtype=float)
    waveform_arr = np.asarray(waveform_mV, dtype=float)
    default_half_window_ns = 0.075

    if t_arr.size == 0 or waveform_arr.size == 0:
        return -default_half_window_ns, default_half_window_ns
    if t_arr.size == 1 or waveform_arr.size == 1:
        t0 = float(t_arr[0])
        return t0 - default_half_window_ns, t0 + default_half_window_ns

    peak_idx = int(np.argmin(np.abs(t_arr - float(peak_time_ns))))
    diffs = np.diff(t_arr)
    positive_diffs = diffs[diffs > 0]
    dt_ns = float(np.min(positive_diffs)) if positive_diffs.size else 0.0
    min_window_ns = max(0.15, 6.0 * dt_ns)

    baseline_mV = float(np.percentile(waveform_arr, 5))
    peak_mV = float(waveform_arr[peak_idx])
    excursion_mV = peak_mV - baseline_mV

    if not np.isfinite(excursion_mV) or excursion_mV <= 0.0:
        half_window = min_window_ns / 2.0
        center = float(t_arr[peak_idx])
        return (
            max(float(t_arr[0]), center - half_window),
            min(float(t_arr[-1]), center + half_window),
        )

    active_threshold_mV = baseline_mV + 0.05 * excursion_mV
    active = waveform_arr >= active_threshold_mV
    active[peak_idx] = True

    left = peak_idx
    while left > 0 and active[left - 1]:
        left -= 1

    right = peak_idx
    while right < waveform_arr.size - 1 and active[right + 1]:
        right += 1

    active_span_ns = float(t_arr[right] - t_arr[left])
    pad_ns = max(active_span_ns, 4.0 * dt_ns, min_window_ns / 4.0)

    t_min = max(float(t_arr[0]), float(t_arr[left] - pad_ns))
    t_max = min(float(t_arr[-1]), float(t_arr[right] + pad_ns))

    if t_max - t_min < min_window_ns:
        half_window = min_window_ns / 2.0
        center = float(t_arr[peak_idx])
        t_min = max(float(t_arr[0]), center - half_window)
        t_max = min(float(t_arr[-1]), center + half_window)

    return t_min, t_max


def generate_plots(
    result: DecompositionResult,
    plot_dir: str,
    show: bool = False,
    heatmap_layers: Optional[List[str]] = None,
    max_stripes: int = 500,
    verbose: bool = False,
    graph: Optional[Any] = None,
) -> None:
    """Generate analysis plots.

    Creates:
    1. Design-wide total current plot
    2. Waveform plots: Time-domain decomposition with total current for each worst instance
    3. Aggressor contribution bar plots (per victim)
    4. Peak IR-drop stripe heatmaps (if peak_ir_drop_per_node available)
    5. Peak IR-drop overlay heatmaps (per victim, if peak_ir_drop_per_node available)

    Args:
        result: DecompositionResult with analysis data
        plot_dir: Directory to save plots
        show: If True, display plots interactively
        heatmap_layers: Optional list of layers to generate heatmaps for (None = all)
        max_stripes: Maximum stripes for heatmap (default 500)
        verbose: Print progress
    """
    try:
        import matplotlib
        if not show:
            matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not available, skipping plots")
        return

    os.makedirs(plot_dir, exist_ok=True)

    # Clean stale plots from previous runs so leftover files don't
    # confuse users.
    for pattern in ('waveform_*.png',):
        for old_file in glob.glob(os.path.join(plot_dir, pattern)):
            try:
                os.remove(old_file)
            except OSError:
                pass

    # 1. Design-wide total current plot
    if len(result.total_current_waveform) > 0 and len(result.t_array) > 0:
        fig, ax = plt.subplots(figsize=(10, 5))

        t_ns = result.t_array * 1e9
        current_mA = result.total_current_waveform

        ax.plot(t_ns, current_mA, 'b-', linewidth=1.5)

        # Mark peak current time
        peak_idx = np.argmax(current_mA)
        peak_time = t_ns[peak_idx]
        peak_current = current_mA[peak_idx]
        ax.axvline(peak_time, color='red', linestyle='--', alpha=0.7,
                   label=f'Peak: {peak_current:.2f} mA @ {peak_time:.2f} ns')
        ax.plot(peak_time, peak_current, 'ro', markersize=8)

        ax.set_xlabel('Time (ns)')
        ax.set_ylabel('Total Current (mA)')
        ax.set_title(f'Design-Wide Total Current ({result.net_name})')
        ax.legend(loc='upper right')
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'total_current.png'), dpi=150)
        if show:
            plt.show()
        plt.close()

    # 2. Waveform plots for each instance (with total current on secondary y-axis)
    for i, inst in enumerate(result.worst_instances):
        fig, ax = plt.subplots(figsize=(10, 5))

        t_ns = inst.t_array * 1e9
        t_min, t_max = _compute_waveform_zoom_window_ns(
            t_ns,
            inst.ir_drop_total * 1000,
            inst.peak_time_ns,
        )

        # Left y-axis: IR-drop waveforms
        ax.plot(t_ns, inst.ir_drop_total * 1000, 'k-', linewidth=1.2, label='Total IR-drop')
        ax.plot(t_ns, inst.ir_drop_near * 1000, 'b-', linewidth=0.8, label='Near')
        ax.plot(t_ns, inst.ir_drop_far * 1000, 'r-', linewidth=0.8, label='Far')
        ax.axvline(inst.peak_time_ns, color='gray', linestyle='--', alpha=0.5,
                   label=f'Peak @ {inst.peak_time_ns:.2f} ns')

        ax.set_xlabel('Time (ns)')
        ax.set_ylabel('IR-Drop (mV)')
        ax.set_title(f'IR-Drop Decomposition: {truncate_name(inst.instance_name)}\n'
                     f'Peak: {inst.peak_total_mV:.3f} mV = {inst.peak_near_mV:.3f} (near) + {inst.peak_far_mV:.3f} (far)')
        ax.grid(alpha=0.3)
        ax.set_xlim(t_min, t_max)

        # Right y-axis: Total current (from design-wide waveform)
        if len(result.total_current_waveform) > 0 and len(result.t_array) == len(inst.t_array):
            ax2 = ax.twinx()
            ax2.plot(t_ns, result.total_current_waveform, 'g--', linewidth=0.8, alpha=0.7, label='Total Current')
            ax2.set_ylabel('Total Current (mA)', color='green')
            ax2.tick_params(axis='y', labelcolor='green')

            # Combine legends from both axes
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8)
        else:
            ax.legend(loc='upper right')

        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f'waveform_{i+1}.png'), dpi=300)
        if show:
            plt.show()
        plt.close()

    # 3. Aggressor contribution bar plots (per victim)
    generate_aggressor_contribution_plots(
        result,
        plot_dir,
        show=show,
        max_aggressors=10,
        verbose=verbose,
    )

    generate_peak_ir_drop_plots(
        result,
        result.peak_ir_drop_per_node,
        plot_dir,
        show=show,
        heatmap_layers=heatmap_layers,
        max_stripes=max_stripes,
        verbose=verbose,
        graph=graph,
    )

    print(f"Plots saved to: {plot_dir}")


def postprocess_saved_decomposition_outputs(
    results_json_path: str | Path,
    *,
    peak_ir_drop_data_path: Optional[str | Path] = None,
    plot_dir: Optional[str | Path] = None,
    heatmap_layers: Optional[List[str]] = None,
    max_stripes: int = 500,
    show: bool = False,
    verbose: bool = False,
    graph: Optional[Any] = None,
) -> str:
    """Regenerate plots that can be derived from saved decomposition outputs."""
    results_path = Path(results_json_path)
    saved_result = load_saved_decomposition_result(results_path)
    target_plot_dir = Path(plot_dir) if plot_dir is not None else results_path.parent / 'plots'
    explicit_peak_ir_drop_data_path = peak_ir_drop_data_path is not None

    generate_aggressor_contribution_plots(
        saved_result,
        target_plot_dir,
        show=show,
        max_aggressors=10,
        verbose=verbose,
    )

    try:
        peak_data_path = resolve_peak_ir_drop_data_path(
            results_path,
            saved_result,
            peak_ir_drop_data_path=peak_ir_drop_data_path,
        )
    except FileNotFoundError:
        if explicit_peak_ir_drop_data_path:
            raise
        recovered_plot_dir = recover_legacy_peak_ir_drop_plot_artifacts(
            results_path,
            plot_dir=plot_dir,
            verbose=verbose,
        )
        if recovered_plot_dir is not None:
            return recovered_plot_dir
        raise

    peak_ir_drop_per_node = load_peak_ir_drop_data(peak_data_path)

    if verbose:
        print(f"Post-processing saved decomposition output: {results_path}")
        print(f"Loading persisted peak IR-drop data: {peak_data_path}")
        print(f"Writing plots to: {target_plot_dir}")

    generate_peak_ir_drop_plots(
        saved_result,
        peak_ir_drop_per_node,
        str(target_plot_dir),
        show=show,
        heatmap_layers=heatmap_layers,
        max_stripes=max_stripes,
        verbose=verbose,
        graph=graph,
    )
    return str(target_plot_dir)


# =============================================================================
# Graph Loading (with pkl cache support)
# =============================================================================

def load_pdn_graph(
    netlist_dir: str,
    net: Optional[str] = None,
    use_cache: bool = True,
    verbose: bool = False,
) -> Tuple[Any, float]:
    """Load PDN graph from pickle cache or parse from netlist.

    Automatically detects and converts NetworkX graphs to Rustworkx format
    for compatibility with the current solver infrastructure.

    Args:
        netlist_dir: Path to PDN netlist directory
        net: Optional net name to filter during parsing (ignored when loading from cache)
        use_cache: If True, load from pdn_graph.pkl if available
        verbose: Print progress

    Returns:
        Tuple of (graph, load_time_seconds)
    """
    from pathlib import Path
    from parser.netlist import NetlistParser, load_pdn_pickle
    from graph.converter import detect_graph_type, ensure_rustworkx_graph

    netlist_path = Path(netlist_dir)
    pkl_path = netlist_path / 'pdn_graph.pkl'

    t0 = time_module.perf_counter()

    if use_cache and pkl_path.exists():
        if verbose:
            print(f"Loading cached graph from {pkl_path}...")
        graph = load_pdn_pickle(str(pkl_path))

        # Detect graph type and convert if needed
        graph_type = detect_graph_type(graph)
        if verbose:
            print(f"  Graph type: {graph_type}")

        if graph_type == 'networkx':
            if verbose:
                print("  Converting NetworkX graph to Rustworkx format...")
            graph = ensure_rustworkx_graph(graph, verbose=False)
            if verbose:
                print("  Conversion complete!")

        if verbose:
            print(f"  Nodes: {graph.number_of_nodes()}, Edges: {graph.number_of_edges()}")
    else:
        if verbose:
            if use_cache:
                print(f"No cached graph found, parsing netlist: {netlist_dir}")
            else:
                print(f"Parsing netlist: {netlist_dir}")
            if net:
                print(f"  Filtering for net: {net}")
        parser = NetlistParser(netlist_dir, net_filter=net)
        graph = parser.parse()

    load_time = time_module.perf_counter() - t0
    return graph, load_time


# =============================================================================
# Main Analysis Function
# =============================================================================

def analyze_dynamic_irdrop_decomposition(
    netlist_dir: str,
    net: str = 'VDD',
    t_start: float = 0.0,
    t_end: float = 100e-9,
    dt: float = 0.1e-9,
    top_k: int = 5,
    window_percent: float = 10.0,
    integration_method: str = 'trap',
    instances: Optional[List[str]] = None,
    use_cache: bool = True,
    verbose: bool = False,
    aggressor_top_k: int = 0,
    adjoint_method: str = 'dynamic',
    adjoint_memory_window: int = 20,
    smooth_sources: bool = True,
) -> Tuple[DecompositionResult, Any]:
    """Analyze dynamic IR-drop and decompose into near/far contributions.

    Args:
        netlist_dir: Path to PDN netlist directory
        net: Power net name (default 'VDD')
        t_start: Start time in seconds
        t_end: End time in seconds
        dt: Time step in seconds
        top_k: Number of worst instances to analyze
        window_percent: Window size as percentage of design dimensions
        integration_method: 'trap' (Trapezoidal) or 'be' (Backward Euler)
        instances: Optional list of instance/node names to analyze (skips initial transient)
        use_cache: If True, load from pdn_graph.pkl if available (default True)
        verbose: Print progress
        aggressor_top_k: Number of top aggressors to identify per victim (0 = disabled)
        adjoint_method: 'dynamic' (default) or 'static' for adjoint analysis
        adjoint_memory_window: Number of time steps for dynamic adjoint memory
        smooth_sources: If True (default), apply PWL smoothing to current source waveforms

    Returns:
        Tuple of (DecompositionResult, graph) where graph is the PDN graph for plotting.
    """
    # Force Backward Euler when aggressor analysis is enabled
    # (dynamic adjoint uses BE internally for consistency)
    if aggressor_top_k > 0 and integration_method != 'be':
        if verbose:
            print("Note: Forcing Backward Euler integration for adjoint consistency")
        integration_method = 'be'
    timings: Dict[str, float] = {}
    peak_ir_drop_per_node: Dict[str, float] = {}  # Will be populated from initial transient
    t0_total = time_module.perf_counter()

    # Import core modules
    from model.factory import create_model_from_pdn
    from analysis.transient_solver import TransientIRDropSolver, IntegrationMethod
    from analysis.adjoint_sensitivity import AdjointSensitivitySolver

    # Load graph (from cache or parse)
    graph, load_time = load_pdn_graph(netlist_dir, net=net, use_cache=use_cache, verbose=verbose)
    timings['parse'] = load_time

    # Create model
    t0_model = time_module.perf_counter()
    model = create_model_from_pdn(graph, net)
    timings['model'] = time_module.perf_counter() - t0_model

    # Get current sources
    graph_dict = None
    if hasattr(graph, 'graph') and isinstance(graph.graph, dict):
        graph_dict = graph.graph
    elif hasattr(graph, '_attrs'):
        graph_dict = graph._attrs

    if graph_dict is None:
        raise RuntimeError("Cannot access graph metadata")

    # Try raw objects first, then fall back to serialized
    current_sources = graph_dict.get('_instance_sources_objects', {})
    if not current_sources:
        # Need to reconstruct from serialized
        from parser.current_sources import CurrentSource
        serialized = graph_dict.get('instance_sources', {})
        current_sources = {k: CurrentSource.from_dict(v) for k, v in serialized.items()}

    if verbose:
        print(f"Found {len(current_sources)} current sources")

    # Extract locations
    t0_loc = time_module.perf_counter()
    instance_coords, grid_bounds = extract_instance_locations(current_sources)
    timings['extract_locations'] = time_module.perf_counter() - t0_loc

    if verbose:
        print(f"Grid bounds: {grid_bounds}")

    # Create solver (use vectorize_threshold=0 to enable solve_transient_multi_rhs).
    # When aggressor analysis is enabled, do not clear graph metadata so the adjoint
    # solver can read _instance_sources_objects when created after the decomposition loop.
    solver = TransientIRDropSolver(
        model, graph, vectorize_threshold=0,
        clear_graph_metadata=(aggressor_top_k == 0),
    )

    # Preprocess current sources for smoothing (reusable across all solves)
    smoothed_sources = None
    if smooth_sources:
        if verbose:
            print("Preprocessing current sources with waveform smoothing...")
        t0_smooth = time_module.perf_counter()
        smoothed_sources = solver.preprocess_sources(
            dt=dt,
            t_start=t_start,
            t_end=t_end,
            compact_threshold=1e-12,
        )
        timings['preprocess_smoothing'] = time_module.perf_counter() - t0_smooth

    # Build source name list and index mapping
    source_names = list(current_sources.keys())
    name_to_idx = {name: i for i, name in enumerate(source_names)}
    n_sources = len(source_names)

    # Variables for total current waveform (populated from initial transient or first decomposition)
    total_current_waveform: np.ndarray = np.array([])
    result_t_array: np.ndarray = np.array([])

    # Determine worst instances
    if instances:
        # User provided list - skip initial analysis
        if verbose:
            print(f"Using {len(instances)} pre-defined instances")
        worst_instances_raw = resolve_instance_list(instances, current_sources, instance_coords)

        # Warn about overlapping windows
        selected_windows = []
        for inst, node, x, y, _ in worst_instances_raw:
            window = compute_window_for_instance(x, y, grid_bounds, window_percent)
            overlaps = [windows_intersect(window, w) for w in selected_windows]
            if any(overlaps):
                print(f"Warning: Window for {inst} overlaps with previously selected instances")
            selected_windows.append(window)

        worst_instances = worst_instances_raw

    else:
        # Run initial transient to find worst instances
        if verbose:
            print("Running initial transient analysis to find worst instances...")

        t0_initial = time_module.perf_counter()

        # Create mask for all sources (single mask)
        mask_all = np.ones(n_sources, dtype=bool)

        # Get all source nodes for tracking (limit to avoid memory issues)
        all_source_nodes = list(set(
            getattr(current_sources[name], 'node1', None)
            for name in source_names
            if getattr(current_sources[name], 'node1', None) is not None
        ))

        # Limit tracked nodes to reasonable number
        max_track = min(2000, len(all_source_nodes))
        track_nodes_initial = all_source_nodes[:max_track]

        method_enum = IntegrationMethod.TRAPEZOIDAL if integration_method == 'trap' else IntegrationMethod.BACKWARD_EULER

        # Run single-mask solve to get peak IR-drop
        results = solver.solve_transient_multi_rhs(
            t_start=t_start,
            t_end=t_end,
            dt=dt,
            source_masks=mask_all[np.newaxis, :],  # Shape: (1, n_sources)
            method=method_enum,
            track_nodes=track_nodes_initial,
            verbose=verbose,
            smoothed_sources=smoothed_sources,
        )
        initial_result = results[0]

        timings['initial_transient'] = time_module.perf_counter() - t0_initial

        # Find worst instances with spatial separation
        worst_instances = find_worst_instances_spatially_separated(
            initial_result.peak_ir_drop_per_node,
            current_sources,
            instance_coords,
            grid_bounds,
            top_k,
            window_percent,
        )

        if verbose:
            print(f"Found {len(worst_instances)} spatially-separated worst instances")

        # Preserve peak IR-drop per node for heatmap generation
        peak_ir_drop_per_node = dict(initial_result.peak_ir_drop_per_node)

        # Capture total current waveform from initial transient
        total_current_waveform = np.array(initial_result.total_current_per_time)
        result_t_array = initial_result.t_array.copy()

        # Free initial transient result - no longer needed (~80 MB)
        del initial_result, results

    # Decomposition analysis for each worst instance (merged with adjoint analysis
    # so result_full can flow directly into analyze_victim(transient_result=...) and
    # avoid an O(observation_time/dt) per-victim forward sweep).
    decompositions: List[InstanceDecomposition] = []
    cum_transient_s = 0.0
    cum_adjoint_s = 0.0

    method_enum = IntegrationMethod.TRAPEZOIDAL if integration_method == 'trap' else IntegrationMethod.BACKWARD_EULER

    # Construct adjoint solver/context once before the loop, gated on aggressor_top_k > 0
    # so we don't pay setup cost when adjoint analysis is disabled.
    adjoint_solver = None
    adjoint_ctx = None
    if aggressor_top_k > 0:
        if verbose:
            print()
            print("Running adjoint sensitivity analysis for top aggressors...")

        # Create adjoint solver from transient solver (shares RC system)
        adjoint_solver = AdjointSensitivitySolver.from_transient_solver(solver)

        # Prepare adjoint context for batch solving (reuses LU factorization).
        # The integration method MUST match solve_transient_multi_rhs above so
        # that the BE/TR forward operator and its dual share the same A, B
        # matrices -- otherwise contribution sums are dual to a different
        # operator than result_full was integrated under, breaking
        # attribution_efficiency for trap mode.
        adjoint_ctx = adjoint_solver.prepare(dt=dt, method=method_enum)

    for rank, (inst_name, node, x, y, _) in enumerate(worst_instances, 1):
        t_iter_start = time_module.perf_counter()
        if verbose:
            print(f"Analyzing instance {rank}/{len(worst_instances)}: {inst_name}")

        window = compute_window_for_instance(x, y, grid_bounds, window_percent)
        near_names, far_names = partition_sources_by_window(
            current_sources, instance_coords, window
        )

        if verbose:
            print(f"  Near sources: {len(near_names)}, Far sources: {len(far_names)}")

        # Build boolean masks
        mask_all = np.ones(n_sources, dtype=bool)
        mask_near = np.array([name in near_names for name in source_names], dtype=bool)
        mask_far = np.array([name in far_names for name in source_names], dtype=bool)

        source_masks = np.stack([mask_all, mask_near, mask_far])  # (3, n_sources)

        # Run multi-RHS transient
        results = solver.solve_transient_multi_rhs(
            t_start=t_start,
            t_end=t_end,
            dt=dt,
            source_masks=source_masks,
            method=method_enum,
            track_nodes=[node],
            verbose=verbose,
            smoothed_sources=smoothed_sources,
        )
        result_full, result_near, result_far = results

        # Compute statistics
        decomp = compute_decomposition_stats(
            inst_name, node, x, y, window,
            result_full, result_near, result_far,
            len(near_names), len(far_names),
        )
        decompositions.append(decomp)

        # Capture total current from first decomposition if not already set
        # (handles case where instances are pre-defined and initial transient is skipped)
        if len(total_current_waveform) == 0:
            total_current_waveform = np.array(result_full.total_current_per_time)
            result_t_array = result_full.t_array.copy()

        cum_transient_s += time_module.perf_counter() - t_iter_start

        # Adjoint aggressor analysis for this victim (if enabled). Runs while
        # result_full is still alive so we can pass it as transient_result and
        # skip the per-victim forward sweep inside analyze_victim.
        if aggressor_top_k > 0:
            t_adj_start = time_module.perf_counter()
            if verbose:
                print(f"  Analyzing aggressors for victim {rank}/{len(worst_instances)}: {decomp.node}")

            # Get peak time in seconds
            peak_time_s = decomp.peak_time_ns * 1e-9

            # Run adjoint analysis within the near-window
            try:
                attribution = adjoint_solver.analyze_victim(
                    victim_node=decomp.node,
                    observation_time=peak_time_s,
                    memory_window=adjoint_memory_window,
                    dt=dt,
                    top_k=aggressor_top_k,
                    spatial_window=decomp.window_bounds,
                    use_static=(adjoint_method == 'static'),
                    context=adjoint_ctx,
                    method=method_enum,
                    transient_result=result_full,
                    smoothed_sources=smoothed_sources,
                )

                # Compute distances and build AggressorResult list
                aggressor_results: List[AggressorResult] = []
                for agg in attribution.top_aggressors:
                    agg_x, agg_y = parse_node_coordinates(agg.node)
                    if agg_x is not None and agg_y is not None:
                        distance = math.sqrt((agg_x - decomp.x)**2 + (agg_y - decomp.y)**2)
                    else:
                        distance = 0.0

                    aggressor_results.append(AggressorResult(
                        node=agg.node,
                        contribution_mV=agg.contribution_mV,
                        contribution_pct=agg.contribution_pct,
                        source_names=agg.source_names,
                        distance_um=distance,
                    ))

                # Update decomposition with aggressor data
                decomp.top_aggressors = aggressor_results
                decomp.self_contribution_mV = attribution.self_contribution_mV
                decomp.self_contribution_pct = attribution.self_contribution_pct
                decomp.attribution_efficiency = attribution.attribution_efficiency

            except Exception as e:
                if verbose:
                    print(f"    Warning: Adjoint analysis failed for {decomp.node}: {e}")

            cum_adjoint_s += time_module.perf_counter() - t_adj_start

        # Free TransientResult objects - waveforms already extracted (~80 MB per instance).
        # Released only after adjoint is done with result_full.
        del results, result_full, result_near, result_far

    timings['decomposition'] = cum_transient_s
    if aggressor_top_k > 0:
        timings['adjoint_analysis'] = cum_adjoint_s
    timings['total'] = time_module.perf_counter() - t0_total

    return (
        DecompositionResult(
            netlist_dir=netlist_dir,
            net_name=net,
            method='transient',
            integration_method=integration_method,
            t_start_ns=t_start * 1e9,
            t_end_ns=t_end * 1e9,
            dt_ns=dt * 1e9,
            window_percent=window_percent,
            grid_bounds=grid_bounds,
            worst_instances=decompositions,
            timings=timings,
            peak_ir_drop_per_node=peak_ir_drop_per_node,
            total_current_waveform=total_current_waveform,
            t_array=result_t_array,
            smooth_sources=smooth_sources,
        ),
        graph,
    )


# =============================================================================
# Config File Loading
# =============================================================================

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Dict with configuration parameters.

    Raises:
        ImportError: If PyYAML not installed.
        FileNotFoundError: If config file doesn't exist.
    """
    if not HAS_YAML:
        raise ImportError(
            "PyYAML is required for config file support. Install with: pip install pyyaml"
        )

    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def merge_config_with_args(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Merge config file values with CLI arguments.

    CLI arguments take precedence over config file values.

    Args:
        config: Dict from config file
        args: Parsed CLI arguments

    Returns:
        Dict with merged configuration.
    """
    result = {}

    # Basic parameters
    result['netlist_dir'] = args.netlist_dir or config.get('netlist_dir')
    result['net'] = args.net or config.get('net', 'VDD')

    # Time parameters (from config.time or CLI)
    time_config = config.get('time', {})
    result['t_start'] = parse_time_value(args.start_time) if args.start_time else parse_time_value(time_config.get('start', '0ns'))
    result['t_end'] = parse_time_value(args.end_time) if args.end_time else parse_time_value(time_config.get('end', '100ns'))
    result['dt'] = parse_time_value(args.dt) if args.dt else parse_time_value(time_config.get('dt', '100ps'))

    # Analysis parameters
    analysis_config = config.get('analysis', {})
    result['top_k'] = args.top_k if args.top_k is not None else analysis_config.get('top_k', 5)
    result['window_percent'] = args.window_percent if args.window_percent is not None else analysis_config.get('window_percent', 10.0)
    result['integration_method'] = args.integration or analysis_config.get('integration', 'trap')

    # Aggressor analysis parameters
    result['aggressor_top_k'] = args.aggressor_top_k if args.aggressor_top_k != 0 else analysis_config.get('aggressor_top_k', 0)
    result['adjoint_method'] = args.adjoint_method or analysis_config.get('adjoint_method', 'dynamic')
    result['adjoint_memory_window'] = args.adjoint_memory_window if args.adjoint_memory_window != 20 else analysis_config.get('adjoint_memory_window', 20)

    # Waveform smoothing (CLI takes precedence, then config, default True)
    # args.smooth_sources is None when neither --smooth nor --no-smooth is specified
    if args.smooth_sources is not None:
        result['smooth_sources'] = args.smooth_sources
    else:
        result['smooth_sources'] = analysis_config.get('smooth_sources', True)

    # Instances
    if args.instances:
        result['instances'] = [s.strip() for s in args.instances.split(',')]
    elif args.instances_file:
        with open(args.instances_file, 'r') as f:
            result['instances'] = [line.strip() for line in f if line.strip()]
    elif analysis_config.get('instances'):
        result['instances'] = analysis_config['instances']
    elif analysis_config.get('instances_file'):
        with open(analysis_config['instances_file'], 'r') as f:
            result['instances'] = [line.strip() for line in f if line.strip()]
    else:
        result['instances'] = None

    # Solver configuration
    solver_config = config.get('solver', {})
    result['solver'] = {
        'backend': args.backend or solver_config.get('backend', 'auto'),
        'ordering': args.cholmod_ordering or solver_config.get('ordering'),
        'mode': args.cholmod_mode or solver_config.get('mode'),
        'use_long': args.cholmod_use_long if args.cholmod_use_long is not None else solver_config.get('use_long'),
    }

    # Cache configuration (--no-cache CLI flag overrides config)
    # Default is True (use cache), --no-cache sets it to False
    if args.no_cache:
        result['use_cache'] = False
    else:
        result['use_cache'] = config.get('use_cache', True)

    # Output configuration
    output_config = config.get('output', {})
    result['output_dir'] = args.output_dir or output_config.get('output_dir', './irdrop_decomp_results')
    result['no_plot'] = args.no_plot if args.no_plot else output_config.get('no_plot', False)
    result['verbose'] = args.verbose or output_config.get('verbose', False)

    # Heatmap layers
    if args.heatmap_layers:
        result['heatmap_layers'] = [s.strip() for s in args.heatmap_layers.split(',')]
    else:
        result['heatmap_layers'] = output_config.get('heatmap_layers')

    # Max stripes for heatmap
    result['max_stripes'] = args.max_stripes if args.max_stripes != 500 else output_config.get('max_stripes', 500)

    return result


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """Command-line interface for dynamic IR-drop decomposition analysis."""
    parser = argparse.ArgumentParser(
        description='Analyze dynamic IR-drop and decompose into near/far contributions.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with time units
  python -m analysis.dynamic_irdrop_decomposition ./netlist/netlist_test --end-time 100ns --dt 100ps

  # With config file
  python -m analysis.dynamic_irdrop_decomposition --config config.yaml

  # Override config with CLI
  python -m analysis.dynamic_irdrop_decomposition --config config.yaml --top-k 10

  # Pre-defined instances
  python -m analysis.dynamic_irdrop_decomposition ./netlist/netlist_test --instances "inst1,inst2,inst3"

  # Re-generate saved-result peak IR-drop plots only
  python -m analysis.dynamic_irdrop_decomposition --saved-results ./results/ir_drop_decomp/results.json
        """
    )

    # Positional argument
    parser.add_argument('netlist_dir', nargs='?', help='Path to PDN netlist directory')

    # Config file
    parser.add_argument('--config', type=str, help='Path to YAML config file')
    parser.add_argument(
        '--saved-results',
        type=str,
        help='Path to saved results.json for post-processing peak IR-drop plots only',
    )
    parser.add_argument(
        '--peak-ir-drop-data',
        type=str,
        help='Optional path to persisted peak IR-drop data sidecar for --saved-results',
    )

    # Basic parameters
    parser.add_argument('--net', type=str, help='Power net name (default: VDD)')
    parser.add_argument('--start-time', type=str, help='Start time (e.g., 0ns)')
    parser.add_argument('--end-time', type=str, help='End time (e.g., 100ns)')
    parser.add_argument('--dt', type=str, help='Time step (e.g., 100ps)')

    # Analysis parameters
    parser.add_argument('--top-k', type=int, help='Number of worst instances to analyze')
    parser.add_argument('--window-percent', type=float, help='Window size as %% of design')
    parser.add_argument('--integration', choices=['trap', 'be'], help='Integration method')

    # Aggressor analysis parameters
    parser.add_argument('--aggressor-top-k', type=int, default=0,
                        help='Number of top aggressors per victim (0=disabled, default)')
    parser.add_argument('--adjoint-method', choices=['dynamic', 'static'], default='dynamic',
                        help='Adjoint analysis method (default: dynamic)')
    parser.add_argument('--adjoint-memory-window', type=int, default=20,
                        help='Memory window for dynamic adjoint (default: 20 time steps)')

    # Waveform smoothing (default=None to allow config file to take effect)
    parser.add_argument('--smooth', dest='smooth_sources', action='store_true',
                        default=None,
                        help='Enable current source waveform smoothing (default: enabled)')
    parser.add_argument('--no-smooth', dest='smooth_sources', action='store_false',
                        help='Disable current source waveform smoothing')

    # Pre-defined instances
    parser.add_argument('--instances', type=str, help='Comma-separated list of instance/node names')
    parser.add_argument('--instances-file', type=str, help='File with instance/node names (one per line)')

    # Cache control
    parser.add_argument('--no-cache', action='store_true',
                        help='Force re-parsing netlist (ignore pdn_graph.pkl cache)')

    # Solver backend
    parser.add_argument('--backend', choices=['auto', 'splu', 'cholmod'], help='Solver backend')
    parser.add_argument('--cholmod-ordering', type=str, help='CHOLMOD ordering method')
    parser.add_argument('--cholmod-mode', type=str, help='CHOLMOD factorization mode')
    parser.add_argument('--cholmod-use-long', action='store_true', default=None,
                        help='Use 64-bit indices for CHOLMOD')

    # Output
    parser.add_argument('--output-dir', '-o', type=str,
                        help='Output directory for results (default: ./irdrop_decomp_results)')
    parser.add_argument('--no-plot', action='store_true', help='Skip plot generation')
    parser.add_argument('--heatmap-layers', type=str,
                        help='Comma-separated list of layers for heatmaps (e.g., "M1,M2")')
    parser.add_argument('--max-stripes', type=int, default=500,
                        help='Maximum stripes for heatmap (default: 500)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')

    args = parser.parse_args()

    if args.saved_results:
        if args.netlist_dir:
            parser.error("netlist_dir cannot be used with --saved-results")
        if args.config:
            parser.error("--config is not supported with --saved-results; pass plotting flags directly")
        if args.no_plot:
            parser.error("--no-plot cannot be used with --saved-results")
        heatmap_layers = (
            [s.strip() for s in args.heatmap_layers.split(',')]
            if args.heatmap_layers else None
        )
        return postprocess_saved_decomposition_outputs(
            args.saved_results,
            peak_ir_drop_data_path=args.peak_ir_drop_data,
            plot_dir=args.output_dir,
            heatmap_layers=heatmap_layers,
            max_stripes=args.max_stripes,
            show=False,
            verbose=args.verbose,
        )

    # Load config if provided
    config = {}
    if args.config:
        config = load_config(args.config)

    # Merge config with CLI args
    merged = merge_config_with_args(config, args)

    # Validate required parameters
    if not merged['netlist_dir']:
        parser.error("netlist_dir is required (via CLI or config file)")

    # Configure solver backend
    configure_solver_backend(merged['solver'])

    # Create output directory
    output_dir = merged['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    # Define output paths
    log_file = os.path.join(output_dir, 'analysis.log')
    json_file = os.path.join(output_dir, 'results.json')

    # Create logger
    logger = Logger(log_file)

    try:
        # Run analysis
        if merged['verbose']:
            logger.log("Starting dynamic IR-drop decomposition analysis...")
            logger.log(f"  Netlist: {merged['netlist_dir']}")
            logger.log(f"  Net: {merged['net']}")
            logger.log(f"  Time: {merged['t_start']*1e9:.2f} ns to {merged['t_end']*1e9:.2f} ns, dt={merged['dt']*1e9:.3f} ns")
            logger.log(f"  Window: {merged['window_percent']}%")
            logger.log(f"  Smoothing: {'enabled' if merged['smooth_sources'] else 'disabled'}")
            if merged['aggressor_top_k'] > 0:
                logger.log(f"  Aggressor analysis: top-{merged['aggressor_top_k']} per victim ({merged['adjoint_method']} method)")
            logger.log(f"  Output dir: {output_dir}")
            logger.log()

        result, graph = analyze_dynamic_irdrop_decomposition(
            netlist_dir=merged['netlist_dir'],
            net=merged['net'],
            t_start=merged['t_start'],
            t_end=merged['t_end'],
            dt=merged['dt'],
            top_k=merged['top_k'],
            window_percent=merged['window_percent'],
            integration_method=merged['integration_method'],
            instances=merged['instances'],
            use_cache=merged['use_cache'],
            verbose=merged['verbose'],
            aggressor_top_k=merged['aggressor_top_k'],
            adjoint_method=merged['adjoint_method'],
            adjoint_memory_window=merged['adjoint_memory_window'],
            smooth_sources=merged['smooth_sources'],
        )

        # Print results
        print_results(result, logger)

        # Save JSON
        result.save_json(json_file)
        logger.log(f"\nResults saved to: {json_file}")

        # Generate plots
        if not merged['no_plot']:
            plot_dir = os.path.join(output_dir, 'plots')
            generate_plots(
                result,
                plot_dir,
                show=False,
                heatmap_layers=merged.get('heatmap_layers'),
                max_stripes=merged.get('max_stripes', 500),
                verbose=merged['verbose'],
                graph=graph,
            )

        logger.log(f"Log saved to: {log_file}")

        return result

    finally:
        logger.close()


if __name__ == '__main__':
    main()

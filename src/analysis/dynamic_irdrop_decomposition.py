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
import json
import math
import os
import re
import sys
import time as time_module
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

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
            # Note: total_current_waveform and t_array are NOT serialized to keep JSON compact.
            # Use the dataclass attributes directly for plotting.
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def save_json(self, path: str) -> None:
        """Save results to JSON file."""
        with open(path, 'w') as f:
            f.write(self.to_json())


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
    # confuse users (e.g. old aggressors_*.png when adjoint is now disabled).
    for pattern in ('waveform_*.png', 'aggressors_*.png'):
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

        # Zoom window: 1ns centered on peak time
        ZOOM_WINDOW_NS = 1.0
        half_window = ZOOM_WINDOW_NS / 2
        t_min = max(t_ns[0], inst.peak_time_ns - half_window)
        t_max = min(t_ns[-1], inst.peak_time_ns + half_window)

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

    # 3. Aggressor contribution bar plots (per victim, sorted by distance)
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    # Adaptive thresholds for plot readability
    LABEL_ALL_THRESHOLD = 20      # Show all labels if ≤ this many aggressors
    TARGET_LABELS = 15            # Target number of x-axis labels when crowded
    ANNOTATE_THRESHOLD = 20       # Annotate all bars if ≤ this many aggressors
    TOP_N_ANNOTATE = 10           # Number of top contributors to annotate when crowded
    EDGE_LINE_THRESHOLD = 50      # Remove bar edges if > this many aggressors

    for i, inst in enumerate(result.worst_instances):
        if not inst.top_aggressors:
            continue

        # Sort aggressors by distance
        sorted_aggressors = sorted(inst.top_aggressors, key=lambda a: a.distance_um)
        n_aggressors = len(sorted_aggressors)
        distances = [agg.distance_um for agg in sorted_aggressors]
        contributions = [agg.contribution_mV for agg in sorted_aggressors]

        # Adaptive figure width based on number of aggressors
        fig_width = max(12, min(24, 12 + (n_aggressors - 20) * 0.1))
        fig, ax = plt.subplots(figsize=(fig_width, 6))

        # Create bar positions and labels
        x_pos = np.arange(n_aggressors)
        cmap = matplotlib.colormaps.get_cmap('viridis')
        bar_colors = cmap(np.linspace(0.2, 0.8, n_aggressors))

        # Adaptive bar edge styling
        edge_color = 'black' if n_aggressors <= EDGE_LINE_THRESHOLD else 'none'
        edge_width = 0.5 if n_aggressors <= EDGE_LINE_THRESHOLD else 0

        bars = ax.bar(x_pos, contributions, color=bar_colors,
                      edgecolor=edge_color, linewidth=edge_width)

        # Adaptive value annotations - only annotate top contributors when crowded
        if n_aggressors <= ANNOTATE_THRESHOLD:
            # Annotate all bars
            for bar, contrib in zip(bars, contributions):
                height = bar.get_height()
                ax.annotate(f'{contrib:.2f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=8)
        else:
            # Only annotate top N contributors by magnitude
            contrib_array = np.array(contributions)
            top_indices = set(np.argsort(contrib_array)[-TOP_N_ANNOTATE:])
            for idx, (bar, contrib) in enumerate(zip(bars, contributions)):
                if idx in top_indices:
                    height = bar.get_height()
                    ax.annotate(f'{contrib:.2f}',
                               xy=(bar.get_x() + bar.get_width() / 2, height),
                               xytext=(0, 2),
                               textcoords="offset points",
                               ha='center', va='bottom', fontsize=7, rotation=45)

        # Adaptive x-axis labels
        if n_aggressors <= LABEL_ALL_THRESHOLD:
            # Show all labels
            x_labels = [f'{d:.0f}' for d in distances]
            ax.set_xticks(x_pos)
            ax.set_xticklabels(x_labels, fontsize=9)
        else:
            # Show subset of labels to avoid overlap
            label_stride = max(1, n_aggressors // TARGET_LABELS)
            visible_ticks = x_pos[::label_stride]
            visible_labels = [f'{distances[j]:.0f}' for j in range(0, n_aggressors, label_stride)]
            ax.set_xticks(visible_ticks)
            ax.set_xticklabels(visible_labels, fontsize=8, rotation=45, ha='right')

        ax.set_xlabel('Distance to Victim (um)')
        ax.set_ylabel('Contribution (mV)')

        # Enhanced title with more context
        total_agg_contrib = sum(agg.contribution_mV for agg in inst.top_aggressors)
        ax.set_title(f'Aggressor Contributions vs Distance\n'
                     f'Victim #{i+1}: {truncate_name(inst.instance_name)} (Node: {inst.node})\n'
                     f'Peak IR-drop: {inst.peak_total_mV:.3f} mV | '
                     f'Self: {inst.self_contribution_mV:.3f} mV | '
                     f'Aggressors: {total_agg_contrib:.3f} mV')
        ax.grid(axis='y', alpha=0.3)

        # Add a color bar legend for distance
        sm = ScalarMappable(cmap='viridis',
                            norm=Normalize(vmin=min(distances), vmax=max(distances)))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.6)
        cbar.set_label('Distance (um)')

        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f'aggressors_{i+1}.png'), dpi=150)
        if show:
            plt.show()
        plt.close()

    # 4. Peak IR-drop stripe heatmaps (if peak_ir_drop_per_node available)
    if result.peak_ir_drop_per_node:
        from visualization.stripe_heatmap import plot_stripe_heatmap

        # Helper function to extract layer from node name
        def extract_layer_from_node(node: str) -> Optional[str]:
            parts = str(node).split('_')
            if len(parts) >= 3:
                return parts[2]
            return None

        # Build markers from worst instances
        markers = []
        for inst in result.worst_instances:
            layer = extract_layer_from_node(inst.node)
            if layer:
                markers.append((inst.x, inst.y, layer))

        # Build windows from worst instances
        windows = []
        for inst in result.worst_instances:
            layer = extract_layer_from_node(inst.node)
            if layer and inst.window_bounds:
                x_min, x_max, y_min, y_max = inst.window_bounds
                windows.append((x_min, x_max, y_min, y_max, layer))

        if verbose:
            print(f"Generating peak IR-drop heatmaps...")

        plot_stripe_heatmap(
            node_values=result.peak_ir_drop_per_node,
            layers=heatmap_layers,
            plot_dir=plot_dir,
            max_stripes=max_stripes,
            title_prefix='Peak IR-Drop',
            value_label='Peak IR-Drop (mV)',
            value_scale=1000.0,  # V to mV
            markers=markers,
            windows=windows,
            show=show,
            verbose=verbose,
            graph=graph,
        )

    print(f"Plots saved to: {plot_dir}")


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

    # Decomposition analysis for each worst instance
    decompositions: List[InstanceDecomposition] = []
    t0_decomp = time_module.perf_counter()

    method_enum = IntegrationMethod.TRAPEZOIDAL if integration_method == 'trap' else IntegrationMethod.BACKWARD_EULER

    for rank, (inst_name, node, x, y, _) in enumerate(worst_instances, 1):
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

        # Free TransientResult objects - waveforms already extracted (~80 MB per instance)
        del results, result_full, result_near, result_far

    timings['decomposition'] = time_module.perf_counter() - t0_decomp

    # Adjoint aggressor analysis (if enabled)
    if aggressor_top_k > 0 and decompositions:
        if verbose:
            print()
            print("Running adjoint sensitivity analysis for top aggressors...")

        t0_adjoint = time_module.perf_counter()

        # Create adjoint solver from transient solver (shares RC system)
        adjoint_solver = AdjointSensitivitySolver.from_transient_solver(solver)

        # Prepare adjoint context for batch solving (reuses LU factorization)
        adjoint_ctx = adjoint_solver.prepare(dt=dt)

        for rank, decomp in enumerate(decompositions, 1):
            if verbose:
                print(f"  Analyzing aggressors for victim {rank}/{len(decompositions)}: {decomp.node}")

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

        timings['adjoint_analysis'] = time_module.perf_counter() - t0_adjoint
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
        """
    )

    # Positional argument
    parser.add_argument('netlist_dir', nargs='?', help='Path to PDN netlist directory')

    # Config file
    parser.add_argument('--config', type=str, help='Path to YAML config file')

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

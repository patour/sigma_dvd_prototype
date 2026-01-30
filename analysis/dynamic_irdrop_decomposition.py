#!/usr/bin/env python3
"""Dynamic IR-Drop Decomposition Analysis.

Analyzes dynamic IR-drop in a PDN netlist and decomposes the IR-drop at worst-case
instances into contributions from "near" instances (within a local window) and
"far" instances (outside the window).

This analysis helps identify whether IR-drop issues are caused by local current
density or by distributed grid resistance effects.

Usage:
    # Via command line arguments (with time units)
    python -m analysis.dynamic_irdrop_decomposition ./pdn/netlist_test \\
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

    # Via config file
    python -m analysis.dynamic_irdrop_decomposition --config config.yaml

Example output:
    ================================================================================
    DYNAMIC IR-DROP DECOMPOSITION ANALYSIS RESULTS
    ================================================================================
    Netlist: ./pdn/netlist_test
    Method: transient (RC)
    Time range: 0.00 ns - 100.00 ns (dt=0.10 ns)
    Window size: 10.0% of design

    TOP-5 WORST INSTANCES WITH NEAR/FAR DECOMPOSITION
    ================================================================================
    Rank  Instance                 Peak(mV)  Near(mV)  Far(mV)  Near%  Far%
    --------------------------------------------------------------------------------
    1     i_cpu_core:VDD:...       12.345    8.234     4.111    66.7%  33.3%
    ...
"""

from __future__ import annotations

import argparse
import json
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

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
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
            'waveforms': {
                't_ns': (self.t_array * 1e9).tolist(),
                'total_mV': (self.ir_drop_total * 1000).tolist(),
                'near_mV': (self.ir_drop_near * 1000).tolist(),
                'far_mV': (self.ir_drop_far * 1000).tolist(),
            },
        }


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

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            'netlist_dir': self.netlist_dir,
            'net_name': self.net_name,
            'method': self.method,
            'integration_method': self.integration_method,
            't_start_ns': self.t_start_ns,
            't_end_ns': self.t_end_ns,
            'dt_ns': self.dt_ns,
            'window_percent': self.window_percent,
            'grid_bounds': list(self.grid_bounds),
            'worst_instances': [inst.to_dict() for inst in self.worst_instances],
            'timings': self.timings,
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
    from core.unified_solver import (
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

def print_results(result: DecompositionResult) -> None:
    """Print results to console in formatted table."""
    print("=" * 80)
    print("DYNAMIC IR-DROP DECOMPOSITION ANALYSIS RESULTS")
    print("=" * 80)
    print(f"Netlist: {result.netlist_dir}")
    print(f"Net: {result.net_name}")
    print(f"Method: {result.method} ({result.integration_method.upper()})")
    print(f"Time range: {result.t_start_ns:.2f} ns - {result.t_end_ns:.2f} ns (dt={result.dt_ns:.2f} ns)")
    print(f"Window size: {result.window_percent:.1f}% of design")
    print(f"Grid bounds: ({result.grid_bounds[0]:.0f}, {result.grid_bounds[1]:.0f}) x "
          f"({result.grid_bounds[2]:.0f}, {result.grid_bounds[3]:.0f})")
    print()

    print(f"TOP-{len(result.worst_instances)} WORST INSTANCES WITH NEAR/FAR DECOMPOSITION")
    print("=" * 80)
    print(f"{'Rank':<5} {'Instance':<30} {'Peak(mV)':<10} {'Near(mV)':<10} {'Far(mV)':<10} {'Near%':<7} {'Far%':<7}")
    print("-" * 80)

    for i, inst in enumerate(result.worst_instances, 1):
        inst_short = inst.instance_name[:28] + ".." if len(inst.instance_name) > 30 else inst.instance_name
        print(f"{i:<5} {inst_short:<30} {inst.peak_total_mV:<10.3f} {inst.peak_near_mV:<10.3f} "
              f"{inst.peak_far_mV:<10.3f} {inst.near_fraction_at_peak:<7.1f} {inst.far_fraction_at_peak:<7.1f}")

    print()

    # Summary statistics
    if result.worst_instances:
        avg_near_pct = np.mean([inst.near_fraction_at_peak for inst in result.worst_instances])
        avg_far_pct = np.mean([inst.far_fraction_at_peak for inst in result.worst_instances])
        print(f"Average near contribution: {avg_near_pct:.1f}%")
        print(f"Average far contribution: {avg_far_pct:.1f}%")

    print()
    print("Timing breakdown:")
    for key, val in result.timings.items():
        print(f"  {key}: {val:.3f} s")


# =============================================================================
# Plotting Functions
# =============================================================================

def generate_plots(
    result: DecompositionResult,
    plot_dir: str,
    show: bool = False,
) -> None:
    """Generate analysis plots.

    Creates:
    1. Bar chart: Near vs Far contributions for all worst instances
    2. Waveform plots: Time-domain decomposition for each worst instance
    3. Spatial map: Worst instance locations with their analysis windows (if matplotlib available)

    Args:
        result: DecompositionResult with analysis data
        plot_dir: Directory to save plots
        show: If True, display plots interactively
    """
    try:
        import matplotlib
        if not show:
            matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
    except ImportError:
        print("Warning: matplotlib not available, skipping plots")
        return

    os.makedirs(plot_dir, exist_ok=True)

    # 1. Bar chart: Near vs Far contributions
    fig, ax = plt.subplots(figsize=(12, 6))
    n_inst = len(result.worst_instances)
    x = np.arange(n_inst)
    width = 0.35

    near_drops = [inst.peak_near_mV for inst in result.worst_instances]
    far_drops = [inst.peak_far_mV for inst in result.worst_instances]
    labels = [f"#{i+1}\n{inst.instance_name[:15]}..." if len(inst.instance_name) > 15
              else f"#{i+1}\n{inst.instance_name}"
              for i, inst in enumerate(result.worst_instances)]

    bars_near = ax.bar(x - width/2, near_drops, width, label='Near', color='royalblue')
    bars_far = ax.bar(x + width/2, far_drops, width, label='Far', color='coral')

    ax.set_xlabel('Instance')
    ax.set_ylabel('Peak IR-Drop (mV)')
    ax.set_title(f'Near vs Far IR-Drop Contributions ({result.net_name})')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'near_far_comparison.png'), dpi=150)
    if show:
        plt.show()
    plt.close()

    # 2. Waveform plots for each instance
    for i, inst in enumerate(result.worst_instances):
        fig, ax = plt.subplots(figsize=(10, 5))

        t_ns = inst.t_array * 1e9
        ax.plot(t_ns, inst.ir_drop_total * 1000, 'k-', linewidth=2, label='Total')
        ax.plot(t_ns, inst.ir_drop_near * 1000, 'b-', linewidth=1.5, label='Near')
        ax.plot(t_ns, inst.ir_drop_far * 1000, 'r-', linewidth=1.5, label='Far')

        ax.axvline(inst.peak_time_ns, color='gray', linestyle='--', alpha=0.5, label=f'Peak @ {inst.peak_time_ns:.2f} ns')

        ax.set_xlabel('Time (ns)')
        ax.set_ylabel('IR-Drop (mV)')
        ax.set_title(f'IR-Drop Decomposition: {inst.instance_name[:40]}\n'
                     f'Peak: {inst.peak_total_mV:.3f} mV = {inst.peak_near_mV:.3f} (near) + {inst.peak_far_mV:.3f} (far)')
        ax.legend(loc='upper right')
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f'waveform_{i+1}.png'), dpi=150)
        if show:
            plt.show()
        plt.close()

    # 3. Spatial map with windows
    fig, ax = plt.subplots(figsize=(10, 8))

    # Draw grid bounds
    x_min, x_max, y_min, y_max = result.grid_bounds
    ax.set_xlim(x_min - (x_max - x_min) * 0.05, x_max + (x_max - x_min) * 0.05)
    ax.set_ylim(y_min - (y_max - y_min) * 0.05, y_max + (y_max - y_min) * 0.05)

    # Draw analysis windows
    colors = plt.cm.tab10(np.linspace(0, 1, len(result.worst_instances)))
    for i, inst in enumerate(result.worst_instances):
        wx_min, wx_max, wy_min, wy_max = inst.window_bounds
        rect = patches.Rectangle(
            (wx_min, wy_min), wx_max - wx_min, wy_max - wy_min,
            linewidth=2, edgecolor=colors[i], facecolor=colors[i], alpha=0.2
        )
        ax.add_patch(rect)

        # Mark instance location
        ax.plot(inst.x, inst.y, 'o', markersize=10, color=colors[i],
                label=f'#{i+1}: {inst.peak_total_mV:.2f} mV')

    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    ax.set_title(f'Worst Instance Locations with Analysis Windows\n{result.net_name}')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'spatial_map.png'), dpi=150)
    if show:
        plt.show()
    plt.close()

    # 4. Pie chart: Average near/far breakdown
    fig, ax = plt.subplots(figsize=(8, 8))

    avg_near = np.mean([inst.near_fraction_at_peak for inst in result.worst_instances])
    avg_far = np.mean([inst.far_fraction_at_peak for inst in result.worst_instances])

    sizes = [avg_near, avg_far]
    labels = [f'Near\n{avg_near:.1f}%', f'Far\n{avg_far:.1f}%']
    colors_pie = ['royalblue', 'coral']
    explode = (0.05, 0.05)

    ax.pie(sizes, explode=explode, labels=labels, colors=colors_pie,
           autopct='', startangle=90)
    ax.set_title(f'Average Near/Far Contribution at Peak\n(Window = {result.window_percent}% of design)')

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'pie_chart.png'), dpi=150)
    if show:
        plt.show()
    plt.close()

    print(f"Plots saved to: {plot_dir}")


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
    verbose: bool = False,
) -> DecompositionResult:
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
        verbose: Print progress

    Returns:
        DecompositionResult with analysis data.
    """
    timings: Dict[str, float] = {}
    t0_total = time_module.perf_counter()

    # Import PDN parser and core modules
    from pdn.pdn_parser import NetlistParser
    from core import create_model_from_pdn
    from core.transient_solver import TransientIRDropSolver, IntegrationMethod

    # Parse netlist
    t0_parse = time_module.perf_counter()
    if verbose:
        print(f"Parsing netlist: {netlist_dir}")
    parser = NetlistParser(netlist_dir)
    graph = parser.parse()
    timings['parse'] = time_module.perf_counter() - t0_parse

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
        from pdn.pdn_parser import CurrentSource
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

    # Create solver (use vectorize_threshold=0 to enable solve_transient_multi_rhs)
    solver = TransientIRDropSolver(model, graph, vectorize_threshold=0)

    # Build source name list and index mapping
    source_names = list(current_sources.keys())
    name_to_idx = {name: i for i, name in enumerate(source_names)}
    n_sources = len(source_names)

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
            verbose=False,
        )
        result_full, result_near, result_far = results

        # Compute statistics
        decomp = compute_decomposition_stats(
            inst_name, node, x, y, window,
            result_full, result_near, result_far,
            len(near_names), len(far_names),
        )
        decompositions.append(decomp)

    timings['decomposition'] = time_module.perf_counter() - t0_decomp
    timings['total'] = time_module.perf_counter() - t0_total

    return DecompositionResult(
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

    # Output configuration
    output_config = config.get('output', {})
    result['output'] = args.output or output_config.get('json_file')
    result['plot'] = args.plot or output_config.get('plot', False)
    result['plot_dir'] = args.plot_dir or output_config.get('plot_dir', './plots')
    result['verbose'] = args.verbose or output_config.get('verbose', False)

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
  python -m analysis.dynamic_irdrop_decomposition ./pdn/netlist_test --end-time 100ns --dt 100ps

  # With config file
  python -m analysis.dynamic_irdrop_decomposition --config config.yaml

  # Override config with CLI
  python -m analysis.dynamic_irdrop_decomposition --config config.yaml --top-k 10

  # Pre-defined instances
  python -m analysis.dynamic_irdrop_decomposition ./pdn/netlist_test --instances "inst1,inst2,inst3"
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

    # Pre-defined instances
    parser.add_argument('--instances', type=str, help='Comma-separated list of instance/node names')
    parser.add_argument('--instances-file', type=str, help='File with instance/node names (one per line)')

    # Solver backend
    parser.add_argument('--backend', choices=['auto', 'splu', 'cholmod'], help='Solver backend')
    parser.add_argument('--cholmod-ordering', type=str, help='CHOLMOD ordering method')
    parser.add_argument('--cholmod-mode', type=str, help='CHOLMOD factorization mode')
    parser.add_argument('--cholmod-use-long', action='store_true', help='Use 64-bit indices for CHOLMOD')

    # Output
    parser.add_argument('--output', '-o', type=str, help='Output JSON file path')
    parser.add_argument('--plot', action='store_true', help='Generate plots')
    parser.add_argument('--plot-dir', type=str, help='Directory for plots')
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

    # Run analysis
    if merged['verbose']:
        print("Starting dynamic IR-drop decomposition analysis...")
        print(f"  Netlist: {merged['netlist_dir']}")
        print(f"  Net: {merged['net']}")
        print(f"  Time: {merged['t_start']*1e9:.2f} ns to {merged['t_end']*1e9:.2f} ns, dt={merged['dt']*1e9:.3f} ns")
        print(f"  Window: {merged['window_percent']}%")
        print()

    result = analyze_dynamic_irdrop_decomposition(
        netlist_dir=merged['netlist_dir'],
        net=merged['net'],
        t_start=merged['t_start'],
        t_end=merged['t_end'],
        dt=merged['dt'],
        top_k=merged['top_k'],
        window_percent=merged['window_percent'],
        integration_method=merged['integration_method'],
        instances=merged['instances'],
        verbose=merged['verbose'],
    )

    # Print results
    print_results(result)

    # Save JSON
    if merged['output']:
        result.save_json(merged['output'])
        print(f"\nResults saved to: {merged['output']}")

    # Generate plots
    if merged['plot']:
        generate_plots(result, merged['plot_dir'], show=False)

    return result


if __name__ == '__main__':
    main()

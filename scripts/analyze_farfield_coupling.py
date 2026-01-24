#!/usr/bin/env python
"""Analyze far-field to local boundary coupling in PDN grids.

This script runs experiments to determine whether far-field switching influences
a local boundary region through a small number of smooth spatial patterns.

Usage:
    python scripts/analyze_farfield_coupling.py --netlist ./pdn/netlist_test --net VDD
    python scripts/analyze_farfield_coupling.py --netlist ./pdn/netlist_test --net VDD --validate
    python scripts/analyze_farfield_coupling.py --netlist ./pdn/netlist_test --net VDD --sweep-position
    python scripts/analyze_farfield_coupling.py --netlist ./pdn/netlist_test --net VDD --sweep-size
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pdn.pdn_parser import NetlistParser
from core import create_model_from_pdn, UnifiedIRDropSolver
from core.farfield_analysis import (
    run_farfield_analysis,
    compute_window_bounds,
    define_window_and_boundary,
    partition_farfield_by_distance_rings,
    generate_block_injections,
    compute_boundary_response_matrix,
    analyze_response_matrix,
)
from core.tiling import TileManager


def plot_singular_values(results: dict, output_path: str = None, show: bool = False):
    """Plot singular value spectrum."""
    sigma = results['svd']['singular_values']
    sigma_norm = sigma / sigma[0]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Linear scale
    ax = axes[0]
    ax.semilogy(np.arange(len(sigma_norm)) + 1, sigma_norm, 'b.-', markersize=8)
    ax.axhline(0.01, color='r', linestyle='--', label='1% threshold')
    ax.axhline(0.001, color='orange', linestyle='--', label='0.1% threshold')
    ax.set_xlabel('Mode index k')
    ax.set_ylabel(r'$\sigma_k / \sigma_1$')
    ax.set_title('Singular Value Spectrum')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, len(sigma_norm) + 1)

    # Cumulative energy
    ax = axes[1]
    cum_energy = results['svd']['cumulative_energy']
    ax.plot(np.arange(len(cum_energy)) + 1, cum_energy * 100, 'g.-', markersize=8)
    ax.axhline(90, color='r', linestyle='--', label='90%')
    ax.axhline(95, color='orange', linestyle='--', label='95%')
    ax.axhline(99, color='purple', linestyle='--', label='99%')
    ax.set_xlabel('Number of modes')
    ax.set_ylabel('Cumulative energy (%)')
    ax.set_title('Cumulative Energy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, len(cum_energy) + 1)
    ax.set_ylim(0, 105)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_dominant_modes(results: dict, n_modes: int = 6, output_path: str = None, show: bool = False):
    """Plot dominant boundary voltage patterns (left singular vectors)."""
    U = results['svd']['U']
    sigma = results['svd']['singular_values']
    boundary_ports = results['boundary_ports']
    port_coords = results['port_coords']

    coords = np.array([port_coords[p] for p in boundary_ports])

    n_modes = min(n_modes, U.shape[1])
    n_cols = 3
    n_rows = (n_modes + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
    axes = np.atleast_2d(axes)

    for i in range(n_modes):
        row, col = i // n_cols, i % n_cols
        ax = axes[row, col]

        u = U[:, i]
        sc = ax.scatter(coords[:, 0], coords[:, 1], c=u, cmap='RdBu_r',
                       s=30, edgecolors='k', linewidths=0.5)
        plt.colorbar(sc, ax=ax, shrink=0.8)

        smoothness = results['svd']['smoothness_scores'][i] if i < len(results['svd']['smoothness_scores']) else {}
        grad_energy = smoothness.get('gradient_energy', 0)

        ax.set_title(f'Mode {i+1}: $\\sigma$={sigma[i]:.2e}\ngrad_energy={grad_energy:.2f}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal')

    # Hide unused subplots
    for i in range(n_modes, n_rows * n_cols):
        row, col = i // n_cols, i % n_cols
        axes[row, col].set_visible(False)

    plt.suptitle('Dominant Boundary Voltage Patterns (Left Singular Vectors)', y=1.02)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_setup_visualization(results: dict, output_path: str = None, show: bool = False):
    """Visualize the experiment setup: window, boundary ports, far-field blocks."""
    port_coords = results['port_coords']
    boundary_ports = results['boundary_ports']
    window_bounds = results['window_bounds']
    grid_bounds = results['grid_bounds']
    block_centers = results['block_centers']

    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot all ports
    all_ports = list(port_coords.keys())
    all_coords = np.array([port_coords[p] for p in all_ports])
    ax.scatter(all_coords[:, 0], all_coords[:, 1], c='lightgray', s=10, label='All ports', alpha=0.5)

    # Plot boundary ports (B)
    boundary_coords = np.array([port_coords[p] for p in boundary_ports])
    ax.scatter(boundary_coords[:, 0], boundary_coords[:, 1], c='blue', s=30, label=f'B (|B|={len(boundary_ports)})')

    # Plot window rectangle
    wx_min, wx_max, wy_min, wy_max = window_bounds
    rect = plt.Rectangle((wx_min, wy_min), wx_max - wx_min, wy_max - wy_min,
                         fill=False, edgecolor='blue', linewidth=2, linestyle='--', label='Window W')
    ax.add_patch(rect)

    # Plot block centers
    if block_centers:
        bc = np.array(block_centers)
        ax.scatter(bc[:, 0], bc[:, 1], c='red', s=100, marker='x', linewidths=2, label=f'Block centers (K={len(block_centers)})')

    # Plot grid bounds
    gx_min, gx_max, gy_min, gy_max = grid_bounds
    rect_grid = plt.Rectangle((gx_min, gy_min), gx_max - gx_min, gy_max - gy_min,
                              fill=False, edgecolor='black', linewidth=1, linestyle='-')
    ax.add_patch(rect_grid)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Experiment Setup')
    ax.legend(loc='upper right')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")

    if show:
        plt.show()
    else:
        plt.close()


def run_position_sweep(
    solver: UnifiedIRDropSolver,
    partition_layer: str,
    window_fraction: float = 0.25,
    n_blocks_x: int = 3,
    n_blocks_y: int = 3,
    n_random_patterns: int = 3,
    total_current: float = 1.0,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """Run analysis for multiple window positions."""
    positions = ['center', 'corner_ll', 'corner_ur', 'corner_lr', 'corner_ul']
    results = {}

    for pos in positions:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Position: {pos}")
            print('='*60)

        try:
            res = run_farfield_analysis(
                solver, partition_layer,
                window_position=pos,
                window_fraction=window_fraction,
                n_blocks_x=n_blocks_x,
                n_blocks_y=n_blocks_y,
                n_random_patterns=n_random_patterns,
                total_current=total_current,
                seed=seed,
                validate=False,
                verbose=verbose,
            )
            results[pos] = res
        except ValueError as e:
            print(f"  Skipped: {e}")
            results[pos] = None

    return results


def run_size_sweep(
    solver: UnifiedIRDropSolver,
    partition_layer: str,
    window_position: str = 'center',
    n_blocks_x: int = 3,
    n_blocks_y: int = 3,
    n_random_patterns: int = 3,
    total_current: float = 1.0,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """Run analysis for multiple window sizes."""
    fractions = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
    results = {}

    for frac in fractions:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Window fraction: {frac*100:.0f}%")
            print('='*60)

        try:
            res = run_farfield_analysis(
                solver, partition_layer,
                window_position=window_position,
                window_fraction=frac,
                n_blocks_x=n_blocks_x,
                n_blocks_y=n_blocks_y,
                n_random_patterns=n_random_patterns,
                total_current=total_current,
                seed=seed,
                validate=False,
                verbose=verbose,
            )
            results[frac] = res
        except ValueError as e:
            print(f"  Skipped: {e}")
            results[frac] = None

    return results


def run_distance_analysis(
    solver: UnifiedIRDropSolver,
    partition_layer: str,
    window_position: str = 'center',
    window_fraction: float = 0.25,
    n_rings: int = 4,
    n_random_patterns: int = 3,
    total_current: float = 1.0,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """Analyze rank contribution from different distance rings."""
    model = solver.model

    # Setup
    top_nodes, bottom_nodes, port_nodes, _ = model._decompose_at_layer(partition_layer)
    tile_manager = TileManager(model)
    bottom_coords, port_coords, grid_bounds = tile_manager.extract_bottom_grid_coordinates(
        bottom_nodes, port_nodes
    )
    window_bounds = compute_window_bounds(port_coords, grid_bounds, window_position, window_fraction)
    boundary_ports_set, boundary_ports = define_window_and_boundary(
        port_nodes, port_coords, window_bounds
    )

    if verbose:
        print(f"Boundary ports |B|: {len(boundary_ports)}")

    # Partition by distance rings
    rings = partition_farfield_by_distance_rings(
        bottom_nodes, bottom_coords, window_bounds, grid_bounds, n_rings
    )

    if verbose:
        for i, ring in enumerate(rings):
            print(f"Ring {i}: {len(ring)} nodes")

    # Prepare solver context once for efficient batch solving across all rings
    if verbose:
        print("Preparing solver context for batch solving...")

    coupled_context = solver.prepare_hierarchical_coupled(
        partition_layer=partition_layer,
        solver='gmres',
        tol=1e-8,
        maxiter=500,
        preconditioner='block_diagonal',
    )

    results = {'rings': [], 'ring_sizes': [len(r) for r in rings]}

    for ring_idx, ring_nodes in enumerate(rings):
        if len(ring_nodes) == 0:
            if verbose:
                print(f"\nRing {ring_idx}: empty, skipping")
            results['rings'].append(None)
            continue

        if verbose:
            print(f"\n{'='*60}")
            print(f"Ring {ring_idx} (distance band {ring_idx})")
            print('='*60)

        # Generate patterns for this ring only
        patterns, _ = generate_block_injections(
            [ring_nodes], n_random_patterns, total_current=total_current, seed=seed
        )

        if verbose:
            print(f"  Patterns: {len(patterns)}")

        # Compute response matrix using prepared context
        H = compute_boundary_response_matrix(
            solver, patterns, boundary_ports, partition_layer, verbose=False,
            context=coupled_context
        )

        # Analyze
        svd_res = analyze_response_matrix(H, boundary_ports, port_coords)

        if verbose:
            print(f"  Effective rank (1%): {svd_res['effective_rank_1pct']}")
            print(f"  Rank for 99% energy: {svd_res['rank_99pct_energy']}")

        results['rings'].append({
            'H': H,
            'svd': svd_res,
            'n_nodes': len(ring_nodes),
        })

    return results


def plot_position_sweep_comparison(results: dict, output_path: str = None, show: bool = False):
    """Plot comparison of SVD results across window positions."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Singular value comparison
    ax = axes[0]
    for pos, res in results.items():
        if res is None:
            continue
        sigma = res['svd']['singular_values']
        sigma_norm = sigma / sigma[0]
        ax.semilogy(np.arange(len(sigma_norm)) + 1, sigma_norm, '.-', label=pos, markersize=6)

    ax.axhline(0.01, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Mode index k')
    ax.set_ylabel(r'$\sigma_k / \sigma_1$')
    ax.set_title('Singular Values by Position')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Effective rank comparison
    ax = axes[1]
    positions = []
    ranks_1pct = []
    ranks_99pct = []
    n_boundary = []

    for pos, res in results.items():
        if res is None:
            continue
        positions.append(pos)
        ranks_1pct.append(res['svd']['effective_rank_1pct'])
        ranks_99pct.append(res['svd']['rank_99pct_energy'])
        n_boundary.append(res['svd']['n_boundary'])

    x = np.arange(len(positions))
    width = 0.35
    ax.bar(x - width/2, ranks_1pct, width, label='Rank (1% thresh)')
    ax.bar(x + width/2, ranks_99pct, width, label='Rank (99% energy)')
    ax.set_xticks(x)
    ax.set_xticklabels(positions, rotation=45)
    ax.set_ylabel('Effective Rank')
    ax.set_title('Effective Rank by Position')
    ax.legend()

    # Add |B| annotation
    for i, (n, r) in enumerate(zip(n_boundary, ranks_99pct)):
        ax.annotate(f'|B|={n}', (i, r + 0.5), ha='center', fontsize=8)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_size_sweep_scaling(results: dict, output_path: str = None, show: bool = False):
    """Plot how effective rank scales with window size."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    fractions = []
    n_boundary = []
    ranks_1pct = []
    ranks_99pct = []

    for frac, res in sorted(results.items()):
        if res is None:
            continue
        fractions.append(frac)
        n_boundary.append(res['svd']['n_boundary'])
        ranks_1pct.append(res['svd']['effective_rank_1pct'])
        ranks_99pct.append(res['svd']['rank_99pct_energy'])

    # Rank vs fraction
    ax = axes[0]
    ax.plot(np.array(fractions) * 100, ranks_99pct, 'bo-', label='Rank (99% energy)', markersize=8)
    ax.plot(np.array(fractions) * 100, ranks_1pct, 'rs--', label='Rank (1% thresh)', markersize=8)
    ax.set_xlabel('Window fraction (%)')
    ax.set_ylabel('Effective Rank')
    ax.set_title('Rank vs Window Size')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Rank vs |B|
    ax = axes[1]
    ax.plot(n_boundary, ranks_99pct, 'bo-', label='Rank (99% energy)', markersize=8)
    ax.plot(n_boundary, ranks_1pct, 'rs--', label='Rank (1% thresh)', markersize=8)
    ax.plot([0, max(n_boundary)], [0, max(n_boundary)], 'k--', alpha=0.3, label='rank = |B|')
    ax.set_xlabel('|B| (number of boundary ports)')
    ax.set_ylabel('Effective Rank')
    ax.set_title('Rank vs Boundary Size')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_distance_analysis(results: dict, output_path: str = None, show: bool = False):
    """Plot how rank varies with distance from window."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ring_indices = []
    ranks_1pct = []
    ranks_99pct = []
    ring_sizes = results['ring_sizes']

    for i, ring_res in enumerate(results['rings']):
        if ring_res is None:
            continue
        ring_indices.append(i)
        ranks_1pct.append(ring_res['svd']['effective_rank_1pct'])
        ranks_99pct.append(ring_res['svd']['rank_99pct_energy'])

    # Rank vs ring index
    ax = axes[0]
    ax.plot(ring_indices, ranks_99pct, 'bo-', label='Rank (99% energy)', markersize=10)
    ax.plot(ring_indices, ranks_1pct, 'rs--', label='Rank (1% thresh)', markersize=10)
    ax.set_xlabel('Ring index (0=closest to window)')
    ax.set_ylabel('Effective Rank')
    ax.set_title('Rank vs Distance from Window')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Singular value spectra for each ring
    ax = axes[1]
    for i, ring_res in enumerate(results['rings']):
        if ring_res is None:
            continue
        sigma = ring_res['svd']['singular_values']
        if len(sigma) > 0:
            sigma_norm = sigma / sigma[0]
            ax.semilogy(np.arange(len(sigma_norm)) + 1, sigma_norm, '.-',
                       label=f'Ring {i} ({ring_sizes[i]} nodes)', markersize=6)

    ax.axhline(0.01, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Mode index k')
    ax.set_ylabel(r'$\sigma_k / \sigma_1$')
    ax.set_title('Singular Values by Distance Ring')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")

    if show:
        plt.show()
    else:
        plt.close()


def print_summary(results: dict):
    """Print summary of analysis results."""
    svd = results['svd']
    config = results['config']

    print("\n" + "=" * 60)
    print("ANALYSIS SUMMARY")
    print("=" * 60)

    print(f"\nConfiguration:")
    print(f"  Partition layer: {config['partition_layer']}")
    print(f"  Window position: {config['window_position']}")
    print(f"  Window fraction: {config['window_fraction']*100:.0f}%")
    print(f"  Far-field blocks: {config['n_blocks_x']}x{config['n_blocks_y']}")
    print(f"  Random patterns/block: {config['n_random_patterns']}")

    print(f"\nDimensions:")
    print(f"  Boundary ports |B|: {svd['n_boundary']}")
    print(f"  Injection patterns: {svd['n_patterns']}")
    print(f"  H matrix shape: ({svd['n_boundary']}, {svd['n_patterns']})")

    print(f"\nSVD Results:")
    print(f"  Effective rank (1% threshold): {svd['effective_rank_1pct']}")
    print(f"  Effective rank (0.1% threshold): {svd['effective_rank_01pct']}")
    print(f"  Rank for 90% energy: {svd['rank_90pct_energy']}")
    print(f"  Rank for 95% energy: {svd['rank_95pct_energy']}")
    print(f"  Rank for 99% energy: {svd['rank_99pct_energy']}")
    print(f"  Decay rate: {svd['decay_rate']:.3f}")
    print(f"  Compression ratio (1%): {svd['compression_ratio_1pct']:.1f}x")
    print(f"  Compression ratio (99%): {svd['compression_ratio_99pct']:.1f}x")

    print(f"\nTop 5 singular values:")
    sigma_0 = svd['singular_values'][0] if len(svd['singular_values']) > 0 and svd['singular_values'][0] > 0 else 1.0
    for i, s in enumerate(svd['singular_values'][:5]):
        print(f"  σ_{i+1} = {s:.4e} ({s/sigma_0*100:.1f}%)")

    print(f"\nSmoothness of dominant modes:")
    if svd['smoothness_scores']:
        for i, sm in enumerate(svd['smoothness_scores'][:5]):
            print(f"  Mode {i+1}: gradient_energy={sm['gradient_energy']:.3f}, TV={sm['total_variation_2d']:.3f}")
    else:
        print("  (No smoothness data available)")

    if results['validation'] is not None:
        val = results['validation']
        print(f"\nValidation (vs flat solve):")
        print(f"  Max diff: {val['max_diff_mV']:.3f} mV")
        print(f"  Mean diff: {val['mean_diff_mV']:.3f} mV")
        print(f"  RMSE: {val['rmse']*1000:.3f} mV")

    # Conclusion
    print("\n" + "-" * 60)
    if svd['rank_99pct_energy'] < svd['n_boundary'] * 0.3:
        print("CONCLUSION: Far-field coupling is LOW-RANK")
        print(f"  Only {svd['rank_99pct_energy']} modes capture 99% of energy")
        print(f"  Compression ratio: {svd['compression_ratio_99pct']:.1f}x")
    elif svd['rank_99pct_energy'] < svd['n_boundary'] * 0.6:
        print("CONCLUSION: Far-field coupling has MODERATE rank")
        print(f"  {svd['rank_99pct_energy']} modes capture 99% of energy")
    else:
        print("CONCLUSION: Far-field coupling is HIGH-RANK (limited compression)")
        print(f"  {svd['rank_99pct_energy']} modes needed for 99% energy")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze far-field to local boundary coupling in PDN grids',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic analysis
  python scripts/analyze_farfield_coupling.py --netlist ./pdn/netlist_test --net VDD

  # With validation against flat solve
  python scripts/analyze_farfield_coupling.py --netlist ./pdn/netlist_test --net VDD --validate

  # Sweep window positions
  python scripts/analyze_farfield_coupling.py --netlist ./pdn/netlist_test --net VDD --sweep-position

  # Sweep window sizes
  python scripts/analyze_farfield_coupling.py --netlist ./pdn/netlist_test --net VDD --sweep-size

  # Distance ring analysis
  python scripts/analyze_farfield_coupling.py --netlist ./pdn/netlist_test --net VDD --distance-rings
        """
    )

    parser.add_argument('--netlist', required=True, help='Path to PDN netlist directory')
    parser.add_argument('--net', default='VDD', help='Power net name (default: VDD)')
    parser.add_argument('--partition-layer', default='M2', help='Partition layer (default: M2)')

    parser.add_argument('--window-position', default='center',
                       choices=['center', 'corner_ll', 'corner_ur', 'corner_lr', 'corner_ul'],
                       help='Window position (default: center)')
    parser.add_argument('--window-fraction', type=float, default=0.25,
                       help='Window area as fraction of grid (default: 0.25)')

    parser.add_argument('--n-blocks-x', type=int, default=3, help='Far-field blocks in x (default: 3)')
    parser.add_argument('--n-blocks-y', type=int, default=3, help='Far-field blocks in y (default: 3)')
    parser.add_argument('--n-random', type=int, default=3, help='Random patterns per block (default: 3)')
    parser.add_argument('--total-current', type=float, default=1.0, help='Total current per pattern in mA (default: 1.0)')

    parser.add_argument('--validate', action='store_true', help='Validate against flat solve')
    parser.add_argument('--sweep-position', action='store_true', help='Sweep window positions')
    parser.add_argument('--sweep-size', action='store_true', help='Sweep window sizes')
    parser.add_argument('--distance-rings', action='store_true', help='Analyze by distance rings')
    parser.add_argument('--n-rings', type=int, default=4, help='Number of distance rings (default: 4)')

    parser.add_argument('--output-dir', default='./farfield_analysis_output',
                       help='Output directory for plots (default: ./farfield_analysis_output)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    parser.add_argument('--quiet', action='store_true', help='Suppress verbose output')

    args = parser.parse_args()

    verbose = not args.quiet

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load netlist
    if verbose:
        print(f"Loading netlist from {args.netlist}...")

    parser_obj = NetlistParser(args.netlist, validate=True)
    graph = parser_obj.parse()
    model = create_model_from_pdn(graph, args.net)
    solver = UnifiedIRDropSolver(model)

    if verbose:
        print(f"Model: {len(model.graph)} nodes, vdd={model.vdd}V")

    # Run analysis
    if args.sweep_position:
        if verbose:
            print("\nRunning position sweep...")

        results = run_position_sweep(
            solver, args.partition_layer,
            window_fraction=args.window_fraction,
            n_blocks_x=args.n_blocks_x,
            n_blocks_y=args.n_blocks_y,
            n_random_patterns=args.n_random,
            total_current=args.total_current,
            seed=args.seed,
            verbose=verbose,
        )

        plot_position_sweep_comparison(results, output_dir / 'position_sweep.png')

        # Print summary for each position
        for pos, res in results.items():
            if res is not None:
                print(f"\n--- {pos} ---")
                print(f"  |B|={res['svd']['n_boundary']}, "
                      f"rank(1%)={res['svd']['effective_rank_1pct']}, "
                      f"rank(99%)={res['svd']['rank_99pct_energy']}")

    elif args.sweep_size:
        if verbose:
            print("\nRunning size sweep...")

        results = run_size_sweep(
            solver, args.partition_layer,
            window_position=args.window_position,
            n_blocks_x=args.n_blocks_x,
            n_blocks_y=args.n_blocks_y,
            n_random_patterns=args.n_random,
            total_current=args.total_current,
            seed=args.seed,
            verbose=verbose,
        )

        plot_size_sweep_scaling(results, output_dir / 'size_sweep.png')

        # Print summary
        print("\n" + "=" * 60)
        print("SIZE SWEEP SUMMARY")
        print("=" * 60)
        print(f"{'Fraction':>10} {'|B|':>8} {'Rank(1%)':>10} {'Rank(99%)':>10} {'Compression':>12}")
        for frac, res in sorted(results.items()):
            if res is not None:
                print(f"{frac*100:>10.0f}% {res['svd']['n_boundary']:>8} "
                      f"{res['svd']['effective_rank_1pct']:>10} "
                      f"{res['svd']['rank_99pct_energy']:>10} "
                      f"{res['svd']['compression_ratio_99pct']:>12.1f}x")

    elif args.distance_rings:
        if verbose:
            print("\nRunning distance ring analysis...")

        results = run_distance_analysis(
            solver, args.partition_layer,
            window_position=args.window_position,
            window_fraction=args.window_fraction,
            n_rings=args.n_rings,
            n_random_patterns=args.n_random,
            total_current=args.total_current,
            seed=args.seed,
            verbose=verbose,
        )

        plot_distance_analysis(results, output_dir / 'distance_rings.png')

        # Print summary
        print("\n" + "=" * 60)
        print("DISTANCE RING SUMMARY")
        print("=" * 60)
        print(f"{'Ring':>6} {'Nodes':>8} {'Rank(1%)':>10} {'Rank(99%)':>10}")
        for i, ring_res in enumerate(results['rings']):
            if ring_res is not None:
                print(f"{i:>6} {results['ring_sizes'][i]:>8} "
                      f"{ring_res['svd']['effective_rank_1pct']:>10} "
                      f"{ring_res['svd']['rank_99pct_energy']:>10}")

    else:
        # Single analysis
        if verbose:
            print("\nRunning single analysis...")

        results = run_farfield_analysis(
            solver, args.partition_layer,
            window_position=args.window_position,
            window_fraction=args.window_fraction,
            n_blocks_x=args.n_blocks_x,
            n_blocks_y=args.n_blocks_y,
            n_random_patterns=args.n_random,
            total_current=args.total_current,
            seed=args.seed,
            validate=args.validate,
            verbose=verbose,
        )

        # Generate plots
        plot_setup_visualization(results, output_dir / 'setup.png')
        plot_singular_values(results, output_dir / 'singular_values.png')
        plot_dominant_modes(results, n_modes=6, output_path=output_dir / 'dominant_modes.png')

        # Print summary
        print_summary(results)

    print(f"\nPlots saved to: {output_dir}/")


if __name__ == '__main__':
    main()

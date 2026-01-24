"""Far-field to local boundary coupling analysis.

This module provides utilities to analyze whether far-field switching influences
a local boundary region through a small number of smooth/low-frequency spatial patterns.

Key functions:
- define_window_and_boundary: Define rectangular window W and boundary set B
- partition_farfield_into_blocks: Partition M1 nodes outside W into K blocks
- generate_block_injections: Create current injection patterns for each block
- compute_boundary_response_matrix: Solve coupled system and extract boundary voltages
- analyze_response_matrix: SVD analysis of the response matrix H
- validate_against_flat_solve: Compare coupled solver against flat solve
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from scipy.spatial import KDTree

from .tiling import TileManager
from .unified_solver import UnifiedIRDropSolver


def define_window_and_boundary(
    port_nodes: Set[Any],
    port_coords: Dict[Any, Tuple[float, float]],
    window_bounds: Tuple[float, float, float, float],
) -> Tuple[Set[Any], List[Any]]:
    """Define a rectangular locality window W and boundary set B.

    B = ALL ports inside the window W (not just perimeter).

    Args:
        port_nodes: Set of all port node IDs
        port_coords: Dict mapping port node -> (x, y) coordinates
        window_bounds: (x_min, x_max, y_min, y_max) defining window W

    Returns:
        boundary_ports: Set of ALL ports inside window W (this is B)
        boundary_port_list: Ordered list for indexing into H matrix
    """
    x_min, x_max, y_min, y_max = window_bounds

    boundary_ports = set()

    for p in port_nodes:
        if p not in port_coords:
            continue
        x, y = port_coords[p]

        # B = all ports inside window W
        if x_min <= x <= x_max and y_min <= y <= y_max:
            boundary_ports.add(p)

    # Sort for deterministic ordering
    boundary_port_list = sorted(boundary_ports, key=lambda p: (port_coords[p][0], port_coords[p][1]))

    return boundary_ports, boundary_port_list


def compute_window_bounds(
    port_coords: Dict[Any, Tuple[float, float]],
    grid_bounds: Tuple[float, float, float, float],
    position: str = 'center',
    window_fraction: float = 0.25,
) -> Tuple[float, float, float, float]:
    """Compute window bounds for a given position and size.

    Args:
        port_coords: Dict mapping port node -> (x, y) coordinates
        grid_bounds: (x_min, x_max, y_min, y_max) of the entire grid
        position: Window position - 'center', 'corner_ll', 'corner_ur', 'corner_lr', 'corner_ul'
        window_fraction: Window size as fraction of grid dimensions (0.0 to 1.0)

    Returns:
        (x_min, x_max, y_min, y_max) for the window
    """
    gx_min, gx_max, gy_min, gy_max = grid_bounds
    grid_width = gx_max - gx_min
    grid_height = gy_max - gy_min

    window_width = grid_width * np.sqrt(window_fraction)
    window_height = grid_height * np.sqrt(window_fraction)

    if position == 'center':
        cx = (gx_min + gx_max) / 2
        cy = (gy_min + gy_max) / 2
    elif position == 'corner_ll':  # lower-left
        cx = gx_min + window_width / 2
        cy = gy_min + window_height / 2
    elif position == 'corner_ur':  # upper-right
        cx = gx_max - window_width / 2
        cy = gy_max - window_height / 2
    elif position == 'corner_lr':  # lower-right
        cx = gx_max - window_width / 2
        cy = gy_min + window_height / 2
    elif position == 'corner_ul':  # upper-left
        cx = gx_min + window_width / 2
        cy = gy_max - window_height / 2
    else:
        raise ValueError(f"Unknown position: {position}")

    return (
        cx - window_width / 2,
        cx + window_width / 2,
        cy - window_height / 2,
        cy + window_height / 2,
    )


def partition_farfield_into_blocks(
    tile_manager: TileManager,
    bottom_nodes: Set[Any],
    bottom_coords: Dict[Any, Tuple[float, float]],
    window_bounds: Tuple[float, float, float, float],
    grid_bounds: Tuple[float, float, float, float],
    n_blocks_x: int,
    n_blocks_y: int,
) -> Tuple[List[Set[Any]], List[Tuple[float, float]]]:
    """Partition M1 nodes outside the window into K blocks.

    Reuses TileManager.generate_uniform_tile_bounds() for tiling.

    Args:
        tile_manager: TileManager instance
        bottom_nodes: Set of all bottom-grid (M1) nodes
        bottom_coords: Dict mapping node -> (x, y) coordinates
        window_bounds: (x_min, x_max, y_min, y_max) defining window W
        grid_bounds: (x_min, x_max, y_min, y_max) of entire grid
        n_blocks_x: Number of blocks in x direction
        n_blocks_y: Number of blocks in y direction

    Returns:
        blocks: List of sets, each containing M1 node IDs for that block
        block_centers: List of (cx, cy) for each block
    """
    # Generate uniform tile bounds for entire grid
    tile_bounds = tile_manager.generate_uniform_tile_bounds(
        n_blocks_x, n_blocks_y, grid_bounds
    )

    wx_min, wx_max, wy_min, wy_max = window_bounds

    # Assign bottom-grid nodes to tiles, excluding those inside window
    blocks = []
    block_centers = []

    for tile in tile_bounds:
        block_nodes = set()
        for node in bottom_nodes:
            if node not in bottom_coords:
                continue
            x, y = bottom_coords[node]

            # Skip nodes inside window W
            if wx_min <= x <= wx_max and wy_min <= y <= wy_max:
                continue

            # Check if in this tile
            if tile.x_min <= x <= tile.x_max and tile.y_min <= y <= tile.y_max:
                block_nodes.add(node)

        if block_nodes:  # Only add non-empty blocks
            blocks.append(block_nodes)
            block_centers.append((
                (tile.x_min + tile.x_max) / 2,
                (tile.y_min + tile.y_max) / 2,
            ))

    return blocks, block_centers


def partition_farfield_by_distance_rings(
    bottom_nodes: Set[Any],
    bottom_coords: Dict[Any, Tuple[float, float]],
    window_bounds: Tuple[float, float, float, float],
    grid_bounds: Tuple[float, float, float, float],
    n_rings: int = 4,
) -> List[Set[Any]]:
    """Partition far-field into concentric rings by distance from window.

    Args:
        bottom_nodes: Set of all bottom-grid nodes
        bottom_coords: Dict mapping node -> (x, y) coordinates
        window_bounds: (x_min, x_max, y_min, y_max) defining window W
        grid_bounds: (x_min, x_max, y_min, y_max) of entire grid
        n_rings: Number of distance rings

    Returns:
        rings: List of n_rings sets, ring[0] is closest to window
    """
    wx_min, wx_max, wy_min, wy_max = window_bounds
    gx_min, gx_max, gy_min, gy_max = grid_bounds

    # Compute max distance from window edge to grid edge
    max_dist = max(
        wx_min - gx_min,
        gx_max - wx_max,
        wy_min - gy_min,
        gy_max - wy_max,
    )

    ring_width = max_dist / n_rings

    def distance_to_window(x: float, y: float) -> float:
        """Compute distance from point to window boundary."""
        # Distance to nearest window edge
        dx = max(wx_min - x, 0, x - wx_max)
        dy = max(wy_min - y, 0, y - wy_max)
        return np.sqrt(dx * dx + dy * dy)

    rings: List[Set[Any]] = [set() for _ in range(n_rings)]

    for node in bottom_nodes:
        if node not in bottom_coords:
            continue
        x, y = bottom_coords[node]

        # Skip nodes inside window
        if wx_min <= x <= wx_max and wy_min <= y <= wy_max:
            continue

        dist = distance_to_window(x, y)
        ring_idx = min(int(dist / ring_width), n_rings - 1)
        rings[ring_idx].add(node)

    return rings


def generate_block_injections(
    blocks: List[Set[Any]],
    n_random_patterns: int = 3,
    total_current: float = 1.0,
    seed: Optional[int] = None,
) -> Tuple[List[Dict[Any, float]], List[int]]:
    """Generate injection patterns for each far-field block.

    Patterns per block:
    - Uniform: total_current distributed equally among block's M1 nodes
    - Random: n_random_patterns with random weights (normalized to total_current)

    Args:
        blocks: List of sets, each containing node IDs for a block
        n_random_patterns: Number of random patterns per block
        total_current: Total current to inject (mA)
        seed: Random seed for reproducibility

    Returns:
        patterns: List of current injection dicts {node: current_mA}
        block_indices: List indicating which block each pattern belongs to
    """
    rng = np.random.default_rng(seed)
    patterns = []
    block_indices = []

    for block_idx, block_nodes in enumerate(blocks):
        block_list = list(block_nodes)
        n_block = len(block_list)
        if n_block == 0:
            continue

        # Uniform injection
        uniform_current = total_current / n_block
        patterns.append({node: uniform_current for node in block_list})
        block_indices.append(block_idx)

        # Random injections
        for _ in range(n_random_patterns):
            weights = rng.random(n_block)
            weights /= weights.sum()
            patterns.append({
                node: total_current * w
                for node, w in zip(block_list, weights)
            })
            block_indices.append(block_idx)

    return patterns, block_indices


def compute_boundary_response_matrix(
    solver: UnifiedIRDropSolver,
    injection_patterns: List[Dict[Any, float]],
    boundary_ports: List[Any],
    partition_layer: str,
    verbose: bool = False,
    context: Optional[Any] = None,
) -> np.ndarray:
    """Solve coupled system for each injection pattern and extract boundary voltages.

    Uses batched solving with prepare/solve_prepared for efficiency.

    Args:
        solver: UnifiedIRDropSolver instance
        injection_patterns: List of {node: current} dicts
        boundary_ports: Ordered list of boundary port nodes
        partition_layer: Layer for hierarchical decomposition
        verbose: Print progress
        context: Optional pre-computed CoupledHierarchicalSolverContext.
                 If None, will be created and cached internally.

    Returns:
        H: Boundary response matrix (|B| × n_patterns)
           H[:, k] = v_p^{(k)}|_B = boundary voltages for pattern k
    """
    n_patterns = len(injection_patterns)
    n_boundary = len(boundary_ports)
    H = np.zeros((n_boundary, n_patterns))

    # Prepare context once for efficient batch solving
    if context is None:
        context = solver.prepare_hierarchical_coupled(
            partition_layer=partition_layer,
            solver='gmres',
            tol=1e-8,
            maxiter=500,
            preconditioner='block_diagonal',
        )

    for k, currents in enumerate(injection_patterns):
        if verbose:
            print(f"  Solving pattern {k+1}/{n_patterns}...")

        # Solve using prepared context (reuses LU factorizations and operators)
        result = solver.solve_hierarchical_coupled_prepared(
            current_injections=currents,
            context=context,
            verbose=False,
        )

        # Extract boundary port voltages
        for i, port in enumerate(boundary_ports):
            if hasattr(result, 'port_voltages') and port in result.port_voltages:
                H[i, k] = result.port_voltages[port]
            else:
                H[i, k] = result.voltages.get(port, 0.0)

    return H


def compute_boundary_smoothness(
    u: np.ndarray,
    coords: np.ndarray,
) -> Dict[str, float]:
    """Compute 2D smoothness metrics for a voltage pattern over window ports.

    Args:
        u: Voltage pattern vector (|B|,)
        coords: Port coordinates (|B|, 2) inside window W

    Returns:
        Dict with smoothness metrics:
        - gradient_energy: ||∇u||₂ / std(u) (lower = smoother)
        - total_variation_2d: Sum of |u_i - u_j| for k-nearest neighbors
    """
    if len(u) < 2:
        return {
            'total_variation_2d': 0.0,
            'gradient_energy': 0.0,
        }

    # Build KD-tree for nearest neighbor queries
    tree = KDTree(coords)

    # Compute gradient-like metric using k nearest neighbors
    k = min(6, len(u) - 1)  # 6-connectivity approximation
    distances, indices = tree.query(coords, k=k+1)  # +1 because self is included

    # Total variation: sum of |u_i - u_j| for neighbors
    tv = 0.0
    for i in range(len(u)):
        for j_idx in range(1, k+1):  # Skip self (index 0)
            j = indices[i, j_idx]
            tv += abs(u[i] - u[j])
    tv /= 2  # Each edge counted twice

    # Gradient energy: RMS of local differences normalized by distance
    grad_sq_sum = 0.0
    count = 0
    for i in range(len(u)):
        for j_idx in range(1, k+1):
            j = indices[i, j_idx]
            d = distances[i, j_idx]
            if d > 0:
                grad_sq_sum += ((u[i] - u[j]) / d) ** 2
                count += 1

    u_std = np.std(u)
    if u_std < 1e-12:
        gradient_energy = 0.0
    else:
        gradient_energy = np.sqrt(grad_sq_sum / max(count, 1)) / u_std

    return {
        'total_variation_2d': float(tv),
        'gradient_energy': float(gradient_energy),
    }


def analyze_response_matrix(
    H: np.ndarray,
    boundary_ports: List[Any],
    port_coords: Dict[Any, Tuple[float, float]],
) -> Dict[str, Any]:
    """Compute SVD of H and analyze the spectral structure.

    Args:
        H: Boundary response matrix (|B| × n_patterns)
        boundary_ports: Ordered list of boundary port nodes
        port_coords: Dict mapping port node -> (x, y)

    Returns:
        results: Dict containing singular values, ranks, and smoothness metrics
    """
    U, sigma, Vt = np.linalg.svd(H, full_matrices=False)

    # Handle edge case of zero matrix
    if len(sigma) == 0 or sigma[0] == 0:
        return {
            'singular_values': sigma,
            'effective_rank_1pct': 0,
            'effective_rank_01pct': 0,
            'rank_90pct_energy': 0,
            'rank_99pct_energy': 0,
            'error': 'All singular values are zero',
        }

    # Effective rank at various thresholds
    effective_rank_1pct = int(np.sum(sigma > 0.01 * sigma[0]))
    effective_rank_01pct = int(np.sum(sigma > 0.001 * sigma[0]))

    # Cumulative energy
    energy = sigma ** 2
    total_energy = np.sum(energy)
    cumulative_energy = np.cumsum(energy) / total_energy

    rank_90pct = int(np.searchsorted(cumulative_energy, 0.90)) + 1
    rank_95pct = int(np.searchsorted(cumulative_energy, 0.95)) + 1
    rank_99pct = int(np.searchsorted(cumulative_energy, 0.99)) + 1

    # Analyze smoothness of top singular vectors (boundary patterns)
    boundary_coords_array = np.array([port_coords[p] for p in boundary_ports])
    smoothness_scores = []
    for i in range(min(10, len(sigma))):
        smoothness = compute_boundary_smoothness(U[:, i], boundary_coords_array)
        smoothness_scores.append(smoothness)

    # Compute decay rate (linear fit on log scale)
    if len(sigma) > 1:
        log_sigma = np.log10(sigma / sigma[0] + 1e-15)
        indices = np.arange(len(sigma))
        # Linear regression: log(sigma) = a * k + b
        coeffs = np.polyfit(indices, log_sigma, 1)
        decay_rate = -coeffs[0]  # Negative slope = decay rate
    else:
        decay_rate = 0.0

    return {
        'singular_values': sigma,
        'U': U,
        'Vt': Vt,
        'effective_rank_1pct': effective_rank_1pct,
        'effective_rank_01pct': effective_rank_01pct,
        'rank_90pct_energy': rank_90pct,
        'rank_95pct_energy': rank_95pct,
        'rank_99pct_energy': rank_99pct,
        'cumulative_energy': cumulative_energy,
        'smoothness_scores': smoothness_scores,
        'decay_rate': decay_rate,
        'n_boundary': len(boundary_ports),
        'n_patterns': H.shape[1],
        'compression_ratio_1pct': len(boundary_ports) / max(effective_rank_1pct, 1),
        'compression_ratio_99pct': len(boundary_ports) / max(rank_99pct, 1),
    }


def validate_against_flat_solve(
    solver: UnifiedIRDropSolver,
    injection_patterns: List[Dict[Any, float]],
    boundary_ports: List[Any],
    H_coupled: np.ndarray,
    verbose: bool = False,
    context: Optional[Any] = None,
) -> Dict[str, Any]:
    """Validate coupled solver results against flat (non-hierarchical) solve.

    Uses batched flat solving with prepare/solve_prepared for efficiency.

    Args:
        solver: UnifiedIRDropSolver instance
        injection_patterns: List of {node: current} dicts
        boundary_ports: Ordered list of boundary port nodes
        H_coupled: Response matrix from coupled solver
        verbose: Print progress
        context: Optional pre-computed FlatSolverContext.
                 If None, will be created and cached internally.

    Returns:
        Dict with max_diff, mean_diff, rmse between coupled and flat solutions
    """
    n_patterns = len(injection_patterns)
    H_flat = np.zeros_like(H_coupled)

    # Prepare context once for efficient batch solving
    if context is None:
        context = solver.prepare_flat()

    for k, currents in enumerate(injection_patterns):
        if verbose:
            print(f"  Flat solve pattern {k+1}/{n_patterns}...")

        # Flat solve using prepared context (reuses LU factorization)
        result_flat = solver.solve_prepared(currents, context)

        # Extract boundary port voltages
        for i, port in enumerate(boundary_ports):
            H_flat[i, k] = result_flat.voltages.get(port, 0.0)

    # Compare
    diff = np.abs(H_coupled - H_flat)

    return {
        'max_diff': float(np.max(diff)),
        'mean_diff': float(np.mean(diff)),
        'rmse': float(np.sqrt(np.mean(diff ** 2))),
        'max_diff_mV': float(np.max(diff) * 1000),
        'mean_diff_mV': float(np.mean(diff) * 1000),
        'H_flat': H_flat,
    }


def run_farfield_analysis(
    solver: UnifiedIRDropSolver,
    partition_layer: str,
    window_position: str = 'center',
    window_fraction: float = 0.25,
    n_blocks_x: int = 3,
    n_blocks_y: int = 3,
    n_random_patterns: int = 3,
    total_current: float = 1.0,
    seed: Optional[int] = 42,
    validate: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run complete far-field coupling analysis.

    This is the main entry point for the analysis.

    Args:
        solver: UnifiedIRDropSolver instance
        partition_layer: Layer for hierarchical decomposition (e.g., 'M2')
        window_position: 'center', 'corner_ll', 'corner_ur', etc.
        window_fraction: Window area as fraction of grid area
        n_blocks_x: Number of far-field blocks in x direction
        n_blocks_y: Number of far-field blocks in y direction
        n_random_patterns: Random patterns per block
        total_current: Current per injection (mA)
        seed: Random seed
        validate: If True, compare against flat solve
        verbose: Print progress

    Returns:
        Dict with analysis results, including SVD metrics
    """
    model = solver.model

    if verbose:
        print("Setting up analysis...")

    # Decompose at partition layer to get port nodes
    top_nodes, bottom_nodes, port_nodes, _ = model._decompose_at_layer(partition_layer)

    if verbose:
        print(f"  Partition layer: {partition_layer}")
        print(f"  Top nodes: {len(top_nodes)}, Bottom nodes: {len(bottom_nodes)}, Ports: {len(port_nodes)}")

    # Use TileManager to extract coordinates
    tile_manager = TileManager(model)
    bottom_coords, port_coords, grid_bounds = tile_manager.extract_bottom_grid_coordinates(
        bottom_nodes, port_nodes
    )

    if verbose:
        print(f"  Grid bounds: x=[{grid_bounds[0]:.0f}, {grid_bounds[1]:.0f}], y=[{grid_bounds[2]:.0f}, {grid_bounds[3]:.0f}]")

    # Compute window bounds
    window_bounds = compute_window_bounds(port_coords, grid_bounds, window_position, window_fraction)

    if verbose:
        print(f"  Window ({window_position}, {window_fraction*100:.0f}%): "
              f"x=[{window_bounds[0]:.0f}, {window_bounds[1]:.0f}], "
              f"y=[{window_bounds[2]:.0f}, {window_bounds[3]:.0f}]")

    # Define boundary set B (all ports in window)
    boundary_ports_set, boundary_ports = define_window_and_boundary(
        port_nodes, port_coords, window_bounds
    )

    if verbose:
        print(f"  Boundary ports |B|: {len(boundary_ports)}")

    if len(boundary_ports) == 0:
        raise ValueError("No ports found inside window. Try larger window_fraction or different position.")

    # Partition far-field into blocks
    blocks, block_centers = partition_farfield_into_blocks(
        tile_manager, bottom_nodes, bottom_coords,
        window_bounds, grid_bounds, n_blocks_x, n_blocks_y
    )

    if verbose:
        print(f"  Far-field blocks: {len(blocks)}")
        for i, b in enumerate(blocks):
            print(f"    Block {i}: {len(b)} nodes")

    if len(blocks) == 0:
        raise ValueError("No far-field blocks found. Window may cover entire grid.")

    # Generate injection patterns
    patterns, block_indices = generate_block_injections(
        blocks, n_random_patterns, total_current, seed
    )

    if verbose:
        print(f"  Total injection patterns: {len(patterns)}")

    # Prepare solver contexts for batch solving
    if verbose:
        print("Preparing solver contexts...")

    coupled_context = solver.prepare_hierarchical_coupled(
        partition_layer=partition_layer,
        solver='gmres',
        tol=1e-8,
        maxiter=500,
        preconditioner='block_diagonal',
    )

    # Compute boundary response matrix using batch solving
    if verbose:
        print("Computing boundary response matrix H...")

    H = compute_boundary_response_matrix(
        solver, patterns, boundary_ports, partition_layer, verbose,
        context=coupled_context
    )

    if verbose:
        print(f"  H shape: {H.shape}")

    # SVD analysis
    if verbose:
        print("Analyzing response matrix (SVD)...")

    svd_results = analyze_response_matrix(H, boundary_ports, port_coords)

    if verbose:
        print(f"  Singular values (top 5): {svd_results['singular_values'][:5]}")
        print(f"  Effective rank (1%): {svd_results['effective_rank_1pct']}")
        print(f"  Rank for 99% energy: {svd_results['rank_99pct_energy']}")
        print(f"  Compression ratio (99%): {svd_results['compression_ratio_99pct']:.1f}x")
        print(f"  Decay rate: {svd_results['decay_rate']:.3f}")

    # Optional validation using batch flat solving
    validation_results = None
    if validate:
        if verbose:
            print("Validating against flat solve...")

        flat_context = solver.prepare_flat()
        validation_results = validate_against_flat_solve(
            solver, patterns, boundary_ports, H, verbose,
            context=flat_context
        )

        if verbose:
            print(f"  Max diff vs flat: {validation_results['max_diff_mV']:.3f} mV")
            print(f"  RMSE vs flat: {validation_results['rmse']*1000:.3f} mV")

    return {
        'H': H,
        'boundary_ports': boundary_ports,
        'port_coords': port_coords,
        'window_bounds': window_bounds,
        'grid_bounds': grid_bounds,
        'blocks': blocks,
        'block_centers': block_centers,
        'block_indices': block_indices,
        'patterns': patterns,
        'svd': svd_results,
        'validation': validation_results,
        'config': {
            'partition_layer': partition_layer,
            'window_position': window_position,
            'window_fraction': window_fraction,
            'n_blocks_x': n_blocks_x,
            'n_blocks_y': n_blocks_y,
            'n_random_patterns': n_random_patterns,
            'total_current': total_current,
            'seed': seed,
        },
    }
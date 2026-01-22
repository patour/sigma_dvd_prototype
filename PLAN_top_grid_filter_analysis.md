# Plan: Validate Far-Field → Local Boundary Coupling is Low-Rank

## Objective

Determine whether **far-field switching** influences a **local boundary region** through a small number of smooth/low-frequency spatial patterns.

**Core Question**: Given a locality window W with boundary ports B, does the matrix H of far-field block responses (restricted to B) have low rank?

---

## Background: Coupled System Formulation

### Using the Existing Coupled Hierarchical Solver

Instead of explicitly computing the coupled system operator `L_c`, we use the existing `solve_hierarchical_coupled()` method which already computes port voltages `v_p`:

```python
result = solver.solve_hierarchical_coupled(
    current_injections=s_k,  # Inject currents into far-field block k
    partition_layer='M2',
    ...
)
v_p = result.port_voltages  # Extract port voltages directly
```

This avoids building L_c explicitly and reuses the well-tested coupled solver infrastructure.

### Setup

```
┌─────────────────────────────────────────────────┐
│                   Port Grid                      │
│                                                  │
│   ┌───────────────┐                              │
│   │   Window W    │  ← Near-field (local region)│
│   │  ● ● ● ● ●    │                              │
│   │  ● ● ● ● ●    │  ← B = ALL ports inside W   │
│   │  ● ● ● ● ●    │    (not just perimeter)     │
│   │  ● ● ● ● ●    │                              │
│   └───────────────┘                              │
│                                                  │
│   [Block 1] [Block 2] ... [Block K]             │
│        ↑ Far-field blocks (tiles outside W)     │
└─────────────────────────────────────────────────┘
```

### Methodology

1. **Define locality window W**: A 2D rectangular region (reuse `TileManager` tiling)
2. **Define boundary set B**: ALL port nodes inside W (the full window, not just perimeter)
3. **Partition far-field into K blocks**: Use `TileManager.generate_uniform_tile_bounds()` for blocks outside W
4. **For each block k**: Create injection pattern `s_k` (uniform or random distribution to M1 nodes in block)
5. **Solve via coupled solver**: `result = solver.solve_hierarchical_coupled(s_k, ...)`
6. **Extract boundary response**: `h_k = {result.port_voltages[p] for p in B}`
7. **Stack responses**: `H = [h_1, h_2, ..., h_K]` where H is |B| × K
8. **Compute SVD(H)**: Analyze singular value decay
9. **(Optional) Validate**: Compare against flat solve using existing `TileManager.validate_tiled_accuracy()`

### Hypothesis

If H has **low numerical rank**, then:
- Far-field influences on the local boundary B can be represented by few smooth patterns
- The dominant left singular vectors of H are the "principal boundary voltage modes"
- Hierarchical/tile-based approximations are justified

---

## Validation Plan

### Phase 1: Infrastructure Setup

#### Task 1.1: Create Analysis Module
Create `core/farfield_analysis.py` with utilities for:
- Reusing `TileManager` for coordinate extraction and tiling
- Defining locality windows and extracting boundary port sets
- Generating injection patterns (uniform, random) for M1 nodes in far-field blocks
- Running coupled hierarchical solver and extracting port voltages
- Computing and analyzing SVD of the response matrix H
- Optional validation against flat solver

#### Task 1.2: Create Analysis Script
Create `scripts/analyze_farfield_coupling.py` for running experiments:
- CLI interface with configurable window size, block count, injection types
- PDN netlist input only (no synthetic grid support needed)
- Output plots and numerical metrics
- Save results for reproducibility

---

### Phase 2: Far-Field Block Response Analysis

#### Task 2.1: Reuse TileManager for Setup

Leverage existing `TileManager` infrastructure for coordinate extraction and tiling:

```python
from core.tiling import TileManager
from core import create_model_from_pdn, UnifiedIRDropSolver

# Create model and solver
model = create_model_from_pdn(graph, 'VDD')
solver = UnifiedIRDropSolver(model)

# Decompose at partition layer to get port nodes
top_nodes, bottom_nodes, port_nodes, _ = model._decompose_at_layer(partition_layer)

# Use TileManager to extract coordinates
tile_manager = TileManager(model)
bottom_coords, port_coords, grid_bounds = tile_manager.extract_bottom_grid_coordinates(
    bottom_nodes, port_nodes
)
```

#### Task 2.2: Define Locality Window W and Boundary Set B

```python
def define_window_and_boundary(
    port_nodes: Set[Any],
    port_coords: Dict[Any, Tuple[float, float]],
    window_bounds: Tuple[float, float, float, float],  # (x_min, x_max, y_min, y_max)
) -> Tuple[Set[Any], List[Any]]:
    """
    Define a rectangular locality window W and boundary set B.

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

    return boundary_ports, sorted(boundary_ports)
```

#### Task 2.3: Partition Far-Field into K Blocks (Reuse TileManager)

```python
def partition_farfield_into_blocks(
    tile_manager: TileManager,
    bottom_nodes: Set[Any],
    bottom_coords: Dict[Any, Tuple[float, float]],
    window_bounds: Tuple[float, float, float, float],
    grid_bounds: Tuple[float, float, float, float],
    n_blocks_x: int,
    n_blocks_y: int,
) -> List[Set[Any]]:
    """
    Partition M1 nodes outside the window into K blocks.

    Reuses TileManager.generate_uniform_tile_bounds() for the far-field region.

    Returns:
        blocks: List of K sets, each containing M1 node IDs for that block
    """
    # Compute far-field bounds (exclude window region)
    # This may require multiple rectangular regions around the window
    # For simplicity: use full grid and filter out window nodes later

    # Generate uniform tile bounds for entire grid
    tile_bounds = tile_manager.generate_uniform_tile_bounds(
        n_blocks_x, n_blocks_y, grid_bounds
    )

    # Assign bottom-grid nodes to tiles
    blocks = []
    for tile in tile_bounds:
        block_nodes = set()
        for node, (x, y) in bottom_coords.items():
            # Skip nodes inside window
            wx_min, wx_max, wy_min, wy_max = window_bounds
            if wx_min <= x <= wx_max and wy_min <= y <= wy_max:
                continue

            # Check if in this tile
            if tile.x_min <= x <= tile.x_max and tile.y_min <= y <= tile.y_max:
                block_nodes.add(node)

        if block_nodes:  # Only add non-empty blocks
            blocks.append(block_nodes)

    return blocks
```

#### Task 2.4: Generate Injection Patterns s_k

For each far-field block k, create injection patterns using M1 nodes:

```python
def generate_block_injections(
    blocks: List[Set[Any]],
    n_random_patterns: int = 3,
    total_current: float = 1.0,  # mA
    seed: Optional[int] = None,
) -> List[Dict[Any, float]]:
    """
    Generate injection patterns for each far-field block.

    Patterns per block:
    - Uniform: total_current distributed equally among block's M1 nodes
    - Random: n_random_patterns with random weights (normalized to total_current)

    Returns:
        patterns: List of current injection dicts {node: current_mA}
    """
    rng = np.random.default_rng(seed)
    patterns = []

    for block_nodes in blocks:
        block_list = list(block_nodes)
        n_block = len(block_list)
        if n_block == 0:
            continue

        # Uniform injection
        uniform_current = total_current / n_block
        patterns.append({node: uniform_current for node in block_list})

        # Random injections
        for _ in range(n_random_patterns):
            weights = rng.random(n_block)
            weights /= weights.sum()
            patterns.append({
                node: total_current * w
                for node, w in zip(block_list, weights)
            })

    return patterns
```

#### Task 2.5: Solve Using Coupled Hierarchical Solver

```python
def compute_boundary_response_matrix(
    solver: UnifiedIRDropSolver,
    injection_patterns: List[Dict[Any, float]],
    boundary_ports: List[Any],
    partition_layer: str,
    verbose: bool = False,
) -> np.ndarray:
    """
    Solve coupled system for each injection pattern and extract boundary voltages.

    Uses existing solve_hierarchical_coupled() - no explicit L_c needed.

    Args:
        solver: UnifiedIRDropSolver instance
        injection_patterns: List of {node: current} dicts
        boundary_ports: Ordered list of boundary port nodes
        partition_layer: Layer for hierarchical decomposition

    Returns:
        H: Boundary response matrix (|B| × n_patterns)
           H[:, k] = v_p^{(k)}|_B = boundary voltages for pattern k
    """
    n_patterns = len(injection_patterns)
    n_boundary = len(boundary_ports)
    H = np.zeros((n_boundary, n_patterns))

    for k, currents in enumerate(injection_patterns):
        if verbose:
            print(f"  Solving pattern {k+1}/{n_patterns}...")

        # Solve using coupled hierarchical solver
        result = solver.solve_hierarchical_coupled(
            current_injections=currents,
            partition_layer=partition_layer,
            solver='gmres',
            tol=1e-8,
            maxiter=500,
            preconditioner='block_diagonal',
            verbose=False,
        )

        # Extract boundary port voltages
        for i, port in enumerate(boundary_ports):
            H[i, k] = result.port_voltages.get(port, result.voltages.get(port, 0.0))

    return H
```

#### Task 2.6: SVD Analysis of Response Matrix H

```python
def analyze_response_matrix(
    H: np.ndarray,
    boundary_ports: List[Any],
    port_coords: Dict[Any, Tuple[float, float]],
) -> Dict[str, Any]:
    """
    Compute SVD of H and analyze the spectral structure.

    Returns:
        results: Dict containing singular values, ranks, and smoothness metrics
    """
    U, sigma, Vt = np.linalg.svd(H, full_matrices=False)

    # Handle edge case of zero matrix
    if sigma[0] == 0:
        return {
            'singular_values': sigma,
            'effective_rank_1pct': 0,
            'rank_99pct_energy': 0,
            'error': 'All singular values are zero',
        }

    # Effective rank at various thresholds
    effective_rank_1pct = int(np.sum(sigma > 0.01 * sigma[0]))
    effective_rank_01pct = int(np.sum(sigma > 0.001 * sigma[0]))

    # Cumulative energy
    energy = sigma ** 2
    cumulative_energy = np.cumsum(energy) / np.sum(energy)
    rank_90pct = int(np.searchsorted(cumulative_energy, 0.90)) + 1
    rank_99pct = int(np.searchsorted(cumulative_energy, 0.99)) + 1

    # Analyze smoothness of top singular vectors (boundary patterns)
    boundary_coords_array = np.array([port_coords[p] for p in boundary_ports])
    smoothness_scores = []
    for i in range(min(10, len(sigma))):
        smoothness = compute_boundary_smoothness(U[:, i], boundary_coords_array)
        smoothness_scores.append(smoothness)

    return {
        'singular_values': sigma,
        'U': U,
        'Vt': Vt,
        'effective_rank_1pct': effective_rank_1pct,
        'effective_rank_01pct': effective_rank_01pct,
        'rank_90pct_energy': rank_90pct,
        'rank_99pct_energy': rank_99pct,
        'cumulative_energy': cumulative_energy,
        'smoothness_scores': smoothness_scores,
        'n_boundary': len(boundary_ports),
        'n_patterns': H.shape[1],
    }
```

#### Task 2.7: Smoothness Analysis of Boundary Modes (2D)

Since B contains ALL ports inside the window W (a 2D region), we use 2D smoothness metrics:

```python
def compute_boundary_smoothness(
    u: np.ndarray,
    coords: np.ndarray,
) -> Dict[str, float]:
    """
    Compute 2D smoothness metrics for a voltage pattern over window ports.

    Args:
        u: Voltage pattern vector (|B|,)
        coords: Port coordinates (|B|, 2) inside window W

    Returns:
        Dict with smoothness metrics:
        - gradient_energy: ||∇u||₂ / ||u||₂ (lower = smoother)
        - total_variation_2d: Sum of |u_i - u_j| for k-nearest neighbors
    """
    from scipy.spatial import KDTree

    # Build KD-tree for nearest neighbor queries
    tree = KDTree(coords)

    # Compute gradient-like metric using k nearest neighbors
    k = min(6, len(u) - 1)  # 6-connectivity approximation
    distances, indices = tree.query(coords, k=k+1)  # +1 because self is included

    # Total variation: sum of |u_i - u_j| for neighbors
    tv = 0.0
    for i in range(len(u)):
        for j in indices[i, 1:]:  # Skip self (index 0)
            tv += abs(u[i] - u[j])
    tv /= 2  # Each edge counted twice

    # Gradient energy: RMS of local differences
    grad_sq_sum = 0.0
    count = 0
    for i in range(len(u)):
        for j in indices[i, 1:]:
            d = distances[i, list(indices[i]).index(j)]
            if d > 0:
                grad_sq_sum += ((u[i] - u[j]) / d) ** 2
                count += 1

    gradient_energy = np.sqrt(grad_sq_sum / max(count, 1)) / (np.std(u) + 1e-12)

    return {
        'total_variation_2d': float(tv),
        'gradient_energy': float(gradient_energy),
    }
```

#### Task 2.8: Optional Validation Against Flat Solver

```python
def validate_against_flat_solve(
    solver: UnifiedIRDropSolver,
    injection_patterns: List[Dict[Any, float]],
    boundary_ports: List[Any],
    H_coupled: np.ndarray,
) -> Dict[str, Any]:
    """
    Validate coupled solver results against flat (non-hierarchical) solve.

    Uses existing solver.solve() for ground truth comparison.

    Returns:
        Dict with max_diff, mean_diff, rmse between coupled and flat solutions
    """
    n_patterns = len(injection_patterns)
    H_flat = np.zeros_like(H_coupled)

    for k, currents in enumerate(injection_patterns):
        # Flat solve
        result_flat = solver.solve(currents)

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
        'H_flat': H_flat,
    }
```

---

### Phase 3: Numerical Experiments (PDN Netlists Only)

#### Task 3.1: PDN Netlist Experiments

Use real PDN grids from available netlists:
```python
from pdn_parser import NetlistParser
from core import create_model_from_pdn, UnifiedIRDropSolver

parser = NetlistParser('./pdn/netlist_test', validate=True)
graph = parser.parse()
model = create_model_from_pdn(graph, 'VDD')
solver = UnifiedIRDropSolver(model)
```

**Experiment matrix:**

| Netlist | Window Position | Window Size | # Blocks | # Random/Block | Expected Rank |
|---------|----------------|-------------|----------|----------------|---------------|
| netlist_test | center | 25% of grid | 8 (2×4) | 3 | TBD |
| netlist_test | corner | 25% of grid | 8 | 3 | TBD |
| netlist_test | near pad | 25% of grid | 8 | 3 | TBD |
| (larger netlist if available) | center | 10%, 25%, 50% | varies | 3-5 | TBD |

**Key factors to investigate:**
- Non-uniform grid spacing in real PDN
- Anisotropic resistances (metal layer orientation: H vs V)
- Irregular port placement
- Effect of pad proximity

#### Task 3.2: Window Position Sweep

Test how window position affects the low-rank structure:
```python
# Define multiple window positions
positions = [
    'center',      # Center of port grid
    'corner_ll',   # Lower-left corner
    'corner_ur',   # Upper-right corner
    'near_pad',    # Adjacent to a voltage source pad
    'edge',        # Along one edge
]

for position in positions:
    window_bounds = compute_window_bounds(port_coords, grid_bounds, position, window_fraction=0.25)
    # Run analysis
    # Compare effective rank across positions
```

**Question**: Is the rank position-dependent, or uniformly low across the grid?

#### Task 3.3: Window Size Scaling

Vary window size to understand scaling:
```python
window_fractions = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]

for fraction in window_fractions:
    window_bounds = compute_window_bounds(..., window_fraction=fraction)
    # Boundary size |B| grows with perimeter
    # Measure: effective rank vs |B|
```

**Questions:**
- Does effective rank grow with O(1), O(√|B|), or O(|B|)?
- What's the compression ratio (|B| / effective_rank)?

#### Task 3.4: Block Distance Analysis

For a fixed window, partition far-field into distance-based rings:
```python
def partition_by_distance_rings(
    bottom_coords, window_bounds, n_rings=4
) -> List[Set[Any]]:
    """Partition far-field into concentric rings by distance from window."""
    # Ring 0: immediately outside window
    # Ring 1: next band further out
    # ...
```

For each ring:
```python
# Inject only into ring k
# Measure H_k = boundary response
# Compute rank of H_k
# Compare: rank should decrease with distance
```

**Expected**: More distant rings contribute fewer effective modes (faster singular value decay)

#### Task 3.5: Validation Against Flat Solve

For each experiment configuration:
```python
validation = validate_against_flat_solve(solver, patterns, boundary_ports, H_coupled)

print(f"Max diff vs flat: {validation['max_diff_mV']:.3f} mV")
print(f"RMSE vs flat: {validation['rmse']*1000:.3f} mV")
```

**Purpose**: Ensure the coupled hierarchical solver is giving correct results (within tolerance)

---

### Phase 4: Visualization and Reporting

#### Task 4.1: Plots to Generate

1. **Singular value spectrum** (log scale)
   ```
   σ_k / σ_1 vs k
   ```
   - Mark effective rank thresholds (1%, 0.1%)
   - Compare different window sizes on same plot

2. **Cumulative energy plot**
   ```
   Σ_{i≤k} σ_i² / Σ_all σ_i² vs k
   ```
   - Mark 90%, 95%, 99% energy thresholds
   - Annotate rank at each threshold

3. **Dominant boundary voltage patterns** (2D heatmaps)
   - Top 4-6 left singular vectors U[:, k] mapped to port (x, y) coordinates in W
   - Visualize as 2D scatter/heatmap over the window region
   - Annotate with singular value magnitude
   - Expect: low-index modes are smooth (gradual spatial variation), high-index are oscillatory

4. **Smoothness analysis of modes**
   - For each dominant mode: plot pattern + FFT power spectrum
   - Show that low-index modes are smooth, high-index are oscillatory

5. **Rank vs distance plot**
   - Effective rank from blocks at distance d from window
   - Expected: decreasing rank with distance

6. **Window size scaling**
   - Plot effective rank vs window perimeter |B|
   - Expected: sublinear growth (rank << |B|)

7. **Comparison: uniform vs random injections**
   - Do random patterns reveal additional modes beyond uniform?
   - If not, uniform patterns suffice

#### Task 4.2: Summary Metrics Table

| Metric | Definition | How to Compute |
|--------|------------|----------------|
| Effective rank (ε=1%) | # modes with σ_k > 0.01·σ_1 | `sum(sigma > 0.01*sigma[0])` |
| Effective rank (ε=0.1%) | # modes with σ_k > 0.001·σ_1 | `sum(sigma > 0.001*sigma[0])` |
| Rank for 90% energy | # modes to capture 90% | `searchsorted(cumsum(σ²)/sum(σ²), 0.9) + 1` |
| Rank for 99% energy | # modes to capture 99% | `searchsorted(cumsum(σ²)/sum(σ²), 0.99) + 1` |
| Decay rate | Slope of log(σ_k) vs k | Linear fit on log scale |
| Compression ratio | |B| / effective_rank | Indicates potential speedup |
| Mode smoothness | Gradient energy of U[:, k] | `‖∇u‖₂ / std(u)` (2D, lower = smoother) |

#### Task 4.3: Summary Report

Create `docs/farfield_lowrank_analysis.md` with:

1. **Problem Statement**
   - Why we care about far-field → boundary coupling
   - Connection to hierarchical solver efficiency

2. **Mathematical Formulation**
   - Definition of L_c, window W, boundary B, blocks
   - The response matrix H and its SVD interpretation

3. **Experimental Results**
   - Synthetic grid results (tables + key plots)
   - PDN netlist results
   - Scaling behavior

4. **Key Findings**
   - Is far-field coupling low-rank? At what threshold?
   - How does rank scale with window size and distance?
   - Are the dominant modes smooth?

5. **Implications**
   - Justification for hierarchical approximations
   - Potential for low-rank compression in solver
   - Recommendations for tile/window sizing

---

## Implementation Order

```
Phase 1 (Infrastructure)
    ├── Task 1.1: core/farfield_analysis.py
    │     - define_window_and_boundary()
    │     - partition_farfield_into_blocks()  [reuses TileManager]
    │     - generate_block_injections()
    │     - compute_boundary_response_matrix()  [uses solve_hierarchical_coupled]
    │     - analyze_response_matrix()
    │     - compute_boundary_smoothness()
    │     - validate_against_flat_solve()
    │
    └── Task 1.2: scripts/analyze_farfield_coupling.py
          - CLI interface (argparse)
          - Experiment runner
          - Plotting utilities
          │
Phase 2 (Core Analysis) - Implemented in Phase 1 module
    ├── Task 2.1: Setup using TileManager
    ├── Task 2.2: Define window W and boundary B
    ├── Task 2.3: Partition far-field into K blocks
    ├── Task 2.4: Generate injection patterns s_k
    ├── Task 2.5: Solve via coupled solver, extract H matrix
    ├── Task 2.6: SVD analysis of H
    ├── Task 2.7: Smoothness analysis of modes
    └── Task 2.8: Optional validation against flat solve
          │
Phase 3 (Experiments) - PDN Netlists Only
    ├── Task 3.1: PDN netlist experiments
    ├── Task 3.2: Window position sweep
    ├── Task 3.3: Window size scaling
    ├── Task 3.4: Block distance analysis
    └── Task 3.5: Validation against flat solve
          │
Phase 4 (Reporting)
    ├── Task 4.1: Generate all plots
    ├── Task 4.2: Compute summary metrics
    └── Task 4.3: Write documentation
```

---

## Expected Outcomes

### If Hypothesis Confirmed (Low-Rank Far-Field → Boundary Coupling)

- **Singular values**: Rapid decay (e.g., σ_5 < 0.1·σ_1 for typical grids)
- **Effective rank**: Small relative to |B| (e.g., rank ≈ 5-10 even for |B| = 50)
- **Dominant modes**: Smooth boundary patterns (low Laplacian energy, low total variation)
- **Distance dependence**: Rank decreases as far-field blocks move further away

**Implication**: Far-field switching influences local boundary through O(1) or O(log n) smooth modes → hierarchical low-rank approximations are justified

### If Hypothesis Rejected

- **Singular values**: Flat decay (many modes of similar magnitude)
- **Effective rank**: Large, approaching |B| or K
- **Modes**: Oscillatory, high-frequency content

**Implication**: Far-field details matter at the boundary → need full resolution or different decomposition strategy

### Partial Confirmation (Likely Scenario)

- Strong low-rank structure for **distant** far-field blocks
- Weaker compression for **nearby** far-field (close to window)
- Anisotropic behavior based on metal routing direction (H vs V layers)

**Implication**: Use distance-adaptive approximation; nearby requires more modes, distant can use very few

---

## Dependencies

**Existing infrastructure to reuse:**
- `core/tiling.py`: `TileManager` for coordinate extraction and tile generation
- `core/unified_solver.py`: `UnifiedIRDropSolver.solve_hierarchical_coupled()` for coupled solve
- `core/unified_solver.py`: `UnifiedIRDropSolver.solve()` for flat solve validation
- `core/unified_model.py`: `UnifiedPowerGridModel._decompose_at_layer()` for port extraction
- `pdn_parser/`: `NetlistParser` for loading PDN netlists

**Standard libraries:**
- `numpy.linalg`: SVD
- `matplotlib`: Visualization
- `argparse`: CLI for analysis script

---

## Estimated Complexity

| Phase | Key Deliverables | Lines of Code (est.) |
|-------|-----------------|---------------------|
| Phase 1 | `farfield_analysis.py` | ~300 |
| Phase 1 | `scripts/analyze_farfield_coupling.py` | ~200 |
| Phase 3 | Experiment configurations (in script) | ~100 |
| Phase 4 | Plots + documentation | ~100 |
| **Total** | | ~700 lines |

**Note**: Complexity reduced by reusing TileManager and existing solver infrastructure.

---

## Future Work

### Temporal Analysis (Deferred)
- Build frequency-dependent admittance `Y(jω) = G + jωC`
- Analyze how H(jω) rank structure changes with frequency
- Identify temporal bandwidth of low-rank approximation

### Algorithmic Exploitation
If low-rank confirmed:
- Precompute dominant boundary modes for each far-field region
- Use low-rank update formulas for fast local solves
- Integrate with tiled hierarchical solver for speedup
- Implement "skeleton" approximation using representative far-field injections

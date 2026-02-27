# Solver Package

Flat, hierarchical, coupled, and tiled IR-drop solvers. Entry point: `UnifiedIRDropSolver` in `unified_solver.py`.

## Key Classes

- **UnifiedIRDropSolver**: Orchestrates all solve modes — `solve()` (flat), `solve_hierarchical()`, `solve_hierarchical_coupled()`, `solve_hierarchical_tiled()`
- **BlockMatrixSystem** (`coupled_system.py`): Block-partitioned conductance matrix (port/interior splits)
- **SchurComplementOperator** (`coupled_system.py`): Matrix-free Schur complement for coupled solver
- **CoupledSystemOperator** (`coupled_system.py`): Full coupled top-grid + Schur complement operator
- **CurrentAggregator** (`current_aggregation.py`): Distributes load currents to ports using shortest-path or effective resistance weighting
- **TileManager** (`tiling.py`): Tile generation, connectivity validation, and result merging
- **UnifiedPartitioner** (`unified_partitioner.py`): Layer-based and spatial grid partitioning
- **UnifiedEffectiveResistanceCalculator** (`effective_resistance.py`): Pairwise and single-node effective resistance
- **UnifiedStatistics** (`statistics.py`): Netlist statistics (node/edge counts, R/C/L/I totals)
- **PDNSolver** (`pdn_solver.py`): Standalone DC solver CLI

## Hierarchical Solve (Layer Decomposition)

```python
# Partition at layer boundary for faster bottom-grid solves
hier_result = solver.solve_hierarchical(
    load_currents,
    partition_layer='M2',      # or integer layer index
    top_k=5,                   # ports per load for current aggregation
    weighting="shortest_path", # "shortest_path" or "effective"
    verbose=True,              # print timing breakdown
)
print(f"Ports: {len(hier_result.port_nodes)}")
```

**Parameters:**
- `partition_layer`: Layer name (string like `'M2'`) or integer index
- `top_k`: Number of nearest ports per load for current aggregation (default 5)
- `weighting`: `"shortest_path"` (default) or `"effective"` (more accurate but slower)
- `rmax`: Maximum resistance distance for shortest_path weighting
- `use_fast_builder`: If True (default), use vectorized subgrid builder (~10x speedup)

## Coupled Hierarchical Solve (Exact)

Solves full coupled system iteratively using matrix-free Schur complement. Exact up to solver tolerance (~0.02 uV error).

```python
coupled_result = solver.solve_hierarchical_coupled(
    load_currents,
    partition_layer='M2',
    solver='gmres',            # 'gmres' or 'bicgstab'
    tol=1e-8,                  # Iterative solver tolerance
    maxiter=500,               # Max iterations
    preconditioner='block_diagonal',  # 'none', 'block_diagonal', or 'ilu'
    verbose=True,
)
print(f"Converged in {coupled_result.iterations} iterations")
print(f"Final residual: {coupled_result.final_residual:.2e}")
```

**Coupled vs Uncoupled:**
- **Uncoupled (`solve_hierarchical`)**: Approximates port currents via weighted distribution, then solves top/bottom grids independently. Fast but ~0.5 mV error.
- **Coupled (`solve_hierarchical_coupled`)**: Iterative Schur complement. Exact up to tolerance. Slower but accurate.

**Solver Parameters:**
- `solver`: `'cg'` (SPD, best for large), `'gmres'` (default, robust), `'bicgstab'` (often faster)
- `tol`: Residual tolerance (default 1e-8)
- `maxiter`: Max iterations (default 500)
- `preconditioner`: `'block_diagonal'` (default), `'ilu'`, `'amg'` (requires pyamg), `'none'`

**Recommended configurations:**
- Small (<100K nodes): `solver='bicgstab', preconditioner='ilu'`
- Large (>1M nodes): `solver='cg', preconditioner='amg'` (O(1) iterations)

**Note:** CG requires SPD preconditioner. Use `'amg'` or `'block_diagonal'` with CG, not `'ilu'`.

**UnifiedCoupledHierarchicalResult Fields:**
- All fields from `UnifiedHierarchicalResult` plus:
- `iterations`, `final_residual`, `converged`, `preconditioner_type`
- `timings`: Dict with 'factor_bottom', 'build_rhs', 'iterative_solve', 'recover_bottom'

## Tiled Hierarchical Solve (PDN only)

Exploits spatial locality by tiling the bottom-grid. **Only for PDN graphs** (string node names).

```python
tiled_result = solver.solve_hierarchical_tiled(
    current_injections=load_currents,
    partition_layer='M2',
    N_x=2, N_y=2,              # Tile grid dimensions
    halo_percent=0.2,          # Halo size as fraction of tile
    top_k=5,
    n_workers=4,               # Parallel workers (default: CPU count)
    parallel_backend='thread', # 'thread' or 'process'
    validate_against_flat=True,
)
print(f"Max diff vs flat: {tiled_result.validation_stats['max_diff']*1000:.3f} mV")
```

## Batch Solving (Multiple Current Scenarios)

Use prepare/solve_prepared to cache expensive precomputation (LU factorization, block matrices, operators):

```python
solver = UnifiedIRDropSolver(model)

# Prepare once (expensive)
ctx = solver.prepare_flat()

# Solve multiple scenarios (cheap: reuses cached factorization)
for scenario in current_scenarios:
    result = solver.solve_prepared(ctx, scenario)
```

**Available methods:**

| Method | Context Class | Use Case |
|--------|---------------|----------|
| `prepare_flat()` | `FlatSolverContext` | Multiple flat solves |
| `prepare_hierarchical()` | `HierarchicalSolverContext` | Multiple hierarchical solves |
| `prepare_hierarchical_coupled()` | `CoupledHierarchicalSolverContext` | Multiple coupled solves |
| `prepare_hierarchical_tiled()` | `TiledHierarchicalSolverContext` | Multiple tiled solves |

**Hierarchical batch example:**
```python
ctx = solver.prepare_hierarchical(partition_layer='M2', top_k=5)
for scenario in current_scenarios:
    result = solver.solve_hierarchical_prepared(ctx, scenario)
```

**Coupled batch example:**
```python
ctx = solver.prepare_hierarchical_coupled(partition_layer='M2', preconditioner='block_diagonal')
for scenario in current_scenarios:
    result = solver.solve_hierarchical_coupled_prepared(ctx, scenario)
```

**NOTE:** All standard `solve*()` methods also accept an optional `context` parameter. If not provided, they create a temporary context internally.

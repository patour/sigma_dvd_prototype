# `src/solver/` — flat / hierarchical / coupled / tiled solvers (validation oracles)

> These solvers are **validation reference paths** — exact flat-LU and iterative Schur baselines used to verify the distributed DDM solver. The distributed DDM path (see `src/distributed/`) is the primary production solver for large PDNs. Root `CLAUDE.md` covers conventions, units, and distributed pitfalls. This file is the API reference for the in-process solver hierarchy.
>
> `solver/coupled_system.py` and `solver/interface_assembly.py` are now **re-export shims** — all math has moved to `src/pgmath/`. Import from `pgmath` directly in new code; the shims exist for backward compat only.

## Mode comparison

| Mode | Method | When | Accuracy |
|---|---|---|---|
| `solve()` | direct LU on reduced system | small/medium grids, baseline | exact |
| `solve_hierarchical()` | layer-partitioned, weighted port currents | fast bottom-grid solves; tolerable approximation | ~0.5 mV error |
| `solve_hierarchical_coupled()` | matrix-free Schur complement, iterative | exact + still benefits from layer partitioning | exact to tol (~0.02 µV) |
| `solve_hierarchical_tiled()` | spatial tiling of bottom grid, parallel | PDN graphs only (string nodes); spatial locality | matches flat within validation_stats |
| `prepare_distributed()` / `solve_distributed_prepared()` | hands off to DDM | many tiles, multi-process | exact (DDM is structurally exact) |

All of these accept an optional `context=` to reuse cached precomputation. See *Batch solving* below.

## Hierarchical (uncoupled, approximate)

```python
hier = solver.solve_hierarchical(
    load_currents,
    partition_layer='M2',         # str layer name or int index
    top_k=5,                      # nearest ports per load
    weighting='shortest_path',    # or 'effective' (more accurate, slower)
    rmax=None,                    # cutoff for shortest-path weighting
    use_fast_builder=True,        # vectorized subgrid builder (~10x)
    verbose=False,
)
```

Approximates port currents by distributing each load over `top_k` ports, then solves top and bottom grids independently. Fast; ~0.5 mV error vs flat.

## Hierarchical coupled (exact, iterative)

```python
res = solver.solve_hierarchical_coupled(
    load_currents,
    partition_layer='M2',
    solver='gmres',                  # 'cg', 'gmres', 'bicgstab'
    tol=1e-8, maxiter=500,
    preconditioner='block_diagonal', # 'none', 'block_diagonal', 'ilu', 'amg'
)
```

Recipes:

- Small (<100K nodes): `solver='bicgstab', preconditioner='ilu'`
- Large (>1M nodes): `solver='cg', preconditioner='amg'` (O(1) iterations; needs `pyamg`)
- **CG needs an SPD preconditioner** — use `amg` or `block_diagonal`, never `ilu`.

Returns `UnifiedCoupledHierarchicalResult` with `iterations`, `final_residual`, `converged`, `timings`.

## Tiled (PDN-only, spatially partitioned)

```python
res = solver.solve_hierarchical_tiled(
    current_injections=load_currents,
    partition_layer='M2',
    N_x=2, N_y=2,                # tile grid
    halo_percent=0.2,            # halo as fraction of tile
    top_k=5,
    n_workers=None,              # default = CPU count
    parallel_backend='thread',   # or 'process'
    validate_against_flat=True,
)
```

Only works with PDN string nodes. `validation_stats['max_diff']` reports peak deviation vs flat in volts.

## Batch solving (`prepare_*` / `solve_*_prepared`)

Build the expensive bits once, solve many scenarios cheap:

| Mode | Prepare | Solve |
|---|---|---|
| flat | `prepare_flat()` → `FlatSolverContext` | `solve_prepared(ctx, currents)` |
| hierarchical | `prepare_hierarchical(...)` → `HierarchicalSolverContext` | `solve_hierarchical_prepared(ctx, currents)` |
| coupled | `prepare_hierarchical_coupled(...)` → `CoupledHierarchicalSolverContext` | `solve_hierarchical_coupled_prepared(ctx, currents)` |
| tiled | `prepare_hierarchical_tiled(...)` → `TiledHierarchicalSolverContext` | `solve_hierarchical_tiled_prepared(ctx, currents)` |
| distributed | `prepare_distributed(metadata=…)` | `solve_distributed_prepared(ctx, currents)` |

The non-`_prepared` `solve*()` methods also accept `context=`; they build a temporary one if not given.

## Coupled-system internals

`coupled_system.py` and `coupled_operators.py` together implement the matrix-free Schur path.

- `BlockMatrixSystem` — block-partitioned conductance matrix `[[G_ii, G_ip], [G_pi, G_pp]]`.
- `extract_block_matrices(...)` — builder used by the flat coupled solver. Has `exclude_port_to_port` flag.
- `build_block_system_from_edges(...)` — builder used by **distributed** tile workers; no `exclude_port_to_port` flag, includes all edges. Don't confuse the two.
- `build_grounded_capacitance_diags(...)` — diagonal C contribution for transient.
- `compute_explicit_schur(block_system)` — uses partial Cholesky when CHOLMOD is active, falls back to chunked multi-RHS with splu otherwise. CHOLMOD-only path sets `lu_ii` via the solve_L/Lt truncation trick.
- `compute_reduced_rhs(...)` — port-only RHS, shape `(n_ports,)`, not full system.
- `recover_bottom_voltages(...)` — interior recovery from port voltages.
- `SchurComplementOperator`, `CoupledSystemOperator` — `LinearOperator`s used by iterative solvers; preconditioners (`block_diagonal`, `ilu`, `amg`) wrap them.

## CHOLMOD backend (module-level)

```python
from solver.coupled_system import (
    set_use_cholmod, set_cholmod_mode,
    set_cholmod_ordering, set_cholmod_use_long,
    get_active_backend,
)
```

| Setting | Effect |
|---|---|
| `use_cholmod=True/False/None` | force on/off; None = auto |
| `cholmod_mode='auto'/'simplicial'/'supernodal'` | factorization strategy |
| `cholmod_ordering='amd'/'metis'/...` | fill-reducing ordering |
| `cholmod_use_long=True/False` | 64-bit indices (large problems) |

These are **module globals** — they do not cross process boundaries automatically. For Ray distributed workers, use `TileWorker.configure(...)` (handled by `create_distributed_model`).

## Other utilities

- `current_aggregation.CurrentAggregator` — distributes load currents to ports (shortest-path or effective-resistance weighting). Pad nodes are rejected.
- `tiling.TileManager` — tile generation, halo expansion, connectivity validation, result merge.
- `unified_partitioner.UnifiedPartitioner` — layer-based and spatial partitioning. Pads excluded; balance ratio enforced ≤ 3.5.
- `effective_resistance.UnifiedEffectiveResistanceCalculator` — pairwise + single-node R_eff; raises `ValueError` on pad nodes.
- `interface_assembly.py` — distributed interface assembly: `assemble_schur_complement_system`, `build_interface_package_matrices`, `find_interface_islands`, `apply_island_penalty`, `detect_interface_islands`. Uses `np.repeat`/`np.tile` for vectorized COO scatter-add (don't drop into Python loops here).
- `statistics.UnifiedStatistics` — node/edge counts, R/C/L/I totals.
- `pdn_solver.py` (`pdn-solve` CLI) — standalone DC solver if you don't want the unified interface.

## Schur regularization

`_compute_schur_partial(...)` adds a 1e-5 mS port-diagonal regularization for tiles whose full per-tile matrix is PSD-but-not-SPD (no ground connection). It's subtracted back from S after extraction. `G_ii` alone is always SPD, so interior solves don't need this. The default value is exposed via `set_partial_factor_reg_resistance` / `get_partial_factor_reg_resistance`.

## Floating islands & ground

- Ground node `'0'` is excluded from the conductance matrix but I-type edges to it are preserved (used during current extraction).
- Capacitive edges do NOT contribute to connectivity for island detection; a node connected to pads only via caps is floating.
- DC island detection penalty (`apply_island_penalty`) keeps the largest connected component plus components with ≥5 interface nodes (`MIN_INTERFACE_NODES_KEEP`).

# Distributed Package

Tile-based domain decomposition solver for multi-tile PDN netlists. Decomposes into per-tile subproblems coupled through a Schur complement interface system.

## File Organization

Mixin pattern keeps files under ~800 lines:

| File | Role |
|------|------|
| `solver.py` | `DistributedDDMSolver` — DC orchestration (`prepare`, `solve_dc`) |
| `solver_td.py` | Time-domain mixin: `preprocess_sources`, `solve_quasi_static`, `prepare_transient`, `solve_transient` |
| `solver_adjoint.py` | Adjoint mixin: `analyze_adjoint_static`, `analyze_adjoint` (backward sweep) |
| `tile_worker.py` | `TileWorker` — per-tile actor wrapping `BlockMatrixSystem` |
| `tile_worker_td.py` | Time-domain mixin: VCS init, transient factor/RHS, peak tracking, current node masking |
| `tile_worker_adjoint.py` | Adjoint worker mixin: terminal/step RHS, lambda recovery, contribution accumulation |
| `tile_parsing.py` | Stateless parsing: `TileData`, `_parse_tile_ckt`, `_iter_instance_sources`, `_parse_node_xy` |
| `model.py` | `DistributedPowerGridModel`, `ParsedTileBundle`, `create_distributed_model` |
| `parser.py` | `DistributedNetlistParser` — parse + dump tiles to pkl |
| `result.py` | Context/result dataclasses (`DistributedSolverContext`, `DistributedTransientContext`, etc.) |
| `result_factorization.py` | `factor()`, `release()`, `save()`, `load()`, `refactor()` implementations |
| `backend.py` | `LocalBackend`, `RayBackend` — compute abstraction |
| `heatmap.py` | Tile-parallel pre-binned stripe heatmap pipeline |
| `cli.py` | CLI: `python -m distributed {parse,solve,run}` with `--mode dc/quasi-static/transient` |

**Re-exports**: `tile_worker.py` re-exports all symbols from `tile_parsing.py` for backward compat. `__init__.py` re-exports the full public API.

## Context Lifecycle

```
prepare()  -->  DistributedSolverContext   (DC)
                  .factor() / .release() / .save() / .load() / .refactor()

prepare_transient()  -->  DistributedTransientContext   (transient)
                           .factor() / .release() / .save() / .load() / .refactor()
```

- `prepare()` and `prepare_transient()` are **independent** — caller manages both
- `solve_transient(trans_ctx, dc_context=dc_ctx)` does NOT release `dc_context`
- Two IC paths: `dc_context` (solve DC for IC) or `ic_voltages` (skip DC) — mutually exclusive
- `save()` must be called BEFORE `release()` (release clears `S_global`)
- After `load()`, call `refactor()` to rebuild coordinator LU; workers need separate `factor()`
- Topology (`DistributedTopologyContext`) is cached on `solver._topology` — computed once, reused

## Circular Import Rules

- `parser.py` cannot import from `model.py` at module level (model.py imports from parser.py). Use lazy imports inside functions.
- `result_factorization.py` uses `TYPE_CHECKING` guard for imports from `result.py` and `model.py`.

## Ray Worker Gotchas

- Module-level globals (CHOLMOD settings, regularization) do NOT propagate to Ray workers (separate processes). Use `TileWorker.configure(settings)` during `create_distributed_model`. CHOLMOD backend settings (`use_cholmod`, `cholmod_mode`, `cholmod_ordering`, `cholmod_use_long`) are now propagated automatically via the settings dict.
- `tile_parsing.py` duplicates unit constants (`R_TO_KOHM`, `C_TO_FF`, `I_TO_MA`) from `parser.py` to avoid circular imports.

## Transient Numerics

- Tile caps are grounded (diagonal C). `C_ip = C_pi = 0`. Package caps can couple (general sparse).
- Dirichlet RHS in time loop: use `rhs_dirichlet_G` (G-only), NOT `rhs_dirichlet_interface` (A-based, includes cap terms). BE: `+rhs_d_G`, TR: `+2*rhs_d_G`.
- Unit scaling: `dt_scaled = dt_seconds * 1e12` (ps). `C_coeff = 1/dt_scaled` (BE) or `2/dt_scaled` (TR).
- Tile matrix may be PSD (not SPD) without ground connections. `_compute_schur_partial()` adds 1e-5 mS regularization.

## Current Node Masking (Near/Far Decomposition)

`TileWorker.set_current_node_mask(mask)` / `build_node_mask_for_window(x0, x1, y0, y1, inside=True)` enable spatially-filtered transient solves. The mask is applied in both `evaluate_and_get_reduced_rhs` (QS) and `get_transient_reduced_rhs` (transient) after `evaluate_at_time(t)`. The transient factorization (A = G + C*C) is independent of currents and can be reused across masked solves.

## Key Entry Points

```python
from distributed import create_distributed_model, load_distributed_partitions, DistributedDDMSolver

# From pre-parsed pkl
model = load_distributed_partitions('./pkl_dir', backend='local')
solver = DistributedDDMSolver(model)

# DC
ctx = solver.prepare()
result = solver.solve_dc(ctx)
ctx.release()

# Transient
dc_ctx = solver.prepare()
trans_ctx = solver.prepare_transient(dt=100e-12, method='BE')
sources = solver.preprocess_sources(time_step=100e-12, t_end=10e-9)
result = solver.solve_transient(trans_ctx, dc_context=dc_ctx)
trans_ctx.release()
dc_ctx.release()
```

# `src/distributed/` — distributed DDM solver

> Root `CLAUDE.md` already covers the file map, the context-lifecycle skeleton, and the long list of distributed pitfalls (Dirichlet RHS in transient, partial Cholesky, Ray globals, save-before-release, island detection with caps, etc.). This file is the deeper internals reference: numerics, near/far decomposition, heatmap pipeline, CLI surface.

## Numerics & matrix structure

- Per-tile full matrix is `[[G_ii, G_ip], [G_pi, G_pp]]`. `G_ii` is always SPD; the full matrix may be PSD-only (no ground), so `_compute_schur_partial` adds a 1e-5 mS port-diagonal regularization and subtracts it back from S after extraction.
- Tile capacitors are grounded — diagonal C, with `C_ip = C_pi = 0`. Package caps can couple (general sparse).
- Transient time loop scaling: `dt_scaled = dt_seconds * 1e12` (ps). `C_coeff = 1/dt_scaled` (BE) or `2/dt_scaled` (TR).
- `BlockMatrixSystem.lu_ii` is the factored interior-solve callable; supports batch multi-RHS (matrix input).
- `build_block_system_from_edges` returns `rhs_dirichlet` of shape `(n_ports + n_interior,)`, not `(n_ports,)`.
- `compute_explicit_schur` requires `factor_interior()` first; uses batch solve, not column-by-column.

## Topology context

`DistributedTopologyContext` is immutable, computed once on first `prepare()` / `prepare_transient()`, cached on `solver._topology`. Both DC and transient contexts share it.

```
solver.prepare()  ─┐
                   ├─→  DistributedTopologyContext  (interface graph, islands, ownership)
solver.prepare_transient(dt, method)  ─┘
```

DC and transient contexts are otherwise **independent** lifecycles — caller manages both. `solve_transient(trans_ctx, dc_context=dc_ctx)` does NOT release `dc_ctx`. Two IC paths are mutually exclusive: `dc_context=` (DC solve for IC) vs `ic_voltages=` (skip DC).

## Save / load / refactor

```
ctx.factor()      # builds + factors (coordinator LU + worker tile factors)
ctx.save(path)    # serializes assembled S_global + per-worker state
ctx.release()     # frees factorizations and clears S_global
ctx.load(path)    # restores S_global + worker state
ctx.refactor()    # rebuilds coordinator LU from saved S_global
                  # (workers still need a separate factor() afterwards)
```

`save()` must run **before** `release()` because release clears `S_global`. After `load()`, both `refactor()` (coordinator) and `factor()` (workers) are required before the next solve.

## Near/far decomposition (`decomposition.py`)

Spatially partitions the solve so far-field current sources are folded into a static contribution and only near-field sources are stepped.

- `decompose_near_far(...)` — top-level pipeline
- `find_worst_nodes_separated(...)` — pick spatially-separated victim candidates (10% min separation)
- `extract_instance_locations_from_peaks(...)` — bridge from quasi-static peaks to instance coords
- `analyze_distributed_decomposition(...)` — diagnostic / validation helper

Tile-side: `TileWorker.set_current_node_mask(mask)` + `build_node_mask_for_window(x0, x1, y0, y1, inside=True)` enable spatially-filtered transient solves. Mask is applied in both `evaluate_and_get_reduced_rhs` (QS) and `get_transient_reduced_rhs` (transient) **after** `evaluate_at_time(t)`. The transient factorization (`A = G + α·C`) is current-independent and reused across masked solves.

## Heatmap pipeline (`heatmap.py`)

Root mentions the high-level pipeline; here are the building blocks:

| Step | Function | Where it runs |
|---|---|---|
| Per-layer metadata (bbox, stripe coords, edge orientation) | `TileWorker.get_layer_metadata()` | one round-trip per worker |
| Global bin spec (resolves stripe boundaries across tiles) | `build_global_bin_spec(...)` → `GlobalBinSpec` (collection of `LayerBinSpec`) | coordinator |
| Per-tile pre-binning | `prebin_tile(...)` | stateless + picklable, runs as Ray `map_func` |
| Boundary ownership (avoid double-counting) | `compute_boundary_ownership(...)` | coordinator |
| Merge & render | `merge_tile_prebins(...)`, `_render_merged_heatmaps(...)`, `render_from_prebinned_stripe_data(...)` (in `visualization/stripe_heatmap.py`) | coordinator |
| Top-level entry points | `plot_distributed_heatmaps(...)`, `plot_distributed_td_heatmaps(...)` | coordinator |

`prebin_tile` is intentionally stateless and uses lazy imports so it pickles cleanly to Ray workers. Current heatmaps skip layers with no current sources (all-zero bin check) — upper metals typically have none. For binning, filter out-of-range with `valid_mask`, never `np.clip` (clamping corrupts edge bin values).

## CLI (`python -m distributed`, `cli.py`)

Subcommands: `parse`, `solve`, `run`, `decompose`. Important flags:

```bash
python -m distributed solve <pkl_dir> \
    --backend {local,ray} \
    --mode {dc,quasi-static,transient} \
    --t-end 10ns --dt 100ps --n-points 11 \
    --plot [--plot-layers M0,M1] [--max-stripes 2000] \
    --config solver.yaml \
    --verbose
```

YAML config supports per-role solver settings (coordinator vs tile workers) — see `_apply_yaml_role_configs`. CLI also supports file logging (`_setup_logging`, `_add_file_logging`) and writes a top-K worst IR-drop report via the shared `reports.topk_irdrop.generate_topk_report`.

## Backends (`backend.py`)

Both implement the `ComputeBackend` ABC:

- `LocalBackend` — in-process; useful for tests and small models
- `RayBackend` — multi-process via Ray; workers are `TileWorker` actors

Backend is selected by `create_distributed_model(metadata, backend='local'|'ray')` or by `load_distributed_partitions(path, backend=...)`. Module-level globals (CHOLMOD settings, regularization) do NOT propagate to Ray workers — `TileWorker.configure(settings)` is called once during `create_distributed_model` to push them through. CHOLMOD knobs (`use_cholmod`, `cholmod_mode`, `cholmod_ordering`, `cholmod_use_long`) are propagated automatically via the settings dict.

## Tile parsing (`tile_parsing.py`)

Stateless, picklable functions used by both the parser and Ray workers:

- `TileData` — parsed tile container
- `_parse_tile_ckt(...)` — element parser
- `_iter_instance_sources(...)` — iterates filtered `instanceModels` entries (handles gzip + net filter). **Always reuse this**, don't reimplement gzip/filter logic.
- `_parse_node_xy('1000_2000_M1') -> (1000.0, 2000.0)` — shared coordinate parser. Don't duplicate.
- `_parse_instance_models`, `_iter_instance_capacitors`, `_parse_instance_capacitors`, `parse_tile_with_instances`, `parse_and_dump_tile`

Unit constants (`R_TO_KOHM`, `C_TO_FF`, `I_TO_MA`) are duplicated here from `parser.py` to avoid the circular import (`pdn_parser` → `core` → `core.distributed`).

## Tile worker mixins

`TileWorker` is composed via mixins to keep each file under ~800 lines:

- `_TimeDomainMixin` (`tile_worker_td.py`) — VCS init, transient factor/RHS/recovery, current-node masking
- `_PeakTrackingMixin` (`tile_worker_peak.py`) — peak init, dict/array update, accessors, fused recover+peak
- `_AdjointWorkerMixin` (`tile_worker_adjoint.py`) — terminal/step RHS, lambda recovery, contribution accumulation

`tile_worker.py` re-exports all `tile_parsing.py` symbols for backward compat.

## Solver mixins

`DistributedDDMSolver(_AdjointMixin, _SolverTimeDomainMixin)`:

- `_SolverTimeDomainMixin` (`solver_td.py`) — `preprocess_sources`, `solve_quasi_static`, `prepare_transient`, `solve_transient`
- `_AdjointMixin` (`solver_adjoint.py`) — `analyze_adjoint_static`, `analyze_adjoint`

## Result types (`result.py`)

- `DistributedTopologyContext` — immutable shared topology (see above)
- `DistributedSolverContext` — DC; `factor`/`release`/`save`/`load`/`refactor`
- `DistributedTransientContext` — transient; same lifecycle methods
- `DistributedSmoothedSources` — coordinator-side handle to preprocessed VCS (data lives on workers)
- `DistributedSolveResult` — flat DC result
- `DistributedQuasiStaticResult` — lazy peak collection from workers; `as_flat()` / `as_per_tile()` / `dump()`
- `DistributedTransientResult` — extends QS result with RC transient metadata
- `TileSolveResult` — per-tile slice of a solve result

`result_factorization.py` holds the `factor`/`release`/`save`/`load`/`refactor` implementations split out from the dataclasses to keep `result.py` minimal. Uses `TYPE_CHECKING` guards to import `result.py` and `model.py` types.

## Net filter & metadata

Net filtering happens at parse time (`PowerGridMetaData` set by the distributed parser's `net_filter`). `create_distributed_model(metadata, backend='local')` reads net info from there, not from graph attributes. `solver.prepare_distributed(metadata=…)` (on the *unified* solver) is the corresponding entry point.

## Tests

- `tests/distributed/test_distributed_solver.py` — 41 tests (38 unit/validation + 3 benchmark)
- `tests/distributed/test_distributed_heatmap.py` — 39 tests
- `tests/distributed/test_distributed_td_heatmap.py` — time-domain heatmap pipeline
- `tests/distributed/test_time_domain*.py` — quasi-static + transient validation
- `tests/distributed/test_adjoint_integration.py` — adjoint sensitivity validation
- `tests/distributed/test_distributed_cli.py` — CLI surface
- Integration: `*_integration.py` files (mark `@pytest.mark.integration`)

`tests/distributed/test_time_domain.py::_build_two_tile_distributed_model` is the standard fixture for minimal 2-tile models with optional cap edges.

## Benchmark snapshot

`netlist_sampled` benchmark: Ray with 9 workers ≈ 2.7× total speedup; `factor_tiles` dominates the prepare phase.

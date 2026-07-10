# `src/distributed/` — distributed DDM solver

> Root `CLAUDE.md` already covers the file map, context-lifecycle skeleton, distributed pitfalls (Dirichlet RHS, partial Cholesky, Ray globals, save-before-release, island detection, Phase-A/B features). This file is the deeper internals reference: numerics, phase-folded RHS, symbolic reuse, interface solver, retiling, streaming assembly, near/far decomposition, heatmap pipeline, CLI surface.

## Numerics & matrix structure

- Per-tile full matrix is `[[G_ii, G_ip], [G_pi, G_pp]]`. `G_ii` is always SPD; the full matrix may be PSD-only (no ground), so `_compute_schur_partial` adds a 1e-5 mS port-diagonal regularization and subtracts it back from S after extraction.
- Tile capacitors are grounded — diagonal C, with `C_ip = C_pi = 0`. Package caps can couple (general sparse).
- Transient time loop scaling: `dt_scaled = dt_seconds * 1e12` (ps). `C_coeff = 1/dt_scaled` (BE) or `2/dt_scaled` (TR).
- `BlockMatrixSystem.lu_ii` is the factored interior-solve callable; supports batch multi-RHS (matrix input).
- `build_block_system_from_edges` returns `rhs_dirichlet` of shape `(n_ports + n_interior,)`, not `(n_ports,)`.
- `compute_explicit_schur` requires `factor_interior()` first; uses batch solve, not column-by-column.

## Topology context

`DistributedTopologyContext` is immutable, computed once on first `prepare()` / `prepare_transient()`, cached on `solver._topology`. Both DC and transient contexts share it. Island detection results are cached here (A7 — computed once, not re-run for transient prepare).

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

## A2 — Phase-folded RHS (step columns)

`VectorizedCurrentSources` sources are periodic (smoothing retains period, folds delay). With `use_step_columns=True` (default), the coordinator probes workers via `get_period_info()` then calls `precompute_step_columns(t_start, dt, m)` once before the loop. Workers build a `(n_active_nodes, m)` float64 table `C` (Fortran order, port-first); per step `k`, the worker gathers column `C[:, (k + phase0) % m]` instead of calling `evaluate_at_time`.

**Tiers (auto-selected by `get_period_info` probe):**
1. **Single period** — one table of `m = round(P/dt)` columns. Build via smoothed-grid direct scatter when sources are already smoothed (build ≈ free); else vectorized `evaluate_at_times(t_grid)`; else scalar loop fallback.
2. **Multi-period** — per-group tables, summed per step.
3. **Aperiodic or table exceeds `max_table_mb`** — chunked window (`W ≈ 512` steps), streamed; still hoists segment search from inner loop. When sources are smoothed-grid-aligned (`_smoothed_grid_alignment`), windows are built via `_gather_window_direct` (index gather, wrap-at-m; requires per-row `cnt >= m+1`, else falls back to `evaluate_at_times_for_rows`). Fast-path metadata (`_fast_path`, `_ts_m`, `_m_fold`) is stored in the table dict so on-demand window rebuilds never re-probe.
4. **Skipped** — chunked tier + single window (`n_steps <= W`) + no fast path + no reusable cache: `precompute_step_columns` returns `{'tier': 'skipped'}` and the loop uses per-step `evaluate_at_time` (exact by construction; a single-window build costs the same evaluate work while allocating a multi-GB intermediate).

**Cross-transient reuse (worker-side cache)**: the table is a pure function of (active sources, dt, tier grid) — the near/far mask is post-gather, so one table serves all decomposition victims and all ~6 `solve_transient` calls in a decompose run. `precompute_step_columns` caches key = sources-version counter + `(dt, max_table_mb)` (+ `t_start` for chunked); on a hit it returns `{'reused': True}` without rebuilding. Phase tables are reused across *any* dt-grid-aligned `t_start` (phase0 recomputed); chunked reuse extends `n_steps` monotonically so window rebuilds never clamp.

**Invalidation rule**: `_step_cols` and the reuse cache key are cleared (and `_sources_version` bumped) by `init_vectorized_sources`, `smooth_sources`, `use_smoothed_sources`, and `use_raw_sources`. Call `precompute_step_columns` after any of these, not before.

**Equivalence tolerance**: column gather vs direct `evaluate_at_time` ≤ 1e-9 mA (fp-modulo roundoff from `t % P`). End-to-end transient results equal to flag-off ≤ 1e-12 V.

Near/far mask (`_current_node_mask`) is applied post-gather, so the same `C` table serves all decomposition victims without rebuilding.

## A4 — Symbolic and assembly-pattern reuse

**Tile symbolic reuse**: `_compute_schur_partial` caches the CHOLMOD symbolic object (`_symbolic_ii`) from the DC factor. Transient factor (`A_ii = G_ii + C_coeff·diag(c)`) shares `G_ii`'s sparsity pattern, so only a numeric refactor is needed. If the pattern check fails (e.g., after retiling a context that was saved pre-split), falls back to full re-analyze — correct but slower. Also benefits `refactor()` after `load()`.

**Interface assembly-pattern reuse**: `assemble_schur_complement_system` caches its COO/CSR index arrays on the `DistributedTopologyContext`. Subsequent calls (transient prepare, refactor) substitute values into the cached pattern without re-computing sparsity structure.

## A5 — Smoothed-VCS disk cache

Cache path: `<pkl_dir>/vcs_tile_<id>_smoothed_<hash>.pkl` where `hash` covers `(time_step, t_start, t_end, compact_threshold, SMOOTHING_CODE_VERSION)`.

**Invalidation rule**: bump `SMOOTHING_CODE_VERSION` (integer constant in `tile_worker_td.py`) whenever smoothing logic changes. The old hash no longer matches, so all tiles silently rebuild on next run. The raw VCS cache (separate file, no version suffix) is not affected.

`preprocess_sources(smooth='auto')` skips smoothing when `time_step` ≤ the smallest PWL segment (no aliasing risk). Always pass `smooth=False` when running the equivalence suite to keep source inputs identical on both sides.

## B1 — Balanced retiling

`retile.py` runs inside `DistributedNetlistParser.parse_and_dump()`. Tiles with `n_interior > max_interior` are recursively bisected by node coordinates (`_parse_node_xy` from `tile_parsing.py`).

- Parent `(x, y)` yields sub-tiles with 3-tuple IDs `(x, y, k)`.
- `_tile_id_str(tile_id)` converts any-length tuple to `'_'`-joined slug for filenames, VCS cache paths, and log messages.
- `_try_axis_split` sweeps coordinate-value transition points only (O(distinct_coord_values), not O(n)); tiles with identical coordinates are left unsplit with a warning.
- `split_tile(tile_data, max_interior, alpha=0.5)` is the public entry in `retile.py`; `alpha` controls balance vs. cut-cost trade-off. `parser._apply_tile_splits()` calls it for each oversized tile.
- `create_distributed_model(..., tiles_per_worker='auto')` packs tiles into `PackedTileWorker` groups when tile count exceeds actor budget.

**Exactness**: DC/QS exact. Transient FP noise ≤ 2e-14 V for one-level bisections (BRCM-class); up to ~60 nV (BE) / ~6 µV (TR) for very aggressive four-level splits — below integration-method truncation error.

## B2 — Iterative interface solve

`interface_iterative.py`: `InterfaceCGSolver` implements CG on the SPD global Schur `S_global` with block-Jacobi preconditioner (per-tile diagonal `S_i` blocks). `auto_select_interface_solver(n_interface)` returns `'direct'` when `n_interface < AUTO_CG_N_INTERFACE_THRESHOLD` (200,000) and estimated factor memory is within budget; else `'cg'`.

Override via `model.settings['interface_solver'] = 'direct'|'cg'|'auto'` or YAML config. The resolved mode is stored as `ctx._interface_solver_mode` and propagated through `save()`/`load()`. Adjoint code checks `_interface_solver_mode` before choosing the solve path.

Warm-start from previous step's `v_gamma` (transient changes slowly → typically few iterations/step on smooth waveforms).

## B3 — Streaming Schur assembly

`streaming_assembly=False` (default): assemble `S_global` fully in memory before factoring — straightforward, requires full `n_interface² × density` RAM.

`streaming_assembly=True`: workers cache `S_i` via `factor_and_cache_schur()` and return it as COO shards via `get_schur_coo_shards(n_shards, tile_index_map)` (first call uses a two-pass index-only/data-only protocol: `get_schur_coo_indices_only` + `get_schur_data_flat`); coordinator accumulates into a pre-allocated CSR using the A4 cached assembly pattern, freeing each shard immediately. Peak memory is proportional to one tile's shard, not the sum of all `S_i`.

`streaming_assembly='auto'`: switches to streaming when the estimated `S_i` peak exceeds `STREAMING_ASSEMBLY_AUTO_BYTES` (default 512 MB). Override via `model.settings['streaming_assembly_auto_bytes']`.

**Constraint**: `streaming_assembly=True` is incompatible with assembling `S_global` for `interface_solver='cg'` (CG uses `S_global` explicitly in the non-tilewise-matvec mode). When `interface_solver='cg'` with tilewise matvec, `S_global` assembly can be skipped entirely.

The transient path reuses the B3 `factor_transient_and_cache_schur()` code path for the streaming DC-factor structure.

## Near/far decomposition (`decomposition.py`)

Spatially partitions the solve so far-field current sources are folded into a static contribution and only near-field sources are stepped.

- `decompose_near_far(...)` — top-level pipeline
- `find_worst_nodes_separated(...)` — pick spatially-separated victim candidates (10% min separation)
- `extract_instance_locations_from_peaks(...)` — bridge from quasi-static peaks to instance coords
- `analyze_distributed_decomposition(...)` — diagnostic / validation helper

Tile-side: `TileWorker.set_current_node_mask(mask)` + `build_node_mask_for_window(x0, x1, y0, y1, inside=True)` enable spatially-filtered transient solves. Mask is applied post-column-gather (A2), so the same phase table `C` serves all victims. The transient factorization (`A = G + α·C`) is current-independent and reused across masked solves.

**A6**: victim waveforms are now captured during the Phase 2b main sweep via `_PeakTrackingMixin.get_tracked_waveforms`; the redundant Phase-3 all-sources transient in `decomposition.py` is eliminated.

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

## CLI (`sigma-dvd` / `python -m distributed`, `cli.py`)

Subcommands: `parse`, `solve`, `run`, `decompose`. Important flags:

```bash
sigma-dvd solve <pkl_dir> \
    --backend {local,ray} \
    --mode {dc,quasi-static,transient} \
    --t-end 10ns --dt 100ps --n-points 11 \
    --max-interior 400000 \
    --tiles-per-worker auto \
    --plot [--plot-layers M0,M1] [--max-stripes 2000] \
    --config solver.yaml \
    --verbose
```

YAML config supports per-role solver settings (coordinator vs tile workers) — see `_apply_yaml_role_configs`. `interface_solver`, `streaming_assembly`, `use_step_columns`, `max_table_mb`, CHOLMOD knobs are all settable via YAML. CLI also supports file logging (`_setup_logging`, `_add_file_logging`) and writes a top-K worst IR-drop report via the shared `reports.topk_irdrop.generate_topk_report`.

## Backends (`backend.py`)

Both implement the `ComputeBackend` ABC:

- `LocalBackend` — in-process; useful for tests and small models; supports `PackedTileWorker` (tiles_per_worker > 1)
- `RayBackend` — multi-process via Ray; workers are `TileWorker` actors; `PackedTileWorkerActor` routes batched calls

Backend is selected by `create_distributed_model(metadata, backend='local'|'ray')` or by `load_distributed_partitions(path, backend=...)`. Module-level globals (CHOLMOD settings, regularization) do NOT propagate to Ray workers — `TileWorker.configure(settings)` is called once during `create_distributed_model` to push them through. CHOLMOD knobs (`use_cholmod`, `cholmod_mode`, `cholmod_ordering`, `cholmod_use_long`) are propagated automatically via the settings dict. `RayBackend` also sets `OMP_NUM_THREADS`/`OPENBLAS_NUM_THREADS` per actor via `runtime_env` from `threads_per_worker`.

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

- `_TimeDomainMixin` (`tile_worker_td.py`) — VCS init, smooth/cache, `precompute_step_columns`, transient factor/RHS/recovery, current-node masking
- `_PeakTrackingMixin` (`tile_worker_peak.py`) — peak init, dict/array update, accessors, fused recover+peak
- `_AdjointWorkerMixin` (`tile_worker_adjoint.py`) — terminal/step RHS, lambda recovery, contribution accumulation

`tile_worker.py` re-exports all `tile_parsing.py` symbols for backward compat. The module docstring describes math delegation to `pgmath` (not `solver/coupled_system.py` — that is a shim).

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
- `tests/validation/test_equivalence.py` — flat-vs-distributed equivalence gate (marker `validation`)
- Integration: `*_integration.py` files (mark `@pytest.mark.integration`)

`tests/distributed/test_time_domain.py::_build_two_tile_distributed_model` is the standard fixture for minimal 2-tile models with optional cap edges.

## Benchmark snapshot (netlist_sampled, post Phase A–B)

Measured on `netlist_sampled` (Ray, 9 workers, 100 steps, BE). Baseline captured in `scripts/benchmark/baselines/perf_netlist_sampled.json`.

| Metric | Pre-refactor (BRCM-scale estimate) | Post-refactor (netlist_sampled measured) |
|--------|-----------------------------------|------------------------------------------|
| Transient loop total | –31% vs pre-A baseline | loop_total ≈ 5.1 s (100 steps) |
| Transient prepare | –70% vs pre-A baseline | trans_prepare ≈ 1.1 s |
| Smoothing (first run) | –98% cached (A5) | smooth ≈ 0.17 s first run / ~seconds cached |
| DC prepare | –12% vs pre-A baseline | dc_prepare ≈ 0.79 s |
| Per-step RHS | A2 step-cols: ~1164× on periodic waveforms | rhs ≈ 28 ms/step |

All 68/68 notebook regression metrics within tolerance. BRCM re-measurement pending bundle access (no BRCM netlist on this host).

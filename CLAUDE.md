# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Static and dynamic IR-drop analysis prototype for multi-layer power grids. Supports synthetic grids and real PDN netlists; quasi-static (batch DC) and transient RC analysis. The **distributed DDM solver** is the primary production path for large PDNs (100M+ nodes); flat/hierarchical/tiled solvers serve as validation oracles.

`src/` is installed editable (`pip install -e .`) with `pythonpath = ["src"]` set in `pyproject.toml`, so imports are `from distributed import ...` / `from pgmath.block_system import ...`, not `from src.distributed...`. (`solver/coupled_system.py`, `solver/interface_assembly.py`, and the factor block of `solver/unified_solver.py` are re-export shims over `pgmath`.)

## Commands

```bash
# Install (editable, with test deps)
uv pip install -e ".[test]"

# Primary CLI (distributed-first)
sigma-dvd solve ./netlist/netlist_sampled/distributed_pkl --backend ray --mode transient \
    --t-end 10ns --dt 100ps --verbose
sigma-dvd parse  ./netlist/netlist_test --net VDD
sigma-dvd run    ./netlist/netlist_sampled/distributed_pkl --backend ray --verbose

# Equivalent long form
python -m distributed solve ./netlist/netlist_sampled/distributed_pkl --backend ray --plot --verbose
python -m distributed solve ./pkl_dir --mode quasi-static --t-end 100ns --n-points 11 --verbose
python -m distributed solve ./pkl_dir --mode transient --t-end 10ns --dt 100ps --verbose

# Validation reference CLIs (flat path)
python -m parser.pdn_parser ./netlist/netlist_test --net VDD
python -m solver.pdn_solver --input graph.pkl --net VDD --output ./results
python -m analysis.dynamic_irdrop_decomposition ./netlist/netlist_test --net VDD --end-time 100ns --dt 100ps

# Unit tests (fast, <160s) — preferred during iteration
pytest -m unit

# Validation reference paths only
pytest -m validation

# Performance benchmarks
pytest -m benchmark

# Specific module
pytest tests/solver/test_hierarchical_solver.py
pytest tests/distributed/test_distributed_solver.py

# Single test
pytest tests/legacy/test_irdrop.py::TestIRDrop::test_no_load_currents_all_pad_voltage -v

# Integration tests (~6 min) — only when relevant area changed
pytest -m integration

# Everything (~10 min) — final validation only
pytest

# Performance benchmark on netlist_sampled
python scripts/benchmark/run_perf_baseline.py \
    --pkl-dir netlist/netlist_sampled/distributed_pkl \
    [--compare scripts/benchmark/baselines/perf_netlist_sampled.json --max-regress 10%]
```

**Test priority:** run individual unit test files for the affected area first, then `pytest -m unit`. Integration files only when changes touch that area. Bare `pytest` is a final check, not an iteration step.

## Architecture

### Distributed DDM data flow (primary path)

```
DistributedNetlistParser.parse_and_dump()
  └─ _apply_tile_splits(max_interior=...)         # B1 balanced retiling (calls retile.split_tile)
       └─ ParsedTileBundle  →  create_distributed_model(bundle)
            └─ DistributedDDMSolver
                 ├─ prepare()                     # DC factor + island cache (A7)
                 │    └─ DistributedSolverContext
                 ├─ preprocess_sources(smooth='auto')  # A5 smoothed-VCS disk cache
                 ├─ solve_quasi_static()          # phase-folded step columns (A2)
                 ├─ prepare_transient(dt, method) # symbolic reuse (A4)
                 │    └─ DistributedTransientContext
                 └─ solve_transient(ctx, dc_context=dc_ctx)  # A1 array exchange
```

**Context lifecycle (DC + transient are independent):**
```python
dc_ctx = solver.prepare()                                  # builds + factors
result = solver.solve_dc(dc_ctx)
trans_ctx = solver.prepare_transient(dt=100e-12, method='BE')
sources = solver.preprocess_sources(time_step=100e-12, t_end=10e-9)
trans_result = solver.solve_transient(trans_ctx, dc_context=dc_ctx)  # OR ic_voltages=...
trans_ctx.release(); dc_ctx.release()                      # caller owns both
```

**Key constraint:** pads (voltage sources) are Dirichlet BCs at Vdd, eliminated via Schur complement. LU factorization is cached for batch solves.

Heatmap pipeline: `get_layer_metadata()` → `build_global_bin_spec()` → `prebin_tile()` (per tile, picklable, `map_func`-able) → `merge_tile_prebins()` → `render_from_prebinned_stripe_data()`.

### Validation reference paths

- **Synthetic**: `generate_power_grid()` → `create_model_from_synthetic(G, pads, vdd)` → `UnifiedIRDropSolver`
- **PDN**: `NetlistParser.parse()` → `create_model_from_pdn(graph, net_name)` → `UnifiedIRDropSolver`
- **Multi-net PDN**: `create_multi_net_models(graph)` → iterate

### Module Tree (high-level)

- `src/pgmath/` — **shared math layer** (no solver/distributed/analysis imports): `block_system.py` (`BlockMatrixSystem`, `build_block_system_from_edges`, `extract_block_matrices`, `compute_reduced_rhs`, `recover_bottom_voltages`); `schur.py` (`compute_explicit_schur`, `assemble_schur_complement_system`, interface island utilities); `factor.py` (`SparseFactorAdapter`, `SolverBackendConfig`, `_factor_conductance_matrix`, CHOLMOD constants)
- `src/graph/` — rustworkx wrappers, networkx↔rustworkx conversion (`ensure_rustworkx_graph`)
- `src/model/` — `UnifiedPowerGridModel`, factories (`create_model_from_pdn/synthetic/multi_net`), result/context dataclasses
- `src/solver/` — `UnifiedIRDropSolver` (`solve`, `solve_hierarchical`, `solve_hierarchical_coupled`, `solve_hierarchical_tiled`); `SchurComplementOperator`, `CoupledSystemOperator`, `CurrentAggregator`, `TileManager`, `pdn_solver` CLI. **Validation oracle** — `solver/coupled_system.py` and `solver/interface_assembly.py` are re-export shims for `pgmath`.
- `src/analysis/` — `DynamicIRDropSolver` (quasi-static), `TransientIRDropSolver` (BE/TR), `AdjointSensitivitySolver`, `PWLSmoother`, `VectorizedCurrentSources`, `farfield_analysis`
- `src/parser/` — `NetlistParser` (gzip, parallel, net-filter), `sampled_netlist`, `parallel`, `edge_attrs` (memory-optimized slotted edges), `current_sources` (`Pulse`, `PWL`, `CurrentSource`)
- `src/distributed/` — DDM solver; see *Distributed solver* below
- `src/reports/` — shared report writers: `floating_nodes`, `topk_irdrop`
- `src/visualization/` — `UnifiedPlotter`, `DynamicPlotter`, `PDNPlotter`, `stripe_heatmap` (`parse_node_info`, `render_from_prebinned_stripe_data`)
- `src/legacy/` — **validation oracle** — original synthetic-grid path: `generate_power_grid`, `PowerGridModel`, `IRDropSolver`, `StimulusGenerator`, `GridPartitioner`

### Distributed solver (`src/distributed/`)

DDM is algebraically exact for any partition (DC, QS, and transient). The flat-vs-distributed comparison gives 0 µV diff. For B1 split-vs-unsplit: DC/QS exact; transient FP noise grows with the number of interface (cut) nodes — ≤ 2e-14 V for one-level bisections (max_interior ≈ 8000 on a 135K-node PDN), up to ~60 nV (BE) / ~6 µV (TR) for very aggressive four-level splits. Files use a mixin pattern to keep each under ~800 lines:

| File | Role |
|------|------|
| `solver.py` | `DistributedDDMSolver` — DC orchestration (`prepare`, `solve_dc`) |
| `solver_td.py` | Time-domain mixin: `preprocess_sources`, `solve_quasi_static`, `prepare_transient`, `solve_transient` |
| `solver_adjoint.py` | `analyze_adjoint_static`, `analyze_adjoint` |
| `tile_worker.py` / `tile_worker_td.py` / `tile_worker_peak.py` / `tile_worker_adjoint.py` | Per-tile actor wrapping `BlockMatrixSystem` (mixins for transient, peak tracking, adjoint) |
| `tile_parsing.py` | Stateless parsing: `TileData`, `_parse_tile_ckt`, `_iter_instance_sources`, `_parse_node_xy` |
| `model.py` | `DistributedPowerGridModel`, `ParsedTileBundle`, `create_distributed_model` |
| `parser.py` | `DistributedNetlistParser` |
| `retile.py` | B1 tile splitting: `split_tile` (public entry), `_try_axis_split`, `_tile_id_str` |
| `interface_iterative.py` | B2 CG interface solver: `InterfaceCGSolver`, `auto_select_interface_solver` |
| `result.py` / `result_factorization.py` | Context/result dataclasses; `factor`/`release`/`save`/`load`/`refactor` |
| `backend.py` | `LocalBackend`, `RayBackend`, `PackedTileWorker` |
| `heatmap.py` | Tile-parallel pre-binned stripe heatmap pipeline |
| `cli.py` | `sigma-dvd` / `python -m distributed {parse,solve,run,decompose}` |

## pytest markers

| Marker | Meaning |
|--------|---------|
| `unit` | Fast, isolated; no external netlist needed; preferred during iteration |
| `integration` | Slow; needs real netlist data (`netlist_test` / `netlist_sampled`) |
| `validation` | Validation reference path tests (hierarchical, tiled, legacy, equivalence suite) |
| `benchmark` | Performance throughput tests (slow; guarded by `run_perf_baseline.py`) |

All four are registered in `pyproject.toml`. `tests/validation/test_equivalence.py` carries `validation` at module level.

## Critical conventions

### Node types
- **Synthetic**: `NodeID(layer, idx)` frozen dataclass
- **PDN**: strings like `'1000_2000_M1'`, `'VDD_vsrc'`, `'0'` (ground)

### Unit system (PDN)
- Resistance kΩ, capacitance fF, inductance nH, current mA
- Conductance matrix in mS so `G·V = I` is self-consistent

### Current sign convention
- Input: positive = sink drawing from grid (`currents[node] = +1.0` mA)
- Solver internally negates for the nodal equation
- IR-drop is always reported as `Vdd − V_node` (positive = drop below Vdd)

## Pitfalls (project-specific, not generic)

### General
- **Plotting**: `plot_ir_drop_map(G, voltages, vdd=1.0, ...)` takes a scalar `vdd`, not the pad list.
- **Stimulus area**: `StimulusGenerator(graph=G, ...)` must receive `graph` if you use the `area` parameter.
- **R_eff queries**: pad nodes are rejected in pairwise calculations (raises `ValueError`).
- **PDN current extraction**: use `model.extract_current_sources()` to get load currents from I-type edges.
- **Headless plotting**: `show=False` for batch; `tests/conftest.py` sets matplotlib `Agg` backend.
- **Legacy pickles**: old `pdn_graph.pkl` files contain NetworkX graphs — call `ensure_rustworkx_graph(graph)` first.
- **Optimized edge attrs (default)**: `elem_name` is *not* stored on most resistors. Use `data.get('elem_name', '')`, never `data['elem_name']`. Computed properties (`.tile_id`, `.net_type`) unpack on the fly and are 4–5× slower than dict access — cache in hot loops.
- **Lazy factorization (default)**: `create_model_from_pdn(..., lazy_factor=True)` defers LU until first flat solve. Pass `lazy_factor=False` only if you need eager factorization (backward compat / flat-only).

### Distributed solver — core
- **Island detection**: must exclude ground node `'0'` from BFS (ground edges are diagonal-only). Capacitive edges do **not** contribute to connectivity for island detection — but DO filter cap edges when removing island nodes.
- **Boundary current partitioning**: external current injection on boundary nodes must go to exactly one tile to avoid double-counting.
- **Transient Dirichlet RHS in time loop**: use `rhs_dirichlet_G` (G-only), NOT `rhs_dirichlet_interface` (A-based, includes cap terms). BE: `+rhs_d_G`. TR: `+2*rhs_d_G`. Pad-cap history cancels because pads hold constant voltage.
- **Tile matrix SPD**: per-tile `[[G_ii, G_ip], [G_pi, G_pp]]` may be PSD (not SPD) for tiles with no ground connection. `_compute_schur_partial()` adds 1e-5 mS port-diagonal regularization and subtracts it from S after extraction. `G_ii` alone is always SPD.
- **Partial Cholesky Schur**: `compute_explicit_schur(block_system)` uses partial Cholesky when CHOLMOD is active; falls back to chunked multi-RHS with splu. CHOLMOD-only path sets `lu_ii` via the solve_L/Lt truncation trick.
- **`build_block_system_from_edges` vs `extract_block_matrices`**: tile worker uses the former (no `exclude_port_to_port` param, includes all edges). Flat coupled solver uses the latter. Don't confuse.
- **`solve_quasi_static` default smoothing**: calling without `smoothed_sources` triggers `preprocess_sources(smooth=True)` and silently overwrites `_active_sources` on workers. Always pass an explicit handle if VCS is already initialized.
- **Context save/load**: `save()` must run **before** `release()` (release clears `S_global`). After `load()`, call `refactor()` to rebuild the coordinator LU; workers need a separate `factor()`.
- **Ray worker globals**: module-level globals do **not** propagate to Ray workers (separate processes). Use `TileWorker.configure(settings)` during `create_distributed_model`. CHOLMOD settings (`use_cholmod`, `cholmod_mode`, `cholmod_ordering`, `cholmod_use_long`) are propagated automatically via the settings dict.
- **Notebook cwd for Ray**: workers inherit the driver's cwd. Notebooks must `os.chdir` to the project root before creating the model so tile metadata relative paths resolve.
- **Circular imports inside `distributed/`**: `parser.py` cannot import `model.py` at module level (model.py imports from parser.py). Use lazy imports. `result_factorization.py` uses `TYPE_CHECKING` for `result.py`/`model.py`.
- **`_parse_current_source_line`** (in `parser/current_sources.py`) returns Amperes; conversion to mA happens post-parse in `_prepare_instance_source` (`current_sources.py`, also `parallel.py`). Multiply by `I_TO_MA` if you call it directly.
- **`_parse_node_xy`**: shared utility in `distributed/tile_parsing.py` for `'1000_2000_M1'` → `(1000.0, 2000.0)`. Don't reimplement.

### Distributed solver — Phase A/B features
- **`use_step_columns` invalidation**: `_step_cols` is cleared by `init_vectorized_sources`, `smooth_sources`, and `use_smoothed_sources`. If you call any of these after `precompute_step_columns`, rebuild the table. `use_step_columns=True` (default) is propagated via `TileWorker.configure`; set `max_table_mb` (default 512) to cap per-worker memory.
- **Array exchange port-gather convention**: `_precompute_port_gathers` returns `(port_gather, pad_mask)` per tile — computed once before the time loop. Per step: `v_arr = np.where(pad_mask, vdd, v_gamma[port_gather])`. Pad/Dirichlet ports are not in `interface_node_to_idx`, so `port_gather[j]` is 0 (dummy) for them; `pad_mask[j]` gates the `np.where`.
- **Smoothed-VCS cache key**: `vcs_tile_X_Y_smoothed_<hash(time_step,t_start,t_end,compact_threshold,SMOOTHING_CODE_VERSION)>.pkl`. Bump `SMOOTHING_CODE_VERSION` in `tile_worker_td.py` whenever smoothing logic changes; existing caches invalidate automatically.
- **Symbolic reuse fallback**: A4 caches the CHOLMOD symbolic analysis from the DC factor for reuse in the transient factor (`G + α·C` shares `G`'s sparsity). If the sparsity pattern changes (e.g., after retiling), the symbolic check fails and falls back to full re-analyze — correct but slower.
- **`interface_solver` auto thresholds**: `'auto'` (default) selects direct CHOLMD/SuperLU when `n_interface < 200,000` and estimated factor memory is within budget; else CG. Override via `model.settings['interface_solver'] = 'direct'|'cg'|'auto'` or `--config solver.yaml`. Threshold is `AUTO_CG_N_INTERFACE_THRESHOLD = 200_000` in `interface_iterative.py`.
- **`streaming_assembly` semantics**: `False` (default) — assemble full `S_global` in memory before factoring. `True` — tile shards arrive in batches and are accumulated into a pre-allocated CSR using the A4 cached assembly pattern; peak memory is proportional to one shard, not the full matrix. `'auto'` — switches to streaming when estimated `S_i` peak exceeds `streaming_assembly_auto_bytes`. Incompatible with `interface_solver='cg'` without prior `S_global` assembly.
- **B1 retiling — 3-tuple tile IDs**: `retile.split_tile(tile_data, max_interior)` is the public entry; `parser._apply_tile_splits()` calls it for each oversized tile. Parent `(x, y)` yields sub-tiles `(x, y, k)`. `_tile_id_str` converts any-length tuple to `'_'`-joined slug used for filenames and VCS cache keys. VCS cache filenames include the full tile ID slug, so sub-tiles never collide with the parent cache.
- **B1 split exactness**: DC/QS exact; BE/TR FP noise ≤ 2e-14 V for one-level bisection. For aggressive splits (4+ levels), noise can reach ~60 nV (BE) / ~6 µV (TR). This is below integration-method truncation error and is NOT a physics bug.
- **B1 unsplit tiles warning**: `_apply_tile_splits` emits `WARNING ... n_tiles_over_max ... tiles still have n_interior > max_interior` when a tile cannot be bisected (all candidates cut coupling caps, or all interior nodes share identical coordinates). These tiles remain oversized and factor slowly but are otherwise correct.
- **B1 candidate sweep**: `_try_axis_split` uses a transition-point sweep (not exhaustive O(n)) for large tiles (n > 1000). Only tries cut positions at coordinate-value transitions; O(distinct_coord_values) not O(n). Cheap 0-interior pre-check skipped if uncoordinated interior nodes exist.
- **TR IC lesson**: the interior IC must be recovered using the *same VCS RHS* that was used at the final quasi-static step — `_last_qs_rhs_i` stored on the worker. Using static `tile_data.current_injections` instead causes an IC inconsistency between interface and interior recoveries. TR amplifies this via stiff-node period-2 oscillation (BE damps it quickly). Pre-fix distributed TR diff was ~0.5 mV; post-fix ≤ 1.4e-15 V (machine precision).

### Visualization / heatmaps
- **Current heatmaps**: only render layers that actually have current sources (all-zero check). Upper metals typically have none.
- **`np.digitize` binning**: filter out-of-range with `valid_mask`, NOT `np.clip` — clamping corrupts edge bin values.

## PDN netlist format

```
netlist_dir/
  ckt.sp                  # Top-level includes
  tile_X_Y.ckt            # R/C/L/I/V elements
  tile_X_Y.nd             # Node coords: x y layer node_name
  package.ckt             # Package-level connections
  instanceModels_X_Y.sp   # Instance current source models
  pg_net_voltage          # Power-net voltages, e.g. "VDD 1.0"
  additional_vsrcs        # Extra vsrc definitions
  decap_cell_list         # Decap cell instance names
  switch_cell_list        # Power-switch cell names
```

**Element syntax** (units are kΩ / fF / nH / mA / V):
```
R_name n1 n2 <kOhm>
C_name n1 n2 <fF>
L_name n1 n2 <nH>
I_name n1 n2 <mA>          # current source (load)
V_name n+ n- <V>           # voltage source (pad)
X_inst subckt n1 n2 ...    # subcircuit instance
```

**Node naming:** `<x>_<y>_<layer>` (e.g. `1000_2000_M1`).

## File landmarks

- **Notebooks**: `notebooks/irdrop_decomposition_pdn.ipynb`, `notebooks/irdrop_decomposition.ipynb`, `notebooks/irdrop_decomposition_unified_model.ipynb`
- **Parity notebooks** (must reproduce baseline JSONs): `notebooks/dynamic_irdrop_decomposition.ipynb`, `notebooks/transient_analysis_validation.ipynb`, `notebooks/distributed_transient_analysis_validation.ipynb`, `notebooks/distributed_dynamic_irdrop_decomposition.ipynb`
- **Test netlists**: `netlist/netlist_test/` (small PDN, integration), `netlist/netlist_small/` (minimal unit fixtures), `netlist/netlist_sampled/` (distributed benchmark)
- **Equivalence suite**: `tests/validation/test_equivalence.py` (marker `validation`)
- **Perf baseline**: `scripts/benchmark/baselines/perf_netlist_sampled.json`; runner: `scripts/benchmark/run_perf_baseline.py`
- **API exports**: each `src/<pkg>/__init__.py`
- **Copilot equivalent**: `.github/copilot-instructions.md` (overlapping content; this file is authoritative for Claude)

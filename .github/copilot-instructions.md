# Power Grid IR-Drop Analysis Prototype

Static and dynamic IR-drop analysis prototype for multi-layer power grids. Supports synthetic grids and real PDN netlists; quasi-static (batch DC) and transient RC analysis. The **distributed DDM solver** (`src/distributed/`) is the primary production path for large PDNs (100M+ nodes); flat/hierarchical/tiled solvers in `src/solver/` and `src/legacy/` serve as validation oracles.

`src/` is installed editable (`pip install -e .`) with `pythonpath = ["src"]` set in `pyproject.toml`, so imports are `from distributed import ...` / `from pgmath.block_system import ...`, **not** `from src.distributed...`. (`solver/coupled_system.py`, `solver/interface_assembly.py`, and the factor block of `solver/unified_solver.py` are re-export shims over `pgmath`.)

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
- `src/distributed/` — DDM solver; see *Distributed solver internals* below
- `src/reports/` — shared report writers: `floating_nodes`, `topk_irdrop`
- `src/visualization/` — `UnifiedPlotter`, `DynamicPlotter`, `PDNPlotter`, `stripe_heatmap` (`parse_node_info`, `render_from_prebinned_stripe_data`)
- `src/legacy/` — **validation oracle** — original synthetic-grid path: `generate_power_grid`, `PowerGridModel`, `IRDropSolver`, `StimulusGenerator`, `GridPartitioner`

## Critical Domain Conventions

### Node Types
- **Synthetic**: `NodeID(layer, idx)` frozen dataclass keys the graph
- **PDN**: string node names like `'1000_2000_M1'`, `'VDD_vsrc'`, `'0'` (ground)

### Unit System (PDN)
- Resistance kΩ, Capacitance fF, Inductance nH, Current mA
- Conductance matrix in mS (milli-Siemens) so `G·V = I` is self-consistent

### Current Sign Convention (CRITICAL)
- **Input**: Positive current = sink drawing from grid (`currents[node] = +1.0 mA`)
- **Internal**: Solver negates for the nodal equation
- **IR-drop**: Always `Vdd - V_node` (positive = voltage dropped below Vdd)

## Common Pitfalls

### General
- **Plotting**: `plot_ir_drop_map(G, voltages, vdd=1.0, ...)` requires scalar `vdd`, NOT the pad list
- **Stimulus area**: `StimulusGenerator(graph=G, ...)` must pass `graph` if using the `area` parameter
- **R_eff queries**: pad nodes rejected in pairwise calculations (raises `ValueError`)
- **PDN current extraction**: use `model.extract_current_sources()` to get load currents from I-type edges
- **Headless plotting**: `show=False` for batch; `tests/conftest.py` sets matplotlib `Agg` backend
- **Legacy pickles**: old `pdn_graph.pkl` files may contain NetworkX graphs — call `ensure_rustworkx_graph(graph)` first before `create_model_from_pdn()`
- **Optimized edge attrs (default)**: `elem_name` is *not* stored on most resistors. Use `data.get('elem_name', '')`, never `data['elem_name']`. Computed properties (`.tile_id`, `.net_type`) unpack on the fly and are 4-5x slower than dict access — cache in hot loops
- **Lazy factorization (default)**: `create_model_from_pdn(..., lazy_factor=True)` defers LU until first flat solve. Pass `lazy_factor=False` only if you need eager factorization (backward compat / flat-only)
- **PDN ground node**: Ground is `'0'` string; excluded from the conductance matrix but preserved for I-type edges
- **Gaussian degeneracy**: `StimulusGenerator` falls back to uniform if weights sum to zero

### Distributed solver — core (`src/distributed/`)
- **Island detection**: must exclude ground node `'0'` from BFS (ground edges are diagonal-only). Capacitive edges do **not** contribute to connectivity for island detection — but DO filter cap edges when removing island nodes
- **Boundary current partitioning**: external current injection on boundary nodes must go to exactly one tile to avoid double-counting
- **Transient Dirichlet RHS in time loop**: use `rhs_dirichlet_G` (G-only), NOT `rhs_dirichlet_interface` (A-based, includes cap terms). BE: `+rhs_d_G`. TR: `+2*rhs_d_G`. Pad-cap history cancels because pads hold constant voltage
- **Tile matrix SPD**: per-tile `[[G_ii, G_ip], [G_pi, G_pp]]` may be PSD (not SPD) for tiles with no ground connection. `_compute_schur_partial()` adds 1e-5 mS port-diagonal regularization and subtracts it from S after extraction. `G_ii` alone is always SPD
- **Partial Cholesky Schur**: `compute_explicit_schur(block_system)` uses partial Cholesky when CHOLMOD is active; falls back to chunked multi-RHS with splu. CHOLMOD-only path sets `lu_ii` via the solve_L/Lt truncation trick. Requires `factor_interior()` first; uses batch solve, not column-by-column
- **`build_block_system_from_edges` vs `extract_block_matrices`**: tile worker uses the former (no `exclude_port_to_port` param, includes all edges). Flat coupled solver uses the latter. Don't confuse. `build_block_system_from_edges` returns `rhs_dirichlet` of shape `(n_ports + n_interior,)`, not `(n_ports,)`
- **`solve_quasi_static` default smoothing**: calling without `smoothed_sources` triggers `preprocess_sources(smooth=True)` and silently overwrites `_active_sources` on workers. Always pass an explicit handle if VCS is already initialized
- **Context save/load**: `save()` must run **before** `release()` (release clears `S_global`). After `load()`, call `refactor()` to rebuild the coordinator LU; workers need a separate `factor()`
- **Ray worker globals**: module-level globals do **not** propagate to Ray workers (separate processes). Use `TileWorker.configure(settings)` during `create_distributed_model`. CHOLMOD settings (`use_cholmod`, `cholmod_mode`, `cholmod_ordering`, `cholmod_use_long`) are propagated automatically via the settings dict
- **Notebook cwd for Ray**: workers inherit the driver's cwd. Notebooks must `os.chdir` to the project root before creating the model so tile metadata relative paths resolve
- **Circular imports inside `distributed/`**: `parser.py` cannot import `model.py` at module level (model.py imports from parser.py). Use lazy imports. `result_factorization.py` uses `TYPE_CHECKING` for `result.py`/`model.py`
- **`_parse_current_source_line`** (in `parser/current_sources.py`) returns Amperes; conversion to mA happens post-parse in `_prepare_instance_source` (`current_sources.py`, also `parallel.py`). Multiply by `I_TO_MA` if you call it directly
- **`_parse_node_xy`**: shared utility in `distributed/tile_parsing.py` for `'1000_2000_M1'` → `(1000.0, 2000.0)`. Don't reimplement
- **Tile capacitors are grounded** — diagonal C, with `C_ip = C_pi = 0`. Package caps can couple (general sparse)
- **Transient time loop scaling**: `dt_scaled = dt_seconds * 1e12` (ps). `C_coeff = 1/dt_scaled` (BE) or `2/dt_scaled` (TR)
- **Topology context**: `DistributedTopologyContext` is immutable, computed once on first `prepare()`/`prepare_transient()`, cached on `solver._topology`. Both DC and transient contexts share it; island detection is cached here (A7). DC and transient contexts are otherwise **independent** lifecycles — `solve_transient(trans_ctx, dc_context=dc_ctx)` does NOT release `dc_ctx`. Two IC paths are mutually exclusive: `dc_context=` (DC solve for IC) vs `ic_voltages=` (skip DC)

### Distributed solver — Phase A/B features
- **`use_step_columns` invalidation**: `_step_cols` is cleared by `init_vectorized_sources`, `smooth_sources`, `use_smoothed_sources`, and `use_raw_sources` (each also bumps the worker's `_sources_version`, invalidating the cross-transient reuse cache). Rebuild after calling any of these. `use_step_columns=True` (default) is propagated via `TileWorker.configure`; set `max_table_mb` (default 512) to cap per-worker memory. Cross-transient reuse (worker-side cache) means one phase table serves all decomposition victims and all `solve_transient` calls in a decompose run — the near/far mask (`_current_node_mask`) is applied post-gather
- **Array exchange port-gather convention**: `_precompute_port_gathers` returns `(port_gather, pad_mask)` per tile — computed once before the time loop. Per step: `v_arr = np.where(pad_mask, vdd, v_gamma[port_gather])`. Pad/Dirichlet ports are not in `interface_node_to_idx`, so `port_gather[j]` is 0 (dummy) for them; `pad_mask[j]` gates the `np.where`
- **Smoothed-VCS cache key**: `vcs_tile_X_Y_smoothed_<hash(time_step,t_start,t_end,compact_threshold,SMOOTHING_CODE_VERSION)>.pkl`. Bump `SMOOTHING_CODE_VERSION` in `tile_worker_td.py` whenever smoothing logic changes; existing caches invalidate automatically. `preprocess_sources(smooth='auto')` skips smoothing when `time_step` ≤ the smallest PWL segment. Always pass `smooth=False` for the equivalence suite
- **Symbolic reuse (A4)**: `_compute_schur_partial` caches the CHOLMOD symbolic object (`_symbolic_ii`) from the DC factor; the transient factor (`A_ii = G_ii + C_coeff·diag(c)`) shares `G_ii`'s sparsity so only a numeric refactor is needed. If the sparsity pattern changes (e.g., after retiling), falls back to full re-analyze — correct but slower. `assemble_schur_complement_system` similarly caches its COO/CSR index arrays on the topology context
- **`interface_solver` auto thresholds**: `'auto'` (default) selects direct CHOLMOD/SuperLU when `n_interface < 200,000` and estimated factor memory is within budget; else CG (`InterfaceCGSolver`, block-Jacobi preconditioned, warm-started from the previous step's `v_gamma`). Override via `model.settings['interface_solver'] = 'direct'|'cg'|'auto'` or `--config solver.yaml`. Threshold is `AUTO_CG_N_INTERFACE_THRESHOLD = 200_000` in `interface_iterative.py`
- **`streaming_assembly` semantics**: `False` (default) — assemble full `S_global` in memory before factoring. `True` — tile shards arrive as COO batches and accumulate into a pre-allocated CSR using the A4 cached assembly pattern; peak memory is proportional to one shard, not the full matrix. `'auto'` — switches to streaming when estimated `S_i` peak exceeds `streaming_assembly_auto_bytes` (default 512MB). Incompatible with `interface_solver='cg'` without prior `S_global` assembly (fine when CG uses tilewise matvec)
- **B1 retiling — 3-tuple tile IDs**: `retile.split_tile(tile_data, max_interior, alpha=0.5)` is the public entry; `parser._apply_tile_splits()` calls it for each oversized tile. Parent `(x, y)` yields sub-tiles `(x, y, k)`. `_tile_id_str` converts any-length tuple to `'_'`-joined slug used for filenames and VCS cache keys, so sub-tiles never collide with the parent cache. `_try_axis_split` sweeps coordinate-value transition points only (O(distinct_coord_values), not O(n)); tiles with identical coordinates are left unsplit with a warning
- **B1 split exactness**: DC/QS exact; BE/TR FP noise ≤ 2e-14 V for one-level bisection (max_interior ≈ 8000 on a 135K-node PDN); up to ~60nV (BE) / ~6µV (TR) for very aggressive four-level splits — below integration-method truncation error, NOT a physics bug
- **TR IC lesson**: the interior IC must be recovered using the *same VCS RHS* that was used at the final quasi-static step — `_last_qs_rhs_i` stored on the worker. Using static `tile_data.current_injections` instead causes an IC inconsistency between interface and interior recoveries; TR amplifies this via stiff-node period-2 oscillation (BE damps it quickly)

### Visualization / heatmaps
- **Current heatmaps**: only render layers that actually have current sources (all-zero check). Upper metals typically have none
- **`np.digitize` binning**: filter out-of-range with `valid_mask`, NOT `np.clip` — clamping corrupts edge bin values
- Pipeline building blocks: `TileWorker.get_layer_metadata()` (per worker) → `build_global_bin_spec()` → `GlobalBinSpec`/`LayerBinSpec` → `prebin_tile()` (stateless, picklable, Ray `map_func`-able) → `compute_boundary_ownership()` (avoid double-counting) → `merge_tile_prebins()` → `render_from_prebinned_stripe_data()` (`visualization/stripe_heatmap.py`) → `plot_distributed_heatmaps()` / `plot_distributed_td_heatmaps()`

## Distributed solver internals (`src/distributed/`)

DDM is algebraically exact for any partition (DC, QS, and transient); the flat-vs-distributed comparison gives 0µV diff. Files use a mixin pattern to keep each under ~800 lines:

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
| `decomposition.py` | Near/far decomposition: `decompose_near_far`, `find_worst_nodes_separated`, `extract_instance_locations_from_peaks`, `analyze_distributed_decomposition` |
| `cli.py` | `sigma-dvd` / `python -m distributed {parse,solve,run,decompose}` |

**Backends** (`backend.py`, both implement `ComputeBackend`): `LocalBackend` (in-process, tests/small models, supports `PackedTileWorker`), `RayBackend` (multi-process; `TileWorker`/`PackedTileWorkerActor` actors; sets `OMP_NUM_THREADS`/`OPENBLAS_NUM_THREADS` per actor via `runtime_env` from `threads_per_worker`). Selected via `create_distributed_model(metadata, backend='local'|'ray')` or `load_distributed_partitions(path, backend=...)`.

**CLI flags** (`sigma-dvd solve <pkl_dir> --backend {local,ray} --mode {dc,quasi-static,transient} --t-end 10ns --dt 100ps --n-points 11 --tiles-per-worker auto --plot [--plot-layers M0,M1] [--max-stripes 2000] --config solver.yaml --verbose`). `--max-interior` (B1 retiling) is a **parse-time** flag only (`parse`/`run`), since splitting happens inside `parse_and_dump()` and is baked into the pkl bundle: `sigma-dvd parse <netlist_dir> --net VDD --backend ray --max-interior 400000 -o <netlist_dir>/distributed_pkl_split`. YAML config supports per-role solver settings (coordinator vs tile workers, `_apply_yaml_role_configs`); `interface_solver`, `streaming_assembly`, `use_step_columns`, `max_table_mb`, CHOLMOD knobs are all settable via YAML.

**Near/far decomposition**: tile-side `TileWorker.set_current_node_mask(mask)` + `build_node_mask_for_window(x0, x1, y0, y1, inside=True)` enable spatially-filtered transient solves; mask applied post-column-gather (A2) so the same phase table serves all victims. A6: victim waveforms captured during the Phase 2b main sweep via `_PeakTrackingMixin.get_tracked_waveforms`, eliminating a redundant Phase-3 all-sources transient.

**Result types** (`result.py`): `DistributedTopologyContext` (immutable shared topology), `DistributedSolverContext`/`DistributedTransientContext` (`factor`/`release`/`save`/`load`/`refactor`), `DistributedSmoothedSources` (coordinator handle to preprocessed VCS; data lives on workers), `DistributedSolveResult`, `DistributedQuasiStaticResult` (lazy peak collection; `as_flat()`/`as_per_tile()`/`dump()`), `DistributedTransientResult`, `TileSolveResult`.

**Net filter & metadata**: net filtering happens at parse time (`PowerGridMetaData` set by the distributed parser's `net_filter`). `create_distributed_model(metadata, backend='local')` reads net info from there, not from graph attributes.

## pytest markers

| Marker | Meaning |
|--------|---------|
| `unit` | Fast, isolated; no external netlist needed; preferred during iteration |
| `integration` | Slow; needs real netlist data (`netlist_test` / `netlist_sampled`) |
| `validation` | Validation reference path tests (hierarchical, tiled, legacy, equivalence suite) |
| `benchmark` | Performance throughput tests (slow; guarded by `run_perf_baseline.py`) |

All four are registered in `pyproject.toml`. `tests/validation/test_equivalence.py` carries `validation` at module level.

**Distributed tests**: `tests/distributed/test_distributed_solver.py` (41 tests: 38 unit/validation + 3 benchmark), `test_distributed_heatmap.py` (39 tests), `test_distributed_td_heatmap.py`, `test_time_domain*.py` (quasi-static + transient), `test_adjoint_integration.py`, `test_distributed_cli.py`. `tests/distributed/test_time_domain.py::_build_two_tile_distributed_model` is the standard fixture for minimal 2-tile models with optional cap edges.

## PDN Netlist Format

Directory structure:
```
netlist_dir/
  ckt.sp              # Top-level circuit includes
  tile_0_0.ckt        # Tile subcircuit with R/C/L/I/V elements
  tile_0_0.nd         # Node coordinate mapping (x y layer node_name)
  package.ckt         # Package-level connections
  instanceModels_0_0.sp  # Instance current source models
  pg_net_voltage      # Power net voltage specs (VDD 1.0, VSS 0.0)
  additional_vsrcs    # Extra voltage source definitions
  decap_cell_list     # Decap cell instance names
  switch_cell_list    # Power switch cell names
```

**Element Syntax in `.ckt` files** (units are kΩ / fF / nH / mA / V):
```spice
R_name node1 node2 <resistance_kOhm>
C_name node1 node2 <capacitance_fF>
L_name node1 node2 <inductance_nH>
I_name node1 node2 <current_mA>       # Current source (instance load)
V_name node+ node- <voltage_V>        # Voltage source (pad)
X_inst subckt node1 node2 ...         # Subcircuit instance
```

**Node naming convention:** `<x>_<y>_<layer>` (e.g., `1000_2000_M1`)

## Typical Workflow Patterns

### PDN Netlist Analysis (validation / flat path)
```python
from parser.netlist import NetlistParser
from model.factory import create_model_from_pdn
from solver.unified_solver import UnifiedIRDropSolver

parser = NetlistParser('./netlist/netlist_test', validate=True)
graph = parser.parse()
model = create_model_from_pdn(graph, 'VDD')  # vdd auto-extracted from graph
load_currents = model.extract_current_sources()

solver = UnifiedIRDropSolver(model)
result = solver.solve(load_currents)
print(f"Max IR-drop: {max(result.ir_drop.values()):.4f} V")
```

### Distributed PDN Analysis (production path)
```python
from distributed import DistributedNetlistParser, create_distributed_model

parser = DistributedNetlistParser('./netlist/netlist_sampled', net_filter=['VDD'])
bundle = parser.parse_and_dump('./netlist/netlist_sampled/distributed_pkl', max_interior=400_000)
model = create_distributed_model(bundle, backend='ray')

solver = model.solver
dc_ctx = solver.prepare()
dc_result = solver.solve_dc(dc_ctx)

trans_ctx = solver.prepare_transient(dt=100e-12, method='BE')
solver.preprocess_sources(time_step=100e-12, t_end=10e-9)
trans_result = solver.solve_transient(trans_ctx, dc_context=dc_ctx)
trans_ctx.release(); dc_ctx.release()
```

### Synthetic Grid Analysis
```python
from legacy.generate_power_grid import generate_power_grid
from model.factory import create_model_from_synthetic
from solver.unified_solver import UnifiedIRDropSolver
from legacy import StimulusGenerator

G, loads, pads = generate_power_grid(K=3, N0=12, I_N=150, N_vsrc=4, seed=42)
model = create_model_from_synthetic(G, pads, vdd=1.0)

stim_gen = StimulusGenerator(load_nodes=list(loads.keys()), vdd=1.0, seed=10, graph=G)
meta = stim_gen.generate(total_power=1.5, percent=0.3, distribution="gaussian")

solver = UnifiedIRDropSolver(model)
result = solver.solve(meta.currents)
```

### Hierarchical Solve (Layer Decomposition, validation oracle)
```python
hier_result = solver.solve_hierarchical(
    load_currents,
    partition_layer='M2',  # or integer layer index
    top_k=5,               # ports per load for current aggregation
    weighting="shortest_path",
    verbose=True,
)
```

### Multi-Net PDN Analysis
```python
from parser.netlist import NetlistParser
from model.factory import create_multi_net_models
from solver.unified_solver import UnifiedIRDropSolver

parser = NetlistParser('./netlist/netlist_test', validate=True)
graph = parser.parse()

models = create_multi_net_models(graph, net_filter=['VDD', 'VSS'])
for net_name, model in models.items():
    solver = UnifiedIRDropSolver(model)
    currents = model.extract_current_sources()
    result = solver.solve(currents)
    print(f"{net_name}: Max IR-drop = {max(result.ir_drop.values())*1000:.2f} mV")
```

## Testing & Validation

**Run tests**: `pytest -m unit` (fast iteration); see Commands above for the full matrix.

**Test layout mirrors `src/`:**
```
tests/
  graph/          # test_rx_graph, test_rx_algorithms
  model/          # test_unified_core
  solver/         # test_hierarchical_solver, test_coupled_hierarchical_solver, ...
  analysis/       # test_dynamic_solver, test_transient_solver, test_adjoint_sensitivity, ...
  parser/         # test_pdn_parser, test_parallel_parser, test_edge_attrs, ...
  distributed/    # test_distributed_solver, test_distributed_heatmap, test_time_domain*, ...
  visualization/  # test_pdn_plotter, test_stripe_heatmap
  legacy/         # test_irdrop, test_partitioner
  validation/     # test_equivalence (flat-vs-distributed equivalence gate)
```

**Test netlists:** `netlist/netlist_test/` (small PDN, integration), `netlist/netlist_small/` (minimal unit fixtures), `netlist/netlist_sampled/` (distributed benchmark).

**Key invariants tested**:
- Zero load -> all nodes at pad voltage
- R_eff symmetry: `R(u,v) == R(v,u)` and triangle inequality
- Partition balance ratio <= 3.5; pads excluded from partitions
- Floating island detection removes disconnected components
- Flat-vs-distributed equivalence within tolerance (0µV DC/QS; ≤2e-14V transient for one-level B1 splits)

## File Landmarks

- **Notebooks**: `notebooks/irdrop_decomposition_pdn.ipynb`, `notebooks/irdrop_decomposition.ipynb`, `notebooks/irdrop_decomposition_unified_model.ipynb`
- **Parity notebooks** (must reproduce baseline JSONs): `notebooks/dynamic_irdrop_decomposition.ipynb`, `notebooks/transient_analysis_validation.ipynb`, `notebooks/distributed_transient_analysis_validation.ipynb`, `notebooks/distributed_dynamic_irdrop_decomposition.ipynb`
- **Tests**: `tests/{graph,model,solver,analysis,parser,distributed,visualization,legacy,validation}/test_*.py`
- **Test netlists**: `netlist/netlist_test/`, `netlist/netlist_small/`, `netlist/netlist_sampled/`
- **Equivalence suite**: `tests/validation/test_equivalence.py` (marker `validation`)
- **Perf baseline**: `scripts/benchmark/baselines/perf_netlist_sampled.json`; runner: `scripts/benchmark/run_perf_baseline.py`
- **API exports**: each `src/<pkg>/__init__.py`
- **Scripts**: `scripts/analysis/`, `scripts/solver/`, `scripts/parser/`, `scripts/benchmark/`, `scripts/distributed/`
- **Per-directory guides**: `CLAUDE.md` (root, authoritative for Claude Code), `src/distributed/CLAUDE.md` (distributed internals deep dive), plus `tests/CLAUDE.md`, `src/{analysis,model,parser,solver,visualization}/CLAUDE.md` — check the relevant one when working deep in a subpackage; this file is the Copilot equivalent and stays in sync with root `CLAUDE.md`.

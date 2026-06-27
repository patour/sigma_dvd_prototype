# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Static and dynamic IR-drop analysis prototype for multi-layer power grids. Supports synthetic grids and real PDN netlists; quasi-static (batch DC) and transient RC analysis; flat, hierarchical, coupled, tiled, and distributed (DDM) solvers.

`src/` is installed editable (`pip install -e .`) with `pythonpath = ["src"]` set in `pyproject.toml`, so imports are `from solver.unified_solver import ...`, not `from src.solver...`.

## Commands

```bash
# Install (editable, with test deps)
uv pip install -e ".[test]"

# Unit tests (fast, <160s) — preferred during iteration
pytest -m unit

# Specific module
pytest tests/solver/test_hierarchical_solver.py
pytest tests/distributed/test_distributed_solver.py

# Single test
pytest tests/legacy/test_irdrop.py::TestIRDrop::test_no_load_currents_all_pad_voltage -v

# Integration tests (~6 min) — only when relevant area changed
pytest -m integration

# Everything (~10 min) — final validation only
pytest

# CLIs
python -m parser.pdn_parser ./netlist/netlist_test --net VDD
python -m solver.pdn_solver --input graph.pkl --net VDD --output ./results
python -m analysis.dynamic_irdrop_decomposition ./netlist/netlist_test --net VDD --end-time 100ns --dt 100ps
python -m distributed solve ./netlist/netlist_sampled/distributed_pkl --backend ray --plot --verbose
python -m distributed solve ./pkl_dir --mode quasi-static --t-end 100ns --n-points 11 --verbose
python -m distributed solve ./pkl_dir --mode transient --t-end 10ns --dt 100ps --verbose
```

**Test priority:** run individual unit test files for the affected area first, then `pytest -m unit`. Integration files only when changes touch that area. Bare `pytest` is a final check, not an iteration step.

## Architecture

### Data Flow

- **Synthetic**: `generate_power_grid()` → `create_model_from_synthetic(G, pads, vdd)` → `UnifiedIRDropSolver`
- **PDN**: `NetlistParser.parse()` → `create_model_from_pdn(graph, net_name)` → `UnifiedIRDropSolver`
- **Multi-net PDN**: `create_multi_net_models(graph)` → iterate
- **Distributed (DDM)**: `DistributedNetlistParser.parse_and_dump()` → `ParsedTileBundle` → `create_distributed_model(bundle)` → `DistributedDDMSolver`

**Key constraint:** pads (voltage sources) are Dirichlet BCs at Vdd, eliminated via Schur complement. LU factorization is cached for batch solves.

### Module Tree (high-level)

- `src/graph/` — rustworkx wrappers, networkx↔rustworkx conversion (`ensure_rustworkx_graph`)
- `src/model/` — `UnifiedPowerGridModel`, factories (`create_model_from_pdn/synthetic/multi_net`), result/context dataclasses
- `src/solver/` — `UnifiedIRDropSolver` (`solve`, `solve_hierarchical`, `solve_hierarchical_coupled`, `solve_hierarchical_tiled`); `BlockMatrixSystem`, `SchurComplementOperator`, `CoupledSystemOperator`, `CurrentAggregator`, `TileManager`, `pdn_solver` CLI
- `src/analysis/` — `DynamicIRDropSolver` (quasi-static), `TransientIRDropSolver` (BE/TR), `AdjointSensitivitySolver`, `PWLSmoother`, `VectorizedCurrentSources`, `farfield_analysis`
- `src/parser/` — `NetlistParser` (gzip, parallel, net-filter), `sampled_netlist`, `parallel`, `edge_attrs` (memory-optimized slotted edges), `current_sources` (`Pulse`, `PWL`, `CurrentSource`)
- `src/distributed/` — DDM solver, see *Distributed solver* below
- `src/reports/` — shared report writers: `floating_nodes`, `topk_irdrop`
- `src/visualization/` — `UnifiedPlotter`, `DynamicPlotter`, `PDNPlotter`, `stripe_heatmap` (`parse_node_info`, `render_from_prebinned_stripe_data`)
- `src/legacy/` — original synthetic-grid path: `generate_power_grid`, `PowerGridModel`, `IRDropSolver`, `StimulusGenerator`, `GridPartitioner`

### Distributed solver (`src/distributed/`)

DDM is structurally exact — matches the flat solver to floating-point precision (0 µV diff on validation). Files use a mixin pattern to keep each under ~800 lines:

| File | Role |
|------|------|
| `solver.py` | `DistributedDDMSolver` — DC orchestration (`prepare`, `solve_dc`) |
| `solver_td.py` | Time-domain mixin: `preprocess_sources`, `solve_quasi_static`, `prepare_transient`, `solve_transient` |
| `solver_adjoint.py` | `analyze_adjoint_static`, `analyze_adjoint` |
| `tile_worker.py` / `tile_worker_td.py` / `tile_worker_peak.py` / `tile_worker_adjoint.py` | Per-tile actor wrapping `BlockMatrixSystem` (mixins for transient, peak tracking, adjoint) |
| `tile_parsing.py` | Stateless parsing: `TileData`, `_parse_tile_ckt`, `_iter_instance_sources`, `_parse_node_xy` |
| `model.py` | `DistributedPowerGridModel`, `ParsedTileBundle`, `create_distributed_model` |
| `parser.py` | `DistributedNetlistParser` |
| `result.py` / `result_factorization.py` | Context/result dataclasses; `factor`/`release`/`save`/`load`/`refactor` |
| `backend.py` | `LocalBackend`, `RayBackend` |
| `heatmap.py` | Tile-parallel pre-binned stripe heatmap pipeline |
| `cli.py` | `python -m distributed {parse,solve,run}` |

**Context lifecycle (DC + transient are independent):**
```python
dc_ctx = solver.prepare()                                  # builds + factors
result = solver.solve_dc(dc_ctx)
trans_ctx = solver.prepare_transient(dt=100e-12, method='BE')
sources = solver.preprocess_sources(time_step=100e-12, t_end=10e-9)
trans_result = solver.solve_transient(trans_ctx, dc_context=dc_ctx)  # OR ic_voltages=...
trans_ctx.release(); dc_ctx.release()                      # caller owns both
```

Heatmap pipeline: `get_layer_metadata()` → `build_global_bin_spec()` → `prebin_tile()` (per tile, picklable, `map_func`-able) → `merge_tile_prebins()` → `render_from_prebinned_stripe_data()`.

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

### Distributed solver
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
- **Test netlists**: `netlist/netlist_test/` (small PDN, integration), `netlist/netlist_small/` (minimal unit fixtures), `netlist/netlist_sampled/` (distributed benchmark)
- **API exports**: each `src/<pkg>/__init__.py`
- **Copilot equivalent**: `.github/copilot-instructions.md` (overlapping content; this file is authoritative for Claude)

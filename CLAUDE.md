# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Static and dynamic IR-drop analysis prototype for multi-layer power grids. Supports both synthetic grids and real PDN netlists. Includes quasi-static (batch DC) and transient RC analysis.

**Source Layout (`src/` packages, installed via `pip install -e .`):**
1. **`src/graph/`** - Rustworkx graph wrappers and conversion utilities
2. **`src/model/`** - `UnifiedPowerGridModel`, adapters, factory functions
3. **`src/solver/`** - Flat, hierarchical, coupled, and tiled solvers
4. **`src/analysis/`** - Dynamic, transient, adjoint analysis; PWL smoothing
5. **`src/parser/`** - SPICE-like netlist parsing (`NetlistParser`)
6. **`src/distributed/`** - Distributed DDM solver (tile-based domain decomposition), includes `heatmap.py` for tile-parallel pre-binned stripe heatmaps
7. **`src/visualization/`** - Plotters (`UnifiedPlotter`, `DynamicPlotter`, `PDNPlotter`)
8. **`src/legacy/`** - Original synthetic grid modules (originally `irdrop/`)
9. **`src/reports/`** - Shared report generators (floating nodes, top-K IR-drop)

## Commands

```bash
# Install (editable)
uv pip install -e ".[test]"

# Run unit tests (fast, <160s)
pytest -m unit

# Run a specific integration test file
pytest tests/distributed/test_distributed_integration.py -v

# Run ALL integration tests (~6 min, slow — only when needed)
pytest -m integration

# Run everything (~10 min, slow — only as a final check)
pytest

# Run specific test module
pytest tests/solver/test_hierarchical_solver.py
pytest tests/parser/test_pdn_parser.py
pytest tests/distributed/test_distributed_solver.py

# Run single test
pytest tests/legacy/test_irdrop.py::TestIRDrop::test_no_load_currents_all_pad_voltage -v

# Run CLI tools
python -m parser.pdn_parser ./netlist/netlist_test --net VDD
python -m solver.pdn_solver --input graph.pkl --net VDD --output ./results
python -m analysis.dynamic_irdrop_decomposition ./netlist/netlist_test --net VDD --end-time 100ns --dt 100ps

# Run distributed solver with heatmaps
python -m distributed solve ./netlist/netlist_sampled/distributed_pkl --backend ray --plot --verbose

# Run distributed quasi-static analysis
python -m distributed solve ./netlist/netlist_sampled/distributed_pkl --mode quasi-static --t-end 100ns --n-points 11 --verbose

# Run distributed transient analysis
python -m distributed solve ./netlist/netlist_sampled/distributed_pkl --mode transient --t-end 10ns --dt 100ps --verbose
```

> **Test priority:** First run individual unit test files related to the affected
> area, then `pytest -m unit` for full unit coverage. Run individual integration
> test files only when changes affect that area. Run bare `pytest` only as a
> final validation step, not during intermediate steps.

## Architecture

### Data Flow
- **Synthetic**: `generate_power_grid()` -> `create_model_from_synthetic(G, pads, vdd)` -> `UnifiedIRDropSolver`
- **PDN**: `NetlistParser.parse()` -> `create_model_from_pdn(graph, net_name)` -> `UnifiedIRDropSolver`
- **Multi-Net**: `NetlistParser.parse()` -> `create_multi_net_models(graph)` -> iterate models
- **Distributed**: `DistributedNetlistParser.parse_and_dump()` -> `ParsedTileBundle` -> `create_distributed_model(bundle)` -> `DistributedDDMSolver`
- **Distributed QS**: `ctx = prepare()` -> `preprocess_sources()` -> `solve_quasi_static(ctx, t_array)` -> `DistributedQuasiStaticResult` (peaks lazy on workers)
- **Distributed Transient**: `dc_ctx = prepare()` -> `trans_ctx = prepare_transient(dt, method)` -> `preprocess_sources()` -> `solve_transient(trans_ctx, dc_context=dc_ctx)` -> `DistributedTransientResult`
- **Distributed Transient (ic_voltages)**: `trans_ctx = prepare_transient(dt, method)` -> `solve_transient(trans_ctx, ic_voltages=dc_result.voltages)` -> `DistributedTransientResult`
- **Distributed Adjoint**: `dc_ctx = prepare()` -> `solver.analyze_adjoint_static(victim, T, dc_ctx)` or `solver.analyze_adjoint(victim, T, trans_ctx, trans_result)` -> `AdjointAttribution`

**Key Constraint:** Pads (voltage sources) are Dirichlet BCs at Vdd, eliminated via Schur complement. LU factorization cached for batch solves.

### Module Structure

```
src/
├── graph/
│   ├── rx_graph.py             # RustworkxMultiDiGraphWrapper
│   ├── rx_algorithms.py        # Graph algorithms (dijkstra, components, etc.)
│   └── converter.py            # NetworkX <-> Rustworkx conversion
├── model/
│   ├── unified_model.py        # UnifiedPowerGridModel, grid decomposition
│   ├── factory.py              # create_model_from_* functions
│   ├── node_adapter.py         # NodeInfoExtractor
│   ├── edge_adapter.py         # EdgeInfoExtractor, ElementType
│   └── solver_results.py       # Result/context data classes
├── solver/
│   ├── unified_solver.py       # UnifiedIRDropSolver (orchestration)
│   ├── coupled_system.py       # BlockMatrixSystem, Schur math, grounded cap diags, re-exports
│   ├── coupled_operators.py    # SchurComplementOperator, CoupledSystemOperator, preconditioners
│   ├── interface_assembly.py   # Distributed interface assembly, island detection, package matrices
│   ├── current_aggregation.py  # CurrentAggregator
│   ├── tiling.py               # TileManager for parallel tiling
│   ├── unified_partitioner.py  # Layer/spatial partitioning
│   ├── effective_resistance.py # Pairwise effective resistance
│   ├── statistics.py           # Netlist statistics
│   └── pdn_solver.py           # Standalone PDN DC solver CLI
├── analysis/
│   ├── dynamic_solver.py       # DynamicIRDropSolver (batch DC)
│   ├── transient_solver.py     # TransientIRDropSolver (RC)
│   ├── adjoint_sensitivity.py  # IR-drop attribution
│   ├── farfield_analysis.py    # Far-field to local boundary coupling analysis
│   ├── pwl_smoothing.py        # PWLSmoother
│   ├── vectorized_sources.py   # VectorizedCurrentSources
│   └── dynamic_irdrop_decomposition.py  # CLI for dynamic IR-drop decomposition
├── parser/
│   ├── netlist.py              # NetlistParser (main entry point)
│   ├── pdn_parser.py           # CLI entry point for PDN parsing
│   ├── sampled_netlist.py      # Sampled multi-tile netlist generator
│   ├── spice_lexer.py          # SPICE element line tokenizer
│   ├── current_sources.py      # CurrentSource, Pulse, PWL
│   ├── graph_builder.py        # Builds rustworkx graph from tokens
│   ├── metadata.py             # Net voltage, vsrc metadata
│   ├── parallel.py             # Parallel tile parsing
│   └── edge_attrs.py           # Memory-optimized edge attributes
├── distributed/
│   ├── model.py                # DistributedPowerGridModel, ParsedTileBundle
│   ├── solver.py               # DistributedDDMSolver (DC + time-domain mixin)
│   ├── solver_td.py            # Time-domain mixin: preprocess_sources, solve_quasi_static, solve_transient
│   ├── parser.py               # DistributedNetlistParser
│   ├── tile_worker.py          # Per-tile BlockMatrixSystem actor (+ time-domain mixin)
│   ├── tile_worker_td.py       # Time-domain mixin: VCS, transient factor/RHS, peak tracking
│   ├── tile_parsing.py         # TileData, stateless parsing functions (_parse_tile_ckt, etc.)
│   ├── backend.py              # Local/Ray compute backends
│   ├── heatmap.py              # Distributed stripe heatmap pipeline (prebin/merge/render)
│   ├── cli.py                  # CLI: python -m distributed {solve,run,parse} with --mode dc/quasi-static/transient
│   ├── result.py               # Result/context classes (DC, quasi-static, transient) + dataclasses
│   └── result_factorization.py # Factorization, save/load/refactor logic for context classes
├── reports/
│   ├── floating_nodes.py          # Floating nodes detection and reporting
│   └── topk_irdrop.py             # Top-K worst IR-drop report (shared by flat and distributed)
├── visualization/
│   ├── unified_plotter.py      # UnifiedPlotter (voltage/IR-drop heatmaps)
│   ├── dynamic_plotter.py      # DynamicPlotter (time-domain results)
│   ├── pdn_plotter.py          # PDNPlotter (layer-wise heatmaps)
│   └── stripe_heatmap.py       # Stripe-based heatmap visualization
└── legacy/
    ├── generate_power_grid.py  # Synthetic K-layer grid generator
    ├── power_grid_model.py     # Original PowerGridModel
    ├── solver.py               # IRDropSolver
    ├── stimulus.py             # StimulusGenerator
    ├── grid_partitioner.py     # GridPartitioner
    ├── effective_resistance.py # EffectiveResistanceCalculator
    ├── regional_voltage_solver.py # Regional IR-drop via effective resistance
    └── plot.py                 # plot_voltage_map, plot_ir_drop_map
```

### Key Classes (one-line summaries)

- **UnifiedPowerGridModel**: Handles both NodeID and string nodes; auto-detects floating islands
- **UnifiedIRDropSolver**: Flat, hierarchical, coupled, and tiled solves with batch support
- **BlockMatrixSystem / SchurComplementOperator / CoupledSystemOperator**: Coupled solver internals
- **CurrentAggregator**: Distributes load currents to ports (shortest-path or effective resistance)
- **TileManager**: Tile generation, connectivity validation, and result merging
- **DynamicIRDropSolver**: Quasi-static analysis via batch DC solves at discrete time points
- **TransientIRDropSolver**: Transient RC analysis (Backward Euler or Trapezoidal)
- **AdjointSensitivitySolver**: IR-drop attribution to aggressor current sources
- **PWLSmoother**: Analytical triangular low-pass filter for waveform preprocessing
- **NetlistParser**: SPICE-like tile-based netlist parsing with parallel support
- **ParsedTileBundle**: Lightweight coordinator-side metadata for distributed model creation (no tile data)
- **DistributedTopologyContext**: Immutable topology shared by DC and transient contexts
- **DistributedSolverContext**: Active DC context with `factor()` / `release()` / `save()` / `load()` / `refactor()`
- **DistributedTransientContext**: Active transient context with same lifecycle methods
- **DistributedSmoothedSources**: Coordinator-side handle for preprocessed VCS (data lives on workers)
- **DistributedQuasiStaticResult**: Lazy peak collection from workers; `as_flat()` / `as_per_tile()` / `dump()`
- **DistributedTransientResult**: Extends quasi-static result with RC transient metadata
- **generate_topk_report**: Shared top-K IR-drop report writer (used by both PDNSolver and DistributedDDMSolver)

### Legacy Module (src/legacy/)
- `generate_power_grid()`: Creates K-layer resistor mesh with `NodeID` keys
- `PowerGridModel`, `IRDropSolver`: Original classes (prefer unified versions in `src/solver/`)
- `GridPartitioner`: Structured slab partitioning along via rows/columns

## Critical Conventions

### Node Types
- **Synthetic**: `NodeID(layer, idx)` frozen dataclass keys the graph
- **PDN**: String node names like `'1000_2000_M1'`, `'VDD_vsrc'`, `'0'` (ground)

### Unit System (PDN)
- Resistance: kOhm, Capacitance: fF, Inductance: nH, Current: mA
- Conductance matrix in mS (milli-Siemens) for self-consistent G*V = I

### Current Sign Convention (CRITICAL)
- **Input**: Positive current = sink drawing from grid (`currents[node] = +1.0 mA`)
- **Internal**: Solver negates for nodal equation
- **IR-drop**: Always `Vdd - V_node` (positive = voltage dropped below Vdd)

### Common Pitfalls
- **Plotting**: `plot_ir_drop_map(G, voltages, vdd=1.0, ...)` requires scalar `vdd`, NOT pad list
- **Stimulus area**: `StimulusGenerator(graph=G, ...)` must pass graph if using `area` parameter
- **R_eff queries**: Pad nodes rejected in pairwise calculations (raises `ValueError`)
- **PDN current extraction**: Use `model.extract_current_sources()` to get load currents from I-type edges
- **Headless plotting**: Use `show=False` for batch/headless runs. Matplotlib backend is set to `Agg` in test runners.
- **Legacy pickle files**: Old `pdn_graph.pkl` files contain NetworkX graphs. Use `ensure_rustworkx_graph()` to convert before creating models.
- **Edge elem_name access**: With optimized edges (default), `elem_name` is None for most resistors. Use `data.get('elem_name', '')` not `data['elem_name']`.
- **Instance model iteration**: Use `_iter_instance_sources()` from `distributed/tile_worker.py` to iterate filtered instanceModels entries — don't duplicate the gzip/net-filter logic.
- **np.digitize binning**: Use `valid_mask` filter for out-of-range indices, NOT `np.clip`. Clamping corrupts edge bin values.
- **Current heatmaps**: Only plot layers that have current sources (check for non-zero bins). Upper metal layers typically have none.
- **Distributed circular imports**: `distributed/parser.py` cannot import from `distributed/model.py` at module level (model.py already imports from parser.py). Use lazy imports inside functions.
- **Transient Dirichlet RHS**: In the distributed transient time loop, use `rhs_dirichlet_G` (G-only), NOT `rhs_dirichlet_interface` (A-based, includes cap terms). BE: `+rhs_d_G`, TR: `+2*rhs_d_G`. Pad capacitance history cancels because pads hold constant voltage.
- **Island detection with caps**: Capacitive edges do NOT contribute to connectivity for island detection. A node connected to pads only via caps is floating. But DO filter cap edges when removing island nodes.
- **Context lifecycle**: `solve_dc(ctx)`, `solve_quasi_static(ctx)`, `solve_transient(trans_ctx, dc_context=dc_ctx)` — context is REQUIRED (first positional). Caller creates via `prepare()` / `prepare_transient()` and must `release()` when done.
- **Transient IC paths**: `solve_transient` takes `dc_context` OR `ic_voltages` (mutually exclusive). `dc_context` does a DC solve for IC; `ic_voltages` skips DC entirely.
- **prepare_transient() is independent**: Does NOT internally call `prepare()`. Caller manages DC and transient contexts separately.
- **solve_transient does NOT release dc_context**: Caller owns the lifecycle of both contexts.
- **Context save/load**: `save()` must be called BEFORE `release()` (release clears S_global). After `load()`, call `refactor()` to rebuild coordinator LU from saved S_global. Workers need separate `factor()`.
- **Ray worker globals**: Module-level globals (CHOLMOD settings, regularization) do NOT propagate to Ray workers (separate Python processes). Use `TileWorker.configure(settings)` called once during `create_distributed_model` to push settings to workers. CHOLMOD backend settings (`use_cholmod`, `cholmod_mode`, `cholmod_ordering`, `cholmod_use_long`) are now propagated automatically.
- **Tile matrix SPD**: Per-tile full matrix `[[G_ii, G_ip], [G_pi, G_pp]]` may be PSD (not SPD) for tiles without ground connections. `_compute_schur_partial()` adds 1e-5 mS regularization to port diagonals and subtracts it from S after extraction. `G_ii` alone is always SPD (diagonal includes connections to ports).
- **Partial Cholesky Schur path**: `compute_explicit_schur(block_system)` automatically uses partial Cholesky when CHOLMOD backend is active; falls back to chunked multi-RHS when splu is used. Factors full `[interior, ports]` matrix and extracts `S = L22 @ L22.T`. Sets `lu_ii` via solve_L/Lt truncation trick. CHOLMOD-only.
- **`build_block_system_from_edges` vs `extract_block_matrices`**: The tile worker uses `build_block_system_from_edges` (no `exclude_port_to_port` param — includes all edges). The flat coupled solver uses `extract_block_matrices` (has `exclude_port_to_port` flag). Don't confuse them.
- **`solve_quasi_static` default smoothing**: Calling without `smoothed_sources` triggers `preprocess_sources(smooth=True)`, silently overwriting `_active_sources` on workers. Always pass a `smoothed_sources` handle if VCS is already initialized.
- **Notebook cwd for Ray**: Ray workers inherit the driver's cwd. Notebooks must `os.chdir` to the project root before creating the model so tile metadata relative paths resolve. Pattern: `os.chdir(Path(__file__).parent.parent if '__file__' in dir() else Path('..'))`
- **`_parse_node_xy` shared utility**: Use `from distributed.tile_parsing import _parse_node_xy` for PDN coordinate parsing (`'1000_2000_M1'` → `(1000.0, 2000.0)`). Do not duplicate this function.

## Typical Workflow Patterns

### PDN Netlist Analysis (Recommended)
```python
from parser.netlist import NetlistParser
from model.factory import create_model_from_pdn
from solver.unified_solver import UnifiedIRDropSolver

parser = NetlistParser('./netlist/netlist_test', validate=True)
graph = parser.parse()
model = create_model_from_pdn(graph, 'VDD')  # vdd auto-extracted
load_currents = model.extract_current_sources()

solver = UnifiedIRDropSolver(model)
result = solver.solve(load_currents)
print(f"Max IR-drop: {max(result.ir_drop.values()):.4f} V")
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

## File Landmarks

- **Notebooks**: `notebooks/irdrop_decomposition_pdn.ipynb`, `notebooks/irdrop_decomposition.ipynb`, `notebooks/irdrop_decomposition_unified_model.ipynb`
- **Tests**: `tests/{graph,model,solver,analysis,parser,distributed,visualization,legacy}/test_*.py`
- **API exports**: `src/*/__init__.py` (package-level public APIs)
- **Scripts**: `scripts/analysis/`, `scripts/solver/`, `scripts/parser/`
- **Reference**: `DEPRECATION.md` (historical old->new import mappings)

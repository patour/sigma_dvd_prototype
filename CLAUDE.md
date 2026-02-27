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
6. **`src/distributed/`** - Distributed DDM solver (tile-based domain decomposition)
7. **`src/visualization/`** - Plotters (`UnifiedPlotter`, `DynamicPlotter`, `PDNPlotter`)
8. **`src/legacy/`** - Original synthetic grid modules (originally `irdrop/`)

## Commands

```bash
# Install (editable)
uv pip install -e ".[test]"

# Run all tests (fast, ~984 tests)
pytest

# Run slow integration tests
pytest tests/solver/test_hierarchical_integration.py tests/analysis/test_dynamic_integration.py tests/parser/test_pdn_integration.py

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
```

## Architecture

### Data Flow
- **Synthetic**: `generate_power_grid()` -> `create_model_from_synthetic(G, pads, vdd)` -> `UnifiedIRDropSolver`
- **PDN**: `NetlistParser.parse()` -> `create_model_from_pdn(graph, net_name)` -> `UnifiedIRDropSolver`
- **Multi-Net**: `NetlistParser.parse()` -> `create_multi_net_models(graph)` -> iterate models

**Key Constraint:** Pads (voltage sources) are Dirichlet BCs at Vdd, eliminated via Schur complement. LU factorization cached for batch solves.

### Source Packages (src/) - Use These for New Code

**Module Structure:**
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
│   ├── coupled_system.py       # Block matrices, Schur complement
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
│   ├── pwl_smoothing.py        # PWLSmoother
│   └── vectorized_sources.py   # VectorizedCurrentSources
├── parser/
│   ├── netlist.py              # NetlistParser (main entry point)
│   ├── spice_lexer.py          # SPICE element line tokenizer
│   ├── current_sources.py      # CurrentSource, Pulse, PWL
│   ├── graph_builder.py        # Builds rustworkx graph from tokens
│   ├── metadata.py             # Net voltage, vsrc metadata
│   ├── parallel.py             # Parallel tile parsing
│   └── edge_attrs.py           # Memory-optimized edge attributes
├── distributed/
│   ├── model.py                # DistributedPowerGridModel
│   ├── solver.py               # DistributedDDMSolver
│   ├── parser.py               # DistributedNetlistParser
│   ├── tile_worker.py          # Per-tile BlockMatrixSystem actor
│   ├── backend.py              # Local/Ray compute backends
│   └── result.py               # Result/context dataclasses
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
    └── plot.py                 # plot_voltage_map, plot_ir_drop_map
```

**Key Classes:**
- **UnifiedPowerGridModel**: Handles both NodeID and string nodes; auto-detects floating islands
- **UnifiedIRDropSolver**: `solve()` for flat, `solve_hierarchical()` for layer-decomposed (approximate), `solve_hierarchical_coupled()` for exact coupled solve, `solve_hierarchical_tiled()` for parallel tiled solving. Supports batch solving via `prepare_*()` / `solve_*_prepared()` methods.
- **BlockMatrixSystem**: Block-partitioned conductance matrix (port/interior splits)
- **SchurComplementOperator**: Matrix-free Schur complement for coupled solver
- **CoupledSystemOperator**: Full coupled top-grid + Schur complement operator
- **CurrentAggregator**: Distributes load currents to ports using shortest-path or effective resistance weighting
- **TileManager**: Manages tile generation, connectivity validation, and result merging for tiled solving
- **NodeInfoExtractor / EdgeInfoExtractor**: Adapt different graph representations
- **UnifiedStatistics**: Compute netlist statistics (node/edge counts, R/C/L/I totals)
- **UnifiedPartitioner**: Layer-based and spatial grid partitioning
- **UnifiedPlotter**: Voltage/IR-drop heatmap generation
- **UnifiedEffectiveResistanceCalculator**: Pairwise and single-node effective resistance
- **DynamicIRDropSolver**: Quasi-static analysis via batch DC solves at discrete time points
- **TransientIRDropSolver**: Transient RC analysis with Backward Euler or Trapezoidal integration
- **PWLSmoother**: Analytical triangular low-pass filter for waveform preprocessing
- **SmoothedWaveformCache**: Cached smoothed waveforms for reuse across analyses
- **DynamicPlotter**: Heatmap and time series plotting for dynamic analysis results

**Factory Functions:**
```python
from model.factory import create_model_from_synthetic, create_model_from_pdn, create_multi_net_models

# From synthetic grid
model = create_model_from_synthetic(G, pads, vdd=1.0)

# From PDN netlist (single net) - vdd auto-extracted
model = create_model_from_pdn(graph, 'VDD')

# From PDN netlist (all nets)
models = create_multi_net_models(graph)  # {'VDD': model, 'VSS': model}

# Eager factorization (for backward compat or flat solver-only workflows)
model = create_model_from_pdn(graph, 'VDD', lazy_factor=False)
```

**Lazy Factorization (default):**
Model creation uses `lazy_factor=True` by default, deferring LU factorization until the first flat solve. This provides ~4.6x faster model creation when using hierarchical solvers (which build their own systems). Set `lazy_factor=False` for backward compatibility or when using only flat solves.

**Result Data Classes:**
- `UnifiedSolveResult`: Basic solve result with voltages, ir_drop, metadata
- `UnifiedHierarchicalResult`: Hierarchical result with port_nodes, port_voltages, port_currents, aggregation_map
- `UnifiedCoupledHierarchicalResult`: Coupled solver result with iterations, final_residual, converged, timings
- `TiledBottomGridResult`: Tiled solve result with tiles, per_tile_solve_times, validation_stats

**Solver Context Classes (for batch solving):**
- `FlatSolverContext`: Caches reduced system LU factorization for repeated flat solves
- `HierarchicalSolverContext`: Caches top/bottom grid systems and shortest-path distances
- `CoupledHierarchicalSolverContext`: Caches block matrices, Schur complement operator, preconditioner
- `TiledHierarchicalSolverContext`: Caches top-grid system, tile structure, path distances

**Enums:**
- `GridSource.SYNTHETIC`, `GridSource.PDN_NETLIST`: Source type detection
- `ElementType.RESISTOR`, `ElementType.CAPACITOR`, `ElementType.INDUCTOR`, `ElementType.CURRENT_SOURCE`

**Graph Converter (for legacy pickle files):**
```python
from graph.converter import detect_graph_type, ensure_rustworkx_graph

# Detect graph type from pickle
graph_type = detect_graph_type(graph)  # Returns 'networkx', 'rustworkx', or 'unknown'

# Auto-convert NetworkX to Rustworkx if needed
graph = ensure_rustworkx_graph(graph)  # No-op if already Rustworkx
```

### Parser Module (src/parser/)
- **NetlistParser** (`parser.netlist`): Parses SPICE-like tile-based netlists with gzip support (parallel parsing available)
- **PDNSolver** (`solver.pdn_solver`): Standalone DC solver (use if you don't need unified interface)
- **PDNPlotter** (`visualization.pdn_plotter`): Layer-wise heatmap generation with advanced features
- **parallel.py** (`parser.parallel`): Worker functions and data classes for parallel tile parsing
- **edge_attrs.py** (`parser.edge_attrs`): Memory-optimized edge attribute classes (ResistorEdge, CapacitorEdge, etc.)

**Current Source Data Structures (from instanceModels*.sp):**
- `InstanceInfo`: Parsed instance name with net/pin/tile location info
- `Pulse`: Pulse waveform with `evaluate(time)` and `get_dc()` methods
- `PWL`: Piece-wise linear waveform with `evaluate(time)` and `get_dc()` methods
- `CurrentSource`: Full current source with DC value, static_value, pulses, PWLs

**Accessing Current Source Data:**
```python
# By default, parser stores raw CurrentSource objects (memory efficient)
graph = parser.parse()
raw_sources = graph.graph.get('_instance_sources_objects', {})

# Access CurrentSource objects directly
for name, src in raw_sources.items():
    static_ma = src.get_static_current()      # DC analysis
    current_at_t = src.get_current_at_time(1e-9)  # Transient at 1ns

# For portable pickle files, use store_instance_sources=True (serializes to dicts)
parser = NetlistParser('./netlist_dir', store_instance_sources=True)
graph = parser.parse()
instance_sources = graph.graph.get('instance_sources', {})  # Serialized dicts
```

**Memory Optimization for Large Netlists:**
The default `store_instance_sources=False` avoids serializing CurrentSource objects to dicts, saving ~60% parse-time memory for large netlists (1.7GB -> 1.1GB for 1M sources). The dynamic/transient solvers automatically handle both formats.

**Edge Attribute Memory Optimization:**
By default, edge attributes use specialized slotted dataclasses (`parser/edge_attrs.py`) instead of dicts, reducing memory by ~95% per edge. Critical for 100M+ edge netlists (~65 GB → ~4 GB).

```python
from parser.graph_builder import get_use_optimized_edges, set_use_optimized_edges

# Check current mode (default: True)
print(get_use_optimized_edges())  # True

# Disable for backward compatibility or small netlists
set_use_optimized_edges(False)
```

**Edge Classes:**
- `ResistorEdge`: Die resistors (no elem_name stored) - 99.9% of resistors
- `ResistorEdgeWithName`: Package resistors matching `vsrc_resistor_pattern` (e.g., 'rs')
- `CapacitorEdge`, `InductorEdge`, `CurrentSourceEdge`, `VoltageSourceEdge`

**Important:** With optimized edges, `elem_name` is only stored for:
- Voltage sources (always)
- Resistors matching `vsrc_resistor_pattern` (default 'rs') for vsrc node identification

Use `.get('elem_name', '')` instead of `['elem_name']` for safe access:
```python
for u, v, data in graph.edges(data=True):
    elem_name = data.get('elem_name', '')  # Safe: returns '' if not stored
    # NOT: data['elem_name']  # May raise KeyError for die resistors
```

**Runtime Trade-off:** Computed properties (`.tile_id`, `.net_type`) are ~4-5x slower than dict access due to on-the-fly unpacking. For hot loops accessing same edges repeatedly, either cache values locally or use `set_use_optimized_edges(False)` for small netlists.

**Pickle Compatibility:**
Both modes support pickle. The difference is portability:
- `store_instance_sources=False` (default): Pickle works but requires `parser` module when loading
- `store_instance_sources=True`: Pickle is portable (no module dependency), better for long-term storage/sharing

**Parallel Parsing (for large netlists with many tiles):**
```python
# Enable parallel parsing for ~6-8x speedup on 100+ tiles
parser = NetlistParser('./netlist_dir', parallel=True, n_workers=8)
graph = parser.parse()

# With net filter and custom chunk size
parser = NetlistParser('./netlist_dir', parallel=True, n_workers=4,
                       net_filter='VDD', chunk_size=10000)
```

Parallel parsing uses `multiprocessing.Pool` with:
- Memory-mapped file access for plain text files (gzip fallback for compressed)
- Chunk-based processing within large tiles
- Bulk graph operations for efficient merge phase
- Full equivalence with sequential parsing (same graph output)

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

### Hierarchical Solve (Layer Decomposition)
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

**Hierarchical Solver Parameters:**
- `partition_layer`: Layer name (string like `'M2'`) or integer index
- `top_k`: Number of nearest ports per load for current aggregation (default 5)
- `weighting`: `"shortest_path"` (default) or `"effective"` (more accurate but slower)
- `rmax`: Maximum resistance distance for shortest_path weighting
- `use_fast_builder`: If True (default), use vectorized subgrid builder (~10x speedup)

### Coupled Hierarchical Solve (Exact)
For exact solutions (up to iterative tolerance) without current aggregation approximation:

```python
# Coupled solve using matrix-free Schur complement
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

**Coupled vs Uncoupled Hierarchical:**
- **Uncoupled (`solve_hierarchical`)**: Approximates port currents via weighted distribution, then solves top/bottom grids independently. Fast but introduces ~0.5 mV error from current aggregation.
- **Coupled (`solve_hierarchical_coupled`)**: Solves the full coupled system iteratively using matrix-free Schur complement. Exact up to solver tolerance (~0.02 µV error). Slower but highly accurate.

**Coupled Solver Parameters:**
- `solver`: Iterative solver type:
  - `'cg'`: Conjugate Gradient - optimal for SPD systems, recommended for large problems
  - `'gmres'` (default): GMRES - robust, works for non-symmetric systems
  - `'bicgstab'`: BiCGSTAB - often faster than GMRES
- `tol`: Residual tolerance for iterative solver (default 1e-8)
- `maxiter`: Maximum iterations before raising RuntimeError (default 500)
- `preconditioner`: Preconditioner type:
  - `'block_diagonal'` (default): Fast, diagonal approximation
  - `'ilu'`: Incomplete LU - better for ill-conditioned systems
  - `'amg'`: Algebraic Multigrid - best for large problems (requires pyamg)
  - `'none'`: No preconditioning

**Recommended configurations:**
- Small problems (<100K nodes): `solver='bicgstab', preconditioner='ilu'`
- Large problems (>1M nodes): `solver='cg', preconditioner='amg'` (O(1) iterations)

**Note:** CG requires an SPD preconditioner. Use `'amg'` or `'block_diagonal'` with CG, not `'ilu'`.

**UnifiedCoupledHierarchicalResult Fields:**
- All fields from `UnifiedHierarchicalResult` plus:
- `iterations`: Number of iterative solver iterations
- `final_residual`: Final residual norm
- `converged`: Boolean indicating convergence
- `preconditioner_type`: Preconditioner used
- `timings`: Dict with 'factor_bottom', 'build_rhs', 'iterative_solve', 'recover_bottom'

### Tiled Hierarchical Solve (PDN only)
For large PDN grids, exploit spatial locality by tiling the bottom-grid:

```python
# Tiled solve with 2x2 grid and 20% halo
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

**NOTE:** Tiled solving is only supported for PDN graphs (string node names). Synthetic grids with `NodeID` raise `ValueError`.

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

### Batch Solving (Multiple Current Scenarios)
For solving multiple current scenarios efficiently, use the prepare/solve_prepared pattern to cache expensive precomputation (LU factorization, block matrices, operators):

```python
from solver.unified_solver import UnifiedIRDropSolver
from model.solver_results import FlatSolverContext

solver = UnifiedIRDropSolver(model)

# Prepare once (expensive: builds and factors matrices)
ctx = solver.prepare_flat()

# Solve multiple scenarios (cheap: reuses cached factorization)
for scenario in current_scenarios:
    result = solver.solve_prepared(ctx, scenario)
    print(f"Max IR-drop: {max(result.ir_drop.values()):.4f} V")
```

**Available prepare/solve_prepared methods:**
| Method | Context Class | Use Case |
|--------|---------------|----------|
| `prepare_flat()` | `FlatSolverContext` | Multiple flat solves |
| `prepare_hierarchical()` | `HierarchicalSolverContext` | Multiple hierarchical solves |
| `prepare_hierarchical_coupled()` | `CoupledHierarchicalSolverContext` | Multiple coupled solves |
| `prepare_hierarchical_tiled()` | `TiledHierarchicalSolverContext` | Multiple tiled solves |

**Hierarchical batch solving example:**
```python
# Prepare hierarchical solver (caches top/bottom systems, path distances)
ctx = solver.prepare_hierarchical(partition_layer='M2', top_k=5)

# Solve multiple scenarios
for scenario in current_scenarios:
    result = solver.solve_hierarchical_prepared(ctx, scenario)
```

**Coupled batch solving example:**
```python
# Prepare coupled solver (caches block matrices, Schur complement, preconditioner)
ctx = solver.prepare_hierarchical_coupled(partition_layer='M2', preconditioner='block_diagonal')

# Solve multiple scenarios
for scenario in current_scenarios:
    result = solver.solve_hierarchical_coupled_prepared(ctx, scenario)
```

**NOTE:** All standard `solve*()` methods also accept an optional `context` parameter for backward compatibility. If not provided, they create a temporary context internally.

### Dynamic IR-Drop Analysis (Time-Domain)

For time-varying currents, use the dynamic solvers which evaluate current sources at discrete time points.

#### Quasi-Static Analysis (Batch DC Solves)
```python
from parser.netlist import NetlistParser
from model.factory import create_model_from_pdn
from analysis.dynamic_solver import DynamicIRDropSolver

parser = NetlistParser('./netlist_dir')
graph = parser.parse()
model = create_model_from_pdn(graph, 'VDD')

solver = DynamicIRDropSolver(model, graph)
result = solver.solve_quasi_static(
    t_start=0, t_end=100e-9, n_points=101,
    method='flat',           # 'flat' or 'hierarchical'
    n_worst_nodes=10,        # Track N worst-case nodes
    track_nodes=['node1'],   # Store full waveforms for these nodes
)

print(f"Peak IR-drop: {result.peak_ir_drop*1000:.2f} mV at t={result.peak_ir_drop_time*1e9:.2f} ns")
print(f"Peak node: {result.peak_ir_drop_node}")
```

**QuasiStaticResult Fields:**
- `t_array`: Time points array
- `peak_ir_drop`, `peak_ir_drop_time`, `peak_ir_drop_node`: Global peak info
- `worst_nodes`: List of (node, max_drop, time) for top N worst nodes
- `max_ir_drop_per_time`: Max IR-drop at each time step
- `total_current_per_time`, `total_vsrc_current_per_time`: Aggregate currents
- `peak_ir_drop_per_node`, `peak_current_per_node`: Spatial peaks for heatmaps
- `tracked_waveforms`, `tracked_ir_drop`: Waveforms for selected nodes

#### Transient RC Analysis (with Capacitance)
```python
from analysis.transient_solver import TransientIRDropSolver, IntegrationMethod

solver = TransientIRDropSolver(model, graph)
result = solver.solve_transient(
    t_start=0, t_end=100e-9, dt=0.1e-9,
    method=IntegrationMethod.BACKWARD_EULER,  # or TRAPEZOIDAL
    n_worst_nodes=10,
    track_nodes=['node1'],
)

# Capacitors provide smoothing, so peak may be lower than quasi-static
print(f"Transient peak: {result.peak_ir_drop*1000:.2f} mV")
```

**Transient vs Quasi-Static:**
- **Quasi-static**: Ignores capacitance, solves independent DC problems at each time point. Faster, useful for steady-state approximation.
- **Transient**: Includes capacitance via implicit time integration (Backward Euler or Trapezoidal). Captures decoupling effects but slower.

**TransientResult Fields:**
- Same as QuasiStaticResult, plus `integration_method`
- Timings include `build_rc`, `factor`, `solve`

#### Dynamic Analysis Plotting
```python
from visualization.dynamic_plotter import DynamicPlotter

# Peak IR-drop heatmap (worst IR-drop each node saw across time)
DynamicPlotter.plot_peak_ir_drop_heatmap(
    model, result, layer='M1',
    title='Peak IR-Drop During Transient',
    save_path='peak_ir_drop.png'
)

# Peak current heatmap
DynamicPlotter.plot_peak_current_heatmap(
    model, result, layer='M1',
    save_path='peak_current.png'
)

# Time series of aggregate metrics
DynamicPlotter.plot_time_series(
    result, metrics=['max_ir_drop', 'total_current', 'vsrc_current'],
    save_path='time_series.png'
)

# Node waveforms (for tracked nodes)
DynamicPlotter.plot_node_waveforms(
    result, plot_ir_drop=True,
    save_path='waveforms.png'
)
```

#### PWL Waveform Smoothing (Preprocessing)

Apply analytical triangular low-pass filter to current waveforms before dynamic/transient analysis. This removes high-frequency content that causes numerical noise while preserving DC and low-frequency behavior.

**Algorithm:** Convolves each PWL segment with a triangular window (width = 2×time_step) using exact closed-form integration. Pulse waveforms are first converted to PWL. A compaction phase removes redundant collinear points after filtering.

**Basic Usage (Automatic):**
```python
from analysis.dynamic_solver import DynamicIRDropSolver

solver = DynamicIRDropSolver(model, graph)

# Preprocess waveforms (returns VectorizedCurrentSources with smoothed PWLs)
smoothed = solver.preprocess_sources(
    time_step=0.1e-9,      # Filter window = 2 × time_step
    t_start=0,
    t_end=100e-9,
    compact_threshold=1e-12,  # Collinearity threshold for compaction
)

# Use smoothed sources (reusable across multiple analyses)
result = solver.solve_quasi_static(
    t_start=0, t_end=100e-9, n_points=101,
    smoothed_sources=smoothed,  # Pass pre-smoothed sources
)
```

**Reusing Smoothed Sources Across Solvers:**
```python
# Preprocess once
smoothed = solver.preprocess_sources(time_step=0.1e-9, t_start=0, t_end=100e-9)

# Reuse for multiple quasi-static analyses
result1 = solver.solve_quasi_static(..., smoothed_sources=smoothed)
result2 = solver.solve_quasi_static(..., smoothed_sources=smoothed)

# Also works with transient solver
from analysis.transient_solver import TransientIRDropSolver
trans = TransientIRDropSolver(model, graph)
trans_smoothed = trans.preprocess_sources(time_step=0.1e-9, t_start=0, t_end=100e-9)
result3 = trans.solve_transient(..., smoothed_sources=trans_smoothed)
```

**Manual Smoothing (Low-Level API):**
```python
from analysis.pwl_smoothing import PWLSmoother, smooth_pwl_points, compact_pwl, pulse_to_pwl_points
from parser.current_sources import PWL, Pulse

smoother = PWLSmoother(time_step=0.1e-9, compact_threshold=1e-12)

# Smooth a single PWL waveform
pwl = PWL(points=[(0, 0), (1e-9, 1), (2e-9, 0)], period=10e-9)
smoothed_pwl = smoother.smooth_pwl(pwl, t_start=0, t_end=10e-9)

# Convert pulse to PWL and smooth
pulse = Pulse(v1=0, v2=1, delay=1e-9, rt=0.1e-9, ft=0.1e-9, width=2e-9, period=10e-9)
smoothed_from_pulse = smoother.smooth_pulse(pulse, t_start=0, t_end=10e-9)

# Direct function calls
points = [(0, 0), (1e-9, 1), (2e-9, 0)]
smoothed = smooth_pwl_points(points, period=10e-9, time_step=0.1e-9, t_start=0, t_end=10e-9)
compacted = compact_pwl(smoothed, threshold=1e-12)
```

**SmoothedWaveformCache for Batch Analysis:**
```python
# Create cache from VectorizedCurrentSources
vec_sources = solver._vec_sources  # Internal vectorized sources
cache = smoother.create_smoothed_cache(vec_sources, t_start=0, t_end=100e-9)

# Check cache compatibility
if cache.is_compatible(time_step=0.1e-9, t_start=0, t_end=100e-9):
    smoothed = smoother.apply_cache_to_sources(vec_sources, cache)
```

**Key Functions:**
| Function | Description |
|----------|-------------|
| `smooth_pwl_points()` | Apply triangular LP filter to PWL points |
| `compact_pwl()` | Remove collinear/redundant points |
| `pulse_to_pwl_points()` | Convert Pulse to PWL representation |
| `triangular_window()` | Evaluate triangle window function |
| `analytical_triangle_pwl_integral()` | Exact triangle×PWL integral |

**Smoothing Effect:**
- Preserves DC average (energy conservation)
- Removes high-frequency content above ~1/(2×time_step)
- Reduces numerical noise in transient analysis
- Compaction reduces memory for long simulations

#### Adjoint Sensitivity Analysis (IR-Drop Attribution)
For identifying which aggressor current sources contribute most to IR-drop at a victim node:

```python
from analysis.transient_solver import TransientIRDropSolver
from analysis.adjoint_sensitivity import AdjointSensitivitySolver

trans = TransientIRDropSolver(model, graph)
result = trans.solve_transient(t_start=0, t_end=100e-9, dt=1e-9)

# Analyze the worst node at peak time
victim = result.peak_ir_drop_node
T = result.peak_ir_drop_time

adjoint = AdjointSensitivitySolver.from_transient_solver(trans)
attribution = adjoint.analyze_victim(
    victim_node=victim,
    observation_time=T,
    memory_window=20,      # L time steps to look back
    dt=1e-9,
    top_k=10,              # Return top 10 aggressors
    spatial_window=(x_v - 500, x_v + 500, y_v - 500, y_v + 500),  # Optional spatial filter
)

print(f"Victim: {attribution.victim_node}")
print(f"Total IR-drop: {attribution.ir_drop_at_T:.2f} mV")
print(f"Self-contribution: {attribution.self_contribution_mV:.3f} mV ({attribution.self_contribution_pct:.1f}%)")

for i, agg in enumerate(attribution.top_aggressors, 1):
    print(f"  {i}. {agg.node}: {agg.contribution_mV:.3f} mV ({agg.contribution_pct:.1f}%)")
```

**Two Methods Available:**
1. **Dynamic Adjoint** (`analyze_victim`): Propagates sensitivities backward through RC network's memory. For stiff systems (τ << dt), the dynamic method correctly converges to the static sensitivity result.
2. **Static Sensitivity** (`analyze_victim_static` or `use_static=True`): Uses steady-state sensitivity (G^-1). Faster than dynamic for stiff RC systems.

**Initial Condition Options:**
- `initial_condition='zero'` (default): Assumes V=VDD at start (zero IR-drop baseline). Computes contributions to the **total** IR-drop at observation time T.
- `initial_condition='dc'`: Starts from DC operating point. Computes contributions to the **incremental** IR-drop (above the DC baseline from static currents).

Use `'dc'` mode when you want to analyze switching-induced IR-drop separately from the baseline drop caused by static leakage currents.

**When to Use Static vs Dynamic:**
- Use `use_static=True` for faster analysis when RC time constant is much smaller than time step (τ << dt)
- For typical PDN grids with very small resistances, both methods give the same result; static is faster
- Dynamic method captures time-varying effects for grids with significant decoupling capacitor effects

**Vectorization Threshold:**
```python
# Force vectorized current evaluation (faster for many sources)
adjoint = AdjointSensitivitySolver(model, graph, vectorize_threshold=0)

# Or disable vectorization (uses raw CurrentSource objects)
adjoint = AdjointSensitivitySolver(model, graph, vectorize_threshold=100000)
```
Default threshold is 10000 sources. Both modes produce identical results.

**Example with Static Method (Recommended for Most PDNs):**
```python
attribution = adjoint.analyze_victim_static(
    victim_node=victim,
    observation_time=T,
    top_k=10,
)
# OR equivalently:
attribution = adjoint.analyze_victim(victim, T, use_static=True, top_k=10)
```

**Example with DC Initial Condition (Incremental Attribution):**
```python
# Analyze switching-induced IR-drop (contributions above DC baseline)
attribution = adjoint.analyze_victim(
    victim_node=victim,
    observation_time=T,
    initial_condition='dc',  # Start from DC operating point
    top_k=10,
)

print(f"Total IR-drop at T: {attribution.ir_drop_at_T:.2f} mV")
print(f"DC baseline IR-drop: {attribution.dc_ir_drop_mV:.2f} mV")
incremental = attribution.ir_drop_at_T - attribution.dc_ir_drop_mV
print(f"Incremental IR-drop: {incremental:.2f} mV")
# Contributions are attributed to incremental IR-drop (switching activity)
```

**AdjointAttribution Fields:**
- `victim_node`, `observation_time`, `ir_drop_at_T`: Victim info (always total IR-drop: VDD - V_T)
- `memory_window`, `t_array`: Time window analyzed
- `self_contribution_mV`, `self_contribution_pct`: Victim's own current contribution
- `top_aggressors`: List of `AggressorContribution` (see below)
- `attribution_efficiency`: Ratio of total_attributed / IR-drop being attributed (~1.0 for static)
- `initial_condition`: Initial condition used (`'zero'` or `'dc'`)
- `dc_ir_drop_mV`: DC baseline IR-drop (only when `initial_condition='dc'`). Incremental = `ir_drop_at_T - dc_ir_drop_mV`

**AggressorContribution Fields:**
- `node`: Aggressor node name
- `contribution_mV`: Contribution in mV. In 'zero' mode: total (from I(t)). In 'dc' mode: incremental (from ΔI = I(t) - I_DC)
- `contribution_pct`: Percentage of attributed IR-drop
- `source_names`: List of current source instance names
- `current_waveform`: Optional I(t) waveform over memory window
- `static_contribution_mV`: Static (DC) contribution in mV (only in 'dc' mode). Total = `contribution_mV + static_contribution_mV`

**Batch Attribution (multiple victims):**
```python
ctx = adjoint.prepare(dt=1e-9)  # Prepare once, caches LU factorization
for victim in victims:
    result = adjoint.analyze_victim(victim, T, context=ctx)
```

## Testing

**Test layout mirrors `src/`:**
```
tests/
├── graph/          # test_rx_graph, test_rx_algorithms
├── model/          # test_unified_core
├── solver/         # test_hierarchical_solver, test_coupled_hierarchical_solver,
│                   # test_batch_solving, test_regional_solver, test_pdn_solver,
│                   # test_hierarchical_integration (slow), test_tiled_accuracy
├── analysis/       # test_dynamic_solver, test_transient_solver, test_transient_multi_rhs,
│                   # test_adjoint_sensitivity, test_pwl_smoothing, test_vectorized_sources,
│                   # test_smoothing_source_idx, test_dynamic_integration (slow)
├── parser/         # test_pdn_parser, test_parallel_parser, test_edge_attrs,
│                   # test_parser_regression, test_pdn_integration (slow)
├── distributed/    # test_distributed_solver
├── visualization/  # test_pdn_plotter, test_stripe_heatmap
├── legacy/         # test_irdrop, test_partitioner
└── fixtures.py     # Factory functions for edge case testing
```

**Test netlists:** `netlist/netlist_test/` (small PDN), `netlist/netlist_small/` (minimal unit tests).

**Key invariants tested:**
- Zero load -> all nodes at pad voltage
- R_eff symmetry: `R(u,v) == R(v,u)` and triangle inequality
- Partition balance ratio <= 3.5; pads excluded from partitions
- Floating island detection removes disconnected components

**Test helper:** `build_small()` creates standard test grid (K=3, N0=8, I_N=80)

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

**Element syntax in `.ckt` files:**
```spice
R_name node1 node2 <resistance_kOhm>
C_name node1 node2 <capacitance_fF>
L_name node1 node2 <inductance_nH>
I_name node1 node2 <current_mA>
V_name node+ node- <voltage_V>
X_inst subckt node1 node2 ...
```

**Current source syntax in `instanceModels*.sp` (enhanced):**
```spice
I_name node+ node- <dc_mA> [static_value=<mA>] [pulse(v1,v2,delay,rt,ft,width,period)] [pwl(t1 v1 t2 v2 ...)]
```
- `static_value=`: Additional static current component
- `pulse(...)`: Periodic pulse waveform (values in Amperes)
- `pwl(...)`: Piece-wise linear waveform with optional `pwl_period=` and `pwl_delay=`

**Node naming convention:** `<x>_<y>_<layer>` (e.g., `1000_2000_M1`)

**Boundary nodes (multi-tile stitching):**
Nodes shared across tile boundaries are marked with `*` prefix in `.ckt` files:
```spice
R_bnd_M1 *900_2000_M1 *1000_2000_M1 8    # Cross-tile resistor (both nodes starred)
r 800_2000_M1 *900_2000_M1 8              # Internal-to-boundary resistor (one starred)
```

The `*` prefix signals the parser to track these nodes for tile stitching:
- Parser strips the `*` prefix when creating graph nodes
- Tracks `boundary_node1`/`boundary_node2` flags in edge attributes
- Merges matching boundary nodes across tiles during graph construction

## PDNPlotter Advanced Features

| Feature | Description |
|---------|-------------|
| Net Type Detection | Auto-detects power vs ground from naming |
| Layer Orientation | Auto-detects 'H'/'V'/'MIXED' from resistor edge angles |
| Anisotropic Binning | Orientation-aware bins: thin perpendicular to routing |
| Stripe Consolidation | Merges adjacent stripes when count exceeds threshold |
| Worst Node Selection | Finds spatially-separated worst-case nodes |

**Colormap Conventions:**
| Mode | Colormap | Aggregation | Units |
|------|----------|-------------|-------|
| IR-Drop (power) | `RdYlGn_r` | Max per bin | mV |
| Ground-Bounce (VSS) | `RdYlGn_r` | Max per bin | mV |
| Voltage (power) | `RdYlGn` | Min per bin | V |
| Current | `hot_r` | Sum per bin | mA |

## File Landmarks

- **Notebooks**: `notebooks/irdrop_decomposition_pdn.ipynb`, `notebooks/irdrop_decomposition.ipynb`, `notebooks/irdrop_decomposition_unified_model.ipynb`
- **Tests**: `tests/{graph,model,solver,analysis,parser,distributed,visualization,legacy}/test_*.py`
- **API exports**: `src/*/__init__.py` (package-level public APIs)
- **Scripts**: `scripts/analysis/`, `scripts/solver/`, `scripts/parser/`
- **Reference**: `DEPRECATION.md` (historical old→new import mappings)

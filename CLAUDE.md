# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Static IR-drop analysis prototype for multi-layer power grids. Supports both synthetic grids and real PDN netlists.

**Three Subsystems:**
1. **`core/`** - Unified model supporting BOTH synthetic grids and PDN netlists (use this for new code)
2. **`irdrop/`** - Original synthetic grid generation, solving, partitioning
3. **`pdn/`** - SPICE-like netlist parsing (`NetlistParser`) and standalone DC solver (`PDNSolver`)

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run all tests (fast)
python run_all_tests.py

# Run slow integration tests
python run_all_integration_tests.py

# Run specific test module
python -m unittest tests.test_irdrop
python -m unittest tests.test_hierarchical_solver
python -m unittest tests.test_pdn_parser

# Run single test
python -m unittest tests.test_irdrop.TestIRDrop.test_no_load_currents_all_pad_voltage

# Run examples
python example_ir_drop.py
python example_partitioning.py
python example_effective_resistance.py
```

## Architecture

### Data Flow
- **Synthetic**: `generate_power_grid()` -> `create_model_from_synthetic(G, pads, vdd)` -> `UnifiedIRDropSolver`
- **PDN**: `NetlistParser.parse()` -> `create_model_from_pdn(graph, net_name)` -> `UnifiedIRDropSolver`
- **Multi-Net**: `NetlistParser.parse()` -> `create_multi_net_models(graph)` -> iterate models

**Key Constraint:** Pads (voltage sources) are Dirichlet BCs at Vdd, eliminated via Schur complement. LU factorization cached for batch solves.

### Core Module (core/) - Preferred for New Code

**Module Structure:**
```
core/
├── unified_model.py        # UnifiedPowerGridModel, grid decomposition
├── unified_solver.py       # UnifiedIRDropSolver (orchestration)
├── solver_results.py       # Result data classes (UnifiedSolveResult, etc.)
├── coupled_system.py       # Block matrices, Schur complement for coupled solver
├── current_aggregation.py  # CurrentAggregator for port current distribution
├── tiling.py               # TileManager, solve_single_tile for parallel tiling
├── graph_converter.py      # NetworkX <-> Rustworkx conversion utilities
├── factory.py              # create_model_from_* functions
├── node_adapter.py         # NodeInfoExtractor
├── edge_adapter.py         # EdgeInfoExtractor, ElementType
├── rx_graph.py             # RustworkxMultiDiGraphWrapper
└── __init__.py             # Public API exports
```

**Key Classes:**
- **UnifiedPowerGridModel**: Handles both NodeID and string nodes; auto-detects floating islands
- **UnifiedIRDropSolver**: `solve()` for flat, `solve_hierarchical()` for layer-decomposed (approximate), `solve_hierarchical_coupled()` for exact coupled solve, `solve_hierarchical_tiled()` for parallel tiled solving
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

**Factory Functions:**
```python
from core import create_model_from_synthetic, create_model_from_pdn, create_multi_net_models

# From synthetic grid
model = create_model_from_synthetic(G, pads, vdd=1.0)

# From PDN netlist (single net) - vdd auto-extracted
model = create_model_from_pdn(graph, 'VDD')

# From PDN netlist (all nets)
models = create_multi_net_models(graph)  # {'VDD': model, 'VSS': model}
```

**Result Data Classes:**
- `UnifiedSolveResult`: Basic solve result with voltages, ir_drop, metadata
- `UnifiedHierarchicalResult`: Hierarchical result with port_nodes, port_voltages, port_currents, aggregation_map
- `UnifiedCoupledHierarchicalResult`: Coupled solver result with iterations, final_residual, converged, timings
- `TiledBottomGridResult`: Tiled solve result with tiles, per_tile_solve_times, validation_stats

**Enums:**
- `GridSource.SYNTHETIC`, `GridSource.PDN_NETLIST`: Source type detection
- `ElementType.RESISTOR`, `ElementType.CAPACITOR`, `ElementType.INDUCTOR`, `ElementType.CURRENT_SOURCE`

**Graph Converter (for legacy pickle files):**
```python
from core import detect_graph_type, ensure_rustworkx_graph

# Detect graph type from pickle
graph_type = detect_graph_type(graph)  # Returns 'networkx', 'rustworkx', or 'unknown'

# Auto-convert NetworkX to Rustworkx if needed
graph = ensure_rustworkx_graph(graph)  # No-op if already Rustworkx
```

### PDN Module (pdn/)
- **NetlistParser**: Parses SPICE-like tile-based netlists with gzip support
- **PDNSolver**: Standalone DC solver (use if you don't need unified interface)
- **PDNPlotter**: Layer-wise heatmap generation with advanced features

### IRDrop Module (irdrop/) - Original Synthetic
- `generate_power_grid()`: Creates K-layer resistor mesh with `NodeID` keys
- `PowerGridModel`, `IRDropSolver`: Original classes (prefer `core/` unified versions)
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

## Typical Workflow Patterns

### PDN Netlist Analysis (Recommended)
```python
from pdn_parser import NetlistParser
from core import create_model_from_pdn, UnifiedIRDropSolver

parser = NetlistParser('./pdn/netlist_test', validate=True)
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
- `solver`: `'gmres'` (default) or `'bicgstab'`. GMRES is generally more robust.
- `tol`: Residual tolerance for iterative solver (default 1e-8)
- `maxiter`: Maximum iterations before raising RuntimeError (default 500)
- `preconditioner`: `'block_diagonal'` (default), `'none'`, or `'ilu'`

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
from generate_power_grid import generate_power_grid
from core import create_model_from_synthetic, UnifiedIRDropSolver
from irdrop import StimulusGenerator

G, loads, pads = generate_power_grid(K=3, N0=12, I_N=150, N_vsrc=4, seed=42)
model = create_model_from_synthetic(G, pads, vdd=1.0)

stim_gen = StimulusGenerator(load_nodes=list(loads.keys()), vdd=1.0, seed=10, graph=G)
meta = stim_gen.generate(total_power=1.5, percent=0.3, distribution="gaussian")

solver = UnifiedIRDropSolver(model)
result = solver.solve(meta.currents)
```

## Testing

**Test modules:** `test_irdrop.py`, `test_partitioner.py`, `test_pdn_parser.py`, `test_pdn_solver.py`, `test_pdn_plotter.py`, `test_unified_core.py`, `test_hierarchical_solver.py`, `test_coupled_hierarchical_solver.py`, `test_hierarchical_integration.py` (slow), `test_regional_solver.py`

**Test fixtures:** `tests/fixtures.py` provides factory functions for edge case testing scenarios.

**Test netlists:** `pdn/netlist_test/` (small PDN), `pdn/netlist_small/` (minimal unit tests)

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

**Node naming convention:** `<x>_<y>_<layer>` (e.g., `1000_2000_M1`)

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

- **Examples**: `example_ir_drop.py`, `example_partitioning.py`, `example_effective_resistance.py`, `example_regional_voltage.py`
- **Notebooks**: `irdrop_decomposition_pdn.ipynb`, `irdrop_decomposition.ipynb`, `irdrop_decomposition_unified_model.ipynb`
- **Tests**: `tests/test_*.py`, `test_tiled_accuracy.py` (standalone accuracy validation)
- **API exports**: `core/__init__.py`, `irdrop/__init__.py`
- **Documentation**: `REGIONAL_SOLVER_USAGE.md` (near-field/far-field decomposition guide)

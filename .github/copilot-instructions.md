# Power Grid IR-Drop Analysis Prototype

AI coding guide for static IR-drop analysis, effective resistance, and hierarchical solving for multi-layer power grids. Supports both synthetic grids and real PDN netlists.

## Architecture Overview

**Source packages (`src/`, installed via `pip install -e .`):**
1. **`src/graph/`** - Rustworkx graph wrappers and conversion utilities
2. **`src/model/`** - `UnifiedPowerGridModel`, adapters, factory functions
3. **`src/solver/`** - Flat, hierarchical, coupled, and tiled solvers
4. **`src/analysis/`** - Dynamic, transient, adjoint analysis; PWL smoothing
5. **`src/parser/`** - SPICE-like netlist parsing (`NetlistParser`)
6. **`src/distributed/`** - Distributed DDM solver (tile-based domain decomposition)
7. **`src/visualization/`** - Plotters (`UnifiedPlotter`, `DynamicPlotter`, `PDNPlotter`)
8. **`src/legacy/`** - Original synthetic grid modules

**Data Flow:**
- Synthetic: `generate_power_grid()` -> `create_model_from_synthetic(G, pads, vdd)` -> `UnifiedIRDropSolver`
- PDN: `NetlistParser.parse()` -> `create_model_from_pdn(graph, net_name)` -> `UnifiedIRDropSolver`
- Multi-Net: `NetlistParser.parse()` -> `create_multi_net_models(graph)` -> iterate models

**Key Constraint:** Pads (voltage sources) are Dirichlet BCs at Vdd, eliminated via Schur complement. LU factorization cached for batch solves.

## Critical Domain Conventions

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

## Module Responsibilities

### Model (`src/model/`)

```python
from model.factory import (create_model_from_pdn, create_model_from_synthetic,
                           create_multi_net_models)
from solver.unified_solver import UnifiedIRDropSolver

# PDN netlist (vdd auto-extracted from pg_net_voltage file or voltage source edges)
from parser.netlist import NetlistParser
parser = NetlistParser('./netlist/netlist_test', validate=True)
graph = parser.parse()
model = create_model_from_pdn(graph, 'VDD')  # vdd auto-detected
load_currents = model.extract_current_sources()  # Get I-type edge currents

# Flat solve
solver = UnifiedIRDropSolver(model)
result = solver.solve(load_currents)

# Hierarchical solve (partition at layer boundary)
hier_result = solver.solve_hierarchical(load_currents, partition_layer='M2', top_k=5)
```

**Key Classes:**
- `UnifiedPowerGridModel`: Handles both NodeID and string nodes; auto-detects floating islands
- `UnifiedIRDropSolver`: `solve()` for flat, `solve_hierarchical()` for layer-decomposed (approximate), `solve_hierarchical_coupled()` for exact coupled solve, `solve_hierarchical_tiled()` for parallel tiled solving
- `BlockMatrixSystem`: Block-partitioned conductance matrix (port/interior splits)
- `SchurComplementOperator`: Matrix-free Schur complement for coupled solver
- `CoupledSystemOperator`: Full coupled top-grid + Schur complement operator
- `CurrentAggregator`: Distributes load currents to ports using shortest-path or effective resistance weighting
- `TileManager`: Manages bottom-grid tile generation, halo expansion, and connectivity validation
- `NodeInfoExtractor` / `EdgeInfoExtractor`: Adapt different graph representations
- `UnifiedStatistics`: Compute netlist statistics (node/edge counts, R/C/L/I totals)
- `UnifiedPartitioner`: Layer-based and spatial grid partitioning
- `UnifiedPlotter`: Voltage/IR-drop heatmap generation
- `UnifiedEffectiveResistanceCalculator`: Pairwise and single-node effective resistance

**Factory Functions:**
- `create_model_from_synthetic(graph, pad_nodes, vdd)`: For synthetic grids
- `create_model_from_pdn(graph, net_name)`: For PDN netlists (vdd auto-extracted)
- `create_multi_net_models(graph, net_filter=None)`: Batch create models for all nets
- `create_model_from_graph(graph, pads, vdd, auto_detect_source=True)`: Auto-detecting factory

**Enums:**
- `GridSource.SYNTHETIC`, `GridSource.PDN_NETLIST`: Source type detection
- `ElementType.RESISTOR`, `ElementType.CAPACITOR`, `ElementType.INDUCTOR`, `ElementType.CURRENT_SOURCE`

**Graph Converter (for legacy pickle files):**
```python
from graph.converter import detect_graph_type, ensure_rustworkx_graph

# Detect graph type
graph_type = detect_graph_type(graph)  # Returns 'networkx', 'rustworkx', or 'unknown'

# Auto-convert if needed (safe to call on any graph)
graph = ensure_rustworkx_graph(graph, verbose=True)
```

### Parser (`src/parser/`)
- **`NetlistParser`**: Parses SPICE-like tile-based netlists with gzip support
- **`PDNSolver`** (`solver.pdn_solver`): Standalone DC solver (use if you don't need unified interface)
- **`PDNPlotter`** (`visualization.pdn_plotter`): Layer-wise heatmap generation with advanced features
- **Graph metadata**: `graph.graph['net_connectivity']`, `graph.graph['vsrc_nodes']`, `graph.graph['instance_node_map']`

**PDNPlotter Advanced Features:**

| Feature | Description |
|---------|-------------|
| **Net Type Detection** | Auto-detects power (`VDD`, `VCC`, `VDDA`, etc.) vs ground (`VSS`, `GND`) from naming |
| **Layer Orientation** | Auto-detects `'H'` (horizontal), `'V'` (vertical), `'MIXED'` from resistor edge angles (+-15 deg tolerance) |
| **Anisotropic Binning** | Orientation-aware bins: thin perpendicular to routing, wide along routing |
| **Stripe Consolidation** | Merges adjacent stripes when count exceeds `max_stripes` threshold |
| **Worst Node Selection** | Finds spatially-separated worst-case nodes (10% min separation) |

**Heatmap Generation Methods:**
```python
from visualization.pdn_plotter import PDNPlotter

plotter = PDNPlotter(graph, graph.graph.get('net_connectivity', {}))

# IR-drop heatmaps with anisotropic binning
plotter.generate_layer_heatmaps(
    net_name='VDD',
    output_path='./results',
    anisotropic_bins=True,
    bin_aspect_ratio=50,
    show_irdrop=True,
)
```

```bash
# CLI usage
python -m solver.pdn_solver --input graph.pkl --net VDD --output ./results
python -m solver.pdn_solver --input graph.pkl --net VDD --output ./results --show-voltage
python -m solver.pdn_solver --input graph.pkl --net VDD --output ./results --stripe-mode
```

### Legacy (`src/legacy/`)
- `generate_power_grid()`: Creates K-layer resistor mesh with `NodeID` keys
- `PowerGridModel`, `IRDropSolver`: Original classes (prefer unified versions in `src/solver/`)
- `GridPartitioner`: Structured slab partitioning along via rows/columns

## Typical Workflow Patterns

### PDN Netlist Analysis (Recommended)
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

### Hierarchical Solve (Layer Decomposition)
```python
# Partition at layer boundary for faster bottom-grid solves
hier_result = solver.solve_hierarchical(
    load_currents,
    partition_layer='M2',  # or integer layer index
    top_k=5,               # ports per load for current aggregation
    weighting="shortest_path",  # default: inverse of least-resistive path
    verbose=True,          # print timing breakdown
)
print(f"Ports: {len(hier_result.port_nodes)}")
```

### Coupled Hierarchical Solve (Exact)
```python
coupled_result = solver.solve_hierarchical_coupled(
    load_currents,
    partition_layer='M2',
    solver='gmres',
    tol=1e-8,
    maxiter=500,
    preconditioner='block_diagonal',
    verbose=True,
)
print(f"Converged in {coupled_result.iterations} iterations")
print(f"Final residual: {coupled_result.final_residual:.2e}")
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

**Element Syntax in `.ckt` files:**
```spice
R_name node1 node2 <resistance_kOhm>
C_name node1 node2 <capacitance_fF>
L_name node1 node2 <inductance_nH>
I_name node1 node2 <current_mA>       # Current source (instance load)
V_name node+ node- <voltage_V>        # Voltage source (pad)
X_inst subckt node1 node2 ...         # Subcircuit instance
```

**Node naming convention:** `<x>_<y>_<layer>` (e.g., `1000_2000_M1`)

## Testing & Validation

**Run tests**: `python run_all_tests.py`

**Test layout mirrors `src/`:**
```
tests/
  graph/          # test_rx_graph, test_rx_algorithms
  model/          # test_unified_core
  solver/         # test_hierarchical_solver, test_coupled_hierarchical_solver, ...
  analysis/       # test_dynamic_solver, test_transient_solver, test_adjoint_sensitivity, ...
  parser/         # test_pdn_parser, test_parallel_parser, test_edge_attrs, ...
  distributed/    # test_distributed_solver
  visualization/  # test_pdn_plotter, test_stripe_heatmap
  legacy/         # test_irdrop, test_partitioner
```

**Test netlists:** `netlist/netlist_test/` (small PDN), `netlist/netlist_small/` (minimal unit tests).

**Key invariants tested**:
- Zero load -> all nodes at pad voltage
- R_eff symmetry: `R(u,v) == R(v,u)` and triangle inequality
- Partition balance ratio <= 3.5; pads excluded from partitions
- Floating island detection removes disconnected components

## Common Pitfalls

1. **Pad vs vdd confusion**: `plot_ir_drop_map(G, voltages, vdd=1.0, ...)` needs float, not list
2. **Area filtering**: Must pass `graph=G` to `StimulusGenerator` to use `area` parameter
3. **PDN ground node**: Ground is `'0'` string; excluded from conductance matrix but preserved for I-type edges
4. **Gaussian degeneracy**: Falls back to uniform if weights sum to zero
5. **Legacy pickle files**: Old `.pkl` files may contain NetworkX graphs. Use `ensure_rustworkx_graph(graph)` to auto-convert before passing to `create_model_from_pdn()`

## File Landmarks

- **Notebooks**: `notebooks/irdrop_decomposition_pdn.ipynb` (PDN hierarchical), `notebooks/irdrop_decomposition.ipynb` (synthetic)
- **Tests**: `tests/{graph,model,solver,analysis,parser,distributed,visualization,legacy}/test_*.py`
- **Test netlists**: `netlist/netlist_test/`, `netlist/netlist_small/`
- **API exports**: `src/*/__init__.py` (package-level public APIs)
- **Scripts**: `scripts/analysis/`, `scripts/solver/`, `scripts/parser/`

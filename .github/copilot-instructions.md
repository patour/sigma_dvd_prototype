# Power Grid IR-Drop Analysis Prototype

AI coding guide for static IR-drop analysis, effective resistance, and hierarchical solving for multi-layer power grids. Supports both synthetic grids and real PDN netlists.

## Architecture Overview

**Three Subsystems:**
1. **`core/`** - Unified model supporting BOTH synthetic grids and PDN netlists (use this for new code)
2. **`irdrop/`** - Original synthetic grid generation, solving, partitioning  
3. **`pdn/`** - SPICE-like netlist parsing (`NetlistParser`) and standalone DC solver (`PDNSolver`)

**Data Flow:**
- Synthetic: `generate_power_grid()` → `create_model_from_synthetic(G, pads, vdd)` → `UnifiedIRDropSolver`
- PDN: `NetlistParser.parse()` → `create_model_from_pdn(graph, net_name)` → `UnifiedIRDropSolver`
- Multi-Net: `NetlistParser.parse()` → `create_multi_net_models(graph)` → iterate models

**Key Constraint:** Pads (voltage sources) are Dirichlet BCs at Vdd, eliminated via Schur complement. LU factorization cached for batch solves.

## Critical Domain Conventions

### Node Types
- **Synthetic**: `NodeID(layer, idx)` frozen dataclass keys the graph
- **PDN**: String node names like `'1000_2000_M1'`, `'VDD_vsrc'`, `'0'` (ground)

### Unit System (PDN)
- Resistance: kOhm, Capacitance: fF, Inductance: nH, Current: mA
- Conductance matrix in mS (milli-Siemens) for self-consistent G·V = I

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

### `core/` - Unified Model (Preferred for New Code)
```python
from core import (create_model_from_pdn, create_model_from_synthetic, 
                  create_multi_net_models, UnifiedIRDropSolver)

# PDN netlist (vdd auto-extracted from pg_net_voltage file or voltage source edges)
parser = NetlistParser('./pdn/netlist_test', validate=True)
graph = parser.parse()
model = create_model_from_pdn(graph, 'VDD')  # vdd auto-detected, no parameter needed
load_currents = model.extract_current_sources()  # Get I-type edge currents

# Flat solve
solver = UnifiedIRDropSolver(model)
result = solver.solve(load_currents)

# Hierarchical solve (partition at layer boundary)
hier_result = solver.solve_hierarchical(load_currents, partition_layer='M2', top_k=5)
```

**Key Classes:**
- `UnifiedPowerGridModel`: Handles both NodeID and string nodes; auto-detects floating islands
- `UnifiedIRDropSolver`: `solve()` for flat, `solve_hierarchical()` for layer-decomposed
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

### `pdn/` - PDN Netlist Parsing
- **`NetlistParser`**: Parses SPICE-like tile-based netlists with gzip support
- **`PDNSolver`**: Standalone DC solver (use if you don't need unified interface)
- **`PDNPlotter`**: Layer-wise heatmap generation with advanced features (see below)
- **Graph metadata**: `graph.graph['net_connectivity']`, `graph.graph['vsrc_nodes']`, `graph.graph['instance_node_map']`

**PDNPlotter Advanced Features:**

| Feature | Description |
|---------|-------------|
| **Net Type Detection** | Auto-detects power (`VDD`, `VCC`, `VDDA`, etc.) vs ground (`VSS`, `GND`) from naming |
| **Layer Orientation** | Auto-detects `'H'` (horizontal), `'V'` (vertical), `'MIXED'` from resistor edge angles (±15° tolerance) |
| **Anisotropic Binning** | Orientation-aware bins: thin perpendicular to routing, wide along routing |
| **Stripe Consolidation** | Merges adjacent stripes when count exceeds `max_stripes` threshold |
| **Worst Node Selection** | Finds spatially-separated worst-case nodes (10% min separation) |

**Colormap & Aggregation Conventions:**
| Mode | Colormap | Aggregation | Units |
|------|----------|-------------|-------|
| IR-Drop (power) | `RdYlGn_r` | Max per bin | mV |
| Ground-Bounce (VSS) | `RdYlGn_r` | Max per bin | mV |
| Voltage (power) | `RdYlGn` | Min per bin | V |
| Voltage (ground) | `RdYlGn` | Max per bin | V |
| Current | `hot_r` | Sum per bin | mA |

**Heatmap Generation Methods:**
```python
from pdn_plotter import PDNPlotter

plotter = PDNPlotter(graph, graph.graph.get('net_connectivity', {}))

# IR-drop heatmaps with anisotropic binning
plotter.generate_layer_heatmaps(
    net_name='VDD',
    output_path='./results',
    anisotropic_bins=True,        # Enable orientation-aware binning
    bin_aspect_ratio=50,          # Ratio for anisotropic bin dimensions
    show_irdrop=True,             # True=mV IR-drop, False=Voltage
)

# Stripe-based visualization (auto-consolidates if > max_stripes)
plotter.generate_stripe_heatmaps(
    net_name='VDD',
    output_path='./results',
    max_stripes=50,               # Consolidation threshold
    layer_orientations={'M1': 'V', 'M2': 'H'},  # Manual override
)

# Current source distribution
plotter.generate_current_heatmaps(
    net_name='VDD',
    output_path='./results',
    anisotropic_bins=True,
)
```

```bash
# CLI usage
python pdn/pdn_solver.py --input graph.pkl --net VDD --output ./results
python pdn/pdn_solver.py --input graph.pkl --net VDD --output ./results --show-voltage  # Voltage mode
python pdn/pdn_solver.py --input graph.pkl --net VDD --output ./results --stripe-mode   # Stripe mode
```

### `irdrop/` - Synthetic Grids (Original)
- `generate_power_grid()`: Creates K-layer resistor mesh with `NodeID` keys
- `PowerGridModel`, `IRDropSolver`: Original classes (use `core/` unified versions instead)
- `GridPartitioner`: Structured slab partitioning along via rows/columns

### `generate_power_grid.py`
Constructs K-layer resistor mesh. Returns `(G, loads, pads)` where loads map `NodeID → current`.
- Stripe count halves per layer: `ceil(N0/2^ℓ)`
- Load nodes on layer 0 only; pads on top layer (K-1)

## PDN Netlist Format

The PDN parser reads SPICE-like tile-based netlists. Directory structure:
```
netlist_dir/
├── ckt.sp              # Top-level circuit includes
├── tile_0_0.ckt        # Tile subcircuit with R/C/L/I/V elements
├── tile_0_0.nd         # Node coordinate mapping (x y layer node_name)
├── package.ckt         # Package-level connections
├── instanceModels_0_0.sp  # Instance current source models
├── pg_net_voltage      # Power net voltage specs (VDD 1.0, VSS 0.0)
├── additional_vsrcs    # Extra voltage source definitions
├── decap_cell_list     # Decap cell instance names
└── switch_cell_list    # Power switch cell names
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

**Graph metadata after parsing:**
- `graph.graph['net_connectivity']`: `{net_name: [nodes]}`
- `graph.graph['vsrc_nodes']`: Voltage source node names
- `graph.graph['layer_stats']`: Per-layer node/resistor counts

## Typical Workflow Patterns

### PDN Netlist Analysis (Recommended)
```python
from pdn_parser import NetlistParser
from core import create_model_from_pdn, UnifiedIRDropSolver

parser = NetlistParser('./pdn/netlist_test', validate=True)
graph = parser.parse()
model = create_model_from_pdn(graph, 'VDD')  # vdd auto-extracted from graph
load_currents = model.extract_current_sources()

solver = UnifiedIRDropSolver(model)
result = solver.solve(load_currents)
print(f"Max IR-drop: {max(result.ir_drop.values()):.4f} V")
```

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

**Hierarchical Solver Parameters:**
- `partition_layer`: Layer name (string like `'M2'`) or integer index. Nodes at/above this layer form "top-grid"; below form "bottom-grid". Use `model.get_all_layers()` to see available layers.
- `top_k`: Number of nearest ports per load for current aggregation (default 5). Higher values improve accuracy at cost of denser port current distribution. Start with 3-5.
- `weighting`: How to distribute load current to ports:
  - `"shortest_path"` (default): Inverse of least-resistive path length. Good balance of accuracy and speed.
  - `"effective"`: Inverse of effective resistance. Most accurate but slower.
  - `"uniform"`: Equal weight to all k ports. Simple but less accurate.
- `rmax`: Maximum resistance distance for `shortest_path` weighting. Paths beyond this are ignored. `None` = no limit.
- `verbose`: If `True`, print timing breakdown (decompose, aggregate, solve_top, solve_bottom).
- `use_fast_builder`: If `True` (default), use vectorized subgrid builder (~10x speedup). Set `False` for debugging.

**Hierarchical Result Fields:**
- `hier_result.voltages`: All node voltages (combined top + bottom)
- `hier_result.ir_drop`: IR-drop at each node (`Vdd - voltage`)
- `hier_result.port_nodes`: Set of port nodes at partition boundary
- `hier_result.port_voltages`: Voltage at each port (from top-grid solve)
- `hier_result.port_currents`: Aggregated current injected at each port
- `hier_result.aggregation_map`: `{load_node: [(port, weight, current), ...]}` for debugging

### Tiled Hierarchical Solve (PDN only)
For large PDN grids, exploit spatial locality by tiling the bottom-grid into independent subproblems:

```python
# Tiled solve with 2x2 grid and 20% halo
tiled_result = solver.solve_hierarchical_tiled(
    current_injections=load_currents,
    partition_layer='M2',
    N_x=2, N_y=2,           # Tile grid dimensions
    halo_percent=0.2,       # Halo size as fraction of tile
    top_k=5,
    weighting='shortest_path',
    n_workers=4,            # Parallel workers (default: CPU count)
    parallel_backend='thread',   # 'thread' (default) or 'process'
    validate_against_flat=True,  # Compare against non-tiled solve
    progress_callback=lambda done, total, tid: print(f"Tile {tid}: {done}/{total}"),
)
print(f"Max diff vs flat: {tiled_result.validation_stats['max_diff']*1000:.3f} mV")

# Validation mode: skip top-grid solve, use pre-computed port voltages
flat_result = solver.solve_hierarchical(load_currents, partition_layer='M2', top_k=5)
tiled_result = solver.solve_hierarchical_tiled(
    current_injections=load_currents,
    partition_layer='M2',
    N_x=2, N_y=2,
    override_port_voltages=flat_result.port_voltages,  # Skip top-grid solve
    validate_against_flat=True,
)
```

**Tiled Solver Parameters:**
- `N_x, N_y`: Number of tiles in X and Y dimensions
- `halo_percent`: Halo region size as fraction of tile dimension (0.1-0.3 typical)
- `min_ports_per_tile`: Minimum port nodes per tile (default: `ceil(total_ports / (N_x × N_y))`). Tiles with fewer ports have boundaries expanded.
- `n_workers`: Number of parallel workers (default: `os.cpu_count()`)
- `parallel_backend`: `'thread'` (default, safe for unit tests) or `'process'` (for production parallelism)
- `override_port_voltages`: Optional `Dict[node, voltage]` to skip top-grid solve and use provided port voltages as Dirichlet BCs. Useful for validation/debugging.
- `validate_against_flat`: If True, compare results against non-tiled bottom-grid solve
- `progress_callback`: `Callable[[int, int, int], None]` - called with (completed, total, tile_id)
- `verbose`: Print timing and statistics

**Tile Constraints (Automatic):**
- **Phase 1 - Boundary Expansion**: Tiles with < `min_ports_per_tile` ports have boundaries expanded to include more ports
- **Phase 2 - Tile Merging**: Tiles with zero current sources are merged with neighbors (rare edge case)
- Halo regions use port voltages from top-grid solve (or `override_port_voltages`) as Dirichlet BCs
- Warnings emitted when halo is severely clipped at grid boundaries (< 25% of expected area)

**TiledBottomGridResult Fields (extends UnifiedHierarchicalResult):**
- `tiles`: List of `BottomGridTile` objects with bounds, core_nodes, halo_nodes, port_nodes
- `per_tile_solve_times`: `{tile_id: time_ms}` for performance profiling
- `halo_clip_warnings`: List of tile IDs where halo was significantly clipped at grid boundary
- `tiling_params`: `{'N_x': int, 'N_y': int, 'halo_percent': float}`
- `validation_stats`: `{'max_diff': float, 'mean_diff': float, 'rmse': float, ...}` if validate_against_flat=True

**NOTE:** Tiled solving is only supported for PDN graphs (string node names). Synthetic grids with `NodeID` tuples raise `ValueError`.

### Multi-Net PDN Analysis
Process all power/ground nets in a single PDN with batch model creation:

```python
from pdn_parser import NetlistParser
from core import create_multi_net_models, UnifiedIRDropSolver

parser = NetlistParser('./pdn/netlist_test', validate=True)
graph = parser.parse()

# Create models for specific nets (or all nets if net_filter=None)
models = create_multi_net_models(graph, net_filter=['VDD', 'VSS'])

for net_name, model in models.items():
    solver = UnifiedIRDropSolver(model)
    currents = model.extract_current_sources()
    result = solver.solve(currents)
    print(f"{net_name}: Max IR-drop = {max(result.ir_drop.values())*1000:.2f} mV")
```

## Testing & Validation

**Run tests**: `python -m unittest discover -s tests -p 'test_*.py'`
**Run all tests**: `python run_all_tests.py`

**Test modules**: `test_irdrop.py`, `test_partitioner.py`, `test_pdn_parser.py`, `test_pdn_solver.py`, `test_pdn_plotter.py`, `test_unified_core.py`, `test_hierarchical_solver.py`, `test_hierarchical_integration.py` (slow tests), `test_regional_solver.py`

**Key invariants tested**:
- Zero load → all nodes at pad voltage
- R_eff symmetry: `R(u,v) == R(v,u)` and triangle inequality
- Partition balance ratio ≤3.5; pads excluded from partitions
- Floating island detection removes disconnected components

**Test helpers**: `build_small()` creates standard test grid (K=3, N0=8, I_N=80)

## Development Workflow

**Environment setup**:
```bash
conda activate pyspice  # or your Python env
pip install -r requirements.txt
```

**Run examples**:
```bash
python example_ir_drop.py
python example_partitioning.py
python example_effective_resistance.py
```

**Debug/Testing with Pylance MCP (PREFERRED)**:
For quick debugging and testing, use the `pylanceRunCodeSnippet` MCP tool instead of terminal commands. Benefits:
- Uses correct Python interpreter configured for workspace
- No shell escaping issues with complex code
- Clean stdout/stderr output
- Runs in workspace context with proper imports

Example debug snippets:
```python
# Check module version/state
from core.unified_model import UnifiedPowerGridModel
print(UnifiedPowerGridModel.__module__)

# Quick PDN parsing test
from pdn_parser import NetlistParser
parser = NetlistParser('./pdn/netlist_test', validate=True)
graph = parser.parse()
print(f"Nodes: {graph.number_of_nodes()}, Edges: {graph.number_of_edges()}")

# Inspect edge types
from collections import Counter
edge_types = Counter(d.get('type') for u, v, d in graph.edges(data=True))
print(edge_types)
```

**Notebook module reloading (CRITICAL)**:
When modifying core modules, reload ALL dependencies in order:
```python
import importlib
import core.node_adapter, core.edge_adapter, core.unified_model, core.factory
importlib.reload(core.node_adapter)
importlib.reload(core.edge_adapter)  # Must reload before unified_model!
importlib.reload(core.unified_model)
importlib.reload(core.factory)
```
Failure to reload `edge_adapter` causes stale cached versions with broken island detection.

## Common Pitfalls

1. **Pad vs vdd confusion**: `plot_ir_drop_map(G, voltages, vdd=1.0, ...)` needs float, not list
2. **Area filtering**: Must pass `graph=G` to `StimulusGenerator` to use `area` parameter
3. **Notebook stale imports**: Always reload `core.edge_adapter` before `core.unified_model`
4. **PDN ground node**: Ground is `'0'` string; excluded from conductance matrix but preserved for I-type edges
5. **Gaussian degeneracy**: Falls back to uniform if weights sum to zero

## File Landmarks

- **Examples**: `example_ir_drop.py`, `example_partitioning.py`, `example_effective_resistance.py`, `example_regional_voltage.py`
- **Notebooks**: `irdrop_decomposition_pdn.ipynb` (PDN hierarchical), `irdrop_decomposition.ipynb` (synthetic)
- **Tests**: `tests/test_*.py` (11 test modules), `test_tiled_accuracy.py` (standalone accuracy validation)
- **Test netlist**: `pdn/netlist_test/` (small PDN for testing), `pdn/netlist_small/` (minimal PDN for unit tests)
- **API exports**: `core/__init__.py`, `irdrop/__init__.py`
- **Documentation**: `REGIONAL_SOLVER_USAGE.md` (near-field/far-field decomposition guide)

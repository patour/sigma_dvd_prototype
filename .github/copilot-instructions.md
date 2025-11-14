# IR-Drop Analysis Prototype - AI Coding Guide

## Architecture Overview

This is a **static IR-drop analysis toolkit** for multi-layer power grids. The workflow is:
1. **Generate** artificial power grid → 2. **Build** conductance model → 3. **Stimulate** with current loads → 4. **Solve** for voltages → 5. **Visualize** results

**Core modules** (`irdrop/`):
- `power_grid_model.py`: Constructs sparse conductance matrix (G) from resistor network; reduces system by treating pads as fixed Dirichlet boundary conditions (Vdd=1.0V). Pre-factorizes G_uu via LU for fast repeated solves.
- `stimulus.py`: Generates current sink patterns (Dict[node, current]) from total power budget; supports uniform/gaussian distribution, spatial filtering by rectangular area.
- `solver.py`: Orchestrates solve workflow; returns `IRDropResult` with voltages and ir_drop dicts.
- `plot.py`: Matplotlib scatter heatmaps filtered by layer; uses node `xy` attribute from graph.

**Grid generator** (`generate_power_grid.py`):
- Creates K-layer mesh with alternating orientation (H/V/H/V...); stripe count halves per layer (N₀, N₀/2, N₀/4...).
- Layer 0 = horizontal load-bearing stripes; loads uniformly distributed (never at via intersections).
- Returns `(nx.Graph, loads_dict, pads_list)` with edge attribute `resistance` and node attributes `layer`, `xy`, `node_type`.

## Critical Conventions

### Sign Convention for Currents
**Stimuli currents are POSITIVE for sinks** (nodes drawing current). Internally, `PowerGridModel.solve_voltages()` converts to negative injections (`I_node = -I_load`) to match nodal equation `G*V = I` where sources inject positive current.

### Node Identification
Nodes are `NodeID(layer, idx)` frozen dataclasses. Grid structure:
- **Layer 0**: horizontal stripes, contains all load nodes
- **Top layer (K-1)**: vertical stripes, contains all pad (Vdd) nodes
- **Vias**: connect adjacent layer intersections; via resistance halves per layer

### Matrix Reduction Pattern
Pads are eliminated via Schur complement: `G_uu * V_u = I_u - G_up * V_p`. The factorized `G_uu` is cached in `ReducedSystem` and reused across stimuli. For batch solves, use `solve_batch()` which shares the factorization and constant term `(-G_up * V_p)`.

## Development Workflows

### Running Examples
```csh
python example_ir_drop.py                           # Single run demo
python -m unittest discover -s tests -p 'test_*.py' # Full test suite
```

### Interactive Exploration
Use `proto.ipynb` for experimentation. Typical workflow:
1. Generate grid with `generate_power_grid(K, N0, I_N, N_vsrc, ...)`
2. Build model: `model = PowerGridModel(G, pad_nodes=pads, vdd=1.0)`
3. Create stimulus: `stim_gen.generate(total_power, percent=0.3, distribution="gaussian", area=(x1,y1,x2,y2))`
4. Solve: `result = solver.solve(meta.currents, meta)`
5. Plot: `plot_voltage_map(G, result.voltages, layer=0, show=False)` (use `show=False` in notebooks/headless)

### Testing Patterns
Tests use `build_small()` fixture (K=3, N0=8). Key assertions:
- Zero load → all nodes at pad voltage (test_no_load_currents_all_pad_voltage)
- Current conservation: `sum(currents) ≈ total_power / vdd` (test_uniform_power_distribution_current_sum)
- Monotonicity: higher power → lower min voltage (test_batch_min_voltage_monotonic)

## API Usage Patterns

### Stimulus Generation with Spatial Filtering
```python
# Filter loads to rectangular area before sampling
meta = stim_gen.generate(
    total_power=1.5, 
    count=10,                          # Exact node count (overrides percent)
    distribution="gaussian",           # or "uniform"
    gaussian_loc=1.0, gaussian_scale=0.2,  # Normal(loc, scale) weights
    area=(0.4, 0.4, 0.6, 0.6)         # (x_min, y_min, x_max, y_max)
)
```
**Note**: `area` filters candidates BEFORE sampling; requires `graph=G` in StimulusGenerator constructor.

### Batch Solves for Parameter Sweeps
```python
powers = [0.5, 1.0, 2.0]
metas = stim_gen.generate_batch(powers, percent=0.3, distribution="gaussian")
results = solver.solve_batch([m.currents for m in metas], metas)
# Results preserve order; access via: results[i].voltages, results[i].ir_drop
```

### Plotting Headless (CI/Notebooks)
Always pass `show=False` and `matplotlib.use("Agg")` is set by default in `plot.py`:
```python
fig, ax = plot_voltage_map(G, voltages, layer=0, show=False)
fig.savefig("output.png")  # Manual save
```

## Common Gotchas

- **Type error in plot_ir_drop_map**: Must pass `vdd` as numeric float, not `pads` list. Signature is `plot_ir_drop_map(G, voltages, vdd, layer=None)`.
- **Load nodes only on layer 0**: Attempting to place loads on upper layers breaks assumptions; `generate_power_grid` enforces this.
- **Via nodes excluded from loads**: `I_N` parameter controls tap insertion attempts; algorithm never places loads at via intersections.

## Key Files to Reference

- `example_ir_drop.py`: Canonical usage workflow (generate → model → stimulate → solve → summarize)
- `tests/test_irdrop.py`: Validation patterns and edge cases
- `irdrop/__init__.py`: Public API exports
- `README.md`: Installation, equation background (G*V=I), distribution modes

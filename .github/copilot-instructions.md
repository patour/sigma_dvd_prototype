# Power Grid IR-Drop Analysis Prototype

AI coding guide for static IR-drop analysis, effective resistance computation, and spatial partitioning of multi-layer power grids.

## Architecture Overview

**Data Flow**: Generate grid → Build conductance model → Generate stimuli → Solve voltages/resistances → Partition (optional) → Visualize

**Core Pipeline**: `generate_power_grid()` produces NetworkX graph with resistor edges → `PowerGridModel` builds sparse Laplacian with Schur reduction for pad nodes → `IRDropSolver` or `EffectiveResistanceCalculator` compute results → `plot.py` visualizes

**Key Constraint**: Pads (voltage sources) are Dirichlet boundary conditions at Vdd, eliminated from system via Schur complement. LU factorization is cached and reused across batch solves for performance.

## Critical Domain Conventions

### Node & Resistance Model
- **NodeID structure**: `NodeID(layer, idx)` frozen dataclass keys the graph
- **Load placement**: Loads exist ONLY on layer 0, inserted mid-segment (never at via/intersection nodes)
- **Pad placement**: Pads on top layer (K-1), evenly spaced along boundaries; may truncate if `N_vsrc` exceeds available endpoints
- **Resistance scaling**: Each layer up halves stripe and via resistances (`R_layer = max_R / 2^layer`)

### Current Sign Convention (CRITICAL)
- **Input**: Positive current = sink drawing from grid (e.g., `currents[node] = +1.0`)
- **Internal**: Solver negates to `-I` for nodal equation `G·V = I` (injections are negative for sinks)
- **IR-drop**: Always `Vdd - V_node` (positive means voltage dropped below Vdd)

### Common Type Errors
- **Plotting**: `plot_ir_drop_map(G, voltages, vdd=1.0, ...)` requires scalar `vdd`, NOT pad list (common mistake)
- **Stimulus area**: `StimulusGenerator(graph=G, ...)` must pass graph if using `area=(x0,y0,x1,y1)` parameter
- **R_eff queries**: Pad nodes rejected in pairwise resistance calculations (raises `ValueError`)

## Module Responsibilities

### `generate_power_grid.py`
Constructs K-layer resistor mesh with alternating H/V orientation. Stripe count halves per layer (ceil(N0/2^ℓ), min 1). Returns `(G, loads, pads)` where loads map `NodeID → current`.

**Key parameters**: `K` (layers), `N0` (layer-0 stripes), `I_N` (load insertion attempts), `N_vsrc` (pad count), resistances, `seed`

### `irdrop/power_grid_model.py`
Builds sparse conductance matrix `G` from resistor edges (`g = 1/R`). Performs Schur reduction to eliminate pad nodes (fixed Vdd). Caches `(G_uu, G_up, lu)` in `ReducedSystem`.

**Critical**: Sign flip on input currents; batch solves reuse constant term `−G_up·V_p`

### `irdrop/solver.py`
Single/batch voltage solves returning `IRDropResult(voltages, ir_drop, metadata)`. Use `solver.summarize(result)` for min_voltage/max_drop/avg_drop stats.

### `irdrop/stimulus.py`
Allocates total_power across load nodes. `distribution='uniform'` splits evenly; `'gaussian'` uses |N(0,1)| weights. Optional `percent` samples subset; `area` rectangle filters candidates (requires `graph`).

### `irdrop/effective_resistance.py`
Batch R_eff computation. Pairs `(u, None)` compute to ground (uses diagonal of `G_uu^-1`); `(u, v)` pairwise. Solves sparse systems per unique node (no dense inversion). Rejects pad nodes for pairwise.

### `irdrop/grid_partitioner.py`
**Structured slab partitioning** (deterministic). Axis options: `'x'` (vertical slabs), `'y'` (horizontal), `'auto'` (selects best balance).

**Constraints**: 
- Separators = via rows/columns (no loads)
- X-axis: layer-0 vias; Y-axis: layer-1+ vias (fewer separators)
- Via-column snapping can cause load imbalance (ratio ≤3.5 single axis; <2.0 auto)
- Pads never partitioned; degree≤1 nodes excluded as separators

**Connectivity**: May have disconnected interior regions with loads (geometric clustering artifact). Use `get_partition_connectivity_info(G, pads)` to check.

### `irdrop/plot.py`
Voltage/IR-drop/current heatmaps. Headless default (`Agg` backend). Current map derives `I = (V_u - V_v)/R` per edge. Pass `show=False` for batch/headless.

## Typical Workflow Patterns

### Single Stimulus Solve
```python
G, loads, pads = generate_power_grid(K=3, N0=12, I_N=150, N_vsrc=4, 
                                     max_stripe_res=1.0, max_via_res=0.1,
                                     seed=42, plot=False)
model = PowerGridModel(G, pad_nodes=pads, vdd=1.0)
stim_gen = StimulusGenerator(load_nodes=list(loads.keys()), vdd=1.0, seed=10, graph=G)
meta = stim_gen.generate(total_power=1.5, percent=0.3, distribution="gaussian")
solver = IRDropSolver(model)
result = solver.solve(meta.currents)
print(solver.summarize(result))
```

### Batch Power Sweep (Reuses LU)
```python
powers = [0.5, 1.0, 2.0, 4.0]
metas = stim_gen.generate_batch(powers, percent=0.25, distribution='uniform')
results = solver.solve_batch([m.currents for m in metas], metas)
for m, r in zip(metas, results):
    print(f"P={m.total_power}W: {solver.summarize(r)}")
```

### Effective Resistance Batch
```python
calc = EffectiveResistanceCalculator(model)
nodes = list(loads.keys())
# Mix ground and pairwise queries
pairs = [(nodes[0], None), (nodes[1], None), (nodes[0], nodes[1])]
reff = calc.compute_batch(pairs)  # Returns numpy array
```

### Structured Partitioning
```python
from irdrop.grid_partitioner import GridPartitioner
partitioner = GridPartitioner(G, loads, pads, seed=42)
result = partitioner.partition(P=4, axis='auto', balance_tolerance=0.15)
# Check: result.load_balance_ratio, len(result.separator_nodes)
fig, ax = partitioner.visualize_partitions(result, show=False)
```

## Testing & Validation

**Run tests**: `python -m unittest discover -s tests -p 'test_*.py'`

**Key invariants tested** (see `tests/test_irdrop.py`, `tests/test_partitioner.py`):
- Zero load → all nodes at pad voltage
- Monotonic min voltage decrease with increasing power
- R_eff symmetry: `R(u,v) == R(v,u)`
- R_eff triangle inequality: `R(u,w) <= R(u,v) + R(v,w)`
- Partition balance ratio ≤3.5 (via-column constraints); auto <2.0
- Pads excluded from partitions; separators have no loads
- Graph topology preserved (no edge removal)

**Test helpers**: `build_small()` creates standard test grid (K=3, N0=8, I_N=80)

## Performance Guidelines

- **Reuse PowerGridModel**: Build once, solve many stimuli (LU factorization cached)
- **Batch effective resistance**: Group queries to amortize sparse solves (one per unique node)
- **Partitioning**: Structured O(N log N) in load nodes; faster and more reliable than k-means
- **Plotting**: Use `show=False` for headless; filter via nodes in current maps if too dense

## Extension Points

### New Stimulus Distribution
Add branch in `StimulusGenerator.generate()`. Must normalize total current to `total_power / vdd`. Return updated `StimulusMeta` with selected nodes.

### New Metrics
Derive from `result.voltages` or `result.ir_drop` post-solve. Keep batch-aware (operate on list of results).

### New Partition Strategy
Return `PartitionResult` with `partitions`, `separator_nodes`, `boundary_edges`. Must preserve graph (no edge removal). Follow structured pattern for determinism.

## Common Pitfalls

1. **Pad vs vdd confusion**: `plot_ir_drop_map(G, voltages, vdd=1.0, ...)` needs float, not list
2. **Area filtering**: Must pass `graph=G` to `StimulusGenerator` to use `area` parameter
3. **Partition count**: Cannot exceed load node count; raises ValueError
4. **Gaussian degeneracy**: Falls back to uniform if weights sum to zero
5. **Empty plots**: Very small node subsets may produce empty scatter; check `len(nodes) > 0`

## File Landmarks

- **Examples**: `example_ir_drop.py`, `example_partitioning.py`, `example_effective_resistance.py`
- **Tests**: `tests/test_irdrop.py` (16 tests), `tests/test_partitioner.py`
- **API exports**: `irdrop/__init__.py` defines public interface
- **Documentation**: `EFFECTIVE_RESISTANCE_SUMMARY.md`, `GRID_PARTITIONING.md`
- **Dependencies**: `requirements.txt` (networkx, matplotlib, numpy, scipy, pytest)

## Development Workflow

**Install**: `pip install -r requirements.txt` (use csh shell)
**Run examples**: `python example_ir_drop.py` or `example_partitioning.py`
**Test**: `python -m unittest discover -s tests -p 'test_*.py'`
**Debug**: Use headless plots (`show=False`) then `fig.savefig()` for batch inspection

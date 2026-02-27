# Model Package

Power grid model, factory functions, adapters, and result data classes.

## Key Classes

- **UnifiedPowerGridModel** (`unified_model.py`): Handles both NodeID and string nodes; auto-detects floating islands
- **NodeInfoExtractor** (`node_adapter.py`): Adapts different graph node representations
- **EdgeInfoExtractor** (`edge_adapter.py`): Adapts different graph edge representations

## Factory Functions

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

## Result Data Classes (`solver_results.py`)

- `UnifiedSolveResult`: Basic solve result with voltages, ir_drop, metadata
- `UnifiedHierarchicalResult`: Hierarchical result with port_nodes, port_voltages, port_currents, aggregation_map
- `UnifiedCoupledHierarchicalResult`: Coupled solver result with iterations, final_residual, converged, timings
- `TiledBottomGridResult`: Tiled solve result with tiles, per_tile_solve_times, validation_stats

## Solver Context Classes (`solver_results.py`)

For batch solving (see `src/solver/CLAUDE.md` for usage):

- `FlatSolverContext`: Caches reduced system LU factorization for repeated flat solves
- `HierarchicalSolverContext`: Caches top/bottom grid systems and shortest-path distances
- `CoupledHierarchicalSolverContext`: Caches block matrices, Schur complement operator, preconditioner
- `TiledHierarchicalSolverContext`: Caches top-grid system, tile structure, path distances

## Enums (`edge_adapter.py`)

- `GridSource.SYNTHETIC`, `GridSource.PDN_NETLIST`: Source type detection
- `ElementType.RESISTOR`, `ElementType.CAPACITOR`, `ElementType.INDUCTOR`, `ElementType.CURRENT_SOURCE`

## Graph Converter (`../graph/converter.py`)

For legacy pickle files containing NetworkX graphs:

```python
from graph.converter import detect_graph_type, ensure_rustworkx_graph

# Detect graph type from pickle
graph_type = detect_graph_type(graph)  # Returns 'networkx', 'rustworkx', or 'unknown'

# Auto-convert NetworkX to Rustworkx if needed
graph = ensure_rustworkx_graph(graph)  # No-op if already Rustworkx
```

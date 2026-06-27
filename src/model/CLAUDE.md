# `src/model/` — model layer

> Root `CLAUDE.md` covers the data-flow skeleton, units, and sign convention. This file is the API and internals reference for the model layer.

## Public API

```python
from model.factory import (
    create_model_from_synthetic,    # synthetic NodeID grids
    create_model_from_pdn,          # PDN string-node graphs
    create_multi_net_models,        # all nets in a parsed PDN graph
    create_model_from_graph,        # auto-detecting (synthetic vs PDN)
)
```

| Factory | Inputs | Notes |
|---------|--------|-------|
| `create_model_from_synthetic(graph, pad_nodes, vdd, lazy_factor=True)` | rustworkx graph + explicit pad list + scalar Vdd | NodeID-based |
| `create_model_from_pdn(graph, net_name, lazy_factor=True)` | parsed PDN graph + net string | Vdd auto-extracted from `pg_net_voltage` / vsrc edges |
| `create_multi_net_models(graph, net_filter=None)` | parsed PDN graph + optional net allowlist | returns `{net_name: model}` |
| `create_model_from_graph(graph, pads, vdd, auto_detect_source=True)` | mixed | dispatches based on node-key type |

### Lazy factorization

Default is `lazy_factor=True` — LU factorization is deferred until the first flat solve. ~4.6× faster model creation when only hierarchical/distributed solves will follow (those build their own systems). Set `lazy_factor=False` for backward compat or pure flat-solve workflows.

The factor function is resolved through `_get_factor_func()` which honors the active CHOLMOD backend (see `src/solver/CLAUDE.md`).

## `UnifiedPowerGridModel` (`unified_model.py`)

- Handles both `NodeID` and string nodes through the adapter pair (`NodeInfoExtractor`, `EdgeInfoExtractor`).
- Auto-detects floating islands and excludes them from the conductance matrix; ground node `'0'` is special-cased (excluded from G but preserved for I-type edges).
- `extract_current_sources()` pulls load currents from I-type edges and returns `{node: mA}` — use this instead of reaching into the graph yourself.
- `EdgeArrayCache` and `UnifiedReducedSystem` are internal; don't touch from user code.

## Adapters (`node_adapter.py`, `edge_adapter.py`)

These hide the synthetic/PDN graph differences from the rest of the stack:

- `NodeInfoExtractor` → `UnifiedNodeInfo(coords, layer, kind, …)`
- `EdgeInfoExtractor` → `UnifiedEdgeInfo(element_type, value, …)`

Two enums you'll touch:

- `GridSource.{SYNTHETIC, PDN_NETLIST}` — set when the model is built.
- `ElementType.{RESISTOR, CAPACITOR, INDUCTOR, CURRENT_SOURCE, VOLTAGE_SOURCE}` — element classifier on edges.

## Result / context dataclasses (`solver_results.py`)

This package owns *all* solver result types so the solver package can import them without circular dependencies. The solver package re-exports them.

**Results:**

- `UnifiedSolveResult` — flat: `voltages`, `ir_drop`, `metadata`
- `UnifiedHierarchicalResult` — adds `port_nodes`, `port_voltages`, `port_currents`, `aggregation_map`
- `UnifiedCoupledHierarchicalResult` — extends with `iterations`, `final_residual`, `converged`, `preconditioner_type`, `timings`
- `TiledBottomGridResult` — extends with `tiles: List[BottomGridTile]`, `per_tile_solve_times`, `validation_stats`
- `TileBounds`, `BottomGridTile`, `TileSolveResult` — used inside tiled results

**Contexts (for batch solving):**

- `FlatSolverContext` — caches reduced-system LU
- `HierarchicalSolverContext` — caches top + bottom systems and shortest-path distances
- `CoupledHierarchicalSolverContext` — caches block matrices, Schur operator, preconditioner
- `TiledHierarchicalSolverContext` — caches top-grid system, tile structure, path distances

Distributed contexts live in `src/distributed/result.py` because their lifecycle (factor / release / save / load / refactor across workers) doesn't fit a dataclass-only file.

## Graph conversion (`../graph/converter.py`)

Older pickle artifacts contain NetworkX graphs. The factories tolerate either, but:

```python
from graph.converter import detect_graph_type, ensure_rustworkx_graph

graph = ensure_rustworkx_graph(graph)  # no-op if already rustworkx
```

Always call this at load boundaries; rustworkx is the internal representation everywhere downstream.

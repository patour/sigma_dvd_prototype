# Import Path Migration Reference (COMPLETED)

**Migration Status: COMPLETE.** The old shim directories (`core/`, `pdn/`,
`irdrop/`, `analysis/`, and root-level `generate_power_grid.py`) have been
deleted. All code must use the canonical `src/` package imports listed below.

This document is retained as a historical reference for the old-to-new
import path mappings.

## Import Path Mappings

### Graph (`core/` → `src/graph/`)

| Old Path | New Canonical Path |
|----------|-------------------|
| `core.rx_graph` | `graph.rx_graph` |
| `core.rx_algorithms` | `graph.rx_algorithms` |
| `core.graph_converter` | `graph.converter` |

### Model (`core/` → `src/model/`)

| Old Path | New Canonical Path |
|----------|-------------------|
| `core.unified_model` | `model.unified_model` |
| `core.node_adapter` | `model.node_adapter` |
| `core.edge_adapter` | `model.edge_adapter` |
| `core.factory` | `model.factory` |
| `core.solver_results` | `model.solver_results` |

### Solver (`core/` → `src/solver/`)

| Old Path | New Canonical Path |
|----------|-------------------|
| `core.unified_solver` | `solver.unified_solver` |
| `core.coupled_system` | `solver.coupled_system` |
| `core.tiling` | `solver.tiling` |
| `core.current_aggregation` | `solver.current_aggregation` |
| `core.unified_partitioner` | `solver.unified_partitioner` |
| `core.effective_resistance` | `solver.effective_resistance` |
| `core.statistics` | `solver.statistics` |
| `pdn.pdn_solver` | `solver.pdn_solver` |

### Analysis (`core/` → `src/analysis/`)

| Old Path | New Canonical Path |
|----------|-------------------|
| `core.dynamic_solver` | `analysis.dynamic_solver` |
| `core.transient_solver` | `analysis.transient_solver` |
| `core.adjoint_sensitivity` | `analysis.adjoint_sensitivity` |
| `core.pwl_smoothing` | `analysis.pwl_smoothing` |
| `core.vectorized_sources` | `analysis.vectorized_sources` |
| `core.farfield_analysis` | `analysis.farfield_analysis` |

### Parser (`pdn/` → `src/parser/`)

| Old Path | New Canonical Path |
|----------|-------------------|
| `pdn.pdn_parser` | `parser.netlist` (main parser) |
| `pdn.netlist` | `parser.netlist` |
| `pdn.spice_lexer` | `parser.spice_lexer` |
| `pdn.current_sources` | `parser.current_sources` |
| `pdn.graph_builder` | `parser.graph_builder` |
| `pdn.metadata` | `parser.metadata` |
| `pdn.parallel_parser` | `parser.parallel` |
| `pdn.edge_attrs` | `parser.edge_attrs` |
| `pdn.pdn_plotter` | `visualization.pdn_plotter` |
| `pdn.generate_sampled_netlist` | `parser.generate_sampled_netlist` |

### Distributed (`core.distributed` → `src/distributed/`)

| Old Path | New Canonical Path |
|----------|-------------------|
| `core.distributed` | `distributed` |
| `core.distributed.distributed_model` | `distributed.model` |
| `core.distributed.distributed_solver` | `distributed.solver` |
| `core.distributed.distributed_parser` | `distributed.parser` |
| `core.distributed.tile_worker` | `distributed.tile_worker` |
| `core.distributed.backend` | `distributed.backend` |
| `core.distributed.result` | `distributed.result` |

### Visualization (`core/` → `src/visualization/`)

| Old Path | New Canonical Path |
|----------|-------------------|
| `core.unified_plotter` | `visualization.unified_plotter` |
| `core.dynamic_plotter` | `visualization.dynamic_plotter` |
| `analysis.stripe_heatmap` | `visualization.stripe_heatmap` |

### Legacy (`irdrop/` → `src/legacy/`)

| Old Path | New Canonical Path |
|----------|-------------------|
| `irdrop` | `legacy` |
| `irdrop.power_grid_model` | `legacy.power_grid_model` |
| `irdrop.solver` | `legacy.solver` |
| `irdrop.stimulus` | `legacy.stimulus` |
| `irdrop.effective_resistance` | `legacy.effective_resistance` |
| `irdrop.grid_partitioner` | `legacy.grid_partitioner` |
| `irdrop.regional_voltage_solver` | `legacy.regional_voltage_solver` |
| `irdrop.plot` | `legacy.plot` |
| `generate_power_grid` | `legacy.generate_power_grid` |

### Aggregated (`core` package) -- REMOVED

The `core` package previously aggregated all public symbols. It has been
deleted. Import directly from the specific package instead
(e.g., `from model.factory import create_model_from_pdn`).

## Data Directory

| Old Path | New Path |
|----------|----------|
| `pdn/netlist_test/` | `netlist/netlist_test/` |
| `pdn/netlist_small/` | `netlist/netlist_small/` |
| `pdn/netlist_multi_tile/` | `netlist/netlist_multi_tile/` |
| `pdn/netlist_sampled/` | `netlist/netlist_sampled/` |

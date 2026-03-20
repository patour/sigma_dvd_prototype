# Tests

Test layout mirrors `src/`:

```
tests/
├── conftest.py     # Sets matplotlib Agg backend for headless testing
├── fixtures.py     # Factory functions for edge case testing
├── graph/          # test_rx_graph, test_rx_algorithms
├── model/          # test_unified_core
├── solver/         # test_hierarchical_solver, test_coupled_hierarchical_solver,
│                   # test_batch_solving, test_regional_solver, test_pdn_solver,
│                   # test_interface_islands, test_tiled_accuracy,
│                   # test_hierarchical_integration (integration)
├── analysis/       # test_dynamic_solver, test_transient_solver, test_transient_multi_rhs,
│                   # test_adjoint_sensitivity, test_pwl_smoothing, test_vectorized_sources,
│                   # test_smoothing_source_idx, test_dynamic_integration (integration)
├── parser/         # test_pdn_parser, test_parallel_parser, test_edge_attrs,
│                   # test_parser_regression, test_pdn_integration (integration)
├── distributed/    # test_distributed_solver, test_distributed_heatmap,
│                   # test_distributed_cli, test_time_domain,
│                   # test_distributed_integration, test_time_domain_integration (integration)
├── reports/        # test_floating_nodes, test_topk_irdrop,
│                   # test_floating_nodes_consistency (integration)
├── visualization/  # test_pdn_plotter, test_stripe_heatmap
└── legacy/         # test_irdrop, test_partitioner
```

## Test Netlists

- `netlist/netlist_test/` — Small PDN for integration tests
- `netlist/netlist_small/` — Minimal unit test fixtures

## Key Invariants Tested

- Zero load -> all nodes at pad voltage
- R_eff symmetry: `R(u,v) == R(v,u)` and triangle inequality
- Partition balance ratio <= 3.5; pads excluded from partitions
- Floating island detection removes disconnected components

## Test Helper

`build_small()` in `fixtures.py` creates standard test grid (K=3, N0=8, I_N=80).
`_build_two_tile_distributed_model()` in `distributed/test_time_domain.py` creates a minimal 2-tile distributed model with optional cap edges for unit tests.

## Running Tests

```bash
# Run unit tests (fast, <160s)
pytest -m unit

# Run a specific integration test file
pytest tests/distributed/test_distributed_integration.py -v

# Run ALL integration tests (~6 min, slow — only when needed)
pytest -m integration

# Run everything (~10 min, slow — only as a final check)
pytest

# Specific module
pytest tests/solver/test_hierarchical_solver.py

# Single test
pytest tests/legacy/test_irdrop.py::TestIRDrop::test_no_load_currents_all_pad_voltage -v
```

> **Test priority:** First run individual unit test files related to the affected
> area, then `pytest -m unit` for full unit coverage. Run individual integration
> test files only when changes affect that area. Run bare `pytest` only as a
> final validation step, not during intermediate steps.

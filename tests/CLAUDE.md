# Tests

Test layout mirrors `src/`:

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

## Running Tests

```bash
# All tests (~984 tests)
pytest

# Slow integration tests
pytest tests/solver/test_hierarchical_integration.py tests/analysis/test_dynamic_integration.py tests/parser/test_pdn_integration.py

# Specific module
pytest tests/solver/test_hierarchical_solver.py

# Single test
pytest tests/legacy/test_irdrop.py::TestIRDrop::test_no_load_currents_all_pad_voltage -v
```

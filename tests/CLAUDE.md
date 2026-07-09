# `tests/`

> Root `CLAUDE.md` lists the `pytest` invocations and the test-priority rule (unit first, integration only for affected area, bare `pytest` as final check). This file documents the layout, fixtures, and conventions.

## Layout

Mirrors `src/`. ~984 tests total.

```
tests/
├── conftest.py     # forces matplotlib 'Agg' backend
├── fixtures.py     # canonical netlist paths + small-graph factories
├── graph/          # rustworkx wrappers, NX↔RX conversion
├── model/          # UnifiedPowerGridModel + factories
├── solver/         # flat / hierarchical / coupled / tiled / batch / regional / interface_islands
├── analysis/       # dynamic / transient / adjoint / PWL smoothing / vectorized sources
├── parser/         # PDN parser, parallel parser, edge attrs, regression
├── distributed/    # DDM solver, heatmap, time domain, adjoint, CLI
├── reports/        # floating nodes, top-K IR-drop
├── visualization/  # PDN plotter, stripe heatmap
└── legacy/         # original synthetic IR-drop, partitioner
```

## Markers

| Marker | Meaning | Fixtures |
|---|---|---|
| `@pytest.mark.unit` | fast, isolated; no external netlist needed | minimal in-memory graphs from `fixtures.py` |
| `@pytest.mark.integration` | slow; needs real netlist data; one of the `_integration.py` files | uses `NETLIST_TEST` / `NETLIST_SMALL` / `NETLIST_MULTI_TILE` / `NETLIST_SAMPLED` |

Both markers are registered in `pyproject.toml`. Files named `test_<topic>_integration.py` are integration; everything else is unit.

## Test netlists

`fixtures.py` defines canonical paths — **always import these instead of hardcoding `pdn/...` paths**:

```python
from fixtures import (
    NETLIST_TEST,          # netlist/netlist_test/    — small PDN, integration
    NETLIST_SMALL,         # netlist/netlist_small/   — minimal unit fixtures
    NETLIST_MULTI_TILE,    # netlist/netlist_multi_tile/
    NETLIST_SAMPLED,       # netlist/netlist_sampled/ — distributed benchmark
)
```

## Helper factories

`fixtures.create_minimal_pdn_graph(scenario)` builds tiny PDN graphs for edge cases that aren't reachable through `netlist_small`:

| Scenario | Triggers |
|---|---|
| `'tile_merging'` | Phase-3 merge: 4×4 M1 + 2×2 M2 with loads clustered in one corner; 2×2 tiling produces 0-load tiles |
| `'path_expansion'` | Sparse vias → locally disconnected core nodes → halo path expansion |
| `'severe_halo_clip'` | 6×6 grid with 3×3 tiling clipping halos >70% |

`tests/distributed/test_time_domain.py::_build_two_tile_distributed_model(...)` is the standard fixture for minimal 2-tile distributed models with optional cap edges. Reuse it; don't roll your own.

## Invariants the suite guards

These are the load-bearing checks — break one and you almost certainly broke physics:

- Zero load → all nodes at pad voltage
- `R_eff(u, v) == R_eff(v, u)` and triangle inequality (within tolerance)
- Partition balance ratio ≤ 3.5; pads excluded from partitions
- Floating-island detection removes disconnected components
- DDM exactness: distributed solver matches flat to floating-point precision (0 µV diff) on validation graphs; B1 split-vs-unsplit is DC-exact and transient machine-precision (≤ 2e-14 V) for one-level bisections; `TestForcedSplitNetlistSampled` (integration) guards this
- PWL smoothing equivalence: `TestSparseVsDenseEquivalence`, `TestSparseSmoothingFunctions` in `tests/analysis/test_pwl_smoothing.py`

## Naming

- `test_<topic>.py` — unit
- `test_<topic>_integration.py` — integration

Within a file, classes group related scenarios (`TestHierarchicalCoupledExactness`, `TestTransientStability`, etc.). Keep that pattern when adding new tests.

## Running a single case

```bash
pytest tests/legacy/test_irdrop.py::TestIRDrop::test_no_load_currents_all_pad_voltage -v
pytest tests/distributed/test_distributed_solver.py -k "schur and not benchmark" -v
```

`-m "unit and not integration"` is implicit when you pass `-m unit` (integration tests aren't picked up unless explicitly requested), but `-k` is the easiest way to slice by name within a marker.

## Known flakey tests

- `tests/analysis/test_pwl_smoothing.py::TestSmoothedEvaluationPerformance::test_smoothed_evaluate_at_time_within_2x_original` — measures `evaluate_at_time` speed (not smoothing); marginally fails the ~2.0× threshold on slow runners.

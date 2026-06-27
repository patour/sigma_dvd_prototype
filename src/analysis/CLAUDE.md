# `src/analysis/` — dynamic, transient, adjoint, far-field

> Root `CLAUDE.md` covers conventions and lists the unit/sign rules. This file is the API reference for time-domain analysis built on top of the model + solver layers.

## Quasi-static (batch DC)

`DynamicIRDropSolver` ignores capacitance and solves an independent DC problem at each time point. Cheap, correct steady-state envelope.

```python
from analysis.dynamic_solver import DynamicIRDropSolver

dyn = DynamicIRDropSolver(model, graph)
res = dyn.solve_quasi_static(
    t_start=0, t_end=100e-9, n_points=101,
    method='flat',         # or 'hierarchical'
    n_worst_nodes=10,      # track top-N worst-case nodes
    track_nodes=['1000_2000_M1'],   # full waveforms for these nodes
)
```

`QuasiStaticResult` fields: `t_array`, `peak_ir_drop`, `peak_ir_drop_time`, `peak_ir_drop_node`, `worst_nodes` (list of `(node, max_drop, time)`), `max_ir_drop_per_time`, `total_current_per_time`, `total_vsrc_current_per_time`, `peak_ir_drop_per_node`, `peak_current_per_node`, `tracked_waveforms`, `tracked_ir_drop`.

## Transient RC (with capacitance)

`TransientIRDropSolver` does implicit time integration over `G + α·C`.

```python
from analysis.transient_solver import TransientIRDropSolver, IntegrationMethod

trn = TransientIRDropSolver(model, graph)
res = trn.solve_transient(
    t_start=0, t_end=100e-9, dt=0.1e-9,
    method=IntegrationMethod.BACKWARD_EULER,   # or TRAPEZOIDAL
    n_worst_nodes=10,
    track_nodes=['1000_2000_M1'],
)
```

`TransientResult` adds `integration_method` and timings (`build_rc`, `factor`, `solve`). Capacitors smooth waveforms, so peaks are typically lower than quasi-static.

There's also `solve_transient_multi_rhs(...)` for batched RHS over a fixed factor — useful for sensitivity sweeps.

## PWL smoothing (`pwl_smoothing.py`)

Analytical convolution of a triangular low-pass window with PWL/Pulse waveforms. Used to avoid aliasing when the solver step is coarser than waveform features.

Public types:

- `PWLSmoother` — main entry point
- `SmoothingConfig` — half-width, output sampling, mode
- `SmoothedWaveformCache` — keyed cache for repeated smoothing across solves
- `triangular_window(t, t_center, half_width)` — analytical kernel
- `smooth_pwl_points(...)`, `pulse_to_pwl_points(...)`, `compact_pwl(...)` — utility helpers

**Hot-path internals (don't touch unless profiling):**

- `_smooth_pwl_sparse` collects all output times (boundary + active) into a single array for one vectorized pass per segment.
- `_compact_chunk_vectorized` returns flat arrays `(kept_times_flat, kept_values_flat, offsets, counts)` — not per-waveform lists.
- `_compact_and_append` accepts those flat arrays + offsets and does one bulk `tolist()` per chunk.
- `_compact_arrays` is the numpy-native equivalent of `compact_pwl(list(zip(times, values)))`.

68 tests in `tests/analysis/test_pwl_smoothing.py`. The `TestSparseVsDenseEquivalence` and `TestSparseSmoothingFunctions` suites are the equivalence backbone. `TestSmoothedEvaluationPerformance.test_smoothed_evaluate_at_time_within_2x_original` is a flaky perf test (it measures `evaluate_at_time` speed, not smoothing) that marginally fails the ~2.0× threshold.

## Vectorized current sources (`vectorized_sources.py`)

`VectorizedCurrentSources` evaluates many `CurrentSource`s simultaneously into a single dense vector at a given time. Used by both the dynamic solver and the distributed time-domain workers. Holds optional source-index → tile mappings.

## Adjoint sensitivity (`adjoint_sensitivity.py`)

Attribute IR-drop at a victim node back to aggressor current sources via a backward sweep.

```python
from analysis.adjoint_sensitivity import AdjointSensitivitySolver

adj = AdjointSensitivitySolver(model, graph)
attribution = adj.analyze(victim_node, t_window, ...)   # returns AdjointAttribution
```

Types: `AdjointSolverContext`, `AdjointAttribution`, `AggressorContribution`. The DDM equivalent is in `src/distributed/solver_adjoint.py`.

## Far-field analysis (`farfield_analysis.py`)

Decompose a global solve into a local window + far-field boundary coupling. Used to validate distributed near/far decomposition and to study how block injections far from a victim influence its boundary response.

Pipeline (top-level helpers):

1. `define_window_and_boundary(...)` — window box + boundary-node detection
2. `partition_farfield_into_blocks(...)` or `partition_farfield_by_distance_rings(...)`
3. `generate_block_injections(...)` — synthetic injection scenarios
4. `compute_boundary_response_matrix(...)` — boundary response per block
5. `analyze_response_matrix(...)` + `compute_boundary_smoothness(...)`
6. `validate_against_flat_solve(...)` — sanity-check against ground-truth flat
7. `run_farfield_analysis(...)` — full pipeline driver

## CLI: `dynamic_irdrop_decomposition`

```bash
python -m analysis.dynamic_irdrop_decomposition ./netlist/netlist_test \
    --net VDD --end-time 100ns --dt 100ps
```

Performs aggressor decomposition over the whole netlist. Public types: `AggressorResult`, `InstanceDecomposition`, `DecompositionResult`. Uses `parse_time_value` / `format_time_ns` helpers for CLI time strings. `configure_solver_backend(config)` flips CHOLMOD settings before solving.

Window-based partitioning helpers (`extract_instance_locations`, `compute_window_for_instance`, `windows_intersect`, `partition_sources_by_window`) are reused by the distributed near/far decomposition path.

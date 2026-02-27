# Analysis Package

Dynamic (time-domain) IR-drop analysis: quasi-static batch DC, transient RC, PWL smoothing, and adjoint sensitivity attribution.

## Quasi-Static Analysis (Batch DC Solves)

`DynamicIRDropSolver` in `dynamic_solver.py` — ignores capacitance, solves independent DC problems at each time point.

```python
from parser.netlist import NetlistParser
from model.factory import create_model_from_pdn
from analysis.dynamic_solver import DynamicIRDropSolver

parser = NetlistParser('./netlist_dir')
graph = parser.parse()
model = create_model_from_pdn(graph, 'VDD')

solver = DynamicIRDropSolver(model, graph)
result = solver.solve_quasi_static(
    t_start=0, t_end=100e-9, n_points=101,
    method='flat',           # 'flat' or 'hierarchical'
    n_worst_nodes=10,        # Track N worst-case nodes
    track_nodes=['node1'],   # Store full waveforms for these nodes
)

print(f"Peak IR-drop: {result.peak_ir_drop*1000:.2f} mV at t={result.peak_ir_drop_time*1e9:.2f} ns")
print(f"Peak node: {result.peak_ir_drop_node}")
```

**QuasiStaticResult Fields:**
- `t_array`: Time points array
- `peak_ir_drop`, `peak_ir_drop_time`, `peak_ir_drop_node`: Global peak info
- `worst_nodes`: List of (node, max_drop, time) for top N worst nodes
- `max_ir_drop_per_time`: Max IR-drop at each time step
- `total_current_per_time`, `total_vsrc_current_per_time`: Aggregate currents
- `peak_ir_drop_per_node`, `peak_current_per_node`: Spatial peaks for heatmaps
- `tracked_waveforms`, `tracked_ir_drop`: Waveforms for selected nodes

## Transient RC Analysis (with Capacitance)

`TransientIRDropSolver` in `transient_solver.py` — includes capacitance via implicit time integration.

```python
from analysis.transient_solver import TransientIRDropSolver, IntegrationMethod

solver = TransientIRDropSolver(model, graph)
result = solver.solve_transient(
    t_start=0, t_end=100e-9, dt=0.1e-9,
    method=IntegrationMethod.BACKWARD_EULER,  # or TRAPEZOIDAL
    n_worst_nodes=10,
    track_nodes=['node1'],
)

# Capacitors provide smoothing, so peak may be lower than quasi-static
print(f"Transient peak: {result.peak_ir_drop*1000:.2f} mV")
```

**Transient vs Quasi-Static:**
- **Quasi-static**: Ignores capacitance, independent DC at each time point. Faster, steady-state approximation.
- **Transient**: Implicit time integration (Backward Euler or Trapezoidal). Captures decoupling effects but slower.

**TransientResult Fields:**
- Same as QuasiStaticResult, plus `integration_method`
- Timings include `build_rc`, `factor`, `solve`

## Dynamic Analysis Plotting

```python
from visualization.dynamic_plotter import DynamicPlotter

# Peak IR-drop heatmap (worst IR-drop each node saw across time)
DynamicPlotter.plot_peak_ir_drop_heatmap(
    model, result, layer='M1',
    title='Peak IR-Drop During Transient',
    save_path='peak_ir_drop.png'
)

# Peak current heatmap
DynamicPlotter.plot_peak_current_heatmap(
    model, result, layer='M1',
    save_path='peak_current.png'
)

# Time series of aggregate metrics
DynamicPlotter.plot_time_series(
    result, metrics=['max_ir_drop', 'total_current', 'vsrc_current'],
    save_path='time_series.png'
)

# Node waveforms (for tracked nodes)
DynamicPlotter.plot_node_waveforms(
    result, plot_ir_drop=True,
    save_path='waveforms.png'
)
```

## PWL Waveform Smoothing (Preprocessing)

`PWLSmoother` in `pwl_smoothing.py` — analytical triangular low-pass filter for current waveforms.

**Algorithm:** Convolves each PWL segment with a triangular window (width = 2*time_step) using exact closed-form integration. Pulse waveforms are first converted to PWL. A compaction phase removes redundant collinear points after filtering.

**Basic Usage (Automatic):**
```python
from analysis.dynamic_solver import DynamicIRDropSolver

solver = DynamicIRDropSolver(model, graph)

# Preprocess waveforms (returns VectorizedCurrentSources with smoothed PWLs)
smoothed = solver.preprocess_sources(
    time_step=0.1e-9,      # Filter window = 2 x time_step
    t_start=0,
    t_end=100e-9,
    compact_threshold=1e-12,  # Collinearity threshold for compaction
)

# Use smoothed sources (reusable across multiple analyses)
result = solver.solve_quasi_static(
    t_start=0, t_end=100e-9, n_points=101,
    smoothed_sources=smoothed,  # Pass pre-smoothed sources
)
```

**Reusing Smoothed Sources Across Solvers:**
```python
# Preprocess once
smoothed = solver.preprocess_sources(time_step=0.1e-9, t_start=0, t_end=100e-9)

# Reuse for multiple quasi-static analyses
result1 = solver.solve_quasi_static(..., smoothed_sources=smoothed)
result2 = solver.solve_quasi_static(..., smoothed_sources=smoothed)

# Also works with transient solver
from analysis.transient_solver import TransientIRDropSolver
trans = TransientIRDropSolver(model, graph)
trans_smoothed = trans.preprocess_sources(time_step=0.1e-9, t_start=0, t_end=100e-9)
result3 = trans.solve_transient(..., smoothed_sources=trans_smoothed)
```

**Manual Smoothing (Low-Level API):**
```python
from analysis.pwl_smoothing import PWLSmoother, smooth_pwl_points, compact_pwl, pulse_to_pwl_points
from parser.current_sources import PWL, Pulse

smoother = PWLSmoother(time_step=0.1e-9, compact_threshold=1e-12)

# Smooth a single PWL waveform
pwl = PWL(points=[(0, 0), (1e-9, 1), (2e-9, 0)], period=10e-9)
smoothed_pwl = smoother.smooth_pwl(pwl, t_start=0, t_end=10e-9)

# Convert pulse to PWL and smooth
pulse = Pulse(v1=0, v2=1, delay=1e-9, rt=0.1e-9, ft=0.1e-9, width=2e-9, period=10e-9)
smoothed_from_pulse = smoother.smooth_pulse(pulse, t_start=0, t_end=10e-9)

# Direct function calls
points = [(0, 0), (1e-9, 1), (2e-9, 0)]
smoothed = smooth_pwl_points(points, period=10e-9, time_step=0.1e-9, t_start=0, t_end=10e-9)
compacted = compact_pwl(smoothed, threshold=1e-12)
```

**SmoothedWaveformCache for Batch Analysis:**
```python
# Create cache from VectorizedCurrentSources
vec_sources = solver._vec_sources  # Internal vectorized sources
cache = smoother.create_smoothed_cache(vec_sources, t_start=0, t_end=100e-9)

# Check cache compatibility
if cache.is_compatible(time_step=0.1e-9, t_start=0, t_end=100e-9):
    smoothed = smoother.apply_cache_to_sources(vec_sources, cache)
```

**Key Functions:**

| Function | Description |
|----------|-------------|
| `smooth_pwl_points()` | Apply triangular LP filter to PWL points |
| `compact_pwl()` | Remove collinear/redundant points |
| `pulse_to_pwl_points()` | Convert Pulse to PWL representation |
| `triangular_window()` | Evaluate triangle window function |
| `analytical_triangle_pwl_integral()` | Exact triangle x PWL integral |

**Smoothing Effect:**
- Preserves DC average (energy conservation)
- Removes high-frequency content above ~1/(2*time_step)
- Reduces numerical noise in transient analysis
- Compaction reduces memory for long simulations

## Adjoint Sensitivity Analysis (IR-Drop Attribution)

`AdjointSensitivitySolver` in `adjoint_sensitivity.py` — identifies which aggressor current sources contribute most to IR-drop at a victim node.

```python
from analysis.transient_solver import TransientIRDropSolver
from analysis.adjoint_sensitivity import AdjointSensitivitySolver

trans = TransientIRDropSolver(model, graph)
result = trans.solve_transient(t_start=0, t_end=100e-9, dt=1e-9)

# Analyze the worst node at peak time
victim = result.peak_ir_drop_node
T = result.peak_ir_drop_time

adjoint = AdjointSensitivitySolver.from_transient_solver(trans)
attribution = adjoint.analyze_victim(
    victim_node=victim,
    observation_time=T,
    memory_window=20,      # L time steps to look back
    dt=1e-9,
    top_k=10,              # Return top 10 aggressors
    spatial_window=(x_v - 500, x_v + 500, y_v - 500, y_v + 500),  # Optional spatial filter
)

print(f"Victim: {attribution.victim_node}")
print(f"Total IR-drop: {attribution.ir_drop_at_T:.2f} mV")
print(f"Self-contribution: {attribution.self_contribution_mV:.3f} mV ({attribution.self_contribution_pct:.1f}%)")

for i, agg in enumerate(attribution.top_aggressors, 1):
    print(f"  {i}. {agg.node}: {agg.contribution_mV:.3f} mV ({agg.contribution_pct:.1f}%)")
```

**Two Methods:**
1. **Dynamic Adjoint** (`analyze_victim`): Propagates sensitivities backward through RC network's memory. For stiff systems (tau << dt), converges to static result.
2. **Static Sensitivity** (`analyze_victim_static` or `use_static=True`): Uses steady-state G^-1. Faster for stiff RC systems.

**Initial Condition Options:**
- `initial_condition='zero'` (default): V=VDD at start. Computes contributions to **total** IR-drop at T.
- `initial_condition='dc'`: DC operating point start. Computes **incremental** IR-drop (above DC baseline).

Use `'dc'` to analyze switching-induced IR-drop separately from static leakage.

**When to Use Static vs Dynamic:**
- `use_static=True` for faster analysis when tau << dt (typical PDN grids)
- Dynamic method captures time-varying effects for grids with significant decoupling

**Vectorization Threshold:**
```python
# Force vectorized current evaluation (faster for many sources)
adjoint = AdjointSensitivitySolver(model, graph, vectorize_threshold=0)

# Or disable vectorization (uses raw CurrentSource objects)
adjoint = AdjointSensitivitySolver(model, graph, vectorize_threshold=100000)
```
Default threshold is 10000 sources. Both modes produce identical results.

**Static Method Example (Recommended for Most PDNs):**
```python
attribution = adjoint.analyze_victim_static(
    victim_node=victim,
    observation_time=T,
    top_k=10,
)
# OR equivalently:
attribution = adjoint.analyze_victim(victim, T, use_static=True, top_k=10)
```

**DC Initial Condition Example (Incremental Attribution):**
```python
attribution = adjoint.analyze_victim(
    victim_node=victim,
    observation_time=T,
    initial_condition='dc',
    top_k=10,
)

print(f"Total IR-drop at T: {attribution.ir_drop_at_T:.2f} mV")
print(f"DC baseline IR-drop: {attribution.dc_ir_drop_mV:.2f} mV")
incremental = attribution.ir_drop_at_T - attribution.dc_ir_drop_mV
print(f"Incremental IR-drop: {incremental:.2f} mV")
```

**AdjointAttribution Fields:**
- `victim_node`, `observation_time`, `ir_drop_at_T`: Victim info (always total: VDD - V_T)
- `memory_window`, `t_array`: Time window analyzed
- `self_contribution_mV`, `self_contribution_pct`: Victim's own current contribution
- `top_aggressors`: List of `AggressorContribution`
- `attribution_efficiency`: Ratio of total_attributed / IR-drop (~1.0 for static)
- `initial_condition`: `'zero'` or `'dc'`
- `dc_ir_drop_mV`: DC baseline (only when `initial_condition='dc'`). Incremental = `ir_drop_at_T - dc_ir_drop_mV`

**AggressorContribution Fields:**
- `node`: Aggressor node name
- `contribution_mV`: In 'zero' mode: total. In 'dc' mode: incremental (from delta_I)
- `contribution_pct`: Percentage of attributed IR-drop
- `source_names`: List of current source instance names
- `current_waveform`: Optional I(t) waveform over memory window
- `static_contribution_mV`: Static (DC) contribution (only in 'dc' mode). Total = `contribution_mV + static_contribution_mV`

**Batch Attribution (multiple victims):**
```python
ctx = adjoint.prepare(dt=1e-9)  # Prepare once, caches LU factorization
for victim in victims:
    result = adjoint.analyze_victim(victim, T, context=ctx)
```

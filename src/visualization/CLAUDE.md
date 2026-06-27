# `src/visualization/` — plotters

> Root `CLAUDE.md` notes the `Agg` backend setup and the heatmap-binning pitfalls. This file is the API reference for the four plotter modules.

## Modules

| File | Class / Functions | Use for |
|---|---|---|
| `unified_plotter.py` | `UnifiedPlotter`, `plot_voltage_map`, `plot_ir_drop_map` | static DC results (synthetic + PDN) |
| `dynamic_plotter.py` | `DynamicPlotter` (+ `plot_peak_ir_drop_heatmap`, `plot_peak_current_heatmap`, `plot_time_series`, `plot_node_waveforms`) | dynamic / transient results |
| `pdn_plotter.py` | `PDNPlotter` | layer-wise PDN heatmaps with orientation-aware bins |
| `stripe_heatmap.py` | helpers (see below) | shared stripe-binning kernel; reused by distributed pipeline |

## `UnifiedPlotter` / `plot_ir_drop_map`

```python
from visualization import plot_voltage_map, plot_ir_drop_map

plot_ir_drop_map(G, voltages, vdd=1.0, ...)   # vdd is SCALAR, not the pad list
```

Common headless usage: pass `show=False`, `save_path='out.png'`. `tests/conftest.py` already forces `matplotlib.use('Agg')`.

## `DynamicPlotter`

```python
from visualization.dynamic_plotter import DynamicPlotter

DynamicPlotter.plot_peak_ir_drop_heatmap(model, result, layer='M1',
    title='Peak IR-Drop During Transient', save_path='peak_ir_drop.png')

DynamicPlotter.plot_peak_current_heatmap(model, result, layer='M1',
    save_path='peak_current.png')

DynamicPlotter.plot_time_series(result,
    metrics=['max_ir_drop', 'total_current', 'vsrc_current'],
    save_path='time_series.png')

DynamicPlotter.plot_node_waveforms(result, nodes=[...], save_path='waveforms.png')
```

Works on both `QuasiStaticResult` and `TransientResult` (both expose `peak_*_per_node`, `t_array`, `total_current_per_time`, `tracked_waveforms`, etc.).

## `PDNPlotter` advanced features

| Feature | What it does |
|---|---|
| Net-type detection | Auto-classifies power (`VDD`/`VCC`/`VDDA`/…) vs ground (`VSS`/`GND`) from naming |
| Layer orientation | Detects `'H'`, `'V'`, `'MIXED'` from resistor edge angles (±15° tolerance) |
| Anisotropic binning | Bins thin perpendicular to routing, wide along it |
| Stripe consolidation | Merges adjacent stripes when count exceeds `max_stripes` |
| Worst-node selection | Finds spatially-separated worst nodes (10% min separation) |

```python
plotter = PDNPlotter(graph, graph.graph.get('net_connectivity', {}))
plotter.generate_layer_heatmaps(
    net_name='VDD',
    output_path='./results',
    anisotropic_bins=True,
    bin_aspect_ratio=50,
    show_irdrop=True,
)
```

## `stripe_heatmap.py` — shared kernel

These helpers are reused by the distributed heatmap pipeline. Do not reimplement them locally:

| Function | Role |
|---|---|
| `parse_node_info('1000_2000_M1') -> (1000.0, 2000.0, 'M1')` | canonical node-name parser for plotting |
| `extract_node_data_vectorized(...)` | bulk per-node coord/value extraction |
| `detect_orientation_from_edges(...)`, `detect_orientation_from_coords(...)` | layer orientation detection |
| `group_nodes_into_stripes(...)`, `consolidate_stripes(...)` | stripe construction |
| `aggregate_stripe_bins(...)`, `aggregate_fast_imshow(...)` | per-stripe aggregation |
| `plot_stripe_heatmap(...)` | one-shot stripe plotter (in-process) |
| `render_from_prebinned_stripe_data(...)` | renderer used by `distributed/heatmap.py` after merge |

## Colormap conventions

| Mode | Colormap | Aggregation | Units |
|---|---|---|---|
| IR-drop (power net) | `RdYlGn_r` | max per bin | mV |
| Ground bounce (ground net) | `RdYlGn_r` | max per bin | mV |
| Voltage (power net) | `RdYlGn` | min per bin | V |
| Current | `hot_r` | sum per bin | mA |

Stick to these so flat and distributed renders look consistent.

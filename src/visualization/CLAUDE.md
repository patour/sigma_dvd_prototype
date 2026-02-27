# Visualization Package

Plotting utilities for IR-drop analysis results.

## Key Classes

- **UnifiedPlotter** (`unified_plotter.py`): Voltage/IR-drop heatmap generation for static results
- **DynamicPlotter** (`dynamic_plotter.py`): Heatmap and time series plotting for dynamic/transient results (see `src/analysis/CLAUDE.md` for usage)
- **PDNPlotter** (`pdn_plotter.py`): Layer-wise heatmap generation with advanced features
- **stripe_heatmap.py**: Stripe-based heatmap visualization

## Headless Plotting

Use `show=False` for batch/headless runs. Matplotlib backend is set to `Agg` in test runners.

## PDNPlotter Advanced Features

| Feature | Description |
|---------|-------------|
| Net Type Detection | Auto-detects power vs ground from naming |
| Layer Orientation | Auto-detects 'H'/'V'/'MIXED' from resistor edge angles |
| Anisotropic Binning | Orientation-aware bins: thin perpendicular to routing |
| Stripe Consolidation | Merges adjacent stripes when count exceeds threshold |
| Worst Node Selection | Finds spatially-separated worst-case nodes |

## Colormap Conventions

| Mode | Colormap | Aggregation | Units |
|------|----------|-------------|-------|
| IR-Drop (power) | `RdYlGn_r` | Max per bin | mV |
| Ground-Bounce (VSS) | `RdYlGn_r` | Max per bin | mV |
| Voltage (power) | `RdYlGn` | Min per bin | V |
| Current | `hot_r` | Sum per bin | mA |

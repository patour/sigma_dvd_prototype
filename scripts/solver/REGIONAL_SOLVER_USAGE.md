# Regional IR-Drop Solver Usage Guide

## Overview

`regional_solver.py` is a command-line script for performing regional IR-drop analysis on partitioned power grids. It demonstrates the `RegionalIRDropSolver` which efficiently computes IR-drops at a subset of load nodes within a partitioned region using effective resistance and boundary conditions.

The regional solver decomposes IR-drop into:
- **Near-field contribution**: From loads within the region
- **Far-field contribution**: From loads outside the region

This enables efficient localized analysis without solving the entire grid.

## Quick Start

### Basic Usage (Default Parameters)
```bash
python regional_solver.py
```

This runs with default parameters from `proto.ipynb`:
- 3-layer grid (K=3)
- 16 stripes in layer 0 (N0=16)
- 1000 load insertion attempts
- 3 partitions (P=3), Y-axis partitioning
- 10% of loads activated with Gaussian distribution
- 1.0W total power
- Analyzes 5 nodes in partition 0

### With Visualizations
```bash
python regional_solver.py --plot_partition --plot_irdrop --plot_current
```

Generates three PNG files:
- `regional_solver_partitions.png` - Partition visualization
- `regional_solver_irdrop.png` - IR-drop heatmap with subset S highlighted
- `regional_solver_current.png` - Current map with active loads

### Custom Configuration
```bash
python regional_solver.py \
    --num_partitions 4 \
    --partition_axis auto \
    --total_power 2.0 \
    --load_percent 0.2 \
    --subset_size 10 \
    --plot_partition \
    --plot_irdrop
```

## Key Parameters

### Grid Generation
- `--K` - Number of metal layers (default: 3)
- `--N0` - Number of stripes in layer 0 (default: 16)
- `--I_N` - Load insertion attempts (default: 1000)
- `--N_vsrc` - Number of voltage source pads (default: 8)
- `--max_stripe_res` - Max stripe resistance in Ω (default: 5.0)
- `--max_via_res` - Max via resistance in Ω (default: 0.1)
- `--grid_seed` - Grid generation seed (default: 7)

### Partitioning
- `--num_partitions` - Number of partitions P (default: 3)
- `--partition_axis` - Axis for partitioning: x, y, or auto (default: y)
- `--partition_id` - Which partition to analyze, 0-indexed (default: 0)
- `--partition_seed` - Partitioner seed (default: 42)

### Stimulus Generation
- `--total_power` - Total power in Watts (default: 1.0)
- `--load_percent` - Fraction of loads to activate, 0-1 (default: 0.1)
- `--distribution` - Current distribution: uniform or gaussian (default: gaussian)
- `--gaussian_loc` - Gaussian mean (default: 1.0)
- `--gaussian_scale` - Gaussian std dev (default: 0.2)
- `--stimulus_seed` - Stimulus generation seed (default: 100)

### Regional Analysis
- `--subset_size` - Number of nodes in subset S (default: 5)
- `--subset_seed` - Subset selection seed (default: 4)

### Visualization
- `--plot_grid` - Show grid during generation
- `--plot_partition` - Generate partition visualization
- `--plot_irdrop` - Generate IR-drop map
- `--plot_current` - Generate current map
- `--show_plots` - Display plots interactively (GUI)
- `--min_current` - Minimum current threshold for current map in A (default: 5e-3)

## Output

The script produces:

1. **Console output** showing:
   - Grid statistics
   - Partition details
   - Regional IR-drop results (voltage, total drop, near/far contributions)
   - Validation against full solver
   - Error statistics (mean, max, RMS, relative error)

2. **PNG files** (when visualization flags are set):
   - Partition visualization with colored regions
   - IR-drop heatmap with subset S nodes highlighted in red
   - Current map showing edge currents and load magnitudes

## Examples

### High Power Scenario
```bash
python regional_solver.py --total_power 5.0 --load_percent 0.3 --plot_irdrop
```

### Different Partition Analysis
```bash
python regional_solver.py --partition_id 1 --subset_size 8
```

### Uniform Distribution
```bash
python regional_solver.py --distribution uniform --load_percent 0.15
```

### Large Grid
```bash
python regional_solver.py --K 4 --N0 20 --I_N 2000 --num_partitions 4
```

### Interactive Mode
```bash
python regional_solver.py --plot_partition --plot_irdrop --show_plots
```

## Validation

The script automatically validates regional solver results against the full IR-drop solver. Typical errors are on the order of 1e-14 V (machine precision), confirming mathematical correctness.

Error metrics reported:
- Mean absolute error
- Max absolute error
- RMS error
- Mean and max relative error (%)

## Integration with Notebook

This script implements the exact workflow from `proto.ipynb` (Regional IR-Drop Solver section) with all parameters exposed as command-line arguments. Default values match those used in the notebook for reproducibility.

## Related Files

- `proto.ipynb` - Interactive notebook with regional solver demonstrations
- `example_regional_voltage.py` - Alternative regional solver example
- `irdrop/regional_voltage_solver.py` - Core implementation
- `GRID_PARTITIONING.md` - Partitioning strategy documentation
- `EFFECTIVE_RESISTANCE_SUMMARY.md` - Effective resistance theory

## Help

For complete parameter documentation:
```bash
python regional_solver.py --help
```

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Static IR-drop analysis prototype for multi-layer power grids. Two main subsystems:
1. **irdrop/**: Synthetic power grid generation, IR-drop solving, effective resistance, and partitioning
2. **pdn/**: SPICE-like netlist parsing for real PDN netlists, DC voltage solving

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run all tests
python run_all_tests.py

# Run specific test module
python -m unittest tests.test_irdrop
python -m unittest tests.test_partitioner
python -m unittest tests.test_pdn_parser

# Run single test
python -m unittest tests.test_irdrop.TestIRDrop.test_no_load_currents_all_pad_voltage

# Run examples
python example_ir_drop.py
python example_partitioning.py
python example_effective_resistance.py
```

## Architecture

### Data Flow
`generate_power_grid()` → NetworkX graph with resistor edges → `PowerGridModel` (sparse Laplacian + Schur reduction) → `IRDropSolver` or `EffectiveResistanceCalculator` → results → `plot.py` visualization

### Key Classes (irdrop/)
- **PowerGridModel**: Builds conductance matrix from resistor graph, eliminates pad nodes via Schur complement. Caches LU factorization for batch solves.
- **IRDropSolver**: Single/batch voltage solves. Returns `IRDropResult(voltages, ir_drop, metadata)`.
- **StimulusGenerator**: Allocates power across load nodes (uniform or gaussian distribution).
- **EffectiveResistanceCalculator**: Batch R_eff computation to ground or between node pairs.
- **GridPartitioner**: Structured slab partitioning along via rows/columns.
- **RegionalIRDropSolver**: Hierarchical IR-drop with sub-grid decomposition.

### PDN Module (pdn/)
- **NetlistParser**: Parses SPICE-like PDN netlists (supports gzip, tile-based parallel parsing, subcircuit flattening).
- **PDNSolver**: DC voltage solver for parsed netlists (sparse direct or iterative CG).
- **PDNPlotter**: Layer-wise heatmap visualization.

## Critical Conventions

### NodeID Structure
`NodeID(layer, idx)` frozen dataclass used as graph node keys.

### Current Sign Convention
- **Input**: Positive current = sink drawing from grid (`currents[node] = +1.0`)
- **Internal**: Solver negates for nodal equation `G·V = I`
- **IR-drop**: Always `Vdd - V_node`

### Common Mistakes
- `plot_ir_drop_map(G, voltages, vdd=1.0, ...)` requires scalar `vdd`, NOT pad list
- `StimulusGenerator(graph=G, ...)` must pass graph if using `area=(x0,y0,x1,y1)` parameter
- Pad nodes rejected in pairwise R_eff calculations (raises `ValueError`)

### Headless Plotting
Use `show=False` for batch/headless runs. Matplotlib backend is set to `Agg` in test runners.

## Test Helpers

`build_small()` in tests creates standard test grid: `K=3, N0=8, I_N=80` (3 layers, 8 stripes, 80 loads).

Key invariants tested:
- Zero load → all nodes at pad voltage
- Min voltage decreases monotonically with power
- R_eff symmetry and triangle inequality
- Partition balance ratio constraints

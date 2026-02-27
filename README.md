# Power Grid IR-Drop Analysis Prototype

Static and dynamic IR-drop analysis for multi-layer power grids. Supports both synthetic grids and real PDN netlists. Includes quasi-static (batch DC) and transient RC analysis.

## Installation

```bash
pip install -e .
```

## Source Layout (`src/` packages)

1. **`src/graph/`** - Rustworkx graph wrappers and conversion utilities
2. **`src/model/`** - `UnifiedPowerGridModel`, adapters, factory functions
3. **`src/solver/`** - Flat, hierarchical, coupled, and tiled solvers
4. **`src/analysis/`** - Dynamic, transient, adjoint analysis; PWL smoothing
5. **`src/parser/`** - SPICE-like netlist parsing (`NetlistParser`)
6. **`src/distributed/`** - Distributed DDM solver (tile-based domain decomposition)
7. **`src/visualization/`** - Plotters (`UnifiedPlotter`, `DynamicPlotter`, `PDNPlotter`)
8. **`src/legacy/`** - Original synthetic grid modules

## Usage Examples

### PDN Netlist Analysis (Recommended)

```python
from parser.netlist import NetlistParser
from model.factory import create_model_from_pdn
from solver.unified_solver import UnifiedIRDropSolver

parser = NetlistParser('./netlist/netlist_test', validate=True)
graph = parser.parse()
model = create_model_from_pdn(graph, 'VDD')
load_currents = model.extract_current_sources()

solver = UnifiedIRDropSolver(model)
result = solver.solve(load_currents)
print(f"Max IR-drop: {max(result.ir_drop.values()):.4f} V")
```

### Synthetic Grid Analysis (Legacy)

```python
from legacy.generate_power_grid import generate_power_grid
from legacy import PowerGridModel, StimulusGenerator, IRDropSolver

G, loads, pads = generate_power_grid(K=3, N0=12, I_N=150, N_vsrc=4,
                                     max_stripe_res=1.0, max_via_res=0.1,
                                     load_current=1.0, seed=7, plot=False)

model = PowerGridModel(G, pad_nodes=pads, vdd=1.0)
stim_gen = StimulusGenerator(load_nodes=list(loads.keys()), vdd=1.0, seed=42)

meta = stim_gen.generate(total_power=1.2, percent=0.30, distribution="gaussian")
solver = IRDropSolver(model)
result = solver.solve(meta.currents, metadata={"power": meta.total_power})
print("Min voltage:", min(result.voltages.values()))
```

## Concepts

Equation: G V = I where G is the nodal conductance matrix assembled from resistors (g = 1/R). Pads are fixed at V_DD and eliminated to form the reduced system.

Stimulus currents are positive for *sinks* (nodes drawing current). Internally these are converted to negative injections in the solver to match the sign convention of the nodal equation.

IR-drop per node: `Vdd - V_node`.

## Running Tests

```bash
# All tests (~984 tests)
python run_all_tests.py

# Specific test module
python -m unittest tests.legacy.test_irdrop
python -m unittest tests.parser.test_pdn_parser
python -m unittest tests.solver.test_hierarchical_solver
```

## License

Prototype code for internal evaluation. Adjust licensing as needed.

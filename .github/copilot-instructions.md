# AI Coding Guide – Power Grid Analysis Prototype

Concise reference for agents working on static IR-drop, effective resistance, and spatial partitioning over artificial multi-layer power grids.

## 1. Big Picture Workflow
Generate grid (`generate_power_grid.py`) → Build reduced conductance model (`PowerGridModel`) → Create stimuli (`StimulusGenerator`) → Solve voltages (`IRDropSolver`) or resistances (`EffectiveResistanceCalculator`) → (Optionally) partition grid (`GridPartitioner`) → Visualize (`plot.py`). Reuse LU factorization for batch speed.

## 2. Core Modules & Roles
`generate_power_grid.py`: Constructs layered resistor mesh; NodeID(layer, idx) keys; loads inserted mid‑segment only on layer 0; pads placed on top layer endpoints (ordered, possible truncation if requested count exceeds). Resistances halve per higher layer.
`power_grid_model.py`: Builds sparse G; Schur reduction removing pad (Dirichlet) nodes; caches (G_uu, G_up, lu). Sign flip: input sink current +I becomes −I in RHS.
`solver.py`: Single/batch voltage solves returning `IRDropResult` (voltages, ir_drop, metadata); summary helpers.
`stimulus.py`: Uniform or Gaussian allocation of total_power → currents; `area` rectangle filters candidates before sampling if graph provided.
`effective_resistance.py`: Batch R_eff: to ground uses diagonal of G_uu^{-1}; pairwise uses solved columns of inverse (no direct dense inversion). Rejects pad nodes for pairwise queries.
`grid_partitioner.py`: Structured slab partitioning (deterministic, connectivity enforced) with axis selection and optimized separator placement. Boundaries snap to via rows/columns; endpoint nodes (degree≤1) excluded as separators. X-axis uses layer-0 via nodes; Y-axis preferentially uses layer-1+ via nodes for fewer separators (can have zero layer-0 separators). Axis parameter: 'x' (vertical slabs), 'y' (horizontal slabs), 'auto' (choose better balance). Via-column constraints can cause load imbalance (ratio up to 3.5 for single axis; auto typically achieves <2.0). Pads never partitioned; separators carry no loads.
`plot.py`: Voltage / IR-drop / current maps; headless default (`Agg`). Current map derives I=(V_u−V_v)/R per edge; optional via filtering.

## 3. Critical Conventions
Currents: positive = sink draw; solver internally negates for G·V=I. IR-drop = Vdd − V_node. Loads exist only on layer 0 and never at via (intersection) nodes. Via & stripe resistances halve per ascending layer. Do not pass pad list where a scalar `vdd` is required (common plotting mistake). Batch solves reuse LU and constant term (−G_up·V_p).

## 4. Typical Usage Snippets
Build + solve one stimulus:
```python
G, loads, pads = generate_power_grid(K=3, N0=12, I_N=150, N_vsrc=4, max_stripe_res=1.0, max_via_res=0.1, load_current=1.0, seed=7, plot=False)
model = PowerGridModel(G, pad_nodes=pads, vdd=1.0)
stim_gen = StimulusGenerator(load_nodes=list(loads.keys()), vdd=1.0, seed=42, graph=G)
meta = stim_gen.generate(total_power=1.2, percent=0.30, distribution="gaussian", area=(0.2,0.2,0.8,0.8))
solver = IRDropSolver(model)
result = solver.solve(meta.currents)
```
Batch sweep:
```python
metas = stim_gen.generate_batch([0.5, 1.0, 2.0], percent=0.25)
results = solver.solve_batch([m.currents for m in metas], metas)
```
Effective resistance mix:
```python
calc = EffectiveResistanceCalculator(model)
pairs = [(list(loads.keys())[0], None), (list(loads.keys())[1], list(loads.keys())[2])]
reff = calc.compute_batch(pairs)
```
Partition (structured slabs, 4 regions):
```python
from irdrop.grid_partitioner import GridPartitioner
part = GridPartitioner(G, loads, pads).partition(P=4, axis='auto')  # auto selects best X or Y
```

## 5. Testing & Validation
Run all tests: `python -m unittest discover -s tests -p 'test_*.py'`. Key invariants: zero-load → all pad voltage; monotonic min voltage with increasing power; R_eff symmetry & positivity; partition load balance ratio ≤3.5 (via-column boundary constraint trades balance for separation guarantee); pads excluded from partitions; graph topology unchanged by partitioning; cross-partition paths require separator traversal.

## 6. Common Gotchas / Edge Cases
Passing pad list instead of float `vdd` to `plot_ir_drop_map`; selecting area without providing `graph` to `StimulusGenerator`; requesting more partitions than load nodes; including pad nodes in effective resistance pairwise queries (raises ValueError); extremely small node subsets leading to empty plots (raise early); Gaussian weights can degenerate → fallback uniform.

## 7. Extension Hooks
Add stimulus distribution: implement new branch in `StimulusGenerator.generate` (ensure normalization of total current). New metrics: derive from voltages or R_eff post-solve (keep batch usage to avoid refactor). New partition strategy: follow structured pattern returning `PartitionResult` preserving graph (no edge removals).

## 8. Performance Notes
Avoid rebuilding `PowerGridModel` for each stimulus; reuse `model.reduced.lu`. Batch effective resistance solves iterate LU with unit RHS columns—prefer grouping queries. Partitioning (structured) is O(N log N) over load-node x sorting; kmeans path heavier and less reliable.

## 9. File Landmarks
Examples: `example_ir_drop.py`, `example_partitioning.py`, `example_effective_resistance.py`; Tests: `tests/test_irdrop.py`, `tests/test_partitioner.py`; Core API exports in `irdrop/__init__.py`.

Questions or missing clarity? Ask for: more on separator promotion, adding multi-domain Vdd, or current visualization scaling.

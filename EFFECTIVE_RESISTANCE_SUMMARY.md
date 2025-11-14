# Effective Resistance Module - Summary

## Overview
Created a new module `irdrop/effective_resistance.py` that computes effective resistance between node pairs in the power grid network. The implementation is optimized for batch computation to efficiently handle large numbers of node pairs.

## Module: `irdrop/effective_resistance.py`

### Class: `EffectiveResistanceCalculator`

Computes effective resistance (R_eff) between nodes using the reduced conductance matrix from `PowerGridModel`.

#### Key Features:
- **Batch computation**: Efficiently processes multiple node pairs in a single call
- **Two computation modes**:
  - Node to ground (pads): `(node, None)` → computes R_eff from node to voltage sources
  - Node to node: `(node1, node2)` → computes R_eff between two nodes
- **Mixed batches**: Can combine both types in a single batch operation
- **Efficient sparse linear algebra**: Uses factorized conductance matrix and solves multiple systems simultaneously

#### API:

```python
from irdrop import EffectiveResistanceCalculator

# Initialize with a PowerGridModel
calc = EffectiveResistanceCalculator(model)

# Compute batch of effective resistances
pairs = [
    (node1, None),      # R_eff to ground
    (node2, node3),     # R_eff between nodes
    (node4, None),      # R_eff to ground
]
reff = calc.compute_batch(pairs)  # Returns numpy array

# Single computation (convenience method)
reff_single = calc.compute_single(node1, node2)
```

#### Input Format:
- `pairs`: List or numpy array of tuples `(u, v)` where:
  - `u`: NodeID (source node)
  - `v`: NodeID or None
    - If `None`: computes R_eff from u to ground (voltage sources)
    - If NodeID: computes R_eff between u and v

#### Output:
- Returns numpy array of effective resistances with same length as input
- R_eff[k] corresponds to pairs[k]
- Values in Ohms (Ω)

## Mathematical Background

The effective resistance between nodes u and v is computed using:

```
R_eff(u,v) = (e_u - e_v)^T * G^(-1) * (e_u - e_v)
           = G^(-1)[u,u] + G^(-1)[v,v] - 2*G^(-1)[u,v]
```

For resistance to ground (with pads eliminated via Schur complement):
```
R_eff(u, ground) = (G_uu^(-1))[u,u]
```

The implementation solves sparse linear systems `G_uu * X = e_i` for each unique node, then combines results according to the formula above.

## Implementation Efficiency

**Batch optimization strategies:**
1. Collects unique nodes across all pairs
2. Solves one sparse linear system per unique node
3. Reuses the pre-factorized conductance matrix from PowerGridModel
4. Assembles results by indexing into solution vectors (no redundant solves)

**Performance:** On test grids (~140 nodes), processes ~50 pairs in < 1ms

## Tests

Added comprehensive test suite in `tests/test_irdrop.py`:

### Test Coverage:
- `test_ground_resistance_positive`: Validates R_eff > 0 for non-pad nodes
- `test_pad_to_ground_is_zero`: Verifies pads have zero resistance to ground
- `test_node_to_node_symmetry`: Checks R_eff(u,v) = R_eff(v,u)
- `test_node_to_node_positive`: Validates R_eff > 0 between distinct nodes
- `test_node_to_self_is_zero`: Checks R_eff(u,u) ≈ 0
- `test_triangle_inequality`: Validates R(u,w) ≤ R(u,v) + R(v,w)
- `test_batch_mixed_types`: Tests mixed ground/node-to-node batches
- `test_batch_large`: Efficiency test with large batches
- `test_single_convenience_method`: Validates single-pair API
- `test_empty_batch`: Edge case handling
- `test_invalid_node_raises`: Error handling for invalid inputs
- `test_consistency_with_voltage_solve`: Cross-validation with IR-drop solver

**All 16 tests pass** (12 new + 4 existing IR-drop tests)

## Examples

### Command-line Example
`example_effective_resistance.py` - Comprehensive demonstration with:
- Resistance to ground computation
- Node-to-node resistance
- Mixed batch operations
- Large batch efficiency testing
- Symmetry validation

Run with: `python3 example_effective_resistance.py`

### Notebook Example
Added section to `proto.ipynb` demonstrating:
- Basic setup with EffectiveResistanceCalculator
- Computing R_eff to ground
- Computing R_eff between node pairs
- Mixed batch usage

## Package Integration

Updated `irdrop/__init__.py` to export:
```python
from .effective_resistance import EffectiveResistanceCalculator

__all__ = [
    # ...existing exports...
    "EffectiveResistanceCalculator",
]
```

## Usage Example

```python
from generate_power_grid import generate_power_grid
from irdrop import PowerGridModel, EffectiveResistanceCalculator

# Generate grid
G, loads, pads = generate_power_grid(K=3, N0=8, I_N=50, N_vsrc=3, 
                                     max_stripe_res=1.0, max_via_res=0.1,
                                     plot=False)

# Build model
model = PowerGridModel(G, pad_nodes=pads, vdd=1.0)

# Create calculator
calc = EffectiveResistanceCalculator(model)

# Compute effective resistances
nodes = list(loads.keys())[:10]
pairs = [
    (nodes[0], None),      # to ground
    (nodes[1], nodes[2]),  # node-to-node
    (nodes[3], None),      # to ground
]

reff = calc.compute_batch(pairs)
print(f"R_eff values: {reff}")
```

## Files Created/Modified

### New Files:
- `irdrop/effective_resistance.py` - Main module implementation
- `example_effective_resistance.py` - Standalone example script
- `EFFECTIVE_RESISTANCE_SUMMARY.md` - This document

### Modified Files:
- `irdrop/__init__.py` - Added EffectiveResistanceCalculator export
- `tests/test_irdrop.py` - Added TestEffectiveResistance test class (12 tests)
- `proto.ipynb` - Added demonstration section

## Notes

- Pad nodes cannot be used in node-to-node pairs (raises ValueError) since they are fixed voltage sources
- Pad nodes have zero resistance to ground by definition
- The implementation automatically handles the sign conventions and boundary conditions from PowerGridModel
- Numerical precision is excellent (~1e-10 for symmetry checks)

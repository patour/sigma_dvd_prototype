# Small PDN Test Netlist (VDD_XLV)

A compact PDN netlist generated for faster development, testing, and debugging of the IR-drop solver. Includes isolated islands for testing floating node detection.

## Netlist Statistics

| Metric | Value |
|--------|-------|
| Nodes (total) | 6,574 |
| - Main grid | 6,540 |
| - Isolated islands | 34 |
| Resistors | 16,694 |
| Current Sources | 1,017 |
| - Main grid loads | 1,000 |
| - Island loads | 17 |
| Voltage Sources | 1 |
| Total Load Current | 21.5 A |
| Current Distribution | Gaussian (μ=21.6mA, σ=10.2mA) |
| Max IR-drop | ~10 mV |
| Nominal Vdd | 0.75 V |

## Layer Structure

| Layer | Grid Size | Nodes | Description |
|-------|-----------|-------|-------------|
| M1 | 50×50 | 2,500 | Lowest layer, load injection |
| M2 | 50×50 | 2,500 | Second metal layer |
| M3 | 25×25 | 625 | Third metal layer |
| M4 | 25×25 | 625 | Fourth metal layer |
| M5 | 12×12 | 144 | Top layer, bump connections |
| Package | - | 145 | Tap nodes + vsrc |

## Pitch/Grid Spacing

- M1, M2: 2000 units
- M3, M4: 4000 units  
- M5: 8000 units (bump pitch)

## Isolated Islands

Three small disconnected regions are included to test floating node detection:

| Island | Grid | Nodes | Loads | Location (x) |
|--------|------|-------|-------|--------------|
| island1 | 3×3 M1 | 9 | 5 | x=110000, y=10000-14000 |
| island2 | 3×3 M1 | 9 | 5 | x=110000, y=30000-34000 |
| island3 | 4×4 M1 | 16 | 7 | x=110000, y=50000-56000 |

**Note**: Islands are automatically detected and filtered by `UnifiedPowerGridModel` during model creation. A warning is logged: "Removed 3 floating island(s) with 34 nodes".

## Performance Comparison

| Metric | Original VDD_XLV | Small Netlist | Speedup |
|--------|------------------|---------------|---------|
| Nodes | ~500K | 6.5K | 77x smaller |
| Flat Solve | 1.57s | 0.007s | 224x faster |

## Usage

```python
from pdn.pdn_parser import NetlistParser
from core import create_model_from_pdn, UnifiedIRDropSolver

# Parse
parser = NetlistParser('./pdn/netlist_small', validate=True)
graph = parser.parse()

# Create model (islands are auto-filtered with warning)
model = create_model_from_pdn(graph, 'VDD_XLV')

# Get loads (only main grid loads, island loads filtered)
load_currents = model.extract_current_sources()
print(f"Active loads: {len(load_currents)}")  # 1000 (not 1017)

# Solve
solver = UnifiedIRDropSolver(model)
result = solver.solve(load_currents)
print(f"Max IR-drop: {max(result.ir_drop.values())*1000:.2f} mV")  # ~10 mV
```

## Files

- `ckt.sp` - Main netlist file with includes and parameters
- `tile_0_0.ckt` - Die mesh (5-layer resistor network + 3 isolated islands)
- `tile_0_0.nd` - Node-to-net mapping for all die nodes (including islands)
- `package.ckt` - Package model with voltage source and bump connections
- `instanceModels_0_0.sp` - Current source definitions (1017 loads, Gaussian distribution)
- `pg_net_voltage` - Net voltage specifications
- `additional_vsrcs` - Additional voltage sources (empty)
- `decap_cell_list` - Decap cell list (empty)
- `switch_cell_list` - Power switch list (empty)

## Notes

- Generated with seed=42 for reproducible load placement
- Loads distributed randomly across 10% of M1 nodes
- Single voltage source at `VDD_XLV_vsrc` node
- All die nodes connected to `VDD_XLV` net

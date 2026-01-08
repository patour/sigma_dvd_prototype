# PDN Test Netlist

Minimal test netlist for validating PDN parser, solver, and plotter functionality.

## Structure

- **Topology**: Single 1×1 tile with 5×5 grid on two metal layers (M1, M2)
- **Layers**: 
  - M1: Horizontal routing (5 rows)
  - M2: Vertical routing (5 columns)
  - Via connections at all 25 grid intersections
- **Total nodes**: 50 die nodes + package nodes + ground
- **Power distribution**: VDD rail at 0.75V with 9 bump connections
- **Load**: 17 current sources totaling ~38 mA, clustered to create IR-drop gradients

## Files

| File | Purpose | Format |
|------|---------|--------|
| `ckt.sp` | Main netlist entry point | Text SPICE |
| `tile_0_0.ckt` | Die resistor/capacitor mesh | Text SPICE |
| `tile_0_0.nd` | Node-to-net mapping (all VDD) | Text (6-column) |
| `package.ckt` | VRM probe network + bump connections | Text SPICE |
| `instanceModels_0_0.sp` | Current source load distribution | Text SPICE |
| `pg_net_voltage` | Net voltage specifications | Text |
| `additional_vsrcs` | Extra voltage sources (empty) | Text |
| `decap_cell_list` | Decap definitions (empty) | Text |
| `switch_cell_list` | Power switch definitions (empty) | Text |

## Usage

### Parse Only
```bash
cd pdn
python pdn_parser.py --netlist-dir ./netlist_test --validate --verbose
```

### Parse and Solve
```bash
python pdn_solver.py --netlist-dir ./netlist_test --output ./test_results --verbose
```

### Parse, Solve, and Plot
```bash
python pdn_solver.py --netlist-dir ./netlist_test --output ./test_results \
    --plot-bin-size 500 --verbose
```

### Plot All Layers
```bash
python pdn_solver.py --netlist-dir ./netlist_test --output ./test_results \
    --verbose --no-anisotropic-bins
```

### Stripe Mode Plotting
```bash
python pdn_solver.py --netlist-dir ./netlist_test --output ./test_results \
    --verbose --stripe-mode --max-stripes 10
```

## Expected Results

### Parser Output
- **Nodes**: ~60 (50 die + package nodes)
- **Elements**: ~200 (resistors, capacitors, via connections)
- **Layers**: M1, M2, M1-M2 (inter-layer), package
- **Current sources**: 17 sources, ~38 mA total
- **Voltage sources**: 2 (VDD, VSS)

### Solver Output
- **Min voltage**: ~0.73-0.74V (at high-load cluster near 2000_2000)
- **Max IR-drop**: ~10-20 mV (depends on resistor values)
- **Avg voltage**: ~0.745V

### Plotter Output
- **Voltage heatmaps**: Show gradient from bumps (0.75V) to load centers
- **Current heatmaps**: Show load clustering in bottom-left and center regions
- **Layer orientation**: M1 horizontal stripes, M2 vertical stripes

## Validation Checklist

- [ ] Parser runs without errors
- [ ] All 50 die nodes mapped to VDD net
- [ ] Union-find propagates VDD net type to package nodes
- [ ] No floating nodes detected
- [ ] 9 voltage source nodes identified (bumps + VRM)
- [ ] Solver converges with realistic voltages (0.73-0.75V range)
- [ ] IR-drop shows spatial gradient (higher drop near load clusters)
- [ ] Heatmaps display with proper orientation (M1 horizontal, M2 vertical)
- [ ] Anisotropic binning adapts to layer orientation
- [ ] Stripe mode produces consolidated stripe plots

## Design Rationale

### Resistor Values
- **M1 stripe resistance**: 0.020 Ω (20 mΩ per segment)
- **M2 stripe resistance**: 0.015 Ω (15 mΩ, lower for vertical routing)
- **Via resistance**: 0.005 Ω (5 mΩ per via)
- **Package distribution**: 0.002 Ω (2 mΩ)
- Chosen to produce measurable IR-drop (~10-20 mV) without extreme values

### Current Distribution
- **Heavy cluster** (2000_2000): 20 mA concentrated load
- **Medium cluster** (3000_3000): 9 mA moderate load
- **Light periphery**: 9 mA distributed I/O load
- Creates realistic spatial IR-drop gradient for validation

### Grid Size
- 5×5 grid chosen for:
  - Fast parsing (<1 second)
  - Sufficient complexity to test layer connectivity
  - Small enough for manual verification
  - Large enough to show spatial voltage gradients

### Bump Placement
- 4 corner bumps (primary supply)
- 4 edge center bumps (secondary supply)
- 1 center bump (tertiary supply)
- Strategy: Surround load clusters to test solver accuracy

## Modifications for Extended Testing

### Add Third Layer
Add M3 vertical routing to `tile_0_0.ckt` and update `.nd` file.

### Increase Grid Density
Change coordinates from 1000-step to 500-step (10×10 grid).

### Add Second Net (VSS)
Map some nodes to VSS in `.nd` file and add VSS bump connections.

### Multi-Tile Test
Change `.partition_info 1 1` to `2 2` and create tiles for 2×2 grid.

### Add Gzip Compression
```bash
gzip tile_0_0.ckt
gzip tile_0_0.nd
gzip instanceModels_0_0.sp
```
Parser auto-detects gzipped files via magic number.

## Test Results

Successfully validates:
- ✅ Parser: Loads 56 nodes, 118 elements
- ✅ Solver: Computes voltage drop (~1mV max)
- ✅ Plotter: Generates voltage and current heatmaps
- ✅ Reports: Top-10 worst IR-drop analysis

Expected output in `results/`:
- `voltage_heatmap_VDD.png` - 2 layer voltage maps
- `current_heatmap_VDD.png` - Current source distribution
- `topk_irdrop_VDD.txt` - Top-10 worst nodes report


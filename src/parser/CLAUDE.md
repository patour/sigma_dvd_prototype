# Parser Package

SPICE-like tile-based netlist parsing. Entry point: `NetlistParser` in `netlist.py`.

## Key Classes

- **NetlistParser** (`netlist.py`): Main parser — gzip support, parallel parsing, net filtering
- **pdn_parser.py**: CLI entry point for PDN netlist parsing
- **sampled_netlist.py**: Sampled multi-tile netlist generator from netlist_minion
- **parallel.py**: Worker functions and data classes for parallel tile parsing
- **edge_attrs.py**: Memory-optimized edge attribute classes (ResistorEdge, CapacitorEdge, etc.)
- **spice_lexer.py**: SPICE element line tokenizer
- **graph_builder.py**: Builds rustworkx graph from tokens
- **metadata.py**: Net voltage, vsrc metadata extraction
- **current_sources.py**: CurrentSource, Pulse, PWL data structures

## Current Source Data Structures (from instanceModels*.sp)

- `InstanceInfo`: Parsed instance name with net/pin/tile location info
- `Pulse`: Pulse waveform with `evaluate(time)` and `get_dc()` methods
- `PWL`: Piece-wise linear waveform with `evaluate(time)` and `get_dc()` methods
- `CurrentSource`: Full current source with DC value, static_value, pulses, PWLs

**Accessing Current Source Data:**
```python
# By default, parser stores raw CurrentSource objects (memory efficient)
graph = parser.parse()
raw_sources = graph.graph.get('_instance_sources_objects', {})

# Access CurrentSource objects directly
for name, src in raw_sources.items():
    static_ma = src.get_static_current()      # DC analysis
    current_at_t = src.get_current_at_time(1e-9)  # Transient at 1ns

# For portable pickle files, use store_instance_sources=True (serializes to dicts)
parser = NetlistParser('./netlist_dir', store_instance_sources=True)
graph = parser.parse()
instance_sources = graph.graph.get('instance_sources', {})  # Serialized dicts
```

**Memory Optimization for Large Netlists:**
The default `store_instance_sources=False` avoids serializing CurrentSource objects to dicts, saving ~60% parse-time memory for large netlists (1.7GB -> 1.1GB for 1M sources). The dynamic/transient solvers automatically handle both formats.

## Edge Attribute Memory Optimization

By default, edge attributes use specialized slotted dataclasses (`edge_attrs.py`) instead of dicts, reducing memory by ~95% per edge. Critical for 100M+ edge netlists (~65 GB -> ~4 GB).

```python
from parser.graph_builder import get_use_optimized_edges, set_use_optimized_edges

# Check current mode (default: True)
print(get_use_optimized_edges())  # True

# Disable for backward compatibility or small netlists
set_use_optimized_edges(False)
```

**Edge Classes:**
- `ResistorEdge`: Die resistors (no elem_name stored) — 99.9% of resistors
- `ResistorEdgeWithName`: Package resistors matching `vsrc_resistor_pattern` (e.g., 'rs')
- `CapacitorEdge`, `InductorEdge`, `CurrentSourceEdge`, `VoltageSourceEdge`

**Important:** With optimized edges, `elem_name` is only stored for:
- Voltage sources (always)
- Resistors matching `vsrc_resistor_pattern` (default 'rs') for vsrc node identification

Use `.get('elem_name', '')` instead of `['elem_name']` for safe access:
```python
for u, v, data in graph.edges(data=True):
    elem_name = data.get('elem_name', '')  # Safe: returns '' if not stored
    # NOT: data['elem_name']  # May raise KeyError for die resistors
```

**Runtime Trade-off:** Computed properties (`.tile_id`, `.net_type`) are ~4-5x slower than dict access due to on-the-fly unpacking. For hot loops, cache values locally or use `set_use_optimized_edges(False)`.

**Pickle Compatibility:**
- `store_instance_sources=False` (default): Pickle works but requires `parser` module when loading
- `store_instance_sources=True`: Portable pickle (no module dependency), better for long-term storage

## Parallel Parsing (for large netlists with many tiles)

```python
# Enable parallel parsing for ~6-8x speedup on 100+ tiles
parser = NetlistParser('./netlist_dir', parallel=True, n_workers=8)
graph = parser.parse()

# With net filter and custom chunk size
parser = NetlistParser('./netlist_dir', parallel=True, n_workers=4,
                       net_filter='VDD', chunk_size=10000)
```

Parallel parsing uses `multiprocessing.Pool` with:
- Memory-mapped file access for plain text files (gzip fallback for compressed)
- Chunk-based processing within large tiles
- Bulk graph operations for efficient merge phase
- Full equivalence with sequential parsing (same graph output)

## PDN Netlist Format

Directory structure:
```
netlist_dir/
  ckt.sp              # Top-level circuit includes
  tile_0_0.ckt        # Tile subcircuit with R/C/L/I/V elements
  tile_0_0.nd         # Node coordinate mapping (x y layer node_name)
  package.ckt         # Package-level connections
  instanceModels_0_0.sp  # Instance current source models
  pg_net_voltage      # Power net voltage specs (VDD 1.0, VSS 0.0)
  additional_vsrcs    # Extra voltage source definitions
  decap_cell_list     # Decap cell instance names
  switch_cell_list    # Power switch cell names
```

**Element syntax in `.ckt` files:**
```spice
R_name node1 node2 <resistance_kOhm>
C_name node1 node2 <capacitance_fF>
L_name node1 node2 <inductance_nH>
I_name node1 node2 <current_mA>
V_name node+ node- <voltage_V>
X_inst subckt node1 node2 ...
```

**Current source syntax in `instanceModels*.sp` (enhanced):**
```spice
I_name node+ node- <dc_mA> [static_value=<mA>] [pulse(v1,v2,delay,rt,ft,width,period)] [pwl(t1 v1 t2 v2 ...)]
```
- `static_value=`: Additional static current component
- `pulse(...)`: Periodic pulse waveform (values in Amperes)
- `pwl(...)`: Piece-wise linear waveform with optional `pwl_period=` and `pwl_delay=`

**Node naming convention:** `<x>_<y>_<layer>` (e.g., `1000_2000_M1`)

**Boundary nodes (multi-tile stitching):**
Nodes shared across tile boundaries are marked with `*` prefix in `.ckt` files:
```spice
R_bnd_M1 *900_2000_M1 *1000_2000_M1 8    # Cross-tile resistor (both nodes starred)
r 800_2000_M1 *900_2000_M1 8              # Internal-to-boundary resistor (one starred)
```

The `*` prefix signals the parser to track these nodes for tile stitching:
- Parser strips the `*` prefix when creating graph nodes
- Tracks `boundary_node1`/`boundary_node2` flags in edge attributes
- Merges matching boundary nodes across tiles during graph construction

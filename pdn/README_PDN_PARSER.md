# PDN Netlist Parser

A Python tool to parse Power Delivery Network (PDN) netlists in SPICE-like format and convert them to NetworkX graphs for analysis.

## Features

- **Automatic gzip detection**: Handles compressed files with or without `.gz` extension (checks magic bytes `0x1f8b`)
- **Tile-based parsing**: Supports partitioned netlists with progress tracking
- **3D layer support**: Extracts and tracks layer information from node names (X_Y_LAYER format)
- **Subcircuit flattening**: Expands hierarchical subcircuits with automatic naming
- **Instance-to-node mapping**: Tracks which nodes each current source connects to
- **Validation**: Optional sanity checks for shorts, floating nodes, and merged nodes
- **Package model support**: Distinguishes package nodes from die nodes
- **Voltage source node identification**: Automatically identifies nodes connected to voltage sources via zero-resistance paths
- **Layer-based statistics**: Computes per-layer breakdowns of nodes and elements
- **Visualization**: Grid-based heatmap plotting for individual layers or all layers
- **FSDB metadata extraction**: Captures waveform file references for current sources
- **Error recovery**: Continues parsing after errors (with warnings) unless strict mode enabled

## Installation

### Requirements

```bash
# Required
pip install networkx

# Optional (for progress bars)
pip install tqdm

# Optional (for visualization)
pip install matplotlib numpy

# Optional (for memory profiling)
pip install memory_profiler
```

### No Installation Required

The parser is a standalone Python script. Just download `pdn_parser.py` and run it:

```bash
python pdn_parser.py --help
```

## Usage

### Basic Usage

```bash
# Parse netlist from current directory (looks for ckt.sp)
python pdn_parser.py

# Specify netlist directory
python pdn_parser.py --netlist-dir /path/to/netlist

# Save output graph
python pdn_parser.py --netlist-dir /path/to/netlist --output pdn.graphml
```

### With Validation

```bash
# Enable sanity checks
python pdn_parser.py --netlist-dir /path/to/netlist --validate

# Strict mode (fail on warnings)
python pdn_parser.py --netlist-dir /path/to/netlist --validate --strict
```

### Advanced Options

```bash
# Filter specific power net
python pdn_parser.py --netlist-dir /path/to/netlist --net vdd

# Verbose output
python pdn_parser.py --netlist-dir /path/to/netlist --verbose

# Memory profiling
python pdn_parser.py --netlist-dir /path/to/netlist --profile-memory

# Configure voltage source identification
python pdn_parser.py --netlist-dir /path/to/netlist --vsrc-resistor-pattern rs --vsrc-depth-limit 3
```

### Visualization Options

```bash
# Plot a specific layer
python pdn_parser.py --netlist-dir /path/to/netlist --plot-layer 5 --plot-output layer5.png

# Plot all layers
python pdn_parser.py --netlist-dir /path/to/netlist --plot-all-layers --plot-output layers/

# Plot with custom bin size and statistic
python pdn_parser.py --netlist-dir /path/to/netlist --plot-layer M1 \
    --plot-bin-size 1000 --plot-statistic total_capacitance

# Plot specific net on a layer
python pdn_parser.py --netlist-dir /path/to/netlist --plot-layer 3 --net vdd
```

### Example with MED CrystalX Testcase

```bash
# Parse the example testcase
python pdn_parser.py \
    --netlist-dir ./netlist_data \
    --validate \
    --output crystalx_pdn.pkl \
    --verbose
```

## Netlist Format

The parser supports SPICE-like netlists with PDN-specific extensions:

### Top-Level File (`ckt.sp`)

```spice
* PDN Netlist
.die_area 0 0 10000000 10000000
.partition_info 10 10
.include tile_0_0.ckt
.include tile_0_1.ckt
...
.include instanceModels_0_0.sp
.include package.ckt
.parameter vdd=1.8
```

### Tile Files (`tile_X_Y.ckt`)

```spice
* Tile 0 0
r_grid_1 1000_2000 1010_2000 0.001
c_grid_1 1000_2000 0 1e-12
*r_boundary *3000_2000 1100_2000 0.002
```

Note: Nodes prefixed with `*` are boundary nodes requiring stitching across tiles.

### Instance Model Files (`instanceModels_X_Y.sp`)

```spice
* Current sources
i_cpu:inst1:vdd:0 1000_2000 0 PWL(0 0 1n 5e-3 2n 2e-3)
i_cpu:inst2:vdd:0 1050_2000 0 fsdb /path/to/waveform.fsdb 1.0 0.0
```

### Supported Elements

| Element | Syntax | Example |
|---------|--------|---------|
| Resistor | `R<name> <n1> <n2> <value>` | `R1 vdd gnd 1e-3` |
| Capacitor | `C<name> <n1> <n2> <value> [model]` | `C1 vdd 0 10e-12` |
| Inductor | `L<name> <n1> <n2> <value>` | `L1 pkg die 1e-9` |
| Mutual Inductor | `K<name> L1 L2 <coupling>` | `K1 L1 L2 0.9` |
| Voltage Source | `V<name> <n+> <n-> <dc> [AC ...]` | `Vvdd vdd 0 1.8` |
| Current Source | `I<name> <n+> <n-> <dc> [PWL ...] [fsdb ...]` | `I1 vdd 0 1e-3` |
| VCVS | `E<name> <out+> <out-> <in+> <in-> <gain>` | `E1 out1 out2 in1 in2 2.0` |
| VCCS | `G<name> <out+> <out-> <in+> <in-> <gm>` | `G1 out1 out2 in1 in2 0.01` |
| CCCS | `F<name> <out+> <out-> Vprobe <gain>` | `F1 out1 out2 Vx 5.0` |
| CCVS | `H<name> <out+> <out-> Vprobe <gain>` | `H1 out1 out2 Vx 100` |
| Subcircuit | `.subckt <name> <pins...>` ... `.ends` | See below |
| Subcircuit Instance | `X<name> <nodes...> <subckt>` | `X1 n1 n2 decap` |

### Dot Commands

- `.partition_info N M` - Define N×M tile grid
- `.include <file>` - Include another file
- `.parameter <name>=<value>` - Define parameter
- `.subckt <name> <pins...>` - Start subcircuit definition
- `.ends` - End subcircuit definition
- `.flag_boundary` - Mark following nodes as boundary nodes

## Layer Support

The parser supports 3D node naming to track layer information in multi-layer PDN structures.

### Node Naming Conventions

**3D Format (with layer):**
```
X_Y_LAYER
```
Examples:
- `1000_2000_5` - Node at (1000, 2000) on layer 5
- `500_750_M1` - Node at (500, 750) on metal layer M1
- `2000_3000_AP` - Node at (2000, 3000) on redistribution layer AP

**2D Format (backward compatible):**
```
X_Y
```
Examples:
- `1000_2000` - Node at (1000, 2000), no layer specified
- `vdd_500_750` - Named node with coordinates

**Layer Identifier:**
- Stored as **string** (supports both numeric like `'5'` and named layers like `'M1'`, `'AP'`)
- Extracted as third underscore-separated component
- Nodes without layer identifier have `layer=None`

### Voltage Source Node Identification

The parser automatically identifies nodes that are electrically equivalent to voltage sources by tracing zero-resistance connections from package resistors.

**Identification Method:**
1. Find all nodes directly connected to voltage source elements (V)
2. Identify zero-valued resistors with configurable name pattern (default: `'rs'`)
3. Propagate voltage source marking through zero-resistance paths up to configurable depth (default: 3 hops)

**Configuration:**
```bash
# Change resistor pattern (default: 'rs' exact match)
--vsrc-resistor-pattern rs_pkg

# Change depth limit (default: 3)
--vsrc-depth-limit 5
```

**Example Package Model:**
```spice
* package.ckt
Vvdd vdd_pkg 0 1.8
rs vdd_pkg vdd_die 0.0    # Zero-valued resistor connects to die
```

Result: Both `vdd_pkg` and `vdd_die` nodes marked with `is_vsrc_node=True`

### Layer Statistics

The parser automatically computes per-layer statistics during parsing.

**Statistics Computed:**
- Node count per layer
- Voltage source node count per layer  
- Element counts per layer (resistors, capacitors, inductors, current sources)

**Access via API:**
```python
layer_stats = graph.graph['layer_stats']
# Returns: {layer_id: {'nodes': N, 'vsrc_nodes': V, 'resistors': R, ...}}
```

**CLI Output:**
```
Layer Statistics (5 layers)
============================================================
Layer           Nodes  Vsrc    Res      Cap    Ind  Isrc
------------------------------------------------------------
1              12450     8    45320   23100     0   156
2              13200    12    48900   25400     0   189
3              11800     6    43200   21900     0   142
M1              8500     4    31000   15200     0    98
AP              3200    20    11500    5800    45    45
```

## Python API

### Basic Parsing

```python
from pdn_parser import NetlistParser
import networkx as nx

# Parse netlist
parser = NetlistParser('/path/to/netlist/dir', validate=True)
graph = parser.parse()

# Graph is a NetworkX MultiDiGraph
print(f"Nodes: {graph.number_of_nodes()}")
print(f"Edges: {graph.number_of_edges()}")
```

### Querying the Graph

```python
# Find all resistors
resistors = [(u, v, d) for u, v, d in graph.edges(data=True) 
             if d['type'] == 'R']

# Find high-value resistors (potential opens)
opens = [(u, v, d) for u, v, d in resistors 
         if d['value'] > 1.0]  # > 1 KOhm

# Find all capacitors connected to a node
node = 'vdd_1000_2000'
caps = [(u, v, d) for u, v, d in graph.edges(data=True) 
        if d['type'] == 'C' and (u == node or v == node)]

# Total capacitance on a node
total_cap = sum(d['value'] for u, v, d in caps)  # in fF

# Find current sources with FSDB waveforms
isrcs_with_fsdb = [(u, v, d) for u, v, d in graph.edges(data=True)
                   if d['type'] == 'I' and 'fsdb_path' in d]
```

### Instance-to-Node Mapping

```python
# Get mapping dictionary
inst_map = graph.graph['instance_node_map']

# Find nodes for specific instance
instance_name = 'i_cpu_core:inst1:vdd:0'
nodes = inst_map.get(instance_name, [])
print(f"Instance {instance_name} connects to nodes: {nodes}")

# Find all instances connected to a node
node = 'vdd_1000_2000'
instances = [inst for inst, nodes in inst_map.items() if node in nodes]
```

### Filtering Nodes

```python
# Get package nodes only
pkg_nodes = [n for n, d in graph.nodes(data=True) 
             if d.get('is_package', False)]

# Get boundary nodes
boundary_nodes = [n for n, d in graph.nodes(data=True) 
                  if d.get('is_boundary', False)]

# Get voltage source nodes
vsrc_nodes = [n for n, d in graph.nodes(data=True)
              if d.get('is_vsrc_node', False)]

# Or get from metadata
vsrc_nodes_set = graph.graph.get('vsrc_nodes', set())

# Get nodes from specific layer
layer_5_nodes = [n for n, d in graph.nodes(data=True) 
                 if d.get('layer') == '5']

# Get nodes from multiple layers
metal_layers = ['M1', 'M2', 'M3']
metal_nodes = [n for n, d in graph.nodes(data=True)
               if d.get('layer') in metal_layers]

# Get nodes from specific tile
tile_nodes = [n for n, d in graph.nodes(data=True) 
              if d.get('tile_id') == (0, 0)]

# Get nodes with coordinates in range on specific layer
nodes_in_region = [n for n, d in graph.nodes(data=True)
                   if d.get('layer') == '3' and
                   d.get('x') and d.get('y') and
                   1000 <= d['x'] <= 2000 and
                   1000 <= d['y'] <= 2000]

# Use filter_by_layer helper function
from pdn_parser import filter_by_layer
layer_3_graph = filter_by_layer(graph, '3')
layer_3_vdd = filter_by_layer(graph, '3', net='vdd')
```

### Path Analysis

```python
import networkx as nx

# Find resistive path between two nodes
path = nx.shortest_path(graph, 'vrm_node', 'cell_node')
print(f"Path: {' -> '.join(path)}")

# Calculate path resistance
path_resistance = 0
for i in range(len(path) - 1):
    u, v = path[i], path[i+1]
    # Get all edges between u and v
    for key, data in graph[u][v].items():
        if data['type'] == 'R':
            path_resistance += data['value']
print(f"Total resistance: {path_resistance:.2e} KOhm")

# Find all paths (up to certain length)
all_paths = nx.all_simple_paths(graph, 'vrm', 'node', cutoff=10)

# Check connectivity
is_connected = nx.has_path(graph.to_undirected(), 'vrm', 'node')
```

### Graph Metadata

```python
# Access metadata
metadata = graph.graph

# Voltage sources
vsrc_dict = metadata['vsrc_dict']
for name, attrs in vsrc_dict.items():
    print(f"Vsource {name}: DC={attrs['dc']}V")

# Voltage source nodes (identified via zero-resistance paths)
vsrc_nodes = metadata.get('vsrc_nodes', set())
print(f"Found {len(vsrc_nodes)} voltage source nodes")

# Parameters
parameters = metadata['parameters']
vdd = float(parameters.get('vdd', 1.8))

# Tile grid
tile_grid = metadata['tile_grid']  # (N, M) or None
if tile_grid:
    print(f"Tile grid: {tile_grid[0]} × {tile_grid[1]}")

# Merged nodes (from validation)
merged_nodes = metadata['merged_nodes']
for from_node, to_node, merge_type in merged_nodes:
    print(f"Merged: {from_node} -> {to_node}")

# Mutual inductors
mutual_inductors = metadata['mutual_inductors']
for k_name, (l1, l2, coupling) in mutual_inductors.items():
    print(f"{k_name}: {l1} <-> {l2}, k={coupling}")

# Statistics
stats = metadata['stats']
print(f"Resistors: {stats['resistors']}")
print(f"Capacitors: {stats['capacitors']}")
print(f"Current sources: {stats['isources']}")
print(f"Voltage source nodes: {stats['vsrc_nodes']}")

# Layer statistics
layer_stats = metadata.get('layer_stats', {})
for layer_id, layer_stat in layer_stats.items():
    if layer_id is not None:  # Skip 2D-only nodes
        print(f"Layer {layer_id}:")
        print(f"  Nodes: {layer_stat['nodes']}")
        print(f"  Vsrc nodes: {layer_stat['vsrc_nodes']}")
        print(f"  Resistors: {layer_stat['resistors']}")
        print(f"  Capacitors: {layer_stat['capacitors']}")

# Find layer with most nodes
if layer_stats:
    layers_with_counts = [(k, v['nodes']) for k, v in layer_stats.items() if k is not None]
    if layers_with_counts:
        max_layer = max(layers_with_counts, key=lambda x: x[1])
        print(f"Layer with most nodes: {max_layer[0]} ({max_layer[1]} nodes)")

# Total capacitance per layer
for layer_id, layer_stat in layer_stats.items():
    if layer_id is not None:
        total_cap = layer_stat['capacitors']  # Count of capacitors
        print(f"Layer {layer_id}: {total_cap} capacitors")
```

### Export and Save

```python
# Save as GraphML (XML format, portable)
nx.write_graphml(graph, 'pdn.graphml')

# Load GraphML
graph = nx.read_graphml('pdn.graphml')

# Save as pickle (faster, Python-specific)
import pickle
with open('pdn.pkl', 'wb') as f:
    pickle.dump(graph, f)

# Load pickle
with open('pdn.pkl', 'rb') as f:
    graph = pickle.load(f)

# Export to other formats
nx.write_gexf(graph, 'pdn.gexf')  # Gephi format
nx.write_graphml(graph, 'pdn.graphml')  # GraphML

# Export node list to CSV
import pandas as pd
nodes_df = pd.DataFrame([
    {'name': n, **d} for n, d in graph.nodes(data=True)
])
nodes_df.to_csv('nodes.csv', index=False)

# Export edge list to CSV
edges_df = pd.DataFrame([
    {'from': u, 'to': v, 'key': k, **d}
    for u, v, k, d in graph.edges(keys=True, data=True)
])
edges_df.to_csv('edges.csv', index=False)
```

## Visualization

The parser includes grid-based visualization utilities for analyzing PDN layers. For large networks with millions of nodes, the visualization uses spatial binning to aggregate statistics into manageable heatmaps.

### Plot Single Layer

```python
from pdn_parser import plot_layer
import networkx as nx

# Load parsed graph
graph = nx.read_graphml('pdn.graphml')

# Plot layer with default settings (node count)
plot_layer(graph, layer_id='5', output_file='layer5.png')

# Plot specific net on a layer
plot_layer(graph, layer_id='M1', net='vdd', output_file='layer_m1_vdd.png')

# Plot with custom bin size
plot_layer(graph, layer_id='3', bin_size=1000, output_file='layer3_binned.png')

# Plot different statistics
plot_layer(graph, layer_id='5', statistic='total_capacitance', 
          output_file='layer5_cap.png')

plot_layer(graph, layer_id='M1', statistic='avg_voltage',
          output_file='layer_m1_voltage.png')

# Show plot interactively instead of saving
plot_layer(graph, layer_id='5')  # output_file=None shows plot
```

### Plot All Layers

```python
from pdn_parser import plot_all_layers

# Plot all layers in a single figure
plot_all_layers(graph, output_dir='./plots')

# Plot all layers for specific net
plot_all_layers(graph, net='vdd', output_dir='./plots')

# Show interactively
plot_all_layers(graph, show=True)
```

### Visualization Statistics

The `statistic` parameter controls what is displayed in the heatmap:

| Statistic | Description | Use Case |
|-----------|-------------|----------|
| `node_count` | Number of nodes per bin (default) | Overall structure, density |
| `avg_voltage` | Average node voltage per bin | Voltage drop analysis |
| `total_capacitance` | Sum of capacitances per bin (fF) | Decoupling capacitance distribution |
| `avg_resistance` | Average resistance per bin (KOhm) | Resistance uniformity |

### Handling Large Networks

For large PDN netlists (>100K nodes per layer), the visualization automatically uses spatial binning:

**Automatic Bin Size Calculation:**
```python
bin_size = (max_coord - min_coord) / sqrt(num_nodes)
```

This targets approximately 100-1000 bins per layer for reasonable plot sizes.

**Manual Bin Size Control:**
```python
# Small bins (high resolution, slower for large networks)
plot_layer(graph, '5', bin_size=100)

# Large bins (faster, lower resolution)
plot_layer(graph, '5', bin_size=5000)

# Auto-calculate (recommended)
plot_layer(graph, '5', bin_size=None)  # Default
```

**Performance Tips:**
- Use larger bin sizes (>1000) for very large layers (>1M nodes)
- Filter to specific nets before plotting: `filter_by_layer(graph, '5', net='vdd')`
- Save to file instead of interactive display for batch processing
- Use `plot_all_layers()` to generate overview plots efficiently

### Customization

```python
# Custom colormap
plot_layer(graph, '5', colormap='plasma')  # Options: viridis, plasma, inferno, magma, jet

# Custom title
plot_layer(graph, '5', title='VDD Layer 5 - Node Density')

# High-resolution output
plot_layer(graph, '5', output_file='layer5_hires.png')  # Default DPI=150
```

### Example: Comparing Layers

```python
import matplotlib.pyplot as plt
from pdn_parser import filter_by_layer

# Compare node count across layers
layer_stats = graph.graph['layer_stats']
layers = sorted([k for k in layer_stats.keys() if k is not None])
node_counts = [layer_stats[l]['nodes'] for l in layers]

plt.figure(figsize=(10, 6))
plt.bar(layers, node_counts)
plt.xlabel('Layer')
plt.ylabel('Node Count')
plt.title('Node Distribution Across Layers')
plt.savefig('layer_comparison.png')
```

## Node Attributes

Each node in the graph has the following attributes:

- `name` (str): Node name
- `x` (int or None): X coordinate (extracted from name or tile)
- `y` (int or None): Y coordinate
- `layer` (str or None): Layer identifier (e.g., '5', 'M1', 'AP') from X_Y_LAYER naming
- `is_boundary` (bool): True if boundary node requiring stitching
- `is_package` (bool): True if node is in package model
- `is_vsrc_node` (bool): True if node is connected to voltage source via zero-resistance path
- `net_type` (str or None): Power net type (e.g., 'vdd', 'vss')
- `voltage` (float or None): Node voltage (if known)
- `tile_id` (tuple or None): (x, y) tile ID where node is defined

## Edge Attributes

Each edge (element) has the following attributes:

- `type` (str): Element type ('R', 'C', 'L', 'V', 'I', 'E', 'F', 'G', 'H')
- `value` (float): Element value (KOhm, fF, nH, V, mA, etc.)
- `elem_name` (str): Element name from netlist
- `tile_id` (tuple or None): Tile where element is defined

Additional attributes for specific element types:

**Current Sources (I):**
- `dc` (float): DC current value (mA)
- `inst_x`, `inst_y` (int): Coordinates extracted from instance name
- `fsdb_path` (str): Path to FSDB waveform file
- `fsdb_coeff` (float): Waveform coefficient
- `fsdb_shift` (float): Waveform time shift
- `pwl` (str): PWL waveform definition

**Voltage Sources (V):**
- `dc` (float): DC voltage value
- `ac` (float): AC magnitude
- `portid` (str): Port ID for S-parameter extraction
- `pwl` (str): PWL waveform definition

**Capacitors (C):**
- `nlcap_model` (str): Nonlinear capacitor model name

**Controlled Sources (E/F/G/H):**
- `source_type` (str): Type of controlled source
- `ctrl_pos`, `ctrl_neg` (str): Control node names

## Validation

The `--validate` flag enables sanity checks:

### Short Detection

Identifies resistors below threshold (< 1e-6 KOhm = 1e-3 Ohm):

```
WARNING: Found 5 shorted resistors:
  r_short1: vdd_100_200 <-> vdd_110_200 = 1.23e-07 KOhm
  r_short2: gnd_100_200 <-> gnd_110_200 = 5.67e-08 KOhm
  ...
```

### Floating Node Detection

Identifies nodes not connected to any voltage source:

```
WARNING: Found 123 floating nodes (not connected to voltage source)
First 10: node_100_200, node_150_300, ...
```

### Merged Node Tracking

Records nodes merged during stitching or short removal:

```python
merged_nodes = graph.graph['merged_nodes']
# [(from_node, to_node, merge_type), ...]
```

Merge types:
- `0`: Stitch merge (boundary node stitching)
- `1`: Short merge (shorted resistor removal)
- `2`: Ground merge (grounded node removal)

## Error Handling

### Default Behavior (Warning Mode)

Errors are logged as warnings, parsing continues:

```bash
WARNING: Invalid resistor line: R1 node1
WARNING: Failed to parse tile 5_7: unexpected EOF
```

### Strict Mode

Errors cause immediate failure:

```bash
python pdn_parser.py --netlist-dir /path --strict
ERROR: Invalid resistor line: R1 node1
```

Use strict mode for validation or when netlist correctness is critical.

## Performance

### Memory Usage

For large netlists (millions of nodes):
- Enable memory profiling: `--profile-memory`
- Monitor peak usage with external tools

### Parsing Speed

Typical performance on modern hardware:
- Small netlist (1K nodes): < 1 second
- Medium netlist (100K nodes): 5-10 seconds  
- Large netlist (1M+ nodes): 1-5 minutes

Progress bars show real-time parsing status.

## Troubleshooting

### Import Errors

```
ERROR: NetworkX is required. Install with: pip install networkx
```
Solution: `pip install networkx`

### File Not Found

```
FileNotFoundError: Main netlist file not found: /path/ckt.sp
```
Solution: Ensure `ckt.sp` exists in the specified directory

### Gzip Errors

```
WARNING: Failed to parse tile 0_0: gzip decompression error
```
Solution: File may be corrupted or not actually gzipped

### Node Name Collisions

```
ERROR: Node name collisions detected:
  Node 'vdd_100_200' appears in: tile_0_0, tile_0_1
```
Solution: Check netlist for duplicate node definitions (may indicate netlist bug)

## Differences from C++ Parser

This Python parser is based on the mpower C++ parser but has some differences:

1. **No MPI support**: Single-process only (no distributed parsing)
2. **Simplified validation**: Basic short/floating checks (no full DC analysis)
3. **Limited subcircuit support**: Flattening is simplified
4. **No FSDB reading**: Only stores FSDB paths, doesn't read waveform data
5. **No device models**: BSIM4, nonlinear caps not fully supported
6. **Enhanced visualization**: Grid-based heatmap plotting (not in C++ parser)
7. **Layer support parity**: Both parsers support X_Y_LAYER node naming

For production simulation, use the C++ parser. This Python tool is for:
- Graph analysis and visualization
- Layer-based PDN analysis
- Prototyping new analysis algorithms  
- Interfacing with Python-based tools
- Educational purposes

## Contributing

This is a standalone tool based on the mpower simulator parser. Enhancements welcome:

- Better subcircuit expansion
- FSDB waveform reading
- More validation checks
- Performance optimizations
- Additional export formats

## License

Based on mpower simulator parser implementation.

## Contact

For questions about the underlying netlist format or mpower simulator, refer to the main simulator documentation.

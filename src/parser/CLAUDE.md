# `src/parser/` — SPICE-like PDN netlist parser

> Root `CLAUDE.md` covers the PDN directory layout, element syntax, and node naming. This file is the API and internals reference for the parser.

## Entry points

- `parser.netlist.NetlistParser` — programmatic
- `parser.pdn_parser` — CLI (`python -m parser.pdn_parser …` / `pdn-parse` console script)

```python
from parser.netlist import NetlistParser

parser = NetlistParser(
    netlist_dir,
    validate=False,                  # full structural validation pass (slow)
    strict=False,                    # raise on parse warnings
    parallel=False, n_workers=None,  # multiprocessing for tiles
    chunk_size=None,                 # chunk size inside large tiles
    net_filter=None,                 # 'VDD' or ['VDD','VSS'] — drops other nets
    store_instance_sources=False,    # serialize CurrentSource → dict (see below)
)
graph = parser.parse()
```

Returns a `RustworkxMultiDiGraphWrapper` whose `.graph` (top-level) carries metadata: `net_connectivity`, `vsrc_nodes`, `instance_node_map`, `pg_net_voltage`, `_instance_sources_objects` (or serialized `instance_sources`), and the `PowerGridMetaData` used by the distributed parser.

## Memory-optimized edge attributes (`edge_attrs.py`)

Default ON. Specialized slotted dataclasses replace dicts (~95% per-edge memory cut, critical for 100M+ edge netlists ≈ 65 GB → ≈ 4 GB).

```python
from parser.graph_builder import get_use_optimized_edges, set_use_optimized_edges

set_use_optimized_edges(False)   # disable for tiny netlists or backward compat
```

**Edge classes:**

| Class | When used | Stores `elem_name`? |
|---|---|---|
| `ResistorEdge` | die resistors (~99.9% of resistors) | no |
| `ResistorEdgeWithName` | resistors matching `vsrc_resistor_pattern` (default `'rs'`) | yes |
| `CapacitorEdge`, `InductorEdge`, `CurrentSourceEdge` | passives & I-edges | no |
| `VoltageSourceEdge` | vsrc edges | yes (always) |

**Always access via `.get`:**

```python
elem_name = data.get('elem_name', '')   # safe: '' for die resistors
# data['elem_name']                     # KeyError on die resistors
```

Computed properties (`.tile_id`, `.net_type`) unpack on the fly and are 4–5× slower than dict access — cache values locally inside hot loops.

## Pickle compatibility

| Mode | Pickle behavior | When to use |
|------|-----------------|-------------|
| `store_instance_sources=False` (default) | Stores raw `CurrentSource` objects in `graph.graph['_instance_sources_objects']`. Pickle works but loader must import `parser`. | In-process / same-version use; saves ~60% parse-time RAM on 1M-source netlists (1.7 GB → 1.1 GB). |
| `store_instance_sources=True` | Serializes to plain dicts in `graph.graph['instance_sources']`. | Long-term archival, sharing pickles across environments. |

Both dynamic and transient solvers handle both formats transparently.

## Current-source data structures (`current_sources.py`)

Parsed from `instanceModels_*.sp`. Returned by `_parse_current_source_line` and held on `CurrentSource`:

- `Pulse(...)` — pulse waveform; `evaluate(time)`, `get_dc()`
- `PWL(...)` — piecewise-linear; `evaluate(time)`, `get_dc()`
- `CurrentSource` — full source with `dc_value`, `static_value`, `pulses`, `pwls`, plus `get_static_current()` and `get_current_at_time(t)`
- `_DCOnlyCurrentSource` — internal lightweight variant when `optimize_dc_only=True`
- `InstanceInfo` — parsed instance name (net, pin, tile location)

```python
sources = graph.graph.get('_instance_sources_objects', {})
for name, src in sources.items():
    static_ma = src.get_static_current()
    i_t = src.get_current_at_time(1e-9)
```

`_parse_current_source_line` returns **Amperes**; conversion to mA happens post-parse in `_prepare_instance_source` (`current_sources.py`, mirrored in `parallel.py`). If you call it directly, multiply by `I_TO_MA` yourself.

## Parallel parsing (`parallel.py`, `_parse_*_worker`)

For large netlists with many tiles:

```python
parser = NetlistParser(netlist_dir, parallel=True, n_workers=8,
                       net_filter='VDD', chunk_size=10000)
```

Mechanics:

- Memory-mapped file access for plain text (gzip falls back to streaming)
- Chunk-based processing inside large tiles
- Bulk graph operations during merge
- Output is bit-equivalent to sequential parsing

Roughly 6–8× speedup on 100+ tile netlists.

## Tunables (module-level globals)

| Function | Effect |
|---|---|
| `set_use_optimized_edges(bool)` | Slotted edge classes vs dicts |
| `set_apply_wscale(bool)` | Apply per-edge `wscale` weighting on resistors |
| `set_optimize_dc_only(bool)` | Use `_DCOnlyCurrentSource` (drops Pulse/PWL waveform data) — large memory win for static-only flows |

Module-level globals do **not** propagate to spawned multiprocessing workers automatically; the parser passes them through worker args explicitly. They also do not propagate to Ray workers — see `src/distributed/CLAUDE.md`.

## Sub-files at a glance

| File | Role |
|---|---|
| `netlist.py` | `NetlistParser`, the orchestrator |
| `pdn_parser.py` | CLI + helpers (also where mA conversion lives) |
| `graph_builder.py` | builds the rustworkx graph from token streams |
| `spice_lexer.py` | element-line tokenizer |
| `parallel.py` | worker functions + `TileParseResult` / `InstanceParseResult` |
| `edge_attrs.py` | slotted edge classes |
| `metadata.py` | `pg_net_voltage`, `additional_vsrcs`, vsrc node detection |
| `current_sources.py` | `Pulse`, `PWL`, `CurrentSource`, `InstanceInfo` |
| `sampled_netlist.py` | sampled multi-tile netlist generator |

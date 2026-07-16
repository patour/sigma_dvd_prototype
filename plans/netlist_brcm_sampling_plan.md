# Plan: Generate a ~10x-smaller sampled netlist from `netlist_brcm`

## REVISED 2026-07-16 — node-drop sampling replaced by contraction/coarsening

**§4.2/§4.3 below (and the original §4.2/§4.3 decisions-table rows in §3) are superseded.** The
originally-implemented algorithm — layer-stratified independent random NODE sampling (§4.2) +
BFS/via-chain connectivity repair for mandatory nodes only (§4.3) — was run end-to-end and
measured in `docs/brcm_distributed_runtime_optimization.md` §7.5. It produced an invalid proxy:

- Nodes reduced 10× but **resistors 26×**, so **R/node collapsed 1.87 → 0.72** (below the ≥1.0
  needed for connectivity — an edge only survives a node-drop pass if BOTH endpoints are
  independently kept, so P(edge survives) ≈ retention² for optional-optional pairs, while
  mandatory-only repair can't rescue the rest).
- **83% of sampled nodes were dropped as resistively floating** at parse time (10,899 global
  islands, 11,034 penalized) — an effective 60× reduction instead of the intended 10×.
- Interface Schur density collapsed 9.9% → 0.51% (224× nnz drop), inverting the per-step profile
  (interface solve 51%→5% of BRCM's step share) — the sampled netlist no longer exercises the
  bottleneck it exists to proxy for. Non-physical solve (max_drop 1.29 V on a 0.76 V rail).

**Fix (implemented in `src/parser/brcm_tile_sampler.py`): contraction/coarsening.** Instead of
independently keeping/dropping nodes and their edges, every optional node is assigned to a small
geometric cell (per layer), split into connected components within that cell (so two disconnected
mesh fragments sharing a cell are never spuriously merged), and each resulting cluster collapses
to one representative node with parallel-merged conductances/capacitances. This preserves
connectivity **by construction** (a quotient of a connected graph is connected — no BFS repair
phase needed, just a cheap mandatory-node degree sanity assertion), preserves total capacitance
exactly, and keeps R/node close to the source ratio because mesh edges map to coarse-mesh edges
instead of vanishing. Fully deterministic (no RNG in the node/edge contraction passes); the
`base_seed` parameter now affects only Pass 4's current-source down-sampling.

## 1. Problem statement

`netlist/netlist_brcm` is a real, production-scale PDN netlist (not a synthetic
grid): 6x6 = 36 tiles, ~30.8M nodes, ~57.6M R/C edges, ~1.68M current sources
(from the existing `distributed_pkl/parse_20260710_205356.log`). We need a new,
standalone netlist directory `netlist/netlist_brcm_sampled/` that:

- Has the **same 36 tiles** (6x6 grid, same `.die_area`, same absolute
  coordinates — no spatial cropping/re-tiling).
- Has the **same number of pad (voltage source) nodes** as the source.
- Is **~10x smaller** in node/resistor/capacitor/current-source count per
  tile, while preserving the statistical character ("similar distribution")
  of the RC mesh and current sources.
- Can be parsed fresh via `sigma-dvd parse` / `DistributedNetlistParser` and
  solved (DC/quasi-static) without floating islands or missing pad
  attachments.

This is architecturally different from `src/parser/sampled_netlist.py`
(built for `netlist_minion`), which spatially **crops** a single-tile die and
**invents new synthetic pads/tiles**. `netlist_brcm` already has 36 real
tiles and 309 real pads anchored at fixed absolute coordinates, so the
correct approach here is **in-place sparsification** of each existing tile,
not spatial cropping — and, given the ~30M-node scale, a **streaming,
per-tile, text-level** approach rather than loading a monolithic graph
(which is exactly what the distributed architecture avoids for large PDNs).

## 2. Key facts discovered during analysis

- **Nets**: `pg_net_voltage` defines 4 nets — `VDD_VAR` (0.76V), `VSS` (0V),
  `pad_PLL_VDD1p5` (1.5V), `pad_PLL_VSS` (0V).
- **Pads**: all defined in the single top-level `package.ckt` (included by
  `ckt.sp`; per-tile `package_X_Y.ckt` files exist on disk but are **not**
  read by `DistributedNetlistParser._parse_package`, which only opens
  `netlist_dir/package.ckt`).
  - VDD_VAR: 267 bump pads (`v_bmpary_bmp_VDD_VAR_*`) + 42 probe pads
    (`v_bmpary_prb_VDD_VAR_*`) = **309 voltage sources**.
  - Each pad is a small closed loop of symbolic nodes
    (`..._vsrc` / bare name / `..._probe` / `..._int`) that is anchored to
    the real conductive die mesh by exactly **one zero-ohm `rs` line**:
    `rs <die_coord_node> <bump_node> 0` (e.g.
    `rs 1197000_449800_86 bmpary_bmp_VDD_VAR_0_1 0`). These 309
    `<die_coord_node>` targets (layer `86`, the top bump/RDL layer) are
    **hard anchors** — they must survive sampling or that pad becomes
    disconnected.
- **Tile files** (`tile_X_Y.ckt`/`.nd`, `instanceModels_X_Y.sp`) are gzip
  SPICE text, values in raw **Ohms / Farads / Amperes** (parser converts
  via `R_TO_KOHM=1e-3`, `C_TO_FF=1e15`, `I_TO_MA=1e3`).
  - `.ckt` header: `.node_count <n>` then `.flag_boundary` (both effectively
    informational/no-ops to the parser) then `r`/`c` element lines.
  - Boundary (shared, 2+ tile) nodes are marked with a **literal `*` prefix**
    directly on the node token in the element line, e.g.
    `r 3181040_928205_43 *3192000_928205_43 127.66`. 137,900 such nodes
    exist design-wide per the parse log.
  - **Via nodes**: same `(x, y)` coordinate, different layer, connected by a
    resistor, e.g. `r 17480_15200_57 17480_15200_59 5.2877`. Regular
    same-layer routing nodes have no such edge.
  - `.nd` format: `<node_name> <v1> <v2> <v3> <v4> <net_name>` — only
    tokens[0] and tokens[5] are used by the parser; the 4 middle fields can
    be carried through unchanged.
  - Tile capacitors are always grounded (`c node 0 value`) — no coupling
    caps at tile level (per root `CLAUDE.md`), so dropping a cap never
    breaks DC connectivity.
- **Unused auxiliary files** (confirmed via `grep` across `src/`, no
  matches): `package_X_Y.ckt`, `instanceList_X_Y`, `portList_X_Y`,
  `decap_cell_list`, `switch_cell_list`, `gated_nets`, `probe_list`,
  `switch_cell.model`, `*.ctotal`. Safe to omit entirely.
- `ckt.sp` **is** read by `DistributedNetlistParser` (for `.partition_info`
  and `.include` tile/instance discovery), so it must be regenerated
  correctly (not just decorative).
- **`netlist_brcm/distributed_pkl` already exists and is VDD_VAR-filtered**
  — verified by loading it directly:
  - `distributed_pkl/metadata.pkl` -> `{'metadata': PowerGridMetaData, 'boundary_nodes': ...}`.
    `metadata.net_name == 'VDD_VAR'`, `metadata.vdd == 0.76`, 36 tile
    configs.
  - `metadata.package_data` (`PackageData`) already has
    `pad_nodes` (309, the bump/probe symbolic vsrc nodes) and
    `die_attachment_nodes` (309, the real coordinate nodes each pad's `rs`
    line anchors to, all layer `86`) **precomputed** — no need to
    hand-parse `package.ckt`'s `rs` lines ourselves.
  - `distributed_pkl/tile_X_Y.pkl` -> a `TileData` object per tile:
    `all_nodes: Set[str]` (VDD_VAR only), `resistive_edges: List[(u,v,g_mS)]`
    (**conductance** in mS, not resistance), `capacitive_edges:
    List[(u,v,c_fF)]`, `boundary_nodes: Set[str]` (already identified — no
    need to scan for `*` prefixes), `current_injections: Dict[node, mA]`.
    Verified against tile `(1,5)`: 203,097 nodes, 378,651 resistive edges,
    210,959 capacitive edges, 3,371 boundary nodes, 8,369 current
    injections (exactly matches the parse log's per-tile current-source
    count).
  - **Caveat**: `current_injections` is a **flattened per-node DC scalar**
    only — it does not preserve original instance names, `pulse(...)`, or
    `pwl(...)` waveform data (those exist only in the raw
    `instanceModels_X_Y.sp` text). Per user decision, current sources are
    therefore sampled from the **raw text file**, not from this pkl field
    (see §4.4); the pkl is used only for mesh topology and node
    classification.
  - Verified per-tile layer distribution is **not monotonic in raw layer
    index** but is strongly bimodal by depth: e.g. tile (1,5) layers
    43-55 (bulk/bottom routing) each have thousands-to-tens-of-thousands of
    nodes (layer 51: 33,131; layer 47: 33,130; ... layer 45: 16,105), while
    layers 57 and up drop off sharply toward the pad layer (layer 61/65/69:
    7,726 each; ... layer 81: 20; **layer 86 (the pad-anchor layer): 11**).
    This confirms "bottom layers have the most nodes" and motivates
    layer-stratified sampling (§4.2).

## 3. Confirmed decisions (from user Q&A)

| Decision | Choice |
|---|---|
| Net scope | **VDD_VAR only** (single net, ~309 pads) |
| `package.ckt` / `ckt.sp` header | Copied unchanged, **filtered to VDD_VAR-only** lines/parameters; same `.die_area`, `.partition_info 6 6`, absolute coordinates |
| **Data source for mesh + classification** | **`distributed_pkl/tile_X_Y.pkl` (`TileData`) and `distributed_pkl/metadata.pkl` (`PackageData.die_attachment_nodes`)** — already VDD_VAR-filtered, decompressed, unit-converted (mS/fF/mA), with boundary nodes and pad-anchor nodes precomputed. Avoids re-parsing/re-filtering raw gzip text for the mesh |
| **Data source for current sources** | **Raw `instanceModels_X_Y.sp` text** (not the pkl) — the pkl's `current_injections` is a flattened per-node DC scalar with no instance names/waveforms, so pulse/PWL/wscale fidelity requires reading the original SPICE text |
| Die-mesh sampling method | **REVISED: layer-stratified geometric contraction/coarsening** (not grid/pitch decimation, not random edge sampling, not independent node-drop sampling — see the banner at the top of this doc) |
| **Layer stratification** | Rank each tile's layers by their **original node count** (descending = bulk/bottom-of-stack layers first). Assign each layer a retention weight **inversely related to its count** (same budget/bisection math as the original design), so sparse upper layers (nearest the pad/RDL layer, e.g. layer 86 with only 11 nodes in tile (1,5)) contract to **~100% singleton representatives**, while dense bulk layers absorb most of the reduction via geometric-cell clustering |
| Reduction ratio granularity | **Per-tile independently** (~10x per tile, preserving each tile's original relative size) |
| Mandatory-keep nodes | Pad-anchor (`die_attachment_nodes`) nodes, boundary (`TileData.boundary_nodes`) nodes, and current-source-bearing nodes are always retained as **singleton cluster representatives** — never absorbed into a cluster and never absorbing other nodes |
| Current sources | **Force-keep** any node with a current source attached (even if contraction wouldn't have made it a representative), then **independently down-sample the current source instances themselves by ~10x** among those force-kept nodes (values/waveforms copied verbatim, not rescaled) |
| Connectivity repair | **REVISED: not needed** — contraction preserves connectivity by construction (a quotient of a connected graph is connected). Replaced by a cheap mandatory-node degree sanity `AssertionError` in `contract_tile_edges` (fail loud if a mandatory node ends up with zero degree post-contraction — a contraction bug, not a data condition) |
| Current source net filtering | `instanceModels_X_Y.sp` interleaves **multiple nets'** current sources in the same file (instance names are structured as `...:<NET>:...`, e.g. `i_..._bank0:VDD_VAR:VDD:0:0:0:0:0`). Only lines whose structured name field is **`VDD_VAR`** are considered for classification and down-sampling; all other nets' lines are fully ignored (reusing `parser.spice_lexer._has_structured_instance_names` / `_fast_instance_net_filter` + `parser.current_sources._prepare_instance_source`, not a hand-rolled substring check) |
| Capacitor-follows-current-source | Any node whose **surviving** (post-10%-down-sampling) VDD_VAR current source is kept **must** retain its grounded capacitor line in the output tile `.ckt`, even though the node itself was already force-kept for the broader "current-source-bearing" reason — treated as an explicit, separately-verified invariant rather than an implicit side effect |
| Auxiliary unused files | **Omit entirely** from output |
| Validation | **Full**: `sigma-dvd parse` the sampled netlist -> `distributed_pkl`, run a DC/quasi-static solve, verify all 309 pad-anchor nodes and all boundary nodes survived and are connected, sanity-check IR-drop magnitude |
| Output location | **`netlist/netlist_brcm_sampled/`** (new sibling directory — `netlist_brcm` is a symlink into a shared external path and must not be written into) |

## 4. Detailed algorithm (per tile, executed independently for all 36 tiles)

For each `tile_X_Y`:

### 4.1 Pass 1 — load + classify nodes (from `distributed_pkl`, not raw text)

1. Load `distributed_pkl/tile_X_Y.pkl` (`TileData`): `all_nodes`,
   `resistive_edges` (mS), `capacitive_edges` (fF), `boundary_nodes`,
   `current_injections`. This is already VDD_VAR-filtered, decompressed,
   and unit-converted — no raw-text scanning needed for the mesh.
2. Build an adjacency structure (plain dict/array, not `rustworkx`) from
   `resistive_edges`, keyed by node, for the connected-component split
   (isolated-node detection + per-cell CC-split) reused in Pass 2.
3. Classify each node in `all_nodes`:
   - **via** vs **non-via**: a node is `via` if it has >=1 resistive edge to
     another node sharing the same `(x, y)` but a different layer (parsed
     from the node name suffix).
   - **layer**: parsed directly from the node name (e.g.
     `1197000_449800_86` -> layer `86`).
   - **boundary**: node is in `TileData.boundary_nodes` (already computed
     — no `*`-prefix scanning needed).
   - **pad-anchor**: node is in `metadata.pkl`'s
     `package_data.die_attachment_nodes` (309 design-wide; filter to those
     whose coordinates fall within this tile's bounds).
   - **current-source-bearing**: determined separately in Pass 4 from the
     *raw* `instanceModels_X_Y.sp` text (VDD_VAR-filtered), since the pkl's
     `current_injections` doesn't preserve enough info for later waveform
     reconstruction — Pass 4 feeds its node set back into the
     mandatory-keep set before Pass 2 runs.
4. **Mandatory-keep set** = pad-anchor ∪ boundary ∪ current-source-bearing
   (VDD_VAR only, from Pass 4's pre-scan). **Optional pool** = all
   remaining nodes in the tile, grouped by layer.

### 4.2 Pass 2 — layer-stratified geometric contraction (REVISED, supersedes the
original node-sampling text — see the banner at the top of this doc)

Implemented as `contract_tile_nodes` in `src/parser/brcm_tile_sampler.py`. Fully
deterministic — **no RNG anywhere in this pass**.

1. **Per-layer retention targets `r_L`** — identical budget math to the original design:
   compute the tile's target kept-node count as `target_kept = round(ratio * len(all_nodes))`.
   If `|mandatory-keep set| >= target_kept`, keep the mandatory set as-is (log a warning — this
   tile's reduction will be less than the target ratio). Otherwise distribute
   `remaining = target_kept - |mandatory-keep set|` across the optional pool using per-layer
   weights `w_L = n_L ** (-alpha)` (sparser layers weighted higher) and a single scale factor `s`
   solved by bisection so `sum_L( min(1.0, w_L * s) * n_L ) ~= remaining`. Unlike the original
   design, `r_L = min(1.0, w_L * s)` is now interpreted as "fraction of layer `L`'s optional nodes
   that remain as **cluster representatives**", i.e. expected cluster size on layer `L` is
   `~= 1 / r_L` — not an independent per-node keep probability.
2. **Geometric cell assignment** (per layer `L` with `r_L < 1.0`; layers with `r_L >= 1.0`, or
   absent from the optional pool, make every optional node its own singleton cluster — no
   coordinate parsing needed): parse `(x, y)` for each optional node on `L`. Unparseable-xy nodes
   each become their own singleton cluster (never merged with anything, parseable or not).
   Compute the bbox of the parseable nodes and a cell edge length
   `h = sqrt(bbox_area / (n_parseable * r_L))` so the expected cluster occupies one cell; nodes
   are bucketed by `(floor((x-x0)/h), floor((y-y0)/h))`. Degenerate (zero-width and/or
   zero-height) bboxes fall back to a 1-D or single-shared-cell assignment along whichever axis
   actually varies.
3. **Connected-component split within each (layer, cell) bucket**: union-find restricted to
   resistive edges where BOTH endpoints are optional-pool members of the SAME bucket — this is
   what prevents two disconnected mesh fragments that happen to land in the same geometric cell
   from being spuriously merged (no fabricated connection is ever created).
4. **Representative selection** (deterministic, no RNG): the cluster member minimizing
   `(squared distance to the cluster's coordinate centroid, node_name)`; a cluster with no
   coordinate-parseable member falls back to the lexicographically smallest name.
5. **Originally-isolated nodes** (zero resistive edges in the tile's own adjacency, independent of
   `r_L`): optional isolates are **dropped entirely** (the production parser would island-drop
   them anyway); mandatory isolates are **kept as self-mapped singletons** (never given a
   fabricated edge), logged once with a count.
6. `kept_nodes = mandatory-keep set ∪ {cluster representatives}` (minus dropped optional
   isolates). Mandatory nodes are **always singleton representatives** — never absorbed into a
   cluster and never absorbing other nodes, so boundary/pad-anchor/current-source node names
   survive verbatim for cross-tile stitching, package hookup, and Pass-4 consistency.

### 4.3 Pass 3 — remap edges through the contraction (REVISED, supersedes the
original filter+repair text)

Implemented as `contract_tile_edges`. **No repair phase** — contraction already guarantees every
kept node keeps a path to the rest of the tile (a quotient of a connected graph is connected), so
this pass only needs to remap and merge edges through Pass 2's `node_to_rep` map
(`rep('0') = '0'`, ground is its own fixed representative).

1. **Resistors**: for each `(u, v, g_mS)`, map `ru, rv = rep(u), rep(v)`. If `ru == rv` the edge
   is intra-cluster and dropped (counted). Otherwise accumulate parallel conductance into
   `G[(min(ru,rv), max(ru,rv))] += g_mS` — independently-ordered mappings of the same physical
   edge (e.g. arriving via different original endpoints) still merge into one coarse edge.
2. **Capacitors**: same remap/merge (general case, though tile capacitors are always grounded per
   root `CLAUDE.md`) — grounded caps of absorbed cluster members accumulate onto their
   representative, so total tile capacitance is preserved exactly (minus caps on dropped
   optional isolates, counted separately, never conserved).
3. **Mandatory-degree sanity check**: every mandatory node that had resistive degree > 0 in the
   original (pre-contraction) adjacency must have degree > 0 in the contracted edge list; a
   violation raises `AssertionError` — this indicates a bug in the contraction itself, not a data
   condition, and is deliberately NOT downgraded to a warning-and-continue.

### 4.4 Pass 4 — current sources (from raw text, pre-scanned before Pass 2)

1. Iterate `instanceModels_X_Y.sp` line by line (raw gzip text, *not* the
   pkl); for each line, apply the fast structured-name net filter
   (`_has_structured_instance_names` / `_fast_instance_net_filter` for
   `VDD_VAR`) to select only VDD_VAR-net lines — lines for other nets are
   skipped entirely (not written, not counted, not sampled, and never
   force-keep a node). Use `_prepare_instance_source(line)` only to extract
   `node_pos` for classification/keep decisions; the **original raw line
   text is retained verbatim** for output (no reconstruction of
   `pulse(...)`/`pwl(...)`/`static_value=`/`wscale=` syntax needed — this
   avoids any risk of a serializer round-trip bug).
2. **Pre-scan** (runs before Pass 2): collect the set of all `node_pos`
   values from VDD_VAR lines — this feeds the "current-source-bearing"
   mandatory-keep category in Pass 1/§4.1.
3. **Down-sample** (runs after Pass 2, once `kept_nodes` is final): among
   VDD_VAR lines whose `node_pos` is in `kept_nodes` (guaranteed, since
   current-source-bearing nodes are force-kept), randomly keep ~10% of the
   lines (fixed seed), writing the kept lines verbatim.
4. Nodes whose current source(s) all got dropped simply remain as ordinary
   (zero-current) resistor-mesh nodes — fine, they're still needed for
   `.nd`/mesh consistency (kept for another mandatory reason, or because
   losing a source doesn't remove the node from `kept_nodes`).
5. **Capacitor-follows-current-source invariant**: for every current-source
   line that survives this down-sampling, explicitly verify that its
   `node_pos`'s grounded capacitor line (if `TileData.capacitive_edges`
   originally had one) is present in the tile's sampled `.ckt` output.
   Because current-source-bearing nodes are unconditionally force-kept
   (§4.1/§4.2) and capacitor retention in §4.3 keys off `kept_nodes`
   membership, this holds by construction — but treat it as an **explicit,
   separately asserted invariant** (fail loud if violated) rather than an
   unverified side effect, since it's a specific correctness requirement.

### 4.5 Write outputs

- `tile_X_Y.ckt` (gzip): updated `.node_count`, `.flag_boundary`, `r`/`c`
  lines regenerated from the contracted `kept_nodes`/edge sets
  (converting `TileData`'s mS/fF back to raw Ohms/Farads:
  `R_ohm = 1000/g_mS`, `C_farad = c_fF * 1e-15`), `*` prefix re-added for
  nodes in `TileData.boundary_nodes`.
- `tile_X_Y.nd` (gzip): filtered to `kept_nodes`; since the pkl doesn't
  carry the `.nd` file's 4 cosmetic middle fields, these lines are copied
  verbatim from the **original raw `.nd` file** for kept nodes (cheap —
  `.nd` is small relative to `.ckt` and the parser only actually consumes
  `tokens[0]`/`tokens[5]`, so exact fidelity of the middle fields is a
  nice-to-have, not a correctness requirement).
- `instanceModels_X_Y.sp` (gzip): filtered current-source lines, written
  verbatim from the raw source text (§4.4).
- `package.ckt`: single top-level file, VDD_VAR-only lines copied verbatim
  from the source (bump/probe loops + `rs` anchors + `.print` lines),
  unchanged coordinates. Cross-check against `metadata.pkl`'s
  `package_data.pad_nodes`/`die_attachment_nodes` counts (309/309) as a
  sanity check that the text filter matches the already-parsed data.
- `ckt.sp`: regenerated header (`.partition_info 6 6`, `.die_area ...`,
  `.parameter VDD_VAR 0.76`, `vVDD_VAR ...`) + `.include` list for the 36
  tile files + 36 instance-model files + `package.ckt`.
- `pg_net_voltage`: single `VDD_VAR 0.76` line.
- `additional_vsrcs`: empty (matches source).
- No `package_X_Y.ckt`, `instanceList_X_Y`, `portList_X_Y`,
  `decap_cell_list`, `switch_cell_list`, `gated_nets`, `probe_list`,
  `switch_cell.model`, `.ctotal` files.

## 5. Implementation

- New module, e.g. `src/parser/brcm_tile_sampler.py` (or similar name —
  adjustable), implemented as a standalone script/CLI (not reusing
  `sampled_netlist.py`'s graph-based class, since the algorithm and data
  scale/inputs are fundamentally different: pkl-driven per-tile mesh
  sampling + raw-text current-source handling, vs. loading one monolithic
  pickle graph for a single-tile synthetic-like source).
- **Primary data source = `distributed_pkl/`** (already produced by
  `sigma-dvd parse ./netlist/netlist_brcm --net VDD_VAR`): load
  `metadata.pkl` once (for `package_data.die_attachment_nodes`, tile grid),
  then `tile_X_Y.pkl` per tile (for `all_nodes`/`resistive_edges`/
  `capacitive_edges`/`boundary_nodes`) — this is what makes Passes 1-3 fast
  and avoids re-parsing ~57.6M raw R/C element lines.
- **Current sources only** still read the raw gzip
  `instanceModels_X_Y.sp` text directly (§4.4), reusing
  `parser.spice_lexer._has_structured_instance_names` /
  `_fast_instance_net_filter` for the VDD_VAR structured-name filter, and
  `parser.current_sources._prepare_instance_source` to extract `node_pos`
  for classification/keep decisions (output still uses the raw line text,
  not a reconstructed one) — mirrors exactly how
  `distributed.tile_parsing._iter_instance_sources` filters sources in the
  production parser, keeping sampling semantics consistent with how
  `sigma-dvd parse --net VDD_VAR` will later read the sampled output.
- Processes tiles independently (embarrassingly parallel — could optionally
  use a process pool across the 36 tiles, given each tile is 200K-1.6M
  nodes and fully self-contained).
- Uses plain Python dicts/arrays (not `rustworkx`/`networkx`) for per-tile
  adjacency to keep memory bounded per tile.
- Pass 2/3 (geometric contraction, §4.2/§4.3) are fully deterministic with
  **no RNG at all** — same input always produces byte-identical output. The
  configurable `--seed`/`base_seed` affects **only** Pass 4's current-source
  down-sampling (§4.4).

## 6. Validation

1. Run `sigma-dvd parse ./netlist/netlist_brcm_sampled --net VDD_VAR
   --backend ray -o ./netlist/netlist_brcm_sampled/distributed_pkl`
   (or `--backend local` if scale allows) and confirm:
   - No "floating island" errors/warnings beyond expected.
   - No "No pad (voltage source) nodes found" warning.
   - Per-tile node/edge/current-source counts are roughly ~10x smaller than
     the corresponding source-tile counts in
     `distributed_pkl/parse_20260710_205356.log`.
   - All 309 pad-anchor nodes are present and connected (assert against the
     anchor list computed during generation).
   - **Capacitor-follows-current-source invariant** holds: for a sample of
     surviving current sources, their node's grounded capacitor line exists
     in the corresponding sampled tile `.ckt`.
2. Run a DC or quasi-static distributed solve
   (`sigma-dvd solve ... --mode dc` or `quasi-static`) and sanity-check the
   resulting IR-drop magnitude is physically reasonable (same order as the
   full netlist, not NaN/exploding, no isolated-node solver errors).
3. Report final statistics (nodes/R/C/current-sources per tile and total,
   before/after, reduction ratio achieved) similar to
   `sampled_netlist.py`'s `compute_statistics()`.
4. **Acceptance criteria from the §7.5 failure analysis** (the reason this
   redesign exists — a re-measurement that doesn't clear these is not a valid
   proxy, regardless of how close it gets to 10x node reduction):
   - **Parse-time islands ≈ 0** (`Islands penalized ≈ 0` in the parse log, not
     the 10,899/11,034 seen pre-contraction).
   - **Kept-interior ≈ node count** — i.e. the parser's post-island-removal
     interior node count should match `sampling_report.json`'s
     `total_sampled_nodes`, not collapse to a fraction of it (pre-contraction:
     514K of an intended 3.08M, an effective 60x instead of 10x).
   - **R/node within ~±30% of the source's** — `sampling_report.json`'s
     `totals.r_per_node_before` vs `totals.r_per_node_after` (pre-contraction:
     1.87 → 0.72, a 61% collapse and below the ≥1.0 connectivity floor).
   - **Per-step profile share sanity**: a short transient re-run's RHS/solve/
     recovery percentage split should stay within ~±10 points of full BRCM's
     32/51/16 (§7.5) — if the interface-solve share collapses toward single
     digits, the sample no longer exercises the bottleneck it exists to proxy.
   - Seeds now affect **only** Pass 4's current-source down-sampling — a
     re-run with a different `--seed` should reproduce identical node/edge
     counts and topology, differing only in which current-source lines survive.

## 7. Open/adjustable items for review

- Exact new module name/location (`src/parser/brcm_tile_sampler.py` is a
  placeholder suggestion).
- Random seed value and whether to expose sampling ratio as a CLI
  parameter (default ~10%).
- Whether `local` or `ray` backend is preferred for the validation solve
  (depends on available compute in this environment).

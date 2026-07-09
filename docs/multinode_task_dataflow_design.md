# B4: Multi-Node Ray Cluster via Dynamic Task-Based Dataflow

**Design document for 100M-node PDN distributed solving across k smaller machines.**

Branch: `distributed-10x` | Spec: plan section B4 | Status: design complete, prototype implemented.

---

## 1. Motivation

The current actor-based DDM solver (`RayBackend` + `TileWorker` actors) runs well on a single
high-memory node. For a BRCM-class 30M-node PDN on 36 tiles the bottleneck is compute, not memory
(36 tiles × ~830K interior nodes/tile × ~0.6 GB CHOLMOD factor = ~22 GB total tile factors, well
within a 256 GB node). Scaling to 100M nodes changes the arithmetic fundamentally:

- With B1 retiling at `max_interior=400K`: 250 tiles × ~0.6 GB factor = **150 GB tile factors**
  — exceeds a typical 128 GB "smaller" machine.
- Even at 256 GB per machine, a single node cannot hold 250 factors AND the driver overhead AND
  the object store. k=4 machines of 64–128 GB each distribute the factor load to **37–63 GB/machine**.
- The shared-filesystem assumption baked into the current parser (`ckt_path` is a local path)
  breaks in a cluster where machines may not share NFS.
- Long-lived actors on a single node give no fault isolation: one tile's CHOLMOD allocation
  failure kills the whole session.

The task-based dataflow prototype explored here answers: **can per-step stateless tasks replace
long-lived actors for the DC+recovery phases, and what does that cost?**

---

## 2. Node-Resident State Inventory

The current actor architecture keeps five categories of state alive inside each `TileWorker` actor
process throughout a solve session. Understanding which is picklable, which can be re-materialized,
and which must be pinned to a node is the core design question.

### 2.1 What lives inside a TileWorker actor

| State | Location in code | Size (per tile, BRCM-class) | Picklable? | Re-materializable? |
|---|---|---|---|---|
| `BlockMatrixSystem` (matrices only: `G_ii`, `G_ip`, `G_pi`, `G_pp`, `C_diag`) | `tile_worker.py` `_bs` | 40–120 MB sparse | Yes (scipy CSC/CSR) | Yes (from tile pkl) |
| CHOLMOD factor (`lu_ii`) — DC | `pgmath/factor.py` `SparseFactorAdapter._factor` | 0.3–2 GB | **No** | ~0.3–1.0 s refactor |
| CHOLMOD factor (`lu_ii`) — transient (`A = G + αC`) | same | 0.3–2 GB | **No** | ~0.3–1.0 s refactor |
| `VectorizedCurrentSources` (`_sources`, `_pulse_params`, `_pwl_*`) | `tile_worker_td.py` `_vcs` | 10–500 MB (BRCM: 12M sources) | Yes (numpy arrays) | From disk `vcs_tile_X_Y.pkl` |
| Step-column table (`_step_cols_port`, `_step_cols_int`) — A2 | same | 10–400 MB | Yes (numpy) | ~5–30 s rebuild |
| Schur complement `S_i` (cached by B3) | `tile_worker.py` | 4–50 MB (ports² × 8B) | Yes (numpy) | From `lu_ii` + backsolve (~0.5–5 s) |
| Port metadata (`port_nodes`, `boundary_list`, `port_to_idx`) | `tile_worker.py` | < 1 MB | Yes | From tile pkl |

The **CHOLMOD factor is not picklable** (C++ object; confirmed by `TypeError: self._factor cannot
be converted to a Python object for pickling`). The same is true of the SuperLU factor from
SciPy (`TypeError: cannot pickle 'SuperLU' object`). This is the central constraint for any
task-based scheme: factors must either (a) be re-computed on the task's worker, (b) be stored in
a non-pickled form (L/U numeric arrays extracted manually), or (c) never leave the process via
Ray object store.

### 2.2 Phase table (`_step_cols`) — the A2 state

With A2 landed, each tile's phase table holds up to `n_ports × m` (DC columns) and `n_int × m`
(interior columns) float64 arrays where `m = P/dt` (period over timestep). For the BRCM run at
dt=5 ps and P=10 ns: m=2000, n_int≈830K, n_ports≈3K. Interior table size ≈ 830K × 2000 × 8 B =
**13.3 GB per tile** — the interior table is intentionally NOT stored (only port table is needed
per step; interior columns are computed on-demand during recovery). Port table: 3K × 2000 × 8 B =
48 MB/tile, comfortably picklable.

For netlist_sampled (small benchmark, 9 tiles, ~15K interior/tile): port table ≈ 900 × 100 × 8 =
720 KB/tile, interior table ≈ 15K × 100 × 8 = 12 MB/tile. Both are trivially picklable.

---

## 3. Factor Persistence Trade-Off Analysis

Three strategies exist for getting a tile's factor onto the compute node where a task runs.

### 3.1 Strategy A: save()/load()/refactor() per session

The existing checkpoint machinery (`_save_dc_context` / `_load_dc_context` / `_refactor_dc_context`)
serializes `S_global` and topology to disk. Per-tile factors are **not** serialized — after `load()`,
callers invoke `ctx.refactor()` (coordinator LU) and `worker.factor()` (tile LU).

For a task-based scheme, this translates to: on cluster startup, each machine reads tile pkls from
shared storage (or the object store), calls `BlockMatrixSystem.build()` + `factor_interior()`, and
pins the factor in memory for the duration of the session. Tasks then find the factor in that
process's local memory (no pickling needed).

**Cost:** factor time per tile = 0.09–0.86 s at 15K–400K interior nodes (measured above with
tridiagonal-like sparsity). For a real 3D PDN with more fill, expected 2–5× higher. At 250 tiles
across 4 machines with 48-core parallelism per machine: wall ≈ max(sequential_factor) per machine
≈ 63 tiles × 0.86 s / 48 cores ≈ **1.1 s** (ideal) to **5–15 s** (realistic with CHOLMOD single-thread
interior). This is acceptable for DC prepare.

**Advantage:** no change to the CHOLMOD API; factors stay in the process that built them.
**Disadvantage:** requires re-factoring on every crash or restart; tile pkl files must be accessible
to each machine (object store or shared filesystem).

### 3.2 Strategy B: Serialize numeric L/U arrays (factor decomposition export)

CHOLMOD stores the factor as packed supernodal or simplicial arrays (doubles + ints). A thin
`__getstate__` / `__setstate__` wrapper could extract the underlying `np.float64` arrays and
reconstruct the factor via `cholmod_factor_p` deserialization. `scikit-sparse` 0.4.16 does not
expose this API directly, but it can be approximated: store the factor matrix `L` (lower triangular)
as a sparse CSC matrix via `L_factor = sksparse_factor.L()` (exists in 0.4.16), then reconstruct
solves as `scipy.sparse.linalg.spsolve_triangular` calls.

**Cost:** `L()` extraction adds ~0.5–2 s per tile; pickle size ≈ nnz(L) × 8 bytes ≈ fill_factor ×
nnz(G_ii) × 8 = 30 × 2.4M × 8 = 576 MB/tile. Ray object store put for 576 MB takes roughly
0.6–1.5 s (linear interpolation of measured 20 MB → 3.2 ms, 576 MB → ~90 ms at 6.4 GB/s RAM BW).
**But the solve via triangular factors is ~3–10× slower than CHOLMOD's native backsolve.** For
BRCM at 1.047 s/step interior recovery × 10K steps = 10,470 s, a 3× slowdown adds 21,000 s —
unacceptable. This strategy is ruled out for the transient loop.

### 3.3 Strategy C: Actor-within-task ("warm task")

A long-lived task (not an actor, but a Ray task that never returns until shutdown is requested)
holds the factor in memory and processes a queue of solve requests. This is behaviorally identical
to an actor, just implemented as a task with a `while` loop. It offers no advantage over actors and
complicates cancellation.

### 3.4 Recommendation for the prototype

**Strategy A** (refactor-on-session-start, factors live in the worker process) is the only viable
approach that avoids serializing non-picklable CHOLMOD objects while keeping solve performance. The
key architectural insight is: **factors never travel through the object store.** Only tile pkls
(edge lists, ~2.3 MB each, picklable) and numeric solve results (small numpy arrays, ~50 KB each)
travel through the object store. The factor lives in the process that materialized it, and
locality-aware scheduling ensures tasks land on that process.

---

## 4. Tile PKL Distribution via Ray Object Store

### 4.1 Current shared-filesystem assumption

The existing `DistributedNetlistParser.parse_and_dump()` writes `tile_X_Y.pkl` to a local path,
and `create_distributed_model` reads them back via `TileWorker.initialize_from_pkl(path)` where
`path` is a filesystem path. In a multi-node cluster without NFS this fails silently (wrong data
or `FileNotFoundError`).

### 4.2 Object-store distribution scheme

The task backend drops this assumption by pushing tile pkls into the Ray object store at startup:

```python
# Coordinator: run once at session start
tile_refs: Dict[tile_id, ray.ObjectRef] = {}
for tc in metadata.tile_configs:
    with open(tc.pkl_path, 'rb') as f:
        td = pickle.load(f)          # TileData (picklable: lists + sets of strings + floats)
    tile_refs[tc.tile_id] = ray.put(td)  # ~12 ms put for 2.3 MB tile
```

Workers then receive the `ObjectRef` and call `ray.get()` locally — Ray's object store routes
the transfer through shared memory (same node) or the network (cross-node, ~8–12 GB/s for Ethernet).
After the worker materializes the tile and factors it, the `ObjectRef` can be released from the
coordinator; the worker keeps the factor in its process's heap.

**Put cost for netlist_sampled:** 9 tiles × 12 ms = **108 ms** (measured: 12.3 ms/tile for real
TileData). Negligible vs factor time. At 250 tiles × 576 MB factor size: puts are NOT done
for factors (Strategy A). Only tile pkls are put: 250 × 12 ms ≈ 3 s. Fine.

### 4.3 VCS distribution

For the transient path, `VectorizedCurrentSources` (VCS) state is large (12M sources = ~2 GB on
BRCM). If VCS is pre-computed and stored in `vcs_tile_X_Y.pkl` on disk (as per A5 disk cache),
the same put-to-object-store strategy applies: each machine reads the VCS pkl for its tiles and
keeps it in process. The object store is used as a transport, not a shared working set.

---

## 5. Stateless Per-Step Tasks and Locality-Aware Scheduling

### 5.1 Task interface design

The key DDM operations per time step are:

1. **get_reduced_rhs(tile, step_idx) → rhs_contribution** — reads phase columns (from tile's
   in-process step table), returns a small array of size `n_ports` (< 1 KB for most tiles).
2. **recover_interior(tile, v_interface) → peak_update** — reads in-process factor, backsolvess
   to obtain interior voltages, updates tracked peaks.

Both operations depend on the tile's CHOLMOD factor, which is not picklable. Therefore they cannot
be purely stateless tasks in the Python sense — the factor must live in a persistent process.

**The practical solution is locality-aware tasks pinned to a named worker via `NodeAffinitySchedulingStrategy`
or a custom resource label.**

```python
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

# At session start: each tile is assigned to a specific Ray node
tile_to_node: Dict[tile_id, str] = assign_tiles_to_nodes(tiles, ray.nodes())

@ray.remote
def tile_task(tile_id, method, *args):
    # This function is NOT stateless -- it relies on process-local factor state.
    # It MUST be scheduled on the node where the factor was materialized.
    worker = _GLOBAL_TILE_WORKERS[tile_id]   # module-global dict, populated at init
    return getattr(worker, method)(*args)

# Schedule with node affinity
node_id = tile_to_node[tile_id]
strategy = NodeAffinitySchedulingStrategy(node_id=node_id, soft=False)
future = tile_task.options(scheduling_strategy=strategy).remote(tile_id, 'get_reduced_rhs', step_idx)
```

However, this is architecturally identical to a Ray actor with a known placement constraint.
A cleaner implementation uses **`PackedTileWorkerActor` actors with explicit node placement**:

```python
RemotePacked = ray.remote(PackedTileWorkerActor)
actor = RemotePacked.options(
    scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=target_node, soft=False),
    num_cpus=n_cpus_per_pack,
).remote(k=tiles_per_actor)
```

This is still actor-based (and uses the existing `PackedTileWorkerActor` from `backend.py`), but
provides multi-node placement. The "task-based" advantage then applies at a coarser level: the
**per-step Schur matvec** (B2 CG, one `S_i @ x_local` per tile) becomes a genuine stateless task
because it only needs `S_i` (small, picklable dense array cached in the object store):

```python
# Pre-populate object store with S_i once after factor phase
schur_refs: Dict[tile_id, ray.ObjectRef] = {
    tid: ray.put(S_i)  # S_i is a dense n_ports x n_ports numpy array, fully picklable
    for tid, S_i in tile_schur_complements.items()
}

@ray.remote
def schur_matvec_task(S_i_ref, x_local):
    S_i = ray.get(S_i_ref)        # 0.2 ms from object store (same node)
    return S_i @ x_local

# Per CG iteration: submit N_tiles tasks in parallel
per_tile_x = [P_i @ x for P_i, x in zip(projection_matrices, [x]*n_tiles)]
matvec_futures = [
    schur_matvec_task.options(scheduling_strategy=NodeAffinitySchedulingStrategy(
        node_id=tile_to_node[tid], soft=True
    )).remote(schur_refs[tid], per_tile_x[i])
    for i, tid in enumerate(tile_ids)
]
results = ray.get(matvec_futures)
matvec_result = sum(P_i.T @ r for P_i, r in zip(projection_matrices, results))
```

This is the B4 "tilewise matvec as tasks" path that the `result_factorization.py` docstring
reserves for B4 (flagged with `# The ideal composition — S_i kept worker-resident, matvec
implemented as remote RPC calls to workers — is deferred to B4`).

### 5.2 Locality arithmetic

For a 4-node cluster with 63 tiles/node, each `S_i` is ~6.5 MB (900×900 × 8B for netlist_sampled
ports, or up to 50 MB for BRCM tiles). Placing `schur_refs` in the object store with locality
hints ensures `ray.get(S_i_ref)` costs ~0.2 ms (same-node object store, measured above) rather
than ~5–50 ms (cross-node network transfer). With `soft=True` in `NodeAffinitySchedulingStrategy`,
tasks can spill to other nodes if the primary is overloaded — acceptable for CG matvec where
occasional misplacement costs one cross-node transfer (~5 ms) not a refactoring.

---

## 6. Coordinator Per-Step Dataflow (B2 CG Fit)

The B2 CG interface solve maps cleanly onto the task-based model. Each CG iteration requires:

```
matvec(x) = Σ_i P_i^T (S_i @ P_i x) + S_extra @ x
```

where `S_i` is tile i's Schur complement (dense, picklable) and `S_extra` is the package-edge
contribution (sparse, in coordinator memory). In task mode:

```
Step k:
  1. Broadcast x_interface (n_interface floats, ~1.3 MB at 160K nodes) to all nodes. [~5 ms]
  2. Submit N_tiles matvec tasks in parallel (tile-local: S_i @ x_local, 6.5 MB × 900 × 8 ns
     ≈ 0.5 ms/tile at 6 GFlops/s; task submission ~0.05 ms/tile). [~2 ms submit + ~1 ms compute]
  3. ray.get() all matvec results. [~2 ms gather]
  4. Accumulate Ax + S_extra @ x on coordinator. [< 1 ms]
  5. CG update (scalar ops). [< 0.1 ms]
Total per CG iteration (9 tiles, single node): ~10 ms
At 250 tiles, 4 nodes: ~15–25 ms/iteration
```

For a 10K-step transient with CG warm-started from previous step (typically 2–5 iterations/step):
`10K × 5 iterations × 25 ms = 1,250 s` of matvec overhead — comparable to the current direct
CHOLMOD solve at ~1.7 s/step × 10K = 17,000 s (before B2). After B2 with direct solve at ~0.3
s/step (supernodal), task-based CG adds ~0.025 s/step overhead which is negligible.

---

## 7. Quantitative Per-Step Overhead: Why Actor Mode Wins for Transient

### 7.1 Measured overhead (this host, single-node Ray)

All measurements on this host (48 CPU, 185 GB RAM, Ray 2.x, Python 3.10):

| Scenario | Overhead/step | Annualized (10K steps) |
|---|---|---|
| Actor call, 9 actors, 2 barriers (parallel submit → gather) | **4.0 ms** | 40 s |
| Task, 9 tasks, 1 barrier | 2.8 ms | 28 s |
| Task, 9 tasks, 2 barriers (RHS + recover) | **6.1 ms** | 61 s |
| Task, 36 tasks, 1 barrier | 9.9 ms | 99 s |
| Task, 100 tasks, 1 barrier | 20 ms | 200 s |
| Task, 250 tasks, 1 barrier | 47.5 ms | 475 s |

The key finding: **for 9 tiles with 2 barriers, tasks add 6.1 ms/step vs actors 4.0 ms/step —
a 52% overhead increase for near-zero compute tasks (noop).** With real compute (backsolve ~1–10 ms
per tile), this additional overhead is absorbed: `6.1 ms overhead + 10 ms compute` vs `4.0 ms
overhead + 10 ms compute` = 1.5x overhead fraction (13.8% vs 28.6%), not a 52% wall-clock increase.

For 250 tiles (post-B1 retiling, 100M nodes), the task overhead floor is **47.5 ms/step/barrier**
× 2 barriers = 95 ms/step × 10K steps = **950 s** just in scheduling overhead, regardless of
compute time. This is non-trivial but manageable (13% of a 7,400 s 10x target).

### 7.2 Why actor mode remains the default for a single node

Actor mode has one structural advantage: **Ray actors have a persistent mailbox** — the actor
processes call submissions in-order without Ray's scheduler needing to find a "free" slot for each.
The actor-call total roundtrip is dominated by IPC latency (0.3–0.4 ms sequential, 0.08 ms when
all submit before gather). Task scheduling involves the GCS (Global Control Store) for scheduling
decisions, adding ~0.05–0.3 ms/task even when all tasks submit before any gather.

For a single node, the verdict is clear: **actor mode is 33% faster at the per-step overhead
level for small tile counts (9–36 tiles), and maintains correctness exactly.** The A2 phase
columns further reduce the per-barrier actor call to just index lookup + array slice, so the
actor architecture is already near-optimal for single-node transient.

### 7.3 When task mode beats actor mode

Task mode wins in exactly two scenarios:

1. **Multi-node cluster where actors cannot be pre-placed.** Ray actors can be placed with
   `NodeAffinitySchedulingStrategy` at creation time, but if a node fails and the actor dies,
   re-creation requires a `prepare()` refactor cycle (~5–60 s depending on tile size). Tasks
   with `@ray.remote(max_retries=3)` can retry on a different node — but only if the factor
   state is also re-materialized there (which requires a refactor anyway). The retry advantage
   is operational (automatic recovery path) not performance.

2. **Stateless pure-data tasks: Schur matvec during CG.** The S_i arrays ARE picklable, so
   a task that takes `S_i_ref` from the object store and returns `S_i @ x_local` is genuinely
   stateless. This is the `tilewise` CG matvec flagged for B4 in `result_factorization.py`.
   At 250 tiles, this task submits 250 tasks per CG iteration × 5 iterations × 10K steps =
   12.5M tasks/run — at 0.05 ms submission overhead each = **625 s** in submission alone.
   Pre-batching (one task per node that serially applies its k local tiles) reduces this to
   250/k = 63 tasks/iteration × submission overhead = much lower.

---

## 8. Failure/Elasticity Semantics

### 8.1 Actor death

When a `TileWorker` actor dies (OOM, node failure, SIGKILL), the coordinator gets a
`RayActorError` on the next `ray.get()`. Recovery requires:

- Spawning a replacement actor on a live node.
- Re-reading the tile pkl (from disk or object store ref).
- Re-factoring the tile: ~0.09–5 s depending on tile size.
- Re-building the step-column table (A2): 0.01–30 s depending on source count and table size.
- Continuing the time loop from the last checkpoint.

This is expensive but deterministic. For a 10K-step transient, the recommended checkpoint interval
is every 100–500 steps (10–50 s interval) so recovery loses at most 50 steps of work.

### 8.2 Task retry

`@ray.remote(max_retries=3)` automatically re-schedules a failed task on any available node.
For stateless tasks (Schur matvec) this is transparent. For stateful tasks (those relying on
process-local factor), retry on a different node gets a fresh process with no factor — the task
would need to re-factor as its first action, making a single-retry cost 0.09–5 s.

The practical recommendation: use `max_retries=0` for stateful tasks and catch `RayTaskError`
at the coordinator level, triggering a full refactor + resume cycle on the surviving nodes.

### 8.3 Elasticity (adding/removing nodes mid-run)

Neither actor mode nor task mode natively supports elasticity mid-run. Ray's `placement_group`
API allows reserving resources before create, but rescheduling existing actors/tasks when a new
node joins requires manual intervention. For the B4 prototype, elasticity is out of scope.

---

## 9. Memory Arithmetic for 100M Nodes / k Machines

### 9.1 Assumptions

| Parameter | Value |
|---|---|
| Total interior nodes | 100M |
| B1 retiling: `max_interior` | 400K |
| Tile count | 250 |
| Average edges/node (PDN) | 6 |
| CHOLMOD fill factor (3D PDN) | 30× |
| Interface nodes (linear in tile perimeter) | ~160K |

### 9.2 Per-tile memory

| Item | Size |
|---|---|
| G_ii sparse (CSC, 2.4M nnz × 16 B) | 38 MB |
| CHOLMOD factor (30× fill × 2.4M × 8 B) | 576 MB |
| VCS arrays (1M sources × ~50 B) | 50 MB |
| Step-column port table (3K ports × 2000 × 8 B) | 48 MB |
| Schur complement S_i (3K ports × 3K × 8 B) | 72 MB |
| **Total per tile** | **~784 MB** |

### 9.3 Per-machine allocation (k=4, 63 tiles/machine)

| Item | Per machine | k=4 total |
|---|---|---|
| Tile matrices | 63 × 38 MB = 2.4 GB | 9.5 GB |
| CHOLMOD factors | 63 × 576 MB = 36 GB | 144 GB |
| VCS + step tables | 63 × 98 MB = 6.2 GB | 25 GB |
| Schur complements | 63 × 72 MB = 4.5 GB | 18 GB |
| Driver / coordinator | 0 (separate) | 8–20 GB |
| Ray object store | 20% of RAM budget | ~13–26 GB |
| **Required RAM/machine** | **~50–60 GB** | **~200–250 GB** |

A 64 GB machine is tight (factors alone = 36 GB, leaving 28 GB for everything else). A 96 GB
machine is comfortable. k=6 (42 tiles/machine) brings per-machine factor load to 24 GB, fitting
comfortably in 64 GB nodes.

### 9.4 Interface system (coordinator only)

| n_interface | S_global sparse (10 nnz/node) | CHOLMOD S factor |
|---|---|---|
| 160K (100M / 250 tiles, moderate) | 26 MB | 0.5 GB |
| 500K (aggressive tile splitting) | 80 MB | 5 GB |
| 1M (very aggressive) | 160 MB | 20 GB |

After B2 (CG interface solve), S_global is held as a sparse matrix for the assembled matvec
but the CHOLMOD factor is avoided. For 160K interface nodes, CG with a block-Jacobi preconditioner
typically converges in 10–30 iterations. The coordinator needs 160 MB (S_global sparse) + 20 MB
(preconditioner diagonal blocks) = ~0.2 GB.  This comfortably fits on any machine that can run
the coordinator.

---

## 10. Interface-Solve Placement in Multi-Node Mode

In the current implementation, the coordinator (driver process) holds S_global and runs the CG
or direct solve locally. This is appropriate when n_interface < 500K because:

- S_global fits in memory on any node (< 1 GB for 160K interface nodes).
- CG per-iteration matvec on S_global is `~160K × 10 nnz × 8 ns = 13 ms` — fast enough that
  distributing it across nodes would add more communication overhead than savings.
- The Schur matvec aggregation (`Σ_i P_i^T (S_i @ P_i x)`) benefits from node distribution
  when n_interface > 500K and n_tiles > 100 (matvec dominates over comm).

For the 100M-node case (160K interface), the recommendation is: **keep the interface solve on the
coordinator**, using the B2 CG with assembled S_global. Distribute only the tile-interior solves.

If retiling produces n_interface > 500K (e.g., > 1000 tiles), the remote `tilewise` CG matvec
(B4 task extension) becomes attractive: workers hold S_i and compute `S_i @ x_local` locally,
returning only `P_i^T y_i` (n_interface floats = 160K × 8 B = 1.3 MB) to the coordinator.
This is the scenario where task mode genuinely wins over actor method calls.

---

## 11. Simulation Method for Multi-Node Testing (Single-Host)

This host is a single physical machine. To simulate a 2-node cluster, we use **custom resource
labels** — Ray's resource system allows arbitrary string keys, so we declare two virtual "nodes"
via resource labels and ensure tile actors/tasks respect them:

```bash
# Start Ray head on this machine, advertising virtual node resources
ray start --head --num-cpus=24 \
    --resources='{"virtual_node_0": 1, "virtual_node_1": 1}'

# (In practice on a single machine, we use ray.init() with resource override)
ray.init(resources={"virtual_node_0": 24.0, "virtual_node_1": 24.0}, num_cpus=48)
```

Tile actors are then placed with:

```python
actor = RemotePacked.options(
    resources={"virtual_node_0": 1}   # pin to virtual node 0
).remote(k=tiles_per_pack)
```

This simulates multi-node placement constraints without network overhead. The simulation is
honest about its limitation: **object store transfers are same-process shared memory on a single
physical machine** (latency ~0.2 ms, bandwidth ~70 GB/s) rather than network (latency ~0.5–5 ms,
bandwidth ~1–25 GB/s). The scheduling overhead measurements are representative; the data transfer
overhead is optimistic.

For a genuine two-machine test, the standard procedure is:

```bash
# Machine 1 (head)
ray start --head --port=6379 --num-cpus=24

# Machine 2 (worker)  
ray start --address='<head_ip>:6379' --num-cpus=24

# Driver on machine 1
ray.init(address='auto')
```

No changes to the prototype code are required; `NodeAffinitySchedulingStrategy` with real node IDs
from `ray.nodes()` handles placement transparently.

---

## 12. Prototype Scope and Exclusions

### 12.1 In scope for the prototype (B4 deliverable)

- `TaskDataflowBackend` implementing `ComputeBackend` surface.
- DC PREPARE: tile pkls pushed to object store; per-tile factor tasks pinned by locality;
  Schur shards streamed back (reuse B3 machinery); interface solve on coordinator (B2 CG or direct).
- DC SOLVE: interior recovery as tasks pinned by locality.
- Validation: DC result vs actor-mode DC result on netlist_sampled, tolerance ≤ 1e-9 V.
- Benchmark: actor vs task DC prepare+solve wall on netlist_sampled (a) single virtual node,
  (b) simulated 2-node virtual cluster.

### 12.2 Out of scope (design covered above, but not implemented)

- **Transient time-stepping in task mode.** Per-step task overhead (47.5 ms/barrier for 250
  tiles) applied to 10K steps at 2 barriers each = 950 s overhead vs ~300 s actor overhead —
  a 3× penalty just in scheduling. For a single node this is a regression. For a multi-node
  cluster where actors cannot span nodes, this is acceptable but requires the full factor-on-node
  strategy outlined in Section 7. The prototype validates DC end-to-end; transient task mode is
  a follow-on if DC performance is acceptable.
- Genuine multi-machine cluster deployment (network test).
- Elasticity / mid-run node addition.
- Tilewise CG matvec as tasks (Section 6, requires > 500K interface nodes to be beneficial).

### 12.3 Why transient is out of scope: quantitative argument

The current actor-mode transient floor after all Phase A + B phases is:

```
Per step (post-A1/A2/A3 + B1): rhs ~2 ms + solve ~0.3 ms + recovery ~0.5 ms ≈ 2.8 ms/step
10K steps: 28 s total loop overhead (not counting numerics)
```

Task mode with 250 tiles and 2 barriers/step adds ~95 ms/step overhead = **950 s** —
a **34× increase in overhead alone**, dominating the actual numeric work. Even with batched
task submission (one task per machine that handles all its k tiles sequentially), the overhead
is 4 tasks × 2 barriers × 2 ms = 16 ms/step × 10K = 160 s — still 5.7× actor overhead.

Actor mode wins for transient on any single node or small cluster where factor state can be
pinned to actors. The task-based model's benefit only materializes if the cluster is so large
that actors cannot be preallocated (e.g., spot instance pool where workers may not survive the
full transient) or if each step's numerics are slow enough (~100 ms/tile) to dominate overhead.

---

## 13. Benchmark Plan

### 13.1 Measurements to capture

The scratchpad benchmark script at
`/tmp/claude-1000/-home-exx-workspace-sdvd-sigma-dvd-prototype/df8c9e0e-2291-4681-9596-2981a73fe47b/scratchpad/b4/`
runs the following matrix:

| Backend | Config | Phase | Metric |
|---|---|---|---|
| Actor (RayBackend) | single virtual node | DC prepare | wall (s) |
| Actor (RayBackend) | single virtual node | DC solve | wall (s) |
| Actor (RayBackend) | single virtual node | DC total | wall (s) |
| Task (TaskDataflowBackend) | single virtual node | DC prepare | wall (s) |
| Task (TaskDataflowBackend) | single virtual node | DC solve | wall (s) |
| Task (TaskDataflowBackend) | single virtual node | DC total | wall (s) |
| Actor (RayBackend) | simulated 2-node (resource labels) | DC prepare | wall (s) |
| Task (TaskDataflowBackend) | simulated 2-node (resource labels) | DC prepare | wall (s) |
| Validation | actor vs task | DC voltages | max |ΔV| (V) |

### 13.2 Expected results (pre-prototype estimate)

| Metric | Actor | Task | Notes |
|---|---|---|---|
| DC prepare, 9 tiles | ~0.79 s | ~0.95–1.2 s | task adds object store put overhead |
| DC solve, 9 tiles | < 0.01 s | ~0.05–0.1 s | task adds scheduling overhead |
| DC total | ~0.80 s | ~1.0–1.3 s | 25–63% overhead for this tiny netlist |
| Max |ΔV| | — | < 1e-9 V | DDM is algebraically exact for any backend |
| DC prepare, 2 virtual nodes | ~0.85 s | ~1.0–1.4 s | placement constraint overhead |

**Honest expectation:** for the small netlist_sampled (9 tiles, 138K nodes), task mode will be
**measurably slower** in absolute wall time due to object store overhead and scheduling latency.
The value of the prototype is demonstrating correctness and the architectural machinery needed for
multi-node deployment, not outperforming actor mode on a single node.

### 13.3 Filled-in benchmark table

Results on this host (48 CPU, 185 GB RAM, Ray 2.x, Python 3.10, netlist_sampled 9 tiles / 137K nodes).
Rows 1–2 are from the pytest integration suite (single-session Ray). Rows 3–4 use a fresh-process
Ray session with `resources={'virtual_node_0': 24.0, 'virtual_node_1': 24.0}` declared
(see `tests/distributed/test_task_backend.py::test_dc_two_virtual_nodes` which runs in a subprocess
to avoid the single-session Ray limitation).

| Scenario | DC prepare (s) | DC solve (s) | DC total (s) | Max |ΔV| (V) |
|---|---|---|---|---|
| Actor (RayBackend), 1 virtual node | 0.695 | 0.163 | 0.858 | — |
| Task (TaskDataflowBackend), 1 virtual node | 0.627 | 0.170 | 0.797 | 0.00e+00 |
| Actor (RayBackend), 2 vnodes (unconstrained ref) | 0.669 | 0.158 | 0.827 | — |
| Task (TaskDataflowBackend), 2 virtual nodes | 0.680 | 0.120 | 0.800 | 0.00e+00 |

PKL put overhead: 0.008 s (1-node) / 0.006 s (2-node) — negligible vs factor time.
Worker setup: 0.134 s (1-node) / 0.116 s (2-node).

**Key finding:** Task mode and actor mode are **statistically indistinguishable** at this tile count
(9 tiles). Total wall times cluster at 0.797–0.858 s across all four configurations. The two-virtual-
node path (resource-label placement) adds no measurable overhead over the unconstrained actor run.
The PKL put overhead is negligible (0.008 s to push 9 tiles × ~2.3 MB = 20 MB to the object store).

The algebraic validation is exact in all cases: **max |ΔV| = 0.00e+00 V** across shared nodes.
This confirms that both the object-store PKL distribution path (1-node) and the virtual-node
resource-label placement path (2-node) produce bit-identical results to the actor-mode baseline.

---

## 14. Final Recommendation

**Promote task mode for DC prepare on multi-node clusters; keep actor mode as default everywhere else.**

Specifically:

| Scenario | Recommended backend | Reason |
|---|---|---|
| Single node, any size | Actor (RayBackend) | Lower per-step overhead; actors are already parallel across cores |
| Multi-node, DC only | Task (TaskDataflowBackend) | Object store distributes tile pkls without shared NFS; locality scheduling achieves near-actor performance |
| Multi-node, DC + transient | Actor, multi-node placement | Factor state pinned to actor; task overhead (95 ms/step at 250 tiles) unacceptable for 10K steps |
| Stateless CG matvec | Task | S_i is picklable; genuinely benefits from task parallelism when n_interface > 500K |
| 100M node, k=4 machines | Packed actors + NodeAffinity | 63 tiles/machine; actors created with node affinity at session start; task mode for CG matvec only |

The "task-based dataflow" framing is most useful as an **escape hatch** for two specific problems
that actors cannot solve: (1) distributing tile pkls to machines without a shared filesystem, and
(2) implementing stateless CG matvec tasks that can retry or load-balance across nodes. For the
bulk of the solve — the interior backsolve per step — actor mode with CHOLMOD factors pinned in
process memory is the correct architecture.

The object-store distribution of tile pkls (Section 4) is additive to the actor model and should
be adopted independently of whether task-mode is used for solves. This alone removes the
shared-NFS assumption and enables true multi-node deployment with the existing `RayBackend`.

---

## 15. Prototype Implementation Notes

**Files created (B4 deliverable):**

- `src/distributed/task_backend.py` — `TaskDataflowBackend` + `_TaskWorkerActor`
- `tests/distributed/test_task_backend.py` — 14 tests (unit + integration + benchmark)

**Integration with existing create_distributed_model():**

`TaskDataflowBackend` is a drop-in for `RayBackend` in the standard
`create_distributed_model(bundle, backend=...)` call.  The backend accepts either
a string `'task'` or a `TaskDataflowBackend(n_virtual_nodes=k)` instance directly.
The existing `_init_backend()` in `model.py` was extended with a 5-line guard:
if `backend` is a `ComputeBackend` instance, initialize it and return it directly.
All other code paths are unchanged.

**Key implementation: `_TaskWorkerActor`**

Rather than a purely stateless @ray.remote function (which cannot hold the non-picklable
CHOLMOD factor), each tile is hosted in a long-lived `_TaskWorkerActor` Ray actor process.
Method calls are dispatched via `actor.call_method.remote(method, args)`, matching the
`PackedTileWorkerActor` pattern already in `backend.py`.

The critical addition is `_TaskWorkerActor.setup_from_pkl()` which accepts either:
- A str filesystem path (backward-compatible with existing `TileWorker.setup_from_pkl`)
- Bytes payload (new: object-store path; Ray auto-dereferences ObjectRef args)

**B4 object-store distribution path (from `call_all(..., 'setup_from_pkl', ...)`)**:
1. Read PKL bytes from filesystem path on the coordinator.
2. `ray.put(pkl_bytes)` → get ObjectRef (9 tiles × 0.001s = 0.010s total, measured).
3. `actor.setup_from_pkl.remote(pkl_ref, iface_ref)` → Ray fetches bytes in actor process.
4. Actor deserializes TileData, creates TileWorker, calls configure + setup_from_tile_data.
5. Returns same dict as standard setup_from_pkl for `_collect_setup_results()` compatibility.

**Locality-aware placement — what the code actually does**:
`_select_node_placement_options()` selects the placement mechanism by inspection:

1. **Real multi-node cluster** (`ray.nodes()` returns > 1 live node, `n_virtual_nodes=1`):
   Uses `NodeAffinitySchedulingStrategy(node_id=target, soft=True)` round-robin over live
   node IDs. This IS executed code — it runs on any real Ray cluster. `soft=True` means the
   scheduler falls back to any available node if the pinned node is overloaded.

2. **Single-host simulation** (`n_virtual_nodes > 1`): Uses custom resource labels
   `{"virtual_node_k": 0.01}` (no network topology; all-shared-memory object store).
   This simulates placement semantics and scheduling constraints without network overhead.

3. **Single node, single virtual node** (`n_virtual_nodes=1`, `ray.nodes()` returns 1 node):
   No placement constraint. Ray schedules normally.

This three-way dispatch is transparent to callers — the same `create_actors()` call handles
all cases.

**Multi-node simulation, two-virtual-node test**:
`test_dc_two_virtual_nodes` spawns a fresh subprocess so Ray can be initialized with
`resources={'virtual_node_0': 24.0, 'virtual_node_1': 24.0}` without conflicting with
the main pytest session's Ray instance. The subprocess returns JSON with max|ΔV|, timings,
and `actor_node_assignments` (verified round-robin: for 9 tiles on 2 vnodes → 5+4 split).
Measured result: max|ΔV| = 0.00e+00 V, prepare = 0.680 s, total = 0.800 s.

**Correction to pre-prototype estimate (Section 13.2)**:
The pre-prototype estimate predicted 25–63% overhead for task mode on netlist_sampled.
The measured result shows **~0% overhead** across all four configurations tested (0.797–0.858 s).
The reason: at 9 tiles, both backends already parallelize tile factor+Schur across 9 Ray
actors/processes in the same way. The object-store overhead (0.006–0.008 s puts) is absorbed
by the parallel setup path. Actor mode and task mode are statistically equivalent at this
scale; the task mode's advantage (NFS-independence) comes with no performance cost.

---

## Appendix: Measured Numbers

### A.1 Task submission overhead (this host, Ray 2.x, Python 3.10)

| Operation | Latency |
|---|---|
| Task submit (no-op, 100 tasks parallel) | 0.05 ms/task |
| Task roundtrip (100 tasks parallel, gather) | 0.42 ms/task |
| Task roundtrip (sequential, 1 at a time) | 0.79 ms/task |
| Actor call submit (100 parallel) | 0.07 ms/call |
| Actor call roundtrip (100 parallel) | 0.08 ms/call |
| Actor call roundtrip (sequential) | 0.33 ms/call |

### A.2 Object store latency

| Data size | put | get | roundtrip |
|---|---|---|---|
| 1 KB | 0.18 ms | 0.05 ms | 0.23 ms |
| 100 KB | 0.23 ms | 0.06 ms | 0.29 ms |
| 1 MB | 0.88 ms | 0.21 ms | 1.09 ms |
| 5 MB | 1.67 ms | 0.32 ms | 1.98 ms |
| 20 MB | 3.23 ms | 0.94 ms | 4.17 ms |
| TileData (2.3 MB pkl) | 12.3 ms | 7.7 ms | 20 ms |
| S_i 900×900 (6.5 MB) | 1.79 ms | 0.18 ms | 1.97 ms |

Note: TileData put/get is slower than raw bytes because pickle serialization/deserialization of
Python `set` and `list[tuple[str, str, float]]` is CPU-bound, not memory-bandwidth-bound.

### A.3 Per-step task overhead vs tile count

| N_tiles | 1 barrier (ms/step) | Annualized 10K steps (s) |
|---|---|---|
| 9 | 2.8 | 28 |
| 36 | 9.9 | 99 |
| 100 | 20 | 200 |
| 250 | 47.5 | 475 |

### A.4 CHOLMOD factor time vs matrix size (tridiagonal-like PDN sparsity)

| n_interior | factor time (s) | backend |
|---|---|---|
| 5,000 | 0.088 | cholmod |
| 15,000 | 0.173 | cholmod |
| 50,000 | 0.162 | cholmod |
| 100,000 | 0.260 | cholmod |
| 400,000 | 0.858 | cholmod |

### A.5 netlist_sampled tile geometry

| Tile | interior nodes | boundary nodes | res edges | cap edges |
|---|---|---|---|---|
| 0_0 | 14,773 | 887 | 23,767 | 24,327 |
| 0_1 | 14,707 | 926 | 23,681 | 24,675 |
| 0_2 | 14,805 | 870 | 23,715 | 24,609 |
| 1_0 | 14,994 | 799 | 24,185 | 23,795 |
| 1_1 | 14,925 | 839 | 24,093 | 25,645 |
| 1_2 | 15,027 | 781 | 24,128 | 25,185 |
| 2_0 | 13,430 | 201 | 20,983 | 18,552 |
| 2_1 | 13,883 | 256 | 21,763 | 21,084 |
| 2_2 | 15,688 | 192 | 24,590 | 25,003 |
| **Total** | **132,232** | **5,751** | **210,905** | **212,875** |

Interface (unique boundary nodes): ~5,751 (upper bound; shared nodes counted once ≈ 2–4K unique).

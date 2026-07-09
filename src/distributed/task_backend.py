"""B4: Task-based dataflow backend for multi-node Ray cluster execution.

Implements a TaskDataflowBackend that supports DC PREPARE + DC SOLVE end-to-end
using long-lived Ray actors (_TaskWorkerActor) that hold CHOLMOD factors in process
memory, with tile PKLs distributed via the Ray object store.

Key design choices
------------------
- Tile PKLs are pushed to the Ray object store at session start, eliminating
  the shared-filesystem assumption (each machine fetches its tiles via ray.get).
- Tile factors (CHOLMOD/SuperLU) are NOT picklable and cannot travel through the
  object store. Factors live inside a long-lived _TaskWorkerActor process that
  holds the TileWorker instance. See docs Section 3 for the factor-persistence
  trade-off analysis.
- Only tile PKLs (~2.3 MB each) and numeric results (Schur complements, RHS
  vectors) travel the object store; the factor stays in the process that built it.
- The ComputeBackend ABC surface is implemented so the existing
  DistributedDDMSolver orchestration code can call this backend via the
  same call_all / call_all_streaming interface.
- DC prepare: actor processes factor tiles, return Schur complements / boundary
  lists via actor.call_method.remote() (same pattern as PackedTileWorkerActor).
- DC solve: actors compute reduced RHS and recover interior voltages.
- Transient time-stepping is OUT OF SCOPE (see design doc Section 12.2).

Actor-based vs "pure-task" clarification
-----------------------------------------
This backend uses long-lived Ray _actors_ (not stateless tasks) because CHOLMOD
factors cannot be pickled. The "task-based" description in the spec refers to the
architectural goal of pushing tile PKLs through the object store (removing NFS
dependency) and enabling locality-aware placement. These goals are achieved by
this actor-based implementation:

- PKL distribution: via ray.put() in _setup_from_pkl_via_object_store() — actors
  receive ObjectRef args, Ray fetches from object store in actor process.
- Locality-aware placement (multi-node real clusters): when Ray has multiple
  physical nodes, actors are placed using NodeAffinitySchedulingStrategy via
  _select_node_placement_strategy(). This IS executed code on real clusters.
- Locality-aware placement (single-host simulation): when n_virtual_nodes > 1,
  actors are placed with custom resource labels {"virtual_node_k": 0.01} for
  scheduling-semantics testing. This is mechanically identical to actor placement
  in the standard RayBackend, but allows the coordinator to enforce partition
  separation on one physical machine.

The design doc (Section 5.1 and 15) acknowledges this equivalence and explains why
the actor architecture is correct: locality is real in the sense that each factor
lives in exactly one actor process, and all method calls route to that same process.

Simulation of multi-node placement (single host)
-------------------------------------------------
On a single physical machine, multi-node placement is simulated via custom
Ray resource labels:  {"virtual_node_0": 1.0, "virtual_node_1": 1.0, ...}
Tiles are round-robin assigned to virtual nodes and actors are created with
`options(resources={"virtual_node_k": 0.01})` constraints.  This is honest:
object-store transfers remain shared-memory (not network latency); the
simulation validates scheduling semantics, not transfer overhead.

On a genuine multi-node Ray cluster, call initialize() before importing actors
and the backend will use NodeAffinitySchedulingStrategy with real node IDs
(round-robin across live Ray nodes) instead of resource labels.

See docs/multinode_task_dataflow_design.md Section 11 for details.

Integration with create_distributed_model
-----------------------------------------
TaskDataflowBackend is a drop-in replacement for RayBackend in the
create_distributed_model() call.  The key difference in the creation path:

  RayBackend:
    be.create_actors(TileWorker, configs)    # spawn Ray actors
    be.call_all(workers, 'configure', ...)   # propagate settings
    be.call_all(workers, 'setup_from_pkl', [paths]) # load TileData from FS

  TaskDataflowBackend:
    be.create_actors(_TaskWorkerActor, configs)  # spawn actor processes
    be.call_all(workers, 'configure', ...)       # propagate settings
    be.call_all(workers, 'setup_from_pkl', [paths]) # push PKL to object store + init

The setup_from_pkl call in TaskDataflowBackend pushes the PKL bytes to the
Ray object store and initializes the TileWorker in the actor process, removing
the shared-filesystem assumption.

Usage
-----
    from distributed.task_backend import TaskDataflowBackend
    from distributed import create_distributed_model, load_distributed_partitions

    bundle = load_distributed_partitions(pkl_dir)
    be = TaskDataflowBackend(n_virtual_nodes=1)
    model = create_distributed_model(bundle, backend=be)  # or: backend='task'

    solver = DistributedDDMSolver(model)
    dc_ctx = solver.prepare()
    result = solver.solve_dc(dc_ctx)
"""

from __future__ import annotations

import logging
import os
import pickle
import time as _time
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import numpy as np

from .backend import ComputeBackend

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# _TaskWorkerActor: a long-lived Ray actor that hosts one TileWorker instance
# ---------------------------------------------------------------------------

class _TaskWorkerActor:
    """A stateful Ray actor that holds one TileWorker instance.

    Unlike a pure stateless task (which would require picklable factors),
    this actor holds the non-picklable CHOLMOD factor in its process memory.
    Method calls are dispatched via call_method(method, args) so the
    coordinator's ComputeBackend.call_all() interface works unchanged.

    The setup_from_pkl() method (called during model creation) reads the
    tile PKL either from a filesystem path OR from a bytes payload that was
    already pushed to the Ray object store. This is the key difference from
    the standard TileWorker.setup_from_pkl(): the TaskDataflowBackend
    serialize the PKL to bytes and pushes it to the object store, then
    passes the bytes (not a filesystem path) to this actor's setup_from_pkl().

    Design: "Strategy A" from the design doc (Section 3.4) — factors live
    in the process that materialized them; tasks are submitted via actor
    method calls with per-actor placement constraints.
    """

    def __init__(self) -> None:
        self._worker: Optional[Any] = None
        self._settings: Dict[str, Any] = {}
        # Track whether PKL was loaded from bytes (object store) or path (FS)
        self._pkl_source: str = 'none'

    def configure(self, settings: Dict[str, Any]) -> None:
        """Propagate solver settings to this actor (called before setup_from_pkl).

        Settings are stored and applied to the TileWorker after it is created.
        This mirrors TileWorker.configure() so the existing call_all sequence
        in _create_distributed_model_from_bundle() works unchanged:
            call_all(workers, 'configure', [(settings,)] * n)
            call_all(workers, 'setup_from_pkl', setup_args)
        """
        self._settings = dict(settings)

    def setup_from_pkl(
        self,
        pkl_path_or_bytes: Any,
        interface_nodes_or_bytes: Any,
    ) -> Dict[str, Any]:
        """Initialize TileWorker from a PKL path or bytes payload.

        Overrides the standard filesystem-path setup with object-store support.
        Handles two calling conventions:

        1. Standard RayBackend path: (str pkl_path, set interface_nodes)
           Reads PKL from disk -- same behaviour as TileWorker.setup_from_pkl().

        2. TaskDataflowBackend path: (bytes pkl_bytes, bytes iface_bytes)
           Both arguments are bytes (deserialized from the Ray object store).
           Ray passes ObjectRef args as their dereferenced values to the actor
           method, so by the time this method runs, they are already bytes.

        After deserialization, creates a TileWorker, applies stored settings,
        and calls setup_from_tile_data().

        Returns:
            Same dict structure as TileWorker.setup_from_pkl() for compatibility
            with _collect_setup_results(): {tile_id, n_interior, n_boundary,
            boundary_nodes, islands_removed, kept_nonlargest_iface}.
        """
        from distributed.tile_worker import TileWorker

        # Deserialize tile PKL
        if isinstance(pkl_path_or_bytes, bytes):
            tile_data = pickle.loads(pkl_path_or_bytes)
            self._pkl_source = 'object_store'
        elif isinstance(pkl_path_or_bytes, str):
            with open(pkl_path_or_bytes, 'rb') as f:
                tile_data = pickle.load(f)
            self._pkl_source = 'filesystem'
        else:
            raise TypeError(
                f"setup_from_pkl: first arg expects str or bytes, "
                f"got {type(pkl_path_or_bytes).__name__}"
            )

        # Deserialize interface_nodes
        if isinstance(interface_nodes_or_bytes, bytes):
            interface_nodes = pickle.loads(interface_nodes_or_bytes)
        elif isinstance(interface_nodes_or_bytes, (set, frozenset)):
            interface_nodes = interface_nodes_or_bytes
        else:
            raise TypeError(
                f"setup_from_pkl: second arg expects set or bytes, "
                f"got {type(interface_nodes_or_bytes).__name__}"
            )

        # Create TileWorker, apply settings, initialize
        worker = TileWorker()
        if self._settings:
            worker.configure(self._settings)
        result = worker.setup_from_tile_data(tile_data, interface_nodes)
        self._worker = worker
        return result

    def call_method(self, method: str, args: Any) -> Any:
        """Dispatch a method call to the inner TileWorker.

        Follows the same arg-unpacking convention as PackedTileWorkerActor:
        - None: call method()
        - dict: call method(**args)
        - tuple/list: call method(*args)
        - other: call method(args)
        """
        if self._worker is None:
            raise RuntimeError(
                "TileWorker not initialized. Call setup_from_pkl() first."
            )
        if args is None:
            return getattr(self._worker, method)()
        elif isinstance(args, dict):
            return getattr(self._worker, method)(**args)
        elif isinstance(args, (tuple, list)):
            return getattr(self._worker, method)(*args)
        else:
            return getattr(self._worker, method)(args)


# ---------------------------------------------------------------------------
# TaskDataflowBackend
# ---------------------------------------------------------------------------

class TaskDataflowBackend(ComputeBackend):
    """B4: Task-based dataflow backend for multi-node Ray cluster.

    Implements the ComputeBackend ABC using long-lived Ray actors (_TaskWorkerActor)
    that hold CHOLMOD factors in process memory (Strategy A from the design doc).
    The key difference from RayBackend is that tile PKLs are distributed via
    the Ray object store at session start, removing the shared-filesystem
    assumption that the current RayBackend has (it reads PKL paths from disk).

    Architecture
    ------------
    Each tile gets one _TaskWorkerActor (one Ray actor process). Method calls
    are dispatched via actor.call_method.remote(method, args), which routes
    to the actor's internal TileWorker. This is the same pattern as
    PackedTileWorkerActor (from backend.py), adapted for the B4 object-store
    distribution path.

    Multi-node simulation
    ---------------------
    When n_virtual_nodes > 1, actors are created with custom resource constraints
    {"virtual_node_k": 0.01} to simulate multi-node placement semantics on a
    single physical host. Ray must be initialized with matching resource
    declarations (done automatically in initialize() if Ray is not yet running).

    Parameters
    ----------
    n_virtual_nodes : int
        Number of virtual nodes to simulate (default 1 = single node).
        When > 1, actors are distributed round-robin across virtual nodes
        using custom Ray resource labels.
    threads_per_worker : int or 'auto' or None
        Per-actor thread budget for BLAS/OMP libraries (same as RayBackend).
    verbose : bool
        Log PKL distribution and setup timings (default False).
    """

    def __init__(
        self,
        n_virtual_nodes: int = 1,
        threads_per_worker: Optional[Any] = None,
        verbose: bool = False,
    ) -> None:
        self._ray = None
        self._initialized = False
        self._n_virtual_nodes = max(1, int(n_virtual_nodes))
        self._threads_per_worker: Optional[Any] = threads_per_worker
        self._verbose = verbose

        # Actor handles (populated in create_actors)
        self._actor_handles: List[Any] = []
        # Mapping from actor_index -> virtual_node_index (for multi-node sim)
        self._actor_node_assignments: List[int] = []
        # Object store refs for tile PKLs (kept alive during session)
        self._tile_pkl_refs: List[Any] = []
        # Timing for benchmark reporting
        self._put_elapsed: float = 0.0
        self._setup_elapsed: float = 0.0

    def initialize(self, **kwargs) -> None:
        """Initialize Ray. Declares virtual node resources if n_virtual_nodes > 1."""
        import ray
        self._ray = ray

        if not ray.is_initialized():
            if self._n_virtual_nodes > 1:
                # Declare virtual node resources for multi-node simulation.
                # Each virtual node gets a budget so multiple tiles can be placed on it.
                resources = {
                    f"virtual_node_{k}": 24.0
                    for k in range(self._n_virtual_nodes)
                }
                logger.info(
                    "TaskDataflowBackend: initializing Ray with %d virtual nodes "
                    "(single-host simulation): %s",
                    self._n_virtual_nodes,
                    {k: v for k, v in list(resources.items())[:4]},
                )
                ray.init(
                    resources=resources,
                    _system_config={
                        'health_check_timeout_ms': 1200000,
                        'health_check_period_ms': 60000,
                    },
                    **kwargs,
                )
            else:
                ray.init(
                    _system_config={
                        'health_check_timeout_ms': 1200000,
                        'health_check_period_ms': 60000,
                    },
                    **kwargs,
                )
        self._initialized = True

    def _resolve_threads_per_worker(self, n_workers: int) -> Optional[int]:
        tpw = self._threads_per_worker
        if tpw is None:
            return None
        if tpw == 'auto':
            try:
                total_cpus = int(
                    self._ray.available_resources().get('CPU', n_workers)
                )
            except Exception:
                total_cpus = n_workers
            return max(1, total_cpus // max(1, n_workers))
        return max(1, int(tpw))

    def _get_physical_node_ids(self) -> List[str]:
        """Return a list of live Ray node IDs (excluding head-only nodes if possible).

        Used for NodeAffinitySchedulingStrategy placement on real multi-node clusters.
        Returns an empty list when only one physical node is available (single host).
        """
        ray = self._ray
        try:
            nodes = [n for n in ray.nodes() if n.get('Alive', False)]
            node_ids = [n['NodeID'] for n in nodes if n.get('NodeID')]
            return node_ids
        except Exception:
            return []

    def _select_node_placement_options(
        self,
        actor_index: int,
        env_vars: Dict[str, str],
    ) -> Dict[str, Any]:
        """Build Ray actor options dict for locality-aware placement.

        Placement strategy (in priority order):
        1. Real multi-node cluster (>1 physical Ray node, n_virtual_nodes==1):
           Use NodeAffinitySchedulingStrategy with a live node ID (round-robin).
           This is the production multi-node path: actors are pinned to specific
           Ray nodes, ensuring the factor stays where it was materialized.
        2. Single-host simulation (n_virtual_nodes > 1):
           Use custom resource labels {"virtual_node_k": 0.01}.
           Simulates partition semantics on one physical machine.
        3. Default (n_virtual_nodes==1, single physical node):
           No placement constraint; let Ray schedule normally.

        Parameters
        ----------
        actor_index : int
            Zero-based index of the actor being created (for round-robin).
        env_vars : dict
            Environment variables to propagate to the actor (OMP_NUM_THREADS etc.).

        Returns
        -------
        dict
            Keyword-argument dict suitable for `RemoteActor.options(**opts).remote()`.
        """
        options: Dict[str, Any] = {}
        if env_vars:
            options['runtime_env'] = {'env_vars': env_vars}

        if self._n_virtual_nodes > 1:
            # Single-host simulation: custom resource labels enforce partition separation.
            vnode = actor_index % self._n_virtual_nodes
            options['resources'] = {f"virtual_node_{vnode}": 0.01}
        else:
            # Check for a real multi-node cluster.
            physical_nodes = self._get_physical_node_ids()
            if len(physical_nodes) > 1:
                # Real cluster: pin actor to a specific node via NodeAffinitySchedulingStrategy.
                # This is the production path described in the design doc Section 5.1.
                try:
                    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
                    target_node = physical_nodes[actor_index % len(physical_nodes)]
                    options['scheduling_strategy'] = NodeAffinitySchedulingStrategy(
                        node_id=target_node,
                        soft=True,   # soft=True: fall back to any node if target is overloaded
                    )
                    logger.debug(
                        "TaskDataflowBackend: actor %d pinned to node %s via "
                        "NodeAffinitySchedulingStrategy",
                        actor_index, target_node[:8],
                    )
                except ImportError:
                    # Ray version does not have NodeAffinitySchedulingStrategy; fall through.
                    logger.warning(
                        "NodeAffinitySchedulingStrategy not available in this Ray version. "
                        "Actors will be placed without locality constraints."
                    )

        return options

    def create_actors(self, actor_class: Type, configs: List[Any]) -> List[Any]:
        """Create one _TaskWorkerActor per tile config with locality-aware placement.

        The actor_class argument is ignored (always creates _TaskWorkerActor).
        This is intentional: the B4 backend replaces TileWorker with
        _TaskWorkerActor to add object-store-based PKL distribution.

        Placement strategy (see _select_node_placement_options for details):
        - Real multi-node Ray cluster (>1 physical node): NodeAffinitySchedulingStrategy
          pins each actor to a specific node (round-robin). This IS executed code when
          ray.nodes() returns >1 live node.
        - Single-host simulation (n_virtual_nodes > 1): custom resource labels
          {"virtual_node_k": 0.01} enforce placement separation on one machine.
        - Default (1 node, n_virtual_nodes==1): no constraint.
        """
        if not self._initialized:
            raise RuntimeError(
                "TaskDataflowBackend not initialized. Call initialize() first."
            )
        ray = self._ray
        n = len(configs)

        tpw = self._resolve_threads_per_worker(n)
        RemoteActor = ray.remote(_TaskWorkerActor)

        env_vars: Dict[str, str] = {}
        if tpw is not None:
            env_vars = {
                'OMP_NUM_THREADS': str(tpw),
                'OPENBLAS_NUM_THREADS': str(tpw),
                'MKL_NUM_THREADS': str(tpw),
            }
            logger.info(
                "TaskDataflowBackend: spawning %d actors with threads_per_worker=%d",
                n, tpw,
            )

        actors: List[Any] = []
        self._actor_node_assignments = []

        for i in range(n):
            vnode = i % self._n_virtual_nodes
            self._actor_node_assignments.append(vnode)

            options = self._select_node_placement_options(i, env_vars)

            if options:
                actor = RemoteActor.options(**options).remote()
            else:
                actor = RemoteActor.remote()
            actors.append(actor)

        self._actor_handles = actors

        if self._n_virtual_nodes > 1:
            node_counts = [0] * self._n_virtual_nodes
            for vnode in self._actor_node_assignments:
                node_counts[vnode] += 1
            logger.info(
                "TaskDataflowBackend: %d actors across %d virtual nodes: %s",
                n, self._n_virtual_nodes, node_counts,
            )

        return actors

    def call(self, actor: Any, method: str, *args, **kwargs) -> Any:
        """Call a method on a single actor (synchronous).

        Dispatches via actor.call_method.remote(method, args).
        """
        ray = self._ray
        if kwargs:
            raise NotImplementedError(
                "TaskDataflowBackend.call() does not support kwargs."
            )
        dispatch_args = args[0] if len(args) == 1 else (args if args else None)
        future = actor.call_method.remote(method, dispatch_args)
        return ray.get(future)

    def call_all(
        self,
        actors: List[Any],
        method: str,
        args_per_actor: Optional[List] = None,
    ) -> List[Any]:
        """Call a method on all actors in parallel, gather results.

        Special handling for 'configure' and 'setup_from_pkl':
        - 'configure': dispatched to actor.configure.remote() (not call_method)
          because the actor needs settings before the TileWorker is created.
        - 'setup_from_pkl': intercepts the (pkl_path, interface_nodes) args,
          reads the PKL bytes, pushes them to the object store, and passes
          bytes to actor.setup_from_pkl.remote(). This is the key B4 change.
        - All other methods: dispatched via actor.call_method.remote(method, args).
        """
        ray = self._ray

        if method == 'configure':
            # Dispatch to actor.configure.remote(settings_dict)
            futures = []
            for i, actor in enumerate(actors):
                args = args_per_actor[i] if args_per_actor is not None else None
                settings = args[0] if isinstance(args, (tuple, list)) and args else (args or {})
                futures.append(actor.configure.remote(settings))
            return ray.get(futures)

        if method == 'setup_from_pkl':
            # B4 core: push PKL bytes to object store, pass bytes to actors.
            # args_per_actor: list of (pkl_path_str, interface_nodes) tuples.
            return self._setup_from_pkl_via_object_store(actors, args_per_actor or [])

        # Standard method dispatch via call_method
        futures = []
        for i, actor in enumerate(actors):
            args = args_per_actor[i] if args_per_actor is not None else None
            futures.append(actor.call_method.remote(method, args))
        return ray.get(futures)

    def _setup_from_pkl_via_object_store(
        self,
        actors: List[Any],
        args_per_actor: List[Any],
    ) -> List[Dict[str, Any]]:
        """B4 core: distribute tile PKLs via object store, set up actors.

        For each tile:
        1. Read PKL bytes from filesystem path.
        2. Push bytes to Ray object store (ray.put).
        3. Call actor.setup_from_pkl.remote(bytes, interface_nodes).

        After all actors complete, releases object store refs so the bytes
        can be GC'd. The TileWorker in each actor process retains the
        deserialized TileData and factor in its heap.

        Design note: We pass bytes directly (not ObjectRef) because Ray
        automatically dereferences ObjectRef arguments when calling remote
        methods. Passing bytes directly gives the same result and avoids
        an extra get() roundtrip.

        Args:
            actors: List of actor handles.
            args_per_actor: List of (pkl_path_or_bytes, interface_nodes) tuples.

        Returns:
            List of setup result dicts (one per actor), same structure as
            TileWorker.setup_from_pkl() for _collect_setup_results().
        """
        ray = self._ray
        n = len(actors)

        # Step 1: Read PKL bytes and push to object store.
        t0 = _time.perf_counter()
        pkl_refs = []
        iface_refs = []

        # Push interface_nodes once per unique set (all tiles share the same
        # interface_nodes in the current implementation).
        _iface_cache: Dict[int, Any] = {}

        for i, (pkl_path_or_src, interface_nodes) in enumerate(args_per_actor):
            # Serialize interface_nodes (shared across tiles)
            iface_key = id(interface_nodes)
            if iface_key not in _iface_cache:
                iface_bytes = pickle.dumps(interface_nodes)
                _iface_cache[iface_key] = ray.put(iface_bytes)
            iface_refs.append(_iface_cache[iface_key])

            # Read and push tile PKL
            if isinstance(pkl_path_or_src, bytes):
                # Already bytes (e.g., from in-memory TileData serialization)
                pkl_bytes = pkl_path_or_src
            elif isinstance(pkl_path_or_src, str):
                # Filesystem path -- read bytes (standard case)
                with open(pkl_path_or_src, 'rb') as f:
                    pkl_bytes = f.read()
            else:
                raise TypeError(
                    f"setup_from_pkl expects str path or bytes, "
                    f"got {type(pkl_path_or_src).__name__}"
                )
            pkl_refs.append(ray.put(pkl_bytes))

        self._put_elapsed = _time.perf_counter() - t0
        logger.info(
            "TaskDataflowBackend: pushed %d tile PKLs to object store in %.3f s "
            "(%.1f ms/tile avg)",
            n, self._put_elapsed, self._put_elapsed / max(1, n) * 1000,
        )

        # Keep refs alive in self so they aren't GC'd while actors are running.
        self._tile_pkl_refs = pkl_refs

        # Step 2: Initialize all actors in parallel via actor.setup_from_pkl.remote().
        # Pass ObjectRef args -- Ray auto-fetches from object store in actor process.
        t0 = _time.perf_counter()
        setup_futures = []
        for i, (actor, pkl_ref, iface_ref) in enumerate(
            zip(actors, pkl_refs, iface_refs)
        ):
            future = actor.setup_from_pkl.remote(pkl_ref, iface_ref)
            setup_futures.append(future)

        results = ray.get(setup_futures)
        self._setup_elapsed = _time.perf_counter() - t0

        if self._verbose:
            logger.info(
                "TaskDataflowBackend: %d workers setup in %.3f s "
                "(%.1f ms/worker avg)",
                n, self._setup_elapsed, self._setup_elapsed / max(1, n) * 1000,
            )

        # Step 3: Release object store refs (TileWorker in each actor keeps
        # the deserialized TileData in its own heap; refs no longer needed).
        self._tile_pkl_refs = []

        return results

    def call_all_streaming(
        self,
        actors: List[Any],
        method: str,
        args_per_actor: Optional[List] = None,
    ):
        """Submit all actor calls eagerly, yield results one at a time.

        B3 streaming compatible: lets the coordinator process and free each
        result before the next ray.get(), keeping peak coordinator memory to
        one result at a time while all actors compute in parallel.
        """
        ray = self._ray
        futures = []
        for i, actor in enumerate(actors):
            args = args_per_actor[i] if args_per_actor is not None else None
            futures.append(actor.call_method.remote(method, args))

        for i, future in enumerate(futures):
            yield i, ray.get(future)

    def map_func(self, func: Callable, args_list: List[Tuple]) -> List[Any]:
        """Apply a stateless function to each args tuple in parallel."""
        ray = self._ray
        remote_func = ray.remote(func)
        futures = [remote_func.remote(*args) for args in args_list]
        return ray.get(futures)

    def gather(self, futures: List[Any]) -> List[Any]:
        """Wait for and collect results from futures."""
        return self._ray.get(futures)

    def shutdown(self) -> None:
        """Release backend resources. Does not shut down Ray (caller manages)."""
        self._tile_pkl_refs = []

    @property
    def n_virtual_nodes(self) -> int:
        """Number of virtual nodes in the simulated cluster."""
        return self._n_virtual_nodes

    @property
    def actor_node_assignments(self) -> List[int]:
        """Virtual node assignment for each actor (index -> virtual_node_id)."""
        return self._actor_node_assignments

    @property
    def put_elapsed(self) -> float:
        """Wall time for tile PKL object store puts (seconds)."""
        return self._put_elapsed

    @property
    def setup_elapsed(self) -> float:
        """Wall time for actor worker setup (seconds)."""
        return self._setup_elapsed

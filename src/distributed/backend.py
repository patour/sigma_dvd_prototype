"""Compute backend abstraction for local vs distributed execution.

Provides a thin abstraction over local (in-process) and Ray-based execution.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Worker packing support (B1 V1: in-process packed workers)
# ---------------------------------------------------------------------------

class PackedTileWorker:
    """Owns k TileWorker instances in-process and fans method calls out serially.

    V1 implementation: all k workers live in the same process.  No parallelism
    is added — the benefit is reducing the number of remote Ray actors when the
    ``RayBackend`` wraps ``PackedTileWorker`` actors in a future iteration.

    Workers are routed via the paired ``VirtualWorkerHandle`` objects returned
    by ``handles()``.  The coordinator's ``call_all`` dispatches each handle
    to the appropriate inner worker transparently.
    """

    def __init__(self, workers: List[Any]) -> None:
        self._workers: List[Any] = list(workers)

    def call_worker(self, local_idx: int, method: str, args: Any) -> Any:
        """Call *method* on the inner worker at *local_idx*.

        *args* follows the same convention as ``call_all``'s ``args_per_actor``
        elements: a tuple/list (unpacked as ``*args``), a dict (unpacked as
        ``**kwargs``), ``None`` (no arguments), or a single non-iterable value
        (passed as a positional argument).
        """
        w = self._workers[local_idx]
        if args is None:
            return getattr(w, method)()
        elif isinstance(args, dict):
            return getattr(w, method)(**args)
        elif isinstance(args, (tuple, list)):
            return getattr(w, method)(*args)
        else:
            return getattr(w, method)(args)

    def handles(self) -> List['VirtualWorkerHandle']:
        """Return one :class:`VirtualWorkerHandle` per inner worker."""
        return [VirtualWorkerHandle(self, i) for i in range(len(self._workers))]

    def __len__(self) -> int:
        return len(self._workers)


class VirtualWorkerHandle:
    """A logical tile-worker slot inside a :class:`PackedTileWorker`.

    The coordinator's ``workers`` list may contain these proxies instead of
    bare actor references when ``tiles_per_worker`` packing is enabled.
    :meth:`~LocalBackend.call_all` detects them and routes each call to the
    correct inner worker.
    """

    def __init__(self, packed: PackedTileWorker, local_idx: int) -> None:
        self._packed = packed
        self._local_idx = local_idx

    @property
    def physical_actor(self) -> PackedTileWorker:
        return self._packed

    @property
    def local_idx(self) -> int:
        return self._local_idx


class PackedTileWorkerActor:
    """k TileWorker instances owned in-process, designed for use as a Ray actor.

    Each ``PackedTileWorkerActor`` holds *k* uninitialized ``TileWorker`` instances.
    Calls are routed to the correct inner worker via
    ``call_worker(local_idx, method, args)``.

    Typical RayBackend usage::

        RemotePacked = ray.remote(PackedTileWorkerActor)
        actor = RemotePacked.remote(batch_size)
        # Then route all per-tile calls via VirtualWorkerHandle → call_worker.remote

    This reduces n Ray actors to ceil(n/k), cutting actor-creation and
    scheduling overhead when tile count >> core count.
    """

    def __init__(self, k: int) -> None:
        # Lazy import: avoids circular import in Ray worker processes where
        # distributed/__init__.py imports from backend.py.
        from distributed.tile_worker import TileWorker  # noqa: PLC0415
        self._pack = PackedTileWorker([TileWorker() for _ in range(k)])

    def call_worker(self, local_idx: int, method: str, args: Any) -> Any:
        """Route *method* to the inner TileWorker at *local_idx*."""
        return self._pack.call_worker(local_idx, method, args)

    def size(self) -> int:
        """Number of inner TileWorker instances."""
        return len(self._pack)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class ComputeBackend(ABC):
    """Abstract compute backend for local vs distributed execution."""

    @abstractmethod
    def initialize(self, **kwargs) -> None:
        """Initialize the backend."""

    @abstractmethod
    def create_actors(self, actor_class: Type, configs: List[Any]) -> List[Any]:
        """Create actor instances from configs.

        Args:
            actor_class: The class to instantiate
            configs: Per-actor configuration

        Returns:
            List of actor handles
        """

    @abstractmethod
    def call(self, actor: Any, method: str, *args, **kwargs) -> Any:
        """Call a method on an actor.

        Args:
            actor: Actor handle
            method: Method name to call
            *args, **kwargs: Arguments to pass

        Returns:
            Method return value
        """

    @abstractmethod
    def call_all(self, actors: List[Any], method: str, args_per_actor: Optional[List] = None) -> List[Any]:
        """Call a method on all actors, optionally with different args per actor.

        Args:
            actors: List of actor handles
            method: Method name to call
            args_per_actor: If provided, per-actor argument tuples. If None,
                calls method() with no args on each actor.

        Returns:
            List of results, one per actor
        """

    @abstractmethod
    def map_func(self, func: Callable, args_list: List[Tuple]) -> List[Any]:
        """Apply a stateless function to each args tuple in parallel.

        Args:
            func: A picklable function to call
            args_list: List of argument tuples, one per invocation

        Returns:
            List of results, one per invocation
        """

    @abstractmethod
    def gather(self, futures: List[Any]) -> List[Any]:
        """Wait for and collect results from futures."""

    def call_all_streaming(
        self,
        actors: List[Any],
        method: str,
        args_per_actor: Optional[List] = None,
    ):
        """Iterate over per-actor results one at a time (generator).

        B3: Used for streaming Schur assembly.  Unlike ``call_all`` which
        waits for ALL results then returns them as a list (peak memory =
        sum of all results), this generator yields one result per actor in
        order, allowing the coordinator to process and discard each result
        before fetching the next.

        Args:
            actors: List of actor handles.
            method: Method name to call.
            args_per_actor: Optional per-actor argument list (same convention
                as ``call_all``).

        Yields:
            (actor_index, result) pairs in order (0, r0), (1, r1), ...
        """
        # Default implementation: sequential (subclasses may override for
        # pipelined execution — Ray subclass fetches futures eagerly in
        # submission order to overlap compute and transfer).
        results = self.call_all(actors, method, args_per_actor)
        yield from enumerate(results)

    @abstractmethod
    def shutdown(self) -> None:
        """Release backend resources."""


class LocalBackend(ComputeBackend):
    """In-process sequential execution. Zero dependencies.

    ``threads_per_worker`` is intentionally a no-op for this backend: all
    workers share the calling process, so changing thread-count env vars
    would affect the coordinator too and would have no effect on parallel
    throughput (workers run sequentially).  Use the Ray backend for true
    parallel multi-worker execution with per-actor thread budgets.
    """

    def initialize(self, **kwargs) -> None:
        pass

    def create_actors(self, actor_class: Type, configs: List[Any]) -> List[Any]:
        return [actor_class() for _ in configs]

    def call(self, actor: Any, method: str, *args, **kwargs) -> Any:
        return getattr(actor, method)(*args, **kwargs)

    def call_all(self, actors: List[Any], method: str, args_per_actor: Optional[List] = None) -> List[Any]:
        results = []
        for i, actor in enumerate(actors):
            args = args_per_actor[i] if args_per_actor is not None else None
            if isinstance(actor, VirtualWorkerHandle):
                result = actor.physical_actor.call_worker(actor.local_idx, method, args)
            elif args is None:
                result = getattr(actor, method)()
            elif isinstance(args, dict):
                result = getattr(actor, method)(**args)
            elif isinstance(args, (tuple, list)):
                result = getattr(actor, method)(*args)
            else:
                result = getattr(actor, method)(args)
            results.append(result)
        return results

    def call_all_streaming(
        self,
        actors: List[Any],
        method: str,
        args_per_actor: Optional[List] = None,
    ):
        """Truly sequential per-actor iteration.

        B3: Unlike the base-class default (which calls ``self.call_all()`` and
        materialises ALL results before yielding the first), this override
        calls each actor serially and yields immediately, so the coordinator
        can process and free each result before the next actor is invoked.

        Peak coordinator memory for the streaming path is therefore
        O(one actor's result) rather than O(sum of all actors' results).
        """
        for i, actor in enumerate(actors):
            args = args_per_actor[i] if args_per_actor is not None else None
            if isinstance(actor, VirtualWorkerHandle):
                result = actor.physical_actor.call_worker(actor.local_idx, method, args)
            elif args is None:
                result = getattr(actor, method)()
            elif isinstance(args, dict):
                result = getattr(actor, method)(**args)
            elif isinstance(args, (tuple, list)):
                result = getattr(actor, method)(*args)
            else:
                result = getattr(actor, method)(args)
            yield i, result

    def map_func(self, func: Callable, args_list: List[Tuple]) -> List[Any]:
        return [func(*args) for args in args_list]

    def gather(self, futures: List[Any]) -> List[Any]:
        return futures  # Already resolved

    def shutdown(self) -> None:
        pass


class RayBackend(ComputeBackend):
    """Ray-based distributed execution. Lazy import of ray.

    Per-actor thread budgets
    ------------------------
    When ``threads_per_worker`` is set (via :attr:`threads_per_worker`), each
    actor's ``runtime_env`` is configured with ``OMP_NUM_THREADS``,
    ``OPENBLAS_NUM_THREADS``, and ``MKL_NUM_THREADS`` so that BLAS/OMP
    libraries inside each worker process use at most that many threads.
    This prevents over-subscription when multiple workers share one node.

    Use ``threads_per_worker='auto'`` to set the value to
    ``max(1, available_cpus // n_workers)`` at actor-creation time.

    The coordinator process environment is *not* touched.
    """

    def __init__(self, threads_per_worker: Optional[Any] = None):
        self._ray = None
        self._initialized = False
        # int, 'auto', or None (no-op)
        self._threads_per_worker: Optional[Any] = threads_per_worker

    @property
    def threads_per_worker(self) -> Optional[Any]:
        """Current threads_per_worker setting (int, 'auto', or None)."""
        return self._threads_per_worker

    @threads_per_worker.setter
    def threads_per_worker(self, value: Optional[Any]) -> None:
        self._threads_per_worker = value

    def initialize(self, **kwargs) -> None:
        import ray
        self._ray = ray
        if not ray.is_initialized():
            # Increase health check timeout for long-running Schur complement computations
            # Default is 30s which is too short for large tiles (~2.78M nodes)
            system_config = kwargs.pop('_system_config', {})
            system_config.setdefault('health_check_timeout_ms', 1200000)  # 20 minutes
            system_config.setdefault('health_check_period_ms', 60000)     # 1 minute
            ray.init(_system_config=system_config, **kwargs)
        self._initialized = True

    def _resolve_threads_per_worker(self, n_workers: int) -> Optional[int]:
        """Resolve 'auto' to an integer, or return the stored int / None."""
        tpw = self._threads_per_worker
        if tpw is None:
            return None
        if tpw == 'auto':
            ray = self._ray
            try:
                total_cpus = int(ray.available_resources().get('CPU', n_workers))
            except Exception:
                total_cpus = n_workers
            return max(1, total_cpus // max(1, n_workers))
        return max(1, int(tpw))

    def create_actors(self, actor_class: Type, configs: List[Any]) -> List[Any]:
        if not self._initialized:
            raise RuntimeError("Backend not initialized. Call initialize() first.")
        ray = self._ray
        n = len(configs)

        tpw = self._resolve_threads_per_worker(n)
        RemoteClass = ray.remote(actor_class)

        if tpw is not None:
            env_vars = {
                'OMP_NUM_THREADS': str(tpw),
                'OPENBLAS_NUM_THREADS': str(tpw),
                'MKL_NUM_THREADS': str(tpw),
            }
            logger.info(
                "RayBackend: spawning %d actors with threads_per_worker=%d "
                "(OMP/OPENBLAS/MKL_NUM_THREADS=%d)",
                n, tpw, tpw,
            )
            return [
                RemoteClass.options(runtime_env={'env_vars': env_vars}).remote()
                for _ in configs
            ]
        else:
            return [RemoteClass.remote() for _ in configs]

    def call(self, actor: Any, method: str, *args, **kwargs) -> Any:
        ray = self._ray
        future = getattr(actor, method).remote(*args, **kwargs)
        return ray.get(future)

    def call_all(self, actors: List[Any], method: str, args_per_actor: Optional[List] = None) -> List[Any]:
        ray = self._ray
        futures = []
        for i, actor in enumerate(actors):
            args = args_per_actor[i] if args_per_actor is not None else None
            if isinstance(actor, VirtualWorkerHandle):
                # Physical actor is a ray.remote(PackedTileWorkerActor) handle.
                # Route via call_worker.remote(local_idx, method, args).
                # PackedTileWorkerActor.call_worker handles arg unpacking internally.
                future = actor.physical_actor.call_worker.remote(
                    actor.local_idx, method, args
                )
            elif args is None:
                future = getattr(actor, method).remote()
            elif isinstance(args, dict):
                future = getattr(actor, method).remote(**args)
            elif isinstance(args, (tuple, list)):
                future = getattr(actor, method).remote(*args)
            else:
                future = getattr(actor, method).remote(args)
            futures.append(future)
        return ray.get(futures)

    def call_all_streaming(
        self,
        actors: List[Any],
        method: str,
        args_per_actor: Optional[List] = None,
    ):
        """Submit all actor calls eagerly, then yield results one at a time.

        B3: Submits all ``method.remote(...)`` calls upfront (so workers
        compute in parallel), then yields (index, result) pairs by calling
        ``ray.get(future)`` one at a time in submission order.  This lets the
        coordinator process / free each result before the next ``ray.get``
        returns, keeping only one result live at a time while still running
        all workers concurrently.
        """
        ray = self._ray
        futures = []
        for i, actor in enumerate(actors):
            args = args_per_actor[i] if args_per_actor is not None else None
            if isinstance(actor, VirtualWorkerHandle):
                future = actor.physical_actor.call_worker.remote(
                    actor.local_idx, method, args
                )
            elif args is None:
                future = getattr(actor, method).remote()
            elif isinstance(args, dict):
                future = getattr(actor, method).remote(**args)
            elif isinstance(args, (tuple, list)):
                future = getattr(actor, method).remote(*args)
            else:
                future = getattr(actor, method).remote(args)
            futures.append(future)

        for i, future in enumerate(futures):
            yield i, ray.get(future)

    def map_func(self, func: Callable, args_list: List[Tuple]) -> List[Any]:
        ray = self._ray
        remote_func = ray.remote(func)
        futures = [remote_func.remote(*args) for args in args_list]
        return ray.get(futures)

    def gather(self, futures: List[Any]) -> List[Any]:
        return self._ray.get(futures)

    def shutdown(self) -> None:
        pass  # Don't shut down Ray - caller manages lifecycle

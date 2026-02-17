"""Compute backend abstraction for local vs distributed execution.

Provides a thin abstraction over local (in-process) and Ray-based execution.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

logger = logging.getLogger(__name__)


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
    def gather(self, futures: List[Any]) -> List[Any]:
        """Wait for and collect results from futures."""

    @abstractmethod
    def shutdown(self) -> None:
        """Release backend resources."""


class LocalBackend(ComputeBackend):
    """In-process sequential execution. Zero dependencies."""

    def initialize(self, **kwargs) -> None:
        pass

    def create_actors(self, actor_class: Type, configs: List[Any]) -> List[Any]:
        return [actor_class() for _ in configs]

    def call(self, actor: Any, method: str, *args, **kwargs) -> Any:
        return getattr(actor, method)(*args, **kwargs)

    def call_all(self, actors: List[Any], method: str, args_per_actor: Optional[List] = None) -> List[Any]:
        results = []
        for i, actor in enumerate(actors):
            if args_per_actor is not None:
                args = args_per_actor[i]
                if isinstance(args, dict):
                    result = getattr(actor, method)(**args)
                elif isinstance(args, (tuple, list)):
                    result = getattr(actor, method)(*args)
                else:
                    result = getattr(actor, method)(args)
            else:
                result = getattr(actor, method)()
            results.append(result)
        return results

    def gather(self, futures: List[Any]) -> List[Any]:
        return futures  # Already resolved

    def shutdown(self) -> None:
        pass


class RayBackend(ComputeBackend):
    """Ray-based distributed execution. Lazy import of ray."""

    def __init__(self):
        self._ray = None
        self._initialized = False

    def initialize(self, **kwargs) -> None:
        import ray
        self._ray = ray
        if not ray.is_initialized():
            ray.init(**kwargs)
        self._initialized = True

    def create_actors(self, actor_class: Type, configs: List[Any]) -> List[Any]:
        if not self._initialized:
            raise RuntimeError("Backend not initialized. Call initialize() first.")
        ray = self._ray
        # Create a Ray remote class from the actor class
        RemoteClass = ray.remote(actor_class)
        return [RemoteClass.remote() for _ in configs]

    def call(self, actor: Any, method: str, *args, **kwargs) -> Any:
        ray = self._ray
        future = getattr(actor, method).remote(*args, **kwargs)
        return ray.get(future)

    def call_all(self, actors: List[Any], method: str, args_per_actor: Optional[List] = None) -> List[Any]:
        ray = self._ray
        futures = []
        for i, actor in enumerate(actors):
            if args_per_actor is not None:
                args = args_per_actor[i]
                if isinstance(args, dict):
                    future = getattr(actor, method).remote(**args)
                elif isinstance(args, (tuple, list)):
                    future = getattr(actor, method).remote(*args)
                else:
                    future = getattr(actor, method).remote(args)
            else:
                future = getattr(actor, method).remote()
            futures.append(future)
        return ray.get(futures)

    def gather(self, futures: List[Any]) -> List[Any]:
        return self._ray.get(futures)

    def shutdown(self) -> None:
        pass  # Don't shut down Ray - caller manages lifecycle

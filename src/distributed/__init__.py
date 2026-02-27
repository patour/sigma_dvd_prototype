"""Distributed IR-drop via tile-boundary domain decomposition.

Provides distributed DDM solver that decomposes multi-tile PDN netlists
into per-tile subproblems coupled through a Schur complement interface.
"""

from .result import (
    TileSolveResult,
    DistributedSolveResult,
    DistributedSolverContext,
)
from .backend import (
    ComputeBackend,
    LocalBackend,
    RayBackend,
)
from .parser import (
    DistributedNetlistParser,
    TileConfig,
    PackageData,
    PowerGridMetaData,
)
from .tile_worker import (
    TileWorker,
    TileData,
)
from .model import (
    DistributedPowerGridModel,
    create_distributed_model,
)
from .solver import (
    DistributedDDMSolver,
)

__all__ = [
    # Result / Context
    "TileSolveResult",
    "DistributedSolveResult",
    "DistributedSolverContext",
    # Backends
    "ComputeBackend",
    "LocalBackend",
    "RayBackend",
    # Parser
    "DistributedNetlistParser",
    "TileConfig",
    "PackageData",
    "PowerGridMetaData",
    # Worker
    "TileWorker",
    "TileData",
    # Model
    "DistributedPowerGridModel",
    "create_distributed_model",
    # Solver
    "DistributedDDMSolver",
]

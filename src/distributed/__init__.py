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
    parse_tile_with_instances,
)
from .model import (
    DistributedPowerGridModel,
    create_distributed_model,
    load_distributed_partitions,
)
from .solver import (
    DistributedDDMSolver,
)
from .heatmap import (
    LayerBinSpec,
    GlobalBinSpec,
    build_global_bin_spec,
    prebin_tile,
    merge_tile_prebins,
    compute_boundary_ownership,
    plot_distributed_heatmaps,
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
    "parse_tile_with_instances",
    # Model
    "DistributedPowerGridModel",
    "create_distributed_model",
    "load_distributed_partitions",
    # Solver
    "DistributedDDMSolver",
    # Heatmap
    "LayerBinSpec",
    "GlobalBinSpec",
    "build_global_bin_spec",
    "prebin_tile",
    "merge_tile_prebins",
    "compute_boundary_ownership",
    "plot_distributed_heatmaps",
]

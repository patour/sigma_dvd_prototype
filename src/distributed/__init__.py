"""Distributed IR-drop via tile-boundary domain decomposition.

Provides distributed DDM solver that decomposes multi-tile PDN netlists
into per-tile subproblems coupled through a Schur complement interface.
"""

from .result import (
    TileSolveResult,
    DistributedSolveResult,
    DistributedTopologyContext,
    DistributedSolverContext,
    DistributedSmoothedSources,
    DistributedTransientContext,
    DistributedQuasiStaticResult,
    DistributedTransientResult,
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
    parse_and_dump_tile,
)
from .model import (
    ParsedTileBundle,
    DistributedPowerGridModel,
    create_distributed_model,
    load_distributed_partitions,
)
from .solver import (
    DistributedDDMSolver,
)
from .decomposition import (
    analyze_distributed_decomposition,
)
from .heatmap import (
    LayerBinSpec,
    GlobalBinSpec,
    build_global_bin_spec,
    prebin_tile,
    merge_tile_prebins,
    compute_boundary_ownership,
    plot_distributed_heatmaps,
    plot_distributed_td_heatmaps,
)

__all__ = [
    # Result / Context
    "TileSolveResult",
    "DistributedSolveResult",
    "DistributedTopologyContext",
    "DistributedSolverContext",
    "DistributedSmoothedSources",
    "DistributedTransientContext",
    "DistributedQuasiStaticResult",
    "DistributedTransientResult",
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
    "parse_and_dump_tile",
    # Model
    "ParsedTileBundle",
    "DistributedPowerGridModel",
    "create_distributed_model",
    "load_distributed_partitions",
    # Solver
    "DistributedDDMSolver",
    # Decomposition
    "analyze_distributed_decomposition",
    # Heatmap
    "LayerBinSpec",
    "GlobalBinSpec",
    "build_global_bin_spec",
    "prebin_tile",
    "merge_tile_prebins",
    "compute_boundary_ownership",
    "plot_distributed_heatmaps",
    "plot_distributed_td_heatmaps",
]

"""Distributed power grid model.

Analogous to UnifiedPowerGridModel but holds distributed tile data via
worker actors instead of a monolithic graph.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from .backend import ComputeBackend, LocalBackend, RayBackend
from .parser import PackageData, PowerGridMetaData, TileConfig
from .tile_worker import TileWorker

logger = logging.getLogger(__name__)


@dataclass
class DistributedPowerGridModel:
    """Distributed power grid model. Holds workers, package, metadata.

    Analogous to UnifiedPowerGridModel which holds graph + pads + vdd.
    Created by create_distributed_model() factory, consumed by DistributedDDMSolver.
    """

    # Workers (hold per-tile graph data)
    backend: ComputeBackend
    workers: List[Any]  # TileWorker handles (objects or Ray actors)

    # Interface topology
    interface_nodes: Set[str]  # All boundary + promoted M13 nodes
    tile_boundary_nodes: Dict[Tuple[int, int], List[str]]  # per-tile boundary lists
    tile_interior_counts: Dict[Tuple[int, int], int]  # per-tile interior node counts

    # Package / voltage sources
    package_data: PackageData

    # Metadata
    metadata: PowerGridMetaData
    island_stats: Dict[Tuple[int, int], Dict] = field(default_factory=dict)

    @property
    def tile_ids(self) -> List[Tuple[int, int]]:
        return [tc.tile_id for tc in self.metadata.tile_configs]

    @property
    def pad_nodes(self) -> Set[str]:
        return self.package_data.pad_nodes

    @property
    def vdd(self) -> float:
        return self.package_data.vdd

    @property
    def net_name(self) -> str:
        return self.package_data.net_name

    @property
    def n_tiles(self) -> int:
        return len(self.metadata.tile_configs)

    @property
    def tile_grid(self) -> Tuple[int, int]:
        return self.metadata.tile_grid

    def shutdown(self):
        """Release backend resources."""
        self.backend.shutdown()


def create_distributed_model(
    metadata: PowerGridMetaData,
    backend: str = 'local',
    n_workers: Optional[int] = None,
    **backend_kwargs,
) -> DistributedPowerGridModel:
    """Factory function for distributed model (analogous to create_model_from_pdn).

    1. Create backend (LocalBackend or RayBackend)
    2. Compute interface_nodes = boundary_nodes union package.die_attachment_nodes
    3. Create TileWorker actors via backend
    4. Workers parse tile files, classify boundary/interior, detect floating islands
    5. Collect per-tile boundary node lists and metadata
    6. Return DistributedPowerGridModel

    Args:
        metadata: PowerGridMetaData from DistributedNetlistParser
        backend: 'local' or 'ray'
        n_workers: Number of workers (only for ray backend)
        **backend_kwargs: Extra kwargs passed to backend.initialize()
    """
    t_start = time.perf_counter()

    # 1. Create backend
    if backend == 'ray':
        be = RayBackend()
        be.initialize(**backend_kwargs)
    else:
        be = LocalBackend()
        be.initialize()

    # 2. Collect boundary nodes from tile .ckt files (fast pre-scan)
    from .parser import DistributedNetlistParser
    parser = DistributedNetlistParser(str(metadata.tile_configs[0].ckt_path).rsplit('/', 1)[0])
    boundary_nodes = parser.collect_boundary_nodes(metadata.tile_configs)

    # Interface = boundary nodes + die attachment nodes from package
    interface_nodes = boundary_nodes | metadata.package_data.die_attachment_nodes
    logger.info(
        f"Interface: {len(interface_nodes)} nodes "
        f"({len(boundary_nodes)} boundary + {len(metadata.package_data.die_attachment_nodes)} die attachment)"
    )

    # 3. Create workers
    workers = be.create_actors(TileWorker, metadata.tile_configs)

    # 4. Setup workers (parse tiles, classify nodes)
    tile_configs_as_dicts = [
        {
            'tile_id': list(tc.tile_id),
            'ckt_path': tc.ckt_path,
            'nd_path': tc.nd_path,
            'instance_path': tc.instance_path,
            'net_filter': tc.net_filter,
        }
        for tc in metadata.tile_configs
    ]

    setup_args = [
        (cfg, interface_nodes) for cfg in tile_configs_as_dicts
    ]
    setup_results = be.call_all(workers, 'setup', setup_args)

    # 5. Collect results
    tile_boundary_nodes: Dict[Tuple[int, int], List[str]] = {}
    tile_interior_counts: Dict[Tuple[int, int], int] = {}
    island_stats: Dict[Tuple[int, int], Dict] = {}

    for result in setup_results:
        tid = tuple(result['tile_id'])
        tile_boundary_nodes[tid] = result['boundary_nodes']
        tile_interior_counts[tid] = result['n_interior']
        island_stats[tid] = {'islands_removed': result['islands_removed']}

    total_interior = sum(tile_interior_counts.values())
    total_boundary = sum(len(v) for v in tile_boundary_nodes.values())
    t_elapsed = time.perf_counter() - t_start
    logger.info(
        f"Model created in {t_elapsed:.3f}s: {len(workers)} tiles, "
        f"{total_interior} interior nodes, {total_boundary} boundary entries, "
        f"{len(interface_nodes)} interface nodes"
    )

    return DistributedPowerGridModel(
        backend=be,
        workers=workers,
        interface_nodes=interface_nodes,
        tile_boundary_nodes=tile_boundary_nodes,
        tile_interior_counts=tile_interior_counts,
        package_data=metadata.package_data,
        metadata=metadata,
        island_stats=island_stats,
    )

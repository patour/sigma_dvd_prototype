"""Distributed power grid model.

Analogous to UnifiedPowerGridModel but holds distributed tile data via
worker actors instead of a monolithic graph.
"""

from __future__ import annotations

import logging
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from .backend import ComputeBackend, LocalBackend, RayBackend
from .parser import PackageData, PowerGridMetaData, TileConfig
from .tile_worker import TileData, TileWorker

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


def load_distributed_partitions(
    pkl_dir: str,
) -> Tuple[PowerGridMetaData, Set[str], Dict[Tuple[int, int], TileData]]:
    """Load pre-parsed tile partitions and metadata from .pkl files.

    Reads metadata.pkl (PowerGridMetaData + boundary_nodes) and all
    tile_X_Y.pkl files (TileData) produced by parse_and_dump().

    .. warning::
        Uses ``pickle.load()`` which can execute arbitrary code. Only load
        .pkl files that you produced yourself or received from a trusted
        source. Never load .pkl files from untrusted / user-supplied paths.

    Args:
        pkl_dir: Directory containing metadata.pkl and tile_X_Y.pkl files

    Returns:
        Tuple of (metadata, boundary_nodes, tile_data_dict) where
        tile_data_dict maps tile_id -> TileData

    Raises:
        FileNotFoundError: If metadata.pkl is missing
        TypeError: If loaded objects have unexpected types
        ValueError: If no tile .pkl files are found
    """
    import re
    pkl_path = Path(pkl_dir)

    # Load metadata
    meta_pkl = pkl_path / 'metadata.pkl'
    if not meta_pkl.exists():
        raise FileNotFoundError(f"metadata.pkl not found in {pkl_dir}")

    # WARNING: pickle.load() can execute arbitrary code. Only load files
    # from trusted sources. See https://docs.python.org/3/library/pickle.html
    with open(meta_pkl, 'rb') as f:
        meta_bundle = pickle.load(f)

    # Type-check the metadata bundle
    if not isinstance(meta_bundle, dict) or 'metadata' not in meta_bundle:
        raise TypeError(
            f"metadata.pkl must contain a dict with a 'metadata' key, "
            f"got {type(meta_bundle).__name__}"
        )
    if not isinstance(meta_bundle['metadata'], PowerGridMetaData):
        raise TypeError(
            f"metadata.pkl['metadata'] must be a PowerGridMetaData instance, "
            f"got {type(meta_bundle['metadata']).__name__}"
        )

    metadata: PowerGridMetaData = meta_bundle['metadata']
    boundary_nodes: Set[str] = meta_bundle['boundary_nodes']

    # Discover and load tile .pkl files
    tile_pkl_re = re.compile(r'tile_(\d+)_(\d+)\.pkl$')
    tile_data_dict: Dict[Tuple[int, int], TileData] = {}

    for p in sorted(pkl_path.glob('tile_*.pkl')):
        m = tile_pkl_re.match(p.name)
        if m:
            tid = (int(m.group(1)), int(m.group(2)))
            with open(p, 'rb') as f:
                td = pickle.load(f)
            if not isinstance(td, TileData):
                raise TypeError(
                    f"{p.name} must contain a TileData instance, "
                    f"got {type(td).__name__}"
                )
            tile_data_dict[tid] = td

    if not tile_data_dict:
        raise ValueError(f"No tile_X_Y.pkl files found in {pkl_dir}")

    logger.info(
        f"Loaded {len(tile_data_dict)} tile partitions + metadata from {pkl_dir}"
    )

    return metadata, boundary_nodes, tile_data_dict


def create_distributed_model(
    metadata: PowerGridMetaData,
    backend: str = 'local',
    n_workers: Optional[int] = None,
    use_pkl: bool = False,
    pkl_dir: Optional[str] = None,
    boundary_nodes: Optional[Set[str]] = None,
    tile_data_dict: Optional[Dict[Tuple[int, int], TileData]] = None,
    **backend_kwargs,
) -> DistributedPowerGridModel:
    """Factory function for distributed model (analogous to create_model_from_pdn).

    1. Create backend (LocalBackend or RayBackend)
    2. Compute interface_nodes = boundary_nodes union package.die_attachment_nodes
    3. Create TileWorker actors via backend
    4. Workers parse tile files (or load TileData), classify boundary/interior, detect floating islands
    5. Collect per-tile boundary node lists and metadata
    6. Return DistributedPowerGridModel

    Args:
        metadata: PowerGridMetaData from DistributedNetlistParser
        backend: 'local' or 'ray'
        n_workers: Number of workers (only for ray backend)
        use_pkl: If True, use pre-parsed TileData from pkl files instead of
            re-parsing .ckt files. Requires either (pkl_dir) or
            (boundary_nodes + tile_data_dict) to be provided.
        pkl_dir: Directory containing tile_X_Y.pkl files (used when use_pkl=True).
            If provided, boundary_nodes and tile_data_dict are loaded from it.
        boundary_nodes: Pre-collected boundary nodes (used when use_pkl=True).
            Alternative to pkl_dir when TileData is already in memory.
        tile_data_dict: Pre-loaded tile data dict (used when use_pkl=True).
            Alternative to pkl_dir when TileData is already in memory.
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

    # 2. Collect boundary nodes
    if use_pkl:
        # Load from pkl if directory given but data not provided
        if pkl_dir is not None and (boundary_nodes is None or tile_data_dict is None):
            _, boundary_nodes, tile_data_dict = load_distributed_partitions(pkl_dir)
        if boundary_nodes is None or tile_data_dict is None:
            raise ValueError(
                "use_pkl=True requires either pkl_dir or both "
                "boundary_nodes and tile_data_dict"
            )
    else:
        # Original path: pre-scan .ckt files for boundary nodes
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

    # 4. Setup workers
    if use_pkl:
        # Validate all expected tile IDs are present in tile_data_dict
        expected_ids = {tc.tile_id for tc in metadata.tile_configs}
        available_ids = set(tile_data_dict.keys())
        missing_ids = expected_ids - available_ids
        if missing_ids:
            raise ValueError(
                f"tile_data_dict is missing {len(missing_ids)} tile(s) "
                f"expected by metadata.tile_configs: {sorted(missing_ids)}"
            )

        # Use pre-parsed TileData — no file I/O on workers
        setup_args = [
            (tile_data_dict[tc.tile_id], interface_nodes)
            for tc in metadata.tile_configs
        ]
        setup_results = be.call_all(workers, 'setup_from_tile_data', setup_args)
    else:
        # Original path: parse tile files on each worker
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

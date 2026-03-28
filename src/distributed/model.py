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
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

from .backend import ComputeBackend, LocalBackend, RayBackend
from .parser import PackageData, PowerGridMetaData, TileConfig
from .tile_worker import TileData, TileWorker

if TYPE_CHECKING:
    from solver.unified_solver import SolverBackendConfig

logger = logging.getLogger(__name__)


@dataclass
class ParsedTileBundle:
    """Lightweight coordinator-side metadata produced by parse_and_dump().

    Contains only what the coordinator needs to orchestrate workers --
    no per-tile graph data. Workers load their own TileData from pkl files
    via ``setup_from_pkl()``.
    """

    metadata: PowerGridMetaData
    shared_boundary_nodes: Set[str]
    pkl_dir: str


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
    tile_kept_nonlargest_iface: Dict[Tuple[int, int], List[str]] = field(default_factory=dict)

    # Per-role solver backend configs (None = use module globals)
    coordinator_solver_config: Optional[SolverBackendConfig] = field(default=None, repr=False)
    worker_solver_config: Optional[SolverBackendConfig] = field(default=None, repr=False)

    # Internal: temp pkl_dir created by legacy shim (cleaned up in shutdown)
    _owns_pkl_dir: Optional[str] = field(default=None, repr=False)

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
        """Release backend resources and clean up temp directories."""
        self.backend.shutdown()
        if self._owns_pkl_dir:
            import shutil
            shutil.rmtree(self._owns_pkl_dir, ignore_errors=True)
            self._owns_pkl_dir = None


def load_distributed_partitions(
    pkl_dir: str,
) -> ParsedTileBundle:
    """Load pre-parsed metadata from .pkl files and return a ParsedTileBundle.

    Reads only ``metadata.pkl`` (PowerGridMetaData + boundary_nodes).
    Tile .pkl files are loaded lazily by workers via ``setup_from_pkl()``.

    .. warning::
        Uses ``pickle.load()`` which can execute arbitrary code. Only load
        .pkl files that you produced yourself or received from a trusted
        source. Never load .pkl files from untrusted / user-supplied paths.

    Args:
        pkl_dir: Directory containing metadata.pkl and tile_X_Y.pkl files

    Returns:
        ParsedTileBundle with metadata, shared_boundary_nodes, and pkl_dir.

    Raises:
        FileNotFoundError: If metadata.pkl is missing
        TypeError: If loaded objects have unexpected types
    """
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

    logger.info(
        f"Loaded metadata from {pkl_dir}: {len(metadata.tile_configs)} tiles, "
        f"{len(boundary_nodes)} shared boundary nodes"
    )

    return ParsedTileBundle(
        metadata=metadata,
        shared_boundary_nodes=boundary_nodes,
        pkl_dir=str(pkl_path),
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _init_backend(
    backend: str, backend_kwargs: Dict[str, Any],
) -> ComputeBackend:
    """Create and initialize compute backend."""
    if backend == 'ray':
        be = RayBackend()
        be.initialize(**backend_kwargs)
    else:
        be = LocalBackend()
        be.initialize()
    return be


def _collect_setup_results(
    setup_results: List[Dict[str, Any]],
) -> Tuple[
    Dict[Tuple[int, int], List[str]],
    Dict[Tuple[int, int], int],
    Dict[Tuple[int, int], Dict],
    Dict[Tuple[int, int], List[str]],
]:
    """Parse worker setup results into per-tile dicts.

    Returns:
        (tile_boundary_nodes, tile_interior_counts, island_stats,
         tile_kept_nonlargest_iface)
    """
    tile_boundary_nodes: Dict[Tuple[int, int], List[str]] = {}
    tile_interior_counts: Dict[Tuple[int, int], int] = {}
    island_stats: Dict[Tuple[int, int], Dict] = {}
    tile_kept_nonlargest_iface: Dict[Tuple[int, int], List[str]] = {}

    for result in setup_results:
        tid = tuple(result['tile_id'])
        tile_boundary_nodes[tid] = result['boundary_nodes']
        tile_interior_counts[tid] = result['n_interior']
        island_stats[tid] = {'islands_removed': result['islands_removed']}
        tile_kept_nonlargest_iface[tid] = result.get('kept_nonlargest_iface', [])

    return tile_boundary_nodes, tile_interior_counts, island_stats, tile_kept_nonlargest_iface


def create_distributed_model(
    bundle_or_metadata,
    backend: str = 'local',
    n_workers: Optional[int] = None,
    coordinator_solver_config: Optional[SolverBackendConfig] = None,
    worker_solver_config: Optional[SolverBackendConfig] = None,
    # Legacy kwargs (deprecated -- use ParsedTileBundle instead)
    use_pkl: bool = False,
    pkl_dir: Optional[str] = None,
    boundary_nodes: Optional[Set[str]] = None,
    tile_data_dict: Optional[Dict[Tuple[int, int], TileData]] = None,
    **backend_kwargs,
) -> DistributedPowerGridModel:
    """Factory function for distributed model (analogous to create_model_from_pdn).

    Primary path (recommended):
        Pass a ``ParsedTileBundle`` as the first argument. Workers load
        their own TileData from pkl files via ``setup_from_pkl()``.

    Legacy path (deprecated):
        Pass a ``PowerGridMetaData`` as the first argument with optional
        ``use_pkl``, ``pkl_dir``, ``boundary_nodes``, ``tile_data_dict``
        kwargs. Emits ``DeprecationWarning``.

    Args:
        bundle_or_metadata: A ``ParsedTileBundle`` (new) or
            ``PowerGridMetaData`` (legacy, deprecated).
        backend: 'local' or 'ray'
        n_workers: Number of workers (only for ray backend)
        coordinator_solver_config: Backend settings for the coordinator
            (interface factorization).  ``None`` = use module globals.
        worker_solver_config: Backend settings for tile workers.
            ``None`` = use module globals.
        use_pkl: (Deprecated) If True, use pre-parsed TileData.
        pkl_dir: (Deprecated) Directory containing tile_X_Y.pkl files.
        boundary_nodes: (Deprecated) Pre-collected boundary nodes.
        tile_data_dict: (Deprecated) Pre-loaded tile data dict.
        **backend_kwargs: Extra kwargs passed to backend.initialize()
    """
    import warnings

    # ── Backward-compat shim: adapt legacy PowerGridMetaData call ──
    _legacy_temp_dir = None
    if isinstance(bundle_or_metadata, PowerGridMetaData):
        warnings.warn(
            "Passing PowerGridMetaData to create_distributed_model() is "
            "deprecated. Pass a ParsedTileBundle instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        bundle = _adapt_legacy_args(
            bundle_or_metadata,
            use_pkl=use_pkl,
            pkl_dir=pkl_dir,
            boundary_nodes=boundary_nodes,
            tile_data_dict=tile_data_dict,
        )
        # Track temp dirs created by sub-cases 2 & 3 for cleanup
        if bundle.pkl_dir != pkl_dir:
            _legacy_temp_dir = bundle.pkl_dir
    elif isinstance(bundle_or_metadata, ParsedTileBundle):
        bundle = bundle_or_metadata
    else:
        raise TypeError(
            f"Expected ParsedTileBundle or PowerGridMetaData, "
            f"got {type(bundle_or_metadata).__name__}"
        )

    model = _create_distributed_model_from_bundle(
        bundle, backend,
        coordinator_solver_config=coordinator_solver_config,
        worker_solver_config=worker_solver_config,
        **backend_kwargs,
    )
    if _legacy_temp_dir:
        model._owns_pkl_dir = _legacy_temp_dir
    return model


def _adapt_legacy_args(
    metadata: PowerGridMetaData,
    use_pkl: bool,
    pkl_dir: Optional[str],
    boundary_nodes: Optional[Set[str]],
    tile_data_dict: Optional[Dict[Tuple[int, int], TileData]],
) -> ParsedTileBundle:
    """Convert legacy create_distributed_model() kwargs into a ParsedTileBundle.

    Handles three sub-cases:
    1. use_pkl + pkl_dir -> load metadata from pkl, return bundle
    2. use_pkl + boundary_nodes + tile_data_dict -> dump to temp dir, return bundle
    3. Not use_pkl -> scan .ckt for boundary nodes, dump tiles to temp dir, return bundle
    """
    import tempfile

    if use_pkl and pkl_dir is not None:
        # Sub-case 1: pkl_dir exists, just load metadata
        loaded = load_distributed_partitions(pkl_dir)
        return loaded

    if use_pkl and boundary_nodes is not None and tile_data_dict is not None:
        # Sub-case 2: in-memory TileData, dump to temp dir
        tmp_dir = tempfile.mkdtemp(prefix='dist_model_')
        _dump_tile_data_to_dir(metadata, boundary_nodes, tile_data_dict, tmp_dir)
        return ParsedTileBundle(
            metadata=metadata,
            shared_boundary_nodes=boundary_nodes,
            pkl_dir=tmp_dir,
        )

    if use_pkl:
        raise ValueError(
            "use_pkl=True requires either pkl_dir or both "
            "boundary_nodes and tile_data_dict"
        )

    # Sub-case 3: no pkl, scan .ckt files
    from .parser import DistributedNetlistParser
    parser = DistributedNetlistParser(
        str(metadata.tile_configs[0].ckt_path).rsplit('/', 1)[0]
    )
    bnd_nodes = parser.collect_shared_boundary_nodes(metadata.tile_configs)

    # Need to dump tiles to a temp dir so workers can use setup_from_pkl
    tmp_dir = tempfile.mkdtemp(prefix='dist_model_')
    # Parse tiles and dump
    from .backend import LocalBackend
    from .tile_worker import parse_tile_with_instances
    be = LocalBackend()
    be.initialize()

    args_list = [
        (tc.ckt_path, tc.nd_path, tc.net_filter, tc.tile_id, tc.instance_path)
        for tc in metadata.tile_configs
    ]
    tile_results = be.map_func(parse_tile_with_instances, args_list)
    tile_data_dict_local = {}
    for tc, td in zip(metadata.tile_configs, tile_results):
        tile_data_dict_local[tc.tile_id] = td
    _dump_tile_data_to_dir(metadata, bnd_nodes, tile_data_dict_local, tmp_dir)
    return ParsedTileBundle(
        metadata=metadata,
        shared_boundary_nodes=bnd_nodes,
        pkl_dir=tmp_dir,
    )


def _dump_tile_data_to_dir(
    metadata: PowerGridMetaData,
    boundary_nodes: Set[str],
    tile_data_dict: Dict[Tuple[int, int], TileData],
    output_dir: str,
) -> None:
    """Dump TileData + metadata to a directory (helper for legacy shim)."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for tid, td in tile_data_dict.items():
        x, y = tid
        with open(out / f'tile_{x}_{y}.pkl', 'wb') as f:
            pickle.dump(td, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(out / 'metadata.pkl', 'wb') as f:
        pickle.dump(
            {'metadata': metadata, 'boundary_nodes': boundary_nodes},
            f, protocol=pickle.HIGHEST_PROTOCOL,
        )


def _create_distributed_model_from_bundle(
    bundle: ParsedTileBundle,
    backend: str = 'local',
    coordinator_solver_config: Optional[SolverBackendConfig] = None,
    worker_solver_config: Optional[SolverBackendConfig] = None,
    **backend_kwargs,
) -> DistributedPowerGridModel:
    """Core factory: create model from a ParsedTileBundle.

    Single code path: workers load their TileData from pkl files via
    ``setup_from_pkl()``.
    """
    t_start = time.perf_counter()
    metadata = bundle.metadata

    # 1. Create backend
    be = _init_backend(backend, backend_kwargs)

    # 2. Compute interface nodes
    die_net_map = getattr(metadata.package_data, 'die_attachment_net_map', {})
    interface_nodes = (
        bundle.shared_boundary_nodes
        | set(die_net_map.keys())
        | metadata.package_data.die_attachment_nodes
    )
    logger.info(
        f"Interface: {len(interface_nodes)} nodes "
        f"({len(bundle.shared_boundary_nodes)} boundary + "
        f"{len(metadata.package_data.die_attachment_nodes)} die attachment)"
    )

    # 3. Create workers
    workers = be.create_actors(TileWorker, metadata.tile_configs)

    # 3b. Propagate solver settings to workers
    from solver.coupled_system import get_partial_factor_reg_resistance

    # Build worker settings dict: use explicit config or snapshot globals.
    # partial_factor_reg_ohms is a separate concern (coupled_system.py),
    # not included in SolverBackendConfig; always inherited from globals.
    from solver.unified_solver import SolverBackendConfig as _SBC
    solver_settings = {
        **(worker_solver_config or _SBC.from_globals()).to_dict(),
        'partial_factor_reg_ohms': get_partial_factor_reg_resistance(),
    }
    be.call_all(workers, 'configure', [(solver_settings,)] * len(workers))

    # 4. Setup workers: each loads its own .pkl and builds block system
    pkl_path = Path(bundle.pkl_dir)
    setup_args = [
        (str(pkl_path / f'tile_{tc.tile_id[0]}_{tc.tile_id[1]}.pkl'), interface_nodes)
        for tc in metadata.tile_configs
    ]
    setup_results = be.call_all(workers, 'setup_from_pkl', setup_args)

    # 5. Collect results
    (tile_boundary_nodes, tile_interior_counts,
     island_stats, tile_kept_nonlargest_iface) = _collect_setup_results(setup_results)

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
        tile_kept_nonlargest_iface=tile_kept_nonlargest_iface,
        coordinator_solver_config=coordinator_solver_config,
        worker_solver_config=worker_solver_config,
    )

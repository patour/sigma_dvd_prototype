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

from .backend import ComputeBackend, LocalBackend, PackedTileWorker, RayBackend, VirtualWorkerHandle
from .parser import PackageData, PowerGridMetaData, TileConfig
from .tile_worker import TileData, TileWorker

if TYPE_CHECKING:
    from pgmath.factor import SolverBackendConfig

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
    tile_boundary_nodes: Dict[tuple, List[str]]  # per-tile boundary lists
    tile_interior_counts: Dict[tuple, int]  # per-tile interior node counts

    # Package / voltage sources
    package_data: PackageData

    # Metadata
    metadata: PowerGridMetaData
    island_stats: Dict[tuple, Dict] = field(default_factory=dict)
    tile_kept_nonlargest_iface: Dict[tuple, List[str]] = field(default_factory=dict)

    # Per-role solver backend configs (None = use module globals)
    coordinator_solver_config: Optional[SolverBackendConfig] = field(default=None, repr=False)
    worker_solver_config: Optional[SolverBackendConfig] = field(default=None, repr=False)

    # B2: Coordinator-side solver settings dict.
    # Keys recognised:
    #   'interface_solver': 'direct' | 'cg' | 'auto' (default 'auto')
    #   'interface_matvec_mode': 'assembled' | 'tilewise' (default 'assembled')
    #   'interface_preconditioner': 'block_jacobi' | 'jacobi' | 'none' | 'amg'
    #   'interface_cg_rtol': float (default 1e-10)
    # These affect only the coordinator; they are NOT propagated to workers.
    settings: Dict[str, Any] = field(default_factory=dict, repr=False)

    # Internal: temp pkl_dir created by legacy shim (cleaned up in shutdown)
    _owns_pkl_dir: Optional[str] = field(default=None, repr=False)

    # pkl_dir: the output directory that contains tile_*.pkl files for this model.
    # Used as the default VCS cache directory in preprocess_sources() so that
    # sub-tile VCS caches land in the PKL output dir, NOT in the source netlist dir.
    # None when created from legacy paths that don't have an output dir concept.
    pkl_dir: Optional[str] = field(default=None, repr=False)

    @property
    def tile_ids(self) -> List[tuple]:
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
    backend: str,
    backend_kwargs: Dict[str, Any],
    threads_per_worker: Any = None,
) -> ComputeBackend:
    """Create and initialize compute backend.

    Args:
        backend: 'local' or 'ray'
        backend_kwargs: Extra kwargs passed to backend.initialize()
        threads_per_worker: For RayBackend — number of BLAS/OMP threads per
            actor process (int), or 'auto' for ``max(1, cpus // n_workers)``.
            Has no effect on LocalBackend (workers run in-process).
    """
    if backend == 'ray':
        be = RayBackend(threads_per_worker=threads_per_worker)
        be.initialize(**backend_kwargs)
    else:
        be = LocalBackend()
        be.initialize()
    return be


def _collect_setup_results(
    setup_results: List[Dict[str, Any]],
) -> Tuple[
    Dict[tuple, List[str]],
    Dict[tuple, int],
    Dict[tuple, Dict],
    Dict[tuple, List[str]],
]:
    """Parse worker setup results into per-tile dicts.

    Returns:
        (tile_boundary_nodes, tile_interior_counts, island_stats,
         tile_kept_nonlargest_iface)
    """
    tile_boundary_nodes: Dict[tuple, List[str]] = {}
    tile_interior_counts: Dict[tuple, int] = {}
    island_stats: Dict[tuple, Dict] = {}
    tile_kept_nonlargest_iface: Dict[tuple, List[str]] = {}

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
    threads_per_worker: Any = None,
    use_step_columns: bool = True,
    max_table_mb: float = 512.0,
    tiles_per_worker: Any = None,
    # Legacy kwargs (deprecated -- use ParsedTileBundle instead)
    use_pkl: bool = False,
    pkl_dir: Optional[str] = None,
    boundary_nodes: Optional[Set[str]] = None,
    tile_data_dict: Optional[Dict[tuple, TileData]] = None,
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
        n_workers: Number of workers (only for ray backend; currently
            informational — tile count is determined by the bundle).
        coordinator_solver_config: Backend settings for the coordinator
            (interface factorization).  ``None`` = use module globals.
        worker_solver_config: Backend settings for tile workers.
            ``None`` = use module globals.
        threads_per_worker: For RayBackend — threads per actor process.
            int: explicit count.  'auto': ``max(1, cpus // n_workers)``.
            ``None`` (default): no env override, system defaults apply.
            No effect on LocalBackend (workers are in-process).
        tiles_per_worker: V1 in-process packing (LocalBackend only).
            int: pack every *k* tile workers into one :class:`PackedTileWorker`.
            ``'auto'``: ``ceil(n_tiles / cpu_count())``.
            ``None`` (default): no packing; one actor per tile.
            Ignored silently for RayBackend (not yet implemented).
            The coordinator ``call_all`` API is unchanged — the workers list
            still has one entry per tile, backed by
            :class:`VirtualWorkerHandle` proxies when packing is active.
        use_step_columns: Enable A2 phase-folded step-column table on all
            workers (default True).  Propagated via ``TileWorker.configure``
            so Ray workers receive the setting even though module globals don't
            propagate.  Can be overridden per-solve via the ``use_step_columns``
            kwarg on ``solve_transient``/``solve_quasi_static``.
        max_table_mb: Per-worker memory budget for the phase-table (MB,
            default 512).  Tables estimated to exceed this fall back to the
            chunked-window tier.  Propagated via ``TileWorker.configure``.
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
        threads_per_worker=threads_per_worker,
        use_step_columns=use_step_columns,
        max_table_mb=max_table_mb,
        tiles_per_worker=tiles_per_worker,
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
    tile_data_dict: Optional[Dict[tuple, TileData]],
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
    tile_data_dict: Dict[tuple, TileData],
    output_dir: str,
) -> None:
    """Dump TileData + metadata to a directory (helper for legacy shim)."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for tid, td in tile_data_dict.items():
        tile_str = '_'.join(str(c) for c in tid)
        with open(out / f'tile_{tile_str}.pkl', 'wb') as f:
            pickle.dump(td, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(out / 'metadata.pkl', 'wb') as f:
        pickle.dump(
            {'metadata': metadata, 'boundary_nodes': boundary_nodes},
            f, protocol=pickle.HIGHEST_PROTOCOL,
        )


def _resolve_k(tiles_per_worker: Any, n: int) -> int:
    """Convert tiles_per_worker ('auto' or int) to a k value >= 1."""
    import math
    import os

    if tiles_per_worker == 'auto':
        k = math.ceil(n / max(1, os.cpu_count() or 1))
    else:
        k = int(tiles_per_worker)
    return max(1, k)


def _pack_workers(workers: List[Any], tiles_per_worker: Any, be: ComputeBackend) -> List[Any]:
    """Wrap *workers* into :class:`PackedTileWorker` groups (LocalBackend).

    Returns a list of :class:`VirtualWorkerHandle` objects of the same length
    as *workers*, one per tile, pointing into the appropriate packed actor.

    For :class:`RayBackend`, packing is handled upstream in
    :func:`_create_packed_ray_workers` before ``setup_from_pkl``.  Calling this
    function with a ``RayBackend`` is a no-op with a warning.
    """
    import math

    if isinstance(be, RayBackend):
        logger.warning(
            "tiles_per_worker is a no-op for RayBackend at this stage; "
            "Ray packing should be applied before setup_from_pkl.",
        )
        return workers

    n = len(workers)
    k = _resolve_k(tiles_per_worker, n)

    if k <= 1:
        # No-op: k=1 means one tile per actor (same as unpacked)
        return workers

    handles: List[Any] = []
    for i in range(0, n, k):
        batch = workers[i:i + k]
        packed = PackedTileWorker(batch)
        handles.extend(packed.handles())

    logger.info(
        "LocalBackend: packed %d tile workers into %d PackedTileWorker groups "
        "(tiles_per_worker=%s, k=%d)",
        n, math.ceil(n / k), tiles_per_worker, k,
    )
    return handles


def _create_packed_ray_workers(
    be: 'RayBackend',
    n_tiles: int,
    tiles_per_worker: Any,
) -> Optional[List[Any]]:
    """Create packed Ray actors for RayBackend when tiles_per_worker > 1.

    Returns a list of :class:`~distributed.backend.VirtualWorkerHandle` objects
    (one per tile) that route calls through ``PackedTileWorkerActor.call_worker``
    on the Ray side.  Returns *None* when k <= 1 (no packing needed).

    The caller must have already built *solver_settings* and *setup_args* (one
    per tile) and must call ``be.call_all(handles, ...)`` for configure and
    setup_from_pkl AFTER this function returns the handles.
    """
    import math

    from distributed.backend import PackedTileWorkerActor, VirtualWorkerHandle

    k = _resolve_k(tiles_per_worker, n_tiles)
    if k <= 1:
        return None

    tpw = be._resolve_threads_per_worker(math.ceil(n_tiles / k))
    RemotePacked = be._ray.remote(PackedTileWorkerActor)

    if tpw is not None:
        env_vars = {
            'OMP_NUM_THREADS': str(tpw),
            'OPENBLAS_NUM_THREADS': str(tpw),
            'MKL_NUM_THREADS': str(tpw),
        }
        def _make(batch_size):
            return RemotePacked.options(
                runtime_env={'env_vars': env_vars}
            ).remote(batch_size)
    else:
        def _make(batch_size):
            return RemotePacked.remote(batch_size)

    handles: List[Any] = []
    n_packed = 0
    for i in range(0, n_tiles, k):
        batch_size = min(k, n_tiles - i)
        packed_actor = _make(batch_size)
        n_packed += 1
        for j in range(batch_size):
            handles.append(VirtualWorkerHandle(packed_actor, j))

    logger.info(
        "RayBackend: packed %d tiles into %d Ray actors "
        "(tiles_per_worker=%s, k=%d, threads_per_worker=%s)",
        n_tiles, n_packed, tiles_per_worker, k, tpw,
    )
    return handles


def _create_distributed_model_from_bundle(
    bundle: ParsedTileBundle,
    backend: str = 'local',
    coordinator_solver_config: Optional[SolverBackendConfig] = None,
    worker_solver_config: Optional[SolverBackendConfig] = None,
    threads_per_worker: Any = None,
    use_step_columns: bool = True,
    max_table_mb: float = 512.0,
    tiles_per_worker: Any = None,
    **backend_kwargs,
) -> DistributedPowerGridModel:
    """Core factory: create model from a ParsedTileBundle.

    Single code path: workers load their TileData from pkl files via
    ``setup_from_pkl()``.
    """
    t_start = time.perf_counter()
    metadata = bundle.metadata

    # 1. Create backend
    be = _init_backend(backend, backend_kwargs, threads_per_worker=threads_per_worker)

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

    # 3. Build solver settings and per-tile setup args (needed for all paths)
    from pgmath.block_system import get_partial_factor_reg_resistance
    from pgmath.factor import SolverBackendConfig as _SBC

    # Build worker settings dict: use explicit config or snapshot globals.
    # partial_factor_reg_ohms is a separate concern (block_system.py),
    # not included in SolverBackendConfig; always inherited from globals.
    # A2 step-column settings (use_step_columns, max_table_mb) must be
    # propagated here so Ray workers receive them — module-level globals
    # do NOT propagate to Ray worker processes.
    solver_settings = {
        **(worker_solver_config or _SBC.from_globals()).to_dict(),
        'partial_factor_reg_ohms': get_partial_factor_reg_resistance(),
        'use_step_columns': use_step_columns,
        'max_table_mb': max_table_mb,
    }

    pkl_path = Path(bundle.pkl_dir)
    setup_args = [
        (str(pkl_path / f'tile_{"_".join(str(c) for c in tc.tile_id)}.pkl'), interface_nodes)
        for tc in metadata.tile_configs
    ]

    n_tiles = len(metadata.tile_configs)

    if tiles_per_worker is not None and isinstance(be, RayBackend):
        # RayBackend packing: create packed Ray actors BEFORE setup_from_pkl
        # so that n_tiles / k Ray actors are spawned, each owning k in-process
        # TileWorkers.  VirtualWorkerHandle proxies are used as the workers list;
        # be.call_all routes each call via PackedTileWorkerActor.call_worker.remote.
        packed_handles = _create_packed_ray_workers(be, n_tiles, tiles_per_worker)
        if packed_handles is not None:
            workers = packed_handles
        else:
            # k <= 1: fall back to standard one-actor-per-tile
            workers = be.create_actors(TileWorker, metadata.tile_configs)
        # configure + setup route through VirtualWorkerHandle → call_worker.remote
        be.call_all(workers, 'configure', [(solver_settings,)] * len(workers))
        setup_results = be.call_all(workers, 'setup_from_pkl', setup_args)
        tiles_per_worker = None  # Packing done; skip the LocalBackend step below.
    else:
        # Standard path: one actor (in-process or Ray) per tile.
        workers = be.create_actors(TileWorker, metadata.tile_configs)
        be.call_all(workers, 'configure', [(solver_settings,)] * len(workers))
        setup_results = be.call_all(workers, 'setup_from_pkl', setup_args)

    # 5. Collect results
    (tile_boundary_nodes, tile_interior_counts,
     island_stats, tile_kept_nonlargest_iface) = _collect_setup_results(setup_results)

    # 6. Optional: pack multiple tile workers into one physical actor (LocalBackend)
    # For RayBackend this was already done above (tiles_per_worker set to None).
    if tiles_per_worker is not None:
        workers = _pack_workers(workers, tiles_per_worker, be)

    total_interior = sum(tile_interior_counts.values())
    total_boundary = sum(len(v) for v in tile_boundary_nodes.values())
    t_elapsed = time.perf_counter() - t_start
    logger.info(
        f"Model created in {t_elapsed:.3f}s: {len(workers)} tile-worker handles, "
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
        pkl_dir=bundle.pkl_dir,
    )

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

    # Stage 1e: parse-time connectivity summaries + the interface set they
    # were computed against.  ``None`` for bundles parsed before Stage 1e
    # (or loaded via the legacy no-pkl shim) -- always means "no summaries
    # available", forcing the legacy Schur-BFS fallback at model creation.
    connectivity_summary_version: Optional[int] = None
    parser_interface_set: Optional[Set[str]] = None
    component_summaries: Optional[List[Dict[str, Any]]] = None

    # Finding R4: parse-time WHOLE-TILE pre-clean removal diagnostics for
    # tiles that were subsequently B1-split, keyed by the PARENT tile's own
    # (x, y) id.  ``{'islands_removed': int, 'removed_nodes': Set[str]}``.
    # The parent tile no longer exists as a worker after splitting, so this
    # is surfaced as a SYNTHETIC island_stats[parent_id] entry at model
    # creation (model._merge_parent_removal_diagnostics) instead of being
    # misattributed to whichever sub-tile happens to be first.
    parent_removal_diagnostics: Optional[Dict[tuple, Dict[str, Any]]] = None


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

    # Stage 1e: resolved ONCE at model creation (never re-derived downstream).
    # 'summaries' -- workers skipped _remove_floating_islands (trusted
    #   pre_cleaned_full) and prepare() should use
    #   pgmath.schur.detect_interface_islands_from_summaries instead of the
    #   Schur-BFS.  'schur_bfs' -- full legacy path (workers ran removal;
    #   prepare() must run the BFS).  See model._resolve_island_detection for
    #   the fallback matrix (missing summaries / version mismatch / trust
    #   assertion failure / explicit override all resolve to 'schur_bfs').
    island_detection_mode: str = field(default='schur_bfs', repr=False)
    # Non-None only when island_detection_mode == 'summaries'.
    component_summaries: Optional[List[Dict[str, Any]]] = field(default=None, repr=False)

    # Per-role solver backend configs (None = use module globals)
    coordinator_solver_config: Optional[SolverBackendConfig] = field(default=None, repr=False)
    worker_solver_config: Optional[SolverBackendConfig] = field(default=None, repr=False)

    # B2/Stage 1: Coordinator-side solver settings dict.
    # Keys recognised:
    #   'interface_solver': 'direct' | 'cg' | 'auto' (default 'auto')
    #   'interface_matvec_mode': 'auto' | 'assembled' | 'tilewise' (default
    #       'auto' -- resolves to 'tilewise' whenever per-tile dense Schur
    #       blocks are available, else 'assembled'; see
    #       interface_iterative.resolve_matvec_mode). This is also the
    #       precondition the 'interface_preconditioner' default below keys
    #       off (two_level only activates for CG + tilewise).
    #   'interface_preconditioner': 'auto' | 'block_jacobi' | 'jacobi' |
    #       'none' | 'amg' | 'two_level' (default 'auto' -- Stage 3: resolves
    #       to 'two_level' for CG+tilewise, else 'block_jacobi'; see
    #       interface_iterative.resolve_preconditioner)
    #   'interface_coarse_geneo_k': int (default 4; Stage 3 two_level knob)
    #   'interface_coarse_geneo_tol': float (default 1e-6)
    #   'interface_coarse_eps_rank': float (default 1e-12)
    #   'interface_coarse_max_cols': int (default 4096)
    #   'interface_coarse_max_bytes': 'auto' | int bytes (default 'auto' =
    #       min(8 GB, 0.1 * total RAM) via psutil) -- byte-based guard on the
    #       coarse build's dense (n x T') arrays, distinct from
    #       interface_coarse_max_cols (a column-count cap that does not
    #       scale with n)
    #   'interface_cg_rtol': float (default 1e-8; Stage 0 sweep validated)
    #   'interface_cg_atol': float (default 1e-14)
    #   'interface_cg_maxiter': int | None (default None -> 3 * n_interface)
    #   'interface_cg_strict': bool (default True)
    #   'interface_factor_memory_budget': 'auto' | int bytes
    #       (default 'auto' = min(32 GB, 0.4 * total RAM) via psutil)
    #   'interface_block_jacobi_max_bytes': 'auto' | int bytes
    #       (default 'auto' = min(8 GB, 0.1 * total RAM) via psutil)
    #   'interface_drop_s_global': bool (default False) -- opt-in "never
    #       assemble S_global" factor path (item 3). Requires explicit
    #       interface_solver='cg', interface_matvec_mode in
    #       ('tilewise', 'auto'), and summaries-based island detection;
    #       falls back to the normal S_global-assembling path with a WARNING
    #       when preconditions aren't met. Stage 2 scope: DC-only
    #       (_factor_dc_context). Transient factor (_factor_transient_context)
    #       always assembles S_global normally and emits a WARNING if this is
    #       set True -- TD never-assemble lands in a later stage.
    # These affect only the coordinator; they are NOT propagated to workers
    # (Ray module-globals do not propagate -- see CLAUDE.md pitfalls).
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

    # Stage 1e: connectivity summaries are additive keys -- bundles written
    # before Stage 1e (or by the legacy no-pkl shim) simply lack them, and
    # `.get(..., None)` degrades to "no summaries available" exactly as if
    # this were an old-format metadata.pkl (see model._resolve_island_detection).
    connectivity_summary_version = meta_bundle.get('connectivity_summary_version')
    parser_interface_set = meta_bundle.get('parser_interface_set')
    component_summaries = meta_bundle.get('component_summaries')
    parent_removal_diagnostics = meta_bundle.get('parent_removal_diagnostics')

    logger.info(
        f"Loaded metadata from {pkl_dir}: {len(metadata.tile_configs)} tiles, "
        f"{len(boundary_nodes)} shared boundary nodes"
        + (
            f", {len(component_summaries)} connectivity summaries"
            if component_summaries is not None else ""
        )
    )

    return ParsedTileBundle(
        metadata=metadata,
        shared_boundary_nodes=boundary_nodes,
        pkl_dir=str(pkl_path),
        connectivity_summary_version=connectivity_summary_version,
        parser_interface_set=parser_interface_set,
        component_summaries=component_summaries,
        parent_removal_diagnostics=parent_removal_diagnostics,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _init_backend(
    backend,
    backend_kwargs: Dict[str, Any],
    threads_per_worker: Any = None,
) -> ComputeBackend:
    """Create and initialize compute backend.

    Args:
        backend: 'local', 'ray', 'task', or a ComputeBackend instance.
            When a ComputeBackend instance is passed directly (e.g.,
            TaskDataflowBackend()), it is initialized with backend_kwargs
            and returned as-is. This supports B4 task-based backends
            without modifying the existing string-dispatch path.
        backend_kwargs: Extra kwargs passed to backend.initialize()
        threads_per_worker: For RayBackend — number of BLAS/OMP threads per
            actor process (int), or 'auto' for ``max(1, cpus // n_workers)``.
            Has no effect on LocalBackend (workers run in-process).
    """
    # Accept ComputeBackend instances directly (B4 task backend and future backends)
    if isinstance(backend, ComputeBackend):
        be = backend
        if not getattr(be, '_initialized', False):
            be.initialize(**backend_kwargs)
        return be
    if backend == 'ray':
        be = RayBackend(threads_per_worker=threads_per_worker)
        be.initialize(**backend_kwargs)
    elif backend == 'task':
        # B4: task-based dataflow backend (alias for TaskDataflowBackend())
        from .task_backend import TaskDataflowBackend
        be = TaskDataflowBackend(threads_per_worker=threads_per_worker)
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


def _merge_parent_removal_diagnostics(
    island_stats: Dict[tuple, Dict],
    bundle: 'ParsedTileBundle',
) -> None:
    """Finding R4: surface parse-time WHOLE-TILE pre-clean removals (paid
    BEFORE a tile was B1-split) under a synthetic entry keyed by the
    PARENT's own (x, y) tile id, instead of misattributing them to
    sub_tiles[0] (whose spatial region may have nothing to do with where
    the removed components actually lay).

    Mode-independent: ``bundle.parent_removal_diagnostics`` is parse-time
    bundle metadata, unaffected by which island_detection mode a later
    model creation chooses -- so calling this after EVERY
    ``_collect_setup_results`` (both the initial setup and the F10
    re-setup-with-trust-off path) keeps 'auto'/'summaries' and 'schur_bfs'
    reporting IDENTICAL totals for the same bundle, matching finding R2's
    cross-mode equality requirement.  In-place update of *island_stats*.
    """
    for parent_tid, diag in (bundle.parent_removal_diagnostics or {}).items():
        island_stats[parent_tid] = {
            'islands_removed': diag.get('islands_removed', 0),
            'is_parent_synthetic_entry': True,
        }


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
    island_detection: str = 'auto',
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
        island_detection: ``'auto'`` (default) | ``'summaries'`` | ``'schur_bfs'``.
            Decided ONCE here, all-new-or-all-legacy (Stage 1e).  ``'auto'``
            and ``'summaries'`` both attempt the parse-time-summary fast path
            (union-find island detection, worker-side removal skipped) and
            fall back to the legacy Schur-BFS path when the bundle lacks
            summaries, the summary version is stale, or the trust assertion
            (``interface_nodes == bundle.parser_interface_set``) fails.
            ``'schur_bfs'`` forces the legacy path unconditionally.  See
            ``model._resolve_island_detection`` for the full fallback matrix.
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
        island_detection=island_detection,
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

    if not getattr(be, 'supports_inprocess_packing', False):
        logger.warning(
            "tiles_per_worker is a no-op for %s (remote-actor backend); "
            "packing should be applied before setup_from_pkl or is not supported.",
            type(be).__name__,
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


def _resolve_island_detection(
    bundle: ParsedTileBundle,
    interface_nodes: Set[str],
    island_detection: str,
) -> Tuple[bool, str]:
    """Decide ONCE whether to trust the bundle's Stage 1e connectivity summaries.

    All-new-or-all-legacy (never a mix): returns ``(trust, resolved_mode)``
    where ``resolved_mode`` is ``'summaries'`` (workers skip
    ``_remove_floating_islands``; ``prepare()`` uses the union-find) or
    ``'schur_bfs'`` (full legacy path).  ``trust`` is the boolean threaded to
    workers via ``TileWorker.configure({'pre_cleaned_full_trusted': trust})``.

    Fallback matrix (any one of these forces ``'schur_bfs'``):
      * ``island_detection == 'schur_bfs'`` (explicit override).
      * ``bundle.component_summaries is None`` (no summaries -- old bundle,
        or parsed via the legacy no-pkl shim).
      * ``bundle.connectivity_summary_version != CONNECTIVITY_SUMMARY_VERSION``
        (stale summaries from a bundle parsed before a summary-semantics change).
      * ``interface_nodes != bundle.parser_interface_set`` (trust assertion
        failure -- guards the .ckt-rescan fallback, model.py interface-node
        derivation drift, or a stale/edited metadata.pkl).
    """
    from .parser import CONNECTIVITY_SUMMARY_VERSION

    if island_detection == 'schur_bfs':
        logger.info(
            "island_detection='schur_bfs' (explicit): using the legacy "
            "worker-side island removal + coordinator Schur-BFS path."
        )
        return False, 'schur_bfs'

    if island_detection not in ('auto', 'summaries'):
        raise ValueError(
            f"Unknown island_detection setting: {island_detection!r}. "
            "Expected 'auto', 'summaries', or 'schur_bfs'."
        )

    if bundle.component_summaries is None or bundle.parser_interface_set is None:
        logger.info(
            "island_detection=%r: bundle has no Stage 1e connectivity "
            "summaries (parsed before Stage 1e, or via the legacy no-pkl "
            "path); falling back to the legacy Schur-BFS island detection. "
            "Re-parse (DistributedNetlistParser.parse_and_dump) to enable "
            "the fast summaries path.",
            island_detection,
        )
        return False, 'schur_bfs'

    if bundle.connectivity_summary_version != CONNECTIVITY_SUMMARY_VERSION:
        logger.info(
            "island_detection=%r: bundle connectivity_summary_version=%s != "
            "current CONNECTIVITY_SUMMARY_VERSION=%s; falling back to the "
            "legacy Schur-BFS island detection. Re-parse to refresh the "
            "connectivity summaries.",
            island_detection, bundle.connectivity_summary_version,
            CONNECTIVITY_SUMMARY_VERSION,
        )
        return False, 'schur_bfs'

    if interface_nodes != bundle.parser_interface_set:
        n_sym_diff = len(interface_nodes ^ bundle.parser_interface_set)
        logger.warning(
            "island_detection=%r: trust assertion FAILED -- model-creation "
            "interface_nodes (%d nodes) != bundle.parser_interface_set "
            "(%d nodes), symmetric difference=%d. Falling back to the legacy "
            "Schur-BFS island detection for correctness. This can happen if "
            "metadata.pkl was edited/stale, or model.py's interface-node "
            "derivation has drifted from the parser's.",
            island_detection, len(interface_nodes),
            len(bundle.parser_interface_set), n_sym_diff,
        )
        return False, 'schur_bfs'

    return True, 'summaries'


def _verify_summary_structural_completeness(
    component_summaries: Optional[List[Dict[str, Any]]],
    interface_nodes: Set[str],
    tile_boundary_nodes: Dict[tuple, List[str]],
) -> bool:
    """Stage 1e finding F10: runtime guarantee that summaries structurally
    cover every (tile, interface-port) incidence.

    The Stage 1e exactness argument (pgmath.schur module docstring) ASSUMES
    that every port a tile worker actually reports (``boundary_nodes``,
    intersected with the finalized ``interface_nodes``) is also present as an
    interface-candidate in that tile's own connectivity summaries.  Findings
    F1/F5 fixed the two concrete ways this assumption could be violated;
    this function turns the assumption into a cheap, enforced runtime check
    (no extra worker round-trip -- ``tile_boundary_nodes`` was already
    returned by the ``setup_from_pkl`` RPC) instead of trusting it blindly,
    so any FUTURE regression in the pre-clean/summary machinery degrades to
    the legacy Schur-BFS path (loudly) rather than corrupting results.

    Known residuals (verification-reviewed, accepted):
    - Split-induced-shared nodes: a node '*'-declared in only ONE tile (so
      absent from the step-0 scan's shared set) that becomes interface only
      because that tile was B1-split, while sitting unmarked inside a
      DIFFERENT unsplit tile's floating fragment, can have that fragment
      removed at parse time.  This is NOT a Stage 1e regression -- legacy
      worker-time removal deletes the identical fragment (same single
      cross-tile candidate, same threshold) -- and a true divergence would
      need >=4 such split-induced-shared unmarked nodes in one fragment.
      Removed-fragment interiors are absent from ``all_nodes``, so this
      port-coverage check cannot (and need not) see them.
    - Perf, not correctness: a bundle that genuinely trips this check pays
      one extra worker re-setup and permanently runs the legacy Schur-BFS
      (238 s proxy / ~900 s BRCM) instead of the fast union-find -- correct
      but slow, and the INFO/WARNING logs say so.

    Returns:
        True if every tile's worker-reported ports are a subset of that
        tile's summarized interface-candidates (or *component_summaries* is
        None, i.e. summaries were never trusted in the first place). False on
        any violation (caller should route the whole model to legacy).
    """
    if component_summaries is None:
        return True

    per_tile_candidates: Dict[tuple, Set[str]] = {}
    for summary in component_summaries:
        tid = summary.get('tile_id')
        if tid is None:
            continue
        per_tile_candidates.setdefault(tuple(tid), set()).update(
            summary.get('candidates', ())
        )

    ok = True
    for tid, boundary_list in tile_boundary_nodes.items():
        worker_ports = set(boundary_list) & interface_nodes
        if not worker_ports:
            continue
        summarized = per_tile_candidates.get(tuple(tid), set())
        missing = worker_ports - summarized
        if missing:
            ok = False
            _preview = sorted(missing)[:5]
            logger.warning(
                "Stage 1e structural-completeness check FAILED for tile %s: "
                "%d worker-reported port(s) not present in that tile's "
                "summarized interface candidates (e.g. %s%s). Summaries are "
                "structurally incomplete for this bundle -- routing the "
                "WHOLE model to the legacy Schur-BFS island-detection path "
                "for correctness.",
                tid, len(missing), _preview,
                ' ...' if len(missing) > len(_preview) else '',
            )

    return ok


def _create_distributed_model_from_bundle(
    bundle: ParsedTileBundle,
    backend: str = 'local',
    coordinator_solver_config: Optional[SolverBackendConfig] = None,
    worker_solver_config: Optional[SolverBackendConfig] = None,
    threads_per_worker: Any = None,
    use_step_columns: bool = True,
    max_table_mb: float = 512.0,
    tiles_per_worker: Any = None,
    island_detection: str = 'auto',
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

    # 2. Compute interface nodes.
    # Finding F13: single-sourced formula (tile_parsing.compute_interface_nodes),
    # also used by DistributedNetlistParser.parse_and_dump's parser_interface_set
    # -- must mirror it exactly or the Stage 1e trust assertion below spuriously
    # fails and silently disables the summaries fast path.
    from .tile_parsing import compute_interface_nodes
    die_net_map = getattr(metadata.package_data, 'die_attachment_net_map', {})
    interface_nodes = compute_interface_nodes(
        bundle.shared_boundary_nodes, die_net_map,
        metadata.package_data.die_attachment_nodes,
    )
    logger.info(
        f"Interface: {len(interface_nodes)} nodes "
        f"({len(bundle.shared_boundary_nodes)} boundary + "
        f"{len(metadata.package_data.die_attachment_nodes)} die attachment)"
    )

    # 2b. Stage 1e: decide ONCE whether to trust the bundle's connectivity
    # summaries (all-new-or-all-legacy).  Must happen here (not later, via
    # model.settings) because the trust flag needs to reach workers'
    # setup_from_pkl() below via the SAME configure() call as everything else.
    island_trust, island_detection_mode = _resolve_island_detection(
        bundle, interface_nodes, island_detection,
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
        # Stage 1e: reaches TileWorker.configure() -> _build_block_system,
        # which skips _remove_floating_islands only when BOTH this is True
        # AND the tile's own TileData.pre_cleaned_full is True.
        'pre_cleaned_full_trusted': island_trust,
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
    _merge_parent_removal_diagnostics(island_stats, bundle)

    # 5b. Finding F10: enforce (not just assume) that the summaries
    # structurally cover every (tile, interface-port) incidence, now that we
    # have the worker-reported port sets for free (no extra round-trip).
    # Only meaningful when trust was granted (workers actually skipped
    # _remove_floating_islands based on it); if trust already resolved to
    # 'schur_bfs' there's nothing to double-check.  On violation, workers
    # already ran with removal skipped based on stale trust -- re-run setup
    # with the trust flag OFF so _remove_floating_islands actually executes,
    # and downgrade the model to the full legacy path end-to-end (never a
    # mix of trusted-worker-state + legacy island detection).
    if island_trust and not _verify_summary_structural_completeness(
        bundle.component_summaries, interface_nodes, tile_boundary_nodes,
    ):
        island_trust = False
        island_detection_mode = 'schur_bfs'
        solver_settings['pre_cleaned_full_trusted'] = False
        be.call_all(workers, 'configure', [(solver_settings,)] * len(workers))
        setup_results = be.call_all(workers, 'setup_from_pkl', setup_args)
        (tile_boundary_nodes, tile_interior_counts,
         island_stats, tile_kept_nonlargest_iface) = _collect_setup_results(setup_results)
        _merge_parent_removal_diagnostics(island_stats, bundle)

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
        f"{len(interface_nodes)} interface nodes, island_detection={island_detection_mode}"
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
        island_detection_mode=island_detection_mode,
        component_summaries=(bundle.component_summaries if island_trust else None),
    )

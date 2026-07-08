"""Factorization, checkpoint, and refactor helpers for distributed contexts.

Extracted from result.py to keep file sizes manageable (mixin pattern).
All functions take the context as the first argument and mutate it in place,
mirroring the original method bodies exactly.
"""

from __future__ import annotations

import logging
import os
import pickle
import time as _time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

import numpy as np
import scipy.sparse as sp

if TYPE_CHECKING:
    from .model import DistributedPowerGridModel
    from .result import (
        DistributedSolverContext,
        DistributedTopologyContext,
        DistributedTransientContext,
    )

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _save_role_configs(model: Optional['DistributedPowerGridModel']) -> dict:
    """Extract per-role solver configs from model for checkpoint serialization.

    Serializes configs as plain dicts (via ``to_dict()``) rather than raw
    dataclass instances, ensuring forward compatibility if fields change.
    """
    if model is None:
        return {'coordinator_solver_config': None, 'worker_solver_config': None}
    return {
        'coordinator_solver_config': (
            model.coordinator_solver_config.to_dict()
            if model.coordinator_solver_config is not None else None
        ),
        'worker_solver_config': (
            model.worker_solver_config.to_dict()
            if model.worker_solver_config is not None else None
        ),
    }


def _restore_role_configs(
    data: dict, model: Optional['DistributedPowerGridModel'],
) -> None:
    """Restore per-role solver configs from checkpoint data onto model.

    Only restores a config if the model's field is currently None
    (i.e., don't overwrite explicitly-set configs).  Accepts both
    plain dicts (new format) and ``SolverBackendConfig`` instances
    (legacy pickled checkpoints).
    """
    if model is None:
        return
    from pgmath.factor import SolverBackendConfig

    for attr in ('coordinator_solver_config', 'worker_solver_config'):
        saved = data.get(attr)
        if saved is not None and getattr(model, attr) is None:
            if isinstance(saved, dict):
                saved = SolverBackendConfig.from_dict(saved)
            setattr(model, attr, saved)


def _default_checkpoint_path(
    ctx: 'DistributedSolverContext | DistributedTransientContext',
    filename: str,
) -> str:
    """Derive default checkpoint path from model metadata."""
    if (
        ctx.model is not None
        and ctx.model.metadata.tile_configs
    ):
        ckt_path = ctx.model.metadata.tile_configs[0].ckt_path
        parent = os.path.dirname(os.path.dirname(ckt_path))
        return os.path.join(parent, 'checkpoint', filename)
    return os.path.join('.', 'checkpoint', filename)


# ---------------------------------------------------------------------------
# DC context: factor / save / load / refactor
# ---------------------------------------------------------------------------


def _factor_dc_context(ctx: 'DistributedSolverContext', verbose: bool = False) -> None:
    """Factor tiles + assemble/factor interface system (DC). Populates ctx in place."""
    from .result import DistributedTopologyContext

    if ctx.model is None:
        raise RuntimeError(
            "Cannot factor without a model reference. "
            "Either pass model= to __init__ or set self.model."
        )
    timings: Dict[str, Any] = {}
    model = ctx.model

    # 1. Factor tiles and compute Schur complements (parallel on workers)
    t0 = _time.perf_counter()
    schur_results = model.backend.call_all(
        model.workers, 'factor_and_compute_schur'
    )
    timings['factor_tiles'] = _time.perf_counter() - t0

    # Organize results by tile
    tile_schur_complements: Dict[Any, np.ndarray] = {}
    tile_port_node_lists: Dict[Any, List[str]] = {}
    per_tile_stats: List[Dict[str, Any]] = []

    for i, (S_i, boundary_list, tile_stats) in enumerate(schur_results):
        tid = model.metadata.tile_configs[i].tile_id
        tile_schur_complements[tid] = S_i
        tile_port_node_lists[tid] = boundary_list
        per_tile_stats.append(tile_stats)

    # Coordinator-side DEBUG: per-tile factor/schur details
    from pgmath.block_system import _format_bytes
    for i, ts in enumerate(per_tile_stats):
        tid = model.metadata.tile_configs[i].tile_id
        n_ii = ts.get('n_interior', 0)
        n_pp = ts.get('n_ports', 0)
        G_ii_nnz = ts.get('G_ii_nnz', 0)
        density = (G_ii_nnz / (n_ii * n_ii) * 100) if n_ii > 0 else 0.0
        logger.debug(
            "Tile %s factor_and_compute_schur:\n"
            "  G_ii: %s x %s, nnz=%s (density %.5f%%), %s\n"
            "  G_pp: %s x %s, nnz=%s\n"
            "  G_pi: %s x %s, nnz=%s  |  G_ip: %s x %s, nnz=%s\n"
            "  Block system memory: %s\n"
            "  factor_interior: %.3fs  |  backend: %s\n"
            "  compute_schur: %.3fs  |  Schur: %s dense (%s)",
            tid,
            f"{n_ii:,}", f"{n_ii:,}", f"{G_ii_nnz:,}", density,
            _format_bytes(ts.get('mem_bytes', 0)),
            f"{n_pp:,}", f"{n_pp:,}", f"{ts.get('G_pp_nnz', 0):,}",
            f"{n_pp:,}", f"{n_ii:,}", f"{ts.get('G_pi_nnz', 0):,}",
            f"{n_ii:,}", f"{n_pp:,}", f"{ts.get('G_ip_nnz', 0):,}",
            _format_bytes(ts.get('mem_bytes', 0)),
            ts.get('factor_interior_s', 0), ts.get('factorization_backend_info', 'n/a'),
            ts.get('compute_schur_s', 0),
            "%s x %s" % (f"{ts.get('schur_shape', (0, 0))[0]:,}",
                         f"{ts.get('schur_shape', (0, 0))[1]:,}"),
            _format_bytes(ts.get('schur_mem_bytes', 0)),
        )

    # 2. Assemble global interface system
    t0 = _time.perf_counter()
    from pgmath.schur import assemble_schur_complement_system

    S_global, rhs_dirichlet, interface_nodes, interface_node_to_idx = (
        assemble_schur_complement_system(
            tile_schur_complements=tile_schur_complements,
            tile_port_node_lists=tile_port_node_lists,
            extra_edges=model.package_data.package_edges,
            dirichlet_nodes=model.pad_nodes,
            dirichlet_voltage=model.vdd,
        )
    )
    timings['assemble_interface'] = _time.perf_counter() - t0

    # 2b. Global interface island detection
    t0 = _time.perf_counter()
    from pgmath.schur import detect_interface_islands

    S_global, rhs_dirichlet, island_nodes = detect_interface_islands(
        S_global, rhs_dirichlet, interface_nodes, interface_node_to_idx,
        pad_nodes=model.pad_nodes,
        extra_edges=model.package_data.package_edges,
        dirichlet_voltage=model.vdd,
    )
    timings['detect_interface_islands'] = _time.perf_counter() - t0

    if island_nodes:
        logger.warning(
            "Penalized %d interface island nodes (shorted to %.3f V)",
            len(island_nodes), model.vdd,
        )

    if verbose and island_nodes:
        # Cross-check: tile-level kept non-largest components vs global islands
        tile_flagged: Set[str] = set()
        for nodes in model.tile_kept_nonlargest_iface.values():
            tile_flagged.update(nodes)

        confirmed = tile_flagged & island_nodes
        saved = tile_flagged - island_nodes
        if confirmed:
            logger.info(
                "%d tile-flagged interface nodes confirmed as global islands",
                len(confirmed),
            )
        if saved:
            logger.info(
                "%d tile-flagged interface nodes connected to pads through other tiles",
                len(saved),
            )

    # 3. Factor interface system
    t0 = _time.perf_counter()
    from pgmath.factor import _factor_conductance_matrix

    interface_lu_result = _factor_conductance_matrix(
        S_global, verbose=False, config=model.coordinator_solver_config,
    )
    timings['factor_interface'] = _time.perf_counter() - t0

    # 4. Build tile index maps (local port indices -> global interface indices)
    tile_index_maps: Dict[Tuple[int, int], np.ndarray] = {}
    for tid, boundary_list in tile_port_node_lists.items():
        local_to_global = np.array(
            [interface_node_to_idx[n] for n in boundary_list if n in interface_node_to_idx],
            dtype=np.int32,
        )
        tile_index_maps[tid] = local_to_global

    timings['total_prepare'] = sum(timings.values())

    # --- Build solver_stats ---
    from pgmath.block_system import _sparse_mem_bytes, _format_bytes as _fb

    # Interface system stats
    n_unknowns = S_global.shape[0]
    iface_nnz = S_global.nnz
    iface_density_pct = (
        (iface_nnz / (n_unknowns * n_unknowns) * 100)
        if n_unknowns > 0 else 0.0
    )
    iface_mem_bytes = _sparse_mem_bytes(S_global)

    interface_stats: Dict[str, Any] = {
        'n_unknowns': n_unknowns,
        'nnz': iface_nnz,
        'density_pct': iface_density_pct,
        'mem_bytes': iface_mem_bytes,
        'factor_time_s': timings['factor_interface'],
        'backend': interface_lu_result.backend,
        'backend_info': interface_lu_result.backend_info,
        'islands_penalized': len(island_nodes),
    }

    # Aggregate per-tile stats (min/mean/max)
    from distributed.solver import _minmeanmax
    aggregate_stats: Dict[str, Any] = {}
    if per_tile_stats:
        for key in ('n_interior', 'n_ports', 'G_ii_nnz', 'mem_bytes',
                    'schur_mem_bytes', 'factor_interior_s'):
            values = [s.get(key, 0) for s in per_tile_stats]
            lo, avg, hi = _minmeanmax(values)
            aggregate_stats[key] = {'min': lo, 'mean': avg, 'max': hi,
                                    'total': float(np.sum(values))}

    solver_stats: Dict[str, Any] = {
        'per_tile': per_tile_stats,
        'interface': interface_stats,
        'aggregate': aggregate_stats,
    }
    timings['solver_stats'] = solver_stats

    # --- Verbose INFO logging ---
    if verbose:
        from distributed.solver import _fmt_count
        n_tiles = len(per_tile_stats)
        logger.info("=== Distributed DDM Prepare Statistics ===")
        logger.info("Tiles: %d", n_tiles)
        if aggregate_stats:
            _ag = aggregate_stats
            lo, avg, hi = _ag['n_interior']['min'], _ag['n_interior']['mean'], _ag['n_interior']['max']
            logger.info("  Interior nodes:  %s / %s / %s  (min/mean/max)",
                        _fmt_count(lo), _fmt_count(avg), _fmt_count(hi))
            lo, avg, hi = _ag['n_ports']['min'], _ag['n_ports']['mean'], _ag['n_ports']['max']
            logger.info("  Port nodes:      %s / %s / %s",
                        _fmt_count(lo), _fmt_count(avg), _fmt_count(hi))
            lo, avg, hi = _ag['G_ii_nnz']['min'], _ag['G_ii_nnz']['mean'], _ag['G_ii_nnz']['max']
            logger.info("  G_ii nnz:        %s / %s / %s",
                        _fmt_count(lo), _fmt_count(avg), _fmt_count(hi))
            lo, avg, hi = _ag['mem_bytes']['min'], _ag['mem_bytes']['mean'], _ag['mem_bytes']['max']
            total = _ag['mem_bytes']['total']
            logger.info("  G_ii memory:     %s / %s / %s  (total: %s)",
                        _fb(lo), _fb(avg), _fb(hi), _fb(total))
            lo, avg, hi = _ag['schur_mem_bytes']['min'], _ag['schur_mem_bytes']['mean'], _ag['schur_mem_bytes']['max']
            total = _ag['schur_mem_bytes']['total']
            logger.info("  Schur memory:    %s / %s / %s  (total: %s)",
                        _fb(lo), _fb(avg), _fb(hi), _fb(total))
            lo, avg, hi = _ag['factor_interior_s']['min'], _ag['factor_interior_s']['mean'], _ag['factor_interior_s']['max']
            logger.info("  Factor time:     %.3fs / %.3fs / %.3fs", lo, avg, hi)
            # Backend: use the first tile's backend_info as representative
            backend_info = per_tile_stats[0].get('factorization_backend_info', 'n/a')
            logger.info("  Factor backend:  %s", backend_info)
            logger.info("  Factor + Schur wall time: %.3fs", timings['factor_tiles'])

        logger.info("Interface system: %s unknowns, %s nnz (density %.3f%%), %s",
                    _fmt_count(n_unknowns), _fmt_count(iface_nnz),
                    iface_density_pct, _fb(iface_mem_bytes))
        logger.info("  Backend: %s", interface_lu_result.backend_info)
        logger.info("  Factor time: %.3fs", timings['factor_interface'])
        logger.info("  Islands penalized: %d", len(island_nodes))
        logger.info("  Assemble time: %.3fs", timings['assemble_interface'])
        logger.info("  Detect islands time: %.3fs", timings['detect_interface_islands'])
        logger.info("=== Total Prepare: %.3fs ===", timings['total_prepare'])

    # 5. Build package G matrix for topology
    from pgmath.schur import build_interface_package_matrices
    n_interface = len(interface_nodes)
    G_pkg_uu, _ = build_interface_package_matrices(
        package_edges=model.package_data.package_edges,
        package_cap_edges=model.package_data.package_cap_edges,
        interface_node_to_idx=interface_node_to_idx,
        n_interface=n_interface,
        dirichlet_nodes=model.pad_nodes,
    )

    # Populate context fields
    ctx._interface_lu = interface_lu_result.solve
    ctx._interface_nodes = interface_nodes
    ctx._interface_node_to_idx = interface_node_to_idx
    ctx._rhs_dirichlet_interface = rhs_dirichlet
    ctx._tile_index_maps = tile_index_maps
    ctx._removed_interface_nodes = island_nodes
    ctx._S_global = S_global
    ctx.timings = timings

    # Build topology context (if not already provided)
    if ctx.topology is None:
        ctx.topology = DistributedTopologyContext(
            interface_nodes=interface_nodes,
            interface_node_to_idx=interface_node_to_idx,
            tile_index_maps=tile_index_maps,
            rhs_dirichlet_G=rhs_dirichlet,
            G_package_uu=G_pkg_uu if G_pkg_uu.nnz > 0 else None,
            removed_interface_nodes=island_nodes,
        )

    ctx.is_factored = True


def _save_dc_context(ctx: 'DistributedSolverContext', path: Optional[str] = None) -> str:
    """Save DC context metadata (topology, S_global, timings) to disk."""
    if path is None:
        path = _default_checkpoint_path(ctx, 'dc_context.pkl')

    if ctx._S_global is None:
        raise RuntimeError(
            "Cannot save: S_global is None (was release() called before "
            "save()?). Save before release(), or re-factor first."
        )

    save_data = {
        'type': 'DistributedSolverContext',
        'version': 1,
        'topology': ctx.topology,
        'S_global': ctx._S_global,
        'timings': ctx.timings,
        **_save_role_configs(ctx.model),
    }

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(save_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    logger.info("Saved DC context to %s", path)
    return path


def _load_dc_context(
    cls: type,
    model: 'DistributedPowerGridModel',
    path: str,
) -> 'DistributedSolverContext':
    """Load DC context from disk. Returns unfactored context (call refactor())."""
    with open(path, 'rb') as f:
        data = pickle.load(f)

    if data.get('type') != 'DistributedSolverContext':
        raise ValueError(
            f"Expected DistributedSolverContext checkpoint, "
            f"got {data.get('type')!r}"
        )

    ctx = cls(model=model, topology=data['topology'])
    ctx._S_global = data.get('S_global')
    ctx.timings = data.get('timings', {})
    _restore_role_configs(data, model)
    # NOT factored -- caller must call refactor() or factor()
    logger.info("Loaded DC context from %s (is_factored=False)", path)
    return ctx


def _refactor_dc_context(ctx: 'DistributedSolverContext', verbose: bool = False) -> None:
    """Rebuild coordinator LU from saved S_global (DC). Workers must already be factored."""
    if ctx._S_global is None:
        raise RuntimeError(
            "Cannot refactor without S_global. Use factor() for a "
            "full factorization, or load a checkpoint that includes "
            "S_global."
        )
    from pgmath.factor import _factor_conductance_matrix

    coord_config = ctx.model.coordinator_solver_config if ctx.model is not None else None
    t0 = _time.perf_counter()
    interface_lu_result = _factor_conductance_matrix(
        ctx._S_global, verbose=verbose, config=coord_config,
    )
    elapsed = _time.perf_counter() - t0

    ctx._interface_lu = interface_lu_result.solve
    ctx.is_factored = True
    logger.info(
        "Refactored coordinator LU from saved S_global in %.3fs", elapsed,
    )


# ---------------------------------------------------------------------------
# Transient context: factor / save / load / refactor
# ---------------------------------------------------------------------------


def _factor_transient_context(
    ctx: 'DistributedTransientContext', verbose: bool = False
) -> None:
    """Factor transient A = G + C_coeff*C on tiles and assemble interface system."""
    from .result import DistributedTopologyContext

    if ctx.model is None:
        raise RuntimeError(
            "Cannot factor without a model reference. "
            "Either pass model= to __init__ or set self.model."
        )
    timings: Dict[str, Any] = {}
    model = ctx.model
    tile_configs = model.metadata.tile_configs
    method = ctx.integration_method

    # 1. Factor transient system on all workers (parallel)
    dt_scaled = ctx.dt_scaled
    C_coeff = ctx.C_coeff

    t0 = _time.perf_counter()
    trans_args = [(dt_scaled, method)] * len(tile_configs)
    schur_results = model.backend.call_all(
        model.workers, 'factor_transient_system', trans_args,
    )
    timings['factor_transient_tiles'] = _time.perf_counter() - t0

    # Organize results by tile
    tile_schur_complements: Dict[Any, np.ndarray] = {}
    tile_port_node_lists: Dict[Any, List[str]] = {}
    total_tile_cap = 0.0
    per_tile_stats: List[Dict[str, Any]] = []
    for i, (S_A_i, port_list, tile_cap, tile_stats) in enumerate(schur_results):
        tid = tile_configs[i].tile_id
        tile_schur_complements[tid] = S_A_i
        tile_port_node_lists[tid] = port_list
        total_tile_cap += tile_cap
        per_tile_stats.append(tile_stats)

    # Coordinator-side DEBUG: per-tile transient factor details
    from pgmath.block_system import _format_bytes
    from distributed.solver import _fmt_count
    for i, ts in enumerate(per_tile_stats):
        tid = tile_configs[i].tile_id
        n_pp = ts.get('n_ports', 0)
        logger.debug(
            "Tile %s factor_transient_system:\n"
            "  A_ii: nnz=%s  |  A_pp: nnz=%s\n"
            "  C_ii cap nodes: %d / %d  |  C_pp cap nodes: %d / %d\n"
            "  Total tile cap: %.1f fF  |  C_coeff: %.4f\n"
            "  factor_interior: %.3fs  |  backend: %s\n"
            "  compute_schur: %.3fs  |  Schur: %s x %s dense (%s)",
            tid,
            _fmt_count(ts.get('A_ii_nnz', 0)), _fmt_count(ts.get('A_pp_nnz', 0)),
            ts.get('c_ii_cap_nodes', 0), ts.get('n_interior', 0),
            ts.get('c_pp_cap_nodes', 0), n_pp,
            ts.get('total_cap_fF', 0), ts.get('C_coeff', 0),
            ts.get('factor_interior_s', 0), ts.get('factorization_backend_info', 'n/a'),
            ts.get('compute_schur_s', 0),
            f"{n_pp:,}", f"{n_pp:,}",
            _format_bytes(ts.get('schur_mem_bytes', 0)),
        )

    # 2. Build combined package edges: resistive + effective cap edges
    t0 = _time.perf_counter()
    pkg_res_edges = model.package_data.package_edges
    pkg_cap_edges = model.package_data.package_cap_edges

    combined_edges = list(pkg_res_edges)
    has_cap = total_tile_cap > 0
    for u, v, c_fF in pkg_cap_edges:
        if c_fF > 0:
            combined_edges.append((u, v, C_coeff * c_fF))
            has_cap = True
    ctx.has_capacitance = has_cap

    # 3. Assemble transient interface system
    from pgmath.schur import (
        assemble_schur_complement_system,
        build_interface_package_matrices,
        detect_interface_islands,
    )

    S_global, rhs_dirichlet_A, interface_nodes, interface_node_to_idx = (
        assemble_schur_complement_system(
            tile_schur_complements=tile_schur_complements,
            tile_port_node_lists=tile_port_node_lists,
            extra_edges=combined_edges,
            dirichlet_nodes=model.pad_nodes,
            dirichlet_voltage=model.vdd,
        )
    )

    # Also compute G-only Dirichlet RHS (without cap contributions)
    # needed for correct transient RHS formulation
    _, rhs_dirichlet_G, _, _ = assemble_schur_complement_system(
        tile_schur_complements=tile_schur_complements,
        tile_port_node_lists=tile_port_node_lists,
        extra_edges=list(pkg_res_edges),
        dirichlet_nodes=model.pad_nodes,
        dirichlet_voltage=model.vdd,
    )

    # Island detection on transient system
    S_global, rhs_dirichlet_A, island_nodes = detect_interface_islands(
        S_global, rhs_dirichlet_A, interface_nodes, interface_node_to_idx,
        pad_nodes=model.pad_nodes,
        extra_edges=combined_edges,
        dirichlet_voltage=model.vdd,
    )

    if island_nodes:
        logger.warning(
            "Transient: penalized %d interface island nodes",
            len(island_nodes),
        )

    timings['assemble_transient_interface'] = _time.perf_counter() - t0

    # 4. Build package G and C matrices for transient RHS history terms
    n_interface = len(interface_nodes)
    G_pkg_uu, C_pkg_uu = build_interface_package_matrices(
        package_edges=pkg_res_edges,
        package_cap_edges=pkg_cap_edges,
        interface_node_to_idx=interface_node_to_idx,
        n_interface=n_interface,
        dirichlet_nodes=model.pad_nodes,
    )

    # 5. Factor transient interface system
    t0 = _time.perf_counter()
    from pgmath.factor import _factor_conductance_matrix

    interface_lu_result = _factor_conductance_matrix(
        S_global, verbose=verbose, config=model.coordinator_solver_config,
    )
    timings['factor_transient_interface'] = _time.perf_counter() - t0

    # 6. Build tile index maps
    tile_index_maps: Dict[Tuple[int, int], np.ndarray] = {}
    for tid, port_list in tile_port_node_lists.items():
        local_to_global = np.array(
            [interface_node_to_idx[n] for n in port_list
             if n in interface_node_to_idx],
            dtype=np.int32,
        )
        tile_index_maps[tid] = local_to_global

    timings['total_prepare_transient'] = sum(
        v for k, v in timings.items()
        if k != 'total_prepare_transient' and isinstance(v, (int, float))
    )

    # --- Build solver_stats ---
    from distributed.solver import _minmeanmax
    from pgmath.block_system import _sparse_mem_bytes, _format_bytes as _fb

    # Interface system stats
    n_unknowns = S_global.shape[0]
    iface_nnz = S_global.nnz
    iface_density_pct = (
        (iface_nnz / (n_unknowns * n_unknowns) * 100)
        if n_unknowns > 0 else 0.0
    )
    iface_mem_bytes = _sparse_mem_bytes(S_global)

    interface_stats: Dict[str, Any] = {
        'n_unknowns': n_unknowns,
        'nnz': iface_nnz,
        'density_pct': iface_density_pct,
        'mem_bytes': iface_mem_bytes,
        'factor_time_s': timings['factor_transient_interface'],
        'backend': interface_lu_result.backend,
        'backend_info': interface_lu_result.backend_info,
        'has_capacitance': has_cap,
        'pkg_C_uu_nnz': C_pkg_uu.nnz if C_pkg_uu is not None else 0,
        'pkg_G_uu_nnz': G_pkg_uu.nnz if G_pkg_uu is not None else 0,
    }

    # Aggregate per-tile stats (min/mean/max)
    aggregate_stats: Dict[str, Any] = {}
    if per_tile_stats:
        for key in ('A_ii_nnz', 'A_pp_nnz', 'total_cap_fF',
                    'schur_mem_bytes', 'factor_interior_s'):
            values = [s.get(key, 0) for s in per_tile_stats]
            lo, avg, hi = _minmeanmax(values)
            aggregate_stats[key] = {'min': lo, 'mean': avg, 'max': hi,
                                    'total': float(np.sum(values))}

    solver_stats: Dict[str, Any] = {
        'per_tile': per_tile_stats,
        'interface': interface_stats,
        'aggregate': aggregate_stats,
    }
    timings['solver_stats'] = solver_stats

    # --- Verbose INFO logging ---
    if verbose:
        n_tiles = len(per_tile_stats)
        logger.info("=== Distributed DDM Prepare Transient Statistics ===")
        logger.info("Method: %s  |  dt: %.1f ps  |  C_coeff: %.4f",
                    method, dt_scaled, C_coeff)
        logger.info("Tiles: %d", n_tiles)
        if aggregate_stats:
            _ag = aggregate_stats
            lo, avg, hi = _ag['A_ii_nnz']['min'], _ag['A_ii_nnz']['mean'], _ag['A_ii_nnz']['max']
            logger.info("  A_ii nnz:         %s / %s / %s  (min/mean/max)",
                        _fmt_count(lo), _fmt_count(avg), _fmt_count(hi))
            lo, avg, hi = _ag['A_pp_nnz']['min'], _ag['A_pp_nnz']['mean'], _ag['A_pp_nnz']['max']
            logger.info("  A_pp nnz:         %s / %s / %s",
                        _fmt_count(lo), _fmt_count(avg), _fmt_count(hi))
            lo, avg, hi = _ag['total_cap_fF']['min'], _ag['total_cap_fF']['mean'], _ag['total_cap_fF']['max']
            total_cap = _ag['total_cap_fF']['total']
            logger.info("  Total cap:        %.0f fF / %.0f fF / %.0f fF  (total: %.0f fF)",
                        lo, avg, hi, total_cap)
            lo, avg, hi = _ag['schur_mem_bytes']['min'], _ag['schur_mem_bytes']['mean'], _ag['schur_mem_bytes']['max']
            total_sm = _ag['schur_mem_bytes']['total']
            logger.info("  Schur memory:     %s / %s / %s  (total: %s)",
                        _fb(lo), _fb(avg), _fb(hi), _fb(total_sm))
            lo, avg, hi = _ag['factor_interior_s']['min'], _ag['factor_interior_s']['mean'], _ag['factor_interior_s']['max']
            logger.info("  Factor time:      %.3fs / %.3fs / %.3fs", lo, avg, hi)
            # Backend: use the first tile's backend_info as representative
            backend_info = per_tile_stats[0].get('factorization_backend_info', 'n/a')
            logger.info("  Factor backend:   %s", backend_info)
            logger.info("  Factor + Schur wall time: %.3fs", timings['factor_transient_tiles'])

        logger.info("Transient interface: %s unknowns, %s nnz (density %.3f%%), %s",
                    _fmt_count(n_unknowns), _fmt_count(iface_nnz),
                    iface_density_pct, _fb(iface_mem_bytes))
        logger.info("  Backend: %s", interface_lu_result.backend_info)
        logger.info("  Factor time: %.3fs", timings['factor_transient_interface'])
        logger.info("  Has capacitance: %s", has_cap)
        logger.info("  Package C_uu nnz: %d  |  Package G_uu nnz: %d",
                    interface_stats['pkg_C_uu_nnz'],
                    interface_stats['pkg_G_uu_nnz'])
        logger.info("  Assemble time: %.3fs", timings['assemble_transient_interface'])
        if 'detect_interface_islands' in timings:
            logger.info("  Detect islands time: %.3fs", timings['detect_interface_islands'])
        logger.info("=== Total Prepare: %.3fs ===", timings['total_prepare_transient'])

    # Populate context fields
    ctx._interface_lu = interface_lu_result.solve
    ctx._interface_nodes = interface_nodes
    ctx._interface_node_to_idx = interface_node_to_idx
    ctx.rhs_dirichlet_A = rhs_dirichlet_A
    ctx._rhs_dirichlet_G = rhs_dirichlet_G
    ctx._tile_index_maps = tile_index_maps
    ctx._removed_interface_nodes = island_nodes
    ctx._S_global = S_global
    ctx.C_package_uu = C_pkg_uu if C_pkg_uu.nnz > 0 else None
    ctx._G_package_uu = G_pkg_uu if G_pkg_uu.nnz > 0 else None
    ctx.timings = timings

    # Build topology context if not already provided
    if ctx.topology is None:
        ctx.topology = DistributedTopologyContext(
            interface_nodes=interface_nodes,
            interface_node_to_idx=interface_node_to_idx,
            tile_index_maps=tile_index_maps,
            rhs_dirichlet_G=rhs_dirichlet_G,
            G_package_uu=G_pkg_uu if G_pkg_uu.nnz > 0 else None,
            removed_interface_nodes=island_nodes,
        )

    ctx.is_factored = True


def _save_transient_context(
    ctx: 'DistributedTransientContext', path: Optional[str] = None
) -> str:
    """Save transient context metadata (topology, S_global, integration params) to disk."""
    if path is None:
        path = _default_checkpoint_path(ctx, 'transient_context.pkl')

    if ctx._S_global is None:
        raise RuntimeError(
            "Cannot save: S_global is None (was release() called before "
            "save()?). Save before release(), or re-factor first."
        )

    save_data = {
        'type': 'DistributedTransientContext',
        'version': 1,
        'topology': ctx.topology,
        'S_global': ctx._S_global,
        'dt_scaled': ctx.dt_scaled,
        'integration_method': ctx.integration_method,
        'has_capacitance': ctx.has_capacitance,
        'rhs_dirichlet_A': ctx.rhs_dirichlet_A,
        'rhs_dirichlet_G': ctx._rhs_dirichlet_G,
        'C_package_uu': ctx.C_package_uu,
        'G_package_uu': ctx._G_package_uu,
        'timings': ctx.timings,
        **_save_role_configs(ctx.model),
    }

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(save_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    logger.info("Saved transient context to %s", path)
    return path


def _load_transient_context(
    cls: type,
    model: 'DistributedPowerGridModel',
    path: str,
) -> 'DistributedTransientContext':
    """Load transient context from disk. Returns unfactored context (call refactor())."""
    with open(path, 'rb') as f:
        data = pickle.load(f)

    if data.get('type') != 'DistributedTransientContext':
        raise ValueError(
            f"Expected DistributedTransientContext checkpoint, "
            f"got {data.get('type')!r}"
        )

    ctx = cls(
        model=model,
        topology=data['topology'],
        dt_scaled=data['dt_scaled'],
        integration_method=data['integration_method'],
    )
    ctx._S_global = data.get('S_global')
    ctx.has_capacitance = data.get('has_capacitance', False)
    ctx.rhs_dirichlet_A = data.get('rhs_dirichlet_A')
    ctx._rhs_dirichlet_G = data.get('rhs_dirichlet_G')
    ctx.C_package_uu = data.get('C_package_uu')
    ctx._G_package_uu = data.get('G_package_uu')
    ctx.timings = data.get('timings', {})
    _restore_role_configs(data, model)
    # NOT factored -- caller must call refactor() or factor()
    logger.info(
        "Loaded transient context from %s (is_factored=False)", path,
    )
    return ctx


def _refactor_transient_context(
    ctx: 'DistributedTransientContext', verbose: bool = False
) -> None:
    """Rebuild coordinator LU from saved S_global (transient). Workers must already be factored."""
    if ctx._S_global is None:
        raise RuntimeError(
            "Cannot refactor without S_global. Use factor() for a "
            "full factorization, or load a checkpoint that includes "
            "S_global."
        )
    from pgmath.factor import _factor_conductance_matrix

    coord_config = ctx.model.coordinator_solver_config if ctx.model is not None else None
    t0 = _time.perf_counter()
    interface_lu_result = _factor_conductance_matrix(
        ctx._S_global, verbose=verbose, config=coord_config,
    )
    elapsed = _time.perf_counter() - t0

    ctx._interface_lu = interface_lu_result.solve
    ctx.is_factored = True
    logger.info(
        "Refactored transient coordinator LU from saved S_global "
        "in %.3fs", elapsed,
    )

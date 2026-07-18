"""Factorization, checkpoint, and refactor helpers for distributed contexts.

Extracted from result.py to keep file sizes manageable (mixin pattern).
All functions take the context as the first argument and mutate it in place,
mirroring the original method bodies exactly.

Interface-solver setting (B2)
------------------------------
The ``interface_solver`` setting controls how the global Schur complement
S_global is solved at the coordinator.  Three values:

  'direct' -- direct CHOLMOD/SuperLU factorization (existing behaviour).
  'cg'     -- iterative CG via InterfaceCGSolver (removes the factor).
  'auto'   -- select based on n_interface + estimated factor memory.
              DEFAULT is 'auto', which resolves to 'direct' for the
              netlist_sampled benchmark (n_interface ~2-4K << 200K threshold).
              EXISTING MODELS THEREFORE GET IDENTICAL BEHAVIOUR.

The setting is read from ``model.settings.get('interface_solver', 'auto')``
(same dict plumbed via TileWorker.configure / YAML / CLI ``--interface-solver``).
If the model has no settings dict, the default is 'auto'.

Adjoint note (v1)
-----------------
analyze_adjoint* uses ctx._interface_lu.  CG mode is compatible for DC
adjoint because CG provides the same solve-callable interface.  Tilewise
CG adjoint would need per-tile S_i blocks at adjoint time; that is not
implemented in v1 -- the assembled CG mode works fine.  The context stores
``_interface_solver_mode`` so the adjoint code can check if needed.

Streaming assembly (B3) vs CG tilewise matvec: non-composition
---------------------------------------------------------------
``streaming_assembly=True`` and ``interface_matvec_mode='tilewise'`` are
**mutually exclusive** in the current implementation.

  - ``tilewise`` CG matvec requires the per-tile dense S_i blocks at
    coordinator side to form ``matvec(x) = sum_i P_i^T (S_i (P_i x))``.
    Building ``S_extra`` (the package-edge contribution) also needs S_i.
  - Streaming assembly intentionally never gathers S_i to the coordinator —
    that is its entire purpose (coordinator memory peak reduction).

When both are requested, the code automatically falls back to
``interface_matvec_mode='assembled'`` and logs a WARNING.  ``assembled``
mode assembles S_global (which streaming did produce) and applies it as a
sparse matvec, so CG still avoids the CHOLMOD factor but retains S_global
in coordinator memory.

The ideal composition — S_i kept worker-resident, matvec implemented as
remote RPC calls to workers — is deferred to B4 (multi-node Ray task-
dataflow).  Workers already hold S_i in their factored state (accessible via
``get_schur_data_flat``), so a B4 remote-matvec extension would not require
re-factoring.

Setting combinations and their coordinator memory footprint:

  streaming=False, solver=direct   -- baseline: all S_i gathered + CHOLMOD factor
  streaming=False, solver=cg/assembled -- all S_i gathered + S_global (no factor)
  streaming=False, solver=cg/tilewise  -- all S_i kept coordinator-side for matvec
  streaming=True,  solver=direct   -- S_global only (no S_i, no factor)   [RECOMMENDED]
  streaming=True,  solver=cg/assembled -- S_global only (no S_i, no factor) [SAME AS ABOVE]
  streaming=True,  solver=cg/tilewise  -- falls back to 'assembled' (see above)
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
# Streaming assembly budget (B3)
# ---------------------------------------------------------------------------
#
# When streaming_assembly='auto', streaming is enabled when the estimated peak
# memory for bulk assembly (sum of all dense S_i bytes) exceeds this threshold.
# Default: 512 MB.  Overridable via monkeypatch in tests or via model settings.
STREAMING_ASSEMBLY_AUTO_BYTES: int = 512 * 1024 * 1024  # 512 MB

# Number of row-shards per tile for streaming assembly.  Larger values = more
# round-trips to workers but smaller per-shard memory.  Default 4 is a
# conservative choice: splits each tile's S_i into at most 4 row slices so the
# coordinator never holds more than 1/4 of a single tile's S_i at once.
STREAMING_ASSEMBLY_N_SHARDS: int = 4


# ---------------------------------------------------------------------------
# Helper: read interface_solver setting from model (or default 'auto')
# ---------------------------------------------------------------------------

def _get_interface_solver_setting(model: Optional[Any]) -> str:
    """Read the interface_solver setting from model.settings (default 'auto').

    The model may or may not have a .settings dict.  Fall back gracefully
    so that models created before B2 land continue to get 'auto' behaviour
    (which maps to 'direct' for small interface systems).
    """
    if model is None:
        return 'auto'
    settings = getattr(model, 'settings', None)
    if settings is None:
        return 'auto'
    return settings.get('interface_solver', 'auto')


def _detect_islands_dispatch(
    model: Optional[Any],
    S_global: sp.csr_matrix,
    rhs_dirichlet: np.ndarray,
    interface_nodes: List[str],
    interface_node_to_idx: Dict[str, int],
    extra_edges: Optional[List[Tuple[str, str, float]]],
) -> Tuple[sp.csr_matrix, np.ndarray, Set[str]]:
    """Dispatch island detection: Stage 1e summaries union-find vs Schur-BFS.

    ``model.island_detection_mode`` is resolved ONCE at model creation
    (``model._resolve_island_detection``, all-new-or-all-legacy) — this
    function must NOT re-derive or override it.  Callers (DC and transient
    factor paths, both cache-miss branches) are otherwise unchanged: same
    return shape as :func:`pgmath.schur.detect_interface_islands`, same A7
    topology-context caching wrapping this call.

    Finding F15: ``model`` is typed ``Optional`` but every branch below
    dereferences ``model.pad_nodes``/``model.vdd`` -- island detection
    (either strategy) is fundamentally model-scoped (it needs the model's
    pad set and supply voltage), so ``model=None`` is NOT a supported input.
    Raise loudly and immediately here instead of letting a confusing
    ``AttributeError`` surface deep inside whichever branch runs next.
    """
    if model is None:
        raise ValueError(
            "_detect_islands_dispatch requires a non-None model (island "
            "detection needs model.pad_nodes/model.vdd); got model=None."
        )
    _mode = getattr(model, 'island_detection_mode', 'schur_bfs')
    _summaries = getattr(model, 'component_summaries', None)

    # Finding R6: island_detection_mode == 'summaries' with
    # component_summaries is None is a FORBIDDEN mixed state -- mode
    # resolution in model._resolve_island_detection is the only legitimate
    # place these two fields are set, and it always sets them together
    # (summaries iff mode == 'summaries').  Because DistributedPowerGridModel
    # is a plain dataclass, the two fields can go out of sync via direct
    # construction, field mutation, or a future serialization path that
    # drops the non-repr summaries field.  Silently degrading to the legacy
    # Schur-BFS here (the prior behaviour) would lose the tile-resident-pad
    # rescue and produce silently wrong voltages -- exactly the bug the
    # summaries path exists to fix -- with no warning that anything is
    # amiss.  Raise loudly instead.
    if _mode == 'summaries' and _summaries is None:
        raise ValueError(
            "_detect_islands_dispatch: forbidden mixed state -- "
            "model.island_detection_mode == 'summaries' but "
            "model.component_summaries is None. These two fields must only "
            "ever be set together by model._resolve_island_detection; this "
            "indicates a corrupted or hand-constructed DistributedPowerGridModel, "
            "or a serialization path that dropped the non-repr summaries "
            "field. Refusing to silently degrade to the legacy Schur-BFS "
            "path (which would lose the tile-resident-pad rescue and could "
            "produce silently wrong voltages)."
        )

    if _mode == 'summaries' and _summaries is not None:
        from pgmath.schur import apply_island_penalty, detect_interface_islands_from_summaries
        island_nodes = detect_interface_islands_from_summaries(
            component_summaries=_summaries,
            interface_node_to_idx=interface_node_to_idx,
            pad_nodes=model.pad_nodes,
            extra_edges=extra_edges,
        )
        if not island_nodes:
            return S_global, rhs_dirichlet, set()
        S_fixed, rhs_fixed = apply_island_penalty(
            S_global, rhs_dirichlet, island_nodes,
            interface_node_to_idx, model.vdd,
        )
        return S_fixed, rhs_fixed, island_nodes

    from pgmath.schur import detect_interface_islands
    return detect_interface_islands(
        S_global, rhs_dirichlet, interface_nodes, interface_node_to_idx,
        pad_nodes=model.pad_nodes, extra_edges=extra_edges,
        dirichlet_voltage=model.vdd,
    )


def _coerce_bool(value: Any, setting_name: str) -> bool:
    """Coerce a YAML/CLI-sourced value to bool, rejecting garbage loudly.

    Real Python bools pass through unchanged.  Strings are matched
    case-insensitively against common true/false tokens.  This avoids the
    classic ``bool(x)`` pitfall where ``bool('false')`` is ``True`` because
    any non-empty string is truthy -- a real risk here because YAML often
    delivers quoted strings for booleans (e.g. ``interface_cg_strict:
    "false"``).

    Args:
        value: The raw settings value.
        setting_name: Name of the setting, used in the error message.

    Returns:
        The coerced bool.

    Raises:
        ValueError: If ``value`` cannot be unambiguously interpreted as a
            bool.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ('true', '1', 'yes', 'on'):
            return True
        if v in ('false', '0', 'no', 'off'):
            return False
    raise ValueError(
        f"Invalid boolean setting {setting_name}={value!r}: expected a bool "
        f"or one of 'true'/'false'/'1'/'0'/'yes'/'no'/'on'/'off' "
        f"(case-insensitive)."
    )


def _coerce_int(value: Any, setting_name: str) -> int:
    """Coerce a YAML/CLI-sourced numeric setting to int, rejecting garbage.

    Accepts real ints/floats and numeric strings (including exponent
    notation, e.g. '500', '5e2') via ``int(float(value))`` -- PyYAML 1.1
    parses bare-exponent floats like ``1e-10`` as strings, so this mirrors
    the defensive ``float()`` coercion already used for rtol/atol.  Bools
    are rejected explicitly (Python's ``bool`` is an ``int`` subclass, so
    ``int(float(True))`` would silently resolve to 1 instead of raising).

    Args:
        value: The raw settings value.
        setting_name: Name of the setting, used in the error message.

    Returns:
        The coerced int.

    Raises:
        ValueError: If ``value`` cannot be interpreted as an int.
    """
    if isinstance(value, bool):
        raise ValueError(
            f"Invalid integer setting {setting_name}={value!r}: bool is not "
            f"a valid integer value."
        )
    try:
        return int(float(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid integer setting {setting_name}={value!r}: expected an "
            f"integer (or numeric string)."
        ) from exc


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
# B3: Streaming Schur assembly helpers
# ---------------------------------------------------------------------------


def _get_streaming_assembly_setting(model: Optional[Any]) -> Any:
    """Read the streaming_assembly setting from model.settings (default False).

    Values:
      False   -- off (default): use bulk assemble_schur_complement_system.
      True    -- always stream, regardless of memory estimate.
      'auto'  -- stream when estimated S_i memory sum > STREAMING_ASSEMBLY_AUTO_BYTES.
    """
    if model is None:
        return False
    settings = getattr(model, 'settings', None)
    if settings is None:
        return False
    return settings.get('streaming_assembly', False)


def _estimate_schur_peak_bytes(per_tile_stats: List[Dict[str, Any]]) -> int:
    """Estimate peak coordinator memory if all S_i are gathered at once.

    Uses the schur_mem_bytes reported by each tile's stats dict.  Returns 0
    when stats are unavailable (safe: won't trigger auto-streaming).
    """
    return sum(int(s.get('schur_mem_bytes', 0)) for s in per_tile_stats)


def _should_stream(model: Optional[Any], per_tile_stats: List[Dict[str, Any]]) -> bool:
    """Decide whether to use streaming assembly for this prepare() call.

    Returns True when streaming_assembly is True (forced) or 'auto' and the
    estimated peak S_i memory exceeds STREAMING_ASSEMBLY_AUTO_BYTES.
    """
    setting = _get_streaming_assembly_setting(model)
    if setting is False:
        return False
    if setting is True:
        return True
    # 'auto'
    auto_budget = int(
        getattr(model, 'settings', {}).get(
            'streaming_assembly_auto_bytes', STREAMING_ASSEMBLY_AUTO_BYTES
        )
        if model is not None else STREAMING_ASSEMBLY_AUTO_BYTES
    )
    est = _estimate_schur_peak_bytes(per_tile_stats)
    if est > auto_budget:
        logger.info(
            "streaming_assembly='auto': estimated S_i memory %.1f MB > "
            "budget %.1f MB — enabling streaming assembly.",
            est / 1024 ** 2, auto_budget / 1024 ** 2,
        )
        return True
    logger.debug(
        "streaming_assembly='auto': estimated S_i memory %.1f MB <= "
        "budget %.1f MB — using bulk assembly.",
        est / 1024 ** 2, auto_budget / 1024 ** 2,
    )
    return False


def _compute_rhs_dirichlet_from_edges(
    extra_edges: Optional[List],
    unknown_list: List[str],
    unknown_to_idx: Dict[str, int],
    dirichlet_nodes: Optional[Set[str]],
    dirichlet_voltage: float,
) -> np.ndarray:
    """Compute Dirichlet RHS from extra (package) edges WITHOUT tile S_i.

    B3: Used in the streaming transient path to compute rhs_dirichlet_G
    (G-only Dirichlet contribution) after streaming assembly has already
    determined the interface node ordering.  Since tile Schur complements
    only contribute to the unknown-unknown block and NOT to the
    unknown-Dirichlet coupling (G_ud), this function provides an exact
    rhs_dirichlet from package edges alone.

    Args:
        extra_edges: List of (u, v, g) package-edge triples.
        unknown_list: Ordered list of unknown interface node names.
        unknown_to_idx: {node: index} for unknown nodes.
        dirichlet_nodes: Set of Dirichlet (pad) node names.
        dirichlet_voltage: Voltage applied to all Dirichlet nodes (Vdd).

    Returns:
        rhs_dirichlet (shape n_unknown,) = -(G_ud @ V_d) from extra edges.
    """
    n_unknown = len(unknown_list)
    rhs = np.zeros(n_unknown, dtype=np.float64)

    if not extra_edges or n_unknown == 0:
        return rhs

    ground_node = '0'
    dirichlet_set = set(dirichlet_nodes) if dirichlet_nodes else set()

    for u, v, g in extra_edges:
        if g <= 0:
            continue
        # Only edges that couple an unknown to a Dirichlet node contribute.
        u_is_d = u in dirichlet_set
        v_is_d = v in dirichlet_set
        u_in_u = u in unknown_to_idx
        v_in_u = v in unknown_to_idx

        if u_is_d and v_in_u:
            # Edge (Dirichlet u) -- [g] -- (unknown v):
            # G_ud[v, u] = -g  (off-diagonal), so rhs = -(G_ud @ V_d) = -(-g * V_d) = +g*V_d
            iv = unknown_to_idx[v]
            rhs[iv] += g * dirichlet_voltage
        if v_is_d and u_in_u:
            # Edge (unknown u) -- [g] -- (Dirichlet v):
            # G_ud[u, v] = -g  (off-diagonal), so rhs = -(G_ud @ V_d) = -(-g * V_d) = +g*V_d
            iu = unknown_to_idx[u]
            rhs[iu] += g * dirichlet_voltage

    return rhs


def _build_s_extra_coo(
    S_global_csr: sp.csr_matrix,
    tile_schur_complements: Dict[Any, np.ndarray],
    tile_index_maps: Dict[Any, np.ndarray],
    n_iface: int,
) -> sp.csr_matrix:
    """Vectorized S_extra = S_global - sum_i P_i^T S_i P_i (B2 follow-up).

    Replaces the O(n_ports^2)-per-tile nested Python loop (LIL construction)
    with a single vectorized COO scatter-add.  The result is the package-edge
    contribution that is NOT captured by per-tile Schur complements and needs
    to be added as S_extra in the tilewise CG matvec.

    Args:
        S_global_csr: Assembled global Schur matrix (CSR).
        tile_schur_complements: {tile_id: S_i dense array}.
        tile_index_maps: {tile_id: int32 global-index array}.
        n_iface: Number of interface unknowns.

    Returns:
        Sparse CSR matrix S_extra (entries below 1e-15 eliminated).
    """
    # Vectorized COO assembly of sum_i P_i^T S_i P_i
    coo_rows_parts: List[np.ndarray] = []
    coo_cols_parts: List[np.ndarray] = []
    coo_data_parts: List[np.ndarray] = []

    for tid, S_i in tile_schur_complements.items():
        idx = tile_index_maps[tid]
        n_local = len(idx)
        global_rows = np.repeat(idx, n_local)
        global_cols = np.tile(idx, n_local)
        coo_rows_parts.append(global_rows)
        coo_cols_parts.append(global_cols)
        coo_data_parts.append(np.asarray(S_i, dtype=np.float64).ravel())

    if coo_rows_parts:
        all_rows = np.concatenate(coo_rows_parts).astype(np.int32)
        all_cols = np.concatenate(coo_cols_parts).astype(np.int32)
        all_data = np.concatenate(coo_data_parts).astype(np.float64)
        S_tile_sum = sp.coo_matrix(
            (all_data, (all_rows, all_cols)), shape=(n_iface, n_iface)
        ).tocsr()
    else:
        S_tile_sum = sp.csr_matrix((n_iface, n_iface), dtype=np.float64)

    S_extra = (S_global_csr - S_tile_sum).tocsr()
    S_extra.eliminate_zeros()
    return S_extra


def _build_csr_scatter_pattern(
    G_full_csr: 'sp.csr_matrix',
    tile_index_maps: Dict[Any, np.ndarray],
    extra_rows: np.ndarray,
    extra_cols: np.ndarray,
) -> Dict[str, Any]:
    """Precompute scatter indices mapping each COO entry to its CSR data position.

    B3: After the first (COO-path) streaming call, this helper records where in
    the G_full CSR data array each per-tile row×col entry and each extra-edge
    entry lands.  Subsequent streaming calls use these precomputed indices to
    scatter-add directly into a preallocated float64 array — peak coordinator
    memory is then exactly one tile's shard at a time (plus the O(nnz)
    preallocated buffer, which is the same size as the final G_full.data).

    The CSR data array is sorted by row and within each row by column
    (guaranteed by scipy's tocsr()).  The position of entry (r, c) is:
        indptr[r] + searchsorted(indices[indptr[r]:indptr[r+1]], c)

    For each tile, every entry in the tile's S_i contributes a
    (global_row, global_col) pair.  We compute all of them at once using
    vectorised searchsorted.

    Args:
        G_full_csr: The assembled full G matrix (CSR, sorted column indices).
        tile_index_maps: {tile_id: int32 local→global index array}.
        extra_rows: int32 COO row indices for the extra-edge entries.
        extra_cols: int32 COO col indices for the extra-edge entries.

    Returns:
        Pattern dict with keys:
          'indptr'            : G_full_csr.indptr copy (int32/int64)
          'indices'           : G_full_csr.indices copy (int32/int64)
          'n_full'            : G_full_csr.shape[0]
          'nnz'               : G_full_csr.nnz
          'tile_scatter_idxs' : {tile_id: int64 flat position array}
          'extra_scatter_idx' : int64 flat position array for extra entries
    """
    indptr = G_full_csr.indptr
    indices = G_full_csr.indices

    def _coo_to_csr_positions(
        rows: np.ndarray, cols: np.ndarray,
    ) -> np.ndarray:
        """Return the flat CSR data positions for a set of (row, col) pairs."""
        positions = np.empty(len(rows), dtype=np.int64)
        # Group by row for vectorised searchsorted
        # (rows may repeat, so we iterate over unique rows)
        if len(rows) == 0:
            return positions
        # Sort by row for grouped processing
        order = np.argsort(rows, kind='stable')
        rows_s = rows[order]
        cols_s = cols[order]
        # Use searchsorted within each row's column slice
        # Most tiles have all entries in a small number of rows — batch.
        unique_rows, row_starts = np.unique(rows_s, return_index=True)
        row_ends = np.append(row_starts[1:], len(rows_s))
        for i, r in enumerate(unique_rows):
            r = int(r)
            lo = int(row_starts[i])
            hi = int(row_ends[i])
            row_cols = cols_s[lo:hi]
            csr_lo = int(indptr[r])
            csr_hi = int(indptr[r + 1])
            row_idx_slice = indices[csr_lo:csr_hi]
            offsets = np.searchsorted(row_idx_slice, row_cols)
            positions[order[lo:hi]] = csr_lo + offsets
        return positions

    # Per-tile scatter indices
    tile_scatter_idxs: Dict[Any, np.ndarray] = {}
    for tid, idx in tile_index_maps.items():
        n_local = len(idx)
        global_rows = np.repeat(idx, n_local).astype(np.int32)
        global_cols = np.tile(idx, n_local).astype(np.int32)
        tile_scatter_idxs[tid] = _coo_to_csr_positions(global_rows, global_cols)

    # Extra-edge scatter indices
    if len(extra_rows) > 0:
        extra_scatter_idx = _coo_to_csr_positions(
            extra_rows.astype(np.int32), extra_cols.astype(np.int32)
        )
    else:
        extra_scatter_idx = np.empty(0, dtype=np.int64)

    return {
        'indptr': indptr.copy(),
        'indices': indices.copy(),
        'n_full': G_full_csr.shape[0],
        'nnz': G_full_csr.nnz,
        'tile_scatter_idxs': tile_scatter_idxs,
        'extra_scatter_idx': extra_scatter_idx,
    }


def _stream_assemble_schur(
    model: Any,
    tile_port_node_lists: Dict[Any, List[str]],
    per_tile_stats: List[Dict[str, Any]],
    extra_edges: Optional[List],
    dirichlet_nodes: Optional[Set[str]],
    dirichlet_voltage: float,
    assembly_cache: Dict[str, Any],
) -> Tuple[
    sp.csr_matrix,
    np.ndarray,
    List[str],
    Dict[str, int],
    Dict[Any, np.ndarray],  # tile_schur_complements (None — not held)
]:
    """Stream-assemble S_global by fetching COO shards one tile at a time.

    B3: Avoids holding all dense S_i simultaneously in coordinator memory.

    Three internal paths:

    FAST PATH (CSR scatter pattern cached from a prior call):
      - Preallocate G_full_data = np.zeros(pattern.nnz).
      - Call ``get_schur_data_flat`` on each tile via ``call_all_streaming``:
        yields (tile_idx, flat_float64_array) one at a time.
      - Scatter-add each tile's float64 flat array directly into G_full_data
        using the cached scatter positions; free the flat array immediately.
      - Scatter-add extra-edge values.
      - Reconstruct G_full as CSR from (G_full_data, pattern.indptr, pattern.indices).
      Peak extra coordinator memory: one tile's float64 flat array
      (n_ports^2 * 8 bytes) + preallocated G_full_data (nnz * 8 bytes).

    FIRST CALL — two-pass (no scatter pattern yet):
      Pass 1 — index-only pre-pass (no float64 on the wire):
        - Call ``get_schur_coo_indices_only`` on each tile via ``call_all_streaming``:
          yields (tile_idx, (global_rows, global_cols)) one at a time.
        - Accumulate int32 rows/cols across all tiles (no float64 values).
        - After all tiles are collected, add extra-edge rows/cols.
        - Build COO→CSR with dummy (all-ones) data to obtain indptr/indices.
        - Compute CSR scatter positions via ``_build_csr_scatter_pattern``.
        - Free accumulated rows/cols arrays.
      Pass 2 — data-only (fast path after pattern is cached):
        - Preallocate G_full_data = np.zeros(pattern.nnz).
        - Call ``get_schur_data_flat`` on each tile via ``call_all_streaming``
          (scatter pattern now known; see fast path above).
      Peak extra coordinator memory (two-pass first call):
        - Pre-pass accumulation: O(sum_i n_ports_i^2 * 8 bytes) int32 rows+cols
          = O(nnz * 8 bytes).  No float64 during this phase.
        - After pre-pass: scatter pattern (nnz * ~12 bytes) + preallocated buffer
          (nnz * 8 bytes).  Row/col accumulations freed before data pass starts.
        - Data pass: one tile's float64 flat array at a time.
        Combined first-call peak ≈ O(nnz * 20 bytes):
          preallocated buffer (8 B/entry) + scatter pattern (~12 B/entry).
          Compared with old single-pass COO: O(nnz * 16 bytes) float+indices held
          simultaneously PLUS tocsr output simultaneously = O(nnz * 28+ bytes).
        This bounds first-call peak to the same asymptotic class as subsequent calls.

    Returns:
        (S_global, rhs_dirichlet, interface_nodes, interface_node_to_idx,
         tile_index_maps)
    where tile_index_maps are the per-tile local->global index arrays built
    during Step 2 (stored for downstream use: S_extra, CG solver setup, etc.).

    NOTE: tile_schur_complements are NOT returned — they were never gathered.
    Callers that need them for CG tilewise mode must call factor_and_compute_schur
    via the normal (bulk) path instead.
    """
    workers = model.workers
    tile_configs = model.metadata.tile_configs
    backend = model.backend

    n_shards_setting = (
        getattr(model, 'settings', {}).get(
            'streaming_assembly_n_shards', STREAMING_ASSEMBLY_N_SHARDS
        )
        if model is not None else STREAMING_ASSEMBLY_N_SHARDS
    )
    n_shards = max(1, int(n_shards_setting))

    # Step 1: Build node index universe (same logic as assemble_schur_complement_system)
    all_interface_nodes: Set[str] = set()
    ground_node = '0'
    if dirichlet_nodes is None:
        dirichlet_nodes = set()

    for node_list in tile_port_node_lists.values():
        all_interface_nodes.update(node_list)
    if extra_edges:
        for u, v, g in extra_edges:
            if u != ground_node:
                all_interface_nodes.add(u)
            if v != ground_node:
                all_interface_nodes.add(v)

    dirichlet_set = dirichlet_nodes & all_interface_nodes
    unknown_nodes = all_interface_nodes - dirichlet_set

    # --- A4 cache reuse (same logic as assemble_schur_complement_system) ----
    _use_cached_idx = False
    if assembly_cache and 'full_node_to_idx' in assembly_cache:
        if (assembly_cache.get('n_unknown') == len(unknown_nodes)
                and assembly_cache.get('n_full') == len(unknown_nodes) + len(dirichlet_set)):
            _use_cached_idx = True

    if _use_cached_idx:
        full_node_to_idx = assembly_cache['full_node_to_idx']
        unknown_list = assembly_cache['unknown_list']
        dirichlet_list = assembly_cache['dirichlet_list']
        n_unknown = assembly_cache['n_unknown']
        n_dirichlet = len(dirichlet_set)
        n_full = assembly_cache['n_full']
    else:
        unknown_list = sorted(unknown_nodes)
        dirichlet_list = sorted(dirichlet_set)
        all_ordered = unknown_list + dirichlet_list
        full_node_to_idx = {n: i for i, n in enumerate(all_ordered)}
        n_unknown = len(unknown_list)
        n_dirichlet = len(dirichlet_list)
        n_full = n_unknown + n_dirichlet

    if n_full == 0:
        empty_S = sp.csr_matrix((0, 0))
        empty_rhs = np.zeros(0, dtype=np.float64)
        return empty_S, empty_rhs, [], {}, {}

    # Step 2: Build tile index maps (local port -> global full-matrix index)
    _cached_l2g = assembly_cache.get('tile_local_to_global', {}) if (assembly_cache and _use_cached_idx) else {}
    tile_index_maps: Dict[Any, np.ndarray] = {}
    _new_l2g: Dict[Any, np.ndarray] = {}
    _cache_valid = True

    for tid, node_list in tile_port_node_lists.items():
        n_local = len(node_list)
        if tid in _cached_l2g:
            l2g = _cached_l2g[tid]
            if len(l2g) == n_local:
                local_to_global = l2g
            else:
                _cache_valid = False
                local_to_global = np.array(
                    [full_node_to_idx[n] for n in node_list], dtype=np.int32
                )
        else:
            local_to_global = np.array(
                [full_node_to_idx[n] for n in node_list], dtype=np.int32
            )
        tile_index_maps[tid] = local_to_global
        _new_l2g[tid] = local_to_global

    # Step 3a: Precompute extra-edge COO entries (always recomputed; small).
    extra_rows_l: List[int] = []
    extra_cols_l: List[int] = []
    extra_data_l: List[float] = []

    if extra_edges:
        for u, v, g in extra_edges:
            if g <= 0:
                continue
            if u == ground_node:
                if v in full_node_to_idx:
                    iv = full_node_to_idx[v]
                    extra_rows_l.append(iv)
                    extra_cols_l.append(iv)
                    extra_data_l.append(g)
                continue
            if v == ground_node:
                if u in full_node_to_idx:
                    iu = full_node_to_idx[u]
                    extra_rows_l.append(iu)
                    extra_cols_l.append(iu)
                    extra_data_l.append(g)
                continue
            if u not in full_node_to_idx or v not in full_node_to_idx:
                continue
            iu, iv = full_node_to_idx[u], full_node_to_idx[v]
            extra_rows_l += [iu, iv, iu, iv]
            extra_cols_l += [iv, iu, iu, iv]
            extra_data_l += [-g, -g, g, g]

    extra_rows_arr = np.array(extra_rows_l, dtype=np.int32) if extra_rows_l else np.empty(0, dtype=np.int32)
    extra_cols_arr = np.array(extra_cols_l, dtype=np.int32) if extra_cols_l else np.empty(0, dtype=np.int32)
    extra_data_arr = np.array(extra_data_l, dtype=np.float64) if extra_data_l else np.empty(0, dtype=np.float64)

    # -------------------------------------------------------------------------
    # FAST PATH: CSR scatter pattern cached from a previous call.
    #
    # Uses get_schur_data_flat() — sends ONLY float64 values (8 B/entry)
    # instead of get_schur_coo_shards() which also sends int32 rows + int32
    # cols (16 B/entry total, 2x the dense S_i size).  The scatter pattern
    # already encodes the (row, col) → G_full_data position mapping, so
    # global indices are not needed on the wire.
    #
    # Peak coordinator extra memory per tile:
    #   one float64 flat array (n_ports^2 * 8 bytes) + no rows/cols overhead.
    #   After scatter-add the array is freed; next tile's data arrives.
    # -------------------------------------------------------------------------
    _scatter_pattern = assembly_cache.get('_csr_scatter_pattern') if assembly_cache else None
    # The extra-edge set MUST be part of the pattern validity check.
    # The single _csr_scatter_pattern cache slot is shared between the DC
    # prepare (extra_edges = resistive package edges only) and the transient
    # prepare (extra_edges = resistive + C_coeff-scaled cap edges).  If a
    # model has package cap edges and streaming_assembly=True, the transient
    # prepare will see a cached DC pattern whose sparsity structure does NOT
    # include the cap columns — scatter-adding N_transient values into an
    # N_dc-sized pattern causes an index-out-of-bounds crash or, worse, silent
    # corruption.  We guard against this by hashing the extra-edge (row, col)
    # index arrays and including the hash + entry count in the validity check.
    _extra_edge_fingerprint = (len(extra_rows_arr), int(np.sum(extra_rows_arr.astype(np.int64)) + np.sum(extra_cols_arr.astype(np.int64))))
    _pattern_valid = (
        _scatter_pattern is not None
        and _cache_valid
        and _use_cached_idx
        and _scatter_pattern.get('n_full') == n_full
        and set(_scatter_pattern.get('tile_scatter_idxs', {}).keys()) == set(tile_index_maps.keys())
        and _scatter_pattern.get('extra_edge_fingerprint') == _extra_edge_fingerprint
    )

    if _pattern_valid:
        # Preallocate G_full data array; indptr/indices come from pattern.
        pat = _scatter_pattern
        G_full_data = np.zeros(pat['nnz'], dtype=np.float64)
        tile_scatter_idxs = pat['tile_scatter_idxs']
        extra_scatter_idx = pat['extra_scatter_idx']

        # Stream flat data tile-by-tile (no rows/cols on the wire).
        # Each result is a 1-D float64 array of length n_ports^2 in row-major
        # order.  scatter_idx[tid] maps each entry to its G_full_data position.
        for i, tile_data_flat in backend.call_all_streaming(
            workers, 'get_schur_data_flat'
        ):
            tid = tile_configs[i].tile_id
            scatter_idx = tile_scatter_idxs[tid]
            np.add.at(G_full_data, scatter_idx, tile_data_flat)
            del tile_data_flat

        # Scatter-add extra-edge values.
        if len(extra_scatter_idx) > 0:
            np.add.at(G_full_data, extra_scatter_idx, extra_data_arr)

        # Reconstruct G_full as CSR from (data, indptr, indices).
        G_full = sp.csr_matrix(
            (G_full_data, pat['indices'], pat['indptr']),
            shape=(n_full, n_full),
        )
        logger.debug(
            "B3 streaming (fast-path, data-only): scatter-add into preallocated CSR "
            "(%d nnz, n_full=%d) — 8 B/entry (no rows/cols wire overhead)",
            pat['nnz'], n_full,
        )

    else:
        # -------------------------------------------------------------------------
        # FIRST CALL (or pattern invalidated): Two-pass bounded-peak path.
        #
        # Pass 1 (index-only, no float64 on the wire):
        #   Stream get_schur_coo_indices_only() one tile at a time to accumulate
        #   only int32 rows/cols (8 B/entry, no float64 values).  After collecting
        #   all tiles, build the CSR sparsity pattern via COO→CSR with dummy data.
        #   Compute and cache scatter positions.  Free all accumulated index arrays.
        #
        # Pass 2 (data-only, scatter into preallocated buffer):
        #   Once the scatter pattern is in assembly_cache, take the same fast path
        #   as subsequent calls: stream get_schur_data_flat() one tile at a time
        #   and scatter-add into a preallocated G_full_data buffer.
        #
        # Peak extra coordinator memory:
        #   Pre-pass accumulation: O(nnz * 8 bytes) for int32 rows+cols (no float64).
        #   After pre-pass: pattern (~12 B/nnz) + preallocated buffer (8 B/nnz).
        #   Data pass: one tile's float64 flat (n_ports^2 * 8 B) at a time.
        #   Combined: O(nnz * 20 bytes) vs old single-pass O(nnz * 28+ bytes).
        # -------------------------------------------------------------------------

        # Build index args: one per tile with the tile_index_map.
        idx_args = [
            (tile_index_maps[tc.tile_id],)
            for tc in tile_configs
        ]

        # --- Pass 1: index-only pre-pass (accumulate int32 rows/cols, NO float64) ---
        coo_rows_parts: List[np.ndarray] = []
        coo_cols_parts: List[np.ndarray] = []

        for i, (idx_rows, idx_cols) in backend.call_all_streaming(
            workers, 'get_schur_coo_indices_only', idx_args
        ):
            coo_rows_parts.append(idx_rows)
            coo_cols_parts.append(idx_cols)
            del idx_rows, idx_cols

        # Add extra-edge indices
        if len(extra_rows_arr) > 0:
            coo_rows_parts.append(extra_rows_arr)
            coo_cols_parts.append(extra_cols_arr)

        # Build CSR sparsity pattern from index-only COO (dummy all-ones data).
        if coo_rows_parts:
            all_rows = np.concatenate(coo_rows_parts).astype(np.int32)
            all_cols = np.concatenate(coo_cols_parts).astype(np.int32)
            # Free the per-part lists now that they're concatenated.
            del coo_rows_parts, coo_cols_parts
            dummy_data = np.ones(len(all_rows), dtype=np.float64)
            G_full_pattern_csr = sp.coo_matrix(
                (dummy_data, (all_rows, all_cols)), shape=(n_full, n_full)
            ).tocsr()
            del all_rows, all_cols, dummy_data
        else:
            G_full_pattern_csr = sp.csr_matrix((n_full, n_full), dtype=np.float64)

        # Compute and cache scatter positions from the index-only CSR pattern.
        _new_pattern = _build_csr_scatter_pattern(
            G_full_csr=G_full_pattern_csr,
            tile_index_maps=tile_index_maps,
            extra_rows=extra_rows_arr,
            extra_cols=extra_cols_arr,
        )
        del G_full_pattern_csr  # free pattern CSR (indptr/indices kept in _new_pattern)

        # Stamp the extra-edge fingerprint into the pattern so _pattern_valid
        # can reject a stale DC pattern when transient prepare arrives with
        # a different extra-edge set (e.g., + cap edges).
        _new_pattern['extra_edge_fingerprint'] = _extra_edge_fingerprint

        if assembly_cache is not None and _cache_valid:
            assembly_cache['_csr_scatter_pattern'] = _new_pattern
            logger.debug(
                "B3 streaming (first-call two-pass): built scatter pattern "
                "via index-only pre-pass (%d nnz, %d tiles).",
                _new_pattern['nnz'], len(tile_index_maps),
            )

        # --- Pass 2: data-only (fast path now that scatter pattern is cached) ---
        _scatter_pattern = _new_pattern
        G_full_data = np.zeros(_scatter_pattern['nnz'], dtype=np.float64)
        tile_scatter_idxs = _scatter_pattern['tile_scatter_idxs']
        extra_scatter_idx = _scatter_pattern['extra_scatter_idx']

        for i, tile_data_flat in backend.call_all_streaming(
            workers, 'get_schur_data_flat'
        ):
            tid = tile_configs[i].tile_id
            scatter_idx = tile_scatter_idxs[tid]
            np.add.at(G_full_data, scatter_idx, tile_data_flat)
            del tile_data_flat

        # Scatter-add extra-edge values.
        if len(extra_scatter_idx) > 0:
            np.add.at(G_full_data, extra_scatter_idx, extra_data_arr)

        # Build G_full CSR from (data, indptr, indices).
        G_full = sp.csr_matrix(
            (G_full_data, _scatter_pattern['indices'], _scatter_pattern['indptr']),
            shape=(n_full, n_full),
        )
        logger.debug(
            "B3 streaming (first-call two-pass): assembled G_full "
            "(%d nnz, n_full=%d) — index pre-pass + data scatter.",
            G_full.nnz, n_full,
        )

    u_idx = np.arange(n_unknown)
    S_global = G_full[np.ix_(u_idx, u_idx)].tocsr()

    if n_dirichlet > 0:
        d_idx = np.arange(n_unknown, n_full)
        G_ud = G_full[np.ix_(u_idx, d_idx)].tocsr()
        V_d = np.full(n_dirichlet, dirichlet_voltage, dtype=np.float64)
        rhs_dirichlet = -(G_ud @ V_d)
    else:
        rhs_dirichlet = np.zeros(n_unknown, dtype=np.float64)

    unknown_to_idx = {n: i for i, n in enumerate(unknown_list)}

    # Build interface-only tile_index_maps for callers (S_extra, CG solver, etc.).
    #
    # The tile_index_maps built above (Step 2) use full_node_to_idx indices
    # (0..n_full-1), where Dirichlet/pad nodes are placed at n_unknown..n_full-1.
    # These are correct for the ASSEMBLY pass (get_schur_coo_indices_only places
    # entries in the right block of G_full).  However, callers that receive the
    # returned tile_index_maps expect *interface-unknown* indices (0..n_unknown-1),
    # consistent with what the bulk path builds at result_factorization.py ~:1058
    # using `if n in interface_node_to_idx`.  A pad node on a tile boundary has
    # full_node_to_idx[pad] >= n_unknown, so passing it downstream causes:
    #   - S_extra scatter: indices >= n_iface → out-of-bounds or wrong entry
    #   - CG tilewise matvec: same corruption
    #   - Tile index map stored on the context: corrupted for save/load paths
    # The fix mirrors the bulk path exactly: filter to unknowns only, remap to
    # unknown_to_idx (which IS interface_node_to_idx after assembly).
    interface_tile_index_maps: Dict[Any, np.ndarray] = {}
    for tid, node_list in tile_port_node_lists.items():
        interface_tile_index_maps[tid] = np.array(
            [unknown_to_idx[n] for n in node_list if n in unknown_to_idx],
            dtype=np.int32,
        )

    # Update assembly cache (same as assemble_schur_complement_system)
    if assembly_cache is not None and _cache_valid:
        assembly_cache['tile_local_to_global'] = _new_l2g
        assembly_cache['full_node_to_idx'] = full_node_to_idx
        assembly_cache['unknown_list'] = unknown_list
        assembly_cache['dirichlet_list'] = dirichlet_list
        assembly_cache['n_unknown'] = n_unknown
        assembly_cache['n_full'] = n_full

    return S_global, rhs_dirichlet, unknown_list, unknown_to_idx, interface_tile_index_maps


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

    # 1. Factor tiles and compute Schur complements (parallel on workers).
    #
    # B3: When streaming_assembly is enabled, use factor_and_cache_schur() to
    # factor tiles and cache S_i on workers WITHOUT returning S_i to the
    # coordinator.  S_global is then assembled by streaming COO shards
    # tile-by-tile.  When streaming_assembly is False (default), use the
    # existing factor_and_compute_schur() path which gathers all dense S_i
    # at once.  This path remains byte-identical to pre-B3 behaviour.
    #
    # B3 'auto' fix: get lightweight size stats BEFORE factoring interior
    # (via get_schur_size_stats()) so the streaming decision is made without
    # first gathering any S_i.  This eliminates the blocker where 'auto' used
    # to run factor_and_compute_schur first (hitting the memory peak), then
    # logged a note that streaming would be used "next time".
    t0 = _time.perf_counter()

    _streaming_setting = _get_streaming_assembly_setting(model)

    if _streaming_setting == 'auto':
        # Cheap stats round-trip (no factoring, no S_i) to decide streaming.
        size_stats_raw = model.backend.call_all(
            model.workers, 'get_schur_size_stats'
        )
        _use_streaming_dc = _should_stream(model, size_stats_raw)
        if verbose or _use_streaming_dc:
            _est = _estimate_schur_peak_bytes(size_stats_raw)
            logger.info(
                "streaming_assembly='auto': estimated S_i peak %.1f MB -> "
                "streaming=%s", _est / 1024 ** 2, _use_streaming_dc,
            )
    else:
        _use_streaming_dc = bool(_streaming_setting)

    tile_schur_complements: Dict[Any, np.ndarray] = {}
    tile_port_node_lists: Dict[Any, List[str]] = {}
    per_tile_stats: List[Dict[str, Any]] = []

    if _use_streaming_dc:
        # streaming_assembly=True (or 'auto' resolved to True):
        # factor_and_cache_schur — S_i stays on workers, not sent to coordinator.
        cache_results = model.backend.call_all(
            model.workers, 'factor_and_cache_schur'
        )
        timings['factor_tiles'] = _time.perf_counter() - t0

        for i, (boundary_list, tile_stats) in enumerate(cache_results):
            tid = model.metadata.tile_configs[i].tile_id
            tile_port_node_lists[tid] = boundary_list
            per_tile_stats.append(tile_stats)

    else:
        # Bulk path (streaming_assembly=False or 'auto' resolved to False):
        # factor_and_compute_schur — gathers all dense S_i to coordinator.
        # This is byte-identical to pre-B3 behaviour.
        schur_results = model.backend.call_all(
            model.workers, 'factor_and_compute_schur'
        )
        timings['factor_tiles'] = _time.perf_counter() - t0

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

    # A4: Determine assembly cache.
    # When topology already exists (repeated DC prepare or post-transient-first):
    #   reuse or lazily initialise the cache on the topology object.
    # When topology is None (first DC prepare): create a local dict that will be
    #   stored on the new DistributedTopologyContext created at the end of this
    #   function.  This ensures prepare_transient() finds a pre-populated cache
    #   and its primary assemble_schur_complement_system call gets a cache HIT
    #   (skipping full_node_to_idx rebuild and per-tile local_to_global lookups).
    if ctx.topology is not None:
        if getattr(ctx.topology, '_assembly_cache', None) is None:
            ctx.topology._assembly_cache = {}
        _asm_cache: Dict[str, Any] = ctx.topology._assembly_cache
    else:
        # Will be attached to the topology object created at step 5 below.
        _asm_cache = {}

    if _use_streaming_dc:
        # B3 streaming path: COO shards streamed tile-by-tile.
        # tile_schur_complements is empty; S_i lives on workers.
        (
            S_global, rhs_dirichlet, interface_nodes, interface_node_to_idx,
            _streaming_tile_index_maps,
        ) = _stream_assemble_schur(
            model=model,
            tile_port_node_lists=tile_port_node_lists,
            per_tile_stats=per_tile_stats,
            extra_edges=model.package_data.package_edges,
            dirichlet_nodes=model.pad_nodes,
            dirichlet_voltage=model.vdd,
            assembly_cache=_asm_cache,
        )
        logger.debug(
            "DC streaming assembly complete: S_global %dx%d, nnz=%d",
            S_global.shape[0], S_global.shape[1], S_global.nnz,
        )
    else:
        from pgmath.schur import assemble_schur_complement_system
        S_global, rhs_dirichlet, interface_nodes, interface_node_to_idx = (
            assemble_schur_complement_system(
                tile_schur_complements=tile_schur_complements,
                tile_port_node_lists=tile_port_node_lists,
                extra_edges=model.package_data.package_edges,
                dirichlet_nodes=model.pad_nodes,
                dirichlet_voltage=model.vdd,
                assembly_cache=_asm_cache,
            )
        )
        _streaming_tile_index_maps = None  # not used in bulk path

    timings['assemble_interface'] = _time.perf_counter() - t0

    # 2b. Global interface island detection (DC mode — resistive-only extra_edges).
    # Cache is keyed on topology.island_nodes (the DC field).  A second DC
    # prepare() call skips the BFS via cache hit.
    # Cross-mode reuse NOTE: when package_cap_edges is empty (combined_edges ==
    # resistive_edges) the BFS result is identical for DC and transient, so
    # the cache is also written to island_nodes_td to enable transient skipping.
    # When cap edges are present, island_nodes_td is NOT set here; the transient
    # path runs its own BFS so it uses the correct (smaller) island set.
    t0 = _time.perf_counter()
    _cached_islands: Optional[Set[str]] = (
        getattr(ctx.topology, 'island_nodes', None)
        if ctx.topology is not None else None
    )
    if _cached_islands is not None:
        # Cache hit: apply penalty from prior DC BFS result; skip re-detection.
        island_nodes = _cached_islands
        if island_nodes:
            from pgmath.schur import apply_island_penalty
            S_global, rhs_dirichlet = apply_island_penalty(
                S_global, rhs_dirichlet, island_nodes,
                interface_node_to_idx, model.vdd,
            )
            logger.debug(
                "DC island detection: cache hit (%d islands), BFS skipped.",
                len(island_nodes),
            )
    else:
        # Cache miss: Stage 1e summaries union-find, or legacy Schur-BFS
        # (resistive-only extra_edges) -- resolved once at model creation.
        S_global, rhs_dirichlet, island_nodes = _detect_islands_dispatch(
            model, S_global, rhs_dirichlet, interface_nodes, interface_node_to_idx,
            extra_edges=model.package_data.package_edges,
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

    # 3. Build tile index maps (local port indices -> global interface indices).
    # Must be built BEFORE step 4 (interface solver setup) so that CG tilewise
    # mode can reference tile_index_maps when constructing InterfaceCGSolver.
    # B3: When streaming was used, tile_index_maps were already built during
    # _stream_assemble_schur (stored in _streaming_tile_index_maps).  Reuse
    # them directly to avoid rebuilding.  For the bulk path, build as before.
    if _streaming_tile_index_maps is not None:
        tile_index_maps: Dict[Tuple[int, int], np.ndarray] = _streaming_tile_index_maps
    else:
        tile_index_maps = {}
        for tid, boundary_list in tile_port_node_lists.items():
            local_to_global = np.array(
                [interface_node_to_idx[n] for n in boundary_list if n in interface_node_to_idx],
                dtype=np.int32,
            )
            tile_index_maps[tid] = local_to_global

    # 4. Factor (or set up iterative solver for) interface system.
    #
    # B2: The interface_solver setting controls whether a direct CHOLMD/SuperLU
    # factorization or iterative CG is used.  'auto' selects direct for small
    # systems (n_interface < 200K) and CG for large ones.  The default 'auto'
    # maps to 'direct' for netlist_sampled (n_interface ~2-4K), so existing
    # model behaviour is unchanged.
    t0 = _time.perf_counter()
    _iface_solver_setting = _get_interface_solver_setting(model)
    _model_settings = getattr(model, 'settings', {}) if model is not None else {}

    _cg_stats: Dict[str, Any] = {}
    _cg_solver = None  # InterfaceCGSolver if CG is used, else None

    # Resolve 'auto' here (before branching) so we can log it
    if _iface_solver_setting == 'auto':
        from .interface_iterative import (
            auto_select_interface_solver, resolve_factor_memory_budget_bytes,
        )
        _resolved_budget_bytes = resolve_factor_memory_budget_bytes(
            _model_settings.get('interface_factor_memory_budget', 'auto')
        )
        _iface_resolved_mode = auto_select_interface_solver(
            len(interface_nodes), S_global,
            factor_memory_budget_bytes=_resolved_budget_bytes,
        )
    else:
        _iface_resolved_mode = _iface_solver_setting

    if _iface_resolved_mode == 'direct':
        from pgmath.factor import _factor_conductance_matrix
        interface_lu_result = _factor_conductance_matrix(
            S_global, verbose=False, config=model.coordinator_solver_config,
        )
    else:
        # CG mode -- routed through build_interface_solver (Stage 1d: the
        # single factory used by both DC and transient factor/refactor paths).
        from .interface_iterative import (
            build_interface_solver, resolve_block_jacobi_max_bytes,
        )
        _matvec_mode = _model_settings.get('interface_matvec_mode', 'assembled')
        _preconditioner = _model_settings.get('interface_preconditioner', 'block_jacobi')
        _cg_rtol = float(_model_settings.get('interface_cg_rtol', 1e-8))
        _cg_atol = float(_model_settings.get('interface_cg_atol', 1e-14))
        _cg_maxiter_raw = _model_settings.get('interface_cg_maxiter', None)
        _cg_maxiter = (
            None if _cg_maxiter_raw is None
            else _coerce_int(_cg_maxiter_raw, 'interface_cg_maxiter')
        )
        _cg_strict = _coerce_bool(
            _model_settings.get('interface_cg_strict', True), 'interface_cg_strict'
        )
        _bj_max_bytes = resolve_block_jacobi_max_bytes(
            _model_settings.get('interface_block_jacobi_max_bytes', 'auto')
        )

        # For tilewise mode, compute the package-edge contribution not included
        # in per-tile Schur complements.  S_extra = S_global - sum_i(P_i^T S_i P_i).
        # B2 follow-up: replaced nested Python loop (LIL) with vectorized COO.
        # B3: When streaming was used, tile_schur_complements is empty; tilewise
        # CG mode is not available (can't build S_extra without S_i).  Fall back
        # to 'assembled' mode and log a warning.
        _S_extra: Optional[sp.spmatrix] = None
        if _matvec_mode == 'tilewise':
            if _use_streaming_dc or not tile_schur_complements:
                # Non-composition: streaming_assembly=True did not gather S_i to the
                # coordinator, so tilewise matvec (which needs per-tile S_i blocks) is
                # not possible.  Fall back to 'assembled' mode: CG still avoids the
                # CHOLMOD factor, but S_global remains in coordinator memory.
                # The ideal composition (S_i worker-resident RPC matvec) is deferred
                # to B4.  See module docstring for the full compatibility matrix.
                logger.warning(
                    "streaming_assembly=True is incompatible with "
                    "interface_matvec_mode='tilewise': S_i blocks were not gathered "
                    "to the coordinator (streaming intentionally avoids this). "
                    "Falling back to matvec_mode='assembled' — CG still avoids the "
                    "CHOLMOD factor; only the ~50GB S_global itself is held. "
                    "For a fully worker-resident matvec, see B4 (future work)."
                )
                _matvec_mode = 'assembled'
            else:
                # B2 follow-up: vectorized COO S_extra construction
                _S_extra = _build_s_extra_coo(
                    S_global_csr=S_global.tocsr(),
                    tile_schur_complements=tile_schur_complements,
                    tile_index_maps=tile_index_maps,
                    n_iface=len(interface_nodes),
                )

        _cg_solve_callable, _cg_resolved_mode, _cg_solver = build_interface_solver(
            S_global=S_global,
            interface_solver='cg',
            tile_schur_complements=tile_schur_complements if not _use_streaming_dc else None,
            tile_index_maps=tile_index_maps,
            S_extra=_S_extra,
            matvec_mode=_matvec_mode,
            preconditioner=_preconditioner,
            rtol=_cg_rtol,
            atol=_cg_atol,
            maxiter=_cg_maxiter,
            strict=_cg_strict,
            block_jacobi_max_bytes=_bj_max_bytes,
            verbose=verbose,
            cg_stats_dict=_cg_stats,
        )

        # Synthetic stats object (matching SparseFactorAdapter fields used below)
        class _CGSolveResult:
            backend = 'cg'
            backend_info = (
                f"CG/{_cg_solver.matvec_mode}/"
                f"precond={_cg_solver.preconditioner}"
            )
            resolved_mode = 'cg'
            solve = _cg_solve_callable

        interface_lu_result = _CGSolveResult()
        if verbose:
            logger.info(
                "Interface CG solver: mode=%s, precond=%s, rtol=%.2e, n=%d",
                _matvec_mode, _preconditioner, _cg_rtol, len(interface_nodes),
            )

    timings['factor_interface'] = _time.perf_counter() - t0

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
        'resolved_mode': interface_lu_result.resolved_mode,
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
    # B2: store resolved interface solver mode and optional CG solver
    ctx._interface_solver_mode = _iface_resolved_mode
    ctx._cg_solver = _cg_solver  # InterfaceCGSolver or None

    # Build topology context (if not already provided)
    # Cross-mode cache: when package_cap_edges is empty, DC and transient BFS
    # produce identical results, so pre-populate island_nodes_td here to let
    # prepare_transient() skip its BFS.  When cap edges are present, leave
    # island_nodes_td=None so transient runs its own mode-correct BFS.
    _pkg_has_cap = bool(model.package_data.package_cap_edges)
    if ctx.topology is None:
        # A4: _asm_cache was populated by the DC assemble call above.
        # Storing it on the topology lets prepare_transient() reuse the
        # per-tile local_to_global index arrays and full_node_to_idx map,
        # avoiding a full rebuild for its primary assemble call.
        ctx.topology = DistributedTopologyContext(
            interface_nodes=interface_nodes,
            interface_node_to_idx=interface_node_to_idx,
            tile_index_maps=tile_index_maps,
            rhs_dirichlet_G=rhs_dirichlet,
            G_package_uu=G_pkg_uu if G_pkg_uu.nnz > 0 else None,
            removed_interface_nodes=island_nodes,
            island_nodes=island_nodes,  # DC-mode cache
            island_nodes_td=None if _pkg_has_cap else island_nodes,  # transient cross-mode (safe when no caps)
            _assembly_cache=_asm_cache,  # A4: pass DC-populated cache
        )
    elif getattr(ctx.topology, 'island_nodes', None) is None:
        # Topology exists (from prior prepare or loaded from old checkpoint)
        # but DC island_nodes not yet cached — store the freshly-computed result.
        ctx.topology.island_nodes = island_nodes
        if not _pkg_has_cap and getattr(ctx.topology, 'island_nodes_td', None) is None:
            # Also populate transient cache (safe: same BFS inputs when no caps).
            ctx.topology.island_nodes_td = island_nodes

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
        # B2: persist the resolved interface solver mode so refactor() can
        # reconstruct the same callable type.  Per-tile S_i blocks are NOT
        # saved (too large; always recomputed in tilewise mode).
        'interface_solver_mode': getattr(ctx, '_interface_solver_mode', 'direct'),
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
    # B2: restore interface_solver_mode (default 'direct' for old checkpoints)
    ctx._interface_solver_mode = data.get('interface_solver_mode', 'direct')
    ctx._cg_solver = None  # Re-created on refactor()
    _restore_role_configs(data, model)
    # NOT factored -- caller must call refactor() or factor()
    logger.info("Loaded DC context from %s (is_factored=False)", path)
    return ctx


def _refactor_dc_context(ctx: 'DistributedSolverContext', verbose: bool = False) -> None:
    """Rebuild coordinator solve callable from saved S_global (DC).

    For direct mode: rebuilds the CHOLMOD/SuperLU LU factorization.
    For CG mode: reconstructs the InterfaceCGSolver (assembled matvec only;
      tilewise mode requires a full factor() to re-obtain per-tile S_i blocks).

    Workers must already be factored before calling this.
    """
    if ctx._S_global is None:
        raise RuntimeError(
            "Cannot refactor without S_global. Use factor() for a "
            "full factorization, or load a checkpoint that includes "
            "S_global."
        )

    # Determine the mode from the context (set during factor() or load())
    _mode = getattr(ctx, '_interface_solver_mode', 'direct')
    # Allow override via model settings (e.g. if user changes setting between
    # save and load+refactor).
    _model_setting = _get_interface_solver_setting(ctx.model)
    if _model_setting != 'auto':
        # Explicit setting overrides the stored mode
        _mode = _model_setting

    coord_config = ctx.model.coordinator_solver_config if ctx.model is not None else None
    t0 = _time.perf_counter()

    if _mode == 'direct':
        from pgmath.factor import _factor_conductance_matrix
        interface_lu_result = _factor_conductance_matrix(
            ctx._S_global, verbose=verbose, config=coord_config,
        )
        ctx._interface_lu = interface_lu_result.solve
        ctx._cg_solver = None
        ctx._interface_solver_mode = 'direct'
    else:
        # CG assembled mode (no per-tile S_i needed) -- routed through
        # build_interface_solver (Stage 1d).
        from .interface_iterative import (
            build_interface_solver, resolve_block_jacobi_max_bytes,
        )
        _model_settings = getattr(ctx.model, 'settings', {}) if ctx.model is not None else {}
        _cg_rtol = float(_model_settings.get('interface_cg_rtol', 1e-8))
        _cg_atol = float(_model_settings.get('interface_cg_atol', 1e-14))
        _cg_maxiter_raw = _model_settings.get('interface_cg_maxiter', None)
        _cg_maxiter = (
            None if _cg_maxiter_raw is None
            else _coerce_int(_cg_maxiter_raw, 'interface_cg_maxiter')
        )
        _cg_strict = _coerce_bool(
            _model_settings.get('interface_cg_strict', True), 'interface_cg_strict'
        )
        _preconditioner = _model_settings.get('interface_preconditioner', 'block_jacobi')
        _bj_max_bytes = resolve_block_jacobi_max_bytes(
            _model_settings.get('interface_block_jacobi_max_bytes', 'auto')
        )
        # Finding 1: tile_index_maps is required for block_jacobi ownership
        # assignment even in 'assembled' matvec mode (InterfaceCGSolver uses
        # it to assign each interface node to an owning tile, then extracts
        # that tile's principal submatrix from the assembled S_global).
        # ctx.topology is restored from the saved checkpoint by load(), so
        # tile_index_maps is normally available here -- pass it through so
        # the preconditioner rebuilt on refactor() actually matches the one
        # built during the original factor().  If topology genuinely lacks
        # it (e.g. a pre-B2 checkpoint), degrade gracefully to the diagonal
        # 'jacobi' preconditioner rather than silently building none at all.
        _tile_index_maps = (
            getattr(ctx.topology, 'tile_index_maps', None)
            if ctx.topology is not None else None
        )
        if _preconditioner == 'block_jacobi' and not _tile_index_maps:
            logger.warning(
                "Refactor: tile_index_maps unavailable on ctx.topology (missing "
                "or predates this field), so the 'block_jacobi' preconditioner "
                "cannot be rebuilt after load()+refactor(). Degrading to "
                "'jacobi' (diagonal) for this refactor -- expect MORE CG "
                "iterations per solve than block_jacobi would need. Call "
                "factor() (not just refactor()) to restore full block_jacobi "
                "support."
            )
            _preconditioner = 'jacobi'
        _cg_stats: Dict[str, Any] = {}
        _solve_callable, _resolved_backend, cg_solver = build_interface_solver(
            S_global=ctx._S_global,
            interface_solver='cg',
            matvec_mode='assembled',  # tilewise needs per-tile S_i; use assembled on refactor
            tile_index_maps=_tile_index_maps,
            preconditioner=_preconditioner,
            rtol=_cg_rtol,
            atol=_cg_atol,
            maxiter=_cg_maxiter,
            strict=_cg_strict,
            block_jacobi_max_bytes=_bj_max_bytes,
            verbose=verbose,
            cg_stats_dict=_cg_stats,
        )
        ctx._interface_lu = _solve_callable
        ctx._cg_solver = cg_solver
        ctx._interface_solver_mode = 'cg'

    elapsed = _time.perf_counter() - t0
    ctx.is_factored = True
    logger.info(
        "Refactored DC coordinator solve (%s) from saved S_global in %.3fs",
        _mode, elapsed,
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

    # 1. Factor transient system on all workers (parallel).
    #
    # B3: Same auto/streaming logic as DC.  When streaming_assembly is True
    # or 'auto' resolves to True (via lightweight get_schur_size_stats()),
    # use factor_transient_and_cache_schur() which keeps S_A on the worker.
    # Assembly then streams COO shards via _stream_assemble_schur().
    dt_scaled = ctx.dt_scaled
    C_coeff = ctx.C_coeff

    _streaming_setting_td = _get_streaming_assembly_setting(model)

    if _streaming_setting_td == 'auto':
        # Lightweight round-trip to decide streaming WITHOUT factoring.
        size_stats_raw = model.backend.call_all(
            model.workers, 'get_schur_size_stats'
        )
        _use_streaming_td = _should_stream(model, size_stats_raw)
        if verbose or _use_streaming_td:
            _est = _estimate_schur_peak_bytes(size_stats_raw)
            logger.info(
                "Transient streaming_assembly='auto': estimated S_A peak "
                "%.1f MB -> streaming=%s", _est / 1024 ** 2, _use_streaming_td,
            )
    else:
        _use_streaming_td = bool(_streaming_setting_td)

    t0 = _time.perf_counter()
    trans_args = [(dt_scaled, method)] * len(tile_configs)

    tile_schur_complements: Dict[Any, np.ndarray] = {}
    tile_port_node_lists: Dict[Any, List[str]] = {}
    total_tile_cap = 0.0
    per_tile_stats: List[Dict[str, Any]] = []

    if _use_streaming_td:
        # Streaming path: S_A stays on workers.
        cache_results = model.backend.call_all(
            model.workers, 'factor_transient_and_cache_schur', trans_args,
        )
        timings['factor_transient_tiles'] = _time.perf_counter() - t0

        for i, (port_list, tile_cap, tile_stats) in enumerate(cache_results):
            tid = tile_configs[i].tile_id
            tile_port_node_lists[tid] = port_list
            total_tile_cap += tile_cap
            per_tile_stats.append(tile_stats)

    else:
        # Bulk path: gathers all dense S_A_i to coordinator.
        schur_results = model.backend.call_all(
            model.workers, 'factor_transient_system', trans_args,
        )
        timings['factor_transient_tiles'] = _time.perf_counter() - t0

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
    )

    # A4: Determine assembly cache.
    # When topology already exists (post-DC or repeated transient): reuse or
    #   lazily initialise the cache on the topology.  The DC call already
    #   populated it, so the primary (combined_edges) assemble below gets
    #   a cache HIT and skips the full_node_to_idx + local_to_global rebuild.
    # When topology is None (transient-first, no prior DC prepare): create a
    #   local dict that will be stored on the new DistributedTopologyContext
    #   created at the end of this function.  The G-only secondary assemble
    #   below then also gets a cache HIT.
    if ctx.topology is not None:
        if getattr(ctx.topology, '_assembly_cache', None) is None:
            ctx.topology._assembly_cache = {}
        _asm_cache: Dict[str, Any] = ctx.topology._assembly_cache
    else:
        # Will be attached to the topology object created below.
        _asm_cache = {}

    _streaming_tile_index_maps_td = None

    if _use_streaming_td:
        # B3 streaming transient: COO shards of S_A streamed tile-by-tile.
        # combined_edges includes the effective cap contribution (C_coeff * C).
        (
            S_global, rhs_dirichlet_A, interface_nodes, interface_node_to_idx,
            _streaming_tile_index_maps_td,
        ) = _stream_assemble_schur(
            model=model,
            tile_port_node_lists=tile_port_node_lists,
            per_tile_stats=per_tile_stats,
            extra_edges=combined_edges,
            dirichlet_nodes=model.pad_nodes,
            dirichlet_voltage=model.vdd,
            assembly_cache=_asm_cache,
        )
        # Compute G-only Dirichlet RHS from package resistive edges alone
        # (tile S_A_i only contribute to the unknown-unknown block, not G_ud).
        unknown_to_idx_td = {n: i for i, n in enumerate(interface_nodes)
                             if n not in (model.pad_nodes or set())}
        # Build sorted unknown list (interface_nodes is already unknowns only)
        rhs_dirichlet_G = _compute_rhs_dirichlet_from_edges(
            extra_edges=list(pkg_res_edges),
            unknown_list=list(interface_nodes),
            unknown_to_idx={n: i for i, n in enumerate(interface_nodes)},
            dirichlet_nodes=model.pad_nodes,
            dirichlet_voltage=model.vdd,
        )
        logger.debug(
            "Transient streaming assembly complete: S_global %dx%d, nnz=%d",
            S_global.shape[0], S_global.shape[1], S_global.nnz,
        )
    else:
        S_global, rhs_dirichlet_A, interface_nodes, interface_node_to_idx = (
            assemble_schur_complement_system(
                tile_schur_complements=tile_schur_complements,
                tile_port_node_lists=tile_port_node_lists,
                extra_edges=combined_edges,
                dirichlet_nodes=model.pad_nodes,
                dirichlet_voltage=model.vdd,
                assembly_cache=_asm_cache,
            )
        )

        # Also compute G-only Dirichlet RHS (without cap contributions)
        # needed for correct transient RHS formulation.
        # Reuse the same cache — the node ordering from the first call is consistent.
        _, rhs_dirichlet_G, _, _ = assemble_schur_complement_system(
            tile_schur_complements=tile_schur_complements,
            tile_port_node_lists=tile_port_node_lists,
            extra_edges=list(pkg_res_edges),
            dirichlet_nodes=model.pad_nodes,
            dirichlet_voltage=model.vdd,
            assembly_cache=_asm_cache,
        )

    # Island detection on transient system (resistive + cap extra_edges).
    # Cache lookup order:
    #   1. island_nodes_td  — transient-mode cache (always mode-correct).
    #   2. island_nodes     — DC-mode cache; ONLY reused when package_cap_edges
    #                         is empty, because cap edges can bridge components
    #                         that are resistively-disconnected, making the
    #                         transient island set a strict subset of the DC set.
    #                         Reusing the DC result for transient when caps exist
    #                         would over-penalise cap-bridged interface nodes.
    # Old-format checkpoints (no island_nodes_td field) fall back gracefully
    # via getattr(..., None).
    _t_island = _time.perf_counter()
    _td_has_cap = bool(model.package_data.package_cap_edges)
    _cached_islands_td: Optional[Set[str]] = None
    if ctx.topology is not None:
        # 1. Mode-correct transient cache
        _cached_islands_td = getattr(ctx.topology, 'island_nodes_td', None)
        if _cached_islands_td is None and not _td_has_cap:
            # 2. Cross-mode DC cache (safe only when combined_edges == resistive_edges)
            _cached_islands_td = getattr(ctx.topology, 'island_nodes', None)

    if _cached_islands_td is not None:
        # Cache hit: apply penalty from prior BFS result; skip re-detection.
        island_nodes = _cached_islands_td
        if island_nodes:
            from pgmath.schur import apply_island_penalty
            S_global, rhs_dirichlet_A = apply_island_penalty(
                S_global, rhs_dirichlet_A, island_nodes,
                interface_node_to_idx, model.vdd,
            )
            logger.debug(
                "Transient island detection: cache hit (%d islands), BFS skipped.",
                len(island_nodes),
            )
    else:
        # Cache miss: Stage 1e summaries union-find, or legacy Schur-BFS
        # (resistive + cap extra_edges) -- resolved once at model creation.
        S_global, rhs_dirichlet_A, island_nodes = _detect_islands_dispatch(
            model, S_global, rhs_dirichlet_A, interface_nodes, interface_node_to_idx,
            extra_edges=combined_edges,
        )
    timings['detect_interface_islands'] = _time.perf_counter() - _t_island

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

    # 5. Build tile index maps (must be before step 6 so CG tilewise can use them).
    # B3: Reuse streaming-built maps when available.
    if _streaming_tile_index_maps_td is not None:
        tile_index_maps: Dict[Tuple[int, int], np.ndarray] = _streaming_tile_index_maps_td
    else:
        tile_index_maps = {}
        for tid, port_list in tile_port_node_lists.items():
            local_to_global = np.array(
                [interface_node_to_idx[n] for n in port_list
                 if n in interface_node_to_idx],
                dtype=np.int32,
            )
            tile_index_maps[tid] = local_to_global

    # 6. Factor (or set up iterative solver for) transient interface system.
    # B2: Same auto-select logic as DC context (same model settings apply).
    t0 = _time.perf_counter()
    _iface_solver_setting_td = _get_interface_solver_setting(model)
    _model_settings_td = getattr(model, 'settings', {}) if model is not None else {}

    _cg_stats_td: Dict[str, Any] = {}
    _cg_solver_td = None

    if _iface_solver_setting_td == 'auto':
        from .interface_iterative import (
            auto_select_interface_solver, resolve_factor_memory_budget_bytes,
        )
        _resolved_budget_bytes_td = resolve_factor_memory_budget_bytes(
            _model_settings_td.get('interface_factor_memory_budget', 'auto')
        )
        _iface_resolved_mode_td = auto_select_interface_solver(
            len(interface_nodes), S_global,
            factor_memory_budget_bytes=_resolved_budget_bytes_td,
        )
    else:
        _iface_resolved_mode_td = _iface_solver_setting_td

    if _iface_resolved_mode_td == 'direct':
        from pgmath.factor import _factor_conductance_matrix
        interface_lu_result = _factor_conductance_matrix(
            S_global, verbose=verbose, config=model.coordinator_solver_config,
        )
    else:
        # CG mode for transient -- routed through build_interface_solver
        # (Stage 1d: the same factory used by the DC factor/refactor paths).
        from .interface_iterative import (
            build_interface_solver, resolve_block_jacobi_max_bytes,
        )
        _matvec_mode_td = _model_settings_td.get('interface_matvec_mode', 'assembled')
        _preconditioner_td = _model_settings_td.get('interface_preconditioner', 'block_jacobi')
        _cg_rtol_td = float(_model_settings_td.get('interface_cg_rtol', 1e-8))
        _cg_atol_td = float(_model_settings_td.get('interface_cg_atol', 1e-14))
        _cg_maxiter_td_raw = _model_settings_td.get('interface_cg_maxiter', None)
        _cg_maxiter_td = (
            None if _cg_maxiter_td_raw is None
            else _coerce_int(_cg_maxiter_td_raw, 'interface_cg_maxiter')
        )
        _cg_strict_td = _coerce_bool(
            _model_settings_td.get('interface_cg_strict', True), 'interface_cg_strict'
        )
        _bj_max_bytes_td = resolve_block_jacobi_max_bytes(
            _model_settings_td.get('interface_block_jacobi_max_bytes', 'auto')
        )

        # For tilewise mode: compute S_extra (package-edge contribution).
        # B2 follow-up: replaced nested Python loop (LIL) with vectorized COO.
        # B3: When streaming was used, tile_schur_complements is empty; fall back.
        # See module docstring for the streaming vs CG-tilewise compatibility matrix.
        _S_extra_td: Optional[sp.spmatrix] = None
        if _matvec_mode_td == 'tilewise':
            if _use_streaming_td or not tile_schur_complements:
                # Non-composition: streaming_assembly=True did not gather S_i to the
                # coordinator, so tilewise matvec (which needs per-tile S_i) is not
                # possible.  Fall back to 'assembled'; CG still avoids the CHOLMOD
                # factor.  Worker-resident RPC matvec is deferred to B4.
                logger.warning(
                    "Transient streaming_assembly=True is incompatible with "
                    "interface_matvec_mode='tilewise': S_i blocks were not gathered. "
                    "Falling back to matvec_mode='assembled'. "
                    "See module docstring for the compatibility matrix."
                )
                _matvec_mode_td = 'assembled'
            else:
                _S_extra_td = _build_s_extra_coo(
                    S_global_csr=S_global.tocsr(),
                    tile_schur_complements=tile_schur_complements,
                    tile_index_maps=tile_index_maps,
                    n_iface=len(interface_nodes),
                )

        _cg_solve_callable_td, _cg_resolved_mode_td, _cg_solver_td = build_interface_solver(
            S_global=S_global,
            interface_solver='cg',
            tile_schur_complements=tile_schur_complements if not _use_streaming_td else None,
            tile_index_maps=tile_index_maps,
            S_extra=_S_extra_td,
            matvec_mode=_matvec_mode_td,
            preconditioner=_preconditioner_td,
            rtol=_cg_rtol_td,
            atol=_cg_atol_td,
            maxiter=_cg_maxiter_td,
            strict=_cg_strict_td,
            block_jacobi_max_bytes=_bj_max_bytes_td,
            verbose=verbose,
            cg_stats_dict=_cg_stats_td,
        )

        class _CGSolveResultTD:
            backend = 'cg'
            backend_info = (
                f"CG/{_cg_solver_td.matvec_mode}/"
                f"precond={_cg_solver_td.preconditioner}"
            )
            resolved_mode = 'cg'
            solve = _cg_solve_callable_td

        interface_lu_result = _CGSolveResultTD()
        if verbose:
            logger.info(
                "Transient interface CG solver: mode=%s, precond=%s, rtol=%.2e, n=%d",
                _matvec_mode_td, _preconditioner_td, _cg_rtol_td, len(interface_nodes),
            )

    timings['factor_transient_interface'] = _time.perf_counter() - t0

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
        'resolved_mode': interface_lu_result.resolved_mode,
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
    # B2: store resolved interface solver mode and optional CG solver.
    # The transient time loop uses warm-start from v_gamma_old; the CG solver
    # retains the last solution as x0 automatically via InterfaceCGSolver.
    ctx._interface_solver_mode = _iface_resolved_mode_td
    ctx._cg_solver = _cg_solver_td  # InterfaceCGSolver or None

    # Build topology context if not already provided.
    # Cross-mode cache: when package_cap_edges is empty, DC and transient BFS
    # produce identical results, so pre-populate island_nodes (DC field) to
    # let a subsequent DC prepare() skip its BFS.  When cap edges are present,
    # leave island_nodes=None so DC runs its own mode-correct BFS.
    if ctx.topology is None:
        # A4: _asm_cache was populated by the primary transient assemble call
        # above.  Storing it on the topology lets subsequent prepare() or
        # prepare_transient() calls reuse the index maps without rebuilding.
        ctx.topology = DistributedTopologyContext(
            interface_nodes=interface_nodes,
            interface_node_to_idx=interface_node_to_idx,
            tile_index_maps=tile_index_maps,
            rhs_dirichlet_G=rhs_dirichlet_G,
            G_package_uu=G_pkg_uu if G_pkg_uu.nnz > 0 else None,
            removed_interface_nodes=island_nodes,
            island_nodes=None if _td_has_cap else island_nodes,  # DC cross-mode (safe when no caps)
            island_nodes_td=island_nodes,  # transient-mode cache
            _assembly_cache=_asm_cache,  # A4: pass transient-populated cache
        )
    else:
        # Topology already exists (shared with DC context or loaded checkpoint).
        # Populate mode-specific fields if not yet cached.
        if getattr(ctx.topology, 'island_nodes_td', None) is None:
            ctx.topology.island_nodes_td = island_nodes
        if not _td_has_cap and getattr(ctx.topology, 'island_nodes', None) is None:
            # Also populate DC cache (safe: same BFS inputs when no caps).
            ctx.topology.island_nodes = island_nodes

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
        # B2: persist the resolved interface solver mode
        'interface_solver_mode': getattr(ctx, '_interface_solver_mode', 'direct'),
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
    # B2: restore interface_solver_mode (default 'direct' for old checkpoints)
    ctx._interface_solver_mode = data.get('interface_solver_mode', 'direct')
    ctx._cg_solver = None  # Re-created on refactor()
    _restore_role_configs(data, model)
    # NOT factored -- caller must call refactor() or factor()
    logger.info(
        "Loaded transient context from %s (is_factored=False)", path,
    )
    return ctx


def _refactor_transient_context(
    ctx: 'DistributedTransientContext', verbose: bool = False
) -> None:
    """Rebuild coordinator solve callable from saved S_global (transient).

    For direct mode: rebuilds the CHOLMD/SuperLU LU factorization.
    For CG mode: reconstructs the InterfaceCGSolver (assembled matvec only).

    Workers must already be factored before calling this.
    """
    if ctx._S_global is None:
        raise RuntimeError(
            "Cannot refactor without S_global. Use factor() for a "
            "full factorization, or load a checkpoint that includes "
            "S_global."
        )

    _mode = getattr(ctx, '_interface_solver_mode', 'direct')
    _model_setting = _get_interface_solver_setting(ctx.model)
    if _model_setting != 'auto':
        _mode = _model_setting

    coord_config = ctx.model.coordinator_solver_config if ctx.model is not None else None
    t0 = _time.perf_counter()

    if _mode == 'direct':
        from pgmath.factor import _factor_conductance_matrix
        interface_lu_result = _factor_conductance_matrix(
            ctx._S_global, verbose=verbose, config=coord_config,
        )
        ctx._interface_lu = interface_lu_result.solve
        ctx._cg_solver = None
        ctx._interface_solver_mode = 'direct'
    else:
        # CG assembled mode -- routed through build_interface_solver (Stage 1d).
        from .interface_iterative import (
            build_interface_solver, resolve_block_jacobi_max_bytes,
        )
        _model_settings = getattr(ctx.model, 'settings', {}) if ctx.model is not None else {}
        _cg_rtol = float(_model_settings.get('interface_cg_rtol', 1e-8))
        _cg_atol = float(_model_settings.get('interface_cg_atol', 1e-14))
        _cg_maxiter_raw = _model_settings.get('interface_cg_maxiter', None)
        _cg_maxiter = (
            None if _cg_maxiter_raw is None
            else _coerce_int(_cg_maxiter_raw, 'interface_cg_maxiter')
        )
        _cg_strict = _coerce_bool(
            _model_settings.get('interface_cg_strict', True), 'interface_cg_strict'
        )
        _preconditioner = _model_settings.get('interface_preconditioner', 'block_jacobi')
        _bj_max_bytes = resolve_block_jacobi_max_bytes(
            _model_settings.get('interface_block_jacobi_max_bytes', 'auto')
        )
        # Finding 1: see the matching comment in _refactor_dc_context -- the
        # 'block_jacobi' preconditioner needs tile_index_maps for ownership
        # assignment even in 'assembled' matvec mode; without it, CG silently
        # runs unpreconditioned.  ctx.topology is restored by load(), so it
        # is normally available; degrade gracefully with a WARNING if not.
        _tile_index_maps = (
            getattr(ctx.topology, 'tile_index_maps', None)
            if ctx.topology is not None else None
        )
        if _preconditioner == 'block_jacobi' and not _tile_index_maps:
            logger.warning(
                "Refactor: tile_index_maps unavailable on ctx.topology (missing "
                "or predates this field), so the 'block_jacobi' preconditioner "
                "cannot be rebuilt after load()+refactor(). Degrading to "
                "'jacobi' (diagonal) for this refactor -- expect MORE CG "
                "iterations per solve than block_jacobi would need. Call "
                "factor() (not just refactor()) to restore full block_jacobi "
                "support."
            )
            _preconditioner = 'jacobi'
        _cg_stats: Dict[str, Any] = {}
        _solve_callable, _resolved_backend, cg_solver = build_interface_solver(
            S_global=ctx._S_global,
            interface_solver='cg',
            matvec_mode='assembled',
            tile_index_maps=_tile_index_maps,
            preconditioner=_preconditioner,
            rtol=_cg_rtol,
            atol=_cg_atol,
            maxiter=_cg_maxiter,
            strict=_cg_strict,
            block_jacobi_max_bytes=_bj_max_bytes,
            verbose=verbose,
            cg_stats_dict=_cg_stats,
        )
        ctx._interface_lu = _solve_callable
        ctx._cg_solver = cg_solver
        ctx._interface_solver_mode = 'cg'

    elapsed = _time.perf_counter() - t0
    ctx.is_factored = True
    logger.info(
        "Refactored transient coordinator solve (%s) from saved S_global "
        "in %.3fs", _mode, elapsed,
    )

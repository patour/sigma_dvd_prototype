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

Adjoint note
------------
``solver_adjoint.py``'s adjoint solves call ``ctx.interface_lu(global_rhs)``
generically, with no mode check -- see the "Adjoint note" in
``interface_iterative.py``'s module docstring for the full explanation. This
works in direct, CG/assembled, AND CG/tilewise mode: the D1/D2 fixes make
tilewise CG's matvec exactly correct, so there is no unsupported case to
force to direct here.

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
from dataclasses import dataclass
import time as _time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import scipy.sparse as sp

# Finding 6: single canonical island-penalty-conductance constant (pgmath.schur
# owns it; safe at module level -- pgmath never imports distributed/, so no
# circular-import risk). Used as both a function default value (needs a
# module-level import, not the lazy per-function style used elsewhere in this
# file) and inline at every never-assemble/refactor stamping site.
from pgmath.schur import ISLAND_PENALTY_CONDUCTANCE

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


def _close_existing_cg_solver(ctx: Any) -> None:
    """Finding 6: close() the OUTGOING InterfaceCGSolver's persistent thread
    pool (and, in never-assemble mode, its retained per-tile Schur block
    dict) before replacing ``ctx._cg_solver``.

    The S12 lifecycle-parity fix added this close-before-replace step to
    ``release()`` and both ``refactor()`` sites, but not to the three
    ``factor()`` sites (``_factor_dc_context_no_s_global``,
    ``_factor_dc_context``, ``_factor_transient_context``). A session that
    calls ``ctx.factor()`` again on an already-factored context (no
    ``release()`` in between -- e.g. after changing ``model.settings`` CG
    tolerances) previously dropped the old ``InterfaceCGSolver`` into a
    reference cycle (solver -> LinearOperator -> bound matvec/apply
    closures -> solver), reclaimed only by an eventual cyclic-GC pass
    (the ``weakref.finalize`` safety net, not guaranteed timing) -- keeping
    the outgoing solver's thread pool and full ``tile_schur_complements``
    dict alive in the meantime, doubling coordinator memory per repeated
    ``factor()`` call.
    """
    _old_cg_solver = getattr(ctx, '_cg_solver', None)
    if _old_cg_solver is not None:
        _close = getattr(_old_cg_solver, 'close', None)
        if callable(_close):
            _close()


@dataclass(frozen=True)
class _InterfaceCgSettings:
    """Finding 13: bundle of the ~9-key interface-CG settings block, read
    from ``model.settings`` the same way at every call site.

    See :func:`_read_interface_cg_settings`.
    """

    preconditioner: str
    cg_rtol: float
    cg_atol: float
    cg_maxiter: Optional[int]
    cg_strict: bool
    bj_max_bytes: int
    # NN/BDD work package: 'two_level' base selection + Neumann knobs.
    two_level_base: Any
    neumann_max_bytes: int
    neumann_weight: Any
    neumann_reg: float
    # Docs Sec 7.13 recommended change 2: per-N-iteration true-residual
    # progress logging (0 = disabled).
    progress_every: int
    matvec_threads: Any
    matvec_dtype: Any
    strict_dtype_rtol: bool
    # Stage 3: two-level coarse-space preconditioner knobs.
    geneo_k: int
    geneo_tol: float
    eps_rank: float
    max_cols: int
    max_bytes: int
    # A-DEF2 work package: deflated apply mode + warm-start extrapolation.
    apply_mode: str
    deflated_reproject_every: int
    warm_start_extrapolation: bool


def _read_interface_cg_settings(model: Optional[Any]) -> '_InterfaceCgSettings':
    """Finding 13: single source of truth for the interface-CG settings
    block previously duplicated (with slight, drift-prone variations) at
    five call sites: ``_factor_dc_context_no_s_global``,
    ``_factor_dc_context``, ``_refactor_dc_context``,
    ``_factor_transient_context``, ``_refactor_transient_context``. Each
    site independently read/coerced ``interface_preconditioner``,
    ``interface_cg_rtol``, ``interface_cg_atol``, ``interface_cg_maxiter``,
    ``interface_cg_strict``, ``interface_block_jacobi_max_bytes``,
    ``matvec_threads``, ``interface_matvec_dtype``, and
    ``interface_strict_dtype_rtol`` -- adding a new setting (as this diff's
    own Stage 2 additions did) meant touching all five, and missing one
    would silently give that path a stale default.

    Deliberately does NOT include ``interface_matvec_mode`` or
    ``interface_factor_memory_budget`` -- those are read at different
    points (or, for ``interface_matvec_mode``, not at all) by different
    call sites: the never-assemble DC factor path forces tilewise
    unconditionally and never reads ``interface_matvec_mode``, so folding
    it in here would misrepresent which sites actually consume it.

    Stage 3: ``preconditioner``'s default changed from the literal
    ``'block_jacobi'`` to ``'auto'`` -- this is the "no explicit setting"
    signal that ``build_interface_solver``'s ``resolve_preconditioner()``
    (the ONE place 'auto' is resolved) turns into 'two_level' for CG+
    tilewise, or 'block_jacobi' otherwise (unchanged behaviour for
    'assembled'/'direct'). An explicit ``interface_preconditioner`` setting
    (YAML/CLI/``model.settings``) always overrides this and passes through
    unchanged. Note this is independent of ``cli.py``'s own
    ``_IFACE_SETTING_DEFAULTS['interface_preconditioner']`` (also 'auto' as
    of Stage 3) -- both defaults were changed together so CLI-driven and
    direct-``model.settings``-driven runs get the same resolved behaviour.

    Args:
        model: The ``DistributedPowerGridModel`` (or ``None`` -- every
            built-in default is returned in that case, matching every call
            site's own None-model fallback).

    Returns:
        An :class:`_InterfaceCgSettings` with every field resolved to its
        documented default when unset.
    """
    # Finding 15: lazy import (matching this function's existing
    # resolve_block_jacobi_max_bytes import) -- interface_coarse.py has no
    # internal-package dependencies of its own, so a module-level import
    # here would create no cycle, but result_factorization.py's convention
    # for interface_iterative/interface_coarse is a lazy, function-local
    # import (see the sibling import on the next line), kept consistent
    # here rather than special-cased.
    from . import interface_coarse
    from . import interface_iterative
    from .interface_iterative import (
        resolve_block_jacobi_max_bytes, resolve_coarse_max_bytes,
        resolve_neumann_max_bytes,
    )

    _settings = getattr(model, 'settings', None) if model is not None else None
    _settings = _settings or {}
    _cg_maxiter_raw = _settings.get('interface_cg_maxiter', None)
    _preconditioner_raw = _settings.get('interface_preconditioner', 'auto')
    return _InterfaceCgSettings(
        # Finding 1 (round 2): normalize (strip + lower) HERE, ONCE, at
        # settings-read time -- not just inside resolve_preconditioner
        # (which does its own normalization too, defense in depth, but
        # every OTHER literal comparison against this value -- the refactor
        # degrade guards and coarse-space-lost warning condition in this
        # module -- compared the RAW string against lowercase literals like
        # ``'block_jacobi'``/``'two_level'``/``'auto'``.  A sloppy-but-valid
        # YAML scalar (``'Two_Level'``, or ``' two_level '`` with a
        # trailing space from a quoted scalar -- resolve_preconditioner
        # itself already tolerates both) would build correctly at factor()
        # time (resolve_preconditioner normalizes) but silently bypass
        # these OTHER guards after a save()/load()+refactor() cycle, since
        # they never saw the canonical lowercase form.
        preconditioner=(
            _preconditioner_raw.strip().lower()
            if isinstance(_preconditioner_raw, str) else _preconditioner_raw
        ),
        cg_rtol=float(_settings.get('interface_cg_rtol', 1e-8)),
        cg_atol=float(_settings.get('interface_cg_atol', 1e-14)),
        cg_maxiter=(
            None if _cg_maxiter_raw is None
            else _coerce_int(_cg_maxiter_raw, 'interface_cg_maxiter')
        ),
        cg_strict=_coerce_bool(
            _settings.get('interface_cg_strict', True), 'interface_cg_strict',
        ),
        bj_max_bytes=resolve_block_jacobi_max_bytes(
            _settings.get('interface_block_jacobi_max_bytes', 'auto')
        ),
        # NN/BDD work package: raw two_level_base/neumann_weight values pass
        # through UNNORMALIZED (InterfaceCGSolver.__init__ is the single
        # validation/normalization point -- same round-3-finding-9 rationale
        # as apply_mode below); the byte budget resolves here like its
        # bj_max_bytes sibling.
        two_level_base=_settings.get('interface_two_level_base', 'auto'),
        neumann_max_bytes=resolve_neumann_max_bytes(
            _settings.get('interface_neumann_max_bytes', 'auto')
        ),
        neumann_weight=_settings.get('interface_neumann_weight', None),
        neumann_reg=_coerce_float(
            _settings.get(
                'interface_neumann_reg',
                interface_iterative.DEFAULT_NEUMANN_REG,
            ),
            'interface_neumann_reg',
        ),
        # Passed through raw; InterfaceCGSolver.__init__'s int() is the
        # single coercion/validation point (<= 0 disables).
        progress_every=_settings.get('interface_cg_progress_every', 0),
        matvec_threads=_settings.get('matvec_threads', 'auto'),
        matvec_dtype=_settings.get('interface_matvec_dtype', 'float64'),
        strict_dtype_rtol=_coerce_bool(
            _settings.get('interface_strict_dtype_rtol', True),
            'interface_strict_dtype_rtol',
        ),
        # Finding 15 (round 1) / Finding 9 (round 2): the canonical defaults
        # live on interface_coarse.py's DEFAULT_GENEO_K/DEFAULT_GENEO_TOL/
        # DEFAULT_EPS_RANK/DEFAULT_MAX_COLS -- read from there (a genuinely
        # dynamic attribute lookup: this function body runs fresh on every
        # call, unlike a def-time default) instead of re-hardcoding the
        # literals a second time.  interface_iterative.py's own
        # InterfaceCGSolver.__init__/build_interface_solver resolve from the
        # SAME source, dynamically, via None-sentinel defaults (Finding 9)
        # -- not bound at def time -- so a future retune of interface_
        # coarse.py's DEFAULT_* stays in sync across CLI-driven,
        # model.settings-driven, AND direct-constructor call sites.
        geneo_k=_coerce_int(
            _settings.get(
                'interface_coarse_geneo_k', interface_coarse.DEFAULT_GENEO_K,
            ),
            'interface_coarse_geneo_k',
        ),
        # Finding 2 (round 2): _coerce_float (not a bare float()) -- rejects
        # a YAML-1.1 bool (e.g. 'interface_coarse_geneo_tol: yes') instead
        # of silently coercing it to 1.0 (float(True) == 1.0), matching the
        # bool-rejection already applied to every sibling coercer in this
        # function (_coerce_int for geneo_k/max_cols, _coerce_bool for
        # cg_strict/strict_dtype_rtol).
        geneo_tol=_coerce_float(
            _settings.get(
                'interface_coarse_geneo_tol', interface_coarse.DEFAULT_GENEO_TOL,
            ),
            'interface_coarse_geneo_tol',
        ),
        eps_rank=_coerce_float(
            _settings.get(
                'interface_coarse_eps_rank', interface_coarse.DEFAULT_EPS_RANK,
            ),
            'interface_coarse_eps_rank',
        ),
        max_cols=_coerce_int(
            _settings.get(
                'interface_coarse_max_cols', interface_coarse.DEFAULT_MAX_COLS,
            ),
            'interface_coarse_max_cols',
        ),
        # Finding 5: byte-based guard (see resolve_coarse_max_bytes) on the
        # dense Z_dense/SZ arrays -- 'auto' = min(8 GB, 0.1x total RAM).
        max_bytes=resolve_coarse_max_bytes(
            _settings.get('interface_coarse_max_bytes', 'auto')
        ),
        # A-DEF2 work package: same "read the canonical default dynamically"
        # pattern as the geneo_k/geneo_tol/eps_rank/max_cols reads above --
        # interface_coarse.DEFAULT_APPLY_MODE/DEFAULT_DEFLATED_REPROJECT_EVERY
        # are the single canonical source, also consumed dynamically by
        # InterfaceCGSolver's own None-sentinel constructor defaults.
        #
        # Round-3 code review finding 9 (cleanup): pass the raw settings
        # value through UNNORMALIZED -- InterfaceCGSolver.__init__ already
        # does `_v.strip().lower() if isinstance(_v, str) else _v` on
        # `interface_coarse_apply_mode` (it is the single validation point
        # for this value, also raising ValueError for anything outside
        # {'additive', 'deflated'}), so normalizing here too was a
        # duplicate of that exact logic in a second place that could
        # silently diverge from it later. Every consumer of this field
        # (`apply_mode=_cg_settings.apply_mode` at all six
        # ``build_interface_solver`` call sites) threads it straight into
        # ``InterfaceCGSolver(interface_coarse_apply_mode=...)``, so the
        # constructor's own normalization always runs regardless.
        apply_mode=_settings.get(
            'interface_coarse_apply_mode', interface_coarse.DEFAULT_APPLY_MODE,
        ),
        deflated_reproject_every=_coerce_int(
            _settings.get(
                'interface_deflated_reproject_every',
                interface_coarse.DEFAULT_DEFLATED_REPROJECT_EVERY,
            ),
            'interface_deflated_reproject_every',
        ),
        # interface_warm_start_extrapolation's canonical default lives in
        # interface_iterative.py (DEFAULT_WARM_START_EXTRAPOLATION), not
        # interface_coarse.py -- see that constant's docstring for why.
        warm_start_extrapolation=_coerce_bool(
            _settings.get(
                'interface_warm_start_extrapolation',
                interface_iterative.DEFAULT_WARM_START_EXTRAPOLATION,
            ),
            'interface_warm_start_extrapolation',
        ),
    )


def _island_idx_array(
    island_nodes: Optional[Any], interface_node_to_idx: Dict[str, int],
) -> Optional[np.ndarray]:
    """Stage 3: convert an island-node-name set to a global interface-index
    array for the 'two_level' preconditioner's coarse-space row zeroing.

    Shared by every ``build_interface_solver`` call site so the same
    ``(island_nodes, interface_node_to_idx) -> island_idx`` conversion isn't
    duplicated five times with slight drift. Returns ``None`` for an empty/
    ``None`` island set (the common case) so callers can pass it straight
    through as ``island_idx=`` without an extra ``if``.
    """
    if not island_nodes:
        return None
    idx = np.array(
        sorted(
            interface_node_to_idx[n] for n in island_nodes
            if n in interface_node_to_idx
        ),
        dtype=np.int64,
    )
    return idx if idx.size else None


def _check_summaries_mixed_state(model: Any) -> None:
    """Finding R6 guard: raise on the forbidden 'summaries' mixed state.

    Factored out of :func:`_detect_islands_dispatch` (S6) so the "never
    assemble S_global" factor path (:func:`_factor_dc_context_no_s_global`,
    item 3) -- which cannot route through the full dispatch because it has
    no ``S_global`` to hand to the Schur-BFS branch / ``apply_island_
    penalty`` -- still gets the SAME guard instead of calling
    ``detect_interface_islands_from_summaries`` directly and silently
    islanding the whole interface on a corrupted model.

    See :func:`_detect_islands_dispatch`'s docstring for the full rationale.
    """
    _mode = getattr(model, 'island_detection_mode', 'schur_bfs')
    _summaries = getattr(model, 'component_summaries', None)
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

    # Finding R6 (factored into _check_summaries_mixed_state, S6): raise
    # loudly on the forbidden 'summaries' mixed state instead of silently
    # degrading to the legacy Schur-BFS path.
    _check_summaries_mixed_state(model)

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
        # Finding 8 (same root cause as interface_iterative.py's memory-
        # budget/matvec_threads coercers): a '.inf' value (PyYAML parses to
        # float('inf')) makes int(float(...)) raise OverflowError, not
        # TypeError/ValueError -- must be caught here too, e.g. for
        # 'interface_cg_maxiter: .inf'.
        return int(float(value))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"Invalid integer setting {setting_name}={value!r}: expected an "
            f"integer (or numeric string)."
        ) from exc


def _coerce_float(value: Any, setting_name: str) -> float:
    """Coerce a YAML/CLI-sourced numeric setting to float, rejecting garbage.

    Finding 2 (round 2): mirrors :func:`_coerce_int`'s bool-rejection idiom
    -- the Stage 3 coarse-space float knobs (``interface_coarse_geneo_tol``,
    ``interface_coarse_eps_rank``) were being read via a bare ``float(...)``
    call, which (unlike every sibling coercer in this module) silently
    accepts a YAML-1.1 bool (``interface_coarse_geneo_tol: yes`` parses to
    Python ``True``) and coerces it to ``1.0`` instead of raising --
    ``float(True) == 1.0``, the exact classic pitfall ``_coerce_int``'s own
    docstring calls out for the int case.

    Args:
        value: The raw settings value.
        setting_name: Name of the setting, used in the error message.

    Returns:
        The coerced float.

    Raises:
        ValueError: If ``value`` cannot be interpreted as a float.
    """
    if isinstance(value, bool):
        raise ValueError(
            f"Invalid float setting {setting_name}={value!r}: bool is not "
            f"a valid float value."
        )
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid float setting {setting_name}={value!r}: expected a "
            f"float (or numeric string)."
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


def _compute_interface_ordering(
    tile_port_node_lists: Dict[Any, List[str]],
    extra_edges: Optional[List[Tuple[str, str, float]]],
    dirichlet_nodes: Optional[Set[str]],
    dirichlet_voltage: float,
    ground_node: str = '0',
) -> Tuple[List[str], Dict[str, int], np.ndarray]:
    """Item 3: interface unknown ordering + Dirichlet RHS from PORT NAME
    LISTS alone -- no per-tile S_i values are touched, so this never needs
    to gather (let alone assemble) anything Schur-related.  O(number of
    distinct interface node names), used by the "never assemble S_global"
    factor path (``interface_drop_s_global``).

    Mirrors ``assemble_schur_complement_system``'s own Step 1 (unknown vs.
    Dirichlet node-universe split) exactly, so the resulting ordering is
    IDENTICAL to what the normal (S_global-assembling) path would produce
    for the same tiles/package edges/pads -- this function just stops short
    of ever building G_full/S_global.  ``_compute_rhs_dirichlet_from_edges``
    already provides the exact G-only Dirichlet RHS (tile Schur complements
    never contribute to the unknown-Dirichlet coupling G_ud -- see that
    function's docstring), so no S_i involvement is needed for the RHS
    either.

    Returns:
        ``(unknown_list, unknown_to_idx, rhs_dirichlet)``.
    """
    # Finding 14: shared unknown/Dirichlet node-universe split -- see
    # pgmath.schur.compute_interface_node_split's docstring for why this was
    # extracted out of independently-maintained copies here and in
    # assemble_schur_complement_system.
    from pgmath.schur import compute_interface_node_split

    _all_interface_nodes, dirichlet_set, unknown_nodes = compute_interface_node_split(
        tile_port_node_lists, extra_edges, dirichlet_nodes, ground_node,
    )
    unknown_list = sorted(unknown_nodes)
    unknown_to_idx = {n: i for i, n in enumerate(unknown_list)}

    rhs_dirichlet = _compute_rhs_dirichlet_from_edges(
        extra_edges=extra_edges,
        unknown_list=unknown_list,
        unknown_to_idx=unknown_to_idx,
        dirichlet_nodes=dirichlet_set,
        dirichlet_voltage=dirichlet_voltage,
    )
    return unknown_list, unknown_to_idx, rhs_dirichlet


def _build_s_extra_direct(
    interface_node_to_idx: Dict[str, int],
    n_iface: int,
    package_edges: Optional[List[Tuple[str, str, float]]],
    dirichlet_nodes: Optional[Set[str]],
    island_nodes: Optional[Set[str]] = None,
    package_cap_edges: Optional[List[Tuple[str, str, float]]] = None,
    C_coeff: float = 0.0,
    penalty_conductance: float = ISLAND_PENALTY_CONDUCTANCE,
    return_package_matrices: bool = False,
) -> Union[sp.csr_matrix, Tuple[sp.csr_matrix, Tuple[sp.csr_matrix, sp.csr_matrix]]]:
    """D2: build S_extra by DIRECT STAMPING (package edges + island-penalty
    diagonal), replacing the old ``S_global - sum_i P_i^T S_i P_i`` giant
    subtraction (~25 GB temporaries at 107-tile BRCM scale, plus FP
    cancellation residue left as spurious nnz -- interface_solve_
    acceleration_plan.md D2).

    Mode-dependent (the review-confirmed gap D2 also fixes): DC passes
    ``package_cap_edges=None``/``C_coeff=0`` so only the resistive
    ``package_edges`` contribute; transient passes ``package_cap_edges``
    plus the CURRENT ``ctx.C_coeff`` (1/dt_ps BE, 2/dt_ps TR) -- S_extra^TD
    therefore depends on dt/method and must be rebuilt inside every
    ``prepare_transient`` call (this function is O(package-edge count) and
    trivially cheap -- never shared with a DC-context instance).

    Mathematically identical to the old subtraction (verified by the
    equivalence tests below): :func:`pgmath.schur.build_interface_package_matrices`
    already stamps package conductances/capacitances into EXACTLY the
    unknown-unknown block using the same nodal-stamp convention
    ``assemble_schur_complement_system``'s ``extra_edges`` loop uses,
    restricted to ``interface_node_to_idx`` (unknown-only) -- an edge
    touching a Dirichlet node contributes only to ``rhs_dirichlet``
    (computed separately, unchanged by this function), not to S_extra.

    The island-penalty diagonal is stamped here too (not just onto
    S_global) so the tilewise operator (``sum_i S_i_kept + S_extra``)
    matches the assembled operator (``S_global``, which DOES carry the
    penalty via ``apply_island_penalty``) exactly -- ``sum_i S_i`` never
    carries the penalty on its own (workers don't know about interface
    islands, which are a coordinator-side, cross-tile concept).  This is
    the piece the tilewise-without-S_global path (item 3) needs, since
    there it is the ONLY place the penalty can be applied.

    Args:
        interface_node_to_idx: Unknown-only interface node ordering.
        n_iface: Number of interface unknowns.
        package_edges: Resistive package edges ``(u, v, g_mS)``.
        dirichlet_nodes: Pad node set.
        island_nodes: Interface nodes to penalize (diagonal-only), or None.
        package_cap_edges: Package capacitor edges ``(u, v, c_fF)`` (transient
            mode only; None/empty for DC).
        C_coeff: Capacitance coefficient (1/dt_ps BE, 2/dt_ps TR); 0 for DC
            (skips the capacitive term entirely, even if package_cap_edges
            is non-empty, so callers may pass the model's cap edges
            unconditionally and control mode purely via C_coeff).
        penalty_conductance: Island-penalty diagonal magnitude (mS);
            default ``ISLAND_PENALTY_CONDUCTANCE``, matching
            ``apply_island_penalty``.
        return_package_matrices: Finding 12: when True, also return the
            ``(G_pkg_uu, C_pkg_uu)`` pair this function stamps internally
            (via ``build_interface_package_matrices``), so a caller that
            ALSO needs them afterward (e.g. for
            ``ctx.C_package_uu``/``ctx._G_package_uu``, the transient
            RHS history terms) can reuse this single stamping pass
            instead of calling ``build_interface_package_matrices`` a
            second time with identical arguments.

    Returns:
        Sparse CSR matrix S_extra (entries below scipy's default zero
        elimination threshold removed via ``eliminate_zeros()``), or, if
        ``return_package_matrices``, the tuple
        ``(S_extra, (G_pkg_uu, C_pkg_uu))``.
    """
    from pgmath.schur import build_interface_package_matrices

    G_pkg_uu, C_pkg_uu = build_interface_package_matrices(
        package_edges=package_edges or [],
        package_cap_edges=package_cap_edges or [],
        interface_node_to_idx=interface_node_to_idx,
        n_interface=n_iface,
        dirichlet_nodes=dirichlet_nodes or set(),
    )
    S_extra = G_pkg_uu.tocsr()
    if C_coeff and package_cap_edges:
        S_extra = (S_extra + C_coeff * C_pkg_uu).tocsr()

    if island_nodes:
        indices = np.array(
            [interface_node_to_idx[n] for n in island_nodes
             if n in interface_node_to_idx],
            dtype=np.intp,
        )
        if len(indices):
            penalty_vals = np.full(len(indices), penalty_conductance, dtype=np.float64)
            penalty_matrix = sp.coo_matrix(
                (penalty_vals, (indices, indices)), shape=(n_iface, n_iface),
            ).tocsr()
            S_extra = (S_extra + penalty_matrix).tocsr()

    S_extra = S_extra.tocsr()
    S_extra.eliminate_zeros()
    if return_package_matrices:
        return S_extra, (G_pkg_uu, C_pkg_uu)
    return S_extra


def _gather_kept_tile_schur_streaming(
    model: Any,
    interface_node_to_idx: Dict[str, int],
    *,
    transient_dt_scaled: Optional[float] = None,
    transient_method: Optional[str] = None,
    dirichlet_voltage: Optional[float] = None,
    n_iface: Optional[int] = None,
    out_port_count: Optional[Dict[Any, int]] = None,
    out_tile_cap: Optional[Dict[Any, float]] = None,
) -> Tuple[
    Dict[Any, np.ndarray], Dict[Any, np.ndarray], Dict[Any, np.ndarray],
    Optional[np.ndarray],
]:
    """Re-gather D1-safe per-tile dense Schur blocks from FACTORED workers.

    Streams one tile's full (unsliced) ``S_i`` in flight at a time via
    ``call_all_streaming`` (the same protocol ``_stream_assemble_schur`` and
    ``scripts/benchmark/microbench/bench_interface_matvec.py``'s
    ``gather_tiles`` use) and immediately kept-position-slices it (D1 fix)
    before fetching the next tile -- coordinator peak memory is bounded by
    the sum of the KEPT blocks plus one in-flight full block, never all
    unsliced blocks simultaneously.

    Used by:
      - ``refactor()`` (item 9): rebuilds tilewise CG instead of silently
        downgrading to assembled when workers are already factored (from a
        prior ``factor()`` call in this session) but the coordinator's own
        ``tile_schur_complements`` was never saved (S_i is never persisted,
        by design -- ``save()``/``load()`` only carry ``S_global``).
      - The "never assemble S_global" factor path (item 3,
        ``interface_drop_s_global``): the SAME streaming gather, just called
        at initial ``factor()`` time instead of ``refactor()`` time.

    Args:
        model: Live ``DistributedPowerGridModel`` with attached (already
            interior-factored) workers.
        interface_node_to_idx: Unknown-only interface node ordering.
        transient_dt_scaled: When provided (with ``transient_method``),
            re-factors the TRANSIENT system (``A = G + C_coeff*C``) via
            ``factor_transient_system(dt_scaled, method)`` instead of the DC
            system via ``factor_and_compute_schur()``.  Must be provided
            together with ``transient_method``.
        transient_method: 'be' or 'trap'; see ``transient_dt_scaled``.
        dirichlet_voltage: When provided (with ``n_iface``), ALSO accumulate
            and return the Dirichlet RHS contribution from each tile's OWN
            pad-adjacent ``S_i`` entries -- i.e. the ``-S_i[kept, dirichlet]
            @ V_d`` term that a tile with a Dirichlet pad directly on its own
            port list contributes to ``G_ud`` (and hence ``rhs_dirichlet``)
            in the normal ``assemble_schur_complement_system`` path.  The
            "never assemble S_global" factor path (item 3) needs this: its
            ``_compute_interface_ordering`` pre-pass only sees package
            ``extra_edges`` (no S_i values yet), so it is blind to a
            tile-resident pad's contribution -- this streaming pass is the
            only place that contribution is ever observable without
            assembling G_full.  Omit (default None) when the caller doesn't
            need it (e.g. ``refactor()``, which reuses the already-fixed
            ``rhs_dirichlet`` from the original ``factor()`` call).
        n_iface: Number of interface unknowns; required together with
            ``dirichlet_voltage``.
        out_port_count: Optional mutable dict; when provided, populated
            in-place with ``{tid: len(port_nodes)}`` (S2/S13) -- the tile's
            FULL port count at the moment ``kept_pos`` was built, for
            ``ctx.tile_port_count`` / ``filter_kept_rhs`` validation.  A
            side-channel output (not part of the return tuple) so existing
            callers that only want the first four values are unaffected.
        out_tile_cap: Optional mutable dict; when provided AND
            ``transient_dt_scaled`` is set, populated in-place with
            ``{tid: total_cap_fF}`` -- the per-tile total capacitance
            ``factor_transient_system`` already computes and returns as its
            3rd tuple element (silently discarded otherwise).  Used by the
            TD "never assemble S_global" factor path
            (:func:`_factor_transient_context_no_s_global`) to derive
            ``ctx.has_capacitance`` without a separate round-trip.  Ignored
            in DC mode (``transient_dt_scaled is None``) -- there is no
            per-tile cap total to report.

    Returns:
        ``(tile_schur_complements, tile_index_maps, tile_kept_port_pos,
        rhs_dirichlet_from_tiles)`` -- the first three keyed by tile_id,
        D1-consistent by construction (built together via
        ``kept_position_slice``); the fourth is ``None`` unless
        ``dirichlet_voltage``/``n_iface`` were provided.
    """
    from .interface_iterative import kept_position_slice

    tile_schur_complements: Dict[Any, np.ndarray] = {}
    tile_index_maps: Dict[Any, np.ndarray] = {}
    tile_kept_port_pos: Dict[Any, np.ndarray] = {}

    is_transient = transient_dt_scaled is not None
    if is_transient != (transient_method is not None):
        raise ValueError(
            "_gather_kept_tile_schur_streaming: transient_dt_scaled and "
            "transient_method must be provided together."
        )
    _want_tile_rhs = dirichlet_voltage is not None
    if _want_tile_rhs != (n_iface is not None):
        raise ValueError(
            "_gather_kept_tile_schur_streaming: dirichlet_voltage and "
            "n_iface must be provided together."
        )
    rhs_dirichlet_from_tiles = (
        np.zeros(n_iface, dtype=np.float64) if _want_tile_rhs else None
    )

    if is_transient:
        n_tiles = len(model.metadata.tile_configs)
        stream = model.backend.call_all_streaming(
            model.workers, 'factor_transient_system',
            [(transient_dt_scaled, transient_method)] * n_tiles,
        )
    else:
        stream = model.backend.call_all_streaming(
            model.workers, 'factor_and_compute_schur',
        )

    for i, result in stream:
        tid = model.metadata.tile_configs[i].tile_id
        if is_transient:
            S_i, port_nodes, _total_cap, _stats = result
            if out_tile_cap is not None:
                out_tile_cap[tid] = _total_cap
        else:
            S_i, port_nodes, _stats = result
        idx, S_kept, kept_pos = kept_position_slice(
            S_i, port_nodes, interface_node_to_idx,
        )
        # T3 fix: kept_position_slice() treats ANY port node absent from
        # interface_node_to_idx as "Dirichlet, drop its row/column" -- that
        # equivalence only holds when interface_node_to_idx is the CURRENT
        # ordering. When this function is called with a STALE ordering
        # (refactor() re-gathering LIVE worker port lists against a
        # checkpoint's saved interface_node_to_idx after the model was
        # re-parsed with a different retile/topology), a genuinely-live
        # interface port that simply isn't in the stale ordering gets
        # silently reclassified as a pad and dropped -- corrupting the
        # tilewise operator with no error. Any dropped port that is NOT a
        # real Dirichlet pad (model.pad_nodes) is exactly that mismatch;
        # fail loudly instead of silently scattering wrong voltages.
        if len(kept_pos) < len(port_nodes):
            _kept_pos_set = {int(p) for p in kept_pos}
            _dropped_nodes = [
                nd for p, nd in enumerate(port_nodes) if p not in _kept_pos_set
            ]
            _bad_dropped = [nd for nd in _dropped_nodes if nd not in model.pad_nodes]
            if _bad_dropped:
                raise ValueError(
                    f"_gather_kept_tile_schur_streaming: tile {tid!r} has "
                    f"port node(s) {_bad_dropped!r} missing from the "
                    f"supplied interface_node_to_idx ordering, but "
                    f"{'they are' if len(_bad_dropped) > 1 else 'it is'} "
                    f"NOT a Dirichlet pad (model.pad_nodes). This indicates "
                    f"a model/checkpoint topology mismatch -- e.g. "
                    f"refactor() re-gathering the LIVE workers' port lists "
                    f"against a LOADED checkpoint's STALE interface "
                    f"ordering after the model was re-parsed with a "
                    f"different retile. Silently dropping this port would "
                    f"corrupt the tilewise interface operator. Call "
                    f"factor() (not refactor()) to rebuild from the "
                    f"model's current topology."
                )
        tile_schur_complements[tid] = S_kept
        tile_index_maps[tid] = idx
        tile_kept_port_pos[tid] = kept_pos
        if out_port_count is not None:
            out_port_count[tid] = len(port_nodes)

        if _want_tile_rhs and len(port_nodes) > len(kept_pos):
            # This tile has >=1 Dirichlet/pad port directly on its own port
            # list (D1 scenario) -- every position NOT in kept_pos is
            # Dirichlet (see kept_position_slice: "not kept" <=> "is
            # Dirichlet" for tile ports, since a tile's port list is a
            # subset of all_interface_nodes).  Contribute
            # -S_i[kept, dirichlet] @ V_d to the unknown positions, exactly
            # matching assemble_schur_complement_system's G_ud slice for
            # this tile's own embedded S_i.
            n_ports = len(port_nodes)
            all_pos = np.arange(n_ports, dtype=np.int64)
            dirichlet_pos = np.setdiff1d(all_pos, kept_pos, assume_unique=False)
            S_arr = np.asarray(S_i, dtype=np.float64)
            contrib = -dirichlet_voltage * S_arr[np.ix_(kept_pos, dirichlet_pos)].sum(axis=1)
            rhs_dirichlet_from_tiles[idx] += contrib

    return (
        tile_schur_complements, tile_index_maps, tile_kept_port_pos,
        rhs_dirichlet_from_tiles,
    )


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
    Dict[Any, np.ndarray],  # tile_kept_port_pos (Finding 4)
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
         tile_index_maps, tile_kept_port_pos)
    where tile_index_maps are the per-tile local->global index arrays built
    during Step 2 (stored for downstream use: S_extra, CG solver setup, etc.),
    and tile_kept_port_pos (Finding 4) are the per-tile POSITIONS within the
    tile's full port list (``tile_port_node_lists[tid]``) that are kept
    (non-Dirichlet) -- the streaming-path mirror of
    ``kept_position_slice()``'s ``kept_pos`` return value, needed by
    ``solve_dc``'s D1 reduced-RHS scatter (solver.py) whenever a tile has a
    pad directly on its own port list and streaming assembly was used.

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
        return empty_S, empty_rhs, [], {}, {}, {}

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
    # Finding 4: also compute tile_kept_port_pos -- the streaming-path mirror
    # of kept_position_slice()'s kept_pos return, i.e. the POSITIONS within
    # each tile's full port list (node_list, may include a tile-resident
    # pad/Dirichlet port) that are kept (non-Dirichlet).  Populating this
    # here (instead of leaving it empty) lets solve_dc's D1 reduced-RHS
    # scatter (solver.py's ctx.tile_kept_port_pos lookup) recover a
    # tile-resident pad-on-port scenario on the streaming DC path exactly
    # as it already does on the bulk (non-streaming) path.
    interface_tile_index_maps: Dict[Any, np.ndarray] = {}
    interface_tile_kept_port_pos: Dict[Any, np.ndarray] = {}
    for tid, node_list in tile_port_node_lists.items():
        kept_pos_list = [p for p, nd in enumerate(node_list) if nd in unknown_to_idx]
        interface_tile_index_maps[tid] = np.array(
            [unknown_to_idx[node_list[p]] for p in kept_pos_list], dtype=np.int32,
        )
        interface_tile_kept_port_pos[tid] = np.array(kept_pos_list, dtype=np.int64)

    # Update assembly cache (same as assemble_schur_complement_system)
    if assembly_cache is not None and _cache_valid:
        assembly_cache['tile_local_to_global'] = _new_l2g
        assembly_cache['full_node_to_idx'] = full_node_to_idx
        assembly_cache['unknown_list'] = unknown_list
        assembly_cache['dirichlet_list'] = dirichlet_list
        assembly_cache['n_unknown'] = n_unknown
        assembly_cache['n_full'] = n_full

    return (
        S_global, rhs_dirichlet, unknown_list, unknown_to_idx,
        interface_tile_index_maps, interface_tile_kept_port_pos,
    )


def _get_drop_s_global_setting(model: Optional[Any]) -> bool:
    """Read the ``interface_drop_s_global`` setting (default False)."""
    if model is None:
        return False
    settings = getattr(model, 'settings', None)
    if settings is None:
        return False
    return _coerce_bool(
        settings.get('interface_drop_s_global', False), 'interface_drop_s_global',
    )


def _can_use_no_s_global_path(model: Any) -> Tuple[bool, str]:
    """Check the preconditions for the "never assemble S_global" factor path
    (item 3): ``interface_solver`` must resolve unambiguously to 'cg' (an
    explicit 'cg', not 'auto' -- 'auto' would need S_global for its own
    memory-estimate decision, a chicken-and-egg problem this path
    deliberately sidesteps rather than half-solves), ``interface_matvec_mode``
    must be 'tilewise' or 'auto', and island detection must be the Stage 1e
    summaries union-find (the legacy Schur-BFS fundamentally needs
    S_global's nonzero structure -- see interface_iterative.py module
    docstring).

    Returns ``(ok, reason)`` -- ``reason`` is a human-readable explanation
    used in the fallback WARNING when ``ok`` is False.
    """
    settings = getattr(model, 'settings', {}) or {}
    iface_solver = settings.get('interface_solver', 'auto')
    matvec_mode = settings.get('interface_matvec_mode', 'auto')
    island_mode = getattr(model, 'island_detection_mode', 'schur_bfs')

    if iface_solver != 'cg':
        return False, (
            f"interface_solver={iface_solver!r} (must be explicit 'cg' -- "
            f"'auto' cannot resolve without S_global)"
        )
    # S10: None means "use the auto default" everywhere else in this module
    # (resolve_matvec_mode, the refactor _want_tilewise check both treat
    # None the same as 'auto') -- a programmatically-built settings dict
    # with interface_matvec_mode=None must not be silently rejected here,
    # or a caller who opted into interface_drop_s_global to avoid the
    # >190 GB S_global assembly falls back into exactly that assembly.
    if matvec_mode is None:
        matvec_mode = 'auto'
    if matvec_mode not in ('tilewise', 'auto'):
        return False, f"interface_matvec_mode={matvec_mode!r} (must be 'tilewise' or 'auto')"
    if island_mode != 'summaries':
        return False, (
            f"model.island_detection_mode={island_mode!r} (must be "
            f"'summaries' -- the legacy Schur-BFS needs S_global)"
        )
    return True, ''


def _factor_dc_context_no_s_global(
    ctx: 'DistributedSolverContext', model: Any, verbose: bool = False,
) -> None:
    """Item 3 (Finding 0 upgrade): factor a CG+tilewise DC context WITHOUT
    ever assembling S_global.

    Two lightweight round-trips to the (already interior-factorable)
    workers, neither of which ever holds all per-tile Schur blocks at once:

      1. ``get_port_node_list()`` (no factoring) -- just node NAMES, used to
         compute the interface unknown ordering + G-only Dirichlet RHS via
         :func:`_compute_interface_ordering` (no S_i involvement at all).
      2. ``factor_and_compute_schur()`` streamed one tile at a time via
         :func:`_gather_kept_tile_schur_streaming` -- D1-safe kept-position
         slicing happens immediately per tile, so peak coordinator memory is
         the sum of the KEPT blocks (Stage 0's ~26.5 GB streaming-gather
         number at the 64-tile/167K regime) plus one in-flight full block,
         never the >190 GB non-streaming COO-assembly peak that motivated
         this path (Stage 0 Finding 0).

    Island detection is summaries-only (:func:`_can_use_no_s_global_path`
    is the caller's precondition check) -- the penalty's RHS contribution is
    applied directly here (mirroring ``apply_island_penalty``'s RHS
    formula); the penalty's DIAGONAL contribution is folded into S_extra via
    :func:`_build_s_extra_direct`'s ``island_nodes`` parameter (item 2/D2).
    """
    from .result import DistributedTopologyContext
    from .interface_iterative import build_interface_solver
    from pgmath.schur import detect_interface_islands_from_summaries

    timings: Dict[str, Any] = {}

    # Round-trip 1: port name lists only.
    t0 = _time.perf_counter()
    port_lists_raw = model.backend.call_all(model.workers, 'get_port_node_list')
    tile_port_node_lists: Dict[Any, List[str]] = {
        model.metadata.tile_configs[i].tile_id: pl
        for i, pl in enumerate(port_lists_raw)
    }
    timings['gather_port_lists'] = _time.perf_counter() - t0

    t0 = _time.perf_counter()
    interface_nodes, interface_node_to_idx, rhs_dirichlet = _compute_interface_ordering(
        tile_port_node_lists=tile_port_node_lists,
        extra_edges=model.package_data.package_edges,
        dirichlet_nodes=model.pad_nodes,
        dirichlet_voltage=model.vdd,
    )
    timings['compute_ordering'] = _time.perf_counter() - t0

    # Island detection: summaries union-find only (never touches S_global).
    # S6: route through the SAME Finding-R6 mixed-state guard
    # _detect_islands_dispatch uses, instead of calling
    # detect_interface_islands_from_summaries directly -- this path can't
    # use the full dispatch (it has no S_global for the Schur-BFS branch /
    # apply_island_penalty), but it must not skip the guard: with
    # island_detection_mode == 'summaries' and component_summaries == None
    # (the corrupted state the guard exists to catch),
    # detect_interface_islands_from_summaries would iterate '(None or ())'
    # and island nearly the whole interface silently.
    #
    # Finding 10 (round 2 review): mirror the assembled DC path's topology
    # island cache (a few hundred lines below, in _factor_dc_context) --
    # a second DC prepare() on this model reuses ctx.topology (cached on
    # solver.py's self._topology and passed back in), so a cache hit here
    # skips the summaries union-find entirely, exactly like the assembled
    # path already does. Without this, every DC never-assemble prepare()
    # pays a full summaries union-find over the whole interface even when
    # island_nodes was already computed and cached by a prior prepare() on
    # this same solver -- growing with the 100M-node target scale this
    # path exists for. (DC-mode cache only: resistive-only extra_edges,
    # same as the assembled path's cache key -- the TD never-assemble path
    # has its own independent, mode-correct island_nodes_td cache.)
    t0 = _time.perf_counter()
    _cached_islands_dc: Optional[Set[str]] = (
        getattr(ctx.topology, 'island_nodes', None)
        if ctx.topology is not None else None
    )
    if _cached_islands_dc is not None:
        island_nodes = _cached_islands_dc
        logger.debug(
            "DC island detection (never-assemble path): cache hit (%d "
            "islands), union-find skipped.", len(island_nodes),
        )
    else:
        _check_summaries_mixed_state(model)
        island_nodes = detect_interface_islands_from_summaries(
            component_summaries=model.component_summaries,
            interface_node_to_idx=interface_node_to_idx,
            pad_nodes=model.pad_nodes,
            extra_edges=model.package_data.package_edges,
        )
    if island_nodes:
        penalty_conductance = ISLAND_PENALTY_CONDUCTANCE
        idx = np.array(
            [interface_node_to_idx[n] for n in island_nodes], dtype=np.intp,
        )
        rhs_dirichlet = rhs_dirichlet.copy()
        rhs_dirichlet[idx] += penalty_conductance * model.vdd
        logger.warning(
            "Penalized %d interface island nodes (never-assemble path, "
            "shorted to %.3f V)", len(island_nodes), model.vdd,
        )
    timings['detect_interface_islands'] = _time.perf_counter() - t0

    # Round-trip 2: streaming, D1-safe per-tile Schur gather.  Also
    # accumulates each tile's OWN Dirichlet-adjacent RHS contribution (a
    # tile-resident pad directly on a tile's port list, D1 scenario) --
    # invisible to the package-edges-only rhs_dirichlet computed above.
    t0 = _time.perf_counter()
    tile_port_count: Dict[Any, int] = {}
    (
        tile_schur_complements, tile_index_maps, tile_kept_port_pos,
        rhs_dirichlet_from_tiles,
    ) = _gather_kept_tile_schur_streaming(
        model, interface_node_to_idx,
        dirichlet_voltage=model.vdd, n_iface=len(interface_nodes),
        out_port_count=tile_port_count,
    )
    rhs_dirichlet = rhs_dirichlet + rhs_dirichlet_from_tiles
    timings['factor_tiles'] = _time.perf_counter() - t0

    # D2: S_extra by direct stamping (package edges + island-penalty diagonal).
    # Finding 8 (round 2 review): request return_package_matrices=True (the
    # Finding-12 pattern already used by the TD paths) instead of a second,
    # separate build_interface_package_matrices() call below with
    # identical arguments -- G_pkg_uu doesn't depend on C_coeff/
    # package_cap_edges (build_interface_package_matrices computes G and C
    # independently), so passing package_cap_edges here (C_coeff stays 0.0,
    # the DC default, so the capacitive term is never added to S_extra) is
    # safe and yields the SAME G_pkg_uu the old second call built, from one
    # stamping pass instead of two -- and removes the risk of the two call
    # sites' arguments silently drifting apart.
    t0 = _time.perf_counter()
    S_extra, (G_pkg_uu, _C_pkg_uu_unused) = _build_s_extra_direct(
        interface_node_to_idx=interface_node_to_idx,
        n_iface=len(interface_nodes),
        package_edges=model.package_data.package_edges,
        dirichlet_nodes=model.pad_nodes,
        island_nodes=island_nodes,
        package_cap_edges=model.package_data.package_cap_edges,
        return_package_matrices=True,
    )
    timings['build_s_extra'] = _time.perf_counter() - t0

    # Finding 13: shared settings-reading helper (see its docstring).
    _cg_settings = _read_interface_cg_settings(model)
    _preconditioner = _cg_settings.preconditioner
    _cg_rtol = _cg_settings.cg_rtol
    _cg_atol = _cg_settings.cg_atol
    _cg_maxiter = _cg_settings.cg_maxiter
    _cg_strict = _cg_settings.cg_strict
    _bj_max_bytes = _cg_settings.bj_max_bytes
    _matvec_threads = _cg_settings.matvec_threads
    _matvec_dtype = _cg_settings.matvec_dtype
    _strict_dtype_rtol = _cg_settings.strict_dtype_rtol
    # Stage 3: two-level coarse-space knobs + island indices (for the
    # coarse-space row-zeroing -- see interface_coarse.py's module
    # docstring).  island_nodes was computed above (island detection step).
    _island_idx = _island_idx_array(island_nodes, interface_node_to_idx)

    t0 = _time.perf_counter()
    _cg_stats: Dict[str, Any] = {}
    _cg_solve_callable, _resolved_mode, _cg_solver = build_interface_solver(
        S_global=None,
        interface_solver='cg',
        tile_schur_complements=tile_schur_complements,
        tile_index_maps=tile_index_maps,
        S_extra=S_extra,
        matvec_mode='tilewise',
        preconditioner=_preconditioner,
        rtol=_cg_rtol,
        atol=_cg_atol,
        maxiter=_cg_maxiter,
        strict=_cg_strict,
        block_jacobi_max_bytes=_bj_max_bytes,
        verbose=verbose,
        cg_stats_dict=_cg_stats,
        n_interface=len(interface_nodes),
        matvec_threads=_matvec_threads,
        matvec_dtype=_matvec_dtype,
        strict_dtype_rtol=_strict_dtype_rtol,
        island_idx=_island_idx,
        interface_coarse_geneo_k=_cg_settings.geneo_k,
        interface_coarse_geneo_tol=_cg_settings.geneo_tol,
        interface_coarse_eps_rank=_cg_settings.eps_rank,
        interface_coarse_max_cols=_cg_settings.max_cols,
        interface_coarse_max_bytes=_cg_settings.max_bytes,
        two_level_base=_cg_settings.two_level_base,
        neumann_max_bytes=_cg_settings.neumann_max_bytes,
        neumann_weight=_cg_settings.neumann_weight,
        neumann_reg=_cg_settings.neumann_reg,
        progress_every=_cg_settings.progress_every,
        interface_coarse_apply_mode=_cg_settings.apply_mode,
        interface_deflated_reproject_every=_cg_settings.deflated_reproject_every,
        warm_start_extrapolation=_cg_settings.warm_start_extrapolation,
    )
    timings['factor_interface'] = _time.perf_counter() - t0
    timings['total_prepare'] = sum(
        v for v in timings.values() if isinstance(v, (int, float))
    )

    if verbose:
        logger.info(
            "=== Distributed DDM Prepare Statistics (never-assemble S_global) ===",
        )
        logger.info(
            "Interface system: %d unknowns, %d islands penalized, "
            "matvec_threads=%d, matvec_dtype=%s",
            len(interface_nodes), len(island_nodes),
            _cg_solver.matvec_threads, _cg_solver.matvec_dtype,
        )
        # Per-phase breakdown — the worker-side interior factor + Schur
        # computation happens inside the streaming gather and is otherwise
        # invisible in this path's log (the assembled path prints its own
        # factor_tiles/assembly lines).
        for _phase in ('gather_port_lists', 'compute_ordering',
                       'detect_interface_islands', 'factor_tiles',
                       'build_s_extra', 'build_cg_solver'):
            if _phase in timings:
                logger.info("  %-36s %.3fs", _phase, timings[_phase])
        logger.info("=== Total Prepare: %.3fs ===", timings['total_prepare'])

    ctx._interface_lu = _cg_solve_callable
    ctx._interface_nodes = interface_nodes
    ctx._interface_node_to_idx = interface_node_to_idx
    ctx._rhs_dirichlet_interface = rhs_dirichlet
    ctx._tile_index_maps = tile_index_maps
    ctx._tile_kept_port_pos = tile_kept_port_pos
    ctx._tile_port_count = tile_port_count
    ctx._removed_interface_nodes = island_nodes
    ctx._S_global = None
    # Item 3: marks that S_global was never assembled this factor() call --
    # save() checks this and raises with guidance (S_global is None already,
    # but this flag distinguishes "never built" from "released after build",
    # which matters for the error message's wording).
    ctx._s_global_dropped = True
    ctx.timings = timings
    ctx._interface_solver_mode = 'cg'
    # Finding 6: close the outgoing CG solver (if factor() is being re-run
    # on an already-factored context) before replacing it.
    _close_existing_cg_solver(ctx)
    ctx._cg_solver = _cg_solver

    if ctx.topology is None:
        ctx.topology = DistributedTopologyContext(
            interface_nodes=interface_nodes,
            interface_node_to_idx=interface_node_to_idx,
            tile_index_maps=tile_index_maps,
            rhs_dirichlet_G=rhs_dirichlet,
            G_package_uu=G_pkg_uu if G_pkg_uu.nnz > 0 else None,
            removed_interface_nodes=island_nodes,
            island_nodes=island_nodes,
            tile_kept_port_pos=tile_kept_port_pos,
            tile_port_count=tile_port_count,
        )
    elif getattr(ctx.topology, 'island_nodes', None) is None:
        ctx.topology.island_nodes = island_nodes

    ctx.is_factored = True


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
    model = ctx.model

    # Item 3 (Finding 0 upgrade): interface_drop_s_global='never assemble
    # S_global at all', not 'assemble then free' -- dispatch to the
    # dedicated no-S_global factor path when its preconditions hold
    # (explicit interface_solver='cg', matvec_mode tilewise/auto, summaries-
    # based island detection).  Falls back to the normal path (with a
    # WARNING) otherwise -- the setting is opt-in and must degrade
    # gracefully rather than raise, since 'auto'-everything is still the
    # documented default-safe configuration.
    if _get_drop_s_global_setting(model):
        _ok, _reason = _can_use_no_s_global_path(model)
        if _ok:
            _factor_dc_context_no_s_global(ctx, model, verbose)
            return
        logger.warning(
            "interface_drop_s_global=True but preconditions are not met "
            "(%s); falling back to the normal (S_global-assembling) factor "
            "path.", _reason,
        )

    timings: Dict[str, Any] = {}

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
            _streaming_tile_index_maps, _streaming_tile_kept_port_pos,
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
        _streaming_tile_kept_port_pos = None  # not used in bulk path

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

    # 3. Build tile index maps (local port indices -> global interface indices)
    # AND (D1 fix, item 1) kept-position-slice tile_schur_complements so every
    # tile's S_i dimension always agrees with its (pad-filtered) index map.
    # Must be built BEFORE step 4 (interface solver setup) so that CG tilewise
    # mode can reference tile_index_maps when constructing InterfaceCGSolver.
    # B3: When streaming was used, tile_index_maps were already built during
    # _stream_assemble_schur (stored in _streaming_tile_index_maps), along
    # with tile_kept_port_pos (Finding 4: _streaming_tile_kept_port_pos --
    # populated the same way, mirroring this bulk-path loop, so solve_dc's
    # D1 reduced-RHS scatter can recover a tile-resident pad-on-port
    # scenario on the streaming DC path too).  tile_schur_complements is
    # empty (S_i was never gathered) -- nothing to slice there.  For the
    # bulk path, build both together via kept_position_slice so the map and
    # the (now-sliced) S_i can never disagree.
    from .interface_iterative import kept_position_slice
    tile_kept_port_pos: Dict[Any, np.ndarray] = {}
    if _streaming_tile_index_maps is not None:
        tile_index_maps: Dict[Tuple[int, int], np.ndarray] = _streaming_tile_index_maps
        tile_kept_port_pos = _streaming_tile_kept_port_pos or {}
    else:
        tile_index_maps = {}
        for tid, boundary_list in tile_port_node_lists.items():
            idx, S_kept, kept_pos = kept_position_slice(
                tile_schur_complements[tid], boundary_list, interface_node_to_idx,
            )
            tile_index_maps[tid] = idx
            tile_schur_complements[tid] = S_kept
            tile_kept_port_pos[tid] = kept_pos

    # S2/S13: per-tile FULL port count (tile_port_node_lists already holds
    # the full, unfiltered port list regardless of streaming vs bulk path)
    # -- recorded alongside tile_kept_port_pos for filter_kept_rhs's
    # drift-detection validation.
    tile_port_count: Dict[Any, int] = {
        tid: len(node_list) for tid, node_list in tile_port_node_lists.items()
    }

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
        from .interface_iterative import build_interface_solver, resolve_matvec_mode
        # Item 8: default changed 'assembled' -> 'auto' (tilewise whenever
        # per-tile Schur blocks are available -- resolved below, once
        # tile_schur_complements' post-streaming-fallback availability is known).
        _matvec_mode_setting = _model_settings.get('interface_matvec_mode', 'auto')
        # Finding 13: shared settings-reading helper (see its docstring).
        _cg_settings = _read_interface_cg_settings(model)
        _preconditioner = _cg_settings.preconditioner
        _cg_rtol = _cg_settings.cg_rtol
        _cg_atol = _cg_settings.cg_atol
        _cg_maxiter = _cg_settings.cg_maxiter
        _cg_strict = _cg_settings.cg_strict
        _bj_max_bytes = _cg_settings.bj_max_bytes
        _matvec_threads = _cg_settings.matvec_threads
        _matvec_dtype = _cg_settings.matvec_dtype
        _strict_dtype_rtol = _cg_settings.strict_dtype_rtol

        # 'auto' resolves to 'tilewise' whenever per-tile Schur blocks are
        # available (post-streaming-fallback -- streaming never gathers S_i).
        _has_tile_blocks = (not _use_streaming_dc) and bool(tile_schur_complements)
        _matvec_mode = resolve_matvec_mode(_matvec_mode_setting, _has_tile_blocks)

        # D2 (item 2): direct-stamping S_extra (package edges + island-penalty
        # diagonal) -- replaces the old S_global-minus-sum_i-S_i subtraction.
        # B3: When streaming was used, tile_schur_complements is empty;
        # tilewise CG mode is not available (no per-tile S_i to matvec
        # against).  Fall back to 'assembled' mode and log a warning.
        _S_extra: Optional[sp.spmatrix] = None
        if _matvec_mode == 'tilewise':
            if not _has_tile_blocks:
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
                _S_extra = _build_s_extra_direct(
                    interface_node_to_idx=interface_node_to_idx,
                    n_iface=len(interface_nodes),
                    package_edges=model.package_data.package_edges,
                    dirichlet_nodes=model.pad_nodes,
                    island_nodes=island_nodes,
                )

        # Stage 3: island indices for the 'two_level' preconditioner's
        # coarse-space row zeroing -- independent of matvec_mode (two_level
        # works in both 'assembled' and 'tilewise' CG matvec modes).
        _island_idx = _island_idx_array(island_nodes, interface_node_to_idx)

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
            matvec_threads=_matvec_threads,
            matvec_dtype=_matvec_dtype,
            strict_dtype_rtol=_strict_dtype_rtol,
            island_idx=_island_idx,
            interface_coarse_geneo_k=_cg_settings.geneo_k,
            interface_coarse_geneo_tol=_cg_settings.geneo_tol,
            interface_coarse_eps_rank=_cg_settings.eps_rank,
            interface_coarse_max_cols=_cg_settings.max_cols,
            interface_coarse_max_bytes=_cg_settings.max_bytes,
            two_level_base=_cg_settings.two_level_base,
            neumann_max_bytes=_cg_settings.neumann_max_bytes,
            neumann_weight=_cg_settings.neumann_weight,
            neumann_reg=_cg_settings.neumann_reg,
            progress_every=_cg_settings.progress_every,
            interface_coarse_apply_mode=_cg_settings.apply_mode,
            interface_deflated_reproject_every=_cg_settings.deflated_reproject_every,
            warm_start_extrapolation=_cg_settings.warm_start_extrapolation,
        )

        # Synthetic stats object (matching SparseFactorAdapter fields used below)
        class _CGSolveResult:
            backend = 'cg'
            backend_info = (
                f"CG/{_cg_solver.matvec_mode}/"
                f"precond={_cg_solver.preconditioner_label}"
            )
            resolved_mode = 'cg'
            solve = _cg_solve_callable

        interface_lu_result = _CGSolveResult()
        if verbose:
            logger.info(
                "Interface CG solver: mode=%s, precond=%s, rtol=%.2e, n=%d",
                _matvec_mode, _cg_solver.preconditioner_label, _cg_rtol, len(interface_nodes),
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
    # D1 fix (item 1): always overwritten fresh (mirrors _tile_index_maps),
    # consumed by solve_dc's RHS scatter.
    ctx._tile_kept_port_pos = tile_kept_port_pos
    # S2/S13: parity with tile_kept_port_pos.
    ctx._tile_port_count = tile_port_count
    ctx._removed_interface_nodes = island_nodes
    ctx._S_global = S_global
    # S7: this IS the normal (S_global-assembling) factor path -- reset the
    # never-assemble flag so a context previously factored with
    # interface_drop_s_global=True, then re-factored (not merely
    # refactor()'d) with the setting off, can save() again.  Without this,
    # the flag stuck True forever and save() kept raising even though
    # ctx._S_global is now a valid, assembled matrix.
    ctx._s_global_dropped = False
    ctx.timings = timings
    # B2: store resolved interface solver mode and optional CG solver
    ctx._interface_solver_mode = _iface_resolved_mode
    # Finding 6: close the outgoing CG solver (if factor() is being re-run
    # on an already-factored context) before replacing it.
    _close_existing_cg_solver(ctx)
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
            tile_kept_port_pos=tile_kept_port_pos,  # D1 fix: persisted for save/load
            tile_port_count=tile_port_count,  # S2/S13: persisted for save/load
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

    if getattr(ctx, '_s_global_dropped', False):
        raise RuntimeError(
            "Cannot save: this context was factored with "
            "interface_drop_s_global=True, which never assembles S_global "
            "at all (not merely 'assemble then free') -- there is nothing "
            "for save() to persist that would let load()+refactor() rebuild "
            "the interface solve without workers.  Options: (1) don't call "
            "save() for never-assemble contexts -- workers already hold "
            "everything needed, so refactor() alone (no save/load) rebuilds "
            "tilewise via a streaming re-gather; (2) re-factor with "
            "interface_drop_s_global=False if a checkpoint is required."
        )

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


def _warn_if_coarse_space_lost(
    preconditioner: str,
    matvec_mode: str,
    want_tilewise: bool,
    matvec_mode_setting: Any,
    workers_attached: bool,
) -> None:
    """Finding 6 (round 2) / Finding 11: shared coarse-space-lost warning
    for BOTH refactor paths (DC and TD) -- previously ~18 lines duplicated
    verbatim in ``_refactor_dc_context`` and ``_refactor_transient_context``,
    built on a false premise the round-1 fix baked in (see below).

    Corrected semantics: warn ONLY when a two_level coarse space is
    GENUINELY lost by this refactor -- NOT merely whenever the resolved
    matvec mode isn't 'tilewise'.  The round-1 guard's premise ("the
    two_level coarse-space preconditioner requires 'tilewise' matvec
    mode") is false: ``preconditioner='two_level'`` works identically in
    'assembled' mode -- ``_augment_with_coarse_space``'s
    ``build_coarse_space`` call uses ``self._linear_op.matmat`` regardless
    of matvec mode (in 'assembled' mode that matmat is the sparse
    S_global matvec), and ``resolve_preconditioner()`` honours an EXPLICIT
    'two_level' setting verbatim in both modes.  Only 'auto' is
    mode-sensitive (it resolves to 'two_level' only for CG+tilewise, else
    'block_jacobi' -- see ``resolve_preconditioner``'s docstring).

    "Genuinely lost" is defined by comparing what the preconditioner
    setting WOULD resolve to at the matvec mode this refactor's request
    INTENDED (``want_tilewise`` -- the mode implied by interface_matvec_
    mode/attachment BEFORE this refactor's actual re-gather outcome is
    applied) against what it ACTUALLY resolves to at the matvec mode this
    refactor ended up with.  If intent would have reached 'two_level' but
    the actual outcome does not, that is a genuine regression from what
    was requested -- warn.  This needs no historical solver-state
    introspection (unreliable after a fresh ``load()`` -- the coarse
    space, like the per-tile Schur blocks it derives from, is never
    persisted, so ``ctx._cg_solver`` may be ``None`` at refactor time):

      - explicit 'two_level' + matvec forced to 'assembled' (regather
        failed): intended == actual == 'two_level' (explicit passthrough
        in both modes) -- NOT lost, no warning.
      - 'auto' + interface_matvec_mode explicitly 'tilewise' but workers
        not attached (regather impossible): intended (at 'tilewise') ==
        'two_level', actual (at 'assembled') == 'block_jacobi' -- LOST,
        warn (this is the "workers-not-attached auto case" -- a genuine
        loss).
      - 'auto' + interface_matvec_mode explicitly 'assembled' (by design,
        not by failure): intended == actual == 'assembled' both ways ==
        'block_jacobi' -- nothing was ever going to be two_level here, no
        warning.

    Distinct from the sibling ``_degrade_preconditioner_if_no_tile_index_
    maps`` guard below (a different failure mode; fires separately).

    Args:
        preconditioner: The (already-normalized, per Finding 1) resolved
            ``interface_preconditioner`` setting.
        matvec_mode: The matvec mode this refactor ACTUALLY resolved to
            ('assembled' or 'tilewise').
        want_tilewise: Whether tilewise was INTENDED for this refactor
            (the caller's ``_want_tilewise`` local, reflecting the
            interface_matvec_mode setting/attachment state, independent of
            whether the re-gather actually succeeded).
        matvec_mode_setting: The raw ``interface_matvec_mode`` setting
            (for the log message only).
        workers_attached: Whether workers were attached (for the log
            message only).
    """
    from .interface_iterative import resolve_preconditioner
    _intended_mode = 'tilewise' if want_tilewise else matvec_mode
    _would_have_resolved = resolve_preconditioner(preconditioner, 'cg', _intended_mode)
    _actually_resolved = resolve_preconditioner(preconditioner, 'cg', matvec_mode)
    if _would_have_resolved == 'two_level' and _actually_resolved != 'two_level':
        logger.warning(
            "Refactor: the two_level coarse-space preconditioner "
            "(interface_preconditioner=%r) would resolve to 'two_level' "
            "had the requested matvec mode been honoured, but this "
            "refactor resolved to %r matvec mode instead "
            "(interface_matvec_mode setting=%r, workers_attached=%s -- "
            "tile Schur blocks could not be re-gathered). The coarse "
            "space is LOST for this refactor: CG falls back to %r. To "
            "restore two_level, attach workers to ctx.model and call "
            "factor() (not just refactor()) for a full re-factorization.",
            preconditioner, matvec_mode, matvec_mode_setting, workers_attached,
            _actually_resolved,
        )


def _degrade_preconditioner_if_no_tile_index_maps(
    preconditioner: str, tile_index_maps: Optional[Dict[Any, np.ndarray]],
) -> str:
    """Finding 11: shared degrade-to-'jacobi' guard for BOTH refactor paths
    (previously duplicated verbatim in ``_refactor_dc_context`` and
    ``_refactor_transient_context``).

    'block_jacobi'/'two_level'/'auto' preconditioners need tile_index_maps
    for ownership assignment even in 'assembled' matvec mode; without it
    (e.g. a pre-B2 checkpoint), CG would otherwise silently run
    unpreconditioned (M=None). 'auto' is included because resolve_
    preconditioner('auto', ...) resolves to 'block_jacobi' whenever
    matvec_mode is forced to 'assembled' -- exactly the case this guard's
    `not tile_index_maps` condition selects for.

    Returns the (possibly-degraded) preconditioner setting.
    """
    if preconditioner in ('block_jacobi', 'two_level', 'auto') and not tile_index_maps:
        logger.warning(
            "Refactor: tile_index_maps unavailable on ctx.topology (missing "
            "or predates this field), so the %r preconditioner "
            "cannot be rebuilt after load()+refactor(). Degrading to "
            "'jacobi' (diagonal) for this refactor -- expect MORE CG "
            "iterations per solve than block_jacobi/two_level would need. "
            "Call factor() (not just refactor()) to restore full "
            "block_jacobi/two_level support.",
            preconditioner,
        )
        return 'jacobi'
    return preconditioner


def _refactor_dc_context(ctx: 'DistributedSolverContext', verbose: bool = False) -> None:
    """Rebuild coordinator solve callable from saved S_global (DC).

    For direct mode: rebuilds the CHOLMOD/SuperLU LU factorization.
    For CG mode: reconstructs the InterfaceCGSolver (assembled matvec only;
      tilewise mode requires a full factor() to re-obtain per-tile S_i blocks).

    Workers must already be factored before calling this.

    Item 3: a context factored via ``interface_drop_s_global`` never has
    ``ctx._S_global`` (by design -- that is the whole point).  Such a
    context CAN still be refactored, but only by re-gathering S_i from
    already-factored workers (the tilewise branch below); there is no
    saved-S_global fallback available for it, and switching to 'direct'
    mode is impossible without a full ``factor()``.
    """
    _workers_attached_early = bool(ctx.model is not None and ctx.model.workers)
    _was_never_assembled = getattr(ctx, '_s_global_dropped', False)
    if ctx._S_global is None and not (_was_never_assembled and _workers_attached_early):
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

    if _mode == 'direct' and ctx._S_global is None:
        raise RuntimeError(
            "Cannot refactor to interface_solver='direct': S_global was "
            "never assembled for this context (interface_drop_s_global=True "
            "at factor() time). Call factor() for a full re-factorization, "
            "or keep interface_solver='cg' for this refactor()."
        )

    coord_config = ctx.model.coordinator_solver_config if ctx.model is not None else None
    t0 = _time.perf_counter()

    # S12: close() the OUTGOING CG solver's persistent thread pool before
    # replacing ctx._cg_solver below -- mirror release()'s handling (the
    # lifecycle-parity checklist).  Without this, a session that alternates
    # factor()/solve/refactor() without release() accumulates one live
    # thread pool per refactor() call, reclaimed only by the weakref.
    # finalize GC safety net (not guaranteed timing).
    _old_cg_solver = getattr(ctx, '_cg_solver', None)
    if _old_cg_solver is not None:
        _close = getattr(_old_cg_solver, 'close', None)
        if callable(_close):
            _close()

    if _mode == 'direct':
        from pgmath.factor import _factor_conductance_matrix
        interface_lu_result = _factor_conductance_matrix(
            ctx._S_global, verbose=verbose, config=coord_config,
        )
        ctx._interface_lu = interface_lu_result.solve
        ctx._cg_solver = None
        ctx._interface_solver_mode = 'direct'
    else:
        # CG mode -- routed through build_interface_solver (Stage 1d).
        from .interface_iterative import build_interface_solver
        # Finding 13: shared settings-reading helper (see its docstring).
        # _model_settings (raw dict) is kept alive below for the
        # interface_matvec_mode read, which this helper deliberately
        # excludes.
        _model_settings = getattr(ctx.model, 'settings', {}) if ctx.model is not None else {}
        _cg_settings = _read_interface_cg_settings(ctx.model)
        _cg_rtol = _cg_settings.cg_rtol
        _cg_atol = _cg_settings.cg_atol
        _cg_maxiter = _cg_settings.cg_maxiter
        _cg_strict = _cg_settings.cg_strict
        _preconditioner = _cg_settings.preconditioner
        _bj_max_bytes = _cg_settings.bj_max_bytes
        _matvec_threads = _cg_settings.matvec_threads
        _matvec_dtype = _cg_settings.matvec_dtype
        _strict_dtype_rtol = _cg_settings.strict_dtype_rtol

        # tile_index_maps from the (possibly stale, pre-re-gather) topology
        # -- used for block-Jacobi ownership in the 'assembled' fallback
        # path, and as the "was tilewise ever used" signal.
        _topology_tile_index_maps = (
            getattr(ctx.topology, 'tile_index_maps', None)
            if ctx.topology is not None else None
        )

        # Item 9: when workers are ALREADY factored (attached to the model
        # from a prior factor() call in this session -- refactor()'s
        # documented precondition), re-gather D1-safe per-tile S_i via the
        # streaming protocol and rebuild TILEWISE instead of silently
        # downgrading to assembled.  Only S_global (not S_i) is ever saved,
        # so this re-gather is the only way tilewise survives a
        # save()/release()/load()/refactor() cycle within one process, or a
        # release()-without-reload refactor() in the same session.
        _matvec_mode_setting = _model_settings.get('interface_matvec_mode', 'auto')
        _workers_attached = bool(ctx.model is not None and ctx.model.workers)
        # A context whose S_global was never assembled (interface_drop_
        # s_global) has no 'assembled' fallback available at all -- force
        # tilewise regardless of the matvec_mode setting (the top-of-
        # function guard already ensured workers are attached in this case,
        # else it raised).
        _want_tilewise = (
            _was_never_assembled
            or _matvec_mode_setting == 'tilewise'
            or (_matvec_mode_setting in (None, 'auto') and _workers_attached)
        )
        # Finding 5: 'auto' (the default) resolves to tilewise whenever
        # workers are merely ATTACHED (bool(model.workers)), not whether
        # they are actually FACTORED (refactor()'s documented precondition).
        # The re-gather below (_gather_kept_tile_schur_streaming) calls
        # factor_and_compute_schur() on every worker, which performs a FULL
        # interior factorization + dense Schur computation -- the dominant
        # cost of factor() -- not a cheap coordinator-only rebuild. This can
        # silently turn a documented "seconds" refactor() into the same
        # multi-minute/hour cost as factor_tiles, and is wasted work if the
        # caller also calls factor_and_compute_schur() separately afterward
        # (a real risk: this module's own docstrings show that pattern).
        # Detecting "already factored" would need new per-worker RPC state
        # this Stage doesn't otherwise track; a loud WARNING (fired only
        # for the implicit 'auto' path, not an explicit interface_matvec_
        # mode='tilewise' request) gives visibility without that larger,
        # riskier change.
        if (
            _want_tilewise and _workers_attached and not _was_never_assembled
            and _matvec_mode_setting in (None, 'auto')
        ):
            logger.warning(
                "Refactor: interface_matvec_mode='auto' (default) resolved "
                "to 'tilewise' because workers are attached -- this "
                "re-gather calls factor_and_compute_schur() on every "
                "worker (full interior factorization + dense Schur), NOT a "
                "cheap coordinator-only rebuild. If workers were already "
                "factored this session, this repeats that work; if they "
                "were not, refactor()'s documented precondition (workers "
                "already factored) was not actually met. Set "
                "interface_matvec_mode='assembled' explicitly to force the "
                "cheap S_global-only rebuild instead."
            )

        # S5: mirror factor()'s streaming-assembly guard (_has_tile_blocks =
        # (not _use_streaming_dc) and bool(tile_schur_complements)).  Under
        # streaming_assembly, factor() never gathers dense per-tile S_i to
        # the coordinator at all -- that IS the memory bound streaming
        # exists to provide.  Without this guard, refactor() (e.g. after
        # save()/release()/load(), or just to change CG tolerances) would
        # unconditionally re-gather and sum every tile's kept dense Schur
        # block via _gather_kept_tile_schur_streaming, silently reintroducing
        # the exact OOM risk streaming_assembly=True was set to avoid.  Skip
        # the re-gather (fall back to 'assembled' + warning) unless this
        # context's never-assemble path forces tilewise (that path has no
        # 'assembled' fallback to begin with, so streaming_assembly is moot).
        if _want_tilewise and not _was_never_assembled and _workers_attached:
            _streaming_setting = _get_streaming_assembly_setting(ctx.model)
            if _streaming_setting == 'auto':
                _size_stats_raw = ctx.model.backend.call_all(
                    ctx.model.workers, 'get_schur_size_stats'
                )
                _refactor_would_stream = _should_stream(ctx.model, _size_stats_raw)
            else:
                _refactor_would_stream = bool(_streaming_setting)
            if _refactor_would_stream:
                logger.warning(
                    "Refactor: interface_matvec_mode=%r requests tilewise, "
                    "but streaming_assembly is active for this model -- "
                    "re-gathering dense per-tile S_i to the coordinator "
                    "would defeat the memory bound streaming_assembly=True "
                    "provides. Falling back to 'assembled'. Call factor() "
                    "with streaming_assembly disabled for a full "
                    "re-factorization if tilewise is required.",
                    _matvec_mode_setting,
                )
                _want_tilewise = False

        _tile_schur_complements: Optional[Dict[Any, np.ndarray]] = None
        _tile_index_maps = _topology_tile_index_maps
        _S_extra: Optional[sp.spmatrix] = None
        _matvec_mode = 'assembled'
        # Stage 3: hoisted OUT of the `if _want_tilewise` branch below --
        # island indices (for 'two_level' coarse-space row zeroing) are
        # needed regardless of which matvec mode this refactor ends up
        # resolving to (assembled or tilewise). S1: use this context's own
        # DC-mode island set, NOT topology.removed_interface_nodes (that
        # shared field is set once, at whichever mode -- DC or TD -- first
        # created the topology, and is NOT updated afterward; DC and TD
        # island sets genuinely differ when package_cap_edges exist).
        # topology.island_nodes is the DC-mode cache, kept mode-correct
        # independently of factor order (see _factor_dc_context).
        _interface_node_to_idx = ctx.interface_node_to_idx
        _island_nodes = (
            getattr(ctx.topology, 'island_nodes', None)
            if ctx.topology is not None else None
        )
        if _island_nodes is None:
            _island_nodes = ctx._removed_interface_nodes or set()
        _island_idx = _island_idx_array(_island_nodes, _interface_node_to_idx)

        if _want_tilewise and _workers_attached:
            _fresh_port_count: Dict[Any, int] = {}
            (
                _tile_schur_complements, _fresh_tile_index_maps, _fresh_kept_pos, _,
            ) = _gather_kept_tile_schur_streaming(
                ctx.model, _interface_node_to_idx, out_port_count=_fresh_port_count,
            )
            _tile_index_maps = _fresh_tile_index_maps
            # S14: keep _tile_index_maps and _tile_kept_port_pos in sync on
            # the context, mirroring factor() (which always sets both
            # together).  Leaving _tile_index_maps stale (pointing at the
            # shared topology's copy) while _tile_kept_port_pos is refreshed
            # would desync the D1-consistent pair solve_dc's RHS scatter
            # relies on if worker port lists ever drift between factor()
            # and refactor().
            ctx._tile_index_maps = _fresh_tile_index_maps
            ctx._tile_kept_port_pos = _fresh_kept_pos
            # S2/S13: parity with _tile_kept_port_pos -- refresh the
            # validation reference too, or filter_kept_rhs would keep
            # comparing against a stale (or empty) port count after a
            # refactor()-only session.
            ctx._tile_port_count = _fresh_port_count
            _S_extra = _build_s_extra_direct(
                interface_node_to_idx=_interface_node_to_idx,
                n_iface=len(_interface_node_to_idx),
                package_edges=ctx.model.package_data.package_edges,
                dirichlet_nodes=ctx.model.pad_nodes,
                island_nodes=_island_nodes,
            )
            _matvec_mode = 'tilewise'
        elif _matvec_mode_setting == 'tilewise' and not _workers_attached:
            logger.warning(
                "Refactor: interface_matvec_mode='tilewise' requested but no "
                "workers are attached to re-gather S_i (they must already be "
                "factored -- refactor()'s documented precondition). Falling "
                "back to 'assembled'. Call factor() for a full "
                "re-factorization to restore tilewise."
            )

        # Finding 6 (round 2) / Finding 11: shared helper -- see its
        # docstring for the corrected "genuinely lost" semantics (the
        # round-1 guard here fired whenever matvec_mode != 'tilewise',
        # which is a false premise: explicit 'two_level' rebuilds the
        # coarse space fine in 'assembled' mode too).
        _warn_if_coarse_space_lost(
            _preconditioner, _matvec_mode, _want_tilewise,
            _matvec_mode_setting, _workers_attached,
        )

        # Finding 1 (round 1) / Finding 11: shared helper -- tile_index_maps
        # is required for block_jacobi/two_level ownership assignment even
        # in 'assembled' matvec mode; degrade to 'jacobi' if unavailable
        # (e.g. a pre-B2 checkpoint) rather than silently building no
        # preconditioner at all.
        _preconditioner = _degrade_preconditioner_if_no_tile_index_maps(
            _preconditioner, _tile_index_maps,
        )
        _cg_stats: Dict[str, Any] = {}
        _solve_callable, _resolved_backend, cg_solver = build_interface_solver(
            S_global=ctx._S_global,
            interface_solver='cg',
            matvec_mode=_matvec_mode,
            tile_schur_complements=_tile_schur_complements,
            tile_index_maps=_tile_index_maps,
            S_extra=_S_extra,
            preconditioner=_preconditioner,
            rtol=_cg_rtol,
            atol=_cg_atol,
            maxiter=_cg_maxiter,
            strict=_cg_strict,
            block_jacobi_max_bytes=_bj_max_bytes,
            verbose=verbose,
            cg_stats_dict=_cg_stats,
            n_interface=len(ctx.interface_node_to_idx) if ctx._S_global is None else None,
            matvec_threads=_matvec_threads,
            matvec_dtype=_matvec_dtype,
            strict_dtype_rtol=_strict_dtype_rtol,
            island_idx=_island_idx,
            interface_coarse_geneo_k=_cg_settings.geneo_k,
            interface_coarse_geneo_tol=_cg_settings.geneo_tol,
            interface_coarse_eps_rank=_cg_settings.eps_rank,
            interface_coarse_max_cols=_cg_settings.max_cols,
            interface_coarse_max_bytes=_cg_settings.max_bytes,
            two_level_base=_cg_settings.two_level_base,
            neumann_max_bytes=_cg_settings.neumann_max_bytes,
            neumann_weight=_cg_settings.neumann_weight,
            neumann_reg=_cg_settings.neumann_reg,
            progress_every=_cg_settings.progress_every,
            interface_coarse_apply_mode=_cg_settings.apply_mode,
            interface_deflated_reproject_every=_cg_settings.deflated_reproject_every,
            warm_start_extrapolation=_cg_settings.warm_start_extrapolation,
        )
        ctx._interface_lu = _solve_callable
        ctx._cg_solver = cg_solver
        ctx._interface_solver_mode = 'cg'
        # Item 3: still true after a successful tilewise re-gather refactor
        # (S_global remains unassembled); a downgrade to 'assembled' would
        # have raised above (no fallback exists), so reaching here with
        # _was_never_assembled means the re-gather succeeded.
        ctx._s_global_dropped = _was_never_assembled

    elapsed = _time.perf_counter() - t0
    ctx.is_factored = True
    logger.info(
        "Refactored DC coordinator solve (%s) from saved S_global in %.3fs",
        _mode, elapsed,
    )


# ---------------------------------------------------------------------------
# Transient context: factor / save / load / refactor
# ---------------------------------------------------------------------------


def _stamp_worker_transient_factor(
    model: Optional['DistributedPowerGridModel'], dt_scaled: float, method: str,
) -> None:
    """Finding 1: record the ``(dt_scaled, method)`` the model's WORKERS are
    currently factored for (worker-side transient block system
    ``A = G + C_coeff*C``, set by a ``factor_transient_system()`` RPC).

    Workers are shared, single-owner state across every live
    ``DistributedTransientContext`` for this model -- whichever context most
    recently ran ``factor_transient_system()`` on the workers (via its own
    ``factor()``, or a never-assembled ``refactor()``'s tilewise re-gather)
    wins, silently invalidating every OTHER live transient context's
    worker-side factorization: per-step RHS (``get_transient_reduced_rhs_arr``)
    and interior recovery (``recover_transient_and_update_peaks_arr``) both
    use whatever the workers are CURRENTLY factored for, not whichever
    dt/method the calling context was originally prepared with.

    ``solve_transient()`` (``solver_td.py``) reads this stamp and compares
    it against its own context's ``(dt_scaled, integration_method)`` before
    doing any per-step work, raising a loud ``RuntimeError`` on mismatch
    instead of silently solving with mismatched coordinator/worker state.

    Call this at every site that runs ``factor_transient_system()`` (or
    ``factor_transient_and_cache_schur()``) on the workers:
    ``_factor_transient_context`` (assembled path, both the streaming and
    bulk branches), ``_factor_transient_context_no_s_global`` (never-assemble
    factor path, via its ``_gather_kept_tile_schur_streaming`` call), and
    ``_refactor_transient_context``'s tilewise re-gather branch (both the
    assembled-context 'auto'-resolved-to-tilewise case and the
    never-assembled case).

    Finding 1 (round 2 review): callers MUST invalidate the stamp to None
    (``model._worker_transient_factor_key = None``) BEFORE dispatching the
    worker-side ``factor_transient_system()`` RPC, and only call THIS
    function (writing the real key) after every worker has completed. A
    mid-gather failure then leaves the stamp at None -- "workers'
    transient factorization state unknown/absent" -- instead of a stale
    value that could pass a naive not-None guard while some tiles hold the
    NEW dt/method and others the OLD one.

    Finding 3 (round 2 review): ``method`` is canonicalized via
    ``.lower()`` here (and independently by the guard,
    ``_check_worker_transient_stamp`` in ``solver_td.py``, on its own
    context-side key) since every numeric consumer treats any value other
    than ``'trap'`` as Backward Euler -- ``'be'`` and ``'BE'`` are
    numerically identical configurations and must compare equal.
    """
    if model is not None:
        model._worker_transient_factor_key = (dt_scaled, method.lower())


def _invalidate_worker_transient_stamp(
    model: Optional['DistributedPowerGridModel'],
) -> None:
    """Finding 1 (round 2 review): reset the worker transient-factor stamp
    to None BEFORE dispatching a ``factor_transient_system()`` /
    ``factor_transient_and_cache_schur()`` RPC to the workers.

    Call this immediately before EVERY dispatch site that
    :func:`_stamp_worker_transient_factor` documents (factor, never-
    assemble factor, never-assembled refactor's tilewise re-gather); pair
    it with a :func:`_stamp_worker_transient_factor` call written only
    after that same dispatch's gather/``call_all`` returns successfully.
    If the dispatch raises partway through (e.g. some tiles already
    re-factored at the new dt/method, others not yet reached), the stamp
    is left at None -- "workers' transient factorization state unknown/
    absent" -- rather than the stale pre-call value, which could otherwise
    describe an inconsistent mix of old- and new-dt tiles as if it were a
    single valid factorization. See ``_check_worker_transient_stamp``
    (``solver_td.py``) for how None is treated by the guard.
    """
    if model is not None:
        model._worker_transient_factor_key = None


def _build_td_package_edge_sets(
    pkg_res_edges: List[Tuple[str, str, float]],
    pkg_cap_edges: List[Tuple[str, str, float]],
    C_coeff: float,
) -> Tuple[List[Tuple[str, str, float]], List[Tuple[str, str, float]]]:
    """Finding 9 (round 2 review): shared C_coeff-weighted, c_fF>0-filtered
    package-cap edge construction -- the "combined_edges = resistive +
    C_coeff-weighted package caps" pattern used by the TD interface
    ordering, S_extra^TD stamping, and the rhs_dirichlet_G linearity delta.

    Previously this exact three-line filter+weight (`c_fF > 0` then
    `C_coeff * c_fF`) was hand-duplicated at four sites
    (:func:`_derive_td_dirichlet_vectors`, the never-assemble factor's
    combined_edges construction, the assembled factor's cap-only-edges
    construction, and the refactor's never-assembled-branch cap-only-edges
    reconstruction) and had to be edited in lockstep -- exactly the drift
    risk :func:`_derive_td_dirichlet_vectors`'s docstring describes for the
    derivation itself: a future edit to the filter or weighting applied to
    only one copy would make a ``release()``-``refactor()``'d never-
    assembled context numerically diverge from a freshly ``factor()``'d
    one with no error.

    Args:
        pkg_res_edges: Resistive package edges ``(u, v, g_mS)`` (pass ``[]``
            when only ``cap_only_edges`` is needed).
        pkg_cap_edges: Package capacitor edges ``(u, v, c_fF)``.
        C_coeff: Capacitance coefficient (1/dt_ps BE, 2/dt_ps TR).

    Returns:
        ``(combined_edges, cap_only_edges)`` -- ``combined_edges`` is
        ``pkg_res_edges`` with the weighted cap-only edges appended (the TD
        node universe for the interface ordering / S_extra^TD stamping);
        ``cap_only_edges`` is the weighted-cap-only subset alone (consumed
        by the rhs_dirichlet_G linearity delta).
    """
    cap_only_edges = [
        (u, v, C_coeff * c_fF) for u, v, c_fF in pkg_cap_edges if c_fF > 0
    ]
    combined_edges = list(pkg_res_edges) + cap_only_edges
    return combined_edges, cap_only_edges


def _derive_td_dirichlet_vectors(
    *,
    model: 'DistributedPowerGridModel',
    interface_nodes: List[str],
    interface_node_to_idx: Dict[str, int],
    rhs_dirichlet_pkg: np.ndarray,
    rhs_dirichlet_from_tiles: np.ndarray,
    pkg_cap_edges: List[Tuple[str, str, float]],
    C_coeff: float,
    island_nodes: Optional[Set[str]],
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Finding 7: shared TD Dirichlet-vector derivation, used identically by
    :func:`_factor_transient_context_no_s_global` (factor) and
    :func:`_refactor_transient_context`'s never-assembled branch (refactor).

    Previously ~35 lines of this derivation were hand-duplicated between the
    two functions and had to be patched TWICE in lockstep during this work
    package's implementation (see the "Major finding (review round 1/2)"
    comments each call site used to carry) -- exactly the drift risk this
    extraction removes: a future edit to the delta ordering, the cap-edge
    filter, or the penalty formula that touches only one copy would make a
    ``release()``-``refactor()``'d context silently diverge from a freshly
    ``factor()``'d one.

    Given the package-edges-only Dirichlet contribution
    (``rhs_dirichlet_pkg``, computed from ``combined_edges`` = resistive +
    C_coeff-weighted package caps -- via ``_compute_interface_ordering`` in
    the factor path, or ``_compute_rhs_dirichlet_from_edges`` directly in
    the refactor path, since refactor must NOT re-derive the already-fixed
    interface ordering) and the tile-embedded ("D1") contribution
    (``rhs_dirichlet_from_tiles``, from a streaming factor/re-gather), this
    derives:

      1. The PRE-penalty A-based Dirichlet RHS: ``rhs_dirichlet_pkg +
         rhs_dirichlet_from_tiles``.
      2. ``rhs_dirichlet_G`` via the Finding-12-style linearity delta: exact
         because the tile-embedded contribution does not depend on which
         package ``extra_edges`` set fixed the (already-final) interface
         ordering -- only the PACKAGE-edge term changes between
         ``combined_edges`` and the resistive-only set. This MUST run
         before the island penalty is folded into ``rhs_dirichlet_A``
         (step 3), or ``rhs_dirichlet_G`` would inherit the penalty too and
         the separate per-step ``island_penalty_rhs`` vector (step 3) would
         double-count it every time-loop step.
      3. The POST-penalty ``rhs_dirichlet_A`` (island rows get
         ``+ISLAND_PENALTY_CONDUCTANCE * vdd``) and the separate per-step
         ``island_penalty_rhs`` vector (T1 fix), which the time loop adds
         EXACTLY ONCE per step for both BE and TR (never folded into
         ``rhs_dirichlet_G``, which TR scales by 2 -- that would double the
         penalty forcing term under TR).

    Args:
        model: For ``pad_nodes``/``vdd``.
        interface_nodes: Ordered interface-unknown node list (already
            fixed -- NOT recomputed here).
        interface_node_to_idx: ``{node: index}`` for the same ordering.
        rhs_dirichlet_pkg: Package-edges-only Dirichlet contribution (from
            ``combined_edges``), computed by the caller.
        rhs_dirichlet_from_tiles: Tile-embedded ("D1") Dirichlet
            contribution, from a streaming factor/re-gather.
        pkg_cap_edges: Package capacitor edges ``(u, v, c_fF)``.
        C_coeff: Capacitance coefficient (1/dt_ps BE, 2/dt_ps TR).
        island_nodes: Interface nodes to penalize, or None/empty.

    Returns:
        ``(rhs_dirichlet_A, rhs_dirichlet_G, island_penalty_rhs)`` --
        ``island_penalty_rhs`` is ``None`` when there are no islands.
    """
    # Finding 9: shared combined/cap-only edge construction (see
    # _build_td_package_edge_sets's docstring) -- only cap_only_edges is
    # needed here, so pkg_res_edges is passed as [] (combined_edges would
    # just equal cap_only_edges then, and is discarded).
    _, cap_only_edges = _build_td_package_edge_sets([], pkg_cap_edges, C_coeff)
    rhs_dirichlet_A = rhs_dirichlet_pkg + rhs_dirichlet_from_tiles

    if cap_only_edges:
        rhs_from_cap_only = _compute_rhs_dirichlet_from_edges(
            extra_edges=cap_only_edges,
            unknown_list=interface_nodes,
            unknown_to_idx=interface_node_to_idx,
            dirichlet_nodes=model.pad_nodes,
            dirichlet_voltage=model.vdd,
        )
        rhs_dirichlet_G = rhs_dirichlet_A - rhs_from_cap_only
    else:
        rhs_dirichlet_G = rhs_dirichlet_A.copy()

    island_penalty_rhs: Optional[np.ndarray] = None
    if island_nodes:
        island_pen_idx = np.array(
            [interface_node_to_idx[n] for n in island_nodes
             if n in interface_node_to_idx],
            dtype=np.intp,
        )
        rhs_dirichlet_A = rhs_dirichlet_A.copy()
        rhs_dirichlet_A[island_pen_idx] += ISLAND_PENALTY_CONDUCTANCE * model.vdd
        island_penalty_rhs = np.zeros(len(interface_node_to_idx), dtype=np.float64)
        island_penalty_rhs[island_pen_idx] = ISLAND_PENALTY_CONDUCTANCE * model.vdd

    return rhs_dirichlet_A, rhs_dirichlet_G, island_penalty_rhs


def _factor_transient_context_no_s_global(
    ctx: 'DistributedTransientContext', model: Any, verbose: bool = False,
) -> None:
    """TD extension of item 3: factor a CG+tilewise TRANSIENT context
    WITHOUT ever assembling S_global^TD.

    Mirrors :func:`_factor_dc_context_no_s_global`'s two-round-trip
    streaming design (port-name-lists-only ordering pre-pass, then one
    streamed factor-and-gather pass with immediate kept-position slicing),
    extended for the transient system the way :func:`_factor_transient_context`
    extends the assembled DC path:

      - The interface ordering (and the package-edges-only piece of the
        Dirichlet RHS) is computed from ``combined_edges`` = resistive +
        C_coeff-weighted package caps, NOT DC's resistive-only
        ``package_edges`` -- cap-edge endpoints can add unknown nodes to the
        interface universe that DC's ordering never sees (see
        ``compute_interface_node_split``).
      - Island detection uses ``combined_edges`` too (TD-mode, cap-aware),
        via the same summaries union-find as the DC never-assemble path --
        never the DC-mode resistive-only set (:func:`_can_use_no_s_global_path`
        is the caller's precondition check, same as DC).
      - The tile-embedded ("D1", a Dirichlet pad directly on a tile's own
        port list) contribution to the Dirichlet RHS is gathered from the
        TRANSIENT per-tile Schur (``factor_transient_system``, A = G +
        C_coeff*C), not the DC one -- :func:`_gather_kept_tile_schur_streaming`'s
        ``transient_dt_scaled``/``transient_method`` kwargs route the
        streamed RPC to the transient worker method.
      - ``rhs_dirichlet_G`` (the vector the time loop actually reads -- see
        the "Transient Dirichlet RHS in time loop" CLAUDE.md pitfall) is
        derived from ``rhs_dirichlet_A`` by the SAME linearity delta
        :func:`_factor_transient_context`'s Finding 12 uses: for a FIXED
        interface ordering, the tile-embedded contribution to the
        Dirichlet-unknown coupling does not depend on which package
        ``extra_edges`` set was used to fix that ordering (only the
        PACKAGE-edge piece does) -- so
        ``rhs_dirichlet_G = rhs_dirichlet_A - rhs_from_cap_only_edges``,
        where the subtracted term is computed from the cap-only edges alone
        (no S_i involvement, via :func:`_compute_rhs_dirichlet_from_edges`).
        This needs only ONE tile-gather pass, not two -- the same reasoning
        the streaming branch of the assembled path already relies on, now
        extended one level further (never gathering S_global at all).
      - ``S_extra^TD`` is built by :func:`_build_s_extra_direct`'s
        mode-dependent TD branch (resistive package edges + C_coeff-weighted
        package caps + island-penalty diagonal) -- rebuilt on every
        ``prepare_transient()`` call, exactly as the assembled path's
        S_extra^TD is (dt/method-dependent, never shared with a DC
        instance).

    The per-step ``island_penalty_rhs`` vector (T1 fix) is built directly
    here from the same island index array used for the diagonal stamp,
    instead of via ``apply_island_penalty`` (which operates on an assembled
    S_global this path never builds).
    """
    from .result import DistributedTopologyContext
    from .interface_iterative import build_interface_solver
    # Finding 12: build_interface_package_matrices is no longer imported
    # directly here -- _build_s_extra_direct(return_package_matrices=True)
    # below returns the (G_pkg_uu, C_pkg_uu) pair from its own single
    # stamping pass.
    from pgmath.schur import detect_interface_islands_from_summaries

    timings: Dict[str, Any] = {}
    tile_configs = model.metadata.tile_configs
    method = ctx.integration_method
    dt_scaled = ctx.dt_scaled
    C_coeff = ctx.C_coeff

    pkg_res_edges = model.package_data.package_edges
    pkg_cap_edges = model.package_data.package_cap_edges
    # Finding 9: shared combined/cap-only edge construction (see
    # _build_td_package_edge_sets's docstring).
    combined_edges, _cap_only_edges_f = _build_td_package_edge_sets(
        pkg_res_edges, pkg_cap_edges, C_coeff,
    )
    has_cap = bool(_cap_only_edges_f)
    # Note: the cap-only-edges list (package caps alone, weighted) used by
    # the rhs_dirichlet_G linearity delta is independently rebuilt by
    # _derive_td_dirichlet_vectors (Finding 7) via the same helper, from
    # pkg_cap_edges/C_coeff.

    # Round-trip 1: port name lists only (no factoring).
    t0 = _time.perf_counter()
    port_lists_raw = model.backend.call_all(model.workers, 'get_port_node_list')
    tile_port_node_lists: Dict[Any, List[str]] = {
        tile_configs[i].tile_id: pl for i, pl in enumerate(port_lists_raw)
    }
    timings['gather_port_lists'] = _time.perf_counter() - t0

    # Interface ordering + package-edges-only Dirichlet RHS, from
    # combined_edges (the TD node universe -- may include cap-edge
    # endpoints DC's resistive-only ordering never sees).
    t0 = _time.perf_counter()
    interface_nodes, interface_node_to_idx, rhs_dirichlet_pkg = _compute_interface_ordering(
        tile_port_node_lists=tile_port_node_lists,
        extra_edges=combined_edges,
        dirichlet_nodes=model.pad_nodes,
        dirichlet_voltage=model.vdd,
    )
    timings['compute_ordering'] = _time.perf_counter() - t0

    # Island detection: summaries union-find, TD-mode (combined_edges) --
    # same guard as the DC never-assemble path (S6).  Detection ONLY here
    # (island_nodes set) -- penalty APPLICATION is deferred until after
    # rhs_dirichlet_G is derived below (see the comment there for why: this
    # mirrors _factor_transient_context's exact ordering, where Finding 12's
    # linearity delta runs BEFORE apply_island_penalty).
    #
    # Finding 11: mirror the assembled path's topology island cache (A7 --
    # islands computed once, not re-run every prepare_transient) instead of
    # unconditionally re-detecting. Cache lookup order, identical to the
    # assembled path's (see its "Island detection on transient system"
    # comment a few hundred lines below in _factor_transient_context):
    #   1. island_nodes_td  -- transient-mode cache (always mode-correct).
    #   2. island_nodes     -- DC-mode cache; ONLY reused when
    #                          package_cap_edges is empty (cap edges can
    #                          bridge components that are resistively
    #                          disconnected, making the transient island set
    #                          a strict subset of the DC set -- reusing the
    #                          DC result when caps exist would
    #                          over-penalise cap-bridged interface nodes).
    t0 = _time.perf_counter()
    _td_has_cap = bool(pkg_cap_edges)
    _cached_islands_td: Optional[Set[str]] = None
    if ctx.topology is not None:
        _cached_islands_td = getattr(ctx.topology, 'island_nodes_td', None)
        if _cached_islands_td is None and not _td_has_cap:
            _cached_islands_td = getattr(ctx.topology, 'island_nodes', None)

    if _cached_islands_td is not None:
        island_nodes = _cached_islands_td
        logger.debug(
            "Transient (never-assemble) island detection: cache hit (%d "
            "islands), union-find skipped.", len(island_nodes),
        )
    else:
        _check_summaries_mixed_state(model)
        island_nodes = detect_interface_islands_from_summaries(
            component_summaries=model.component_summaries,
            interface_node_to_idx=interface_node_to_idx,
            pad_nodes=model.pad_nodes,
            extra_edges=combined_edges,
        )
    timings['detect_interface_islands'] = _time.perf_counter() - t0

    # Round-trip 2: streaming TRANSIENT factor + D1-safe kept-slice gather,
    # plus each tile's own pad-adjacent RHS contribution to rhs_dirichlet_A
    # and its total capacitance (for ctx.has_capacitance).
    t0 = _time.perf_counter()
    tile_port_count: Dict[Any, int] = {}
    tile_cap: Dict[Any, float] = {}
    # Finding 1 (round 2 review): invalidate the stamp to None BEFORE
    # dispatching factor_transient_system() to the workers, not just before
    # writing the real key after -- a failure partway through this streamed
    # per-tile gather (some tiles already re-factored at dt_scaled/method,
    # others not yet reached) must leave the stamp at None ("state unknown")
    # rather than the stale pre-call value, which could describe a mix of
    # old- and new-dt tiles as if it were a single consistent factorization.
    _invalidate_worker_transient_stamp(model)
    (
        tile_schur_complements, tile_index_maps, tile_kept_port_pos,
        rhs_dirichlet_from_tiles,
    ) = _gather_kept_tile_schur_streaming(
        model, interface_node_to_idx,
        transient_dt_scaled=dt_scaled, transient_method=method,
        dirichlet_voltage=model.vdd, n_iface=len(interface_nodes),
        out_port_count=tile_port_count, out_tile_cap=tile_cap,
    )
    # Finding 1: every worker just successfully ran
    # factor_transient_system(dt_scaled, method) -- stamp the model so
    # solve_transient()/analyze_adjoint() (any context, this one or
    # another) can detect a stale worker/coordinator dt/method pairing.
    # Only reached if the gather above completed without raising.
    _stamp_worker_transient_factor(model, dt_scaled, method)
    timings['factor_transient_tiles'] = _time.perf_counter() - t0
    if sum(tile_cap.values()) > 0:
        has_cap = True

    # Finding 7: shared derivation (linearity delta + penalty ordering) --
    # see _derive_td_dirichlet_vectors's docstring for why the ordering
    # matters and why this used to be independently duplicated in
    # _refactor_transient_context.
    rhs_dirichlet_A, rhs_dirichlet_G, island_penalty_rhs = _derive_td_dirichlet_vectors(
        model=model,
        interface_nodes=interface_nodes,
        interface_node_to_idx=interface_node_to_idx,
        rhs_dirichlet_pkg=rhs_dirichlet_pkg,
        rhs_dirichlet_from_tiles=rhs_dirichlet_from_tiles,
        pkg_cap_edges=pkg_cap_edges,
        C_coeff=C_coeff,
        island_nodes=island_nodes,
    )
    if island_nodes:
        logger.warning(
            "Transient: penalized %d interface island nodes (never-assemble "
            "path, shorted to %.3f V)", len(island_nodes), model.vdd,
        )

    # D2: S_extra^TD by direct stamping -- mode-dependent (resistive package
    # edges + C_coeff-weighted package caps + island-penalty diagonal).
    # Finding 12: request the (G_pkg_uu, C_pkg_uu) pair this already stamps
    # internally so the separate build_interface_package_matrices() call
    # below (for ctx.C_package_uu/_G_package_uu) doesn't repeat the same
    # package-edge stamping pass a second time.
    t0 = _time.perf_counter()
    S_extra, (G_pkg_uu, C_pkg_uu) = _build_s_extra_direct(
        interface_node_to_idx=interface_node_to_idx,
        n_iface=len(interface_nodes),
        package_edges=pkg_res_edges,
        dirichlet_nodes=model.pad_nodes,
        island_nodes=island_nodes,
        package_cap_edges=pkg_cap_edges,
        C_coeff=C_coeff,
        return_package_matrices=True,
    )
    timings['build_s_extra'] = _time.perf_counter() - t0

    # Finding 13: shared settings-reading helper.
    _cg_settings = _read_interface_cg_settings(model)
    _preconditioner = _cg_settings.preconditioner
    _cg_rtol = _cg_settings.cg_rtol
    _cg_atol = _cg_settings.cg_atol
    _cg_maxiter = _cg_settings.cg_maxiter
    _cg_strict = _cg_settings.cg_strict
    _bj_max_bytes = _cg_settings.bj_max_bytes
    _matvec_threads = _cg_settings.matvec_threads
    _matvec_dtype = _cg_settings.matvec_dtype
    _strict_dtype_rtol = _cg_settings.strict_dtype_rtol
    _island_idx = _island_idx_array(island_nodes, interface_node_to_idx)

    t0 = _time.perf_counter()
    _cg_stats: Dict[str, Any] = {}
    _cg_solve_callable, _resolved_mode, _cg_solver = build_interface_solver(
        S_global=None,
        interface_solver='cg',
        tile_schur_complements=tile_schur_complements,
        tile_index_maps=tile_index_maps,
        S_extra=S_extra,
        matvec_mode='tilewise',
        preconditioner=_preconditioner,
        rtol=_cg_rtol,
        atol=_cg_atol,
        maxiter=_cg_maxiter,
        strict=_cg_strict,
        block_jacobi_max_bytes=_bj_max_bytes,
        verbose=verbose,
        cg_stats_dict=_cg_stats,
        n_interface=len(interface_nodes),
        matvec_threads=_matvec_threads,
        matvec_dtype=_matvec_dtype,
        strict_dtype_rtol=_strict_dtype_rtol,
        island_idx=_island_idx,
        interface_coarse_geneo_k=_cg_settings.geneo_k,
        interface_coarse_geneo_tol=_cg_settings.geneo_tol,
        interface_coarse_eps_rank=_cg_settings.eps_rank,
        interface_coarse_max_cols=_cg_settings.max_cols,
        interface_coarse_max_bytes=_cg_settings.max_bytes,
        two_level_base=_cg_settings.two_level_base,
        neumann_max_bytes=_cg_settings.neumann_max_bytes,
        neumann_weight=_cg_settings.neumann_weight,
        neumann_reg=_cg_settings.neumann_reg,
        progress_every=_cg_settings.progress_every,
        interface_coarse_apply_mode=_cg_settings.apply_mode,
        interface_deflated_reproject_every=_cg_settings.deflated_reproject_every,
        warm_start_extrapolation=_cg_settings.warm_start_extrapolation,
    )
    timings['factor_transient_interface'] = _time.perf_counter() - t0
    timings['total_prepare_transient'] = sum(
        v for v in timings.values() if isinstance(v, (int, float))
    )

    # Finding 12: G_pkg_uu/C_pkg_uu already came back from the
    # _build_s_extra_direct(return_package_matrices=True) call above -- no
    # second build_interface_package_matrices() pass needed here.

    if verbose:
        logger.info(
            "=== Distributed DDM Prepare Transient Statistics "
            "(never-assemble S_global) ===",
        )
        logger.info(
            "Method: %s  |  dt: %.1f ps  |  C_coeff: %.4f", method, dt_scaled, C_coeff,
        )
        logger.info(
            "Transient interface: %d unknowns, %d islands penalized, "
            "matvec_threads=%d, matvec_dtype=%s",
            len(interface_nodes), len(island_nodes),
            _cg_solver.matvec_threads, _cg_solver.matvec_dtype,
        )
        for _phase in ('gather_port_lists', 'compute_ordering',
                       'detect_interface_islands', 'factor_transient_tiles',
                       'build_s_extra', 'factor_transient_interface'):
            if _phase in timings:
                logger.info("  %-36s %.3fs", _phase, timings[_phase])
        logger.info("=== Total Prepare: %.3fs ===", timings['total_prepare_transient'])

    # Populate context fields (mirrors _factor_transient_context's field set).
    ctx._interface_lu = _cg_solve_callable
    ctx._interface_nodes = interface_nodes
    ctx._interface_node_to_idx = interface_node_to_idx
    ctx.rhs_dirichlet_A = rhs_dirichlet_A
    ctx._rhs_dirichlet_G = rhs_dirichlet_G
    ctx.island_penalty_rhs = island_penalty_rhs
    ctx._tile_index_maps = tile_index_maps
    ctx._tile_kept_port_pos = tile_kept_port_pos
    ctx._tile_port_count = tile_port_count
    ctx._removed_interface_nodes = island_nodes
    ctx._S_global = None
    # Item 3 (TD extension): marks that S_global^TD was never assembled this
    # factor() call -- save() checks this and raises with guidance.
    ctx._s_global_dropped = True
    ctx.has_capacitance = has_cap
    ctx.C_package_uu = C_pkg_uu if C_pkg_uu.nnz > 0 else None
    ctx._G_package_uu = G_pkg_uu if G_pkg_uu.nnz > 0 else None
    ctx.timings = timings
    ctx._interface_solver_mode = 'cg'
    # Finding 6: close the outgoing CG solver (if factor() is being re-run
    # on an already-factored context) before replacing it.
    _close_existing_cg_solver(ctx)
    ctx._cg_solver = _cg_solver

    # Finding 11: mirror the assembled path's cross-mode cache population
    # (see its "Build topology context if not already provided" comment) --
    # when package_cap_edges is empty, DC and transient BFS/union-find
    # produce identical results, so pre-populate island_nodes (the DC-mode
    # field) too, letting a subsequent DC prepare() skip its own detection.
    # When cap edges are present, leave island_nodes unset so DC still runs
    # its own mode-correct detection.
    if ctx.topology is None:
        ctx.topology = DistributedTopologyContext(
            interface_nodes=interface_nodes,
            interface_node_to_idx=interface_node_to_idx,
            tile_index_maps=tile_index_maps,
            rhs_dirichlet_G=rhs_dirichlet_G,
            G_package_uu=G_pkg_uu if G_pkg_uu.nnz > 0 else None,
            removed_interface_nodes=island_nodes,
            island_nodes=None if _td_has_cap else island_nodes,  # DC cross-mode (safe when no caps)
            island_nodes_td=island_nodes,
            tile_kept_port_pos=tile_kept_port_pos,
            tile_port_count=tile_port_count,
        )
    else:
        if getattr(ctx.topology, 'island_nodes_td', None) is None:
            ctx.topology.island_nodes_td = island_nodes
        if not _td_has_cap and getattr(ctx.topology, 'island_nodes', None) is None:
            ctx.topology.island_nodes = island_nodes

    ctx.is_factored = True


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

    # Item 3 (TD extension): interface_drop_s_global's "never assemble
    # S_global" path now covers BOTH DC and transient factor -- dispatch to
    # the dedicated no-S_global transient factor path when its preconditions
    # hold (same check as DC: explicit interface_solver='cg', matvec_mode
    # tilewise/auto, summaries-based island detection). Falls back to the
    # normal (S_global-assembling) path with a WARNING otherwise, mirroring
    # _factor_dc_context's dispatch exactly.
    if _get_drop_s_global_setting(model):
        _ok, _reason = _can_use_no_s_global_path(model)
        if _ok:
            _factor_transient_context_no_s_global(ctx, model, verbose)
            return
        logger.warning(
            "interface_drop_s_global=True but preconditions are not met "
            "(%s); falling back to the normal (S_global-assembling) factor "
            "path.", _reason,
        )

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

    # Finding 1 (round 2 review): invalidate BEFORE dispatching either
    # branch's factor_transient_*() call_all -- see
    # _invalidate_worker_transient_stamp's docstring.
    _invalidate_worker_transient_stamp(model)

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

    # Finding 1: workers just ran factor_transient_system/
    # factor_transient_and_cache_schur(dt_scaled, method) -- stamp the model
    # so solve_transient() can detect a stale worker/coordinator pairing.
    _stamp_worker_transient_factor(model, dt_scaled, method)

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

    # Finding 9: shared combined/cap-only edge construction (see
    # _build_td_package_edge_sets's docstring) -- _cap_only_edges is reused
    # below (~ the Finding-12 linearity-delta comment) instead of being
    # rebuilt a second time with the same filter/weighting.
    combined_edges, _cap_only_edges = _build_td_package_edge_sets(
        pkg_res_edges, pkg_cap_edges, C_coeff,
    )
    has_cap = total_tile_cap > 0 or bool(_cap_only_edges)
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
    _streaming_tile_kept_port_pos_td = None

    # Finding 12 (hoisted out of the streaming-only branch, S3's own
    # linearity trick): G_full = G_full_tiles + G_full_extra(E), where
    # G_full_tiles depends only on the tile S_A_i data and is IDENTICAL
    # whether E = combined_edges or E = pkg_res_edges (same tile_schur_
    # complements either way). So rhs_dirichlet_G = rhs_dirichlet_A - (rhs
    # contribution of the cap-only edges alone), computed via
    # _compute_rhs_dirichlet_from_edges (package-edges-only, no tile S_i
    # involvement) applied to combined_edges' cap-only DELTA over
    # pkg_res_edges. This holds for BOTH the streaming and bulk assembly
    # paths (pkg_res_edges/pkg_cap_edges/C_coeff are all already computed
    # above, before the streaming/bulk split), so it is computed once here
    # and reused by both branches below -- the bulk path no longer needs a
    # second full assemble_schur_complement_system() call (a complete
    # second global Schur COO scatter of every tile's S_A_i block) solely
    # to obtain this vector. (_cap_only_edges itself was already built
    # above, step 2, via _build_td_package_edge_sets -- Finding 9.)

    if _use_streaming_td:
        # B3 streaming transient: COO shards of S_A streamed tile-by-tile.
        # combined_edges includes the effective cap contribution (C_coeff * C).
        (
            S_global, rhs_dirichlet_A, interface_nodes, interface_node_to_idx,
            _streaming_tile_index_maps_td, _streaming_tile_kept_port_pos_td,
        ) = _stream_assemble_schur(
            model=model,
            tile_port_node_lists=tile_port_node_lists,
            per_tile_stats=per_tile_stats,
            extra_edges=combined_edges,
            dirichlet_nodes=model.pad_nodes,
            dirichlet_voltage=model.vdd,
            assembly_cache=_asm_cache,
        )
        # S3: Compute G-only Dirichlet RHS.  _compute_rhs_dirichlet_from_edges
        # (package resistive edges alone, as this branch used to call it
        # directly) is WRONG on its own -- it assumes tile Schur complements
        # never contribute to the unknown-Dirichlet coupling G_ud, but a
        # Dirichlet pad directly on a tile's own port list (D1 scenario) DOES
        # contribute -S_A_i[kept, pad] via that tile's own S_A_i, exactly as
        # the bulk (non-streaming) path's rhs_dirichlet_A already captures
        # (it re-embeds tile_schur_complements once, for combined_edges).
        #
        # A second _stream_assemble_schur pass can't reproduce that
        # contribution directly: the B3 streaming protocol frees each
        # tile's cached S_A after the FIRST get_schur_data_flat() read
        # (bounded worker memory is the entire point of
        # streaming_assembly=True), so the tile data is gone by the time a
        # second pass would need it -- the linearity trick above is used
        # here for exactly that reason.
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

    # Finding 12: rhs_dirichlet_G via the linearity delta above (shared by
    # both branches), instead of a second assemble_schur_complement_system
    # call in the bulk path (which used to re-scatter every tile's S_A_i a
    # second time solely to obtain this vector).
    if _cap_only_edges:
        _rhs_from_cap_only = _compute_rhs_dirichlet_from_edges(
            extra_edges=_cap_only_edges,
            unknown_list=list(interface_nodes),
            unknown_to_idx=interface_node_to_idx,
            dirichlet_nodes=model.pad_nodes,
            dirichlet_voltage=model.vdd,
        )
        rhs_dirichlet_G = rhs_dirichlet_A - _rhs_from_cap_only
    else:
        rhs_dirichlet_G = rhs_dirichlet_A.copy()

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
        # T1 fix: build a SEPARATE per-step island-penalty RHS vector.
        # apply_island_penalty() (called above, both the cache-hit branch
        # and inside _detect_islands_dispatch) folds penalty*vdd into
        # rhs_dirichlet_A only. rhs_dirichlet_A is the A-based (G + cap)
        # Dirichlet RHS used by DC-style solves, but the transient time
        # loop (solver_td.py) never reads rhs_dirichlet_A during stepping
        # -- it uses rhs_dirichlet_G (BE: +1x, TR: +2x per the documented
        # convention). So without this vector, penalized interface-island
        # rows carry the 1e5 mS penalty on the A-diagonal but get NO
        # matching per-step forcing term, and decay from Vdd toward 0
        # within a few steps.
        #
        # Do NOT fold this into rhs_dirichlet_G either: TR scales
        # rhs_dirichlet_G by 2 each step, so folding the penalty in there
        # would double-count it under TR. Instead this vector is added to
        # global_rhs EXACTLY ONCE per step, for BOTH BE and TR, in the
        # time loop.
        _island_pen_idx = np.array(
            [interface_node_to_idx[n] for n in island_nodes], dtype=np.intp,
        )
        island_penalty_rhs = np.zeros(len(interface_nodes), dtype=np.float64)
        # Finding 6: ISLAND_PENALTY_CONDUCTANCE (pgmath.schur) is the single
        # canonical source for this magnitude -- must match
        # apply_island_penalty's penalty_conductance default, since both
        # branches above call it with default arguments.
        island_penalty_rhs[_island_pen_idx] = ISLAND_PENALTY_CONDUCTANCE * model.vdd
    else:
        island_penalty_rhs = None

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

    # 5. Build tile index maps (must be before step 6 so CG tilewise can use them)
    # AND (D1 fix, item 1) kept-position-slice tile_schur_complements -- see
    # the matching comment in _factor_dc_context.
    # B3: Reuse streaming-built maps when available (tile_schur_complements
    # is empty in that case -- nothing to slice). Finding 4: also reuse the
    # streaming-built tile_kept_port_pos (mirrors the DC path fix) instead of
    # leaving it empty.
    from .interface_iterative import kept_position_slice
    tile_kept_port_pos: Dict[Any, np.ndarray] = {}
    if _streaming_tile_index_maps_td is not None:
        tile_index_maps: Dict[Tuple[int, int], np.ndarray] = _streaming_tile_index_maps_td
        tile_kept_port_pos = _streaming_tile_kept_port_pos_td or {}
    else:
        tile_index_maps = {}
        for tid, port_list in tile_port_node_lists.items():
            idx, S_kept, kept_pos = kept_position_slice(
                tile_schur_complements[tid], port_list, interface_node_to_idx,
            )
            tile_index_maps[tid] = idx
            tile_schur_complements[tid] = S_kept
            tile_kept_port_pos[tid] = kept_pos

    # S2/S13: per-tile FULL port count -- see the matching comment in
    # _factor_dc_context.
    tile_port_count: Dict[Any, int] = {
        tid: len(node_list) for tid, node_list in tile_port_node_lists.items()
    }

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
        from .interface_iterative import build_interface_solver, resolve_matvec_mode
        # Item 8: default 'auto' (resolved below).
        _matvec_mode_setting_td = _model_settings_td.get('interface_matvec_mode', 'auto')
        # Finding 13: shared settings-reading helper (see its docstring).
        _cg_settings_td = _read_interface_cg_settings(model)
        _preconditioner_td = _cg_settings_td.preconditioner
        _cg_rtol_td = _cg_settings_td.cg_rtol
        _cg_atol_td = _cg_settings_td.cg_atol
        _cg_maxiter_td = _cg_settings_td.cg_maxiter
        _cg_strict_td = _cg_settings_td.cg_strict
        _bj_max_bytes_td = _cg_settings_td.bj_max_bytes
        _matvec_threads_td = _cg_settings_td.matvec_threads
        _matvec_dtype_td = _cg_settings_td.matvec_dtype
        _strict_dtype_rtol_td = _cg_settings_td.strict_dtype_rtol

        _has_tile_blocks_td = (not _use_streaming_td) and bool(tile_schur_complements)
        _matvec_mode_td = resolve_matvec_mode(_matvec_mode_setting_td, _has_tile_blocks_td)

        # D2 (item 2): direct-stamping S_extra.  MODE-DEPENDENT (review-
        # confirmed gap): transient stamps combined_edges = resistive +
        # C_coeff*package-cap edges -- passed here as package_edges (the
        # resistive part) + package_cap_edges/C_coeff (the capacitive part,
        # via C_pkg_uu * C_coeff, mathematically identical to stamping
        # (u, v, C_coeff*c_fF) triples directly -- see build_interface_
        # package_matrices' docstring cross-reference in _build_s_extra_direct).
        # S_extra^TD therefore depends on dt/method and is rebuilt HERE, on
        # every prepare_transient() call -- never shared with a DC instance.
        # B3: When streaming was used, tile_schur_complements is empty; fall back.
        # See module docstring for the streaming vs CG-tilewise compatibility matrix.
        _S_extra_td: Optional[sp.spmatrix] = None
        if _matvec_mode_td == 'tilewise':
            if not _has_tile_blocks_td:
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
                _S_extra_td = _build_s_extra_direct(
                    interface_node_to_idx=interface_node_to_idx,
                    n_iface=len(interface_nodes),
                    package_edges=pkg_res_edges,
                    dirichlet_nodes=model.pad_nodes,
                    island_nodes=island_nodes,
                    package_cap_edges=pkg_cap_edges,
                    C_coeff=C_coeff,
                )

        # Stage 3: island indices for the 'two_level' preconditioner's
        # coarse-space row zeroing -- independent of matvec_mode.
        _island_idx_td = _island_idx_array(island_nodes, interface_node_to_idx)

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
            matvec_threads=_matvec_threads_td,
            matvec_dtype=_matvec_dtype_td,
            strict_dtype_rtol=_strict_dtype_rtol_td,
            island_idx=_island_idx_td,
            interface_coarse_geneo_k=_cg_settings_td.geneo_k,
            interface_coarse_geneo_tol=_cg_settings_td.geneo_tol,
            interface_coarse_eps_rank=_cg_settings_td.eps_rank,
            interface_coarse_max_cols=_cg_settings_td.max_cols,
            interface_coarse_max_bytes=_cg_settings_td.max_bytes,
            two_level_base=_cg_settings_td.two_level_base,
            neumann_max_bytes=_cg_settings_td.neumann_max_bytes,
            neumann_weight=_cg_settings_td.neumann_weight,
            neumann_reg=_cg_settings_td.neumann_reg,
            progress_every=_cg_settings_td.progress_every,
            interface_coarse_apply_mode=_cg_settings_td.apply_mode,
            interface_deflated_reproject_every=_cg_settings_td.deflated_reproject_every,
            warm_start_extrapolation=_cg_settings_td.warm_start_extrapolation,
        )

        class _CGSolveResultTD:
            backend = 'cg'
            backend_info = (
                f"CG/{_cg_solver_td.matvec_mode}/"
                f"precond={_cg_solver_td.preconditioner_label}"
            )
            resolved_mode = 'cg'
            solve = _cg_solve_callable_td

        interface_lu_result = _CGSolveResultTD()
        if verbose:
            logger.info(
                "Transient interface CG solver: mode=%s, precond=%s, rtol=%.2e, n=%d",
                _matvec_mode_td, _cg_solver_td.preconditioner_label, _cg_rtol_td, len(interface_nodes),
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
    # T1 fix: separate per-step island-penalty RHS (see the block above
    # where it is built); solve_transient adds it once per step.
    ctx.island_penalty_rhs = island_penalty_rhs
    ctx._tile_index_maps = tile_index_maps
    # D1 fix (item 1): stored for parity with the DC context; consumed by
    # solve_transient's D1 RHS-scatter filtering (S2 fix), and also feeds a
    # shared topology's tile_kept_port_pos (below) whichever mode factors
    # first, matching the existing tile_index_maps pattern.
    ctx._tile_kept_port_pos = tile_kept_port_pos
    # S2/S13: parity with tile_kept_port_pos.
    ctx._tile_port_count = tile_port_count
    ctx._removed_interface_nodes = island_nodes
    ctx._S_global = S_global
    # S7 (TD extension, mirrors _factor_dc_context): this IS the normal
    # (S_global-assembling) factor path -- reset the never-assemble flag so
    # a context previously factored with interface_drop_s_global=True, then
    # re-factored (not merely refactor()'d) with the setting off, can
    # save() again.
    ctx._s_global_dropped = False
    ctx.C_package_uu = C_pkg_uu if C_pkg_uu.nnz > 0 else None
    ctx._G_package_uu = G_pkg_uu if G_pkg_uu.nnz > 0 else None
    ctx.timings = timings
    # B2: store resolved interface solver mode and optional CG solver.
    # The transient time loop uses warm-start from v_gamma_old; the CG solver
    # retains the last solution as x0 automatically via InterfaceCGSolver.
    ctx._interface_solver_mode = _iface_resolved_mode_td
    # Finding 6: close the outgoing CG solver (if factor() is being re-run
    # on an already-factored context) before replacing it.
    _close_existing_cg_solver(ctx)
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
            tile_kept_port_pos=tile_kept_port_pos,  # D1 fix: persisted for save/load
            tile_port_count=tile_port_count,  # S2/S13: persisted for save/load
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

    if getattr(ctx, '_s_global_dropped', False):
        raise RuntimeError(
            "Cannot save: this transient context was factored with "
            "interface_drop_s_global=True, which never assembles S_global^TD "
            "at all (not merely 'assemble then free') -- there is nothing "
            "for save() to persist that would let load()+refactor() rebuild "
            "the interface solve without workers.  Options: (1) don't call "
            "save() for never-assemble contexts -- workers already hold "
            "everything needed, so refactor() alone (no save/load) rebuilds "
            "tilewise via a streaming re-gather; (2) re-factor with "
            "interface_drop_s_global=False if a checkpoint is required."
        )

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
        # T1 fix: persist the separate per-step island-penalty RHS.
        'island_penalty_rhs': ctx.island_penalty_rhs,
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
    # T1 fix: pre-fix checkpoints lack this key -- default to None, which
    # solve_transient treats as "no island-penalty RHS to add" (matches
    # such a checkpoint's islands, if any, being a pre-existing correctness
    # gap in the saved rhs_dirichlet_A/G already; refactor() cannot recover
    # it without re-running factor()).
    ctx.island_penalty_rhs = data.get('island_penalty_rhs')
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
    For CG mode: reconstructs the InterfaceCGSolver (assembled matvec only,
      unless this context was never-assembled -- see below).

    Workers must already be factored before calling this.

    Item 3 (TD extension): a context factored via ``interface_drop_s_global``
    never has ``ctx._S_global`` (by design).  Such a context CAN still be
    refactored, but only by re-gathering S_A_i from already-factored workers
    (the tilewise branch below); there is no saved-S_global fallback
    available for it, and switching to 'direct' mode is impossible without a
    full ``factor()`` -- mirrors ``_refactor_dc_context``'s Item 3 handling
    exactly.
    """
    _workers_attached_early = bool(ctx.model is not None and ctx.model.workers)
    _was_never_assembled = getattr(ctx, '_s_global_dropped', False)
    if ctx._S_global is None and not (_was_never_assembled and _workers_attached_early):
        if _was_never_assembled:
            # Finding 9: for a never-assembled context, "load a checkpoint
            # that includes S_global" is a dead end by construction --
            # save() unconditionally refuses to persist a never-assembled
            # context (see its error message), so no such checkpoint can
            # ever exist. Tell the user the actual recovery path directly
            # instead of pointing at an impossible route.
            raise RuntimeError(
                "Cannot refactor: this transient context was factored with "
                "interface_drop_s_global=True (S_global^TD was never "
                "assembled), and its workers are not attached "
                "(ctx.model is None or ctx.model.workers is empty) -- "
                "refactor()'s only recovery path for a never-assembled "
                "context (a streaming re-gather from already-factored "
                "workers) is unavailable. There is NO checkpoint route: "
                "save() refuses to persist a never-assembled context. The "
                "only way to recover is a full prepare_transient()/"
                "factor() call with live, attached workers."
            )
        raise RuntimeError(
            "Cannot refactor without S_global. Use factor() for a "
            "full factorization, or load a checkpoint that includes "
            "S_global."
        )

    _mode = getattr(ctx, '_interface_solver_mode', 'direct')
    _model_setting = _get_interface_solver_setting(ctx.model)
    if _model_setting != 'auto':
        _mode = _model_setting

    if _mode == 'direct' and ctx._S_global is None:
        raise RuntimeError(
            "Cannot refactor to interface_solver='direct': S_global was "
            "never assembled for this context (interface_drop_s_global=True "
            "at factor() time). Call factor() for a full re-factorization, "
            "or keep interface_solver='cg' for this refactor()."
        )

    coord_config = ctx.model.coordinator_solver_config if ctx.model is not None else None
    t0 = _time.perf_counter()
    # Finding 7 (round 2 review): tracks whether the CG branch's tilewise
    # re-gather actually ran (factor_transient_system() re-run on every
    # attached worker) vs a rebuild from saved S_global -- read at the end
    # of this function to log the accurate completion message. Stays False
    # for the direct-mode branch (always "from saved S_global") and the
    # CG branch's 'assembled' sub-case.
    _did_tilewise_regather = False

    # S12: close() the OUTGOING CG solver's persistent thread pool before
    # replacing ctx._cg_solver below -- mirror release()'s handling (the
    # lifecycle-parity checklist).  Without this, a session that alternates
    # factor()/solve/refactor() without release() accumulates one live
    # thread pool per refactor() call, reclaimed only by the weakref.
    # finalize GC safety net (not guaranteed timing).
    _old_cg_solver = getattr(ctx, '_cg_solver', None)
    if _old_cg_solver is not None:
        _close = getattr(_old_cg_solver, 'close', None)
        if callable(_close):
            _close()

    if _mode == 'direct':
        from pgmath.factor import _factor_conductance_matrix
        interface_lu_result = _factor_conductance_matrix(
            ctx._S_global, verbose=verbose, config=coord_config,
        )
        ctx._interface_lu = interface_lu_result.solve
        ctx._cg_solver = None
        ctx._interface_solver_mode = 'direct'
    else:
        # CG mode -- routed through build_interface_solver (Stage 1d).
        from .interface_iterative import build_interface_solver
        # Finding 13: shared settings-reading helper (see its docstring).
        # _model_settings (raw dict) is kept alive below for the
        # interface_matvec_mode read, which this helper deliberately
        # excludes.
        _model_settings = getattr(ctx.model, 'settings', {}) if ctx.model is not None else {}
        _cg_settings = _read_interface_cg_settings(ctx.model)
        _cg_rtol = _cg_settings.cg_rtol
        _cg_atol = _cg_settings.cg_atol
        _cg_maxiter = _cg_settings.cg_maxiter
        _cg_strict = _cg_settings.cg_strict
        _preconditioner = _cg_settings.preconditioner
        _bj_max_bytes = _cg_settings.bj_max_bytes
        _matvec_threads = _cg_settings.matvec_threads
        _matvec_dtype = _cg_settings.matvec_dtype
        _strict_dtype_rtol = _cg_settings.strict_dtype_rtol

        _topology_tile_index_maps = (
            getattr(ctx.topology, 'tile_index_maps', None)
            if ctx.topology is not None else None
        )

        # Item 9: same tilewise-rebuild-via-streaming-re-gather as the DC
        # refactor path, using the TRANSIENT Schur (A = G + C_coeff*C) via
        # factor_transient_system(dt_scaled, method) instead of the DC Schur.
        _matvec_mode_setting = _model_settings.get('interface_matvec_mode', 'auto')
        _workers_attached = bool(ctx.model is not None and ctx.model.workers)
        # Item 3 (TD extension): a never-assembled context has no
        # 'assembled' fallback at all (no S_global) -- force tilewise
        # regardless of the matvec_mode setting, mirroring
        # _refactor_dc_context's identical guard.  The top-of-function guard
        # already ensured workers are attached in this case, else it raised.
        _want_tilewise = (
            _was_never_assembled
            or _matvec_mode_setting == 'tilewise'
            or (_matvec_mode_setting in (None, 'auto') and _workers_attached)
        )
        # Finding 5 (transient twin): see the matching comment in
        # _refactor_dc_context -- 'auto' resolving to tilewise whenever
        # workers are merely ATTACHED (not verified to be FACTORED) can
        # silently re-run the full per-worker interior factorization +
        # dense transient Schur inside what is documented as a cheap
        # coordinator-only rebuild.
        #
        # Finding 1 (round 2): PREVIOUSLY this warning was suppressed for a
        # never-assembled context (`... and not _was_never_assembled`) on
        # the premise that "tilewise there is intentional/unconditional,
        # not an implicit 'auto' surprise" -- but that premise misses the
        # actual hazard: regardless of WHY tilewise was chosen, this
        # re-gather unconditionally re-factors EVERY attached worker's
        # transient block system at THIS context's (dt_scaled,
        # integration_method), silently clobbering the worker-side
        # transient factorization any OTHER live DistributedTransientContext
        # for this model (prepared at a different dt/method) depends on --
        # and for a never-assembled context this can ALSO silently override
        # an explicit interface_matvec_mode='assembled' setting, since a
        # never-assembled context has no 'assembled' fallback at all. The
        # warning is now un-suppressed for the never-assembled branch (it
        # already fires for the assembled-path implicit-'auto' case).
        # solve_transient() (solver_td.py) independently guards the actual
        # silent-wrong-voltage consequence via a worker dt/method stamp
        # check (raises RuntimeError on mismatch, converting the
        # pre-existing silent hazard into a loud error) -- this warning is
        # the proactive heads-up at refactor() time; the RuntimeError is the
        # backstop at solve_transient() time.
        if (
            _want_tilewise and _workers_attached
            and (_was_never_assembled or _matvec_mode_setting in (None, 'auto'))
        ):
            _extra_never_assembled_note = ""
            if _was_never_assembled:
                _extra_never_assembled_note = (
                    " This context was factored with "
                    "interface_drop_s_global=True: it has no 'assembled' "
                    "fallback at all, so tilewise is forced unconditionally"
                )
                if _matvec_mode_setting == 'assembled':
                    _extra_never_assembled_note += (
                        ", OVERRIDING the explicit "
                        "interface_matvec_mode='assembled' setting"
                    )
                _extra_never_assembled_note += "."
            # Finding 6 (round 2 review): restore the actionable remedy for
            # the assembled-context-under-'auto' case -- explicitly setting
            # interface_matvec_mode='assembled' is still the only way to
            # avoid this full per-worker re-factorization on every
            # refactor() call for such a context (it has no equivalent
            # remedy: a never-assembled context has no 'assembled' fallback
            # to opt into at all, so this sentence is omitted there).
            _assembled_remedy_note = (
                "" if _was_never_assembled else (
                    " Set interface_matvec_mode='assembled' explicitly to "
                    "force the cheap S_global-only rebuild instead."
                )
            )
            logger.warning(
                "Refactor: %s -- this re-gather calls "
                "factor_transient_system(dt_scaled=%.6g, method=%r) on "
                "EVERY attached worker (full interior factorization + "
                "dense transient Schur), NOT a cheap coordinator-only "
                "rebuild. This SILENTLY INVALIDATES the worker-side "
                "transient factorization of any OTHER live "
                "DistributedTransientContext for this model prepared at a "
                "different (dt, method) -- a subsequent solve_transient() "
                "on that other context now raises a loud RuntimeError "
                "(dt/method stamp mismatch) instead of silently solving "
                "with mismatched coordinator/worker state.%s If workers "
                "were already factored at this dt/method this session, "
                "this repeats that work for no benefit; if they were not, "
                "refactor()'s documented precondition (workers already "
                "factored) was not actually met.%s",
                (
                    "interface_matvec_mode='auto' (default) resolved to "
                    "'tilewise' because workers are attached"
                    if not _was_never_assembled else
                    "this is a never-assembled (interface_drop_s_global=True) "
                    "context's refactor()"
                ),
                ctx.dt_scaled, ctx.integration_method,
                _extra_never_assembled_note,
                _assembled_remedy_note,
            )

        # S5: mirror factor()'s streaming-assembly guard -- see the matching
        # comment in _refactor_dc_context.  Excluded for a never-assembled
        # context (mirrors the DC guard's `not _was_never_assembled`): such
        # a context has no 'assembled' fallback to downgrade to, so the
        # streaming_assembly OOM-avoidance this guard exists for is moot --
        # the never-assemble path's own streaming re-gather already provides
        # the same memory bound regardless of the model's streaming_assembly
        # setting.
        if _want_tilewise and not _was_never_assembled and _workers_attached:
            _streaming_setting_td_rf = _get_streaming_assembly_setting(ctx.model)
            if _streaming_setting_td_rf == 'auto':
                _size_stats_raw_td_rf = ctx.model.backend.call_all(
                    ctx.model.workers, 'get_schur_size_stats'
                )
                _refactor_would_stream_td = _should_stream(
                    ctx.model, _size_stats_raw_td_rf,
                )
            else:
                _refactor_would_stream_td = bool(_streaming_setting_td_rf)
            if _refactor_would_stream_td:
                logger.warning(
                    "Refactor: interface_matvec_mode=%r requests tilewise, "
                    "but streaming_assembly is active for this model -- "
                    "re-gathering dense per-tile S_A_i to the coordinator "
                    "would defeat the memory bound streaming_assembly=True "
                    "provides. Falling back to 'assembled'. Call factor() "
                    "with streaming_assembly disabled for a full "
                    "re-factorization if tilewise is required.",
                    _matvec_mode_setting,
                )
                _want_tilewise = False

        _tile_schur_complements: Optional[Dict[Any, np.ndarray]] = None
        _tile_index_maps = _topology_tile_index_maps
        _S_extra: Optional[sp.spmatrix] = None
        _matvec_mode = 'assembled'
        # Stage 3: hoisted OUT of the `if _want_tilewise` branch below (see
        # the matching comment in _refactor_dc_context) -- island indices
        # are needed regardless of resolved matvec mode. S1: use this
        # context's own TD-mode island set (topology.island_nodes_td), not
        # the shared first-mode-wins topology.removed_interface_nodes.
        _interface_node_to_idx = ctx.interface_node_to_idx
        _island_nodes = (
            getattr(ctx.topology, 'island_nodes_td', None)
            if ctx.topology is not None else None
        )
        if _island_nodes is None:
            _island_nodes = ctx._removed_interface_nodes or set()
        _island_idx = _island_idx_array(_island_nodes, _interface_node_to_idx)

        if _want_tilewise and _workers_attached:
            # Finding 7 (round 2 review): remember that this branch actually
            # re-ran factor_transient_system() on the workers (a tilewise
            # re-gather, NOT a rebuild from saved S_global) -- read at the
            # end of this function to log the correct completion message.
            _did_tilewise_regather = True
            _fresh_port_count: Dict[Any, int] = {}
            # Finding 1 (round 2 review): invalidate BEFORE dispatching the
            # re-gather's factor_transient_system() RPCs -- see
            # _invalidate_worker_transient_stamp's docstring. A failure
            # partway through this re-gather must leave the stamp at None,
            # not the stale pre-refactor value (which could describe a mix
            # of tiles re-factored at THIS context's dt/method and tiles
            # still at whatever the previous stamp claimed).
            _invalidate_worker_transient_stamp(ctx.model)
            (
                _tile_schur_complements, _fresh_tile_index_maps, _fresh_kept_pos,
                _rhs_dirichlet_from_tiles_rf,
            ) = _gather_kept_tile_schur_streaming(
                ctx.model, _interface_node_to_idx,
                transient_dt_scaled=ctx.dt_scaled,
                transient_method=ctx.integration_method,
                out_port_count=_fresh_port_count,
                # Major finding (review round 2): a never-assembled
                # context's rhs_dirichlet_A/_rhs_dirichlet_G were cleared
                # by release() (result.py) -- unlike an assembled context,
                # which reuses its already-fixed rhs_dirichlet_A from the
                # original factor() call (or a load()'d checkpoint), a
                # never-assembled context has no other refactor-time
                # source for the tile-embedded ("D1") Dirichlet
                # contribution. Request it from this same streaming
                # re-gather instead of a second round-trip. None for an
                # assembled context (unused there -- see the docstring).
                dirichlet_voltage=ctx.model.vdd if _was_never_assembled else None,
                n_iface=len(_interface_node_to_idx) if _was_never_assembled else None,
            )
            # Finding 1: this re-gather just re-ran
            # factor_transient_system(dt_scaled, method) on every attached
            # worker at THIS context's dt/method -- stamp the model (see the
            # un-suppressed warning right above this branch and
            # _stamp_worker_transient_factor's docstring for why this can
            # invalidate OTHER live transient contexts' worker-side state).
            # Only reached if the re-gather above completed without raising.
            _stamp_worker_transient_factor(
                ctx.model, ctx.dt_scaled, ctx.integration_method,
            )
            _tile_index_maps = _fresh_tile_index_maps
            # S14: keep the D1-consistent pair in sync (see the matching
            # comment in _refactor_dc_context).
            ctx._tile_index_maps = _fresh_tile_index_maps
            ctx._tile_kept_port_pos = _fresh_kept_pos
            # S2/S13: parity with _tile_kept_port_pos.
            ctx._tile_port_count = _fresh_port_count
            # Finding 12: always request the (G_pkg_uu, C_pkg_uu) pair this
            # already stamps internally -- free when not needed (the
            # assembled-context tilewise-refactor branch below just
            # discards it), and removes the second
            # build_interface_package_matrices() pass the never-assembled
            # branch used to make with IDENTICAL arguments.
            _S_extra, (_G_pkg_uu_rf, _C_pkg_uu_rf) = _build_s_extra_direct(
                interface_node_to_idx=_interface_node_to_idx,
                n_iface=len(_interface_node_to_idx),
                package_edges=ctx.model.package_data.package_edges,
                dirichlet_nodes=ctx.model.pad_nodes,
                island_nodes=_island_nodes,
                package_cap_edges=ctx.model.package_data.package_cap_edges,
                C_coeff=ctx.C_coeff,
                return_package_matrices=True,
            )
            _matvec_mode = 'tilewise'
            if _was_never_assembled:
                # Major finding (review round 1): a never-assembled context
                # has no save()/load() route to persist C_package_uu /
                # _G_package_uu / island_penalty_rhs -- release() clears all
                # three to None (result.py) and this tilewise re-gather is
                # its ONLY refactor path (unlike an assembled context, which
                # only ever reaches refactor() with S_global intact or via
                # load(), both of which already carry these vectors).
                # Without rebuilding them here, a post-refactor
                # solve_transient silently drops the package-cap history
                # term, the TR package-G term, and the island-penalty
                # forcing term (solver_td.py's time loop only guards each
                # with `if ... is not None`, so the omission is silent, not
                # an error). Rebuild identically to _factor_transient_context
                # (~:3581 for the package matrices, ~:3566-3573 for the
                # penalty vector) so a refactored context is numerically
                # identical to the original factor().
                ctx.C_package_uu = _C_pkg_uu_rf if _C_pkg_uu_rf.nnz > 0 else None
                ctx._G_package_uu = _G_pkg_uu_rf if _G_pkg_uu_rf.nnz > 0 else None

                # Major finding (review round 2): release() (result.py)
                # clears BOTH ctx.rhs_dirichlet_A and ctx._rhs_dirichlet_G
                # to None, and unlike C_package_uu/_G_package_uu/
                # island_penalty_rhs above, nothing below used to rebuild
                # them -- the rhs_dirichlet_G property (result.py) then
                # silently fell back to ctx.topology.rhs_dirichlet_G, which
                # is only correct for a TD-created topology. In the
                # documented DC-first lifecycle (CLAUDE.md: dc_ctx =
                # solver.prepare(); prepare_transient() shares that same
                # topology object), the fallback instead returns the DC
                # context's rhs_dirichlet_G -- the DC never-assemble value
                # already has the island penalty folded in (G-based, not
                # A-based) -- so the time loop's separate
                # `+island_penalty_rhs` term (solver_td.py) would double
                # -count the penalty, and any package-cap Dirichlet term
                # would be silently wrong. Rebuild EXACTLY as
                # _factor_transient_context_no_s_global does: package-edges
                # -only contribution (from combined_edges, the resistive +
                # C_coeff-weighted-cap set fixing this same ordering) --
                # computed here (refactor must NOT re-derive the
                # already-fixed interface ordering, unlike the factor path's
                # _compute_interface_ordering call) -- plus the
                # tile-embedded ("D1") contribution just re-gathered above.
                # Finding 7: the linearity delta + penalty-ordering
                # derivation itself is shared with
                # _factor_transient_context_no_s_global via
                # _derive_td_dirichlet_vectors (see its docstring for why
                # the ordering matters).
                _pkg_res_edges_rf = ctx.model.package_data.package_edges
                _pkg_cap_edges_rf = ctx.model.package_data.package_cap_edges
                # Finding 9: shared combined/cap-only edge construction (see
                # _build_td_package_edge_sets's docstring).
                _combined_edges_rf, _cap_only_edges_rf = _build_td_package_edge_sets(
                    _pkg_res_edges_rf, _pkg_cap_edges_rf, ctx.C_coeff,
                )
                _interface_nodes_rf = ctx.interface_nodes
                _rhs_dirichlet_pkg_rf = _compute_rhs_dirichlet_from_edges(
                    extra_edges=_combined_edges_rf,
                    unknown_list=_interface_nodes_rf,
                    unknown_to_idx=_interface_node_to_idx,
                    dirichlet_nodes=ctx.model.pad_nodes,
                    dirichlet_voltage=ctx.model.vdd,
                )
                (
                    _rhs_dirichlet_A_rf, _rhs_dirichlet_G_rf, _island_penalty_rhs_rf,
                ) = _derive_td_dirichlet_vectors(
                    model=ctx.model,
                    interface_nodes=_interface_nodes_rf,
                    interface_node_to_idx=_interface_node_to_idx,
                    rhs_dirichlet_pkg=_rhs_dirichlet_pkg_rf,
                    rhs_dirichlet_from_tiles=_rhs_dirichlet_from_tiles_rf,
                    pkg_cap_edges=_pkg_cap_edges_rf,
                    C_coeff=ctx.C_coeff,
                    island_nodes=_island_nodes,
                )
                ctx.island_penalty_rhs = _island_penalty_rhs_rf
                ctx.rhs_dirichlet_A = _rhs_dirichlet_A_rf
                ctx._rhs_dirichlet_G = _rhs_dirichlet_G_rf
        elif _matvec_mode_setting == 'tilewise' and not _workers_attached:
            logger.warning(
                "Refactor: interface_matvec_mode='tilewise' requested but no "
                "workers are attached to re-gather S_i (they must already be "
                "factored -- refactor()'s documented precondition). Falling "
                "back to 'assembled'. Call factor() for a full "
                "re-factorization to restore tilewise."
            )

        # Finding 6 (round 2) / Finding 11: shared helper -- see the
        # matching comment/docstring in _refactor_dc_context.
        _warn_if_coarse_space_lost(
            _preconditioner, _matvec_mode, _want_tilewise,
            _matvec_mode_setting, _workers_attached,
        )

        # Finding 1 (round 1) / Finding 11: shared helper -- see the
        # matching comment in _refactor_dc_context.
        _preconditioner = _degrade_preconditioner_if_no_tile_index_maps(
            _preconditioner, _tile_index_maps,
        )
        _cg_stats: Dict[str, Any] = {}
        _solve_callable, _resolved_backend, cg_solver = build_interface_solver(
            S_global=ctx._S_global,
            interface_solver='cg',
            matvec_mode=_matvec_mode,
            tile_schur_complements=_tile_schur_complements,
            tile_index_maps=_tile_index_maps,
            S_extra=_S_extra,
            # Item 3 (TD extension): S_global is None for a never-assembled
            # context -- build_interface_solver requires n_interface
            # explicitly in that case (mirrors _refactor_dc_context).
            n_interface=len(ctx.interface_node_to_idx) if ctx._S_global is None else None,
            island_idx=_island_idx,
            interface_coarse_geneo_k=_cg_settings.geneo_k,
            interface_coarse_geneo_tol=_cg_settings.geneo_tol,
            interface_coarse_eps_rank=_cg_settings.eps_rank,
            interface_coarse_max_cols=_cg_settings.max_cols,
            interface_coarse_max_bytes=_cg_settings.max_bytes,
            two_level_base=_cg_settings.two_level_base,
            neumann_max_bytes=_cg_settings.neumann_max_bytes,
            neumann_weight=_cg_settings.neumann_weight,
            neumann_reg=_cg_settings.neumann_reg,
            progress_every=_cg_settings.progress_every,
            interface_coarse_apply_mode=_cg_settings.apply_mode,
            interface_deflated_reproject_every=_cg_settings.deflated_reproject_every,
            warm_start_extrapolation=_cg_settings.warm_start_extrapolation,
            preconditioner=_preconditioner,
            rtol=_cg_rtol,
            atol=_cg_atol,
            maxiter=_cg_maxiter,
            strict=_cg_strict,
            block_jacobi_max_bytes=_bj_max_bytes,
            verbose=verbose,
            cg_stats_dict=_cg_stats,
            matvec_threads=_matvec_threads,
            matvec_dtype=_matvec_dtype,
            strict_dtype_rtol=_strict_dtype_rtol,
        )
        ctx._interface_lu = _solve_callable
        ctx._cg_solver = cg_solver
        ctx._interface_solver_mode = 'cg'
        # Item 3 (TD extension): still true after a successful tilewise
        # re-gather refactor (S_global remains unassembled); a downgrade to
        # 'assembled' would have raised above (no fallback exists for a
        # never-assembled context), so reaching here with
        # _was_never_assembled means the re-gather succeeded. Mirrors
        # _refactor_dc_context's identical assignment.
        ctx._s_global_dropped = _was_never_assembled

    elapsed = _time.perf_counter() - t0
    ctx.is_factored = True
    # Finding 7 (round 2 review): branch the completion log -- "from saved
    # S_global" is only true when this refactor rebuilt from ctx._S_global
    # (direct mode, or the CG branch's 'assembled' matvec sub-case); the
    # tilewise re-gather sub-case (_did_tilewise_regather) re-ran
    # factor_transient_system() on every attached worker instead and never
    # touched a saved S_global (there may not even be one -- a never-
    # assembled context has none by construction), which directly
    # contradicts the "from saved S_global" wording and the loud
    # tilewise-re-gather WARNING already logged moments earlier.
    if _did_tilewise_regather:
        logger.info(
            "Refactored transient coordinator solve (%s) via tilewise "
            "re-gather from workers (factor_transient_system() re-run on "
            "every attached worker; no saved S_global was read or exists) "
            "in %.3fs", _mode, elapsed,
        )
    else:
        logger.info(
            "Refactored transient coordinator solve (%s) from saved S_global "
            "in %.3fs", _mode, elapsed,
        )

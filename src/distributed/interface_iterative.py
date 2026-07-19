"""Iterative interface solve for large-scale DDM.

Provides InterfaceCGSolver, a wrapper around scipy.sparse.linalg.cg that
replaces the direct CHOLMOD/SuperLU factorization of the global Schur
complement S_global.  Two coordinator-side matvec modes are supported:

  'assembled'  -- matvec on the assembled sparse S_global (removes the
                  dominant ~100-300 GB CHOLMOD factor; keeps ~50 GB S_global).
  'tilewise'   -- matvec = sum_i P_i^T (S_i @ (P_i @ x)) using the per-tile
                  dense S_i blocks + tile_index_maps.  Avoids global assembly
                  entirely (removes both the ~50 GB S_global and its factor).

Worker-side RPC matvec (S_i lives on the worker) is out of scope -- noted as
future work for B4.

Preconditioners (pluggable, via ``preconditioner`` keyword):
  'block_jacobi' (default) -- block-Jacobi from the owner's principal S_i
                              submatrix for each interface node.  Ownership:
                              first tile whose tile_index_map includes the node.

                              Memory note: each owned block is a k_i x k_i
                              dense Cholesky factor.  Total factor memory scales
                              as Sum(k_i^2) ≈ n^2/T for T balanced tiles.  At
                              n=1M interface nodes with T=1000 tiles this is
                              ~8 GB for factors alone — the same coordinator-
                              memory class CG is designed to avoid.  For
                              netlist_sampled (n~2K, max block 478) the factor
                              is ~5.9 MB (negligible).

                              When the estimated factor memory exceeds
                              ``BLOCK_JACOBI_MAX_FACTOR_BYTES`` (default 4 GB),
                              _build_block_jacobi() automatically falls back to
                              'jacobi' (diagonal) and logs a WARNING.  For
                              1M-interface systems with few tiles, prefer
                              'jacobi' explicitly or reduce tile size via B1
                              splitting so T is large enough that individual
                              block sizes are manageable.
  'jacobi'                  -- diagonal of S_global (assembled) or the summed
                              tile S_i diagonals (tilewise).  Cheap, scales
                              to any interface size; weaker than block_jacobi
                              but does not use O(n^2/T) memory.
  'none'                    -- identity (no preconditioning).
  'amg'                     -- algebraic multigrid via pyamg (lazy import,
                              skipped gracefully when pyamg is not installed).
  'two_level'               -- Stage 3: block_jacobi PLUS an additive coarse-
                              space correction (partition-of-unity + GenEO-
                              lite columns) -- fixes the block_jacobi CG
                              stagnation Stage 2 measured at large split
                              regimes (genuine near-null eigendirections in
                              the cho-factored ownership blocks).  See
                              ``interface_coarse.py`` and this class's
                              "Stage 3" docstring section below.  'auto' (the
                              resolved default for CG+tilewise, see
                              :func:`resolve_preconditioner`) picks this
                              automatically; small/'assembled' systems keep
                              the legacy 'block_jacobi' default.

Save / load semantics for CG mode
-----------------------------------
CG mode has no factor to save (that is the whole point).  On save():
  - S_global is stored as usual (used by assembled mode; can also be used to
    refactor into direct mode later).
  - The ``interface_solver`` setting is stored so that load() + refactor()
    reconstructs a CG solve callable, not an LU factor.
On load() + refactor() in CG mode:
  - ``refactor`` rebuilds a CG callable from S_global (for assembled mode) or
    from S_global + tile_schur_complements (for tilewise mode).
  - Per-tile S_i blocks are NOT saved (too large; always recomputed).
  - Simplest correct behaviour: after load(), callers must call factor()
    for tilewise mode (which re-runs factor_and_compute_schur on workers),
    or refactor() for assembled mode (which rebuilds the CG op from S_global).

Adjoint note
------------
``solver_adjoint.py``'s ``analyze_adjoint_static``/``analyze_adjoint`` call
``ctx.interface_lu(global_rhs)`` generically -- there is no special-casing of
CG vs direct, and no check of ``ctx._interface_solver_mode``.  This works
transparently in EVERY mode (direct, CG/assembled, CG/tilewise) because
``interface_lu`` is always a plain linear-solve callable with the same call
signature, regardless of what builds it: a direct LU factorization, an
``InterfaceCGSolver`` wrapping the assembled ``S_global``, or an
``InterfaceCGSolver`` wrapping per-tile ``S_i`` blocks via the tilewise
matvec.  Tilewise CG adjoint is not a special/unsupported case that needs
forcing to direct -- the D1 (pad-port kept-position slicing) and D2
(direct-stamped ``S_extra``) fixes make the tilewise matvec exactly correct,
so ``ctx.interface_lu`` returns the same answer as direct/assembled mode for
the adjoint's DC-shaped solves too (same coefficient matrix as the forward
DC solve).

Stage 2 -- threaded tilewise matvec + threaded block-Jacobi apply
-------------------------------------------------------------------
D1 fix (pad-port corruption)
    ``tile_index_maps`` (built by ``result_factorization.py``) filters
    Dirichlet/pad nodes out of each tile's port list, but the per-tile dense
    Schur block ``S_i`` returned by the tile worker still has ALL ports
    (including pads) as rows/columns.  Pairing the filtered map with the
    full ``S_i`` is a dimension mismatch (or, worse, silent index
    corruption if sizes happen to coincide).  The fix lives at the call
    site (``result_factorization.py``'s ``_kept_position_slice``, reused
    here as ``kept_position_slice``): each tile's ``S_i`` is sliced to
    ``S_i[np.ix_(pos, pos)]`` -- dropping exactly the rows/columns whose
    node is NOT in ``interface_node_to_idx`` -- BEFORE it ever reaches this
    class.  ``InterfaceCGSolver.__init__`` additionally asserts
    ``S_i.shape[0] == len(tile_index_maps[tid])`` per tile so any caller
    that forgets the slicing step fails loudly (a clear ``ValueError``)
    instead of a cryptic numpy broadcast error deep inside a matvec.

Threaded tilewise matvec (design + measured scatter decision)
    ``matvec_threads`` controls a **persistent** ``ThreadPoolExecutor``
    (lazy-built on first tilewise matvec/BJ-apply call, closed via
    :meth:`InterfaceCGSolver.close` or a ``weakref.finalize`` safety net).
    Work is a static LPT (longest-processing-time) partition of tiles by
    ``n_ports**2`` (proxy for per-tile GEMV cost) across
    ``matvec_threads`` bins, computed once at construction.

    Scatter design (measured on this host, `docs`/plan §Stage 2 report has
    the full numbers): each thread accumulates into a **compact buffer**
    sized to the UNION of global interface indices its assigned tiles
    touch (precomputed once), not a full ``(n_threads, n)`` array that must
    be zero-filled and reduced via ``acc.sum(axis=0)`` every call.  Stage 0
    measured the naive full-row-accumulator design INVERT above 8 threads
    (zero-fill + reduction cost growing with thread count outpacing the
    GEMV work); the compact-touched-index design was measured on this host
    to match the naive design's throughput at <=8 threads and pull ahead at
    16-32 threads (e.g. ~26.6ms vs ~31.6ms at 32 threads on a 150K-interface
    /60-tile synthetic), while being architecturally immune to the
    zero-fill-scales-with-n_threads*n failure mode.  ``matvec_threads='auto'``
    resolves to ``min(8, cpu_count, n_tiles)`` -- 8, not 32 -- because Stage 0
    measured best throughput at 8 threads on the BRCM-class proxy; the
    original Stage 2 sketch's ``min(32, ...)`` predates that measurement.

fp32 critical path (BRCM host is CPU-only)
    ``matvec_dtype='float32'`` stores each tile's ``S_i`` as float32 and
    casts the (small) gathered ``x`` slice to float32 per tile so the GEMV
    itself (``S_i @ x_local``) stays entirely in float32 and hits the BLAS
    ``sgemv`` fast path; the float32 result is then accumulated into the
    float64 running total.  A naive mixed-dtype ``S_i`` (float32) times
    ``x`` (float64) call falls OFF the BLAS fast path in numpy (silently
    promotes to a slow elementwise loop) -- Stage 0 measured this ~10x
    SLOWER than fp64, which is why both operands of the GEMV must share
    dtype.  Measured on this host (150K-interface/60-tile synthetic,
    8 threads): fp32 ~1.7-2.0x the fp64 throughput, matching the plan's "at
    least ~2x" expectation.  fp32 residual floor is ~1e-7 relative, so
    ``matvec_dtype='float32'`` is enforced to pair with ``rtol >= 1e-7``
    (``ValueError`` otherwise; override with ``strict_dtype_rtol=False``).

SPD-safe block-Jacobi fallback
    Singular/indefinite owned blocks previously fell back to
    ``np.linalg.pinv``, which can retain small NEGATIVE eigenvalues from FP
    noise on a numerically indefinite block -- silently voiding CG's
    convergence guarantee (a preconditioner must be SPD).  The fallback now
    eigendecomposes the (symmetric) block, clips eigenvalues to
    ``>= eps_rel * lambda_max``, and applies ``V @ diag(1/w_clipped) @ V.T``
    -- guaranteed PSD (in fact SPD after clipping away non-positive modes).

Stage 3 -- two-level coarse-space preconditioner ('two_level')
-------------------------------------------------------------------
``preconditioner='two_level'`` layers an additive coarse-space correction on
top of block-Jacobi: ``M^-1 = M_bj^-1 + Z S_c^+ Z^T`` (see
``interface_coarse.py`` for the full derivation/motivation -- Stage 2
measured cold block-Jacobi CG STAGNATING at the mi200k_v2 split regime due to
genuine near-null eigendirections in the cho-factored ownership blocks). The
coarse space (``Z``, partition-of-unity + GenEO-lite columns; ``S_c``, its
small eigh-factored pseudo-inverse) is built AFTER ``self._linear_op`` exists
(``_augment_with_coarse_space``, called from ``__init__`` right after
``_build_linear_op()``) because ``S Z`` uses the solver's own matmat --
building the block-Jacobi component first (as for plain ``'block_jacobi'``)
and layering the coarse term on second avoids disturbing the existing
fp64-read-before-fp32-cast ordering invariant documented above. Never
persisted (rebuilt on every ``factor()``/``refactor()``, like the per-tile
Schur blocks it derives from). If T' (PoU + GenEO columns) exceeds ``interface_coarse_max_cols``, falls
back to the PoU-only rung first (WARNING, GenEO columns dropped for this
solve -- PoU-only is a strictly smaller space and typically still fits).
Degrades further to the plain (possibly memory-budget-downgraded-to-'jacobi')
base preconditioner with a WARNING -- never raises prepare() -- only when the
coarse build itself fails outright (PoU-only T' ALSO exceeds
``interface_coarse_max_cols``, no usable ``tile_index_maps``, or ``S_c`` has
no positive spectrum).

BJ-apply perf fix (permuted-contiguous GEMV)
-------------------------------------------------------------------
Stage 2 measured the threaded block-Jacobi apply at only 1.4x speedup (701 ms
vs 990 ms serial, 64 blocks / n=167,659) despite ``cho_solve`` itself
releasing the GIL (LAPACK ``dpotrs``): the PER-BLOCK fancy-index gather
(``x[global_idx]``) and scatter (``result[global_idx] = ...``) do NOT release
the GIL and are RANDOM-access (cache-miss-bound) at a cost comparable to the
O(k^2) solve itself at these block sizes -- serializing across threads even
though the solve itself parallelizes fine. The fix: do exactly ONE gather and
ONE scatter per ``apply()`` call (not one per block) via a single global
permutation array (block-Jacobi ownership is a partition, so concatenating
every block's global index array is a valid partial permutation of
``[0, n)``), and materialize each block's dense (pseudo-)inverse ONCE at
build time (replacing -- not duplicating -- the cho factor payload, so total
memory stays the same order as before) so each block's apply is a single
contiguous-slice GEMV (BLAS-2, releases the GIL) instead of two triangular
solves plus scattered indexing. Measured on a synthetic proportional to the
mi200k_v2 regime (64 blocks, ~2674 avg block size, n~171K, 8 threads, BLAS
pinned to 1 thread throughout via ``threadpool_limits``): the OLD design's
own threading is NEGATIVE (serial ~255 ms, 8-thread ~470-490 ms -- concurrent
``cho_solve`` calls across threads contend rather than parallelize, even
with per-thread disjoint data); the fix is faster BOTH serially (~120 ms,
~2.1x -- a streaming GEMV beats two triangular-solve passes even
single-threaded) AND when threaded (~47-52 ms, a further ~2.3-2.5x over its
own serial, ~9-10x over the OLD design's threaded number). See
``_build_block_jacobi``'s ``_bj_perm``/``_bj_offsets``/``_bj_solve_threaded``
for the implementation.

Tilewise without ever assembling S_global (Finding 0 upgrade)
    ``interface_drop_s_global`` (bool, default False) changes the CG+tilewise
    factor path from "assemble S_global then optionally free it" to "never
    assemble it at all": interface node ordering + Dirichlet RHS are derived
    from tile PORT NAME LISTS alone (no S_i values), per-tile dense Schur
    blocks are gathered via the streaming (one-tile-in-flight)
    ``call_all_streaming`` protocol with kept-position slicing applied
    immediately per tile (peak coordinator memory never includes an
    unsliced full-S_i-sum or an assembled S_global CSR), and S_extra is
    built by direct stamping (D2, below) rather than a giant
    S_global-minus-sum subtraction.  Requires
    ``model.island_detection_mode == 'summaries'`` (the union-find island
    detector, which never touches S_global) -- with the legacy Schur-BFS
    detector this setting falls back to the normal assembled path with a
    WARNING, since that BFS fundamentally needs S_global's nonzero
    structure.  ``save()`` raises with guidance when S_global was never
    assembled; ``refactor()`` re-gathers via the same streaming protocol.
"""

from __future__ import annotations

import logging
import os
import threading
import time
import weakref
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import threadpoolctl

from . import interface_coarse

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stage 2: threaded matvec/BJ-apply thread-count resolution.
#
# Stage 0 microbenchmark (docs §7.7 / plan report) measured the threaded
# tilewise matvec on the BRCM-class proxy: best throughput at 8 threads,
# INVERTED scaling above 8 (per-thread accumulator zero-fill + final
# reduction growing with thread count outpaces the GEMV work per thread on
# this host's memory subsystem).  8, not 32, is therefore the 'auto' cap.
# ---------------------------------------------------------------------------
DEFAULT_MATVEC_THREADS_CAP: int = 8

# fp32 tilewise matvec residual floor (Stage 0 empirical + this host's
# fp32-vs-fp64 comparison): pairing matvec_dtype='float32' with a CG rtol
# tighter than this is not meaningful (the matvec itself introduces ~1e-7
# relative error) and risks CG failing to converge to its requested
# tolerance.  Enforced in InterfaceCGSolver.__init__ unless
# strict_dtype_rtol=False.
FP32_MATVEC_MIN_RTOL: float = 1e-7

# ---------------------------------------------------------------------------
# Threshold for auto-select: use CG when n_interface >= this value.
# Overridable via monkeypatch in tests.
# ---------------------------------------------------------------------------
AUTO_CG_N_INTERFACE_THRESHOLD: int = 200_000

# ---------------------------------------------------------------------------
# Sentinel: factor-memory budget (bytes) for auto-select.
# CHOLMOD supernodal factor is roughly 5-10x S_global memory.
# At 50 GB S_global (138K interface) CHOLMOD factor = 100-300 GB.
# We use a 32 GB budget as the auto cutover (conservative for coordinator).
#
# This module-level constant is the FALLBACK used when no settings dict is
# available (e.g. direct calls to auto_select_interface_solver() without an
# explicit factor_memory_budget_bytes).  When a model.settings dict IS
# available, the ``interface_factor_memory_budget`` setting (Stage 1c)
# resolves to ``min(AUTO_CG_FACTOR_MEMORY_BUDGET_BYTES, 0.4 * total_RAM)``
# via ``resolve_factor_memory_budget_bytes()`` and is passed explicitly by
# callers (result_factorization.py).  See that function for the host-aware
# 'auto' default.
# ---------------------------------------------------------------------------
AUTO_CG_FACTOR_MEMORY_BUDGET_BYTES: int = 32 * 1024 ** 3  # 32 GB

# ---------------------------------------------------------------------------
# Block-Jacobi preconditioner memory budget.
#
# Each owned block is a k_i x k_i dense float64 Cholesky factor.  Total
# factor memory scales as Sum(k_i^2) = n^2/T for T balanced tiles.
# At n=1M interface nodes with T=1000 tiles: ~8 GB just for factors.
# With fewer or larger tiles the cost grows quickly into the coordinator-
# memory class that CG is designed to avoid.
#
# When the estimated block-Jacobi factor memory exceeds this threshold,
# _build_block_jacobi() falls back to 'jacobi' (diagonal) automatically
# and logs a WARNING.  Override via monkeypatch in tests.
#
# Default 4 GB is conservative: keeps the preconditioner build out of the
# multi-GB danger zone while still allowing block_jacobi on all realistic
# netlist_sampled / BRCM-class systems where n_interface is 2-140K.
# (Measured 5.9 MB on netlist_sampled 2015-interface max-block-478 system.)
#
# This module-level constant is the FALLBACK used by InterfaceCGSolver
# instances that don't receive an explicit ``block_jacobi_max_bytes``
# constructor argument (preserves legacy/monkeypatch-based test behaviour).
# When a model.settings dict IS available, the
# ``interface_block_jacobi_max_bytes`` setting (Stage 1c) resolves to
# ``min(8 GB, 0.1 * total_RAM)`` via ``resolve_block_jacobi_max_bytes()``
# and is passed explicitly through the ``block_jacobi_max_bytes`` constructor
# argument by callers (result_factorization.py).
# ---------------------------------------------------------------------------
BLOCK_JACOBI_MAX_FACTOR_BYTES: int = 4 * 1024 ** 3  # 4 GB

# Host-aware 'auto' caps (Stage 1c) — independent of the legacy module
# constants above so that changing these does not alter the fallback path
# monkeypatched by existing tests.
_AUTO_FACTOR_MEMORY_BUDGET_CAP_BYTES: int = 32 * 1024 ** 3   # 32 GB
_AUTO_FACTOR_MEMORY_BUDGET_FRACTION: float = 0.4              # of total RAM
_AUTO_BLOCK_JACOBI_MAX_BYTES_CAP: int = 8 * 1024 ** 3          # 8 GB
_AUTO_BLOCK_JACOBI_MAX_BYTES_FRACTION: float = 0.1             # of total RAM

# Finding 5 (Stage 3): host-aware 'auto' cap/fraction for
# interface_coarse_max_bytes -- same order as the block-Jacobi budget above
# (both guard coordinator-resident dense arrays), no legacy floor (brand
# new setting, no pre-existing fixed constant to never regress below).
_AUTO_COARSE_MAX_BYTES_CAP: int = 8 * 1024 ** 3                # 8 GB
_AUTO_COARSE_MAX_BYTES_FRACTION: float = 0.1                   # of total RAM
# Finding 9 (round 2): NO module-level COARSE_MAX_BYTES_DEFAULT constant
# here -- a prior version snapshotted interface_coarse.DEFAULT_MAX_BYTES at
# IMPORT time (a plain module-level assignment), which defeats
# `monkeypatch.setattr(interface_coarse, 'DEFAULT_MAX_BYTES', ...)` despite
# a comment on the old constant claiming the opposite ("read dynamically so
# test monkeypatches keep working" -- it did not).  The fallback used when
# InterfaceCGSolver receives no explicit interface_coarse_max_bytes now
# reads ``interface_coarse.DEFAULT_MAX_BYTES`` directly at the point of use
# (``_augment_with_coarse_space``, below) -- genuinely dynamic, mirroring
# BLOCK_JACOBI_MAX_FACTOR_BYTES's role for block_jacobi (that constant IS
# safe to snapshot at module level because it is defined in THIS module, so
# tests monkeypatch it directly with no import-time-copy indirection).


_warned_no_psutil = False


def _get_total_ram_bytes() -> int:
    """Return total system RAM in bytes via psutil.

    Falls back to a conservative 64 GB assumption (logging a WARNING once)
    if psutil is unavailable, so host-aware 'auto' sizing degrades
    gracefully rather than crashing.
    """
    global _warned_no_psutil
    try:
        import psutil  # lazy import; declared in pyproject.toml [project] deps
        return int(psutil.virtual_memory().total)
    except Exception:
        if _warned_no_psutil:
            return 64 * 1024 ** 3
        _warned_no_psutil = True
        logger.warning(
            "psutil unavailable; falling back to a conservative 64 GB RAM "
            "assumption for interface-solver memory-budget auto-sizing. "
            "Install psutil or set interface_factor_memory_budget / "
            "interface_block_jacobi_max_bytes explicitly."
        )
        return 64 * 1024 ** 3


def _resolve_memory_budget_bytes(
    setting: Any,
    auto_cap_bytes: int,
    auto_fraction: float,
    floor_bytes: int = 0,
) -> int:
    """Resolve a 'auto' | int | float | numeric-string memory-budget setting.

    'auto' (or None) -> max(floor_bytes, min(auto_cap_bytes,
    auto_fraction * total_RAM_bytes)) via psutil.  ``floor_bytes`` lets a
    caller enforce a legacy minimum so the host-aware auto formula never
    regresses below a previously-fixed constant (finding 9: the block-Jacobi
    budget must never auto-resolve below the legacy 4 GB floor).

    Any other value is coerced via ``int(float(setting))`` so plausible
    numeric strings (including exponent notation like '2e9', which PyYAML
    1.1 parses as a *string* rather than a float) are accepted the same way
    an actual float would be.  Bools are rejected explicitly BEFORE this
    coercion: Python's ``bool`` is a subclass of ``int``, so
    ``int(float(True))`` would silently resolve to 1 byte instead of
    raising -- a classic YAML-1.1 pitfall (``interface_factor_memory_budget:
    yes`` parses to Python ``True``).
    """
    if setting is None or (
        isinstance(setting, str) and setting.strip().lower() == 'auto'
    ):
        total_ram = _get_total_ram_bytes()
        return max(floor_bytes, min(auto_cap_bytes, int(auto_fraction * total_ram)))
    if isinstance(setting, bool):
        raise ValueError(
            f"Invalid memory-budget setting {setting!r}: expected 'auto' or "
            f"an integer byte count (bool is not a valid byte count)."
        )
    try:
        # Finding 8: a YAML '.inf' value (PyYAML parses to float('inf'), a
        # plausible way to spell "unlimited budget") makes int(float(...))
        # raise OverflowError ("cannot convert float infinity to integer"),
        # not TypeError/ValueError -- must be caught here too, or prepare()
        # crashes with a raw traceback instead of this actionable message.
        return int(float(setting))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"Invalid memory-budget setting {setting!r}: expected 'auto' or "
            f"an integer byte count."
        ) from exc


def resolve_factor_memory_budget_bytes(setting: Any = 'auto') -> int:
    """Resolve the ``interface_factor_memory_budget`` setting to bytes.

    'auto' (default) = min(32 GB, 0.4 * total system RAM), computed via
    psutil.  Consumed by ``auto_select_interface_solver`` to decide between
    'direct' and 'cg'.
    """
    return _resolve_memory_budget_bytes(
        setting,
        _AUTO_FACTOR_MEMORY_BUDGET_CAP_BYTES,
        _AUTO_FACTOR_MEMORY_BUDGET_FRACTION,
    )


def resolve_block_jacobi_max_bytes(setting: Any = 'auto') -> int:
    """Resolve the ``interface_block_jacobi_max_bytes`` setting to bytes.

    'auto' (default) = max(4 GB legacy floor, min(8 GB, 0.1 * total system
    RAM)), computed via psutil.  The legacy 4 GB floor (the fixed
    pre-Stage-1c ``BLOCK_JACOBI_MAX_FACTOR_BYTES`` constant) ensures the
    host-aware auto formula never resolves to a SMALLER budget than the
    previously-hardcoded value on hosts with <40 GB RAM -- i.e. 'auto' can
    only ever be as-good-or-better than the legacy behaviour, never a
    silent downgrade of block_jacobi -> diagonal jacobi on modest hosts.
    Consumed by ``InterfaceCGSolver._build_block_jacobi`` to decide whether
    to fall back from 'block_jacobi' to the 'jacobi' diagonal
    preconditioner.
    """
    return _resolve_memory_budget_bytes(
        setting,
        _AUTO_BLOCK_JACOBI_MAX_BYTES_CAP,
        _AUTO_BLOCK_JACOBI_MAX_BYTES_FRACTION,
        floor_bytes=BLOCK_JACOBI_MAX_FACTOR_BYTES,
    )


def resolve_coarse_max_bytes(setting: Any = 'auto') -> int:
    """Resolve the ``interface_coarse_max_bytes`` setting to bytes (Finding 5).

    'auto' (default) = min(8 GB, 0.1 * total system RAM), computed via
    psutil -- same cap/fraction as :func:`resolve_block_jacobi_max_bytes`
    (both bound coordinator-resident dense-array allocations of the same
    order). Guards the two dense ``(n, T')`` fp64 arrays
    ``interface_coarse.build_coarse_space`` allocates (``Z_dense`` + ``SZ``)
    -- unlike ``interface_coarse_max_cols`` (a column-count cap), this scales
    with ``n`` too, so it is the guard that actually bounds coordinator
    memory at the "never assemble S_global" / large-n regime this feature
    must not regress.
    """
    return _resolve_memory_budget_bytes(
        setting,
        _AUTO_COARSE_MAX_BYTES_CAP,
        _AUTO_COARSE_MAX_BYTES_FRACTION,
    )


# ---------------------------------------------------------------------------
# Stage 2 helpers: thread-count / dtype / matvec-mode resolution, LPT
# partitioning, kept-position (D1) slicing, SPD-safe pseudo-inverse.
# ---------------------------------------------------------------------------


def resolve_matvec_threads(setting: Any, n_tiles: int) -> int:
    """Resolve the ``matvec_threads`` setting to a concrete thread count.

    'auto' (default) = ``min(DEFAULT_MATVEC_THREADS_CAP, cpu_count, n_tiles)``
    -- capped at 8 (not 32), per the Stage 0 measurement documented in the
    module docstring.  An explicit integer is honoured as-is (not capped) --
    callers who override the default get exactly what they asked for.

    Args:
        setting: ``None``/``'auto'``, or an explicit positive integer
            (also accepts numeric strings, matching the other Stage 1
            memory-budget coercion helpers).
        n_tiles: Number of tiles contributing to the tilewise matvec.  Used
            only for the 'auto' cap (no point spinning up more threads than
            tiles to partition across).

    Returns:
        A positive integer thread count (never less than 1).
    """
    if setting is None or (
        isinstance(setting, str) and setting.strip().lower() == 'auto'
    ):
        cpu = os.cpu_count() or 1
        return max(1, min(DEFAULT_MATVEC_THREADS_CAP, cpu, max(1, n_tiles)))
    if isinstance(setting, bool):
        raise ValueError(
            f"Invalid matvec_threads setting {setting!r}: expected 'auto' "
            f"or a positive integer (bool is not valid)."
        )
    try:
        # Finding 8: same OverflowError-on-.inf gap as
        # _resolve_memory_budget_bytes (e.g. 'matvec_threads: .inf').
        val = int(float(setting))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"Invalid matvec_threads setting {setting!r}: expected 'auto' "
            f"or a positive integer."
        ) from exc
    if val < 1:
        raise ValueError(
            f"Invalid matvec_threads setting {setting!r}: must be >= 1."
        )
    return val


def resolve_matvec_dtype(setting: Any) -> np.dtype:
    """Resolve the ``interface_matvec_dtype`` setting to a numpy dtype.

    Accepts ``'float64'``/``'float32'`` (strings), or the numpy dtype /
    type objects directly.  Default (``None``) is float64.
    """
    if setting is None:
        return np.dtype(np.float64)
    if isinstance(setting, str):
        key = setting.strip().lower()
        if key in ('float64', 'fp64', 'f64', 'double'):
            return np.dtype(np.float64)
        if key in ('float32', 'fp32', 'f32', 'single'):
            return np.dtype(np.float32)
        raise ValueError(
            f"Invalid interface_matvec_dtype {setting!r}: expected "
            f"'float64' or 'float32'."
        )
    dt = np.dtype(setting)
    if dt not in (np.dtype(np.float64), np.dtype(np.float32)):
        raise ValueError(
            f"Invalid interface_matvec_dtype {setting!r}: expected "
            f"float64 or float32."
        )
    return dt


def resolve_matvec_mode(setting: Any, has_tile_blocks: bool) -> str:
    """Resolve ``interface_matvec_mode`` -- 'auto' | 'assembled' | 'tilewise'.

    Item 8: default changed from a hardcoded 'assembled' to 'auto', which
    picks 'tilewise' whenever per-tile dense Schur blocks are available
    (the common, memory-cheaper, now-threaded-and-D1-safe path) and falls
    back to 'assembled' only when they are not (e.g. streaming assembly
    without ``interface_drop_s_global``).  Explicit values are honoured
    verbatim -- 'auto' never overrides an explicit caller choice.
    """
    if setting is None or (
        isinstance(setting, str) and setting.strip().lower() == 'auto'
    ):
        return 'tilewise' if has_tile_blocks else 'assembled'
    if setting not in ('assembled', 'tilewise'):
        raise ValueError(
            f"Invalid interface_matvec_mode {setting!r}: expected 'auto', "
            f"'assembled', or 'tilewise'."
        )
    return setting


def resolve_preconditioner(
    setting: Any, interface_solver_resolved: str, matvec_mode_resolved: str,
) -> str:
    """Resolve the ``interface_preconditioner`` setting -- 'auto' | explicit.

    Stage 3 resolved default: 'auto' (or ``None``) resolves to 'two_level'
    when the resolved interface solver is 'cg' AND the resolved matvec mode
    is 'tilewise' (the regime Stage 2 measured block-Jacobi CG stagnating
    in); otherwise ('assembled' CG, or -- moot, since this is only called
    from the CG branch -- 'direct') it resolves to the legacy 'block_jacobi'
    default. An explicit (non-'auto') value is always honoured verbatim, so
    small systems that already resolve to 'direct' (never reaching this
    function) and any caller/YAML/CLI setting that names a preconditioner
    explicitly are completely unaffected -- this is the ONLY place 'auto' is
    resolved (see ``build_interface_solver``'s docstring).
    """
    # Finding 12: normalize (strip + lower) EVERY string value before
    # validation, not just the 'auto' sentinel -- the pre-fix code left an
    # inconsistent-coercion trap where 'auto'/' AUTO ' was silently accepted
    # but an equivalently-sloppy explicit value (e.g. 'Two_Level ', a
    # trailing space from a quoted YAML scalar) raised ValueError deep
    # inside prepare()/factor() at solve time.
    _norm = setting.strip().lower() if isinstance(setting, str) else setting
    if _norm is None or _norm == 'auto':
        if interface_solver_resolved == 'cg' and matvec_mode_resolved == 'tilewise':
            return 'two_level'
        return 'block_jacobi'
    if _norm not in ('block_jacobi', 'jacobi', 'none', 'amg', 'two_level'):
        raise ValueError(
            f"Invalid interface_preconditioner {setting!r}: expected 'auto', "
            f"'block_jacobi', 'jacobi', 'none', 'amg', or 'two_level'."
        )
    return _norm


def _lpt_partition(costs: List[float], n_bins: int) -> List[List[int]]:
    """Longest-processing-time greedy load-balancing partition.

    Returns a list of ``n_bins`` lists, each containing the indices (into
    ``costs``) assigned to that bin, chosen greedily by always adding the
    next-most-expensive remaining item to the currently-least-loaded bin.
    Same algorithm as the Stage 0 prototype
    (``scripts/benchmark/microbench/bench_interface_matvec.py``).
    """
    n_bins = max(1, n_bins)
    if not costs:
        return [[] for _ in range(n_bins)]
    order = np.argsort(costs)[::-1]
    bins: List[List[int]] = [[] for _ in range(n_bins)]
    loads = np.zeros(n_bins)
    for i in order:
        b = int(np.argmin(loads))
        bins[b].append(int(i))
        loads[b] += costs[i]
    return bins


def kept_position_slice(
    S_i: np.ndarray,
    port_nodes: List[str],
    interface_node_to_idx: Dict[str, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """D1 fix: drop Dirichlet/pad rows+columns from a tile's dense Schur block.

    ``port_nodes`` is the tile's FULL port list (as returned by
    ``factor_and_compute_schur`` / ``bs.port_nodes``), which may include
    Dirichlet (pad) nodes -- those are not part of the interface unknown
    set (``interface_node_to_idx``), so ``S_i`` (shape
    ``(len(port_nodes), len(port_nodes))``) is one row/column too big per
    pad port for the tilewise-CG matvec, which indexes with the
    pad-filtered ``tile_index_maps[tid]``.

    Mathematically exact (not an approximation): slicing away a Dirichlet
    row/column from a SUMMED matrix (``S_global = sum_i S_i_embedded``) is
    identical to summing the sliced blocks, because slicing is a linear
    projection that commutes with addition.  The dropped pad's contribution
    to the unknown-unknown block already lives entirely in the assembly's
    ``rhs_dirichlet`` (via ``-G_ud @ V_d``), not in ``S_i``'s own
    unknown-unknown entries -- see interface_solve_acceleration_plan.md D1.

    Args:
        S_i: Dense ``(n_ports, n_ports)`` Schur block (full port order,
            same order as ``port_nodes``).
        port_nodes: Tile's full port name list, same order as ``S_i``'s
            rows/columns.
        interface_node_to_idx: Global interface (unknown-only) ordering.

    Returns:
        ``(idx, S_kept, kept_pos)`` -- ``idx`` (int32, global interface
        indices, kept order), ``S_kept`` (float64, contiguous, ``(k, k)``
        where ``k <= n_ports``), and ``kept_pos`` (int64, POSITIONS within
        ``port_nodes``/the original full-port-order arrays such as
        ``get_reduced_rhs()``'s return value -- callers filtering a
        full-port-length vector, e.g. ``solve_dc``'s RHS scatter, index with
        this).  ``k`` may be 0 (a tile whose entire port list is Dirichlet
        pads contributes nothing to the tilewise matvec).
    """
    n_ports = len(port_nodes)
    if n_ports == 0:
        empty = np.empty(0, dtype=np.int64)
        return np.empty(0, dtype=np.int32), np.zeros((0, 0), dtype=np.float64), empty

    kept_pos_list = [p for p, nd in enumerate(port_nodes) if nd in interface_node_to_idx]
    kept_pos = np.array(kept_pos_list, dtype=np.int64)
    idx = np.array(
        [interface_node_to_idx[port_nodes[p]] for p in kept_pos_list], dtype=np.int32,
    )
    S_arr = np.asarray(S_i, dtype=np.float64)
    if len(kept_pos_list) == n_ports:
        S_kept = np.ascontiguousarray(S_arr)
    else:
        S_kept = np.ascontiguousarray(S_arr[np.ix_(kept_pos, kept_pos)])
    return idx, S_kept, kept_pos


def filter_kept_rhs(
    g_i: np.ndarray,
    tid: Any,
    idx_map: np.ndarray,
    kept_pos_map: Dict[Any, np.ndarray],
    port_count_map: Optional[Dict[Any, int]] = None,
    *,
    caller: str = "",
) -> np.ndarray:
    """D1/S2/S13-safe filter: reduce a full-port-order per-tile vector to kept positions.

    Shared by every RHS/terminal-vector scatter site that pairs a tile's
    reduced vector (``get_reduced_rhs``/``evaluate_and_get_reduced_rhs``/
    the adjoint terminal-RHS, all returned in the tile's FULL port order,
    possibly including a Dirichlet/pad port directly on the tile's own port
    list) with the pad-FILTERED ``idx_map`` (``tile_index_maps[tid]``) --
    see ``kept_position_slice`` and interface_solve_acceleration_plan.md D1.
    ``solve_dc`` originally had this filter inline; Stage 2's D1 follow-up
    (S2) found the sibling QS/transient/adjoint scatter sites lacked it
    entirely, and Stage 2's own inline version (S13) under-validated: it
    accepted any ``kept_pos`` whose length matched ``idx_map`` without
    checking that ``kept_pos`` was actually built for THIS ``g_i`` -- a
    stale/drifted context (e.g. workers re-set-up after a retile, or a
    mixed pre/post-Stage-2 session) with a coincidentally same-length,
    wrong-valued ``kept_pos`` would silently scatter ``g_i`` to the wrong
    interface rows instead of raising.

    Args:
        g_i: Full-port-order per-tile vector as returned by the worker RPC.
        tid: Tile id (dict key into ``idx_map``'s owning ``tile_index_maps``,
            ``kept_pos_map``, and ``port_count_map``).
        idx_map: This tile's (pad-filtered) global interface index array
            (``ctx.tile_index_maps[tid]``).
        kept_pos_map: ``ctx.tile_kept_port_pos`` -- per-tile positions within
            the FULL port order that survived pad-filtering.
        port_count_map: ``ctx.tile_port_count`` -- per-tile FULL port count
            recorded at the same factor()-time call that built
            ``kept_pos_map[tid]``.  When provided (non-empty) and the tile
            has an entry, ``len(g_i)`` is validated against it before
            slicing.  Pass ``None`` (or an empty/missing-tid dict) only for
            legacy pre-Stage-2 contexts that never recorded it -- the
            length-only fast path and the ``kept_pos``-length check still
            catch the common cases, just without the extra drift guard.
        caller: Short label (e.g. ``"solve_quasi_static"``) included in the
            error message for easier diagnosis.

    Returns:
        ``g_i`` unchanged (fast path, no tile-resident pad port) or
        ``g_i[kept_pos]`` (validated slice down to ``len(idx_map)``).

    Raises:
        ValueError: On any length mismatch that ``kept_pos``/``port_count``
            cannot resolve -- a genuinely inconsistent context. No silent
            'repair' is attempted beyond the documented kept-position slice.
    """
    if len(g_i) == len(idx_map):
        # Fast path: tile has no Dirichlet/pad port on its own port list
        # (the overwhelmingly common case) -- already matches. T4: still
        # validate against port_count_map when available -- a drifted tile
        # (e.g. re-set-up after a retile) whose CURRENT full port count no
        # longer matches what was recorded at factor() time can produce a
        # g_i whose length only COINCIDENTALLY equals len(idx_map); this is
        # exactly the "lengths happen to coincide" corruption the S13
        # slow-path check below exists to catch, except the fast path used
        # to return before ever consulting port_count_map. A consistent
        # (non-drifted) full-count tile always satisfies
        # len(g_i) == port_count == len(idx_map); if port_count_map has an
        # entry for this tile and it differs from len(g_i), that invariant
        # is broken regardless of the len(g_i) == len(idx_map) coincidence.
        _expected_full_fast = port_count_map.get(tid) if port_count_map else None
        if _expected_full_fast is not None and _expected_full_fast != len(g_i):
            raise ValueError(
                f"{caller}: tile {tid!r} reduced-vector length {len(g_i)} "
                f"coincidentally equals index map length {len(idx_map)} "
                f"(fast path), but the FULL port count recorded when "
                f"tile_kept_port_pos[{tid!r}] was built at factor() time is "
                f"{_expected_full_fast} != {len(g_i)}. Slicing this g_i "
                f"unfiltered would silently scatter to the wrong interface "
                f"rows -- this indicates a drifted/stale context (e.g. "
                f"workers re-set-up after a retile). Re-run factor() (not "
                f"just refactor()) to rebuild a consistent context."
            )
        return g_i

    kept_pos = kept_pos_map.get(tid) if kept_pos_map else None
    if kept_pos is None or len(kept_pos) != len(idx_map):
        raise ValueError(
            f"{caller}: tile {tid!r} reduced-vector length {len(g_i)} != "
            f"index map length {len(idx_map)}, and no usable "
            f"tile_kept_port_pos entry is available to recover (D1 "
            f"pad-on-port fix requires a Stage-2 factor() context; a "
            f"pre-Stage-2 checkpoint or backward-compat direct "
            f"construction cannot recover here)."
        )

    expected_full = port_count_map.get(tid) if port_count_map else None
    if expected_full is not None and len(g_i) != expected_full:
        raise ValueError(
            f"{caller}: tile {tid!r} reduced-vector length {len(g_i)} != "
            f"the FULL port count {expected_full} recorded when "
            f"tile_kept_port_pos[{tid!r}] was built at factor() time. "
            f"kept_pos happens to have the right length ({len(kept_pos)}) "
            f"to match idx_map, but slicing with it would silently "
            f"scatter to the wrong interface rows -- this indicates a "
            f"drifted/stale context (e.g. workers re-set-up after a "
            f"retile). Re-run factor() (not just refactor()) to rebuild "
            f"a consistent context."
        )
    if kept_pos.size and int(kept_pos.max()) >= len(g_i):
        raise ValueError(
            f"{caller}: tile {tid!r} kept_pos max index {int(kept_pos.max())} "
            f"is out of bounds for reduced-vector length {len(g_i)} -- "
            f"inconsistent context (see tile_kept_port_pos docs)."
        )
    return g_i[kept_pos]


def _spd_safe_pseudo_solve_factor(
    sub: np.ndarray, eps_rel: float = 1e-10,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Eigh-based PSD-safe pseudo-inverse factor for an indefinite BJ block.

    Replaces the old ``np.linalg.pinv`` fallback (item 7): ``pinv`` of a
    numerically indefinite symmetric matrix can retain tiny NEGATIVE
    eigenvalues from floating-point noise, so the resulting "preconditioner"
    is not actually PSD -- silently voiding CG's convergence guarantee
    (a valid preconditioner for SPD CG must itself be SPD).

    Thin wrapper over :func:`_spd_safe_pseudo_solve_factor_ex` that drops
    its third return value (the raw, unclipped spectrum ``w``).  Kept as a
    separate name -- rather than inlined at call sites -- because it is
    directly unit-tested (``TestSPDSafeFallback``) against the 2-tuple
    ``(V, inv_w)`` contract; production (``_build_block_jacobi``) calls the
    ``_ex`` form directly so it can reuse ``w`` for GenEO-lite enrichment
    without a second eigendecomposition of the same block. Because this is
    a thin wrapper, it is NOT on `_build_block_jacobi`'s call path -- tests
    that need to intercept the eigh-fallback failure mode in that path must
    monkeypatch :func:`_spd_safe_pseudo_solve_factor_ex`, not this function.

    Returns ``None`` only if the block has no positive spectrum at all
    (``lambda_max <= 0``) -- callers should skip the block (identity
    fallback) in that degenerate case.
    """
    result = _spd_safe_pseudo_solve_factor_ex(sub, eps_rel=eps_rel)
    if result is None:
        return None
    V, inv_w, _w = result
    return V, inv_w


def _spd_safe_pseudo_solve_factor_ex(
    sub: np.ndarray, eps_rel: float = 1e-10,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Same as :func:`_spd_safe_pseudo_solve_factor` but also returns the
    raw (unclipped) eigenvalues ``w``.

    A separate function -- not a change to the original's return contract --
    because :func:`_spd_safe_pseudo_solve_factor` is directly unit-tested
    (``TestSPDSafeFallback``) against a 2-tuple ``(V, inv_w)`` return, and
    :func:`_spd_safe_pseudo_solve_factor` now delegates HERE (dropping
    ``w``) so there is exactly one implementation.  Stage 3's GenEO-lite
    enrichment needs the block's full spectrum to pick its lowest eigenpairs
    WITHOUT a second eigendecomposition of the same block (the "don't
    recompute" requirement -- a block that already fell into this
    indefinite-block fallback path has already paid for a full ``eigh``);
    ``_build_block_jacobi`` calls THIS ``_ex`` form directly (always, not
    only when GenEO reuse is needed) so there is exactly one eigh call per
    indefinite block either way.
    """
    sub_sym = 0.5 * (sub + sub.T)
    w, V = np.linalg.eigh(sub_sym)
    lam_max = float(np.max(w)) if w.size else 0.0
    if lam_max <= 0:
        return None
    eps = eps_rel * lam_max
    w_clipped = np.clip(w, eps, None)
    inv_w = 1.0 / w_clipped
    return V, inv_w, w


def _finalize_pool(pool: ThreadPoolExecutor) -> None:
    """Module-level finalizer callback (must not close over ``self``).

    Used as the ``weakref.finalize`` target so a garbage-collected
    ``InterfaceCGSolver`` that never called :meth:`InterfaceCGSolver.close`
    explicitly still releases its thread pool -- a safety net, not the
    primary release mechanism (``release()`` on the owning context calls
    ``close()`` directly; see the lifecycle-parity checklist in the plan).
    """
    pool.shutdown(wait=False)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class InterfaceCGSolver:
    """Iterative CG solver for the global Schur complement interface system.

    Wraps scipy.sparse.linalg.cg with a LinearOperator.  The returned
    callable has the same signature as the direct LU callable:
        v_gamma = solver(rhs)

    Parameters
    ----------
    n_interface : int
        Size of the interface system (number of unknowns).
    matvec_mode : str
        'assembled' or 'tilewise'.
    S_global : sp.csc_matrix, optional
        Assembled sparse Schur matrix.  Required for mode='assembled'.
        Also used for 'jacobi' preconditioner in tilewise mode.
    tile_schur_complements : dict, optional
        {tile_id: S_i (dense np.ndarray)}.  Required for mode='tilewise'.
    tile_index_maps : dict, optional
        {tile_id: np.ndarray(int32)} mapping local port index -> global
        interface index.  Required for mode='tilewise'.
    preconditioner : str
        'block_jacobi' (default), 'jacobi', 'none', 'amg'.
    rtol : float
        Relative tolerance for CG convergence (default 1e-8, validated by the
        Stage 0 sweep: 166 nV max error vs direct on the BRCM-class proxy —
        see docs §7.7).
    maxiter : int or None
        Maximum CG iterations.  None = 3 * n_interface.
    x0 : np.ndarray or None
        Initial guess for the next solve.  Warm-start support.
    stats_dict : dict or None
        If provided, iteration counts and timing are written here.
        Updated in-place on each solve call.
    block_jacobi_max_bytes : int or None
        Per-instance override for the block-Jacobi factor-memory budget
        (bytes).  ``None`` (default) falls back to the module-level
        ``BLOCK_JACOBI_MAX_FACTOR_BYTES`` constant (read dynamically, so
        test monkeypatches of that module attribute still work).  Callers
        that have a ``model.settings['interface_block_jacobi_max_bytes']``
        value should resolve it via ``resolve_block_jacobi_max_bytes()``
        and pass the result here.
    """

    def __init__(
        self,
        n_interface: int,
        matvec_mode: str = 'assembled',
        S_global: Optional[sp.spmatrix] = None,
        tile_schur_complements: Optional[Dict[Any, np.ndarray]] = None,
        tile_index_maps: Optional[Dict[Any, np.ndarray]] = None,
        S_extra: Optional[sp.spmatrix] = None,
        preconditioner: str = 'block_jacobi',
        rtol: float = 1e-8,
        atol: float = 1e-14,
        maxiter: Optional[int] = None,
        x0: Optional[np.ndarray] = None,
        stats_dict: Optional[Dict[str, Any]] = None,
        strict: bool = True,
        block_jacobi_max_bytes: Optional[int] = None,
        matvec_threads: Any = 'auto',
        matvec_dtype: Any = 'float64',
        strict_dtype_rtol: bool = True,
        island_idx: Optional[np.ndarray] = None,
        # Finding 9: None-sentinel defaults, NOT interface_coarse.DEFAULT_*
        # bound at def time (module-import time) -- resolved dynamically in
        # the constructor body instead, so
        # monkeypatch.setattr(interface_coarse, 'DEFAULT_GENEO_K', ...)
        # (etc.) actually takes effect for callers that don't pass these
        # explicitly. See the constructor body's Finding 9 comment.
        interface_coarse_geneo_k: Optional[int] = None,
        interface_coarse_geneo_tol: Optional[float] = None,
        interface_coarse_eps_rank: Optional[float] = None,
        interface_coarse_max_cols: Optional[int] = None,
        interface_coarse_max_bytes: Optional[int] = None,
    ) -> None:
        """
        Parameters
        ----------
        island_idx : np.ndarray, optional
            Stage 3: global interface indices of interface-island nodes
            (penalized via ``S_extra``'s 1e5 mS diagonal).  Only consumed by
            ``preconditioner='two_level'`` -- zeroed out of every coarse-
            space column (PoU and GenEO alike) so the penalty does not leak
            into ``S_c``.  See ``interface_coarse.py``'s module docstring.
        interface_coarse_geneo_k : int
            Stage 3: max GenEO-lite eigenpairs enriched per block-Jacobi
            ownership block (default 4; 0 disables GenEO, leaving a PoU-only
            coarse space).  Only used by ``preconditioner='two_level'``.
        interface_coarse_geneo_tol : float
            Stage 3: relative eigenvalue threshold (fraction of the block's
            own ``lambda_max``) below which an eigenpair is enriched
            (default 1e-6).
        interface_coarse_eps_rank : float
            Stage 3: ``S_c`` eigenvalues <= this fraction of ``S_c``'s
            ``lambda_max`` are treated as structural rank deficiency (e.g.
            the checkerboard null space of an even-multiplicity PoU basis)
            and dropped from the pseudo-inverse (default 1e-12) -- distinct
            knob from ``interface_coarse_geneo_tol``.
        interface_coarse_max_cols : int
            Stage 3: hard cap on T' (default 4096); exceeding it first
            falls back to a PoU-only coarse space (WARNING, GenEO columns
            dropped) -- the coarse space is disabled entirely (degrades to
            plain block-Jacobi) only if the PoU-only column count ALONE
            still exceeds the cap.
        interface_coarse_max_bytes : int
            Stage 3 (Finding 5): byte-based guard on the two dense
            ``(n, T')`` fp64 arrays the coarse build allocates (default
            'auto' -> ``resolve_coarse_max_bytes``, min(8 GB, 0.1x total
            RAM)); same two-rung PoU-only-then-disable degradation as
            ``interface_coarse_max_cols`` above, since a large-n/modest-T'
            system can exceed it even when comfortably under the column cap.
        S_extra : sp.spmatrix, optional
            Additional sparse contribution to the matvec, added on top of the
            per-tile Schur sum.  For ``matvec_mode='tilewise'`` this carries the
            package-edge contribution (G_pkg_uu) that is NOT included in the
            per-tile Schur complements.  Ignored in assembled mode (where S_global
            already contains all contributions).
        atol : float
            Absolute tolerance for CG convergence.  The CG stopping criterion is
            ``||r|| <= max(rtol * ||b||, atol)``.  The default (1e-14) prevents
            CG from burning all maxiter iterations when the RHS is near-zero
            (e.g. early transient steps with no active sources), where
            ``rtol * ||b|| -> 0`` would otherwise make convergence unreachable.
            Set to 0 to recover the pure rtol behaviour.
        strict : bool
            If True (default), raise ``RuntimeError`` when CG does not converge
            within maxiter iterations.  The error message includes the relative
            residual so it appears clearly in logs.  Set to False to demote to a
            warning (not recommended for production; useful only for unit-testing
            the non-convergence path).
        matvec_threads : int or 'auto'
            Stage 2: number of threads for the tilewise matvec and the
            block-Jacobi apply (shared persistent ``ThreadPoolExecutor``,
            lazily built).  'auto' (default) = ``min(8, cpu_count, n_tiles)``
            -- see :func:`resolve_matvec_threads` and the module docstring
            for the Stage 0 measurement behind the cap of 8.  Only used in
            'tilewise' matvec mode (and for block-Jacobi regardless of
            matvec mode).
        matvec_dtype : 'float64' or 'float32'
            Stage 2: storage dtype for each tile's ``S_i`` block in tilewise
            mode ('float64' default).  'float32' halves per-tile storage
            and roughly doubles GEMV throughput on a CPU-only host (see
            module docstring) at the cost of a ~1e-7 relative residual
            floor -- paired with ``strict_dtype_rtol`` (below).  Ignored in
            'assembled' mode.
        strict_dtype_rtol : bool
            If True (default) and ``matvec_dtype='float32'``, raise
            ``ValueError`` when ``rtol < FP32_MATVEC_MIN_RTOL`` (1e-7) --
            fp32's own matvec error floor makes a tighter CG tolerance
            unachievable/meaningless.  Set False to override (e.g. for a
            deliberate accuracy study).
        """
        if matvec_mode not in ('assembled', 'tilewise'):
            raise ValueError(
                f"matvec_mode must be 'assembled' or 'tilewise', got {matvec_mode!r}"
            )
        if matvec_mode == 'assembled' and S_global is None:
            raise ValueError("matvec_mode='assembled' requires S_global")
        if matvec_mode == 'tilewise' and (
            tile_schur_complements is None or tile_index_maps is None
        ):
            raise ValueError(
                "matvec_mode='tilewise' requires tile_schur_complements "
                "and tile_index_maps"
            )
        if preconditioner not in (
            'block_jacobi', 'jacobi', 'none', 'amg', 'two_level',
        ):
            raise ValueError(
                f"preconditioner must be one of 'block_jacobi', 'jacobi', "
                f"'none', 'amg', 'two_level'; got {preconditioner!r}"
            )

        # D1 (item 1): fail loudly, HERE, if a caller passed a tile Schur
        # block whose shape doesn't match its (pad-filtered) index map --
        # this is exactly the corruption pattern the kept-position slicing
        # (result_factorization.py, kept_position_slice() above) fixes at
        # the source; catching it here turns a silent numpy broadcast
        # failure deep inside the CG matvec into an actionable error at
        # construction time.
        if matvec_mode == 'tilewise':
            for tid, S_i in tile_schur_complements.items():
                idx = tile_index_maps[tid]
                if S_i.shape != (len(idx), len(idx)):
                    raise ValueError(
                        f"InterfaceCGSolver: tile {tid!r} Schur block shape "
                        f"{S_i.shape} != (len(tile_index_maps[tid]),)*2 = "
                        f"({len(idx)}, {len(idx)}). This is the D1 pad-port "
                        f"corruption pattern (interface_solve_acceleration_"
                        f"plan.md) -- the caller must slice S_i to kept "
                        f"(non-Dirichlet) port positions via "
                        f"kept_position_slice() before constructing "
                        f"InterfaceCGSolver."
                    )

        self.n = n_interface
        self.matvec_mode = matvec_mode
        self.S_global = S_global
        self.tile_schur_complements = tile_schur_complements
        self.tile_index_maps = tile_index_maps
        self.S_extra = S_extra  # package-edge contribution (tilewise mode)
        self.preconditioner = preconditioner
        # The originally-requested preconditioner, kept even if a runtime
        # memory-budget downgrade (see _build_block_jacobi) replaces
        # self.preconditioner with a weaker one.  Callers/logs that need to
        # know what the USER asked for (as opposed to what is actually
        # active) should read this attribute (finding 11).
        self.requested_preconditioner = preconditioner
        self.rtol = rtol
        self.atol = atol
        self.strict = strict
        self.maxiter = maxiter if maxiter is not None else max(3 * n_interface, 100)
        self._x0: Optional[np.ndarray] = x0
        self._stats: Dict[str, Any] = stats_dict if stats_dict is not None else {}
        # Stage 1c: per-instance block-Jacobi memory budget override.  None
        # means "use the module-level BLOCK_JACOBI_MAX_FACTOR_BYTES constant
        # dynamically" (preserves monkeypatch-based test behaviour).
        self.block_jacobi_max_bytes: Optional[int] = block_jacobi_max_bytes

        # --- Stage 3: two-level coarse-space state --------------------------
        # ``_two_level_requested`` is captured BEFORE _build_block_jacobi()
        # runs because that builder may itself downgrade self.preconditioner
        # 'block_jacobi' -> 'jacobi' (memory budget) -- the coarse-space
        # augmentation step (after _build_linear_op(), below) must still
        # fire in that case (diagonal + coarse is a valid, useful
        # combination), so it cannot key off self.preconditioner alone.
        self._two_level_requested: bool = preconditioner == 'two_level'
        self._island_idx: Optional[np.ndarray] = (
            np.asarray(island_idx, dtype=np.int64)
            if island_idx is not None and len(island_idx) else None
        )
        # Finding 9: None-sentinel constructor defaults (see the signature
        # above) resolved HERE, via a live attribute lookup on
        # interface_coarse, rather than binding interface_coarse.DEFAULT_*
        # as def-time default parameter values -- def-time defaults are
        # evaluated exactly once, at module-import time, so
        # `monkeypatch.setattr(interface_coarse, 'DEFAULT_GENEO_K', ...)`
        # would have no effect on them.
        self._geneo_k: int = int(
            interface_coarse_geneo_k if interface_coarse_geneo_k is not None
            else interface_coarse.DEFAULT_GENEO_K
        )
        self._geneo_tol: float = float(
            interface_coarse_geneo_tol if interface_coarse_geneo_tol is not None
            else interface_coarse.DEFAULT_GENEO_TOL
        )
        self._eps_rank: float = float(
            interface_coarse_eps_rank if interface_coarse_eps_rank is not None
            else interface_coarse.DEFAULT_EPS_RANK
        )
        self._max_cols: int = int(
            interface_coarse_max_cols if interface_coarse_max_cols is not None
            else interface_coarse.DEFAULT_MAX_COLS
        )
        # Stage 1c-style per-instance override (Finding 5): None means "use
        # interface_coarse.DEFAULT_MAX_BYTES dynamically" (read at the point
        # of use in _augment_with_coarse_space, mirrors
        # block_jacobi_max_bytes above -- also a None-sentinel).
        self._coarse_max_bytes: Optional[int] = interface_coarse_max_bytes
        # Whether _build_block_jacobi() should pay for GenEO eigendecomposition
        # at all -- both a 'two_level' request AND a nonzero k are required
        # (k=0 is the documented "GenEO disabled, PoU-only" knob).
        self._want_geneo: bool = self._two_level_requested and self._geneo_k > 0
        # Per-block GenEO eigenpairs, ``[(global_idx, V_k, w_k), ...]`` --
        # populated by _build_block_jacobi() (only when
        # self._two_level_requested and self._geneo_k > 0), consumed by
        # _augment_with_coarse_space() after _build_linear_op() runs.
        self._geneo_pairs: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        self._coarse: Optional[interface_coarse.CoarseSpace] = None
        # Set True inside _build_block_jacobi() iff the 'two_level' base
        # component itself downgraded to the diagonal 'jacobi' fallback
        # (memory budget) -- used only for the human-readable
        # preconditioner_label below.
        self._bj_downgraded: bool = False
        # Finding 8: set inside _augment_with_coarse_space() to whether the
        # base (pre-coarse) preconditioner builder returned None outright
        # (e.g. all block-Jacobi blocks failed) -- distinct from
        # _bj_downgraded (block_jacobi -> jacobi); used so the degrade
        # path/label never mislabels a truly unpreconditioned base as
        # 'block_jacobi'.
        self._bj_was_none: bool = False
        # Finding 13: the base component's ACTUAL name ('none' | 'jacobi' |
        # 'block_jacobi'), derived ONCE from _bj_was_none/_bj_downgraded in
        # _augment_with_coarse_space (the only place both flags are known)
        # instead of being independently re-derived by the same if/elif
        # chain in both the degrade-path warning and preconditioner_label.
        # Placeholder default here; always overwritten before being read
        # (both consumers only run after _augment_with_coarse_space has).
        self._base_precond_label: str = 'block_jacobi'

        # --- Stage 2: threaded matvec/BJ-apply state -----------------------
        self.matvec_dtype: np.dtype = resolve_matvec_dtype(matvec_dtype)
        # S9: matvec_dtype is documented as IGNORED in 'assembled' mode (pure
        # fp64 sparse matvec on S_global -- see _build_linear_op) -- only
        # 'tilewise' mode actually casts S_i to fp32 and hits the ~1e-7
        # residual floor this strict check exists to prevent an unachievable
        # rtol for.  Gate on matvec_mode so a shared solver.yaml with
        # interface_matvec_dtype='float32' (intended for tilewise runs)
        # doesn't abort an assembled-mode fallback run that would have
        # converged fine in fp64.
        if (
            self.matvec_mode == 'tilewise'
            and self.matvec_dtype == np.dtype(np.float32)
            and strict_dtype_rtol
        ):
            if rtol < FP32_MATVEC_MIN_RTOL:
                raise ValueError(
                    f"matvec_dtype='float32' requires rtol >= "
                    f"{FP32_MATVEC_MIN_RTOL:.0e} (fp32 tilewise matvec has a "
                    f"~1e-7 relative residual floor -- a tighter CG rtol is "
                    f"unachievable). Got rtol={rtol:.2e}. Raise rtol, use "
                    f"matvec_dtype='float64', or pass "
                    f"strict_dtype_rtol=False to override."
                )
        # n_tiles for the 'auto' thread-count cap: prefer tile_index_maps
        # (present whenever block-Jacobi ownership is computed, in BOTH
        # matvec modes) over tile_schur_complements (tilewise-only) so an
        # assembled-mode solver with many tiles' worth of BJ ownership
        # blocks still gets a sensible 'auto' thread count for the
        # threaded BJ apply (item 5), not just the threaded matvec.
        if tile_index_maps:
            n_tiles = len(tile_index_maps)
        elif tile_schur_complements:
            n_tiles = len(tile_schur_complements)
        else:
            n_tiles = 0
        self.matvec_threads: int = resolve_matvec_threads(matvec_threads, n_tiles)
        self._pool: Optional[ThreadPoolExecutor] = None
        self._pool_lock = threading.Lock()

        # Build preconditioner BEFORE the linear op (order matters for fp32
        # tilewise storage, see below).  The block_jacobi/jacobi fallback
        # paths (S_global unavailable, tilewise-only) read
        # self.tile_schur_complements in fp64 -- they must run before
        # _build_linear_op()'s tilewise state prep converts those blocks to
        # fp32 and frees the fp64 originals in place.  Neither builder
        # depends on the other's *output* (only this ordering of reads),
        # so swapping them is safe.
        self._M: Optional[spla.LinearOperator] = self._build_preconditioner()
        # Build operator (tilewise mode: converts/frees tile_schur_complements
        # to matvec_dtype in place -- see _prepare_tilewise_matvec_state).
        self._linear_op: spla.LinearOperator = self._build_linear_op()

        # Stage 3: layer the coarse-space correction on top of the just-built
        # base (block_jacobi, or its jacobi downgrade) preconditioner -- must
        # run AFTER self._linear_op exists (S @ Z uses its matmat).  Degrades
        # to the plain base preconditioner (WARNING, never raises) if the
        # coarse build itself fails.
        if self._two_level_requested:
            self._M = self._augment_with_coarse_space()

        # Track cumulative iteration stats
        self._total_iters: int = 0
        self._total_solves: int = 0

    # ------------------------------------------------------------------
    # Thread pool lifecycle (Stage 2)
    # ------------------------------------------------------------------

    def _get_pool(self) -> ThreadPoolExecutor:
        """Lazily build the persistent matvec/BJ-apply thread pool.

        A ``weakref.finalize`` safety net is registered on first build so a
        garbage-collected solver that never called :meth:`close` explicitly
        still releases its threads; the primary release path is still
        ``close()`` (called by the owning context's ``release()``).
        """
        if self._pool is None:
            with self._pool_lock:
                if self._pool is None:
                    pool = ThreadPoolExecutor(
                        max_workers=self.matvec_threads,
                        thread_name_prefix='ifacecg-mv',
                    )
                    self._pool = pool
                    weakref.finalize(self, _finalize_pool, pool)
        return self._pool

    def close(self) -> None:
        """Shut down the persistent thread pool, if one was built.

        Idempotent.  Must be called by the owning context's ``release()``
        (lifecycle-parity checklist) so repeated prepare/release cycles do
        not accumulate live thread pools.
        """
        pool = self._pool
        self._pool = None
        if pool is not None:
            pool.shutdown(wait=True)

    # ------------------------------------------------------------------
    # LinearOperator / preconditioner construction
    # ------------------------------------------------------------------

    def _build_linear_op(self) -> spla.LinearOperator:
        """Build the matvec/matmat LinearOperator for S_global.

        'assembled' mode is unchanged (sparse matvec on S_global).
        'tilewise' mode (Stage 2) precomputes an LPT partition of tiles by
        ``n_ports**2`` across ``self.matvec_threads`` bins and, for each
        bin, the UNION of global interface indices its tiles touch (the
        compact-accumulator scatter design -- see module docstring for the
        measured decision).  ``matvec_threads <= 1`` uses a plain serial
        loop (no pool overhead).
        """
        n = self.n

        if self.matvec_mode == 'assembled':
            # Convert to CSR for efficient matvec
            S = self.S_global.tocsr()

            def _matvec(x: np.ndarray) -> np.ndarray:
                return S @ x

            def _matmat(X: np.ndarray) -> np.ndarray:
                return S @ X

            return spla.LinearOperator(
                shape=(n, n), matvec=_matvec, matmat=_matmat, dtype=np.float64,
            )

        # --- 'tilewise' mode -------------------------------------------------
        self._prepare_tilewise_matvec_state()
        return spla.LinearOperator(
            shape=(n, n),
            matvec=self._tilewise_matvec,
            matmat=self._tilewise_matmat,
            dtype=np.float64,
        )

    def _prepare_tilewise_matvec_state(self) -> None:
        """Precompute the tilewise-matvec working set (Stage 2).

        Populates:
          ``self._tiles``       -- list of (idx int64, S_i dtype-cast) pairs.
          ``self._mv_partition``-- LPT partition of tile indices into
                                   ``self.matvec_threads`` bins by cost n_p^2.
          ``self._mv_touched``  -- per-bin sorted-unique array of global
                                   interface indices the bin's tiles touch
                                   (the compact-accumulator scatter target).
          ``self._S_extra_csr`` -- CSR S_extra (or None).

        Finding 3 (fp32 storage): when ``dtype`` is float32, each tile's
        fp64 block is cast and the fp64 original is immediately freed by
        overwriting the ``self.tile_schur_complements[tid]`` entry in
        place, so peak coordinator memory during this loop is one tile's
        block (fp64 + fp32 transiently), not the whole dict held in both
        dtypes at once -- at BRCM scale, keeping both would be 12.5 GB +
        6.2 GB, strictly worse than fp64-only.  After this method returns,
        ``self.tile_schur_complements`` holds only fp32 arrays; the only
        readers of that dict are the block_jacobi/jacobi preconditioner
        builders, which are guaranteed to have already run (see the
        build-order comment in ``__init__``), and no caller keeps its own
        reference to the fp64 arrays past constructing this solver (the
        dict object itself is shared with the caller, but callers never
        read it again after passing it in -- see the callers in
        ``result_factorization.py``).
        """
        dtype = self.matvec_dtype
        free_fp64 = dtype != np.float64
        tiles: List[Tuple[np.ndarray, np.ndarray]] = []
        for tid, S_i in self.tile_schur_complements.items():
            idx = np.asarray(self.tile_index_maps[tid], dtype=np.int64)
            S_arr = np.asarray(S_i)
            S_cast = np.ascontiguousarray(S_arr.astype(dtype, copy=False))
            tiles.append((idx, S_cast))
            if free_fp64:
                self.tile_schur_complements[tid] = S_cast
        self._tiles = tiles

        costs = [float(S.shape[0]) ** 2 for _, S in tiles]
        # Finding 15: cap the bin count at len(tiles) -- an explicit
        # matvec_threads > n_tiles otherwise produces permanently empty LPT
        # bins whose no-op work items are still dispatched through the
        # thread pool on EVERY matvec call (pure dispatch overhead in the
        # innermost hot loop, exactly the regime the module's own Stage 0
        # note says inverts scaling). Pool size itself (self.matvec_threads,
        # via _get_pool()) is left uncapped -- only the partition/dispatch
        # bin count shrinks to match the actual amount of parallelizable
        # work.
        self._mv_n_bins = min(self.matvec_threads, len(tiles)) if tiles else 1
        self._mv_partition = _lpt_partition(costs, self._mv_n_bins)

        touched: List[np.ndarray] = []
        for t in range(self._mv_n_bins):
            part = self._mv_partition[t]
            if part:
                touched.append(
                    np.unique(np.concatenate([tiles[i][0] for i in part]))
                )
            else:
                touched.append(np.empty(0, dtype=np.int64))
        self._mv_touched = touched

        self._S_extra_csr = (
            self.S_extra.tocsr() if self.S_extra is not None else None
        )

    def _tilewise_extra_term(self, X: np.ndarray) -> np.ndarray:
        """S_extra @ X (or a zero array), broadcasting to X's shape."""
        if self._S_extra_csr is not None:
            return np.asarray(self._S_extra_csr @ X, dtype=np.float64)
        return np.zeros_like(X, dtype=np.float64)

    def _tilewise_matvec(self, x: np.ndarray) -> np.ndarray:
        if self.matvec_threads <= 1 or len(self._tiles) < 2:
            return self._tilewise_matvec_serial(x)
        return self._tilewise_matvec_threaded(x)

    def _tilewise_matvec_serial(self, x: np.ndarray) -> np.ndarray:
        """Serial tilewise matvec (also the n_threads<=1 code path).

        Scatter-add via ``np.bincount``: ~10-30x faster than ``np.add.at``
        for dense index maps with repeated global indices (same bottleneck
        A1 replaced in ``solver_td.py`` for the transient RHS).
        """
        n = self.n
        dtype = self.matvec_dtype
        result = self._tilewise_extra_term(x)
        for idx, S_i in self._tiles:
            x_local = x[idx]
            if dtype != np.float64:
                x_local = x_local.astype(dtype, copy=False)
            y_local = np.asarray(S_i @ x_local, dtype=np.float64)
            result += np.bincount(idx, weights=y_local, minlength=n)
        return result

    def _tilewise_matvec_threaded(self, x: np.ndarray) -> np.ndarray:
        """Threaded tilewise matvec: LPT partition + compact per-thread buffers.

        Each thread's work is gather (``x[idx]``) -> GEMV
        (``S_i @ x_local``, releases the GIL inside BLAS) -> scatter into
        its OWN compact buffer (sized to its touched-index union, not the
        full interface) -- avoiding the Stage-0-flagged inverted scaling
        from a full ``(n_threads, n)`` accumulator's zero-fill + reduction
        cost growing with thread count.  BLAS is pinned to 1 thread for the
        duration of the pool region to avoid nested-parallelism oversubscription.
        """
        n = self.n
        dtype = self.matvec_dtype
        tiles = self._tiles
        part = self._mv_partition
        touched = self._mv_touched
        # Finding 15: dispatch over the CAPPED bin count (min(matvec_
        # threads, n_tiles), computed once in _prepare_tilewise_matvec_
        # state), not self.matvec_threads -- otherwise an explicit
        # matvec_threads > n_tiles dispatches permanently-empty no-op work
        # items through the pool on every matvec call.
        n_threads = self._mv_n_bins
        local_bufs: List[np.ndarray] = [None] * n_threads  # type: ignore[list-item]

        def work(t: int) -> None:
            u = touched[t]
            if len(u) == 0:
                local_bufs[t] = np.zeros(0, dtype=np.float64)
                return
            buf = np.zeros(len(u), dtype=np.float64)
            for i in part[t]:
                idx, S_i = tiles[i]
                x_local = x[idx]
                if dtype != np.float64:
                    x_local = x_local.astype(dtype, copy=False)
                y_local = np.asarray(S_i @ x_local, dtype=np.float64)
                pos = np.searchsorted(u, idx)
                buf += np.bincount(pos, weights=y_local, minlength=len(u))
            local_bufs[t] = buf

        pool = self._get_pool()
        with threadpoolctl.threadpool_limits(1):
            list(pool.map(work, range(n_threads)))

        result = self._tilewise_extra_term(x)
        for t in range(n_threads):
            u = touched[t]
            if len(u):
                result[u] += local_bufs[t]
        return result

    def _tilewise_matmat(self, X: np.ndarray) -> np.ndarray:
        """Per-tile GEMM + scatter (item 6): ``S_global-equivalent @ X``.

        Used by Stage 3's coarse-space setup (``S @ Z``, T'+1 columns) and
        exercised directly by the matmat-vs-column-matvecs equivalence
        test.  Threaded when ``matvec_threads > 1`` (same LPT
        partition/touched-index buffers as the matvec, generalised to
        multiple columns).
        """
        X = np.asarray(X, dtype=np.float64)
        squeeze = X.ndim == 1
        X2 = X.reshape(self.n, 1) if squeeze else X
        n, k = X2.shape[0], X2.shape[1]

        if self.matvec_threads <= 1 or len(self._tiles) < 2:
            result = self._tilewise_extra_term(X2)
            dtype = self.matvec_dtype
            for idx, S_i in self._tiles:
                X_local = X2[idx, :]
                if dtype != np.float64:
                    X_local = X_local.astype(dtype, copy=False)
                Y_local = np.asarray(S_i @ X_local, dtype=np.float64)
                np.add.at(result, idx, Y_local)
            return result[:, 0] if squeeze else result

        dtype = self.matvec_dtype
        tiles = self._tiles
        part = self._mv_partition
        touched = self._mv_touched
        # Finding 15: see the matching comment in _tilewise_matvec_threaded.
        n_threads = self._mv_n_bins
        local_bufs: List[np.ndarray] = [None] * n_threads  # type: ignore[list-item]

        def work(t: int) -> None:
            u = touched[t]
            if len(u) == 0:
                local_bufs[t] = np.zeros((0, k), dtype=np.float64)
                return
            buf = np.zeros((len(u), k), dtype=np.float64)
            for i in part[t]:
                idx, S_i = tiles[i]
                X_local = X2[idx, :]
                if dtype != np.float64:
                    X_local = X_local.astype(dtype, copy=False)
                Y_local = np.asarray(S_i @ X_local, dtype=np.float64)
                pos = np.searchsorted(u, idx)
                np.add.at(buf, pos, Y_local)
            local_bufs[t] = buf

        pool = self._get_pool()
        with threadpoolctl.threadpool_limits(1):
            list(pool.map(work, range(n_threads)))

        result = self._tilewise_extra_term(X2)
        for t in range(n_threads):
            u = touched[t]
            if len(u):
                result[u, :] += local_bufs[t]
        return result[:, 0] if squeeze else result

    def _build_preconditioner(self) -> Optional[spla.LinearOperator]:
        """Build preconditioner LinearOperator."""
        n = self.n

        if self.preconditioner == 'none':
            return None

        if self.preconditioner == 'jacobi':
            # Extract diagonal from S_global (assembled) or sum tile diagonals
            if self.S_global is not None:
                diag = np.array(self.S_global.diagonal(), dtype=np.float64)
            else:
                diag = np.zeros(n, dtype=np.float64)
                for tid, S_i in self.tile_schur_complements.items():
                    idx = self.tile_index_maps[tid]
                    np.add.at(diag, idx, np.diag(S_i))
                # T2/S4: never-assemble mode (S_global is None) needs
                # S_extra's diagonal (island 1e5 penalties + package
                # conductances) added in too -- S = sum_i P_i^T S_i P_i +
                # S_extra, so the tile-diagonal sum alone omits exactly the
                # terms that, in assembled mode, live in S_global's
                # diagonal. This is the same S4 fix already applied to the
                # sibling _build_jacobi_fallback (the memory-budget-downgrade
                # path); this is the EXPLICIT preconditioner='jacobi' branch,
                # which was missed. Use self.S_extra (set in __init__,
                # before this builder runs), not self._S_extra_csr (only
                # populated later by _build_linear_op()).
                if self.S_extra is not None:
                    diag += np.asarray(
                        self.S_extra.tocsr().diagonal(), dtype=np.float64
                    )
            diag = np.where(diag > 0, diag, 1.0)  # guard against zero diagonal

            def _msolve(x: np.ndarray) -> np.ndarray:
                return x / diag

            return spla.LinearOperator(shape=(n, n), matvec=_msolve, dtype=np.float64)

        if self.preconditioner == 'block_jacobi':
            return self._build_block_jacobi()

        if self.preconditioner == 'two_level':
            # Stage 3: the coarse-space TERM is layered on top AFTER
            # self._linear_op exists (see _augment_with_coarse_space, called
            # from __init__ right after _build_linear_op()) -- this builder
            # only produces the BASE component here, identical to
            # 'block_jacobi' (including its own memory-budget downgrade to
            # 'jacobi', which self._build_block_jacobi() applies to
            # self.preconditioner directly; _two_level_requested -- captured
            # before this call -- is what the post-linear-op step keys off
            # of, not self.preconditioner, so the coarse term still gets
            # added even after a downgrade).
            return self._build_block_jacobi()

        if self.preconditioner == 'amg':
            return self._build_amg()

        return None

    def _augment_with_coarse_space(self) -> Optional[spla.LinearOperator]:
        """Stage 3: layer ``Z S_c^+ Z^T`` on top of the already-built base
        (block_jacobi, or its jacobi memory-budget downgrade) preconditioner.

        Must run AFTER ``self._linear_op`` exists (the coarse build's
        ``S @ Z`` matmat needs the fully-prepared tilewise/assembled matvec
        state -- see the module docstring's Stage 3 ordering note).
        ``build_coarse_space`` itself falls back to a PoU-only coarse space
        (still returned here, not ``None``) when GenEO pushes T' over the
        cap/byte budget; this method only degrades to the plain base
        preconditioner (``self._M``, already built) with a WARNING -- NEVER
        RAISES -- both when ``build_coarse_space`` returns ``None`` (PoU-only
        T'/bytes ALSO exceed budget, no usable ``tile_index_maps``, ``S_c``
        has no positive spectrum, ...) and (Finding 6) when it raises ANY
        exception outright (e.g. ``MemoryError``/``LinAlgError`` from the
        dense ``Z.toarray()``/``eigh(S_c)`` calls) -- a pathological S_c must
        degrade this (new-default, 'auto'-resolved) code path the same way
        the sibling block-Jacobi eigh fallback degrades (see the "Finding 7"
        comment in ``_build_block_jacobi`` for the matching precedent).
        """
        base_M = self._M
        # Finding 8: remember whether the base builder returned None
        # OUTRIGHT (e.g. every block-Jacobi block failed) -- distinct from
        # self._bj_downgraded (block_jacobi -> jacobi) -- so the degrade
        # path/label below can report 'none' instead of falsely claiming
        # 'block_jacobi' is active.
        self._bj_was_none = base_M is None
        # Finding 13: derive the base label ONCE here (the only place
        # _bj_was_none and _bj_downgraded are both known) -- see
        # self._base_precond_label's docstring in __init__.
        if self._bj_was_none:
            self._base_precond_label = 'none'
        elif self._bj_downgraded:
            self._base_precond_label = 'jacobi'
        else:
            self._base_precond_label = 'block_jacobi'
        base_apply = base_M.matvec if base_M is not None else (lambda r: r.copy())

        # Finding 9: read interface_coarse.DEFAULT_MAX_BYTES dynamically
        # (attribute lookup at call time), NOT a module-level snapshot --
        # see the removed COARSE_MAX_BYTES_DEFAULT constant's comment.
        _max_bytes = (
            self._coarse_max_bytes if self._coarse_max_bytes is not None
            else interface_coarse.DEFAULT_MAX_BYTES
        )
        # Finding 7: matvec_dtype only actually affects the matmat's fp
        # precision in 'tilewise' mode (S9 -- 'assembled' mode's sparse
        # matvec on S_global is always fp64 regardless of the dtype
        # setting), so only floor eps_rank when that matmat is genuinely
        # running in fp32.
        _coarse_matvec_dtype = (
            self.matvec_dtype if self.matvec_mode == 'tilewise'
            else np.dtype(np.float64)
        )
        try:
            coarse = interface_coarse.build_coarse_space(
                matmat=self._linear_op.matmat,
                tile_index_maps=self.tile_index_maps or {},
                n=self.n,
                island_idx=self._island_idx,
                geneo_pairs=self._geneo_pairs,
                max_cols=self._max_cols,
                eps_rank=self._eps_rank,
                max_bytes=_max_bytes,
                matvec_dtype=_coarse_matvec_dtype,
            )
        except Exception as exc:
            # Finding 6: build_coarse_space is documented "never raises" at
            # this call site, but its own dense allocation/eigh calls can
            # raise MemoryError/LinAlgError/ValueError on a pathological S_c
            # -- catch-all here (matching the sibling block-Jacobi eigh
            # fallback's convention) so a coarse-build failure degrades to
            # the base preconditioner instead of aborting prepare()/factor()
            # entirely.
            logger.warning(
                "InterfaceCGSolver: two_level coarse-space build raised "
                "%s: %s; degrading to the plain base preconditioner for "
                "this solve.",
                type(exc).__name__, exc,
            )
            coarse = None

        if coarse is None:
            # The base component's ACTUAL name -- self.preconditioner is
            # still 'two_level' at this point (only the success path below
            # overwrites it; _build_block_jacobi() only ever rewrites it to
            # 'jacobi' on a memory-budget downgrade, never back to
            # 'block_jacobi'), so use self._base_precond_label (Finding 13:
            # derived once, above) rather than trusting self.preconditioner
            # (Finding 8: must not claim 'block_jacobi' when the base
            # builder actually returned None).
            _actual = self._base_precond_label
            logger.warning(
                "InterfaceCGSolver: two_level coarse-space build failed or "
                "was disabled (see preceding WARNING); degrading to the "
                "plain %r preconditioner for this solve.",
                _actual,
            )
            self._coarse = None
            self.preconditioner = _actual
            return base_M

        self._coarse = coarse
        self.preconditioner = 'two_level'
        n = self.n

        def _msolve(r: np.ndarray) -> np.ndarray:
            return base_apply(r) + coarse.apply(r)

        logger.info(
            "InterfaceCGSolver: two_level preconditioner active -- %s",
            self.preconditioner_label,
        )
        return spla.LinearOperator(shape=(n, n), matvec=_msolve, dtype=np.float64)

    @property
    def preconditioner_label(self) -> str:
        """Human-readable preconditioner description for stats/logs, e.g.
        ``"two_level(bj+geneo k=4, T'=321, rank=320)"``.  Falls back to the
        plain ``self.preconditioner`` string when no coarse space is active
        (including a degraded two_level request)."""
        if self._coarse is not None:
            # Finding 13: self._base_precond_label (derived once in
            # _augment_with_coarse_space -- see its docstring) is 'none' |
            # 'jacobi' | 'block_jacobi'; the label uses the short 'bj' form
            # for the 'block_jacobi' case only (Finding 8: 'none' must not
            # fall through to 'bj' when the base builder returned None
            # outright -- coarse-only correction, no block/diagonal base at
            # all).
            base = (
                'bj' if self._base_precond_label == 'block_jacobi'
                else self._base_precond_label
            )
            return f"two_level({base}+geneo k={self._coarse.n_geneo_cols}, T'={self._coarse.n_cols}, rank={self._coarse.rank})"
        return self.preconditioner

    def _build_block_jacobi(self) -> Optional[spla.LinearOperator]:
        """Block-Jacobi preconditioner.

        Two implementation paths:
        1. S_global available (assembled mode or tilewise with S_global passed):
           uses the actual principal submatrices of S_global for each tile's
           owned nodes.  These are the TRUE block-diagonal blocks of the global
           system.  Much better preconditioner than path 2.
        2. S_global not available (tilewise-only): falls back to per-tile
           Schur complement submatrices.  These are NOT the diagonal blocks of
           S_global (they miss cross-tile couplings and package edges), so they
           are a weaker approximation.

        Ownership assignment: each interface node is assigned to the FIRST
        tile (in iteration order of tile_index_maps keys) whose tile_index_map
        includes it.  The owner contributes the block.

        Each per-owner submatrix is factored via scipy.linalg.cho_factor (dense
        Cholesky) for numerically robust small-block solves.  Falls back to
        pinv on singular blocks (1e-12 diagonal regularization applied first).

        Memory guard: when the estimated factor memory across all blocks exceeds
        ``BLOCK_JACOBI_MAX_FACTOR_BYTES`` (default 4 GB), falls back to the
        'jacobi' (diagonal) preconditioner automatically and logs a WARNING.
        This prevents the block-Jacobi from re-introducing the coordinator-
        memory pressure that CG mode is designed to remove.  Scaling analysis:
        for T balanced tiles with n total interface nodes, total factor memory
        is Sum(k_i^2) * 8 bytes ≈ n^2/T * 8 bytes.  At n=1M, T=1000 this is
        ~8 GB; with fewer/larger tiles it grows rapidly.  Use 'jacobi' or
        increase T (via B1 splitting) when targeting 1M+ interface nodes.

        Returns None only on a complete build failure; triggers logged warning.
        """
        n = self.n

        # ---- 1. Assign ownership (first-seen tile for each global index) ----
        owner_tile: Dict[int, Any] = {}
        if self.tile_index_maps:
            for tid, idx in self.tile_index_maps.items():
                for g in idx:
                    gi = int(g)
                    if gi not in owner_tile:
                        owner_tile[gi] = tid
        elif self.tile_schur_complements is None:
            # Neither tile_index_maps nor tile_schur_complements: cannot build
            return None

        # ---- 2. Collect owned global indices per tile ----
        tile_owned: Dict[Any, List[int]] = {}
        for g, tid in owner_tile.items():
            tile_owned.setdefault(tid, []).append(g)

        # ---- 2b. Pre-flight memory estimate ----
        # Sum(k_i^2) * 8 bytes = total dense Cholesky factor memory.
        # Each k_i is the number of nodes owned by tile i.  Use the
        # owned-count dict (already built) to estimate without allocating.
        #
        # Stage 1c: use the per-instance override when provided (resolved
        # from the 'interface_block_jacobi_max_bytes' setting), else fall
        # back to the module-level constant (read dynamically so test
        # monkeypatches of the module attribute keep working).
        _max_bytes = (
            self.block_jacobi_max_bytes
            if self.block_jacobi_max_bytes is not None
            else BLOCK_JACOBI_MAX_FACTOR_BYTES
        )
        _max_block_pre = max(len(v) for v in tile_owned.values()) if tile_owned else 0
        est_factor_bytes = sum(
            len(owned) * len(owned) * 8 for owned in tile_owned.values()
        )
        # Finding 10 (round 1) / Finding 8 (round 2): the materialization
        # loop below (section 4) transiently holds, for the block currently
        # being symmetrized (`Sinv = 0.5 * (Sinv + Sinv.T)`), FOUR k x k
        # arrays at once: the not-yet-replaced cho factor (`payload`,
        # already counted in est_factor_bytes above), the freshly-built
        # dense inverse from cho_solve (`Sinv`), the `Sinv + Sinv.T` sum
        # temporary, and the `0.5 * (...)` result array that `Sinv` gets
        # reassigned to (the OLD Sinv object from cho_solve is not
        # reclaimed until this expression finishes evaluating) -- three
        # arrays BEYOND the counted cho factor, not two.  Add the
        # worst-case extra (the single largest block's three transient
        # arrays) so this pre-flight estimate does not underestimate the
        # actual build-time peak.
        est_factor_bytes += 3 * _max_block_pre * _max_block_pre * 8
        if est_factor_bytes > _max_bytes:
            max_block = _max_block_pre
            # Finding 5: this builder also serves preconditioner='two_level'
            # requests (dispatched from _build_preconditioner's two_level
            # branch, which calls this method to build the BASE component --
            # see that branch's comment) -- name the ACTUAL requested
            # preconditioner (self.requested_preconditioner), not a
            # hardcoded 'block_jacobi', and -- when the request was
            # 'two_level' -- say so explicitly: this fallback path
            # (_build_jacobi_fallback) never reaches the per-block
            # cho_factor loop below, so no blocks get cho-factored and
            # GenEO enrichment is silently skipped entirely for this factor
            # (self._geneo_pairs stays empty); the coarse space built
            # afterward by _augment_with_coarse_space is PoU-only, not
            # PoU+GenEO.
            _is_two_level = self.requested_preconditioner == 'two_level'
            logger.warning(
                "Block-Jacobi: estimated factor memory %.1f GB exceeds budget "
                "%.1f GB (setting 'interface_block_jacobi_max_bytes', "
                "n_interface=%d, T=%d tiles, max_block=%d). Downgrading "
                "requested preconditioner %r -> 'jacobi' "
                "(diagonal) for this solve -- expect MORE CG iterations per "
                "solve than block_jacobi would have needed (diagonal is a "
                "strictly weaker preconditioner).%s "
                "Scaling: Sum(k_i^2)*8 bytes ~ n^2/T*8; at n=1M, T=1000 this "
                "is ~8 GB. To keep block_jacobi, reduce interface size via B1 "
                "tile splitting or raise 'interface_block_jacobi_max_bytes'.",
                est_factor_bytes / 1024 ** 3,
                _max_bytes / 1024 ** 3,
                n,
                len(tile_owned),
                max_block,
                self.requested_preconditioner,
                (
                    " GenEO enrichment is SKIPPED for this factor (no "
                    "blocks are cho-factored on this fallback path); the "
                    "two_level coarse space -- if it can still be built at "
                    "all -- will be PoU-only, not PoU+GenEO."
                    if _is_two_level else ""
                ),
            )
            # Fall back to diagonal preconditioner (cheap, always safe).
            # Keep self.requested_preconditioner == 'block_jacobi' (set in
            # __init__) so callers/logs can distinguish "asked for" from
            # "actually in use"; downgrade self.preconditioner so every
            # downstream report (backend_info, verbose logs) reflects reality.
            self.preconditioner = 'jacobi'
            self._bj_downgraded = True
            return self._build_jacobi_fallback()

        logger.debug(
            "Block-Jacobi: estimated factor memory %.1f MB "
            "(n=%d, T=%d tiles, max_block=%d).",
            est_factor_bytes / 1024 ** 2,
            n,
            len(tile_owned),
            _max_block_pre,
        )

        # ---- 3. Extract submatrices and build Cholesky factors ----
        _block_factors: List[Tuple[np.ndarray, Any, bool]] = []

        # Path 1: use actual diagonal blocks of S_global (preferred)
        _use_s_global = self.S_global is not None
        if _use_s_global:
            S_csr = self.S_global.tocsr()
        # S4: never-assemble mode (S_global is None) needs S_extra's
        # contribution (island 1e5 penalties + package conductances) added
        # into each owned block too -- S = sum_i P_i^T S_i P_i + S_extra, so
        # a block built from tile S_i alone omits exactly the terms that, in
        # assembled mode, live in S_global's diagonal blocks.  Use
        # self.S_extra (available from __init__, before self._S_extra_csr
        # is populated by _build_linear_op() later).
        _S_extra_csr_early = (
            self.S_extra.tocsr() if self.S_extra is not None else None
        )

        for tile_or_node_group, owned_global in tile_owned.items():
            owned_global_arr = np.array(sorted(owned_global), dtype=np.int32)

            if _use_s_global:
                # Extract true diagonal block from S_global
                sub = S_csr[np.ix_(owned_global_arr, owned_global_arr)].toarray()
            elif (self.tile_schur_complements is not None
                  and self.tile_index_maps is not None):
                # Fallback: use per-tile Schur block (less accurate)
                tid = tile_or_node_group
                S_i = np.asarray(self.tile_schur_complements[tid], dtype=np.float64)
                idx_full = self.tile_index_maps[tid]
                global_to_local: Dict[int, int] = {
                    int(g): loc for loc, g in enumerate(idx_full)
                }
                owned_local = np.array(
                    [global_to_local[g] for g in owned_global_arr], dtype=np.int32
                )
                sub = S_i[np.ix_(owned_local, owned_local)].copy()
                if _S_extra_csr_early is not None:
                    sub += _S_extra_csr_early[
                        np.ix_(owned_global_arr, owned_global_arr)
                    ].toarray()
            else:
                continue

            # Regularize to ensure SPD
            sub += 1e-12 * np.eye(sub.shape[0])

            # Finding 3: local (within-this-block) island mask, used by
            # BOTH GenEO call sites below so the eigen-analysis runs on the
            # physical (non-penalty-inflated) spectrum -- see
            # interface_coarse.geneo_lowest_eigenpairs's island_local_mask
            # docstring for the full rationale.  Island nodes CAN appear in
            # an ownership block (see interface_coarse.py's module
            # docstring, "Island rows in GenEO columns").
            island_local_mask = (
                np.isin(owned_global_arr, self._island_idx)
                if self._island_idx is not None else None
            )

            try:
                cho = la.cho_factor(sub, lower=False, check_finite=False)
            except la.LinAlgError:
                # SPD-safe fallback (item 7): eigh-based PSD projection,
                # NOT np.linalg.pinv.  pinv of a numerically indefinite
                # symmetric block can retain tiny negative eigenvalues from
                # FP noise -- silently voiding CG's convergence guarantee
                # (a preconditioner must itself be SPD).  See
                # _spd_safe_pseudo_solve_factor's docstring.
                logger.warning(
                    "Block-Jacobi: owned block size %d is singular/"
                    "indefinite; using eigh-based SPD-safe pseudo-inverse "
                    "fallback (clipped to >= eps*lambda_max).",
                    sub.shape[0],
                )
                # S11: eigh itself can raise LinAlgError (eigenvalue
                # non-convergence, NaN/Inf-contaminated block) -- the old
                # pre-Stage-2 code wrapped its np.linalg.pinv fallback in a
                # catch-all and skipped the block on any failure; this eigh-
                # based replacement must be just as resilient, not let a
                # pathological block crash the whole prepare().
                #
                # Finding 7: S11 narrowed this to `except la.LinAlgError`,
                # but np.linalg.eigh (and the 0.5*(sub+sub.T) allocation
                # ahead of it) can also raise other exceptions on a
                # pathological block -- e.g. MemoryError on a large block
                # under coordinator memory pressure, or ValueError on
                # NaN/Inf-contaminated input. Restore the catch-all so any
                # such failure degrades to "skip this block" instead of
                # aborting the entire prepare()/factor() call.
                try:
                    # Stage 3: use the ``_ex`` form (returns the raw spectrum
                    # ``w`` too) so GenEO-lite can reuse it below WITHOUT a
                    # second eigh call on the same block ("don't recompute" --
                    # see interface_coarse.py's module docstring and
                    # _spd_safe_pseudo_solve_factor_ex's docstring).
                    eigh_factor_ex = _spd_safe_pseudo_solve_factor_ex(sub)
                except Exception as exc:
                    logger.warning(
                        "Block-Jacobi: eigh-based SPD-safe fallback itself "
                        "failed on block size %d (%s: %s); skipping (falls "
                        "back to identity for this block's nodes).",
                        sub.shape[0], type(exc).__name__, exc,
                    )
                    eigh_factor_ex = None
                if eigh_factor_ex is not None:
                    V_eigh, inv_w_eigh, w_eigh = eigh_factor_ex
                    _block_factors.append(
                        (owned_global_arr, 'eigh', (V_eigh, inv_w_eigh))
                    )
                    if self._want_geneo:
                        try:
                            V_k, w_k = interface_coarse.geneo_lowest_eigenpairs(
                                sub, k=self._geneo_k, tol=self._geneo_tol,
                                precomputed=(w_eigh, V_eigh),
                                island_local_mask=island_local_mask,
                            )
                            if V_k.shape[1] > 0:
                                self._geneo_pairs.append((owned_global_arr, V_k, w_k))
                        except Exception as exc:
                            logger.warning(
                                "Block-Jacobi: GenEO-lite enrichment failed "
                                "on the eigh-fallback block size %d (%s: "
                                "%s); skipping enrichment for this block "
                                "(keeps the eigh factor).",
                                sub.shape[0], type(exc).__name__, exc,
                            )
                else:
                    logger.warning(
                        "Block-Jacobi: block size %d has no positive "
                        "spectrum (lambda_max <= 0); skipping (falls back "
                        "to identity for this block's nodes).",
                        sub.shape[0],
                    )
            else:
                # Finding 2: the cho-factor SUCCESS path lives in this
                # `else:` clause (not inline after cho_factor() inside the
                # try), and the GenEO call below has its OWN narrow
                # try/except -- neither can land in the `except
                # la.LinAlgError` above.  Before this fix, GenEO ran INSIDE
                # the try, AFTER the 'cho' entry was already appended to
                # _block_factors; a LinAlgError raised by GenEO's own dense
                # np.linalg.eigh calls (interface_coarse.py's ARPACK-
                # ArpackNoConvergence/exception fallbacks) was caught by
                # `except la.LinAlgError` above, which appended a SECOND
                # ('eigh') entry for the SAME owned_global_arr -- breaking
                # the block-ownership partition _bj_perm/_bj_offsets rely
                # on (the block would be gathered/applied twice, and
                # result[perm] = y_perm would double-write those nodes).
                _block_factors.append((owned_global_arr, 'cho', cho))
                if self._want_geneo:
                    try:
                        V_k, w_k = interface_coarse.geneo_lowest_eigenpairs(
                            sub, cho=cho, k=self._geneo_k, tol=self._geneo_tol,
                            island_local_mask=island_local_mask,
                        )
                        if V_k.shape[1] > 0:
                            self._geneo_pairs.append((owned_global_arr, V_k, w_k))
                    except Exception as exc:
                        logger.warning(
                            "Block-Jacobi: GenEO-lite enrichment failed on "
                            "block size %d (%s: %s); skipping enrichment "
                            "for this block (keeps the cho factor -- "
                            "block-Jacobi itself is unaffected).",
                            sub.shape[0], type(exc).__name__, exc,
                        )

        if not _block_factors:
            logger.warning(
                "Block-Jacobi: all blocks failed; falling back to no preconditioner."
            )
            return None

        # ---- 4. Materialize each block's dense (pseudo-)inverse, in place --
        # BJ-apply perf fix (Stage 3): the OLD apply (kept as _bj_apply_one's
        # cho_solve/two-GEMV forms up through Stage 2) gathered
        # ``x[global_idx]`` and scattered ``result[global_idx] = ...`` once
        # PER BLOCK -- fancy-index random-access numpy ops that do not
        # release the GIL, measured (Stage 2) to serialize threads down to a
        # 1.4x speedup despite cho_solve's own LAPACK call releasing the
        # GIL fine. Replaced by materializing each block's dense SPD
        # (pseudo-)inverse HERE (cho_solve(cho, eye(k)) for 'cho' blocks; the
        # already-available V @ diag(inv_w) @ V.T for 'eigh' blocks) --
        # REPLACING (not duplicating) the cho-factor payload in
        # _block_factors, so total memory stays Sum(k_i^2)*8 bytes, the same
        # order as before -- so apply becomes ONE global gather + a loop of
        # CONTIGUOUS-slice GEMVs (BLAS-2, releases the GIL) + ONE global
        # scatter (see _bj_solve_serial/_bj_solve_threaded below).
        _bj_perm_parts: List[np.ndarray] = []
        _bj_inv_blocks: List[np.ndarray] = []
        _offsets = [0]
        for _bi, (global_idx, kind, payload) in enumerate(_block_factors):
            k = len(global_idx)
            if kind == 'cho':
                Sinv = la.cho_solve(payload, np.eye(k), check_finite=False)
                # Finding 9: cho_solve's explicitly-formed inverse is not
                # exactly symmetric (unlike the backward-stable triangular
                # solves the old apply-time cho_solve(cho, x) design used
                # directly) -- on the ill-conditioned blocks this feature
                # targets (near-null eigenvalues ~1e-10 relative, cond up to
                # ~1e10 per the module docstring) the asymmetry is large
                # enough to weaken PCG's SPD-preconditioner assumption.
                # Cheap, exact fix: symmetrize in place.  ('eigh' blocks
                # below are already exactly symmetric by construction --
                # V @ diag(inv_w) @ V.T -- so this is only needed here.)
                Sinv = 0.5 * (Sinv + Sinv.T)
            else:
                V, inv_w = payload
                Sinv = (V * inv_w) @ V.T
            Sinv = np.ascontiguousarray(Sinv)
            _block_factors[_bi] = (global_idx, kind, Sinv)
            _bj_perm_parts.append(global_idx)
            _bj_inv_blocks.append(Sinv)
            _offsets.append(_offsets[-1] + k)

        self._bj_block_factors = _block_factors
        self._bj_perm = (
            np.concatenate(_bj_perm_parts) if _bj_perm_parts
            else np.empty(0, dtype=np.int64)
        )
        self._bj_offsets = np.array(_offsets, dtype=np.int64)
        self._bj_inv_blocks = _bj_inv_blocks

        # ---- 5. LPT partition of blocks by k^2 for threaded apply ----------
        # Threading is safe WITHOUT accumulation: block ownership is a
        # partition (each interface node owned by exactly one block), so
        # concurrent threads write to disjoint SLICES of y_perm below --
        # no race, no reduction step needed (item 5).
        # Finding 15: cap the bin count at len(_block_factors) -- see the
        # matching comment in _prepare_tilewise_matvec_state.
        self._bj_n_bins = min(self.matvec_threads, len(_block_factors))
        self._bj_partition = _lpt_partition(
            [float(len(idx)) ** 2 for idx, _, _ in _block_factors],
            self._bj_n_bins,
        )

        def _bj_solve_serial(x: np.ndarray) -> np.ndarray:
            result = x.copy()  # identity for any unowned nodes
            perm = self._bj_perm
            if len(perm):
                offs = self._bj_offsets
                x_perm = x[perm]                      # ONE gather
                y_perm = np.empty_like(x_perm)
                for i, Sinv in enumerate(self._bj_inv_blocks):
                    s, e = offs[i], offs[i + 1]
                    y_perm[s:e] = Sinv @ x_perm[s:e]   # contiguous GEMV
                result[perm] = y_perm                  # ONE scatter
            return result

        def _bj_solve_threaded(x: np.ndarray) -> np.ndarray:
            result = x.copy()  # identity for any unowned nodes
            perm = self._bj_perm
            if len(perm) == 0:
                return result
            offs = self._bj_offsets
            inv_blocks = self._bj_inv_blocks
            x_perm = x[perm]                           # ONE gather
            y_perm = np.empty_like(x_perm)

            def work(t: int) -> None:
                for bi in self._bj_partition[t]:
                    s, e = offs[bi], offs[bi + 1]
                    # Disjoint contiguous slice per block -- no lock needed;
                    # dgemv releases the GIL so this genuinely parallelizes,
                    # unlike the old per-block fancy-index gather/scatter.
                    y_perm[s:e] = inv_blocks[bi] @ x_perm[s:e]

            pool = self._get_pool()
            with threadpoolctl.threadpool_limits(1):
                # Finding 15: dispatch over the CAPPED bin count, not
                # self.matvec_threads -- see the matching comment in
                # _prepare_tilewise_matvec_state.
                list(pool.map(work, range(self._bj_n_bins)))
            result[perm] = y_perm                       # ONE scatter
            return result

        def _bj_solve(x: np.ndarray) -> np.ndarray:
            if self.matvec_threads <= 1 or len(_block_factors) < 2:
                return _bj_solve_serial(x)
            return _bj_solve_threaded(x)

        return spla.LinearOperator(
            shape=(n, n), matvec=_bj_solve, dtype=np.float64
        )

    def _build_jacobi_fallback(self) -> Optional[spla.LinearOperator]:
        """Build a diagonal (Jacobi) preconditioner as a fallback.

        Used when block_jacobi memory exceeds the budget threshold, or as an
        explicit alternative. Cheap and scalable to any interface size.
        """
        n = self.n
        if self.S_global is not None:
            diag = np.array(self.S_global.diagonal(), dtype=np.float64)
        elif self.tile_schur_complements is not None and self.tile_index_maps is not None:
            diag = np.zeros(n, dtype=np.float64)
            for tid, S_i in self.tile_schur_complements.items():
                idx = self.tile_index_maps[tid]
                # Scatter-add diagonal entries using bincount (fast path)
                diag += np.bincount(idx, weights=np.diag(S_i), minlength=n)
            # S4: S_global is None in never-assemble mode -- S = sum_i P_i^T
            # S_i P_i + S_extra, so the true diagonal also needs S_extra's
            # diagonal (island 1e5 penalties + package conductances), not
            # just the summed tile S_i diagonals.  Omitting it under-scales
            # the preconditioner on every islanded/package-coupled node.
            # NB: use self.S_extra (set in __init__ before this builder
            # runs), NOT self._S_extra_csr -- that cache is only populated
            # later, inside _build_linear_op()/_prepare_tilewise_matvec_
            # state(), which __init__ calls AFTER the preconditioner.
            if self.S_extra is not None:
                diag += np.asarray(
                    self.S_extra.tocsr().diagonal(), dtype=np.float64
                )
        else:
            return None  # cannot build any preconditioner

        diag = np.where(diag > 0, diag, 1.0)

        def _diag_solve(x: np.ndarray) -> np.ndarray:
            return x / diag

        return spla.LinearOperator(shape=(n, n), matvec=_diag_solve, dtype=np.float64)

    def _build_amg(self) -> Optional[spla.LinearOperator]:
        """AMG preconditioner via pyamg (optional dependency).

        Requires the assembled S_global (CSR format).  Returns None if
        pyamg is not installed or S_global is not available.
        """
        try:
            import pyamg  # type: ignore[import]
        except ImportError:
            logger.info(
                "pyamg not installed; falling back to block_jacobi preconditioner."
            )
            return self._build_block_jacobi()

        if self.S_global is None:
            logger.warning(
                "AMG preconditioner requires S_global (assembled mode); "
                "falling back to block_jacobi."
            )
            return self._build_block_jacobi()

        try:
            ml = pyamg.smoothed_aggregation_solver(self.S_global.tocsr())
            M_op = ml.aspreconditioner(cycle='V')
            return M_op
        except Exception as exc:
            logger.warning(
                "AMG setup failed (%s); falling back to block_jacobi.", exc
            )
            return self._build_block_jacobi()

    # ------------------------------------------------------------------
    # Solve (callable interface matching direct LU)
    # ------------------------------------------------------------------

    def __call__(self, rhs: np.ndarray) -> np.ndarray:
        """Solve S_global @ x = rhs via CG.

        Args:
            rhs: Right-hand-side vector, shape (n_interface,).

        Returns:
            Solution vector x, shape (n_interface,).

        Raises:
            RuntimeError: If CG fails to converge within maxiter.
        """
        t0 = time.perf_counter()

        x0 = self._x0

        iters: List[int] = [0]
        # Debug/observability knob (set attribute directly, e.g. from a
        # measurement script): log progress every `progress_every` CG
        # iterations, including the TRUE relative residual (costs one extra
        # matvec per report -- e.g. 0.5% overhead at progress_every=200).
        # 0 (default) disables.
        _progress_every = int(getattr(self, 'progress_every', 0) or 0)
        _rhs_norm = float(np.linalg.norm(rhs)) if _progress_every else 0.0

        def _callback(xk: np.ndarray) -> None:
            iters[0] += 1
            if _progress_every and iters[0] % _progress_every == 0:
                rel = (float(np.linalg.norm(rhs - self._linear_op.matvec(xk)))
                       / max(_rhs_norm, 1e-300))
                logger.info(
                    "InterfaceCG progress: iter %d, true rel_res=%.3e, "
                    "elapsed=%.1fs (target rtol=%.1e)",
                    iters[0], rel, time.perf_counter() - t0, self.rtol,
                )

        result, info = spla.cg(
            self._linear_op,
            rhs,
            x0=x0,
            rtol=self.rtol,
            atol=self.atol,
            maxiter=self.maxiter,
            M=self._M,
            callback=_callback,
        )

        elapsed = time.perf_counter() - t0
        n_iter = iters[0]

        if info < 0:
            raise RuntimeError(
                f"InterfaceCGSolver: CG failed with illegal input (info={info})."
            )

        # Accumulate stats before any possible raise so callers that catch the
        # RuntimeError can still inspect stats['cg_failed'] / 'total_cg_failures'.
        self._total_iters += n_iter
        self._total_solves += 1
        self._stats['last_cg_iters'] = n_iter
        self._stats['last_cg_time_s'] = elapsed
        self._stats['last_cg_info'] = info
        self._stats.setdefault('total_cg_iters', 0)
        self._stats['total_cg_iters'] = self._total_iters
        self._stats.setdefault('total_cg_solves', 0)
        self._stats['total_cg_solves'] = self._total_solves

        _failed = info > 0
        self._stats['cg_failed'] = _failed
        self._stats.setdefault('total_cg_failures', 0)

        if info > 0:
            # Compute actual relative residual so the error message is informative.
            r_norm = float(np.linalg.norm(rhs - self._linear_op.matvec(result)))
            b_norm = float(np.linalg.norm(rhs))
            rel_res = r_norm / b_norm if b_norm > 0 else r_norm
            self._stats['last_cg_rel_residual'] = rel_res
            self._stats['total_cg_failures'] += 1

            if self.strict:
                raise RuntimeError(
                    f"InterfaceCGSolver: CG did not converge after {n_iter} "
                    f"iterations (rtol={self.rtol:.2e}, atol={self.atol:.2e}, "
                    f"rel_residual={rel_res:.3e}). "
                    f"Increase maxiter, relax tolerances, or improve the "
                    f"preconditioner. Set strict=False to demote to a warning."
                )
            else:
                logger.warning(
                    "InterfaceCGSolver: CG did not converge after %d iterations "
                    "(rtol=%.2e, rel_residual=%.3e). "
                    "Using best iterate (strict=False).",
                    n_iter, self.rtol, rel_res,
                )

        # Warm-start: update x0 for next call
        self._x0 = result.copy()

        logger.debug(
            "InterfaceCGSolver: %d iters, info=%d, %.4fs (warm=%s)",
            n_iter, info, elapsed, x0 is not None,
        )

        return result

    # ------------------------------------------------------------------
    # Warm-start control
    # ------------------------------------------------------------------

    def reset_warm_start(self) -> None:
        """Clear the warm-start initial guess (force cold start next call)."""
        self._x0 = None

    def set_x0(self, x0: Optional[np.ndarray]) -> None:
        """Explicitly set the warm-start guess for the next solve call."""
        self._x0 = x0 if x0 is None else np.asarray(x0, dtype=np.float64).copy()

    # ------------------------------------------------------------------
    # Stats access
    # ------------------------------------------------------------------

    @property
    def stats(self) -> Dict[str, Any]:
        """Read-only view of accumulated stats (same dict as stats_dict)."""
        return self._stats

    @property
    def total_iterations(self) -> int:
        """Total CG iterations across all calls."""
        return self._total_iters

    @property
    def total_solves(self) -> int:
        """Total solve calls."""
        return self._total_solves


# ---------------------------------------------------------------------------
# Auto-select helper (used by result_factorization)
# ---------------------------------------------------------------------------


def auto_select_interface_solver(
    n_interface: int,
    S_global: Optional[sp.spmatrix] = None,
    n_interface_threshold: int = AUTO_CG_N_INTERFACE_THRESHOLD,
    factor_memory_budget_bytes: int = AUTO_CG_FACTOR_MEMORY_BUDGET_BYTES,
) -> str:
    """Decide whether to use 'direct' or 'cg' for the interface solve.

    Rules:
      1. If n_interface < n_interface_threshold AND estimated factor memory
         fits within factor_memory_budget_bytes: return 'direct'.
      2. Otherwise: return 'cg'.

    The factor memory estimate uses S_global.nnz * 8 * factor_fill_ratio
    (empirical fill ratio 5 for supernodal CHOLMOD on PDN-class matrices).

    Args:
        n_interface: Number of interface unknowns.
        S_global: Assembled Schur matrix (optional; used for memory estimate).
        n_interface_threshold: Cutover size (default 200_000).
        factor_memory_budget_bytes: Memory budget in bytes (default 32 GB).

    Returns:
        'direct' or 'cg'.
    """
    # Finding 4 (SKIPPED -- host-RAM-dependent auto selection is intended
    # Stage 1c behaviour) is mitigated by always logging the resolved
    # budget and resulting mode at INFO level, regardless of which branch
    # is taken, so a host-dependent flip from 'direct' to 'cg' (or vice
    # versa) is visible in the log rather than silent.
    if n_interface < n_interface_threshold:
        if S_global is not None:
            # Rough factor memory: nnz * bytes_per_double * fill_ratio
            fill_ratio = 5.0
            est_bytes = S_global.nnz * 8 * fill_ratio
            if est_bytes > factor_memory_budget_bytes:
                logger.info(
                    "auto_select: n_interface=%d < threshold but estimated "
                    "factor memory %.1f GB > budget %d bytes (%.1f GB); "
                    "resolved_mode=cg.",
                    n_interface,
                    est_bytes / 1024 ** 3,
                    factor_memory_budget_bytes,
                    factor_memory_budget_bytes / 1024 ** 3,
                )
                return 'cg'
            logger.info(
                "auto_select: n_interface=%d < threshold %d, estimated "
                "factor memory %.1f GB fits budget %d bytes (%.1f GB); "
                "resolved_mode=direct.",
                n_interface, n_interface_threshold,
                est_bytes / 1024 ** 3,
                factor_memory_budget_bytes,
                factor_memory_budget_bytes / 1024 ** 3,
            )
        else:
            logger.info(
                "auto_select: n_interface=%d < threshold %d (no S_global "
                "for memory estimate); resolved_mode=direct.",
                n_interface, n_interface_threshold,
            )
        return 'direct'

    logger.info(
        "auto_select: n_interface=%d >= threshold %d; budget=%d bytes "
        "(%.1f GB); resolved_mode=cg.",
        n_interface, n_interface_threshold,
        factor_memory_budget_bytes, factor_memory_budget_bytes / 1024 ** 3,
    )
    return 'cg'


# ---------------------------------------------------------------------------
# Factory: build an interface_lu callable from CG or direct
# ---------------------------------------------------------------------------


def build_interface_solver(
    S_global: Optional[sp.spmatrix],
    interface_solver: str = 'auto',
    tile_schur_complements: Optional[Dict[Any, np.ndarray]] = None,
    tile_index_maps: Optional[Dict[Any, np.ndarray]] = None,
    S_extra: Optional[sp.spmatrix] = None,
    matvec_mode: str = 'auto',
    preconditioner: str = 'block_jacobi',
    rtol: float = 1e-8,
    atol: float = 1e-14,
    maxiter: Optional[int] = None,
    strict: bool = True,
    x0: Optional[np.ndarray] = None,
    block_jacobi_max_bytes: Optional[int] = None,
    factor_memory_budget_bytes: Optional[int] = None,
    n_interface_threshold: int = AUTO_CG_N_INTERFACE_THRESHOLD,
    coordinator_solver_config: Optional[Any] = None,
    verbose: bool = False,
    cg_stats_dict: Optional[Dict[str, Any]] = None,
    n_interface: Optional[int] = None,
    matvec_threads: Any = 'auto',
    matvec_dtype: Any = 'float64',
    strict_dtype_rtol: bool = True,
    island_idx: Optional[np.ndarray] = None,
    # Finding 9: None-sentinel defaults (NOT interface_coarse.DEFAULT_*
    # bound at def/import time) -- forwarded as-is to InterfaceCGSolver,
    # which resolves them dynamically (see its constructor's Finding 9
    # comment). A def-time snapshot here would silently defeat
    # monkeypatch.setattr(interface_coarse, 'DEFAULT_GENEO_K', ...) even
    # after InterfaceCGSolver itself was fixed, since this factory's own
    # default would already be a concrete (stale) int/float by the time
    # InterfaceCGSolver.__init__ ever sees it.
    interface_coarse_geneo_k: Optional[int] = None,
    interface_coarse_geneo_tol: Optional[float] = None,
    interface_coarse_eps_rank: Optional[float] = None,
    interface_coarse_max_cols: Optional[int] = None,
    interface_coarse_max_bytes: Optional[int] = None,
) -> Tuple[Callable, str, Optional['InterfaceCGSolver']]:
    """Build an interface solve callable (direct LU or CG).

    Stage 1d: this is the single entry point for constructing an
    ``InterfaceCGSolver`` (or a direct factor).  ``_factor_dc_context``,
    ``_factor_transient_context``, ``_refactor_dc_context``, and
    ``_refactor_transient_context`` (all in ``result_factorization.py``) are
    routed through this factory so there is exactly one place that builds
    the CG solve callable.

    Args:
        S_global: Assembled sparse Schur matrix (CSR or CSC).  May be
            ``None`` ONLY when ``interface_solver='cg'`` (not 'auto' or
            'direct' -- both need S_global) and ``matvec_mode`` resolves to
            'tilewise' with ``tile_schur_complements`` provided -- the
            "never assemble S_global" path (item 3).  ``n_interface`` must
            be passed explicitly in that case.
        interface_solver: 'direct', 'cg', or 'auto'.
        tile_schur_complements: Per-tile dense Schur blocks (for tilewise).
        tile_index_maps: Per-tile index maps (for tilewise and block_jacobi
            ownership).
        S_extra: Package-edge contribution not captured by per-tile Schur
            complements (tilewise mode only; ignored otherwise).
        matvec_mode: 'auto' (default; item 8 -- 'tilewise' when
            ``tile_schur_complements`` is provided, else 'assembled'),
            'assembled', or 'tilewise' (only used when cg selected).
        preconditioner: 'auto' (Stage 3 -- resolves to 'two_level' when the
            resolved CG matvec mode is 'tilewise', else 'block_jacobi'; see
            :func:`resolve_preconditioner`, the ONE place this resolution
            happens), 'block_jacobi', 'jacobi', 'none', 'amg', or
            'two_level'.
        rtol: CG relative convergence tolerance (default 1e-8).
        atol: CG absolute convergence tolerance floor (default 1e-14).
        maxiter: Max CG iterations (default None -> 3 * n_interface).
        strict: Raise on CG non-convergence (default True).
        x0: Optional initial warm-start guess.
        block_jacobi_max_bytes: Resolved block-Jacobi memory budget (bytes);
            None falls back to the module-level constant.
        factor_memory_budget_bytes: Direct-factor memory budget (bytes),
            used only when ``interface_solver='auto'``.  May be a resolved
            byte count, ``'auto'``, or ``None`` -- all are routed through
            ``resolve_factor_memory_budget_bytes()`` so the factory's own
            'auto' branch is host-aware (matches the host-aware resolution
            already performed by callers), not the raw module-level
            ``AUTO_CG_FACTOR_MEMORY_BUDGET_BYTES`` constant.
        n_interface_threshold: Cutover size used only when
            ``interface_solver='auto'``.
        coordinator_solver_config: SolverBackendConfig (for direct path).
        verbose: Log the resolved backend.
        cg_stats_dict: Mutable dict for CG statistics.
        n_interface: Interface system size.  Required when ``S_global`` is
            ``None`` (the "never assemble" path); otherwise inferred from
            ``S_global.shape[0]`` and used only as a consistency check.
        matvec_threads: Stage 2 ``InterfaceCGSolver`` thread count ('auto'
            default -- see :func:`resolve_matvec_threads`).
        matvec_dtype: Stage 2 tilewise storage dtype ('float64' default).
        strict_dtype_rtol: Enforce the fp32-matvec/rtol pairing (default
            True) -- see :class:`InterfaceCGSolver`.
        island_idx: Stage 3: global interface indices of interface-island
            nodes, for the 'two_level' preconditioner's coarse-space row
            zeroing (ignored otherwise).
        interface_coarse_geneo_k, interface_coarse_geneo_tol,
        interface_coarse_eps_rank, interface_coarse_max_cols,
        interface_coarse_max_bytes: Stage 3 'two_level' knobs -- see
            :class:`InterfaceCGSolver`.

    Returns:
        (solve_callable, resolved_mode, cg_solver_or_none)
        resolved_mode is 'direct' or 'cg'.
        cg_solver_or_none is the InterfaceCGSolver when cg is used, else None.
    """
    if S_global is not None:
        n = S_global.shape[0]
        if n_interface is not None and n_interface != n:
            raise ValueError(
                f"build_interface_solver: n_interface={n_interface} != "
                f"S_global.shape[0]={n}."
            )
    else:
        if n_interface is None:
            raise ValueError(
                "build_interface_solver: n_interface must be provided "
                "explicitly when S_global is None."
            )
        n = n_interface
        if interface_solver != 'cg':
            raise ValueError(
                f"build_interface_solver: S_global is None, which is only "
                f"supported for interface_solver='cg' (got "
                f"{interface_solver!r}) -- 'direct' and 'auto' both need "
                f"S_global for the direct-factor memory estimate/factor."
            )

    # Resolve 'auto'.  Route through resolve_factor_memory_budget_bytes()
    # (host-aware: min(32 GB, 0.4*RAM) via psutil) rather than the raw
    # AUTO_CG_FACTOR_MEMORY_BUDGET_BYTES constant, so the factory's own
    # 'auto' branch behaves identically to callers that pre-resolve the
    # budget externally (finding 12).  factor_memory_budget_bytes may
    # already be a resolved int (idempotent through this call) or an
    # unresolved 'auto'/None passthrough.
    if interface_solver == 'auto':
        _budget = resolve_factor_memory_budget_bytes(
            factor_memory_budget_bytes
            if factor_memory_budget_bytes is not None
            else 'auto'
        )
        resolved = auto_select_interface_solver(
            n, S_global,
            n_interface_threshold=n_interface_threshold,
            factor_memory_budget_bytes=_budget,
        )
    else:
        resolved = interface_solver

    if resolved == 'direct':
        from pgmath.factor import _factor_conductance_matrix
        t0 = time.perf_counter()
        result = _factor_conductance_matrix(
            S_global, verbose=verbose, config=coordinator_solver_config,
        )
        elapsed = time.perf_counter() - t0
        if verbose:
            logger.info(
                "Interface direct factor: %s, %.3fs, n=%d",
                result.backend_info, elapsed, n,
            )
        return result.solve, 'direct', None

    # CG path
    if cg_stats_dict is None:
        cg_stats_dict = {}

    # Item 8: resolve 'auto' matvec_mode -- 'tilewise' when tile Schur
    # blocks are available, else 'assembled'.  Explicit values pass through
    # resolve_matvec_mode() unchanged.
    _has_tile_blocks = bool(tile_schur_complements) and tile_index_maps is not None
    matvec_mode = resolve_matvec_mode(matvec_mode, _has_tile_blocks)

    # For tilewise mode: require tile_schur_complements + tile_index_maps
    # (unless S_global is None, in which case 'assembled' is impossible --
    # this can only happen via a caller bug, since the "never assemble"
    # path always provides tile blocks; raise instead of silently building
    # a solver with no matvec at all).
    if matvec_mode == 'tilewise' and not _has_tile_blocks:
        if S_global is None:
            raise ValueError(
                "build_interface_solver: S_global is None and "
                "tile_schur_complements/tile_index_maps were not provided "
                "-- cannot build any matvec."
            )
        logger.warning(
            "build_interface_solver: matvec_mode='tilewise' requested but "
            "tile_schur_complements or tile_index_maps not provided; "
            "falling back to 'assembled' mode."
        )
        matvec_mode = 'assembled'
    if matvec_mode == 'assembled' and S_global is None:
        raise ValueError(
            "build_interface_solver: matvec_mode resolved to 'assembled' "
            "but S_global is None."
        )

    # Stage 3: the ONE place 'auto' preconditioner resolution happens --
    # 'two_level' when CG + tilewise, else the legacy 'block_jacobi' default.
    # An explicit (non-'auto') caller value passes through unchanged.
    preconditioner = resolve_preconditioner(preconditioner, resolved, matvec_mode)

    t0 = time.perf_counter()
    cg_solver = InterfaceCGSolver(
        n_interface=n,
        matvec_mode=matvec_mode,
        S_global=S_global,
        tile_schur_complements=tile_schur_complements,
        tile_index_maps=tile_index_maps,
        S_extra=S_extra,
        preconditioner=preconditioner,
        rtol=rtol,
        atol=atol,
        maxiter=maxiter,
        x0=x0,
        stats_dict=cg_stats_dict,
        strict=strict,
        block_jacobi_max_bytes=block_jacobi_max_bytes,
        matvec_threads=matvec_threads,
        matvec_dtype=matvec_dtype,
        strict_dtype_rtol=strict_dtype_rtol,
        island_idx=island_idx,
        interface_coarse_geneo_k=interface_coarse_geneo_k,
        interface_coarse_geneo_tol=interface_coarse_geneo_tol,
        interface_coarse_eps_rank=interface_coarse_eps_rank,
        interface_coarse_max_cols=interface_coarse_max_cols,
        interface_coarse_max_bytes=interface_coarse_max_bytes,
    )
    elapsed = time.perf_counter() - t0
    if verbose:
        logger.info(
            "Interface CG solver built: mode=%s, precond=%s, rtol=%.2e, n=%d, "
            "threads=%d, dtype=%s, build_time=%.3fs",
            matvec_mode, cg_solver.preconditioner_label, rtol, n,
            cg_solver.matvec_threads, cg_solver.matvec_dtype, elapsed,
        )

    return cg_solver, 'cg', cg_solver

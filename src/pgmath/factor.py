"""Sparse matrix factorization utilities for power-grid solvers.

This module owns the single source-of-truth for CHOLMOD backend globals,
SparseFactorAdapter, SolverBackendConfig, and _factor_conductance_matrix.
Extracted from solver/unified_solver.py so that distributed code can import
here without pulling in the 3K-line flat solver.

RULE: this module imports ONLY numpy / scipy / stdlib / optional sksparse.
      It must never import from solver/, distributed/, or analysis/.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# ---------------------------------------------------------------------------
# Optional high-performance Cholesky factorization via sksparse/cholmod
# ---------------------------------------------------------------------------
try:
    from sksparse.cholmod import cholesky as _cholmod_cholesky
    HAS_CHOLMOD = True
except ImportError:
    HAS_CHOLMOD = False
    _cholmod_cholesky = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Module-level globals: single source of truth for all CHOLMOD settings.
# unified_solver.py re-exports these functions; callers using either path
# mutate the same module globals here.
# ---------------------------------------------------------------------------

# None = auto-detect (use cholmod if available)
# True = force cholmod (raises ImportError if unavailable)
# False = force splu (SuperLU)
_USE_CHOLMOD: Optional[bool] = None

# 'auto', 'simplicial', or 'supernodal'
_CHOLMOD_MODE: str = 'auto'

# 'default', 'natural', 'amd', 'metis', 'nesdis', 'colamd', 'best'
_CHOLMOD_ORDERING: str = 'default'

# None = auto; True = 64-bit; False = 32-bit
_CHOLMOD_USE_LONG: Optional[bool] = None

_VALID_CHOLMOD_MODES = ('auto', 'simplicial', 'supernodal')
_VALID_CHOLMOD_ORDERINGS = (
    'default', 'natural', 'amd', 'metis', 'nesdis', 'colamd', 'best',
)


# ---------------------------------------------------------------------------
# Getter / setter API (used by tile_worker.py configure(), cli.py, etc.)
# ---------------------------------------------------------------------------

def set_use_cholmod(value: Optional[bool]) -> None:
    """Set the global cholmod usage setting.

    Args:
        value: If True, use cholmod; If False, use splu; If None, auto-detect.

    Raises:
        ImportError: If value=True but sksparse.cholmod is not installed.
    """
    global _USE_CHOLMOD
    if value is True and not HAS_CHOLMOD:
        raise ImportError(
            "Cannot enable cholmod: sksparse.cholmod is not installed. "
            "Install with: pip install scikit-sparse"
        )
    _USE_CHOLMOD = value


def get_use_cholmod() -> Optional[bool]:
    """Get the current global cholmod usage setting."""
    return _USE_CHOLMOD


def set_cholmod_mode(mode: str) -> None:
    """Set the cholmod factorization mode ('auto', 'simplicial', 'supernodal')."""
    global _CHOLMOD_MODE
    if mode not in _VALID_CHOLMOD_MODES:
        raise ValueError(
            f"Invalid cholmod mode '{mode}'. Must be one of: {_VALID_CHOLMOD_MODES}"
        )
    _CHOLMOD_MODE = mode


def get_cholmod_mode() -> str:
    """Get the current cholmod factorization mode."""
    return _CHOLMOD_MODE


def set_cholmod_ordering(ordering: str) -> None:
    """Set the cholmod fill-reducing ordering method."""
    global _CHOLMOD_ORDERING
    if ordering not in _VALID_CHOLMOD_ORDERINGS:
        raise ValueError(
            f"Invalid cholmod ordering '{ordering}'. "
            f"Must be one of: {_VALID_CHOLMOD_ORDERINGS}"
        )
    _CHOLMOD_ORDERING = ordering


def get_cholmod_ordering() -> str:
    """Get the current cholmod fill-reducing ordering method."""
    return _CHOLMOD_ORDERING


def set_cholmod_use_long(value: Optional[bool]) -> None:
    """Set the cholmod index type (None=auto, True=64-bit, False=32-bit)."""
    global _CHOLMOD_USE_LONG
    _CHOLMOD_USE_LONG = value


def get_cholmod_use_long() -> Optional[bool]:
    """Get the current cholmod index type setting."""
    return _CHOLMOD_USE_LONG


def get_active_backend() -> str:
    """Return 'cholmod' or 'splu' depending on current settings."""
    if _USE_CHOLMOD is None:
        return 'cholmod' if HAS_CHOLMOD else 'splu'
    return 'cholmod' if _USE_CHOLMOD else 'splu'


# ---------------------------------------------------------------------------
# SolverBackendConfig
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SolverBackendConfig:
    """Immutable bundle of solver backend settings.

    Used to pass per-role (coordinator vs worker) settings explicitly,
    rather than relying on module-level globals.  Fields with ``None``
    values trigger auto-detection at point of use.  Use
    :meth:`from_globals` to snapshot the current module globals into a
    config.
    """

    use_cholmod: Optional[bool] = None
    cholmod_mode: str = 'auto'
    cholmod_ordering: str = 'default'
    cholmod_use_long: Optional[bool] = None

    def __post_init__(self) -> None:
        if self.cholmod_mode not in _VALID_CHOLMOD_MODES:
            raise ValueError(
                f"Invalid cholmod mode '{self.cholmod_mode}'. "
                f"Must be one of: {_VALID_CHOLMOD_MODES}"
            )
        if self.cholmod_ordering not in _VALID_CHOLMOD_ORDERINGS:
            raise ValueError(
                f"Invalid cholmod ordering '{self.cholmod_ordering}'. "
                f"Must be one of: {_VALID_CHOLMOD_ORDERINGS}"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Return a dict with CHOLMOD backend keys for TileWorker.configure()."""
        import dataclasses as _dc
        return _dc.asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'SolverBackendConfig':
        """Construct from a dict, ignoring unknown keys."""
        return cls(
            use_cholmod=d.get('use_cholmod'),
            cholmod_mode=d.get('cholmod_mode') or 'auto',
            cholmod_ordering=d.get('cholmod_ordering') or 'default',
            cholmod_use_long=d.get('cholmod_use_long'),
        )

    @classmethod
    def from_globals(cls) -> 'SolverBackendConfig':
        """Snapshot the current module-level globals into a config."""
        return cls(
            use_cholmod=_USE_CHOLMOD,
            cholmod_mode=_CHOLMOD_MODE,
            cholmod_ordering=_CHOLMOD_ORDERING,
            cholmod_use_long=_CHOLMOD_USE_LONG,
        )


# ---------------------------------------------------------------------------
# SparseFactorAdapter
# ---------------------------------------------------------------------------

class SparseFactorAdapter:
    """Unified interface for sparse matrix factorization backends.

    Wraps either sksparse.cholmod Cholesky factor or scipy splu factor
    to provide a consistent ``.solve(rhs)`` interface.

    Attributes:
        backend: String indicating which backend is used ('cholmod' or 'splu')
    """

    def __init__(self, factor: Any, backend: str, backend_info: str = ''):
        self._factor = factor
        self._backend = backend
        self._backend_info = backend_info

    @property
    def backend(self) -> str:
        """Return the backend name ('cholmod' or 'splu')."""
        return self._backend

    @property
    def backend_info(self) -> str:
        """Return human-readable backend configuration string."""
        return self._backend_info

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        """Solve A x = rhs using the cached factorization.

        Supports both vector (1D) and matrix (2D) right-hand sides.
        """
        if self._backend == 'cholmod':
            return self._factor.solve_A(rhs)
        else:
            if rhs.ndim == 1:
                return self._factor.solve(rhs)
            else:
                return np.column_stack(
                    [self._factor.solve(rhs[:, i]) for i in range(rhs.shape[1])]
                )

    def __call__(self, rhs: np.ndarray) -> np.ndarray:
        """Allow using the adapter as a callable (backward compatibility)."""
        return self.solve(rhs)


# ---------------------------------------------------------------------------
# _factor_conductance_matrix
# ---------------------------------------------------------------------------

def _factor_conductance_matrix(
    matrix: sp.spmatrix,
    verbose: bool = False,
    use_cholmod: Optional[bool] = None,
    config: Optional[SolverBackendConfig] = None,
) -> SparseFactorAdapter:
    """Factor a conductance matrix using the best available backend.

    Uses sksparse.cholmod for Cholesky factorization if available,
    otherwise falls back to scipy.sparse.linalg.splu.

    The conductance matrix must be symmetric positive definite (SPD).
    In debug mode, symmetry is verified before factorization.

    Args:
        matrix: Sparse conductance matrix (will be converted to CSC if needed)
        verbose: If True, print which backend is being used and timing info
        use_cholmod: If True, force cholmod (raises if unavailable).
                     If False, force splu. If None (default), use the global
                     setting from set_use_cholmod().  Takes precedence over
                     ``config.use_cholmod`` when both are provided.
        config: Optional :class:`SolverBackendConfig` supplying all backend
                settings.  When provided, its fields are used instead of
                module-level globals.  When ``None`` (default), module
                globals are used (backward-compatible behavior).

    Returns:
        SparseFactorAdapter wrapping the factorization

    Raises:
        ImportError: If use_cholmod=True but sksparse not installed
        ValueError: If matrix is not symmetric (in debug mode)
    """
    # Resolve effective settings: explicit use_cholmod > config > globals
    if config is not None:
        eff_mode = config.cholmod_mode
        eff_ordering = config.cholmod_ordering
        eff_use_long = config.cholmod_use_long
        eff_use_cholmod = config.use_cholmod if use_cholmod is None else use_cholmod
    else:
        eff_mode = _CHOLMOD_MODE
        eff_ordering = _CHOLMOD_ORDERING
        eff_use_long = _CHOLMOD_USE_LONG
        eff_use_cholmod = _USE_CHOLMOD if use_cholmod is None else use_cholmod

    # Auto-convert to CSC format (required by both backends)
    t_csc_start = time.perf_counter()
    if not sp.isspmatrix_csc(matrix):
        matrix = matrix.tocsc()
    t_csc_end = time.perf_counter()
    csc_time_ms = (t_csc_end - t_csc_start) * 1000

    # Debug-mode symmetry check
    if __debug__:
        diff = matrix - matrix.T
        sym_err = sp.linalg.norm(diff, 'fro')
        mat_norm = sp.linalg.norm(matrix, 'fro')
        if mat_norm > 0:
            rel_err = sym_err / mat_norm
            assert rel_err < 1e-10, (
                f"Conductance matrix is not symmetric: "
                f"||A - A^T|| / ||A|| = {rel_err:.2e}"
            )

    # Determine which backend to use
    if eff_use_cholmod is None:
        eff_use_cholmod = HAS_CHOLMOD
    elif eff_use_cholmod and not HAS_CHOLMOD:
        raise ImportError(
            "sksparse.cholmod requested but not available. "
            "Install with: pip install scikit-sparse"
        )

    t_factor_start = time.perf_counter()
    if eff_use_cholmod:
        factor = _cholmod_cholesky(
            matrix,
            mode=eff_mode,
            ordering_method=eff_ordering,
            use_long=eff_use_long,
        )
        backend = 'cholmod'
        use_long_str = (
            'auto' if eff_use_long is None else ('64-bit' if eff_use_long else '32-bit')
        )
        backend_info = (
            f"cholmod(mode={eff_mode}, ordering={eff_ordering}, idx={use_long_str})"
        )
    else:
        factor = spla.splu(matrix)
        backend = 'splu'
        backend_info = 'splu'
    t_factor_end = time.perf_counter()
    factor_time_ms = (t_factor_end - t_factor_start) * 1000

    if verbose:
        print(
            f"Matrix factorization ({backend_info}): "
            f"CSC conversion {csc_time_ms:.2f} ms, "
            f"factorization {factor_time_ms:.2f} ms"
        )

    return SparseFactorAdapter(factor, backend, backend_info)

"""Distributed interface assembly and island detection for coupled solving.

SHIM: All implementations have moved to pgmath.schur.
This module re-exports everything for backward compatibility so that all
existing ``from solver.interface_assembly import X`` statements work unchanged.

Split from coupled_system.py for maintainability. All public names are
re-exported from coupled_system so existing imports keep working.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Re-export from pgmath.schur (single canonical source of truth)
# ---------------------------------------------------------------------------
from pgmath.schur import (  # noqa: F401
    assemble_schur_complement_system,
    build_interface_package_matrices,
    find_interface_islands,
    apply_island_penalty,
    detect_interface_islands,
)

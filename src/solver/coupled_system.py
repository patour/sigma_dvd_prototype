"""Block-partitioned matrices and core Schur complement math.

SHIM: All implementations have moved to pgmath.block_system and pgmath.schur.
This module re-exports everything for backward compatibility so that all existing
``from solver.coupled_system import X`` statements continue to work unchanged.

Mathematical Formulation:
------------------------
Given a block-partitioned conductance matrix where:
- t: top-grid interior nodes (above partition layer, excluding ports)
- b: bottom-grid interior nodes (below partition layer)
- p: port nodes (at partition layer)

Top-grid (after eliminating pads):
    G^T = [[G^T_pp, G^T_pt],
           [G^T_tp, G^T_tt]]

Bottom-grid:
    G^B = [[G^B_pp, G^B_pb],
           [G^B_bp, G^B_bb]]

The Schur complement of the bottom-grid interior onto ports is:
    S^B = G^B_pp - G^B_pb * inv(G^B_bb) * G^B_bp
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Re-export from pgmath.block_system
# ---------------------------------------------------------------------------
from pgmath.block_system import (  # noqa: F401
    BlockMatrixSystem,
    extract_block_matrices,
    build_block_system_from_edges,
    compute_reduced_rhs,
    recover_bottom_voltages,
    build_grounded_capacitance_diags,
    set_partial_factor_reg_resistance,
    get_partial_factor_reg_resistance,
    _sparse_mem_bytes,
    _format_bytes,
    _PARTIAL_FACTOR_REG_OHMS,
    _CHUNKED_MAX_MEMORY_GB,
)

# ---------------------------------------------------------------------------
# Re-export Schur complement computation from pgmath.schur
# ---------------------------------------------------------------------------
from pgmath.schur import (  # noqa: F401
    compute_explicit_schur,
    _compute_schur_partial,
    _make_partial_interior_solve,
)

# ---------------------------------------------------------------------------
# Re-exports from split-out modules for backward compatibility.
# All existing ``from solver.coupled_system import X`` continue to work.
# ---------------------------------------------------------------------------

from .coupled_operators import (  # noqa: F401
    SchurComplementOperator,
    CoupledSystemOperator,
    BlockDiagonalPreconditioner,
    ILUPreconditioner,
    AMGPreconditioner,
    HAS_PYAMG,
)

from .interface_assembly import (  # noqa: F401
    assemble_schur_complement_system,
    build_interface_package_matrices,
    find_interface_islands,
    apply_island_penalty,
    detect_interface_islands,
)

"""pgmath — power-grid math primitives.

Shared math for the distributed DDM solver: factorization backends,
block-matrix systems, and Schur complement utilities.

No imports from solver/, distributed/, analysis/, or model/.
"""

from pgmath.factor import (
    HAS_CHOLMOD,
    _VALID_CHOLMOD_MODES,
    _VALID_CHOLMOD_ORDERINGS,
    SolverBackendConfig,
    SparseFactorAdapter,
    _factor_conductance_matrix,
    set_use_cholmod,
    get_use_cholmod,
    set_cholmod_mode,
    get_cholmod_mode,
    set_cholmod_ordering,
    get_cholmod_ordering,
    set_cholmod_use_long,
    get_cholmod_use_long,
    get_active_backend,
)

from pgmath.block_system import (
    BlockMatrixSystem,
    build_block_system_from_edges,
    extract_block_matrices,
    compute_reduced_rhs,
    recover_bottom_voltages,
    build_grounded_capacitance_diags,
    set_partial_factor_reg_resistance,
    get_partial_factor_reg_resistance,
    _sparse_mem_bytes,
    _format_bytes,
    _CHUNKED_MAX_MEMORY_GB,
    _PARTIAL_FACTOR_REG_OHMS,
)

from pgmath.schur import (
    compute_explicit_schur,
    _compute_schur_partial,
    _make_partial_interior_solve,
    assemble_schur_complement_system,
    build_interface_package_matrices,
    find_interface_islands,
    apply_island_penalty,
    detect_interface_islands,
)

__all__ = [
    # factor.py
    'HAS_CHOLMOD',
    '_VALID_CHOLMOD_MODES',
    '_VALID_CHOLMOD_ORDERINGS',
    'SolverBackendConfig',
    'SparseFactorAdapter',
    '_factor_conductance_matrix',
    'set_use_cholmod',
    'get_use_cholmod',
    'set_cholmod_mode',
    'get_cholmod_mode',
    'set_cholmod_ordering',
    'get_cholmod_ordering',
    'set_cholmod_use_long',
    'get_cholmod_use_long',
    'get_active_backend',
    # block_system.py
    'BlockMatrixSystem',
    'build_block_system_from_edges',
    'extract_block_matrices',
    'compute_reduced_rhs',
    'recover_bottom_voltages',
    'build_grounded_capacitance_diags',
    'set_partial_factor_reg_resistance',
    'get_partial_factor_reg_resistance',
    '_sparse_mem_bytes',
    '_format_bytes',
    # schur.py
    'compute_explicit_schur',
    '_compute_schur_partial',
    '_make_partial_interior_solve',
    'assemble_schur_complement_system',
    'build_interface_package_matrices',
    'find_interface_islands',
    'apply_island_penalty',
    'detect_interface_islands',
]

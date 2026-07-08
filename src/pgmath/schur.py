"""Schur complement computation and interface assembly for DDM solving.

Extracted from solver/coupled_system.py and solver/interface_assembly.py.
Contains:
  - _make_partial_interior_solve  (CHOLMOD-based interior solve via truncation trick)
  - _compute_schur_partial        (partial Cholesky Schur complement)
  - compute_explicit_schur        (dispatch: partial Cholesky or chunked multi-RHS)
  - assemble_schur_complement_system   (global DDM interface assembly)
  - build_interface_package_matrices   (package G and C at interface unknowns)
  - find_interface_islands             (BFS island detection)
  - apply_island_penalty               (diagonal penalty for islands)
  - detect_interface_islands           (combined find + apply)

RULE: this module imports ONLY numpy / scipy / stdlib / optional sksparse and
      other pgmath sub-modules.  It must never import from solver/, distributed/,
      analysis/, or model/.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import scipy.sparse as sp

from pgmath.block_system import (
    BlockMatrixSystem,
    _CHUNKED_MAX_MEMORY_GB,
    _format_bytes,
    _sparse_mem_bytes,
    _PARTIAL_FACTOR_REG_OHMS,
)
from pgmath.factor import (
    HAS_CHOLMOD,
    SparseFactorAdapter,
    get_use_cholmod,
    get_cholmod_mode,
    get_cholmod_ordering,
    get_cholmod_use_long,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# _make_partial_interior_solve
# ---------------------------------------------------------------------------

def _make_partial_interior_solve(
    cholmod_factor: Any,
    P_ii: np.ndarray,
    n_interior: int,
    n_ports: int,
) -> Callable[[np.ndarray], np.ndarray]:
    """Create an interior solve closure using the full-matrix Cholesky factor.

    The closure implements the solve_L / solve_Lt truncation trick: given a
    Cholesky factor L of the full permuted matrix ``G_perm = L @ L.T`` with
    interior nodes ordered first and ports last, an interior solve
    ``G_ii^{-1} @ b`` is obtained by:

    1. Permute b to AMD order, pad with port zeros.
    2. Forward-solve: ``y = L^{-1} @ b_pad``.
    3. Zero the port portion of y.
    4. Backward-solve: ``x = L^{-T} @ y``.
    5. Un-permute the interior portion of x.
    """
    inv_P_ii = np.argsort(P_ii)
    n_full = n_interior + n_ports

    def solve(b: np.ndarray) -> np.ndarray:
        if b.ndim == 1:
            b_pad = np.zeros(n_full, dtype=np.float64)
            b_pad[:n_interior] = b[P_ii]
            y = cholmod_factor.solve_L(b_pad, use_LDLt_decomposition=False)
            y[n_interior:] = 0.0
            x = cholmod_factor.solve_Lt(y, use_LDLt_decomposition=False)
            return x[:n_interior][inv_P_ii]
        else:
            if sp.issparse(b):
                b = b.toarray()
            n_rhs = b.shape[1]
            b_pad = np.zeros((n_full, n_rhs), dtype=np.float64)
            b_pad[:n_interior] = b[P_ii]
            y = cholmod_factor.solve_L(b_pad, use_LDLt_decomposition=False)
            y[n_interior:] = 0.0
            x = cholmod_factor.solve_Lt(y, use_LDLt_decomposition=False)
            return x[:n_interior][inv_P_ii]

    return solve


# ---------------------------------------------------------------------------
# _compute_schur_partial
# ---------------------------------------------------------------------------

def _compute_schur_partial(
    block_system: BlockMatrixSystem,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Compute Schur complement via partial Cholesky of the full block matrix.

    Factors ``G_full = [[G_ii, G_ip], [G_pi, G_pp]]`` with interior nodes in a
    fill-reducing (AMD/METIS) order and ports appended last.  The Schur
    complement is then ``S = L22 @ L22.T`` where L22 is the lower-right
    ``n_ports x n_ports`` block of the Cholesky factor L.

    As a side-effect, sets ``block_system.lu_ii`` and
    ``block_system.factor_adapter``.
    """
    from sksparse.cholmod import cholesky as cholmod_cholesky, analyze as cholmod_analyze

    n_i = block_system.n_interior
    n_p = block_system.n_ports
    ordering = get_cholmod_ordering()
    mode = get_cholmod_mode()
    use_long = get_cholmod_use_long()

    logger.debug(
        "_compute_schur_partial: n_interior=%d, n_ports=%d, ordering=%s, mode=%s",
        n_i, n_p, ordering, mode,
    )

    # Phase 1: Analyze G_ii to obtain fill-reducing permutation
    t0 = time.perf_counter()
    G_ii_csc = block_system.G_ii.tocsc()
    symbolic = cholmod_analyze(
        G_ii_csc, mode=mode, ordering_method=ordering, use_long=use_long,
    )
    P_ii = symbolic.P()
    t_analyze = time.perf_counter() - t0

    # Phase 2: Build full permuted matrix and factor
    t0 = time.perf_counter()
    G_full = sp.bmat(
        [[block_system.G_ii, block_system.G_ip],
         [block_system.G_pi, block_system.G_pp]],
        format='csc',
    )
    full_perm = np.concatenate([P_ii, n_i + np.arange(n_p)])
    G_perm = G_full[np.ix_(full_perm, full_perm)].tocsc()

    # Regularize port diagonals (tiles without ground connection are PSD, not SPD)
    R_TO_KOHM = 1e-3
    r_kohm = _PARTIAL_FACTOR_REG_OHMS * R_TO_KOHM
    g_reg = 1.0 / r_kohm
    diag = G_perm.diagonal()
    diag[n_i:] += g_reg
    G_perm.setdiag(diag)

    factor = cholmod_cholesky(
        G_perm, mode=mode, ordering_method='natural', use_long=use_long,
    )
    t_factor = time.perf_counter() - t0

    # Phase 3: Extract L22 and compute S
    t0 = time.perf_counter()
    L = factor.L()
    L22 = L[n_i:, n_i:].toarray()
    S = L22 @ L22.T
    np.fill_diagonal(S, S.diagonal() - g_reg)
    t_extract = time.perf_counter() - t0

    # Side-effects: set lu_ii and factor_adapter on block_system
    block_system.lu_ii = _make_partial_interior_solve(factor, P_ii, n_i, n_p)
    use_long_str = 'auto' if use_long is None else ('64-bit' if use_long else '32-bit')
    info_str = (
        f"cholmod(partial, mode={mode}, ordering={ordering}, "
        f"natural, idx={use_long_str})"
    )
    block_system.factor_adapter = SparseFactorAdapter(factor, 'cholmod', info_str)

    stats = {
        'path': 'partial_cholesky',
        'ordering': ordering,
        'mode': mode,
        'chunk_size': 0,
        'n_chunks': 0,
        'schur_mem_bytes': S.nbytes,
        'full_factor_nnz': L.nnz,
        'L22_dense_shape': L22.shape,
        'analyze_s': t_analyze,
        'factor_s': t_factor,
        'extract_s': t_extract,
    }
    logger.debug(
        "_compute_schur_partial: S %dx%d (%s), L nnz=%d, "
        "analyze %.3fs, factor %.3fs, extract %.3fs",
        n_p, n_p, _format_bytes(S.nbytes), L.nnz,
        t_analyze, t_factor, t_extract,
    )
    return S, stats


# ---------------------------------------------------------------------------
# compute_explicit_schur
# ---------------------------------------------------------------------------

def compute_explicit_schur(
    block_system: BlockMatrixSystem,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Compute explicit Schur complement S = G_pp - G_pi * inv(G_ii) * G_ip.

    Two computation paths are available, selected automatically:

    **Partial Cholesky** (when CHOLMOD backend is active): Factors the full
    block matrix ``[[G_ii, G_ip], [G_pi, G_pp]]`` with a fill-reducing
    interior ordering and ports last, then extracts S = L22 @ L22.T.
    Sets ``lu_ii`` as a side-effect.

    **Chunked multi-RHS** (when splu backend is active): Solves
    ``G_ii^{-1} @ G_ip`` column-by-column in chunks. Calls
    ``factor_interior()`` internally if ``lu_ii`` is not already set.
    """
    n_ports = block_system.n_ports
    n_interior = block_system.n_interior

    if n_interior == 0:
        S = block_system.G_pp.toarray()
        return S, {'path': 'trivial', 'schur_mem_bytes': S.nbytes}

    use_partial = (HAS_CHOLMOD and get_use_cholmod() is not False)

    if use_partial:
        return _compute_schur_partial(block_system)

    # Chunked multi-RHS path (splu fallback)
    t_factor = 0.0
    if block_system.lu_ii is None:
        t_f0 = time.perf_counter()
        block_system.factor_interior()
        t_factor = time.perf_counter() - t_f0

    INT_MAX = 2**31 - 1
    bytes_per_col = n_interior * 8 * 0.5
    memory_chunk = max(1, int(_CHUNKED_MAX_MEMORY_GB * 1e9 / bytes_per_col))
    index_chunk = max(1, INT_MAX // max(n_interior, 1))
    chunk_size = min(memory_chunk, index_chunk, n_ports, 256)
    chunk_size = max(chunk_size, min(32, n_ports))

    n_chunks = (n_ports + chunk_size - 1) // chunk_size

    logger.debug(
        "compute_explicit_schur: n_ports=%d, n_interior=%d, chunk_size=%d, n_chunks=%d",
        n_ports, n_interior, chunk_size, n_chunks,
    )

    S = block_system.G_pp.toarray()
    G_ip_csc = block_system.G_ip.tocsc()
    for start in range(0, n_ports, chunk_size):
        end = min(start + chunk_size, n_ports)
        G_ip_chunk = G_ip_csc[:, start:end].toarray()
        Z_chunk = block_system.lu_ii(G_ip_chunk)
        chunk_result = block_system.G_pi @ Z_chunk
        S[:, start:end] -= chunk_result.toarray() if sp.issparse(chunk_result) else chunk_result

    stats = {
        'path': 'chunked',
        'factor_s': t_factor,
        'chunk_size': chunk_size,
        'n_chunks': n_chunks,
        'schur_mem_bytes': S.nbytes,
    }
    logger.debug(
        "compute_explicit_schur: Schur %dx%d, memory %s",
        n_ports, n_ports, _format_bytes(S.nbytes),
    )
    return S, stats


# ---------------------------------------------------------------------------
# assemble_schur_complement_system  (from solver/interface_assembly.py)
# ---------------------------------------------------------------------------

def assemble_schur_complement_system(
    tile_schur_complements: Dict[Any, np.ndarray],
    tile_port_node_lists: Dict[Any, List[str]],
    extra_edges: Optional[List[Tuple[str, str, float]]] = None,
    dirichlet_nodes: Optional[Set[str]] = None,
    dirichlet_voltage: float = 0.0,
    ground_node: str = '0',
) -> Tuple[sp.csr_matrix, np.ndarray, List[str], Dict[str, int]]:
    """Assemble global Schur complement system from per-subdomain contributions.

    Standard finite-element scatter-add:
        S_global = sum_i P_i^T @ S_i @ P_i + G_extra

    Uses COO format for assembly, converts to CSR for factorization.
    """
    if dirichlet_nodes is None:
        dirichlet_nodes = set()

    all_interface_nodes: Set[str] = set()
    for tile_id, node_list in tile_port_node_lists.items():
        all_interface_nodes.update(node_list)

    if extra_edges:
        for u, v, g in extra_edges:
            if u != ground_node:
                all_interface_nodes.add(u)
            if v != ground_node:
                all_interface_nodes.add(v)

    dirichlet_set = dirichlet_nodes & all_interface_nodes
    unknown_nodes = all_interface_nodes - dirichlet_set

    unknown_list = sorted(unknown_nodes)
    dirichlet_list = sorted(dirichlet_set)

    all_ordered = unknown_list + dirichlet_list
    full_node_to_idx = {n: i for i, n in enumerate(all_ordered)}

    n_unknown = len(unknown_list)
    n_dirichlet = len(dirichlet_list)
    n_full = n_unknown + n_dirichlet

    if n_full == 0:
        return (
            sp.csr_matrix((0, 0)),
            np.zeros(0, dtype=np.float64),
            [],
            {},
        )

    coo_rows, coo_cols, coo_data = [], [], []

    # 1. Scatter-add tile Schur complements
    for tile_id, S_i in tile_schur_complements.items():
        node_list = tile_port_node_lists[tile_id]
        n_local = len(node_list)
        assert S_i.shape == (n_local, n_local), (
            f"Tile {tile_id}: Schur shape {S_i.shape} != ({n_local}, {n_local})"
        )

        local_to_global = np.array(
            [full_node_to_idx[n] for n in node_list], dtype=np.int32
        )

        gi_grid = np.repeat(local_to_global, n_local)
        gj_grid = np.tile(local_to_global, n_local)
        vals = S_i.ravel()

        nonzero = vals != 0.0
        coo_rows.extend(gi_grid[nonzero].tolist())
        coo_cols.extend(gj_grid[nonzero].tolist())
        coo_data.extend(vals[nonzero].tolist())

    # 2. Add extra edges (package conductances)
    if extra_edges:
        for u, v, g in extra_edges:
            if g <= 0:
                continue
            if u == ground_node:
                if v in full_node_to_idx:
                    iv = full_node_to_idx[v]
                    coo_rows.append(iv)
                    coo_cols.append(iv)
                    coo_data.append(g)
                continue
            if v == ground_node:
                if u in full_node_to_idx:
                    iu = full_node_to_idx[u]
                    coo_rows.append(iu)
                    coo_cols.append(iu)
                    coo_data.append(g)
                continue
            if u not in full_node_to_idx or v not in full_node_to_idx:
                continue
            iu, iv = full_node_to_idx[u], full_node_to_idx[v]
            coo_rows.extend([iu, iv, iu, iv])
            coo_cols.extend([iv, iu, iu, iv])
            coo_data.extend([-g, -g, g, g])

    if coo_data:
        G_full = sp.coo_matrix(
            (coo_data, (coo_rows, coo_cols)), shape=(n_full, n_full)
        ).tocsr()
    else:
        G_full = sp.csr_matrix((n_full, n_full))

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

    return S_global, rhs_dirichlet, unknown_list, unknown_to_idx


# ---------------------------------------------------------------------------
# build_interface_package_matrices  (from solver/interface_assembly.py)
# ---------------------------------------------------------------------------

def build_interface_package_matrices(
    package_edges: List[Tuple[str, str, float]],
    package_cap_edges: List[Tuple[str, str, float]],
    interface_node_to_idx: Dict[str, int],
    n_interface: int,
    dirichlet_nodes: Set[str],
    ground_node: str = '0',
) -> Tuple[sp.csr_matrix, sp.csr_matrix]:
    """Build package G and C matrices at interface unknowns.

    Package resistors produce ``G_package_uu`` following standard nodal
    conductance stamping.  Package capacitors CAN be coupling (non-grounded),
    so ``C_package_uu`` is a general sparse matrix.
    """
    shape = (n_interface, n_interface)

    # Build G_package_uu
    g_rows: List[int] = []
    g_cols: List[int] = []
    g_data: List[float] = []

    for u, v, g in package_edges:
        if g <= 0:
            continue
        if u == ground_node:
            if v in interface_node_to_idx:
                iv = interface_node_to_idx[v]
                g_rows.append(iv); g_cols.append(iv); g_data.append(g)
            continue
        if v == ground_node:
            if u in interface_node_to_idx:
                iu = interface_node_to_idx[u]
                g_rows.append(iu); g_cols.append(iu); g_data.append(g)
            continue

        u_is_unknown = u in interface_node_to_idx
        v_is_unknown = v in interface_node_to_idx
        u_is_dirichlet = u in dirichlet_nodes
        v_is_dirichlet = v in dirichlet_nodes

        if u_is_unknown and v_is_unknown:
            iu, iv = interface_node_to_idx[u], interface_node_to_idx[v]
            g_rows.extend([iu, iv, iu, iv])
            g_cols.extend([iv, iu, iu, iv])
            g_data.extend([-g, -g, g, g])
        elif u_is_unknown and v_is_dirichlet:
            iu = interface_node_to_idx[u]
            g_rows.append(iu); g_cols.append(iu); g_data.append(g)
        elif v_is_unknown and u_is_dirichlet:
            iv = interface_node_to_idx[v]
            g_rows.append(iv); g_cols.append(iv); g_data.append(g)

    if g_data:
        G_package_uu = sp.coo_matrix(
            (g_data, (g_rows, g_cols)), shape=shape
        ).tocsr()
    else:
        G_package_uu = sp.csr_matrix(shape)

    # Build C_package_uu
    c_rows: List[int] = []
    c_cols: List[int] = []
    c_data: List[float] = []

    for u, v, c_fF in package_cap_edges:
        if c_fF <= 0:
            continue

        if u == ground_node and v == ground_node:
            continue
        if u == ground_node:
            if v in interface_node_to_idx:
                iv = interface_node_to_idx[v]
                c_rows.append(iv); c_cols.append(iv); c_data.append(c_fF)
            continue
        if v == ground_node:
            if u in interface_node_to_idx:
                iu = interface_node_to_idx[u]
                c_rows.append(iu); c_cols.append(iu); c_data.append(c_fF)
            continue

        u_is_unknown = u in interface_node_to_idx
        v_is_unknown = v in interface_node_to_idx
        u_is_dirichlet = u in dirichlet_nodes
        v_is_dirichlet = v in dirichlet_nodes

        if u_is_dirichlet and v_is_dirichlet:
            continue

        if u_is_unknown and v_is_unknown:
            iu, iv = interface_node_to_idx[u], interface_node_to_idx[v]
            c_rows.extend([iu, iv, iu, iv])
            c_cols.extend([iu, iv, iv, iu])
            c_data.extend([c_fF, c_fF, -c_fF, -c_fF])
        elif u_is_unknown and (v_is_dirichlet or v == ground_node):
            iu = interface_node_to_idx[u]
            c_rows.append(iu); c_cols.append(iu); c_data.append(c_fF)
        elif v_is_unknown and (u_is_dirichlet or u == ground_node):
            iv = interface_node_to_idx[v]
            c_rows.append(iv); c_cols.append(iv); c_data.append(c_fF)

    if c_data:
        C_package_uu = sp.coo_matrix(
            (c_data, (c_rows, c_cols)), shape=shape
        ).tocsr()
    else:
        C_package_uu = sp.csr_matrix(shape)

    return G_package_uu, C_package_uu


# ---------------------------------------------------------------------------
# find_interface_islands  (from solver/interface_assembly.py)
# ---------------------------------------------------------------------------

def find_interface_islands(
    S_global: sp.csr_matrix,
    interface_nodes: List[str],
    interface_node_to_idx: Dict[str, int],
    pad_nodes: Set[str],
    extra_edges: Optional[List[Tuple[str, str, float]]] = None,
) -> Set[str]:
    """Detect interface nodes with no resistive path to any pad node.

    Builds an adjacency graph from S_global off-diagonal nonzeros plus any
    virtual pad nodes reachable through extra_edges.  Runs BFS and returns
    nodes in components that contain no pad node.
    """
    n = len(interface_nodes)
    if n == 0:
        return set()

    adj: Dict[str, Set[str]] = {node: set() for node in interface_nodes}

    S_coo = S_global.tocoo()
    for r, c in zip(S_coo.row, S_coo.col):
        if r != c:
            u_name = interface_nodes[r]
            v_name = interface_nodes[c]
            adj[u_name].add(v_name)
            adj[v_name].add(u_name)

    virtual_pads: Set[str] = set()
    if extra_edges:
        for u, v, g in extra_edges:
            if g <= 0:
                continue
            if u == '0' or v == '0':
                continue

            u_is_pad = u in pad_nodes
            v_is_pad = v in pad_nodes
            u_is_iface = u in interface_node_to_idx
            v_is_iface = v in interface_node_to_idx

            if u_is_iface and v_is_iface:
                adj[u].add(v)
                adj[v].add(u)
            elif u_is_iface and v_is_pad:
                if v not in adj:
                    adj[v] = set()
                adj[u].add(v)
                adj[v].add(u)
                virtual_pads.add(v)
            elif v_is_iface and u_is_pad:
                if u not in adj:
                    adj[u] = set()
                adj[v].add(u)
                adj[u].add(v)
                virtual_pads.add(u)

    all_graph_nodes = set(interface_nodes) | virtual_pads
    visited: Set[str] = set()
    island_nodes: Set[str] = set()

    for start in all_graph_nodes:
        if start in visited:
            continue

        component: Set[str] = set()
        has_pad = False
        queue = deque([start])
        visited.add(start)

        while queue:
            node = queue.popleft()
            component.add(node)
            if node in pad_nodes:
                has_pad = True
            for neighbor in adj.get(node, set()):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        if not has_pad:
            island_nodes.update(component & set(interface_nodes))

    return island_nodes


# ---------------------------------------------------------------------------
# apply_island_penalty  (from solver/interface_assembly.py)
# ---------------------------------------------------------------------------

def apply_island_penalty(
    S_global: sp.csr_matrix,
    rhs_dirichlet: np.ndarray,
    island_nodes: Set[str],
    interface_node_to_idx: Dict[str, int],
    dirichlet_voltage: float,
    penalty_conductance: float = 1e5,
) -> Tuple[sp.csr_matrix, np.ndarray]:
    """Apply a diagonal penalty to island nodes to make S_global non-singular."""
    if not island_nodes:
        return S_global, rhs_dirichlet

    indices = np.array(
        [interface_node_to_idx[n] for n in island_nodes], dtype=np.intp
    )

    n = S_global.shape[0]
    penalty_vals = np.full(len(indices), penalty_conductance, dtype=np.float64)
    penalty_matrix = sp.coo_matrix(
        (penalty_vals, (indices, indices)), shape=(n, n)
    ).tocsr()

    S_fixed = S_global + penalty_matrix

    rhs_fixed = rhs_dirichlet.copy()
    rhs_fixed[indices] += penalty_conductance * dirichlet_voltage

    return S_fixed, rhs_fixed


# ---------------------------------------------------------------------------
# detect_interface_islands  (from solver/interface_assembly.py)
# ---------------------------------------------------------------------------

def detect_interface_islands(
    S_global: sp.csr_matrix,
    rhs_dirichlet: np.ndarray,
    interface_nodes: List[str],
    interface_node_to_idx: Dict[str, int],
    pad_nodes: Set[str],
    extra_edges: Optional[List[Tuple[str, str, float]]] = None,
    *,
    dirichlet_voltage: float,
    penalty_conductance: float = 1e5,
) -> Tuple[sp.csr_matrix, np.ndarray, Set[str]]:
    """Detect interface islands and apply diagonal penalty in one call.

    Convenience wrapper that calls :func:`find_interface_islands` followed
    by :func:`apply_island_penalty`.  If no islands are found, the inputs
    are returned unchanged (zero-copy fast path).
    """
    island_nodes = find_interface_islands(
        S_global, interface_nodes, interface_node_to_idx,
        pad_nodes, extra_edges,
    )

    if not island_nodes:
        return S_global, rhs_dirichlet, set()

    S_fixed, rhs_fixed = apply_island_penalty(
        S_global, rhs_dirichlet, island_nodes,
        interface_node_to_idx, dirichlet_voltage, penalty_conductance,
    )

    return S_fixed, rhs_fixed, island_nodes

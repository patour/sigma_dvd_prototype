"""Distributed interface assembly and island detection for coupled solving.

Contains the global Schur complement assembly (scatter-add from per-tile
contributions) and interface island detection/penalty logic.

Split from coupled_system.py for maintainability. All public names are
re-exported from coupled_system so existing imports keep working.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import scipy.sparse as sp


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

    Where P_i maps tile-local port indices to global indices.
    extra_edges adds conductance from non-tile sources (e.g., package resistors).
    Dirichlet nodes are eliminated via RHS contribution.

    Uses COO format for assembly, converts to CSR for factorization.

    Args:
        tile_schur_complements: Dict mapping tile_id -> dense Schur complement (n_ports_i x n_ports_i)
        tile_port_node_lists: Dict mapping tile_id -> ordered list of port node names
        extra_edges: Optional list of (u, v, conductance) for non-tile conductances
        dirichlet_nodes: Optional set of nodes with fixed voltage
        dirichlet_voltage: Voltage at Dirichlet nodes (default 0.0)
        ground_node: Ground node name (default '0')

    Returns:
        Tuple of:
        - S_global: CSR matrix of the assembled interface system (unknowns only)
        - rhs_dirichlet: Dirichlet RHS contribution for unknowns
        - ordered_node_list: Ordered list of unknown node names
        - node_to_idx_map: Dict mapping node name -> index in unknowns
    """
    if dirichlet_nodes is None:
        dirichlet_nodes = set()

    # Collect all interface nodes from tiles and extra edges
    all_interface_nodes: Set[str] = set()
    for tile_id, node_list in tile_port_node_lists.items():
        all_interface_nodes.update(node_list)

    if extra_edges:
        for u, v, g in extra_edges:
            if u != ground_node:
                all_interface_nodes.add(u)
            if v != ground_node:
                all_interface_nodes.add(v)

    # Classify: unknowns vs Dirichlet
    dirichlet_set = dirichlet_nodes & all_interface_nodes
    unknown_nodes = all_interface_nodes - dirichlet_set

    unknown_list = sorted(unknown_nodes)
    dirichlet_list = sorted(dirichlet_set)

    # Full ordering: unknowns first, then Dirichlet
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

    # Build full system in COO format (unknowns + Dirichlet), then extract blocks
    coo_rows, coo_cols, coo_data = [], [], []

    # 1. Scatter-add tile Schur complements
    for tile_id, S_i in tile_schur_complements.items():
        node_list = tile_port_node_lists[tile_id]
        n_local = len(node_list)
        assert S_i.shape == (n_local, n_local), (
            f"Tile {tile_id}: Schur shape {S_i.shape} != ({n_local}, {n_local})"
        )

        # Map local indices to global
        local_to_global = np.array(
            [full_node_to_idx[n] for n in node_list], dtype=np.int32
        )

        # Vectorized scatter: expand local_to_global into row/col index grids
        gi_grid = np.repeat(local_to_global, n_local)
        gj_grid = np.tile(local_to_global, n_local)
        vals = S_i.ravel()

        # Filter out zeros
        nonzero = vals != 0.0
        coo_rows.extend(gi_grid[nonzero].tolist())
        coo_cols.extend(gj_grid[nonzero].tolist())
        coo_data.extend(vals[nonzero].tolist())

    # 2. Add extra edges (package conductances)
    if extra_edges:
        for u, v, g in extra_edges:
            if g <= 0:
                continue

            # Ground handling: diagonal only
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

    # Build full matrix
    if coo_data:
        G_full = sp.coo_matrix(
            (coo_data, (coo_rows, coo_cols)), shape=(n_full, n_full)
        ).tocsr()
    else:
        G_full = sp.csr_matrix((n_full, n_full))

    # Extract unknown-unknown block
    u_idx = np.arange(n_unknown)
    S_global = G_full[np.ix_(u_idx, u_idx)].tocsr()

    # Dirichlet RHS: rhs = -G_ud * V_d
    if n_dirichlet > 0:
        d_idx = np.arange(n_unknown, n_full)
        G_ud = G_full[np.ix_(u_idx, d_idx)].tocsr()
        V_d = np.full(n_dirichlet, dirichlet_voltage, dtype=np.float64)
        rhs_dirichlet = -(G_ud @ V_d)
    else:
        rhs_dirichlet = np.zeros(n_unknown, dtype=np.float64)

    unknown_to_idx = {n: i for i, n in enumerate(unknown_list)}

    return S_global, rhs_dirichlet, unknown_list, unknown_to_idx


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
    conductance stamping (same pattern used inside
    :func:`assemble_schur_complement_system` for ``extra_edges``).

    Package capacitors CAN be coupling (non-grounded), so ``C_package_uu``
    is a general sparse matrix (not necessarily diagonal).  The C matrix is
    symmetric and built via standard nodal analysis stamping:

    * Grounded cap at node *i*:  ``C[i, i] += c_fF``
    * Coupling cap between nodes *i* and *j*:
      ``C[i, i] += c_fF``, ``C[j, j] += c_fF``,
      ``C[i, j] -= c_fF``, ``C[j, i] -= c_fF``

    Edges where both nodes are Dirichlet or ground are skipped.  Edges with
    one Dirichlet terminal contribute only to the unknown node's diagonal.

    Args:
        package_edges: List of ``(u, v, conductance)`` for package resistors.
        package_cap_edges: List of ``(u, v, capacitance_fF)`` for package caps.
        interface_node_to_idx: Dict mapping unknown interface node name to
            row/column index in the output matrices.
        n_interface: Number of unknown interface nodes (matrix dimension).
        dirichlet_nodes: Set of Dirichlet (pad) node names that have been
            eliminated from the unknown system.
        ground_node: Name of the ground node (default ``'0'``).

    Returns:
        Tuple of ``(G_package_uu, C_package_uu)`` sparse CSR matrices, each
        of shape ``(n_interface, n_interface)``.
    """
    shape = (n_interface, n_interface)

    # ---- Build G_package_uu from resistive package edges ----
    g_rows: List[int] = []
    g_cols: List[int] = []
    g_data: List[float] = []

    for u, v, g in package_edges:
        if g <= 0:
            continue

        # Ground handling: diagonal only on the non-ground node
        if u == ground_node:
            if v in interface_node_to_idx:
                iv = interface_node_to_idx[v]
                g_rows.append(iv)
                g_cols.append(iv)
                g_data.append(g)
            continue
        if v == ground_node:
            if u in interface_node_to_idx:
                iu = interface_node_to_idx[u]
                g_rows.append(iu)
                g_cols.append(iu)
                g_data.append(g)
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
            # Dirichlet terminal eliminated; only diagonal contribution
            iu = interface_node_to_idx[u]
            g_rows.append(iu)
            g_cols.append(iu)
            g_data.append(g)
        elif v_is_unknown and u_is_dirichlet:
            iv = interface_node_to_idx[v]
            g_rows.append(iv)
            g_cols.append(iv)
            g_data.append(g)
        # Both Dirichlet or neither in system: skip

    if g_data:
        G_package_uu = sp.coo_matrix(
            (g_data, (g_rows, g_cols)), shape=shape
        ).tocsr()
    else:
        G_package_uu = sp.csr_matrix(shape)

    # ---- Build C_package_uu from capacitive package edges ----
    c_rows: List[int] = []
    c_cols: List[int] = []
    c_data: List[float] = []

    for u, v, c_fF in package_cap_edges:
        if c_fF <= 0:
            continue

        # Ground handling: grounded cap -> diagonal only on the non-ground node
        if u == ground_node and v == ground_node:
            continue
        if u == ground_node:
            if v in interface_node_to_idx:
                iv = interface_node_to_idx[v]
                c_rows.append(iv)
                c_cols.append(iv)
                c_data.append(c_fF)
            continue
        if v == ground_node:
            if u in interface_node_to_idx:
                iu = interface_node_to_idx[u]
                c_rows.append(iu)
                c_cols.append(iu)
                c_data.append(c_fF)
            continue

        u_is_unknown = u in interface_node_to_idx
        v_is_unknown = v in interface_node_to_idx
        u_is_dirichlet = u in dirichlet_nodes
        v_is_dirichlet = v in dirichlet_nodes

        # Both Dirichlet: skip entirely
        if u_is_dirichlet and v_is_dirichlet:
            continue

        if u_is_unknown and v_is_unknown:
            # Coupling cap between two unknowns: full nodal stamp
            iu, iv = interface_node_to_idx[u], interface_node_to_idx[v]
            c_rows.extend([iu, iv, iu, iv])
            c_cols.extend([iu, iv, iv, iu])
            c_data.extend([c_fF, c_fF, -c_fF, -c_fF])
        elif u_is_unknown and (v_is_dirichlet or v == ground_node):
            # One terminal is Dirichlet/ground: only diagonal on unknown
            iu = interface_node_to_idx[u]
            c_rows.append(iu)
            c_cols.append(iu)
            c_data.append(c_fF)
        elif v_is_unknown and (u_is_dirichlet or u == ground_node):
            iv = interface_node_to_idx[v]
            c_rows.append(iv)
            c_cols.append(iv)
            c_data.append(c_fF)
        # Both outside the system: skip

    if c_data:
        C_package_uu = sp.coo_matrix(
            (c_data, (c_rows, c_cols)), shape=shape
        ).tocsr()
    else:
        C_package_uu = sp.csr_matrix(shape)

    return G_package_uu, C_package_uu


def find_interface_islands(
    S_global: sp.csr_matrix,
    interface_nodes: List[str],
    interface_node_to_idx: Dict[str, int],
    pad_nodes: Set[str],
    extra_edges: Optional[List[Tuple[str, str, float]]] = None,
) -> Set[str]:
    """Detect interface nodes with no resistive path to any pad node.

    Builds an adjacency graph from the off-diagonal nonzeros of S_global
    plus any virtual pad nodes reachable through extra_edges. Runs BFS
    connected-component analysis and returns nodes in components that
    contain no pad node (i.e., "islands" that would make S_global singular).

    Args:
        S_global: Assembled global Schur complement (CSR, n_unknown x n_unknown).
        interface_nodes: Ordered list of unknown node names matching S_global rows.
        interface_node_to_idx: Dict mapping node name -> index in interface_nodes.
        pad_nodes: Set of pad (Dirichlet) node names. These are *not* in S_global
            but may appear in extra_edges.
        extra_edges: Optional list of (u, v, conductance) tuples from package
            resistors or other non-tile sources. Pad endpoints create virtual
            adjacency into the interface graph.

    Returns:
        Set of island node names (interface unknowns with no path to any pad).
        Empty set if all interface nodes can reach at least one pad.
    """
    n = len(interface_nodes)
    if n == 0:
        return set()

    # Build adjacency list: interface unknowns + virtual pad nodes
    adj: Dict[str, Set[str]] = {node: set() for node in interface_nodes}

    # 1. Adjacency from S_global off-diagonal nonzeros
    S_coo = S_global.tocoo()
    for r, c in zip(S_coo.row, S_coo.col):
        if r != c:
            u_name = interface_nodes[r]
            v_name = interface_nodes[c]
            adj[u_name].add(v_name)
            adj[v_name].add(u_name)

    # 2. Adjacency from extra_edges (introduces virtual pad nodes)
    virtual_pads: Set[str] = set()
    if extra_edges:
        for u, v, g in extra_edges:
            if g <= 0:
                continue
            # Skip ground node (same pattern as tile_worker.py)
            if u == '0' or v == '0':
                continue

            u_is_pad = u in pad_nodes
            v_is_pad = v in pad_nodes
            u_is_iface = u in interface_node_to_idx
            v_is_iface = v in interface_node_to_idx

            # Only add edge if it connects an interface unknown to a pad,
            # or two interface unknowns (already covered by S_global but
            # extra_edges may add additional connections).
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

    # 3. BFS connected components
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
            # All interface unknowns in this component are islands
            # (exclude virtual pads, though a pad-only island is impossible here)
            island_nodes.update(component & set(interface_nodes))

    return island_nodes


def apply_island_penalty(
    S_global: sp.csr_matrix,
    rhs_dirichlet: np.ndarray,
    island_nodes: Set[str],
    interface_node_to_idx: Dict[str, int],
    dirichlet_voltage: float,
    penalty_conductance: float = 1e5,
) -> Tuple[sp.csr_matrix, np.ndarray]:
    """Apply a diagonal penalty to island nodes to make S_global non-singular.

    For each island node i, adds penalty_conductance to S_global[i, i] and
    penalty_conductance * dirichlet_voltage to rhs_dirichlet[i]. This
    effectively shorts the island nodes to the supply voltage through a
    large conductance, preventing a singular interface matrix.

    Args:
        S_global: Assembled global Schur complement (CSR, n x n).
        rhs_dirichlet: Right-hand side vector of shape (n,).
        island_nodes: Set of island node names to penalise.
        interface_node_to_idx: Dict mapping node name -> index in S_global.
        dirichlet_voltage: Supply voltage to pin island nodes to (V).
        penalty_conductance: Conductance added to diagonal (default 1e5,
            matching GMAX used elsewhere in this module).

    Returns:
        Tuple of (S_fixed, rhs_fixed). If island_nodes is empty the inputs
        are returned unchanged (zero-copy fast path).
    """
    if not island_nodes:
        return S_global, rhs_dirichlet

    # Build index array for island nodes
    indices = np.array(
        [interface_node_to_idx[n] for n in island_nodes], dtype=np.intp
    )

    n = S_global.shape[0]

    # Diagonal penalty via COO -> CSR addition
    penalty_vals = np.full(len(indices), penalty_conductance, dtype=np.float64)
    penalty_matrix = sp.coo_matrix(
        (penalty_vals, (indices, indices)), shape=(n, n)
    ).tocsr()

    S_fixed = S_global + penalty_matrix

    # RHS contribution: penalty_conductance * V_dd at island indices
    rhs_fixed = rhs_dirichlet.copy()
    rhs_fixed[indices] += penalty_conductance * dirichlet_voltage

    return S_fixed, rhs_fixed


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
    by :func:`apply_island_penalty`. If no islands are found the inputs are
    returned unchanged (zero-copy fast path).

    Args:
        S_global: Assembled global Schur complement (CSR, n x n).
        rhs_dirichlet: Right-hand side vector of shape (n,).
        interface_nodes: Ordered list of unknown node names.
        interface_node_to_idx: Dict mapping node name -> index.
        pad_nodes: Set of pad (Dirichlet) node names.
        extra_edges: Optional list of (u, v, conductance) tuples.
        dirichlet_voltage: Voltage to pin island nodes to. Required;
            callers must pass this explicitly (typically ``model.vdd``).
        penalty_conductance: Conductance for diagonal penalty (default 1e5).

    Returns:
        Tuple of (S_fixed, rhs_fixed, island_nodes). island_nodes is the
        set of detected island node names (empty if none found).
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

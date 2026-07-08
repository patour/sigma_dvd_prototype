"""Tile worker for distributed DDM solver.

Thin stateful actor that holds per-tile BlockMatrixSystem and delegates
ALL math to solver/coupled_system.py building blocks.

Parsing helpers (TileData, _parse_tile_ckt, etc.) live in tile_parsing.py
and are re-exported here for backward compatibility.

Time-domain methods (VCS, smoothing, transient, peak tracking) live in
tile_worker_td.py and are mixed in via _TimeDomainMixin.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Re-exports from tile_parsing for backward compatibility.
# All existing ``from distributed.tile_worker import X`` continue to work.
# ---------------------------------------------------------------------------

from .tile_parsing import (  # noqa: F401
    TileData,
    R_TO_KOHM,
    C_TO_FF,
    I_TO_MA,
    _is_gzip_file,
    _load_nd_file,
    _parse_tile_ckt,
    _iter_instance_sources,
    _parse_instance_models,
    _iter_instance_capacitors,
    _parse_instance_capacitors,
    parse_tile_with_instances,
    parse_and_dump_tile,
)

from .tile_worker_td import _TimeDomainMixin
from .tile_worker_adjoint import _AdjointWorkerMixin


class TileWorker(_AdjointWorkerMixin, _TimeDomainMixin):
    """Per-tile actor for distributed DDM. Thin wrapper that delegates
    all math to coupled_system.py building blocks.

    Time-domain methods (init_vectorized_sources, smooth_sources,
    evaluate_and_get_reduced_rhs, factor_transient_system,
    get_transient_reduced_rhs, get_transient_interior_voltages,
    set_initial_voltages, init_peak_tracking, update_peak_stats,
    get_peak_stats, get_tracked_waveforms, use_smoothed_sources) are
    provided by _TimeDomainMixin in tile_worker_td.py.

    Adjoint sensitivity methods (has_node, init_adjoint_source_mappings,
    filter_adjoint_sources, reset_adjoint_state) are provided by
    _AdjointWorkerMixin in tile_worker_adjoint.py.
    """

    def __init__(self):
        self._tile_data: Optional[TileData] = None
        self._block_system = None
        self._rhs_dirichlet = None
        self._interface_nodes: Optional[Set[str]] = None
        self._removed_island_nodes: Set[str] = set()

        # --- Time-domain state (Phase 4) ---
        # VectorizedCurrentSources (raw and smoothed)
        self._vec_sources = None
        self._smoothed_sources = None
        self._active_sources = None  # Points to _vec_sources or _smoothed_sources

        # Transient system (A = G + C_coeff * C)
        self._transient_block_system = None
        self._c_pp_diag: Optional[np.ndarray] = None
        self._c_ii_diag: Optional[np.ndarray] = None
        self._total_cap: float = 0.0
        self._C_coeff: float = 0.0
        self._dt_scaled: float = 0.0
        self._transient_method: str = 'be'
        # Transient time-stepping state
        self._v_interior_old: Optional[np.ndarray] = None
        self._last_f_i: Optional[np.ndarray] = None
        # Quasi-static: cached interior RHS from last evaluate_and_get_reduced_rhs
        self._last_qs_rhs_i: Optional[np.ndarray] = None
        # Per-node current mask (float64): 1.0 = keep, 0.0 = zero out
        self._current_node_mask: Optional[np.ndarray] = None

        # Peak tracking (dict-based for QS, array-based for transient)
        self._peak_per_node: Dict[str, Tuple[float, float]] = {}
        self._peak_vdd: float = 0.0
        self._peak_tracking_active: bool = False
        self._tracked_nodes: Optional[Set[str]] = None
        self._tracked_waveforms: Dict[str, List[float]] = {}
        # Vectorized peak arrays (init_peak_tracking sets these)
        self._peak_drops_array: Optional[np.ndarray] = None
        self._peak_times_array: Optional[np.ndarray] = None
        self._peak_use_arrays: bool = False
        self._tracked_node_indices: Dict[str, int] = {}

        # --- A2: Phase-folded step-column table (tile_worker_td.py) --------
        # None until precompute_step_columns() is called.
        self._step_col_table: Optional[Dict] = None
        # Settings (propagated via configure() from coordinator settings).
        self._use_step_columns: bool = True
        self._max_table_mb: float = 512.0
        # Pre-allocated current buffer for buffer-reuse per step (n_nodes,).
        # Allocated lazily on first use.
        self._current_buf: Optional[np.ndarray] = None

        # --- Adjoint sensitivity state ---
        self._init_adjoint_state()

    def configure(self, settings: Dict[str, Any]) -> None:
        """Apply solver configuration on this worker (call once after creation).

        Propagates solver backend settings (CHOLMOD mode, ordering, etc.)
        so that Ray workers match the driver-side configuration.
        """
        from pgmath.block_system import set_partial_factor_reg_resistance
        from pgmath.factor import (
            set_use_cholmod,
            set_cholmod_mode,
            set_cholmod_ordering,
            set_cholmod_use_long,
        )
        if 'partial_factor_reg_ohms' in settings:
            set_partial_factor_reg_resistance(settings['partial_factor_reg_ohms'])
        if 'use_cholmod' in settings:
            set_use_cholmod(settings['use_cholmod'])
        if 'cholmod_mode' in settings:
            set_cholmod_mode(settings['cholmod_mode'])
        if 'cholmod_ordering' in settings:
            set_cholmod_ordering(settings['cholmod_ordering'])
        if 'cholmod_use_long' in settings:
            set_cholmod_use_long(settings['cholmod_use_long'])
        # A2 step-column table settings
        if 'use_step_columns' in settings:
            self._use_step_columns = bool(settings['use_step_columns'])
        if 'max_table_mb' in settings:
            self._max_table_mb = float(settings['max_table_mb'])

    def setup(
        self,
        tile_config_dict: Dict[str, Any],
        interface_nodes: Set[str],
    ) -> Dict[str, Any]:
        """Parse tile and build block system (without factoring).

        Args:
            tile_config_dict: Dict with tile_id, ckt_path, nd_path, instance_path, net_filter
            interface_nodes: Set of all global interface nodes (boundary + die attachment)

        Returns:
            Dict with tile metadata: {tile_id, boundary_nodes, n_interior, n_boundary, islands_removed}
        """
        tc = tile_config_dict
        tile_id = tuple(tc['tile_id'])

        self._tile_data = parse_tile_with_instances(
            ckt_path=tc['ckt_path'],
            nd_path=tc.get('nd_path'),
            net_filter=tc.get('net_filter'),
            tile_id=tile_id,
            instance_path=tc.get('instance_path'),
        )

        return self._build_block_system(interface_nodes)

    def setup_from_pkl(self, pkl_path, interface_nodes):
        """Load TileData from .pkl, build block system. Worker-side I/O."""
        import pickle
        with open(pkl_path, 'rb') as f:
            self._tile_data = pickle.load(f)
        return self._build_block_system(interface_nodes)

    def setup_from_tile_data(
        self,
        tile_data: TileData,
        interface_nodes: Set[str],
    ) -> Dict[str, Any]:
        """Build block system from pre-parsed TileData (no file I/O).

        Accepts a TileData that was previously produced by _parse_tile_ckt()
        and _parse_instance_models() (e.g., loaded from a .pkl file).
        Performs island detection and builds BlockMatrixSystem, identical to
        the bottom half of setup().

        Args:
            tile_data: Pre-parsed tile data (edges, nodes, currents)
            interface_nodes: Set of all global interface nodes (boundary + die attachment)

        Returns:
            Dict with tile metadata: {tile_id, boundary_nodes, n_interior, n_boundary, islands_removed}
        """
        self._tile_data = tile_data
        return self._build_block_system(interface_nodes)

    def _build_block_system(
        self,
        interface_nodes: Set[str],
    ) -> Dict[str, Any]:
        """Shared logic: island detection + BlockMatrixSystem construction.

        Called by both setup() and setup_from_tile_data() after tile_data
        has been populated.

        Args:
            interface_nodes: Set of all global interface nodes

        Returns:
            Dict with tile metadata: {tile_id, boundary_nodes, n_interior, n_boundary, islands_removed}
        """
        self._interface_nodes = interface_nodes

        # Classify: port nodes = interface nodes present in this tile
        port_nodes_local = interface_nodes & self._tile_data.all_nodes

        # Floating island detection: remove components not connected to any port/boundary
        islands_removed = 0
        kept_nonlargest_iface: Set[str] = set()
        if port_nodes_local:
            islands_removed, kept_nonlargest_iface = self._remove_floating_islands(port_nodes_local)

        # Build BlockMatrixSystem from edges (no factorization yet)
        from pgmath.block_system import build_block_system_from_edges
        self._block_system, self._rhs_dirichlet = build_block_system_from_edges(
            edges=self._tile_data.resistive_edges,
            port_nodes=port_nodes_local,
            dirichlet_nodes=None,
            dirichlet_voltage=0.0,
            ground_node='0',
        )

        return {
            'tile_id': self._tile_data.tile_id,
            'boundary_nodes': self._block_system.port_nodes,
            'n_interior': self._block_system.n_interior,
            'n_boundary': self._block_system.n_ports,
            'islands_removed': islands_removed,
            'kept_nonlargest_iface': sorted(kept_nonlargest_iface),
        }

    # Minimum interface node count for a non-largest component to be kept.
    MIN_INTERFACE_NODES_KEEP = 5

    def _remove_floating_islands(self, port_nodes: Set[str]) -> Tuple[int, Set[str]]:
        """Remove disconnected components that are isolated fragments.

        Keeps the largest component plus any component with enough interface
        node connectivity (>= MIN_INTERFACE_NODES_KEEP interface nodes) to
        be a legitimate cross-tile strip.

        Returns:
            Tuple of (islands_removed, kept_nonlargest_iface).
        """
        # Build adjacency
        adj: Dict[str, Set[str]] = {}
        for u, v, g in self._tile_data.resistive_edges:
            if u == '0' or v == '0':
                continue
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)

        # Find connected components via BFS
        visited: Set[str] = set()
        components: List[Set[str]] = []
        for start_node in self._tile_data.all_nodes:
            if start_node in visited or start_node == '0':
                continue
            comp: Set[str] = set()
            queue = [start_node]
            while queue:
                node = queue.pop()
                if node in visited:
                    continue
                visited.add(node)
                comp.add(node)
                for nb in adj.get(node, set()):
                    if nb not in visited:
                        queue.append(nb)
            components.append(comp)

        if len(components) <= 1:
            return 0, set()

        largest = max(components, key=len)
        removed_nodes: Set[str] = set()
        kept_nonlargest_iface: Set[str] = set()
        islands_removed = 0
        for comp in components:
            if comp is largest:
                continue
            n_interface = len(comp & port_nodes)
            if n_interface >= self.MIN_INTERFACE_NODES_KEEP:
                kept_nonlargest_iface.update(comp & port_nodes)
                continue
            removed_nodes.update(comp)
            islands_removed += 1

        if not removed_nodes:
            return 0, kept_nonlargest_iface

        self._removed_island_nodes = removed_nodes.copy()
        self._tile_data.all_nodes -= removed_nodes
        self._tile_data.boundary_nodes -= removed_nodes
        self._tile_data.resistive_edges = [
            (u, v, g) for u, v, g in self._tile_data.resistive_edges
            if u not in removed_nodes and v not in removed_nodes
        ]
        self._tile_data.capacitive_edges = [
            (u, v, c) for u, v, c in self._tile_data.capacitive_edges
            if u not in removed_nodes and v not in removed_nodes
        ]
        for node in removed_nodes:
            self._tile_data.current_injections.pop(node, None)

        return islands_removed, kept_nonlargest_iface

    def factor_and_compute_schur(self) -> Tuple[Any, List[str], Dict[str, Any]]:
        """Factor interior and compute explicit Schur complement.

        Path selection (partial Cholesky vs chunked multi-RHS) is handled
        automatically by ``compute_explicit_schur()`` based on the active
        solver backend (CHOLMOD vs splu).  Both paths set ``lu_ii`` as a
        side-effect, so downstream RHS/recovery calls work transparently.

        Returns:
            Tuple of (S_i as numpy array, boundary_node_list, stats dict)
        """
        from pgmath.schur import compute_explicit_schur
        from pgmath.block_system import _format_bytes, _sparse_mem_bytes

        bs = self._block_system
        t_total_start = time.perf_counter()

        t0 = time.perf_counter()
        S, schur_stats = compute_explicit_schur(bs)
        schur_time = time.perf_counter() - t0

        # Both paths report factor timing in schur_stats: partial path
        # via 'analyze_s'+'factor_s', chunked path via 'factor_s' alone.
        factor_time = schur_stats.get('factor_s', 0) + schur_stats.get('analyze_s', 0)

        total_time = time.perf_counter() - t_total_start

        schur_path = schur_stats['path']

        # Build stats dict
        stats = dict(bs.stats())  # n_ports, n_interior, G_ii_nnz, etc.
        stats.update({
            'factor_interior_s': factor_time,
            'compute_schur_s': schur_time,
            'total_s': total_time,
            'schur_shape': S.shape,
            'schur_mem_bytes': schur_stats['schur_mem_bytes'],
            'schur_path': schur_path,
        })

        # Backend info from factor_adapter (may be None if n_interior == 0)
        fa = bs.factor_adapter
        if fa is not None:
            stats['factorization_backend'] = fa.backend
            stats['factorization_backend_info'] = fa.backend_info
        else:
            stats['factorization_backend'] = 'n/a'
            stats['factorization_backend_info'] = 'n/a'

        # DEBUG logging: per-tile matrix characteristics
        tid = self._tile_data.tile_id if self._tile_data else '?'
        n_ii = stats['n_interior']
        n_pp = stats['n_ports']
        G_ii_nnz = stats['G_ii_nnz']
        density = (G_ii_nnz / (n_ii * n_ii) * 100) if n_ii > 0 else 0.0
        G_ii_mem = _sparse_mem_bytes(bs.G_ii) if hasattr(bs.G_ii, 'data') else 0
        logger.debug(
            "Tile %s factor_and_compute_schur:\n"
            "  G_ii: %s x %s, nnz=%s (density %.5f%%), %s\n"
            "  G_pp: %s x %s, nnz=%s\n"
            "  G_pi: %s x %s, nnz=%s  |  G_ip: %s x %s, nnz=%s\n"
            "  Block system memory: %s\n"
            "  factor_interior: %.3fs  |  path: %s  |  backend: %s\n"
            "  compute_schur: %.3fs  |  Schur: %s x %s dense (%s)",
            tid,
            f"{n_ii:,}", f"{n_ii:,}", f"{G_ii_nnz:,}", density, _format_bytes(G_ii_mem),
            f"{n_pp:,}", f"{n_pp:,}", f"{stats['G_pp_nnz']:,}",
            f"{n_pp:,}", f"{n_ii:,}", f"{stats['G_pi_nnz']:,}",
            f"{n_ii:,}", f"{n_pp:,}", f"{stats['G_ip_nnz']:,}",
            _format_bytes(stats['mem_bytes']),
            factor_time, schur_path, stats['factorization_backend_info'],
            schur_time, f"{S.shape[0]:,}", f"{S.shape[1]:,}",
            _format_bytes(schur_stats['schur_mem_bytes']),
        )

        return S, list(bs.port_nodes), stats

    def get_reduced_rhs(self, current_injections: Optional[Dict[str, float]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Compute reduced RHS for this tile.

        Args:
            current_injections: Override current injections. If None, uses tile's own.

        Returns:
            Tuple of (reduced RHS numpy array (n_ports,), stats dict)
        """
        from pgmath.block_system import compute_reduced_rhs

        if current_injections is None:
            current_injections = self._tile_data.current_injections

        t0 = time.perf_counter()
        rhs = compute_reduced_rhs(
            self._block_system, current_injections, self._rhs_dirichlet,
        )
        rhs_time = time.perf_counter() - t0

        n_currents = sum(1 for v in current_injections.values() if v != 0.0)
        rhs_norm = float(np.linalg.norm(rhs))

        stats = {
            'rhs_time_s': rhs_time,
            'n_currents': n_currents,
            'rhs_norm': rhs_norm,
        }

        tid = self._tile_data.tile_id if self._tile_data else '?'
        logger.debug(
            "Tile %s get_reduced_rhs:\n"
            "  time: %.3fs  |  n_currents: %s  |  rhs_norm: %.2f",
            tid, rhs_time, f"{n_currents:,}", rhs_norm,
        )

        return rhs, stats

    def get_interior_voltages(
        self,
        boundary_voltages_dict: Dict[str, float],
        current_injections: Optional[Dict[str, float]] = None,
    ) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """Recover interior voltages from boundary voltages.

        Args:
            boundary_voltages_dict: Dict mapping boundary node -> voltage
            current_injections: Optional override currents (node -> mA).

        Returns:
            Tuple of (Dict mapping all tile nodes -> voltage, stats dict)
        """
        from pgmath.block_system import recover_bottom_voltages

        t0 = time.perf_counter()

        currents = current_injections if current_injections is not None else self._tile_data.current_injections

        port_voltages = np.zeros(self._block_system.n_ports, dtype=np.float64)
        for node, idx in self._block_system.port_to_idx.items():
            if node in boundary_voltages_dict:
                port_voltages[idx] = boundary_voltages_dict[node]

        interior_voltages = recover_bottom_voltages(
            self._block_system,
            port_voltages,
            currents,
            self._rhs_dirichlet,
        )

        all_voltages = dict(interior_voltages)
        for node in self._block_system.port_nodes:
            if node in boundary_voltages_dict:
                all_voltages[node] = boundary_voltages_dict[node]

        recovery_time = time.perf_counter() - t0

        n_nodes = len(all_voltages)
        if all_voltages:
            vals = list(all_voltages.values())
            v_min = min(vals)
            v_max = max(vals)
        else:
            v_min = 0.0
            v_max = 0.0

        stats = {
            'recovery_time_s': recovery_time,
            'n_nodes': n_nodes,
            'v_min': v_min,
            'v_max': v_max,
        }

        tid = self._tile_data.tile_id if self._tile_data else '?'
        logger.debug(
            "Tile %s get_interior_voltages:\n"
            "  time: %.3fs  |  n_nodes: %s  |  v_range: [%.3f, %.3f]",
            tid, recovery_time, f"{n_nodes:,}", v_min, v_max,
        )

        return all_voltages, stats

    def get_layer_metadata(self) -> Dict[str, Dict]:
        """Per-layer spatial extents, orientation, and stripe coordinates."""
        if self._tile_data is None:
            raise RuntimeError(
                "get_layer_metadata() called before setup(); no tile data loaded"
            )

        from visualization.stripe_heatmap import parse_node_info

        node_info: Dict[str, Tuple[float, float, str]] = {}
        layer_xs: Dict[str, List[float]] = {}
        layer_ys: Dict[str, List[float]] = {}

        for node in self._tile_data.all_nodes:
            x, y, layer = parse_node_info(node)
            if x is None:
                continue
            node_info[node] = (x, y, layer)
            layer_xs.setdefault(layer, []).append(x)
            layer_ys.setdefault(layer, []).append(y)

        layer_h: Dict[str, int] = {}
        layer_v: Dict[str, int] = {}
        layer_d: Dict[str, int] = {}

        for u, v, _g in self._tile_data.resistive_edges:
            u_info = node_info.get(u)
            v_info = node_info.get(v)
            if u_info is None or v_info is None:
                continue
            ux, uy, u_layer = u_info
            vx, vy, v_layer = v_info
            if u_layer != v_layer:
                continue

            dx = abs(vx - ux)
            dy = abs(vy - uy)
            if dx == 0 and dy == 0:
                continue

            if dy == 0:
                layer_h[u_layer] = layer_h.get(u_layer, 0) + 1
            elif dx == 0:
                layer_v[u_layer] = layer_v.get(u_layer, 0) + 1
            else:
                layer_d[u_layer] = layer_d.get(u_layer, 0) + 1

        result: Dict[str, Dict] = {}
        for layer in sorted(layer_xs):
            xs = layer_xs[layer]
            ys = layer_ys[layer]
            result[layer] = {
                'bbox': (min(xs), max(xs), min(ys), max(ys)),
                'n_nodes': len(xs),
                'stripe_coords_h': sorted(set(ys)),
                'stripe_coords_v': sorted(set(xs)),
                'edge_orientation': (
                    layer_h.get(layer, 0),
                    layer_v.get(layer, 0),
                    layer_d.get(layer, 0),
                ),
            }

        return result

    def get_current_injections(self) -> Dict[str, float]:
        """Return per-tile current injection dict ``{node: mA}``."""
        if self._tile_data is None:
            raise RuntimeError(
                "get_current_injections() called before setup(); no tile data loaded"
            )
        return dict(self._tile_data.current_injections)

    @property
    def tile_id(self) -> Tuple[int, int]:
        return self._tile_data.tile_id if self._tile_data else None

    @property
    def n_interior(self) -> int:
        return self._block_system.n_interior if self._block_system else 0

    @property
    def n_boundary(self) -> int:
        return self._block_system.n_ports if self._block_system else 0

    def get_floating_nodes_data(self) -> Dict[str, Any]:
        """Return floating node data for report aggregation."""
        if self._tile_data is None:
            raise RuntimeError(
                "get_floating_nodes_data() called before setup(); no tile data loaded"
            )
        return {
            'removed_nodes': list(self._removed_island_nodes),
            'tile_id': self._tile_data.tile_id,
        }

    def lookup_instance_names(
        self,
        target_nodes: Set[str],
        instance_path: Optional[str],
        nd_path: Optional[str] = None,
        net_filter: Optional[str] = None,
    ) -> Dict[str, str]:
        """Look up instance names for a set of target nodes from instanceModels file."""
        node_to_instance: Dict[str, str] = {}

        for prepared in _iter_instance_sources(instance_path, net_filter, nd_path):
            if prepared.node_pos in target_nodes:
                node_to_instance[prepared.node_pos] = prepared.cs.name

        return node_to_instance

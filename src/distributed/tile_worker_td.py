"""Time-domain mixin for TileWorker.

Provides VCS creation, PWL smoothing, quasi-static evaluation, transient
factorization/RHS/recovery, and peak tracking.  Kept in a separate file
to keep tile_worker.py under 500 lines; the public API is surfaced on
TileWorker via inheritance.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class _TimeDomainMixin:
    """Mixin providing time-domain methods for TileWorker.

    Expects the host class to provide:
        _block_system, _rhs_dirichlet, _tile_data, _vec_sources,
        _smoothed_sources, _active_sources, _transient_block_system,
        _c_pp_diag, _c_ii_diag, _total_cap, _C_coeff, _dt_scaled,
        _transient_method, _rhs_dirichlet_transient, _v_interior_old,
        _last_f_i, _peak_per_node, _peak_vdd,
        _peak_tracking_active, _tracked_nodes, _tracked_waveforms,
        get_reduced_rhs()
    """

    # --- 4a. Vectorized current source creation + disk caching ---------

    def init_vectorized_sources(
        self,
        instance_path: Optional[str],
        nd_path: Optional[str] = None,
        net_filter: Optional[str] = None,
        pkl_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Build VectorizedCurrentSources from instance model file.

        Supports optional disk caching via pkl_dir.

        Args:
            instance_path: Path to instanceModels*.sp file.
            nd_path: Path to .nd file for net filtering.
            net_filter: Optional lowercase net name to filter by.
            pkl_dir: Optional directory for VCS pickle cache.

        Returns:
            Stats dict: ``{n_sources, n_nodes, cached}``.
        """
        n_ports = self._block_system.n_ports
        n_interior = self._block_system.n_interior
        n_nodes = n_ports + n_interior

        if self._vec_sources is not None:
            stats = self._vec_sources.get_statistics()
            stats['n_nodes'] = n_nodes
            stats['cached'] = True
            return stats

        import os
        import pickle
        from analysis.vectorized_sources import VectorizedCurrentSources
        from .tile_parsing import _iter_instance_sources

        x, y = self._tile_data.tile_id

        # Try loading from disk cache
        if pkl_dir is not None:
            cache_path = os.path.join(pkl_dir, f'vcs_tile_{x}_{y}.pkl')
            if os.path.isfile(cache_path):
                logger.debug("Tile (%d,%d): loading VCS from %s", x, y, cache_path)
                with open(cache_path, 'rb') as f:
                    self._vec_sources = pickle.load(f)
                self._active_sources = self._vec_sources
                stats = self._vec_sources.get_statistics()
                stats['n_nodes'] = n_nodes
                stats['cached'] = True
                return stats

        # Build node_to_idx: ports [0..n_ports), interior [n_ports..n_total)
        node_to_idx: Dict[str, int] = dict(self._block_system.port_to_idx)
        for node, idx in self._block_system.interior_to_idx.items():
            node_to_idx[node] = idx + n_ports

        # Collect full CurrentSource objects from instance file
        sources_dict: Dict[str, Any] = {}
        for prepared in _iter_instance_sources(instance_path, net_filter, nd_path):
            sources_dict[prepared.cs.name] = prepared.cs

        self._vec_sources = VectorizedCurrentSources.from_current_sources(
            sources_dict, node_to_idx, n_nodes,
        )
        self._active_sources = self._vec_sources

        # Save to disk cache
        if pkl_dir is not None:
            os.makedirs(pkl_dir, exist_ok=True)
            cache_path = os.path.join(pkl_dir, f'vcs_tile_{x}_{y}.pkl')
            logger.debug("Tile (%d,%d): saving VCS to %s", x, y, cache_path)
            with open(cache_path, 'wb') as f:
                pickle.dump(self._vec_sources, f, protocol=pickle.HIGHEST_PROTOCOL)

        stats = self._vec_sources.get_statistics()
        stats['n_nodes'] = n_nodes
        stats['cached'] = False
        return stats

    # --- 4b. PWL smoothing ---------------------------------------------

    def smooth_sources(
        self,
        time_step: float,
        t_start: float,
        t_end: float,
    ) -> Dict[str, Any]:
        """Apply triangular low-pass smoothing to vectorized sources.

        Args:
            time_step: Filter window = 2 * time_step (seconds).
            t_start: Simulation start time (seconds).
            t_end: Simulation end time (seconds).

        Returns:
            Stats dict with smoothing parameters.
        """
        if self._vec_sources is None:
            raise RuntimeError(
                "smooth_sources() called before init_vectorized_sources()"
            )
        self._smoothed_sources = self._vec_sources.create_smoothed_copy(
            time_step, t_start, t_end,
        )
        self._active_sources = self._smoothed_sources
        return {'time_step': time_step, 't_start': t_start, 't_end': t_end}

    # --- 4c. Source selection -------------------------------------------

    def use_smoothed_sources(self, use_smoothed: bool = True) -> None:
        """Switch between raw and smoothed vectorized sources."""
        if use_smoothed:
            if self._smoothed_sources is None:
                raise RuntimeError(
                    "No smoothed sources; call smooth_sources() first"
                )
            self._active_sources = self._smoothed_sources
        else:
            if self._vec_sources is None:
                raise RuntimeError(
                    "No raw sources; call init_vectorized_sources() first"
                )
            self._active_sources = self._vec_sources

    # --- 4d. Quasi-static evaluate + reduced RHS -----------------------

    def evaluate_and_get_reduced_rhs(
        self, t: float,
    ) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        """Evaluate time-domain currents and compute reduced RHS.

        Falls back to static DC path when no VCS is loaded.  Caches the
        interior RHS vector so that the subsequent
        :meth:`recover_and_update_peaks` call recovers interior voltages
        using the same time-varying currents (not the static DC values).

        Args:
            t: Time in seconds.

        Returns:
            ``(g, total_current, stats)`` -- reduced RHS ``(n_ports,)``,
            scalar sum of currents (mA), and stats dict.
        """
        t0 = time.perf_counter()

        n_ports = self._block_system.n_ports
        n_interior = self._block_system.n_interior

        if self._active_sources is None:
            self._last_qs_rhs_i = None
            rhs, _rhs_stats = self.get_reduced_rhs()
            total = sum(self._tile_data.current_injections.values())
            eval_time = time.perf_counter() - t0
            stats = {
                'eval_time_s': eval_time,
                'n_active_sources': sum(
                    1 for v in self._tile_data.current_injections.values()
                    if v != 0.0
                ),
            }
            return rhs, total, stats

        current_array = self._active_sources.evaluate_at_time(t)
        total_current = float(np.sum(current_array))

        I_p = current_array[:n_ports]
        I_i = current_array[n_ports:n_ports + n_interior]

        rhs_d_p = self._rhs_dirichlet[:n_ports]
        rhs_d_i = self._rhs_dirichlet[n_ports:n_ports + n_interior]

        rhs_p = -I_p + rhs_d_p
        rhs_i = -I_i + rhs_d_i

        # Cache interior RHS for consistent recovery in recover_and_update_peaks
        self._last_qs_rhs_i = rhs_i

        if n_interior > 0 and self._block_system.lu_ii is not None:
            v_i = self._block_system.lu_ii(rhs_i)
            g = rhs_p - self._block_system.G_pi @ v_i
        else:
            g = rhs_p

        eval_time = time.perf_counter() - t0
        n_active = int(np.count_nonzero(current_array))
        stats = {
            'eval_time_s': eval_time,
            'n_active_sources': n_active,
        }

        tid = self._tile_data.tile_id if self._tile_data else '?'
        logger.debug(
            "Tile %s evaluate_and_get_reduced_rhs:\n"
            "  time: %.3fs  |  n_active_sources: %s  |  total_current: %.2f mA",
            tid, eval_time, f"{n_active:,}", total_current,
        )

        return g, total_current, stats

    # --- 4e. Transient system factorization ----------------------------

    def factor_transient_system(
        self,
        dt_scaled: float,
        method: str = 'be',
    ) -> Tuple[np.ndarray, List[str], float, Dict[str, Any]]:
        """Build and factor A = G + C_coeff * diag(C), return Schur complement.

        Args:
            dt_scaled: Time step in pico-seconds (dt_seconds * 1e12).
            method: ``'be'`` (Backward Euler) or ``'trap'`` (Trapezoidal).

        Returns:
            ``(S_A_dense, port_list, total_cap, stats)`` -- dense Schur
            complement, ordered port node names, total capacitance (fF),
            and stats dict.
        """
        from solver.coupled_system import (
            BlockMatrixSystem,
            build_grounded_capacitance_diags,
            compute_explicit_schur,
            _format_bytes,
        )
        import scipy.sparse as sp_mod

        bs = self._block_system
        n_ports = bs.n_ports
        n_interior = bs.n_interior

        # Build capacitance diagonals
        self._c_pp_diag, self._c_ii_diag, self._total_cap = (
            build_grounded_capacitance_diags(
                self._tile_data.capacitive_edges,
                bs.port_to_idx, bs.interior_to_idx,
                n_ports, n_interior,
            )
        )

        # Integration coefficient
        C_coeff = (2.0 if method == 'trap' else 1.0) / dt_scaled
        self._C_coeff = C_coeff
        self._dt_scaled = dt_scaled
        self._transient_method = method

        # A_ii = G_ii + C_coeff * diag(c_ii)
        if n_interior > 0:
            A_ii = bs.G_ii + sp_mod.diags(self._c_ii_diag * C_coeff, format='csr')
        else:
            A_ii = bs.G_ii

        # A_pp = G_pp + C_coeff * diag(c_pp)
        if n_ports > 0:
            A_pp = bs.G_pp + sp_mod.diags(self._c_pp_diag * C_coeff, format='csr')
        else:
            A_pp = bs.G_pp

        # Off-diagonal blocks unchanged (grounded caps -> C_pi = C_ip = 0)
        transient_bs = BlockMatrixSystem(
            G_pp=A_pp, G_pi=bs.G_pi, G_ip=bs.G_ip, G_ii=A_ii,
            port_nodes=list(bs.port_nodes),
            interior_nodes=list(bs.interior_nodes),
            port_to_idx=dict(bs.port_to_idx),
            interior_to_idx=dict(bs.interior_to_idx),
            lu_ii=None,
        )

        t0 = time.perf_counter()
        transient_bs.factor_interior()
        factor_time = time.perf_counter() - t0

        self._transient_block_system = transient_bs

        t0 = time.perf_counter()
        S_A, schur_stats = compute_explicit_schur(transient_bs)
        schur_time = time.perf_counter() - t0

        # Capacitance stats
        c_ii_cap_nodes = int(np.count_nonzero(self._c_ii_diag))
        c_pp_cap_nodes = int(np.count_nonzero(self._c_pp_diag))

        # Backend info from factor_adapter
        fa = transient_bs.factor_adapter
        if fa is not None:
            backend = fa.backend
            backend_info = fa.backend_info
        else:
            backend = 'n/a'
            backend_info = 'n/a'

        A_ii_nnz = A_ii.nnz if sp_mod.issparse(A_ii) else 0
        A_pp_nnz = A_pp.nnz if sp_mod.issparse(A_pp) else 0

        stats = {
            'factor_interior_s': factor_time,
            'compute_schur_s': schur_time,
            'total_cap_fF': self._total_cap,
            'A_ii_nnz': A_ii_nnz,
            'A_pp_nnz': A_pp_nnz,
            'schur_mem_bytes': schur_stats['schur_mem_bytes'],
            'schur_chunk_size': schur_stats['chunk_size'],
            'factorization_backend': backend,
            'factorization_backend_info': backend_info,
            'n_ports': n_ports,
            'n_interior': n_interior,
            'c_ii_cap_nodes': c_ii_cap_nodes,
            'c_pp_cap_nodes': c_pp_cap_nodes,
            'C_coeff': C_coeff,
        }

        tid = self._tile_data.tile_id if self._tile_data else '?'
        logger.debug(
            "Tile %s factor_transient_system:\n"
            "  A_ii: %s x %s, nnz=%s  |  A_pp: %s x %s, nnz=%s\n"
            "  C_ii cap nodes: %s / %s  |  C_pp cap nodes: %s / %s\n"
            "  Total tile cap: %.0f fF  |  C_coeff: %.4f\n"
            "  factor_interior: %.3fs  |  backend: %s\n"
            "  compute_schur: %.3fs  |  Schur: %s x %s dense (%s)  |  chunk_size: %s",
            tid,
            f"{n_interior:,}", f"{n_interior:,}", f"{A_ii_nnz:,}",
            f"{n_ports:,}", f"{n_ports:,}", f"{A_pp_nnz:,}",
            f"{c_ii_cap_nodes:,}", f"{n_interior:,}",
            f"{c_pp_cap_nodes:,}", f"{n_ports:,}",
            self._total_cap, C_coeff,
            factor_time, backend_info,
            schur_time, f"{S_A.shape[0]:,}", f"{S_A.shape[1]:,}",
            _format_bytes(schur_stats['schur_mem_bytes']),
            schur_stats['chunk_size'],
        )

        return S_A, list(bs.port_nodes), self._total_cap, stats

    # --- 4f. Transient reduced RHS ------------------------------------

    def get_transient_reduced_rhs(
        self,
        t: float,
        boundary_v_old: Dict[str, float],
    ) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        """Compute reduced RHS for one transient time step (BE or TR).

        Args:
            t: Current time in seconds.
            boundary_v_old: Previous-step port voltages ``{node: V}``.

        Returns:
            ``(g, total_current, stats)`` -- reduced RHS ``(n_ports,)``,
            scalar sum of currents (mA), and stats dict.
        """
        t0 = time.perf_counter()

        if self._transient_block_system is None:
            raise RuntimeError(
                "get_transient_reduced_rhs() requires factor_transient_system()"
            )

        bs = self._block_system
        tbs = self._transient_block_system
        n_ports = bs.n_ports
        n_interior = bs.n_interior

        # Evaluate currents
        if self._active_sources is not None:
            current_array = self._active_sources.evaluate_at_time(t)
            I_p = current_array[:n_ports]
            I_i = current_array[n_ports:n_ports + n_interior]
        else:
            I_p = np.zeros(n_ports, dtype=np.float64)
            I_i = np.zeros(n_interior, dtype=np.float64)
            for node, cur in self._tile_data.current_injections.items():
                if node in bs.port_to_idx:
                    I_p[bs.port_to_idx[node]] += cur
                elif node in bs.interior_to_idx:
                    I_i[bs.interior_to_idx[node]] += cur
            current_array = np.concatenate([I_p, I_i])

        total_current = float(np.sum(current_array))

        # Previous-step voltages
        v_p_old = np.zeros(n_ports, dtype=np.float64)
        for node, idx in bs.port_to_idx.items():
            if node in boundary_v_old:
                v_p_old[idx] = boundary_v_old[node]

        v_i_old = self._v_interior_old if self._v_interior_old is not None else np.zeros(n_interior, dtype=np.float64)

        rhs_d_p = self._rhs_dirichlet[:n_ports]
        rhs_d_i = self._rhs_dirichlet[n_ports:n_ports + n_interior]

        if self._transient_method == 'trap':
            f_i = (
                -2.0 * I_i
                + self._C_coeff * self._c_ii_diag * v_i_old
                - bs.G_ii @ v_i_old - bs.G_ip @ v_p_old
                + 2.0 * rhs_d_i
            )
            f_p = (
                -2.0 * I_p
                + self._C_coeff * self._c_pp_diag * v_p_old
                - bs.G_pi @ v_i_old - bs.G_pp @ v_p_old
                + 2.0 * rhs_d_p
            )
        else:  # Backward Euler
            f_i = -I_i + self._C_coeff * self._c_ii_diag * v_i_old + rhs_d_i
            f_p = -I_p + self._C_coeff * self._c_pp_diag * v_p_old + rhs_d_p

        self._last_f_i = f_i

        if n_interior > 0 and tbs.lu_ii is not None:
            g = f_p - tbs.G_pi @ tbs.lu_ii(f_i)
        else:
            g = f_p

        rhs_time = time.perf_counter() - t0
        rhs_norm = float(np.linalg.norm(g))
        stats = {
            'rhs_time_s': rhs_time,
            'total_current': total_current,
            'rhs_norm': rhs_norm,
        }

        tid = self._tile_data.tile_id if self._tile_data else '?'
        logger.debug(
            "Tile %s get_transient_reduced_rhs:\n"
            "  time: %.3fs  |  total_current: %.2f mA  |  rhs_norm: %.2f",
            tid, rhs_time, total_current, rhs_norm,
        )

        return g, total_current, stats

    # --- 4g. Transient interior recovery + state update ----------------

    def get_transient_interior_voltages(
        self,
        boundary_voltages_dict: Dict[str, float],
    ) -> Dict[str, float]:
        """Recover interior voltages and update state for next step.

        Args:
            boundary_voltages_dict: Solved port voltages for current step.

        Returns:
            Dict mapping all tile nodes -> voltage.
        """
        if self._transient_block_system is None:
            raise RuntimeError(
                "get_transient_interior_voltages() requires factor_transient_system()"
            )
        if self._last_f_i is None:
            raise RuntimeError(
                "get_transient_interior_voltages() needs a preceding "
                "get_transient_reduced_rhs() call"
            )

        tbs = self._transient_block_system
        bs = self._block_system
        n_ports = bs.n_ports

        v_p = np.zeros(n_ports, dtype=np.float64)
        for node, idx in bs.port_to_idx.items():
            if node in boundary_voltages_dict:
                v_p[idx] = boundary_voltages_dict[node]

        if bs.n_interior > 0 and tbs.lu_ii is not None:
            v_i = tbs.lu_ii(self._last_f_i - tbs.G_ip @ v_p)
        else:
            v_i = np.array([], dtype=np.float64)

        self._v_interior_old = v_i.copy()

        all_voltages: Dict[str, float] = {}
        for i, node in enumerate(bs.interior_nodes):
            all_voltages[node] = float(v_i[i])
        for node in bs.port_nodes:
            if node in boundary_voltages_dict:
                all_voltages[node] = boundary_voltages_dict[node]
        return all_voltages

    # --- 4h. State management ------------------------------------------

    def set_initial_voltages(self, voltages: Dict[str, float]) -> None:
        """Initialize time-stepping state from a voltage dict."""
        bs = self._block_system
        v_i = np.zeros(bs.n_interior, dtype=np.float64)
        for node, idx in bs.interior_to_idx.items():
            if node in voltages:
                v_i[idx] = voltages[node]
        self._v_interior_old = v_i

    def recover_and_set_initial_voltages(
        self,
        boundary_voltages_dict: Dict[str, float],
    ) -> Dict[str, Any]:
        """Recover DC interior voltages and set as transient IC in one step.

        Combines get_interior_voltages + set_initial_voltages without the
        coordinator round-trip.  The interior voltage array stays on the
        worker.

        Args:
            boundary_voltages_dict: Port/boundary voltages from DC interface
                solve.

        Returns:
            Stats dict with recovery metadata.
        """
        from solver.coupled_system import recover_bottom_voltages

        bs = self._block_system

        port_voltages = np.zeros(bs.n_ports, dtype=np.float64)
        for node, idx in bs.port_to_idx.items():
            if node in boundary_voltages_dict:
                port_voltages[idx] = boundary_voltages_dict[node]

        interior_voltages = recover_bottom_voltages(
            bs, port_voltages,
            self._tile_data.current_injections,
            self._rhs_dirichlet,
        )

        # Set _v_interior_old directly from recovered voltages
        v_i = np.zeros(bs.n_interior, dtype=np.float64)
        for node, idx in bs.interior_to_idx.items():
            if node in interior_voltages:
                v_i[idx] = interior_voltages[node]
        self._v_interior_old = v_i

        return {'n_interior': bs.n_interior, 'n_ports': bs.n_ports}

    # --- 4k. Factorization lifecycle -------------------------------------

    def clear_dc_factorization(self) -> Dict[str, Any]:
        """Free DC LU factorization from _block_system. G matrices preserved.

        Clears lu_ii and factor_adapter from the block system, freeing the
        LU memory while preserving G_ii, G_pp, G_pi, G_ip matrices for
        potential re-factorization.

        Returns:
            Stats dict with keys: had_lu (bool), n_interior (int).
        """
        had_lu = False
        n_interior = 0
        if self._block_system is not None:
            n_interior = self._block_system.n_interior
            if self._block_system.lu_ii is not None:
                had_lu = True
                self._block_system.lu_ii = None
            if self._block_system.factor_adapter is not None:
                self._block_system.factor_adapter = None
        return {'had_lu': had_lu, 'n_interior': n_interior}

    def clear_transient_factorization(self) -> Dict[str, Any]:
        """Free transient block system entirely.

        Removes the _transient_block_system (A = G + C_coeff * C), freeing
        both matrices and LU factorization. The base _block_system (DC G matrices)
        is preserved.

        Also clears transient time-stepping state (_last_f_i and _v_interior_old).
        Callers must call set_initial_voltages() before restarting transient
        integration.

        Returns:
            Stats dict with keys: had_transient_system (bool).
        """
        had = self._transient_block_system is not None
        self._transient_block_system = None
        # Clear transient time-stepping state that depends on transient system
        self._last_f_i = None
        self._v_interior_old = None
        return {'had_transient_system': had}

    # --- 4i. Peak tracking ---------------------------------------------

    def init_peak_tracking(
        self,
        track_nodes: Optional[List[str]] = None,
        vdd: float = 0.0,
    ) -> None:
        """Initialize per-node peak IR-drop tracking."""
        if vdd <= 0:
            raise ValueError(f"vdd must be positive, got {vdd}")
        self._peak_per_node = {}
        self._peak_vdd = vdd
        self._peak_tracking_active = True
        self._tracked_nodes = set(track_nodes) if track_nodes else None
        self._tracked_waveforms = {}

    def update_peak_stats(
        self,
        voltages: Dict[str, float],
        t: float,
    ) -> float:
        """Update per-node peaks and tracked waveforms.  Returns step max drop."""
        max_drop = 0.0
        vdd = self._peak_vdd
        for node, v in voltages.items():
            drop = vdd - v
            if drop > max_drop:
                max_drop = drop
            prev = self._peak_per_node.get(node)
            if prev is None or drop > prev[0]:
                self._peak_per_node[node] = (drop, t)
            if self._tracked_nodes is not None and node in self._tracked_nodes:
                self._tracked_waveforms.setdefault(node, []).append(v)
        return max_drop

    def get_peak_stats(self) -> Dict[str, Tuple[float, float]]:
        """Return ``{node: (max_drop, time_of_max)}``."""
        return self._peak_per_node

    def get_tracked_waveforms(self) -> Dict[str, List[float]]:
        """Return ``{node: [v0, v1, ...]}`` for tracked nodes."""
        return self._tracked_waveforms

    # --- 4j. Combined recover + peak update (coordinator convenience) ----

    def recover_and_update_peaks(
        self,
        boundary_voltages_dict: Dict[str, float],
        t: float,
    ) -> float:
        """Recover interior voltages and update peak stats in one call.

        When called after :meth:`evaluate_and_get_reduced_rhs`, uses the
        cached interior RHS (which includes time-varying VCS currents) to
        recover interior voltages consistently with the reduced RHS that
        was used to solve the interface system.

        Falls back to :meth:`get_interior_voltages` (static DC currents)
        when no cached interior RHS is available.

        Args:
            boundary_voltages_dict: Solved port voltages ``{node: V}``.
            t: Current time in seconds.

        Returns:
            Scalar max IR-drop for this tile at this time step.
        """
        if self._last_qs_rhs_i is not None:
            # Use cached interior RHS from evaluate_and_get_reduced_rhs
            voltages = self._recover_interior_from_cached_rhs(
                boundary_voltages_dict
            )
            self._last_qs_rhs_i = None  # Consumed; prevent stale use
        else:
            voltages, _recovery_stats = self.get_interior_voltages(boundary_voltages_dict)

        if self._peak_tracking_active:
            return self.update_peak_stats(voltages, t)
        vdd = self._peak_vdd if self._peak_vdd > 0 else 0.0
        if vdd > 0:
            return max((vdd - v for v in voltages.values()), default=0.0)
        return 0.0

    def _recover_interior_from_cached_rhs(
        self,
        boundary_voltages_dict: Dict[str, float],
    ) -> Dict[str, float]:
        """Recover interior voltages using the cached interior RHS.

        This produces interior voltages consistent with the time-varying
        currents evaluated in the last :meth:`evaluate_and_get_reduced_rhs`
        call, avoiding the static-current mismatch that would occur when
        using :meth:`get_interior_voltages` in the quasi-static loop.

        Args:
            boundary_voltages_dict: Solved port voltages ``{node: V}``.

        Returns:
            Dict mapping all tile nodes (interior + boundary) -> voltage.
        """
        bs = self._block_system
        n_ports = bs.n_ports

        # Build port voltage array
        v_p = np.zeros(n_ports, dtype=np.float64)
        for node, idx in bs.port_to_idx.items():
            if node in boundary_voltages_dict:
                v_p[idx] = boundary_voltages_dict[node]

        # v_i = inv(G_ii) @ (rhs_i - G_ip @ v_p)
        if bs.n_interior > 0 and bs.lu_ii is not None:
            v_i = bs.lu_ii(self._last_qs_rhs_i - bs.G_ip @ v_p)
        else:
            v_i = np.array([], dtype=np.float64)

        # Build all-node voltage dict
        all_voltages: Dict[str, float] = {}
        for i, node in enumerate(bs.interior_nodes):
            all_voltages[node] = float(v_i[i])
        for node in bs.port_nodes:
            if node in boundary_voltages_dict:
                all_voltages[node] = boundary_voltages_dict[node]

        return all_voltages

    def recover_transient_and_update_peaks(
        self,
        boundary_voltages_dict: Dict[str, float],
        t: float,
    ) -> float:
        """Recover transient interior voltages and update peak stats in one call.

        Combines :meth:`get_transient_interior_voltages` with
        :meth:`update_peak_stats` to avoid two round-trips per time step
        in the transient loop.

        Args:
            boundary_voltages_dict: Solved port voltages ``{node: V}``.
            t: Current time in seconds.

        Returns:
            Scalar max IR-drop for this tile at this time step.
        """
        voltages = self.get_transient_interior_voltages(boundary_voltages_dict)
        if self._peak_tracking_active:
            return self.update_peak_stats(voltages, t)
        vdd = self._peak_vdd if self._peak_vdd > 0 else 0.0
        if vdd > 0:
            return max((vdd - v for v in voltages.values()), default=0.0)
        return 0.0

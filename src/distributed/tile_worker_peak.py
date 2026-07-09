"""Peak tracking mixin for TileWorker.

Provides init_peak_tracking, update_peak_stats, peak data accessors,
and the combined recover + peak update methods used in the time loop.
Extracted from tile_worker_td.py to keep files under 800 lines.

The transient recovery path uses vectorized numpy arrays instead of
per-node Python dicts, avoiding ~800ms/step overhead on large tiles.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class _PeakTrackingMixin:
    """Mixin providing peak IR-drop tracking for TileWorker.

    Expects the host class to provide:
        _block_system, _transient_block_system, _last_f_i,
        _v_interior_old, _last_qs_rhs_i,
        _peak_per_node, _peak_vdd, _peak_tracking_active,
        _peak_drops_array, _peak_times_array, _peak_use_arrays,
        _tracked_nodes, _tracked_waveforms, _tracked_node_indices,
        get_interior_voltages()
    """

    # --- Peak tracking init -----------------------------------------------

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

        # Vectorized peak arrays: [0..n_ports) ports, [n_ports..) interior
        bs = self._block_system
        if bs is not None:
            n = bs.n_ports + bs.n_interior
            self._peak_drops_array = np.zeros(n, dtype=np.float64)
            self._peak_times_array = np.full(n, -1.0, dtype=np.float64)
        else:
            self._peak_drops_array = None
            self._peak_times_array = None
        self._peak_use_arrays = False  # True once transient path updates arrays

        # Precompute indices for tracked nodes
        self._tracked_node_indices: Dict[str, int] = {}
        if track_nodes and bs is not None:
            for node in track_nodes:
                if node in bs.port_to_idx:
                    self._tracked_node_indices[node] = bs.port_to_idx[node]
                elif node in bs.interior_to_idx:
                    self._tracked_node_indices[node] = (
                        bs.interior_to_idx[node] + bs.n_ports
                    )

    # --- Dict-based peak update (quasi-static path) -----------------------

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

    # --- Peak data accessors ----------------------------------------------

    def _rebuild_peak_dict(self) -> Dict[str, Tuple[float, float]]:
        """Reconstruct per-node peak dict from vectorized arrays."""
        bs = self._block_system
        pa = self._peak_drops_array
        pt = self._peak_times_array
        result: Dict[str, Tuple[float, float]] = {}
        for node, idx in bs.port_to_idx.items():
            if pa[idx] > 0 or pt[idx] >= 0:
                result[node] = (float(pa[idx]), float(pt[idx]))
        for node, idx in bs.interior_to_idx.items():
            ai = idx + bs.n_ports
            if pa[ai] > 0 or pt[ai] >= 0:
                result[node] = (float(pa[ai]), float(pt[ai]))
        return result

    def get_peak_stats(self) -> Dict[str, Tuple[float, float]]:
        """Return ``{node: (max_drop, time_of_max)}``."""
        if self._peak_use_arrays and self._peak_drops_array is not None:
            return self._rebuild_peak_dict()
        return self._peak_per_node

    def prebin_peak_data(
        self,
        bin_spec: 'GlobalBinSpec',
        exclude_set: Optional[Set[str]] = None,
    ) -> Dict[str, List[Optional[np.ndarray]]]:
        """Pre-bin peak IR-drop data into stripe bins for heatmap rendering.

        Converts stored per-node peak drops back to voltages
        (``voltage = vdd - drop``) before delegating to
        :func:`distributed.heatmap.prebin_tile`, which expects voltage
        inputs and internally computes IR-drop in mV.
        """
        if not self._peak_tracking_active:
            return {}
        vdd = self._peak_vdd
        if self._peak_use_arrays and self._peak_drops_array is not None:
            peak_dict = self._rebuild_peak_dict()
        else:
            peak_dict = self._peak_per_node
        if not peak_dict:
            return {}
        voltages = {node: vdd - drop for node, (drop, _t) in peak_dict.items()}
        from .heatmap import prebin_tile
        return prebin_tile(voltages, bin_spec, exclude_set)

    def get_top_k_peaks(self, k: int) -> List[Tuple[str, float, float]]:
        """Return top-K nodes by peak IR-drop as ``[(node, drop, time), ...]``."""
        if self._peak_use_arrays and self._peak_drops_array is not None:
            drops = self._peak_drops_array
            n = min(k, len(drops))
            if n == 0:
                return []
            idx = np.argpartition(drops, -n)[-n:]
            idx = idx[np.argsort(drops[idx])[::-1]]
            bs = self._block_system
            result = []
            for i in idx:
                i = int(i)
                if i < bs.n_ports:
                    node = bs.port_nodes[i]
                else:
                    node = bs.interior_nodes[i - bs.n_ports]
                result.append((node, float(drops[i]),
                               float(self._peak_times_array[i])))
            return result
        if not self._peak_per_node:
            return []
        items = [(node, drop, t) for node, (drop, t)
                 in self._peak_per_node.items()]
        items.sort(key=lambda x: x[1], reverse=True)
        return items[:k]

    def get_tracked_waveforms(self) -> Dict[str, List[float]]:
        """Return ``{node: [v0, v1, ...]}`` for tracked nodes."""
        return self._tracked_waveforms

    # --- QS combined recover + peak update --------------------------------

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

    # --- QS combined recover + peak update (array-based) -----------------

    def recover_and_update_peaks_arr(
        self,
        v_p: np.ndarray,
        t: float,
    ) -> float:
        """Recover interior voltages and update peak stats (array-based, QS).

        Array-based variant of :meth:`recover_and_update_peaks`: takes
        ``v_p`` as a pre-gathered float64 ndarray ``(n_ports,)`` instead
        of a boundary voltage dict, eliminating the per-port Python
        dict-lookup loop for port voltage construction.

        Args:
            v_p: Pre-gathered port voltages, shape ``(n_ports,)``.
                ``v_p[j]`` is the voltage at the tile's j-th port (in
                ``bs.port_nodes`` order). Pad/Dirichlet ports should
                already be filled with ``vdd`` by the coordinator.
            t: Current time in seconds.

        Returns:
            Scalar max IR-drop for this tile at this time step.
        """
        bs = self._block_system
        n_ports = bs.n_ports

        if self._last_qs_rhs_i is not None:
            # Use cached interior RHS from evaluate_and_get_reduced_rhs
            if bs.n_interior > 0 and bs.lu_ii is not None:
                v_i = bs.lu_ii(self._last_qs_rhs_i - bs.G_ip @ v_p)
            else:
                v_i = np.array([], dtype=np.float64)
            self._last_qs_rhs_i = None  # Consumed; prevent stale use
        else:
            # No VCS case: static DC recovery using tile current injections
            from pgmath.block_system import recover_bottom_voltages
            interior_voltages = recover_bottom_voltages(
                bs, v_p,
                self._tile_data.current_injections,
                self._rhs_dirichlet,
            )
            if bs.n_interior > 0:
                v_i = np.zeros(bs.n_interior, dtype=np.float64)
                for node, idx in bs.interior_to_idx.items():
                    if node in interior_voltages:
                        v_i[idx] = interior_voltages[node]
            else:
                v_i = np.array([], dtype=np.float64)

        # Build all-node voltage dict for peak tracking (same as dict path)
        all_voltages: Dict[str, float] = {}
        for j, node in enumerate(bs.port_nodes):
            all_voltages[node] = float(v_p[j])
        for i, node in enumerate(bs.interior_nodes):
            all_voltages[node] = float(v_i[i]) if i < len(v_i) else 0.0

        if self._peak_tracking_active:
            return self.update_peak_stats(all_voltages, t)
        vdd = self._peak_vdd if self._peak_vdd > 0 else 0.0
        if vdd > 0:
            return max((vdd - v for v in all_voltages.values()), default=0.0)
        return 0.0

    # --- Transient combined recover + peak update (vectorized) ------------

    def recover_transient_and_update_peaks(
        self,
        boundary_voltages_dict: Dict[str, float],
        t: float,
    ) -> float:
        """Recover transient interior voltages and update peak stats.

        Vectorized implementation: does the sparse solve inline and updates
        peak tracking with numpy array ops, avoiding the creation of a
        per-node Python dict (~1M+ entries) that dominated runtime.

        Args:
            boundary_voltages_dict: Solved port voltages ``{node: V}``.
            t: Current time in seconds.

        Returns:
            Scalar max IR-drop for this tile at this time step.
        """
        if self._transient_block_system is None:
            raise RuntimeError(
                "recover_transient_and_update_peaks() requires "
                "factor_transient_system()"
            )
        if self._last_f_i is None:
            raise RuntimeError(
                "recover_transient_and_update_peaks() needs a preceding "
                "get_transient_reduced_rhs() call"
            )

        tbs = self._transient_block_system
        bs = self._block_system
        n_ports = bs.n_ports
        n_interior = bs.n_interior
        vdd = self._peak_vdd if self._peak_vdd > 0 else 0.0

        # Build port voltage array (small: typically ~3K entries)
        v_p = np.zeros(n_ports, dtype=np.float64)
        for node, idx in bs.port_to_idx.items():
            if node in boundary_voltages_dict:
                v_p[idx] = boundary_voltages_dict[node]

        # Sparse solve for interior voltages
        if n_interior > 0 and tbs.lu_ii is not None:
            v_i = tbs.lu_ii(self._last_f_i - tbs.G_ip @ v_p)
        else:
            v_i = np.array([], dtype=np.float64)

        # Update state for next time step
        self._v_interior_old = v_i.copy()

        # Compute IR-drops vectorized (no per-node Python loop)
        max_drop = 0.0
        drops_p = np.empty(0, dtype=np.float64)
        drops_i = np.empty(0, dtype=np.float64)
        if vdd > 0:
            if n_ports > 0:
                drops_p = vdd - v_p
                max_drop = float(np.max(drops_p))
            if len(v_i) > 0:
                drops_i = vdd - v_i
                max_drop = max(max_drop, float(np.max(drops_i)))

        # Vectorized peak update using numpy arrays
        if (self._peak_tracking_active
                and self._peak_drops_array is not None):
            self._peak_use_arrays = True
            pa = self._peak_drops_array
            pt = self._peak_times_array

            if n_ports > 0:
                mask_p = drops_p > pa[:n_ports]
                pa[:n_ports][mask_p] = drops_p[mask_p]
                pt[:n_ports][mask_p] = t

            if len(drops_i) > 0:
                pa_i = pa[n_ports:n_ports + n_interior]
                pt_i = pt[n_ports:n_ports + n_interior]
                mask_i = drops_i > pa_i
                pa_i[mask_i] = drops_i[mask_i]
                pt_i[mask_i] = t

            # Tracked waveforms (number set by caller via init_peak_tracking)
            if self._tracked_node_indices:
                for node, idx in self._tracked_node_indices.items():
                    if idx < n_ports:
                        v = float(v_p[idx])
                    elif idx - n_ports < len(v_i):
                        v = float(v_i[idx - n_ports])
                    else:
                        continue
                    self._tracked_waveforms.setdefault(node, []).append(v)

        return max_drop

    # --- Transient combined recover + peak update (array-based) ----------

    def recover_transient_and_update_peaks_arr(
        self,
        v_p: np.ndarray,
        t: float,
    ) -> float:
        """Recover transient interior voltages and update peaks (array-based).

        Array-based variant of :meth:`recover_transient_and_update_peaks`:
        takes ``v_p`` as a pre-gathered float64 ndarray ``(n_ports,)``
        instead of a boundary voltage dict, eliminating the per-port
        Python dict-lookup loop for port voltage construction.

        Args:
            v_p: Pre-gathered port voltages, shape ``(n_ports,)``.
                ``v_p[j]`` is the voltage at the tile's j-th port (in
                ``bs.port_nodes`` order). Pad/Dirichlet ports should
                already be filled with ``vdd`` by the coordinator.
            t: Current time in seconds.

        Returns:
            Scalar max IR-drop for this tile at this time step.
        """
        if self._transient_block_system is None:
            raise RuntimeError(
                "recover_transient_and_update_peaks_arr() requires "
                "factor_transient_system()"
            )
        if self._last_f_i is None:
            raise RuntimeError(
                "recover_transient_and_update_peaks_arr() needs a preceding "
                "get_transient_reduced_rhs_arr() call"
            )

        tbs = self._transient_block_system
        bs = self._block_system
        n_ports = bs.n_ports
        n_interior = bs.n_interior
        vdd = self._peak_vdd if self._peak_vdd > 0 else 0.0

        # v_p: direct ndarray input — no per-port dict-lookup loop needed

        # Sparse solve for interior voltages
        if n_interior > 0 and tbs.lu_ii is not None:
            v_i = tbs.lu_ii(self._last_f_i - tbs.G_ip @ v_p)
        else:
            v_i = np.array([], dtype=np.float64)

        # Update state for next time step
        self._v_interior_old = v_i.copy()

        # Compute IR-drops vectorized (no per-node Python loop)
        max_drop = 0.0
        drops_p = np.empty(0, dtype=np.float64)
        drops_i = np.empty(0, dtype=np.float64)
        if vdd > 0:
            if n_ports > 0:
                drops_p = vdd - v_p
                max_drop = float(np.max(drops_p))
            if len(v_i) > 0:
                drops_i = vdd - v_i
                max_drop = max(max_drop, float(np.max(drops_i)))

        # Vectorized peak update using numpy arrays
        if (self._peak_tracking_active
                and self._peak_drops_array is not None):
            self._peak_use_arrays = True
            pa = self._peak_drops_array
            pt = self._peak_times_array

            if n_ports > 0:
                mask_p = drops_p > pa[:n_ports]
                pa[:n_ports][mask_p] = drops_p[mask_p]
                pt[:n_ports][mask_p] = t

            if len(drops_i) > 0:
                pa_i = pa[n_ports:n_ports + n_interior]
                pt_i = pt[n_ports:n_ports + n_interior]
                mask_i = drops_i > pa_i
                pa_i[mask_i] = drops_i[mask_i]
                pt_i[mask_i] = t

            # Tracked waveforms (number set by caller via init_peak_tracking)
            if self._tracked_node_indices:
                for node, idx in self._tracked_node_indices.items():
                    if idx < n_ports:
                        v = float(v_p[idx])
                    elif idx - n_ports < len(v_i):
                        v = float(v_i[idx - n_ports])
                    else:
                        continue
                    self._tracked_waveforms.setdefault(node, []).append(v)

        return max_drop

"""Adjoint sensitivity mixin for TileWorker.

Provides source name/index mappings, spatial filtering, and adjoint state
management for distributed adjoint sensitivity analysis.  Kept in a
separate file following the mixin pattern established by tile_worker_td.py.

The public API is surfaced on TileWorker via inheritance.

Index alignment contract
------------------------
``init_adjoint_source_mappings()`` must reproduce the *exact* source
ordering used by ``VectorizedCurrentSources.from_current_sources()``
in ``init_vectorized_sources()``.  Both methods:

1. Collect ``sources_dict[prepared.cs.name] = prepared.cs`` from
   ``_iter_instance_sources(instance_path, net_filter, nd_path)``.
2. Iterate ``sources_dict.items()`` (insertion-order, Python 3.7+).
3. Skip sources whose ``src.node1 not in node_to_idx``.
4. Assign a monotonically increasing ``source_idx`` to each kept source.

Any deviation breaks ``evaluate_per_source_at_time(t)[idx]`` lookups.
Runtime validation in ``recover_and_accumulate_adjoint()`` checks
``n_adjoint == n_vcs`` and raises if misaligned.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

import numpy as np

from .tile_parsing import _parse_node_xy

if TYPE_CHECKING:
    from parser.current_sources import PreparedSource

logger = logging.getLogger(__name__)


class _AdjointWorkerMixin:
    """Mixin providing adjoint sensitivity methods for TileWorker.

    Expects the host class to provide:
        _block_system, _tile_data, _vec_sources,
        _block_system.port_to_idx, _block_system.interior_to_idx,
        _block_system.port_nodes, _block_system.interior_nodes
    """

    # ------------------------------------------------------------------
    # State initialisation (called from TileWorker.__init__)
    # ------------------------------------------------------------------

    def _init_adjoint_state(self) -> None:
        """Initialise all adjoint-related state fields to safe defaults.

        Called once from ``TileWorker.__init__``.  All fields are reset
        to the same values by ``reset_adjoint_state()``.
        """
        # Source name <-> VCS index mappings (populated by init_adjoint_source_mappings)
        self._adjoint_source_name_to_idx: Dict[str, int] = {}
        self._adjoint_source_idx_to_name: List[str] = []

        # Source name -> attached node name
        self._adjoint_source_to_node: Dict[str, str] = {}

        # Node name -> list of attached source names
        self._adjoint_node_to_sources: Dict[str, List[str]] = {}

        # Per-step adjoint variable for interior nodes (set during backward sweep)
        self._adjoint_lambda_interior: Optional[np.ndarray] = None

        # Accumulated per-source contributions (source_name -> float, in V)
        self._adjoint_contributions: Dict[str, float] = {}

        # Spatial window boolean mask over sources (n_sources,) or None
        self._adjoint_source_mask: Optional[np.ndarray] = None

        # Per-source (time, current) waveform snapshots for reporting
        self._adjoint_waveforms: Dict[str, List[Tuple[float, float]]] = {}

    # ------------------------------------------------------------------
    # Node membership query
    # ------------------------------------------------------------------

    def has_node(self, node: str) -> bool:
        """Check whether *node* belongs to this tile (port or interior).

        Args:
            node: Node name string.

        Returns:
            True if *node* is a port or interior node of this tile's
            block system.
        """
        if self._block_system is None:
            return False
        return (
            node in self._block_system.port_to_idx
            or node in self._block_system.interior_to_idx
        )

    # ------------------------------------------------------------------
    # Source mapping construction
    # ------------------------------------------------------------------

    def init_adjoint_source_mappings(
        self,
        instance_path: Optional[str],
        nd_path: Optional[str] = None,
        net_filter: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Build source-name to VCS-index mappings for adjoint lookups.

        The iteration order and skip predicate **must** mirror
        ``init_vectorized_sources()`` exactly so that
        ``_adjoint_source_name_to_idx[name]`` equals the index returned
        by ``VectorizedCurrentSources.evaluate_per_source_at_time(t)``.

        Algorithm
        ---------
        1. Build the same ``node_to_idx`` mapping that VCS uses:
           ports at ``[0, n_ports)`` and interior at ``[n_ports, n_total)``.
        2. Iterate ``_iter_instance_sources`` with the same arguments to
           build ``sources_dict`` keyed by ``prepared.cs.name`` (preserving
           insertion order).
        3. Walk ``sources_dict.items()`` and skip entries whose
           ``src.node1 not in node_to_idx`` -- same skip predicate as
           ``VectorizedCurrentSources.from_current_sources()``.
        4. For every kept source, record name-to-index, index-to-name,
           source-to-node, and node-to-sources mappings.

        Args:
            instance_path: Path to instanceModels*.sp file.
            nd_path: Path to .nd file for net filtering.
            net_filter: Optional lowercase net name to filter by.

        Returns:
            Stats dict with ``n_sources``, ``n_nodes_with_sources``,
            ``n_skipped`` counts.
        """
        from .tile_parsing import _iter_instance_sources

        bs = self._block_system
        if bs is None:
            raise RuntimeError(
                "init_adjoint_source_mappings() called before setup(); "
                "no block system available"
            )

        # 1. Reconstruct the same node_to_idx that VCS uses
        n_ports = bs.n_ports
        node_to_idx: Dict[str, int] = dict(bs.port_to_idx)
        for node, idx in bs.interior_to_idx.items():
            node_to_idx[node] = idx + n_ports

        # 2. Collect sources in insertion order (same as init_vectorized_sources)
        sources_dict: Dict[str, Any] = {}
        prepared: PreparedSource
        for prepared in _iter_instance_sources(instance_path, net_filter, nd_path):
            sources_dict[prepared.cs.name] = prepared.cs

        # 3. Walk sources_dict with same skip predicate as
        #    VectorizedCurrentSources.from_current_sources()
        name_to_idx: Dict[str, int] = {}
        idx_to_name: List[str] = []
        source_to_node: Dict[str, str] = {}
        node_to_sources: Dict[str, List[str]] = {}

        source_idx = 0
        n_skipped = 0
        for name, src in sources_dict.items():
            node = src.node1
            if node not in node_to_idx:
                n_skipped += 1
                continue

            name_to_idx[name] = source_idx
            idx_to_name.append(name)
            source_to_node[name] = node

            if node not in node_to_sources:
                node_to_sources[node] = []
            node_to_sources[node].append(name)

            source_idx += 1

        self._adjoint_source_name_to_idx = name_to_idx
        self._adjoint_source_idx_to_name = idx_to_name
        self._adjoint_source_to_node = source_to_node
        self._adjoint_node_to_sources = node_to_sources

        # Clear any stale mask / contributions from a prior run
        self._adjoint_source_mask = None
        self._adjoint_contributions = {}
        self._adjoint_waveforms = {}
        self._adjoint_lambda_interior = None

        n_sources = len(name_to_idx)
        n_nodes_with_sources = len(node_to_sources)

        tid = self._tile_data.tile_id if self._tile_data else '?'
        logger.debug(
            "Tile %s init_adjoint_source_mappings: "
            "n_sources=%d, n_nodes_with_sources=%d, n_skipped=%d",
            tid, n_sources, n_nodes_with_sources, n_skipped,
        )

        return {
            'n_sources': n_sources,
            'n_nodes_with_sources': n_nodes_with_sources,
            'n_skipped': n_skipped,
        }

    # ------------------------------------------------------------------
    # Spatial filtering
    # ------------------------------------------------------------------

    def filter_adjoint_sources(
        self,
        spatial_window: Optional[Tuple[float, float, float, float]],
    ) -> int:
        """Build a boolean mask selecting sources within *spatial_window*.

        Source coordinates are derived from the node each source is
        attached to.  PDN node names encode coordinates as
        ``<x>_<y>_<layer>`` (e.g. ``'1000_2000_M1'``).

        Passing ``None`` clears the mask so all sources are active.

        Args:
            spatial_window: ``(x_min, x_max, y_min, y_max)`` bounding
                box in the same coordinate units as node names, or
                ``None`` to clear filtering.

        Returns:
            Number of sources that pass the filter (i.e. are active).
        """
        n_sources = len(self._adjoint_source_idx_to_name)
        if n_sources == 0:
            self._adjoint_source_mask = None
            return 0

        if spatial_window is None:
            self._adjoint_source_mask = None
            return n_sources

        x_min, x_max, y_min, y_max = spatial_window
        mask = np.zeros(n_sources, dtype=np.bool_)

        for idx, name in enumerate(self._adjoint_source_idx_to_name):
            node = self._adjoint_source_to_node.get(name)
            if node is None:
                continue
            x, y = _parse_node_xy(node)
            if x is None:
                continue
            if x_min <= x <= x_max and y_min <= y <= y_max:
                mask[idx] = True

        self._adjoint_source_mask = mask
        n_active = int(mask.sum())

        tid = self._tile_data.tile_id if self._tile_data else '?'
        logger.debug(
            "Tile %s filter_adjoint_sources: "
            "window=(%.0f, %.0f, %.0f, %.0f), active=%d / %d",
            tid, x_min, x_max, y_min, y_max, n_active, n_sources,
        )

        return n_active

    # ------------------------------------------------------------------
    # State reset
    # ------------------------------------------------------------------

    def reset_adjoint_state(self) -> None:
        """Clear per-analysis adjoint state (contributions, masks, waveforms).

        Call between successive adjoint analyses on the same tile to
        avoid leaking accumulated contributions or stale masks.

        Note: This preserves the source mappings (populated by
        ``init_adjoint_source_mappings``) which are reused across analyses.
        Only per-analysis state is cleared.
        """
        # Preserve persistent mappings:
        # - _adjoint_source_name_to_idx (populated by init_adjoint_source_mappings)
        # - _adjoint_source_idx_to_name (populated by init_adjoint_source_mappings)
        # - _adjoint_source_to_node (populated by init_adjoint_source_mappings)
        # - _adjoint_node_to_sources (populated by init_adjoint_source_mappings)

        # Clear per-analysis state:
        self._adjoint_lambda_interior = None
        self._adjoint_contributions = {}
        self._adjoint_source_mask = None
        self._adjoint_waveforms = {}

        tid = self._tile_data.tile_id if getattr(self, '_tile_data', None) else '?'
        logger.debug("Tile %s reset_adjoint_state: cleared per-analysis state", tid)

    # ------------------------------------------------------------------
    # Shared accumulation loop
    # ------------------------------------------------------------------

    def _accumulate_contributions(
        self,
        lambda_i: np.ndarray,
        lambda_p: np.ndarray,
        t: float,
        include_waveforms: bool,
    ) -> int:
        """Accumulate per-source adjoint contributions from lambda vectors.

        This is the shared inner loop used by all three recover-and-accumulate
        methods (dynamic, terminal, static).  For each active source *j*:

            ``contributions[j] += lambda[node_j] * I_j(t)``

        where ``lambda[node_j]`` is looked up from *lambda_i* (interior) or
        *lambda_p* (port), and ``I_j(t)`` comes from the VCS evaluation.

        Args:
            lambda_i: Recovered interior lambda vector (may be empty).
            lambda_p: Interface lambda vector of shape ``(n_ports,)``.
            t: Time point for current evaluation (seconds).
            include_waveforms: Record ``(t, I_j)`` per source for reporting.

        Returns:
            Number of sources accumulated.

        Raises:
            RuntimeError: If adjoint/VCS index alignment is broken.
        """
        if self._vec_sources is None or len(self._adjoint_source_idx_to_name) == 0:
            return 0

        bs = self._block_system

        # Validate index alignment (defensive check -- see module docstring)
        n_adjoint = len(self._adjoint_source_idx_to_name)
        n_vcs = self._vec_sources.n_sources
        if n_adjoint != n_vcs:
            raise RuntimeError(
                f"Index alignment mismatch: adjoint has {n_adjoint} sources, "
                f"VCS has {n_vcs}. Ensure init_adjoint_source_mappings() and "
                f"init_vectorized_sources() use the same arguments."
            )

        per_source_currents = self._vec_sources.evaluate_per_source_at_time(t)
        n_accumulated = 0

        for source_idx, source_name in enumerate(self._adjoint_source_idx_to_name):
            if self._adjoint_source_mask is not None:
                if not self._adjoint_source_mask[source_idx]:
                    continue

            I_j = per_source_currents[source_idx]

            node_j = self._adjoint_source_to_node.get(source_name)
            if node_j is None:
                continue

            # Look up lambda at the node where source j is attached
            lambda_at_node: float = 0.0
            if node_j in bs.interior_to_idx:
                idx = bs.interior_to_idx[node_j]
                if idx < len(lambda_i):
                    lambda_at_node = lambda_i[idx]
            elif node_j in bs.port_to_idx:
                idx = bs.port_to_idx[node_j]
                lambda_at_node = lambda_p[idx]

            # Discrete adjoint contribution (no dt factor):
            # lambda has units of kOhm (from A^{-1} where A is in mS)
            # current has units of mA
            # lambda * current has units: kOhm * mA = V
            contribution = lambda_at_node * I_j

            if source_name not in self._adjoint_contributions:
                self._adjoint_contributions[source_name] = 0.0
            self._adjoint_contributions[source_name] += contribution
            n_accumulated += 1

            if include_waveforms:
                if source_name not in self._adjoint_waveforms:
                    self._adjoint_waveforms[source_name] = []
                self._adjoint_waveforms[source_name].append((t, float(I_j)))

        return n_accumulated

    # ------------------------------------------------------------------
    # Adjoint backward sweep RHS computation
    # ------------------------------------------------------------------

    def get_adjoint_terminal_reduced_rhs(
        self,
        victim_node: str,
        use_transient: bool,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Build terminal condition reduced RHS for adjoint backward sweep.

        Computes the Schur-reduced RHS from the terminal condition
        ``A @ lambda_{L-1} = e_victim`` where ``e_victim`` is a unit
        vector at the victim node.

        Three cases:
        - **Victim is interior on this tile**: Reduced RHS =
          ``-A_pi @ lu_A_ii^{-1}(e_i)`` where ``e_i`` has 1.0 at victim.
        - **Victim is a port on this tile**: Reduced RHS = ``e_p``
          (unit vector at victim's port index).
        - **Victim is not on this tile**: Reduced RHS = zeros.

        Args:
            victim_node: The victim node name.
            use_transient: If True, use transient block system (A = G + C*C).
                If False, use DC block system (G matrices).

        Returns:
            Tuple of ``(reduced_rhs, stats)`` where ``reduced_rhs`` has
            shape ``(n_ports,)`` and ``stats`` contains
            ``{'victim_type': 'interior' | 'port' | 'absent', 'victim_idx': int | None}``.

        Raises:
            RuntimeError: If the required block system is not factored.
        """
        if use_transient:
            bs = self._transient_block_system
            if bs is None:
                raise RuntimeError(
                    "get_adjoint_terminal_reduced_rhs(use_transient=True) "
                    "requires factor_transient_system() to be called first"
                )
            if bs.lu_ii is None and bs.n_interior > 0:
                raise RuntimeError(
                    "get_adjoint_terminal_reduced_rhs(use_transient=True) "
                    "requires transient interior factorization"
                )
        else:
            bs = self._block_system
            if bs is None:
                raise RuntimeError(
                    "get_adjoint_terminal_reduced_rhs() called before setup(); "
                    "no block system available"
                )
            if bs.lu_ii is None:
                raise RuntimeError(
                    "get_adjoint_terminal_reduced_rhs(use_transient=False) "
                    "requires factor_and_compute_schur() to be called first"
                )

        n_ports = bs.n_ports
        n_interior = bs.n_interior

        # Case 1: victim is an interior node on this tile
        if victim_node in bs.interior_to_idx:
            victim_idx = bs.interior_to_idx[victim_node]
            e_i = np.zeros(n_interior, dtype=np.float64)
            e_i[victim_idx] = 1.0

            # Reduced RHS: g = 0 - A_pi @ lu_A_ii^{-1}(e_i) = -A_pi @ lu_ii(e_i)
            # Note: bs.G_pi is the off-diagonal conductance block. For transient,
            # A_pi == G_pi because tile caps are grounded (C_pi = 0).
            if n_interior > 0 and bs.lu_ii is not None:
                v_i = bs.lu_ii(e_i)
                g = -bs.G_pi @ v_i
            else:
                g = np.zeros(n_ports, dtype=np.float64)

            stats = {'victim_type': 'interior', 'victim_idx': victim_idx}
            return g, stats

        # Case 2: victim is a port node on this tile
        if victim_node in bs.port_to_idx:
            victim_idx = bs.port_to_idx[victim_node]
            g = np.zeros(n_ports, dtype=np.float64)
            g[victim_idx] = 1.0

            stats = {'victim_type': 'port', 'victim_idx': victim_idx}
            return g, stats

        # Case 3: victim is not on this tile
        g = np.zeros(n_ports, dtype=np.float64)
        stats = {'victim_type': 'absent', 'victim_idx': None}
        return g, stats

    def get_adjoint_step_reduced_rhs(
        self,
        lambda_p_old: Dict[str, float],
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Compute non-terminal step reduced RHS for adjoint backward sweep.

        For non-terminal steps, the adjoint equation is:
            ``A @ lambda_k = B @ lambda_{k+1}``

        where ``B = C_coeff * C`` (capacitive mass matrix). Since tile
        capacitors are grounded (diagonal C), we have:
            - ``B_ii = C_coeff * diag(c_ii)``
            - ``B_pp = C_coeff * diag(c_pp)``
            - ``B_ip = B_pi = 0`` (no off-diagonal coupling)

        The Schur reduction gives:
            ``g = s_p - A_pi @ lu_A_ii^{-1}(s_i)``

        where:
            - ``s_i = C_coeff * c_ii * lambda_i_old`` (element-wise)
            - ``s_p = C_coeff * c_pp * lambda_p_old`` (element-wise)

        **Note:** Uses ``_adjoint_lambda_interior`` (stored from previous
        recover step) for ``lambda_i_old``. The adjoint backward sweep
        has NO Dirichlet RHS (pads have fixed voltage, zero adjoint).

        Args:
            lambda_p_old: Port lambda values from previous step,
                keyed by node name.

        Returns:
            Tuple of ``(reduced_rhs, stats)`` where ``reduced_rhs`` has
            shape ``(n_ports,)`` and ``stats`` includes ``'s_i'`` and
            ``'s_p'`` arrays for use in recovery.

        Raises:
            RuntimeError: If transient system not factored or lambda_interior
                not available from a prior recover call.
        """
        if self._transient_block_system is None:
            raise RuntimeError(
                "get_adjoint_step_reduced_rhs() requires "
                "factor_transient_system() to be called first"
            )
        if self._adjoint_lambda_interior is None:
            raise RuntimeError(
                "get_adjoint_step_reduced_rhs() requires lambda_interior "
                "from a prior recover_and_accumulate_adjoint() call"
            )

        tbs = self._transient_block_system
        bs = self._block_system
        n_ports = bs.n_ports
        n_interior = bs.n_interior

        C_coeff = self._C_coeff
        c_ii_diag = self._c_ii_diag
        c_pp_diag = self._c_pp_diag

        lambda_p_arr = np.zeros(n_ports, dtype=np.float64)
        for node, idx in bs.port_to_idx.items():
            if node in lambda_p_old:
                lambda_p_arr[idx] = lambda_p_old[node]

        # Source terms from B @ lambda_old (B is diagonal)
        # s_i = C_coeff * c_ii_diag * lambda_i_old (element-wise)
        # s_p = C_coeff * c_pp_diag * lambda_p_old (element-wise)
        s_i = C_coeff * c_ii_diag * self._adjoint_lambda_interior
        s_p = C_coeff * c_pp_diag * lambda_p_arr

        # Schur reduction: g = s_p - A_pi @ lu_A_ii^{-1}(s_i)
        # Note: Use transient block system's lu_ii and G_pi (which is A_pi)
        if n_interior > 0 and tbs.lu_ii is not None:
            v_i = tbs.lu_ii(s_i)
            g = s_p - tbs.G_pi @ v_i
        else:
            g = s_p

        stats = {
            'C_coeff': C_coeff,
            's_i_norm': float(np.linalg.norm(s_i)),
            's_p_norm': float(np.linalg.norm(s_p)),
            'g_norm': float(np.linalg.norm(g)),
            's_i': s_i,
            's_p': s_p,
        }

        tid = self._tile_data.tile_id if self._tile_data else '?'
        logger.debug(
            "Tile %s get_adjoint_step_reduced_rhs: "
            "C_coeff=%.4f, s_i_norm=%.2e, s_p_norm=%.2e, g_norm=%.2e",
            tid, C_coeff, stats['s_i_norm'], stats['s_p_norm'], stats['g_norm'],
        )

        return g, stats

    # ------------------------------------------------------------------
    # Adjoint lambda recovery and contribution accumulation
    # ------------------------------------------------------------------

    def recover_and_accumulate_adjoint(
        self,
        lambda_p_dict: Dict[str, float],
        t: float,
        source_term_i: Optional[np.ndarray],
        source_term_p: Optional[np.ndarray],
        include_waveforms: bool,
    ) -> Dict[str, Any]:
        """Recover interior lambda and accumulate per-source contributions.

        Performs two operations:
        1. Recovers interior lambda values from interface lambda_p using the
           standard Schur complement back-substitution.
        2. Accumulates per-source contributions to the total IR-drop at the
           victim node.

        **Lambda recovery math:**

        For terminal step (source term is e_victim):
            ``lambda_i = lu_ii(e_i - A_ip @ lambda_p)`` if victim is interior
            ``lambda_i = -lu_ii(A_ip @ lambda_p)`` otherwise

        For non-terminal step (source term is B @ lambda_old):
            ``lambda_i = lu_ii(s_i - A_ip @ lambda_p)``
            where ``s_i = C_coeff * c_ii * lambda_i_old``

        **Contribution accumulation (discrete adjoint, no dt factor):**
            ``contributions[j] += lambda_at_node_j * I_j(t)``

        where ``lambda_at_node_j`` is the lambda value at the node where
        source j is attached, and ``I_j(t)`` is the current of source j
        at time t.

        Args:
            lambda_p_dict: Interface lambda values from coordinator solve,
                keyed by node name.
            t: Current time point (seconds).
            source_term_i: Interior source term from RHS computation.
                For terminal step with interior victim: e_i (unit vector).
                For non-terminal step: s_i = C_coeff * c_ii * lambda_i_old.
                None if absent (zero contribution from interior).
            source_term_p: Port source term.
                For terminal step with port victim: e_p (unit vector).
                For non-terminal step: s_p = C_coeff * c_pp * lambda_p_old.
                None if absent (zero contribution from ports).
            include_waveforms: Whether to record (t, I) waveform data per source.

        Returns:
            Stats dict with ``'n_accumulated'`` (number of sources accumulated),
            ``'lambda_i_norm'`` (norm of recovered interior lambda).

        Note:
            Stores the recovered ``lambda_i`` in ``_adjoint_lambda_interior``
            for use by the next ``get_adjoint_step_reduced_rhs()`` call.
        """
        tbs = self._transient_block_system
        bs = self._block_system

        if tbs is None:
            raise RuntimeError(
                "recover_and_accumulate_adjoint() requires "
                "factor_transient_system() to be called first"
            )

        n_ports = bs.n_ports
        n_interior = bs.n_interior

        lambda_p = np.zeros(n_ports, dtype=np.float64)
        for node, idx in bs.port_to_idx.items():
            if node in lambda_p_dict:
                lambda_p[idx] = lambda_p_dict[node]

        # Recover interior lambda: lambda_i = lu_ii(s_i - A_ip @ lambda_p)
        # where s_i is the interior source term (e_i for terminal, B_ii @ lambda_i_old for non-terminal)
        # A_ip is tbs.G_ip (transient block system, since tile caps are grounded: A_ip = G_ip)
        if n_interior > 0 and tbs.lu_ii is not None:
            # Compute A_ip @ lambda_p
            a_ip_lambda_p = tbs.G_ip @ lambda_p

            if source_term_i is not None:
                # lambda_i = lu_ii(s_i - A_ip @ lambda_p)
                lambda_i = tbs.lu_ii(source_term_i - a_ip_lambda_p)
            else:
                # No interior source term: lambda_i = -lu_ii(A_ip @ lambda_p)
                lambda_i = tbs.lu_ii(-a_ip_lambda_p)
        else:
            lambda_i = np.array([], dtype=np.float64)

        # Store for next get_adjoint_step_reduced_rhs() call
        self._adjoint_lambda_interior = lambda_i

        n_accumulated = self._accumulate_contributions(
            lambda_i, lambda_p, t, include_waveforms,
        )

        lambda_i_norm = float(np.linalg.norm(lambda_i)) if len(lambda_i) > 0 else 0.0

        tid = self._tile_data.tile_id if self._tile_data else '?'
        logger.debug(
            "Tile %s recover_and_accumulate_adjoint: "
            "t=%.2e, n_accumulated=%d, lambda_i_norm=%.2e",
            tid, t, n_accumulated, lambda_i_norm,
        )

        return {
            'n_accumulated': n_accumulated,
            'lambda_i_norm': lambda_i_norm,
        }

    def recover_and_accumulate_adjoint_terminal(
        self,
        lambda_p_dict: Dict[str, float],
        t: float,
        victim_node: str,
        include_waveforms: bool,
    ) -> Dict[str, Any]:
        """Recover interior lambda and accumulate for terminal step.

        This is the terminal step variant where the source term is e_victim
        (unit vector at victim node) rather than B @ lambda_old.

        For terminal step:
            - If victim is interior: lambda_i = lu_ii(e_i - A_ip @ lambda_p)
            - Otherwise: lambda_i = -lu_ii(A_ip @ lambda_p)

        Args:
            lambda_p_dict: Interface lambda values from coordinator solve,
                keyed by node name.
            t: Current time point (seconds), typically observation_time.
            victim_node: The victim node name (used to construct e_victim).
            include_waveforms: Whether to record (t, I) waveform data per source.

        Returns:
            Stats dict with ``'n_accumulated'``, ``'lambda_i_norm'``.

        Raises:
            RuntimeError: If transient block system not factored.
        """
        tbs = self._transient_block_system
        bs = self._block_system

        if tbs is None:
            raise RuntimeError(
                "recover_and_accumulate_adjoint_terminal() requires "
                "factor_transient_system() to be called first"
            )

        n_ports = bs.n_ports
        n_interior = bs.n_interior

        lambda_p = np.zeros(n_ports, dtype=np.float64)
        for node, idx in bs.port_to_idx.items():
            if node in lambda_p_dict:
                lambda_p[idx] = lambda_p_dict[node]

        e_i = np.zeros(n_interior, dtype=np.float64)
        victim_is_interior = victim_node in bs.interior_to_idx
        if victim_is_interior:
            victim_idx = bs.interior_to_idx[victim_node]
            e_i[victim_idx] = 1.0

        # Recover interior lambda: lambda_i = lu_ii(e_i - A_ip @ lambda_p)
        # A_ip = tbs.G_ip (tile caps are grounded, so A_ip = G_ip)
        if n_interior > 0 and tbs.lu_ii is not None:
            a_ip_lambda_p = tbs.G_ip @ lambda_p
            lambda_i = tbs.lu_ii(e_i - a_ip_lambda_p)
        else:
            lambda_i = np.array([], dtype=np.float64)

        # Store for next get_adjoint_step_reduced_rhs() call
        self._adjoint_lambda_interior = lambda_i

        n_accumulated = self._accumulate_contributions(
            lambda_i, lambda_p, t, include_waveforms,
        )

        lambda_i_norm = float(np.linalg.norm(lambda_i)) if len(lambda_i) > 0 else 0.0

        tid = self._tile_data.tile_id if self._tile_data else '?'
        logger.debug(
            "Tile %s recover_and_accumulate_adjoint_terminal: "
            "t=%.2e, victim=%s (is_interior=%s), n_accumulated=%d, lambda_i_norm=%.2e",
            tid, t, victim_node, victim_is_interior, n_accumulated, lambda_i_norm,
        )

        return {
            'n_accumulated': n_accumulated,
            'lambda_i_norm': lambda_i_norm,
        }

    # ------------------------------------------------------------------
    # Static adjoint recovery and accumulation (DC path)
    # ------------------------------------------------------------------

    def recover_and_accumulate_adjoint_static(
        self,
        lambda_p_dict: Dict[str, float],
        victim_node: str,
        t: float,
        include_waveforms: bool,
    ) -> Dict[str, Any]:
        """Recover interior lambda and accumulate contributions (static/DC path).

        This is the static adjoint version that uses the DC block system (G)
        instead of the transient block system (A = G + C*C).

        For static adjoint, we solve:
            ``G @ lambda = e_victim``

        The Schur reduction gives:
            ``lambda_i = lu_G_ii^{-1}(e_i - G_ip @ lambda_p)``

        where ``e_i`` has 1.0 at victim if victim is interior, else zeros.

        Args:
            lambda_p_dict: Interface lambda values from coordinator solve,
                keyed by node name.
            victim_node: The victim node name (needed to construct e_i).
            t: Current time point (seconds) for current evaluation.
            include_waveforms: Whether to record (t, I) waveform data per source.

        Returns:
            Stats dict with ``'n_accumulated'``, ``'lambda_i_norm'``.

        Raises:
            RuntimeError: If DC block system not factored.
        """
        bs = self._block_system
        if bs is None:
            raise RuntimeError(
                "recover_and_accumulate_adjoint_static() called before setup(); "
                "no block system available"
            )
        if bs.lu_ii is None and bs.n_interior > 0:
            raise RuntimeError(
                "recover_and_accumulate_adjoint_static() requires "
                "factor_and_compute_schur() to be called first"
            )

        n_ports = bs.n_ports
        n_interior = bs.n_interior

        lambda_p = np.zeros(n_ports, dtype=np.float64)
        for node, idx in bs.port_to_idx.items():
            if node in lambda_p_dict:
                lambda_p[idx] = lambda_p_dict[node]

        e_i = np.zeros(n_interior, dtype=np.float64)
        if victim_node in bs.interior_to_idx:
            victim_idx = bs.interior_to_idx[victim_node]
            e_i[victim_idx] = 1.0

        # Recover interior lambda: lambda_i = lu_ii(e_i - G_ip @ lambda_p)
        if n_interior > 0 and bs.lu_ii is not None:
            g_ip_lambda_p = bs.G_ip @ lambda_p
            lambda_i = bs.lu_ii(e_i - g_ip_lambda_p)
        else:
            lambda_i = np.array([], dtype=np.float64)

        # Store for reference (though static doesn't use it for recursion)
        self._adjoint_lambda_interior = lambda_i

        n_accumulated = self._accumulate_contributions(
            lambda_i, lambda_p, t, include_waveforms,
        )

        lambda_i_norm = float(np.linalg.norm(lambda_i)) if len(lambda_i) > 0 else 0.0

        tid = self._tile_data.tile_id if self._tile_data else '?'
        logger.debug(
            "Tile %s recover_and_accumulate_adjoint_static: "
            "t=%.2e, n_accumulated=%d, lambda_i_norm=%.2e",
            tid, t, n_accumulated, lambda_i_norm,
        )

        return {
            'n_accumulated': n_accumulated,
            'lambda_i_norm': lambda_i_norm,
        }

    # ------------------------------------------------------------------
    # Adjoint results retrieval
    # ------------------------------------------------------------------

    def get_adjoint_results(
        self,
    ) -> Tuple[Dict[str, float], Dict[str, str], Dict[str, List[Tuple[float, float]]]]:
        """Return accumulated contributions and supporting data.

        Returns:
            Tuple of:
            - contributions: Dict mapping source_name to total contribution (V).
              Note: Values are in Volts, NOT millivolts. Caller should multiply
              by 1000 if mV output is desired.
            - source_to_node: Dict mapping source_name to node_name (for
              coordinate lookup by the caller).
            - waveforms: Dict mapping source_name to list of (time, current)
              tuples. Empty if ``include_waveforms=False`` was used during
              accumulation.
        """
        return (
            dict(self._adjoint_contributions),
            dict(self._adjoint_source_to_node),
            dict(self._adjoint_waveforms),
        )

"""Effective resistance computation for unified power grids.

Computes effective resistance between node pairs or from nodes to ground
(voltage source nodes).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import scipy.sparse.linalg as spla

from .unified_model import UnifiedPowerGridModel


class UnifiedEffectiveResistanceCalculator:
    """Compute effective resistance for unified power grids.

    Effective resistance R_eff(u, v) is the voltage drop across nodes u and v
    when 1A of current is injected at u and extracted at v.

    For node-to-ground: R_eff(u, ground) = G_uu^-1[u, u] where G_uu is the
    reduced conductance matrix.

    Example:
        model = create_model_from_synthetic(G, pads, vdd=1.0)
        calc = UnifiedEffectiveResistanceCalculator(model)

        # Resistance from node to ground
        r_to_ground = calc.compute_single(node, target=None)

        # Resistance between two nodes
        r_pairwise = calc.compute_single(node1, node2)

        # Batch computation
        pairs = [(n1, None), (n2, None), (n1, n2)]
        results = calc.compute_batch(pairs)
    """

    def __init__(self, model: UnifiedPowerGridModel):
        """Initialize calculator with a unified model.

        Args:
            model: UnifiedPowerGridModel instance (must have pads for grounding)
        """
        self.model = model
        self._cache: Dict[Any, np.ndarray] = {}  # node -> solution vector

    def compute_single(
        self,
        source: Any,
        target: Optional[Any] = None,
    ) -> float:
        """Compute effective resistance for a single query.

        Args:
            source: Source node
            target: Target node, or None for resistance to ground (pads)

        Returns:
            Effective resistance in Ohms

        Raises:
            ValueError: If source or target is a pad node
        """
        rs = self.model.reduced
        pad_set = set(rs.pad_nodes)

        if source in pad_set:
            raise ValueError(f"Source node {source} is a pad (resistance to ground = 0)")

        if target is not None and target in pad_set:
            raise ValueError(f"Target node {target} is a pad (resistance to ground = 0)")

        if source not in rs.index_unknown:
            raise ValueError(f"Source node {source} not in model")

        if target is not None and target not in rs.index_unknown:
            raise ValueError(f"Target node {target} not in model")

        # Get or compute G_uu^-1 column for source
        x_s = self._get_solution(source)

        if target is None:
            # Resistance to ground = diagonal of G_uu^-1
            return float(x_s[rs.index_unknown[source]])
        else:
            # Pairwise resistance
            x_t = self._get_solution(target)
            i_s = rs.index_unknown[source]
            i_t = rs.index_unknown[target]

            # R_eff(s, t) = G^-1[s,s] + G^-1[t,t] - 2*G^-1[s,t]
            r_eff = x_s[i_s] + x_t[i_t] - 2.0 * x_s[i_t]
            return float(max(0.0, r_eff))  # Ensure non-negative

    def compute_batch(
        self,
        pairs: Sequence[Tuple[Any, Optional[Any]]],
    ) -> np.ndarray:
        """Compute effective resistance for multiple queries.

        Args:
            pairs: List of (source, target) tuples. target=None means to ground.

        Returns:
            numpy array of resistance values

        Example:
            pairs = [(node1, None), (node2, None), (node1, node2)]
            results = calc.compute_batch(pairs)
            # results[0] = R_eff(node1, ground)
            # results[1] = R_eff(node2, ground)
            # results[2] = R_eff(node1, node2)
        """
        results = np.zeros(len(pairs), dtype=float)

        for i, (source, target) in enumerate(pairs):
            try:
                results[i] = self.compute_single(source, target)
            except ValueError:
                results[i] = 0.0  # Pad nodes have 0 resistance to ground

        return results

    def compute_to_ground(self, nodes: Sequence[Any]) -> Dict[Any, float]:
        """Compute resistance to ground for multiple nodes.

        Args:
            nodes: Sequence of nodes

        Returns:
            Dict mapping node -> resistance to ground
        """
        results = {}
        for node in nodes:
            try:
                results[node] = self.compute_single(node, target=None)
            except ValueError:
                results[node] = 0.0
        return results

    def _get_solution(self, node: Any) -> np.ndarray:
        """Get or compute G_uu^-1 * e_node.

        Args:
            node: Node to get solution for

        Returns:
            Solution vector (column of G_uu^-1)
        """
        if node in self._cache:
            return self._cache[node]

        rs = self.model.reduced

        # Create unit vector
        e = np.zeros(len(rs.unknown_nodes), dtype=float)
        e[rs.index_unknown[node]] = 1.0

        # Solve G_uu * x = e
        x = rs.lu(e)

        self._cache[node] = x
        return x

    def clear_cache(self) -> None:
        """Clear the solution cache."""
        self._cache.clear()


# Convenience function for backward compatibility

def compute_effective_resistance(
    model: UnifiedPowerGridModel,
    source: Any,
    target: Optional[Any] = None,
) -> float:
    """Compute effective resistance (convenience function).

    Args:
        model: UnifiedPowerGridModel
        source: Source node
        target: Target node or None for ground

    Returns:
        Effective resistance in Ohms
    """
    calc = UnifiedEffectiveResistanceCalculator(model)
    return calc.compute_single(source, target)

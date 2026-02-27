"""Stimulus generation for static IR-drop analysis.

Provides utilities to create current sink vectors (node->current) given a
total power P and either a percentage or explicit count of load nodes to
activate. Currents are distributed uniformly or using a Gaussian weight.

Convention: Returned currents are positive numbers representing current DRAWN
from the grid at that node (sink). Solver layer converts to injection sign.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence
import random
import numpy as np


@dataclass
class StimulusMeta:
    total_power: float
    total_current: float
    vdd: float
    selected_nodes: List
    distribution: str
    currents: Dict


class StimulusGenerator:
    def __init__(self, load_nodes: Sequence, vdd: float = 1.0, seed: int | None = None, graph=None):
        """Stimulus generator.

        Parameters
        ----------
        load_nodes : Sequence
            Iterable of candidate load (tap) nodes.
        vdd : float
            Supply voltage used to convert power -> current.
        seed : int | None
            RNG seed for reproducible node sampling.
        graph : networkx.Graph | None
            Optional reference graph to enable geometric filtering by area.
            If provided, nodes must have 'xy' attribute.
        """
        self.load_nodes = list(load_nodes)
        self.vdd = float(vdd)
        self.rng = random.Random(seed)
        self.graph = graph

    def _filter_area(self, nodes: Sequence, area: tuple | None) -> List:
        """Filter nodes by rectangular area (x_min, y_min, x_max, y_max).

        Returns nodes whose xy lies strictly inside the box (inclusive on edges).
        If graph or area is None, returns nodes unchanged.
        """
        if area is None or self.graph is None:
            return list(nodes)
        x_min, y_min, x_max, y_max = area
        selected = []
        for n in nodes:
            data = self.graph.nodes.get(n)
            if not data:
                continue
            xy = data.get("xy")
            if not xy:
                continue
            x, y = xy
            if x_min <= x <= x_max and y_min <= y <= y_max:
                selected.append(n)
        return selected

    def _select_nodes(self, percent: float | None, count: int | None, area: tuple | None) -> List:
        if not self.load_nodes:
            return []
        # If area specified, restrict candidate pool before sampling
        candidates = self._filter_area(self.load_nodes, area) if area else list(self.load_nodes)
        if not candidates:
            return []
        if count is None and percent is None:
            base = candidates
        elif count is not None:
            count = max(0, min(count, len(candidates)))
            base = self.rng.sample(candidates, count) if count > 0 else []
        else:
            p = max(0.0, min(float(percent), 1.0))
            c = int(round(p * len(candidates)))
            c = max(0, min(c, len(candidates)))
            if c == 0 and p > 0:
                c = 1
            base = self.rng.sample(candidates, c) if c > 0 else []
        return base

    def generate(
        self,
        total_power: float,
        percent: float | None = None,
        count: int | None = None,
        distribution: str = "uniform",
        gaussian_loc: float = 1.0,
        gaussian_scale: float = 1.0,
        area: tuple | None = None,
        seed: int | None = None,
    ) -> StimulusMeta:
        """Generate one stimulus mapping node->current (Amps).

        total_power: Watts consumed by selected load nodes.
        percent / count: choose nodes (exclusive; count takes precedence if provided).
        distribution: 'uniform' or 'gaussian'.
        gaussian_loc/scale: parameters for Normal(loc, scale) used when gaussian.
        area: optional (x_min, y_min, x_max, y_max) rectangle; only nodes inside are used.
        seed: optional RNG seed for this specific generation call. If provided, 
              overrides the generator's state to ensure reproducible node selection
              and current distribution.
        """
        assert total_power >= 0.0, "Power must be non-negative"
        
        # Save current RNG state if seed is provided
        saved_state = None
        if seed is not None:
            saved_state = self.rng.getstate()
            self.rng.seed(seed)
        
        # Save numpy RNG state if using gaussian distribution
        saved_np_state = None
        if distribution == "gaussian" and seed is not None:
            saved_np_state = np.random.get_state()
            np.random.seed(seed)
        
        try:
            nodes = self._select_nodes(percent if count is None else None, count, area)
            if not nodes:
                return StimulusMeta(total_power, 0.0, self.vdd, [], distribution, {})
            total_current = total_power / self.vdd if self.vdd > 0 else 0.0
            currents: Dict = {}
            if distribution == "uniform":
                each = total_current / len(nodes)
                for n in nodes:
                    currents[n] = each
            elif distribution == "gaussian":
                raw = np.abs(np.random.normal(loc=gaussian_loc, scale=gaussian_scale, size=len(nodes)))
                s = raw.sum()
                if s <= 0:
                    # fallback uniform
                    each = total_current / len(nodes)
                    for n in nodes:
                        currents[n] = each
                else:
                    weights = raw / s
                    for n, w in zip(nodes, weights):
                        currents[n] = w * total_current
            else:
                raise ValueError(f"Unknown distribution: {distribution}")
            return StimulusMeta(total_power, total_current, self.vdd, nodes, distribution, currents)
        finally:
            # Restore RNG states if they were saved
            if saved_state is not None:
                self.rng.setstate(saved_state)
            if saved_np_state is not None:
                np.random.set_state(saved_np_state)

    def generate_batch(
        self,
        powers: Sequence[float],
        percent: float | None = None,
        count: int | None = None,
        distribution: str = "uniform",
        gaussian_loc: float = 1.0,
        gaussian_scale: float = 1.0,
        area: tuple | None = None,
        seed: int | None = None,
    ) -> List[StimulusMeta]:
        metas: List[StimulusMeta] = []
        for P in powers:
            metas.append(
                self.generate(
                    total_power=P,
                    percent=percent,
                    count=count,
                    distribution=distribution,
                    gaussian_loc=gaussian_loc,
                    gaussian_scale=gaussian_scale,
                    area=area,
                    seed=seed,
                )
            )
        return metas

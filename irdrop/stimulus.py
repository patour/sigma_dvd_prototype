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
    def __init__(self, load_nodes: Sequence, vdd: float = 1.0, seed: int | None = None):
        self.load_nodes = list(load_nodes)
        self.vdd = float(vdd)
        self.rng = random.Random(seed)

    def _select_nodes(self, percent: float | None, count: int | None) -> List:
        if not self.load_nodes:
            return []
        if count is None and percent is None:
            # default: use all
            return list(self.load_nodes)
        if count is not None:
            count = max(0, min(count, len(self.load_nodes)))
            return self.rng.sample(self.load_nodes, count) if count > 0 else []
        # percent path
        p = max(0.0, min(float(percent), 1.0))
        c = int(round(p * len(self.load_nodes)))
        c = max(0, min(c, len(self.load_nodes)))
        if c == 0 and p > 0:
            c = 1  # ensure at least one if p>0
        return self.rng.sample(self.load_nodes, c) if c > 0 else []

    def generate(
        self,
        total_power: float,
        percent: float | None = None,
        count: int | None = None,
        distribution: str = "uniform",
        gaussian_loc: float = 1.0,
        gaussian_scale: float = 1.0,
    ) -> StimulusMeta:
        """Generate one stimulus mapping node->current (Amps).

        total_power: Watts consumed by selected load nodes.
        percent / count: choose nodes (exclusive; count takes precedence if provided).
        distribution: 'uniform' or 'gaussian'.
        gaussian_loc/scale: parameters for Normal(loc, scale) used when gaussian.
        """
        assert total_power >= 0.0, "Power must be non-negative"
        nodes = self._select_nodes(percent if count is None else None, count)
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

    def generate_batch(
        self,
        powers: Sequence[float],
        percent: float | None = None,
        count: int | None = None,
        distribution: str = "uniform",
        gaussian_loc: float = 1.0,
        gaussian_scale: float = 1.0,
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
                )
            )
        return metas

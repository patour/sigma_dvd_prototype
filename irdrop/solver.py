"""High-level IR-drop solver orchestration.

Provides IRDropSolver which wraps a PowerGridModel and exposes convenient
methods returning voltage & IR-drop results for single and batch stimuli.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

from .power_grid_model import PowerGridModel


@dataclass
class IRDropResult:
    voltages: Dict  # node -> voltage
    ir_drop: Dict   # node -> (vdd - voltage)
    metadata: Dict  # stimulus meta or arbitrary info


class IRDropSolver:
    def __init__(self, model: PowerGridModel):
        self.model = model

    def solve(self, stimulus: Dict, metadata: Dict | None = None) -> IRDropResult:
        voltages = self.model.solve_voltages(stimulus)
        drops = self.model.ir_drop(voltages, self.model.vdd)
        return IRDropResult(voltages=voltages, ir_drop=drops, metadata=metadata or {})

    def solve_batch(self, stimuli: Sequence[Dict], metas: Sequence | None = None) -> List[IRDropResult]:
        volt_list = self.model.solve_batch(stimuli)
        results: List[IRDropResult] = []
        for i, volts in enumerate(volt_list):
            drops = self.model.ir_drop(volts, self.model.vdd)
            meta = metas[i] if metas is not None else {}
            results.append(IRDropResult(voltages=volts, ir_drop=drops, metadata=meta))
        return results

    @staticmethod
    def summarize(result: IRDropResult) -> Dict:
        """Return basic summary stats: min voltage, max drop, average drop."""
        volt_values = list(result.voltages.values())
        drop_values = list(result.ir_drop.values())
        return {
            "min_voltage": float(np.min(volt_values)) if volt_values else None,
            "max_drop": float(np.max(drop_values)) if drop_values else None,
            "avg_drop": float(np.mean(drop_values)) if drop_values else None,
        }

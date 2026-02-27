"""IR-drop analysis package for power grid simulation.

Exports core classes for building models, generating stimuli, and solving IR-drop.
"""

from .power_grid_model import PowerGridModel, ReducedSystem, HierarchicalSolveResult
from .stimulus import StimulusGenerator, StimulusMeta
from .solver import IRDropSolver, IRDropResult
from .effective_resistance import EffectiveResistanceCalculator
from .regional_voltage_solver import RegionalIRDropSolver
from .grid_partitioner import GridPartitioner
from .plot import plot_voltage_map, plot_ir_drop_map

__all__ = [
    "PowerGridModel",
    "ReducedSystem",
    "HierarchicalSolveResult",
    "StimulusGenerator",
    "StimulusMeta",
    "IRDropSolver",
    "IRDropResult",
    "EffectiveResistanceCalculator",
    "RegionalIRDropSolver",
    "GridPartitioner",
    "plot_voltage_map",
    "plot_ir_drop_map",
]


"""Unified power grid model core package.

Provides a unified interface for both synthetic power grids and PDN netlists.
"""

from .node_adapter import NodeInfoExtractor, UnifiedNodeInfo
from .edge_adapter import EdgeInfoExtractor, UnifiedEdgeInfo, ElementType
from .unified_model import UnifiedPowerGridModel, UnifiedReducedSystem, GridSource, EdgeArrayCache
from .factory import (
    create_model_from_synthetic,
    create_model_from_pdn,
    create_multi_net_models,
    create_model_from_graph,
)
from .unified_solver import UnifiedIRDropSolver, UnifiedSolveResult, UnifiedHierarchicalResult
from .unified_plotter import UnifiedPlotter, plot_voltage_map, plot_ir_drop_map
from .unified_partitioner import UnifiedPartitioner, UnifiedPartition, UnifiedPartitionResult
from .statistics import UnifiedStatistics, GridStats, LayerStats
from .effective_resistance import UnifiedEffectiveResistanceCalculator, compute_effective_resistance

__all__ = [
    # Node adapter
    "NodeInfoExtractor",
    "UnifiedNodeInfo",
    # Edge adapter
    "EdgeInfoExtractor",
    "UnifiedEdgeInfo",
    "ElementType",
    # Unified model
    "UnifiedPowerGridModel",
    "UnifiedReducedSystem",
    "GridSource",
    "EdgeArrayCache",
    # Factory functions
    "create_model_from_synthetic",
    "create_model_from_pdn",
    "create_multi_net_models",
    "create_model_from_graph",
    # Solver
    "UnifiedIRDropSolver",
    "UnifiedSolveResult",
    "UnifiedHierarchicalResult",
    # Plotter
    "UnifiedPlotter",
    "plot_voltage_map",
    "plot_ir_drop_map",
    # Partitioner
    "UnifiedPartitioner",
    "UnifiedPartition",
    "UnifiedPartitionResult",
    # Statistics
    "UnifiedStatistics",
    "GridStats",
    "LayerStats",
    # Effective resistance
    "UnifiedEffectiveResistanceCalculator",
    "compute_effective_resistance",
]

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
from .unified_solver import (
    UnifiedIRDropSolver,
    UnifiedSolveResult,
    UnifiedHierarchicalResult,
    UnifiedCoupledHierarchicalResult,
    TiledBottomGridResult,
    TileBounds,
    BottomGridTile,
)
from .solver_results import (
    FlatSolverContext,
    HierarchicalSolverContext,
    CoupledHierarchicalSolverContext,
    TiledHierarchicalSolverContext,
)
from .coupled_system import (
    BlockMatrixSystem,
    SchurComplementOperator,
    CoupledSystemOperator,
    extract_block_matrices,
    AMGPreconditioner,
    HAS_PYAMG,
)
from .unified_plotter import UnifiedPlotter, plot_voltage_map, plot_ir_drop_map
from .unified_partitioner import UnifiedPartitioner, UnifiedPartition, UnifiedPartitionResult
from .statistics import UnifiedStatistics, GridStats, LayerStats
from .effective_resistance import UnifiedEffectiveResistanceCalculator, compute_effective_resistance
from .graph_converter import (
    detect_graph_type,
    is_networkx_graph,
    is_rustworkx_graph,
    convert_networkx_to_rustworkx,
    ensure_rustworkx_graph,
)
from .dynamic_solver import (
    DynamicIRDropSolver,
    QuasiStaticResult,
)
from .transient_solver import (
    TransientIRDropSolver,
    TransientResult,
    RCSystem,
    IntegrationMethod,
)
from .dynamic_plotter import (
    DynamicPlotter,
    plot_peak_ir_drop_heatmap,
    plot_peak_current_heatmap,
    plot_time_series,
    plot_node_waveforms,
)

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
    "UnifiedCoupledHierarchicalResult",
    "TiledBottomGridResult",
    "TileBounds",
    "BottomGridTile",
    # Solver contexts for batch solving
    "FlatSolverContext",
    "HierarchicalSolverContext",
    "CoupledHierarchicalSolverContext",
    "TiledHierarchicalSolverContext",
    # Coupled system
    "BlockMatrixSystem",
    "SchurComplementOperator",
    "CoupledSystemOperator",
    "extract_block_matrices",
    "AMGPreconditioner",
    "HAS_PYAMG",
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
    # Graph converter
    "detect_graph_type",
    "is_networkx_graph",
    "is_rustworkx_graph",
    "convert_networkx_to_rustworkx",
    "ensure_rustworkx_graph",
    # Dynamic solver (quasi-static)
    "DynamicIRDropSolver",
    "QuasiStaticResult",
    # Transient solver (RC)
    "TransientIRDropSolver",
    "TransientResult",
    "RCSystem",
    "IntegrationMethod",
    # Dynamic plotter
    "DynamicPlotter",
    "plot_peak_ir_drop_heatmap",
    "plot_peak_current_heatmap",
    "plot_time_series",
    "plot_node_waveforms",
]

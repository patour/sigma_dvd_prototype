"""Analysis scripts for PDN IR-drop analysis.

This package provides scripts for advanced analysis of power delivery networks,
including dynamic IR-drop decomposition into near and far contributions.
"""

from .dynamic_irdrop_decomposition import (
    analyze_dynamic_irdrop_decomposition,
    InstanceDecomposition,
    DecompositionResult,
    parse_time_value,
)

from .stripe_heatmap import (
    plot_stripe_heatmap,
    parse_node_info,
    detect_orientation_from_coords,
)

__all__ = [
    'analyze_dynamic_irdrop_decomposition',
    'InstanceDecomposition',
    'DecompositionResult',
    'parse_time_value',
    'plot_stripe_heatmap',
    'parse_node_info',
    'detect_orientation_from_coords',
]

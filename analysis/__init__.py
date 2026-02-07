"""Analysis scripts for PDN IR-drop analysis.

This package provides scripts for advanced analysis of power delivery networks,
including dynamic IR-drop decomposition into near and far contributions.

Uses lazy imports (PEP 562) to avoid RuntimeWarning when running modules directly
with `python -m analysis.<module>`.
"""

__all__ = [
    'analyze_dynamic_irdrop_decomposition',
    'InstanceDecomposition',
    'DecompositionResult',
    'parse_time_value',
    'plot_stripe_heatmap',
    'parse_node_info',
    'detect_orientation_from_coords',
]


def __getattr__(name):
    if name in ('analyze_dynamic_irdrop_decomposition', 'InstanceDecomposition',
                'DecompositionResult', 'parse_time_value'):
        from .dynamic_irdrop_decomposition import (
            analyze_dynamic_irdrop_decomposition,
            InstanceDecomposition,
            DecompositionResult,
            parse_time_value,
        )
        globals().update({
            'analyze_dynamic_irdrop_decomposition': analyze_dynamic_irdrop_decomposition,
            'InstanceDecomposition': InstanceDecomposition,
            'DecompositionResult': DecompositionResult,
            'parse_time_value': parse_time_value,
        })
        return globals()[name]

    if name in ('plot_stripe_heatmap', 'parse_node_info', 'detect_orientation_from_coords'):
        from .stripe_heatmap import (
            plot_stripe_heatmap,
            parse_node_info,
            detect_orientation_from_coords,
        )
        globals().update({
            'plot_stripe_heatmap': plot_stripe_heatmap,
            'parse_node_info': parse_node_info,
            'detect_orientation_from_coords': detect_orientation_from_coords,
        })
        return globals()[name]

    raise AttributeError(f"module 'analysis' has no attribute {name!r}")

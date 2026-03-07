"""Reports package for IR-drop analysis.

This package provides report generation utilities for power grid analysis,
including floating nodes detection and statistics.
"""

from .floating_nodes import (
    FloatingNodesData,
    collect_floating_nodes_distributed,
    collect_floating_nodes_flat,
    generate_floating_nodes_report,
)
from .topk_irdrop import generate_topk_report

__all__ = [
    'FloatingNodesData',
    'collect_floating_nodes_distributed',
    'collect_floating_nodes_flat',
    'generate_floating_nodes_report',
    'generate_topk_report',
]

"""Node adapter for unified power grid model.

Extracts unified node information from various node representations:
- Synthetic grids: NodeID(layer, idx) dataclass with xy attribute in graph
- PDN netlists: String names with X_Y_LAYER format
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import networkx as nx


# Type alias for coordinates
Coordinate = Tuple[float, float]

# Type alias for layer identifiers (can be int or str)
LayerID = Union[int, str]


@dataclass(frozen=True)
class UnifiedNodeInfo:
    """Unified node information extracted from any node representation.

    Attributes:
        x: X coordinate (None if unavailable)
        y: Y coordinate (None if unavailable)
        layer: Layer identifier (int for synthetic, str for PDN, None if unavailable)
        original_node: Original node object for identity preservation
    """
    x: Optional[float]
    y: Optional[float]
    layer: Optional[LayerID]
    original_node: Any

    @property
    def xy(self) -> Optional[Coordinate]:
        """Return (x, y) tuple if both coordinates are available."""
        if self.x is not None and self.y is not None:
            return (self.x, self.y)
        return None

    @property
    def layer_numeric(self) -> Optional[int]:
        """Convert layer to numeric if possible.

        Returns:
            Integer layer number, or None if not convertible.
        """
        if isinstance(self.layer, int):
            return self.layer
        if isinstance(self.layer, str):
            # Try direct conversion
            if self.layer.isdigit():
                return int(self.layer)
            # Try extracting number from layer name (e.g., "M1" -> 1)
            match = re.search(r'(\d+)', self.layer)
            if match:
                return int(match.group(1))
        return None

    def __hash__(self) -> int:
        """Hash based on original node for identity."""
        return hash(self.original_node)

    def __eq__(self, other: Any) -> bool:
        """Equality based on original node."""
        if isinstance(other, UnifiedNodeInfo):
            return self.original_node == other.original_node
        return False


class NodeInfoExtractor:
    """Extracts unified node info from various node representations.

    Supports:
    - Synthetic grids with NodeID objects (layer, idx attributes)
    - PDN netlists with string node names (X_Y_LAYER format)
    - Generic graphs with node attributes (x, y, layer)

    Caches results for performance.
    """

    # Pattern for PDN node names: X_Y_LAYER (e.g., "1000_2000_M1")
    PDN_3D_PATTERN = re.compile(r'^(\d+)_(\d+)_(.+?)(?:_.*)?$')
    # Pattern for 2D node names: X_Y (e.g., "1000_2000")
    PDN_2D_PATTERN = re.compile(r'^(\d+)_(\d+)$')

    def __init__(self, graph: nx.Graph):
        """Initialize extractor with a graph.

        Args:
            graph: NetworkX graph (Graph or MultiDiGraph)
        """
        self.graph = graph
        self._cache: Dict[Any, UnifiedNodeInfo] = {}

    def get_info(self, node: Any) -> UnifiedNodeInfo:
        """Extract unified info from any node type.

        Args:
            node: Node identifier (NodeID, string, or other hashable)

        Returns:
            UnifiedNodeInfo with extracted coordinates and layer.
        """
        if node in self._cache:
            return self._cache[node]

        info = self._extract_info(node)
        self._cache[node] = info
        return info

    def get_xy(self, node: Any) -> Optional[Coordinate]:
        """Get (x, y) coordinates for a node.

        Args:
            node: Node identifier

        Returns:
            (x, y) tuple or None if unavailable.
        """
        return self.get_info(node).xy

    def get_layer(self, node: Any) -> Optional[LayerID]:
        """Get layer identifier for a node.

        Args:
            node: Node identifier

        Returns:
            Layer (int or str) or None if unavailable.
        """
        return self.get_info(node).layer

    def clear_cache(self) -> None:
        """Clear the info cache."""
        self._cache.clear()

    def _extract_info(self, node: Any) -> UnifiedNodeInfo:
        """Extract node info based on node type.

        Args:
            node: Node identifier

        Returns:
            UnifiedNodeInfo with extracted data.
        """
        # Get node data from graph if available
        node_data = self.graph.nodes.get(node, {})

        # Try NodeID-style (synthetic grids) - has layer and idx attributes
        if hasattr(node, 'layer') and hasattr(node, 'idx'):
            return self._extract_from_nodeid(node, node_data)

        # Try string node (PDN style)
        if isinstance(node, str):
            return self._extract_from_string(node, node_data)

        # Fallback: try graph attributes only
        return self._extract_from_attrs(node, node_data)

    def _extract_from_nodeid(self, node: Any, node_data: Dict) -> UnifiedNodeInfo:
        """Extract info from NodeID-style object.

        NodeID has 'layer' and 'idx' attributes.
        Coordinates come from 'xy' attribute in graph node data.
        """
        layer = node.layer

        # Get coordinates from graph node data
        xy = node_data.get('xy')
        if xy is not None:
            x, y = xy[0], xy[1]
        else:
            x = node_data.get('x')
            y = node_data.get('y')

        return UnifiedNodeInfo(x=x, y=y, layer=layer, original_node=node)

    def _extract_from_string(self, node: str, node_data: Dict) -> UnifiedNodeInfo:
        """Extract info from string node name.

        Tries:
        1. Explicit attributes in graph data (x, y, layer)
        2. Parse from name using X_Y_LAYER pattern
        3. Parse from name using X_Y pattern
        """
        # First try explicit attributes
        x = node_data.get('x')
        y = node_data.get('y')
        layer = node_data.get('layer')

        # If coords missing, try parsing from name
        if x is None or y is None:
            # Try 3D pattern: X_Y_LAYER
            match = self.PDN_3D_PATTERN.match(node)
            if match:
                x = float(match.group(1))
                y = float(match.group(2))
                if layer is None:
                    layer = match.group(3)
            else:
                # Try 2D pattern: X_Y
                match = self.PDN_2D_PATTERN.match(node)
                if match:
                    x = float(match.group(1))
                    y = float(match.group(2))

        return UnifiedNodeInfo(x=x, y=y, layer=layer, original_node=node)

    def _extract_from_attrs(self, node: Any, node_data: Dict) -> UnifiedNodeInfo:
        """Extract info from graph attributes only.

        Fallback for nodes that are neither NodeID nor string.
        """
        x = node_data.get('x')
        y = node_data.get('y')
        layer = node_data.get('layer')

        # Try 'xy' tuple attribute
        if x is None or y is None:
            xy = node_data.get('xy')
            if xy is not None:
                x, y = xy[0], xy[1]

        return UnifiedNodeInfo(x=x, y=y, layer=layer, original_node=node)

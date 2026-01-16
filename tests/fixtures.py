"""Test fixtures for hierarchical solver unit tests.

Provides factory functions to create minimal PDN graphs for specific
edge case testing scenarios that cannot be triggered with netlist_small.
"""

from typing import Dict, List, Set, Tuple, Any
import networkx as nx


def create_minimal_pdn_graph(
    scenario: str
) -> Tuple[nx.MultiDiGraph, List[str], Dict[str, float]]:
    """Create minimal PDN graph for specific edge case testing.

    Args:
        scenario: One of:
            - 'tile_merging': 4x4 M1 + 2x2 M2 grid with loads clustered in
              one corner. 2x2 tiling produces tiles with 0 loads -> Phase 3 merge.
            - 'path_expansion': Grid with sparse vias causing locally disconnected
              core nodes -> triggers halo path expansion.
            - 'severe_halo_clip': 6x6 grid where 3x3 tiling clips halos >70%.
            - 'basic': Simple 3x3 M1 + 2x2 M2 for basic tests.

    Returns:
        (graph, pad_nodes, load_currents) tuple ready for create_model_from_pdn()
    """
    if scenario == 'tile_merging':
        return _create_tile_merging_graph()
    elif scenario == 'path_expansion':
        return _create_path_expansion_graph()
    elif scenario == 'severe_halo_clip':
        return _create_severe_halo_clip_graph()
    elif scenario == 'basic':
        return _create_basic_graph()
    else:
        raise ValueError(f"Unknown scenario: {scenario}")


def _create_basic_graph() -> Tuple[nx.MultiDiGraph, List[str], Dict[str, float]]:
    """Create basic 3x3 M1 + 2x2 M2 grid for simple tests."""
    G = nx.MultiDiGraph()
    
    # Add ground node
    G.add_node('0')
    
    # M1 layer: 3x3 grid (coordinates 0, 2000, 4000)
    m1_coords = [0, 2000, 4000]
    m1_nodes = []
    for x in m1_coords:
        for y in m1_coords:
            node = f'{x}_{y}_M1'
            m1_nodes.append(node)
            G.add_node(node)
    
    # M2 layer: 2x2 grid at corners
    m2_coords = [0, 4000]
    m2_nodes = []
    for x in m2_coords:
        for y in m2_coords:
            node = f'{x}_{y}_M2'
            m2_nodes.append(node)
            G.add_node(node)
    
    # Add horizontal M1 resistors
    R_HORIZ = 0.01  # kOhm
    for y in m1_coords:
        for i in range(len(m1_coords) - 1):
            x1, x2 = m1_coords[i], m1_coords[i + 1]
            n1, n2 = f'{x1}_{y}_M1', f'{x2}_{y}_M1'
            G.add_edge(n1, n2, type='R', value=R_HORIZ)
            G.add_edge(n2, n1, type='R', value=R_HORIZ)
    
    # Add vertical M1 resistors
    for x in m1_coords:
        for i in range(len(m1_coords) - 1):
            y1, y2 = m1_coords[i], m1_coords[i + 1]
            n1, n2 = f'{x}_{y1}_M1', f'{x}_{y2}_M1'
            G.add_edge(n1, n2, type='R', value=R_HORIZ)
            G.add_edge(n2, n1, type='R', value=R_HORIZ)
    
    # Add M2 horizontal/vertical resistors
    R_M2 = 0.005  # kOhm (lower resistance for upper layer)
    for y in m2_coords:
        for i in range(len(m2_coords) - 1):
            x1, x2 = m2_coords[i], m2_coords[i + 1]
            n1, n2 = f'{x1}_{y}_M2', f'{x2}_{y}_M2'
            G.add_edge(n1, n2, type='R', value=R_M2)
            G.add_edge(n2, n1, type='R', value=R_M2)
    for x in m2_coords:
        for i in range(len(m2_coords) - 1):
            y1, y2 = m2_coords[i], m2_coords[i + 1]
            n1, n2 = f'{x}_{y1}_M2', f'{x}_{y2}_M2'
            G.add_edge(n1, n2, type='R', value=R_M2)
            G.add_edge(n2, n1, type='R', value=R_M2)
    
    # Add vias (M1 to M2) at corners
    R_VIA = 0.001  # kOhm
    for x in m2_coords:
        for y in m2_coords:
            m1_node = f'{x}_{y}_M1'
            m2_node = f'{x}_{y}_M2'
            G.add_edge(m1_node, m2_node, type='R', value=R_VIA)
            G.add_edge(m2_node, m1_node, type='R', value=R_VIA)
    
    # Add voltage source at top-right M2
    pad_node = 'VDD_vsrc'
    G.add_node(pad_node)
    G.add_edge(pad_node, '4000_4000_M2', type='R', value=0.0001)
    G.add_edge('4000_4000_M2', pad_node, type='R', value=0.0001)
    
    # Add current sources (loads) - distribute across M1
    load_currents = {
        '0_0_M1': 10.0,      # mA
        '2000_2000_M1': 15.0,
        '4000_0_M1': 10.0,
    }
    for node, current in load_currents.items():
        G.add_edge(node, '0', type='I', value=current)
    
    # Store metadata
    G.graph['net_connectivity'] = {'VDD': list(m1_nodes) + list(m2_nodes) + [pad_node]}
    G.graph['vsrc_nodes'] = [pad_node]
    G.graph['parameters'] = {'VDD': 1.0}  # Nominal voltage
    
    return G, [pad_node], load_currents


def _create_tile_merging_graph() -> Tuple[nx.MultiDiGraph, List[str], Dict[str, float]]:
    """Create graph where 2x2 tiling produces tiles with 0 loads.
    
    Structure:
    - 4x4 M1 grid (16 nodes) at coords 0, 2000, 4000, 6000
    - 2x2 M2 grid (4 nodes) at corners 0, 6000
    - Loads ONLY in bottom-left quadrant (x <= 2000, y <= 2000)
    - With 2x2 tiling, tiles (1,0), (0,1), (1,1) have 0 loads -> triggers merging
    """
    G = nx.MultiDiGraph()
    G.add_node('0')
    
    # M1 layer: 4x4 grid
    m1_coords = [0, 2000, 4000, 6000]
    m1_nodes = []
    for x in m1_coords:
        for y in m1_coords:
            node = f'{x}_{y}_M1'
            m1_nodes.append(node)
            G.add_node(node)
    
    # M2 layer: 2x2 grid at corners
    m2_coords = [0, 6000]
    m2_nodes = []
    for x in m2_coords:
        for y in m2_coords:
            node = f'{x}_{y}_M2'
            m2_nodes.append(node)
            G.add_node(node)
    
    # Add M1 horizontal resistors
    R_HORIZ = 0.01  # kOhm
    for y in m1_coords:
        for i in range(len(m1_coords) - 1):
            x1, x2 = m1_coords[i], m1_coords[i + 1]
            n1, n2 = f'{x1}_{y}_M1', f'{x2}_{y}_M1'
            G.add_edge(n1, n2, type='R', value=R_HORIZ)
            G.add_edge(n2, n1, type='R', value=R_HORIZ)
    
    # Add M1 vertical resistors
    for x in m1_coords:
        for i in range(len(m1_coords) - 1):
            y1, y2 = m1_coords[i], m1_coords[i + 1]
            n1, n2 = f'{x}_{y1}_M1', f'{x}_{y2}_M1'
            G.add_edge(n1, n2, type='R', value=R_HORIZ)
            G.add_edge(n2, n1, type='R', value=R_HORIZ)
    
    # Add M2 resistors
    R_M2 = 0.005  # kOhm
    for y in m2_coords:
        for i in range(len(m2_coords) - 1):
            x1, x2 = m2_coords[i], m2_coords[i + 1]
            n1, n2 = f'{x1}_{y}_M2', f'{x2}_{y}_M2'
            G.add_edge(n1, n2, type='R', value=R_M2)
            G.add_edge(n2, n1, type='R', value=R_M2)
    for x in m2_coords:
        for i in range(len(m2_coords) - 1):
            y1, y2 = m2_coords[i], m2_coords[i + 1]
            n1, n2 = f'{x}_{y1}_M2', f'{x}_{y2}_M2'
            G.add_edge(n1, n2, type='R', value=R_M2)
            G.add_edge(n2, n1, type='R', value=R_M2)
    
    # Add vias at ALL corners (M1 to M2)
    R_VIA = 0.001  # kOhm
    for x in m2_coords:
        for y in m2_coords:
            m1_node = f'{x}_{y}_M1'
            m2_node = f'{x}_{y}_M2'
            G.add_edge(m1_node, m2_node, type='R', value=R_VIA)
            G.add_edge(m2_node, m1_node, type='R', value=R_VIA)
    
    # Add voltage source at top-right M2
    pad_node = 'VDD_vsrc'
    G.add_node(pad_node)
    G.add_edge(pad_node, '6000_6000_M2', type='R', value=0.0001)
    G.add_edge('6000_6000_M2', pad_node, type='R', value=0.0001)
    
    # Add current sources ONLY in bottom-left quadrant
    # With 2x2 tiling (split at x=3000, y=3000), only tile (0,0) has loads
    load_currents = {
        '0_0_M1': 10.0,
        '0_2000_M1': 10.0,
        '2000_0_M1': 10.0,
        '2000_2000_M1': 10.0,
    }
    for node, current in load_currents.items():
        G.add_edge(node, '0', type='I', value=current)
    
    G.graph['net_connectivity'] = {'VDD': m1_nodes + m2_nodes + [pad_node]}
    G.graph['vsrc_nodes'] = [pad_node]
    G.graph['parameters'] = {'VDD': 1.0}  # Nominal voltage
    
    return G, [pad_node], load_currents


def _create_path_expansion_graph() -> Tuple[nx.MultiDiGraph, List[str], Dict[str, float]]:
    """Create graph with sparse vias causing locally disconnected core nodes.
    
    Structure:
    - 4x4 M1 grid
    - 2x2 M2 grid at corners
    - Via ONLY at (6000, 6000) - NOT at (0, 0)
    - Load at (0, 0) M1 which is far from only via
    - With small halo, core node (0,0) may be locally disconnected from port
    """
    G = nx.MultiDiGraph()
    G.add_node('0')
    
    # M1 layer: 4x4 grid
    m1_coords = [0, 2000, 4000, 6000]
    m1_nodes = []
    for x in m1_coords:
        for y in m1_coords:
            node = f'{x}_{y}_M1'
            m1_nodes.append(node)
            G.add_node(node)
    
    # M2 layer: 2x2 grid
    m2_coords = [0, 6000]
    m2_nodes = []
    for x in m2_coords:
        for y in m2_coords:
            node = f'{x}_{y}_M2'
            m2_nodes.append(node)
            G.add_node(node)
    
    # Add M1 resistors
    R_HORIZ = 0.01
    for y in m1_coords:
        for i in range(len(m1_coords) - 1):
            x1, x2 = m1_coords[i], m1_coords[i + 1]
            n1, n2 = f'{x1}_{y}_M1', f'{x2}_{y}_M1'
            G.add_edge(n1, n2, type='R', value=R_HORIZ)
            G.add_edge(n2, n1, type='R', value=R_HORIZ)
    for x in m1_coords:
        for i in range(len(m1_coords) - 1):
            y1, y2 = m1_coords[i], m1_coords[i + 1]
            n1, n2 = f'{x}_{y1}_M1', f'{x}_{y2}_M1'
            G.add_edge(n1, n2, type='R', value=R_HORIZ)
            G.add_edge(n2, n1, type='R', value=R_HORIZ)
    
    # Add M2 resistors
    R_M2 = 0.005
    for y in m2_coords:
        for i in range(len(m2_coords) - 1):
            x1, x2 = m2_coords[i], m2_coords[i + 1]
            n1, n2 = f'{x1}_{y}_M2', f'{x2}_{y}_M2'
            G.add_edge(n1, n2, type='R', value=R_M2)
            G.add_edge(n2, n1, type='R', value=R_M2)
    for x in m2_coords:
        for i in range(len(m2_coords) - 1):
            y1, y2 = m2_coords[i], m2_coords[i + 1]
            n1, n2 = f'{x}_{y1}_M2', f'{x}_{y2}_M2'
            G.add_edge(n1, n2, type='R', value=R_M2)
            G.add_edge(n2, n1, type='R', value=R_M2)
    
    # Add via ONLY at top-right corner - sparse connectivity
    R_VIA = 0.001
    G.add_edge('6000_6000_M1', '6000_6000_M2', type='R', value=R_VIA)
    G.add_edge('6000_6000_M2', '6000_6000_M1', type='R', value=R_VIA)
    
    # Also add via at (0, 6000) to ensure some connectivity
    G.add_edge('0_6000_M1', '0_6000_M2', type='R', value=R_VIA)
    G.add_edge('0_6000_M2', '0_6000_M1', type='R', value=R_VIA)
    
    # Add voltage source at top-right M2
    pad_node = 'VDD_vsrc'
    G.add_node(pad_node)
    G.add_edge(pad_node, '6000_6000_M2', type='R', value=0.0001)
    G.add_edge('6000_6000_M2', pad_node, type='R', value=0.0001)
    
    # Add loads distributed across M1 - including far corner
    load_currents = {
        '0_0_M1': 10.0,       # Far from vias - may trigger path expansion
        '2000_2000_M1': 10.0,
        '6000_6000_M1': 10.0,
    }
    for node, current in load_currents.items():
        G.add_edge(node, '0', type='I', value=current)
    
    G.graph['net_connectivity'] = {'VDD': m1_nodes + m2_nodes + [pad_node]}
    G.graph['vsrc_nodes'] = [pad_node]
    G.graph['parameters'] = {'VDD': 1.0}  # Nominal voltage
    
    return G, [pad_node], load_currents


def _create_severe_halo_clip_graph() -> Tuple[nx.MultiDiGraph, List[str], Dict[str, float]]:
    """Create 6x6 grid where 3x3 tiling severely clips corner halos.
    
    With 3x3 tiling and 50% halo, corner tiles have only 25% of expected
    halo area (clipped on two sides), triggering severe clip warning.
    """
    G = nx.MultiDiGraph()
    G.add_node('0')
    
    # M1 layer: 6x6 grid
    m1_coords = [0, 2000, 4000, 6000, 8000, 10000]
    m1_nodes = []
    for x in m1_coords:
        for y in m1_coords:
            node = f'{x}_{y}_M1'
            m1_nodes.append(node)
            G.add_node(node)
    
    # M2 layer: 3x3 grid
    m2_coords = [0, 5000, 10000]
    m2_nodes = []
    for x in m2_coords:
        for y in m2_coords:
            node = f'{x}_{y}_M2'
            m2_nodes.append(node)
            G.add_node(node)
    
    # Add M1 horizontal resistors
    R_HORIZ = 0.01
    for y in m1_coords:
        for i in range(len(m1_coords) - 1):
            x1, x2 = m1_coords[i], m1_coords[i + 1]
            n1, n2 = f'{x1}_{y}_M1', f'{x2}_{y}_M1'
            G.add_edge(n1, n2, type='R', value=R_HORIZ)
            G.add_edge(n2, n1, type='R', value=R_HORIZ)
    
    # Add M1 vertical resistors
    for x in m1_coords:
        for i in range(len(m1_coords) - 1):
            y1, y2 = m1_coords[i], m1_coords[i + 1]
            n1, n2 = f'{x}_{y1}_M1', f'{x}_{y2}_M1'
            G.add_edge(n1, n2, type='R', value=R_HORIZ)
            G.add_edge(n2, n1, type='R', value=R_HORIZ)
    
    # Add M2 resistors
    R_M2 = 0.005
    for y in m2_coords:
        for i in range(len(m2_coords) - 1):
            x1, x2 = m2_coords[i], m2_coords[i + 1]
            n1, n2 = f'{x1}_{y}_M2', f'{x2}_{y}_M2'
            G.add_edge(n1, n2, type='R', value=R_M2)
            G.add_edge(n2, n1, type='R', value=R_M2)
    for x in m2_coords:
        for i in range(len(m2_coords) - 1):
            y1, y2 = m2_coords[i], m2_coords[i + 1]
            n1, n2 = f'{x}_{y1}_M2', f'{x}_{y2}_M2'
            G.add_edge(n1, n2, type='R', value=R_M2)
            G.add_edge(n2, n1, type='R', value=R_M2)
    
    # Add vias at M2 corners closest to M1 grid points
    R_VIA = 0.001
    via_connections = [
        ('0_0_M1', '0_0_M2'),
        ('10000_0_M1', '10000_0_M2'),
        ('0_10000_M1', '0_10000_M2'),
        ('10000_10000_M1', '10000_10000_M2'),
        ('4000_4000_M1', '5000_5000_M2'),  # Center via (approximate)
    ]
    for m1, m2 in via_connections:
        if m1 in G.nodes() and m2 in G.nodes():
            G.add_edge(m1, m2, type='R', value=R_VIA)
            G.add_edge(m2, m1, type='R', value=R_VIA)
    
    # Add voltage source
    pad_node = 'VDD_vsrc'
    G.add_node(pad_node)
    G.add_edge(pad_node, '10000_10000_M2', type='R', value=0.0001)
    G.add_edge('10000_10000_M2', pad_node, type='R', value=0.0001)
    
    # Add loads distributed across grid
    load_currents = {
        '0_0_M1': 5.0,
        '4000_4000_M1': 10.0,
        '8000_8000_M1': 5.0,
        '2000_6000_M1': 5.0,
        '6000_2000_M1': 5.0,
    }
    for node, current in load_currents.items():
        G.add_edge(node, '0', type='I', value=current)
    
    G.graph['net_connectivity'] = {'VDD': m1_nodes + m2_nodes + [pad_node]}
    G.graph['vsrc_nodes'] = [pad_node]
    G.graph['parameters'] = {'VDD': 1.0}  # Nominal voltage
    
    return G, [pad_node], load_currents


def create_floating_island_graph() -> Tuple[nx.MultiDiGraph, List[str], Dict[str, float]]:
    """Create graph with an isolated floating island (no path to pad).
    
    Returns:
        (graph, pad_nodes, load_currents) - note that island loads are included
        but will be filtered out by the model's island detection.
    """
    G = nx.MultiDiGraph()
    G.add_node('0')
    
    # Main connected grid: 3x3 M1 + 2x2 M2
    m1_coords = [0, 2000, 4000]
    for x in m1_coords:
        for y in m1_coords:
            G.add_node(f'{x}_{y}_M1')
    
    m2_coords = [0, 4000]
    for x in m2_coords:
        for y in m2_coords:
            G.add_node(f'{x}_{y}_M2')
    
    # Add M1 resistors
    R = 0.01
    for y in m1_coords:
        for i in range(len(m1_coords) - 1):
            x1, x2 = m1_coords[i], m1_coords[i + 1]
            n1, n2 = f'{x1}_{y}_M1', f'{x2}_{y}_M1'
            G.add_edge(n1, n2, type='R', value=R)
            G.add_edge(n2, n1, type='R', value=R)
    for x in m1_coords:
        for i in range(len(m1_coords) - 1):
            y1, y2 = m1_coords[i], m1_coords[i + 1]
            n1, n2 = f'{x}_{y1}_M1', f'{x}_{y2}_M1'
            G.add_edge(n1, n2, type='R', value=R)
            G.add_edge(n2, n1, type='R', value=R)
    
    # Add M2 resistors
    for y in m2_coords:
        for i in range(len(m2_coords) - 1):
            x1, x2 = m2_coords[i], m2_coords[i + 1]
            n1, n2 = f'{x1}_{y}_M2', f'{x2}_{y}_M2'
            G.add_edge(n1, n2, type='R', value=0.005)
            G.add_edge(n2, n1, type='R', value=0.005)
    for x in m2_coords:
        for i in range(len(m2_coords) - 1):
            y1, y2 = m2_coords[i], m2_coords[i + 1]
            n1, n2 = f'{x}_{y1}_M2', f'{x}_{y2}_M2'
            G.add_edge(n1, n2, type='R', value=0.005)
            G.add_edge(n2, n1, type='R', value=0.005)
    
    # Add vias
    for x in m2_coords:
        for y in m2_coords:
            m1, m2 = f'{x}_{y}_M1', f'{x}_{y}_M2'
            G.add_edge(m1, m2, type='R', value=0.001)
            G.add_edge(m2, m1, type='R', value=0.001)
    
    # Add voltage source
    pad_node = 'VDD_vsrc'
    G.add_node(pad_node)
    G.add_edge(pad_node, '4000_4000_M2', type='R', value=0.0001)
    G.add_edge('4000_4000_M2', pad_node, type='R', value=0.0001)
    
    # === FLOATING ISLAND ===
    # Add isolated 2x2 M1 grid at x=10000 (not connected to main grid)
    island_coords = [10000, 12000]
    for x in island_coords:
        for y in island_coords:
            G.add_node(f'{x}_{y}_M1')
    
    # Connect island nodes to each other
    for y in island_coords:
        n1, n2 = f'10000_{y}_M1', f'12000_{y}_M1'
        G.add_edge(n1, n2, type='R', value=R)
        G.add_edge(n2, n1, type='R', value=R)
    for x in island_coords:
        n1, n2 = f'{x}_10000_M1', f'{x}_12000_M1'
        G.add_edge(n1, n2, type='R', value=R)
        G.add_edge(n2, n1, type='R', value=R)
    
    # Add loads on both main grid and island
    load_currents = {
        '0_0_M1': 10.0,
        '2000_2000_M1': 10.0,
        '10000_10000_M1': 5.0,  # Island load - will be filtered
        '12000_12000_M1': 5.0,  # Island load - will be filtered
    }
    for node, current in load_currents.items():
        G.add_edge(node, '0', type='I', value=current)
    
    G.graph['net_connectivity'] = {'VDD': [n for n in G.nodes() if n != '0']}
    G.graph['vsrc_nodes'] = [pad_node]
    G.graph['parameters'] = {'VDD': 1.0}  # Nominal voltage
    
    return G, [pad_node], load_currents

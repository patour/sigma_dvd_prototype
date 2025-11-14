#!/usr/bin/env python3
"""
Example: Structured Grid Partitioning

Demonstrates deterministic slab partitioning using the GridPartitioner.
Partitions are formed by slicing along X (vertical) or Y (horizontal) axis of 
layer-0 load nodes; separators are entire via rows/columns at internal boundaries 
(excluding loads & pads). No edges are removed; graph topology preserved.

The 'axis' parameter allows choosing partitioning orientation:
  - 'x': Vertical slabs (partition by X coordinate)
  - 'y': Horizontal slabs (partition by Y coordinate)  
  - 'auto': Automatically choose axis with better load balance
"""

from generate_power_grid import generate_power_grid
from irdrop import GridPartitioner


def main():
    print("=" * 70)
    print("Grid Partitioning Example")
    print("=" * 70)
    
    # Generate a power grid
    print("\n1. Generating power grid...")
    G, loads, pads = generate_power_grid(
        K=3,           # 3 layers
        N0=16,         # 16 initial stripes on layer 0
        I_N=100,       # ~100 load insertion attempts
        N_vsrc=8,      # 8 voltage source pads
        max_stripe_res=5.0,
        max_via_res=0.1,
        load_current=1.0,
        seed=42,
        plot=False
    )
    
    print(f"   Grid: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"   Loads: {len(loads)}, Pads: {len(pads)}")
    
    # Create partitioner
    print("\n2. Creating partitioner...")
    partitioner = GridPartitioner(G, loads, pads, seed=42)
    
    # Partition into 4 regions
    print("\n3. Partitioning grid into P=4 regions (auto axis selection)...")
    result = partitioner.partition(P=4, axis='auto')
    
    print(f"\n{result}")
    
    # Show partition details
    print("\n4. Partition Details:")
    print(f"   {'ID':<4} {'Loads':<8} {'Separators':<12} {'Interior':<10}")
    print("   " + "-" * 40)
    for partition in result.partitions:
        print(f"   {partition.partition_id:<4} "
              f"{partition.num_loads:<8} "
              f"{len(partition.separator_nodes):<12} "
              f"{len(partition.interior_nodes):<10}")
    
    # Verify properties
    print("\n5. Verification:")
    total_loads = sum(p.num_loads for p in result.partitions)
    print(f"   ✓ Total loads across partitions: {total_loads}/{len(loads)}")
    print(f"   ✓ Load balance ratio: {result.load_balance_ratio:.3f}")
    print(f"   ✓ Separator nodes: {len(result.separator_nodes)}")
    print(f"   ✓ Boundary edges: {len(result.boundary_edges)}")
    print(f"   ✓ Graph edges preserved: {G.number_of_edges()} (no removal)")
    
    # Verify separators have no loads
    sep_with_loads = result.separator_nodes & set(loads.keys())
    print(f"   ✓ Separators with loads: {len(sep_with_loads)} (expected: 0)")
    
    # Check connectivity
    print("\n   Connectivity Analysis:")
    conn_info = result.get_partition_connectivity_info(G, set(pads))
    all_connected = all(info['connected'] for info in conn_info.values())
    for pid, info in conn_info.items():
        status = "✓" if info['connected'] else "⚠"
        print(f"   {status} Partition {pid}: {info['connectivity_ratio']:.1%} connected " 
              f"({info.get('reachable_count', info['interior_count'])}/{info['interior_count']} nodes)")
    
    if not all_connected:
        print(f"   ⚠ Some partitions have disconnected interior regions (contains isolated loads)")
    else:
        print(f"   ✓ All partitions have fully connected interiors")
    
    # Test different partition counts
    print("\n6. Testing different partition counts and axis modes:")
    print(f"   {'P':<4} {'Axis':<6} {'Load Distribution':<30} {'Balance':<10} {'Edges':<10}")
    print("   " + "-" * 70)
    
    for P in [2, 3, 4, 6]:
        # Need fresh grid for each test
        G_test, loads_test, pads_test = generate_power_grid(
            K=3, N0=16, I_N=100, N_vsrc=8,
            max_stripe_res=5.0, max_via_res=0.1,
            load_current=1.0, seed=42, plot=False
        )
        
        partitioner_test = GridPartitioner(G_test, loads_test, pads_test, seed=42)
        result_test = partitioner_test.partition(P=P, axis='auto')
        
        load_counts = [p.num_loads for p in result_test.partitions]
        load_str = str(load_counts)
        
        print(f"   {P:<4} {'auto':<6} {load_str:<30} "
              f"{result_test.load_balance_ratio:<10.3f} "
              f"{len(result_test.boundary_edges):<10}")
    
    # Compare X vs Y axis for P=4
    print("\n   Comparing X vs Y axis for P=4:")
    print(f"   {'Axis':<6} {'Load Distribution':<30} {'Balance':<10} {'Separators':<25}")
    print("   " + "-" * 75)
    
    for axis_mode in ['x', 'y', 'auto']:
        G_test, loads_test, pads_test = generate_power_grid(
            K=3, N0=16, I_N=100, N_vsrc=8,
            max_stripe_res=5.0, max_via_res=0.1,
            load_current=1.0, seed=42, plot=False
        )
        partitioner_test = GridPartitioner(G_test, loads_test, pads_test, seed=42)
        result_test = partitioner_test.partition(P=4, axis=axis_mode)
        load_counts = [p.num_loads for p in result_test.partitions]
        load_str = str(load_counts)
        
        # Count separators per layer
        total_seps = len(result_test.separator_nodes)
        layer_counts = {}
        for layer in range(3):
            layer_counts[layer] = len([n for n in result_test.separator_nodes 
                                        if G_test.nodes[n].get('layer', 0) == layer])
        sep_str = f"L0:{layer_counts[0]} L1:{layer_counts[1]} L2:{layer_counts[2]}"
        
        print(f"   {axis_mode:<6} {load_str:<30} {result_test.load_balance_ratio:<10.3f} {sep_str:<25}")
    
    # Visualize partitions
    print("\n7. Generating visualization...")
    fig, ax = partitioner.visualize_partitions(result, figsize=(12, 10), show=False)
    
    # Save figure
    output_file = "partition_visualization.png"
    fig.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"   Saved visualization to: {output_file}")
    
    print("\n" + "=" * 70)
    print("Partitioning complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()

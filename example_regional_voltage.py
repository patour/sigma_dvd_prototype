#!/usr/bin/env python3
"""
Example: Regional Voltage Solver with Partitioned Power Grids

Demonstrates the RegionalVoltageSolver for computing DC voltages at load nodes
within a partitioned region using effective resistance and boundary conditions.
"""

import numpy as np
from generate_power_grid import generate_power_grid
from irdrop import (
    PowerGridModel, 
    EffectiveResistanceCalculator,
    GridPartitioner,
    RegionalVoltageSolver,
    IRDropSolver
)


def main():
    print("=" * 70)
    print("Regional Voltage Solver Example")
    print("=" * 70)
    
    # Generate a power grid
    print("\n1. Generating power grid (K=3 layers, N0=12 stripes)...")
    G, loads, pads = generate_power_grid(
        K=3,
        N0=12,
        I_N=80,
        N_vsrc=4,
        max_stripe_res=1.0,
        max_via_res=0.1,
        load_current=0.01,  # Reduced from 1.0 to get realistic voltages
        seed=42,
        plot=False
    )
    
    print(f"   Grid has {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"   Load nodes: {len(loads)}, Pad nodes: {len(pads)}")
    
    # Build power grid model
    print("\n2. Building power grid model...")
    model = PowerGridModel(G, pad_nodes=pads, vdd=1.0)
    
    # Create effective resistance calculator
    print("\n3. Creating effective resistance calculator...")
    calc = EffectiveResistanceCalculator(model)
    
    # Partition the grid
    print("\n4. Partitioning the grid...")
    partitioner = GridPartitioner(G, load_nodes=loads, pad_nodes=pads)
    partition_result = partitioner.partition(P=4, axis='x')
    
    print(f"   Created {partition_result.num_partitions} partitions")
    print(f"   Load balance ratio: {partition_result.load_balance_ratio:.2f}")
    print(f"   Total separator nodes: {len(partition_result.separator_nodes)}")
    
    # Create regional voltage solver
    print("\n5. Creating regional voltage solver...")
    regional_solver = RegionalVoltageSolver(calc, model)
    
    # Select a partition to analyze
    partition = partition_result.partitions[0]
    print(f"\n6. Analyzing Partition {partition.partition_id}...")
    print(f"   Interior nodes: {len(partition.interior_nodes)}")
    print(f"   Separator nodes: {len(partition.separator_nodes)}")
    print(f"   Load nodes: {len(partition.load_nodes)}")
    
    # Define the subset S (first 5 load nodes in this partition)
    partition_loads = sorted(list(partition.load_nodes), key=lambda n: (n.layer, n.idx))
    S = set(partition_loads[:5])
    print(f"\n   Selected subset S with {len(S)} load nodes: {partition_loads[:5]}")
    
    # Define region R (all nodes in partition)
    R = partition.all_nodes
    
    # Define boundary A (separator nodes for this partition)
    A = partition.separator_nodes
    
    # Define currents for near loads (in S)
    I_S = {node: loads[node] for node in S if node in loads}
    
    # Define currents for far loads (all other loads not in S)
    all_other_loads = set(loads.keys()) - S
    I_F = {node: loads[node] for node in all_other_loads}
    
    print(f"\n   Near loads (S): {len(I_S)} nodes")
    print(f"   Far loads (F): {len(I_F)} nodes")
    print(f"   Boundary nodes (A): {len(A)} nodes")
    
    # Compute voltages using regional solver
    print("\n7. Computing voltages using regional solver...")
    voltages_regional = regional_solver.compute_voltages(S, R, A, I_S, I_F)
    
    print(f"\n   Voltages computed for {len(voltages_regional)} nodes")
    print("\n   Sample voltages (first 5 nodes):")
    for i, node in enumerate(partition_loads[:5]):
        v = voltages_regional[node]
        ir_drop = 1.0 - v  # Assuming Vdd = 1.0V
        print(f"      Node {node}: V = {v:.6f} V, IR-drop = {ir_drop:.6f} V")
    
    # Compare with full IR-drop solver
    print("\n" + "=" * 70)
    print("Validation: Compare with Full IR-Drop Solver")
    print("=" * 70)
    
    print("\n8. Running full IR-drop solver for comparison...")
    full_solver = IRDropSolver(model)
    
    # Create stimulus with all loads as a Dict
    stimulus_dict = {node: loads[node] for node in loads}
    
    result_full = full_solver.solve(stimulus_dict)
    
    # Debug: check what's in result_full
    print(f"\n   Sample from full solver result:")
    sample_nodes = list(S)[:2]
    for node in sample_nodes:
        print(f"      {node}: voltage={result_full.voltages.get(node, 'N/A')}, ir_drop={result_full.ir_drop.get(node, 'N/A')}")
    
    # Extract voltages for nodes in S from full solution
    voltages_full = {}
    for node in S:
        if node in result_full.voltages:
            voltages_full[node] = result_full.voltages[node]
    
    print("\n9. Comparing results...")
    print("\n   Node-by-node comparison:")
    print(f"   {'Node':<15} {'Regional':<12} {'Full Solver':<12} {'Difference':<12}")
    print("   " + "-" * 55)
    
    errors = []
    for node in partition_loads[:min(10, len(partition_loads))]:  # Show first 10
        if node not in S:
            continue
        v_regional = voltages_regional.get(node, 0.0)
        v_full = voltages_full.get(node, 0.0)
        diff = abs(v_regional - v_full)
        errors.append(diff)
        print(f"   {str(node):<15} {v_regional:<12.6f} {v_full:<12.6f} {diff:<12.6e}")
    
    if errors:
        print(f"\n   Error statistics:")
        print(f"      Mean absolute error: {np.mean(errors):.6e} V")
        print(f"      Max absolute error:  {np.max(errors):.6e} V")
        print(f"      RMS error:           {np.sqrt(np.mean(np.array(errors)**2)):.6e} V")
    
    # Display statistics
    print("\n" + "=" * 70)
    print("Statistics")
    print("=" * 70)
    
    v_regional_vals = list(voltages_regional.values())
    v_full_vals = [voltages_full[n] for n in voltages_regional.keys()]
    
    print(f"\nRegional Solver:")
    print(f"   Min voltage: {np.min(v_regional_vals):.6f} V")
    print(f"   Max voltage: {np.max(v_regional_vals):.6f} V")
    print(f"   Mean voltage: {np.mean(v_regional_vals):.6f} V")
    print(f"   Max IR-drop: {1.0 - np.min(v_regional_vals):.6f} V")
    
    print(f"\nFull Solver (for same nodes):")
    print(f"   Min voltage: {np.min(v_full_vals):.6f} V")
    print(f"   Max voltage: {np.max(v_full_vals):.6f} V")
    print(f"   Mean voltage: {np.mean(v_full_vals):.6f} V")
    print(f"   Max IR-drop: {1.0 - np.min(v_full_vals):.6f} V")
    
    print("\n" + "=" * 70)
    print("✓ Regional voltage solver demonstration complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()

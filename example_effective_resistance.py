#!/usr/bin/env python3
"""
Example: Computing effective resistance in power grids

Demonstrates the EffectiveResistanceCalculator for computing R_eff between
node pairs in a multi-layer power grid.
"""

import numpy as np
from generate_power_grid import generate_power_grid
from irdrop import PowerGridModel, EffectiveResistanceCalculator


def main():
    print("=" * 70)
    print("Effective Resistance Computation Example")
    print("=" * 70)
    
    # Generate a small power grid
    print("\n1. Generating power grid (K=3 layers, N0=8 stripes)...")
    G, loads, pads = generate_power_grid(
        K=3,
        N0=8,
        I_N=50,
        N_vsrc=3,
        max_stripe_res=1.0,
        max_via_res=0.1,
        load_current=1.0,
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
    
    # Example 1: Compute R_eff from several nodes to ground (pads)
    print("\n" + "=" * 70)
    print("Example 1: Resistance to Ground (Pads)")
    print("=" * 70)
    
    load_nodes = list(loads.keys())[:10]
    ground_pairs = [(node, None) for node in load_nodes]
    
    print(f"\nComputing R_eff to ground for {len(ground_pairs)} nodes...")
    reff_ground = calc.compute_batch(ground_pairs)
    
    print("\nResults (first 5):")
    for i in range(min(5, len(reff_ground))):
        node = load_nodes[i]
        print(f"   R_eff({node} → ground) = {reff_ground[i]:.6f} Ω")
    
    print(f"\nStatistics:")
    print(f"   Min R_eff to ground: {np.min(reff_ground):.6f} Ω")
    print(f"   Max R_eff to ground: {np.max(reff_ground):.6f} Ω")
    print(f"   Mean R_eff to ground: {np.mean(reff_ground):.6f} Ω")
    
    # Example 2: Compute R_eff between pairs of nodes
    print("\n" + "=" * 70)
    print("Example 2: Node-to-Node Resistance")
    print("=" * 70)
    
    # Create pairs of adjacent nodes
    node_pairs = [(load_nodes[i], load_nodes[i+1]) for i in range(0, min(10, len(load_nodes)-1))]
    
    print(f"\nComputing R_eff between {len(node_pairs)} node pairs...")
    reff_pairs = calc.compute_batch(node_pairs)
    
    print("\nResults (first 5):")
    for i in range(min(5, len(reff_pairs))):
        u, v = node_pairs[i]
        print(f"   R_eff({u} ↔ {v}) = {reff_pairs[i]:.6f} Ω")
    
    print(f"\nStatistics:")
    print(f"   Min R_eff between pairs: {np.min(reff_pairs):.6f} Ω")
    print(f"   Max R_eff between pairs: {np.max(reff_pairs):.6f} Ω")
    print(f"   Mean R_eff between pairs: {np.mean(reff_pairs):.6f} Ω")
    
    # Example 3: Mixed batch computation
    print("\n" + "=" * 70)
    print("Example 3: Mixed Batch (Ground + Node-to-Node)")
    print("=" * 70)
    
    mixed_pairs = [
        (load_nodes[0], None),           # to ground
        (load_nodes[1], load_nodes[2]),  # node-to-node
        (load_nodes[3], None),           # to ground
        (load_nodes[4], load_nodes[5]),  # node-to-node
        (load_nodes[6], load_nodes[7]),  # node-to-node
    ]
    
    print(f"\nComputing R_eff for mixed batch of {len(mixed_pairs)} pairs...")
    reff_mixed = calc.compute_batch(mixed_pairs)
    
    print("\nResults:")
    for i, (u, v) in enumerate(mixed_pairs):
        if v is None:
            print(f"   R_eff({u} → ground) = {reff_mixed[i]:.6f} Ω")
        else:
            print(f"   R_eff({u} ↔ {v}) = {reff_mixed[i]:.6f} Ω")
    
    # Example 4: Single computation (convenience method)
    print("\n" + "=" * 70)
    print("Example 4: Single Computation")
    print("=" * 70)
    
    u, v = load_nodes[0], load_nodes[1]
    reff_single = calc.compute_single(u, v)
    print(f"\nR_eff({u} ↔ {v}) = {reff_single:.6f} Ω")
    
    # Example 5: Large batch efficiency test
    print("\n" + "=" * 70)
    print("Example 5: Large Batch Efficiency")
    print("=" * 70)
    
    # Create a large batch
    large_batch = []
    for i in range(len(load_nodes) - 1):
        large_batch.append((load_nodes[i], load_nodes[i+1]))
    for node in load_nodes:
        large_batch.append((node, None))
    
    print(f"\nComputing R_eff for large batch of {len(large_batch)} pairs...")
    import time
    start = time.time()
    reff_large = calc.compute_batch(large_batch)
    elapsed = time.time() - start
    
    print(f"   Completed in {elapsed:.4f} seconds")
    print(f"   Average time per pair: {elapsed/len(large_batch)*1000:.2f} ms")
    print(f"   Results shape: {reff_large.shape}")
    
    # Validation: Check symmetry
    print("\n" + "=" * 70)
    print("Validation: Symmetry Check")
    print("=" * 70)
    
    u, v = load_nodes[0], load_nodes[1]
    pairs_sym = [(u, v), (v, u)]
    reff_sym = calc.compute_batch(pairs_sym)
    
    print(f"\nR_eff({u} ↔ {v}) = {reff_sym[0]:.10f} Ω")
    print(f"R_eff({v} ↔ {u}) = {reff_sym[1]:.10f} Ω")
    print(f"Difference: {abs(reff_sym[0] - reff_sym[1]):.2e} Ω")
    print(f"Symmetric: {np.allclose(reff_sym[0], reff_sym[1])}")
    
    print("\n" + "=" * 70)
    print("Example completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()

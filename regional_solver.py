"""Regional IR-Drop Solver Example

Demonstrates the RegionalIRDropSolver which computes IR-drops at a subset of 
load nodes within a partitioned region using effective resistance and boundary 
conditions. Validates results against the full IR-drop solver.

The regional solver decomposes IR-drop into near-field (loads within region) 
and far-field (loads outside region) contributions, enabling efficient 
localized analysis.
"""

import numpy as np
import argparse
from generate_power_grid import generate_power_grid
from irdrop import (
    PowerGridModel,
    StimulusGenerator,
    IRDropSolver,
    RegionalIRDropSolver,
    GridPartitioner,
    EffectiveResistanceCalculator,
    plot_ir_drop_map,
    plot_current_map,
)


def main():
    parser = argparse.ArgumentParser(
        description="Regional IR-Drop Solver Demonstration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Grid generation parameters (defaults from proto.ipynb)
    parser.add_argument("--K", type=int, default=3, help="Number of metal layers")
    parser.add_argument("--N0", type=int, default=16, help="Number of stripes in layer 0")
    parser.add_argument("--I_N", type=int, default=1000, help="Number of load insertion attempts")
    parser.add_argument("--N_vsrc", type=int, default=8, help="Number of voltage sources (pads)")
    parser.add_argument("--max_stripe_res", type=float, default=5.0, help="Max stripe resistance (Ω) in layer 0")
    parser.add_argument("--max_via_res", type=float, default=0.1, help="Max via resistance (Ω) between L0-L1")
    parser.add_argument("--load_current", type=float, default=0.01, help="Default load current (A)")
    parser.add_argument("--grid_seed", type=int, default=7, help="Seed for grid generation")
    
    # Partitioning parameters
    parser.add_argument("--num_partitions", type=int, default=3, help="Number of partitions (P)")
    parser.add_argument("--partition_axis", type=str, default="y", choices=["x", "y", "auto"],
                        help="Partitioning axis")
    parser.add_argument("--partition_seed", type=int, default=42, help="Seed for partitioner")
    parser.add_argument("--partition_id", type=int, default=0, help="Which partition to analyze (0-indexed)")
    
    # Stimulus parameters
    parser.add_argument("--total_power", type=float, default=1.0, help="Total power (W)")
    parser.add_argument("--load_percent", type=float, default=0.1, help="Percentage of loads to activate (0-1)")
    parser.add_argument("--distribution", type=str, default="gaussian", choices=["uniform", "gaussian"],
                        help="Current distribution type")
    parser.add_argument("--gaussian_loc", type=float, default=1.0, help="Gaussian mean")
    parser.add_argument("--gaussian_scale", type=float, default=0.2, help="Gaussian std dev")
    parser.add_argument("--stimulus_seed", type=int, default=100, help="Seed for stimulus generation")
    
    # Regional solver parameters
    parser.add_argument("--subset_size", type=int, default=5, help="Number of nodes in subset S")
    parser.add_argument("--subset_seed", type=int, default=4, help="Seed for subset selection")
    parser.add_argument("--near_max_distance", type=float, default=10.0, help="Max distance for near-field loads from subset S")
    
    # Visualization parameters
    parser.add_argument("--vdd", type=float, default=1.0, help="Supply voltage (V)")
    parser.add_argument("--plot_grid", action="store_true", help="Plot grid during generation")
    parser.add_argument("--plot_partition", action="store_true", help="Plot partition visualization")
    parser.add_argument("--plot_irdrop", action="store_true", help="Plot IR-drop map")
    parser.add_argument("--plot_current", action="store_true", help="Plot current map")
    parser.add_argument("--min_current", type=float, default=5e-3, help="Minimum current for current map (A)")
    parser.add_argument("--show_plots", action="store_true", help="Show plots interactively")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Regional IR-Drop Solver Example")
    print("=" * 80)
    
    # Step 1: Generate power grid
    print("\n[1] Generating power grid...")
    G, loads, pads = generate_power_grid(
        K=args.K,
        N0=args.N0,
        I_N=args.I_N,
        N_vsrc=args.N_vsrc,
        max_stripe_res=args.max_stripe_res,
        max_via_res=args.max_via_res,
        load_current=args.load_current,
        seed=args.grid_seed,
        plot=args.plot_grid,
    )
    print(f"Grid: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"Loads: {len(loads)}, Pads: {len(pads)}")
    
    # Step 2: Build model and calculator
    print("\n[2] Building power grid model...")
    model = PowerGridModel(G, pad_nodes=pads, vdd=args.vdd)
    calc = EffectiveResistanceCalculator(model)
    
    # Step 3: Partition the grid
    print(f"\n[3] Partitioning grid into {args.num_partitions} regions (axis={args.partition_axis})...")
    partitioner = GridPartitioner(G, loads, pads, seed=args.partition_seed)
    part_result = partitioner.partition(P=args.num_partitions, axis=args.partition_axis)
    
    print(f"Created {part_result.num_partitions} partitions")
    print(f"Load balance ratio: {part_result.load_balance_ratio:.2f}")
    
    # Select partition for analysis
    if args.partition_id >= part_result.num_partitions:
        print(f"WARNING: partition_id {args.partition_id} >= {part_result.num_partitions}, using partition 0")
        args.partition_id = 0
    
    partition = part_result.partitions[args.partition_id]
    print(f"\nAnalyzing Partition {partition.partition_id}:")
    print(f"  Interior nodes: {len(partition.interior_nodes)}")
    print(f"  Separator nodes: {len(partition.separator_nodes)}")
    print(f"  Load nodes: {len(partition.load_nodes)}")
    
    if args.plot_partition:
        print("\nPlotting partition visualization...")
        fig, ax = partitioner.visualize_partitions(part_result, figsize=(12, 10), show=args.show_plots)
        if not args.show_plots:
            import matplotlib.pyplot as plt
            fig.savefig("regional_solver_partitions.png", dpi=150, bbox_inches="tight")
            print("  Saved: regional_solver_partitions.png")
            plt.close(fig)
    
    # Step 4: Generate stimulus
    print(f"\n[4] Generating stimulus ({args.total_power}W, {args.load_percent*100:.1f}% loads, {args.distribution})...")
    stim_gen = StimulusGenerator(load_nodes=list(loads.keys()), vdd=args.vdd, seed=args.grid_seed, graph=G)
    meta = stim_gen.generate(
        total_power=args.total_power,
        percent=args.load_percent,
        distribution=args.distribution,
        gaussian_loc=args.gaussian_loc,
        gaussian_scale=args.gaussian_scale,
        seed=args.stimulus_seed,
    )
    print(f"Active loads: {len(meta.currents)} nodes")
    print(f"Total current: {meta.total_current:.6f} A")
    
    # Step 5: Select subset S for regional analysis
    print(f"\n[5] Selecting subset S ({args.subset_size} nodes from partition {args.partition_id})...")
    partition_loads = sorted(list(partition.load_nodes), key=lambda n: (n.layer, n.idx))
    
    if len(partition_loads) < args.subset_size:
        print(f"WARNING: Partition has only {len(partition_loads)} load nodes, using all")
        S = set(partition_loads)
    else:
        rng = np.random.RandomState(args.subset_seed)
        S = set(rng.choice(partition_loads, size=args.subset_size, replace=False))
    
    print(f"Subset S: {len(S)} nodes")
    
    # Step 6: Prepare regional solver inputs
    print("\n[6] Preparing regional solver inputs...")
    R = partition.all_nodes
    A = partition.separator_nodes
    active_loads = meta.currents
    
    # Compute positions for subset S nodes
    print(f"Filtering near loads within distance {args.near_max_distance} from subset S...")
    S_positions = np.array([G.nodes[node]['xy'] for node in S])
    
    # Filter loads in R by distance to subset S
    loads_in_R = set(partition.load_nodes)
    I_R = {}
    for node in loads_in_R:
        if node in active_loads:
            node_pos = np.array(G.nodes[node]['xy'])
            # Compute minimum distance to any node in S
            distances = np.linalg.norm(S_positions - node_pos, axis=1)
            min_distance = np.min(distances)
            if min_distance <= args.near_max_distance:
                I_R[node] = active_loads[node]
    
    all_far_loads = set(active_loads.keys()) - set(I_R.keys())
    I_F = {node: active_loads[node] for node in all_far_loads}
    
    print(f"Region R: {len(R)} nodes")
    print(f"Boundary A: {len(A)} nodes")
    print(f"Near loads (in R, within {args.near_max_distance}): {len(I_R)} nodes")
    print(f"Far loads (outside near region): {len(I_F)} nodes")
    
    # Step 7: Compute IR-drops using regional solver
    print("\n[7] Computing IR-drops with regional solver...")
    regional_solver = RegionalIRDropSolver(calc)
    ir_drops_regional, drop_near, drop_far = regional_solver.compute_ir_drops(S, R, A, I_R, I_F)
    
    print(f"\nRegional IR-drop results for {len(ir_drops_regional)} nodes:")
    print(f"{'Node':<25} {'Voltage':<12} {'IR-drop':<12} {'Near':<12} {'Far':<12}")
    print("-" * 73)
    for node in sorted(S, key=lambda n: (n.layer, n.idx)):
        drop = ir_drops_regional[node]
        drop_n = drop_near[node]
        drop_f = drop_far[node]
        voltage = args.vdd - drop
        print(f"{str(node):<25} {voltage:<12.6f} {drop:<12.6f} {drop_n:<12.6f} {drop_f:<12.6f}")
    
    # Step 8: Validation against full solver
    print("\n[8] Validating against full IR-drop solver...")
    full_solver = IRDropSolver(model)
    result_full = full_solver.solve(active_loads)
    
    ir_drops_full = {node: result_full.ir_drop[node] for node in S}
    
    print("\nComparison: Regional vs Full Solver")
    print(f"{'Node':<25} {'Regional':<12} {'Full':<12} {'Error':<12} {'Rel Error %':<12}")
    print("-" * 85)
    
    errors = []
    rel_errors = []
    for node in sorted(S, key=lambda n: (n.layer, n.idx)):
        drop_regional = ir_drops_regional[node]
        drop_full = ir_drops_full[node]
        error = abs(drop_regional - drop_full)
        rel_error = (error / drop_full * 100) if drop_full > 1e-12 else 0.0
        errors.append(error)
        rel_errors.append(rel_error)
        print(f"{str(node):<25} {drop_regional:<12.6f} {drop_full:<12.6f} {error:<12.2e} {rel_error:<12.2f}")
    
    print(f"\nError Statistics:")
    print(f"  Mean absolute error:  {np.mean(errors):.2e} V")
    print(f"  Max absolute error:   {np.max(errors):.2e} V")
    print(f"  RMS error:            {np.sqrt(np.mean(np.array(errors)**2)):.2e} V")
    print(f"  Mean relative error:  {np.mean(rel_errors):.2f}%")
    print(f"  Max relative error:   {np.max(rel_errors):.2f}%")
    
    # Step 9: Visualizations
    if args.plot_irdrop:
        print("\n[9] Plotting IR-drop map...")
        import matplotlib.pyplot as plt
        fig, ax = plot_ir_drop_map(G, result_full.voltages, vdd=args.vdd, layer=0, show=False)
        
        # Highlight subset S nodes
        for node in S:
            if node.layer == 0:
                x, y = G.nodes[node]['xy']
                ax.scatter([x], [y], c='red', s=100, marker='o', 
                          edgecolors='black', linewidths=2, zorder=10, label='Subset S')
        
        # Remove duplicate labels
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
        
        if args.show_plots:
            plt.show()
        else:
            fig.savefig("regional_solver_irdrop.png", dpi=150, bbox_inches="tight")
            print("  Saved: regional_solver_irdrop.png")
            plt.close(fig)
    
    if args.plot_current:
        print("\n[10] Plotting current map...")
        import matplotlib.pyplot as plt
        fig, ax = plot_current_map(
            G,
            result_full.voltages,
            layer=0,
            min_current=args.min_current,
            loads_current=active_loads,
            show=False,
        )
        
        if args.show_plots:
            plt.show()
        else:
            fig.savefig("regional_solver_current.png", dpi=150, bbox_inches="tight")
            print("  Saved: regional_solver_current.png")
            plt.close(fig)
    
    print("\n" + "=" * 80)
    print("Regional IR-Drop Solver Example Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()

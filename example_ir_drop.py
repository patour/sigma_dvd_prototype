"""Example usage of IR-drop analysis workflow."""

from generate_power_grid import generate_power_grid
from irdrop import PowerGridModel, StimulusGenerator, IRDropSolver


def main():
    # Generate a modest grid
    G, loads, pads = generate_power_grid(
        K=3,
        N0=12,
        I_N=200,  # number of tap insertion attempts (affects load nodes)
        N_vsrc=4,
        max_stripe_res=1.0,
        max_via_res=0.1,
        load_current=1.0,
        seed=11,
        plot=False,
    )
    print(f"Grid: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges; Loads={len(loads)} Pads={len(pads)}")

    # Build model (pads at 1.0V)
    model = PowerGridModel(G, pad_nodes=pads, vdd=1.0)

    # Stimulus generator using discovered load nodes
    stim_gen = StimulusGenerator(load_nodes=list(loads.keys()), vdd=1.0, seed=7)

    # Create batch of stimuli with varying total power
    powers = [0.5, 1.0, 2.0]  # Watts
    metas = stim_gen.generate_batch(powers, percent=0.3, distribution="gaussian")
    stimuli = [m.currents for m in metas]

    solver = IRDropSolver(model)
    results = solver.solve_batch(stimuli, metas)

    for m, r in zip(metas, results):
        summary = solver.summarize(r)
        print(
            f"P={m.total_power:.2f}W currents on {len(m.selected_nodes)} nodes -> minV={summary['min_voltage']:.4f}V maxDrop={summary['max_drop']:.4f}V avgDrop={summary['avg_drop']:.4f}V"
        )


if __name__ == "__main__":
    main()

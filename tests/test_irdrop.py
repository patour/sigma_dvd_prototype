import math
import unittest

from generate_power_grid import generate_power_grid
from irdrop import PowerGridModel, StimulusGenerator, IRDropSolver, plot_voltage_map, plot_ir_drop_map


def build_small():
    G, loads, pads = generate_power_grid(
        K=3, N0=8, I_N=80, N_vsrc=3,
        max_stripe_res=1.0, max_via_res=0.1,
        load_current=1.0, seed=3, plot=False
    )
    return G, loads, pads


class TestIRDrop(unittest.TestCase):
    def test_no_load_currents_all_pad_voltage(self):
        G, loads, pads = build_small()
        model = PowerGridModel(G, pad_nodes=pads, vdd=1.0)
        solver = IRDropSolver(model)
        result = solver.solve({})
        self.assertTrue(all(0.0 <= v <= 1.0 for v in result.voltages.values()))
        for p in pads:
            self.assertTrue(math.isclose(result.voltages[p], 1.0))

    def test_uniform_power_distribution_current_sum(self):
        G, loads, pads = build_small()
        model = PowerGridModel(G, pad_nodes=pads, vdd=1.0)
        stim_gen = StimulusGenerator(load_nodes=list(loads.keys()), vdd=1.0, seed=10)
        P = 2.0
        meta = stim_gen.generate(total_power=P, percent=0.5, distribution='uniform')
        self.assertTrue(math.isclose(sum(meta.currents.values()), P / 1.0, rel_tol=1e-9))

    def test_batch_min_voltage_monotonic(self):
        G, loads, pads = build_small()
        model = PowerGridModel(G, pad_nodes=pads, vdd=1.0)
        stim_gen = StimulusGenerator(load_nodes=list(loads.keys()), vdd=1.0, seed=2)
        powers = [0.5, 1.0, 2.0]
        metas = stim_gen.generate_batch(powers, percent=0.4, distribution='gaussian')
        solver = IRDropSolver(model)
        results = solver.solve_batch([m.currents for m in metas], metas)
        min_voltages = [min(r.voltages.values()) for r in results]
        self.assertTrue(min_voltages[0] >= min_voltages[1] >= min_voltages[2])

    def test_plot_functions(self):
        G, loads, pads = build_small()
        model = PowerGridModel(G, pad_nodes=pads, vdd=1.0)
        stim_gen = StimulusGenerator(load_nodes=list(loads.keys()), vdd=1.0, seed=5)
        meta = stim_gen.generate(total_power=1.0, percent=0.4, distribution='uniform')
        solver = IRDropSolver(model)
        result = solver.solve(meta.currents)
        fig1, _ = plot_voltage_map(G, result.voltages, show=False)
        fig2, _ = plot_ir_drop_map(G, result.voltages, vdd=1.0, show=False)
        # Just ensure figures were created
        self.assertIsNotNone(fig1)
        self.assertIsNotNone(fig2)


if __name__ == '__main__':
    unittest.main()

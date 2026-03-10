#!/usr/bin/env python3
"""
Integration tests for PDN Solver.

These tests use real netlists and generate actual output files.
They are slower than unit tests and are run separately.
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless testing

import unittest
import sys
import tempfile
import shutil
from pathlib import Path

import pytest

from parser.netlist import NetlistParser

pytestmark = pytest.mark.integration
from solver.pdn_solver import PDNSolver, PDNSolveResult, SolveResults


class TestReportGeneration(unittest.TestCase):
    """Test report generation with real netlist"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_generate_reports(self):
        """Test report generation"""
        netlist_dir = Path(__file__).parent.parent.parent / 'netlist' / 'netlist_small'
        if not netlist_dir.exists():
            self.skipTest("netlist_small not found")

        parser = NetlistParser(str(netlist_dir))
        graph = parser.parse()

        solver = PDNSolver(graph, verbose=False)
        results = solver.solve()

        # Generate reports
        solver.generate_reports(output_dir=self.temp_dir, top_k=10)

        # Check that report files were created
        net_name = list(results.net_results.keys())[0]
        report_file = Path(self.temp_dir) / f'topk_irdrop_{net_name}.txt'
        self.assertTrue(report_file.exists())

        # Check report content
        with open(report_file, 'r') as f:
            content = f.read()
            self.assertIn('Top-10 Worst IR-Drop Report', content)
            self.assertIn(net_name, content)
            self.assertIn('Backend:', content)
            self.assertIn('Solve Time:', content)

    def test_generate_heatmaps(self):
        """Test heatmap generation"""
        netlist_dir = Path(__file__).parent.parent.parent / 'netlist' / 'netlist_small'
        if not netlist_dir.exists():
            self.skipTest("netlist_small not found")

        parser = NetlistParser(str(netlist_dir))
        graph = parser.parse()

        solver = PDNSolver(graph, verbose=False)
        results = solver.solve()

        # Generate reports with heatmaps
        solver.generate_reports(output_dir=self.temp_dir, top_k=10)

        # Check that heatmap files were created
        output_path = Path(self.temp_dir)
        net_name = list(results.net_results.keys())[0]

        # Should have voltage/irdrop heatmaps per layer
        heatmap_files = list(output_path.glob(f'*_heatmap_{net_name}_layer_*.png'))
        self.assertGreater(len(heatmap_files), 0, "Should create at least one heatmap")

    def test_stripe_mode_reports(self):
        """Test stripe mode report generation"""
        netlist_dir = Path(__file__).parent.parent.parent / 'netlist' / 'netlist_small'
        if not netlist_dir.exists():
            self.skipTest("netlist_small not found")

        parser = NetlistParser(str(netlist_dir))
        graph = parser.parse()

        solver = PDNSolver(graph, verbose=False)
        results = solver.solve()

        # Generate reports with stripe mode
        solver.generate_reports(
            output_dir=self.temp_dir,
            top_k=10,
            stripe_mode=True,
            max_stripes=50,
        )

        # Check that report files were created
        net_name = list(results.net_results.keys())[0]
        report_file = Path(self.temp_dir) / f'topk_irdrop_{net_name}.txt'
        self.assertTrue(report_file.exists())


class TestNetlistSmallFullAnalysis(unittest.TestCase):
    """Full analysis test using netlist_small"""

    @classmethod
    def setUpClass(cls):
        netlist_dir = Path(__file__).parent.parent.parent / 'netlist' / 'netlist_small'
        if not netlist_dir.exists():
            raise unittest.SkipTest("netlist_small not found")

        parser = NetlistParser(str(netlist_dir))
        cls.graph = parser.parse()

    def test_full_solve_with_profiling(self):
        """Test full solve with timing and memory profiling"""
        solver = PDNSolver(
            self.graph,
            verbose=True,
            profile_memory=True,
        )
        results = solver.solve()

        # Verify results
        self.assertGreater(len(results.net_results), 0)
        self.assertGreater(results.total_solve_time, 0)
        self.assertGreater(results.total_memory_peak_mb, 0)

        # Check per-net results
        for net_name, result in results.net_results.items():
            self.assertGreater(result.num_nodes, 0)
            self.assertGreater(result.timings.total, 0)
            self.assertGreater(result.memory.peak_mb, 0)

    def test_cholmod_vs_splu_consistency(self):
        """Test that cholmod and splu produce consistent results"""
        # Solve with splu
        solver_splu = PDNSolver(self.graph, verbose=False, use_cholmod=False)
        results_splu = solver_splu.solve()

        # Re-parse graph (solver modifies it)
        netlist_dir = Path(__file__).parent.parent.parent / 'netlist' / 'netlist_small'
        parser = NetlistParser(str(netlist_dir))
        graph2 = parser.parse()

        # Try cholmod if available
        try:
            solver_cholmod = PDNSolver(graph2, verbose=False, use_cholmod=True)
            results_cholmod = solver_cholmod.solve()

            # Compare results
            for net_name in results_splu.net_results:
                result_splu = results_splu.net_results[net_name]
                result_cholmod = results_cholmod.net_results[net_name]

                # Voltages should match within tolerance
                self.assertAlmostEqual(
                    result_splu.max_ir_drop,
                    result_cholmod.max_ir_drop,
                    places=6,
                    msg=f"Max IR-drop mismatch for {net_name}"
                )

        except ImportError:
            self.skipTest("Cholmod not available")


if __name__ == '__main__':
    unittest.main()

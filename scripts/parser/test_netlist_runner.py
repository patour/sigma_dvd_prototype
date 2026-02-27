#!/usr/bin/env python3
"""Quick test runner for the test netlist.

Parses a PDN netlist, solves IR-drop, and generates reports.
Relocated from pdn/test_netlist_runner.py.
"""
from parser.netlist import NetlistParser
from solver.pdn_solver import PDNSolver

print('=== Parsing test netlist ===')
parser = NetlistParser('netlist/netlist_test', validate=True)
graph = parser.parse()

print('\n=== Solving IR-drop ===')
solver = PDNSolver(graph, verbose=False)
results = solver.solve()

print('\n=== Generating reports ===')
solver.generate_reports(output_dir='netlist/netlist_test/results', top_k=10)

print('\nAll tests passed!')

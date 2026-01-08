#!/usr/bin/env python3
"""Quick test runner for the test netlist"""

import sys
sys.path.insert(0, 'pdn')

from pdn_parser import NetlistParser
from pdn_solver import PDNSolver

print('=== Parsing test netlist ===')
parser = NetlistParser('pdn/netlist_test', validate=True)
graph = parser.parse()

print('\n=== Solving IR-drop ===')
solver = PDNSolver(graph, verbose=False)
results = solver.solve()

print('\n=== Generating reports ===')
solver.generate_reports(output_dir='pdn/netlist_test/results', top_k=10)

print('\n✅ All tests passed!')

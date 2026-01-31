#!/usr/bin/env python3
"""Run all unit tests (excluding slow integration tests).

For integration tests, run: python run_all_integration_tests.py
"""
import unittest
import sys

# Configure matplotlib before importing test modules
import matplotlib
matplotlib.use('Agg')

loader = unittest.TestLoader()

# Discover all test modules except integration tests
print("Discovering unit test modules in ./tests directory...")
print("(Skipping *_integration.py - run run_all_integration_tests.py for those)")
suite = unittest.TestSuite()

# Load specific test modules, excluding integration tests
test_modules = [
    'tests.test_batch_solving',
    'tests.test_coupled_hierarchical_solver',
    'tests.test_dynamic_solver',
    'tests.test_hierarchical_solver',
    'tests.test_irdrop',
    'tests.test_partitioner',
    'tests.test_pdn_parser',
    'tests.test_pdn_plotter',
    'tests.test_pdn_solver',
    'tests.test_regional_solver',
    'tests.test_rx_algorithms',
    'tests.test_rx_graph',
    'tests.test_transient_multi_rhs',
    'tests.test_transient_solver',
    'tests.test_unified_core',
    'tests.test_vectorized_sources',
]

for module in test_modules:
    try:
        suite.addTests(loader.loadTestsFromName(module))
    except Exception as e:
        print(f"Warning: Could not load {module}: {e}")

print(f"\nRunning {suite.countTestCases()} tests...")
print("="*70)

runner = unittest.TextTestRunner(verbosity=1)
result = runner.run(suite)

print("\n" + "="*70)
print("TEST SUMMARY")
print("="*70)
print(f"Tests run: {result.testsRun}")
print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
print(f"Failures: {len(result.failures)}")
print(f"Errors: {len(result.errors)}")

if result.failures:
    print("\nFAILED TESTS:")
    for test, _ in result.failures:
        print(f"  ❌ {test}")

if result.errors:
    print("\nERRORS:")
    for test, _ in result.errors:
        print(f"  ❌ {test}")

print("="*70)
sys.exit(0 if result.wasSuccessful() else 1)

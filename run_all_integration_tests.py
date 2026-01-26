#!/usr/bin/env python3
"""Run integration tests (slow tests that use real netlists).

These tests are separated from unit tests because they take longer to run.
For quick unit tests, run: python run_all_tests.py
"""
import unittest
import sys

# Configure matplotlib before importing test modules
import matplotlib
matplotlib.use('Agg')

loader = unittest.TestLoader()

print("Running integration tests...")
print("(These tests use real netlists and may take a while)")
print("="*70)

suite = unittest.TestSuite()

# Load integration test modules
integration_modules = [
    'tests.test_hierarchical_integration',
    'tests.test_dynamic_integration',
]

for module in integration_modules:
    try:
        suite.addTests(loader.loadTestsFromName(module))
    except Exception as e:
        print(f"Warning: Could not load {module}: {e}")

print(f"\nRunning {suite.countTestCases()} integration tests...")
print("="*70)

runner = unittest.TextTestRunner(verbosity=2)
result = runner.run(suite)

print("\n" + "="*70)
print("INTEGRATION TEST SUMMARY")
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

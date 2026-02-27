#!/usr/bin/env python3
"""Run all unit tests (excluding slow integration tests).

For integration tests, run: python run_all_integration_tests.py
"""
import sys

# Configure matplotlib before importing test modules
import matplotlib
matplotlib.use('Agg')

import pytest

print("Running unit tests via pytest...")
print("(Skipping *_integration.py - run run_all_integration_tests.py for those)")
print("=" * 70)

exit_code = pytest.main([
    "tests",
    "-q",
])

sys.exit(exit_code)

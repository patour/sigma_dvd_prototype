#!/usr/bin/env python3
"""Run integration tests (slow tests that use real netlists).

These tests are separated from unit tests because they take longer to run.
For quick unit tests, run: python run_all_tests.py
"""
import sys

# Configure matplotlib before importing test modules
import matplotlib
matplotlib.use('Agg')

import pytest

print("Running integration tests via pytest...")
print("(These tests use real netlists and may take a while)")
print("=" * 70)

# Note: pyproject.toml [tool.pytest.ini_options] addopts includes --ignore flags
# for these integration files and --tb=short. When we pass explicit file paths
# below, pytest still appends addopts, so:
#   - The --ignore flags are harmless (they don't match the explicit paths).
#   - The --tb=short from addopts applies automatically (no need to repeat it).
# We use -o addopts= to clear the inherited addopts entirely, avoiding the
# confusing presence of --ignore flags that contradict the explicit file list.
exit_code = pytest.main([
    "tests/solver/test_hierarchical_integration.py",
    "tests/analysis/test_dynamic_integration.py",
    "tests/parser/test_pdn_integration.py",
    "-o", "addopts=",
    "--tb=short",
    "-v",
])

sys.exit(exit_code)

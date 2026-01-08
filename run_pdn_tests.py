#!/usr/bin/env python3
"""Test runner for PDN module tests"""

import sys
import unittest

# Run all PDN tests
if __name__ == '__main__':
    # Discover and run tests
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test modules
    suite.addTests(loader.loadTestsFromName('tests.test_pdn_parser'))
    suite.addTests(loader.loadTestsFromName('tests.test_pdn_solver'))
    suite.addTests(loader.loadTestsFromName('tests.test_pdn_plotter'))
    
    # Run with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Exit with error code if tests failed
    sys.exit(0 if result.wasSuccessful() else 1)

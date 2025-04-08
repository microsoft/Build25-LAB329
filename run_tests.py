#!/usr/bin/env python
"""
Test runner for Distillation project.
This script runs all unit tests in the project.

Usage:
    python run_tests.py             # Run all tests
    python run_tests.py data        # Run only tests with 'data' in the name
    python run_tests.py -v          # Run tests with verbose output
"""

import unittest
import sys
import os

if __name__ == "__main__":
    # Add project root to path for imports
    sys.path.append(os.path.abspath(os.path.dirname(__file__)))
    
    # Discover all tests in the tests directory
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover('tests', pattern='test_*.py')
    
    # If arguments are provided, filter tests
    if len(sys.argv) > 1 and sys.argv[1] != "-v":
        pattern = sys.argv[1]
        filtered_suite = unittest.TestSuite()
        for test in test_suite:
            for test_case in test:
                if pattern.lower() in test_case.id().lower():
                    filtered_suite.addTest(test_case)
        test_suite = filtered_suite
    
    # Determine verbosity
    verbosity = 2 if "-v" in sys.argv else 1
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(test_suite)
    
    # Exit with appropriate status code
    sys.exit(not result.wasSuccessful())
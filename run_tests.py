#!/usr/bin/env python3
"""
Test runner script for DP5.
Run this script to execute all tests.
"""

import os
import sys
import pytest

if __name__ == "__main__":
    # Add current directory to path
    sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
    
    # Run all tests
    print("Running DP5 tests...")
    
    # Parse command line arguments
    args = ["-v"]  # Verbose output by default
    
    # Add any additional arguments passed to this script
    args.extend(sys.argv[1:])
    
    # Add the tests directory
    args.append("tests")
    
    # Run pytest with the arguments
    exit_code = pytest.main(args)
    
    # Exit with the same code as pytest
    sys.exit(exit_code) 
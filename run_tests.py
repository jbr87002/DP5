#!/usr/bin/env python3
"""
Test runner script for DP5.
Run this script to execute all tests.
"""

import os
import sys
import pytest

if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
    
    print("Running DP5 tests...")
    
    args = ["-v"]  # Verbose output by default
    
    args.extend(sys.argv[1:])
    
    args.append("tests")
    
    exit_code = pytest.main(args)
    
    sys.exit(exit_code) 
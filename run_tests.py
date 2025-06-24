#!/usr/bin/env python3
"""
Test runner script for nano-genie

This script provides an easy way to run tests for the entire project.
"""

import subprocess
import sys
import os
import argparse
from pathlib import Path


def run_tests(test_type="all", verbose=False, coverage=False):
    """
    Run tests based on the specified type
    
    Args:
        test_type: Type of tests to run ("all", "unit", "integration", "video_tokenizer", "lam", "dynamics", "pipeline")
        verbose: Whether to run with verbose output
        coverage: Whether to run with coverage reporting
    """
    
    # Base pytest command
    cmd = [sys.executable, "-m", "pytest"]
    
    # Add verbose flag if requested
    if verbose:
        cmd.append("-v")
    
    # Add coverage if requested
    if coverage:
        cmd.extend(["--cov=src", "--cov-report=html", "--cov-report=term"])
    
    # Add specific test files based on type
    if test_type == "all":
        cmd.append("tests/")
    elif test_type == "unit":
        cmd.extend(["-m", "unit"])
    elif test_type == "integration":
        cmd.extend(["-m", "integration"])
    elif test_type == "video_tokenizer":
        cmd.append("tests/test_video_tokenizer.py")
    elif test_type == "lam":
        cmd.append("tests/test_lam.py")
    elif test_type == "dynamics":
        cmd.append("tests/test_dynamics.py")
    elif test_type == "pipeline":
        cmd.append("tests/test_full_pipeline.py")
    elif test_type == "cpu":
        cmd.extend(["-m", "cpu"])
    elif test_type == "gpu":
        cmd.extend(["-m", "gpu"])
    else:
        print(f"Unknown test type: {test_type}")
        return False
    
    # Print the command being run
    print(f"Running: {' '.join(cmd)}")
    print("-" * 60)
    
    # Run the tests
    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode == 0
    except KeyboardInterrupt:
        print("\nTests interrupted by user")
        return False
    except Exception as e:
        print(f"Error running tests: {e}")
        return False


def check_dependencies():
    """Check if required dependencies are available"""
    required_packages = ["pytest", "torch", "numpy", "h5py"]
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Missing required packages: {', '.join(missing_packages)}")
        print("Please install them with: pip install " + " ".join(missing_packages))
        return False
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Run tests for nano-genie")
    parser.add_argument(
        "--type", 
        choices=["all", "unit", "integration", "video_tokenizer", "lam", "dynamics", "pipeline", "cpu", "gpu"],
        default="all",
        help="Type of tests to run"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Run tests with verbose output"
    )
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Run tests with coverage reporting"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run only quick tests (exclude slow tests)"
    )
    
    args = parser.parse_args()
    
    # Check if we're in the right directory
    if not os.path.exists("src"):
        print("Error: Please run this script from the nano-genie root directory")
        return 1
    
    # Check dependencies
    if not check_dependencies():
        return 1
    
    # Build the test command
    cmd_args = [args.type]
    if args.verbose:
        cmd_args.append("--verbose")
    if args.coverage:
        cmd_args.append("--coverage")
    if args.quick:
        cmd_args.append("--quick")
    
    # Run the tests
    success = run_tests(*cmd_args)
    
    if success:
        print("\n" + "=" * 60)
        print("✅ All tests passed!")
        print("=" * 60)
        return 0
    else:
        print("\n" + "=" * 60)
        print("❌ Some tests failed!")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main()) 
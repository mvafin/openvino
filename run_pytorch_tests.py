#!/usr/bin/env python3
"""
Script to run PyTorch layer tests for OpenVINO.

This script provides a convenient way to run PyTorch layer tests
with various configuration options.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Run PyTorch layer tests for OpenVINO",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run precommit tests on CPU with FP32 precision
  %(prog)s

  # Run all tests on CPU and GPU with FP32 and FP16 precision
  %(prog)s --device CPU GPU --precision FP32 FP16 --marker ""

  # Run a specific test file
  %(prog)s --test-file test_add.py

  # Run with parallel execution
  %(prog)s --parallel
        """
    )
    
    parser.add_argument(
        '--device',
        nargs='+',
        default=['CPU'],
        choices=['CPU', 'GPU'],
        help='Device(s) to run tests on (default: CPU)'
    )
    
    parser.add_argument(
        '--precision',
        nargs='+',
        default=['FP32'],
        choices=['FP32', 'FP16'],
        help='Precision(s) to use for inference (default: FP32)'
    )
    
    parser.add_argument(
        '--marker',
        default='precommit',
        help='Pytest marker to filter tests (default: precommit). Use empty string for all tests.'
    )
    
    parser.add_argument(
        '--test-file',
        help='Specific test file to run (e.g., test_add.py)'
    )
    
    parser.add_argument(
        '--parallel',
        action='store_true',
        help='Run tests in parallel using pytest-xdist'
    )
    
    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Verbose output'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print the command that would be executed without running it'
    )
    
    args = parser.parse_args()
    
    # Set environment variables
    os.environ['TEST_DEVICE'] = ';'.join(args.device)
    os.environ['TEST_PRECISION'] = ';'.join(args.precision)
    
    # Get the repository root
    repo_root = Path(__file__).parent.resolve()
    layer_tests_dir = repo_root / 'tests' / 'layer_tests'
    
    if not layer_tests_dir.exists():
        print(f"Error: Layer tests directory not found: {layer_tests_dir}", file=sys.stderr)
        return 1
    
    # Build pytest command
    cmd = [
        sys.executable, '-m', 'pytest',
        'pytorch_tests' if not args.test_file else f'pytorch_tests/{args.test_file}',
    ]
    
    # Add marker filter
    if args.marker:
        cmd.extend(['-m', args.marker])
    
    # Add verbosity
    if args.verbose:
        cmd.append('-v')
    
    # Add parallel execution
    if args.parallel:
        cmd.extend(['-n', 'logical'])
    
    # Add output options
    cmd.extend([
        '--tb=short',
        '--junitxml=TEST-pytorch.xml',
        '--html=TEST-pytorch.html',
        '--self-contained-html',
    ])
    
    # Print configuration
    print("=" * 50)
    print("PyTorch Layer Tests Configuration")
    print("=" * 50)
    print(f"TEST_DEVICE: {os.environ['TEST_DEVICE']}")
    print(f"TEST_PRECISION: {os.environ['TEST_PRECISION']}")
    print(f"Marker: {args.marker if args.marker else 'all tests'}")
    print(f"Test file: {args.test_file if args.test_file else 'all tests'}")
    print(f"Parallel: {args.parallel}")
    print(f"Working directory: {layer_tests_dir}")
    print("=" * 50)
    print()
    
    # Print command
    print("Command:")
    print(' '.join(cmd))
    print()
    
    if args.dry_run:
        print("Dry run - not executing")
        return 0
    
    # Run the tests
    try:
        result = subprocess.run(
            cmd,
            cwd=layer_tests_dir,
            check=False
        )
        return result.returncode
    except KeyboardInterrupt:
        print("\nTest execution interrupted by user")
        return 130
    except Exception as e:
        print(f"Error running tests: {e}", file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())

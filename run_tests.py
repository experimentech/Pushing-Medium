#!/usr/bin/env python3
"""
Quick test runner for the Pushing-Medium gravitational model test suite.

This script provides convenient access to the full 59-test validation suite
that verifies GR equivalence, galaxy dynamics, and extended physics capabilities.

Usage:
    python run_tests.py              # Run all tests with standard output
    python run_tests.py -v           # Verbose output
    python run_tests.py -q           # Quiet output  
    python run_tests.py --report     # Generate comprehensive test report
    python run_tests.py --benchmarks # Run only GR benchmark tests
    python run_tests.py --galaxy     # Run only galaxy dynamics tests
"""

import subprocess
import sys
import argparse
from pathlib import Path

def run_pytest(args_list):
    """Run pytest with given arguments"""
    cmd = ['pytest'] + args_list
    print(f"Running: {' '.join(cmd)}")
    print("=" * 60)
    
    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    return result.returncode

def main():
    parser = argparse.ArgumentParser(
        description="Run Pushing-Medium gravitational model test suite (59 tests)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Verbose test output')
    parser.add_argument('-q', '--quiet', action='store_true', 
                       help='Quiet test output')
    parser.add_argument('--report', action='store_true',
                       help='Generate comprehensive test report')
    parser.add_argument('--benchmarks', action='store_true',
                       help='Run only GR benchmark tests')
    parser.add_argument('--galaxy', action='store_true',
                       help='Run only galaxy dynamics tests')
    parser.add_argument('--lensing', action='store_true',
                       help='Run only light deflection and lensing tests')
    parser.add_argument('--core', action='store_true',
                       help='Run only core physics tests')
    
    args = parser.parse_args()
    
    if args.report:
        print("🧪 Generating comprehensive test report...")
        return subprocess.run([sys.executable, 'test_results_table.py']).returncode
    
    # Build pytest arguments
    pytest_args = ['tests']
    
    if args.verbose:
        pytest_args.append('-v')
    elif args.quiet:
        pytest_args.append('-q')
    
    # Category-specific test selection
    if args.benchmarks:
        pytest_args = ['tests/test_passed_benchmarks.py', 'tests/test_gr_classical.py']
        if args.verbose:
            pytest_args.append('-v')
        elif args.quiet:
            pytest_args.append('-q')
    elif args.galaxy:
        pytest_args = ['tests/test_galaxy_rotation.py', 'tests/test_fitting.py', 
                      'tests/test_halo_fitting.py', 'tests/test_scaling_relations.py']
        if args.verbose:
            pytest_args.append('-v')
        elif args.quiet:
            pytest_args.append('-q')
    elif args.lensing:
        pytest_args = ['tests/test_deflection_convergence.py', 'tests/test_fermat_helper.py',
                      'tests/test_weak_bending_numeric.py', 'tests/test_strong_field_trend.py']
        if args.verbose:
            pytest_args.append('-v')
        elif args.quiet:
            pytest_args.append('-q')
    elif args.core:
        pytest_args = ['tests/test_pm_core.py', 'tests/test_pm_vs_gr.py', 
                      'tests/test_index_from_density.py']
        if args.verbose:
            pytest_args.append('-v')
        elif args.quiet:
            pytest_args.append('-q')
    
    # Run the tests
    return run_pytest(pytest_args)

if __name__ == "__main__":
    exit_code = main()
    print(f"\n{'='*60}")
    if exit_code == 0:
        print("✅ All tests passed! Pushing-Medium model validation successful.")
    else:
        print("❌ Some tests failed. Check output above for details.")
    
    print(f"\nFor detailed documentation, see TESTING.md")
    print(f"For comprehensive test report, run: python run_tests.py --report")
    
    sys.exit(exit_code)
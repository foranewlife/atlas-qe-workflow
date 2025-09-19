#!/usr/bin/env python3
"""
Test runner for ATLAS-QE workflow system.

This script runs all unit tests and provides a summary of results.
"""

import unittest
import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def run_specific_tests():
    """Run specific test modules."""
    test_modules = [
        'tests.test_parameter_set',
        'tests.test_pseudopotential',
        'tests.test_structure_generation'
    ]

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    for module in test_modules:
        try:
            tests = loader.loadTestsFromName(module)
            suite.addTests(tests)
            print(f"✓ Loaded tests from {module}")
        except Exception as e:
            print(f"✗ Failed to load {module}: {e}")

    return suite

def run_all_tests():
    """Run all tests in the tests directory."""
    loader = unittest.TestLoader()
    suite = loader.discover('tests', pattern='test_*.py')
    return suite

def main():
    """Main test runner."""
    print("🧪 Running ATLAS-QE Workflow Tests")
    print("=" * 50)

    # Try to run specific tests first
    try:
        suite = run_specific_tests()
        test_count = suite.countTestCases()

        if test_count == 0:
            print("No tests found, trying directory discovery...")
            suite = run_all_tests()
            test_count = suite.countTestCases()

        if test_count == 0:
            print("❌ No tests found!")
            return False

        print(f"\n🚀 Running {test_count} tests...\n")

        # Run tests with detailed output
        runner = unittest.TextTestRunner(
            verbosity=2,
            stream=sys.stdout,
            descriptions=True,
            failfast=False
        )

        result = runner.run(suite)

        # Print summary
        print("\n" + "=" * 50)
        print("📊 Test Summary")
        print("=" * 50)

        total_tests = result.testsRun
        failures = len(result.failures)
        errors = len(result.errors)
        skipped = len(result.skipped) if hasattr(result, 'skipped') else 0
        passed = total_tests - failures - errors - skipped

        print(f"Total tests:  {total_tests}")
        print(f"Passed:      {passed}")
        print(f"Failed:      {failures}")
        print(f"Errors:      {errors}")
        print(f"Skipped:     {skipped}")

        success_rate = (passed / total_tests * 100) if total_tests > 0 else 0
        print(f"Success rate: {success_rate:.1f}%")

        if failures > 0:
            print("\n❌ Failed tests:")
            for test, traceback in result.failures:
                print(f"  - {test}")

        if errors > 0:
            print("\n💥 Error tests:")
            for test, traceback in result.errors:
                print(f"  - {test}")

        if failures == 0 and errors == 0:
            print("\n🎉 All tests passed!")
            return True
        else:
            print(f"\n⚠️  {failures + errors} test(s) failed")
            return False

    except Exception as e:
        print(f"❌ Test execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
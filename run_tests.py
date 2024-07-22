import unittest
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

if __name__ == '__main__':
    test_loader = unittest.TestLoader()
    try:
        test_suite = test_loader.discover('tests', pattern='test_*.py')
    except Exception as e:
        print(f"An error occurred while discovering tests: {e}")
        sys.exit(1)

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    if not result.wasSuccessful():
        sys.exit(1)
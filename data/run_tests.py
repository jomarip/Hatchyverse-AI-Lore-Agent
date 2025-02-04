#!/usr/bin/env python3
"""Script to run the retrieval system tests."""

import pytest
import sys
import logging
from pathlib import Path

def setup_logging():
    """Configure logging for test runs."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('test_run.log')
        ]
    )

def main():
    """Run the test suite."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Ensure the tests directory is in the Python path
    test_dir = Path(__file__).parent / "tests"
    sys.path.insert(0, str(test_dir.parent))
    
    logger.info("Starting test run...")
    
    # Run pytest with verbosity and test coverage
    args = [
        "-v",  # verbose output
        "--cov=src",  # measure code coverage for src directory
        "--cov-report=term-missing",  # show lines missing coverage
        "--cov-report=html",  # generate HTML coverage report
        str(test_dir / "test_retrieval_system.py")
    ]
    
    exit_code = pytest.main(args)
    
    if exit_code == 0:
        logger.info("All tests passed successfully!")
    else:
        logger.error(f"Tests failed with exit code: {exit_code}")
    
    return exit_code

if __name__ == "__main__":
    sys.exit(main()) 
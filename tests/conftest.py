import os
import sys

# Ensure src/ and tests/ are on sys.path at import time
repo_root = os.path.dirname(os.path.dirname(__file__))
src_path = os.path.join(repo_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)
tests_dir = os.path.dirname(__file__)
if tests_dir not in sys.path:
    sys.path.insert(0, tests_dir)

def pytest_sessionstart(session):
    # Nothing extra; kept for potential future session init.
    pass


import pytest
from fixtures.calibration import load_or_calibrate


@pytest.fixture(scope="session")
def pm_calibration():
    """Session-scoped calibration constants for PM tests."""
    return load_or_calibrate()

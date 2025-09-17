import math
from pathlib import Path

from galaxy_dynamics.data import load_sparc_mock, KPC_TO_M, KM_TO_M


def test_load_sparc_mock_default_first_galaxy():
    csv_path = Path('tests/data/mock_sparc.csv')
    rc = load_sparc_mock(csv_path)
    # Should pick first galaxy (GAL_A) by default
    assert rc.name == 'GAL_A'
    assert len(rc.radii_m) == len(rc.v_obs_ms) == len(rc.v_err_ms)
    assert len(rc.radii_m) > 0
    # Radii should be sorted and increasing
    assert all(rc.radii_m[i] < rc.radii_m[i+1] for i in range(len(rc.radii_m)-1))
    # First radius 0.5 kpc -> meters
    expected_first = 0.5 * KPC_TO_M
    assert math.isclose(rc.radii_m[0], expected_first, rel_tol=1e-12)
    # Velocity conversion: 30 km/s -> m/s
    assert math.isclose(rc.v_obs_ms[0], 30 * KM_TO_M, rel_tol=1e-12)
    # Error conversion: 2 km/s -> m/s
    assert math.isclose(rc.v_err_ms[0], 2 * KM_TO_M, rel_tol=1e-12)
    # Surface densities retained raw
    assert rc.sigma_star[0] == 150
    assert rc.sigma_gas[0] == 10


def test_load_sparc_mock_select_second():
    csv_path = Path('tests/data/mock_sparc.csv')
    rc = load_sparc_mock(csv_path, galaxy_name='GAL_B')
    assert rc.name == 'GAL_B'
    # Should have 5 radii for GAL_B in file
    assert len(rc.radii_m) == 5
    # Check last velocity corresponds to 95 km/s
    assert math.isclose(rc.v_obs_ms[-1], 95 * KM_TO_M, rel_tol=1e-12)
    # Ensure meta recorded source and file
    assert rc.meta.get('source') == 'mock_sparc'
    assert rc.meta.get('file')


def test_load_sparc_mock_bad_file():
    from pytest import raises
    with raises(FileNotFoundError):
        load_sparc_mock('tests/data/does_not_exist.csv')

import math
from pathlib import Path

from galaxy_dynamics.data import load_sparc_real, KM_TO_M, KPC_TO_M


def test_load_sparc_real_single_default():
    path = Path('tests/data/sparc_sample.csv')
    rc = load_sparc_real(path)
    assert rc.name == 'GAL_X'
    assert rc.distance_mpc == 12.3
    assert rc.inclination_deg == 60
    assert len(rc.radii_m) == 5
    # check radius conversion for 0.5 kpc first row
    assert math.isclose(rc.radii_m[0], 0.5 * KPC_TO_M, rel_tol=1e-12)
    # check observed velocity conversion 40 km/s -> m/s
    assert math.isclose(rc.v_obs_ms[0], 40 * KM_TO_M, rel_tol=1e-12)
    # components present
    assert 'stars' in rc.components and 'gas' in rc.components
    assert len(rc.components['stars']) == len(rc.radii_m)


def test_load_sparc_real_select_galaxy():
    path = Path('tests/data/sparc_sample.csv')
    rc = load_sparc_real(path, galaxy_name='GAL_Y')
    assert rc.name == 'GAL_Y'
    # bulge and bar components should exist with finite entries
    assert 'bulge' in rc.components and 'bar' in rc.components
    assert any(math.isfinite(v) for v in rc.components['bulge'])
    assert rc.distance_mpc == 9.5
    assert rc.inclination_deg == 72


def test_load_sparc_real_dict_return():
    path = Path('tests/data/sparc_sample.csv')
    data = load_sparc_real(path, return_dict=True)
    assert set(data.keys()) == {'GAL_X','GAL_Y'}
    assert data['GAL_X'].components['stars'][0] > 0
    assert data['GAL_Y'].components['gas'][0] > 0
    # Ensure metadata carried
    assert data['GAL_X'].distance_mpc == 12.3

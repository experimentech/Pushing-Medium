import math
from pushing_medium.core import plummer_density, plummer_mass_enclosed


def test_plummer_density_normalization_ratio():
    # Check scaling by comparing densities at 0 and large r
    M, a = 1.0, 0.5
    rho = plummer_density(M, a)
    r0 = rho(0,0,0)
    rfar = rho(10*a,0,0)
    assert r0 > rfar > 0


def test_plummer_mass_enclosed_limits():
    M, a = 1.0, 1.0
    assert abs(plummer_mass_enclosed(M, a, 0.0) - 0.0) < 1e-12
    assert abs(plummer_mass_enclosed(M, a, 1e6) - M) / M < 1e-6

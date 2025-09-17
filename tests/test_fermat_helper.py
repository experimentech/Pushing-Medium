import math

from pushing_medium.core import (
    fermat_deflection_static_index,
    index_deflection_numeric,
    pm_deflection_angle_point_mass,
)


def approx(a, b, rel=1e-2, abs_tol=1e-12):
    return abs(a - b) <= max(abs_tol, rel * max(abs(a), abs(b)))


def test_fermat_matches_numeric_and_gr(pm_calibration):
    # Sun-like lens, grazing-ish impact parameter
    M = 1.989e30
    b = 6.96e8
    mu = pm_calibration.mu_coeff
    a1 = index_deflection_numeric(M, b, mu=mu, z_max=2e10, steps=2500)
    a2 = fermat_deflection_static_index(M, b, mu=mu, z_max=2e10, steps=2500)
    assert approx(a1, a2, rel=1e-12)
    a_gr = pm_deflection_angle_point_mass(M, b)
    assert approx(a1, a_gr, rel=5e-2)

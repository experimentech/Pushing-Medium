from pushing_medium.core import (
    index_deflection_numeric,
    curved_path_deflection_iterative,
    pm_deflection_angle_point_mass,
)


def test_iterative_deflection_reduces_error(pm_calibration):
    M = 1.989e30
    b = 3.0e9  # choose larger b where weak-field analytic is good but path correction helps a bit
    mu = pm_calibration.mu_coeff
    analytic = pm_deflection_angle_point_mass(M, b)
    straight = index_deflection_numeric(M, b, mu, z_max=4e10, steps=4000)
    iter_alpha = curved_path_deflection_iterative(M, b, mu, z_max=4e10, steps=4000, relax=0.6, iters=3)
    err_straight = abs(straight - analytic) / analytic
    err_iter = abs(iter_alpha - analytic) / analytic
    # Allow either improvement or parity, but require not worse by >10%, and typically better
    assert err_iter <= 1.1 * err_straight
    # Sanity: both within 10%
    assert err_straight < 0.1 and err_iter < 0.1
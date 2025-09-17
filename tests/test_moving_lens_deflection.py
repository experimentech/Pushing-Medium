from pushing_medium.core import (
    moving_lens_deflection_first_order,
    fermat_deflection_static_index,
)

def approx(a,b,rel=1e-2,abs_tol=1e-14):
    return abs(a-b) <= max(abs_tol, rel*max(abs(a),abs(b)))


def test_moving_lens_linear_v_scaling(pm_calibration):
    M = 1.989e30
    b = 6.96e8
    mu = pm_calibration.mu_coeff
    kF = pm_calibration.k_Fizeau
    v = 3.0e4  # 30 km/s
    base = fermat_deflection_static_index(M, b, mu=mu, z_max=2e10, steps=2000)
    moved = moving_lens_deflection_first_order(M, b, mu, v, kF, z_max=2e10, steps=2000)
    # Expected ratio ~ 1 + kF * v/c
    expected = base * (1 + kF * v / 299792458.0)
    assert approx(moved, expected, rel=2e-3)
    # Ensure increase for positive v
    assert moved > base

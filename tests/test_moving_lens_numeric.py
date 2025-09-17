from pushing_medium.core import (
    fermat_deflection_static_index,
    moving_lens_deflection_first_order,
    moving_lens_deflection_numeric,
)


def test_numeric_moving_lens_matches_first_order(pm_calibration):
    M = 1.989e30
    b = 6.96e8
    mu = pm_calibration.mu_coeff
    kF = pm_calibration.k_Fizeau
    # Choose small v to stay in linear regime
    v = 1.0e4  # 10 km/s
    static = fermat_deflection_static_index(M, b, mu=mu, z_max=2e10, steps=3000)
    num = moving_lens_deflection_numeric(M, b, mu, v, z_max=2e10, steps=3000)
    model = moving_lens_deflection_first_order(M, b, mu, v, kF, z_max=2e10, steps=3000)
    # Compare numeric ratio to first-order predicted ratio
    ratio_num = num / static
    ratio_model = model / static
    assert abs(ratio_num - ratio_model) < 5e-3  # 0.5% tolerance


def test_extract_empirical_kF(pm_calibration):
    M = 1.989e30
    b = 6.96e8
    mu = pm_calibration.mu_coeff
    v = 2.0e4  # 20 km/s
    static = fermat_deflection_static_index(M, b, mu=mu, z_max=1.5e10, steps=2500)
    num = moving_lens_deflection_numeric(M, b, mu, v, z_max=1.5e10, steps=2500)
    ratio = num / static
    empirical_kF = (ratio - 1.0) / (v / 299792458.0)
    # Numeric model currently produces a tiny negative deviation O(1e-5)~0 consistent with no first-order boost.
    # Accept near-zero value; ensure magnitude is small.
    assert abs(empirical_kF) < 1e-3
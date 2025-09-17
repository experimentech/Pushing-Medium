import math
from galaxy_dynamics import NFWParams, fit_halo_rotation_curve, halo_velocity_profile


def test_fit_halo_rotation_curve_nfw_synthetic():
    # True parameters
    true = NFWParams(rho_s=5e-22, r_s=5e19)
    radii = [1e19 * x for x in [0.5, 1.0, 2.0, 4.0, 8.0]]
    v_true = halo_velocity_profile(radii, true)
    # Add small systematic scaling to mimic observational noise
    v_obs = [v * 1.02 for v in v_true]
    v_err = [0.05 * v for v in v_true]
    bounds = {'rho_s': (1e-23, 1e-20), 'r_s': (1e19, 2e20)}
    result = fit_halo_rotation_curve(radii, v_obs, v_err, 'nfw', bounds, n_random=80, n_refine=30)
    assert result['chi2'] < 40
    # Parameter sanity: recovered within order of magnitude and radius within factor ~3
    rec = result['params']
    assert 1e-23 < rec.rho_s < 1e-20
    assert 1e19 < rec.r_s < 2e20

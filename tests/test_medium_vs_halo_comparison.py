from galaxy_dynamics import (
    DiskParams, MediumParams, rotation_curve, fit_rotation_curve,
    NFWParams, fit_halo_rotation_curve, halo_velocity_profile
)


def test_medium_vs_halo_comparison_structure():
    # Synthetic medium-based curve
    disk = DiskParams(M_d=5e40, R_d=5e19)
    medium = MediumParams(v_inf=1.6e5, r_s=5e19, r_c=1e19, m=2.0)
    radii = [1e19 * x for x in [0.5, 1.0, 2.0, 4.0, 8.0]]
    v_obs = rotation_curve(radii, disk, medium)
    v_err = [0.05 * v for v in v_obs]
    # Fit medium model back
    disk_bounds = {'M_d': (1e40, 1e41), 'R_d': (2e19, 1e20)}
    medium_bounds = {'v_inf': (1e5, 2.5e5), 'r_s': (2e19, 1e20), 'r_c': (5e18, 5e19), 'm': (1.0, 3.0)}
    rc_like = type('RC', (), {
        'radii_m': radii,
        'v_obs_ms': v_obs,
        'v_err_ms': v_err,
        'sigma_star': [float('nan')]*len(radii),
        'sigma_gas': [float('nan')]*len(radii)
    })()
    med_fit = fit_rotation_curve(rc_like, disk_bounds, medium_bounds, n_random=40, n_refine=15)
    # Fit NFW-only as baseline
    halo_bounds = {'rho_s': (1e-24, 1e-20), 'r_s': (1e19, 2e20)}
    halo_fit = fit_halo_rotation_curve(radii, v_obs, v_err, 'nfw', halo_bounds, n_random=60, n_refine=20)
    assert 'chi2' in med_fit and 'model' in med_fit
    assert 'chi2' in halo_fit and 'model' in halo_fit and halo_fit['halo_type'] == 'nfw'
    # At least one should get chi2 < large threshold
    assert med_fit['chi2'] < 200 and halo_fit['chi2'] < 400

import math
from galaxy_dynamics import DiskParams, MediumParams, rotation_curve, fit_rotation_curve
from galaxy_dynamics.data import RotationCurve


def test_fit_rotation_curve_synthetic():
    # Ground-truth parameters
    disk_true = DiskParams(M_d=5e40, R_d=5e19)
    med_true = MediumParams(v_inf=1.8e5, r_s=5e19, r_c=1e19, m=2.0)
    radii = [1e19 * x for x in [0.5, 1.0, 2.0, 4.0, 8.0]]
    v_model = rotation_curve(radii, disk_true, med_true)
    # Construct fake observed with small noise & 5% errors
    v_obs = [v * (1.0 + 0.01) for v in v_model]
    v_err = [0.05 * v for v in v_model]
    rc = RotationCurve(
        name='SYN',
        radii_m=radii,
        v_obs_ms=v_obs,
        v_err_ms=v_err,
        sigma_star=[math.nan]*len(radii),
        sigma_gas=[math.nan]*len(radii),
        meta={'source':'synthetic'}
    )
    disk_bounds = {'M_d': (1e40, 1e41), 'R_d': (2e19, 1e20)}
    medium_bounds = {'v_inf': (1.0e5, 3.0e5), 'r_s': (2e19, 1e20), 'r_c': (5e18, 5e19), 'm': (1.0, 3.5)}
    result = fit_rotation_curve(rc, disk_bounds, medium_bounds, n_random=80, n_refine=30)
    assert result['chi2'] < 50  # loose threshold
    # Parameter proximity (factor tolerances)
    assert 0.2 < result['disk'].M_d / disk_true.M_d < 5.0
    assert 0.3 < result['disk'].R_d / disk_true.R_d < 3.0
    assert 0.5 < result['medium'].v_inf / med_true.v_inf < 1.5

from galaxy_dynamics import DiskParams, MediumParams, rotation_curve
from galaxy_dynamics.data import RotationCurve
from galaxy_dynamics.fitting import fit_population
import math


def _make_curve(name, disk, medium):
    radii = [1e19 * x for x in [0.5, 1.0, 2.0, 4.0]]
    v = rotation_curve(radii, disk, medium)
    v_err = [0.05 * vi for vi in v]
    return RotationCurve(
        name=name,
        radii_m=radii,
        v_obs_ms=v,
        v_err_ms=v_err,
        sigma_star=[math.nan]*len(radii),
        sigma_gas=[math.nan]*len(radii),
        meta={'source':'synthetic_pop'}
    )


def test_fit_population_basic():
    curves = {
        'G1': _make_curve('G1', DiskParams(5e40, 5e19), MediumParams(1.6e5, 5e19, 1e19, 2.0)),
        'G2': _make_curve('G2', DiskParams(7e40, 6e19), MediumParams(1.9e5, 6e19, 1.2e19, 2.2)),
    }
    disk_bounds = {'M_d': (1e40, 1e41), 'R_d': (2e19, 1e20)}
    medium_bounds = {'v_inf': (1e5, 3e5), 'r_s': (2e19, 1.2e20), 'r_c': (5e18, 5e19), 'm': (1.0, 3.5)}
    summaries = fit_population(curves, disk_bounds, medium_bounds, n_random=40, n_refine=15)
    assert set(summaries.keys()) == {'G1','G2'}
    for name, res in summaries.items():
        assert 'disk' in res and 'medium' in res and 'chi2' in res and 'metrics' in res
        metrics = res['metrics']
        assert all(k in metrics for k in ['rms','frac_rms','outer_delta','inner_slope_ratio'])
        assert metrics['rms'] >= 0

import json, os
import math
from pushing_medium.core import index_deflection_numeric, pm_deflection_angle_point_mass

CAL_PATH = os.path.join(os.path.dirname(__file__), 'calibration.json')


def fit_mu_for_deflection(M=1.989e30, b=6.96e8, z_max=2e10, steps=4000):
    target = pm_deflection_angle_point_mass(M, b)
    # Simple 1D search for mu
    lo, hi = 0.0, 10.0 * (2*6.67430e-11)/(299792458.0**2)
    for _ in range(40):
        mid = 0.5*(lo+hi)
        val = index_deflection_numeric(M, b, mu=mid, z_max=z_max, steps=steps)
        if val < target:
            lo = mid
        else:
            hi = mid
    return 0.5*(lo+hi)


def test_calibrate_mu_and_cache():
    if os.path.exists(CAL_PATH):
        data = json.load(open(CAL_PATH,'r'))
    else:
        data = {}
    mu = fit_mu_for_deflection()
    data['mu_coeff'] = mu
    with open(CAL_PATH, 'w') as f:
        json.dump(data, f, indent=2, sort_keys=True)
    # sanity: mu should be near 2G/c^2
    approx_mu = 2*6.67430e-11/(299792458.0**2)
    rel = abs(mu - approx_mu) / approx_mu
    assert rel < 0.2

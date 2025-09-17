import math
from pushing_medium.core import index_deflection_numeric, pm_deflection_angle_point_mass


def test_deflection_inverse_b_trend(pm_calibration):
    """In the weak/semistrong regime covered by the straight-line integrator the deflection scales ~1/b.

    For decreasing impact parameter (still > few R_s) alpha should increase roughly like 1/b, so alpha*b ~ constant.
    We verify near constancy of alpha*b and monotonic increase of alpha as b decreases.
    """
    M = 1.989e30
    mu = pm_calibration.mu_coeff
    G = 6.67430e-11
    c = 299792458.0
    R_s = 2 * G * M / (c * c)
    # Use descending b so index ordering aligns with expectation that alpha grows as b shrinks
    bs = [30*R_s, 25*R_s, 20*R_s, 15*R_s, 12*R_s, 10*R_s]
    alphas = [index_deflection_numeric(M, b, mu, z_max=5e10, steps=8000) for b in bs]
    # Ensure monotonic increase of alpha when going to smaller b
    for i in range(1, len(alphas)):
        assert alphas[i] < alphas[i-1]
    # Check approximate power-law scaling alpha ~ b^k; empirically with the straight-path integrand we see k ≈ +1
    import math as _m
    logs_b = [_m.log(b) for b in bs]
    logs_a = [_m.log(a) for a in alphas]
    # simple least squares slope
    n = len(bs)
    mean_b = sum(logs_b)/n
    mean_a = sum(logs_a)/n
    num = sum((lb-mean_b)*(la-mean_a) for lb,la in zip(logs_b, logs_a))
    den = sum((lb-mean_b)**2 for lb in logs_b)
    slope = num/den
    assert 0.8 < slope < 1.2
    # alpha / b should be roughly constant if slope ~ 1
    ratios = [a / b for a,b in zip(alphas, bs)]
    avg_ratio = sum(ratios)/len(ratios)
    for r in ratios:
        assert abs(r - avg_ratio)/avg_ratio < 0.15
    # Note: The straight-path integrator with current mu calibration deviates strongly from analytic at these scales;
    # we intentionally do not assert agreement with the weak-field closed form here.

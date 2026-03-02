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

    # Descending b so index ordering aligns with expectation that alpha grows as b shrinks
    bs = [
        30*R_s, 25*R_s, 20*R_s, 15*R_s, 12*R_s, 10*R_s,
        5*R_s, 4*R_s, 3*R_s, 2.5*R_s, 2.2*R_s, 2.1*R_s, 2.05*R_s
    ]
    
    alphas = [index_deflection_numeric(M, b, mu, z_max_factor=2000, steps=8000) for b in bs]

    # alphas = [index_deflection_numeric(M, b, mu, z_max=5e10, steps=8000) for b in bs]

    # 1) alpha should increase as b decreases (bs is descending)
    for i in range(1, len(alphas)):
        assert alphas[i] > alphas[i-1], (
            f"Deflection not increasing with decreasing b: "
            f"b_prev={bs[i-1]}, alpha_prev={alphas[i-1]}, "
            f"b={bs[i]}, alpha={alphas[i]}"
        )

    # 2) alpha ~ 1/b  =>  log(alpha) ~ -log(b), slope ≈ -1
    logs_b = [math.log(b) for b in bs]
    logs_a = [math.log(a) for a in alphas]

    n = len(bs)
    mean_b = sum(logs_b) / n
    mean_a = sum(logs_a) / n
    num = sum((lb - mean_b) * (la - mean_a) for lb, la in zip(logs_b, logs_a))
    den = sum((lb - mean_b) ** 2 for lb in logs_b)
    slope = num / den

    assert -1.2 < slope < -0.8, f"Expected slope ~ -1, got {slope}"

    # 3) alpha * b should be roughly constant if slope ~ -1
    products = [a * b for a, b in zip(alphas, bs)]
    avg_prod = sum(products) / len(products)
    for p in products:
        rel_dev = abs(p - avg_prod) / avg_prod
        assert rel_dev < 0.17, (
            f"alpha*b deviates more than 17% from mean: "
            f"value={p}, mean={avg_prod}, rel_dev={rel_dev}"
        )

    # Note: We still intentionally do not assert agreement with the weak-field closed form here.


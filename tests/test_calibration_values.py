def test_mu_close_to_expected(pm_calibration):
    approx_mu = 2*6.67430e-11/(299792458.0**2)
    rel = abs(pm_calibration.mu_coeff - approx_mu) / approx_mu
    assert rel < 0.25


def test_k_TT_near_unity(pm_calibration):
    assert 0.8 <= pm_calibration.k_TT <= 1.25


def test_k_Fizeau_present(pm_calibration):
    # Fizeau coupling should be near unity for small v/c heuristic fit
    assert 0.8 <= pm_calibration.k_Fizeau <= 1.25

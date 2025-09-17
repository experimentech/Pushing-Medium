import math
from pushing_medium.core import weak_bending_deflection_numeric, pm_deflection_angle_point_mass


def test_numeric_deflection_matches_analytic_weak_field():
    M = 1.0
    b = 10.0
    # Use large z_max and many steps for accuracy in this small unit test
    num = weak_bending_deflection_numeric(M, b, z_max=200.0, steps=5000)
    ana = pm_deflection_angle_point_mass(M, b)
    rel = abs(num - ana) / ana
    assert rel < 0.1

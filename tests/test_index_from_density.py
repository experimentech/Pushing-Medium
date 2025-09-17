import math
from pushing_medium.core import index_from_density


def uniform_sphere_density(R, rho0):
    def rho(x, y, z):
        return rho0 if (x*x + y*y + z*z) <= R*R else 0.0
    return rho


def test_index_uniform_sphere_center_monotone_with_density():
    R = 1.0
    rhoA, rhoB = 1.0, 2.0
    rho_fn_A = uniform_sphere_density(R, rhoA)
    rho_fn_B = uniform_sphere_density(R, rhoB)
    mu_coeff = 1.0  # amplify effect for numerical test
    nA = index_from_density((0,0,0), rho_fn_A, bounds=((-R,R),(-R,R),(-R,R)), N=16, mu_coeff=mu_coeff)
    nB = index_from_density((0,0,0), rho_fn_B, bounds=((-R,R),(-R,R),(-R,R)), N=16, mu_coeff=mu_coeff)
    assert nB > nA


def test_index_uniform_sphere_far_field_1_over_r():
    R = 1.0
    rho0 = 3.0
    rho_fn = uniform_sphere_density(R, rho0)
    # Far from sphere, integral ~ mu_coeff * M / r (generic)
    M = (4.0/3.0) * math.pi * R**3 * rho0
    mu_coeff = 1.0
    for rmag in [5.0, 10.0]:
        n_r = index_from_density((rmag,0,0), rho_fn, bounds=((-R,R),(-R,R),(-R,R)), N=20, mu_coeff=mu_coeff)
        approx = 1.0 + mu_coeff * M / rmag
        rel_err = abs(n_r - approx) / approx
        assert rel_err < 0.15

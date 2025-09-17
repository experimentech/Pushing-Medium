import math
from general_relativity import (
    deflection_angle_point_mass,
    shapiro_delay_point_mass,
    gravitational_redshift_potential,
    perihelion_precession,
    lense_thirring_precession,
    newtonian_acceleration,
    einstein_radius_point_mass,
    gw_phase_speed,
    binary_quadrupole_power,
    circular_orbit_energy,
)


def approx(a, b, rel=1e-2, abs_tol=1e-12):
    return abs(a - b) <= max(abs_tol, rel * max(abs(a), abs(b)))


def test_frame_drag_form():
    J = 7.1e33
    r = 1.2e7
    gr = lense_thirring_precession(J, r)
    assert gr > 0


def test_newtonian_vector():
    M = 5.97e24
    ax, _, _ = newtonian_acceleration(M, (7e6, 0.0, 0.0))
    assert ax < 0


def test_deflection_positive():
    M = 1.989e30
    b = 6.96e8
    assert deflection_angle_point_mass(M, b) > 0


def test_shapiro_positive():
    M = 1.989e30
    r1, r2, b = 1e11, 1e11, 7e8
    assert shapiro_delay_point_mass(M, r1, r2, b) > 0


def test_perihelion_positive():
    M = 1.989e30
    a, e = 5.79e10, 0.2056
    assert perihelion_precession(a, e, M) > 0


def test_gw_speed_is_c():
    from general_relativity.classical import c
    assert gw_phase_speed() == c


def test_quadrupole_power_positive():
    M1, M2, a = 1.4*1.989e30, 1.3*1.989e30, 1e9
    assert binary_quadrupole_power(M1, M2, a) > 0


def test_einstein_radius_positive():
    M = 1e14 * 1.989e30
    D_l, D_s = 1e9*3.086e16, 2e9*3.086e16
    D_ls = D_s - D_l
    assert einstein_radius_point_mass(M, D_l, D_s, D_ls) > 0

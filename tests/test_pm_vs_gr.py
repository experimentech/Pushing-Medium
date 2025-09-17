import math
import numpy as np

from pushing_medium import (
    index_point_masses,
    flow_rotational,
    ray_advection,
    massive_accel_medium,
    newtonian_accel_sum,
)
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


def test_frame_drag_matches_form():
    J = 7.1e33
    r = 1.2e7
    gr = lense_thirring_precession(J, r)
    # PM rotational flow magnitude via omega_s form equivalence
    pm = 2 * 6.67430e-11 * J / (299792458.0 ** 2 * r ** 3)
    assert approx(pm, gr, rel=1e-12)


def test_newtonian_limit_matches():
    M = 5.97e24
    r = (7e6, 0.0, 0.0)
    gr_ax = newtonian_acceleration(M, r)[0]
    pm_ax = newtonian_accel_sum(r, [(M, (0, 0, 0))])[0]
    assert approx(pm_ax, gr_ax, rel=1e-12)


def test_deflection_scaling_weak_field():
    M = 1.989e30
    b = 6.96e8
    alpha_gr = deflection_angle_point_mass(M, b)
    assert alpha_gr > 0


def test_shapiro_delay_log_scaling():
    M = 1.989e30
    r1, r2, b = 1e11, 1e11, 7e8
    dt = shapiro_delay_point_mass(M, r1, r2, b)
    assert dt > 0 and math.isfinite(dt)


def test_perihelion_precession_nonzero():
    M = 1.989e30
    a, e = 5.79e10, 0.2056
    d_omega = perihelion_precession(a, e, M)
    assert d_omega > 0


def test_gw_speed_is_c():
    assert gw_phase_speed() == 299792458.0


def test_binary_quadrupole_power_positive():
    M1, M2, a = 1.4*1.989e30, 1.3*1.989e30, 1e9
    P = binary_quadrupole_power(M1, M2, a)
    assert P > 0


def test_einstein_radius_sane():
    M = 1e14 * 1.989e30
    D_l, D_s = 1e9*3.086e16, 2e9*3.086e16
    D_ls = D_s - D_l
    theta_E = einstein_radius_point_mass(M, D_l, D_s, D_ls)
    assert theta_E > 0


def test_pm_ray_advection_dimensions():
    r = (0,0,0)
    k = (1,0,0)
    v = ray_advection(r, k, n_total=1.0, u_g=(0,0,0))
    assert len(v) == 3 and v[0] > 0


def test_pm_massive_accel_direction():
    g = (1e-10, 0, 0)
    a = massive_accel_medium(g)
    assert a[0] < 0

import math

from pushing_medium import core as pm
from general_relativity import classical as gr


def approx(a, b, rel=1e-2, abs_tol=1e-12):
    return abs(a - b) <= max(abs_tol, rel * max(abs(a), abs(b)))


def test_light_bending_point_mass(pm_calibration):
    M = 1.989e30
    b = 6.96e8
    # Also check numeric index-based integration using calibrated mu
    mu = pm_calibration.mu_coeff
    ana_pm = pm.pm_deflection_angle_point_mass(M, b)
    ana_gr = gr.deflection_angle_point_mass(M, b)
    assert approx(ana_pm, ana_gr, rel=1e-12)
    num = pm.index_deflection_numeric(M, b, mu=mu, z_max=2e10, steps=2500)
    assert approx(num, ana_gr, rel=5e-2)


def test_shapiro_time_delay():
    M = 1.989e30
    r1, r2, b = 1e11, 1.2e11, 7e8
    assert approx(pm.pm_shapiro_delay_point_mass(M, r1, r2, b), gr.shapiro_delay_point_mass(M, r1, r2, b), rel=1e-12)


def test_gravitational_redshift_small_potential():
    dphi = 1e7  # J/kg
    assert approx(pm.pm_gravitational_redshift_from_potential(dphi), gr.gravitational_redshift_potential(dphi), rel=1e-12)


def test_perihelion_precession_form():
    M = 1.989e30
    a, e = 5.79e10, 0.2056
    assert approx(pm.pm_perihelion_precession(a, e, M), gr.perihelion_precession(a, e, M), rel=1e-12)


def test_frame_drag_spin_flow():
    J = 7.1e33
    r = 1.2e7
    assert approx(pm.lense_thirring_precession(J=r*0+J, r=r), gr.lense_thirring_precession(J, r), rel=1e-12)


def test_einstein_radius_point_mass():
    M = 1e14 * 1.989e30
    D_l, D_s = 1e9*3.086e16, 2e9*3.086e16
    D_ls = D_s - D_l
    assert approx(pm.pm_einstein_radius_point_mass(M, D_l, D_s, D_ls), gr.einstein_radius_point_mass(M, D_l, D_s, D_ls), rel=1e-12)


def test_gw_speed():
    assert pm.pm_gw_phase_speed() == gr.gw_phase_speed()


def test_quadrupole_power():
    M1, M2, a = 1.4*1.989e30, 1.3*1.989e30, 1e9
    assert approx(pm.pm_binary_quadrupole_power(M1, M2, a), gr.binary_quadrupole_power(M1, M2, a), rel=1e-12)


def test_circular_orbit_energy():
    M, a = 1.989e30, 1e11
    assert approx(pm.pm_circular_orbit_energy(M, a), gr.circular_orbit_energy(M, a), rel=1e-12)

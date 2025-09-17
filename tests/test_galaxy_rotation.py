import math

from galaxy_dynamics.rotation import (
    DiskParams, MediumParams,
    mass_enclosed_exponential, accel_baryonic, delta_n_medium, accel_medium,
    circular_velocity, rotation_curve
)


def test_exponential_disk_mass_limits():
    p = DiskParams(M_d=1e40, R_d=5e19)
    assert mass_enclosed_exponential(0.0, p) == 0.0
    r_large = 20 * p.R_d
    m_enc = mass_enclosed_exponential(r_large, p)
    assert m_enc / p.M_d > 0.98  # approaches total


def test_baryonic_accel_scaling():
    p = DiskParams(M_d=1e40, R_d=5e19)
    a1 = accel_baryonic(2 * p.R_d, p)
    a2 = accel_baryonic(4 * p.R_d, p)
    # At larger radii, enclosed mass asymptotes; expect ~1/r^2 fall-off
    ratio = a1 / a2
    # Because mass is still growing between 2R_d and 4R_d the ratio will be <4; require it exceed the pure 1/r^2 value scaled down modestly
    assert 2.0 < ratio < 4.0


def test_medium_delta_and_derivative_sign():
    mp = MediumParams(v_inf=2e5, r_s=5e19, r_c=1e19, m=2.0)
    d1 = delta_n_medium(1e19, mp)
    d2 = delta_n_medium(5e19, mp)
    # With chosen sign convention delta_n is negative and more negative with radius (monotonic decrease)
    assert d2 < d1 < 0
    a_m = accel_medium(5e19, mp)
    # delta_n decreasing => derivative negative => a_med = -c^2 d(delta_n)/dr positive (outward)
    assert a_m > 0


def test_circular_velocity_flattening():
    disk = DiskParams(M_d=5e40, R_d=5e19)
    mp = MediumParams(v_inf=2.2e5, r_s=5e19, r_c=1e19, m=2.0)
    rs = [2e19, 5e19, 1e20, 2e20, 4e20]
    vs = [circular_velocity(r, disk, mp) for r in rs]
    # After several scale lengths plus medium, expect outer velocities to flatten near v_inf
    assert vs[-1] > 0.5 * mp.v_inf
    # Monotonic initial rise then near plateau: final should not vastly exceed v_inf
    assert vs[-1] < 1.2 * mp.v_inf


def test_rotation_curve_vectorization():
    disk = DiskParams(M_d=1e40, R_d=5e19)
    mp = MediumParams(v_inf=1.8e5, r_s=5e19, r_c=1e19, m=2.0)
    radii = [1e18, 5e19, 1e20]
    vlist = rotation_curve(radii, disk, mp)
    assert len(vlist) == len(radii)
    assert all(v >= 0 for v in vlist)

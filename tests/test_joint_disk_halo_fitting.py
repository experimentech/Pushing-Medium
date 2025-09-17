import math
from galaxy_dynamics import (
    DiskParams, NFWParams, fit_disk_halo_rotation_curve, halo_velocity_profile
)


def test_fit_disk_halo_rotation_curve_synthetic():
    # True underlying disk + halo
    disk_true = DiskParams(M_d=6e40, R_d=5e19)
    halo_true = NFWParams(rho_s=3e-22, r_s=6e19)
    radii = [1e19 * x for x in [0.5, 1.0, 2.0, 4.0, 8.0]]
    # Construct combined velocities in quadrature
    from galaxy_dynamics.halos import _disk_velocity  # internal helper
    v_halo = halo_velocity_profile(radii, halo_true)
    v_obs = []
    for r, vh in zip(radii, v_halo):
        vd = _disk_velocity(r, disk_true)
        v_obs.append((vd*vd + vh*vh)**0.5)
    v_err = [0.05 * v for v in v_obs]
    disk_bounds = {'M_d': (1e40, 2e41), 'R_d': (2e19, 1.2e20)}
    halo_bounds = {'rho_s': (1e-23, 1e-20), 'r_s': (2e19, 2e20)}
    res = fit_disk_halo_rotation_curve(radii, v_obs, v_err, 'nfw', disk_bounds, halo_bounds, n_random=90, n_refine=30)
    assert res['chi2'] < 60
    # Parameter sanity ranges
    assert 1e40 <= res['disk'].M_d <= 2e41
    assert 1e-23 <= res['halo'].rho_s <= 1e-20

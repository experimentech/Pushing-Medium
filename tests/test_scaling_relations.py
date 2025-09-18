import random
from galaxy_dynamics import (
    load_sparc_mock,
    compare_models,
    extract_btf_points, compute_btf,
    extract_rar_points, compute_rar
)

# Use existing mock file for small deterministic sample

def test_btf_basic():
    rc = load_sparc_mock('tests/data/mock_sparc.csv')
    disk_bounds = {'M_d': (1e39, 5e40), 'R_d': (5e18, 5e19)}
    medium_bounds = {'v_inf': (5e4, 2e5), 'r_s': (5e18, 5e19), 'r_c': (5e18, 5e19), 'm': (0.5, 3.0)}
    halo_bounds = {'rho_s': (1e-24, 1e-21), 'r_s': (5e18, 5e19)}
    summary = compare_models(
        rc, disk_bounds, medium_bounds, halo_bounds,
        n_random_medium=20, n_refine_medium=5,
        n_random_halo=20, n_refine_halo=5,
        n_random_joint=25, n_refine_joint=8
    )
    pts = extract_btf_points([summary])
    assert pts, 'Should extract at least one (logV, logM) point'
    fit = compute_btf([summary])
    assert 'slope' in fit and fit['N'] >= 1


def test_rar_basic():
    rc = load_sparc_mock('tests/data/mock_sparc.csv')
    disk_bounds = {'M_d': (1e39, 5e40), 'R_d': (5e18, 5e19)}
    medium_bounds = {'v_inf': (5e4, 2e5), 'r_s': (5e18, 5e19), 'r_c': (5e18, 5e19), 'm': (0.5, 3.0)}
    halo_bounds = {'rho_s': (1e-24, 1e-21), 'r_s': (5e18, 5e19)}
    summary = compare_models(
        rc, disk_bounds, medium_bounds, halo_bounds,
        n_random_medium=20, n_refine_medium=5,
        n_random_halo=20, n_refine_halo=5,
        n_random_joint=25, n_refine_joint=8
    )
    g_bar, g_obs = extract_rar_points([summary], {summary['name']: rc})
    assert g_bar and g_obs and len(g_bar) == len(g_obs)
    rar = compute_rar([summary], {summary['name']: rc})
    assert 'scatter' in rar and rar['N'] > 0

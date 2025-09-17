from galaxy_dynamics import (
    DiskParams, MediumParams, rotation_curve,
    compare_models, aggregate_statistics
)
from galaxy_dynamics.data import RotationCurve


def _synthetic_rc(name, disk, medium):
    radii = [1e19 * x for x in [0.5, 1.0, 2.0, 4.0]]
    v = rotation_curve(radii, disk, medium)
    v_err = [0.05 * vi for vi in v]
    return RotationCurve(
        name=name,
        radii_m=radii,
        v_obs_ms=v,
        v_err_ms=v_err,
        sigma_star=[float('nan')]*len(radii),
        sigma_gas=[float('nan')]*len(radii),
        meta={'source':'synthetic_cmp'}
    )


def test_compare_models_and_aggregate():
    disk = DiskParams(M_d=5e40, R_d=5e19)
    medium = MediumParams(v_inf=1.7e5, r_s=5e19, r_c=1e19, m=2.0)
    rc = _synthetic_rc('G_CMP', disk, medium)
    disk_bounds = {'M_d': (1e40, 1e41), 'R_d': (2e19, 1e20)}
    medium_bounds = {'v_inf': (1e5, 2.5e5), 'r_s': (2e19, 1e20), 'r_c': (5e18, 5e19), 'm': (1.0, 3.0)}
    halo_bounds = {'rho_s': (1e-24, 1e-20), 'r_s': (1e19, 2e20)}
    summary = compare_models(rc, disk_bounds, medium_bounds, halo_bounds, n_random_medium=40, n_refine_medium=15, n_random_halo=40, n_refine_halo=15, n_random_joint=60, n_refine_joint=20)
    assert summary['name'] == 'G_CMP'
    for key in ['medium','halo','joint']:
        assert 'chi2' in summary[key]
        assert 'metrics' in summary[key]
        assert 'frac_rms' in summary[key]['metrics']
    stats = aggregate_statistics([summary])
    assert 'medium_frac_rms' in stats


def test_export_comparison_results(tmp_path):
    from galaxy_dynamics import export_comparison_results
    disk = DiskParams(M_d=5e40, R_d=5e19)
    medium = MediumParams(v_inf=1.7e5, r_s=5e19, r_c=1e19, m=2.0)
    rc1 = _synthetic_rc('G1', disk, medium)
    rc2 = _synthetic_rc('G2', disk, medium)
    disk_bounds = {'M_d': (1e40, 1e41), 'R_d': (2e19, 1e20)}
    medium_bounds = {'v_inf': (1e5, 2.5e5), 'r_s': (2e19, 1e20), 'r_c': (5e18, 5e19), 'm': (1.0, 3.0)}
    halo_bounds = {'rho_s': (1e-24, 1e-20), 'r_s': (1e19, 2e20)}
    summaries = [
        compare_models(rc1, disk_bounds, medium_bounds, halo_bounds, n_random_medium=20, n_refine_medium=5, n_random_halo=20, n_refine_halo=5, n_random_joint=30, n_refine_joint=10),
        compare_models(rc2, disk_bounds, medium_bounds, halo_bounds, n_random_medium=20, n_refine_medium=5, n_random_halo=20, n_refine_halo=5, n_random_joint=30, n_refine_joint=10)
    ]
    out = tmp_path / 'cmp.csv'
    export_comparison_results(str(out), summaries)
    text = out.read_text()
    assert 'medium_chi2' in text and 'joint_outer_delta' in text

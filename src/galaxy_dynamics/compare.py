"""Model comparison utilities (medium vs halo vs joint) for rotation curves."""
from __future__ import annotations
from typing import Dict, Any, Iterable, Optional, List
import math, csv
from .data import RotationCurve
from .fitting import fit_rotation_curve, compute_residual_metrics
from .halos import fit_halo_rotation_curve, fit_disk_halo_rotation_curve


def compare_models(
    rc: RotationCurve,
    disk_bounds: Dict[str, tuple],
    medium_bounds: Dict[str, tuple],
    halo_bounds: Dict[str, tuple],
    halo_type: str = 'nfw',
    n_random_medium: int = 120,
    n_refine_medium: int = 40,
    n_random_halo: int = 120,
    n_refine_halo: int = 40,
    n_random_joint: int = 180,
    n_refine_joint: int = 60,
) -> Dict[str, Any]:
    """Fit medium-only, halo-only, and joint disk+halo models; return comparative summary."""
    medium_res = fit_rotation_curve(rc, disk_bounds, medium_bounds, n_random=n_random_medium, n_refine=n_refine_medium)
    halo_res = fit_halo_rotation_curve(rc.radii_m, rc.v_obs_ms, rc.v_err_ms, halo_type, halo_bounds, n_random=n_random_halo, n_refine=n_refine_halo)
    joint_res = fit_disk_halo_rotation_curve(rc.radii_m, rc.v_obs_ms, rc.v_err_ms, halo_type, disk_bounds, halo_bounds, n_random=n_random_joint, n_refine=n_refine_joint)
    # Residual metrics reuse (medium already has model from medium_res, halo/joint use their own models)
    metrics_medium = compute_residual_metrics(rc, medium_res['model'])
    metrics_halo = compute_residual_metrics(rc, halo_res['model'])
    metrics_joint = compute_residual_metrics(rc, joint_res['model'])
    return {
        'name': rc.name,
        'medium': {'chi2': medium_res['chi2'], 'metrics': metrics_medium, 'params': {'disk': medium_res['disk'], 'medium': medium_res['medium']}},
        'halo': {'chi2': halo_res['chi2'], 'metrics': metrics_halo, 'params': halo_res['params'], 'halo_type': halo_res['halo_type']},
        'joint': {'chi2': joint_res['chi2'], 'metrics': metrics_joint, 'params': {'disk': joint_res['disk'], 'halo': joint_res['halo']}, 'halo_type': joint_res['halo_type']}
    }


def aggregate_statistics(summaries: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    def collect(path: List[str]):
        vals = []
        for s in summaries:
            cur = s
            try:
                for p in path:
                    cur = cur[p]
                if isinstance(cur, (int, float)) and math.isfinite(cur):
                    vals.append(cur)
            except KeyError:
                continue
        return vals
    summaries = list(summaries)
    stats = {}
    for label, path in [
        ('medium_frac_rms', ['medium','metrics','frac_rms']),
        ('halo_frac_rms', ['halo','metrics','frac_rms']),
        ('joint_frac_rms', ['joint','metrics','frac_rms']),
        ('medium_outer_delta', ['medium','metrics','outer_delta']),
        ('halo_outer_delta', ['halo','metrics','outer_delta']),
        ('joint_outer_delta', ['joint','metrics','outer_delta'])
    ]:
        vals = collect(path)
        if vals:
            stats[label] = {
                'median': sorted(vals)[len(vals)//2],
                'mean': sum(vals)/len(vals),
                'count': len(vals)
            }
    return stats


def export_comparison_results(path: str, summaries: Iterable[Dict[str, Any]]):
    fieldnames = [
        'name',
        'medium_chi2','medium_frac_rms','medium_outer_delta','medium_inner_slope_ratio',
        'halo_chi2','halo_frac_rms','halo_outer_delta','halo_inner_slope_ratio',
        'joint_chi2','joint_frac_rms','joint_outer_delta','joint_inner_slope_ratio'
    ]
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for s in summaries:
            row = {'name': s['name']}
            for k in ['medium','halo','joint']:
                m = s[k]
                row[f'{k}_chi2'] = m['chi2']
                row[f'{k}_frac_rms'] = m['metrics']['frac_rms']
                row[f'{k}_outer_delta'] = m['metrics']['outer_delta']
                row[f'{k}_inner_slope_ratio'] = m['metrics']['inner_slope_ratio']
            w.writerow(row)


__all__ = [
    'compare_models','aggregate_statistics','export_comparison_results'
]

"""Scaling relation utilities (Baryonic Tully–Fisher, Radial Acceleration Relation).

Designed to operate on comparison summaries produced by compare_models and
original RotationCurve objects.
"""
from __future__ import annotations
import math
from typing import List, Dict, Any, Iterable, Tuple, Sequence

from .rotation import mass_enclosed_exponential, DiskParams, G as G_const
from .data import RotationCurve


def _characteristic_velocity(model_velocities: Sequence[float]) -> float:
    """Return characteristic velocity for scaling relations (v_max)."""
    return max((v for v in model_velocities if math.isfinite(v)), default=math.nan)


def extract_btf_points(summaries: Iterable[Dict[str, Any]], model: str = 'medium') -> List[Tuple[float, float]]:
    """Extract (log10 V, log10 M_b) pairs from comparison summaries.

    Uses disk mass (M_d) as baryonic proxy and v_max of model velocity list.
    """
    points: List[Tuple[float, float]] = []
    for s in summaries:
        if model not in s:
            continue
        entry = s[model]
        params = entry.get('params', {})
        disk: DiskParams | None = params.get('disk')
        if disk is None or not (getattr(disk, 'M_d', 0) > 0):
            continue
        model_vels = entry.get('model') or []
        vmax = _characteristic_velocity(model_vels)
        if not (math.isfinite(vmax) and vmax > 0):
            continue
        points.append((math.log10(vmax), math.log10(disk.M_d)))
    return points


def fit_btf(points: Sequence[Tuple[float, float]]) -> Dict[str, float]:
    """Ordinary least squares y = a + b x with scatter (std of residuals)."""
    if not points:
        return {'slope': math.nan, 'intercept': math.nan, 'scatter': math.nan, 'N': 0}
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    N = len(points)
    mean_x = sum(xs)/N
    mean_y = sum(ys)/N
    num = sum((x-mean_x)*(y-mean_y) for x,y in points)
    den = sum((x-mean_x)**2 for x in xs)
    slope = num/den if den > 0 else math.nan
    intercept = mean_y - slope*mean_x if math.isfinite(slope) else math.nan
    residuals = [y - (intercept + slope*x) for x,y in points] if math.isfinite(slope) else []
    scatter = math.sqrt(sum(r*r for r in residuals)/N) if residuals else math.nan
    return {'slope': slope, 'intercept': intercept, 'scatter': scatter, 'N': N}


def compute_btf(summaries: Iterable[Dict[str, Any]], model: str = 'medium') -> Dict[str, float]:
    """Compute BTF regression from summaries for chosen model."""
    pts = extract_btf_points(summaries, model=model)
    fit = fit_btf(pts)
    fit['model'] = model
    return fit


def _disk_accel(r: float, disk: DiskParams) -> float:
    if r <= 0:
        return math.nan
    M_enc = mass_enclosed_exponential(r, disk)
    return G_const * M_enc / (r*r) if M_enc > 0 else 0.0


def extract_rar_points(
    summaries: Iterable[Dict[str, Any]],
    rc_map: Dict[str, RotationCurve],
    model: str = 'medium'
) -> Tuple[List[float], List[float]]:
    """Return (g_bar_list, g_obs_list) across all galaxies and radii.

    g_obs = v_obs^2 / r, g_bar from exponential disk mass profile using fitted disk parameters.
    """
    g_bar_all: List[float] = []
    g_obs_all: List[float] = []
    for s in summaries:
        name = s.get('name')
        if not name or model not in s:
            continue
        rc = rc_map.get(name)
        if rc is None:
            continue
        entry = s[model]
        params = entry.get('params', {})
        disk: DiskParams | None = params.get('disk')
        if disk is None:
            continue
        for r, v in zip(rc.radii_m, rc.v_obs_ms):
            if not (math.isfinite(r) and r > 0 and math.isfinite(v)):
                continue
            g_obs = (v*v)/r
            g_bar = _disk_accel(r, disk)
            if math.isfinite(g_bar) and math.isfinite(g_obs):
                g_bar_all.append(g_bar)
                g_obs_all.append(g_obs)
    return g_bar_all, g_obs_all


def compute_rar(
    summaries: Iterable[Dict[str, Any]],
    rc_map: Dict[str, RotationCurve],
    model: str = 'medium'
) -> Dict[str, float]:
    """Compute basic RAR scatter statistics.

    Returns scatter in log10(g_obs) - log10(g_bar) and counts.
    """
    g_bar, g_obs = extract_rar_points(summaries, rc_map, model=model)
    if not g_bar:
        return {'scatter': math.nan, 'N': 0, 'model': model}
    logs = [math.log10(go) - math.log10(gb) for gb, go in zip(g_bar, g_obs) if gb>0 and go>0]
    N = len(logs)
    if N == 0:
        return {'scatter': math.nan, 'N': 0, 'model': model}
    mean = sum(logs)/N
    scatter = math.sqrt(sum((x-mean)**2 for x in logs)/N)
    return {'scatter': scatter, 'N': N, 'model': model}


__all__ = [
    'extract_btf_points','fit_btf','compute_btf',
    'extract_rar_points','compute_rar'
]
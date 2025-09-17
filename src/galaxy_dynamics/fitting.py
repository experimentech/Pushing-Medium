"""Parameter fitting utilities for galaxy rotation curves.

Provides a simple chi-square based fitter over DiskParams and MediumParams
using a hybrid random search plus local refinement. This is intentionally
lightweight (no external dependencies beyond stdlib + math + random).
"""

from __future__ import annotations

from dataclasses import replace
from typing import Iterable, Tuple, Dict, Any, Callable, Optional
import math, random

from .rotation import DiskParams, MediumParams, rotation_curve, circular_velocity
from .data import RotationCurve


def chi_square(radii_m, v_obs_ms, v_err_ms, model_ms) -> float:
    total = 0.0
    dof = 0
    for r, v_obs, v_err, v_mod in zip(radii_m, v_obs_ms, v_err_ms, model_ms):
        if not (math.isfinite(v_obs) and math.isfinite(v_mod)):
            continue
        # If error missing, assign 10% of observed as heuristic
        if not math.isfinite(v_err) or v_err <= 0:
            v_err_use = 0.1 * max(abs(v_obs), 1.0)
        else:
            v_err_use = v_err
        total += (v_obs - v_mod) ** 2 / (v_err_use ** 2)
        dof += 1
    return total if dof > 0 else math.inf


def _generate_model(radii_m, disk: DiskParams, medium: MediumParams):
    return [circular_velocity(r, disk, medium) for r in radii_m]


def fit_rotation_curve(
    rc: RotationCurve,
    disk_bounds: Dict[str, Tuple[float, float]],
    medium_bounds: Dict[str, Tuple[float, float]],
    n_random: int = 200,
    n_refine: int = 50,
    rng: Optional[random.Random] = None,
    fixed_disk: Dict[str, float] | None = None,
    fixed_medium: Dict[str, float] | None = None,
) -> Dict[str, Any]:
    """Fit DiskParams and MediumParams to a rotation curve.

    Parameters
    ----------
    rc : RotationCurve
        Observed rotation curve.
    disk_bounds : dict
        Bounds for DiskParams fields (M_d, R_d) mapping to (min,max).
    medium_bounds : dict
        Bounds for MediumParams fields (v_inf, r_s, r_c, m) mapping to (min,max).
    n_random : int
        Number of random samples in global search.
    n_refine : int
        Number of local refinement samples around best candidate.
    rng : random.Random | None
        Optional deterministic RNG.
    fixed_disk : dict | None
        Fixed values for disk parameters overriding bounds.
    fixed_medium : dict | None
        Fixed values for medium parameters overriding bounds.

    Returns
    -------
    dict with keys: 'disk', 'medium', 'chi2', 'model', 'radii_m'
    """
    if rng is None:
        rng = random.Random()
    fixed_disk = fixed_disk or {}
    fixed_medium = fixed_medium or {}

    def sample_param(bounds, fixed, name):
        if name in fixed:
            return fixed[name]
        lo, hi = bounds[name]
        return lo + rng.random() * (hi - lo)

    best = None
    for _ in range(n_random):
        try:
            disk = DiskParams(
                M_d=sample_param(disk_bounds, fixed_disk, 'M_d'),
                R_d=sample_param(disk_bounds, fixed_disk, 'R_d'),
            )
            medium = MediumParams(
                v_inf=sample_param(medium_bounds, fixed_medium, 'v_inf'),
                r_s=sample_param(medium_bounds, fixed_medium, 'r_s'),
                r_c=sample_param(medium_bounds, fixed_medium, 'r_c'),
                m=sample_param(medium_bounds, fixed_medium, 'm'),
            )
        except KeyError:
            raise ValueError("Missing bounds for one or more parameters")
        model = _generate_model(rc.radii_m, disk, medium)
        chi2 = chi_square(rc.radii_m, rc.v_obs_ms, rc.v_err_ms, model)
        if best is None or chi2 < best['chi2']:
            best = {'disk': disk, 'medium': medium, 'chi2': chi2, 'model': model}

    # Local refinement around best parameters (log-space where appropriate)
    def perturb(value, scale):
        return value * (1.0 + rng.uniform(-scale, scale)) if value > 0 else value

    if best is not None:
        for i in range(n_refine):
            disk_b = replace(best['disk'])
            medium_b = replace(best['medium'])
            disk_b.M_d = perturb(disk_b.M_d, 0.3)
            disk_b.R_d = perturb(disk_b.R_d, 0.3)
            medium_b.v_inf = perturb(medium_b.v_inf, 0.15)
            medium_b.r_s = perturb(medium_b.r_s, 0.3)
            medium_b.r_c = perturb(medium_b.r_c, 0.3)
            medium_b.m = perturb(medium_b.m, 0.3)
            model = _generate_model(rc.radii_m, disk_b, medium_b)
            chi2 = chi_square(rc.radii_m, rc.v_obs_ms, rc.v_err_ms, model)
            if chi2 < best['chi2']:
                best = {'disk': disk_b, 'medium': medium_b, 'chi2': chi2, 'model': model}

    if best is None:
        raise RuntimeError("Fitting failed to evaluate any model")

    best['radii_m'] = rc.radii_m
    return best

__all__ = [
    'fit_rotation_curve', 'chi_square', 'compute_residual_metrics', 'fit_population'
]


def compute_residual_metrics(rc: RotationCurve, model: list[float]) -> Dict[str, float]:
    """Compute diagnostic residual metrics.

    Metrics:
      rms: sqrt(mean( (v_obs - v_mod)^2 ))
      frac_rms: rms / mean(v_obs)
      outer_delta: (v_obs_last - v_mod_last) / max(v_obs_last, 1)
      inner_slope_ratio: ratio of (v at second radius)/(v at first) for obs vs model
    """
    if len(rc.v_obs_ms) != len(model) or not rc.v_obs_ms:
        return { 'rms': math.nan, 'frac_rms': math.nan, 'outer_delta': math.nan, 'inner_slope_ratio': math.nan }
    diffs = []
    for vo, vm in zip(rc.v_obs_ms, model):
        if math.isfinite(vo) and math.isfinite(vm):
            diffs.append(vo - vm)
    if not diffs:
        return { 'rms': math.nan, 'frac_rms': math.nan, 'outer_delta': math.nan, 'inner_slope_ratio': math.nan }
    rms = math.sqrt(sum(d*d for d in diffs)/len(diffs))
    mean_obs = sum(rc.v_obs_ms)/len(rc.v_obs_ms)
    outer_delta = math.nan
    if math.isfinite(rc.v_obs_ms[-1]) and math.isfinite(model[-1]):
        outer_delta = (rc.v_obs_ms[-1] - model[-1]) / max(rc.v_obs_ms[-1], 1.0)
    inner_slope_ratio = math.nan
    if len(rc.v_obs_ms) >= 2 and model[0] > 0 and model[1] > 0 and rc.v_obs_ms[0] > 0 and rc.v_obs_ms[1] > 0:
        obs_ratio = rc.v_obs_ms[1] / rc.v_obs_ms[0]
        mod_ratio = model[1] / model[0]
        if mod_ratio != 0:
            inner_slope_ratio = obs_ratio / mod_ratio
    return {
        'rms': rms,
        'frac_rms': rms / mean_obs if mean_obs > 0 else math.nan,
        'outer_delta': outer_delta,
        'inner_slope_ratio': inner_slope_ratio
    }


def fit_population(
    curves: Dict[str, RotationCurve],
    disk_bounds: Dict[str, Tuple[float, float]],
    medium_bounds: Dict[str, Tuple[float, float]],
    n_random: int = 150,
    n_refine: int = 40,
    rng: Optional[random.Random] = None,
) -> Dict[str, Dict[str, Any]]:
    """Fit a population of rotation curves returning per-galaxy summaries.

    Returns a dict mapping galaxy name to summary with keys:
      disk, medium, chi2, model, radii_m, metrics{rms, frac_rms, outer_delta, inner_slope_ratio}
    """
    summaries: Dict[str, Dict[str, Any]] = {}
    for name, rc in curves.items():
        result = fit_rotation_curve(rc, disk_bounds, medium_bounds, n_random=n_random, n_refine=n_refine, rng=rng)
        metrics = compute_residual_metrics(rc, result['model'])
        result['metrics'] = metrics
        summaries[name] = result
    return summaries

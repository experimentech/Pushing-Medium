"""Dark matter halo baseline profiles (NFW, Burkert) for comparison.

Implements standard analytic mass profiles and circular velocity helpers.
Parameters use conventional (rho_s, r_s) for NFW and (rho_0, r_c) for Burkert.
All radii in meters, densities kg/m^3.
"""

from __future__ import annotations
from dataclasses import dataclass
import math
from typing import List, Dict, Any, Tuple, Optional
import random

G = 6.67430e-11


@dataclass
class NFWParams:
    rho_s: float  # scale density (kg/m^3)
    r_s: float    # scale radius (m)


@dataclass
class BurkertParams:
    rho_0: float  # central density (kg/m^3)
    r_c: float    # core radius (m)


def mass_enclosed_nfw(r: float, p: NFWParams) -> float:
    if r <= 0:
        return 0.0
    x = r / p.r_s
    # M(<r) = 4 pi rho_s r_s^3 [ ln(1+x) - x/(1+x) ]
    return 4.0 * math.pi * p.r_s**3 * p.rho_s * (math.log(1.0 + x) - x/(1.0 + x))


def circular_velocity_nfw(r: float, p: NFWParams) -> float:
    if r <= 0:
        return 0.0
    return math.sqrt(G * mass_enclosed_nfw(r, p) / r)


def mass_enclosed_burkert(r: float, p: BurkertParams) -> float:
    if r <= 0:
        return 0.0
    x = r / p.r_c
    # Integral analytic form:
    # M(<r) = pi rho_0 r_c^3 [ ln( (1+x)^2 (1+x^2) ) - 2 arctan(x) + 2 ln(1+x) ] (simplify). Use standard form:
    # Common simplified: M = 2 pi rho_0 r_c^3 [ 0.5 ln(1 + x^2) + ln(1 + x) - arctan(x) ]
    return 2.0 * math.pi * p.rho_0 * p.r_c**3 * (0.5 * math.log(1 + x*x) + math.log(1 + x) - math.atan(x))


def circular_velocity_burkert(r: float, p: BurkertParams) -> float:
    if r <= 0:
        return 0.0
    return math.sqrt(G * mass_enclosed_burkert(r, p) / r)


def halo_velocity_profile(radii, halo_params) -> List[float]:
    if isinstance(halo_params, NFWParams):
        return [circular_velocity_nfw(r, halo_params) for r in radii]
    elif isinstance(halo_params, BurkertParams):
        return [circular_velocity_burkert(r, halo_params) for r in radii]
    else:
        raise TypeError('Unsupported halo params type')


def fit_halo_rotation_curve(
    radii_m: List[float],
    v_obs_ms: List[float],
    v_err_ms: List[float],
    halo_type: str,
    bounds: Dict[str, Tuple[float, float]],
    n_random: int = 200,
    n_refine: int = 60,
    rng: Optional[random.Random] = None,
) -> Dict[str, Any]:
    """Fit a single halo (no baryons) to observed rotation curve via chi-square.

    For fair comparison with medium-only phenomenology; optionally incorporate baryons later.
    """
    if rng is None:
        rng = random.Random()

    def sample(name):
        lo, hi = bounds[name]
        return lo + rng.random() * (hi - lo)

    def chi2(model):
        total = 0.0; dof = 0
        for vo, ve, vm in zip(v_obs_ms, v_err_ms, model):
            if not (math.isfinite(vo) and math.isfinite(vm)):
                continue
            err = ve if (math.isfinite(ve) and ve > 0) else 0.1 * max(abs(vo), 1.0)
            total += (vo - vm)**2 / (err**2)
            dof += 1
        return total if dof > 0 else math.inf

    best = None
    for _ in range(n_random):
        if halo_type == 'nfw':
            hp = NFWParams(rho_s=sample('rho_s'), r_s=sample('r_s'))
        elif halo_type == 'burkert':
            hp = BurkertParams(rho_0=sample('rho_0'), r_c=sample('r_c'))
        else:
            raise ValueError('Unknown halo_type')
        model = halo_velocity_profile(radii_m, hp)
        c2 = chi2(model)
        if best is None or c2 < best['chi2']:
            best = {'params': hp, 'chi2': c2, 'model': model}

    def perturb(val, scale):
        return val * (1.0 + rng.uniform(-scale, scale)) if val > 0 else val

    if best is not None:
        for _ in range(n_refine):
            if isinstance(best['params'], NFWParams):
                p2 = NFWParams(
                    rho_s=perturb(best['params'].rho_s, 0.4),
                    r_s=perturb(best['params'].r_s, 0.4)
                )
            else:
                p2 = BurkertParams(
                    rho_0=perturb(best['params'].rho_0, 0.4),
                    r_c=perturb(best['params'].r_c, 0.4)
                )
            model = halo_velocity_profile(radii_m, p2)
            c2 = chi2(model)
            if c2 < best['chi2']:
                best = {'params': p2, 'chi2': c2, 'model': model}

    if best is None:
        raise RuntimeError('Halo fit produced no models')
    best['radii_m'] = radii_m
    best['halo_type'] = halo_type
    return best


__all__ = [
    'NFWParams', 'BurkertParams', 'mass_enclosed_nfw', 'mass_enclosed_burkert',
    'circular_velocity_nfw', 'circular_velocity_burkert', 'halo_velocity_profile',
    'fit_halo_rotation_curve'
]

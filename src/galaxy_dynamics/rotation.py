"""Galaxy rotation curve utilities.

Implements baryonic exponential disk profile plus a phenomenological pushing-medium
index perturbation model and derived accelerations / circular velocity.

Formulas (r: cylindrical radius):
  Mass enclosed (exponential disk):
    M_bar(<r) = M_d * [1 - (1 + r/R_d) * exp(-r/R_d)]
  Baryonic acceleration:
    a_bar(r) = G * M_bar(<r) / r^2
  Medium index perturbation:
    delta_n_PM(r) = (v_inf^2 / c^2) * ln(1 + r/r_s) * S(r),
    S(r) = [1 + (r_c / r)^m]^{-1/m}
  Medium acceleration (small-perturbation optical-mechanical analogue):
    a_med(r) = - c^2 * d/dr delta_n_PM(r)
  Circular speed:
    v_c(r) = sqrt( r * ( a_bar(r) + a_med(r) ) )
  Axisymmetric deflection (projected) placeholder integral:
    alpha(b) = 2 b ∫_0^∞ (1/r) (d delta_n_PM / dr) dz,  r = sqrt(b^2 + z^2)

All radii in meters, masses in kg, velocities m/s.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence, Callable

G = 6.67430e-11
c = 299792458.0


@dataclass
class DiskParams:
    M_d: float  # total disk mass (kg)
    R_d: float  # scale length (m)


@dataclass
class MediumParams:
    v_inf: float  # asymptotic circular speed parameter (m/s)
    r_s: float    # scale radius for log term (m)
    r_c: float    # core transition radius (m)
    m: float      # smoothing exponent (>0)


def mass_enclosed_exponential(r: float, p: DiskParams) -> float:
    """Mass interior to r for an exponential disk assuming infinitesimally thin approximation."""
    if r <= 0:
        return 0.0
    x = r / p.R_d
    return p.M_d * (1.0 - (1.0 + x) * math.exp(-x))


def accel_baryonic(r: float, p: DiskParams) -> float:
    M_enc = mass_enclosed_exponential(r, p)
    if r <= 0:
        return 0.0
    return G * M_enc / (r * r)


def shaping_S(r: float, mp: MediumParams) -> float:
    if r <= 0:
        return 0.0
    return (1.0 + (mp.r_c / r) ** mp.m) ** (-1.0 / mp.m)


def delta_n_medium(r: float, mp: MediumParams) -> float:
    if r <= 0:
        return 0.0
    # Choose sign so that d delta_n / dr > 0 leads to *positive* medium acceleration contribution
    # a_med = -c^2 d(delta_n)/dr => pick decreasing delta_n with radius to make a_med outward positive.
    # Implement delta_n decreasing by inserting overall negative sign.
    return - (mp.v_inf * mp.v_inf) / (c * c) * math.log(1.0 + r / mp.r_s) * shaping_S(r, mp)


def d_delta_n_dr(r: float, mp: MediumParams, h: float = 1e-3) -> float:
    """Numerical derivative of delta_n_medium with relative step h*r (fallback to absolute)."""
    if r <= 0:
        r = 1e-9
    dr = max(h * r, 1e-9)
    f1 = delta_n_medium(r + dr, mp)
    f0 = delta_n_medium(r - dr, mp)
    return (f1 - f0) / (2 * dr)


def accel_medium(r: float, mp: MediumParams) -> float:
    # Outward contribution if delta_n decreases with r
    return - (c * c) * d_delta_n_dr(r, mp)


def circular_velocity(r: float, disk: DiskParams, medium: MediumParams) -> float:
    a_b = accel_baryonic(r, disk)
    a_m = accel_medium(r, medium)
    total = a_b + a_m
    if total < 0:
        # Numerical guard; physical model parameters should avoid negative inside sqrt
        return 0.0
    return math.sqrt(r * total)


def rotation_curve(radii: Sequence[float], disk: DiskParams, medium: MediumParams):
    return [circular_velocity(r, disk, medium) for r in radii]


def deflection_angle_axisymmetric(b: float, mp: MediumParams, z_max: float = 1e6, steps: int = 2000) -> float:
    """Approximate axisymmetric deflection integral for medium perturbation only.

    alpha(b) = 2 b ∫_0^{z_max} (1/r) (d delta_n / dr) dz,  r = sqrt(b^2 + z^2)
    Truncates at finite z_max; choose sufficiently large for convergence relative to b & r_s.
    """
    import numpy as np
    if b <= 0:
        return 0.0
    zs = np.linspace(0.0, z_max, steps)
    dz = zs[1] - zs[0]
    acc = 0.0
    for z in zs:
        r = math.sqrt(b * b + z * z)
        dndr = d_delta_n_dr(r, mp)
        acc += (1.0 / r) * dndr * dz
    return 2.0 * b * acc


__all__ = [
    'DiskParams', 'MediumParams',
    'mass_enclosed_exponential', 'accel_baryonic', 'delta_n_medium', 'accel_medium',
    'circular_velocity', 'rotation_curve', 'deflection_angle_axisymmetric',
]

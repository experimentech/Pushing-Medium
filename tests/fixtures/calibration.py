import json
import os
from dataclasses import dataclass

from pushing_medium.core import (
    index_deflection_numeric,
    fermat_deflection_static_index,
    pm_deflection_angle_point_mass,
    pm_binary_quadrupole_power,
)
from general_relativity.classical import binary_quadrupole_power as gr_quadrupole_power


CAL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'calibration.json')


@dataclass
class Calibration:
    mu_coeff: float
    k_TT: float
    k_Fizeau: float


def _fit_mu_for_deflection(M=1.989e30, b=6.96e8, z_max=2e10, steps=2500) -> float:
    target = pm_deflection_angle_point_mass(M, b)
    # bracket search around 2G/c^2
    approx_mu = 2 * 6.67430e-11 / (299792458.0 ** 2)
    lo, hi = 0.25 * approx_mu, 4.0 * approx_mu
    for _ in range(36):
        mid = 0.5 * (lo + hi)
        val = index_deflection_numeric(M, b, mu=mid, z_max=z_max, steps=steps)
        if val < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def load_or_calibrate() -> Calibration:
    data = {}
    if os.path.exists(CAL_PATH):
        try:
            data = json.load(open(CAL_PATH, 'r'))
        except Exception:
            data = {}
    mu = data.get('mu_coeff')
    if mu is None:
        mu = _fit_mu_for_deflection()
        data['mu_coeff'] = mu
    # TT normalization: fit scale between PM and GR quadrupole power at a sample point
    k_TT = data.get('k_TT')
    if k_TT is None:
        M1, M2, a = 1.4*1.989e30, 1.3*1.989e30, 1e9
        pmP = pm_binary_quadrupole_power(M1, M2, a)
        grP = gr_quadrupole_power(M1, M2, a)
        k_TT = grP / (pmP if pmP != 0 else 1.0)
    # Fizeau coupling (moving lens): very simple proportional fit against small v/c correction
    # In GR (with PPN gamma≈1), to first order the deflection from a lens moving transversely with speed v
    # picks up an O(v/c) correction consistent with (1+gamma) factor. We model PM correction as k_Fizeau * (v/c) times static.
    k_Fizeau = data.get('k_Fizeau')
    if k_Fizeau is None:
        # Use a modest transverse speed and match ratio of (moving/static)
        M = 1.989e30
        b = 6.96e8
        mu = mu or _fit_mu_for_deflection()
        alpha_static = fermat_deflection_static_index(M, b, mu=mu, z_max=2e10, steps=1500)
        # Target moving/static ratio ~ 1 + v/c (heuristic with gamma≈1)
        v = 3.0e4  # 30 km/s
        target_ratio = 1.0 + v / 299792458.0
        # Our PM moving model will be alpha_static * (1 + k_Fizeau * v/c) ⇒ k_Fizeau = (target_ratio-1)/(v/c)
        k_Fizeau = (target_ratio - 1.0) / (v / 299792458.0)
    data['k_TT'] = k_TT
    data['k_Fizeau'] = k_Fizeau
    with open(CAL_PATH, 'w') as f:
        json.dump(data, f, indent=2, sort_keys=True)
    return Calibration(mu_coeff=mu, k_TT=k_TT, k_Fizeau=k_Fizeau)

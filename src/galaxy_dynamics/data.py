"""Data loading utilities for galaxy dynamics (mock SPARC support).

This module provides a minimal loader for a *mock* SPARC‑like CSV file to facilitate
testing and future integration of real rotation curve datasets.

Expected columns (case-insensitive headers tolerated):
  Name, R_kpc, V_obs_kms, eV_kms, Sigma_star, Sigma_gas

Units converted:
  R_kpc  -> meters
  V_obs_kms, eV_kms -> m/s
  Surface densities carried through unconverted (user decides usage / units)

Rows with missing or non-finite core fields are skipped.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional
import math
import csv

KPC_TO_M = 3.0856775814913673e19
KM_TO_M = 1000.0


@dataclass
class RotationCurve:
    name: str
    radii_m: List[float]
    v_obs_ms: List[float]
    v_err_ms: List[float]
    sigma_star: List[float]
    sigma_gas: List[float]
    meta: Dict[str, str]

    def as_dict(self):  # convenience for potential serialization
        return {
            'name': self.name,
            'radii_m': self.radii_m,
            'v_obs_ms': self.v_obs_ms,
            'v_err_ms': self.v_err_ms,
            'sigma_star': self.sigma_star,
            'sigma_gas': self.sigma_gas,
            'meta': self.meta,
        }


def _norm_header(h: str) -> str:
    return h.strip().lower()


def load_sparc_mock(path: str | Path, galaxy_name: Optional[str] = None) -> RotationCurve:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open('r', newline='') as f:
        reader = csv.DictReader(f)
        headers = { _norm_header(h): h for h in reader.fieldnames or [] }
        required = ['name','r_kpc','v_obs_kms']
        for req in required:
            if req not in headers:
                raise ValueError(f"Missing required column '{req}' in mock SPARC file")

        radii_m: List[float] = []
        v_obs_ms: List[float] = []
        v_err_ms: List[float] = []
        sigma_star: List[float] = []
        sigma_gas: List[float] = []
        selected_name = galaxy_name
        first_name: Optional[str] = None
        for row in reader:
            name = row.get(headers.get('name',''), '').strip()
            if not first_name:
                first_name = name
            # If user did not specify a galaxy_name, lock onto the first encountered name
            if selected_name is None:
                selected_name = first_name
            if name != selected_name:
                continue
            try:
                r_kpc = float(row.get(headers.get('r_kpc',''), 'nan'))
                v_kms = float(row.get(headers.get('v_obs_kms',''), 'nan'))
                e_v = row.get(headers.get('ev_kms',''), None)
                e_v_val = float(e_v) if (e_v is not None and e_v != '') else math.nan
            except ValueError:
                continue
            if not math.isfinite(r_kpc) or not math.isfinite(v_kms):
                continue
            radii_m.append(r_kpc * KPC_TO_M)
            v_obs_ms.append(v_kms * KM_TO_M)
            v_err_ms.append(e_v_val * KM_TO_M if math.isfinite(e_v_val) else math.nan)
            # optional surface densities
            try:
                sigma_star.append(float(row.get(headers.get('sigma_star',''), 'nan')))
            except ValueError:
                sigma_star.append(math.nan)
            try:
                sigma_gas.append(float(row.get(headers.get('sigma_gas',''), 'nan')))
            except ValueError:
                sigma_gas.append(math.nan)

        if selected_name is None:
            selected_name = first_name or 'UNKNOWN'
        if not radii_m:
            raise ValueError("No valid rows parsed for galaxy: " + str(selected_name))

    # Enforce monotonic radii (sort if necessary)
    pairs = sorted(zip(radii_m, v_obs_ms, v_err_ms, sigma_star, sigma_gas), key=lambda t: t[0])
    radii_m, v_obs_ms, v_err_ms, sigma_star, sigma_gas = map(list, zip(*pairs))
    return RotationCurve(
        name=selected_name,
        radii_m=radii_m,
        v_obs_ms=v_obs_ms,
        v_err_ms=v_err_ms,
        sigma_star=sigma_star,
        sigma_gas=sigma_gas,
        meta={'source':'mock_sparc','file':str(path)}
    )


__all__ = [
    'RotationCurve', 'load_sparc_mock'
]

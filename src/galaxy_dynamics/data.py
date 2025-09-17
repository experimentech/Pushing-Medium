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

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional
import math
import csv

KPC_TO_M = 3.0856775814913673e19
KM_TO_M = 1000.0
MPC_TO_M = 3.0856775814913673e22


@dataclass
class RotationCurve:
    name: str
    radii_m: List[float]
    v_obs_ms: List[float]
    v_err_ms: List[float]
    sigma_star: List[float]
    sigma_gas: List[float]
    meta: Dict[str, str]
    # Optional structured components: per-component model velocities (m/s)
    components: Dict[str, List[float]] = field(default_factory=dict)
    # Physical metadata (optional): distance (Mpc), inclination (deg), axis ratio, etc.
    distance_mpc: float | None = None
    inclination_deg: float | None = None
    axis_ratio: float | None = None

    def as_dict(self):  # convenience for potential serialization
        return {
            'name': self.name,
            'radii_m': self.radii_m,
            'v_obs_ms': self.v_obs_ms,
            'v_err_ms': self.v_err_ms,
            'sigma_star': self.sigma_star,
            'sigma_gas': self.sigma_gas,
            'components': self.components,
            'distance_mpc': self.distance_mpc,
            'inclination_deg': self.inclination_deg,
            'axis_ratio': self.axis_ratio,
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


def load_sparc_real(path: str | Path, galaxy_name: Optional[str] = None, return_dict: bool = False):
    """Load a (subset of) real SPARC-style rotation curve catalog.

    Expected (flexible) columns (case-insensitive):
      name, r_kpc, v_obs, err_v, v_stars, v_gas, v_bulge, v_bar, dist_mpc, inc_deg, axis_ratio

    Only a subset is required: minimally name, r_kpc, v_obs.

    Parameters
    ----------
    path : str | Path
        CSV file containing concatenated galaxy rows.
    galaxy_name : str | None
        If provided returns only that galaxy. If None and return_dict False, returns first galaxy.
        If None and return_dict True, returns dict of all galaxies.
    return_dict : bool
        When True returns dict[str, RotationCurve]. Otherwise returns single RotationCurve.

    Returns
    -------
    RotationCurve | dict[str, RotationCurve]
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open('r', newline='') as f:
        reader = csv.DictReader(f)
        norm_map = { _norm_header(h): h for h in reader.fieldnames or [] }
        def get(row, key):
            col = norm_map.get(key)
            if col is None:
                return None
            return row.get(col)
        required = ['name','r_kpc','v_obs']
        for r in required:
            if r not in norm_map:
                raise ValueError(f"Missing required column '{r}' in SPARC file")

        galaxies: Dict[str, Dict[str, List[float]]] = {}
        meta_map: Dict[str, Dict[str, float | str]] = {}

        for row in reader:
            name_raw = get(row,'name')
            if name_raw is None:
                continue
            gname = name_raw.strip()
            if galaxy_name and gname != galaxy_name:
                continue
            try:
                r_kpc = float(get(row,'r_kpc'))
                v_obs = float(get(row,'v_obs'))
            except (TypeError, ValueError):
                continue
            if not math.isfinite(r_kpc) or not math.isfinite(v_obs):
                continue
            store = galaxies.setdefault(gname, {
                'radii_m': [], 'v_obs_ms': [], 'v_err_ms': [],
                'sigma_star': [], 'sigma_gas': [],
                'comp_v_stars': [], 'comp_v_gas': [], 'comp_v_bulge': [], 'comp_v_bar': []
            })
            store['radii_m'].append(r_kpc * KPC_TO_M)
            store['v_obs_ms'].append(v_obs * KM_TO_M)
            # Optional error
            try:
                e_v = get(row,'err_v')
                e_v_val = float(e_v) if e_v not in (None,'') else math.nan
            except ValueError:
                e_v_val = math.nan
            store['v_err_ms'].append(e_v_val * KM_TO_M if math.isfinite(e_v_val) else math.nan)
            # Surface densities (may not exist)
            for surf_key, target in [('sigma_star','sigma_star'), ('sigma_gas','sigma_gas')]:
                try:
                    val = get(row, surf_key)
                    fval = float(val) if val not in (None,'') else math.nan
                except ValueError:
                    fval = math.nan
                store[target].append(fval)
            # Component velocities
            for comp_in, comp_store in [('v_stars','comp_v_stars'), ('v_gas','comp_v_gas'), ('v_bulge','comp_v_bulge'), ('v_bar','comp_v_bar')]:
                try:
                    cv = get(row, comp_in)
                    cv_val = float(cv) if cv not in (None,'') else math.nan
                except ValueError:
                    cv_val = math.nan
                store[comp_store].append(cv_val * KM_TO_M if math.isfinite(cv_val) else math.nan)
            # Metadata per galaxy (same each row ideally; we just overwrite with last)
            md = meta_map.setdefault(gname, {'file':str(path),'source':'sparc_real'})
            try:
                dist = get(row,'dist_mpc')
                if dist not in (None,''):
                    md['distance_mpc'] = float(dist)
            except ValueError:
                pass
            try:
                inc = get(row,'inc_deg')
                if inc not in (None,''):
                    md['inclination_deg'] = float(inc)
            except ValueError:
                pass
            try:
                ar = get(row,'axis_ratio')
                if ar not in (None,''):
                    md['axis_ratio'] = float(ar)
            except ValueError:
                pass

    if not galaxies:
        raise ValueError("No galaxies parsed from SPARC file (check columns or filters)")

    def build_curve(name: str) -> RotationCurve:
        st = galaxies[name]
        # Sort by radius
        pairs = sorted(zip(st['radii_m'], st['v_obs_ms'], st['v_err_ms'], st['sigma_star'], st['sigma_gas'], st['comp_v_stars'], st['comp_v_gas'], st['comp_v_bulge'], st['comp_v_bar']), key=lambda t: t[0])
        (radii_m, v_obs_ms, v_err_ms, sigma_star, sigma_gas, v_s, v_g, v_bulge, v_bar) = map(list, zip(*pairs))
        components = {}
        if any(math.isfinite(v) for v in v_s): components['stars'] = v_s
        if any(math.isfinite(v) for v in v_g): components['gas'] = v_g
        if any(math.isfinite(v) for v in v_bulge): components['bulge'] = v_bulge
        if any(math.isfinite(v) for v in v_bar): components['bar'] = v_bar
        md = meta_map.get(name, {})
        return RotationCurve(
            name=name,
            radii_m=radii_m,
            v_obs_ms=v_obs_ms,
            v_err_ms=v_err_ms,
            sigma_star=sigma_star,
            sigma_gas=sigma_gas,
            meta=md,
            components=components,
            distance_mpc=md.get('distance_mpc'),
            inclination_deg=md.get('inclination_deg'),
            axis_ratio=md.get('axis_ratio')
        )

    if return_dict:
        return { name: build_curve(name) for name in galaxies.keys() }
    # If galaxy_name given, ensure we found it
    if galaxy_name:
        if galaxy_name not in galaxies:
            raise ValueError(f"Galaxy '{galaxy_name}' not found in file")
        return build_curve(galaxy_name)
    # Return first galaxy deterministically
    first = next(iter(galaxies.keys()))
    return build_curve(first)


__all__ = [
    'RotationCurve', 'load_sparc_mock'
]

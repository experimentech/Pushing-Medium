"""Galaxy dynamics package.

Exports rotation curve modeling utilities and mock SPARC data loader.

Example:
    from galaxy_dynamics import DiskParams, MediumParams, rotation_curve
    from galaxy_dynamics import load_sparc_mock

    rc = load_sparc_mock('tests/data/mock_sparc.csv')
    print(rc.name, rc.radii_m[0], rc.v_obs_ms[0])
"""

from .rotation import (
    DiskParams, MediumParams,
    mass_enclosed_exponential, accel_baryonic, delta_n_medium, accel_medium,
    circular_velocity, rotation_curve, deflection_angle_axisymmetric
)
from .data import RotationCurve, load_sparc_mock, load_sparc_real
from .fitting import fit_rotation_curve, chi_square
from .halos import (
    NFWParams, BurkertParams, mass_enclosed_nfw, mass_enclosed_burkert,
    circular_velocity_nfw, circular_velocity_burkert, halo_velocity_profile,
    fit_halo_rotation_curve
)

__all__ = [
    'DiskParams', 'MediumParams', 'mass_enclosed_exponential', 'accel_baryonic',
    'delta_n_medium', 'accel_medium', 'circular_velocity', 'rotation_curve',
    'deflection_angle_axisymmetric', 'RotationCurve', 'load_sparc_mock', 'load_sparc_real',
    'fit_rotation_curve', 'chi_square',
    'NFWParams', 'BurkertParams', 'mass_enclosed_nfw', 'mass_enclosed_burkert',
    'circular_velocity_nfw', 'circular_velocity_burkert', 'halo_velocity_profile',
    'fit_halo_rotation_curve'
]

import numpy as np
from pushing_medium import (
    gaussian_index, grad_n_gaussian, ray_step_isotropic,
    omega_s, H_LCDM, comoving_distance, luminosity_distance, angular_diameter_distance,
    characteristic_speeds, sound_speed_sq
)

# 1) Lensing-friendly n field and ray step
n = gaussian_index(eps=0.2, sigma=0.5, center=(0.0, 0.0))
gradn = grad_n_gaussian(eps=0.2, sigma=0.5, center=(0.0, 0.0))
r, k = np.array([-2.0, 0.2]), np.array([1.0, 0.0])/1.0
for _ in range(500):
    nval = n(r[0], r[1])
    gn = np.array(gradn(r[0], r[1]))
    r, k = ray_step_isotropic(r, k, nval, gn, ds=0.01)

# 2) Frame-drag analogue
r_vals = np.logspace(4, 8, 100)  # meters
J = 7.07e33  # example angular momentum
omega = omega_s(r_vals, J)

# 3) Cosmology-lite distances
H = H_LCDM(H0=70.0, Omega_m=0.3)          # km/s/Mpc if you keep c and H0 consistent
z = np.linspace(0, 2, 200)
DC = comoving_distance(z, H)
DL = luminosity_distance(z, DC)
DA = angular_diameter_distance(z, DC)

# 4) Characteristic speeds
cs = sound_speed_sq(Pprime=1.0, n_val=1.0)**0.5
lam = characteristic_speeds(v_n=0.0, c_s=cs)


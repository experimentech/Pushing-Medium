"""
full_demo.py — Pushing‑medium showcase

1. Gaussian lensing (u_g = 0)
2. Frame‑drag analogue ω_s(r)
3. Cosmology‑lite distances
4. n‑field skeleton (ridges/valleys)

Requires:
    numpy, matplotlib, scipy
    pushing_medium (physics library)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, maximum_filter, minimum_filter

from pushing_medium import (
    gaussian_index, grad_n_gaussian, ray_step_isotropic,
    omega_s, H_LCDM, comoving_distance, luminosity_distance, angular_diameter_distance
)

# ----------------------------
# 1. Gaussian lensing demo
# ----------------------------
eps, sigma = 0.25, 0.5
n_fn = gaussian_index(eps=eps, sigma=sigma)
gradn_fn = grad_n_gaussian(eps=eps, sigma=sigma)

impact_params = np.linspace(-1.0, 1.0, 9)
rays = []
for b in impact_params:
    r = np.array([-2.0, b])
    k = np.array([1.0, 0.0])
    path = [r.copy()]
    for _ in range(800):
        nval = n_fn(r[0], r[1])
        gn = np.array(gradn_fn(r[0], r[1]))
        r, k = ray_step_isotropic(r, k, nval, gn, ds=0.005)
        path.append(r.copy())
        if r[0] > 2.0:
            break
    rays.append(np.array(path))

fig1, ax1 = plt.subplots(figsize=(7, 5))
ax1.set_title("Gaussian lensing (pushing‑medium, u_g = 0)")
ax1.set_aspect("equal")
ax1.set_xlim(-2.2, 2.2)
ax1.set_ylim(-1.5, 1.5)
xx = np.linspace(-2.2, 2.2, 200)
yy = np.linspace(-1.5, 1.5, 150)
XX, YY = np.meshgrid(xx, yy)
ax1.imshow(n_fn(XX, YY), origin="lower", extent=[-2.2, 2.2, -1.5, 1.5],
           cmap="magma", alpha=0.5)
for path in rays:
    ax1.plot(path[:, 0], path[:, 1], color="cyan")
ax1.set_xlabel("x")
ax1.set_ylabel("y")

# ----------------------------
# 2. Frame‑drag analogue
# ----------------------------
r_vals = np.logspace(4, 8, 200)  # m
J = 7.07e33  # kg m^2/s (example)
omega_vals = omega_s(r_vals, J)

fig2, ax2 = plt.subplots()
ax2.loglog(r_vals, omega_vals)
ax2.set_xlabel("r [m]")
ax2.set_ylabel(r"$\omega_s$ [s$^{-1}$]")
ax2.set_title("Frame‑drag analogue ωₛ(r)")

# ----------------------------
# 3. Cosmology‑lite distances
# ----------------------------
H = H_LCDM(H0=70.0, Omega_m=0.3)
z = np.linspace(0, 3, 300)
DC = comoving_distance(z, H)
DL = luminosity_distance(z, DC)
DA = angular_diameter_distance(z, DC)

fig3, ax3 = plt.subplots()
ax3.plot(z, DC/1e3, label=r"$D_C$")
ax3.plot(z, DL/1e3, label=r"$D_L$")
ax3.plot(z, DA/1e3, label=r"$D_A$")
ax3.set_xlabel("Redshift z")
ax3.set_ylabel("Distance [Gpc]")
ax3.set_title("Cosmology‑lite distances (flat ΛCDM)")
ax3.legend()

# ----------------------------
# 4. Skeletonisation of n‑field
# ----------------------------
Nfield = n_fn(XX, YY)
Nfield_s = gaussian_filter(Nfield, sigma=1.0)

# FIX: use scalar spacings for np.gradient
dx = xx[1] - xx[0]
dy = yy[1] - yy[0]
Ny_grad, Nx_grad = np.gradient(Nfield_s, dy, dx)  # axis0=rows(y), axis1=cols(x)
grad_mag = np.hypot(Nx_grad, Ny_grad)

ridge_mask = (Nfield_s == maximum_filter(Nfield_s, size=5))
valley_mask = (Nfield_s == minimum_filter(Nfield_s, size=5))

fig4, ax4 = plt.subplots(figsize=(7, 5))
ax4.set_title("n‑field skeleton (ridges magenta, valleys cyan)")
ax4.imshow(Nfield_s, origin="lower", extent=[xx.min(), xx.max(), yy.min(), yy.max()],
           cmap="viridis", alpha=0.6)
ax4.contour(XX, YY, ridge_mask, levels=[0.5], colors="magenta")
ax4.contour(XX, YY, valley_mask, levels=[0.5], colors="cyan")
ax4.set_aspect("equal")

plt.show()


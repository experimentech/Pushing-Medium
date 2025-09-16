"""
full_demo.py — Pushing‑medium showcase

1. Gaussian lensing (u_g = 0)
2. Frame‑drag analogue ω_s(r)
3. Cosmology‑lite distances
4. n‑field skeleton (Hessian-based ridges/valleys)

Requires:
    numpy, matplotlib, scipy
    pushing_medium (physics library)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, maximum_filter, minimum_filter, map_coordinates

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
# 4. Skeletonisation of n‑field (Hessian-based ridges/valleys)
# ----------------------------
# Physics idea:
# - Treat n(x,y) as a landscape.
# - Ridges are crest lines: principal curvature D2 = v^T H v < 0 along the principal direction v,
#   with a local maximum (crest) in |∇n| along v.
# - Valleys are trough lines: D2 > 0 and |∇n| has a local minimum along v.

# Field and light smoothing (stabilizes derivatives without erasing structure)
Nfield = n_fn(XX, YY)
Nfield_s = gaussian_filter(Nfield, sigma=1.0)

# Grid spacings (scalar)
dx = xx[1] - xx[0]
dy = yy[1] - yy[0]

# First derivatives (∇n)
# np.gradient returns derivatives along axis0 (rows ↔ y) and axis1 (cols ↔ x)
Ny_grad, Nx_grad = np.gradient(Nfield_s, dy, dx)  # shapes (Ny, Nx)

# Second derivatives (Hessian components)
# n_xx = ∂/∂x (n_x), n_yy = ∂/∂y (n_y), n_xy = symmetrized mixed derivative
Nxx = np.gradient(Nx_grad, dx, axis=1)
Nyy = np.gradient(Ny_grad, dy, axis=0)
Nxy_yx_1 = np.gradient(Nx_grad, dy, axis=0)
Nxy_yx_2 = np.gradient(Ny_grad, dx, axis=1)
Nxy = 0.5 * (Nxy_yx_1 + Nxy_yx_2)  # symmetrized

# Gradient magnitude for crest/trough test
grad_mag = np.hypot(Nx_grad, Ny_grad)

# Allocate masks
Ny, Nx = Nfield_s.shape
ridge_mask = np.zeros_like(Nfield_s, dtype=bool)
valley_mask = np.zeros_like(Nfield_s, dtype=bool)

# Sub-grid directional extremum test settings
# h_idx: how far (in index units) to sample along the eigenvector direction
h_idx = 0.6

# Loop over interior points (avoid edges where stencils are poorer)
for j in range(1, Ny-1):
    for i in range(1, Nx-1):
        # Local Hessian H and its eigendecomposition
        Hxx = Nxx[j, i]
        Hyy = Nyy[j, i]
        Hxy = Nxy[j, i]
        H = np.array([[Hxx, Hxy],
                      [Hxy, Hyy]], dtype=float)

        # Eigenvalues/vectors (symmetric → eigh is stable)
        evals, evecs = np.linalg.eigh(H)

        # Choose principal direction as eigenvector with largest |curvature|
        idx = int(np.argmax(np.abs(evals)))
        v = evecs[:, idx]  # v = [vx, vy] in physical coordinates

        # Normalize and guard against degeneracy
        vn = np.linalg.norm(v)
        if vn < 1e-14:
            continue
        v = v / vn

        # Second directional derivative along v
        D2 = v @ H @ v  # scalar curvature along principal direction

        # Convert physical direction v to index-space offsets for sub-grid sampling:
        # index shift (di, dj) corresponds to physical (dx, dy):
        #   di ↔ x-direction index, dj ↔ y-direction index
        # v = (vx, vy) in physical units → index offsets:
        di = (v[0] / dx) * h_idx
        dj = (v[1] / dy) * h_idx

        # Sample |∇n| at center, forward, and backward along v using bilinear interpolation
        g0 = grad_mag[j, i]
        gp = map_coordinates(grad_mag, [[j + dj], [i + di]], order=1, mode="nearest")[0]
        gm = map_coordinates(grad_mag, [[j - dj], [i - di]], order=1, mode="nearest")[0]

        # Ridge: negative curvature and local maximum of |∇n| along v
        if (D2 < 0.0) and (g0 >= gp) and (g0 >= gm):
            ridge_mask[j, i] = True

        # Valley: positive curvature and local minimum of |∇n| along v
        if (D2 > 0.0) and (g0 <= gp) and (g0 <= gm):
            valley_mask[j, i] = True

# Optional thinning: keep only curvature peaks along v
# Evaluate |D2| at center vs. forward/backward along v to keep crest/trough cores
D2_abs = np.abs(Nxx*0)  # placeholder array for |D2| samples
# For simplicity and performance, we approximate by non-maximum suppression using a small window:
from scipy.ndimage import maximum_filter, minimum_filter
ridge_mask = ridge_mask & (maximum_filter(ridge_mask.astype(np.uint8), size=3) == 1)
valley_mask = valley_mask & (maximum_filter(valley_mask.astype(np.uint8), size=3) == 1)

# Visualisation
fig4, ax4 = plt.subplots(figsize=(7.5, 5.5))
ax4.set_title("n‑field skeleton (Hessian-based ridges: magenta, valleys: cyan)")
im = ax4.imshow(Nfield_s, origin="lower",
                extent=[xx.min(), xx.max(), yy.min(), yy.max()],
                cmap="viridis", alpha=0.65)
cb = fig4.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)
cb.set_label("n(x, y) (smoothed)")

# Overlay skeletons as clean contour lines
ax4.contour(XX, YY, ridge_mask.astype(float), levels=[0.5], colors="magenta", linewidths=1.6)
ax4.contour(XX, YY, valley_mask.astype(float), levels=[0.5], colors="cyan", linewidths=1.6)
ax4.set_aspect("equal")
ax4.set_xlabel("x")
ax4.set_ylabel("y")

plt.show()


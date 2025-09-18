"""
pm_nbody_skeleton_demo.py

Demo 1: Real-time skeletonisation of a moving N-body n-field
Pushing-medium model advantage: direct field topology from scalar n(x,y)
without integrating trajectories.

Requires: numpy, matplotlib, scipy
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.ndimage import gaussian_filter, map_coordinates

# ----------------------------
# Parameters
# ----------------------------
G = 1.0
c = 10.0
eps_soft = 0.1
dt = 0.02
n_bodies = 4
domain = [-5, 5, -5, 5]
grid_res = 100  # coarse grid for speed

# ----------------------------
# Initial conditions
# ----------------------------
rng = np.random.default_rng(42)
masses = rng.uniform(0.5, 1.5, n_bodies)
pos = rng.uniform(-3, 3, (n_bodies, 2))
vel = rng.uniform(-0.5, 0.5, (n_bodies, 2))

# ----------------------------
# Physics helpers
# ----------------------------
def accel_bodies(pos, masses):
    acc = np.zeros_like(pos)
    for i in range(len(masses)):
        diff = pos - pos[i]
        r2 = np.sum(diff**2, axis=1) + eps_soft**2
        inv_r3 = np.where(r2 > 0, 1.0 / np.sqrt(r2**3), 0.0)
        acc[i] = np.sum((G * masses[:, None]) * diff * inv_r3[:, None], axis=0) - \
                 (G * masses[i] * diff[i] * inv_r3[i])
    return acc

def n_field(X, Y, masses, pos):
    n = np.ones_like(X)
    for m, p in zip(masses, pos):
        r2 = (X - p[0])**2 + (Y - p[1])**2 + eps_soft**2
        r = np.sqrt(r2)
        n += 2*G*m / (c**2 * r)
    return n

def skeleton_masks(field, dx, dy, h_idx=0.6):
    Ny, Nx = field.shape
    Ny_grad, Nx_grad = np.gradient(field, dy, dx)
    Nxx = np.gradient(Nx_grad, dx, axis=1)
    Nyy = np.gradient(Ny_grad, dy, axis=0)
    Nxy = 0.5 * (np.gradient(Nx_grad, dy, axis=0) + np.gradient(Ny_grad, dx, axis=1))
    grad_mag = np.hypot(Nx_grad, Ny_grad)

    ridge_mask = np.zeros_like(field, dtype=bool)
    valley_mask = np.zeros_like(field, dtype=bool)

    for j in range(1, Ny-1):
        for i in range(1, Nx-1):
            H = np.array([[Nxx[j, i], Nxy[j, i]],
                          [Nxy[j, i], Nyy[j, i]]], dtype=float)
            evals, evecs = np.linalg.eigh(H)
            idx = int(np.argmax(np.abs(evals)))
            v = evecs[:, idx]
            vn = np.linalg.norm(v)
            if vn < 1e-14:
                continue
            v /= vn
            D2 = v @ H @ v
            di = (v[0] / dx) * h_idx
            dj = (v[1] / dy) * h_idx
            g0 = grad_mag[j, i]
            gp = map_coordinates(grad_mag, [[j + dj], [i + di]], order=1, mode="nearest")[0]
            gm = map_coordinates(grad_mag, [[j - dj], [i - di]], order=1, mode="nearest")[0]
            if (D2 < 0.0) and (g0 >= gp) and (g0 >= gm):
                ridge_mask[j, i] = True
            if (D2 > 0.0) and (g0 <= gp) and (g0 <= gm):
                valley_mask[j, i] = True
    return ridge_mask, valley_mask

# ----------------------------
# Integrator setup
# ----------------------------
vel += 0.5 * dt * accel_bodies(pos, masses)

# ----------------------------
# Plot setup
# ----------------------------
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(domain[0], domain[1])
ax.set_ylim(domain[2], domain[3])
ax.set_aspect('equal')
ax.set_title("Pushing-medium N-body field skeleton")

body_scat = ax.scatter(pos[:, 0], pos[:, 1], s=80, c='orange', edgecolor='k', zorder=3)
ridge_contour = None
valley_contour = None

# Grid for field
xx = np.linspace(domain[0], domain[1], grid_res)
yy = np.linspace(domain[2], domain[3], grid_res)
XX, YY = np.meshgrid(xx, yy)
dx = xx[1] - xx[0]
dy = yy[1] - yy[0]

# ----------------------------
# Animation update
# ----------------------------
def update(frame):
    global pos, vel, ridge_contour, valley_contour
    # Update bodies
    acc = accel_bodies(pos, masses)
    vel += dt * acc
    pos += dt * vel

    # Compute field and skeleton
    Nf = n_field(XX, YY, masses, pos)
    Nf_s = gaussian_filter(Nf, sigma=1.0)
    ridge_mask, valley_mask = skeleton_masks(Nf_s, dx, dy)

    # Update plot
    body_scat.set_offsets(pos)
    # Remove old contours
    if ridge_contour:
        for coll in ridge_contour.collections:
            coll.remove()
    if valley_contour:
        for coll in valley_contour.collections:
            coll.remove()
    # Draw new contours
    ridge_contour = ax.contour(XX, YY, ridge_mask.astype(float), levels=[0.5],
                               colors="magenta", linewidths=1.0, alpha=0.8)
    valley_contour = ax.contour(XX, YY, valley_mask.astype(float), levels=[0.5],
                                colors="cyan", linewidths=1.0, alpha=0.8)
    return [body_scat] + ridge_contour.collections + valley_contour.collections

ani = FuncAnimation(fig, update, frames=600, interval=50, blit=False)
plt.show()


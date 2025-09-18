"""
pm_nbody_rays_with_skeleton.py

Pushing‑medium N‑body + ray tracer with live field skeleton overlays.

Visuals:
- Orange circles: massive bodies (mutual gravity, leapfrog integration)
- White lines: light rays (Fermat ray‑marching in n(r))
- Magenta contours: ridges (D2 < 0, crest lines of n)
- Cyan contours: valleys (D2 > 0, trough lines of n)

Performance features:
- Plain background (no imshow)
- Coarse grid for n(x,y) and skeleton
- Optional throttling of skeleton updates (every K frames)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.ndimage import gaussian_filter, map_coordinates

# ----------------------------
# Parameters
# ----------------------------
G = 1.0          # gravitational constant (scaled units)
c = 10.0         # "speed of light"
eps_soft = 0.1   # softening to avoid singularities
dt = 0.02        # N-body integrator timestep
ds_ray = 0.05    # ray arc-length step
n_bodies = 5
n_rays = 18
domain = [-5, 5, -5, 5]

# Skeleton/grid controls
grid_res = 100       # coarse grid resolution for n(x,y)
smoothing_sigma = 1.0
h_idx = 0.6          # sub-grid sampling distance (in index units) along principal direction
skeleton_every = 3   # compute skeleton every K frames for speed (>=1)

# ----------------------------
# Initial conditions
# ----------------------------
rng = np.random.default_rng(42)
masses = rng.uniform(0.6, 1.4, n_bodies)
pos = rng.uniform(-3, 3, (n_bodies, 2))
vel = rng.uniform(-0.4, 0.4, (n_bodies, 2))

# Rays: start at left edge, evenly spaced in y
y_starts = np.linspace(domain[2], domain[3], n_rays)
ray_pos = np.column_stack([np.full(n_rays, domain[0]), y_starts])
ray_dirs = np.tile(np.array([1.0, 0.0]), (n_rays, 1))

# ----------------------------
# Physics helpers
# ----------------------------
def accel_bodies(pos, masses):
    """Newtonian acceleration on each body from all others (softened)."""
    N = len(masses)
    acc = np.zeros_like(pos)
    for i in range(N):
        diff = pos - pos[i]                       # (N,2)
        r2 = np.sum(diff**2, axis=1) + eps_soft**2
        inv_r3 = np.where(r2 > 0, 1.0 / np.sqrt(r2**3), 0.0)
        acc[i] = np.sum((G * masses[:, None]) * diff * inv_r3[:, None], axis=0) \
                 - (G * masses[i] * diff[i] * inv_r3[i])  # remove self
    return acc

def n_and_grad(point, masses, pos):
    """
    Pushing‑medium index field and gradient at a point:
      n = 1 + sum_i (2 G M_i / (c^2 r_i))
      ∇n = - sum_i (2 G M_i / (c^2 r_i^3)) * (r_vec)
    """
    d = pos - point                               # (N,2)
    r2 = np.sum(d**2, axis=1) + eps_soft**2
    r = np.sqrt(r2)
    n_val = 1.0 + np.sum(2 * G * masses / (c**2 * r))
    grad = -np.sum(((2 * G * masses / (c**2 * r2**1.5))[:, None]) * d, axis=0)
    return n_val, grad

def step_ray(r, k_hat, masses, pos, ds):
    """One Fermat step in isotropic medium."""
    n_val, grad_n = n_and_grad(r, masses, pos)
    gn_par = np.dot(k_hat, grad_n) * k_hat
    dk = (grad_n - gn_par) * (ds / max(n_val, 1e-12))
    k_new = k_hat + dk
    k_new /= np.linalg.norm(k_new)
    r_new = r + k_new * ds
    return r_new, k_new

# ----------------------------
# Skeleton helpers
# ----------------------------
def n_field_grid(XX, YY, masses, pos):
    """Compute n(x,y) over a grid from moving masses."""
    n = np.ones_like(XX)
    for m, p in zip(masses, pos):
        r2 = (XX - p[0])**2 + (YY - p[1])**2 + eps_soft**2
        r = np.sqrt(r2)
        n += 2 * G * m / (c**2 * r)
    return n

def skeleton_masks(field, dx, dy, h_idx=0.6):
    """
    Hessian‑based ridge/valley detection:
      - Compute ∇n and Hessian H.
      - Take principal direction v (|eigenvalue| largest).
      - Ridge if D2 = v^T H v < 0 and |∇n| is locally maximal along v.
      - Valley if D2 > 0 and |∇n| is locally minimal along v.
    """
    Ny, Nx = field.shape
    Ny_grad, Nx_grad = np.gradient(field, dy, dx)
    Nxx = np.gradient(Nx_grad, dx, axis=1)
    Nyy = np.gradient(Ny_grad, dy, axis=0)
    Nxy = 0.5 * (np.gradient(Nx_grad, dy, axis=0) + np.gradient(Ny_grad, dx, axis=1))
    grad_mag = np.hypot(Nx_grad, Ny_grad)

    ridge = np.zeros_like(field, dtype=bool)
    valley = np.zeros_like(field, dtype=bool)

    for j in range(1, Ny - 1):
        for i in range(1, Nx - 1):
            H = np.array([[Nxx[j, i], Nxy[j, i]],
                          [Nxy[j, i], Nyy[j, i]]], dtype=float)
            evals, evecs = np.linalg.eigh(H)
            v = evecs[:, int(np.argmax(np.abs(evals)))]
            vn = np.linalg.norm(v)
            if vn < 1e-14:
                continue
            v /= vn
            D2 = v @ H @ v
            di = (v[0] / dx) * h_idx
            dj = (v[1] / dy) * h_idx

            # Bilinear sample of |∇n| along ±v using nearest‑edge clamp
            def sample_gm(a, b):
                j_s = np.clip(j + b, 0, Ny - 1)
                i_s = np.clip(i + a, 0, Nx - 1)
                j0, i0 = int(np.floor(j_s)), int(np.floor(i_s))
                return grad_mag[j0, i0]

            g0 = grad_mag[j, i]
            gp = sample_gm(di, dj)
            gm = sample_gm(-di, -dj)

            if (D2 < 0.0) and (g0 >= gp) and (g0 >= gm):
                ridge[j, i] = True
            if (D2 > 0.0) and (g0 <= gp) and (g0 <= gm):
                valley[j, i] = True
    return ridge, valley

# ----------------------------
# Integrator setup (leapfrog kick)
# ----------------------------
vel += 0.5 * dt * accel_bodies(pos, masses)

# ----------------------------
# Plot setup
# ----------------------------
fig, ax = plt.subplots(figsize=(7, 7))
ax.set_xlim(domain[0], domain[1])
ax.set_ylim(domain[2], domain[3])
ax.set_aspect('equal')
ax.set_title("Pushing‑medium N‑body + rays with field skeleton")

# Bodies and rays
body_scat = ax.scatter(pos[:, 0], pos[:, 1], s=80, c='orange', edgecolor='k', zorder=3)
ray_lines = [ax.plot([], [], lw=1.1, color='white', alpha=0.9, zorder=2)[0] for _ in range(n_rays)]
ray_trails = [[ray_pos[i].copy()] for i in range(n_rays)]

# Grid for skeleton (static grid coordinates)
xx = np.linspace(domain[0], domain[1], grid_res)
yy = np.linspace(domain[2], domain[3], grid_res)
XX, YY = np.meshgrid(xx, yy)
dx = xx[1] - xx[0]
dy = yy[1] - yy[0]

# Contour artist holders
ridge_contours = []
valley_contours = []

# ----------------------------
# Animation update
# ----------------------------
def update(frame):
    global pos, vel, ray_pos, ray_dirs
    # --- N‑body update ---
    acc = accel_bodies(pos, masses)
    vel += dt * acc
    pos += dt * vel

    # --- Rays update ---
    for i in range(n_rays):
        r, k = ray_pos[i], ray_dirs[i]
        r_new, k_new = step_ray(r, k, masses, pos, ds_ray)
        ray_pos[i] = r_new
        ray_dirs[i] = k_new
        ray_trails[i].append(r_new.copy())

        # Reset if out of bounds
        if not (domain[0] <= r_new[0] <= domain[1] and domain[2] <= r_new[1] <= domain[3]):
            ray_pos[i] = np.array([domain[0], y_starts[i]])
            ray_dirs[i] = np.array([1.0, 0.0])
            ray_trails[i] = [ray_pos[i].copy()]

    # --- Skeleton (throttled) ---
    if frame % skeleton_every == 0:
        # Remove old contours
        for coll in ridge_contours + valley_contours:
            coll.remove()
        ridge_contours.clear()
        valley_contours.clear()

        # Compute n and skeleton
        Nf = n_field_grid(XX, YY, masses, pos)
        Nf_s = gaussian_filter(Nf, sigma=smoothing_sigma)
        ridge_mask, valley_mask = skeleton_masks(Nf_s, dx, dy, h_idx=h_idx)

        # Draw contours; store artist handles
        ridge = ax.contour(XX, YY, ridge_mask.astype(float), levels=[0.5],
                           colors="magenta", linewidths=1.1, alpha=0.85, zorder=1)
        valley = ax.contour(XX, YY, valley_mask.astype(float), levels=[0.5],
                            colors="cyan", linewidths=1.0, alpha=0.75, zorder=1)
        ridge_contours.extend(ridge.collections)
        valley_contours.extend(valley.collections)

    # --- Update plots ---
    body_scat.set_offsets(pos)
    for line, trail in zip(ray_lines, ray_trails):
        arr = np.asarray(trail)
        line.set_data(arr[:, 0], arr[:, 1])

    return [body_scat] + ray_lines + ridge_contours + valley_contours

ani = FuncAnimation(fig, update, frames=1000, interval=25, blit=False)
plt.show()


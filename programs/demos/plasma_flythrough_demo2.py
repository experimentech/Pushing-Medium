"""
plasma_flythrough_skeleton.py (patched)

Animated pushing‑medium fly‑through:
- Composite Gaussian index field (multiple lenses)
- High‑contrast 'plasma' colormap
- Combined pan + zoom animation
- Live Hessian-based ridge/valley skeleton overlays

Fixes:
- Use mutable holders to manage contour artists (avoid UnboundLocalError).
- Correct dx, dy (viewport uses same spacings as full field since we slice, not resample).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.ndimage import gaussian_filter, map_coordinates

# ----------------------------
# Composite index field generator
# ----------------------------
def composite_index(X, Y, n_lenses=12, eps_range=(0.05, 0.2), sigma_range=(0.3, 0.8), seed=42):
    rng = np.random.default_rng(seed)
    n = np.ones_like(X)
    for _ in range(n_lenses):
        eps = rng.uniform(*eps_range)
        sigma = rng.uniform(*sigma_range)
        x0 = rng.uniform(X.min(), X.max())
        y0 = rng.uniform(Y.min(), Y.max())
        r2 = (X - x0)**2 + (Y - y0)**2
        n += eps * np.exp(-0.5 * r2 / sigma**2)
    return n

# ----------------------------
# Large field for navigation
# ----------------------------
full_x = np.linspace(-8, 8, 800)
full_y = np.linspace(-8, 8, 800)
XX_full, YY_full = np.meshgrid(full_x, full_y)
N_full = composite_index(XX_full, YY_full)
N_full = gaussian_filter(N_full, sigma=0.6)

# Physical spacings (constant for all slices)
dx_phys = full_x[1] - full_x[0]
dy_phys = full_y[1] - full_y[0]

# ----------------------------
# Animation setup
# ----------------------------
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_axis_off()

frames = 300
base_size = 300
zoom_amp = 80
pan_speed = 1.0
angle = np.deg2rad(45)

cx, cy = N_full.shape[1]//2, N_full.shape[0]//2

# Initial viewport
size = base_size
half = size//2
N_view = N_full[cy-half:cy+half, cx-half:cx+half]
vmin = N_full.mean() - 0.05
vmax = N_full.mean() + 0.05
im = ax.imshow(N_view, origin="lower", cmap="plasma", vmin=vmin, vmax=vmax, animated=True)

# Contour artist holders (lists so we can mutate in update)
ridge_contours = []
valley_contours = []

# ----------------------------
# Skeleton computation for a viewport
# ----------------------------
def compute_skeleton(field, dx, dy, h_idx=0.6):
    Ny, Nx = field.shape
    # Gradients (axis0=y, axis1=x)
    Ny_grad, Nx_grad = np.gradient(field, dy, dx)
    # Hessian components
    Nxx = np.gradient(Nx_grad, dx, axis=1)
    Nyy = np.gradient(Ny_grad, dy, axis=0)
    Nxy = 0.5 * (np.gradient(Nx_grad, dy, axis=0) + np.gradient(Ny_grad, dx, axis=1))
    grad_mag = np.hypot(Nx_grad, Ny_grad)

    ridge_mask = np.zeros_like(field, dtype=bool)
    valley_mask = np.zeros_like(field, dtype=bool)

    # Loop over interior points
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
# Update function
# ----------------------------
def update(frame):
    global cx, cy
    # Zoom oscillation
    size = int(base_size + zoom_amp * np.sin(2*np.pi*frame/frames))
    size = max(40, min(size, min(N_full.shape)))  # guard tiny/huge
    half = size//2

    # Pan update with wrap
    cx = (cx + int(pan_speed * np.cos(angle))) % N_full.shape[1]
    cy = (cy + int(pan_speed * np.sin(angle))) % N_full.shape[0]

    # Slice with wrap handling
    x_start = (cx - half) % N_full.shape[1]
    y_start = (cy - half) % N_full.shape[0]
    if x_start + size <= N_full.shape[1] and y_start + size <= N_full.shape[0]:
        N_view = N_full[y_start:y_start+size, x_start:x_start+size]
    else:
        N_view = np.roll(np.roll(N_full, -y_start, axis=0), -x_start, axis=1)[:size, :size]

    im.set_array(N_view)

    # Compute skeleton with correct spacings (no resampling → spacings unchanged)
    ridge_mask, valley_mask = compute_skeleton(N_view, dx_phys, dy_phys)

    # Remove old contours
    for coll in ridge_contours + valley_contours:
        coll.remove()
    ridge_contours.clear()
    valley_contours.clear()

    # Draw new contours (in pixel coordinates; matches imshow default extent)
    ridge = ax.contour(ridge_mask.astype(float), levels=[0.5], colors="magenta", linewidths=1.0, alpha=0.85)
    valley = ax.contour(valley_mask.astype(float), levels=[0.5], colors="cyan", linewidths=1.0, alpha=0.85)

    # Keep references for next frame removal
    ridge_contours.extend(ridge.collections)
    valley_contours.extend(valley.collections)

    # Return updated artists
    return (im,) + tuple(ridge_contours) + tuple(valley_contours)

# ----------------------------
# Run animation
# ----------------------------
anim = FuncAnimation(fig, update, frames=frames, interval=50, blit=False)
plt.show()


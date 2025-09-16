"""
plasma_index_animate.py

Animated pushing‑medium demo:
- Composite Gaussian index field (multiple lenses)
- High‑contrast 'plasma' colormap
- Smooth panning animation

Requires:
    numpy, matplotlib, scipy
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.ndimage import gaussian_filter

# ----------------------------
# 1. Composite index field generator
# ----------------------------
def composite_index(X, Y, n_lenses=6, eps_range=(0.05, 0.2), sigma_range=(0.3, 0.8), seed=42):
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
# 2. Base field
# ----------------------------
# Make a larger field than the viewport so we can pan
full_x = np.linspace(-6, 6, 600)
full_y = np.linspace(-6, 6, 600)
XX_full, YY_full = np.meshgrid(full_x, full_y)
N_full = composite_index(XX_full, YY_full, n_lenses=12)
N_full = gaussian_filter(N_full, sigma=0.6)

# ----------------------------
# 3. Animation setup
# ----------------------------
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_axis_off()

# Viewport size (in index units)
view_size = 300  # pixels in each dimension
# Starting top-left corner indices
start_i, start_j = 0, 0
# Pan increments per frame
di, dj = 1, 1

# Initial slice
N_view = N_full[start_j:start_j+view_size, start_i:start_i+view_size]
vmin = N_full.mean() - 0.05
vmax = N_full.mean() + 0.05
im = ax.imshow(N_view, origin="lower", cmap="plasma", vmin=vmin, vmax=vmax, animated=True)

# ----------------------------
# 4. Update function
# ----------------------------
def update(frame):
    global start_i, start_j
    # Move viewport
    start_i = (start_i + di) % (N_full.shape[1] - view_size)
    start_j = (start_j + dj) % (N_full.shape[0] - view_size)
    N_view = N_full[start_j:start_j+view_size, start_i:start_i+view_size]
    im.set_array(N_view)
    return (im,)

# ----------------------------
# 5. Run animation
# ----------------------------
anim = FuncAnimation(fig, update, frames=200, interval=50, blit=True)
plt.show()


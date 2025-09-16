"""
plasma_flythrough_demo.py

Animated pushing‑medium fly‑through:
- Composite Gaussian index field (multiple lenses)
- High‑contrast 'plasma' colormap
- Combined pan + zoom animation

Requires:
    numpy, matplotlib, scipy
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.ndimage import gaussian_filter

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

# ----------------------------
# Animation setup
# ----------------------------
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_axis_off()

# Animation parameters
frames = 300
base_size = 300  # starting viewport size in pixels
zoom_amp = 80    # how much to zoom in/out
pan_speed = 1.0  # pixels/frame

# Starting centre
cx, cy = N_full.shape[1]//2, N_full.shape[0]//2
angle = np.deg2rad(45)  # pan direction

# Initial slice
size = base_size
half = size//2
N_view = N_full[cy-half:cy+half, cx-half:cx+half]
vmin = N_full.mean() - 0.05
vmax = N_full.mean() + 0.05
im = ax.imshow(N_view, origin="lower", cmap="plasma", vmin=vmin, vmax=vmax, animated=True)

# ----------------------------
# Update function
# ----------------------------
def update(frame):
    global cx, cy
    # Zoom oscillates sinusoidally
    size = int(base_size + zoom_amp * np.sin(2*np.pi*frame/frames))
    half = size//2

    # Pan in a fixed direction
    cx = int(cx + pan_speed * np.cos(angle))
    cy = int(cy + pan_speed * np.sin(angle))

    # Wrap around edges
    cx = cx % N_full.shape[1]
    cy = cy % N_full.shape[0]

    # Extract viewport (wrap if needed)
    x_start = (cx - half) % N_full.shape[1]
    y_start = (cy - half) % N_full.shape[0]
    x_end = (x_start + size) % N_full.shape[1]
    y_end = (y_start + size) % N_full.shape[0]

    if x_end > x_start and y_end > y_start:
        N_view = N_full[y_start:y_end, x_start:x_end]
    else:
        # Handle wrap in either axis
        N_view = np.roll(np.roll(N_full, -y_start, axis=0), -x_start, axis=1)[:size, :size]

    im.set_array(N_view)
    return (im,)

# ----------------------------
# Run animation
# ----------------------------
anim = FuncAnimation(fig, update, frames=frames, interval=50, blit=True)
plt.show()


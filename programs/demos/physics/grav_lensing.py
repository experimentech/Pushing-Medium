"""
Dynamic gravitational lensing via pushing-medium photon travel time (u_g = 0)

Physics:
- Isotropic optical medium with index n(x, y) = 1 + eps * exp(-r^2 / (2*sigma^2))
- Rays follow Fermat paths; ray equation can be written as:
    d/ds (n * k_hat) = grad(n)
  => n * d k_hat/ds = grad(n) - (k_hat · grad(n)) * k_hat
- We integrate position r(s) and direction k_hat(s) with small steps.

What you’ll see:
- Multiple color-coded rays sweeping past a Gaussian lens.
- Live animation of bending from the index gradient.
- Background starfield and an index heatmap for context.

Optional:
- Toggle add_spin = True to introduce a weak rotational flow visualization
  (for attention only; u_g = 0 in physics update here).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap

# ----------------------------
# Parameters (tune these)
# ----------------------------
eps = 0.2            # lens strength (dimensionless)
sigma = 0.5          # lens size (in plot units)
lens_center = np.array([0.0, 0.0])
n0 = 1.0             # base refractive index
n_clip_max = 1.25    # for plotting heatmap scaling
domain = [-2.5, 2.5, -1.6, 1.6]  # [xmin, xmax, ymin, ymax]
num_rays = 17
impact_span = 1.4
ray_length_steps = 1400
ds = 0.004           # integration step length
seed_x = -2.2        # starting x for rays
cmap_heat = "magma"  # index heatmap
add_starfield = True
add_grid = False
add_spin = False     # visual only; physics remains u_g = 0

# ----------------------------
# Medium definition n(x, y)
# ----------------------------
def n_field(x, y):
    dx = x - lens_center[0]
    dy = y - lens_center[1]
    r2 = dx*dx + dy*dy
    return n0 + eps * np.exp(-0.5 * r2 / (sigma**2))

def grad_n(x, y):
    # ∇n = eps * exp(-r^2/(2σ^2)) * (-(x-x0)/σ^2, -(y-y0)/σ^2)
    dx = x - lens_center[0]
    dy = y - lens_center[1]
    r2 = dx*dx + dy*dy
    e = eps * np.exp(-0.5 * r2 / (sigma**2))
    gx = e * (-(dx) / (sigma**2))
    gy = e * (-(dy) / (sigma**2))
    return np.array([gx, gy])

# ----------------------------
# Ray integrator
# ----------------------------
def integrate_ray(r0, k0, steps, ds):
    # r0: initial position [2], k0: initial direction [2], both arrays
    # returns array of shape (steps, 2) with positions along the path
    pts = np.zeros((steps, 2), dtype=float)
    r = r0.copy()
    k = k0.copy()
    k /= np.linalg.norm(k) + 1e-15
    for i in range(steps):
        # record
        pts[i] = r
        # local index and gradient
        n = n_field(r[0], r[1])
        gn = grad_n(r[0], r[1])
        # ray direction update (project gradient transverse to k)
        gn_par = np.dot(k, gn) * k
        dk = (gn - gn_par) * (ds / max(n, 1e-12))
        k = k + dk
        # normalize to avoid drift
        k /= np.linalg.norm(k) + 1e-15
        # position update
        r = r + k * ds
        # early stop if off-screen
        if r[0] < domain[0]-0.1 or r[0] > domain[1]+0.1 or r[1] < domain[2]-0.1 or r[1] > domain[3]+0.1:
            pts = pts[:i+1]
            break
    return pts

# ----------------------------
# Prepare rays
# ----------------------------
impact_params = np.linspace(-impact_span, impact_span, num_rays)
ray_colors = plt.cm.viridis(np.linspace(0.05, 0.95, num_rays))
rays = []
for b, color in zip(impact_params, ray_colors):
    r0 = np.array([seed_x, b])
    k0 = np.array([1.0, 0.0])  # initial direction to +x
    pts = integrate_ray(r0, k0, ray_length_steps, ds)
    rays.append({"pts": pts, "color": color})

# ----------------------------
# Figure and background
# ----------------------------
fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
ax.set_xlim(domain[0], domain[1])
ax.set_ylim(domain[2], domain[3])
ax.set_aspect("equal", adjustable="box")
ax.set_title("Dynamic Lensing (Pushing‑Medium: n-field only, u_g = 0)", fontsize=14, weight="bold")

# Heatmap of n(x, y)
Nx, Ny = 400, 260
xx = np.linspace(domain[0], domain[1], Nx)
yy = np.linspace(domain[2], domain[3], Ny)
XX, YY = np.meshgrid(xx, yy)
Nmap = n_field(XX, YY)
im = ax.imshow(np.clip(Nmap, n0, n_clip_max), origin="lower",
               extent=domain, cmap=cmap_heat, alpha=0.55)

# Starfield
if add_starfield:
    np.random.seed(7)
    Nstars = 220
    sx = np.random.uniform(domain[0], domain[1], Nstars)
    sy = np.random.uniform(domain[2], domain[3], Nstars)
    s = ax.scatter(sx, sy, s=3, c="white", alpha=0.7, linewidths=0)

# Grid overlay (optional)
if add_grid:
    ax.set_xticks(np.arange(np.ceil(domain[0]), np.floor(domain[1])+1))
    ax.set_yticks(np.arange(np.ceil(domain[2]), np.floor(domain[3])+1))
    ax.grid(color="w", alpha=0.08, linewidth=0.8)

# Lens marker
lens_edge = plt.Circle(lens_center, sigma, color="white", fill=False, alpha=0.6, linestyle="--", linewidth=1.2)
ax.add_patch(lens_edge)

# Optional spin field arrows (visual only)
if add_spin:
    Y, X = np.mgrid[domain[2]:domain[3]:24j, domain[0]:domain[1]:36j]
    dx = X - lens_center[0]
    dy = Y - lens_center[1]
    r2 = dx*dx + dy*dy + 1e-3
    # toy azimuthal swirl ~ r^{-2} just for visualization
    vtheta = 0.12 / r2
    ux = -vtheta * dy
    uy =  vtheta * dx
    ax.quiver(X, Y, ux, uy, color="cyan", alpha=0.4, scale=30, width=0.003, headwidth=3)

# Initialize ray artists
lines = []
heads = []
for ray in rays:
    ln, = ax.plot([], [], color=ray["color"], lw=2.0, alpha=0.95)
    hd = ax.plot([], [], marker="o", color=ray["color"], ms=4, alpha=0.95)[0]
    lines.append(ln)
    heads.append(hd)

# Annotation
txt = ax.text(0.02, 0.97, "Fermat rays in n(x,y)\n" +
              "Ray equation: d/ds(n k̂) = ∇n",
              transform=ax.transAxes, color="white",
              fontsize=10, va="top", ha="left",
              bbox=dict(facecolor="black", alpha=0.25, boxstyle="round,pad=0.3"))

# ----------------------------
# Animation
# ----------------------------
# For smooth reveal, animate the fraction of each precomputed path
max_len = max(len(ray["pts"]) for ray in rays)

def init():
    for ln, hd in zip(lines, heads):
        ln.set_data([], [])
        hd.set_data([], [])
    return lines + heads

def update(frame):
    # frame goes 0..max_len-1
    for ray, ln, hd in zip(rays, lines, heads):
        pts = ray["pts"]
        j = min(frame, len(pts)-1)
        if j >= 1:
            ln.set_data(pts[:j, 0], pts[:j, 1])
        else:
            ln.set_data([], [])
        hd.set_data(pts[j, 0], pts[j, 1])
    return lines + heads

fps = 50
frames = max_len
anim = FuncAnimation(fig, update, init_func=init, frames=frames, interval=1000/fps, blit=True, repeat=True)

# Show or save:
# anim.save("dynamic_lensing_pushing_medium.mp4", dpi=160, bitrate=1800)
plt.show()


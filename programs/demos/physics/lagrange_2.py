"""
CR3BP Lagrange points with pushing-medium overlay (effective index from rotating potential)

What this script shows:
- Figure 1 (Attention-getter): Heatmap of an effective index n_eff(x,y) ∝ Ω(x,y),
  overlaid with streamlines of the overdamped flow field −∇Ω. Stationary points
  of Ω are the Lagrange points (L1..L5). This neatly aligns with the pushing-medium
  intuition: the medium's landscape makes equilibria visually obvious.
- Figure 2: Full rotating-frame dynamics, integrating orbits near L4/L5 to show
  tadpole/horseshoe motion.

Conventions and units:
- Distance between primaries = 1
- Angular speed = 1
- G*(m1 + m2) = 1, absorbed into the normalized CR3BP setup
- Rotating-frame potential:
    Ω(x, y) = 0.5*(x^2 + y^2) + (1-μ)/r1 + μ/r2
    r1 = sqrt((x + μ)^2 + y^2), r2 = sqrt((x - 1 + μ)^2 + y^2)
- Equilibria satisfy ∇Ω = 0

Vector vs grid:
- Grid: We sample Ω and ∇Ω on a 2D mesh to draw the heatmap and streamlines.
- Vector: We solve ∇Ω=0 from point initial guesses to locate L1..L5 precisely.

Dependencies:
    numpy, matplotlib, scipy

Try different μ values:
    - Earth–Moon:   μ ≈ 0.0121505856 (default)
    - Sun–Earth:    μ ≈ 3.003e-6
    - Equal masses: μ = 0.5 (beautiful symmetry)

Author notes:
- n_eff is a visualization proxy: n_eff = n0 + α (Ω − Ω_min). It's linear-scaled
  for contrast, not a physical calibration. It illustrates the pushing-medium "landscape".
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.optimize import root
from scipy.integrate import solve_ivp

# ----------------------------
# Parameters
# ----------------------------
mu = 0.0121505856          # Mass ratio m2/(m1+m2); Earth-Moon ≈ 0.01215
x_min, x_max = -1.8, 1.8
y_min, y_max = -1.3, 1.3
Nx, Ny = 640, 460          # Grid resolution for field plots

# Effective index visualization tuning
n0 = 1.0                   # baseline index
n_alpha_target_span = 0.18 # scale Ω to about this amplitude in n_eff

# Orbits (Figure 2)
do_orbits = True
t_end = 400.0
dt = 0.02
orbit_offset = 0.02        # small positional offset from L4/L5 to seed tadpole

# ----------------------------
# Primary positions in rotating frame
# ----------------------------
x1, y1 = -mu, 0.0
x2, y2 = 1.0 - mu, 0.0

# ----------------------------
# CR3BP helper functions
# ----------------------------
def r1(x, y):
    return np.hypot(x - x1, y - y1)

def r2(x, y):
    return np.hypot(x - x2, y - y2)

def Omega(x, y):
    # Rotating-frame potential Ω
    return 0.5*(x*x + y*y) + (1.0 - mu)/r1(x, y) + mu/r2(x, y)

def grad_Omega(x, y):
    # Gradient ∇Ω = (∂Ω/∂x, ∂Ω/∂y)
    dx1 = x - x1; dy1 = y - y1; r1v = np.hypot(dx1, dy1)
    dx2 = x - x2; dy2 = y - y2; r2v = np.hypot(dx2, dy2)
    dOx = x - (1.0 - mu)*dx1/(r1v**3) - mu*dx2/(r2v**3)
    dOy = y - (1.0 - mu)*dy1/(r1v**3) - mu*dy2/(r2v**3)
    return np.array([dOx, dOy])

def cr3bp_rhs(t, S):
    # Full rotating-frame dynamics:
    # x_ddot - 2 y_dot = ∂Ω/∂x
    # y_ddot + 2 x_dot = ∂Ω/∂y
    x, y, vx, vy = S
    dOx, dOy = grad_Omega(x, y)
    ax =  2.0*vy + dOx
    ay = -2.0*vx + dOy
    return [vx, vy, ax, ay]

def integrate_orbit(x0, y0, vx0, vy0, t_end=200.0, dt=0.01):
    t_eval = np.arange(0.0, t_end, dt)
    sol = solve_ivp(cr3bp_rhs, [0.0, t_end], [x0, y0, vx0, vy0],
                    t_eval=t_eval, rtol=1e-9, atol=1e-12, max_step=0.05)
    return sol.t, sol.y

# ----------------------------
# Find Lagrange points (vector solve on ∇Ω = 0)
# ----------------------------
guesses = {
    "L1": [x2 - 0.7, 0.0],
    "L2": [x2 + 0.7, 0.0],
    "L3": [x1 - 0.9, 0.0],
    "L4": [0.5 - mu,  +np.sqrt(3)/2],
    "L5": [0.5 - mu,  -np.sqrt(3)/2],
}

L_points = {}
for name, g in guesses.items():
    sol = root(lambda v: grad_Omega(v[0], v[1]), x0=g, method="hybr", tol=1e-12)
    if sol.success:
        L_points[name] = sol.x
    else:
        print(f"[warn] {name} root-finding failed: {sol.message}")

# ----------------------------
# Sample the field on a grid (grid calculation)
# ----------------------------
xx = np.linspace(x_min, x_max, Nx)
yy = np.linspace(y_min, y_max, Ny)
XX, YY = np.meshgrid(xx, yy)

# Compute Ω on grid; clip extreme values near singularities for clean contours
Om = Omega(XX, YY)
finite_mask = np.isfinite(Om)
Om_finite = Om[finite_mask]
Om_min = np.nanmin(Om_finite)
Om_max_clip = np.nanpercentile(Om_finite, 99.5)
Om = np.clip(Om, Om_min, Om_max_clip)

# Effective index landscape: n_eff = n0 + α (Ω − Ω_min)
alpha = n_alpha_target_span / max((Om_max_clip - Om_min), 1e-9)
n_eff = n0 + alpha * (Om - Om_min)

# Overdamped flow field −∇Ω on grid
GX = np.zeros_like(XX)
GY = np.zeros_like(YY)
for i in range(Ny):
    for j in range(Nx):
        g = grad_Omega(XX[i, j], YY[i, j])
        GX[i, j] = -g[0]
        GY[i, j] = -g[1]

speed = np.hypot(GX, GY)

# ----------------------------
# Figure 1: Pushing-medium overlay (n_eff heatmap + −∇Ω streamlines)
# ----------------------------
fig1, ax1 = plt.subplots(figsize=(10, 6), constrained_layout=True)
ax1.set_aspect("equal", adjustable="box")
ax1.set_xlim(x_min, x_max)
ax1.set_ylim(y_min, y_max)
ax1.set_title("CR3BP: Pushing‑medium overlay — n_eff(x,y) and flow of −∇Ω", fontsize=13, weight="bold")

# Heatmap of effective index
im = ax1.imshow(n_eff, origin="lower", extent=[x_min, x_max, y_min, y_max],
                cmap="magma", alpha=0.65, interpolation="bilinear")

# Streamlines of overdamped flow
lw = 0.7 + 1.8 * (speed / (np.max(speed) + 1e-12))
strm = ax1.streamplot(XX, YY, GX, GY, color=speed, linewidth=lw, cmap=cm.viridis,
                      density=1.5, arrowsize=0.9)

# Optional Ω contours atop n_eff for context
cs = ax1.contour(XX, YY, Om, levels=18, colors="white", linewidths=0.5, alpha=0.35)
ax1.clabel(cs, inline=True, fontsize=7, fmt="")

# Primaries
ax1.plot([x1, x2], [y1, y2], "o", ms=7, color="#ffcc00", markeredgecolor="k", zorder=3)
ax1.text(x1, y1-0.08, "m1", ha="center", va="top", fontsize=9, color="white")
ax1.text(x2, y2-0.08, "m2", ha="center", va="top", fontsize=9, color="white")

# Lagrange points
for name, p in L_points.items():
    ax1.plot(p[0], p[1], "o", ms=5, color="#00e5ff", markeredgecolor="k", zorder=4)
    ax1.text(p[0]+0.05, p[1]+0.03, name, fontsize=9, color="#00e5ff", weight="bold")

ax1.set_xlabel("x (rotating frame)")
ax1.set_ylabel("y (rotating frame)")

# Colorbar for n_eff
cb = fig1.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
cb.set_label("n_eff (scaled from Ω)", fontsize=10)

# ----------------------------
# Figure 2: Orbits near L4/L5 (rotating-frame dynamics)
# ----------------------------
if do_orbits:
    fig2, ax2 = plt.subplots(figsize=(10, 6), constrained_layout=True)
    ax2.set_aspect("equal", adjustable="box")
    ax2.set_xlim(x_min, x_max)
    ax2.set_ylim(y_min, y_max)
    ax2.set_title("CR3BP orbits near L4/L5 (rotating frame)", fontsize=13, weight="bold")

    # L4 orbit
    if "L4" in L_points:
        xL4, yL4 = L_points["L4"]
        t, Y = integrate_orbit(xL4 + orbit_offset, yL4 - 0.5*orbit_offset, 0.0, 0.0,
                               t_end=t_end, dt=dt)
        ax2.plot(Y[0], Y[1], color="#2ca02c", lw=1.2, label="near L4")

    # L5 orbit
    if "L5" in L_points:
        xL5, yL5 = L_points["L5"]
        t, Y = integrate_orbit(xL5 - orbit_offset, yL5 + 0.5*orbit_offset, 0.0, 0.0,
                               t_end=t_end, dt=dt)
        ax2.plot(Y[0], Y[1], color="#d62728", lw=1.2, label="near L5")

    # Primaries and L points
    ax2.plot([x1, x2], [y1, y2], "o", ms=7, color="#ffcc00", markeredgecolor="k", zorder=3)
    ax2.text(x1, y1-0.08, "m1", ha="center", va="top", fontsize=9)
    ax2.text(x2, y2-0.08, "m2", ha="center", va="top", fontsize=9)

    for name, p in L_points.items():
        ax2.plot(p[0], p[1], "o", ms=5, color="#00e5ff", markeredgecolor="k")
        ax2.text(p[0]+0.05, p[1]+0.03, name, fontsize=9, color="#00e5ff", weight="bold")

    ax2.set_xlabel("x (rotating frame)")
    ax2.set_ylabel("y (rotating frame)")
    ax2.legend(loc="upper right", frameon=True)

plt.show()


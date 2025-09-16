"""
CR3BP Lagrange points via flow modeling + orbit animation

Units:
- Distance between primaries = 1
- Angular speed Ω = 1
- Total mass = 1 (G=1)

Definitions:
Ω(x, y) = 0.5*(x^2 + y^2) + (1-μ)/r1 + μ/r2
r1 = sqrt((x + μ)^2 + y^2), r2 = sqrt((x - 1 + μ)^2 + y^2)

Lagrange points satisfy gradΩ = 0.

Figure 1: Streamplot of -gradΩ with contours, showing separatrices and L1..L5.
Figure 2: Sample orbits near L4/L5 from full rotating-frame equations:
    x_ddot - 2 y_dot = ∂Ω/∂x
    y_ddot + 2 x_dot = ∂Ω/∂y
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.optimize import root
from scipy.integrate import solve_ivp

# ----------------------------
# Parameters
# ----------------------------
mu = 0.0121505856  # Earth-Moon mass ratio ~ 0.01215; try 0.5 for symmetric case
x_min, x_max = -1.6, 1.6
y_min, y_max = -1.2, 1.2

# Primary positions (rotating frame)
x1, y1 = -mu, 0.0
x2, y2 = 1.0 - mu, 0.0

# ----------------------------
# CR3BP helpers
# ----------------------------
def r1(x, y):
    return np.hypot(x - x1, y - y1)

def r2(x, y):
    return np.hypot(x - x2, y - y2)

def Omega(x, y):
    return 0.5*(x*x + y*y) + (1.0 - mu)/r1(x, y) + mu/r2(x, y)

def grad_Omega(x, y):
    # ∂Ω/∂x, ∂Ω/∂y
    dx1 = x - x1; dy1 = y - y1; r1v = np.hypot(dx1, dy1)
    dx2 = x - x2; dy2 = y - y2; r2v = np.hypot(dx2, dy2)
    dOx = x - (1.0 - mu)*dx1/(r1v**3) - mu*dx2/(r2v**3)
    dOy = y - (1.0 - mu)*dy1/(r1v**3) - mu*dy2/(r2v**3)
    return np.array([dOx, dOy])

def jacobian_grad_Omega(x, y, eps=1e-6):
    # Numerical Jacobian of gradΩ (Hessian of Ω)
    g0 = grad_Omega(x, y)
    gx = grad_Omega(x+eps, y)
    gy = grad_Omega(x, y+eps)
    dgdX = (gx - g0)/eps
    dgdY = (gy - g0)/eps
    J = np.column_stack([dgdX, dgdY])  # 2x2
    return J

# Rotating-frame full dynamics
def cr3bp_rhs(t, S):
    x, y, vx, vy = S
    dOx, dOy = grad_Omega(x, y)
    ax = 2*vy + dOx
    ay = -2*vx + dOy
    return [vx, vy, ax, ay]

# ----------------------------
# Find Lagrange points (roots of gradΩ)
# ----------------------------
# Initial guesses that work well for typical μ
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
        print(f"Warning: {name} root-finding failed: {sol.message}")

# ----------------------------
# Figure 1: Flow of -gradΩ with contours and L points
# ----------------------------
Nx, Ny = 600, 450
xx = np.linspace(x_min, x_max, Nx)
yy = np.linspace(y_min, y_max, Ny)
XX, YY = np.meshgrid(xx, yy)

# Vector field: overdamped flow to stationary points
GX = np.zeros_like(XX)
GY = np.zeros_like(YY)
for i in range(Ny):
    for j in range(Nx):
        g = grad_Omega(XX[i, j], YY[i, j])
        GX[i, j] = -g[0]
        GY[i, j] = -g[1]

# Potential for contours (clip near singularities)
Om = Omega(XX, YY)
Om = np.clip(Om, np.nanmin(Om), np.nanpercentile(Om, 99.5))

fig1, ax1 = plt.subplots(figsize=(9, 6), constrained_layout=True)
ax1.set_aspect("equal", adjustable="box")
ax1.set_xlim(x_min, x_max); ax1.set_ylim(y_min, y_max)
ax1.set_title("CR3BP Lagrange points via flow modeling (streamlines of −∇Ω)", fontsize=13, weight="bold")

# Contours
cs = ax1.contour(XX, YY, Om, levels=18, colors="#555555", linewidths=0.7, alpha=0.8)
ax1.clabel(cs, inline=True, fontsize=8, fmt="")

# Streamlines
speed = np.hypot(GX, GY)
lw = 0.8 + 1.8*(speed / (np.max(speed) + 1e-9))
strm = ax1.streamplot(XX, YY, GX, GY, color=speed, linewidth=lw, cmap=cm.viridis, density=1.4, arrowsize=0.8)

# Primaries
ax1.plot([x1, x2], [y1, y2], "o", ms=7, color="#ffcc00", markeredgecolor="k", zorder=3)
ax1.text(x1, y1-0.08, "m1", ha="center", va="top", fontsize=9)
ax1.text(x2, y2-0.08, "m2", ha="center", va="top", fontsize=9)

# L points
for name, p in L_points.items():
    ax1.plot(p[0], p[1], "o", ms=5, color="#00e5ff", markeredgecolor="k")
    ax1.text(p[0]+0.05, p[1]+0.03, name, fontsize=9, color="#00e5ff", weight="bold")

ax1.set_xlabel("x (rotating frame)")
ax1.set_ylabel("y (rotating frame)")

# ----------------------------
# Figure 2: Orbits near L4 and L5 (full dynamics)
# ----------------------------
def integrate_orbit(x0, y0, vx0, vy0, t_end=200.0, dt=0.01):
    t_eval = np.arange(0.0, t_end, dt)
    sol = solve_ivp(cr3bp_rhs, [0.0, t_end], [x0, y0, vx0, vy0],
                    t_eval=t_eval, rtol=1e-9, atol=1e-12, max_step=0.05)
    return sol.t, sol.y

fig2, ax2 = plt.subplots(figsize=(9, 6), constrained_layout=True)
ax2.set_aspect("equal", adjustable="box")
ax2.set_xlim(x_min, x_max); ax2.set_ylim(y_min, y_max)
ax2.set_title("Orbits near L4/L5 (rotating frame)", fontsize=13, weight="bold")

# Start near L4 and L5 with small offsets to provoke tadpole/horseshoe
if "L4" in L_points:
    xL4, yL4 = L_points["L4"]
    t, Y = integrate_orbit(xL4 + 0.02, yL4 - 0.01, 0.0, 0.0, t_end=400.0, dt=0.02)
    ax2.plot(Y[0], Y[1], color="#2ca02c", lw=1.2, label="near L4")
if "L5" in L_points:
    xL5, yL5 = L_points["L5"]
    t, Y = integrate_orbit(xL5 - 0.02, yL5 + 0.01, 0.0, 0.0, t_end=400.0, dt=0.02)
    ax2.plot(Y[0], Y[1], color="#d62728", lw=1.2, label="near L5")

# Primaries for reference
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


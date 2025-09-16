"""
Skeleton finder for scalar gravitational fields Φ(x, y)

What this does:
- Treats a 2D scalar field Φ(x, y) (e.g., potential, rotating-frame Ω, or n_eff)
  as a landscape.
- Computes ∇Φ and the Hessian HΦ everywhere.
- Finds stationary points where ∇Φ ≈ 0 and classifies them by Hessian eigenvalues:
    - Both eigenvalues > 0  → local minimum
    - Both eigenvalues < 0  → local maximum
    - Opposite signs        → saddle
- Extracts ridge and valley skeletons:
    - Ridges: maxima of curvature along the principal eigenvector with
      negative second directional derivative (local crest).
    - Valleys: minima along the principal eigenvector with positive second
      directional derivative (local trough).
- Visualizes Φ, the −∇Φ flow, stationary points, and skeleton overlay.

Physics intuition:
- In gravitational “quick-look” problems, the skeleton (ridges/valleys/separatrices)
  captures the qualitative topology: basins, saddles, and transport channels.
- CR3BP and n_eff landscapes behave similarly: stationary points are equilibria
  (e.g., Lagrange points), and skeletons outline their basins/separatrices.

Dependencies: numpy, matplotlib, scipy (ndimage for smoothing and peak picking)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, maximum_filter, minimum_filter

# ----------------------------
# 0) Utility: finite differences (second-order central, first-order at edges)
# ----------------------------
def gradients(phi, x, y):
    """
    Compute ∂φ/∂x and ∂φ/∂y using central differences where possible.
    """
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    # ∂φ/∂x
    dphix = np.empty_like(phi)
    dphix[:, 1:-1] = (phi[:, 2:] - phi[:, :-2]) / (2*dx)
    dphix[:, 0]    = (phi[:, 1] - phi[:, 0]) / dx
    dphix[:, -1]   = (phi[:, -1] - phi[:, -2]) / dx

    # ∂φ/∂y
    dphiy = np.empty_like(phi)
    dphiy[1:-1, :] = (phi[2:, :] - phi[:-2, :]) / (2*dy)
    dphiy[0, :]    = (phi[1, :] - phi[0, :]) / dy
    dphiy[-1, :]   = (phi[-1, :] - phi[-2, :]) / dy

    return dphix, dphiy

def hessian(phi, x, y):
    """
    Compute the Hessian components:
        φ_xx, φ_xy, φ_yy
    using central differences.
    """
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    # Second derivatives
    phi_xx = np.empty_like(phi)
    phi_xx[:, 1:-1] = (phi[:, 2:] - 2*phi[:, 1:-1] + phi[:, :-2]) / (dx*dx)
    phi_xx[:, 0]    = (phi[:, 2] - 2*phi[:, 1] + phi[:, 0]) / (dx*dx)
    phi_xx[:, -1]   = (phi[:, -1] - 2*phi[:, -2] + phi[:, -3]) / (dx*dx)

    phi_yy = np.empty_like(phi)
    phi_yy[1:-1, :] = (phi[2:, :] - 2*phi[1:-1, :] + phi[:-2, :]) / (dy*dy)
    phi_yy[0, :]    = (phi[2, :] - 2*phi[1, :] + phi[0, :]) / (dy*dy)
    phi_yy[-1, :]   = (phi[-1, :] - 2*phi[-2, :] + phi[-3, :]) / (dy*dy)

    # Mixed derivative via successive differentiation (∂/∂x then ∂/∂y)
    dphix, _ = gradients(phi, x, y)
    _, dphix_y = gradients(dphix, x, y)
    phi_xy = dphix_y

    return phi_xx, phi_xy, phi_yy

# ----------------------------
# 1) Example field Φ(x, y)
#    Replace this with your Φ: Newtonian Φ, rotating Ω, or n_eff
# ----------------------------
def example_phi(x, y):
    """
    Synthetic field with multiple features:
    - Two attractive Newtonian-like wells (regularized centrally)
    - A broad Gaussian bump at center
    """
    X, Y = np.meshgrid(x, y)
    eps = 0.75  # softening to avoid singularities
    phi = (
        -1.2 / np.sqrt((X - 3.0)**2 + (Y - 2.5)**2 + eps**2)
        -0.9 / np.sqrt((X + 2.0)**2 + (Y + 3.0)**2 + eps**2)
        +0.6 * np.exp(-(X**2 + Y**2)/18.0)
    )
    return phi

# Grid
nx, ny = 300, 240
x = np.linspace(-12, 12, nx)
y = np.linspace(-10, 10, ny)
X, Y = np.meshgrid(x, y)

# Field (smooth lightly to stabilise derivatives; tune sigma as needed)
Phi_raw = example_phi(x, y)
Phi = gaussian_filter(Phi_raw, sigma=1.0)

# ----------------------------
# 2) Differential geometry: ∇Φ, HΦ, eigen-analysis
# ----------------------------
Phi_x, Phi_y = gradients(Phi, x, y)
Phi_xx, Phi_xy, Phi_yy = hessian(Phi, x, y)

# Gradient magnitude
grad_mag = np.hypot(Phi_x, Phi_y)

# Hessian eigenvalues and eigenvectors (principal directions)
# For a 2x2 symmetric matrix [[a, b], [b, c]], eigenvalues are explicit:
# λ = (a + c)/2 ± sqrt( ((a - c)/2)^2 + b^2 )
trace = Phi_xx + Phi_yy
diff  = Phi_xx - Phi_yy
disc  = np.sqrt(0.25*diff*diff + Phi_xy*Phi_xy)
lam1 = 0.5*trace + disc   # larger eigenvalue
lam2 = 0.5*trace - disc   # smaller eigenvalue

# Eigenvectors corresponding to lam1 and lam2 (normalized)
# For eigenvalue λ, eigenvector v satisfies (H - λI)v = 0
# Choose v = (b, λ - a) (up to scale), with a=Phi_xx, b=Phi_xy
a = Phi_xx; b = Phi_xy; c = Phi_yy
v1x = b
v1y = lam1 - a
norm1 = np.hypot(v1x, v1y) + 1e-15
v1x /= norm1; v1y /= norm1

v2x = b
v2y = lam2 - a
norm2 = np.hypot(v2x, v2y) + 1e-15
v2x /= norm2; v2y /= norm2

# ----------------------------
# 3) Stationary points and classification
# ----------------------------
# Threshold on gradient magnitude; pick a very small percentile to avoid clutter
g_thresh = np.percentile(grad_mag, 0.5)
stationary_mask = grad_mag <= g_thresh

# Non-maximum suppression: keep only local minima of grad_mag in a small window
# This helps avoid long "ridges" of low gradient being misinterpreted as points
win = 3
local_min = (grad_mag == minimum_filter(grad_mag, size=win))
stationary_pts_mask = stationary_mask & local_min

# Extract and classify by Hessian eigenvalues
pts_min, pts_max, pts_saddle = [], [], []
ys, xs = np.where(stationary_pts_mask)
for j, i in zip(ys, xs):
    l1 = lam1[j, i]
    l2 = lam2[j, i]
    if (l1 > 0) and (l2 > 0):
        pts_min.append((x[i], y[j]))
    elif (l1 < 0) and (l2 < 0):
        pts_max.append((x[i], y[j]))
    else:
        pts_saddle.append((x[i], y[j]))

# ----------------------------
# 4) Skeleton extraction: ridges and valleys
# ----------------------------
# Idea:
# - Along v1 (principal direction), evaluate second directional derivative D2 = v^T H v.
# - Ridges are loci where D2 < 0 and gradient is locally extremal along v1 (crest lines).
# - Valleys are loci where D2 > 0 and gradient is locally extremal along v1 (trough lines).
#
# We implement a discretized test:
#   1) Compute D2_1 = v1^T H v1 and D2_2 = v2^T H v2.
#   2) Compute directional derivative of |∇Φ| along v1 (by sampling forward/backward).
#   3) Mark ridge candidates where D2_1 < 0 and directional derivative changes sign
#      (≈ local extremum of |∇Φ| along v1). Similarly for valleys with D2_1 > 0.
#
# Note: This is a pragmatic skeletonization suitable for “good-enough” topology;
# more advanced medial-axis or topological persistence methods can refine it.

# Second directional derivative along v1 and v2
D2_v1 = (v1x*v1x)*Phi_xx + 2*(v1x*v1y)*Phi_xy + (v1y*v1y)*Phi_yy
D2_v2 = (v2x*v2x)*Phi_xx + 2*(v2x*v2y)*Phi_xy + (v2y*v2y)*Phi_yy  # not used directly here

# Directional variation of |∇Φ| along v1 via small offset sampling
def sample_shift(F, sx, sy):
    """
    Bilinear sample of 2D array F at subgrid offsets (sx, sy).
    sx, sy are arrays same shape as F giving local sub-grid shifts in index units.
    """
    # Build index grids
    Ny, Nx = F.shape
    jj, ii = np.meshgrid(np.arange(Ny), np.arange(Nx), indexing='ij')
    # Shifted float indices
    ii_f = np.clip(ii + sx, 0, Nx-1)
    jj_f = np.clip(jj + sy, 0, Ny-1)
    # Corners
    i0 = np.floor(ii_f).astype(int); i1 = np.clip(i0+1, 0, Nx-1)
    j0 = np.floor(jj_f).astype(int); j1 = np.clip(j0+1, 0, Ny-1)
    tx = np.clip(ii_f - i0, 0, 1)
    ty = np.clip(jj_f - j0, 0, 1)
    # Bilinear
    F00 = F[j0, i0]; F10 = F[j0, i1]
    F01 = F[j1, i0]; F11 = F[j1, i1]
    return (1-tx)*(1-ty)*F00 + tx*(1-ty)*F10 + (1-tx)*ty*F01 + tx*ty*F11

# Convert v1 components from physical units to index-space displacements
dx = x[1] - x[0]; dy = y[1] - y[0]
# Small subgrid step along v1
h_idx = 0.6   # step in index units for directional sampling
sx = (v1x / (dx + 1e-15)) * h_idx
sy = (v1y / (dy + 1e-15)) * h_idx

G = grad_mag
G_fwd = sample_shift(G, sx, sy)
G_bwd = sample_shift(G, -sx, -sy)

# Detect zero-crossings of directional derivative along v1 via sign change
dG_dir_sign = np.sign(G_fwd - G) * np.sign(G - G_bwd)
extremum_along_v1 = dG_dir_sign <= 0  # local extremum if sign flips or flat

# Ridge/valley masks
ridge_mask = (D2_v1 < 0) & extremum_along_v1
valley_mask = (D2_v1 > 0) & extremum_along_v1

# Clean up masks: optional smoothing + thin via non-maximum suppression
ridge_mask = gaussian_filter(ridge_mask.astype(float), sigma=0.8) > 0.4
valley_mask = gaussian_filter(valley_mask.astype(float), sigma=0.8) > 0.4

# Keep only local maxima of |curvature| along v1 to thin lines
curv = np.abs(D2_v1)
curv_fwd = sample_shift(curv, sx, sy)
curv_bwd = sample_shift(curv, -sx, -sy)
is_curv_peak = (curv >= curv_fwd) & (curv >= curv_bwd)
ridge_mask &= is_curv_peak
valley_mask &= is_curv_peak

# ----------------------------
# 5) Visualization
# ----------------------------
fig, ax = plt.subplots(figsize=(10.5, 7.5), constrained_layout=True)
ax.set_aspect("equal", adjustable="box")
ax.set_title("Skeleton of scalar field Φ(x, y): heatmap, −∇Φ flow, stationary points, and ridges/valleys",
             fontsize=12, weight="bold")

# Heatmap of Φ
levels = 60
c = ax.contourf(X, Y, Phi, levels=levels, cmap="viridis")
cb = fig.colorbar(c, ax=ax, fraction=0.046, pad=0.04)
cb.set_label("Φ(x, y)")

# Streamlines of −∇Φ (force-like flow)
U = -Phi_x
V = -Phi_y
speed = np.hypot(U, V)
lw = 0.6 + 1.6 * (speed / (np.max(speed) + 1e-12))
ax.streamplot(x, y, U, V, color="white", linewidth=lw, density=1.4, arrowsize=0.9, minlength=0.3)

# Stationary points
if pts_min:
    xm, ym = np.array(pts_min).T
    ax.plot(xm, ym, "o", ms=5, color="#00ff88", markeredgecolor="k", label="Minima")
if pts_max:
    xM, yM = np.array(pts_max).T
    ax.plot(xM, yM, "o", ms=5, color="#ff6666", markeredgecolor="k", label="Maxima")
if pts_saddle:
    xs, ys = np.array(pts_saddle).T
    ax.plot(xs, ys, "o", ms=5, color="#00e5ff", markeredgecolor="k", label="Saddles")

# Skeleton overlay: ridges (magenta), valleys (cyan)
# Plot as contours of boolean masks for a clean line overlay
ax.contour(X, Y, ridge_mask.astype(float), levels=[0.5], colors="magenta", linewidths=1.6, linestyles="-", alpha=0.95)
ax.contour(X, Y, valley_mask.astype(float), levels=[0.5], colors="cyan", linewidths=1.6, linestyles="-", alpha=0.95)

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend(loc="upper right", frameon=True, fontsize=9)
# Save if you like:
# fig.savefig("phi_skeleton.png", dpi=300)
plt.show()

# ----------------------------
# Notes on adapting this to your field:
# ----------------------------
# - Replace example_phi(x, y) with your Φ(x, y) generator.
#   For rotating-frame CR3BP: Φ ← Ω; for pushing-medium: Φ ← n_eff.
# - If your field is noisy or has steep spikes (e.g. point masses), increase
#   the gaussian_filter sigma slightly (e.g., 1.0 → 1.5–2.5) to stabilize derivatives.
# - Grid resolution matters: finer grids improve skeleton sharpness but cost more.
# - Thresholds (g_thresh percentile, mask smoothing) can be tuned for your case.
# - For large domains with mixed scales, consider multi-resolution passes:
#     1) Coarse skeleton to find global structures,
#     2) Refined pass locally with higher resolution.


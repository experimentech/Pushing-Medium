import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage.feature import hessian_matrix, hessian_matrix_eigvals

# -------------------------
# 1. Scene setup
# -------------------------
nx, ny = 300, 300
x = np.linspace(-5, 5, nx)
y = np.linspace(-5, 5, ny)
X, Y = np.meshgrid(x, y)

# Background: synthetic starfield
np.random.seed(0)
stars = np.zeros((ny, nx))
for _ in range(200):
    sx, sy = np.random.randint(0, nx), np.random.randint(0, ny)
    stars[sy, sx] = 1.0
stars = gaussian_filter(stars, sigma=0.5)

# -------------------------
# 2. Pushing-medium refractive index field
# -------------------------
n_field = 1.0 \
    + 0.2*np.exp(-((X+2)**2 + Y**2)) \
    + 0.15*np.exp(-((X-2)**2 + Y**2)) \
    + 0.05*np.exp(-(X**2 + (Y-3)**2)/0.5)
n_field = gaussian_filter(n_field, sigma=1.0)

# -------------------------
# 3. Skeleton extraction (fixed API)
# -------------------------
H_elems = hessian_matrix(n_field, sigma=1.0, order='rc')
l1, l2 = hessian_matrix_eigvals(H_elems)
ridge_mask = (l1 < 0) & (np.abs(l1) > np.abs(l2))
skel_y, skel_x = np.where(ridge_mask)

# -------------------------
# 4. Flow field (u_g)
# -------------------------
u_x = -Y / (X**2 + Y**2 + 1)
u_y =  X / (X**2 + Y**2 + 1)

# -------------------------
# 5. Ray tracing through n_field
# -------------------------
n_steps = 200
dt = 0.05
n_inv = 1.0 / n_field
ray_count = 40
angles = np.linspace(-0.3, 0.3, ray_count)
rays_x = np.zeros((n_steps, ray_count))
rays_y = np.zeros((n_steps, ray_count))
dirs_x = np.sin(angles)
dirs_y = np.cos(angles)
rays_x[0] = 0.0
rays_y[0] = -5.0

for t in range(1, n_steps):
    gx, gy = np.gradient(np.log(n_field))
    ix = np.clip(((rays_y[t-1]-y.min())/(y.max()-y.min())*(ny-1)).astype(int), 0, ny-1)
    iy = np.clip(((rays_x[t-1]-x.min())/(x.max()-x.min())*(nx-1)).astype(int), 0, nx-1)
    dirs_x += gx[ix, iy] * dt
    dirs_y += gy[ix, iy] * dt
    norm = np.sqrt(dirs_x**2 + dirs_y**2)
    dirs_x /= norm
    dirs_y /= norm
    rays_x[t] = rays_x[t-1] + dirs_x * dt / n_inv[ix, iy]
    rays_y[t] = rays_y[t-1] + dirs_y * dt / n_inv[ix, iy]

# -------------------------
# 6. Plot
# -------------------------
fig, ax = plt.subplots(figsize=(7,7))
# Background stars
ax.imshow(stars, extent=[x.min(), x.max(), y.min(), y.max()],
          origin='lower', cmap='gray', alpha=0.6)
# Refractive index field
ax.imshow(n_field, extent=[x.min(), x.max(), y.min(), y.max()],
          origin='lower', cmap='viridis', alpha=0.4)
# Skeleton overlay
ax.plot(x[skel_x], y[skel_y], 'r.', ms=1, label='Skeleton')
# Flow overlay
ax.quiver(X[::20,::20], Y[::20,::20], u_x[::20,::20], u_y[::20,::20],
          color='white', alpha=0.6, scale=20, label='Flow')
# Ray paths
for i in range(ray_count):
    ax.plot(rays_x[:,i], rays_y[:,i], 'w-', lw=0.8, alpha=0.8)

ax.set_xlim(-5,5)
ax.set_ylim(-5,5)
ax.set_title("PM Ray-tracing with Skeleton & Flow Overlays")
ax.legend()
plt.show()


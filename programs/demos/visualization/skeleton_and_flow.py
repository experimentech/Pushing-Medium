import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, sobel
from skimage.feature import hessian_matrix, hessian_matrix_eigvals

# -------------------------
# 1. Build a sample refractive index field n(x,y)
# -------------------------
nx, ny = 200, 200
x = np.linspace(-5, 5, nx)
y = np.linspace(-5, 5, ny)
X, Y = np.meshgrid(x, y)

# Example: two Gaussian wells + a ridge
n0 = 1.0
n_field = n0 + 0.2*np.exp(-((X+2)**2 + (Y)**2)) \
              + 0.15*np.exp(-((X-2)**2 + (Y)**2)) \
              + 0.05*np.exp(-(X**2 + (Y-3)**2)/0.5)

# Smooth for cleaner Hessian behaviour
n_field = gaussian_filter(n_field, sigma=1.0)

# -------------------------
# 2. Skeleton extraction via Hessian ridge detection
# -------------------------
# Compute Hessian components
Hxx, Hxy, Hyy = hessian_matrix(n_field, sigma=1.0, order='rc')
# Eigenvalues of Hessian
l1, l2 = hessian_matrix_eigvals(Hxx, Hxy, Hyy)

# Ridge criterion: one large negative eigenvalue (ridge direction)
ridge_mask = (l1 < 0) & (np.abs(l1) > np.abs(l2))

# -------------------------
# 3. Define a flow field u_g(x,y)
# -------------------------
# Example: a rotating vortex-like flow
u_x = -Y / (X**2 + Y**2 + 1)
u_y =  X / (X**2 + Y**2 + 1)

# -------------------------
# 4. Flow map integration (Euler stepper for demo)
# -------------------------
def integrate_flow(x0, y0, u_x, u_y, X, Y, dt=0.1, steps=50):
    """Integrate flow from initial points (x0,y0) over given steps."""
    xf, yf = np.copy(x0), np.copy(y0)
    for _ in range(steps):
        # Interpolate velocities
        ux = np.interp(xf, X[0], u_x[int(ny/2)])  # crude 1D interp for demo
        uy = np.interp(yf, Y[:,0], u_y[:,int(nx/2)])
        xf += ux * dt
        yf += uy * dt
    return xf, yf

# Seed grid for flow map
seed_x, seed_y = np.meshgrid(np.linspace(-4, 4, 20),
                             np.linspace(-4, 4, 20))
flow_x, flow_y = integrate_flow(seed_x, seed_y, u_x, u_y, X, Y)

# -------------------------
# 5. Plotting
# -------------------------
fig, ax = plt.subplots(figsize=(7,7))
# Background: refractive index field
im = ax.imshow(n_field, extent=[x.min(), x.max(), y.min(), y.max()],
               origin='lower', cmap='viridis')

# Skeleton overlay
skel_y, skel_x = np.where(ridge_mask)
ax.plot(x[skel_x], y[skel_y], 'r.', ms=1, label='Skeleton points')

# Flow map overlay (arrows from seeds to final positions)
for i in range(seed_x.shape[0]):
    for j in range(seed_x.shape[1]):
        ax.arrow(seed_x[i,j], seed_y[i,j],
                 flow_x[i,j]-seed_x[i,j],
                 flow_y[i,j]-seed_y[i,j],
                 head_width=0.1, head_length=0.1, fc='white', ec='white', alpha=0.6)

ax.set_title("Skeleton (red) over refractive index with flow map arrows")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend()
fig.colorbar(im, ax=ax, label="n(x,y)")
plt.show()


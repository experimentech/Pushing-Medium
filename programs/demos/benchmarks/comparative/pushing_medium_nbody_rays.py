"""
pushing_medium_nbody_rays.py

Lightweight N-body + ray tracer demo in the pushing-medium model:
- A few massive bodies move under Newtonian gravity (leapfrog integration)
- A bundle of light rays bends in real time via Fermat's principle in n(r)
- Plain background for speed

Requires: numpy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ----------------------------
# Parameters
# ----------------------------
G = 1.0          # gravitational constant (scaled units)
c = 10.0         # "speed of light" in these units
eps_soft = 0.1   # softening length to avoid singularities
dt = 0.02        # timestep
n_bodies = 5
n_rays = 15
domain = [-5, 5, -5, 5]

# ----------------------------
# Initial conditions
# ----------------------------
rng = np.random.default_rng(42)
masses = rng.uniform(0.5, 1.5, n_bodies)
pos = rng.uniform(-3, 3, (n_bodies, 2))
vel = rng.uniform(-0.5, 0.5, (n_bodies, 2))

# Rays: start at left edge, evenly spaced in y
ray_pos = np.zeros((n_rays, 2))
ray_vel = np.zeros((n_rays, 2))
ray_pos[:, 0] = domain[0]
ray_pos[:, 1] = np.linspace(domain[2], domain[3], n_rays)
ray_vel[:, 0] = 1.0  # unit x-direction

# ----------------------------
# Physics helpers
# ----------------------------
def accel_bodies(pos, masses):
    """Newtonian acceleration on each body from all others."""
    acc = np.zeros_like(pos)
    for i in range(len(masses)):
        diff = pos - pos[i]
        r2 = np.sum(diff**2, axis=1) + eps_soft**2
        inv_r3 = np.where(r2 > 0, 1.0 / np.sqrt(r2**3), 0.0)
        # Sum over all j != i
        acc[i] = np.sum((G * masses[:, None]) * diff * inv_r3[:, None], axis=0) - \
                 (G * masses[i] * diff[i] * inv_r3[i])  # remove self-term
    return acc

def n_and_grad(r_point, masses, pos):
    """
    Compute n(r) and grad n for pushing-medium model:
    n = 1 + sum_i (2GM / (c^2 * r_i))
    """
    diff = pos - r_point
    r2 = np.sum(diff**2, axis=1) + eps_soft**2
    r = np.sqrt(r2)
    n_val = 1.0 + np.sum(2*G*masses / (c**2 * r))
    # grad n = - sum_i (2GM / (c^2 * r^3)) * (r_vec)
    grad = -np.sum(((2*G*masses / (c**2 * r2**1.5))[:, None]) * diff, axis=0)
    return n_val, grad

def step_ray(r, k_hat, masses, pos, ds):
    """One Fermat step for a ray in isotropic medium."""
    n_val, grad_n = n_and_grad(r, masses, pos)
    gn_par = np.dot(k_hat, grad_n) * k_hat
    dk = (grad_n - gn_par) * (ds / max(n_val, 1e-12))
    k_new = k_hat + dk
    k_new /= np.linalg.norm(k_new)
    r_new = r + k_new * ds
    return r_new, k_new

# ----------------------------
# Integrator setup
# ----------------------------
# Leapfrog: start with half-step velocity update
vel += 0.5 * dt * accel_bodies(pos, masses)

# ----------------------------
# Plot setup
# ----------------------------
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(domain[0], domain[1])
ax.set_ylim(domain[2], domain[3])
ax.set_aspect('equal')
ax.set_title("Pushing-medium N-body + ray tracer")

body_scat = ax.scatter(pos[:, 0], pos[:, 1], s=80, c='orange', edgecolor='k', zorder=3)
ray_lines = [ax.plot([], [], lw=1, color='cyan')[0] for _ in range(n_rays)]

# Store ray trails
ray_trails = [ [ray_pos[i].copy()] for i in range(n_rays) ]
ray_dirs = [ray_vel[i] / np.linalg.norm(ray_vel[i]) for i in range(n_rays)]

# ----------------------------
# Animation update
# ----------------------------
def update(frame):
    global pos, vel, ray_pos, ray_dirs

    # --- Update bodies ---
    acc = accel_bodies(pos, masses)
    vel += dt * acc
    pos += dt * vel

    # --- Update rays ---
    for i in range(n_rays):
        r, k = ray_pos[i], ray_dirs[i]
        r_new, k_new = step_ray(r, k, masses, pos, ds=0.05)
        ray_pos[i] = r_new
        ray_dirs[i] = k_new
        ray_trails[i].append(r_new.copy())
        # Reset ray if it leaves domain
        if not (domain[0] <= r_new[0] <= domain[1] and domain[2] <= r_new[1] <= domain[3]):
            ray_pos[i] = np.array([domain[0], np.linspace(domain[2], domain[3], n_rays)[i]])
            ray_dirs[i] = np.array([1.0, 0.0])
            ray_trails[i] = [ray_pos[i].copy()]

    # --- Update plot ---
    body_scat.set_offsets(pos)
    for line, trail in zip(ray_lines, ray_trails):
        arr = np.array(trail)
        line.set_data(arr[:, 0], arr[:, 1])

    return [body_scat] + ray_lines

ani = FuncAnimation(fig, update, frames=600, interval=20, blit=True)
plt.show()


"""
nbody_rays_side_by_side.py

Side-by-side N-body + ray tracer:
- Left: Pushing-medium model (n-field + Fermat stepper)
- Right: Weak-field GR (Newtonian potential deflection)

Requires: numpy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ----------------------------
# Parameters
# ----------------------------
G = 1.0
c = 10.0
eps_soft = 0.1
dt = 0.02
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
ray_pos_PM = np.zeros((n_rays, 2))
ray_vel_PM = np.zeros((n_rays, 2))
ray_pos_GR = np.zeros((n_rays, 2))
ray_vel_GR = np.zeros((n_rays, 2))
y_starts = np.linspace(domain[2], domain[3], n_rays)
ray_pos_PM[:, 0] = domain[0]
ray_pos_PM[:, 1] = y_starts
ray_vel_PM[:, 0] = 1.0
ray_pos_GR[:] = ray_pos_PM
ray_vel_GR[:, 0] = 1.0

# ----------------------------
# Physics helpers
# ----------------------------
def accel_bodies(pos, masses):
    acc = np.zeros_like(pos)
    for i in range(len(masses)):
        diff = pos - pos[i]
        r2 = np.sum(diff**2, axis=1) + eps_soft**2
        inv_r3 = np.where(r2 > 0, 1.0 / np.sqrt(r2**3), 0.0)
        acc[i] = np.sum((G * masses[:, None]) * diff * inv_r3[:, None], axis=0) - \
                 (G * masses[i] * diff[i] * inv_r3[i])
    return acc

def n_and_grad(r_point, masses, pos):
    diff = pos - r_point
    r2 = np.sum(diff**2, axis=1) + eps_soft**2
    r = np.sqrt(r2)
    n_val = 1.0 + np.sum(2*G*masses / (c**2 * r))
    grad = -np.sum(((2*G*masses / (c**2 * r2**1.5))[:, None]) * diff, axis=0)
    return n_val, grad

def step_ray_PM(r, k_hat, masses, pos, ds):
    n_val, grad_n = n_and_grad(r, masses, pos)
    gn_par = np.dot(k_hat, grad_n) * k_hat
    dk = (grad_n - gn_par) * (ds / max(n_val, 1e-12))
    k_new = k_hat + dk
    k_new /= np.linalg.norm(k_new)
    r_new = r + k_new * ds
    return r_new, k_new

def step_ray_GR(r, v, masses, pos, ds):
    # Weak-field: transverse acceleration from Newtonian potential
    acc = np.zeros(2)
    for m, p in zip(masses, pos):
        diff = p - r
        r2 = np.sum(diff**2) + eps_soft**2
        rmag = np.sqrt(r2)
        # Only transverse component affects light path
        acc += (4*G*m / (c**2 * r2)) * (diff / rmag)
    v_new = v + acc * ds
    v_new /= np.linalg.norm(v_new)
    r_new = r + v_new * ds
    return r_new, v_new

# ----------------------------
# Integrator setup
# ----------------------------
vel += 0.5 * dt * accel_bodies(pos, masses)

# ----------------------------
# Plot setup
# ----------------------------
fig, (axL, axR) = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
for ax in (axL, axR):
    ax.set_xlim(domain[0], domain[1])
    ax.set_ylim(domain[2], domain[3])
    ax.set_aspect('equal')

axL.set_title("Pushing-medium")
axR.set_title("Weak-field GR")

body_scat_L = axL.scatter(pos[:, 0], pos[:, 1], s=80, c='orange', edgecolor='k', zorder=3)
body_scat_R = axR.scatter(pos[:, 0], pos[:, 1], s=80, c='orange', edgecolor='k', zorder=3)

ray_lines_PM = [axL.plot([], [], lw=1, color='cyan')[0] for _ in range(n_rays)]
ray_lines_GR = [axR.plot([], [], lw=1, color='lime')[0] for _ in range(n_rays)]

ray_trails_PM = [[ray_pos_PM[i].copy()] for i in range(n_rays)]
ray_dirs_PM = [ray_vel_PM[i] / np.linalg.norm(ray_vel_PM[i]) for i in range(n_rays)]
ray_trails_GR = [[ray_pos_GR[i].copy()] for i in range(n_rays)]
ray_dirs_GR = [ray_vel_GR[i] / np.linalg.norm(ray_vel_GR[i]) for i in range(n_rays)]

# ----------------------------
# Animation update
# ----------------------------
def update(frame):
    global pos, vel, ray_pos_PM, ray_dirs_PM, ray_pos_GR, ray_dirs_GR

    # Update bodies
    acc = accel_bodies(pos, masses)
    vel += dt * acc
    pos += dt * vel

    # Update rays: PM
    for i in range(n_rays):
        r, k = ray_pos_PM[i], ray_dirs_PM[i]
        r_new, k_new = step_ray_PM(r, k, masses, pos, ds=0.05)
        ray_pos_PM[i] = r_new
        ray_dirs_PM[i] = k_new
        ray_trails_PM[i].append(r_new.copy())
        if not (domain[0] <= r_new[0] <= domain[1] and domain[2] <= r_new[1] <= domain[3]):
            ray_pos_PM[i] = np.array([domain[0], y_starts[i]])
            ray_dirs_PM[i] = np.array([1.0, 0.0])
            ray_trails_PM[i] = [ray_pos_PM[i].copy()]

    # Update rays: GR
    for i in range(n_rays):
        r, v = ray_pos_GR[i], ray_dirs_GR[i]
        r_new, v_new = step_ray_GR(r, v, masses, pos, ds=0.05)
        ray_pos_GR[i] = r_new
        ray_dirs_GR[i] = v_new
        ray_trails_GR[i].append(r_new.copy())
        if not (domain[0] <= r_new[0] <= domain[1] and domain[2] <= r_new[1] <= domain[3]):
            ray_pos_GR[i] = np.array([domain[0], y_starts[i]])
            ray_dirs_GR[i] = np.array([1.0, 0.0])
            ray_trails_GR[i] = [ray_pos_GR[i].copy()]

    # Update plots
    body_scat_L.set_offsets(pos)
    body_scat_R.set_offsets(pos)
    for line, trail in zip(ray_lines_PM, ray_trails_PM):
        arr = np.array(trail)
        line.set_data(arr[:, 0], arr[:, 1])
    for line, trail in zip(ray_lines_GR, ray_trails_GR):
        arr = np.array(trail)
        line.set_data(arr[:, 0], arr[:, 1])

    return [body_scat_L, body_scat_R] + ray_lines_PM + ray_lines_GR

ani = FuncAnimation(fig, update, frames=600, interval=20, blit=True)
plt.show()


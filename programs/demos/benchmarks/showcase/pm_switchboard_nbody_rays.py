"""
pm_switchboard_nbody_rays.py

Pushing-medium N-body + ray tracer with live "physics switchboard":
- Toggle static index field n(x,y)
- Toggle flow field u_g(x,y) (frame-drag analogue; simple swirl model)
- Toggle TT wave perturbation (small time-dependent index ripple)

Visuals:
- Orange circles: massive bodies (mutual gravity, leapfrog integration)
- White lines: light rays (Fermat ray-marching in n)
- Status HUD: shows which effects are currently enabled

Notes on models (kept simple for real-time clarity):
- Static n field: n = 1 + sum_i 2GM_i/(c^2 r_i), grad n used for ray bending.
- Flow field u_g: a small, divergence-free swirl around masses ~ (ẑ × r)/(|r|^2 + eps^2);
  used to advect rays in addition to geometric step to mimic frame-drag-like advection.
- TT wave: small sinusoidal perturbation δn_tt(x,y,t) added to n; contributes to grad n.

Requires: numpy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['keymap.fullscreen'] = []  # disable fullscreen shortcut
from matplotlib.animation import FuncAnimation

# ----------------------------
# Parameters
# ----------------------------
G = 1.0            # gravitational constant (scaled units)
c = 10.0           # "speed of light"
eps_soft = 0.12    # softening to avoid singularities
dt = 0.02          # N-body integrator timestep
ds_ray = 0.05      # ray arc-length step
n_bodies = 5
n_rays = 24
domain = [-5, 5, -5, 5]

# Flow model parameters (frame-drag analogue)
flow_strength = 0.4   # scales u_g magnitude
flow_dt_scale = 0.6   # how strongly u_g advects rays per frame (units of dt)

# TT wave parameters (small index ripple)
tt_amp = 0.015        # amplitude of δn_tt
tt_k = np.array([0.8, 0.5])  # wavevector (rad/unit)
tt_omega = 0.8        # angular frequency
tt_phase0 = 0.0

# Rendering / trail
trail_maxlen = 200    # limit trail length per ray (keeps draw cost bounded)

# ----------------------------
# Initial conditions
# ----------------------------
rng = np.random.default_rng(42)
masses = rng.uniform(0.6, 1.4, n_bodies)
pos = rng.uniform(-3, 3, (n_bodies, 2))
vel = rng.uniform(-0.35, 0.35, (n_bodies, 2))

# Rays: start at left edge, evenly spaced in y
y_starts = np.linspace(domain[2], domain[3], n_rays)
ray_pos = np.column_stack([np.full(n_rays, domain[0]), y_starts])
ray_dirs = np.tile(np.array([1.0, 0.0]), (n_rays, 1))
ray_trails = [[ray_pos[i].copy()] for i in range(n_rays)]

# ----------------------------
# Toggles (keyboard-controlled)
# ----------------------------
use_n = True         # static index field
use_flow = True      # flow advection
use_tt = True        # TT wave perturbation

# ----------------------------
# Physics helpers
# ----------------------------
def accel_bodies(pos, masses):
    """Newtonian acceleration on each body from all others (softened)."""
    N = len(masses)
    acc = np.zeros_like(pos)
    for i in range(N):
        d = pos - pos[i]                         # (N,2)
        r2 = np.sum(d**2, axis=1) + eps_soft**2
        inv_r3 = np.where(r2 > 0, 1.0 / np.sqrt(r2**3), 0.0)
        acc[i] = np.sum((G * masses[:, None]) * d * inv_r3[:, None], axis=0) \
                 - (G * masses[i] * d[i] * inv_r3[i])  # remove self
    return acc

def n_and_grad_static(point, masses, pos):
    """
    Static pushing-medium index field and gradient at a point:
      n = 1 + sum_i (2 G M_i / (c^2 r_i))
      ∇n = - sum_i (2 G M_i / (c^2 r_i^3)) * (r_vec)
    """
    d = pos - point                               # (N,2)
    r2 = np.sum(d**2, axis=1) + eps_soft**2
    r = np.sqrt(r2)
    n_val = 1.0 + np.sum(2 * G * masses / (c**2 * r))
    grad = -np.sum(((2 * G * masses / (c**2 * r2**1.5))[:, None]) * d, axis=0)
    return n_val, grad

def ug_flow(point, masses, pos):
    """
    Simple divergence-free swirl flow around each mass (2D analogue of frame-drag):
      u_g ~ flow_strength * (ẑ × r_vec) / (|r|^2 + eps^2)
    Summed over masses. Units are arbitrary but small relative to c.
    """
    d = point - pos                                # (N,2) vector from mass to point
    r2 = np.sum(d**2, axis=1) + eps_soft**2
    # ẑ × r = (-dy, dx)
    swirl = np.column_stack([-d[:,1], d[:,0]]) / r2[:, None]
    u = flow_strength * np.sum((masses[:, None]) * swirl, axis=0)
    return u

def tt_delta_n_and_grad(point, t):
    """
    TT 'wave' as a small sinusoidal index perturbation:
      δn_tt = A sin(k⋅x - ωt + φ0)
      ∇δn_tt = A cos(k⋅x - ωt + φ0) * k
    Lightweight and purely demonstrative (kept small).
    """
    phase = tt_k @ point + (-tt_omega * t + tt_phase0)
    dn = tt_amp * np.sin(phase)
    grad = tt_amp * np.cos(phase) * tt_k
    return dn, grad

def step_ray(r, k_hat, masses, pos, t, ds, use_n=True, use_flow=True, use_tt=True):
    """
    One ray step combining:
      - Fermat bending from grad n (static + TT contribution).
      - Optional advection by u_g (frame-drag analogue).
    """
    # Accumulate n and grad n as needed
    n_val = 1.0
    grad_n = np.zeros(2)

    if use_n:
        n_s, g_s = n_and_grad_static(r, masses, pos)
        n_val += (n_s - 1.0)
        grad_n += g_s

    if use_tt:
        dn_tt, g_tt = tt_delta_n_and_grad(r, t)
        n_val += dn_tt
        grad_n += g_tt

    # Fermat direction update: bend by component of grad n perpendicular to k
    # dk = (I - k k^T) grad n * (ds / n)
    gn_par = np.dot(k_hat, grad_n) * k_hat
    dk = (grad_n - gn_par) * (ds / max(n_val, 1e-12))
    k_new = k_hat + dk
    k_new /= np.linalg.norm(k_new)

    # Baseline geometric advance
    r_new = r + k_new * ds

    # Optional flow advection (small displacement per frame)
    if use_flow:
        u = ug_flow(r, masses, pos)
        r_new = r_new + u * (flow_dt_scale * dt)

    return r_new, k_new

# ----------------------------
# Integrator setup (leapfrog kick)
# ----------------------------
vel += 0.5 * dt * accel_bodies(pos, masses)
sim_time = 0.0

# ----------------------------
# Plot setup
# ----------------------------
fig, ax = plt.subplots(figsize=(7, 7))
ax.set_xlim(domain[0], domain[1])
ax.set_ylim(domain[2], domain[3])
ax.set_aspect('equal')
ax.set_title("Pushing-medium N-body + rays — modular physics switchboard")

body_scat = ax.scatter(pos[:, 0], pos[:, 1], s=80, c='orange', edgecolor='k', zorder=3)
ray_lines = [ax.plot([], [], lw=1.1, color='white', alpha=0.95, zorder=2)[0] for _ in range(n_rays)]

# HUD for toggle status
status_txt = ax.text(
    0.02, 0.98,
    "",
    transform=ax.transAxes,
    va='top', ha='left',
    fontsize=10, color='w',
    bbox=dict(boxstyle="round,pad=0.25", fc="black", ec="gray", alpha=0.6),
    zorder=4
)

# ----------------------------
# Utility: reset rays
# ----------------------------
def reset_rays():
    global ray_pos, ray_dirs, ray_trails
    ray_pos = np.column_stack([np.full(n_rays, domain[0]), y_starts])
    ray_dirs = np.tile(np.array([1.0, 0.0]), (n_rays, 1))
    ray_trails = [[ray_pos[i].copy()] for i in range(n_rays)]

# ----------------------------
# Keyboard controls
# ----------------------------
def on_key(event):
    global use_n, use_flow, use_tt
    if event.key in ('q', 'escape'):
        plt.close(event.canvas.figure)
        return
    if event.key == 'n':
        use_n = not use_n
    elif event.key == 'f':
        use_flow = not use_flow
    elif event.key == 't':
        use_tt = not use_tt
    elif event.key == 'r':
        reset_rays()
    update_status()

def update_status():
    status = [
        f"n-field: {'ON' if use_n else 'off'}",
        f"u_g flow: {'ON' if use_flow else 'off'}",
        f"TT waves: {'ON' if use_tt else 'off'}",
        "keys: n/f/t toggle, r reset, q quit"
    ]
    status_txt.set_text("\n".join(status))
    fig.canvas.draw_idle()

fig.canvas.mpl_connect('key_press_event', on_key)
update_status()

# ----------------------------
# Animation update
# ----------------------------
def update(frame):
    global pos, vel, sim_time, ray_pos, ray_dirs, ray_trails

    # --- N-body update ---
    acc = accel_bodies(pos, masses)
    vel += dt * acc
    pos += dt * vel
    sim_time += dt

    # --- Rays update ---
    for i in range(n_rays):
        r, k = ray_pos[i], ray_dirs[i]
        r_new, k_new = step_ray(r, k, masses, pos, sim_time, ds_ray,
                                use_n=use_n, use_flow=use_flow, use_tt=use_tt)
        ray_pos[i] = r_new
        ray_dirs[i] = k_new
        ray_trails[i].append(r_new.copy())
        if len(ray_trails[i]) > trail_maxlen:
            ray_trails[i].pop(0)

        # Reset if out of bounds
        if not (domain[0] <= r_new[0] <= domain[1] and domain[2] <= r_new[1] <= domain[3]):
            ray_pos[i] = np.array([domain[0], y_starts[i]])
            ray_dirs[i] = np.array([1.0, 0.0])
            ray_trails[i] = [ray_pos[i].copy()]

    # --- Update plot ---
    body_scat.set_offsets(pos)
    for line, trail in zip(ray_lines, ray_trails):
        arr = np.asarray(trail)
        line.set_data(arr[:, 0], arr[:, 1])

    return [body_scat] + ray_lines + [status_txt]

ani = FuncAnimation(fig, update, frames=100000, interval=25, blit=True)
plt.show()


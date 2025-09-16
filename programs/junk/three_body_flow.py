#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Three-body flow topology (Sun–Earth–Moon) in a co-rotating frame.
- Builds a substrate-inspired inflow field from three sinks.
- Switches to Sun–Earth rotating frame (Omega).
- Finds & classifies stagnation points (saddle vs center/spiral).
- Plots streamlines, separatrix-like structure, and tracer advection.

Install deps: pip install numpy scipy matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.ndimage import minimum_filter

# ---------------------------
# Config and domain
# ---------------------------
# Grid size (increase for cleaner separatrices, costs time)
Nx, Ny = 700, 500
x = np.linspace(-2.0, 3.0, Nx)
y = np.linspace(-2.0, 2.0, Ny)
X, Y = np.meshgrid(x, y)

# Regularization to avoid singular cores
EPS = 1e-3

# Rotating frame angular rate (1 rev per unit time ~ 2π)
# You can tune this to see bifurcations in the stagnation set
OMEGA = 2.0 * np.pi

# Mass-like sink strengths (nondimensional)
M_SUN, M_EARTH, M_MOON = 1000.0, 1.0, 0.0123

# Positions (Sun at origin, Earth at ~1 AU, Moon slightly offset)
SUN = np.array([0.0, 0.0])
EARTH = np.array([1.0, 0.0])

def moon_pos(phase_rad=0.0, r_em=0.00257):
    # Place the Moon on a small circle around Earth (phase in radians)
    return EARTH + r_em * np.array([np.cos(phase_rad), np.sin(phase_rad)])

# ---------------------------
# Flow field definition
# ---------------------------
def add_inflow(U, V, pos, strength):
    dx = pos[0] - X
    dy = pos[1] - Y
    r2 = dx*dx + dy*dy + EPS*EPS
    inv_r32 = strength / np.power(r2, 1.5)
    U += dx * inv_r32
    V += dy * inv_r32
    return U, V

def flow_field(omega, moon_xy):
    # Superposed radial inflows (substrate-inspired) + co-rotating frame correction
    U = np.zeros_like(X)
    V = np.zeros_like(Y)
    U, V = add_inflow(U, V, SUN,   M_SUN)
    U, V = add_inflow(U, V, EARTH, M_EARTH)
    U, V = add_inflow(U, V, moon_xy, M_MOON)
    # Rotating frame: u_Omega = u - Omega x r = (U + omega*Y, V - omega*X)
    Uo = U + omega * Y
    Vo = V - omega * X
    return Uo, Vo

def jacobian(Uo, Vo, dx, dy):
    # Partial derivatives for classification
    Ux = np.gradient(Uo, dx, axis=1)
    Uy = np.gradient(Uo, dy, axis=0)
    Vx = np.gradient(Vo, dx, axis=1)
    Vy = np.gradient(Vo, dy, axis=0)
    return Ux, Uy, Vx, Vy

def find_stagnations(Uo, Vo, speed_thresh=3e-3, sep=9):
    S = np.hypot(Uo, Vo)
    # Local minima in speed below threshold (simple, robust)
    local_min = (S == minimum_filter(S, size=sep))
    cand = np.argwhere((S < speed_thresh) & local_min)
    return cand, S

def classify_points(cand, Ux, Uy, Vx, Vy):
    labels = []
    for (iy, ix) in cand:
        J = np.array([[Ux[iy, ix], Uy[iy, ix]],
                      [Vx[iy, ix], Vy[iy, ix]]], dtype=float)
        eig = np.linalg.eigvals(J)
        if np.all(np.isreal(eig)):
            lbl = 'saddle' if eig[0]*eig[1] < 0 else 'node'
        else:
            tr = np.trace(J)
            lbl = 'center' if np.abs(tr) < 1e-4 else ('spiral+' if tr > 0 else 'spiral-')
        labels.append((iy, ix, lbl))
    return labels

# ---------------------------
# Tracer advection (RK4) in the rotating frame
# ---------------------------
def make_vector_field_sampler(omega, moon_xy):
    # Cache a field for fast bilinear sampling
    Uo, Vo = flow_field(omega, moon_xy)
    def sample(xp, yp):
        # Bilinear interpolation on grid; clamp to domain
        if xp <= x[0] or xp >= x[-1] or yp <= y[0] or yp >= y[-1]:
            return 0.0, 0.0
        i = np.searchsorted(x, xp) - 1
        j = np.searchsorted(y, yp) - 1
        i = np.clip(i, 0, Nx-2)
        j = np.clip(j, 0, Ny-2)
        x1, x2 = x[i], x[i+1]
        y1, y2 = y[j], y[j+1]
        tx = (xp - x1) / (x2 - x1 + 1e-16)
        ty = (yp - y1) / (y2 - y1 + 1e-16)
        # corners
        U00, U10 = Uo[j, i],   Uo[j, i+1]
        U01, U11 = Uo[j+1, i], Uo[j+1, i+1]
        V00, V10 = Vo[j, i],   Vo[j, i+1]
        V01, V11 = Vo[j+1, i], Vo[j+1, i+1]
        U = (1-tx)*(1-ty)*U00 + tx*(1-ty)*U10 + (1-tx)*ty*U01 + tx*ty*U11
        V = (1-tx)*(1-ty)*V00 + tx*(1-ty)*V10 + (1-tx)*ty*V01 + tx*ty*V11
        return float(U), float(V)
    return sample, Uo, Vo

def rk4_advect(x0, y0, sampler, dt=0.002, steps=20000):
    xs = np.empty(steps); ys = np.empty(steps)
    xs[0], ys[0] = x0, y0
    for k in range(1, steps):
        xi, yi = xs[k-1], ys[k-1]
        k1x, k1y = sampler(xi, yi)
        k2x, k2y = sampler(xi + 0.5*dt*k1x, yi + 0.5*dt*k1y)
        k3x, k3y = sampler(xi + 0.5*dt*k2x, yi + 0.5*dt*k2y)
        k4x, k4y = sampler(xi + dt*k3x, yi + dt*k3y)
        xs[k] = xi + (dt/6.0)*(k1x + 2*k2x + 2*k3x + k4x)
        ys[k] = yi + (dt/6.0)*(k1y + 2*k2y + 2*k3y + k4y)
        # stop if leaving domain
        if not (x[0] < xs[k] < x[-1] and y[0] < ys[k] < y[-1]):
            xs = xs[:k]; ys = ys[:k]
            break
    return xs, ys

# ---------------------------
# Plotting helpers
# ---------------------------
def plot_all(moon_xy, labels, Uo, Vo, S, tracer=None):
    fig, ax = plt.subplots(1, 1, figsize=(10, 7), constrained_layout=True)
    ax.set_title("Three-body flow topology in Sun–Earth rotating frame")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_aspect('equal')
    # Background: speed (log-ish stretch)
    Sm = np.log10(1e-6 + S)
    cs = ax.imshow(Sm, extent=[x[0], x[-1], y[0], y[-1]],
                   origin='lower', cmap='magma', alpha=0.85)
    cbar = fig.colorbar(cs, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("log10 speed")

    # Streamlines
    skip = 2  # subsample for speed
    ax.streamplot(x[::skip], y[::skip], Uo[::skip, ::skip], Vo[::skip, ::skip],
                  color='white', density=1.2, linewidth=0.6, arrowsize=0.6)

    # Bodies
    ax.scatter([SUN[0], EARTH[0], moon_xy[0]],
               [SUN[1], EARTH[1], moon_xy[1]],
               c=['gold', 'deepskyblue', 'silver'], s=[80, 40, 20], zorder=5, edgecolor='k', linewidth=0.5)
    ax.add_patch(Circle(SUN, 0.04, fc='gold', ec='k', lw=0.5, alpha=0.4))
    ax.add_patch(Circle(EARTH, 0.02, fc='deepskyblue', ec='k', lw=0.5, alpha=0.4))
    ax.add_patch(Circle(moon_xy, 0.008, fc='silver', ec='k', lw=0.5, alpha=0.4))

    # Stagnation points
    for (iy, ix, lbl) in labels:
        px, py = x[ix], y[iy]
        if lbl == 'saddle':
            ax.scatter(px, py, marker='x', c='yellow', s=60, zorder=6, label='saddle' if 'saddle' not in [l.get_text() for l in ax.legend_.texts] if ax.legend_ else True)
        elif lbl.startswith('spiral'):
            ax.scatter(px, py, marker='o', facecolors='none', edgecolors='cyan', s=60, zorder=6)
        elif lbl == 'center':
            ax.scatter(px, py, marker='o', c='cyan', s=30, zorder=6)
        else:
            ax.scatter(px, py, marker='.', c='lime', s=20, zorder=6)

    # Tracer path
    if tracer is not None:
        xs, ys = tracer
        ax.plot(xs, ys, c='white', lw=1.2, alpha=0.9, label='tracer')

    ax.set_xlim(x[0], x[-1]); ax.set_ylim(y[0], y[-1])
    # Legend (build manually to avoid duplications)
    handles = [
        plt.Line2D([], [], marker='x', ls='', c='yellow', label='saddle'),
        plt.Line2D([], [], marker='o', ls='', c='cyan', label='center/spiral'),
        plt.Line2D([], [], c='white', lw=1.2, label='tracer'),
    ]
    ax.legend(handles=handles, loc='upper right')
    plt.show()

# ---------------------------
# Main: compute, classify, advect
# ---------------------------
if __name__ == "__main__":
    phase = np.deg2rad(30.0)  # try 0, 30, 60, ... to see phase effects
    MOON = moon_pos(phase_rad=phase)

    # Build field and Jacobian
    Uo, Vo = flow_field(OMEGA, MOON)
    dx, dy = x[1] - x[0], y[1] - y[0]
    Ux, Uy, Vx, Vy = jacobian(Uo, Vo, dx, dy)

    # Find and classify stagnation points
    cand, S = find_stagnations(Uo, Vo, speed_thresh=3e-3, sep=9)
    labels = classify_points(cand, Ux, Uy, Vx, Vy)

    # Pick a center-like point (triangular region) for tracer seeding
    center_like = [(ix, iy) for (iy, ix, lbl) in labels if lbl in ('center', 'spiral-', 'spiral+')]
    tracer = None
    if center_like:
        # Seed a tracer slightly offset from the first center-like point
        ix, iy = center_like[0]
        x0, y0 = x[ix] + 0.02, y[iy]
        sampler, Uc, Vc = make_vector_field_sampler(OMEGA, MOON)
        xs, ys = rk4_advect(x0, y0, sampler, dt=0.002, steps=20000)
        tracer = (xs, ys)

    # Plot everything
    plot_all(MOON, labels, Uo, Vo, S, tracer=tracer)

    # Notes:
    # - Increase Nx, Ny for sharper separatrices.
    # - Tighten speed_thresh as you refine the grid.
    # - Sweep 'phase' to watch the stagnation set move over a synodic-like cycle.
    # - To stress-test, vary OMEGA, M_MOON, and EPS; watch for saddle-center pair creation/annihilation (bifurcations).


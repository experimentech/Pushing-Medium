"""Bullet Cluster: GR vs Pushing-Medium (flow) comparison demo (analytic toy setup).

This script builds a toy "Bullet Cluster" lens made of two Plummer-profile clusters in the
lens plane and compares:
  - Standard GR thin-lens deflection (weak field, spherical Plummer lens)
  - A Pushing-Medium (PM) moving-lens analogue where the subcluster carries a transverse flow

Key improvements over the initial version:
  - Uses analytic Plummer surface density and deflection instead of noisy mass sampling
  - Calibrates the PM moving-lens correction by calling `moving_lens_deflection_numeric`
    on a grid of impact parameters and interpolates the ratio at runtime
  - Produces more informative figures: column density, GR and PM deflection magnitudes, and
    a fractional difference map with streamlines overlaid

This remains an illustrative demo; to reproduce observational data exactly, replace the
Plummer parameters with measured mass maps.

Usage:
    python programs/demos/benchmarks/bullet_cluster_comparison.py

Outputs:
    programs/demos/benchmarks/out/bullet_cluster_comparison.png

If required packages are missing the script prints installation guidance.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

# Ensure repo src/ is importable
repo_root = Path(__file__).resolve().parents[3]
src_path = str(repo_root / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

try:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")  # render to file without X server
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
except Exception:
    print("Missing Python scientific packages. Please install into your project venv:\n")
    print("  .venv/bin/pip install numpy matplotlib")
    raise

try:
    from pushing_medium import core as pm
except Exception as exc:
    raise RuntimeError("Could not import pushing_medium.core; ensure src/ is on PYTHONPATH") from exc

G = pm.G
c = pm.c

kpc = 3.085677581e19  # metres
M_SUN = 1.98847e30
ARCSEC_PER_RAD = (180.0 / math.pi) * 3600.0


def plummer_surface_density(
    mass: float,
    softening: float,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    x0: float,
    y0: float,
) -> np.ndarray:
    """Analytic projected surface density Sigma(R) for a Plummer sphere."""
    dx = x_grid - x0
    dy = y_grid - y0
    r2 = dx * dx + dy * dy
    return (mass * softening * softening) / (math.pi * (r2 + softening * softening) ** 2)


def plummer_deflection(
    mass: float,
    softening: float,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    x0: float,
    y0: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (alpha_x, alpha_y) for a Plummer lens in the weak-field limit."""
    dx = x_grid - x0
    dy = y_grid - y0
    r2 = dx * dx + dy * dy
    pref = 4.0 * G * mass / (c * c)
    denom = r2 + softening * softening
    alpha_x = pref * dx / denom
    alpha_y = pref * dy / denom
    return alpha_x, alpha_y


def calibrate_pm_ratio(
    mass: float,
    softening: float,
    mu_coeff: float,
    v_transverse: float,
    b_min: float,
    b_max: float,
    samples: int = 160,
    z_max_factor: float = 6.0,
    steps: int = 1200,
) -> tuple[np.ndarray, np.ndarray]:
    """Tabulate alpha_PM / alpha_static for a moving Plummer-like lens."""
    if v_transverse == 0.0:
        return np.array([b_min, b_max]), np.ones(2)

    impact_params = np.linspace(b_min, b_max, samples)
    ratios = np.ones_like(impact_params)
    z_max = z_max_factor * softening

    for idx, b in enumerate(impact_params):
        static_val = pm.fermat_deflection_static_index(mass, float(b), mu=mu_coeff, z_max=z_max, steps=steps)
        moving_val = pm.moving_lens_deflection_numeric(
            mass,
            float(b),
            mu=mu_coeff,
            v_transverse=v_transverse,
            z_max=z_max,
            steps=steps,
        )
        ratios[idx] = moving_val / static_val if static_val > 0 else 1.0
    return impact_params, ratios


def main() -> None:
    # Cluster setup (toy values only)
    mass_main = 1.0e15 * M_SUN
    mass_sub = 3.0e14 * M_SUN
    soft_main = 150.0 * kpc
    soft_sub = 80.0 * kpc
    separation = 800.0 * kpc

    # Field grid (physical metres, but treated as projected coordinates)
    nx = ny = 220
    field_size = 2200.0 * kpc
    xs = np.linspace(-field_size / 2.0, field_size / 2.0, nx)
    ys = np.linspace(-field_size / 2.0, field_size / 2.0, ny)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")

    # Cluster centres
    x_main, y_main = -separation / 2.0, 0.0
    x_sub, y_sub = +separation / 2.0, 0.0

    # Surface density maps (kg m^-2)
    sigma_main = plummer_surface_density(mass_main, soft_main, xx, yy, x_main, y_main)
    sigma_sub = plummer_surface_density(mass_sub, soft_sub, xx, yy, x_sub, y_sub)
    sigma_total = sigma_main + sigma_sub

    # GR deflection field (radians)
    ax_main_gr, ay_main_gr = plummer_deflection(mass_main, soft_main, xx, yy, x_main, y_main)
    ax_sub_gr, ay_sub_gr = plummer_deflection(mass_sub, soft_sub, xx, yy, x_sub, y_sub)
    alpha_x_gr = ax_main_gr + ax_sub_gr
    alpha_y_gr = ay_main_gr + ay_sub_gr

    # PM correction for moving subcluster
    v_sub = 3000.0 * 1000.0  # 3000 km/s
    mu_coeff = 2.0 * G / (c * c)
    r_sub = np.hypot(xx - x_sub, yy - y_sub)
    nonzero = r_sub[r_sub > 0.0]
    b_min = float(np.min(nonzero)) if nonzero.size else 1.0 * kpc
    b_max = float(np.max(r_sub)) if nonzero.size else 2000.0 * kpc
    b_samples, ratio_samples = calibrate_pm_ratio(mass_sub, soft_sub, mu_coeff, v_sub, b_min, b_max)
    ratio_grid = np.interp(
        r_sub.ravel(),
        b_samples,
        ratio_samples,
        left=ratio_samples[0],
        right=ratio_samples[-1],
    ).reshape(r_sub.shape)

    alpha_x_pm = ax_main_gr + ax_sub_gr * ratio_grid
    alpha_y_pm = ay_main_gr + ay_sub_gr * ratio_grid

    # Diagnostics
    mag_gr = np.hypot(alpha_x_gr, alpha_y_gr)
    mag_pm = np.hypot(alpha_x_pm, alpha_y_pm)
    mag_diff = mag_pm - mag_gr
    mag_frac = np.where(mag_gr > 0.0, mag_diff / mag_gr, 0.0)

    mag_gr_arcsec = mag_gr * ARCSEC_PER_RAD
    mag_pm_arcsec = mag_pm * ARCSEC_PER_RAD
    mag_diff_mas = mag_diff * ARCSEC_PER_RAD * 1000.0  # milli-arcseconds

    extent = [xs[0] / kpc, xs[-1] / kpc, ys[0] / kpc, ys[-1] / kpc]
    fig, axs = plt.subplots(2, 2, figsize=(13, 11))

    # Panel 1: surface density
    ax = axs[0, 0]
    im0 = ax.imshow(
        sigma_total / (M_SUN / (kpc ** 2)),
        origin="lower",
        extent=extent,
        cmap="inferno",
        norm=LogNorm(),
    )
    ax.set_title("Projected surface density Sigma (Msun / kpc^2)")
    fig.colorbar(im0, ax=ax, label="Sigma [Msun / kpc^2]")
    ax.plot([x_main / kpc], [y_main / kpc], marker="o", color="#00ffff", label="main")
    ax.plot([x_sub / kpc], [y_sub / kpc], marker="o", color="#ff00ff", label="sub")
    ax.legend(loc="upper right")

    # Panel 2: GR deflection
    ax = axs[0, 1]
    im1 = ax.imshow(
        np.maximum(mag_gr_arcsec, 1e-6),
        origin="lower",
        extent=extent,
        cmap="magma",
        norm=LogNorm(),
    )
    ax.set_title("GR deflection |alpha| (arcsec)")
    fig.colorbar(im1, ax=ax, label="|alpha| [arcsec]")
    norm_gr = mag_gr / (np.max(mag_gr) + 1e-30)
    lw_gr = 0.6 + 1.4 * norm_gr
    ax.streamplot(xx / kpc, yy / kpc, alpha_x_gr, alpha_y_gr, color="white", linewidth=lw_gr, density=1.2, arrowsize=0.8)

    # Panel 3: PM deflection
    ax = axs[1, 0]
    im2 = ax.imshow(
        np.maximum(mag_pm_arcsec, 1e-6),
        origin="lower",
        extent=extent,
        cmap="viridis",
        norm=LogNorm(),
    )
    ax.set_title("PM-flow deflection |alpha| (arcsec)")
    fig.colorbar(im2, ax=ax, label="|alpha| [arcsec]")
    norm_pm = mag_pm / (np.max(mag_pm) + 1e-30)
    lw_pm = 0.6 + 1.4 * norm_pm
    ax.streamplot(xx / kpc, yy / kpc, alpha_x_pm, alpha_y_pm, color="white", linewidth=lw_pm, density=1.2, arrowsize=0.8)

    # Panel 4: fractional difference
    ax = axs[1, 1]
    im3 = ax.imshow(mag_frac * 100.0, origin="lower", extent=extent, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    ax.set_title("Fractional change (PM / GR - 1) [%]")
    fig.colorbar(im3, ax=ax, label="Delta alpha / alpha_GR [%]")

    for axis in axs.flat:
        axis.set_xlabel("x [kpc]")
        axis.set_ylabel("y [kpc]")
        axis.set_aspect("equal", adjustable="box")

    fig.suptitle("Toy Bullet Cluster: GR vs PM moving-lens comparison", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    outdir = repo_root / "programs" / "demos" / "benchmarks" / "out"
    outdir.mkdir(parents=True, exist_ok=True)
    outfile = outdir / "bullet_cluster_comparison.png"
    plt.savefig(outfile, dpi=220)
    print(f"Wrote {outfile}")


if __name__ == "__main__":
    main()
src_path = str(repo_root / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

try:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")  # render to file without X server
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
except Exception:
    print("Missing Python scientific packages. Please install into your project venv:\n")
    print("  .venv/bin/pip install numpy matplotlib")
    raise

try:
    from pushing_medium import core as pm
except Exception:
    raise RuntimeError("Could not import pushing_medium.core; ensure src/ is on PYTHONPATH")

G = pm.G
c = pm.c

kpc = 3.085677581e19  # metres
M_SUN = 1.98847e30
ARCSEC_PER_RAD = (180.0 / math.pi) * 3600.0


def plummer_surface_density(M: float, a_soft: float, X: np.ndarray, Y: np.ndarray, centre_x: float, centre_y: float) -> np.ndarray:
    """Analytic Σ(R) for Plummer sphere projected on sky: Σ(R) = (M a^2) / [π (R^2 + a^2)^2]."""
    dx = X - centre_x
    dy = Y - centre_y
    R2 = dx * dx + dy * dy
    return (M * a_soft * a_soft) / (math.pi * (R2 + a_soft * a_soft) ** 2)


def plummer_deflection(M: float, a_soft: float, X: np.ndarray, Y: np.ndarray, centre_x: float, centre_y: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (alpha_x, alpha_y) for a spherical Plummer mass in weak-field GR."""
    dx = X - centre_x
    dy = Y - centre_y
    R2 = dx * dx + dy * dy
    pref = 4.0 * G * M / (c * c)
    denom = R2 + a_soft * a_soft
    alpha_x = pref * dx / denom
    alpha_y = pref * dy / denom
    return alpha_x, alpha_y


def calibrate_pm_ratio(
    M: float,
    a_soft: float,
    mu_coeff: float,
    v_transverse: float,
    b_min: float,
    b_max: float,
    samples: int = 200,
    z_max_factor: float = 6.0,
    steps: int = 1200,
) -> tuple[np.ndarray, np.ndarray]:
    """Return sampled impact parameters b and corresponding PM/GR deflection ratios for a lens.

    We compute the static Fermat deflection (matching the index model n = 1 + mu M / r)
    and the moving-lens correction using `moving_lens_deflection_numeric`, then tabulate
    the ratio α_PM / α_static. The ratio is near unity but encodes first-order v/c effects.
    """

    if v_transverse == 0.0:
        return np.array([b_min, b_max]), np.ones(2)

    bs = np.linspace(b_min, b_max, samples)
    ratios = np.ones_like(bs)
    z_max = z_max_factor * a_soft

    for i, b in enumerate(bs):
        static = pm.fermat_deflection_static_index(M, float(b), mu=mu_coeff, z_max=z_max, steps=steps)
        moving = pm.moving_lens_deflection_numeric(M, float(b), mu=mu_coeff, v_transverse=v_transverse, z_max=z_max, steps=steps)
        if static <= 0:
            ratios[i] = 1.0
        else:
            ratios[i] = moving / static
    return bs, ratios


def main() -> None:
    # Cluster parameters (toy values resembling Bullet Cluster order-of-magnitude)
    M_main = 1.0e15 * M_SUN
    M_sub = 3.0e14 * M_SUN
    a_main = 150.0 * kpc
    a_sub = 80.0 * kpc
    sep = 800.0 * kpc

    # Lens/source distance placeholders (used only if converting angles to arcsec) – we keep thin-lens units.
    # Field grid
    nx = ny = 220
    field_size = 2200.0 * kpc
    xs = np.linspace(-field_size / 2.0, field_size / 2.0, nx)
    ys = np.linspace(-field_size / 2.0, field_size / 2.0, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")

    # Centres of main/sub clusters
    x_main, y_main = -sep / 2.0, 0.0
    x_sub, y_sub = +sep / 2.0, 0.0

    # Surface density maps (kg / m^2)
    sigma_main = plummer_surface_density(M_main, a_main, XX, YY, x_main, y_main)
    sigma_sub = plummer_surface_density(M_sub, a_sub, XX, YY, x_sub, y_sub)
    sigma_total = sigma_main + sigma_sub

    # GR deflection field (radians)
    ax_main_gr, ay_main_gr = plummer_deflection(M_main, a_main, XX, YY, x_main, y_main)
    ax_sub_gr, ay_sub_gr = plummer_deflection(M_sub, a_sub, XX, YY, x_sub, y_sub)
    alpha_x_gr = ax_main_gr + ax_sub_gr
    alpha_y_gr = ay_main_gr + ay_sub_gr

    # PM moving-lens correction for the subcluster
    v_sub = 3000.0 * 1000.0  # 3000 km/s transverse speed
    mu_coeff = 2.0 * G / (c * c)
    R_sub = np.hypot(XX - x_sub, YY - y_sub)
    b_min = max(0.1 * kpc, float(np.min(R_sub[R_sub > 0.0])))
    b_max = float(np.max(R_sub))
    bs, ratio_samples = calibrate_pm_ratio(M_sub, a_sub, mu_coeff, v_sub, b_min=b_min, b_max=b_max, samples=150)
    ratio_interp = np.interp(R_sub.ravel(), bs, ratio_samples, left=ratio_samples[0], right=ratio_samples[-1])
    ratio_grid = ratio_interp.reshape(R_sub.shape)

    # Apply ratio only to moving subcluster contribution; main cluster stationary
    alpha_x_pm = ax_main_gr + ax_sub_gr * ratio_grid
    alpha_y_pm = ay_main_gr + ay_sub_gr * ratio_grid

    # Diagnostics: magnitudes in radians + arcsec
    mag_gr = np.hypot(alpha_x_gr, alpha_y_gr)
    mag_pm = np.hypot(alpha_x_pm, alpha_y_pm)
    mag_diff = mag_pm - mag_gr
    mag_frac = np.where(mag_gr > 0.0, mag_diff / mag_gr, 0.0)

    mag_gr_arcsec = mag_gr * ARCSEC_PER_RAD
    mag_pm_arcsec = mag_pm * ARCSEC_PER_RAD
    mag_diff_mas = mag_diff * ARCSEC_PER_RAD * 1000.0  # milli-arcsec difference

    # Plotting setup
    extent = [xs[0] / kpc, xs[-1] / kpc, ys[0] / kpc, ys[-1] / kpc]
    fig, axs = plt.subplots(2, 2, figsize=(13, 11))

    # Surface density map
    ax = axs[0, 0]
    im0 = ax.imshow(sigma_total / (M_SUN / (kpc ** 2)), origin="lower", extent=extent, cmap="inferno", norm=LogNorm())
    ax.set_title("Projected surface density Σ (Msun / kpc²)")
    fig.colorbar(im0, ax=ax, label="Σ [Msun / kpc²]")
    ax.plot([x_main / kpc], [y_main / kpc], marker="o", color="#00FFFF", label="main mass")
    ax.plot([x_sub / kpc], [y_sub / kpc], marker="o", color="#FF00FF", label="sub mass")
    ax.legend(loc="upper right")

    # GR deflection magnitude + streamlines
    ax = axs[0, 1]
    im1 = ax.imshow(np.maximum(mag_gr_arcsec, 1e-6), origin="lower", extent=extent, cmap="magma", norm=LogNorm())
    ax.set_title("GR thin-lens deflection |α| (arcsec)")
    cb1 = fig.colorbar(im1, ax=ax, label="|α| [arcsec]")
    speed_scaled = mag_gr / (np.max(mag_gr) + 1e-30)
    lw = 0.8 + 1.5 * speed_scaled
    ax.streamplot(XX / kpc, YY / kpc, alpha_x_gr, alpha_y_gr, color="white", linewidth=lw, density=1.2, arrowsize=0.8)

    # PM deflection magnitude + streamlines
    ax = axs[1, 0]
    im2 = ax.imshow(np.maximum(mag_pm_arcsec, 1e-6), origin="lower", extent=extent, cmap="viridis", norm=LogNorm())
    ax.set_title("PM-flow deflection |α| (arcsec)")
    fig.colorbar(im2, ax=ax, label="|α| [arcsec]")
    speed_pm_scaled = mag_pm / (np.max(mag_pm) + 1e-30)
    lw_pm = 0.8 + 1.5 * speed_pm_scaled
    ax.streamplot(XX / kpc, YY / kpc, alpha_x_pm, alpha_y_pm, color="white", linewidth=lw_pm, density=1.2, arrowsize=0.8)

    # Fractional difference map
    ax = axs[1, 1]
    im3 = ax.imshow(mag_frac * 100.0, origin="lower", extent=extent, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    ax.set_title("Fractional change (PM/GR − 1) [%]")
    fig.colorbar(im3, ax=ax, label="Δα / α_GR [%]")

    for ax in axs.flat:
        ax.set_xlabel("x [kpc]")
        ax.set_ylabel("y [kpc]")
        ax.set_aspect("equal", adjustable="box")

    fig.suptitle("Toy Bullet Cluster: GR vs Pushing-Medium moving-lens comparison", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    outdir = repo_root / "programs" / "demos" / "benchmarks" / "out"
    outdir.mkdir(parents=True, exist_ok=True)
    outfile = outdir / "bullet_cluster_comparison.png"
    plt.savefig(outfile, dpi=220)
    print(f"Wrote {outfile}")


if __name__ == "__main__":
    main()

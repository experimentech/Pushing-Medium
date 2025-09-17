import math
from typing import List, Sequence, Tuple

G = 6.67430e-11
c = 299792458.0


def index_point_masses(r: Sequence[float], masses: List[Tuple[float, Sequence[float]]], mu_scale: float = None) -> float:
    """n(r) = 1 + sum_i mu_i / |r - r_i|, with mu_i = 2 G M_i / c^2 (or chosen scaling)."""
    if mu_scale is None:
        mu = lambda M: 2 * G * M / (c * c)
    else:
        mu = mu_scale
    x, y, z = r
    n = 1.0
    for M, ri in masses:
        dx, dy, dz = x - ri[0], y - ri[1], z - ri[2]
        d = math.sqrt(dx * dx + dy * dy + dz * dz) + 1e-12
        n += mu(M) / d
    return n


def index_from_density(
    r: Sequence[float],
    density_fn,
    bounds: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]],
    N: int = 32,
    mu_coeff: float = 2 * G / (c * c),
    eps: float = 1e-9,
):
    """
    Compute n(r) = 1 + (2G/c^2) ∫ ρ(r') / |r - r'| d^3r' via simple grid integration.

    Parameters
    - r: observation point (x,y,z)
    - density_fn: callable rho(x,y,z) [kg/m^3]
    - bounds: ((x_min,x_max),(y_min,y_max),(z_min,z_max)) integration box covering nonzero density
    - N: grid resolution per axis (total samples N^3)
    - mu_coeff: coefficient for mapping to index (defaults to 2G/c^2)
    - eps: small softening to avoid singularity when r'≈r

    Notes: This is a crude Riemann-sum integrator intended for tests; for production, use adaptive quadrature or analytic kernels.
    """
    (x0, x1), (y0, y1), (z0, z1) = bounds
    x, y, z = r
    dx = (x1 - x0) / N
    dy = (y1 - y0) / N
    dz = (z1 - z0) / N
    dV = dx * dy * dz
    acc = 0.0
    # cell-centered sampling
    for i in range(N):
        xi = x0 + (i + 0.5) * dx
        for j in range(N):
            yj = y0 + (j + 0.5) * dy
            for k in range(N):
                zk = z0 + (k + 0.5) * dz
                rho = density_fn(xi, yj, zk)
                if rho == 0.0:
                    continue
                rx, ry, rz = x - xi, y - yj, z - zk
                inv_r = 1.0 / math.sqrt(rx * rx + ry * ry + rz * rz + eps * eps)
                acc += rho * inv_r * dV
    return 1.0 + mu_coeff * acc


def flow_rotational(r: Sequence[float], spins: List[Tuple[Sequence[float], Sequence[float]]]):
    """u_g(r) = sum_i Omega_i × (r - r_i)."""
    ux = uy = uz = 0.0
    x, y, z = r
    for ri, Omega in spins:
        rx, ry, rz = x - ri[0], y - ri[1], z - ri[2]
        Ox, Oy, Oz = Omega
        ux += Oy * rz - Oz * ry
        uy += Oz * rx - Ox * rz
        uz += Ox * ry - Oy * rx
    return (ux, uy, uz)


def flow_translational_retarded(r, t, source_fn):
    """Placeholder for retarded translational flow; user supplies source_fn returning u_g at (r,t)."""
    return source_fn(r, t)


def ray_direction_update(grad_ln_n_perp: Sequence[float]):
    """d k_hat / ds = ∇_⊥ ln n_total. Caller computes the perpendicular gradient."""
    return tuple(grad_ln_n_perp)


def ray_advection(r: Sequence[float], k_hat: Sequence[float], n_total: float, u_g: Sequence[float]):
    """dr/dt = c k_hat / n_total + u_g."""
    vx = c * k_hat[0] / n_total + u_g[0]
    vy = c * k_hat[1] / n_total + u_g[1]
    vz = c * k_hat[2] / n_total + u_g[2]
    return (vx, vy, vz)


def massive_accel_medium(grad_ln_n: Sequence[float]):
    """a_med = - c^2 ∇ ln n_total."""
    return tuple(-c * c * g for g in grad_ln_n)


def newtonian_accel_sum(r: Sequence[float], masses: List[Tuple[float, Sequence[float]]]):
    """a_grav = - sum_i G M_i (r - r_i) / |r - r_i|^3."""
    ax = ay = az = 0.0
    x, y, z = r
    for M, ri in masses:
        dx, dy, dz = x - ri[0], y - ri[1], z - ri[2]
        r2 = dx * dx + dy * dy + dz * dz
        r3 = (r2 + 1e-24) ** 1.5
        scale = -G * M / r3
        ax += scale * dx
        ay += scale * dy
        az += scale * dz
    return (ax, ay, az)


# --- GR-mapped helper functions for testbench comparisons ---

def pm_deflection_angle_point_mass(M: float, b: float) -> float:
    """Weak-field light deflection (radians): 4GM/(c^2 b)."""
    return 4 * G * M / (c * c * b)


def pm_shapiro_delay_point_mass(M: float, r1: float, r2: float, b: float) -> float:
    """Shapiro delay in static field: Δt ≈ (2GM/c^3) ln(4 r1 r2 / b^2)."""
    return (2 * G * M / (c ** 3)) * math.log(4 * r1 * r2 / (b * b))


def pm_gravitational_redshift_from_potential(delta_phi: float) -> float:
    """Gravitational redshift z ≈ Δϕ/c^2 (small potentials)."""
    return delta_phi / (c * c)


def pm_perihelion_precession(a: float, e: float, M: float) -> float:
    """Per orbit perihelion advance (radians): 6πGM/(c^2 a (1-e^2))."""
    return 6 * math.pi * G * M / (c * c * a * (1 - e * e))


def pm_einstein_radius_point_mass(M: float, D_l: float, D_s: float, D_ls: float) -> float:
    """Einstein angle (radians): θ_E = sqrt(4GM D_ls / (c^2 D_l D_s))."""
    return math.sqrt(4 * G * M * D_ls / (c * c * D_l * D_s))


def pm_gw_phase_speed() -> float:
    """GW phase/group speed in PM TT sector: c."""
    return c


def pm_binary_quadrupole_power(M1: float, M2: float, a: float) -> float:
    """Quadrupole power (circular binary): (32/5) G^4 (M1 M2)^2 (M1+M2) / (c^5 a^5)."""
    return (32.0 / 5.0) * (G ** 4) * ((M1 * M2) ** 2) * (M1 + M2) / (c ** 5 * a ** 5)


def pm_circular_orbit_energy(M: float, a: float) -> float:
    """Specific orbital energy (per unit mass), Newtonian limit: −GM/(2a)."""
    return -G * M / (2 * a)


def lense_thirring_precession(J: float, r: float) -> float:
    """Frame-drag angular velocity ω = 2 G J / (c^2 r^3) (same as GR form)."""
    return 2 * G * J / (c * c * r ** 3)


# --- Analytic density helpers (Plummer) ---

def plummer_density(M: float, a: float):
    """Return rho(r) for a Plummer sphere: rho(r) = (3M/4π a^3) (1 + r^2/a^2)^(-5/2)."""
    coeff = 3 * M / (4 * math.pi * a ** 3)

    def rho(x, y, z):
        r2 = x * x + y * y + z * z
        return coeff * (1 + r2 / (a * a)) ** (-2.5)

    return rho


def plummer_mass_enclosed(M: float, a: float, r: float) -> float:
    """M(<r) = M r^3 / (r^2 + a^2)^(3/2)."""
    return M * (r ** 3) / (r * r + a * a) ** 1.5


def weak_bending_deflection_numeric(M: float, b: float, z_max: float = 50.0, steps: int = 10000) -> float:
    """
    Numerically estimate light deflection in the weak field using the PM index model n = 1 + mu M / r
    with mu = 2G/c^2, i.e., d kx/dz ≈ ∂/∂x ln n. This reproduces 4GM/(c^2 b).
    """
    mu = 2 * G / (c * c)
    return index_deflection_numeric(M, b, mu=mu, z_max=z_max, steps=steps)


def index_deflection_numeric(M: float, b: float, mu: float, z_max: float = 50.0, steps: int = 5000) -> float:
        """
        Numerically estimate light deflection using the PM index model:
            n(r) = 1 + mu * M / r,  ln n = ln(1 + mu M / r)
        Small-angle approximation: d kx/dz ≈ ∂/∂x ln n, with x=b, y=0, z as path parameter.
        Deflection α ≈ ∫ (∂ ln n / ∂x) dz over z in [-z_max, z_max].
        """
        import numpy as np

        zs = np.linspace(-z_max, z_max, steps)
        dz = zs[1] - zs[0]
        alpha = 0.0
        for z in zs:
                r = math.sqrt(b * b + z * z) + 1e-30
                n_val = 1.0 + (mu * M) / r
                dndx = - (mu * M) * b / (r ** 3)
                dlnndx = dndx / n_val
                alpha += dlnndx * dz
        return float(abs(alpha))

from __future__ import annotations
import numpy as np
from typing import Callable, Tuple, Optional

# -----------------------------
# Physical constants
# -----------------------------
c: float = 299_792_458.0          # m/s
G: float = 6.67430e-11            # m^3/(kg s^2)

# -----------------------------
# Index fields n(x) and derivatives
# -----------------------------
def gaussian_index(eps: float, sigma: float, center: Tuple[float, float] = (0.0, 0.0)
                   ) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """
    n(x,y) = 1 + eps * exp(-((x-x0)^2 + (y-y0)^2)/(2 sigma^2))
    Returns a callable n(x, y).
    """
    x0, y0 = center
    sig2 = sigma * sigma
    def n(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        dx = x - x0; dy = y - y0
        r2 = dx*dx + dy*dy
        return 1.0 + eps * np.exp(-0.5 * r2 / sig2)
    return n

def grad_n_gaussian(eps: float, sigma: float, center: Tuple[float, float] = (0.0, 0.0)
                    ) -> Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    ∇n(x,y) for the Gaussian index: grad = eps*exp(-r^2/(2σ^2)) * (-(dx)/σ^2, -(dy)/σ^2)
    Returns a callable (nx, ny)(x, y).
    """
    x0, y0 = center
    sig2 = sigma * sigma
    def grad(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        dx = x - x0; dy = y - y0
        r2 = dx*dx + dy*dy
        e = eps * np.exp(-0.5 * r2 / sig2)
        nx = e * (-(dx) / sig2)
        ny = e * (-(dy) / sig2)
        return nx, ny
    return grad

def hessian_n_gaussian(eps: float, sigma: float, center: Tuple[float, float] = (0.0, 0.0)
                       ) -> Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Hessian H(n) for Gaussian index. Returns (n_xx, n_xy, n_yy).
    """
    x0, y0 = center
    sig2 = sigma * sigma
    def hess(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        dx = x - x0; dy = y - y0
        r2 = dx*dx + dy*dy
        e = eps * np.exp(-0.5 * r2 / sig2)
        common = e / (sig2*sig2)
        n_xx = common * (dx*dx - sig2)
        n_yy = common * (dy*dy - sig2)
        n_xy = common * (dx*dy)
        return n_xx, n_xy, n_yy
    return hess

def grad_n_central(n_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
                   x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Central-difference gradient of a generic n(x,y) over grid (x,y) (1D arrays).
    Returns (n_x, n_y) sampled on meshgrid(X,Y).
    """
    X, Y = np.meshgrid(x, y)
    N = n_fn(X, Y)
    dx = x[1] - x[0]; dy = y[1] - y[0]
    n_x = np.empty_like(N); n_y = np.empty_like(N)
    n_x[:, 1:-1] = (N[:, 2:] - N[:, :-2]) / (2*dx)
    n_x[:, 0]    = (N[:, 1] - N[:, 0]) / dx
    n_x[:, -1]   = (N[:, -1] - N[:, -2]) / dx
    n_y[1:-1, :] = (N[2:, :] - N[:-2, :]) / (2*dy)
    n_y[0, :]    = (N[1, :] - N[0, :]) / dy
    n_y[-1, :]   = (N[-1, :] - N[-2, :]) / dy
    return n_x, n_y

def hessian_n_central(n_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
                      x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Central-difference Hessian of a generic n(x,y) over grid (x,y).
    Returns (n_xx, n_xy, n_yy).
    """
    X, Y = np.meshgrid(x, y)
    N = n_fn(X, Y)
    dx = x[1] - x[0]; dy = y[1] - y[0]
    n_xx = np.empty_like(N); n_yy = np.empty_like(N)
    n_xx[:, 1:-1] = (N[:, 2:] - 2*N[:, 1:-1] + N[:, :-2]) / (dx*dx)
    n_xx[:, 0]    = (N[:, 2] - 2*N[:, 1] + N[:, 0]) / (dx*dx)
    n_xx[:, -1]   = (N[:, -1] - 2*N[:, -2] + N[:, -3]) / (dx*dx)
    n_yy[1:-1, :] = (N[2:, :] - 2*N[1:-1, :] + N[:-2, :]) / (dy*dy)
    n_yy[0, :]    = (N[2, :] - 2*N[1, :] + N[0, :]) / (dy*dy)
    n_yy[-1, :]   = (N[-1, :] - 2*N[-2, :] + N[-3, :]) / (dy*dy)
    # mixed via successive diffs
    N_x = np.empty_like(N)
    N_x[:, 1:-1] = (N[:, 2:] - N[:, :-2]) / (2*dx)
    N_x[:, 0]    = (N[:, 1] - N[:, 0]) / dx
    N_x[:, -1]   = (N[:, -1] - N[:, -2]) / dx
    n_xy = np.empty_like(N)
    n_xy[1:-1, :] = (N_x[2:, :] - N_x[:-2, :]) / (2*dy)
    n_xy[0, :]    = (N_x[1, :] - N_x[0, :]) / dy
    n_xy[-1, :]   = (N_x[-1, :] - N_x[-2, :]) / dy
    return n_xx, n_xy, n_yy

# -----------------------------
# Rotational flow (frame-drag analogue)
# -----------------------------
def omega_s(r: np.ndarray, J: float) -> np.ndarray:
    """
    ω_s(r) = 2 G J / (c^2 r^3). Input r>0 (m), J=|angular momentum| (kg m^2/s).
    Returns ω_s in s^-1.
    """
    return 2.0 * G * J / (c*c * np.power(r, 3))

# -----------------------------
# Translational flow operator: u_g wave equation helper
# (∂_t^2 u_g)/c^2 - ∇^2 u_g = κ_J * J_TT
# These are utility pieces, so you can assemble solvers.
# -----------------------------
def laplacian2d(F: np.ndarray, dx: float, dy: float) -> np.ndarray:
    L = np.zeros_like(F)
    L[1:-1, 1:-1] = (
        (F[1:-1, 2:] - 2*F[1:-1, 1:-1] + F[1:-1, :-2]) / (dx*dx) +
        (F[2:, 1:-1] - 2*F[1:-1, 1:-1] + F[:-2, 1:-1]) / (dy*dy)
    )
    # Neumann copy at edges (simple)
    L[:, 0] = L[:, 1]; L[:, -1] = L[:, -2]
    L[0, :] = L[1, :]; L[-1, :] = L[-2, :]
    return L

def laplacian3d(F: np.ndarray, dx: float, dy: float, dz: float) -> np.ndarray:
    L = np.zeros_like(F)
    L[1:-1, 1:-1, 1:-1] = (
        (F[1:-1, 1:-1, 2:] - 2*F[1:-1, 1:-1, 1:-1] + F[1:-1, 1:-1, :-2]) / (dz*dz) +
        (F[1:-1, 2:, 1:-1] - 2*F[1:-1, 1:-1, 1:-1] + F[1:-1, :-2, 1:-1]) / (dy*dy) +
        (F[2:, 1:-1, 1:-1] - 2*F[1:-1, 1:-1, 1:-1] + F[:-2, 1:-1, 1:-1]) / (dx*dx)
    )
    # Neumann copy at edges
    L[:, :, 0] = L[:, :, 1]; L[:, :, -1] = L[:, :, -2]
    L[:, 0, :] = L[:, 1, :]; L[:, -1, :] = L[:, -2, :]
    L[0, :, :] = L[1, :, :]; L[-1, :, :] = L[-2, :, :]
    return L

def wave_operator_u(U: np.ndarray, U_prev: np.ndarray, dt: float,
                    lap_fn: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """
    Discrete wave operator on u_g: (1/c^2) d2U/dt^2 - ∇^2 U
    using central time difference.
    Supply lap_fn(U) that returns ∇^2 U (component-wise if vector field).
    """
    d2t = (U - 2*U_prev + U_prev)  # placeholder to show shape; see note below
    # NOTE: This is a placeholder pattern; in practice store U_next, U_curr, U_prev and compute:
    # d2t = (U_next - 2*U_curr + U_prev) / (dt*dt)
    # Here we keep interface minimal so you can wire to your integrator.
    return (d2t / (c*c)) - lap_fn(U)

# -----------------------------
# Fermat functional and ray update
# -----------------------------
def fermat_integrand(n_val: float, u_dot_k: float) -> float:
    """
    Differential contribution to travel time:
      dT = n ds - (1/c) (n^2 - 1) (u_g · k_hat) ds
    Return the integrand f = n - (1/c) (n^2 - 1) (u_g · k_hat).
    """
    return n_val - (1.0 / c) * (n_val*n_val - 1.0) * u_dot_k

def ray_step_isotropic(r: np.ndarray, k_hat: np.ndarray,
                       n_val: float, grad_n: np.ndarray, ds: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Single ray step in isotropic medium (u_g = 0 case):
      d/ds (n k̂) = ∇n  ⇒  n dk̂/ds = ∇n - (k̂·∇n) k̂
    Returns updated (r_next, k_hat_next).
    """
    gn = grad_n
    gn_par = np.dot(k_hat, gn) * k_hat
    dk = (gn - gn_par) * (ds / max(n_val, 1e-15))
    k_new = k_hat + dk
    k_new /= np.linalg.norm(k_new) + 1e-15
    r_new = r + k_new * ds
    return r_new, k_new

# -----------------------------
# Symmetric-hyperbolic blocks
# -----------------------------
def sound_speed_sq(Pprime: float, n_val: float) -> float:
    """
    c_s^2 = dP/dn evaluated at n=n_val. Input Pprime = dP/dn(n_val).
    """
    return Pprime

def symmetrizer_blocks(Pprime_over_n: float, inv_n: float
                       ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fluid and TT blocks of the symmetrizer (diagonal forms):
      S_fl = diag(P'(n)/n, (1/n) I3)
      S_TT = diag(I, c^2 I)
    Returns (S_fl, S_TT).
    """
    S_fl = np.diag([Pprime_over_n, inv_n, inv_n, inv_n])
    S_TT = np.block([
        [np.eye(1),            np.zeros((1, 1))],
        [np.zeros((1, 1)),     (c*c) * np.eye(1)]
    ])
    # Note: S_TT here is a 2x2 placeholder per component (h, q) channel to keep API simple.
    return S_fl, S_TT

def energy_density(n_val: float, m_vec: np.ndarray,
                   Pprime_over_n: float,
                   p_tt_sq: float, q_tt_sq: float) -> float:
    """
    𝔈 = 1/2 [ (P'(n)/n) n^2 + |m|^2/n + |p_TT|^2 + c^2 |q_TT|^2 ]
    Inputs:
      n_val: scalar n
      m_vec: 3-vector momentum m = n v
      p_tt_sq: sum of squares of TT time derivatives
      q_tt_sq: sum of squares of spatial derivatives
    """
    term_fl = 0.5 * (Pprime_over_n * (n_val**2) + np.dot(m_vec, m_vec) / max(n_val, 1e-15))
    term_tt = 0.5 * (p_tt_sq + (c*c) * q_tt_sq)
    return term_fl + term_tt

def energy_flux(U: np.ndarray, S_mat: np.ndarray, A_k: np.ndarray) -> float:
    """
    𝔽^k = 1/2 U^T S A^k U. Provide your assembled state U, symmetrizer S, and flux Jacobian A^k.
    This helper just computes the quadratic form.
    """
    return 0.5 * float(U.T @ (S_mat @ (A_k @ U)))

def characteristic_speeds(v_n: float, c_s: float) -> Tuple[float, float, float, float, float]:
    """
    Returns (λ_minus, λ_0, λ_plus, λ_TT_minus, λ_TT_plus)
      λ_± = v_n ± c_s
      λ_0 = v_n
      λ_TT = ± c
    """
    return (v_n - c_s, v_n, v_n + c_s, -c, +c)

# -----------------------------
# TT wave helpers (scalar channel utility)
# -----------------------------
def wave_operator_scalar(H: np.ndarray, H_prev: np.ndarray, dt: float,
                         lap_fn: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """
    Discrete wave operator on scalar H:
      (1/c^2) ∂_t^2 H - ∇^2 H
    using central time difference stencil. As with wave_operator_u, wire into your
    integrator; this returns the operator applied to the current field placeholder.
    """
    d2t = (H - 2*H_prev + H_prev)  # placeholder pattern; see note in wave_operator_u
    return (d2t / (c*c)) - lap_fn(H)

# -----------------------------
# Cosmology-lite
# -----------------------------
def H_LCDM(H0: float, Omega_m: float, Omega_L: Optional[float] = None
           ) -> Callable[[np.ndarray], np.ndarray]:
    """
    H(z) = H0 * sqrt(Ω_m (1+z)^3 + Ω_Λ + Ω_k (1+z)^2)
    If Omega_L is None, assume flat: Ω_Λ = 1 - Ω_m.
    Returns a callable H(z) with z >= 0 (same units as H0).
    """
    if Omega_L is None:
        Omega_L = 1.0 - Omega_m
    Omega_k = max(0.0, 1.0 - Omega_m - Omega_L)
    def H_of_z(z: np.ndarray) -> np.ndarray:
        zp1 = 1.0 + z
        return H0 * np.sqrt(Omega_m * zp1**3 + Omega_L + Omega_k * zp1**2)
    return H_of_z

def comoving_distance(z: np.ndarray, H_of_z: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """
    D_C(z) = ∫_0^z c / H(z') dz' using simple composite trapezoid.
    """
    z = np.asarray(z)
    Hz = H_of_z(z)
    # Trapezoid in z; assumes monotone z
    integrand = c / np.maximum(Hz, 1e-30)
    # cumulative trapz
    dc = np.zeros_like(z, dtype=float)
    if len(z) > 1:
        dz = np.diff(z)
        dc[1:] = np.cumsum(0.5 * (integrand[:-1] + integrand[1:]) * dz)
    return dc

def luminosity_distance(z: np.ndarray, D_C: np.ndarray) -> np.ndarray:
    """
    D_L = (1+z) D_C
    """
    return (1.0 + z) * D_C

def angular_diameter_distance(z: np.ndarray, D_C: np.ndarray) -> np.ndarray:
    """
    D_A = D_C / (1+z)
    """
    return D_C / np.maximum(1.0 + z, 1e-30)

def growth_rhs(a: float, delta: float, delta_prime: float,
               H_of_a: Callable[[float], float],
               Omega_m_of_a: Callable[[float], float]) -> float:
    """
    RHS of growth ODE in ln a:
      δ'' + (2 + H'/H) δ' - (3/2) Ω_m(a) δ = 0
    Rearranged for δ'' = ...
    Inputs:
      a: scale factor
      delta: δ(a)
      delta_prime: dδ/d ln a
      H_of_a: callable H(a)
      Omega_m_of_a: callable Ω_m(a)
    """
    # Numerical derivative H'/H in ln a: (d ln H)/(d ln a)
    h = 1e-4
    Ha = H_of_a(a)
    Hp = H_of_a(a * np.exp(h))
    Hm = H_of_a(a * np.exp(-h))
    dlnH_dlnA = (np.log(Hp) - np.log(Hm)) / (2*h)
    return - (2.0 + dlnH_dlnA) * delta_prime + 1.5 * Omega_m_of_a(a) * delta


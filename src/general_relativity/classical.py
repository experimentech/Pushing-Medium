import math

G = 6.67430e-11
c = 299792458.0


def deflection_angle_point_mass(M, b):
    """GR light deflection (radians) for a point mass in weak field: alpha = 4GM/(c^2 b)."""
    return 4 * G * M / (c * c * b)


def shapiro_delay_point_mass(M, r1, r2, b):
    """
    Shapiro time delay (seconds) for signal passing with impact parameter b near mass M.
    Uses standard logarithmic formula: Δt = (2GM/c^3) ln[(r1 + r2 + D)/(r1 + r2 - D)] with D≈sqrt(r1^2 + r2^2 - 2 r1 r2 cosθ).
    For simplicity assume small-angle path with closest approach b and large r1,r2: Δt ≈ (2GM/c^3) ln(4 r1 r2 / b^2).
    """
    return (2 * G * M / (c ** 3)) * math.log(4 * r1 * r2 / (b * b))


def gravitational_redshift_potential(delta_phi):
    """Gravitational redshift z ≈ Δϕ/c^2 for small potentials (frequency shift)."""
    return delta_phi / (c * c)


def perihelion_precession(a, e, M):
    """Per orbit perihelion advance (radians): Δω = 6πGM/(c^2 a (1-e^2))."""
    return 6 * math.pi * G * M / (c * c * a * (1 - e * e))


def lense_thirring_precession(J, r):
    """Frame-drag angular velocity ω = 2 G J / (c^2 r^3)."""
    return 2 * G * J / (c * c * r ** 3)


def newtonian_acceleration(M, r_vec):
    """Newtonian acceleration vector: a = - G M r / |r|^3."""
    rx, ry, rz = r_vec
    r3 = (rx * rx + ry * ry + rz * rz) ** 1.5
    if r3 == 0:
        return (0.0, 0.0, 0.0)
    scale = -G * M / r3
    return (scale * rx, scale * ry, scale * rz)


def einstein_radius_point_mass(M, D_l, D_s, D_ls):
    """Einstein angle (radians): θ_E = sqrt(4GM D_ls / (c^2 D_l D_s))."""
    return math.sqrt(4 * G * M * D_ls / (c * c * D_l * D_s))


def gw_phase_speed():
    """GW phase/group speed in GR vacuum: c."""
    return c


def binary_quadrupole_power(M1, M2, a):
    """Quadrupole power radiated by a circular binary: P = (32/5) G^4 (M1 M2)^2 (M1+M2) / (c^5 a^5)."""
    return (32.0 / 5.0) * (G ** 4) * ((M1 * M2) ** 2) * (M1 + M2) / (c ** 5 * a ** 5)


def circular_orbit_energy(M, a):
    """Specific orbital energy (per unit mass) for circular orbit in Newtonian limit: E = - GM/(2a)."""
    return -G * M / (2 * a)

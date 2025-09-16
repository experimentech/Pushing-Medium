import numpy as np

# Bodies (positions r_i, strengths Γ_i)
Gamma = np.array([1000.0, 1.0, 0.0123])
R = np.array([[0.0, 0.0], [1.0, 0.0], [1.00257, 0.0]])  # Sun, Earth, Moon
eps2 = 1e-6
Omega = 2*np.pi  # rotating frame rate

def u(r):
    # Substrate-inspired inflow sum (vector, no grid)
    d = R - r  # (N,2)
    r2 = np.sum(d*d, axis=1) + eps2
    inv_r32 = Gamma / (r2**1.5)
    return (inv_r32[:,None] * d).sum(axis=0)

def uOmega(r):
    ux, uy = u(r)
    x, y = r
    return np.array([ux + Omega*y, uy - Omega*x])

def J(r):
    # Analytic Jacobian of uOmega
    d = R - r
    r2 = np.sum(d*d, axis=1) + eps2
    r5 = r2**2.5
    # For each body: ∂/∂x of (Gamma * dx / r^3) = Gamma * ( (1/r^3) - 3*dx^2/r^5 )
    dx, dy = d[:,0], d[:,1]
    g = Gamma
    uxx = (g*(1.0/r2**1.5 - 3*dx*dx/r5)).sum()
    uxy = (g*(-3*dx*dy/r5)).sum()
    uyx = uxy
    uyy = (g*(1.0/r2**1.5 - 3*dy*dy/r5)).sum()
    # Add rotating-frame derivatives
    # d/dx (ux + Omega*y) = uxx ; d/dy(...) = uxy + Omega
    # d/dx (uy - Omega*x) = uyx - Omega ; d/dy(...) = uyy
    return np.array([[uxx, uxy + Omega],
                     [uyx - Omega, uyy]])

def newton_root(r0, tol=1e-10, itmax=50):
    r = r0.copy()
    for _ in range(itmax):
        F = uOmega(r)
        if np.linalg.norm(F) < tol:
            return r, True
        Jm = J(r)
        try:
            step = np.linalg.solve(Jm, F)
        except np.linalg.LinAlgError:
            return r, False
        r = r - step
        if np.linalg.norm(step) < tol:
            return r, True
    return r, False

def classify(rstar):
    eig = np.linalg.eigvals(J(rstar))
    if np.all(np.isreal(eig)):
        return 'saddle' if np.prod(eig) < 0 else 'node'
    return 'center/spiral'

# Example: seed near expected Sun–Earth L1/L2/L4/L5 locations and refine
seeds = [
    np.array([0.99, 0.0]),   # L1-ish
    np.array([1.01, 0.0]),   # L2-ish
    np.array([0.0, 0.0]),    # near Sun-side
    np.array([0.5,  +0.866]),# L4-ish (~60 deg)
    np.array([0.5,  -0.866]) # L5-ish
]
stagnations = []
for s in seeds:
    rstar, ok = newton_root(s)
    if ok:
        label = classify(rstar)
        stagnations.append((rstar, label))

# Manifold tracing from a saddle r*
def trace_manifold(rstar, kind='unstable', ds=1e-3, max_steps=20000):
    # eigenvectors of J at r*: choose unstable (positive real) or stable (negative real)
    w, V = np.linalg.eig(J(rstar))
    idx = np.argmax(w.real) if kind=='unstable' else np.argmin(w.real)
    v0 = V[:, idx].real
    v0 /= np.linalg.norm(v0)
    # small offset to leave the linear neighborhood
    r = rstar + 1e-5 * v0
    path = [r.copy()]
    for _ in range(max_steps):
        # Follow integral curve of uOmega
        v = uOmega(r)
        nv = np.linalg.norm(v)
        if nv < 1e-12 or nv > 1e3:  # bail out if stagnant or far
            break
        r = r + ds * v / nv  # arc-length-ish step
        path.append(r.copy())
        if np.linalg.norm(r) > 5.0:  # domain exit
            break
    return np.array(path)


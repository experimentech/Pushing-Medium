# vector_skeleton.py
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Callable

@dataclass
class Body:
    pos: np.ndarray  # shape (2,)
    gamma: float     # sink strength (mass-like)

@dataclass
class Stag:
    r: np.ndarray      # position (2,)
    label: str         # 'saddle', 'center/spiral', 'node'
    J: np.ndarray      # Jacobian 2x2
    eigvals: np.ndarray  # eigenvalues
    eigvecs: np.ndarray  # eigenvectors (columns)

def moon_pos(phase: float, earth: np.ndarray, r_em: float = 0.00257) -> np.ndarray:
    return earth + r_em * np.array([np.cos(phase), np.sin(phase)])

class VectorFlow:
    def __init__(self, bodies: List[Body], omega: float, eps2: float = 1e-6):
        self.bodies = bodies
        self.omega = omega
        self.eps2 = eps2

    def u(self, r: np.ndarray) -> np.ndarray:
        # Substrate-inspired inflow sum: Σ Γ Δ / (|Δ|^2+eps2)^(3/2)
        v = np.zeros(2)
        for b in self.bodies:
            d = b.pos - r
            r2 = np.dot(d, d) + self.eps2
            v += (b.gamma / (r2**1.5)) * d
        return v

    def uOmega(self, r: np.ndarray) -> np.ndarray:
        ux, uy = self.u(r)
        x, y = r
        return np.array([ux + self.omega * y, uy - self.omega * x])

    def J(self, r: np.ndarray) -> np.ndarray:
        uxx = uxy = uyy = 0.0
        for b in self.bodies:
            d = b.pos - r
            dx, dy = d[0], d[1]
            r2 = dx*dx + dy*dy + self.eps2
            r5 = r2**2.5
            c = b.gamma
            uxx += c * (1.0 / (r2**1.5) - 3.0 * dx*dx / r5)
            uyy += c * (1.0 / (r2**1.5) - 3.0 * dy*dy / r5)
            uxy += c * (-3.0 * dx*dy / r5)
        # Rotating frame derivatives:
        # d/dx (ux + Ωy) = uxx ; d/dy(...) = uxy + Ω
        # d/dx (uy - Ωx) = uxy - Ω ; d/dy(...) = uyy
        return np.array([[uxx, uxy + self.omega],
                         [uxy - self.omega, uyy]])

    def newton_root(self, r0: np.ndarray, tol: float = 1e-12, itmax: int = 50) -> Tuple[np.ndarray, bool]:
        r = r0.astype(float)
        for _ in range(itmax):
            F = self.uOmega(r)
            if np.linalg.norm(F) < tol:
                return r, True
            Jm = self.J(r)
            try:
                step = np.linalg.solve(Jm, F)
            except np.linalg.LinAlgError:
                return r, False
            r = r - step
            if np.linalg.norm(step) < tol:
                return r, True
        return r, False

    def classify(self, rstar: np.ndarray) -> Stag:
        Jm = self.J(rstar)
        w, V = np.linalg.eig(Jm)
        label = 'node'
        if np.all(np.isreal(w)):
            label = 'saddle' if (w[0].real * w[1].real) < 0 else 'node'
        else:
            tr = np.trace(Jm).real
            label = 'center/spiral'
        return Stag(r=rstar, label=label, J=Jm, eigvals=w, eigvecs=V)

    def trace_manifold(self, stag: Stag, kind: str = 'unstable',
                       ds: float = 1e-3, max_steps: int = 20000,
                       rmax: float = 5.0) -> np.ndarray:
        # pick eigenvector with largest/smallest real part
        w = stag.eigvals
        V = stag.eigvecs
        idx = np.argmax(w.real) if kind == 'unstable' else np.argmin(w.real)
        v0 = V[:, idx].real
        v0 /= np.linalg.norm(v0) + 1e-18
        r = stag.r + 1e-6 * v0
        path = [r.copy()]
        for _ in range(max_steps):
            v = self.uOmega(r)
            n = np.linalg.norm(v)
            if n < 1e-14:  # stuck
                break
            r = r + ds * (v / n)  # approximate arc-length step
            path.append(r.copy())
            if np.linalg.norm(r) > rmax:
                break
        return np.array(path)

def seeds_two_body(earth: np.ndarray) -> List[np.ndarray]:
    # Heuristic seeds near L1/L2/L4/L5 for Sun–Earth only
    return [
        earth + np.array([-0.02, 0.0]),  # L1-ish
        earth + np.array([+0.02, 0.0]),  # L2-ish
        0.5 * earth + np.array([ 0.0,  +0.866]),  # L4-ish
        0.5 * earth + np.array([ 0.0,  -0.866]),  # L5-ish
    ]

def continue_stagnations(flow_factory: Callable[[float], VectorFlow],
                         phases: np.ndarray,
                         r0_list: List[np.ndarray],
                         tol: float = 1e-10) -> Dict[float, List[Stag]]:
    """Track stagnations across phases using previous solutions as seeds."""
    results: Dict[float, List[Stag]] = {}
    prev_roots = r0_list
    for ph in phases:
        vf = flow_factory(ph)
        stags = []
        for s in prev_roots:
            r, ok = vf.newton_root(s, tol=tol)
            if ok:
                stags.append(vf.classify(r))
        results[ph] = stags
        # use current roots as seeds for next phase
        prev_roots = [st.r for st in stags]
    return results


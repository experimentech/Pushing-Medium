import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict

# -------- Vector core --------
@dataclass
class Body:
    pos: np.ndarray
    gamma: float

class VectorFlow:
    def __init__(self, bodies: List[Body], omega: float, eps2: float = 1e-6):
        self.bodies, self.omega, self.eps2 = bodies, omega, eps2

    def u(self, r):
        v = np.zeros(2)
        for b in self.bodies:
            d = b.pos - r
            r2 = d@d + self.eps2
            v += (b.gamma / (r2**1.5)) * d
        return v

    def uOmega(self, r):
        ux, uy = self.u(r); x, y = r
        return np.array([ux + self.omega*y, uy - self.omega*x])

    def J(self, r):
        uxx = uxy = uyy = 0.0
        for b in self.bodies:
            d = b.pos - r; dx, dy = d
            r2 = dx*dx + dy*dy + self.eps2; r5 = r2**2.5; c = b.gamma
            uxx += c*(1.0/(r2**1.5) - 3*dx*dx/r5)
            uyy += c*(1.0/(r2**1.5) - 3*dy*dy/r5)
            uxy += c*(-3*dx*dy/r5)
        return np.array([[uxx, uxy + self.omega],[uxy - self.omega, uyy]])

    def newton_root(self, r0, tol=1e-12, itmax=50):
        r = r0.astype(float)
        for _ in range(itmax):
            F = self.uOmega(r)
            if np.linalg.norm(F) < tol: return r, True
            step = np.linalg.solve(self.J(r), F)
            r = r - step
            if np.linalg.norm(step) < tol: return r, True
        return r, False

def moon_pos(phase, earth_pos, r_em=0.00257):
    return earth_pos + r_em*np.array([np.cos(phase), np.sin(phase)])

def seeds_two_body(earth):
    return [earth+np.array([-0.02,0]), earth+np.array([0.02,0]),
            0.5*earth+np.array([0, +0.866]), 0.5*earth+np.array([0, -0.866])]

def classify(J):
    w = np.linalg.eigvals(J)
    if np.all(np.isreal(w)): return 'saddle' if w[0].real*w[1].real < 0 else 'node'
    return 'center'

def trace_manifold(vf, rstar, kind='unstable', ds=1e-3, steps=30000, rmax=5.0):
    w, V = np.linalg.eig(vf.J(rstar))
    idx = np.argmax(w.real) if kind=='unstable' else np.argmin(w.real)
    v0 = V[:,idx].real; v0 /= (np.linalg.norm(v0)+1e-18)
    r = rstar + 1e-6*v0; path=[r.copy()]
    for _ in range(steps):
        v = vf.uOmega(r); n=np.linalg.norm(v)
        if n<1e-14: break
        r = r + ds*(v/n)
        path.append(r.copy())
        if np.linalg.norm(r)>rmax: break
    return np.array(path)

# -------- Confidence ensemble (coarse scouts) --------
def coarse_grid_confidence(vf_factory, phase, jitters=12, Nx=300, Ny=220, speed_thresh=3e-3):
    x0, x1, y0, y1 = -2.0, 3.0, -2.0, 2.0
    hits = []  # stagnation detections
    for k in range(jitters):
        # jitter grid origin and eps/omega slightly
        jx = (np.random.rand()-0.5)*((x1-x0)/Nx)
        jy = (np.random.rand()-0.5)*((y1-y0)/Ny)
        eps_scale = np.random.uniform(0.85, 1.15)
        omega_scale = np.random.uniform(0.99, 1.01)
        vf = vf_factory(phase, eps_scale, omega_scale)
        x = np.linspace(x0+jx, x1+jx, Nx)
        y = np.linspace(y0+jy, y1+jy, Ny)
        X, Y = np.meshgrid(x, y)
        U = np.zeros_like(X); V = np.zeros_like(Y)
        for j in range(Ny):
            for i in range(Nx):
                U[j,i], V[j,i] = vf.uOmega(np.array([X[j,i], Y[j,i]]))
        S = np.hypot(U,V)
        # local minima mask (3x3) + threshold
        from scipy.ndimage import minimum_filter
        local_min = (S == minimum_filter(S, size=5))
        cand = np.argwhere((S < speed_thresh) & local_min)
        pts = np.array([[x[i], y[j]] for (j,i) in cand])
        hits.append(pts)
    # confidence via kernel density around hits
    if not any(len(h)>0 for h in hits): return np.zeros((0,3))
    all_pts = np.vstack([h for h in hits if len(h)>0])
    # simple mean-shift clustering
    from sklearn.cluster import MeanShift
    ms = MeanShift(bandwidth=0.02).fit(all_pts)
    centers = ms.cluster_centers_
    labels = ms.labels_
    # confidence = cluster frequency / jitters
    conf = []
    for c in range(len(centers)):
        freq = (labels == c).sum() / max(1,len(all_pts))
        conf.append([centers[c,0], centers[c,1], freq])
    return np.array(conf)

# -------- Keyframe + refinement driver --------
def make_flow_factory(omega_base=2*np.pi):
    SUN = Body(np.array([0.0,0.0]), 1000.0)
    EARTH = Body(np.array([1.0,0.0]), 1.0)
    def vf_factory(phase, eps_scale=1.0, omega_scale=1.0):
        MOON = Body(moon_pos(phase, EARTH.pos), 0.0123)
        vf = VectorFlow([SUN, EARTH, MOON], omega=omega_base*omega_scale, eps2=1e-6*eps_scale)
        return vf
    return vf_factory

def keyframe(phase, vf_factory):
    vf = vf_factory(phase)
    seeds = seeds_two_body(np.array([1.0,0.0])) + [np.array([0.6,0.2]), np.array([0.6,-0.2])]
    roots=[]
    for s in seeds:
        r, ok = vf.newton_root(s)
        if ok and all(np.linalg.norm(r - rr) > 1e-3 for rr,_ in roots):
            roots.append((r, classify(vf.J(r))))
    manifolds=[]
    for r,lbl in roots:
        if lbl=='saddle':
            manifolds.append(trace_manifold(vf, r, 'unstable'))
            manifolds.append(trace_manifold(vf, r, 'stable'))
    return dict(roots=roots, manifolds=manifolds)

def refine_from_confidence(vf_factory, phase, conf_pts, conf_low=0.3, conf_high=0.6):
    vf = vf_factory(phase)
    roots=[]
    for x,y,cf in conf_pts:
        if cf>=conf_high or (conf_low<=cf<conf_high):
            r, ok = vf.newton_root(np.array([x,y]))
            if ok and all(np.linalg.norm(r - rr) > 1e-4 for rr,_ in roots):
                roots.append((r, classify(vf.J(r))))
    manifolds=[]
    for r,lbl in roots:
        if lbl=='saddle':
            manifolds.append(trace_manifold(vf, r, 'unstable'))
            manifolds.append(trace_manifold(vf, r, 'stable'))
    return dict(roots=roots, manifolds=manifolds)

if __name__ == "__main__":
    np.random.seed(7)
    vf_factory = make_flow_factory()
    # I-frame keyframe at phase 30°
    phase = np.deg2rad(30.0)
    K = keyframe(phase, vf_factory)
    # Ensemble confidence (coarse scouts)
    C = coarse_grid_confidence(vf_factory, phase, jitters=10, Nx=260, Ny=180)
    # Refinement guided by confidence
    R = refine_from_confidence(vf_factory, phase, C)
    print("Keyframe roots:", [(r.tolist(), lbl) for r,lbl in K['roots']])
    print("Refined roots:", [(r.tolist(), lbl) for r,lbl in R['roots']])
    print("Manifolds traced (counts):", len(K['manifolds']), "→", len(R['manifolds']))


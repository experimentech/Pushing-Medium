# benchmark.py
import time
import numpy as np
from vector_skeleton import Body, VectorFlow, seeds_two_body, moon_pos, continue_stagnations
from grid_layer import grid_flow, grid_stagnations, hausdorff

def make_flow(phase: float, omega: float) -> VectorFlow:
    SUN = Body(pos=np.array([0.0, 0.0]), gamma=1000.0)
    EARTH = Body(pos=np.array([1.0, 0.0]), gamma=1.0)
    MOON = Body(pos=moon_pos(phase, EARTH.pos), gamma=0.0123)
    return VectorFlow([SUN, EARTH, MOON], omega=omega, eps2=1e-6)

def vector_skeleton_bench(phase: float, omega: float):
    vf = make_flow(phase, omega)
    # seeds: two-body heuristic plus rotated copies to catch off-axis points
    seeds = seeds_two_body(np.array([1.0, 0.0])) + [np.array([0.6, 0.2]), np.array([0.6, -0.2])]
    t0 = time.perf_counter()
    roots = []
    for s in seeds:
        r, ok = vf.newton_root(s)
        if ok:
            roots.append(vf.classify(r))
    # deduplicate close roots
    uniq = []
    for st in roots:
        if all(np.linalg.norm(st.r - u.r) > 1e-3 for u in uniq):
            uniq.append(st)
    # manifolds from saddles
    manifolds = []
    for st in uniq:
        if st.label == 'saddle':
            manifolds.append(vf.trace_manifold(st, 'unstable'))
            manifolds.append(vf.trace_manifold(st, 'stable'))
    t1 = time.perf_counter()
    return dict(time=t1-t0, count=len(uniq), stagnations=uniq, manifolds=manifolds)

def grid_skeleton_bench(phase: float, omega: float, Nx=700, Ny=500):
    vf = make_flow(phase, omega)
    x = np.linspace(-2.0, 3.0, Nx)
    y = np.linspace(-2.0, 2.0, Ny)
    t0 = time.perf_counter()
    U, V, S = grid_flow(vf, x, y)
    cand = grid_stagnations(U, V, S, speed_thresh=3e-3, sep=9)
    t1 = time.perf_counter()
    # map grid candidates to coordinates
    pts = np.array([[x[ix], y[iy]] for (ix, iy) in cand])
    return dict(time=t1-t0, points=pts, grid=(x, y, U, V, S))

def hybrid_bench(phase: float, omega: float, Nx=500, Ny=350):
    # vector seeds + coarse grid refinement near separatrices
    vres = vector_skeleton_bench(phase, omega)
    vf = make_flow(phase, omega)
    x = np.linspace(-2.0, 3.0, Nx)
    y = np.linspace(-2.0, 2.0, Ny)
    t0 = time.perf_counter()
    # sample only in bands around manifolds (tube of width w)
    w = 0.03
    mask = np.zeros((Ny, Nx), dtype=bool)
    X, Y = np.meshgrid(x, y)
    for path in vres['manifolds']:
        if path.shape[0] < 2:
            continue
        # distance to polyline (approx)
        for k in range(0, path.shape[0], 15):
            px, py = path[k]
            mask |= ((X - px)**2 + (Y - py)**2) <= (w*w)
    # compute flow only where mask True
    U = np.zeros((Ny, Nx)); V = np.zeros((Ny, Nx)); S = np.zeros((Ny, Nx))
    idxs = np.argwhere(mask)
    for (j, i) in idxs:
        U[j, i], V[j, i] = vf.uOmega(np.array([x[i], y[j]]))
        S[j, i] = np.hypot(U[j, i], V[j, i])
    t1 = time.perf_counter()
    return dict(time=t1-t0, sampled=int(mask.sum()), mask=mask, vector=vres)

def compare_vector_vs_grid(phase: float, omega: float):
    v = vector_skeleton_bench(phase, omega)
    g = grid_skeleton_bench(phase, omega)
    # Hausdorff distance between stagnation sets (vector vs grid)
    vpts = np.array([st.r for st in v['stagnations']])
    gpts = g['points'] if g['points'].size else np.zeros((0,2))
    hd = hausdorff(vpts, gpts) if (len(vpts) > 0 and len(gpts) > 0) else np.nan
    return dict(vector_time=v['time'], grid_time=g['time'],
                v_count=len(vpts), g_count=len(gpts), hausdorff=hd)

if __name__ == "__main__":
    omega = 2*np.pi
    phase = np.deg2rad(30.0)
    res = compare_vector_vs_grid(phase, omega)
    print("Vector vs Grid:", res)
    hyb = hybrid_bench(phase, omega)
    print("Hybrid sampling time (coarse grid near manifolds):", hyb['time'], "with cells:", hyb['sampled'])


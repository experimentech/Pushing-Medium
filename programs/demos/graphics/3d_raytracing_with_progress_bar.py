import numpy as np
import imageio.v2 as imageio
from tqdm import tqdm   # <-- progress bar

# ------------------------------
# 1) Camera and render settings
# ------------------------------
W, H = 256, 256
fov = 60.0 * np.pi/180.0
cam_pos = np.array([0.0, 0.0, -4.0])
cam_dir = np.array([0.0, 0.0, 1.0])
cam_up  = np.array([0.0, 1.0, 0.0])

max_steps = 800
ds = 0.01
hit_epsilon = 0.002
far_limit = 12.0
bg_color = np.array([0.03, 0.05, 0.08])

# ------------------------------
# 2) Pushing-medium refractive field
# ------------------------------
masses = [
    (np.array([-1.2, 0.5,  1.0]), 0.50),
    (np.array([ 1.0, -0.6, 1.5]), 0.35),
]
eps = 1e-4

def n_field(p):
    n = 1.0
    for c, mu in masses:
        rvec = p - c
        r = np.linalg.norm(rvec) + eps
        n += mu / r
    return n

def grad_ln_n(p):
    n = 1.0
    grad = np.zeros(3)
    for c, mu in masses:
        rvec = p - c
        r = np.linalg.norm(rvec) + eps
        n += mu / r
        grad += -mu * rvec / (r**3)
    return grad / max(n, 1e-8)

def project_perp(khat, v):
    return v - khat * np.dot(khat, v)

# ------------------------------
# 3) Scene: SDF geometry
# ------------------------------
def sd_sphere(p, r):
    return np.linalg.norm(p) - r

def sd_round_box(p, b, r):
    q = np.abs(p) - b
    return np.linalg.norm(np.maximum(q, 0.0)) + np.minimum(np.max(q), 0.0) - r

def scene_sdf(p):
    ps = p - np.array([-0.8, -0.1, 2.2])
    d1 = sd_sphere(ps, 0.7)
    pb = p - np.array([1.0, 0.2, 2.8])
    d2 = sd_round_box(pb, np.array([0.6, 0.4, 0.8]), 0.15)
    k = 0.25
    h = np.clip(0.5 + 0.5*(d2 - d1)/k, 0.0, 1.0)
    return (1-h)*d2 + h*d1 - k*h*(1-h)

def estimate_normal(p):
    e = 1e-3
    dx = np.array([e, 0, 0])
    dy = np.array([0, e, 0])
    dz = np.array([0, 0, e])
    n = np.array([
        scene_sdf(p + dx) - scene_sdf(p - dx),
        scene_sdf(p + dy) - scene_sdf(p - dy),
        scene_sdf(p + dz) - scene_sdf(p - dz)
    ])
    return n / max(np.linalg.norm(n), 1e-8)

# ------------------------------
# 4) Lighting/shading
# ------------------------------
light_pos = np.array([-2.5, 2.8, -1.5])
light_col = np.array([1.3, 1.2, 1.1])
ambient = 0.08
spec_power = 32.0
albedo_obj1 = np.array([0.8, 0.3, 0.2])
albedo_obj2 = np.array([0.2, 0.6, 0.9])

def which_object(p):
    ps = p - np.array([-0.8, -0.1, 2.2])
    d1 = sd_sphere(ps, 0.7)
    pb = p - np.array([1.0, 0.2, 2.8])
    d2 = sd_round_box(pb, np.array([0.6, 0.4, 0.8]), 0.15)
    return 1 if d1 < d2 else 2

def shade(p, vdir):
    n = estimate_normal(p)
    ldir = light_pos - p
    ldist = np.linalg.norm(ldir)
    ldir /= max(ldist, 1e-8)
    diff = max(np.dot(n, ldir), 0.0)
    h = (ldir - vdir); h /= max(np.linalg.norm(h), 1e-8)
    spec = max(np.dot(n, h), 0.0) ** spec_power
    att = 1.0 / (0.2 + 0.02*ldist + 0.01*ldist*ldist)
    base = albedo_obj1 if which_object(p) == 1 else albedo_obj2
    return np.clip(ambient*base + att*(diff*base + 0.3*spec*light_col), 0, 1)

# ------------------------------
# 5) Primary ray integration
# ------------------------------
def render_pixel(ix, iy):
    cw = cam_dir / np.linalg.norm(cam_dir)
    cu = np.cross(cw, cam_up); cu /= np.linalg.norm(cu)
    cv = np.cross(cu, cw)
    px = ((ix + 0.5)/W * 2 - 1) * np.tan(fov*0.5) * (W/H)
    py = ((iy + 0.5)/H * 2 - 1) * np.tan(fov*0.5)
    k = (cu*px + cv*py + cw)
    k /= np.linalg.norm(k)
    p = cam_pos.copy()

    for _ in range(max_steps):
        d = scene_sdf(p)
        if d < hit_epsilon:
            return shade(p, k)
        if np.linalg.norm(p) > far_limit:
            return bg_color
        g_perp = project_perp(k, grad_ln_n(p))
        k = k + g_perp * ds
        k /= max(np.linalg.norm(k), 1e-8)
        p = p + k * (ds / max(n_field(p), 1e-6))
    return bg_color

# ------------------------------
# 6) Render loop with progress bar
# ------------------------------
img = np.zeros((H, W, 3), dtype=np.float32)
for iy in tqdm(range(H), desc="Rendering", unit="row"):
    for ix in range(W):
        img[iy, ix] = render_pixel(ix, iy)

# Save
img = np.clip(img, 0, 1) ** (1.0/2.2)
imageio.imwrite("pm_raytrace_3d.png", (img*255).astype(np.uint8))
print("Saved pm_raytrace_3d.png")


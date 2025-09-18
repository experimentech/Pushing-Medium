import numpy as np
import matplotlib.pyplot as plt

# Scene setup
W, H = 128, 128
plane_z = 0.0
light_pos = np.array([0.0, 0.8, 0.5])
light_col = np.array([1.0, 0.9, 0.7])

# Simple PM refractive index field: two Gaussian wells
def n_field(p):
    wells = [
        (np.array([-0.3, 0.0, 0.3]), 0.3, 0.15),
        (np.array([ 0.4, 0.0, 0.6]), 0.25, 0.12)
    ]
    n = 1.0
    for c, amp, sigma in wells:
        r2 = np.sum((p - c)**2)
        n += amp * np.exp(-r2/(2*sigma**2))
    return n

def grad_ln_n(p, eps=1e-3):
    base = n_field(p)
    grad = np.zeros(3)
    for i in range(3):
        dp = np.zeros(3); dp[i] = eps
        grad[i] = (n_field(p+dp) - n_field(p-dp)) / (2*eps)
    return grad / max(base, 1e-8)

# Simple flow field (vortex)
def flow_field(p):
    return np.array([-p[1], p[0], 0.0]) * 0.2

# Geometry: two boxes on plane
def hit_scene(p):
    # plane
    if abs(p[2] - plane_z) < 1e-3:
        return True, np.array([0,0,1])
    # box 1
    if -0.5 < p[0] < -0.2 and 0 < p[1] < 0.4 and 0 <= p[2] <= 0.4:
        return True, np.array([0,0,1])
    # box 2
    if 0.2 < p[0] < 0.5 and 0 < p[1] < 0.4 and 0 <= p[2] <= 0.4:
        return True, np.array([0,0,1])
    return False, None

# Photon tracing
lightmap = np.zeros((H, W, 3), dtype=np.float32)
n_photons = 20000
max_bounces = 3
ds = 0.02

for _ in range(n_photons):
    # random initial direction
    theta = np.random.uniform(0, 2*np.pi)
    phi = np.random.uniform(0, np.pi/2)
    dir = np.array([np.cos(theta)*np.sin(phi),
                    np.sin(theta)*np.sin(phi),
                    np.cos(phi)])
    pos = light_pos.copy()
    col = light_col.copy()

    for bounce in range(max_bounces):
        for step in range(500):
            # bend by PM gradient
            g_perp = grad_ln_n(pos) - dir*np.dot(dir, grad_ln_n(pos))
            dir += g_perp * ds
            dir += flow_field(pos) * ds  # advect
            dir /= np.linalg.norm(dir)
            pos += dir * ds / n_field(pos)

            hit, normal = hit_scene(pos)
            if hit:
                # project to image plane (top-down view)
                ix = int((pos[0] + 1) * 0.5 * W)
                iy = int((pos[1] + 1) * 0.5 * H)
                if 0 <= ix < W and 0 <= iy < H:
                    lightmap[iy, ix] += col
                # diffuse bounce
                theta = np.random.uniform(0, 2*np.pi)
                phi = np.random.uniform(0, np.pi/2)
                dir = np.array([np.cos(theta)*np.sin(phi),
                                np.sin(theta)*np.sin(phi),
                                np.cos(phi)])
                pos += normal * 1e-3
                col *= 0.6  # energy loss
                break

# Visualise
plt.imshow(np.clip(lightmap / lightmap.max(), 0, 1)**(1/2.2))
plt.title("PM radiosity-style photon map (top-down)")
plt.axis('off')
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import imageio.v2 as imageio

# Scene setup
W, H = 128, 128
plane_z = 0.0
light_pos = np.array([0.0, 0.8, 0.5])
light_col = np.array([1.0, 0.9, 0.7])

# PM refractive index field: two Gaussian wells
def n_field(p, t):
    wells = [
        (np.array([-0.3 + 0.1*np.sin(0.5*t), 0.0, 0.3]), 0.3, 0.15),
        (np.array([ 0.4 + 0.1*np.cos(0.3*t), 0.0, 0.6]), 0.25, 0.12)
    ]
    n = 1.0
    for c, amp, sigma in wells:
        r2 = np.sum((p - c)**2)
        n += amp * np.exp(-r2/(2*sigma**2))
    return n

def grad_ln_n(p, t, eps=1e-3):
    base = n_field(p, t)
    grad = np.zeros(3)
    for i in range(3):
        dp = np.zeros(3); dp[i] = eps
        grad[i] = (n_field(p+dp, t) - n_field(p-dp, t)) / (2*eps)
    return grad / max(base, 1e-8)

# Flow field (rotating vortex)
def flow_field(p, t):
    return np.array([-p[1], p[0], 0.0]) * (0.2 + 0.1*np.sin(0.2*t))

# Geometry: two boxes on plane
def hit_scene(p):
    if abs(p[2] - plane_z) < 1e-3:
        return True, np.array([0,0,1])
    if -0.5 < p[0] < -0.2 and 0 < p[1] < 0.4 and 0 <= p[2] <= 0.4:
        return True, np.array([0,0,1])
    if 0.2 < p[0] < 0.5 and 0 < p[1] < 0.4 and 0 <= p[2] <= 0.4:
        return True, np.array([0,0,1])
    return False, None

# Animation parameters
n_photons = 5000
max_bounces = 3
ds = 0.02
n_frames = 20

frames = []

for frame in tqdm(range(n_frames), desc="Animating PM radiosity", unit="frame"):
    t = frame * 0.5  # time parameter
    lightmap = np.zeros((H, W, 3), dtype=np.float32)

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
                g_perp = grad_ln_n(pos, t) - dir*np.dot(dir, grad_ln_n(pos, t))
                dir += g_perp * ds
                dir += flow_field(pos, t) * ds
                dir /= np.linalg.norm(dir)
                pos += dir * ds / n_field(pos, t)

                hit, normal = hit_scene(pos)
                if hit:
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
                    col *= 0.6
                    break

    img = np.clip(lightmap / lightmap.max(), 0, 1)**(1/2.2)
    frames.append((img*255).astype(np.uint8))

# Save animation
imageio.mimsave("pm_radiosity_anim.gif", frames, fps=5)
print("Saved pm_radiosity_anim.gif")


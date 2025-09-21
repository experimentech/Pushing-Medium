#!/usr/bin/env python3

"""
Torch-accelerated PM radiosity-style photon demo (GPU-ready).

Overview
- Emits photons from a point light, bends via PM gradient (optional),
  does a few diffuse bounces, and accumulates energy into the camera image
  by projecting hit points to screen coordinates.
- Vectorized and chunked to handle many photons efficiently.

This is a simple photon map visualizer, not a full GI solution.
"""
from __future__ import annotations

import math
import argparse
import os
import torch
import torch.nn.functional as F
import imageio.v2 as imageio


def normalize(v, eps=1e-12):
    return v / (v.norm(dim=-1, keepdim=True) + eps)


class PMField:
    def __init__(self, centers, mus, eps=1e-4, device='cpu', dtype=torch.float32):
        self.centers = torch.tensor(centers, device=device, dtype=dtype)
        self.mus = torch.tensor(mus, device=device, dtype=dtype)
        self.eps = torch.tensor(eps, device=device, dtype=dtype)

    def n(self, p: torch.Tensor) -> torch.Tensor:
        rvec = p[..., None, :] - self.centers
        r = rvec.norm(dim=-1) + self.eps
        return 1.0 + (self.mus / r).sum(dim=-1)

    def grad_ln_n(self, p: torch.Tensor) -> torch.Tensor:
        rvec = p[..., None, :] - self.centers
        r = rvec.norm(dim=-1) + self.eps
        inv_r3 = (self.mus / (r**3))[..., None]
        grad_n = (-inv_r3 * rvec).sum(dim=-2)
        n = 1.0 + (self.mus / r).sum(dim=-1, keepdim=True)
        return grad_n / (n + 1e-12)


def sd_sphere(p: torch.Tensor, r: float) -> torch.Tensor:
    return p.norm(dim=-1) - r


def sd_round_box(p: torch.Tensor, b: torch.Tensor, r: float) -> torch.Tensor:
    q = p.abs() - b
    return torch.linalg.vector_norm(torch.clamp(q, min=0.0), dim=-1) + torch.clamp(q.max(dim=-1).values, max=0.0) - r


def scene_sdf_and_material(p: torch.Tensor):
    # sphere
    ps = p - torch.tensor([-0.8, -0.1, 2.2], device=p.device, dtype=p.dtype)
    d1 = sd_sphere(ps, 0.7)
    # rounded box
    pb = p - torch.tensor([1.0, 0.2, 2.8], device=p.device, dtype=p.dtype)
    d2 = sd_round_box(pb, torch.tensor([0.6, 0.4, 0.8], device=p.device, dtype=p.dtype), 0.15)
    # smooth min objects
    k = 0.25
    h = (0.5 + 0.5 * (d2 - d1) / k).clamp(0.0, 1.0)
    d_obj = (1 - h) * d2 + h * d1 - k * h * (1 - h)
    # plane y = -1
    d_plane = p[..., 1] + 1.0
    d = torch.minimum(d_obj, d_plane)
    # material id: 0 plane, 1 sphere, 2 box
    is_plane = (d_plane <= d_obj)
    mat = torch.zeros_like(d, dtype=torch.int64)
    mat[~is_plane & (d1 < d2)] = 1
    mat[~is_plane & (d2 <= d1)] = 2
    return d, mat


def estimate_normal(p: torch.Tensor) -> torch.Tensor:
    e = 1e-3
    dx = torch.tensor([e, 0, 0], device=p.device, dtype=p.dtype)
    dy = torch.tensor([0, e, 0], device=p.device, dtype=p.dtype)
    dz = torch.tensor([0, 0, e], device=p.device, dtype=p.dtype)
    nx = scene_sdf_and_material(p + dx)[0] - scene_sdf_and_material(p - dx)[0]
    ny = scene_sdf_and_material(p + dy)[0] - scene_sdf_and_material(p - dy)[0]
    nz = scene_sdf_and_material(p + dz)[0] - scene_sdf_and_material(p - dz)[0]
    n = torch.stack([nx, ny, nz], dim=-1)
    return normalize(n)


def material_albedo(p: torch.Tensor, mat: torch.Tensor) -> torch.Tensor:
    """Return per-hit RGB albedo based on material id.
    mat: 0 plane, 1 sphere, 2 box.
    Plane uses a checker grid in x,z; objects use fixed colors.
    """
    albedo_obj1 = torch.tensor([0.8, 0.3, 0.2], device=p.device, dtype=p.dtype)
    albedo_obj2 = torch.tensor([0.2, 0.6, 0.9], device=p.device, dtype=p.dtype)
    # Plane checker
    grid_u = torch.floor((p[..., 0] * 3.0) % 2.0)
    grid_v = torch.floor((p[..., 2] * 3.0) % 2.0)
    grid = ((grid_u + grid_v) % 2.0)
    plane_b = 0.2 + 0.6 * (1.0 - grid)  # alternate bright/dark
    plane_rgb = torch.stack([plane_b, plane_b * 0.9, plane_b * 0.7], dim=-1)
    # Collect
    out = torch.zeros_like(plane_rgb)
    is_plane = (mat == 0)
    is_sph = (mat == 1)
    is_box = (mat == 2)
    if is_plane.any():
        out[is_plane] = plane_rgb[is_plane]
    if is_sph.any():
        out[is_sph] = albedo_obj1
    if is_box.any():
        out[is_box] = albedo_obj2
    return out


def project_to_screen(points, cam_pos, cu, cv, cw, fov_rad, W, H, device, dtype):
    # Camera space components
    v = points - cam_pos[None, :]
    x = (v * cu[None, :]).sum(dim=-1)
    y = (v * cv[None, :]).sum(dim=-1)
    z = (v * cw[None, :]).sum(dim=-1)
    # Perspective divide
    eps = torch.tensor(1e-6, device=device, dtype=dtype)
    tan = math.tan(0.5 * fov_rad)
    px = x / (z + eps)
    py = y / (z + eps)
    sx = (px / (tan * (W / H)) + 1.0) * 0.5 * W
    sy = (py / tan + 1.0) * 0.5 * H
    return sx, sy, z


def make_camera_std(W, H, fov_deg, cam_pos, cam_dir, cam_up, device, dtype):
    fov = math.radians(fov_deg)
    cam_pos = torch.tensor(cam_pos, device=device, dtype=dtype)
    cam_dir = torch.tensor(cam_dir, device=device, dtype=dtype)
    cam_up = torch.tensor(cam_up, device=device, dtype=dtype)
    cw = normalize(cam_dir)
    cu = normalize(torch.cross(cam_up, cw))
    cv = torch.cross(cw, cu)
    return cam_pos, cu, cv, cw, fov


def hemisphere_cosine(n, count):
    # cosine-weighted hemisphere around normal n
    u1 = torch.rand(count, device=n.device, dtype=n.dtype)
    u2 = torch.rand(count, device=n.device, dtype=n.dtype)
    r = torch.sqrt(u1)
    theta = 2 * math.pi * u2
    local = torch.stack([r * torch.cos(theta), r * torch.sin(theta), torch.sqrt(1 - u1)], dim=-1)
    # build orthonormal basis from n
    w = normalize(n)
    a = torch.tensor([0.0, 1.0, 0.0], device=n.device, dtype=n.dtype)
    u = normalize(torch.cross(a.expand_as(w), w))
    v = torch.cross(w, u)
    return normalize(u * local[..., :1] + v * local[..., 1:2] + w * local[..., 2:3])


def sky_background(dir: torch.Tensor, device, dtype):
    t = (dir[..., 1] * 0.5 + 0.5).clamp(0, 1)
    sky_top = torch.tensor([0.1, 0.2, 0.4], device=device, dtype=dtype)
    sky_bot = torch.tensor([0.02, 0.04, 0.08], device=device, dtype=dtype)
    grad = t[..., None] * sky_top + (1 - t[..., None]) * sky_bot
    h = (dir * torch.tensor([12.9898, 78.233, 45.164], device=device, dtype=dtype)).sum(dim=-1)
    s = torch.frac(torch.sin(h) * 43758.5453)
    stars = (s > 0.996).to(dtype)[..., None] * 0.8
    return (grad + stars).clamp(0, 1)


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(description="Torch PM Radiosity Demo")
    ap.add_argument('--width', type=int, default=512)
    ap.add_argument('--height', type=int, default=512)
    ap.add_argument('--fov', type=float, default=60.0)
    ap.add_argument('--photons', type=int, default=40000)
    ap.add_argument('--bounces', type=int, default=2)
    ap.add_argument('--steps', type=int, default=400)
    ap.add_argument('--ds', type=float, default=0.01)
    ap.add_argument('--hit-eps', type=float, default=0.01)
    ap.add_argument('--far', type=float, default=20.0)
    ap.add_argument('--bend-strength', type=float, default=0.5)
    ap.add_argument('--no-bend', action='store_true')
    ap.add_argument('--chunk', type=int, default=65536)
    ap.add_argument('--light-pos', type=str, default='-2.5,2.8,-1.5', help='Light position x,y,z')
    ap.add_argument('--light-dir', type=str, default='0,-0.6,1.0', help='Initial photon direction (x,y,z), normalized with jitter')
    ap.add_argument('--cam-pos', type=str, default=None, help='Camera position x,y,z')
    ap.add_argument('--cam-dir', type=str, default=None, help='Camera forward x,y,z')
    ap.add_argument('--accum', type=str, default='camera', choices=['camera','topdown'], help='How to accumulate photon hits')
    ap.add_argument('--x-range', type=str, default='-3,3', help='Top-down x range (min,max)')
    ap.add_argument('--z-range', type=str, default='0,6', help='Top-down z range (min,max)')
    ap.add_argument('--rotate', type=str, default='none', choices=['none','90ccw','90cw','180'], help='Rotate final image for presentation')
    ap.add_argument('--flip-x', action='store_true', help='Mirror final image horizontally')
    ap.add_argument('--flip-y', action='store_true', help='Mirror final image vertically')
    ap.add_argument('--albedo-bleed', action='store_true', help='Tint bounced photon power by surface albedo')
    ap.add_argument('--filter-sigma', type=float, default=0.0, help='Optional Gaussian blur sigma (pixels) on the photon map')
    ap.add_argument('--out', type=str, default='pm_radiosity_torch.png')
    ap.add_argument('--device', type=str, default='auto', choices=['auto','cpu','cuda'])
    ap.add_argument('--dtype', type=str, default='fp32', choices=['fp16','fp32'])
    args = ap.parse_args()

    dev = 'cuda' if (args.device == 'cuda' or (args.device == 'auto' and torch.cuda.is_available())) else 'cpu'
    dtype = torch.float16 if args.dtype == 'fp16' and dev == 'cuda' else torch.float32

    # Resolve output path: if user passed a bare filename (no directory),
    # save alongside this script to avoid cluttering the workspace root.
    script_dir = os.path.dirname(__file__)
    out_arg = args.out or 'pm_radiosity_torch.png'
    if os.path.isabs(out_arg) or os.path.dirname(out_arg):
        out_path = out_arg
    else:
        out_path = os.path.join(script_dir, out_arg)

    # PM field (same as other demo)
    pm = PMField([(-1.2, 0.5, 1.0), (1.0, -0.6, 1.5)], [0.50, 0.35], device=dev, dtype=dtype)

    # Camera for projection
    cam_p = (0.0, 0.3, -4.0) if args.cam_pos is None else tuple(map(float, args.cam_pos.split(',')))
    cam_d = (0.0, -0.05, 1.0) if args.cam_dir is None else tuple(map(float, args.cam_dir.split(',')))
    cam_pos, cu, cv, cw, fov = make_camera_std(args.width, args.height, args.fov, cam_p, cam_d, (0.0, 1.0, 0.0), dev, dtype)

    # Light
    lp = tuple(map(float, args.light_pos.split(',')))
    light_pos = torch.tensor(lp, device=dev, dtype=dtype)
    light_col = torch.tensor([1.0, 0.9, 0.7], device=dev, dtype=dtype)

    # Accumulator image
    H, W = args.height, args.width
    img = torch.zeros(H, W, 3, device=dev, dtype=dtype)
    if args.accum == 'topdown':
        xr_str = getattr(args, 'x_range')
        zr_str = getattr(args, 'z_range')
        xmin, xmax = map(float, xr_str.split(','))
        zmin, zmax = map(float, zr_str.split(','))
        xr = xmax - xmin
        zr = zmax - zmin

    # Photon simulation in chunks
    remaining = args.photons
    total_hits = 0
    while remaining > 0:
        n0 = min(remaining, args.chunk)
        remaining -= n0

        # Sample initial directions (cone around user-provided light direction)
        ld = tuple(map(float, args.light_dir.split(',')))
        base_dir = normalize(torch.tensor(ld, device=dev, dtype=dtype))[None, :].expand(n0, -1)
        jitter = normalize(base_dir + 0.2 * torch.randn(n0, 3, device=dev, dtype=dtype))
        pos = light_pos[None, :].expand(n0, -1).clone()
        direc = jitter.clone()
        power = light_col[None, :].expand(n0, -1).clone()

        # Trace photons over bounces
        hits_this_chunk = 0  # accumulate hits across all bounces for this chunk
        b = 0
        while b <= args.bounces and pos.shape[0] > 0:
            n = pos.shape[0]
            active = torch.ones(n, dtype=torch.bool, device=dev)
            pi = pos.clone()
            ki = direc.clone()
            powi = power.clone()

            # Prepare next-bounce containers
            next_pos = []
            next_dir = []
            next_pow = []

            for s in range(args.steps):
                if not active.any():
                    break
                ai = active
                p_prev = pi.clone()
                # Bend
                if not args.no_bend and args.bend_strength > 0:
                    g = pm.grad_ln_n(pi[ai])
                    g_perp = g - (ki[ai] * (ki[ai] * g).sum(dim=-1, keepdim=True))
                    ki[ai] = normalize(ki[ai] + args.ds * args.bend_strength * g_perp)
                # Advance
                nloc = pm.n(pi[ai])[..., None]
                pi[ai] = pi[ai] + ki[ai] * (args.ds / (nloc + 1e-12))

                # Hit test (SDF threshold for objects/plane)
                d, mat_ids = scene_sdf_and_material(pi[ai])
                hit = d < args.hit_eps

                # Robust plane crossing: detect sign change across y=-1
                plane_prev = p_prev[ai][..., 1] + 1.0
                plane_curr = pi[ai][..., 1] + 1.0
                crossed = (plane_prev > 0) & (plane_curr <= 0)

                any_hitlike = hit | crossed
                if any_hitlike.any():
                    # Prepare hit points from threshold hits
                    hp_thresh = pi[ai][hit]
                    pw_thresh = powi[ai][hit]
                    mat_thresh = mat_ids[hit]
                    # Prepare plane-crossing hits by interpolation
                    hp_plane = None
                    pw_plane = None
                    mat_plane = None
                    if crossed.any():
                        t = (-(plane_prev[crossed])) / (plane_curr[crossed] - plane_prev[crossed] + 1e-12)
                        p0 = p_prev[ai][crossed]
                        p1 = pi[ai][crossed]
                        hp_plane = p0 + t[..., None] * (p1 - p0)
                        pw_plane = powi[ai][crossed]
                        mat_plane = torch.zeros(hp_plane.shape[0], dtype=torch.int64, device=hp_plane.device)

                    if hp_plane is not None and hp_plane.shape[0] > 0:
                        hit_pts = torch.cat([hp_thresh, hp_plane], dim=0) if hp_thresh.shape[0] > 0 else hp_plane
                        hit_pow = torch.cat([pw_thresh, pw_plane], dim=0) if hp_thresh.shape[0] > 0 else pw_plane
                        mat_all = torch.cat([mat_thresh, mat_plane], dim=0) if hp_thresh.shape[0] > 0 else mat_plane
                    else:
                        hit_pts = hp_thresh
                        hit_pow = pw_thresh
                        mat_all = mat_thresh

                    hits_added = 0
                    if args.accum == 'camera':
                        # Project to screen and accumulate (z>0 only)
                        sx, sy, z = project_to_screen(hit_pts, cam_pos, cu, cv, cw, fov, W, H, dev, dtype)
                        front = z > 0
                        if front.any():
                            ix = sx[front].long().clamp(0, W-1)
                            iy = sy[front].long().clamp(0, H-1)
                            img[iy, ix] += hit_pow[front]
                            hits_added = int(front.count_nonzero().item())
                    else:
                        # Top-down accumulation: map x,z within the specified ranges
                        xw = hit_pts[..., 0]
                        zw = hit_pts[..., 2]
                        in_range = (xw >= xmin) & (xw <= xmax) & (zw >= zmin) & (zw <= zmax)
                        if in_range.any():
                            xw = xw[in_range]
                            zw = zw[in_range]
                            hp = hit_pow[in_range]
                            ix = (((xw - xmin) / max(xr, 1e-9)) * (W - 1)).long().clamp(0, W-1)
                            iy = (((zw - zmin) / max(zr, 1e-9)) * (H - 1)).long().clamp(0, H-1)
                            img[iy, ix] += hp
                            hits_added = int(ix.numel())
                        else:
                            hits_added = 0
                    hits_this_chunk += hits_added

                    # Prepare bounce photons (if more bounces remain)
                    if b < args.bounces:
                        nrm = estimate_normal(hit_pts)
                        new_dir = hemisphere_cosine(nrm, hit_pts.shape[0])
                        new_pos = hit_pts + nrm * 1e-3
                        new_pow = hit_pow * 0.6
                        if args.albedo_bleed:
                            alb = material_albedo(hit_pts, mat_all)
                            new_pow = new_pow * alb
                        next_pos.append(new_pos)
                        next_dir.append(new_dir)
                        next_pow.append(new_pow)

                    # Deactivate those that hit for the rest of this bounce
                    sub_idx = torch.where(ai)[0]
                    active[sub_idx[any_hitlike]] = False

                # Far termination
                far = pi[ai].norm(dim=-1) > args.far
                if far.any():
                    sub_idx = torch.where(ai)[0]
                    active[sub_idx[far]] = False

            # Prepare for next bounce
            if b < args.bounces and len(next_pos) > 0:
                pos = torch.cat(next_pos, dim=0)
                direc = torch.cat(next_dir, dim=0)
                power = torch.cat(next_pow, dim=0)
                b += 1
            else:
                break
        total_hits += hits_this_chunk
        print(f"Chunk photons: {n0}, accumulated hits in chunk: {hits_this_chunk}")

    # Tone and save
    # Optional blur
    if args.filter_sigma and args.filter_sigma > 0:
        # approximate Gaussian via separable conv with kernel size ~ 6*sigma+1
        sigma = float(args.filter_sigma)
        ksize = max(3, int(6 * sigma) | 1)  # odd
        x = torch.linspace(- (ksize // 2), (ksize // 2), ksize, device=dev, dtype=dtype)
        g = torch.exp(-0.5 * (x / max(sigma, 1e-6))**2)
        g = (g / g.sum()).view(1, 1, 1, -1)  # horizontal
        img_b = img.permute(2, 0, 1).unsqueeze(0)  # [1,3,H,W]
        pad = ksize // 2
        img_b = F.pad(img_b, (pad, pad, 0, 0), mode='reflect')
        img_b = F.conv2d(img_b, g.expand(3, 1, 1, -1), groups=3)
        gT = g.transpose(-1, -2)
        img_b = F.pad(img_b, (0, 0, pad, pad), mode='reflect')
        img_b = F.conv2d(img_b, gT.expand(3, 1, -1, 1), groups=3)
        img = img_b.squeeze(0).permute(1, 2, 0).contiguous()

    img_np = (img / (img.max() + 1e-8)).clamp(0, 1) ** (1/2.2)
    img_np = img_np.cpu().numpy()
    if args.rotate != 'none':
        if args.rotate == '90ccw':
            img_np = img_np.transpose(1,0,2)[::-1,:,:]
        elif args.rotate == '90cw':
            img_np = img_np.transpose(1,0,2)[:,::-1,:]
        elif args.rotate == '180':
            img_np = img_np[::-1, ::-1, :]
    if args.flip_x:
        img_np = img_np[:, ::-1, :]
    if args.flip_y:
        img_np = img_np[::-1, :, :]
    imageio.imwrite(out_path, (img_np * 255).astype('uint8'))
    print(f"Saved {out_path} (total hits: {total_hits})")


if __name__ == '__main__':
    main()

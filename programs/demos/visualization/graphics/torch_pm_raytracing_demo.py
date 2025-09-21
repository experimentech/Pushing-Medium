#!/usr/bin/env python3
 
"""
Torch-accelerated PM ray tracing demo (vectorized, GPU-ready).

Highlights
- All pixels traced in parallel as tensors (optionally in chunks to fit memory).
- Curved rays through a pushing-medium index field n(x) with analytic ∇ ln n.
- Simple SDF scene (sphere + rounded box) and lightweight shading.
- Runs on CUDA if available; otherwise CPU.

This aims for speed and clarity first; visuals can be improved later.
"""
from __future__ import annotations

import math
import argparse
import os
from typing import Tuple

import torch
import imageio.v2 as imageio


# ------------------------------
# 1) Camera and render settings
# ------------------------------
def make_camera(W: int, H: int, fov_deg: float, cam_pos, cam_dir, cam_up, device, dtype, orient: str = 'standard'):
    fov = math.radians(fov_deg)
    cam_pos = torch.tensor(cam_pos, device=device, dtype=dtype)
    cam_dir = torch.tensor(cam_dir, device=device, dtype=dtype)
    cam_up = torch.tensor(cam_up, device=device, dtype=dtype)

    if orient == 'standard':
       # Conventional camera: floor at bottom (image y downwards)
       cw = cam_dir / (cam_dir.norm() + 1e-12)               # forward
       cu = torch.cross(cam_up, cw); cu = cu / (cu.norm() + 1e-12)  # right
       cv = torch.cross(cw, cu)                               # up

       xs = torch.arange(W, device=device, dtype=dtype)
       ys = torch.arange(H, device=device, dtype=dtype)
       grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')  # [H,W]
       px = ((grid_x + 0.5) / W * 2 - 1) * math.tan(0.5 * fov) * (W / H)
       py = ((grid_y + 0.5) / H * 2 - 1) * math.tan(0.5 * fov)

       k = (cu[None, None, :] * px[..., None] -
           cv[None, None, :] * py[..., None] +
           cw[None, None, :])  # [H,W,3]
       k = k / (k.norm(dim=-1, keepdim=True) + 1e-12)
       k = k.contiguous()
       p = cam_pos[None, None, :].expand(H, W, 3).contiguous()
       return p, k, cam_pos, cw, cu, cv

    # Legacy orientation (matches earlier outputs)
    cw = cam_dir / (cam_dir.norm() + 1e-12)
    cu = torch.cross(cw, cam_up); cu = cu / (cu.norm() + 1e-12)
    cv = torch.cross(cu, cw)

    i = torch.arange(W, device=device, dtype=dtype)
    j = torch.arange(H, device=device, dtype=dtype)
    px = ((i + 0.5) / W * 2 - 1) * math.tan(0.5 * fov) * (W / H)
    py = ((j + 0.5) / H * 2 - 1) * math.tan(0.5 * fov)
    grid_x, grid_y = torch.meshgrid(px, py, indexing='xy')  # [W,H]

    k = (cu[None, None, :] * grid_x[..., None] +
        cv[None, None, :] * grid_y[..., None] +
        cw[None, None, :])  # [W,H,3]
    k = k / (k.norm(dim=-1, keepdim=True) + 1e-12)
    k = k.contiguous()
    p = cam_pos[None, None, :].expand(W, H, 3).contiguous()
    return p, k, cam_pos, cw, cu, cv


# ---------------------------------
# 2) Pushing-medium refractive field
#     n = 1 + sum(mu_i / r_i), with analytic ∇ ln n
# ---------------------------------
class PMField:
    def __init__(self, centers, mus, eps=1e-4, device='cpu', dtype=torch.float32):
        self.centers = torch.tensor(centers, device=device, dtype=dtype)  # [M,3]
        self.mus = torch.tensor(mus, device=device, dtype=dtype)          # [M]
        self.eps = torch.tensor(eps, device=device, dtype=dtype)

    def n(self, p: torch.Tensor) -> torch.Tensor:
        # p: [...,3]
        rvec = p[..., None, :] - self.centers  # [...,M,3]
        r = rvec.norm(dim=-1) + self.eps       # [...,M]
        n = 1.0 + (self.mus / r).sum(dim=-1)   # [...]
        return n

    def grad_ln_n(self, p: torch.Tensor) -> torch.Tensor:
        rvec = p[..., None, :] - self.centers     # [...,M,3]
        r = rvec.norm(dim=-1) + self.eps          # [...,M]
        # grad n = sum(-mu * rvec / r^3)
        inv_r3 = (self.mus / (r**3))[..., None]   # [...,M,1]
        grad_n = (-inv_r3 * rvec).sum(dim=-2)     # [...,3]
        n = 1.0 + (self.mus / r).sum(dim=-1, keepdim=True)  # [...,1]
        return grad_n / (n + 1e-12)


def project_perp(khat: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    # khat, v: [...,3]
    return v - khat * (khat * v).sum(dim=-1, keepdim=True)


# ---------------------------------
# 3) Scene: SDF geometry (torch)
# ---------------------------------
def sd_sphere(p: torch.Tensor, r: float) -> torch.Tensor:
    return p.norm(dim=-1) - r


def sd_round_box(p: torch.Tensor, b: torch.Tensor, r: float) -> torch.Tensor:
    # rounded box: distance to box with corner radius r
    q = p.abs() - b
    # max(q,0) in L2, plus min(max(q),0) - r
    return torch.linalg.vector_norm(torch.clamp(q, min=0.0), dim=-1) + torch.clamp(q.max(dim=-1).values, max=0.0) - r


def scene_sdf(p: torch.Tensor, include_plane: bool = True) -> torch.Tensor:
    # Sphere centered at (-0.8, -0.1, 2.2), radius 0.7
    ps = p - torch.tensor([-0.8, -0.1, 2.2], device=p.device, dtype=p.dtype)
    d1 = sd_sphere(ps, 0.7)
    # Rounded box centered at (1.0, 0.2, 2.8), size (0.6,0.4,0.8), radius 0.15
    pb = p - torch.tensor([1.0, 0.2, 2.8], device=p.device, dtype=p.dtype)
    d2 = sd_round_box(pb, torch.tensor([0.6, 0.4, 0.8], device=p.device, dtype=p.dtype), 0.15)
    # Ground plane y = -1 (distance to plane is p.y + 1)
    d_plane = p[...,1] + 1.0
    # Smooth min
    k = 0.25
    h = (0.5 + 0.5 * (d2 - d1) / k).clamp(0.0, 1.0)
    d = (1 - h) * d2 + h * d1 - k * h * (1 - h)
    # Combine with plane via standard min (no smoothing for crisp contact)
    if include_plane:
        return torch.minimum(d, d_plane)
    else:
        return d


def estimate_normal(p: torch.Tensor, include_plane: bool = True) -> torch.Tensor:
    # Finite difference normal of SDF (vectorized)
    e = 1e-3
    dx = torch.tensor([e, 0, 0], device=p.device, dtype=p.dtype)
    dy = torch.tensor([0, e, 0], device=p.device, dtype=p.dtype)
    dz = torch.tensor([0, 0, e], device=p.device, dtype=p.dtype)
    nx = scene_sdf(p + dx, include_plane) - scene_sdf(p - dx, include_plane)
    ny = scene_sdf(p + dy, include_plane) - scene_sdf(p - dy, include_plane)
    nz = scene_sdf(p + dz, include_plane) - scene_sdf(p - dz, include_plane)
    n = torch.stack([nx, ny, nz], dim=-1)
    return n / (n.norm(dim=-1, keepdim=True) + 1e-12)


# ---------------------------------
# 4) Lighting/shading (torch)
# ---------------------------------
light_pos = (-2.5, 2.8, -1.5)
light_col = (1.3, 1.2, 1.1)
ambient = 0.08
spec_power = 32.0
albedo_obj1 = (0.8, 0.3, 0.2)
albedo_obj2 = (0.2, 0.6, 0.9)


def which_object(p: torch.Tensor) -> torch.Tensor:
    ps = p - torch.tensor([-0.8, -0.1, 2.2], device=p.device, dtype=p.dtype)
    d1 = sd_sphere(ps, 0.7)
    pb = p - torch.tensor([1.0, 0.2, 2.8], device=p.device, dtype=p.dtype)
    d2 = sd_round_box(pb, torch.tensor([0.6, 0.4, 0.8], device=p.device, dtype=p.dtype), 0.15)
    return (d1 < d2).to(p.dtype)  # 1.0 if obj1 else 0.0


def shade(p: torch.Tensor, vdir: torch.Tensor, device, dtype, include_plane: bool = True) -> torch.Tensor:
    n = estimate_normal(p, include_plane)
    lpos = torch.tensor(light_pos, device=device, dtype=dtype)
    lcol = torch.tensor(light_col, device=device, dtype=dtype)
    ldir = lpos - p
    ldist = ldir.norm(dim=-1, keepdim=True)
    ldir = ldir / (ldist + 1e-12)

    diff = torch.clamp((n * ldir).sum(dim=-1, keepdim=True), min=0.0)
    h = (ldir - vdir)
    h = h / (h.norm(dim=-1, keepdim=True) + 1e-12)
    spec = torch.clamp((n * h).sum(dim=-1, keepdim=True), min=0.0) ** spec_power

    att = 1.0 / (0.2 + 0.02 * ldist + 0.01 * ldist * ldist)
    base1 = torch.tensor(albedo_obj1, device=device, dtype=dtype)
    base2 = torch.tensor(albedo_obj2, device=device, dtype=dtype)
    obj1 = which_object(p)[..., None]
    # Robust plane-vs-object decision at hit: compare SDF distances
    # Recompute object smooth-min distance
    ps = p - torch.tensor([-0.8, -0.1, 2.2], device=p.device, dtype=p.dtype)
    d1 = sd_sphere(ps, 0.7)
    pb = p - torch.tensor([1.0, 0.2, 2.8], device=p.device, dtype=p.dtype)
    d2 = sd_round_box(pb, torch.tensor([0.6, 0.4, 0.8], device=p.device, dtype=p.dtype), 0.15)
    k = 0.25
    h = (0.5 + 0.5 * (d2 - d1) / k).clamp(0.0, 1.0)
    d_obj = (1 - h) * d2 + h * d1 - k * h * (1 - h)
    d_plane = p[...,1] + 1.0
    is_plane = (d_plane <= d_obj)[..., None]
    if not include_plane:
        is_plane = torch.zeros_like(is_plane, dtype=torch.bool)
    grid_u = torch.floor((p[...,0] * 3.0) % 2.0)
    grid_v = torch.floor((p[...,2] * 3.0) % 2.0)
    grid = ((grid_u + grid_v) % 2.0)[..., None]
    plane_col = 0.2 + 0.1 * grid
    base_plane = torch.cat([plane_col, plane_col*0.9, plane_col*0.7], dim=-1).to(dtype)
    base_obj = obj1 * base1 + (1 - obj1) * base2
    base = torch.where(is_plane, base_plane, base_obj)
    color = ambient * base + att * (diff * base + 0.3 * spec * lcol)
    return color.clamp(0.0, 1.0)


# ---------------------------------
# 5) Primary ray integration with bending (vectorized)
# ---------------------------------
@torch.inference_mode()
def render(pm: PMField,
           W: int = 512,
           H: int = 512,
           fov_deg: float = 60.0,
           cam_pos=(0.0, 0.0, -4.0),
           cam_dir=(0.0, 0.0, 1.0),
           cam_up=(0.0, 1.0, 0.0),
           max_steps: int = 500,
           ds: float = 0.01,
           hit_epsilon: float = 0.01,
           far_limit: float = 16.0,
           bg_color=(0.03, 0.05, 0.08),
           chunk_size: int | None = None,
           bend: bool = True,
           bend_strength: float = 1.0,
           orient: str = 'standard',
           device: str = 'cpu',
           dtype=torch.float32,
           include_plane: bool = True) -> torch.Tensor:
    """
    Returns: image tensor [H,W,3] in [0,1].
    """
    p, k, _, _, _, _ = make_camera(W, H, fov_deg, cam_pos, cam_dir, cam_up, device, dtype, orient=orient)

    # Flatten to [N,3]
    N = W * H
    p = p.view(N, 3)
    k = k.view(N, 3)
    out = torch.zeros(N, 3, device=device, dtype=dtype)
    hit_mask_global = torch.zeros(N, dtype=torch.bool, device=device)
    bg = torch.tensor(bg_color, device=device, dtype=dtype)

    def sky_background(ki: torch.Tensor) -> torch.Tensor:
        # Simple sky gradient + procedural stars from ray direction
        t = (ki[...,1] * 0.5 + 0.5).clamp(0,1)  # up vector contribution
        sky_top = torch.tensor([0.1, 0.2, 0.4], device=device, dtype=dtype)
        sky_bot = torch.tensor([0.02, 0.04, 0.08], device=device, dtype=dtype)
        grad = t[..., None] * sky_top + (1 - t[..., None]) * sky_bot
        # Stars: hash of direction -> random speckles
        h = (ki * torch.tensor([12.9898,78.233,45.164], device=device, dtype=dtype)).sum(dim=-1)
        s = torch.frac(torch.sin(h) * 43758.5453)
        stars = (s > 0.996).to(dtype)[..., None] * 0.8
        return (grad + stars).clamp(0,1)

    def process_chunk(idx: torch.Tensor):
        pi = p[idx]
        ki = k[idx]
        # Track mapping to original chunk-local indices [0..len(idx)-1]
        active_idx = torch.arange(idx.numel(), device=device)
        color = sky_background(ki).clone()
        hit_mask = torch.zeros(idx.numel(), dtype=torch.bool, device=device)
        plane_hits = 0
        object_hits = 0
        for _ in range(max_steps):
            if pi.numel() == 0:
                break
            d = scene_sdf(pi, include_plane)
            hit = d < hit_epsilon
            if hit.any():
                # Count plane vs object at hit points for diagnostics
                hp = pi[hit]
                # recompute object smooth-min vs plane
                ps = hp - torch.tensor([-0.8, -0.1, 2.2], device=device, dtype=dtype)
                d1 = sd_sphere(ps, 0.7)
                pb = hp - torch.tensor([1.0, 0.2, 2.8], device=device, dtype=dtype)
                d2 = sd_round_box(pb, torch.tensor([0.6, 0.4, 0.8], device=device, dtype=dtype), 0.15)
                ksm = 0.25
                hsm = (0.5 + 0.5 * (d2 - d1) / ksm).clamp(0.0, 1.0)
                d_obj = (1 - hsm) * d2 + hsm * d1 - ksm * hsm * (1 - hsm)
                d_plane_hit = hp[...,1] + 1.0
                is_plane_hit = d_plane_hit <= d_obj
                if include_plane:
                    plane_hits += int(is_plane_hit.count_nonzero().item())
                    object_hits += int((~is_plane_hit).count_nonzero().item())
                else:
                    object_hits += int(hit.count_nonzero().item())

                col = shade(hp, ki[hit], device, dtype, include_plane)
                color[active_idx[hit]] = col
                hit_mask[active_idx[hit]] = True
                keep = ~hit
                pi = pi[keep]
                ki = ki[keep]
                active_idx = active_idx[keep]

            if pi.numel() == 0:
                break

            # Far termination
            far = pi.norm(dim=-1) > far_limit
            if far.any():
                color[active_idx[far]] = bg
                keep = ~far
                pi = pi[keep]
                ki = ki[keep]
                active_idx = active_idx[keep]

            if pi.numel() == 0:
                break

            # Bend by PM gradient (optional)
            if bend:
                g = pm.grad_ln_n(pi)
                g_perp = project_perp(ki, g)
                ki = ki + bend_strength * g_perp * ds
                ki = ki / (ki.norm(dim=-1, keepdim=True) + 1e-12)

            # Advance by ds / n
            nloc = pm.n(pi)
            pi = pi + ki * (ds / (nloc[..., None] + 1e-12))

        # Any remaining active rays get background
        if active_idx.numel() > 0:
            color[active_idx] = bg
        return color, hit_mask, plane_hits, object_hits

    if chunk_size is None:
        idx = torch.arange(N, device=device)
        col, mask, ph, oh = process_chunk(idx)
        out[:] = col
        hit_mask_global[idx] = mask
        plane_total = ph
        object_total = oh
    else:
        plane_total = 0
        object_total = 0
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            idx = torch.arange(start, end, device=device)
            col, mask, ph, oh = process_chunk(idx)
            out[start:end] = col
            hit_mask_global[start:end] = mask
            plane_total += ph
            object_total += oh

    if orient == 'standard':
        img = out.view(H, W, 3)
    else:
        img = out.view(W, H, 3).permute(1, 0, 2)
    # Print stats for diagnostics
    hits = hit_mask_global.count_nonzero().item()
    total = N
    pct = 100.0 * hits / max(total, 1)
    if include_plane:
        print(f"Hit pixels: {hits}/{total} ({pct:.2f}%) | plane hits: {plane_total}, object hits: {object_total}")
    else:
        print(f"Hit pixels: {hits}/{total} ({pct:.2f}%) | object-only mode, hits: {object_total}")
    return img.clamp(0.0, 1.0)


def main():
    parser = argparse.ArgumentParser(description="Torch PM ray tracing demo")
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--height', type=int, default=512)
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--ds', type=float, default=0.01)
    parser.add_argument('--fov', type=float, default=60.0)
    parser.add_argument('--cam-pos', type=str, default=None, help='Override camera position as x,y,z')
    parser.add_argument('--cam-dir', type=str, default=None, help='Override camera direction as x,y,z')
    parser.add_argument('--cam-up',  type=str, default=None, help='Override camera up vector as x,y,z')
    parser.add_argument('--hit-eps', type=float, default=0.01)
    parser.add_argument('--far', type=float, default=16.0)
    parser.add_argument('--chunk', type=int, default=65536, help='rays per chunk (None for all)')
    parser.add_argument('--out', type=str, default='pm_raytrace_torch.png')
    parser.add_argument('--device', type=str, default='auto', choices=['auto','cpu','cuda'])
    parser.add_argument('--dtype', type=str, default='fp32', choices=['fp16','fp32'])
    parser.add_argument('--no-bend', action='store_true', help='Disable PM curvature (debug)')
    parser.add_argument('--rotate', type=str, default='180', choices=['none','90ccw','90cw','180'], help='Rotate final image for presentation')
    parser.add_argument('--orient', type=str, default='standard', choices=['standard','legacy'], help='Camera/image mapping convention')
    parser.add_argument('--flip-x', action='store_true', help='Mirror final image horizontally')
    parser.add_argument('--flip-y', action='store_true', help='Mirror final image vertically')
    parser.add_argument('--bend-strength', type=float, default=1.0, help='Scale curvature strength (0..1 recommended)')
    parser.add_argument('--no-plane', action='store_true', help='Disable the ground plane (debug objects)')
    args = parser.parse_args()

    dev = 'cuda' if (args.device == 'cuda' or (args.device == 'auto' and torch.cuda.is_available())) else 'cpu'
    dtype = torch.float16 if args.dtype == 'fp16' and dev == 'cuda' else torch.float32

    # PM masses (same as minimal demo)
    centers = [(-1.2, 0.5, 1.0), (1.0, -0.6, 1.5)]
    mus = [0.50, 0.35]
    pm = PMField(centers, mus, eps=1e-4, device=dev, dtype=dtype)

    # Camera overrides
    cam_pos = (0.0, 0.0, -4.0)
    cam_dir = (0.0, 0.0, 1.0)
    cam_up  = (0.0, 1.0, 0.0)
    if args.cam_pos:
        cam_pos = tuple(map(float, args.cam_pos.split(',')))
    if args.cam_dir:
        cam_dir = tuple(map(float, args.cam_dir.split(',')))
    if args.cam_up:
        cam_up = tuple(map(float, args.cam_up.split(',')))

    # Resolve output path: save alongside this script if user provided a bare filename
    script_dir = os.path.dirname(__file__)
    out_arg = args.out or 'pm_raytrace_torch.png'
    if os.path.isabs(out_arg) or os.path.dirname(out_arg):
        out_path = out_arg
    else:
        out_path = os.path.join(script_dir, out_arg)

    img = render(pm,
                 W=args.width,
                 H=args.height,
                 fov_deg=args.fov,
                 cam_pos=cam_pos,
                 cam_dir=cam_dir,
                 cam_up=cam_up,
                 max_steps=args.steps,
                 ds=args.ds,
                 hit_epsilon=args.hit_eps,
                 far_limit=args.far,
                 bend=(not args.no_bend),
                 bend_strength=args.bend_strength,
                 orient=args.orient,
                 chunk_size=args.chunk if args.chunk > 0 else None,
                 device=dev,
                 dtype=dtype,
                 include_plane=(not args.no_plane))

    # Gamma and save
    img_np = (img.clamp(0, 1) ** (1.0 / 2.2)).detach().cpu().numpy()
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
    print(f"Saved {out_path} [{args.width}x{args.height}] on {dev} ({str(dtype).split('.')[-1]})")


if __name__ == '__main__':
    main()

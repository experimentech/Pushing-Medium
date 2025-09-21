# Torch-based Pushing-Medium Demos

This folder contains GPU-ready demos implemented in PyTorch:

- `torch_pm_raytracing_demo.py` — fast curved-ray SDF renderer using a pushing-medium index field n(x) and analytic ∇ln n.
- `torch_pm_radiosity_demo.py` — simple radiosity-style photon demo with optional bending and diffuse bounces; supports camera and top-down accumulation.

Both scripts run on CPU or CUDA and chunk work to manage memory.

## Quick start

Render a quick ray-traced image (no bend):

```bash
python3 programs/demos/visualization/graphics/torch_pm_raytracing_demo.py \
  --width 640 --height 360 --steps 450 --ds 0.012 \
  --hit-eps 0.01 --far 16 --no-bend \
  --rotate 180 --out pm_raytrace_nobend.png \
  --device auto --dtype fp32
```

Mild bend (0.5):

```bash
python3 programs/demos/visualization/graphics/torch_pm_raytracing_demo.py \
  --width 640 --height 360 --steps 500 --ds 0.012 \
  --hit-eps 0.01 --far 16 \
  --bend-strength 0.5 \
  --rotate 180 --out pm_raytrace_bend.png \
  --device auto --dtype fp32
```

Top-down radiosity map (fast, debug-friendly):

```bash
python3 programs/demos/visualization/graphics/torch_pm_radiosity_demo.py \
  --width 256 --height 256 \
  --photons 40000 --bounces 1 \
  --steps 400 --ds 0.02 --hit-eps 0.02 --far 20 \
  --no-bend \
  --accum topdown --x-range=-3,3 --z-range=0,6 \
  --out pm_radiosity_topdown.png \
  --device auto --dtype fp32
```

Camera-accumulated radiosity (sparser, try some blur):

```bash
python3 programs/demos/visualization/graphics/torch_pm_radiosity_demo.py \
  --width 512 --height 512 \
  --photons 120000 --bounces 2 \
  --steps 500 --ds 0.016 --hit-eps 0.02 --far 24 \
  --bend-strength 0.5 \
  --accum camera \
  --albedo-bleed --filter-sigma 0.6 \
  --rotate 180 \
  --out pm_radiosity_camera_bend.png \
  --device auto --dtype fp32
```

Note: If `--out` is a bare filename, images are saved alongside these scripts, not in the repo root. Pass an absolute path or a folder if you want to save elsewhere.

## Key options

- `--no-bend` or `--bend-strength <v>`: Disable or scale PM curvature.
- `--steps`, `--ds`, `--far`, `--hit-eps`: Control integrator step size and limits.
- `--device {auto,cpu,cuda}` and `--dtype {fp32,fp16}`: Choose runtime.
- Raytracer: `--orient`, `--rotate`, `--flip-x/y` to match earlier reference images.
- Raytracer: `--orient`, `--rotate`, `--flip-x/y` to match earlier reference images.
  - `--no-plane`: Hide the ground plane (useful to confirm objects are visible)
  - `--no-object-priority`: Disable object-first hit resolution (defaults to object-priority ON)
- Radiosity: `--accum {camera,topdown}`, `--x-range`, `--z-range` (topdown only), `--albedo-bleed`, `--filter-sigma`.

## Tips

- For negative ranges in `--x-range`/`--z-range`, use equals syntax (e.g., `--x-range=-3,3`).
- CUDA + fp16 can be very fast when available, but keep an eye on precision.
- Camera-accumulated radiosity is inherently sparse at low photon counts — increase photons, bounces, and add mild blur for readability.

### Orientation note

The raytracer’s default `--rotate` is `none`. If you prefer the floor visually at the bottom (matching earlier results), add `--rotate 180`.

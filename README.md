# Pushing‑Medium Gravity

An alternative gravitational model built around a refractive‑index‑like medium that “pushes” matter and bends light. This repo contains:
- Core equations and notes (LaTeX/Markdown/TXT)
- A set of Python demos and experiments, including a PMFlow‑BNN analogue network for noisy MNIST

This is an exploratory codebase — treat it as a sandbox. As a physical model, believe what you like; as a computational playground, it works and is fun to tinker with.

## Repository layout

```
docs/
	latex/        # LaTeX sources for formula sheets and notes
	markdown/     # Rendered/parallel Markdown versions of docs
	image_formatted/  # Pre-rendered figures/screenshots

programs/
	demos/
		BNN/        # PMFlow‑BNN demo (noisy MNIST, robustness, attractor probe)
		comparitive_demos/
		propaganda_demos/
		...         # Many small demonstrations (lensing, plasma, skeletons, etc.)
	junk/         # Scratch scripts
	misc/         # Small utilities/experiments
```

Some LaTeX files are rough or may not compile on the first try. Prefer the content under `docs/markdown` and `docs/image_formatted` if you just want to read.

## Quick start (demos)

Requirements vary by demo. Most non‑NN demos use NumPy/Matplotlib/SciPy. The PMFlow‑BNN demo adds PyTorch and torchvision.

### Install (minimal)

Optional, for common demos:

```bash
pip install numpy scipy matplotlib tqdm
```

For the PMFlow‑BNN demo:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install tqdm
```

Adjust the PyTorch install command to match your platform/CUDA if you have a GPU.

### Run the PMFlow‑BNN demo

Script: `programs/demos/BNN/pmflow_bnn_always_plastic.py`

- Saves accuracy curves to `acc_curves.png`
- Saves a simple “attractor energy” plot to `bnn_attractor_energy.png`
- Prints training/test accuracy and a robustness score under noise/occlusion

Environment flags:
- `EPOCHS` — number of training epochs (default 5)
- `HEADLESS` — set to `1` to disable interactive plotting (saves images instead)
- `NUM_WORKERS` — DataLoader workers (default 2; set to 0 if you have issues)
- `DATA_DIR` — where to download/cache MNIST (default `./data`)

Example (short, headless CPU run):

```bash
HEADLESS=1 EPOCHS=1 NUM_WORKERS=0 python programs/demos/BNN/pmflow_bnn_always_plastic.py
```

Notes:
- CPU runs will take a few minutes per epoch. On GPU it’s much faster.
- The script prints a clear start/end banner and progress bars so it shouldn’t feel “silent.”

### Other demos

Explore `programs/demos` for lensing, plasma index, skeleton flow, Lagrange points, and more. A short guide lives at `programs/demos/README.md`.

Example scripts you can try:
- `programs/demos/pushing_medium.py`
- `programs/demos/grav_lensing.py`
- `programs/demos/plasma_index_demo.py`

## Core equations (pointer)

See `docs/latex/all_formulas.tex` (or the Markdown equivalents under `docs/markdown`). At a glance, the model uses:
- A static refractive index from mass points or continuous density
- Optional flow field (frame‑drag analogue)
- Ray equations from Fermat’s principle
- An optical–mechanical analogue for massive particle acceleration

These are mapped into small, runnable demos wherever practical.

## Troubleshooting

- “Script exits with no window”: run headless (`HEADLESS=1`) and check the saved PNGs; interactive windows are disabled in headless mode.
- MNIST download problems: set `DATA_DIR` to a writable location; ensure internet access on first run.
- Slow CPU training: start with `EPOCHS=1` and `NUM_WORKERS=0` to sanity‑check; then scale up.

## License

Public domain / unrestricted use as originally stated: do whatever you like.



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

## Physics libraries & testbench

The repository now includes two lightweight Python packages (src layout) used by an automated pytest testbench:

- `pushing_medium` — refractive index / flow model core functions (index from point masses or density, ray bending integrators, massive acceleration analogue, Plummer helpers, moving-lens approximations).
- `general_relativity` — baseline weak‑field GR formulae (deflection, Shapiro delay, perihelion precession, frame‑drag, Einstein radius, quadrupole power, etc.).

Calibration values (fitted once then cached in `tests/calibration.json`):
- `mu_coeff` — index scaling (≈ 2G/c^2) fitted by matching numeric weak‑bending deflection to analytic 4GM/(c^2 b).
- `k_TT` — tensor (quadrupole) normalization (≈ 1) via power ratio.
- `k_Fizeau` — current heuristic (≈ 1) but numeric moving‑lens integrator shows negligible first‑order boost; will be refined with improved path integration.

Run all tests:
```bash
pytest -q
```

## Ray bending helpers

Implemented numerical small‑angle integrators:
- `index_deflection_numeric(M, b, mu, ...)` — integrates ∂x ln n along a straight reference path.
- `fermat_deflection_static_index(...)` — thin wrapper for clarity when used in Fermat‑style discussions.
- `moving_lens_deflection_first_order(...)` — scales static deflection by (1 + k_Fizeau v/c) (model placeholder).
- `moving_lens_deflection_numeric(...)` — explicit integration with a time‑dependent lens position x_L(z)=v z / c (straight path; first‑order in v/c).

Current limitations:
- Straight‑line path assumption breaks down nearer to a few Schwarzschild radii; we added a trend test instead of enforcing analytic GR there.
- Moving‑lens numeric model shows a tiny negative O(1e‑5) fractional change ⇒ effective first‑order boost ~ 0 with present approximation.
- No iterative curved‑path correction yet; planned (two‑pass or RK update of transverse position).

## Convergence & validation tests

Key test categories (all under `tests/` and passing):
- Analytic parity (PM vs GR weak‑field deflection, Shapiro, redshift, perihelion, frame‑drag, lensing angle, GW power/speed, orbit energy).
- Calibration verification (μ near 2G/c^2, k_TT ≈ 1, k_Fizeau present).
- Numeric deflection convergence vs step count and z_max.
- Moving lens: numeric vs first‑order ratio consistency for small v.
- Strong‑field trend: deflection growth pattern & fitted log–log slope across 30→10 R_s (documentation of current integrator behavior).

## Planned next steps

- Iterative / curved‑path ray integrator for improved near‑critical bending.
- Empirical re‑derivation of k_Fizeau from path‑corrected moving lens optics.
- Adaptive z integration (smaller dz near closest approach) to speed convergence.
- Optional effective metric export (mapping n, u_g to isotropic weak‑field metric components for comparison tables).

## Effective metric & PPN draft

An initial helper maps the scalar index to an approximate isotropic weak‑field metric: \(g_{tt}\approx -1/n^2\), \(g_{rr}\approx n^2\). Expanding for \(n=1+\epsilon\) gives:
\[
g_{tt} \approx -1 + 2\epsilon + O(\epsilon^2), \qquad g_{rr} \approx 1 + 2\epsilon + O(\epsilon^2)
\]
Identifying \(\epsilon = \mu M / r\) (positive) yields PPN-like \(\gamma \approx 1\) and (with no explicit quadratic term yet) \(\beta \approx 1\). Tests (`test_ppn_parameters.py`) assert these within tight tolerances and verify Lorentzian signature.

Limitations: This mapping is a heuristic; full covariant reconstruction and higher-order (β) effects are not yet modeled.

If you’d like those implemented, open an issue or keep the chat session going.

## Troubleshooting

- “Script exits with no window”: run headless (`HEADLESS=1`) and check the saved PNGs; interactive windows are disabled in headless mode.
- MNIST download problems: set `DATA_DIR` to a writable location; ensure internet access on first run.
- Slow CPU training: start with `EPOCHS=1` and `NUM_WORKERS=0` to sanity‑check; then scale up.

## License

Public domain / unrestricted use as originally stated: do whatever you like.



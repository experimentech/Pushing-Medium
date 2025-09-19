# Pushing‑Medium Gravity

Concise implementation of a refractive‑index + flow ("pushing‑medium") gravitational analogue together with a weak‑field General Relativity (GR) baseline and a reproducible testbench. The focus is parity with classical weak‑field observables (deflection, time delay, perihelion precession, frame‑drag, lensing scale, GW power) and incremental tooling for ray tracing and parameter extraction.

## Repository Overview

```
src/
  pushing_medium/         # Medium index/flow, ray & deflection helpers, calibration hooks
  general_relativity/     # Weak-field GR comparison formulae
tests/                    # pytest suite (calibration, parity, convergence, PPN, strong-field trends)
docs/                     # LaTeX + Markdown technical notes
programs/demos/           # Example scripts (lensing, plasma index, skeleton flows, BNN demo)
```

Primary documentation sources: `docs/markdown/` (readable) and `docs/latex/` (original formula sheets).

## Installation

Core (analysis/tests):
```bash
pip install -r requirements.txt  # if present, otherwise install numpy scipy matplotlib pytest
```

BNN demo (optional):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

## Testing

The repository includes a comprehensive test suite with **59 tests** covering all aspects of the gravitational model:

```bash
# Run all tests (recommended)
pytest tests -v

# Quick test run  
pytest tests -q

# Run specific test categories
pytest tests/test_passed_benchmarks.py -v    # Core GR comparison tests
pytest tests/test_galaxy_rotation.py -v     # Galaxy dynamics tests

# Generate detailed test report
python test_results_table.py
pytest -q
```

Minimal BNN run:
```bash
HEADLESS=1 EPOCHS=1 NUM_WORKERS=0 python programs/demos/BNN/pmflow_bnn_always_plastic.py
```

**Current Status:** ✅ All 59 tests pass (100% success rate)

For detailed testing documentation, see [TESTING.md](TESTING.md).

## Core Physical Model (outline)

1. Refractive index field: \(n(\mathbf{r}) = 1 + \sum_i \mu_i/|\mathbf{r}-\mathbf{r}_i|\) or continuous integral analogue.
2. Optional flow \(\mathbf{u}_g\) (rotational + translational placeholders) for frame‑drag analogues.
3. Light propagation: small‑angle Fermat updates using \(\partial_x \ln n\) integrals (several numeric variants).
4. Massive particle analogue: acceleration term \(-c^2 \nabla \ln n\) plus Newtonian superposition.

## Packages & Calibration

`pushing_medium` (index, flows, deflection, iterative & moving‑lens integrators, Plummer helpers, effective metric, PPN estimates)

`general_relativity` (deflection, Shapiro delay, perihelion precession, frame‑drag, Einstein radius, GW speed/power, orbit energy)

Calibration (cached at `tests/calibration.json`):
- `mu_coeff`: fitted numeric deflection → analytic match (target 4GM/(c^2 b)).
- `k_TT`: quadrupole power normalization (≈1).
- `k_Fizeau`: provisional; presently near unity heuristic, numeric moving‑lens indicates negligible first‑order term.

## Ray Bending Implementations

- Straight-line integral: `index_deflection_numeric`
- Fermat wrapper: `fermat_deflection_static_index`
- First-order moving lens: `moving_lens_deflection_first_order`
- Numeric moving lens (time-shifted lens): `moving_lens_deflection_numeric`
- Curved-path iterative refinement: `curved_path_deflection_iterative`

Limitations: small-angle assumption; near-critical (few R_s) behavior approximate; moving-lens path currently straight or first-order corrected only.

## Galaxy Dynamics (Rotation Curves)

Module `galaxy_dynamics.rotation` provides utilities for preliminary disk + medium modified rotation curve modeling:

- Exponential disk mass interior: \(M(<r)=M_d[1-(1+r/R_d)e^{-r/R_d}]\)
- Baryonic acceleration: \(a_{bar}=G M(<r)/r^2\)
- Phenomenological medium perturbation: \(\delta n_{PM}= - (v_\infty^2/c^2) \ln(1+r/r_s) S(r)\) with smoothing \(S(r)=[1+(r_c/r)^m]^{-1/m}\)
- Medium acceleration (small perturbation): \(a_{med}= -c^2\,d\delta n_{PM}/dr\) (sign chosen so outer contribution supports flattening)
- Circular speed: \(v_c(r)=\sqrt{r(a_{bar}+a_{med})}\)

Example:
```python
from galaxy_dynamics.rotation import DiskParams, MediumParams, rotation_curve
disk = DiskParams(M_d=5e40, R_d=5e19)
medium = MediumParams(v_inf=2.2e5, r_s=5e19, r_c=1e19, m=2.0)
radii = [1e19, 5e19, 1e20]
vc = rotation_curve(radii, disk, medium)
```

Tests cover mass convergence, acceleration signs, and approximate outer flattening behavior. SPARC ingestion and fitting pipeline are planned next.

## Test Coverage (pytest)

- GR parity: deflection, Shapiro delay, redshift (potential), perihelion, frame-drag, Einstein radius, GW speed/power, orbit energy.
- Calibration: μ, k_TT, k_Fizeau presence.
- Convergence: step & domain extent (z_max) refinement.
- Moving lens: ratio consistency (first-order vs numeric).
- Iterative vs straight-line: accuracy sanity at moderate b.
- Strong-field trend: qualitative scaling (documentation only).
- PPN extraction: gamma, beta ≈ 1; metric signature checks.

## Roadmap (short)

- Adaptive ray integration (near closest approach refinement)
- Path-corrected moving-lens calibration (refined k_Fizeau)
- Comprehensive error tables & benchmarking harness
- Effective metric higher-order terms (β beyond leading) & parameter scanning
- Wave / dispersion prototype (TT perturbations)

## Effective Metric & PPN

Heuristic mapping: \(g_{tt}\approx -1/n^2\), \(g_{rr}\approx n^2\). Expansion for \(n=1+\epsilon\) gives matching leading coefficients ⇒ \(\gamma\approx1\). No non-linear term modeled ⇒ \(\beta\approx1\). Tests verify gamma, beta, and metric signature; higher-order refinement is future work.

## Troubleshooting (selected)

- Headless plotting: set `HEADLESS=1` (images saved to PNGs).
- Dataset issues: set `DATA_DIR` to a writable path.
- Slow BNN run: start with `EPOCHS=1` and `NUM_WORKERS=0`.

## License

Unrestricted use / public domain style (see repository for any clarifications).



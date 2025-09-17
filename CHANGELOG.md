# Changelog

All notable additions to the evolving pushing‑medium / GR comparison testbench.

## Unreleased (current)
### Added
- Physics packages: `pushing_medium`, `general_relativity` (src/ layout) with core weak‑field & PM helper functions.
- Calibration fixture (`tests/fixtures/calibration.py`) producing cached `mu_coeff`, `k_TT`, `k_Fizeau`.
- Ray bending helpers: `index_deflection_numeric`, `fermat_deflection_static_index`, `moving_lens_deflection_first_order`, `moving_lens_deflection_numeric`.
- Convergence tests for deflection (steps and z_max refinement).
- Moving lens numeric vs first‑order scaling tests; empirical k_Fizeau observation (≈0 first‑order boost in current straight-path approximation).
- Strong‑field trend test (30→10 R_s) capturing present integrator scaling (slope ~ +1 over sampled range).
- Additional documentation in README on calibration, ray tracing, convergence, limitations, and roadmap.

### Changed
- README expanded with physics library overview and planned next steps.
- Calibration values test updated to tolerate heuristic k_Fizeau and near‑zero numeric boost.

### Known Limitations
- Straight path assumption for deflection; no iterative curved ray solution yet.
- Moving lens numeric integrator omits transverse path correction (only time‑shifted lens position).
- Strong‑field test does not assert GR analytic deflection (intended; future iterative solver planned).

### Planned
- Iterative curved-path ray tracer for near‑critical bending.
- Adaptive integration and path refinement for speed/accuracy.
- Re‑fit k_Fizeau using improved moving-lens solver.
- Effective metric export & comparison tables.
- Optional wave (TT) propagation demos with dispersion toggles.

# The Pushing-Medium Model: Technical Overview

## Abstract

The Pushing-Medium model presents an alternative theory of gravitation that replaces Einstein's curved spacetime with a flat Euclidean background containing a medium with variable refractive index and flow fields. This model uses optical analogies to reproduce gravitational phenomena, offering computational advantages and intuitive understanding while maintaining compatibility with observed gravitational effects in weak-field regimes.

## TL;DR (Quick Orientation)
Goal: Provide a flat-space refractive-index + flow-field alternative that reproduces key weak‑field gravitational observables (deflection, perihelion shift analogues, Shapiro‑like delays, rotation curves) while enabling direct comparison to dark-matter halo paradigms.

Core Ingredients:
- Refractive index perturbation n = 1 + Σ 2GM/(c^2 r) + phenomenological medium term for galactic scales.
- Optional flow field for frame‑dragging analogues.
- Optical–mechanical analogy supplies effective accelerations for massive bodies.

Main Python Modules (galaxy layer):
- rotation.py: Exponential disk + medium acceleration, circular velocities, rotation curve synthesis.
- data.py: SPARC-style loader (mock + real) returning RotationCurve objects with metadata.
- fitting.py: Medium parameter fitter, population metrics (chi2, frac_rms, outer_delta, inner_slope_ratio).
- halos.py: NFW & Burkert profiles, halo-only and joint disk+halo fitting.
- compare.py: Unified multi-model comparison, aggregation, CSV export.

Key Functions to Know:
compare_models -> run medium, halo, joint fits for one galaxy.
aggregate_statistics -> compute medians/means across a sample.
fit_rotation_curve / fit_halo_rotation_curve / fit_disk_halo_rotation_curve -> individual model fitting backends.

Typical Workflow:
1. Load curves: curves = load_sparc_real('sparc.csv', return_dict=True)
2. Define parameter bounds (disk, medium, halo).
3. For each galaxy: summary = compare_models(rc, ...)
4. Aggregate: stats = aggregate_statistics(summaries)
5. Export: export_comparison_results('results.csv', summaries)
6. Inspect medium vs halo/joint frac_rms and outer_delta distributions.

Current Metrics:
- chi2: raw chi-square (relative comparison across identical data vectors).
- frac_rms: RMS / mean(V_obs) for scale-free residual magnitude.
- outer_delta: captures asymptotic plateau match quality.
- inner_slope_ratio: inner rise shape comparison (core/cusp analogue).

Status Snapshot (Sep 2025):
- Full comparison pipeline implemented (Section 11.12).
- 71 tests passing covering fitting, halos, joint models, comparison, CSV export.
- Documentation up through model comparison; scaling relations & information criteria planned.

Next Planned Extensions (high value):
- Baryonic Tully–Fisher & Radial Acceleration Relation extraction from summaries.
- Information criteria (AIC/BIC) to penalize parameter count.
- Parallel batch fitting & bootstrap confidence intervals.

Use this TL;DR when returning later: jump directly to Sections 10–11 for galaxy dynamics and comparisons.

---

## 1. Theoretical Foundation

### 1.1 Core Concept

The Pushing-Medium model is based on the premise that gravitational effects arise not from spacetime curvature, but from variations in a physical medium's properties:

- **Background**: Flat, fixed Euclidean space-time
- **Medium Properties**: Spatially and temporally varying refractive index $n(\mathbf{r}, t)$
- **Additional Effects**: Optional flow field $\mathbf{u}_g(\mathbf{r}, t)$ for frame-dragging analogues
- **Wave Phenomena**: Explicit wave perturbations for gravitational wave analogues

### 1.2 Philosophical Differences from General Relativity

| Aspect | General Relativity | Pushing-Medium Model |
|--------|-------------------|---------------------|
| **Nature of Space-time** | Dynamic, curved manifold | Static, flat Euclidean background |
| **Gravitational Field** | Encoded in metric tensor $g_{\mu\nu}$ | Encoded in refractive index $n$ and flow $\mathbf{u}_g$ |
| **Field Equations** | Einstein field equations (covariant) | Phenomenological formulas for $n$ and $\mathbf{u}_g$ |
| **Matter Motion** | Timelike geodesics | Modified Newtonian dynamics |
| **Light Propagation** | Null geodesics | Fermat's principle in moving medium |

---

## 2. Mathematical Formulation

### 2.1 Static Refractive Index Field

**Point Mass Configuration:**
$$
n(\mathbf{r}) = 1 + \sum_i \frac{\mu_i}{|\mathbf{r} - \mathbf{r}_i|}, \quad \mu_i = \frac{2GM_i}{c^2}
$$

**Continuous Mass Distribution:**
$$
n(\mathbf{r}) = 1 + \frac{2G}{c^2} \int \frac{\rho(\mathbf{r}')}{|\mathbf{r} - \mathbf{r}'|} \, d^3 r'
$$

**Interpretation:** Masses raise the local refractive index; higher index corresponds to effective time dilation / slower coordinate light speed.

### 2.2 Gradient of Refractive Index

$$
\nabla n(\mathbf{r}) = -\sum_i \frac{2GM_i}{c^2 |\mathbf{r} - \mathbf{r}_i|^3} (\mathbf{r} - \mathbf{r}_i)
$$

**Interpretation:** The gradient direction specifies attractive response; magnitude sets bending or acceleration strength.

### 2.3 Flow Field (Frame-Dragging Analogue)

**Rotational Flow:**
$$
\mathbf{u}_g(\mathbf{r}) = \sum_i \boldsymbol{\Omega}_i \times (\mathbf{r} - \mathbf{r}_i)
$$

**Angular Velocity from Spin:**
$$
\omega_s(r) = \frac{2GJ}{c^2 r^3}
$$

**Interpretation:** Mimics Lense–Thirring‑like frame influence by advecting rays and massive trajectories.

### 2.4 Gravitational Wave Analogues

**Time-Dependent Perturbation:**
$$
\delta n_{\mathrm{wave}}(\mathbf{r}, t) = A \cos(\mathbf{k} \cdot \mathbf{r} - \omega t + \phi_0)
$$

**TT Wave Equation:**
$$
\frac{\partial^2 h}{\partial t^2} - c^2 \frac{\partial^2 h}{\partial x^2} = 0
$$

**Interpretation:** TT‑style perturbations propagate at speed $c$; phenomenological, not derived from covariant field equations.

### 2.5 Total Field Configuration

$$
n_{\mathrm{total}}(\mathbf{r}, t) = n(\mathbf{r}) + \delta n_{\mathrm{wave}}(\mathbf{r}, t)
$$

---

## 3. Dynamics and Motion

### 3.1 Light Ray Propagation

**Ray Equation (Fermat's Principle):**
$$
\frac{d\hat{\mathbf{k}}}{ds} = \nabla_{\!\perp} \ln n_{\mathrm{total}}(\mathbf{r}, t)
$$

**Ray Velocity with Flow Advection:**
$$
\frac{d\mathbf{r}}{dt} = \frac{c\,\hat{\mathbf{k}}}{n_{\mathrm{total}}} + \mathbf{u}_g(\mathbf{r}, t)
$$

**Fermat Functional:**
$$
T = \int n\,ds - \frac{1}{c} \int (n^2 - 1)(\mathbf{u}_g \cdot \hat{\mathbf{k}})\,ds
$$

**Interpretation:** Rays extremize optical travel time producing lensing and time‑delay analogues.

### 3.1.1 Current implementation status
- Static index: `fermat_deflection_static_index` integrates $\partial_x \ln n$ along a straight reference path and numerically reproduces $4GM/(c^2 b)$.
- Moving lens: present code exposes a heuristic $k_{\mathrm{Fizeau}}$ (near unity) and a straight‑path translational correction; higher fidelity path‑corrected advection is planned.

### 3.2 Massive Particle Motion

**Optical-Mechanical Acceleration:**
$$
\mathbf{a}_{\mathrm{med}}(\mathbf{r}, t) = -c^2 \nabla \ln n_{\mathrm{total}}(\mathbf{r}, t)
$$

**Newtonian Gravity Component:**
$$
\mathbf{a}_{\mathrm{grav}}(\mathbf{r}) = -\sum_i \frac{GM_i (\mathbf{r} - \mathbf{r}_i)}{|\mathbf{r} - \mathbf{r}_i|^3}
$$

**Combined Equation of Motion:**
$$
\frac{d^2 \mathbf{r}}{dt^2} = \mathbf{a}_{\mathrm{grav}} + \mathbf{a}_{\mathrm{med}} + \text{(flow coupling)}
$$

**Interpretation:** Motion combines Newtonian inverse‑square attraction with an additional gradient‑index term.

---

## 4. Computational Implementation

### 4.1 Core Algorithm Structure

The model is implemented through modular components:

1. **Index Field Calculation**: Direct evaluation of $n(\mathbf{r})$ from mass distributions
2. **Gradient Computation**: Analytical or numerical derivatives of the index field
3. **Ray Integration**: Numerical integration of ray equations using RK4 or similar methods
4. **N-body Dynamics**: Leapfrog or Verlet integration for massive particle motion
5. **Flow Field Evaluation**: Direct computation of $\mathbf{u}_g$ from source configurations

### 4.2 Key Implementation Features

**Gaussian Index Profiles** (for smooth mass distributions):
```python
def gaussian_index(eps, sigma, center):
    def n(x, y):
        r2 = (x - center[0])**2 + (y - center[1])**2
        return 1.0 + eps * np.exp(-0.5 * r2 / sigma**2)
    return n
```

**Regularization** (to avoid singularities):
- Softening parameters $\epsilon$ in denominators
- Smooth cutoff functions near point masses

**Modular Effects**:
- Toggle-able frame-dragging effects
- Optional gravitational wave injection
- Configurable mass distributions

### 4.3 Performance Characteristics

**Computational Advantages:**
- Direct field evaluation (no PDE solving)
- Embarrassingly parallel ray tracing
- Fast N-body integration in flat space
- Real-time visualization capabilities

**Computational Complexity:**
- $O(N)$ for field evaluation at single point
- $O(N \times M)$ for $M$ field points and $N$ sources
- Linear scaling with number of rays or particles

---

## 5. Physical Phenomena and Applications

### 5.1 Light Deflection and Lensing

The framework reproduces:
- **Gravitational Lensing**: Rays bend toward regions of higher refractive index
- **Shapiro Delay**: Light travels slower through regions of higher index
- **Multiple Images**: Complex lens configurations produce multiple ray paths

### 5.2 Frame-Dragging Effects

Through the flow field $\mathbf{u}_g$:
- **Lense-Thirring Precession**: Analogous to GR frame-dragging
- **Rotational Coupling**: Spinning masses affect nearby trajectories
- **Orbital Precession**: Additional precession terms from medium flow

### 5.3 Gravitational Wave Analogues

**Wave Propagation:**
- Explicit TT perturbations in the medium
- Phase velocity equal to $c$
- Polarization effects through directional index variations

**Detection Signatures:**
- Changes in ray travel times
- Modulation of particle trajectories
- Interferometric phase shifts

### 5.4 Cosmological Applications

**Distance Relations:**
$$
D_C(z) = \int_0^z \frac{c\,dz'}{H(z')}, \quad D_L = (1+z)D_C, \quad D_A = \frac{D_C}{1+z}
$$

**Expanding Medium Model:**
- Substrate density evolution: $\rho_s(t) \propto a(t)^{-3}$
- Modified Hubble law through medium expansion
- Alternative to dark energy through medium properties

---

## 6. Comparison with General Relativity

### 6.1 Theoretical Differences

| Property | General Relativity | Pushing-Medium Model |
|----------|-------------------|---------------------|
| **Covariance** | Fully covariant under general coordinate transformations | Not manifestly covariant; uses preferred flat coordinates |
| **Field Equations** | Einstein equations: $G_{\mu\nu} = 8\pi T_{\mu\nu}$ | Phenomenological recipes for $n$ and $\mathbf{u}_g$ |
| **Causality** | Light cones from metric | Characteristic speeds from medium properties |
| **Strong Field Regime** | Exact nonlinear solutions available | Validity unclear; requires empirical validation |

### 6.2 Empirical Regime (current scope)

**Weak Field Agreement:**
- Recovers Newton's law in appropriate limits
- Reproduces classical tests: perihelion precession, light deflection, time delay
- Frame-dragging effects match Lense-Thirring predictions

**Potential Differences:**
- Strong-field regime behavior may differ
- Gravitational wave signatures could show deviations
- N-body dynamics include additional medium-mediated forces

### 6.3 Observational Tests

**Current Compatibility:**
- Solar system tests: Mercury perihelion precession
- Light deflection during solar eclipses
- Gravitational time dilation measurements
- Frame-dragging from Gravity Probe B

**Future Discriminating Tests:**
- Strong-field regime observations (black hole vicinity)
- Precise gravitational wave astronomy
- Multi-messenger astronomy correlations
- Ultra-precise solar system ephemeris

---

## 7. Advantages and Limitations

### 7.1 Advantages (summary)

**Conceptual:**
- Intuitive optical analogies
- Clear separation of effects (static, flow, waves)
- Pedagogically accessible
- Modular implementation

**Computational:**
- Fast numerical evaluation
- Real-time simulation capabilities
- Scalable to large N-body systems
- Interactive parameter exploration

**Practical:**
- No need for numerical relativity
- Direct connection to laboratory optics
- Straightforward experimental analogues
- Educational value for teaching

### 7.2 Limitations (summary)

**Theoretical:**
- Lack of manifest covariance
- No derivation from fundamental principles
- Phenomenological rather than ab initio
- Unknown behavior in strong-field regimes

**Empirical:**
- Limited to weak-field validation
- No guarantee of exact GR recovery
- Potential conflicts in extreme conditions
- Requires careful calibration of parameters

**Computational:**
- May require regularization techniques
- Potential instabilities in extreme configurations
- Limited to classical (non-quantum) regimes

---

## 8. Research Directions and Extensions

### 8.1 Theoretical Development

**Covariant Formulation:**
- Development of manifestly covariant version
- Connection to gauge theories
- Derivation from more fundamental principles

**Quantum Extensions:**
- Quantum field theory in the medium
- Particle creation/annihilation processes
- Quantum gravitational effects

### 8.2 Empirical Testing

**Precision Tests:**
- Laboratory analogues using metamaterials
- Precision timing of pulsar signals
- Advanced gravitational wave analysis
- Solar system ephemeris improvements

**Strong-Field Exploration:**
- Black hole shadow observations
- Neutron star merger dynamics
- Extreme mass ratio inspirals

### 8.3 Computational Enhancements

**Advanced Algorithms:**
- Adaptive mesh refinement for index fields
- GPU acceleration for large-scale simulations
- Machine learning for parameter optimization
- Hybrid analytical-numerical methods

---

## 9. Summary

Current implementation: weak‑field lensing/time‑delay/ perihelion / frame‑drag / GW power analogues validated numerically against classical GR forms. Framework is phenomenological; strong‑field and fully covariant behavior remain open. Emphasis is computational transparency and modular extensibility.

---

## 10. Unexplored Predictions & Experimental Tests

Outlined below are candidate phenomena for future discrimination; each includes rationale, indicative signature, and a schematic test.

1) Medium microphysics and constitutive relations
- Why it matters: unlike GR, the pushing‑medium requires a substrate model (compressible fluid, elastic field, condensate) and a constitutive relation linking substrate state to $n(\rho_s,P_s)$. Different microphysics predict measurable differences.
- Signature: frequency‑dependent wave propagation, dissipative losses, or mode conversion during strong events.
- Test: compare waveforms from merger simulations across different constitutive models; look for systematic deviations from GR templates.

2) Dispersion, attenuation and birefringence of gravitational signals
- Why it matters: the medium can be dispersive or anisotropic, producing frequency and polarization dependent propagation that vacuum GR forbids.
- Signature: frequency‑dependent arrival times, amplitude roll‑off, or polarization mixing in GW detectors.
- Test: search LIGO/Virgo/KAGRA catalogues for residual dispersion beyond current constraints; run injection/recovery studies with simple dispersive models.

3) Preferred frame and small Lorentz violations
- Why it matters: a physical substrate defines a rest frame, so tiny frame‑dependent effects (anisotropic propagation, seasonal modulation) are allowed.
- Signature: minute anisotropy in light speed or GW speed relative to the CMB/rest frame; sidereal modulation of precision clocks/tests.
- Test: re‑analyse high‑precision Lorentz invariance experiments and multi‑messenger timing (EM vs GW) for correlated residuals.

4) Local energy/momentum bookkeeping and back‑reaction
- Why it matters: the medium provides a local substrate energy density and momentum flux, enabling alternate accounts of gravitational wave energy loss and back‑reaction.
- Signature: different ringdown damping rates or energy partitioning during mergers.
- Test: compute substrate energy budget in toy merger models and compare predicted damping/ringdown with observed waveforms.

5) Horizon microstructure, echoes, and near‑horizon phenomenology
- Why it matters: horizons in a medium picture could correspond to refractive trapping, phase changes, or shock fronts, not geometric singularities.
- Signature: echoes, small deviations of black‑hole shadows, or modified near‑horizon EM/GW scattering.
- Test: ray‑tracing around a substrate model that stiffens near a compact core to simulate echoes/shadow changes.

6) Possible equivalence‑principle violations
- Why it matters: substrate coupling could depend weakly on composition or internal structure, violating strong or weak EP at some level.
- Signature: composition‑dependent free‑fall, anomalous tidal responses.
- Test: precision torsion‑balance, lunar laser ranging, and pulsar timing differential tests targeted at predicted coupling forms.

7) Nonlinear substrate dynamics and emergent effects (shocks, turbulence)
- Why it matters: hydrodynamic or elastic substrate equations can create shocks, turbulence, or instabilities that have no direct analogue in GR PDEs.
- Signature: transient, broadband emissions or stochastic backgrounds from substrate turbulence during violent events.
- Test: evolve simple nonlinear substrate PDEs for merger analogues and quantify stochastic emission spectra.

8) Cosmological re‑interpretation: vacuum energy and expansion
- Why it matters: substrate pressure/density naturally supplies new degrees of freedom for cosmic acceleration, potentially replacing or altering Λ interpretations.
- Signature: subtle changes in distance–redshift relations or growth of structure.
- Test: fit simple substrate cosmologies to SN/BAO/CMB compressed constraints and evaluate parameter degeneracies with dark energy.

9) Light‑propagation subtleties (Fermat vs null geodesic corrections)
- Why it matters: advection by $\mathbf{u}_g$, anisotropic indices, or higher‑order index corrections can produce departures from GR null geodesics.
- Signature: polarization‑dependent lensing, small extra time delays, microlensing residuals.
- Test: high‑precision lensing/time‑delay systems and polarimetric monitoring.

10) Quantum field coupling and semiclassical effects

---

## 11. Galaxy Rotation Curve Data Integration (SPARC)

### 11.1 Motivation
Testing the Pushing-Medium model against observed galaxy rotation curves requires a standardized ingestion path for high-quality catalogs. The SPARC (Spitzer Photometry & Accurate Rotation Curves) dataset provides homogeneous photometry and rotation measurements, decomposed into baryonic components (stellar disk, bulge, gas) with geometric metadata (distance, inclination, axis ratio). This repository now includes loaders for both a lightweight mock format (for rapid development) and a more feature-complete real-style format.

### 11.2 Data Structures
Rotation curves are represented by the `RotationCurve` dataclass:
- `radii_m`: galactocentric radii (meters)
- `v_obs_ms`: observed circular velocity (m/s)
- `v_err_ms`: observational uncertainty (m/s, NaN if absent)
- `sigma_star`, `sigma_gas`: surface density proxies (unit-preserving placeholders)
- `components`: optional dict of modeled component velocities (`stars`, `gas`, `bulge`, `bar`) each in m/s
- `distance_mpc`, `inclination_deg`, `axis_ratio`: optional geometric metadata

### 11.3 Loaders
1. Mock loader (development & tests):
    `from galaxy_dynamics import load_sparc_mock`
    Expected minimal columns: `Name,R_kpc,V_obs_kms,...` (small synthetic CSVs).

2. Real-style loader (component & metadata aware):
    `from galaxy_dynamics.data import load_sparc_real`
    Flexible, case-insensitive columns (subset required):
    - Required: `name, r_kpc, v_obs`
    - Optional: `err_v, v_stars, v_gas, v_bulge, v_bar, dist_mpc, inc_deg, axis_ratio, sigma_star, sigma_gas`

### 11.4 Usage Examples
Single galaxy (first in file):
```python
from galaxy_dynamics.data import load_sparc_real
rc = load_sparc_real('sparc_catalog.csv')
print(rc.name, rc.radii_m[0], rc.v_obs_ms[0])
```

Select a specific galaxy:
```python
rc = load_sparc_real('sparc_catalog.csv', galaxy_name='UGC01234')
if 'stars' in rc.components:
     v_stars = rc.components['stars']
```

Load all galaxies into a dict:
```python
curves = load_sparc_real('sparc_catalog.csv', return_dict=True)
print(len(curves), 'galaxies loaded')
```

### 11.5 Planned Extensions
Planned higher-level utilities will include:
- Baryonic mass model synthesis and conversion of component velocities to enclosed mass profiles
- Parameter estimation / χ² fitting against Pushing-Medium rotational predictions (`DiskParams`, `MediumParams`)
- Automated inclination correction and deprojection checks
- Quality cuts (minimum radial extent, error filtering, outer flatness metrics)

### 11.6 Validation Tests
The test suite now exercises:
- Unit conversion correctness (kpc→m, km/s→m/s)
- Component velocity presence and length matching `radii_m`
- Galaxy selection versus dictionary return modes
- Metadata propagation (distance, inclination, axis ratio)

### 11.7 Workflow Toward Model Confrontation
1. Load rotation curves (mock or real SPARC style)
2. (Future) Convert component velocities into baryonic acceleration curves
3. Generate Pushing-Medium predicted circular velocity via `circular_velocity` using `DiskParams` + `MediumParams`
4. Fit medium parameters (e.g., asymptotic velocity scale) by minimizing residuals
5. Aggregate residual diagnostics across sample (RMS, outer slope, inner rise)
6. Compare against standard dark matter halo fits (prospective baseline)

This staged approach will isolate where the current phenomenological medium succeeds or diverges relative to empirical galaxy dynamics.

### 11.8 Parameter Fitting Utility
The module provides a lightweight chi-square fitter `fit_rotation_curve` combining a global random search with a local perturbative refinement stage. It estimates `DiskParams` (exponential disk mass and scale length) and `MediumParams` (asymptotic velocity scale and shaping radii) directly from an observed `RotationCurve`.

Example:
```python
from galaxy_dynamics import load_sparc_real, fit_rotation_curve

rc = load_sparc_real('sparc_catalog.csv', galaxy_name='GAL_X')
disk_bounds = {'M_d': (1e40, 2e41), 'R_d': (2e19, 1.2e20)}
medium_bounds = {'v_inf': (8e4, 3e5), 'r_s': (2e19, 1e20), 'r_c': (5e18, 5e19), 'm': (1.0, 4.0)}
result = fit_rotation_curve(rc, disk_bounds, medium_bounds, n_random=300, n_refine=80)
print('Chi2:', result['chi2'])
print('Disk M_d:', result['disk'].M_d, 'R_d:', result['disk'].R_d)
print('Medium v_inf:', result['medium'].v_inf)
```

Returned dictionary keys:
- `disk`: best-fit `DiskParams`
- `medium`: best-fit `MediumParams`
- `chi2`: minimized chi-square value
- `model`: list of model circular velocities (m/s)
- `radii_m`: radii used for the fit (meters)

Calibration / robustness suggestions:
- Provide conservative bounds to avoid unrealistic degenerate solutions.
- Increase `n_random` for broad, multi-modal landscapes; `n_refine` for local polishing.
- Use logarithmic parameter transforms externally if spans exceed ~3 decades (current perturbation is linear percentage-based).
- Consider pre-filtering inner radii where baryonic modeling uncertainties (e.g., beam smearing) dominate.

Planned improvements:
- Optional weighting by radial leverage or fractional errors.
- Inclusion of halo comparison baselines (e.g., NFW, Burkert) for relative residual analysis.
- Batch fitting helper returning population statistics and residual distributions.

### 11.9 Residual Diagnostics & Population Fitting
To evaluate systematic performance across many galaxies, the fitting module supplies:

1. `compute_residual_metrics(rc, model)` — produces a dictionary of scalar diagnostics:
   - `rms`: Root-mean-square absolute velocity residual.
   - `frac_rms`: RMS normalized by mean observed velocity (scale-invariant comparison).
   - `outer_delta`: Relative residual at the outermost sampled radius; probes asymptotic matching.
   - `inner_slope_ratio`: Ratio of observed inner rise (v2/v1) to model inner rise; highlights core mismatch.

2. `fit_population(curves, disk_bounds, medium_bounds, n_random=..., n_refine=...)` — iterates `fit_rotation_curve` over a dictionary `{name: RotationCurve}` returning per-galaxy summaries augmented with `metrics`.

Example:
```python
from galaxy_dynamics import load_sparc_real
from galaxy_dynamics.fitting import fit_population

curves = load_sparc_real('sparc_catalog.csv', return_dict=True)
disk_bounds = {'M_d': (5e39, 5e41), 'R_d': (1e19, 2e20)}
medium_bounds = {'v_inf': (5e4, 3.5e5), 'r_s': (1e19, 2e20), 'r_c': (5e18, 8e19), 'm': (0.8, 4.0)}
summaries = fit_population(curves, disk_bounds, medium_bounds, n_random=120, n_refine=40)
for name, s in summaries.items():
    print(name, s['chi2'], s['metrics']['frac_rms'], s['metrics']['outer_delta'])
```

Interpretation guidelines:
- High `frac_rms` indicates global mismatch; inspect residual distribution vs radius.
- Large positive `outer_delta` (> +0.1) means model under-predicts outer plateau; negative indicates over-prediction.
- `inner_slope_ratio` > 1: observed rises faster than model (need steeper inner mass or medium taper); < 1: model too steep.
- Combine metrics into a selection filter (e.g., discard fits with frac_rms > 0.2 before population statistics).

Planned population extensions:
- Export tabular summaries (CSV / JSON) with percentile aggregates.
- Bootstrapped uncertainty estimates via re-sampling radii.
- Residual shape classification (e.g., concave vs convex error trend) for clustering analyses.

### 11.10 Halo Baselines & Comparative Performance
To assess whether the pushing‑medium phenomenology can obviate dark matter halos, we include standard halo benchmarks:

Implemented profiles:
- **NFW**: $(\rho_s, r_s)$ with mass $M(<r)=4\pi\rho_s r_s^3[\ln(1+x)-x/(1+x)]$, $x=r/r_s$.
- **Burkert** (cored): $(\rho_0, r_c)$ with $M(<r)=2\pi\rho_0 r_c^3[\tfrac{1}{2}\ln(1+x^2)+\ln(1+x)-\arctan x]$.

API summary:
```python
from galaxy_dynamics import (
    NFWParams, BurkertParams, fit_halo_rotation_curve,
    halo_velocity_profile, fit_rotation_curve
)

# Fit halo-only (no baryons) baseline
result_nfw = fit_halo_rotation_curve(radii, v_obs, v_err, 'nfw', bounds={'rho_s':(...), 'r_s':(...)})

# Medium + disk already via fit_rotation_curve(...)
```

Comparison methodology (initial minimal set):
1. Fit medium+disk model (phenomenological) and halo-only models separately.
2. Record per-galaxy metrics: $\chi^2$, fractional RMS, outer_delta.
3. Construct summary statistics: median frac_rms (medium) vs (NFW) vs (Burkert).
4. Count galaxies where medium improves $\chi^2$ by > threshold (e.g. $\Delta\chi^2 > 5$) relative to best halo.
5. Evaluate outer_delta distributions: systematic positive bias would indicate medium underestimates outer support compared to halos.

Interpretation guidelines:
- If medium median frac_rms ≲ halo median within statistical scatter and no strong outer bias, the phenomenological medium is competitive absent an explicit halo mass component.
- Large scatter or need for extreme parameter values (e.g. $m$ or $r_s$ uncorrelated with disk $R_d$) weakens explanatory power.
- Future: incorporate disk + halo fits to ensure fair comparison (currently halo-only exaggerates medium advantage if baryons significantly shape inner curve).

Planned extensions:
- Dual-component fits (baryons + NFW) for a fully balanced likelihood comparison.
- Information criteria (AIC, BIC) to penalize extra degrees of freedom in medium vs halo parameterizations.
- Scaling relation diagnostics (Baryonic Tully–Fisher, Radial Acceleration Relation) extracted from fitted parameter sets.

Limitations of current baseline:
- No explicit baryonic contribution included in halo fit (intentionally harsh test for medium inner slope fidelity).
- Simple random + refinement search may miss narrow degeneracy valleys (future: log-prior sampling / MCMC optional module).

### 11.11 Joint Disk + Halo Fitting
For balanced comparison against the medium model, we introduce a joint fitter combining an exponential baryonic disk with a dark halo (NFW or Burkert):

Model composition:
$$v_{\text{tot}}^2(r) = v_{\text{disk}}^2(r) + v_{\text{halo}}^2(r)$$
with the same exponential disk mass profile used by the medium comparison and an analytic halo velocity profile.

API:
```python
from galaxy_dynamics import fit_disk_halo_rotation_curve

res = fit_disk_halo_rotation_curve(
    radii_m, v_obs_ms, v_err_ms,
    halo_type='nfw',
    disk_bounds={'M_d': (1e40, 2e41), 'R_d': (2e19, 1.2e20)},
    halo_bounds={'rho_s': (1e-23, 1e-20), 'r_s': (2e19, 2e20)},
    n_random=250, n_refine=80
)
print(res['chi2'], res['disk'], res['halo'])
```

Returned keys:
- `disk`: best-fit `DiskParams`
- `halo`: best-fit halo parameters (`NFWParams` or `BurkertParams`)
- `model`: list of fitted total velocities (m/s)
- `chi2`: minimized objective value
- `halo_type`: tag indicating which halo profile was used

Usage recommendations:
- Start with broad, physically plausible bounds (avoid artificially constraining degeneracies between $M_d$ and $\rho_s$ or $r_s$).
- Increase `n_random` for galaxies with extended flat regions (degeneracy widened by outer plateau).
- Compare AIC/BIC proxies later to penalize additional parameters relative to the medium model.

Interpreting results:
- If joint disk+halo significantly outperforms medium (large systematic reduction in frac_rms and outer_delta neutral), the medium needs refinement or extra constraints.
- If medium matches or beats joint fits across many galaxies without invoking unseen mass, it strengthens the hypothesis of a substrate-induced modification.

Future extensions:
- Add optional priors linking disk scale $R_d$ and halo scale $r_s$ (empirical concentration-like relations).
- Provide simultaneous multi-model evaluation returning a ranked list per galaxy.
- Incorporate anisotropy or flaring corrections for the disk as additional (optional) parameters.

### 11.12 Model Comparison Pipeline
The repository includes a unified comparison utility to evaluate (a) medium+disk, (b) halo-only, and (c) joint disk+halo fits for each galaxy.

Core functions:
- `compare_models(rc, disk_bounds, medium_bounds, halo_bounds, halo_type='nfw', ...)` → returns structured summary for a single `RotationCurve`.
- `aggregate_statistics(summaries)` → aggregates median/mean metrics across a set of summaries.
- `export_comparison_results(path, summaries)` → writes a CSV with per-galaxy chi-square and key residual metrics.

Example:
```python
from galaxy_dynamics import (
    load_sparc_real, compare_models, aggregate_statistics,
    export_comparison_results
)

curves = load_sparc_real('sparc_catalog.csv', return_dict=True)
disk_bounds = {'M_d': (5e39, 5e41), 'R_d': (1e19, 2e20)}
medium_bounds = {'v_inf': (5e4, 3.5e5), 'r_s': (1e19, 2e20), 'r_c': (5e18, 8e19), 'm': (0.8, 4.0)}
halo_bounds = {'rho_s': (1e-24, 1e-20), 'r_s': (1e19, 3e20)}

summaries = []
for name, rc in list(curves.items())[:25]:  # first 25 galaxies
    summaries.append(compare_models(rc, disk_bounds, medium_bounds, halo_bounds))

stats = aggregate_statistics(summaries)
print('Median medium frac_rms:', stats['medium_frac_rms']['median'])
print('Median halo frac_rms:', stats['halo_frac_rms']['median'])

export_comparison_results('model_comparison.csv', summaries)
```

Metrics tracked per model:
- `chi2`: raw chi-square (no dof normalization; same datasets allow relative comparison).
- `frac_rms`: scale-insensitive residual magnitude.
- `outer_delta`: sign / magnitude of outermost point mismatch (plateau fidelity).
- `inner_slope_ratio`: relative shape of inner rise (core/cusp tension analogue).

Suggested interpretation workflow:
1. Verify medium vs joint disk+halo frac_rms medians.
2. Inspect outer_delta bias (systematic sign indicates asymptotic mismatch).
3. Examine distribution of `m` (medium) vs `r_s` (halo) for plausible scaling trends with `R_d`.
4. Apply information criteria (future) to weigh parameter count differences.

Planned enhancements:
- Optional parallel execution for large galaxy samples.
- Confidence interval estimation via bootstrap over radii or Monte Carlo noise realizations.
- Automated scaling relation extraction (e.g., Baryonic Tully–Fisher) directly from summary objects.




## 11. Skeletons & flow‑map modelling

This subsection explains how to extract the qualitative transport skeleton of substrate fields used in the pushing‑medium model. "Skeletons" here means the set of stationary points (equilibria), their linearized classification, and the invariant manifolds (stable/unstable separatrices) and ridge/valley lines that organise transport and lensing.

### 11.1 Conceptual summary

- For scalar landscapes (for example, an effective potential Φ(x,y) or an index field `n_eff`), skeletons include minima, maxima, saddles, and the ridge/valley lines (crests and troughs) that separate basins.
- For vector flows (substrate velocity `u` or rotating‑frame `uOmega`), skeletons are built from stagnation points where `uOmega(r)=0`; the Jacobian at a stagnation point gives eigenvectors whose unstable/stable manifolds trace separatrices in the flow.

### 11.2 Where to find the code

The repository already contains two complementary implementations:

- `Physics/vector_skeleton.py` — root‑finding, Jacobian, classification, and manifold tracing for 2D rotating‑frame substrate flows. Key API elements:
  - `Body(pos, gamma)` — positional sources and strengths
  - `VectorFlow(bodies, omega, eps2=...)` — flow factory
  - `VectorFlow.u(r)` — base substrate inflow
  - `VectorFlow.uOmega(r)` — flow including rotating‑frame term
  - `VectorFlow.J(r)` — analytic Jacobian of `uOmega`
  - `VectorFlow.newton_root(r0, tol, itmax)` — Newton solver for stagnations
  - `VectorFlow.classify(rstar)` — eigenvalue/eigenvector analysis and label
  - `VectorFlow.trace_manifold(stag, kind, ds, max_steps, rmax)` — trace stable/unstable manifolds
  - `seeds_two_body(...)`, `continue_stagnations(...)` — helpers for seeding and parameter sweeps

- `Physics/demos/skeleton_finder.py` — grid‑based scalar skeleton extraction (ridges/valleys and stationary point detection) using finite differences and image‑processing style thinning. Use this when working from `n_eff(x,y)` or any scalar Φ defined on a grid.

### 11.3 Practical workflow

1. Choose representation
    - If you have an analytic or particle-sourced flow, use `VectorFlow` and the stagnation/manifold machinery.
    - If you have a gridded scalar field (e.g., `n_eff` on a fixed 2D grid), run `skeleton_finder.py` to extract ridges/valleys and stationary points.

2. Seed and locate stagnations
    - Use physics‑informed seeds such as `seeds_two_body(...)` for two‑body-like setups, or derive seeds from a coarse grid scan (look for sign changes in `uOmega` components or small |uOmega| regions).
    - Run `VectorFlow.newton_root(seed)` to converge to stagnation points. If the Jacobian is singular or Newton fails, try slightly different seeds or increase softening `eps2`.

3. Classify and inspect linearization
    - Use `VectorFlow.classify(r)` to obtain eigenvalues/eigenvectors and an initial label (saddle/node/center). Plot eigenvectors at the stagnation location to confirm orientation.

4. Trace manifolds
    - For saddles, call `trace_manifold(stag, 'unstable')` and `'stable'` to integrate outward along eigenvector directions. Overlay the traced polylines on streamplots or scalar contours to verify they align with separatrices.

5. Parameter sweeps
    - Use `continue_stagnations(flow_factory, phases, r0_list)` to follow stagnations as bodies move (e.g., sweep the moon phase) and observe bifurcations or annihilations of equilibria.

### 11.4 Numerical tips and tuning

- Softening (`eps2`): increase if Newton fails near point masses; typical start values range 1e‑6 → 1e‑3 depending on geometric units.
- Newton tolerance `tol`: 1e‑10–1e‑12 gives high precision; relax if solver unstable (1e‑8). Monitor ||uOmega(r)|| and step norm.
- Manifold step size `ds`: 1e‑3 → 5e‑3 is usually stable; consider adaptive stepping (smaller ds where curvature large).
- `rmax` and `max_steps`: bound the tracing domain to prevent runaway traces into far field; choose based on your system scale.
- Jacobian conditioning: compute cond(J) or eigenvalue gaps — near‑singular J indicates potential failure or nearby bifurcation.

### 11.5 Diagnostics and tests

- Plot streamlines of `u` or `uOmega` and overlay stagnation points and manifold polylines — the manifolds should sit on separatrices between streamline basins.
- Validate against scalar skeletons: for cases where `n_eff` (or Φ) is defined, run `skeleton_finder.py` and compare ridge/valley lines with manifolds from `VectorFlow`.
- Track conservation: if flows are derived from a potential, verify curl ≈ 0 away from rotating frame contributions; large curl signals non‑conservative behaviour.

### 11.6 Common failure modes & remedies

- Newton fails (LinAlgError): increase `eps2`, perturb seed, fall back to quasi‑Newton (Broyden) or use `continue_stagnations` with neighbouring successful roots as seeds.
- Manifold tracing stalls (very small |v|): reduce `ds`, or stop and treat as terminus (manifold reached another equilibrium or an asymptotic region).
- Spirals / complex eigenvalues: manifold tracing along real eigenvectors is not applicable — treat as local spirals and plot phase portraits instead.

### 11.7 Supplementary implementation notes (potential additions)
- Example `VectorFlow` usage snippet (classification + manifolds).
- Guidance on parameter selection (`eps2`, `tol`, `ds`, `rmax`).

- Why it matters: a substrate implies modified vacuum structure which can alter semiclassical predictions (Hawking radiation, vacuum polarization).
- Signature: modified particle emission spectra near compact objects; decoherence signatures.
- Test: model simple quantum fields on a dispersive medium background to estimate deviations from standard semiclassical predictions.

Short prioritized project concepts
- GW dispersion demo: implement a toy 1D wave propagation code with tunable dispersion to generate templates and bounds (useful and fast).
- Effective metric mapping: derive conditions under which index+flow map to an effective metric and parameterize deviations (PPN‑style).
- Horizon echo simulation: ray tracing on a model substrate that stiffens near a core to look for echoes in time series.

Observational constraints and cautions
- Many constraints (GW speed, Lorentz tests, solar‑system PPN) are already tight; pushing‑medium parameter space must be chosen to respect these. Some signatures may be degenerate with astrophysical noise.
- Care is required to avoid ad‑hoc tuning: focus on physically motivated constitutive relations and explore falsifiable predictions.

### Local perturbations with remote amplification

The pushing‑medium formulae contain several explicit mechanisms that allow small, local perturbations to produce large remote effects. Below are the primary amplification channels and short computational/experimental checks you can run.

1) Nonlocal kernel (1/|r−r'|)
- Source: continuous index formula
    $$n(\mathbf{r}) = 1 + \frac{2G}{c^2} \int \frac{\rho(\mathbf{r}')}{|\mathbf{r}-\mathbf{r}'|} \, d^3r'$$
- Mechanism: a point perturbation Δρ at r0 changes n at r by Δn(r) ∝ Δm/|r−r0|. Although decaying, this is nonzero arbitrarily far away and can be exploited by focusing geometries or resonant paths.
- Quick check: compute Δn(r) analytically for a point Δm and plot vs |r−r0|; integrate along a ray bundle to estimate accumulated phase shift.

2) Path accumulation (Fermat functional)
- Source: Fermat functional and ray advection
    $$T=\int n\,ds - \frac{1}{c}\int (n^2-1)(\mathbf{u}_g\cdot\hat{\mathbf{k}})\,ds$$
- Mechanism: small Δn along a long or critical path integrates to a large ΔT (phase/time delay). Interferometers and long baseline rays are especially sensitive.
- Quick check: add a narrow Δn bump into a lensing simulation and measure ΔT for long‑baseline vs short‑baseline rays.

3) Focusing and caustics
- Source: spatial dependence of n in ray equations
- Mechanism: lenses concentrate energy at caustics; tiny lens changes shift caustic location producing huge intensity changes at the focal plane.
- Quick check: ray trace through an asymmetric lens and compute focal intensity sensitivity to small core perturbations.

4) Flow-induced Doppler and directional effects
- Source: advection term in ray velocity and wave Doppler shifts from u_g
- Mechanism: small u_g along an extended region causes cumulative Doppler shifts or preferred transmission for one direction, potentially moving waves into resonance at remote sites.
- Quick check: simulate 1D wave packet crossing a localized u_g and measure transmitted amplitude and frequency shift vs u_g.

5) Nonlinear thresholding and phase change
- Source: substrate constitutive nonlinearity n(ρ_s,P_s)
- Mechanism: local seed perturbations may nucleate a phase that expands (like a combustion front), converting small local changes into macroscopic regions of altered n.
- Quick check: evolve a bistable 1D substrate PDE with a small localized perturbation and monitor front propagation.

6) Parametric / resonant amplification via time modulation
- Source: time‑dependent n(r,t)
- Mechanism: local periodic pumping can parametrically amplify certain spatial modes, producing large distant responses if resonance conditions are met.
- Quick check: 1D Floquet simulation with local time modulation and measure transmitted mode amplitudes.

Practical notes
- Even though these channels permit amplification, physical constraints (1/|r| decay, causality, observational bounds) limit dramatic long‑range amplification unless combined with focusing, resonance, or nonlinear growth.
- These mechanisms are useful for designing sensitive tests: interferometric setups, long‑baseline timing, and lensing caustics are promising probes.

### Information leakage & substrate channels

The pushing‑medium picture explicitly exposes substrate channels that can carry, degrade, or reroute information compared with the pure‑geometry picture of GR. Below are the principal leakage channels, their physical origin, observational signatures, and quick checks.

1) Nonlocal encoding via the Poisson kernel
- Origin: $n(\mathbf{r})$ depends on the integral of mass/substrate over space, so local events imprint a global field.
- Signature: small local mass/substrate changes produce distant Δn and therefore phase/time shifts detectable by long‑baseline interferometry.
- Check: analytic Δn(r) for a localized Δm and integrated ray delay.

2) Substrate internal modes (phonons, vorticity)
- Origin: substrate supports microscopic excitations; gravitational disturbances can excite them, transferring information out of the bulk gravitational channel.
- Signature: excess damping, thermal-like emission, or delayed substrate radiative tails.
- Check: couple a wave equation to a damped harmonic oscillator bath and measure energy transfer.

3) Dissipative loss and entropy production
- Origin: viscous or resistive substrate dynamics convert coherent waves into heat/noise.
- Signature: decoherence, mismatch between emitted gravitational energy and observed coherent signal.
- Check: 1D viscous wave simulation measuring coherent vs dissipated energy.

4) Mode conversion and polarization leakage
- Origin: anisotropic/inhomogeneous n and u_g allow mixing between polarization channels and scalar substrate modes.
- Signature: cross‑polarization signals, reduced signal fidelity in detectors tuned to GR polarizations.
- Check: propagate polarized wave through layered anisotropic index and measure mixing fractions.

5) Flow transport and advective routing
- Origin: u_g advects substrate excitations and wavepackets, moving information along flow lines away from naive geometrical predictions.
- Signature: Doppler‑shifted components, direction‑dependent arrival times.
- Check: simulate wave crossing a Gaussian flow patch and record transmitted spectra vs direction.

6) Porous horizons and tunnelling leakage
- Origin: refractive trapping regions may be frequency‑selective and leaky, allowing partial tunnelling of information across would‑be horizons.
- Signature: echoes, early/late weak signals correlated with merger events.
- Check: ray/wave tunnelling analysis across an index well with frequency‑dependent transmissivity.

7) Disorder trapping and late‑time leakage
- Origin: disorder in n(r) can localize energy and release it slowly via tunnelling, spreading information over long times.
- Signature: long‑tail, frequency‑dependent late emission.
- Check: 1D random‑index FDTD with localized injection and measurement of late‑time leakage.

8) Resonant parametric channels
- Origin: local time‑dependent modulation can pump resonant distant modes, transferring information via narrow frequency channels.
- Signature: stimulated distant emission at resonant frequencies; high sensitivity to modulation parameters.
- Check: Floquet analysis of a local modulated patch and remote response measurement.

Practical constraints
- Any viable model must respect causality and observational bounds (GW speed, PPN limits). Leakage may be subtle and often converts coherent information into thermal/stochastic noise rather than usable signal.



---

## 11. Gravitational Manipulation, Valves & Diodes

One of the intuitions that the pushing‑medium picture makes natural is that, if gravity is mediated by a material substrate with spatially varying refractive index and flow, it may be possible in principle to design structures that steer, block, or permit one‑way propagation of gravitational effects in an analogous way to optical devices (lenses, isolators, and diodes). Below we summarize concepts, simple toy models, feasible experiments, and fundamental caveats.

### 11.1 Terminology and concepts
- Gravitational manipulation: active or passive shaping of the substrate state (index $n$ and flow $\mathbf{u}_g$) to control trajectories of rays/particles and wave propagation.
- Gravitational valve / diode: a configuration that allows transmission of gravitational influence (rays or waves) in one direction but strongly suppresses it in the opposite direction (non‑reciprocal behaviour).

### 11.2 Mechanisms available in the pushing‑medium picture
- Index gradients and asymmetric geometry: spatially non‑symmetric index profiles can produce strongly direction‑dependent ray deflections (geometric valve).
- Flow advection (non‑reciprocity): a background flow $\mathbf{u}_g(\mathbf{r})$ breaks time‑reversal symmetry for wave/ray propagation and can produce Doppler‑shifted, non‑reciprocal transmission (flow diode).
- Active modulation / time‑dependence: time‑periodic modulation of $n$ or $\mathbf{u}_g$ (parametric pumping) can produce one‑way bandgaps and isolation similar to Floquet isolators in photonics.
- Nonlinear substrate response: thresholded or hysteretic substrate behaviour (phase change, stiffening) can act like a valve that opens for strong incident flux but closes for weak or reverse flux.

### 11.3 Simple toy models (equations and intuition)
- Asymmetric lens (passive): a 2D index profile
    $$n(x,y)=1+\varepsilon\,\frac{1+\alpha\,\mathrm{tanh}(x/L)}{\sqrt{x^2+y^2+\epsilon^2}}$$
    For large positive $x$ the effective lensing strength differs from the negative side, producing direction‑dependent deflection angles for rays traversing the region.
- Flow diode (advection): add a localized flow patch
    $$\mathbf{u}_g(x,y)=U_0\,e^{-r^2/\sigma^2}\,\hat{x}$$
    Waves/rays propagating with the flow are Doppler‑boosted and can pass; those against the flow encounter an effective barrier (blue/red shifted dispersion relations).
- Active Floquet valve: time modulation
    $$n(\mathbf{r},t)=n_0(\mathbf{r})+\delta n(\mathbf{r})\cos(\Omega t)$$
    With suitable spatial asymmetry this can create directional bandgaps for wave propagation (one‑way windows in frequency/wavevector space).

### 11.4 Simulation & experiment suggestions (small projects)
- Ray‑tracing demo: implement a 2D ray tracer through the asymmetric lens above and show forward vs reverse transmission maps (transmission coefficient vs impact parameter).
- Wave packet simulation: 1D/2D finite‑difference propagation of small amplitude waves in presence of a Gaussian flow patch and measure transmission asymmetry vs flow speed $U_0$ and frequency.
- Laboratory analogue: build a metamaterial plate or water‑tank experiment where an index analogue (sound speed or refractive index) is varied spatially and temporally to show non‑reciprocal wave transmission.

### 11.5 Practical limitations and fundamental constraints
- Causality and energy: any valve/diode must respect causality — the substrate characteristic speeds (signal speed) must not allow superluminal signalling. Energy conservation requires attention to active modulation sources.
- Observational constraints: large, static one‑way gravitational devices would likely have been noticed astrophysically; realistic devices are small and subtle and must avoid violating solar‑system bounds.
- Reciprocity in linear, time‑invariant conservative systems: passive, linear, lossless media are reciprocal. Non‑reciprocity requires either flow (breaking T‑symmetry), nonlinearity, loss/gain, or time‑dependence.

### 11.6 Potential applications (speculative)
- Gravitational shielding analogue (local redirection of trajectories for debris mitigation) — likely tiny effects only.
- Directional gravitational couplers for laboratory tabletop experiments — allow controlled injection or extraction of substrate waves.
- Black‑hole mimics with asymmetric echoes — testbeds for echo searches.

### 11.7 Safety, ethics, and claims
- Any claim about controllable, engineered gravitational forces should be treated cautiously. Even in the pushing‑medium picture the magnitudes are likely tiny for plausible substrate parameter choices.
- Emphasize lab analogues and scaled demonstrations (acoustic, optical, or fluid) rather than claims of practical propulsion or macroscopic gravity control.

---

## 12. Further Unexplored Phenomena

The pushing‑medium framework suggests several additional phenomena that are natural to explore but not typically discussed in GR. Each offers different observational or experimental handles.

1) Topological defects (vortices, domain walls)
- Why: defects create long‑range phase structures in $n$ and $\mathbf{u}_g$, producing Aharonov–Bohm–like effects for rays and waves.
- Quick test: introduce a vortex u_g in a 2D wave simulation and look for quantized phase jumps or wavefront splitting.

2) Substrate anisotropic inertia / effective mass changes
- Why: local substrate state may modify inertial response, producing apparent mass variations or modified orbital frequencies.
- Quick test: integrate two‑body orbits where one body passes through a high‑n patch and measure period shifts.

3) Modified gravitational memory (viscoelastic substrate)
- Why: substrate relaxation can store residual strain, changing permanent displacement (memory) signatures from transient waves.
- Quick test: propagate a TT pulse in a viscoelastic 1D medium and measure residual particle displacements.

4) Tidal heating and viscous dissipation
- Why: substrate viscosity converts orbital energy to heat, altering inspiral rates and emitted signals.
- Quick test: add a simple viscous drag term to orbit integrators and compare inspiral time vs vacuum GR.

5) Mode conversion and extra polarizations
- Why: anisotropic, layered or nonlinear media mix polarizations, producing detectable non‑GR polarization content.
- Quick test: wave propagation through anisotropic layered n(r) and measure polarization mixing.

6) Anderson localization of substrate waves
- Why: disorder in n(r) can localize waves, suppressing long‑range GW transport.
- Quick test: 1D random‑index FDTD to map localization length vs disorder strength.

7) Stimulated emission and resonance (Cherenkov‑like effects)
- Why: moving masses can excite substrate modes, leading to resonant energy transfer or stimulated emission.
- Quick test: sweep source speed in 1D medium and look for thresholds where transmitted energy grows rapidly.

8) Superradiant scattering from rotating flow regions
- Why: rotating u_g regions can amplify incident waves, analogous to black‑hole superradiance.
- Quick test: scatter wave packets off a rotating flow patch and measure amplification vs rotation rate.

9) Quantum decoherence via substrate coupling
- Why: substrate fluctuations serve as a decohering environment for quantum fields or interferometers.
- Quick test: model stochastic substrate noise and compute expected decoherence rates for atom interferometry near masses.

10) Chaotic collective dynamics from substrate back‑reaction
- Why: coupled N‑body + substrate feedback can introduce new resonances and chaos beyond Newtonian expectations.
- Quick test: compute Lyapunov exponents for small N systems with and without substrate coupling.

11) Porous / leaky horizon analogues
- Why: substrate horizons may be frequency‑dependent or leaky, producing echo structures and partial transparency.
- Quick test: ray/wave propagation near an index well with absorption/dispersion to look for echoes.

12) Catalytic large emissions from metastable substrate modes
- Why: metastable substrate modes could be triggered by small seeds to emit large energy bursts.
- Quick test: bistable PDE with localized trigger and measurement of emitted wave energy after nucleation.

These topics can be expanded with quantitative scaling relations in future revisions.



## References and Further Reading

### Primary Sources
- Core formula documentation in `all_formulas.tex`
- Comparison table in `differences.tex`
- Implementation examples in `Physics/demos/`

### Key Demonstrations
- Gravitational lensing: `demos/grav_lensing.py`
- N-body dynamics: `demos/pushing_medium_nbody_rays.py`
- Three-body flow topology: `three_body_flow.py`
- Cosmological distances: `demos/cosmology_lite_distances.py`

### Computational Framework
- Core physics library: `demos/library/physics.py`
- Utility functions: `demos/library/examples.py`
- Structure documentation: `demos/library/structure.txt`

---

*This document provides a comprehensive overview of the Pushing-Medium gravitational model, covering its theoretical foundations, mathematical formulation, computational implementation, and relationship to General Relativity. It serves as both a technical reference and an introduction to this alternative approach to gravitational physics.*
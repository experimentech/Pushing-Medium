# Pushing–Medium Model: Complete Formula Documentation and Comparison to General Relativity

This document presents all key formulas of the Pushing–Medium gravitational model, with explanations and a functional comparison to General Relativity (GR).

---

## 1. Static Refractive Index Field from Point Masses

$$
n(\mathbf{r}) = 1 + \sum_i \frac{\mu_i}{|\mathbf{r} - \mathbf{r}_i|}, \quad \mu_i = \frac{2GM_i}{c^2}
$$
**Interpretation:**
Masses increase the refractive index locally, with the effect falling off as $1/r$. This is analogous to the gravitational potential in Newtonian gravity, but here it modifies the medium's optical properties.

**GR Comparison:**
GR uses the metric tensor $g_{\mu\nu}$ to encode curvature, not a refractive index. The gravitational potential is a weak-field approximation.

---

## 2. Gradient of Refractive Index

$$
\nabla n(\mathbf{r}) = - \sum_i \frac{2GM_i}{c^2 |\mathbf{r} - \mathbf{r}_i|^3} (\mathbf{r} - \mathbf{r}_i)
$$
**Interpretation:**
Describes how the refractive index changes in space, determining the direction and strength of gravitational effects.

**GR Comparison:**
In GR, the gradient of the metric determines geodesic deviation and tidal forces.

---

## 3. Flow Field (Frame–Drag Analogue)

$$
\mathbf{u}_g(\mathbf{r}) = \sum_i \boldsymbol{\Omega}_i \times (\mathbf{r} - \mathbf{r}_i)
$$
**Interpretation:**
Models frame-dragging effects by advecting rays and particles, similar to the Lense-Thirring effect in GR.

**GR Comparison:**
Frame-dragging arises naturally from off-diagonal metric terms (e.g., Kerr metric).

---

## 4. Time-Dependent Wave Perturbation

$$
\delta n_{\mathrm{wave}}(\mathbf{r}, t) = A \cos(\mathbf{k} \cdot \mathbf{r} - \omega t)
$$
**Interpretation:**
Represents gravitational waves as explicit modulations of the refractive index.

**GR Comparison:**
Gravitational waves are oscillations of spacetime curvature, derived from Einstein's equations.

---

## 5. Total Refractive Index Field

$$
n_{\mathrm{total}}(\mathbf{r}, t) = n(\mathbf{r}) + \delta n_{\mathrm{wave}}(\mathbf{r}, t)
$$
**Interpretation:**
Sum of static and dynamic contributions to the medium's refractive index.

---

## 6. Ray Equation of Motion (Fermat's Principle)

$$
\frac{d\hat{\mathbf{k}}}{ds} = \nabla_{\!\perp} \ln n_{\mathrm{total}}(\mathbf{r}, t)
$$
**Interpretation:**
Light rays bend toward regions of higher refractive index, following Fermat's principle.

**GR Comparison:**
Light follows null geodesics in curved spacetime; bending emerges from the metric.

---

## 7. Ray Advection by Flow Field

$$
\frac{d\mathbf{r}}{dt} = \frac{c\,\hat{\mathbf{k}}}{n_{\mathrm{total}}} + \mathbf{u}_g(\mathbf{r}, t)
$$
**Interpretation:**
Ray velocity is affected by both the refractive index and the flow field.

---

## 8. Massive Particle Acceleration (Optical–Mechanical Analogy)

$$
\mathbf{a}_{\mathrm{med}}(\mathbf{r}, t) = -c^2 \nabla \ln n_{\mathrm{total}}(\mathbf{r}, t)
$$
**Interpretation:**
Massive particles experience acceleration due to gradients in the refractive index, analogous to gravity.

**GR Comparison:**
Massive bodies follow timelike geodesics in curved spacetime.

---

## 9. Newtonian Gravity Term for Matter

$$
\mathbf{a}_{\mathrm{grav}}(\mathbf{r}) = - \sum_i \frac{G M_i (\mathbf{r} - \mathbf{r}_i)}{|\mathbf{r} - \mathbf{r}_i|^3}
$$
**Interpretation:**
Standard Newtonian gravitational acceleration, included for reference and comparison.

---

## Functional Comparison Table

| Aspect | Pushing–Medium Model | General Relativity |
|--------|----------------------|-------------------|
| Underlying picture | Flat, fixed Euclidean background. Gravity-like effects from medium with varying refractive index $n(\mathbf{r},t)$ and flow field $\mathbf{u}_g$. | Spacetime is dynamic and curved. Matter/energy changes curvature; curvature dictates motion. |
| Core variables | Scalar $n(\mathbf{r},t)$ + vector $\mathbf{u}_g(\mathbf{r},t)$ fields. Optional wave perturbations. | Metric tensor $g_{\mu\nu}(x^\alpha)$ encodes all distances, times, and causal structure. |
| Source–effect link | Sources (masses) plugged into formulas for $n$ and $\mathbf{u}_g$. Phenomenological, not covariant. | Einstein field equations link stress–energy to curvature. |
| Motion of matter | Newtonian $N$-body integration in flat space. | Timelike geodesics in curved spacetime. |
| Motion of light | Fermat’s principle: rays bend toward higher $n$; $\nabla n$ changes direction. Optional advection by $\mathbf{u}_g$. | Null geodesics in curved spacetime. |
| Frame–drag analogue | Explicit $\mathbf{u}_g$ term advects rays/particles. | Off-diagonal metric terms produce frame-dragging. |
| Gravitational waves | Added as explicit modulations of $n$, not emergent. | Oscillations of spacetime curvature, propagating at $c$. |
| Time evolution | Medium changes as sources move; $n$ and $\mathbf{u}_g$ recomputed each tick. | Metric evolves via coupled PDEs with matter dynamics. |
| Computational load | Lightweight: direct evaluation of $n$, $\nabla n$, $\mathbf{u}_g$ and ray/body integration. | Heavy: solve nonlinear PDEs for $g_{\mu\nu}$; numerical relativity. |
| Strengths | Intuitive, modular, fast to simulate, pedagogically clear. | Physically complete, matches all tested regimes, Lorentz invariant. |
| Limitations | Not covariant; no guarantee in strong-field/high-velocity regimes; wave/drag effects are modelled by hand. | Computationally expensive; less intuitive; exact solutions rare. |

---

**Summary:**
The Pushing–Medium model offers an intuitive, modular, and computationally efficient alternative to General Relativity, using optical analogies and explicit formulas. However, it lacks the covariant foundation and completeness of GR, especially in strong-field and relativistic regimes.

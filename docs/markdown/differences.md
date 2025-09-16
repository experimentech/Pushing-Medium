
---

| **Aspect** | **Pushing‑Medium Model** | **General Relativity** |
|------------|--------------------------|------------------------|
| **Underlying picture** | Flat, fixed Euclidean background. Gravity‑like effects come from a medium with spatially/temporally varying refractive index \(n(\mathbf{r},t)\) and optional flow field \(\mathbf{u}_g\). | Spacetime itself is dynamic and curved. Matter/energy changes curvature; curvature dictates motion. |
| **Core variables** | Scalar \(n(\mathbf{r},t)\) + vector \(\mathbf{u}_g(\mathbf{r},t)\) fields. Optional explicit wave perturbations. | Metric tensor \(g_{\mu\nu}(x^\alpha)\) encodes all distances, times, and causal structure. |
| **Source–effect link** | Sources (masses) are plugged into chosen formulas for \(n\) and \(\mathbf{u}_g\). Relationship is phenomenological, not derived from a covariant field equation. | Einstein field equations \(G_{\mu\nu} = \frac{8\pi G}{c^4}T_{\mu\nu}\) link stress–energy to curvature in a self‑consistent way. |
| **Motion of matter** | Massive bodies: Newtonian \(N\)-body integration in flat space. | Massive bodies: timelike geodesics in curved spacetime. |
| **Motion of light** | Fermat’s principle in a medium: rays bend toward higher \(n\); perpendicular component of \(\nabla n\) changes direction. Optional advection by \(\mathbf{u}_g\). | Null geodesics in curved spacetime; bending emerges from the metric. |
| **Frame‑drag analogue** | Explicit \(\mathbf{u}_g\) term advects rays/particles. | Off‑diagonal metric terms (e.g., Kerr metric) produce frame‑dragging naturally. |
| **Gravitational waves** | Added as explicit sinusoidal modulations of \(n\) (TT‑like), not emergent. | Oscillations of spacetime curvature itself, propagating at \(c\), derived from Einstein equations. |
| **Time evolution** | Medium changes because sources move; \(n\) and \(\mathbf{u}_g\) recomputed each tick. | Metric evolves according to coupled PDEs with matter dynamics. |
| **Computational load** | Lightweight: direct evaluation of \(n\), \(\nabla n\), \(\mathbf{u}_g\) and ray/body integration. Modular toggles for effects. | Heavy: solve nonlinear PDEs for \(g_{\mu\nu}\) in 3+1D; numerical relativity for general cases. |
| **Strengths** | Intuitive optical analogy, modular, interactive, pedagogically clear, fast to simulate. | Physically complete within its domain, matches all tested regimes, built‑in Lorentz invariance. |
| **Limitations** | Not derived from a covariant theory; no guarantee in strong‑field/high‑velocity regimes; wave/drag effects are modelled by hand. | Computationally expensive; less intuitive for newcomers; exact solutions rare. |

---


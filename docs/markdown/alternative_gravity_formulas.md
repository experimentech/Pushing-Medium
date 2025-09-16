# Alternative Gravity Model: Key Formulas

This document presents the main formulas used in the alternative gravitational model explored in this project. The notation follows standard conventions for physics and vector calculus.

---

## 1. Refractive Index Field

$$
n(r) = 1 + \sum_{i=1}^N \left[ \frac{2GM_i}{c^2 |\mathbf{r} - \mathbf{r}_i|} \right]
$$

## 2. Gradient of Refractive Index

$$
\nabla n(r) = - \sum_{i=1}^N \left[ \frac{2GM_i}{c^2 |\mathbf{r} - \mathbf{r}_i|^3} (\mathbf{r} - \mathbf{r}_i) \right]
$$

## 3. Gravitational Flow Field

$$
u_g(r) = \sum_{i=1}^N \left[ M_i \frac{\hat{z} \times (\mathbf{r} - \mathbf{r}_i)}{|\mathbf{r} - \mathbf{r}_i|^2 + \epsilon^2} \right]
$$

## 4. Transverse-Traceless Perturbation

$$
\delta n_{TT}(r, t) = A \sin(\mathbf{k} \cdot \mathbf{r} - \omega t + \phi_0)
$$

---

* $G$ — Gravitational constant
* $c$ — Speed of light
* $M_i$ — Mass of body $i$
* $\mathbf{r}$ — Field point
* $\mathbf{r}_i$ — Position of body $i$
* $\hat{z}$ — Unit vector in $z$ direction
* $\epsilon$ — Regularization parameter
* $A$ — Amplitude
* $\mathbf{k}$ — Wave vector
* $\omega$ — Angular frequency
* $\phi_0$ — Phase offset

---

This summary provides a readable reference for the main mathematical structures in the model.

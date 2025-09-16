# Pushing‑Medium Gravity — Demonstration Suite

This repository contains a set of **self‑contained Python scripts** that bring the core equations of the pushing‑medium gravitational model to life.  
Each script is designed to be:

- **Minimal** — no unnecessary dependencies beyond NumPy, Matplotlib, and SciPy.
- **Transparent** — every formula is annotated with its origin in the model.
- **Reproducible** — fixed seeds and parameters for consistent results.

---

## 📂 Contents

### 1. `cosmology_lite_distances.py`
**Demonstrates:** Cosmology‑lite distance relations  
**Formulae:**  


\[
D_C(z) = \int_0^z \frac{c\,dz'}{H(z')},\quad D_L = (1+z)D_C,\quad D_A = \frac{D_C}{1+z}
\]

  
**What it does:**  
Plots comoving, luminosity, and angular‑diameter distances for \(0 \le z \le 3\) using a simple \(H(z)\) from the expanding‑substrate ansatz.  
**Output:** PNG plot of \(D_C, D_L, D_A\) vs \(z\).

---

### 2. `substrate_fluid_1d.py`
**Demonstrates:** Substrate continuity and momentum equations  
**Formulae:**  


\[
\partial_t n + \partial_x(n v) = 0
\]

  


\[
\partial_t(n v) + \partial_x(n v^2 + P(n)) = 0
\]

  
**What it does:**  
Simulates a 1D density pulse evolving under a barotropic equation of state \(P(n) = n^2\). Shows density and velocity profiles over time.  
**Output:** Animated or static plot sequence.

---

### 3. `gravito_magnetic_spin.py`
**Demonstrates:** Rotational gravito‑magnetic flow (frame‑drag)  
**Formula:**  


\[
\omega_s(r) = \frac{2 G J}{c^2 r^3}
\]

  
**What it does:**  
Plots \(\omega_s(r)\) for a given angular momentum \(J\) over a range of radii.  
**Output:** Log‑log plot showing \(r^{-3}\) fall‑off.

---

### 4. `photon_travel_time.py`
**Demonstrates:** Fermat functional in a static lens  
**Formula:**  


\[
T = \int n\,ds - \frac{1}{c} \int (n^2 - 1)(\mathbf{u}_g \cdot \hat{\mathbf{k}})\,ds
\]

  
**What it does:**  
Computes travel time vs impact parameter for a Gaussian index profile \(n(r) = 1 + \varepsilon e^{-r^2}\) with \(\mathbf{u}_g = 0\).  
**Output:** Travel‑time curve showing lensing delay.

---

### 5. `tt_wave_propagation.py`
**Demonstrates:** TT gravitational wave sector  
**Formula:**  


\[
\frac{\partial^2 h}{\partial t^2} - c^2 \frac{\partial^2 h}{\partial x^2} = 0
\]

  
**What it does:**  
Simulates a Gaussian TT pulse propagating at \(c\) in 1D.  
**Output:** Animation or frame sequence of wave motion.

---

## 🛠 Requirements

- Python 3.8+
- NumPy
- SciPy
- Matplotlib

Install dependencies:
```bash
pip install numpy scipy matplotlib


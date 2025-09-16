import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from matplotlib import style

style.use('seaborn-v0_8')

# Constants
H0 = 70e3 / 3.086e22  # s^-1
c = 299792458  # m/s
Omega_m = 0.3
Omega_lambda = 0.7

# Hubble parameter
H = lambda z: H0 * np.sqrt(Omega_m * (1 + z)**3 + Omega_lambda)

# Comoving distance
D_C = lambda z: c * quad(lambda z_: 1 / H(z_), 0, z)[0]

z_vals = np.linspace(0, 3, 100)
DC_vals = np.array([D_C(z) for z in z_vals])
DL_vals = (1 + z_vals) * DC_vals
DA_vals = DC_vals / (1 + z_vals)

# Plot
plt.figure(figsize=(8, 6))
plt.plot(z_vals, DC_vals / 3.086e22, label='D_C (Gpc)')
plt.plot(z_vals, DL_vals / 3.086e22, label='D_L (Gpc)')
plt.plot(z_vals, DA_vals / 3.086e22, label='D_A (Gpc)')
plt.title('Cosmological Distances vs Redshift')
plt.xlabel('Redshift z')
plt.ylabel('Distance (Gpc)')
plt.legend()
plt.grid(True)
plt.savefig('./cosmology_distances.png')
print("Saved cosmology distances plot as cosmology_distances.png")


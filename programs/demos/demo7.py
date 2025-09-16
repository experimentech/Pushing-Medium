import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import os

style.use('seaborn-v0_8')

# Constants
G = 6.67430e-11  # m^3 kg^-1 s^-2
c = 299792458    # m/s
J = 1e42          # kg m^2/s (example angular momentum)

# Radius range
r = np.linspace(1e6, 1e8, 500)  # meters
omega_s = 2 * G * J / (c**2 * r**3)

# Ensure output directory exists
#os.makedirs('/mnt/data', exist_ok=True)

# Plot
plt.figure(figsize=(8, 6))
plt.plot(r, omega_s)
plt.title('Gravito-Magnetic Spin Flow ω_s(r)')
plt.xlabel('Radius r (m)')
plt.ylabel('ω_s (rad/s)')
plt.grid(True)
plt.tight_layout()
plt.savefig('./spin_flow.png')
print("Saved spin flow plot as spin_flow.png")


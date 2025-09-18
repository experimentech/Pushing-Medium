import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

style.use('seaborn-v0_8')

# Constants
c = 299792458  # m/s
epsilon = 1e-6

# Impact parameter range
b_vals = np.linspace(0.1, 5.0, 100)
T_vals = []

# Integrate along straight line path for each impact parameter
for b in b_vals:
    s = np.linspace(-10, 10, 1000)
    r = np.sqrt(s**2 + b**2)
    n = 1 + epsilon * np.exp(-r**2)
    ds = s[1] - s[0]
    T = np.sum(n) * ds - (1/c) * np.sum((n**2 - 1) * 0) * ds  # u_g · k̂ = 0
    T_vals.append(T)

# Plot
plt.figure(figsize=(8, 6))
plt.plot(b_vals, T_vals)
plt.title('Photon Travel Time vs Impact Parameter')
plt.xlabel('Impact Parameter b')
plt.ylabel('Travel Time T (s)')
plt.grid(True)
plt.savefig('./photon_travel_time.png')
print("Saved photon travel time plot as photon_travel_time.png")


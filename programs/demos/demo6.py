import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

style.use('seaborn-v0_8')

# Parameters
nx = 200
nt = 300
dx = 0.1
dt = 0.005
gamma = 2.0  # Pressure law exponent

# Initial conditions
x = np.linspace(0, nx*dx, nx)
n = np.ones(nx)
v = np.zeros(nx)
n[nx//2-10:nx//2+10] = 2.0  # Initial density bump

# Pressure function
P = lambda n: np.clip(n, 1e-6, None)**gamma

# Arrays to store results
n_history = []
v_history = []

# Time evolution
for t in range(nt):
    # Compute pressure
    p = P(n)

    # Compute fluxes
    flux_n = n * v
    flux_p = p

    # Update velocity using momentum equation (Euler method)
    dp = np.diff(flux_p)
    n_avg = 0.5 * (n[1:] + n[:-1])
    dv = -dt/dx * dp / np.clip(n_avg, 1e-6, None)
    v[1:-1] += dv[:-1]

    # Update density using continuity equation
    dflux_n = np.diff(flux_n)
    dn = -dt/dx * dflux_n
    n[1:-1] += dn[:-1]

    # Store snapshots
    if t % 30 == 0:
        n_history.append(n.copy())
        v_history.append(v.copy())

# Plot results
fig, axs = plt.subplots(2, 1, figsize=(10, 8))
for i in range(len(n_history)):
    axs[0].plot(x, n_history[i], label=f'Time {i*30*dt:.2f}')
    axs[1].plot(x, v_history[i], label=f'Time {i*30*dt:.2f}')

axs[0].set_title('Density Evolution')
axs[1].set_title('Velocity Evolution')
for ax in axs:
    ax.set_xlabel('x')
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.savefig('./fluid_dynamics.png')
print("Saved fluid dynamics plot as fluid_dynamics.png")


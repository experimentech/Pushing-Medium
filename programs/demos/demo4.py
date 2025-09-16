import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style

style.use('seaborn-v0_8')

# Parameters
nx = 400
dx = 0.1
dt = 0.01
c = 1.0
nt = 300

x = np.linspace(0, nx*dx, nx)

# Initial conditions: Gaussian pulse
h = np.exp(-((x - nx*dx/2)**2) / 2)
h_new = h.copy()
h_old = h.copy()

# Storage for animation
frames = []
fig, ax = plt.subplots()
line, = ax.plot(x, h)
ax.set_ylim(-1, 1)
ax.set_title('TT Wave Propagation')
ax.set_xlabel('x')
ax.set_ylabel('h')

# Update function
def update(frame):
    global h, h_new, h_old
    h_new[1:-1] = 2*h[1:-1] - h_old[1:-1] + (c*dt/dx)**2 * (h[2:] - 2*h[1:-1] + h[:-2])
    h_old = h.copy()
    h = h_new.copy()
    line.set_ydata(h)
    return line,

ani = animation.FuncAnimation(fig, update, frames=nt, blit=True)
ani.save('./tt_wave_propagation.mp4', fps=30)
print("Saved TT wave propagation animation as tt_wave_propagation.mp4")


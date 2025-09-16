import matplotlib.pyplot as plt
from matplotlib.table import Table
import matplotlib as mpl

# Use a clean style for professional appearance
plt.style.use('seaborn-v0_8')
mpl.rcParams['font.size'] = 14

# Define the data for the table
data = [
    [r"$R_{\mu\nu} - \frac{1}{2}g_{\mu\nu}R = \frac{8\pi G}{c^4}T_{\mu\nu}$", r"Medium stress-energy ↔ curvature response", "Einstein Field Equations ↔ Medium dynamics"],
    [r"$g_{\mu\nu}$", r"Effective medium metric", "Spacetime geometry ↔ Medium structure"],
    [r"$T_{\mu\nu}$", r"Stress-energy in medium", "Energy-momentum ↔ Medium forces"],
    [r"$\nabla_{\mu}T^{\mu\nu} = 0$", r"Conservation of medium flow", "Energy conservation ↔ Flow continuity"],
    [r"$\Gamma^{\lambda}_{\mu\nu}$", r"Medium connection coefficients", "Geodesics ↔ Medium path guidance"],
    [r"$ds^2 = g_{\mu\nu}dx^{\mu}dx^{\nu}$", r"Medium interval measure", "Spacetime interval ↔ Medium distance"],
    [r"$\Box \phi = 0$", r"Wave propagation in medium", "Scalar field ↔ Medium wave dynamics"],
    [r"$\delta S = 0$", r"Medium action principle", "Least action ↔ Medium equilibrium"],
]

# Create figure and axis
fig, ax = plt.subplots(figsize=(14, 8))
ax.set_axis_off()

# Create the table
table = Table(ax, bbox=[0, 0, 1, 1])

# Column headers
columns = ["GR Concept / Equation", "Pushing-Medium Analogue", "Mapping"]
col_widths = [0.4, 0.35, 0.25]

# Add header row
for i, col in enumerate(columns):
    table.add_cell(0, i, width=col_widths[i], height=0.1, text=col, loc='center', facecolor='#d3d3d3')

# Add data rows
for row_idx, row in enumerate(data):
    for col_idx, cell_text in enumerate(row):
        table.add_cell(row_idx + 1, col_idx, width=col_widths[col_idx], height=0.1, 
                       text=cell_text, loc='center', facecolor='white')

# Auto scale the table
for i in range(len(data) + 1):
    table.auto_set_font_size(False)
    table.set_fontsize(14)

ax.add_table(table)
plt.title("GR vs Pushing-Medium: Formula Comparison", fontsize=18, weight='bold')

# Save the image
plt.savefig("/mnt/data/gr_vs_pushing_medium_comparison.png", dpi=300, bbox_inches='tight')
plt.close()
print("Image saved as gr_vs_pushing_medium_comparison.png")


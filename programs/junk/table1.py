import matplotlib.pyplot as plt
import pandas as pd

def save_table_as_png(data, title, filename):
    fig, ax = plt.subplots(figsize=(12, len(data)*0.6 + 2))
    ax.axis('off')
    table = ax.table(cellText=data.values,
                     colLabels=data.columns,
                     loc='center',
                     cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.5)
    plt.title(title, fontsize=16, weight='bold')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

# Example: condensed GR ↔ Pushing‑Medium map
data = pd.DataFrame({
    "GR Concept / Equation": [
        "Metric: ds² = gμν dxμ dxν",
        "Geodesics",
        "Christoffels",
        "Einstein eq.",
        "GW propagation",
        "Frame‑drag (spin)",
        "Moving‑lens gravito‑magnetism",
        "Conservation",
        "Cosmological distances",
        "Growth of structure"
    ],
    "Pushing‑Medium Analogue": [
        "n(ρ_s), u_g",
        "Fermat functional",
        "∇n, ∇u_g",
        "Symmetric‑hyperbolic PDEs",
        "TT wave channel",
        "ω_s(r) = 2GJ/(c² r³)",
        "Retarded u_g in Fermat",
        "Energy identity",
        "Same integrals with H(z) from substrate",
        "Same ODE with effective Ω_m(a)"
    ],
    "Mapping": [
        "Metric ↔ index + flow",
        "Geodesic ↔ optical path",
        "Connection ↔ gradients",
        "Curvature ↔ substrate dynamics",
        "TT ↔ TT",
        "Identical",
        "Same vt/c term",
        "Conservation ↔ flux balance",
        "Identical form",
        "Same structure"
    ]
})

save_table_as_png(data, "GR ↔ Pushing‑Medium Formula Map (Condensed)", "formula_map_condensed.png")


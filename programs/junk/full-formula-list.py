import matplotlib.pyplot as plt
from textwrap import wrap

# Full formula list: [Section, Formula] — mathtext-safe
formulae = [
    ["Substrate fluid (continuity)", r"$\partial_t n + \partial_j(n v_j) = 0$"],
    ["Substrate fluid (momentum)", r"$\partial_t(n v_i) + \partial_j\!\left(n v_i v_j + \delta_{ij}P(n)\right) = \mathcal{F}_i$"],
    ["Sound speed", r"$c_s^2 = \frac{dP}{dn} \leq c^2$"],
    ["Rotational flow (spin)", r"$\omega_s(r) = \frac{2 G J}{c^2 r^3}$"],
    ["Translational flow (mass current)", r"$\frac{1}{c^2}\frac{\partial^2 \mathbf{u}_g}{\partial t^2} - \nabla^2 \mathbf{u}_g = \kappa_J\,\mathbf{J}_{\mathrm{TT}},\ \mathbf{J}=\rho\,\mathbf{v}$"],
    ["Photon travel time", r"$T = \int n\,ds - \frac{1}{c}\int (n^2 - 1)\,(\mathbf{u}_g\cdot\hat{\mathbf{k}})\,ds$"],
    ["TT wave equation", r"$\Box h_{ij}^{\mathrm{TT}} = \mathcal{S}_{ij}^{\mathrm{TT}},\quad \Box \equiv \frac{1}{c^2}\partial_t^2 - \nabla^2$"],
    ["First-order vars (wave)", r"$p_{ij}=\partial_t h_{ij}^{\mathrm{TT}},\quad q_{kij}=\partial_k h_{ij}^{\mathrm{TT}}$"],
    ["State vector", r"$\mathbf{U}=(n,\ m_i,\ h_{ij}^{\mathrm{TT}},\ p_{ij},\ q_{kij}),\ m_i\equiv n v_i$"],
    ["System form", r"$A^0(\mathbf{U})\,\partial_t \mathbf{U} + A^k(\mathbf{U})\,\partial_k \mathbf{U} = \mathbf{R}(\mathbf{U})$"],
    ["Symmetrizer", r"$S_{\mathrm{fl}}=\mathrm{diag}\!\left(\frac{P'(n)}{n},\ \frac{1}{n}\mathbb{I}_3\right),\ S_{\mathrm{TT}}=\mathrm{diag}\!\left(\mathbb{I},\ c^2\mathbb{I}\right)$"],
    ["Energy norm", r"$\mathcal{E}=\tfrac{1}{2}\left[\frac{P'(n)}{n}\,n^2+\frac{|m|^2}{n}+|p_{ij}|^2+c^2|q_{kij}|^2\right]$"],
    ["Energy identity", r"$\partial_t \mathcal{E} + \partial_k \mathcal{F}^k = \mathbf{U}^{\!\top} S\,\mathbf{R}(\mathbf{U}),\quad \mathcal{F}^k=\tfrac{1}{2}\mathbf{U}^{\!\top} S A^k \mathbf{U}$"],
    ["Characteristic speeds", r"$\lambda_\pm=v_n\pm c_s,\ \lambda_0=v_n,\ \lambda_{\mathrm{TT}}=\pm c$"],
    ["Hubble law analogue", r"$H^2(a)=\left(\frac{\dot{a}}{a}\right)^2=\frac{8\pi G_{\mathrm{eff}}}{3}\rho_{\mathrm{tot}} - \frac{k}{a^2} + \frac{\Lambda_{\mathrm{eff}}}{3}$"],
    ["Distances", r"$D_C(z)=\int_0^z \frac{c\,dz'}{H(z')},\quad D_L=(1+z)D_C,\quad D_A=\frac{D_C}{1+z}$"],
    ["Growth equation", r"$\delta'' + \left(2 + \frac{H'}{H}\right)\delta' - \frac{3}{2}\,\Omega_m(a)\,\delta = 0,\quad (\,{}' \equiv d/d\ln a\,)$"]
]

# Wrap long section names for neatness
wrapped = [[ "\n".join(wrap(sec, 30)), form] for sec, form in formulae]

fig, ax = plt.subplots(figsize=(14, len(wrapped)*0.6 + 2))
ax.axis('off')

table = ax.table(cellText=wrapped,
                 colLabels=["Section", "Formula"],
                 cellLoc='left',
                 loc='center')

table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 1.4)

plt.title("Full Formula List — Pushing‑Medium Model", fontsize=16, weight='bold', pad=20)
plt.tight_layout()
plt.savefig("full_formula_list.png", dpi=300, bbox_inches='tight')
plt.close()


from fpdf import FPDF

pdf = FPDF()
pdf.add_page()

# Register fonts
pdf.add_font('NotoSans', '', '/usr/share/fonts/truetype/noto/NotoSansMono-Regular.ttf', uni=True)
pdf.add_font('NotoSans', 'B', '/usr/share/fonts/truetype/noto/NotoSansMono-Bold.ttf', uni=True)
pdf.set_font('NotoSans', 'B', 16)
pdf.cell(0, 10, 'Pushing-Medium Gravity Model: Formula Sheet & Comparison', ln=True, align='C')

pdf.set_font('NotoSans', 'B', 12)
pdf.cell(0, 10, 'Core Governing Equations (Pushing-Medium Gravity Model)', ln=True)
pdf.set_font('NotoSans', '', 11)
formulas = [
    "1. Gravitational Force: F = p * A",
    "2. Pressure Gradient Force: F = -∇P",
    "3. Medium Density Relation: ρ_g = P / c^2",
    "4. Acceleration due to Gravity: g = -∇(P / ρ)",
    "5. Potential Energy: U = -∫(∇P / ρ) · dr",
    "6. Field Equation Analogy: ∇²P = 4πGρ_m",
    "7. Energy-Momentum Transfer: T^μν_medium = P * u^μ u^ν"
]
for formula in formulas:
    pdf.multi_cell(0, 8, formula)

pdf.ln(5)
pdf.set_font('NotoSans', 'B', 12)
pdf.cell(0, 10, 'Comparison with General Relativity', ln=True)
pdf.set_font('NotoSans', '', 11)

# Table header
pdf.set_fill_color(200, 220, 255)
pdf.cell(45, 8, 'Quantity', border=1, fill=True)
pdf.cell(55, 8, 'GR Expression', border=1, fill=True)
pdf.cell(55, 8, 'Pushing-Medium', border=1, fill=True)
pdf.cell(35, 8, 'Notes', border=1, ln=True, fill=True)

# Table rows
comparison_data = [
    ('Gravitational Force', 'F = Gm1m2/r^2', 'F = p * A', 'Different mechanism'),
    ('Field Equation', 'G_{μν} = 8πGT_{μν}', '∇²P = 4πGρ_m', 'Analogous structure'),
    ('Acceleration', 'g = -∇Φ', 'g = -∇(P / ρ)', 'Similar form'),
    ('Energy Density', 'ρ = T^{00}', 'ρ_g = P / c^2', 'Different interpretation'),
    ('Potential Energy', 'U = -Gm1m2/r', 'U = -∫(∇P / ρ) · dr', 'Integral vs point-based'),
    ('Stress-Energy Tensor', 'T^{μν}', 'T^μν_medium = P * u^μ u^ν', 'Medium-based formulation')
]

for row in comparison_data:
    for item in row:
        pdf.cell(45 if item == row[0] else 55 if item == row[1] or item == row[2] else 35, 8, item, border=1)
    pdf.ln()

pdf.output('./pushing_medium_vs_gravity_model.pdf')


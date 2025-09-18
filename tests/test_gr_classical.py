from general_relativity.classical import binary_quadrupole_power


def test_quadrupole_power_positive():
    M1, M2, a = 1.4*1.989e30, 1.3*1.989e30, 1e9
    assert binary_quadrupole_power(M1, M2, a) > 0

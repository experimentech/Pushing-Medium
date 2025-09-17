from pushing_medium import (
    index_point_masses,
    flow_rotational,
    ray_advection,
    massive_accel_medium,
    newtonian_accel_sum,
)


def test_index_point_masses_increases_with_mass():
    # Use a custom mu_scale to avoid cancellation at double precision near 1.0
    mu_scale = lambda M: 1.0 * M  # dimensionless test scaling
    n1 = index_point_masses((1,0,0), [(1.0, (0,0,0))], mu_scale=mu_scale)
    n2 = index_point_masses((1,0,0), [(2.0, (0,0,0))], mu_scale=mu_scale)
    assert n2 > n1


def test_flow_rotational_zero_far_from_zero_spin():
    u = flow_rotational((1,2,3), [])
    assert u == (0.0, 0.0, 0.0)


def test_ray_advection_dimensions():
    v = ray_advection((0,0,0), (1,0,0), n_total=1.0, u_g=(0,0,0))
    assert len(v) == 3 and v[0] > 0


def test_massive_accel_medium_direction():
    a = massive_accel_medium((1e-10, 0, 0))
    assert a[0] < 0


def test_newtonian_accel_sum_matches_sign():
    a = newtonian_accel_sum((1,0,0), [(1.0, (0,0,0))])
    assert a[0] < 0

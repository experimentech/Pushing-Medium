from pushing_medium import (
    index_point_masses,
    flow_rotational,
    ray_advection,
    massive_accel_medium,
    newtonian_accel_sum,
)


def test_index_point_masses_scaling():
    n1 = index_point_masses((1,0,0), [(1.0,(0,0,0))])
    n2 = index_point_masses((1,0,0), [(2.0,(0,0,0))])
    assert n2 > n1


def test_flow_rotational_zero_no_spins():
    assert flow_rotational((0,0,0), []) == (0.0,0.0,0.0)


def test_ray_advection_positive_x():
    v = ray_advection((0,0,0), (1,0,0), n_total=1.0, u_g=(0,0,0))
    assert v[0] > 0 and len(v) == 3


def test_massive_accel_medium_direction():
    a = massive_accel_medium((1e-10,0,0))
    assert a[0] < 0


def test_newtonian_accel_sum_sign():
    a = newtonian_accel_sum((1,0,0), [(1.0,(0,0,0))])
    assert a[0] < 0

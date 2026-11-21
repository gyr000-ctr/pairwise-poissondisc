import math

import pytest

import poissondisc

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

def _p(r=0.5, seed=1):
    return poissondisc.PoissonDisc(r, seed)


def _pts():
    return _p().sample(10.0, 10.0, k=10)


def _check_nonempty():
    return len(_pts()) > 0


def _check_min_distance():
    pts = _pts()
    r = 0.5
    return all(math.dist(a, b) >= r - 1e-9 for i, a in enumerate(pts) for b in pts[i + 1:])


def _check_in_bounds():
    return all(0.0 <= x <= 10.0 and 0.0 <= y <= 10.0 for x, y in _pts())


def _check_deterministic():
    return _pts() == _pts()


def _check_different_seed():
    return _p(0.5, 1).sample(10.0, 10.0) != _p(0.5, 2).sample(10.0, 10.0)


def _check_density():
    pts = _p(0.5, 3).sample(10.0, 10.0)
    return len(pts) >= 20


def _check_bad_radius():
    return _raises(lambda: poissondisc.PoissonDisc(0.0, 1))


def _check_bad_area():
    return _raises(lambda: _p().sample(0.0, 10.0))


def _check_k_small():
    return len(_p().sample(5.0, 5.0, k=1)) > 0


def _check_type():
    return isinstance(_pts(), list)


def _check_points_pairs():
    return all(len(pt) == 2 for pt in _pts())


def _check_no_mutation():
    p = _p()
    a = p.sample(5.0, 5.0)
    b = p.sample(5.0, 5.0)
    return a == b


def _check_small_area():
    return isinstance(_p(0.1).sample(1.0, 1.0), list)


def _check_float():
    return isinstance(_pts()[0][0], float)


def _check_count_positive():
    return len(_p(0.2, 5).sample(3.0, 3.0)) >= 1

CASES = [("nonempty", "_check_nonempty"), ("min_distance", "_check_min_distance"), ("in_bounds", "_check_in_bounds"), ("deterministic", "_check_deterministic"), ("different_seed", "_check_different_seed"), ("density", "_check_density"), ("bad_radius", "_check_bad_radius"), ("bad_area", "_check_bad_area"), ("k_small", "_check_k_small"), ("type", "_check_type"), ("points_pairs", "_check_points_pairs"), ("no_mutation", "_check_no_mutation"), ("small_area", "_check_small_area"), ("float", "_check_float"), ("count_positive", "_check_count_positive")]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert globals()[check]() is True

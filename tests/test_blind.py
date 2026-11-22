import numpy as np

from mismatch import blind as BL
from mismatch import burgers as B
from mismatch.alignment import term_responses
from mismatch.simulators import simulate_batch, sobol_thetas

LOW, SPAN = np.array([0.05, 0.5, 0.1, -1.0]), np.array([0.45, 1.5, 2.4, 2.0])


def burg_thetas(n, seed):
    return B.from_unit((sobol_thetas(n, seed=seed) - LOW) / SPAN)


def test_library_sizes_follow_the_rule():
    assert len(BL.PEND_LIB) == 13            # 9 monomials of degree 1-3 + 4 Fourier terms
    assert len(BL.BURG_LIB) == 12            # {1, u, u^2, u^3} x {1, u_x, u_xx}
    assert "1" in BL.BURG_LIB and "u^3*u_xx" in BL.BURG_LIB


def test_pools_have_ten_terms_each():
    assert len(BL.PEND_POOL) == 10 and len(BL.BURG_POOL) == 10


def test_draw_is_deterministic_and_without_replacement():
    a, b = BL.draw("0123456789abcdef"), BL.draw("0123456789abcdef")
    assert a == b
    assert len(set(a["pendulum"])) == 3 and len(set(a["burgers"])) == 3
    assert sorted(a["pendulum_order"]) == sorted(BL.PEND_POOL)
    assert BL.draw("fedcba9876543210") != a


def test_eps_zero_is_nominal():
    th = sobol_thetas(4, seed=7)
    assert np.array_equal(BL.pend_batch(th, "constant torque", 0.0), simulate_batch(th, "S1", 0.0))
    bt = burg_thetas(4, 7)
    assert np.array_equal(BL.burg_batch(bt, "drift", 0.0), B.simulate_batch(bt))


def test_every_pool_term_gives_a_finite_response():
    th = sobol_thetas(4, seed=11)
    for t in BL.PEND_POOL:
        r = BL.pend_response(th, t)
        assert np.all(np.isfinite(r)) and np.linalg.norm(r) > 0, t
    bt = burg_thetas(4, 11)
    for t in BL.BURG_POOL:
        r = BL.burg_response(bt, t)
        assert np.all(np.isfinite(r)) and np.linalg.norm(r) > 0, t


def test_libraries_give_responses():
    th = sobol_thetas(4, seed=3)
    names, resp, _ = term_responses(th, terms=BL.PEND_LIB)
    assert len(names) == 13 and np.all(np.isfinite(resp))
    names, resp, _ = B.term_responses(burg_thetas(4, 3), terms=BL.BURG_LIB)
    assert len(names) == 12 and np.all(np.isfinite(resp))

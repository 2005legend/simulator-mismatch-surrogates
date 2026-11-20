import numpy as np

from mismatch import burgers as B
from mismatch.surrogate import scale_inputs

TH = np.array([[0.10, 1.5, 1.0, 0.0], [0.2, 1.0, 0.5, 1.0], [0.3, 0.5, 0.0, 2.0]])


def test_initial_condition_and_shape():
    u = B.solve(TH)
    assert u.shape == (3, B.N_OT, B.N_OX)
    assert np.allclose(u[:, 0], B.initial(TH, B.X_OUT))


def test_grid_converged_at_hardest_corner():
    a, b = B.solve(TH[:1]), B.solve(TH[:1], n_x=512, dt=0.001)
    assert np.abs(a - b).max() < 1e-5


def test_batch_equals_single_and_eps_zero_is_nominal():
    full = B.simulate_batch(TH)
    assert np.allclose(full[1], B.simulate_batch(TH[1:2])[0])
    assert np.array_equal(B.simulate_batch(TH, "P1", 0.0), full)


def test_missing_terms_scale_linearly_for_small_eps():
    f0 = B.simulate_batch(TH)
    for s in ("P1", "P2"):
        d1 = B.simulate_batch(TH, s, 1e-3) - f0
        d2 = B.simulate_batch(TH, s, 2e-3) - f0
        assert np.allclose(d2, 2 * d1, rtol=0, atol=0.02 * np.abs(d1).max())   # second-order effect about 0.6% at eps = 1e-3


def test_viscous_decay_and_residual_floor():
    f0 = B.simulate_batch(TH)
    energy = np.sum(f0.reshape(3, B.N_OT, B.N_OX) ** 2, axis=2)
    assert np.all(np.diff(energy, axis=1) <= 1e-9)            # Burgers energy never grows
    noisy = f0 + 0.05 * np.random.default_rng(0).normal(size=f0.shape)
    assert np.all(B.residual_score(noisy, TH) > 3 * B.residual_score(f0, TH))


def test_box_maps_to_unit_scaling():
    x = scale_inputs(B.to_unit_box(np.stack([B.BOX_LOW, B.BOX_HIGH])))
    assert np.allclose(x, [[-1] * 4, [1] * 4])


def test_term_responses_match_true_term_direction():
    names, resp, base = B.term_responses(TH)
    assert resp.shape == (len(B.TERMS), 3, B.N_OUT)
    d = (B.simulate_batch(TH, "P2", 1e-3) - base) / 1e-3
    assert np.allclose(resp[names.index("sin(3x)")], d, atol=1e-3 * np.abs(d).max())
    # P1 is h = -u|u|, so its response is minus the u|u| response
    d1 = (B.simulate_batch(TH, "P1", 1e-3) - base) / 1e-3
    assert np.allclose(-resp[names.index("u|u|")], d1, atol=1e-2 * np.abs(d1).max())

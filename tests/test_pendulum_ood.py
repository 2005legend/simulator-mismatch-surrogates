import numpy as np

from mismatch import burgers_ood as BO
from mismatch import pendulum_ood as PO
from mismatch.alignment import term_responses
from mismatch.simulators import simulate_batch, sobol_thetas

TH = sobol_thetas(8, seed=12345)


def test_eps_zero_is_nominal():
    for t in ("Q1", "Q2"):
        assert np.array_equal(PO.simulate_batch(TH, t, 0.0), simulate_batch(TH, "S1", 0.0))


def test_terms_move_the_trajectory_linearly():
    for t in ("Q1", "Q2"):
        r1, r2 = PO.response(TH, t, 1e-3), PO.response(TH, t, 1e-4)
        assert np.linalg.norm(r1) > 0
        assert np.linalg.norm(r1 - r2) / np.linalg.norm(r2) < 0.01


def test_constant_torque_shifts_equilibrium_positive():
    # a positive constant torque pushes phi towards positive values at late times
    r = PO.response(TH, "Q1")
    assert np.mean(r[:, -50:]) > 0


def test_out_of_span_of_dictionary_term_is_small():
    names, resp, _ = term_responses(TH)
    share = BO.out_of_span(resp[names.index("w")], resp)
    assert np.max(share) < 1e-6


def test_phase_guard():
    import pytest
    from scripts.run_pendulum_ood import check_phase
    check_phase([999], "dev")
    with pytest.raises(SystemExit):
        check_phase([1], "dev")
    with pytest.raises(SystemExit):
        check_phase([0], "pilot", tag_ok=False)
    with pytest.raises(SystemExit):
        check_phase([0, 1], "pilot", tag_ok=True)
    with pytest.raises(SystemExit):
        check_phase([21], "main", tag_ok=True)
    check_phase([1, 20], "main", tag_ok=True)

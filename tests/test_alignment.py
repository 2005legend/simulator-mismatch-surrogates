import numpy as np
import pytest
import torch

from mismatch.alignment import TERMS, alignment_range, response_norm, term_responses
from mismatch.dynamics import fit_term, simulate_with_term
from mismatch.metrics import rel_errors
from mismatch.simulators import simulate_batch, sobol_thetas

torch.set_num_threads(1)
SEED = 999  # development seed; never 0, 101, 102 or 1-10


@pytest.fixture(scope="module")
def setup():
    thetas = sobol_thetas(64, seed=SEED)
    names, responses, base = term_responses(thetas)
    return thetas, names, responses, base


def test_rk4_rollout_matches_reference_solver(setup):
    thetas, _, _, base = setup
    assert np.median(rel_errors(base, simulate_batch(thetas, "S1", 0.0))) < 1e-3


def test_linear_response_predicts_small_discrepancy(setup):
    thetas, names, responses, base = setup
    eps = 0.002
    f_nom, f_true = simulate_batch(thetas, "S1", 0.0), simulate_batch(thetas, "S1", eps)
    predicted = f_nom - eps * responses[names.index("w|w|")]      # S1 adds -eps * w|w|
    assert np.median(rel_errors(predicted, f_true)) < 0.1 * np.median(rel_errors(f_nom, f_true))


def test_right_family_aligns_and_wrong_family_does_not(setup):
    thetas, names, responses, base = setup
    f_nom = simulate_batch(thetas, "S1", 0.0)
    norm = response_norm(responses, base)
    for system, eps, right, wrong in (("S1", 0.01, "w|w|", "tanh(w/0.1)"),
                                      ("S2", 0.0026, "tanh(w/0.1)", "w|w|")):
        d = rel_errors(f_nom, simulate_batch(thetas, system, eps))
        res = alignment_range(d, names, responses, base, n_mix=0)["per_term"]
        assert res[right] > 0.9
        assert res[wrong] < 0.3
    assert norm.shape == (len(TERMS), len(thetas))


def test_alignment_range_bounds(setup):
    _, names, responses, base = setup
    signal = response_norm(responses, base)[names.index("p^3")]
    res = alignment_range(signal, names, responses, base, n_mix=200, seed=1)
    assert res["per_term"]["p^3"] == pytest.approx(1.0)
    assert res["max"] == pytest.approx(1.0)
    assert res["min"] <= res["term_min"] <= res["term_max"] <= res["max"]
    assert -1.0 <= res["min"] < 0.5
    again = alignment_range(signal, names, responses, base, n_mix=200, seed=1)
    assert again["min"] == res["min"] and again["mix_quantiles"] == res["mix_quantiles"]


def test_single_term_dictionary_has_no_mixtures(setup):
    thetas, _, _, _ = setup
    names, responses, base = term_responses(thetas, {"w": TERMS["w"]})
    res = alignment_range(np.arange(len(thetas)), names, responses, base)
    assert res["mix_quantiles"] is None and res["min"] == res["max"]


def test_fit_term_reduces_discrepancy():
    thetas = sobol_thetas(8, seed=SEED * 1000)
    test = sobol_thetas(40, seed=SEED + 1)
    eps = 0.04
    rng = np.random.default_rng(SEED)
    clean = simulate_batch(thetas, "S1", eps)
    y = clean + 0.01 * clean.std(axis=1, keepdims=True) * rng.normal(size=clean.shape)
    term = fit_term(thetas, y, seed=SEED, lbfgs_iters=10)
    f_true = simulate_batch(test, "S1", eps)
    before = np.median(rel_errors(simulate_batch(test, "S1", 0.0), f_true))
    after = np.median(rel_errors(simulate_with_term(test, term), f_true))
    assert after < 0.5 * before

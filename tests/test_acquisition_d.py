import numpy as np
import pytest
import torch

from mismatch import config
from mismatch.acquisition import Problem, make_oracle
from mismatch.acquisition_d import (ARMS_D, observed_states, run_cell_d, state_coverage_scores,
                                    term_disagreement_scores)
from mismatch.dynamics import Term
from mismatch.simulators import simulate_batch, sobol_thetas
from mismatch.surrogate import Ensemble, FitConfig

torch.set_num_threads(1)
SEED = 999  # development seed; never 0, 101, 102 or 1-10


def test_identical_terms_have_zero_disagreement():
    pool = sobol_thetas(16, seed=SEED)
    t = Term()
    assert np.allclose(term_disagreement_scores([t, t, t], pool), 0.0)


def test_different_terms_disagree_most_where_they_differ():
    pool = sobol_thetas(32, seed=SEED)
    a, b = Term(), Term()
    with torch.no_grad():
        b.net[-1].bias.fill_(0.5)                # a constant extra acceleration
    scores = term_disagreement_scores([a, b], pool)
    assert scores.shape == (32,) and np.all(scores > 0)


def test_state_coverage_is_zero_on_observed_and_grows_with_distance():
    thetas = sobol_thetas(6, seed=SEED)
    y = simulate_batch(thetas, "S1", 0.0)
    observed = observed_states(thetas, y)
    assert observed.shape == (6 * 40, 2)
    own = state_coverage_scores(thetas, observed)
    assert np.all(own < 0.05)                          # spline omega is close to the exact one
    far = thetas.copy()
    far[:, 2] = 2.4                                    # all released from near the top
    far[:, 3] = 0.9
    near = thetas.copy()
    near[:, 2], near[:, 3] = 0.15, 0.0                 # small swings only
    obs_small = observed_states(near, simulate_batch(near, "S1", 0.0))
    assert state_coverage_scores(far, obs_small).mean() > state_coverage_scores(near, obs_small).mean()


@pytest.fixture(scope="module")
def small_problem():
    train = sobol_thetas(64, seed=SEED)
    nominal = Ensemble(n_members=2, cfg=FitConfig(hidden=32, max_epochs=100, patience=40),
                       base_seed=SEED).fit(train, simulate_batch(train, "S1", 0.0))
    test = sobol_thetas(30, seed=SEED + 1)
    return Problem(nominal=nominal, oracle=make_oracle("S1", 0.04, seed=SEED), test_thetas=test,
                   test_true=simulate_batch(test, "S1", 0.04), seed=SEED, n_init=4, batch=4,
                   rounds=2, pool_size=48)


def test_all_arms_run_and_share_the_first_fit(small_problem):
    out = run_cell_d(small_problem, k_terms=2, lbfgs_iters=2, gm_steps=60)
    assert set(out) == set(ARMS_D)
    design = small_problem.design()
    first = {a: r["log"][0]["median"] for a, r in out.items()}
    assert len(set(first.values())) == 1, first          # same data, same seed, same fit
    for arm, res in out.items():
        assert [row["n"] for row in res["log"]] == [4, 8, 12]
        np.testing.assert_allclose(res["thetas"][:4], design[:4])
        assert len(np.unique(res["thetas"], axis=0)) == 12
        assert res["budget_auc"] > 0 and res["budget_auc_corrected"] > 0
        assert np.all(res["thetas"] >= config.BOX_LOW) and np.all(res["thetas"] <= config.BOX_HIGH)
    np.testing.assert_allclose(out["sobol"]["thetas"], design[:12])


def test_cell_is_reproducible(small_problem):
    a = run_cell_d(small_problem, arms=("state_coverage",), k_terms=2, lbfgs_iters=2, gm_steps=60)
    b = run_cell_d(small_problem, arms=("state_coverage",), k_terms=2, lbfgs_iters=2, gm_steps=60)
    np.testing.assert_allclose(a["state_coverage"]["thetas"], b["state_coverage"]["thetas"])
    assert a["state_coverage"]["budget_auc"] == pytest.approx(b["state_coverage"]["budget_auc"],
                                                              rel=1e-6)

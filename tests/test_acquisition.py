import numpy as np
import pytest
import torch

from mismatch import config
from mismatch.acquisition import (ARMS, Problem, make_oracle, r_hat_lower, run_cell,
                                  sample_batch, update_gate)
from mismatch.gp import PCAGP, ScalarGP
from mismatch.metrics import budget_auc, rel_errors
from mismatch.signals import residual_score, spline_residual_score
from mismatch.simulators import simulate_batch, sobol_thetas
from mismatch.surrogate import Ensemble, FitConfig

torch.set_num_threads(1)
SEED = 999  # development seed; never 0, 101, 102 or 1-10


def test_sample_batch_no_repeats_and_uniform_when_k_zero():
    rng = np.random.default_rng(0)
    idx = sample_batch(np.arange(50.0), 10, 2.0, rng)
    assert len(set(idx)) == 10
    counts = np.bincount(np.concatenate(
        [sample_batch(np.arange(20.0), 5, 0, rng) for _ in range(400)]), minlength=20)
    assert counts.min() > 50          # score 0 still picked when k = 0


def test_sample_batch_concentrates_with_k_and_greedy():
    rng = np.random.default_rng(1)
    scores = np.linspace(0.0, 1.0, 200)
    mean_k1 = np.mean([scores[sample_batch(scores, 8, 1, rng)].mean() for _ in range(200)])
    mean_k4 = np.mean([scores[sample_batch(scores, 8, 4, rng)].mean() for _ in range(200)])
    assert mean_k4 > mean_k1 > 0.5
    assert set(sample_batch(scores, 8, np.inf, rng)) == set(range(192, 200))


def test_sample_batch_all_zero_scores_falls_back_to_uniform():
    idx = sample_batch(np.zeros(30), 8, 2.0, np.random.default_rng(2))
    assert len(set(idx)) == 8


def test_budget_auc_is_geometric_mean_error():
    assert budget_auc([8, 16, 24], [0.1, 0.1, 0.1]) == pytest.approx(0.1)
    # log error linear in n: trapezoid is exact, mean log = log of the middle value
    assert budget_auc([8, 32, 56], [1e-1, 1e-2, 1e-3]) == pytest.approx(1e-2)
    assert budget_auc([8, 56], [0.2, 0.2]) < budget_auc([8, 56], [0.4, 0.4])


def test_ode_residual_small_on_nominal_and_larger_on_true():
    th = sobol_thetas(8, seed=SEED)
    nominal = simulate_batch(th, "S1", 0.0)
    true = simulate_batch(th, "S1", 0.2)
    # scale: the restoring term (g/L) sin(phi)
    scale = np.mean(np.abs(config.G / th[:, [1]] * np.sin(nominal)), axis=1)
    assert np.all(residual_score(nominal, th) < 0.05 * scale)
    assert np.all(residual_score(true, th) > residual_score(nominal, th))


def test_spline_residual_tracks_missing_term_under_noise():
    th = sobol_thetas(6, seed=SEED)
    rng = np.random.default_rng(3)
    noisy = lambda y: y + 0.01 * y.std(axis=1, keepdims=True) * rng.normal(size=y.shape)
    clean = spline_residual_score(noisy(simulate_batch(th, "S1", 0.0)), th)
    drag = spline_residual_score(noisy(simulate_batch(th, "S1", 0.5)), th)
    assert np.median(drag) > np.median(clean)


def test_gate_is_monotone_and_needs_16_points_for_arm_7():
    assert update_gate(0, 0.5, 8) == 0
    assert update_gate(0, 1.5, 8) == 1
    assert update_gate(0, 3.0, 8) == 1      # above 2 but fewer than 16 points
    assert update_gate(1, 3.0, 16) == 2
    assert update_gate(2, 0.1, 24) == 2     # never switches back
    assert update_gate(1, 0.1, 24) == 1


def test_r_hat_recovers_ratio_and_debiases_noise():
    rng = np.random.default_rng(4)
    n, n_t = 40, 201
    f_nom = rng.normal(size=(n, n_t))
    sigma = np.full(n, 0.05)
    d = 0.2 * rng.normal(size=(n, n_t))
    y = f_nom - d + sigma[:, None] * rng.normal(size=(n, n_t))
    true_rel = np.median(np.linalg.norm(d, axis=1) / np.linalg.norm(y, axis=1))
    r_hat, lower = r_hat_lower(y, f_nom, sigma, 0.1, rng)
    assert r_hat == pytest.approx(true_rel / 0.1, rel=0.05)
    assert lower <= r_hat
    # pure noise, no discrepancy: debiased estimate is near zero
    y0 = f_nom + sigma[:, None] * rng.normal(size=(n, n_t))
    assert r_hat_lower(y0, f_nom, sigma, 0.1, rng)[0] < 0.1


def test_pca_gp_interpolates_and_variance_grows_away_from_data():
    rng = np.random.default_rng(5)
    th = config.BOX_LOW + rng.random((30, 4)) * (config.BOX_HIGH - config.BOX_LOW)
    t = np.linspace(0, 1, 40)
    y = np.sin(3 * t)[None, :] * th[:, [2]] + np.cos(5 * t)[None, :] * th[:, [0]]
    gp = PCAGP().fit(th, y)
    mean, var_train = gp.predict(th)
    assert rel_errors(mean, y).max() < 0.05
    far = config.BOX_LOW + rng.random((30, 4)) * (config.BOX_HIGH - config.BOX_LOW)
    assert gp.predict(far)[1].mean() > var_train.mean()


def test_scalar_gp_fits_smooth_function():
    rng = np.random.default_rng(6)
    th = config.BOX_LOW + rng.random((40, 4)) * (config.BOX_HIGH - config.BOX_LOW)
    y = th[:, 2] ** 2
    assert np.abs(ScalarGP().fit(th, y).predict(th) - y).max() < 0.1


def test_oracle_noise_depends_only_on_point():
    oracle = make_oracle("S1", 0.05, seed=SEED)
    th = sobol_thetas(4, seed=SEED)
    y_all, s_all = oracle(th)
    y_rev, _ = oracle(th[::-1])
    np.testing.assert_allclose(y_all, y_rev[::-1])
    clean = simulate_batch(th, "S1", 0.05)
    np.testing.assert_allclose(s_all, config.NOISE_MAIN * clean.std(axis=1))
    assert not np.allclose(y_all, clean)


@pytest.fixture(scope="module")
def small_problem():
    train = sobol_thetas(96, seed=SEED)
    nominal = Ensemble(n_members=2, cfg=FitConfig(hidden=32, max_epochs=150, patience=50),
                       base_seed=SEED).fit(train, simulate_batch(train, "S1", 0.0))
    test = sobol_thetas(40, seed=SEED + 1)
    return Problem(nominal=nominal, oracle=make_oracle("S1", 0.05, seed=SEED),
                   test_thetas=test, test_true=simulate_batch(test, "S1", 0.05), seed=SEED,
                   rounds=2, pool_size=64, disc_members=2,
                   disc_cfg=FitConfig(hidden=16, max_epochs=60, min_val=10 ** 9))


def test_all_arms_run_with_shared_initial_points(small_problem):
    out = run_cell(small_problem)
    assert set(out) == set(ARMS)
    design = small_problem.design()
    for res in out.values():
        assert [row["n"] for row in res["log"]] == [8, 16, 24]
        np.testing.assert_allclose(res["thetas"][:8], design[:8])
        assert len(np.unique(res["thetas"], axis=0)) == 24
        assert res["budget_auc"] > 0
        assert all(row["median"] <= row["p95"] <= row["p99"] for row in res["log"])
    # same initial points and same first correction fit rule -> arm 2 follows the design
    np.testing.assert_allclose(out[2]["thetas"], design)
    assert len(out[8]["gate_log"]) == 2
    assert all(np.all(res["thetas"] >= config.BOX_LOW) for res in out.values())


def test_cell_is_reproducible(small_problem):
    a = run_cell(small_problem, arms=(3, 7))
    b = run_cell(small_problem, arms=(7, 3))
    for arm in (3, 7):
        np.testing.assert_allclose(a[arm]["thetas"], b[arm]["thetas"])
        assert a[arm]["budget_auc"] == pytest.approx(b[arm]["budget_auc"], rel=1e-4)


def test_horizon_option_scores_only_first_points(small_problem):
    from dataclasses import replace
    short = replace(small_problem, n_t=81, _pool_err={})
    out = run_cell(short, arms=(2, 4, 7, 8, 9))
    full = run_cell(small_problem, arms=(2,))
    assert all(len(res["log"]) == 3 for res in out.values())
    # same Sobol points, different scored horizon -> different error
    np.testing.assert_allclose(out[2]["thetas"], full[2]["thetas"])
    assert out[2]["uncorrected"]["median"] != full[2]["uncorrected"]["median"]


def test_arms_share_fit_noise_on_identical_data(small_problem):
    out = run_cell(small_problem)
    first = {arm: res["log"][0]["median"] for arm, res in out.items()}
    assert len(set(first.values())) == 1, first      # same 8 points, same seed, same error


def test_stream_seeds_are_independent():
    from mismatch.simulators import stream_seed
    seeds = [stream_seed(s, p, i) for s in (0, 1, 2, 101) for p in
             ("train", "test", "design", "pool") for i in (0, 1, 2)]
    assert len(set(seeds)) == len(seeds)
    a = sobol_thetas(56, seed=stream_seed(0, "design"))
    b = sobol_thetas(512, seed=stream_seed(0, "train"))[:56]
    assert not np.allclose(a, b)
    assert not np.allclose(sobol_thetas(500, seed=stream_seed(3, "test")),
                           sobol_thetas(512, seed=stream_seed(4, "train"))[:500])

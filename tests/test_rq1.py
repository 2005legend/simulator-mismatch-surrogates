import numpy as np
import pytest
import torch

from mismatch.acquisition import make_oracle
from mismatch.rq1 import (auroc, ause, conformal_quantile, rq1_cell, sigma_rel, spearman)
from mismatch.simulators import simulate_batch, sobol_thetas
from mismatch.surrogate import Ensemble, FitConfig

torch.set_num_threads(1)
SEED = 999  # development seed; never 0, 101, 102 or 1-10
FAST = FitConfig(hidden=32, max_epochs=150, patience=50)


def test_auroc_perfect_and_chance():
    err = np.arange(100.0)
    assert auroc(err, err) == 1.0
    assert auroc(-err, err) == 0.0
    rng = np.random.default_rng(0)
    assert 0.4 < auroc(rng.normal(size=2000), rng.normal(size=2000)) < 0.6


def test_spearman_monotone():
    x = np.linspace(0, 1, 50)
    assert spearman(x, x ** 3) == pytest.approx(1.0)
    assert spearman(x, -x) == pytest.approx(-1.0)


def test_ause_is_zero_for_oracle_and_positive_for_reverse():
    err = np.random.default_rng(1).exponential(size=500)
    assert ause(err, err) == pytest.approx(0.0, abs=1e-12)
    assert ause(-err, err) > 0.1
    assert ause(np.random.default_rng(2).normal(size=500), err) > 0.0


def test_conformal_quantile_gives_nominal_coverage():
    rng = np.random.default_rng(3)
    covered = []
    for _ in range(300):
        cal, test = rng.exponential(size=200), rng.exponential(size=2000)
        covered.append(np.mean(test <= conformal_quantile(cal, 0.9)))
    assert 0.88 < np.mean(covered) < 0.93


def test_sigma_rel_scales_with_sigma():
    f = np.ones((4, 10))
    assert np.allclose(sigma_rel(2 * np.ones((4, 10)), f), 2 * sigma_rel(np.ones((4, 10)), f))


@pytest.fixture(scope="module")
def cell_records():
    train = sobol_thetas(128, seed=SEED)
    y = simulate_batch(train, "S1", 0.0)
    nominal = Ensemble(n_members=2, cfg=FAST, base_seed=SEED).fit(train, y)
    mc = Ensemble(n_members=1, cfg=FitConfig(hidden=32, max_epochs=150, patience=50,
                                             dropout=0.1), base_seed=SEED + 1).fit(train, y)
    test = sobol_thetas(60, seed=SEED + 7)
    f_true = simulate_batch(test, "S1", 0.02)
    oracle = make_oracle("S1", 0.02, seed=SEED)
    full = rq1_cell(nominal, mc, test, f_true, oracle, SEED, n_mc=10, disc_members=2,
                    n_cal_true=(8, 16))
    sim_only = rq1_cell(nominal, mc, test, f_true, oracle, SEED, include_true_data=False, n_mc=10)
    return full, sim_only


def test_cell_has_every_signal(cell_records):
    full, sim_only = cell_records
    names = {(r["signal"], r["n_cal"]) for r in full}
    for expected in [("ens_sigma", 0), ("mc_sigma", 0), ("ode_resid", 0), ("conf_nom", 0),
                     ("conf_true", 8), ("conf_true", 16), ("disc_sigma", 8), ("gp_std", 16)]:
        assert expected in names
    assert {r["signal"] for r in sim_only} == {"ens_sigma", "mc_sigma", "ode_resid",
                                               "ens_sigma_gauss", "mc_sigma_gauss", "conf_nom"}


def test_cell_values_are_valid(cell_records):
    full, _ = cell_records
    for r in full:
        if "auroc_e" in r:
            assert 0.0 <= r["auroc_e"] <= 1.0 and -1.0 <= r["spearman_e"] <= 1.0
            assert r["r"] > 0
        else:
            assert 0.0 <= r["coverage"] <= 1.0 and r["width"] > 0


def test_mc_dropout_passes_differ():
    train = sobol_thetas(64, seed=SEED)
    mc = Ensemble(n_members=1, cfg=FitConfig(hidden=32, max_epochs=50, patience=20, dropout=0.2),
                  base_seed=SEED).fit(train, simulate_batch(train, "S1", 0.0))
    p = mc.members[0].predict_mc(train[:5], n_passes=6, seed=1)
    assert p.shape == (6, 5, 201) and p.std(axis=0).mean() > 0
    q = mc.members[0].predict_mc(train[:5], n_passes=6, seed=1)
    np.testing.assert_allclose(p, q)


def test_runner_refuses_registered_seeds():
    from scripts.run_rq1 import check_seeds
    for bad in ([0], [101], [5], [999, 3]):
        with pytest.raises(SystemExit):
            check_seeds(bad)
    check_seeds([999, 998, 997])                  # development seeds are fine
    check_seeds([0, 1], allow_registered=True)    # explicit opt-in

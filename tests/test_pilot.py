import numpy as np
import pytest

from mismatch import config
from mismatch.pilot import (HYPER_GRID, choose_hyperparameters, choose_k, gcv_lambda, log_bisect,
                            ols_slope, paired_t_power, pca_components, seed_count_rule)
from mismatch.surrogate import FitConfig


def test_log_bisect_finds_the_target_and_checks_the_bracket():
    assert log_bisect(lambda x: x ** 2, 1.0, 0.01, 100.0) == pytest.approx(1.0, rel=2e-3)
    assert log_bisect(lambda x: 3 * x, 0.3, 1e-4, 1.0) == pytest.approx(0.1, rel=2e-3)
    with pytest.raises(ValueError):
        log_bisect(lambda x: x, 5.0, 0.01, 1.0)


def test_power_matches_standard_tables_and_behaves_monotonically():
    assert paired_t_power(1.0, 1.0, 10, 0.05) == pytest.approx(0.80, abs=0.02)   # d = 1, n = 10
    assert paired_t_power(0.0, 1.0, 10, 0.05) == pytest.approx(0.05, abs=0.005)  # no effect: level
    assert paired_t_power(1.0, 1.0, 10, 0.05 / 6) < paired_t_power(1.0, 1.0, 10, 0.05)
    assert paired_t_power(1.0, 1.0, 20, 0.05) > paired_t_power(1.0, 1.0, 10, 0.05)
    assert paired_t_power(-1.0, 1.0, 10, 0.05) == pytest.approx(paired_t_power(1.0, 1.0, 10, 0.05))


def test_seed_count_rule_branches():
    strong = seed_count_rule([1.0, 0.8, 1.2])                  # mean 1, sd 0.2: easy
    assert strong["n_seeds"] == 10 and strong["reached_target"]
    weak = seed_count_rule([0.3, 1.2, -0.6])                   # mean 0.3, sd ~0.9: hopeless
    assert weak["n_seeds"] == 20 and not weak["reached_target"]
    middle = seed_count_rule([0.5, 1.0, 0.0, 0.8, 0.2])        # needs more than 10, fewer than 20
    assert 10 < middle["n_seeds"] <= 20 or middle["n_seeds"] == 20
    assert middle["power_at_n_min"] < 0.8
    assert seed_count_rule([0.4, 0.4, 0.4])["n_seeds"] == 10   # no spread
    with pytest.raises(ValueError):
        seed_count_rule([1.0])


def test_pca_components_counts_rank_and_respects_the_cap():
    rng = np.random.default_rng(0)
    low = rng.normal(size=(30, 3)) @ rng.normal(size=(3, 50))
    assert pca_components(low + 1e-6 * rng.normal(size=low.shape)) == 3
    high = rng.normal(size=(40, 20)) @ rng.normal(size=(20, 60))
    assert pca_components(high, max_components=12) == 12
    assert pca_components(rng.normal(size=(4, 50)), max_components=12) <= 3     # at most n - 1


def test_choose_k_and_hyperparameters_take_the_lowest_mean():
    best, means = choose_k({1: [0.05, 0.06], 2: [0.04, 0.05], 4: [0.07, 0.09]})
    assert best == 2 and means[2] == pytest.approx(0.045)
    label, _ = choose_hyperparameters({"default": [0.02, 0.03], "hidden128": [0.01, 0.02]})
    assert label == "hidden128"


def test_grid_entries_are_valid_network_settings():
    for overrides in HYPER_GRID.values():
        assert FitConfig(**overrides).hidden > 0


def test_ols_slope():
    x = np.linspace(0, 1, 20)
    assert ols_slope(x, 2 * x + 1) == pytest.approx(2.0)


def test_gcv_prefers_more_smoothing_for_noisier_data():
    t = config.T_GRID
    rng = np.random.default_rng(1)
    base = np.stack([np.sin(0.5 * t + p) * np.exp(-0.1 * t) for p in np.linspace(0, 3, 6)])
    clean = base + 1e-4 * rng.normal(size=base.shape)
    noisy = base + 5e-2 * rng.normal(size=base.shape)
    grid = np.logspace(-8, 0, 9)
    lam_clean, lam_noisy = gcv_lambda(clean, grid=grid), gcv_lambda(noisy, grid=grid)
    assert lam_noisy > lam_clean
    assert grid[0] <= lam_clean <= grid[-1]

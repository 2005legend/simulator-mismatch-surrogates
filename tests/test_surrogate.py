import numpy as np
import pytest
import torch

from mismatch import config
from mismatch.surrogate import Ensemble, FitConfig, scale_inputs

torch.set_num_threads(1)
FAST = FitConfig(hidden=32, max_epochs=300, patience=50)


def _toy_data(n, seed=0):
    rng = np.random.default_rng(seed)
    thetas = config.BOX_LOW + rng.random((n, 4)) * (config.BOX_HIGH - config.BOX_LOW)
    t = np.linspace(0, 1, 20)
    y = np.sin(2 * np.pi * t[None, :] * thetas[:, [1]]) * thetas[:, [2]]
    return thetas, y


def test_scale_inputs_maps_box_to_unit_cube():
    np.testing.assert_allclose(scale_inputs(config.BOX_LOW), -1.0)
    np.testing.assert_allclose(scale_inputs(config.BOX_HIGH), 1.0)


def test_default_size_is_about_30k_parameters():
    n = Ensemble(n_members=1).n_parameters
    assert 25_000 < n < 35_000


def test_shapes():
    thetas, y = _toy_data(64)
    ens = Ensemble(n_out=20, n_members=3, cfg=FAST).fit(thetas, y)
    mean, sigma = ens.predict(thetas[:5])
    assert mean.shape == sigma.shape == (5, 20)
    assert ens.member_predictions(thetas[:5]).shape == (3, 5, 20)
    assert np.all(sigma >= 0)


def test_learns_better_than_mean_baseline():
    thetas, y = _toy_data(256)
    test_t, test_y = _toy_data(100, seed=1)
    mean, _ = Ensemble(n_out=20, n_members=2, cfg=FAST).fit(thetas, y).predict(test_t)
    mse = np.mean((mean - test_y) ** 2)
    baseline = np.mean((y.mean(axis=0) - test_y) ** 2)
    assert mse < 0.5 * baseline


def test_same_seed_gives_same_predictions():
    thetas, y = _toy_data(48)
    a = Ensemble(n_out=20, n_members=1, cfg=FAST, base_seed=7).fit(thetas, y).predict(thetas)[0]
    b = Ensemble(n_out=20, n_members=1, cfg=FAST, base_seed=7).fit(thetas, y).predict(thetas)[0]
    np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)


def test_members_differ():
    thetas, y = _toy_data(48)
    ens = Ensemble(n_out=20, n_members=2, cfg=FAST).fit(thetas, y)
    p = ens.member_predictions(thetas)
    assert not np.allclose(p[0], p[1])


def test_tiny_dataset_trains_without_validation_split():
    thetas, y = _toy_data(6)
    ens = Ensemble(n_out=20, n_members=1, cfg=FAST).fit(thetas, y)
    assert ens.members[0].epochs_run == FAST.max_epochs


def test_batched_fit_matches_independent_fits():
    thetas, y = _toy_data(96)
    test_t, _ = _toy_data(30, seed=5)
    cfg = FitConfig(hidden=32, max_epochs=80, patience=40)
    a = Ensemble(n_out=20, n_members=3, cfg=cfg, base_seed=4, batched=False).fit(thetas, y)
    b = Ensemble(n_out=20, n_members=3, cfg=cfg, base_seed=4, batched=True).fit(thetas, y)
    pa, pb = a.member_predictions(test_t), b.member_predictions(test_t)
    assert np.median(np.abs(pa - pb)) / np.abs(pa).mean() < 1e-3
    assert [m.epochs_run for m in a.members] == [m.epochs_run for m in b.members]


def test_batched_fit_without_validation_split_and_with_dropout_fallback():
    thetas, y = _toy_data(6)
    ens = Ensemble(n_out=20, n_members=3, cfg=FAST, base_seed=2).fit(thetas, y)
    assert all(m.epochs_run == FAST.max_epochs for m in ens.members)
    drop = FitConfig(hidden=32, max_epochs=40, patience=20, dropout=0.1)
    thetas, y = _toy_data(48)
    mc = Ensemble(n_out=20, n_members=2, cfg=drop, base_seed=2).fit(thetas, y)   # sequential path
    assert mc.members[0].predict_mc(thetas[:4], n_passes=3).shape == (3, 4, 20)


def test_batched_members_are_distinct_and_reproducible():
    thetas, y = _toy_data(64)
    a = Ensemble(n_out=20, n_members=3, cfg=FAST, base_seed=9).fit(thetas, y)
    b = Ensemble(n_out=20, n_members=3, cfg=FAST, base_seed=9).fit(thetas, y)
    pa, pb = a.member_predictions(thetas), b.member_predictions(thetas)
    np.testing.assert_allclose(pa, pb, rtol=1e-5, atol=1e-6)
    assert not np.allclose(pa[0], pa[1])

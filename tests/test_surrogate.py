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

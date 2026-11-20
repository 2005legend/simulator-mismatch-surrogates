import numpy as np
import torch

from mismatch import burgers as B
from mismatch import operator as O


def test_profiles_scaled_and_simulate_shapes():
    c = O.sample_coefs(4, 0)
    y, u0 = O.simulate(c)
    assert y.shape == (4, B.N_OT - 1, O.N_OP) and u0.shape == (4, O.N_OP)
    assert np.allclose(np.abs(O.profiles(c, B.grid()[0])).max(axis=1), O.AMP)


def test_operator_ensemble_learns_and_members_differ():
    torch.set_num_threads(1)
    c = O.sample_coefs(48, 1)
    y, u0 = O.simulate(c)
    ens = O.OperatorEnsemble(n_members=2, base_seed=3, epochs=30, patience=30).fit(u0, y)
    mean, sd = ens.predict(u0)
    base = np.linalg.norm(y - y.mean(axis=0))
    assert np.linalg.norm(mean - y) < 0.6 * base
    assert np.all(sd >= 0) and sd.mean() > 0


def test_missing_term_changes_operator_output():
    c = O.sample_coefs(3, 2)
    y0, _ = O.simulate(c)
    y1, _ = O.simulate(c, "P2", 0.05)
    assert np.abs(y1 - y0).max() > 1e-3

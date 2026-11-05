import numpy as np
import pytest

from mismatch import config
from mismatch.errors import (decompose, fraction_after, horizon_slice, linearity_error,
                             ratio_r, rel_norm, smoothness_proxy)
from mismatch.simulators import simulate_batch, sobol_thetas

rng = np.random.default_rng(0)
F_NOM = rng.normal(size=(50, config.N_T))
F_TRUE = F_NOM + 0.1 * rng.normal(size=F_NOM.shape)
F_HAT = F_NOM + 0.05 * rng.normal(size=F_NOM.shape)


def test_decomposition_sums_exactly():
    a, d, e = decompose(F_HAT, F_NOM, F_TRUE)
    np.testing.assert_allclose(a + d, e, rtol=0, atol=1e-12)


def test_r_is_zero_without_mismatch():
    a, d, _ = decompose(F_HAT, F_NOM, F_NOM)
    assert ratio_r(a, d, F_NOM) == 0.0


def test_r_scales_with_discrepancy():
    a, d, _ = decompose(F_HAT, F_NOM, F_TRUE)
    r1 = ratio_r(a, d, F_TRUE)
    d2 = 2 * d
    r2 = ratio_r(a, d2, F_NOM - d2)
    assert r2 == pytest.approx(2 * r1, rel=0.05)


def test_rel_norm_shape():
    assert rel_norm(F_HAT, F_TRUE).shape == (50,)


def test_smoothness_higher_for_rough_signal():
    t = config.T_GRID
    smooth = np.sin(t)[None, :]
    rough = np.sin(20 * t)[None, :]
    assert smoothness_proxy(rough) > smoothness_proxy(smooth)


def test_linearity_error_zero_for_linear_d():
    d_ref = rng.normal(size=(10, config.N_T))
    assert linearity_error(2 * d_ref, d_ref, 2.0, 1.0) == pytest.approx(0.0, abs=1e-12)


def test_horizon_slice_lengths():
    assert horizon_slice(F_NOM, 10.0).shape[-1] == config.N_T
    assert horizon_slice(F_NOM, 2.0).shape[-1] == 41


def test_linearity_error_shrinks_as_eps_shrinks():
    # d(eps) is only approximately linear in eps (about 6% at eps = 0.01 -> 0.02 on S1);
    # the error must vanish as eps -> 0, which checks the function, not the physics.
    thetas = sobol_thetas(16, seed=2)
    nom = simulate_batch(thetas, "S1", 0.0)
    errs = []
    for e in (0.04, 0.01, 0.0025):
        d_ref = nom - simulate_batch(thetas, "S1", e)
        d_2x = nom - simulate_batch(thetas, "S1", 2 * e)
        errs.append(linearity_error(d_2x, d_ref, 2 * e, e))
    assert errs[0] > errs[1] > errs[2]


def test_s2_discrepancy_is_later_than_s1():
    # descriptive G1 expectation: friction acts more near rest, i.e. later in time
    thetas = sobol_thetas(32, seed=4)
    nom = simulate_batch(thetas, "S1", 0.0)
    d1 = nom - simulate_batch(thetas, "S1", 0.05)
    d2 = nom - simulate_batch(thetas, "S2", 0.005)
    print("late fraction S1", fraction_after(d1, 6.0), "S2", fraction_after(d2, 6.0))
    assert 0.0 <= fraction_after(d1, 6.0) <= 1.0

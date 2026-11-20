import numpy as np
from scipy import stats as sps

from mismatch import stats


def test_holm_matches_hand_computation():
    adj = stats.holm([0.01, 0.04, 0.03, 0.005])
    # sorted: 0.005*4=0.02, 0.01*3=0.03, 0.03*2=0.06, 0.04*1 -> max(0.06, 0.04)=0.06
    assert np.allclose(adj, [0.03, 0.06, 0.06, 0.02])


def test_paired_t_matches_scipy():
    x = np.random.default_rng(0).normal(0.3, 1, 20)
    assert np.isclose(stats.paired_t(x)["p"], sps.ttest_1samp(x, 0).pvalue)


def test_bca_ci_covers_mean_and_is_ordered():
    x = np.random.default_rng(1).normal(1.0, 0.5, 20)
    lo, hi = stats.bca_ci(x, n_boot=2000)
    assert lo < x.mean() < hi


def test_sign_test_all_positive():
    s = stats.sign_test(np.ones(10))
    assert s["n_pos"] == 10 and np.isclose(s["p"], 2 * 0.5 ** 10)


def test_tost_equivalent_and_not():
    rng = np.random.default_rng(2)
    assert stats.tost(rng.normal(0, 0.05, 20), 0.1)["equivalent"]
    assert not stats.tost(rng.normal(0.3, 0.05, 20), 0.1)["equivalent"]


def test_family_decision_rule():
    rng = np.random.default_rng(3)
    rep = {"big": stats.test_report(rng.normal(1, 0.3, 20)),
           "null": stats.test_report(rng.normal(0, 1, 20))}
    out = stats.family(rep, {"big": +1, "null": +1})
    assert out["big"]["supported"] and not out["null"]["supported"]


def test_family_wrong_direction_is_not_supported():
    rng = np.random.default_rng(4)
    out = stats.family({"b1": stats.test_report(rng.normal(1, 0.3, 20))}, {"b1": -1})
    assert not out["b1"]["supported"] and out["b1"]["opposite"]


def test_binom_ci_contains_proportion():
    lo, hi = stats.binom_ci(9, 10)
    assert lo < 0.9 < hi

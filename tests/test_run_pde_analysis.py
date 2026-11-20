"""Synthetic check of the PDE analysis (Family P, PA2, PD1) decision paths."""

import numpy as np

from scripts import run_pde_analysis as P


def fake_rank(n_seeds=12, falling=True, seed=0):
    rng = np.random.default_rng(seed)
    rank = {}
    for k in range(1, n_seeds + 1):
        for s in P.SYSTEMS:
            for t in P.TERMS:
                for m in P.SEVEN:
                    r = m * np.exp(rng.normal(0, 0.05))
                    x = np.log(r)
                    auc = 0.6 + 0.3 / (1 + np.exp((x - np.log(2.0)) * (2 if falling else -2)))
                    rank[(k, s, t, m, "ens_sigma")] = {"r": r, "auroc_e": auc + rng.normal(0, 0.01),
                                                       "spearman_e": 0.2 + rng.normal(0, 0.02),
                                                       "spearman_d": 0.2 + rng.normal(0, 0.02)}
    return rank


def test_family_p_supported_when_falling():
    res = P.family_p(fake_rank(), list(range(1, 13)))
    assert all(v["supported"] and not v["opposite"] for v in res.values())


def test_family_p_opposite_when_rising():
    res = P.family_p(fake_rank(falling=False), list(range(1, 13)))
    assert all(v["opposite"] and not v["supported"] for v in res.values())


def test_pa2_and_pd1():
    pred = {"fits": {f"{s}_{t}_ens_sigma": {"r_half": 2.0, "converged": True} for s in P.SYSTEMS
                     for t in P.TERMS}}
    res = P.pa2(fake_rank(), list(range(1, 13)), pred, n_boot=200)
    for v in res.values():
        assert v["verdict"] == "success", v
    far = {"fits": {k: {"r_half": 8.0, "converged": True} for k in pred["fits"]}}
    assert all(v["verdict"] == "fail" for v in P.pa2(fake_rank(), list(range(1, 13)), far, 200).values())
    d = P.pd1(fake_rank(), list(range(1, 13)))
    assert all(v["equivalent"] for v in d.values())

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
                                               "ens_sigma_gauss", "mc_sigma_gauss", "conf_nom", "cell_summary"}


def test_cell_values_are_valid(cell_records):
    full, _ = cell_records
    for r in [x for x in full if x["signal"] != "cell_summary"]:
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


from mismatch.rq1 import cell_summary, d3_records, debiased_discrepancy
from mismatch.signals import residual_profile, residual_score, windowed_residual_score


@pytest.fixture(scope="module")
def cell_arrays():
    train = sobol_thetas(128, seed=SEED)
    y = simulate_batch(train, "S1", 0.0)
    nominal = Ensemble(n_members=2, cfg=FAST, base_seed=SEED).fit(train, y)
    mc = Ensemble(n_members=1, cfg=FitConfig(hidden=32, max_epochs=150, patience=50,
                                             dropout=0.1), base_seed=SEED + 1).fit(train, y)
    test = sobol_thetas(60, seed=SEED + 7)
    f_true = simulate_batch(test, "S1", 0.02)
    oracle = make_oracle("S1", 0.02, seed=SEED)
    records, arrays = rq1_cell(nominal, mc, test, f_true, oracle, SEED, include_true_data=False,
                               d3=True, n_mc=10, n_cal_true=(8, 16), return_arrays=True)
    return records, arrays, nominal, test, f_true


def test_full_window_equals_full_arrays(cell_arrays):
    _, arrays, _, _, _ = cell_arrays
    for key in ("A", "D", "E", "ens_sigma", "mc_sigma", "ode_resid"):
        np.testing.assert_allclose(arrays[f"{key}_10"], arrays[key], rtol=1e-10)
    assert arrays["A"].shape == (60,)


def test_short_window_is_the_prefix_norm(cell_arrays):
    _, arrays, nominal, test, f_true = cell_arrays
    a = nominal.predict(test)[0] - simulate_batch(test, "S1", 0.0)
    expected = np.linalg.norm(a[:, :41], axis=1) / np.linalg.norm(f_true[:, :41], axis=1)
    np.testing.assert_allclose(arrays["A_2"], expected, rtol=1e-10)


def test_windowed_residual_matches_full_score():
    thetas = sobol_thetas(10, seed=SEED)
    traj = simulate_batch(thetas, "S1", 0.05)
    profile = residual_profile(traj, thetas)
    np.testing.assert_allclose(windowed_residual_score(profile, traj.shape[1]),
                               residual_score(traj, thetas), rtol=1e-12)


def test_cell_summary_matches_arrays(cell_arrays):
    records, arrays, _, _, _ = cell_arrays
    s = next(r for r in records if r["signal"] == "cell_summary")
    A, D = arrays["A"], arrays["D"]
    assert s["kappa"] == pytest.approx(np.sqrt(np.mean(A ** 2)) / np.std(D))
    assert s["cv_d"] == pytest.approx(np.std(D) / np.mean(D))
    assert s["r"] == pytest.approx(np.median(D) / np.median(A))
    assert cell_summary(A, np.zeros_like(D), 0.0)["kappa"] is None      # no spread in d


def test_d3_records_have_the_expected_shape(cell_arrays):
    records, _, _, _, _ = cell_arrays
    d3 = [r for r in records if r["signal"] == "d3"]
    assert [r["n_cal"] for r in d3] == [8, 16]
    for r in d3:
        assert -1.0 <= r["rho_hat"] <= 1.0 and r["ci_lo"] <= r["ci_hi"]
        assert r["aligned"] == (r["ci_lo"] > 0)


def test_d3_detects_a_monotone_relation_and_not_noise():
    rng = np.random.default_rng(3)
    dhat = rng.random(32)
    aligned = d3_records(dhat * 2 + 1, dhat, (16, 32), seed=1)
    assert all(r["aligned"] and r["rho_hat"] > 0.99 for r in aligned)
    noise = d3_records(rng.random(32), dhat, (16,), seed=1)
    assert not noise[0]["aligned"]


def test_debiased_discrepancy_removes_noise_energy():
    rng = np.random.default_rng(4)
    f = rng.normal(size=(5, 201))
    d = 0.1 * rng.normal(size=(5, 201))
    exact = debiased_discrepancy(f + d, f, np.zeros(5))
    np.testing.assert_allclose(exact, np.linalg.norm(d, axis=1) / np.linalg.norm(f + d, axis=1))
    sigma = np.full(5, 0.05)
    noisy = f + 0.05 * rng.normal(size=f.shape)
    assert np.all(debiased_discrepancy(noisy, f, sigma) < 0.02)         # near zero discrepancy


def test_phase_guard():
    from scripts.run_rq1 import check_phase
    check_phase([999, 998], "dev")
    for bad in ([0], [101], [5], [999, 3]):
        with pytest.raises(SystemExit):
            check_phase(bad, "dev")
    check_phase([0, 101, 102], "pilot")
    for bad in ([1], [0, 5], [999]):
        with pytest.raises(SystemExit):
            check_phase(bad, "pilot")
    check_phase([1, 10], "main", pred_tag_exists=True)
    with pytest.raises(SystemExit):
        check_phase([1], "main", pred_tag_exists=False)        # no pred-v1 tag yet
    check_phase([11, 20], "main", pred_tag_exists=True)              # raised to 20 by the pilot
    for bad in ([0], [21], [999]):
        with pytest.raises(SystemExit):
            check_phase(bad, "main", pred_tag_exists=True)
    with pytest.raises(SystemExit):
        check_phase([999], "other")


def test_extra_levels_only_at_the_extra_n_nom():
    from scripts.run_rq1 import cell_plan
    assert cell_plan(512, (0.0, 1.0, 4.0, 16.0)) == [(0.0, True, False), (1.0, True, False),
                                                      (4.0, False, True), (16.0, False, True)]
    assert [m for m, _, _ in cell_plan(128, (0.0, 1.0, 4.0, 16.0))] == [0.0, 1.0]


def test_runner_writes_per_seed_files_arrays_and_resumes(tmp_path):
    import json
    from scripts.run_rq1 import run
    small = FitConfig(hidden=32, max_epochs=60, patience=30)
    kw = dict(n_test=30, n_mc=4, nominal_members=2, nominal_cfg=small,
              mc_cfg=FitConfig(hidden=32, max_epochs=60, patience=30, dropout=0.1),
              disc_members=2, n_cal_true=(8, 16), extra_nnom=64)
    eps_ref = {"S1": 0.01, "S2": 0.0026}
    run([999], [64], ["S1"], (0.0, 1.0, 4.0), eps_ref, str(tmp_path), **kw)
    path = tmp_path / "rq1_seed999.jsonl"
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    assert {r["mult"] for r in rows} == {0.0, 1.0, 4.0}
    assert all(r["seed"] == 999 and r["n_nom"] == 64 and r["system"] == "S1" for r in rows)
    at4 = [r for r in rows if r["mult"] == 4.0]
    assert any(r["signal"] == "d3" for r in at4)                       # real-point records at extra levels
    assert not any(r["signal"] == "conf_true" for r in at4)            # but no fits on true data
    assert any(r["signal"] == "conf_true" for r in rows if r["mult"] == 1.0)
    npz = np.load(tmp_path / "arrays" / "seed999_n64_S1_x1.npz")
    assert npz["A"].shape == (30,) and "ens_sigma_4" in npz.files
    n_lines = len(rows)
    run([999], [64], ["S1"], (0.0, 1.0, 4.0), eps_ref, str(tmp_path), **kw)
    assert len(open(path, encoding="utf-8").readlines()) == n_lines    # finished cells are skipped

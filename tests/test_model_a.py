import numpy as np
import pytest

from mismatch.model_a import (blindness_horizon, closed_form_midpoint, cosines, empirical_auroc,
                              fit_4pl, four_pl, gaussian_auroc, gaussian_inputs, use_fallback)
from mismatch.rq1 import auroc
from mismatch.surrogate import FitConfig

NOALIGN = {"s_A": 0.8, "s_D": 0.5, "rho_sa": 0.8, "rho_sg": 0.0, "rho_ag": 0.0}


def test_gaussian_auroc_moves_from_a_to_d():
    assert gaussian_auroc(NOALIGN, 1e-3, n_draws=100000) > 0.85          # ranks a at small r
    assert gaussian_auroc(NOALIGN, 1e3, n_draws=100000) == pytest.approx(0.5, abs=0.01)
    aligned = {**NOALIGN, "rho_sa": 0.6, "rho_sg": 0.6}                   # valid correlation matrix
    small, large = (gaussian_auroc(aligned, r, n_draws=100000) for r in (1e-3, 1e3))
    assert large > 0.75 and large == pytest.approx(small, abs=0.01)     # aligned: stays high


def test_closed_form_midpoint_matches_monte_carlo():
    rs = np.exp(np.linspace(-4, 4, 33))
    fit = fit_4pl(rs, [gaussian_auroc(NOALIGN, r, n_draws=100000) for r in rs])
    assert fit["converged"]
    assert fit["r_half"] == pytest.approx(closed_form_midpoint(0.8, 0.5), rel=0.05)


def test_fit_4pl_recovers_parameters():
    r = np.exp(np.linspace(-3, 3, 25))
    y = four_pl(np.log(r), 0.45, 0.9, 0.3, 2.0)
    fit = fit_4pl(r, y)
    assert fit["converged"] and fit["r_half"] == pytest.approx(np.exp(0.3), rel=1e-3)
    assert fit["slope"] == pytest.approx(2.0, rel=1e-3) and fit["lo"] == pytest.approx(0.45, abs=1e-3)
    rising = fit_4pl(r, four_pl(np.log(r), 0.9, 0.5, 0.0, 2.0))         # aligned: AUROC rises
    assert rising["converged"] and rising["r_half"] == pytest.approx(1.0, rel=1e-3)
    assert rising["plateau_large_r"] == pytest.approx(0.9, abs=1e-3)       # side, not parameter name
    assert fit["plateau_large_r"] == pytest.approx(0.45, abs=1e-3)


def test_inputs_and_fallback():
    rng = np.random.default_rng(0)
    A = np.exp(rng.normal(0, 0.3, 400)); D = np.exp(rng.normal(0, 0.2, 400))
    inp = gaussian_inputs(A ** 0.5, A, D)
    assert inp["rho_sa"] == pytest.approx(1.0) and inp["s_A"] == pytest.approx(0.6, rel=0.1)
    small = {"median_cos": 0.0, "median_norm_point_cos": 0.1}
    assert not use_fallback(inp, small)
    assert use_fallback(inp, {"median_cos": 0.5, "median_norm_point_cos": 0.0})
    assert use_fallback({**inp, "s_A": 1.5}, small)


def test_empirical_and_cosines():
    rng = np.random.default_rng(1)
    f = rng.normal(size=(50, 20)); a = 0.1 * rng.normal(size=(50, 20)); sig = np.linalg.norm(a, axis=1)
    rel = np.linalg.norm(a, axis=1) / np.linalg.norm(f, axis=1)
    assert empirical_auroc(sig, a, np.zeros_like(a), f) == pytest.approx(auroc(sig, rel))
    assert cosines(a, a)["median_cos"] == pytest.approx(1.0)
    assert cosines(a, -a)["median_norm_point_cos"] == pytest.approx(-1.0)


def test_blindness_horizon():
    w = ["2", "4", "6"]
    assert blindness_horizon(w, {"2": 0.9, "4": 0.6, "6": 0.5}, dict.fromkeys(w, 0.9),
                             dict.fromkeys(w, 0.5)) == "4"
    assert blindness_horizon(w, dict.fromkeys(w, 0.9), dict.fromkeys(w, 0.9),
                             dict.fromkeys(w, 0.5)) == "> 10 s"


def test_runner_end_to_end_on_a_dev_seed(tmp_path):
    from scripts.run_model_a import run
    small = FitConfig(hidden=16, max_epochs=30, patience=10)
    params = {"nominal_cfg": small, "eps_ref": {"S1": 0.01, "S2": 0.003}}
    res = run(params, str(tmp_path), n_draws=2000, grid=(0.25, 0.5, 1.0, 2.0, 8.0, 32.0), n_test=30,
              members=2, collapse_nnom=(64, 128), seed=999,
              mc_cfg=FitConfig(hidden=16, max_epochs=30, patience=10, dropout=0.1), n_mc=5)
    s1 = res["systems"]["S1"]
    assert set(s1["fallback"]) == {"ens_sigma", "mc_sigma", "ode_resid"}
    assert set(s1["P_C"]["blindness_horizon"]) == {"0.25", "0.5", "1", "2"}
    assert 0 <= s1["exact"]["ens_sigma"]["empirical"]["auroc_at_mult"]["1"] <= 1
    assert "collapse_predicted" in res["collapse"]["verdict"]["S2"]["ens_sigma"]
    assert (tmp_path / "model_a_seed999.json").exists()

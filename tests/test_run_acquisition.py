import json

import numpy as np
import torch

from mismatch.surrogate import FitConfig
from scripts.run_acquisition import run

torch.set_num_threads(1)
SMALL = FitConfig(hidden=32, max_epochs=60, patience=30)
DISC = FitConfig(hidden=16, max_epochs=40, min_val=10 ** 9)
EPS = {"S1": 0.01, "S2": 0.0026}
SEED = 999  # development seed; never 0, 101, 102 or 1-10


def _rows(path):
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def test_family_a_runner_files_and_resume(tmp_path):
    kw = dict(n_nom=64, n_test=30, nominal_members=2, nominal_cfg=SMALL, arms=(2, 3, 8),
              problem_kw=dict(n_init=4, batch=4, rounds=2, pool_size=48, disc_members=2,
                              disc_cfg=DISC))
    run("A", [SEED], ["S1"], (0.0, 1.0), EPS, str(tmp_path), **kw)
    path = tmp_path / "acqA_seed999.jsonl"
    rows = _rows(path)
    assert {(r["mult"], r["arm"]) for r in rows} == {(m, a) for m in (0.0, 1.0) for a in (2, 3, 8)}
    for r in rows:
        assert [x["n"] for x in r["log"]] == [4, 8, 12]
        assert r["budget_auc"] > 0 and "uncorrected_median" in r and r["family"] == "A"
    assert all(len(r["gate_log"]) == 2 for r in rows if r["arm"] == 8)
    npz = np.load(tmp_path / "arrays" / "acqA_seed999_S1_x1.npz")
    assert set(npz.files) == {"thetas_sobol", "thetas_nominal_sigma", "thetas_r_gated"}
    assert npz["thetas_sobol"].shape == (12, 4)
    run("A", [SEED], ["S1"], (0.0, 1.0), EPS, str(tmp_path), **kw)
    assert len(_rows(path)) == len(rows)                       # finished cells are skipped


def test_family_f_runner_records_both_errors(tmp_path):
    kw = dict(n_nom=64, n_test=30, nominal_members=2, nominal_cfg=SMALL,
              arms=("sobol", "state_coverage"), f_kw=dict(k_terms=2, lbfgs_iters=2, gm_steps=40),
              problem_kw=dict(n_init=4, batch=4, rounds=2, pool_size=48))
    run("F", [SEED], ["S1"], (4.0,), EPS, str(tmp_path), **kw)
    rows = _rows(tmp_path / "acqF_seed999.jsonl")
    assert {r["name"] for r in rows} == {"sobol", "state_coverage"}
    for r in rows:
        assert r["family"] == "F" and r["mult"] == 4.0
        assert r["budget_auc"] > 0 and r["budget_auc_corrected"] > 0
        assert [x["n"] for x in r["log"]] == [4, 8, 12]
    npz = np.load(tmp_path / "arrays" / "acqF_seed999_S1_x4.npz")
    assert set(npz.files) == {"thetas_sobol", "thetas_state_coverage"}

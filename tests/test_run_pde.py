import json
import os

import numpy as np
import pytest

from mismatch import operator as O
from scripts import run_pde as rp
from scripts.run_pilot import load_params


def test_phase_guard():
    with pytest.raises(SystemExit):
        rp.check_phase([1], "dev")
    with pytest.raises(SystemExit):
        rp.check_phase([5], "pilot")
    with pytest.raises(SystemExit):
        rp.check_phase([3], "main", tags={"prereg-v1.7": True, "pred-v2": False})
    rp.check_phase([3], "main", tags={"prereg-v1.7": True, "pred-v2": True})
    rp.check_phase([999], "dev")


def test_noise_depends_only_on_seed_and_theta():
    y = np.ones((3, 10)) * np.arange(10)
    th = np.arange(12.0).reshape(3, 4)
    a, _ = rp.noisy(y, th, 7)
    b, _ = rp.noisy(y[::-1], th[::-1], 7)
    assert np.allclose(a, b[::-1])


def test_dev_end_to_end_both_systems(tmp_path, monkeypatch):
    monkeypatch.setattr(rp, "N_TEST", 40)
    monkeypatch.setattr(rp, "N_NOM", 48)
    monkeypatch.setattr(rp, "N_CAL_NOMINAL", 20)
    orig = O.OperatorEnsemble.__init__
    monkeypatch.setattr(O.OperatorEnsemble, "__init__",
                        lambda self, **kw: orig(self, **{**kw, "epochs": 3, "patience": 3}))
    cfg = load_params("pilot_results/frozen_params.json")["nominal_cfg"]
    objs = [rp.Burgers(cfg), rp.Operator()]
    eps_ref = {"B": {"P1": 0.003, "P2": 0.005}, "O": {"P1": 0.003, "P2": 0.005}}
    rp.run_cells(objs, [999], eps_ref, str(tmp_path), mults=(0.0, 1.0, 16.0), members=2)
    recs = [json.loads(l) for l in open(tmp_path / "pde_seed999.jsonl")]
    keys = {(r["system"], r["term"], r["mult"]) for r in recs}
    assert len(keys) == 2 * 2 * 3
    sig = {(r["system"], r["signal"]) for r in recs}
    assert ("B", "mc_sigma") in sig and ("O", "mc_sigma") not in sig
    assert any(r["signal"] == "d3" for r in recs)
    r1 = [r for r in recs if r["signal"] == "ens_sigma" and r["mult"] == 1.0]
    assert all(r["r"] > 0 for r in r1)
    assert os.path.exists(tmp_path / "arrays" / "seed999_O_P2_x16.npz")
    n_before = len(recs)
    rp.run_cells(objs, [999], eps_ref, str(tmp_path), mults=(0.0, 1.0, 16.0), members=2)  # resumes
    assert len(open(tmp_path / "pde_seed999.jsonl").readlines()) == n_before

import json

import pytest

from mismatch.surrogate import FitConfig
from scripts import run_pilot as rp

SMALL = FitConfig(hidden=16, max_epochs=40, patience=20)
SEED = 999


def test_eps_stage_brackets_r_equal_one(monkeypatch):
    monkeypatch.setattr(rp, "SYSTEMS", ("S1",))
    monkeypatch.setattr(rp, "EPS_BRACKET", (1e-4, 20.0))      # the tiny test surrogate is inaccurate
    out = rp.stage_eps(seed=SEED, n_nom=64, n_test=30, members=2, cfg=SMALL)
    assert 1e-4 < out["eps_ref"]["S1"] < 20.0


def test_hyper_stage_picks_a_grid_label():
    grid = {"default": {}, "hidden32": {"hidden": 32}}
    out = rp.stage_hyper(seed=SEED, n_nom=64, base={"max_epochs": 40, "patience": 20}, grid=grid)
    assert out["best"] in grid and set(out["mean_val_loss"]) == set(grid)


def test_cached_stage_is_computed_once(tmp_path):
    calls = []
    f = lambda: calls.append(1) or {"x": 1}
    assert rp.cached(str(tmp_path), "t", f)["x"] == 1 and rp.cached(str(tmp_path), "t", f)["x"] == 1
    assert len(calls) == 1


def test_load_params_roundtrip(tmp_path):
    path = tmp_path / "p.json"
    json.dump({"eps_ref": {"S1": 0.01, "S2": 0.003}, "k": 2.0, "spline_lam": 1e-3,
               "gp_components": 5, "nominal_cfg": {"hidden": 64}}, open(path, "w"))
    p = rp.load_params(str(path))
    assert p["nominal_cfg"].hidden == 64 and p["problem_kw"]["gp_components"] == 5 and p["k"] == 2.0

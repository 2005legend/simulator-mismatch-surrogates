"""End-to-end check of scripts/run_analysis.py on synthetic records covering every cell, so each
test path runs before any main-run data is read. Values are made up; only the plumbing and the
direction logic are checked."""

import json
import os

import numpy as np

from scripts import run_analysis as ra

SEEDS = list(range(1, 7))


def _auc(r, s):
    # S1 rises with r, S2 falls (the dev-seed pattern)
    x = np.log(r) if r > 0 else -5.0
    return 0.7 + (0.25 if s == "S1" else -0.35) / (1 + np.exp(-1.5 * x))


def write_synthetic(root, rng):
    rq1, acq = os.path.join(root, "rq1"), os.path.join(root, "acq")
    os.makedirs(os.path.join(rq1, "arrays"))
    os.makedirs(acq)
    for seed in SEEDS:
        recs, arecs, frecs = [], [], []
        for s in ra.SYSTEMS:
            for n in ra.NNOM:
                mults = (0.0,) + (ra.SEVEN if n == 512 else ra.FROZEN)
                for m in mults:
                    r = m * np.exp(rng.normal(0, 0.05))
                    base = {"seed": seed, "n_nom": n, "system": s, "mult": m, "eps": m, "n_cal": 0}
                    for sig in ("ens_sigma", "mc_sigma", "ode_resid"):
                        recs.append({**base, "signal": sig, "r": r,
                                     "auroc_e": _auc(r, s) + rng.normal(0, 0.01),
                                     "spearman_e": 0.5, "spearman_a": 0.6,
                                     "spearman_d": 0.5 + rng.normal(0, 0.01) if m else float("nan")})
                    recs.append({**base, "signal": "cell_summary", "r": r, "kappa": 1 / max(r, 1e-3),
                                 "cv_d": 0.5, "median_a": 0.02, "rms_a": 0.03, "sd_d": 0.01,
                                 "mean_d": 0.02})
                    if n == 512 and m == 16.0:
                        for nc in (8, 16, 32, 64):
                            recs.append({**base, "signal": "d3", "n_cal": nc, "rho_hat": 0.5,
                                         "ci_lo": 0.1, "ci_hi": 0.9, "aligned": s == "S1"})
                    if n == 512:
                        z = {}
                        for w in ra.WINDOWS:
                            A = rng.lognormal(0, 0.3, 50)
                            D = m * A * rng.lognormal(0, 0.3, 50)
                            z.update({f"A_{w}": A, f"D_{w}": D, f"E_{w}": A + D,
                                      f"ens_sigma_{w}": A * rng.lognormal(0, 0.2, 50)})
                        np.savez(os.path.join(rq1, "arrays", f"seed{seed}_n512_{s}_x{m:g}.npz"), **z)
            for m in (0.0,) + ra.FROZEN:
                for arm, name in ra.ARM_A.items():
                    auc = 0.02 * np.exp(rng.normal(0, 0.05) + (0.1 if name == "r_gated" else 0))
                    arecs.append({"seed": seed, "family": "A", "system": s, "mult": m, "arm": arm,
                                  "name": name, "budget_auc": auc, "uncorrected_median": 0.02})
            for name in ("random", "sobol", "nominal_sigma", "term_disagreement", "state_coverage"):
                frecs.append({"seed": seed, "family": "F", "system": s, "mult": 4.0, "arm": name,
                              "name": name, "budget_auc": 0.01 * np.exp(rng.normal(0, 0.1))})
        for fn, rr in ((os.path.join(rq1, f"rq1_seed{seed}.jsonl"), recs),
                       (os.path.join(acq, f"acqA_seed{seed}.jsonl"), arecs),
                       (os.path.join(acq, f"acqF_seed{seed}.jsonl"), frecs)):
            with open(fn, "w", encoding="utf-8") as fh:
                for r in rr:
                    fh.write(json.dumps(r) + "\n")
    return rq1, acq


def test_run_analysis_covers_every_test(tmp_path):
    rq1, acq = write_synthetic(str(tmp_path), np.random.default_rng(0))
    res = ra.run(rq1, acq, SEEDS, str(tmp_path / "out"), n_boot=50,
                 pred_path="predictions/model_a_seed0.json", pilot_params="pilot_results/stage_fcv.json")
    B = res["family_B"]
    assert set(B) == {"B1_S1", "B1_S2", "B2_S1", "B2_S2"}
    assert B["B1_S2"]["supported"] and not B["B1_S1"]["supported"] and B["B1_S1"]["opposite"]
    assert all("verdict" in B[k] for k in ("B2_S1", "B2_S2"))
    assert set(res["family_A"]) == {f"{t}_{s}" for t in ("A1", "A2", "A3") for s in ra.SYSTEMS}
    assert res["family_C"]["C1_S1"]["mean"] > 0 and not res["family_C"]["C1_S1"]["supported"]
    assert set(res["family_F"]) >= {"F1", "F2", "F3", "F4", "stop_rule"}
    assert "inconclusive" in res["family_F"]["stop_rule"] or "beats" in res["family_F"]["stop_rule"]
    assert res["D"]["D2_S1"]["success"] and res["D"]["D2_S2"]["success"]
    assert res["D"]["D3_S1"]["success"] and res["D"]["D3_S2"]["success"]
    assert res["P_A"]["S1"]["ens_sigma"]["fit"]["converged"]
    assert res["P_C"]["S2"]["observed_horizon"]
    assert res["r_reuse_check"]["cells_checked"] == 2 * len(SEEDS)

"""Development check of the alignment-range tests on System 3, Burgers (dev seed 999 only).
Refits the dev nominal ensemble, then at 4 x and 16 x a rough eps_ref (r about 1) computes the
R1/R1s/R1c numbers and the R3a/R3b term selection from n = 8..64 noisy true fields, to set the
v1.7 thresholds the way v1.6 set them for the pendulum.

Usage:  python -m scripts.dev_burgers_alignment
"""

import json
import os

import numpy as np
import torch
from scipy.stats import qmc, spearmanr

from mismatch import burgers as B
from mismatch.alignment import alignment_range, response_norm
from mismatch.rq1 import debiased_discrepancy
from mismatch.surrogate import Ensemble
from scripts.run_pilot import load_params

DEV_SEED = 999
EPS_REF = {"P1": 0.0034, "P2": 0.0056}       # rough r = 1 from dev_burgers_feasibility


def rho(a, b):
    return float(spearmanr(a, b)[0])


def main():
    torch.set_num_threads(1)
    cfg = load_params(os.path.join("pilot_results", "frozen_params.json"))["nominal_cfg"]
    sob = lambda n, s: B.from_unit(qmc.Sobol(4, scramble=True, seed=s).random(n))
    train, test, cal = sob(512, DEV_SEED), sob(512, DEV_SEED + 1)[:500], sob(64, DEV_SEED + 2)
    ens = Ensemble(n_out=B.N_OUT, n_members=5, cfg=cfg, base_seed=DEV_SEED).fit(
        B.to_unit_box(train), B.simulate_batch(train))
    mean, sigma = ens.predict(B.to_unit_box(test))
    sig = {"ens_sigma": sigma.mean(axis=1), "pde_resid": B.residual_score(mean, test)}
    names, resp, base = B.term_responses(test)
    _, cresp, cbase = B.term_responses(cal)
    cnorm = response_norm(cresp, cbase)
    tnorm = dict(zip(names, response_norm(resp, base)))
    f_nom, c_nom = base, B.simulate_batch(cal)
    rng = np.random.default_rng(DEV_SEED)
    out = {}
    for system, e_ref in EPS_REF.items():
        fam = B.TRUE_FAMILY[system]
        keep = [i for i, n in enumerate(names) if n not in fam]
        out[system] = {}
        for mult in (1.0, 4.0, 16.0):
            eps = mult * e_ref
            f_true = B.simulate_batch(test, system, eps)
            D = np.linalg.norm(f_nom - f_true, axis=1) / np.linalg.norm(f_true, axis=1)
            row = {}
            for sname, s in sig.items():
                full = alignment_range(s, names, resp, base, seed=DEV_SEED)
                red = alignment_range(s, [names[i] for i in keep], resp[keep], base, seed=DEV_SEED)
                act = rho(s, D)
                fp = float(np.mean([full["per_term"][n] for n in fam]))
                row[sname] = {"actual": act, "range_full": [full["min"], full["max"]],
                              "range_reduced": [red["min"], red["max"]],
                              "band_reduced": [red["mix_quantiles"][0], red["mix_quantiles"][2]],
                              "family_pred": fp, "gap": abs(act - fp)}
            yc = B.simulate_batch(cal, system, eps)
            sd = B.NOISE * yc.std(axis=1)
            yc = yc + sd[:, None] * rng.normal(size=yc.shape)
            dhat = debiased_discrepancy(yc, c_nom, sd)
            r3 = {}
            for lab, idx in (("R3a", list(range(len(names)))), ("R3b", keep)):
                r3[lab] = {}
                for n in (8, 16, 32, 64):
                    c = {names[i]: rho(dhat[:n], cnorm[i, :n]) for i in idx}
                    ch = max(c, key=lambda k: c[k] if np.isfinite(c[k]) else -9)
                    r3[lab][n] = (ch, rho(tnorm[ch], D))
            row["R3"], row["sigma_stat"] = r3, rho(sig["ens_sigma"], D)
            out[system][mult] = row
            e = row["ens_sigma"]
            print(f"{system} x{mult:g}: sigma actual {e['actual']:+.2f} range {np.round(e['range_full'], 2)} "
                  f"family {e['family_pred']:+.2f} gap {e['gap']:.2f} | resid actual {row['pde_resid']['actual']:+.2f} "
                  f"family {row['pde_resid']['family_pred']:+.2f} gap {row['pde_resid']['gap']:.2f}", flush=True)
            print(f"   reduced range sigma {np.round(e['range_reduced'], 2)} band {np.round(e['band_reduced'], 2)}")
            print(f"   R3a {r3['R3a']}\n   R3b {r3['R3b']}", flush=True)
    os.makedirs(os.path.join("runs", "dev_burgers"), exist_ok=True)
    json.dump(out, open(os.path.join("runs", "dev_burgers", "alignment.json"), "w"), indent=1, default=str)


if __name__ == "__main__":
    main()

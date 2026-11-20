"""Development feasibility for System 3, Burgers (dev seed 999 only; not pre-registered).

Fits the nominal ensemble (frozen pendulum settings, 5 members) on 512 nominal space-time fields,
then sweeps eps on both missing terms and prints r, the ranking of the total error by ensemble
sigma and by the PDE residual, and the two limits rho(s, A) and rho(s, D). Used to choose the
eps grid, eps_ref and settings for addendum v1.7.

Usage:  python -m scripts.dev_burgers_feasibility
"""

import json
import os
import time

import numpy as np
import torch
from scipy.stats import qmc, spearmanr

from mismatch import burgers as B
from mismatch.rq1 import auroc
from mismatch.surrogate import Ensemble
from scripts.run_pilot import load_params

DEV_SEED = 999
EPS = {"P1": (0.003, 0.01, 0.03, 0.1, 0.3, 1.0), "P2": (0.001, 0.003, 0.01, 0.03, 0.1, 0.3)}


def main():
    torch.set_num_threads(1)
    cfg = load_params(os.path.join("pilot_results", "frozen_params.json"))["nominal_cfg"]
    train = B.from_unit(qmc.Sobol(4, scramble=True, seed=DEV_SEED).random(512))
    test = B.from_unit(qmc.Sobol(4, scramble=True, seed=DEV_SEED + 1).random(500))
    t0 = time.perf_counter()
    y, f_nom = B.simulate_batch(train), B.simulate_batch(test)
    print(f"1012 solves: {time.perf_counter() - t0:.1f} s", flush=True)
    t0 = time.perf_counter()
    ens = Ensemble(n_out=B.N_OUT, n_members=5, cfg=cfg, base_seed=DEV_SEED).fit(B.to_unit_box(train), y)
    print(f"ensemble fit: {time.perf_counter() - t0:.1f} s", flush=True)
    mean, sigma = ens.predict(B.to_unit_box(test))
    a = mean - f_nom
    sig = {"ens_sigma": sigma.mean(axis=1), "pde_resid": B.residual_score(mean, test)}
    out = {"median_rel_A": float(np.median(np.linalg.norm(a, axis=1) / np.linalg.norm(f_nom, axis=1)))}
    print(f"median relative approximation error {out['median_rel_A']:.4f}")
    for system, levels in EPS.items():
        print(f"\n{system}: eps | r | AUROC(E) sigma, resid | rho(s,E) rho(s,A) rho(s,D) for sigma | same for resid")
        out[system] = []
        for eps in levels:
            f_true = B.simulate_batch(test, system, eps)
            d = f_nom - f_true
            ref = np.linalg.norm(f_true, axis=1)
            A, D, E = (np.linalg.norm(v, axis=1) / ref for v in (a, d, a + d))
            r = float(np.median(D) / np.median(A))
            row = {"eps": eps, "r": r}
            for name, s in sig.items():
                row[name] = {"auroc": auroc(s, E), **{f"rho_{k}": float(spearmanr(s, v)[0])
                                                      for k, v in (("E", E), ("A", A), ("D", D))}}
            out[system].append(row)
            f = lambda n: f"{row[n]['rho_E']:+.2f} {row[n]['rho_A']:+.2f} {row[n]['rho_D']:+.2f}"
            print(f"{eps:6.3f} | {r:7.2f} | {row['ens_sigma']['auroc']:.3f} {row['pde_resid']['auroc']:.3f} | "
                  f"{f('ens_sigma')} | {f('pde_resid')}", flush=True)
    os.makedirs(os.path.join("runs", "dev_burgers"), exist_ok=True)
    json.dump(out, open(os.path.join("runs", "dev_burgers", "feasibility.json"), "w"), indent=1)


if __name__ == "__main__":
    main()

"""Gate G3 for System 4 (dev seed 999 only): fit time per FNO member on CPU and median relative
error at n = 512 training profiles, plus a first look at ranking under mismatch.
Pass: at most 300 s per member and at most 5% median relative error.

Usage:  python -m scripts.dev_operator_gate [n_members]
"""

import json
import os
import sys
import time

import numpy as np
import torch
from scipy.stats import spearmanr

from mismatch import operator as O
from mismatch.rq1 import auroc

DEV_SEED = 999


def main():
    torch.set_num_threads(1)
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    ctr, cte = O.sample_coefs(512, DEV_SEED), O.sample_coefs(500, DEV_SEED + 1)
    t0 = time.perf_counter()
    ytr, u0tr = O.simulate(ctr)
    yte, u0te = O.simulate(cte)
    print(f"solves: {time.perf_counter() - t0:.1f} s", flush=True)
    ens = O.OperatorEnsemble(n_members=M, base_seed=DEV_SEED).fit(u0tr, ytr)
    mean, sd = ens.predict(u0te)
    flat = lambda z: z.reshape(len(z), -1)
    a = flat(mean - yte)
    A = np.linalg.norm(a, axis=1) / np.linalg.norm(flat(yte), axis=1)
    res = {"fit_seconds": ens.fit_seconds, "median_rel_err": float(np.median(A)),
           "gate_pass": bool(max(ens.fit_seconds) <= 300 and np.median(A) <= 0.05)}
    print(f"fit seconds per member {[round(t) for t in ens.fit_seconds]}, median rel error "
          f"{res['median_rel_err']:.4f}, gate {'PASS' if res['gate_pass'] else 'FAIL'}", flush=True)
    s = flat(sd).mean(axis=1)
    res["sweep"] = {}
    for system, levels in (("P1", (0.01, 0.03, 0.1, 0.3, 1.0)), ("P2", (0.003, 0.01, 0.03, 0.1, 0.3))):
        rows = []
        for eps in levels:
            yt, _ = O.simulate(cte, system, eps)
            d = flat(yte - yt)
            ref = np.linalg.norm(flat(yt), axis=1)
            Aa, D, E = (np.linalg.norm(v, axis=1) / ref for v in (a, d, a + d))
            row = {"eps": eps, "r": float(np.median(D) / np.median(Aa)), "auroc": auroc(s, E),
                   "rho_E": float(spearmanr(s, E)[0]), "rho_A": float(spearmanr(s, Aa)[0]),
                   "rho_D": float(spearmanr(s, D)[0])}
            rows.append(row)
            print(f"{system} eps {eps:6.3f} r {row['r']:7.2f} AUROC {row['auroc']:.3f} "
                  f"rho E/A/D {row['rho_E']:+.2f} {row['rho_A']:+.2f} {row['rho_D']:+.2f}", flush=True)
        res["sweep"][system] = rows
    os.makedirs(os.path.join("runs", "dev_burgers"), exist_ok=True)
    json.dump(res, open(os.path.join("runs", "dev_burgers", "operator_gate.json"), "w"), indent=1)


if __name__ == "__main__":
    main()

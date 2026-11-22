"""Post hoc, exploratory (2026-11-22): the quantities Proposition 1 actually uses, next to r.

For every system-term case and level, over seeds 1-20 (n_nom = 512, 500 test points):
  r      = median D / median A
  q_D    = fraction of pairs i < j with |D_i - D_j| <= A_i + A_j   (Kendall bound 2 q_D)
  kappa  = rms(A) / sd(D)                                        (Pearson bound 2 kappa / (1 - kappa))
  obs    = |tau(sigma, E) - tau(sigma, D)| for ensemble sigma
Reported as the mean over seeds. Output: r_diagnostics.json.
Answers the reviewer point that r is a descriptive index, not a sufficient statistic.
"""
import json
import os

import numpy as np
from scipy.stats import kendalltau

LEVELS = (1.0, 4.0, 16.0)
CASES = {"S1": "runs/rq1_main/arrays/seed{k}_n512_S1_x{m:g}.npz",
         "S2": "runs/rq1_main/arrays/seed{k}_n512_S2_x{m:g}.npz",
         **{c: f"runs/pde_main/arrays/seed{{k}}_{c}_x{{m:g}}.npz" for c in ("B_P1", "B_P2", "O_P1", "O_P2")},
         **{c: f"runs/pde_ood_main/arrays/seed{{k}}_{c}_x{{m:g}}.npz" for c in ("B_P3", "B_P4")},
         **{c: f"runs/pend_ood_main/arrays/seed{{k}}_{c}_x{{m:g}}.npz" for c in ("P_Q1", "P_Q2")},
         **{f"blind_{c}": f"runs/blind_main_{c[0]}/arrays/seed{{k}}_{c}_x{{m:g}}.npz"
            for c in ("P_aerodynamic_drag", "P_pivot_vibration", "B_forcing_sin_3x",
                      "B_cubic_damping", "B_quadratic_damping")}}


def q_d(A, D):
    i, j = np.triu_indices(len(D), 1)
    return float(np.mean(np.abs(D[i] - D[j]) <= A[i] + A[j]))


out = {}
for case, pat in CASES.items():
    out[case] = {}
    for m in LEVELS:
        rows = []
        for k in range(1, 21):
            f = pat.format(k=k, m=m)
            if not os.path.exists(f):
                continue
            a = np.load(f)
            A, D, E, s = a["A"], a["D"], a["E"], a["ens_sigma"]
            rows.append((np.median(D) / np.median(A), q_d(A, D),
                         np.sqrt(np.mean(A ** 2)) / np.std(D),
                         abs(kendalltau(s, E)[0] - kendalltau(s, D)[0])))
        if rows:
            v = np.mean(rows, axis=0)
            out[case][f"{m:g}"] = {"n": len(rows), "r": float(v[0]), "q_D": float(v[1]),
                                   "bound_2qD": float(2 * v[1]), "kappa": float(v[2]), "obs_tau_gap": float(v[3])}
            print(case, m, {k2: round(x, 3) for k2, x in out[case][f"{m:g}"].items()}, flush=True)
json.dump(out, open(os.path.join("theory", "checks", "r_diagnostics.json"), "w"), indent=1)

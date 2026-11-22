"""Post hoc, exploratory (2026-11-22): at 16 x eps_ref, rho(s, E) - rho(s, D) for every signal,
in Spearman's rho and Kendall's tau, over seeds 1-20, for all system-term cases (eight; P_Q1 and P_Q2 added 2026-11-22 after v1.9).
The pre-registered tests (D1, PD1, OD1) covered ensemble sigma only. Output: limit_all_signals.json."""
import json, os
import numpy as np
from scipy.stats import kendalltau, spearmanr

CASES = {"S1": ("runs/rq1_main/arrays/seed{k}_n512_S1_x16.npz", ("ens_sigma", "mc_sigma", "ode_resid")),
         "S2": ("runs/rq1_main/arrays/seed{k}_n512_S2_x16.npz", ("ens_sigma", "mc_sigma", "ode_resid")),
         **{c: (f"runs/pde_main/arrays/seed{{k}}_{c}_x16.npz", ("ens_sigma", "mc_sigma", "pde_resid"))
            for c in ("B_P1", "B_P2", "O_P1", "O_P2")},
         **{c: (f"runs/pde_ood_main/arrays/seed{{k}}_{c}_x16.npz", ("ens_sigma", "mc_sigma", "pde_resid"))
            for c in ("B_P3", "B_P4")},
         **{c: (f"runs/pend_ood_main/arrays/seed{{k}}_{c}_x16.npz", ("ens_sigma", "mc_sigma", "pde_resid"))
            for c in ("P_Q1", "P_Q2")}}
out = {}
for case, (pat, sigs) in CASES.items():
    for s in sigs:
        ds, dk = [], []
        for k in range(1, 21):
            a = np.load(pat.format(k=k))
            if s not in a:
                continue
            ds.append(spearmanr(a[s], a["E"])[0] - spearmanr(a[s], a["D"])[0])
            dk.append(kendalltau(a[s], a["E"])[0] - kendalltau(a[s], a["D"])[0])
        if ds:
            out[f"{case}:{s}"] = {"n": len(ds), "spearman_mean": float(np.mean(ds)), "spearman_max_abs": float(np.max(np.abs(ds))),
                                  "kendall_mean": float(np.mean(dk)), "kendall_max_abs": float(np.max(np.abs(dk)))}
json.dump(out, open(os.path.join("theory", "checks", "limit_all_signals.json"), "w"), indent=1)
print(max(v["spearman_max_abs"] for v in out.values()), max(v["kendall_max_abs"] for v in out.values()), len(out))

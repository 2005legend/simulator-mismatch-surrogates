"""Development look for addendum v1.8 (dev seed 999 only): out-of-dictionary terms on System 3.

Fits the System 3 nominal MLP ensemble (frozen settings), finds a rough eps_ref (r = 1) for the
OOD terms P3 (-u_x) and P4 (u_xx) by bisection, then at 1, 4 and 16 x reports r, AUROC and the
alignment of ensemble sigma and the PDE residual with ||d||, whether the full-dictionary
alignment range contains it, the out-of-span fraction of the term's response, and the R3 term
selection from 16 noisy fields at 4 x. Output: runs/dev_burgers_ood/dev_seed999.json.

Usage:  python -m scripts.dev_burgers_ood
"""

import json
import os

import numpy as np
import torch

from mismatch import burgers as B
from mismatch import burgers_ood as BO
from mismatch.alignment import alignment_range, response_norm
from mismatch.pilot import log_bisect
from mismatch.rq1 import debiased_discrepancy
from mismatch.simulators import stream_seed
from scripts.run_alignment import rho, select_term
from scripts.run_pde import N_TEST, Burgers, models, noisy, pde_cell
from scripts.run_pilot import load_params

DEV_SEED = 999
BRACKET = {"P3": (1e-5, 0.5), "P4": (1e-5, 0.1)}


class BurgersOOD(Burgers):
    def sim(self, th, term="P3", eps=0.0):
        return B.simulate_batch(th) if eps == 0 else BO.simulate_batch(th, term, eps)


def main():
    torch.set_num_threads(4)
    cfg = load_params(os.path.join("pilot_results", "frozen_params.json"))["nominal_cfg"]
    so = BurgersOOD(cfg)
    nominal, mc = models(so, DEV_SEED)
    test = so.sample(N_TEST, stream_seed(DEV_SEED, "test"))
    f_nom = so.sim(test)
    mean, _ = so.predict(nominal, test)
    names, resp, base = B.term_responses(test)
    cal = so.sample(64, stream_seed(DEV_SEED, "calibration", 1))
    _, cresp, cbase = B.term_responses(cal)
    cal_norms, test_norms = response_norm(cresp, cbase), dict(zip(names, response_norm(resp, base)))
    f_cal = so.sim(cal)
    out = {"seed": DEV_SEED, "terms": {}}
    for term in ("P3", "P4"):
        def r_of(eps):
            ft = so.sim(test, term, eps)
            A = np.linalg.norm(mean - f_nom, axis=1) / np.linalg.norm(ft, axis=1)
            D = np.linalg.norm(f_nom - ft, axis=1) / np.linalg.norm(ft, axis=1)
            return float(np.median(D) / np.median(A))
        eps_ref = log_bisect(r_of, 1.0, *BRACKET[term], n_iter=10)
        span = BO.out_of_span(BO.response(test, term), resp)
        res = {"eps_ref": eps_ref, "out_of_span_median": float(np.median(span)),
               "out_of_span_q10_q90": [float(np.quantile(span, 0.1)), float(np.quantile(span, 0.9))],
               "levels": {}}
        for m in (1.0, 4.0, 16.0):
            eps = m * eps_ref
            recs, arrs = pde_cell(so, nominal, mc, test, f_nom, so.sim(test, term, eps), term, eps, DEV_SEED)
            lev = {"r": float(np.median(arrs["D"]) / np.median(arrs["A"]))}
            for sig in ("ens_sigma", "pde_resid"):
                rk = [x for x in recs if x.get("signal") == sig][0]
                rng = alignment_range(arrs[sig], names, resp, base, n_mix=2000, seed=DEV_SEED)
                a = rho(arrs[sig], arrs["D"])
                lev[sig] = {"auroc_e": rk["auroc_e"], "rho_E": rk["spearman_e"], "rho_D": a,
                            "range": [rng["min"], rng["max"]], "inside": bool(rng["min"] <= a <= rng["max"])}
            if m == 4.0:
                y, sd = noisy(so.sim(cal, term, eps), cal, DEV_SEED)
                dhat = debiased_discrepancy(y, f_cal, sd)
                chosen, _ = select_term(dhat[:16], names, cal_norms[:, :16])
                st = rho(test_norms[chosen], arrs["D"])
                lev["R3_n16"] = {"selected": chosen, "stat": st, "sigma_stat": lev["ens_sigma"]["rho_D"],
                                 "beats_sigma": bool(st > lev["ens_sigma"]["rho_D"])}
            res["levels"][f"{m:g}"] = lev
            print(term, m, json.dumps(lev), flush=True)
        out["terms"][term] = res
        print(term, "eps_ref", eps_ref, "out-of-span median", res["out_of_span_median"], flush=True)
    os.makedirs(os.path.join("runs", "dev_burgers_ood"), exist_ok=True)
    json.dump(out, open(os.path.join("runs", "dev_burgers_ood", "dev_seed999.json"), "w"), indent=1)


if __name__ == "__main__":
    main()

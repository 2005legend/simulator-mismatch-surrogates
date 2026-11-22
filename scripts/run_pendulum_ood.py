"""Runner and analysis for addendum v1.9: out-of-dictionary missing terms on the pendulum.

Stages (same design as v1.8 on System 3):
  pilot     eps_ref per term (Q1, Q2) by bisection on r = 1 at n_nom = 512 -> <out>/eps_ref.json, then cells
  cells     levels 0-16 x eps_ref for terms Q1, Q2 for the given seeds
  align     per seed: QR1 (range), QR1L at 16 x and QR1L4 at 4 x (linear response), gap at every
            level, QR3 (selection at 4 x), out-of-span share (keys keep the v1.8 'OR' names)
  analyse   OB1 (Holm over 2), OD1 (TOST), OR1, OR1L, OR3 counts, descriptive -> results.json

Reuses the frozen v1.7 cell machinery (`scripts.run_pde`: run_cells, stage_eps, pde_cell) through
a pendulum adapter, with its term list and brackets replaced for this process only. The
adapter uses the same sample streams, surrogate settings (frozen nominal_cfg, 5 members,
n_nom = 512) and test set as the main pendulum runs, so a seed gives the same surrogate as in
RQ1. The residual signal is the ODE residual; it is stored under the runner's key "pde_resid".
No frozen module is modified.

Usage:  python -m scripts.run_pendulum_ood --phase dev --stage pilot --seeds 999 --out-dir runs/pend_ood_dev
        python -m scripts.run_pendulum_ood --phase pilot --stage pilot --seeds 0 --out-dir runs/pend_ood_pilot
        python -m scripts.run_pendulum_ood --phase main --stage cells --seeds 1 2 3
        python -m scripts.run_pendulum_ood --phase main --stage align --seeds 1-20
        python -m scripts.run_pendulum_ood --phase main --stage analyse --seeds 1-20
"""

import argparse
import glob
import json
import os
from multiprocessing import Pool

import numpy as np
from scipy.stats import binom

from mismatch import burgers_ood as BO
from mismatch import pendulum_ood as PO
from mismatch import stats
from mismatch.alignment import TERMS as DICT_TERMS
from mismatch.alignment import alignment_range, response_norm, term_responses
from mismatch.rq1 import debiased_discrepancy
from mismatch.signals import residual_score
from mismatch.simulators import sobol_thetas, stream_seed
from mismatch.surrogate import Ensemble, FitConfig
from scripts import run_pde as R
from scripts.run_alignment import count, rho, select_term
from scripts.run_pde import N_TEST, noisy
from scripts.run_pilot import load_params

TERMS = ("Q1", "Q2")
BRACKET = {"Q1": (1e-6, 0.1), "Q2": (1e-6, 0.1)}
SEVEN = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
SIGNALS = ("ens_sigma", "pde_resid")
MAIN_SEEDS = set(range(1, 21))
TAG = "prereg-v1.9"


class Pendulum:
    """Pendulum adapter for the frozen v1.7 cell runner; true system adds an OOD term."""
    name, has_mc = "P", True

    def __init__(self, cfg):
        self.cfg = cfg

    def sample(self, n, seed):
        return sobol_thetas(n, seed=seed)

    def sim(self, th, term="Q1", eps=0.0):
        return PO.simulate_batch(th, term, eps)

    def fit(self, th, y, seed, members=5):
        return Ensemble(n_members=members, cfg=self.cfg, base_seed=stream_seed(seed, "nominal")).fit(th, y)

    def fit_mc(self, th, y, seed):
        return Ensemble(n_members=1, cfg=FitConfig(dropout=0.1), base_seed=stream_seed(seed, "mc")).fit(th, y)

    def predict(self, model, th):
        return model.predict(th)

    def predict_mc(self, model, th, seed):
        return model.members[0].predict_mc(th, n_passes=50, seed=seed).std(axis=0)

    def residual(self, mean, th):
        return residual_score(mean, th)


def check_phase(seeds, phase, tag_ok=None):
    seeds = set(seeds)
    if phase == "dev":
        if seeds & (MAIN_SEEDS | {0, 101, 102}):
            raise SystemExit(f"refusing registered seeds {sorted(seeds)} in the dev phase")
        return
    ok = tag_ok if tag_ok is not None else R.tag_exists(TAG)
    if not ok:
        raise SystemExit(f"the v1.9 pilot and main phases need git tag {TAG}")
    if phase == "pilot" and seeds != {0}:
        raise SystemExit("the v1.9 pilot uses seed 0 only")
    if phase == "main" and not seeds <= MAIN_SEEDS:
        raise SystemExit("main seeds are 1-20")


def use_ood_terms():
    """Point the frozen runner at the v1.9 terms for this process."""
    R.TERMS, R.BRACKET = TERMS, BRACKET


# ---------------------------------------------------------------- align

def align_seed(args):
    seed, cell_dir, eps_ref = args
    import torch
    torch.set_num_threads(1)
    so = Pendulum(None)
    test = so.sample(N_TEST, stream_seed(seed, "test"))
    names, resp, base = term_responses(test)
    cal = so.sample(64, stream_seed(seed, "calibration", 1))
    _, cresp, cbase = term_responses(cal)
    cal_norms, test_norms = response_norm(cresp, cbase), dict(zip(names, response_norm(resp, base)))
    f_cal = so.sim(cal)
    recs = []
    for term in TERMS:
        arrs = {m: np.load(os.path.join(cell_dir, "arrays", f"seed{seed}_P_{term}_x{m:g}.npz"))
                for m in SEVEN}
        L = PO.response(test, term)
        lin = np.linalg.norm(L, axis=1) / np.linalg.norm(base, axis=1)
        span = BO.out_of_span(L, resp)
        rec = {"seed": seed, "term": term, "out_of_span_median": float(np.median(span)),
               "out_of_span_q10_q90": [float(np.quantile(span, 0.1)), float(np.quantile(span, 0.9))],
               "signals": {}}
        for sig in SIGNALS:
            s16, D16 = arrs[16.0][sig], arrs[16.0]["D"]
            rng = alignment_range(s16, names, resp, base, n_mix=2000, seed=seed)
            a = rho(s16, D16)
            pred = rho(s16, lin)
            rec["signals"][sig] = {"rho_D_x16": a, "range": [rng["min"], rng["max"]],
                                   "OR1_inside": bool(rng["min"] <= a <= rng["max"]),
                                   "rho_linear": pred, "OR1L_gap": abs(a - pred),
                                   "OR1L_ok": bool(abs(a - pred) <= 0.10),
                                   "OR1L4_gap": abs(rho(arrs[4.0][sig], arrs[4.0]["D"]) - pred),
                                   "gap_by_level": {f"{m:g}": abs(rho(arrs[m][sig], arrs[m]["D"]) - pred)
                                                    for m in SEVEN}}
            rec["signals"][sig]["OR1L4_ok"] = bool(rec["signals"][sig]["OR1L4_gap"] <= 0.10)
        eps = 4.0 * eps_ref[term]
        y, sd = noisy(so.sim(cal, term, eps), cal, seed)
        dhat = debiased_discrepancy(y, f_cal, sd)
        D4 = arrs[4.0]["D"]
        sigma_stat = rho(arrs[4.0]["ens_sigma"], D4)
        sel = {}
        for n in (8, 16, 32, 64):
            chosen, _ = select_term(dhat[:n], names, cal_norms[:, :n])
            st = rho(test_norms[chosen], D4)
            sel[str(n)] = {"selected": chosen, "stat": st, "beats_sigma": bool(st > sigma_stat)}
        rec["OR3"] = {"sigma_stat": sigma_stat, "by_n": sel}
        recs.append(rec)
    return seed, recs


def run_align(seeds, cell_dir, out_dir, eps_ref, workers):
    os.makedirs(out_dir, exist_ok=True)
    jobs = [(s, cell_dir, eps_ref) for s in seeds
            if not os.path.exists(os.path.join(out_dir, f"pend_ood_align_seed{s}.jsonl"))]
    with Pool(workers) as pool:
        for seed, recs in pool.imap_unordered(align_seed, jobs):
            with open(os.path.join(out_dir, f"pend_ood_align_seed{seed}.jsonl"), "w", encoding="utf-8") as fh:
                for r in recs:
                    fh.write(json.dumps(r) + "\n")
            print(f"align seed {seed} done", flush=True)


# ---------------------------------------------------------------- analyse

def analyse(seeds, cell_dir, align_dir, out_dir):
    recs = [json.loads(l) for fn in glob.glob(os.path.join(cell_dir, "pde_seed*.jsonl"))
            for l in open(fn, encoding="utf-8") if l.strip()]
    recs = [r for r in recs if r["seed"] in set(seeds) and r["term"] in TERMS]
    rank = {(r["seed"], r["term"], r["mult"], r["signal"]): r for r in recs if "auroc_e" in r}
    reps, od1, desc = {}, {}, {}
    for t in TERMS:
        vals = []
        for k in seeds:
            v = [rank.get((k, t, m, "ens_sigma")) for m in SEVEN]
            if None not in v:
                vals.append(stats.ols_slope(np.log([x["r"] for x in v]), [x["auroc_e"] for x in v]))
        reps[f"OB1_{t}"] = {**stats.test_report(vals), "n_seeds": len(vals)}
        v16 = [x for x in (rank.get((k, t, 16.0, "ens_sigma")) for k in seeds) if x is not None]
        diff = [x["spearman_e"] - x["spearman_d"] for x in v16]
        od1[f"OD1_{t}"] = {**stats.tost(diff, 0.10), "n": len(diff)}
        mean_auc = {f"{m:g}": float(np.mean([rank[(k, t, m, "ens_sigma")]["auroc_e"] for k in seeds
                                              if (k, t, m, "ens_sigma") in rank])) for m in (0.0,) + SEVEN}
        mean_r = {f"{m:g}": float(np.exp(np.mean(np.log([rank[(k, t, m, "ens_sigma")]["r"] for k in seeds
                                                          if (k, t, m, "ens_sigma") in rank])))) for m in SEVEN}
        d3 = {}
        for n in (8, 16, 32, 64):
            rows = [r for r in recs if r["term"] == t and r["signal"] == "d3" and r["n_cal"] == n]
            d3[str(n)] = {"aligned": int(sum(r["aligned"] for r in rows)), "n_seeds": len(rows)}
        cov = {}
        for sig, n in (("conf_nom", 0), ("conf_true", 16)):
            rows = [r for r in recs if r["term"] == t and r["mult"] == 16.0 and r["signal"] == sig
                    and r["n_cal"] == n]
            cov[f"{sig}_{n}"] = float(np.mean([r["coverage"] for r in rows])) if rows else None
        desc[t] = {"mean_auroc_ens_sigma": mean_auc, "geo_mean_r": mean_r, "D3_x16": d3, "coverage_x16": cov}
    fam = stats.family(reps, {n: -1 for n in reps})
    al = [json.loads(l) for fn in glob.glob(os.path.join(align_dir, "pend_ood_align_seed*.jsonl"))
          for l in open(fn, encoding="utf-8") if l.strip()]
    al = [r for r in al if r["seed"] in set(seeds)]
    align = {}
    for t in TERMS:
        rs = [r for r in al if r["term"] == t]
        if not rs:
            continue
        o = {"n_seeds": len(rs), "out_of_span_median": float(np.median([r["out_of_span_median"] for r in rs]))}
        for sig in SIGNALS:
            v = [r["signals"][sig] for r in rs]
            w = float(np.mean([x["range"][1] - x["range"][0] for x in v]))
            need = int(np.ceil(0.9 * len(v) - 1e-9))
            o[f"OR1_{sig}"] = {**count([x["OR1_inside"] for x in v], 0.9), "mean_width": w,
                               "reference_p": float(binom.sf(need - 1, len(v), w / 2))}
            o[f"OR1L_{sig}"] = {**count([x["OR1L_ok"] for x in v], 0.9),
                                "max_gap": float(max(x["OR1L_gap"] for x in v)),
                                "mean_rho_D_x16": float(np.mean([x["rho_D_x16"] for x in v])),
                                "mean_rho_linear": float(np.mean([x["rho_linear"] for x in v]))}
            o[f"OR1L4_{sig}"] = {**count([x["OR1L4_ok"] for x in v], 0.9),
                                 "max_gap": float(max(x["OR1L4_gap"] for x in v))}
            o[f"gap_by_level_{sig}"] = {f"{m:g}": {"mean": float(np.mean([x["gap_by_level"][f"{m:g}"] for x in v])),
                                                   "max": float(max(x["gap_by_level"][f"{m:g}"] for x in v))}
                                        for m in SEVEN}
        o["OR3"] = count([r["OR3"]["by_n"]["16"]["beats_sigma"] for r in rs], 0.9)
        o["OR3_selected_n16"] = {n: sum(r["OR3"]["by_n"]["16"]["selected"] == n for r in rs) for n in DICT_TERMS}
        o["OR3_by_n"] = {n: {"beats_sigma": int(sum(r["OR3"]["by_n"][n]["beats_sigma"] for r in rs)),
                             "median_stat": float(np.median([r["OR3"]["by_n"][n]["stat"] for r in rs]))}
                         for n in ("8", "16", "32", "64")}
        o["sigma_stat_median"] = float(np.median([r["OR3"]["sigma_stat"] for r in rs]))
        align[t] = o
    res = {"seeds": seeds, "OB1": fam, "OD1": od1, "alignment": align, "descriptive": desc}
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, default=float)
    return res


def parse_seeds(v):
    out = []
    for tok in v:
        a, _, b = tok.partition("-")
        out += list(range(int(a), int(b) + 1)) if b else [int(a)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["dev", "pilot", "main"], required=True)
    ap.add_argument("--stage", choices=["pilot", "cells", "align", "analyse"], required=True)
    ap.add_argument("--seeds", nargs="+", required=True)
    ap.add_argument("--eps-ref", default=os.path.join("pilot_results", "pend_ood_eps_ref.json"))
    ap.add_argument("--params", default=os.path.join("pilot_results", "frozen_params.json"))
    ap.add_argument("--out-dir", default=os.path.join("runs", "pend_ood_main"))
    ap.add_argument("--align-dir", default=os.path.join("runs", "pend_ood_alignment"))
    ap.add_argument("--analysis-dir", default=os.path.join("runs", "pend_ood_analysis"))
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    seeds = parse_seeds(a.seeds)
    check_phase(seeds, a.phase)
    use_ood_terms()
    so = Pendulum(load_params(a.params)["nominal_cfg"])
    if a.stage == "pilot":
        res = R.stage_eps([so], a.out_dir, seed=seeds[0])
        R.run_cells([so], seeds, res, a.out_dir)
    elif a.stage == "cells":
        R.run_cells([so], seeds, json.load(open(a.eps_ref, encoding="utf-8")), a.out_dir)
    elif a.stage == "align":
        eps = json.load(open(a.eps_ref, encoding="utf-8"))["P"]
        run_align(seeds, a.out_dir, a.align_dir, eps, a.workers)
    else:
        analyse(seeds, a.out_dir, a.align_dir, a.analysis_dir)
        print(f"wrote {os.path.join(a.analysis_dir, 'results.json')}")


if __name__ == "__main__":
    main()

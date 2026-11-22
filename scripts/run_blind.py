"""Runner and analysis for addendum v1.10: blind dictionaries and drawn missing terms.

The drawn terms come from pilot_results/blind_draw.json (scripts/draw_blind_terms.py, seeded by the
hash of tag prereg-v1.10a). Each system runs in its own process (`--system P` or `--system B`),
because the frozen cell runner keeps one term list per process.

Stages:
  pilot     eps_ref per drawn term by bisection on r = 1 at n_nom = 512 -> <out>/eps_ref.json, then cells
  cells     levels 0-16 x eps_ref for the drawn terms, given seeds
  align     per seed: limit inputs, range of the BLIND library at 16 x, linear-response gap at every
            level, selection from the BLIND library at 4 x (n = 8, 16, 32, 64), out-of-span share
  analyse   per drawn term (both systems): slope test (Holm over all drawn terms), TOST limit,
            range containment with width and reference p, linear-response gap at 4 x and 16 x,
            selection counts, descriptive -> results.json

Reuses the frozen v1.7 cell machinery (`scripts.run_pde`) through adapters; no frozen module is
modified. The residual signal is stored under the runner key "pde_resid" (ODE residual on the
pendulum, PDE residual on Burgers).

Each system writes its own cell directory (default runs/blind_main_P, runs/blind_main_B) so two
processes never append to the same file; analyse reads both.

Usage:  python -m scripts.run_blind --phase dev --system P --stage pilot --seeds 999 --out-dir runs/blind_dev
"""

import argparse
import glob
import json
import os
from multiprocessing import Pool

import numpy as np
from scipy.stats import binom

from mismatch import blind as BL
from mismatch import burgers as B
from mismatch import burgers_ood as BO
from mismatch import stats
from mismatch.alignment import alignment_range, response_norm, term_responses
from mismatch.rq1 import debiased_discrepancy
from mismatch.simulators import stream_seed
from scripts import run_pde as R
from scripts.run_alignment import count, rho, select_term
from scripts.run_pde import N_TEST, Burgers, noisy
from scripts.run_pendulum_ood import Pendulum
from scripts.run_pilot import load_params

SEVEN = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
SIGNALS = ("ens_sigma", "pde_resid")
MAIN_SEEDS = set(range(1, 21))
TAG = "prereg-v1.10b"
BRACKET = {"P": (1e-6, 0.1), "B": (1e-5, 0.5)}
DRAW = os.path.join("pilot_results", "blind_draw.json")


def slug(name):
    return name.replace(" ", "_")


def drawn(system):
    d = json.load(open(DRAW, encoding="utf-8"))
    return [slug(t) for t in d["pendulum" if system == "P" else "burgers"]]


def unslug(code):
    return code.replace("_", " ")


class BlindPendulum(Pendulum):
    def sim(self, th, term=None, eps=0.0):
        return BL.pend_batch(th, unslug(term) if term else "constant torque", eps)


class BlindBurgers(Burgers):
    def sim(self, th, term=None, eps=0.0):
        return BL.burg_batch(th, unslug(term) if term else "drift", eps)


def make_system(system, cfg):
    return BlindPendulum(cfg) if system == "P" else BlindBurgers(cfg)


def check_phase(seeds, phase, tag_ok=None):
    seeds = set(seeds)
    if phase == "dev":
        if seeds & (MAIN_SEEDS | {0, 101, 102}):
            raise SystemExit(f"refusing registered seeds {sorted(seeds)} in the dev phase")
        return
    ok = tag_ok if tag_ok is not None else R.tag_exists(TAG)
    if not ok:
        raise SystemExit(f"the v1.10 pilot and main phases need git tag {TAG}")
    if phase == "pilot" and seeds != {0}:
        raise SystemExit("the v1.10 pilot uses seed 0 only")
    if phase == "main" and not seeds <= MAIN_SEEDS:
        raise SystemExit("main seeds are 1-20")


def use_terms(system):
    terms = tuple(drawn(system))
    R.TERMS = terms
    R.BRACKET = {t: BRACKET[system] for t in terms}
    return terms

# ---------------------------------------------------------------- align


def _library(system, thetas):
    if system == "P":
        return term_responses(thetas, terms=BL.PEND_LIB)
    return B.term_responses(thetas, terms=BL.BURG_LIB)


def _response(system, thetas, term):
    return BL.pend_response(thetas, unslug(term)) if system == "P" else BL.burg_response(thetas, unslug(term))


def align_seed(args):
    seed, system, cell_dir, eps_ref = args
    import torch
    torch.set_num_threads(1)
    so = make_system(system, None)
    test = so.sample(N_TEST, stream_seed(seed, "test"))
    names, resp, base = _library(system, test)
    cal = so.sample(64, stream_seed(seed, "calibration", 1))
    _, cresp, cbase = _library(system, cal)
    cal_norms, test_norms = response_norm(cresp, cbase), dict(zip(names, response_norm(resp, base)))
    f_cal = so.sim(cal)
    recs = []
    for term in drawn(system):
        arrs = {m: np.load(os.path.join(cell_dir, "arrays", f"seed{seed}_{so.name}_{term}_x{m:g}.npz"))
                for m in SEVEN}
        L = _response(system, test, term)
        lin = np.linalg.norm(L, axis=1) / np.linalg.norm(base, axis=1)
        span = BO.out_of_span(L, resp)
        rec = {"seed": seed, "system": system, "term": term, "out_of_span_median": float(np.median(span)),
               "out_of_span_q10_q90": [float(np.quantile(span, 0.1)), float(np.quantile(span, 0.9))],
               "signals": {}}
        for sig in SIGNALS:
            s16, D16 = arrs[16.0][sig], arrs[16.0]["D"]
            rng = alignment_range(s16, names, resp, base, n_mix=2000, seed=seed)
            a = rho(s16, D16)
            pred = rho(s16, lin)
            rec["signals"][sig] = {
                "rho_D_x16": a, "range": [rng["min"], rng["max"]],
                "inside": bool(rng["min"] <= a <= rng["max"]),
                "rho_linear": pred,
                "gap_by_level": {f"{m:g}": abs(rho(arrs[m][sig], arrs[m]["D"]) - pred) for m in SEVEN}}
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
        rec["selection"] = {"sigma_stat": sigma_stat, "by_n": sel}
        recs.append(rec)
    return seed, system, recs


def run_align(seeds, system, cell_dir, out_dir, eps_ref, workers):
    os.makedirs(out_dir, exist_ok=True)
    jobs = [(s, system, cell_dir, eps_ref) for s in seeds
            if not os.path.exists(os.path.join(out_dir, f"blind_align_seed{s}_{system}.jsonl"))]
    with Pool(workers) as pool:
        for seed, sysn, recs in pool.imap_unordered(align_seed, jobs):
            with open(os.path.join(out_dir, f"blind_align_seed{seed}_{sysn}.jsonl"), "w", encoding="utf-8") as fh:
                for r in recs:
                    fh.write(json.dumps(r) + "\n")
            print(f"align seed {seed} {sysn} done", flush=True)

# ---------------------------------------------------------------- analyse


def analyse(seeds, cell_dir, align_dir, out_dir, margin=0.10, need_frac=0.9):
    recs = [json.loads(l) for d in cell_dir.split(",") for fn in glob.glob(os.path.join(d, "pde_seed*.jsonl"))
            for l in open(fn, encoding="utf-8") if l.strip()]
    recs = [r for r in recs if r["seed"] in set(seeds)]
    rank = {(r["seed"], r["system"], r["term"], r["mult"], r["signal"]): r for r in recs if "auroc_e" in r}
    al = [json.loads(l) for fn in glob.glob(os.path.join(align_dir, "blind_align_seed*.jsonl"))
          for l in open(fn, encoding="utf-8") if l.strip()]
    al = [r for r in al if r["seed"] in set(seeds)]
    reps, per_term = {}, {}
    for system in ("P", "B"):
        for t in drawn(system):
            key = f"{system}:{t}"
            vals = []
            for k in seeds:
                v = [rank.get((k, "P" if system == "P" else "B", t, m, "ens_sigma")) for m in SEVEN]
                if None not in v:
                    vals.append(stats.ols_slope(np.log([x["r"] for x in v]), [x["auroc_e"] for x in v]))
            if vals:
                reps[key] = {**stats.test_report(vals), "n_seeds": len(vals)}
            o = {}
            v16 = [x for x in (rank.get((k, system, t, 16.0, "ens_sigma")) for k in seeds) if x is not None]
            if v16:
                o["limit"] = {**stats.tost([x["spearman_e"] - x["spearman_d"] for x in v16], margin), "n": len(v16)}
                o["mean_auroc_ens_sigma"] = {f"{m:g}": float(np.mean([rank[(k, system, t, m, "ens_sigma")]["auroc_e"]
                                                                      for k in seeds if (k, system, t, m, "ens_sigma") in rank]))
                                             for m in (0.0,) + SEVEN}
                cov = {}
                for sig, n in (("conf_nom", 0), ("conf_true", 16)):
                    rows = [r for r in recs if r["system"] == system and r["term"] == t and r["mult"] == 16.0
                            and r["signal"] == sig and r["n_cal"] == n]
                    cov[f"{sig}_{n}"] = float(np.mean([r["coverage"] for r in rows])) if rows else None
                o["coverage_x16"] = cov
            rs = [r for r in al if r["system"] == system and r["term"] == t]
            if rs:
                o["n_seeds_align"] = len(rs)
                o["out_of_span_median"] = float(np.median([r["out_of_span_median"] for r in rs]))
                for sig in SIGNALS:
                    v = [r["signals"][sig] for r in rs]
                    w = float(np.mean([x["range"][1] - x["range"][0] for x in v]))
                    need = int(np.ceil(need_frac * len(v) - 1e-9))
                    o[f"range_{sig}"] = {**count([x["inside"] for x in v], need_frac), "mean_width": w,
                                         "reference_p": float(binom.sf(need - 1, len(v), w / 2)),
                                         "mean_rho_D_x16": float(np.mean([x["rho_D_x16"] for x in v])),
                                         "mean_rho_linear": float(np.mean([x["rho_linear"] for x in v]))}
                    for lev in ("4", "16"):
                        g = [x["gap_by_level"][lev] for x in v]
                        o[f"linear_x{lev}_{sig}"] = {**count([gi <= margin for gi in g], need_frac), "max_gap": float(max(g))}
                    o[f"gap_by_level_{sig}"] = {f"{m:g}": float(np.mean([x["gap_by_level"][f"{m:g}"] for x in v])) for m in SEVEN}
                lib = BL.PEND_LIB if system == "P" else BL.BURG_LIB
                o["selection_n16"] = count([r["selection"]["by_n"]["16"]["beats_sigma"] for r in rs], need_frac)
                o["selection_strong_n16"] = count([r["selection"]["by_n"]["16"]["stat"] >= 0.90 for r in rs], need_frac)
                o["selected_n16"] = {n: sum(r["selection"]["by_n"]["16"]["selected"] == n for r in rs) for n in lib}
                o["selection_by_n"] = {n: {"beats_sigma": int(sum(r["selection"]["by_n"][n]["beats_sigma"] for r in rs)),
                                           "median_stat": float(np.median([r["selection"]["by_n"][n]["stat"] for r in rs]))}
                                       for n in ("8", "16", "32", "64")}
                o["sigma_stat_median"] = float(np.median([r["selection"]["sigma_stat"] for r in rs]))
            per_term[key] = o
    fam = stats.family(reps, {n: -1 for n in reps}) if reps else {}
    res = {"seeds": seeds, "draw": json.load(open(DRAW, encoding="utf-8")), "slope_family": fam, "terms": per_term}
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
    ap.add_argument("--system", choices=["P", "B"], default="P")
    ap.add_argument("--seeds", nargs="+", required=True)
    ap.add_argument("--eps-ref", default=os.path.join("pilot_results", "blind_eps_ref.json"))
    ap.add_argument("--params", default=os.path.join("pilot_results", "frozen_params.json"))
    ap.add_argument("--out-dir", default=None,
                    help="cell directory; default runs/blind_main_<system>; for analyse, comma-separated list")
    ap.add_argument("--align-dir", default=os.path.join("runs", "blind_alignment"))
    ap.add_argument("--analysis-dir", default=os.path.join("runs", "blind_analysis"))
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    seeds = parse_seeds(a.seeds)
    check_phase(seeds, a.phase)
    if a.out_dir is None:
        a.out_dir = (",".join(os.path.join("runs", f"blind_main_{x}") for x in "PB") if a.stage == "analyse"
                     else os.path.join("runs", f"blind_main_{a.system}"))
    if a.stage == "analyse":
        analyse(seeds, a.out_dir, a.align_dir, a.analysis_dir)
        print(f"wrote {os.path.join(a.analysis_dir, 'results.json')}")
        return
    use_terms(a.system)
    so = make_system(a.system, load_params(a.params)["nominal_cfg"])
    if a.stage == "pilot":
        res = R.stage_eps([so], a.out_dir, seed=seeds[0])
        R.run_cells([so], seeds, res, a.out_dir)
    elif a.stage == "cells":
        R.run_cells([so], seeds, json.load(open(a.eps_ref, encoding="utf-8")), a.out_dir)
    else:
        eps = json.load(open(a.eps_ref, encoding="utf-8"))[a.system]
        run_align(seeds, a.system, a.out_dir, a.align_dir, eps, a.workers)


if __name__ == "__main__":
    main()

"""Analysis of the PDE systems (addendum v1.7 section 5), seeds 1-20.

Stages:
  align      per (seed, system): term responses of the nominal field on the test points and the
             first 64 calibration points; R1/R1s/R1c/R2 numbers for each term (P1, P2) and signal,
             and R3a/R3b term selection at 4 x. One JSON record per (seed, system, term) in
             <align-dir>/pde_align_seed<k>.jsonl (resumable; seeds in parallel).
  analyse    Family P (PB1, Holm over 4), PA2 against pred-v2, PD1 (TOST), PR1, PR1s, PR3a, PR3b,
             and descriptive D3 counts, coverage and widths. Writes <out>/results.json.

Decision rules are those frozen in prereg_addendum_v1_7.md and section 11 of preregistration.md
(2026-11-20): direction-aware Holm, seed-resampled 4PL bootstrap with the >10% drop rule, k of
20 counts. Helpers are reused from scripts.run_analysis and scripts.run_alignment unchanged.

Usage:  python -m scripts.run_pde_analysis --stage align --workers 6
        python -m scripts.run_pde_analysis --stage analyse
"""

import argparse
import glob
import json
import os
from multiprocessing import Pool

import numpy as np

from mismatch import burgers as B
from mismatch import operator as O
from mismatch import stats
from mismatch.alignment import alignment_range, response_norm
from mismatch.rq1 import debiased_discrepancy
from mismatch.simulators import stream_seed
from scripts.run_alignment import count, rho, select_term
from scripts.run_analysis import boot_mid, fit_summary, gmean, seed_fit
from scripts.run_pde import N_TEST, Burgers, Operator, noisy

SYSTEMS = ("B", "O")
TERMS = ("P1", "P2")
SEVEN = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
R1_MULT, R1S_MULTS, R3_MULT, R3_N = 16.0, (4.0, 16.0), 4.0, (8, 16, 32, 64)
SIGNALS = ("ens_sigma", "pde_resid")
DIR_PDE = os.path.join("runs", "pde_main")


# ---------------------------------------------------------------- term responses

def responses(so, th, scale=1e-3):
    """(names, responses (m, n, T), base (n, T)) of the nominal output to each dictionary term."""
    if so.name == "B":
        return B.term_responses(th, scale=scale)
    x = B.grid()[0]

    def run(term):
        c = np.atleast_2d(th)
        out = np.concatenate([B.integrate(O.profiles(c[i:i + 256], x), np.full(len(c[i:i + 256]), O.NU),
                                          term, n_ox=O.N_OP) for i in range(0, len(c), 256)])
        return out[:, 1:].reshape(len(c), -1)
    base = run(None)
    resp = [(run(lambda u, xx, f=fn: scale * f(u, xx)) - base) / scale for fn in B.TERMS.values()]
    return list(B.TERMS), np.stack(resp), base


def system(name):
    return Burgers(None) if name == "B" else Operator()


def align_seed(args):
    seed, name, pde_dir, eps_ref, n_mix = args
    so = system(name)
    test = so.sample(N_TEST, stream_seed(seed, "test"))
    names, resp, base = responses(so, test)
    cal = so.sample(max(R3_N), stream_seed(seed, "calibration", 1))
    _, cresp, cbase = responses(so, cal)
    cal_norms, test_norms = response_norm(cresp, cbase), dict(zip(names, response_norm(resp, base)))
    f_cal_nom = so.sim(cal)
    out = []
    for term in TERMS:
        arrs = {m: np.load(os.path.join(pde_dir, "arrays", f"seed{seed}_{name}_{term}_x{m:g}.npz"))
                for m in sorted({R1_MULT, R3_MULT, *R1S_MULTS})}
        fam = B.TRUE_FAMILY[term]
        keep = [i for i, n in enumerate(names) if n not in fam]
        rec = {"seed": seed, "system": name, "term": term, "signals": {}}
        for sig in SIGNALS:
            s = arrs[R1_MULT][sig]
            full = alignment_range(s, names, resp, base, n_mix=n_mix, seed=seed)
            red = alignment_range(s, [names[i] for i in keep], resp[keep], base, n_mix=n_mix, seed=seed)
            actual = {f"{m:g}": rho(arrs[m][sig], arrs[m]["D"]) for m in arrs}
            fp = float(np.mean([full["per_term"][n] for n in fam]))
            a16 = actual[f"{R1_MULT:g}"]
            rec["signals"][sig] = {
                "actual_rho_D": actual, "range_full": [full["min"], full["max"]],
                "per_term": full["per_term"], "range_reduced": [red["min"], red["max"]],
                "band_reduced_5_95": [red["mix_quantiles"][0], red["mix_quantiles"][2]],
                "family_predicted": fp, "R1_inside": bool(full["min"] <= a16 <= full["max"]),
                "R1s_gap": {f"{m:g}": abs(actual[f"{m:g}"] - fp) for m in R1S_MULTS},
                "R1c_inside_range": bool(red["min"] <= a16 <= red["max"]),
                "R1c_inside_band": bool(red["mix_quantiles"][0] <= a16 <= red["mix_quantiles"][2]),
                "R2_width_full": full["max"] - full["min"], "R2_width_reduced": red["max"] - red["min"]}
        # R3 at 4 x: same calibration stream and noise rule as the runner's conf_true / D3
        eps = R3_MULT * eps_ref[name][term]
        y, sd = noisy(so.sim(cal, term, eps), cal, seed)
        dhat = debiased_discrepancy(y, f_cal_nom, sd)
        D4 = arrs[R3_MULT]["D"]
        sigma_stat = rho(arrs[R3_MULT]["ens_sigma"], D4)
        r3 = {"ens_sigma_stat": sigma_stat}
        for label, drop in (("R3a", ()), ("R3b", fam)):
            idx = [i for i, n in enumerate(names) if n not in drop]
            per_n = {}
            for n in R3_N:
                chosen, _ = select_term(dhat[:n], [names[i] for i in idx], cal_norms[idx][:, :n])
                st = rho(test_norms[chosen], D4)
                per_n[str(n)] = {"selected": chosen, "stat": st, "beats_sigma": bool(st > sigma_stat),
                                 "at_least_0.90": bool(st >= 0.90)}
            r3[label] = per_n
        rec["R3"] = r3
        out.append(rec)
    return seed, name, out


def _one_thread():
    import torch
    torch.set_num_threads(1)


def run_align(seeds, pde_dir, align_dir, eps_ref, workers, n_mix=2000):
    os.makedirs(align_dir, exist_ok=True)

    def done(seed):
        p = os.path.join(align_dir, f"pde_align_seed{seed}.jsonl")
        return ({r["system"] for r in map(json.loads, open(p, encoding="utf-8"))}
                if os.path.exists(p) else set())
    jobs = [(s, n, pde_dir, eps_ref, n_mix) for s in seeds for n in SYSTEMS if n not in done(s)]
    with Pool(workers, initializer=_one_thread) as pool:
        for seed, name, recs in pool.imap_unordered(align_seed, jobs):
            with open(os.path.join(align_dir, f"pde_align_seed{seed}.jsonl"), "a", encoding="utf-8") as fh:
                for r in recs:
                    fh.write(json.dumps(r) + "\n")
            print(f"align seed {seed} {name} done", flush=True)


# ---------------------------------------------------------------- analysis

def load(pde_dir, seeds):
    recs = [json.loads(l) for fn in glob.glob(os.path.join(pde_dir, "pde_seed*.jsonl"))
            for l in open(fn, encoding="utf-8") if l.strip()]
    recs = [r for r in recs if r["seed"] in seeds]
    rank = {(r["seed"], r["system"], r["term"], r["mult"], r["signal"]): r
            for r in recs if "auroc_e" in r}
    return recs, rank


def family_p(rank, seeds):
    """PB1: per-seed OLS slope of ensemble-sigma AUROC on log r (seven levels); slope < 0."""
    reps = {}
    for s in SYSTEMS:
        for t in TERMS:
            vals, miss = [], []
            for k in seeds:
                v = [rank.get((k, s, t, m, "ens_sigma")) for m in SEVEN]
                if None in v:
                    miss.append(k)
                    continue
                vals.append(stats.ols_slope(np.log([x["r"] for x in v]), [x["auroc_e"] for x in v]))
            reps[f"PB1_{s}_{t}"] = {**stats.test_report(vals), "seeds_missing": miss}
    return stats.family(reps, {n: -1 for n in reps})


def pa2(rank, seeds, pred, n_boot):
    """PA2 (ensemble sigma): pred-v2 r_1/2 inside the 90% seed-bootstrap CI and within 1.5 x.
    The other signals get the seed-mean fit beside their pred-v2 value (descriptive)."""
    out = {}
    for s in SYSTEMS:
        for t in TERMS:
            for sig in ("ens_sigma", "mc_sigma", "pde_resid"):
                rows = []
                for k in seeds:
                    v = [rank.get((k, s, t, m, sig)) for m in SEVEN]
                    if None not in v:
                        rows.append(([x["r"] for x in v], [x["auroc_e"] for x in v]))
                if len(rows) < 3:
                    continue
                xr, ys = np.array([r[0] for r in rows]), np.array([r[1] for r in rows])
                f = seed_fit(xr, ys)
                p = pred["fits"].get(f"{s}_{t}_{sig}", {})
                res = {"fit": fit_summary(f), "mean_auroc": ys.mean(axis=0).tolist(),
                       "gmean_r": [gmean(xr[:, j]) for j in range(xr.shape[1])], "n_seeds": len(rows),
                       "pred": {"r_half": p.get("r_half"), "converged": p.get("converged"),
                                "plateau_large_r": p.get("plateau_large_r")}}
                if sig == "ens_sigma":
                    if not f.get("converged"):
                        res.update(verdict="inconclusive",
                                   reason="seed-mean fit failed or midpoint outside the r range")
                    else:
                        vals, dropped = boot_mid(xr, ys, n_boot)
                        res.update(dropped=dropped, n_boot=n_boot)
                        if vals.size == 0 or dropped > 0.1 * n_boot:
                            res.update(verdict="inconclusive", reason=f"{dropped} of {n_boot} resamples dropped")
                        elif not p.get("converged"):
                            res.update(verdict="inconclusive", reason="pred-v2 fit did not converge")
                        else:
                            lo, hi = (float(np.exp(v)) for v in stats.percentile_ci(vals, 0.90))
                            ph = p["r_half"]
                            inside, within = bool(lo <= ph <= hi), bool(1 / 1.5 <= ph / f["r_half"] <= 1.5)
                            res.update(ci90_r_half=[lo, hi], inside_ci=inside, within_1_5=within,
                                       verdict="success" if inside and within else "fail")
                out[f"{s}_{t}_{sig}"] = res
    return out


def pd1(rank, seeds):
    """PD1: rho_S(sigma, E) - rho_S(sigma, D) at 16 x within +/- 0.10 (TOST across seeds)."""
    out = {}
    for s in SYSTEMS:
        for t in TERMS:
            v = [x for x in (rank.get((k, s, t, 16.0, "ens_sigma")) for k in seeds) if x is not None]
            diff = [x["spearman_e"] - x["spearman_d"] for x in v]
            out[f"PD1_{s}_{t}"] = {**stats.tost(diff, 0.10), "n": len(diff),
                                   "mean_rho_E": float(np.mean([x["spearman_e"] for x in v])),
                                   "mean_rho_D": float(np.mean([x["spearman_d"] for x in v]))}
    return out


def descriptive(recs):
    out = {}
    for s in SYSTEMS:
        for t in TERMS:
            mine = [r for r in recs if r["system"] == s and r["term"] == t]
            d3 = {}
            for n in R3_N:
                rows = [r for r in mine if r["signal"] == "d3" and r["n_cal"] == n and r["mult"] == 16.0]
                k = int(sum(r["aligned"] for r in rows))
                d3[str(n)] = {"aligned": k, "n_seeds": len(rows),
                              "ci95": stats.binom_ci(k, len(rows)) if rows else None}
            cov = {}
            for m in (0.0,) + SEVEN:
                for sig, n in [("ens_sigma_gauss", 0), ("conf_nom", 0)] + [("conf_true", n) for n in R3_N]:
                    rows = [r for r in mine if r["mult"] == m and r["signal"] == sig and r["n_cal"] == n]
                    if rows:
                        cov[f"x{m:g}_{sig}_{n}"] = {"coverage": float(np.mean([r["coverage"] for r in rows])),
                                                    "width": float(np.mean([r["width"] for r in rows]))}
            out[f"{s}_{t}"] = {"D3_x16": d3, "intervals": cov}
    return out


def alignment_summary(align_dir, tol=0.20):
    recs = [json.loads(l) for fn in glob.glob(os.path.join(align_dir, "pde_align_seed*.jsonl"))
            for l in open(fn, encoding="utf-8") if l.strip()]
    out = {}
    for s in SYSTEMS:
        for t in TERMS:
            rs = [r for r in recs if r["system"] == s and r["term"] == t]
            if not rs:
                continue
            o = {"n_seeds": len(rs)}
            for sig in SIGNALS:
                v = [r["signals"][sig] for r in rs]
                o[f"PR1_{sig}"] = count([x["R1_inside"] for x in v], 0.9)
                for m in R1S_MULTS:
                    o[f"PR1s_{sig}_x{m:g}"] = {**count([x["R1s_gap"][f"{m:g}"] <= tol for x in v], 0.9),
                                               "max_gap": max(x["R1s_gap"][f"{m:g}"] for x in v)}
                o[f"R1c_{sig}"] = {"inside_range": int(sum(x["R1c_inside_range"] for x in v)),
                                   "inside_band": int(sum(x["R1c_inside_band"] for x in v)), "n": len(v)}
                o[f"R2_{sig}"] = {"width_full_mean": float(np.mean([x["R2_width_full"] for x in v])),
                                  "width_reduced_mean": float(np.mean([x["R2_width_reduced"] for x in v]))}
                o[f"rho_D_x16_{sig}_mean"] = float(np.mean([x["actual_rho_D"]["16"] for x in v]))
                o[f"family_predicted_{sig}_mean"] = float(np.mean([x["family_predicted"] for x in v]))
                o[f"range_full_{sig}_mean"] = np.mean([x["range_full"] for x in v], axis=0).tolist()
            for label in ("R3a", "R3b"):
                x = [r["R3"][label]["16"] for r in rs]
                o[f"P{label}_beats_sigma"] = count([y["beats_sigma"] for y in x], 0.9)
                if label == "R3a":
                    o["PR3a_at_least_0.90"] = count([y["at_least_0.90"] for y in x], 0.9)
                o[f"{label}_selected_n16"] = {n: sum(y["selected"] == n for y in x) for n in B.TERMS}
                o[f"{label}_by_n"] = {n: {"beats_sigma": int(sum(r["R3"][label][n]["beats_sigma"] for r in rs)),
                                          "median_stat": float(np.median([r["R3"][label][n]["stat"] for r in rs]))}
                                      for n in map(str, R3_N)}
            o["ens_sigma_stat_median"] = float(np.median([r["R3"]["ens_sigma_stat"] for r in rs]))
            out[f"{s}_{t}"] = o
    return out


def analyse(pde_dir, align_dir, seeds, pred_path, out_dir, n_boot):
    os.makedirs(out_dir, exist_ok=True)
    recs, rank = load(pde_dir, set(seeds))
    pred = json.load(open(pred_path, encoding="utf-8"))
    res = {"seeds": seeds, "n_boot": n_boot, "family_P": family_p(rank, seeds),
           "PA2": pa2(rank, seeds, pred, n_boot), "PD1": pd1(rank, seeds),
           "alignment": alignment_summary(align_dir), "descriptive": descriptive(recs)}
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
    ap.add_argument("--stage", choices=["align", "analyse"], required=True)
    ap.add_argument("--seeds", nargs="+", default=["1-20"])
    ap.add_argument("--pde-dir", default=DIR_PDE)
    ap.add_argument("--align-dir", default=os.path.join("runs", "pde_alignment"))
    ap.add_argument("--eps-ref", default=os.path.join("pilot_results", "pde_eps_ref.json"))
    ap.add_argument("--pred", default=os.path.join("predictions", "pred_v2_seed0.json"))
    ap.add_argument("--out-dir", default=os.path.join("runs", "pde_analysis"))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--n-mix", type=int, default=2000)
    ap.add_argument("--n-boot", type=int, default=stats.N_BOOT)
    a = ap.parse_args()
    seeds = parse_seeds(a.seeds)
    if a.stage == "align":
        run_align(seeds, a.pde_dir, a.align_dir, json.load(open(a.eps_ref, encoding="utf-8")),
                  a.workers, a.n_mix)
    else:
        analyse(a.pde_dir, a.align_dir, seeds, a.pred, a.out_dir, a.n_boot)
        print(f"wrote {os.path.join(a.out_dir, 'results.json')}")


if __name__ == "__main__":
    main()

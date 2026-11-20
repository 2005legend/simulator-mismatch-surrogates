"""Main analysis of the frozen tests (prereg-v1 section 8; addenda v1.1, v1.2, v1.5, v1.6; the
implementation rules of the 2026-11-20 entry in section 11 of preregistration.md).

Reads RQ1 records and arrays and the acquisition JSONL files; fits nothing new. Writes
<out>/results.json. Pieces whose inputs are missing are reported as `missing`, never guessed.

Usage:  python -m scripts.run_analysis --seeds 1-20 --out-dir runs/analysis
        python -m scripts.run_analysis --rq1-dir runs/rq1_pilot --acq-dir runs/pilot/acq_seeds \
            --seeds 0 101 102 --n-boot 200 --out-dir runs/analysis_dev     (debugging only)
"""

import argparse
import glob
import json
import os

import numpy as np

from mismatch import stats
from mismatch.model_a import fit_4pl
from mismatch.rq1 import auroc

FROZEN = (0.25, 0.5, 1.0, 2.0)
SEVEN = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
NNOM = (128, 512, 2048)
SYSTEMS = ("S1", "S2")
WINDOWS = (2, 4, 6, 8, 10)
LOG15 = float(np.log(1.5))
ARM_A = {1: "random", 2: "sobol", 3: "nominal_sigma", 4: "nominal_residual", 5: "disc_sigma",
         6: "gp_variance", 7: "observed_residual", 8: "r_gated"}


# ---------------------------------------------------------------- loading

def load_jsonl(pattern):
    out = []
    for fn in sorted(glob.glob(pattern)):
        out += [json.loads(l) for l in open(fn, encoding="utf-8") if l.strip()]
    return out


class RQ1:
    def __init__(self, d, seeds):
        recs = [r for r in load_jsonl(os.path.join(d, "rq1_seed*.jsonl")) if r["seed"] in seeds]
        self.rank = {(r["seed"], r["n_nom"], r["system"], r["mult"], r["signal"]): r
                     for r in recs if "auroc_e" in r and r["n_cal"] == 0}
        self.cell = {(r["seed"], r["n_nom"], r["system"], r["mult"]): r
                     for r in recs if r["signal"] == "cell_summary"}
        self.d3 = [r for r in recs if r["signal"] == "d3"]
        self.dir, self.seeds = d, sorted(seeds)

    def get(self, seed, n, s, m, sig="ens_sigma", key="auroc_e"):
        r = self.rank.get((seed, n, s, m, sig))
        return None if r is None else r[key]

    def r(self, seed, n, s, m):
        return self.get(seed, n, s, m, key="r")

    def arrays(self, seed, n, s, m):
        p = os.path.join(self.dir, "arrays", f"seed{seed}_n{n}_{s}_x{m:g}.npz")
        return np.load(p) if os.path.exists(p) else None


def load_acq(d, family, seeds):
    recs = load_jsonl(os.path.join(d, f"acq{family}_seed*.jsonl"))
    return {(r["seed"], r["system"], r["mult"], r["name"]): r for r in recs if r["seed"] in seeds}


# ---------------------------------------------------------------- helpers

def complete_seeds(seeds, fn):
    """Seeds for which fn(seed) returns no None (a missing cell drops the seed and is counted)."""
    keep = [s for s in seeds if fn(s) is not None]
    return keep, [s for s in seeds if s not in keep]


def gmean(v):
    return float(np.exp(np.mean(np.log(v))))


def seed_fit(xr, ys, idx=None):
    """4PL of mean y on geometric-mean r, per cell, over the seeds in idx (rows)."""
    xr, ys = np.asarray(xr), np.asarray(ys)
    if idx is not None:
        xr, ys = xr[idx], ys[idx]
    return fit_4pl([gmean(xr[:, j]) for j in range(xr.shape[1])], ys.mean(axis=0).tolist())


def boot_mid(xr, ys, n_boot, extra=None):
    """Bootstrap log r_1/2 over seeds; extra = second (xr, ys) resampled with the same indices,
    returning the paired difference of midpoints. Dropped resamples are counted."""
    n = np.asarray(xr).shape[0]
    vals, dropped = [], 0
    for idx in stats.boot_indices(n, n_boot):
        f = seed_fit(xr, ys, idx)
        if extra is not None:
            g = seed_fit(*extra, idx)
            ok = f.get("converged") and g.get("converged")
            v = np.log(f["r_half"]) - np.log(g["r_half"]) if ok else None
        else:
            v = np.log(f["r_half"]) if f.get("converged") else None
        if v is None:
            dropped += 1
        else:
            vals.append(float(v))
    return np.array(vals), dropped


def fit_summary(f):
    keep = ("converged", "r_half", "slope", "plateau_small_r", "plateau_large_r")
    return {k: f.get(k) for k in keep}


# ---------------------------------------------------------------- Family B and P-A

def b1(R):
    out = {}
    for s in SYSTEMS:
        cells = [(n, m) for n in NNOM for m in FROZEN]
        def slope(seed):
            v = [(R.r(seed, n, s, m), R.get(seed, n, s, m)) for n, m in cells]
            if any(a is None or b is None for a, b in v):
                return None
            return stats.ols_slope(np.log([a for a, _ in v]), [b for _, b in v])
        keep, miss = complete_seeds(R.seeds, slope)
        out[f"B1_{s}"] = ({**stats.test_report([slope(k) for k in keep]), "seeds_missing": miss}
                          if len(keep) >= 3 else {"missing": True, "seeds_missing": miss})
    return out


def rq2_g(A, seed, s, m, arm):
    a, b = A.get((seed, s, m, arm)), A.get((seed, s, m, "sobol"))
    return None if a is None or b is None else float(np.log(a["budget_auc"] / b["budget_auc"]))


def b2(R, A, n_boot):
    out = {}
    for s in SYSTEMS:
        cells = [(n, m) for n in NNOM for m in FROZEN]
        def row(seed):
            ra = [R.r(seed, n, s, m) for n, m in cells]
            ya = [R.get(seed, n, s, m) for n, m in cells]
            rb = [R.r(seed, 512, s, m) for m in FROZEN]
            yb = [rq2_g(A, seed, s, m, "nominal_sigma") for m in FROZEN]
            return None if None in ra + ya + rb + yb else (ra, ya, rb, yb)
        keep, miss = complete_seeds(R.seeds, row)
        if len(keep) < 3:
            out[f"B2_{s}"] = {"missing": True, "seeds_missing": miss}
            continue
        rows = [row(k) for k in keep]
        xa, ya, xb, yb = (np.array([r[i] for r in rows]) for i in range(4))
        fa, fb = seed_fit(xa, ya), seed_fit(xb, yb)
        res = {"fit_rq1": fit_summary(fa), "fit_rq2": fit_summary(fb), "seeds_missing": miss,
               "n_seeds": len(keep)}
        if not (fa.get("converged") and fb.get("converged")):
            res.update(verdict="inconclusive", reason="a point fit did not converge or its midpoint "
                       "is outside the sampled r range", p=1.0)
        else:
            diff = float(np.log(fa["r_half"]) - np.log(fb["r_half"]))
            vals, dropped = boot_mid(xa, ya, n_boot, extra=(xb, yb))
            res.update(diff_log_rhalf=diff, dropped=dropped, n_boot=n_boot)
            if dropped > 0.1 * n_boot or vals.size == 0:
                res.update(verdict="inconclusive", reason=f"{dropped} of {n_boot} resamples dropped",
                           p=1.0)
            else:
                p = max(float(np.mean(vals <= -LOG15)), float(np.mean(vals >= LOG15)))
                res.update(ci90=stats.percentile_ci(vals, 0.90), p=p)
        out[f"B2_{s}"] = res
    return out


def family_b(B1, B2):
    names = [k for k in list(B1) + list(B2)]
    reps = {**B1, **B2}
    ok = [n for n in names if not reps[n].get("missing") and "p" in reps[n]]
    adj = dict(zip(ok, stats.holm([reps[n]["p"] for n in ok])))
    out = {}
    for n in names:
        r = dict(reps[n])
        if n in adj:
            r["p_holm"] = adj[n]
            if n.startswith("B1"):
                lo, hi = r["ci95_bca"]
                r["supported"] = bool(adj[n] < 0.05 and hi < 0)
                r["opposite"] = bool(adj[n] < 0.05 and lo > 0)
            elif r.get("verdict") != "inconclusive":
                lo, hi = r["ci90"]
                r["verdict"] = ("equivalent" if adj[n] < 0.05 and -LOG15 < lo and hi < LOG15
                                else "not equivalent")
        out[n] = r
    return out


def p_a(R, n_boot, pred_path):
    pred = json.load(open(pred_path, encoding="utf-8")) if os.path.exists(pred_path) else None
    out = {}
    for s in SYSTEMS:
        out[s] = {}
        for sig in ("ens_sigma", "mc_sigma", "ode_resid"):
            def row(seed):
                rr = [R.r(seed, 512, s, m) for m in SEVEN]
                yy = [R.get(seed, 512, s, m, sig) for m in SEVEN]
                return None if None in rr + yy else (rr, yy)
            keep, miss = complete_seeds(R.seeds, row)
            if len(keep) < 3:
                out[s][sig] = {"missing": True}
                continue
            xr = np.array([row(k)[0] for k in keep])
            ys = np.array([row(k)[1] for k in keep])
            f = seed_fit(xr, ys)
            res = {"fit": fit_summary(f), "mean_auroc": ys.mean(axis=0).tolist(),
                   "gmean_r": [gmean(xr[:, j]) for j in range(xr.shape[1])],
                   "seeds_missing": miss}
            if f.get("converged"):
                vals, dropped = boot_mid(xr, ys, n_boot)
                res.update(dropped=dropped, n_boot=n_boot)
                if vals.size and dropped <= 0.1 * n_boot:
                    lo, hi = stats.percentile_ci(vals, 0.90)
                    res["ci90_r_half"] = (float(np.exp(lo)), float(np.exp(hi)))
            if pred is not None:
                ex = pred["systems"][s]["exact"][sig]
                for variant in ("empirical", "gaussian"):
                    pf = ex[variant]["fit_measured_levels"]
                    if not pf.get("converged"):
                        res[f"pred_{variant}"] = {"converged": False}
                        continue
                    ph = pf["r_half"]
                    sc = {"r_half": ph, "slope": pf["slope"]}
                    if f.get("converged") and "ci90_r_half" in res:
                        lo, hi = res["ci90_r_half"]
                        sc["inside_ci"] = bool(lo <= ph <= hi)
                        sc["within_1_5"] = bool(1 / 1.5 <= ph / f["r_half"] <= 1.5)
                        sc["success"] = sc["inside_ci"] and sc["within_1_5"]
                    res[f"pred_{variant}"] = sc
            out[s][sig] = res
    return out


# ---------------------------------------------------------------- v1.5 D-tests

def d_tests(R):
    out = {}
    for s in SYSTEMS:
        def diff(seed):
            e = R.get(seed, 512, s, 16.0, key="spearman_e")
            d = R.get(seed, 512, s, 16.0, key="spearman_d")
            return None if e is None or d is None else e - d
        keep, miss = complete_seeds(R.seeds, diff)
        if len(keep) >= 3:
            out[f"D1_{s}"] = {**stats.tost([diff(k) for k in keep], 0.10), "seeds_missing": miss,
                              "values": [diff(k) for k in keep]}

        def slope(seed):
            v = [(R.r(seed, 512, s, m), R.get(seed, 512, s, m)) for m in SEVEN]
            if any(a is None or b is None for a, b in v):
                return None
            return stats.ols_slope(np.log([a for a, _ in v]), [b for _, b in v])
        keep, miss = complete_seeds(R.seeds, slope)
        if len(keep) >= 3:
            rep = stats.test_report([slope(k) for k in keep])
            want = +1 if s == "S1" else -1
            lo, hi = rep["ci95_bca"]
            rep.update(predicted_sign=want, p_ok=rep["p"] < 0.05,
                       success=bool(rep["p"] < 0.05 and (lo > 0 if want > 0 else hi < 0)),
                       seeds_missing=miss)
            out[f"D2_{s}"] = rep

        d3 = {}
        for n in (8, 16, 32, 64):
            rows = [r for r in R.d3 if r["system"] == s and r["mult"] == 16.0 and r["n_cal"] == n
                    and r["n_nom"] == 512 and r["signal"] == "d3"]
            k, N = sum(r["aligned"] for r in rows), len(rows)
            d3[str(n)] = {"aligned": k, "n_seeds": N,
                          "ci95": stats.binom_ci(k, N) if N else None}
        x = d3["32"]
        frac = (x["aligned"] / x["n_seeds"]) if x["n_seeds"] else None
        d3["prediction"] = ">= 8 of 10 (>= 16 of 20)" if s == "S1" else "<= 2 of 10 (<= 4 of 20)"
        d3["success"] = None if frac is None else bool(frac >= 0.8 if s == "S1" else frac <= 0.2)
        out[f"D3_{s}"] = d3

        d4 = {}
        for n in NNOM:
            for m in (0.0,) + SEVEN:
                cs = [R.cell.get((k, n, s, m)) for k in R.seeds]
                cs = [c for c in cs if c is not None]
                if not cs:
                    continue
                g = lambda key: [c[key] for c in cs if c.get(key) is not None]
                rho = lambda key: [R.get(k, n, s, m, key=key) for k in R.seeds
                                   if R.get(k, n, s, m, key=key) is not None]
                e, a, dd = rho("spearman_e"), rho("spearman_a"), rho("spearman_d")
                def fin(v):
                    v = [x for x in v if x is not None and np.isfinite(x)]
                    return float(np.mean(v)) if v else None
                d4[f"n{n}_x{m:g}"] = {
                    "kappa": fin(g("kappa")), "cv_d": fin(g("cv_d")), "r": fin(g("r")),
                    "rho_E": fin(e), "rho_A": fin(a), "rho_D": fin(dd),
                    "dist_to_A_limit": None if fin(e) is None or fin(a) is None else abs(fin(e) - fin(a)),
                    "dist_to_D_limit": None if fin(e) is None or fin(dd) is None else abs(fin(e) - fin(dd)),
                    "n_seeds": len(cs)}
        out[f"D4_{s}"] = d4
    return out


# ---------------------------------------------------------------- Family A, C1, Family F

def family_a(R, A):
    reps, dirs, miss_all = {}, {}, {}
    for s in SYSTEMS:
        for test, arm in (("A1", "nominal_sigma"), ("A2", "nominal_residual")):
            def slope(seed):
                v = [(R.r(seed, 512, s, m), rq2_g(A, seed, s, m, arm)) for m in FROZEN]
                if any(a is None or b is None for a, b in v):
                    return None
                return stats.ols_slope(np.log([a for a, _ in v]), [b for _, b in v])
            keep, miss = complete_seeds(R.seeds, slope)
            if len(keep) >= 3:
                reps[f"{test}_{s}"], dirs[f"{test}_{s}"] = stats.test_report([slope(k) for k in keep]), +1
                miss_all[f"{test}_{s}"] = miss

        def a3(seed):
            x, y = A.get((seed, s, 2.0, "observed_residual")), A.get((seed, s, 2.0, "gp_variance"))
            return None if x is None or y is None else x["budget_auc"] - y["budget_auc"]
        keep, miss = complete_seeds(R.seeds, a3)
        if len(keep) >= 3:
            reps[f"A3_{s}"], dirs[f"A3_{s}"] = stats.test_report([a3(k) for k in keep]), -1
            miss_all[f"A3_{s}"] = miss
    out = stats.family(reps, dirs) if reps else {}
    for k in out:
        out[k]["seeds_missing"] = miss_all[k]
    return out


def check_r_reuse(R, A):
    """Family A's uncorrected error at eps = 0 must equal RQ1's median ||a|| (same surrogate)."""
    worst = 0.0
    n = 0
    for (seed, s, m, name), rec in A.items():
        if m == 0.0 and name == "sobol":
            c = R.cell.get((seed, 512, s, 0.0))
            if c is not None:
                worst = max(worst, abs(rec["uncorrected_median"] - c["median_a"]) / c["median_a"])
                n += 1
    assert n == 0 or worst < 1e-6, f"Family A and RQ1 surrogates differ (rel diff {worst:.2e})"
    return {"cells_checked": n, "max_rel_diff": worst}


def c1(A, seeds):
    out = {}
    mults = (0.0,) + FROZEN
    for s in SYSTEMS:
        keep = [k for k in seeds if all(A.get((k, s, m, ARM_A[j])) for m in mults for j in range(1, 9))]
        if len(keep) < 3:
            out[f"C1_{s}"] = {"missing": True}
            continue
        auc = {(k, m, j): A[(k, s, m, ARM_A[j])]["budget_auc"] for k in keep for m in mults
               for j in range(1, 9)}
        best = {m: min(range(1, 8), key=lambda j: np.mean([auc[(k, m, j)] for k in keep]))
                for m in mults}
        regret = {j: [np.mean([auc[(k, m, j)] - auc[(k, m, best[m])] for m in mults]) for k in keep]
                  for j in range(1, 9)}
        fixed = min(range(1, 8), key=lambda j: np.mean(regret[j]))
        diff = np.array(regret[8]) - np.array(regret[fixed])
        rep = stats.family({"C1": stats.test_report(diff)}, {"C1": -1})["C1"]
        rep.update(best_fixed_arm=ARM_A[fixed], mean_regret={ARM_A[j]: float(np.mean(regret[j]))
                                                            for j in regret})
        out[f"C1_{s}"] = rep
    return out


def family_f(F, seeds, pilot_cv_max):
    reps, dirs = {}, {}
    for test, arm, s in (("F1", "term_disagreement", "S1"), ("F2", "term_disagreement", "S2"),
                         ("F3", "state_coverage", "S1"), ("F4", "state_coverage", "S2")):
        g = [rq2_g(F, k, s, 4.0, arm) for k in seeds]
        g = [v for v in g if v is not None]
        if len(g) >= 3:
            reps[test], dirs[test] = stats.test_report(g), -1
    out = stats.family(reps, dirs) if reps else {}
    if len(out) == 4:
        any_sup = any(v["supported"] for v in out.values())
        out["stop_rule"] = ("an arm beats Sobol" if any_sup else
                            "no arm beats Sobol; " + ("points chosen do not matter (CV < 10%)"
                                                      if pilot_cv_max < 0.10 else
                                                      f"inconclusive at this fit budget (pilot CV max {pilot_cv_max:.2f})"))
    return out


# ---------------------------------------------------------------- P-C (v1.2 G1)

def p_c(R, pred_path):
    pred = json.load(open(pred_path, encoding="utf-8")) if os.path.exists(pred_path) else None
    out = {}
    for s in SYSTEMS:
        table = {}
        for w in WINDOWS:
            per_m = {}
            for m in (0.0,) + SEVEN:
                aucs, rs = [], []
                for k in R.seeds:
                    z = R.arrays(k, 512, s, m)
                    if z is None or f"E_{w}" not in z:
                        continue
                    aucs.append(auroc(z[f"ens_sigma_{w}"], z[f"E_{w}"]))
                    if m > 0:
                        rs.append(np.median(z[f"D_{w}"]) / np.median(z[f"A_{w}"]))
                if aucs:
                    per_m[f"{m:g}"] = {"auroc": float(np.mean(aucs)), "n_seeds": len(aucs),
                                       "r": gmean(rs) if rs else 0.0}
            if not per_m:
                continue
            lv = [f"{m:g}" for m in SEVEN if f"{m:g}" in per_m]
            f = fit_4pl([per_m[k]["r"] for k in lv], [per_m[k]["auroc"] for k in lv]) if len(lv) >= 4 else {}
            plateau = f["plateau_large_r"] if f.get("converged") else per_m.get("16", {}).get("auroc")
            table[str(w)] = {"levels": per_m, "plateau_large_r": plateau,
                             "auroc_eps0": per_m.get("0", {}).get("auroc")}
        if not table:
            out[s] = {"missing": True}
            continue
        horizon = {}
        for m in FROZEN:
            h = "> 10 s"
            for w in WINDOWS:
                t = table.get(str(w))
                if t and t["auroc_eps0"] is not None and t["plateau_large_r"] is not None \
                        and f"{m:g}" in t["levels"] \
                        and t["levels"][f"{m:g}"]["auroc"] < 0.5 * (t["auroc_eps0"] + t["plateau_large_r"]):
                    h = str(w)
                    break
            horizon[f"{m:g}"] = h
        res = {"windows": table, "observed_horizon": horizon}
        if pred is not None:
            ph = pred["systems"][s]["P_C"]["blindness_horizon"]
            idx = lambda h: len(WINDOWS) if h == "> 10 s" else WINDOWS.index(int(float(h)))
            close = [abs(idx(horizon[k]) - idx(ph[k])) <= 1 for k in horizon]
            res.update(predicted_horizon=ph, levels_equal_or_adjacent=int(sum(close)),
                       success=None if s == "S1" else bool(sum(close) >= 3),
                       note="not applicable on S1 (section 11, 2026-11-20)" if s == "S1" else "")
        out[s] = res
    return out


# ---------------------------------------------------------------- collapse (descriptive)

def collapse(R):
    out = {}
    for s in SYSTEMS:
        out[s] = {}
        for n in NNOM:
            pts = {}
            for m in FROZEN:
                a = [R.get(k, n, s, m) for k in R.seeds]
                r = [R.r(k, n, s, m) for k in R.seeds]
                ok = [(x, y) for x, y in zip(r, a) if x is not None and y is not None]
                if ok:
                    pts[f"{m:g}"] = {"r": gmean([x for x, _ in ok]),
                                     "auroc": float(np.mean([y for _, y in ok])), "n_seeds": len(ok)}
            out[s][str(n)] = pts
    return out


# ---------------------------------------------------------------- main

def parse_seeds(v):
    out = []
    for tok in v:
        if "-" in tok:
            a, b = map(int, tok.split("-"))
            out += list(range(a, b + 1))
        else:
            out.append(int(tok))
    return out


def run(rq1_dir, acq_dir, seeds, out_dir, n_boot, pred_path, pilot_params):
    os.makedirs(out_dir, exist_ok=True)
    R = RQ1(rq1_dir, set(seeds))
    A = load_acq(acq_dir, "A", set(seeds))
    F = load_acq(acq_dir, "F", set(seeds))
    fcv = json.load(open(pilot_params, encoding="utf-8")) if os.path.exists(pilot_params) else None
    cv_max = (max(v["cv"] for s in fcv["fit_to_fit"].values() for v in s.values()) if fcv else 1.0)
    B1 = b1(R)
    res = {"seeds": sorted(seeds), "n_boot": n_boot,
           "r_reuse_check": check_r_reuse(R, A),
           "family_B": family_b(B1, b2(R, A, n_boot)),
           "P_A": p_a(R, n_boot, pred_path),
           "D": d_tests(R),
           "family_A": family_a(R, A),
           "family_C": c1(A, sorted(seeds)),
           "family_F": family_f(F, sorted(seeds), cv_max),
           "P_C": p_c(R, pred_path),
           "collapse": collapse(R)}
    with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, default=float)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rq1-dir", default=os.path.join("runs", "rq1_main"))
    ap.add_argument("--acq-dir", default=os.path.join("runs", "acq_main"))
    ap.add_argument("--seeds", nargs="+", default=["1-20"])
    ap.add_argument("--n-boot", type=int, default=stats.N_BOOT)
    ap.add_argument("--pred", default=os.path.join("predictions", "model_a_seed0.json"))
    ap.add_argument("--pilot-fcv", default=os.path.join("pilot_results", "stage_fcv.json"))
    ap.add_argument("--out-dir", default=os.path.join("runs", "analysis"))
    a = ap.parse_args()
    run(a.rq1_dir, a.acq_dir, parse_seeds(a.seeds), a.out_dir, a.n_boot, a.pred, a.pilot_fcv)
    print(f"wrote {os.path.join(a.out_dir, 'results.json')}")


if __name__ == "__main__":
    main()

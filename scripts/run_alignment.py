"""Alignment-range tests of addendum v1.6 section 5: R1, R1s, R1c, R2 and R3a/R3b.

Simulator work only; no network is fitted. Per seed and system it reads the saved RQ1 arrays
(n_nom = 512: the signals and the true discrepancy norm D on the 500 test points), computes the
linear response of the nominal trajectory to each dictionary term on the test points and on the
first 64 calibration points (the same stream and oracle as RQ1, so the same noisy observations),
and writes one JSON record per (seed, system) to <out>/alignment_seed<k>.jsonl. `--summarise`
applies the frozen criteria (k of 10 read as 2k of 20; section 11, 2026-11-20).

Usage:  python -m scripts.run_alignment --seeds 1-20 --rq1-dir runs/rq1_main --out-dir runs/alignment
        python -m scripts.run_alignment --summarise --out-dir runs/alignment
"""

import argparse
import glob
import json
import os

import numpy as np
import torch
from scipy.stats import spearmanr

from mismatch import stats
from mismatch.acquisition import make_oracle
from mismatch.alignment import TERMS, alignment_range, response_norm, term_responses
from mismatch.rq1 import debiased_discrepancy
from mismatch.simulators import simulate_batch, sobol_thetas, stream_seed
from scripts.run_pilot import load_params

SIGNALS = ("ens_sigma", "ode_resid")
TRUE_FAMILY = {"S1": ("w|w|", "w^3"), "S2": ("tanh(w/0.1)", "tanh(w/0.5)")}
R1_MULT, R1S_MULTS, R3_MULT = 16.0, (4.0, 16.0), 4.0
R3_N = (8, 16, 32, 64)
N_TEST = 500


def rho(a, b):
    return float(spearmanr(a, b)[0])


def subset(names, responses, drop):
    keep = [i for i, n in enumerate(names) if n not in drop]
    return [names[i] for i in keep], responses[keep]


def select_term(dhat, names, cal_norms):
    """R3: the dictionary term whose response norm at the calibration points correlates most
    with the debiased observed discrepancy norm."""
    with np.errstate(all="ignore"):
        c = {n: rho(dhat, v) for n, v in zip(names, cal_norms)}
    c = {n: (v if np.isfinite(v) else -np.inf) for n, v in c.items()}
    return max(c, key=c.get), c


def seed_record(seed, system, rq1_dir, eps_ref, n_mix=2000):
    arrs = {}
    for m in sorted({R1_MULT, R3_MULT, *R1S_MULTS}):
        p = os.path.join(rq1_dir, "arrays", f"seed{seed}_n512_{system}_x{m:g}.npz")
        if not os.path.exists(p):
            return None
        arrs[m] = np.load(p)
    test = sobol_thetas(N_TEST, seed=stream_seed(seed, "test"))
    names, resp, base = term_responses(test)
    fam = TRUE_FAMILY[system]
    names_red, resp_red = subset(names, resp, fam)
    rec = {"seed": seed, "system": system, "signals": {}}

    for sig in SIGNALS:
        s = arrs[R1_MULT][sig]
        full = alignment_range(s, names, resp, base, n_mix=n_mix, seed=seed)
        red = alignment_range(s, names_red, resp_red, base, n_mix=n_mix, seed=seed)
        actual = {f"{m:g}": rho(arrs[m][sig], arrs[m]["D"]) for m in arrs}
        fam_pred = float(np.mean([full["per_term"][n] for n in fam]))
        a16 = actual[f"{R1_MULT:g}"]
        rec["signals"][sig] = {
            "actual_rho_D": actual,
            "range_full": [full["min"], full["max"]], "per_term": full["per_term"],
            "range_reduced": [red["min"], red["max"]],
            "band_reduced_5_95": [red["mix_quantiles"][0], red["mix_quantiles"][2]],
            "family_predicted": fam_pred,
            "R1_inside": bool(full["min"] <= a16 <= full["max"]),
            "R1s_gap": {f"{m:g}": abs(actual[f"{m:g}"] - fam_pred) for m in R1S_MULTS},
            "R1c_inside_range": bool(red["min"] <= a16 <= red["max"]),
            "R1c_inside_band": bool(red["mix_quantiles"][0] <= a16 <= red["mix_quantiles"][2]),
            "R2_width_full": full["max"] - full["min"], "R2_width_reduced": red["max"] - red["min"]}

    # R3: hypothesis selection from the first n real trajectories at 4 x eps_ref
    eps = R3_MULT * eps_ref[system]
    cal = sobol_thetas(max(R3_N), seed=stream_seed(seed, "calibration", 1))
    y, s_obs = make_oracle(system, eps, seed=seed)(cal)
    dhat = debiased_discrepancy(y, simulate_batch(cal, "S1", 0.0), s_obs)
    _, cal_resp, cal_base = term_responses(cal)
    cal_norms = response_norm(cal_resp, cal_base)
    test_norms = dict(zip(names, response_norm(resp, base)))
    D4 = arrs[R3_MULT]["D"]
    sigma_stat = rho(arrs[R3_MULT]["ens_sigma"], D4)
    r3 = {"ens_sigma_stat": sigma_stat}
    for label, drop in (("R3a", ()), ("R3b", fam)):
        idx = [i for i, n in enumerate(names) if n not in drop]
        sub = [names[i] for i in idx]
        per_n = {}
        for n in R3_N:
            chosen, corr = select_term(dhat[:n], sub, cal_norms[idx][:, :n])
            stat = rho(test_norms[chosen], D4)
            per_n[str(n)] = {"selected": chosen, "stat": stat, "beats_sigma": bool(stat > sigma_stat),
                             "at_least_0.90": bool(stat >= 0.90)}
        r3[label] = per_n
    rec["R3"] = r3
    return rec


def done_keys(path):
    if not os.path.exists(path):
        return set()
    return {(r["seed"], r["system"]) for r in map(json.loads, open(path, encoding="utf-8"))}


def run(seeds, systems, rq1_dir, out_dir, eps_ref, n_mix=2000):
    torch.set_num_threads(1)
    os.makedirs(out_dir, exist_ok=True)
    for seed in seeds:
        path = os.path.join(out_dir, f"alignment_seed{seed}.jsonl")
        done = done_keys(path)
        for system in systems:
            if (seed, system) in done:
                continue
            rec = seed_record(seed, system, rq1_dir, eps_ref, n_mix)
            if rec is None:
                print(f"seed {seed} {system}: arrays missing, skipped", flush=True)
                continue
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + "\n")
            print(f"seed {seed} {system} done", flush=True)


def count(flags, need_frac):
    k, n = int(sum(flags)), len(flags)
    return {"k": k, "n": n, "ci95": stats.binom_ci(k, n) if n else None,
            "required": f">= {int(np.ceil(need_frac * n - 1e-9))} of {n}",
            "met": bool(n and k >= np.ceil(need_frac * n - 1e-9))}


def summarise(out_dir, tol=0.20):
    recs = [json.loads(l) for fn in sorted(glob.glob(os.path.join(out_dir, "alignment_seed*.jsonl")))
            for l in open(fn, encoding="utf-8") if l.strip()]
    out = {}
    for system in TRUE_FAMILY:
        rs = [r for r in recs if r["system"] == system]
        if not rs:
            continue
        o = {"n_seeds": len(rs)}
        for sig in SIGNALS:
            v = [r["signals"][sig] for r in rs]
            o[f"R1_{sig}"] = count([x["R1_inside"] for x in v], 0.9)
            for m in R1S_MULTS:
                o[f"R1s_{sig}_x{m:g}"] = {**count([x["R1s_gap"][f"{m:g}"] <= tol for x in v], 0.9),
                                          "max_gap": max(x["R1s_gap"][f"{m:g}"] for x in v)}
            o[f"R1c_{sig}"] = {"inside_range": int(sum(x["R1c_inside_range"] for x in v)),
                               "inside_band": int(sum(x["R1c_inside_band"] for x in v)), "n": len(v)}
            o[f"R2_{sig}"] = {"width_full_mean": float(np.mean([x["R2_width_full"] for x in v])),
                              "width_reduced_mean": float(np.mean([x["R2_width_reduced"] for x in v]))}
        for label in ("R3a", "R3b"):
            x = [r["R3"][label]["16"] for r in rs]
            o[f"{label}_beats_sigma"] = count([y["beats_sigma"] for y in x], 0.9)
            if label == "R3a":
                o["R3a_at_least_0.90"] = count([y["at_least_0.90"] for y in x], 0.9)
            o[f"{label}_selected_n16"] = {t: sum(y["selected"] == t for y in x) for t in TERMS}
            o[f"{label}_by_n"] = {n: {"beats_sigma": int(sum(r["R3"][label][n]["beats_sigma"] for r in rs)),
                                      "median_stat": float(np.median([r["R3"][label][n]["stat"] for r in rs]))}
                                  for n in map(str, R3_N)}
        out[system] = o
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    return out


def parse_seeds(v):
    out = []
    for tok in v:
        a, _, b = tok.partition("-")
        out += list(range(int(a), int(b) + 1)) if b else [int(a)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", default=["1-20"])
    ap.add_argument("--systems", nargs="+", default=list(TRUE_FAMILY))
    ap.add_argument("--rq1-dir", default=os.path.join("runs", "rq1_main"))
    ap.add_argument("--params", default=os.path.join("pilot_results", "frozen_params.json"))
    ap.add_argument("--out-dir", default=os.path.join("runs", "alignment"))
    ap.add_argument("--n-mix", type=int, default=2000)
    ap.add_argument("--summarise", action="store_true")
    a = ap.parse_args()
    if not a.summarise:
        run(parse_seeds(a.seeds), a.systems, a.rq1_dir, a.out_dir, load_params(a.params)["eps_ref"],
            a.n_mix)
    print(json.dumps(summarise(a.out_dir), indent=1)[:4000])


if __name__ == "__main__":
    main()

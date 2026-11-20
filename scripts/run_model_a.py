"""Compute the frozen predictions P-A (v1.1 section 2, exact-d per v1.4) and P-C (v1.2 section 1)
from pilot seed 0 only, and write them to <out>/model_a_seed0.json.

Inputs: the eps = 0 surrogates of seed 0 (nominal ensemble, MC-dropout model) refitted exactly as
`scripts/run_rq1.py` fits them (checked against the pilot's saved arrays), and simulator
discrepancies d(eps) on a log grid of eps. No surrogate trained or scored at eps > 0 is used.

Usage:  python -m scripts.run_model_a --params pilot_results/frozen_params.json --out-dir runs/model_a
"""

import argparse
import json
import os
import time

import numpy as np
import torch

from mismatch import config
from mismatch.model_a import (blindness_horizon, closed_form_midpoint, cosines, empirical_auroc,
                              fit_4pl, gaussian_auroc, gaussian_inputs, rel_norms, use_fallback)
from mismatch.rq1 import WINDOWS, auroc
from mismatch.signals import residual_profile, windowed_residual_score
from mismatch.simulators import simulate_batch, sobol_thetas, stream_seed
from mismatch.surrogate import Ensemble, FitConfig
from scripts.run_pilot import load_params

SEED = 0
SYSTEMS = ("S1", "S2")
SIGNALS = ("ens_sigma", "mc_sigma", "ode_resid")
SWEEP = (0.25, 0.5, 1.0, 2.0)                    # the frozen nonzero levels (prereg-v1)
EXTRA = (4.0, 8.0, 16.0)                         # extra levels (v1.5), reported too
GRID = tuple(float(2.0 ** k) for k in np.arange(-8, 6.01, 0.5))   # eps / eps_ref, 1/256 to 64
COLLAPSE_NNOM = (128, 512, 2048)
nominal_sim = lambda th: simulate_batch(th, "S1", 0.0)


def window_k(h):
    return int(round(h / (config.T_GRID[1] - config.T_GRID[0]))) + 1


def fit_models(n_nom, cfg, seed=SEED, members=5, mc_cfg=None):
    train = sobol_thetas(n_nom, seed=stream_seed(seed, "train"))
    y = nominal_sim(train)
    nominal = Ensemble(n_members=members, cfg=cfg, base_seed=stream_seed(seed, "nominal")).fit(train, y)
    mc = Ensemble(n_members=1, cfg=mc_cfg or FitConfig(dropout=0.1),
                  base_seed=stream_seed(seed, "mc")).fit(train, y)
    return nominal, mc


def surrogate_side(nominal, mc, test, f_nom, k=None, seed=SEED, n_mc=50):
    """a and the signals on the first k time points (all points if k is None); MC dropout is
    skipped when mc is None."""
    mean, sigma = nominal.predict(test)
    k = k or mean.shape[1]
    a = (mean - f_nom)[:, :k]
    sig = {"ens_sigma": sigma[:, :k].mean(axis=1),
           "ode_resid": windowed_residual_score(residual_profile(mean, test), k)}
    if mc is not None:
        mcs = mc.members[0].predict_mc(test, n_passes=n_mc, seed=seed).std(axis=0)
        sig["mc_sigma"] = mcs[:, :k].mean(axis=1)
    return a, sig


def curve(signals, a, d_by_mult, f_true_by_mult, grid, n_draws):
    """Predicted AUROC on the grid for both variants of Model A, per signal."""
    out = {s: {"r": [], "gaussian": [], "empirical": []} for s in signals}
    for m in grid:
        d, f_true = d_by_mult[m], f_true_by_mult[m]
        A, D = rel_norms(a, f_true), rel_norms(d, f_true)
        r = float(np.median(D) / np.median(A))
        for s, sig in signals.items():
            out[s]["r"].append(r)
            out[s]["gaussian"].append(gaussian_auroc(gaussian_inputs(sig, A, D), r, n_draws=n_draws))
            out[s]["empirical"].append(empirical_auroc(sig, a, d, f_true))
    return out


def summarise(c, A0, signals, grid, levels):
    """4PL fits, level predictions and the eps = 0 AUROC for one curve set."""
    res = {}
    for s, sig in signals.items():
        entry = {"auroc_eps0": auroc(sig, A0), "grid": c[s],
                 "r_at_mult": {f"{m:g}": c[s]["r"][grid.index(m)] for m in levels}}
        for variant in ("gaussian", "empirical"):
            at = {f"{m:g}": c[s][variant][grid.index(m)] for m in levels}
            entry[variant] = {"fit": fit_4pl(c[s]["r"], c[s][variant]), "auroc_at_mult": at,
                              # same seven levels the main-run fit will use (like for like)
                              "fit_measured_levels": fit_4pl([entry["r_at_mult"][k] for k in at],
                                                             list(at.values()))}
        res[s] = entry
    return res


def run(params, out_dir, n_draws=10 ** 6, collapse=True, grid=GRID, n_test=500, members=5,
        collapse_nnom=COLLAPSE_NNOM, mc_cfg=None, n_mc=50, seed=SEED):
    torch.set_num_threads(1)
    os.makedirs(out_dir, exist_ok=True)
    grid = tuple(grid)
    levels = tuple(m for m in SWEEP + EXTRA if m in grid)
    cfg, eps_ref = params["nominal_cfg"], params["eps_ref"]
    t0 = time.perf_counter()
    nominal, mc = fit_models(512 if 512 in collapse_nnom else collapse_nnom[0], cfg, seed, members, mc_cfg)
    test = sobol_thetas(n_test, seed=stream_seed(seed, "test"))
    f_nom = nominal_sim(test)
    a, signals = surrogate_side(nominal, mc, test, f_nom, seed=seed, n_mc=n_mc)
    result = {"seed": seed, "eps_ref": eps_ref, "grid_mults": list(grid), "n_draws": n_draws,
              "systems": {}}

    pilot = os.path.join("runs", "rq1_pilot", "arrays", f"seed{seed}_n512_S1_x0.npz")
    if os.path.exists(pilot) and n_test == 500:     # the refit must reproduce the pilot's arrays
        z = np.load(pilot)
        result["reproduces_pilot_max_rel_diff"] = {
            "A": float(np.max(np.abs(z["A"] - rel_norms(a, f_nom)) / z["A"])),
            **{s: float(np.max(np.abs(z[s] - signals[s]) / np.abs(z[s]))) for s in SIGNALS}}

    for system in SYSTEMS:
        f_true = {m: simulate_batch(test, system, m * eps_ref[system]) for m in grid}
        d = {m: f_nom - f_true[m] for m in grid}
        d_lin = {m: m * d[1.0] for m in grid}            # linear variant, d(eps) = eps g
        A1, D1 = rel_norms(a, f_true[1.0]), rel_norms(d[1.0], f_true[1.0])
        inp_ref = {s: gaussian_inputs(sig, A1, D1) for s, sig in signals.items()}
        cos = cosines(a, d[1.0])
        sysres = {"inputs_at_eps_ref": inp_ref, "cosine": cos,
                  "fallback": {s: use_fallback(inp_ref[s], cos) for s in signals},
                  "closed_form_r_half": closed_form_midpoint(inp_ref["ens_sigma"]["s_A"],
                                                             inp_ref["ens_sigma"]["s_D"]),
                  "linearity_error": {f"{m:g}": float(np.median(
                      np.linalg.norm(d[m] - d_lin[m], axis=1) / np.linalg.norm(d[m], axis=1)))
                      for m in levels}}
        A0 = rel_norms(a, f_nom)
        for name, dd in (("exact", d), ("linear", d_lin)):
            sysres[name] = summarise(curve(signals, a, dd, f_true, grid, n_draws), A0, signals,
                                     grid, levels)
        # P-C: nominal sigma per window, exact d, variant chosen by the window's own fallback
        pc = {}
        for h in WINDOWS:
            k = window_k(h)
            aw, sw = surrogate_side(nominal, None, test, f_nom, k)
            sw = {"ens_sigma": sw["ens_sigma"]}
            dw = {m: d[m][:, :k] for m in grid}
            fw = {m: f_true[m][:, :k] for m in grid}
            fb = use_fallback(gaussian_inputs(sw["ens_sigma"], rel_norms(aw, fw[1.0]),
                                              rel_norms(dw[1.0], fw[1.0])), cosines(aw, dw[1.0]))
            c = curve(sw, aw, dw, fw, grid, n_draws)["ens_sigma"]
            variant = "empirical" if fb else "gaussian"
            fit = fit_4pl(c["r"], c[variant])
            pc[f"{h:g}"] = {"fallback": fb, "variant": variant,
                            "auroc_eps0": auroc(sw["ens_sigma"], rel_norms(aw, f_nom[:, :k])),
                            "plateau_large_r": fit["plateau_large_r"] if fit.get("converged") else c[variant][-1],
                            "fit": fit, "r": c["r"], "auroc_grid": c[variant],
                            "auroc_at_mult": {f"{m:g}": c[variant][grid.index(m)]
                                              for m in SWEEP if m in grid}}
        horizons = {f"{m:g}": blindness_horizon(
            list(pc), {w: pc[w]["auroc_at_mult"][f"{m:g}"] for w in pc},
            {w: pc[w]["auroc_eps0"] for w in pc}, {w: pc[w]["plateau_large_r"] for w in pc})
            for m in SWEEP if m in grid}
        k6 = window_k(6)
        late = np.sum(d[1.0][:, k6:] ** 2, axis=1) / np.sum(d[1.0] ** 2, axis=1)
        sysres["P_C"] = {"windows": pc, "blindness_horizon": horizons,
                         "fraction_d2_after_6s_median": float(np.median(late))}
        result["systems"][system] = sysres
        print(f"{system} done at {time.perf_counter() - t0:.0f} s", flush=True)

    if collapse:                                     # eps = 0 inputs across n_nom (v1.1)
        col = {}
        for n_nom in collapse_nnom:
            nm, mm = (nominal, mc) if n_nom == 512 else fit_models(n_nom, cfg, seed, members, mc_cfg)
            an, sn = surrogate_side(nm, mm, test, f_nom, seed=seed, n_mc=n_mc)
            A0n = rel_norms(an, f_nom)
            col[str(n_nom)] = {system: {s: gaussian_inputs(v, A0n, rel_norms(
                f_nom - simulate_batch(test, system, eps_ref[system]), f_nom)) for s, v in sn.items()}
                for system in SYSTEMS}
            print(f"collapse n_nom {n_nom} done at {time.perf_counter() - t0:.0f} s", flush=True)
        ref = "512" if "512" in col else str(collapse_nnom[0])
        verdict = {}
        for system in SYSTEMS:
            verdict[system] = {}
            for s in SIGNALS:
                vals = {q: [col[str(n)][system][s][q] for n in collapse_nnom]
                        for q in ("s_A", "rho_sa", "rho_ag")}
                spread = {q: float((max(v) - min(v)) / abs(col[ref][system][s][q]))
                          for q, v in vals.items()}
                verdict[system][s] = {"relative_spread": spread,
                                      "collapse_predicted": all(x < 0.10 for x in spread.values())}
        result["collapse"] = {"inputs": col, "verdict": verdict}
    result["seconds"] = round(time.perf_counter() - t0, 1)
    with open(os.path.join(out_dir, f"model_a_seed{seed}.json"), "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=1)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default=os.path.join("pilot_results", "frozen_params.json"))
    ap.add_argument("--out-dir", default=os.path.join("runs", "model_a"))
    args = ap.parse_args()
    run(load_params(args.params), args.out_dir)


if __name__ == "__main__":
    main()

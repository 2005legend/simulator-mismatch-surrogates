"""Render predictions_v1.md from predictions/model_a_seed0.json (output of scripts/run_model_a.py).

Usage:  python -m scripts.write_predictions
"""

import json
import subprocess

LV = ["0.25", "0.5", "1", "2", "4", "8", "16"]
SWEEP = ["0.25", "0.5", "1", "2"]


def f(x):
    return f"{x:.3f}"


def rh(fit):
    return f(fit["r_half"]) if fit.get("converged") else "no fit"


def sl(fit):
    return f"{fit['slope']:.2f}" if fit.get("converged") else "-"


def render(R, head):
    L = ["# Frozen predictions v1 (`pred-v1`): Blind to the Missing Term", ""]
    L.append(
        f"Computed 2026-11-20 from **pilot seed 0 only** (code at commit `{head}` plus the Model A "
        "files committed with this document), before any seed 1-20 run. Status: **frozen 2026-11-20** by the "
        "commit tagged `pred-v1`; changes after this go in section 11 of `preregistration.md`. Inputs: the eps = 0 nominal ensemble and "
        "MC-dropout surrogates of seed 0 (n_nom = 512, settings from `pilot_results/frozen_params.json`), "
        "refitted exactly as the RQ1 runner fits them (they reproduce the pilot's saved arrays: largest "
        "relative difference 1.5e-12), and simulator discrepancies d(eps) on 29 log-spaced levels from "
        "eps_ref/256 to 64 eps_ref. No surrogate trained at eps > 0 is used. Raw output: "
        "`predictions/model_a_seed0.json`. Code: `mismatch/model_a.py`, `scripts/run_model_a.py`, "
        "`scripts/write_predictions.py`.")
    L += ["", "Main seeds are 1-20 (the pilot's seed-count rule; changelog 2026-11-20), so every "
          "'seeds 1-10' in v1.1 and v1.2 reads 'seeds 1-20'.", ""]

    L += ["## 1. Fallback decision (v1.1 section 2, frozen rule)", "",
          "| System | s_A | s_D | median cos(a, d) | cos at median-norm point | Fallback |",
          "|---|---|---|---|---|---|"]
    for s, v in R["systems"].items():
        i, c = v["inputs_at_eps_ref"]["ens_sigma"], v["cosine"]
        L.append(f"| {s} | {f(i['s_A'])} | {f(i['s_D'])} | {f(c['median_cos'])} | "
                 f"{f(c['median_norm_point_cos'])} | **yes** (s_A > 1) |")
    L += ["", "The fallback fires on both systems and for all three signals, so **the prediction is the "
          "empirical plug-in**: the seed-0 arrays are recombined at each eps, e_i = a_i + d_i(eps), and "
          "AUROC is computed directly. The Gaussian-copula variant is reported as secondary.", "",
          "**What this means for the test (stated before any main run).** The nominal surrogate is "
          "never retrained at eps > 0, so the plug-in AUROC at each eps equals the RQ1 AUROC that pilot "
          "seed 0 itself shows (checked: identical to four decimals at every level). Under the fallback, "
          "P-A predicts the main seeds' transition from one seed's arrays: it tests whether the transition "
          "is a stable property of the setup across seeds, not whether the Gaussian model is right. The "
          "Gaussian variant is the model-based prediction and is scored alongside it as secondary.", ""]

    L += ["## 2. P-A, primary: nominal ensemble sigma, exact d, empirical plug-in", "",
          "Success (v1.1): the predicted r_1/2 lies inside the 90% bootstrap CI of r_1/2 from seeds 1-20 "
          "(10,000 resamples) and within a factor of 1.5 of its point estimate. Positive slope = AUROC "
          "falls with r; negative = it rises (the aligned case).", "",
          "| System | r_1/2, fit on measured levels **(primary)** | r_1/2, fit on full grid | "
          "slope (levels / grid) | AUROC plateau small r / large r (grid fit) | AUROC at eps = 0 |",
          "|---|---|---|---|---|---|"]
    for s, v in R["systems"].items():
        e = v["exact"]["ens_sigma"]
        g, l = e["empirical"]["fit"], e["empirical"]["fit_measured_levels"]
        L.append(f"| {s} | **{rh(l)}** | {rh(g)} | {sl(l)} / {sl(g)} | {f(g['plateau_small_r'])} / "
                 f"{f(g['plateau_large_r'])} | {f(e['auroc_eps0'])} |")
    L += ["", "Predicted AUROC (event ||e|| > 80th percentile) and r at each level (eps as a multiple of "
          "eps_ref):", "", "| System | quantity | " + " | ".join(LV) + " |",
          "|---|---|" + "---|" * len(LV)]
    for s, v in R["systems"].items():
        e = v["exact"]["ens_sigma"]
        L.append(f"| {s} | r | " + " | ".join(f(e["r_at_mult"][k]) for k in LV) + " |")
        L.append(f"| {s} | AUROC | " + " | ".join(f(e["empirical"]["auroc_at_mult"][k]) for k in LV) + " |")
    L += ["", "**Decision A (author, 2026-11-20, before any main run).** The main-run r_1/2 is fitted to the "
          "seven measured levels (r about 0.25 to 13). The primary prediction fits the predicted curve "
          "to the same seven levels, so both sides are computed the same way. The full-grid fit, read "
          "literally from v1.1 ('its 4PL midpoint'), is kept beside it. They differ by about 10%.", ""]

    L += ["## 3. P-A, secondary", "",
          "| System | signal | variant | r_1/2 (levels) | slope (levels) | r_1/2 (grid) |",
          "|---|---|---|---|---|---|"]
    for s, v in R["systems"].items():
        for sig in ("ens_sigma", "mc_sigma", "ode_resid"):
            for var, dv in (("exact", "empirical"), ("exact", "gaussian"), ("linear", "empirical")):
                if sig == "ens_sigma" and (var, dv) == ("exact", "empirical"):
                    continue
                e = v[var][sig][dv]
                L.append(f"| {s} | {sig} | {var} d, {dv} | {rh(e['fit_measured_levels'])} | "
                         f"{sl(e['fit_measured_levels'])} | {rh(e['fit'])} |")
    L += ["", "The Gaussian variant misses the S1 shape (it peaks near 1 x eps_ref and then falls, and its "
          "seven-level fit gives an unstable slope), the failure the fallback rule anticipates when s_A > 1.",
          "", "Closed form, no alignment (v1.1), r_1/2 = 3^(1/4) (s_A/s_D)^(1/2): " +
          ", ".join(f"{s} {f(v['closed_form_r_half'])}" for s, v in R["systems"].items()) +
          ". It ignores the measured alignment (S1 rho_sg = "
          f"{f(R['systems']['S1']['inputs_at_eps_ref']['ens_sigma']['rho_sg'])}) and is for reference only.",
          "", "Linearity error of d(eps) = eps g (median ||d - eps g|| / ||d||, v1.4):", "",
          "| System | " + " | ".join(LV) + " |", "|---|" + "---|" * len(LV)]
    for s, v in R["systems"].items():
        L.append(f"| {s} | " + " | ".join(f(v["linearity_error"][k]) for k in LV) + " |")

    L += ["", "## 4. Collapse across n_nom (v1.1)", "",
          "Rule: the curves for n_nom in {128, 512, 2048} collapse onto one curve in r if and only if s_A, "
          "rho_sa and rho_ag at eps = 0 differ by under 10% across n_nom (relative spread = (max - min) / "
          "|value at 512|).", "", "| System | signal | s_A | rho_sa | rho_ag | collapse predicted |",
          "|---|---|---|---|---|---|"]
    for s, vv in R["collapse"]["verdict"].items():
        for sig, x in vv.items():
            sp = x["relative_spread"]
            L.append(f"| {s} | {sig} | {f(sp['s_A'])} | {f(sp['rho_sa'])} | {f(sp['rho_ag'])} | "
                     f"{'yes' if x['collapse_predicted'] else '**no**'} |")
    L += ["", "Collapse is **not predicted** for any system or signal: s_A is stable (6% spread) but the "
          "correlations change with n_nom. Under prereg-v1 the extra n_nom cells are S1 only, at the level "
          "whose pilot r is nearest 1 (1 x eps_ref).", ""]

    L += ["## 5. P-C: blindness horizon (v1.2 section 1, secondary)", "",
          "Model A per prefix window (the fallback fires in every window, so the plug-in is used), nominal "
          "ensemble sigma, exact d. t* = smallest window whose AUROC is below the midpoint between that "
          "window's eps = 0 AUROC and its large-r plateau.", ""]
    for s, v in R["systems"].items():
        pc = v["P_C"]
        L += [f"**{s}** (median fraction of ||d||^2 after 6 s at eps_ref: "
              f"{f(pc['fraction_d2_after_6s_median'])}).", "",
              "| window (s) | AUROC eps = 0 | large-r plateau | 0.25 | 0.5 | 1 | 2 |",
              "|---|---|---|---|---|---|---|"]
        for w, x in pc["windows"].items():
            L.append(f"| {w} | {f(x['auroc_eps0'])} | {f(x['plateau_large_r'])} | " +
                     " | ".join(f(x["auroc_at_mult"][k]) for k in SWEEP) + " |")
        hz = ", ".join(f"{k} x: {h}" + ("" if h == "> 10 s" else " s")
                       for k, h in pc["blindness_horizon"].items())
        L += ["", f"Predicted t*: {hz}.", ""]
    L += ["The directional statement holds in the simulator: more of ||d|| arrives after 6 s for S2 than "
          "for S1.", "",
          "**Decision B (author, 2026-11-20, before any main run).** The frozen t* definition assumes AUROC falls "
          "towards its large-r plateau. On S1 it rises (the aligned case: the plateau is above 0.92 in every "
          "window), so read literally t* = 2 s at every level, which only says AUROC has not yet risen "
          "halfway, not that the signal goes blind. The literal value stays in the table, and the P-C test "
          "is reported as **not applicable on S1** because its large-r plateau lies above its eps = 0 "
          "AUROC in every window; S2 is scored as frozen. On S2 t* is '> 10 s' at all "
          "four levels: the curve has not reached its midpoint by 2 x eps_ref.", ""]

    L += ["## 6. Not predicted (cut, logged in v1.6)", "",
          "P-B (Model B), the SPI bounds (C2, C3) and the 5% noise analysis were cut before any pilot run "
          "(addendum v1.6, `preregistration.md` section 11) and have no entry here.", ""]
    return "\n".join(L)


def main():
    R = json.load(open("predictions/model_a_seed0.json", encoding="utf-8"))
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                          text=True).stdout.strip()
    open("predictions_v1.md", "w", encoding="utf-8").write(render(R, head))


if __name__ == "__main__":
    main()

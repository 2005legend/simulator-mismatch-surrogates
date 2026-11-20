"""Paper figures from the analysis outputs (no new fits). Static print figures (PDF + PNG).

  fig1_transition   nominal-sigma AUROC vs r at n_nom = 512, seed mean with 95% band over seeds,
                    against the frozen P-A prediction (seed-0 plug-in); one panel per system
  fig2_collapse     the same for n_nom = 128, 512, 2048 at the frozen levels (v1.1 collapse check)
  fig3_alignment    alignment range per system and signal, with the actual correlation per seed
                    and the true family's simulator-only prediction (R1, R1s)
  fig4_selection    R3: correlation of the selected term with the true discrepancy norm vs n,
                    with and without the true family, against ensemble sigma
  fig5_acquisition  Family A: G = log(budget-AUC arm / Sobol) vs r for arms 3 and 4

Palette: the validated reference categorical slots 1-3 (blue, orange, aqua; all-pairs CVD pass);
aqua is below 3:1 contrast, so every series also has its own marker and a direct label.

Usage:  python -m scripts.make_figures --rq1-dir runs/rq1_main --acq-dir runs/acq_main \
            --alignment-dir runs/alignment --out-dir figures
"""

import argparse
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from scripts.run_analysis import FROZEN, NNOM, SEVEN, SYSTEMS, RQ1, load_acq, parse_seeds, rq2_g  # noqa: E402

C = ["#2a78d6", "#eb6834", "#1baf7a"]
MARK = ["o", "s", "^"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SYS_LABEL = {"S1": "S1: quadratic drag", "S2": "S2: Coulomb friction"}

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "legend.fontsize": 8,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "lines.linewidth": 2, "lines.markersize": 5,
    "figure.facecolor": "white", "savefig.bbox": "tight", "savefig.dpi": 200})


def save(fig, out, name):
    os.makedirs(out, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"))
    plt.close(fig)


def seed_curve(R, s, n, mults, sig="ens_sigma"):
    """Geometric-mean r, mean AUROC and 95% t-interval half-width over seeds, per level."""
    rs, ms, hs = [], [], []
    for m in mults:
        pts = [(R.r(k, n, s, m), R.get(k, n, s, m, sig)) for k in R.seeds]
        pts = [(a, b) for a, b in pts if a is not None and b is not None]
        if len(pts) < 2:
            continue
        r = np.array([a for a, _ in pts])
        y = np.array([b for _, b in pts])
        rs.append(float(np.exp(np.mean(np.log(r)))))
        ms.append(float(y.mean()))
        hs.append(float(1.96 * y.std(ddof=1) / np.sqrt(y.size)))
    return np.array(rs), np.array(ms), np.array(hs)


def label_end(ax, x, y, text):
    ax.annotate(text, (x[-1], y[-1]), xytext=(4, 0), textcoords="offset points", va="center",
                fontsize=8, color=INK)


def fig1(R, pred, out):
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.6), sharey=True)
    for ax, s in zip(axes, SYSTEMS):
        r, m, h = seed_curve(R, s, 512, SEVEN)
        if r.size:
            ax.fill_between(r, m - h, m + h, color=C[0], alpha=0.18, linewidth=0)
            ax.plot(r, m, color=C[0], marker=MARK[0], label=f"observed, mean of {len(R.seeds)} seeds")
        if pred:
            g = pred["systems"][s]["exact"]["ens_sigma"]
            ax.plot(g["grid"]["r"], g["grid"]["empirical"], color=C[1], linestyle="--", linewidth=1.5,
                    label="predicted (pilot seed 0)")
        ax.axhline(0.5, color=INK2, linewidth=0.8, linestyle=":")
        ax.set_xscale("log")
        ax.set_xlim(0.1, 20)
        ax.set_title(SYS_LABEL[s], loc="left")
        ax.set_xlabel("r = median ||d|| / median ||a||")
    axes[0].set_ylabel("AUROC, nominal ensemble σ")
    axes[0].legend(frameon=False, loc="lower left")
    save(fig, out, "fig1_transition")


def fig2(R, out):
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.6), sharey=True)
    for ax, s in zip(axes, SYSTEMS):
        for i, n in enumerate(NNOM):
            r, m, h = seed_curve(R, s, n, FROZEN)
            if r.size:
                ax.errorbar(r, m, yerr=h, color=C[i], marker=MARK[i], capsize=2, linewidth=1.5,
                            label=f"n_nom = {n}")
                label_end(ax, r, m, str(n))
        ax.set_xscale("log")
        ax.set_title(SYS_LABEL[s], loc="left")
        ax.set_xlabel("r")
    axes[0].set_ylabel("AUROC, nominal ensemble σ")
    axes[0].legend(frameon=False, loc="best")
    save(fig, out, "fig2_collapse")


def load_alignment(d):
    return [json.loads(l) for fn in sorted(glob.glob(os.path.join(d, "alignment_seed*.jsonl")))
            for l in open(fn, encoding="utf-8") if l.strip()]


def fig3(recs, out):
    if not recs:
        return
    fig, ax = plt.subplots(figsize=(4.6, 2.8))
    rows = [(s, sig) for s in SYSTEMS for sig in ("ens_sigma", "ode_resid")]
    names = {"ens_sigma": "ensemble σ", "ode_resid": "ODE residual"}
    for i, (s, sig) in enumerate(rows):
        v = [r["signals"][sig] for r in recs if r["system"] == s]
        if not v:
            continue
        lo = np.mean([x["range_full"][0] for x in v])
        hi = np.mean([x["range_full"][1] for x in v])
        ax.plot([lo, hi], [i, i], color=GRID, linewidth=7, solid_capstyle="round", zorder=1)
        act = [x["actual_rho_D"]["16"] for x in v]
        jit = np.random.default_rng(i).uniform(-0.12, 0.12, len(act))
        ax.scatter(act, np.full(len(act), i) + jit, s=14, color=C[0], marker=MARK[0], zorder=3,
                   label="actual, 16 × ε_ref" if i == 0 else None)
        ax.scatter([np.mean([x["family_predicted"] for x in v])], [i], s=60, color=C[1], marker="|",
                   linewidths=2.5, zorder=4, label="true-family prediction" if i == 0 else None)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{s}, {names[g]}" for s, g in rows])
    ax.set_xlim(-1, 1)
    ax.axvline(0, color=INK2, linewidth=0.8, linestyle=":")
    ax.set_xlabel("Spearman correlation with ||d||  (grey bar: alignment range)")
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)
    save(fig, out, "fig3_alignment")


def fig4(recs, out):
    if not recs:
        return
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.5), sharey=True)
    ns = ["8", "16", "32", "64"]
    for ax, s in zip(axes, SYSTEMS):
        rs = [r for r in recs if r["system"] == s]
        if not rs:
            continue
        for i, (lab, txt) in enumerate((("R3a", "selected, full dictionary"),
                                        ("R3b", "selected, true family removed"))):
            y = np.array([[r["R3"][lab][n]["stat"] for n in ns] for r in rs])
            ax.plot([int(n) for n in ns], np.median(y, axis=0), color=C[i], marker=MARK[i], label=txt)
        sig = np.median([r["R3"]["ens_sigma_stat"] for r in rs])
        ax.axhline(sig, color=C[2], linestyle="--", linewidth=1.5, label="ensemble σ")
        ax.set_xscale("log", base=2)
        ax.set_xticks([8, 16, 32, 64], ["8", "16", "32", "64"])
        ax.set_title(SYS_LABEL[s], loc="left")
        ax.set_xlabel("real trajectories n")
    axes[0].set_ylabel("Spearman with ||d||, median over seeds")
    axes[0].legend(frameon=False, loc="lower right")
    save(fig, out, "fig4_selection")


def fig5(R, A, out):
    if not A:
        return
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.5), sharey=True)
    for ax, s in zip(axes, SYSTEMS):
        for i, (arm, txt) in enumerate((("nominal_sigma", "arm 3: ensemble σ"),
                                        ("nominal_residual", "arm 4: ODE residual"))):
            xs, ms, hs = [], [], []
            for m in FROZEN:
                pts = [(R.r(k, 512, s, m), rq2_g(A, k, s, m, arm)) for k in R.seeds]
                pts = [(a, b) for a, b in pts if a is not None and b is not None]
                if len(pts) < 2:
                    continue
                g = np.array([b for _, b in pts])
                xs.append(np.exp(np.mean(np.log([a for a, _ in pts]))))
                ms.append(g.mean())
                hs.append(1.96 * g.std(ddof=1) / np.sqrt(g.size))
            if xs:
                ax.errorbar(xs, ms, yerr=hs, color=C[i], marker=MARK[i], capsize=2, linewidth=1.5,
                            label=txt)
        ax.axhline(0, color=INK2, linewidth=0.8, linestyle=":")
        ax.set_xscale("log")
        ax.xaxis.set_minor_formatter(plt.NullFormatter())
        ax.set_xticks([0.25, 0.5, 1, 2])
        ax.set_xticklabels(["0.25", "0.5", "1", "2"])
        ax.set_title(SYS_LABEL[s], loc="left")
        ax.set_xlabel("r")
    axes[0].set_ylabel("G = log(AUC arm / AUC Sobol)")
    axes[0].legend(frameon=False, loc="best")
    save(fig, out, "fig5_acquisition")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rq1-dir", default=os.path.join("runs", "rq1_main"))
    ap.add_argument("--acq-dir", default=os.path.join("runs", "acq_main"))
    ap.add_argument("--alignment-dir", default=os.path.join("runs", "alignment"))
    ap.add_argument("--pred", default=os.path.join("predictions", "model_a_seed0.json"))
    ap.add_argument("--seeds", nargs="+", default=["1-20"])
    ap.add_argument("--out-dir", default="figures")
    a = ap.parse_args()
    seeds = set(parse_seeds(a.seeds))
    R = RQ1(a.rq1_dir, seeds)
    A = load_acq(a.acq_dir, "A", seeds)
    pred = json.load(open(a.pred, encoding="utf-8")) if os.path.exists(a.pred) else None
    al = [r for r in load_alignment(a.alignment_dir) if r["seed"] in seeds]
    fig1(R, pred, a.out_dir)
    fig2(R, a.out_dir)
    fig3(al, a.out_dir)
    fig4(al, a.out_dir)
    fig5(R, A, a.out_dir)
    print(f"figures written to {a.out_dir}")


if __name__ == "__main__":
    main()

"""Paper figures for the PDE systems (addendum v1.7), in the style of scripts/make_figures.py.

fig6_pde_transition  AUROC of nominal ensemble sigma vs r, mean of seeds 1-20 with 95% t-interval,
                     against the pred-v2 curve of pilot seed 0, for Burgers and the operator, P1/P2.
fig7_pde_alignment   rho_S(signal, ||d||) at 16 x per seed, the dictionary alignment range and the
                     true-family prediction, for ensemble sigma and the PDE residual.

Usage:  python -m scripts.make_pde_figures
"""

import argparse
import glob
import json
import os

import numpy as np

from scripts.make_figures import C, GRID, INK2, MARK, plt, save
from scripts.run_pde_analysis import SEVEN, SYSTEMS, TERMS, load

LABEL = {"B": "Burgers (MLP)", "O": "Operator (FNO)"}
TERM = {"P1": "P1: −u|u|", "P2": "P2: sin 3x"}


def curve(rank, seeds, s, t, sig="ens_sigma"):
    rs, ms, hs = [], [], []
    for m in SEVEN:
        pts = [rank.get((k, s, t, m, sig)) for k in seeds]
        pts = [p for p in pts if p is not None]
        r = np.array([p["r"] for p in pts])
        y = np.array([p["auroc_e"] for p in pts])
        rs.append(float(np.exp(np.mean(np.log(r)))))
        ms.append(float(y.mean()))
        hs.append(float(1.96 * y.std(ddof=1) / np.sqrt(y.size)))
    return np.array(rs), np.array(ms), np.array(hs)


def fig6(rank, seeds, pred, out):
    fig, axes = plt.subplots(2, 2, figsize=(6.6, 4.8), sharex=True)
    for i, s in enumerate(SYSTEMS):
        for j, t in enumerate(TERMS):
            ax = axes[i, j]
            r, m, h = curve(rank, seeds, s, t)
            ax.fill_between(r, m - h, m + h, color=C[0], alpha=0.18, linewidth=0)
            ax.plot(r, m, color=C[0], marker=MARK[0], label=f"observed, mean of {len(seeds)} seeds")
            p = pred["fits"][f"{s}_{t}_ens_sigma"]
            ax.plot(p["r"], p["auroc"], color=C[1], linestyle="--", linewidth=1.5,
                    marker=MARK[1], markersize=3, label="predicted (pilot seed 0)")
            ax.axhline(0.5, color=INK2, linewidth=0.8, linestyle=":")
            ax.set_xscale("log")
            ax.set_title(f"{LABEL[s]}, {TERM[t]}", loc="left")
            if i == 1:
                ax.set_xlabel("r = median ||d|| / median ||a||")
            if j == 0:
                ax.set_ylabel("AUROC, ensemble σ")
    axes[0, 0].legend(frameon=False, loc="lower left")
    save(fig, out, "fig6_pde_transition")


def fig7(align_dir, out):
    recs = [json.loads(l) for fn in sorted(glob.glob(os.path.join(align_dir, "pde_align_seed*.jsonl")))
            for l in open(fn, encoding="utf-8") if l.strip()]
    if not recs:
        return
    names = {"ens_sigma": "ensemble σ", "pde_resid": "PDE residual"}
    rows = [(s, t, sig) for s in SYSTEMS for t in TERMS for sig in ("ens_sigma", "pde_resid")]
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    for i, (s, t, sig) in enumerate(rows):
        v = [r["signals"][sig] for r in recs if r["system"] == s and r["term"] == t]
        lo = np.mean([x["range_full"][0] for x in v])
        hi = np.mean([x["range_full"][1] for x in v])
        ax.plot([lo, hi], [i, i], color=GRID, linewidth=7, solid_capstyle="round", zorder=1)
        act = [x["actual_rho_D"]["16"] for x in v]
        jit = np.random.default_rng(i).uniform(-0.15, 0.15, len(act))
        ax.scatter(act, np.full(len(act), i) + jit, s=12, color=C[0], marker=MARK[0], zorder=3,
                   label="actual, 16 × ε_ref" if i == 0 else None)
        ax.scatter([np.mean([x["family_predicted"] for x in v])], [i], s=60, color=C[1], marker="|",
                   linewidths=2.5, zorder=4, label="true-family prediction" if i == 0 else None)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{s} {t}, {names[g]}" for s, t, g in rows])
    ax.set_xlim(-1, 1)
    ax.axvline(0, color=INK2, linewidth=0.8, linestyle=":")
    ax.set_xlabel("Spearman correlation with ||d||  (grey bar: alignment range)")
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)
    save(fig, out, "fig7_pde_alignment")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pde-dir", default=os.path.join("runs", "pde_main"))
    ap.add_argument("--align-dir", default=os.path.join("runs", "pde_alignment"))
    ap.add_argument("--pred", default=os.path.join("predictions", "pred_v2_seed0.json"))
    ap.add_argument("--out", default="figures")
    a = ap.parse_args()
    seeds = list(range(1, 21))
    _, rank = load(a.pde_dir, set(seeds))
    fig6(rank, seeds, json.load(open(a.pred, encoding="utf-8")), a.out)
    fig7(a.align_dir, a.out)
    print(f"figures written to {a.out}")


if __name__ == "__main__":
    main()

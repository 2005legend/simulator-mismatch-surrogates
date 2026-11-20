"""RQ1 runner: signals scored against the true-system error, per (seed, n_nom, system, eps).

Resumable: every finished cell is appended to the seed's JSONL file and skipped on a rerun;
the per-point arrays of each cell go to an npz file next to it. One file per seed, so seeds
can run as separate processes without interleaving lines.

Phases (addendum v1.6 and the plan): `dev` accepts only development seeds; `pilot` only
0, 101, 102; `main` only 1-10 and refuses to start unless the git tag `pred-v1` exists.

eps levels are multiples of eps_ref. Levels up to 2 x use all signals (prereg-v1). Levels
above 2 x (addendum v1.5) run only at n_nom = 512 and compute the simulator-only signals
plus the real-point records (D3).

Usage (development, one seed per process):
  python -m scripts.run_rq1 --phase dev --seeds 999 --n-nom 512 --out-dir runs/rq1_dev
"""

import argparse
import json
import os
import subprocess
import time

import numpy as np
import torch

from mismatch.acquisition import make_oracle
from mismatch.rq1 import N_CAL_TRUE, rq1_cell
from mismatch.simulators import simulate_batch, sobol_thetas, stream_seed
from mismatch.surrogate import Ensemble, FitConfig

PILOT_SEEDS = {0, 101, 102}
MAIN_SEEDS = set(range(1, 11))
FROZEN_MAX_MULT = 2.0
EXTRA_NNOM = 512
DEFAULT_MULTS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
DEV_EPS_REF = {"S1": 0.01, "S2": 0.0026}      # rough development values, not the pilot's


def _tag_exists(name):
    out = subprocess.run(["git", "tag", "--list", name], capture_output=True, text=True)
    return out.stdout.strip() == name


def check_phase(seeds, phase, pred_tag_exists=None):
    """Refuse seeds that do not belong to the phase; the main phase also needs `pred-v1`."""
    seeds = set(seeds)
    if phase == "dev":
        bad = sorted(seeds & (PILOT_SEEDS | MAIN_SEEDS))
        if bad:
            raise SystemExit(f"refusing registered seeds {bad} in the dev phase: use a development "
                             "seed such as 999, or --phase pilot (0, 101, 102) or main (1-10).")
    elif phase == "pilot":
        if not seeds <= PILOT_SEEDS:
            raise SystemExit(f"the pilot phase accepts only seeds 0, 101, 102, not {sorted(seeds)}.")
    elif phase == "main":
        if not seeds <= MAIN_SEEDS:
            raise SystemExit(f"the main phase accepts only seeds 1-10, not {sorted(seeds)}.")
        if pred_tag_exists is None:
            pred_tag_exists = _tag_exists("pred-v1")
        if not pred_tag_exists:
            raise SystemExit("the main phase needs the git tag pred-v1 (committed after the pilot).")
    else:
        raise SystemExit(f"unknown phase {phase!r}")


def cell_plan(n_nom, mults, extra_nnom=EXTRA_NNOM):
    """(mult, include_true_data, d3) for each level allowed at this n_nom."""
    plan = []
    for m in mults:
        if m <= FROZEN_MAX_MULT:
            plan.append((m, True, False))
        elif n_nom == extra_nnom:
            plan.append((m, False, True))
    return plan


def seed_file(out_dir, seed):
    return os.path.join(out_dir, f"rq1_seed{seed}.jsonl")


def load_done(path):
    done = set()
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            rec = json.loads(line)
            done.add((rec["seed"], rec["n_nom"], rec["system"], rec["mult"]))
    return done


def run(seeds, n_noms, systems, mults, eps_ref, out_dir, n_test=500, n_mc=50,
        nominal_members=5, nominal_cfg=None, mc_cfg=None, disc_members=5,
        n_cal_true=N_CAL_TRUE, extra_nnom=EXTRA_NNOM):
    torch.set_num_threads(1)
    os.makedirs(os.path.join(out_dir, "arrays"), exist_ok=True)
    nominal_sim = lambda th: simulate_batch(th, "S1", 0.0)
    for seed in seeds:
        path = seed_file(out_dir, seed)
        done = load_done(path)
        test = sobol_thetas(n_test, seed=stream_seed(seed, "test"))
        f_nom = nominal_sim(test)
        true_cache = {}
        for n_nom in n_noms:
            todo = [(s, m, tr, d3) for s in systems
                    for (m, tr, d3) in cell_plan(n_nom, mults, extra_nnom)
                    if (seed, n_nom, s, m) not in done]
            if not todo:
                continue
            t0 = time.perf_counter()
            train = sobol_thetas(n_nom, seed=stream_seed(seed, "train"))
            y_train = nominal_sim(train)
            nominal = Ensemble(n_members=nominal_members, cfg=nominal_cfg,
                               base_seed=stream_seed(seed, "nominal")).fit(train, y_train)
            mc_model = Ensemble(n_members=1, cfg=mc_cfg or FitConfig(dropout=0.1),
                                base_seed=stream_seed(seed, "mc")).fit(train, y_train)
            print(f"seed {seed} n_nom {n_nom}: surrogates fitted in "
                  f"{time.perf_counter() - t0:.0f} s", flush=True)
            for system, mult, include_true, d3 in todo:
                eps = mult * eps_ref[system]
                key = (system, mult)
                if key not in true_cache:
                    true_cache[key] = f_nom if mult == 0 else simulate_batch(test, system, eps)
                records, arrays = rq1_cell(
                    nominal, mc_model, test, true_cache[key], make_oracle(system, eps, seed=seed),
                    seed, n_mc=n_mc, include_true_data=include_true, d3=d3, return_arrays=True,
                    disc_members=disc_members, n_cal_true=n_cal_true, nominal_sim=nominal_sim)
                np.savez_compressed(os.path.join(
                    out_dir, "arrays", f"seed{seed}_n{n_nom}_{system}_x{mult:g}.npz"), **arrays)
                with open(path, "a", encoding="utf-8") as fh:
                    for rec in records:
                        fh.write(json.dumps({"seed": seed, "n_nom": n_nom, "system": system,
                                             "mult": mult, "eps": eps, **rec}) + chr(10))
                print(f"  {system} x{mult:g} done ({len(records)} records)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["dev", "pilot", "main"], default="dev")
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--n-nom", type=int, nargs="+", default=[512])
    ap.add_argument("--systems", nargs="+", default=["S1", "S2"])
    ap.add_argument("--mults", type=float, nargs="+", default=list(DEFAULT_MULTS))
    ap.add_argument("--eps-ref-s1", type=float)
    ap.add_argument("--eps-ref-s2", type=float)
    ap.add_argument("--params", help="frozen_params.json from the pilot")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    check_phase(args.seeds, args.phase)
    extra = {}
    if args.params:
        from scripts.run_pilot import load_params
        fp = load_params(args.params)
        args.eps_ref_s1, args.eps_ref_s2 = fp["eps_ref"]["S1"], fp["eps_ref"]["S2"]
        extra = {"nominal_cfg": fp["nominal_cfg"]}
    if args.phase == "dev":
        eps_ref = {"S1": args.eps_ref_s1 or DEV_EPS_REF["S1"],
                   "S2": args.eps_ref_s2 or DEV_EPS_REF["S2"]}
    else:
        if args.eps_ref_s1 is None or args.eps_ref_s2 is None:
            raise SystemExit("pilot and main phases need --eps-ref-s1 and --eps-ref-s2 from the pilot.")
        eps_ref = {"S1": args.eps_ref_s1, "S2": args.eps_ref_s2}
    out_dir = args.out_dir or os.path.join("runs", f"rq1_{args.phase}")
    run(args.seeds, args.n_nom, args.systems, args.mults, eps_ref, out_dir, **extra)


if __name__ == "__main__":
    main()

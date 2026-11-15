"""RQ1 runner: signals scored against the true-system error, per (seed, n_nom, system, eps).

Resumable: every finished cell is appended to a JSONL file and skipped on a rerun.
The pilot seeds (0, 101, 102) and main seeds (1-10) are refused unless --allow-registered
is given, so a development run cannot touch them by accident.

eps levels are multiples of eps_ref. Levels up to 2 x use all signals (prereg-v1); levels
above 2 x (addendum v1.5) compute only the simulator-only signals, which need no true data.

Usage (development):
  python -m scripts.run_rq1 --seeds 999 --n-nom 512 --out runs/rq1_dev.jsonl
"""

import argparse
import json
import os
import time

import torch

from mismatch.acquisition import make_oracle
from mismatch.rq1 import rq1_cell
from mismatch.simulators import simulate_batch, sobol_thetas, stream_seed
from mismatch.surrogate import Ensemble, FitConfig

REGISTERED = {0, 101, 102} | set(range(1, 11))
FROZEN_MAX_MULT = 2.0
DEFAULT_MULTS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)


def check_seeds(seeds, allow_registered=False):
    bad = sorted(set(seeds) & REGISTERED)
    if bad and not allow_registered:
        raise SystemExit(f"refusing registered seeds {bad}: pilot seeds are 0, 101, 102 and main "
                         "seeds are 1-10. Use a development seed (e.g. 999) or --allow-registered.")


def load_done(path):
    done = set()
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            rec = json.loads(line)
            done.add((rec["seed"], rec["n_nom"], rec["system"], rec["mult"]))
    return done


def run(seeds, n_noms, systems, mults, eps_ref, out, n_test=500, n_mc=50):
    torch.set_num_threads(1)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    done = load_done(out)
    nominal_sim = lambda th: simulate_batch(th, "S1", 0.0)
    for seed in seeds:
        test = sobol_thetas(n_test, seed=stream_seed(seed, "test"))
        f_nom = nominal_sim(test)
        for n_nom in n_noms:
            todo = [(s, m) for s in systems for m in mults if (seed, n_nom, s, m) not in done]
            if not todo:
                continue
            t0 = time.perf_counter()
            train = sobol_thetas(n_nom, seed=stream_seed(seed, "train"))
            y_train = nominal_sim(train)
            nominal = Ensemble(n_members=5, base_seed=seed).fit(train, y_train)
            mc_model = Ensemble(n_members=1, cfg=FitConfig(dropout=0.1),
                                base_seed=seed * 10 + 7).fit(train, y_train)
            print(f"seed {seed} n_nom {n_nom}: surrogates fitted in "
                  f"{time.perf_counter() - t0:.0f} s", flush=True)
            for system, mult in todo:
                eps = mult * eps_ref[system]
                f_true = f_nom if mult == 0 else simulate_batch(test, system, eps)
                records = rq1_cell(nominal, mc_model, test, f_true,
                                   make_oracle(system, eps, seed=seed), seed, n_mc=n_mc,
                                   include_true_data=mult <= FROZEN_MAX_MULT,
                                   nominal_sim=nominal_sim)
                with open(out, "a", encoding="utf-8") as fh:
                    for rec in records:
                        fh.write(json.dumps({"seed": seed, "n_nom": n_nom, "system": system,
                                             "mult": mult, "eps": eps, **rec}) + chr(10))
                print(f"  {system} x{mult:g} done ({len(records)} records)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--n-nom", type=int, nargs="+", default=[512])
    ap.add_argument("--systems", nargs="+", default=["S1", "S2"])
    ap.add_argument("--mults", type=float, nargs="+", default=list(DEFAULT_MULTS))
    ap.add_argument("--eps-ref-s1", type=float, default=0.01)    # rough dev value
    ap.add_argument("--eps-ref-s2", type=float, default=0.0026)  # rough dev value
    ap.add_argument("--out", default=os.path.join("runs", "rq1_dev.jsonl"))
    ap.add_argument("--allow-registered", action="store_true")
    args = ap.parse_args()
    check_seeds(args.seeds, args.allow_registered)
    run(args.seeds, args.n_nom, args.systems, args.mults,
        {"S1": args.eps_ref_s1, "S2": args.eps_ref_s2}, args.out)


if __name__ == "__main__":
    main()

"""Development run of Family D acquisition (not the pilot; dev seed 999 only).

One system at 4 x a rough eps_ref, all five arms, full-size pools (4,096), 8 initial points
then 6 rounds of 8, learned-term correction on the full 10 s horizon. Appends one JSON line
per arm to runs/dev_family_d.jsonl. Never the pilot seeds (0, 101, 102) or main seeds (1-10).

Usage: python -m scripts.dev_family_d --system S2
"""

import argparse
import json
import os
import time

import torch

from mismatch.acquisition import Problem, make_oracle
from mismatch.acquisition_d import ARMS_D, run_arm_d
from mismatch.simulators import simulate_batch, sobol_thetas, stream_seed
from mismatch.surrogate import Ensemble

DEV_SEED = 999
EPS_REF = {"S1": 0.01, "S2": 0.0026}      # rough dev values, not the pilot's
OUT = os.path.join("runs", "dev_family_d.jsonl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", choices=["S1", "S2"], required=True)
    ap.add_argument("--mult", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=DEV_SEED)
    args = ap.parse_args()
    assert args.seed not in set(range(0, 11)) | {101, 102}, "registered seed"
    torch.set_num_threads(1)
    os.makedirs("runs", exist_ok=True)

    train = sobol_thetas(512, seed=stream_seed(args.seed, "train"))
    test = sobol_thetas(500, seed=stream_seed(args.seed, "test"))
    nominal = Ensemble(n_members=5, base_seed=args.seed).fit(train, simulate_batch(train, "S1", 0.0))
    eps = args.mult * EPS_REF[args.system]
    problem = Problem(nominal=nominal, oracle=make_oracle(args.system, eps, seed=args.seed),
                      test_thetas=test, test_true=simulate_batch(test, args.system, eps),
                      seed=args.seed)
    init_obs = problem.oracle(problem.design()[:problem.n_init])
    for arm in ARMS_D:
        t0 = time.perf_counter()
        res = run_arm_d(problem, arm, init_obs)
        rec = {"system": args.system, "mult": args.mult, "seed": args.seed, "arm": arm,
               "seconds": time.perf_counter() - t0, "budget_auc": res["budget_auc"],
               "budget_auc_corrected": res["budget_auc_corrected"],
               "median_by_n": [(r["n"], r["median"], r["corrected_median"]) for r in res["log"]]}
        with open(OUT, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + chr(10))
        print(f"{args.system} {arm}: auc {res['budget_auc']:.4f}, final {res['log'][-1]['median']:.4f}"
              f", {rec['seconds']:.0f} s", flush=True)


if __name__ == "__main__":
    main()

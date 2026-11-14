"""Development check: can acquisition arms separate on a 4 s horizon? (not the pilot)

Dev seeds 999, 998, 997 only: never the pilot seeds (0, 101, 102) or main seeds (1-10).
Arms 1-7, S1 and S2 at 1x and 4x a rough eps_ref, output-space correction (width 32,
1,500 epochs), scored on the first 4 s (81 points). Reports the paired log ratio of
budget-AUC to Sobol per arm: mean and SD over the three seeds.

Usage: python -m scripts.dev_arm_separation
"""

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

from mismatch.acquisition import ARMS, Problem, make_oracle, run_arm
from mismatch.simulators import simulate_batch, sobol_thetas
from mismatch.surrogate import Ensemble, FitConfig

DEV_SEEDS = (999, 998, 997)
EPS_REF = {"S1": 0.01, "S2": 0.0026}
MULTIPLES = (1, 4)
ARMS_RUN = (1, 2, 3, 4, 5, 6, 7)
N_T = 81
DISC = FitConfig(hidden=32, max_epochs=1500, min_val=10 ** 9)
LOG = os.path.join("runs", "dev_arm_separation.jsonl")   # runs/ is git-ignored


def run_seed(seed):
    torch.set_num_threads(1)
    train = sobol_thetas(512, seed=seed)
    test = sobol_thetas(500, seed=seed + 5000)
    nominal = Ensemble(n_members=5, base_seed=seed).fit(train, simulate_batch(train, "S1", 0.0))
    out = {}
    for system in ("S1", "S2"):
        for m in MULTIPLES:
            eps = m * EPS_REF[system]
            problem = Problem(nominal=nominal, oracle=make_oracle(system, eps, seed=seed),
                              test_thetas=test, test_true=simulate_batch(test, system, eps),
                              seed=seed, n_t=N_T, disc_cfg=DISC)
            init_obs = problem.oracle(problem.design()[:problem.n_init])
            cell = {}
            for arm in ARMS_RUN:
                res = run_arm(problem, arm, init_obs)
                cell[arm] = {"auc": res["budget_auc"], "final": res["log"][-1]["median"],
                             "uncorrected": res["uncorrected"]["median"]}
            out[f"{system}_{m}"] = cell
            with open(LOG, "a", encoding="utf-8") as fh:      # survives a crash
                fh.write(json.dumps({"seed": seed, "cell": f"{system}_{m}", "arms": cell}) + "\n")
            print(f"seed {seed} {system} x{m} done", file=sys.stderr, flush=True)
    return seed, out


def load_done():
    done = {}
    if os.path.exists(LOG):
        for line in open(LOG, encoding="utf-8"):
            rec = json.loads(line)
            done.setdefault(rec["seed"], {})[rec["cell"]] = {int(a): v for a, v in rec["arms"].items()}
    return done


def main():
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    n_cells = 2 * len(MULTIPLES)
    todo = [s for s in DEV_SEEDS if len(load_done().get(s, {})) < n_cells]
    if todo:
        with ProcessPoolExecutor(max_workers=2) as pool:
            list(pool.map(run_seed, todo))
    results = load_done()
    for cell in results[DEV_SEEDS[0]]:
        unc = np.mean([results[s][cell][2]["uncorrected"] for s in DEV_SEEDS])
        sob = np.mean([results[s][cell][2]["final"] for s in DEV_SEEDS])
        print(f"\n{cell}: uncorrected {unc:.4f}, Sobol at n = 56 {sob:.4f}")
        print("arm, log(budget-AUC / Sobol) per seed, mean, SD, final error mean")
        for arm in ARMS_RUN:
            g = np.array([np.log(results[s][cell][arm]["auc"] / results[s][cell][2]["auc"])
                          for s in DEV_SEEDS])
            fin = np.mean([results[s][cell][arm]["final"] for s in DEV_SEEDS])
            print(f"{arm} {ARMS[arm]:18s} {np.round(g, 3)} {g.mean():+.3f} {g.std(ddof=1):.3f} "
                  f"{fin:.4f}")


if __name__ == "__main__":
    main()

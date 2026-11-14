"""Development smoke run of the full-size acquisition loop (not the pre-registered pilot).

Dev seed 999 only: never the pilot seeds (0, 101, 102) or main seeds (1-10). One cell per
system at a rough eps_ref, all nine arms, k = 1. Checks that every arm runs at full size
and measures time per arm. With one seed the arm ranking means nothing.

Usage: python -m scripts.dev_acquisition_smoke
"""

import time

import numpy as np
import torch

from mismatch.acquisition import ARMS, Problem, make_oracle, run_arm
from mismatch.simulators import simulate_batch, sobol_thetas
from mismatch.surrogate import Ensemble

DEV_SEED = 999
EPS_REF = {"S1": 0.01, "S2": 0.0026}   # rough dev values, not the pilot's


def nominal_sim(thetas):
    return simulate_batch(thetas, "S1", 0.0, n_jobs=6 if len(thetas) > 200 else 1)


def main():
    torch.set_num_threads(1)
    train = sobol_thetas(512, seed=DEV_SEED)
    test = sobol_thetas(500, seed=DEV_SEED + 1)
    t0 = time.perf_counter()
    nominal = Ensemble(n_members=5, base_seed=DEV_SEED).fit(train, nominal_sim(train))
    print(f"nominal ensemble (5 members): {time.perf_counter() - t0:.0f} s", flush=True)

    pool_err = {}
    for system in ("S1", "S2"):
        eps = EPS_REF[system]
        problem = Problem(nominal=nominal, oracle=make_oracle(system, eps, seed=DEV_SEED),
                          test_thetas=test,
                          test_true=simulate_batch(test, system, eps, n_jobs=6),
                          seed=DEV_SEED, nominal_sim=nominal_sim, _pool_err=pool_err)
        init_obs = problem.oracle(problem.design()[:problem.n_init])
        print(f"\n{system}, eps = {eps}: arm, median error at n = 8 / 32 / 56, "
              "budget-AUC, ratio to Sobol, seconds", flush=True)
        results = {}
        for arm in ARMS:
            t0 = time.perf_counter()
            results[arm] = run_arm(problem, arm, init_obs)
            results[arm]["seconds"] = time.perf_counter() - t0
        sobol_auc = results[2]["budget_auc"]
        for arm, res in results.items():
            med = [row["median"] for row in res["log"]]
            print(f"{arm} {res['name']:18s} {med[0]:.4f} {med[3]:.4f} {med[6]:.4f} | "
                  f"{res['budget_auc']:.4f} {res['budget_auc'] / sobol_auc:5.2f} | "
                  f"{res['seconds']:5.1f}", flush=True)
        print("arm 8 gate:", [(g["n"], round(g["r_hat"], 2), round(g["lower"], 2), g["gate"])
                              for g in results[8]["gate_log"]], flush=True)


if __name__ == "__main__":
    main()

"""Acquisition runner: Family A (output-space correction, arms 1-8) and Family F (a learned
missing term inside the ODE, five arms), per (seed, system, eps).

Same pattern as `run_rq1.py`: phases (dev, pilot, main), one JSONL file per seed and family,
finished cells skipped on a rerun, the design points chosen by every arm saved to an npz so
that design clustering (E2) and the density-ratio analysis (G3) need no rerun.

Family A uses the five frozen eps levels at n_nom = 512; Family F (addendum v1.6) uses 4 x
eps_ref. The batch exponent k and eps_ref come from the pilot; the dev phase uses rough
development values.

Usage (development):
  python -m scripts.run_acquisition --family A --phase dev --seeds 999 --out-dir runs/acq_dev
"""

import argparse
import json
import os
import time

import numpy as np
import torch

from mismatch.acquisition import ARMS, Problem, make_oracle, run_cell
from mismatch.acquisition_d import ARMS_D, run_cell_d
from mismatch.simulators import simulate_batch, sobol_thetas, stream_seed
from mismatch.surrogate import Ensemble
from scripts.run_rq1 import DEV_EPS_REF, check_phase

FAMILY_A_MULTS = (0.0, 0.25, 0.5, 1.0, 2.0)
FAMILY_F_MULTS = (4.0,)
FAMILY_A_ARMS = (1, 2, 3, 4, 5, 6, 7, 8)       # arm 9 is cut (addendum v1.6 section 6)


def seed_file(out_dir, family, seed):
    return os.path.join(out_dir, f"acq{family}_seed{seed}.jsonl")


def load_done(path):
    done = set()
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            rec = json.loads(line)
            done.add((rec["seed"], rec["system"], rec["mult"]))
    return done


def run(family, seeds, systems, mults, eps_ref, out_dir, k=1.0, n_nom=512, n_test=500,
        nominal_members=5, nominal_cfg=None, problem_kw=None, arms=None, f_kw=None):
    """Run one family over seeds x systems x mults; see the module docstring."""
    assert family in ("A", "F")
    torch.set_num_threads(1)
    os.makedirs(os.path.join(out_dir, "arrays"), exist_ok=True)
    problem_kw, f_kw = problem_kw or {}, f_kw or {}
    nominal_sim = lambda th: simulate_batch(th, "S1", 0.0)
    for seed in seeds:
        path = seed_file(out_dir, family, seed)
        done = load_done(path)
        todo = [(s, m) for s in systems for m in mults if (seed, s, m) not in done]
        if not todo:
            continue
        t0 = time.perf_counter()
        train = sobol_thetas(n_nom, seed=stream_seed(seed, "train"))
        nominal = Ensemble(n_members=nominal_members, cfg=nominal_cfg,
                           base_seed=stream_seed(seed, "nominal")).fit(train, nominal_sim(train))
        test = sobol_thetas(n_test, seed=stream_seed(seed, "test"))
        pool_err = {}                                   # arm 8's pool denominator, shared by cells
        print(f"seed {seed}: nominal surrogate fitted in {time.perf_counter() - t0:.0f} s", flush=True)
        for system, mult in todo:
            eps = mult * eps_ref[system]
            problem = Problem(nominal=nominal, oracle=make_oracle(system, eps, seed=seed),
                              test_thetas=test, test_true=simulate_batch(test, system, eps),
                              seed=seed, k=k, _pool_err=pool_err, **problem_kw)
            t1 = time.perf_counter()
            if family == "A":
                names = ARMS
                results = run_cell(problem, arms=tuple(arms or FAMILY_A_ARMS))
            else:
                names = {a: a for a in ARMS_D}
                results = run_cell_d(problem, arms=tuple(arms or ARMS_D), **f_kw)
            seconds = time.perf_counter() - t1
            design = {}
            with open(path, "a", encoding="utf-8") as fh:
                for arm, res in results.items():
                    rec = {"seed": seed, "family": family, "system": system, "mult": mult, "eps": eps,
                           "arm": arm, "name": names[arm], "budget_auc": res["budget_auc"],
                           "log": res["log"], "cell_seconds": seconds}
                    if family == "A":
                        rec["uncorrected_median"] = res["uncorrected"]["median"]
                        rec["gate_log"] = res["gate_log"]
                    else:
                        rec["budget_auc_corrected"] = res["budget_auc_corrected"]
                    fh.write(json.dumps(rec) + chr(10))
                    design[f"thetas_{names[arm]}"] = res["thetas"]
            np.savez_compressed(os.path.join(
                out_dir, "arrays", f"acq{family}_seed{seed}_{system}_x{mult:g}.npz"), **design)
            print(f"  {family} {system} x{mult:g} done in {seconds:.0f} s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=["A", "F"], required=True)
    ap.add_argument("--phase", choices=["dev", "pilot", "main"], default="dev")
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--systems", nargs="+", default=["S1", "S2"])
    ap.add_argument("--mults", type=float, nargs="+")
    ap.add_argument("--k", type=float, help="batch exponent from the pilot")
    ap.add_argument("--eps-ref-s1", type=float)
    ap.add_argument("--eps-ref-s2", type=float)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    check_phase(args.seeds, args.phase)
    if args.phase == "dev":
        eps_ref = {"S1": args.eps_ref_s1 or DEV_EPS_REF["S1"], "S2": args.eps_ref_s2 or DEV_EPS_REF["S2"]}
        k = args.k if args.k is not None else 1.0
    else:
        if None in (args.eps_ref_s1, args.eps_ref_s2, args.k):
            raise SystemExit("pilot and main phases need --eps-ref-s1, --eps-ref-s2 and --k from the pilot.")
        eps_ref, k = {"S1": args.eps_ref_s1, "S2": args.eps_ref_s2}, args.k
    mults = args.mults or (FAMILY_A_MULTS if args.family == "A" else FAMILY_F_MULTS)
    out_dir = args.out_dir or os.path.join("runs", f"acq_{args.phase}")
    run(args.family, args.seeds, args.systems, mults, eps_ref, out_dir, k=k)


if __name__ == "__main__":
    main()

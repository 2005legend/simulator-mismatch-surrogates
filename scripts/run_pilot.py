"""Pilot orchestrator (prereg-v1 section 9, addenda v1.1 and v1.6). Pilot seeds 0, 101, 102 only.

Stages, each saved to <out>/stage_<name>.json and skipped on a rerun (run in this order):
  hyper  MLP setting with the lowest validation loss (seed 0, 2 members, simulator data only)
  eps    eps_ref per system: the eps where r = median||d|| / median||a|| = 1 (seed 0, n_nom = 512,
         the chosen MLP setting)
  aux    spline smoothing (GCV) and GP component count, from the pilot's true trajectories
  k      batch exponent k in {1, 2, 4}: lowest mean budget-AUC of arms 3-7 at 1 x eps_ref
  seeds  main-run seed count from the paired slope of arm 3 against Sobol over eps (arms 2, 3)
  fcv    Family F fit-to-fit variation (3 fit seeds, n = 8, 24, 56)
`finalize` writes frozen_params.json with provenance. Runners read it through --params.

Usage:  python -m scripts.run_pilot --stage all --out-dir runs/pilot
"""

import argparse
import json
import os
import subprocess
import time

import numpy as np
import torch

from mismatch.acquisition import make_oracle
from mismatch.dynamics import fit_term, simulate_with_term
from mismatch.metrics import rel_errors
from mismatch.pilot import (HYPER_GRID, K_CANDIDATES, choose_hyperparameters, choose_k, gcv_lambda,
                            log_bisect, ols_slope, pca_components, seed_count_rule)
from mismatch.simulators import simulate_batch, sobol_thetas, stream_seed
from mismatch.surrogate import Ensemble, FitConfig
from scripts.run_acquisition import run as run_acq
from scripts.run_rq1 import PILOT_SEEDS, check_phase

SYSTEMS = ("S1", "S2")
EPS_BRACKET = (1e-4, 1.0)
F_SIZES = (8, 24, 56)
SEED_MULTS = (0.25, 0.5, 1.0, 2.0)
nominal_sim = lambda th: simulate_batch(th, "S1", 0.0)


def stage_path(out, name):
    return os.path.join(out, f"stage_{name}.json")


def cached(out, name, fn):
    path = stage_path(out, name)
    if os.path.exists(path):
        return json.load(open(path, encoding="utf-8"))
    t0 = time.perf_counter()
    result = fn()
    result["_seconds"] = round(time.perf_counter() - t0, 1)
    json.dump(result, open(path, "w", encoding="utf-8"), indent=1)
    print(f"stage {name} done in {result['_seconds']} s", flush=True)
    return result


def fit_nominal(seed, n_nom, members=5, cfg=None):
    train = sobol_thetas(n_nom, seed=stream_seed(seed, "train"))
    return Ensemble(n_members=members, cfg=cfg, base_seed=stream_seed(seed, "nominal")).fit(
        train, nominal_sim(train))


def ratio_r(nominal, test, system, eps):
    """r = median||d|| / median||a||, both relative to ||f_true||, as in the RQ1 summary."""
    f_true, f_nom = simulate_batch(test, system, eps), nominal_sim(test)
    denom = np.linalg.norm(f_true, axis=1)
    a = np.linalg.norm(nominal.predict(test)[0] - f_nom, axis=1) / denom
    d = np.linalg.norm(f_nom - f_true, axis=1) / denom
    return float(np.median(d) / np.median(a))


def stage_eps(seed=0, n_nom=512, n_test=500, members=5, cfg=None):
    nominal = fit_nominal(seed, n_nom, members, cfg)
    test = sobol_thetas(n_test, seed=stream_seed(seed, "test"))
    out = {s: log_bisect(lambda e: ratio_r(nominal, test, s, e), 1.0, *EPS_BRACKET) for s in SYSTEMS}
    return {"eps_ref": out}


def stage_hyper(seed=0, n_nom=512, base=None, grid=None):
    grid = grid or HYPER_GRID
    train = sobol_thetas(n_nom, seed=stream_seed(seed, "train"))
    y = nominal_sim(train)
    losses = {}
    for label, over in grid.items():
        cfg = FitConfig(**{**(base or {}), **over})
        ens = Ensemble(n_members=2, cfg=cfg, base_seed=stream_seed(seed, "nominal")).fit(train, y)
        losses[label] = [m.best_val for m in ens.members]
    best, means = choose_hyperparameters(losses)
    return {"best": best, "overrides": grid[best], "mean_val_loss": means}


def stage_aux(eps_ref, seeds, n_pts=56, n_nom=512, cfg=None, members=5):
    lams, gp_k = [], {s: [] for s in SYSTEMS}
    for seed in seeds:
        nominal = fit_nominal(seed, n_nom, members, cfg)
        x = sobol_thetas(n_pts, seed=stream_seed(seed, "design"))
        for s in SYSTEMS:
            y, _ = make_oracle(s, eps_ref[s], seed=seed)(x)
            lams.append(gcv_lambda(y[:10]))                      # 10 trajectories per cell
            gp_k[s].append(pca_components(y - nominal.predict(x)[0]))
    lam = float(np.exp(np.mean(np.log(lams))))                   # geometric mean over cells
    return {"spline_lam": lam, "spline_lam_all": lams, "gp_components_by_system": gp_k,
            "gp_components": int(max(max(v) for v in gp_k.values()))}


def collect_auc(dirpath, family="A"):
    """{(seed, system, mult, arm_name): budget_auc} from a runner's JSONL files."""
    out = {}
    for fn in os.listdir(dirpath):
        if fn.startswith(f"acq{family}_seed") and fn.endswith(".jsonl"):
            for line in open(os.path.join(dirpath, fn), encoding="utf-8"):
                r = json.loads(line)
                out[(r["seed"], r["system"], r["mult"], r["name"])] = r["budget_auc"]
    return out


def stage_k(out, eps_ref, seeds, **kw):
    arms, auc = (3, 4, 5, 6, 7), {}
    for k in K_CANDIDATES:
        d = os.path.join(out, f"acq_k{k}")
        run_acq("A", seeds, SYSTEMS, (1.0,), eps_ref, d, k=float(k), arms=arms, **kw)
        auc[k] = list(collect_auc(d).values())
    best, means = choose_k(auc)
    return {"k": float(best), "mean_budget_auc": means}


def stage_seeds(out, eps_ref, seeds, k, **kw):
    d = os.path.join(out, "acq_seeds")
    run_acq("A", seeds, SYSTEMS, SEED_MULTS, eps_ref, d, k=k, arms=(2, 3), **kw)
    auc = collect_auc(d)
    slopes = []
    for seed in seeds:
        xs, ys = [], []
        for s in SYSTEMS:
            for m in SEED_MULTS:
                xs.append(np.log(m))
                ys.append(np.log(auc[(seed, s, m, "nominal_sigma")] / auc[(seed, s, m, "sobol")]))
        slopes.append(ols_slope(xs, ys))
    return {"slopes": slopes, **seed_count_rule(slopes)}


def stage_fcv(eps_ref, seed=0, fit_seeds=3, n_test=100, iters=20, gm_steps=500, sizes=F_SIZES):
    test = sobol_thetas(n_test, seed=stream_seed(seed, "test"))
    res = {}
    for s in SYSTEMS:
        truth = simulate_batch(test, s, eps_ref[s])
        oracle = make_oracle(s, eps_ref[s], seed=seed)
        res[s] = {}
        for n in sizes:
            x = sobol_thetas(n, seed=stream_seed(seed, "design"))
            y, _ = oracle(x)
            errs = []
            for j in range(fit_seeds):
                term = fit_term(x, y, seed=stream_seed(seed, "fit", j), lbfgs_iters=iters,
                                gm_steps=gm_steps)
                errs.append(float(np.median(rel_errors(simulate_with_term(test, term), truth))))
            res[s][str(n)] = {"errors": errs, "cv": float(np.std(errs, ddof=1) / np.mean(errs))}
    return {"fit_to_fit": res}


SOURCE_FILES = ("mismatch/pilot.py", "scripts/run_pilot.py", "mismatch/surrogate.py",
                "mismatch/acquisition.py", "mismatch/acquisition_d.py", "mismatch/dynamics.py",
                "mismatch/simulators.py", "scripts/run_acquisition.py", "scripts/run_rq1.py")


def git_commit():
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()


def git_dirty():
    out = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout
    return bool(out.strip())


def source_hashes():
    """sha256 of the code the pilot ran, so a later commit can be checked against it."""
    import hashlib
    return {f: hashlib.sha256(open(f, "rb").read()).hexdigest() for f in SOURCE_FILES if os.path.exists(f)}


def finalize(out):
    st = {n: json.load(open(stage_path(out, n), encoding="utf-8"))
          for n in ("eps", "hyper", "aux", "k", "seeds", "fcv")}
    params = {"eps_ref": st["eps"]["eps_ref"], "k": st["k"]["k"],
              "spline_lam": st["aux"]["spline_lam"], "gp_components": st["aux"]["gp_components"],
              "nominal_cfg": st["hyper"]["overrides"], "n_seeds": st["seeds"]["n_seeds"],
              "provenance": {"pilot_seeds": sorted(PILOT_SEEDS), "git_commit": git_commit(),
                             "git_dirty": git_dirty(), "source_sha256": source_hashes(),
                             "stages": sorted(st)}}
    json.dump(params, open(os.path.join(out, "frozen_params.json"), "w", encoding="utf-8"), indent=1)
    return params


def load_params(path):
    """Frozen parameters for the runners: eps_ref, k, nominal FitConfig, Problem keywords."""
    p = json.load(open(path, encoding="utf-8"))
    return {"eps_ref": p["eps_ref"], "k": p["k"], "nominal_cfg": FitConfig(**p["nominal_cfg"]),
            "problem_kw": {"spline_lam": p["spline_lam"], "gp_components": p["gp_components"]}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["all", "eps", "hyper", "aux", "k", "seeds", "fcv", "finalize"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 101, 102])
    ap.add_argument("--out-dir", default=os.path.join("runs", "pilot"))
    args = ap.parse_args()
    check_phase(args.seeds, "pilot")
    torch.set_num_threads(1)
    out = args.out_dir
    os.makedirs(out, exist_ok=True)
    want = lambda n: args.stage in ("all", n)
    # hyperparameters first, so eps_ref is calibrated on the surrogate the main runs will use
    nominal_cfg = FitConfig(**cached(out, "hyper", stage_hyper)["overrides"])
    eps_ref = cached(out, "eps", lambda: stage_eps(cfg=nominal_cfg))["eps_ref"]
    kw = {"nominal_cfg": nominal_cfg}
    if want("aux"):
        cached(out, "aux", lambda: stage_aux(eps_ref, args.seeds, cfg=nominal_cfg))
    if want("k") or want("seeds"):
        k = cached(out, "k", lambda: stage_k(out, eps_ref, args.seeds, **kw))["k"]
    if want("seeds"):
        cached(out, "seeds", lambda: stage_seeds(out, eps_ref, args.seeds, k, **kw))
    if want("fcv"):
        cached(out, "fcv", lambda: stage_fcv(eps_ref))
    if args.stage in ("all", "finalize"):
        print(json.dumps(finalize(out), indent=1))


if __name__ == "__main__":
    main()

"""PDE runner for addendum v1.7: Systems 3 (parametric Burgers, MLP) and 4 (FNO operator).

Stages:
  pilot   seed 0 only: eps_ref per (system, term) by bisection on r = 1 at n_nom = 512, written
          to <out>/eps_ref.json, then the full cell sweep on seed 0 (the input of pred-v2)
  cells   the sweep (eps in {0, .25, .5, 1, 2, 4, 8, 16} x eps_ref) for the given seeds
Records per cell follow the pendulum RQ1 runner (same scoring functions from mismatch.rq1);
arrays A, D, E and the signals are saved per cell. Resumable per (seed, system, term, mult).

Phases: dev (seeds other than 0-20, 101, 102), pilot (seed 0), main (1-20; needs tags
prereg-v1.7 and pred-v2).

Usage:  python -m scripts.run_pde --phase pilot --seeds 0 --stage pilot --out-dir runs/pde_pilot
        python -m scripts.run_pde --phase main --seeds 1 --eps-ref runs/pde_pilot/eps_ref.json \
            --out-dir runs/pde_main
"""

import argparse
import json
import os
import subprocess
import time

import numpy as np
import torch

from mismatch import burgers as B
from mismatch import config
from mismatch import operator as O
from mismatch.pilot import log_bisect
from mismatch.rq1 import (N_CAL_NOMINAL, N_CAL_TRUE, _interval_record, _rank_record,
                          cell_summary, conformal_quantile, d3_records, debiased_discrepancy,
                          sigma_rel)
from mismatch.simulators import sobol_thetas, stream_seed
from mismatch.surrogate import Ensemble, FitConfig
from scripts.run_pilot import load_params

MULTS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
TERMS = ("P1", "P2")
BRACKET = {"P1": (1e-5, 1.0), "P2": (1e-6, 1.0)}
N_TEST, N_NOM = 500, 512
MAIN_SEEDS, PILOT_SEEDS = set(range(1, 21)), {0}


def noisy(clean, thetas, seed):
    """1% noise per trajectory; the draw depends only on (seed, theta), as in make_oracle."""
    sd = B.NOISE * clean.std(axis=1)
    out = np.empty_like(clean)
    for i, th in enumerate(np.atleast_2d(thetas)):
        key = np.frombuffer(np.ascontiguousarray(th, dtype=np.float64).tobytes(), dtype=np.uint32).tolist()
        out[i] = clean[i] + sd[i] * np.random.default_rng([seed] + key).normal(size=clean.shape[1])
    return out, sd


class Burgers:
    """System 3 adapter."""
    name, has_mc = "B", True

    def __init__(self, cfg):
        self.cfg = cfg

    def sample(self, n, seed):
        # the shared Sobol stream (pendulum box) mapped affinely onto this box
        u = (sobol_thetas(n, seed=seed) - config.BOX_LOW) / (config.BOX_HIGH - config.BOX_LOW)
        return B.from_unit(u)

    def sim(self, th, term="P1", eps=0.0):
        return B.simulate_batch(th, term, eps)

    def fit(self, th, y, seed, members=5):
        return Ensemble(n_out=B.N_OUT, n_members=members, cfg=self.cfg,
                        base_seed=stream_seed(seed, "nominal")).fit(B.to_unit_box(th), y)

    def fit_mc(self, th, y, seed):
        return Ensemble(n_out=B.N_OUT, n_members=1, cfg=FitConfig(dropout=0.1),
                        base_seed=stream_seed(seed, "mc")).fit(B.to_unit_box(th), y)

    def predict(self, model, th):
        return model.predict(B.to_unit_box(th))

    def predict_mc(self, model, th, seed):
        return model.members[0].predict_mc(B.to_unit_box(th), n_passes=50, seed=seed).std(axis=0)

    def residual(self, mean, th):
        return B.residual_score(mean, th)


class Operator:
    """System 4 adapter; theta = Fourier coefficients of the initial profile."""
    name, has_mc = "O", False
    X64 = np.linspace(0.0, 2 * np.pi, O.N_OP, endpoint=False)

    def sample(self, n, seed):
        return O.sample_coefs(n, seed)

    def sim(self, th, term="P1", eps=0.0):
        y, _ = O.simulate(th, term, eps)
        return y.reshape(len(y), -1)

    def u0(self, th):
        return O.profiles(th, self.X64)

    def fit(self, th, y, seed, members=5):
        return O.OperatorEnsemble(n_members=members, base_seed=stream_seed(seed, "nominal")).fit(
            self.u0(th), y.reshape(len(y), B.N_OT - 1, O.N_OP))

    def predict(self, model, th):
        m, s = model.predict(self.u0(th))
        return m.reshape(len(m), -1), s.reshape(len(s), -1)

    def residual(self, mean, th):
        u = np.concatenate([self.u0(th)[:, None], mean.reshape(len(mean), B.N_OT - 1, O.N_OP)], axis=1)
        return B.residual_field(u, np.full(len(u), O.NU)).mean(axis=1)


def rel(x, ref):
    return np.linalg.norm(x, axis=1) / np.linalg.norm(ref, axis=1)


def tag_exists(name):
    out = subprocess.run(["git", "tag", "--list", name], capture_output=True, text=True).stdout
    return out.strip() == name


def check_phase(seeds, phase, tags=None):
    seeds = set(seeds)
    if phase == "dev" and seeds & (MAIN_SEEDS | PILOT_SEEDS | {101, 102}):
        raise SystemExit(f"refusing registered seeds {sorted(seeds)} in the dev phase")
    if phase == "pilot" and not seeds <= PILOT_SEEDS:
        raise SystemExit("the PDE pilot uses seed 0 only")
    if phase == "main":
        if not seeds <= MAIN_SEEDS:
            raise SystemExit("main PDE seeds are 1-20")
        tags = tags if tags is not None else {t: tag_exists(t) for t in ("prereg-v1.7", "pred-v2")}
        missing = [t for t, ok in tags.items() if not ok]
        if missing:
            raise SystemExit(f"the main PDE phase needs git tags {missing}")


def models(sysobj, seed, members=5):
    train = sysobj.sample(N_NOM, stream_seed(seed, "train"))
    y = sysobj.sim(train)
    t0 = time.perf_counter()
    nominal = sysobj.fit(train, y, seed, members)
    mc = sysobj.fit_mc(train, y, seed) if sysobj.has_mc else None
    print(f"seed {seed} {sysobj.name}: surrogates fitted in {time.perf_counter() - t0:.0f} s", flush=True)
    return nominal, mc


def ratio_r(sysobj, nominal, test, f_nom, term, eps):
    f_true = sysobj.sim(test, term, eps)
    mean, _ = sysobj.predict(nominal, test)
    return float(np.median(rel(f_nom - f_true, f_true)) / np.median(rel(mean - f_nom, f_true)))


def pde_cell(sysobj, nominal, mc, test, f_nom, f_true, term, eps, seed, d3=False):
    mean, sigma = sysobj.predict(nominal, test)
    a, d = mean - f_nom, f_nom - f_true
    A, D, E = rel(a, f_true), rel(d, f_true), rel(a + d, f_true)
    r = float(np.median(D) / np.median(A))
    signals = {"ens_sigma": sigma.mean(axis=1), "pde_resid": sysobj.residual(mean, test)}
    if mc is not None:
        signals["mc_sigma"] = sysobj.predict_mc(mc, test, seed).mean(axis=1)
    recs = [_rank_record(n, 0, s, E, A, D, r) for n, s in signals.items()]
    recs.append(cell_summary(A, D, r))
    recs.append(_interval_record("ens_sigma_gauss", 0, sigma_rel(sigma, mean), E, 1.645))
    cal = sysobj.sample(N_CAL_NOMINAL, stream_seed(seed, "calibration", 0))
    cm, cs = sysobj.predict(nominal, cal)
    cn = sysobj.sim(cal)
    q = conformal_quantile(rel(cm - cn, cn) / sigma_rel(cs, cm))
    recs.append(_interval_record("conf_nom", 0, sigma_rel(sigma, mean), E, q))
    cal_t = sysobj.sample(max(N_CAL_TRUE), stream_seed(seed, "calibration", 1))
    y_t, sd_t = noisy(sysobj.sim(cal_t, term, eps), cal_t, seed)
    ctm, cts = sysobj.predict(nominal, cal_t)
    for n in N_CAL_TRUE:
        qt = conformal_quantile(rel(ctm[:n] - y_t[:n], y_t[:n]) / sigma_rel(cts[:n], ctm[:n]))
        recs.append(_interval_record("conf_true", n, sigma_rel(sigma, mean), E, qt))
    if d3:
        dhat = debiased_discrepancy(y_t, sysobj.sim(cal_t), sd_t)
        recs += d3_records(cts.mean(axis=1), dhat, N_CAL_TRUE, seed)
    return recs, {"A": A, "D": D, "E": E, **signals}


def done_keys(path):
    if not os.path.exists(path):
        return set()
    return {(r["seed"], r["system"], r["term"], r["mult"]) for r in map(json.loads, open(path, encoding="utf-8"))}


def run_cells(sysobjs, seeds, eps_ref, out_dir, mults=MULTS, members=5):
    os.makedirs(os.path.join(out_dir, "arrays"), exist_ok=True)
    for seed in seeds:
        path = os.path.join(out_dir, f"pde_seed{seed}.jsonl")
        done = done_keys(path)
        for so in sysobjs:
            todo = [(t, m) for t in TERMS for m in mults if (seed, so.name, t, m) not in done]
            if not todo:
                continue
            nominal, mc = models(so, seed, members)
            test = so.sample(N_TEST, stream_seed(seed, "test"))
            f_nom = so.sim(test)
            for term, m in todo:
                eps = m * eps_ref[so.name][term]
                f_true = f_nom if m == 0 else so.sim(test, term, eps)
                recs, arrays = pde_cell(so, nominal, mc, test, f_nom, f_true, term, eps, seed,
                                        d3=(m == 16.0))
                np.savez_compressed(os.path.join(out_dir, "arrays",
                                                 f"seed{seed}_{so.name}_{term}_x{m:g}.npz"), **arrays)
                with open(path, "a", encoding="utf-8") as fh:
                    for r in recs:
                        fh.write(json.dumps({"seed": seed, "system": so.name, "term": term, "mult": m,
                                             "eps": eps, **r}) + "\n")
                print(f"  seed {seed} {so.name} {term} x{m:g} done", flush=True)


def stage_eps(sysobjs, out_dir, seed=0, members=5):
    path = os.path.join(out_dir, "eps_ref.json")
    res = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}
    for so in sysobjs:
        if so.name in res:
            continue
        nominal, _ = models(so, seed, members)
        test = so.sample(N_TEST, stream_seed(seed, "test"))
        f_nom = so.sim(test)
        res[so.name] = {t: log_bisect(lambda e: ratio_r(so, nominal, test, f_nom, t, e), 1.0, *BRACKET[t])
                        for t in TERMS}
        print(f"eps_ref {so.name}: {res[so.name]}", flush=True)
        os.makedirs(out_dir, exist_ok=True)
        json.dump(res, open(path, "w", encoding="utf-8"), indent=1)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["dev", "pilot", "main"], required=True)
    ap.add_argument("--stage", choices=["pilot", "cells"], default="cells")
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--systems", nargs="+", default=["B", "O"])
    ap.add_argument("--eps-ref", help="eps_ref.json from the pilot (cells stage)")
    ap.add_argument("--params", default=os.path.join("pilot_results", "frozen_params.json"))
    ap.add_argument("--members", type=int, default=5)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    check_phase(a.seeds, a.phase)
    torch.set_num_threads(1)
    cfg = load_params(a.params)["nominal_cfg"]
    objs = [{"B": Burgers(cfg), "O": Operator()}[s] for s in a.systems]
    if a.stage == "pilot":
        eps_ref = stage_eps(objs, a.out_dir, seed=a.seeds[0], members=a.members)
    else:
        eps_ref = json.load(open(a.eps_ref, encoding="utf-8"))
    run_cells(objs, a.seeds, eps_ref, a.out_dir, members=a.members)


if __name__ == "__main__":
    main()

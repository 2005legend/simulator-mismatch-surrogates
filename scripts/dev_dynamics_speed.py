"""Development speed test for the dynamics-space correction (dev seed 999 only).

Reference: 300 Adam steps of trajectory matching take 85-150 s per fit
(scripts/dev_dynamics_correction.py). Faster candidates:
  gm       gradient matching: spline derivatives of each observed trajectory give a direct
           regression target h = phi'' + c phi' + (g/L) sin(phi); no ODE solves
  gm+adam  gm, then 30 Adam steps of trajectory matching
  gm+lbfgs gm, then L-BFGS trajectory matching (20 iterations)
  lbfgs    L-BFGS trajectory matching from zero (25 iterations)
Scored by the learned ODE's median error against the true system on the test set.

Usage: python -m scripts.dev_dynamics_speed
"""

import time

import numpy as np
import torch
from scipy.interpolate import make_smoothing_spline

from mismatch import config
from mismatch.acquisition import make_oracle
from mismatch.metrics import rel_errors
from mismatch.simulators import simulate_batch, sobol_thetas
from mismatch.dynamics import Term, rollout

DEV_SEED = 999
CASES = (("S1", 0.04), ("S2", 0.0026), ("S2", 0.0104))
EDGE = 10


def gm_targets(thetas, y):
    xs, hs = [], []
    for th, traj in zip(thetas, y):
        spl = make_smoothing_spline(config.T_GRID, traj)
        phi, d1, d2 = spl(config.T_GRID), spl.derivative(1)(config.T_GRID), spl.derivative(2)(config.T_GRID)
        h = d2 + th[0] * d1 + (config.G / th[1]) * np.sin(phi)
        xs.append(np.stack([phi, d1], axis=1)[EDGE:-EDGE])
        hs.append(h[EDGE:-EDGE])
    return torch.from_numpy(np.concatenate(xs)), torch.from_numpy(np.concatenate(hs))


def fit_gm(thetas, y, seed, steps=500):
    torch.manual_seed(seed)
    term = Term()
    torch.nn.init.normal_(term.net[-1].weight, std=0.01)
    x, h = gm_targets(thetas, y)
    opt = torch.optim.Adam(term.parameters(), lr=1e-2)
    for _ in range(steps):
        opt.zero_grad()
        torch.mean((term(x[:, 0], x[:, 1]) - h) ** 2).backward()
        opt.step()
    return term


def traj_loss(term, x, target, scale):
    return torch.mean(((rollout(x, term) - target) / scale) ** 2)


def refine_adam(term, thetas, y, iters=30, lr=1e-3):
    x, target = torch.from_numpy(thetas), torch.from_numpy(y)
    scale = target.std(dim=1, keepdim=True)
    opt = torch.optim.Adam(term.parameters(), lr=lr)
    for _ in range(iters):
        opt.zero_grad()
        traj_loss(term, x, target, scale).backward()
        torch.nn.utils.clip_grad_norm_(term.parameters(), 1.0)
        opt.step()
    return term


def refine_lbfgs(term, thetas, y, iters=20):
    x, target = torch.from_numpy(thetas), torch.from_numpy(y)
    scale = target.std(dim=1, keepdim=True)
    opt = torch.optim.LBFGS(term.parameters(), lr=1.0, max_iter=iters,
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = traj_loss(term, x, target, scale)
        loss.backward()
        return loss

    opt.step(closure)
    return term


def main():
    torch.set_num_threads(1)
    test = sobol_thetas(500, seed=DEV_SEED + 1)
    design = sobol_thetas(56, seed=DEV_SEED * 1000)
    f_nom = simulate_batch(test, "S1", 0.0)
    xt = torch.from_numpy(test)

    def score(term):
        with torch.no_grad():
            return np.median(rel_errors(rollout(xt, term).numpy(), f_true))

    methods = {
        "gm": lambda th, y, s: fit_gm(th, y, s),
        "gm+adam": lambda th, y, s: refine_adam(fit_gm(th, y, s), th, y),
        "gm+lbfgs": lambda th, y, s: refine_lbfgs(fit_gm(th, y, s), th, y),
        "lbfgs": lambda th, y, s: refine_lbfgs(Term(), th, y, iters=25),
    }
    for system, eps in CASES:
        f_true = simulate_batch(test, system, eps)
        y, _ = make_oracle(system, eps, seed=DEV_SEED)(design)
        print(f"\n{system} eps={eps}: nominal ODE alone {np.median(rel_errors(f_nom, f_true)):.4f}",
              flush=True)
        for n in (8, 56):
            row = []
            for name, fit in methods.items():
                t0 = time.perf_counter()
                term = fit(design[:n], y[:n], DEV_SEED + n)
                row.append(f"{name} {score(term):.4f} ({time.perf_counter() - t0:.0f} s)")
            print(f"  n={n:2d}: " + " | ".join(row), flush=True)


if __name__ == "__main__":
    main()

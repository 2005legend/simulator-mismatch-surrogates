"""Development test of a dynamics-space correction (not pre-registered, dev seed 999 only).

Instead of an additive correction on the output trajectory, learn the missing term h in
  phi'' = -c phi' - (g/L) sin(phi) + h(phi, phi')
with a small network, by fitting simulated trajectories (differentiable RK4) to the n
noisy true observations over the full 10 s. Two corrected predictions are scored:
  hybrid:    the learned ODE integrated at the test point;
  surrogate: nominal surrogate + [learned-ODE trajectory - nominal-ODE trajectory].
The second keeps the frozen form "corrected = nominal surrogate + delta".

Usage: python -m scripts.dev_dynamics_correction
"""

import time

import numpy as np
import torch
from torch import nn

from mismatch import config
from mismatch.acquisition import make_oracle
from mismatch.metrics import rel_errors
from mismatch.simulators import simulate_batch, sobol_thetas
from mismatch.surrogate import Ensemble

DEV_SEED = 999
CASES = (("S1", 0.01), ("S1", 0.04), ("S2", 0.0026), ("S2", 0.0104))
N_STEPS = config.N_T - 1
DT = config.T_END / N_STEPS


class Term(nn.Module):
    """h(phi, omega); the last layer starts at zero so training starts at the nominal ODE."""

    def __init__(self, hidden=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, hidden), nn.SiLU(), nn.Linear(hidden, hidden),
                                 nn.SiLU(), nn.Linear(hidden, 1)).double()
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, phi, omega):
        return self.net(torch.stack([phi / np.pi, omega / 5.0], dim=1))[:, 0]


def rollout(thetas, term=None):
    """RK4 on the 201-point grid for a batch of thetas (n, 4). Returns phi (n, 201)."""
    c, length, phi, omega = thetas.unbind(dim=1)
    k = config.G / length

    def f(p, w):
        acc = -c * w - k * torch.sin(p)
        return w, acc + term(p, w) if term is not None else acc

    out = [phi]
    for _ in range(N_STEPS):
        p1, w1 = f(phi, omega)
        p2, w2 = f(phi + 0.5 * DT * p1, omega + 0.5 * DT * w1)
        p3, w3 = f(phi + 0.5 * DT * p2, omega + 0.5 * DT * w2)
        p4, w4 = f(phi + DT * p3, omega + DT * w3)
        phi = phi + DT / 6 * (p1 + 2 * p2 + 2 * p3 + p4)
        omega = omega + DT / 6 * (w1 + 2 * w2 + 2 * w3 + w4)
        out.append(phi)
    return torch.stack(out, dim=1)


def fit_term(thetas, y, seed, iters=300, lr=3e-3):
    torch.manual_seed(seed)
    term = Term()
    x, target = torch.from_numpy(thetas), torch.from_numpy(y)
    scale = target.std(dim=1, keepdim=True)
    opt = torch.optim.Adam(term.parameters(), lr=lr)
    for _ in range(iters):
        opt.zero_grad()
        loss = torch.mean(((rollout(x, term) - target) / scale) ** 2)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(term.parameters(), 1.0)
        opt.step()
    return term, float(loss)


def main():
    torch.set_num_threads(1)
    train = sobol_thetas(512, seed=DEV_SEED)
    test = sobol_thetas(500, seed=DEV_SEED + 1)
    design = sobol_thetas(56, seed=DEV_SEED * 1000)
    f_nom_test = simulate_batch(test, "S1", 0.0)
    with torch.no_grad():
        nom_roll = rollout(torch.from_numpy(test)).numpy()
    print(f"RK4 vs reference solver, nominal, median rel error: "
          f"{np.median(rel_errors(nom_roll, f_nom_test)):.2e}", flush=True)

    nominal = Ensemble(n_members=5, base_seed=DEV_SEED).fit(train, simulate_batch(train, "S1", 0.0))
    surr = nominal.predict(test)[0]

    for system, eps in CASES:
        f_true = simulate_batch(test, system, eps)
        y, _ = make_oracle(system, eps, seed=DEV_SEED)(design)
        print(f"\n{system} eps={eps}: uncorrected surrogate {np.median(rel_errors(surr, f_true)):.4f}, "
              f"nominal ODE alone {np.median(rel_errors(f_nom_test, f_true)):.4f}", flush=True)
        for n in (8, 24, 56):
            t0 = time.perf_counter()
            term, loss = fit_term(design[:n], y[:n], seed=DEV_SEED + n)
            with torch.no_grad():
                hybrid = rollout(torch.from_numpy(test), term).numpy()
            corrected = surr + (hybrid - nom_roll)
            print(f"  n={n:2d}: hybrid {np.median(rel_errors(hybrid, f_true)):.4f} | "
                  f"surrogate + delta {np.median(rel_errors(corrected, f_true)):.4f} | "
                  f"p95 {np.quantile(rel_errors(corrected, f_true), 0.95):.4f} | "
                  f"train loss {loss:.2e} | {time.perf_counter() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()

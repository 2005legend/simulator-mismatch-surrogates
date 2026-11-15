"""Differentiable pendulum solver and dynamics-space correction.

The nominal ODE is phi'' = -c phi' - (g/L) sin(phi). `rollout` integrates it with RK4 on the
201-point grid, optionally with an extra term h(phi, phi') added to the acceleration. That
term can be a fixed function (structural perturbation, see alignment.py) or a small network
fitted to observed true trajectories (the dynamics-space correction).

Fitting is two-step: gradient matching (spline derivatives give a direct regression target
for h, no ODE solves), then a few L-BFGS iterations of trajectory matching.
"""

import numpy as np
import torch
from scipy.interpolate import make_smoothing_spline
from torch import nn

from mismatch import config

N_STEPS = config.N_T - 1
DT = config.T_END / N_STEPS
_EDGE = 10   # spline second derivatives are unreliable near the ends


class Term(nn.Module):
    """h(phi, omega). The last layer starts at zero, so an unfitted Term is the nominal ODE."""

    def __init__(self, hidden=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, hidden), nn.SiLU(), nn.Linear(hidden, hidden),
                                 nn.SiLU(), nn.Linear(hidden, 1)).double()
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, phi, omega):
        return self.net(torch.stack([phi / np.pi, omega / 5.0], dim=1))[:, 0]


def rollout(thetas, term=None, return_omega=False):
    """RK4 on the time grid for a batch of thetas (n, 4): phi as a tensor (n, 201), or
    (phi, omega) when return_omega is set."""
    c, length, phi, omega = thetas.unbind(dim=1)
    k = config.G / length

    def f(p, w):
        acc = -c * w - k * torch.sin(p)
        return w, acc + term(p, w) if term is not None else acc

    out, vel = [phi], [omega]
    for _ in range(N_STEPS):
        p1, w1 = f(phi, omega)
        p2, w2 = f(phi + 0.5 * DT * p1, omega + 0.5 * DT * w1)
        p3, w3 = f(phi + 0.5 * DT * p2, omega + 0.5 * DT * w2)
        p4, w4 = f(phi + DT * p3, omega + DT * w3)
        phi = phi + DT / 6 * (p1 + 2 * p2 + 2 * p3 + p4)
        omega = omega + DT / 6 * (w1 + 2 * w2 + 2 * w3 + w4)
        out.append(phi)
        vel.append(omega)
    if return_omega:
        return torch.stack(out, dim=1), torch.stack(vel, dim=1)
    return torch.stack(out, dim=1)


def simulate_with_term(thetas, term=None, return_omega=False):
    """NumPy wrapper around `rollout` without gradients."""
    with torch.no_grad():
        out = rollout(torch.as_tensor(np.atleast_2d(thetas), dtype=torch.float64), term,
                      return_omega)
    return tuple(o.numpy() for o in out) if return_omega else out.numpy()


def _gm_targets(thetas, y, lam):
    xs, hs = [], []
    t = config.T_GRID
    for th, traj in zip(thetas, y):
        spl = make_smoothing_spline(t, traj, lam=lam)
        phi, d1, d2 = spl(t), spl.derivative(1)(t), spl.derivative(2)(t)
        xs.append(np.stack([phi, d1], axis=1)[_EDGE:-_EDGE])
        hs.append((d2 + th[0] * d1 + (config.G / th[1]) * np.sin(phi))[_EDGE:-_EDGE])
    return torch.from_numpy(np.concatenate(xs)), torch.from_numpy(np.concatenate(hs))


def fit_gradient_matching(thetas, y, seed=0, steps=500, lam=None):
    """Regress h on spline-derivative targets. Fast, approximate."""
    torch.manual_seed(seed)
    term = Term()
    nn.init.normal_(term.net[-1].weight, std=0.01)
    x, h = _gm_targets(thetas, y, lam)
    opt = torch.optim.Adam(term.parameters(), lr=1e-2)
    for _ in range(steps):
        opt.zero_grad()
        torch.mean((term(x[:, 0], x[:, 1]) - h) ** 2).backward()
        opt.step()
    return term


def refine_trajectory_matching(term, thetas, y, iters=20):
    """L-BFGS on the mismatch between simulated and observed trajectories."""
    x, target = torch.from_numpy(np.asarray(thetas)), torch.from_numpy(np.asarray(y))
    scale = target.std(dim=1, keepdim=True)
    opt = torch.optim.LBFGS(term.parameters(), lr=1.0, max_iter=iters,
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.mean(((rollout(x, term) - target) / scale) ** 2)
        loss.backward()
        return loss

    opt.step(closure)
    return term


def fit_term(thetas, y, seed=0, lbfgs_iters=20, lam=None, gm_steps=500):
    """Two-step fit of the missing term from observed true trajectories."""
    term = fit_gradient_matching(thetas, y, seed=seed, steps=gm_steps, lam=lam)
    return refine_trajectory_matching(term, thetas, y, iters=lbfgs_iters)

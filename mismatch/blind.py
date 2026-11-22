"""Addendum v1.10: rule-based dictionaries and pools of missing terms (the blind-dictionary test).

The dictionaries used in v1.6-v1.8 were written knowing the true missing terms. Here:

  * the dictionaries are fixed by published default rules, not chosen by hand:
      PEND_LIB  SINDy default library (Brunton et al. 2016) for the pendulum state (p, w):
                polynomials of degree 1-3 plus sin p, cos p, sin 2p, cos 2p (13 terms);
      BURG_LIB  PDE-FIND default library (Rudy et al. 2017) for Burgers:
                {1, u, u^2, u^3} x {1, u_x, u_xx} (12 terms);
  * the missing terms come from fixed pools (PEND_POOL, BURG_POOL) and are drawn at random
    after this module and its addendum are frozen, with a seed taken from the hash of the
    freeze commit (`draw`), so nobody can choose them. Before the freeze, the pools were
    screened only for solver stability: dispersion u_xxx (diverges from eps = 0.01) and
    hyperviscosity -u_xxxx (diverges at eps = 0.001) with the frozen explicit Burgers solver
    were replaced by a localized source and saturating damping.

Frozen modules are not modified: the pendulum true system is solved with solve_ivp exactly as
`mismatch.simulators` does (same tolerances), with h(t, p, w) added to the acceleration; the
Burgers true system uses `mismatch.burgers.solve` with a term(u, x).
"""

import numpy as np
import torch
from scipy.integrate import solve_ivp

from mismatch import burgers as B
from mismatch import config
from mismatch.burgers_ood import _dx
from mismatch.simulators import simulate_batch as pend_nominal

G = config.G

# ------------------------------------------------------------------ dictionaries (fixed by rule)

PEND_LIB = {
    "p": lambda p, w: p,
    "w": lambda p, w: w,
    "p^2": lambda p, w: p ** 2,
    "p*w": lambda p, w: p * w,
    "w^2": lambda p, w: w ** 2,
    "p^3": lambda p, w: p ** 3,
    "p^2*w": lambda p, w: p ** 2 * w,
    "p*w^2": lambda p, w: p * w ** 2,
    "w^3": lambda p, w: w ** 3,
    "sin(p)": lambda p, w: torch.sin(p),
    "cos(p)": lambda p, w: torch.cos(p),
    "sin(2p)": lambda p, w: torch.sin(2 * p),
    "cos(2p)": lambda p, w: torch.cos(2 * p),
}


def _burg_lib():
    lib = {}
    powers = {"1": lambda u: np.ones_like(u), "u": lambda u: u, "u^2": lambda u: u ** 2, "u^3": lambda u: u ** 3}
    derivs = {"1": lambda u: np.ones_like(u), "u_x": lambda u: _dx(u, 1), "u_xx": lambda u: _dx(u, 2)}
    for pn, pf in powers.items():
        for dn, df in derivs.items():
            name = pn if dn == "1" else (dn if pn == "1" else f"{pn}*{dn}")
            lib[name] = (lambda u, x, pf=pf, df=df: pf(u) * df(u))
    return lib


BURG_LIB = _burg_lib()

# ------------------------------------------------------------------ pools of missing terms

PEND_POOL = {
    "quadratic drag":       lambda t, p, w: -w * abs(w),
    "Coulomb friction":     lambda t, p, w: -G * np.tanh(w / 0.05),
    "Stribeck friction":    lambda t, p, w: -G * np.tanh(w / 0.05) * (0.5 + 0.5 * np.exp(-(w / 0.3) ** 2)),
    "linear damping":       lambda t, p, w: -w,
    "cubic spring":         lambda t, p, w: -p ** 3,
    "constant torque":      lambda t, p, w: G,
    "periodic drive":       lambda t, p, w: G * np.sin(1.5 * t),
    "pivot vibration":      lambda t, p, w: -G * np.cos(5.0 * t) * np.sin(p),
    "aerodynamic drag":     lambda t, p, w: -w * np.sqrt(abs(w)),
    "offset restoring":     lambda t, p, w: -G * np.sin(p - 0.2),
}

BURG_POOL = {
    "quadratic damping":    lambda u, x: -u * np.abs(u),
    "forcing sin 3x":       lambda u, x: np.sin(3 * x) * np.ones_like(u),
    "linear damping":       lambda u, x: -u,
    "drift":                lambda u, x: -_dx(u, 1),
    "extra viscosity":      lambda u, x: _dx(u, 2),
    "localized source":     lambda u, x: np.exp(-4.0 * (x - np.pi) ** 2) * np.ones_like(u),
    "saturating damping":   lambda u, x: -np.tanh(2.0 * u),
    "varying viscosity":    lambda u, x: (1 + 0.5 * np.sin(x)) * _dx(u, 2) + 0.5 * np.cos(x) * _dx(u, 1),
    "forcing cos 2x":       lambda u, x: np.cos(2 * x) * np.ones_like(u),
    "cubic damping":        lambda u, x: -u ** 3,
}

N_DRAW = 3


def draw(commit_hash, n=N_DRAW):
    """The frozen draw rule: a generator seeded by the first 12 hex digits of the part-A freeze
    commit; n pendulum terms, then n Burgers terms, without replacement, followed by the full
    order of each pool (the redraw sequence for terms found numerically unusable)."""
    rng = np.random.default_rng(int(commit_hash[:12], 16))
    pend = [list(PEND_POOL)[i] for i in rng.permutation(len(PEND_POOL))]
    burg = [list(BURG_POOL)[i] for i in rng.permutation(len(BURG_POOL))]
    return {"seed_hex": commit_hash[:12], "pendulum": pend[:n], "burgers": burg[:n],
            "pendulum_order": pend, "burgers_order": burg}

# ------------------------------------------------------------------ true systems with a pool term


def pend_simulate(theta, term, eps, t_grid=config.T_GRID):
    c, L, phi0, omega0 = (float(v) for v in theta)
    h = PEND_POOL[term]

    def rhs(t, y):
        p, w = y
        return (w, -c * w - (G / L) * np.sin(p) + eps * h(t, p, w))

    sol = solve_ivp(rhs, (t_grid[0], t_grid[-1]), (phi0, omega0), t_eval=t_grid,
                    method="RK45", rtol=config.RTOL, atol=config.ATOL)
    if not sol.success:
        raise RuntimeError(f"solver failed for theta={theta}: {sol.message}")
    return sol.y[0]


def pend_batch(thetas, term, eps=0.0):
    thetas = np.atleast_2d(thetas)
    if eps == 0:
        return pend_nominal(thetas, "S1", 0.0)
    return np.stack([pend_simulate(th, term, eps) for th in thetas])


def pend_response(thetas, term, scale=1e-3):
    return (pend_batch(thetas, term, scale) - pend_nominal(np.atleast_2d(thetas), "S1", 0.0)) / scale


def burg_batch(thetas, term, eps=0.0, chunk=256):
    th = np.atleast_2d(thetas)
    if eps == 0:
        return B.simulate_batch(th)
    f = (lambda u, x, h=BURG_POOL[term]: eps * h(u, x))
    parts = [B.solve(th[i:i + chunk], f) for i in range(0, th.shape[0], chunk)]
    return np.concatenate(parts).reshape(th.shape[0], B.N_OUT)


def burg_response(thetas, term, scale=1e-3):
    return (burg_batch(thetas, term, scale) - B.simulate_batch(np.atleast_2d(thetas))) / scale

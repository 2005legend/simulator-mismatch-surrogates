"""Out-of-dictionary missing terms for the pendulum (draft for addendum v1.9).

The frozen pendulum dictionary (`mismatch.alignment.TERMS`) holds functions of the state
(phi, phi') only. The two terms here are not functions of the state, so their responses are
not in its span:

  Q1  g                  constant torque (tilted pivot or steady wind)
  Q2  g sin(OMEGA_F t)   periodic external drive, below the natural frequencies

The true system adds eps * h to the acceleration. Solver, grid, tolerances and the parameter box
are those of `mismatch.simulators`, which is not modified.
"""

import numpy as np
from scipy.integrate import solve_ivp

from mismatch import config
from mismatch.simulators import simulate_batch as nominal_batch

OMEGA_F = 1.5   # rad/s; natural frequencies sqrt(g/L) lie in [2.2, 4.4] for L in [0.5, 2]

MISSING_OOD = {"Q1": lambda t: config.G,
               "Q2": lambda t: config.G * np.sin(OMEGA_F * t)}


def simulate(theta, term, eps=0.0, t_grid=config.T_GRID):
    """One trajectory of the true system with an OOD term (phi on t_grid)."""
    c, L, phi0, omega0 = (float(v) for v in theta)
    h = MISSING_OOD[term]

    def rhs(t, y):
        phi, omega = y
        return (omega, -c * omega - (config.G / L) * np.sin(phi) + eps * h(t))

    sol = solve_ivp(rhs, (t_grid[0], t_grid[-1]), (phi0, omega0), t_eval=t_grid,
                    method="RK45", rtol=config.RTOL, atol=config.ATOL)
    if not sol.success:
        raise RuntimeError(f"solver failed for theta={theta}: {sol.message}")
    return sol.y[0]


def simulate_batch(thetas, term, eps=0.0):
    """(n, N_T); eps = 0 returns the nominal simulator exactly."""
    thetas = np.atleast_2d(thetas)
    if eps == 0:
        return nominal_batch(thetas, "S1", 0.0)
    return np.stack([simulate(th, term, eps) for th in thetas])


def response(thetas, term, scale=1e-3):
    """Linear response (n, N_T) of the nominal trajectory to the OOD term, per unit coefficient."""
    return (simulate_batch(thetas, term, scale) - nominal_batch(np.atleast_2d(thetas), "S1", 0.0)) / scale

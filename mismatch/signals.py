"""Nominal-ODE residual signals (prereg-v1 sections 6 and 7).

The nominal ODE is phi'' + c phi' + (g/L) sin(phi) = 0. Its residual is evaluated either
on a surrogate prediction (finite differences; RQ1 signal 3, arm 4) or on an observed
true trajectory (smoothing-spline derivatives; arm 7).
"""

import numpy as np
from scipy.interpolate import make_smoothing_spline

from mismatch import config

_EDGE = 2  # points dropped at each end, where one-sided differences are less accurate


def ode_residual(traj, thetas, t_grid=config.T_GRID):
    """Residual of the nominal ODE on trajectories (n, T), by finite differences."""
    traj, thetas = np.atleast_2d(traj), np.atleast_2d(thetas)
    c, length = thetas[:, [0]], thetas[:, [1]]
    dt = t_grid[1] - t_grid[0]
    d1 = np.gradient(traj, dt, axis=1)
    d2 = np.gradient(d1, dt, axis=1)
    return d2 + c * d1 + (config.G / length) * np.sin(traj)


def residual_score(traj, thetas, t_grid=config.T_GRID):
    """Time-averaged residual magnitude, one value per trajectory."""
    return np.mean(np.abs(ode_residual(traj, thetas, t_grid)[:, _EDGE:-_EDGE]), axis=1)


def spline_residual_score(traj, thetas, t_grid=config.T_GRID, lam=None):
    """Same score on noisy observed trajectories, with derivatives from a smoothing spline.

    lam=None selects the smoothing by generalised cross-validation per trajectory; the
    pilot fixes a single value (prereg-v1 section 9).
    """
    traj, thetas = np.atleast_2d(traj), np.atleast_2d(thetas)
    out = np.empty(len(traj))
    for i, (y, th) in enumerate(zip(traj, thetas)):
        spl = make_smoothing_spline(t_grid, y, lam=lam)
        res = (spl.derivative(2)(t_grid) + th[0] * spl.derivative(1)(t_grid)
               + (config.G / th[1]) * np.sin(spl(t_grid)))
        out[i] = np.mean(np.abs(res[_EDGE:-_EDGE]))
    return out

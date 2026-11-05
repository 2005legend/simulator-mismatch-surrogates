"""Nominal and true pendulum simulators.

Nominal:  phi'' + c phi' + (g/L) sin(phi) = 0
S1 true:  nominal - eps * phi' |phi'|              (quadratic drag)
S2 true:  nominal - eps * g * tanh(phi' / delta)   (smoothed Coulomb friction)

eps = 0 gives the nominal simulator exactly.
"""

from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats import qmc

from mismatch import config


def _missing_term(system, eps, omega, delta):
    if eps == 0.0:
        return 0.0
    if system == "S1":
        return -eps * omega * abs(omega)
    if system == "S2":
        return -eps * config.G * np.tanh(omega / delta)
    raise ValueError(f"unknown system {system!r}")


def simulate(theta, system="S1", eps=0.0, delta=config.DELTA_S2, t_grid=config.T_GRID,
             return_omega=False):
    """Solve one trajectory. theta = (c, L, phi0, omega0). Returns phi on t_grid."""
    c, L, phi0, omega0 = (float(v) for v in theta)

    def rhs(_t, y):
        phi, omega = y
        acc = -c * omega - (config.G / L) * np.sin(phi) + _missing_term(system, eps, omega, delta)
        return (omega, acc)

    sol = solve_ivp(rhs, (t_grid[0], t_grid[-1]), (phi0, omega0), t_eval=t_grid,
                    method="RK45", rtol=config.RTOL, atol=config.ATOL)
    if not sol.success:
        raise RuntimeError(f"solver failed for theta={theta}: {sol.message}")
    return (sol.y[0], sol.y[1]) if return_omega else sol.y[0]


def _simulate_args(args):
    theta, system, eps, delta = args
    return simulate(theta, system, eps, delta)


def simulate_batch(thetas, system="S1", eps=0.0, delta=config.DELTA_S2, n_jobs=1):
    """Solve many trajectories; returns an array of shape (n, N_T)."""
    thetas = np.atleast_2d(thetas)
    jobs = [(th, system, eps, delta) for th in thetas]
    if n_jobs == 1:
        out = [_simulate_args(j) for j in jobs]
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            out = list(ex.map(_simulate_args, jobs, chunksize=max(1, len(jobs) // (4 * n_jobs))))
    return np.stack(out)


def sobol_thetas(n, seed, low=config.BOX_LOW, high=config.BOX_HIGH, skip=0):
    """Scrambled Sobol points in the parameter box. `skip` continues a sequence."""
    sampler = qmc.Sobol(d=len(low), scramble=True, seed=seed)
    if skip:
        sampler.fast_forward(skip)
    return qmc.scale(sampler.random(n), low, high)


def add_noise(trajectories, frac, rng):
    """Gaussian noise with std = frac * per-trajectory std of phi."""
    traj = np.atleast_2d(trajectories)
    scale = frac * traj.std(axis=1, keepdims=True)
    return traj + rng.normal(size=traj.shape) * scale

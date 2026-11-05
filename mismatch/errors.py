"""Error decomposition and the dial r (prereg-v1 section 5, v1.1 section 2, v1.2 section 1).

For trajectories on a shared time grid, arrays have shape (n_points, n_times):
    a = f_hat_nom - f_nom      approximation error
    d = f_nom - f_true         structural discrepancy
    e = f_hat_nom - f_true     total error, e = a + d exactly
Norms are relative L2 over time, divided by ||f_true||.
"""

import numpy as np

from mismatch import config

_TINY = 1e-12


def decompose(f_hat_nom, f_nom, f_true):
    a = f_hat_nom - f_nom
    d = f_nom - f_true
    e = f_hat_nom - f_true
    return a, d, e


def rel_norm(x, ref):
    """Row-wise ||x|| / ||ref||."""
    return np.linalg.norm(x, axis=-1) / (np.linalg.norm(ref, axis=-1) + _TINY)


def ratio_r(a, d, f_true):
    """r = median ||d|| / median ||a|| over the test set (relative norms)."""
    return float(np.median(rel_norm(d, f_true)) / (np.median(rel_norm(a, f_true)) + _TINY))


def smoothness_proxy(d, t_grid=config.T_GRID):
    """Median over points of the discrete H1-seminorm / L2 ratio of d (reported, not tested)."""
    dt = np.diff(t_grid)
    grad = np.diff(d, axis=-1) / dt
    h1 = np.linalg.norm(grad, axis=-1)
    l2 = np.linalg.norm(d, axis=-1) + _TINY
    return float(np.median(h1 / l2))


def linearity_error(d_eps, d_ref, eps, eps_ref):
    """Median ||d(eps) - (eps/eps_ref) d(eps_ref)|| / ||d(eps)|| (reported with P-A/P-B)."""
    if eps == 0.0:
        return 0.0
    approx = (eps / eps_ref) * d_ref
    return float(np.median(np.linalg.norm(d_eps - approx, axis=-1)
                           / (np.linalg.norm(d_eps, axis=-1) + _TINY)))


def horizon_slice(x, horizon, t_grid=config.T_GRID):
    """Prefix window t in [0, horizon] along the last axis (G1 windows)."""
    n = int(np.searchsorted(t_grid, horizon, side="right"))
    return x[..., :n]


def fraction_after(d, t_split, t_grid=config.T_GRID):
    """Fraction of ||d||^2 accumulated after t_split (G1 descriptive statement)."""
    k = int(np.searchsorted(t_grid, t_split, side="right"))
    total = (d**2).sum(axis=-1) + _TINY
    return float(np.median((d[..., k:] ** 2).sum(axis=-1) / total))

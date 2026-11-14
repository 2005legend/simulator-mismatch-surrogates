"""Gaussian-process models: ARD Matern-5/2 on scaled theta (prereg-v1 section 4).

ScalarGP regresses one number per point (arms 7 and 9). PCAGP models trajectory residuals
through independent GPs on their leading PCA components (arm 6, GP reference).
"""

import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from mismatch.surrogate import scale_inputs


def _gpr(n_dim, seed):
    kernel = (ConstantKernel(1.0, (1e-3, 1e3))
              * Matern(length_scale=np.ones(n_dim), length_scale_bounds=(0.05, 20.0), nu=2.5)
              + WhiteKernel(1e-3, (1e-8, 1e1)))
    return GaussianProcessRegressor(kernel, normalize_y=True, random_state=seed)


def _fit(gp, x, y):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        return gp.fit(x, y)


class ScalarGP:
    def __init__(self, seed=0):
        self.seed, self.gp = seed, None

    def fit(self, thetas, y):
        x = scale_inputs(thetas)
        self.gp = _fit(_gpr(x.shape[1], self.seed), x, np.asarray(y, dtype=float))
        return self

    def predict(self, thetas, return_std=False):
        return self.gp.predict(scale_inputs(thetas), return_std=return_std)


class PCAGP:
    """Independent GPs on the leading PCA components of (n, T) targets.

    n_components=None keeps the smallest number explaining `var_keep` of the variance,
    capped at `max_components`; the pilot fixes the number (prereg-v1 section 9).
    """

    def __init__(self, n_components=None, var_keep=0.999, max_components=12, seed=0):
        self.n_components, self.var_keep = n_components, var_keep
        self.max_components, self.seed = max_components, seed

    def fit(self, thetas, targets):
        x = scale_inputs(thetas)
        self.mean_ = targets.mean(axis=0)
        _, sing, vt = np.linalg.svd(targets - self.mean_, full_matrices=False)
        ratio = np.cumsum(sing ** 2) / np.sum(sing ** 2)
        n_comp = self.n_components or int(np.searchsorted(ratio, self.var_keep) + 1)
        n_comp = max(1, min(n_comp, self.max_components, len(targets) - 1))
        self.basis_ = vt[:n_comp]                                   # orthonormal rows
        scores = (targets - self.mean_) @ self.basis_.T
        self.gps_ = [_fit(_gpr(x.shape[1], self.seed + j), x, scores[:, j])
                     for j in range(n_comp)]
        return self

    def predict(self, thetas):
        """Returns (mean (n, T), total predictive variance (n,))."""
        x = scale_inputs(thetas)
        out = [gp.predict(x, return_std=True) for gp in self.gps_]
        mu = np.stack([m for m, _ in out], axis=1)
        var = np.sum(np.stack([s for _, s in out], axis=1) ** 2, axis=1)
        return self.mean_ + mu @ self.basis_, var

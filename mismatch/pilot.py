"""Pilot rules (prereg-v1 section 9, addenda v1.1 and v1.6) as plain functions.

The orchestrator in `scripts/run_pilot.py` runs the pilot seeds and applies these rules; the
functions take arrays and return numbers, so each rule can be tested on its own.

Rules covered here:
  eps_ref        the eps at which r = median||d|| / median||a|| equals 1       -> log_bisect
  spline smoothing  generalised cross-validation on the pilot's true trajectories -> gcv_lambda
  GP components  smallest number keeping 99.9% of the residual variance         -> pca_components
  batch exponent k  the value in {1, 2, 4} with the lowest mean budget-AUC      -> choose_k
  MLP settings   the candidate with the lowest validation loss                  -> choose_hyperparameters
  seed count     10, raised to at most 20 if power for half the pilot slope < 80% -> seed_count_rule
"""

import numpy as np
from scipy import stats
from scipy.interpolate import make_smoothing_spline

from mismatch import config

K_CANDIDATES = (1, 2, 4)
# One-at-a-time variations around the default network (FitConfig); label -> overrides.
HYPER_GRID = {
    "default": {},
    "hidden64": {"hidden": 64},
    "hidden128": {"hidden": 128},
    "layers2": {"n_hidden_layers": 2},
    "layers4": {"n_hidden_layers": 4},
    "lr3e-3": {"lr": 3e-3},
    "epochs4000": {"max_epochs": 4000},
}


def log_bisect(f, target, lo, hi, n_iter=14):
    """x in [lo, hi] with f(x) close to target, for increasing f, by bisection on log x.

    Raises ValueError if the target is not bracketed: f(lo) <= target <= f(hi) must hold.
    """
    flo, fhi = f(lo), f(hi)
    if not (flo <= target <= fhi):
        raise ValueError(f"target {target} is not bracketed: f({lo}) = {flo}, f({hi}) = {fhi}")
    a, b = np.log(lo), np.log(hi)
    for _ in range(n_iter):
        m = 0.5 * (a + b)
        if f(float(np.exp(m))) < target:
            a = m
        else:
            b = m
    return float(np.exp(0.5 * (a + b)))


def gcv_lambda(trajs, t_grid=config.T_GRID, grid=None):
    """Smoothing parameter of the cubic smoothing spline chosen by generalised cross-validation.

    GCV(lam) = n RSS(lam) / (n - tr H(lam))^2, with H the smoother matrix (its trace depends
    only on the time grid). Each trajectory's curve is divided by its own total variance so
    that all trajectories count equally; the returned lam minimises the sum over trajectories.
    """
    grid = np.logspace(-9, 0, 19) if grid is None else np.asarray(grid)
    n = len(t_grid)
    eye = np.eye(n)
    trace = np.array([sum(make_smoothing_spline(t_grid, eye[j], lam=lam)(t_grid)[j]
                          for j in range(n)) for lam in grid])
    total = np.zeros(len(grid))
    for y in np.atleast_2d(trajs):
        scale = np.sum((y - y.mean()) ** 2)
        for i, lam in enumerate(grid):
            rss = np.sum((y - make_smoothing_spline(t_grid, y, lam=lam)(t_grid)) ** 2)
            total[i] += n * rss / (n - trace[i]) ** 2 / scale
    return float(grid[int(np.argmin(total))])


def pca_components(residuals, var_keep=0.999, max_components=12):
    """Smallest number of principal components keeping `var_keep` of the variance of
    (n, T) residuals, capped at `max_components` and at n - 1."""
    centered = np.asarray(residuals, dtype=float)
    centered = centered - centered.mean(axis=0)
    sing = np.linalg.svd(centered, compute_uv=False)
    ratio = np.cumsum(sing ** 2) / np.sum(sing ** 2)
    k = int(np.searchsorted(ratio, var_keep) + 1)
    return max(1, min(k, max_components, len(centered) - 1))


def choose_k(auc_by_k):
    """k with the lowest mean budget-AUC. `auc_by_k` maps k to the list of budget-AUC values
    over arms 3-7, pilot seeds and systems. Returns (best k, mean per k)."""
    means = {k: float(np.mean(v)) for k, v in auc_by_k.items()}
    return min(means, key=means.get), means


def choose_hyperparameters(val_loss_by_label):
    """Label with the lowest mean validation loss. Returns (label, mean loss per label)."""
    means = {k: float(np.mean(v)) for k, v in val_loss_by_label.items()}
    return min(means, key=means.get), means


def paired_t_power(effect, sd, n, alpha):
    """Two-sided power of a one-sample t-test on n paired differences with true mean
    `effect` and standard deviation `sd`, at level `alpha` (noncentral t)."""
    df = n - 1
    tcrit = stats.t.ppf(1 - alpha / 2, df)
    ncp = abs(effect) * np.sqrt(n) / sd
    return float(stats.nct.sf(tcrit, df, ncp) + stats.nct.cdf(-tcrit, df, ncp))


def seed_count_rule(slopes, alpha=0.05 / 6, n_min=10, n_max=20, target=0.8):
    """The frozen seed-count rule (prereg-v1 section 9): 10 seeds, raised to at most 20 if the
    pilot's paired SD of the A1 slope gives under 80% power to detect a slope of half the pilot
    mean at alpha = 0.05 / 6. `slopes` are the per-pilot-seed slopes (at least 2)."""
    slopes = np.asarray(slopes, dtype=float)
    if len(slopes) < 2:
        raise ValueError("need at least two pilot slopes for a standard deviation")
    mean, sd = float(slopes.mean()), float(slopes.std(ddof=1))
    effect = 0.5 * abs(mean)
    if sd < 1e-12 * max(abs(mean), 1e-12) or effect == 0:   # no spread (float noise included)
        return {"mean": mean, "sd": sd, "effect": effect, "power_at_n_min": None,
                "n_seeds": n_min, "power_at_n": None, "reached_target": effect > 0}
    n = next((m for m in range(n_min, n_max + 1)
              if paired_t_power(effect, sd, m, alpha) >= target), n_max)
    return {"mean": mean, "sd": sd, "effect": effect,
            "power_at_n_min": paired_t_power(effect, sd, n_min, alpha), "n_seeds": n,
            "power_at_n": paired_t_power(effect, sd, n, alpha),
            "reached_target": paired_t_power(effect, sd, n, alpha) >= target}


def ols_slope(x, y):
    """Least-squares slope of y on x."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    return float(np.polyfit(x, y, 1)[0])

"""Model A: predicted RQ1 ranking transition (prereg addendum v1.1 section 2, v1.4, v1.2 P-C).

A signal s ranks the total error e = a + d. Model A predicts how its AUROC (event ||e|| above
its 80th percentile) moves with r = median||d|| / median||a||, from inputs measured on pilot
seed 0 only: the eps = 0 surrogate (a, s) and the simulator (d), never a surrogate trained or
scored at eps > 0.

Two variants, both computed:
  gaussian   (log s, log||a||^2, log||d||^2) trivariate Gaussian; log||e||^2 linearised at the
             medians with weight w = 1 / (1 + r^2) on log||a||^2; AUROC by Monte Carlo.
  empirical  the frozen fallback: recombine the per-point arrays, e_i = a_i + d_i(eps), and
             compute AUROC directly.
The fallback rule decides which one is the prediction: empirical if s_A > 1, s_D > 1 or the
|a-g cosine| > 0.3 on pilot seed 0.

Norms are relative L2 over the trajectory divided by ||f_true|| (as in RQ1).
"""

import numpy as np
from scipy.optimize import curve_fit

from mismatch.rq1 import auroc

FALLBACK_SD = 1.0
FALLBACK_COS = 0.3


def rel_norms(x, f_true):
    return np.linalg.norm(x, axis=1) / np.linalg.norm(f_true, axis=1)


def gaussian_inputs(signal, A, D):
    """s_A, s_D and the three correlations of (log s, log A^2, log D^2)."""
    S, X, Y = np.log(signal), np.log(A ** 2), np.log(D ** 2)
    c = np.corrcoef(np.stack([S, X, Y]))
    return {"s_A": float(X.std(ddof=1)), "s_D": float(Y.std(ddof=1)),
            "rho_sa": float(c[0, 1]), "rho_sg": float(c[0, 2]), "rho_ag": float(c[1, 2])}


def cosines(a, d):
    """Per-point cosine between a_i and d_i, and the cosine at the median-norm point (the point
    whose ||a|| ||d|| is the median); both are reported and both feed the fallback rule."""
    na, nd = np.linalg.norm(a, axis=1), np.linalg.norm(d, axis=1)
    cos = np.sum(a * d, axis=1) / (na * nd + 1e-300)
    i = int(np.argsort(na * nd)[len(cos) // 2])
    return {"median_cos": float(np.median(cos)), "median_abs_cos": float(np.median(np.abs(cos))),
            "median_norm_point_cos": float(cos[i])}


def use_fallback(inp, cos):
    """Frozen rule: the closed-form linearisation is not trusted if s_A or s_D exceeds 1 or the
    |a-g cosine| exceeds 0.3. Both cosine readings are checked (the conservative choice)."""
    big_cos = max(abs(cos["median_cos"]), abs(cos["median_norm_point_cos"])) > FALLBACK_COS
    return bool(inp["s_A"] > FALLBACK_SD or inp["s_D"] > FALLBACK_SD or big_cos)


def gaussian_auroc(inp, r, n_draws=10 ** 6, seed=0, quantile=0.8):
    """Monte-Carlo AUROC of s for the event Z > its quantile, Z = w X + (1 - w) Y."""
    rng = np.random.default_rng(seed)
    sd = np.array([1.0, inp["s_A"], inp["s_D"]])
    corr = np.array([[1.0, inp["rho_sa"], inp["rho_sg"]],
                     [inp["rho_sa"], 1.0, inp["rho_ag"]],
                     [inp["rho_sg"], inp["rho_ag"], 1.0]])
    S, X, Y = rng.multivariate_normal(np.zeros(3), corr * np.outer(sd, sd), size=n_draws,
                                      method="eigh").T
    w = 1.0 / (1.0 + r ** 2)
    return auroc(S, w * X + (1 - w) * Y, quantile)


def closed_form_midpoint(s_A, s_D):
    """No-alignment midpoint r_1/2 = 3^(1/4) (s_A / s_D)^(1/2) (v1.1 section 2)."""
    return float(3 ** 0.25 * np.sqrt(s_A / s_D))


def empirical_auroc(signal, a, d, f_true, quantile=0.8):
    """Fallback plug-in: AUROC of the signal for the recombined error e_i = a_i + d_i."""
    return auroc(signal, rel_norms(a + d, f_true), quantile)


def four_pl(x, lo, hi, mid, slope):
    """4PL in x = log r: hi at small r, lo at large r, midpoint mid (slope > 0 means falling)."""
    return lo + (hi - lo) / (1.0 + np.exp(slope * (x - mid)))


def fit_4pl(r, auc):
    """4PL fit of AUROC on log r. Returns r_1/2 = exp(mid), the slope (natural-log units, signed:
    positive when AUROC falls with r), the plateaus, and `converged` (fit succeeded and the
    midpoint lies inside the r range)."""
    x, y = np.log(np.asarray(r, float)), np.asarray(auc, float)
    order = np.argsort(x)
    x, y = x[order], y[order]
    mid0 = x[np.argmin(np.abs(y - 0.5 * (y[0] + y[-1])))]
    try:
        p, _ = curve_fit(four_pl, x, y, p0=[y[-1], y[0], mid0, 2.0 * np.sign(y[0] - y[-1] or 1)],
                         maxfev=20000)
    except RuntimeError:
        return {"converged": False}
    lo, hi, mid, slope = (float(v) for v in p)
    small, large = (hi, lo) if slope > 0 else (lo, hi)      # plateaus by side, whatever the sign
    return {"converged": bool(x[0] <= mid <= x[-1]), "r_half": float(np.exp(mid)),
            "slope": slope, "lo": lo, "hi": hi, "plateau_small_r": small, "plateau_large_r": large}


def blindness_horizon(windows, auc_by_window, auc0_by_window, plateau_by_window):
    """P-C: smallest window whose AUROC is below the midpoint between that window's eps = 0
    AUROC and its large-r plateau; '> 10 s' if none."""
    for h in windows:
        if auc_by_window[h] < 0.5 * (auc0_by_window[h] + plateau_by_window[h]):
            return h
    return "> 10 s"

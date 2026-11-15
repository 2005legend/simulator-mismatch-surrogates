"""RQ1: how well do uncertainty signals rank the surrogate's error on the true system?

Signals (prereg-v1 section 6) for the nominal surrogate predicting the true system:
  1 ens_sigma    nominal ensemble sigma (mean over time of the per-point std)
  2 mc_sigma     MC-dropout sigma from a separate dropout-trained surrogate
  3 ode_resid    nominal-ODE residual of the surrogate's prediction
  4 conf_nom     split conformal calibrated on nominal simulations (coverage and width only)
  5 conf_true    split conformal calibrated on n_cal true points (coverage and width only)
  6 disc_sigma   discrepancy-ensemble sigma after fitting n true points
  7 gp_std       GP discrepancy predictive std, same n

Each ranking signal is scored against the total error ||e|| (AUROC for the event
||e|| > its 80th percentile, Spearman, AUSE) and against ||a|| and ||d|| alone.
Norms are relative L2 over the trajectory, normalised by ||f_true||.
"""

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from mismatch.acquisition import DISC_CFG
from mismatch.errors import decompose, rel_norm
from mismatch.gp import PCAGP
from mismatch.signals import residual_score
from mismatch.simulators import simulate_batch, sobol_thetas, stream_seed
from mismatch.surrogate import Ensemble

N_CAL_TRUE = (8, 16, 32, 64)
N_CAL_NOMINAL = 200
COVER = 0.90
Z90 = 1.645       # one-sided Gaussian multiplier used for sigma-based intervals


def auroc(score, error, quantile=0.8):
    """AUROC of `score` for the event error > its `quantile` (ranking within the cell)."""
    event = error > np.quantile(error, quantile)
    return float(roc_auc_score(event, score))


def spearman(a, b):
    return float(spearmanr(a, b)[0])


def ause(signal, error, n_steps=20):
    """Area between the sparsification curve of `signal` and the oracle curve.

    The curve is the mean error of the points kept after removing the fraction f with the
    highest signal (oracle: highest error). Normalised by the mean error with nothing removed.
    """
    error = np.asarray(error, dtype=float)
    n = len(error)

    def curve(order):
        sorted_err = error[order]
        tail_mean = np.cumsum(sorted_err[::-1])[::-1] / np.arange(n, 0, -1)
        return np.array([tail_mean[int(f * n)] for f in np.linspace(0, 0.95, n_steps)])

    by_signal = curve(np.argsort(-np.asarray(signal)))      # remove highest signal first
    oracle = curve(np.argsort(-error))
    return float(np.mean(by_signal - oracle) / error.mean())


def conformal_quantile(scores, level=COVER):
    """Finite-sample split-conformal quantile of nonconformity scores."""
    scores = np.sort(np.asarray(scores))
    k = int(np.ceil((len(scores) + 1) * level))
    return float(scores[min(k, len(scores)) - 1])


def sigma_rel(sigma_traj, f_hat):
    """Relative norm of a per-time sigma against the prediction (scale for intervals)."""
    return np.linalg.norm(sigma_traj, axis=1) / np.linalg.norm(f_hat, axis=1)


def _rank_record(name, n_cal, signal, E, A, D, r):
    return {"signal": name, "n_cal": n_cal, "r": r,
            "auroc_e": auroc(signal, E), "spearman_e": spearman(signal, E),
            "ause_e": ause(signal, E),
            "auroc_a": auroc(signal, A), "spearman_a": spearman(signal, A),
            "auroc_d": auroc(signal, D), "spearman_d": spearman(signal, D)}


def _interval_record(name, n_cal, scale, E, q):
    return {"signal": name, "n_cal": n_cal, "coverage": float(np.mean(E <= q * scale)),
            "width": float(np.mean(q * scale))}


def rq1_cell(nominal, mc_model, test_thetas, f_true, oracle, system_seed, include_true_data=True,
             n_mc=50, disc_members=5, n_cal_true=N_CAL_TRUE, nominal_sim=None):
    """All RQ1 records for one (system, eps, n_nom, seed) cell.

    `f_true` are noise-free true trajectories on the test set; `oracle` returns noisy true
    trajectories for calibration points. `nominal_sim(thetas)` simulates the nominal system.
    With include_true_data=False only the simulator-only signals (1-4) are computed.
    """
    nominal_sim = nominal_sim or (lambda th: simulate_batch(th, "S1", 0.0))
    f_nom = nominal_sim(test_thetas)
    mean, sigma = nominal.predict(test_thetas)
    a, d, e = decompose(mean, f_nom, f_true)
    A, D, E = (rel_norm(x, f_true) for x in (a, d, e))
    r = float(np.median(D) / np.median(A))

    ens = sigma.mean(axis=1)
    mc = mc_model.members[0].predict_mc(test_thetas, n_passes=n_mc, seed=system_seed).std(axis=0)
    signals = {"ens_sigma": ens, "mc_sigma": mc.mean(axis=1),
               "ode_resid": residual_score(mean, test_thetas)}
    out = [_rank_record(name, 0, sig, E, A, D, r) for name, sig in signals.items()]

    # Gaussian intervals from sigma (relative norm), coverage against the total error
    for name, traj_sigma in (("ens_sigma", sigma), ("mc_sigma", mc)):
        out.append(_interval_record(name + "_gauss", 0, sigma_rel(traj_sigma, mean), E, Z90))

    # split conformal on nominal simulations, normalised by ensemble sigma
    cal_th = sobol_thetas(N_CAL_NOMINAL, seed=stream_seed(system_seed, "calibration", 0))
    cal_mean, cal_sigma = nominal.predict(cal_th)
    cal_nom = nominal_sim(cal_th)
    cal_err = rel_norm(cal_mean - cal_nom, cal_nom)
    q = conformal_quantile(cal_err / sigma_rel(cal_sigma, cal_mean))
    out.append(_interval_record("conf_nom", 0, sigma_rel(sigma, mean), E, q))

    if not include_true_data:
        return out

    cal_all = sobol_thetas(max(n_cal_true), seed=stream_seed(system_seed, "calibration", 1))
    y_all, _ = oracle(cal_all)
    resid_all = y_all - nominal.predict(cal_all)[0]
    for n in n_cal_true:
        x_cal, y_cal, resid = cal_all[:n], y_all[:n], resid_all[:n]
        # conformal on n true points, same sigma scale
        c_mean, c_sigma = nominal.predict(x_cal)
        c_err = rel_norm(c_mean - y_cal, y_cal)
        q_true = conformal_quantile(c_err / sigma_rel(c_sigma, c_mean))
        out.append(_interval_record("conf_true", n, sigma_rel(sigma, mean), E, q_true))
        # discrepancy ensemble and GP fitted on the n residuals
        disc = Ensemble(n_out=resid.shape[1], n_members=disc_members, cfg=DISC_CFG,
                        base_seed=system_seed * 10000 + n).fit(x_cal, resid)
        out.append(_rank_record("disc_sigma", n, disc.predict(test_thetas)[1].mean(axis=1),
                                E, A, D, r))
        gp = PCAGP(seed=system_seed).fit(x_cal, resid)
        out.append(_rank_record("gp_std", n, np.sqrt(gp.predict(test_thetas)[1]), E, A, D, r))
    return out

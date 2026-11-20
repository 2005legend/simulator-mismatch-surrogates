"""Frozen statistics for the main analysis (prereg-v1 section 8, addenda v1.1, v1.5, v1.6).

The unit of replication is the seed: every test takes one value per seed.
  paired_t      two-sided one-sample t-test of per-seed values (differences or slopes) against 0
  holm          Holm step-down adjustment within one family
  bca_ci        BCa bootstrap CI of the mean over seeds (10,000 resamples)
  sign_test     exact two-sided sign test (robustness check)
  tost          two one-sided tests for equivalence within +/- margin
  binom_ci      exact (Clopper-Pearson) interval for a count
  family        Holm within a family plus the frozen decision rule: supported iff the Holm p is
                below alpha and the 95% BCa CI excludes 0
"""

import numpy as np
from scipy import stats

N_BOOT = 10_000
BOOT_SEED = 20261120


def paired_t(x):
    x = np.asarray(x, float)
    t, p = stats.ttest_1samp(x, 0.0)
    return {"n": int(x.size), "mean": float(x.mean()), "sd": float(x.std(ddof=1)), "t": float(t),
            "p": float(p)}


def holm(pvals):
    """Holm-adjusted p-values, same order as the input."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = p.size
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def bca_ci(x, level=0.95, n_boot=N_BOOT, seed=BOOT_SEED, statistic=np.mean):
    x = np.asarray(x, float)
    if np.ptp(x) == 0:
        v = float(statistic(x))
        return (v, v)
    res = stats.bootstrap((x,), statistic, n_resamples=n_boot, confidence_level=level,
                          method="BCa", random_state=np.random.default_rng(seed))
    return (float(res.confidence_interval.low), float(res.confidence_interval.high))


def percentile_ci(values, level=0.90):
    """Percentile interval of an array of bootstrap replicates."""
    a = (1 - level) / 2
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    return (float(np.quantile(v, a)), float(np.quantile(v, 1 - a)))


def boot_indices(n, n_boot=N_BOOT, seed=BOOT_SEED):
    """Seed-level resampling indices (shared across statistics so CIs are paired)."""
    return np.random.default_rng(seed).integers(0, n, size=(n_boot, n))


def sign_test(x):
    x = np.asarray(x, float)
    x = x[x != 0]
    k = int((x > 0).sum())
    return {"n_pos": k, "n_nonzero": int(x.size),
            "p": float(stats.binomtest(k, x.size, 0.5).pvalue) if x.size else 1.0}


def tost(x, margin, alpha=0.05):
    """Equivalence of mean(x) to 0 within +/- margin: both one-sided t-tests at alpha."""
    x = np.asarray(x, float)
    n, m, se = x.size, x.mean(), x.std(ddof=1) / np.sqrt(x.size)
    if se == 0:
        p_lo = p_hi = 0.0 if abs(m) < margin else 1.0
    else:
        p_lo = float(stats.t.sf((m + margin) / se, n - 1))     # H0: mean <= -margin
        p_hi = float(stats.t.cdf((m - margin) / se, n - 1))    # H0: mean >= +margin
    p = max(p_lo, p_hi)
    return {"mean": float(m), "margin": margin, "p_lower": p_lo, "p_upper": p_hi, "p": p,
            "equivalent": bool(p < alpha)}


def binom_ci(k, n, level=0.95):
    ci = stats.binomtest(int(k), int(n)).proportion_ci(confidence_level=level, method="exact")
    return (float(ci.low), float(ci.high))


def ols_slope(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    xc = x - x.mean()
    return float(np.sum(xc * (y - y.mean())) / np.sum(xc ** 2))


def test_report(x):
    """Paired t, 95% BCa CI and sign test for one set of per-seed values."""
    return {**paired_t(x), "ci95_bca": bca_ci(x), "sign": sign_test(x),
            "values": [float(v) for v in x]}


def family(reports, directions, alpha=0.05):
    """Apply Holm within a family {name: test_report} and the frozen decision rule.

    directions[name] is +1 (hypothesis: mean > 0) or -1 (mean < 0). A test is supported only if
    its Holm-adjusted p is below alpha and the whole 95% BCa CI lies on the hypothesised side of
    0; a significant effect in the other direction is reported as `opposite`, never supported.
    """
    names = list(reports)
    adj = holm([reports[n]["p"] for n in names])
    out = {}
    for n, pa in zip(names, adj):
        r, d = dict(reports[n]), directions[n]
        lo, hi = r["ci95_bca"]
        side = lo > 0 if d > 0 else hi < 0
        other = hi < 0 if d > 0 else lo > 0
        r.update(p_holm=pa, direction=d, supported=bool(pa < alpha and side),
                 opposite=bool(pa < alpha and other))
        out[n] = r
    return out

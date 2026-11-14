"""RQ2 metrics (prereg-v1 section 7)."""

import numpy as np


def rel_errors(pred, truth):
    """Relative L2 error per trajectory."""
    return np.linalg.norm(pred - truth, axis=1) / np.linalg.norm(truth, axis=1)


def summarize(err):
    return {"median": float(np.median(err)), "p95": float(np.quantile(err, 0.95)),
            "p99": float(np.quantile(err, 0.99))}


def budget_auc(ns, median_errors):
    """Budget-AUC of the median error: trapezoid on log error over the budget n.

    Returned as exp(area / budget range), i.e. the geometric-mean error across the
    budget, so it is positive and the ratio of two arms' values is meaningful. The log of
    that ratio is the mean difference in log error.
    """
    ns = np.asarray(ns, dtype=float)
    log_err = np.log(np.asarray(median_errors, dtype=float))
    area = np.sum(0.5 * (log_err[1:] + log_err[:-1]) * np.diff(ns))
    return float(np.exp(area / (ns[-1] - ns[0])))

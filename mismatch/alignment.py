"""Alignment range: a no-real-data stress test for an uncertainty signal.

For a small missing term eps*g, the discrepancy is the simulator's linear response to g.
That response is computable from the simulator alone for any hypothesised g. Given a
dictionary of plausible missing terms, the alignment range of a signal s is the smallest
and largest rank correlation between s and the response norm, over single terms and over
sparse mixtures of them. A wide range means the signal's usefulness under mismatch depends
on physics the simulator cannot reveal.
"""

import numpy as np
import torch
from scipy.stats import rankdata

from mismatch.dynamics import simulate_with_term

# Hypothesised missing terms h(phi, omega) added to the acceleration.
TERMS = {
    "w": lambda p, w: w,
    "w|w|": lambda p, w: w * torch.abs(w),
    "w^3": lambda p, w: w ** 3,
    "tanh(w/0.1)": lambda p, w: torch.tanh(w / 0.1),
    "tanh(w/0.5)": lambda p, w: torch.tanh(w / 0.5),
    "p": lambda p, w: p,
    "p^3": lambda p, w: p ** 3,
    "sin(2p)": lambda p, w: torch.sin(2 * p),
    "p*w": lambda p, w: p * w,
}


def term_responses(thetas, terms=None, scale=1e-3):
    """Linear response of the nominal trajectory to each term.

    Returns (names, responses (m, n, T), base (n, T)); responses are per unit coefficient,
    estimated by a finite perturbation of size `scale`.
    """
    terms = terms or TERMS
    base = simulate_with_term(thetas)
    out = []
    for fn in terms.values():
        moved = simulate_with_term(thetas, lambda p, w, f=fn: scale * f(p, w))
        out.append((moved - base) / scale)
    return list(terms), np.stack(out), base


def response_norm(responses, base, coef=None):
    """Relative norm per test point of the response to sum_j coef_j * term_j."""
    resp = responses if coef is None else np.tensordot(coef, responses, axes=1)
    return np.linalg.norm(resp, axis=-1) / np.linalg.norm(base, axis=-1)


def _spearman(rank_signal, values):
    r = rankdata(values)
    a, b = rank_signal - rank_signal.mean(), r - r.mean()
    return float(a @ b / np.sqrt((a @ a) * (b @ b)))


def alignment_range(signal, names, responses, base, n_mix=2000, max_terms=3, seed=0):
    """Spearman correlation of `signal` with the response norm over the dictionary.

    Mixtures are random signed combinations of 2..max_terms terms, each term first scaled to
    unit median response so no single term dominates by its units.
    """
    rs = rankdata(signal)
    single = response_norm(responses, base)                       # (m, n)
    per_term = {n: _spearman(rs, v) for n, v in zip(names, single)}

    unit = responses / np.median(single, axis=1)[:, None, None]
    rng = np.random.default_rng(seed)
    mixes = []
    for _ in range(n_mix if len(names) > 1 else 0):
        k = rng.integers(2, min(max_terms, len(names)) + 1)
        coef = np.zeros(len(names))
        coef[rng.choice(len(names), size=k, replace=False)] = rng.normal(size=k)
        mixes.append(_spearman(rs, response_norm(unit, base, coef)))
    everything = list(per_term.values()) + mixes
    return {"per_term": per_term,
            "term_min": min(per_term.values()), "term_max": max(per_term.values()),
            "min": min(everything), "max": max(everything),
            "mix_quantiles": tuple(np.quantile(mixes, (0.05, 0.5, 0.95))) if mixes else None}

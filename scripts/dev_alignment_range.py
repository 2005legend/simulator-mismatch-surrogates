"""Development run: alignment range of simulator-only signals (dev seed 999, no real data).

Computes, for nominal ensemble sigma and for the nominal-ODE residual of the surrogate
prediction, the smallest and largest Spearman correlation with the linear-response norm
over a dictionary of nine hypothesised missing terms and their sparse mixtures. For
reference it then shows where the two true systems fall (this part uses true simulations
and is only a check of the range, not part of the stress test).

Usage: python -m scripts.dev_alignment_range
"""

import numpy as np
import torch
from scipy.stats import spearmanr

from mismatch.alignment import alignment_range, term_responses
from mismatch.metrics import rel_errors
from mismatch.signals import residual_score
from mismatch.simulators import simulate_batch, sobol_thetas
from mismatch.surrogate import Ensemble

DEV_SEED = 999


def main():
    torch.set_num_threads(1)
    train = sobol_thetas(512, seed=DEV_SEED)
    test = sobol_thetas(500, seed=DEV_SEED + 1)
    nominal = Ensemble(n_members=5, base_seed=DEV_SEED).fit(train, simulate_batch(train, "S1", 0.0))
    mean, sigma = nominal.predict(test)
    signals = {"ensemble sigma": sigma.mean(axis=1), "ODE residual": residual_score(mean, test)}

    names, responses, base = term_responses(test)
    f_nom = simulate_batch(test, "S1", 0.0)
    truth = {"S1": rel_errors(f_nom, simulate_batch(test, "S1", 0.04)),
             "S2": rel_errors(f_nom, simulate_batch(test, "S2", 0.0104))}
    for name, sig in signals.items():
        res = alignment_range(sig, names, responses, base, seed=DEV_SEED)
        q = res["mix_quantiles"]
        print(f"\n{name}: range over single terms [{res['term_min']:+.2f}, {res['term_max']:+.2f}], "
              f"with mixtures [{res['min']:+.2f}, {res['max']:+.2f}], "
              f"mixture 5/50/95%: {q[0]:+.2f} / {q[1]:+.2f} / {q[2]:+.2f}")
        print("  per term: " + ", ".join(f"{n} {v:+.2f}" for n, v in res["per_term"].items()))
        print("  actual correlation with the true discrepancy: "
              + ", ".join(f"{s} {spearmanr(sig, d)[0]:+.2f}" for s, d in truth.items()), flush=True)


if __name__ == "__main__":
    main()

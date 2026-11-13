"""Development timing check (not the pre-registered pilot).

Uses dev seed 999 only: never the pilot seeds (0, 101, 102) or main seeds (1-10).
Measures single-thread fit time of one nominal surrogate member (n_nom = 512) and one
discrepancy member (56 points), and the approximation error of the nominal member.

Usage: python -m scripts.dev_timing_pilot
"""

import time

import numpy as np
import torch

from mismatch.errors import rel_norm
from mismatch.simulators import simulate_batch, sobol_thetas
from mismatch.surrogate import Ensemble, FitConfig

DEV_SEED = 999


def main():
    torch.set_num_threads(1)
    train = sobol_thetas(512, seed=DEV_SEED)
    test = sobol_thetas(500, seed=DEV_SEED + 1)
    t0 = time.perf_counter()
    y_train = simulate_batch(train, "S1", 0.0, n_jobs=6)
    y_test = simulate_batch(test, "S1", 0.0, n_jobs=6)
    print(f"simulations: {time.perf_counter() - t0:.1f} s for 1012 trajectories")

    for k in range(2):
        ens = Ensemble(n_members=1, base_seed=DEV_SEED * 10 + k).fit(train, y_train)
        m = ens.members[0]
        mean, _ = ens.predict(test)
        a = rel_norm(mean - y_test, y_test)
        print(f"nominal member {k}: {m.fit_seconds:.1f} s, {m.epochs_run} epochs, "
              f"{ens.n_parameters} params, median rel||a|| = {np.median(a):.4f}, "
              f"p90 = {np.quantile(a, 0.9):.4f}")

    small = FitConfig(hidden=32)
    rng = np.random.default_rng(DEV_SEED)
    pts = train[:56]
    resid = rng.normal(scale=0.05, size=(56, y_train.shape[1]))
    ens = Ensemble(n_members=1, cfg=small, base_seed=DEV_SEED).fit(pts, resid)
    m = ens.members[0]
    print(f"discrepancy member (56 pts, width 32): {m.fit_seconds:.1f} s, {m.epochs_run} epochs")


if __name__ == "__main__":
    main()

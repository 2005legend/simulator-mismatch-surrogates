"""Development check of the ranking sandwich bound (not the pre-registered pilot).

Dev seed 999 only: never the pilot seeds (0, 101, 102) or main seeds (1-10).

With per-point norms A = ||a||, D = ||d||, E = ||a + d|| (all divided by ||f_true||), the
triangle inequality gives |E - D| <= A and |E - A| <= D. For any signal s this implies
  Pearson:  |rho(s,E) - rho(s,D)| <= 2k/(1-k),  k = rms(A)/sd(D)   (large-r side)
            |rho(s,E) - rho(s,A)| <= 2k'/(1-k'), k' = rms(D)/sd(A)  (small-r side)
  Kendall:  |tau(s,E) - tau(s,D)| <= 2 P(|D_i - D_j| <= A_i + A_j)
            |tau(s,E) - tau(s,A)| <= 2 P(|A_i - A_j| <= D_i + D_j)
This script measures both sides on the pendulum to see how tight the bounds are.

Usage: python -m scripts.dev_bound_check
"""

import numpy as np
import torch
from scipy.stats import kendalltau, pearsonr
from sklearn.metrics import roc_auc_score

from mismatch.simulators import simulate_batch, sobol_thetas
from mismatch.surrogate import Ensemble

DEV_SEED = 999
EPS_REF = {"S1": 0.01, "S2": 0.0026}          # rough dev values, not the pilot's
MULTIPLES = (0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32)


def norms(x, ref):
    return np.linalg.norm(x, axis=1) / np.linalg.norm(ref, axis=1)


def pair_prob(x, y_sum):
    """P(|x_i - x_j| <= y_i + y_j) over distinct pairs."""
    i, j = np.triu_indices(len(x), k=1)
    return np.mean(np.abs(x[i] - x[j]) <= y_sum[i] + y_sum[j])


def pearson_bound(num, den_sd):
    k = np.sqrt(np.mean(num ** 2)) / den_sd
    return 2 * k / (1 - k) if k < 1 else np.inf


def main():
    torch.set_num_threads(1)
    train = sobol_thetas(512, seed=DEV_SEED)
    test = sobol_thetas(500, seed=DEV_SEED + 1)
    y_train = simulate_batch(train, "S1", 0.0, n_jobs=6)
    f_nom = simulate_batch(test, "S1", 0.0, n_jobs=6)
    ens = Ensemble(n_members=3, base_seed=DEV_SEED).fit(train, y_train)
    mean, sigma = ens.predict(test)
    s = sigma.mean(axis=1)
    a = mean - f_nom

    for system in ("S1", "S2"):
        print(f"\n{system}: columns = eps/eps_ref, r, CV(D), "
              "rho(s,E), rho(s,A), rho(s,D), Pearson bound to D, to A, "
              "tau(s,E), tau(s,A), tau(s,D), Kendall bound to D, to A, AUROC(s,E)")
        for m in MULTIPLES:
            f_true = simulate_batch(test, system, m * EPS_REF[system], n_jobs=6)
            d = f_nom - f_true
            A, D, E = norms(a, f_true), norms(d, f_true), norms(a + d, f_true)
            r = np.median(D) / np.median(A)
            rho = [pearsonr(s, v)[0] for v in (E, A, D)]
            tau = [kendalltau(s, v)[0] for v in (E, A, D)]
            pb_d, pb_a = pearson_bound(A, D.std()), pearson_bound(D, A.std())
            kb_d, kb_a = 2 * pair_prob(D, A), 2 * pair_prob(A, D)
            auroc = roc_auc_score(E > np.quantile(E, 0.8), s)
            print(f"{m:6.3f} {r:7.2f} {D.std() / D.mean():5.2f} | "
                  f"{rho[0]:6.3f} {rho[1]:6.3f} {rho[2]:6.3f} | {pb_d:6.2f} {pb_a:6.2f} | "
                  f"{tau[0]:6.3f} {tau[1]:6.3f} {tau[2]:6.3f} | {kb_d:5.2f} {kb_a:5.2f} | "
                  f"{auroc:5.3f}")


if __name__ == "__main__":
    main()

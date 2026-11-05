"""Diagnostic (not part of any pre-registered analysis): solver timing and how far the
discrepancy d(eps) is from linear in eps, for both systems.

Usage: python scripts/diag_solver_and_linearity.py
"""

import time

import numpy as np

from mismatch.errors import linearity_error, rel_norm
from mismatch.simulators import simulate_batch, sobol_thetas


def main():
    thetas = sobol_thetas(256, seed=11)

    t0 = time.perf_counter()
    nom = simulate_batch(thetas[:64], "S1", 0.0)
    serial = (time.perf_counter() - t0) / 64

    t0 = time.perf_counter()
    simulate_batch(sobol_thetas(1024, seed=12), "S2", 0.05, n_jobs=6)
    parallel = (time.perf_counter() - t0) / 1024
    print(f"solve time: serial {serial*1e3:.1f} ms/traj, 6 workers {parallel*1e3:.2f} ms/traj")

    nom = simulate_batch(thetas, "S1", 0.0, n_jobs=6)
    grids = {"S1": [0.005, 0.01, 0.02, 0.05, 0.1, 0.2], "S2": [0.0005, 0.001, 0.002, 0.005, 0.01, 0.02]}
    for system, eps_list in grids.items():
        print(f"\n{system}: eps  median rel||d||  linearity(eps -> 2 eps)  linearity(eps -> 4 eps)")
        for e in eps_list:
            d1 = nom - simulate_batch(thetas, system, e, n_jobs=6)
            d2 = nom - simulate_batch(thetas, system, 2 * e, n_jobs=6)
            d4 = nom - simulate_batch(thetas, system, 4 * e, n_jobs=6)
            true1 = nom - d1
            print(f"  {e:<7g} {np.median(rel_norm(d1, true1)):>10.4f} "
                  f"{linearity_error(d2, d1, 2*e, e):>14.3f} {linearity_error(d4, d1, 4*e, e):>16.3f}")


if __name__ == "__main__":
    main()

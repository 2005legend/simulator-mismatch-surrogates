"""Development feasibility test for a PDE system (not pre-registered, dev seed 999 only).

Nominal: 1D viscous Burgers, u_t + u u_x = nu u_xx, periodic on [0, 2 pi), pseudo-spectral
on 256 points. True system adds one missing term at strength eps:
  P1: -eps * u|u|      (state-dependent nonlinear damping)
  P2: +eps * sin(3x)   (fixed forcing, independent of the state)
theta = (nu, A1, A2, phase), u0 = A1 sin(x) + A2 sin(2x + phase). Output: u(x, T=2) on 64 points.

Measures solve time, fit time, approximation error, whether r can be dialled with eps, and
whether nominal ensemble sigma keeps ranking the total error as r grows.

Usage: python -m scripts.dev_burgers_pilot
"""

import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
from scipy.integrate import solve_ivp
from scipy.stats import pearsonr, qmc
from sklearn.metrics import roc_auc_score

from mismatch import config
from mismatch.surrogate import Ensemble

DEV_SEED = 999
N_X, N_OUT, T_END = 256, 64, 2.0
X = np.linspace(0.0, 2 * np.pi, N_X, endpoint=False)
K = np.fft.rfftfreq(N_X, d=1.0 / N_X)
LOW = np.array([0.03, 0.5, 0.0, 0.0])
HIGH = np.array([0.15, 1.5, 1.0, np.pi])
EPS = {"P1": (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0), "P2": (0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5)}


def solve(args):
    theta, system, eps = args
    nu, a1, a2, phase = theta

    def rhs(_, u):
        u_hat = np.fft.rfft(u)
        ux = np.fft.irfft(1j * K * u_hat, n=N_X)
        uxx = np.fft.irfft(-(K ** 2) * u_hat, n=N_X)
        out = -u * ux + nu * uxx
        if eps:
            out += -eps * u * np.abs(u) if system == "P1" else eps * np.sin(3 * X)
        return out

    u0 = a1 * np.sin(X) + a2 * np.sin(2 * X + phase)
    sol = solve_ivp(rhs, (0.0, T_END), u0, method="RK45", rtol=1e-6, atol=1e-8, t_eval=[T_END])
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.y[::N_X // N_OUT, 0]


def solve_batch(thetas, system, eps, pool):
    return np.array(list(pool.map(solve, [(t, system, eps) for t in thetas], chunksize=8)))


def unit_sobol(n, seed):
    return qmc.Sobol(d=4, scramble=True, seed=seed).random(n)


def main():
    torch.set_num_threads(1)
    u_train, u_test = unit_sobol(512, DEV_SEED), unit_sobol(300, DEV_SEED + 1)
    train, test = LOW + u_train * (HIGH - LOW), LOW + u_test * (HIGH - LOW)
    # The surrogate scales inputs with the pendulum box, so pass unit-cube points mapped into it.
    to_box = lambda u: config.BOX_LOW + u * (config.BOX_HIGH - config.BOX_LOW)

    t0 = time.perf_counter()
    solve((train[0], "P1", 0.0))
    print(f"one serial solve: {time.perf_counter() - t0:.2f} s")

    with ProcessPoolExecutor(max_workers=6) as pool:
        t0 = time.perf_counter()
        y_train = solve_batch(train, "P1", 0.0, pool)
        f_nom = solve_batch(test, "P1", 0.0, pool)
        print(f"812 nominal solves, 6 workers: {time.perf_counter() - t0:.1f} s")

        ens = Ensemble(n_out=N_OUT, n_members=3, base_seed=DEV_SEED).fit(to_box(u_train), y_train)
        mean, sigma = ens.predict(to_box(u_test))
        print("fit seconds per member:", [round(t, 1) for t in ens.fit_seconds],
              "epochs:", [m.epochs_run for m in ens.members])
        s, a = sigma.mean(axis=1), mean - f_nom

        for system in ("P1", "P2"):
            print(f"\n{system}: eps, r, median rel A, rho(s,E), rho(s,A), rho(s,D), AUROC(s,E)")
            for eps in EPS[system]:
                f_true = solve_batch(test, system, eps, pool)
                d = f_nom - f_true
                ref = np.linalg.norm(f_true, axis=1)
                A, D, E = (np.linalg.norm(v, axis=1) / ref for v in (a, d, a + d))
                rho = [pearsonr(s, v)[0] for v in (E, A, D)]
                auroc = roc_auc_score(E > np.quantile(E, 0.8), s)
                print(f"{eps:6.3f} {np.median(D) / np.median(A):7.2f} {np.median(A):6.3f} | "
                      f"{rho[0]:6.3f} {rho[1]:6.3f} {rho[2]:6.3f} | {auroc:5.3f}")


if __name__ == "__main__":
    main()

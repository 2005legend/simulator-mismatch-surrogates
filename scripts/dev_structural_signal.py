"""Development test of a structural-perturbation signal (dev seed 999 only, no real data).

Idea: ensemble sigma perturbs the network's weights, so it tracks approximation error.
To track the discrepancy without real data, perturb the simulator's physics instead: add
K random plausible missing terms h_k(phi, omega) to the nominal ODE and score each test
point by how much its trajectory moves,
    s_struct(theta) = rms_k || f_{nom + h_k}(theta) - f_nom(theta) || / || f_nom(theta) ||.
The prior over terms is a random combination of generic basis functions. To avoid handing
the answer to the signal, a "leave-family-out" prior drops the basis functions of the true
term's family. Reported: Spearman correlation with the true discrepancy norm D on S1 and S2.

Usage: python -m scripts.dev_structural_signal
"""

import numpy as np
import torch
from scipy.stats import spearmanr

from mismatch.metrics import rel_errors
from mismatch.simulators import simulate_batch, sobol_thetas
from mismatch.dynamics import rollout

DEV_SEED = 999
K = 32
SCALE = 1e-3          # tiny, so the response is linear and no rollout diverges

BASIS = {
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
FAMILY = {"S1": ("w|w|", "w^3"), "S2": ("tanh(w/0.1)", "tanh(w/0.5)")}
# rough magnitudes over the box, so each basis function contributes comparably
NORM = {"w": 3.0, "w|w|": 9.0, "w^3": 27.0, "tanh(w/0.1)": 1.0, "tanh(w/0.5)": 1.0,
        "p": 1.5, "p^3": 3.0, "sin(2p)": 1.0, "p*w": 4.0}


def struct_signal(xt, base, names, rng):
    moves = []
    for _ in range(K):
        coef = {n: SCALE * rng.normal() / NORM[n] for n in names}
        term = lambda p, w, c=coef: sum(c[n] * BASIS[n](p, w) for n in c)
        with torch.no_grad():
            moves.append(rel_errors(rollout(xt, term).numpy(), base))
    return np.sqrt(np.mean(np.square(moves), axis=0))


def main():
    torch.set_num_threads(1)
    test = sobol_thetas(500, seed=DEV_SEED + 1)
    xt = torch.from_numpy(test)
    with torch.no_grad():
        base = rollout(xt).numpy()
    f_nom = simulate_batch(test, "S1", 0.0)
    rng = np.random.default_rng(DEV_SEED)

    full = struct_signal(xt, base, list(BASIS), rng)
    velocity_only = struct_signal(xt, base, ["w", "p", "p*w"], rng)
    for system, eps_list in (("S1", (0.01, 0.04, 0.16)), ("S2", (0.0026, 0.0104, 0.0416))):
        loo_names = [n for n in BASIS if n not in FAMILY[system]]
        loo = struct_signal(xt, base, loo_names, rng)
        print(f"\n{system}: Spearman correlation with the true discrepancy norm D")
        for eps in eps_list:
            d = rel_errors(f_nom, simulate_batch(test, system, eps))
            row = [f"{name} {spearmanr(sig, d)[0]:+.3f}"
                   for name, sig in (("full prior", full), ("leave-family-out", loo),
                                     ("linear terms only", velocity_only))]
            print(f"  eps={eps}: " + " | ".join(row), flush=True)
        # one hypothesised term at a time, at the middle eps
        d = rel_errors(f_nom, simulate_batch(test, system, eps_list[1]))
        single = {n: spearmanr(struct_signal(xt, base, [n], rng), d)[0] for n in BASIS}
        print("  single-term sensitivity vs D: "
              + ", ".join(f"{n} {v:+.2f}" for n, v in single.items()), flush=True)


if __name__ == "__main__":
    main()

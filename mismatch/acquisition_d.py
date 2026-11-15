"""Family D acquisition (addendum v1.6 section 4): choosing real experiments when the
correction is a missing term learned inside the ODE.

Same loop as `acquisition.py` (8 initial points, then rounds of 8 from a shared fresh pool of
Sobol candidates, same batch rule), but the correction is a small network h(phi, omega) added
to the nominal acceleration, fitted to the observed noisy trajectories after every round.

Arms: random, sobol, nominal_sigma (nominal ensemble disagreement, as in Family A),
term_disagreement (K learned terms fitted with different seeds on the same data, spread of
their rollouts over the candidate pool) and state_coverage (distance of a candidate's nominal
trajectory, in the (phi, omega/3) plane, to the states the observed trajectories already cover).

Scored quantity: median relative L2 error of the learned-ODE rollout against the noise-free
true system on the test set, full horizon. The corrected surrogate (nominal surrogate plus the
learned-ODE rollout minus the nominal-ODE rollout) is logged alongside.
"""

import numpy as np
from scipy.interpolate import make_smoothing_spline
from scipy.spatial import cKDTree

from mismatch import config
from mismatch.acquisition import Problem, sample_batch
from mismatch.dynamics import fit_term, simulate_with_term
from mismatch.metrics import budget_auc, rel_errors, summarize
from mismatch.simulators import stream_seed

ARMS_D = {"random": 1, "sobol": 2, "nominal_sigma": 3, "term_disagreement": 15,
          "state_coverage": 16}
COVERAGE_TIMES = np.linspace(0, config.N_T - 1, 40).astype(int)   # 40 sample times
OMEGA_SCALE = 3.0


def term_disagreement_scores(terms, pool):
    """Mean over time of the across-term standard deviation of the rollouts, per candidate."""
    runs = np.stack([simulate_with_term(pool, t) for t in terms])
    return runs.std(axis=0).mean(axis=1)


def observed_states(thetas, y):
    """(phi, omega/3) at the coverage times for each observed trajectory.

    Only phi is observed; omega comes from a smoothing spline (smoothing chosen by GCV).
    """
    t = config.T_GRID
    states = []
    for traj in y:
        spl = make_smoothing_spline(t, traj)
        states.append(np.stack([spl(t), spl.derivative(1)(t) / OMEGA_SCALE], axis=1)[COVERAGE_TIMES])
    return np.concatenate(states)


def state_coverage_scores(pool, observed):
    """Mean over the coverage times of the distance, in (phi, omega/3), from the candidate's
    nominal-ODE state to the nearest observed state."""
    phi, omega = simulate_with_term(pool, return_omega=True)
    query = np.stack([phi[:, COVERAGE_TIMES], omega[:, COVERAGE_TIMES] / OMEGA_SCALE], axis=-1)
    dist, _ = cKDTree(observed).query(query.reshape(-1, 2))
    return dist.reshape(len(pool), -1).mean(axis=1)


def run_arm_d(problem, arm, init_obs=None, k_terms=5, lbfgs_iters=20, gm_steps=500):
    """Run one Family D arm. `init_obs` = (y, sigma) of the shared initial points."""
    p = problem
    rng = np.random.default_rng([p.seed, ARMS_D[arm]])
    design = p.design()
    x = design[:p.n_init]
    y, _ = init_obs if init_obs is not None else p.oracle(x)
    test, truth = p.test_thetas, p.test_true
    nominal_ode = simulate_with_term(test)
    surrogate = p.nominal.predict(test)[0]

    log = []
    for rnd in range(p.rounds + 1):
        fit_seed = stream_seed(p.seed, "fit", rnd)       # same for every arm: paired fits
        term = fit_term(x, y, seed=fit_seed, lbfgs_iters=lbfgs_iters, gm_steps=gm_steps)
        learned = simulate_with_term(test, term)
        err = rel_errors(learned, truth)
        corrected = rel_errors(surrogate + (learned - nominal_ode), truth)
        log.append({"n": len(x), **summarize(err), "corrected_median": float(np.median(corrected))})
        if rnd == p.rounds:
            break

        pool = p.pool(rnd + 1)
        if arm == "random":
            new_x = pool[rng.choice(len(pool), size=p.batch, replace=False)]
        elif arm == "sobol":
            lo = p.n_init + rnd * p.batch
            new_x = design[lo:lo + p.batch]
        else:
            if arm == "nominal_sigma":
                scores = p.nominal.predict(pool)[1].mean(axis=1)
            elif arm == "term_disagreement":
                terms = [term] + [fit_term(x, y, seed=stream_seed(p.seed, "score", rnd * 10 + j),
                                           lbfgs_iters=lbfgs_iters,
                                           gm_steps=gm_steps) for j in range(1, k_terms)]
                scores = term_disagreement_scores(terms, pool)
            elif arm == "state_coverage":
                scores = state_coverage_scores(pool, observed_states(x, y))
            else:
                raise ValueError(f"unknown arm {arm}")
            new_x = pool[sample_batch(scores, p.batch, p.k, rng)]
        new_y, _ = p.oracle(new_x)
        x, y = np.vstack([x, new_x]), np.vstack([y, new_y])

    ns = [r["n"] for r in log]
    return {"arm": arm, "log": log, "thetas": x,
            "budget_auc": budget_auc(ns, [r["median"] for r in log]),
            "budget_auc_corrected": budget_auc(ns, [r["corrected_median"] for r in log])}


def run_cell_d(problem, arms=tuple(ARMS_D), **kw):
    """All Family D arms on one (system, eps, seed) cell with a shared initial set."""
    init_obs = problem.oracle(problem.design()[:problem.n_init])
    return {arm: run_arm_d(problem, arm, init_obs, **kw) for arm in arms}

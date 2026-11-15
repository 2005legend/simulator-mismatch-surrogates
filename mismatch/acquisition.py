"""RQ2 acquisition loop and arms 1-9 (prereg-v1 section 7, v1.1 section 4, v1.2 section 5).

Loop: 8 initial true points (scrambled Sobol, shared across arms in a seed), then 6 rounds
of 8. Each round every arm scores the same fresh pool of 4,096 Sobol candidates and the
correction model (discrepancy ensemble) is refitted. Corrected prediction = nominal
surrogate + correction.
"""

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from mismatch import config
from mismatch.gp import PCAGP, ScalarGP
from mismatch.metrics import budget_auc, rel_errors, summarize
from mismatch.signals import residual_score, spline_residual_score
from mismatch.simulators import simulate_batch, sobol_thetas, stream_seed
from mismatch.surrogate import Ensemble, FitConfig

ARMS = {1: "random", 2: "sobol", 3: "nominal_sigma", 4: "nominal_residual",
        5: "disc_sigma", 6: "gp_variance", 7: "observed_residual", 8: "r_gated", 9: "deup"}

# Small MLP for the correction; no held-out split because n starts at 8. Pilot may retune.
DISC_CFG = FitConfig(hidden=32, max_epochs=300, min_val=10 ** 9)


def sample_batch(scores, n, k, rng):
    """Indices of n candidates drawn without replacement with probability ~ score^k.

    k = 0 is uniform; k = inf is the greedy top-n.
    """
    s = np.clip(np.asarray(scores, dtype=float), 0.0, None)
    if k == 0 or not np.any(s > 0):
        return rng.choice(len(s), size=n, replace=False)
    if np.isinf(k):
        return np.argsort(-s)[:n]
    w = (s / s.max()) ** k + 1e-12
    return rng.choice(len(s), size=n, replace=False, p=w / w.sum())


def make_oracle(system, eps, noise_frac=config.NOISE_MAIN, seed=0):
    """True-system experiment: thetas -> (noisy trajectories, noise std per trajectory).

    The noise for a point depends only on (seed, theta), so every arm that queries the
    same point sees the same observation, whatever order the arms run in.
    """
    def oracle(thetas):
        thetas = np.atleast_2d(thetas)
        clean = simulate_batch(thetas, system, eps)
        sigma = noise_frac * clean.std(axis=1)
        noisy = np.empty_like(clean)
        for i, th in enumerate(thetas):
            key = np.frombuffer(np.ascontiguousarray(th, dtype=np.float64).tobytes(),
                                dtype=np.uint32).tolist()
            rng = np.random.default_rng([seed] + key)
            noisy[i] = clean[i] + sigma[i] * rng.normal(size=clean.shape[1])
        return noisy, sigma
    return oracle


def r_hat_lower(y, f_nom, sigma, denom, rng, n_boot=1000, level=0.90):
    """Estimate of r and the lower end of its percentile-bootstrap CI (v1.1 section 4).

    Numerator: median debiased discrepancy norm relative to ||y||. Denominator: median
    relative approximation error of the nominal surrogate over the candidate pool.
    """
    n_t = y.shape[1]
    sq = np.sum((y - f_nom) ** 2, axis=1) - n_t * sigma ** 2
    num = np.sqrt(np.maximum(sq, 0.0)) / np.linalg.norm(y, axis=1)
    boots = np.median(num[rng.integers(0, len(num), size=(n_boot, len(num)))], axis=1)
    return np.median(num) / denom, np.quantile(boots, (1 - level) / 2) / denom


def update_gate(gate, lower, n_obs):
    """Monotone three-way rule: 0 = arm 3, 1 = Sobol, 2 = arm 7. Never switches back."""
    if lower > 2 and n_obs >= 16:
        return 2
    if lower > 1:
        return max(gate, 1)
    return gate


@dataclass
class Problem:
    nominal: Ensemble                     # fitted nominal surrogate (shared across arms)
    oracle: Callable                      # thetas -> (noisy true trajectories, noise std)
    test_thetas: np.ndarray
    test_true: np.ndarray                 # noise-free true trajectories on the test set
    seed: int
    k: float = 1.0                        # batch-rule exponent, frozen in the pilot
    n_init: int = 8
    batch: int = 8
    rounds: int = 6
    pool_size: int = 4096
    disc_cfg: FitConfig = field(default_factory=lambda: DISC_CFG)
    disc_members: int = 5
    spline_lam: Optional[float] = None    # arm 7 smoothing; None = GCV per trajectory
    gp_components: Optional[int] = None   # arm 6 PCA components; None = 99.9% variance
    nominal_sim: Callable = lambda th: simulate_batch(th, "S1", 0.0)
    n_t: Optional[int] = None             # score only the first n_t time points (horizon)
    _pool_err: dict = field(default_factory=dict)

    def cut(self, traj):
        return traj if self.n_t is None else traj[:, :self.n_t]

    @property
    def t_grid(self):
        return config.T_GRID if self.n_t is None else config.T_GRID[:self.n_t]

    def nom(self, thetas):
        """Nominal surrogate (mean, sigma) on the scored horizon."""
        mean, sigma = self.nominal.predict(thetas)
        return self.cut(mean), self.cut(sigma)

    def design(self):
        """Initial points followed by the Sobol continuation used by arm 2."""
        return sobol_thetas(self.n_init + self.rounds * self.batch,
                            seed=stream_seed(self.seed, "design"))

    def pool(self, rnd):
        """Candidate pool for round rnd (1-based), shared by all arms."""
        return sobol_thetas(self.pool_size, seed=stream_seed(self.seed, "pool", rnd))

    def pool_nominal_error(self, rnd):
        """Median relative approximation error over the round's pool (arm 8 denominator)."""
        key = (rnd, self.n_t)
        if key not in self._pool_err:
            pool = self.pool(rnd)
            f_nom = self.cut(self.nominal_sim(pool))
            self._pool_err[key] = float(np.median(rel_errors(self.nom(pool)[0], f_nom)))
        return self._pool_err[key]


def _scores(arm, p, pool, x, y, disc, deup, seed):
    if arm == 3:
        return p.nom(pool)[1].mean(axis=1)
    if arm == 4:
        return residual_score(p.nom(pool)[0], pool, p.t_grid)
    if arm == 5:
        return disc.predict(pool)[1].mean(axis=1)
    if arm == 6:
        resid = y - p.nom(x)[0]
        return PCAGP(n_components=p.gp_components, seed=seed).fit(x, resid).predict(pool)[1]
    if arm == 7:
        mag = spline_residual_score(y, x, t_grid=p.t_grid, lam=p.spline_lam)
        return ScalarGP(seed=seed).fit(x, mag).predict(pool)
    if arm == 9:
        gp = ScalarGP(seed=seed).fit(np.vstack(deup[0]), np.log(np.concatenate(deup[1])))
        return np.exp(gp.predict(pool))
    raise ValueError(f"arm {arm} has no score")


def run_arm(problem, arm, init_obs=None):
    """Run one arm. Returns per-round errors, budget-AUC, the chosen points and gate log."""
    p = problem
    rng = np.random.default_rng([p.seed, arm])
    design = p.design()
    x = design[:p.n_init]
    y, sigma = init_obs if init_obs is not None else p.oracle(x)
    y, truth = p.cut(y), p.cut(p.test_true)
    nominal_test = p.nom(p.test_thetas)[0]

    gate, gate_log = 0, []
    f_nom_obs = p.cut(p.nominal_sim(x)) if arm == 8 else None
    # Arm 9: errors of points before they entered training. The initial points are scored
    # against the uncorrected nominal surrogate, which has not seen them either.
    deup = ([x], [rel_errors(p.nom(x)[0], y)]) if arm == 9 else None

    log = []
    for rnd in range(p.rounds + 1):
        fit_seed = stream_seed(p.seed, "fit", rnd)                 # same for every arm: paired fits
        score_seed = stream_seed(p.seed, "score", arm * 100 + rnd)  # arm-specific, score models only
        disc = Ensemble(n_out=y.shape[1], n_members=p.disc_members, cfg=p.disc_cfg,
                        base_seed=fit_seed).fit(x, y - p.nom(x)[0])
        err = rel_errors(nominal_test + disc.predict(p.test_thetas)[0], truth)
        log.append({"n": len(x), **summarize(err)})
        if rnd == p.rounds:
            break

        pool = p.pool(rnd + 1)
        sobol_block = design[p.n_init + rnd * p.batch: p.n_init + (rnd + 1) * p.batch]
        rule = arm
        if arm == 8:
            r_hat, lower = r_hat_lower(y, f_nom_obs, sigma, p.pool_nominal_error(rnd + 1), rng)
            gate = update_gate(gate, lower, len(x))
            gate_log.append({"n": len(x), "r_hat": float(r_hat), "lower": float(lower),
                             "gate": gate})
            rule = {0: 3, 1: 2, 2: 7}[gate]

        if rule == 1:
            new_x = pool[rng.choice(len(pool), size=p.batch, replace=False)]
        elif rule == 2:
            new_x = sobol_block
        else:
            scores = _scores(rule, p, pool, x, y, disc, deup, score_seed)
            new_x = pool[sample_batch(scores, p.batch, p.k, rng)]

        new_y, new_sigma = p.oracle(new_x)
        new_y = p.cut(new_y)
        if arm == 9:
            pred = p.nom(new_x)[0] + disc.predict(new_x)[0]
            deup[0].append(new_x)
            deup[1].append(rel_errors(pred, new_y))
        if arm == 8:
            f_nom_obs = np.vstack([f_nom_obs, p.cut(p.nominal_sim(new_x))])
        x, y, sigma = np.vstack([x, new_x]), np.vstack([y, new_y]), np.concatenate([sigma, new_sigma])

    ns, med = [r["n"] for r in log], [r["median"] for r in log]
    uncorrected = summarize(rel_errors(nominal_test, truth))
    return {"arm": arm, "name": ARMS[arm], "log": log, "budget_auc": budget_auc(ns, med),
            "uncorrected": uncorrected, "thetas": x, "gate_log": gate_log}


def run_cell(problem, arms=tuple(ARMS)):
    """Run several arms on one (system, eps, seed) cell with a shared initial set."""
    init_obs = problem.oracle(problem.design()[:problem.n_init])
    return {arm: run_arm(problem, arm, init_obs) for arm in arms}

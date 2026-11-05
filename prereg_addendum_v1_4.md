# Pre-registration addendum v1.4: Blind to the Missing Term

Drafted 2026-11-05. Status: **frozen 2026-11-05** by the git commit tagged `prereg-v1.4`, before any pilot run (pilot starts 2026-11-25).

**Scope.** One deviation to how the frozen predictions P-A (v1.1 section 2), P-B (v1.1 section 3) and P-C (v1.2 section 1) are computed. No test, success criterion, family, arm, seed or metric changes.

## Why

Prediction inputs in v1.1 assumed the discrepancy is linear in the mismatch strength, d(ε) ≈ ε·g with g = d(ε_ref)/ε_ref. A simulator check on 2026-11-05 (`scripts/diag_solver_and_linearity.py`, 256 Sobol points, no learned model involved) showed it is not: the missing terms shift the oscillation's phase, and the shift accumulates over the 10 s horizon.

| System | ε | median relative ‖d‖ | linearity error ε → 2ε | ε → 4ε |
|---|---|---|---|---|
| S1 | 0.01 | 0.040 | 0.067 | 0.197 |
| S1 | 0.02 | 0.077 | 0.123 | 0.358 |
| S1 | 0.05 | 0.176 | 0.248 | 0.710 |
| S2 | 0.002 | 0.031 | 0.024 | 0.077 |
| S2 | 0.005 | 0.077 | 0.069 | 0.223 |
| S2 | 0.01 | 0.156 | 0.155 | 0.494 |

The pre-registered sweep spans a factor of 8 in ε (0.25 to 2 × ε_ref). If ε_ref falls where the discrepancy is 5-10% of the trajectory, the linear approximation is 20-40% off at the ends of the sweep, so the predictions could fail for a known, uninteresting reason.

## Change

The predictions use the **exact simulator discrepancy at each ε** instead of its linear extrapolation:

- d(ε) = f_nom − f_true,ε is computed with the simulators on pilot seed 0's test points, at every ε level of the sweep. This uses no learned model trained at ε > 0, so the predictions remain genuine predictions.
- **P-A and P-C (Model A).** The surrogate-side inputs are unchanged and still come from the ε = 0 surrogate of pilot seed 0: s_A, ρ_sa and the signal values. The discrepancy-side inputs are recomputed at each ε from d(ε): s_D(ε), ρ_sg(ε), ρ_ag(ε), the median ‖d(ε)‖ and hence r(ε). Model A's Gaussian-copula step, closed form and Monte-Carlo fallback rule are otherwise unchanged. For P-C this is done separately in each horizon window.
- **P-B (Model B).** The corrected-error expression for a fixed design and frozen-hyperparameter GP/kernel-ridge corrector is evaluated as a quadratic form in the exact a + d(ε) at each ε, instead of the expansion Q_aa + 2εQ_ag + ε²Q_gg. The crossover is located by interpolation in log ε over the five sweep levels plus a finer simulator-only grid (20 log-spaced ε values spanning the sweep), which needs no learned model at ε > 0. The plateau signs, the H2-null condition and the S2 direction prediction are read from these curves as before.
- **Linear variant kept as a secondary comparison.** P-A, P-B and P-C are also computed with the original linear approximation and reported beside the exact-d versions, with the linearity error. **The exact-d versions are the ones scored against v1.1's and v1.2's success criteria.**

## Order of events

Unchanged. Both variants of P-A, P-B and P-C are computed from pilot seed 0 and committed under `pred-v1` before any seed 1-10 run starts.

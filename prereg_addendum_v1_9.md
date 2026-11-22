# Pre-registration addendum v1.9: Blind to the Missing Term (out-of-dictionary terms on the pendulum)

Drafted 2026-11-22. Status: **frozen 2026-11-22** by the git commit tagged `prereg-v1.9`, after the author's approval of the draft as written, and before any run on pilot seed 0 or main seeds 1-20 with the new terms. Nothing in `prereg-v1` to `prereg-v1.8`, `pred-v1` or `pred-v2` changes.

**Why this addendum exists.** It is written after every result of v1 to v1.8 was seen, including the v1.8 out-of-dictionary results on System 3. A reviewer would ask whether those results hold on a second system with different physics. The pendulum dictionary (`mismatch.alignment.TERMS`) holds functions of the state (phi, phi') only, so a missing term that is not a function of the state lies partly outside its span. This addendum runs the v1.8 design on the pendulum with two such terms. Its tests are frozen before any registered seed is run with them.

## 1. Disclosure: development look (dev seed 999 only)

`scripts/run_pendulum_ood.py --phase dev`, all stages, output `runs/pend_ood_dev/`; plus a one-off diagnostic of the linear-response gap at 1, 4 and 16 x on the same seed.

| Quantity | Q1 (constant torque) | Q2 (periodic drive) |
|---|---|---|
| Rough eps_ref (r = 1, n_nom = 512) | 0.00503 | 0.00464 |
| Share of the term's response outside the dictionary span, median (10-90%) | 0.73 (0.38-0.86) | 0.50 (0.17-0.88) |
| AUROC of ensemble sigma at 0, 1, 4, 16 x | 0.600, 0.558, 0.371, 0.222 | 0.600, 0.543, 0.498, 0.380 |
| Slope of AUROC on log r (seven nonzero levels) | -0.103 | -0.056 |
| rho(sigma, E) - rho(sigma, D) at 16 x | 0.011 | 0.034 |
| rho(sigma, D) at 16 x; full-dictionary range | -0.446; [-0.60, 0.82] (inside) | -0.422; [-0.60, 0.82] (inside) |
| rho(residual, D) at 16 x; range | -0.338; [-0.57, 0.75] (inside) | -0.331; [-0.57, 0.75] (inside) |
| Linear-response gap abs(rho(sigma, D) - rho(sigma, \|\|L[h]\|\|)) at 1, 4, 16 x | 0.009, 0.04, **0.14** | 0.006, 0.026, 0.085 |
| Same gap for the residual at 16 x | 0.148 | 0.085 |
| Selection from 16 real trajectories at 4 x: selected term, statistic vs sigma | tanh(w/0.1), 0.84 vs -0.34 (beats) | p, 0.53 vs -0.36 (beats) |
| Coverage at 16 x: nominal-calibrated / calibrated on 16 real trajectories | 0.0% / 98.4% | 1.0% / 98.8% |

Reading of the dev look, stated before any registered run:

- Ensemble sigma is anti-aligned with the discrepancy for both terms (rho about -0.4 at 16 x), so ranking falls below chance, as on S2.
- The linear-response mechanism (v1.8 OR1L) holds at small eps and breaks at large eps on the pendulum: the gap grows with the level (0.009 at 1 x, 0.14 at 16 x for Q1). Normalising by the true instead of the nominal trajectory does not close it (16 x, Q1: -0.290 against -0.304), so the cause is nonlinearity of the discrepancy in eps, not the normaliser. The gap roughly doubles with each doubling of eps (Q1, sigma: 0.001, 0.004, 0.009, 0.021, 0.041, 0.079, 0.142 at 0.25-16 x; Q2: 0.000 to 0.085), as expected from a second-order term. This is the "small eps" condition of Proposition 2 failing at 16 x; on System 3 it did not fail (largest gap 0.0062).
- Selection beats sigma easily here because sigma is negative; the test is weaker than on System 3 for that reason, and this is stated now.

Checks of the new terms: eps = 0 gives the nominal simulator exactly; responses at steps 1e-3 and 1e-4 differ by under 1%; a dictionary term has zero out-of-span share; a positive constant torque shifts late-time phi upward (`tests/test_pendulum_ood.py`, 5 tests).

## 2. Systems

The pendulum of v1 (`mismatch/simulators.py`, nominal solver, box, grid, tolerances, noise 1% as frozen), with two new missing terms in `mismatch/pendulum_ood.py`. The true system adds eps * h to the angular acceleration:

- **Q1:** h = g (constant torque: a tilted pivot or a steady wind).
- **Q2:** h = g sin(1.5 t) (periodic external drive; 1.5 rad/s lies below the natural frequencies sqrt(g/L) in [2.2, 4.4] rad/s, so there is no resonance).

Neither is a function of (phi, phi'). S1 and S2 are not rerun. Surrogates are the frozen n_nom = 512 settings with the same sample streams as RQ1, so each seed's nominal ensemble equals its RQ1 ensemble.

## 3. Pilot and levels

Seed 0: eps_ref per term by bisection on r = 1 at n_nom = 512 (brackets [1e-6, 0.1] for both), written to `pilot_results/pend_ood_eps_ref.json`. No transition prediction is frozen. Levels eps in {0, 0.25, 0.5, 1, 2, 4, 8, 16} x eps_ref; seeds 1-20; same signals, metrics and arrays as v1.8 (ensemble sigma, MC dropout, ODE residual, conformal, D3).

## 4. Tests (all secondary; family Q)

"k of 20" counts use 20 seeds. Signals for the alignment tests: ensemble sigma and the ODE residual. Names follow v1.8 with a Q prefix.

- **QB1 (Holm over 2, alpha 0.05):** per seed, slope of ensemble-sigma AUROC on log r over the seven nonzero levels, hypothesis slope < 0, for Q1 and Q2. Supported only if Holm p < 0.05 and the 95% BCa CI lies below 0. Dev: falls on both.
- **QD1 (limit):** at 16 x, rho_S(sigma, E) - rho_S(sigma, D) within +/- 0.10, TOST across seeds, per term. Dev: 0.011 and 0.034.
- **QR1 (bracketing out of the dictionary):** at 16 x, rho_S(signal, D) inside the full-dictionary alignment range (nine terms, 2,000 mixtures) in at least 18 of 20 seeds, per term and signal. The range is wide (dev: 1.42 of 2 for sigma, 1.32 for the residual), so a uniformly placed value lands inside about 71% and 66% of the time; the paper reports the widths and the reference probability next to the counts, as in v1.8.
- **QR1L (linear-response mechanism at 16 x, same rule as v1.8 OR1L):** abs(rho_S(signal, D) - rho_S(signal, ||L[h]||)) <= 0.10 in at least 18 of 20 seeds, per term and signal; L[h] by finite difference, step 1e-3. **Expectation stated now: likely not met on Q1 (dev gap 0.14), borderline on Q2 (dev 0.085).** It is run and reported either way.
- **QR1L4 (the same at 4 x, new):** the same rule at 4 x eps_ref, per term and signal. Expectation: met on both (dev gaps 0.04 and 0.026). Together with QR1L this locates where linearity breaks.
- **QR3 (selection from 16 real trajectories at 4 x):** as v1.8 OR3. Expectation: met on both, but weak, because sigma's own statistic is negative.
- **Descriptive:** the gap at every level 0.25-16 x; out-of-span share; selected terms; D3 counts; coverage of nominal and real-data conformal intervals; geometric-mean r per level.

## 5. Compute

Dev seed: about 7 minutes of cells and 2 minutes of alignment per seed. Seeds 1-20 in four parallel processes: about 1 to 1.5 hours in all. Nothing is cut if compute runs short; the run is resumable.

## 6. Order of events

Author review; freeze v1.9 (tag `prereg-v1.9`); pilot seed 0 (eps_ref only); seeds 1-20; analysis. The paper reports v1.9 next to v1.8, after all earlier results, and says it was written after they were seen.

## 7. Design decision (per project rules)

- **What matters:** whether the v1.8 findings generalise to a second system. With 20 seeds an 18-of-20 count has a 95% interval of about 0.68 to 0.99. That is enough to tell "holds" from "breaks" for QR1L, where the dev gap (0.14) sits well above the 0.10 margin.
- **Options considered:** (a) the v1.8 design unchanged, OR1L at 16 x only; (b) as (a) plus the same test at 4 x (recommended); (c) a smaller eps range (up to 4 x) so linearity holds, which would hide the failure the dev look found.
- **Recommendation:** (b). It keeps comparability with v1.8 and measures, rather than avoids, the nonlinear regime. What would change it: a reason to think 4 x is also nonlinear on main seeds, which the dev look does not suggest.
- **What cannot be known yet:** the seed-to-seed spread of the gap; the dev look is one seed.

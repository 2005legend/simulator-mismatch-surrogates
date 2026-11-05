# Pre-registration: Blind to the Missing Term (v1)

Drafted 2026-11-04. Status: **frozen 2026-11-04** by the git commit tagged `prereg-v1`, before any pilot or main-grid run. Values marked as proposed defaults (parameter box, δ) may change only by the pilot rules in section 9; every other change goes in section 11. The pilot (section 9) may run before freezing only to set the quantities listed there; pilot seed 0 is excluded from every analysis.

Companion files: `project_goal_v2.md` (scope and plan), `week1_novelty_multifidelity.md` (novelty check).

## 1. Questions

- **RQ1.** As the ratio r of structural discrepancy to approximation error grows, how well does each uncertainty signal rank the surrogate's true-system error, and how well is it calibrated?
- **RQ2.** With a budget of at most 56 true-system experiments used to fit a correction model, which acquisition rule gives the lowest corrected error, and how does each rule's gap to Sobol design change with r?

## 2. Hypotheses

- **H1 (ranking).** Signals built only from the nominal simulator (nominal ensemble σ, MC dropout, nominal-ODE residual) lose ranking ability as r grows: their AUROC for the large-error event falls with log r.
- **H2 (acquisition).** The corrected-error gap between each nominal-only acquisition arm (arms 3 and 4) and Sobol grows with log r.
- **H3 (rescue).** The observed-residual arm (arm 7) reaches lower corrected error than the GP discrepancy arm (arm 6) at the largest r.
- **H4 (matched transitions).** The r at which nominal σ loses half its ranking ability equals, within a factor of 1.5, the r at which arm 3 falls behind Sobol by half its final gap.

The mechanism behind H1 (an ensemble's spread cannot see a bias that all members share) is known. The claims are the measured rates in H1-H2, the rescue in H3 and the match in H4.

## 3. Systems and data

State φ(t) on t ∈ [0, 10] s, 201 equispaced points, g = 9.81. Solver: scipy `solve_ivp` (RK45, rtol 1e-8, atol 1e-10).

| ID | Nominal simulator | True system adds | Note |
|---|---|---|---|
| S1 | φ'' + c φ' + (g/L) sin φ = 0 | − ε·φ'·abs(φ') on the right-hand side (quadratic drag) | Discrepancy largest at high velocity |
| S2 | same | − ε·g·tanh(φ'/δ) (smoothed Coulomb friction) | Discrepancy largest at low amplitude; δ fixed in the pilot (default 0.05 rad/s) |

- **Parameter box θ = (c, L, φ0, ω0)** (proposed defaults, to confirm in the pilot so that all trajectories stay non-chaotic and resolvable): c ∈ [0.05, 0.5] s⁻¹, L ∈ [0.5, 2.0] m, φ0 ∈ [0.1, 2.5] rad, ω0 ∈ [−1, 1] rad/s.
- **Mismatch levels:** ε ∈ {0, 0.25, 0.5, 1, 2} × ε_ref, with ε_ref set per system in the pilot (section 9).
- **Observation noise on true-system data:** Gaussian, σ_obs = 1% of the per-trajectory standard deviation of φ (main grid); 5% as a secondary analysis on S1 (section 8).
- **Nominal training data:** n_nom ∈ {128, 512, 2048} scrambled Sobol points, noise-free nominal simulations.
- **Test set:** 2,000 scrambled Sobol points in the box, true-system trajectories without noise, shared across arms within a seed. Never queried by any arm.

## 4. Models

- **Nominal surrogate:** MLP deep ensemble, M = 5, about 30k parameters, θ → 201-point trajectory. Same architecture and schedule for all cells; hyperparameters frozen in the pilot.
- **Correction model:** additive discrepancy ensemble δ(θ) (small MLP, M = 5) fitted on (true observation − nominal-surrogate prediction) at the queried true points. Corrected prediction = f̂_nom + δ.
- **GP reference:** GP with ARD Matérn-5/2 kernel on the same residuals, outputs reduced to the leading PCA components (number fixed in the pilot to keep 99.9% variance).

## 5. Error decomposition and the dial r

For each test point: approximation error a = f̂_nom − f_nom, discrepancy d = f_nom − f_true, total error e = f̂_nom − f_true = a + d. Norms are relative L2 over the trajectory, normalised by ‖f_true‖.

- **r (per seed and cell)** = median‖d‖ / median‖a‖ over the test set. r = 0 when ε = 0.
- **Smoothness proxy** (reported, not tested): ratio of discrete H¹-seminorm to L2 norm of d, median over the test set.

## 6. RQ1 design

**Cells:** 2 systems × 5 ε × 3 n_nom = 30 cells, 10 seeds each.

**Uncertainty signals** (for the nominal surrogate predicting the true system):
1. Nominal ensemble σ (mean over time of per-point std).
2. MC dropout σ (dropout 0.1, 50 passes; a separate dropout-trained surrogate).
3. Nominal-ODE residual of the surrogate's prediction (finite differences on the predicted trajectory).
4. Split conformal calibrated on 200 nominal simulations.
5. Split conformal calibrated on n_cal ∈ {8, 16, 32, 64} true-system points (Sobol, disjoint from the test set).
6. Discrepancy-ensemble σ after fitting on n ∈ {8, 16, 32, 64} true points.
7. GP discrepancy predictive std, same n.

**Metrics** per cell, against the total error ‖e‖:
- AUROC for the event ‖e‖ > τ, with τ the 80th percentile of ‖e‖ in that cell (so AUROC measures ranking within the cell);
- Spearman correlation between signal and ‖e‖;
- AUSE (area under the sparsification-error curve);
- 90% coverage and mean width (conformal signals; Gaussian intervals from σ for the others);
- the same AUROC and Spearman computed against ‖a‖ alone and ‖d‖ alone.

## 7. RQ2 design

**Cells:** 2 systems × 5 ε at n_nom = 512, 10 seeds. Extra S1 cells at n_nom ∈ {128, 2048} at the ε level whose pilot r is nearest 1.

**Loop:** 8 initial true points (scrambled Sobol, shared across arms in a seed), then 6 rounds of 8 (n_max = 56). Each round every arm scores the same fresh pool of 4,096 Sobol candidates; the correction model is refitted after every round.

**Arms:**
1. Random.
2. Sobol (continuation of the initial sequence).
3. Nominal ensemble σ.
4. Nominal-ODE residual of the surrogate prediction.
5. Discrepancy-ensemble σ (retrained each round).
6. GP discrepancy predictive variance.
7. Observed-residual model: nominal-ODE residual computed on each observed true trajectory (derivatives from a smoothing spline, smoothing set in the pilot), its time-averaged magnitude regressed on θ with a GP, candidates scored by the predicted magnitude.

**Batch rule for arms 3-7:** stochastic batch sampling without replacement with probability ∝ score^k (k frozen in the pilot), so that no arm loses only because greedy batches cluster.

**Metrics** on the test set after each round, for the corrected prediction: median, p95 and p99 relative L2 error; budget-AUC of the median error over n ∈ {8, 16, ..., 56} (trapezoid on log error).

## 8. Analysis plan

**Unit of replication:** seed (seed 0 = pilot, excluded; seeds 1-10 = main). Within a seed all arms share the initial set, pools, test set and nominal surrogate.

**Family A: acquisition (6 tests, Holm at α = 0.05).** For each system:
- **A1 (arm 3), A2 (arm 4):** per seed, regress G_k = log(budget-AUC_arm / budget-AUC_Sobol) on log r_k over the four nonzero ε levels. Test mean slope > 0 with a two-sided paired t-test across seeds. (Using the slope, not the raw gap, removes any offset at ε = 0, where greedy residual or error acquisition can already lose to random selection.)
- **A3 (arm 7 vs arm 6):** at ε = 2·ε_ref, per-seed difference in budget-AUC; two-sided paired t-test.

**Family B: ranking and matched transitions (4 tests, Holm at α = 0.05).** For each system:
- **B1:** per seed, slope of nominal-σ AUROC on log r over the nonzero r points (12 cells per system); test mean slope < 0, paired t-test.
- **B2 (H4):** fit a 4-parameter logistic in log r to (a) mean nominal-σ AUROC across the 15 RQ1 cells and (b) mean G for arm 3 across the RQ2 cells. Take each midpoint log r½. Equivalence test (two one-sided tests, 90% bootstrap CI over seeds, 10,000 resamples) on log r½(a) − log r½(b) with margin ±log 1.5. **If either fit does not converge, or a midpoint falls outside the sampled r range, B2 is reported as inconclusive, not as a failure or a success.**

**Effect sizes:** every test also reports the mean difference or slope with a 95% bootstrap CI (10,000 resamples, BCa) and an exact sign test as a robustness check.

**Decision rule:** a hypothesis is "supported" only if its Holm-adjusted p < 0.05 and the CI excludes 0 (or, for B2, the equivalence CI lies inside the margin). No improvement is claimed when a CI includes 0.

**Exploratory (labelled as such):** all other arm pairs, all other RQ1 signals and metrics, p95/p99 errors, 5% observation noise on S1 (arms 2, 6 and 7, all ε), the n_nom ∈ {128, 2048} RQ2 cells, the smoothness proxy, parameter-extrapolation test sets.

## 9. Pilot (seed 0 only) and what it may set

The pilot runs S1 and S2 with 3 seeds (seed 0 plus two throwaway seeds 101 and 102, both excluded) and sets only the following, by these rules:

| Quantity | Rule |
|---|---|
| Parameter box | Shrink the defaults only if any trajectory fails to solve or leaves the regime the 201-point grid resolves |
| ε_ref (per system) | The value at which r = 1 at n_nom = 512, by bisection on the pilot surrogate |
| δ (S2) | Default 0.05 rad/s unless the solver step count exceeds 10× the S1 count |
| MLP hyperparameters, epochs | Lowest validation loss on nominal data at n_nom = 512 |
| Exponent k (batch rule) | Value in {1, 2, 4} with the best mean budget-AUC across arms 3-7 at ε_ref |
| Spline smoothing (arm 7) | Generalised cross-validation on the pilot's true trajectories |
| GP PCA components | Smallest number keeping 99.9% variance |
| **Seed count** | 10, raised to at most 20 if the pilot's paired SD of the A1 slope gives under 80% power to detect a slope of half the pilot mean at α = 0.05/6 |

Fit time per network is also measured here; if the projected compute exceeds 425 CPU-hours, the 5% noise analysis is dropped first, then the n_nom ∈ {128, 2048} RQ2 cells. Any such cut is recorded in section 11.

## 10. Pre-registered outcomes

| Result | Reading |
|---|---|
| A1/A2 supported on both systems, B1 supported | Nominal-only tools go blind as r grows, and this hurts acquisition: headline result |
| B2 supported (equivalence) | The ranking failure and the acquisition failure happen at the same r: strongest version of the paper |
| B2 equivalence fails (CI outside margin) | The two failures are linked but happen at different r; report both midpoints |
| B2 inconclusive | Report the curves; no claim about matching |
| A1/A2 supported on S1 but not S2 (or the reverse) | Alignment between uncertainty and discrepancy depends on where the discrepancy lives; report as geometric coincidence |
| A3 supported | Residuals computed on observed true data rescue physics-guided acquisition |
| A3 not supported | Report it; consistent with residual acquisition being fragile (2610.27130) |
| Nothing supported | Report as a null with the achieved power from the pilot variance |

## 11. Deviations

Any change after freezing is listed here with the date, what changed and why, before the affected results are looked at.

- 2026-11-04 (before any pilot run): addendum `prereg_addendum_v1_1.md`, frozen the same day by tag `prereg-v1.1`. Adds frozen predictions P-A (RQ1 transition, Model A) and P-B (RQ2 transition and the S2 direction of H4, Model B); arm 8 (r-gated acquisition); discrepancy-augmented SPI; a new Family C of 3 tests with its own Holm correction; exploratory blocks; a cut order; and clarifies that "half its ranking ability" means the 4PL midpoint. Why: a post-freeze literature review showed H1 is largely computable from the design and the bare claim is published, so the contribution moves to predicting the transitions. Families A and B, systems, arms 1-7, seeds and metrics are unchanged.
- 2026-11-04 (before any pilot run): addendum `prereg_addendum_v1_2.md`, frozen the same day by tag `prereg-v1.2`. Adds secondary analyses G1 (horizon-resolved transition, frozen prediction P-C), G2 (identify vs correct, replacing v1.1's system-ID baseline) and G3 (measured misspecification amplification); exploratory G5 (settling-time error) and arm 9 (DEUP-style, S1, cut first); records SPI's worst-case bounds for test C2 without changing it; updates the cut order. Why: a second literature pass found these questions listed as open future work in 2606.11988, 2606.22062, 2401.12216 and 2506.07805. Families A, B and C and all earlier predictions are unchanged.
- 2026-11-05 (before any pilot run): addendum `prereg_addendum_v1_3.md`, frozen the same day by tag `prereg-v1.3`. Exploratory only: E1, semigroup error (arXiv:2605.26324) as an extra nominal-only RQ1 signal, expected to go blind like nominal σ; E2, design-clustering metrics per arm. No extra fits; no confirmatory or secondary test changes. Why: an external gap report proposed a semigroup acquisition filter and a design-collapse claim; its cited source says semigroup error cannot see the true dynamics, so the claim is tested rather than adopted.
- 2026-11-05 (before any pilot run): addendum `prereg_addendum_v1_4.md`, frozen the same day by tag `prereg-v1.4`. Deviation: P-A, P-B and P-C use the exact simulator discrepancy d(ε) at each ε instead of the linear approximation ε·g; the linear variant is reported as a secondary comparison. Why: a simulator-only check (no learned model) showed d(ε) is 20-40% off linear across an 8× sweep, which would make the predictions miss for a known reason. No test, criterion, family, arm, seed or metric changes.

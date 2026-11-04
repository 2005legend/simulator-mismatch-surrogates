# Pre-registration addendum v1.1: Blind to the Missing Term

Drafted 2026-11-04. Status: **draft, not yet frozen.** It is frozen by a git commit tagged `prereg-v1.1`, which must land before any pilot run is analysed (deadline 2026-11-24; pilot starts 2026-11-25). It amends `preregistration.md` (tag `prereg-v1`) and is logged in that file's section 11.

**Why this addendum exists.** A literature review after `prereg-v1` (report: `reports/Mismatch study novelty upgrade.md`) found that H1 is largely computable from the design and that the bare "uncertainty misses bias" claim is already published (arXiv:2505.23506, 2608.13633). This addendum adds frozen quantitative predictions for the transitions, two methods, and exploratory blocks. **Nothing in Families A and B, the systems, arms 1-7, seeds or metrics of `prereg-v1` changes.** No pilot or main-grid data existed when this was written.

## 1. Clarification of `prereg-v1`

- "The r at which nominal σ loses half its ranking ability" (H4) means the midpoint, in log r, of the 4-parameter logistic fitted to AUROC against log r, halfway between the fitted lower and upper plateaus. This is the same quantity test B2 uses.

## 2. Prediction P-A: the RQ1 ranking transition (Model A)

**Model.** For a signal s (nominal ensemble σ; separately, MC dropout σ and the nominal residual), treat (log‖a‖², log‖g‖², log s) over the test set as trivariate Gaussian, where a = f̂_nom − f_nom and g = d(ε_ref)/ε_ref with d(ε) = f_true,ε − f_nom. The model assumes d(ε) ≈ ε·g. Because d is only approximately linear in ε for an ODE, its linearity error (‖d(ε) − ε·g‖ / ‖d(ε)‖, median over the test set, computed from the simulator at every ε at no fit cost) is reported beside every P-A and P-B result. The total error is e = a + d.

**Inputs** (measured on pilot seed 0 only, n_nom = 512, using the ε = 0 surrogate and simulator-computed g; no ε > 0 surrogate result is used):
1. s_A: standard deviation of log‖a‖²
2. s_D: standard deviation of log‖g‖²
3. ρ_sa = corr(log s, log‖a‖²)
4. ρ_sg = corr(log s, log‖g‖²)
5. ρ_ag = corr(log‖a‖², log‖g‖²)
plus the median-norm cosine between a and g, used only for the fallback rule below.

**Predicted quantities, per system and per signal:**
- The AUROC-versus-log r curve for the event ‖e‖ > 80th percentile, computed by Monte Carlo (10⁶ draws) from the fitted Gaussian with log‖e‖² linearised at the medians (weight w = 1/(1 + r²)).
- Its 4PL midpoint r½ (closed form in the no-alignment case: r½ = 3^(1/4)·(s_A/s_D)^(1/2); general case by Monte Carlo).
- Its 4PL slope in natural-log units (predicted near 2).
- Collapse: curves for n_nom ∈ {128, 512, 2048} collapse onto one curve in r **if and only if** s_A, ρ_sa and ρ_ag measured at ε = 0 differ by less than 10% across n_nom. Whether this holds is checked at ε = 0 before any ε > 0 cell is examined, and the collapse prediction is stated accordingly.

**Fallback (frozen now).** If on pilot seed 0, s_A or s_D exceeds 1.0, or the absolute a-g cosine exceeds 0.3, the closed-form linearisation is not trusted. The prediction is then made with the empirical Monte-Carlo plug-in instead: recombine the seed-0 arrays σ_i, a_i, g_i at each ε and compute AUROC directly. Which variant was used is reported.

**Success criterion (per system, nominal ensemble σ, the primary signal):** the predicted r½ lies inside the empirical 90% bootstrap CI of r½ from seeds 1-10 (10,000 resamples) **and** within a factor of 1.5 of its point estimate. The slope prediction is reported against its CI but is secondary. P-A results for the other two nominal-only signals are secondary.

## 3. Prediction P-B: the RQ2 acquisition transition (Model B)

**Model.** Replace the discrepancy ensemble with the arm-6 GP / kernel ridge corrector, hyperparameters frozen at their pilot-seed-0 values at ε_ref. Arms 2, 3 and 4 do not use true-system data to choose points, so for each seed their designs are fixed. The expected integrated squared error of the corrected surrogate on the test set is then exactly Q_aa + 2εQ_ag + ε²Q_gg + N for each arm (quadratic forms in the saved a, g and the 56-point design).

**Predicted quantities, per system:**
1. The sign of each plateau of G = log(IMSE_arm / IMSE_Sobol) for arms 3 and 4 at ε → 0 and at large ε.
2. The crossover ε (positive root of the quadratic difference), if one exists, and the effective-ratio midpoint r_eff² = ε²Q_gg/Q_aa.
3. Whether H2 should be null: predicted null when an arm's Q_gg ratio to Sobol equals its Q_aa ratio within 10%.
4. **The H4 direction.** On S1, |log r½(RQ2) − log r½(RQ1)| < log 1.5. On S2, log r½(RQ2) < log r½(RQ1), i.e. the acquisition transition comes at a smaller raw r than the ranking transition, because the friction discrepancy is rougher.

**Inputs** are computed from pilot seed 0 and the fitted pilot GP hyperparameters only.

**Success criteria.** (a) Plateau signs and crossover existence match the MLP-corrected results on seeds 1-10 (secondary, descriptive). (b) S2 direction: the 90% bootstrap CI of log r½(RQ2) − log r½(RQ1) on S2 lies entirely below 0. (c) S1: the equivalence test B2 of `prereg-v1`, unchanged. (b) is the primary P-B criterion. It is tested when the S2 grid is complete (planned for the March paper).

## 4. New method arm: arm 8, r-gated acquisition (S1 grid; S2 if compute allows)

**Estimate of r after each round.**
- Numerator: median over the n observed true points of the debiased discrepancy norm, sqrt(max(‖y_i − f_nom(θ_i)‖² − T·σ_obs², 0)), relative to ‖y_i‖ (T = 201 time points; σ_obs known).
- Denominator: median over the current 4,096-point candidate pool of ‖f̂_nom(θ) − f_nom(θ)‖ / ‖f_nom(θ)‖ (nominal solves only).
- 90% CI by percentile bootstrap over the observed points (1,000 resamples).

**Rule (three-way, monotone, checked after every round, never switches back):**
1. Start with arm 3 (nominal ensemble σ).
2. Switch to Sobol (continuing the shared sequence) once the CI lower bound of r̂ exceeds 1.
3. Switch to arm 7 (observed residual) once the CI lower bound of r̂ exceeds 2 **and** at least 16 true points have been observed.

Batch rule, pool, initial set and budget are the same as arms 1-7 (`prereg-v1` section 7). Thresholds 1 and 2 are fixed now and are not tuned in the pilot.

## 5. New method: discrepancy-augmented synthetic-powered predictive inference (SPI)

Applied post hoc to the RQ1 models; no new fits.
- **Score:** relative L2 trajectory error of the corrected prediction.
- **Target coverage:** 90% (α = 0.1). **SPI β = 0.4** (the value in arXiv:2505.13432). Implementation from the authors' public code.
- **Synthetic scores, plain SPI:** 2,000 nominal simulations (Sobol, disjoint from training and test sets), scored as if they were true points.
- **Synthetic scores, augmented SPI:** the same 2,000 θ, with the nominal trajectory plus δ_m(θ) from a member m drawn uniformly from the discrepancy ensemble fitted on the same n_cal true points.
- **Real calibration points:** n_cal ∈ {8, 16, 32, 64} true points (Sobol, disjoint from the test set), as in `prereg-v1` signal 5.
- Compared with true-only split conformal at the same n_cal.

## 6. Family C (3 tests, Holm-corrected within the family at α = 0.05, S1 only)

Families A and B of `prereg-v1` are unchanged and corrected separately.

- **C1 (gate).** For each arm j and seed, regret_j = mean over the 5 ε levels of (budget-AUC_j − budget-AUC of the arm with the lowest seed-mean budget-AUC at that ε, among arms 1-7). Test: arm 8's regret is lower than that of the fixed arm with the lowest mean regret. Two-sided paired t-test across seeds 1-10.
- **C2 (SPI validity).** Per seed, coverage of augmented SPI averaged over the 5 ε levels, at n_cal = 8 and at n_cal = 16 (the larger of the two p-values is the test's p-value). Test: coverage > 0.87 (one-sided t-test, H0: coverage ≤ 0.87). Mean widths are reported beside it.
- **C3 (SPI dose-response).** Per seed, slope of log mean interval width on log r over the four nonzero ε levels, for augmented and for plain SPI at n_cal = 16. Test: augmented slope < plain slope, two-sided paired t-test.

Effect sizes with 95% BCa bootstrap CIs and exact sign tests are reported as in `prereg-v1` section 8. The same three tests on S2 are reported as secondary replications, without correction.

## 7. Exploratory blocks (no confirmatory tests)

- **Acquisition-concentration (Goodhart) curve.** Pilot: arm 3 on S1 with k ∈ {0, 1, 4, greedy}. Gate: run the reduced main sweep (k ∈ {0, 1, 4, greedy} × ε ∈ {0.5, 1, 2}·ε_ref × 10 seeds, arm 3, S1; about 3,600 fits) only if, at some ε, the pilot's best-mean k is 1 or 4 (an interior optimum). Fit gain against sqrt(KL(q_k ‖ uniform pool)). The pilot's choice of the frozen k for arms 3-7 (`prereg-v1` section 9) is not affected.
- **Exact error decomposition.** For every arm and ε on S1, split corrected error into misspecification bias, estimation error and the cross term of arXiv:2506.07805, using one large-n (2,048 true points) discrepancy fit per cell as the best-in-class reference.
- **System-identification baseline.** Refit c and L by least squares on the observed true points, then apply the arm-2 (Sobol) design with the refitted nominal model. Reported beside arms 1-8.

## 8. Cut order if compute exceeds the 425 CPU-hour ceiling

Measured in the timing pilot. Cut in this order until the projection fits:
1. The k-sweep main run (keep its pilot).
2. Arm 8 on S2.
3. The system-identification baseline.
4. Then the cuts already listed in `prereg-v1` section 9 (5% noise analysis, then the extra n_nom RQ2 cells).

The S3 PDE system is not part of this addendum. It gets its own addendum `prereg-v1.2`, frozen before its pilot is analysed.

## 9. Order of events

1. Commit this file with tag `prereg-v1.1` before 2026-11-24.
2. Run the pilot (seeds 0, 101, 102; all excluded from analysis).
3. Compute the P-A and P-B predicted numbers from seed 0. Commit them in `predictions_v1.md` with tag `pred-v1` **before any seed 1-10 run starts**.
4. Run the main grid and test.

Tags are local, so the paper says "pre-registered in the project repository before the pilot", not external pre-registration.

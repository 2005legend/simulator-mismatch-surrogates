# Pre-registration addendum v1.2: Blind to the Missing Term

Drafted 2026-11-04. Status: **frozen 2026-11-04** by the git commit tagged `prereg-v1.2`, before any pilot run (pilot starts 2026-11-25). It amends `preregistration.md` (tag `prereg-v1`) and `prereg_addendum_v1_1.md` (tag `prereg-v1.1`) and is logged in section 11 of `preregistration.md`.

**Why this addendum exists.** A second literature pass (`gap_mining_2026-11-04.md`) mined the limitations and future-work sections of the closest 2024-2026 papers. It found open questions our design can answer at little or no extra cost, and a risk in test C2 of v1.1. No pilot or main-grid data existed when this was written.

**What does not change.** Families A, B and C, their tests and Holm corrections, the systems, arms 1-8, seeds, metrics and predictions P-A and P-B are unchanged. Everything below is a **secondary** analysis (one pre-stated test each, uncorrected, reported as secondary) or **exploratory** (no test).

## 1. G1: horizon-resolved transition ("time to blindness") — secondary, with frozen prediction P-C

**Motivation.** Sale et al. (arXiv:2606.11988) state, qualitatively and on this same pendulum with viscous vs Coulomb friction, that structural error dominates at long prediction horizons and that Coulomb friction shows mainly near rest. This analysis makes that quantitative.

**Windows.** Prefix horizons T_h ∈ {2, 4, 6, 8, 10} s of each 201-point trajectory (t ∈ [0, T_h]). For each window, recompute from saved arrays (no new fits): ‖a‖, ‖d‖, ‖e‖, r(T_h), every RQ1 signal restricted to the window (σ averaged over the window's time points), and the RQ1 metrics.

**Prediction P-C (frozen in `pred-v1`, computed from pilot seed 0 only).** Apply Model A (section 2 of v1.1, including its fallback rule) separately to each window, using window-specific inputs. For each system and each nonzero ε level, P-C states:
- the predicted nominal-σ AUROC in each window;
- the predicted blindness horizon t*(ε): the smallest T_h whose predicted AUROC is below the midpoint between that window's predicted ε = 0 AUROC and its predicted large-r plateau. If no window qualifies, t*(ε) = "> 10 s".

**Secondary test (per system).** Over the four nonzero ε levels, compare the observed t*(ε) from seeds 1-10 (computed the same way from seed-mean AUROC) with the predicted t*(ε). Success: the observed and predicted horizons are equal or adjacent windows for at least 3 of the 4 ε levels. Reported with the full window-by-ε table.

**Directional statement (descriptive, not a test).** The fraction of ‖d‖ accumulated after t = 6 s is reported per system, computed from the simulator alone. It is expected to be larger for S2 (friction acts near rest) than for S1; this is a property of the simulators, stated so that P-C's per-system differences can be read against it.

## 2. G2: identify vs correct under structural mismatch — secondary

**Motivation.** 2606.22062 leaves the structural-mismatch case of "identify first or correct first" as future work. The nominal inputs c and L are known here, but the missing terms are dissipative, so an *effective* damping can absorb part of them.

**Identification.** From the true points available at n = 24 and n = 56 of **arm 2 (Sobol)** in each seed, fit an effective damping map c′ = a·c + b (two scalars) by least squares between the nominal ODE trajectories at (c′, L, φ0, ω0) and the observed true trajectories. Multi-start (5 starts), bounds chosen so that c′ stays within [0.01, 0.6] over the box (slightly wider than the training box; extrapolation is reported).

**Three predictors, each evaluated on the test set at n = 24 and n = 56:**
1. **Identify only:** f̂_nom(a·c + b, L, φ0, ω0). No network fits.
2. **Correct only:** arm 2's corrected predictor, f̂_nom(θ) + δ(θ) (already computed).
3. **Identify then correct:** f̂_nom(θ′) + δ′(θ), with δ′ a new discrepancy ensemble (same architecture, M = 5) fitted on the residuals after identification. About 1,000 extra fits (2 systems × 5 ε × 10 seeds × 2 n × 5 members).

**Secondary test (per system).** At n = 56 and ε = 2·ε_ref: median relative L2 error of "identify then correct" vs "correct only", two-sided paired t-test across seeds 1-10. All three predictors are reported against r at every ε and both n.

This replaces the "system-identification baseline" in the exploratory block of v1.1 section 7.

## 3. G3: measured misspecification amplification — secondary

**Motivation.** Amortila et al. (arXiv:2401.12216) ask how common misspecification amplification is in practice; Tang et al. (arXiv:2506.07805) leave the tightness of their bound open.

**Density ratio.** For each arm, ε and seed, fit a logistic-regression classifier with quadratic features of θ to separate the final 56-point design from the 2,000 test points (class-balanced weights). The estimated density ratio at each test point follows from the classifier odds. Summary: C_eff = 95th percentile of the estimated ratio over test points (the supremum is too unstable).

**Prediction (stated now).** At large mismatch, arms whose designs are less representative of the test distribution pay more: the slope of log(corrected error_arm / corrected error_Sobol) on log C_eff is **positive**.

**Secondary test (per system).** At ε = 2·ε_ref, per seed, regress log(error_arm/error_Sobol) on log C_eff across arms 3-8; test mean slope > 0 with a one-sided t-test across seeds 1-10. The same slope at ε = 0 is reported as the control (expected near 0). The relation to Model B's prediction is reported descriptively.

## 4. G5: derived-quantity error — exploratory

For every arm and cell, report the error in **settling time**: the first time after which |φ(t)| < 0.05 rad for the rest of the window, compared between corrected prediction and true trajectory (trajectories that never settle within 10 s are flagged and reported separately). No test.

## 5. G4: DEUP-style arm 9 — exploratory, S1 only, cut first

Arm 9 follows Lahlou et al. (arXiv:2102.08501). At each round, the errors observed at newly acquired true points *before* they were added to training (‖y − corrected prediction‖, out-of-sample by construction) are accumulated; a GP on θ is fitted to their logarithm; candidates are scored by the predicted error and sampled with the frozen batch rule. Same initial set, pool and budget as arms 1-8. About 1,500 fits. No test.

## 6. Test C2 of v1.1: recorded worst-case floor (no change to the test)

SPI's worst-case coverage (Theorem 3.5 of arXiv:2505.13432) at α = 0.1, β = 0.4 lies well below 0.9 for small real calibration sets, and misaligned synthetic scores push coverage toward it. Test C2 (coverage > 0.87 at n_cal = 8 and 16) is **kept unchanged**. Before the pilot, the Theorem 3.5 lower and upper bounds for m ∈ {8, 16}, N = 2,000, α = 0.1, β = 0.4 are computed with the authors' public code (no data needed) and recorded in `predictions_v1.md` under the `pred-v1` tag. If C2 fails, the paper reports observed coverage relative to those bounds.

## 7. Updated cut order

If the timing pilot projects more than the 425 CPU-hour ceiling, cut in this order:
1. Arm 9 (G4).
2. The k-sweep main run (v1.1).
3. Arm 8 on S2 (v1.1).
4. G2 on S2 (keep S1).
5. Then the cuts listed in `prereg-v1` section 9.

G1, G3 and G5 need no fits and are never cut.

## 8. Order of events (updated)

1. Commit this file with tag `prereg-v1.2` before the pilot starts.
2. Compute the SPI bounds of section 6 (data-free).
3. Run the pilot (seeds 0, 101, 102; all excluded).
4. Compute P-A, P-B and P-C from pilot seed 0; commit them with the SPI bounds in `predictions_v1.md`, tag `pred-v1`, **before any seed 1-10 run starts**.
5. Run the main grid and test.

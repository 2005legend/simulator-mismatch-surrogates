# Experimental log: Blind to the Missing Term

All numbers below are copied from the committed results files `results_main.md` (pendulum) and `results_pde.md` (PDE systems) of the project repository. Do not round them further or change them. Do not add results that are not here.

## 1. Pre-registration trail

| Tag | Date | Content |
|---|---|---|
| prereg-v1 | 2026-11-04 | Hypotheses H1-H4, pendulum systems S1/S2, Families A (acquisition, Holm over 6) and B (ranking, Holm over 4), decision rules |
| prereg-v1.1 to v1.4 | 2026-11-04 to 11-13 | Frozen transition predictions P-A (Model A), r-gated arm 8 and test C1, horizon prediction P-C, clarifications |
| prereg-v1.5 | 2026-11-13 | After a development-seed look (seed 999 only): main claim changed to convergence to alignment; extra levels 4, 8, 16 x eps_ref; secondary predictions D1-D3 |
| prereg-v1.6 | 2026-11-15 | Family F (acquisition for a learned missing term, 4 tests, own Holm), alignment-range predictions R1, R1s, R3a, R3b; cut list (model B, C2, C3, 5% noise and others, reported as not run) |
| pred-v1 | 2026-11-20 | Transition predictions from pilot seed 0 (pendulum) |
| prereg-v1.7 | 2026-11-20 | PDE Systems 3 and 4, Family P (PB1), PA2, PD1, PR1, PR1s, PR3a, PR3b; all PDE tests secondary |
| pred-v2 | 2026-11-20 | Transition predictions from pilot seed 0 (PDE) |

Seeds: development 997-999 (never in results), pilot 0, 101, 102, main 1-20. The pilot's seed-count rule gave 20 main seeds. "k of 10" criteria are read as "2k of 20". Deviations are logged in section 11 of the pre-registration, each before the affected result was examined; one is that the original RQ1 grid script missed the n_nom = 128 and 2048 cells, which were added and run before any eps > 0 result was examined.

## 2. Systems and models

**Pendulum (S1, S2).** phi'' + c phi' + (g/L) sin phi = 0, g = 9.81, theta = (c, L, phi0, omega0) in [0.05, 0.5] x [0.5, 2.0] x [0.1, 2.5] x [-1, 1]; output phi on 201 points over [0, 10] s (RK45, rtol 1e-8). True system adds -eps phi'|phi'| (S1, quadratic drag; discrepancy largest at high velocity) or -eps g tanh(phi'/0.05) (S2, smoothed Coulomb friction; largest at low amplitude). eps_ref (r = 1 at n_nom = 512, from the pilot): S1 0.00544, S2 0.00140. Levels eps in {0, 0.25, 0.5, 1, 2} x eps_ref for all frozen tests, plus {4, 8, 16} x eps_ref for simulator-only ranking signals. Nominal surrogate: MLP deep ensemble, 5 members, 4 hidden layers, trained on n_nom in {128, 512, 2048} Sobol simulations; test set 500 Sobol points per seed. Observation noise on real data: 1% of each trajectory's standard deviation.

**System 3: parametric Burgers.** u_t + u u_x = nu u_xx, periodic on [0, 2 pi); theta = (nu, A1, A2, phase) in [0.10, 0.30] x [0.5, 1.5] x [0, 1] x [0, pi], u0 = A1 sin x + A2 sin(2x + phase). Pseudo-spectral, 256 points, 2/3 dealiasing, integrating-factor RK4, dt = 0.002; output u on 32 points at 21 times in [0, 2]. True system adds eps h with P1 h = -u|u| or P2 h = sin 3x. Surrogate: the pendulum's MLP ensemble settings, n_nom = 512. eps_ref: P1 0.00359, P2 0.00606.

**System 4: neural operator.** Same equation and missing terms, nu = 0.15; input an initial profile from a random 6-mode Fourier series scaled to max |u0| = 1.5 on 64 points; output u at 20 times on 64 points. Surrogate: ensemble of 5 one-dimensional Fourier neural operators (width 32, 16 modes, 4 layers, Adam 1e-3, 400 epochs, early stopping). eps_ref: P1 0.00580, P2 0.00677.

**Signals.** Nominal ensemble sigma (mean over output of per-point std); MC-dropout sigma (dropout 0.1, 50 passes; not on System 4); the governing-equation residual of the surrogate's prediction (ODE residual for the pendulum, PDE residual for Burgers); split conformal calibrated on 200 nominal simulations; split conformal calibrated on n in {8, 16, 32, 64} noisy real observations.

**Metrics.** AUROC for the event "E above its 80th percentile in the cell"; Spearman correlation with E, A and D; coverage and width of 90% intervals. r = median D / median A per seed and cell.

**Statistics.** Seed is the unit. Paired t-test across 20 seeds, 95% BCa bootstrap CI (10,000 resamples), exact sign test, Holm within each family. A test is supported only if Holm p < 0.05 and the CI lies on the hypothesised side; a significant effect the other way is reported as "opposite". 4PL fits in log r on seed means; seed-resampled bootstrap, a fit is "inconclusive" if more than 10% of resamples fail. Equivalence by TOST.

## 3. Pendulum results (20 seeds)

### Family B: ranking (primary H1, H4; Holm over 4)

| Test | Mean | 95% BCa CI | Holm p | Sign (+/20) | Verdict |
|---|---|---|---|---|---|
| B1 S1, slope of AUROC on log r (hyp. < 0) | +0.050 | [0.045, 0.057] | < 1e-4 | 20/20 | opposite |
| B1 S2 | -0.056 | [-0.062, -0.048] | < 1e-4 | 0/20 | supported |
| B2 S1, log r_1/2 (ranking) - log r_1/2 (acquisition), equivalence within log 1.5 | -0.399 | - | 1.0 | - | inconclusive (2,168 of 10,000 resamples dropped) |
| B2 S2 | -0.171 | - | 1.0 | - | inconclusive (4,277 dropped) |

### Family A: acquisition (primary H2, H3; Holm over 6)

Acquisition loop: 8 initial real points, then 6 rounds of 8 (56 total), pools of 4,096 Sobol candidates, correction model refitted each round. G = log(budget-AUC of arm / budget-AUC of Sobol); A1/A2 test the slope of G on log r (hyp. > 0); A3 tests arm 7 minus arm 6 budget-AUC at 2 x eps_ref (hyp. < 0).

| Test | Mean | 95% BCa CI | Holm p | Sign (+/20) | Verdict |
|---|---|---|---|---|---|
| A1 S1 (arm 3, nominal sigma) | 0.028 | [0.015, 0.041] | 0.0023 | 16/20 | supported |
| A2 S1 (arm 4, nominal residual) | 0.024 | [0.007, 0.042] | 0.0500 | 16/20 | not supported |
| A3 S1 (observed residual vs GP) | 0.008 | [0.003, 0.013] | 0.0299 | 17/20 | opposite |
| A1 S2 | -0.027 | [-0.039, -0.018] | 0.0006 | 2/20 | opposite |
| A2 S2 | -0.034 | [-0.060, -0.016] | 0.0299 | 5/20 | opposite |
| A3 S2 | 0.001 | [-0.000, 0.003] | 0.1051 | 13/20 | not supported |

At every level and on both systems, arms 3 and 4 have G between +0.08 and +0.18 (worse than Sobol); the slope tests only ask whether that gap grows with r. Figure fig5.

### Family C (C1, r-gated arm 8 vs best fixed arm, hyp. < 0)

| Test | Mean regret difference | 95% BCa CI | Holm p | Verdict |
|---|---|---|---|---|
| C1 S1 | 0.006 | [0.004, 0.007] | < 1e-4 | opposite (best fixed arm: Sobol) |
| C1 S2 | 0.004 | [0.003, 0.005] | < 1e-4 | opposite (best fixed arm: Sobol) |

### Family F: acquisition for a learned missing term (Holm over 4)

Arms F5 (disagreement among 5 learned terms) and F6 (state-space coverage) vs Sobol; one correction fit of 20 L-BFGS iterations per round (author's cost decision, pilot fit-to-fit CV up to 0.39).

| Test | Mean G | 95% BCa CI | Holm p | Verdict |
|---|---|---|---|---|
| F1 (F5, S1) | 0.184 | [0.072, 0.339] | 0.055 | not supported |
| F2 (F5, S2) | 0.123 | [-0.005, 0.258] | 0.283 | not supported |
| F3 (F6, S1) | 0.047 | [-0.067, 0.132] | 0.750 | not supported |
| F4 (F6, S2) | 0.003 | [-0.134, 0.142] | 0.964 | not supported |

Pre-registered stop rule: no arm beats Sobol, so Family F is inconclusive at this fit budget.

### Transition prediction P-A (secondary; pred-v1 from pilot seed 0)

Success requires the predicted r_1/2 inside the 90% bootstrap CI of the 20-seed r_1/2 and within a factor 1.5.

| System, signal | Observed r_1/2 | 90% CI | Predicted | Verdict |
|---|---|---|---|---|
| S1, ensemble sigma (primary) | 0.681 | [0.588, 0.762] | 0.912 | fail (within 1.5x, outside CI) |
| S1, MC dropout | 0.954 | [0.875, 1.030] | 1.246 | fail |
| S1, ODE residual | 0.631 | [0.545, 0.704] | 0.712 | fail |
| S2, ensemble sigma (primary) | 2.393 | [2.259, 2.537] | 2.589 | fail (within 1.5x, outside CI) |
| S2, MC dropout | 2.772 | [2.520, 3.058] | 3.060 | fail |
| S2, ODE residual | 2.479 | [2.332, 2.644] | 2.827 | fail |

Mean AUROC of ensemble sigma at levels 0.25-16 x eps_ref: S1 0.717, 0.780, 0.839, 0.894, 0.925, 0.937, 0.936 (r 0.27 to 11.1); S2 0.677, 0.664, 0.632, 0.551, 0.460, 0.400, 0.386 (r 0.26 to 14.1). Figure fig1.

### D-tests (secondary, dev-informed, v1.5)

- D1 (limit), rho_S(sigma, E) - rho_S(sigma, D) at 16 x within +/- 0.10 (TOST): S1 mean 0.033, equivalent; S2 mean -0.004, equivalent.
- D2 (direction), per-seed slope of AUROC on log r over the seven nonzero levels: S1 +0.062 (CI [0.056, 0.069], 20/20 positive, success); S2 -0.084 (CI [-0.087, -0.080], 0/20 positive, success).
- D3 (32 real points detect alignment): S1 aligned in 20/20 seeds at n = 32 and 64 (12/20 at n = 8, 19/20 at n = 16); S2 aligned in 0/20 at every n. Both as predicted.
- D4 (descriptive): at 16 x, distance of rho(sigma, E) to the D limit 0.033 (S1), 0.004 (S2). Curves from n_nom = 128, 512 and 2048 fall on one curve in r (fig2).
- P-C (horizon, S2): predicted and observed blindness horizon agree at all four levels; success.

### Alignment range (secondary, v1.6)

Dictionary of nine candidate terms; range over single terms and 2,000 random mixtures. At 16 x:

| | R1 inside range | R1s gap to true-family prediction <= 0.20 (4x, 16x) | R3a selected term beats sigma / >= 0.90 (n = 16) | R3b (true family removed) beats sigma |
|---|---|---|---|---|
| S1 | 20/20 (sigma and ODE residual) | 20/20 (max gap 0.017) | 20/20 / 20/20 (w|w| selected 20/20) | 20/20 (p^3 selected 19/20) |
| S2 | 20/20 | 20/20 (max gap 0.148) | 20/20 / 20/20 (tanh(w/0.1) 20/20) | 20/20 (sin 2p 19/20) |

Median statistic of the selected term: R3a 1.00 (S1), 0.998 (S2); R3b 0.85 (S1), 0.59 (S2); ensemble sigma 0.70 (S1), -0.11 (S2). Figures fig3, fig4.

## 4. PDE results (Systems 3 and 4, 20 seeds; all secondary)

### Family P: PB1, slope of ensemble-sigma AUROC on log r < 0 (Holm over 4)

| Combination | Mean slope | 95% BCa CI | Holm p | Seeds with slope > 0 | Verdict |
|---|---|---|---|---|---|
| Burgers P1 | -0.0616 | [-0.0659, -0.0571] | 5.4e-16 | 0/20 | supported |
| Burgers P2 | -0.0904 | [-0.0938, -0.0870] | 4.5e-21 | 0/20 | supported |
| Operator P1 | -0.0090 | [-0.0127, -0.0046] | 0.00037 | 3/20 | supported (weak; AUROC falls then partly recovers) |
| Operator P2 | -0.0540 | [-0.0594, -0.0490] | 9.8e-14 | 0/20 | supported |

### PA2 (transfer of the pilot-seed prediction, ensemble sigma)

| Combination | Observed r_1/2 | 90% CI | Predicted | Verdict | Large-r AUROC observed / predicted |
|---|---|---|---|---|---|
| Burgers P1 | 3.642 | [3.507, 3.783] | 3.703 | success | 0.729 / 0.743 |
| Burgers P2 | 2.185 | [2.104, 2.267] | 2.697 | fail | 0.617 / 0.584 |
| Operator P1 | 0.525 | [0.518, 8.013] | 0.502 | fail (just below CI; midpoint poorly identified, flagged before the runs) | 0.627 / 0.645 |
| Operator P2 | 1.272 | [1.179, 1.376] | 1.027 | fail | 0.551 / 0.508 |

Figure fig6.

### PD1 (limit; TOST within +/- 0.10 at 16 x)

| Combination | Mean difference | Equivalent | rho(sigma, E) / rho(sigma, D) |
|---|---|---|---|
| Burgers P1 | 0.022 | yes | 0.18 / 0.16 |
| Burgers P2 | 0.016 | yes | 0.11 / 0.09 |
| Operator P1 | -0.016 | yes | 0.06 / 0.08 |
| Operator P2 | 0.003 | yes | -0.12 / -0.12 |

### Alignment (PR1, PR1s, PR3a, PR3b; need >= 18 of 20)

All four combinations: PR1, PR1s (4x and 16x), PR3a (beats sigma and >= 0.90) and PR3b met in 20 of 20 seeds, for ensemble sigma and the PDE residual. Largest PR1s gap: 0.046 (Burgers P1 sigma), 0.070 (Burgers P1 residual), 0.035 (operator P1), 0.003 (P2 cases). The PDE residual is aligned with D on Burgers P1 (rho 0.43) and anti-aligned on Burgers P2 (rho -0.27). Term selected at n = 16 (R3a): Burgers P1 u|u| 20/20; operator P1 u|u| 20/20; Burgers P2 sin 3x 16, cos 3x 1, sin 2x 3; operator P2 cos 3x 9, sin 2x 7, sin 3x 4. Caveat stated before the runs: on P2 the sin(kx) responses are nearly collinear, so PR3a/PR3b there are close to automatic, and a term outside the true family (sin 2x) is sometimes selected and still passes. Figure fig7.

### Descriptive

- Conformal intervals calibrated on nominal simulations cover 0% of true errors at 16 x in all four PDE combinations; calibrated on 16 real fields they cover 94.2% to 96.9%.
- D3 on the PDE systems (low power, no prediction): at 16 x, aligned in 0 to 4 of 20 seeds for n = 8 to 64.

## 5. Not run (cut before any registered-seed result, with reason in the pre-registration)

Model B and prediction P-B; synthetic-powered intervals and tests C2, C3; the k-curve main run; arm 9; the semigroup signal; extra n_nom acquisition cells; the 5% noise analysis; discrepancy ensemble, GP and acquisition on the PDE systems.

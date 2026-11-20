# Frozen predictions v1 (`pred-v1`): Blind to the Missing Term

Computed 2026-11-20 from **pilot seed 0 only** (code at commit `883997b` plus the Model A files committed with this document), before any seed 1-20 run. Status: **frozen 2026-11-20** by the commit tagged `pred-v1`; changes after this go in section 11 of `preregistration.md`. Inputs: the eps = 0 nominal ensemble and MC-dropout surrogates of seed 0 (n_nom = 512, settings from `pilot_results/frozen_params.json`), refitted exactly as the RQ1 runner fits them (they reproduce the pilot's saved arrays: largest relative difference 1.5e-12), and simulator discrepancies d(eps) on 29 log-spaced levels from eps_ref/256 to 64 eps_ref. No surrogate trained at eps > 0 is used. Raw output: `predictions/model_a_seed0.json`. Code: `mismatch/model_a.py`, `scripts/run_model_a.py`, `scripts/write_predictions.py`.

Main seeds are 1-20 (the pilot's seed-count rule; changelog 2026-11-20), so every 'seeds 1-10' in v1.1 and v1.2 reads 'seeds 1-20'.

## 1. Fallback decision (v1.1 section 2, frozen rule)

| System | s_A | s_D | median cos(a, d) | cos at median-norm point | Fallback |
|---|---|---|---|---|---|
| S1 | 1.769 | 2.091 | 0.168 | -0.265 | **yes** (s_A > 1) |
| S2 | 1.784 | 1.161 | 0.064 | -0.022 | **yes** (s_A > 1) |

The fallback fires on both systems and for all three signals, so **the prediction is the empirical plug-in**: the seed-0 arrays are recombined at each eps, e_i = a_i + d_i(eps), and AUROC is computed directly. The Gaussian-copula variant is reported as secondary.

**What this means for the test (stated before any main run).** The nominal surrogate is never retrained at eps > 0, so the plug-in AUROC at each eps equals the RQ1 AUROC that pilot seed 0 itself shows (checked: identical to four decimals at every level). Under the fallback, P-A predicts the main seeds' transition from one seed's arrays: it tests whether the transition is a stable property of the setup across seeds, not whether the Gaussian model is right. The Gaussian variant is the model-based prediction and is scored alongside it as secondary.

## 2. P-A, primary: nominal ensemble sigma, exact d, empirical plug-in

Success (v1.1): the predicted r_1/2 lies inside the 90% bootstrap CI of r_1/2 from seeds 1-20 (10,000 resamples) and within a factor of 1.5 of its point estimate. Positive slope = AUROC falls with r; negative = it rises (the aligned case).

| System | r_1/2, fit on measured levels **(primary)** | r_1/2, fit on full grid | slope (levels / grid) | AUROC plateau small r / large r (grid fit) | AUROC at eps = 0 |
|---|---|---|---|---|---|
| S1 | **0.912** | 0.835 | -1.49 / -1.60 | 0.704 / 0.944 | 0.705 |
| S2 | **2.589** | 2.308 | 2.06 / 2.48 | 0.704 / 0.370 | 0.705 |

Predicted AUROC (event ||e|| > 80th percentile) and r at each level (eps as a multiple of eps_ref):

| System | quantity | 0.25 | 0.5 | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|---|---|---|
| S1 | r | 0.260 | 0.513 | 1.000 | 1.936 | 3.578 | 6.353 | 10.528 |
| S1 | AUROC | 0.739 | 0.787 | 0.836 | 0.891 | 0.930 | 0.946 | 0.941 |
| S2 | r | 0.250 | 0.500 | 1.000 | 1.975 | 3.857 | 7.536 | 13.393 |
| S2 | AUROC | 0.699 | 0.691 | 0.668 | 0.567 | 0.448 | 0.384 | 0.343 |

**Decision A (author, 2026-11-20, before any main run).** The main-run r_1/2 is fitted to the seven measured levels (r about 0.25 to 13). The primary prediction fits the predicted curve to the same seven levels, so both sides are computed the same way. The full-grid fit, read literally from v1.1 ('its 4PL midpoint'), is kept beside it. They differ by about 10%.

## 3. P-A, secondary

| System | signal | variant | r_1/2 (levels) | slope (levels) | r_1/2 (grid) |
|---|---|---|---|---|---|
| S1 | ens_sigma | exact d, gaussian | 0.513 | 33.91 | 0.430 |
| S1 | ens_sigma | linear d, empirical | 0.949 | -1.79 | 0.848 |
| S1 | mc_sigma | exact d, empirical | 1.246 | -1.27 | 1.173 |
| S1 | mc_sigma | exact d, gaussian | 0.613 | -4.58 | 0.584 |
| S1 | mc_sigma | linear d, empirical | 1.258 | -1.58 | 1.158 |
| S1 | ode_resid | exact d, empirical | 0.712 | -1.13 | 0.821 |
| S1 | ode_resid | exact d, gaussian | 0.511 | 39.19 | 0.408 |
| S1 | ode_resid | linear d, empirical | 0.799 | -1.38 | 0.865 |
| S2 | ens_sigma | exact d, gaussian | 1.436 | 2.04 | 1.384 |
| S2 | ens_sigma | linear d, empirical | 2.586 | 1.96 | 2.386 |
| S2 | mc_sigma | exact d, empirical | 3.060 | 1.90 | 2.630 |
| S2 | mc_sigma | exact d, gaussian | 1.444 | 1.72 | 1.387 |
| S2 | mc_sigma | linear d, empirical | 3.226 | 1.75 | 2.895 |
| S2 | ode_resid | exact d, empirical | 2.827 | 1.87 | 2.603 |
| S2 | ode_resid | exact d, gaussian | 1.633 | 1.83 | 1.674 |
| S2 | ode_resid | linear d, empirical | 2.628 | 1.86 | 2.439 |

The Gaussian variant misses the S1 shape (it peaks near 1 x eps_ref and then falls, and its seven-level fit gives an unstable slope), the failure the fallback rule anticipates when s_A > 1.

Closed form, no alignment (v1.1), r_1/2 = 3^(1/4) (s_A/s_D)^(1/2): S1 1.211, S2 1.631. It ignores the measured alignment (S1 rho_sg = 0.768) and is for reference only.

Linearity error of d(eps) = eps g (median ||d - eps g|| / ||d||, v1.4):

| System | 0.25 | 0.5 | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|---|---|
| S1 | 0.029 | 0.020 | 0.000 | 0.039 | 0.115 | 0.264 | 0.546 |
| S2 | 0.013 | 0.008 | 0.000 | 0.017 | 0.051 | 0.127 | 0.311 |

## 4. Collapse across n_nom (v1.1)

Rule: the curves for n_nom in {128, 512, 2048} collapse onto one curve in r if and only if s_A, rho_sa and rho_ag at eps = 0 differ by under 10% across n_nom (relative spread = (max - min) / |value at 512|).

| System | signal | s_A | rho_sa | rho_ag | collapse predicted |
|---|---|---|---|---|---|
| S1 | ens_sigma | 0.064 | 0.581 | 2.285 | **no** |
| S1 | mc_sigma | 0.064 | 3.029 | 2.285 | **no** |
| S1 | ode_resid | 0.064 | 0.505 | 2.285 | **no** |
| S2 | ens_sigma | 0.064 | 0.581 | 0.384 | **no** |
| S2 | mc_sigma | 0.064 | 3.029 | 0.384 | **no** |
| S2 | ode_resid | 0.064 | 0.505 | 0.384 | **no** |

Collapse is **not predicted** for any system or signal: s_A is stable (6% spread) but the correlations change with n_nom. Under prereg-v1 the extra n_nom cells are S1 only, at the level whose pilot r is nearest 1 (1 x eps_ref).

## 5. P-C: blindness horizon (v1.2 section 1, secondary)

Model A per prefix window (the fallback fires in every window, so the plug-in is used), nominal ensemble sigma, exact d. t* = smallest window whose AUROC is below the midpoint between that window's eps = 0 AUROC and its large-r plateau.

**S1** (median fraction of ||d||^2 after 6 s at eps_ref: 0.459).

| window (s) | AUROC eps = 0 | large-r plateau | 0.25 | 0.5 | 1 | 2 |
|---|---|---|---|---|---|---|
| 2 | 0.561 | 0.920 | 0.568 | 0.595 | 0.637 | 0.733 |
| 4 | 0.585 | 0.937 | 0.616 | 0.666 | 0.780 | 0.838 |
| 6 | 0.593 | 0.950 | 0.685 | 0.755 | 0.818 | 0.872 |
| 8 | 0.663 | 0.943 | 0.719 | 0.769 | 0.824 | 0.883 |
| 10 | 0.705 | 0.944 | 0.739 | 0.787 | 0.836 | 0.891 |

Predicted t*: 0.25 x: 2 s, 0.5 x: 2 s, 1 x: 2 s, 2 x: 2 s.

**S2** (median fraction of ||d||^2 after 6 s at eps_ref: 0.679).

| window (s) | AUROC eps = 0 | large-r plateau | 0.25 | 0.5 | 1 | 2 |
|---|---|---|---|---|---|---|
| 2 | 0.561 | 0.256 | 0.556 | 0.556 | 0.548 | 0.543 |
| 4 | 0.585 | 0.261 | 0.584 | 0.585 | 0.596 | 0.558 |
| 6 | 0.593 | 0.298 | 0.616 | 0.635 | 0.611 | 0.557 |
| 8 | 0.663 | 0.329 | 0.674 | 0.665 | 0.637 | 0.549 |
| 10 | 0.705 | 0.370 | 0.699 | 0.691 | 0.668 | 0.567 |

Predicted t*: 0.25 x: > 10 s, 0.5 x: > 10 s, 1 x: > 10 s, 2 x: > 10 s.

The directional statement holds in the simulator: more of ||d|| arrives after 6 s for S2 than for S1.

**Decision B (author, 2026-11-20, before any main run).** The frozen t* definition assumes AUROC falls towards its large-r plateau. On S1 it rises (the aligned case: the plateau is above 0.92 in every window), so read literally t* = 2 s at every level, which only says AUROC has not yet risen halfway, not that the signal goes blind. The literal value stays in the table, and the P-C test is reported as **not applicable on S1** because its large-r plateau lies above its eps = 0 AUROC in every window; S2 is scored as frozen. On S2 t* is '> 10 s' at all four levels: the curve has not reached its midpoint by 2 x eps_ref.

## 6. Not predicted (cut, logged in v1.6)

P-B (Model B), the SPI bounds (C2, C3) and the 5% noise analysis were cut before any pilot run (addendum v1.6, `preregistration.md` section 11) and have no entry here.

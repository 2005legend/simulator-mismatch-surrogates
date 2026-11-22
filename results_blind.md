# Blind-dictionary results (addendum v1.10): rule-based libraries, drawn missing terms

Seeds 1-20. Draw from the hash of tag `prereg-v1.10a` (commit df14e5c3da49, `pilot_results/blind_draw.json`); pilot seed 0 set eps_ref only (`pilot_results/blind_eps_ref.json`). Rules: `prereg_addendum_v1_10.md` part B, frozen by tag `prereg-v1.10b` before any registered run. Generated from `runs/blind_analysis/results.json` (`scripts/run_blind.py --stage analyse`). All tests are secondary (family BL). Dictionaries: SINDy default library for the pendulum (13 terms), PDE-FIND default library for Burgers (12 terms). The residual is the ODE residual (pendulum) or the PDE residual (Burgers).

## Summary: how many of the six drawn terms meet each test

| Test | Met | Terms not met |
|---|---|---|
| BL1 limit (TOST +/- 0.10, sigma, 16 x) | 6 of 6 | - |
| BLR1 inside blind range, sigma and residual (>= 18/20) | 6 of 6 | - |
| BLR1L linear response at 4 x, both signals (>= 18/20) | 6 of 6 | - |
| BLR1L linear response at 16 x, both signals (>= 18/20) | 5 of 6 | P constant torque |
| BLR3 selected term beats sigma (>= 18/20) | 6 of 6 | - |
| BLR3s selected term statistic >= 0.90 (>= 18/20) | 3 of 6 | P aerodynamic drag, P pivot vibration, B quadratic damping |

For comparison, the hand-written dictionaries of v1.6-v1.9 met their range test in every case (R1, PR1, OR1, QR1) and their selection-beats-sigma test in all cases but one (OR3 on P4, 9 of 20).

## Per term

| Drawn term | eps_ref | Out-of-span share (median) | sigma-AUROC slope on log r [95% BCa CI], seeds > 0 | BL1 mean diff |
|---|---|---|---|---|
| P aerodynamic drag | 0.007882 | 0.00 | +0.065 [0.059, 0.072], 20/20 | 0.0087 (equivalent) |
| P pivot vibration | 0.011083 | 0.16 | -0.021 [-0.027, -0.015], 1/20 | 0.0282 (equivalent) |
| P constant torque | 0.005310 | 0.00 | -0.109 [-0.112, -0.105], 0/20 | 0.0376 (equivalent) |
| B forcing sin 3x | 0.006059 | 0.64 | -0.090 [-0.094, -0.087], 0/20 | 0.0156 (equivalent) |
| B cubic damping | 0.003601 | 0.00 | -0.069 [-0.073, -0.065], 0/20 | 0.0152 (equivalent) |
| B quadratic damping | 0.003596 | 0.01 | -0.062 [-0.066, -0.057], 0/20 | 0.0215 (equivalent) |

The slope is descriptive (no directional test was frozen, because the dev look showed both signs).

| Drawn term | signal | BLR1 inside, width, reference p | mean rho(s,D) / rho(s,linear) at 16 x | BLR1L 4 x: count, largest gap | BLR1L 16 x: count, largest gap |
|---|---|---|---|---|---|
| P aerodynamic drag | sigma | 20/20, 1.49, 0.080 (met) | 0.725 / 0.726 | 20/20, 0.003 (met) | 20/20, 0.008 (met) |
| P aerodynamic drag | residual | 20/20, 1.40, 0.037 (met) | 0.687 / 0.690 | 20/20, 0.001 (met) | 20/20, 0.009 (met) |
| P pivot vibration | sigma | 20/20, 1.49, 0.080 (met) | -0.115 / -0.162 | 20/20, 0.021 (met) | 20/20, 0.071 (met) |
| P pivot vibration | residual | 20/20, 1.40, 0.037 (met) | -0.052 / -0.079 | 20/20, 0.016 (met) | 20/20, 0.056 (met) |
| P constant torque | sigma | 20/20, 1.49, 0.080 (met) | -0.456 / -0.318 | 20/20, 0.044 (met) | 0/20, 0.151 (not met) |
| P constant torque | residual | 20/20, 1.40, 0.037 (met) | -0.343 / -0.190 | 20/20, 0.046 (met) | 0/20, 0.164 (not met) |
| B forcing sin 3x | sigma | 20/20, 1.10, 0.001 (met) | 0.094 / 0.095 | 20/20, 0.001 (met) | 20/20, 0.001 (met) |
| B forcing sin 3x | residual | 20/20, 1.40, 0.036 (met) | -0.271 / -0.271 | 20/20, 0.000 (met) | 20/20, 0.001 (met) |
| B cubic damping | sigma | 20/20, 1.10, 0.001 (met) | 0.223 / 0.219 | 20/20, 0.002 (met) | 20/20, 0.006 (met) |
| B cubic damping | residual | 20/20, 1.40, 0.036 (met) | 0.546 / 0.545 | 20/20, 0.001 (met) | 20/20, 0.002 (met) |
| B quadratic damping | sigma | 20/20, 1.10, 0.001 (met) | 0.162 / 0.162 | 20/20, 0.001 (met) | 20/20, 0.003 (met) |
| B quadratic damping | residual | 20/20, 1.40, 0.036 (met) | 0.426 / 0.430 | 20/20, 0.002 (met) | 20/20, 0.006 (met) |

## Selection from 16 noisy observations at 4 x

| Drawn term | BLR3 beats sigma | BLR3s statistic >= 0.90 | median statistic vs sigma | selected terms (n = 16) | beats sigma at n = 8, 16, 32, 64 |
|---|---|---|---|---|---|
| P aerodynamic drag | 20/20 (met) | 15/20 (not met) | 0.933 vs 0.715 | w^3 9, p*w^2 6, w 5 | 20, 20, 20, 20 |
| P pivot vibration | 20/20 (met) | 0/20 (not met) | 0.620 vs -0.150 | p 14, sin(p) 5, p^2 1 | 20, 20, 20, 20 |
| P constant torque | 20/20 (met) | 20/20 (met) | 0.904 vs -0.365 | cos(p) 20 | 20, 20, 20, 20 |
| B forcing sin 3x | 20/20 (met) | 20/20 (met) | 0.925 vs 0.103 | 1 20 | 20, 20, 20, 20 |
| B cubic damping | 20/20 (met) | 20/20 (met) | 1.000 vs 0.217 | u^3 20 | 20, 20, 20, 20 |
| B quadratic damping | 20/20 (met) | 15/20 (not met) | 0.947 vs 0.157 | u^3 15, u*u_x 4, u^3*u_x 1 | 20, 20, 20, 20 |

## Expectations frozen in part B, and what happened

- BL1, BLR1, BLR1L at 4 x, BLR3: expected met on all six; met on all six.
- BLR1L at 16 x: expected not met on constant torque, borderline on pivot vibration, met on the others; not met on constant torque (0/20), met on pivot vibration (largest gap 0.071) and on the others.
- BLR3s: expected met on aerodynamic drag, forcing sin 3x, cubic damping and quadratic damping, not met on pivot vibration, uncertain on constant torque. Met on forcing sin 3x, cubic damping and constant torque; **not met on aerodynamic drag (15/20) and quadratic damping (15/20), contrary to expectation**; not met on pivot vibration (0/20), as expected. In the failing seeds the observations selected a weaker library term (w or p w^2 instead of w^3; u u_x instead of u^3).
- Range containment is weak evidence on the pendulum (reference p 0.080 for sigma, 0.037 for the residual) and stronger for sigma on Burgers (0.001).
- The draw repeated constant torque, already tested in v1.9 (Q1). With the same seeds, term and eps rule, its limit, AUROC and linear-response results reproduce v1.9 exactly; it adds a consistency check, not new evidence. Its range and selection results are new (blind library).

## Descriptive

| Drawn term | sigma AUROC at 0, 1, 4, 16 x | coverage at 16 x of 90% conformal: nominal-calibrated / 16 observations |
|---|---|---|
| P aerodynamic drag | 0.680, 0.816, 0.912, 0.930 | 2.0% / 96.5% |
| P pivot vibration | 0.680, 0.705, 0.639, 0.619 | 3.2% / 94.9% |
| P constant torque | 0.680, 0.622, 0.438, 0.271 | 0.0% / 94.3% |
| B forcing sin 3x | 0.943, 0.885, 0.681, 0.620 | 0.0% / 96.9% |
| B cubic damping | 0.943, 0.940, 0.760, 0.717 | 0.1% / 95.4% |
| B quadratic damping | 0.943, 0.952, 0.813, 0.727 | 0.0% / 96.8% |

## Summary

With dictionaries fixed by published default rules and missing terms drawn by a seed nobody chose, the limit held on all six drawn terms and the linear-response mechanism held at 4 x on all six. At 16 x it failed only for the constant torque, as on v1.9. Most drawn terms turned out to lie almost entirely inside the generic libraries when measured by their response (only forcing sin 3x, 0.64, and pivot vibration, 0.16, were substantially outside). Selection from 16 noisy observations always beat ensemble sigma, but it found a term with statistic at least 0.90 in at least 18 of 20 seeds for only three of the six drawn terms.

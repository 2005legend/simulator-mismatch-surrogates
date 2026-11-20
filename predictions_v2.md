# Frozen predictions v2 (`pred-v2`): PDE systems 3 and 4

Status: **frozen 2026-11-20** by the commit tagged `pred-v2`, after the author's approval of the draft as written. Computed 2026-11-20 from **pilot seed 0 only** (runner `scripts/run_pde.py` at commit `30f7ab2`, `--phase pilot`), before any seed 1-20 PDE run. Changes after this go in section 11 of `preregistration.md`. Raw output: `runs/pde_pilot/pde_seed0.jsonl`; eps_ref: `pilot_results/pde_eps_ref.json` (copy of `runs/pde_pilot/eps_ref.json`, used by the main run); fits: `predictions/pred_v2_seed0.json` (`mismatch/model_a.fit_4pl`, the function used for `pred-v1`).

## 1. eps_ref (r = 1 at n_nom = 512, bisection on seed 0)

| System | P1 (-u\|u\|) | P2 (sin 3x) |
|---|---|---|
| 3, Burgers | 0.003595 | 0.006059 |
| 4, operator | 0.005797 | 0.006767 |

Measured r at 1x: 1.00 in all four cases.

## 2. Plug-in rule (addendum v1.7, section 3)

The nominal surrogate is never retrained at eps > 0, so Model A's empirical plug-in at each level equals seed 0's measured AUROC (as for `pred-v1`). The prediction is the 4PL fit of seed 0's nominal-ensemble-sigma AUROC on log r over the seven nonzero levels. As for `pred-v1`, this tests whether the transition is a stable property of the setup across seeds.

## 3. PA2 primary: nominal ensemble sigma

| System | term | predicted r_1/2 | 4PL slope | plateau small r / **large r** | AUROC at eps = 0 | fit |
|---|---|---|---|---|---|---|
| 3 | P1 | **3.70** | 4.74 | 0.947 / **0.743** | 0.944 | converged |
| 3 | P2 | **2.70** | 1.94 | 0.954 / **0.584** | 0.944 | converged |
| 4 | P1 | **0.50** | 30.5 (step) | 0.755 / **0.645** | 0.797 | converged, midpoint poorly identified (see 5) |
| 4 | P2 | **1.03** | 3.03 | 0.786 / **0.508** | 0.797 | converged |

Measured AUROC (event \|\|e\|\| above its 80th percentile) by eps multiple:

| System | term | quantity | 0.25 | 0.5 | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|---|---|---|---|
| 3 | P1 | r | 0.25 | 0.50 | 1.00 | 1.99 | 3.96 | 7.81 | 15.18 |
| 3 | P1 | AUROC | 0.941 | 0.946 | 0.946 | 0.944 | 0.826 | 0.763 | 0.730 |
| 3 | P2 | r | 0.25 | 0.50 | 1.00 | 2.00 | 4.00 | 7.99 | 15.95 |
| 3 | P2 | AUROC | 0.948 | 0.941 | 0.910 | 0.822 | 0.695 | 0.636 | 0.588 |
| 4 | P1 | r | 0.25 | 0.50 | 1.00 | 1.99 | 3.93 | 7.67 | 14.70 |
| 4 | P1 | AUROC | 0.755 | 0.700 | 0.641 | 0.610 | 0.632 | 0.650 | 0.690 |
| 4 | P2 | r | 0.25 | 0.50 | 1.00 | 2.00 | 4.00 | 7.97 | 15.94 |
| 4 | P2 | AUROC | 0.785 | 0.753 | 0.656 | 0.533 | 0.524 | 0.509 | 0.500 |

**Predicted large-r AUROC** (the 4PL large-r plateau): 0.74, 0.58, 0.64, 0.51 in the order of the table above.

## 4. Secondary signals (reported, not tested under PA2)

| System | term | signal | r_1/2 | large-r plateau | rho_S(signal, D) at 16x |
|---|---|---|---|---|---|
| 3 | P1 | MC-dropout sigma | 6.66 | 0.814 | +0.23 |
| 3 | P1 | PDE residual | fit fails (AUROC does not fall: 0.85 to 0.92 to 0.85) | n/a | +0.42 |
| 3 | P2 | MC-dropout sigma | 2.91 | 0.574 | -0.04 |
| 3 | P2 | PDE residual | 2.38 | 0.371 | -0.26 |
| 4 | P1 | PDE residual | 0.50 | 0.646 | +0.12 |
| 4 | P2 | PDE residual | 1.05 | 0.485 | -0.21 |

At 16x, rho_S(ens sigma, E) and rho_S(ens sigma, D) were 0.19 / 0.14 (3, P1), 0.12 / 0.09 (3, P2), 0.12 / 0.14 (4, P1) and -0.18 / -0.18 (4, P2): the limit that PD1 tests.

## 5. Readings stated before any main PDE run

- **Family P (PB1, slope < 0).** Seed 0's OLS slopes of ensemble-sigma AUROC on log r are -0.058, -0.098, -0.016 and -0.076. System 4 P1 is the weakest: the curve falls to 0.61 at 2x and then recovers to 0.69 at 16x, so PB1 there may not reach significance.
- **System 4, P1.** The AUROC drop happens below the smallest measured level (0.80 at eps = 0, already 0.755 at 0.25x), and the curve is not monotone. The 4PL returns a near-step (slope 30) with its midpoint between the first two levels. The rule's "converged" condition holds, but the midpoint is poorly pinned down. PA2 is likely to be fragile there (a main-seed fit could land anywhere from about 0.25 to 1, or fail and be scored inconclusive under the frozen rule). It is scored by the frozen rule without change.
- **System 3, P1, PDE residual.** It is aligned with the missing term (rho_S with D +0.42), so its AUROC does not fall. This is the case the theory predicts for an aligned signal, as in the dev look.
- **System 3, P2, PDE residual.** It is anti-aligned (rho_S with D -0.26), and its large-r AUROC falls below 0.5 (0.38 at 16x), as in the dev look.

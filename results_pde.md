# PDE results (addendum v1.7): Systems 3 (Burgers, MLP) and 4 (FNO operator)

Seeds 1-20, 10,000 bootstrap resamples. Rules: `prereg_addendum_v1_7.md` section 5, `pred-v2`. Generated from `runs/pde_analysis/results.json` (`scripts/run_pde_analysis.py`). All PDE tests are secondary to the pendulum families.

## Family P: PB1, slope of ensemble-sigma AUROC on log r < 0 (Holm over 4)

| combination | mean slope | 95% BCa CI | p | Holm p | seeds with slope > 0 | verdict |
|---|---|---|---|---|---|---|
| B_P1 | -0.0616 | [-0.0659, -0.0571] | 1.8e-16 | 5.4e-16 | 0/20 | **supported** |
| B_P2 | -0.0904 | [-0.0938, -0.0870] | 1.1e-21 | 4.5e-21 | 0/20 | **supported** |
| O_P1 | -0.0090 | [-0.0127, -0.0046] | 0.00037 | 0.00037 | 3/20 | **supported** |
| O_P2 | -0.0540 | [-0.0594, -0.0490] | 4.9e-14 | 9.8e-14 | 0/20 | **supported** |

## PA2: pred-v2 r_1/2 vs seeds 1-20 (ensemble sigma)

| combination | observed r_1/2 | 90% CI | predicted (seed 0) | inside CI | within 1.5x | resamples dropped | verdict | large-r AUROC obs / pred |
|---|---|---|---|---|---|---|---|---|
| B_P1 | 3.642 | [3.507, 3.783] | 3.703 | True | True | 0 | **success** | 0.729 / 0.743 |
| B_P2 | 2.185 | [2.104, 2.267] | 2.697 | False | True | 0 | **fail** | 0.617 / 0.584 |
| O_P1 | 0.525 | [0.518, 8.013] | 0.502 | False | True | 385 | **fail** | 0.627 / 0.645 |
| O_P2 | 1.272 | [1.179, 1.376] | 1.027 | False | True | 0 | **fail** | 0.551 / 0.508 |

Secondary signals (descriptive, seed-mean 4PL r_1/2 vs pred-v2):

- B_P1_mc_sigma: observed 4.490 (converged True), predicted 6.655
- B_P1_pde_resid: observed 0.543 (converged True), predicted fit failed
- B_P2_mc_sigma: observed 2.262 (converged True), predicted 2.914
- B_P2_pde_resid: observed 1.899 (converged True), predicted 2.379
- O_P1_pde_resid: observed 0.528 (converged True), predicted 0.505
- O_P2_pde_resid: observed 1.249 (converged True), predicted 1.053

## PD1: rho_S(sigma, E) - rho_S(sigma, D) at 16x within +/- 0.10 (TOST)

| combination | mean difference | TOST p | equivalent | mean rho(sigma,E) / rho(sigma,D) |
|---|---|---|---|---|
| B_P1 | 0.022 | 5e-21 | **True** | 0.18 / 0.16 |
| B_P2 | 0.016 | 2.2e-28 | **True** | 0.11 / 0.09 |
| O_P1 | -0.016 | 1.3e-14 | **True** | 0.06 / 0.08 |
| O_P2 | 0.003 | 1.6e-36 | **True** | -0.12 / -0.12 |

## Alignment: PR1, PR1s, PR3a, PR3b (need >= 18 of 20)

| combination | PR1 sigma / resid | PR1s 4x, 16x sigma (max gap) | PR1s 4x, 16x resid (max gap) | PR3a beats sigma / >= 0.90 | PR3b beats sigma | R3a selected (n = 16) | R3b selected | rho(sigma, D) at 16x |
|---|---|---|---|---|---|---|---|---|
| B_P1 | 20 / 20 | 20, 20 (0.046) | 20, 20 (0.070) | 20 / 20 | 20 | u|u| 20 | u^2 16, u sin(x) 4 | 0.16 |
| B_P2 | 20 / 20 | 20, 20 (0.003) | 20, 20 (0.002) | 20 / 20 | 20 | sin(2x) 3, sin(3x) 16, cos(3x) 1 | sin(2x) 20 | 0.09 |
| O_P1 | 20 / 20 | 20, 20 (0.035) | 20, 20 (0.028) | 20 / 20 | 20 | u|u| 20 | u 8, u^2 11, u sin(x) 1 | 0.08 |
| O_P2 | 20 / 20 | 20, 20 (0.003) | 20, 20 (0.003) | 20 / 20 | 20 | sin(2x) 7, sin(3x) 4, cos(3x) 9 | sin(x) 1, sin(2x) 19 | -0.12 |

Caveat stated before the runs (v1.7 section 5): on P2 the sin(kx) responses are nearly collinear, so PR3a/PR3b there are close to automatic; R3a on P2 often selects sin(2x), a term outside the true family, and still passes.

## Descriptive: D3 (aligned of 20 at 16x) and coverage at 16x

| combination | D3 n = 8 / 16 / 32 / 64 | nominal conformal coverage | true-data conformal (n = 16) coverage |
|---|---|---|---|
| B_P1 | 0 / 2 / 1 / 4 | 0.000 | 0.968 |
| B_P2 | 1 / 0 / 0 / 1 | 0.000 | 0.969 |
| O_P1 | 1 / 1 / 1 / 2 | 0.000 | 0.942 |
| O_P2 | 0 / 0 / 0 / 1 | 0.000 | 0.946 |

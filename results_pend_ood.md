# Out-of-dictionary results on the pendulum (addendum v1.9): terms Q1 and Q2

Seeds 1-20 (pilot seed 0 set eps_ref only: Q1 0.005310, Q2 0.004706, `pilot_results/pend_ood_eps_ref.json`). Rules: `prereg_addendum_v1_9.md` section 4, frozen by tag `prereg-v1.9` before any registered run. Generated from `runs/pend_ood_analysis/results.json` (`scripts/run_pendulum_ood.py --stage analyse`). All tests are secondary (family Q). The residual signal is the ODE residual (stored under the runner key `pde_resid`).

- **Q1:** true system adds eps * g (constant torque). Median out-of-span share of its response: 0.73.
- **Q2:** true system adds eps * g sin(1.5 t) (periodic drive). Median out-of-span share: 0.49.

## QB1: slope of ensemble-sigma AUROC on log r < 0 (Holm over 2)

| term | mean slope | 95% BCa CI | Holm p | seeds with slope > 0 | verdict |
|---|---|---|---|---|---|
| Q1 | -0.1085 | [-0.1125, -0.1047] | 6.5e-22 | 0/20 | **supported** |
| Q2 | -0.0807 | [-0.0842, -0.0760] | 2.2e-19 | 0/20 | **supported** |

## QD1: at 16 x, rho_S(sigma, E) - rho_S(sigma, D) within +/- 0.10 (TOST)

| term | mean difference | TOST p | verdict |
|---|---|---|---|
| Q1 | 0.0376 | 6.0e-16 | **equivalent** |
| Q2 | 0.0445 | 7.1e-16 | **equivalent** |

## QR1: rho_S(signal, D) at 16 x inside the full-dictionary alignment range (>= 18 of 20)

| term | signal | inside | mean width (of 2) | reference p | verdict |
|---|---|---|---|---|---|
| Q1 | ensemble sigma | 20/20 | 1.37 | 0.026 | **met** |
| Q1 | ODE residual | 20/20 | 1.32 | 0.015 | **met** |
| Q2 | ensemble sigma | 20/20 | 1.37 | 0.026 | **met** |
| Q2 | ODE residual | 20/20 | 1.32 | 0.015 | **met** |

## QR1L and QR1L4: abs(rho_S(signal, D) - rho_S(signal, ||L[h]||)) <= 0.10 (>= 18 of 20), at 16 x and at 4 x

| term | signal | QR1L (16 x): within 0.10, largest gap | mean rho(s, D) / rho(s, linear) at 16 x | QR1L4 (4 x): within 0.10, largest gap |
|---|---|---|---|---|
| Q1 | ensemble sigma | 0/20, 0.151, **not met** | -0.456 / -0.318 | 20/20, 0.044, **met** |
| Q1 | ODE residual | 0/20, 0.164, **not met** | -0.343 / -0.190 | 20/20, 0.046, **met** |
| Q2 | ensemble sigma | 20/20, 0.088, **met** | -0.466 / -0.387 | 20/20, 0.027, **met** |
| Q2 | ODE residual | 20/20, 0.091, **met** | -0.341 / -0.256 | 20/20, 0.028, **met** |

Expectation frozen in v1.9: QR1L likely not met on Q1, borderline on Q2; QR1L4 met on both. All three held.

Mean gap by level (ensemble sigma):

| term | 0.25 | 0.5 | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|---|---|
| Q1 | 0.001 | 0.004 | 0.009 | 0.020 | 0.040 | 0.076 | 0.137 |
| Q2 | 0.000 | 0.002 | 0.005 | 0.012 | 0.024 | 0.046 | 0.079 |

Reading: the linear-response prediction holds at small eps and degrades as eps grows, roughly doubling with each doubling of eps; on Q1 it exceeds the 0.10 margin at 16 x in every seed. This is the small-eps condition of Proposition 2 failing in the nonlinear regime. On System 3 (v1.8) the largest gap at 16 x was 0.0062.

## QR3: term selected from 16 noisy trajectories at 4 x beats ensemble sigma (>= 18 of 20)

| term | beats sigma | selections (n = 16) | median statistic vs sigma | verdict |
|---|---|---|---|---|
| Q1 | 20/20 | tanh(w/0.1) 19, tanh(w/0.5) 1 | 0.836 vs -0.365 | **met** |
| Q2 | 20/20 | tanh(w/0.1) 11, p 7, tanh(w/0.5) 1, sin(2p) 1 | 0.616 vs -0.423 | **met** |

As stated in v1.9, this test is weak here: ensemble sigma is anti-aligned with the discrepancy, so any positively aligned term beats it.

## Descriptive

Mean ensemble-sigma AUROC and geometric-mean r by level (x eps_ref):

| term | | 0 | 0.25 | 0.5 | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|---|---|---|---|
| Q1 | AUROC | 0.680 | 0.678 | 0.666 | 0.622 | 0.541 | 0.438 | 0.346 | 0.271 |
| Q1 | r | - | 0.26 | 0.52 | 1.03 | 2.04 | 3.99 | 7.66 | 14.42 |
| Q2 | AUROC | 0.680 | 0.681 | 0.677 | 0.647 | 0.586 | 0.505 | 0.432 | 0.380 |
| Q2 | r | - | 0.26 | 0.51 | 1.02 | 2.03 | 3.99 | 7.77 | 15.04 |

Q1: coverage at 16 x of 90% conformal intervals: nominal-calibrated 0.0%, calibrated on 16 noisy trajectories 94.3%. D3 aligned seeds at n = 8, 16, 32, 64: 0, 0, 0, 0 (sigma is anti-aligned, so 0 is expected).

Q2: coverage at 16 x of 90% conformal intervals: nominal-calibrated 0.6%, calibrated on 16 noisy trajectories 94.5%. D3 aligned seeds at n = 8, 16, 32, 64: 0, 0, 0, 0 (sigma is anti-aligned, so 0 is expected).

## Summary

On a second system with different physics, the limit (QB1, QD1) and bracketing (QR1) hold for both out-of-dictionary terms, as on System 3. The linear-response mechanism holds at 4 x on both terms but fails at 16 x on the constant torque in every seed, with the gap growing smoothly with eps: the mechanism is a small-eps result, and the pendulum leaves that regime within the studied range while Burgers did not. Selection beats sigma, but only because sigma is anti-aligned.

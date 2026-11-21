# Out-of-dictionary results (addendum v1.8): System 3, missing terms P3 and P4

Seeds 1-20 (pilot seed 0 set eps_ref only: P3 0.002216, P4 0.002062, `pilot_results/pde_ood_eps_ref.json`). Rules: `prereg_addendum_v1_8.md` section 4, frozen by tag `prereg-v1.8` before any registered run. Generated from `runs/pde_ood_analysis/results.json` (`scripts/run_pde_ood.py --stage analyse`). All tests are secondary (family O).

- **P3:** true system adds eps * (-u_x) (missing uniform drift). Median out-of-span share of its response: 0.36.
- **P4:** true system adds eps * u_xx (missing extra diffusion). Median out-of-span share: 0.22.

## OB1: slope of ensemble-sigma AUROC on log r < 0 (Holm over 2)

| term | mean slope | 95% BCa CI | Holm p | seeds with slope > 0 | verdict |
|---|---|---|---|---|---|
| P3 | -0.0401 | [-0.0427, -0.0377] | 3.1e-17 | 0/20 | **supported** |
| P4 | -0.0334 | [-0.0365, -0.0309] | 2.8e-15 | 0/20 | **supported** |

## OD1: at 16 x, rho_S(sigma, E) - rho_S(sigma, D) within +/- 0.10 (TOST)

| term | mean difference | TOST p | verdict |
|---|---|---|---|
| P3 | 0.0245 | 1.1e-17 | **equivalent** |
| P4 | 0.0052 | 3.9e-22 | **equivalent** |

## OR1: rho_S(signal, D) at 16 x inside the full-dictionary alignment range (>= 18 of 20)

The range depends on the signal and the dictionary, not on the missing term, so its width is the same for P3 and P4. Reference probability: chance of 18 or more of 20 inside if the value were placed uniformly on [-1, 1] (stated in v1.8 as about 0.001 for sigma and 0.02 for the residual; the measured widths came out a little narrower than in the dev look).

| term | signal | inside | mean width (of 2) | reference p | verdict |
|---|---|---|---|---|---|
| P3 | ensemble sigma | 20/20 | 1.01 | 0.0002 | **met** |
| P3 | PDE residual | 20/20 | 1.30 | 0.013 | **met** |
| P4 | ensemble sigma | 20/20 | 1.01 | 0.0002 | **met** |
| P4 | PDE residual | 20/20 | 1.30 | 0.013 | **met** |

Reading: the range is wide, so containment is weak evidence on its own; the counts are reported next to the widths and reference probabilities as promised in v1.8.

## OR1L: |rho_S(signal, D) - rho_S(signal, ||L[h]||)| <= 0.10 at 16 x (>= 18 of 20)

L[h] is the finite-difference linear response of the nominal output to the true term (step 1e-3).

| term | signal | within 0.10 | largest gap | mean rho(signal, D) | mean rho(signal, linear) | verdict |
|---|---|---|---|---|---|---|
| P3 | ensemble sigma | 20/20 | 0.0004 | 0.325 | 0.325 | **met** |
| P3 | PDE residual | 20/20 | 0.0003 | 0.210 | 0.210 | **met** |
| P4 | ensemble sigma | 20/20 | 0.0062 | 0.508 | 0.504 | **met** |
| P4 | PDE residual | 20/20 | 0.0040 | 0.331 | 0.329 | **met** |

Reading: the mechanism of Proposition 2 (alignment at large r is set by the missing term's linear response) holds when the term is outside the dictionary; the margin was 0.10 and the largest gap is 0.006.

## OR3: term selected from 16 real fields at 4 x beats ensemble sigma (>= 18 of 20)

| term | beats sigma | selections (n = 16) | median statistic vs sigma | verdict |
|---|---|---|---|---|
| P3 | 19/20 | u^2 11, u\|u\| 6, u^3 1, sin(3x) 1, u sin(x) 1 | 0.590 vs 0.329 | **met** |
| P4 | 9/20 | sin(3x) 17, cos(3x) 3 | 0.503 vs 0.505 | **not met** (expected in v1.8) |

Beats-sigma count by number of real fields (descriptive):

| term | n = 8 | n = 16 | n = 32 | n = 64 |
|---|---|---|---|---|
| P3 | 20 | 19 | 20 | 20 |
| P4 | 1 | 9 | 9 | 9 |

Reading: on P4 more data does not help; the best dictionary term is about as aligned as sigma itself, so the limit is the dictionary, not the sample size.

## Descriptive

Mean ensemble-sigma AUROC by level (x eps_ref):

| term | 0 | 0.25 | 0.5 | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|---|---|---|
| P3 | 0.943 | 0.945 | 0.946 | 0.950 | 0.941 | 0.868 | 0.818 | 0.798 |
| P4 | 0.943 | 0.943 | 0.941 | 0.936 | 0.908 | 0.863 | 0.834 | 0.824 |

D3 at 16 x (seeds aligned, of 20) by number of real points: P3 2, 2, 9, 19 and P4 8, 12, 18, 20 at n = 8, 16, 32, 64.

Coverage at 16 x: conformal intervals calibrated on nominal simulations cover 0.0% of true outputs (both terms); intervals calibrated on real data cover 96.8% (P3) and 97.4% (P4), nominal 95%.

## Summary

Out of the dictionary, the limit (OB1, OD1) and the linear-response mechanism (OR1L) hold on both terms in every seed. Bracketing (OR1) also holds in every seed, but the range is wide. Choosing a term from a few real fields (OR3) beats sigma on P3 but not on P4, where no dictionary term resembles extra diffusion closely enough; this was predicted before the run.

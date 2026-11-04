# Blind to the missing term: experiment design (v1)

Working title: *Blind to the Missing Term: How Structural Simulator Mismatch Breaks Uncertainty, Data Acquisition and Failure Search for Physics Surrogates.*

Status: design v1, 2026-11-01. It is based on a full-text novelty check of about 30 papers. Notes:
- Data acquisition: {{artifact:56506713-39fb-4e14-a054-e7c1c771dfdc}}
- Uncertainty: {{artifact:5763c665-8d2d-48ed-a8f3-50bc2c907003}}
- Failure search: {{artifact:1cf23b35-ae60-4992-a1e1-5e8b524d6453}}

Earlier landscape report: {{artifact:64a5e73d-48b4-44e4-8a1a-f2c643cd2d8a}}.

## 1. Novelty position (honest)

Each of the three parts exists on its own. Nobody has published the combination.

| Part | Closest prior work (verified, full text) | What it already does | Our delta |
|---|---|---|---|
| A. Data acquisition under mismatch | Zhu & Hainy, arXiv:2603.16756 (Sept 2026) | GP with Kennedy-O'Hagan discrepancy picks physical experiments vs space-filling, including a Duffing oscillator whose simulator lacks the cubic term (one fixed mismatch strength) | Neural surrogates; residual and ensemble scores; a sweep over the mismatch strength ε to locate the crossover ε* |
| A (secondary) | Polanska et al., arXiv:2605.21348 | PDE-residual acquisition, always against the exact PDE | Residual computed against a wrong (nominal) equation |
| B. Uncertainty under mismatch | Gopakumar et al., arXiv:2408.09881; Mouli et al., arXiv:2403.10642 | Conformal recovers coverage only when calibrated in the shifted regime; ensemble spread tracks error under parameter extrapolation | Controlled structural term with strength ε; calibration source (simulator vs n true points) as a factor; ranking quality (AUROC, Spearman), not just coverage |
| C. Failure search | Roy et al., arXiv:2610.07529 (Oct 2026) | Targeted search finds 1.06-3.10x larger neural-operator errors than random, over hardware tolerances | Physical parameters with an operating distribution; failure-probability estimates with CIs at matched true-system budgets; mismatch-driven errors; checks the surrogate's own uncertainty |

**Claim to make:** one controlled mismatch dial ε across all three tools, with one error decomposition explaining when and why each tool goes blind. Also report how many true-system observations restore each tool. Do NOT claim that ensemble uncertainty tracks error under parameter extrapolation; Mouli et al. showed that.

## 2. Setting (fixes the flaw the check found)

There are two information sources:
- **Nominal simulator:** cheap, unlimited, missing a term.
- **True system:** expensive. Each query is an "experiment" counted against a budget n. Here it is simulated as the nominal ODE plus the missing term at strength ε.

Without true-system queries, ε never enters training, and ε* would follow analytically from one run (flagged by the acquisition checker).

**Error decomposition (the central object):**

f_hat(θ) − f_true(θ) = [f_hat − f_nom] + [f_nom − f_true] = approximation error + structural discrepancy.

## 3. Systems

| ID | Nominal model | True system adds | Why |
|---|---|---|---|
| S1 | Damped pendulum: φ'' + c φ' + (g/L) sin φ = 0 | ε·φ'·abs(φ') (quadratic drag) | Smooth discrepancy that grows with velocity, so it is θ-dependent |
| S2 | Same as S1 | ε·g·tanh(φ'/δ) (smoothed Coulomb friction) | Nearly non-smooth discrepancy that matters most at low amplitude, the opposite region to S1 |
| S3 (optional) | Linear damped oscillator | ε·x³ (Duffing) | Replication anchor against Zhu & Hainy only; not a headline system |

- **Parameters θ:** (c, L, φ0, ω0) in a fixed box, with an operating distribution p(θ) for part C.
- **Mismatch levels:** ε ∈ {0, 0.25, 0.5, 1, 2}, each a multiple of a reference ε_ref. ε_ref is chosen in a pilot so that the median discrepancy norm equals the median approximation error of the nominal surrogate at n_nominal = 512. This puts the expected crossover inside the sweep.
- **Surrogate:** MLP deep ensemble (M=5, about 30k parameters) mapping θ to a trajectory on a fixed time grid, pretrained on 512 Sobol nominal simulations.
- **Correction model:** an additive discrepancy network δ(θ) (small MLP ensemble) fitted on true minus nominal-surrogate residuals at the queried true points. The GP / Kennedy-O'Hagan discrepancy model is a reference arm.

## 4. Research questions, arms and metrics

### RQ1: Does uncertainty see the missing term?

Uncertainty sources:
- nominal ensemble σ;
- MC dropout;
- split conformal calibrated on nominal simulations;
- split conformal calibrated on n ∈ {8, 16, 32, 64} true points;
- discrepancy-ensemble σ;
- GP-KOH predictive variance;
- nominal-ODE residual of the surrogate prediction (NeuroForge-type check, arXiv:2607.10333).

Metrics, all reported against ε:
- 90% coverage;
- AUROC for the event error > τ;
- Spearman(score, error);
- AUROC computed separately against each error component (approximation vs discrepancy).

Parameter extrapolation is reported as a separate arm and cited to Mouli et al.

Stated up front: nominal σ(θ) does not depend on ε, so nominal-calibrated coverage must decay. The open question is whether σ *aligns* with where the discrepancy lives. S1 and S2 put the discrepancy in opposite regions of θ, so the alignment is testable rather than assumed.

### RQ2: Which experiments should we run on the true system?

Budget: n = 8 initial Sobol points, then 6 rounds of 8 (n_max = 56).

Acquisition arms:
1. Random.
2. Sobol / maximin.
3. Nominal ensemble σ (AL4PDE/SBAL-style; hypothesised blind to ε).
4. Nominal residual of the surrogate prediction (Polanska-style; hypothesised blind to ε).
5. Discrepancy-ensemble σ.
6. GP-KOH discrepancy variance (Zhu & Hainy-style reference).
7. Observed-residual model: regress the nominal-ODE residual measured on already-observed TRUE trajectories (this is the missing forcing term) and query where its predicted magnitude is high, i.e. a physics-guided score that can see the mismatch. This arm is the most novel in the design and needs a full-text check of arXiv:2610.27130 (reported to show residual-driven acquisition harming a surrogate, from a keyword scan only).

Metrics, on held-out true-system test sets (ID box, near and far parameter extrapolation): median, p95, p99 and max relative L2 error after correction, and budget-AUC over n.

**Crossover ε\*:** the smallest ε at which arms 3-4 fall below Sobol, with the CI excluding 0.

### RQ3: Can we find where the corrected surrogate still fails?

- **Failure event:** error(θ) > τ_ε. τ is fixed per ε and system so that P_fail lies in [1e-4, 1e-2] under p(θ).
- **Ground truth:** 1e6 true-system simulations per system and ε.
- **Methods at matched true-system budgets B ∈ {100, 300, 1000}:**
  - Monte Carlo with Clopper-Pearson CIs;
  - Sobol QMC;
  - cross-entropy method / improved CE;
  - adaptive importance sampling;
  - Bayesian Safety Validation (open-source; Moss et al., arXiv:2305.02449);
  - AK-MCS;
  - uncertainty-guided IS with proposal ∝ p(θ)·σ(θ)^k.
- **Metrics:** relative error of the P_fail estimate, CI coverage over repeats, fraction of failure mass located, and worst error found (reported as a lower bound only).

### Unifying hypothesis (the paper's main claim if it holds)

- **H\*:** tools calibrated only on the nominal simulator (nominal σ, nominal residual, nominal-calibrated conformal, σ-guided failure search) track the approximation component and not the discrepancy component. Their usefulness collapses near the ε where the discrepancy norm overtakes the approximation error.
- The ε at which RQ1's AUROC drops is predicted to match RQ2's crossover ε\*. That link is the most novel testable claim.

## 5. Statistics (same protocol as the accepted paper; only the method is new)

- 5 seeds per cell, paired by seed: shared initial sets, candidate pools and test sets.
- Percentile-bootstrap 95% CIs with 10,000 resamples.
- Primary contrasts, declared in advance and Holm-corrected:
  - RQ2: arm 3 vs Sobol, arm 4 vs Sobol, arm 7 vs Sobol and arm 7 vs arm 6, at each ε, on S1 and S2;
  - RQ1: nominal σ AUROC at ε=0 vs ε=2;
  - RQ3: CE and BSV vs MC relative error at B=300.
- Everything else is exploratory. No improvement claim when the CI includes 0. Acquisition hyperparameters (k, τ quantile, grid) are frozen on a pilot seed that is excluded from the analysis.

## 6. Pre-registered outcomes

| Outcome | Reading |
|---|---|
| Nominal-only arms lose to Sobol above ε\*, and RQ1 AUROC on the discrepancy component stays near 0.5 | H\* supported; headline result |
| RQ1 and RQ2 transition points coincide (CIs overlap) | Mechanism link supported; strongest version of the paper |
| Nominal σ still tracks discrepancy in S1 but not S2 | Alignment is system-dependent; report it as "uncertainty works by geometric coincidence" |
| Observed-residual arm 7 beats GP-KOH arm 6 | Physics-guided acquisition can be rescued by computing residuals on true data |
| Arm 7 no better than Sobol | Report it; consistent with arXiv:2610.27130 |
| CE/BSV estimates are no better than MC at B ≤ 300 | Failure search does not pay at these budgets; still a useful negative result |
| All CIs include 0 | Tight null with power analysis; suitable for evaluation or negative-results venues |

## 7. Compute (estimates)

Only the 0.0215 s per pendulum simulation was measured, by a researcher on CPU. MLP fit times (10-60 s per member) are assumptions to replace with a 10-minute timing pilot.

- **RQ2 grid:** 2 systems × 5 ε × 7 arms × 5 seeds × 6 rounds × 5 members = 10,500 discrepancy-net fits, plus 50 nominal pretraining fits. That is 29.3-175.8 CPU-hours.
- **RQ1:** reuses the RQ2 models; conformal and GP fits are negligible.
- **RQ3 ground truth:** 1e7 true simulations, about 59.7 CPU-hours, embarrassingly parallel. It can be cut by about 5× by using 2e5 simulations wherever P_fail ≥ 1e-3.
- **RQ3 method runs:** at most 25.1 CPU-hours of simulation.
- **Total:** roughly 115-260 CPU-hours. Across 8-12 parallel workers (RAM, about 5 GB free, is the limit, not cores), that is 1-3 days of wall time.
- **GPU:** Kaggle's T4 is optional (S3 or a 1D Burgers extension only).

## 8. Scope tiers and timeline (about 18 weeks)

| Tier | Contents | Target |
|---|---|---|
| MVP | S1 only; RQ1 and RQ2; 5 ε levels | ICLR 2027 workshop (deadline estimated late Jan to Feb 2027; unannounced) |
| Full | S1 and S2; RQ1-RQ3; S3 anchor | ICML 2027 workshops / ML4PS 2027 (dates unknown); arXiv by about March 2027 |

| Weeks | Work | Gate |
|---|---|---|
| 1 | Full-text read of arXiv:2610.27130 and the Jaber et al. SSRN 7272715 conformal model-form paper (both unverified); pre-registration committed | Reframe if either covers the RQ1-RQ2 link |
| 2-4 | Simulators with the ε switch, error-decomposition tooling, nominal pretraining, timing pilot, ε_ref calibration | ε_ref puts the expected crossover inside the sweep |
| 5-7 | RQ1 and RQ2 on S1 (MVP) | Go/no-go on H\* |
| 8-11 | S2, RQ3, S3 anchor | |
| 12-14 | Statistics, figures (key figure: AUROC by error component vs ε) | |
| 14-18 | Writing; clean repo with one-command reproduction | |

## 9. Caveats

- "Not found" in the novelty check means not found within budget. arXiv rate-limited the data-acquisition search (4 of 15 date-sorted queries ran; OpenAlex and web searches filled the gap). The full texts were read selectively (methods, experiments, appendix).
- Unverified: arXiv:2610.27130 (keyword scan only); Jaber et al. SSRN 7272715; older GP calibration papers (abstract or title only). See the notes for the full list.
- Watch monthly for new work from the groups behind Zhu & Hainy, Roy et al., Gopakumar et al., Polanska et al. and McShannon & Dietrich.

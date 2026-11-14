# Stage 1 notes (2026-11-13): literature, ranking bound, Burgers feasibility

Status: development notes. Everything numerical here uses dev seed 999 only, 3 ensemble members and rough ε_ref values. No pilot seed (0, 101, 102) or main seed (1-10) was used. Nothing here is a pre-registered result.

## 1. Literature check for the PDE / neural-operator version (first pass, abstracts only)

Four web searches. No paper found that sweeps mismatch strength for a neural PDE surrogate and measures uncertainty ranking or true-system acquisition against it. Closest, to read in full before any novelty claim:

| Paper | What it does (from abstract / search summary) | Overlap |
|---|---|---|
| arXiv:2606.03469, Physics-guided correction for operator learning under model misspecification | Approximate physics prior plus a learned correction operator | Correction method under misspecification; no r sweep seen. **Read in full.** |
| arXiv:2603.24948, Latent representation learning based model correction and UQ for PDEs | Latent GP for the discrepancy term with predictive bounds | Discrepancy UQ for PDEs. **Read in full.** |
| arXiv:2606.17513, Geometry-aware post-hoc UQ in operator learning | GP on the residual of a frozen neural operator ("model inadequacy") | Surrogate-vs-simulator error (our a), not simulator-vs-truth (our d) |
| arXiv:2502.05372, Active learning of model discrepancy with BED | Already read (Yang/Chen/Wu) | Known neighbour |
| arXiv:2408.01536 (AL4PDE), 2503.03178, 2606.09949 | Active learning for neural PDE surrogates | Simulator assumed correct |
| arXiv:2610.09255, The Symbol of the Surrogate | Surrogates can learn the solver's numerical errors | Motivation: agreement with the solver is not agreement with physics |
| arXiv:2510.22063 (Jain & Bates), 2210.16938, DEUP 2102.08501 | Ensembles capture procedural variance, not bias | States the mechanism; no ranking bound in r found |
| Mikkola et al. 2023, rMFBO | "No harm" bound vs single-fidelity BO with unreliable sources (GP, optimisation) | Nearest neighbour of an arm-8 guarantee; full text not yet checked |

Not yet done: a search of COLT/NeurIPS theory for a ranking bound under misspecification; a full-text check of rMFBO. The first two rows were read in full on 2026-11-13 (section 6).

## 2. The ranking law (derivation)

Per test point x: a = f̂_nom − f_nom, d = f_nom − f_true, e = a + d. Norms A = ‖a‖, D = ‖d‖, E = ‖e‖ (same normaliser). s(x) is any signal.

**T1, limits and sandwich (no assumptions).** The triangle inequality gives |E − D| ≤ A and |E − A| ≤ D pointwise. Hence, with κ = rms(A)/sd(D) and κ′ = rms(D)/sd(A):

- |ρ(s,E) − ρ(s,D)| ≤ 2κ/(1−κ) for κ < 1 (large-r side);
- |ρ(s,E) − ρ(s,A)| ≤ 2κ′/(1−κ′) for κ′ < 1 (small-r side);
- rank version: |τ(s,E) − τ(s,D)| ≤ 2·P(|D_i − D_j| ≤ A_i + A_j), and the same with A and D swapped (a pair can only change order between E and D if its D gap is smaller than A_i + A_j).

Proof of the Pearson line: write E = D + R with |R| ≤ A, so sd(R) ≤ rms(A) and |sd(E) − sd(D)| ≤ rms(A). Then ρ(s,E) − ρ(s,D) = ρ(s,D)(sd(D)/sd(E) − 1) + ρ(s,R)·sd(R)/sd(E), and both terms are at most rms(A)/sd(E) ≤ rms(A)/(sd(D) − rms(A)).

Consequence: as r grows, the ranking ability of any signal converges to its correlation with the discrepancy norm, ρ(s,D). Since κ = (1/r) × (rms A / median A) × (median D / sd D), the transition is set by r **times the spread of D**, not by r alone: a discrepancy of nearly constant size changes the ranking late.

**T2, non-identifiability (to write up formally).** A simulator-only signal s is a function of the nominal simulator and the surrogate only. Two true systems with the same nominal simulator and the same r give the same s but can have any ρ(s,D), including negative. So no simulator-only signal can have a ranking guarantee at large r, and no amount of simulator data fixes it. S1 and S2 are the empirical witness (section 3).

**T3, exact identity for squared error.** E² = A² + D² + 2⟨a,d⟩, so cov(s,E²) = cov(s,A²) + cov(s,D²) + 2cov(s,⟨a,d⟩): the correlation with squared error is an exact weighted mix of three correlations. With d ≈ ε·g the weights scale as 1, ε², ε.

**T4, what real data buys (rough).** The large-r limit ρ(s,D) can be estimated from n real points. By the Fisher z approximation, telling ρ = 0 from ρ = 0.5 at one-sided α = 0.05 and 90% power needs about n ≈ ((1.645 + 1.28)/atanh 0.5)² + 3 ≈ 31 real experiments. A back-of-envelope figure, not a result.

## 3. Numerical check on the pendulum (dev seed 999, n_nom = 512, 500 test points, s = nominal ensemble σ)

`python -m scripts.dev_bound_check`

| System | ε/ε_ref | r | ρ(s,E) | ρ(s,A) | ρ(s,D) | AUROC(s,E) |
|---|---|---|---|---|---|---|
| S1 | 0.125 | 0.15 | 0.40 | 0.40 | 0.78 | 0.69 |
| S1 | 1 | 1.13 | 0.70 | 0.42 | 0.79 | 0.88 |
| S1 | 4 | 3.84 | 0.79 | 0.47 | 0.79 | 0.93 |
| S1 | 32 | 13.6 | 0.72 | 0.58 | 0.73 | 0.92 |
| S2 | 0.125 | 0.14 | 0.38 | 0.39 | 0.00 | 0.68 |
| S2 | 1 | 1.12 | 0.26 | 0.36 | −0.02 | 0.58 |
| S2 | 4 | 4.33 | 0.01 | 0.30 | −0.05 | 0.39 |
| S2 | 32 | 18.4 | −0.07 | 0.25 | −0.06 | 0.40 |

What this shows:

1. **The limits hold.** ρ(s,E) starts at ρ(s,A) and ends at ρ(s,D) on both systems.
2. **The worst-case rates are loose.** The Pearson bound is infinite for r below about 2-4 and above 1 for most of the range; the Kendall bound drops below 1 only for r above about 4. T1 gives the limits, not a usable rate. A sharper rate needs a model (the Model A copula, or T3).
3. **On S1 nominal σ does not go blind: it gets better** (AUROC 0.69 → 0.92), because σ and quadratic drag both grow with amplitude, ρ(s,D) ≈ 0.78. **On S2 it falls below chance** (AUROC ≈ 0.35-0.40), ρ(s,D) ≈ 0.
4. So the accurate statement is not "simulator-only signals go blind as r grows". It is: **their ranking converges to an alignment ρ(s,D) that the simulator cannot reveal**, which can be excellent or worse than chance.

Consequence for the frozen pre-registration: H1 / test B1 (AUROC slope on log r below 0) will probably be supported on S2 and **fail on S1**. The frozen tests are not changed. This was seen on a dev seed before the pilot, which must be disclosed in the paper; any new prediction written now (for example "the large-r AUROC equals the value implied by ρ(s,D)") goes in a new dated addendum and is labelled as written after this dev-seed look.

## 4. Burgers feasibility (dev seed 999)

`python -m scripts.dev_burgers_pilot`. Nominal: u_t + u u_x = ν u_xx, periodic, pseudo-spectral on 256 points, output u(x, T = 2) on 64 points, θ = (ν, A1, A2, phase). Missing terms: P1 = −ε u|u| (state-dependent damping), P2 = +ε sin 3x (fixed forcing).

- Solve: 0.13 s serial; 812 solves in 62 s on 6 workers.
- Fit: 12.4 s per member (same MLP; all three hit the 1,500-epoch cap). Median relative approximation error 0.5%.
- r is dialled by ε on both terms (P1: r 2.7 → 118 for ε 0.01 → 1; P2: r 0.95 → 99 for ε 0.005 → 0.5). The ε grid should start lower to cover r < 1.
- Same law: ρ(s,A) = 0.88; ρ(s,E) falls from 0.80-0.87 to 0.16-0.32, matching ρ(s,D) (0.16-0.36). AUROC falls from 0.93-0.95 to about 0.76, not to chance, because ρ(s,D) is positive here.
- The transition is slower in r than on the pendulum (ρ(s,E) is halfway at r ≈ 10), consistent with T1's "r times spread of D" reading; not yet checked quantitatively.

Feasible on the laptop. An MLP on a 4-parameter family is not a neural operator; a function-to-function setup (FNO or DeepONet on random initial conditions) is a later step and will need a GPU for 2D.

## 5. What stage 1 changes

- Headline candidate: "ranking under mismatch converges to an alignment the simulator cannot reveal; r and the spread of D set when; about n real points tell you which case you are in."
- Theory section: T1 (limits), T2 (non-identifiability), T3 (exact squared-error identity), Model A for the curve in between, T4 for the real-data budget.
- Open: is T1 + T2 "obvious"? T1 is elementary. The non-obvious content is the S1/S2 contrast, the spread-of-D dependence and a prediction of the whole curve.

## 6. Full-text reads of the two closest PDE papers (2026-11-13)

Read through a web-fetch pass of the arXiv HTML. For 2603.24948 the body of Section 5 (conclusion) was only seen as a summary, not verbatim.

| | arXiv:2606.03469v2 (Ma, Boullé, Yang, Wu, Guo; 1 Aug 2026) | arXiv:2603.24948v1 (Zhou, Feng, Guo, Wu; 26 Mar 2026) |
|---|---|---|
| Method | Approximate physics prior plus a DeepONet correction operator, trained jointly with a DeepONet solution operator | Latent-variable model with a shared encoder, a neural-operator solution decoder and a neural-network correction decoder |
| Misspecification | Diffusion-reaction (wrong reaction term), Burgers (extra cubic term; advection omitted; diffusion omitted), cavity flow (Newtonian prior for a power-law fluid), hyperelasticity (linear prior) | ODE and reaction-diffusion (wrong reaction term), channel and cavity flow (Newtonian model for a power-law fluid) |
| Strength varied over levels? | No. One setting per case; sweeps are over data amounts and noise only | No. One fixed misspecification per case |
| Approximation vs discrepancy split, or a ratio like r? | No | No |
| UQ | Deep ensembles (5-10 members), shown as bands; no ranking or calibration metric | Monte Carlo over latent samples, compared with ensemble PINNs and B-PINNs; coverage judged by eye, no ranking or calibration metric |
| Choice of true-system samples | None; fixed or random | None; random or uniform |
| Theorems or bounds | None | None |
| Stated future work | Time-dependent multiphysics, inverse problems, real experimental data | Correction is "weakly identifiable"; expressiveness and training cost of the correction network (from summary) |

Reading: neither overlaps with the study. Both are correction methods at a single mismatch level, and both present ensemble or latent-sample uncertainty as informative without measuring whether it ranks the true error. That unmeasured step is what this project measures. Useful for the PDE stage: their test cases (Burgers with a term omitted, power-law cavity flow with a Newtonian prior) are ready-made mismatch pairs that reviewers in this area already accept, and a strength dial can be added to each (for example the power-law index n moving away from 1).

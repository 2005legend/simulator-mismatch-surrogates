# Pre-registration addendum v1.7: Blind to the Missing Term (PDE systems)

Drafted 2026-11-20. Status: **DRAFT, not frozen.** It will be frozen by the git commit tagged `prereg-v1.7` after the author's review, and before any PDE run on a pilot or main seed. It adds two systems; nothing in `prereg-v1` to `prereg-v1.6` or `pred-v1` changes.

**Scope.** (1) Disclosure of the PDE development looks (dev seed 999 only). (2) Systems 3 and 4: definitions and settings. (3) A pilot on seed 0 that sets eps_ref and freezes predictions `pred-v2` before any main PDE run. (4) The RQ1-type sweep, signals and metrics on the PDE systems. (5) Tests and predictions, all secondary to the pendulum's frozen families, with one new Holm family. No acquisition on the PDE systems.

## 1. Disclosure: PDE development looks (dev seed 999 only)

No pilot seed (0, 101, 102) or main seed (1-20) has been used on Burgers or the operator. ε values below are rough guesses, not pilot values.

| Look | What was seen |
|---|---|
| Solver check (`mismatch/burgers.py`, `tests/test_burgers.py`) | The dev pilot of 2026-11-13 used nu down to 0.03 on 256 points, where shocks (width about 0.01) are not resolved. The box is now nu in [0.10, 0.30]; 256 and 512 points agree to 1e-7. 64 solves take 2 s. |
| System 3 feasibility (`scripts/dev_burgers_feasibility.py`, MLP ensemble of 5, frozen pendulum settings, n_nom = 512) | Median approximation error 0.25%; fit 122 s. P1: eps 0.003 to 1 gives r 0.9 to 156; rho_S(sigma, E) falls from +0.84 to +0.16 (its limit rho_S(sigma, D) = +0.18); AUROC 0.96 to 0.73. P2: eps 0.001 to 0.3 gives r 0.18 to 53; rho_S(sigma, E) +0.85 to +0.07 (limit +0.08); AUROC 0.95 to 0.61. The PDE residual is aligned on P1 (+0.46 to D) and anti-aligned on P2 (-0.30; AUROC 0.35 at large r). |
| Alignment check (`scripts/dev_burgers_alignment.py`, rough eps_ref P1 0.0034, P2 0.0056; 1, 4, 16 x) | Range of rho_S(sigma, term response) over the nine-term dictionary [-0.41, +0.57]; actual +0.18 (P1) and +0.08 (P2) inside it; gap to the mean true-family prediction 0.00 to 0.02 (sigma) and up to 0.06 (residual). R3a: selected term statistic 0.95 to 1.00 at every n; R3b (true family removed): 0.64 to 0.75 on P1 (u^2 or u sin x), 0.98 on P2 (sin 2x), against sigma 0.18 and 0.08. |
| Operator gate G3 (`scripts/dev_operator_gate.py`, FNO ensemble of 5, n = 512 profiles) | **Pass**: fit 159 to 217 s per member (limit 300), median relative error 0.31% (limit 5%). P1: eps 0.01 to 1 gives r 2.4 to 125; rho_S(sigma, E) +0.16 to +0.10 (limit +0.12); AUROC 0.69 to 0.66. P2: eps 0.003 to 0.3 gives r 0.6 to 62; rho_S(sigma, E) +0.59 to -0.18 (limit -0.18); AUROC 0.81 to 0.47. |

Reading: the pendulum's law holds on both PDE systems in this look. The ranking converges to the signal's alignment with the discrepancy, which the simulator alone cannot reveal. These are dev-informed, so the predictions below are labelled as such.

## 2. Systems

**System 3: parametric Burgers** (`mismatch/burgers.py`). Nominal u_t + u u_x = nu u_xx, periodic on [0, 2 pi); true system adds eps h with **P1** h = -u|u| (state-dependent damping) or **P2** h = sin 3x (fixed forcing). theta = (nu, A1, A2, phase) in [0.10, 0.30] x [0.5, 1.5] x [0, 1] x [0, pi], u0 = A1 sin x + A2 sin(2x + phase). Pseudo-spectral, 256 points, 2/3 dealiasing, integrating-factor RK4, dt = 0.002. Output: u on 32 points at 21 times in [0, 2] (672 values). Surrogate: the pendulum's MLP ensemble (5 members, frozen pilot settings `pilot_results/frozen_params.json`), MC-dropout model as for the pendulum; inputs mapped affinely onto [-1, 1].

**System 4: neural operator** (`mismatch/operator.py`). Same equation and missing terms, nu = 0.15 fixed. Input: an initial profile from a random Fourier series (6 modes, a_k, b_k ~ N(0, k^-2), scaled to max |u0| = 1.5) on 64 points. Output: u at the 20 times after 0 on 64 points. Surrogate: an ensemble of 5 one-dimensional Fourier neural operators (width 32, 16 modes, 4 layers; Adam 1e-3, cosine schedule, 400 epochs, early stopping patience 40 on a 10% validation split). No MC dropout.

Seeds: the same stream rule as the pendulum (`stream_seed(seed, name)`); training, test (500 points) and calibration streams are disjoint. Observation noise 1% of each trajectory's standard deviation, as for the pendulum.

## 3. Pilot (seed 0 only) and pred-v2

On seed 0, per system and term: eps_ref = the eps where r = 1 at n_nom = 512, by bisection on the pilot surrogate (the pendulum rule). Settings are not retuned (frozen pendulum MLP settings; FNO settings as in section 2). Then Model A's empirical plug-in (the pendulum fallback, used here directly because s_A > 1 is expected) is computed from seed 0's arrays on the eps grid, and the predicted r_1/2 (4PL on the seven measured levels) and the predicted large-r AUROC are frozen in `predictions_v2.md` by the tag `pred-v2` before any seed 1-20 PDE run.

## 4. Sweep, signals, metrics

Levels: eps in {0, 0.25, 0.5, 1, 2, 4, 8, 16} x eps_ref; n_nom = 512 only; seeds 1-20; 500 test points. Signals: (1) nominal ensemble sigma; (2) MC-dropout sigma (System 3 only); (3) PDE residual of the surrogate prediction (time-averaged RMS of u_t + u u_x - nu u_xx on the output grid); (4) split conformal on 200 nominal simulations; (5) split conformal on n in {8, 16, 32, 64} true points; D3 real-point records at 16 x. Not run on the PDE systems: discrepancy ensemble, GP, acquisition. Metrics as in RQ1: AUROC (event above the 80th percentile), Spearman with E, A, D, AUSE, coverage and width; per-point arrays saved.

## 5. Tests and predictions (all secondary to the pendulum families)

Per system s in {3, 4} and term in {P1, P2} (four combinations). "k of 20" counts use 20 seeds.

- **Family P (Holm over 4, alpha = 0.05). PB1:** per seed, slope of nominal-sigma AUROC on log r over the seven nonzero levels; hypothesis slope < 0 on all four combinations (dev: falls everywhere). Supported only if the Holm p < 0.05 and the 95% BCa CI lies below 0.
- **PA2 (transfer, pred-v2):** the pred-v2 r_1/2 lies inside the 90% bootstrap CI of the seeds 1-20 r_1/2 (10,000 resamples) and within a factor of 1.5 of its point estimate (the P-A rule). Inconclusive if a fit fails or the midpoint is outside the sampled r range.
- **PD1 (limit):** at 16 x, rho_S(sigma, E) - rho_S(sigma, D) within +/- 0.10, TOST across seeds.
- **PR1 (bracketing):** at 16 x, rho_S(signal, D) lies inside the full-dictionary alignment range (`mismatch/burgers.py` TERMS, 2,000 mixtures) in at least 18 of 20 seeds, for ensemble sigma and the PDE residual.
- **PR1s (strict):** at 4 x and 16 x, |rho_S(signal, D) - mean true-family prediction| <= 0.20 in at least 18 of 20 seeds (true families: P1 {u|u|, u^3}; P2 {sin 3x, cos 3x}). Dev gaps were at most 0.06.
- **PR3a / PR3b:** at 4 x, n = 16 real fields: the selected term's statistic exceeds ensemble sigma's in at least 18 of 20 seeds; for PR3a also at least 0.90 in at least 18 of 20.
- **Descriptive:** D3 counts at n = 8 to 64 (low power where rho_S(sigma, D) is near 0.1 to 0.2, so no prediction), D4 quantities, coverage, R1c and widths.

**Caveats stated now.** The dictionary was written knowing both true families; on P2 the sin(kx) responses are nearly collinear, so PR3b there is close to automatic (dev 0.98) and will be reported with that caveat. System 4 has the operator gate pass on one dev seed only.

## 6. Compute (estimate from dev timings)

System 3: about 4 minutes per seed (fits and about 20,000 solves), about 1.5 CPU-hours for 20 seeds. System 4: about 17 minutes per seed (5 FNO fits), about 6 CPU-hours. Alignment and R3: under 1 CPU-hour. If compute falls short, System 4 is reduced to 10 seeds first (logged in section 11 before results are seen).

## 7. Order of events

Author review, then freeze v1.7 (tag `prereg-v1.7`), then the PDE runner (`scripts/run_pde.py`, tested on dev seed 999), then the pilot on seed 0 and `pred-v2`, then seeds 1-20.

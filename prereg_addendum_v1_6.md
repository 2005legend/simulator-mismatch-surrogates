# Pre-registration addendum v1.6: Blind to the Missing Term

Drafted 2026-11-15. Status: **DRAFT, not frozen.** Section 9 lists what must be settled before it can be frozen by a git commit tagged `prereg-v1.6`, which must happen before any pilot run.

**Scope.** (1) A disclosure of further development-seed looks since v1.5. (2) Clarifications of how frozen definitions are implemented. (3) A change to how acquisition (RQ2) is analysed: Family A runs exactly as frozen, and a new Family D tests acquisition for a dynamics-space correction. (4) Secondary predictions about the alignment range. No frozen test, family, hypothesis (H1-H4), seed, or the Holm structure of Families A, B and C changes.

## 1. Disclosure: development-seed looks since v1.5

All on development seeds 999, 998 and 997. No pilot seed (0, 101, 102) and no main seed (1-10) was used. Rough ε_ref values (S1 0.01, S2 0.0026) were development guesses, not the pilot's.

| Look | What was seen (single seeds unless stated) |
|---|---|
| Output-space correction, full 10 s horizon, n_nom = 512 | With 56 true trajectories, the corrected error was no better than the uncorrected surrogate (S1: 0.063 uncorrected, 0.066-0.077 corrected). Five correction models tried; none clearly better. |
| Same, correction scored on the first 4 s, arms 1-7, 3 seeds, 12 cells | No arm beat Sobol; mean log(budget-AUC ratio) from -0.02 to +0.16 against between-seed SDs of 0.01 to 0.14. The nominal-residual arm was worse than Sobol in all 12 seed-cells. No clear growth of the gap from 1 x to 4 x for arms 3 and 4 (arm 3 on S1 rose from +0.055 to +0.072, inside the noise). A paired-seed rerun is incomplete and is reported separately. |
| Dynamics-space correction (a small network for a missing term inside the ODE, fitted to 8-56 noisy true trajectories, full 10 s) | Median error against the true system falls 1.3 to 7.6 times from the nominal ODE with 8 trajectories (smallest on S2 at the rough eps_ref, largest on S1 at 4 x) and a further 2 to 4 times by 56 (S1 at 4 x: 0.147 to 0.019 to 0.0049). A two-step fit takes 9-19 s against 85-150 s for plain gradient training. |
| Alignment range (section 5), n_nom = 512, 500 test points | Nominal ensemble sigma: range of Spearman correlation with the linear-response norm over nine hypothesised terms and their mixtures, [-0.61, +0.78]; with the true family removed, [-0.63, +0.74] for S1 and [-0.58, +0.77] for S2. Actual correlations with the true discrepancy: +0.73 (S1), -0.17 (S2), inside both ranges. |
| RQ1 smoke run (all seven frozen signals, mismatch 0, 1, 4, 16 x) | Ranking: nominal sigma AUROC S1 0.76, 0.90, 0.93, 0.93; S2 0.76, 0.64, 0.42, 0.41. MC dropout S2 down to 0.25. Simulator-calibrated conformal coverage 0.90 at zero mismatch, 0.34/0.33 at 1 x, near zero at 16 x. |
| Seed-stream defect | `sobol_thetas(56, seed=0)` equals the first 56 training points of seed 0, and a seed's test set equalled the next seed's training set. Fixed by deriving every stream from (seed, purpose, index) (section 3). No registered seed was affected because none had been run. |

**Consequences stated now.** (i) Family A (arms 3, 4 and 7 against Sobol, output-space correction) will most likely show no effect, because the correction cannot learn enough from 56 points; this is an expectation, not a change to A1-A3. (ii) Section 4 is a response to that evidence and is labelled **dev-informed**. (iii) The numbers in sections 5-6 were set after these looks.

## 2. Clarifications (implementation of frozen definitions, not changes)

1. **Budget-AUC** (prereg-v1 section 7: "trapezoid on log error") is implemented as exp of the trapezoid average of log median error over n = 8, 16, ..., 56, so the analysis ratio in A1/A2, log(AUC_arm / AUC_Sobol), equals the difference in trapezoid-averaged log median error.
2. **Correction model for all RQ2 arms** is the small MLP discrepancy ensemble (3 hidden layers of width 32 as in the surrogate code, M = 5, 300 epochs, no held-out split because n starts at 8). Arm 6's GP is used only to score candidates, not to correct.
3. **Arm 9** (exploratory) scores its first 8 points against the uncorrected nominal surrogate.
4. **Fit seeds** of the correction ensemble are the same for every arm, so arms are compared on identical fits. Arm-specific randomness is limited to batch sampling and score models.
5. **RQ1 conventions not fixed by prereg-v1.** Gaussian intervals use 1.645 times the relative norm of sigma against the relative total error. Conformal intervals are normalised by ensemble sigma (score = relative error divided by relative sigma). Calibration points are the first 8, 16, 32 and 64 of one scrambled Sobol stream, observed with the frozen 1% noise; the nominal calibration uses 200 simulator points. MC dropout: dropout 0.1, 50 passes, one network.
6. **Event for AUROC**: relative total error above its 80th percentile within each cell (frozen).
7. **Semigroup signal** (v1.3, exploratory): not built; will be reported as "not run" unless built before the pilot.

## 3. Seed streams

Every random stream is `stream_seed(seed, purpose, index)`, a hash of the replication seed, a purpose label (train, test, design, pool, noise, calibration) and an index. Pilot seed 0 and throwaway seeds 101 and 102 and main seeds 1-10 keep their frozen roles; only the derivation of data from them is specified. A runner refuses these seeds unless explicitly told otherwise.

## 4. RQ2: Family A as frozen, plus Family D

**Family A** (A1-A3 and the C1 gate of v1.1) runs unchanged: output-space correction, full 10 s horizon, arms 1-8 (arm 9 exploratory).

**Family D (new, 4 tests, its own Holm correction at alpha = 0.05).** Acquisition for a dynamics-space correction.

- *Correction model.* A small network h(phi, omega) added to the nominal acceleration, fitted to the observed noisy trajectories by gradient matching followed by 20 L-BFGS iterations of trajectory matching (code: `mismatch/dynamics.py`). Refitted after every round. Hyperparameters fixed in the pilot.
- *Arms.* 1 random; 2 Sobol; 3 nominal ensemble sigma; D5 disagreement among K = 5 learned terms (different seeds, same data), averaged over the candidate's trajectory; D6 state-coverage: the score of a candidate is the distance from its nominal-ODE trajectory, sampled at 40 times in the (phi, omega / 3) plane, to the nearest point of the observed trajectories so far (a space-filling rule in state space instead of parameter space). Same initial 8 points, 6 rounds of 8, pools of 4,096 and batch rule as prereg-v1 section 7.
- *Scored quantity.* Median relative L2 error of the learned-ODE rollout against the noise-free true system on the test set, full 10 s; budget-AUC as in clarification 1. The corrected surrogate (nominal surrogate plus the learned ODE minus the nominal ODE) is reported alongside; on the development seed it sits at the surrogate's approximation floor, so arms cannot separate on it.
- *Cells.* S1 and S2 at 4 x eps_ref, n_nom = 512, seeds 1-10.
- *Tests.* D1-D2: arm D5 vs Sobol on S1 and on S2. D3-D4: arm D6 vs Sobol on S1 and on S2. Paired t-test across seeds on G = log(budget-AUC_arm / budget-AUC_Sobol); supported only if Holm-adjusted p < 0.05 and the 95% BCa CI excludes 0 (frozen decision rule).
- *Stated stop rule.* If no arm beats Sobol in either system (no D-test supported), the paper reports acquisition as a secondary section whose finding is that where the correction is placed matters and which points are chosen does not; Koermer et al. (arXiv:2301.10228) is prior evidence for such a null. No further acquisition analysis is added after seeing results.

## 5. Alignment range: definition and secondary predictions (dev-informed)

**Definition** (code: `mismatch/alignment.py`). A dictionary of nine hypothesised missing terms h(phi, omega): omega, omega|omega|, omega^3, tanh(omega/0.1), tanh(omega/0.5), phi, phi^3, sin 2phi, phi omega. For each term, the linear response of the nominal trajectory (finite perturbation of size 1e-3). The alignment range of a signal is the minimum and maximum Spearman correlation between the signal and the relative response norm over the single terms and over 2,000 random signed mixtures of 2 to 3 terms (each term scaled to unit median response). It uses no true-system data.

**R1 (bracketing).** For nominal ensemble sigma and the nominal-ODE residual, at 16 x eps_ref and n_nom = 512, on each system and in each of seeds 1-10, the Spearman correlation between the signal and the true discrepancy norm lies inside the alignment range, computed (a) with the full dictionary and (b) with the true family removed from the dictionary (S1: omega|omega|, omega^3; S2: both tanh terms). Prediction: at least 9 of 10 seeds in each of the 8 (signal, system, dictionary) combinations. Secondary, uncorrected, reported as a count with exact binomial intervals.

**R2 (width, descriptive).** Report the width of each range. The development-seed widths are about 1.3-1.4 on a scale whose maximum is 2, which makes bracketing easy; R1 therefore supports the claim "a simulator-only signal cannot be certified without real data", not "the range predicts the sign". The paper will say so.

**R3 (hypothesis selection, exploratory, no test).** From n = 8, 16 and 32 real trajectories, choose the single dictionary term whose response norm has the highest Spearman correlation with the observed debiased discrepancy norms (as in v1.1 section 4), build the signal from that term, and report its Spearman correlation with the true discrepancy norm on the test set next to ensemble sigma and next to a learned-term ensemble (Family D model) at the same n.

## 6. Cut order and schedule

The working window is about ten days from 2026-11-14. If compute or time falls short, analyses are removed in this order, each recorded in section 11 of `preregistration.md` before the affected results are looked at: arm 9; G3 (measured amplification); SPI and test C2; P-B (the S2 direction of H4); G2 beyond the pendulum; Family D arm D6. Not cut: Families A, B and C tests that remain computable, RQ1 with its seven signals, the alignment range.

## 7. Out of scope here

PDE systems (1D Burgers with a missing term, a neural-operator surrogate) get their own addendum, v1.7, frozen before any run beyond feasibility checks. The semigroup signal, conformal calibration under acquisition-chosen points, and any guarantee for arm 8 are not covered.

## 8. Order of events

Freeze v1.6, then run the pilot (seeds 0, 101, 102) to set eps_ref, network settings, the batch exponent, spline smoothing and the seed count, then commit `pred-v1`, then freeze v1.7 before the PDE runs, then main seeds 1-10.

## 9. To settle before freezing

1. The paired-seed rerun of the 4 s arm-separation check finishes and is logged (it does not change Family A, but it is part of the disclosure in section 1).
2. Arms D5 and D6 and the Family D runner are built and tested (they do not exist yet).
3. The author reviews the numbers in section 5 (9 of 10 seeds; 2,000 mixtures; the dictionary) and section 4 (arm definitions, cells, test count). These are judgments and are fixed once tagged.
4. The nine-term dictionary needs a stated selection rule in the paper: it was chosen by me before looking at the range, and includes both true families, which is why the true-family-removed version is pre-registered next to it.

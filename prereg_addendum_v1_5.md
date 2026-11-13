# Pre-registration addendum v1.5: Blind to the Missing Term

Drafted 2026-11-13. Status: **frozen 2026-11-13** by the git commit tagged `prereg-v1.5`, after the author's review and before any pilot run.

**Scope.** (1) A disclosure. (2) Three secondary predictions and one exploratory analysis about where uncertainty ranking ends up at large r. (3) Three extra ε levels for the simulator-only RQ1 signals, needing no new network fits. No frozen test, success criterion, family, arm, seed or metric changes. Families A, B and C are run and reported exactly as frozen.

## 1. Disclosure: a development-seed look before the pilot

On 2026-11-13, while checking a derivation, the nominal ensemble σ was scored against the true-system error at ε > 0 on development seed 999 (3 members, n_nom = 512, 500 test points, rough ε_ref of 0.01 for S1 and 0.0026 for S2; `scripts/dev_bound_check.py`, tables in `stage1_notes_2026-11-13.md`). No pilot seed (0, 101, 102) and no main seed (1-10) was used. Observed:

- S1: AUROC of nominal σ for the total error rose with r (0.69 at r ≈ 0.15 to about 0.92 at r ≈ 4-14).
- S2: it fell below chance (0.68 to about 0.35-0.40).

The frozen hypothesis H1 and test B1 predict a negative AUROC slope on log r on both systems. On this evidence **B1 will probably not be supported on S1.** B1 is not changed. Everything in sections 3-5 was written after this look and is labelled **dev-informed**: these are predictions for seeds 1-10 made with knowledge of one development seed, and the paper will say so.

## 2. Why: the ranking limit

For a test point with approximation error a, discrepancy d and total error e = a + d, write A = ‖a‖, D = ‖d‖, E = ‖e‖. The triangle inequality gives |E − D| ≤ A and |E − A| ≤ D. So for any signal s, with κ = rms(A)/sd(D):

|ρ(s,E) − ρ(s,D)| ≤ 2κ/(1 − κ) for κ < 1,

and the mirror statement with A and D swapped. Ranking of the total error therefore moves from ρ(s,A) at small r to ρ(s,D) at large r. A simulator-only signal is a function of the nominal simulator and the surrogate alone, so two true systems with the same r can give it any ρ(s,D). "Going blind" is the special case ρ(s,D) ≈ 0. The worst-case rate above is loose on the dev seed (vacuous for r below about 2-4), so it fixes the limits, not the curve between them; Model A (v1.1, v1.4) remains the prediction of the curve.

**Wording change.** The paper's main claim becomes: *as r grows, the ranking ability of a simulator-only signal converges to its alignment with the discrepancy, which the simulator cannot reveal and which can be high (S1) or below chance (S2).* The title may change. H1-H4 keep their frozen wording and are reported against it.

## 3. Extra ε levels (RQ1, simulator-only signals)

For RQ1 signals 1-4 of `prereg-v1` section 6 (nominal ensemble σ, MC dropout σ, nominal-ODE residual, nominal-calibrated conformal) and the semigroup signal of v1.3, three levels are added: ε ∈ {4, 8, 16} × ε_ref, at n_nom = 512, both systems, seeds 1-10. These need true-system simulations only, on the same test sets and the same nominal surrogates. No new fits.

Reason: the frozen sweep stops at 2·ε_ref (r ≈ 2), and on the dev seed the limit is reached at r ≈ 4 or beyond. The frozen B1 and B2 use only the frozen levels.

## 4. Secondary predictions (dev-informed; one test each, uncorrected, reported as secondary)

All use nominal ensemble σ and Spearman correlation ρ_S, per seed, n_nom = 512.

- **D1 (limit).** At ε = 16·ε_ref, ρ_S(σ, E) equals ρ_S(σ, D) within ±0.10. Test: two one-sided tests on the per-seed difference across seeds 1-10, α = 0.05, separately for S1 and S2. The margin 0.10 is a judgment (dev-seed differences were about 0.01); it is fixed now.
- **D2 (opposite directions).** Per seed, slope of nominal-σ AUROC on log r over all seven nonzero ε levels. Prediction: mean slope > 0 on S1 and < 0 on S2. Two-sided paired t-test on each. This restates B1 for S2 and contradicts it for S1; B1 stays the primary test.
- **D3 (real data reveals the limit).** From the n ∈ {8, 16, 32, 64} true-system calibration points of RQ1 signal 5 at ε = 16·ε_ref, compute ρ̂_n = ρ_S(σ, D̂), with D̂ the debiased observed discrepancy norm of v1.1 section 4. Decision rule: call the signal "aligned" if the lower end of the 90% percentile-bootstrap CI of ρ̂_n (1,000 resamples) is above 0. Prediction: at n = 32 the rule says "aligned" in at least 8 of 10 seeds on S1 and in at most 2 of 10 on S2. Reported for all four n with exact binomial intervals. The back-of-envelope figure behind n = 32 is a Fisher-z calculation for ρ = 0.5 against 0 (about 31 points); it is not a result.

## 5. Exploratory (no test)

- **D4 (what sets the transition).** At each ε level report κ = rms(A)/sd(D), the coefficient of variation of D, and the distance of ρ_S(σ, E) from its two limits. Question: does the midpoint of the transition line up better with κ ≈ 1 than with r ≈ 1 across S1, S2 and the three n_nom values?
- The same limits (D1) for the other simulator-only signals and for the v1.2 horizon windows.
- **Acquisition.** The dev look did not cover RQ2. The reasoning of section 2 suggests arm 3 may not fall behind Sobol on S1, so A1 may not be supported on S1. This is an expectation, not a test; A1 is run as frozen.

## 6. What this addendum does not cover

PDE systems, neural-operator surrogates and any guarantee for arm 8 are not pre-registered here. A feasibility run on 1D Burgers with a missing term exists on dev seed 999 (`scripts/dev_burgers_pilot.py`). Any PDE study gets its own addendum before runs beyond feasibility.

## 7. Order of events

`prereg-v1.5` is tagged before the pilot. The pilot, `pred-v1` and the main seeds follow as in v1.4. The pilot date of 2026-11-25 is no longer binding (changelog, 2026-11-13).

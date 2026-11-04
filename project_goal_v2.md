# Blind to the missing term: goal, abstract and plan (v2 draft)

Status: v2, 2026-11-04. Scope option B approved by the author on 2026-11-04. Replaces the scope of "Blind to the missing term design.md" (v1, 2026-11-01). The abstract and pre-registration are frozen only after the week-1 novelty gates pass.

## Changelog

- 2026-11-04: Chose option B (RQ1 + RQ2 on S1 + S2, r as the dial, 10 seeds, 6 primary contrasts; failure search deferred to a second paper). Reason: v1's 5 seeds could not pass its own corrected contrasts, and RQ3 diluted the story at high cost. See "Scope decision" below.
- 2026-11-04: Week-1 novelty gate passed provisionally (9 papers read; see week1_novelty_multifidelity.md). Design changes adopted from it: H* worded as a measured dose-response, not a new mechanism (ensemble blindness to shared bias is textbook); RQ2 contrasts are slopes against log r so the ε = 0 offset cancels; RQ1 adds AUSE and reports calibration separately; report a discrepancy-smoothness proxy; avoid the term "model-form uncertainty".
- 2026-11-04: Drafted preregistration.md (not yet frozen). It supersedes items 4-5 below: primary tests are two Holm families, A (acquisition, 6 tests) and B (ranking and matched transitions, 4 tests), using paired t-tests on per-seed slopes with BCa bootstrap CIs; the exact sign test is a robustness check only, since with 10 seeds it needs all 10 in one direction to pass Holm. 1% observation noise is the main grid; 5% is exploratory on S1.

## Project goal

Measure when the signals used to judge a physics surrogate (ensemble spread, conformal intervals calibrated on simulations, the residual of the governing equation) stop pointing at its real errors because the simulator behind them is missing a term, and find how many true-system experiments, chosen by which rule, bring them back.

## Working title

Blind to the Missing Term: When Uncertainty and Physics Residuals Stop Guiding Data Collection for Physics Surrogates

## Abstract (pre-results draft)

Neural surrogates of physical systems are trained on a simulator, and the same simulator is then used to decide where the surrogate can be trusted: ensemble disagreement, conformal intervals calibrated on simulated data, and the residual of the governing equation. If the simulator leaves out a term that the real system has, these signals can see the surrogate's approximation error but not the gap between simulator and system. We study this with a controlled dial. A damped-pendulum simulator is paired with a "true" system that adds quadratic drag (S1) or smoothed Coulomb friction (S2) at strength ε. The two terms put the discrepancy in opposite regions of parameter space, so any agreement between uncertainty and discrepancy can be tested rather than assumed. We split each surrogate error into an approximation part and a structural part, and index every experiment by their ratio r, which we move with both ε and the number of nominal simulations. We ask two questions. First, how well does each uncertainty signal rank the true error (AUROC, Spearman correlation, 90% coverage) as r grows? Second, when at most 56 true-system experiments are spent to fit a correction model, which acquisition rule gives the lowest corrected error: Sobol design, nominal ensemble variance, nominal residual, discrepancy-ensemble variance, a Gaussian-process discrepancy model, or a residual computed on the observed true trajectories? We predict that the nominal-only signals lose their ranking ability and fall behind Sobol design at the same value of r. [Results go here.] Hypotheses, primary contrasts and analysis are pre-registered; every cell uses 10 paired seeds and bootstrap confidence intervals.

## Changes from v1, and why

1. **The dial is r, not ε alone.** Nominal simulations are cheap and unlimited, so the approximation error is a choice (v1 fixed it at 512 simulations). A reviewer can say "run more nominal simulations and only the discrepancy is left". Sweeping n_nominal ∈ {128, 512, 2048} together with ε, and plotting everything against r = ‖f_nom − f_true‖ / ‖f̂ − f_nom‖ (medians), answers that. If curves from different (ε, n_nominal) pairs collapse onto one curve in r, that is direct evidence for the mechanism. It also gives 15 r-points on S1 instead of 5, which the transition fit needs.
2. **"The two transitions coincide" gets a real test.** v1 asked for overlapping CIs, which almost always happens with 5 seeds and 5 ε levels. v2 fits a logistic curve in log r to each RQ1 and RQ2 metric, takes the midpoint r½, and reports a bootstrap CI on the difference between the two midpoints with a pre-set equivalence margin (proposed: a factor of 1.5 in r).
3. **RQ3 (failure search) moves to a follow-up paper.** It needs 1e7 ground-truth simulations and a different community's baselines, and its main comparison (rare-event estimators beat Monte Carlo at P_fail ≈ 1e-4 with 100-1000 samples) is largely textbook. It is a good second paper on the same code.
4. **10 seeds instead of 5.** With 5 paired seeds the exact sign test cannot go below p = 0.0625 two-sided, and a percentile bootstrap on 5 values is anticonservative (if all 5 differences are positive, every resample is positive). With 10 seeds the sign test reaches p = 0.002, which can survive a Holm correction over about 6 contrasts. Final count is set from the pilot's between-seed variance.
5. **Fewer primary contrasts (6, down from about 42).** Per system: arm 3 vs Sobol and arm 4 vs Sobol, tested as the slope of the corrected-error gap against log r; arm 7 vs arm 6 at the top r level. Everything else is exploratory.
6. **Static nominal arms get batch diversity.** Nominal σ and nominal residual never see true data, so they are fixed designs. Greedy top-8 picks would cluster, and AL4PDE already shows greedy Top-K can raise error with no mismatch at all. Use stochastic batch sampling (SBAL-style) so a loss is not just a batching artefact.
7. **True-system observations get noise (1% and 5%).** Arm 7 takes the nominal-ODE residual of observed true trajectories. With noise-free data and exact derivatives this hands the arm the missing term. With noise it needs finite differences or smoothing, which is the realistic case.
8. **Novelty check adds multi-fidelity active learning.** The setting (cheap nominal simulator, few expensive true queries, choose which true queries) is a multi-fidelity design problem, and v1's search did not cover that literature. Search "multi-fidelity active learning", "deep multi-fidelity", "multi-fidelity Bayesian optimization neural", and read the closest 2-3 in full. Also read Yang, Chen, Wu (arXiv:2502.05372) in full; it is acquisition under structural error with an NN correction.

## Scope decision (record)

**What matters.** The paper needs one clean, defensible result on RQ1 + RQ2. The risk that most threatens it is not compute but a reviewer reading H* as true by construction. The r-collapse and the transition-match test are the parts that are not obvious, so the design is built around them.

**Options.**

| Option | Contents | Time / compute (estimate) | Pros | Cons |
|---|---|---|---|---|
| A. v1 as written | RQ1-RQ3, S1-S3, 5 seeds | 18 weeks, 115-260 CPU-h | Widest paper | Stats too weak for the primary contrasts; RQ3 thins the story; full version misses ICLR workshops |
| B. Focused (recommended) | RQ1 + RQ2, S1 + S2, r sweep, 10 seeds; RQ3 later | ~16 weeks, 70-420 CPU-h | Answerable questions; mechanism test; S1-only cut ready for an ICLR 2027 workshop | No failure-search result in paper 1 |
| C. Minimum | RQ1 + RQ2 on S1 only, 5 seeds | ~10 weeks, ~20-60 CPU-h | Fastest | Can't test the S1-vs-S2 alignment; primary contrasts underpowered |

**Decision (2026-11-04): B**, with the S1-only subset written up as the workshop version in late January. I would switch to C only if the timing pilot shows fits above about 60 s each and the cluster of 8-12 workers can't finish the grid in time. I would go back toward A only if a quick RQ3 pilot shows something surprising, such as σ-guided failure search being worse than plain Monte Carlo.

**Not known yet.** Between-seed variance (sets the seed count), MLP fit time (sets compute; 10-60 s is a guess), and where r ≈ 1 falls in ε for each system. The week-3/4 pilot measures all three.

**Open item for the author:** the 2026-11-01 landscape report ranked the learnability candidate first (24 vs 21). v1 chose the mismatch study. [add: the reason for that choice, one line]

## Compute (estimates; only the 0.0215 s pendulum simulation is measured)

- RQ2 main grid: 2 systems × 5 ε × 7 arms × 10 seeds × 6 rounds × 5 members = 21,000 discrepancy-net fits.
- RQ2 n_nominal check: S1, ε at the level nearest r = 1, n_nominal ∈ {128, 2048}, 7 arms × 10 seeds × 6 rounds × 5 members = 4,200 fits.
- RQ1 extra: nominal pretraining for 3 n_nominal × 2 systems × 10 seeds × 5 members = 300 fits.
- Total about 25,500 fits at 10-60 s, so 70-425 CPU-hours, or 1-2 days on 10 workers at the upper end. RAM (about 5 GB free) caps the worker count.

## Next steps

| Dates | Work | Gate / output |
|---|---|---|
| Nov 4-10 (wk 1) | Full-text reads: arXiv:2610.27130, Jaber et al. SSRN 7272715, Yang/Chen/Wu 2502.05372, top 2-3 multi-fidelity AL papers. Freeze the goal and abstract. Commit `preregistration.md` with H*, the 6 primary contrasts, the equivalence margin and the r definition. | Reframe if any paper already sweeps mismatch strength for NN-surrogate acquisition or UQ ranking |
| Nov 11-24 (wk 2-3) | Code: simulators S1/S2 with ε and noise switches; error decomposition; nominal ensemble training; acquisition loop with all 7 arms; unit tests (ε = 0 gives zero discrepancy; decomposition sums exactly). Timing pilot (10 min). | Measured fit time replaces the 10-60 s guess |
| Nov 25-Dec 1 (wk 4) | Pilot: S1, 3 seeds, all ε, n_nominal = 512. Calibrate ε_ref so r = 1 sits mid-sweep. Freeze hyperparameters on the pilot seed. | Seed count from measured variance. Go/no-go: does nominal-σ AUROC on the discrepancy component drop at all? |
| Dec 2-Dec 29 (wk 5-8) | Full S1 grid (RQ1 + RQ2 + n_nominal check). | S1 results and figures |
| Dec 30-Jan 26 (wk 9-12) | S2 grid in the background; write the S1 workshop paper (4 pages). | Workshop draft ready before the estimated late-Jan / Feb deadlines |
| Jan 27-Mar 2027 (wk 13-18) | S2 analysis, transition-match test across systems, full paper, clean repo with one-command reproduction, arXiv post. | arXiv preprint by end of March |

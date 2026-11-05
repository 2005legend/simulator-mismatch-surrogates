# Pre-registration addendum v1.3 (exploratory only): Blind to the Missing Term

Drafted 2026-11-05. Status: **frozen 2026-11-05** by the git commit tagged `prereg-v1.3`, before any pilot run (pilot starts 2026-11-25).

**Scope.** Two exploratory analyses. No confirmatory or secondary test changes; Families A, B and C, P-A, P-B, P-C and G1-G5 of earlier tags are untouched. Both analyses re-score saved arrays and need **no extra network fits**. Neither can be cut by compute.

**Source.** An external gap report (pasted 2026-11-05) proposed a semigroup-consistency acquisition filter and a design-collapse claim. Checked against its cited sources: Shikhman (arXiv:2605.26324) states that semigroup error measures internal coherence, not agreement with the true dynamics, and the true pendulum with drag or friction is itself autonomous, so a missing term does not break the semigroup law. The analyses below test that reasoning rather than adopt the report's claim.

## E1: semigroup error as an additional nominal-only RQ1 signal

**Signal.** For each test point and the nominal surrogate f̂ (inputs c, L, φ0, ω0; output φ on the 201-point grid over [0, 10] s), and for split times s ∈ {2, 4} s:
1. Read φ̂(s) from the direct prediction and ω̂(s) by central finite difference of the predicted trajectory.
2. Composed prediction: f̂(c, L, φ̂(s), ω̂(s)) on [0, 10 − s].
3. Direct prediction: f̂(c, L, φ0, ω0) on [s, 10].
4. Semigroup error E_sg = ‖composed − direct‖ / (‖direct‖ + 10⁻⁶), averaged over the two split times (definition of 2605.26324, Definition 1).

If the composed initial state (φ̂(s), ω̂(s)) falls outside the training box, the point gets no E_sg; the fraction excluded is reported per cell.

**Metrics.** The RQ1 metrics of `prereg-v1` section 6 (AUROC for the large-error event, Spearman, AUSE) against ‖e‖, ‖a‖ and ‖d‖ separately, in every RQ1 cell.

**Expectation, stated now (exploratory, no test).** E_sg tracks the approximation error a, not the discrepancy d. Its AUROC against total error should therefore fall with log r in the same way as nominal ensemble σ, and its Spearman correlation with ‖d‖ should stay near zero at every ε. If instead E_sg keeps ranking true error at large r, that contradicts the reasoning above and is reported as such.

## E2: design clustering of each acquisition arm

For each arm, ε and seed, on the final 56-point design with θ rescaled to the unit cube:
- δ_min: minimum pairwise distance;
- mean nearest-neighbour distance;
- the same two quantities for the arm-2 (Sobol) design as reference.

Reported descriptively against r and alongside G3's density-ratio summary C_eff. Purpose: check directly whether any arm's design collapses into clusters as r grows, a claim made without evidence in the external report. No test.

## Order of events

Unchanged from `prereg-v1.2` section 8. This file is committed and tagged before the pilot.

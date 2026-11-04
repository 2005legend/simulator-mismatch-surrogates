# Gap mining: risk papers and recent future-work sections (2026-11-04)

Purpose: (1) close the three "could narrow the novelty" reads flagged in `reports/Mismatch study novelty upgrade.md`; (2) mine the limitations and future-work sections of the closest recent papers for gaps our setup can fill cheaply. All papers below were read in full text through alphaXiv unless marked.

## 1. The three risk papers: none predicts the transition

| Paper | What it does | Threat to our claim | Use |
|---|---|---|---|
| DEUP, Lahlou et al. (arXiv:2102.08501, TMLR 2023) | Epistemic uncertainty = predicted excess risk from a secondary error predictor trained on observed out-of-sample errors; argues variance misses bias; used for acquisition in sequential optimisation and RL. | **Low.** No prediction of when variance stops ranking error, no dial, no physics. It is a natural baseline reviewers will ask for. Its own future work: "theoretically studying the properties of DEUP under model misspecification". | Add a DEUP-style arm (gap G4). Cite for "variance misses bias". |
| Amortila, Cao, Krishnamurthy (arXiv:2401.12216, COLT 2024) | Under L∞ misspecification and covariate shift, least squares suffers "misspecification amplification": test error ≥ C∞·ε²; proposes disagreement-based regression (DBR), not computationally tractable. | **Low.** Worst-case regression bound; no acquisition, no ranking. Future work explicitly asks to "study empirical issues, to understand how pervasive and problematic misspecification amplification is". | Gap G3: measure amplification per acquisition arm. |
| 2605.21348 (Polanska et al.) | Already read in full in v1 (residual acquisition against the exact PDE). | None new. | — |

Also read: Sloman, Caprio, Kaski (arXiv:2505.23496): a loose epistemic-error bound with model-restriction, data-scarcity and shift terms; no transition prediction (low threat). Gollakota et al. (arXiv:2502.20375): a loss predictor beats a model's own uncertainty estimate if and only if the model fails multicalibration; this gives RQ1 a formal reading (an error predictor trained on true points can only help where the simulator-trained model is miscalibrated on an identifiable region of θ).

**Verdict:** the core novelty claims (predicting the ranking and acquisition transitions; arm 7; r-gated acquisition) survive. Estimated novelty stays at the v3 level (~58-60 for the workshop scope).

## 2. Future-work statements that line up with our setup

| Source | Their stated limitation / future work | What our setup can do |
|---|---|---|
| Sale, Bülte, Czaja, Stiller, Hüllermeier, "What Uncertainties Do We Need for Dynamical Systems?" (arXiv:2606.11988, EIML@ICML 2026) | Uses **the same damped pendulum with viscous vs Coulomb friction** as its structural-uncertainty example. Says structural uncertainty "cannot be answered from inside any single model, nor with additional data" and is "notoriously difficult to represent (and quantify)". Claims, qualitatively, that structural uncertainty dominates at long prediction horizons and Coulomb friction shows only near rest. Proposes uncertainty in derived quantities (time to stabilisation) as an opportunity. | Quantify their qualitative claims: horizon-resolved transition (G1), derived-quantity metric. Our true points answer "cannot be quantified from inside the model". |
| Rizvi & Tomar, "How Should a Sim-to-Real Transfer Budget Be Spent?" (arXiv:2606.22062, IROS 2026 workshop) | Pendulum, sim-to-sim, 10 paired seeds; only parameter mismatch. Limitation and future work: test "where this ordering changes under structural model mismatch", i.e. whether identifying parameters or something else should come first. | Identify vs correct under structural mismatch, as a dose-response in r (G2). |
| Tang, Sloman, Kaski (arXiv:2506.07805, AISTATS 2026) | Their threshold τ is fixed regardless of misspecification degree, and they speculate this causes the shrinking advantage; future work: choose hyperparameters automatically. Bound tightness untested. | Arm 8 is an r-adaptive rule (already planned); G3 measures their amplification term directly. |
| Amortila et al. (above) | Empirical prevalence of misspecification amplification unknown. | G3. |
| SPI, Bashari et al. (arXiv:2505.13432) | Assumes i.i.d. real calibration points; extending beyond i.i.d. is future work. Experiments only with images and tabular data. At α = 0.1, β = 0.4, m = 15, their guaranteed coverage range is [0.812, 0.937] and misaligned classes sit near the floor. | **Risk to Family C test C2** (section 4). Paper 2: SPI with acquisition-chosen true points (G6). |
| Chen et al., digital-twin review (arXiv:2611.04269, 3 Nov 2026) | Open questions: "which experiments justify their cost" when combining simulation and physical experiments (RQ 7.1.3); "statistically grounded criteria for when sufficient evidence has accumulated to accept an update" (RQ 7.2.1). | RQ2 and the r̂ gate answer these directly; cite in the introduction. |

A search for horizon-resolved uncertainty reliability under structural error (alphaXiv, 12 results, titles and abstracts only) found long-rollout stability and diagnostics papers (e.g. 2605.26324, 2609.19674) but none measuring when simulator-only uncertainty stops ranking true error as a function of horizon. Treat G1 as open, pending a full-text check of the closest two.

## 3. Gaps ranked for adoption

| ID | Gap | Why it is new | Extra fits | Effort | Fits before Jan? |
|---|---|---|---|---|---|
| **G1** | **Horizon-resolved transition ("time to blindness").** Compute r(T_h), AUROC and the arm-vs-Sobol gap on prefix windows t ∈ [0, T_h], T_h ∈ {2, 4, 6, 8, 10} s. Model A predicts the horizon t* at which r(T_h) crosses r½, from simulator quantities alone. Prediction: S2's t* is later than S1's relative to its r curve, because Coulomb friction acts near rest. | Turns Sale et al.'s qualitative "structural uncertainty dominates at long horizons" into a measured, predicted quantity, on their own example system. Not found in search. | 0 (re-scoring saved trajectories) | ~3 days | **Yes** |
| **G2** | **Identify vs correct under structural mismatch.** Upgrade the planned system-ID baseline to a three-way comparison at n = 24 and n = 56: (a) refit c, L by least squares and retrain the nominal surrogate; (b) the discrepancy correction (current); (c) refit then correct. Report the gain of each per true experiment against r. | Exactly the open question in 2606.22062, and the "parameter estimation vs discrepancy correction" choice the digital-twin review (sec. 6.4) says should depend on the mismatch. | ~1,000 (S1 + S2) | ~1 week | S1 yes; S2 March |
| **G3** | **Measured misspecification amplification.** For each arm and ε, estimate the design-vs-test density ratio (classifier-based, with a bootstrap CI) and regress corrected error on (density ratio × r²). Compare with Model B. | Answers the empirical-prevalence question in Amortila et al. and the bound-tightness question in Tang et al., on a neural surrogate. | 0 | ~2 days | **Yes** |
| G4 | **DEUP-style arm** (an error predictor fitted on observed |y − corrected prediction| at the true points, acquire where predicted error is high). | Baseline reviewers will ask for; not a novelty gain on its own. | ~1,500 (S1) | ~3 days | S1 yes |
| G5 | **Derived-quantity metric:** error in time to stop / energy-decay time, alongside trajectory L2. | Sale et al.'s suggested direction; cheap. | 0 | ~1 day | Yes |
| G6 | **SPI with acquisition-chosen calibration points**, using the known sampling probabilities of the batch rule as weights (feedback-covariate-shift conformal; Fannjiang et al. 2022, PNAS, known from memory, not re-read). | SPI's stated non-i.i.d. future work, made concrete. | 0 | 2-3 weeks of theory checking | No (paper 2 / March) |

**Recommendation:** adopt G1, G3 and G5 for January (no fits); G2 on S1 for January and S2 for March; G4 as exploratory if compute allows; G6 to paper 2.

## 4. A risk found in the frozen addendum: test C2

`prereg-v1.1` test C2 asks augmented-SPI coverage > 0.87 at n_cal = 8 and 16, averaged over ε. SPI's worst-case coverage at α = 0.1, β = 0.4 is well below 0.9 for small m (their range at m = 15 is [0.812, 0.937]), and misaligned synthetic scores push coverage toward that floor. At large r, plain SPI is expected to be misaligned, and augmented SPI only partly fixes that. So C2 may fail because of a known property of SPI, not because the augmentation is poor.

What to do (decide before the pilot, log in section 11):
- **Option 1 (recommended):** compute SPI's Theorem 3.5 worst-case floor for m ∈ {8, 16}, N = 2,000, β = 0.4 now (no data needed), record it, and keep C2 unchanged. If C2 fails, the paper reports coverage relative to that floor. No change to a frozen test.
- **Option 2:** amend C2 in `prereg-v1.2` to "coverage ≥ the Theorem 3.5 floor + 0.02". Cleaner test, but it changes a frozen confirmatory test (allowed before the pilot, but visible).
- **Option 3:** leave as is. Risk: an uninterpretable failure.

## 5. Handling against the pre-registration

All adopted items go into one `prereg-v1.2` addendum, committed before the pilot (2026-11-25):
- G1 and G3 as **secondary analyses with frozen predictions** (t* per system from Model A; the sign of the amplification slope), predicted numbers committed in `pred-v1` with the P-A/P-B numbers.
- G2 as a **secondary** comparison (one paired test per system at n = 56: refit-then-correct vs correct-only), not added to Family C, to keep Family C at 3 tests.
- G4 and G5 as exploratory.
- Cut order update: G4 is cut before anything in `prereg-v1.1`'s list.
- The C2 decision from section 4.

Compute: G2 + G4 ≈ +2,500 fits on top of the v3 total (~+50% over the original 25,500). The timing pilot decides whether G4 survives.

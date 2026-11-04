# Part B novelty check (full text): does surrogate uncertainty track true error under structural mismatch and parameter extrapolation?

Date: 2026-11-01. Scope: Part B of the SciML workshop project. A deep-ensemble MLP surrogate maps theta to a trajectory. It is trained on a NOMINAL simulator, but the true system adds a structural term at strength epsilon (for example quadratic drag or Coulomb friction).

## Deviations / limits (read first)
- Full texts were read from arXiv HTML (arxiv.org/html/<id>). The DOI route failed with CrossRef 404s. Reading was targeted: abstract, setup, methods, the relevant experiments/appendix, and keyword passes (misspecif*, model-form, structural, shift, OOD, ensemble, conformal, exchangeab*). No paper was read line by line from start to finish.
- arXiv API: 20 queries plus 4 extra. 12 first hit HTTP 429 and were retried with back-off; all 24 eventually returned. Results were capped at 30 per query and sorted by submission date.
- OpenAlex: 12 queries, 25 results each (from 2024-01-01); 287 unique works. Titles were screened, and only near-hits were opened.
- Web search: 2 queries.
- Three non-arXiv items were judged from title or OpenAlex abstract only, marked [abstract-only] below. For two SSRN items no abstract was available ([title-only]).
- All "absent" calls are therefore **provisional**: they mean not found in the texts and searches above.

## 1. Assigned papers

### 1.1 Mouli, Maddix, Alizadeh, Gupta, Stuart, Mahoney, Wang, "Using Uncertainty Quantification to Characterize and Improve Out-of-Domain Learning for PDEs", arXiv:2403.10642 (v2 12 Jun 2024; ICML 2024 per the task brief, venue not stated in the HTML text read)
- **What it does:** neural operators (FNO) map PDE coefficients to solutions. Training uses an under-specified parameter range, and tests use parameter values outside it at small, medium and large shifts. The PDEs are the GPME family (heat, porous medium, Stefan), linear advection and Darcy flow. It shows that cheap UQ methods (MC dropout, Laplace, variance-NO and others) fail OOD. Deep ensembles (EnsembleNO) produce uncertainty that is "well-correlated with prediction errors" (n-MeRCI about 0.05 vs 0.8 on heat). It proposes DiverseNO (multi-head diversity) and Operator-ProbConserv (uses the variance to apply conservation constraints).
- **Model-form shift?** No. Full-text keyword pass: 0 hits for "model-form" and "misspecif"; the only "structur" hit is unrelated. The OOD task is defined purely as test inputs (PDE parameters) outside the training support. The governing equation is the same at train and test. The appendix (B–F) adds more PDEs and shift magnitudes, still only parameter shift.
- **Verdict: PARTIAL (on the parameter-extrapolation arm of Part B).** It already shows that ensemble spread tracks error under parameter extrapolation for operator surrogates, using a correlation-type metric across increasing shift.
- **Delta to claim:** a structural or missing-term shift at fixed theta. The input theta is unchanged, so epistemic spread has no input-space signal to react to. The paper must cite Mouli et al. as the parameter-extrapolation reference and should not present that arm as new.

### 1.2 Ma, Azizzadenesheli, Anandkumar, "Calibrated Uncertainty Quantification for Operator Learning via Conformal Prediction", arXiv:2402.01960 (Feb 2024)
- **What it does:** risk-controlling quantile neural operator (UQNO), a split-conformal, function-space coverage guarantee. Experiments are Darcy flow (data-rich) and 3D car surface pressure (data-scarce).
- **Shift/misspecification?** None: 0 hits for "shift" and "misspecif". Coverage relies on the calibration set being exchangeable with test data.
- **Verdict: NO OVERLAP** with the question; it is a method citation for conformal UQ on operators.
- **Delta:** the project tests what happens when calibration data come from the nominal simulator but test data come from the true system, which breaks exchangeability by an amount set by epsilon.

### 1.3 McShannon & Dietrich, "Silent Failures in Physics-Informed Neural Networks: Parameter Poisoning and the Limits of Loss-Based Validation", arXiv:2606.25151 (Jun 2026)
- **What it does:** PINNs are trained with wrong PDE parameters (Burgers, cavity Navier–Stokes, convection–diffusion). They reach low loss but give wrong solutions (up to 71% fixed-sweep, 128% adversarial). Six detection defenses are tested; a "same-parameter ensemble" of three models fails ("ensemble std 0.470 vs error 0.175"), as does a parameter-jitter ensemble. It proposes a post-hoc sweep of residual loss over parameter values.
- **Verdict: PARTIAL.** It shows ensembles do not detect misspecified physics. But the misspecification is a wrong parameter value in a PINN, not a missing structural term in a data-driven theta-to-trajectory surrogate. The ensemble has only three members, and nothing is swept over a mismatch magnitude for UQ-error tracking. There is no conformal method, no coverage-vs-epsilon curve and no rank-correlation analysis.
- **Delta:** the epsilon-sweep curves (coverage and rank correlation vs epsilon), conformal calibrated on simulator vs true data, and a supervised surrogate rather than a PINN. Cite it as converging evidence that loss- and ensemble-based self-checks are blind to wrong physics.

### 1.4 Payot, Pasquato, Trani, Hezaveh, Perreault-Levasseur, "Active learning meets fractal decision boundaries: a cautionary tale from the Sitnikov three-body problem", arXiv:2311.18010 (v1 29 Nov 2023, astro-ph.GA; venue not verified)
- **What it does:** uncertainty-sampling and ranked-batch active learning for a stability classifier on a chaotic three-body problem. AL "does not result in improved performance" over random; the authors attribute this to the fractal decision boundary.
- **Verdict: NO OVERLAP for Part B** (no mismatch, no UQ-error tracking). It is relevant to Part A as a precedent for AL losing to random sampling, but the cause there is chaos, not model mismatch.
- **Delta:** n/a for B. For A, mismatch-induced failure is a different mechanism from fractal-boundary failure.

### 1.5 Dale, Li, DeCost, Wan, Han, Fehlis, Hattrick-Simpers, "When Active Learning Fails, Uncalibrated Out of Distribution Uncertainty Quantification Might Be the Problem", arXiv:2511.17760 (Nov 2025)
- **What it does:** materials property tasks (solubility, bandgap, formation energy) with ALIGNN deep ensembles, XGBoost, RF and NN. ID/OOD splits come from a prior split protocol or from clustering. ID-fitted calibration "can degrade OOD uncertainty estimates". Calibrated-uncertainty AL on OOD data generally fails to beat random or uncalibrated selection.
- **Verdict: PARTIAL.** It shows that UQ calibrated in-domain fails to transfer OOD and that this hurts AL. The OOD here is covariate/chemistry shift in tabular or graph data, not a structural physics mismatch, and there is no sweep over mismatch magnitude.
- **Delta:** a controlled, continuous, known-magnitude structural mismatch (epsilon) in a dynamical system, which yields a dose-response curve rather than a single ID/OOD split. Cite it as the closest "calibration does not transfer, so AL fails" precedent and link Parts A and B through it.

### 1.6 Jiang, Xia, Yu, Sun, et al., "CAAL: Confidence-Aware Active Learning for Heteroscedastic Atmospheric Regression", arXiv:2602.11825 (Feb 2026)
- **What it does:** AL on particle-resolved aerosol simulations and observations. It decouples mean and variance training and weights epistemic uncertainty by predicted aleatoric uncertainty. Its distribution-shift content is only a reference citation (1 hit).
- **Verdict: NO OVERLAP for B.** Heteroscedastic noise is not the same thing as model-form mismatch.
- **Delta:** n/a. At most, cite it as an AL acquisition baseline variant for Part A.

## 2. New papers from fresh searches (not in the earlier scan)

### 2.1 Gopakumar, Gray, Oskarsson, Zanisi, Giles, Kusner, Pamela, Deisenroth, "Uncertainty Quantification of Surrogate Models using Conformal Prediction", arXiv:2408.09881; Machine Learning: Science and Technology 7(1), 2026 (full text read)
- **What it does:** split CP (CQR, absolute-error, std scores) on many surrogates (MLP, U-Net, FNO, ViT, GNN). It includes an OOD test where the wave equation is solved "with half the wave velocity used during training". Uncalibrated MC dropout fails there, while CP with **new calibration data drawn from the shifted regime** restores coverage. It also covers zero-shot MPP on MHD. It states that the key requirement is exchangeability between calibration and prediction, not similarity to training data, and gives a tokamak L-mode vs H-mode example where non-exchangeable calibration fails.
- **Verdict: PARTIAL, and the closest paper for the conformal half of Part B.** It already establishes (a) that ensemble- or dropout-type uncertainty under-covers in a new physics regime and (b) that CP recovers coverage if you can calibrate on the new regime.
- **Delta:** (i) calibration on the **nominal simulator** while deployment is on the **true** system, with coverage loss mapped as a function of epsilon. Gopakumar et al. always recalibrate in-regime or show a single non-exchangeable case. (ii) The **sample complexity of recovery:** how many true-system calibration points restore coverage at each epsilon. (iii) **Conditional and ranking** quality (Spearman, AUROC for err > tau), not just marginal coverage.

### 2.2 Ma, Boullé, Yang, Wu, Guo, "Physics-guided correction for operator learning under model misspecification", arXiv:2606.03469 (v2 Aug 2026)
- **What it does:** DeepONet correction operator for misspecified governing equations: 1D diffusion-reaction, 1D Burgers (Cases A–C, including a perturbed advection nonlinearity), 2D cavity flow and hyperelasticity. Deep ensembles (10 seeds) are used only to put error bars on the *corrected* model under sparse or noisy data.
- **Verdict: PARTIAL (setting only).** It uses the same "nominal physics lacks a term" setup but is about correction, not about whether UQ detects mismatch.
- **Delta:** the project measures UQ-error tracking of the *uncorrected* nominal surrogate vs epsilon. Its misspecification cases are a useful template for the structural-term design.

### 2.3 Swinburne & Perez, "Parameter uncertainties for imperfect surrogate models in the low-noise regime", MLST 2024 (doi 10.1088/2632-2153/ad9fce) [abstract-only, plus the IOP page snippet]; and Perez, Subramanyam, Maliyov, Swinburne, "Uncertainty Quantification for Misspecified Machine Learned Interatomic Potentials", arXiv:2502.07104 (full text, abstract and introduction read)
- **What they do:** they argue that Bayesian (and by extension ensemble) uncertainty ignores misspecification and vanishes as data grow. They propose POPS (pointwise-optimal parameter sets), a misspecification-aware ensemble that bounds test errors, and demonstrate it on interatomic potentials.
- **Verdict: PARTIAL (conceptual).** The principle that standard Bayesian or ensemble UQ misses model-form error is already published. Their misspecification is, however, surrogate-class vs simulator (the surrogate cannot fit the simulator), not simulator vs reality.
- **Delta:** in the project the surrogate fits the simulator well and the gap is simulator vs truth, which no amount of simulator data can reveal. **POPS-style misspecification-aware ensembles are a natural extra baseline.**

### 2.4 NeuroForge: "Self-Auditing Neural CFD Surrogates with Calibrated Physics-Residual Trust", arXiv:2607.10333 (Jul 2026), full text, abstract and section map read
- **What it does:** it uses the RANS residual of a CFD surrogate prediction as a trust signal and finds positive rank correlation with field error across backbones. It adds split-conformal calibration of that trust signal, and the paper also reports a deep-ensemble band.
- **Verdict: PARTIAL (method family).** Its residual is computed against the *assumed* equations, so by construction it cannot see a missing structural term. That is the same blindness the project tests.
- **Delta / design hook:** include a "nominal-physics residual" trust score as a baseline and show it is epsilon-blind too, or not.

### 2.5 Other items screened

| Paper | Read | Overlap | Note |
|---|---|---|---|
| Paplhám, Waegeman, Hüllermeier, Franc, "Evaluating Epistemic Uncertainty: Beyond OOD Detection and Active Learning", arXiv:2607.14817 | abstract and section map | none (vision/tabular) | Argues OOD detection and AL are misaligned proxies for the quality of epistemic uncertainty; supports evaluating via a reject option / regret. |
| Lauer, Blake, Freund, "Soft Checksums to Flag Untrustworthy ML Surrogate Predictions", arXiv:2412.03497 | abstract | none (input-OOD) | Possible cheap self-check baseline. |
| "Into the danger zone: stable extrapolation...", arXiv:2609.36709 | section map | none (theory of OOD extrapolation) | — |
| "The Dynamic-Probabilistic Consistency Gap in Chaotic Surrogate Modeling", arXiv:2605.31547 | section map | none | — |
| "Exploration & Experimentation with a Misspecified Simulator", arXiv:2605.21458 | abstract | none for B (causal / operations) | Weakly relevant to A: when to collect real data under simulator drift. |
| "Benchmarking Multi-fidelity Neural Operators ... with Non-trivial Fidelity Differences", arXiv:2608.04708 | abstract | none for B | LF/HF governing-equation discrepancy, but not UQ tracking. |
| Thorat & Nayek, model-form error inference (SSRN 7191284), and 2601.05297 | title / abstract | none | Model-form error *correction* with a GP latent-force model in structural dynamics; a KOH-type baseline family. |
| Jaber, Vallot, Dairay, Mougeot, "Conformal risk control for model-form uncertainty in parametric non-intrusive reduced-order models", SSRN 7272715 (2026) | **[title-only — no abstract available; NOT verified]** | potentially PARTIAL | Needs a manual read: CP for model-form uncertainty in ROMs could be close to the conformal half of Part B. **Highest-priority unread item.** |
| Ford & Kulcinski, "Calibrated Uncertainty, Conformal Prediction, and Abstention for Safety-Critical Fusion Control...", Zenodo 2026 | abstract-only | none substantive | Non-peer-reviewed; input-OOD abstention. |
| Physics Attention Transformer, C-Mod to SPARC, arXiv:2608.24785 | web snippet | none | Ensemble spread vs error under machine shift; it explicitly excludes model-form uncertainty from its intervals. |
| Crash-surrogate MC dropout vs deep ensemble, arXiv:2607.18294 | web snippet | none | Ensembles accurate but overconfident in-distribution. |

## 3. Overall verdict for Part B
**PARTIALLY TAKEN** (confidence about 0.65, provisional on the unread SSRN 7272715).

Each building block is published:
- The parameter-extrapolation arm: ensembles track error OOD for neural operators (Mouli et al. 2024).
- CP restores coverage only when calibration is in-regime (Gopakumar et al. 2024/2026).
- Bayesian or ensemble UQ ignores misspecification (Swinburne & Perez 2024; Perez et al. 2025).
- Ensembles and residuals do not detect wrong physics in PINNs (McShannon & Dietrich 2026).
- ID-calibrated UQ fails OOD and breaks AL (Dale et al. 2025).

No paper found does the specific thing: a **controlled, continuous structural-term sweep (epsilon)** on a theta-to-trajectory surrogate trained on a nominal simulator, measuring **ensemble and conformal coverage, error-rank correlation and failure-flag AUROC as functions of epsilon**, with the calibration source (simulator vs n true points) as a factor.

## 4. Design recommendations that fall out of the reading
1. **Acknowledge the tautology up front.** With theta unchanged, the ensemble sigma(theta) is *identical* for every epsilon, because it depends only on the nominal training data. Marginal coverage must therefore decay as the discrepancy delta_epsilon(theta) grows. The non-trivial question is whether sigma(theta) is *accidentally aligned* with delta_epsilon(theta): for example, does quadratic drag matter most at high velocity or large amplitude, where ensembles may also disagree? Report the rank correlation corr(sigma, |delta_epsilon|) and when it holds. Without this framing, reviewers will call Part B obvious.
2. **Separate the two shifts.** Run epsilon at in-range theta and theta-extrapolation at epsilon = 0 as separate arms, then the 2x2 combination. Cite Mouli et al. for the second arm and do not claim it.
3. **Conformal arms (as Gopakumar et al. require):**
   - (a) Split CP calibrated on nominal-simulator data. Expect coverage to fall with epsilon.
   - (b) CP calibrated on n true-system points, n in {5, 10, 20, 50}. Report the n needed to restore 1-alpha at each epsilon.
   - (c) Optionally, weighted or adaptive CP.
4. **Baselines reviewers will expect:**
   - deep-ensemble sigma;
   - MC dropout (cheap-UQ foil);
   - KOH / GP discrepancy fitted on the same n true points (the classical model-form-error answer);
   - a POPS-style misspecification-aware ensemble (optional);
   - a nominal-physics residual trust score (NeuroForge-type; expected to be epsilon-blind).
5. **Metrics:**
   - coverage vs epsilon;
   - interval width;
   - Spearman(sigma, error);
   - AUROC / AUPRC of sigma for the event ||f_hat - f_true|| > tau, which reuses Part C's failure event and ties B to C;
   - risk-coverage (selective prediction) curves, following Paplhám et al.'s critique of OOD-detection and AL proxies.
6. **Link to Part A:** Dale et al. (calibration fails to transfer, so AL does not beat random) is the natural precedent. Report whether the epsilon at which the sigma-error correlation collapses predicts the Part A crossover epsilon*. That cross-part result is the most novel claim available.
7. **Must-read before submission:** Jaber et al., SSRN 7272715 (conformal risk control for model-form uncertainty in ROMs). Not verified here.

## 5. Citations verified in this note
Bibliographic details for 2403.10642, 2402.01960, 2606.25151, 2311.18010, 2511.17760, 2602.11825, 2408.09881, 2606.03469, 2502.07104, 2607.10333, 2607.14817, 2412.03497, 2609.36709, 2605.31547, 2605.21458 and 2608.04708 were checked against arXiv HTML title pages or API records. MLST 10.1088/2632-2153/ad9fce was checked via OpenAlex. SSRN 7272715 and 7191284 are title and author only, via OpenAlex; their content is unverified.

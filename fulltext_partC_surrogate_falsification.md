# Part C full-text novelty check: rare-event and falsification search for regions where a trained surrogate's own error is large

Date of check: 2026-11-01. Scope: Part C only. Part C treats the surrogate's own error, ||f_hat(theta) − f_true(theta)|| > tau, as the failure event over physical parameters theta. It uses CE, adaptive IS or BSV-style search to find failure regions and estimate P_fail with fewer true-system calls than MC, then audits whether the surrogate's ensemble or conformal uncertainty flags those regions. The true system is the nominal simulator plus a structural term of strength epsilon.

## Deviations and limits (read first)
- `fetch_article_fulltext` returned CrossRef 404 for all arXiv DOIs. Full texts were taken from arXiv HTML (arxiv.org/html/<id>) instead. This substitutes the source, not the content.
- I read full texts selectively: methods, experiments, the limitations and discussion sections, and keyword-in-context passages. I did not read every line. Appendices were checked only by keyword.
- Search budget:
  - web_search: 3 queries (the tool limit).
  - arXiv API: 18 queries, ≤25 results each, sorted by submittedDate.
  - OpenAlex: 10 queries, ≤15 results each, from 2023 onward.
  - Total: about 220 records screened by title. Coverage is not exhaustive, so "not found" means not found within this budget.
- The following were checked on **abstract only** and are marked so below: Qin et al. 2023 (ACM TCPS), Kotz & Åkesson 2605.07541, Renganathan & Booth 2603.20959, DreamTest 2610.04494, Biagiola & Tonella 2023, Shikhman 2601.11428, Bastos & O'Hagan 2009, and Roy et al. 2603.22525 (abstract plus the citing passage in 2610.07529).

## Assigned papers

### 1. Moss, Kochenderfer, Gariel, Dubois — "Bayesian Safety Validation for Failure Probability Estimation of Black-Box Systems", arXiv:2305.02449 (v2 Jun 2024; AIAA Aviation 2023; J. Aerospace Inf. Sys. DOI 10.2514/1.I011395)
- Authors: the HTML rendering dropped the first author. The arXiv API metadata gives Robert J. Moss, Mykel J. Kochenderfer, Maxime Gariel (+ Arthur Dubois in HTML).
- **What it does.** A GP surrogate is fit to a *Boolean* failure indicator of a black-box system. Three acquisition functions (FSAR) drive the search: uncertainty exploration, boundary refinement, and sampling of the failure region. P_fail comes from self-normalized importance sampling. Baselines are MC and PMC. Test systems are toy problems, a stochastic decision-making system, and a CNN runway detector searched over a 2-D parametric simulator space (glide slope, distance). In the runway case, failure is a misdetection (false negative). Open source: BayesianSafetyValidation.jl.
- **Verdict: PARTIAL (method only).** This is exactly the machinery Part C would use: BO-based failure search plus an IS failure-probability estimate over a low-dimensional parameter space, with "failure" defined as an ML model being wrong. It is not applied to a physics surrogate's regression error, and there is no model-form mismatch and no comparison with the model's own uncertainty.
- Evidence phrase: "frames the problem of black-box safety validation as a Bayesian optimization problem".
- **Delta to claim.** Part C does not contribute a new falsification algorithm. It applies BSV/CE/AIS to a new failure event (surrogate regression error under structural simulator mismatch) and asks a new question: does the surrogate's own UQ predict the falsifier's findings? **BSV must be a baseline**, and its code is open.

### 2. Delecki, Corso, Kochenderfer — "Model-based Validation as Probabilistic Inference", arXiv:2305.09930 (May 2023; L4DC 2023)
- **What it does.** Estimates the distribution over failure *trajectories* of sequential systems as Bayesian inference over disturbance trajectories. It uses HMC with automatic-differentiation gradients through the simulator and gradient smoothing. Test cases: an inverted pendulum with an NN controller, an AV scenario, and a POMDP lunar lander. The system under test is a controller; the dynamics are known and differentiable. Future work mentions surrogate models (citing Qin et al. 2022).
- **Verdict: NO OVERLAP** with A/B. **Weak PARTIAL** with C (it samples a failure distribution, which is the methodology family, not the object). The failure is a safety-spec violation of a closed-loop system, not surrogate error.
- Evidence phrase: "frame estimating the distribution over failure trajectories for sequential systems as Bayesian inference".
- **Delta.** Part C's failure event is defined by surrogate-vs-truth discrepancy over static physical parameters. HMC on a smoothed failure indicator is a possible extra method if f_hat and the true simulator are differentiable (both are in the pendulum/Duffing setting). Optional, and not needed for the claim.

### 3. Pickering, Guth, Karniadakis, Sapsis — "Discovering and forecasting extreme events via active learning in neural operators", arXiv:2204.02488 (v2 Sep 2022; Nature Computational Science 2022)
- **What it does.** A DeepONet ensemble surrogate is trained with uncertainty sampling and likelihood-weighted uncertainty sampling (US-LW), which multiplies ensemble variance by a "danger score" p_x/p_y. The goal is to learn the tails of the *system's* output PDF (SIR pandemic, MMT rogue waves, ship fatigue). The error metric is log-PDF error of the system's QoI.
- **Verdict: PARTIAL for Part A** (ensemble-variance and output-weighted active learning with deep ensembles), **NO OVERLAP for Part C**. The rare events are extremes of the physical system, not regions where the surrogate is wrong, and there is no model-form mismatch.
- Evidence phrase: "both highly uncertain and high magnitude regions are sampled".
- **Delta.** In Part C the "extreme" is surrogate error. Note for Part A: US-LW is a natural additional acquisition baseline, and reviewers in this community will know it.

### 4. Gao & Karniadakis — "Proposal-Guided Greedy Surrogate Refinement for PDE-Driven High-Dimensional Rare-Event Estimation", arXiv:2605.15356 (May 2026, math.NA)
- **What it does.** An encoder+NN surrogate of the limit-state function is refined locally along an improved cross-entropy (ICE-vMFNM) adaptive IS proposal. A greedy latent-space selection balances closeness to the estimated failure boundary against diversity. The paper gives one-step proposal-stability bounds and a "surrogate-induced misclassification" error decomposition (Lemma 4.2, Theorem 2). Tests run up to 100-D (diffusion, semilinear heat, heat conduction). Baselines are CMC, true-model ICE, and random refinement.
- **Verdict: NO OVERLAP in object (reverse direction)**, PARTIAL in machinery. This is classic surrogate-assisted reliability: the surrogate estimates the *system's* P_fail, and surrogate error matters only where it flips sign(g). There is no structural mismatch; the high-fidelity model is the truth.
- Evidence phrase: "a misclassification can occur only when the surrogate error changes the sign of g".
- **Delta.** Part C's failure event *is* the surrogate error itself, and truth differs structurally from the training simulator, so refining on nominal data cannot remove the discrepancy. Their bound linking estimator error to local surrogate error under the proposal is a useful related-work citation. Their ICE baseline is the CE baseline Part C should implement.

## Fresh-search findings (2025–2026 emphasis)

### Closest new hit: Roy, Yoo, Chakraborty, Alam — "Targeted search shows that random-device testing underestimates worst-case error in a simulated wave-based neural operator", arXiv:2610.07529 (5 Oct 2026, cs.LG). FULL TEXT read (abstract, intro, 4.6 search protocol, discussion and limitations).
- **What it does.** A hybrid FNO runs its spectral layers on simulated optical 4f hardware with 32 toleranced fabrication knobs. For 120 models, it compares the worst of 200/1000 MC (and Sobol, budget-matched) random devices against devices found by search (DE, CMA-ES, PGD, coordinate ascent). Searched devices reached 1.06–3.10× the MC maximum. The paper reports a Clopper–Pearson bound on the probability of drawing such a device, Wilks 95/95 limits, and a negative feasibility study of a Taylor-model certificate. It explicitly separates "random testing estimates how often errors exceed a threshold" from "search lower-bounds how large they can be." It does **not** use IS, CE or BSV to estimate the exceedance probability. It runs no ensemble or conformal UQ check (0 occurrences of "ensemble"). There is no model-form mismatch between training simulator and truth. The searched variables are *hardware tolerances of the surrogate's own implementation*, not physical parameters of the modeled system.
- **Verdict: PARTIAL for Part C.** It establishes "targeted search finds surrogate-error exceedances that random testing misses" for a neural-operator surrogate.
- Evidence phrase: "worst-device search gives a lower bound on how large they can be".
- **Delta.** Part C differs in four ways:
  1. The search space is physical-parameter space theta with an operational density.
  2. It estimates P(err > tau) with rare-event estimators (CE/AIS/BSV) plus CIs, against MC and Sobol at matched true-system call budgets. Roy et al. explicitly leave this estimation step to plain MC.
  3. The error source is structural simulator mismatch of controlled strength epsilon.
  4. It audits whether ensemble or conformal UQ flags the failure regions.
- Cite this paper prominently and adopt its reporting conventions: budget-matched Sobol baseline, worst-case lower bound *and* frequency estimate.

### Adversarial robustness of neural operators (input-function attacks). PARTIAL at most, different object.
- **Roy, Kobayashi, Chakraborty, Rizwan-uddin**, arXiv:2603.22525 (Mar 2026; abstract plus citing passage). Gradient-free DE finds sparse (<1%), physically plausible boundary-condition perturbations that raise FNO/DeepONet error from about 1.5% to 37–63%. Attacks pass z-score anomaly detection. *Piece*: search for inputs where the surrogate's error vs. truth is large, plus "undetected by standard checks." *Not*: a physical-parameter distribution, P_fail estimation, model-form mismatch, or ensemble/conformal UQ.
- **Roy, Chakraborty, Alam**, arXiv:2604.13316 (Apr 2026, FULL TEXT keyword-checked). Active learning by DE attacks: new training data is generated where attacks succeed, combined with an input-denoising bottleneck, on viscous Burgers. *Piece*: surrogate-failure search feeding data acquisition (a link between C and A). Uses no UQ ("ensemble" 0×), no failure probability.
- **Sun**, arXiv:2510.18989 (v4 Oct 2026, FULL TEXT keyword-checked). "Solver-integrated" adversarial attack: maximizes model-solver error under norm-bounded perturbations of input *functions* (ICs, coefficient fields), with the solver recomputing truth at the perturbed input. Used for adversarial training on Burgers/Darcy. *Piece*: worst-case search on the true surrogate error ||f_hat(x) − f_true(x)||. This is the same error functional as Part C, but local (norm ball), with no probability measure, no rare-event estimate, no mismatch, and no UQ.
- **Huang, Ma et al. (StablePDENet)**, arXiv:2601.06472 (v2 Aug 2026). Physics-informed PGD adversary on input perturbations plus a residual-sensitivity regularizer. It is a training method, so NO OVERLAP with C beyond motivation.
- **How the project differs from all four.** They attack *input functions* inside norm balls, ask "how bad can it get locally," and use the result for robustness or training. Part C searches a *probability-weighted physical-parameter space*, estimates a *failure probability with CIs*, attributes failures to *structural discrepancy vs. approximation error*, and checks *UQ calibration on the found failures*.

### PINN-specific
- **Shi, Li, Guo, Wu (WbAR)**, arXiv:2310.11789: white-box adversarial search for high-residual collocation points during PINN training. This is adaptive collocation, NO OVERLAP with C.
- **McShannon & Dietrich, "Silent Failures in PINNs: Parameter Poisoning…"**, arXiv:2606.25151 (Jun 2026, FULL TEXT keyword-checked). PDE parameters are misspecified before training; models reach low loss but are wrong (up to 128% under adversarial search). A same-parameter ensemble fails to detect this: "ensemble std 0.470 vs error 0.175", meaning members disagree more with each other than they deviate from the clean baseline. *Relevant to Part B, not C*. It is parametric (not structural) misspecification in PINNs, and it shows ensembles do not detect misspecified physics. That is supporting evidence for Part C's predicted finding, so cite it.

### Surrogate error estimation (Part B adjacent)
- **Nguyen, Serino et al. (LANL), GeoQ**, arXiv:2608.21652 (Aug 2026, FULL TEXT keyword-checked). Conditional upper-quantile error estimator for scientific surrogates built from representation-space geometry and support density, with comparisons against ensembles, conformal and GPs. It has no failure search, no rare-event estimation and no model-form mismatch ("misspecif" 0×, "failure" 0×). *Piece for B/C*: an alternative self-reported error signal. GeoQ-like or kNN-support scores could be a third "would UQ have flagged it" detector besides ensemble and conformal (optional).

### Reliability analysis with surrogates (the reverse direction). Explained, NO OVERLAP in object.
- AK-MCS family; **Jaber, Chabridon et al., AK-MCS-C²**, arXiv:2606.20191 (Jun 2026, FULL TEXT keyword-checked): Kriging plus cross-conformal (J+GP) intervals as the learning function and certified two-sided P_fail bounds. **Bousquet**, arXiv:2403.17505 (2024 review of conservative rare-event probabilities with surrogates). **Renganathan & Booth**, arXiv:2603.20959; Booth & Renganathan arXiv:2410.04496 (abstract only). **Gao & Karniadakis** 2605.15356 (above). **Li & Xiu** hybrid method and Neural Operator Hybrid (arXiv:2304.11894, from web search).
- **How Part C differs.** In reliability analysis the *system* g(x) defines failure, and the surrogate is a cheap stand-in whose errors are a nuisance controlled near the limit state. The surrogate is trained on the true model, so it has no structural mismatch. In Part C the failure event is defined *by* the surrogate's deviation from truth: g(theta) = tau − ||f_hat(theta) − f_true(theta)||. The surrogate is the system under test, and truth contains physics absent from its training simulator. Machinery transfers directly: AK-MCS, ICE, BSV and conformal certification can all be pointed at g. The question being asked is different, and these methods become baselines.

### Statistical verification and CPS falsification
- **Qin, Xia, Zutshi, Fan, Deshmukh**, "Statistical Verification using Surrogate Models and Conformal Inference and a Comparison with Risk-Aware Verification", ACM TCPS 2023 (DOI 10.1145/3635160), *abstract only*. It uses a learned surrogate of a CPS with conformal bounds on surrogate error to verify the *system*. Again the reverse direction, but it is the closest precedent for "conformal bound on surrogate error over a parameter space." Cite it.
- **Kotz & Åkesson**, "Learning Neural Hybrid Surrogates for Gradient-Based Falsification", arXiv:2605.07541 (abstract only). Surrogates are used *for* falsifying CPS requirements, the reverse direction. ARCH-COMP falsification reports (2023–25) concern STL falsification of CPS models and have no surrogate-error target.
- **Model-based RL / learned dynamics.** No hit applied rare-event search to the error of a learned dynamics model. DreamTest (arXiv:2610.04494, abstract only) and Biagiola & Tonella 2023 (abstract only) use learned or world-model surrogates *to test RL agents*, the reverse direction.

### Emulator validation (computer-experiments / UQ)
- **Bastos & O'Hagan 2009** (Technometrics 51:425–438, abstract only) provide GP emulator diagnostics on held-out validation runs: Mahalanobis distance, pivoted Cholesky errors. Marrel & Iooss 2024 (RESS review of GP estimation and validation) is title-level only. I found **no** paper that adaptively *designs* validation runs to hunt for regions of large emulator error with a rare-event estimator. Within this budget, the "sequential validation design" literature I found is about building emulators (sequential *training* design), not validating them.

## Overall verdict for Part C
**PARTIALLY TAKEN. "Empty" is overturned at the level of pieces; the full combination is still open.** Confidence about 0.7. I am not more confident because of the search budget and because several hits were checked on abstract only.
- Taken pieces:
  - Targeted search beats random testing at finding large surrogate errors (Roy et al. 2610.07529; adversarial neural-operator papers).
  - BO/IS failure-probability estimation where failure is "an ML model is wrong" over a low-dimensional simulator-parameter space (BSV 2305.02449, runway detector).
  - Conformal or surrogate-error bounds inside reliability estimation (AK-MCS-C², Qin et al.).
  - Ensembles failing to detect misspecified physics (2606.25151, PINN parametric case).
- Not found anywhere: rare-event *estimation* of P(||f_hat − f_true|| > tau) over physical parameters for a physics surrogate whose truth contains a **structural** term absent from the training simulator, swept over epsilon, together with an audit of whether ensemble or conformal UQ flags the falsifier-found regions.
- **Single closest paper:** Roy et al. 2026, arXiv:2610.07529. **Delta:** physical-parameter space with an operational density, not hardware tolerances; CE/AIS/BSV P_fail estimation with CIs at matched call budgets, not worst-case lower bounds plus MC frequency; structural mismatch of controlled strength epsilon as the error source; UQ-flagging audit.

## Design recommendations that follow from the reading
1. **Baselines reviewers will expect:**
   - crude MC with Clopper–Pearson CIs;
   - budget-matched Sobol/LHS;
   - CE/ICE adaptive IS;
   - BSV (open source, Julia; a Python reimplementation of FSAR + SNIS is small);
   - AK-MCS-style GP active learning on g;
   - a **"self-UQ-guided" search** that ranks theta by ensemble std or conformal width and evaluates truth only there. This last one makes Part B's question operational: if UQ-guided search finds the failures as cheaply as CE, the surrogate's UQ suffices; if not, that is the result.
2. **Report both quantities**, following Roy et al.: P_fail estimate with CI vs. true-call budget, and the worst-found error as a lower bound. Use the relative error of P_fail against a large-MC reference, as in BSV and Gao & Karniadakis.
3. **Decompose failures.** Surrogate error = approximation error (reducible with more nominal data) + structural discrepancy δ_eps(theta) (irreducible from nominal data). Estimate each separately: compare f_hat against the nominal simulator and against truth. The predicted, testable claim is that ensemble or conformal UQ calibrated on nominal data tracks the first term but is blind to the second. Supporting evidence for this mechanism is in 2606.25151's ensemble result. This decomposition is the paper's distinctive contribution, so make it a figure.
4. **Choose tau carefully.** At large epsilon the discrepancy region may stop being rare. Fix tau per epsilon so that P_fail is roughly 1e-2 to 1e-4 under the operational density, or report P_fail(tau) curves. Otherwise CE vs MC comparisons become trivial or meaningless.
5. **Accounting.** Every evaluation of the failure indicator costs one *true-system* call (f_hat is free). State this cost model explicitly; the reliability literature counts high-fidelity calls the same way.
6. **Related-work framing:** "reverse direction" paragraph (AK-MCS, Gao & Karniadakis, AK-MCS-C², Bousquet, Qin et al.); "adversarial neural operators" paragraph (Roy ×2, Sun, StablePDENet); "safety validation" paragraph (BSV, Delecki, Corso et al. survey); and Pickering et al. for ensemble-based extreme-event active learning (Part A link).

## Bibliographic verification status
Verified from arXiv full-text HTML headers or the arXiv API: 2305.02449, 2305.09930, 2204.02488, 2605.15356, 2610.07529, 2510.18989, 2604.13316, 2601.06472, 2606.25151, 2608.21652, 2403.17505, 2606.20191, 2310.11789.
Verified from arXiv API or OpenAlex metadata only (abstract): 2603.22525, 2605.07541, 2603.20959, 2410.04496, 2610.04494, 2601.11428, 10.1145/3635160 (Qin et al.), 10.1145/3631970 (Biagiola & Tonella), 10.1198/TECH.2009.08019 (Bastos & O'Hagan).

# Novelty improvement: research report

*Generated: 2026-11-14 | Sources: 14 (4 read in depth, 10 by abstract or search summary) | Confidence: Medium*

Tools: the firecrawl and exa tools the skill expects are not installed, so this used a paper index (two searches, four deep reads) and two web searches. Deep reads return the most relevant pages of each paper, not always every page.

## Executive summary

Three findings change what the paper can claim. First, a 2025 theory paper frames "can anything predict this model's loss?" as a question about what information the loss predictor has; that language fits this project exactly and no one has applied it to a simulator with missing physics. Second, a 2026 benchmark paper on multi-fidelity neural operators states that almost every existing test case differs only by grid resolution and asks for benchmarks whose fidelity gap comes from the governing equations, with uncertainty studied on top; the planned operator system with a missing-term dial is that benchmark. Third, a development-seed test run for this report shows that a simulator-only signal built by perturbing the simulator's physics tracks the missing-term error almost perfectly when the hypothesised term is of the right family (Spearman 0.97-1.00) and can be anti-aligned when it is not (down to -0.68). That makes the paper's negative result constructive: the simulator fully determines how sensitive each prediction is to any hypothesised missing term; what it cannot supply is which term is missing.

## 1. Loss prediction theory gives the result a home

- Gollakota et al. show that a loss predictor can beat a model's own loss estimate only by exposing a multicalibration failure, and that extra information the model's trainer did not have is what makes a real advantage possible ([When does a predictor know its own loss?](https://www.alphaxiv.org/abs/2502.20375), pp. 3-6). The paper covers binary classification with proper losses; regression and a misspecified data source are not treated.
- Reading for this project: a simulator-only uncertainty signal is a loss predictor restricted to simulator-computable information; real measurements are the external information. The alignment law (ranking converges to the correlation with the discrepancy norm) is the regression, ranking-metric counterpart. This is an inference from their framework, not a result they state.
- Related evidence: ensemble disagreement decouples from accumulated model error in latent rollouts ([Biased Dreams](https://www.alphaxiv.org/abs/2604.25416), read in depth, pp. 1, 8-9, which also calls a formal characterisation future work). Abstracts only: variance that is calibrated but does not localise error in CT reconstruction ([2607.13682](https://www.alphaxiv.org/abs/2607.13682)); a call to evaluate epistemic uncertainty beyond OOD detection and active learning ([2607.14817](https://www.alphaxiv.org/abs/2607.14817)).

## 2. The operator community is asking for this benchmark

- Faza et al. benchmark multi-fidelity neural operators and observe that existing test cases get their fidelity gap from grid resolution only; they build one "governing-equation-like" case by truncating a POD basis, find that most strategies stop beating a high-fidelity-only baseline as the gap grows, and list as future work benchmark cases with differences in the governing equations and uncertainty quantification across fidelities ([2608.04708](https://www.alphaxiv.org/abs/2608.04708), pp. 2-3, 13, 17-18).
- A Bayesian multi-fidelity DeepONet with active learning uses a frozen low-fidelity operator plus a residual learner on experimental data ([2602.13257](https://arxiv.org/abs/2602.13257), search summary only, unverified).
- Reading: Burgers with a missing term at a controlled strength, with a neural-operator surrogate, is a governing-equation fidelity gap with a dial. Releasing the simulator pairs and the dial as a small open benchmark is a contribution reviewers of an applied journal value.

## 3. Experimental design for missing physics exists, but not this question

- Plate, Martensen and Sager formulate optimal experimental design for universal differential equations with Fisher-information criteria and dimension reduction, and compare against equidistant sampling and uncontrolled experiments ([2408.07143](https://www.alphaxiv.org/abs/2408.07143), pp. 1, 3-4, 14). Strouwen and Micluța-Câmpeanu choose experiments to discriminate candidate structures ([2604.01231](https://www.alphaxiv.org/abs/2604.01231), read on 2026-11-14).
- Neither varies how large the missing term is, uses a space-filling baseline over initial conditions, or involves a neural surrogate of the simulator. So "when does designed experimentation for missing physics beat a space-filling design, as mismatch grows" is open, but the area is active and both must be cited as the nearest neighbours.

## 4. Validation with few real samples is a neighbouring line

- Sim2Val uses simulator metrics as a control variate for real-world estimates, and SureSim frames combining real and simulated evaluations as prediction-powered inference, reporting that the benefit vanishes when the gap is too large ([Sim2Val](https://arxiv.org/abs/2506.20553), [SureSim](https://arxiv.org/abs/2510.04354); search summaries only, unverified). A March 2026 paper gives closed-form sample-size formulas for prediction-powered estimators ([2603.16041](https://arxiv.org/html/2603.16041), search summary only).
- These estimate average performance. None tests whether an uncertainty signal's ranking can be trusted. The planned "how many real points to know which case you are in" test is distinct, and should cite these and borrow the sample-size framing.

## 5. New test: perturb the physics, not the weights

Run for this report on development seed 999 (`scripts/dev_structural_signal.py`; no pilot or main seed, no network training). For each hypothesised missing term, the nominal ODE is perturbed by a tiny multiple of that term and each test point is scored by how far its trajectory moves. Spearman correlation of that score with the true discrepancy norm:

| Hypothesised term | True system S1 (quadratic drag) | True system S2 (Coulomb friction) |
|---|---|---|
| ω·\|ω\| | +1.00 | −0.42 |
| ω³ | +0.98 | −0.51 |
| tanh(ω/0.1) | −0.34 | +0.99 |
| tanh(ω/0.5) | −0.21 | +0.97 |
| ω | +0.78 | +0.16 |
| φ³ | +0.84 | −0.51 |
| sin 2φ | −0.68 | +0.63 |
| φ·ω | +0.65 | −0.65 |
| φ | +0.32 | +0.35 |
| Random mix of all nine | +0.17 | +0.73 |
| Random mix without the true family | −0.19 | +0.07 |

For reference, nominal ensemble σ reached about +0.78 on S1 and about 0 on S2 in the earlier dev-seed check (Pearson, so not directly comparable).

What this shows, with one seed and one run:

1. For small mismatch the discrepancy is the simulator's sensitivity applied to the missing term. The sensitivity is computable from the simulator alone; the term is the only unknown.
2. A physics-perturbation signal is near-perfect with the right hypothesis and can be worse than useless with the wrong one. A generic mix does not rescue it. This is the non-identifiability result in constructive form.
3. It suggests two things the paper could add: (a) an **alignment range**, the best and worst correlation of any given signal over a dictionary of plausible missing terms, computed with no real data, as a stress test for an uncertainty signal; (b) using a few real trajectories to pick the hypothesis, which needs far less data than learning the term from scratch. Neither has been tested beyond this table.

Related ideas exist elsewhere and must be cited: perturbed-physics ensembles in climate modelling, stochastic-operator model-form uncertainty (Soize and co-workers), and basis perturbations for reduced-order models ([2608.03360](https://www.alphaxiv.org/abs/2608.03360)). I did not find them applied to ranking the true-system error of a neural surrogate; that search was not exhaustive.

## Key takeaways

- **Add the alignment range (stress test).** No training, uses code that exists, works on the pendulum now and on Burgers by perturbing the PDE right-hand side. My estimate is that this is the largest novelty gain available inside the 10-day window, because it turns "you cannot know" into "here is how bad it can get, computed before any experiment".
- **Frame the theory in loss-prediction terms** and state the linear-response identity (discrepancy = sensitivity operator applied to the missing term) as a proposition. Writing only.
- **Package the systems as a benchmark** with the mismatch dial, citing the multi-fidelity operator paper's call.
- **Keep acquisition on the dynamics correction** and cite the two experimental-design papers as nearest neighbours; the open question is the dependence on mismatch strength.
- **Do not claim** that perturbing physics is new in general, or anything about the four conformal-operator papers, which are still unread.

## Gaps in this research

- No search of theory venues (COLT, ALT) for an existing ranking bound under misspecification.
- Sim2Val, SureSim, the PPI power paper and the Bayesian multi-fidelity DeepONet were seen only as search summaries.
- The perturbed-physics literature in climate and structural dynamics was recalled, not searched.

## Sources

1. [When does a predictor know its own loss? (2502.20375)](https://www.alphaxiv.org/abs/2502.20375) — loss prediction equals multicalibration auditing; read in depth.
2. [Optimal Experimental Design for Universal Differential Equations (2408.07143)](https://www.alphaxiv.org/abs/2408.07143) — Fisher-information design for hybrid models; read in depth.
3. [Biased Dreams (2604.25416)](https://www.alphaxiv.org/abs/2604.25416) — ensemble disagreement decouples from accumulated error; read in depth.
4. [On Multifidelity Neural Operator for Data-Driven PDE Predictions (2608.04708)](https://www.alphaxiv.org/abs/2608.04708) — benchmark and call for governing-equation gaps; read in depth.
5. [Experimental Design for Missing Physics (2604.01231)](https://www.alphaxiv.org/abs/2604.01231) — structure discrimination for missing physics; read 2026-11-14.
6. [Conformal risk control for model-form uncertainty (2608.03360)](https://www.alphaxiv.org/abs/2608.03360) — basis perturbation for reduced models; read 2026-11-14.
7. [When Variance Is Not an Error Map (2607.13682)](https://www.alphaxiv.org/abs/2607.13682) — abstract only.
8. [Evaluating Epistemic Uncertainty: Beyond OOD Detection and Active Learning (2607.14817)](https://www.alphaxiv.org/abs/2607.14817) — abstract only.
9. [Epistemic Uncertainty Is Not the Reducible Kind (2606.12646)](https://www.alphaxiv.org/abs/2606.12646) — abstract only.
10. [Sim2Val (2506.20553)](https://arxiv.org/abs/2506.20553) — search summary only.
11. [SureSim (2510.04354)](https://arxiv.org/abs/2510.04354) — search summary only.
12. [Power analysis for prediction-powered inference (2603.16041)](https://arxiv.org/html/2603.16041) — search summary only.
13. [Bayesian multi-fidelity neural operator (2602.13257)](https://arxiv.org/abs/2602.13257) — search summary only.
14. [Active learning with Bayesian multi-fidelity Laplace neural operators (2502.00550)](https://www.arxiv.org/pdf/2502.00550) — search result only.

## Methodology

Two paper-index searches and two web searches, four papers read in depth, one new numerical test. Sub-questions: (1) is there theory on when uncertainty ranks error under misspecification; (2) what exists on experimental design for hybrid models with a learned missing term; (3) how many real samples validate a simulator-trained model; (4) what do multi-fidelity operator benchmarks lack; (5) can a simulator-only signal target the discrepancy.

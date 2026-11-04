# Mismatch study: where novelty is weakest, and reframings from neighbouring fields

Research date 2026-11-04. Read in full (main text plus the appendix pages the tool returned): 2608.13633, 2505.23506, 2606.22062, 2609.38526, 2210.10760, 2405.08719, 2406.03154, 2506.07805, 2609.01418. Seen as titles and abstracts only, in search listings: 2503.12315, 2509.05724, 2509.23385, 2606.16923, 2502.04949, 2507.13495, 2409.02628, 2610.19832. Papers known only because a paper I read cites them (not read; no IDs checked unless given): Sugiyama 2005, Bach 2006, Amortila et al. 2024, Sloman et al. 2022, Ward et al. 2022, Cannon et al. 2022 (2209.01845), Wagenmaker et al. NeurIPS 2024, Qu et al. AISTATS 2025, AdaptSim (Ren et al. CoRL 2023), Kirchhof et al. 2025.

## Q1. Strongest reviewer objections (blunt list, with the evidence a reviewer would cite)

### Takeaway
The weakest point is not the toy system. It is that H1, and much of H2, follow from the design and can be computed without the expensive grid. The nominal ensemble σ(θ) and the nominal residual are fixed functions of θ that do not depend on ε. So the "dose-response in r" is a mixing curve between corr(σ, |a|) and corr(σ, |d|). Mainstream UQ, SBI, BOED and hybrid-RL papers from 2024-2026 already state the general claim ("epistemic estimates miss bias"; "active learning on a misspecified model amplifies error") and in some cases give bounds for it.

### Cited Findings
**O1. "The central claim is textbook, and in general form it is already a published position paper."**
- Jiménez, Jürgens, Waegeman (2505.23506, ICML-style position paper, v4 Feb 2026) argue that approximation error and estimation bias "are both not included in existing epistemic uncertainty estimators", so a complete representation of epistemic uncertainty is "fundamentally unattainable". They show that under systematic bias, deep ensembles at once overestimate aleatoric and underestimate epistemic uncertainty. Their appendix shows epistemic estimates shrink with N even where bias stays large, and their Table 1 reports bias "frequently larger than –and largely uncorrelated with–" the epistemic estimates, across 8 methods (bootstrap, GP, EDL, Laplace, MC dropout, VI, etc.). — [2505.23506](https://arxiv.org/abs/2505.23506)
- The same paper: deep ensembles capture "procedural uncertainty only"; the data-resampling component "is entirely absent". Their recommendations: assess model bias "separately through held-out validation", and treat epistemic estimates "as lower bounds". — [2505.23506](https://arxiv.org/abs/2505.23506)
- The PHYSTAT/VERaiPHY community report says models trained on simulations "inherit their misspecifications", and that unknown unknowns "cannot be captured by standard uncertainty propagation and may introduce biases invisible to conventional diagnostics". It lists "structural" misspecification (missing terms, wrong functional relationships) as its own category. — [2608.13633](https://arxiv.org/abs/2608.13633)
- Already in the project notes: 2610.35560 (Appendix B) states ensemble blindness to shared bias as a standard fact.

**O2. "H1 is true by construction and can be predicted without training."** (Inference from the project's own design, checked against prereg sections 5-6.) Nominal σ, MC dropout and the nominal residual are trained only on nominal data, so they are the same function of θ in every ε cell. The total error is e = a + d with d growing in ε, so AUROC(σ, ‖e‖) moves from its ε = 0 value toward AUROC(σ, ‖d‖) as r grows. The endpoint is fixed by the geometry of each system: S1's drag sits at high velocity, S2's friction at low amplitude. A reviewer can say B1 tests arithmetic. 2505.23506 already shows the mechanism for deep ensembles in general.

**O3. "Prior evidence says H2 will be null."** Koermer et al. (2301.10228, Supplement C.3, already in project notes) varied bias at 3 levels and found "the relative pecking order of those methods is the same". Romor et al. (2610.27130, project notes) found that residual and worst-error rules hurt with no mismatch at all, and no rule beat random in warm-up. Tang, Sloman, Kaski (2506.07805, AISTATS 2026) report generalization error as a ratio to random design across misspecification degrees 0/1/2 (polynomial DGP, Appendix E.1.3-E.1.4, Fig. 10). That is another small "degree of misspecification" sweep against a random baseline. — [2506.07805](https://arxiv.org/abs/2506.07805)

**O4. "Active learning under misspecification already has theory; your H2 is a corollary."**
- 2506.07805 splits generalization error into misspecification bias (B), estimation bias (C) and a new error (de-)amplification term (A). Building on Amortila et al. (2024), it bounds the error by the covariate shift C∞ = sup d_test/d_train times the misspecification B∞ = ‖f̄ − f*‖∞. It says active learning uses the model "twice: once to make inferences to fit training data, and again to select the data", so misspecification has a "double impact". It cites Sugiyama 2005, Bach 2006, Ali et al. 2014, Vincent & Rainforth 2017 and Farquhar et al. 2021 for "model misspecification can produce poor quality datasets". — [2506.07805](https://arxiv.org/abs/2506.07805)
- Their remedy (R-I, R-IDeA) multiplies EIG by an MMD representativeness ratio and a de-amplification sigmoid. It beats BAD and random on polynomial, source-location and pharmacokinetic tasks. — [2506.07805](https://arxiv.org/abs/2506.07805)
- Caveat a reviewer may miss and the author should state: in 2506.07805 the *learner* is misspecified. In this project the correction learner (δ-net) is flexible, and only the *acquisition score* comes from the misspecified nominal model. That is a real difference, but it has to be argued; it is not obvious.

**O5. "Targeted real-data collection where the simulator is wrong is solved, with guarantees, in sim-to-real RL."**
- Ni & Kamgarpour (2609.01418) estimate a "mismatch region" B from real transitions, use simulator transitions elsewhere, and collect real data only where needed. Real-sample complexity scales with |B| and with the separation gap σ_s − ε_s. Empirically, iterations grow "approximately linearly with |B|". At small separation the method falls back to the no-simulator baseline. They cite a bandit result (Cheung & Lyu): with no prior information about the mismatch, no hybrid method is guaranteed to beat a purely online one. — [2609.01418](https://arxiv.org/abs/2609.01418)
- The same paper cites Wagenmaker et al. (NeurIPS 2024) and Qu et al. (AISTATS 2025) as sim-guided real-world exploration that corrects mismatch. Not read. — [2609.01418](https://arxiv.org/abs/2609.01418)
- A reviewer would map arm 7 (observed residual) onto "estimate the mismatch region and sample there".

**O6. "Pendulum plus missing friction is the standard toy; the setting is too narrow."**
- RoPE (Wehenkel et al., ICML 2025) uses a frictionless-pendulum simulator against a "real" damped pendulum with friction (Task C). — [2405.08719](https://arxiv.org/abs/2405.08719)
- Rizvi & Tomar (2606.22062, IROS 2026 Sim2Real workshop) run sim-to-sim pendulum budget allocation with 10 seeds and paired bootstrap, the same statistical style as this project. — [2606.22062](https://arxiv.org/abs/2606.22062)
- One 4-D ODE family and two missing terms will draw "does it generalise beyond a 4-parameter ODE?"

**O7. "Continuously widening simulation gaps are already studied in SBI."** Schmitt et al. (2406.03154) implement "continuously widening simulation gaps, as opposed to a single discrete misspecification". Posterior error rises with misspecification severity, and their MMD detector score tracks it. They also find a trade-off: overcomplete summaries detect more misspecification but give larger posterior error. — [2406.03154](https://arxiv.org/abs/2406.03154)

**O8. "r is not observable, so the result can't be acted on."** (Inference.) r = median‖d‖/median‖a‖ needs f_true on a test set. A practitioner can't compute it before choosing experiments. Reviewers will ask for an observable proxy, e.g. the residual of the first 8 true trajectories, or the Schmitt-style MMD between nominal and true trajectories. Schmitt et al. show their MMD rises with gap severity and tracks posterior error, which suggests such a proxy is feasible. — [2406.03154](https://arxiv.org/abs/2406.03154)

**O9. "Terminology and framing ignore the ML vocabulary."** 2608.13633: what physicists call "systematics", statisticians call "model misspecification" and ML calls "distribution shift" are "overlapping descriptions of the same underlying problem". In ML terms the project's setting is concept shift: p(θ) is unchanged and only p(y | θ) changes. Input-space signals (ensemble spread, distance to training data) can't respond to a shift that leaves the input distribution unchanged. 2505.23506 cites Li et al. (2025): uncertainty-based OOD methods "conflate high uncertainty with being far from training data". — [2608.13633](https://arxiv.org/abs/2608.13633); [2505.23506](https://arxiv.org/abs/2505.23506)

**O10. "Missing baselines."** Likely reviewer requests: a representativeness- or misspecification-robust acquisition (R-I/R-IDeA, 2506.07805), a KOH-style design (2301.10228), and a "system-identify first" arm (refit c, L on true data). 2606.22062 finds that a few identification rollouts close most of the gap when the model family is right, and that broad randomisation "did not substitute for measurement". — [2506.07805](https://arxiv.org/abs/2506.07805); [2606.22062](https://arxiv.org/abs/2606.22062)

### Inferences
- Ranked by how much each hurts: O2 > O1 > O4 > O3 > O5 > O8 > O6 > O10 > O7 > O9.
- O2 can be turned into a strength. Pre-compute the *predicted* RQ1 curve from the ε = 0 ensemble and the simulator-computed d(θ), with no retraining, and test whether the trained-cell results match it. "Blindness is predictable from cheap quantities" is a more useful claim than "blindness exists".
- O4's caveat (flexible learner, misspecified acquisition score) is the cleanest novelty statement left. Neither 2506.07805 nor Koermer et al. separates "score from a misspecified model" from "misspecified learner".

### Gaps
- I could not reach OpenReview reviews of related submissions, so the objections are inferred from the literature, not quoted from real reviews.
- Not read: Sugiyama 2005, Bach 2006, Amortila et al. 2024 (the misspecification-amplification bound), Sloman et al. 2022, Ward et al. 2022, Cannon et al. 2022, Kirchhof et al. 2025. All known only through papers that cite them. Amortila et al. is the one to read before submission.
- I found no paper on concept-shift detection by input-space uncertainty specifically; O9 is partly inference.

## Q2. Uncertainty under misspecification in mainstream ML: who already states the central claim in general form?

### Takeaway
2505.23506 states it in general form for supervised regression and classification. 2608.13633 states it for ML in physics. Neither measures *how fast* ranking ability decays with a bias/variance ratio, or what this does to acquisition. So the project's measured dose-response and its acquisition link remain open, but the bare claim is taken.

### Cited Findings
- Bias-variance decomposition for proper scoring rules: aleatoric + generalized bias + generalized variance. The bias term is "as central to epistemic uncertainty as the variance component". — [2505.23506](https://arxiv.org/abs/2505.23506)
- Their call to action asks for evaluation "against reference distributions that decompose variance into different sources", and for claims to "specify which components are captured". — [2505.23506](https://arxiv.org/abs/2505.23506)
- Other critiques they cite: second-order/evidential identifiability problems (Bengs et al. 2022; Jürgens et al. 2024, 2402.09056), epistemic uncertainty collapse as models grow (Fellaji & Pennerath 2024, 2409.02628, title and abstract only), and correlated aleatoric/epistemic estimates (Mucsányi et al. 2024). — [2505.23506](https://arxiv.org/abs/2505.23506)
- They say credal sets and conformal prediction don't fix this: conformal "addresses a different problem". — [2505.23506](https://arxiv.org/abs/2505.23506)

### Inferences
- The project already has what 2505.23506 asks for: an exact split e = a + d per test point. RQ1 can be framed as the first test of their taxonomy at a controlled bias/variance ratio, in a SciML forward-surrogate setting.

### Gaps
- Bengs/Hüllermeier credal critiques are known only through 2505.23506's citations. I did not read them.

## Q3. Sim-to-real: is the project a special case of something already solved?

### Takeaway
Partly. Tabular hybrid RL (2609.01418) solves "spend real samples where sim ≠ real" with guarantees under a separation assumption, and finds cost scales with mismatch-region *size*. The pendulum budget paper (2606.22062) studies only the well-specified case and explicitly leaves structural mismatch open. Continuous-θ neural surrogates with a structural missing term are not covered.

### Cited Findings
- 2606.22062: sweeps identification rollouts n ∈ {0, 5, 10, 25, 50} × randomisation width w ∈ {0, ..., 0.5} on a pendulum, 10 seeds. Conclusion: "measure first". Limitations: "no structural model mismatch for randomization to absorb", and "Future work should test where this ordering changes under structural model mismatch". It cites AdaptSim and Valassakis et al. as evidence that the ordering may flip under irreducible gaps. — [2606.22062](https://arxiv.org/abs/2606.22062)
- 2609.01418: assumes every transition mismatch is either ≤ ε_s or ≥ σ_s (separation). Cost term (i) scales with |B|; term (ii) with the separation gap. Experiments vary |B| ∈ {4, ..., 100} and wind strength p_wind (σ_s from 0.13 to 0.72); smaller σ_s needs more iterations. — [2609.01418](https://arxiv.org/abs/2609.01418)

### Inferences
- That hybrid-RL work has two dials, mismatch *support* (|B|) and mismatch *magnitude* (σ_s). The project has only magnitude (r). S1 and S2 differ in where the mismatch lives but not in a controlled support fraction. Adding a support dial (e.g. a localised bump in the missing term) would line the result up with this theory.

### Gaps
- Not read: Wagenmaker et al. 2024, Qu et al. 2025, AdaptSim, Valassakis 2020, or the "how much real data" literature (e.g. Shakerimov et al. 2023, cited by 2606.22062: 20-50 real episodes help on a pendulum).

## Q4. SBI misspecification: could the project be recast as an SBI question, and is that space crowded?

### Takeaway
It is crowded: at least 8 papers from 2024-2026 in one search, plus a 2026 review. The task is also different (posterior over θ from observations, not a forward surrogate over θ). One gap in SBI matches RQ2 well: no paper I read selects the real *calibration set* actively. RoPE uses i.i.d. calibration sets and names non-i.i.d. calibration as future work.

### Cited Findings
- RoPE: calibration-set sizes 10-1000, i.i.d. from p*(θ, x_o); includes a pendulum-with-friction task. Discussion: calibration data collected from "a different, potentially biased, distribution" is a challenge; disjoint support "may fail". — [2405.08719](https://arxiv.org/abs/2405.08719)
- Schmitt et al.: unsupervised MMD misspecification score; with as few as N = 5 real series, Type II error is essentially zero (COVID model). — [2406.03154](https://arxiv.org/abs/2406.03154)
- Crowding (titles and abstracts only): 2503.12315 (SBI under misspecification review), 2509.05724 (variational robust amortised SBI), 2509.23385 (flow-matching calibration), 2606.16923 (MA-SBI), 2502.04949 (domain adaptation for amortised Bayesian inference), 2507.13495 (deep-ensemble SBI calibration and misspecification detection), 2610.40770 (cosmic dawn misspecification diagnosis). — [alphaXiv search listing](https://www.alphaxiv.org/abs/2503.12315)

### Inferences
- Don't recast the paper as SBI: the change of task explodes the scope and the area is busy. Cite RoPE and Schmitt in related work. Mention "active choice of the real calibration set" as a direction the result informs.

### Gaps
- Ward et al. robust NPE and Cannon et al. not read in full.

## Q5. Reward-model / verifier misspecification: does "simulator = proxy reward" give a scaling-law result?

### Takeaway
Yes, and it is the most promising reframing, because the project already has an optimisation-pressure knob: the batch-sampling exponent k in p ∝ score^k. Gao et al. and Aouad et al. give functional forms for gold reward against optimisation pressure (KL) and proxy quality (data M). I found no SciML or active-learning paper that measures acquisition gain against acquisition concentration and proxy-gold correlation.

### Cited Findings
- Gao, Schulman, Hilton (2210.10760, ICML 2023): synthetic gold-RM setup. Gold score R_bon(d) = d(α − βd) and R_RL(d) = d(α − β log d), with d = √KL and KL_bon = log n − (n−1)/n. The coefficients scale smoothly with RM size, "not as cleanly" with data size. Regressional Goodhart: with proxy X̂ = X + Z, E[X | X̂] shrinks by Var(X)/(Var(X)+Var(Z)). α is read as regressional and β as extremal Goodhart. — [2210.10760](https://arxiv.org/abs/2210.10760)
- Aouad, El Gadarri, Farias (2609.38526): gain Θ(√min{log M, K}). The upper bound is σ√(2 min{K, log(1+2pM)}) and is tight up to √2. Empirically, √log M fits with R² of 0.95-0.99 across 0.6B-4B proxies, BTL noise, BoN and tilting. Recommendation: set K ~ log(pM). They list "adaptive preference collection" as future work. — [2609.38526](https://arxiv.org/abs/2609.38526)
- 2610.19832 (title and abstract only) recasts overoptimisation as rent-seeking where "frequent audits curb the rent", an analogue of spending true experiments to audit a proxy. — [alphaXiv listing](https://www.alphaxiv.org/abs/2610.19832)

### Inferences
- Mapping: nominal score s(θ) = proxy reward; true-error reduction from querying θ = gold reward; the acquisition distribution q_k ∝ s^k over the pool = optimised policy; KL(q_k ‖ uniform pool) = optimisation pressure (k = 0 is random; k → ∞ is greedy, like BoN with large n); r (or 1 − ρ(s, |e|)) = proxy-gold gap.
- Testable prediction: gain over random/Sobol, as a function of √KL, is an inverted U. The peak KL* falls as r rises. At large r the curve becomes monotone decreasing (pure extremal Goodhart: the score concentrates samples where d is small). This turns the yes/no H2 into a law with fitted (α(r), β(r)). It also connects to the author's RLVR gate work: a gate is a choice of KL budget against a proxy.
- The regressional-Goodhart formula predicts α ∝ ρ(s, true gain). ρ can be measured at ε = 0 and at each ε, so α(r) can be checked out of sample.

### Gaps
- No paper found that applies Goodhart scaling laws to active learning or experimental design. This is an absence of evidence after one search, not proof.

## Q6. Which framing gives the most general, surprising claim without exploding scope?

### Takeaway
Recommended: keep RQ1 and RQ2 as frozen, and add two exploratory layers. (a) "Acquisition from a simulator is optimisation against a proxy": sweep k and report gain against KL and r in a Gao/Aouad form. (b) Exact Tang-style decomposition of corrected error into misspecification, estimation and (de-)amplification terms, which the project can compute because f_true is known. Lead with the O2 strength: blindness is predictable from cheap quantities.

### Cited Findings (closest prior work per reframing, all read in full)
| # | Reframing | Closest prior work | What it did | Still open |
|---|---|---|---|---|
| R1 | Goodhart / proxy-optimisation law for acquisition | 2210.10760; 2609.38526 | Functional forms for gold reward vs KL and vs data M, in RLHF | Any acquisition or design setting; a proxy-gold gap as a continuous dial (r); adaptive collection (named as future work in 2609.38526) |
| R2 | Misspecification (de-)amplification in active learning | 2506.07805 (AISTATS 2026) | Error decomposition B + C + A; bound in C∞·B∞; R-IDeA; 3 misspecification degrees vs random on toy models | Neural surrogates; misspecified *score* with a flexible learner; exact computation of A on a dense dial; whether de-amplification explains S1-vs-S2 differences |
| R3 | Mismatch support vs magnitude (hybrid sim-to-real) | 2609.01418; 2606.22062 | Cost ∝ \|B\| and separation gap (tabular); "measure first" holds only when well-specified | Continuous θ, structural missing term; the "identify vs correct" order under structural mismatch (named open in 2606.22062) |
| R4 | Test of the epistemic-incompleteness taxonomy | 2505.23506 | Position plus 1-D demos: bias missing from epistemic estimates | A controlled bias/variance ratio sweep with exact decomposition; its downstream effect on acquisition |
| R5 | Active choice of the real calibration set in SBI | 2405.08719; 2406.03154 | i.i.d. calibration sets of 10-1000 points; MMD detection | Active selection; not recommended as the main frame (crowded, change of task) |

### Inferences
- **Most general claim available:** "When a cheap model scores where to spend expensive true experiments, the gain follows a predictable proxy-optimisation law. The law's coefficients are set by the bias/variance ratio r, and the best acquisition concentration shrinks as r grows." This holds in any field with a simulator, so it suits TMLR. It is falsifiable on the existing pendulum code.
- **Cost estimate (a guess until the timing pilot):** a k sweep for arm 3 only, on S1, at k ∈ {0, 0.5, 1, 2, 4, 8, greedy} × 5 ε × 10 seeds × 6 rounds × 5 members ≈ 10,500 extra fits, about +40% on the current ~25,500. Arm 3's score is static, so only the δ-net refits cost anything. The R2 decomposition needs no extra fits: it is post-hoc arithmetic on stored predictions, plus the best-in-class δ̄, which needs one large-n δ fit per cell.
- **Design-decision flag (per the user's CLAUDE.md):**
  - *What matters:* the paper needs one non-obvious result. Under the frozen plan, the most likely outcome is that H1 is supported but reads as obvious (O2), and that H2 is null or weak (Koermer; Romor). R1 gives a positive, structured result even if H2 is null, because an inverted U with a falling KL* is informative either way.
  - *Options:*
    - (i) Frozen plan only: no cost, weakest novelty.
    - (ii) Add the R1 k sweep on S1 plus the R2 decomposition: about +40% compute, 1-2 weeks; strongest gain.
    - (iii) Add R3, a support-fraction dial: needs a new missing-term family; adds systems and risks the January workshop.
  - *Recommendation:* (ii). Log it in prereg section 11 as an exploratory addition *before* the pilot. Run k ∈ {0, 1, 4, greedy} in the pilot first to see whether gain against KL is non-monotone at any ε. Switch to (i) if the pilot gain curve is flat in k at every ε, because then there is no law to fit.
  - *Not known yet:* between-seed variance of gain at high k (greedy batches are noisier), and whether the KL of a 4,096-candidate pool distribution is a stable pressure measure. The pilot measures both.
- R3 is the best bridge to robotics or sim-to-real workshops (2606.22062 names the open question), but it adds scope. Keep it for paper 2, alongside the deferred RQ3.

### Gaps
- Whether workshop reviewers in SciML value a Goodhart framing is a judgement call. I found no evidence either way.
- Fitted forms (d(α − βd), √log M) come from LLM settings; whether any of them fits acquisition gain is unknown until the pilot.

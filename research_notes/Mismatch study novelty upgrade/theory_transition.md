# Theory that could predict the transition r* for "Blind to the Missing Term"

Scope: theory that would let the study predict, not just measure, when simulator-only uncertainty (RQ1) and acquisition (RQ2) fail as r = median‖d‖ / median‖a‖ grows. Written 2026-11-04.

Reading depth:
- **Full text (theorems and method sections) via alphaXiv:** Tang, Sloman & Kaski 2506.07805; Sanz-Alonso & Yang 2312.09225; Sloman, Caprio & Kaski 2505.23496; Lin & Reimherr 2402.14966; Bogunovic & Krause 2111.05008; Barber, Candès, Ramdas & Tibshirani 2202.13415; Baek, Berchuck & Mukherjee 2306.03783.
- **Abstract or secondary source only:** Box & Draper 1959 (the PDF could not be text-extracted; numbers come from secondary summaries); Sugiyama 2006 JMLR; Bogunovic's citation of Lattimore et al.; Amortila et al. 2024 (known only through Tang et al.); DEUP 2102.08501; "When does a predictor know its own loss?" 2502.20375; Dette & Melas; Wiens 2512.21806.

**Key structural fact, used throughout (derived from the project's own design, not from a paper).** The nominal surrogate f̂, its ensemble σ, MC dropout, the nominal-ODE residual and the approximation error a never see true-system data. So none of them depend on ε. For S1 and S2, d(θ) = f_nom(θ) − f_true(θ; ε) can be computed exactly at every test point, because both simulators are available.

Consequence: once a, σ and g are saved on the test set, the whole RQ1 curve for the nominal-only signals is a deterministic recombination of saved arrays. For S1 and S2, ε scales the missing term, not d itself; d is nonlinear in ε, so it is computed exactly per ε level rather than as ε·g (cheap, since it needs only simulator runs).
- A "prediction" is only worth something if it comes from a few summary statistics measured at ε = 0 (or in the pilot) and then holds on held-out cells (other n_nom, other system, other seeds).
- Otherwise the reviewer reply is "you just computed it".

The same holds for any static acquisition arm (arms 3 and 4, and arm 6 with frozen GP hyperparameters) when the correction model is a linear smoother (KRR or GP).

---

## 1. What results exist on UQ and calibration under misspecification or label-function shift, where the error has a component shared by all members (bias)? Any closed-form AUROC or Spearman-of-uncertainty-vs-error results?

### Takeaway
There are decompositions and bounds that separate a misspecification-bias term from an estimation term, and they say variance-type uncertainty tracks only the latter. I found **no paper that derives a closed-form AUROC or Spearman between an uncertainty signal and the true error as a function of bias size**. The closest are coverage-gap bounds (conformal), regret and confidence-width inflation bounds (GP bandits), and risk-vs-posterior-variance asymptotics (random features). All are about size or coverage, not ranking.

### Cited Findings
- **Tang, Sloman & Kaski (AISTATS 2026), Proposition 1:** test MSE splits exactly into three parts — [arXiv:2506.07805](https://arxiv.org/abs/2506.07805):
  - misspecification bias B = E(f̄ − f*)², fixed and "unaffected by the training data";
  - estimation term C = E(f̂ − f̄)²;
  - an "error (de-)amplification" cross term A = 2E[(f̄ − f*)(f̂ − f̄)].
- In the same paper, Theorem 1 tightens a bound of Amortila et al. (2024). There, misspecification error is amplified by the density ratio C∞ = sup d_test/d_train, with misspecification measured as B∞ = ‖f̄ − f*‖∞ — [arXiv:2506.07805](https://arxiv.org/abs/2506.07805) (Amortila et al. read only through this citation).
- **Sloman, Caprio & Kaski (2505.23496):** a decompositional epistemic-error bound, Pr(e ≥ α + B + C + D) ≤ V[Q_T]/α² — [arXiv:2505.23496](https://arxiv.org/abs/2505.23496):
  - e is the total-variation (TV) distance between the predictive distribution and the target task;
  - B = model restriction, C = data scarcity, D = distribution shift.
  - It bounds error size. It says nothing about whether uncertainty ranks error.
  - Conformal prediction is explicitly outside its setting.
- **Sanz-Alonso & Yang (2312.09225), Theorem 3.1 Part II:** when f lies outside the kernel's RKHS, the kriging error picks up a term ‖f_N − f‖∞ (distance to the RKHS) — [arXiv:2312.09225](https://arxiv.org/abs/2312.09225):
  - it is multiplied by √n·[h^s/√(σ_min(K)+λ) + h^{d/2}];
  - the GP posterior variance P(x) does not contain this term, since P depends only on the design and the kernel.
- **Bogunovic & Krause (NeurIPS 2021), Lemma 2:** if the truth is ε-close in sup-norm to an RKHS ball, the posterior mean computed on true data deviates from the well-specified one by at most ε√t/√λ · σ_t(x) — [arXiv:2111.05008](https://arxiv.org/abs/2111.05008):
  - so valid confidence bounds must be inflated to (β_t + ε√t/√λ)σ_t(x);
  - they also show that standard GP-UCB confidence bounds "are no longer valid" under misspecification;
  - Ω(εT) regret is unavoidable.
- **Baek, Berchuck & Mukherjee (2306.03783):** for random-features regression in proportional asymptotics, the expected posterior predictive variance S² and the MAP risk R generally converge to different limits — [arXiv:2306.03783](https://arxiv.org/abs/2306.03783):
  - when the number of parameters is much larger than the number of samples, they agree only on one side of a signal-to-noise phase transition ρ* (Prop. 4);
  - the mismatch comes from f_d not belonging to the random-features class (a nonlinear component F*²).
  - The analysis concerns average size (width vs risk), not per-input ranking.
- **Barber et al. (Ann. Stat. 2023), Theorem 2:** for non-exchangeable split or full conformal, Coverage gap ≤ Σ w̃_i · d_TV(R(Z), R(Z^i)) — [arXiv:2202.13415](https://arxiv.org/abs/2202.13415):
  - R are residual vectors and Z^i swaps test point and calibration point i;
  - for independent points this is ≤ 2 Σ w̃_i d_TV((X_i,Y_i),(X_{n+1},Y_{n+1}));
  - with unweighted split conformal, the gap is ≤ Σ d_TV/(n+1);
  - the authors warn that if the test point "bears no resemblance" to the calibration data, no guarantee is possible without further assumptions.
- Barber et al. contrast their fixed weights with the data-dependent weights w(X_i) used by Tibshirani et al. (2019) for covariate shift — [arXiv:2202.13415](https://arxiv.org/abs/2202.13415). Tibshirani et al. was not read directly.
- **Box & Draper (1959, JASA 54(287):622-654):**
  - In the cases they studied, the best design when both variance and bias are present is "very nearly the same" as the design that ignores variance and minimises bias alone — [Box & Draper PDF host](https://sites.ualberta.ca/~dwiens/stat578/papers/Box%20&%20Draper.pdf); [scilit record](https://scilit.com/publications/16c29e7b314f65dc7afd9b601249caa1).
  - Draper & Guttman (1992), as summarised in secondary sources: even when variance error is several times larger than bias error, the best design is much closer to the all-bias design than to the all-variance design — [search summary via Dette & Melas](https://eldorado.tu-dortmund.de/items/c3424464-02fd-4280-ae01-69eea7b141da).
  - Secondary/abstract level only: the PDF could not be text-extracted.

### Inferences
- **Why no closed-form ranking result exists, and why one is easy here.** All the bounds above are worst-case sizes. Ranking (AUROC, Spearman) depends on the joint distribution over inputs of the signal and the two error parts. That is a distributional question, and it becomes tractable under a Gaussian or lognormal copula. This is the gap the project can fill (Model A in section 4).
- **The misspecification term is design-independent.** Tang et al.'s B and Sanz-Alonso & Yang's RKHS-distance term do not shrink with nominal data. So r grows with n_nom by construction, and any variance signal (which shrinks with n) loses calibration faster at larger n_nom. This predicts the direction of the n_nom effect on coverage. It does not predict that ranking curves collapse in r (see the Model A collapse conditions).
- **Bogunovic & Krause give a crude bound-based threshold for RQ1.** Their result has the form |error(x)| ≤ (β + ε√t/√λ)σ(x) + ε: a σ-proportional part plus a constant ε part. σ can only rank error while the spread of the σ-part across x exceeds ε. That suggests r* ~ (β · IQR(σ)) / typical(σ). The constants β and λ are loose, so use this only as a sanity check, not as the prediction.
- **Conformal under label-function shift.** Covariate-shift weighting does not apply: here P(Y|X) changes, not P(X). Barber's TV bound becomes vacuous when d is large relative to the residual spread. For a σ-normalised split-conformal interval [f̂ ± qσ], Model A gives an exact coverage formula (section 4), which is more useful than the bound.

### Gaps
- No paper found with closed-form AUROC or Spearman of uncertainty vs error under bias. Searched alphaXiv (two queries) for ensemble/GP/random-features uncertainty-error correlation under misspecification.
- "When does a predictor know its own loss?" ([arXiv:2502.20375](https://arxiv.org/abs/2502.20375)) and DEUP ([arXiv:2102.08501](https://arxiv.org/abs/2102.08501)) were seen at listing or abstract level only. DEUP predicts error directly instead of using variance, which is close in spirit to arms 5 and 7. Read both before claiming novelty for the ranking theory.
- Clarté et al. (2023, AISTATS / MLST) on exact asymptotic calibration in high-dimensional models were cited by Baek et al. but not read. They concern classification calibration, probably not ranking under bias.

---

## 2. Theory of active learning and optimal design under misspecification. Is there a crossover condition between variance-driven and space-filling designs as bias grows?

### Takeaway
The classical response-surface literature (Box & Draper 1959 and the "all-bias design") is the only source that gives an explicit bias-vs-variance design trade-off. Its answer is that a small amount of bias already pushes the optimal design to the moment-matching (space-filling-like) design. Modern ML papers give decompositions, robust criteria and bounds, plus small empirical sweeps. **None gives a crossover r for "variance-driven acquisition falls behind Sobol".** It can be derived in closed form in toy models (Model C) and computed semi-analytically for KRR/GP correction models (Model B).

### Cited Findings
- **Box & Draper 1959:** the integrated MSE J = V (integrated variance) + B (integrated squared bias); the all-bias design matches the moments of the region of interest; and the J-optimal design is "very nearly" the all-bias design in typical cases — [Box & Draper PDF host](https://sites.ualberta.ca/~dwiens/stat578/papers/Box%20&%20Draper.pdf) (secondary/abstract level).
- **Dette & Melas (2008):** give sufficient conditions for a design to be all-bias, with spline-regression examples — [TU Dortmund record](https://eldorado.tu-dortmund.de/items/c3424464-02fd-4280-ae01-69eea7b141da) (abstract level).
- **Wiens and co-author, "Optimal Robust Bounded Bias and Bounded Variance Designs" (2025):** minimax designs over classes of alternative response models — [arXiv:2512.21806](https://arxiv.org/abs/2512.21806) (abstract level only).
- **Tang, Sloman & Kaski (2506.07805):** — [arXiv:2506.07805](https://arxiv.org/abs/2506.07805)
  - They show standard Bayesian optimal design (expected information gain, EIG) "is not robust to model misspecification": it gives unrepresentative and possibly error-amplifying training sets. They propose R-IDeA (EIG × MMD representativeness × de-amplification).
  - **Appendix E.1.4 sweeps misspecification degree at 3 levels (0, 1, 2)** on a polynomial toy (truth quadratic or cubic, model linear, 10 designs). At degree 0 every method's error ratio to Random is about 1. Under misspecification the ratio for BAD/EIG "still remains above 1", with "large variation" (instability).
  - With representativeness weighted low (λ = 0.25), both of their robust methods do worse than Random.
- **Sugiyama (JMLR 2006, ALICE):** traditional active learning with ordinary least squares assumes a correct model; importance-weighted least squares is robust to misspecification; ALICE uses the conditional expectation of generalisation error given the training inputs — [JMLR v7 sugiyama06a](https://www.jmlr.org/papers/v7/sugiyama06a.html); related NIPS 2005 paper "Active Learning for Misspecified Models" — [NeurIPS 2005](https://proceedings.neurips.cc/paper/2005/hash/0c1c995b77ea7312f887ddd9f9d35de5-Abstract.html) (abstract level).
- **Bogunovic & Krause:** in misspecified GP bandits, phased GP uncertainty sampling (posterior-variance-driven, elimination-based) achieves R*_T = Õ(B√(γ_T T) + √((ln(1/δ)+γ_T)γ_T T) + εT√γ_T) without knowing ε. The εT term is unavoidable (Ω(εT)). With a linear kernel this recovers the Õ(εT√d) rate of Lattimore et al. — [arXiv:2111.05008](https://arxiv.org/abs/2111.05008).
- **Koermer et al. (Supplement C.3; read earlier, see week1_novelty_multifidelity.md):** bias amplitude α ∈ {0,1,2}; "pecking order" of designs unchanged — [arXiv:2301.10228](https://arxiv.org/abs/2301.10228).

### Inferences
- **Add Tang et al. 2506.07805 to the novelty table.** It is a second prior 3-level misspecification sweep for an active-design rule vs random, finding that EIG-type design does worse than random under misspecification. It is a polynomial toy with no surrogate and no space-filling baseline, so it narrows the "first sweep" claim further but not the core claim (dense r, neural surrogate, true-system experiments, ranking).
- **Crossover in the Box–Draper toy (Model C, section 4).** Fit a line to h(x) = εx² on x ~ U[−1,1], with n points split at ±s and noise σ_η. Then:
  - V = (σ_η²/n)(1 + 1/(3s²)) and B = ε²(s⁴ − 2s²/3 + 1/5);
  - the variance-optimal design (s = 1) stays J-optimal only while γ := nε²/σ_η² ≤ 1/4;
  - it does worse than the moment-matching all-bias design (s² = 1/3) once γ > 3/2, which is where B/V at s = 1 reaches 0.6.
  - So the variance-driven design loses once integrated squared bias is about 60% of integrated variance. My derivation; check the algebra.
- **Static arms are designs, not adaptive rules, under a linear-smoother correction model.** GP posterior variance does not depend on y, so arm 6 with frozen hyperparameters is a fixed design (greedy max-variance is close to a kernel-maximin, space-filling design). Arms 3 and 4 never see true data. For these arms the corrected IMSE as a function of ε is an exact ratio of quadratics (Model B), so the arm-vs-Sobol crossover is the root of a quadratic in ε.
- **A possible explanation of Koermer's null.** In Model B the large-ε gap tends to a plateau set by how the design treats the discrepancy shape g, not by its size. If the discrepancy-aware and simulator-only designs give the same ratio on g as on the ε = 0 target, the ranking never flips. Their single smooth sinusoidal bias is the case where that is likely. This is a hypothesis, not checked against their supplement numbers.

### Gaps
- Exact numerical results in Box & Draper (1959) and Draper & Guttman (1992) were not read (PDF not text-extractable; the JSTOR original is needed).
- Amortila et al. (2024, "misspecification amplification") is known only through Tang et al.; no arXiv ID verified here.
- The "active learning is harmful under misspecification" results in Farquhar et al. (2021, "On statistical bias in active learning") and Vincent & Rainforth (2017) are cited by Tang et al. but were not read.
- Williams et al. (2011, RESS) is still unread (see week-1 notes).

---

## 3. Kernel, NTK and random-features analyses of residual (multi-fidelity) learning: sample complexity of learning a discrepancy (valid alternatives to the withdrawn κ^(2γ) bound), and how predictive variance relates to bias.

### Takeaway
The valid replacement for the withdrawn 2602.01176 bound is the **hypothesis/offset transfer learning (OTL)** minimax theory for kernel ridge regression. Lin & Reimherr (2402.14966) give matching lower and upper bounds, in which the target-sample cost depends on the offset's Sobolev smoothness m and its norm relative to the source signal. Random-features asymptotics (Baek et al.) exist for posterior variance vs risk, but they are heavy random-matrix machinery and do not cover a discrepancy that is never seen in training. I recommend against an NTK/random-features derivation for a 4-6 week solo project.

### Cited Findings
- **Lin & Reimherr, "Smoothness Adaptive Hypothesis Transfer Learning" (ICML 2024), Theorem 3:** with f_T, f_S ∈ H^{m0} and offset f_δ = f_T − f_S ∈ H^m (m ≥ m0), ‖f_δ‖_{H^m} ≤ h, the minimax excess risk is ≍ n_S^{−2m0/(2m0+d)} + n_T^{−2m/(2m+d)} · ξ(h, f_S) — [arXiv:2402.14966](https://arxiv.org/abs/2402.14966):
  - ξ = h²/‖f_S‖²_{H^{m0}};
  - the two-phase estimator SATL (fixed-bandwidth Gaussian kernels in both phases, smoothness chosen by validation) attains this up to log factors.
- **Same paper:** using one kernel of smoothness m0 for both phases makes the offset error n_T^{−2γm0/(2γm0+d)} with γ = min{2, m/m0}, "never faster than" n_T^{−2m/(2m+d)}. This is the KRR saturation effect — [arXiv:2402.14966](https://arxiv.org/abs/2402.14966).
- **Same paper, Proposition 1 (misspecified-kernel target-only KRR):** a Matérn kernel of smoothness m0′ keeps the optimal rate if m0′ ≥ m0/2; below that, saturation gives Ω(n^{−4m0′/(4m0′+d)}) — [arXiv:2402.14966](https://arxiv.org/abs/2402.14966).
- **Sanz-Alonso & Yang:** for quasi-uniform (space-filling) designs, fill distance h_n ≍ n^{−1/d}, and the error bounds are written in fill distance. That is the formal sense in which space-filling designs control the worst-case interpolation error for a fixed RKHS smoothness — [arXiv:2312.09225](https://arxiv.org/abs/2312.09225).
- **Baek et al.:** expected posterior predictive variance → S² = F1²/(1 − χζ²) + F*² + τ² (Prop. 1); risk-variance agreement is governed by an SNR phase transition when parameters greatly outnumber samples, and holds when samples greatly outnumber parameters (Prop. 4) — [arXiv:2306.03783](https://arxiv.org/abs/2306.03783).
- **Withdrawn status of 2602.01176** (κ^(2γ) bound): recorded in the user's week-1 notes; not re-checked here.

### Inferences
- **Scaling law for RQ2 from OTL.**
  - The correction net learns h = −(a + d) on n true points; θ has dimension 4.
  - If h ∈ H^m, the corrected error should fall as n^{−2m/(2m+4)}, about n^{−0.5} for m = 2 and n^{−1/3} for m = 1, times ‖h‖²_{H^m}.
  - Two testable consequences:
    - (i) S2's near-nonsmooth tanh(φ'/δ) term should show a shallower log-error vs log-n slope than S1's quadratic drag at equal r;
    - (ii) at large r the prefactor scales like ε²‖g‖²_{H^m}, so budget-AUC should grow about linearly in log ε with slope ≈ 2 once d dominates.
  - Power warning: n spans only 8 to 56 (factor 7), so the slopes will be noisy. Treat them as exploratory or descriptive.
- **The smoothness proxy already in the pre-registration is the right covariate.** OTL theory says cost depends on ‖·‖_{H^m} and m, not on L2 size. That is why r (an L2 ratio) can fail to collapse across S1 and S2. The H¹/L2 ratio is a cheap stand-in for m.
- **Why not NTK or random features.** Baek et al.'s asymptotic setup (Gaussian-process nonlinear component, proportional limits, Stieltjes-transform fixed points) cannot be reused here. The pendulum surrogate is 4-D, low-dimensional and not in a proportional regime. A KRR/GP stand-in fitted on the actual arrays (Model B) is both closer and cheaper. An empirical NTK of the discrepancy MLP could replace the Matérn kernel in Model B as a robustness check (1-2 days with `torch.func` on CPU).

### Gaps
- No paper found on sample complexity of learning a discrepancy when the *acquisition* is driven by the source model's uncertainty (as opposed to i.i.d. target samples). OTL theory assumes random target sampling from the test marginal.
- Did not find a valid published multi-fidelity analogue of the κ^(2γ) ratio. The OTL quantity ξ(h, f_S) = h²/‖f_S‖² is the closest valid analogue (an RKHS-angle measure).

---

## 4. Concrete tractable models in which r* can be derived or computed semi-analytically; assumptions; how to test on the pendulum runs.

### Takeaway
Three models, in order of payoff per week of work:
- **(A)** a two-source lognormal/Gaussian-copula model for RQ1 ranking, which gives a closed form r½ ≈ 3^{1/4}(s_A/s_D)^{1/2} in the simplest case;
- **(B)** an exact linear-smoother (KRR/GP) IMSE model for RQ2, where each static arm's gap to Sobol is a ratio of quadratics in ε;
- **(C)** a Box–Draper toy with a closed-form crossover nε²/σ² = 3/2, for intuition.

Models A and B both predict a logistic-like curve in log r with slope set by the exponent 2 (from squared norms). That supplies a theoretical justification for the pre-registered 4-parameter logistic and a testable prediction of its slope. Model B also predicts **when H4 (matched transitions) should fail**: when a and d differ in smoothness relative to the correction kernel, which is the S2 case.

All derivations below are mine (not from a paper) and should be checked.

### Cited Findings
- Squared-norm additivity, bias/variance split and the cross term follow the decomposition in Tang et al. Prop. 1 — [arXiv:2506.07805](https://arxiv.org/abs/2506.07805).
- KRR/GP posterior variance depends on design and kernel only; the misspecified part enters the error through ‖f_N − f‖ terms — [arXiv:2312.09225](https://arxiv.org/abs/2312.09225); [arXiv:2111.05008](https://arxiv.org/abs/2111.05008).
- The bias-vs-variance design criterion J = V + B — Box & Draper 1959 — [PDF host](https://sites.ualberta.ca/~dwiens/stat578/papers/Box%20&%20Draper.pdf) (secondary level).
- Offset-learning rates for the correction model — [arXiv:2402.14966](https://arxiv.org/abs/2402.14966).

### Inferences

**Model A — RQ1 ranking: two-source copula model (about 1 week).**
- *Setup.* Per test point i:
  - A_i = ‖a_i‖², D_i = ‖g_i‖², where d_i = ε g_i (or the exact d at each ε);
  - signal s_i (nominal σ, dropout σ or residual).
  - With 201-point trajectories, ‖e‖² = A + ε²D + 2ε⟨a,g⟩. Assume the cosine c_i = ⟨a,g⟩/(‖a‖‖g‖) is small; measure it.
- *Assume* (log A, log D, log s) is trivariate Gaussian with:
  - log-spreads s_A, s_D;
  - baseline skill ρ_a = corr(log s, log A);
  - alignment ρ_d = corr(log s, log D), expected positive or negative depending on S1 vs S2;
  - ρ_AD = corr(log A, log D).
- *Delta method.* Write log‖e‖² = log(e^X + e^Y), with X = log A and Y = 2 log ε + log D.
  - Linearise at the medians: log‖e‖² ≈ w X + (1−w) Y + const, with **w = 1/(1 + r²)**, using r = median‖d‖/median‖a‖ (the median commutes with monotone maps).
  - Then the Pearson correlation on the log/Gaussian scale is
    ρ(r) = [w s_A ρ_a + (1−w) s_D ρ_d] / √(w² s_A² + (1−w)² s_D² + 2w(1−w) s_A s_D ρ_AD).
  - Map to Spearman with ρ_S = (6/π) arcsin(ρ/2). Map to AUROC for the top-20% event by evaluating the bivariate-normal orthant probability at the 80th-percentile threshold (scipy `multivariate_normal.cdf`).
- *Closed form in the simplest case* (ρ_d = 0, ρ_AD = 0, "half" = half of the baseline correlation):
  - 3w² = k²(1−w)² with k = s_D/s_A;
  - so **r½ = 3^{1/4} · (s_A/s_D)^{1/2} ≈ 1.32 · (s_A/s_D)^{1/2}**;
  - with equal spreads, r½ ≈ 1.32.
  - With ρ_d ≠ 0, the curve runs from ρ_a down to the asymptote ρ_d. The 4PL midpoint is where ρ(r) is halfway between them; solve numerically (one line with `brentq`).
- *Shape prediction.* w(r) = 1/(1+r²) is a logistic in log r with slope 2. Apart from the normalising denominator, the ranking curve is a logistic in log r with Hill coefficient near 2.
  - **Testable:** the fitted 4PL slope in B2 should be about 2 (per natural-log unit of r). This prediction does not come for free from the data.
- *Collapse prediction (the r-collapse claim in the goal file).* Curves from different n_nom collapse onto one curve in r **iff** s_A, ρ_a and ρ_AD do not change with n_nom.
  - This can be checked at ε = 0 alone, before any ε > 0 cell is examined.
  - If ρ_a changes with n_nom (likely: ensemble skill often changes with data size), the theory predicts by how much the curves shift.
- *Coverage formula (calibration, separate from ranking).* For a Gaussian or σ-scaled interval [f̂ ± qσ_i] and a scalar summary e_i = c σ_i z + d_i, with z ~ N(0,1) and c² = mean(A/σ²) at ε = 0:
  - Coverage(ε) = mean_i [Φ((qσ_i − d_i)/(cσ_i)) − Φ((−qσ_i − d_i)/(cσ_i))].
  - This is exact under the assumption and needs only ε = 0 quantities plus d.
  - For trajectory-norm intervals, use a noncentral-χ version: ‖e‖² ≈ c²σ²χ²_k(λ = ‖d‖²/(c²σ²)), with effective dof k estimated at ε = 0.
- *Assumptions to state:*
  - (i) lognormal-ish marginals;
  - (ii) a Gaussian copula;
  - (iii) small ⟨a,d⟩ cosine;
  - (iv) w linearised at the medians, so accuracy drops when s_A or s_D > ~1.
  - Check each on pilot seed 0.
- *Test protocol.*
  - Measure (s_A, s_D, ρ_a, ρ_d, ρ_AD, c) on pilot seed 0 at n_nom = 512 only.
  - Predict r½ and the AUROC curve for seeds 1-10, all n_nom, both systems.
  - Report prediction error in log r against the pre-registered ±log 1.5 margin.
  - "Prediction confirmed" means the predicted r½ lies inside the empirical 90% bootstrap CI and within a factor 1.5.
  - Also report a fully empirical Monte-Carlo plug-in (resample z, recombine actual σ_i and d_i). It separates copula error from linearisation error.

**Model B — RQ2 acquisition: exact IMSE of a linear-smoother correction (2-3 weeks).**
- *Setup.* Replace the discrepancy ensemble by KRR/GP (the arm-6 model, on the same PCA outputs) with frozen hyperparameters fitted at ε_ref in the pilot.
  - For a design X (n points from the arm), the corrected-error field is r_X(θ) = h(θ) − k_X(θ)ᵀ(K_X + λI)⁻¹(h(X) + η), with h = −(a + εg).
  - Expected IMSE over noise and test points: IMSE_X(ε) = Q_aa(X) + 2ε Q_ag(X) + ε² Q_gg(X) + N(X).
  - The Q are quadratic forms in the saved arrays: Q_uv = mean_θ[(u − S_X u)(v − S_X v)], with S_X the smoother. N(X) = σ_η² mean_θ ‖k_X(θ)ᵀ(K_X+λI)⁻¹‖².
  - All are computable from pool and test-set arrays with no retraining.
  - With true-system noise σ_obs proportional to the trajectory SD (pre-registration §3), N depends weakly on ε; include it exactly by simulation.
- *Static arms* (3, 4, 6-frozen) give designs X_arm that do not depend on ε. Sobol is also fixed. So
  - G(ε) = log[IMSE_arm(ε)/IMSE_Sob(ε)] runs from the plateau log[(Q_aa+N)^arm/(Q_aa+N)^Sob] at ε → 0 to log[Q_gg^arm/Q_gg^Sob] at ε → ∞;
  - the transition is logistic-like in log ε with slope 2;
  - its midpoint is at the **effective ratio** r_eff² = ε² Q_gg/(Q_aa + N) ≈ 1.
  - The crossover where the arm falls behind Sobol is the positive root of ΔQ_gg ε² + 2ΔQ_ag ε + Δ(Q_aa+N) = 0, if it exists.
- *What it predicts that is not obvious:*
  - (i) **whether** H2 can hold at all: it needs the two plateaus on opposite sides of 0, or at least ordered. If Q_gg^arm/Q_gg^Sob ≈ (Q_aa+N)^arm/(Q_aa+N)^Sob, H2 is predicted null. This is the Koermer case.
  - (ii) **H4 matching:** RQ1's transition sits at raw r ≈ 1.3 (Model A), RQ2's at r_eff ≈ 1. These coincide only if Q_gg/Q_aa ≈ median‖g‖²/median‖a‖², i.e. if a and g leave similar fractions unexplained by the smoother. That requires similar smoothness relative to the kernel.
    - For S2 (rough friction), Q_gg per unit norm should be larger, so the RQ2 transition happens at *smaller* raw r than the RQ1 one.
    - So **the theory predicts H4 holds better on S1 than S2, and the sign of the S2 mismatch**. This is the kind of sharp, falsifiable prediction that answers "true by construction".
  - (iii) Adaptive arms (5, 7, and 6 with refitted hyperparameters) fall outside the exact formula. Treat the formula as a reference curve and report the deviation.
- *MLP transfer test.* Check that the KRR prediction tracks the MLP correction:
  - compare predicted vs measured G(ε) per seed at the 5 ε levels;
  - optionally swap the Matérn kernel for the empirical NTK of the discrepancy MLP at initialisation.
- *Assumptions:* linear smoother; frozen hyperparameters; design fixed across ε; stochastic batch sampling handled by averaging IMSE over the sampled designs (Monte Carlo over the batch randomisation, cheap).
- *Cost:* n ≤ 56 and pool 4,096, so each IMSE is a 56×56 solve plus 2,000×56 products: milliseconds on CPU. The full ε × arm × seed table takes minutes.

**Model C — Box–Draper toy, closed form (2-3 days; intuition figure).**
- Setup: h(x) = εx² + noise σ_η, fit a line on x ~ U[−1,1], n points at ±s.
  - V = (σ_η²/n)(1 + 1/(3s²)) and B = ε²(s⁴ − 2s²/3 + 1/5).
  - Optimality: s⁴(s² − 1/3) = 1/(6γ), with γ = nε²/σ_η².
  - The extreme (variance-optimal) design is optimal iff γ ≤ 1/4.
  - It does worse than the moment-matching design (s² = 1/3) iff γ > 3/2.
- Use it in the paper's intro or appendix to make "variance-driven designs lose once bias is a modest fraction of variance" concrete. It reproduces the Box–Draper message in two lines.
- It is a correction-model-class misspecification story, not the "nominal σ looks in the wrong place" story; Model B covers that one.

**Feasibility for one student in 4-6 weeks:**
- Model A: derivation 2-3 days, Monte-Carlo validation on synthetic lognormals 2 days, pendulum test 2 days.
- Model B: implementation 1 week, validation against MLP 1 week.
- Model C: 2 days.
- Total about 4 weeks, all on a laptop CPU; no Kaggle GPU needed.

**Design decision under time pressure (flagged per the author's standing rule):**
- *What matters:* the theory is only "prediction" if its inputs are frozen before the main grid is examined. The pre-registration is frozen (tag `prereg-v1`). The clean route is to add, in §11 Deviations, a dated entry before the pilot is analysed:
  - which pilot-seed-0 quantities feed Models A and B;
  - which predictions (r½ for RQ1 per system, r_eff crossover and plateau signs for arms 3 and 4, 4PL slope ≈ 2, sign of the S2 H4 mismatch);
  - the success criterion (within the ±log 1.5 margin).
- *Options:*
  - (1) A only (1 week): cheapest, upgrades RQ1;
  - (2) A + B (about 4 weeks): upgrades both RQs and gives an H4 prediction; recommended;
  - (3) A + B + NTK random-features asymptotics (8+ weeks): not doable alone at this deadline.
- *Recommendation:* option 2. I would drop to option 1 if the workshop-version deadline (late January) is at risk, putting Model B in the full paper.
- *Not known yet:* whether the lognormal/copula assumptions hold, and how large the a·d cosine is. Both are measurable on pilot seed 0 in under an hour. Record the decision in project_goal_v2.md's changelog.

### Gaps
- None of Models A-C is checked against data. The linearisation in Model A can be poor if s_A or s_D > ~1 (heavy-tailed errors); the Monte-Carlo plug-in is the fallback.
- "Loses half its ranking ability" is ambiguous in the pre-registration: half of AUROC − 0.5, or the 4PL midpoint between plateaus. Model A uses the 4PL midpoint, consistent with B2. State this in the §11 entry.
- How well a KRR proxy predicts MLP-ensemble correction behaviour at n ≤ 56 is unknown. It is an empirical question for the pilot.

---

## 5. Does any paper already derive such a transition? Closest work and the gap.

### Takeaway
No. No paper found derives (a) the bias-to-approximation ratio at which a variance-type uncertainty signal loses its ranking of true error, or (b) the bias level at which variance-driven acquisition falls behind space-filling design, let alone tests whether (a) and (b) coincide. The closest are listed below.

### Cited Findings
Closest works, with the gap for each:
- **Box & Draper 1959 / Draper & Guttman 1992:** an explicit bias-vs-variance design trade-off, under which the all-bias (moment-matching) design wins under modest bias — [PDF host](https://sites.ualberta.ca/~dwiens/stat578/papers/Box%20&%20Draper.pdf) (secondary level).
  - Gap: polynomial response surfaces; the bias is in the fitted model class, not in a simulator that guides acquisition; no uncertainty ranking.
- **Tang, Sloman & Kaski 2026:** a decomposition with a design-dependent amplification term, and a 3-level misspecification sweep showing EIG design worse than random — [arXiv:2506.07805](https://arxiv.org/abs/2506.07805).
  - Gap: an upper bound, not a transition point; a polynomial toy; no ranking.
- **Koermer et al. 2023:** a 3-level bias sweep with design ranking unchanged — [arXiv:2301.10228](https://arxiv.org/abs/2301.10228).
  - Gap: empirical only; designs simulator runs.
- **Bogunovic & Krause 2021:** confidence widths must be inflated by ε√t/√λ, and Ω(εT) regret is unavoidable — [arXiv:2111.05008](https://arxiv.org/abs/2111.05008).
  - Gap: worst-case bounds; no ranking or crossover vs space-filling.
- **Baek et al. 2023:** a phase transition in agreement between posterior variance and risk (an SNR threshold ρ*) — [arXiv:2306.03783](https://arxiv.org/abs/2306.03783).
  - Gap: average size, not per-input ranking; the misspecified component is present in the training data.

### Inferences
- The defensible novelty claim would be: "a two-parameter-family prediction of the ranking transition r½ (closed form in the simplest case) and an exact linear-smoother prediction of the acquisition crossover, both fed only by ε = 0 pilot quantities and confirmed or refuted on held-out seeds, n_nom and systems".
- This would most likely move the novelty estimate up, because it turns H* from "true by construction" into a falsifiable quantitative law. The size of the move cannot be estimated without results.
- The strongest single outcome would be Model B correctly predicting the *sign* of the H4 mismatch on S2, before the main grid is examined.

### Gaps
- Searches covered alphaXiv (5 discovery queries) and two web searches. JASA/Technometrics full texts beyond Box & Draper were not searched directly (no Technometrics database access). A search on Google Scholar for "integrated mean squared error" + "model discrepancy" + "sequential design" + "crossover" is still worth doing before submission.
- Not read: Wiens 2512.21806 (minimax robust designs; abstract only); Amortila et al. 2024; Farquhar et al. 2021; Vincent & Rainforth 2017; DEUP 2102.08501; 2502.20375.

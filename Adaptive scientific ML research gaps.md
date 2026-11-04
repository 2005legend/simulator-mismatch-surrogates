# Steer simulation budgets by learnability, not hardness

Most of the brainstorm is already taken. Every term of the proposed score S(θ)=a·U+b·R+d·B has been published as an acquisition signal for physics surrogates. Uncertainty-driven active learning (AL) over PDE parameters has an ICLR 2025 benchmark ([AL4PDE](https://arxiv.org/abs/2408.01536)). PDE-residual acquisition appeared in May 2026 ([Polanska et al.](https://arxiv.org/abs/2605.21348)). Diversity/coverage batch selection is standard. Online, difficulty-guided parameter sampling exists in [Breed](https://arxiv.org/abs/2410.05860) and [OGAS](https://arxiv.org/abs/2606.09949). Training-distribution design for distribution shift exists in [Guerra, Nelsen & Yang](https://arxiv.org/abs/2505.21626). A weighted-sum score, or "AL vs random on a toy PDE", would therefore be judged incremental.

The defensible space is not a new score. It is the question of *which kind of signal* should drive simulation acquisition, and how that choice behaves under shift. The recommended goal transfers the correction the RL and RLVR community already made, "hard is not the same as learnable", to simulation acquisition for physics surrogates. Concretely, it compares learning-progress (ALP) and p(1−p)-learnability signals against loss, residual and ensemble-variance signals. The test systems are parametrised oscillators that contain regions unlearnable at the given budget: a chaotic band and a noisy sub-box. Evaluation uses in-distribution (ID), near, far, combined and structural-mismatch test splits. The researchers found **zero direct hits** for learning-progress, learnability or regret signals in physics-surrogate acquisition, with an estimated **~80% confidence** that this is unaddressed on arXiv.

The fit with your background is unusually tight. Your per-prompt EMA solve-rate gate is a learnability estimator, and your cluster-level gates are region-level budget allocators. The estimated cost (12.5–75 CPU-hours plus an optional 7.5–30 T4-hours) fits the stated hardware. A second strong option, "physics-misinformed acquisition" (residual acquisition under structural mismatch), shares the same codebase and serves as the pivot if a pilot falsifies the main premise. Every novelty claim here rests mostly on abstracts. Full texts of the 6–8 closest papers must be read before committing.

## Every term of S(θ) is already published; the gaps are in signal type and evaluation

Read the brainstorm as a list of components, and most of them have an owner. The researchers agree across all five notes that **novelty must come from the question and the evaluation, not from the score**. The table below lists what is taken, with the closest prior work.

| Brainstorm component | Status | Closest prior work |
|---|---|---|
| Monte Carlo / space-filling training data | Standard baseline | Random, Sobol and LHS are the default comparators in AL4PDE ([2408.01536](https://arxiv.org/abs/2408.01536)) |
| Uncertainty term U | Taken | AL4PDE (SBAL, LCMD, BAIT; up to 71% lower average error vs random) ([mlanthology](https://mlanthology.org/iclr/2025/musekamp2025iclr-active/)); Winovich et al. ([2503.03178](https://arxiv.org/abs/2503.03178)) |
| Physics-residual term R | **Taken (May 2026)** | Polanska et al.: PDE-residual acquisition on Burgers and 2D compressible NS, ICLR 2026 AI&PDE workshop ([2605.21348](https://arxiv.org/abs/2605.21348)) |
| Coverage/diversity term B | Taken | LCMD feature diversity in AL4PDE; diversity-aware residual collocation via QUBO ([2603.06761](https://arxiv.org/abs/2603.06761)) |
| Combined weighted score S(θ) | **Taken in pieces, low novelty** | EI-RAR/EI-Grad combines expected improvement and residual ([DOI](https://doi.org/10.1109/ACCESS.2024.3422224)); Lee & Jeong mix a learned high-residual distribution with a base distribution ([2610.09126](https://arxiv.org/abs/2610.09126)) |
| Difficulty-guided online parameter sampling | Taken | Breed: adaptive multiple importance sampling guided by training-loss statistics ([2410.05860](https://arxiv.org/abs/2410.05860)); OGAS: a diffusion model maps a loss/uncertainty difficulty signal to solver parameters, up to 308 parameters ([2606.09949](https://arxiv.org/abs/2606.09949)) |
| Designing training data for distribution shift | Occupied, with theory | Guerra et al. optimise the training distribution for a *known* family of deployment regimes ([2505.21626](https://arxiv.org/abs/2505.21626)) |
| Budget-matched comparison | Standard; overhead-honest ledger exists only for PINNs | Long & Fan's operator-budget ledger, PINN collocation only ([DOI](https://doi.org/10.3390/app16199508)) |
| Shift taxonomy | Exists at PDE scale; ad hoc for low-dimensional ODEs | SIMSHIFT ([2506.12007](https://arxiv.org/abs/2506.12007)); neural-operator stress test ([2601.11428](https://arxiv.org/abs/2601.11428)) |
| Damped pendulum as testbed | Common toy | Neural Context Flows use pendulum and Lotka–Volterra with a few hand-chosen OOD points ([2405.02154](https://arxiv.org/html/2405.02154v6)) |

The crowdedness map below merges the five notes' verdicts. Hit counts come from the researchers' strict arXiv abstract queries (2023-01 to 2026-11) and loose OpenAlex searches. Rate limiting left several queries unanswered, so **every count is provisional and order-of-magnitude only**. Where two notes report different counts for similar queries (53 vs 59 for adaptive sampling + physics-informed; 177 vs 68 for UQ of neural operators), the query strings differed. Both numbers are given.

| Sub-niche | Verdict | Evidence (provisional) |
|---|---|---|
| Residual-adaptive collocation sampling for PINNs | **Very crowded** | 53–59 strict hits, 130 on the broad query, OpenAlex 1,507–2,075; several new variants a month in 2026 ([RAD/RAR-D](https://arxiv.org/abs/2207.10289), [R3](https://arxiv.org/abs/2207.02338)) |
| Curriculum PINNs (hand-scheduled or residual-scheduled) | **Crowded** | 53 hits for curriculum + physics-informed ([Krishnapriyan et al.](https://arxiv.org/abs/2109.01050), [CGMPINN](https://arxiv.org/abs/2605.19263)) |
| New architectures / meta-learning for OOD physical parameters | **Very crowded** | Estimated 30+ method papers; LEADS→CoDA→GEPS→Zebra→operator-splitting lineage plus ≥8 independent 2025–26 papers ([CoDA](https://arxiv.org/abs/2202.01889), [NOVA](https://arxiv.org/abs/2601.19091), [Late Fusion NO](https://arxiv.org/abs/2604.16721)) |
| UQ for neural operators / conformal for operators | **Very crowded / crowded** | 177 (or 68) UQ hits; 28 conformal-operator and 72 conformal-PDE hits, ≥6 in 2026 alone ([Ma et al.](https://arxiv.org/abs/2402.01960), [Mouli et al.](https://arxiv.org/abs/2403.10642)) |
| Hybrid / grey-box architectures | **Very crowded** | 40 UDE hits; active groups ([APHYNITY](https://arxiv.org/abs/2010.04456), [OrthoReg](https://arxiv.org/abs/2606.19145)) |
| Learnability / difficulty filtering in LLM-RLVR | **Very crowded** | 103 + 26 hits, 2025–26 ([Foster et al.](https://arxiv.org/abs/2502.12272), [DAPO](https://arxiv.org/abs/2503.14476)) |
| AL / parameter selection for PDE/operator surrogates, ID error | **Moderate, fast-growing** | 10 strict hits for AL + neural operator; ~10–25 real papers incl. AL4PDE, STAP, MRA-FNO, Polanska, OGAS, Breed ([STAP](https://arxiv.org/abs/2511.18107)) |
| Correcting misspecified physics in PINNs/operators | **Moderate, rising** | ~15 hits, 4 of the top 8 from 2026 ([Zou et al.](https://arxiv.org/abs/2310.10776), [Ma et al.](https://arxiv.org/abs/2606.03469)) |
| PDE-scale OOD benchmarks | **Moderate** | SIMSHIFT 2025, stress test 2026 ([2506.12007](https://arxiv.org/abs/2506.12007), [2601.11428](https://arxiv.org/abs/2601.11428)) |
| Training-distribution design for deployment shift | **Sparse but occupied** | Guerra et al. is a direct, theory-backed entry; ≤3 related works ([2505.21626](https://arxiv.org/abs/2505.21626)) |
| AL explicitly evaluated under parameter shift / extrapolation | **Sparse** | 0/0/0 strict hits for AL × {distribution shift, extrapolation, target distribution} × PDE/surrogate; 1 hit for AL × neural operator × OOD, where OOD is only an evaluation aspect ([2503.03178](https://arxiv.org/abs/2503.03178)) |
| Overhead-charged ("budget-honest") comparison for parameter-space AL | **Sparse** | Only the PINN-collocation ledger found ([Long & Fan](https://doi.org/10.3390/app16199508)) |
| Systematic parameter-shift protocol for low-dimensional ODE surrogates | **Sparse** | <5 found, none systematic ([CODE](https://arxiv.org/html/2511.15619)) |
| AL on parametrised chaotic systems | **Sparse** | 2 hits, incl. a cautionary classification tale ([Payot et al.](https://arxiv.org/abs/2311.18010)) |
| DRO / minimax over PDE parameters | **Sparse** | <5; group DRO appears only as a training detail ([2607.21932](https://arxiv.org/abs/2607.21932)) |
| Residual/physics-informed acquisition under structural mismatch | **Sparse, ≈empty** | 6 hits, 0 relevant (AL + physics + misspecification); 2 off-topic SBI hits |
| UQ tracking error under structural (model-form) mismatch | **Sparse, ≈empty** | 16 hits, 0 test neural-surrogate UQ against error under mismatch |
| Environment selection for meta-learned dynamics models | **Empty in search** | 0 direct hits; LEADS, CoDA and NCF fix training environments by hand ([LEADS](https://arxiv.org/abs/2106.04546)) |
| Surrogate error as the failure event (falsification / rare-event search) | **Empty in search** | 4 hits, all about *system* failure ([Bayesian Safety Validation](https://arxiv.org/abs/2305.02449)) |
| Selective prediction / abstention for physics surrogates | **Empty in search** | 0 relevant of 120 + 28 + 7 hits |
| Learning-progress / learnability / regret acquisition for physics surrogates; noisy-TV in SciML AL | **Empty in search** | 0 direct hits across `"learning progress"`, regret, `"environment design"` and `"prioritized level replay"` × PDE/PINN/operator/surrogate |

One pattern is worth acting on. Crowding is concentrated where groups build *methods*: architectures, UQ estimators, collocation heuristics. The sparse cells all sit where the *question* changes: which signal, evaluated under which shift, charged at which cost. That is good news for a solo diagnostic paper, because the open questions there are empirical and cheap to answer on ODE testbeds.

## Five candidate questions survive merging, and one is a direct transfer of your RLVR work

The five notes proposed 21 research questions between them. Many overlap. Deduplicated, they collapse into five distinct candidates. All five share the same parametrised-oscillator testbed, so the second-ranked candidate can serve as a pivot without new infrastructure.

### Candidate 1: learnability over hardness (learning-progress and p(1−p) acquisition when some regions are unlearnable)

This merges RL-curriculum RQ1/RQ2/RQ4, feasibility RQ3 and the regime-boundary diagnostic from the OOD note.

The **hypothesis** concerns a training box Θ_train that contains regions unlearnable at the given budget. One kind is aleatoric: injected heteroscedastic noise. The other is deterministic: trajectories in a chaotic band at long horizons. In such a box, loss-, residual- and ensemble-variance acquisition over-allocate simulations to those regions and lose accuracy elsewhere, whereas learning-progress and learnability signals avoid the sink. The RL literature documented exactly this failure. Rutherford et al. showed that practical UED regret proxies "do not correlate with regret but with success rate", and proposed training on scenarios the agent can sometimes, but not always, solve ([No Regrets/SFL](https://arxiv.org/abs/2408.15099)). Foster et al. carried the idea into LLM RL ([2502.12272](https://arxiv.org/abs/2502.12272)). Bae et al. lower-bounded policy improvement by success variance ([2504.03380](https://arxiv.org/abs/2504.03380)).

The **closest prior work** in SciML uses only the pre-correction signals. Breed steers by training-loss statistics ([2410.05860](https://arxiv.org/abs/2410.05860)). OGAS steers by loss or uncertainty and itself reports **tail gains with a trade-off in average error** ([2606.09949](https://arxiv.org/abs/2606.09949)), which is the symptom the RL corrections were designed to fix. AL4PDE uses uncertainty and feature diversity, and Polanska et al. use the residual. MUSEL is the only regression work that includes learning progress in a composite score, and it is robotics, not physics ([2412.02331](https://arxiv.org/abs/2412.02331)). Generic deep AL is known to prefer noisier distributions ([2211.00928](https://arxiv.org/html/2211.00928)). The deterministic-chaos case is the novel twist. It has no aleatoric noise, so a CAAL-style split of epistemic from aleatoric uncertainty ([2602.11825](https://arxiv.org/pdf/2602.11825)) cannot rescue variance-based acquisition there.

The **precise delta** is threefold. This would be the first learning-progress or learnability arm in surrogate acquisition, the first test of the noisy-TV failure in its deterministic-chaos form, and the first stratification of signal effects into ID and OOD.

The **minimal experiment** uses a driven damped pendulum with a chaotic band and an unforced pendulum with a noisy sub-box. The surrogate is an MLP ensemble. The design runs 9 acquisition arms × 5 seeds at a matched budget of 512 simulations.

**Compute** is about 4,500 CPU fits, estimated at 12.5–75 CPU-hours. That figure is arithmetic on a 10–60 s per-fit estimate, not a measurement.

The **scoop risk** is moderate. The Raffin group (Breed, OGAS) or the AL4PDE maintainers could add a learning-progress term. The **failure risk** is that learning progress is noisy at small budgets. A **negative result** still contributes: the first test of whether the "No Regrets" diagnosis transfers to supervised surrogates, plus budget-allocation maps that show where each signal spends simulations.

### Candidate 2: physics-misinformed acquisition (residual acquisition under structural mismatch)

This merges AL-note RQ-C, simulator-note RQ-A/B/D, feasibility RQ2 and OOD RQ4. Three of the five notes converged on it independently.

The **hypothesis**: the residual term uses a nominal ODE, while data come from a system with an unmodelled term of strength ε (quadratic drag or Coulomb friction). As ε grows, the rank correlation between residual score and true error collapses, and residual-weighted acquisition falls below Sobol at some crossover ε*.

The **closest prior work** assumes the physics is exact: Polanska et al. ([2605.21348](https://arxiv.org/abs/2605.21348)) and WbAR ([2310.11789](https://arxiv.org/abs/2310.11789)). Misspecification has been studied for *fitting*, not for *data selection*. Zou et al. show that misspecified PINNs "cannot fit the data and satisfy the physical model at the same time" ([2310.10776](https://arxiv.org/html/2310.10776)). McShannon & Dietrich show low-loss silent failures ([2606.25151](https://arxiv.org/abs/2606.25151)). Strouwen & Micluţa-Câmpeanu design experiments to *discover* missing physics, not to improve surrogate accuracy ([2604.01231](https://arxiv.org/abs/2604.01231)).

A natural diagnostic section tests whether ensemble uncertainty keeps tracking error under structural mismatch. Mouli et al. showed it does under *parameter* shift ([2403.10642](https://arxiv.org/abs/2403.10642)). Whether it survives structural shift is untested.

**Compute** is 10–40 CPU-hours according to the simulator note, or 60–120 CPU-hours with the full five-level ε sweep according to the feasibility note.

The **risks** are that reviewers call the result "obvious" and that groups active in 2026 extend their work: McShannon & Dietrich, Polanska et al. A **negative result** is still useful: robustness of residual acquisition to misspecification would be a reassuring, reportable finding. It maps onto the RLVR "misspecified verifier" intuition.

### Candidate 3: an ID–OOD trade-off audit of acquisition rules, with a full-cost ledger

This merges AL-note RQ-A/B/D, OOD RQ2, RL RQ2 and feasibility RQ1. It is the safest candidate.

The **hypothesis**: uncertainty and residual AL inside Θ_train improve ID error but give no gain, or a loss, on far, corner and structural shifts. Once ensemble retraining and pool scoring are charged, AL beats Sobol only above a break-even simulator cost c*.

The **closest prior work** reports data efficiency in-box ([AL4PDE](https://arxiv.org/abs/2408.01536)). Guerra et al. need a known deployment family ([2505.21626](https://arxiv.org/abs/2505.21626)). In materials science, calibrated-uncertainty AL generally failed to beat random for OOD generalisation ([Dale et al.](https://arxiv.org/abs/2511.17760)). Long & Fan audit cost only for PINN collocation ([DOI](https://doi.org/10.3390/app16199508)).

The **weakness** is distinctiveness. Several 2026 individual GitHub repositories already run budget-matched "AL vs random" comparisons on toy PDEs ([ABAL](https://github.com/a-kwo/adaptive-balance-active-learning), [pde-surrogate-active-learning](https://github.com/DavidLaurenti/pde-surrogate-active-learning)). The finding could also reduce to "boundary sampling helps near-OOD". The recommendation therefore keeps this candidate as the **evaluation axis** of Candidate 1 rather than as its own paper.

### Candidate 4: which training environments to simulate for meta-learned dynamics models

This comes from OOD-note RQ1.

The **hypothesis**: for CoDA/NCF-style context models, adaptive selection of the K training environments improves few-shot adaptation to far and corner environments, and the best rule differs from the best ID rule.

The **gap** is real: LEADS, CoDA and NCF hand-pick fixed environment grids ([CoDA](https://arxiv.org/abs/2202.01889), [NCF](https://arxiv.org/abs/2405.02154)). The only data-design ablation found varies the *number* of environments, not where they are placed ([2509.24332](https://arxiv.org/pdf/2509.24332)).

The **risk** is engineering. Re-implementing a meta-learned dynamics model solo adds weeks, and the effects may be small. The scoop source is the well-resourced Sorbonne/MLIA lineage.

### Candidate 5: falsifying the surrogate by treating its own error as the failure event

This comes from simulator-note RQ-C.

The **hypothesis**: safety-validation estimators (cross-entropy method, adaptive importance sampling, Bayesian Safety Validation-style BO) estimate the failure mass P(‖f̂(θ)−f(θ)‖>τ) with far fewer simulator calls than Monte Carlo. The surrogate's self-reported uncertainty misses this region under structural shift.

The **closest prior work** targets *system* failures or extremes, not surrogate error: [Moss et al.](https://arxiv.org/abs/2305.02449), [Delecki et al.](https://arxiv.org/abs/2305.09930) (an inverted pendulum), [Pickering et al.](https://arxiv.org/abs/2204.02488) and [Gao & Karniadakis](https://arxiv.org/abs/2605.15356).

The idea is novel and cheap, but it belongs to a different community (safety validation), sits furthest from your background, and is easy to misread as "AL with an error acquisition".

### Scored comparison

Scores run from 1 (worst) to 5 (best). For crowdedness, 5 means empty. Fit means fit with the RLVR background. CV value means signal to a Master's admissions committee. The scores are synthesis judgements, not measurements.

| Candidate | Novelty | Crowdedness | Feasibility | Fit (RLVR) | CV value | Total |
|---|---|---|---|---|---|---|
| 1. Learnability over hardness (LP, p(1−p), noisy-TV/chaos) | 5 | 5 | 4 | 5 | 5 | **24** |
| 2. Physics-misinformed acquisition under structural mismatch | 4 | 4 | 5 | 4 | 4 | **21** |
| 4. Environment selection for meta-learned dynamics | 4 | 4 | 3 | 3 | 4 | 18 |
| 3. ID–OOD audit + full-cost ledger | 3 | 3 | 5 | 3 | 3 | 17 |
| 5. Surrogate falsification / failure-boundary estimation | 4 | 4 | 4 | 2 | 3 | 17 |

Candidate 2 is the strongest alternative. It is simpler to run, and its "mismatch dial" ε gives clean dose-response plots. It loses on novelty because the result is easier to predict, and because the misspecified-physics literature accelerated through 2026. Candidate 1 asks a question whose answer is not obvious. Learning progress could lose to plain uncertainty, and the chaotic case breaks the standard epistemic/aleatoric argument. It is also the only candidate where your RLVR gates transfer as working code, not as an analogy.

## The recommended goal: test whether the RL "learnable, not hard" correction transfers to physics-surrogate data

**Working title.** *Hard Is Not Learnable: Learning-Progress and Learnability Signals for Simulation Acquisition in Physics Surrogates under Chaos, Noise and Parameter Shift.*

**RQ1.** At a matched simulator budget, do learning-progress (ALP) and learnability (p(1−p)) acquisition signals spend fewer simulations in budget-unlearnable regions than loss, residual and ensemble-variance signals? Do they also reach lower error on learnable in-distribution and out-of-distribution test splits?

**RQ2.** Does the ranking of acquisition signals change between the in-distribution split and the shifted splits (near, far, combined/corner, noise, structural mismatch)? In other words, does the choice of signal matter more out of distribution?

**Hypotheses.**
- **H1 (budget sink).** In systems that contain an unlearnable region, loss-, residual- and variance-driven arms put a larger share of their budget into that region than its volume share. ALP and learnability arms do not.
- **H2 (accuracy).** That reallocation lowers budget-AUC error on ID-learnable and far/combined-OOD splits.
- **H3 (no-harm control).** Where no unlearnable region exists, ALP and learnability stay within the confidence interval of Sobol.
- **H4 (ID–OOD interaction).** Differences between arms shrink in-box as the budget grows, but persist on the OOD splits. This is a testable mechanism that the RL-transfer note proposed ([UED framing](https://arxiv.org/abs/2408.15099)).
- **H5 (diagnostic).** Replicating the "No Regrets" diagnosis, per-cell loss priority correlates weakly with the next round's error reduction, and learnability or learning-progress priority correlates better.

### Systems

| ID | System | Parameters θ | Built-in unlearnable region | Role |
|---|---|---|---|---|
| S1 | Unforced damped pendulum | damping, natural frequency/length, initial amplitude | Heteroscedastic observation noise injected in a sub-box (~15–20% of Θ_train volume) | Aleatoric noisy-TV case |
| S2 | Driven damped pendulum (or forced Duffing) | damping, drive amplitude, drive frequency | Chaotic band at long horizon, labelled once from a dense largest-Lyapunov-exponent map; the short-horizon variant is the within-system control | Deterministic-unlearnable case, the novel one |
| S3 (optional, T4) | 1D viscous Burgers, ν-parametrised, small FNO-1D | ν, initial-condition family | None by design | No-harm control and a check that results are not specific to ODE toys |

Use your own scipy simulators for S1 and S2. This gives full control over noise, mismatch and regime labels, and makes a cleaner CV repository. AL4PDE is the only benchmark with built-in pool-based AL, but its install pins CUDA 12.1, jaxlib-CUDA and jax-cfd on Linux ([AL4PDE README](https://github.com/dmusekamp/al4pde)). That makes it a risky fit for Kaggle images, which is an inference, not a verified incompatibility. For S3, a self-written solver or APEBench/Exponax is the lower-risk route ([APEBench](https://github.com/tum-pbs/apebench)).

The **surrogate** is an MLP mapping θ (optionally with t) to a trajectory, about 30k parameters, as a deep ensemble with M=5. The architecture, ensemble size and training schedule are identical across arms. Your own design choices then cannot leak into the comparison.

### Acquisition arms

The design has nine arms plus one reference. The brainstorm's composite score S(θ) stays in as a baseline, not as the contribution.

| Arm | Signal | Lineage |
|---|---|---|
| Uniform random | none | Monte Carlo baseline |
| Scrambled Sobol | space-filling | **Primary comparator** |
| Loss priority | per-cell probe loss (PLR analogue) | Breed/OGAS-style ([2410.05860](https://arxiv.org/abs/2410.05860)) |
| Ensemble variance | epistemic disagreement | AL4PDE-style ([2408.01536](https://arxiv.org/abs/2408.01536)) |
| Physics residual | nominal-ODE residual of the predicted trajectory | Polanska-style ([2605.21348](https://arxiv.org/abs/2605.21348)) |
| Composite S(θ)=a·U+b·R+d·B | rank-normalised, a=b=d fixed in advance | Your original brainstorm |
| Aleatoric-aware epistemic | epistemic variance weighted by aleatoric confidence | CAAL-like "just use epistemic" baseline ([2602.11825](https://arxiv.org/pdf/2602.11825)) |
| ALP | absolute change in per-cell probe loss over the last k rounds, GMM or grid bandit | ALP-GMM ([1910.07224](https://arxiv.org/abs/1910.07224)) |
| Learnability p(1−p) | per-cell EMA of the "solve rate": the fraction of newly simulated θ whose *pre-acquisition* prediction fell within tolerance τ | SFL / frontier of learnability ([2408.15099](https://arxiv.org/abs/2408.15099)); **your O-SELF EMA gate, transplanted** |
| Oracle-mask Sobol (reference only) | Sobol that excludes the known unlearnable region | Upper bound for what avoiding the sink could buy |

### Budget-matching rule

Every arm in a seed starts from the same 32 Sobol points. It then acquires 10 rounds of 48 simulations, for **512 simulator calls in total**. Results are reported at the checkpoints 128, 256 and 512.

All arms score the same per-round candidate pool of 4,096 Sobol points, redrawn identically for each seed. Acquisition is confined to Θ_train, and no test region is ever queryable.

Every acquisition hyperparameter is frozen on a pilot seed that is excluded from the final analysis: τ, the cell grid, the window k, the EMA decay, and the weights a, b, d.

A secondary ledger records wall-clock time and FLOPs for retraining plus scoring, and reports the break-even simulator cost c*. This extends Long & Fan's operator-budget idea to parameter-space acquisition ([DOI](https://doi.org/10.3390/app16199508)). The simulator is cheap, so "expensive labels" are simulated by the fixed budget. AL4PDE makes the same choice, and the paper must say so.

### Test splits, metrics and statistics

Test sets are labelled with a high-accuracy solver and never seen by acquisition:

- **ID-learnable**: uniform in the box, excluding the unlearnable region.
- **ID-unlearnable**: the unlearnable region, reported separately.
- **Near-OOD**: 0–10% of the box width outside the box along one parameter.
- **Far-OOD**: 10–30% of the box width outside.
- **Combined/corner**: two or more parameters outside the box at once.
- **Noise**: the true system observed with noise of σ ∈ {1%, 5%}.
- **Structural mismatch**: the "true" system adds quadratic drag or Coulomb friction with ε ∈ {0.05, 0.1, 0.2}, while the surrogate trains on the nominal simulator.

Every split is also stratified by dynamical regime (regular vs chaotic, under- vs over-damped) and by distance to the training support. The OOD note predicts that error grows smoothly with distance within a regime but jumps across bifurcations, in line with holomorphic-extrapolation theory ([Adcock et al.](https://arxiv.org/abs/2609.36709)). That stratification costs nothing extra.

The **primary metric** is budget-AUC of the median relative L2 trajectory error over the checkpoints 128, 256 and 512, on ID-learnable and far-OOD. Secondary metrics:
- final and 90th-percentile error;
- the share of the budget spent in the unlearnable region (the mechanism metric for H1);
- the per-cell Spearman correlation between priority at round t and error reduction from t to t+1 (H5);
- Spearman(ensemble σ, error) per split, which bridges to Candidate 2;
- the cost ledger.

**Statistics** follow your protocol:
- 5 seeds per arm for S1 and S2, and 3 for S3;
- all comparisons paired by seed, with shared initial sets, pools and test sets;
- percentile-bootstrap 95% CIs (10,000 resamples) on the per-seed difference in budget-AUC.

Eight **primary contrasts** are declared in advance: learnability vs ensemble variance, ALP vs ensemble variance, and each of the two vs Sobol (four comparisons), each on S2 ID-learnable and far-OOD. They are Holm-corrected; everything else is labelled exploratory. **No improvement is claimed when the CI includes 0.** With five seeds a bootstrap CI is coarse. If pilot variance is high, raise the primary contrasts to 10 seeds, which is cheap on CPU, rather than relax the rule.

### Compute versus budget

The estimates below are arithmetic from the feasibility note's per-fit ranges and were computed for this report. The only measured number is the researchers' timing of one unforced pendulum trajectory: **0.0215 s on CPU** (scipy `solve_ivp`, t∈[0,10]). That is an internal measurement, not a publication, and the driven, long-horizon S2 will be slower.

| Block | Volume | Estimate |
|---|---|---|
| S1 + S2 main grid | 9 arms × 2 systems × 5 seeds × 10 rounds × 5 members = 4,500 fits | **12.5–75 CPU-h** (10–60 s per fit) |
| S2 short-horizon control | 2,250 fits | ~18.8 CPU-h at 30 s per fit |
| S3 Burgers control (optional) | 5 arms × 3 seeds × 10 rounds × 3 members = 450 fits | **7.5–30 T4-h** (1–4 min per fit, warm-started) |
| Training simulations | ~46,000 trajectories | ~16.5 CPU-min at 0.0215 s (lower bound) |
| Test simulations | ~24,000 trajectories | ~8.6 CPU-min (lower bound) |

That totals roughly 30–95 CPU-hours plus up to 30 T4-hours. Spread over the 16-core laptop, it fits comfortably within weeks. The ~6 GB of free RAM, not the core count, limits parallelism. The T4 hours stay within even a conservative reading of the user-stated (unverified) Kaggle quota. Run a 10-minute timing pilot on both machines before fixing the grid, because torch training times were not measured.

### Phased timeline (about 18 weeks, Nov 2026 to Mar 2027)

| Weeks | Phase | Deliverable and gate |
|---|---|---|
| 1–2 | **Novelty check and pre-registration** | Read the full texts of OGAS, Breed, AL4PDE, Polanska et al., MUSEL, Guerra et al., Payot et al. and the smart-surrogate feasibility study ([2407.07674](https://arxiv.org/pdf/2407.07674)). **Kill or reframe** if any of them contains a learning-progress or learnability arm, or a chaos-sink analysis. Commit a timestamped pre-registration (hypotheses, primary contrasts, splits) to the repo. |
| 2–5 | Infrastructure | Simulators S1/S2 with noise and mismatch switches; Lyapunov regime map; split generator; the AL loop with all arms; unit tests; laptop and T4 timing pilot. |
| 5–7 | Pilot (1–2 seeds) | Freeze hyperparameters on the pilot seed. **Go/no-go:** if ensemble variance does *not* over-allocate to the chaotic band in S2, the H1 premise is falsified. Write that up as a finding and pivot the headline to Candidate 2 using the same code. |
| 7–12 | Full grid | S1/S2 on CPU (background jobs); S3 on Kaggle T4. |
| 12–14 | Diagnostics and statistics | Budget-allocation maps, H5 correlations, regime-stratified error curves, cost ledger, bootstrap CIs. |
| 14–18 | Writing and release | 4-page workshop version plus a longer arXiv version; clean repo with configs, seeds and a one-command reproduction. |

### Pre-registered positive versus negative findings

| Outcome | Counts as | What gets written |
|---|---|---|
| In S1 and S2, learnability or ALP beats ensemble variance **and** Sobol on ID-learnable and far-OOD budget-AUC, CI excluding 0 after Holm; the unlearnable-region budget share is lower (CI excluding 0); S3 shows no harm | **Positive (H1–H3)** | "RL curriculum corrections transfer to surrogate acquisition", with mechanism maps |
| Budget share drops but accuracy CIs include 0 | **Partial / mechanism-only** | Reallocation happens but does not pay at these budgets; report the budget at which it might |
| The variance arm does not over-sample the chaotic band (share ≈ volume share) | **Negative on the premise** | Noisy-TV is not a practical failure for surrogates at this scale; still novel, and the pivot trigger |
| Learning-progress or learnability arms are worse than Sobol (CI excluding 0) | **Negative transfer** | The learning-progress signal is too noisy at small simulator budgets; report the noise level of the estimator |
| Arm rankings differ between ID and OOD splits (sign change or CI separation) | **Positive for RQ2 (H4)**, independent of RQ1 | ID-efficiency rankings do not predict shift robustness |
| Loss priority correlates with subsequent error reduction as well as learnability does | **Negative for H5** | The "No Regrets" diagnosis does not replicate for supervised surrogates |
| All CIs include 0 everywhere | **Null** | Report it as a tight null with power analysis; publishable at evaluation or negative-results venues |

## Venues: ICLR 2027 workshops are tight, while ML4PS 2027 and TMLR are comfortable

All NeurIPS 2026 workshop deadlines have passed. ML4PS 2026 closed on 19 Sept 2026 and runs 11 Dec 2026 in Atlanta, independent of NeurIPS this year. It accepts 4-page short papers and has an explicit Evaluations track, which suits this design ([ML4PS 2026](https://ml4physicalsciences.github.io/2026/)). Representations for the Physical Sciences and New in ML closed on 29 Aug 2026 ([RPS](https://representations-physical-sciences.github.io/workshop-2026/), [NewInML](https://newinml.github.io/NewInML2026NeurIPS/)). New in ML admits only authors without a top-ML-conference paper, which fits a Master's applicant if it recurs.

Precedent is encouraging. ML4PS 2023 accepted a negative AL result on a chaotic toy problem ([Payot et al.](https://arxiv.org/abs/2311.18010)). The ICLR 2026 AI&PDE workshop accepted 1D-Burgers active learning ([2605.21348](https://arxiv.org/abs/2605.21348)) and a 6-page limitations paper ([2606.00677](https://arxiv.org/abs/2606.00677)). AL4PDE itself moved from a NeurIPS'24 workshop to ICLR 2025 ([README](https://github.com/dmusekamp/al4pde)).

| Venue | Date status | Fit |
|---|---|---|
| ICLR 2027 workshops (29–30 Apr 2027) ([ICLR](https://iclr.cc/)) | Workshop list and paper deadlines **unknown**. **Estimated** late Jan to early Feb 2027, by analogy with ICBINB@ICLR2026 (31 Jan 2026 deadline) ([ICBINB 2026](https://sites.google.com/view/icbinb-2026/home?authuser=0)) | Best topical fit if an AI&PDE-type workshop recurs, which is **unknown**. About 12 weeks from now, so only a pilot-plus-partial-grid version is realistic |
| ICBINB (negative results) | Rotating theme; 2026 was LLM-specific; 2027 theme **unknown** | Excellent if the 2027 theme is scientific |
| ICML 2027 workshops | Dates **unknown** (estimated May–June 2027) | Comfortable timing for the full study |
| ML4PS 2027 | **Unknown**; estimated Aug–Sept 2027 from the 2026 cycle | Evaluations track; comfortable timing |
| NeurIPS 2027 workshops (incl. New in ML, if it recurs) | **Unknown**; estimated late Aug–Sept 2027 | Comfortable |
| TMLR | Rolling submissions; **acceptance policy unverified** (the page was blocked during research) | Home for an extended, CI-heavy version |
| arXiv preprint | Any time | Post by about March 2027 regardless; it timestamps priority in a niche with moderate scoop risk |

## Limitations: novelty claims stay provisional until eight full texts are read

The search was incomplete in ways that matter for a novelty claim:
- The arXiv API rate-limited the researchers (HTTP 429). In the AL note, 5 of 15 planned queries ran on the first pass and 6 more on retry. Four never returned and were replaced by loose OpenAlex searches, one of which also failed.
- In the OOD note, roughly half of the planned count queries hit the rate limit.
- The RL note's 41 queries all eventually returned, but only after retries.
- The feasibility note was capped at 4 web searches, and the simulator note at 5. The simulator note did not search Google Scholar, Semantic Scholar, OpenAlex or the JCP/CMAME tables of contents, so journal-only work on misspecified PINNs is under-counted.

**Every hit count and every "sparse/empty" verdict above is therefore provisional.** The arXiv abstract-field counts undercount because of synonyms, and the OpenAlex counts overcount heavily.

The researchers read **abstracts, not full texts**. Every "nobody did X" statement is absence-of-evidence. In particular, nobody checked whether AL4PDE, OGAS, Breed or Polanska et al. contain an OOD split, a learning-progress or staleness term, a timing ledger or a chaos analysis. Nobody checked whether Guerra et al. test extrapolation outside the training support, or whether Mouli et al.'s appendix includes model-form shift. A full-text check of the closest 6–8 papers (listed in the timeline) is **required before any novelty claim**, and the Raffin group's arXiv output should be watched monthly.

The following items are **unverified** and must not be cited as fact:
- Zhao & Li 2022, PMLR v190, on adaptive sampling for dynamical systems.
- PhysiNet: the arXiv ID the researchers had, 2106.14705, turned out to be an unrelated paper.
- The "max-up" worst-case-parameter training papers (Le Boudec et al. 2025; Li et al. 2025b), known only through a low-reliability survey ([2511.04576](https://arxiv.org/html/2511.04576)).
- The ICML 2025 reference "Does in-context operator learning generalize to domain-shifted settings?", which may be OpenReview-only.
- Kennedy & O'Hagan (2001) and Brynjarsdóttir & O'Hagan (2014), cited from background knowledge only.
- The authors and DOI of the CMAME 2025 optimal-experimental-design paper. One note found them via OpenAlex; another did not verify them.
- The authors of the heteroscedastic-GPR AL paper.
- The arXiv ID of PDEInvBench.
- The arXiv ID of the NeurIPS 2025 query-synthesis AL paper.
- SIMSHIFT's venue status.
- The abstract of "High Effort, Low Gain" ([2509.11907](https://arxiv.org/abs/2509.11907)), of which only the title was verified.
- Kaggle's GPU quota (user-stated as 30 h/week; the page was blocked).
- TMLR's review policy.

All compute figures except the 0.0215 s pendulum timing are arithmetic estimates. The crowdedness scores and the candidate ranking are judgements built on this partial evidence. A full-text discovery could move Candidate 1 from "empty" to "moderate" overnight, which is why the timeline puts the kill/reframe gate in week 2.

## Conclusion

The RL community already paid for one lesson the SciML acquisition literature has not yet tested: a priority signal that tracks difficulty spends its budget on things that are either already learned or never learnable. OGAS's own reported tail-versus-average trade-off is the first visible symptom on the surrogate side. Deterministic chaos is a version of the noisy-TV problem that RL rarely faced and that standard uncertainty decompositions cannot detect. That makes the project more than a port: it either extends the RL lesson to a case the RL work never covered, or shows that the lesson has limits.

For your profile the asymmetry is favourable. Your RLVR gates become working baselines, your no-claim-when-the-CI-crosses-zero rule becomes the paper's credibility, and each pre-registered branch of the outcome table yields a reportable result. The one irreversible decision is the week-2 full-text check. It is the cheapest step in the plan, and the only one that can invalidate the whole design.

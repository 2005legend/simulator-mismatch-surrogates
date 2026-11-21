## Problem

Neural surrogates of physical systems are trained on a simulator, and the same simulator is then used to judge where the surrogate can be trusted: the spread of a deep ensemble, Monte Carlo dropout, split conformal intervals calibrated on simulated data, and the residual of the governing equation evaluated on the surrogate's prediction. Real systems often contain a term the simulator leaves out (drag, friction, forcing). These simulator-only signals can see the surrogate's approximation error, but they have no access to the gap between simulator and real system. The question is what they rank, and how well, as that gap grows, and how much real data is needed to recover.

## Core idea

Split each surrogate error into two parts. For a test input, a = f_hat_nom - f_nom is the approximation error (surrogate vs simulator), d = f_nom - f_true is the discrepancy (simulator vs real system), and e = a + d is the total error. Use relative L2 norms A = ||a||, D = ||d||, E = ||e||. Index every experiment by one dial, r = median D / median A over the test set. r is moved both by the strength eps of the missing term and by the number of simulator runs used for training (n_nom), so curves from different (eps, n_nom) can be checked for collapse onto one curve in r.

## Main claim (stated as the result to explain)

As r grows, the ranking ability of any simulator-only signal s converges to its rank correlation with the discrepancy, rho_S(s, D). That correlation is a property of where the missing term acts, which the simulator alone cannot reveal. It can be high (uncertainty looks better than ever), near zero (blind), or negative (worse than chance), depending on the system.

Supporting bound (from the triangle inequality): |E - D| <= A and |E - A| <= D, so with kappa = rms(A) / sd(D), |rho(s, E) - rho(s, D)| <= 2 kappa / (1 - kappa) for kappa < 1, with the mirror statement for small r. So ranking of E moves from rho(s, A) at small r to rho(s, D) at large r. The bound fixes the two limits, not the shape of the curve between them; it is loose (vacuous below r of about 2 to 4).

## Theory (the only statements the paper may make; proofs are in the authors' note and go in the appendix)

- **Proposition 1 (limits).** (a) Kendall form: with q_D = fraction of pairs with |D_i - D_j| <= A_i + A_j, |tau(s, E) - tau(s, D)| <= 2 q_D, for any signal s. Proof: pairs outside q_D keep their order in E, each other pair changes the sum by at most 2. Mirror statement with A and D swapped. (b) Pearson form: with kappa = rms(A) / sd(D) < 1, |corr(s, E) - corr(s, D)| <= 2 kappa / (1 - kappa). (c) Scaling d by lambda: q_D -> 0 as lambda -> infinity, so tau(s, E) -> tau(s, D); as lambda -> 0, tau(s, E) -> tau(s, A).
- **Proposition 2 (bracketing when the missing term is in the dictionary).** If the true system is nominal + eps h, the solution map is differentiable in eps, and h lies in the span of the dictionary terms, then for small enough eps the ranking of D equals the ranking of ||L[h]|| (L = linear response of the output to a term), so tau(s, D) lies inside the exact alignment range (min and max over all unit coefficient vectors). The code samples 9 single terms and 2,000 mixtures, an inner approximation of that range, so the computed range has no guarantee; its containment rates are evidence, not a consequence.
- **Proposition 3 (no guarantee outside the dictionary).** If h is not in the span, tau(s, D) is set by the full response ||L[h]||, which need not lie in the range.
- **Remark.** A simulator-only signal does not depend on h, so whenever the dictionary responses are not all rank-equivalent, two true systems with the same r can give opposite limits.
- Numerical check (post hoc, pendulum, 20 seeds): the Kendall bound holds in every seed and is loose by about 10 x at 16 x eps_ref. Present the bound as the reason the limit exists and the TOST results as the measurement. Do not state rates, constants or further theorems.

## What can be done about it

1. An alignment range: for a dictionary of candidate missing terms, the linear response of the simulator to each term (and to random mixtures of up to three terms) gives a range of possible rho_S(s, D) that can be computed from the simulator alone. It is guaranteed to contain the true value only when the missing term is in the dictionary's span (Proposition 2), and it is wide (about 1.0 of a possible 2 for ensemble sigma).
2. A few real observations (8 to 64) are enough to pick the dictionary term whose response best matches the observed discrepancy, which predicts which way the signal will fail, and to calibrate intervals on real data.

## Study design principles

- Fully computational, controlled "true" systems, so a, d and e are known exactly.
- Pre-registered: hypotheses, tests, Holm families, seeds and decision rules frozen by git tags before any data; every later change logged with a date and reason. Results are reported against the frozen wording, including failed and opposite results.
- Four systems: a damped pendulum with missing quadratic drag (S1) or Coulomb friction (S2); a parametric viscous Burgers equation with a missing damping term -u|u| (P1) or forcing sin 3x (P2), learned by an MLP ensemble (System 3); the same Burgers problem learned by a Fourier neural operator ensemble (System 4).
- An out-of-dictionary test (addendum v1.8, frozen before it was run): on System 3, two missing terms with spatial derivatives, -u_x (drift) and u_xx (extra diffusion), whose responses lie 36% and 22% outside the dictionary's span. It tests whether the limit, the bracketing, the linear-response mechanism and term selection survive when the true term is not a candidate.
- Also tested: whether acquisition rules guided by these signals (ensemble variance, equation residual, discrepancy models, a residual computed on observed real trajectories) choose better real experiments than a space-filling Sobol design.

## Honest framing required

The original primary hypotheses (H1 ranking falls, H2 guided acquisition falls behind Sobol, H3 an observed-residual arm wins, H4 the ranking and acquisition transitions coincide) were frozen first and mostly failed or came out in the opposite direction on one system. The main claim above was added later (addendum v1.5) after a development-seed look, then confirmed on 20 main seeds and on two new PDE systems frozen in advance. The out-of-dictionary test (v1.8) was added after all other results were seen, because full-dictionary bracketing turned out to be nearly automatic when the dictionary contains the true term; it must be described that way. The paper must report the primary hypotheses first, label the main claim as a dev-informed secondary prediction that was then confirmed, and report the negative acquisition results plainly.

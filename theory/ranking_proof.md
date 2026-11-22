# Ranking under a missing term: statements and proofs

Draft, 2026-11-21. Written after all main results were seen; it explains them and states the conditions under which they must hold. It does not change any frozen test.

## Setup and notation

Test inputs theta_1, ..., theta_n. For each i:

- f_nom,i: nominal simulator output; f_true,i: true-system output; f_hat,i: surrogate prediction (all vectors in R^T).
- a_i = f_hat,i - f_nom,i (approximation error), d_i = f_nom,i - f_true,i (discrepancy), e_i = a_i + d_i (total error).
- With the per-point normaliser w_i = ||f_true,i|| > 0: A_i = ||a_i|| / w_i, D_i = ||d_i|| / w_i, E_i = ||e_i|| / w_i. This is the normalisation used in the code (`rel` in `scripts/run_pde.py`, `rel_norms` in `mismatch/model_a.py`).
- s_i: any real-valued signal computed without the true system (ensemble spread, dropout spread, equation residual).
- r = median D / median A.

Rank agreement between two vectors x and y is measured by Kendall's tau,
tau(x, y) = (1 / N) * sum over pairs i < j of sign(x_i - x_j) * sign(y_i - y_j), with N = n(n-1)/2,
and, where stated, by Pearson's correlation. The experiments report Spearman's rho; the two rank correlations are tied by Daniels' inequality, -1 <= 3 tau - 2 rho <= 1, and in all our cells they move together (to be checked numerically, item 4 below).

## Proposition 1 (the large-r limit and the small-r limit)

Triangle inequality, per point: |E_i - D_i| <= A_i and |E_i - A_i| <= D_i, because the same normaliser w_i divides all three norms.

**(a) Kendall form.** Let q_D = (1 / N) * #{pairs i < j : |D_i - D_j| <= A_i + A_j}. Then, for any signal s,

|tau(s, E) - tau(s, D)| <= 2 q_D.

*Proof.* If |D_i - D_j| > A_i + A_j, then E_i - E_j has the sign of D_i - D_j, since |(E_i - E_j) - (D_i - D_j)| <= |E_i - D_i| + |E_j - D_j| <= A_i + A_j. So the pair contributes the same term to tau(s, E) and tau(s, D). Any other pair changes the sum by at most 2 (from +1 to -1 or back, or to or from 0 under ties). Dividing by N gives the bound. ∎

The mirror statement holds with A and D swapped: |tau(s, E) - tau(s, A)| <= 2 q_A, with q_A = (1 / N) * #{pairs : |A_i - A_j| <= D_i + D_j}.

**(b) Pearson form.** With kappa = rms(A) / sd(D) < 1,

|corr(s, E) - corr(s, D)| <= kappa (1 + |corr(s, D)|) / (1 - kappa) <= 2 kappa / (1 - kappa).

*Proof.* Write E = D + delta with |delta_i| <= A_i, so rms(delta) <= rms(A). Then cov(s, E) = cov(s, D) + cov(s, delta) with |cov(s, delta)| <= sd(s) sd(delta) <= sd(s) rms(A), and sd(E) lies in [sd(D) - rms(A), sd(D) + rms(A)]. Dividing, corr(s, E) = (corr(s, D) + c) / t with |c| <= kappa and t in [1 - kappa, 1 + kappa], which gives the bound. ∎

**(c) What "r grows" means.** Fix the inputs and the signal, and scale the discrepancy, d_i -> lambda d_i (for a single missing term, d is linear in eps to first order, Proposition 2). Then D scales by lambda, A is unchanged, r scales by lambda, and q_D -> 0 as lambda -> infinity whenever the D_i are distinct. So tau(s, E) -> tau(s, D). Likewise, as lambda -> 0, tau(s, E) -> tau(s, A). The ranking ability of s moves from its alignment with the approximation error to its alignment with the discrepancy, and only the second depends on the missing term.

**Remark (why these are limits, not a curve).** The bounds are worst-case. On our data they are vacuous below r of about 2 to 4 (v1.5 section 2); they fix the two ends, not the shape between them. Pre-registered predictions of the shape (P-A, PA2) were a separate, empirical question, and transferred from one pilot seed in 1 of 6 cases.

## Proposition 2 (the alignment range brackets the limit when the missing term is in the dictionary)

Assume the true system is the nominal one plus eps h, where h is a term in the governing equation (a function of the state, space and time), and that the solution map is differentiable in eps at eps = 0. Let L_i[h] in R^T be the linear response of the output at theta_i to the term h (the derivative of the solution with respect to eps, at eps = 0). Then d_i = -eps L_i[h] + o(eps).

Dictionary: terms h_1, ..., h_m, with responses R_ij = L_i[h_j]. For a coefficient vector c in R^m define

g_i(c) = || sum_j c_j R_ij || / ||f_nom,i||,

and the exact alignment range

I = [ min over c in S^(m-1) of tau(s, g(c)),  max over c in S^(m-1) of tau(s, g(c)) ],

where S^(m-1) is the unit sphere (tau is unchanged by scaling c, so the sphere covers every direction).

**Claim.** If h lies in the span of the dictionary, h = sum_j c*_j h_j, and the values g_i(c*) are distinct across i, then there is eps_0 > 0 such that for all 0 < eps < eps_0, tau(s, D(eps)) = tau(s, g(c*)) and therefore lies in I.

*Proof.* L_i is linear in the term, so L_i[h] = sum_j c*_j R_ij. Hence D_i(eps) = eps g_i(c*) (||f_nom,i|| / ||f_true,i||) + o(eps), and ||f_true,i|| -> ||f_nom,i|| as eps -> 0. So D_i(eps) / eps -> g_i(c*) for every i. With finitely many points and distinct limits, the ordering of D(eps) equals the ordering of g(c*) for eps small enough, so the two Kendall taus are equal. c* / ||c*|| lies on the sphere, so tau(s, g(c*)) is between the minimum and the maximum over the sphere. ∎

**Corollary (both limits together).** Along any sequence with eps -> 0 and rms(A) / eps -> 0 (the surrogate's approximation error shrinks faster than the mismatch, for example by training on more simulations), Propositions 1 and 2 give tau(s, E) -> tau(s, g(c*)), a value inside I. The simulator and the dictionary determine I; they do not determine where in I the system lands, because that depends on c*, which only data from the true system reveals.

**What the code computes.** `alignment_range` in `mismatch/alignment.py` evaluates Spearman's rho, not Kendall's tau, on the single terms and on 2,000 random signed mixtures of 2 or 3 terms, each term first scaled to unit median response. That set is a finite sample of directions on the sphere, so the computed range is an inner approximation of the exact range: a value inside the exact range can fall just outside the computed one, and the claim gives no guarantee for the computed range. The finite-difference responses (step 1e-3) and the finite eps of the experiments add o(eps) terms. The empirical containment rates (R1, PR1: 20 of 20 in every case) are therefore evidence, not a consequence of the claim.

## Proposition 3 (outside the dictionary there is no guarantee)

If h is not in the span of the dictionary, write L_i[h] = sum_j c_j R_ij + P_i, where P_i is the part of the response not explained by the dictionary. Then tau(s, D) can take any value that tau(s, ||sum_j c_j R_ij + P_i||) can, which need not lie in I. Example already in the data: on pendulum S1 with the true term family removed from the dictionary (R1c), the observed rho_S(ensemble sigma, D) fell inside the reduced range in 8 of 20 seeds and inside its 5-95% band in 0 of 20. On the PDE systems the reduced range still contained it in 20 of 20 seeds, probably because the remaining terms (u^2 next to u|u| and u^3; sin 2x next to sin 3x) have nearly the same responses. How often the range brackets a term that is genuinely outside the dictionary was tested in addendum v1.8 (`results_pde_ood.md`, System 3, seeds 1-20) with two terms partly outside the span (out-of-span share 0.36 for -u_x and 0.22 for u_xx). The range contained rho_S(signal, D) in 20 of 20 seeds for both terms and both signals, but it is wide (1.01 and 1.30 of 2), so this is weak evidence. The stronger result is the mechanism: rho_S(signal, D) at 16 x equalled rho_S(signal, ||L[h]||), computed from the true term's linear response alone, to within 0.006 in every seed. So the limit is set by the linear response whether or not h is in the span; what the span decides is only whether the dictionary can find it. Term selection from 16 real fields beat ensemble sigma on -u_x (19 of 20) but not on u_xx (9 of 20, unchanged at 32 and 64 fields). On the pendulum (addendum v1.9, `results_pend_ood.md`) the same linear-response check holds at 4 x for a constant torque and a periodic drive (largest gaps 0.044 and 0.027) but fails at 16 x for the constant torque in every seed (largest gap 0.151): the gap grows roughly in proportion to eps, so the o(eps) terms of Proposition 2 matter within the studied range there. The limit itself (Proposition 1) held on both terms.

## Remark (the simulator cannot settle the sign)

A simulator-only signal s does not depend on h. For any dictionary with responses that are not all rank-equivalent, I has positive width, so two true systems with the same r can give the same signal opposite limits. The measured widths are large: about 1.3 to 1.4 in Spearman units on the pendulum (R2) and 0.7 to 1.0 for ensemble sigma on the PDE systems, against a maximum possible width of 2.

## Items to do before this goes in the paper

1. **Done 2026-11-21: finite-difference step.** Responses recomputed with step 1e-4 against the step 1e-3 used everywhere, on dev seed 999, 200 test points (`theory/checks/fd_step_check.py`, output `theory/checks/fd_step_check.json`; not part of the pipeline). For every dictionary term on both systems and for P3 and P4, the Spearman correlation between the response norms at the two steps is at least 0.99999, so every ranking used by the alignment tests is unchanged. Median relative difference of the response vectors: 0.0001 to 0.006 for all terms except pendulum w^3 (0.019, largest 0.63 at one point, probably where the cubic term makes the response nonlinear in eps at high velocity; not investigated). The o(eps) terms are not hiding a numerical error.
2. **Done 2026-11-21: differentiability.** Pendulum: the right-hand side, with any dictionary term or the true terms added, is smooth in (phi, phi', eps) (tanh(phi'/0.05) is smooth but steep, and w|w| is once continuously differentiable), so by standard ODE dependence on parameters the solution on [0, 10] s is differentiable in eps at eps = 0 (C^1 is enough for Proposition 2). Burgers: periodic, viscous (nu >= 0.10), smooth bounded initial data; every added term is smooth in u and x except u|u| (C^1) and P4, which shifts nu by eps and keeps the problem parabolic; the semi-discrete pseudo-spectral system is a finite ODE system with a C^1 right-hand side, so the same argument applies to the computed solution. The statements are about the discretised solvers actually used, not the continuum PDE.
3. **Decided 2026-11-21: rank statistic.** The frozen tests use Spearman's rho and stay that way. The paper states the propositions for Kendall's tau (where the proof is exact), gives Daniels' inequality to connect the two, and reports Kendall's tau only in the numerical check of the bound (item 4).
4. Exploratory numerical check on the existing arrays (labelled post hoc), done 2026-11-21 on the pendulum, ensemble sigma, n_nom = 512, 20 seeds. The bound holds in every seed and is loose by about a factor 10 at 16 x:

| System | Level | Observed abs(tau(s,E) - tau(s,D)), mean / max | Bound 2 q_D, mean |
|---|---|---|---|
| S1 | 1 x | 0.061 / 0.143 | 1.46 (vacuous) |
| S1 | 4 x | 0.052 / 0.094 | 0.84 |
| S1 | 16 x | 0.026 / 0.038 | 0.43 |
| S2 | 1 x | 0.240 / 0.305 | 1.98 (vacuous) |
| S2 | 4 x | 0.059 / 0.127 | 1.17 (vacuous) |
| S2 | 16 x | 0.006 / 0.017 | 0.42 |

   Reading: the proposition explains why the limit exists; the convergence seen in the data is much faster than the worst case, so the paper should present the bound as the reason and the TOST results (D1, PD1) as the measurement.

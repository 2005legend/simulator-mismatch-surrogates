# Pre-registration addendum v1.8: Blind to the Missing Term (out-of-dictionary terms)

Drafted 2026-11-21. Status: **frozen 2026-11-21** by the git commit tagged `prereg-v1.8`, after the author's approval of the draft as written, and before any run on pilot seed 0 or main seeds 1-20 with the new terms. Nothing in `prereg-v1` to `prereg-v1.7`, `pred-v1` or `pred-v2` changes.

**Why this addendum exists, stated plainly.** It is written after every main result of v1 to v1.7 was seen. Those results showed that the alignment range brackets the true alignment in 20 of 20 seeds when the dictionary contains the true term, but on pendulum S1 with the true term removed only in 8 of 20 (R1c, descriptive). The PDE dictionary was also written knowing the true terms. A reviewer would ask whether the range and the term selection work for a missing term that is not in the dictionary. This addendum tests that, on terms chosen to lie partly outside the dictionary's span, with the tests frozen before any registered seed is run. A proof note (`theory/ranking_proof.md`, 2026-11-21) states when bracketing is guaranteed (term in the span, small eps) and that it is not guaranteed otherwise.

## 1. Disclosure: development look (dev seed 999 only)

`scripts/dev_burgers_ood.py`, System 3 settings as frozen in v1.7 (MLP ensemble of 5, n_nom = 512, 500 test points), output `runs/dev_burgers_ood/dev_seed999.json`, plus a one-off check of the linear-response prediction on the same seed.

| Quantity | P3 (-u_x) | P4 (u_xx) |
|---|---|---|
| Rough eps_ref (r = 1) | 0.00230 | 0.00212 |
| Share of the term's response outside the dictionary span, median over test points | 0.35 | 0.22 |
| AUROC of ensemble sigma at 1, 4, 16 x | 0.93, 0.85, 0.80 | 0.92, 0.88, 0.85 |
| rho(sigma, E) - rho(sigma, D) at 16 x | 0.009 | 0.009 |
| rho(sigma, D) at 16 x; full-dictionary range | 0.36; [-0.47, 0.64] (inside) | 0.54; [-0.47, 0.64] (inside) |
| rho(residual, D) at 16 x; range | 0.17; [-0.68, 0.66] (inside) | 0.29; [-0.68, 0.66] (inside) |
| rho(sigma, D) predicted from the term's true linear response | 0.364 (actual 0.365) | 0.534 (actual 0.538) |
| R3 at 4 x, n = 16: selected term, its statistic vs sigma's | u^2, 0.60 vs 0.36 (beats) | cos 3x, 0.50 vs 0.54 (does not beat) |

Checks of the new terms: P4 with coefficient eps equals the nominal solver with nu raised by eps (largest difference 1e-10); a dictionary term has zero out-of-span share; P4 is stable up to eps = 0.15 with the frozen time step.

## 2. Systems

System 3 only (parametric Burgers, `mismatch/burgers.py`, all settings as frozen in v1.7), with two new missing terms in `mismatch/burgers_ood.py`:

- **P3:** true system adds eps * (-u_x): transport by a uniform background drift.
- **P4:** true system adds eps * u_xx: extra diffusion (the simulator's viscosity is too low by eps).

Both contain a spatial derivative; the frozen nine-term dictionary contains only local functions of u and x. System 4 (operator) is not run: one system keeps the cost near 2.5 CPU-hours and isolates the dictionary question.

## 3. Pilot and levels

Seed 0: eps_ref per term by bisection on r = 1 at n_nom = 512 (brackets P3 [1e-5, 0.5], P4 [1e-5, 0.1]), written to `pilot_results/pde_ood_eps_ref.json`. No prediction of the transition curve is frozen (PA-type tests transferred in 1 of 6 cases; not repeated). Levels eps in {0, 0.25, 0.5, 1, 2, 4, 8, 16} x eps_ref; seeds 1-20; same signals, metrics and arrays as v1.7 section 4 (ensemble sigma, MC dropout, PDE residual, conformal, D3).

## 4. Tests (all secondary; family O)

"k of 20" counts use 20 seeds. Signals for the alignment tests: ensemble sigma and the PDE residual.

- **OB1 (Holm over 2, alpha 0.05):** per seed, slope of ensemble-sigma AUROC on log r over the seven nonzero levels, hypothesis slope < 0, for P3 and P4. Supported only if Holm p < 0.05 and the 95% BCa CI lies below 0. Dev: falls on both.
- **OD1 (limit):** at 16 x, rho_S(sigma, E) - rho_S(sigma, D) within +/- 0.10, TOST across seeds, per term. Dev: 0.009 on both.
- **OR1 (bracketing out of the dictionary):** at 16 x, rho_S(signal, D) inside the full-dictionary alignment range (nine terms, 2,000 mixtures, as v1.7) in at least 18 of 20 seeds, per term and signal. Dev: inside on both. Stated now: the range is wide (about 1.1 of a possible 2 for sigma and 1.35 for the residual), so a value placed uniformly at random would land inside about 55% and 67% of the time; 18 or more of 20 has probability about 0.001 (sigma) and 0.02 (residual) under that reference. The paper will report the widths and these reference probabilities next to the counts.
- **OR1L (linear-response mechanism):** at 16 x, |rho_S(signal, D) - rho_S(signal, ||L[h]||)| <= 0.10 in at least 18 of 20 seeds, per term and signal, where L[h] is the finite-difference linear response of the nominal output to the true term (step 1e-3) on the test points. This tests Proposition 2's mechanism directly, with the true term in place of a dictionary. Dev: gaps 0.001 to 0.004.
- **OR3 (selection from 16 real fields at 4 x):** the dictionary term selected as in v1.7 (full dictionary; the true term is not in it) has a statistic rho_S(term response norm, D) greater than ensemble sigma's in at least 18 of 20 seeds, per term. Expectation stated from the dev look: likely met on P3, likely **not** met on P4 (dev: 0.50 vs 0.54). It is run and reported either way.
- **Descriptive:** out-of-span share of each term's response (median and 10-90% range per seed); the selected terms; D3 counts; coverage of nominal and real-data conformal intervals. The reduced-range test (R1c) does not apply, since the true term is not in the dictionary.

## 5. Compute

About 4 minutes per seed for the cells (one fit shared by both terms) and about 3 minutes per seed for the alignment and response computations: about 2.5 CPU-hours for seeds 1-20. Nothing is cut if compute runs short; the run is resumable.

## 6. Order of events

Author review; freeze v1.8 (tag `prereg-v1.8`); OOD runner and analysis written and tested on dev seed 999; pilot seed 0 (eps_ref only); seeds 1-20; analysis. The proof note is revised alongside but is not a test.

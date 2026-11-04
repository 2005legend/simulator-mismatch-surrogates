# Simulator mismatch and neural physics surrogates

When do the signals used to trust a neural physics surrogate stop working because the simulator behind them is missing a term? This repository holds a pre-registered study of that question. Working title: *Blind to the Missing Term*.

A damped-pendulum simulator is paired with a "true" system that adds quadratic drag (S1) or smoothed Coulomb friction (S2) at strength ε. Every surrogate error splits exactly into approximation error and structural discrepancy, and results are indexed by their ratio r.

- **RQ1:** how well do uncertainty signals (ensemble spread, MC dropout, equation residual, conformal intervals) rank the true-system error as r grows?
- **RQ2:** with at most 56 true-system experiments, which acquisition rule gives the best corrected surrogate, and how does its gap to a Sobol design change with r?

## Status

Design and pre-registration stage. No experiments have run yet.

| File | What it is |
|---|---|
| `preregistration.md` | Frozen pre-registration (tag `prereg-v1`) |
| `prereg_addendum_v1_1.md` | Addendum v1.1: predictions of the transition points, a mismatch-gated acquisition arm, synthetic-powered conformal intervals (frozen, tag `prereg-v1.1`) |
| `project_goal_v2.md` | Goal, abstract, scope and plan, with a dated changelog |
| `week1_novelty_multifidelity.md` | Full-text novelty check of the closest papers |
| `reports/` | Literature review of novelty gaps and upgrades |
| `research_notes/` | Notes behind the review |
| Earlier notes | `Blind to the missing term design.md` (design v1), `Adaptive scientific ML research gaps.md`, `fulltext_part*.md` |

## Pre-registration

Analyses, hypotheses and success criteria are fixed by git tags before the data they concern exist. Order: `prereg-v1`, then `prereg-v1.1` (before the pilot), then `pred-v1` (predicted numbers computed from pilot seed 0, before the main grid). Changes after freezing are listed with dates in section 11 of `preregistration.md`.

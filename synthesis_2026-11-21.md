# Results synthesis and framing decision (2026-11-21)

All pre-registered runs are finished: pendulum S1/S2 (20 seeds; `results_main.md`) and PDE Systems 3/4 (20 seeds; `results_pde.md`). This page sorts every test by status and sets out the framing choice before any writing starts.

## 1. Scorecard

### Primary hypotheses (prereg-v1, frozen before any data)

| Hypothesis | Test | S1 (quadratic drag) | S2 (Coulomb friction) |
|---|---|---|---|
| H1: nominal-only signals lose ranking as r grows | B1 | **opposite** (AUROC rises, +0.050) | **supported** (-0.056) |
| H2: nominal-guided arms fall behind Sobol as r grows | A1 (arm 3, sigma) | **supported** (slope +0.028) | **opposite** (-0.027) |
| | A2 (arm 4, residual) | not supported (Holm p 0.0500) | **opposite** (-0.034) |
| H3: observed-residual arm beats GP arm at largest r | A3 | **opposite** (worse) | not supported |
| H4: ranking and acquisition transitions coincide | B2 | inconclusive (22% of refits dropped) | inconclusive (43% dropped) |
| C1: r-gated arm beats best fixed arm | C1 | **opposite** (Sobol best) | **opposite** (Sobol best) |
| Family F: learned-term acquisition beats Sobol | F1-F4 | inconclusive at this fit budget (stop rule); F1 leans worse | |

Plain reading: **the primary hypotheses as frozen in prereg-v1 are mostly not supported.** Their direction depends on the system, which is what the alignment law (below) predicts, but that law was added in v1.5 after a development-seed look.

### Secondary, dev-informed predictions (v1.5 to v1.7, frozen before any main run)

| Prediction | Pendulum S1 / S2 | Burgers P1 / P2 | Operator P1 / P2 |
|---|---|---|---|
| Ranking falls (PB1) / direction by system (D2) | D2 met / met | PB1 met / met | PB1 met / met (O-P1 weak: -0.009) |
| Large-r ranking equals alignment with D (D1 / PD1, TOST +/- 0.10) | met / met | met / met | met / met |
| Dictionary range brackets the alignment (R1 / PR1) | 20 of 20 / 20 of 20 | 20 / 20 | 20 / 20 |
| True-family prediction within 0.20 (R1s / PR1s) | 20 / 20 (max gap 0.15) | 20 / 20 (0.07) | 20 / 20 (0.03) |
| Term selection beats sigma from 16 real points (R3a, R3b) | 20 / 20 | 20 / 20 | 20 / 20 (P2 near-automatic: collinear sin kx) |
| 32 real points detect the aligned case (D3) | met / met | descriptive (low power) | descriptive |
| Transition midpoint predicted from one pilot seed (P-A / PA2) | **fail / fail** (within 1.5x, outside the 20-seed CI) | **success / fail** | **fail / fail** |
| Horizon prediction (P-C, S2 only) | n/a / met | - | - |

Plain reading: **the alignment law holds on all six system-term combinations, with every bracketing and limit test met in 20 of 20 seeds.** Precise one-seed transition prediction does not hold: 1 of 6 succeed, all misses within 1.5x but outside a narrow 20-seed CI.

### Descriptive facts worth reporting
- Every nominal-guided and physics-guided acquisition arm is worse than plain Sobol design at every mismatch level on both pendulum systems (fig 5; G = +0.08 to +0.18 in log error).
- Nominal conformal intervals cover 0% of true errors at 16x on the PDE systems; 16 real points restore 94-97%.
- Curves from n_nom = 128, 512, 2048 fall on one curve in r (fig 2): r is the right dial.

## 2. What the paper can honestly claim

1. **Main claim (secondary, dev-informed, confirmed on 4 systems):** as the structural share r grows, the ranking ability of any simulator-only signal (ensemble sigma, MC dropout, the equation's residual) converges to that signal's alignment with the discrepancy, a quantity the simulator cannot reveal. It can rise (S1), fall to chance or below (S2, P2), or barely move (operator P1).
2. **What fixes it:** a handful of real points (16) identifies the right term from a dictionary and restores calibrated intervals.
3. **Negative results, stated plainly:** the pre-registered acquisition hypotheses failed; Sobol design beat every guided arm; transition points cannot be predicted to within a 20-seed CI from one pilot seed.

Required wording: the alignment claim was introduced in v1.5 after a development-seed look. It must be labelled as a dev-informed secondary prediction that was then confirmed on main seeds and on two new systems frozen in advance (v1.7). The primary hypotheses are reported first, as frozen, with their outcomes.

## 3. Framing and venue options

**Author's constraint (2026-11-21):** the venue must be in India, or, if outside India, must allow online or hybrid presentation. Journals need no presentation, so they meet it.

| Option | Framing | Venue and deadline | Meets constraint? | Pros | Cons |
|---|---|---|---|---|---|
| **A. Journal, alignment-law paper (recommended)** | Main claim 1-2 above; acquisition and P-A as reported negative results; full pre-registration trail | TMLR (rolling, no deadline, fully online; accepts sound negative results), or SIAM/ASA J. Uncertainty Quantification, or Data-Centric Engineering (both rolling) | **Yes** (no presentation) | Fits a pre-registered study with mixed outcomes; no page limit; honest ordering (primary first) | Slower review (TMLR about 2-4 months) |
| **D. ASCML 2027 talk, India (add-on to A)** | Contributed talk on the alignment law from an abstract | BITS Pilani, Rajasthan, 21-23 Jan 2027; abstract due **30 Nov 2026**; notification 10 Dec; registration by 15 Dec (student fee INR 3,000) | **Yes** (in India, in person) | Topics include UQ and physics-informed NNs; early feedback; abstract-only keeps the journal paper novel | In person in Pilani; 9 days to the deadline; site does not state that an abstract-only talk is allowed, or the proceedings publisher (ask the organisers); do not submit a full proceedings paper, or the journal version may count as a re-publication |
| B. ICML 2027 | Same claim, 8 pages | Abstract 14 Jan, paper 22 Jan 2027; South America (city unconfirmed) | **No** as far as known: no remote-presentation policy published for 2027 | High visibility | Also: headline set after a dev look; 8 pages too short for the trail |
| C. ICLR 2027 workshop | 4-page note | About 1 Feb 2027; San Francisco (unconfirmed), late April | **No** as far as known: remote participation is up to each workshop's organisers | Cheap, early feedback | Venue and remote option unknown |
| E. CODS 2027, India (later) | Applied data-science paper | ACM IKDD CODS (India, usually December; 2026 deadline was August) | Yes | Indian flagship data-science venue | Next deadline about Aug 2027; weaker topic fit |

**Recommendation: A (TMLR), with D as an optional talk.** The study's strength is its pre-registration and four-system confirmation; its weakness for a conference is that the headline was set after a development look and the frozen primary hypotheses mostly failed. A journal with no deadline lets the paper report that order honestly, and needs no travel. ASCML 2027 is the only matching Indian event in reach; the abstract would be written this week. I would change the recommendation if the ASCML organisers say a full proceedings paper is required to present (then skip D), or if ICML 2027 announces remote presentation and the ranking proof is clean within about 3 weeks (then B becomes possible).

**Not known yet:** how general the ranking proof will be; ASCML's rules on abstract-only talks, online presentation and proceedings (email ascml2027@pilani.bits-pilani.ac.in); ICML 2027's city and remote policy.

**Decision (author, 2026-11-21): A + D.** Full paper to TMLR; abstract-only contributed talk at ASCML 2027 (abstract due 30 Nov 2026), after the organisers confirm an abstract-only talk is allowed. No full proceedings paper. Why: meets the India-or-remote rule, and the journal route lets the paper report the failed primary hypotheses before the confirmed alignment law.

## 4. Next steps (after the decision)
1. Ranking proof from the v1.5 sketch.
2. Method, results (primary first, then secondary), discussion, limitations.
3. Introduction, related work, abstract; full draft (gate G5).
4. paper-orchestra on the draft (read the repo first).

# Pre-registration addendum v1.10: Blind to the Missing Term (blind dictionaries, drawn missing terms)

Part A drafted and frozen 2026-11-22 by the git commit tagged `prereg-v1.10a`, before the missing terms are drawn. Part B (tests and expectations) is added after the draw and a development look on seed 999, and frozen by tag `prereg-v1.10b` before any run on pilot seed 0 or main seeds 1-20. Nothing in `prereg-v1` to `prereg-v1.9`, `pred-v1` or `pred-v2` changes.

**Why this addendum exists.** It is written after every result of v1 to v1.9 was seen. The candidate-term dictionaries of v1.6-v1.9 were written knowing the true missing terms (for example, tanh(w/0.1) is in the pendulum dictionary because S2 is Coulomb friction), so the bracketing and selection results may be favoured by the design. Here neither the dictionary nor the missing terms are chosen by the author with the answer in mind: the dictionaries follow published default rules, and the missing terms are drawn at random from fixed pools after this part is frozen, with a seed nobody can choose.

## Part A

### A1. Dictionaries, fixed by rule (`mismatch/blind.py`)

- **Pendulum, `PEND_LIB` (13 terms):** the default library of SINDy (Brunton, Proctor and Kutz, 2016) for the state (p, w) = (phi, phi'): all monomials of degree 1 to 3 (p, w, p^2, p w, w^2, p^3, p^2 w, p w^2, w^3) and the Fourier terms sin p, cos p, sin 2p, cos 2p. Each term is added to the angular acceleration.
- **Burgers, `BURG_LIB` (12 terms):** the default library of PDE-FIND (Rudy, Brunton, Proctor and Kutz, 2017, Science Advances 3(4), doi 10.1126/sciadv.1602614): {1, u, u^2, u^3} x {1, u_x, u_xx}, derivatives spectral on the periodic grid.

Linear responses use the frozen finite-difference step 1e-3, as before. The alignment range uses the frozen `alignment_range` (single terms and 2,000 random mixtures of 2 or 3 terms).

### A2. Pools of plausible missing terms (`mismatch/blind.py`)

Ten per system, each a physically plausible process a simulator might omit. The true system adds eps * h.

Pendulum (`PEND_POOL`, h(t, p, w) added to the angular acceleration):

| Term | h | In the span of PEND_LIB? |
|---|---|---|
| quadratic drag | -w\|w\| | no (approximated by odd powers of w) |
| Coulomb friction | -g tanh(w / 0.05) | no |
| Stribeck friction | -g tanh(w / 0.05) (0.5 + 0.5 exp(-(w / 0.3)^2)) | no |
| linear damping | -w | yes |
| cubic spring | -p^3 | yes |
| constant torque | g | no |
| periodic drive | g sin(1.5 t) | no (time-dependent) |
| pivot vibration | -g cos(5 t) sin p | no (time-dependent) |
| aerodynamic drag | -w \|w\|^0.5 | no |
| offset restoring | -g sin(p - 0.2) | yes (a combination of sin p and cos p) |

Burgers (`BURG_POOL`, h(u, x) added to u_t):

| Term | h | In the span of BURG_LIB? |
|---|---|---|
| quadratic damping | -u\|u\| | no |
| forcing sin 3x | sin 3x | no (x-dependent) |
| linear damping | -u | yes |
| drift | -u_x | yes |
| extra viscosity | u_xx | yes |
| localized source | exp(-4 (x - pi)^2) | no (x-dependent) |
| saturating damping | -tanh(2u) | no (approximated by u, u^3) |
| varying viscosity | d/dx((1 + 0.5 sin x) u_x) | no (x-dependent) |
| forcing cos 2x | cos 2x | no (x-dependent) |
| cubic damping | -u^3 | yes |

So 3 of 10 pendulum terms and 4 of 10 Burgers terms lie exactly in the span. The author wrote both pools and both libraries and knew this classification when writing them; the draw, not the author, decides which terms are tested.

**Disclosure: stability screening before the freeze.** The first Burgers pool contained dispersion (u_xxx) and hyperviscosity (-u_xxxx). With the frozen explicit Burgers solver, dispersion diverges from eps = 0.01 and hyperviscosity at eps = 0.001, so they were replaced by the localized source and saturating damping before this freeze. The screening used only solver stability on 32 Sobol inputs at eps in {0.001, 0.05, 0.2, 0.5}; no test statistic was computed. All twenty listed terms run at eps up to 0.2 (extra viscosity and varying viscosity diverge at 0.5, the constant torque leaves the plot range at 0.5; the bisection brackets for eps_ref stay below 0.1).

### A3. The draw rule (`mismatch.blind.draw`)

- Let H be the full hash of the commit tagged `prereg-v1.10a`. The generator is `numpy.random.default_rng(int(H[:12], 16))`.
- It draws a random permutation of the pendulum pool, then of the Burgers pool (order of the pool dictionaries as written in `mismatch/blind.py` at that commit). The first 3 entries of each permutation are the drawn terms.
- **Redraw rule:** if a drawn term cannot be run at 16 x eps_ref with the frozen solver, or its eps_ref bisection fails inside its bracket, it is replaced by the next entry of the same permutation. Every replacement is reported.
- `scripts/draw_blind_terms.py` reads the tag, applies the rule and writes `pilot_results/blind_draw.json`; anyone can rerun it.

### A4. What part B will contain

The same design as v1.8 and v1.9, with the blind library as the dictionary: eps_ref per drawn term on pilot seed 0, levels 0-16 x eps_ref, seeds 1-20, the limit test at 16 x, bracketing inside the blind range at 16 x (with widths and reference probabilities), the linear-response gap at 4 x and 16 x, and selection from 16 noisy observations at 4 x against ensemble sigma, plus the out-of-span share of each drawn term against the blind library. Part B states the thresholds and the expectations from a development look on seed 999, before any registered run.

### A5. Design decision (project rules)

- **What matters:** whether bracketing and selection still work when the dictionary was not written with the answer in mind. Six drawn terms (three per system) with 20 seeds each say whether that holds for most drawn terms or only some; they cannot estimate a precise rate over all possible physics, and the paper will say so.
- **Options considered:** both systems with 3 drawn terms each (chosen by the author, 2026-11-22); the pendulum only with 4 terms (cheaper, one system); both systems with 5 terms each (firmer rate, about 1.5 days).
- **Most likely outcome (stated now):** in-span terms bracket and are selected well; out-of-span terms bracket mainly because the range is wide, and selection often fails when no library term behaves like the missing one.
- **What cannot be known yet:** which terms the draw will give, and whether any of them needs the redraw rule.

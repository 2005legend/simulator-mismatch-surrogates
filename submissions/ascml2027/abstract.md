# ASCML 2027 abstract (draft, contributed talk)

**Title.** Blind to the missing term: what simulator-trained uncertainty can and cannot see

**Authors.** Sidaarth [add: surname, affiliation, email]; [add: co-author name, affiliation, if any]

**Research area.** Inverse Problems and Uncertainty Quantification; Machine Learning and Deep Learning

**Abstract.**

Neural surrogates of physical systems are trained on a simulator, and the same simulator is then used to judge where the surrogate can be trusted: ensemble spread, Monte Carlo dropout, and the residual of the governing equation. If the simulator leaves out a term that the real system has, these signals see the surrogate's approximation error but not the gap between simulator and system. We measure what they see as that gap grows, indexed by r, the median simulator-system discrepancy divided by the median approximation error. Systems: a damped pendulum missing quadratic drag or Coulomb friction, and a parametric Burgers equation missing a damping or forcing term, learned by an MLP ensemble and by a Fourier neural operator. A last test used two terms with spatial derivatives (drift, extra diffusion) that lie partly outside our dictionary of candidate terms. All tests were pre-registered and run on 20 seeds.

Once r is large, each signal ranks the true error as well as it correlates with the discrepancy (rank correlations within 0.10 in all eight system-term cases). That correlation can have either sign: with drag the AUROC of ensemble spread rose from 0.72 to 0.94; with friction it fell to 0.39. The simulator alone cannot tell which. The correlation is fixed by the simulator's linear response to the missing term, which predicted it to within 0.006 in every seed, in or out of the dictionary. A range computed from nine candidate terms contained it in every seed, but the range is wide and is guaranteed only when the true term is in the dictionary. Sixteen real observations picked a term that ranked the error better than ensemble spread in every seed when the dictionary held the true term, in 19 of 20 for drift, and in 9 of 20 for extra diffusion, with no gain from more data. On Burgers they restored interval coverage from 0% to 94-97%. Two results were negative: no simulator-guided acquisition rule beat a Sobol design (8-20% higher error), and transition points predicted from one pilot seed fell inside the 20-seed interval in 1 of 6 cases.

<!-- About 350 words; trim if the ASCML limit is lower. Check the ASCML template and word limit before submitting. -->

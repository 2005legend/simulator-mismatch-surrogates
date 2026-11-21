# ASCML 2027 abstract (draft, contributed talk)

**Title.** Blind to the missing term: what simulator-trained uncertainty can and cannot see

**Authors.** Sidaarth [add: surname, affiliation, email]; [add: co-author name, affiliation, if any]

**Research area.** Inverse Problems and Uncertainty Quantification; Machine Learning and Deep Learning

**Abstract.**

Neural surrogates of physical systems are trained on a simulator, and the same simulator is then used to judge where the surrogate can be trusted: ensemble spread, Monte Carlo dropout, and the residual of the governing equation. If the simulator leaves out a term that the real system has, these signals see the surrogate's approximation error but not the gap between simulator and system. We measure what they see as that gap grows. The dial is r, the median size of the simulator-system discrepancy divided by the median approximation error. We study four systems: a damped pendulum with missing quadratic drag or Coulomb friction, a parametric viscous Burgers equation with a missing damping term (-u|u|) or forcing term (sin 3x) learned by an MLP ensemble, and the same Burgers problem learned by a Fourier neural operator. Every test was pre-registered, and each system was run on 20 seeds.

Once r is large, each signal ranks the true error exactly as well as it correlates with the discrepancy: the two rank correlations differed by less than 0.10 in all six system-term cases. That correlation can be positive or negative. With quadratic drag, the AUROC of ensemble spread rose from 0.72 to 0.94; with Coulomb friction it fell to 0.39, below chance. The simulator alone cannot tell which case holds. A range computed from a dictionary of nine candidate terms contained the observed correlation in 20 of 20 seeds per case. Sixteen real observations were enough to choose a dictionary term that ranked the error better than ensemble spread in every seed, and on the Burgers systems they restored interval coverage from 0% to 94-97%. Two results were negative. No simulator-guided acquisition rule beat a Sobol design (8-20% higher error), and transition points predicted from a single pilot seed fell inside the 20-seed interval in only 1 of 6 cases.

<!-- About 275 words. Check the ASCML template and word limit before submitting. -->

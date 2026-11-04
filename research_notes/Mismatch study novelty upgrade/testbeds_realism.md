# Realistic testbeds beyond pendulum ODEs for the structural-mismatch study

Scope: candidate testbeds (synthetic PDEs with a controllable missing term, sim-vs-sim pairs with a natural mismatch, real sim-to-real data) that could sit next to S1/S2 in "Blind to the Missing Term". Research date 2026-11-04, about 18 tool calls. Items marked **[unverified]** come from my background knowledge or my own arithmetic and were not confirmed from a source in this session.

## Q1. PDE testbeds with a tunable missing term, cheap to solve: which libraries make it easy?

### Takeaway
Exponax (the solver underneath APEBench, MIT licence, JAX) is the cheapest route to a PDE version of S1/S2. It covers Burgers, KdV, Kuramoto-Sivashinsky, Fisher-KPP/Allen-Cahn/Gray-Scott reaction-diffusion and 2D Navier-Stokes vorticity, and lets you set physical coefficients, so "true = nominal + ε·(term)" is mostly a matter of setting coefficients. Three 2023-2026 misspecification papers (2606.03469, 2310.10776, 2603.24948) use diffusion-reaction, Burgers and cavity flow with a wrong physics model. None of them releases a benchmark or code, so the user would generate data themselves rather than download it. The name "MFPDE-Bench" could not be confirmed. arXiv:2608.04708 is a multi-fidelity neural-operator benchmark with LF-HF equation discrepancies, but it lists no code or data link.

### Cited Findings
- Exponax supports Burgers and KdV (1D-3D), Kuramoto-Sivashinsky (standard and conservative, 1D-3D), Navier-Stokes vorticity (2D) and velocity (3D), Kolmogorov flow, and reaction-diffusion (Fisher-KPP, Allen-Cahn, Cahn-Hilliard, Gray-Scott, Swift-Hohenberg). MIT licence. Its generic steppers come in "physical coefficients, normalized, and difficulty-based" flavours — [Exponax GitHub](https://github.com/Ceyron/exponax)
- The only speed figure Exponax states: 50 trajectories of 2D KS (200 steps, 128×128) in under a second on a modern GPU. No CPU figure is given. Dividing gives an upper bound of about 20 ms per 2D trajectory on GPU (my arithmetic) — [Exponax GitHub](https://github.com/Ceyron/exponax)
- APEBench: a JAX benchmark for autoregressive neural PDE emulators on periodic domains in 1D-3D, with "≥46 PDEs", a spectral reference solver (Exponax) and differentiable-physics support. MIT licence, NeurIPS 2024, arXiv:2411.00180. No runtime claims on the README — [APEBench GitHub](https://github.com/tum-pbs/apebench)
- PDEBench covers 1D advection, 1D Burgers, 1D reaction-diffusion, 1D diffusion-sorption, 2D Darcy, 2D diffusion-reaction, 2D shallow water, and compressible and incompressible NS. Code is MIT; the README states no data licence or sizes. Generation uses Hydra YAML configs (e.g. `+multi=beta1e0.yaml`), so coefficients look configurable but this is not stated outright. Data: DaRUS doi:10.18419/darus-2986 — [PDEBench GitHub](https://github.com/pdebench/PDEBench)
- arXiv:2606.03469, "Physics-guided correction for operator learning under model misspecification" (Ma, Boullé, Yang, Wu, Guo): a prior operator from an approximate physical model plus a learned additive correction (serial DeepONet), with deep-ensemble UQ. Testbeds are diffusion-reaction, Burgers, cavity flow and hyperelasticity, plus sparse/noisy observation tests. No code link on the abstract page — [arXiv:2606.03469](https://arxiv.org/abs/2606.03469)
- arXiv:2310.10776, "Correcting model misspecification in PINNs" (Zou, Meng, Karniadakis): DNNs learn the gap between assumed physics and data, with B-PINN/ensemble UQ. Examples are non-Newtonian channel flow and cavity flow. No code link — [arXiv:2310.10776](https://arxiv.org/abs/2310.10776)
- arXiv:2603.24948 (Zhou, Feng, Guo, Wu): latent-variable model with a discrepancy decoder for misspecified PDEs. According to a search snippet it deliberately misspecifies a reaction term (not confirmed on the abstract page). Whether it varies misspecification strength is not stated — [arXiv:2603.24948](https://arxiv.org/abs/2603.24948)
- arXiv:2608.04708 (Faza et al., 5 Aug 2026), "Benchmarking Multi-fidelity Neural Operators on Complex PDE Problems with Non-trivial Fidelity Differences": four PDE cases, including a modified Darcy flow with LF-HF governing-equation discrepancies and an unsteady smoke inflow. It reports that feeding LF predictions into HF models can amplify errors when discrepancies are large, and that transfer learning helps consistently. No code or data link is listed — [arXiv:2608.04708](https://arxiv.org/abs/2608.04708)
- A search for "MFPDE-Bench" returned nothing under that name. The closest hits were MDBench (model discovery) and PDEBench — [search: MDBench arXiv:2509.20529](https://arxiv.org/html/2509.20529v2)

### Inferences
- **Best PDE candidate: S3 = 1D Burgers (nominal) vs KdV-Burgers (true, + ε·u_xxx dispersion), or 1D KS-type with a dropped term, on Exponax.** Dispersion moves energy to high wavenumbers and creates oscillatory tails, so the discrepancy has a different spatial and spectral signature from the approximation error. That gives the "discrepancy lives somewhere else" test that S1/S2 provide in θ-space. θ could be (viscosity ν, IC amplitude, IC wavenumber, IC phase), which is four dimensions like the pendulum box. Exponax may not ship a combined Burgers+dispersion stepper under that name; its generic nonlinear stepper with physical coefficients probably allows it, but check the API **[unverified]**.
- **Second PDE candidate: reaction-diffusion with a missing or wrong reaction (Fisher-KPP nominal = pure diffusion + logistic; true adds ε·u³ or an Allen-Cahn term).** It matches the testbeds in 2606.03469 and 2603.24948, so reviewers can relate it to prior misspecification work. Prefer a reaction term over a "heat equation with missing source": the missing-source version is linear, the discrepancy is just an additive field, and a reviewer would call it as toy as the pendulum.
- **Navier-Stokes closure (coarse grid as nominal, fine grid filtered as true):** this is the most realistic (it is the closure problem from the ML4PS list below) but has no clean ε dial; you only get a natural r from the resolution ratio. It is 2D, so data and fits are about 100× bigger. I would use it only as an exploratory "natural-r" point, not a main system.
- **Compute (all [unverified], my estimates):** 1D spectral Burgers/KS at N = 128-256 with about 100-200 saved steps should run at roughly 5-50 ms per trajectory on CPU in JAX once JIT-compiled (batch with `vmap`). Simulation needs per ε level are about (2,048 nominal + 2,000 test + 56 queried) × 10 seeds, so about 40k-80k solves in total: roughly 0.1-1 CPU-h. **Simulation is not the bottleneck; fits are.** With the output reduced to about 32-64 PCA coefficients, the MLP/discrepancy fits should cost about the same as the pendulum fits (201-point output). A reduced grid of 1 system × 3 ε × 4 arms (Sobol, nominal σ, discrepancy σ, observed residual) × 10 seeds × 6 rounds × 5 members = 3,600 fits is about 10-60 CPU-h at the pre-registration's 10-60 s per fit guess. An FNO surrogate instead of an MLP would likely need the Kaggle T4 and adds tuning risk.
- **Arm 7 (observed nominal residual) carries over directly:** for a PDE, the nominal-PDE residual computed on noisy true snapshots needs spatial derivatives (spectral or smoothed). That is a harder and more realistic version of the spline-derivative issue already in the pre-registration.

### Gaps
- MFPDE-Bench: not found under that name. The identity of the "MFPDE-Bench" the user remembers is unconfirmed; it may be 2608.04708 or a different paper.
- The Well (PolymathicAI): not checked this session. From memory it is a roughly 15 TB, 2D/3D collection of high-fidelity physics simulations **[unverified]**. It is too large for 5 GB of RAM and has no nominal/true pair, so I judge it unsuitable.
- JAX-CFD and PDEArena: not checked this session. No runtime or licence numbers.
- I found no stated CPU per-sample runtime for any of the libraries. The pilot has to measure it.
- I have not read the experimental sections of 2606.03469 or 2603.24948, so I do not know whether they sweep the misspecification strength. If they do, that affects novelty. Read section 4 of each.

## Q2. Real sim-to-real datasets with a known missing term and enough parameter variation

### Takeaway
I found no public real-system dataset that has both (a) a known nominal model with a known missing term and (b) a dense sweep of physical parameters θ that could serve as surrogate inputs. Most real system-ID data (nonlinearbenchmark.org, EMPS, robot arms) records one physical system under different input signals. The most usable real case is NeuroBEM (real quadrotor flights, which ship with model-predicted forces and residual columns). It allows pool-based selection over recorded flight segments, but the "θ" there is the flight condition, not system parameters, and the licence is not stated. A battery pair (PyBaMM SPM as nominal, SPMe/DFN as "true", with measured Ecker data as a sanity check) gives a physically meaningful, controllable mismatch at low cost.

### Cited Findings
- nonlinearbenchmark.org lists 15 benchmarks (Nano-Drone, Fine Steering Mirror, Unsteady Fluid Mechanics, Industrial Robot, Two-Story Frame, EMPS, F-16 GVT, Coupled Electric Drives, Cascaded Tanks, Bouc-Wen, Silverbox, Wiener-Hammerstein variants), none of them a pendulum. It warns that some datasets may be unavailable because data.4TU hosting is down, and points to a Python loader at MaartenSchoukens/nonlinear_benchmarks. No sizes or licences on the index page — [nonlinearbenchmark.org](https://www.nonlinearbenchmark.org/)
- NeuroBEM (RSS 2021, UZH RPG): about 1 h 15 min of agile real quadrotor flight at up to 65 km/h, Vicon motion capture, logged at 400 Hz (position, velocity, battery voltage, motor speeds and derivatives). It reports about 50% lower prediction error for its hybrid BEM + NN-residual model than for either part alone — [NeuroBEM project page](https://rpg.ifi.uzh.ch/NeuroBEM.html); [arXiv:2106.08015](https://arxiv.org/abs/2106.08015)
- NeuroBEM processed CSVs: columns 30-35 hold predicted forces/torques and columns 36-41 hold residual forces/torques. Flights are listed in `Flights.txt`. Raw data comes as rosbags and Betaflight logs. **Download size and licence are not stated** — [NeuroBEM readme](https://rpg.ifi.uzh.ch/neuro_bem/Readme.html)
- IRIS (arXiv:2603.16432): 240 real videos (4K, 60 fps) of single- and multi-body systems, each paired with governing equations and independently measured ground-truth parameters. A search snippet mentions pendulum damping and a sliding cone's friction coefficient. The paper says the data is "publicly released", but the abstract page gives no download link or data licence — [arXiv:2603.16432](https://arxiv.org/abs/2603.16432)
- Real Quanser Furuta pendulum and cart-pole trajectories (swing-up then unactuated fall) are used in the differentiable Newton-Euler paper — [arXiv:2010.09802](https://arxiv.org/pdf/2010.09802)
- PyBaMM ships Ecker et al. experimental discharge data (CSV, 1C and 5C) and a notebook comparing DFN to it. It also has a COMSOL-vs-PyBaMM DFN notebook across C-rates — [PyBaMM Ecker notebook](https://docs.pybamm.org/en/stable/source/examples/notebooks/models/compare-ecker-data.html); [PyBaMM COMSOL notebook](https://docs.pybamm.org/en/stable/source/examples/notebooks/models/compare-comsol-discharge-curve.html)
- SPM-type models fit measured LG M50LT impedance poorly at low frequency ("obtaining good fits at low frequencies turns out to be difficult") — [arXiv:2412.10896](https://arxiv.org/html/2412.10896v2)
- BOPTEST is a containerised benchmark for building HVAC control. One study uses the gap between a high-fidelity Modelica model and reduced-order (RC) models to stand in for the model error between predictive models and a real building. Differentiable surrogates of BOPTEST emulators exist (five-zone case) — [par.nsf.gov 10536472](https://par.nsf.gov/servlets/purl/10536472); [arXiv:2301.13447](https://arxiv.org/abs/2301.13447v2); [BOPTEST paper, LBL](https://eta-publications.lbl.gov/bibcite/reference/35036)

### Inferences
- **Battery pair (S-bat): nominal = PyBaMM SPM, true = SPMe or DFN.** The missing term is electrolyte transport (SPM drops it; SPMe adds it back). Its effect grows with C-rate, so r has a physical axis. You can also make a controllable ε by scaling the electrolyte-overpotential contribution, or by mixing (1−ε)·SPM + ε·SPMe as the "true" output, though mixing is less physical. θ = (C-rate, ambient temperature, one or two cell parameters). The measured Ecker curves (1C, 5C) allow an "is the true model closer to reality" check but are far too few for pool-based selection. Licence: PyBaMM is BSD-3 **[unverified]**. DFN solve time is roughly 0.1-1 s per discharge **[unverified]**, so 40k solves is about 1-10 CPU-h. Timeline risk is medium: PyBaMM solver quirks at high C-rate, and a new domain to explain.
- **NeuroBEM as a real-data "case study", not a main system.** Pool = recorded segments; "query" = revealing the measured residual of a segment; nominal = the provided model-predicted forces. 1 h 15 min at 400 Hz is about 1.8 million samples (my arithmetic). This tests RQ1 (does nominal-only uncertainty rank the real residual?) on data a reviewer cannot call a toy. Caveats: θ is the state/command, not system parameters, so the r-sweep is impossible (only one natural r). The licence is unknown, so email the authors before using it in a paper. File sizes are unknown.
- **EMPS / Industrial Robot (nonlinearbenchmark):** friction is the classic missing term, but there is only one physical system and few trajectories, so there is no θ variation. Hosting outages are a risk. Low value for this design.
- **IRIS:** potentially the best "real pendulum with measured damping/friction" source, but only 240 videos split across several systems, and the release status and licence are unconfirmed. Extracting states from video adds a perception error that is a confound. Skip unless the release turns out to include clean trajectories.
- **BOPTEST (RC model vs Modelica emulator):** a credible natural-mismatch pair, but Docker emulators and slow simulations make about 40k true queries unrealistic on 5 GB of RAM. High timeline risk; skip.
- **No real dataset supports the ε dial.** Real data can only add one or a few natural-r points. That fits the paper's framing as an "external validity check at one r", not a new arm of the sweep.

### Gaps
- Real double/single-pendulum datasets with varied length/damping (e.g. the IBM double-pendulum video dataset): not checked this session.
- Cart-pole / quadrotor sim-to-real datasets with physical parameter sweeps: none found.
- NeuroBEM licence, download size and which nominal model (BEM or rigid-body) produced columns 30-35: not stated on the readme.
- PyBaMM licence and DFN/SPMe timing: not verified this session.
- Fluid experiment datasets (e.g. the "Unsteady Fluid Mechanics" entry on nonlinearbenchmark): not opened.

## Q3. Which choice most raises acceptance odds at ML4PS / ICLR AI4Science, and later TMLR? What do recent papers use?

### Takeaway
Recent misspecification and multi-fidelity papers use canonical PDEs (Burgers, diffusion-reaction, cavity flow, Darcy, smoke inflow), not real data. ML4PS 2025 accepted several closure/subgrid and surrogate-UQ papers. A synthetic PDE with a dial therefore matches what the community already accepts and answers the "toy ODE" objection most directly. Real data helps mainly as an extra external-validity point.

### Cited Findings
- ML4PS 2025 (NeurIPS, 6 Dec 2025, San Diego) accepted, among others: "Uncertainty Quantification for Reduced-Order Surrogate Models Applied to Cloud Microphysics", "Physics-Guided Machine Learning For Uncertainty Quantification In Turbulence Models", "Differentiable Physics-Neural Models enable Learning of Non-Markovian Closures for Accelerated Coarse-Grained Physics Simulations", "Multiscale stochastic parameterization with deep Mori-Zwanzig formalism", "Continuous Representations of Baryonic Feedback for Robust Inference from Multiple Simulation Suites", "Adaptive Online Emulation for Accelerating Complex Physical Simulations", "Predictions and Corrections: Neural Predictors with Solver-based Correction for ODEs and PDEs", and two sim-to-survey domain-adaptation papers. Selection was by title only — [ML4PS 2025](https://ml4physicalsciences.github.io/2025)
- ML4PS submissions are up to 4 pages and anonymised (outside the D&B track) — [ML4PS 2025 guidelines](https://ml4physicalsciences.github.io/2025/guidelines.html)
- Testbeds in recent misspecification work: diffusion-reaction, Burgers, cavity flow, hyperelasticity (2606.03469); non-Newtonian channel and cavity flow (2310.10776); modified Darcy and smoke inflow (2608.04708) — sources as in Q1.

### Inferences
- For a late-Jan 2027 workshop deadline, the S1-only cut plus a small S3 (1D Burgers/KdV or reaction-diffusion, reduced grid) is the most valuable addition per hour. It directly addresses "toy ODEs" and uses PDEs that reviewers know from 2606.03469.
- For TMLR (March 2027): S1 + S2 + one PDE system run on the full pre-registered grid, plus one natural-mismatch case (SPM vs DFN battery, or the NeuroBEM case study for RQ1 only). Reviewers at TMLR care more about breadth of evidence than about "real data" for its own sake.
- Under the user's CLAUDE.md rule, this is a scope decision under deadline pressure, so the main agent should flag it before anything is built. Adding a PDE system before the S1 pilot passes risks the S1 timeline (the plan already puts S2 in the background in weeks 9-12). Recommended order: freeze S1, then run the S3 pilot (timing plus where r = 1 falls) in the week-4 window alongside the S1 pilot, then decide. Adding S3 is a deviation from the frozen pre-registration (section 3 defines only S1/S2). It must go in section 11 as an added exploratory or secondary system, or be pre-registered as a separate addendum before its pilot.

### Gaps
- I did not get ICLR 2026 AI4Science or ML4PS 2024 accepted lists, nor acceptance rates. I could not check whether any accepted paper sweeps mismatch strength.
- The ML4PS matches above are by title only; the abstracts were not read.

## Q4. Ranked testbeds with cost and timeline risk

### Takeaway
Ranked by value per hour for this study: (1) 1D Burgers → KdV-Burgers (or KS) with an ε-scaled missing term on Exponax; (2) 1D/2D reaction-diffusion with a missing reaction term on Exponax; (3) battery SPM vs SPMe/DFN in PyBaMM (natural r plus optional ε); (4) NeuroBEM real quadrotor residuals as a single-r RQ1 case study; (5) 2D Kolmogorov flow, coarse vs fine closure, as one natural-r exploratory point on the T4; (6, not recommended) EMPS/BOPTEST/IRIS.

### Cited Findings
- See Q1-Q2 for each source. The only measured runtime on record is the user's 0.0215 s pendulum solve (project_goal_v2.md). Exponax states only "50 2D KS trajectories under 1 s on a modern GPU" — [Exponax GitHub](https://github.com/Ceyron/exponax)

### Inferences (all costs are my estimates **[unverified]**; the pilot must measure them)

| Rank | Testbed | Nominal → true, dial | θ (surrogate input) | Source / licence | Sims cost | Fits (reduced grid ≈3.6k; full ≈25k) | Timeline risk |
|---|---|---|---|---|---|---|---|
| 1 | 1D Burgers → KdV-Burgers (+ε·u_xxx); alt. KS dropped term | ε dial, plus r via n_nom | ν, IC amplitude/wavenumber/phase | Exponax, MIT | ~0.1-1 CPU-h | ~10-60 CPU-h reduced / 70-420 full, with PCA outputs and an MLP | Low-medium (choosing the stepper; keep KS short-horizon to avoid chaos) |
| 2 | Fisher-KPP / Allen-Cahn missing reaction (ε·u³ or ε·u(1−u)) | ε dial | D, reaction rate, IC params | Exponax, MIT; matches 2606.03469 / 2603.24948 testbeds | ~0.1-1 CPU-h (1D) | as row 1 | Low-medium |
| 3 | Battery: SPM → SPMe/DFN | natural r rises with C-rate; optional ε scaling of electrolyte term | C-rate, temperature, 1-2 cell params | PyBaMM (BSD-3 unverified); Ecker data bundled | ~1-10 CPU-h (DFN 0.1-1 s each, unverified) | as row 1 | Medium (solver failures at high C-rate; new domain) |
| 4 | NeuroBEM real quadrotor | single natural r; residual columns provided | flight state/command per segment | rpg.ifi.uzh.ch; licence unknown | none (pool from recordings) | RQ1 only: ~300 fits | Medium (licence, data cleaning; no r sweep) |
| 5 | 2D Kolmogorov coarse → fine (closure) | natural r via resolution | Re, forcing params | Exponax, MIT | GPU; ~20 ms/traj or more (from Exponax KS figure) | T4-bound; only 1-2 cells | High |
| 6 | EMPS / BOPTEST / IRIS | single system or slow emulator or unclear release | — | see Q2 | — | — | High / not recommended |

- The trap in rows 1-2: if the surrogate is an MLP on raw fields (e.g. 128 × 50 outputs), fit times could exceed the 60 s ceiling the pre-registration uses to fall back to option C. Use PCA or low-mode outputs to keep fit cost close to the pendulum's.
- A cheap pilot to decide (about 1 day): implement row 1 in Exponax, time 1,000 solves on CPU, fit one 5-member ensemble at n_nom = 512, and bisect ε to r = 1. Then check that nominal-σ AUROC moves with ε at all (the same go/no-go as the S1 pilot).

### Gaps
- Every runtime in the table except the pendulum and the Exponax GPU figure is an estimate.
- No licence was confirmed for NeuroBEM, the PDEBench data, IRIS or PyBaMM.

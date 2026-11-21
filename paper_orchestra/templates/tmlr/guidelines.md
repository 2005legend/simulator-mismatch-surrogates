# TMLR Paper Writing Guidelines (for this project)

Below are the instructions for writing the LaTeX paper for Transactions on Machine Learning Research (TMLR), followed by rules specific to this study. The project rules take priority over anything else.

## General Setup
* **Document Class:** `\documentclass[10pt]{article}`
* **Required Packages:**
    ```latex
    \usepackage{tmlr}
    \input{math_commands.tex}
    \usepackage{hyperref}
    \usepackage{url}
    \usepackage{graphicx}
    \usepackage{booktabs}
    \usepackage{amsmath,amssymb}
    ```
* **Style Files:** use `tmlr.sty` and `tmlr.bst`. Do not pass `accepted` or `preprint` to the package in the submission version.

## Anonymity (Crucial)
* TMLR is double-blind. Keep the placeholder author block `\name Anonymous Authors \email anonymous@example.com \\ \addr Anonymous Institution` unchanged.
* Do not mention author names, institutions, the project's GitHub repository, or the venue of any related talk.
* Refer to the code as "code will be released on acceptance" or an anonymised link to be added by the authors.

## Length and Structure
* TMLR has no strict page limit; aim for about 12-14 pages of main text. Long proofs, extra tables and the full pre-registration trail go in the appendix after `\appendix`.
* The abstract is one paragraph.
* Optional unnumbered sections before the references: `\subsubsection*{Broader Impact Statement}` (recommended), `\subsubsection*{Author Contributions}`, `\subsubsection*{Acknowledgments}`.

## Formatting
* Do not change margins, font sizes or the text block.
* Headings: `\section`, `\subsection`, `\subsubsection`.
* Citations: natbib is loaded by `tmlr.sty`. Use `\citet{key}` for "Author (Year)" and `\citep{key}` for "(Author, Year)".
* Bibliography: `\bibliography{references}` and `\bibliographystyle{tmlr}`.
* Figures: caption below; tables: caption above. Use `booktabs` rules (`\toprule`, `\midrule`, `\bottomrule`).
* All figures are in the `figures/` directory. Always write `\includegraphics[width=...]{figures/<name>.png}` with the exact names from the figures list.

## Project rules (must follow)

1. **Numbers.** Every number in the paper must appear exactly in `experimental_log.md`. Do not compute new statistics, round differently in a way that changes meaning, or add results, datasets, baselines or ablations that are not in the log.
2. **Order of results.** Report the primary pre-registered hypotheses (H1-H4, Families A and B, C1, Family F) first, with their verdicts as written: supported, not supported, opposite, inconclusive. Then report the secondary predictions (D-tests, alignment range, PDE tests), and say plainly that the main claim was added in addendum v1.5 after a development-seed look and then confirmed on main seeds and on two PDE systems frozen in advance.
3. **Negative results.** State in the abstract and the conclusion that guided acquisition never beat a Sobol design, that the pre-registered acquisition hypotheses were not supported or opposite, and that transition points predicted from one pilot seed matched the 20-seed interval in 1 of 6 cases.
4. **Theory.** Only the triangle-inequality bound in the idea file may be stated, as a proposition with its short proof. Do not state further theorems, rates or guarantees.
5. **Wording.** Use "pre-registered", "secondary", "dev-informed", "opposite direction" as defined in the log. Do not call any result "significant" unless its Holm-adjusted p is below 0.05 and its verdict is "supported". Do not use the words novel, groundbreaking, pivotal, crucial, leverage, delve, robust (except for statistics), or state-of-the-art.
6. **Citations.** Cite only papers found by the literature search with a verified title. Never invent a reference. Related areas to cover: model discrepancy and calibration of computer models (Kennedy and O'Hagan), deep ensembles and MC dropout, conformal prediction, physics-informed neural networks and PDE residuals, neural operators (Fourier neural operator), active learning and Bayesian experimental design for surrogates, multi-fidelity learning, uncertainty under distribution shift.
7. **Limitations section.** Required. Include: controlled synthetic "true" systems only; the main claim was set after a dev look; Family F used a cost-limited fit budget (20 L-BFGS iterations, fit-to-fit CV up to 0.39); the PDE dictionary was written knowing the true terms and its P2 terms are nearly collinear; the operator gate was checked on one development seed; analyses cut before results are listed as not run.
8. **Disclosure.** Add to the acknowledgments: "Parts of the text were drafted with an AI writing tool (PaperOrchestra) and then checked and revised by the authors."

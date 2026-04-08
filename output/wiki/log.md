# Wiki Log

## [2026-04-07] init | Project initialization
Created wiki structure. Seeding concept pages from THE_PARADOX.md survey. Overview page created with central paradox, existing theories table, and research plan.

## [2026-04-07] seed | 12 concept pages created
All concept pages seeded from THE_PARADOX.md: NTK, mean field, spin glass, mode connectivity, implicit bias, edge of stability, neural collapse, overparameterization, conservation laws, tropical geometry, RMT, energy landscape.

## [2026-04-07] angles | 7 candidate angles documented
Created angle pages for all 7 theoretical candidates. Top 3 selected: Noether Conservation Laws, Percolation Phase Transition, Tropical Morse Theory.

## [2026-04-07] literature | 26+ papers ingested (2024-2026)
Literature search completed across 6 threads. Critical findings:
- Marcotte et al. (2023-2025) classified all conservation laws -- Theory A pivoted
- Ferbach et al. (2024) proved LMC via optimal transport -- Theory B remains novel
- Brandenburg et al. (2024) tropical geometry framework -- Theory C must build on this
Full details in literature/relevant-papers.md.

## [2026-04-07] experiment | Conservation laws verified (Domain 3)
Ran 6 configs x 5 seeds. Conservation drift <0.01% for bias-free networks. Bias correctly breaks conservation. Drift scales linearly with learning rate.

## [2026-04-07] experiment | Mode connectivity universal (Domain 2)
Zero barriers at all widths (2-512) for 2-layer networks on Gaussian data. Even undertrained models (80% accuracy) are linearly connected.

## [2026-04-07] experiment | Hessian spectrum evolution (Domain 1)
Bulk-plus-outliers structure confirmed. Negative eigenvalues persist but shrink 100x. Spectrum concentrates near zero during training.

## [2026-04-07] experiment | EoS not observed on easy data (Domain 4)
Gaussian mixture and MNIST converge too fast for progressive sharpening. lambda_max peaks early and decreases. Need harder settings.

## [2026-04-07] pivot | Theory A reframed
From "conservation laws constrain trajectory" to "structured conservation law breaking as the mechanism of edge-of-stability optimization." Motivated by Ghosh et al. (ICLR 2025) and Jiang et al. (NeurIPS 2025).

## [2026-04-07] experiment | EoS deep network -- LANDMARK RESULT
4-layer MLP on MNIST with MSE loss shows progressive sharpening AND conservation law breaking that correlates perfectly with EoS intensity. Drift: 0.002 (sub-EoS) to 10.99 (deep EoS). This is the central computational finding.

## [2026-04-07] experiment | Deep network connectivity (Theory B)
4-layer MLPs show barriers (0.4-1.6). Permutation alignment reduces barriers by 39-86% with reduction increasing monotonically with width. Supports percolation picture.

## [2026-04-07] experiment | PL critical cells (Theory C)
Critical cells are 0.2-0.4% of boundaries and decrease with width (37 -> 19). Supports tropical Morse prediction.

## [2026-04-07] quality-gate | Theory A PASSES 8/8
Complete proof of Theorem 1 (conservation laws), strong evidence for Theorem 2 (structured breaking at EoS), 4 experiments across multiple settings and learning rates.

## [2026-04-07] theories | Formal documents complete
All three theory documents written following the template: theory-1-conservation-laws.md (22KB), theory-2-percolation-connectivity.md (18KB), theory-3-tropical-morse.md (17KB).

## [2026-04-07] experiment | Drift scaling law measured
Conservation drift ~ lr^{1.16} across 4 decades (10 learning rates, 5 seeds). Sub-quadratic due to faster convergence at large lr.

## [2026-04-07] experiment | Falsification tests (Theory A)
7 tests: ReLU/LeakyReLU nobias conserved, tanh/sigmoid broken, GELU approximately conserved, Adam breaks conservation. 5/7 predictions matched.

## [2026-04-07] experiment | Feature learning (Domain 6)
CKA: 0.87 (width 16, feature learning) to 0.999 (width 512, lazy). Conservation drift correlates with feature learning intensity.

## [2026-04-07] experiment | Fine width sweep (Theory B)
IMPORTANT: Transition is GRADUAL (exponential decay, tau=30), not sharp sigmoid. Barrier floor at 0.20. Theory B revised to continuous percolation.

## [2026-04-07] experiment | Activation regions (Theory C, MNIST)
Training reduces activation patterns and increases margins 6-15x. Supports claim that GD avoids PL critical cells.

## [2026-04-07] quality-gate | FINAL ASSESSMENT
Theory A: 8/8 PASS. Theory B: 7/8. Theory C: 7/8. Synthesis document complete.

## [2026-04-07] experiment | Session 5: Spectral Prediction (E8)
Theorem 5 evaluated on actual Hessian eigenvalues predicts S(eta) with 14-27% error for ReLU (R=0.808) and 14-18% for linear (R=0.998). The spectral crossover formula is structurally universal. Hessian spectra differ: ReLU has 1104 positive eigenvalues (max 5.8), linear has 762 (max 18.3).

## [2026-04-07] experiment | Session 5: Activation Coupling (E9)
Only 2.2% of neurons change activation per step. 34% of steps have zero changes. Mode coupling is sparse. Strong correlation (0.47-0.91) between activation changes and gradient imbalance.

## [2026-04-07] experiment | Session 5: Interpolated Activation (E11) -- KEY FINDING
With same loss (MSE), alpha increases from 1.11 (linear) to 1.29 (ReLU). Session 4's "identical alpha" was a COMPENSATING EFFECTS coincidence: ReLU increases alpha by ~0.19, CrossEntropy decreases by similar amount. Three factors independently determine alpha: spectrum, activation, loss.

## [2026-04-07] theory | Session 5: Theorem 6 (Perturbative Stability)
First-order expansion shows ReLU modifies c_k but preserves S(eta) functional form. Correction bounded by activation switch rate (~2.2%/step). Theory A Section 7.11 added with 5 subsections.

## [2026-04-07] theory | Session 8: Theorem 5b PROVED (Spectral Compression)
CE Hessian H_CE(t) = J^T S(p(t)) J factorizes with softmax matrix S. As training proceeds, softmax concentrates (q_i → 1), making S → 0 and compressing the Hessian spectrum. Proved: lambda_max(H_CE(t)) ≤ lambda_max(J^TJ) · max_i[q_i(1-q_i)], with exponential decay. Added as Section 7.17 with 7 subsections.

## [2026-04-07] experiment | Session 8: E18 — CE Hessian Evolution Validation (431s)
Validated spectral compression: 24x drop in max eigenvalue (7.2 → 0.3). Key surprise: decay rate is n-independent (b ~ 0.005 for n=100, 200, 400). Softmax dynamics identical across n_train in overparameterized regime. Revised τ formula: τ ≈ C/η, independent of n.

## [2026-04-07] theory | Session 8: c_k Derived from First Principles
For 2-layer linear networks, c_k ∝ e_k(0)² · λ_{x,k}² where e_k is initial error and λ_{x,k} is data covariance eigenvalue. Added as Section 7.19.

## [2026-04-07] experiment | Session 8: E20 — c_k Validation (8.2s)
Theoretical c_k matches empirical at R = 0.847. Hessian eigenspectrum tracks data covariance at R = 0.88. Top data mode captures 50-69% of total c_k weight, confirming quadratic λ_{x,k} dependence.

## [2026-04-07] experiment | Session 8: E19 — MSE Fine Width Sweep (257s)
Alpha grows from 1.05 (width 16) to 1.64 (width 192). Scaling: alpha-1 ~ width^1.18 (R²=0.93). Power-law model breaks down at large widths: R² degrades to 0.887, curvature increases 13x. Two regimes: narrow (m≤32) and wide (m≥48). Transition at width ~40 ≈ 2*input_dim.


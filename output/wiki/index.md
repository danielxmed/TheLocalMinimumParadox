# Wiki Index

## Overview
- [Research Overview](overview.md) -- Central paradox, existing theories, our novel angles, research status

## Concept Pages
- [Neural Tangent Kernel](concepts/neural-tangent-kernel.md) -- NTK theory: infinite-width linearization, lazy training, convergence guarantees
- [Mean Field Theory](concepts/mean-field-theory.md) -- Wasserstein gradient flow, convexity in measure space, 2-layer limitation
- [Spin Glass Landscape](concepts/spin-glass-landscape.md) -- Kac-Rice formula, Bray-Dean theorem, bad minima exponentially rare
- [Mode Connectivity](concepts/mode-connectivity.md) -- Connected minima, Entezari conjecture, Git Re-Basin, linear mode connectivity
- [Implicit Bias](concepts/implicit-bias.md) -- Max-margin convergence, flat minima, SAM, Dinh critique
- [Edge of Stability](concepts/edge-of-stability.md) -- lambda_max -> 2/eta, progressive sharpening, self-tuning dynamics
- [Neural Collapse](concepts/neural-collapse.md) -- Simplex ETF, Papyan-Han-Donoho discovery, global optimum characterization
- [Overparameterization](concepts/overparameterization.md) -- Lazy vs rich regimes, interpolation threshold, NTK vs mean field scaling
- [Conservation Laws](concepts/conservation-laws.md) -- Balancedness invariant, symmetry teleportation, gradient flow invariants
- [Tropical Geometry](concepts/tropical-geometry.md) -- PL structure of ReLU, linear regions, Newton polytopes, tropical varieties
- [Random Matrix Theory](concepts/random-matrix-theory.md) -- Hessian anatomy, bulk-plus-outliers, spectral universality
- [Energy Landscape](concepts/energy-landscape.md) -- Single funnel hypothesis, protein folding analogy, disconnectivity graphs

## Candidate Angles (Top 3 selected for deep dive marked with *)
- [Noether Conservation Laws](angles/noether-conservation-laws.md)* -- Theory A: symmetries -> conserved quantities -> quasi-convex submanifold
- [Percolation Phase Transition](angles/percolation-phase-transition.md)* -- Theory B: sharp width threshold for sublevel set connectivity
- [Tropical Morse Theory](angles/tropical-morse-theory.md)* -- Theory C: PL Morse theory for ReLU landscape critical cells
- [Edge of Stability as SOC](angles/edge-of-stability-soc.md) -- Angle 4: self-organized criticality at lambda_max = 2/eta
- [RG Flow Along Depth](angles/renormalization-group-depth.md) -- Angle 5: layers as coarse-graining RG transformations
- [Information-Geometric Convexity](angles/information-geometric-convexity.md) -- Angle 6: Fisher-Rao metric makes landscape geodesically convex
- [Data-Dependent Landscape](angles/data-dependent-landscape.md) -- Angle 7: manifold hypothesis reduces effective landscape dimension

## Literature (2024-2026)
*(To be populated during literature search)*

## Synthesis
- [Unified Picture](../theories/synthesis-unified-picture.md) -- How all three theories connect to resolve the paradox

## Experiment Results
- Conservation law verification: drift < 0.003% for nobias networks (5 configs x 5 seeds)
- Conservation drift scaling: drift ~ lr^1.16 across 4 decades (10 learning rates x 5 seeds)
- EoS + conservation breaking: drift 0.002 (sub-EoS) to 10.99 (deep EoS) -- 5500x correlation
- Deep connectivity: barriers 0.4-1.6 with 39-86% reduction after permutation alignment
- Width sweep: universal connectivity for 2-layer (all widths), barriers for 4-layer
- Hessian spectrum: bulk-plus-outliers confirmed, negative eigenvalues shrink 100x during training
- PL critical cells: 0.2-0.4% of boundaries, decreasing with width
- Activation regions: training reduces patterns and increases margins 16x
- **[Session 3] Universality of drift exponent:** alpha ≈ 1.1 stable across widths (16-256) and datasets, increases with depth (1.1→1.7), fundamentally different for Adam (0.6)
- **[Session 3] Grokking + conservation:** Negative result -- 2-layer MLP lacks capacity for modular arithmetic. Requires embedding layers.
- **[Session 5] Spectral prediction (E8):** Theorem 5 predicts S(eta) with 14-27% error for ReLU using the Hessian spectrum at initialization. Log-log correlation R=0.998 (linear), R=0.808 (ReLU). Formula is structurally universal.
- **[Session 5] Activation coupling (E9):** Only 2.2% of neurons change activation per step. 34% of steps have zero changes. Mode coupling is sparse.
- **[Session 5] Interpolated activation (E11):** Alpha increases from 1.11 (linear) to 1.29 (full ReLU) with same MSE loss. Session 4's "identical alpha" was a loss-function compensation coincidence.
- **[Session 6] 4-Way Hessian (E12):** {MSE,CE}x{Linear,ReLU} shows alpha = 1.116, 1.135, 1.276, 1.089. CE barely affects linear (+0.019) but decreases ReLU alpha by 0.188. INTERACTION, not additive.
- **[Session 6] Interpolated loss (E13):** Alpha smoothly transitions from 1.293 (MSE) to 1.076 (CE) for ReLU. 2D grid confirms CE effect scales with activation nonlinearity.
- **[Session 6] Width dependence (E14):** Mode coupling does NOT vanish with width [16-256]. Contradicts Theorem 6's O(1/sqrt(m)) prediction. Mode coupling is fundamental.
- **[Session 7] Width switch rate (E15):** Per-neuron activation switch rate is width-INDEPENDENT at EoS (beta=0.03). Total mode coupling scales as O(m) — extensive quantity. Perturbative framework applies only sub-EoS.
- **[Session 7] Time-dependent Hessian (E16):** CE Hessian compresses 40x during training (max_eig: 5.78->0.14). Using H(t=250) gives R=0.988 for CE prediction (vs 0.808 at t=0). Theorem 5 is UNIVERSAL when appropriate Hessian is used. MSE prediction worsens with later Hessian (best at t=0).
- **[Session 7] Interaction width (E17):** CE clamps alpha near 1.0-1.1 regardless of width. MSE alpha diverges (up to 1.63 at width 128). The "interaction" is CE's spectral self-regularization, not a constant correction term.
- **[Session 8] CE Hessian evolution (E18):** Spectral compression validated: 24x drop in max eigenvalue. Decay rate n-independent (b~0.005). Softmax dynamics identical across n_train={100,200,400} in overparameterized regime. Supports Theorem 5b.
- **[Session 8] c_k validation (E20):** Theoretical c_k ∝ e_k² · λ_{x,k}² matches empirical c_k at R=0.847 for linear networks. Hessian eigenspectrum tracks data covariance at R=0.88. First-principles derivation VALIDATED.
- **[Session 8] MSE fine width sweep (E19):** Alpha grows from 1.05 to 1.64 across widths 16-192 as alpha-1 ~ width^1.18. Power-law model breaks down at large widths (R² drops to 0.887, curvature increases 13x). Two regimes: narrow (m≤32, α~1.05) and wide (m≥48, rapid growth).

## Dead Ends
- EoS on easy data (Gaussian mixture, MNIST subsets): networks converge too fast for progressive sharpening
- Phase transition search for 2-layer networks: threshold is below width=2 for all tested data distributions
- Smooth Morse theory: fundamentally incorrect for ReLU networks (motivates Theory C)
- **[Session 3] Grokking on 2-layer MLP with one-hot encoding:** Cannot learn modular addition (train_acc ≈ random). Needs embedding layers + deeper architecture.

# Paper Structure: Conservation Law Breaking at the Edge of Stability

**Target venue:** NeurIPS/ICML theory track (initial arxiv submission)
**Format:** ~9 pages + appendix, 5 composite figures
**Title (working):** Conservation Law Breaking at the Edge of Stability: A Spectral Theory of Non-Convex Neural Network Optimization

---

## Abstract (~150 words)

Key messages:
1. The paradox: non-convex landscape is NP-hard worst-case, yet SGD works brilliantly
2. Discovery: gradient flow preserves conservation laws C_l = ||W_{l+1}||^2 - ||W_l||^2
3. At Edge of Stability, these laws break with drift ~ eta^alpha (alpha ~ 1.1-1.6)
4. Alpha explained from first principles via spectral crossover formula S(eta) = sum_k c_k f(eta, lambda_k, T)
5. c_k derived: c_k proportional to e_k^2 * lambda_{x,k}^2 (validated R=0.85 for both linear and ReLU)
6. CE self-regularization: softmax compression makes tau = C/eta (n-independent)
7. Width transition at m ~ 2d separates perturbative from non-perturbative regimes

---

## Section 1: Introduction (1.5 pages)

**Opening:** The central paradox — loss landscape of neural networks is provably NP-hard in worst case, yet simple gradient descent finds good solutions with remarkable reliability.

**Our resolution:** Conservation laws act as "guide rails" constraining optimization dynamics. Their BREAKING at the Edge of Stability is the key mechanism that enables escape from bad minima.

**Contributions (one paragraph):**
- Theorems 1-5b: rigorous mathematical framework for conservation laws, their breaking, and the spectral crossover formula
- First-principles derivation of c_k coefficients (parameter-free prediction)
- Proof that CE self-regularizes via spectral compression (tau = C/eta)
- Width-dimension transition at m ~ 2d separating two dynamical regimes
- 23 experiments validating every prediction

**Figure 1:** Overview figure — conservation law drift vs eta showing power-law alpha, with inset showing the EoS regime.
Use: fig5e_drift_scaling (or create composite from fig5 + fig5e)

---

## Section 2: Conservation Laws and Their Breaking (1.5 pages)

**Content:**
- Setup: L-layer homogeneous networks without bias, gradient flow
- **Theorem 1:** C_l = ||W_{l+1}||^2 - ||W_l||^2 is exactly conserved under gradient flow
- Discrete-time breaking: gradient descent introduces drift
- **Theorem 3:** Per-step drift decomposition: Delta C_l = eta^2 * [||dL/dW_{l+1}||^2 - ||dL/dW_l||^2]
- Definition of the gradient imbalance sum S(eta) = sum_t delta_t(theta_t)
- Total drift = eta^2 * S(eta)
- Key insight: S(eta) encodes HOW conservation laws break — it depends on the spectral structure of the loss Hessian

**Experiments:** E1 (conservation verification: drift < 0.003%), E4 (drift scaling over 4 decades)

**Figure 2:** 2-panel composite
- (a) Conservation law verification: C_l stays constant under gradient flow (fig5)
- (b) Drift ~ eta^alpha scaling over 4 decades (fig5e)

---

## Section 3: Spectral Crossover Formula (2 pages)

**Content:**
- **Theorem 4:** Linear networks give alpha = 1.10 (spectral, not from nonlinearity)
- The crossover question: WHY alpha > 1?
- **Theorem 5:** Spectral crossover formula S(eta) = sum_k c_k * (1 - rho_k^{2T}) / (eta * lambda_k * (2 - eta * lambda_k))
  - Where rho_k = 1 - eta * lambda_k, and lambda_k are Hessian eigenvalues
  - The formula is EXACT for linear networks, approximate for ReLU
- **c_k Derivation (Theorem):** For 2-layer linear networks via SVD mode decomposition:
  - c_k proportional to e_k(0)^2 * lambda_{x,k}^2
  - First closed-form, parameter-free prediction for spectral mode weights
  - Top data covariance eigenmode captures 50-69% of total c_k weight
- **ReLU Extension (Session 9):** The linear formula generalizes to ReLU at all tested learning rates with R > 0.80

**Experiments:** E8 (structural universality, 14-27% error), E20 (linear c_k validation, R=0.847), E21 (ReLU c_k validation, R=0.85)

**Figure 3:** 3-panel composite
- (a) Spectral crossover prediction vs measured S(eta) (fig12 spectral prediction)
- (b) Linear c_k validation scatter (fig23 panel a)
- (c) ReLU c_k validation: R vs learning rate (fig24 panel b)

---

## Section 4: Time-Dependent Universality and CE Self-Regularization (2 pages)

**Content:**
- The MSE-CE dichotomy: why does CE keep alpha near 1.0 regardless of width?
- **Theorem 5b:** CE Hessian spectral compression
  - H_CE(t) = J^T S(p(t)) J, where S is the softmax-derived matrix
  - As training proceeds, q_i -> 1, so S -> 0 exponentially
  - lambda_max(H_CE(t)) <= lambda_max(J^T J) * max_i[q_i(1-q_i)]
- **Spectral compression timescale (Section 7.20):**
  - tau = C/eta (n-INDEPENDENT)
  - Proof via NTK: overparameterization ensures lambda_min(K) = Theta(1)
  - Cross-kernel contribution makes per-sample learning rate n-independent
  - Validated by E18 (n-independence) and E23 (tau proportional to 1/eta)
- Universality: the spectral crossover formula with time-dependent c_k(t) works for ANY loss function

**Experiments:** E16 (time-dependent Hessian, CE R=0.988), E18 (24x compression, n-independent decay), E23 (tau vs 1/eta)

**Figure 4:** 3-panel composite
- (a) CE Hessian eigenvalue decay at different n (fig21 panel a)
- (b) Softmax concentration drives compression (fig21 panel d)
- (c) tau vs 1/eta validation (fig26 panel b)

---

## Section 5: Edge of Stability Dichotomy and Width Scaling (1.5 pages)

**Content:**
- **Theorem 6' (Revised):** Two dynamical regimes
  - Sub-EoS: per-neuron switch rate ~ m^{-0.5}, total coupling O(sqrt(m)), perturbative
  - At EoS: per-neuron switch rate width-independent, total coupling O(m), non-perturbative
- Width scaling of alpha: alpha - 1 ~ width^1.18 for MSE (E19)
- Power-law model breaks down at large widths (R^2 degrades 0.999 -> 0.887)
- **Width-dimension transition (Session 9):** Transition at m ~ 2d
  - Coincides with overparameterization threshold
  - Below: perturbative regime, simple power law holds
  - Above: non-perturbative, extensive mode coupling dominates
- CE CLAMPS alpha near 1.0-1.1 regardless of width (E17) — spectral compression prevents large alpha

**Experiments:** E15 (width switch rate), E17 (CE clamping), E19 (MSE fine width sweep), E22 (width-dimension transition)

**Figure 5:** 3-panel composite
- (a) Alpha vs width for MSE, showing divergence (fig22 panel a)
- (b) Alpha vs width/d for d=10,20,40 — curve collapse (fig25 panel a)
- (c) Transition width m* vs d (fig25 panel c)

---

## Section 6: Discussion (0.5 pages)

**Key messages:**
1. CE self-regularization: softmax compression is a BUILT-IN mechanism that prevents pathological optimization
2. Width scaling reveals two regimes, connecting to NTK vs feature learning
3. Implication for practice: learning rate scheduling should respect the EoS boundary
4. Open problems:
   - c_k for ReLU at EoS with extensive mode coupling
   - Extension beyond 2-layer networks
   - Connecting conservation law breaking to generalization
   - Tropical Morse theory and percolation connections (Theories B, C)

---

## Appendix

### A. Full Proofs
- Theorem 1: Conservation law derivation
- Theorem 2': Mean-field quasi-convexity (2-layer, infinite width)
- Theorem 3: Per-step drift decomposition
- Theorem 4: Linear network spectral analysis
- Theorem 5: Spectral crossover formula derivation
- Theorem 5b: CE Hessian factorization and spectral compression
- c_k Theorem: SVD mode decomposition for linear networks

### B. Extended Experimental Results
- Complete parameter configurations for all 23 experiments
- Per-seed results tables
- Additional figures for Theories B and C

### C. Reproducibility
- Code availability (GitHub link)
- Hardware specifications (Intel i5-1038NG7, CPU-only, PyTorch 2.2.2)
- All seeds: [42, 137, 256, 512, 1024]

---

## Figure Budget (5 figures in main paper)

| Figure | Content | Source |
|--------|---------|--------|
| 1 | Overview: drift scaling + conservation verification | fig5 + fig5e |
| 2 | Spectral crossover + c_k validation | fig12 + fig23 + fig24 |
| 3 | CE spectral compression + tau scaling | fig21 + fig26 |
| 4 | Width scaling + dimension transition | fig22 + fig25 |
| 5 | [Optional] EoS switch rate + CE clamping | fig18 + fig20 |

---

## Notation Standardization Checklist

These must be consistent throughout the paper:
- theta: network parameters (all weights)
- W_l: weight matrix of layer l
- C_l = ||W_{l+1}||_F^2 - ||W_l||_F^2: conservation quantity
- eta: learning rate
- alpha: drift exponent (drift ~ eta^alpha)
- S(eta): gradient imbalance sum
- lambda_k: k-th Hessian eigenvalue
- c_k: spectral crossover mode weight
- e_k: initial error in mode k
- lambda_{x,k}: k-th data covariance eigenvalue
- tau: spectral compression timescale
- q_i: correct-class probability for sample i
- J_i: Jacobian for sample i
- H_CE(t): time-dependent CE Hessian
- S(p): softmax-derived matrix
- K or Theta: NTK matrix
- m: hidden layer width
- d: input dimension
- n: number of training samples
- K (italic): number of classes
- T: number of training steps

---

## Timeline for Paper Writing (Session 10)

1. Set up arxiv_submission/ directory structure
2. Write main.tex following this structure
3. Compile bibliography from output/literature/relevant-papers.md
4. Select and compose 5 composite figures
5. Write appendix with full proofs
6. Prepare GitHub repository for public release

# Relevant Papers (2024-2026)

## Search Status
- [x] Thread 1: Feature learning beyond NTK
- [x] Thread 2: Edge of stability
- [x] Thread 3: Mode connectivity
- [x] Thread 4: Conservation laws
- [x] Thread 5: Tropical/algebraic geometry
- [x] Thread 6: Overparameterization phase transitions

All citations verified via arXiv, Semantic Scholar, or conference proceedings.

---

## CRITICAL PAPERS (directly impact our theories)

### Conservation Laws (Theory A)

1. **Marcotte, Gribonval, Peyre.** "Abide by the Law and Follow the Flow: Conservation Laws for Gradient Flows." NeurIPS 2023. arXiv:2307.00144.
   - *Systematic classification framework. State of the art. Finds ALL conservation laws via Lie algebra.*
   - **Impact**: Our Theory A must go beyond classification to explain *why* conservation laws matter for optimization.

2. **Marcotte, Gribonval, Peyre.** "Keep the Momentum: Conservation Laws beyond Euclidean Gradient Flows." ICML 2024. arXiv:2405.12888.
   - *No conservation laws for ReLU + momentum. Temporal dependence in momentum case.*
   - **Impact**: CRITICAL CONSTRAINT. Practical optimizers break all conservation laws.

3. **Marcotte, Gribonval, Peyre.** "Transformative or Conservative? Conservation Laws for ResNets and Transformers." ICML 2025. arXiv:2506.06194.
   - *Complete classification for modern architectures. Skip connections don't create new laws.*

4. **Ghosh, Kwon, Wang, Ravishankar, Qu.** "Learning Dynamics of Deep Matrix Factorization Beyond the Edge of Stability." ICLR 2025. arXiv:2502.20531.
   - *Balancedness BREAKS at EoS. Period-doubling route to chaos. Decay to zero.*
   - **Impact**: PIVOTAL. Conservation law breaking IS the mechanism, not a failure.

5. **Kunin, Sagastuy-Brena, Ganguli, Yamins, Tanaka.** "Neural Mechanics: Symmetry and Broken Conservation Laws in Deep Learning Dynamics." ICLR 2021. arXiv:2012.04728.
   - *Foundational Noether-like framework. Finite learning rates break conservation.*

6. **Zhao, Ganev, Walters, Yu, Dehmamy.** "Symmetries, Flat Minima, and the Conserved Quantities of Gradient Flow." ICLR 2023. arXiv:2210.17216.
   - *Connects conservation to flat minima and convergence rate. Data-dependent symmetries.*

### Edge of Stability (connects to Theory A)

7. **Jiang, Cohen, Li.** "Understanding the Evolution of the NTK at the Edge of Stability." NeurIPS 2025. arXiv:2507.12837.
   - *EoS improves NTK-target alignment. Feature learning IS enhanced by EoS.*
   - **Impact**: Key bridge -- if conservation law breaking enables NTK improvement, this is THE mechanism.

8. **Islamov, Crawshaw, Cohen, Gower.** "Non-Euclidean Gradient Descent Operates at the Edge of Stability." arXiv:2603.05002 (March 2026).
   - *EoS is universal across geometries. Not algorithm-specific.*

9. **Damian, Nichani, Lee.** "Self-Stabilization: The Implicit Bias of GD at the Edge of Stability." ICLR 2023. arXiv:2209.15594.
   - *Cubic feedback mechanism. GD implicitly follows projected GD under S(theta) <= 2/eta.*

10. **Liang, Cloninger, Parhi, Wang.** "Generalization Below the Edge of Stability: The Role of Data Geometry." ICLR 2026. arXiv:2510.18120.
    - *EoS regularization is data-dependent. "Data shatterability" principle.*

### Mode Connectivity (Theory B)

11. **Ferbach, Goujaud, Gidel, Dieuleveut.** "Proving Linear Mode Connectivity via Optimal Transport." AISTATS 2024. arXiv:2310.19103.
    - *First rigorous proof of LMC for finite-width networks. Different mechanism from our percolation.*

12. **Vrabel, Shem-Ur, Oz, Krueger.** "Input Space Mode Connectivity in Deep Neural Networks." arXiv:2409.05800 (2024).
    - *Conjectures percolation for mode connectivity in INPUT space (not parameter space).*
    - **Impact**: Must cite and distinguish. We do parameter space with rigorous threshold.

13. **Zhao, Dehmamy, Walters, Yu.** "Understanding Mode Connectivity via Parameter Space Symmetry." ICML 2025. arXiv:2505.23681.
    - *Symmetry group topology explains connectivity. Explicit connecting curves.*

14. **Ersoy, Cardozo Licha, Wiesner.** "Phase Transitions Reveal Hierarchical Structure in DNNs." arXiv:2512.11866 (Dec 2025).
    - *Phase transitions linked to mode connectivity via saddle points.*

15. **Boursier, Bowditch, Englert, Lazic.** "Benignity of Loss Landscape with Weight Decay Requires Both Large Overparametrization and Initialization." arXiv:2505.22578 (May 2025).
    - *Width >= min(n^d, 2^n) for no spurious minima. Much larger than our Theta(n*d).*
    - **Impact**: Connectivity (our claim) may emerge at LOWER threshold than full benignity.

### Tropical/Algebraic Geometry (Theory C)

16. **Brandenburg, Loho, Montufar.** "The Real Tropical Geometry of Neural Networks." TMLR 2024. arXiv:2403.11871.
    - *Classification fan framework. 0/1-loss sublevel sets are NOT necessarily connected.*
    - **Impact**: Must build on this. Non-connectivity of sublevel sets constrains our claims.

17. **Lezeau, Walker, Cao, Bhatia, Monod.** "Tropical Expressivity of Neural Networks." NeurIPS 2024. arXiv:2405.20174.
    - *OSCAR library for computing tropical representations. Computational tools.*

18. **Grigsby, Lindsey, Masden.** "Local and Global Topological Complexity Measures of ReLU Neural Network Functions." arXiv:2204.06062 (revised 2024).
    - *PL Morse theory for ReLU. Critical cells. Canonical polytopal complex.*
    - **Impact**: Most direct prior art. We must extend this to loss landscapes (not just network functions).

19. **ICML 2025 position paper.** "Algebra Unveils Deep Learning -- An Invitation to Neuroalgebraic Geometry." arXiv:2501.18915.
    - *Broader vision for algebraic geometry in ML.*

### Feature Learning / NTK-Mean Field Bridge

20. **Chen, Yang, Zhao, Gu.** "Global Convergence and Rich Feature Learning in L-Layer Infinite-Width Networks under muP." ICML 2025. arXiv:2503.09565.
    - *LANDMARK: First simultaneous feature learning + global convergence for deep networks.*

21. **Yang, Yu, Zhu, Hayou.** "Tensor Programs VI: Feature Learning in Infinite-Depth Networks." ICLR 2024. arXiv:2310.02244.
    - *Depth-muP extends muP to deep resnets.*

22. **Noci, Meterez et al.** "Why Do Learning Rates Transfer?" arXiv:2402.17457 (Feb 2024).
    - *Sharpness independence under muP. Bridges EoS and feature learning.*

23. **Domine et al.** "From Lazy to Rich: Exact Learning Dynamics in Deep Linear Networks." ICLR 2025. arXiv:2409.14623.
    - *Exact solutions across the full lazy-to-rich spectrum.*

### Overparameterization Phase Transitions

24. **Montanari, Wang.** "Phase Transitions for Feature Learning in Neural Networks." arXiv:2602.01434 (Feb 2026).
    - *Sharp threshold for feature learning. Spectral phase transition in Hessian.*

25. **Annesi, Bocchi, Cammarota.** "Overparametrization Bends the Landscape: BBP Transitions at Initialization." arXiv:2510.18435 (Oct 2025).
    - *BBP spectral phase transition in data-to-parameter ratio.*

26. **Ly, Gong.** "Optimization on Multifractal Loss Landscapes." Nature Communications 16, 3252 (2025).
    - *Multifractal model unifying EoS, anomalous diffusion, degenerate minima.*

---

## NOVELTY ASSESSMENT SUMMARY

| Theory | Prior Art Level | Novel Contribution | Risk |
|--------|----------------|-------------------|------|
| A: Conservation Laws | HIGH (Marcotte et al. 2023-2025 classified all laws) | **PIVOT**: Conservation law BREAKING as optimization mechanism | Medium |
| B: Percolation | LOW (Vrabel et al. input-space conjecture only) | Parameter-space percolation theorem with sharp threshold | Low |
| C: Tropical Morse | MODERATE (Grigsby et al. PL Morse for network functions) | Tropical Morse for LOSS LANDSCAPES with critical cell bounds | Medium |

## RECOMMENDED THEORY A PIVOT

**Old framing**: "Conservation laws constrain trajectories to quasi-convex submanifold"
**New framing**: "Conservation laws serve as guide rails during early training (progressive sharpening phase). At the edge of stability, the discrete-time dynamics break these laws in a structured way that (a) drives the balancedness gap to zero (self-balancing), (b) improves NTK-target alignment, and (c) implicitly regularizes toward flat minima. The PATTERN of conservation law breaking -- which laws break first, how fast, in what order -- determines the quality of the solution found."

This is novel because:
1. Ghosh et al. only showed breaking in LINEAR networks -- we extend to nonlinear
2. Nobody connected breaking to NTK alignment (Jiang et al. 2025) or generalization
3. Nobody characterized the ORDER/PATTERN of breaking across multiple conservation laws
4. Nobody connected conservation law breaking to feature learning onset

---

## SESSION 4 ADDITIONS: Modified Equations, BEA, and EoS Theory

### Backward Error Analysis for Gradient Descent

27. **Barrett & Dherin**, "Implicit Gradient Regularization," ICLR 2021, arXiv:2009.11162.
    Discrete GD follows gradient flow on modified loss: L_tilde = L + (eta/4)||grad L||^2. Predicts O(eta^2) conservation drift. Our finding of eta^1.1 is NOT predicted.

28. **Smith, Dherin, Barrett, De**, "On the Origin of Implicit Regularization in SGD," ICLR 2021, arXiv:2101.12176.
    Extends BEA to SGD with mini-batches. Regularization scales as eta/B.

29. **Li, Tai, E**, "Stochastic Modified Equations and Dynamics of SGD," JMLR 2019, arXiv:1511.06251.
    Rigorous SDE approximation of SGD. Corrections in powers of eta/sqrt(eta). No sub-quadratic effects.

30. **Miyagawa**, "Equation of Motion for Deep Neural Networks," NeurIPS 2022, arXiv:2210.15898.
    Derives counter terms for discretization error in scale/translation-invariant layers.

### Edge of Stability Theory

31. **Cohen, Kaur, Li, Kolter, Talwalkar**, "GD on Neural Networks Occurs at the Edge of Stability," ICLR 2021, arXiv:2103.00065.
    Foundational EoS paper. Lambda_max hovers near 2/eta.

32. **Damian, Nichani, Lee**, "Self-Stabilization at EoS," ICLR 2023, arXiv:2209.15594.
    Cubic Taylor expansion explains EoS dynamics. GD at EoS = projected gradient descent.

33. **Arora, Li, Panigrahi**, "Understanding EoS in Deep Learning," ICML 2022, arXiv:2205.09745.
    GD at EoS follows deterministic flow on minimum-loss manifold.

34. **Song & Yun**, "Trajectory Alignment via Bifurcation Theory," NeurIPS 2023, arXiv:2307.04204.
    EoS explained via bifurcation diagram. Critical exponents could produce non-integer scaling.

35. **Wang, Xu, Zhao, Tao**, "Good Regularity, EoS, Balancing, and Catapult," NeurIPS 2024, arXiv:2310.17087.
    **KEY PAPER**: Unifies EoS with balancing (conservation law breaking) and catapult. Most relevant existing work to our drift exponent finding.

36. **Ahn et al.**, "Learning Threshold Neurons via EoS," NeurIPS 2023, arXiv:2212.07469.
    EoS is NECESSARY for learning certain functions. Sharp phase transition in step size.

### Anomalous Diffusion and Scaling

37. **Kunin et al.**, "Limiting Dynamics of SGD: Modified Loss and Anomalous Diffusion," Neural Computation 2023, arXiv:2107.09133.
    Post-convergence SGD shows anomalous diffusion with non-integer power law exponents. Highly relevant.

38. **Chen, Qu, Gong**, "Anomalous Diffusion Dynamics of Learning in DNNs," Neural Networks 2022, arXiv:2009.10588.
    Superdiffusion during learning, subdiffusion near convergence. Non-integer exponents universal across architectures.

39. **Rod Flow paper**, "Continuous-Time Model for GD at EoS," arXiv:2602.01480, Feb 2026.
    Models GD iterates as an extended 1D object (a "rod"). Correctly predicts critical sharpness threshold.

### Summary: Our drift exponent eta^1.1 is NOT predicted by any existing paper. The decomposition drift = eta^2 * S(eta) with S ~ eta^{-0.84} is a novel finding. The spectral crossover theory (Theorem 5) provides the first explanation.

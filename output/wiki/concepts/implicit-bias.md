---
title: Implicit Bias of Gradient Descent
category: concept
related: [edge-of-stability.md, neural-tangent-kernel.md, overparameterization.md, mode-connectivity.md]
key_papers:
  - Soudry et al. (2018)
  - Gunasekar et al. (2017)
  - Hochreiter, Schmidhuber (1997)
  - Keskar et al. (2017)
  - Dinh et al. (2017)
  - Foret et al. (2021)
  - Zhu et al. (2019)
  - Li et al. (2020)
status: partial
---

# Implicit Bias of Gradient Descent

## Core Idea

Gradient descent on underdetermined (overparameterized) problems does not simply find any interpolating solution -- it exhibits systematic biases toward particular solutions that tend to generalize well, independent of explicit regularization. This implicit bias provides a partial explanation for why SGD-trained neural networks generalize despite having far more parameters than training examples.

Soudry et al. (2018) proved that for logistic loss on linearly separable data, gradient descent iterates converge in direction to the max-margin (hard SVM) solution. Gunasekar et al. (2017) showed that gradient descent on matrix factorization implicitly biases toward low nuclear norm (approximately low-rank) solutions. These results demonstrate that the optimization algorithm itself acts as a regularizer, selecting structured solutions from the manifold of interpolators.

The flat minima hypothesis, originating with Hochreiter and Schmidhuber (1997), posits that SGD preferentially finds wide basins in the loss landscape that correspond to simpler models with better generalization. Keskar et al. (2017) provided empirical support by showing that large-batch SGD finds sharp minima with poor generalization while small-batch SGD finds flat minima with good generalization. However, Dinh et al. (2017) mounted a fundamental critique showing that sharpness measures are not invariant under reparameterization, casting doubt on the theoretical foundations. Despite this, sharpness-aware minimization (SAM, Foret et al. 2021) consistently improves generalization in practice -- a partially unresolved paradox.

## Mathematical Framework

**Max-margin convergence (Soudry et al. 2018).** For logistic loss $\ell(z) = \log(1 + e^{-z})$ on linearly separable data $\{(x_i, y_i)\}_{i=1}^n$, gradient descent iterates $w_t$ satisfy:

$$\frac{w_t}{\|w_t\|} \to \hat{w}_{\text{SVM}} = \arg\max_{\|w\|=1} \min_i y_i (w \cdot x_i)$$

at rate $O(1/\log t)$, with $\|w_t\| = O(\log t)$.

**Nuclear norm bias (Gunasekar et al. 2017).** For matrix factorization $X = UU^\top$, gradient descent initialized near zero implicitly minimizes the nuclear norm $\|X\|_* = \sum_i \sigma_i(X)$, selecting approximately low-rank solutions. Li et al. (2020) refined this to a greedy low-rank learning process.

**SGD as stochastic differential equation (Zhu et al. 2019).** SGD with learning rate $\eta$ and batch size $B$ is modeled as:

$$d\theta = -\nabla L \, dt + \sqrt{\frac{\eta}{B} \Sigma(\theta)} \, dW_t$$

where $\Sigma(\theta) = \mathbb{E}[(\nabla \ell_i - \nabla L)(\nabla \ell_i - \nabla L)^\top]$ is the noise covariance. The key insight: $\Sigma$ aligns with the Hessian, making escape from sharp minima more efficient than isotropic Langevin dynamics would predict. This anisotropic noise structure explains SGD's preference for flat minima.

**Flat minima hypothesis (Hochreiter, Schmidhuber 1997).** Wide basins in the loss landscape generalize better because they correspond to simpler, lower-description-length models. Formally, a flat minimum $\theta^*$ satisfies $\lambda_{\max}(\nabla^2 L(\theta^*)) \ll 1$, meaning the loss is insensitive to parameter perturbations.

**Dinh et al. (2017) critique.** For ReLU networks, the rescaling:

$$W_k \to \alpha W_k, \quad W_{k+1} \to \alpha^{-1} W_{k+1}$$

preserves the network function while making the minimum arbitrarily sharp or flat. Any sharpness measure not invariant to such reparameterization is theoretically vacuous.

**SAM objective (Foret et al. 2021):**

$$\min_w \max_{\|\varepsilon\| \leq \rho} L(w + \varepsilon)$$

This minimax formulation explicitly seeks parameters where the worst-case perturbed loss is small, favoring flat regions.

## What It Explains

Implicit bias explains why gradient descent, despite the existence of infinitely many interpolating solutions in overparameterized networks, consistently finds solutions that generalize. The max-margin and low-rank biases show that gradient descent is not arbitrary -- it selects solutions with specific structural properties (maximum margin, low rank) that align with classical notions of simplicity and regularization. The SGD noise structure further explains why stochastic optimization outperforms full-batch gradient descent for generalization.

## Limitations

1. **Reparameterization non-invariance (Dinh et al. 2017).** The flat minima hypothesis lacks a parameterization-invariant formulation. The rescaling $W_k \to \alpha W_k, W_{k+1} \to \alpha^{-1}W_{k+1}$ creates a fundamental theoretical gap.

2. **The SAM paradox.** SAM empirically works despite the theoretical objections to sharpness measures. Possible resolutions include PAC-Bayesian bounds under specific parameterization conventions and the observation that SGD's natural parameterization may never encounter the pathological rescalings Dinh et al. exploit.

3. **Results are architecture-specific.** The max-margin result applies to linear models with logistic loss; the nuclear norm result applies to matrix factorization. Extension to general deep networks with nonlinear activations remains incomplete.

4. **Implicit bias changes with architecture.** The implicit bias of gradient descent depends on the parameterization, depth, activation function, and loss function. There is no universal characterization.

5. **Does not explain when implicit bias fails.** There are settings where overparameterized networks do not generalize well (e.g., random labels), but the implicit bias framework does not predict these failures.

## Key Results

- **Soudry et al. (2018):** Gradient descent on separable data with logistic loss converges in direction to the $\ell_2$ max-margin solution at rate $O(1/\log t)$.
- **Gunasekar et al. (2017):** Gradient descent on matrix factorization implicitly biases toward low nuclear norm solutions.
- **Li et al. (2020):** Refined the matrix factorization result to a greedy low-rank learning process.
- **Hochreiter, Schmidhuber (1997):** Proposed the flat minima hypothesis linking basin width to generalization.
- **Keskar et al. (2017):** Empirically demonstrated that large-batch SGD finds sharp minima (poor generalization) while small-batch SGD finds flat minima (good generalization).
- **Dinh et al. (2017):** Showed that sharpness measures are not reparameterization-invariant for ReLU networks, undermining the theoretical basis of the flat minima hypothesis.
- **Zhu et al. (2019):** Modeled SGD as an SDE with anisotropic noise covariance aligned with the Hessian, explaining preferential escape from sharp minima.
- **Foret et al. (2021):** Introduced SAM, which consistently improves generalization by explicitly optimizing for flat minima despite the theoretical concerns.

## Connections

- [Edge of Stability](edge-of-stability.md): The progressive sharpening and edge-of-stability phenomena are intimately connected to how gradient descent interacts with the curvature landscape, complementing the implicit bias picture.
- [Neural Tangent Kernel](neural-tangent-kernel.md): In the NTK regime, the implicit bias reduces to minimum-norm interpolation in the RKHS, providing a clean characterization for the lazy regime.
- [Overparameterization](overparameterization.md): Implicit bias is most relevant in the overparameterized regime where the interpolation manifold is high-dimensional and the choice among interpolators matters.
- [Mode Connectivity](mode-connectivity.md): The implicit bias of SGD may explain why independently trained networks end up in the same connected basin.

## Open Questions

1. Is there a reparameterization-invariant measure of "flatness" that rigorously predicts generalization?
2. Why does SAM work despite the Dinh et al. critique? Is the natural parameterization somehow special?
3. Can the implicit bias of gradient descent for deep nonlinear networks be characterized in closed form, beyond the linear and matrix factorization settings?
4. How does the implicit bias interact with architectural choices (skip connections, normalization layers, attention) that change the effective parameterization?
5. What is the precise relationship between SGD's anisotropic noise and the generalization bound -- can the SDE model yield non-vacuous bounds?

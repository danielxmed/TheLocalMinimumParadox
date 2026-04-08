---
title: Neural Tangent Kernel
category: concept
related: [overparameterization.md, mean-field-theory.md, implicit-bias.md, random-matrix-theory.md]
key_papers:
  - Jacot, Gabriel, Hongler (2018)
  - Du et al. (2019)
  - Allen-Zhu, Li, Song (2019)
  - Chizat, Oyallon, Bach (2019)
status: established
---

# Neural Tangent Kernel

## Core Idea

The Neural Tangent Kernel (NTK) framework, introduced by Jacot, Gabriel, and Hongler (2018), provides the most complete rigorous account of why gradient descent converges to global minima in overparameterized neural networks. The central insight is that in the infinite-width limit under NTK parameterization, the kernel defined by the network's parameter gradients converges to a deterministic object and remains frozen during training. This converts the non-convex parameter-space optimization into convex kernel regression in function space, eliminating the problem of bad local minima entirely.

The NTK framework thus explains convergence by showing that sufficiently wide networks behave as linear models around their initialization. Gradient descent in parameter space becomes equivalent to kernel gradient descent with the fixed NTK, inheriting all the convergence guarantees of convex optimization. Du et al. (2019) and Allen-Zhu, Li, and Song (2019) translated this insight into finite-width convergence theorems with explicit width requirements.

However, the NTK regime describes only "lazy training" where parameters barely move from initialization. This fundamental limitation, identified by Chizat, Oyallon, and Bach (2019), means NTK theory explains why networks converge but not why deep learning outperforms kernel methods -- the latter requires feature learning, which NTK forbids.

## Mathematical Framework

The NTK is defined as:

$$K(x, x'; \theta) = \nabla_\theta f(x; \theta)^\top \nabla_\theta f(x'; \theta)$$

In the infinite-width limit, $K$ converges to a deterministic kernel $\Theta^\infty$ and remains constant during training. The training dynamics become linear:

$$\dot{f}_t = -\eta \Theta^\infty (f_t - Y)$$

This has a closed-form solution:

$$f_t - Y = \exp(-\eta \Theta^\infty t)(f_0 - Y)$$

Convergence to zero training loss follows from the positive-definiteness of $\Theta^\infty$, which holds when the data points are distinct and the activation function is non-polynomial.

**Width requirements for finite networks:**
- Two-layer ReLU networks (Du et al. 2019): $m = \Omega(n^6 / \lambda_0^4)$ where $\lambda_0 = \lambda_{\min}(\Theta^\infty)$ and $n$ is the number of training samples.
- Deep networks (Allen-Zhu, Li, Song 2019): SGD finds global minima for width $m \geq \Omega(n^6 L^2)$ where $L$ is the depth, covering fully-connected networks, CNNs, and ResNets.

**Parameterization regimes:**
- NTK parameterization: weights scaled as $O(1/\sqrt{m})$ -- drives networks toward lazy training.
- Mean-field scaling: weights scaled as $O(1/m)$ -- permits genuine feature learning.

## What It Explains

NTK theory explains the convergence aspect of the paradox: why gradient descent reliably reaches zero training loss in overparameterized networks. By mapping the non-convex problem to convex kernel regression, it shows that the apparent non-convexity is an artifact of the parameterization. In function space, the optimization is convex, and the linear convergence rate is exponential: $\|f_t - Y\| \leq \exp(-\eta \lambda_0 t) \|f_0 - Y\|$.

## Limitations

NTK theory alone does not resolve the paradox for several reasons:

1. **No feature learning.** In the lazy training regime, the network is equivalent to kernel regression with fixed random features. No representation learning occurs, yet real networks demonstrably learn hierarchical features that outperform kernel methods.

2. **Unrealistic width requirements.** The required width $m = \Omega(n^6 / \lambda_0^4)$ for two-layer networks is far beyond practical network sizes.

3. **NTK does not explain generalization.** The NTK solution corresponds to minimum-norm interpolation in the RKHS, which does not account for the generalization behavior observed in practice (e.g., the double descent phenomenon).

4. **Lazy regime is not the regime of practice.** Chizat, Oyallon, and Bach (2019) showed that NTK parameterization actively suppresses feature learning. Real networks operate in the "rich" or "feature learning" regime where the kernel evolves during training.

## Key Results

- **Jacot, Gabriel, Hongler (2018):** Proved that the NTK converges to a deterministic limit in the infinite-width limit and remains constant during gradient descent training.
- **Du et al. (2019):** Proved that gradient descent on two-layer ReLU networks with width $m = \Omega(n^6/\lambda_0^4)$ achieves zero training loss at linear rate.
- **Allen-Zhu, Li, Song (2019):** Extended convergence guarantees to deep networks, CNNs, and ResNets via SGD with width $m \geq \Omega(n^6 L^2)$.
- **Chizat, Oyallon, Bach (2019):** Identified the lazy training limitation -- NTK describes a regime where no feature learning occurs, explaining why it cannot account for deep learning's superiority over kernel methods.

## Connections

- [Overparameterization](overparameterization.md): NTK provides the theoretical backbone for understanding why overparameterization convexifies the landscape, but only in the lazy regime.
- [Mean Field Theory](mean-field-theory.md): The alternative infinite-width limit that permits feature learning, operating under different scaling ($O(1/m)$ vs $O(1/\sqrt{m})$).
- [Implicit Bias](implicit-bias.md): NTK solutions correspond to minimum-norm interpolation in the RKHS, connecting to the broader study of gradient descent's implicit regularization.
- [Random Matrix Theory](random-matrix-theory.md): The NTK's spectral properties determine convergence rates and connect to the Hessian analysis of trained networks.

## Open Questions

1. Can the NTK framework be extended to capture feature learning dynamics, perhaps through a time-evolving kernel that tracks the transition from lazy to rich regime?
2. What is the tightest possible width requirement for convergence -- can $\Omega(n^6)$ be reduced to polynomial in $n$ with small exponent?
3. How does the NTK eigenspectrum relate to the generalization properties of the learned function, beyond minimum-norm interpolation?
4. Is there a unified framework that smoothly interpolates between the NTK (lazy) and mean-field (rich) regimes as a function of parameterization scale?

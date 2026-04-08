---
title: Mean Field Theory
category: concept
related: [neural-tangent-kernel.md, overparameterization.md, energy-landscape.md, conservation-laws.md]
key_papers:
  - Mei, Montanari, Nguyen (2018)
  - Chizat, Bach (2018)
  - Chizat (2022)
status: established
---

# Mean Field Theory

## Core Idea

Mean field theory provides an alternative infinite-width limit for neural networks that, unlike the NTK framework, permits genuine feature learning. Introduced independently by Mei, Montanari, and Nguyen (2018) and Chizat and Bach (2018), the approach replaces the discrete collection of neurons with a continuous probability measure over parameter space, reformulating training as a Wasserstein gradient flow. The crucial mathematical insight is that the risk functional is convex in the linear geometry of measures, even though the finite-particle (finite-width) discretization is non-convex.

In the many-particle limit, propagation of chaos ensures that the behavior of individual neurons becomes asymptotically independent, and the finite system tracks the infinite-dimensional convex optimization. This provides a rigorous explanation for why gradient descent converges to global minima while permitting the network to learn data-adapted features -- a capability the NTK regime fundamentally lacks.

Chizat (2022) proved exponential convergence of mean-field Langevin dynamics to global minimizers under a log-Sobolev inequality, giving quantitative rates. However, current rigorous results are limited to two-layer (single hidden layer) architectures, leaving the extension to deep networks as a major open problem.

## Mathematical Framework

The network output for a two-layer network with measure $\rho$ over neuron parameters is:

$$f_\rho(x) = \int \sigma(w \cdot x) \, d\rho(w)$$

where $\sigma$ is the activation function. Training minimizes the risk functional:

$$R(\rho) = \mathbb{E}_{(x,y)} [\ell(f_\rho(x), y)]$$

The training dynamics are formulated as the **Wasserstein gradient flow**:

$$\partial_t \rho_t = \nabla \cdot \left(\rho_t \nabla_w \frac{\delta R}{\delta \rho}\right)$$

where $\frac{\delta R}{\delta \rho}$ is the first variation (functional derivative) of the risk with respect to the measure.

**Key convexity property:** $R(\rho)$ is convex in the linear (weak) topology of measures. For any $\rho_0, \rho_1 \in \mathcal{P}(\mathbb{R}^D)$ and $\lambda \in [0,1]$:

$$R(\lambda \rho_0 + (1-\lambda)\rho_1) \leq \lambda R(\rho_0) + (1-\lambda) R(\rho_1)$$

This convexity holds because $f_\rho$ depends linearly on $\rho$. The non-convexity in the finite-width setting arises from the discretization $\rho = \frac{1}{m}\sum_{j=1}^m \delta_{w_j}$, which restricts to a non-convex subset of measures.

**Propagation of chaos:** In the limit $m \to \infty$, the empirical measure $\hat{\rho}_m = \frac{1}{m}\sum_{j=1}^m \delta_{w_j(t)}$ converges (in Wasserstein distance) to the solution $\rho_t$ of the mean-field PDE, and individual neurons become asymptotically independent.

**Exponential convergence (Chizat 2022):** Under a log-Sobolev inequality for the target measure, mean-field Langevin dynamics converge exponentially to the global minimizer of the regularized risk.

## What It Explains

Mean field theory explains why overparameterized two-layer networks can learn features while still converging to global optima. The convexity in measure space shows that the loss landscape, when viewed at the right level of abstraction, has no spurious local minima. This addresses a key gap in NTK theory: mean field theory accounts for representation learning, explaining (at least for shallow networks) why deep learning outperforms kernel methods.

## Limitations

1. **Limited to two-layer architectures.** Extending mean-field theory to deep networks requires handling the composition of measures across layers, which breaks the linear dependence on any single layer's measure.

2. **Propagation of chaos is asymptotic.** The finite-width approximation error and the rate of convergence to the mean-field limit are not always tight enough to explain practical network sizes.

3. **The log-Sobolev inequality** required for Chizat's exponential convergence result is a strong assumption that may not hold for all data distributions and network architectures.

4. **Computational aspects.** The Wasserstein gradient flow is an infinite-dimensional PDE. Practical optimization uses finite-width networks with SGD, and the gap between the idealized flow and discrete stochastic updates is not fully characterized.

## Key Results

- **Mei, Montanari, Nguyen (2018):** Established the mean-field limit for two-layer networks and proved that the risk functional is convex over probability measures.
- **Chizat, Bach (2018):** Independently developed the Wasserstein gradient flow formulation and proved global convergence in the infinite-width limit.
- **Chizat (2022):** Proved exponential convergence of mean-field Langevin dynamics to global minimizers under a log-Sobolev inequality, providing quantitative convergence rates.

## Connections

- [Neural Tangent Kernel](neural-tangent-kernel.md): NTK operates under $O(1/\sqrt{m})$ scaling (lazy regime), while mean field operates under $O(1/m)$ scaling (feature learning regime). They represent complementary infinite-width limits.
- [Overparameterization](overparameterization.md): Mean field theory provides the theoretical framework for the "rich" regime of overparameterization, where width enables feature learning rather than just linearization.
- [Energy Landscape](energy-landscape.md): The convexity in measure space implies a single-funnel structure when viewed from the appropriate abstraction level.
- [Conservation Laws](conservation-laws.md): The gradient flow structure connects to conserved quantities and symmetry properties of the dynamics.

## Open Questions

1. Can mean-field theory be rigorously extended to deep (multi-layer) networks? What replaces convexity in measure space when layers compose?
2. What is the finite-width correction to the mean-field limit, and does it explain the phase transition between lazy and rich regimes?
3. Under what conditions on the data distribution does the log-Sobolev inequality hold for the mean-field Langevin dynamics?
4. How does the mean-field perspective connect to the empirical observation of neural collapse in the terminal phase of training?
5. Can the Wasserstein gradient flow formulation yield practical algorithms that outperform standard SGD?

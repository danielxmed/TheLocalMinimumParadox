---
title: Conservation Laws and Symmetry
category: concept
related: [mode-connectivity.md, overparameterization.md, energy-landscape.md, tropical-geometry.md]
key_papers:
  - Simsek et al. (2021)
  - Zhao et al. (2022)
  - Grigsby, Lindsey, Rolnick (2023)
  - Kileel et al. (2019)
status: emerging
---

# Conservation Laws and Symmetry

## Core Idea

Neural network parameter spaces possess rich symmetry structures that have profound consequences for the optimization landscape. The most obvious symmetry is the permutation group $G = \prod_k S_{n_k}$ acting by relabeling neurons within each layer, which creates $\prod_k n_k!$ equivalent copies of each minimum. But beyond this discrete group, there are continuous symmetries, conserved quantities under gradient flow, and hidden symmetries specific to piecewise-linear networks.

Zhao et al. (2022) identified conserved quantities for 2-homogeneous networks (including standard ReLU architectures): the "balancedness" invariant $\|W_{\text{out}}\|^2 - \|W_{\text{in}}\|^2$ is preserved under gradient flow. They further showed that symmetry teleportation -- exploiting group actions to jump between equivalent points -- can accelerate optimization. Simsek et al. (2021) proved a striking constructive result: adding a single extra neuron per layer suffices to connect all discrete symmetry-related minima into a single continuous manifold, converting the combinatorial proliferation of equivalent minima into a connected solution set. These results show that symmetry is not merely a source of redundancy but a structural resource that shapes the landscape topology.

Grigsby, Lindsey, and Rolnick (2023) discovered that ReLU networks possess hidden symmetries beyond permutation and rescaling, arising from the piecewise-linear structure, further enriching the symmetry picture. The algebraic geometry of the parameterization map reveals additional structure: the neuromanifold is an algebraic variety (for polynomial activations), and the Fisher information metric degenerates at singular points where neurons coincide.

## Mathematical Framework

**Permutation symmetry.** The discrete symmetry group $G = \prod_k S_{n_k}$ acts on parameter space $\Theta$ by permuting neurons within each layer $k$ (which has $n_k$ neurons). For any $\pi \in G$:

$$f(\cdot; \theta) = f(\cdot; \pi \cdot \theta)$$

Each minimum $\theta^*$ has $|G| = \prod_k n_k!$ equivalent copies, creating an exponential number of equivalent critical points.

**Rescaling symmetry (for ReLU networks).** For consecutive layers with ReLU activation:

$$W_k \to \alpha W_k, \quad W_{k+1} \to \alpha^{-1} W_{k+1} \quad (\alpha > 0)$$

preserves the network function. This continuous symmetry is central to the Dinh et al. (2017) critique of sharpness measures.

**Balancedness invariant (Zhao et al. 2022).** For 2-homogeneous networks under gradient flow $\dot{\theta} = -\nabla L(\theta)$, the following quantity is conserved:

$$\|W_{\text{out}}\|^2 - \|W_{\text{in}}\|^2 = \text{const}$$

More generally, for homogeneous networks of degree $p$, there exist analogous conserved quantities relating the norms of weight matrices across layers. These conservation laws constrain the gradient flow trajectories to specific manifolds within parameter space.

**Symmetry teleportation (Zhao et al. 2022).** Group actions can be used to instantaneously move to equivalent parameter configurations that have different optimization properties (e.g., different gradient norms or curvatures), potentially accelerating convergence.

**Extra neuron connectivity (Simsek et al. 2021).** Adding a single extra neuron per layer is sufficient to connect all $\prod_k n_k!$ permutation-equivalent minima into one continuous manifold. The proof constructs explicit smooth paths through the augmented parameter space, using the extra neuron as a "relay" to continuously interpolate between permuted configurations.

**Hidden symmetries (Grigsby, Lindsey, Rolnick 2023).** ReLU networks possess symmetries beyond permutation and rescaling, arising from the piecewise-linear structure. These hidden symmetries further complicate the fiber structure of the parameterization map $\phi: \Theta \to \mathcal{F}$.

**Algebraic geometry of parameter space.** The image of $\phi: \Theta \to \mathcal{F}$ defines the neuromanifold:
- For polynomial activations, this is an algebraic variety.
- Linear networks produce determinantal varieties (matrices of bounded rank).
- Shallow networks with monomial activation $\sigma(z) = z^k$ produce spaces of symmetric tensors of bounded Waring rank (Kileel et al. 2019).
- The quotient $\Theta/G$ is an orbifold with singular points where neurons coincide, and the Fisher information metric degenerates at these singularities.

## What It Explains

Conservation laws and symmetry explain several aspects of the paradox:

1. The exponential number of critical points (due to $\prod_k n_k!$ permutation copies) is not an obstacle because these are equivalent solutions, and adding a single extra neuron connects them all.

2. The conserved quantities constrain optimization trajectories to lower-dimensional manifolds, reducing the effective complexity of the optimization problem.

3. Symmetry teleportation provides a mechanism for escaping local regions without gradient-based search, suggesting that the effective landscape is simpler than the raw parameter space.

4. The hidden symmetries of ReLU networks imply that the effective number of distinct solutions is even smaller than the permutation count suggests.

## Limitations

1. **Conservation laws are exact only for gradient flow.** Discrete gradient descent (especially with momentum, weight decay, or adaptive learning rates) may only approximately preserve these quantities.

2. **Limited to specific architectures.** The balancedness invariant holds for 2-homogeneous networks but may not extend to architectures with normalization layers, attention mechanisms, or non-homogeneous activations.

3. **Extra neuron result is existential.** Simsek et al. (2021) prove that paths exist but do not show that gradient descent naturally follows them. The practical relevance for optimization dynamics is unclear.

4. **Symmetry teleportation is not standard SGD.** Exploiting symmetry teleportation requires knowledge of the group structure and intentional application, which standard optimizers do not perform.

5. **Singular orbifold structure.** The quotient space $\Theta/G$ has singularities at neuron-coincidence points where the Fisher information degenerates, creating potential difficulties for optimization near these singular loci.

## Key Results

- **Zhao et al. (2022):** Identified conserved quantities under gradient flow for 2-homogeneous networks ($\|W_{\text{out}}\|^2 - \|W_{\text{in}}\|^2 = \text{const}$) and demonstrated symmetry teleportation for optimization acceleration.
- **Simsek et al. (2021):** Proved that adding a single extra neuron per layer connects all permutation-equivalent minima into a single continuous manifold.
- **Grigsby, Lindsey, Rolnick (2023):** Discovered hidden symmetries in ReLU networks beyond permutation and rescaling, arising from piecewise-linear structure.
- **Kileel et al. (2019):** Characterized the algebraic geometry of neural network parameter spaces, including the neuromanifold as an algebraic variety.

## Connections

- [Mode Connectivity](mode-connectivity.md): The Simsek et al. extra neuron result provides a mechanism for mode connectivity -- adding minimal capacity connects all equivalent minima. This complements the empirical mode connectivity results of Draxler et al. and Garipov et al.
- [Overparameterization](overparameterization.md): The extra neuron result shows that minimal overparameterization (one neuron per layer) suffices to qualitatively change the topology of the solution set.
- [Energy Landscape](energy-landscape.md): Conservation laws constrain the gradient flow to specific manifolds, shaping the effective energy landscape and potentially contributing to the single-funnel structure.
- [Tropical Geometry](tropical-geometry.md): The piecewise-linear structure that gives rise to hidden symmetries in ReLU networks is also the foundation for the tropical geometry perspective.

## Open Questions

1. Can conservation laws be identified for non-homogeneous architectures (e.g., networks with batch normalization or attention layers)?
2. How do discrete gradient descent and adaptive optimizers (Adam, AdaGrad) interact with the conserved quantities -- do they approximately preserve them or systematically violate them?
3. Can symmetry teleportation be incorporated into practical optimization algorithms for measurable speedups?
4. What is the complete symmetry group of deep ReLU networks, including all hidden symmetries? Can it be characterized combinatorially?
5. How does the Fisher information degeneracy at orbifold singularities affect optimization dynamics in practice?

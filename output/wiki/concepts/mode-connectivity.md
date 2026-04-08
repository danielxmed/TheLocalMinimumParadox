---
title: Mode Connectivity
category: concept
related: [energy-landscape.md, overparameterization.md, implicit-bias.md, conservation-laws.md]
key_papers:
  - Draxler et al. (2018)
  - Garipov et al. (2018)
  - Frankle et al. (2020)
  - Entezari et al. (2022)
  - Ainsworth et al. (2023)
status: established
---

# Mode Connectivity

## Core Idea

Mode connectivity is the empirical and theoretical observation that independently trained neural network solutions (modes) can be connected by paths through parameter space along which the loss remains nearly constant. This provides direct topological evidence that the loss landscape is far more benign than worst-case theory suggests: rather than being riddled with isolated local minima separated by high barriers, the landscape appears to have a connected low-loss region containing all good solutions.

Draxler et al. (2018) first demonstrated this using the Nudged Elastic Band (NEB) algorithm from computational chemistry, finding essentially flat minimum-energy paths between independently trained minima of VGG, ResNet, and DenseNet on CIFAR-10/100. Garipov et al. (2018) showed that even simple Bezier curves with a single bend suffice to connect modes at constant accuracy. These findings suggest that the apparent multiplicity of local minima is largely an artifact of the discrete symmetry group (neuron permutations) acting on a single connected solution set.

The deeper question -- whether all SGD solutions are fundamentally equivalent up to symmetry -- is captured by the Entezari et al. (2022) conjecture, which proposes linear mode connectivity after permutation alignment. Ainsworth et al. (2023) provided strong empirical support for this conjecture via their Git Re-Basin algorithm.

## Mathematical Framework

**Non-linear mode connectivity.** Given two minima $\theta_A$ and $\theta_B$, a connecting path $\gamma: [0,1] \to \Theta$ satisfies $\gamma(0) = \theta_A$, $\gamma(1) = \theta_B$, and:

$$\max_{t \in [0,1]} L(\gamma(t)) \approx \max(L(\theta_A), L(\theta_B))$$

That is, the loss barrier along the path is negligible.

**Bezier curve parameterization (Garipov et al. 2018).** The connecting path is parameterized as a quadratic Bezier curve:

$$\gamma(t) = (1-t)^2 \theta_A + 2t(1-t)\theta_{\text{bend}} + t^2 \theta_B$$

where $\theta_{\text{bend}}$ is a single learnable midpoint, optimized to minimize $\int_0^1 L(\gamma(t)) \, dt$.

**Linear mode connectivity (Frankle et al. 2020).** Two networks $\theta_A$ and $\theta_B$ are linearly mode connected if:

$$L((1-t)\theta_A + t\theta_B) \leq \max(L(\theta_A), L(\theta_B)) + \epsilon \quad \forall t \in [0,1]$$

for small $\epsilon$. Networks sharing early training (from the same checkpoint) satisfy this; networks from different random initializations do not.

**Entezari et al. (2022) conjecture.** For sufficiently overparameterized networks, all SGD solutions are linearly mode connected after permutation alignment. That is, for any two solutions $\theta_A, \theta_B$, there exists a permutation $\pi \in G = \prod_k S_{n_k}$ such that $\theta_A$ and $\pi(\theta_B)$ are linearly mode connected.

**Symmetry group.** The discrete symmetry group $G = \prod_k S_{n_k}$ acts on parameter space by permuting neurons within each layer. Each minimum has $|G| = \prod_k n_k!$ equivalent copies under this action.

## What It Explains

Mode connectivity explains why the proliferation of local minima (exponential in the number of parameters due to symmetry) does not pose an optimization barrier. If all good solutions lie in a single connected basin (modulo permutation symmetry), then gradient descent starting from any reasonable initialization will flow toward this connected region. The apparent non-convexity is largely a consequence of the discrete symmetry group creating multiple equivalent copies of solutions that are, topologically, part of the same basin.

## Limitations

1. **Empirical, not fully rigorous.** Mode connectivity has been demonstrated experimentally on standard architectures and datasets, but a general theoretical proof is lacking.

2. **The Entezari conjecture is unproven.** While Ainsworth et al. (2023) provided strong empirical evidence via Git Re-Basin, the conjecture remains open. Counterexamples may exist for specific architectures or data distributions.

3. **Does not explain the quality of minima.** Mode connectivity shows that solutions are connected but does not explain why they generalize well or why SGD finds them.

4. **Permutation alignment is computationally hard.** Finding the optimal permutation to align two networks is itself a combinatorial optimization problem (related to graph matching), making it difficult to verify the conjecture at scale.

5. **Linear mode connectivity is stronger than non-linear.** Non-linear paths always exist in high enough dimensions; the significance lies in the flatness of these paths, which linear connectivity makes precise.

## Key Results

- **Draxler et al. (2018):** Used the Nudged Elastic Band algorithm to find essentially flat minimum-energy paths between independently trained VGG, ResNet, and DenseNet minima on CIFAR-10/100.
- **Garipov et al. (2018):** Showed that simple quadratic Bezier curves with one bend suffice to connect modes at constant accuracy; introduced Fast Geometric Ensembling.
- **Frankle et al. (2020):** Introduced linear mode connectivity; showed that networks sharing early training (same checkpoint) are linearly connected, while networks from different initializations are not. This established a connection to the lottery ticket hypothesis and training trajectory structure.
- **Entezari et al. (2022):** Conjectured that all SGD solutions are linearly mode connected after accounting for permutation symmetry.
- **Ainsworth et al. (2023):** Developed the Git Re-Basin algorithm providing strong empirical support for the Entezari conjecture by efficiently finding permutation alignments that enable linear interpolation.

## Connections

- [Energy Landscape](energy-landscape.md): Mode connectivity is the empirical signature of the single-funnel landscape structure identified by Ballard et al. (2017). Connected modes imply a funnel topology.
- [Overparameterization](overparameterization.md): Freeman and Bruna (2017) proved that level sets become connected as width grows, providing a theoretical basis for mode connectivity in overparameterized networks.
- [Implicit Bias](implicit-bias.md): The fact that independently trained networks find connected solutions suggests SGD's implicit bias funnels all trajectories toward the same connected basin.
- [Conservation Laws](conservation-laws.md): Simsek et al. (2021) showed that adding a single extra neuron per layer connects all discrete symmetry-related minima into a continuous manifold, providing a mechanism for mode connectivity.

## Open Questions

1. Can the Entezari conjecture be proven rigorously for specific architectures (e.g., two-layer ReLU networks)?
2. What is the relationship between mode connectivity and generalization -- are better-connected solutions also better-generalizing?
3. How does the connectivity structure change with depth, and does it explain the trainability advantages of ResNets over plain networks?
4. Can mode connectivity be exploited algorithmically for model merging, federated learning, or ensemble construction?
5. What is the precise relationship between the barrier height along connecting paths and the width/overparameterization ratio?

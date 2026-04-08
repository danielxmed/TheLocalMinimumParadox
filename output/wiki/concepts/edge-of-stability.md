---
title: Edge of Stability
category: concept
related: [implicit-bias.md, random-matrix-theory.md, overparameterization.md, neural-collapse.md]
key_papers:
  - Cohen et al. (2021)
  - Keskar et al. (2017)
status: emerging
---

# Edge of Stability

## Core Idea

The edge of stability is a recently identified phenomenon in gradient descent training of neural networks where the largest eigenvalue of the Hessian, $\lambda_{\max}$, self-tunes to hover near the value $2/\eta$ (where $\eta$ is the learning rate). Training proceeds in two distinct phases: first, a progressive sharpening phase where $\lambda_{\max}$ increases steadily; then, upon reaching the threshold $2/\eta$, the system enters a self-tuning regime where $\lambda_{\max}$ oscillates around this critical value while the loss continues to decrease non-monotonically.

This phenomenon is significant because it contradicts the predictions of classical optimization theory. In the quadratic approximation, gradient descent diverges when $\lambda_{\max} > 2/\eta$. Yet in practice, the non-linear dynamics of the loss function allow training to continue productively even when the curvature reaches this classical stability boundary. The network effectively self-regulates its own optimization dynamics.

The edge of stability provides a new lens for understanding the implicit bias of gradient descent: the training dynamics are constrained by the learning rate to operate in regions of bounded curvature, which may connect to the preference for flat minima and good generalization. However, the theoretical understanding of this phenomenon remains incomplete, and its connection to generalization is still unclear.

## Mathematical Framework

**Classical stability threshold.** For gradient descent $\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)$ applied to a quadratic $L(\theta) = \frac{1}{2}\theta^\top H \theta$, convergence requires:

$$\lambda_{\max}(H) < \frac{2}{\eta}$$

When $\lambda_{\max} > 2/\eta$, the quadratic model predicts divergence along the top eigenvector of $H$.

**Progressive sharpening phase.** During early training, the maximum Hessian eigenvalue increases:

$$\lambda_{\max}(\nabla^2 L(\theta_t)) \uparrow \quad \text{until} \quad \lambda_{\max} \approx \frac{2}{\eta}$$

This phase reflects the network moving from the flat initial region toward regions of higher curvature as it fits the training data.

**Edge of stability phase.** Once $\lambda_{\max} \approx 2/\eta$, the dynamics enter a regime where:

1. $\lambda_{\max}$ oscillates around $2/\eta$, neither growing unboundedly nor decreasing substantially.
2. The training loss decreases non-monotonically -- individual steps may increase the loss, but the trend is downward.
3. The trajectory implicitly avoids regions where $\lambda_{\max} \gg 2/\eta$.

**Self-tuning mechanism.** The non-linear dynamics beyond the quadratic approximation provide a self-correcting mechanism: when $\lambda_{\max}$ exceeds $2/\eta$, gradient steps along the high-curvature direction overshoot, effectively pushing the parameters toward regions of lower curvature. This creates an implicit constraint:

$$\lambda_{\max}(\nabla^2 L(\theta_t)) \lessapprox \frac{2}{\eta}$$

throughout the edge-of-stability phase.

## What It Explains

The edge of stability reveals that gradient descent with a fixed learning rate implicitly regularizes the training trajectory by constraining the maximum curvature of the loss landscape along the optimization path. This provides a dynamical mechanism for the flat minima preference: the learning rate acts as an implicit regularizer that prevents the optimization from settling into sharp minima with $\lambda_{\max} > 2/\eta$. Larger learning rates enforce flatter solutions, consistent with the empirical observation that larger learning rates (up to a point) improve generalization.

## Limitations

1. **No rigorous theory.** The edge of stability has been empirically observed and partially analyzed, but a complete mathematical theory explaining why the loss continues to decrease in this regime is lacking.

2. **Connection to generalization is unclear.** While the curvature constraint $\lambda_{\max} \lessapprox 2/\eta$ suggests flatter solutions, the precise relationship between the edge-of-stability dynamics and generalization performance has not been established.

3. **Interaction with SGD noise is not understood.** Most edge-of-stability observations use full-batch gradient descent. How the phenomenon interacts with the stochastic noise of mini-batch SGD is an active research question.

4. **Architecture dependence.** The progressive sharpening rate and the edge-of-stability dynamics may vary significantly across architectures, and the phenomenon has primarily been studied on standard image classification models.

5. **Reparameterization issues.** Like the flat minima hypothesis, $\lambda_{\max}$ is not invariant under all reparameterizations, inheriting the concerns raised by Dinh et al. (2017).

## Key Results

- **Cohen et al. (2021):** Systematically documented the edge-of-stability phenomenon, identifying the progressive sharpening phase and the self-tuning of $\lambda_{\max}$ to $2/\eta$ across multiple architectures and datasets.
- **Keskar et al. (2017):** Earlier work showing that learning rate and batch size affect the sharpness of the minima found, which is consistent with the edge-of-stability mechanism.

## Connections

- [Implicit Bias](implicit-bias.md): The edge of stability provides a dynamical mechanism for the implicit bias toward flat minima -- the learning rate explicitly constrains the curvature of solutions found by gradient descent.
- [Random Matrix Theory](random-matrix-theory.md): The Hessian's spectral structure, particularly the outlier eigenvalues, determines when and how the edge-of-stability regime is reached. The bulk-plus-outliers structure means only a few directions reach the stability threshold.
- [Overparameterization](overparameterization.md): In overparameterized networks, most Hessian eigenvalues are near zero (flat directions), and only a small number of outlier eigenvalues reach $2/\eta$, making the edge-of-stability constraint apply to a low-dimensional subspace.
- [Neural Collapse](neural-collapse.md): The terminal phase dynamics where neural collapse occurs may interact with the edge-of-stability regime, as the curvature structure changes dramatically during the collapse phase.

## Open Questions

1. Can a rigorous theory explain why gradient descent continues to reduce the loss in the edge-of-stability regime, beyond the quadratic approximation?
2. What is the precise relationship between the edge-of-stability dynamics and generalization? Does operating at $\lambda_{\max} \approx 2/\eta$ provably improve test error?
3. How does the edge of stability interact with stochastic noise in mini-batch SGD? Does the noise change the self-tuning dynamics?
4. Can the edge of stability be exploited to design better learning rate schedules or adaptive optimizers?
5. How does the progressive sharpening rate depend on architecture, data distribution, and initialization?

---
title: "Theory A: Noether Conservation Laws for Gradient Flow"
category: angle
status: Phase 2 candidate
related: [conservation-laws.md, edge-of-stability.md, overparameterization.md, implicit-bias.md]
priority: 1 (highest feasibility, concrete math, testable predictions)
---

# Theory A: Noether Conservation Laws for Gradient Flow

## Core Thesis

The continuous symmetries of neural network loss functions (rescaling between layers, neuron permutations, hidden rotations) generate, via a Noether-type theorem adapted for gradient flow, a family of conserved quantities that constrain optimization trajectories to a low-dimensional submanifold where the landscape is effectively quasi-convex.

## Why This Could Resolve the Paradox

The paradox is: GD navigates an N-dimensional non-convex landscape. If conservation laws confine the trajectory to an (N - L + 1)-dimensional submanifold M_C (where L is the number of layers), then the *effective* optimization problem is lower-dimensional and potentially much better structured. The key insight: the non-convexity that makes the full landscape intractable may not exist on the constrained manifold.

This connects to: why GD finds similar solutions from different initializations (they're on the same M_C), why mode connectivity exists (paths between solutions stay on M_C), and why the edge of stability emerges (conservation laws constrain the Hessian spectrum).

## Mathematical Framework

### Known Result: Balancedness Invariant
For a 2-layer network f(x) = W_2 * ReLU(W_1 * x) with no bias (2-homogeneous), gradient flow preserves:

$$C = \|W_2\|_F^2 - \|W_1\|_F^2 = \text{const}$$

This is because ReLU networks are positively homogeneous: f(x; alpha*W_1, W_2/alpha) = f(x; W_1, W_2), generating a continuous symmetry.

### Proposed Extension
For an L-layer network with no bias:

$$f(x) = W_L \cdot \sigma(W_{L-1} \cdot \sigma(\cdots \sigma(W_1 \cdot x)))$$

**Conjecture A.1**: Gradient flow preserves L-1 independent conserved quantities:

$$C_l = \|W_{l+1}\|_F^2 - \|W_l\|_F^2, \quad l = 1, \ldots, L-1$$

These constrain the trajectory to M_C = {theta : C_l(theta) = C_l(theta_0) for all l}.

**Conjecture A.2**: On M_C, the loss function L restricted to M_C has no spurious local minima for sufficiently wide networks (width m > m*(n, d, L)).

**Conjecture A.3**: The geodesic convexity defect of L|_{M_C} vanishes as width m -> infinity, with rate O(1/sqrt(m)).

### Proof Strategy
1. Prove C_l conservation for 2-layer (known) and extend to L layers using induction on depth
2. Derive the full Lie algebra of symmetries for bias-free homogeneous networks
3. Apply Noether's theorem for gradient flows (dissipative Noether theorem)
4. Characterize M_C geometrically (is it a manifold? what is its curvature?)
5. Analyze L|_{M_C}: count critical points, estimate Hessian, bound non-convexity
6. Show width -> infinity limit makes M_C increasingly convex

### Connection to Existing Work
- Builds on: Zhao et al. (2022) symmetry teleportation, Simsek et al. (2021) extra neuron connectivity
- Extends: Known balancedness to full Lie algebra + landscape implications
- Novel: The claim that conservation laws make the *restricted* landscape quasi-convex is new

## Experimental Tests

1. **Conservation verification**: Track C_l during training for bias-free networks at small lr (approximate gradient flow). Check drift < 1%.
2. **Bias as control**: With bias=True, C_l should NOT be conserved. Verify.
3. **Discretization**: Conservation should degrade with large lr. Quantify.
4. **Trajectory dimension**: PCA on training trajectory should reveal low effective dimension consistent with conservation constraints.
5. **Manifold landscape**: Compare optimization difficulty on M_C vs full space.

## Open Questions
- Are there conserved quantities beyond the layer-pair norms?
- What happens when bias is present (most of the symmetry breaks)?
- How does batch normalization interact with conservation laws?
- Can conservation laws explain the edge of stability?

## Feasibility: 5/5 | Novelty: 4/5 | Impact: 5/5

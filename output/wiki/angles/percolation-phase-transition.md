---
title: "Theory B: Percolation Phase Transition for Mode Connectivity"
category: angle
status: Phase 2 candidate
related: [mode-connectivity.md, overparameterization.md, energy-landscape.md, spin-glass-landscape.md]
priority: 2 (clean framework, sharp predictions, directly testable)
---

# Theory B: Percolation Phase Transition for Mode Connectivity

## Core Thesis

There exists a sharp phase transition in network width m at which the sublevel set S_epsilon = {theta : L(theta) <= epsilon} undergoes a percolation transition -- from exponentially many disconnected components to a single connected "giant component." This critical width m* = Theta(n * d / epsilon) coincides with the interpolation threshold, and the transition is sharp (width window O(sqrt(m*))).

## Why This Could Resolve the Paradox

Mode connectivity is observed but has no first-principles explanation. If we can prove a percolation theorem, it would explain:
1. **Why GD works**: Above m*, the landscape is connected, so GD cannot get trapped in isolated basins
2. **Why overparameterization helps**: It pushes the network past the percolation threshold
3. **Why mode connectivity appears suddenly with width**: It's a phase transition, not a gradual change
4. **Why permutation alignment helps**: It accounts for the symmetry group before testing connectivity

This is the first theory to treat mode connectivity as a *phase transition* rather than a property, giving it a sharp mathematical characterization.

## Mathematical Framework

### Setup
- Network: f(x; theta) = W_2 * ReLU(W_1 * x), where W_1 in R^{m x d}, W_2 in R^{K x m}
- Data: (x_i, y_i)_{i=1}^n with x_i in R^d
- Loss: L(theta) = (1/n) sum_i l(f(x_i; theta), y_i)
- Sublevel set: S_epsilon = {theta in R^N : L(theta) <= epsilon}

### Main Conjecture

**Conjecture B.1 (Percolation Transition)**: There exist constants c_1, c_2 > 0 (depending on the data distribution and loss function) such that:
- If m < c_1 * n * d, then S_epsilon has at least exp(c * m) connected components for some c > 0
- If m > c_2 * n * d, then S_epsilon is path-connected

Moreover, the connectivity probability P(S_epsilon is connected) transitions from 0 to 1 in a window of width O(sqrt(n * d)).

**Conjecture B.2 (Linear Connectivity After Alignment)**: For m > c_2 * n * d, any two points theta_1, theta_2 in S_epsilon satisfy: there exists a permutation pi in S_m such that the linear path (1-t)*theta_1 + t*pi(theta_2) stays within S_{epsilon + delta} for delta = O(1/m).

### Proof Strategy

1. **Random geometric graph mapping**: 
   - Sample M points uniformly from S_epsilon
   - Connect theta_i, theta_j if the linear path between them stays in S_{2*epsilon}
   - This defines a random geometric graph G(M, r) in R^N
   
2. **Packing number estimate**: 
   - For m < m*: S_epsilon is a thin algebraic variety, packing number ~ exp(c*m)
   - For m > m*: S_epsilon is a thick convex-like body, packing number ~ 1
   - Key tool: dimension counting. For m neurons fitting n data points, the constraint is n*K equations in m*(d+K) unknowns. When m*(d+K) >> n*K (overparameterization), the solution set is a high-dimensional manifold.

3. **Penrose-type theorem**: Apply the connectivity threshold result for random geometric graphs (Penrose 1999): connectivity occurs when the expected degree exceeds log(M)/M.

4. **Sharp transition**: Use second-moment methods to show the window width is O(sqrt(m*)).

### Connection to Existing Work
- Explains: Draxler et al. (2018), Garipov et al. (2018) mode connectivity
- Extends: Freeman & Bruna (2017) connected level sets (they show connectivity grows with width, we show a *sharp transition*)
- Connects to: Entezari et al. (2022) conjecture (our Conjecture B.2 is a strengthened version with explicit width requirement)
- Novel: The percolation framework and sharp threshold are new

## Experimental Tests

1. **Width sweep**: Train models at widths [4, 8, 16, 32, 64, 128, 256, 512, 1024], measure pairwise linear interpolation barriers. Plot barrier vs width -- look for sharp transition.
2. **Permutation alignment**: Apply Git Re-Basin style alignment, then measure barriers. Transition should be sharper after alignment.
3. **Phase transition fitting**: Fit a sigmoid/logistic model to P(barrier < threshold) vs width. Extract m* and transition width.
4. **Dataset dependence**: Test on Gaussian mixture (vary n, d, K) to see if m* scales as predicted (m* ~ n*d).
5. **Architecture universality**: Test on deep MLPs and CNNs to see if the transition is universal.

## Open Questions
- Does the percolation threshold depend on the loss function (CE vs MSE)?
- Is the transition sharper for certain data distributions?
- Does the percolation picture extend to transformers?
- What is the role of the initialization distribution?
- Can we compute m* exactly for specific architectures?

## Feasibility: 4/5 | Novelty: 5/5 | Impact: 5/5

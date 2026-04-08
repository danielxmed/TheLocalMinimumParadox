---
title: "Theory C: Tropical Morse Theory of ReLU Landscapes"
category: angle
status: Phase 2 candidate
related: [tropical-geometry.md, spin-glass-landscape.md, random-matrix-theory.md, mode-connectivity.md]
priority: 3 (most novel, highest risk/reward)
---

# Theory C: Tropical Morse Theory of ReLU Landscapes

## Core Thesis

ReLU networks compute piecewise-linear functions, and their loss landscapes inherit this PL structure. Classical (smooth) Morse theory -- which underpins the Kac-Rice critical point counting framework -- does not directly apply. We develop a *tropical Morse theory* specifically for ReLU loss landscapes, where critical cells (not smooth critical points) are the fundamental objects. A tropical Kac-Rice formula counts these critical cells, and the tropical discriminant of the network's Newton polytope governs when sublevel sets transition from many disconnected components to a single connected basin.

## Why This Could Resolve the Paradox

The spin-glass / Kac-Rice approach (Choromanska et al. 2015) uses smooth random field theory on functions that are actually piecewise-linear. This is a fundamental mismatch. By developing Morse theory native to the PL geometry of ReLU networks, we:
1. Remove the incorrect smoothness assumption
2. Work with the *actual* structure of the landscape, not an approximation
3. Connect the combinatorial structure of linear regions to landscape benignity
4. Provide a sharp characterization of when critical cells disappear

This would be the first landscape theory that respects the intrinsic geometry of ReLU networks.

## Mathematical Framework

### Tropical Semiring and PL Structure

The tropical semiring (R union {infinity}, min, +) replaces classical arithmetic. A ReLU network with L layers of width m partitions the parameter-input space into at most (2m)^{L*d} cells defined by activation patterns (which neurons are active/inactive). On each cell, the network function is an affine function of the parameters.

### Key Definitions

**Definition (Activation Pattern)**: For a ReLU network f(x; theta) and input x, the activation pattern sigma(x, theta) in {0, 1}^{m*L} records which neurons have positive pre-activation. Each activation pattern defines a linear region in (theta, x)-space.

**Definition (Tropical Critical Cell)**: A critical cell of the PL loss function L(theta) is a maximal face of the canonical polyhedral complex where L is not locally linear -- i.e., where the gradient is not well-defined because multiple linear pieces meet. The index of a critical cell is defined by the local topology change it induces.

**Definition (Tropical Morse Number)**: mu_k(epsilon) = number of critical cells of index k with L(theta) <= epsilon.

### Main Conjecture

**Conjecture C.1 (Tropical Kac-Rice Formula)**: For a 2-layer ReLU network with width m, data (x_i, y_i)_{i=1}^n in general position, and MSE loss:

$$\mathbb{E}[\mu_0(\varepsilon)] \leq \binom{n}{m} \cdot P(\text{activation pattern yields loss} \leq \varepsilon)$$

where the expectation is over random initialization of weights.

**Conjecture C.2 (Tropical Phase Transition)**: There exists m_trop = O(n * d) such that:
- For m < m_trop: E[mu_0(epsilon)] grows exponentially in m
- For m > m_trop: E[mu_0(epsilon)] = 1 (single connected low-loss basin)

The transition is governed by the tropical discriminant of the network's Newton polytope -- specifically, when the number of activation patterns compatible with low loss exceeds the combinatorial complexity threshold.

### Proof Strategy

1. **Formalize the polyhedral decomposition**: The parameter space R^N is divided into cells by the activation patterns. On each cell, L(theta) = ||A_sigma * theta - b_sigma||^2 where A_sigma, b_sigma depend on data and activation pattern sigma.

2. **Define PL Morse theory**: Critical cells occur at boundaries between cells. A critical cell of index k means k "ascending" and N-k "descending" linear pieces meet.

3. **Count critical cells**: The number of critical cells is bounded by the number of cell boundaries, which is at most C(n, m) * 2^{m*L} (combinatorial in width).

4. **Apply tropical discriminant theory**: The tropical discriminant Delta_trop of the Newton polytope controls when affine pieces can achieve loss <= epsilon. When m is large enough, all activation patterns are compatible with zero loss (overparameterization), and the critical cells merge into a single basin.

5. **Show the transition is sharp**: Use concentration arguments on the number of compatible activation patterns.

### Connection to Existing Work
- Extends: PL Morse theory for ReLU networks (recent work on critical cells replacing smooth critical points)
- Complements: Choromanska et al. (2015) spin-glass Kac-Rice (we fix the smoothness mismatch)
- Connects to: Freeman & Bruna (2017) connected level sets (topological perspective)
- Novel: Tropical Kac-Rice formula and tropical discriminant-based phase transition are entirely new

## Experimental Tests

1. **Linear region counting**: For small networks, enumerate all activation patterns as a function of width. Verify combinatorial scaling.
2. **Critical cell enumeration**: For tiny networks (width 2-10, depth 2), exhaustively find all critical cells by checking all cell boundaries. Count them.
3. **Tropical Morse number vs width**: Plot mu_0(epsilon) vs m for fixed epsilon. Look for transition.
4. **PL structure visualization**: Visualize the 2D loss landscape showing the PL cell boundaries, critical cells, and how they evolve with width.
5. **Comparison with smooth Morse**: Compare our tropical critical cell count with the Kac-Rice prediction from the spin-glass approach.

## Open Questions
- Can the tropical Kac-Rice formula be made sharp (not just an upper bound)?
- Does the theory extend to non-ReLU activations (leaky ReLU, GELU approximations)?
- What is the relationship between tropical critical cells and the Hessian spectrum?
- Can tropical geometry explain the edge of stability?
- Is there a tropical version of the Bray-Dean theorem?

## Feasibility: 3/5 | Novelty: 5/5 | Impact: 5/5

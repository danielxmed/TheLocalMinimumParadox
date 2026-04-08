---
title: "Angle 4: Edge of Stability as Self-Organized Criticality"
category: angle
status: candidate (not selected for deep dive)
related: [edge-of-stability.md, conservation-laws.md, implicit-bias.md]
---

# Angle 4: Edge of Stability as Self-Organized Criticality

## Thesis
The edge of stability (lambda_max -> 2/eta) is a self-organized critical (SOC) mechanism analogous to the Bak-Tang-Wiesenfeld sandpile model. At this critical state, the system exhibits power-law correlations in parameter space, enabling efficient landscape navigation. The GD trajectory at the edge of stability performs an optimal exploration-exploitation tradeoff: stable enough to descend, unstable enough to escape poor basins.

## Key Mathematical Idea
- Model the largest Hessian eigenvalue lambda_max(t) as a stochastic process driven by the interaction between gradient descent and the loss landscape curvature
- Show that the fixed point lambda_max = 2/eta is an attractor of this process
- Prove that at this fixed point, the system is at a phase transition between convergent (lambda_max < 2/eta) and divergent (lambda_max > 2/eta) dynamics
- The critical state maximizes the rate of loss decrease subject to stability constraints

## Why Not Selected for Deep Dive
- Overlaps significantly with Theory A (conservation laws could explain EoS)
- SOC is notoriously hard to prove rigorously
- The connection to generalization remains speculative
- But: could be combined with Theory A as a consequence of conservation law constraints

## Novelty: 4/5 | Feasibility: 3/5 | Impact: 4/5

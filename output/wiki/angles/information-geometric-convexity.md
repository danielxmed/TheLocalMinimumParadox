---
title: "Angle 6: Information-Geometric Convexity"
category: angle
status: candidate (not selected for deep dive)
related: [neural-tangent-kernel.md, implicit-bias.md, mean-field-theory.md]
---

# Angle 6: Information-Geometric Convexity

## Thesis
Under the Fisher-Rao metric (the natural geometry of statistical models), the neural network loss landscape is geodesically convex or has bounded non-convexity (quantified by a geodesic convexity defect delta). Gradient descent in Euclidean coordinates implicitly approximates natural gradient descent in this geometry, explaining its effectiveness.

## Key Mathematical Idea
- The Fisher information matrix F(theta) defines a Riemannian metric on parameter space
- The natural gradient is F^{-1} * grad L, which accounts for the curvature of the statistical manifold
- If the loss is geodesically convex under this metric, then natural gradient descent converges to the global minimum
- Euclidean GD approximates natural GD when F is approximately proportional to identity -- which may hold in overparameterized networks due to the NTK structure

## Why Not Selected for Deep Dive
- The Fisher-Rao metric is degenerate for overparameterized networks (rank deficient)
- Computing geodesics in this metric is intractable for deep networks
- Amari's information geometry gives qualitative insights but quantitative results are hard
- The approximation "Euclidean GD ~ natural GD" is only accurate in specific regimes (near NTK)

## Novelty: 3/5 | Feasibility: 2/5 | Impact: 4/5

---
title: "Angle 5: Renormalization Group Flow Along Depth"
category: angle
status: candidate (not selected for deep dive)
related: [neural-tangent-kernel.md, overparameterization.md, energy-landscape.md]
---

# Angle 5: Renormalization Group Flow Along Depth

## Thesis
Each layer of a deep network acts as an RG transformation that coarse-grains the data representation. Fixed points of this RG flow correspond to trainable architectures where each layer's optimization landscape is self-similar. The loss landscape inherits a hierarchical structure where gradient descent is efficient at each scale, analogous to how RG explains universality in statistical physics.

## Key Mathematical Idea
- Define the RG operator T_l: representation at layer l -> representation at layer l+1
- Show that training dynamics at each layer can be decomposed into an RG flow component (changing the representation) and a residual (fitting within the representation)
- Fixed points of T_l correspond to "critical" architectures where representations are scale-invariant
- At these fixed points, the loss landscape restricted to each layer has bounded non-convexity

## Why Not Selected for Deep Dive
- Very ambitious: connecting RG to optimization is a long-standing aspiration
- The RG analogy is attractive but formalizing it is extremely difficult
- Existing work (e.g., Beny, Li-Schwab) provides analogies but not rigorous results
- Better as a long-term direction after establishing Theory A, B, or C

## Novelty: 4/5 | Feasibility: 2/5 | Impact: 5/5

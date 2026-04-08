# Theory: [Title]

**Status:** [Conjecture with Evidence | Proof Sketch | Complete Proof]
**Date:** [YYYY-MM-DD]
**Novelty Check:** [Confirmed novel via web search on DATE | Partially overlaps with PAPER]

---

## 1. Motivation and Intuition

[2–4 paragraphs explaining the core idea in plain language. What gap in the existing theory does this address? What is the key insight? Where did the idea come from? Why should anyone care?]

### Connection to the Central Paradox

[How does this theory contribute to explaining why gradient descent works in non-convex neural network optimization? Which aspect of the paradox does it address?]

---

## 2. Setup and Notation

[Define all mathematical objects, notation, and assumptions before stating the result.]

**Setting:**
- Network architecture: [...]
- Loss function: [...]
- Data distribution: [...]
- Optimization algorithm: [...]

**Notation:**
- $\theta \in \mathbb{R}^N$: [parameters]
- $L(\theta)$: [loss function]
- [... all symbols used ...]

**Assumptions:**
1. **(A1)** [First assumption — state precisely with quantifiers]
2. **(A2)** [Second assumption]
3. [...]

*Discussion of assumptions:* [Are they realistic? Which ones are essential vs. technical conveniences? How do they compare to assumptions in related work?]

---

## 3. Main Result

**Theorem [N].** *[Complete, precise statement of the theorem or conjecture. Include all quantifiers, assumptions, and the conclusion. A mathematician should be able to determine exactly what is being claimed.]*

**Corollary [N.1].** *[If applicable — important consequences.]*

### Interpretation

[What does this theorem mean in concrete terms? What does it predict about training dynamics, landscape structure, or generalization? How does it connect to empirical observations?]

### Comparison with Existing Results

[How does this relate to NTK theory, mean field theory, spin glass results, mode connectivity, implicit bias, etc.? Is it strictly stronger/weaker/orthogonal?]

---

## 4. Proof

*[Rigor level: Complete | Sketch with N gaps | Heuristic]*

### Proof Overview

[1 paragraph describing the high-level proof strategy before diving into details.]

### Step 1: [Description]

[Detailed argument...]

### Step 2: [Description]

[Detailed argument...]

[Continue as needed...]

### [GAP: Description] (if applicable)

*What needs to be shown:* [Precise statement]
*Why we believe it's true:* [Evidence]
*Suggested approach:* [How to close the gap]
*Impact if not closed:* [What happens to the overall result]

### Conclusion of Proof

[Combine the steps to reach the claimed result.] □

---

## 5. Computational Evidence

### Experiment 1: [Name]

**Setup:** [Describe the experimental configuration — architecture, data, hyperparameters]
**Prediction:** [What the theory predicts should happen, quantitatively]
**Code:** `output/code/exp_[name]_v[N].py`
**Results:** `output/experiments/[name]/`

| Configuration | Predicted | Observed (mean ± std) | Seeds |
|---|---|---|---|
| [config 1] | [value] | [value ± error] | 5 |
| [config 2] | [value] | [value ± error] | 5 |
| [...] | [...] | [...] | [...] |

**Figure:** ![Description](../figures/[figure-name].png)

**Analysis:** [Do the results support the theory? Quantitative comparison. Discussion of any discrepancies.]

### Experiment 2: [Name]

[Same structure...]

### Experiment 3: Assumption Ablation / Falsification Attempt

**Purpose:** [Test what happens when assumptions are violated / attempt to find counterexample]
**Setup:** [Configuration that violates assumption AN]
**Prediction:** [Theory predicts this should fail / the predicted behavior should break]
**Results:** [Did it break as predicted? If not, what does that mean?]

---

## 6. Limitations and Open Questions

### Known Limitations
1. [Limitation 1: e.g., "The proof only handles 2-layer networks..."]
2. [Limitation 2: e.g., "Assumption A2 may not hold for batch normalization..."]
3. [...]

### Failure Modes
- [Settings where the theory is known to fail, from falsification attempts]

### Open Questions
1. [Can assumption A3 be removed?]
2. [Does the result extend to deeper networks?]
3. [...]

---

## 7. References

[Only verified, real papers. Format: Author(s), "Title," Venue/arXiv, Year.]

1. [Reference 1]
2. [Reference 2]
3. [...]

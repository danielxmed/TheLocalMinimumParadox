# Proof Standards for Theory Development

This document defines what constitutes a valid proof, proof sketch, and conjecture in this research context.

---

## Hierarchy of Rigor

From most to least rigorous, we recognize the following levels. **All are valuable.** The goal is to be as rigorous as possible while being honest about what level you've achieved.

### Level 1: Complete Proof
- Every step follows from the previous by standard mathematical reasoning.
- All assumptions are stated explicitly at the beginning.
- All intermediate results (lemmas) are proved or cited with a precise reference.
- A competent mathematician in the relevant field could verify every step.
- **This is the gold standard, but not always achievable in a research sprint.**

### Level 2: Proof with Identified Gaps (Target Level)
- The overall proof structure is complete.
- Most steps are rigorous.
- At most 3 gaps are explicitly marked as **[GAP]** with:
  - A precise statement of what needs to be proved to close the gap
  - An explanation of why you believe the gap is closable
  - A suggested approach for closing it
  - Known tools/results that might help
- **This is the target level for this project.** A proof sketch at this level, combined with strong computational evidence, is a substantial research contribution.

### Level 3: Proof Sketch
- The key ideas of the proof are presented.
- The most technically novel steps are worked out.
- Routine steps are indicated but not fully detailed ("by standard concentration inequalities...").
- The overall logical flow is clear.
- **Acceptable when combined with very strong computational evidence.**

### Level 4: Heuristic Argument
- The argument relies on unproven but plausible assumptions.
- Physical intuition or informal reasoning fills in gaps.
- The logic is sound *given* the assumptions.
- **Acceptable as a precursor to rigorous development. Must be labeled as heuristic.**

### Level 5: Conjecture with Evidence
- A precise mathematical statement is made.
- Computational experiments strongly support it.
- No proof or proof sketch is provided.
- **Valuable if the conjecture is surprising and well-supported. Must be labeled as conjecture.**

---

## Proof Writing Standards

### Structure of a Proof

```
**Theorem [N].** [Precise statement with quantifiers and assumptions.]

*Proof.* [The proof body.]

1. [Setup: define notation, restate key assumptions]
2. [Key step 1: the main technical insight]
   - [Sub-steps as needed]
3. [Key step 2: ...]
   ...
N. [Conclusion: combine the steps to reach the claimed result]

□
```

### Rules for Every Proof Step

1. **Justify every equation.** After each equation or inequality, indicate why it holds:
   - "by assumption (A3)"
   - "by Lemma 2"
   - "by the triangle inequality"
   - "by Jensen's inequality applied to the convex function g"
   - "by the chain rule"

2. **State all assumptions up front.** Don't introduce new assumptions mid-proof. If you realize a new assumption is needed, go back and add it to the theorem statement.

3. **Define all notation before using it.** Every symbol that appears in the proof must be defined either in the theorem statement, in a preliminaries section, or at first use.

4. **Be explicit about quantifiers.** "For all ε > 0, there exists N such that..." is precise. "For small ε, N is large enough that..." is not.

5. **Handle edge cases.** If your proof breaks down at a boundary (e.g., width = 1, depth = 1, zero data), either handle the edge case or explicitly exclude it.

---

## Common Proof Techniques for This Domain

### Techniques you're likely to use:

- **Concentration inequalities**: Hoeffding, Bernstein, sub-Gaussian tail bounds, matrix concentration (Vershynin). For showing random quantities concentrate near their expectations.

- **Random matrix theory**: Marchenko-Pastur law, Wigner semicircle, Stieltjes transform. For analyzing Hessian spectra.

- **Covering/packing arguments**: ε-net arguments for uniform convergence over parameter sets.

- **Morse theory / critical point analysis**: Kac-Rice formula, Betti number bounds, sublevel set topology.

- **Comparison arguments**: Showing that a complex process is well-approximated by a simpler one (e.g., GD trajectory stays close to gradient flow ODE).

- **Coupling arguments**: Showing two stochastic processes evolve similarly by constructing them on the same probability space.

- **Induction on architecture**: Proving a property for depth L by assuming it for depth L-1. Particularly natural for ResNets.

- **Energy methods**: Constructing a Lyapunov function that decreases along the optimization trajectory.

- **Convexity lifting**: Showing a non-convex problem in parameter space is convex in a different space (function space, measure space, kernel space).

### When adapting known proof techniques:

1. Clearly state which result you're adapting and from which paper.
2. Identify exactly where the adaptation diverges from the original.
3. Verify that the adapted steps are valid — don't cargo-cult proof steps from a different setting.

---

## Handling Gaps Honestly

A gap in a proof is not a failure — it's a *precisely identified research question*. Gaps should be documented as:

```
**[GAP: Name/description]**

*What needs to be shown:* [Precise mathematical statement]

*Why we believe it's true:*
- [Computational evidence: "In experiments with N = 100, 500, 2000, the bound holds with room to spare."]
- [Analogy: "The analogous result in the matrix completion setting was proved by [Author, Year] using [technique]."]
- [Heuristic argument: "Under the simplifying assumption that X is Gaussian, the result follows from..."]

*Suggested approach for closing:*
- [Specific technique or tool that might work]

*Impact if gap cannot be closed:*
- [What happens to the overall result — does it become a weaker theorem, or does the entire argument collapse?]
```

---

## What Does NOT Count as a Proof

1. **"It is easy to show that..."** — If it's easy, show it. If it's not easy, prove it or mark it as a gap.
2. **Experimental evidence alone.** Experiments support but do not prove. 10,000 confirming experiments cannot substitute for a proof.
3. **Appeal to authority.** "This is known" requires a citation. "Experts believe" is not evidence.
4. **Dimensional/scaling arguments without rigor.** "This scales as O(N²) because there are N² pairs" is an intuition, not a proof. Make it rigorous.
5. **Unverified citations.** "By Theorem 3.2 of [Smith 2023]" requires that Smith 2023 exists and actually contains a Theorem 3.2 that says what you claim.

---

## Special Note on Probabilistic Statements

Many results in this domain are probabilistic: "With probability at least 1 - δ over random initialization..." Be precise about:

- **What is random?** The initialization? The data? The SGD noise?
- **What probability space?** Over which distribution?
- **What is the failure probability?** How does δ scale with the relevant parameters?
- **Is the result almost-sure, in expectation, or high-probability?** These are different and require different proof techniques.

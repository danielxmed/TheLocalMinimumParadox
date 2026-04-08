# Scientific Integrity Rules

These rules are **non-negotiable**. They apply to every phase of the research.

---

## 1. Citation Integrity

- **NEVER fabricate, invent, or hallucinate a citation.** This is the cardinal sin of scientific writing.
- Every paper you reference must be verified to exist via web search before citing.
- If you remember a result but cannot find the paper: write "a known result (citation needed)" or "it has been shown [citation needed] that..."
- If no prior work supports a claim: write "to our knowledge, no prior work has established..." Do not invent a paper to fill the gap.
- When citing, always include: author(s), title, venue or arXiv identifier, year.
- Maintain `output/literature/relevant-papers.md` as a running annotated bibliography.
- When searching for a paper, try multiple query formulations. If 3+ different searches fail to find it, the paper likely doesn't exist as you remember it.

## 2. Honesty About Results

- **Report all results**, including negative ones, failures, and experiments where your theory's predictions were wrong.
- Never cherry-pick seeds, hyperparameters, or experimental configurations to make results look better.
- If your theory works in setting A but fails in setting B, report both. Then analyze *why* it fails in B.
- If a theory you spent hours developing turns out to be wrong, document *why* it's wrong. This is valuable.
- Never present a proof sketch as a complete proof. Always label the level of rigor accurately (see `methodology/proof-standards.md`).
- Clearly distinguish between: (a) proven theorems, (b) conjectures with evidence, (c) heuristic arguments, (d) speculations.

## 3. Reproducibility

- Every experiment must be fully reproducible from saved code + config.
- Record random seeds, library versions, and hardware (CPU/GPU).
- Save all code versions — never overwrite.
- Save raw data (.npy) alongside processed results (.json).
- Another researcher (human or AI) should be able to rerun your experiments and get the same results.

## 4. Intellectual Honesty About Novelty

- Before claiming novelty, search the literature with at least 5 different query formulations.
- If you find prior work that partially overlaps: cite it, explain the overlap, and precisely state what is new.
- "Novel" means the specific claim in the specific generality has not appeared before. Minor reformulations of known results are not novel.
- If your "discovery" turns out to be a known result: document this in the research log and pivot. This is progress, not failure.

## 5. Honesty About Limitations

- Every theory must include an explicit "Limitations" section listing:
  - Assumptions that may be unrealistic
  - Settings where the theory is known or suspected to fail
  - Gaps between the theory's predictions and practical deep learning
  - Open questions that remain
- Do not overclaim. If your result holds for 2-layer networks with Gaussian data, say so. Do not claim it "likely extends to deep networks on real data" without evidence.

## 6. No Self-Deception

- When computational experiments agree with your theory, ask: "Is there a simpler explanation for this observation that doesn't require my theory?"
- When designing experiments, include control experiments that your theory does NOT predict should succeed. If the controls also succeed, your theory may not be the explanation.
- Be especially suspicious of theories that are unfalsifiable — that make no testable prediction that could come out wrong. These are not scientific theories.

## 7. Resource Responsibility

- Do not run experiments that will obviously exceed available compute.
- If an experiment is taking much longer than expected, terminate it and reassess.
- Do not install unnecessary packages.
- Clean up failed experiments (but keep the logs).

## 8. Research Log Discipline

- Update `output/research-log.md` continuously throughout the research process.
- Every significant decision must be documented with reasoning.
- Every dead end must be documented with lessons learned.
- Every insight — even small ones — should be recorded. Insights compound.
- Use timestamps for all entries.
- The research log should tell the complete story of how you arrived at your results, including all the wrong turns.

# The Local Minimum Paradox — Project Instructions

You are an autonomous theoretical research agent. Your mission is defined in `context/INITIAL_PROMPT.md`. Read it fully before doing anything else.

**If this is a continuation session:** Read `NEXT_PROMPT.md` first -- it has the full project state, findings, and suggested next steps from the previous session. Then come back here for operational instructions.

## Quick Reference

- **Mission**: Discover novel theories (with proofs and computational evidence) explaining why gradient descent works in non-convex neural network optimization.
- **Background knowledge**: `context/THE_PARADOX.md`
- **Research methodology**: `methodology/creative-thinking.md`, `methodology/proof-standards.md`
- **Rules**: `rules/integrity.md` (non-negotiable)
- **Templates**: `templates/theory-template.md`
- **Output directory**: `output/` (already populated with prior work)
- **Continuation brief**: `NEXT_PROMPT.md` (read this if prior work exists)

## How to Begin

**First session (output/ is empty):**
1. Read `context/INITIAL_PROMPT.md` completely.
2. Follow the instructions there — they define your entire research loop.
3. Start with Phase 0 (Deep Immersion), then enter the loop.

**Continuation session (output/ has prior work):**
1. Read `NEXT_PROMPT.md` for the state of the project and next steps.
2. Read `output/wiki/overview.md` for the current theory status.
3. Skim the end of `output/research-log.md` for the latest findings.
4. Pick a direction and continue the research loop.
5. Do NOT re-run completed experiments or re-read raw sources already compiled into the wiki.

## Critical Rules

- **Never hallucinate citations.** Verify every paper exists via web search.
- **Never overwrite experiment code.** Version everything (_v1, _v2, ...).
- **Always update `output/research-log.md`.** It's your running journal.
- **Report negative results.** Dead ends are valuable data.
- **Be creative first, rigorous second.** Phase 1 is for imagination; Phase 3 is for proofs.
- **Always update the wiki.** It's the persistent compiled knowledge base.
- **Update NEXT_PROMPT.md** at the end of each session with current state and recommendations.

---

## How Knowledge Is Stored

This project uses a three-layer knowledge architecture to ensure continuity across sessions and prevent re-deriving knowledge from scratch.

### Layer 1: Raw Sources (immutable)

These are the foundational materials. Never modify them.

```
context/
  INITIAL_PROMPT.md          # Mission, research loop, quality gate
  THE_PARADOX.md             # Survey of existing theories and gaps
methodology/
  creative-thinking.md       # Structured creativity techniques
  proof-standards.md         # 5 levels of rigor
rules/
  integrity.md               # 8 non-negotiable rules
templates/
  theory-template.md         # 7-section theory document format
```

### Layer 2: The Wiki (compiled knowledge, LLM-maintained)

The wiki at `output/wiki/` is the **persistent, interlinked knowledge base**. It sits between raw sources and final outputs. The LLM maintains it; the human reads it.

```
output/wiki/
  overview.md               # START HERE -- central paradox, our theories, current status
  index.md                  # Catalog of every wiki page with one-line descriptions
  log.md                    # Chronological record of wiki updates
  concepts/                 # 12 entity pages (NTK, mean field, spin glass, etc.)
  angles/                   # 7 candidate theory pages
  experiments/              # (future) experiment result pages
  literature/               # (future) ingested paper summaries
  dead-ends/                # (future) documented failed approaches
```

**How to use the wiki:**
- **Starting a session:** Read `overview.md` + `index.md` to get the full picture.
- **After a literature search:** Create summary pages in `wiki/literature/`, update relevant concept pages, update `index.md`.
- **After an experiment:** Update the relevant theory's angle page and the experiment results section in `index.md`.
- **After a dead end:** Create a page in `wiki/dead-ends/` documenting what failed and why.
- **Periodically:** Run a "lint" pass -- check for contradictions, orphan pages, missing cross-references, stale claims.

The wiki follows the [LLM Wiki pattern](LLM-WIKI-PROMPT.md): the LLM does all the summarizing, cross-referencing, and maintenance. The human curates sources and directs the analysis.

### Layer 3: Final Outputs (research deliverables)

```
output/
  research-log.md            # Chronological journal of ALL decisions and findings
  theories/
    theory-1-*.md            # Formal theory documents (following templates/)
    theory-2-*.md
    theory-3-*.md
    synthesis-*.md           # How theories connect
  code/
    utils_v1.py              # Shared experimental toolkit (1312 lines)
    exp_*_v[N].py            # Experiment scripts (versioned, never overwritten)
  experiments/
    [name]/
      results.json           # Processed results
      config.json            # Full configuration for reproducibility
      *.npy                  # Raw numerical data
  figures/
    *.png, *.pdf             # Publication-quality figures (300 DPI)
  literature/
    relevant-papers.md       # Verified papers with notes
    bibliography.bib         # BibTeX entries
```

### Layer 4: Session Handoff

```
NEXT_PROMPT.md               # Written at end of each session
                              # Contains: project state, key findings, next steps
                              # Your future self reads this FIRST
```

**Update NEXT_PROMPT.md at the end of every session.** Include:
1. What was accomplished (theories, experiments, proofs)
2. What the key findings were (quantitative results, surprises, failures)
3. Specific next steps with rationale
4. Any new theoretical directions worth exploring
5. Updated file inventory

### How the layers connect

```
Raw Sources (context/, methodology/, rules/)
    |
    | compiled into
    v
Wiki (output/wiki/)          <-- the persistent knowledge graph
    |                             read this instead of re-reading raw sources
    | informs
    v
Final Outputs (theories/, experiments/, figures/)
    |
    | summarized in
    v
Session Handoff (NEXT_PROMPT.md)  <-- read this first in new sessions
```

The key principle: **knowledge is compiled once and kept current**, not re-derived every session. The wiki accumulates. The research log records the journey. NEXT_PROMPT.md captures the current state for the next agent.

---

## Experiment Code Standards

Every experiment script must:
- Begin with a metadata block (name, theory, prediction, date)
- Use seeds [42, 137, 256, 512, 1024]
- Save results as .json + .npy in `output/experiments/`
- Save config.json alongside results
- Never overwrite -- version as _v1, _v2, etc.
- Import shared utilities from `output/code/utils_v1.py`

See `context/INITIAL_PROMPT.md` lines 192-226 for the full specification.

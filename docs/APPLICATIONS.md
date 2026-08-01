# STRING Applications Included in the Final Release

## 1. Interactive six-agent reasoning-graph verification

The root server compiles natural-language reasoning into a graph, runs specialist reviews, conditionally invokes assumption analysis and arbitration, and exposes node-, edge-, finding-, and repair-level details in the interactive interface.

## 2. Chrome extension

`extensions/chrome/` captures selected webpage text and sends it to the local STRING endpoint. The extension is intentionally thin: API keys remain on the local server.

## 3. RAGTruth hallucination detection

The RAGTruth pipeline compares Raw Direct predictions with graph-based source/response alignment. `config/optimized_gate_330.json` contains the development-set relation-specific gate used in the final presentation materials. The score is not a locked-test estimate.

## 4. Reasoning features and journal impact

`applications/impact_factor/` collects article discussion sections, constructs STRING graphs, and exports structural and reasoning-quality metrics. Any association with journal impact is exploratory and vulnerable to confounding by field, article type, length, access, and selection effects.

## 5. AIME math self-revision

`applications/aime_self_revision/` uses STRING as an auditor of a generated solution. Detected graph defects and focused questions are fed into a revision step, after which the revised solution is compiled and audited again. This remains an LLM-centered loop; formal verification of eligible equations is future work.

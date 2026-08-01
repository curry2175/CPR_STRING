# Graph-native Discussion Architecture v036

## Objective

Make the graph an internal shared state and agent communication protocol rather than an output formatting instruction.

## Components

### 1. Semantic Compiler

- Reads the source chunk.
- Extracts source-faithful typed nodes and edges.
- Cannot emit issues or a correctness judgment.

### 2. Shared Reasoning Graph

- Stores compiled nodes and edges.
- Receives localized annotations and graph patches from specialists.
- Keeps specialist work scoped to node/edge ids.

### 3. Conditional specialists

- Logic: implication, contradiction, causal/temporal/necessity/scope relations.
- Evidence: grounding, strength mismatch, unsupported generalization/mechanism.
- Methodology: design, selection, time zero, attrition, estimand and adjustment.
- Quantitative: magnitude, subgroup, interaction, multiplicity, noninferiority/equivalence.

Logic and evidence are always independent reviewers. Methodology and quantitative agents are routed only when graph features indicate relevance.

### 4. Conditional Judge

The Judge is called only when:

- specialists cite overlapping nodes with different issue types;
- a specialist finding is explicitly uncertain; or
- confidence is below the threshold.

### 5. Existing deterministic postprocessor

The longstanding normalization, source-fidelity checks, structural pattern augmentation, issue verification, grouping, graph metrics and public output generation remain unchanged.

## Failure containment

- Specialist failure does not stop other specialists.
- Judge failure falls back to deduplicated specialist findings.
- Compiler-level architecture failure falls back to the legacy single-pass analyzer for that chunk.

## Compatibility principle

No frontend or external evaluation code must understand the internal architecture. Both paths return the same `DiscussionGraphOutput`, which is normalized into the same schema-version `0.27.0` public result.

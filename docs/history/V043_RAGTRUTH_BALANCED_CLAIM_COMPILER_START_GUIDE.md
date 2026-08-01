# v043 RAGTruth balanced claim compiler

## Run

```bat
RUN_RAGTRUTH_RAW_VS_DUAL_GRAPH_BALANCED_NANO_PILOT_WINDOWS.bat
```

The legacy pilot filenames also launch the same v043 evaluator:

```bat
RUN_RAGTRUTH_RAW_VS_DUAL_GRAPH_NANO_PILOT_WINDOWS.bat
RUN_RAGTRUTH_RAW_VS_DUAL_GRAPH_ATOMIC_NANO_PILOT_WINDOWS.bat
```

## Compared methods

```text
nano_raw_direct
Source + Response -> direct hallucination span prediction

nano_dual_graph
Source -> task-complete Evidence Graph
Response -> balanced complete-proposition Claim Graph
Graphs -> alignment -> complete error-clause projection
```

Every API call in both conditions uses `gpt-5.4-nano`.

## Compiler behavior

A sentence is a container. A node is the smallest complete proposition that can receive one factual verdict.

Good:

```text
The trial enrolled 120 patients
improved overall survival
  normalized_claim: The trial improved overall survival
```

Rejected or locally repaired:

```text
finally
Water
Generation 2
too short
```

A whole sentence remains one node when it expresses one inseparable proposition.

## Local repair

The first compiler output is checked in Python. A second nano call occurs only when there is:

- an explicit multi-claim node
- an exact quote that cannot be found
- an incomplete lexical fragment
- a duplicate node id
- a node marked non-eligible
- a reported factual coverage omission
- factual response content but no claim node

The repair call receives only the problematic nodes and related sentences. It returns replacements, additions, or drops. It does not rewrite valid nodes.

A phrase merely containing `and`, `but`, or a relative clause is not sufficient to trigger repair.

## Console diagnostics

Each case prints:

```text
Compiler claims=... | sentences=... | claims/sentence=... |
whole-sentence nodes=... | fragments=... | refinement=YES/local_node_patch
```

Healthy behavior generally means:

- `fragments=0` after repair
- refinement occurs only in a subset of cases, not every case
- whole-sentence nodes remain possible for genuinely single-proposition sentences
- fewer clean-case false positives from transitions and isolated modifiers

## Cache

The default cache file remains:

```text
outputs\ragtruth_raw_vs_dual_graph_nano\generation_cache_v040.json
```

Cache keys include prompt versions. Therefore:

- compatible Raw Direct outputs can be reused
- v043 Source Graph, Response Graph, and Alignment components receive new keys
- v042 graph outputs are not silently reused

## Output

```text
outputs\ragtruth_raw_vs_dual_graph_nano\
  ragtruth_raw_vs_dual_graph_nano_YYYYMMDD_HHMMSS\
    result.json
    cases.jsonl
    report.html
    failures.json
    status.json
```

Useful v043 summary fields include:

```text
incomplete_fragment_node_rate_percent
compiler_refinement_case_rate_percent
problem_text_expanded_to_complete_claim_count
discarded_nonclaim_problem_text_count
clean_false_positive_rate_percent
char_precision_percent
char_recall_percent
char_f1_percent
```

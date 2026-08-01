# v045 Balanced RAGQA + Discussion Hub

## Purpose

This package intentionally separates two uses.

### Quantitative RAGTruth QA benchmark

- Condition A: `nano_raw_direct`
- Condition B: `nano_balanced_dual_graph`
- Both use `gpt-5.4-nano` with low reasoning effort.
- The benchmark does **not** run Source/Response six-agent validation.
- The dual-graph condition uses the v043 task-complete Source Compiler and v043 Balanced Response Compiler with local node repair.

Run:

```bat
cd /d C:\vrg\v045
run_ragqa.bat
```

`--limit 0` evaluates all eligible RAGTruth test-split, good-quality QA cases under the configured response/evidence length safeguards. Component outputs are cached, so rerunning resumes rather than starting over.

### Main Discussion Hub

The Discussion Hub uses the same balanced node philosophy, adapted to the richer Discussion graph schema:

1. Balanced Complete-Proposition Compiler
2. Local repair of only obvious fragment, unresolved-quote, duplicate-id, or likely multi-claim nodes
3. Evidence, Logic, and Target Agents in parallel
4. Assumption Agent only when a concrete missing-premise proposal exists
5. Judge Agent only for material conflicts, uncertainty, or graph-revision proposals
6. Long-document chunking, graph merging, deduplication, and document-level caps remain enabled

Run:

```bat
cd /d C:\vrg\v045
run_hub.bat
```

Open `http://127.0.0.1:8765/discussion-lab`.

## Compiler contract

A node is neither a whole sentence by default nor the shortest token fragment. It is the smallest semantically complete proposition or clause that can receive one verdict. The compiler preserves subject/predicate meaning, negation, modality, numbers, units, population, time, conditions, comparison direction, and causal strength. It does not emit isolated entities, numbers, modifiers, headings, or discourse markers as nodes.

## Transparency fields

Discussion results include:

- `discussion_compiler`
- `discussion_agent_pipeline`
- `architecture_traces`
- `compiler_refinement`

These show whether local compiler repair occurred and which downstream agents ran.

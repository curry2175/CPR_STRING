# v050 Threshold Optimizer Guide

## Your current situation

You can stop the RAGTruth run at case 330. The completed component outputs remain in the generation cache and the partial `cases.jsonl` remains usable.

Do not delete:

- `outputs/`
- `.env`
- `data/`
- `.venv/`

## Step 1 · Apply this update to the existing project folder

Copy the v050 update files over the current v048/v049 folder. Keep the existing `outputs` directory.

## Step 2 · Search for maximum observed F1

Run:

```bat
run_threshold_optimizer.bat
```

This performs no API calls. It automatically selects the newest partial `cases.jsonl` and searches:

1. One global threshold from 0.00 to 1.00 in 0.01 increments.
2. Relation-specific thresholds.
3. Core span versus complete-claim projection.
4. Explicit-label-only versus inferred error labels.

The console displays:

- Raw Direct F1
- Best single global-threshold F1
- Best relation-specific F1
- Delta versus Raw
- Selected thresholds
- Clean false-positive rate
- Hallucination sensitivity

The HTML report opens automatically.

## Step 3 · Resume all 875 cases with the selected gate

```bat
run_ragqa_resume_optimized.bat
```

The runner reads:

```text
outputs\alignment_threshold_optimizer\latest_best_gate.json
```

For previously completed cases, no new API call is needed when the component cache is present. The optimized gate simply changes which stored Alignment findings become submitted spans.

For new cases after the stopping point, normal component calls continue and the same optimized gate is applied.

## Output files

```text
outputs\alignment_threshold_optimizer\
├─ latest_best_gate.json
├─ latest_summary.json
├─ latest_report.html
└─ threshold_opt_<timestamp>\
   ├─ best_gate.json
   ├─ summary.json
   ├─ report.html
   ├─ global_threshold_curve.csv
   ├─ top_configs.csv
   └─ best_reprojected_cases.jsonl
```

## Global versus relation-specific threshold

The report shows both:

- **Best ONE global threshold:** simplest single-number solution.
- **Best relation-specific gate:** usually higher F1 because different error relations receive different cutoffs.

A threshold above 1.00 means the optimizer disabled that relation because submitting its spans reduced F1 on the fitted cases.

## What threshold optimization can and cannot change

It can recover cases where Alignment already produced a problematic relation and localized `problem_text`, but the previous gate suppressed it.

It cannot recover a hallucination that Alignment classified as `supported_by`, `safe_inference`, `not_factual`, `generic_advice`, or `uncertain`, nor a factual clause omitted by the Response Compiler.

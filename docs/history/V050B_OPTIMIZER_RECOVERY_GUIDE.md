# v050b optimizer recovery

## Apply

Extract the hotfix into the current `v048` project folder and overwrite `optimize_alignment_thresholds.py`.
Do not delete `outputs`, `data`, `.env`, or `.venv`.

## Run

```bat
run_threshold_optimizer.bat
```

The optimizer now tries, in order:

1. `cases.jsonl` in the current project.
2. `cases.jsonl` in sibling version folders.
3. Reconstruction from the largest current-project `generation_cache*.json`.
4. Reconstruction from a sibling project's cache if the current project has none.

Recovery is QA-only, matching `run_ragqa.bat`, and makes no API calls.

Expected console lines when cache recovery is used:

```text
[recovery] No cases.jsonl found...
[recovery] Reconstructed N complete QA cases with 0 API calls.
Loaded N completed rows...
```

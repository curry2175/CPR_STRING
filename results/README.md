# Result Artifacts

- `ragtruth_qa_330_console.log`: original console output through case 330
- `ragtruth_qa_330_case_metrics.csv`: parsed case-level metrics
- `response_level_confusion_matrices.csv`: Raw, original DualGraph, optimized DualGraph
- `representative_cases.csv`: selected clean correction, localization uplift, and regression examples
- `summary_330.json`: machine-readable result summary
- `threshold_optimization_console.txt`: optimizer output

The full original `cases.jsonl` is not reconstructable from console text alone because it also contains source/response payloads and Alignment objects. Run `scripts/IMPORT_LOCAL_RUN_WINDOWS.bat` on the original Windows machine to copy the authentic file into this directory.

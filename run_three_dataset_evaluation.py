from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from vrg.evaluation_suite import BINARY_DATASETS, run_three_dataset_evaluation


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run strict binary VRG evaluation on ProofWriter, LegalBench Yes/No tasks, and PubMedQA Yes/No cases."
    )
    parser.add_argument("--datasets", nargs="+", choices=BINARY_DATASETS, default=list(BINARY_DATASETS))
    parser.add_argument("--limit-per-dataset", type=int, default=20, help="0 means all eligible cases")
    parser.add_argument("--legal-tasks", nargs="*", default=[], help="Optional LegalBench task names; e.g. hearsay")
    parser.add_argument("--model", default="gpt-5.6")
    parser.add_argument("--reasoning-effort", choices=["low", "medium", "high"], default="low")
    parser.add_argument("--max-output-tokens", type=int, default=3500)
    parser.add_argument("--repair-iterations", type=int, default=0, choices=[0, 1, 2, 3])
    parser.add_argument("--refresh-datasets", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    result = run_three_dataset_evaluation(
        output_root=root / "outputs" / "evaluation_suite",
        data_root=root / "data" / "downloaded" / "evaluation_suite",
        datasets=args.datasets,
        limit_per_dataset=args.limit_per_dataset,
        legal_tasks=args.legal_tasks,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        max_output_tokens=args.max_output_tokens,
        max_repair_iterations=args.repair_iterations,
        refresh_datasets=args.refresh_datasets,
    )
    print(json.dumps({"run_id": result["run_id"], "summary": result["summary"]}, ensure_ascii=False, indent=2))
    print(f"\nReport: {root / 'outputs' / 'evaluation_suite' / result['run_id'] / 'report.html'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from vrg.ragtruth_catch_review import run_selected_catch_review
from vrg.ragtruth_dual_graph import _catch_candidate_payload, run_ragtruth_raw_vs_dual_graph
from vrg.ragtruth_localization import download_ragtruth_dataset


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the v043 Balanced Source/Response graph and v046 alignment for one RAGTruth case id, "
            "then review that case with the full six-agent graph pipeline. No completed full benchmark is required."
        )
    )
    parser.add_argument("case_id", nargs="?", default="")
    parser.add_argument("--dataset-dir", default="data/ragtruth")
    parser.add_argument("--reasoning-effort", choices=["low", "medium", "high"], default="low")
    parser.add_argument("--max-context-chars", type=int, default=60000)
    parser.add_argument("--max-response-chars", type=int, default=20000)
    parser.add_argument("--generation-cache", default="outputs/ragtruth_raw_vs_dual_graph_nano/generation_cache_v040.json")
    parser.add_argument("--review-cache", default="outputs/ragtruth_case_6agent_reviews/cache_v048.json")
    parser.add_argument("--no-open", action="store_true", help="Do not open the interactive HTML report automatically")
    args = parser.parse_args()

    case_id = str(args.case_id or "").strip()
    if not case_id:
        case_id = input("RAGTruth case id (example: 12046): ").strip()
    if not case_id:
        print("[ERROR] A case id is required.")
        return 2

    root = Path(__file__).resolve().parent
    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.is_absolute():
        dataset_dir = root / dataset_dir
    response_path = dataset_dir / "response.jsonl"
    source_path = dataset_dir / "source_info.jsonl"
    if not response_path.exists() or not source_path.exists():
        print("RAGTruth files are missing; downloading them now...", flush=True)
        download_ragtruth_dataset(dataset_dir, progress=lambda message: print(message, flush=True))

    generation_cache = Path(args.generation_cache)
    if not generation_cache.is_absolute():
        generation_cache = root / generation_cache

    print(f"\n[1/2] Building Balanced graphs and v046 alignment for case {case_id}...", flush=True)
    benchmark = run_ragtruth_raw_vs_dual_graph(
        response_path=response_path,
        source_path=source_path,
        output_root=root / "outputs" / "ragtruth_direct_case_balanced",
        model="gpt-5.4-nano",
        split="",
        quality="",
        task_types=["QA"],
        limit=0,
        reasoning_effort=args.reasoning_effort,
        max_context_chars=args.max_context_chars,
        max_response_chars=args.max_response_chars,
        include_implicit_true=True,
        include_case_ids={case_id},
        require_full_evidence=False,
        generation_cache_path=generation_cache,
        parallel_components=True,
        print_case_comparison=True,
        progress=lambda message: print(message, flush=True),
    )
    cases = benchmark.get("cases") or []
    if not cases:
        print(
            f"[ERROR] Case id {case_id} was not found as an eligible QA record. "
            "Check the id or inspect whether it belongs to a different task type.",
            flush=True,
        )
        return 3

    row = cases[0]
    if str(row.get("case_id")) != case_id:
        print(f"[ERROR] Loaded unexpected case: {row.get('case_id')}", flush=True)
        return 4
    graph_method = (row.get("methods") or {}).get("nano_dual_graph") or {}
    if graph_method.get("status") != "ok":
        print(json.dumps(graph_method.get("error") or {"error": "Balanced graph unavailable"}, ensure_ascii=False, indent=2))
        return 5

    benchmark_run_dir = root / "outputs" / "ragtruth_direct_case_balanced" / benchmark["run_id"]
    candidate = _catch_candidate_payload(row, "direct_case_id_review")
    cases_dir = benchmark_run_dir / "catch_candidates" / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)
    (cases_dir / f"{case_id}.json").write_text(json.dumps(candidate, ensure_ascii=False, indent=2), encoding="utf-8")
    (benchmark_run_dir / "catch_candidates" / "index.json").write_text(
        json.dumps({
            "run_id": benchmark["run_id"],
            "candidate_count": 1,
            "direct_case_mode": True,
            "candidates": [{
                "case_id": case_id,
                "reason": "direct_case_id_review",
                "file": f"cases/{case_id}.json",
            }],
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    review_cache = Path(args.review_cache)
    if not review_cache.is_absolute():
        review_cache = root / review_cache
    print(f"\n[2/2] Running Source/Response six-agent graph review for case {case_id}...", flush=True)
    review = run_selected_catch_review(
        benchmark_run_dir=benchmark_run_dir,
        case_id=case_id,
        output_root=root / "outputs" / "ragtruth_case_6agent_reviews",
        cache_path=review_cache,
        reasoning_effort=args.reasoning_effort,
        max_output_tokens=2600,
        progress=lambda message: print(message, flush=True),
    )
    review_dir = root / "outputs" / "ragtruth_case_6agent_reviews" / review["run_id"]
    summary = {
        "case_id": case_id,
        "balanced_run_id": benchmark["run_id"],
        "six_agent_run_id": review["run_id"],
        "balanced_api_calls_this_run": benchmark.get("summary", {}).get("actual_api_calls_this_run"),
        "six_agent_api_calls_this_run": review.get("api_calls_this_run"),
        "report": str(review_dir / "report.html"),
        "json": str(review_dir / "result.json"),
    }
    print("\nCompleted. Open the interactive HTML graph report below:\n", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if not args.no_open:
        try:
            import webbrowser
            webbrowser.open((review_dir / "report.html").resolve().as_uri())
        except Exception as exc:
            print(f"[WARN] Could not open the report automatically: {exc}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

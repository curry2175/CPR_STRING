from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from vrg.ragtruth_dual_graph import ALIGNMENT_GATE_PROFILES, run_ragtruth_raw_vs_dual_graph
from vrg.ragtruth_localization import download_ragtruth_dataset


def _load_case_ids(root: Path, values: list[str], files: list[str]) -> set[str]:
    case_ids = {str(value) for value in values}
    for raw_path in files:
        path = Path(raw_path)
        if not path.is_absolute():
            path = root / path
        if not path.exists():
            raise FileNotFoundError(f"Excluded-case file not found: {path}")
        text = path.read_text(encoding="utf-8")
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = [line.strip() for line in text.splitlines() if line.strip()]
        if isinstance(payload, dict):
            payload = payload.get("case_ids") or payload.get("excluded_case_ids") or []
        if not isinstance(payload, list):
            raise ValueError(f"Excluded-case file must contain a JSON list or newline-separated ids: {path}")
        case_ids.update(str(value) for value in payload)
    return case_ids


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare gpt-5.4-nano raw Source/Response hallucination localization against a nano dual-graph pipeline "
            "with the v043 balanced complete-proposition compiler and configurable cached-v046 or v049 recall-balanced alignment/projection on RAGTruth. Catch candidates are saved without running six-agent review during the benchmark."
        )
    )
    parser.add_argument("--dataset-dir", default="data/ragtruth")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--model", default="gpt-5.4-nano", choices=["gpt-5.4-nano"])
    parser.add_argument("--limit", type=int, default=24, help="0 means all eligible cases")
    parser.add_argument("--skip-first", type=int, default=0, help="After deterministic sampling, skip this many leading cases; useful for a locked held-out suffix.")
    parser.add_argument("--task-types", nargs="+", default=["QA"])
    parser.add_argument("--split", default="test")
    parser.add_argument("--quality", default="good")
    parser.add_argument("--reasoning-effort", choices=["low", "medium", "high"], default="low")
    parser.add_argument("--max-context-chars", type=int, default=60000)
    parser.add_argument("--max-response-chars", type=int, default=3000)
    parser.add_argument("--exclude-implicit-true", action="store_true")
    parser.add_argument("--require-full-evidence", action="store_true")
    parser.add_argument("--exclude-case-id", action="append", default=[])
    parser.add_argument("--exclude-case-ids-file", action="append", default=[])
    parser.add_argument("--seed", type=int, default=2040)
    parser.add_argument(
        "--generation-cache",
        default="outputs/ragtruth_raw_vs_dual_graph_nano/generation_cache_v040.json",
    )
    parser.add_argument(
        "--force-component",
        action="append",
        default=[],
        choices=["raw_direct", "source_graph", "response_graph", "alignment"],
    )
    parser.add_argument(
        "--alignment-prompt-profile",
        choices=["auto", "v046_cached", "v049_recall"],
        default="v049_recall",
        help="auto reads the recommended prompt profile from --alignment-gate-config; v046_cached reuses existing v046 Alignment cache; v049_recall uses the v049 recall prompt.",
    )
    parser.add_argument(
        "--alignment-gate-profile",
        choices=["v046_conservative", "v049_balanced_recall", "v049_clause_recall"],
        default="v049_balanced_recall",
        help="Built-in post-alignment span submission policy. Ignored when --alignment-gate-config is supplied.",
    )
    parser.add_argument(
        "--alignment-gate-config",
        default="",
        help="JSON created by optimize_alignment_thresholds.py. Its gate_config is registered and used without changing Alignment API outputs.",
    )
    parser.add_argument("--no-parallel-components", action="store_true")
    parser.add_argument(
        "--no-case-comparison",
        action="store_true",
        help="Do not print the per-case Raw vs Dual-Graph score block.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.is_absolute():
        dataset_dir = root / dataset_dir

    def progress(message: str) -> None:
        print(message, flush=True)

    if args.download or args.download_only:
        status = download_ragtruth_dataset(dataset_dir, progress=progress)
        print(json.dumps({"download": status}, ensure_ascii=False, indent=2), flush=True)
    if args.download_only:
        return 0

    cache_path = Path(args.generation_cache)
    if not cache_path.is_absolute():
        cache_path = root / cache_path
    exclude_case_ids = _load_case_ids(root, args.exclude_case_id, args.exclude_case_ids_file)

    alignment_gate_profile = args.alignment_gate_profile
    alignment_prompt_profile = args.alignment_prompt_profile
    if args.alignment_gate_config:
        gate_path = Path(args.alignment_gate_config)
        if not gate_path.is_absolute():
            gate_path = root / gate_path
        payload = json.loads(gate_path.read_text(encoding="utf-8"))
        gate_config = payload.get("gate_config") if isinstance(payload, dict) else None
        if not isinstance(gate_config, dict):
            raise ValueError(f"Alignment gate config must contain an object named gate_config: {gate_path}")
        thresholds = gate_config.get("thresholds") or {}
        required_relations = {
            "contradicted_by", "partially_supported_by", "qualified_by", "not_found_in_source", "requires_assumption"
        }
        if not required_relations.issubset(thresholds):
            missing = sorted(required_relations.difference(thresholds))
            raise ValueError(f"Alignment gate config is missing thresholds: {missing}")
        custom_name = "v050_optimized_from_file"
        ALIGNMENT_GATE_PROFILES[custom_name] = {
            "thresholds": {key: float(value) for key, value in thresholds.items()},
            "infer_error_label": bool(gate_config.get("infer_error_label", True)),
            "partial_span_mode": "claim" if gate_config.get("partial_span_mode") == "claim" else "core",
        }
        alignment_gate_profile = custom_name
        if alignment_prompt_profile == "auto":
            alignment_prompt_profile = str(payload.get("recommended_alignment_prompt_profile") or "v046_cached")
        print(f"Loaded optimized Alignment gate: {gate_path}", flush=True)
        print(json.dumps(ALIGNMENT_GATE_PROFILES[custom_name], ensure_ascii=False, indent=2), flush=True)
    elif alignment_prompt_profile == "auto":
        alignment_prompt_profile = "v046_cached"

    result = run_ragtruth_raw_vs_dual_graph(
        response_path=dataset_dir / "response.jsonl",
        source_path=dataset_dir / "source_info.jsonl",
        output_root=root / "outputs" / "ragtruth_raw_vs_dual_graph_nano",
        model=args.model,
        split=args.split,
        quality=args.quality,
        task_types=args.task_types,
        limit=args.limit,
        skip_first=args.skip_first,
        seed=args.seed,
        reasoning_effort=args.reasoning_effort,
        max_context_chars=args.max_context_chars,
        max_response_chars=args.max_response_chars,
        include_implicit_true=not args.exclude_implicit_true,
        exclude_case_ids=exclude_case_ids,
        require_full_evidence=args.require_full_evidence,
        generation_cache_path=cache_path,
        force_components=set(args.force_component),
        alignment_prompt_profile=alignment_prompt_profile,
        alignment_gate_profile=alignment_gate_profile,
        parallel_components=not args.no_parallel_components,
        print_case_comparison=not args.no_case_comparison,
        progress=progress,
    )
    print(json.dumps({"run_id": result["run_id"], "summary": result["summary"]}, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

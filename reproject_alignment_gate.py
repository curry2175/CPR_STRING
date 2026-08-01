from __future__ import annotations

import argparse
import copy
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from vrg.ragtruth_dual_graph import (
    ALIGNMENT_GATE_PROFILES,
    GRAPH_METHOD,
    RAW_METHOD,
    DualGraphAlignmentOutput,
    ResponseClaimGraphOutput,
    _method_summary,
    _paired_comparison,
    _predictions_from_alignment,
)
from vrg.ragtruth_localization import score_predictions


def _latest_cases_file(root: Path) -> Path:
    candidates = sorted(
        root.glob("outputs/ragtruth_raw_vs_dual_graph_nano/*/cases.jsonl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            "No prior cases.jsonl found. Run run_ragqa.bat at least once, or pass --cases-jsonl explicitly."
        )
    return candidates[0]


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            rows.append(row)
    if not rows:
        raise ValueError(f"No completed cases found in {path}")
    return rows


def _record_by_component(graph_method: dict[str, Any], component: str) -> dict[str, Any] | None:
    for record in graph_method.get("generation_records") or []:
        if record.get("component") == component:
            return record
    return None


def _reproject_row(row: dict[str, Any], profile: str) -> dict[str, Any] | None:
    graph_method = copy.deepcopy((row.get("methods") or {}).get(GRAPH_METHOD) or {})
    if graph_method.get("status") != "ok":
        return None
    response_record = _record_by_component(graph_method, "response_graph")
    alignment_record = _record_by_component(graph_method, "alignment")
    if not response_record or not alignment_record:
        return None
    if response_record.get("status") != "ok" or alignment_record.get("status") != "ok":
        return None

    response_graph = ResponseClaimGraphOutput.model_validate(response_record["parsed"])
    alignment = DualGraphAlignmentOutput.model_validate(alignment_record["parsed"])
    predictions, details = _predictions_from_alignment(
        alignment,
        response_graph,
        str(row.get("response") or ""),
        gate_profile=profile,
    )
    original_details = graph_method.get("details") or {}
    details["source_graph"] = original_details.get("source_graph")
    details["response_compiler_refinement"] = original_details.get("response_compiler_refinement") or {}
    graph_method["predicted_spans"] = predictions
    graph_method["details"] = details
    graph_method["scores"] = score_predictions(
        str(row.get("response") or ""),
        predictions,
        row.get("gold_labels") or [],
    )
    graph_method["offline_reprojection"] = True
    graph_method["alignment_gate_profile"] = profile
    updated = copy.deepcopy(row)
    updated.setdefault("methods", {})[GRAPH_METHOD] = graph_method
    return updated


def _rescue_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {
        "eligible_rows": 0,
        "clean_raw_fp_graph_clean": 0,
        "clean_graph_fp_raw_clean": 0,
        "strict_hallucination_rescue": 0,
        "near_strict_hallucination_rescue": 0,
        "localization_rescue": 0,
        "graph_regression_on_hallucination": 0,
    }
    case_ids = {key: [] for key in counts if key != "eligible_rows"}
    for row in rows:
        raw = ((row.get("methods") or {}).get(RAW_METHOD) or {}).get("scores")
        graph = ((row.get("methods") or {}).get(GRAPH_METHOD) or {}).get("scores")
        if not raw or not graph:
            continue
        counts["eligible_rows"] += 1
        cid = str(row.get("case_id"))
        gold = bool(graph.get("gold_has_hallucination"))
        raw_pred = bool(raw.get("predicted_has_hallucination"))
        graph_pred = bool(graph.get("predicted_has_hallucination"))
        raw_r = float(raw.get("char_recall") or 0.0)
        graph_r = float(graph.get("char_recall") or 0.0)
        raw_f1 = float(raw.get("char_f1") or 0.0)
        graph_f1 = float(graph.get("char_f1") or 0.0)
        if not gold and raw_pred and not graph_pred:
            counts["clean_raw_fp_graph_clean"] += 1
            case_ids["clean_raw_fp_graph_clean"].append(cid)
        if not gold and not raw_pred and graph_pred:
            counts["clean_graph_fp_raw_clean"] += 1
            case_ids["clean_graph_fp_raw_clean"].append(cid)
        if gold and raw_r == 0.0 and graph_r >= 0.50 and graph_f1 >= 0.50:
            counts["strict_hallucination_rescue"] += 1
            case_ids["strict_hallucination_rescue"].append(cid)
        if gold and raw_r <= 0.10 and graph_r >= 0.50 and graph_f1 >= raw_f1 + 0.20:
            counts["near_strict_hallucination_rescue"] += 1
            case_ids["near_strict_hallucination_rescue"].append(cid)
        if gold and raw_r < 0.40 and graph_r >= 0.65 and graph_f1 >= raw_f1 + 0.15:
            counts["localization_rescue"] += 1
            case_ids["localization_rescue"].append(cid)
        if gold and raw_f1 >= graph_f1 + 0.20:
            counts["graph_regression_on_hallucination"] += 1
            case_ids["graph_regression_on_hallucination"].append(cid)
    return {"counts": counts, "case_ids": case_ids}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Re-score completed RAGTruth cases under multiple Alignment gate profiles without any API calls."
    )
    parser.add_argument("--cases-jsonl", default="")
    parser.add_argument(
        "--profiles",
        nargs="+",
        default=["v046_conservative", "v049_balanced_recall", "v049_clause_recall"],
        choices=sorted(ALIGNMENT_GATE_PROFILES),
    )
    parser.add_argument("--seed", type=int, default=2040)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    cases_path = Path(args.cases_jsonl) if args.cases_jsonl else _latest_cases_file(root)
    if not cases_path.is_absolute():
        cases_path = root / cases_path
    rows = _load_rows(cases_path)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = root / "outputs" / "alignment_gate_sweep" / f"gate_sweep_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {
        "source_cases_jsonl": str(cases_path),
        "completed_rows_loaded": len(rows),
        "api_calls": 0,
        "profiles": {},
    }
    csv_rows: list[dict[str, Any]] = []
    best_profile = None
    best_f1 = -1.0

    for profile in args.profiles:
        projected_rows: list[dict[str, Any]] = []
        skipped = 0
        for row in rows:
            updated = _reproject_row(row, profile)
            if updated is None:
                skipped += 1
                continue
            projected_rows.append(updated)
        raw_summary = _method_summary(projected_rows, RAW_METHOD)
        graph_summary = _method_summary(projected_rows, GRAPH_METHOD)
        paired = _paired_comparison(projected_rows, GRAPH_METHOD, seed=args.seed, reference_method=RAW_METHOD)
        rescue = _rescue_summary(projected_rows)
        profile_result = {
            "gate_config": ALIGNMENT_GATE_PROFILES[profile],
            "reprojected_rows": len(projected_rows),
            "skipped_rows": skipped,
            "raw": raw_summary,
            "dual_graph": graph_summary,
            "paired_dual_graph_vs_raw": paired,
            "rescue_analysis": rescue,
        }
        results["profiles"][profile] = profile_result
        graph_f1 = float(graph_summary.get("char_f1_percent") or 0.0)
        if graph_f1 > best_f1:
            best_f1 = graph_f1
            best_profile = profile
        csv_rows.append({
            "profile": profile,
            "n": len(projected_rows),
            "raw_char_f1_percent": raw_summary.get("char_f1_percent"),
            "dual_char_f1_percent": graph_summary.get("char_f1_percent"),
            "dual_clean_false_positive_rate_percent": graph_summary.get("clean_false_positive_rate_percent"),
            "strict_rescue": rescue["counts"]["strict_hallucination_rescue"],
            "near_strict_rescue": rescue["counts"]["near_strict_hallucination_rescue"],
            "localization_rescue": rescue["counts"]["localization_rescue"],
            "hallucination_regression": rescue["counts"]["graph_regression_on_hallucination"],
        })

    results["best_profile_by_current_char_f1"] = best_profile
    results["warning"] = (
        "This is a development-set comparison on already observed cases. Use it to choose a gate, then lock the gate before final held-out evaluation."
    )
    (output_dir / "result.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output_dir / "summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)
    (output_dir / "recommended_profile.txt").write_text(str(best_profile or ""), encoding="utf-8")

    print(f"Loaded {len(rows)} completed cases from: {cases_path}")
    print("No API calls were made.\n")
    for row in csv_rows:
        print(
            f"{row['profile']}: Raw F1={row['raw_char_f1_percent']}% | "
            f"Dual F1={row['dual_char_f1_percent']}% | "
            f"Clean FP={row['dual_clean_false_positive_rate_percent']}% | "
            f"Strict rescue={row['strict_rescue']} | Regression={row['hallucination_regression']}"
        )
    print(f"\nBest current profile by character F1: {best_profile}")
    print(f"Saved: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from vrg.ragtruth_dual_graph import (
    PROMPT_VERSIONS,
    SourceEvidenceGraphOutput,
    _component_key,
    ALIGNMENT_GATE_PROFILES,
    ALIGNMENT_PROMPT_VERSIONS,
    GRAPH_METHOD,
    RAW_METHOD,
    DualGraphAlignmentOutput,
    ResponseClaimGraphOutput,
    _locate_problem_text,
    _node_has_complete_proposition,
    _predictions_from_alignment,
    _problem_text_action,
    _resolve_response_graph,
)
from vrg.ragtruth_localization import (
    DirectSpanOutput,
    _predictions_from_direct,
    build_evidence_card,
    load_ragtruth_cases,
    score_predictions,
)

ERROR_RELATIONS = (
    "contradicted_by",
    "partially_supported_by",
    "qualified_by",
    "not_found_in_source",
    "requires_assumption",
)
PARTIAL_RELATIONS = {"partially_supported_by", "qualified_by", "requires_assumption"}
NON_HALLUCINATION_RELATIONS = {
    "supported_by",
    "safe_inference",
    "not_factual",
    "generic_advice",
    "uncertain",
}


@dataclass(frozen=True)
class CandidateSpan:
    relation: str
    confidence: float
    has_explicit_label: bool
    has_localized_problem: bool
    start: int
    end: int
    label_type: str


@dataclass
class PreparedCase:
    row: dict[str, Any]
    response: str
    gold_intervals: list[tuple[int, int]]
    candidates_by_mode: dict[str, list[CandidateSpan]]
    raw_scores: dict[str, Any]


def _count_jsonl_rows(path: Path) -> int:
    try:
        count = 0
        with path.open("r", encoding="utf-8") as handle:
            for raw in handle:
                line = raw.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    count += 1
        return count
    except OSError:
        return 0


def _dedupe_paths(paths: Iterable[Path]) -> list[Path]:
    found: dict[str, Path] = {}
    for path in paths:
        try:
            key = str(path.resolve()).lower()
        except OSError:
            key = str(path).lower()
        found[key] = path
    return list(found.values())


def _local_cases_candidates(root: Path) -> list[Path]:
    return _dedupe_paths([
        *root.glob("outputs/ragtruth_raw_vs_dual_graph_nano/*/cases.jsonl"),
        *root.glob("outputs/**/cases.jsonl"),
    ])


def _sibling_cases_candidates(root: Path) -> list[Path]:
    parent = root.parent
    if not parent.exists():
        return []
    return _dedupe_paths([
        *parent.glob("*/outputs/ragtruth_raw_vs_dual_graph_nano/*/cases.jsonl"),
        *parent.glob("*/outputs/**/cases.jsonl"),
    ])


def _latest_cases_file(root: Path) -> Path | None:
    local = _local_cases_candidates(root)
    sibling = _sibling_cases_candidates(root)
    candidates = _dedupe_paths([*local, *sibling])
    if not candidates:
        return None
    local_keys = {str(path.resolve()).lower() for path in local}
    ranked = sorted(
        candidates,
        key=lambda p: (_count_jsonl_rows(p), p.stat().st_mtime),
        reverse=True,
    )
    selected = ranked[0]
    print(f"[input] Found {len(ranked)} cases.jsonl candidate(s) in the current/sibling projects.")
    for path in ranked[:8]:
        marker = "current" if str(path.resolve()).lower() in local_keys else "sibling"
        print(f"        rows={_count_jsonl_rows(path):4d}  [{marker}] {path}")
    print(f"[input] Selected the largest completed cases file: {selected}")
    return selected


def _local_cache_candidates(root: Path) -> list[Path]:
    return _dedupe_paths(root.glob("outputs/ragtruth_raw_vs_dual_graph_nano/generation_cache*.json"))


def _sibling_cache_candidates(root: Path) -> list[Path]:
    parent = root.parent
    if not parent.exists():
        return []
    return _dedupe_paths(parent.glob("*/outputs/ragtruth_raw_vs_dual_graph_nano/generation_cache*.json"))


def _cache_score(path: Path) -> tuple[int, int, int, float]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return (0, 0, 0, path.stat().st_mtime)
        raw_n = len(payload.get("raw_direct") or {})
        response_n = len(payload.get("response_graph") or {})
        alignment_n = len(payload.get("alignment") or {})
        return (alignment_n, min(raw_n, response_n, alignment_n), raw_n, path.stat().st_mtime)
    except Exception:
        return (0, 0, 0, path.stat().st_mtime if path.exists() else 0.0)


def _find_dataset_dir(root: Path, cache_path: Path) -> Path:
    candidates = [
        root / "data" / "ragtruth",
        cache_path.parents[2] / "data" / "ragtruth" if len(cache_path.parents) >= 3 else root / "data" / "ragtruth",
    ]
    parent = root.parent
    if parent.exists():
        candidates.extend(path / "data" / "ragtruth" for path in parent.iterdir() if path.is_dir())
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate).lower()
        if key in seen:
            continue
        seen.add(key)
        if (candidate / "response.jsonl").exists() and (candidate / "source_info.jsonl").exists():
            return candidate
    raise FileNotFoundError(
        "RAGTruth dataset files were not found. Expected data/ragtruth/response.jsonl and source_info.jsonl "
        "in the current project or a sibling project folder."
    )


def _recover_cases_from_cache(root: Path, explicit_cache: Path | None = None) -> Path:
    local = _local_cache_candidates(root)
    sibling = _sibling_cache_candidates(root)
    if explicit_cache is not None:
        caches = [explicit_cache]
    else:
        caches = _dedupe_paths([*local, *sibling])
    caches = sorted(caches, key=_cache_score, reverse=True)
    if not caches:
        raise FileNotFoundError(
            "No cases.jsonl and no generation_cache*.json were found in this project or sibling project folders."
        )
    cache_path = caches[0]
    score = _cache_score(cache_path)
    local_keys = {str(path.resolve()).lower() for path in local}
    print(f"[recovery] No cases.jsonl found. Cache candidates:")
    for path in caches[:8]:
        candidate_score = _cache_score(path)
        marker = "current" if str(path.resolve()).lower() in local_keys else "sibling"
        print(f"           alignment={candidate_score[0]:4d} matched={candidate_score[1]:4d} [{marker}] {path}")
    print(f"[recovery] Recovering from the largest completed cache: {cache_path}")
    print(f"[recovery] Cache counts: alignment={score[0]}, matched-min={score[1]}, raw={score[2]}")
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    dataset_dir = _find_dataset_dir(root, cache_path)
    cases, _sampling = load_ragtruth_cases(
        dataset_dir / "response.jsonl",
        dataset_dir / "source_info.jsonl",
        split="test",
        quality="good",
        task_types=["QA"],
        limit=0,
        seed=2040,
        max_response_chars=3000,
        include_implicit_true=True,
        require_full_evidence=True,
        max_context_chars=60000,
    )
    recovered: list[dict[str, Any]] = []
    for index, case in enumerate(cases, 1):
        evidence_card = build_evidence_card(
            case["source_info"], case["response"], max_context_chars=60000, force_full=True
        )
        raw_key = _component_key("raw_direct", {
            "case_id": case["case_id"],
            "response": case["response"],
            "task_instruction": case["task_instruction"],
            "evidence_units": evidence_card["units"],
            "model": "gpt-5.4-nano",
            "reasoning_effort": "low",
            "max_output_tokens": 1800,
        })
        source_key = _component_key("source_graph", {
            "source_id": case["source_id"],
            "task_instruction": case["task_instruction"],
            "evidence_units": evidence_card["units"],
            "model": "gpt-5.4-nano",
            "reasoning_effort": "low",
            "max_output_tokens": 3200,
        })
        response_key = _component_key("response_graph", {
            "case_id": case["case_id"],
            "response": case["response"],
            "task_instruction": case["task_instruction"],
            "model": "gpt-5.4-nano",
            "reasoning_effort": "low",
            "max_output_tokens": 3600,
        })
        raw_record = (cache.get("raw_direct") or {}).get(raw_key)
        source_record = (cache.get("source_graph") or {}).get(source_key)
        response_record = (cache.get("response_graph") or {}).get(response_key)
        if not raw_record or not source_record or not response_record:
            continue
        if any(record.get("status") != "ok" for record in (raw_record, source_record, response_record)):
            continue
        try:
            source_graph = SourceEvidenceGraphOutput.model_validate(source_record["parsed"])
            response_graph = ResponseClaimGraphOutput.model_validate(response_record["parsed"])
        except Exception:
            continue
        alignment_record = None
        for prompt_version in dict.fromkeys([
            ALIGNMENT_PROMPT_VERSIONS.get("v046_cached", ""),
            ALIGNMENT_PROMPT_VERSIONS.get("v049_recall", ""),
            PROMPT_VERSIONS.get("alignment", ""),
        ]):
            if not prompt_version:
                continue
            alignment_key = _component_key("alignment", {
                "case_id": case["case_id"],
                "response": case["response"],
                "task_instruction": case["task_instruction"],
                "evidence_units": evidence_card["units"],
                "source_graph": source_graph.model_dump(),
                "response_graph": response_graph.model_dump(),
                "model": "gpt-5.4-nano",
                "reasoning_effort": "low",
                "max_output_tokens": 2600,
            }, prompt_version=prompt_version)
            candidate = (cache.get("alignment") or {}).get(alignment_key)
            if candidate and candidate.get("status") == "ok":
                alignment_record = candidate
                break
        if alignment_record is None:
            continue
        try:
            parsed_raw = DirectSpanOutput.model_validate(raw_record["parsed"])
            raw_predictions, raw_details = _predictions_from_direct(parsed_raw, case["response"])
            raw_scores = score_predictions(case["response"], raw_predictions, case["gold_labels"])
        except Exception:
            continue
        row = {
            "case_index": index,
            **{key: value for key, value in case.items() if key != "source_info"},
            "evidence_card": evidence_card,
            "methods": {
                RAW_METHOD: {
                    "method": RAW_METHOD,
                    "model": "gpt-5.4-nano",
                    "status": "ok",
                    "predicted_spans": raw_predictions,
                    "details": raw_details,
                    "generation_records": [raw_record],
                    "scores": raw_scores,
                },
                GRAPH_METHOD: {
                    "method": GRAPH_METHOD,
                    "model": "gpt-5.4-nano",
                    "status": "ok",
                    "predicted_spans": [],
                    "details": {},
                    "generation_records": [source_record, response_record, alignment_record],
                    "scores": None,
                },
            },
            "recovered_from_generation_cache": str(cache_path),
        }
        recovered.append(row)
    if not recovered:
        raise RuntimeError(
            "The cache was found, but no complete QA cases could be reconstructed. "
            "This usually means the cache was produced with different run settings."
        )
    output_root = root / "outputs" / "alignment_threshold_optimizer"
    output_root.mkdir(parents=True, exist_ok=True)
    recovered_path = output_root / "recovered_cases_from_cache.jsonl"
    with recovered_path.open("w", encoding="utf-8") as handle:
        for row in recovered:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[recovery] Reconstructed {len(recovered)} complete QA cases with 0 API calls.")
    print(f"[recovery] Saved: {recovered_path}")
    return recovered_path


def _load_rows_tolerant(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    for index, raw in enumerate(lines, 1):
        line = raw.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            # A running process may leave only the final line half-written. Ignore only that line.
            if index == len(lines):
                print(f"[warning] Ignoring incomplete final JSONL line {index} in {path}")
                continue
            raise
        if isinstance(payload, dict):
            rows.append(payload)
    if not rows:
        raise ValueError(f"No completed cases could be loaded from {path}")
    return rows


def _record_by_component(graph_method: dict[str, Any], component: str) -> dict[str, Any] | None:
    for record in graph_method.get("generation_records") or []:
        if record.get("component") == component:
            return record
    return None


def _merge_intervals(intervals: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    cleaned = sorted((max(0, int(a)), max(0, int(b))) for a, b in intervals if int(b) > int(a))
    merged: list[list[int]] = []
    for start, end in cleaned:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(a, b) for a, b in merged]


def _interval_length(intervals: Iterable[tuple[int, int]]) -> int:
    return sum(end - start for start, end in _merge_intervals(intervals))


def _intersection_length(a: Iterable[tuple[int, int]], b: Iterable[tuple[int, int]]) -> int:
    left = _merge_intervals(a)
    right = _merge_intervals(b)
    i = j = total = 0
    while i < len(left) and j < len(right):
        start = max(left[i][0], right[j][0])
        end = min(left[i][1], right[j][1])
        if end > start:
            total += end - start
        if left[i][1] <= right[j][1]:
            i += 1
        else:
            j += 1
    return total


def _candidate_for_item(
    *,
    item: Any,
    node: dict[str, Any],
    response: str,
    mode: str,
) -> CandidateSpan | None:
    if item.relation in NON_HALLUCINATION_RELATIONS:
        return None
    if item.relation not in ERROR_RELATIONS:
        return None
    if not bool(node.get("evaluation_eligible", True)):
        return None
    action = _problem_text_action(item.problem_text, item.relation)
    if action == "discard":
        return None
    location = _locate_problem_text(response, node, item.problem_text)
    if action == "expand_to_claim" and _node_has_complete_proposition(node):
        if isinstance(node.get("start"), int) and isinstance(node.get("end"), int):
            location = (int(node["start"]), int(node["end"]))
    elif (
        mode == "claim"
        and item.relation in PARTIAL_RELATIONS
        and _node_has_complete_proposition(node)
        and isinstance(node.get("start"), int)
        and isinstance(node.get("end"), int)
    ):
        location = (int(node["start"]), int(node["end"]))
    if location is None and _node_has_complete_proposition(node):
        if isinstance(node.get("start"), int) and isinstance(node.get("end"), int):
            location = (int(node["start"]), int(node["end"]))
    if location is None:
        return None
    start, end = int(location[0]), int(location[1])
    if end <= start:
        return None
    label_type = "contradiction" if item.relation == "contradicted_by" or item.label_type == "contradiction" else "unsupported"
    effective_confidence = float(item.confidence) if "confidence" in item.model_fields_set else 1.0
    return CandidateSpan(
        relation=str(item.relation),
        confidence=effective_confidence,
        has_explicit_label=item.label_type in {"unsupported", "contradiction"},
        has_localized_problem=bool(str(item.problem_text or "").strip()),
        start=start,
        end=end,
        label_type=label_type,
    )


def _prepare_case(row: dict[str, Any]) -> PreparedCase | None:
    methods = row.get("methods") or {}
    graph_method = copy.deepcopy(methods.get(GRAPH_METHOD) or {})
    raw_method = methods.get(RAW_METHOD) or {}
    if graph_method.get("status") != "ok" or not isinstance(raw_method.get("scores"), dict):
        return None
    response_record = _record_by_component(graph_method, "response_graph")
    alignment_record = _record_by_component(graph_method, "alignment")
    if not response_record or not alignment_record:
        return None
    if response_record.get("status") != "ok" or alignment_record.get("status") != "ok":
        return None
    response = str(row.get("response") or "")
    response_graph = ResponseClaimGraphOutput.model_validate(response_record["parsed"])
    alignment = DualGraphAlignmentOutput.model_validate(alignment_record["parsed"])
    nodes, _unresolved = _resolve_response_graph(response_graph, response)
    by_mode: dict[str, list[CandidateSpan]] = {"core": [], "claim": []}
    for mode in by_mode:
        seen: set[tuple[str, float, bool, bool, int, int, str]] = set()
        for item in alignment.alignments:
            node = nodes.get(item.response_node_id)
            if node is None:
                continue
            candidate = _candidate_for_item(item=item, node=node, response=response, mode=mode)
            if candidate is None:
                continue
            key = (
                candidate.relation,
                candidate.confidence,
                candidate.has_explicit_label,
                candidate.has_localized_problem,
                candidate.start,
                candidate.end,
                candidate.label_type,
            )
            if key not in seen:
                seen.add(key)
                by_mode[mode].append(candidate)
    gold = [(int(span["start"]), int(span["end"])) for span in row.get("gold_labels") or []]
    return PreparedCase(
        row=row,
        response=response,
        gold_intervals=_merge_intervals(gold),
        candidates_by_mode=by_mode,
        raw_scores=raw_method["scores"],
    )


def _prepare_cases(rows: list[dict[str, Any]]) -> tuple[list[PreparedCase], int]:
    prepared: list[PreparedCase] = []
    skipped = 0
    for row in rows:
        case = _prepare_case(row)
        if case is None:
            skipped += 1
        else:
            prepared.append(case)
    return prepared, skipped


def _normalize_gate_config(config: dict[str, Any]) -> dict[str, Any]:
    thresholds = config.get("thresholds") or {}
    return {
        "thresholds": {relation: float(thresholds.get(relation, 1.01)) for relation in ERROR_RELATIONS},
        "infer_error_label": bool(config.get("infer_error_label", True)),
        "partial_span_mode": "claim" if config.get("partial_span_mode") == "claim" else "core",
    }


def _evaluate_config(cases: list[PreparedCase], config: dict[str, Any]) -> dict[str, Any]:
    config = _normalize_gate_config(config)
    thresholds = config["thresholds"]
    mode = config["partial_span_mode"]
    infer = config["infer_error_label"]
    tp = fp = fn = 0
    clean_n = clean_fp = 0
    hall_n = hall_detected = 0
    strict_rescue = localization_rescue = graph_regression = 0
    response_correct = 0
    per_case: list[dict[str, Any]] = []
    for case in cases:
        selected: list[tuple[int, int]] = []
        selected_records: list[CandidateSpan] = []
        for candidate in case.candidates_by_mode[mode]:
            label_ok = candidate.has_explicit_label or (infer and candidate.has_localized_problem)
            if not label_ok:
                continue
            if candidate.confidence + 1e-12 < float(thresholds.get(candidate.relation, 1.01)):
                continue
            selected.append((candidate.start, candidate.end))
            selected_records.append(candidate)
        predicted = _merge_intervals(selected)
        gold = case.gold_intervals
        pred_len = _interval_length(predicted)
        gold_len = _interval_length(gold)
        case_tp = _intersection_length(predicted, gold)
        case_fp = pred_len - case_tp
        case_fn = gold_len - case_tp
        tp += case_tp
        fp += case_fp
        fn += case_fn
        gold_positive = bool(gold)
        pred_positive = bool(predicted)
        response_correct += int(gold_positive == pred_positive)
        if gold_positive:
            hall_n += 1
            hall_detected += int(pred_positive)
        else:
            clean_n += 1
            clean_fp += int(pred_positive)
        precision = case_tp / (case_tp + case_fp) if case_tp + case_fp else (1.0 if not gold_positive else 0.0)
        recall = case_tp / (case_tp + case_fn) if case_tp + case_fn else (1.0 if not pred_positive else 0.0)
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        raw = case.raw_scores
        raw_r = float(raw.get("char_recall") or 0.0)
        raw_f1 = float(raw.get("char_f1") or 0.0)
        if gold_positive and raw_r == 0.0 and recall >= 0.50 and f1 >= 0.50:
            strict_rescue += 1
        if gold_positive and raw_r < 0.40 and recall >= 0.65 and f1 >= raw_f1 + 0.15:
            localization_rescue += 1
        if gold_positive and raw_f1 >= f1 + 0.20:
            graph_regression += 1
        per_case.append({
            "case_id": str(case.row.get("case_id")),
            "char_tp": case_tp,
            "char_fp": case_fp,
            "char_fn": case_fn,
            "char_precision": precision,
            "char_recall": recall,
            "char_f1": f1,
            "gold_has_hallucination": gold_positive,
            "predicted_has_hallucination": pred_positive,
            "predicted_intervals": predicted,
            "selected_candidate_count": len(selected_records),
        })
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "char_tp": tp,
        "char_fp": fp,
        "char_fn": fn,
        "char_precision": precision,
        "char_recall": recall,
        "char_f1": f1,
        "char_precision_percent": round(precision * 100, 6),
        "char_recall_percent": round(recall * 100, 6),
        "char_f1_percent": round(f1 * 100, 6),
        "clean_n": clean_n,
        "clean_false_positive_n": clean_fp,
        "clean_false_positive_rate_percent": round((clean_fp / clean_n * 100) if clean_n else 0.0, 6),
        "hallucinated_n": hall_n,
        "hallucinated_detected_n": hall_detected,
        "hallucinated_sensitivity_percent": round((hall_detected / hall_n * 100) if hall_n else 0.0, 6),
        "response_accuracy_percent": round((response_correct / len(cases) * 100) if cases else 0.0, 6),
        "strict_rescue": strict_rescue,
        "localization_rescue": localization_rescue,
        "hallucination_regression": graph_regression,
        "per_case": per_case,
    }


def _raw_summary(cases: list[PreparedCase]) -> dict[str, Any]:
    tp = sum(int(c.raw_scores.get("char_tp") or 0) for c in cases)
    fp = sum(int(c.raw_scores.get("char_fp") or 0) for c in cases)
    fn = sum(int(c.raw_scores.get("char_fn") or 0) for c in cases)
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    clean = [c for c in cases if not c.raw_scores.get("gold_has_hallucination")]
    clean_fp = sum(bool(c.raw_scores.get("predicted_has_hallucination")) for c in clean)
    hall = [c for c in cases if c.raw_scores.get("gold_has_hallucination")]
    hall_detected = sum(bool(c.raw_scores.get("predicted_has_hallucination")) for c in hall)
    return {
        "char_tp": tp,
        "char_fp": fp,
        "char_fn": fn,
        "char_precision_percent": round(precision * 100, 6),
        "char_recall_percent": round(recall * 100, 6),
        "char_f1_percent": round(f1 * 100, 6),
        "clean_false_positive_rate_percent": round((clean_fp / len(clean) * 100) if clean else 0.0, 6),
        "hallucinated_sensitivity_percent": round((hall_detected / len(hall) * 100) if hall else 0.0, 6),
    }


def _config_key(config: dict[str, Any]) -> str:
    cfg = _normalize_gate_config(config)
    values = ",".join(f"{relation}:{cfg['thresholds'][relation]:.2f}" for relation in ERROR_RELATIONS)
    return f"{cfg['partial_span_mode']}|infer={int(cfg['infer_error_label'])}|{values}"


def _rank_tuple(result: dict[str, Any], config: dict[str, Any]) -> tuple[float, float, float, float]:
    # F1 is primary. Ties prefer precision, fewer clean FPs, then simpler/global-looking thresholds.
    thresholds = list(_normalize_gate_config(config)["thresholds"].values())
    spread = max(thresholds) - min(thresholds)
    return (
        float(result["char_f1"]),
        float(result["char_precision"]),
        -float(result["clean_false_positive_rate_percent"]),
        -spread,
    )


def _search(
    cases: list[PreparedCase],
    *,
    step: float,
    random_starts: int,
    max_passes: int,
    seed: int,
    progress: bool = True,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    grid = [round(i * step, 6) for i in range(int(round(1.0 / step)) + 1)] + [1.01]
    cache: dict[str, dict[str, Any]] = {}
    leaderboard: dict[str, dict[str, Any]] = {}
    global_curve: list[dict[str, Any]] = []
    evaluation_count = 0

    def evaluate(config: dict[str, Any], phase: str) -> dict[str, Any]:
        nonlocal evaluation_count
        config = _normalize_gate_config(config)
        key = _config_key(config)
        if key not in cache:
            cache[key] = _evaluate_config(cases, config)
            evaluation_count += 1
        result = cache[key]
        current = leaderboard.get(key)
        entry = {
            "phase": phase,
            "config": config,
            "metrics": {k: v for k, v in result.items() if k != "per_case"},
        }
        if current is None or _rank_tuple(result, config) > _rank_tuple(current["metrics"], current["config"]):
            leaderboard[key] = entry
        return result

    best_global_config: dict[str, Any] | None = None
    best_global_result: dict[str, Any] | None = None
    if progress:
        print("[1/3] Exact global-threshold sweep (core/claim × label inference)...", flush=True)
    for mode in ("core", "claim"):
        for infer in (False, True):
            for threshold in grid:
                config = {
                    "thresholds": {relation: threshold for relation in ERROR_RELATIONS},
                    "infer_error_label": infer,
                    "partial_span_mode": mode,
                }
                result = evaluate(config, "global")
                global_curve.append({
                    "threshold": threshold,
                    "partial_span_mode": mode,
                    "infer_error_label": infer,
                    "char_f1_percent": result["char_f1_percent"],
                    "char_precision_percent": result["char_precision_percent"],
                    "char_recall_percent": result["char_recall_percent"],
                    "clean_false_positive_rate_percent": result["clean_false_positive_rate_percent"],
                    "hallucinated_sensitivity_percent": result["hallucinated_sensitivity_percent"],
                })
                if best_global_result is None or _rank_tuple(result, config) > _rank_tuple(best_global_result, best_global_config or config):
                    best_global_config = config
                    best_global_result = result

    if best_global_config is None or best_global_result is None:
        raise RuntimeError("Global threshold search produced no result")

    if progress:
        print(
            f"  best global: F1={best_global_result['char_f1_percent']:.2f}% "
            f"threshold={next(iter(best_global_config['thresholds'].values())):.2f} "
            f"mode={best_global_config['partial_span_mode']} infer={best_global_config['infer_error_label']}",
            flush=True,
        )
        print("[2/3] Relation-specific coordinate search with multiple starts...", flush=True)

    starts: list[dict[str, Any]] = [copy.deepcopy(best_global_config)]
    for profile in ALIGNMENT_GATE_PROFILES.values():
        starts.append(_normalize_gate_config(profile))
    starts.extend([
        {"thresholds": {relation: 0.0 for relation in ERROR_RELATIONS}, "infer_error_label": True, "partial_span_mode": "core"},
        {"thresholds": {relation: 0.0 for relation in ERROR_RELATIONS}, "infer_error_label": True, "partial_span_mode": "claim"},
        {"thresholds": {relation: 1.01 for relation in ERROR_RELATIONS}, "infer_error_label": True, "partial_span_mode": "core"},
    ])
    rng = random.Random(seed)
    for _ in range(random_starts):
        starts.append({
            "thresholds": {relation: rng.choice(grid) for relation in ERROR_RELATIONS},
            "infer_error_label": rng.choice([False, True]),
            "partial_span_mode": rng.choice(["core", "claim"]),
        })

    best_config = copy.deepcopy(best_global_config)
    best_result = best_global_result
    for start_index, start in enumerate(starts, 1):
        current = _normalize_gate_config(start)
        current_result = evaluate(current, f"start_{start_index}")
        # Optimize span mode and inference choice before thresholds.
        for mode in ("core", "claim"):
            for infer in (False, True):
                trial = copy.deepcopy(current)
                trial["partial_span_mode"] = mode
                trial["infer_error_label"] = infer
                trial_result = evaluate(trial, "mode_infer")
                if _rank_tuple(trial_result, trial) > _rank_tuple(current_result, current):
                    current, current_result = trial, trial_result
        for pass_index in range(max_passes):
            improved = False
            relation_order = list(ERROR_RELATIONS)
            rng.shuffle(relation_order)
            for relation in relation_order:
                relation_best = current
                relation_best_result = current_result
                for threshold in grid:
                    trial = copy.deepcopy(current)
                    trial["thresholds"][relation] = threshold
                    trial_result = evaluate(trial, f"coordinate_{pass_index + 1}")
                    if _rank_tuple(trial_result, trial) > _rank_tuple(relation_best_result, relation_best):
                        relation_best, relation_best_result = trial, trial_result
                if _config_key(relation_best) != _config_key(current):
                    current, current_result = relation_best, relation_best_result
                    improved = True
            # Reconsider discrete options after each full threshold pass.
            for mode in ("core", "claim"):
                for infer in (False, True):
                    trial = copy.deepcopy(current)
                    trial["partial_span_mode"] = mode
                    trial["infer_error_label"] = infer
                    trial_result = evaluate(trial, f"discrete_{pass_index + 1}")
                    if _rank_tuple(trial_result, trial) > _rank_tuple(current_result, current):
                        current, current_result = trial, trial_result
                        improved = True
            if not improved:
                break
        if _rank_tuple(current_result, current) > _rank_tuple(best_result, best_config):
            best_config, best_result = copy.deepcopy(current), current_result
        if progress and (start_index == len(starts) or start_index % max(1, len(starts) // 5) == 0):
            print(f"  starts {start_index}/{len(starts)} · best F1={best_result['char_f1_percent']:.2f}%", flush=True)

    # Exhaustive 2D refinement for the two relations most often responsible for recovery.
    if progress:
        print("[3/3] Exact pairwise refinement: partially_supported_by × not_found_in_source...", flush=True)
    pair_base = copy.deepcopy(best_config)
    for partial_threshold in grid:
        for not_found_threshold in grid:
            trial = copy.deepcopy(pair_base)
            trial["thresholds"]["partially_supported_by"] = partial_threshold
            trial["thresholds"]["not_found_in_source"] = not_found_threshold
            trial_result = evaluate(trial, "pairwise_refine")
            if _rank_tuple(trial_result, trial) > _rank_tuple(best_result, best_config):
                best_config, best_result = trial, trial_result

    top_entries = sorted(
        leaderboard.values(),
        key=lambda entry: _rank_tuple(entry["metrics"], entry["config"]),
        reverse=True,
    )
    best_result = cache[_config_key(best_config)]
    best_global_result = cache[_config_key(best_global_config)]
    search_meta = {
        "evaluation_count": evaluation_count,
        "grid_step": step,
        "grid_value_count": len(grid),
        "random_starts": random_starts,
        "max_coordinate_passes": max_passes,
    }
    return (
        {"config": _normalize_gate_config(best_global_config), "metrics": best_global_result},
        {"config": _normalize_gate_config(best_config), "metrics": best_result},
        top_entries,
        global_curve,
        search_meta,
    )


def _detect_prompt_profile(rows: list[dict[str, Any]]) -> tuple[str, str]:
    versions: dict[str, int] = {}
    for row in rows:
        graph = ((row.get("methods") or {}).get(GRAPH_METHOD) or {})
        record = _record_by_component(graph, "alignment")
        if record and record.get("prompt_version"):
            version = str(record["prompt_version"])
            versions[version] = versions.get(version, 0) + 1
    if not versions:
        return "v046_cached", ""
    version = max(versions, key=versions.get)
    for profile, known in ALIGNMENT_PROMPT_VERSIONS.items():
        if version == known:
            return profile, version
    return "v046_cached", version


def _write_best_cases(
    rows: list[dict[str, Any]],
    config: dict[str, Any],
    output_path: Path,
) -> int:
    profile_name = "v050_optimized_from_current_cases"
    ALIGNMENT_GATE_PROFILES[profile_name] = _normalize_gate_config(config)
    written = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            methods = row.get("methods") or {}
            graph_method = copy.deepcopy(methods.get(GRAPH_METHOD) or {})
            response_record = _record_by_component(graph_method, "response_graph")
            alignment_record = _record_by_component(graph_method, "alignment")
            if not response_record or not alignment_record:
                continue
            try:
                response_graph = ResponseClaimGraphOutput.model_validate(response_record["parsed"])
                alignment = DualGraphAlignmentOutput.model_validate(alignment_record["parsed"])
            except Exception:
                continue
            predictions, details = _predictions_from_alignment(
                alignment,
                response_graph,
                str(row.get("response") or ""),
                gate_profile=profile_name,
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
            graph_method["offline_threshold_optimization"] = True
            updated = copy.deepcopy(row)
            updated.setdefault("methods", {})[GRAPH_METHOD] = graph_method
            handle.write(json.dumps(updated, ensure_ascii=False) + "\n")
            written += 1
    return written


def _html_report(summary: dict[str, Any]) -> str:
    raw = summary["raw"]
    global_best = summary["best_global_threshold"]
    optimized = summary["best_relation_specific"]
    cfg = optimized["gate_config"]
    rows = "".join(
        f"<tr><td>{relation}</td><td>{cfg['thresholds'][relation]:.2f}</td></tr>"
        for relation in ERROR_RELATIONS
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Alignment threshold optimizer</title>
<style>
body{{font-family:Segoe UI,Arial,sans-serif;margin:32px;max-width:1000px;color:#172033}}h1{{margin-bottom:4px}}.sub{{color:#667085}}
.grid{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px;margin:24px 0}}.card{{border:1px solid #d8dee9;border-radius:14px;padding:16px}}
.big{{font-size:30px;font-weight:700}}table{{border-collapse:collapse;width:100%}}th,td{{border-bottom:1px solid #e4e7ec;text-align:left;padding:9px}}
.win{{background:#eefbf3}}code{{background:#f2f4f7;padding:2px 5px;border-radius:5px}}@media(max-width:700px){{.grid{{grid-template-columns:1fr}}}}
</style></head><body>
<h1>Alignment threshold optimization</h1><div class="sub">{summary['completed_cases']} completed cases · 0 API calls · fitted directly to these cases</div>
<div class="grid">
<div class="card"><div>Raw Direct F1</div><div class="big">{raw['char_f1_percent']:.2f}%</div></div>
<div class="card"><div>Best single threshold F1</div><div class="big">{global_best['metrics']['char_f1_percent']:.2f}%</div><div>threshold {global_best['global_threshold']:.2f}</div></div>
<div class="card win"><div>Best tuned gate F1</div><div class="big">{optimized['metrics']['char_f1_percent']:.2f}%</div><div>Δ vs Raw {optimized['metrics']['char_f1_percent']-raw['char_f1_percent']:+.2f} pp</div></div>
</div>
<h2>Best relation-specific thresholds</h2><table><thead><tr><th>Relation</th><th>Threshold</th></tr></thead><tbody>{rows}</tbody></table>
<p>Span mode: <code>{cfg['partial_span_mode']}</code> · infer missing error label: <code>{str(cfg['infer_error_label']).lower()}</code></p>
<h2>Trade-offs</h2><table><tbody>
<tr><th>Precision</th><td>{optimized['metrics']['char_precision_percent']:.2f}%</td></tr>
<tr><th>Recall</th><td>{optimized['metrics']['char_recall_percent']:.2f}%</td></tr>
<tr><th>Clean false-positive rate</th><td>{optimized['metrics']['clean_false_positive_rate_percent']:.2f}%</td></tr>
<tr><th>Hallucinated sensitivity</th><td>{optimized['metrics']['hallucinated_sensitivity_percent']:.2f}%</td></tr>
<tr><th>Strict rescue cases</th><td>{optimized['metrics']['strict_rescue']}</td></tr>
<tr><th>Hallucination regressions</th><td>{optimized['metrics']['hallucination_regression']}</td></tr>
</tbody></table>
<p><b>Important:</b> this is the maximum observed F1 on the same cases used to choose the thresholds. It may not transfer to unseen cases.</p>
</body></html>"""


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Search the completed RAGTruth cases for the Alignment threshold configuration with maximum observed character F1. No API calls."
    )
    parser.add_argument("--cases-jsonl", default="")
    parser.add_argument("--generation-cache", default="", help="Optional explicit generation_cache*.json used when no cases.jsonl exists.")
    parser.add_argument("--step", type=float, default=0.01)
    parser.add_argument("--random-starts", type=int, default=12)
    parser.add_argument("--max-passes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2040)
    parser.add_argument("--top-k", type=int, default=100)
    args = parser.parse_args()
    if args.step <= 0 or args.step > 0.25:
        raise ValueError("--step must be >0 and <=0.25")

    root = Path(__file__).resolve().parent
    if args.cases_jsonl:
        cases_path = Path(args.cases_jsonl)
        if not cases_path.is_absolute():
            cases_path = root / cases_path
    else:
        cases_path = _latest_cases_file(root)
        if cases_path is None:
            explicit_cache = Path(args.generation_cache) if args.generation_cache else None
            if explicit_cache is not None and not explicit_cache.is_absolute():
                explicit_cache = root / explicit_cache
            cases_path = _recover_cases_from_cache(root, explicit_cache=explicit_cache)
    rows = _load_rows_tolerant(cases_path)
    prepared, skipped = _prepare_cases(rows)
    if not prepared:
        raise RuntimeError("No cases contained reusable Raw, Response Graph, and Alignment records")
    prompt_profile, prompt_version = _detect_prompt_profile(rows)
    raw = _raw_summary(prepared)
    print(f"Loaded {len(rows)} completed rows; {len(prepared)} eligible for optimization; {skipped} skipped.")
    print(f"Raw Direct micro-F1 on these cases: {raw['char_f1_percent']:.2f}%")
    print("API calls: 0\n")

    global_best, optimized, top_entries, global_curve, search_meta = _search(
        prepared,
        step=args.step,
        random_starts=args.random_starts,
        max_passes=args.max_passes,
        seed=args.seed,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = root / "outputs" / "alignment_threshold_optimizer"
    output_dir = output_root / f"threshold_opt_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    best_global_threshold = next(iter(global_best["config"]["thresholds"].values()))
    best_gate_payload = {
        "schema_version": "0.50.0-alignment-threshold-optimizer",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_cases_jsonl": str(cases_path),
        "completed_cases": len(prepared),
        "fitted_on_same_cases": True,
        "recommended_alignment_prompt_profile": prompt_profile,
        "alignment_prompt_version_observed": prompt_version,
        "gate_config": optimized["config"],
        "metrics_on_fitted_cases": {k: v for k, v in optimized["metrics"].items() if k != "per_case"},
    }
    summary = {
        "schema_version": best_gate_payload["schema_version"],
        "source_cases_jsonl": str(cases_path),
        "completed_rows_loaded": len(rows),
        "completed_cases": len(prepared),
        "skipped_rows": skipped,
        "api_calls": 0,
        "raw": raw,
        "best_global_threshold": {
            "global_threshold": best_global_threshold,
            "gate_config": global_best["config"],
            "metrics": {k: v for k, v in global_best["metrics"].items() if k != "per_case"},
        },
        "best_relation_specific": {
            "gate_config": optimized["config"],
            "metrics": {k: v for k, v in optimized["metrics"].items() if k != "per_case"},
        },
        "search": search_meta,
        "recommended_alignment_prompt_profile": prompt_profile,
        "alignment_prompt_version_observed": prompt_version,
        "warning": "Thresholds were selected to maximize F1 on these same completed cases.",
    }

    (output_dir / "best_gate.json").write_text(json.dumps(best_gate_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "report.html").write_text(_html_report(summary), encoding="utf-8")
    _write_best_cases(rows, optimized["config"], output_dir / "best_reprojected_cases.jsonl")

    with (output_dir / "global_threshold_curve.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(global_curve[0].keys()))
        writer.writeheader()
        writer.writerows(global_curve)

    top_rows: list[dict[str, Any]] = []
    for rank, entry in enumerate(top_entries[: max(1, args.top_k)], 1):
        cfg = entry["config"]
        metrics = entry["metrics"]
        top_rows.append({
            "rank": rank,
            "phase": entry["phase"],
            "char_f1_percent": metrics["char_f1_percent"],
            "char_precision_percent": metrics["char_precision_percent"],
            "char_recall_percent": metrics["char_recall_percent"],
            "clean_false_positive_rate_percent": metrics["clean_false_positive_rate_percent"],
            "hallucinated_sensitivity_percent": metrics["hallucinated_sensitivity_percent"],
            "partial_span_mode": cfg["partial_span_mode"],
            "infer_error_label": cfg["infer_error_label"],
            **{f"threshold_{relation}": cfg["thresholds"][relation] for relation in ERROR_RELATIONS},
        })
    with (output_dir / "top_configs.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(top_rows[0].keys()))
        writer.writeheader()
        writer.writerows(top_rows)

    output_root.mkdir(parents=True, exist_ok=True)
    for source_name, target_name in (
        ("best_gate.json", "latest_best_gate.json"),
        ("summary.json", "latest_summary.json"),
        ("report.html", "latest_report.html"),
    ):
        shutil.copy2(output_dir / source_name, output_root / target_name)
    (output_root / "LATEST_OUTPUT_DIR.txt").write_text(str(output_dir), encoding="utf-8")

    print("\n================ THRESHOLD OPTIMIZATION RESULT ================")
    print(f"Raw Direct F1:             {raw['char_f1_percent']:.2f}%")
    print(
        f"Best ONE global threshold: {global_best['metrics']['char_f1_percent']:.2f}% "
        f"at threshold={best_global_threshold:.2f}, mode={global_best['config']['partial_span_mode']}, "
        f"infer={global_best['config']['infer_error_label']}"
    )
    print(f"Best relation-specific F1: {optimized['metrics']['char_f1_percent']:.2f}%")
    print(f"Delta vs Raw:              {optimized['metrics']['char_f1_percent'] - raw['char_f1_percent']:+.2f} pp")
    print("Thresholds:")
    for relation in ERROR_RELATIONS:
        value = optimized["config"]["thresholds"][relation]
        suffix = " (disabled)" if value > 1.0 else ""
        print(f"  {relation:24s} {value:.2f}{suffix}")
    print(f"Span mode:                 {optimized['config']['partial_span_mode']}")
    print(f"Infer missing error label: {optimized['config']['infer_error_label']}")
    print(f"Clean FP rate:             {optimized['metrics']['clean_false_positive_rate_percent']:.2f}%")
    print(f"Hallucination sensitivity: {optimized['metrics']['hallucinated_sensitivity_percent']:.2f}%")
    print(f"Strict rescue cases:       {optimized['metrics']['strict_rescue']}")
    print(f"Search configs evaluated:  {search_meta['evaluation_count']}")
    print(f"Saved: {output_dir}")
    print(f"Stable best gate: {output_root / 'latest_best_gate.json'}")
    print("===============================================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())

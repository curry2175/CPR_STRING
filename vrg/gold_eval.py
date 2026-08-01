from __future__ import annotations

import csv
import io
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Iterable

from .verifier import verify_case


@dataclass
class GoldEvalOptions:
    prefer_z3: bool = True
    compute_counterfactuals: bool = False


def parse_gold_jsonl(text: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_no, line in enumerate(str(text or "").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at line {line_no}: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"Gold record at line {line_no} must be an object")
        records.append(row)
    if not records:
        raise ValueError("No gold benchmark records were provided")
    return records


def _prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": round(precision * 100, 2), "recall": round(recall * 100, 2), "f1": round(f1 * 100, 2)}


def _classification_metrics(gold: list[str], pred: list[str]) -> dict[str, Any]:
    labels = sorted(set(gold) | set(pred))
    confusion = {g: {p: 0 for p in labels} for g in labels}
    for g, p in zip(gold, pred):
        confusion[g][p] += 1
    per_label: dict[str, Any] = {}
    f1s: list[float] = []
    for label in labels:
        tp = sum(1 for g, p in zip(gold, pred) if g == label and p == label)
        fp = sum(1 for g, p in zip(gold, pred) if g != label and p == label)
        fn = sum(1 for g, p in zip(gold, pred) if g == label and p != label)
        metrics = _prf(tp, fp, fn)
        per_label[label] = {**metrics, "support": sum(g == label for g in gold)}
        f1s.append(metrics["f1"])
    accuracy = sum(g == p for g, p in zip(gold, pred)) / len(gold) * 100 if gold else None
    return {
        "count": len(gold),
        "accuracy_percent": round(accuracy, 2) if accuracy is not None else None,
        "macro_f1_percent": round(sum(f1s) / len(f1s), 2) if f1s else None,
        "labels": labels,
        "per_label": per_label,
        "confusion_matrix": confusion,
    }


def _used_parent_set(node: dict[str, Any]) -> set[str]:
    return set(str(x) for x in (
        list(node.get("source_matches") or [])
        + list(node.get("reasoning_dependencies") or [])
        + list(node.get("reasoning_conflict_dependencies") or [])
    ))


def _acceptable_sets(annotation: dict[str, Any]) -> list[set[str]]:
    raw = annotation.get("acceptable_parent_sets")
    if raw is None and "parents" in annotation:
        raw = [annotation.get("parents") or []]
    if not isinstance(raw, list):
        return []
    result: list[set[str]] = []
    for item in raw:
        if isinstance(item, str):
            item = [x.strip() for x in item.split(",") if x.strip()]
        if isinstance(item, list):
            result.append(set(str(x) for x in item))
    return result


def _best_parent_match(predicted: set[str], acceptable: list[set[str]]) -> tuple[bool | None, float | None, set[str] | None]:
    if not acceptable:
        return None, None, None
    best_f1 = -1.0
    best: set[str] | None = None
    for gold in acceptable:
        tp = len(predicted & gold)
        fp = len(predicted - gold)
        fn = len(gold - predicted)
        f1 = 100.0 if not predicted and not gold else _prf(tp, fp, fn)["f1"]
        if f1 > best_f1:
            best_f1 = f1
            best = gold
    return any(predicted == gold for gold in acceptable), best_f1, best


def _confidence_value(node: dict[str, Any]) -> float:
    value = str(node.get("dependency_confidence") or "")
    return {
        "declared_and_verified": 1.0,
        "inferred_unique": 0.85,
        "inferred_ambiguous": 0.55,
        "declared_but_insufficient": 0.15,
        "no_support": 0.1,
        "not_applicable": 0.0,
    }.get(value, 0.4)


def build_review_queue(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    queue: list[dict[str, Any]] = []
    review_confidence = {"inferred_ambiguous", "declared_but_insufficient", "no_support", "not_applicable"}
    for result in results:
        for node in result.get("nodes") or []:
            if node.get("kind") not in {"reasoning", "answer"}:
                continue
            reasons: list[str] = []
            if node.get("proof_status") in {"untranslatable", "ungrounded", "contradiction"}:
                reasons.append(str(node.get("proof_status")))
            if node.get("chain_status") in {"blocked_by_upstream_error", "insufficient_declared_support"}:
                reasons.append(str(node.get("chain_status")))
            if node.get("dependency_confidence") in review_confidence:
                reasons.append(str(node.get("dependency_confidence")))
            if node.get("atomicity_status") not in {None, "atomic"}:
                reasons.append(str(node.get("atomicity_status")))
            if reasons:
                queue.append({
                    "case_id": result.get("case_id"),
                    "node_id": node.get("id"),
                    "kind": node.get("kind"),
                    "text": node.get("text"),
                    "proof_status": node.get("proof_status"),
                    "chain_status": node.get("chain_status"),
                    "dependency_confidence": node.get("dependency_confidence"),
                    "review_reasons": ";".join(sorted(set(reasons))),
                    "predicted_parents": ",".join(sorted(_used_parent_set(node))),
                })
    return queue


def evaluate_gold_records(records: Iterable[dict[str, Any]], options: GoldEvalOptions | None = None) -> dict[str, Any]:
    options = options or GoldEvalOptions()
    records = list(records)
    proof_gold: list[str] = []
    proof_pred: list[str] = []
    chain_gold: list[str] = []
    chain_pred: list[str] = []
    role_gold: list[str] = []
    role_pred: list[str] = []
    root_tp = root_fp = root_fn = 0
    parent_exact_values: list[bool] = []
    parent_f1_values: list[float] = []
    final_proof_correct: list[bool] = []
    final_chain_correct: list[bool] = []
    node_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    calibration_rows: list[tuple[float, bool]] = []

    for index, record in enumerate(records, 1):
        case = record.get("case") if isinstance(record.get("case"), dict) else record.get("input_case")
        gold = record.get("gold") or record.get("annotations") or {}
        if not isinstance(case, dict) or not isinstance(gold, dict):
            errors.append({"record_index": index, "error": "Record requires case and gold objects"})
            continue
        try:
            result = verify_case(case, prefer_z3=options.prefer_z3, compute_counterfactuals=options.compute_counterfactuals)
        except Exception as exc:
            errors.append({"record_index": index, "case_id": case.get("id"), "error": f"{type(exc).__name__}: {exc}"})
            continue
        results.append(result)
        lookup = {str(node.get("id")): node for node in result.get("nodes") or []}
        gold_nodes = gold.get("nodes") or {}
        if isinstance(gold_nodes, list):
            gold_nodes = {str(row.get("id")): row for row in gold_nodes if isinstance(row, dict)}
        case_node_total = case_node_correct = 0
        for node_id, annotation in gold_nodes.items():
            if not isinstance(annotation, dict):
                continue
            predicted = lookup.get(str(node_id))
            if predicted is None:
                node_rows.append({"case_id": case.get("id"), "node_id": node_id, "missing_prediction": True})
                continue
            row: dict[str, Any] = {
                "case_id": case.get("id"),
                "node_id": node_id,
                "text": predicted.get("text"),
                "missing_prediction": False,
            }
            if annotation.get("proof_status") is not None:
                g = str(annotation["proof_status"]); p = str(predicted.get("proof_status"))
                proof_gold.append(g); proof_pred.append(p)
                row.update({"gold_proof_status": g, "predicted_proof_status": p, "proof_correct": g == p})
                case_node_total += 1; case_node_correct += int(g == p)
            if annotation.get("chain_status") is not None:
                g = str(annotation["chain_status"]); p = str(predicted.get("chain_status"))
                chain_gold.append(g); chain_pred.append(p)
                row.update({"gold_chain_status": g, "predicted_chain_status": p, "chain_correct": g == p})
                case_node_total += 1; case_node_correct += int(g == p)
            if annotation.get("role") is not None:
                g = str(annotation["role"]); p = str(predicted.get("reasoning_role"))
                role_gold.append(g); role_pred.append(p)
                row.update({"gold_role": g, "predicted_role": p, "role_correct": g == p})
            acceptable = _acceptable_sets(annotation)
            predicted_parents = _used_parent_set(predicted)
            exact, best_f1, best_gold = _best_parent_match(predicted_parents, acceptable)
            if exact is not None:
                parent_exact_values.append(bool(exact)); parent_f1_values.append(float(best_f1 or 0))
                calibration_rows.append((_confidence_value(predicted), bool(exact)))
                row.update({
                    "predicted_parents": sorted(predicted_parents),
                    "acceptable_parent_sets": [sorted(x) for x in acceptable],
                    "parent_exact_match": exact,
                    "parent_best_f1_percent": best_f1,
                    "best_gold_parent_set": sorted(best_gold or set()),
                })
            gold_root = bool(annotation.get("root_error", False))
            pred_root = str(node_id) in set(result.get("summary", {}).get("root_error_nodes") or [])
            if gold_root and pred_root: root_tp += 1
            elif not gold_root and pred_root: root_fp += 1
            elif gold_root and not pred_root: root_fn += 1
            row.update({"gold_root_error": gold_root, "predicted_root_error": pred_root})
            node_rows.append(row)

        final_gold = gold.get("final") or {}
        final_node = lookup.get("final", {})
        if final_gold.get("proof_status") is not None:
            final_proof_correct.append(str(final_gold["proof_status"]) == str(final_node.get("proof_status")))
        if final_gold.get("chain_status") is not None:
            final_chain_correct.append(str(final_gold["chain_status"]) == str(final_node.get("chain_status")))
        case_rows.append({
            "case_id": case.get("id"),
            "annotated_node_checks": case_node_total,
            "correct_node_checks": case_node_correct,
            "node_check_accuracy_percent": round(case_node_correct / case_node_total * 100, 2) if case_node_total else None,
            "final_proof_correct": final_proof_correct[-1] if final_gold.get("proof_status") is not None else None,
            "final_chain_correct": final_chain_correct[-1] if final_gold.get("chain_status") is not None else None,
            "reasoning_integrity_score": result.get("summary", {}).get("reasoning_integrity_score"),
        })

    bins = {
        "high_0.80_1.00": [correct for conf, correct in calibration_rows if conf >= 0.8],
        "medium_0.50_0.79": [correct for conf, correct in calibration_rows if 0.5 <= conf < 0.8],
        "low_0.00_0.49": [correct for conf, correct in calibration_rows if conf < 0.5],
    }
    calibration = {
        name: {"count": len(values), "parent_accuracy_percent": round(sum(values) / len(values) * 100, 2) if values else None}
        for name, values in bins.items()
    }
    review_queue = build_review_queue(results)
    summary = {
        "input_records": len(records),
        "evaluated_cases": len(results),
        "failed_cases": len(errors),
        "proof_status": _classification_metrics(proof_gold, proof_pred),
        "chain_status": _classification_metrics(chain_gold, chain_pred),
        "reasoning_role": _classification_metrics(role_gold, role_pred),
        "parent_path_exact_accuracy_percent": round(sum(parent_exact_values) / len(parent_exact_values) * 100, 2) if parent_exact_values else None,
        "parent_path_mean_best_f1_percent": round(sum(parent_f1_values) / len(parent_f1_values), 2) if parent_f1_values else None,
        "root_error_localization": _prf(root_tp, root_fp, root_fn),
        "final_proof_accuracy_percent": round(sum(final_proof_correct) / len(final_proof_correct) * 100, 2) if final_proof_correct else None,
        "final_chain_accuracy_percent": round(sum(final_chain_correct) / len(final_chain_correct) * 100, 2) if final_chain_correct else None,
        "review_queue_count": len(review_queue),
        "dependency_confidence_calibration": calibration,
    }
    return {
        "schema_version": "0.15.0",
        "summary": summary,
        "cases": case_rows,
        "nodes": node_rows,
        "review_queue": review_queue,
        "errors": errors,
        "results": results,
    }


def rows_to_csv(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        cooked = {k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k, v in row.items()}
        writer.writerow(cooked)
    return buffer.getvalue()

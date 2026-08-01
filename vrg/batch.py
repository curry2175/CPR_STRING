from __future__ import annotations

import csv
import io
import json
import math
from collections import Counter
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Iterable

from .verifier import verify_case


@dataclass
class BatchOptions:
    prefer_z3: bool = True
    compute_counterfactuals: bool = False


def parse_jsonl(text: str) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"JSONL line {line_number}: {exc.msg}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"JSONL line {line_number} must contain a JSON object")
        cases.append(value)
    if not cases:
        raise ValueError("No cases were found in the JSONL input")
    return cases


def _percent(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return round(numerator / denominator * 100.0, 2)


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 3) if values else None


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return round(ordered[middle], 3)
    return round((ordered[middle - 1] + ordered[middle]) / 2.0, 3)


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(ordered[0], 3)
    rank = (len(ordered) - 1) * percentile
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return round(ordered[lower], 3)
    weight = rank - lower
    result = ordered[lower] * (1 - weight) + ordered[upper] * weight
    return round(result, 3)


def _root_error_types(result: dict[str, Any]) -> Counter[str]:
    summary = result.get("summary") or {}
    root_ids = set(summary.get("root_error_nodes") or [])
    counter: Counter[str] = Counter()
    for node in result.get("nodes") or []:
        if node.get("id") in root_ids:
            counter[str(node.get("proof_status") or node.get("status") or "unknown")] += 1
    return counter


def _case_row(result: dict[str, Any], runtime_ms: float) -> dict[str, Any]:
    summary = result.get("summary") or {}
    nodes = result.get("nodes") or []
    reasoning_nodes = [node for node in nodes if node.get("kind") == "reasoning"]
    return {
        "case_id": result.get("case_id"),
        "engine": result.get("engine"),
        "predicted_answer": result.get("predicted_answer"),
        "gold_answer": result.get("gold_answer"),
        "answer_correct": bool(result.get("answer_correct")),
        "final_proof_status": summary.get("final_proof_status", summary.get("final_status")),
        "final_chain_status": summary.get("final_chain_status"),
        "all_reasoning_proof_valid": bool(summary.get("all_reasoning_proof_valid")),
        "all_reasoning_chain_valid": bool(summary.get("all_reasoning_chain_valid")),
        "invalid_reasoning_count": int(summary.get("invalid_reasoning_count") or 0),
        "blocked_reasoning_count": int(summary.get("blocked_reasoning_count") or 0),
        "chain_error_count": int(summary.get("chain_error_count") or 0),
        "valid_answer_but_invalid_reasoning": bool(summary.get("valid_answer_but_invalid_reasoning")),
        "semantic_relation_count": int(summary.get("semantic_relation_count") or 0),
        "proof_usable_semantic_relations": int(summary.get("proof_usable_semantic_relations") or 0),
        "advisory_semantic_relations": int(summary.get("advisory_semantic_relations") or 0),
        "semantic_hint_count": int(summary.get("semantic_hint_count") or 0),
        "node_count": len(nodes),
        "reasoning_node_count": len(reasoning_nodes),
        "reasoning_integrity_score": summary.get("reasoning_integrity_score"),
        "ambiguous_dependency_step_count": int(summary.get("ambiguous_dependency_step_count") or 0),
        "insufficient_declared_support_count": int(summary.get("insufficient_declared_support_count") or 0),
        "compound_reasoning_step_count": int(summary.get("compound_reasoning_step_count") or 0),
        "final_minimal_proof_count": int(summary.get("final_minimal_proof_count") or 0),
        "runtime_ms": round(runtime_ms, 3),
    }


def evaluate_cases(cases: Iterable[dict[str, Any]], options: BatchOptions | None = None) -> dict[str, Any]:
    options = options or BatchOptions()
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    full_results: list[dict[str, Any]] = []
    proof_status_counts: Counter[str] = Counter()
    chain_status_counts: Counter[str] = Counter()
    engine_counts: Counter[str] = Counter()
    root_error_counts: Counter[str] = Counter()

    started = perf_counter()
    for index, case in enumerate(cases, start=1):
        case_id = str(case.get("case_id") or case.get("id") or f"case_{index}")
        case_started = perf_counter()
        try:
            result = verify_case(
                case,
                prefer_z3=options.prefer_z3,
                compute_counterfactuals=options.compute_counterfactuals,
            )
            runtime_ms = (perf_counter() - case_started) * 1000.0
            row = _case_row(result, runtime_ms)
            rows.append(row)
            full_results.append(result)
            proof_status_counts[str(row["final_proof_status"] or "unknown")] += 1
            chain_status_counts[str(row["final_chain_status"] or "unknown")] += 1
            engine_counts[str(row["engine"] or "unknown")] += 1
            root_error_counts.update(_root_error_types(result))
        except Exception as exc:  # keep the rest of the batch running
            runtime_ms = (perf_counter() - case_started) * 1000.0
            errors.append(
                {
                    "case_id": case_id,
                    "index": index,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "runtime_ms": round(runtime_ms, 3),
                }
            )

    total_runtime_ms = (perf_counter() - started) * 1000.0
    completed = len(rows)
    total = completed + len(errors)
    runtimes = [float(row["runtime_ms"]) for row in rows]
    correct = sum(1 for row in rows if row["answer_correct"])
    proof_valid = sum(1 for row in rows if row["final_proof_status"] == "valid")
    chain_valid = sum(1 for row in rows if row["final_chain_status"] == "valid")
    reasoning_proof_clean = sum(1 for row in rows if row["all_reasoning_proof_valid"])
    reasoning_chain_clean = sum(1 for row in rows if row["all_reasoning_chain_valid"])
    right_answer_bad_reasoning = sum(1 for row in rows if row["valid_answer_but_invalid_reasoning"])
    cases_with_reasoning_error = sum(1 for row in rows if row["invalid_reasoning_count"] > 0 or row["chain_error_count"] > 0)

    summary = {
        "schema_version": "0.15.0",
        "total_cases": total,
        "completed_cases": completed,
        "failed_cases": len(errors),
        "answer_correct_count": correct,
        "answer_accuracy_percent": _percent(correct, completed),
        "final_proof_valid_count": proof_valid,
        "final_proof_valid_percent": _percent(proof_valid, completed),
        "final_chain_valid_count": chain_valid,
        "final_chain_valid_percent": _percent(chain_valid, completed),
        "reasoning_proof_clean_count": reasoning_proof_clean,
        "reasoning_proof_clean_percent": _percent(reasoning_proof_clean, completed),
        "reasoning_chain_clean_count": reasoning_chain_clean,
        "reasoning_chain_clean_percent": _percent(reasoning_chain_clean, completed),
        "cases_with_reasoning_error_count": cases_with_reasoning_error,
        "cases_with_reasoning_error_percent": _percent(cases_with_reasoning_error, completed),
        "valid_answer_but_invalid_reasoning_count": right_answer_bad_reasoning,
        "valid_answer_but_invalid_reasoning_percent": _percent(right_answer_bad_reasoning, completed),
        "final_proof_status_distribution": dict(sorted(proof_status_counts.items())),
        "final_chain_status_distribution": dict(sorted(chain_status_counts.items())),
        "root_error_type_distribution": dict(sorted(root_error_counts.items())),
        "engine_distribution": dict(sorted(engine_counts.items())),
        "total_nodes": sum(int(row["node_count"]) for row in rows),
        "average_nodes_per_case": _mean([float(row["node_count"]) for row in rows]),
        "average_reasoning_nodes_per_case": _mean([float(row["reasoning_node_count"]) for row in rows]),
        "average_reasoning_integrity_score": _mean([float(row["reasoning_integrity_score"]) for row in rows if row.get("reasoning_integrity_score") is not None]),
        "cases_with_ambiguous_dependencies": sum(int(row.get("ambiguous_dependency_step_count") or 0) > 0 for row in rows),
        "cases_with_insufficient_declared_support": sum(int(row.get("insufficient_declared_support_count") or 0) > 0 for row in rows),
        "cases_with_compound_reasoning_steps": sum(int(row.get("compound_reasoning_step_count") or 0) > 0 for row in rows),
        "average_runtime_ms": _mean(runtimes),
        "median_runtime_ms": _median(runtimes),
        "p95_runtime_ms": _percentile(runtimes, 0.95),
        "total_runtime_ms": round(total_runtime_ms, 3),
        "prefer_z3": options.prefer_z3,
        "compute_counterfactuals": options.compute_counterfactuals,
    }
    return {
        "schema_version": "0.15.0",
        "summary": summary,
        "cases": rows,
        "errors": errors,
        "results": full_results,
    }


def rows_to_csv(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def errors_to_csv(errors: list[dict[str, Any]]) -> str:
    if not errors:
        return ""
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(errors[0].keys()))
    writer.writeheader()
    writer.writerows(errors)
    return output.getvalue()

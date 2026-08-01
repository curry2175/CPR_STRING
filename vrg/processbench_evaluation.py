from __future__ import annotations

import copy
import hashlib
import html
import json
import math
import os
import random
import statistics
import time
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Literal

from pydantic import BaseModel, Field

from .openai_runner import ALLOWED_REASONING_EFFORTS, _load_local_env, _usage_dict


SCHEMA_VERSION = "0.34.0"
CACHE_SCHEMA_VERSION = "0.34.0-processbench-cache"
DATASET_NAME = "Qwen/ProcessBench"
DATASET_SERVER_ROWS = "https://datasets-server.huggingface.co/rows"
PROCESSBENCH_SPLITS = ("gsm8k", "math", "olympiadbench", "omnimath")
METHOD_PROMPT_VERSIONS = {
    "small_direct_step": "v034-processbench-direct-first-error",
    "small_checklist_step": "v034-processbench-step-checklist-no-edges",
    "small_dependency_graph": "v034-processbench-dependency-graph",
    "reference_direct_step": "v034-processbench-direct-first-error",
}
MODEL_PRICE_USD_PER_MILLION: dict[str, tuple[float, float]] = {
    "gpt-5.4-nano": (0.20, 1.25),
    "gpt-5.4-mini": (0.75, 4.50),
    "gpt-5-nano": (0.05, 0.40),
    "gpt-5-mini": (0.25, 2.00),
}


class DirectFirstErrorOutput(BaseModel):
    first_error_step: int = Field(
        description="Zero-based index of the earliest erroneous solution step, or -1 if every step is correct"
    )
    explanation: str = Field(default="", description="Brief public critique of the decisive step")


class StepReview(BaseModel):
    step_index: int
    verdict: Literal["correct", "incorrect", "uncertain"]
    explanation: str = ""


class ChecklistFirstErrorOutput(BaseModel):
    step_reviews: list[StepReview] = Field(default_factory=list)
    first_error_step: int
    summary: str = ""


class DependencyStepNode(BaseModel):
    step_index: int
    depends_on: list[int] = Field(
        default_factory=list,
        description="Zero-based indices of earlier solution steps directly used by this step",
    )
    uses_problem: bool = True
    verdict: Literal["correct", "incorrect", "uncertain"]
    error_type: Literal[
        "none",
        "arithmetic",
        "algebra",
        "logical_inference",
        "misread_problem",
        "unsupported_assumption",
        "contradiction",
        "definition_or_theorem",
        "other",
    ] = "none"
    explanation: str = ""


class DependencyGraphFirstErrorOutput(BaseModel):
    nodes: list[DependencyStepNode] = Field(default_factory=list)
    first_error_step: int
    summary: str = ""


def _stable_hash(payload: Any) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _estimated_cost(model: str, usage: dict[str, Any]) -> float | None:
    prices = MODEL_PRICE_USD_PER_MILLION.get(model)
    if prices is None:
        return None
    input_price, output_price = prices
    return round(
        int(usage.get("input_tokens") or 0) / 1_000_000 * input_price
        + int(usage.get("output_tokens") or 0) / 1_000_000 * output_price,
        8,
    )


def _parse_output(response: Any, output_type: type[BaseModel]) -> BaseModel:
    parsed = getattr(response, "output_parsed", None)
    if parsed is None:
        for output in getattr(response, "output", []) or []:
            if getattr(output, "type", None) != "message":
                continue
            for item in getattr(output, "content", []) or []:
                parsed = getattr(item, "parsed", None)
                if parsed is not None:
                    break
            if parsed is not None:
                break
    if parsed is None:
        output_text = str(getattr(response, "output_text", "") or "").strip()
        if output_text:
            parsed = output_type.model_validate_json(output_text)
    if parsed is None:
        raise ValueError("Model returned no parsed structured output")
    if isinstance(parsed, output_type):
        return parsed
    return output_type.model_validate(parsed)


def _call_parsed(
    client: Any,
    *,
    model: str,
    reasoning_effort: str,
    max_output_tokens: int,
    system: str,
    user: str,
    output_type: type[BaseModel],
) -> tuple[BaseModel, dict[str, Any]]:
    started = time.perf_counter()
    response = client.responses.parse(
        model=model,
        reasoning={"effort": reasoning_effort},
        max_output_tokens=max_output_tokens,
        store=False,
        input=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        text_format=output_type,
    )
    return _parse_output(response, output_type), {
        "usage": _usage_dict(response),
        "latency_ms": round((time.perf_counter() - started) * 1000, 3),
        "response_id": str(getattr(response, "id", "")),
        "model_returned": str(getattr(response, "model", model)),
    }


def _render_case(case: dict[str, Any]) -> str:
    steps = case.get("steps") or []
    step_block = "\n\n".join(f"STEP {index} (zero-based):\n{step}" for index, step in enumerate(steps))
    return f"MATHEMATICAL PROBLEM\n{case['problem']}\n\nPROPOSED SOLUTION\n{step_block}"


def _direct_prompts(case: dict[str, Any]) -> tuple[str, str]:
    system = (
        "You are a rigorous critic of step-by-step mathematical solutions. Identify the earliest solution step that "
        "contains a mathematical, logical, or problem-interpretation error. Step indices are zero-based. A later step "
        "that merely inherits an earlier wrong result is not the first error. Return -1 only when every step is valid "
        "and the solution correctly addresses the problem. Do not solve a different problem, do not use the dataset "
        "label, and keep the explanation concise and inspectable."
    )
    user = _render_case(case) + "\n\nReturn the zero-based first error index, or -1 if all steps are correct."
    return system, user


def _checklist_prompts(case: dict[str, Any]) -> tuple[str, str]:
    system = (
        "You are a rigorous mathematical process critic. Review every supplied solution step sequentially, without "
        "constructing dependency edges. For each zero-based step, mark it correct, incorrect, or uncertain and give a "
        "brief reason. Then identify the earliest step with an intrinsic mathematical, logical, or interpretation error. "
        "Do not call a later step the first error merely because it uses an earlier wrong result. Return -1 only if all "
        "steps are correct. The step_reviews list must cover every supplied step exactly once and in order."
    )
    user = _render_case(case) + "\n\nReturn the full sequential checklist and the zero-based first error index."
    return system, user


def _graph_prompts(case: dict[str, Any]) -> tuple[str, str]:
    system = (
        "You are a rigorous mathematical process verifier that builds a compact dependency graph over the supplied "
        "solution. Create exactly one node for every zero-based solution step. For each node, list only the earlier step "
        "indices it directly depends on, indicate whether it also uses the original problem, and judge the step as "
        "correct, incorrect, or uncertain. Check whether the step follows from the original problem and its direct "
        "parents; distinguish a new local error from propagation of an earlier error. A later step that correctly follows "
        "from an already wrong parent is not the earliest error. Classify the local error type when invalid. Finally "
        "return the earliest intrinsically erroneous step, or -1 only if every step is valid. Dependency indices must be "
        "strictly smaller than the node's own index. Keep explanations short and do not add hidden or invented steps."
    )
    user = _render_case(case) + "\n\nReturn the typed step-dependency graph and zero-based first error index."
    return system, user


def _normalize_prediction(value: Any, step_count: int) -> int | None:
    try:
        prediction = int(value)
    except (TypeError, ValueError):
        return None
    if prediction == -1:
        return -1
    if 0 <= prediction < step_count:
        return prediction
    return None


def _normalize_step_reviews(reviews: list[StepReview], step_count: int) -> dict[str, Any]:
    by_index: dict[int, dict[str, Any]] = {}
    duplicate_indices: list[int] = []
    invalid_indices: list[int] = []
    for review in reviews:
        index = int(review.step_index)
        if not 0 <= index < step_count:
            invalid_indices.append(index)
            continue
        if index in by_index:
            duplicate_indices.append(index)
            continue
        by_index[index] = review.model_dump()
    normalized = [by_index.get(i, {"step_index": i, "verdict": "uncertain", "explanation": "Missing model review"}) for i in range(step_count)]
    derived = next((row["step_index"] for row in normalized if row["verdict"] == "incorrect"), -1)
    return {
        "step_reviews": normalized,
        "derived_first_error_step": derived,
        "coverage_percent": round(100 * len(by_index) / step_count, 2) if step_count else 100.0,
        "duplicate_indices": duplicate_indices,
        "invalid_indices": invalid_indices,
    }


def _normalize_graph_nodes(nodes: list[DependencyStepNode], step_count: int) -> dict[str, Any]:
    by_index: dict[int, dict[str, Any]] = {}
    duplicate_indices: list[int] = []
    invalid_node_indices: list[int] = []
    invalid_dependencies: list[dict[str, int]] = []
    for node in nodes:
        index = int(node.step_index)
        if not 0 <= index < step_count:
            invalid_node_indices.append(index)
            continue
        if index in by_index:
            duplicate_indices.append(index)
            continue
        valid_dependencies: list[int] = []
        for parent in node.depends_on:
            parent_index = int(parent)
            if 0 <= parent_index < index:
                if parent_index not in valid_dependencies:
                    valid_dependencies.append(parent_index)
            else:
                invalid_dependencies.append({"step_index": index, "parent_index": parent_index})
        row = node.model_dump()
        row["depends_on"] = valid_dependencies
        by_index[index] = row
    normalized = [
        by_index.get(
            i,
            {
                "step_index": i,
                "depends_on": [],
                "uses_problem": True,
                "verdict": "uncertain",
                "error_type": "other",
                "explanation": "Missing graph node",
            },
        )
        for i in range(step_count)
    ]
    derived = next((row["step_index"] for row in normalized if row["verdict"] == "incorrect"), -1)
    edge_count = sum(len(row["depends_on"]) + int(bool(row.get("uses_problem"))) for row in normalized)
    return {
        "nodes": normalized,
        "edges": [
            {"source": f"step_{parent}", "target": f"step_{row['step_index']}", "relation": "depends_on"}
            for row in normalized
            for parent in row["depends_on"]
        ]
        + [
            {"source": "problem", "target": f"step_{row['step_index']}", "relation": "uses_problem"}
            for row in normalized
            if row.get("uses_problem")
        ],
        "derived_first_error_step": derived,
        "node_coverage_percent": round(100 * len(by_index) / step_count, 2) if step_count else 100.0,
        "edge_count": edge_count,
        "duplicate_indices": duplicate_indices,
        "invalid_node_indices": invalid_node_indices,
        "invalid_dependencies": invalid_dependencies,
        "dependency_validity_percent": round(
            100 * (edge_count - len(invalid_dependencies)) / max(1, edge_count), 2
        ),
    }


def _run_method(
    case: dict[str, Any],
    *,
    method: str,
    model: str,
    reasoning_effort: str,
    max_output_tokens: int,
    client: Any,
) -> dict[str, Any]:
    if method in {"small_direct_step", "reference_direct_step"}:
        system, user = _direct_prompts(case)
        output_type: type[BaseModel] = DirectFirstErrorOutput
    elif method == "small_checklist_step":
        system, user = _checklist_prompts(case)
        output_type = ChecklistFirstErrorOutput
    elif method == "small_dependency_graph":
        system, user = _graph_prompts(case)
        output_type = DependencyGraphFirstErrorOutput
    else:
        raise ValueError(f"Unknown ProcessBench method: {method}")

    parsed, meta = _call_parsed(
        client,
        model=model,
        reasoning_effort=reasoning_effort,
        max_output_tokens=max_output_tokens,
        system=system,
        user=user,
        output_type=output_type,
    )
    step_count = len(case.get("steps") or [])
    declared = _normalize_prediction(getattr(parsed, "first_error_step", None), step_count)
    details: dict[str, Any] = {"raw_output": parsed.model_dump(), "declared_first_error_step": declared}
    if isinstance(parsed, ChecklistFirstErrorOutput):
        details["checklist"] = _normalize_step_reviews(parsed.step_reviews, step_count)
    if isinstance(parsed, DependencyGraphFirstErrorOutput):
        details["graph"] = _normalize_graph_nodes(parsed.nodes, step_count)
    usage = meta.get("usage") or {}
    return {
        "method": method,
        "model": model,
        "status": "ok",
        "prediction": declared,
        "details": details,
        "usage": usage,
        "api_calls": 1,
        "latency_ms": meta.get("latency_ms"),
        "response_id": meta.get("response_id"),
        "estimated_cost_usd": _estimated_cost(model, usage),
    }


def _score_prediction(gold: int, prediction: int | None) -> dict[str, Any]:
    gold_has_error = gold != -1
    pred_has_error = prediction is not None and prediction != -1
    exact = prediction == gold
    return {
        "gold_label": gold,
        "prediction": prediction,
        "valid_prediction": prediction is not None,
        "exact_correct": exact,
        "gold_has_error": gold_has_error,
        "predicted_has_error": pred_has_error,
        "error_detection_correct": gold_has_error == pred_has_error,
        "step_distance": abs(prediction - gold) if gold_has_error and pred_has_error and prediction is not None else None,
        "within_one_step": bool(gold_has_error and pred_has_error and prediction is not None and abs(prediction - gold) <= 1),
    }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            payload["_split"] = path.stem
            rows.append(payload)
    return rows


def _download_split_via_rows_api(split: str, target: Path, *, progress: Callable[[str], None]) -> dict[str, Any]:
    target.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    offset = 0
    page_size = 100
    total: int | None = None
    while total is None or offset < total:
        query = urllib.parse.urlencode(
            {
                "dataset": DATASET_NAME,
                "config": "default",
                "split": split,
                "offset": offset,
                "length": page_size,
            }
        )
        url = f"{DATASET_SERVER_ROWS}?{query}"
        request = urllib.request.Request(url, headers={"User-Agent": "verified-reasoning-graph-v034"})
        with urllib.request.urlopen(request, timeout=90) as response:  # noqa: S310 - fixed official endpoint
            payload = json.loads(response.read().decode("utf-8"))
        page = payload.get("rows") or []
        total = int(payload.get("num_rows_total") or len(page))
        if not page and offset < total:
            raise RuntimeError(f"Hugging Face rows API returned an empty page for {split} at offset {offset}")
        for item in page:
            row = item.get("row") if isinstance(item, dict) else None
            if isinstance(row, dict):
                rows.append(row)
        offset += len(page)
        progress(f"    {split}: {len(rows)}/{total}")
        if len(page) == 0:
            break
    temp = target.with_suffix(target.suffix + ".part")
    with temp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temp.replace(target)
    return {"path": str(target), "rows": len(rows), "downloaded": True, "source": "huggingface_rows_api"}


def _download_split_via_datasets(split: str, target: Path, *, progress: Callable[[str], None]) -> dict[str, Any]:
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("The optional 'datasets' package is unavailable") from exc
    progress(f"    loading {split} with datasets package fallback")
    dataset = load_dataset(DATASET_NAME, split=split)
    temp = target.with_suffix(target.suffix + ".part")
    with temp.open("w", encoding="utf-8") as handle:
        for row in dataset:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
    temp.replace(target)
    return {"path": str(target), "rows": len(dataset), "downloaded": True, "source": "datasets_package"}


def download_processbench_dataset(
    dataset_dir: Path,
    *,
    splits: list[str] | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    progress = progress or (lambda _message: None)
    selected = splits or list(PROCESSBENCH_SPLITS)
    unknown = sorted(set(selected) - set(PROCESSBENCH_SPLITS))
    if unknown:
        raise ValueError(f"Unknown ProcessBench splits: {unknown}")
    dataset_dir.mkdir(parents=True, exist_ok=True)
    status: dict[str, Any] = {}
    for split in selected:
        target = dataset_dir / f"{split}.jsonl"
        if target.exists() and target.stat().st_size > 1000:
            status[split] = {
                "path": str(target),
                "rows": sum(1 for line in target.open("r", encoding="utf-8") if line.strip()),
                "downloaded": False,
                "source": "existing_file",
            }
            continue
        progress(f"Downloading official ProcessBench split: {split}")
        try:
            status[split] = _download_split_via_rows_api(split, target, progress=progress)
        except Exception as rows_exc:  # noqa: BLE001 - fallback with clear aggregate message
            progress(f"    rows API failed: {type(rows_exc).__name__}: {rows_exc}")
            try:
                status[split] = _download_split_via_datasets(split, target, progress=progress)
            except Exception as datasets_exc:  # noqa: BLE001
                raise RuntimeError(
                    "Could not download ProcessBench. Check internet access, or install the optional datasets package "
                    "with `python -m pip install datasets` and run again. "
                    f"Rows API error: {rows_exc}; datasets fallback error: {datasets_exc}"
                ) from datasets_exc
    return status


def load_processbench_cases(
    dataset_dir: Path,
    *,
    splits: list[str],
    limit: int,
    seed: int,
    balance_correct_error: bool = True,
    max_problem_chars: int = 12_000,
    max_solution_chars: int = 28_000,
    exclude_case_ids: set[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    unknown = sorted(set(splits) - set(PROCESSBENCH_SPLITS))
    if unknown:
        raise ValueError(f"Unknown ProcessBench splits: {unknown}")
    excluded = {str(x) for x in (exclude_case_ids or set())}
    by_split_label: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    skipped = Counter()
    for split in splits:
        path = dataset_dir / f"{split}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Missing ProcessBench split: {path}. Run with --download first.")
        for raw in _load_jsonl(path):
            case_id = str(raw.get("id") or "")
            if not case_id or case_id in excluded:
                skipped["excluded_or_missing_id"] += 1
                continue
            problem = str(raw.get("problem") or "")
            steps = raw.get("steps") or []
            if not isinstance(steps, list) or not steps:
                skipped["missing_steps"] += 1
                continue
            steps = [str(step) for step in steps]
            try:
                label = int(raw.get("label"))
            except (TypeError, ValueError):
                skipped["invalid_label"] += 1
                continue
            if label != -1 and not 0 <= label < len(steps):
                skipped["label_out_of_range"] += 1
                continue
            if len(problem) > max_problem_chars or sum(len(step) for step in steps) > max_solution_chars:
                skipped["too_long"] += 1
                continue
            case = {
                "case_id": case_id,
                "split": split,
                "generator": str(raw.get("generator") or ""),
                "problem": problem,
                "steps": steps,
                "gold_label": label,
                "gold_has_error": label != -1,
                "final_answer_correct": raw.get("final_answer_correct"),
            }
            by_split_label[(split, "error" if label != -1 else "correct")].append(case)

    rng = random.Random(seed)
    for bucket in by_split_label.values():
        bucket.sort(key=lambda row: row["case_id"])
        rng.shuffle(bucket)

    if limit <= 0:
        selected = [row for bucket in by_split_label.values() for row in bucket]
        rng.shuffle(selected)
    else:
        selected = []
        if balance_correct_error:
            keys = [(split, label) for split in splits for label in ("correct", "error")]
        else:
            keys = list(by_split_label)
        base = limit // max(1, len(keys))
        remainder = limit % max(1, len(keys))
        leftovers: list[dict[str, Any]] = []
        for index, key in enumerate(keys):
            target = base + int(index < remainder)
            bucket = by_split_label.get(key, [])
            selected.extend(bucket[:target])
            leftovers.extend(bucket[target:])
        if len(selected) < limit:
            rng.shuffle(leftovers)
            selected.extend(leftovers[: limit - len(selected)])
        rng.shuffle(selected)

    counts = Counter(f"{row['split']}:{'error' if row['gold_has_error'] else 'correct'}" for row in selected)
    return selected, {
        "selected": len(selected),
        "splits": splits,
        "limit": limit,
        "seed": seed,
        "balance_correct_error": balance_correct_error,
        "by_split_and_label": dict(counts),
        "excluded_case_ids_requested": len(excluded),
        "skipped": dict(skipped),
    }


def _cache_key(
    case: dict[str, Any],
    *,
    method: str,
    model: str,
    reasoning_effort: str,
    max_output_tokens: int,
) -> str:
    return _stable_hash(
        {
            "cache_schema": CACHE_SCHEMA_VERSION,
            "prompt_version": METHOD_PROMPT_VERSIONS[method],
            "case_id": case["case_id"],
            "problem": case["problem"],
            "steps": case["steps"],
            "method": method,
            "model": model,
            "reasoning_effort": reasoning_effort,
            "max_output_tokens": max_output_tokens,
        }
    )


def _load_cache(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {"schema_version": CACHE_SCHEMA_VERSION, "methods": {}}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"schema_version": CACHE_SCHEMA_VERSION, "methods": {}}
    if payload.get("schema_version") != CACHE_SCHEMA_VERSION or not isinstance(payload.get("methods"), dict):
        return {"schema_version": CACHE_SCHEMA_VERSION, "methods": {}}
    return payload


def _save_cache(path: Path | None, cache: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    temp.replace(path)


def _harmonic(a: float, b: float) -> float:
    return 0.0 if a + b == 0 else 2 * a * b / (a + b)


def _method_summary(rows: list[dict[str, Any]], method: str) -> dict[str, Any]:
    outputs = [row["methods"].get(method) for row in rows]
    outputs = [output for output in outputs if output and output.get("status") == "ok"]
    scores = [output["scores"] for output in outputs if output.get("scores")]
    error_scores = [score for score in scores if score["gold_has_error"]]
    correct_scores = [score for score in scores if not score["gold_has_error"]]
    error_acc = statistics.mean([score["exact_correct"] for score in error_scores]) if error_scores else 0.0
    correct_acc = statistics.mean([score["exact_correct"] for score in correct_scores]) if correct_scores else 0.0
    tp = sum(score["gold_has_error"] and score["predicted_has_error"] for score in scores)
    fp = sum((not score["gold_has_error"]) and score["predicted_has_error"] for score in scores)
    fn = sum(score["gold_has_error"] and (not score["predicted_has_error"]) for score in scores)
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    detection_f1 = _harmonic(precision, recall)
    distances = [score["step_distance"] for score in error_scores if score["step_distance"] is not None]
    by_split: dict[str, Any] = {}
    for split in PROCESSBENCH_SPLITS:
        split_scores = [row["methods"][method]["scores"] for row in rows if row["split"] == split and row["methods"].get(method, {}).get("status") == "ok"]
        if not split_scores:
            continue
        split_error = [s for s in split_scores if s["gold_has_error"]]
        split_correct = [s for s in split_scores if not s["gold_has_error"]]
        ea = statistics.mean([s["exact_correct"] for s in split_error]) if split_error else 0.0
        ca = statistics.mean([s["exact_correct"] for s in split_correct]) if split_correct else 0.0
        by_split[split] = {
            "n": len(split_scores),
            "exact_accuracy_percent": round(100 * statistics.mean([s["exact_correct"] for s in split_scores]), 2),
            "error_localization_accuracy_percent": round(100 * ea, 2),
            "correct_solution_accuracy_percent": round(100 * ca, 2),
            "official_f1_percent": round(100 * _harmonic(ea, ca), 2),
        }

    usage = Counter()
    costs: list[float] = []
    latencies: list[float] = []
    cache_hits = 0
    api_calls_this_run = 0
    for output in outputs:
        for key in ("input_tokens", "output_tokens", "total_tokens"):
            usage[key] += int((output.get("usage") or {}).get(key) or 0)
        if output.get("estimated_cost_usd") is not None:
            costs.append(float(output["estimated_cost_usd"]))
        if output.get("latency_ms") is not None:
            latencies.append(float(output["latency_ms"]))
        cache_hits += int(bool((output.get("cache") or {}).get("hit")))
        api_calls_this_run += int(output.get("api_calls_this_run") or 0)

    summary: dict[str, Any] = {
        "n": len(scores),
        "n_total": len(rows),
        "n_missing": len(rows) - len(scores),
        "exact_accuracy_percent": round(100 * statistics.mean([score["exact_correct"] for score in scores]), 2) if scores else 0.0,
        "error_localization_accuracy_percent": round(100 * error_acc, 2),
        "correct_solution_accuracy_percent": round(100 * correct_acc, 2),
        "official_f1_percent": round(100 * _harmonic(error_acc, correct_acc), 2),
        "error_detection_precision_percent": round(100 * precision, 2),
        "error_detection_recall_percent": round(100 * recall, 2),
        "error_detection_f1_percent": round(100 * detection_f1, 2),
        "within_one_step_accuracy_percent": round(100 * statistics.mean([s["within_one_step"] for s in error_scores]), 2) if error_scores else 0.0,
        "mean_absolute_step_error": round(statistics.mean(distances), 3) if distances else None,
        "valid_prediction_percent": round(100 * statistics.mean([s["valid_prediction"] for s in scores]), 2) if scores else 0.0,
        "input_tokens": usage["input_tokens"],
        "output_tokens": usage["output_tokens"],
        "total_tokens": usage["total_tokens"],
        "api_calls": len(outputs),
        "api_calls_this_run": api_calls_this_run,
        "cache_hits": cache_hits,
        "mean_latency_ms": round(statistics.mean(latencies), 2) if latencies else None,
        "estimated_cost_usd": round(sum(costs), 6),
        "estimated_cost_usd_this_run": round(sum(float(o.get("estimated_cost_usd_this_run") or 0.0) for o in outputs), 6),
        "by_split": by_split,
    }
    if method == "small_checklist_step":
        coverages = [
            output.get("details", {}).get("checklist", {}).get("coverage_percent")
            for output in outputs
            if output.get("details", {}).get("checklist", {}).get("coverage_percent") is not None
        ]
        consistencies = [
            output.get("prediction") == output.get("details", {}).get("checklist", {}).get("derived_first_error_step")
            for output in outputs
        ]
        summary["mean_step_review_coverage_percent"] = round(statistics.mean(coverages), 2) if coverages else None
        summary["declared_derived_consistency_percent"] = round(100 * statistics.mean(consistencies), 2) if consistencies else None
    if method == "small_dependency_graph":
        graphs = [output.get("details", {}).get("graph", {}) for output in outputs]
        summary["mean_node_coverage_percent"] = round(statistics.mean([g.get("node_coverage_percent", 0) for g in graphs]), 2) if graphs else None
        summary["mean_edge_count"] = round(statistics.mean([g.get("edge_count", 0) for g in graphs]), 2) if graphs else None
        summary["dependency_validity_percent"] = round(statistics.mean([g.get("dependency_validity_percent", 0) for g in graphs]), 2) if graphs else None
        summary["declared_derived_consistency_percent"] = round(
            100 * statistics.mean([
                output.get("prediction") == output.get("details", {}).get("graph", {}).get("derived_first_error_step")
                for output in outputs
            ]),
            2,
        ) if outputs else None
    return summary


def _binomial_two_sided(k: int, n: int) -> float:
    if n <= 0:
        return 1.0
    lower = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    upper = sum(math.comb(n, i) for i in range(k, n + 1)) / (2 ** n)
    return min(1.0, 2 * min(lower, upper))


def _paired_comparison(
    rows: list[dict[str, Any]],
    method: str,
    *,
    reference_method: str = "small_direct_step",
    seed: int = 2034,
) -> dict[str, Any]:
    pairs: list[tuple[bool, bool]] = []
    for row in rows:
        reference = row["methods"].get(reference_method, {})
        candidate = row["methods"].get(method, {})
        if reference.get("status") != "ok" or candidate.get("status") != "ok":
            continue
        pairs.append((bool(reference["scores"]["exact_correct"]), bool(candidate["scores"]["exact_correct"])))
    improved = sum((not ref) and cand for ref, cand in pairs)
    regressed = sum(ref and (not cand) for ref, cand in pairs)
    delta = statistics.mean([cand - ref for ref, cand in pairs]) if pairs else 0.0
    rng = random.Random(seed + sum(ord(ch) for ch in method + reference_method))
    boot: list[float] = []
    if pairs:
        for _ in range(4000):
            sample = [pairs[rng.randrange(len(pairs))] for _ in range(len(pairs))]
            boot.append(statistics.mean([cand - ref for ref, cand in sample]))
        boot.sort()
        low = boot[int(0.025 * (len(boot) - 1))]
        high = boot[int(0.975 * (len(boot) - 1))]
    else:
        low = high = 0.0
    discordant = improved + regressed
    return {
        "reference": reference_method,
        "method": method,
        "n": len(pairs),
        "exact_accuracy_delta_percentage_points": round(100 * delta, 2),
        "bootstrap_95ci_percentage_points": [round(100 * low, 2), round(100 * high, 2)],
        "cases_improved": improved,
        "cases_regressed": regressed,
        "net_improved_cases": improved - regressed,
        "mcnemar_exact_p": round(_binomial_two_sided(min(improved, regressed), discordant), 6),
    }


def _report_html(result: dict[str, Any]) -> str:
    summaries = result["method_summaries"]
    method_labels = {
        "small_direct_step": "Nano direct",
        "small_checklist_step": "Nano checklist",
        "small_dependency_graph": "Nano dependency graph",
        "reference_direct_step": "Mini direct",
    }
    rows = []
    for method, summary in summaries.items():
        rows.append(
            "<tr>"
            f"<td>{html.escape(method_labels.get(method, method))}</td>"
            f"<td>{summary.get('official_f1_percent')}</td>"
            f"<td>{summary.get('error_localization_accuracy_percent')}</td>"
            f"<td>{summary.get('correct_solution_accuracy_percent')}</td>"
            f"<td>{summary.get('exact_accuracy_percent')}</td>"
            f"<td>{summary.get('total_tokens')}</td>"
            f"<td>${summary.get('estimated_cost_usd')}</td>"
            "</tr>"
        )
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>ProcessBench v034</title>
<style>body{{font-family:Arial,sans-serif;max-width:1100px;margin:32px auto;padding:0 18px;color:#202124}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #ddd;padding:9px;text-align:right}}th:first-child,td:first-child{{text-align:left}}th{{background:#f5f6f7}}code{{background:#f1f3f4;padding:2px 4px}}</style></head>
<body><h1>ProcessBench v034</h1><p>Run <code>{html.escape(result['run_id'])}</code>. Primary task: first erroneous step localization (zero-based; -1 means fully correct).</p>
<table><thead><tr><th>Method</th><th>Official F1</th><th>Error localization</th><th>Correct solution</th><th>Overall exact</th><th>Tokens</th><th>Cost</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<p>Official F1 is the harmonic mean of exact error-step accuracy and fully-correct-solution accuracy.</p></body></html>"""


def run_processbench_evaluation(
    *,
    dataset_dir: Path,
    output_root: Path,
    splits: list[str] | None = None,
    small_model: str = "gpt-5.4-nano",
    reference_model: str = "gpt-5.4-mini",
    include_reference: bool = True,
    include_checklist: bool = True,
    limit: int = 48,
    seed: int = 2034,
    balance_correct_error: bool = True,
    reasoning_effort: str = "low",
    max_output_tokens_direct: int = 1400,
    max_output_tokens_structured: int = 3200,
    max_problem_chars: int = 12_000,
    max_solution_chars: int = 28_000,
    exclude_case_ids: set[str] | None = None,
    generation_cache_path: Path | None = None,
    force_methods: list[str] | None = None,
    client: Any = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    _load_local_env()
    progress = progress or (lambda _message: None)
    selected_splits = splits or ["gsm8k", "math"]
    if reasoning_effort not in ALLOWED_REASONING_EFFORTS:
        raise ValueError("reasoning_effort must be low, medium, or high")
    methods = ["small_direct_step"]
    if include_checklist:
        methods.append("small_checklist_step")
    methods.append("small_dependency_graph")
    if include_reference:
        methods.append("reference_direct_step")
    force = set(force_methods or [])
    unknown_force = force - set(methods)
    if unknown_force:
        raise ValueError(f"Cannot force inactive methods: {sorted(unknown_force)}")

    cases, sampling = load_processbench_cases(
        dataset_dir,
        splits=selected_splits,
        limit=limit,
        seed=seed,
        balance_correct_error=balance_correct_error,
        max_problem_chars=max_problem_chars,
        max_solution_chars=max_solution_chars,
        exclude_case_ids=exclude_case_ids,
    )
    cache = _load_cache(generation_cache_path)
    active_client = client

    def get_client() -> Any:
        nonlocal active_client
        if active_client is not None:
            return active_client
        if not os.getenv("OPENAI_API_KEY", "").strip():
            raise ValueError("OPENAI_API_KEY is not configured for an uncached ProcessBench method call")
        from openai import OpenAI
        active_client = OpenAI()
        return active_client

    run_id = f"processbench_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    cache_hits = Counter()
    actual_api_calls = 0

    for case_index, case in enumerate(cases, 1):
        progress(
            f"[{case_index}/{len(cases)}] {case['case_id']} · {case['split']} · "
            f"{'error@' + str(case['gold_label']) if case['gold_has_error'] else 'fully_correct'}"
        )
        row = {"case_index": case_index, **case, "methods": {}}
        for method in methods:
            model = reference_model if method == "reference_direct_step" else small_model
            token_budget = max_output_tokens_structured if method in {"small_checklist_step", "small_dependency_graph"} else max_output_tokens_direct
            key = _cache_key(
                case,
                method=method,
                model=model,
                reasoning_effort=reasoning_effort,
                max_output_tokens=token_budget,
            )
            progress(f"    - {method}")
            if method not in force and key in cache["methods"]:
                output = copy.deepcopy(cache["methods"][key])
                output["cache"] = {"hit": True, "key": key}
                output["api_calls_this_run"] = 0
                output["estimated_cost_usd_this_run"] = 0.0
                cache_hits[method] += 1
                progress("      cache hit · no API call")
            else:
                try:
                    output = _run_method(
                        case,
                        method=method,
                        model=model,
                        reasoning_effort=reasoning_effort,
                        max_output_tokens=token_budget,
                        client=get_client(),
                    )
                    output["cache"] = {"hit": False, "key": key}
                    output["api_calls_this_run"] = 1
                    output["estimated_cost_usd_this_run"] = output.get("estimated_cost_usd") or 0.0
                    cache["methods"][key] = copy.deepcopy(output)
                    _save_cache(generation_cache_path, cache)
                    actual_api_calls += 1
                except Exception as exc:  # noqa: BLE001 - preserved as visible per-case failure
                    output = {
                        "method": method,
                        "model": model,
                        "status": "error",
                        "prediction": None,
                        "scores": None,
                        "api_calls": 1,
                        "api_calls_this_run": 1,
                        "estimated_cost_usd_this_run": 0.0,
                        "error": {"type": type(exc).__name__, "message": str(exc)},
                    }
                    actual_api_calls += 1
                    failures.append({"case_id": case["case_id"], "method": method, **output["error"]})
                    progress(f"      ERROR {type(exc).__name__}: {exc}")
            if output.get("status") == "ok":
                output["scores"] = _score_prediction(case["gold_label"], output.get("prediction"))
            row["methods"][method] = output
        rows.append(row)
        with (run_dir / "cases.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summaries = {method: _method_summary(rows, method) for method in methods}
    paired_vs_direct = {
        method: _paired_comparison(rows, method, reference_method="small_direct_step", seed=seed)
        for method in methods
        if method != "small_direct_step"
    }
    graph_vs_checklist = None
    if include_checklist:
        graph_vs_checklist = _paired_comparison(
            rows,
            "small_dependency_graph",
            reference_method="small_checklist_step",
            seed=seed,
        )
    graph_vs_reference = None
    if include_reference:
        graph_vs_reference = _paired_comparison(
            rows,
            "small_dependency_graph",
            reference_method="reference_direct_step",
            seed=seed,
        )

    result = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": {
            "name": "ProcessBench",
            "huggingface_id": DATASET_NAME,
            "official_repository": "QwenLM/ProcessBench",
            "human_first_error_annotations": True,
            "label_semantics": "zero-based earliest erroneous step; -1 means all steps correct",
        },
        "settings": {
            "small_model": small_model,
            "reference_model": reference_model if include_reference else None,
            "include_reference": include_reference,
            "include_checklist": include_checklist,
            "splits": selected_splits,
            "limit": limit,
            "seed": seed,
            "balance_correct_error": balance_correct_error,
            "reasoning_effort": reasoning_effort,
            "max_output_tokens_direct": max_output_tokens_direct,
            "max_output_tokens_structured": max_output_tokens_structured,
            "max_problem_chars": max_problem_chars,
            "max_solution_chars": max_solution_chars,
            "excluded_case_ids_count": len(exclude_case_ids or set()),
            "lightweight_graph": "one nano call; one node per supplied step plus direct dependency edges; no second audit call",
            "checklist_role": "same-model sequential no-edge ablation",
        },
        "sampling": sampling,
        "method_summaries": summaries,
        "paired_vs_small_direct": paired_vs_direct,
        "paired_graph_vs_checklist": graph_vs_checklist,
        "paired_graph_vs_reference_direct": graph_vs_reference,
        "cache_summary": {
            "generation_cache_path": str(generation_cache_path) if generation_cache_path else None,
            "cache_hits_by_method": dict(cache_hits),
            "actual_api_calls_this_run": actual_api_calls,
        },
        "summary": {
            "completed_cases": len(rows),
            "failed_method_calls": len(failures),
            "primary_method": "small_dependency_graph",
            "primary_official_f1_percent": summaries.get("small_dependency_graph", {}).get("official_f1_percent"),
            "direct_official_f1_percent": summaries.get("small_direct_step", {}).get("official_f1_percent"),
            "checklist_official_f1_percent": summaries.get("small_checklist_step", {}).get("official_f1_percent"),
            "reference_official_f1_percent": summaries.get("reference_direct_step", {}).get("official_f1_percent"),
            "graph_vs_direct_exact_delta_pp": paired_vs_direct.get("small_dependency_graph", {}).get("exact_accuracy_delta_percentage_points"),
            "graph_vs_checklist_exact_delta_pp": (graph_vs_checklist or {}).get("exact_accuracy_delta_percentage_points"),
            "actual_api_calls_this_run": actual_api_calls,
        },
        "cases": rows,
    }
    (run_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "failures.json").write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "report.html").write_text(_report_html(result), encoding="utf-8")
    (run_dir / "status.json").write_text(
        json.dumps(
            {
                "status": "completed" if not failures else "completed_with_failures",
                "run_id": run_id,
                "completed_cases": len(rows),
                "failed_method_calls": len(failures),
                "actual_api_calls_this_run": actual_api_calls,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return result

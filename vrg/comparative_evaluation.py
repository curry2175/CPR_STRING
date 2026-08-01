from __future__ import annotations

import csv
import json
import math
import os
import random
import re
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from .discussion_graph import IssueType, generate_discussion_graph
from .evaluation_suite import (
    BINARY_DATASETS,
    BenchmarkCase,
    BinaryReasoningOutput,
    _macro_f1,
    _norm,
    _parse_binary_response,
    _reasoning_graph,
    load_legalbench_yes_no_cases,
    load_proofwriter_binary_cases,
    load_pubmedqa_binary_cases,
)
from .experiment import (
    FAULT_TYPES,
    _apply_fault,
    _candidate_nodes,
    _difficulty_targets,
    _lock_baseline_chain,
    _oracle_mutation_valid,
    _overall_invalid,
    _predicted_roots,
)
from .hybrid_runner import run_hybrid_proofwriter
from .openai_runner import ALLOWED_REASONING_EFFORTS, DEFAULT_MODEL, _load_local_env, _usage_dict
from .verifier import verify_case


ANSWER_METHODS = ("direct", "self_critique", "graph", "graph_repair")
AUDIT_METHODS = ("plain_critic", "checklist_critic", "graph_verifier")
DISCUSSION_METHODS = ("plain_critic", "structured_critic", "discussion_graph")


class DirectAnswerOutput(BaseModel):
    final_answer: Literal["Yes", "No"]
    brief_reason: str = Field(default="", description="One short public reason, not hidden chain-of-thought")


class SelfCritiqueOutput(BaseModel):
    revised_answer: Literal["Yes", "No"]
    changed_answer: bool
    critique_summary: str = Field(default="", description="Concise public summary of the re-check")


class StructuralRepairOutput(BinaryReasoningOutput):
    repair_summary: str = ""


class CriticOutput(BaseModel):
    has_error: bool
    error_node_ids: list[str] = Field(default_factory=list)
    error_types: list[str] = Field(default_factory=list)
    explanation: str = ""


class DiscussionCriticIssue(BaseModel):
    issue_type: IssueType
    severity: Literal["high", "medium", "low"] = "medium"
    source_spans: list[str] = Field(default_factory=list, description="Short exact quotes from the supplied paragraph")
    explanation: str = ""


class DiscussionCriticOutput(BaseModel):
    has_problem: bool
    issues: list[DiscussionCriticIssue] = Field(default_factory=list)
    overall_summary: str = ""


def _usage_add(total: dict[str, int], usage: dict[str, Any] | None) -> None:
    usage = usage or {}
    for key in ("input_tokens", "output_tokens", "total_tokens"):
        total[key] = int(total.get(key) or 0) + int(usage.get(key) or 0)


def _csv_write(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def stratified_binary_sample(cases: list[BenchmarkCase], limit: int, *, seed: int = 2028) -> tuple[list[BenchmarkCase], dict[str, Any]]:
    """Select a reproducible Yes/No-balanced subset when both labels are available."""
    if limit <= 0 or limit >= len(cases):
        selected = list(cases)
    else:
        rng = random.Random(seed)
        yes = [case for case in cases if case.gold_answer == "Yes"]
        no = [case for case in cases if case.gold_answer == "No"]
        rng.shuffle(yes)
        rng.shuffle(no)
        target_yes = limit // 2
        target_no = limit - target_yes
        chosen_yes = yes[:target_yes]
        chosen_no = no[:target_no]
        remaining = limit - len(chosen_yes) - len(chosen_no)
        pool = yes[len(chosen_yes):] + no[len(chosen_no):]
        rng.shuffle(pool)
        selected = chosen_yes + chosen_no + pool[:remaining]
        rng.shuffle(selected)
    labels = Counter(case.gold_answer for case in selected)
    return selected, {
        "requested": limit,
        "available": len(cases),
        "selected": len(selected),
        "yes": labels.get("Yes", 0),
        "no": labels.get("No", 0),
        "balanced": labels.get("Yes", 0) > 0 and labels.get("No", 0) > 0,
    }


def load_comparison_cases(
    data_root: Path,
    *,
    datasets: list[str],
    limit_per_dataset: int,
    legal_tasks: list[str] | None,
    seed: int,
    refresh: bool,
) -> tuple[list[BenchmarkCase], dict[str, Any]]:
    pool_limit = 0 if limit_per_dataset <= 0 else max(200, limit_per_dataset * 20)
    loaders = {
        "proofwriter": lambda: load_proofwriter_binary_cases(data_root, limit=pool_limit, refresh=refresh),
        "legalbench": lambda: load_legalbench_yes_no_cases(data_root, limit=pool_limit, task_names=legal_tasks, refresh=refresh),
        "pubmedqa": lambda: load_pubmedqa_binary_cases(data_root, limit=pool_limit, refresh=refresh),
    }
    all_cases: list[BenchmarkCase] = []
    sampling: dict[str, Any] = {}
    for offset, dataset in enumerate(datasets):
        pool = loaders[dataset]()
        selected, info = stratified_binary_sample(pool, limit_per_dataset, seed=seed + offset)
        sampling[dataset] = info
        all_cases.extend(selected)
    return all_cases, sampling


def _call_parsed(client: Any, *, model: str, reasoning_effort: str, max_output_tokens: int,
                 system: str, user: str, schema: Any) -> tuple[Any, dict[str, Any]]:
    started = time.perf_counter()
    response = client.responses.parse(
        model=model,
        reasoning={"effort": reasoning_effort},
        max_output_tokens=max_output_tokens,
        store=False,
        input=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        text_format=schema,
    )
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
        text = str(getattr(response, "output_text", "") or "").strip()
        if text:
            parsed = schema.model_validate_json(text)
    if parsed is None:
        raise ValueError("Model returned no parsed output")
    if not isinstance(parsed, schema):
        parsed = schema.model_validate(parsed)
    meta = {
        "usage": _usage_dict(response),
        "latency_ms": round((time.perf_counter() - started) * 1000, 3),
        "response_id": str(getattr(response, "id", "")),
        "model_returned": str(getattr(response, "model", model)),
    }
    return parsed, meta


def _case_prompt(case: BenchmarkCase) -> str:
    return (
        f"Dataset: {case.dataset}\nTask: {case.task}\n\n"
        f"Instruction:\n{case.instruction}\n\n"
        f"Input/context:\n{case.context}\n\n"
        f"Question:\n{case.question}"
    )


def evaluate_direct(case: BenchmarkCase, *, model: str, reasoning_effort: str,
                    max_output_tokens: int, client: Any) -> dict[str, Any]:
    parsed, meta = _call_parsed(
        client,
        model=model,
        reasoning_effort=reasoning_effort,
        max_output_tokens=min(max_output_tokens, 1200),
        schema=DirectAnswerOutput,
        system=(
            "Answer the supplied binary task directly. Use only the supplied text. "
            "Return Yes or No and one brief public reason. Do not create a graph, critique loop, or use outside facts."
        ),
        user=_case_prompt(case),
    )
    return {
        "method": "direct",
        "predicted_answer": parsed.final_answer,
        "answer_correct": parsed.final_answer == case.gold_answer,
        "brief_reason": parsed.brief_reason,
        "usage": meta["usage"],
        "api_calls": 1,
        "latency_ms": meta["latency_ms"],
    }


def evaluate_self_critique(case: BenchmarkCase, direct: dict[str, Any], *, model: str,
                           reasoning_effort: str, max_output_tokens: int, client: Any) -> dict[str, Any]:
    parsed, meta = _call_parsed(
        client,
        model=model,
        reasoning_effort=reasoning_effort,
        max_output_tokens=min(max_output_tokens, 1800),
        schema=SelfCritiqueOutput,
        system=(
            "Re-check a previous Yes/No answer. Use only the supplied task text. Look for overlooked negation, scope, "
            "unsupported assumptions, or an answer that does not follow. Return a revised answer. This is free-form self-critique; do not build a graph."
        ),
        user=(
            _case_prompt(case)
            + f"\n\nPrevious answer: {direct['predicted_answer']}\nPrevious brief reason: {direct.get('brief_reason','')}"
        ),
    )
    return {
        "method": "self_critique",
        "predicted_answer": parsed.revised_answer,
        "answer_correct": parsed.revised_answer == case.gold_answer,
        "changed_answer": parsed.revised_answer != direct["predicted_answer"],
        "critique_summary": parsed.critique_summary,
        "usage": meta["usage"],
        "api_calls": 1,
        "latency_ms": meta["latency_ms"],
    }


def _proofwriter_graph_methods(case: BenchmarkCase, *, model: str, reasoning_effort: str,
                               max_output_tokens: int, client: Any) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    run = run_hybrid_proofwriter({
        "record": dict(case.raw_record),
        "model": model,
        "reasoning_effort": reasoning_effort,
        "max_output_tokens": max_output_tokens,
        "max_repair_iterations": 1,
        "repair_mode": "blind",
        "use_llm_formalizer": True,
        "use_premise_grounder": True,
        "allow_external_premises": False,
        "prefer_z3": True,
    }, client=client)
    initial = run["attempts"][0]
    final = run["attempts"][-1]
    initial_label = initial["analysis"]["classification"]["predicted_label"]
    final_label = final["analysis"]["classification"]["predicted_label"]
    label_map = {"True": "Yes", "False": "No", "Unknown": "Abstain"}
    initial_binary = label_map.get(str(initial_label), "Abstain")
    final_binary = label_map.get(str(final_label), "Abstain")

    initial_calls = [run.get("initial_generation") or {}]
    for source in (
        (run.get("formalization_preflight") or {}).get("api_call"),
        (initial.get("analysis") or {}).get("case_formalization", {}).get("api_call"),
        (initial.get("premise_grounding") or {}).get("api_call"),
    ):
        if source:
            initial_calls.append(source)
    graph_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    seen_call_ids: set[str] = set()
    initial_api_calls = 0
    for call in initial_calls:
        key = str(call.get("response_id") or "") or f"obj:{id(call)}"
        if key in seen_call_ids:
            continue
        seen_call_ids.add(key)
        initial_api_calls += 1
        _usage_add(graph_usage, call.get("usage") or {})
    graph = {
        "method": "graph",
        "predicted_answer": initial_binary,
        "answer_correct": initial_binary == case.gold_answer,
        "graph_pass": bool(initial.get("passed")),
        "verification_type": "formal_z3",
        "root_error_nodes": (initial["analysis"]["verified_graph"].get("summary") or {}).get("root_error_nodes") or [],
        "usage": graph_usage,
        "api_calls": initial_api_calls,
    }
    total_usage = dict((run.get("summary") or {}).get("total_usage") or {})
    graph_repair = {
        "method": "graph_repair",
        "predicted_answer": final_binary,
        "answer_correct": final_binary == case.gold_answer,
        "graph_pass": bool(final.get("passed")),
        "verification_type": "formal_z3",
        "repair_applied": len(run["attempts"]) > 1,
        "repair_count": max(0, len(run["attempts"]) - 1),
        "usage": total_usage,
        "api_calls": int((run.get("summary") or {}).get("api_call_count") or 1),
    }
    return graph, graph_repair, run


def _general_graph_method(case: BenchmarkCase, *, model: str, reasoning_effort: str,
                          max_output_tokens: int, client: Any) -> tuple[dict[str, Any], BinaryReasoningOutput, dict[str, Any]]:
    system = (
        "Solve the strict binary task using only the supplied text. Produce a short public justification graph, not private chain-of-thought. "
        "Each atomic step has an id, direct parent ids, and short exact source quotes. Do not invent missing evidence."
    )
    user = _case_prompt(case) + "\n\nReturn a public reasoning graph and final Yes/No answer."
    parsed, meta = _call_parsed(
        client, model=model, reasoning_effort=reasoning_effort,
        max_output_tokens=max_output_tokens, system=system, user=user, schema=BinaryReasoningOutput,
    )
    nodes, edges, diagnostics = _reasoning_graph(parsed, case.instruction + "\n" + case.context)
    method = {
        "method": "graph",
        "predicted_answer": parsed.final_answer,
        "answer_correct": parsed.final_answer == case.gold_answer,
        "graph_pass": bool(diagnostics.get("graph_pass")),
        "verification_type": "public_reasoning_structural",
        "diagnostics": diagnostics,
        "usage": meta["usage"],
        "api_calls": 1,
        "latency_ms": meta["latency_ms"],
    }
    return method, parsed, {"nodes": nodes, "edges": edges, "diagnostics": diagnostics}


def _general_graph_repair(case: BenchmarkCase, graph_method: dict[str, Any], parsed: BinaryReasoningOutput,
                          graph_data: dict[str, Any], *, model: str, reasoning_effort: str,
                          max_output_tokens: int, client: Any) -> dict[str, Any]:
    if graph_method.get("graph_pass"):
        return {
            **graph_method,
            "method": "graph_repair",
            "repair_applied": False,
            "repair_count": 0,
        }
    feedback = graph_data["diagnostics"]
    repaired, meta = _call_parsed(
        client,
        model=model,
        reasoning_effort=reasoning_effort,
        max_output_tokens=max_output_tokens,
        schema=StructuralRepairOutput,
        system=(
            "Repair a public binary justification after structural verification. Use only the supplied text. "
            "Fix invalid parent ids, unsupported root steps, missing answer support, and reconsider the final Yes/No answer. "
            "Do not assume the previous answer is correct."
        ),
        user=(
            _case_prompt(case)
            + "\n\nPrevious output:\n"
            + parsed.model_dump_json()
            + "\n\nStructural verifier feedback:\n"
            + json.dumps(feedback, ensure_ascii=False)
        ),
    )
    normalized = BinaryReasoningOutput(
        reasoning_steps=repaired.reasoning_steps,
        final_answer=repaired.final_answer,
        answer_depends_on=repaired.answer_depends_on,
    )
    _nodes, _edges, diagnostics = _reasoning_graph(normalized, case.instruction + "\n" + case.context)
    total_usage = dict(graph_method.get("usage") or {})
    _usage_add(total_usage, meta["usage"])
    return {
        "method": "graph_repair",
        "predicted_answer": repaired.final_answer,
        "answer_correct": repaired.final_answer == case.gold_answer,
        "graph_pass": bool(diagnostics.get("graph_pass")),
        "verification_type": "public_reasoning_structural",
        "repair_applied": True,
        "repair_count": 1,
        "repair_summary": repaired.repair_summary,
        "diagnostics": diagnostics,
        "usage": total_usage,
        "api_calls": 2,
        "latency_ms": float(graph_method.get("latency_ms") or 0) + float(meta["latency_ms"]),
    }


def evaluate_answer_case(case: BenchmarkCase, *, model: str, reasoning_effort: str,
                         max_output_tokens: int, client: Any) -> dict[str, Any]:
    direct = evaluate_direct(case, model=model, reasoning_effort=reasoning_effort,
                             max_output_tokens=max_output_tokens, client=client)
    self_critique = evaluate_self_critique(case, direct, model=model, reasoning_effort=reasoning_effort,
                                           max_output_tokens=max_output_tokens, client=client)
    internal: dict[str, Any] = {}
    if case.dataset == "proofwriter":
        graph, graph_repair, graph_run = _proofwriter_graph_methods(
            case, model=model, reasoning_effort=reasoning_effort,
            max_output_tokens=max_output_tokens, client=client,
        )
        internal["proofwriter_graph_run"] = graph_run
    else:
        graph, parsed, graph_data = _general_graph_method(
            case, model=model, reasoning_effort=reasoning_effort,
            max_output_tokens=max_output_tokens, client=client,
        )
        graph_repair = _general_graph_repair(
            case, graph, parsed, graph_data, model=model,
            reasoning_effort=reasoning_effort, max_output_tokens=max_output_tokens, client=client,
        )
    return {
        "dataset": case.dataset,
        "case_id": case.case_id,
        "task": case.task,
        "gold_answer": case.gold_answer,
        "methods": {
            "direct": direct,
            "self_critique": self_critique,
            "graph": graph,
            "graph_repair": graph_repair,
        },
        "metadata": case.metadata,
        "_internal": internal,
    }


def _classification_metrics(rows: list[dict[str, Any]], method: str) -> dict[str, Any]:
    flat = [
        {"gold_answer": row["gold_answer"], "predicted_answer": row["methods"][method]["predicted_answer"]}
        for row in rows if method in row.get("methods", {})
    ]
    n = len(flat)
    correct = sum(item["gold_answer"] == item["predicted_answer"] for item in flat)
    result: dict[str, Any] = {
        "n": n,
        "accuracy_percent": round(correct / n * 100, 2) if n else 0.0,
        "macro_f1_percent": _macro_f1(flat) if flat else 0.0,
    }
    for label in ("Yes", "No"):
        tp = sum(x["gold_answer"] == label and x["predicted_answer"] == label for x in flat)
        fp = sum(x["gold_answer"] != label and x["predicted_answer"] == label for x in flat)
        fn = sum(x["gold_answer"] == label and x["predicted_answer"] != label for x in flat)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        result[label.lower()] = {
            "precision_percent": round(precision * 100, 2),
            "recall_percent": round(recall * 100, 2),
            "f1_percent": round(f1 * 100, 2),
            "support": sum(x["gold_answer"] == label for x in flat),
        }
    usages = [row["methods"][method].get("usage") or {} for row in rows if method in row.get("methods", {})]
    result["input_tokens"] = sum(int(x.get("input_tokens") or 0) for x in usages)
    result["output_tokens"] = sum(int(x.get("output_tokens") or 0) for x in usages)
    result["total_tokens"] = sum(int(x.get("total_tokens") or 0) for x in usages)
    result["api_calls"] = sum(int(row["methods"][method].get("api_calls") or 0) for row in rows if method in row.get("methods", {}))
    graph_rows = [row["methods"][method] for row in rows if row["methods"][method].get("graph_pass") is not None]
    if graph_rows:
        result["graph_pass_percent"] = round(sum(bool(x.get("graph_pass")) for x in graph_rows) / len(graph_rows) * 100, 2)
    return result


def _mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(0, min(b, c) + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def _bootstrap_delta(rows: list[dict[str, Any]], method: str, *, seed: int = 2028,
                     iterations: int = 2000) -> tuple[float, float]:
    diffs = [
        int(row["methods"][method]["answer_correct"]) - int(row["methods"]["direct"]["answer_correct"])
        for row in rows
    ]
    if not diffs:
        return 0.0, 0.0
    rng = random.Random(seed)
    estimates = []
    for _ in range(iterations):
        sample = [diffs[rng.randrange(len(diffs))] for _ in diffs]
        estimates.append(statistics.mean(sample) * 100)
    estimates.sort()
    lo = estimates[int(0.025 * (len(estimates) - 1))]
    hi = estimates[int(0.975 * (len(estimates) - 1))]
    return round(lo, 2), round(hi, 2)


def paired_method_comparison(rows: list[dict[str, Any]], method: str) -> dict[str, Any]:
    direct_correct = [bool(row["methods"]["direct"]["answer_correct"]) for row in rows]
    method_correct = [bool(row["methods"][method]["answer_correct"]) for row in rows]
    corrected = sum((not d) and m for d, m in zip(direct_correct, method_correct))
    regressed = sum(d and (not m) for d, m in zip(direct_correct, method_correct))
    direct_wrong = sum(not x for x in direct_correct)
    direct_right = sum(direct_correct)
    lo, hi = _bootstrap_delta(rows, method)
    return {
        "reference": "direct",
        "method": method,
        "n": len(rows),
        "corrected_initial_errors": corrected,
        "regressed_initially_correct": regressed,
        "correction_rate_percent": round(corrected / direct_wrong * 100, 2) if direct_wrong else None,
        "regression_rate_percent": round(regressed / direct_right * 100, 2) if direct_right else None,
        "net_correction": corrected - regressed,
        "accuracy_delta_percentage_points": round((sum(method_correct) - sum(direct_correct)) / len(rows) * 100, 2) if rows else 0.0,
        "paired_bootstrap_95ci_pp": [lo, hi],
        "mcnemar_discordant_b": regressed,
        "mcnemar_discordant_c": corrected,
        "mcnemar_exact_p": round(_mcnemar_exact(regressed, corrected), 6),
    }


def summarize_answer_comparison(rows: list[dict[str, Any]]) -> dict[str, Any]:
    methods = {method: _classification_metrics(rows, method) for method in ANSWER_METHODS}
    paired = {method: paired_method_comparison(rows, method) for method in ANSWER_METHODS if method != "direct"}
    by_dataset: dict[str, Any] = {}
    for dataset in sorted({row["dataset"] for row in rows}):
        subset = [row for row in rows if row["dataset"] == dataset]
        by_dataset[dataset] = {
            "n": len(subset),
            "gold_yes": sum(row["gold_answer"] == "Yes" for row in subset),
            "gold_no": sum(row["gold_answer"] == "No" for row in subset),
            "methods": {method: _classification_metrics(subset, method) for method in ANSWER_METHODS},
            "paired_vs_direct": {method: paired_method_comparison(subset, method) for method in ANSWER_METHODS if method != "direct"},
        }
    return {"methods": methods, "paired_vs_direct": paired, "datasets": by_dataset}


def _case_as_reasoning_text(case: dict[str, Any]) -> str:
    premises = case.get("premises") or []
    premise_lines = []
    for i, premise in enumerate(premises, 1):
        if isinstance(premise, dict):
            premise_lines.append(f"{premise.get('id') or f'p{i}'}: {premise.get('text') or ''}")
        else:
            premise_lines.append(f"p{i}: {premise}")
    steps = (case.get("llm_output") or {}).get("reasoning_steps") or []
    step_lines = []
    for i, step in enumerate(steps, 1):
        if isinstance(step, dict):
            step_lines.append(f"{step.get('id') or f's{i}'}: {step.get('text') or ''} [parents: {', '.join(step.get('depends_on') or [])}]")
        else:
            step_lines.append(f"s{i}: {step}")
    output = case.get("llm_output") or {}
    return (
        "Premises:\n" + "\n".join(premise_lines)
        + f"\n\nQuestion: {case.get('question','')}"
        + "\n\nPublic reasoning:\n" + "\n".join(step_lines)
        + f"\n\nAnswer: {output.get('answer')} [parents: {', '.join(output.get('answer_depends_on') or [])}]"
    )


def _critic_call(item: dict[str, Any], *, checklist: bool, model: str, reasoning_effort: str,
                 max_output_tokens: int, client: Any) -> dict[str, Any]:
    if checklist:
        system = (
            "Audit the supplied explicit reasoning step by step. Check whether each claim follows from its declared parents, "
            "whether negation/entity/predicate/arguments are preserved, whether parents are missing or wrong, and whether the final answer follows. "
            "Return whether an error exists and the earliest erroneous node id. Do not use outside facts."
        )
    else:
        system = (
            "Review the supplied reasoning and say whether it contains a logical error. If so, identify the erroneous node id. "
            "Use only the supplied premises."
        )
    parsed, meta = _call_parsed(
        client, model=model, reasoning_effort=reasoning_effort,
        max_output_tokens=min(max_output_tokens, 1800), system=system,
        user=_case_as_reasoning_text(item["case"]), schema=CriticOutput,
    )
    return {
        "predicted_fault": bool(parsed.has_error),
        "predicted_root_nodes": sorted(set(_norm(x) for x in parsed.error_node_ids if _norm(x))),
        "error_types": parsed.error_types,
        "explanation": parsed.explanation,
        "usage": meta["usage"],
        "api_calls": 1,
    }


def _make_fault_items(proofwriter_rows: list[dict[str, Any]], *, seed: int, max_items: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    items: list[dict[str, Any]] = []
    fault_cycle = [x for x in FAULT_TYPES if x not in {"step_deletion"}]
    for row in proofwriter_rows:
        run = (row.get("_internal") or {}).get("proofwriter_graph_run")
        if not run or not run.get("attempts") or not run["attempts"][0].get("passed"):
            continue
        adapted = run["attempts"][0]["analysis"]["adapted_case"]
        initial = verify_case(adapted, prefer_z3=True, compute_counterfactuals=False)
        locked = _lock_baseline_chain(adapted, initial)
        baseline = verify_case(locked, prefer_z3=True, compute_counterfactuals=False)
        candidates = _candidate_nodes(locked, baseline)
        targets = _difficulty_targets(candidates, ("upstream", "local"))
        if not targets:
            continue
        difficulty, candidate = targets[0]
        # A clean matched control is necessary for clean false-positive rate.
        items.append({
            "audit_id": f"{row['case_id']}__clean",
            "case_id": row["case_id"], "condition": "clean", "gold_fault": False,
            "expected_root_nodes": [], "fault_type": "none", "difficulty": difficulty,
            "case": locked,
        })
        created = None
        start = rng.randrange(len(fault_cycle))
        for shift in range(len(fault_cycle)):
            fault_type = fault_cycle[(start + shift) % len(fault_cycle)]
            for attempt in range(8):
                applied = _apply_fault(locked, baseline, candidate, fault_type, len(items), attempt)
                if not applied:
                    continue
                mutated, meta = applied
                valid, oracle_note = _oracle_mutation_valid(locked, baseline, mutated, meta, fault_type)
                if valid:
                    created = {
                        "audit_id": f"{row['case_id']}__{fault_type}",
                        "case_id": row["case_id"], "condition": "fault", "gold_fault": True,
                        "expected_root_nodes": [str(meta.get("expected_error_node_id"))],
                        "fault_type": fault_type, "difficulty": difficulty,
                        "oracle_note": oracle_note, "case": mutated,
                    }
                    break
            if created:
                break
        if created:
            items.append(created)
        if max_items > 0 and len(items) >= max_items * 2:
            break
    return items


def _set_scores(predicted: set[str], expected: set[str]) -> tuple[int, int, int]:
    return len(predicted & expected), len(predicted - expected), len(expected - predicted)


def _summarize_audit_method(rows: list[dict[str, Any]], method: str) -> dict[str, Any]:
    records = [row for row in rows if method in row.get("methods", {})]
    tp = sum(row["gold_fault"] and row["methods"][method]["predicted_fault"] for row in records)
    fp = sum((not row["gold_fault"]) and row["methods"][method]["predicted_fault"] for row in records)
    fn = sum(row["gold_fault"] and (not row["methods"][method]["predicted_fault"]) for row in records)
    tn = sum((not row["gold_fault"]) and (not row["methods"][method]["predicted_fault"]) for row in records)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    loc_tp = loc_fp = loc_fn = 0
    exact = 0
    localized_faults = 0
    for row in records:
        if not row["gold_fault"]:
            continue
        expected = set(row["expected_root_nodes"])
        predicted = set(row["methods"][method].get("predicted_root_nodes") or [])
        a, b, c = _set_scores(predicted, expected)
        loc_tp += a; loc_fp += b; loc_fn += c
        if predicted:
            localized_faults += 1
        if predicted == expected:
            exact += 1
    loc_precision = loc_tp / (loc_tp + loc_fp) if loc_tp + loc_fp else 0.0
    loc_recall = loc_tp / (loc_tp + loc_fn) if loc_tp + loc_fn else 0.0
    loc_f1 = 2 * loc_precision * loc_recall / (loc_precision + loc_recall) if loc_precision + loc_recall else 0.0
    usages = [row["methods"][method].get("usage") or {} for row in records]
    clean_n = tp * 0 + sum(not row["gold_fault"] for row in records)
    fault_n = sum(row["gold_fault"] for row in records)
    return {
        "n": len(records), "fault_n": fault_n, "clean_n": clean_n,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision_percent": round(precision * 100, 2),
        "recall_percent": round(recall * 100, 2),
        "f1_percent": round(f1 * 100, 2),
        "clean_false_positive_rate_percent": round(fp / clean_n * 100, 2) if clean_n else None,
        "root_exact_localization_percent": round(exact / fault_n * 100, 2) if fault_n else None,
        "root_node_localization_f1_percent": round(loc_f1 * 100, 2),
        "input_tokens": sum(int(x.get("input_tokens") or 0) for x in usages),
        "output_tokens": sum(int(x.get("output_tokens") or 0) for x in usages),
        "api_calls": sum(int(row["methods"][method].get("api_calls") or 0) for row in records),
    }


def run_reasoning_audit_comparison(proofwriter_rows: list[dict[str, Any]], *, max_cases: int,
                                   seed: int, model: str, reasoning_effort: str,
                                   max_output_tokens: int, client: Any) -> dict[str, Any]:
    items = _make_fault_items(proofwriter_rows, seed=seed, max_items=max_cases)
    rows: list[dict[str, Any]] = []
    for item in items:
        plain = _critic_call(item, checklist=False, model=model, reasoning_effort=reasoning_effort,
                             max_output_tokens=max_output_tokens, client=client)
        checklist = _critic_call(item, checklist=True, model=model, reasoning_effort=reasoning_effort,
                                 max_output_tokens=max_output_tokens, client=client)
        verified = verify_case(item["case"], prefer_z3=True, compute_counterfactuals=False)
        graph = {
            "predicted_fault": _overall_invalid(verified),
            "predicted_root_nodes": _predicted_roots(verified),
            "error_types": [],
            "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            "api_calls": 0,
        }
        rows.append({
            **{key: value for key, value in item.items() if key != "case"},
            "methods": {"plain_critic": plain, "checklist_critic": checklist, "graph_verifier": graph},
        })
    summary = {method: _summarize_audit_method(rows, method) for method in AUDIT_METHODS}
    by_fault: dict[str, Any] = {}
    for fault_type in sorted({row["fault_type"] for row in rows if row["gold_fault"]}):
        subset = [row for row in rows if row["fault_type"] == fault_type or row["condition"] == "clean"]
        by_fault[fault_type] = {method: _summarize_audit_method(subset, method) for method in AUDIT_METHODS}
    return {"summary": summary, "by_fault_type": by_fault, "items": rows}


def _load_discussion_benchmark(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or not _norm(row.get("text")):
            raise ValueError(f"Invalid discussion benchmark line {line_no}")
        row.setdefault("gold_issue_types", [])
        row.setdefault("gold_source_spans", [])
        row.setdefault("label", "flawed" if row["gold_issue_types"] else "clean")
        rows.append(row)
    return rows


def _discussion_critic(text: str, *, structured: bool, model: str, reasoning_effort: str,
                       max_output_tokens: int, client: Any) -> dict[str, Any]:
    if structured:
        system = (
            "Audit this scientific Discussion paragraph without outside fact-checking. Systematically separate observations, study design, analysis methods, limitations, and conclusions. "
            "Check contradiction, causal overclaim, temporal logic, scope, subgroup interpretation, missing-data selection, estimand/method mismatch, noninferiority/equivalence, competing risk, multiplicity, and reproducibility. "
            "Use canonical issue_type labels and quote exact source spans. Do not create a graph."
        )
    else:
        system = (
            "Read this scientific Discussion paragraph and identify any internal logical or methodological reasoning problems. "
            "Do not use outside facts. Quote the relevant text and give canonical issue labels."
        )
    parsed, meta = _call_parsed(
        client, model=model, reasoning_effort=reasoning_effort,
        max_output_tokens=max_output_tokens, system=system, user=text, schema=DiscussionCriticOutput,
    )
    return {
        "predicted_problem": bool(parsed.has_problem or parsed.issues),
        "predicted_issue_types": sorted(set(issue.issue_type for issue in parsed.issues)),
        "source_spans": [span for issue in parsed.issues for span in issue.source_spans],
        "issues": [issue.model_dump() for issue in parsed.issues],
        "usage": meta["usage"], "api_calls": 1,
    }


def _normalize_span(value: str) -> str:
    return re.sub(r"\W+", " ", str(value or "").lower()).strip()


def _span_match(predicted: str, gold: str) -> bool:
    p = _normalize_span(predicted); g = _normalize_span(gold)
    if not p or not g:
        return False
    if p in g or g in p:
        return True
    ps = set(p.split()); gs = set(g.split())
    return len(ps & gs) / max(1, len(ps | gs)) >= 0.5


def _discussion_method_summary(rows: list[dict[str, Any]], method: str) -> dict[str, Any]:
    tp = sum(row["gold_problem"] and row["methods"][method]["predicted_problem"] for row in rows)
    fp = sum((not row["gold_problem"]) and row["methods"][method]["predicted_problem"] for row in rows)
    fn = sum(row["gold_problem"] and (not row["methods"][method]["predicted_problem"]) for row in rows)
    tn = sum((not row["gold_problem"]) and (not row["methods"][method]["predicted_problem"]) for row in rows)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    type_tp = type_fp = type_fn = 0
    localization_hits = 0; localization_total = 0
    exact_source = 0; source_count = 0
    duplicates = 0
    for row in rows:
        gold = set(row["gold_issue_types"])
        pred_list = list(row["methods"][method].get("predicted_issue_types") or [])
        pred = set(pred_list)
        a, b, c = _set_scores(pred, gold); type_tp += a; type_fp += b; type_fn += c
        duplicates += max(0, len(pred_list) - len(pred))
        spans = row["methods"][method].get("source_spans") or []
        for span in spans:
            source_count += 1
            if _normalize_span(span) in _normalize_span(row["text"]):
                exact_source += 1
        if row.get("gold_source_spans"):
            localization_total += 1
            if any(_span_match(p, g) for p in spans for g in row["gold_source_spans"]):
                localization_hits += 1
    type_precision = type_tp / (type_tp + type_fp) if type_tp + type_fp else 0.0
    type_recall = type_tp / (type_tp + type_fn) if type_tp + type_fn else 0.0
    type_f1 = 2 * type_precision * type_recall / (type_precision + type_recall) if type_precision + type_recall else 0.0
    clean_n = sum(not row["gold_problem"] for row in rows)
    usages = [row["methods"][method].get("usage") or {} for row in rows]
    return {
        "n": len(rows), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "detection_precision_percent": round(precision * 100, 2),
        "detection_recall_percent": round(recall * 100, 2),
        "detection_f1_percent": round(f1 * 100, 2),
        "clean_false_positive_rate_percent": round(fp / clean_n * 100, 2) if clean_n else None,
        "issue_type_micro_f1_percent": round(type_f1 * 100, 2),
        "source_localization_hit_percent": round(localization_hits / localization_total * 100, 2) if localization_total else None,
        "exact_source_span_rate_percent": round(exact_source / source_count * 100, 2) if source_count else None,
        "duplicate_issue_count": duplicates,
        "input_tokens": sum(int(x.get("input_tokens") or 0) for x in usages),
        "output_tokens": sum(int(x.get("output_tokens") or 0) for x in usages),
        "api_calls": sum(int(row["methods"][method].get("api_calls") or 0) for row in rows),
    }


def run_discussion_audit_comparison(benchmark_path: Path, *, model: str, reasoning_effort: str,
                                    max_output_tokens: int, client: Any) -> dict[str, Any]:
    cases = _load_discussion_benchmark(benchmark_path)
    rows = []
    for case in cases:
        plain = _discussion_critic(case["text"], structured=False, model=model,
                                   reasoning_effort=reasoning_effort, max_output_tokens=max_output_tokens, client=client)
        structured = _discussion_critic(case["text"], structured=True, model=model,
                                        reasoning_effort=reasoning_effort, max_output_tokens=max_output_tokens, client=client)
        graph_result = generate_discussion_graph(
            case["text"], model=model, reasoning_effort=reasoning_effort,
            max_output_tokens=max(3000, max_output_tokens), client=client,
        )
        graph_issues = graph_result.get("issues") or []
        graph = {
            "predicted_problem": bool(graph_issues),
            "predicted_issue_types": [str(issue.get("issue_type")) for issue in graph_issues],
            "source_spans": [
                str(node.get("source_text") or "")
                for issue in graph_issues
                for node in graph_result.get("nodes") or []
                if str(node.get("id")) in set(str(x) for x in issue.get("node_ids") or [])
            ],
            "issues": graph_issues,
            "usage": graph_result.get("usage") or {},
            "api_calls": int(graph_result.get("api_call_count") or 1),
            "graph_metrics": graph_result.get("graph_metrics") or {},
        }
        rows.append({
            "case_id": case.get("id"), "pair_id": case.get("pair_id"), "label": case.get("label"),
            "text": case["text"], "gold_problem": bool(case.get("gold_issue_types")),
            "gold_issue_types": case.get("gold_issue_types") or [],
            "gold_source_spans": case.get("gold_source_spans") or [],
            "methods": {"plain_critic": plain, "structured_critic": structured, "discussion_graph": graph},
        })
    summary = {method: _discussion_method_summary(rows, method) for method in DISCUSSION_METHODS}
    return {"summary": summary, "cases": rows}


def _answer_html(result: dict[str, Any]) -> str:
    summary = result.get("answer_comparison", {}).get("summary", {})
    method_rows = []
    for method, stats in summary.get("methods", {}).items():
        paired = summary.get("paired_vs_direct", {}).get(method, {})
        method_rows.append(
            f"<tr><td>{method}</td><td>{stats.get('n')}</td><td>{stats.get('accuracy_percent')}%</td>"
            f"<td>{stats.get('macro_f1_percent')}%</td><td>{paired.get('corrected_initial_errors','—')}</td>"
            f"<td>{paired.get('regressed_initially_correct','—')}</td><td>{paired.get('net_correction','—')}</td>"
            f"<td>{paired.get('accuracy_delta_percentage_points','—')}</td><td>{paired.get('mcnemar_exact_p','—')}</td>"
            f"<td>{stats.get('total_tokens')}</td></tr>"
        )
    audit_rows = []
    for method, stats in (result.get("reasoning_audit", {}).get("summary") or {}).items():
        audit_rows.append(
            f"<tr><td>{method}</td><td>{stats.get('precision_percent')}%</td><td>{stats.get('recall_percent')}%</td>"
            f"<td>{stats.get('f1_percent')}%</td><td>{stats.get('clean_false_positive_rate_percent')}%</td>"
            f"<td>{stats.get('root_exact_localization_percent')}%</td><td>{stats.get('root_node_localization_f1_percent')}%</td></tr>"
        )
    discussion_rows = []
    for method, stats in (result.get("discussion_audit", {}).get("summary") or {}).items():
        discussion_rows.append(
            f"<tr><td>{method}</td><td>{stats.get('detection_precision_percent')}%</td><td>{stats.get('detection_recall_percent')}%</td>"
            f"<td>{stats.get('detection_f1_percent')}%</td><td>{stats.get('clean_false_positive_rate_percent')}%</td>"
            f"<td>{stats.get('issue_type_micro_f1_percent')}%</td><td>{stats.get('source_localization_hit_percent')}%</td>"
            f"<td>{stats.get('exact_source_span_rate_percent')}%</td></tr>"
        )
    return f"""<!doctype html><meta charset='utf-8'><title>VRG Comparative Evaluation</title>
<style>body{{font-family:Arial;margin:30px;color:#172033;line-height:1.45}}table{{border-collapse:collapse;width:100%;margin:12px 0 28px}}th,td{{border:1px solid #dbe2ec;padding:8px;text-align:left}}th{{background:#f1f5f9}}.card{{border:1px solid #dbe2ec;border-radius:12px;padding:15px;margin:12px 0}}.note{{color:#475569}}</style>
<h1>VRG Comparative Evaluation</h1><div class='card'><b>Run:</b> {result.get('run_id')}<br><b>Model:</b> {result.get('settings',{}).get('model')}<br><b>Purpose:</b> Paired comparison against direct GPT and free-form critique.</div>
<h2>1. Answer and repair comparison</h2><table><thead><tr><th>Method</th><th>N</th><th>Accuracy</th><th>Macro-F1</th><th>Corrections</th><th>Regressions</th><th>Net correction</th><th>Δ accuracy (pp)</th><th>McNemar p</th><th>Tokens</th></tr></thead><tbody>{''.join(method_rows)}</tbody></table>
<p class='note'>Correction and regression are paired to the same Direct GPT output. Graph and Graph+repair share the same initial graph generation on ProofWriter.</p>
<h2>2. Reasoning-error audit</h2><table><thead><tr><th>Method</th><th>Precision</th><th>Recall</th><th>F1</th><th>Clean FP</th><th>Exact root</th><th>Node localization F1</th></tr></thead><tbody>{''.join(audit_rows)}</tbody></table>
<h2>3. Discussion audit</h2><table><thead><tr><th>Method</th><th>Precision</th><th>Recall</th><th>F1</th><th>Clean FP</th><th>Issue-type F1</th><th>Localization</th><th>Exact source spans</th></tr></thead><tbody>{''.join(discussion_rows)}</tbody></table>
<p class='note'>Do not claim superiority from graph size. The primary claims should rely on paired accuracy changes, error-detection F1, clean false positives, localization, and harmful-regression rates.</p>"""


def _write_human_review_packet(run_dir: Path, discussion: dict[str, Any], *, seed: int) -> None:
    rng = random.Random(seed)
    packet = []
    answer_key = []
    for row in discussion.get("cases") or []:
        for method in DISCUSSION_METHODS:
            output = row["methods"][method]
            record_id = f"hr_{len(packet)+1:04d}"
            packet.append({
                "record_id": record_id,
                "paragraph": row["text"],
                "audit_output": json.dumps(output.get("issues") or [], ensure_ascii=False),
                "reviewer_error_present": "",
                "reviewer_issue_types": "",
                "review_time_seconds": "",
                "usefulness_1_to_5": "",
                "confidence_1_to_5": "",
            })
            answer_key.append({
                "record_id": record_id, "case_id": row["case_id"], "condition": method,
                "gold_problem": row["gold_problem"], "gold_issue_types": ";".join(row["gold_issue_types"]),
            })
    rng.shuffle(packet)
    _csv_write(run_dir / "human_review_packet_blinded.csv", packet)
    _csv_write(run_dir / "human_review_answer_key.csv", answer_key)


def run_comparative_evaluation(
    *,
    output_root: Path,
    data_root: Path,
    discussion_benchmark: Path,
    datasets: list[str] | None = None,
    limit_per_dataset: int = 20,
    legal_tasks: list[str] | None = None,
    audit_cases: int = 20,
    run_answer_comparison: bool = True,
    run_reasoning_audit: bool = True,
    run_discussion_audit: bool = True,
    model: str | None = None,
    reasoning_effort: str = "low",
    max_output_tokens: int = 3500,
    seed: int = 2028,
    refresh_datasets: bool = False,
    client: Any = None,
) -> dict[str, Any]:
    _load_local_env()
    selected = datasets or list(BINARY_DATASETS)
    unknown = sorted(set(selected) - set(BINARY_DATASETS))
    if unknown:
        raise ValueError(f"Unsupported datasets: {unknown}")
    model = _norm(model or os.getenv("OPENAI_MODEL") or DEFAULT_MODEL)
    reasoning_effort = _norm(reasoning_effort).lower() or "low"
    if reasoning_effort not in ALLOWED_REASONING_EFFORTS:
        raise ValueError("reasoning_effort must be low, medium, or high")
    if client is None:
        if not os.getenv("OPENAI_API_KEY", "").strip():
            raise ValueError("OPENAI_API_KEY is not configured")
        from openai import OpenAI
        client = OpenAI()
    output_root.mkdir(parents=True, exist_ok=True)
    data_root.mkdir(parents=True, exist_ok=True)
    run_id = f"comparative_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    result: dict[str, Any] = {
        "schema_version": "0.28.0",
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "settings": {
            "datasets": selected, "limit_per_dataset": limit_per_dataset,
            "legal_tasks": legal_tasks or [], "audit_cases": audit_cases,
            "model": model, "reasoning_effort": reasoning_effort,
            "max_output_tokens": max_output_tokens, "seed": seed,
            "run_answer_comparison": run_answer_comparison,
            "run_reasoning_audit": run_reasoning_audit,
            "run_discussion_audit": run_discussion_audit,
        },
    }
    proofwriter_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    if run_answer_comparison:
        cases, sampling = load_comparison_cases(
            data_root, datasets=selected, limit_per_dataset=limit_per_dataset,
            legal_tasks=legal_tasks, seed=seed, refresh=refresh_datasets,
        )
        answer_rows = []
        cases_path = run_dir / "answer_cases.jsonl"; cases_path.touch()
        for index, case in enumerate(cases, 1):
            try:
                row = evaluate_answer_case(
                    case, model=model, reasoning_effort=reasoning_effort,
                    max_output_tokens=max_output_tokens, client=client,
                )
                row["case_index"] = index
                answer_rows.append(row)
                if case.dataset == "proofwriter":
                    proofwriter_rows.append(row)
                persisted = {key: value for key, value in row.items() if key != "_internal"}
                with cases_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(persisted, ensure_ascii=False) + "\n")
            except Exception as exc:
                failures.append({"stage": "answer_comparison", "dataset": case.dataset, "case_id": case.case_id,
                                 "error_type": type(exc).__name__, "error": str(exc)})
        result["answer_comparison"] = {
            "sampling": sampling,
            "summary": summarize_answer_comparison(answer_rows),
            "cases": [{key: value for key, value in row.items() if key != "_internal"} for row in answer_rows],
        }
    if run_reasoning_audit:
        if not proofwriter_rows:
            cases, _ = load_comparison_cases(
                data_root, datasets=["proofwriter"], limit_per_dataset=max(audit_cases, 10),
                legal_tasks=None, seed=seed, refresh=refresh_datasets,
            )
            for case in cases:
                try:
                    # Generate only graph arms; direct/self-critique are not needed for the audit dataset.
                    graph, graph_repair, graph_run = _proofwriter_graph_methods(
                        case, model=model, reasoning_effort=reasoning_effort,
                        max_output_tokens=max_output_tokens, client=client,
                    )
                    proofwriter_rows.append({
                        "dataset": "proofwriter", "case_id": case.case_id, "task": case.task,
                        "gold_answer": case.gold_answer,
                        "methods": {"graph": graph, "graph_repair": graph_repair},
                        "_internal": {"proofwriter_graph_run": graph_run},
                    })
                except Exception as exc:
                    failures.append({"stage": "audit_graph_generation", "dataset": "proofwriter", "case_id": case.case_id,
                                     "error_type": type(exc).__name__, "error": str(exc)})
        result["reasoning_audit"] = run_reasoning_audit_comparison(
            proofwriter_rows, max_cases=audit_cases, seed=seed, model=model,
            reasoning_effort=reasoning_effort, max_output_tokens=max_output_tokens, client=client,
        )
        with (run_dir / "reasoning_audit_items.jsonl").open("w", encoding="utf-8") as handle:
            for row in result["reasoning_audit"]["items"]:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    if run_discussion_audit:
        result["discussion_audit"] = run_discussion_audit_comparison(
            discussion_benchmark, model=model, reasoning_effort=reasoning_effort,
            max_output_tokens=max(3500, max_output_tokens), client=client,
        )
        with (run_dir / "discussion_audit_cases.jsonl").open("w", encoding="utf-8") as handle:
            for row in result["discussion_audit"]["cases"]:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        _write_human_review_packet(run_dir, result["discussion_audit"], seed=seed)
    result["failures"] = failures
    result["summary"] = {
        "answer_comparison_completed": len((result.get("answer_comparison") or {}).get("cases") or []),
        "reasoning_audit_items": len((result.get("reasoning_audit") or {}).get("items") or []),
        "discussion_audit_cases": len((result.get("discussion_audit") or {}).get("cases") or []),
        "failed_items": len(failures),
    }
    (run_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "report.html").write_text(_answer_html(result), encoding="utf-8")
    (run_dir / "failures.json").write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_root / "latest_comparative.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def list_comparative_runs(output_root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted((p for p in output_root.glob("comparative_*") if p.is_dir()), reverse=True):
        result_path = path / "result.json"
        if not result_path.exists():
            continue
        try:
            data = json.loads(result_path.read_text(encoding="utf-8"))
            rows.append({
                "run_id": data.get("run_id", path.name),
                "created_at": data.get("created_at"),
                **(data.get("summary") or {}),
            })
        except Exception:
            continue
    return rows

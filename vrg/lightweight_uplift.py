from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
import re
import statistics
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Literal

from pydantic import BaseModel, Field

from .discussion_graph import IssueType, generate_discussion_graph
from .issue_resolution import (
    SEMANTICS_VERSION,
    apply_resolution_semantics,
    contains_unsafe_claim,
    graph_from_card,
)
from .openai_runner import ALLOWED_REASONING_EFFORTS, _load_local_env, _usage_dict


SMALL_METHODS = ("small_direct", "small_checklist", "small_graph_structure", "small_graph_insight")
ALL_VARIANTS = ("base", "distance_8", "distance_16")

# Typed discussion graphs are substantially larger than the final lightweight
# audit object. v029 reused the audit output budget (normally 2,200 tokens) and
# only raised it to 3,500, which truncated distance_16 graph JSON. Keep the
# benchmark intervention unchanged and reserve a separate budget for graph
# extraction. A second budget is used only after a truncation-shaped failure.
GRAPH_OUTPUT_TOKEN_BUDGETS = (12_000, 24_000)
SCHEMA_VERSION = "0.31.0"
CACHE_SCHEMA_VERSION = "0.31.0-cache"
METHOD_PROMPT_VERSIONS = {
    "small_direct": "v029-direct",
    "small_checklist": "v029-checklist",
    "small_graph_structure": "v029-graph-structure",
    "small_graph_insight": "v031-resolution-aware-insight",
    "reference_direct": "v029-direct",
}
# These methods are generation-compatible with the v030 result supplied by the user.
# Graph insight is intentionally excluded because its prompt and auditor semantics changed.
V030_REUSABLE_METHODS = {"small_direct", "small_checklist", "small_graph_structure", "reference_direct"}

# Current defaults are deliberately explicit and user-overridable. Cost estimates are
# descriptive only; update these values if provider pricing changes.
MODEL_PRICE_USD_PER_MILLION: dict[str, tuple[float, float]] = {
    "gpt-5.4-nano": (0.20, 1.25),
    "gpt-5.4-mini": (0.75, 4.50),
    "gpt-5-nano": (0.05, 0.40),
    "gpt-5-mini": (0.25, 2.00),
}


class LightweightAuditOutput(BaseModel):
    has_problem: bool
    vulnerable_conclusion: str = Field(
        default="",
        description="Exact contiguous quote of the most vulnerable conclusion. Empty when no problem is present.",
    )
    issue_types: list[IssueType] = Field(default_factory=list)
    evidence_spans: list[str] = Field(
        default_factory=list,
        description="Short exact contiguous quotes that contradict, limit, or qualify the vulnerable conclusion.",
    )
    revised_conclusion: str = Field(
        default="",
        description="A concise safer conclusion grounded only in the supplied paragraph. Empty when no revision is needed.",
    )
    audit_summary: str = ""


def _norm(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _normalize_span(value: str) -> str:
    return re.sub(r"\W+", " ", str(value or "").lower()).strip()


def _sentences(text: str) -> list[str]:
    return [x.strip() for x in re.split(r"(?<=[.!?])\s+", _norm(text)) if x.strip()]


def _span_match(predicted: str, gold: str) -> bool:
    p = _normalize_span(predicted)
    g = _normalize_span(gold)
    if not p or not g:
        return False
    if p in g or g in p:
        return True
    ps, gs = set(p.split()), set(g.split())
    return len(ps & gs) / max(1, len(ps | gs)) >= 0.5


def _usage_add(total: dict[str, int], usage: dict[str, Any] | None) -> None:
    usage = usage or {}
    for key in ("input_tokens", "output_tokens", "total_tokens"):
        total[key] = int(total.get(key) or 0) + int(usage.get(key) or 0)


def _estimated_cost(model: str, usage: dict[str, Any]) -> float | None:
    prices = MODEL_PRICE_USD_PER_MILLION.get(model)
    if not prices:
        return None
    input_price, output_price = prices
    return round(
        int(usage.get("input_tokens") or 0) / 1_000_000 * input_price
        + int(usage.get("output_tokens") or 0) / 1_000_000 * output_price,
        8,
    )


def _has_valid_scores(row: dict[str, Any], method: str) -> bool:
    result = (row.get("methods") or {}).get(method)
    return isinstance(result, dict) and isinstance(result.get("scores"), dict)


def _looks_like_truncated_json(error: Exception) -> bool:
    message = str(error).lower()
    return any(
        marker in message
        for marker in (
            "eof while parsing",
            "unterminated string",
            "unexpected end",
            "max_output_tokens",
            "maximum output",
            "length limit",
            "max tokens",
            "incomplete response",
        )
    )


def _generate_graph_with_retry(
    text: str,
    *,
    model: str,
    reasoning_effort: str,
    custom_instruction: str,
    client: Any,
    progress: Callable[[str], None],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Extract one shared graph, retrying once only for truncated JSON."""
    last_error: Exception | None = None
    for attempt, token_budget in enumerate(GRAPH_OUTPUT_TOKEN_BUDGETS, 1):
        try:
            graph = generate_discussion_graph(
                text,
                model=model,
                reasoning_effort=reasoning_effort,
                max_output_tokens=token_budget,
                custom_instruction=custom_instruction,
                client=client,
            )
            return graph, {
                "attempts": attempt,
                "retried": attempt > 1,
                "successful_max_output_tokens": token_budget,
            }
        except Exception as exc:
            last_error = exc
            if not _looks_like_truncated_json(exc) or attempt == len(GRAPH_OUTPUT_TOKEN_BUDGETS):
                raise
            progress(
                "      graph JSON was truncated; retrying once with "
                f"max_output_tokens={GRAPH_OUTPUT_TOKEN_BUDGETS[attempt]}"
            )
    assert last_error is not None
    raise last_error


def _call_parsed(
    client: Any,
    *,
    model: str,
    reasoning_effort: str,
    max_output_tokens: int,
    system: str,
    user: str,
) -> tuple[LightweightAuditOutput, dict[str, Any]]:
    started = time.perf_counter()
    response = client.responses.parse(
        model=model,
        reasoning={"effort": reasoning_effort},
        max_output_tokens=max_output_tokens,
        store=False,
        input=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        text_format=LightweightAuditOutput,
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
            parsed = LightweightAuditOutput.model_validate_json(text)
    if parsed is None:
        raise ValueError("Model returned no parsed audit output")
    if not isinstance(parsed, LightweightAuditOutput):
        parsed = LightweightAuditOutput.model_validate(parsed)
    return parsed, {
        "usage": _usage_dict(response),
        "latency_ms": round((time.perf_counter() - started) * 1000, 3),
        "response_id": str(getattr(response, "id", "")),
        "model_returned": str(getattr(response, "model", model)),
    }


def load_uplift_benchmark(
    path: Path,
    *,
    variants: list[str] | None = None,
    limit: int = 0,
    seed: int = 2029,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    allowed = set(variants or ALL_VARIANTS)
    unknown = sorted(allowed - set(ALL_VARIANTS))
    if unknown:
        raise ValueError(f"Unknown benchmark variants: {unknown}")
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("variant") not in allowed:
            continue
        if row.get("label") not in {"clean", "flawed"}:
            raise ValueError(f"Invalid label on line {line_no}")
        rows.append(row)
    rng = random.Random(seed)
    rng.shuffle(rows)
    if limit > 0 and limit < len(rows):
        # Balance label and stress variant as far as the requested limit allows.
        groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            groups[(str(row["variant"]), str(row["label"]))].append(row)
        for group in groups.values():
            rng.shuffle(group)
        selected: list[dict[str, Any]] = []
        keys = sorted(groups)
        while len(selected) < limit and any(groups.values()):
            for key in keys:
                if groups[key] and len(selected) < limit:
                    selected.append(groups[key].pop())
        rows = selected
    counts = Counter((str(x["variant"]), str(x["label"])) for x in rows)
    return rows, {
        "selected": len(rows),
        "variants": sorted(allowed),
        "by_variant_label": {f"{k[0]}:{k[1]}": v for k, v in sorted(counts.items())},
    }



def _stable_hash(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _method_cache_key(
    case: dict[str, Any],
    *,
    method: str,
    model: str,
    reasoning_effort: str,
    max_output_tokens: int,
) -> str:
    return _stable_hash({
        "kind": "method",
        "case_id": case.get("id"),
        "text": case.get("text"),
        "method": method,
        "model": model,
        "reasoning_effort": reasoning_effort,
        "max_output_tokens": max_output_tokens,
        "prompt_version": METHOD_PROMPT_VERSIONS[method],
    })


def _graph_cache_key(case: dict[str, Any], *, model: str, reasoning_effort: str) -> str:
    return _stable_hash({
        "kind": "graph",
        "case_id": case.get("id"),
        "text": case.get("text"),
        "model": model,
        "reasoning_effort": reasoning_effort,
        # The v031 downstream semantics are local and do not invalidate the
        # already-paid typed graph extraction from v030.
        "graph_extraction_compatibility": "v027-v030-typed-discussion-card",
    })


def _empty_generation_cache() -> dict[str, Any]:
    return {"schema_version": CACHE_SCHEMA_VERSION, "methods": {}, "graphs": {}}


def _load_generation_cache(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return _empty_generation_cache()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return _empty_generation_cache()
    if not isinstance(data, dict):
        return _empty_generation_cache()
    data.setdefault("methods", {})
    data.setdefault("graphs", {})
    data["schema_version"] = CACHE_SCHEMA_VERSION
    return data


def _save_generation_cache(path: Path | None, cache: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def _load_result_payload(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) and isinstance(data.get("cases"), list) else None


def _import_seed_result(
    cache: dict[str, Any],
    path: Path,
    *,
    current_cases: dict[str, dict[str, Any]],
    small_model: str,
    reference_model: str,
    reasoning_effort: str,
    max_output_tokens: int,
) -> dict[str, int]:
    payload = _load_result_payload(path)
    stats = {"method_entries": 0, "graph_entries": 0}
    if payload is None:
        return stats
    seed_settings = payload.get("settings") or {}
    seed_small = str(seed_settings.get("small_model") or small_model)
    seed_reference = str(seed_settings.get("reference_model") or reference_model)
    seed_effort = str(seed_settings.get("reasoning_effort") or reasoning_effort)
    seed_max_tokens = int(seed_settings.get("max_output_tokens") or max_output_tokens)
    for row in payload.get("cases") or []:
        case_id = str(row.get("case_id") or "")
        case = current_cases.get(case_id)
        if not case or _norm(row.get("text")) != _norm(case.get("text")):
            continue
        methods = row.get("methods") or {}
        for method in V030_REUSABLE_METHODS:
            output = methods.get(method)
            if not isinstance(output, dict) or output.get("status") != "ok":
                continue
            model = seed_reference if method == "reference_direct" else seed_small
            key = _method_cache_key(
                case,
                method=method,
                model=model,
                reasoning_effort=seed_effort,
                max_output_tokens=seed_max_tokens,
            )
            cached = copy.deepcopy(output)
            cached.pop("scores", None)
            cached["cache_origin"] = str(path)
            cache["methods"][key] = cached
            stats["method_entries"] += 1
        graph_card = ""
        structure_output = methods.get("small_graph_structure")
        if isinstance(structure_output, dict):
            graph_card = str(structure_output.get("graph_card") or "")
        if not graph_card:
            insight_output = methods.get("small_graph_insight")
            if isinstance(insight_output, dict):
                graph_card = str(insight_output.get("graph_card") or "")
        graph = graph_from_card(graph_card, str(case.get("text") or ""))
        if graph is not None:
            graph_key = _graph_cache_key(case, model=seed_small, reasoning_effort=seed_effort)
            graph["cache_origin"] = str(path)
            cache["graphs"][graph_key] = graph
            stats["graph_entries"] += 1
    return stats


def _mark_cached_output(output: dict[str, Any], *, origin: str) -> dict[str, Any]:
    row = copy.deepcopy(output)
    row["cache"] = {"hit": True, "origin": origin}
    row["api_calls_this_run"] = 0
    row["usage_this_run"] = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    row["latency_ms_this_run"] = 0.0
    row["estimated_cost_usd_this_run"] = 0.0
    return row

def build_graph_card(graph: dict[str, Any], *, include_verified_insights: bool) -> str:
    nodes = graph.get("nodes") or []
    edges = graph.get("edges") or []
    all_issues = graph.get("issues") or []
    actionable = graph.get("actionable_issues")
    if actionable is None:
        actionable = [x for x in all_issues if x.get("actionable_defect", True)]
    resolved = graph.get("resolved_or_contextual_issues")
    if resolved is None:
        resolved = [x for x in all_issues if not x.get("actionable_defect", True)]
    lines = ["GRAPH STRUCTURE"]
    for node in nodes[:80]:
        lines.append(
            f"- [{node.get('id')}] role={node.get('role')} assertion={node.get('assertion_type')} "
            f"certainty={node.get('certainty')} quote={json.dumps(_norm(node.get('source_text')), ensure_ascii=False)}"
        )
    lines.append("RELATIONS")
    for edge in edges[:120]:
        lines.append(
            f"- {edge.get('source')} --{edge.get('relation')}--> {edge.get('target')}: {_norm(edge.get('rationale'))}"
        )
    metrics = graph.get("graph_metrics") or {}
    size = metrics.get("size") or {}
    structure = metrics.get("structure") or {}
    scores = metrics.get("scores") or {}
    # Backward compatibility with older flat metric fixtures.
    lines.append(
        "GRAPH METRICS: "
        + json.dumps(
            {
                "nodes": size.get("node_count", metrics.get("node_count")),
                "edges": size.get("edge_count", metrics.get("edge_count")),
                "max_depth": structure.get("maximum_depth", metrics.get("max_depth")),
                "max_width": structure.get("maximum_width", metrics.get("max_width")),
                "grounding": scores.get("grounding", metrics.get("grounding_score")),
                "integrity": scores.get("integrity", metrics.get("integrity_score")),
                "fidelity": scores.get("fidelity", metrics.get("fidelity_score")),
            },
            ensure_ascii=False,
        )
    )
    if include_verified_insights:
        lines.append("DETERMINISTIC / TYPED-GRAPH INSIGHTS")
        lines.append(
            "INTERPRETATION RULE: A risk pattern is not automatically an error. "
            "Only items with actionable_defect=true should be treated as reasoning problems. "
            "Acknowledged/resolved items are evidence that the paragraph may be internally safe."
        )
        if not actionable:
            lines.append("- No actionable defect was identified after checking acknowledgement and resolution.")
        node_by_id = {str(x.get("id")): x for x in nodes}
        for issue in actionable[:20]:
            related = [node_by_id.get(str(node_id), {}) for node_id in issue.get("node_ids") or []]
            quotes = [_norm(x.get("source_text")) for x in related if _norm(x.get("source_text"))]
            lines.append(
                f"- actionable_defect=true state=active type={issue.get('issue_type')} severity={issue.get('severity')} "
                f"title={_norm(issue.get('title'))}; linked_quotes={json.dumps(quotes, ensure_ascii=False)}; "
                f"why={_norm(issue.get('explanation'))}; suggested_revision={_norm(issue.get('suggested_revision'))}"
            )
        if resolved:
            lines.append("ACKNOWLEDGED / RESOLVED RISK PATTERNS (NOT ACTIONABLE DEFECTS)")
            for issue in resolved[:12]:
                lines.append(
                    f"- actionable_defect=false state={issue.get('issue_state')} type={issue.get('issue_type')}; "
                    f"reason={_norm(issue.get('resolution_reason'))}"
                )
    card = "\n".join(lines)
    return card[:24000]


def _audit_prompt(text: str, *, mode: str, graph_card: str = "") -> tuple[str, str]:
    common = (
        "Use only the supplied Discussion text and do not fact-check outside sources. "
        "When a problem exists, quote the vulnerable conclusion and limiting evidence exactly from the text, "
        "assign canonical issue labels, and write one concise safer conclusion. "
        "When the paragraph is internally safe, set has_problem=false and do not invent an issue."
    )
    if mode == "direct":
        system = "Audit this scientific Discussion paragraph for reasoning problems. " + common
    elif mode == "checklist":
        system = (
            "Audit this scientific Discussion paragraph systematically. Check causal overclaim, scope generalization, "
            "temporal conflict, subgroup-significance fallacy, attrition/selection bias, landmark/time-zero mismatch, "
            "post-treatment adjustment and estimand mismatch, collider risk, noninferiority/equivalence, competing risk, "
            "multiplicity, reproducibility, effect-magnitude inflation, and unsupported surrogate-to-clinical claims. "
            + common
        )
    elif mode == "graph_structure":
        system = (
            "You are a lightweight downstream auditor. A separate pass converted the paragraph into a typed claim graph. "
            "Use the graph structure to track which evidence, limitation, and conclusion nodes are connected, but make the final judgment yourself. "
            + common
        )
    elif mode == "graph_insight":
        system = (
            "You are a lightweight downstream auditor. A separate pass converted the paragraph into a typed claim graph and a resolution-aware deterministic auditor classified risk patterns. "
            "A methodological risk pattern is not itself an error. Treat only card items marked actionable_defect=true as candidate reasoning problems. "
            "Items marked actionable_defect=false are acknowledged, resolved, or contextual and must not by themselves trigger has_problem=true. "
            "Use quoted text to verify every active item and make the final judgment from the paragraph. "
            + common
        )
    else:
        raise ValueError(f"Unknown audit mode: {mode}")
    user = "DISCUSSION TEXT\n" + text
    if graph_card:
        user += "\n\n" + graph_card
    return system, user


def run_small_method(
    text: str,
    *,
    method: str,
    model: str,
    reasoning_effort: str,
    max_output_tokens: int,
    client: Any,
    graph: dict[str, Any] | None = None,
) -> dict[str, Any]:
    mode_map = {
        "small_direct": "direct",
        "small_checklist": "checklist",
        "small_graph_structure": "graph_structure",
        "small_graph_insight": "graph_insight",
        "reference_direct": "direct",
    }
    mode = mode_map[method]
    card = ""
    extraction_usage: dict[str, Any] = {}
    extraction_calls = 0
    extraction_latency = 0.0
    if mode in {"graph_structure", "graph_insight"}:
        if graph is None:
            raise ValueError("Graph method requires graph output")
        card = build_graph_card(graph, include_verified_insights=mode == "graph_insight")
        extraction_usage = dict(graph.get("usage") or {})
        extraction_calls = int(graph.get("api_call_count") or 1)
        extraction_latency = float(graph.get("latency_ms") or 0.0)
    system, user = _audit_prompt(text, mode=mode, graph_card=card)
    parsed, meta = _call_parsed(
        client,
        model=model,
        reasoning_effort=reasoning_effort,
        max_output_tokens=max_output_tokens,
        system=system,
        user=user,
    )
    usage = dict(extraction_usage)
    _usage_add(usage, meta["usage"])
    return {
        "method": method,
        "model": model,
        "predicted_problem": bool(parsed.has_problem),
        "vulnerable_conclusion": _norm(parsed.vulnerable_conclusion),
        "predicted_issue_types": sorted(set(str(x) for x in parsed.issue_types)),
        "evidence_spans": [_norm(x) for x in parsed.evidence_spans if _norm(x)],
        "revised_conclusion": _norm(parsed.revised_conclusion),
        "audit_summary": _norm(parsed.audit_summary),
        "usage": usage,
        "api_calls": extraction_calls + 1,
        "latency_ms": round(extraction_latency + float(meta["latency_ms"]), 3),
        "estimated_cost_usd": _estimated_cost(model, usage),
        "usage_this_run": dict(meta["usage"]),
        "api_calls_this_run": 1,
        "latency_ms_this_run": round(float(meta["latency_ms"]), 3),
        "estimated_cost_usd_this_run": _estimated_cost(model, meta["usage"]),
        "cache": {"hit": False},
        "graph_card": card if mode in {"graph_structure", "graph_insight"} else "",
    }


RISK_PATTERNS: dict[str, tuple[str, ...]] = {
    "causal_overclaim": (r"\bproves?\b", r"\bestablish(?:es|ed)?\b.*\bcaus", r"\bdirectly causes?\b", r"\bprevents?\b"),
    "necessity_violation": (r"\bnecessary\b", r"\bexclusively\b"),
    "scope_overreach": (r"\ball patients\b", r"\bentire patient population\b", r"\bnearly all\b", r"\bacross the entire\b"),
    "magnitude_inflation": (r"\blarge\b", r"\bdefinitively\b"),
    "evidence_strength_mismatch": (r"\bdefinitively\b", r"\bdemonstrates?\b", r"\bestablish(?:es|ed)?\b"),
    "unsupported_effect_heterogeneity": (r"\bonly in men\b", r"\bno benefit in women\b"),
    "subgroup_significance_fallacy": (r"\bonly in men\b", r"\bno benefit in women\b"),
    "noninferiority_interpretation_error": (r"\bequally effective\b", r"\bequivalent\b"),
    "equivalence_fallacy": (r"\bequally effective\b", r"\bequivalent\b"),
    "reproducibility_conflict": (r"\breproducible\b", r"\brobust\b"),
    "surrogate_to_clinical_overreach": (r"\btherapeutic effect\b",),
}


def _token_f1(a: str, b: str) -> float:
    ta = re.findall(r"[a-z0-9]+", a.lower())
    tb = re.findall(r"[a-z0-9]+", b.lower())
    if not ta or not tb:
        return 0.0
    ca, cb = Counter(ta), Counter(tb)
    overlap = sum((ca & cb).values())
    p = overlap / len(ta)
    r = overlap / len(tb)
    return 2 * p * r / (p + r) if p + r else 0.0


def _revision_proxy(output: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    revision = _norm(output.get("revised_conclusion"))
    if case["label"] == "clean":
        return {"applicable": False, "pass": None, "safe_similarity": None, "risky_marker_present": None, "overcorrected": None}
    # v031 is negation-aware: "does not establish a protective effect" is safe
    # even though it contains the words "establish" and "protective effect".
    risky = any(
        contains_unsafe_claim(revision, str(issue_type))
        for issue_type in case.get("gold_issue_types") or []
    )
    overcorrected = bool(re.search(r"\b(?:no conclusion can be drawn|nothing can be concluded|no inference is possible)\b", revision, flags=re.I))
    similarity = _token_f1(revision, str(case.get("gold_safe_conclusion") or ""))
    passed = bool(revision) and not risky and not overcorrected and similarity >= 0.15
    return {
        "applicable": True,
        "pass": passed,
        "safe_similarity": round(similarity, 4),
        "risky_marker_present": risky,
        "overcorrected": overcorrected,
    }


def score_method(output: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    gold_problem = case["label"] == "flawed"
    predicted_problem = bool(output.get("predicted_problem"))
    pred_types = set(output.get("predicted_issue_types") or [])
    gold_types = set(case.get("gold_issue_types") or [])
    type_hit = bool(pred_types & gold_types) if gold_problem else not pred_types
    conclusion_hit = (
        any(_span_match(output.get("vulnerable_conclusion", ""), str(case.get("gold_target_conclusion") or "")) for _ in [0])
        if gold_problem
        else not _norm(output.get("vulnerable_conclusion"))
    )
    evidence_hit = (
        any(
            _span_match(pred, gold)
            for pred in output.get("evidence_spans") or []
            for gold in case.get("gold_source_spans") or []
        )
        if gold_problem
        else True
    )
    text_norm = _normalize_span(case["text"])
    quote_fields = [output.get("vulnerable_conclusion", "")] + list(output.get("evidence_spans") or [])
    nonempty_quotes = [x for x in quote_fields if _norm(x)]
    exact_quote_rate = (
        sum(_normalize_span(x) in text_norm for x in nonempty_quotes) / len(nonempty_quotes)
        if nonempty_quotes else (1.0 if not gold_problem else 0.0)
    )
    if gold_problem:
        strict_success = predicted_problem and type_hit and conclusion_hit and evidence_hit
    else:
        strict_success = not predicted_problem
    revision = _revision_proxy(output, case)
    return {
        "gold_problem": gold_problem,
        "detection_correct": predicted_problem == gold_problem,
        "type_hit": type_hit,
        "conclusion_localized": conclusion_hit,
        "evidence_localized": evidence_hit,
        "exact_quote_rate": round(exact_quote_rate, 4),
        "strict_audit_success": strict_success,
        "revision": revision,
    }


def _set_micro(rows: list[dict[str, Any]], method: str) -> tuple[int, int, int]:
    tp = fp = fn = 0
    for row in rows:
        gold = set(row["gold_issue_types"])
        pred = set(row["methods"][method].get("predicted_issue_types") or [])
        tp += len(gold & pred)
        fp += len(pred - gold)
        fn += len(gold - pred)
    return tp, fp, fn


def summarize_method(rows: list[dict[str, Any]], method: str) -> dict[str, Any]:
    available = [row for row in rows if _has_valid_scores(row, method)]
    n = len(available)
    flawed = [row for row in available if row["label"] == "flawed"]
    clean = [row for row in available if row["label"] == "clean"]
    tp = sum(row["methods"][method]["scores"]["gold_problem"] and row["methods"][method]["predicted_problem"] for row in available)
    fp = sum((not row["methods"][method]["scores"]["gold_problem"]) and row["methods"][method]["predicted_problem"] for row in available)
    fn = sum(row["methods"][method]["scores"]["gold_problem"] and (not row["methods"][method]["predicted_problem"]) for row in available)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    type_tp, type_fp, type_fn = _set_micro(available, method)
    type_p = type_tp / (type_tp + type_fp) if type_tp + type_fp else 0.0
    type_r = type_tp / (type_tp + type_fn) if type_tp + type_fn else 0.0
    type_f1 = 2 * type_p * type_r / (type_p + type_r) if type_p + type_r else 0.0
    usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    usage_this_run = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    for row in available:
        _usage_add(usage, row["methods"][method].get("usage") or {})
        _usage_add(usage_this_run, row["methods"][method].get("usage_this_run") or {})
    costs = [row["methods"][method].get("estimated_cost_usd") for row in available]
    known_costs = [float(x) for x in costs if x is not None]
    strict_successes = sum(row["methods"][method]["scores"]["strict_audit_success"] for row in available)
    revision_rows = [row for row in flawed if row["methods"][method]["scores"]["revision"]["applicable"]]
    revision_passes = sum(bool(row["methods"][method]["scores"]["revision"]["pass"]) for row in revision_rows)
    result = {
        "n": n,
        "n_total": len(rows),
        "n_missing": len(rows) - n,
        "detection_precision_percent": round(precision * 100, 2),
        "detection_recall_percent": round(recall * 100, 2),
        "detection_f1_percent": round(f1 * 100, 2),
        "clean_false_positive_rate_percent": round(fp / len(clean) * 100, 2) if clean else None,
        "issue_type_micro_f1_percent": round(type_f1 * 100, 2),
        "conclusion_localization_percent": round(
            sum(row["methods"][method]["scores"]["conclusion_localized"] for row in flawed) / len(flawed) * 100, 2
        ) if flawed else None,
        "evidence_localization_percent": round(
            sum(row["methods"][method]["scores"]["evidence_localized"] for row in flawed) / len(flawed) * 100, 2
        ) if flawed else None,
        "strict_audit_success_percent": round(strict_successes / n * 100, 2) if n else 0.0,
        "revision_safety_proxy_percent": round(revision_passes / len(revision_rows) * 100, 2) if revision_rows else None,
        "mean_exact_quote_rate_percent": round(statistics.mean(
            row["methods"][method]["scores"]["exact_quote_rate"] for row in available
        ) * 100, 2) if available else None,
        **usage,
        "api_calls": sum(int(row["methods"][method].get("api_calls") or 0) for row in available),
        "mean_latency_ms": round(statistics.mean(float(row["methods"][method].get("latency_ms") or 0) for row in available), 2) if available else None,
        "estimated_cost_usd": round(sum(known_costs), 6) if known_costs else None,
        "this_run_input_tokens": usage_this_run["input_tokens"],
        "this_run_output_tokens": usage_this_run["output_tokens"],
        "this_run_total_tokens": usage_this_run["total_tokens"],
        "api_calls_this_run": sum(int(row["methods"][method].get("api_calls_this_run") or 0) for row in available),
        "estimated_cost_usd_this_run": round(sum(float(row["methods"][method].get("estimated_cost_usd_this_run") or 0.0) for row in available), 6),
        "cache_hits": sum(bool((row["methods"][method].get("cache") or {}).get("hit")) for row in available),
    }
    if result["estimated_cost_usd"] and strict_successes:
        result["estimated_cost_per_strict_success_usd"] = round(result["estimated_cost_usd"] / strict_successes, 6)
    else:
        result["estimated_cost_per_strict_success_usd"] = None
    by_variant: dict[str, Any] = {}
    for variant in ALL_VARIANTS:
        subset = [row for row in available if row.get("variant") == variant]
        if subset:
            by_variant[variant] = {
                "n": len(subset),
                "strict_audit_success_percent": round(
                    sum(row["methods"][method]["scores"]["strict_audit_success"] for row in subset) / len(subset) * 100, 2
                ),
                "detection_accuracy_percent": round(
                    sum(row["methods"][method]["scores"]["detection_correct"] for row in subset) / len(subset) * 100, 2
                ),
            }
    result["by_variant"] = by_variant
    return result


def _mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(0, min(b, c) + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def paired_uplift(rows: list[dict[str, Any]], method: str, *, seed: int = 2029) -> dict[str, Any]:
    paired = [
        row for row in rows
        if _has_valid_scores(row, "small_direct") and _has_valid_scores(row, method)
    ]
    direct = [bool(row["methods"]["small_direct"]["scores"]["strict_audit_success"]) for row in paired]
    target = [bool(row["methods"][method]["scores"]["strict_audit_success"]) for row in paired]
    corrected = sum((not d) and t for d, t in zip(direct, target))
    regressed = sum(d and (not t) for d, t in zip(direct, target))
    diffs = [int(t) - int(d) for d, t in zip(direct, target)]
    rng = random.Random(seed)
    estimates = []
    if diffs:
        for _ in range(2000):
            sample = [diffs[rng.randrange(len(diffs))] for _ in diffs]
            estimates.append(statistics.mean(sample) * 100)
        estimates.sort()
        lo = estimates[int(0.025 * (len(estimates) - 1))]
        hi = estimates[int(0.975 * (len(estimates) - 1))]
    else:
        lo = hi = 0.0
    return {
        "reference": "small_direct",
        "method": method,
        "n": len(paired),
        "n_total": len(rows),
        "n_missing": len(rows) - len(paired),
        "corrected_direct_failures": corrected,
        "regressed_direct_successes": regressed,
        "net_strict_success_gain": corrected - regressed,
        "strict_success_delta_percentage_points": round(statistics.mean(diffs) * 100, 2) if diffs else 0.0,
        "bootstrap_95ci_percentage_points": [round(lo, 2), round(hi, 2)],
        "mcnemar_exact_p": round(_mcnemar_exact(corrected, regressed), 6),
    }


def _report_html(result: dict[str, Any]) -> str:
    summaries = result.get("method_summaries") or {}
    pairs = result.get("paired_vs_small_direct") or {}
    rows = []
    for method, stats in summaries.items():
        paired = pairs.get(method, {})
        rows.append(
            "<tr>"
            f"<td>{method}</td><td>{stats.get('n')}</td>"
            f"<td>{stats.get('strict_audit_success_percent')}%</td>"
            f"<td>{stats.get('detection_f1_percent')}%</td>"
            f"<td>{stats.get('clean_false_positive_rate_percent')}%</td>"
            f"<td>{stats.get('issue_type_micro_f1_percent')}%</td>"
            f"<td>{stats.get('conclusion_localization_percent')}%</td>"
            f"<td>{stats.get('evidence_localization_percent')}%</td>"
            f"<td>{stats.get('revision_safety_proxy_percent')}%</td>"
            f"<td>{paired.get('strict_success_delta_percentage_points', '—')}</td>"
            f"<td>{paired.get('net_strict_success_gain', '—')}</td>"
            f"<td>{paired.get('mcnemar_exact_p', '—')}</td>"
            f"<td>{stats.get('estimated_cost_usd', '—')}</td>"
            "</tr>"
        )
    variant_rows = []
    for variant in ALL_VARIANTS:
        cells = [f"<td>{variant}</td>"]
        for method in summaries:
            cells.append(f"<td>{(summaries[method].get('by_variant') or {}).get(variant, {}).get('strict_audit_success_percent', '—')}</td>")
        variant_rows.append("<tr>" + "".join(cells) + "</tr>")
    variant_headers = "".join(f"<th>{m}</th>" for m in summaries)
    return f"""<!doctype html><meta charset='utf-8'><title>Lightweight Discussion Uplift</title>
<style>body{{font-family:Arial,sans-serif;margin:28px;color:#172033}}table{{border-collapse:collapse;width:100%;margin:12px 0 28px}}th,td{{border:1px solid #d7dee9;padding:8px;text-align:left}}th{{background:#f3f6fa}}.note{{background:#eef6ff;border:1px solid #bfdbfe;padding:12px;border-radius:10px;line-height:1.5}}code{{background:#f4f4f5;padding:2px 5px}}</style>
<h1>Lightweight-model Discussion Graph uplift · v031</h1>
<p class='note'><b>Primary comparison:</b> the same lightweight model on raw text (<code>small_direct</code>) versus the same lightweight model after typed-graph extraction and deterministic insight generation (<code>small_graph_insight</code>). Strict success requires correct clean/flawed detection; flawed cases additionally require an overlapping issue label, vulnerable-conclusion localization, and limiting-evidence localization.</p>
<p>Small model: <b>{result['settings']['small_model']}</b> · Reference: <b>{result['settings'].get('reference_model')}</b> · Cases: <b>{result['summary']['completed_cases']}</b></p>
<h2>Main outcomes</h2><table><thead><tr><th>Method</th><th>N</th><th>Strict success</th><th>Detection F1</th><th>Clean FP</th><th>Issue F1</th><th>Conclusion loc.</th><th>Evidence loc.</th><th>Revision proxy</th><th>Δ strict pp</th><th>Net gain</th><th>McNemar p</th><th>Est. cost USD</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<h2>Stress-length analysis</h2><table><thead><tr><th>Variant</th>{variant_headers}</tr></thead><tbody>{''.join(variant_rows)}</tbody></table>
<p class='note'>The revision score is a transparent heuristic proxy based on removal of issue-specific risky claims, non-vacuity, and overlap with a matched safe conclusion. Use blinded expert ratings for publication-level claims about revision quality. The synthetic stress set is a development/regression benchmark, not an external validation set. v031 separates active defects from acknowledged or resolved methodological risks.</p>"""


def run_lightweight_uplift(
    *,
    benchmark_path: Path,
    output_root: Path,
    small_model: str = "gpt-5.4-nano",
    reference_model: str = "gpt-5.4-mini",
    include_reference: bool = True,
    variants: list[str] | None = None,
    limit: int = 12,
    reasoning_effort: str = "low",
    max_output_tokens: int = 2200,
    seed: int = 2029,
    include_graph_structure_ablation: bool = True,
    reuse_result_paths: list[Path] | None = None,
    generation_cache_path: Path | None = None,
    force_methods: list[str] | None = None,
    client: Any = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run the lightweight uplift benchmark with selective generation reuse.

    Generation and scoring are separated. Cached model outputs are rescored
    locally, and only methods invalidated by prompt/model/schema changes are
    called again. In v031, v030 direct/checklist/graph-structure/reference
    generations are compatible; graph-insight is intentionally regenerated.
    """
    _load_local_env()
    progress = progress or (lambda _message: None)
    small_model = _norm(small_model)
    reference_model = _norm(reference_model)
    if reasoning_effort not in ALLOWED_REASONING_EFFORTS:
        raise ValueError("reasoning_effort must be low, medium, or high")
    if not small_model:
        raise ValueError("small_model is required")

    active_client = client

    def get_client() -> Any:
        nonlocal active_client
        if active_client is not None:
            return active_client
        if not os.getenv("OPENAI_API_KEY", "").strip():
            raise ValueError("OPENAI_API_KEY is not configured for an uncached method call")
        from openai import OpenAI
        active_client = OpenAI()
        return active_client

    cases, sampling = load_uplift_benchmark(
        benchmark_path, variants=variants, limit=limit, seed=seed
    )
    case_by_id = {str(case["id"]): case for case in cases}
    output_root.mkdir(parents=True, exist_ok=True)
    if generation_cache_path is None:
        generation_cache_path = output_root / "generation_cache_v031.json"
    cache = _load_generation_cache(generation_cache_path)
    seed_import_stats: list[dict[str, Any]] = []
    for seed_path in reuse_result_paths or []:
        stats = _import_seed_result(
            cache,
            Path(seed_path),
            current_cases=case_by_id,
            small_model=small_model,
            reference_model=reference_model,
            reasoning_effort=reasoning_effort,
            max_output_tokens=max_output_tokens,
        )
        seed_import_stats.append({"path": str(seed_path), **stats})
    _save_generation_cache(generation_cache_path, cache)

    forced = set(force_methods or [])
    unknown_forced = forced - set(SMALL_METHODS) - {"reference_direct"}
    if unknown_forced:
        raise ValueError(f"Unknown force_methods: {sorted(unknown_forced)}")

    run_id = f"lightweight_uplift_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    cases_path = run_dir / "cases.jsonl"
    cases_path.touch()
    status_path = run_dir / "status.json"
    status_path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "completed_cases": 0,
                "total_cases": len(cases),
                "small_model": small_model,
                "reference_model": reference_model if include_reference else None,
                "failures": 0,
                "cache_path": str(generation_cache_path),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    progress(f"Run folder: {run_dir}")
    progress(
        f"Cases: {len(cases)} · small={small_model} · "
        f"reference={reference_model if include_reference else 'off'}"
    )
    if seed_import_stats:
        for item in seed_import_stats:
            progress(
                "Reuse seed: "
                f"{item['path']} · methods={item['method_entries']} · graphs={item['graph_entries']}"
            )

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    methods = ["small_direct", "small_checklist"]
    if include_graph_structure_ablation:
        methods.append("small_graph_structure")
    methods.append("small_graph_insight")
    if include_reference:
        methods.append("reference_direct")

    cache_hits_by_method = Counter()
    calls_by_method = Counter()
    graph_cache_hits = 0
    graph_api_extractions = 0

    for index, case in enumerate(cases, 1):
        progress(
            f"[{index}/{len(cases)}] {case['id']} · {case['variant']} · {case['label']}"
        )
        row = {
            "case_index": index,
            "case_id": case["id"],
            "pair_id": case["pair_id"],
            "variant": case["variant"],
            "label": case["label"],
            "text": case["text"],
            "gold_issue_types": case.get("gold_issue_types") or [],
            "gold_source_spans": case.get("gold_source_spans") or [],
            "gold_target_conclusion": case.get("gold_target_conclusion") or "",
            "gold_safe_conclusion": case.get("gold_safe_conclusion") or "",
            "distance_sentences": case.get("distance_sentences"),
            "methods": {},
        }

        graph: dict[str, Any] | None = None
        graph_extraction_meta: dict[str, Any] | None = None
        graph_extraction_error: dict[str, Any] | None = None
        graph_extraction_attempted = False
        graph_key = _graph_cache_key(
            case, model=small_model, reasoning_effort=reasoning_effort
        )
        cached_graph = cache.get("graphs", {}).get(graph_key)
        if isinstance(cached_graph, dict):
            graph = apply_resolution_semantics(cached_graph, case["text"])
            graph_cache_hits += 1
            graph_extraction_attempted = True
            graph_extraction_meta = {
                "status": "reused",
                "cache_hit": True,
                "cache_origin": cached_graph.get("cache_origin", str(generation_cache_path)),
                "attempts": 0,
                "retried": False,
                "successful_max_output_tokens": None,
            }
            row["graph_extraction"] = graph_extraction_meta

        for method in methods:
            try:
                model = reference_model if method == "reference_direct" else small_model
                method_key = _method_cache_key(
                    case,
                    method=method,
                    model=model,
                    reasoning_effort=reasoning_effort,
                    max_output_tokens=max_output_tokens,
                )
                cached_output = cache.get("methods", {}).get(method_key)
                if method not in forced and isinstance(cached_output, dict):
                    progress(f"    - {method} · cache hit (no API call)")
                    output = _mark_cached_output(
                        cached_output,
                        origin=str(cached_output.get("cache_origin") or generation_cache_path),
                    )
                    output["status"] = "ok"
                    output["scores"] = score_method(output, case)
                    row["methods"][method] = output
                    cache_hits_by_method[method] += 1
                    continue

                progress(f"    - {method}")
                if method in {"small_graph_structure", "small_graph_insight"}:
                    if not graph_extraction_attempted:
                        graph_extraction_attempted = True
                        progress(
                            "      extracting one shared typed graph with the same small model"
                        )
                        try:
                            raw_graph, extraction = _generate_graph_with_retry(
                                case["text"],
                                model=small_model,
                                reasoning_effort=reasoning_effort,
                                custom_instruction=(
                                    "This graph will support a lightweight-model benchmark. Preserve exact quotes, "
                                    "distinguish a methodological risk from an actual reasoning defect, and identify "
                                    "whether limitations are acknowledged or resolved by the final conclusion."
                                ),
                                client=get_client(),
                                progress=progress,
                            )
                            graph = apply_resolution_semantics(raw_graph, case["text"])
                            graph_api_extractions += 1
                            graph_extraction_meta = {
                                "status": "ok",
                                "cache_hit": False,
                                **extraction,
                            }
                            row["graph_extraction"] = graph_extraction_meta
                            cache["graphs"][graph_key] = raw_graph
                            _save_generation_cache(generation_cache_path, cache)
                        except Exception as graph_exc:
                            graph_extraction_error = {
                                "error_type": type(graph_exc).__name__,
                                "error": str(graph_exc),
                                "stage": "shared_graph_extraction",
                            }
                            row["graph_extraction"] = {
                                "status": "error",
                                **graph_extraction_error,
                            }
                    if graph_extraction_error is not None:
                        row["methods"][method] = {
                            "method": method,
                            "status": "error",
                            "scores": None,
                            "error": graph_extraction_error,
                            "graph_extraction": graph_extraction_meta,
                        }
                        failures.append(
                            {"case_id": case["id"], "method": method, **graph_extraction_error}
                        )
                        progress(
                            f"      ERROR {graph_extraction_error['error_type']}: "
                            f"{graph_extraction_error['error']}"
                        )
                        continue
                    if graph is None:
                        raise ValueError("Graph method requires a reusable or newly extracted graph")

                output = run_small_method(
                    case["text"],
                    method=method,
                    model=model,
                    reasoning_effort=reasoning_effort,
                    max_output_tokens=max_output_tokens,
                    client=get_client(),
                    graph=graph,
                )
                output["status"] = "ok"
                output["scores"] = score_method(output, case)
                if method in {"small_graph_structure", "small_graph_insight"}:
                    output["graph_extraction"] = graph_extraction_meta
                row["methods"][method] = output
                calls_by_method[method] += int(output.get("api_calls_this_run") or 0)

                cache_copy = copy.deepcopy(output)
                cache_copy.pop("scores", None)
                cache_copy["cache_origin"] = str(run_dir / "result.json")
                cache["methods"][method_key] = cache_copy
                _save_generation_cache(generation_cache_path, cache)
            except Exception as exc:
                error = {
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "stage": "method_execution",
                }
                row["methods"][method] = {
                    "method": method,
                    "status": "error",
                    "scores": None,
                    "error": error,
                }
                failures.append({"case_id": case["id"], "method": method, **error})
                progress(f"      ERROR {type(exc).__name__}: {exc}")

        if graph is not None:
            row["graph_semantics"] = {
                "version": SEMANTICS_VERSION,
                "actionable_issue_count": len(graph.get("actionable_issues") or []),
                "resolved_or_contextual_issue_count": len(
                    graph.get("resolved_or_contextual_issues") or []
                ),
                "active_issue_types": sorted(
                    {
                        str(issue.get("issue_type"))
                        for issue in graph.get("actionable_issues") or []
                    }
                ),
            }
        rows.append(row)
        with cases_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        status_path.write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "completed_cases": index,
                    "total_cases": len(cases),
                    "current_case": case["id"],
                    "failures": len(failures),
                    "cache_hits_by_method": dict(cache_hits_by_method),
                    "api_calls_by_method": dict(calls_by_method),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    summaries = {method: summarize_method(rows, method) for method in methods}
    paired = {
        method: paired_uplift(rows, method, seed=seed)
        for method in methods
        if method != "small_direct"
    }
    graph_extraction_rows = [
        row.get("graph_extraction")
        for row in rows
        if isinstance(row.get("graph_extraction"), dict)
    ]
    graph_extraction_successes = sum(
        item.get("status") in {"ok", "reused"} for item in graph_extraction_rows
    )
    graph_extraction_retries = sum(
        bool(item.get("retried")) for item in graph_extraction_rows
    )
    graph_extraction_summary = {
        "graph_required_cases": len(graph_extraction_rows),
        "successful_cases": graph_extraction_successes,
        "failed_cases": len(graph_extraction_rows) - graph_extraction_successes,
        "api_extracted_cases_this_run": graph_api_extractions,
        "cache_reused_cases_this_run": sum(
            item.get("status") == "reused" for item in graph_extraction_rows
        ),
        "retried_cases": graph_extraction_retries,
        "success_rate_percent": round(
            graph_extraction_successes / len(graph_extraction_rows) * 100, 2
        )
        if graph_extraction_rows
        else None,
    }
    actual_api_calls = sum(
        int(method_result.get("api_calls_this_run") or 0)
        for row in rows
        for method_result in (row.get("methods") or {}).values()
        if isinstance(method_result, dict)
    ) + graph_api_extractions
    actual_cost = sum(
        float(method_result.get("estimated_cost_usd_this_run") or 0.0)
        for row in rows
        for method_result in (row.get("methods") or {}).values()
        if isinstance(method_result, dict)
    )
    result = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "settings": {
            "small_model": small_model,
            "reference_model": reference_model if include_reference else None,
            "include_reference": include_reference,
            "same_model_for_graph_extraction": True,
            "variants": variants or list(ALL_VARIANTS),
            "limit": limit,
            "reasoning_effort": reasoning_effort,
            "max_output_tokens": max_output_tokens,
            "graph_output_token_budgets": list(GRAPH_OUTPUT_TOKEN_BUDGETS),
            "shared_graph_per_case": True,
            "graph_retry_policy": "retry_once_only_on_truncated_json",
            "resolution_semantics_version": SEMANTICS_VERSION,
            "generation_cache_path": str(generation_cache_path),
            "reuse_result_paths": [str(x) for x in reuse_result_paths or []],
            "force_methods": sorted(forced),
            "seed": seed,
            "include_graph_structure_ablation": include_graph_structure_ablation,
            "pricing_assumptions_usd_per_million": {
                model: {"input": prices[0], "output": prices[1]}
                for model, prices in MODEL_PRICE_USD_PER_MILLION.items()
                if model in {small_model, reference_model}
            },
        },
        "sampling": sampling,
        "cache_summary": {
            "seed_imports": seed_import_stats,
            "cache_hits_by_method": dict(cache_hits_by_method),
            "api_calls_by_method_this_run": dict(calls_by_method),
            "graph_cache_hits": graph_cache_hits,
            "actual_api_calls_this_run": actual_api_calls,
            "estimated_cost_usd_this_run": round(actual_cost, 6),
        },
        "method_summaries": summaries,
        "paired_vs_small_direct": paired,
        "graph_extraction_summary": graph_extraction_summary,
        "summary": {
            "completed_cases": len(rows),
            "failed_method_calls": len(failures),
            "primary_method": "small_graph_insight",
            "primary_strict_uplift_pp": paired.get("small_graph_insight", {}).get(
                "strict_success_delta_percentage_points"
            ),
            "primary_net_gain": paired.get("small_graph_insight", {}).get(
                "net_strict_success_gain"
            ),
            "primary_mcnemar_p": paired.get("small_graph_insight", {}).get(
                "mcnemar_exact_p"
            ),
            "actual_api_calls_this_run": actual_api_calls,
            "estimated_cost_usd_this_run": round(actual_cost, 6),
        },
        "cases": rows,
        "failures": failures,
    }
    (run_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (run_dir / "report.html").write_text(_report_html(result), encoding="utf-8")
    (run_dir / "failures.json").write_text(
        json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    status_path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "completed": True,
                **result["summary"],
                "report": str(run_dir / "report.html"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (output_root / "latest_lightweight_uplift.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    progress(f"Completed. Report: {run_dir / 'report.html'}")
    return result


def list_lightweight_uplift_runs(output_root: Path) -> list[dict[str, Any]]:
    rows = []
    if not output_root.exists():
        return rows
    for path in sorted((p for p in output_root.glob("lightweight_uplift_*") if p.is_dir()), reverse=True):
        result_path = path / "result.json"
        status_path = path / "status.json"
        source = result_path if result_path.exists() else status_path
        if not source.exists():
            continue
        try:
            data = json.loads(source.read_text(encoding="utf-8"))
        except Exception:
            continue
        rows.append({
            "run_id": data.get("run_id", path.name),
            **(data.get("summary") or {}),
            "completed_cases": (data.get("summary") or {}).get("completed_cases", data.get("completed_cases", 0)),
            "completed": bool(result_path.exists()),
        })
    return rows

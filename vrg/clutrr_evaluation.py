from __future__ import annotations

import ast
import copy
import hashlib
import html
import json
import math
import random
import statistics
import time
import urllib.parse
import urllib.request
from collections import Counter, defaultdict, deque
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, Field, ValidationError

from .openai_runner import _load_local_env, _usage_dict


SCHEMA_VERSION = "0.35.1"
CACHE_SCHEMA_VERSION = "0.35.0-clutrr-cache"
DATASET_NAME = "CLUTRR/v1"
DATASET_SERVER_ROWS = "https://datasets-server.huggingface.co/rows"
DEFAULT_CONFIG = "gen_train23_test2to10"
DEFAULT_SPLIT = "test"
RELATION_LABELS = (
    "aunt",
    "son-in-law",
    "grandfather",
    "brother",
    "sister",
    "father",
    "mother",
    "grandmother",
    "uncle",
    "daughter-in-law",
    "grandson",
    "granddaughter",
    "father-in-law",
    "mother-in-law",
    "nephew",
    "son",
    "daughter",
    "niece",
    "husband",
    "wife",
    "sister-in-law",
)
METHOD_PROMPT_VERSIONS = {
    "small_direct_relation": "v035-clutrr-direct",
    "small_text_structure": "v035-clutrr-text-structure-no-graph",
    "small_explicit_graph": "v0351-clutrr-explicit-graph-compact-retry",
    "small_text_replay": "v035-clutrr-text-replay",
    "small_graph_replay": "v0351-clutrr-graph-replay-compact-retry",
    "small_shuffled_graph_replay": "v0351-clutrr-shuffled-graph-replay-compact-retry",
    "reference_direct_relation": "v035-clutrr-direct",
}
MODEL_PRICE_USD_PER_MILLION: dict[str, tuple[float, float]] = {
    "gpt-5.4-nano": (0.20, 1.25),
    "gpt-5.4-mini": (0.75, 4.50),
    "gpt-5-nano": (0.05, 0.40),
    "gpt-5-mini": (0.25, 2.00),
}


class RelationAnswerOutput(BaseModel):
    answer_relation: str = Field(description="One allowed CLUTRR relation label")
    explanation: str = ""


class RelationReplayOutput(BaseModel):
    """Minimal replay schema to prevent long-hop answers from wasting tokens on prose."""

    answer_relation: str = Field(description="One allowed CLUTRR relation label")


class TextStructureOutput(BaseModel):
    direct_fact_notes: list[str] = Field(
        default_factory=list,
        description="Plain-language direct family facts extracted from the story; no node-edge notation",
    )
    composition_plan: list[str] = Field(
        default_factory=list,
        description="Plain-language instructions for combining the facts; avoid graph/table notation",
    )
    answer_relation: str
    explanation: str = ""


class KinshipEdge(BaseModel):
    source: str
    relation: str
    target: str


class GraphStructureOutput(BaseModel):
    nodes: list[str] = Field(default_factory=list)
    edges: list[KinshipEdge] = Field(default_factory=list)
    query_path_nodes: list[str] = Field(default_factory=list)
    query_path_relations: list[str] = Field(default_factory=list)
    answer_relation: str
    explanation: str = ""


class StructuredOutputFailure(RuntimeError):
    def __init__(self, message: str, *, api_calls: int):
        super().__init__(message)
        self.api_calls = api_calls


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


def _is_retryable_structured_error(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    markers = (
        "no parsed structured output",
        "eof while parsing",
        "invalid json",
        "json_invalid",
        "output was incomplete",
        "incomplete output",
    )
    return isinstance(exc, (ValueError, ValidationError, json.JSONDecodeError)) or any(marker in text for marker in markers)


def _merge_usage_dicts(usages: list[dict[str, Any]]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for key in ("input_tokens", "output_tokens", "total_tokens"):
        merged[key] = sum(int((usage or {}).get(key) or 0) for usage in usages)
    return merged


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
    """Call structured output with one compact retry when JSON is truncated or absent."""

    started = time.perf_counter()
    usages: list[dict[str, Any]] = []
    response_ids: list[str] = []
    budgets = [max_output_tokens, max(max_output_tokens * 2, max_output_tokens + 1600)]
    last_error: Exception | None = None
    attempts_made = 0
    for attempt_index, budget in enumerate(budgets):
        compact_suffix = ""
        if attempt_index:
            compact_suffix = (
                "\n\nRETRY REQUIREMENT: Return only the required structured fields. "
                "Use the shortest possible values, omit all optional explanation, and do not repeat the input."
            )
        try:
            attempts_made += 1
            response = client.responses.parse(
                model=model,
                reasoning={"effort": reasoning_effort},
                max_output_tokens=budget,
                store=False,
                input=[
                    {"role": "system", "content": system + compact_suffix},
                    {"role": "user", "content": user},
                ],
                text_format=output_type,
            )
            usages.append(_usage_dict(response))
            response_ids.append(str(getattr(response, "id", "")))
            parsed = _parse_output(response, output_type)
            return parsed, {
                "usage": _merge_usage_dicts(usages),
                "latency_ms": round((time.perf_counter() - started) * 1000, 3),
                "response_id": response_ids[-1],
                "response_ids": response_ids,
                "model_returned": str(getattr(response, "model", model)),
                "api_calls": attempts_made,
                "retry_count": attempt_index,
                "max_output_tokens_used": budget,
            }
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            # SDK parse failures can occur before a response object is available.
            if attempt_index == 0 and _is_retryable_structured_error(exc):
                continue
            if _is_retryable_structured_error(exc):
                raise StructuredOutputFailure(str(exc), api_calls=attempts_made) from exc
            raise
    assert last_error is not None
    raise StructuredOutputFailure(str(last_error), api_calls=attempts_made) from last_error


def _literal(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return default
        try:
            return ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return value
    return value


def _normalize_relation(value: Any) -> str | None:
    text = str(value or "").strip().lower().replace("_", "-")
    text = "-".join(text.split())
    aliases = {
        "grand-daughter": "granddaughter",
        "grand-son": "grandson",
        "grand-father": "grandfather",
        "grand-mother": "grandmother",
        "soninlaw": "son-in-law",
        "daughterinlaw": "daughter-in-law",
        "fatherinlaw": "father-in-law",
        "motherinlaw": "mother-in-law",
        "sisterinlaw": "sister-in-law",
    }
    text = aliases.get(text, text)
    return text if text in RELATION_LABELS else None


def _normalize_query(value: Any) -> tuple[str, str] | None:
    parsed = _literal(value, None)
    if isinstance(parsed, (list, tuple)) and len(parsed) == 2:
        return str(parsed[0]), str(parsed[1])
    return None


def _normalize_pairs(value: Any) -> list[tuple[int, int]]:
    parsed = _literal(value, [])
    result: list[tuple[int, int]] = []
    if isinstance(parsed, list):
        for item in parsed:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                try:
                    result.append((int(item[0]), int(item[1])))
                except (TypeError, ValueError):
                    continue
    return result


def _normalize_list(value: Any) -> list[str]:
    parsed = _literal(value, [])
    if isinstance(parsed, (list, tuple)):
        return [str(item) for item in parsed]
    if isinstance(parsed, str) and parsed:
        return [part.strip() for part in parsed.split(",") if part.strip()]
    return []


def _normalize_name(value: Any) -> str:
    text = str(value or "").strip()
    while len(text) >= 2 and ((text[0], text[-1]) in {("[", "]"), ("'", "'"), ('"', '"')}):
        text = text[1:-1].strip()
    return text


def _entity_names(genders: Any) -> list[str]:
    text = str(genders or "")
    names: list[str] = []
    for chunk in text.split(","):
        name = _normalize_name(chunk.split(":", 1)[0])
        if name and name not in names:
            names.append(name)
    return names


def _task_hop(task_name: Any) -> int | None:
    text = str(task_name or "")
    try:
        return int(text.rsplit(".", 1)[1])
    except (IndexError, ValueError):
        return None


def _task_family(task_name: Any) -> int | None:
    text = str(task_name or "")
    try:
        return int(text.split("_", 1)[1].split(".", 1)[0])
    except (IndexError, ValueError):
        return None


def _gold_graph(raw: dict[str, Any]) -> dict[str, Any]:
    names = _entity_names(raw.get("genders"))
    pairs = _normalize_pairs(raw.get("story_edges"))
    relations = [_normalize_relation(x) for x in _normalize_list(raw.get("edge_types"))]
    edges: list[dict[str, str]] = []
    for index, pair in enumerate(pairs):
        relation = relations[index] if index < len(relations) else None
        source_index, target_index = pair
        if relation is None or not (0 <= source_index < len(names) and 0 <= target_index < len(names)):
            continue
        edges.append({"source": names[source_index], "relation": relation, "target": names[target_index]})
    return {
        "nodes": names,
        "edges": edges,
        "query_path_relations": [relation for relation in relations if relation is not None],
    }


def _render_case(case: dict[str, Any]) -> str:
    first, second = case["query"]
    labels = ", ".join(RELATION_LABELS)
    return (
        f"FAMILY STORY\n{case['story']}\n\n"
        f"QUERY PAIR\n({first}, {second})\n\n"
        f"QUESTION\nWhat is {second}'s family relationship to {first}?\n\n"
        f"ALLOWED LABELS\n{labels}"
    )


def _direct_prompts(case: dict[str, Any]) -> tuple[str, str]:
    system = (
        "You solve CLUTRR kinship-relation questions. Read the family story carefully and infer the relation of the "
        "second query person to the first query person. Compose all necessary family relations, ignore irrelevant "
        "wording, and choose exactly one label from the allowed list. Do not output a reciprocal relation in the wrong "
        "direction. Keep the explanation concise."
    )
    return system, _render_case(case)


def _text_structure_prompts(case: dict[str, Any]) -> tuple[str, str]:
    system = (
        "You solve CLUTRR by carefully structuring the story in plain text, but you must not use graph notation, edge "
        "lists, adjacency lists, arrows, JSON triples, or tables. First record each directly stated family fact as a "
        "short natural-language sentence. Then write a plain-language composition plan describing which numbered facts "
        "must be combined. Keep direct_fact_notes limited to explicit story facts and do not insert inferred shortcut "
        "relations. Finally choose the relation of the second query person to the first from the allowed labels."
    )
    return system, _render_case(case) + "\n\nUse plain textual structure only; no graph or table representation."


def _graph_structure_prompts(case: dict[str, Any]) -> tuple[str, str]:
    system = (
        "You solve CLUTRR by constructing an explicit entity-relation graph. Include each named person once as a node. "
        "For every directly stated family fact, add a directed edge source --relation--> target, where target is the "
        "source person's relation label: for example, April --mother--> Lillian means Lillian is April's mother. Do not "
        "add inferred shortcut edges. Then identify the ordered query path from the first query person to the second and "
        "list the edge relations along that path. Finally compose the path and choose exactly one allowed answer label. "
        "Preserve direction carefully and use only names from the story. Keep the output compact: omit prose "
        "explanation, use each direct edge once, and include only the query path needed for composition."
    )
    return system, _render_case(case) + "\n\nReturn the explicit graph, query path, and answer."


def _text_replay_prompts(case: dict[str, Any], representation: dict[str, Any]) -> tuple[str, str]:
    first, second = case["query"]
    system = (
        "You are given only a plain-text structural summary extracted from a CLUTRR story; the original story is hidden. "
        "Use only these notes to infer the relation of the second query person to the first. Choose exactly one allowed "
        "label. Do not assume missing family facts."
    )
    user = (
        f"QUERY: What is {second}'s relationship to {first}?\n\n"
        f"DIRECT FACT NOTES\n" + "\n".join(f"- {x}" for x in representation.get("direct_fact_notes", []))
        + "\n\nCOMPOSITION PLAN\n"
        + "\n".join(f"- {x}" for x in representation.get("composition_plan", []))
        + "\n\nALLOWED LABELS\n"
        + ", ".join(RELATION_LABELS)
    )
    return system, user


def _graph_replay_prompts(case: dict[str, Any], representation: dict[str, Any]) -> tuple[str, str]:
    first, second = case["query"]
    system = (
        "You are given only an extracted CLUTRR entity-relation graph; the original story is hidden. Each edge "
        "source --relation--> target means target is source's stated relation. Infer the relation of the second query "
        "person to the first by traversing and composing the graph. Choose exactly one allowed label and return no "
        "explanation or restatement."
    )
    edges = representation.get("edges") or []
    edge_lines = [f"{e['source']} --{e['relation']}--> {e['target']}" for e in edges]
    user = (
        f"QUERY: What is {second}'s relationship to {first}?\n\n"
        f"NODES\n{', '.join(representation.get('nodes') or [])}\n\n"
        f"EDGES\n" + "\n".join(f"- {line}" for line in edge_lines)
        + "\n\nALLOWED LABELS\n"
        + ", ".join(RELATION_LABELS)
    )
    return system, user


def _normalize_graph_output(parsed: GraphStructureOutput) -> dict[str, Any]:
    nodes: list[str] = []
    for value in parsed.nodes:
        name = _normalize_name(value)
        if name and name not in nodes:
            nodes.append(name)
    edges: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    invalid_relations: list[str] = []
    for edge in parsed.edges:
        relation = _normalize_relation(edge.relation)
        source = _normalize_name(edge.source)
        target = _normalize_name(edge.target)
        if relation is None:
            invalid_relations.append(str(edge.relation))
            continue
        key = (source, relation, target)
        if source and target and key not in seen:
            seen.add(key)
            edges.append({"source": source, "relation": relation, "target": target})
    path_nodes = [_normalize_name(x) for x in parsed.query_path_nodes if _normalize_name(x)]
    path_relations = [relation for relation in (_normalize_relation(x) for x in parsed.query_path_relations) if relation]
    endpoints_in_nodes = sum(int(e["source"] in nodes and e["target"] in nodes) for e in edges)
    path_shape_valid = len(path_nodes) == len(path_relations) + 1 if path_nodes else False
    return {
        "nodes": nodes,
        "edges": edges,
        "query_path_nodes": path_nodes,
        "query_path_relations": path_relations,
        "invalid_relations": invalid_relations,
        "endpoint_validity_percent": round(100 * endpoints_in_nodes / len(edges), 2) if edges else 0.0,
        "path_shape_valid": path_shape_valid,
    }


def _shuffle_graph(graph: dict[str, Any], seed: int, case_id: str) -> dict[str, Any]:
    shuffled = copy.deepcopy(graph)
    edges = shuffled.get("edges") or []
    if len(edges) < 2:
        shuffled["shuffle_changed_edge_count"] = 0
        return shuffled
    rng = random.Random(seed + sum(ord(ch) for ch in case_id))
    original = [(e["source"], e["relation"], e["target"]) for e in edges]
    best = copy.deepcopy(edges)
    best_changes = 0
    for _ in range(40):
        targets = [e["target"] for e in edges]
        relations = [e["relation"] for e in edges]
        rng.shuffle(targets)
        rng.shuffle(relations)
        candidate = []
        for index, edge in enumerate(edges):
            candidate.append({"source": edge["source"], "relation": relations[index], "target": targets[index]})
        changes = sum(tuple((e["source"], e["relation"], e["target"])) != original[i] for i, e in enumerate(candidate))
        if changes > best_changes:
            best, best_changes = candidate, changes
        if changes == len(edges):
            break
    shuffled["edges"] = best
    shuffled["query_path_nodes"] = []
    shuffled["query_path_relations"] = []
    shuffled["shuffle_changed_edge_count"] = best_changes
    return shuffled


def _edge_metrics(predicted: list[dict[str, str]], gold: list[dict[str, str]]) -> dict[str, Any]:
    pred_set = {(e["source"], e["relation"], e["target"]) for e in predicted}
    gold_set = {(e["source"], e["relation"], e["target"]) for e in gold}
    tp = len(pred_set & gold_set)
    precision = tp / len(pred_set) if pred_set else (1.0 if not gold_set else 0.0)
    recall = tp / len(gold_set) if gold_set else 1.0
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {
        "edge_tp": tp,
        "edge_predicted": len(pred_set),
        "edge_gold": len(gold_set),
        "edge_precision": precision,
        "edge_recall": recall,
        "edge_f1": f1,
    }


def _run_initial_method(
    case: dict[str, Any],
    *,
    method: str,
    model: str,
    reasoning_effort: str,
    max_output_tokens: int,
    client: Any,
) -> dict[str, Any]:
    if method in {"small_direct_relation", "reference_direct_relation"}:
        system, user = _direct_prompts(case)
        output_type: type[BaseModel] = RelationAnswerOutput
    elif method == "small_text_structure":
        system, user = _text_structure_prompts(case)
        output_type = TextStructureOutput
    elif method == "small_explicit_graph":
        system, user = _graph_structure_prompts(case)
        output_type = GraphStructureOutput
    else:
        raise ValueError(f"Unknown CLUTRR initial method: {method}")
    parsed, meta = _call_parsed(
        client,
        model=model,
        reasoning_effort=reasoning_effort,
        max_output_tokens=max_output_tokens,
        system=system,
        user=user,
        output_type=output_type,
    )
    raw = parsed.model_dump()
    details: dict[str, Any] = {"raw_output": raw}
    if isinstance(parsed, TextStructureOutput):
        details["text_representation"] = {
            "direct_fact_notes": [str(x) for x in parsed.direct_fact_notes],
            "composition_plan": [str(x) for x in parsed.composition_plan],
        }
    if isinstance(parsed, GraphStructureOutput):
        graph = _normalize_graph_output(parsed)
        details["graph"] = graph
        metrics = _edge_metrics(graph["edges"], case["gold_graph"]["edges"])
        details["graph_metrics"] = {
            **metrics,
            "node_recall": len(set(graph["nodes"]) & set(case["gold_graph"]["nodes"])) / max(1, len(set(case["gold_graph"]["nodes"]))),
            "query_path_relation_exact": graph["query_path_relations"] == case["gold_graph"]["query_path_relations"],
        }
    usage = meta.get("usage") or {}
    return {
        "method": method,
        "model": model,
        "status": "ok",
        "prediction": _normalize_relation(getattr(parsed, "answer_relation", None)),
        "details": details,
        "usage": usage,
        "api_calls": int(meta.get("api_calls") or 1),
        "latency_ms": meta.get("latency_ms"),
        "response_id": meta.get("response_id"),
        "response_ids": meta.get("response_ids") or [meta.get("response_id")],
        "retry_count": int(meta.get("retry_count") or 0),
        "max_output_tokens_used": meta.get("max_output_tokens_used"),
        "estimated_cost_usd": _estimated_cost(model, usage),
    }


def _run_replay_method(
    case: dict[str, Any],
    *,
    method: str,
    representation: dict[str, Any],
    model: str,
    reasoning_effort: str,
    max_output_tokens: int,
    client: Any,
) -> dict[str, Any]:
    if method == "small_text_replay":
        system, user = _text_replay_prompts(case, representation)
    elif method in {"small_graph_replay", "small_shuffled_graph_replay"}:
        system, user = _graph_replay_prompts(case, representation)
    else:
        raise ValueError(f"Unknown CLUTRR replay method: {method}")
    parsed, meta = _call_parsed(
        client,
        model=model,
        reasoning_effort=reasoning_effort,
        max_output_tokens=max_output_tokens,
        system=system,
        user=user,
        output_type=RelationReplayOutput,
    )
    usage = meta.get("usage") or {}
    return {
        "method": method,
        "model": model,
        "status": "ok",
        "prediction": _normalize_relation(parsed.answer_relation),
        "details": {"raw_output": parsed.model_dump(), "representation": representation},
        "usage": usage,
        "api_calls": int(meta.get("api_calls") or 1),
        "latency_ms": meta.get("latency_ms"),
        "response_id": meta.get("response_id"),
        "response_ids": meta.get("response_ids") or [meta.get("response_id")],
        "retry_count": int(meta.get("retry_count") or 0),
        "max_output_tokens_used": meta.get("max_output_tokens_used"),
        "estimated_cost_usd": _estimated_cost(model, usage),
    }


def _score_prediction(gold: str, prediction: str | None) -> dict[str, Any]:
    return {
        "gold_relation": gold,
        "prediction": prediction,
        "valid_prediction": prediction is not None,
        "exact_correct": prediction == gold,
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
            rows.append(payload)
    return rows


def _dataset_file(dataset_dir: Path, config: str, split: str) -> Path:
    safe = config.replace("/", "_")
    return dataset_dir / f"{safe}__{split}.jsonl"


def _download_via_rows_api(
    target: Path,
    *,
    config: str,
    split: str,
    progress: Callable[[str], None],
) -> dict[str, Any]:
    target.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    offset = 0
    page_size = 100
    total: int | None = None
    while total is None or offset < total:
        query = urllib.parse.urlencode(
            {"dataset": DATASET_NAME, "config": config, "split": split, "offset": offset, "length": page_size}
        )
        url = f"{DATASET_SERVER_ROWS}?{query}"
        request = urllib.request.Request(url, headers={"User-Agent": "verified-reasoning-graph-v035"})
        with urllib.request.urlopen(request, timeout=90) as response:  # noqa: S310 - fixed official endpoint
            payload = json.loads(response.read().decode("utf-8"))
        page = payload.get("rows") or []
        total = int(payload.get("num_rows_total") or len(page))
        if not page and offset < total:
            raise RuntimeError(f"Hugging Face rows API returned an empty page at offset {offset}")
        for item in page:
            row = item.get("row") if isinstance(item, dict) else None
            if isinstance(row, dict):
                rows.append(row)
        offset += len(page)
        progress(f"    {split}: {len(rows)}/{total}")
        if not page:
            break
    temp = target.with_suffix(target.suffix + ".part")
    with temp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temp.replace(target)
    return {"path": str(target), "rows": len(rows), "downloaded": True, "source": "huggingface_rows_api"}


def _download_via_datasets(
    target: Path,
    *,
    config: str,
    split: str,
    progress: Callable[[str], None],
) -> dict[str, Any]:
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("The optional 'datasets' package is unavailable") from exc
    progress(f"    loading {config}/{split} with datasets package fallback")
    dataset = load_dataset(DATASET_NAME, config, split=split)
    temp = target.with_suffix(target.suffix + ".part")
    with temp.open("w", encoding="utf-8") as handle:
        for row in dataset:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
    temp.replace(target)
    return {"path": str(target), "rows": len(dataset), "downloaded": True, "source": "datasets_package"}


def download_clutrr_dataset(
    dataset_dir: Path,
    *,
    config: str = DEFAULT_CONFIG,
    split: str = DEFAULT_SPLIT,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    progress = progress or (lambda _message: None)
    target = _dataset_file(dataset_dir, config, split)
    if target.exists() and target.stat().st_size > 1000:
        return {
            "path": str(target),
            "rows": sum(1 for line in target.open("r", encoding="utf-8") if line.strip()),
            "downloaded": False,
            "source": "existing_file",
        }
    progress(f"Downloading official CLUTRR config={config}, split={split}")
    try:
        return _download_via_rows_api(target, config=config, split=split, progress=progress)
    except Exception as rows_exc:  # noqa: BLE001
        progress(f"    rows API failed: {type(rows_exc).__name__}: {rows_exc}")
        try:
            return _download_via_datasets(target, config=config, split=split, progress=progress)
        except Exception as datasets_exc:  # noqa: BLE001
            raise RuntimeError(
                "Could not download CLUTRR. Check internet access, or install the optional datasets package with "
                "`python -m pip install datasets` and run again. "
                f"Rows API error: {rows_exc}; datasets fallback error: {datasets_exc}"
            ) from datasets_exc


def load_clutrr_cases(
    dataset_dir: Path,
    *,
    config: str,
    split: str,
    hop_lengths: list[int],
    per_hop: int,
    seed: int,
    task_family: int = 1,
    max_story_chars: int = 12_000,
    exclude_case_ids: set[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = _dataset_file(dataset_dir, config, split)
    if not path.exists():
        raise FileNotFoundError(f"Missing CLUTRR data: {path}. Run with --download first.")
    excluded = {str(x) for x in (exclude_case_ids or set())}
    requested_hops = sorted(set(int(x) for x in hop_lengths))
    buckets: dict[int, list[dict[str, Any]]] = defaultdict(list)
    skipped = Counter()
    for raw in _load_jsonl(path):
        case_id = str(raw.get("id") or "")
        if not case_id or case_id in excluded:
            skipped["excluded_or_missing_id"] += 1
            continue
        if _task_family(raw.get("task_name")) != task_family:
            skipped["task_family"] += 1
            continue
        hop = _task_hop(raw.get("task_name"))
        if hop not in requested_hops:
            skipped["hop_length"] += 1
            continue
        story = str(raw.get("story") or "")
        query = _normalize_query(raw.get("query"))
        target = _normalize_relation(raw.get("target_text"))
        if not story or query is None or target is None:
            skipped["invalid_core_fields"] += 1
            continue
        if len(story) > max_story_chars:
            skipped["story_too_long"] += 1
            continue
        gold_graph = _gold_graph(raw)
        if not gold_graph["edges"]:
            skipped["missing_gold_graph"] += 1
            continue
        buckets[hop].append(
            {
                "case_id": case_id,
                "story": story,
                "clean_story": str(raw.get("clean_story") or story),
                "query": query,
                "gold_relation": target,
                "hop_length": hop,
                "task_name": str(raw.get("task_name") or ""),
                "f_comb": str(raw.get("f_comb") or ""),
                "proof_state": raw.get("proof_state"),
                "genders": str(raw.get("genders") or ""),
                "gold_graph": gold_graph,
            }
        )
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    by_hop_available: dict[str, int] = {}
    for hop in requested_hops:
        bucket = buckets.get(hop, [])
        rng.shuffle(bucket)
        by_hop_available[str(hop)] = len(bucket)
        selected.extend(bucket if per_hop <= 0 else bucket[:per_hop])
    rng.shuffle(selected)
    return selected, {
        "selected": len(selected),
        "config": config,
        "split": split,
        "task_family": task_family,
        "hop_lengths": requested_hops,
        "per_hop": per_hop,
        "seed": seed,
        "by_hop": dict(Counter(str(row["hop_length"]) for row in selected)),
        "available_by_hop": by_hop_available,
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
    representation: dict[str, Any] | None = None,
) -> str:
    return _stable_hash(
        {
            "cache_schema": CACHE_SCHEMA_VERSION,
            "prompt_version": METHOD_PROMPT_VERSIONS[method],
            "case_id": case["case_id"],
            "story": case["story"] if representation is None else None,
            "query": case["query"],
            "method": method,
            "model": model,
            "reasoning_effort": reasoning_effort,
            "max_output_tokens": max_output_tokens,
            "representation": representation,
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


def _binomial_two_sided(k: int, n: int) -> float:
    if n <= 0:
        return 1.0
    lower = sum(math.comb(n, i) for i in range(0, k + 1)) / (2**n)
    upper = sum(math.comb(n, i) for i in range(k, n + 1)) / (2**n)
    return min(1.0, 2 * min(lower, upper))


def _method_summary(rows: list[dict[str, Any]], method: str) -> dict[str, Any]:
    outputs = [row["methods"].get(method) for row in rows]
    outputs = [o for o in outputs if o and o.get("status") == "ok"]
    scores = [o["scores"] for o in outputs if o.get("scores")]
    by_hop: dict[str, Any] = {}
    for hop in sorted({row["hop_length"] for row in rows}):
        hop_scores = [
            row["methods"][method]["scores"]
            for row in rows
            if row["hop_length"] == hop and row["methods"].get(method, {}).get("status") == "ok"
        ]
        if hop_scores:
            by_hop[str(hop)] = {
                "n": len(hop_scores),
                "accuracy_percent": round(100 * statistics.mean(s["exact_correct"] for s in hop_scores), 2),
                "valid_prediction_percent": round(100 * statistics.mean(s["valid_prediction"] for s in hop_scores), 2),
            }
    short = [
        row["methods"][method]["scores"]["exact_correct"]
        for row in rows
        if row["hop_length"] <= 5 and row["methods"].get(method, {}).get("status") == "ok"
    ]
    long = [
        row["methods"][method]["scores"]["exact_correct"]
        for row in rows
        if row["hop_length"] >= 6 and row["methods"].get(method, {}).get("status") == "ok"
    ]
    usage = Counter()
    costs: list[float] = []
    latencies: list[float] = []
    for output in outputs:
        for key in ("input_tokens", "output_tokens", "total_tokens"):
            usage[key] += int((output.get("usage") or {}).get(key) or 0)
        if output.get("estimated_cost_usd") is not None:
            costs.append(float(output["estimated_cost_usd"]))
        if output.get("latency_ms") is not None:
            latencies.append(float(output["latency_ms"]))
    summary: dict[str, Any] = {
        "n": len(scores),
        "n_total": len(rows),
        "n_missing": len(rows) - len(scores),
        "accuracy_percent": round(100 * statistics.mean(s["exact_correct"] for s in scores), 2) if scores else 0.0,
        "valid_prediction_percent": round(100 * statistics.mean(s["valid_prediction"] for s in scores), 2) if scores else 0.0,
        "short_hop_accuracy_percent": round(100 * statistics.mean(short), 2) if short else None,
        "long_hop_accuracy_percent": round(100 * statistics.mean(long), 2) if long else None,
        "input_tokens": usage["input_tokens"],
        "output_tokens": usage["output_tokens"],
        "total_tokens": usage["total_tokens"],
        "api_calls": len(outputs),
        "api_calls_this_run": sum(int(o.get("api_calls_this_run") or 0) for o in outputs),
        "cache_hits": sum(int(bool((o.get("cache") or {}).get("hit"))) for o in outputs),
        "mean_latency_ms": round(statistics.mean(latencies), 2) if latencies else None,
        "estimated_cost_usd": round(sum(costs), 6),
        "estimated_cost_usd_this_run": round(sum(float(o.get("estimated_cost_usd_this_run") or 0.0) for o in outputs), 6),
        "by_hop": by_hop,
    }
    if method == "small_explicit_graph":
        metrics = [o.get("details", {}).get("graph_metrics", {}) for o in outputs]
        graphs = [o.get("details", {}).get("graph", {}) for o in outputs]
        summary.update(
            {
                "mean_edge_precision_percent": round(100 * statistics.mean(m.get("edge_precision", 0) for m in metrics), 2) if metrics else None,
                "mean_edge_recall_percent": round(100 * statistics.mean(m.get("edge_recall", 0) for m in metrics), 2) if metrics else None,
                "mean_edge_f1_percent": round(100 * statistics.mean(m.get("edge_f1", 0) for m in metrics), 2) if metrics else None,
                "mean_node_recall_percent": round(100 * statistics.mean(m.get("node_recall", 0) for m in metrics), 2) if metrics else None,
                "query_path_relation_exact_percent": round(100 * statistics.mean(bool(m.get("query_path_relation_exact")) for m in metrics), 2) if metrics else None,
                "path_shape_valid_percent": round(100 * statistics.mean(bool(g.get("path_shape_valid")) for g in graphs), 2) if graphs else None,
            }
        )
    if method == "small_shuffled_graph_replay":
        changes = [
            o.get("details", {}).get("representation", {}).get("shuffle_changed_edge_count", 0)
            for o in outputs
        ]
        summary["mean_shuffled_edges_changed"] = round(statistics.mean(changes), 2) if changes else None
    return summary


def _paired_comparison(
    rows: list[dict[str, Any]],
    method: str,
    *,
    reference_method: str,
    seed: int,
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
        "accuracy_delta_percentage_points": round(100 * delta, 2),
        "bootstrap_95ci_percentage_points": [round(100 * low, 2), round(100 * high, 2)],
        "cases_improved": improved,
        "cases_regressed": regressed,
        "net_improved_cases": improved - regressed,
        "mcnemar_exact_p": _binomial_two_sided(min(improved, regressed), discordant),
    }


def _make_report(result: dict[str, Any]) -> str:
    summaries = result["method_summaries"]
    rows = []
    for method, summary in summaries.items():
        rows.append(
            "<tr>"
            f"<td>{html.escape(method)}</td>"
            f"<td>{summary.get('accuracy_percent')}</td>"
            f"<td>{summary.get('short_hop_accuracy_percent')}</td>"
            f"<td>{summary.get('long_hop_accuracy_percent')}</td>"
            f"<td>{summary.get('total_tokens')}</td>"
            f"<td>{summary.get('mean_latency_ms')}</td>"
            f"<td>{summary.get('estimated_cost_usd')}</td>"
            "</tr>"
        )
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{html.escape(result['run_id'])}</title>
<style>body{{font-family:Arial,sans-serif;max-width:1200px;margin:30px auto;padding:0 16px}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #ccc;padding:8px;text-align:left}}pre{{white-space:pre-wrap;background:#f5f5f5;padding:12px}}</style></head>
<body><h1>CLUTRR Graph Representation Evaluation</h1>
<p>Run: {html.escape(result['run_id'])}</p>
<table><thead><tr><th>Method</th><th>Accuracy %</th><th>Short-hop %</th><th>Long-hop %</th><th>Tokens</th><th>Latency ms</th><th>Cost USD</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
<h2>Paired comparisons</h2><pre>{html.escape(json.dumps(result['paired_comparisons'], ensure_ascii=False, indent=2))}</pre>
<h2>Sampling</h2><pre>{html.escape(json.dumps(result['sampling'], ensure_ascii=False, indent=2))}</pre>
</body></html>"""


def run_clutrr_evaluation(
    *,
    dataset_dir: Path,
    output_root: Path,
    config: str = DEFAULT_CONFIG,
    split: str = DEFAULT_SPLIT,
    hop_lengths: list[int] | None = None,
    per_hop: int = 4,
    seed: int = 2035,
    task_family: int = 1,
    small_model: str = "gpt-5.4-nano",
    reference_model: str = "gpt-5.4-mini",
    include_reference: bool = True,
    include_replay_ablation: bool = True,
    reasoning_effort: str = "low",
    max_output_tokens_direct: int = 1000,
    max_output_tokens_structured: int = 2400,
    max_output_tokens_graph: int = 4200,
    max_output_tokens_replay: int = 1000,
    max_output_tokens_graph_replay: int = 1800,
    max_story_chars: int = 12_000,
    exclude_case_ids: set[str] | None = None,
    generation_cache_path: Path | None = None,
    force_methods: list[str] | None = None,
    progress: Callable[[str], None] | None = None,
    client: Any | None = None,
) -> dict[str, Any]:
    progress = progress or (lambda _message: None)
    hops = hop_lengths or list(range(2, 11))
    cases, sampling = load_clutrr_cases(
        dataset_dir,
        config=config,
        split=split,
        hop_lengths=hops,
        per_hop=per_hop,
        seed=seed,
        task_family=task_family,
        max_story_chars=max_story_chars,
        exclude_case_ids=exclude_case_ids,
    )
    if client is None:
        _load_local_env()
        from openai import OpenAI

        client = OpenAI()
    force = set(force_methods or [])
    cache = _load_cache(generation_cache_path)
    cache_methods: dict[str, Any] = cache.setdefault("methods", {})
    run_id = f"clutrr_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    methods = ["small_direct_relation", "small_text_structure", "small_explicit_graph"]
    if include_replay_ablation:
        methods += ["small_text_replay", "small_graph_replay", "small_shuffled_graph_replay"]
    if include_reference:
        methods.append("reference_direct_relation")
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    cache_hits = Counter()
    actual_calls = 0

    def execute_cached(
        case: dict[str, Any],
        *,
        method: str,
        model: str,
        max_tokens: int,
        representation: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        nonlocal actual_calls
        key = _cache_key(
            case,
            method=method,
            model=model,
            reasoning_effort=reasoning_effort,
            max_output_tokens=max_tokens,
            representation=representation,
        )
        method_cache = cache_methods.setdefault(method, {})
        if method not in force and key in method_cache:
            output = copy.deepcopy(method_cache[key])
            output["cache"] = {"hit": True, "key": key}
            output["api_calls_this_run"] = 0
            output["estimated_cost_usd_this_run"] = 0.0
            cache_hits[method] += 1
            return output
        if representation is None:
            output = _run_initial_method(
                case,
                method=method,
                model=model,
                reasoning_effort=reasoning_effort,
                max_output_tokens=max_tokens,
                client=client,
            )
        else:
            output = _run_replay_method(
                case,
                method=method,
                representation=representation,
                model=model,
                reasoning_effort=reasoning_effort,
                max_output_tokens=max_tokens,
                client=client,
            )
        method_cache[key] = copy.deepcopy(output)
        _save_cache(generation_cache_path, cache)
        output["cache"] = {"hit": False, "key": key}
        calls_made = int(output.get("api_calls") or 1)
        output["api_calls_this_run"] = calls_made
        output["estimated_cost_usd_this_run"] = output.get("estimated_cost_usd") or 0.0
        actual_calls += calls_made
        return output

    total = len(cases)
    for index, case in enumerate(cases, 1):
        progress(f"[{index}/{total}] {case['case_id']} · hop={case['hop_length']} · gold={case['gold_relation']}")
        row = {k: copy.deepcopy(v) for k, v in case.items()}
        row["case_index"] = index
        row["methods"] = {}

        def safe_execute(
            *,
            method: str,
            model: str,
            max_tokens: int,
            representation: dict[str, Any] | None = None,
        ) -> dict[str, Any] | None:
            nonlocal actual_calls
            progress(f"    - {method}")
            try:
                output = execute_cached(
                    case,
                    method=method,
                    model=model,
                    max_tokens=max_tokens,
                    representation=representation,
                )
            except Exception as exc:  # noqa: BLE001
                error = f"{type(exc).__name__}: {exc}"
                failed_calls = int(getattr(exc, "api_calls", 1))
                actual_calls += failed_calls
                failures.append({
                    "case_id": case["case_id"],
                    "method": method,
                    "error": error,
                    "api_calls_this_run": failed_calls,
                })
                output = {
                    "method": method,
                    "model": model,
                    "status": "failed",
                    "error": error,
                    "api_calls": failed_calls,
                    "api_calls_this_run": failed_calls,
                }
                progress(f"      FAILED: {error}")
                row["methods"][method] = output
                return None
            row["methods"][method] = output
            return output

        direct = safe_execute(
            method="small_direct_relation", model=small_model, max_tokens=max_output_tokens_direct
        )
        text = safe_execute(
            method="small_text_structure", model=small_model, max_tokens=max_output_tokens_structured
        )
        graph = safe_execute(
            method="small_explicit_graph", model=small_model, max_tokens=max_output_tokens_graph
        )

        if include_replay_ablation:
            if text is not None:
                text_rep = text.get("details", {}).get("text_representation", {})
                safe_execute(
                    method="small_text_replay",
                    model=small_model,
                    max_tokens=max_output_tokens_replay,
                    representation=text_rep,
                )
            else:
                row["methods"]["small_text_replay"] = {
                    "method": "small_text_replay",
                    "model": small_model,
                    "status": "skipped",
                    "reason": "small_text_structure failed",
                    "api_calls_this_run": 0,
                }

            if graph is not None:
                graph_rep = graph.get("details", {}).get("graph", {})
                safe_execute(
                    method="small_graph_replay",
                    model=small_model,
                    max_tokens=max_output_tokens_graph_replay,
                    representation=graph_rep,
                )
                shuffled = _shuffle_graph(graph_rep, seed, case["case_id"])
                safe_execute(
                    method="small_shuffled_graph_replay",
                    model=small_model,
                    max_tokens=max_output_tokens_graph_replay,
                    representation=shuffled,
                )
            else:
                for replay_method in ("small_graph_replay", "small_shuffled_graph_replay"):
                    row["methods"][replay_method] = {
                        "method": replay_method,
                        "model": small_model,
                        "status": "skipped",
                        "reason": "small_explicit_graph failed",
                        "api_calls_this_run": 0,
                    }

        if include_reference:
            safe_execute(
                method="reference_direct_relation",
                model=reference_model,
                max_tokens=max_output_tokens_direct,
            )

        for method, output in row["methods"].items():
            if output.get("status") == "ok":
                output["scores"] = _score_prediction(case["gold_relation"], output.get("prediction"))
        results.append(row)

    summaries = {method: _method_summary(results, method) for method in methods}
    comparisons: dict[str, Any] = {}
    for method in methods:
        if method != "small_direct_relation":
            comparisons[f"{method}_vs_direct"] = _paired_comparison(
                results, method, reference_method="small_direct_relation", seed=seed
            )
    if "small_explicit_graph" in methods and "small_text_structure" in methods:
        comparisons["explicit_graph_vs_text_structure"] = _paired_comparison(
            results, "small_explicit_graph", reference_method="small_text_structure", seed=seed
        )
    if include_replay_ablation:
        comparisons["graph_replay_vs_text_replay"] = _paired_comparison(
            results, "small_graph_replay", reference_method="small_text_replay", seed=seed
        )
        comparisons["graph_replay_vs_shuffled_graph"] = _paired_comparison(
            results, "small_graph_replay", reference_method="small_shuffled_graph_replay", seed=seed
        )

    result = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": {
            "name": "CLUTRR",
            "huggingface_id": DATASET_NAME,
            "official_repository": "facebookresearch/clutrr",
            "config": config,
            "split": split,
            "gold_story_edges": True,
            "gold_query_path_relations": True,
        },
        "settings": {
            "small_model": small_model,
            "reference_model": reference_model,
            "include_reference": include_reference,
            "include_replay_ablation": include_replay_ablation,
            "hop_lengths": hops,
            "per_hop": per_hop,
            "seed": seed,
            "task_family": task_family,
            "reasoning_effort": reasoning_effort,
            "max_output_tokens_direct": max_output_tokens_direct,
            "max_output_tokens_structured": max_output_tokens_structured,
            "max_output_tokens_graph": max_output_tokens_graph,
            "max_output_tokens_replay": max_output_tokens_replay,
            "max_output_tokens_graph_replay": max_output_tokens_graph_replay,
            "representation_isolation": "Replay solvers receive only the extracted text or graph representation; original story hidden",
            "shuffled_graph_control": "Reuses extracted graph with deterministically permuted relation labels and targets",
        },
        "sampling": sampling,
        "method_summaries": summaries,
        "paired_comparisons": comparisons,
        "cache_summary": {
            "generation_cache_path": str(generation_cache_path) if generation_cache_path else None,
            "cache_hits_by_method": dict(cache_hits),
            "actual_api_calls_this_run": actual_calls,
        },
        "summary": {
            "completed_cases": len(results),
            "failed_cases": len({item.get("case_id") for item in failures}),
            "failed_method_calls": len(failures),
            "primary_method": "small_explicit_graph",
            "direct_accuracy_percent": summaries.get("small_direct_relation", {}).get("accuracy_percent"),
            "text_structure_accuracy_percent": summaries.get("small_text_structure", {}).get("accuracy_percent"),
            "graph_accuracy_percent": summaries.get("small_explicit_graph", {}).get("accuracy_percent"),
            "graph_replay_accuracy_percent": summaries.get("small_graph_replay", {}).get("accuracy_percent"),
            "text_replay_accuracy_percent": summaries.get("small_text_replay", {}).get("accuracy_percent"),
            "shuffled_graph_accuracy_percent": summaries.get("small_shuffled_graph_replay", {}).get("accuracy_percent"),
            "actual_api_calls_this_run": actual_calls,
        },
        "cases": results,
    }
    (run_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    with (run_dir / "cases.jsonl").open("w", encoding="utf-8") as handle:
        for row in results:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (run_dir / "failures.json").write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "report.html").write_text(_make_report(result), encoding="utf-8")
    (run_dir / "status.json").write_text(
        json.dumps({"run_id": run_id, "completed": len(results), "failed_cases": len({item.get("case_id") for item in failures}), "failed_method_calls": len(failures)}, indent=2),
        encoding="utf-8",
    )
    return result

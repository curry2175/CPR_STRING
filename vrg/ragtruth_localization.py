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
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Literal

from pydantic import BaseModel, Field

from .openai_runner import ALLOWED_REASONING_EFFORTS, _load_local_env, _usage_dict


SCHEMA_VERSION = "0.33.0"
CACHE_SCHEMA_VERSION = "0.33.0-ragtruth-cache"
RAGTRUTH_RESPONSE_URL = (
    "https://raw.githubusercontent.com/ParticleMedia/RAGTruth/main/dataset/response.jsonl"
)
RAGTRUTH_SOURCE_URL = (
    "https://raw.githubusercontent.com/ParticleMedia/RAGTruth/main/dataset/source_info.jsonl"
)
METHOD_PROMPT_VERSIONS = {
    "small_direct_span": "v033-ragtruth-direct-full-evidence-numeric-boundary",
    "small_checklist_span": "v033-ragtruth-checklist-full-evidence-numeric-boundary",
    "small_claim_graph": "v033-ragtruth-one-pass-claim-graph-minimal-problem-span",
    "reference_direct_span": "v033-ragtruth-direct-full-evidence-numeric-boundary",
}
MODEL_PRICE_USD_PER_MILLION: dict[str, tuple[float, float]] = {
    "gpt-5.4-nano": (0.20, 1.25),
    "gpt-5.4-mini": (0.75, 4.50),
    "gpt-5-nano": (0.05, 0.40),
    "gpt-5-mini": (0.25, 2.00),
}


class SpanPrediction(BaseModel):
    sentence_id: str = Field(description="Answer sentence id such as a1")
    text: str = Field(description="Exact minimal contiguous quote from the answer")
    label_type: Literal["unsupported", "contradiction"]
    evidence_ids: list[str] = Field(
        default_factory=list,
        description="Reference evidence ids that support the contradiction decision; empty is allowed for unsupported claims",
    )
    explanation: str = ""


class DirectSpanOutput(BaseModel):
    hallucinated_spans: list[SpanPrediction] = Field(default_factory=list)
    summary: str = ""


class ClaimNode(BaseModel):
    id: str = Field(description="Sequential claim id such as c1")
    sentence_id: str = Field(description="Answer sentence id such as a1")
    text: str = Field(description="Exact contiguous answer quote expressing one atomic claim")
    relation: Literal["supported", "unsupported", "contradicted"]
    problem_text: str = Field(
        default="",
        description=(
            "For unsupported or contradicted claims, the smallest exact contiguous subspan that is actually wrong. "
            "Leave empty for supported claims."
        ),
    )
    evidence_ids: list[str] = Field(default_factory=list)
    explanation: str = ""


class LightClaimGraphOutput(BaseModel):
    claims: list[ClaimNode] = Field(default_factory=list)
    summary: str = ""


class EvidenceUnit(BaseModel):
    id: str
    text: str
    source_path: str = ""


class AnswerSentence(BaseModel):
    id: str
    text: str
    start: int
    end: int


def _norm(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


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


def _sentence_offsets(text: str, prefix: str = "a") -> list[AnswerSentence]:
    rows: list[AnswerSentence] = []
    pattern = re.compile(r".+?(?:[.!?]+(?=\s|$)|\n+|$)", re.MULTILINE | re.DOTALL)
    for match in pattern.finditer(text):
        raw = match.group(0)
        left = len(raw) - len(raw.lstrip())
        right = len(raw.rstrip())
        start = match.start() + left
        end = match.start() + right
        if end <= start:
            continue
        rows.append(AnswerSentence(id=f"{prefix}{len(rows) + 1}", text=text[start:end], start=start, end=end))
    if not rows and text.strip():
        start = len(text) - len(text.lstrip())
        end = len(text.rstrip())
        rows.append(AnswerSentence(id=f"{prefix}1", text=text[start:end], start=start, end=end))
    return rows


def _split_passages(value: str) -> list[str]:
    matches = list(re.finditer(r"(?i)(?:^|\n\s*)passage\s+\d+\s*:\s*", value))
    if not matches:
        return []
    chunks: list[str] = []
    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(value)
        chunk = _norm(value[start:end])
        if chunk:
            chunks.append(chunk)
    return chunks


def _flatten_source(value: Any, path: str = "source") -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    if value is None:
        rows.append((path, f"{path}: null"))
        return rows
    if isinstance(value, bool):
        rows.append((path, f"{path}: {str(value).lower()}"))
        return rows
    if isinstance(value, (int, float)):
        rows.append((path, f"{path}: {value}"))
        return rows
    if isinstance(value, str):
        passages = _split_passages(value)
        if passages:
            rows.extend((f"{path}.passage_{i}", text) for i, text in enumerate(passages, 1))
            return rows
        sentences = _sentence_offsets(value, prefix="s")
        if len(sentences) <= 1:
            text = _norm(value)
            if text:
                rows.append((path, text))
            return rows
        rows.extend((f"{path}.{sentence.id}", sentence.text) for sentence in sentences)
        return rows
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            rows.extend(_flatten_source(child, child_path))
        return rows
    if isinstance(value, list):
        for index, child in enumerate(value, 1):
            rows.extend(_flatten_source(child, f"{path}[{index}]"))
        return rows
    rows.append((path, f"{path}: {_norm(value)}"))
    return rows


def build_source_units(source_info: Any) -> list[EvidenceUnit]:
    raw = _flatten_source(source_info)
    units: list[EvidenceUnit] = []
    seen: set[str] = set()
    for path, text in raw:
        normalized = _norm(text)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        units.append(EvidenceUnit(id=f"e{len(units) + 1}", text=normalized, source_path=path))
    return units


_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)?")
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "by", "for", "from", "had", "has", "have",
    "he", "her", "his", "i", "in", "into", "is", "it", "its", "of", "on", "or", "our", "she", "that", "the",
    "their", "there", "they", "this", "to", "was", "were", "will", "with", "you", "your",
}


def _tokens(text: str) -> set[str]:
    return {token.lower() for token in _TOKEN_RE.findall(text) if token.lower() not in _STOPWORDS and len(token) > 1}


def _retrieval_score(query: str, evidence: str) -> float:
    q = _tokens(query)
    e = _tokens(evidence)
    if not q or not e:
        return 0.0
    overlap = q & e
    score = len(overlap) / math.sqrt(len(q) * len(e))
    q_numbers = {x for x in q if any(ch.isdigit() for ch in x)}
    e_numbers = {x for x in e if any(ch.isdigit() for ch in x)}
    score += 0.7 * len(q_numbers & e_numbers)
    return score


def _render_evidence_unit(unit: EvidenceUnit) -> str:
    path = _norm(unit.source_path)
    if path:
        return f"[{unit.id}] ({path}) {unit.text}"
    return f"[{unit.id}] {unit.text}"


def build_evidence_card(
    source_info: Any,
    response: str,
    *,
    max_context_chars: int = 40_000,
    max_units: int = 96,
    top_k_per_sentence: int = 5,
    force_full: bool = False,
) -> dict[str, Any]:
    all_units = build_source_units(source_info)
    rendered_all = [_render_evidence_unit(unit) for unit in all_units]
    full_text = "\n".join(rendered_all)
    full_chars = len(full_text)
    if force_full:
        if full_chars > max_context_chars:
            raise ValueError(
                f"Full evidence requires {full_chars} characters, exceeding max_context_chars={max_context_chars}"
            )
        selected = all_units
        mode = "full_required"
    elif full_chars <= max_context_chars and len(all_units) <= max_units:
        selected = all_units
        mode = "full"
    else:
        sentences = _sentence_offsets(response)
        best: dict[str, float] = {}
        for sentence in sentences:
            ranked = sorted(
                ((_retrieval_score(sentence.text, f"{unit.source_path} {unit.text}"), unit) for unit in all_units),
                key=lambda item: (item[0], -int(item[1].id[1:])),
                reverse=True,
            )
            for score, unit in ranked[:top_k_per_sentence]:
                best[unit.id] = max(best.get(unit.id, -1.0), score)
        selected = []
        chars = 0
        for unit in sorted(all_units, key=lambda item: (-best.get(item.id, -1.0), int(item.id[1:]))):
            if unit.id not in best:
                continue
            rendered = _render_evidence_unit(unit)
            additional = len(rendered) + 1
            if selected and (len(selected) >= max_units or chars + additional > max_context_chars):
                continue
            selected.append(unit)
            chars += additional
        mode = "lexical_retrieval_with_paths"
    text = "\n".join(_render_evidence_unit(unit) for unit in selected)
    return {
        "units": [unit.model_dump() for unit in selected],
        "text": text,
        "mode": mode,
        "all_unit_count": len(all_units),
        "selected_unit_count": len(selected),
        "full_source_chars": full_chars,
        "card_chars": len(text),
        "coverage_ratio": round(len(selected) / len(all_units), 4) if all_units else 1.0,
        "full_evidence_used": len(selected) == len(all_units),
    }


def _task_instruction(source: dict[str, Any]) -> str:
    source_info = source.get("source_info")
    if isinstance(source_info, dict):
        question = _norm(source_info.get("question"))
        if question:
            return question
    task_type = str(source.get("task_type") or "").lower()
    if task_type == "summary":
        return "Summarize the supplied source faithfully."
    if task_type == "data2txt":
        return "Describe the supplied structured record faithfully."
    return _norm(source.get("prompt"))[:500] or "Answer using only the supplied evidence."


def download_ragtruth_dataset(dataset_dir: Path, *, progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    progress = progress or (lambda _message: None)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    targets = {
        "response": (RAGTRUTH_RESPONSE_URL, dataset_dir / "response.jsonl"),
        "source_info": (RAGTRUTH_SOURCE_URL, dataset_dir / "source_info.jsonl"),
    }
    status: dict[str, Any] = {}
    for name, (url, path) in targets.items():
        if path.exists() and path.stat().st_size > 1000:
            status[name] = {"path": str(path), "downloaded": False, "bytes": path.stat().st_size}
            continue
        progress(f"Downloading RAGTruth {path.name} ...")
        temp = path.with_suffix(path.suffix + ".part")
        request = urllib.request.Request(url, headers={"User-Agent": "verified-reasoning-graph-v033"})
        try:
            with urllib.request.urlopen(request, timeout=120) as response, temp.open("wb") as output:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
            temp.replace(path)
        except Exception:
            if temp.exists():
                temp.unlink()
            raise
        status[name] = {"path": str(path), "downloaded": True, "bytes": path.stat().st_size}
    return status


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path.name} line {line_no}: {exc}") from exc
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _gold_label_type(label: dict[str, Any]) -> str:
    raw = str(label.get("label_type") or "").lower()
    return "contradiction" if "conflict" in raw or "contradict" in raw else "unsupported"


def _prepare_gold_labels(response: str, labels: list[dict[str, Any]], *, include_implicit_true: bool) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    for label in labels or []:
        if not include_implicit_true and bool(label.get("implicit_true")):
            continue
        try:
            start = int(label.get("start"))
            end = int(label.get("end"))
        except (TypeError, ValueError):
            continue
        start = max(0, min(len(response), start))
        end = max(start, min(len(response), end))
        if end <= start:
            continue
        prepared.append({
            "start": start,
            "end": end,
            "text": response[start:end],
            "label_type": _gold_label_type(label),
            "raw_label_type": str(label.get("label_type") or ""),
            "implicit_true": bool(label.get("implicit_true")),
            "due_to_null": bool(label.get("due_to_null")),
            "meta": str(label.get("meta") or ""),
        })
    return prepared


def load_ragtruth_cases(
    response_path: Path,
    source_path: Path,
    *,
    split: str = "test",
    quality: str = "good",
    task_types: list[str] | None = None,
    limit: int = 24,
    seed: int = 2032,
    max_response_chars: int = 3000,
    include_implicit_true: bool = True,
    exclude_case_ids: set[str] | None = None,
    include_case_ids: set[str] | None = None,
    require_full_evidence: bool = False,
    max_context_chars: int = 60_000,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    sources = {str(row.get("source_id")): row for row in _read_jsonl(source_path)}
    allowed_tasks = {x.lower() for x in (task_types or ["QA", "Summary", "Data2txt"])}
    excluded = {str(value) for value in (exclude_case_ids or set())}
    included = {str(value) for value in (include_case_ids or set())}
    candidates: list[dict[str, Any]] = []
    skipped = Counter()
    for response_row in _read_jsonl(response_path):
        if split and str(response_row.get("split") or "") != split:
            skipped["split"] += 1
            continue
        if quality and str(response_row.get("quality") or "") != quality:
            skipped["quality"] += 1
            continue
        case_id = str(response_row.get("id") or "")
        if included and case_id not in included:
            skipped["not_requested_case_id"] += 1
            continue
        if case_id in excluded:
            skipped["excluded_case_id"] += 1
            continue
        source = sources.get(str(response_row.get("source_id") or ""))
        if source is None:
            skipped["missing_source"] += 1
            continue
        task_type = str(source.get("task_type") or "")
        if allowed_tasks and task_type.lower() not in allowed_tasks:
            skipped["task_type"] += 1
            continue
        response = str(response_row.get("response") or "")
        if not response.strip() or len(response) > max_response_chars:
            skipped["response_length"] += 1
            continue
        if require_full_evidence:
            full_evidence_text = "\n".join(
                _render_evidence_unit(unit) for unit in build_source_units(source.get("source_info"))
            )
            if len(full_evidence_text) > max_context_chars:
                skipped["full_evidence_too_long"] += 1
                continue
        gold_labels = _prepare_gold_labels(
            response,
            list(response_row.get("labels") or []),
            include_implicit_true=include_implicit_true,
        )
        candidates.append({
            "case_id": case_id,
            "source_id": str(response_row.get("source_id")),
            "generator_model": str(response_row.get("model") or ""),
            "temperature": response_row.get("temperature"),
            "task_type": task_type,
            "source_name": str(source.get("source") or ""),
            "task_instruction": _task_instruction(source),
            "source_info": source.get("source_info"),
            "response": response,
            "gold_labels": gold_labels,
            "gold_has_hallucination": bool(gold_labels),
            "split": str(response_row.get("split") or ""),
            "quality": str(response_row.get("quality") or ""),
        })

    rng = random.Random(seed)
    rng.shuffle(candidates)
    if limit > 0 and limit < len(candidates):
        groups: dict[tuple[str, bool], list[dict[str, Any]]] = defaultdict(list)
        for row in candidates:
            groups[(row["task_type"], bool(row["gold_has_hallucination"]))].append(row)
        for rows in groups.values():
            rng.shuffle(rows)
        selected: list[dict[str, Any]] = []
        keys = sorted(groups, key=lambda item: (item[0].lower(), item[1]))
        while len(selected) < limit and any(groups.values()):
            for key in keys:
                if groups[key] and len(selected) < limit:
                    selected.append(groups[key].pop())
        candidates = selected

    counts = Counter((row["task_type"], bool(row["gold_has_hallucination"])) for row in candidates)
    return candidates, {
        "selected": len(candidates),
        "split": split,
        "quality": quality,
        "task_types": sorted({row["task_type"] for row in candidates}),
        "include_implicit_true": include_implicit_true,
        "excluded_case_ids_requested": len(excluded),
        "included_case_ids_requested": sorted(included),
        "require_full_evidence": require_full_evidence,
        "max_context_chars": max_context_chars,
        "by_task_and_label": {
            f"{task}:{'hallucinated' if label else 'clean'}": count
            for (task, label), count in sorted(counts.items())
        },
        "skipped": dict(skipped),
    }


def _answer_block(response: str) -> tuple[list[AnswerSentence], str]:
    sentences = _sentence_offsets(response)
    return sentences, "\n".join(f"[{sentence.id}] {sentence.text}" for sentence in sentences)


def _direct_prompts(case: dict[str, Any], evidence_card: dict[str, Any], *, checklist: bool) -> tuple[str, str]:
    checklist_text = ""
    if checklist:
        checklist_text = (
            "\nBefore returning spans, check separately for: invented entities or attributes; unsupported numbers or dates; "
            "claims contradicting the evidence; unsupported causal, temporal, or scope extensions; and mixed sentences "
            "whose unsupported clause should be isolated rather than marking the whole sentence."
        )
    system = (
        "You are a fine-grained RAG faithfulness auditor. Use only the supplied reference evidence. "
        "Locate hallucinated content inside the answer, not merely whether the answer is globally good or bad. "
        "A hallucinated span is either contradicted by the evidence or unsupported by it. Return the smallest exact "
        "contiguous answer quote that contains the unsupported factual proposition. Do not flag style, omissions, "
        "vagueness, harmless connective language, or a supported claim. If a sentence mixes supported and unsupported "
        "content, return only the unsupported clause. Every returned text must be copied exactly from one answer sentence. "
        "Use label_type='contradiction' only when evidence directly conflicts; otherwise use 'unsupported'. "
        "Never extract a numeric substring from a larger number: for example, do not return '5 stars' from '3.5 stars'."
        + checklist_text
    )
    _sentences, answer_block = _answer_block(case["response"])
    user = (
        f"TASK\n{case['task_instruction']}\n\n"
        f"REFERENCE EVIDENCE\n{evidence_card['text']}\n\n"
        f"ANSWER WITH STABLE SENTENCE IDS\n{answer_block}\n\n"
        "Return all and only hallucinated minimal spans. Use evidence ids when relevant."
    )
    return system, user


def _graph_prompts(case: dict[str, Any], evidence_card: dict[str, Any]) -> tuple[str, str]:
    system = (
        "You build a compact claim-evidence graph for RAG faithfulness auditing. Decompose the answer into minimal atomic "
        "factual claim spans copied exactly from the answer. Ignore pure discourse or style. For each claim, create one "
        "claim node and classify its relation to the supplied evidence as supported, contradicted, or unsupported. "
        "If one sentence contains both supported and unsupported clauses, split them into separate minimal claim nodes. "
        "Use contradicted only for a direct conflict. Use unsupported when no supplied evidence entails the claim. "
        "Supported and contradicted claims should cite the most relevant evidence ids. For every unsupported or "
        "contradicted claim, set problem_text to the smallest exact contiguous subspan that is actually wrong; the rest "
        "of the claim may be supported. Leave problem_text empty for supported claims. Never extract a numeric substring "
        "from a larger number (for example, '5 stars' is not a valid subspan claim inside '3.5 stars'). Keep the graph "
        "compact: do not repeat the same claim, do not reproduce whole paragraphs, and do not add outside knowledge."
    )
    _sentences, answer_block = _answer_block(case["response"])
    user = (
        f"TASK\n{case['task_instruction']}\n\n"
        f"REFERENCE EVIDENCE\n{evidence_card['text']}\n\n"
        f"ANSWER WITH STABLE SENTENCE IDS\n{answer_block}\n\n"
        "Return the compact typed claim-evidence graph. The hallucinated spans will be derived deterministically from "
        "problem_text on claim nodes labeled unsupported or contradicted."
    )
    return system, user


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


def _normalized_with_map(text: str) -> tuple[str, list[int]]:
    normalized: list[str] = []
    mapping: list[int] = []
    previous_space = False
    for index, char in enumerate(text):
        if char.isspace():
            if normalized and not previous_space:
                normalized.append(" ")
                mapping.append(index)
            previous_space = True
        else:
            normalized.append(char.lower())
            mapping.append(index)
            previous_space = False
    while normalized and normalized[-1] == " ":
        normalized.pop()
        mapping.pop()
    return "".join(normalized), mapping


def _valid_quote_boundary(text: str, start: int, end: int, quote: str) -> bool:
    """Reject matches embedded in a larger token or numeric literal."""
    if start < 0 or end > len(text) or end <= start:
        return False
    stripped = quote.strip()
    if not stripped:
        return False
    first = stripped[0]
    last = stripped[-1]
    before = text[start - 1] if start > 0 else ""
    after = text[end] if end < len(text) else ""
    if first.isalnum() and (before.isalnum() or before == "_"):
        return False
    if last.isalnum() and (after.isalnum() or after == "_"):
        return False
    # A model may emit "5 stars" although the answer says "3.5 stars". The
    # substring is textually present but is not an independent numeric claim.
    if first.isdigit() and start > 0:
        if before.isdigit() or before in "+-":
            return False
        if before in ".," and start > 1 and text[start - 2].isdigit():
            return False
    if last.isdigit() and end < len(text):
        if after.isdigit():
            return False
        if after in ".," and end + 1 < len(text) and text[end + 1].isdigit():
            return False
    return True


def _find_valid_literal(region: str, needle: str, *, case_sensitive: bool = True) -> tuple[int, int] | None:
    haystack = region if case_sensitive else region.lower()
    query = needle if case_sensitive else needle.lower()
    cursor = 0
    while query:
        position = haystack.find(query, cursor)
        if position < 0:
            return None
        end = position + len(query)
        if _valid_quote_boundary(region, position, end, region[position:end]):
            return position, end
        cursor = position + 1
    return None


def locate_exact_quote(response: str, quote: str, sentence_id: str = "") -> tuple[int, int] | None:
    quote = str(quote or "")
    if not quote.strip():
        return None
    sentences = {sentence.id: sentence for sentence in _sentence_offsets(response)}
    regions: list[tuple[int, int]] = []
    if sentence_id in sentences:
        sentence = sentences[sentence_id]
        regions.append((sentence.start, sentence.end))
    regions.append((0, len(response)))
    seen_regions: set[tuple[int, int]] = set()
    for region_start, region_end in regions:
        if (region_start, region_end) in seen_regions:
            continue
        seen_regions.add((region_start, region_end))
        region = response[region_start:region_end]
        located = _find_valid_literal(region, quote, case_sensitive=True)
        if located is None:
            located = _find_valid_literal(region, quote, case_sensitive=False)
        if located is not None:
            return region_start + located[0], region_start + located[1]
        normalized_region, region_map = _normalized_with_map(region)
        normalized_quote, _ = _normalized_with_map(quote)
        cursor = 0
        while normalized_quote:
            position = normalized_region.find(normalized_quote, cursor)
            if position < 0:
                break
            start_map = region_map[position]
            end_map = region_map[position + len(normalized_quote) - 1] + 1
            if _valid_quote_boundary(region, start_map, end_map, region[start_map:end_map]):
                return region_start + start_map, region_start + end_map
            cursor = position + 1
    return None


def _predictions_from_direct(parsed: DirectSpanOutput, response: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    seen: set[tuple[int, int, str]] = set()
    for item in parsed.hallucinated_spans:
        located = locate_exact_quote(response, item.text, item.sentence_id)
        if located is None:
            unresolved.append(item.model_dump())
            continue
        start, end = located
        key = (start, end, item.label_type)
        if key in seen:
            continue
        seen.add(key)
        predictions.append({
            "start": start,
            "end": end,
            "text": response[start:end],
            "label_type": item.label_type,
            "sentence_id": item.sentence_id,
            "evidence_ids": list(item.evidence_ids),
            "explanation": item.explanation,
        })
    return predictions, {
        "raw_output": parsed.model_dump(),
        "unresolved_predictions": unresolved,
        "predicted_count": len(parsed.hallucinated_spans),
        "resolved_count": len(predictions),
    }


def _locate_problem_subspan(
    response: str,
    *,
    sentence_id: str,
    claim_location: tuple[int, int] | None,
    problem_text: str,
) -> tuple[int, int] | None:
    if not problem_text.strip():
        return None
    if claim_location is not None:
        claim_start, claim_end = claim_location
        claim_text = response[claim_start:claim_end]
        relative = locate_exact_quote(claim_text, problem_text)
        if relative is not None:
            return claim_start + relative[0], claim_start + relative[1]
    return locate_exact_quote(response, problem_text, sentence_id)


def _predictions_from_graph(parsed: LightClaimGraphOutput, response: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    graph_nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    seen: set[tuple[int, int, str]] = set()
    fallback_count = 0
    for index, item in enumerate(parsed.claims, 1):
        node_id = _norm(item.id) or f"c{index}"
        claim_location = locate_exact_quote(response, item.text, item.sentence_id)
        node = item.model_dump()
        node["id"] = node_id
        if claim_location is not None:
            node["start"], node["end"] = claim_location
            node["resolved_text"] = response[claim_location[0]:claim_location[1]]
        else:
            node["start"] = None
            node["end"] = None
            unresolved.append({**node, "unresolved_field": "text"})
        for evidence_id in item.evidence_ids:
            edges.append({"source": evidence_id, "relation": item.relation, "target": node_id})

        problem_location: tuple[int, int] | None = None
        used_problem_text = False
        if item.relation != "supported":
            problem_location = _locate_problem_subspan(
                response,
                sentence_id=item.sentence_id,
                claim_location=claim_location,
                problem_text=item.problem_text,
            )
            used_problem_text = problem_location is not None
            if problem_location is None and claim_location is not None:
                problem_location = claim_location
                fallback_count += 1
        if problem_location is not None:
            node["problem_start"], node["problem_end"] = problem_location
            node["resolved_problem_text"] = response[problem_location[0]:problem_location[1]]
            node["problem_text_fallback_to_claim"] = not used_problem_text
        else:
            node["problem_start"] = None
            node["problem_end"] = None
            if item.relation != "supported":
                unresolved.append({**node, "unresolved_field": "problem_text"})
        graph_nodes.append(node)

        if item.relation == "supported" or problem_location is None:
            continue
        start, end = problem_location
        label_type = "contradiction" if item.relation == "contradicted" else "unsupported"
        key = (start, end, label_type)
        if key in seen:
            continue
        seen.add(key)
        predictions.append({
            "start": start,
            "end": end,
            "text": response[start:end],
            "label_type": label_type,
            "sentence_id": item.sentence_id,
            "claim_id": node_id,
            "claim_text": item.text,
            "problem_text_requested": item.problem_text,
            "problem_text_fallback_to_claim": not used_problem_text,
            "evidence_ids": list(item.evidence_ids),
            "explanation": item.explanation,
        })
    return predictions, {
        "raw_output": parsed.model_dump(),
        "graph": {"nodes": graph_nodes, "edges": edges},
        "unresolved_claims": unresolved,
        "claim_count": len(parsed.claims),
        "resolved_claim_count": len(parsed.claims) - sum(1 for node in graph_nodes if node.get("start") is None),
        "problem_text_fallback_count": fallback_count,
    }


def _mask(length: int, spans: list[dict[str, Any]]) -> list[bool]:
    values = [False] * length
    for span in spans:
        start = max(0, min(length, int(span.get("start") or 0)))
        end = max(start, min(length, int(span.get("end") or 0)))
        for index in range(start, end):
            values[index] = True
    return values


def _span_iou(left: dict[str, Any], right: dict[str, Any]) -> float:
    start = max(int(left["start"]), int(right["start"]))
    end = min(int(left["end"]), int(right["end"]))
    intersection = max(0, end - start)
    union = max(int(left["end"]), int(right["end"])) - min(int(left["start"]), int(right["start"]))
    return intersection / union if union else 0.0


def _match_spans(predicted: list[dict[str, Any]], gold: list[dict[str, Any]], threshold: float = 0.5) -> list[tuple[int, int, float]]:
    candidates: list[tuple[float, int, int]] = []
    for p_index, pred in enumerate(predicted):
        for g_index, target in enumerate(gold):
            iou = _span_iou(pred, target)
            if iou >= threshold:
                candidates.append((iou, p_index, g_index))
    matches: list[tuple[int, int, float]] = []
    used_pred: set[int] = set()
    used_gold: set[int] = set()
    for iou, p_index, g_index in sorted(candidates, reverse=True):
        if p_index in used_pred or g_index in used_gold:
            continue
        used_pred.add(p_index)
        used_gold.add(g_index)
        matches.append((p_index, g_index, iou))
    return matches


def score_predictions(response: str, predicted: list[dict[str, Any]], gold: list[dict[str, Any]]) -> dict[str, Any]:
    pred_mask = _mask(len(response), predicted)
    gold_mask = _mask(len(response), gold)
    tp = sum(p and g for p, g in zip(pred_mask, gold_mask))
    fp = sum(p and not g for p, g in zip(pred_mask, gold_mask))
    fn = sum((not p) and g for p, g in zip(pred_mask, gold_mask))
    precision = tp / (tp + fp) if tp + fp else (1.0 if not gold else 0.0)
    recall = tp / (tp + fn) if tp + fn else (1.0 if not predicted else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    matches = _match_spans(predicted, gold, threshold=0.5)
    span_precision = len(matches) / len(predicted) if predicted else (1.0 if not gold else 0.0)
    span_recall = len(matches) / len(gold) if gold else (1.0 if not predicted else 0.0)
    span_f1 = 2 * span_precision * span_recall / (span_precision + span_recall) if span_precision + span_recall else 0.0
    exact_matches = sum(
        1 for pred in predicted for target in gold
        if int(pred["start"]) == int(target["start"]) and int(pred["end"]) == int(target["end"])
    )
    type_matches = sum(
        predicted[p_index].get("label_type") == gold[g_index].get("label_type")
        for p_index, g_index, _iou in matches
    )
    predicted_positive = bool(predicted)
    gold_positive = bool(gold)
    return {
        "char_tp": tp,
        "char_fp": fp,
        "char_fn": fn,
        "char_precision": precision,
        "char_recall": recall,
        "char_f1": f1,
        "span_matches_iou50": len(matches),
        "span_precision_iou50": span_precision,
        "span_recall_iou50": span_recall,
        "span_f1_iou50": span_f1,
        "gold_span_count": len(gold),
        "predicted_span_count": len(predicted),
        "exact_span_matches": exact_matches,
        "exact_gold_span_recall": exact_matches / len(gold) if gold else (1.0 if not predicted else 0.0),
        "matched_type_accuracy": type_matches / len(matches) if matches else None,
        "gold_has_hallucination": gold_positive,
        "predicted_has_hallucination": predicted_positive,
        "response_detection_correct": predicted_positive == gold_positive,
    }


def _empty_cache() -> dict[str, Any]:
    return {"schema_version": CACHE_SCHEMA_VERSION, "methods": {}}


def _load_cache(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return _empty_cache()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return _empty_cache()
    if not isinstance(data, dict):
        return _empty_cache()
    data.setdefault("methods", {})
    data["schema_version"] = CACHE_SCHEMA_VERSION
    return data


def _save_cache(path: Path | None, cache: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def _cache_key(
    case: dict[str, Any],
    evidence_card: dict[str, Any],
    *,
    method: str,
    model: str,
    reasoning_effort: str,
    max_output_tokens: int,
) -> str:
    return _stable_hash({
        "case_id": case["case_id"],
        "response": case["response"],
        "task_instruction": case["task_instruction"],
        "evidence_units": evidence_card["units"],
        "method": method,
        "model": model,
        "reasoning_effort": reasoning_effort,
        "max_output_tokens": max_output_tokens,
        "prompt_version": METHOD_PROMPT_VERSIONS[method],
    })


def _run_method(
    case: dict[str, Any],
    evidence_card: dict[str, Any],
    *,
    method: str,
    model: str,
    reasoning_effort: str,
    max_output_tokens: int,
    client: Any,
) -> dict[str, Any]:
    if method == "small_claim_graph":
        system, user = _graph_prompts(case, evidence_card)
        parsed, meta = _call_parsed(
            client,
            model=model,
            reasoning_effort=reasoning_effort,
            max_output_tokens=max(max_output_tokens, 3200),
            system=system,
            user=user,
            output_type=LightClaimGraphOutput,
        )
        predictions, details = _predictions_from_graph(parsed, case["response"])
    else:
        system, user = _direct_prompts(case, evidence_card, checklist=method == "small_checklist_span")
        parsed, meta = _call_parsed(
            client,
            model=model,
            reasoning_effort=reasoning_effort,
            max_output_tokens=max_output_tokens,
            system=system,
            user=user,
            output_type=DirectSpanOutput,
        )
        predictions, details = _predictions_from_direct(parsed, case["response"])
    usage = meta["usage"]
    return {
        "method": method,
        "model": model,
        "status": "ok",
        "predicted_spans": predictions,
        "details": details,
        "usage": usage,
        "api_calls": 1,
        "latency_ms": meta["latency_ms"],
        "response_id": meta["response_id"],
        "estimated_cost_usd": _estimated_cost(model, usage),
    }


def _method_summary(rows: list[dict[str, Any]], method: str) -> dict[str, Any]:
    available = [row for row in rows if isinstance((row.get("methods") or {}).get(method, {}).get("scores"), dict)]
    total = len(rows)
    tp = sum(row["methods"][method]["scores"]["char_tp"] for row in available)
    fp = sum(row["methods"][method]["scores"]["char_fp"] for row in available)
    fn = sum(row["methods"][method]["scores"]["char_fn"] for row in available)
    char_precision = tp / (tp + fp) if tp + fp else 1.0
    char_recall = tp / (tp + fn) if tp + fn else 1.0
    char_f1 = 2 * char_precision * char_recall / (char_precision + char_recall) if char_precision + char_recall else 0.0
    response_tp = sum(
        row["methods"][method]["scores"]["gold_has_hallucination"]
        and row["methods"][method]["scores"]["predicted_has_hallucination"]
        for row in available
    )
    response_fp = sum(
        not row["methods"][method]["scores"]["gold_has_hallucination"]
        and row["methods"][method]["scores"]["predicted_has_hallucination"]
        for row in available
    )
    response_fn = sum(
        row["methods"][method]["scores"]["gold_has_hallucination"]
        and not row["methods"][method]["scores"]["predicted_has_hallucination"]
        for row in available
    )
    response_precision = response_tp / (response_tp + response_fp) if response_tp + response_fp else 1.0
    response_recall = response_tp / (response_tp + response_fn) if response_tp + response_fn else 1.0
    response_f1 = 2 * response_precision * response_recall / (response_precision + response_recall) if response_precision + response_recall else 0.0
    usage = {
        key: sum(int(row["methods"][method].get("usage", {}).get(key) or 0) for row in available)
        for key in ("input_tokens", "output_tokens", "total_tokens")
    }
    costs = [row["methods"][method].get("estimated_cost_usd") for row in available]
    known_costs = [float(value) for value in costs if value is not None]
    by_task: dict[str, Any] = {}
    for task_type in sorted({row["task_type"] for row in available}):
        subset = [row for row in available if row["task_type"] == task_type]
        by_task[task_type] = {
            "n": len(subset),
            "mean_char_f1_percent": round(statistics.mean(row["methods"][method]["scores"]["char_f1"] for row in subset) * 100, 2),
            "response_accuracy_percent": round(statistics.mean(row["methods"][method]["scores"]["response_detection_correct"] for row in subset) * 100, 2),
        }
    unresolved = 0
    predicted_total = 0
    for row in available:
        details = row["methods"][method].get("details") or {}
        unresolved += len(details.get("unresolved_predictions") or details.get("unresolved_claims") or [])
        predicted_total += int(details.get("predicted_count") or details.get("claim_count") or 0)
    return {
        "n": len(available),
        "n_total": total,
        "n_missing": total - len(available),
        "char_precision_percent": round(char_precision * 100, 2),
        "char_recall_percent": round(char_recall * 100, 2),
        "char_f1_percent": round(char_f1 * 100, 2),
        "mean_case_char_f1_percent": round(statistics.mean(row["methods"][method]["scores"]["char_f1"] for row in available) * 100, 2) if available else None,
        "mean_span_f1_iou50_percent": round(statistics.mean(row["methods"][method]["scores"]["span_f1_iou50"] for row in available) * 100, 2) if available else None,
        "mean_exact_gold_span_recall_percent": round(statistics.mean(row["methods"][method]["scores"]["exact_gold_span_recall"] for row in available) * 100, 2) if available else None,
        "response_precision_percent": round(response_precision * 100, 2),
        "response_recall_percent": round(response_recall * 100, 2),
        "response_f1_percent": round(response_f1 * 100, 2),
        "response_accuracy_percent": round(statistics.mean(row["methods"][method]["scores"]["response_detection_correct"] for row in available) * 100, 2) if available else None,
        "clean_false_positive_rate_percent": round(response_fp / sum(not row["methods"][method]["scores"]["gold_has_hallucination"] for row in available) * 100, 2) if any(not row["methods"][method]["scores"]["gold_has_hallucination"] for row in available) else None,
        "matched_type_accuracy_percent": round(statistics.mean(
            score["matched_type_accuracy"]
            for row in available
            if (score := row["methods"][method]["scores"])["matched_type_accuracy"] is not None
        ) * 100, 2) if any(row["methods"][method]["scores"]["matched_type_accuracy"] is not None for row in available) else None,
        "quote_or_claim_resolution_percent": round((predicted_total - unresolved) / predicted_total * 100, 2) if predicted_total else 100.0,
        **usage,
        "api_calls": sum(int(row["methods"][method].get("api_calls") or 0) for row in available),
        "api_calls_this_run": sum(int(row["methods"][method].get("api_calls_this_run") or 0) for row in available),
        "cache_hits": sum(bool((row["methods"][method].get("cache") or {}).get("hit")) for row in available),
        "mean_latency_ms": round(statistics.mean(float(row["methods"][method].get("latency_ms") or 0) for row in available), 2) if available else None,
        "estimated_cost_usd": round(sum(known_costs), 6) if known_costs else None,
        "estimated_cost_usd_this_run": round(sum(float(row["methods"][method].get("estimated_cost_usd_this_run") or 0.0) for row in available), 6),
        "by_task": by_task,
    }


def _paired_comparison(
    rows: list[dict[str, Any]],
    method: str,
    *,
    seed: int,
    reference_method: str = "small_direct_span",
) -> dict[str, Any]:
    paired = [
        row for row in rows
        if isinstance(row["methods"].get(reference_method, {}).get("scores"), dict)
        and isinstance(row["methods"].get(method, {}).get("scores"), dict)
    ]
    diffs = [
        row["methods"][method]["scores"]["char_f1"]
        - row["methods"][reference_method]["scores"]["char_f1"]
        for row in paired
    ]
    rng = random.Random(seed)
    samples: list[float] = []
    if diffs:
        for _ in range(2000):
            samples.append(statistics.mean(diffs[rng.randrange(len(diffs))] for _ in diffs) * 100)
        samples.sort()
        lo = samples[int(0.025 * (len(samples) - 1))]
        hi = samples[int(0.975 * (len(samples) - 1))]
    else:
        lo = hi = 0.0
    improved = sum(diff > 1e-12 for diff in diffs)
    regressed = sum(diff < -1e-12 for diff in diffs)
    return {
        "reference": reference_method,
        "method": method,
        "n": len(paired),
        "mean_case_char_f1_delta_percentage_points": round(statistics.mean(diffs) * 100, 2) if diffs else 0.0,
        "bootstrap_95ci_percentage_points": [round(lo, 2), round(hi, 2)],
        "cases_improved": improved,
        "cases_regressed": regressed,
        "net_improved_cases": improved - regressed,
    }


def _report_html(result: dict[str, Any]) -> str:
    rows = []
    comparisons = result.get("paired_vs_small_direct") or {}
    for method, stats in result["method_summaries"].items():
        comparison = comparisons.get(method, {})
        rows.append(
            "<tr>"
            f"<td>{method}</td><td>{stats['n']}</td><td>{stats['char_f1_percent']}%</td>"
            f"<td>{stats['mean_span_f1_iou50_percent']}%</td><td>{stats['response_f1_percent']}%</td>"
            f"<td>{stats['clean_false_positive_rate_percent']}%</td>"
            f"<td>{comparison.get('mean_case_char_f1_delta_percentage_points', '—')}</td>"
            f"<td>{stats['total_tokens']}</td><td>{stats.get('estimated_cost_usd')}</td>"
            f"<td>{stats['cache_hits']}</td></tr>"
        )
    return f"""<!doctype html><meta charset='utf-8'><title>RAGTruth lightweight localization</title>
<style>body{{font-family:Arial,sans-serif;margin:28px;color:#172033}}table{{border-collapse:collapse;width:100%;margin:12px 0 28px}}th,td{{border:1px solid #d7dee9;padding:8px;text-align:left}}th{{background:#f3f6fa}}.note{{background:#eef6ff;border:1px solid #bfdbfe;padding:12px;border-radius:10px;line-height:1.5}}code{{background:#f4f4f5;padding:2px 5px}}</style>
<h1>RAGTruth hallucination span localization · lightweight graph v033</h1>
<p class='note'>This evaluates where the answer is hallucinated. The compact graph condition uses one small-model call to create atomic claim nodes and evidence edges, then projects each node's minimal problem_text into an answer span. The unseen QA pilot requires complete path-labelled evidence.</p>
<p>Cases: <b>{result['summary']['completed_cases']}</b> · Small model: <b>{result['settings']['small_model']}</b> · Dataset: official RAGTruth test split.</p>
<table><thead><tr><th>Method</th><th>N</th><th>Character F1</th><th>Span F1 @ IoU .5</th><th>Response F1</th><th>Clean FP</th><th>Δ mean case char F1</th><th>Tokens</th><th>Est. USD</th><th>Cache hits</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<p class='note'>Primary metric: micro character-level F1 against the human-annotated RAGTruth spans. Evidence-edge accuracy is not claimed because the original corpus does not provide gold evidence links. Re-running unchanged cases uses the persistent cache and makes no new API call.</p>"""


def run_ragtruth_localization(
    *,
    response_path: Path,
    source_path: Path,
    output_root: Path,
    small_model: str = "gpt-5.4-nano",
    reference_model: str = "gpt-5.4-mini",
    include_reference: bool = True,
    include_checklist: bool = False,
    split: str = "test",
    quality: str = "good",
    task_types: list[str] | None = None,
    limit: int = 24,
    seed: int = 2032,
    reasoning_effort: str = "low",
    max_output_tokens: int = 1800,
    max_context_chars: int = 16_000,
    max_response_chars: int = 3000,
    include_implicit_true: bool = True,
    exclude_case_ids: set[str] | None = None,
    include_case_ids: set[str] | None = None,
    require_full_evidence: bool = False,
    generation_cache_path: Path | None = None,
    force_methods: list[str] | None = None,
    client: Any = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    _load_local_env()
    progress = progress or (lambda _message: None)
    if reasoning_effort not in ALLOWED_REASONING_EFFORTS:
        raise ValueError("reasoning_effort must be low, medium, or high")
    if not response_path.exists() or not source_path.exists():
        raise FileNotFoundError("RAGTruth files are missing. Run with --download first.")
    force = set(force_methods or [])
    methods = ["small_direct_span", "small_claim_graph"]
    if include_checklist:
        methods.insert(1, "small_checklist_span")
    if include_reference:
        methods.append("reference_direct_span")
    unknown_force = force - set(methods)
    if unknown_force:
        raise ValueError(f"Cannot force inactive methods: {sorted(unknown_force)}")

    cases, sampling = load_ragtruth_cases(
        response_path,
        source_path,
        split=split,
        quality=quality,
        task_types=task_types,
        limit=limit,
        seed=seed,
        max_response_chars=max_response_chars,
        include_implicit_true=include_implicit_true,
        exclude_case_ids=exclude_case_ids,
        require_full_evidence=require_full_evidence,
        max_context_chars=max_context_chars,
    )
    cache = _load_cache(generation_cache_path)
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

    run_id = f"ragtruth_localization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    cache_hits = Counter()
    actual_api_calls = 0

    for index, case in enumerate(cases, 1):
        progress(
            f"[{index}/{len(cases)}] {case['case_id']} · {case['task_type']} · "
            f"{'hallucinated' if case['gold_has_hallucination'] else 'clean'}"
        )
        evidence_card = build_evidence_card(
            case["source_info"],
            case["response"],
            max_context_chars=max_context_chars,
            force_full=require_full_evidence,
        )
        row = {
            "case_index": index,
            **{key: value for key, value in case.items() if key != "source_info"},
            "evidence_card": evidence_card,
            "methods": {},
        }
        for method in methods:
            model = reference_model if method == "reference_direct_span" else small_model
            token_budget = max(max_output_tokens, 3200) if method == "small_claim_graph" else max_output_tokens
            key = _cache_key(
                case,
                evidence_card,
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
                        evidence_card,
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
                except Exception as exc:
                    output = {
                        "method": method,
                        "model": model,
                        "status": "error",
                        "predicted_spans": [],
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
                output["scores"] = score_predictions(
                    case["response"],
                    list(output.get("predicted_spans") or []),
                    list(case["gold_labels"]),
                )
            row["methods"][method] = output
        rows.append(row)
        with (run_dir / "cases.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summaries = {method: _method_summary(rows, method) for method in methods}
    paired = {
        method: _paired_comparison(rows, method, seed=seed)
        for method in methods
        if method != "small_direct_span"
    }
    graph_vs_reference = None
    if include_reference:
        graph_vs_reference = _paired_comparison(
            rows,
            "small_claim_graph",
            seed=seed,
            reference_method="reference_direct_span",
        )
    result = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": {
            "name": "RAGTruth",
            "response_path": str(response_path),
            "source_path": str(source_path),
            "official_repository": "ParticleMedia/RAGTruth",
            "human_span_annotations": True,
        },
        "settings": {
            "small_model": small_model,
            "reference_model": reference_model if include_reference else None,
            "include_reference": include_reference,
            "include_checklist": include_checklist,
            "split": split,
            "quality": quality,
            "task_types": task_types or ["QA", "Summary", "Data2txt"],
            "limit": limit,
            "seed": seed,
            "reasoning_effort": reasoning_effort,
            "max_output_tokens_direct": max_output_tokens,
            "max_output_tokens_graph": max(max_output_tokens, 3200),
            "max_context_chars": max_context_chars,
            "max_response_chars": max_response_chars,
            "include_implicit_true": include_implicit_true,
            "require_full_evidence": require_full_evidence,
            "excluded_case_ids_count": len(exclude_case_ids or set()),
            "lightweight_graph": "one small-model call; compact claim nodes plus evidence edges; minimal problem_text projection",
        },
        "sampling": sampling,
        "method_summaries": summaries,
        "paired_vs_small_direct": paired,
        "paired_graph_vs_reference_direct": graph_vs_reference,
        "cache_summary": {
            "generation_cache_path": str(generation_cache_path) if generation_cache_path else None,
            "cache_hits_by_method": dict(cache_hits),
            "actual_api_calls_this_run": actual_api_calls,
        },
        "summary": {
            "completed_cases": len(rows),
            "failed_method_calls": len(failures),
            "primary_method": "small_claim_graph",
            "primary_char_f1_percent": summaries.get("small_claim_graph", {}).get("char_f1_percent"),
            "direct_char_f1_percent": summaries.get("small_direct_span", {}).get("char_f1_percent"),
            "primary_delta_mean_case_char_f1_pp": paired.get("small_claim_graph", {}).get("mean_case_char_f1_delta_percentage_points"),
            "graph_vs_reference_delta_mean_case_char_f1_pp": (
                graph_vs_reference or {}
            ).get("mean_case_char_f1_delta_percentage_points"),
            "actual_api_calls_this_run": actual_api_calls,
        },
        "cases": rows,
    }
    (run_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "failures.json").write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "report.html").write_text(_report_html(result), encoding="utf-8")
    (run_dir / "status.json").write_text(json.dumps({
        "status": "completed" if not failures else "completed_with_failures",
        "run_id": run_id,
        "completed_cases": len(rows),
        "failed_method_calls": len(failures),
        "actual_api_calls_this_run": actual_api_calls,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return result

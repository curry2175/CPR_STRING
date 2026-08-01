from __future__ import annotations

import copy
import hashlib
import json
import os
import random
import re
import statistics
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Literal

from pydantic import BaseModel, Field

from .openai_runner import ALLOWED_REASONING_EFFORTS, _load_local_env, _usage_dict
from .ragtruth_localization import (
    DirectSpanOutput,
    _answer_block,
    _estimated_cost,
    _paired_comparison,
    _parse_output,
    _predictions_from_direct,
    _stable_hash,
    build_evidence_card,
    load_ragtruth_cases,
    locate_exact_quote,
    score_predictions,
)


SCHEMA_VERSION = "0.50.0"
CACHE_SCHEMA_VERSION = "0.50.0-ragtruth-threshold-optimizer-cache"
RAW_METHOD = "nano_raw_direct"
GRAPH_METHOD = "nano_dual_graph"
METHODS = [RAW_METHOD, GRAPH_METHOD]
PROMPT_VERSIONS = {
    "raw_direct": "v040-raw-direct-minimal-span-nano",
    "source_graph": "v043-source-evidence-graph-task-complete",
    "response_graph": "v043-response-balanced-complete-proposition-graph",
    "alignment": "v049-dual-graph-balanced-recall-alignment",
}

ALIGNMENT_PROMPT_VERSIONS = {
    # This exact value preserves cache compatibility with v046. Use it to test
    # a gate-only change without paying for new Alignment calls.
    "v046_cached": "v046-dual-graph-conservative-factuality-gated-alignment",
    # This prompt is less likely to suppress a concrete material unsupported
    # component merely because the surrounding proposition is partly supported.
    "v049_recall": "v049-dual-graph-balanced-recall-alignment",
}

ALIGNMENT_GATE_PROFILES = {
    "v046_conservative": {
        "thresholds": {
            "contradicted_by": 0.55,
            "partially_supported_by": 0.72,
            "qualified_by": 0.75,
            "not_found_in_source": 0.78,
            "requires_assumption": 0.85,
        },
        "infer_error_label": False,
        "partial_span_mode": "core",
    },
    "v049_balanced_recall": {
        "thresholds": {
            "contradicted_by": 0.0,
            "partially_supported_by": 0.50,
            "qualified_by": 0.55,
            "not_found_in_source": 0.58,
            "requires_assumption": 0.72,
        },
        "infer_error_label": True,
        "partial_span_mode": "core",
    },
    "v049_clause_recall": {
        "thresholds": {
            "contradicted_by": 0.0,
            "partially_supported_by": 0.45,
            "qualified_by": 0.50,
            "not_found_in_source": 0.55,
            "requires_assumption": 0.68,
        },
        "infer_error_label": True,
        "partial_span_mode": "claim",
    },
}


class SourceGraphNode(BaseModel):
    id: str = Field(description="Sequential source node id, such as S1")
    node_type: Literal[
        "source_fact",
        "definition",
        "constraint",
        "quantitative_fact",
        "qualified_fact",
    ]
    text: str = Field(description="Canonical proposition supported by the source")
    evidence_ids: list[str] = Field(
        default_factory=list,
        description="Ids of exact evidence units from which this proposition was compiled",
    )


class SourceGraphEdge(BaseModel):
    source_ids: list[str] = Field(default_factory=list)
    target_id: str
    relation: Literal["supports", "derives", "qualifies", "contradicts", "equivalent_to"]


class SourceEvidenceGraphOutput(BaseModel):
    nodes: list[SourceGraphNode] = Field(default_factory=list, max_length=80)
    edges: list[SourceGraphEdge] = Field(default_factory=list, max_length=160)
    summary: str = ""


class ResponseAtomicityCheck(BaseModel):
    single_verdict_possible: bool = Field(
        default=True,
        description="True only when the whole node can receive one supported/unsupported/contradicted verdict",
    )
    contains_multiple_independent_claims: bool = Field(
        default=False,
        description="True when different parts of the node could receive different factual verdicts",
    )
    split_required: bool = Field(
        default=False,
        description="True when this node must be replaced by smaller atomic claim nodes",
    )
    note: str = ""


class ResponseCoverageCheck(BaseModel):
    all_factual_clauses_covered: bool = Field(
        default=True,
        description="True only when every externally verifiable factual clause in the response is represented",
    )
    omitted_factual_text: list[str] = Field(default_factory=list, max_length=20)


class ResponseClaimNode(BaseModel):
    id: str = Field(description="Sequential response node id, such as R1")
    sentence_id: str = Field(description="Stable answer sentence id, such as a1")
    node_type: Literal[
        "claim",
        "quantitative_claim",
        "comparison_claim",
        "causal_claim",
        "qualified_claim",
        "conclusion",
    ]
    text: str = Field(
        description=(
            "Smallest exact contiguous quote from the response that carries one independently checkable claim. "
            "It may be a coordinated clause with an inherited subject, but never a whole multi-claim sentence by default."
        )
    )
    normalized_claim: str = Field(
        default="",
        description="Complete canonical proposition, restoring any subject inherited from sentence context",
    )
    inherited_context: str = Field(
        default="",
        description="Shared subject or context needed to interpret the exact quote; do not add new factual content",
    )
    claim_form: Literal[
        "complete_sentence",
        "independent_clause",
        "shared_subject_clause",
    ] = "independent_clause"
    evaluation_eligible: bool = Field(
        default=True,
        description=(
            "False only for accidental non-propositional material. Discourse markers, headings, and isolated fragments "
            "must not be emitted as eligible factual claim nodes."
        ),
    )
    atomicity_check: ResponseAtomicityCheck = Field(default_factory=ResponseAtomicityCheck)


class ResponseClaimEdge(BaseModel):
    source_ids: list[str] = Field(default_factory=list)
    target_id: str
    relation: Literal["supports", "derives", "qualifies", "causes", "answers", "same_claim_as"]


class ResponseClaimGraphOutput(BaseModel):
    nodes: list[ResponseClaimNode] = Field(default_factory=list, max_length=120)
    edges: list[ResponseClaimEdge] = Field(default_factory=list, max_length=200)
    coverage_check: ResponseCoverageCheck = Field(default_factory=ResponseCoverageCheck)
    summary: str = ""


class ResponseClaimReplacement(BaseModel):
    original_node_id: str
    replacement_nodes: list[ResponseClaimNode] = Field(default_factory=list, max_length=8)


class ResponseClaimRepairOutput(BaseModel):
    replacements: list[ResponseClaimReplacement] = Field(default_factory=list, max_length=30)
    added_nodes: list[ResponseClaimNode] = Field(default_factory=list, max_length=30)
    drop_node_ids: list[str] = Field(default_factory=list, max_length=30)
    all_factual_clauses_covered: bool = True
    remaining_omitted_factual_text: list[str] = Field(default_factory=list, max_length=20)
    note: str = ""


class AlignmentRecord(BaseModel):
    response_node_id: str
    source_node_ids: list[str] = Field(default_factory=list)
    relation: Literal[
        "supported_by",
        "safe_inference",
        "not_factual",
        "generic_advice",
        "uncertain",
        "partially_supported_by",
        "contradicted_by",
        "not_found_in_source",
        "qualified_by",
        "requires_assumption",
    ]
    problem_text: str = Field(
        default="",
        description=(
            "For a problematic response node, the smallest exact contiguous response substring that expresses the "
            "unsupported or contradictory content. Empty for supported nodes."
        ),
    )
    label_type: Literal["none", "unsupported", "contradiction"] = "none"
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    explanation: str = ""


class DualGraphAlignmentOutput(BaseModel):
    alignments: list[AlignmentRecord] = Field(default_factory=list, max_length=120)
    summary: str = ""


def _empty_cache() -> dict[str, Any]:
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "raw_direct": {},
        "source_graph": {},
        "response_graph": {},
        "alignment": {},
    }


def _load_cache(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return _empty_cache()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return _empty_cache()
    if not isinstance(payload, dict):
        return _empty_cache()
    payload["schema_version"] = CACHE_SCHEMA_VERSION
    for bucket in ("raw_direct", "source_graph", "response_graph", "alignment"):
        payload.setdefault(bucket, {})
    return payload


def _save_cache(path: Path | None, cache: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def _component_key(bucket: str, payload: dict[str, Any], *, prompt_version: str | None = None) -> str:
    version = prompt_version or PROMPT_VERSIONS[bucket]
    return _stable_hash({"bucket": bucket, "prompt_version": version, **payload})


def _call_parsed(
    client: Any,
    *,
    model: str,
    reasoning_effort: str,
    max_output_tokens: int,
    system: str,
    user: str,
    output_type: type[BaseModel],
) -> dict[str, Any]:
    started = time.perf_counter()
    total_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    response_ids: list[str] = []
    errors: list[dict[str, str]] = []
    for attempt in range(2):
        token_budget = max_output_tokens if attempt == 0 else max(max_output_tokens + 600, int(max_output_tokens * 1.5))
        retry_suffix = "" if attempt == 0 else " Return concise valid structured output only; omit optional explanation text."
        try:
            response = client.responses.parse(
                model=model,
                reasoning={"effort": reasoning_effort},
                max_output_tokens=token_budget,
                store=False,
                input=[
                    {"role": "system", "content": system + retry_suffix},
                    {"role": "user", "content": user},
                ],
                text_format=output_type,
            )
            usage = _usage_dict(response)
            for key in total_usage:
                total_usage[key] += int(usage.get(key) or 0)
            response_ids.append(str(getattr(response, "id", "")))
            parsed = _parse_output(response, output_type)
            return {
                "status": "ok",
                "parsed": parsed.model_dump(),
                "usage": total_usage,
                "latency_ms": round((time.perf_counter() - started) * 1000, 3),
                "response_id": response_ids[-1] if response_ids else "",
                "response_ids": response_ids,
                "model_returned": str(getattr(response, "model", model)),
                "estimated_cost_usd": _estimated_cost(model, total_usage),
                "api_calls": attempt + 1,
                "retry_count": attempt,
                "retry_errors": errors,
            }
        except Exception as exc:
            errors.append({"type": type(exc).__name__, "message": str(exc)})
            if attempt == 1:
                raise
    raise RuntimeError("Structured output call failed without an exception")


def _raw_prompts(case: dict[str, Any], evidence_card: dict[str, Any]) -> tuple[str, str]:
    system = (
        "You are a fine-grained hallucination span detector. Use only the supplied SOURCE evidence and never use outside "
        "knowledge. Compare the RESPONSE directly against the SOURCE. Mark content only when it is contradicted by the "
        "SOURCE or introduces a factual proposition that the SOURCE does not support. Do not flag paraphrases, style, "
        "omissions, cautious summaries, or harmless connective text. Return the smallest exact contiguous RESPONSE "
        "substring that contains each factual error. Do not return an entire sentence when only an entity, number, unit, "
        "negation, comparison, modifier, temporal phrase, scope phrase, predicate, or clause is wrong. The chosen quote "
        "must include every word needed to express the error but exclude surrounding supported words. If separate errors "
        "occur in one sentence, return separate spans. Copy every span exactly from the RESPONSE sentence. Use "
        "label_type='contradiction' only for direct conflict; otherwise use 'unsupported'. Never extract a numeric substring "
        "from inside a larger numeric literal. Before returning a span, silently test whether correcting or removing only "
        "that substring would eliminate the factual error and whether the span can be shortened further."
    )
    _sentences, answer_block = _answer_block(case["response"])
    user = (
        f"TASK\n{case['task_instruction']}\n\n"
        f"SOURCE EVIDENCE\n{evidence_card['text']}\n\n"
        f"RESPONSE WITH STABLE SENTENCE IDS\n{answer_block}\n\n"
        "Return all and only minimal hallucination spans as structured output."
    )
    return system, user


def _source_graph_prompts(case: dict[str, Any], evidence_card: dict[str, Any]) -> tuple[str, str]:
    system = (
        "You are the SOURCE Evidence Graph Compiler. Convert the supplied SOURCE evidence into a task-complete graph of "
        "factual propositions. Do not inspect or evaluate the response and do not use outside knowledge. The graph should "
        "be compact through deduplication, not through omission: preserve every proposition that could materially support, "
        "contradict, qualify, or constrain an answer to the task. Merge true repetitions and close paraphrases, but never "
        "merge propositions that differ in entity, polarity, number, unit, comparison, condition, population, time, modality, "
        "or causal strength. Create one node per independently verifiable source proposition. Preserve negative facts, "
        "exceptions, limitations, and whether a list or definition is presented as exhaustive. Every node must cite one or "
        "more supplied evidence ids. Use only the schema vocabulary. Do not invent evidence ids or propositions."
    )
    user = (
        f"TASK\n{case['task_instruction']}\n\n"
        f"SOURCE EVIDENCE WITH STABLE IDS\n{evidence_card['text']}\n\n"
        "Compile the task-complete SOURCE Evidence Graph. Deduplicate repeated evidence, but do not discard a relevant "
        "qualifier or proposition merely to make the graph shorter."
    )
    return system, user

def _response_graph_prompts(case: dict[str, Any]) -> tuple[str, str]:
    system = (
        "You are the RESPONSE Balanced Claim Graph Compiler. A sentence is a container, not automatically a node, but a "
        "node is also not the shortest possible text fragment. Convert the RESPONSE into factual claim nodes without seeing "
        "or using the source and without deciding whether any claim is correct. Each eligible node must express the smallest "
        "semantically complete proposition or clause that can receive one supported, unsupported, or contradicted verdict. "
        "Split a sentence only when its parts could genuinely receive different factual verdicts. Keep together a subject, "
        "predicate, object, and all meaning-essential qualifiers such as negation, number and unit, comparator, population, "
        "condition, time, modality, quantifier, and causal strength. For coordinated clauses with a shared subject, node.text "
        "may omit the repeated subject only when it still contains a predicate and its necessary complement; normalized_claim "
        "must restore the subject and inherited_context must record it. Do not emit isolated entities, isolated numbers, bare "
        "adjectives, headings, discourse markers, transitions, or rhetorical phrases as factual nodes. Words such as "
        "'however', 'finally', 'therefore', and 'in conclusion' belong to sentence context, not separate claim nodes. Do not "
        "treat a hedge such as 'evidence is limited' as an independent factual claim unless the response materially asserts "
        "the state or quality of evidence. Do not split merely because a sentence contains 'and', 'but', a relative clause, "
        "or multiple nouns. Preserve a combined expression when splitting would destroy one coherent proposition. Extract "
        "every externally checkable factual proposition, including secondary claims, but do not promote non-factual connective "
        "language into nodes. Merge only true repetitions; do not merge claims that differ in polarity, entity, number, unit, "
        "population, time, modality, comparison direction, or causal strength. Every node.text must be copied exactly and "
        "contiguously from one stable response sentence. Set evaluation_eligible=false only if accidental non-propositional "
        "material must be represented for bookkeeping; normally omit it instead. Before returning, verify that every eligible "
        "node contains a predicate or relation, normalized_claim is a complete proposition, and no eligible node could be split "
        "into parts with different factual verdicts. Set coverage_check.all_factual_clauses_covered=true only when every "
        "externally checkable proposition is represented."
    )
    _sentences, answer_block = _answer_block(case["response"])
    user = (
        f"TASK\n{case['task_instruction']}\n\n"
        f"RESPONSE WITH STABLE SENTENCE IDS\n{answer_block}\n\n"
        "Compile the balanced RESPONSE Claim Graph. Use the smallest complete proposition, not the shortest lexical span. "
        "Example: 'The trial enrolled 120 patients and improved survival' may become 'The trial enrolled 120 patients' and "
        "'improved survival' normalized as 'The trial improved survival'. By contrast, do not create nodes such as '120', "
        "'survival', 'Water', or 'finally'. Return only the structured graph."
    )
    return system, user


_VERB_CUE_RE = re.compile(
    r"\b(?:is|are|was|were|be|been|being|has|have|had|does|do|did|can|could|may|might|must|"
    r"will|would|shall|should|shows?|showed|finds?|found|reports?|reported|suggests?|suggested|"
    r"indicates?|indicated|demonstrates?|demonstrated|includes?|included|enrolls?|enrolled|"
    r"reduces?|reduced|increases?|increased|improves?|improved|causes?|caused|predicts?|predicted|"
    r"associates?|associated|differs?|differed|occurs?|occurred|remains?|remained|allows?|allowed|"
    r"cuts?|cut|uses?|used|leads?|led|makes?|made|gets?|got|becomes?|became|builds?|built|"
    r"watches?|watched|breeds?|bred|possesses?|possessed|\w+(?:ed|ing))\b",
    re.IGNORECASE,
)
_COMPOUND_BOUNDARY_RE = re.compile(r"\s*(?:;|\b(?:and|but|while|whereas)\b|,\s*(?:which|who|whereas|while)\b)\s*", re.IGNORECASE)


def _looks_like_multi_claim_text(text: str) -> bool:
    """Conservatively flag a likely multi-predicate claim for one compiler refinement pass."""
    value = " ".join(str(text or "").split())
    if not value or not _COMPOUND_BOUNDARY_RE.search(value):
        return False
    parts = [piece.strip(" ,;") for piece in _COMPOUND_BOUNDARY_RE.split(value) if piece.strip(" ,;")]
    verb_bearing = sum(bool(_VERB_CUE_RE.search(piece)) for piece in parts)
    relative_clause_boundary = bool(re.search(r",\s*(?:which|who|whereas|while)\b", value, re.IGNORECASE))
    total_verb_cues = len(_VERB_CUE_RE.findall(value))
    return verb_bearing >= 2 or (relative_clause_boundary and total_verb_cues >= 2)


_DISCOURSE_MARKERS = {
    "however", "finally", "therefore", "thus", "moreover", "furthermore", "overall",
    "in conclusion", "to conclude", "in summary", "for example", "for instance",
    "first", "second", "third", "lastly", "additionally", "also", "meanwhile",
}


def _normalized_words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)?", str(text or ""))


def _is_discourse_only(text: str) -> bool:
    value = " ".join(_normalized_words(text)).lower().strip()
    return bool(value) and value in _DISCOURSE_MARKERS


def _is_incomplete_claim_fragment(node: ResponseClaimNode) -> bool:
    """Return True for lexical fragments that cannot carry a factual verdict on their own."""
    text = " ".join(str(node.text or "").split()).strip(" ,;:.!?()[]{}\"'")
    normalized = " ".join(str(node.normalized_claim or node.text or "").split())
    if not node.evaluation_eligible:
        return True
    if not text or _is_discourse_only(text):
        return True
    # The canonical proposition must contain a predicate/relation.
    if not _VERB_CUE_RE.search(normalized):
        return True
    # Shared-subject clauses may omit the subject, but must still contain the predicate.
    if not _VERB_CUE_RE.search(text):
        return True
    words = _normalized_words(text)
    if len(words) == 1:
        return True
    return False


def _response_graph_diagnostics(parsed: ResponseClaimGraphOutput, response: str) -> dict[str, Any]:
    nodes, unresolved = _resolve_response_graph(parsed, response)
    unresolved_ids = [str(item.get("id") or "") for item in unresolved if str(item.get("id") or "")]
    sentences, _block = _answer_block(response)
    sentence_map = {sentence.id: sentence for sentence in sentences}
    node_ids_by_sentence: dict[str, list[str]] = {sentence.id: [] for sentence in sentences}
    whole_sentence_nodes = 0
    explicitly_non_atomic: list[str] = []
    heuristically_compound: list[str] = []
    incomplete_fragments: list[str] = []
    non_eligible_nodes: list[str] = []
    duplicate_ids: list[str] = []
    seen_ids: set[str] = set()
    node_lengths: list[int] = []
    for node in parsed.nodes:
        node_id = str(node.id or "")
        if node_id in seen_ids:
            duplicate_ids.append(node_id)
        seen_ids.add(node_id)
        node_ids_by_sentence.setdefault(node.sentence_id, []).append(node_id)
        check = node.atomicity_check
        if (not check.single_verdict_possible) or check.contains_multiple_independent_claims or check.split_required:
            explicitly_non_atomic.append(node_id)
        if _looks_like_multi_claim_text(node.text):
            heuristically_compound.append(node_id)
        if not node.evaluation_eligible:
            non_eligible_nodes.append(node_id)
        if _is_incomplete_claim_fragment(node):
            incomplete_fragments.append(node_id)
        resolved = nodes.get(node_id) or {}
        start, end = resolved.get("start"), resolved.get("end")
        if isinstance(start, int) and isinstance(end, int) and end > start:
            node_lengths.append(end - start)
            sentence = sentence_map.get(node.sentence_id)
            if sentence and start == sentence.start and end == sentence.end:
                whole_sentence_nodes += 1
    sentence_containers = [
        {
            "id": sentence.id,
            "start": sentence.start,
            "end": sentence.end,
            "text": sentence.text,
            "claim_node_ids": node_ids_by_sentence.get(sentence.id, []),
        }
        for sentence in sentences
    ]
    omitted = [value for value in parsed.coverage_check.omitted_factual_text if str(value).strip()]
    coverage_failed = (not parsed.coverage_check.all_factual_clauses_covered) or bool(omitted)
    # Heuristic compound cues are retained as diagnostics only. They no longer trigger a whole-graph rewrite.
    repair_node_ids = sorted(set(
        explicitly_non_atomic
        + unresolved_ids
        + incomplete_fragments
        + duplicate_ids
        + non_eligible_nodes
    ))
    needs_refinement = bool(
        repair_node_ids
        or coverage_failed
        or (response.strip() and not parsed.nodes)
    )
    return {
        "sentence_containers": sentence_containers,
        "sentence_count": len(sentences),
        "claim_count": len(parsed.nodes),
        "mean_claim_node_chars": round(statistics.mean(node_lengths), 2) if node_lengths else 0.0,
        "claims_per_sentence": round(len(parsed.nodes) / len(sentences), 3) if sentences else 0.0,
        "whole_sentence_node_count": whole_sentence_nodes,
        "whole_sentence_node_rate": whole_sentence_nodes / len(parsed.nodes) if parsed.nodes else 0.0,
        "explicitly_non_atomic_node_ids": explicitly_non_atomic,
        "heuristically_compound_node_ids": heuristically_compound,
        "incomplete_fragment_node_ids": incomplete_fragments,
        "non_eligible_node_ids": non_eligible_nodes,
        "duplicate_node_ids": duplicate_ids,
        "unresolved_node_ids": unresolved_ids,
        "unresolved_node_count": len(unresolved),
        "coverage_failed": coverage_failed,
        "omitted_factual_text": omitted,
        "repair_node_ids": repair_node_ids,
        "needs_refinement": needs_refinement,
    }


def _response_graph_repair_prompts(
    case: dict[str, Any],
    prior: ResponseClaimGraphOutput,
    diagnostics: dict[str, Any],
) -> tuple[str, str]:
    sentences, _block = _answer_block(case["response"])
    sentence_map = {sentence.id: sentence.text for sentence in sentences}
    repair_ids = set(diagnostics.get("repair_node_ids") or [])
    problem_nodes = [node.model_dump() for node in prior.nodes if node.id in repair_ids]
    relevant_sentence_ids = {str(node.get("sentence_id") or "") for node in problem_nodes}
    relevant_sentences = {sid: sentence_map.get(sid, "") for sid in relevant_sentence_ids}
    system = (
        "You are the LOCAL RESPONSE Claim Compiler Repair Agent. Repair only the listed problematic nodes and missing "
        "coverage; do not recompile or subdivide valid nodes. For each problematic node, either replace it with one or more "
        "minimal complete propositions, or drop it when it is only a discourse marker, heading, or other non-factual fragment. "
        "A replacement node must contain a predicate or relation in node.text, and normalized_claim must be a complete factual "
        "proposition. Do not return isolated entities, numbers, adjectives, modifiers, or transition words. Split only when "
        "different parts could receive different factual verdicts. Keep essential scope, modality, negation, numbers, units, "
        "conditions, and causal strength with the relevant claim. Preserve exact sentence ids and exact contiguous quotes. "
        "Use added_nodes only for factual propositions explicitly omitted by the prior graph. Return a local patch, not a full graph."
    )
    user = (
        f"TASK\n{case['task_instruction']}\n\n"
        f"RESPONSE\n{case['response']}\n\n"
        "PROBLEMATIC NODES\n"
        + json.dumps(problem_nodes, ensure_ascii=False)
        + "\n\nRELEVANT SENTENCES\n"
        + json.dumps(relevant_sentences, ensure_ascii=False)
        + "\n\nOMITTED FACTUAL TEXT REPORTED BY PRIOR COMPILER\n"
        + json.dumps(diagnostics.get("omitted_factual_text") or [], ensure_ascii=False)
        + "\n\nReturn replacements only for the listed node ids, optional added_nodes for genuine omitted propositions, "
          "and drop_node_ids for non-factual fragments. Do not modify valid nodes."
    )
    return system, user


def _apply_response_graph_repairs(
    prior: ResponseClaimGraphOutput,
    repair: ResponseClaimRepairOutput,
) -> ResponseClaimGraphOutput:
    replacement_map = {item.original_node_id: list(item.replacement_nodes) for item in repair.replacements}
    drop_ids = set(repair.drop_node_ids)
    new_nodes: list[ResponseClaimNode] = []
    preserved_old_ids: set[str] = set()
    for node in prior.nodes:
        if node.id in drop_ids:
            continue
        if node.id in replacement_map:
            new_nodes.extend(replacement_map[node.id])
            continue
        new_nodes.append(node)
        preserved_old_ids.add(node.id)
    new_nodes.extend(repair.added_nodes)

    # Deduplicate exact repetitions while preserving semantically distinct claims.
    deduped: list[ResponseClaimNode] = []
    seen: set[tuple[str, str, str]] = set()
    for node in new_nodes:
        key = (
            str(node.sentence_id or ""),
            " ".join(str(node.text or "").split()).lower(),
            " ".join(str(node.normalized_claim or "").split()).lower(),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(node)

    old_to_new: dict[str, str] = {}
    renumbered: list[ResponseClaimNode] = []
    for index, node in enumerate(deduped, 1):
        new_id = f"R{index}"
        if node.id in preserved_old_ids:
            old_to_new[node.id] = new_id
        payload = node.model_dump()
        payload["id"] = new_id
        renumbered.append(ResponseClaimNode.model_validate(payload))

    # Preserve only edges whose endpoints were untouched and remain unambiguous.
    repaired_edges: list[ResponseClaimEdge] = []
    for edge in prior.edges:
        if edge.target_id not in old_to_new or any(source_id not in old_to_new for source_id in edge.source_ids):
            continue
        repaired_edges.append(ResponseClaimEdge(
            source_ids=[old_to_new[source_id] for source_id in edge.source_ids],
            target_id=old_to_new[edge.target_id],
            relation=edge.relation,
        ))
    coverage = ResponseCoverageCheck(
        all_factual_clauses_covered=bool(repair.all_factual_clauses_covered),
        omitted_factual_text=list(repair.remaining_omitted_factual_text),
    )
    return ResponseClaimGraphOutput(
        nodes=renumbered,
        edges=repaired_edges,
        coverage_check=coverage,
        summary=prior.summary,
    )


def _combine_component_records(first: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
    combined = copy.deepcopy(second)
    usage = {
        key: int((first.get("usage") or {}).get(key) or 0) + int((second.get("usage") or {}).get(key) or 0)
        for key in ("input_tokens", "output_tokens", "total_tokens")
    }
    combined["usage"] = usage
    combined["latency_ms"] = round(float(first.get("latency_ms") or 0.0) + float(second.get("latency_ms") or 0.0), 3)
    combined["api_calls"] = int(first.get("api_calls") or 1) + int(second.get("api_calls") or 1)
    combined["retry_count"] = int(first.get("retry_count") or 0) + int(second.get("retry_count") or 0)
    combined["retry_errors"] = list(first.get("retry_errors") or []) + list(second.get("retry_errors") or [])
    combined["response_ids"] = list(first.get("response_ids") or []) + list(second.get("response_ids") or [])
    combined["estimated_cost_usd"] = _estimated_cost(str(combined.get("model_returned") or "gpt-5.4-nano"), usage)
    return combined


def _call_response_graph_compiler(
    client: Any,
    *,
    case: dict[str, Any],
    model: str,
    reasoning_effort: str,
    max_output_tokens: int,
    system: str,
    user: str,
) -> dict[str, Any]:
    first = _call_parsed(
        client,
        model=model,
        reasoning_effort=reasoning_effort,
        max_output_tokens=max_output_tokens,
        system=system,
        user=user,
        output_type=ResponseClaimGraphOutput,
    )
    first_graph = ResponseClaimGraphOutput.model_validate(first["parsed"])
    first_diag = _response_graph_diagnostics(first_graph, case["response"])
    if not first_diag["needs_refinement"]:
        first["compiler_refinement"] = {
            "attempted": False,
            "mode": "none",
            "initial_diagnostics": first_diag,
            "final_diagnostics": first_diag,
        }
        return first

    repair_system, repair_user = _response_graph_repair_prompts(case, first_graph, first_diag)
    second = _call_parsed(
        client,
        model=model,
        reasoning_effort=reasoning_effort,
        max_output_tokens=max(900, min(max_output_tokens, 2200)),
        system=repair_system,
        user=repair_user,
        output_type=ResponseClaimRepairOutput,
    )
    repair = ResponseClaimRepairOutput.model_validate(second["parsed"])
    final_graph = _apply_response_graph_repairs(first_graph, repair)
    final_diag = _response_graph_diagnostics(final_graph, case["response"])
    combined = _combine_component_records(first, second)
    combined["parsed"] = final_graph.model_dump()
    combined["compiler_refinement"] = {
        "attempted": True,
        "mode": "local_node_patch",
        "repair_node_ids": list(first_diag.get("repair_node_ids") or []),
        "replacement_groups": len(repair.replacements),
        "added_nodes": len(repair.added_nodes),
        "dropped_nodes": len(repair.drop_node_ids),
        "initial_diagnostics": first_diag,
        "final_diagnostics": final_diag,
        "remaining_quality_warning": bool(final_diag["needs_refinement"]),
    }
    return combined

def _alignment_prompts(
    case: dict[str, Any],
    evidence_card: dict[str, Any],
    source_graph: SourceEvidenceGraphOutput,
    response_graph: ResponseClaimGraphOutput,
    *,
    alignment_prompt_profile: str = "v049_recall",
) -> tuple[str, str]:
    if alignment_prompt_profile not in ALIGNMENT_PROMPT_VERSIONS:
        raise ValueError(f"Unknown alignment prompt profile: {alignment_prompt_profile}")

    common = (
        "You are the Dual-Graph Alignment and Hallucination Localization Agent. Treat the supplied SOURCE "
        "as the authoritative task-bounded evidence and inspect the RESPONSE for factual content that is genuinely wrong, "
        "materially distorted, or newly asserted without support. Align every eligible RESPONSE node by evaluating "
        "normalized_claim as the complete proposition; use node.text only to recover the exact response location. Use only "
        "the supplied source evidence and source graph, never outside knowledge. "
        "\n\nVERDICT POLICY. Use supported_by when the source directly supports the claim. Use safe_inference when the wording is not "
        "explicitly repeated but follows safely from the supplied source through ordinary paraphrase, synthesis across supplied "
        "passages, task-bounded exclusion, or a noncontroversial linguistic inference. Use not_factual for discourse, headings, "
        "answer-organization language, epistemic framing, or other material that does not assert an externally checkable fact. "
        "Use generic_advice for task-appropriate recommendations, procedural guidance, safety advice, or conversational next "
        "steps that do not add a concrete source-dependent factual claim. Use uncertain when support genuinely cannot be resolved "
        "from the supplied material. These five verdicts are NON-HALLUCINATION verdicts and must use label_type='none' with empty "
        "problem_text. "
        "\n\nUse partially_supported_by when the proposition contains both supported material and a material unsupported or distorted "
        "entity, number, unit, polarity, scope, population, condition, time, modality, comparison, causal attribution, or causal "
        "strength. Use contradicted_by for direct incompatibility. Use not_found_in_source only when the response introduces a "
        "concrete externally verifiable proposition that is neither stated nor safely entailed by any supplied source passage. "
        "Mere absence of identical wording is never enough. Use qualified_by when omission or alteration of an essential source "
        "limitation makes the response materially misleading. Use requires_assumption only when a specific additional premise is "
        "necessary. "
        "\n\nFALSE-POSITIVE GUARD. Do not flag appropriate advice, recommendations, procedural steps, rhetorical summaries, cautious "
        "phrasing, answer synthesis, or meta-level language merely because they are not quoted in the source. Do not flag a claim "
        "when the same meaning is distributed across multiple source units. Before using not_found_in_source, check the exact "
        "source passages as well as the compiled source nodes and ask whether the claim is a safe paraphrase or synthesis. "
        "\n\nLOCALIZATION. For a genuinely problematic claim, copy problem_text exactly from the RESPONSE. Return the smallest "
        "semantically complete contiguous substring that expresses the unsupported or contradictory content with its predicate "
        "and required arguments. For a direct substitution or contradiction, the materially wrong value, entity, number, polarity, "
        "scope, or qualifier may be returned alone when it uniquely identifies the error. Never return only a transition word or "
        "harmless modifier."
    )

    if alignment_prompt_profile == "v046_cached":
        policy = (
            " Be conservative. When uncertain, prefer uncertain over unsupported. Unsupported or partial claims below high "
            "confidence should normally be uncertain with no submitted span. Only problematic verdicts may use label_type "
            "unsupported or contradiction."
        )
    else:
        policy = (
            " Balance specificity and recall. Do not downgrade a clearly identified material unsupported component to uncertain "
            "merely because another part of the same proposition is supported. If a concrete unsupported component is present, "
            "use partially_supported_by with label_type='unsupported' and exact problem_text. Use contradicted_by with "
            "label_type='contradiction'. Use not_found_in_source or qualified_by with label_type='unsupported' when their criteria "
            "are met. Confidence reflects certainty in the localized defect, not whether the entire surrounding sentence is wrong. "
            "A supported clause next to an unsupported causal, quantitative, comparative, or scoped addition does not erase the "
            "unsupported addition."
        )

    system = common + policy
    user = (
        f"TASK\n{case['task_instruction']}\n\n"
        f"SOURCE EVIDENCE\n{evidence_card['text']}\n\n"
        f"RESPONSE\n{case['response']}\n\n"
        f"SOURCE EVIDENCE GRAPH\n{json.dumps(source_graph.model_dump(), ensure_ascii=False)}\n\n"
        f"RESPONSE CLAIM GRAPH\n{json.dumps(response_graph.model_dump(), ensure_ascii=False)}\n\n"
        "Return one alignment for every eligible response node. First decide whether the node is factual and source-dependent. "
        "Then distinguish direct support, safe inference, generic advice, uncertainty, partial distortion, contradiction, and "
        "genuinely unsupported new factual content. For every material error relation, identify the exact unsupported or "
        "contradictory response substring rather than suppressing the finding because the rest of the node is supported."
    )
    return system, user

def _resolve_response_graph(parsed: ResponseClaimGraphOutput, response: str) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    resolved: dict[str, dict[str, Any]] = {}
    unresolved: list[dict[str, Any]] = []
    for index, item in enumerate(parsed.nodes, 1):
        node_id = item.id.strip() or f"R{index}"
        location = locate_exact_quote(response, item.text, item.sentence_id)
        node = item.model_dump()
        node["id"] = node_id
        if location is None:
            node["start"] = None
            node["end"] = None
            unresolved.append({**node, "unresolved_field": "text"})
        else:
            node["start"], node["end"] = location
            node["resolved_text"] = response[location[0]:location[1]]
        resolved[node_id] = node
    return resolved, unresolved


def _locate_problem_text(response: str, node: dict[str, Any], problem_text: str) -> tuple[int, int] | None:
    if not problem_text.strip():
        return None
    start = node.get("start")
    end = node.get("end")
    if isinstance(start, int) and isinstance(end, int) and end > start:
        region = response[start:end]
        relative = locate_exact_quote(region, problem_text)
        if relative is not None:
            return start + relative[0], start + relative[1]
    return locate_exact_quote(response, problem_text, str(node.get("sentence_id") or ""))


def _problem_text_action(problem_text: str, relation: str = "") -> str:
    """Classify localization text as keep, expand_to_claim, or discard."""
    text = " ".join(str(problem_text or "").split()).strip(" ,;:.!?()[]{}\"'")
    if not text or _is_discourse_only(text):
        return "discard"
    words = _normalized_words(text)
    if not words:
        return "discard"
    numeric_only = bool(re.fullmatch(r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:\s*%|\s*[A-Za-z]+)?", text))
    if numeric_only:
        return "keep"
    # A bare entity/modifier is too small to express the unsupported proposition.
    if len(words) <= 2 and not _VERB_CUE_RE.search(text):
        # A direct contradiction may differ by exactly one entity/value (e.g., Paris vs Lyon).
        if relation == "contradicted_by":
            return "keep"
        return "expand_to_claim"
    return "keep"


def _node_has_complete_proposition(node: dict[str, Any]) -> bool:
    text = str(node.get("resolved_text") or node.get("text") or "")
    normalized = str(node.get("normalized_claim") or text)
    if _is_discourse_only(text):
        return False
    return bool(_VERB_CUE_RE.search(text) and _VERB_CUE_RE.search(normalized))


def _predictions_from_alignment(
    alignment: DualGraphAlignmentOutput,
    response_graph: ResponseClaimGraphOutput,
    response: str,
    *,
    gate_profile: str = "v046_conservative",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if gate_profile not in ALIGNMENT_GATE_PROFILES:
        raise ValueError(f"Unknown alignment gate profile: {gate_profile}")
    gate_config = ALIGNMENT_GATE_PROFILES[gate_profile]
    nodes, unresolved = _resolve_response_graph(response_graph, response)
    predictions: list[dict[str, Any]] = []
    resolved_alignments: list[dict[str, Any]] = []
    seen: set[tuple[int, int, str]] = set()
    fallback_count = 0
    expanded_count = 0
    discarded_nonclaim_count = 0
    for item in alignment.alignments:
        record = item.model_dump()
        node = nodes.get(item.response_node_id)
        record["response_node"] = node
        if node is None:
            unresolved.append({**record, "unresolved_field": "response_node_id"})
            resolved_alignments.append(record)
            continue
        if not bool(node.get("evaluation_eligible", True)):
            record["ignored_noneligible_node"] = True
            resolved_alignments.append(record)
            continue
        non_hallucination_relations = {
            "supported_by", "safe_inference", "not_factual", "generic_advice", "uncertain"
        }
        relation_thresholds = dict(gate_config["thresholds"])
        if item.relation in non_hallucination_relations:
            record["submission_gate"] = "non_hallucination_relation"
            record["gate_profile"] = gate_profile
            resolved_alignments.append(record)
            continue
        threshold = float(relation_thresholds.get(item.relation, 0.80))
        # Backward-compatible structured records that omit confidence are treated as uncalibrated, not low-confidence.
        effective_confidence = float(item.confidence) if "confidence" in item.model_fields_set else 1.0
        expected_label = "contradiction" if item.relation == "contradicted_by" else "unsupported"
        has_explicit_label = item.label_type in {"unsupported", "contradiction"}
        has_localized_problem = bool(str(item.problem_text or "").strip())
        inferred_label = bool(gate_config.get("infer_error_label")) and has_localized_problem
        problematic = (has_explicit_label or inferred_label) and effective_confidence >= threshold
        if not problematic:
            record["submission_gate"] = "suppressed_low_confidence_or_no_error_label"
            record["submission_threshold"] = threshold
            record["effective_confidence"] = effective_confidence
            record["gate_profile"] = gate_profile
            record["expected_error_label"] = expected_label
            record["inferred_error_label"] = inferred_label and not has_explicit_label
            resolved_alignments.append(record)
            continue
        record["submission_gate"] = "emitted"
        record["submission_threshold"] = threshold
        record["effective_confidence"] = effective_confidence
        record["gate_profile"] = gate_profile
        record["expected_error_label"] = expected_label
        record["inferred_error_label"] = inferred_label and not has_explicit_label

        action = _problem_text_action(item.problem_text, item.relation)
        if action == "discard":
            discarded_nonclaim_count += 1
            record["discarded_nonclaim_problem_text"] = True
            resolved_alignments.append(record)
            continue

        location = _locate_problem_text(response, node, item.problem_text)
        used_fallback = False
        expanded_to_claim = False
        if action == "expand_to_claim" and _node_has_complete_proposition(node):
            if isinstance(node.get("start"), int) and isinstance(node.get("end"), int):
                location = (int(node["start"]), int(node["end"]))
                expanded_to_claim = True
                expanded_count += 1
        elif (
            gate_config.get("partial_span_mode") == "claim"
            and item.relation in {"partially_supported_by", "qualified_by", "requires_assumption"}
            and _node_has_complete_proposition(node)
            and isinstance(node.get("start"), int)
            and isinstance(node.get("end"), int)
        ):
            location = (int(node["start"]), int(node["end"]))
            expanded_to_claim = True
            expanded_count += 1
        if location is None and _node_has_complete_proposition(node):
            if isinstance(node.get("start"), int) and isinstance(node.get("end"), int):
                location = (int(node["start"]), int(node["end"]))
                used_fallback = True
                fallback_count += 1
        if location is None:
            unresolved.append({**record, "unresolved_field": "problem_text"})
            resolved_alignments.append(record)
            continue
        start, end = location
        label_type = "contradiction" if item.relation == "contradicted_by" or item.label_type == "contradiction" else "unsupported"
        key = (start, end, label_type)
        if key not in seen:
            seen.add(key)
            predictions.append({
                "start": start,
                "end": end,
                "text": response[start:end],
                "label_type": label_type,
                "sentence_id": str(node.get("sentence_id") or ""),
                "response_node_id": item.response_node_id,
                "response_node_text": node.get("resolved_text") or node.get("text"),
                "normalized_claim": node.get("normalized_claim") or node.get("text"),
                "alignment_relation": item.relation,
                "source_node_ids": list(item.source_node_ids),
                "problem_text_requested": item.problem_text,
                "problem_text_expanded_to_complete_claim": expanded_to_claim,
                "problem_text_fallback_to_node": used_fallback,
                "confidence": item.confidence,
                "explanation": item.explanation,
            })
        record["problem_start"] = start
        record["problem_end"] = end
        record["resolved_problem_text"] = response[start:end]
        record["problem_text_expanded_to_complete_claim"] = expanded_to_claim
        record["problem_text_fallback_to_node"] = used_fallback
        resolved_alignments.append(record)
    diagnostics = _response_graph_diagnostics(response_graph, response)
    return predictions, {
        "source_graph": None,
        "response_graph": {
            "nodes": list(nodes.values()),
            "edges": [edge.model_dump() for edge in response_graph.edges],
            "coverage_check": response_graph.coverage_check.model_dump(),
            "sentence_containers": diagnostics["sentence_containers"],
        },
        "response_graph_diagnostics": diagnostics,
        "alignments": resolved_alignments,
        "unresolved_claims": unresolved,
        "claim_count": len(response_graph.nodes),
        "predicted_count": len(predictions),
        "problem_text_fallback_count": fallback_count,
        "problem_text_expanded_to_complete_claim_count": expanded_count,
        "discarded_nonclaim_problem_text_count": discarded_nonclaim_count,
        "alignment_gate_profile": gate_profile,
        "alignment_gate_config": copy.deepcopy(gate_config),
    }

def _catch_reason(raw_scores: dict[str, Any] | None, graph_scores: dict[str, Any] | None) -> str | None:
    if not raw_scores or not graph_scores or not graph_scores.get("gold_has_hallucination"):
        return None
    raw_detected = bool(raw_scores.get("predicted_has_hallucination"))
    graph_detected = bool(graph_scores.get("predicted_has_hallucination"))
    raw_f1 = float(raw_scores.get("char_f1") or 0.0)
    graph_f1 = float(graph_scores.get("char_f1") or 0.0)
    raw_recall = float(raw_scores.get("char_recall") or 0.0)
    graph_recall = float(graph_scores.get("char_recall") or 0.0)
    if (not raw_detected) and graph_detected:
        return "raw_detection_miss_dual_hit"
    if graph_detected and graph_f1 >= raw_f1 + 0.20 and graph_recall >= raw_recall + 0.20:
        return "dual_material_uplift"
    return None


def _catch_candidate_payload(row: dict[str, Any], reason: str) -> dict[str, Any]:
    raw = (row.get("methods") or {}).get(RAW_METHOD) or {}
    graph = (row.get("methods") or {}).get(GRAPH_METHOD) or {}
    details = graph.get("details") or {}
    return {
        "schema_version": SCHEMA_VERSION,
        "case_id": row.get("case_id"),
        "source_id": row.get("source_id"),
        "task_type": row.get("task_type"),
        "task_instruction": row.get("task_instruction"),
        "response": row.get("response"),
        "gold_labels": row.get("gold_labels") or [],
        "gold_has_hallucination": row.get("gold_has_hallucination"),
        "catch_reason": reason,
        "evidence_card": row.get("evidence_card") or {},
        "raw": {
            "predicted_spans": raw.get("predicted_spans") or [],
            "scores": raw.get("scores") or {},
        },
        "balanced_dual_graph": {
            "predicted_spans": graph.get("predicted_spans") or [],
            "scores": graph.get("scores") or {},
            "source_graph": details.get("source_graph") or {},
            "response_graph": details.get("response_graph") or {},
            "response_graph_diagnostics": details.get("response_graph_diagnostics") or {},
            "alignments": details.get("alignments") or [],
            "response_compiler_refinement": details.get("response_compiler_refinement") or {},
        },
    }


def _save_catch_candidate(catch_dir: Path, row: dict[str, Any], reason: str) -> dict[str, Any]:
    catch_dir.mkdir(parents=True, exist_ok=True)
    cases_dir = catch_dir / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)
    payload = _catch_candidate_payload(row, reason)
    case_path = cases_dir / f"{payload['case_id']}.json"
    case_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    raw_scores = payload["raw"]["scores"]
    graph_scores = payload["balanced_dual_graph"]["scores"]
    return {
        "case_id": payload["case_id"],
        "reason": reason,
        "raw_char_f1_percent": round(float(raw_scores.get("char_f1") or 0.0) * 100, 2),
        "dual_char_f1_percent": round(float(graph_scores.get("char_f1") or 0.0) * 100, 2),
        "raw_char_recall_percent": round(float(raw_scores.get("char_recall") or 0.0) * 100, 2),
        "dual_char_recall_percent": round(float(graph_scores.get("char_recall") or 0.0) * 100, 2),
        "file": str(case_path.relative_to(catch_dir)),
    }


def _sum_usage(records: list[dict[str, Any]]) -> dict[str, int]:
    return {
        key: sum(int((record.get("usage") or {}).get(key) or 0) for record in records)
        for key in ("input_tokens", "output_tokens", "total_tokens")
    }


def _case_sentence_bounds(response: str) -> list[tuple[int, int]]:
    sentences, _block = _answer_block(response)
    return [(sentence.start, sentence.end) for sentence in sentences]


def _method_summary(rows: list[dict[str, Any]], method: str) -> dict[str, Any]:
    available = [row for row in rows if isinstance((row.get("methods") or {}).get(method, {}).get("scores"), dict)]
    total = len(rows)
    tp = sum(row["methods"][method]["scores"]["char_tp"] for row in available)
    fp = sum(row["methods"][method]["scores"]["char_fp"] for row in available)
    fn = sum(row["methods"][method]["scores"]["char_fn"] for row in available)
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
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
    predicted_lengths: list[int] = []
    gold_lengths: list[int] = []
    full_sentence = 0
    predicted_count = 0
    fallback_count = 0
    expanded_problem_text_count = 0
    discarded_nonclaim_problem_text_count = 0
    unresolved_count = 0
    for row in available:
        response = row["response"]
        sentence_bounds = set(_case_sentence_bounds(response))
        for span in row["methods"][method].get("predicted_spans") or []:
            start, end = int(span["start"]), int(span["end"])
            predicted_lengths.append(max(0, end - start))
            predicted_count += 1
            if (start, end) in sentence_bounds:
                full_sentence += 1
        gold_lengths.extend(max(0, int(span["end"]) - int(span["start"])) for span in row.get("gold_labels") or [])
        details = row["methods"][method].get("details") or {}
        fallback_count += int(details.get("problem_text_fallback_count") or 0)
        expanded_problem_text_count += int(details.get("problem_text_expanded_to_complete_claim_count") or 0)
        discarded_nonclaim_problem_text_count += int(details.get("discarded_nonclaim_problem_text_count") or 0)
        unresolved_count += len(details.get("unresolved_predictions") or details.get("unresolved_claims") or [])
    records: dict[str, dict[str, Any]] = {}
    for row in available:
        for record in row["methods"][method].get("generation_records") or []:
            key = str(record.get("component_key") or "")
            if key:
                records[key] = record
    record_values = list(records.values())
    usage = _sum_usage(record_values)
    known_costs = [float(record["estimated_cost_usd"]) for record in record_values if record.get("estimated_cost_usd") is not None]
    latencies = [float(record.get("latency_ms") or 0.0) for record in record_values]
    clean_n = sum(not row["methods"][method]["scores"]["gold_has_hallucination"] for row in available)
    graph_diagnostics = [
        (row["methods"][method].get("details") or {}).get("response_graph_diagnostics") or {}
        for row in available
        if method == GRAPH_METHOD
    ]
    graph_claim_counts = [int(item.get("claim_count") or 0) for item in graph_diagnostics]
    graph_sentence_counts = [int(item.get("sentence_count") or 0) for item in graph_diagnostics]
    graph_node_chars = [float(item.get("mean_claim_node_chars") or 0.0) for item in graph_diagnostics if int(item.get("claim_count") or 0) > 0]
    graph_whole_sentence_nodes = sum(int(item.get("whole_sentence_node_count") or 0) for item in graph_diagnostics)
    graph_total_nodes = sum(graph_claim_counts)
    graph_explicit_non_atomic = sum(len(item.get("explicitly_non_atomic_node_ids") or []) for item in graph_diagnostics)
    graph_heuristic_compound = sum(len(item.get("heuristically_compound_node_ids") or []) for item in graph_diagnostics)
    graph_incomplete_fragments = sum(len(item.get("incomplete_fragment_node_ids") or []) for item in graph_diagnostics)
    compiler_refinements = [
        (row["methods"][method].get("details") or {}).get("response_compiler_refinement") or {}
        for row in available
        if method == GRAPH_METHOD
    ]
    refinement_attempts = sum(bool(item.get("attempted")) for item in compiler_refinements)
    remaining_quality_warnings = sum(bool(item.get("remaining_quality_warning")) for item in compiler_refinements)
    return {
        "n": len(available),
        "n_total": total,
        "n_missing": total - len(available),
        "char_precision_percent": round(precision * 100, 2),
        "char_recall_percent": round(recall * 100, 2),
        "char_f1_percent": round(f1 * 100, 2),
        "mean_case_char_f1_percent": round(statistics.mean(row["methods"][method]["scores"]["char_f1"] for row in available) * 100, 2) if available else None,
        "mean_span_f1_iou50_percent": round(statistics.mean(row["methods"][method]["scores"]["span_f1_iou50"] for row in available) * 100, 2) if available else None,
        "mean_exact_gold_span_recall_percent": round(statistics.mean(row["methods"][method]["scores"]["exact_gold_span_recall"] for row in available) * 100, 2) if available else None,
        "response_precision_percent": round(response_precision * 100, 2),
        "response_recall_percent": round(response_recall * 100, 2),
        "response_f1_percent": round(response_f1 * 100, 2),
        "response_accuracy_percent": round(statistics.mean(row["methods"][method]["scores"]["response_detection_correct"] for row in available) * 100, 2) if available else None,
        "clean_false_positive_rate_percent": round(response_fp / clean_n * 100, 2) if clean_n else None,
        "mean_predicted_span_chars": round(statistics.mean(predicted_lengths), 2) if predicted_lengths else 0.0,
        "mean_gold_span_chars": round(statistics.mean(gold_lengths), 2) if gold_lengths else 0.0,
        "full_sentence_prediction_rate_percent": round(full_sentence / predicted_count * 100, 2) if predicted_count else 0.0,
        "problem_text_fallback_rate_percent": round(fallback_count / predicted_count * 100, 2) if predicted_count else 0.0,
        "problem_text_expanded_to_complete_claim_count": expanded_problem_text_count,
        "discarded_nonclaim_problem_text_count": discarded_nonclaim_problem_text_count,
        "unresolved_localization_count": unresolved_count,
        "mean_response_claim_node_chars": round(statistics.mean(graph_node_chars), 2) if graph_node_chars else None,
        "mean_claims_per_sentence": round(sum(graph_claim_counts) / sum(graph_sentence_counts), 3) if sum(graph_sentence_counts) else None,
        "whole_sentence_response_node_rate_percent": round(graph_whole_sentence_nodes / graph_total_nodes * 100, 2) if graph_total_nodes else None,
        "explicit_non_atomic_node_rate_percent": round(graph_explicit_non_atomic / graph_total_nodes * 100, 2) if graph_total_nodes else None,
        "heuristic_compound_node_rate_percent": round(graph_heuristic_compound / graph_total_nodes * 100, 2) if graph_total_nodes else None,
        "incomplete_fragment_node_rate_percent": round(graph_incomplete_fragments / graph_total_nodes * 100, 2) if graph_total_nodes else None,
        "compiler_refinement_case_rate_percent": round(refinement_attempts / len(graph_diagnostics) * 100, 2) if graph_diagnostics else None,
        "compiler_remaining_quality_warning_cases": remaining_quality_warnings if graph_diagnostics else None,
        **usage,
        "api_calls": sum(int(record.get("api_calls") or 1) for record in record_values),
        "api_calls_this_run": sum(int(row["methods"][method].get("api_calls_this_run") or 0) for row in available),
        "cache_hits": sum(int(row["methods"][method].get("cache_hits_this_run") or 0) for row in available),
        "mean_component_latency_ms": round(statistics.mean(latencies), 2) if latencies else None,
        "estimated_cost_usd": round(sum(known_costs), 6) if known_costs else None,
        "estimated_cost_usd_this_run": round(sum(float(row["methods"][method].get("estimated_cost_usd_this_run") or 0.0) for row in available), 6),
    }



def _short_span_list(spans: list[dict[str, Any]], *, max_items: int = 6) -> str:
    if not spans:
        return "[none]"
    items: list[str] = []
    for span in spans[:max_items]:
        start = int(span.get("start") or 0)
        end = int(span.get("end") or start)
        text = " ".join(str(span.get("text") or "").split())
        if len(text) > 90:
            text = text[:87] + "..."
        label = str(span.get("label_type") or span.get("type") or "")
        suffix = f" ({label})" if label else ""
        items.append(f"[{start}:{end}] {text!r}{suffix}")
    if len(spans) > max_items:
        items.append(f"... +{len(spans) - max_items} more")
    return "; ".join(items)


def _case_method_line(label: str, output: dict[str, Any]) -> str:
    if output.get("status") != "ok" or not output.get("scores"):
        error = output.get("error") or {}
        message = str(error.get("message") or "method unavailable")
        return f"    {label:<10} ERROR | {message}"
    scores = output["scores"]
    spans = output.get("predicted_spans") or []
    return (
        f"    {label:<10} P={float(scores['char_precision']) * 100:6.2f}% "
        f"R={float(scores['char_recall']) * 100:6.2f}% "
        f"F1={float(scores['char_f1']) * 100:6.2f}% | "
        f"detect={'OK' if scores.get('response_detection_correct') else 'MISS'} | "
        f"spans={_short_span_list(spans)}"
    )


def _running_micro_f1(rows: list[dict[str, Any]], method: str) -> float | None:
    available = [
        row["methods"][method]
        for row in rows
        if row.get("methods", {}).get(method, {}).get("status") == "ok"
        and row["methods"][method].get("scores")
    ]
    if not available:
        return None
    tp = sum(int(item["scores"]["char_tp"]) for item in available)
    fp = sum(int(item["scores"]["char_fp"]) for item in available)
    fn = sum(int(item["scores"]["char_fn"]) for item in available)
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _emit_case_comparison(
    row: dict[str, Any],
    rows_so_far: list[dict[str, Any]],
    *,
    progress: Callable[[str], None],
) -> None:
    raw = row.get("methods", {}).get(RAW_METHOD) or {}
    graph = row.get("methods", {}).get(GRAPH_METHOD) or {}
    gold = row.get("gold_labels") or []
    progress("    +---------------- CASE PERFORMANCE ----------------+")
    progress(f"    Gold       spans={_short_span_list(gold)}")
    progress(_case_method_line("Raw", raw))
    progress(_case_method_line("DualGraph", graph))
    graph_details = graph.get("details") or {}
    compiler_diag = graph_details.get("response_graph_diagnostics") or {}
    compiler_refinement = graph_details.get("response_compiler_refinement") or {}
    if compiler_diag:
        progress(
            "    Compiler   "
            f"claims={int(compiler_diag.get('claim_count') or 0)} | "
            f"sentences={int(compiler_diag.get('sentence_count') or 0)} | "
            f"claims/sentence={float(compiler_diag.get('claims_per_sentence') or 0.0):.2f} | "
            f"whole-sentence nodes={int(compiler_diag.get('whole_sentence_node_count') or 0)} | "
            f"fragments={len(compiler_diag.get('incomplete_fragment_node_ids') or [])} | "
            f"refinement={'YES' if compiler_refinement.get('attempted') else 'NO'}"
            f"/{compiler_refinement.get('mode') or 'none'}"
        )

    raw_scores = raw.get("scores") if raw.get("status") == "ok" else None
    graph_scores = graph.get("scores") if graph.get("status") == "ok" else None
    if raw_scores and graph_scores:
        raw_f1 = float(raw_scores["char_f1"])
        graph_f1 = float(graph_scores["char_f1"])
        delta_pp = (graph_f1 - raw_f1) * 100
        if abs(delta_pp) < 1e-9:
            winner = "TIE"
        elif delta_pp > 0:
            winner = "DUAL_GRAPH"
        else:
            winner = "RAW_DIRECT"
        progress(f"    Case delta DualGraph-Raw = {delta_pp:+.2f} pp | winner={winner}")
    else:
        progress("    Case delta unavailable because one or both methods failed")

    raw_running = _running_micro_f1(rows_so_far, RAW_METHOD)
    graph_running = _running_micro_f1(rows_so_far, GRAPH_METHOD)
    raw_text = "NA" if raw_running is None else f"{raw_running * 100:.2f}%"
    graph_text = "NA" if graph_running is None else f"{graph_running * 100:.2f}%"
    progress(
        f"    Running micro-F1 after {len(rows_so_far)} case(s): "
        f"Raw={raw_text} | DualGraph={graph_text}"
    )
    progress("    +--------------------------------------------------+")

def _report_html(result: dict[str, Any]) -> str:
    comparison = result.get("paired_dual_graph_vs_raw") or {}
    rows = []
    for method, stats in result["method_summaries"].items():
        delta = comparison.get("mean_case_char_f1_delta_percentage_points") if method == GRAPH_METHOD else "—"
        rows.append(
            "<tr>"
            f"<td>{method}</td><td>{stats['n']}</td><td>{stats['char_f1_percent']}%</td>"
            f"<td>{stats['mean_span_f1_iou50_percent']}%</td><td>{stats['clean_false_positive_rate_percent']}%</td>"
            f"<td>{stats['mean_predicted_span_chars']}</td><td>{stats['full_sentence_prediction_rate_percent']}%</td>"
            f"<td>{stats.get('mean_response_claim_node_chars')}</td><td>{stats.get('mean_claims_per_sentence')}</td>"
            f"<td>{stats.get('compiler_refinement_case_rate_percent')}</td>"
            f"<td>{delta}</td><td>{stats['total_tokens']}</td><td>{stats['api_calls']}</td>"
            f"<td>{stats.get('estimated_cost_usd')}</td></tr>"
        )
    return f"""<!doctype html><meta charset='utf-8'><title>RAGTruth raw vs dual graph nano</title>
<style>body{{font-family:Arial,sans-serif;margin:28px;color:#172033}}table{{border-collapse:collapse;width:100%;margin:12px 0 28px}}th,td{{border:1px solid #d7dee9;padding:8px;text-align:left}}th{{background:#f3f6fa}}.note{{background:#eef6ff;border:1px solid #bfdbfe;padding:12px;border-radius:10px;line-height:1.5}}code{{background:#f4f4f5;padding:2px 5px}}</style>
<h1>RAGTruth · raw direct vs dual graph · nano only</h1>
<p class='note'>Both conditions use <b>{result['settings']['model']}</b>. Raw direct compares source and response in one call. Dual graph independently compiles a source evidence graph and response claim graph, aligns them, then projects each problematic node to a minimal complete error clause in the response.</p>
<p>Cases: <b>{result['summary']['completed_cases']}</b> · Primary metric: micro character-level F1.</p>
<table><thead><tr><th>Method</th><th>N</th><th>Character F1</th><th>Span F1 @ IoU .5</th><th>Clean FP</th><th>Mean predicted chars</th><th>Whole-sentence predictions</th><th>Mean claim-node chars</th><th>Claims/sentence</th><th>Compiler refinement %</th><th>Δ mean case F1</th><th>Tokens</th><th>Unique calls</th><th>Est. USD</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<p class='note'>Response sentences are containers; claim nodes are minimal complete propositions rather than sentence-sized bundles or token fragments. Final evaluation spans are complete error clauses, with isolated values retained only for direct substitutions. Local compiler repair touches only problematic nodes. The comparison is a system-level architecture comparison, not a compute-matched causal estimate of graph serialization alone.</p>"""


def run_ragtruth_raw_vs_dual_graph(
    *,
    response_path: Path,
    source_path: Path,
    output_root: Path,
    model: str = "gpt-5.4-nano",
    split: str = "test",
    quality: str = "good",
    task_types: list[str] | None = None,
    limit: int = 24,
    skip_first: int = 0,
    seed: int = 2040,
    reasoning_effort: str = "low",
    max_output_tokens_direct: int = 1800,
    max_output_tokens_source_graph: int = 3200,
    max_output_tokens_response_graph: int = 3600,
    max_output_tokens_alignment: int = 2600,
    max_context_chars: int = 60_000,
    max_response_chars: int = 3000,
    include_implicit_true: bool = True,
    exclude_case_ids: set[str] | None = None,
    include_case_ids: set[str] | None = None,
    require_full_evidence: bool = True,
    generation_cache_path: Path | None = None,
    force_components: set[str] | None = None,
    alignment_prompt_profile: str = "v049_recall",
    alignment_gate_profile: str = "v049_balanced_recall",
    parallel_components: bool = True,
    print_case_comparison: bool = True,
    client: Any = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    _load_local_env()
    progress = progress or (lambda _message: None)
    if model != "gpt-5.4-nano":
        raise ValueError("This comparison intentionally fixes both conditions to gpt-5.4-nano")
    if reasoning_effort not in ALLOWED_REASONING_EFFORTS:
        raise ValueError("reasoning_effort must be low, medium, or high")
    if not response_path.exists() or not source_path.exists():
        raise FileNotFoundError("RAGTruth files are missing. Run the pilot with --download first.")
    force = set(force_components or set())
    allowed_force = {"raw_direct", "source_graph", "response_graph", "alignment"}
    unknown = force - allowed_force
    if unknown:
        raise ValueError(f"Unknown force components: {sorted(unknown)}")
    if alignment_prompt_profile not in ALIGNMENT_PROMPT_VERSIONS:
        raise ValueError(f"Unknown alignment prompt profile: {alignment_prompt_profile}")
    if alignment_gate_profile not in ALIGNMENT_GATE_PROFILES:
        raise ValueError(f"Unknown alignment gate profile: {alignment_gate_profile}")
    alignment_prompt_version = ALIGNMENT_PROMPT_VERSIONS[alignment_prompt_profile]

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
        include_case_ids=include_case_ids,
        require_full_evidence=require_full_evidence,
        max_context_chars=max_context_chars,
    )
    if skip_first < 0:
        raise ValueError("skip_first must be >= 0")
    original_case_count = len(cases)
    if skip_first:
        cases = cases[skip_first:]
        sampling = {
            **sampling,
            "pre_skip_case_count": original_case_count,
            "skip_first": skip_first,
            "post_skip_case_count": len(cases),
            "split_role": "locked_test_suffix",
        }
    cache = _load_cache(generation_cache_path)
    active_client = client

    def get_client() -> Any:
        nonlocal active_client
        if active_client is not None:
            return active_client
        if not os.getenv("OPENAI_API_KEY", "").strip():
            raise ValueError("OPENAI_API_KEY is not configured for an uncached component call")
        from openai import OpenAI
        active_client = OpenAI()
        return active_client

    run_id = f"ragtruth_raw_vs_dual_graph_nano_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    actual_api_calls = 0
    cache_hits_by_component = Counter()
    catch_dir = run_dir / "catch_candidates"
    catch_index: list[dict[str, Any]] = []

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

        raw_key = _component_key("raw_direct", {
            "case_id": case["case_id"],
            "response": case["response"],
            "task_instruction": case["task_instruction"],
            "evidence_units": evidence_card["units"],
            "model": model,
            "reasoning_effort": reasoning_effort,
            "max_output_tokens": max_output_tokens_direct,
        })
        source_key = _component_key("source_graph", {
            "source_id": case["source_id"],
            "task_instruction": case["task_instruction"],
            "evidence_units": evidence_card["units"],
            "model": model,
            "reasoning_effort": reasoning_effort,
            "max_output_tokens": max_output_tokens_source_graph,
        })
        response_key = _component_key("response_graph", {
            "case_id": case["case_id"],
            "response": case["response"],
            "task_instruction": case["task_instruction"],
            "model": model,
            "reasoning_effort": reasoning_effort,
            "max_output_tokens": max_output_tokens_response_graph,
        })

        component_specs: dict[str, tuple[str, str, type[BaseModel], int, str]] = {}
        raw_system, raw_user = _raw_prompts(case, evidence_card)
        source_system, source_user = _source_graph_prompts(case, evidence_card)
        response_system, response_user = _response_graph_prompts(case)
        component_specs["raw_direct"] = (raw_system, raw_user, DirectSpanOutput, max_output_tokens_direct, raw_key)
        component_specs["source_graph"] = (source_system, source_user, SourceEvidenceGraphOutput, max_output_tokens_source_graph, source_key)
        component_specs["response_graph"] = (response_system, response_user, ResponseClaimGraphOutput, max_output_tokens_response_graph, response_key)

        component_results: dict[str, dict[str, Any]] = {}
        missing: list[str] = []
        for name, (_system, _user, _type, _tokens, key) in component_specs.items():
            bucket = cache[name]
            if name not in force and key in bucket:
                component_results[name] = copy.deepcopy(bucket[key])
                cache_hits_by_component[name] += 1
                progress(f"    - {name}: cache hit")
            else:
                missing.append(name)

        if missing:
            try:
                call_client = get_client()
                def call_component_with_client(name: str) -> tuple[str, dict[str, Any]]:
                    system, user, output_type, token_budget, key = component_specs[name]
                    progress(f"    - {name}: nano")
                    if name == "response_graph":
                        result = _call_response_graph_compiler(
                            call_client,
                            case=case,
                            model=model,
                            reasoning_effort=reasoning_effort,
                            max_output_tokens=token_budget,
                            system=system,
                            user=user,
                        )
                    else:
                        result = _call_parsed(
                            call_client,
                            model=model,
                            reasoning_effort=reasoning_effort,
                            max_output_tokens=token_budget,
                            system=system,
                            user=user,
                            output_type=output_type,
                        )
                    result["component"] = name
                    result["component_key"] = key
                    result["model"] = model
                    result["prompt_version"] = PROMPT_VERSIONS[name]
                    return name, result
                if parallel_components and len(missing) > 1:
                    with ThreadPoolExecutor(max_workers=min(3, len(missing))) as executor:
                        futures = {executor.submit(call_component_with_client, name): name for name in missing}
                        for future in as_completed(futures):
                            name, result = future.result()
                            component_results[name] = result
                else:
                    for name in missing:
                        returned_name, result = call_component_with_client(name)
                        component_results[returned_name] = result
                for name in missing:
                    key = component_specs[name][4]
                    cache[name][key] = copy.deepcopy(component_results[name])
                    actual_api_calls += int(component_results[name].get("api_calls") or 1)
                _save_cache(generation_cache_path, cache)
            except Exception as exc:
                failures.append({"case_id": case["case_id"], "stage": "parallel_initial_components", "type": type(exc).__name__, "message": str(exc)})
                progress(f"      ERROR {type(exc).__name__}: {exc}")

        # Raw direct is independently scored even if a graph component failed.
        raw_record = component_results.get("raw_direct")
        if raw_record and raw_record.get("status") == "ok":
            parsed_raw = DirectSpanOutput.model_validate(raw_record["parsed"])
            raw_predictions, raw_details = _predictions_from_direct(parsed_raw, case["response"])
            raw_output = {
                "method": RAW_METHOD,
                "model": model,
                "status": "ok",
                "predicted_spans": raw_predictions,
                "details": raw_details,
                "generation_records": [raw_record],
                "api_calls_this_run": int(raw_record.get("api_calls") or 1) if "raw_direct" in missing else 0,
                "cache_hits_this_run": int("raw_direct" not in missing),
                "estimated_cost_usd_this_run": float(raw_record.get("estimated_cost_usd") or 0.0) if "raw_direct" in missing else 0.0,
            }
            raw_output["scores"] = score_predictions(case["response"], raw_predictions, case["gold_labels"])
        else:
            raw_output = {
                "method": RAW_METHOD,
                "model": model,
                "status": "error",
                "predicted_spans": [],
                "scores": None,
                "generation_records": [],
                "api_calls_this_run": int(raw_record.get("api_calls") or 1) if "raw_direct" in missing else 0,
                "cache_hits_this_run": 0,
                "estimated_cost_usd_this_run": 0.0,
                "error": {"type": "MissingComponent", "message": "raw_direct component unavailable"},
            }
        row["methods"][RAW_METHOD] = raw_output

        source_record = component_results.get("source_graph")
        response_record = component_results.get("response_graph")
        graph_output: dict[str, Any]
        if source_record and response_record and source_record.get("status") == "ok" and response_record.get("status") == "ok":
            source_graph = SourceEvidenceGraphOutput.model_validate(source_record["parsed"])
            response_graph = ResponseClaimGraphOutput.model_validate(response_record["parsed"])
            alignment_key = _component_key("alignment", {
                "case_id": case["case_id"],
                "response": case["response"],
                "task_instruction": case["task_instruction"],
                "evidence_units": evidence_card["units"],
                "source_graph": source_graph.model_dump(),
                "response_graph": response_graph.model_dump(),
                "model": model,
                "reasoning_effort": reasoning_effort,
                "max_output_tokens": max_output_tokens_alignment,
            }, prompt_version=alignment_prompt_version)
            alignment_record = None
            alignment_cache_hit = False
            if "alignment" not in force and alignment_key in cache["alignment"]:
                alignment_record = copy.deepcopy(cache["alignment"][alignment_key])
                alignment_cache_hit = True
                cache_hits_by_component["alignment"] += 1
                progress("    - alignment: cache hit")
            else:
                progress("    - alignment + complete-error-clause projection: nano")
                try:
                    alignment_system, alignment_user = _alignment_prompts(
                        case,
                        evidence_card,
                        source_graph,
                        response_graph,
                        alignment_prompt_profile=alignment_prompt_profile,
                    )
                    alignment_record = _call_parsed(
                        get_client(),
                        model=model,
                        reasoning_effort=reasoning_effort,
                        max_output_tokens=max_output_tokens_alignment,
                        system=alignment_system,
                        user=alignment_user,
                        output_type=DualGraphAlignmentOutput,
                    )
                    alignment_record["component"] = "alignment"
                    alignment_record["component_key"] = alignment_key
                    alignment_record["model"] = model
                    alignment_record["prompt_version"] = alignment_prompt_version
                    alignment_record["alignment_prompt_profile"] = alignment_prompt_profile
                    cache["alignment"][alignment_key] = copy.deepcopy(alignment_record)
                    _save_cache(generation_cache_path, cache)
                    actual_api_calls += int(alignment_record.get("api_calls") or 1)
                except Exception as exc:
                    failures.append({"case_id": case["case_id"], "stage": "alignment", "type": type(exc).__name__, "message": str(exc)})
                    progress(f"      ERROR {type(exc).__name__}: {exc}")
                    alignment_record = None
            if alignment_record and alignment_record.get("status") == "ok":
                parsed_alignment = DualGraphAlignmentOutput.model_validate(alignment_record["parsed"])
                predictions, details = _predictions_from_alignment(
                    parsed_alignment,
                    response_graph,
                    case["response"],
                    gate_profile=alignment_gate_profile,
                )
                details["source_graph"] = source_graph.model_dump()
                details["response_compiler_refinement"] = response_record.get("compiler_refinement") or {}
                generation_records = [source_record, response_record, alignment_record]
                this_run_components = [name for name in ("source_graph", "response_graph") if name in missing]
                if not alignment_cache_hit:
                    this_run_components.append("alignment")
                graph_output = {
                    "method": GRAPH_METHOD,
                    "model": model,
                    "status": "ok",
                    "predicted_spans": predictions,
                    "details": details,
                    "generation_records": generation_records,
                    "api_calls_this_run": sum(
                        int(record.get("api_calls") or 1)
                        for record in generation_records
                        if record.get("component") in this_run_components
                    ),
                    "cache_hits_this_run": 3 - len(this_run_components),
                    "estimated_cost_usd_this_run": round(sum(
                        float(record.get("estimated_cost_usd") or 0.0)
                        for record in generation_records
                        if record.get("component") in this_run_components
                    ), 8),
                }
                graph_output["scores"] = score_predictions(case["response"], predictions, case["gold_labels"])
            else:
                graph_output = {
                    "method": GRAPH_METHOD,
                    "model": model,
                    "status": "error",
                    "predicted_spans": [],
                    "scores": None,
                    "generation_records": [source_record, response_record],
                    "api_calls_this_run": sum(name in missing for name in ("source_graph", "response_graph")),
                    "cache_hits_this_run": sum(name not in missing for name in ("source_graph", "response_graph")),
                    "estimated_cost_usd_this_run": 0.0,
                    "error": {"type": "MissingComponent", "message": "alignment component unavailable"},
                }
        else:
            graph_output = {
                "method": GRAPH_METHOD,
                "model": model,
                "status": "error",
                "predicted_spans": [],
                "scores": None,
                "generation_records": [record for record in (source_record, response_record) if record],
                "api_calls_this_run": sum(name in missing for name in ("source_graph", "response_graph")),
                "cache_hits_this_run": sum(name not in missing for name in ("source_graph", "response_graph")),
                "estimated_cost_usd_this_run": 0.0,
                "error": {"type": "MissingComponent", "message": "source or response graph unavailable"},
            }
        row["methods"][GRAPH_METHOD] = graph_output
        rows.append(row)
        reason = _catch_reason(raw_output.get("scores"), graph_output.get("scores"))
        if reason:
            entry = _save_catch_candidate(catch_dir, row, reason)
            catch_index.append(entry)
            (catch_dir / "index.json").write_text(json.dumps({
                "run_id": run_id,
                "candidate_count": len(catch_index),
                "candidates": catch_index,
            }, ensure_ascii=False, indent=2), encoding="utf-8")
            progress(f"    - catch saved: {reason} → catch_candidates/cases/{case['case_id']}.json")
        if print_case_comparison:
            _emit_case_comparison(row, rows, progress=progress)
        with (run_dir / "cases.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summaries = {method: _method_summary(rows, method) for method in METHODS}
    paired = _paired_comparison(rows, GRAPH_METHOD, seed=seed, reference_method=RAW_METHOD)
    result = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": {
            "name": "RAGTruth",
            "response_path": str(response_path),
            "source_path": str(source_path),
            "human_span_annotations": True,
        },
        "settings": {
            "model": model,
            "both_conditions_use_same_model": True,
            "methods": METHODS,
            "split": split,
            "quality": quality,
            "task_types": task_types or ["QA"],
            "limit": limit,
            "skip_first": skip_first,
            "seed": seed,
            "reasoning_effort": reasoning_effort,
            "max_context_chars": max_context_chars,
            "max_response_chars": max_response_chars,
            "require_full_evidence": require_full_evidence,
            "parallel_initial_components": parallel_components,
            "print_case_comparison": print_case_comparison,
            "dual_graph_pipeline": "task-complete source graph + balanced complete-proposition response graph in parallel, local compiler repair, then configurable alignment prompt + relation-aware recall-balanced projection",
            "alignment_prompt_profile": alignment_prompt_profile,
            "alignment_prompt_version": alignment_prompt_version,
            "alignment_gate_profile": alignment_gate_profile,
            "alignment_gate_config": copy.deepcopy(ALIGNMENT_GATE_PROFILES[alignment_gate_profile]),
            "catch_analysis_policy": "benchmark saves candidates only; selected candidates are reviewed later with six-agent graphs",
            "response_graph_node_unit": "minimal semantically complete independently judgeable factual proposition",
            "response_sentences_are_containers": True,
            "response_compiler_refinement_max_passes": 1,
            "response_compiler_refinement_mode": "local_node_patch",
        },
        "sampling": sampling,
        "method_summaries": summaries,
        "paired_dual_graph_vs_raw": paired,
        "cache_summary": {
            "generation_cache_path": str(generation_cache_path) if generation_cache_path else None,
            "cache_hits_by_component": dict(cache_hits_by_component),
            "actual_api_calls_this_run": actual_api_calls,
        },
        "summary": {
            "completed_cases": len(rows),
            "failed_component_calls": len(failures),
            "raw_char_f1_percent": summaries.get(RAW_METHOD, {}).get("char_f1_percent"),
            "dual_graph_char_f1_percent": summaries.get(GRAPH_METHOD, {}).get("char_f1_percent"),
            "dual_graph_delta_mean_case_char_f1_pp": paired.get("mean_case_char_f1_delta_percentage_points"),
            "dual_graph_mean_claims_per_sentence": summaries.get(GRAPH_METHOD, {}).get("mean_claims_per_sentence"),
            "dual_graph_whole_sentence_response_node_rate_percent": summaries.get(GRAPH_METHOD, {}).get("whole_sentence_response_node_rate_percent"),
            "dual_graph_compiler_refinement_case_rate_percent": summaries.get(GRAPH_METHOD, {}).get("compiler_refinement_case_rate_percent"),
            "actual_api_calls_this_run": actual_api_calls,
            "catch_candidate_count": len(catch_index),
        },
        "catch_candidates": {
            "directory": str(catch_dir),
            "count": len(catch_index),
            "index": catch_index,
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
        "failed_component_calls": len(failures),
        "actual_api_calls_this_run": actual_api_calls,
        "catch_candidate_count": len(catch_index),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return result

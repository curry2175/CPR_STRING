from __future__ import annotations

import copy
import json
import os
import statistics
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Literal

from pydantic import BaseModel, Field

from .openai_runner import ALLOWED_REASONING_EFFORTS, _load_local_env
from .ragtruth_dual_graph import (
    AlignmentRecord,
    DualGraphAlignmentOutput,
    ResponseClaimEdge,
    ResponseClaimGraphOutput,
    ResponseClaimNode,
    SourceEvidenceGraphOutput,
    SourceGraphEdge,
    SourceGraphNode,
    _call_parsed,
    _call_response_graph_compiler,
    _is_incomplete_claim_fragment,
    _method_summary as _legacy_method_summary,
    _predictions_from_alignment,
    _raw_prompts,
    _response_graph_diagnostics,
    _response_graph_prompts,
    _source_graph_prompts,
)
from .ragtruth_localization import (
    DirectSpanOutput,
    _answer_block,
    _paired_comparison,
    _predictions_from_direct,
    _stable_hash,
    build_evidence_card,
    load_ragtruth_cases,
    locate_exact_quote,
    score_predictions,
)


SCHEMA_VERSION = "0.44.0"
CACHE_SCHEMA_VERSION = "0.44.0-ragtruth-true-six-agent-cache"
RAW_METHOD = "nano_raw_direct"
GRAPH_METHOD = "nano_six_agent_dual_graph"
LEGACY_GRAPH_METHOD = "nano_dual_graph"
METHODS = [RAW_METHOD, GRAPH_METHOD]

PROMPT_VERSIONS = {
    "raw_direct": "v040-raw-direct-minimal-span-nano",
    "source_compiler": "v043-source-evidence-graph-task-complete",
    "response_compiler": "v043-response-balanced-complete-proposition-graph-local-repair",
    "source_evidence": "v044-source-evidence-agent-graph-validation",
    "source_logic": "v044-source-logic-agent-graph-validation",
    "source_target": "v044-source-target-agent-graph-validation",
    "source_assumption": "v044-source-assumption-agent-conditional",
    "source_judge": "v044-source-judge-agent-conditional",
    "response_evidence": "v044-response-evidence-agent-graph-validation",
    "response_logic": "v044-response-logic-agent-graph-validation",
    "response_target": "v044-response-target-agent-graph-validation",
    "response_assumption": "v044-response-assumption-agent-conditional",
    "response_judge": "v044-response-judge-agent-conditional",
    "cross_matcher": "v044-cross-graph-evidence-matcher",
    "cross_logic": "v044-cross-graph-logic-verifier",
    "cross_assumption": "v044-cross-graph-assumption-agent-conditional",
    "cross_judge": "v044-cross-graph-judge-agent-conditional",
    "span_projector": "v044-relation-aware-span-projector",
}

GraphKind = Literal["source", "response"]
ValidationSpecialist = Literal["evidence", "logic", "target"]


class GraphPatch(BaseModel):
    id: str
    operation: Literal[
        "drop_node",
        "replace_source_node",
        "replace_response_node",
        "add_source_node",
        "add_response_node",
        "mark_response_ineligible",
        "drop_edge",
        "replace_source_edge",
        "replace_response_edge",
    ]
    target_id: str = ""
    source_node: SourceGraphNode | None = None
    response_node: ResponseClaimNode | None = None
    source_edge: SourceGraphEdge | None = None
    response_edge: ResponseClaimEdge | None = None
    rationale: str = ""
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class GraphValidationFinding(BaseModel):
    id: str
    specialist: ValidationSpecialist
    verdict: Literal["valid", "invalid", "uncertain"]
    severity: Literal["low", "medium", "high"] = "medium"
    issue_type: Literal[
        "source_misalignment",
        "response_misalignment",
        "omitted_proposition",
        "non_propositional_fragment",
        "multi_verdict_node",
        "wrong_edge",
        "unsupported_internal_edge",
        "scope_or_modality_shift",
        "task_irrelevance",
        "task_coverage_gap",
        "duplicate_node",
        "other",
    ] = "other"
    node_ids: list[str] = Field(default_factory=list)
    edge_ids: list[str] = Field(default_factory=list)
    explanation: str = ""
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    patches: list[GraphPatch] = Field(default_factory=list, max_length=12)
    requires_assumption_review: bool = False
    missing_premise: str = ""
    required_for_edge_id: str = ""


class GraphSpecialistReviewOutput(BaseModel):
    specialist: ValidationSpecialist
    reviewed_node_ids: list[str] = Field(default_factory=list)
    reviewed_edge_ids: list[str] = Field(default_factory=list)
    findings: list[GraphValidationFinding] = Field(default_factory=list, max_length=30)
    review_summary: str = ""


class GraphAssumptionAssessment(BaseModel):
    finding_id: str
    disposition: Literal[
        "explicit_in_text",
        "safe_linguistic_inference",
        "missing_but_required",
        "external_knowledge_required",
        "not_actually_required",
        "uncertain",
    ]
    explanation: str = ""
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class GraphAssumptionReviewOutput(BaseModel):
    assessments: list[GraphAssumptionAssessment] = Field(default_factory=list, max_length=30)
    summary: str = ""


class GraphJudgeOutput(BaseModel):
    accepted_finding_ids: list[str] = Field(default_factory=list)
    rejected_finding_ids: list[str] = Field(default_factory=list)
    accepted_patch_ids: list[str] = Field(default_factory=list)
    rejected_patch_ids: list[str] = Field(default_factory=list)
    revised_patches: list[GraphPatch] = Field(default_factory=list, max_length=30)
    rationale: str = ""


class CrossMatchRecord(BaseModel):
    response_node_id: str
    candidate_source_node_ids: list[str] = Field(default_factory=list)
    candidate_evidence_ids: list[str] = Field(default_factory=list)
    match_type: Literal["exact", "semantic", "qualified", "conflicting", "none"] = "none"
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    note: str = ""


class CrossEvidenceMatchOutput(BaseModel):
    matches: list[CrossMatchRecord] = Field(default_factory=list, max_length=120)
    summary: str = ""


class CrossLogicVerdict(BaseModel):
    response_node_id: str
    source_node_ids: list[str] = Field(default_factory=list)
    verdict: Literal[
        "supported_by",
        "partially_supported_by",
        "contradicted_by",
        "not_found_in_source",
        "qualified_by",
        "requires_assumption",
        "safe_inference",
        "not_factual",
        "uncertain",
    ]
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    explanation: str = ""
    changed_dimensions: list[Literal[
        "entity", "number", "unit", "polarity", "scope", "condition", "time",
        "modality", "comparison", "causal_strength", "population", "other",
    ]] = Field(default_factory=list)
    missing_premise: str = ""


class CrossLogicOutput(BaseModel):
    verdicts: list[CrossLogicVerdict] = Field(default_factory=list, max_length=120)
    summary: str = ""


class CrossAssumptionAssessment(BaseModel):
    response_node_id: str
    disposition: Literal[
        "explicit_in_source",
        "safe_linguistic_inference",
        "external_knowledge_required",
        "unsupported_missing_premise",
        "not_actually_required",
        "uncertain",
    ]
    resulting_verdict: Literal[
        "supported_by", "safe_inference", "not_found_in_source", "uncertain"
    ]
    explanation: str = ""
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class CrossAssumptionOutput(BaseModel):
    assessments: list[CrossAssumptionAssessment] = Field(default_factory=list, max_length=60)
    summary: str = ""


class CrossJudgeOutput(BaseModel):
    revised_verdicts: list[CrossLogicVerdict] = Field(default_factory=list, max_length=120)
    rationale: str = ""


def _empty_cache() -> dict[str, Any]:
    return {"schema_version": CACHE_SCHEMA_VERSION, "components": {}}


def _load_cache(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return _empty_cache()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return _empty_cache()
    if not isinstance(payload, dict) or payload.get("schema_version") != CACHE_SCHEMA_VERSION:
        return _empty_cache()
    payload.setdefault("components", {})
    return payload


def _save_cache(path: Path | None, cache: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def _component_key(component: str, payload: dict[str, Any]) -> str:
    return _stable_hash({
        "component": component,
        "prompt_version": PROMPT_VERSIONS[component],
        **payload,
    })


def _record_component(record: dict[str, Any], *, component: str, key: str, model: str) -> dict[str, Any]:
    record = copy.deepcopy(record)
    record["component"] = component
    record["component_key"] = key
    record["model"] = model
    record["prompt_version"] = PROMPT_VERSIONS[component]
    return record


def _cached_parsed_call(
    *,
    cache: dict[str, Any],
    cache_lock: threading.Lock,
    cache_path: Path | None,
    force: set[str],
    component: str,
    payload: dict[str, Any],
    client: Any,
    model: str,
    reasoning_effort: str,
    max_output_tokens: int,
    system: str,
    user: str,
    output_type: type[BaseModel],
) -> tuple[dict[str, Any], bool]:
    key = _component_key(component, payload)
    with cache_lock:
        cached = copy.deepcopy(cache["components"].get(key))
    if component not in force and cached is not None:
        cached["cache_hit"] = True
        return cached, True
    record = _call_parsed(
        client,
        model=model,
        reasoning_effort=reasoning_effort,
        max_output_tokens=max_output_tokens,
        system=system,
        user=user,
        output_type=output_type,
    )
    record = _record_component(record, component=component, key=key, model=model)
    record["cache_hit"] = False
    with cache_lock:
        cache["components"][key] = copy.deepcopy(record)
        _save_cache(cache_path, cache)
    return record, False


def _cached_response_compiler_call(
    *,
    cache: dict[str, Any],
    cache_lock: threading.Lock,
    cache_path: Path | None,
    force: set[str],
    case: dict[str, Any],
    client: Any,
    model: str,
    reasoning_effort: str,
    max_output_tokens: int,
) -> tuple[dict[str, Any], bool]:
    component = "response_compiler"
    payload = {
        "case_id": case["case_id"],
        "response": case["response"],
        "task_instruction": case["task_instruction"],
        "model": model,
        "reasoning_effort": reasoning_effort,
        "max_output_tokens": max_output_tokens,
    }
    key = _component_key(component, payload)
    with cache_lock:
        cached = copy.deepcopy(cache["components"].get(key))
    if component not in force and cached is not None:
        cached["cache_hit"] = True
        return cached, True
    system, user = _response_graph_prompts(case)
    record = _call_response_graph_compiler(
        client,
        case=case,
        model=model,
        reasoning_effort=reasoning_effort,
        max_output_tokens=max_output_tokens,
        system=system,
        user=user,
    )
    record = _record_component(record, component=component, key=key, model=model)
    record["cache_hit"] = False
    with cache_lock:
        cache["components"][key] = copy.deepcopy(record)
        _save_cache(cache_path, cache)
    return record, False


def _validation_original_block(
    kind: GraphKind,
    *,
    case: dict[str, Any],
    evidence_card: dict[str, Any],
) -> str:
    if kind == "source":
        return evidence_card["text"]
    _sentences, block = _answer_block(case["response"])
    return block


def _validation_graph_json(kind: GraphKind, graph: SourceEvidenceGraphOutput | ResponseClaimGraphOutput) -> str:
    return json.dumps(graph.model_dump(), ensure_ascii=False, separators=(",", ":"))


def _validation_specialist_prompts(
    specialist: ValidationSpecialist,
    kind: GraphKind,
    *,
    case: dict[str, Any],
    evidence_card: dict[str, Any],
    graph: SourceEvidenceGraphOutput | ResponseClaimGraphOutput,
) -> tuple[str, str]:
    separation = (
        "You are validating the SOURCE graph only. The RESPONSE is intentionally unavailable. Treat the supplied SOURCE "
        "text as authoritative for this task, but do not automatically trust the compiler's nodes or edges."
        if kind == "source"
        else
        "You are validating the RESPONSE graph only. The SOURCE is intentionally unavailable. Preserve what the response "
        "actually claims without judging whether those claims are externally true or false."
    )
    role_rules = {
        "evidence": (
            "Check grounding fidelity. Verify that every node faithfully represents the supplied original text, preserves "
            "entity, polarity, number, unit, scope, condition, time, modality, comparison, and causal strength, and does not "
            "invent content. For SOURCE nodes, verify evidence_ids and exact supporting evidence. For RESPONSE nodes, verify "
            "that node.text is an exact contiguous quote and normalized_claim restores context without adding facts. Detect "
            "omissions and fragments. Do not assess cross-document factual correctness."
        ),
        "logic": (
            "Check graph structure and internal logic. Verify that each node can receive one factual verdict and each edge "
            "uses a defensible relation. Detect multi-verdict nodes, unsupported internal derivations, wrong edge direction or "
            "strength, and hidden premises. For RESPONSE, assess only internal representation and reasoning, never truth against "
            "the unseen source. Propose an assumption review only for a concrete missing premise tied to an edge."
        ),
        "target": (
            "Check task coverage and relevance. Verify that task-relevant propositions are not omitted and irrelevant material "
            "does not dominate the graph. For RESPONSE, ensure all externally checkable claims and the answer-bearing conclusion "
            "are represented; do not decide whether they are correct. For SOURCE, preserve facts that could support, contradict, "
            "qualify, or constrain an answer."
        ),
    }
    system = (
        f"You are the {specialist.capitalize()} Agent in a true six-agent graph-construction pipeline. {separation} "
        f"{role_rules[specialist]} Be conservative: keep a node or edge unless a concrete defect is supported by the supplied "
        "text. When correction is possible, attach a precise GraphPatch. Add nodes only for clearly omitted propositions. "
        "Never rewrite style merely for preference. Use unique patch ids."
    )
    user = (
        f"TASK\n{case['task_instruction']}\n\n"
        f"GRAPH KIND\n{kind}\n\n"
        f"ORIGINAL TEXT WITH STABLE IDS\n{_validation_original_block(kind, case=case, evidence_card=evidence_card)}\n\n"
        f"CANDIDATE GRAPH\n{_validation_graph_json(kind, graph)}\n\n"
        "Return a graph-local specialist review. Review every node and edge, but report only concrete defects or material uncertainty."
    )
    return system, user


def _assumption_prompts(
    kind: GraphKind,
    *,
    case: dict[str, Any],
    evidence_card: dict[str, Any],
    graph: SourceEvidenceGraphOutput | ResponseClaimGraphOutput,
    findings: list[GraphValidationFinding],
) -> tuple[str, str]:
    system = (
        "You are Agent 4 of 6: the conditional Assumption Agent. Review only concrete missing-premise candidates proposed by "
        "the Logic Agent. Determine whether each premise is explicit in the supplied original text, a safe linguistic inference, "
        "truly missing but required, dependent on external knowledge, not actually required, or uncertain. Do not invent new "
        "premises and do not evaluate the unseen other document."
    )
    user = (
        f"TASK\n{case['task_instruction']}\n\nGRAPH KIND\n{kind}\n\n"
        f"ORIGINAL TEXT\n{_validation_original_block(kind, case=case, evidence_card=evidence_card)}\n\n"
        f"GRAPH\n{_validation_graph_json(kind, graph)}\n\n"
        f"MISSING-PREMISE FINDINGS\n{json.dumps([x.model_dump() for x in findings], ensure_ascii=False)}"
    )
    return system, user


def _judge_prompts(
    kind: GraphKind,
    *,
    case: dict[str, Any],
    evidence_card: dict[str, Any],
    graph: SourceEvidenceGraphOutput | ResponseClaimGraphOutput,
    findings: list[GraphValidationFinding],
    assumptions: list[GraphAssumptionAssessment],
) -> tuple[str, str]:
    system = (
        "You are Agent 6 of 6: the conditional Judge for graph construction. Adjudicate only the supplied graph-local findings "
        "and patches. Preserve the candidate graph unless the original text clearly supports a change. Reject speculative, "
        "stylistic, duplicate, cross-document, or over-broad patches. Accept a patch only when it improves fidelity, atomicity, "
        "coverage, edge validity, or task relevance. For a RESPONSE graph, never judge external truth because the source is not "
        "available. Return accepted/rejected ids and, only when necessary, narrower revised patches."
    )
    user = (
        f"TASK\n{case['task_instruction']}\n\nGRAPH KIND\n{kind}\n\n"
        f"ORIGINAL TEXT\n{_validation_original_block(kind, case=case, evidence_card=evidence_card)}\n\n"
        f"GRAPH\n{_validation_graph_json(kind, graph)}\n\n"
        f"FINDINGS\n{json.dumps([x.model_dump() for x in findings], ensure_ascii=False)}\n\n"
        f"ASSUMPTION ASSESSMENTS\n{json.dumps([x.model_dump() for x in assumptions], ensure_ascii=False)}"
    )
    return system, user


def _normalize_review(review: GraphSpecialistReviewOutput, specialist: ValidationSpecialist) -> GraphSpecialistReviewOutput:
    review.specialist = specialist
    for find_index, finding in enumerate(review.findings, 1):
        finding.specialist = specialist
        finding.id = f"{specialist}_{finding.id or find_index}"
        for patch_index, patch in enumerate(finding.patches, 1):
            patch.id = f"{specialist}_{patch.id or f'p{find_index}_{patch_index}'}"
    return review


def _findings_need_assumption(findings: list[GraphValidationFinding]) -> list[GraphValidationFinding]:
    return [
        finding for finding in findings
        if finding.specialist == "logic"
        and finding.requires_assumption_review
        and finding.missing_premise.strip()
    ]


def _judge_trigger_reasons(findings: list[GraphValidationFinding]) -> list[str]:
    reasons: list[str] = []
    if any(f.patches for f in findings):
        reasons.append("graph_patch_proposed")
    if any(f.verdict == "uncertain" for f in findings):
        reasons.append("material_uncertainty")
    if any(f.severity == "high" for f in findings):
        reasons.append("high_severity_graph_defect")
    target_map: dict[str, set[str]] = {}
    for finding in findings:
        for target in finding.node_ids + finding.edge_ids:
            target_map.setdefault(target, set()).add(finding.specialist)
    if any(len(agents) > 1 for agents in target_map.values()):
        reasons.append("overlapping_specialist_findings")
    return reasons


def _valid_source_node(node: SourceGraphNode, allowed_evidence_ids: set[str]) -> bool:
    return bool(node.text.strip()) and bool(node.evidence_ids) and set(node.evidence_ids).issubset(allowed_evidence_ids)


def _valid_response_node(node: ResponseClaimNode, response: str) -> bool:
    if not node.text.strip() or locate_exact_quote(response, node.text, node.sentence_id) is None:
        return False
    if node.evaluation_eligible and _is_incomplete_claim_fragment(node):
        return False
    return True


def _dedup_source_nodes(nodes: list[SourceGraphNode]) -> list[SourceGraphNode]:
    result: list[SourceGraphNode] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()
    for node in nodes:
        key = (" ".join(node.text.lower().split()), tuple(sorted(node.evidence_ids)))
        if key in seen:
            continue
        seen.add(key)
        result.append(node)
    return result


def _dedup_response_nodes(nodes: list[ResponseClaimNode]) -> list[ResponseClaimNode]:
    result: list[ResponseClaimNode] = []
    seen: set[tuple[str, str, str]] = set()
    for node in nodes:
        key = (
            node.sentence_id,
            " ".join(node.text.lower().split()),
            " ".join((node.normalized_claim or node.text).lower().split()),
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(node)
    return result


def _apply_graph_patches(
    kind: GraphKind,
    graph: SourceEvidenceGraphOutput | ResponseClaimGraphOutput,
    patches: list[GraphPatch],
    *,
    response: str,
    allowed_evidence_ids: set[str],
) -> tuple[SourceEvidenceGraphOutput | ResponseClaimGraphOutput, dict[str, Any]]:
    accepted = {patch.id: patch for patch in patches}
    applied: list[str] = []
    rejected: list[dict[str, str]] = []

    if kind == "source":
        assert isinstance(graph, SourceEvidenceGraphOutput)
        nodes_by_id = {node.id: copy.deepcopy(node) for node in graph.nodes}
        edges = [copy.deepcopy(edge) for edge in graph.edges]
        for patch in accepted.values():
            try:
                if patch.operation == "drop_node" and patch.target_id in nodes_by_id:
                    nodes_by_id.pop(patch.target_id)
                    edges = [e for e in edges if e.target_id != patch.target_id and patch.target_id not in e.source_ids]
                elif patch.operation == "replace_source_node" and patch.target_id in nodes_by_id and patch.source_node:
                    if not _valid_source_node(patch.source_node, allowed_evidence_ids):
                        raise ValueError("replacement source node failed validation")
                    replacement = copy.deepcopy(patch.source_node)
                    replacement.id = patch.target_id
                    nodes_by_id[patch.target_id] = replacement
                elif patch.operation == "add_source_node" and patch.source_node:
                    if not _valid_source_node(patch.source_node, allowed_evidence_ids):
                        raise ValueError("added source node failed validation")
                    temp_id = patch.source_node.id or f"ADD_{patch.id}"
                    nodes_by_id[temp_id] = copy.deepcopy(patch.source_node)
                elif patch.operation == "drop_edge":
                    edges = [e for e in edges if e.target_id != patch.target_id]
                elif patch.operation == "replace_source_edge" and patch.source_edge:
                    edges = [e for e in edges if e.target_id != patch.target_id]
                    edges.append(copy.deepcopy(patch.source_edge))
                else:
                    raise ValueError("patch operation is not applicable to source graph")
                applied.append(patch.id)
            except Exception as exc:
                rejected.append({"patch_id": patch.id, "reason": str(exc)})
        old_nodes = _dedup_source_nodes(list(nodes_by_id.values()))
        old_to_new: dict[str, str] = {}
        new_nodes: list[SourceGraphNode] = []
        for index, node in enumerate(old_nodes, 1):
            old_to_new[node.id] = f"S{index}"
            payload = node.model_dump()
            payload["id"] = f"S{index}"
            new_nodes.append(SourceGraphNode.model_validate(payload))
        new_edges: list[SourceGraphEdge] = []
        seen_edges: set[tuple[tuple[str, ...], str, str]] = set()
        for edge in edges:
            if edge.target_id not in old_to_new or any(x not in old_to_new for x in edge.source_ids):
                continue
            mapped = SourceGraphEdge(
                source_ids=[old_to_new[x] for x in edge.source_ids],
                target_id=old_to_new[edge.target_id],
                relation=edge.relation,
            )
            key = (tuple(mapped.source_ids), mapped.target_id, mapped.relation)
            if key not in seen_edges:
                seen_edges.add(key)
                new_edges.append(mapped)
        return SourceEvidenceGraphOutput(nodes=new_nodes, edges=new_edges, summary=graph.summary), {
            "applied_patch_ids": applied, "rejected_patches": rejected,
        }

    assert isinstance(graph, ResponseClaimGraphOutput)
    nodes_by_id = {node.id: copy.deepcopy(node) for node in graph.nodes}
    edges = [copy.deepcopy(edge) for edge in graph.edges]
    for patch in accepted.values():
        try:
            if patch.operation == "drop_node" and patch.target_id in nodes_by_id:
                nodes_by_id.pop(patch.target_id)
                edges = [e for e in edges if e.target_id != patch.target_id and patch.target_id not in e.source_ids]
            elif patch.operation == "replace_response_node" and patch.target_id in nodes_by_id and patch.response_node:
                if not _valid_response_node(patch.response_node, response):
                    raise ValueError("replacement response node failed validation")
                replacement = copy.deepcopy(patch.response_node)
                replacement.id = patch.target_id
                nodes_by_id[patch.target_id] = replacement
            elif patch.operation == "add_response_node" and patch.response_node:
                if not _valid_response_node(patch.response_node, response):
                    raise ValueError("added response node failed validation")
                temp_id = patch.response_node.id or f"ADD_{patch.id}"
                nodes_by_id[temp_id] = copy.deepcopy(patch.response_node)
            elif patch.operation == "mark_response_ineligible" and patch.target_id in nodes_by_id:
                payload = nodes_by_id[patch.target_id].model_dump()
                payload["evaluation_eligible"] = False
                nodes_by_id[patch.target_id] = ResponseClaimNode.model_validate(payload)
            elif patch.operation == "drop_edge":
                edges = [e for e in edges if e.target_id != patch.target_id]
            elif patch.operation == "replace_response_edge" and patch.response_edge:
                edges = [e for e in edges if e.target_id != patch.target_id]
                edges.append(copy.deepcopy(patch.response_edge))
            else:
                raise ValueError("patch operation is not applicable to response graph")
            applied.append(patch.id)
        except Exception as exc:
            rejected.append({"patch_id": patch.id, "reason": str(exc)})
    old_nodes = _dedup_response_nodes(list(nodes_by_id.values()))
    old_to_new: dict[str, str] = {}
    new_nodes: list[ResponseClaimNode] = []
    for index, node in enumerate(old_nodes, 1):
        old_to_new[node.id] = f"R{index}"
        payload = node.model_dump()
        payload["id"] = f"R{index}"
        new_nodes.append(ResponseClaimNode.model_validate(payload))
    new_edges: list[ResponseClaimEdge] = []
    seen_edges: set[tuple[tuple[str, ...], str, str]] = set()
    for edge in edges:
        if edge.target_id not in old_to_new or any(x not in old_to_new for x in edge.source_ids):
            continue
        mapped = ResponseClaimEdge(
            source_ids=[old_to_new[x] for x in edge.source_ids],
            target_id=old_to_new[edge.target_id],
            relation=edge.relation,
        )
        key = (tuple(mapped.source_ids), mapped.target_id, mapped.relation)
        if key not in seen_edges:
            seen_edges.add(key)
            new_edges.append(mapped)
    return ResponseClaimGraphOutput(
        nodes=new_nodes,
        edges=new_edges,
        coverage_check=graph.coverage_check,
        summary=graph.summary,
    ), {"applied_patch_ids": applied, "rejected_patches": rejected}


def _validate_graph_six_agents(
    kind: GraphKind,
    *,
    case: dict[str, Any],
    evidence_card: dict[str, Any],
    graph: SourceEvidenceGraphOutput | ResponseClaimGraphOutput,
    client: Any,
    model: str,
    reasoning_effort: str,
    max_output_tokens: int,
    cache: dict[str, Any],
    cache_lock: threading.Lock,
    cache_path: Path | None,
    force: set[str],
) -> tuple[SourceEvidenceGraphOutput | ResponseClaimGraphOutput, list[dict[str, Any]], dict[str, Any], int, int]:
    records: list[dict[str, Any]] = []
    cache_hits = 0
    api_calls = 0
    original_hash_payload = {
        "graph_owner_id": case.get("source_id") if kind == "source" else case["case_id"],
        "task": case["task_instruction"],
        "original": _validation_original_block(kind, case=case, evidence_card=evidence_card),
        "graph": graph.model_dump(),
        "model": model,
        "reasoning_effort": reasoning_effort,
    }

    def run_specialist(specialist: ValidationSpecialist) -> tuple[str, dict[str, Any], bool]:
        component = f"{kind}_{specialist}"
        system, user = _validation_specialist_prompts(
            specialist, kind, case=case, evidence_card=evidence_card, graph=graph
        )
        record, hit = _cached_parsed_call(
            cache=cache, cache_lock=cache_lock, cache_path=cache_path, force=force,
            component=component, payload={**original_hash_payload, "max_output_tokens": max_output_tokens},
            client=client, model=model, reasoning_effort=reasoning_effort,
            max_output_tokens=max_output_tokens, system=system, user=user,
            output_type=GraphSpecialistReviewOutput,
        )
        return specialist, record, hit

    reviews: list[GraphSpecialistReviewOutput] = []
    specialist_status: dict[str, Any] = {}
    with ThreadPoolExecutor(max_workers=3, thread_name_prefix=f"{kind}-agent") as executor:
        futures = [executor.submit(run_specialist, specialist) for specialist in ("evidence", "logic", "target")]
        for future in as_completed(futures):
            specialist, record, hit = future.result()
            records.append(record)
            cache_hits += int(hit)
            api_calls += 0 if hit else int(record.get("api_calls") or 1)
            review = _normalize_review(GraphSpecialistReviewOutput.model_validate(record["parsed"]), specialist)
            reviews.append(review)
            specialist_status[specialist] = {
                "status": "cache_hit" if hit else "ok",
                "finding_count": len(review.findings),
                "api_calls": 0 if hit else int(record.get("api_calls") or 1),
            }

    specialist_order = {"evidence": 0, "logic": 1, "target": 2}
    reviews.sort(key=lambda review: specialist_order.get(review.specialist, 99))
    records.sort(key=lambda record: specialist_order.get(str(record.get("component", "")).rsplit("_", 1)[-1], 99))
    findings = [finding for review in reviews for finding in review.findings]
    assumption_findings = _findings_need_assumption(findings)
    assumptions: list[GraphAssumptionAssessment] = []
    assumption_component = f"{kind}_assumption"
    if assumption_findings:
        system, user = _assumption_prompts(
            kind, case=case, evidence_card=evidence_card, graph=graph, findings=assumption_findings
        )
        record, hit = _cached_parsed_call(
            cache=cache, cache_lock=cache_lock, cache_path=cache_path, force=force,
            component=assumption_component,
            payload={**original_hash_payload, "findings": [x.model_dump() for x in assumption_findings]},
            client=client, model=model, reasoning_effort=reasoning_effort,
            max_output_tokens=max(1200, min(2400, max_output_tokens)),
            system=system, user=user, output_type=GraphAssumptionReviewOutput,
        )
        records.append(record)
        cache_hits += int(hit)
        api_calls += 0 if hit else int(record.get("api_calls") or 1)
        assumptions = GraphAssumptionReviewOutput.model_validate(record["parsed"]).assessments
        assumption_status = {
            "status": "cache_hit" if hit else "ok",
            "candidate_count": len(assumption_findings),
            "assessment_count": len(assumptions),
        }
    else:
        assumption_status = {"status": "not_needed", "candidate_count": 0, "assessment_count": 0}

    judge_reasons = _judge_trigger_reasons(findings)
    all_patches = [patch for finding in findings for patch in finding.patches]
    selected_patches: list[GraphPatch] = []
    judge_status: dict[str, Any]
    if findings and judge_reasons:
        component = f"{kind}_judge"
        system, user = _judge_prompts(
            kind, case=case, evidence_card=evidence_card, graph=graph,
            findings=findings, assumptions=assumptions,
        )
        record, hit = _cached_parsed_call(
            cache=cache, cache_lock=cache_lock, cache_path=cache_path, force=force,
            component=component,
            payload={
                **original_hash_payload,
                "findings": [x.model_dump() for x in findings],
                "assumptions": [x.model_dump() for x in assumptions],
            },
            client=client, model=model, reasoning_effort=reasoning_effort,
            max_output_tokens=max(1400, min(3000, max_output_tokens)),
            system=system, user=user, output_type=GraphJudgeOutput,
        )
        records.append(record)
        cache_hits += int(hit)
        api_calls += 0 if hit else int(record.get("api_calls") or 1)
        judged = GraphJudgeOutput.model_validate(record["parsed"])
        patch_map = {patch.id: patch for patch in all_patches}
        selected_patches = [patch_map[x] for x in judged.accepted_patch_ids if x in patch_map]
        selected_patches.extend(judged.revised_patches)
        judge_status = {
            "status": "cache_hit" if hit else "ok",
            "trigger_reasons": judge_reasons,
            "accepted_patch_count": len(selected_patches),
        }
    else:
        judge_status = {
            "status": "not_needed",
            "trigger_reasons": judge_reasons,
            "accepted_patch_count": 0,
        }

    allowed_evidence_ids = {str(unit.get("id") or "") for unit in evidence_card.get("units") or []}
    validated_graph, patch_trace = _apply_graph_patches(
        kind, graph, selected_patches,
        response=case["response"], allowed_evidence_ids=allowed_evidence_ids,
    )
    trace = {
        "kind": kind,
        "architecture": "compiler_evidence_logic_target_assumption_judge",
        "compiler": {
            "version": PROMPT_VERSIONS[f"{kind}_compiler"],
            "status": "completed_before_validation",
        },
        "specialists": specialist_status,
        "assumption": assumption_status,
        "judge": judge_status,
        "finding_count": len(findings),
        "patches_proposed": len(all_patches),
        "patches_selected": len(selected_patches),
        **patch_trace,
        "findings": [x.model_dump() for x in findings],
        "assumption_assessments": [x.model_dump() for x in assumptions],
    }
    return validated_graph, records, trace, api_calls, cache_hits



def _complete_cross_verdicts(
    response_graph: ResponseClaimGraphOutput,
    verdicts: list[CrossLogicVerdict],
) -> tuple[list[CrossLogicVerdict], list[str]]:
    eligible_ids = [node.id for node in response_graph.nodes if node.evaluation_eligible]
    valid_ids = set(eligible_ids)
    by_id: dict[str, CrossLogicVerdict] = {}
    for verdict in verdicts:
        if verdict.response_node_id in valid_ids:
            by_id[verdict.response_node_id] = verdict
    missing: list[str] = []
    completed: list[CrossLogicVerdict] = []
    for node_id in eligible_ids:
        verdict = by_id.get(node_id)
        if verdict is None:
            missing.append(node_id)
            verdict = CrossLogicVerdict(
                response_node_id=node_id,
                verdict="uncertain",
                confidence=0.0,
                explanation="Cross-Graph Logic omitted this eligible response node.",
            )
        completed.append(verdict)
    return completed, missing


def _complete_projector_alignments(
    response_graph: ResponseClaimGraphOutput,
    verdicts: list[CrossLogicVerdict],
    output: DualGraphAlignmentOutput,
) -> tuple[DualGraphAlignmentOutput, list[str]]:
    by_alignment = {row.response_node_id: row for row in output.alignments}
    verdict_by_id = {row.response_node_id: row for row in verdicts}
    node_by_id = {node.id: node for node in response_graph.nodes}
    missing: list[str] = []
    completed: list[AlignmentRecord] = []
    for node in response_graph.nodes:
        if not node.evaluation_eligible:
            continue
        existing = by_alignment.get(node.id)
        if existing is not None:
            completed.append(existing)
            continue
        missing.append(node.id)
        verdict = verdict_by_id.get(node.id)
        if verdict is None or verdict.verdict in {"supported_by", "safe_inference", "not_factual", "uncertain", "requires_assumption"}:
            completed.append(AlignmentRecord(
                response_node_id=node.id,
                source_node_ids=list(verdict.source_node_ids) if verdict else [],
                relation="supported_by",
                problem_text="",
                label_type="none",
                confidence=float(verdict.confidence) if verdict else 0.0,
                explanation="Deterministic conservative fallback for a projector-omitted node.",
            ))
            continue
        relation = verdict.verdict
        if relation not in {"partially_supported_by", "contradicted_by", "not_found_in_source", "qualified_by"}:
            relation = "not_found_in_source"
        completed.append(AlignmentRecord(
            response_node_id=node.id,
            source_node_ids=list(verdict.source_node_ids),
            relation=relation,
            problem_text=node.text,
            label_type="contradiction" if relation == "contradicted_by" else "unsupported",
            confidence=float(verdict.confidence),
            explanation="Projector omitted a final problematic verdict; used the complete claim node as a conservative localization fallback.",
        ))
    return DualGraphAlignmentOutput(alignments=completed, summary=output.summary), missing

def _cross_match_prompts(
    case: dict[str, Any],
    evidence_card: dict[str, Any],
    source_graph: SourceEvidenceGraphOutput,
    response_graph: ResponseClaimGraphOutput,
) -> tuple[str, str]:
    system = (
        "You are the Cross-Graph Evidence Matcher. For every evaluation-eligible RESPONSE node, retrieve the most relevant "
        "validated SOURCE nodes and evidence ids. Do not decide the final hallucination verdict. Use semantic matching, not "
        "word overlap alone. Include potentially conflicting and qualifying evidence. Return none only after checking the exact "
        "source passages. Never use outside knowledge."
    )
    user = (
        f"TASK\n{case['task_instruction']}\n\nSOURCE EVIDENCE\n{evidence_card['text']}\n\n"
        f"VALIDATED SOURCE GRAPH\n{json.dumps(source_graph.model_dump(), ensure_ascii=False)}\n\n"
        f"VALIDATED RESPONSE GRAPH\n{json.dumps(response_graph.model_dump(), ensure_ascii=False)}"
    )
    return system, user


def _cross_logic_prompts(
    case: dict[str, Any],
    evidence_card: dict[str, Any],
    source_graph: SourceEvidenceGraphOutput,
    response_graph: ResponseClaimGraphOutput,
    matches: CrossEvidenceMatchOutput,
) -> tuple[str, str]:
    system = (
        "You are the Cross-Graph Logic Agent. Treat the SOURCE as authoritative for this benchmark and use no outside knowledge. "
        "For every eligible RESPONSE normalized_claim, decide whether it is supported, safely entailed, partially supported, "
        "directly contradicted, qualified by an omitted/changed limitation, genuinely unsupported, not factual, requires one "
        "concrete missing premise, or remains uncertain. A missing lexical match is not enough for unsupported. Re-check the exact "
        "source passages. Safe paraphrase and task-bounded inference are supported. Generic discourse or non-factual advice is "
        "not_factual unless it contains a concrete world claim. Distinguish direct substitutions from wholly unsupported claims. "
        "Use requires_assumption only with a specific premise."
    )
    user = (
        f"TASK\n{case['task_instruction']}\n\nSOURCE EVIDENCE\n{evidence_card['text']}\n\n"
        f"VALIDATED SOURCE GRAPH\n{json.dumps(source_graph.model_dump(), ensure_ascii=False)}\n\n"
        f"VALIDATED RESPONSE GRAPH\n{json.dumps(response_graph.model_dump(), ensure_ascii=False)}\n\n"
        f"EVIDENCE MATCHES\n{json.dumps(matches.model_dump(), ensure_ascii=False)}"
    )
    return system, user


def _cross_assumption_prompts(
    case: dict[str, Any],
    evidence_card: dict[str, Any],
    source_graph: SourceEvidenceGraphOutput,
    response_graph: ResponseClaimGraphOutput,
    verdicts: list[CrossLogicVerdict],
) -> tuple[str, str]:
    system = (
        "You are the conditional Cross-Graph Assumption Agent. Review only response claims marked requires_assumption or uncertain "
        "because of an implicit premise. Decide whether the premise is explicit in SOURCE, a safe linguistic inference, external "
        "knowledge, unsupported and missing, not actually required, or uncertain. Do not add outside knowledge. Return a resulting "
        "verdict for each reviewed response node."
    )
    user = (
        f"TASK\n{case['task_instruction']}\n\nSOURCE EVIDENCE\n{evidence_card['text']}\n\n"
        f"SOURCE GRAPH\n{json.dumps(source_graph.model_dump(), ensure_ascii=False)}\n\n"
        f"RESPONSE GRAPH\n{json.dumps(response_graph.model_dump(), ensure_ascii=False)}\n\n"
        f"CANDIDATE VERDICTS\n{json.dumps([x.model_dump() for x in verdicts], ensure_ascii=False)}"
    )
    return system, user


def _cross_judge_prompts(
    case: dict[str, Any],
    evidence_card: dict[str, Any],
    source_graph: SourceEvidenceGraphOutput,
    response_graph: ResponseClaimGraphOutput,
    matches: CrossEvidenceMatchOutput,
    verdicts: list[CrossLogicVerdict],
    assumptions: list[CrossAssumptionAssessment],
) -> tuple[str, str]:
    system = (
        "You are the conditional Cross-Graph Judge. Adjudicate only uncertain, low-confidence, assumption-dependent, or internally "
        "inconsistent cross-graph verdicts. SOURCE is authoritative. Prefer supported/safe inference when the exact source entails "
        "the response without a new factual premise; prefer unsupported only for genuine new factual content. Preserve direct "
        "contradictions and material qualifier changes. Return one revised verdict per supplied response node, without localizing spans."
    )
    user = (
        f"TASK\n{case['task_instruction']}\n\nSOURCE EVIDENCE\n{evidence_card['text']}\n\n"
        f"SOURCE GRAPH\n{json.dumps(source_graph.model_dump(), ensure_ascii=False)}\n\n"
        f"RESPONSE GRAPH\n{json.dumps(response_graph.model_dump(), ensure_ascii=False)}\n\n"
        f"MATCHES\n{json.dumps(matches.model_dump(), ensure_ascii=False)}\n\n"
        f"LOGIC VERDICTS\n{json.dumps([x.model_dump() for x in verdicts], ensure_ascii=False)}\n\n"
        f"ASSUMPTION ASSESSMENTS\n{json.dumps([x.model_dump() for x in assumptions], ensure_ascii=False)}"
    )
    return system, user


def _span_projector_prompts(
    case: dict[str, Any],
    response_graph: ResponseClaimGraphOutput,
    verdicts: list[CrossLogicVerdict],
) -> tuple[str, str]:
    system = (
        "You are the final Span Projector. Do not reconsider factual verdicts. Convert the supplied final verdict for every eligible "
        "response node into a DualGraphAlignmentOutput and copy problem_text exactly from the RESPONSE. For contradicted_by, return "
        "the smallest differing entity, number, polarity, time, scope, or qualifier when that isolated substitution is the sole error. "
        "For partially_supported_by or qualified_by, return only the unsupported material modifier or clause. For not_found_in_source, "
        "return the smallest semantically complete unsupported proposition or clause, not a discourse marker or bare fragment. For "
        "supported_by, safe_inference, not_factual, or uncertain, emit label_type none and empty problem_text. Map safe_inference and "
        "not_factual to supported_by for the output schema; map uncertain to supported_by unless the final verdict explicitly identifies "
        "a factual error. Use contradiction label only for contradicted_by and unsupported for other problematic verdicts."
    )
    user = (
        f"RESPONSE\n{case['response']}\n\n"
        f"VALIDATED RESPONSE GRAPH\n{json.dumps(response_graph.model_dump(), ensure_ascii=False)}\n\n"
        f"FINAL CROSS-GRAPH VERDICTS\n{json.dumps([x.model_dump() for x in verdicts], ensure_ascii=False)}"
    )
    return system, user


def _apply_cross_assumptions(
    verdicts: list[CrossLogicVerdict],
    assessments: list[CrossAssumptionAssessment],
) -> list[CrossLogicVerdict]:
    by_node = {x.response_node_id: x for x in assessments}
    result: list[CrossLogicVerdict] = []
    for verdict in verdicts:
        assessment = by_node.get(verdict.response_node_id)
        if assessment is None:
            result.append(verdict)
            continue
        payload = verdict.model_dump()
        payload["verdict"] = assessment.resulting_verdict
        payload["confidence"] = max(float(verdict.confidence), float(assessment.confidence))
        payload["explanation"] = f"{verdict.explanation} Assumption review: {assessment.explanation}".strip()
        result.append(CrossLogicVerdict.model_validate(payload))
    return result


def _cross_judge_needed(verdicts: list[CrossLogicVerdict], assumptions: list[CrossAssumptionAssessment]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if any(v.verdict == "uncertain" for v in verdicts):
        reasons.append("uncertain_verdict")
    if any(v.verdict in {"contradicted_by", "partially_supported_by", "not_found_in_source", "qualified_by"} and v.confidence < 0.78 for v in verdicts):
        reasons.append("low_confidence_problematic_verdict")
    if assumptions:
        reasons.append("assumption_review_used")
    return bool(reasons), reasons


def _run_cross_graph_pipeline(
    *,
    case: dict[str, Any],
    evidence_card: dict[str, Any],
    source_graph: SourceEvidenceGraphOutput,
    response_graph: ResponseClaimGraphOutput,
    client: Any,
    model: str,
    reasoning_effort: str,
    max_output_tokens: int,
    cache: dict[str, Any],
    cache_lock: threading.Lock,
    cache_path: Path | None,
    force: set[str],
) -> tuple[DualGraphAlignmentOutput, list[dict[str, Any]], dict[str, Any], int, int]:
    records: list[dict[str, Any]] = []
    api_calls = 0
    cache_hits = 0
    common = {
        "case_id": case["case_id"], "task": case["task_instruction"],
        "response": case["response"], "evidence_units": evidence_card["units"],
        "source_graph": source_graph.model_dump(), "response_graph": response_graph.model_dump(),
        "model": model, "reasoning_effort": reasoning_effort,
    }

    system, user = _cross_match_prompts(case, evidence_card, source_graph, response_graph)
    match_record, hit = _cached_parsed_call(
        cache=cache, cache_lock=cache_lock, cache_path=cache_path, force=force,
        component="cross_matcher", payload=common, client=client, model=model,
        reasoning_effort=reasoning_effort, max_output_tokens=max_output_tokens,
        system=system, user=user, output_type=CrossEvidenceMatchOutput,
    )
    records.append(match_record); cache_hits += int(hit); api_calls += 0 if hit else int(match_record.get("api_calls") or 1)
    matches = CrossEvidenceMatchOutput.model_validate(match_record["parsed"])

    system, user = _cross_logic_prompts(case, evidence_card, source_graph, response_graph, matches)
    logic_record, hit = _cached_parsed_call(
        cache=cache, cache_lock=cache_lock, cache_path=cache_path, force=force,
        component="cross_logic", payload={**common, "matches": matches.model_dump()}, client=client, model=model,
        reasoning_effort=reasoning_effort, max_output_tokens=max_output_tokens,
        system=system, user=user, output_type=CrossLogicOutput,
    )
    records.append(logic_record); cache_hits += int(hit); api_calls += 0 if hit else int(logic_record.get("api_calls") or 1)
    logic = CrossLogicOutput.model_validate(logic_record["parsed"])
    verdicts, logic_missing_node_ids = _complete_cross_verdicts(response_graph, list(logic.verdicts))

    assumption_candidates = [v for v in verdicts if v.verdict in {"requires_assumption", "uncertain"} and v.missing_premise.strip()]
    assumptions: list[CrossAssumptionAssessment] = []
    if assumption_candidates:
        system, user = _cross_assumption_prompts(
            case, evidence_card, source_graph, response_graph, assumption_candidates
        )
        assumption_record, hit = _cached_parsed_call(
            cache=cache, cache_lock=cache_lock, cache_path=cache_path, force=force,
            component="cross_assumption", payload={**common, "candidates": [x.model_dump() for x in assumption_candidates]},
            client=client, model=model, reasoning_effort=reasoning_effort,
            max_output_tokens=max(1200, min(2400, max_output_tokens)),
            system=system, user=user, output_type=CrossAssumptionOutput,
        )
        records.append(assumption_record); cache_hits += int(hit); api_calls += 0 if hit else int(assumption_record.get("api_calls") or 1)
        assumptions = CrossAssumptionOutput.model_validate(assumption_record["parsed"]).assessments
        verdicts = _apply_cross_assumptions(verdicts, assumptions)
        assumption_status = "cache_hit" if hit else "ok"
    else:
        assumption_status = "not_needed"

    judge_needed, judge_reasons = _cross_judge_needed(verdicts, assumptions)
    if judge_needed:
        system, user = _cross_judge_prompts(
            case, evidence_card, source_graph, response_graph, matches, verdicts, assumptions
        )
        judge_record, hit = _cached_parsed_call(
            cache=cache, cache_lock=cache_lock, cache_path=cache_path, force=force,
            component="cross_judge", payload={
                **common, "matches": matches.model_dump(),
                "verdicts": [x.model_dump() for x in verdicts],
                "assumptions": [x.model_dump() for x in assumptions],
            },
            client=client, model=model, reasoning_effort=reasoning_effort,
            max_output_tokens=max(1400, min(3000, max_output_tokens)),
            system=system, user=user, output_type=CrossJudgeOutput,
        )
        records.append(judge_record); cache_hits += int(hit); api_calls += 0 if hit else int(judge_record.get("api_calls") or 1)
        judged = CrossJudgeOutput.model_validate(judge_record["parsed"])
        if judged.revised_verdicts:
            revised_map = {x.response_node_id: x for x in judged.revised_verdicts}
            verdicts = [revised_map.get(v.response_node_id, v) for v in verdicts]
        verdicts, judge_missing_node_ids = _complete_cross_verdicts(response_graph, verdicts)
        judge_status = "cache_hit" if hit else "ok"
    else:
        judge_status = "not_needed"
        judge_missing_node_ids = []

    system, user = _span_projector_prompts(case, response_graph, verdicts)
    projector_record, hit = _cached_parsed_call(
        cache=cache, cache_lock=cache_lock, cache_path=cache_path, force=force,
        component="span_projector", payload={**common, "verdicts": [x.model_dump() for x in verdicts]},
        client=client, model=model, reasoning_effort=reasoning_effort,
        max_output_tokens=max_output_tokens, system=system, user=user,
        output_type=DualGraphAlignmentOutput,
    )
    records.append(projector_record); cache_hits += int(hit); api_calls += 0 if hit else int(projector_record.get("api_calls") or 1)
    output = DualGraphAlignmentOutput.model_validate(projector_record["parsed"])
    output, projector_missing_node_ids = _complete_projector_alignments(response_graph, verdicts, output)
    trace = {
        "matcher": {"status": "completed", "match_count": len(matches.matches)},
        "logic": {"status": "completed", "verdict_count": len(verdicts), "omitted_node_ids_filled": logic_missing_node_ids},
        "assumption": {"status": assumption_status, "candidate_count": len(assumption_candidates)},
        "judge": {"status": judge_status, "trigger_reasons": judge_reasons, "omitted_node_ids_filled": judge_missing_node_ids},
        "projector": {"status": "completed", "alignment_count": len(output.alignments), "omitted_node_ids_filled": projector_missing_node_ids},
        "matches": matches.model_dump(),
        "final_verdicts": [x.model_dump() for x in verdicts],
        "assumption_assessments": [x.model_dump() for x in assumptions],
    }
    return output, records, trace, api_calls, cache_hits


def _alias_rows_for_legacy_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    aliased = copy.deepcopy(rows)
    for row in aliased:
        methods = row.get("methods") or {}
        if GRAPH_METHOD in methods:
            methods[LEGACY_GRAPH_METHOD] = methods[GRAPH_METHOD]
    return aliased


def _method_summary(rows: list[dict[str, Any]], method: str) -> dict[str, Any]:
    if method == GRAPH_METHOD:
        return _legacy_method_summary(_alias_rows_for_legacy_summary(rows), LEGACY_GRAPH_METHOD)
    return _legacy_method_summary(rows, method)


def _short_spans(spans: list[dict[str, Any]], max_items: int = 5) -> str:
    if not spans:
        return "[none]"
    values: list[str] = []
    for span in spans[:max_items]:
        text = " ".join(str(span.get("text") or "").split())
        if len(text) > 80:
            text = text[:77] + "..."
        values.append(f"[{span.get('start')}:{span.get('end')}] {text!r}")
    if len(spans) > max_items:
        values.append(f"... +{len(spans)-max_items} more")
    return "; ".join(values)


def _running_micro_f1(rows: list[dict[str, Any]], method: str) -> float | None:
    scores = [row["methods"][method]["scores"] for row in rows if row.get("methods", {}).get(method, {}).get("scores")]
    if not scores:
        return None
    tp = sum(int(x["char_tp"]) for x in scores); fp = sum(int(x["char_fp"]) for x in scores); fn = sum(int(x["char_fn"]) for x in scores)
    p = tp / (tp + fp) if tp + fp else 1.0; r = tp / (tp + fn) if tp + fn else 1.0
    return 2 * p * r / (p + r) if p + r else 0.0


def _emit_case(row: dict[str, Any], rows: list[dict[str, Any]], progress: Callable[[str], None]) -> None:
    raw = row["methods"][RAW_METHOD]; graph = row["methods"][GRAPH_METHOD]
    progress("    +---------------- CASE PERFORMANCE ----------------+")
    progress(f"    Gold       spans={_short_spans(row.get('gold_labels') or [])}")
    for label, output in (("Raw", raw), ("SixAgent", graph)):
        if output.get("scores"):
            s = output["scores"]
            progress(f"    {label:<10} P={s['char_precision']*100:6.2f}% R={s['char_recall']*100:6.2f}% F1={s['char_f1']*100:6.2f}% | detect={'OK' if s['response_detection_correct'] else 'MISS'} | spans={_short_spans(output.get('predicted_spans') or [])}")
        else:
            progress(f"    {label:<10} ERROR")
    details = graph.get("details") or {}
    diag = details.get("response_graph_diagnostics") or {}
    ref = details.get("response_compiler_refinement") or {}
    progress(
        "    Compiler   "
        f"v043-balanced | claims={diag.get('claim_count', 0)} | fragments={len(diag.get('incomplete_fragment_node_ids') or [])} | "
        f"local-repair={'YES' if ref.get('attempted') else 'NO'}"
    )
    for kind, label in (("source_validation", "Source6"), ("response_validation", "Response6")):
        trace = details.get(kind) or {}
        specialists = trace.get("specialists") or {}
        progress(
            f"    {label:<10} Evidence={specialists.get('evidence',{}).get('status','?')} | "
            f"Logic={specialists.get('logic',{}).get('status','?')} | "
            f"Target={specialists.get('target',{}).get('status','?')} | "
            f"Assumption={trace.get('assumption',{}).get('status','?')} | "
            f"Judge={trace.get('judge',{}).get('status','?')} | patches={len(trace.get('applied_patch_ids') or [])}"
        )
    cross = details.get("cross_graph") or {}
    progress(
        f"    CrossGraph Matcher={cross.get('matcher',{}).get('status','?')} | Logic={cross.get('logic',{}).get('status','?')} | "
        f"Assumption={cross.get('assumption',{}).get('status','?')} | Judge={cross.get('judge',{}).get('status','?')} | Projector={cross.get('projector',{}).get('status','?')}"
    )
    if raw.get("scores") and graph.get("scores"):
        delta = (graph["scores"]["char_f1"] - raw["scores"]["char_f1"]) * 100
        winner = "SIX_AGENT" if delta > 0 else "RAW_DIRECT" if delta < 0 else "TIE"
        progress(f"    Case delta SixAgent-Raw = {delta:+.2f} pp | winner={winner}")
    raw_run = _running_micro_f1(rows, RAW_METHOD); graph_run = _running_micro_f1(rows, GRAPH_METHOD)
    progress(f"    Running micro-F1 after {len(rows)} case(s): Raw={raw_run*100:.2f}% | SixAgent={graph_run*100:.2f}%")
    progress("    +--------------------------------------------------+")


def _report_html(result: dict[str, Any]) -> str:
    raw = result["method_summaries"][RAW_METHOD]
    graph = result["method_summaries"][GRAPH_METHOD]
    return f"""<!doctype html><meta charset='utf-8'><title>RAGTruth true six-agent dual graph</title>
<style>body{{font-family:Arial,sans-serif;margin:28px;color:#172033}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #d7dee9;padding:8px}}th{{background:#f3f6fa}}.note{{background:#eef6ff;padding:12px;border-radius:10px;line-height:1.5}}</style>
<h1>RAGTruth · Raw Direct vs True Six-Agent Dual Graph</h1>
<p class='note'>The Source and Response are independently compiled with the v043 balanced compilers, then each graph is validated by Evidence, Logic, Target, conditional Assumption, and conditional Judge agents. The validated graphs are compared by Cross-Graph Matcher, Logic, conditional Assumption/Judge, and a relation-aware Span Projector. All model calls use {result['settings']['model']}.</p>
<table><tr><th>Method</th><th>N</th><th>Character F1</th><th>Precision</th><th>Recall</th><th>Clean FP</th><th>API calls</th><th>Tokens</th></tr>
<tr><td>{RAW_METHOD}</td><td>{raw['n']}</td><td>{raw['char_f1_percent']}%</td><td>{raw['char_precision_percent']}%</td><td>{raw['char_recall_percent']}%</td><td>{raw['clean_false_positive_rate_percent']}%</td><td>{raw['api_calls']}</td><td>{raw['total_tokens']}</td></tr>
<tr><td>{GRAPH_METHOD}</td><td>{graph['n']}</td><td>{graph['char_f1_percent']}%</td><td>{graph['char_precision_percent']}%</td><td>{graph['char_recall_percent']}%</td><td>{graph['clean_false_positive_rate_percent']}%</td><td>{graph['api_calls']}</td><td>{graph['total_tokens']}</td></tr></table>
<p class='note'>This is a system-level architecture comparison with unequal compute. More agents can improve verification but do not guarantee better performance.</p>"""


def run_ragtruth_raw_vs_true_six_agent_dual_graph(
    *,
    response_path: Path,
    source_path: Path,
    output_root: Path,
    model: str = "gpt-5.4-nano",
    split: str = "test",
    quality: str = "good",
    task_types: list[str] | None = None,
    limit: int = 24,
    seed: int = 2040,
    reasoning_effort: str = "low",
    max_output_tokens_direct: int = 1800,
    max_output_tokens_source_graph: int = 3200,
    max_output_tokens_response_graph: int = 3600,
    max_output_tokens_validation: int = 2600,
    max_output_tokens_cross: int = 2800,
    max_context_chars: int = 60_000,
    max_response_chars: int = 3000,
    include_implicit_true: bool = True,
    exclude_case_ids: set[str] | None = None,
    require_full_evidence: bool = True,
    generation_cache_path: Path | None = None,
    force_components: set[str] | None = None,
    print_case_comparison: bool = True,
    client: Any = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    _load_local_env()
    progress = progress or (lambda _message: None)
    if model != "gpt-5.4-nano":
        raise ValueError("Both conditions are intentionally fixed to gpt-5.4-nano")
    if reasoning_effort not in ALLOWED_REASONING_EFFORTS:
        raise ValueError("reasoning_effort must be low, medium, or high")
    if not response_path.exists() or not source_path.exists():
        raise FileNotFoundError("RAGTruth files are missing. Run with --download first.")
    force = set(force_components or set())
    unknown = force - set(PROMPT_VERSIONS)
    if unknown:
        raise ValueError(f"Unknown force components: {sorted(unknown)}")

    cases, sampling = load_ragtruth_cases(
        response_path, source_path, split=split, quality=quality, task_types=task_types,
        limit=limit, seed=seed, max_response_chars=max_response_chars,
        include_implicit_true=include_implicit_true, exclude_case_ids=exclude_case_ids,
        require_full_evidence=require_full_evidence, max_context_chars=max_context_chars,
    )
    cache = _load_cache(generation_cache_path)
    cache_lock = threading.Lock()
    active_client = client
    client_lock = threading.Lock()

    def get_client() -> Any:
        nonlocal active_client
        if active_client is not None:
            return active_client
        with client_lock:
            if active_client is None:
                if not os.getenv("OPENAI_API_KEY", "").strip():
                    raise ValueError("OPENAI_API_KEY is not configured for an uncached component call")
                from openai import OpenAI
                active_client = OpenAI()
        return active_client

    run_id = f"ragtruth_true_six_agent_dual_graph_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    actual_api_calls = 0
    cache_hits_by_component: Counter[str] = Counter()

    for index, case in enumerate(cases, 1):
        progress(f"[{index}/{len(cases)}] {case['case_id']} · {case['task_type']} · {'hallucinated' if case['gold_has_hallucination'] else 'clean'}")
        evidence_card = build_evidence_card(
            case["source_info"], case["response"], max_context_chars=max_context_chars,
            force_full=require_full_evidence,
        )
        row = {
            "case_index": index,
            **{key: value for key, value in case.items() if key != "source_info"},
            "evidence_card": evidence_card,
            "methods": {},
        }
        try:
            progress("    - initial: Raw Direct + v043 Source Compiler + v043 Balanced Response Compiler (parallel)")

            def raw_call() -> tuple[str, dict[str, Any], bool]:
                system, user = _raw_prompts(case, evidence_card)
                record, hit = _cached_parsed_call(
                    cache=cache, cache_lock=cache_lock, cache_path=generation_cache_path, force=force,
                    component="raw_direct",
                    payload={"case_id": case["case_id"], "response": case["response"], "task": case["task_instruction"], "evidence_units": evidence_card["units"], "model": model, "reasoning_effort": reasoning_effort},
                    client=get_client(), model=model, reasoning_effort=reasoning_effort,
                    max_output_tokens=max_output_tokens_direct, system=system, user=user, output_type=DirectSpanOutput,
                )
                return "raw", record, hit

            def source_compile() -> tuple[str, dict[str, Any], bool]:
                system, user = _source_graph_prompts(case, evidence_card)
                record, hit = _cached_parsed_call(
                    cache=cache, cache_lock=cache_lock, cache_path=generation_cache_path, force=force,
                    component="source_compiler",
                    payload={"source_id": case["source_id"], "task": case["task_instruction"], "evidence_units": evidence_card["units"], "model": model, "reasoning_effort": reasoning_effort},
                    client=get_client(), model=model, reasoning_effort=reasoning_effort,
                    max_output_tokens=max_output_tokens_source_graph, system=system, user=user, output_type=SourceEvidenceGraphOutput,
                )
                return "source", record, hit

            def response_compile() -> tuple[str, dict[str, Any], bool]:
                record, hit = _cached_response_compiler_call(
                    cache=cache, cache_lock=cache_lock, cache_path=generation_cache_path, force=force,
                    case=case, client=get_client(), model=model, reasoning_effort=reasoning_effort,
                    max_output_tokens=max_output_tokens_response_graph,
                )
                return "response", record, hit

            initial: dict[str, tuple[dict[str, Any], bool]] = {}
            with ThreadPoolExecutor(max_workers=3, thread_name_prefix="ragtruth-initial") as executor:
                futures = [executor.submit(fn) for fn in (raw_call, source_compile, response_compile)]
                for future in as_completed(futures):
                    name, record, hit = future.result()
                    initial[name] = (record, hit)
                    actual_api_calls += 0 if hit else int(record.get("api_calls") or 1)
                    cache_hits_by_component[record["component"]] += int(hit)

            raw_record, raw_hit = initial["raw"]
            source_record, source_hit = initial["source"]
            response_record, response_hit = initial["response"]

            raw_parsed = DirectSpanOutput.model_validate(raw_record["parsed"])
            raw_predictions, raw_details = _predictions_from_direct(raw_parsed, case["response"])
            raw_output = {
                "method": RAW_METHOD, "model": model, "status": "ok",
                "predicted_spans": raw_predictions, "details": raw_details,
                "generation_records": [raw_record],
                "api_calls_this_run": 0 if raw_hit else int(raw_record.get("api_calls") or 1),
                "cache_hits_this_run": int(raw_hit),
                "estimated_cost_usd_this_run": 0.0 if raw_hit else float(raw_record.get("estimated_cost_usd") or 0.0),
            }
            raw_output["scores"] = score_predictions(case["response"], raw_predictions, case["gold_labels"])
            row["methods"][RAW_METHOD] = raw_output

            source_graph = SourceEvidenceGraphOutput.model_validate(source_record["parsed"])
            response_graph = ResponseClaimGraphOutput.model_validate(response_record["parsed"])
            progress("    - Source six-agent graph validation + Response six-agent graph validation (parallel)")

            def validate_source():
                return _validate_graph_six_agents(
                    "source", case=case, evidence_card=evidence_card, graph=source_graph,
                    client=get_client(), model=model, reasoning_effort=reasoning_effort,
                    max_output_tokens=max_output_tokens_validation, cache=cache, cache_lock=cache_lock,
                    cache_path=generation_cache_path, force=force,
                )

            def validate_response():
                return _validate_graph_six_agents(
                    "response", case=case, evidence_card=evidence_card, graph=response_graph,
                    client=get_client(), model=model, reasoning_effort=reasoning_effort,
                    max_output_tokens=max_output_tokens_validation, cache=cache, cache_lock=cache_lock,
                    cache_path=generation_cache_path, force=force,
                )

            with ThreadPoolExecutor(max_workers=2, thread_name_prefix="graph-six-agent") as executor:
                f_source = executor.submit(validate_source)
                f_response = executor.submit(validate_response)
                validated_source, source_validation_records, source_trace, source_calls, source_hits = f_source.result()
                validated_response, response_validation_records, response_trace, response_calls, response_hits = f_response.result()
            assert isinstance(validated_source, SourceEvidenceGraphOutput)
            assert isinstance(validated_response, ResponseClaimGraphOutput)
            actual_api_calls += source_calls + response_calls
            for record in source_validation_records + response_validation_records:
                cache_hits_by_component[record["component"]] += int(bool(record.get("cache_hit")))

            progress("    - Cross-Graph Evidence Matcher → Logic → conditional Assumption/Judge → Span Projector")
            alignment, cross_records, cross_trace, cross_calls, cross_hits = _run_cross_graph_pipeline(
                case=case, evidence_card=evidence_card, source_graph=validated_source,
                response_graph=validated_response, client=get_client(), model=model,
                reasoning_effort=reasoning_effort, max_output_tokens=max_output_tokens_cross,
                cache=cache, cache_lock=cache_lock, cache_path=generation_cache_path, force=force,
            )
            actual_api_calls += cross_calls
            for record in cross_records:
                cache_hits_by_component[record["component"]] += int(bool(record.get("cache_hit")))
            predictions, details = _predictions_from_alignment(
                alignment,
                validated_response,
                case["response"],
                gate_profile="v049_balanced_recall",
            )
            details["source_graph"] = validated_source.model_dump()
            details["candidate_source_graph"] = source_graph.model_dump()
            details["candidate_response_graph"] = response_graph.model_dump()
            details["source_validation"] = source_trace
            details["response_validation"] = response_trace
            details["cross_graph"] = cross_trace
            details["response_compiler_refinement"] = response_record.get("compiler_refinement") or {}
            details["response_graph_diagnostics"] = _response_graph_diagnostics(validated_response, case["response"])

            generation_records = [
                source_record, response_record,
                *source_validation_records, *response_validation_records,
                *cross_records,
            ]
            graph_calls_this_run = (
                (0 if source_hit else int(source_record.get("api_calls") or 1))
                + (0 if response_hit else int(response_record.get("api_calls") or 1))
                + source_calls + response_calls + cross_calls
            )
            graph_output = {
                "method": GRAPH_METHOD, "model": model, "status": "ok",
                "predicted_spans": predictions, "details": details,
                "generation_records": generation_records,
                "api_calls_this_run": graph_calls_this_run,
                "cache_hits_this_run": int(source_hit) + int(response_hit) + source_hits + response_hits + cross_hits,
                "estimated_cost_usd_this_run": round(sum(
                    float(record.get("estimated_cost_usd") or 0.0)
                    for record in generation_records if not record.get("cache_hit")
                ), 8),
            }
            graph_output["scores"] = score_predictions(case["response"], predictions, case["gold_labels"])
            row["methods"][GRAPH_METHOD] = graph_output

        except Exception as exc:
            failures.append({"case_id": case["case_id"], "type": type(exc).__name__, "message": str(exc)})
            progress(f"      ERROR {type(exc).__name__}: {exc}")
            row["methods"].setdefault(RAW_METHOD, {"method": RAW_METHOD, "status": "error", "scores": None, "predicted_spans": []})
            row["methods"][GRAPH_METHOD] = {"method": GRAPH_METHOD, "status": "error", "scores": None, "predicted_spans": [], "error": {"type": type(exc).__name__, "message": str(exc)}}

        rows.append(row)
        if print_case_comparison:
            _emit_case(row, rows, progress)
        with (run_dir / "cases.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summaries = {method: _method_summary(rows, method) for method in METHODS}
    alias_rows = _alias_rows_for_legacy_summary(rows)
    paired = _paired_comparison(alias_rows, LEGACY_GRAPH_METHOD, seed=seed, reference_method=RAW_METHOD)
    result = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": {"name": "RAGTruth", "response_path": str(response_path), "source_path": str(source_path), "human_span_annotations": True},
        "settings": {
            "model": model, "both_conditions_use_same_model": True, "methods": METHODS,
            "split": split, "quality": quality, "task_types": task_types or ["QA"], "limit": limit,
            "seed": seed, "reasoning_effort": reasoning_effort,
            "max_context_chars": max_context_chars, "max_response_chars": max_response_chars,
            "require_full_evidence": require_full_evidence,
            "source_pipeline": "v043 Source Compiler -> Evidence/Logic/Target parallel -> conditional Assumption/Judge",
            "response_pipeline": "v043 Balanced Complete-Proposition Compiler with local repair -> Evidence/Logic/Target parallel -> conditional Assumption/Judge",
            "cross_graph_pipeline": "Evidence Matcher -> Logic -> conditional Assumption/Judge -> relation-aware Span Projector",
            "all_model_calls_fixed_to_nano": True,
            "system_level_compute_matched": False,
        },
        "sampling": sampling,
        "method_summaries": summaries,
        "paired_six_agent_vs_raw": paired,
        "cache_summary": {
            "generation_cache_path": str(generation_cache_path) if generation_cache_path else None,
            "cache_hits_by_component": dict(cache_hits_by_component),
            "actual_api_calls_this_run": actual_api_calls,
        },
        "summary": {
            "completed_cases": len(rows), "failed_component_calls": len(failures),
            "raw_char_f1_percent": summaries[RAW_METHOD].get("char_f1_percent"),
            "six_agent_char_f1_percent": summaries[GRAPH_METHOD].get("char_f1_percent"),
            "six_agent_delta_mean_case_char_f1_pp": paired.get("mean_case_char_f1_delta_percentage_points"),
            "six_agent_clean_false_positive_rate_percent": summaries[GRAPH_METHOD].get("clean_false_positive_rate_percent"),
            "actual_api_calls_this_run": actual_api_calls,
        },
        "cases": rows,
    }
    (run_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "failures.json").write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "report.html").write_text(_report_html(result), encoding="utf-8")
    (run_dir / "status.json").write_text(json.dumps({
        "status": "completed" if not failures else "completed_with_failures",
        "run_id": run_id, "completed_cases": len(rows), "failed_component_calls": len(failures),
        "actual_api_calls_this_run": actual_api_calls,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return result

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Literal, get_args

from pydantic import BaseModel, Field

from .discussion_graph import (
    DiscussionEdge,
    DiscussionGraphOutput,
    DiscussionIssue,
    DiscussionNode,
    EdgeRelation,
    IssueType,
    NodeRole,
)


# ---------------------------------------------------------------------------
# v037 pilot contract
# ---------------------------------------------------------------------------
PILOT_AGENT_COUNT = 5
PILOT_AGENT_NAMES = ("compiler", "evidence", "logic", "target", "judge")
PILOT_ARCHITECTURE_NAME = "graph_native_5agents_pilot"
PILOT_PROMPT_VERSION = "discussion_graph_native_5agents_pilot_v1"

NODE_ROLE_VOCAB = tuple(get_args(NodeRole))
EDGE_RELATION_VOCAB = tuple(get_args(EdgeRelation))
PUBLIC_ISSUE_VOCAB = tuple(get_args(IssueType))

ValidationStatus = Literal[
    "unreviewed",
    "valid",
    "conditional",
    "insufficient_information",
    "invalid",
    "not_applicable",
    "superseded",
]
SpecialistName = Literal["evidence", "logic", "target"]
ReviewVerdict = Literal["valid", "invalid", "uncertain"]
PatchOperation = Literal[
    "mark_node_invalid",
    "mark_edge_invalid",
    "mark_conditional",
    "request_missing_premise",
    "change_edge_type",
    "qualify_claim",
    "add_candidate_issue",
]

# Internal, architecture-level issue vocabulary. It is deliberately more
# general than the legacy public IssueType enum. Each internal label maps to a
# backward-compatible public label before the result leaves Discussion Lab.
CANONICAL_ISSUE_VOCAB: tuple[str, ...] = (
    "source_mismatch",
    "hallucinated_content",
    "atomicity_violation",
    "wrong_node_role",
    "scope_distortion",
    "modality_distortion",
    "wrong_edge_type",
    "unsupported_inference",
    "missing_premise",
    "unjustified_assumption",
    "circular_reasoning",
    "contradiction",
    "causal_overclaim",
    "scope_expansion",
    "temporal_error",
    "target_mismatch",
    "irrelevant_reasoning",
    "orphan_conclusion",
    "error_propagation",
    "other",
)

CANONICAL_TO_PUBLIC: dict[str, str] = {
    "source_mismatch": "evidence_strength_mismatch",
    "hallucinated_content": "evidence_strength_mismatch",
    "atomicity_violation": "other",
    "wrong_node_role": "other",
    "scope_distortion": "scope_overreach",
    "modality_distortion": "evidence_strength_mismatch",
    "wrong_edge_type": "other",
    "unsupported_inference": "evidence_strength_mismatch",
    "missing_premise": "evidence_strength_mismatch",
    "unjustified_assumption": "evidence_strength_mismatch",
    "circular_reasoning": "semantic_contradiction",
    "contradiction": "semantic_contradiction",
    "causal_overclaim": "causal_overclaim",
    "scope_expansion": "scope_overreach",
    "temporal_error": "temporal_inversion",
    "target_mismatch": "other",
    "irrelevant_reasoning": "other",
    "orphan_conclusion": "evidence_strength_mismatch",
    "error_propagation": "other",
    "other": "other",
}


class PilotCompilerOutput(BaseModel):
    """Compiler output cannot contain correctness judgments or issues."""

    paragraph_summary: str
    nodes: list[DiscussionNode]
    edges: list[DiscussionEdge]


class MissingNodeProposal(BaseModel):
    node_type: Literal["implicit_assumption_candidate"] = "implicit_assumption_candidate"
    text: str
    required_for_edge_id: str = ""
    rationale: str
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)


class PilotGraphPatch(BaseModel):
    operation: PatchOperation
    target_id: str = ""
    node_ids: list[str] = Field(default_factory=list)
    edge_ids: list[str] = Field(default_factory=list)
    proposed_status: ValidationStatus = "conditional"
    proposed_edge_relation: str = ""
    rationale: str
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)


class PilotFinding(BaseModel):
    id: str
    verdict: ReviewVerdict = "invalid"
    canonical_issue_type: str
    public_issue_type: str
    severity: Literal["high", "medium", "low"]
    title: str
    node_ids: list[str] = Field(default_factory=list)
    edge_ids: list[str] = Field(default_factory=list)
    explanation: str
    logical_pattern: str
    suggested_revision: str = ""
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    missing_node_proposals: list[MissingNodeProposal] = Field(default_factory=list)
    patches: list[PilotGraphPatch] = Field(default_factory=list)


class PilotSpecialistReviewOutput(BaseModel):
    specialist: SpecialistName
    reviewed_node_ids: list[str] = Field(default_factory=list)
    reviewed_edge_ids: list[str] = Field(default_factory=list)
    validated_node_ids: list[str] = Field(default_factory=list)
    validated_edge_ids: list[str] = Field(default_factory=list)
    findings: list[PilotFinding] = Field(default_factory=list)
    review_summary: str = ""


class PilotJudgeOutput(BaseModel):
    accepted_finding_ids: list[str] = Field(default_factory=list)
    rejected_finding_ids: list[str] = Field(default_factory=list)
    revised_findings: list[PilotFinding] = Field(default_factory=list)
    rationale: str = ""


@dataclass
class SharedReasoningGraph5:
    source_text: str
    nodes: list[DiscussionNode]
    edges: list[DiscussionEdge]
    source_alignment: dict[str, dict[str, Any]] = field(default_factory=dict)
    annotations: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    reviews: list[PilotSpecialistReviewOutput] = field(default_factory=list)

    def node_ids(self) -> set[str]:
        return {node.id for node in self.nodes}

    def edge_ids(self) -> set[str]:
        return {edge.id for edge in self.edges}

    def edge_by_id(self) -> dict[str, DiscussionEdge]:
        return {edge.id: edge for edge in self.edges}

    def apply_review(self, review: PilotSpecialistReviewOutput) -> None:
        self.reviews.append(review)
        known_nodes = self.node_ids()
        known_edges = self.edge_ids()
        for finding in review.findings:
            for patch in finding.patches:
                targets = list(patch.node_ids) + list(patch.edge_ids)
                if patch.target_id:
                    targets.append(patch.target_id)
                if not targets:
                    targets = ["graph"]
                for target in targets:
                    if target != "graph" and target not in known_nodes and target not in known_edges:
                        continue
                    self.annotations.setdefault(target, []).append({
                        "specialist": review.specialist,
                        "finding_id": finding.id,
                        "operation": patch.operation,
                        "proposed_status": patch.proposed_status,
                        "proposed_edge_relation": patch.proposed_edge_relation,
                        "rationale": patch.rationale,
                        "confidence": patch.confidence,
                    })


def _extract_parsed(response: Any, schema: type[BaseModel]) -> BaseModel:
    refusal_messages: list[str] = []
    for output in getattr(response, "output", []) or []:
        if getattr(output, "type", None) != "message":
            continue
        for item in getattr(output, "content", []) or []:
            item_type = getattr(item, "type", None)
            if item_type == "refusal":
                refusal_messages.append(str(getattr(item, "refusal", "Model refused")))
                continue
            parsed = getattr(item, "parsed", None)
            if parsed is not None:
                return parsed if isinstance(parsed, schema) else schema.model_validate(parsed)
    output_parsed = getattr(response, "output_parsed", None)
    if output_parsed is not None:
        return output_parsed if isinstance(output_parsed, schema) else schema.model_validate(output_parsed)
    output_text = str(getattr(response, "output_text", "") or "").strip()
    if output_text:
        return schema.model_validate_json(output_text)
    if refusal_messages:
        raise ValueError("OpenAI model refusal: " + " | ".join(refusal_messages))
    raise ValueError(f"OpenAI response contained no parsed {schema.__name__}")


def _call_structured_stage(
    *,
    client: Any,
    model: str,
    reasoning_effort: str,
    max_output_tokens: int,
    system: str,
    user: str,
    schema: type[BaseModel],
    retry_once: bool = True,
) -> tuple[BaseModel, list[Any], float]:
    responses: list[Any] = []
    total_latency = 0.0
    attempts = 2 if retry_once else 1
    last_error: Exception | None = None
    for attempt in range(attempts):
        retry_suffix = ""
        budget = max_output_tokens
        if attempt:
            retry_suffix = (
                "\nThe previous structured response could not be parsed. Return compact valid structured output only. "
                "Use only the fixed vocabularies and keep explanations concise."
            )
            budget = min(30000, max(max_output_tokens + 800, int(max_output_tokens * 1.5)))
        started = time.perf_counter()
        try:
            response = client.responses.parse(
                model=model,
                reasoning={"effort": reasoning_effort},
                max_output_tokens=budget,
                store=False,
                input=[
                    {"role": "system", "content": system + retry_suffix},
                    {"role": "user", "content": user},
                ],
                text_format=schema,
            )
            total_latency += (time.perf_counter() - started) * 1000
            responses.append(response)
            return _extract_parsed(response, schema), responses, round(total_latency, 3)
        except Exception as exc:
            total_latency += (time.perf_counter() - started) * 1000
            last_error = exc
    assert last_error is not None
    raise last_error


def _vocab_text(values: tuple[str, ...]) -> str:
    return ", ".join(values)


def build_compiler_prompt(text: str, custom_instruction: str = "") -> tuple[str, str]:
    system = (
        "You are Agent 1 of 5: the Semantic Compiler. Your only job is to compile the supplied text into a candidate graph. "
        "Do not judge correctness, do not emit issues, and do not repair claims. "
        "ATOMICITY CONTRACT: create a separate node when two propositions could independently be true or false; separate observation from interpretation; "
        "separate an operation from its result when needed; preserve negation, modality, population, comparison, time, numbers, units, and conditions. "
        "Do not split a single comparison or one proposition merely because it contains modifiers. "
        "SOURCE FIDELITY: each source_text must be one exact contiguous quotation from the input. "
        f"Allowed node role vocabulary only: {_vocab_text(NODE_ROLE_VOCAB)}. "
        f"Allowed edge relation vocabulary only: {_vocab_text(EDGE_RELATION_VOCAB)}. "
        "Create candidate edges only for relationships the author explicitly states or clearly uses in reaching a claim. "
        "An edge is a candidate dependency, not a validation result. Do not invent missing assumptions. "
        "Use sequential ids d1, d2, ... and e1, e2, ...."
    )
    if custom_instruction.strip():
        system += "\nAdditional extraction instruction: " + custom_instruction.strip()
    user = (
        "Compile the following text into paragraph_summary, atomic nodes, and candidate edges:\n\n"
        f"{text.strip()}\n\n"
        "Return no issues or verdicts."
    )
    return system, user


def _resolve_source_alignment(text: str, nodes: list[DiscussionNode]) -> dict[str, dict[str, Any]]:
    alignments: dict[str, dict[str, Any]] = {}
    for node in nodes:
        quote = str(node.source_text or "")
        start = text.find(quote) if quote else -1
        status = "exact" if start >= 0 else "unmatched"
        end = start + len(quote) if start >= 0 else -1
        matched = quote if start >= 0 else ""
        if start < 0 and quote:
            # Tolerate whitespace-only discrepancies while preserving an
            # explicit warning for the Evidence Agent.
            pattern = re.escape(quote)
            pattern = pattern.replace(r"\ ", r"\s+")
            match = re.search(pattern, text, flags=re.MULTILINE)
            if match:
                start, end = match.span()
                matched = text[start:end]
                status = "whitespace_normalized"
        alignments[node.id] = {
            "status": status,
            "start": start,
            "end": end,
            "matched_text": matched,
        }
    return alignments


def _compact_graph_json(state: SharedReasoningGraph5, *, include_source_alignment: bool = False) -> str:
    payload: dict[str, Any] = {
        "nodes": [
            {
                "id": n.id,
                "sentence_index": n.sentence_index,
                "source_text": n.source_text,
                "plain_meaning": n.plain_meaning,
                "role": n.role,
                "assertion_type": n.assertion_type,
                "polarity": n.polarity,
                "certainty": n.certainty,
                "subject": n.subject,
                "predicate": n.predicate,
                "object": n.object,
                "population_scope": n.population_scope,
                "time_scope": n.time_scope,
                "analysis_population": n.analysis_population,
                "estimand": n.estimand,
                "causal_role": n.causal_role,
                "methodological_role": n.methodological_role,
                "numeric_mentions": n.numeric_mentions,
            }
            for n in state.nodes
        ],
        "edges": [
            {
                "id": e.id,
                "source": e.source,
                "target": e.target,
                "relation": e.relation,
                "rationale": e.rationale,
                "confidence": e.confidence,
            }
            for e in state.edges
        ],
        "annotations": state.annotations,
    }
    if include_source_alignment:
        payload["source_alignment"] = state.source_alignment
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


_SPECIALIST_RULES: dict[str, dict[str, Any]] = {
    "evidence": {
        "canonical": {
            "source_mismatch", "hallucinated_content", "atomicity_violation", "wrong_node_role",
            "scope_distortion", "modality_distortion", "other",
        },
        "public": {
            "evidence_strength_mismatch", "unsupported_generalization", "magnitude_inflation",
            "scope_overreach", "other",
        },
        "instruction": (
            "Check each node against its exact quoted source span and deterministic source alignment. "
            "Assess quotation fidelity, normalized meaning, atomicity, role assignment, population/scope, modality, polarity, numbers, and units. "
            "Do not judge whether an inference edge is logically valid unless the defect is caused by a source mismatch."
        ),
        "include_alignment": True,
    },
    "logic": {
        "canonical": {
            "wrong_edge_type", "unsupported_inference", "missing_premise", "unjustified_assumption",
            "circular_reasoning", "contradiction", "causal_overclaim", "scope_expansion",
            "temporal_error", "error_propagation", "other",
        },
        "public": {
            "direct_contradiction", "semantic_contradiction", "causal_overclaim",
            "temporal_mechanism_conflict", "temporal_inversion", "temporal_scope_extrapolation",
            "necessity_violation", "exclusivity_conflict", "unsupported_generalization",
            "unsupported_mechanism", "scope_overreach", "evidence_strength_mismatch", "other",
        },
        "instruction": (
            "Check candidate edges and small dependency paths. Ask whether the source nodes, if accepted, justify the target node with the stated relation. "
            "Detect wrong relation type, unsupported jumps, contradiction, circularity, causal strengthening, scope transfer, temporal errors, and upstream error propagation. "
            "When an edge requires an unstated premise, do not invent it as fact: emit a missing_node_proposal of type implicit_assumption_candidate."
        ),
        "include_alignment": False,
    },
    "target": {
        "canonical": {
            "target_mismatch", "irrelevant_reasoning", "orphan_conclusion", "scope_expansion", "other",
        },
        "public": {
            "scope_overreach", "unsupported_generalization", "evidence_strength_mismatch", "other",
        },
        "instruction": (
            "Check conclusion and endpoint alignment. Determine whether the graph's final conclusions are connected to the paragraph's stated objective, question, comparison, or claimed endpoint. "
            "Flag orphan conclusions, conclusions that answer a different question, missing answer links, irrelevant branches used as the main answer, or a conclusion whose scope exceeds the stated target. "
            "Do not repeat generic logical or source-fidelity findings already belonging to other agents."
        ),
        "include_alignment": False,
    },
}


def build_specialist_prompt(specialist: SpecialistName, state: SharedReasoningGraph5) -> tuple[str, str]:
    rule = _SPECIALIST_RULES[specialist]
    system = (
        f"You are the {specialist.capitalize()} Agent in a fixed five-agent graph-native pilot. "
        f"{rule['instruction']} "
        "Review the shared graph, not the entire problem from scratch. Every finding must cite existing node_ids and edge_ids when relevant. "
        "Return no finding for a valid element. Use only the allowed vocabularies. "
        f"Allowed canonical_issue_type values: {', '.join(sorted(rule['canonical']))}. "
        f"Allowed public_issue_type values: {', '.join(sorted(rule['public']))}. "
        "A public_issue_type is only a compatibility label; canonical_issue_type should describe the actual defect. "
        "Use patches to localize the proposed validation state."
    )
    user = "Review this shared reasoning graph:\n" + _compact_graph_json(
        state, include_source_alignment=bool(rule["include_alignment"])
    )
    return system, user


def _normalize_finding_targets(finding: PilotFinding, state: SharedReasoningGraph5) -> PilotFinding | None:
    known_nodes = state.node_ids()
    known_edges = state.edge_ids()
    finding.node_ids = list(dict.fromkeys(x for x in finding.node_ids if x in known_nodes))
    finding.edge_ids = list(dict.fromkeys(x for x in finding.edge_ids if x in known_edges))
    if not finding.node_ids and finding.edge_ids:
        edge_map = state.edge_by_id()
        for edge_id in finding.edge_ids:
            edge = edge_map.get(edge_id)
            if edge:
                finding.node_ids.extend([edge.source, edge.target])
        finding.node_ids = list(dict.fromkeys(x for x in finding.node_ids if x in known_nodes))
    if not finding.node_ids and not finding.edge_ids:
        return None
    return finding


def _valid_findings(review: PilotSpecialistReviewOutput, state: SharedReasoningGraph5) -> list[PilotFinding]:
    rule = _SPECIALIST_RULES[review.specialist]
    allowed_canonical = set(rule["canonical"])
    allowed_public = set(rule["public"])
    rows: list[PilotFinding] = []
    for finding in review.findings:
        if finding.verdict == "valid":
            continue
        if finding.canonical_issue_type not in allowed_canonical:
            continue
        expected_public = CANONICAL_TO_PUBLIC.get(finding.canonical_issue_type, "other")
        if finding.public_issue_type not in allowed_public:
            finding.public_issue_type = expected_public if expected_public in allowed_public else "other"
        if finding.public_issue_type not in PUBLIC_ISSUE_VOCAB:
            finding.public_issue_type = "other"
        normalized = _normalize_finding_targets(finding, state)
        if normalized is None:
            continue
        valid_missing: list[MissingNodeProposal] = []
        for proposal in normalized.missing_node_proposals:
            if proposal.required_for_edge_id and proposal.required_for_edge_id not in state.edge_ids():
                continue
            valid_missing.append(proposal)
        normalized.missing_node_proposals = valid_missing
        rows.append(normalized)
    return rows


def _finding_target_key(finding: PilotFinding) -> tuple[str, ...]:
    ids = finding.edge_ids or finding.node_ids
    return tuple(sorted(ids))


def _finding_signature(finding: PilotFinding) -> tuple[str, tuple[str, ...]]:
    return finding.canonical_issue_type, _finding_target_key(finding)


def _deduplicate_findings(findings: list[PilotFinding]) -> list[PilotFinding]:
    best: dict[tuple[str, tuple[str, ...]], PilotFinding] = {}
    for finding in findings:
        key = _finding_signature(finding)
        current = best.get(key)
        if current is None or finding.confidence > current.confidence:
            best[key] = finding
    return list(best.values())


def _needs_judge(findings: list[PilotFinding]) -> bool:
    if any(f.verdict == "uncertain" or f.confidence < 0.7 for f in findings):
        return True
    by_target: dict[tuple[str, ...], set[str]] = {}
    for finding in findings:
        target = _finding_target_key(finding)
        if not target:
            continue
        by_target.setdefault(target, set()).add(finding.canonical_issue_type)
    # Different defect labels on the exact same graph target are adjudicated;
    # findings on different targets are not treated as disagreement.
    return any(len(labels) > 1 for labels in by_target.values())


def build_judge_prompt(state: SharedReasoningGraph5, findings: list[PilotFinding]) -> tuple[str, str]:
    system = (
        "You are Agent 5 of 5: the conditional Judge. You do not re-read or solve the original text from scratch. "
        "Adjudicate only uncertain or conflicting candidate findings about the same graph target. "
        "Accept supported findings, reject speculative or duplicate ones, or revise them to a narrower target or better label. "
        "Do not add unrelated findings. Preserve the fixed canonical and public vocabularies. "
        f"Canonical vocabulary: {', '.join(CANONICAL_ISSUE_VOCAB)}. "
        f"Public vocabulary: {', '.join(PUBLIC_ISSUE_VOCAB)}."
    )
    payload = {
        "graph": json.loads(_compact_graph_json(state, include_source_alignment=True)),
        "candidate_findings": [finding.model_dump() for finding in findings],
    }
    return system, "Adjudicate these graph-local findings:\n" + json.dumps(
        payload, ensure_ascii=False, separators=(",", ":")
    )


def _to_public_issue(finding: PilotFinding, index: int) -> DiscussionIssue:
    issue_type = finding.public_issue_type
    if issue_type not in PUBLIC_ISSUE_VOCAB:
        issue_type = CANONICAL_TO_PUBLIC.get(finding.canonical_issue_type, "other")
    if issue_type not in PUBLIC_ISSUE_VOCAB:
        issue_type = "other"
    return DiscussionIssue(
        id=f"agent_i{index}",
        issue_type=issue_type,
        severity=finding.severity,
        title=finding.title,
        node_ids=finding.node_ids,
        explanation=finding.explanation,
        logical_pattern=finding.logical_pattern,
        suggested_revision=finding.suggested_revision,
        confidence=finding.confidence,
    )


def run_graph_native_5agents_chunk(
    text: str,
    *,
    model: str,
    reasoning_effort: str,
    max_output_tokens: int,
    custom_instruction: str,
    client: Any,
    judge_enabled: bool = True,
) -> tuple[DiscussionGraphOutput, list[Any], float, dict[str, Any]]:
    """Compiler -> Evidence -> Logic -> Target -> conditional Judge.

    The public DiscussionGraphOutput is unchanged. The source alignments,
    validation proposals, missing-premise candidates, and judge trace remain
    internal so existing UI, endpoints, result JSON consumers, and evaluators
    continue to work.
    """

    all_responses: list[Any] = []
    total_latency = 0.0
    private_trace: dict[str, Any] = {
        "architecture": PILOT_ARCHITECTURE_NAME,
        "agent_count": PILOT_AGENT_COUNT,
        "agents": list(PILOT_AGENT_NAMES),
        "stages": [],
        "vocabulary": {
            "node_roles": list(NODE_ROLE_VOCAB),
            "edge_relations": list(EDGE_RELATION_VOCAB),
            "validation_statuses": list(get_args(ValidationStatus)),
            "canonical_issue_types": list(CANONICAL_ISSUE_VOCAB),
        },
    }

    compiler_system, compiler_user = build_compiler_prompt(text, custom_instruction)
    compiled_raw, responses, latency = _call_structured_stage(
        client=client,
        model=model,
        reasoning_effort=reasoning_effort,
        max_output_tokens=max_output_tokens,
        system=compiler_system,
        user=compiler_user,
        schema=PilotCompilerOutput,
    )
    assert isinstance(compiled_raw, PilotCompilerOutput)
    all_responses.extend(responses)
    total_latency += latency
    private_trace["stages"].append({"stage": "agent:compiler", "status": "ok", "api_calls": len(responses)})

    state = SharedReasoningGraph5(
        source_text=text,
        nodes=compiled_raw.nodes,
        edges=compiled_raw.edges,
        source_alignment=_resolve_source_alignment(text, compiled_raw.nodes),
    )

    all_findings: list[PilotFinding] = []
    for specialist in ("evidence", "logic", "target"):
        system, user = build_specialist_prompt(specialist, state)
        try:
            review_raw, responses, latency = _call_structured_stage(
                client=client,
                model=model,
                reasoning_effort=reasoning_effort,
                max_output_tokens=max(1600, min(5000, max_output_tokens)),
                system=system,
                user=user,
                schema=PilotSpecialistReviewOutput,
            )
            assert isinstance(review_raw, PilotSpecialistReviewOutput)
            review_raw.specialist = specialist
            review_raw.findings = _valid_findings(review_raw, state)
            for finding_index, finding in enumerate(review_raw.findings, 1):
                finding.id = f"{specialist}_{finding.id or finding_index}"
            state.apply_review(review_raw)
            all_findings.extend(review_raw.findings)
            all_responses.extend(responses)
            total_latency += latency
            private_trace["stages"].append({
                "stage": f"agent:{specialist}",
                "status": "ok",
                "api_calls": len(responses),
                "finding_count": len(review_raw.findings),
            })
        except Exception as exc:
            private_trace["stages"].append({
                "stage": f"agent:{specialist}",
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            })

    all_findings = _deduplicate_findings(all_findings)
    if judge_enabled and all_findings and _needs_judge(all_findings):
        system, user = build_judge_prompt(state, all_findings)
        try:
            judged_raw, responses, latency = _call_structured_stage(
                client=client,
                model=model,
                reasoning_effort=reasoning_effort,
                max_output_tokens=max(1800, min(5500, max_output_tokens)),
                system=system,
                user=user,
                schema=PilotJudgeOutput,
            )
            assert isinstance(judged_raw, PilotJudgeOutput)
            all_responses.extend(responses)
            total_latency += latency
            by_id = {finding.id: finding for finding in all_findings}
            selected = [by_id[x] for x in judged_raw.accepted_finding_ids if x in by_id]
            selected.extend(
                finding for finding in judged_raw.revised_findings
                if _normalize_finding_targets(finding, state) is not None
            )
            rejected = set(judged_raw.rejected_finding_ids)
            if not judged_raw.accepted_finding_ids and not judged_raw.revised_findings:
                selected = [finding for finding in all_findings if finding.id not in rejected]
            all_findings = _deduplicate_findings(selected)
            private_trace["stages"].append({
                "stage": "agent:judge",
                "status": "ok",
                "api_calls": len(responses),
            })
        except Exception as exc:
            private_trace["stages"].append({
                "stage": "agent:judge",
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            })
    else:
        private_trace["stages"].append({
            "stage": "agent:judge",
            "status": "not_needed" if judge_enabled else "disabled",
            "api_calls": 0,
        })

    public_issues: list[DiscussionIssue] = []
    for index, finding in enumerate(all_findings, 1):
        try:
            public_issues.append(_to_public_issue(finding, index))
        except Exception:
            continue

    private_trace["source_alignment"] = state.source_alignment
    private_trace["specialists_completed"] = [review.specialist for review in state.reviews]
    private_trace["annotation_target_count"] = len(state.annotations)
    private_trace["missing_premise_candidate_count"] = sum(
        len(finding.missing_node_proposals) for finding in all_findings
    )
    private_trace["final_agent_finding_count"] = len(public_issues)

    output = DiscussionGraphOutput(
        paragraph_summary=compiled_raw.paragraph_summary,
        nodes=compiled_raw.nodes,
        edges=compiled_raw.edges,
        issues=public_issues,
        overall_assessment="potential_issue" if public_issues else "internally_consistent",
    )
    return output, all_responses, round(total_latency, 3), private_trace

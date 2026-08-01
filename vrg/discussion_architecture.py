from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field

from .discussion_graph import (
    DiscussionEdge,
    DiscussionGraphOutput,
    DiscussionIssue,
    DiscussionNode,
)


SpecialistName = Literal["logic", "evidence", "methodology", "quantitative"]
PatchOperation = Literal[
    "mark_node_risk",
    "mark_edge_risk",
    "request_missing_premise",
    "qualify_claim",
    "add_candidate_issue",
]
ReviewVerdict = Literal["valid", "invalid", "uncertain"]


class DiscussionCompilerOutput(BaseModel):
    """Internal compiler output. It intentionally cannot emit issues."""

    paragraph_summary: str
    nodes: list[DiscussionNode]
    edges: list[DiscussionEdge]


class GraphPatch(BaseModel):
    operation: PatchOperation
    target_id: str = ""
    node_ids: list[str] = Field(default_factory=list)
    edge_ids: list[str] = Field(default_factory=list)
    rationale: str
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)


class SpecialistFinding(BaseModel):
    id: str
    verdict: ReviewVerdict = "invalid"
    issue_type: str
    severity: Literal["high", "medium", "low"]
    title: str
    node_ids: list[str]
    edge_ids: list[str] = Field(default_factory=list)
    explanation: str
    logical_pattern: str
    suggested_revision: str = ""
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    patches: list[GraphPatch] = Field(default_factory=list)


class SpecialistReviewOutput(BaseModel):
    specialist: SpecialistName
    reviewed_node_ids: list[str] = Field(default_factory=list)
    findings: list[SpecialistFinding] = Field(default_factory=list)
    review_summary: str = ""


class JudgeOutput(BaseModel):
    accepted_finding_ids: list[str] = Field(default_factory=list)
    rejected_finding_ids: list[str] = Field(default_factory=list)
    revised_findings: list[SpecialistFinding] = Field(default_factory=list)
    rationale: str = ""


@dataclass
class SharedReasoningGraph:
    nodes: list[DiscussionNode]
    edges: list[DiscussionEdge]
    annotations: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    reviews: list[SpecialistReviewOutput] = field(default_factory=list)

    def node_ids(self) -> set[str]:
        return {node.id for node in self.nodes}

    def edge_ids(self) -> set[str]:
        return {edge.id for edge in self.edges}

    def apply_review(self, review: SpecialistReviewOutput) -> None:
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
                "\nYour previous response could not be parsed. Return only compact valid structured output. "
                "Do not repeat source passages and keep explanations concise."
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
        except Exception as exc:  # parsing failures must not crash unrelated agents
            total_latency += (time.perf_counter() - started) * 1000
            last_error = exc
    assert last_error is not None
    raise last_error


def build_compiler_prompt(text: str, custom_instruction: str = "") -> tuple[str, str]:
    system = (
        "You are the Semantic Compiler in a graph-native verification system. "
        "Compile the supplied Discussion-style text into a typed shared reasoning graph. "
        "Do not decide whether the paragraph is correct, do not emit issues, and do not repair it. "
        "Extract atomic public claims, observations, evidence, methods, limitations, assumptions that are explicitly stated, and conclusions. "
        "SOURCE FIDELITY IS MANDATORY: every source_text must be an exact contiguous quote. "
        "Preserve numbers, populations, time points, uncertainty language, analysis names, and study-design details. "
        "Edges must describe only relationships asserted or clearly used by the paragraph. "
        "Do not invent support edges merely because two claims concern the same topic. "
        "Use sequential ids d1, d2, ... and e1, e2, .... "
        "This stage is compilation only; downstream agents perform validation."
    )
    if custom_instruction.strip():
        system += "\nAdditional extraction instruction: " + custom_instruction.strip()
    user = (
        "Compile this text into the shared reasoning graph:\n\n"
        f"{text.strip()}\n\n"
        "Return only paragraph_summary, nodes, and edges."
    )
    return system, user


_SPECIALIST_RULES: dict[str, dict[str, Any]] = {
    "logic": {
        "issues": [
            "direct_contradiction", "semantic_contradiction", "causal_overclaim",
            "temporal_mechanism_conflict", "temporal_inversion", "temporal_scope_extrapolation",
            "necessity_violation", "exclusivity_conflict", "unsupported_generalization",
            "unsupported_mechanism", "scope_overreach", "other",
        ],
        "instruction": (
            "Review logical entailment, contradiction, causal strength, temporal order, necessity, sufficiency, exclusivity, "
            "scope transfer, missing premises, and whether conclusions follow from parent nodes."
        ),
    },
    "evidence": {
        "issues": [
            "evidence_strength_mismatch", "unsupported_generalization", "unsupported_mechanism",
            "surrogate_to_clinical_overreach", "magnitude_inflation", "selective_outcome_reporting", "other",
        ],
        "instruction": (
            "Review whether evidence nodes actually support connected claims, whether uncertainty is preserved, "
            "and whether the conclusion is stronger or broader than the graph evidence."
        ),
    },
    "methodology": {
        "issues": [
            "design_claim_mismatch", "attrition_bias", "informative_missingness", "landmark_selection_bias",
            "time_zero_mismatch", "post_treatment_adjustment", "estimand_mismatch", "collider_bias_risk",
            "competing_risk_misclassification", "reproducibility_conflict", "other",
        ],
        "instruction": (
            "Review study design, eligibility and selection, time zero, attrition, conditioning, adjustment variables, "
            "estimands, competing events, reproducibility, and whether the claimed interpretation matches the design."
        ),
    },
    "quantitative": {
        "issues": [
            "magnitude_inflation", "subgroup_significance_fallacy", "unsupported_effect_heterogeneity",
            "noninferiority_interpretation_error", "equivalence_fallacy", "multiplicity_risk",
            "selective_outcome_reporting", "evidence_strength_mismatch", "other",
        ],
        "instruction": (
            "Review numerical magnitude, statistical significance, subgroup interactions, multiplicity, "
            "noninferiority/equivalence language, and whether quantitative conclusions follow from the graph."
        ),
    },
}


def _compact_graph_json(state: SharedReasoningGraph) -> str:
    payload = {
        "nodes": [
            {
                "id": n.id,
                "source_text": n.source_text,
                "plain_meaning": n.plain_meaning,
                "role": n.role,
                "assertion_type": n.assertion_type,
                "polarity": n.polarity,
                "certainty": n.certainty,
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
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def build_specialist_prompt(specialist: SpecialistName, state: SharedReasoningGraph) -> tuple[str, str]:
    rule = _SPECIALIST_RULES[specialist]
    allowed = ", ".join(rule["issues"])
    system = (
        f"You are the {specialist} specialist in a graph-native multi-agent verification system. "
        "You receive only a compiled shared reasoning graph, not the original document. "
        f"{rule['instruction']} "
        "Do not reconstruct or rewrite the entire paragraph. Review only node-edge validity. "
        "A weak association is not automatically a contradiction. Do not flag a risk that the paragraph already acknowledges and resolves. "
        "Every finding must cite existing node ids; cite edge ids when an edge is the defect. "
        "Use patches to mark the exact node or edge requiring review. "
        "Return no finding when the graph supports the claim. "
        f"Allowed issue_type values for this specialist: {allowed}."
    )
    user = "Review this shared reasoning graph:\n" + _compact_graph_json(state)
    return system, user


def _route_specialists(state: SharedReasoningGraph) -> list[SpecialistName]:
    # Logic and evidence are the general-purpose independent reviewers.
    selected: list[SpecialistName] = ["logic", "evidence"]
    searchable = " ".join(
        " ".join([
            n.source_text,
            n.plain_meaning,
            n.assertion_type,
            n.methodological_role,
            n.causal_role,
            " ".join(n.numeric_mentions),
        ])
        for n in state.nodes
    ).lower()
    methodological_markers = (
        "study_design", "analysis_method", "eligibility", "selection", "estimand", "competing_event",
        "adjustment_variable", "time zero", "landmark", "dropout", "attrition", "cohort", "trial",
    )
    if any(marker in searchable for marker in methodological_markers):
        selected.append("methodology")
    quantitative_markers = (
        "%", "hazard ratio", "confidence interval", "p=", "p <", "significant", "interaction",
        "noninferior", "equivalent", "multiplicity", "subgroup", "odds ratio", "risk ratio",
    )
    if any(marker in searchable for marker in quantitative_markers) or any(n.numeric_mentions for n in state.nodes):
        selected.append("quantitative")
    return selected


def _valid_findings(review: SpecialistReviewOutput, state: SharedReasoningGraph) -> list[SpecialistFinding]:
    known_nodes = state.node_ids()
    known_edges = state.edge_ids()
    allowed = set(_SPECIALIST_RULES[review.specialist]["issues"])
    rows: list[SpecialistFinding] = []
    for finding in review.findings:
        if finding.issue_type not in allowed:
            continue
        finding.node_ids = list(dict.fromkeys(x for x in finding.node_ids if x in known_nodes))
        finding.edge_ids = list(dict.fromkeys(x for x in finding.edge_ids if x in known_edges))
        if not finding.node_ids:
            continue
        if finding.verdict == "valid":
            continue
        rows.append(finding)
    return rows


def _finding_signature(finding: SpecialistFinding) -> tuple[str, tuple[str, ...]]:
    return finding.issue_type, tuple(sorted(finding.node_ids))


def _deduplicate_findings(findings: list[SpecialistFinding]) -> list[SpecialistFinding]:
    best: dict[tuple[str, tuple[str, ...]], SpecialistFinding] = {}
    for finding in findings:
        key = _finding_signature(finding)
        current = best.get(key)
        if current is None or finding.confidence > current.confidence:
            best[key] = finding
    return list(best.values())


def _needs_judge(findings: list[SpecialistFinding]) -> bool:
    if any(f.verdict == "uncertain" or f.confidence < 0.7 for f in findings):
        return True
    for i, left in enumerate(findings):
        left_nodes = set(left.node_ids)
        for right in findings[i + 1 :]:
            if not left_nodes.intersection(right.node_ids):
                continue
            if left.issue_type != right.issue_type:
                return True
    return False


def build_judge_prompt(state: SharedReasoningGraph, findings: list[SpecialistFinding]) -> tuple[str, str]:
    system = (
        "You are the conditional Judge in a graph-native multi-agent verification system. "
        "Resolve only disagreements or uncertain specialist findings. You receive the shared graph and candidate findings, not the original paragraph. "
        "Accept a finding only when the cited graph nodes and edges support it. Reject duplicate, speculative, already-resolved, or over-broad findings. "
        "You may revise a finding to a better-supported issue type or narrower node set. "
        "Do not add unrelated findings."
    )
    user_payload = {
        "graph": json.loads(_compact_graph_json(state)),
        "candidate_findings": [f.model_dump() for f in findings],
    }
    return system, "Adjudicate:\n" + json.dumps(user_payload, ensure_ascii=False, separators=(",", ":"))


def _to_public_issue(finding: SpecialistFinding, index: int) -> DiscussionIssue:
    return DiscussionIssue(
        id=f"agent_i{index}",
        issue_type=finding.issue_type,  # validated downstream against the public schema
        severity=finding.severity,
        title=finding.title,
        node_ids=finding.node_ids,
        explanation=finding.explanation,
        logical_pattern=finding.logical_pattern,
        suggested_revision=finding.suggested_revision,
        confidence=finding.confidence,
    )


def run_graph_native_chunk(
    text: str,
    *,
    model: str,
    reasoning_effort: str,
    max_output_tokens: int,
    custom_instruction: str,
    client: Any,
    judge_enabled: bool = True,
) -> tuple[DiscussionGraphOutput, list[Any], float, dict[str, Any]]:
    """Run compiler -> shared graph -> specialists -> conditional judge.

    The returned DiscussionGraphOutput is deliberately identical to the legacy
    public schema. Internal graph state and agent messages are kept in the
    private trace returned as the fourth element.
    """

    all_responses: list[Any] = []
    latency_total = 0.0
    private_trace: dict[str, Any] = {"architecture": "graph_native_multi_agent", "stages": []}

    compiler_system, compiler_user = build_compiler_prompt(text, custom_instruction)
    compiled, responses, latency = _call_structured_stage(
        client=client,
        model=model,
        reasoning_effort=reasoning_effort,
        max_output_tokens=max_output_tokens,
        system=compiler_system,
        user=compiler_user,
        schema=DiscussionCompilerOutput,
    )
    assert isinstance(compiled, DiscussionCompilerOutput)
    all_responses.extend(responses)
    latency_total += latency
    private_trace["stages"].append({"stage": "compiler", "status": "ok", "api_calls": len(responses)})

    state = SharedReasoningGraph(nodes=compiled.nodes, edges=compiled.edges)
    all_findings: list[SpecialistFinding] = []
    for specialist in _route_specialists(state):
        system, user = build_specialist_prompt(specialist, state)
        try:
            review_raw, responses, latency = _call_structured_stage(
                client=client,
                model=model,
                reasoning_effort=reasoning_effort,
                max_output_tokens=max(1400, min(4500, max_output_tokens)),
                system=system,
                user=user,
                schema=SpecialistReviewOutput,
            )
            assert isinstance(review_raw, SpecialistReviewOutput)
            # Enforce role identity even if the model emits another label.
            review_raw.specialist = specialist
            review_raw.findings = _valid_findings(review_raw, state)
            for finding_index, finding in enumerate(review_raw.findings, 1):
                finding.id = f"{specialist}_{finding.id or finding_index}"
            state.apply_review(review_raw)
            all_findings.extend(review_raw.findings)
            all_responses.extend(responses)
            latency_total += latency
            private_trace["stages"].append({
                "stage": f"specialist:{specialist}",
                "status": "ok",
                "api_calls": len(responses),
                "finding_count": len(review_raw.findings),
            })
        except Exception as exc:
            private_trace["stages"].append({
                "stage": f"specialist:{specialist}",
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
                max_output_tokens=max(1600, min(5000, max_output_tokens)),
                system=system,
                user=user,
                schema=JudgeOutput,
            )
            assert isinstance(judged_raw, JudgeOutput)
            all_responses.extend(responses)
            latency_total += latency
            by_id = {f.id: f for f in all_findings}
            selected = [by_id[x] for x in judged_raw.accepted_finding_ids if x in by_id]
            selected.extend(judged_raw.revised_findings)
            rejected = set(judged_raw.rejected_finding_ids)
            if not judged_raw.accepted_finding_ids and not judged_raw.revised_findings:
                selected = [f for f in all_findings if f.id not in rejected]
            all_findings = _deduplicate_findings(selected)
            private_trace["stages"].append({"stage": "judge", "status": "ok", "api_calls": len(responses)})
        except Exception as exc:
            private_trace["stages"].append({
                "stage": "judge",
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            })

    public_issues: list[DiscussionIssue] = []
    for index, finding in enumerate(all_findings, 1):
        try:
            public_issues.append(_to_public_issue(finding, index))
        except Exception:
            # Invalid issue labels from a specialist are safely discarded; the
            # deterministic postprocessor can still detect structural patterns.
            continue

    output = DiscussionGraphOutput(
        paragraph_summary=compiled.paragraph_summary,
        nodes=compiled.nodes,
        edges=compiled.edges,
        issues=public_issues,
        overall_assessment="potential_issue" if public_issues else "internally_consistent",
    )
    private_trace["specialists"] = [review.specialist for review in state.reviews]
    private_trace["annotation_target_count"] = len(state.annotations)
    private_trace["final_agent_finding_count"] = len(public_issues)
    return output, all_responses, round(latency_total, 3), private_trace

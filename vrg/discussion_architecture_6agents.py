from __future__ import annotations

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher
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
# v045 six-agent Discussion Hub with v043-derived Balanced Compiler
# ---------------------------------------------------------------------------
PILOT_AGENT_COUNT = 6
PILOT_AGENT_NAMES = ("compiler", "evidence", "logic", "assumption", "target", "judge")
PILOT_ARCHITECTURE_NAME = "graph_native_6agents_balanced_compiler"
PILOT_PROMPT_VERSION = "discussion_graph_native_6agents_balanced_compiler_v045"
BALANCED_COMPILER_VERSION = "v043-derived-balanced-complete-proposition-local-repair"

# The model is prompted with a lower configurable cap, while these absolute
# schema caps prevent pathological structured outputs from exhausting memory.
ABSOLUTE_COMPILER_NODE_HARD_CAP = 64
ABSOLUTE_COMPILER_EDGE_HARD_CAP = 160
DEFAULT_COMPILER_NODE_CAP = 32
DEFAULT_COMPILER_EDGE_CAP = 72
DEFAULT_DOCUMENT_NODE_CAP = 120
DEFAULT_DOCUMENT_EDGE_CAP = 260
DEFAULT_NODE_DEDUP_THRESHOLD = 0.96

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


class CompilerNodePriority(BaseModel):
    node_id: str
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    reason: str = ""


class PilotCompilerOutput(BaseModel):
    """Compiler output cannot contain correctness judgments or issues."""

    paragraph_summary: str
    nodes: list[DiscussionNode] = Field(max_length=ABSOLUTE_COMPILER_NODE_HARD_CAP)
    edges: list[DiscussionEdge] = Field(max_length=ABSOLUTE_COMPILER_EDGE_HARD_CAP)
    node_priorities: list[CompilerNodePriority] = Field(default_factory=list)
    omitted_content_summary: str = ""


class DiscussionNodeReplacement(BaseModel):
    original_node_id: str
    replacement_nodes: list[DiscussionNode] = Field(default_factory=list, max_length=8)


class DiscussionCompilerRepairOutput(BaseModel):
    """Local repair patch for the Balanced Discussion Compiler.

    It never recompiles the whole graph. Valid nodes remain untouched.
    """

    replacements: list[DiscussionNodeReplacement] = Field(default_factory=list, max_length=30)
    added_nodes: list[DiscussionNode] = Field(default_factory=list, max_length=30)
    drop_node_ids: list[str] = Field(default_factory=list, max_length=30)
    note: str = ""


class MissingNodeProposal(BaseModel):
    node_type: Literal["implicit_assumption_candidate"] = "implicit_assumption_candidate"
    text: str
    required_for_edge_id: str = ""
    rationale: str
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)


AssumptionDisposition = Literal[
    "accepted_definition",
    "accepted_background",
    "supported_by_context",
    "explicitly_supported",
    "plausible_but_unverified",
    "unsupported_critical_assumption",
    "overly_strong_assumption",
    "circular_assumption",
    "irrelevant_assumption",
]
AssumptionNecessity = Literal["required", "helpful", "not_required", "uncertain"]

ASSUMPTION_DISPOSITION_VOCAB = tuple(get_args(AssumptionDisposition))


class AssumptionCandidate(BaseModel):
    proposal_id: str
    logic_finding_id: str
    text: str
    required_for_edge_id: str = ""
    rationale: str = ""
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)


class AssumptionAssessment(BaseModel):
    proposal_id: str
    logic_finding_id: str
    required_for_edge_id: str = ""
    disposition: AssumptionDisposition
    necessity: AssumptionNecessity = "required"
    proposed_edge_status: ValidationStatus
    supporting_node_ids: list[str] = Field(default_factory=list)
    rationale: str
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)


class PilotAssumptionReviewOutput(BaseModel):
    assessments: list[AssumptionAssessment] = Field(default_factory=list)
    review_summary: str = ""


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
class SharedReasoningGraph6:
    source_text: str
    nodes: list[DiscussionNode]
    edges: list[DiscussionEdge]
    source_alignment: dict[str, dict[str, Any]] = field(default_factory=dict)
    annotations: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    reviews: list[PilotSpecialistReviewOutput] = field(default_factory=list)
    assumption_assessments: list[AssumptionAssessment] = field(default_factory=list)

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


def build_compiler_prompt(
    text: str,
    custom_instruction: str = "",
    *,
    node_cap: int = DEFAULT_COMPILER_NODE_CAP,
    edge_cap: int = DEFAULT_COMPILER_EDGE_CAP,
) -> tuple[str, str]:
    system = (
        "You are Agent 1 of 6: the Balanced Semantic Compiler. Your only job is to compile the supplied Discussion text "
        "into a candidate reasoning graph. Do not judge correctness, do not emit issues, and do not repair the author's claims. "
        "BALANCED NODE CONTRACT: a sentence is a container, not automatically a node, but a node is not the shortest lexical "
        "fragment. Each node must express the smallest semantically complete proposition or clause that can receive one factual, "
        "logical, or methodological verdict. Split only when parts could genuinely receive different verdicts. Keep together the "
        "subject, predicate, object, negation, number and unit, comparator, population, condition, time, modality, quantifier, and "
        "causal strength needed for one complete proposition. For a shared-subject clause, source_text may omit the repeated subject "
        "only when the quoted clause still contains a predicate and necessary complement; plain_meaning must restore the complete "
        "proposition without adding new factual content. Do not emit isolated entities, numbers, adjectives, headings, discourse "
        "markers, transitions, or rhetorical fragments as nodes. Do not split merely because a sentence contains 'and', 'but', a "
        "relative clause, or several nouns; preserve one coherent combined proposition when its parts cannot sensibly receive "
        "different verdicts. Separate observation from interpretation and separate an operation from its claimed result when they "
        "can be evaluated independently. "
        "SOURCE FIDELITY: every source_text must be one exact contiguous quotation from the input. plain_meaning is the complete "
        "canonical proposition; inferred_details may contain only explicitly marked interpretation and must never silently alter "
        "polarity, scope, modality, population, time, numbers, units, comparison direction, or causal strength. "
        "TASK-COMPLETE WITHIN BUDGET: be compact through deduplication, not careless omission. Preserve the main objective, final "
        "conclusions, every distinct claim directly feeding a conclusion, strongest evidence, material counterevidence and "
        "limitations, and critical study-design, population, exposure, analysis, estimand, and temporal conditions. Merge true "
        "repetitions and close restatements; never create two nodes for the same proposition, and never merge propositions "
        "differing in entity, polarity, number, unit, population, "
        "time, modality, comparison, condition, or causal strength. "
        f"NODE BUDGET: return at most {node_cap} nodes and at most {edge_cap} edges. When the text exceeds the budget, preserve "
        "conclusion paths and material qualifiers before peripheral background; summarize intentionally omitted secondary material "
        "in omitted_content_summary. Provide node_priorities for retained nodes. "
        f"Allowed node role vocabulary only: {_vocab_text(NODE_ROLE_VOCAB)}. "
        f"Allowed edge relation vocabulary only: {_vocab_text(EDGE_RELATION_VOCAB)}. "
        "Create candidate edges only for relationships explicitly stated or clearly used by the author. An edge is a candidate "
        "dependency, not a validation result. Do not invent missing assumptions. Before returning, silently verify that every node "
        "is an exact quote, expresses one complete proposition, is not a fragment, and is not duplicated. Use sequential ids d1, "
        "d2, ... and e1, e2, ...."
    )
    if custom_instruction.strip():
        system += "\nAdditional extraction instruction: " + custom_instruction.strip()
    user = (
        "Compile the following Discussion text using the Balanced Complete-Proposition contract. Return paragraph_summary, "
        "task-complete non-duplicative nodes, candidate edges, node priorities, and an omitted-content summary when needed:\n\n"
        f"{text.strip()}\n\n"
        "Return no correctness verdicts or issues."
    )
    return system, user


_BALANCED_VERB_CUE_RE = re.compile(
    r"\b(?:is|are|was|were|be|been|being|has|have|had|does|do|did|can|could|may|might|must|"
    r"will|would|shall|should|shows?|showed|finds?|found|reports?|reported|suggests?|suggested|"
    r"indicates?|indicated|demonstrates?|demonstrated|includes?|included|enrolls?|enrolled|"
    r"reduces?|reduced|increases?|increased|improves?|improved|causes?|caused|predicts?|predicted|"
    r"associates?|associated|differs?|differed|occurs?|occurred|remains?|remained|allows?|allowed|"
    r"uses?|used|leads?|led|becomes?|became|supports?|supported|limits?|limited|confirms?|confirmed|"
    r"\w+(?:ed|ing))\b",
    re.IGNORECASE,
)
_BALANCED_COMPOUND_BOUNDARY_RE = re.compile(
    r"\s*(?:;|\b(?:and|but|while|whereas)\b|,\s*(?:which|who|whereas|while)\b)\s*",
    re.IGNORECASE,
)
_BALANCED_DISCOURSE_MARKERS = {
    "however", "finally", "therefore", "thus", "moreover", "furthermore", "overall",
    "in conclusion", "to conclude", "in summary", "for example", "for instance",
    "first", "second", "third", "lastly", "additionally", "meanwhile",
    "하지만", "그러나", "따라서", "결론적으로", "마지막으로", "예를 들어", "요약하면",
}


def _balanced_words(value: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9가-힣]+(?:['’-][A-Za-z0-9가-힣]+)?", str(value or ""))


def _balanced_is_discourse_only(value: str) -> bool:
    normalized = " ".join(_balanced_words(value)).lower().strip()
    return bool(normalized) and normalized in _BALANCED_DISCOURSE_MARKERS


def _balanced_looks_multi_claim(value: str) -> bool:
    text = " ".join(str(value or "").split())
    if not text or not _BALANCED_COMPOUND_BOUNDARY_RE.search(text):
        return False
    parts = [x.strip(" ,;") for x in _BALANCED_COMPOUND_BOUNDARY_RE.split(text) if x.strip(" ,;")]
    verb_parts = sum(bool(_BALANCED_VERB_CUE_RE.search(x)) for x in parts)
    return verb_parts >= 2


def _balanced_node_reasons(node: DiscussionNode, text: str) -> list[str]:
    reasons: list[str] = []
    quote = str(node.source_text or "").strip()
    meaning = str(node.plain_meaning or quote).strip()
    if not quote:
        reasons.append("empty_source_text")
    elif quote not in text:
        pattern = re.escape(quote).replace(r"\ ", r"\s+")
        if not re.search(pattern, text, flags=re.MULTILINE):
            reasons.append("unresolved_exact_quote")
    if _balanced_is_discourse_only(quote):
        reasons.append("discourse_fragment")
    words = _balanced_words(quote)
    meaning_has_predicate = bool(str(node.predicate or "").strip()) or bool(_BALANCED_VERB_CUE_RE.search(meaning))
    quote_has_predicate = bool(str(node.predicate or "").strip()) or bool(_BALANCED_VERB_CUE_RE.search(quote))
    # Be conservative across languages: only obvious short fragments trigger repair.
    if len(words) <= 1 or (len(words) <= 4 and not meaning_has_predicate and not quote_has_predicate):
        reasons.append("incomplete_fragment")
    if _balanced_looks_multi_claim(quote):
        reasons.append("likely_multiple_independent_claims")
    return list(dict.fromkeys(reasons))


def _balanced_compiler_diagnostics(compiled: PilotCompilerOutput, text: str) -> dict[str, Any]:
    seen: set[str] = set()
    duplicate_ids: list[str] = []
    node_reasons: dict[str, list[str]] = {}
    for node in compiled.nodes:
        if node.id in seen:
            duplicate_ids.append(node.id)
        seen.add(node.id)
        reasons = _balanced_node_reasons(node, text)
        if reasons:
            node_reasons[node.id] = reasons
    repair_ids = sorted(set(node_reasons) | set(duplicate_ids))
    return {
        "compiler_version": BALANCED_COMPILER_VERSION,
        "claim_count": len(compiled.nodes),
        "edge_count": len(compiled.edges),
        "problem_node_ids": repair_ids,
        "node_reasons": node_reasons,
        "duplicate_node_ids": duplicate_ids,
        "needs_refinement": bool(repair_ids),
    }


def _build_balanced_repair_prompt(
    text: str,
    compiled: PilotCompilerOutput,
    diagnostics: dict[str, Any],
) -> tuple[str, str]:
    repair_ids = set(diagnostics.get("problem_node_ids") or [])
    problem_nodes = [node.model_dump() for node in compiled.nodes if node.id in repair_ids]
    system = (
        "You are the Local Balanced Compiler Repair Agent inside the Discussion Hub. Repair only the listed problematic "
        "nodes; never recompile the whole graph and never alter valid nodes. A replacement must be an exact contiguous quote "
        "from the supplied text and the smallest semantically complete proposition that can receive one verdict. It must not "
        "be an isolated entity, number, modifier, heading, transition, or discourse marker. Split only when parts can receive "
        "different verdicts. Preserve negation, modality, population, time, numbers, units, conditions, comparison, and causal "
        "strength. Use the original node id for the first replacement whenever possible so existing edges remain stable. Drop "
        "a node only when it is genuinely non-propositional. Return a local patch only, not a full graph, and do not judge the "
        "author's correctness."
    )
    user = (
        f"DISCUSSION TEXT\n{text}\n\n"
        f"PROBLEMATIC NODES\n{json.dumps(problem_nodes, ensure_ascii=False)}\n\n"
        f"PROBLEM REASONS\n{json.dumps(diagnostics.get('node_reasons') or {}, ensure_ascii=False)}\n\n"
        "Return replacements for listed ids, optional added_nodes only when required to complete a split of a listed node, "
        "and drop_node_ids only for non-factual fragments."
    )
    return system, user


def _apply_balanced_compiler_repair(
    compiled: PilotCompilerOutput,
    repair: DiscussionCompilerRepairOutput,
) -> PilotCompilerOutput:
    replacement_map = {item.original_node_id: list(item.replacement_nodes) for item in repair.replacements}
    drop_ids = set(repair.drop_node_ids)
    priority_map = {row.node_id: row for row in compiled.node_priorities}
    nodes: list[DiscussionNode] = []
    priorities: list[CompilerNodePriority] = []
    extra_counter = 0
    retained_ids: set[str] = set()
    for node in compiled.nodes:
        if node.id in drop_ids:
            continue
        replacements = replacement_map.get(node.id)
        if replacements is None:
            nodes.append(node)
            retained_ids.add(node.id)
            if node.id in priority_map:
                priorities.append(priority_map[node.id])
            continue
        for index, replacement in enumerate(replacements):
            row = replacement.model_copy(deep=True)
            if index == 0:
                row.id = node.id
            else:
                extra_counter += 1
                row.id = f"repair_extra_{extra_counter}"
            nodes.append(row)
            retained_ids.add(row.id)
            old_priority = priority_map.get(node.id)
            priorities.append(CompilerNodePriority(
                node_id=row.id,
                importance=float(old_priority.importance) if old_priority else 0.7,
                reason=(old_priority.reason if old_priority else "Local Balanced Compiler repair"),
            ))
    for added in repair.added_nodes:
        extra_counter += 1
        row = added.model_copy(deep=True)
        row.id = f"repair_added_{extra_counter}"
        nodes.append(row)
        retained_ids.add(row.id)
        priorities.append(CompilerNodePriority(
            node_id=row.id, importance=0.7, reason="Local Balanced Compiler repair addition"
        ))
    edges = [edge for edge in compiled.edges if edge.source in retained_ids and edge.target in retained_ids]
    return PilotCompilerOutput(
        paragraph_summary=compiled.paragraph_summary,
        nodes=nodes,
        edges=edges,
        node_priorities=priorities,
        omitted_content_summary=compiled.omitted_content_summary,
    )

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



def _bounded_env_int(name: str, default: int, *, minimum: int, maximum: int) -> int:
    raw = str(os.getenv(name, "") or "").strip()
    value = default if not raw else int(raw)
    return max(minimum, min(maximum, value))


def _bounded_env_float(name: str, default: float, *, minimum: float, maximum: float) -> float:
    raw = str(os.getenv(name, "") or "").strip()
    value = default if not raw else float(raw)
    return max(minimum, min(maximum, value))


def _semantic_text(node: DiscussionNode) -> str:
    value = str(node.plain_meaning or node.source_text or "").lower()
    value = re.sub(r"\s+", " ", value)
    return re.sub(r"[^\w%+\-./ ]+", "", value).strip()


def _spo_signature(node: DiscussionNode) -> tuple[str, str, str, str]:
    normalize = lambda value: re.sub(r"[^\w]+", " ", str(value or "").lower()).strip()
    return (
        normalize(node.subject),
        normalize(node.predicate),
        normalize(node.object),
        str(node.polarity),
    )


def _nodes_are_duplicates(a: DiscussionNode, b: DiscussionNode, *, threshold: float) -> bool:
    if a.polarity != b.polarity:
        return False
    source_a = re.sub(r"\s+", " ", str(a.source_text or "").lower()).strip()
    source_b = re.sub(r"\s+", " ", str(b.source_text or "").lower()).strip()
    if source_a and source_a == source_b:
        return True
    text_a, text_b = _semantic_text(a), _semantic_text(b)
    if not text_a or not text_b:
        return False
    if text_a == text_b:
        return True
    spo_a, spo_b = _spo_signature(a), _spo_signature(b)
    if all(spo_a[:3]) and spo_a == spo_b:
        return True
    if a.role != b.role or a.assertion_type != b.assertion_type:
        return False
    nums_a = tuple(sorted(str(x) for x in a.numeric_mentions))
    nums_b = tuple(sorted(str(x) for x in b.numeric_mentions))
    if nums_a != nums_b:
        return False
    if min(len(text_a), len(text_b)) < 18:
        return False
    return SequenceMatcher(None, text_a, text_b).ratio() >= threshold


_ROLE_IMPORTANCE: dict[str, float] = {
    "conclusion": 100.0,
    "limitation": 94.0,
    "evidence": 88.0,
    "observation": 84.0,
    "claim": 82.0,
    "study_design": 80.0,
    "analysis_method": 78.0,
    "selection_criterion": 76.0,
    "eligibility_criterion": 74.0,
    "exposure_definition": 74.0,
    "mechanism": 70.0,
}


def _node_scores(
    nodes: list[DiscussionNode],
    edges: list[DiscussionEdge],
    priorities: dict[str, float],
) -> dict[str, float]:
    incoming: dict[str, list[str]] = {node.id: [] for node in nodes}
    outgoing: dict[str, list[str]] = {node.id: [] for node in nodes}
    for edge in edges:
        if edge.source in outgoing and edge.target in incoming:
            outgoing[edge.source].append(edge.target)
            incoming[edge.target].append(edge.source)
    scores: dict[str, float] = {}
    for node in nodes:
        score = _ROLE_IMPORTANCE.get(str(node.role), 60.0)
        score += min(24.0, 4.0 * (len(incoming[node.id]) + len(outgoing[node.id])))
        score += 20.0 * priorities.get(node.id, 0.0)
        if node.numeric_mentions:
            score += 3.0
        lowered = _semantic_text(node)
        if any(token in lowered for token in ("therefore", "overall", "primary", "main finding", "however", "limitation", "결론", "따라서", "주요")):
            score += 5.0
        scores[node.id] = score

    # Preserve the evidence chain feeding conclusions rather than retaining
    # isolated high-scoring restatements.
    frontier = [node.id for node in nodes if node.role == "conclusion"]
    visited = set(frontier)
    depth = 0
    while frontier and depth < 8:
        next_frontier: list[str] = []
        for target in frontier:
            for source in incoming.get(target, []):
                scores[source] = scores.get(source, 0.0) + max(4.0, 18.0 - depth * 2.0)
                if source not in visited:
                    visited.add(source)
                    next_frontier.append(source)
        frontier = next_frontier
        depth += 1
    return scores


def _merge_node_metadata(target: DiscussionNode, source: DiscussionNode) -> None:
    target.confidence = max(float(target.confidence), float(source.confidence))
    target.numeric_mentions = list(dict.fromkeys([*target.numeric_mentions, *source.numeric_mentions]))
    target.inferred_details = list(dict.fromkeys([*target.inferred_details, *source.inferred_details]))
    if not target.why_it_matters and source.why_it_matters:
        target.why_it_matters = source.why_it_matters


def _compact_nodes_edges(
    nodes: list[DiscussionNode],
    edges: list[DiscussionEdge],
    *,
    priorities: dict[str, float] | None,
    node_cap: int,
    edge_cap: int,
    dedup_threshold: float,
) -> tuple[list[DiscussionNode], list[DiscussionEdge], dict[str, str], dict[str, Any]]:
    priorities = priorities or {}
    original_node_count = len(nodes)
    original_edge_count = len(edges)
    pre_scores = _node_scores(nodes, edges, priorities)

    canonical: list[DiscussionNode] = []
    original_to_canonical: dict[str, str] = {}
    duplicate_groups: dict[str, list[str]] = {}
    # Higher-importance nodes become the canonical representative.
    ordered_nodes = sorted(nodes, key=lambda n: (-pre_scores.get(n.id, 0.0), n.sentence_index, n.id))
    for node in ordered_nodes:
        duplicate = next(
            (existing for existing in canonical if _nodes_are_duplicates(node, existing, threshold=dedup_threshold)),
            None,
        )
        if duplicate is None:
            copied = node.model_copy(deep=True)
            canonical.append(copied)
            original_to_canonical[node.id] = copied.id
            duplicate_groups[copied.id] = [node.id]
        else:
            original_to_canonical[node.id] = duplicate.id
            duplicate_groups.setdefault(duplicate.id, [duplicate.id]).append(node.id)
            _merge_node_metadata(duplicate, node)

    edge_best: dict[tuple[str, str, str], DiscussionEdge] = {}
    for edge in edges:
        source = original_to_canonical.get(edge.source)
        target = original_to_canonical.get(edge.target)
        if not source or not target or source == target:
            continue
        key = (source, target, str(edge.relation))
        copied = edge.model_copy(deep=True)
        copied.source, copied.target = source, target
        current = edge_best.get(key)
        if current is None or copied.confidence > current.confidence:
            edge_best[key] = copied
    deduped_edges = list(edge_best.values())

    canonical_priorities: dict[str, float] = {}
    for old_id, canonical_id in original_to_canonical.items():
        canonical_priorities[canonical_id] = max(
            canonical_priorities.get(canonical_id, 0.0), priorities.get(old_id, 0.0)
        )
    scores = _node_scores(canonical, deduped_edges, canonical_priorities)
    selected = sorted(canonical, key=lambda n: (-scores.get(n.id, 0.0), n.sentence_index, n.id))[:node_cap]
    selected_ids = {node.id for node in selected}

    relation_bonus = {
        "contradicts": 18.0,
        "limits": 14.0,
        "does_not_establish": 14.0,
        "evidence_for": 10.0,
        "supports": 8.0,
        "causes": 8.0,
    }
    retained_edges = [
        edge for edge in deduped_edges
        if edge.source in selected_ids and edge.target in selected_ids
    ]
    retained_edges.sort(
        key=lambda e: -(
            scores.get(e.source, 0.0) + scores.get(e.target, 0.0)
            + relation_bonus.get(str(e.relation), 0.0) + float(e.confidence) * 5.0
        )
    )
    retained_edges = retained_edges[:edge_cap]

    # Public output keeps sequential ids and no gaps. Return a complete map so
    # issues from chunk-level specialists can follow merged canonical nodes.
    selected.sort(key=lambda n: (n.sentence_index, n.id))
    canonical_to_final: dict[str, str] = {}
    final_nodes: list[DiscussionNode] = []
    for index, node in enumerate(selected, 1):
        row = node.model_copy(deep=True)
        new_id = f"d{index}"
        canonical_to_final[node.id] = new_id
        row.id = new_id
        final_nodes.append(row)
    final_edges: list[DiscussionEdge] = []
    for index, edge in enumerate(retained_edges, 1):
        if edge.source not in canonical_to_final or edge.target not in canonical_to_final:
            continue
        row = edge.model_copy(deep=True)
        row.id = f"e{index}"
        row.source = canonical_to_final[edge.source]
        row.target = canonical_to_final[edge.target]
        final_edges.append(row)

    full_map = {
        old_id: canonical_to_final[canonical_id]
        for old_id, canonical_id in original_to_canonical.items()
        if canonical_id in canonical_to_final
    }
    meta = {
        "original_node_count": original_node_count,
        "deduplicated_node_count": len(canonical),
        "retained_node_count": len(final_nodes),
        "original_edge_count": original_edge_count,
        "deduplicated_edge_count": len(deduped_edges),
        "retained_edge_count": len(final_edges),
        "duplicate_node_count": max(0, original_node_count - len(canonical)),
        "pruned_node_count": max(0, len(canonical) - len(final_nodes)),
        "duplicate_groups": {k: v for k, v in duplicate_groups.items() if len(v) > 1},
    }
    return final_nodes, final_edges, full_map, meta


def compact_compiler_output(
    compiled: PilotCompilerOutput,
    *,
    node_cap: int,
    edge_cap: int,
    dedup_threshold: float,
) -> tuple[PilotCompilerOutput, dict[str, Any]]:
    priorities = {row.node_id: float(row.importance) for row in compiled.node_priorities}
    nodes, edges, id_map, meta = _compact_nodes_edges(
        compiled.nodes,
        compiled.edges,
        priorities=priorities,
        node_cap=node_cap,
        edge_cap=edge_cap,
        dedup_threshold=dedup_threshold,
    )
    priority_rows = [
        CompilerNodePriority(
            node_id=id_map[row.node_id], importance=row.importance, reason=row.reason
        )
        for row in compiled.node_priorities if row.node_id in id_map
    ]
    return PilotCompilerOutput(
        paragraph_summary=compiled.paragraph_summary,
        nodes=nodes,
        edges=edges,
        node_priorities=priority_rows,
        omitted_content_summary=compiled.omitted_content_summary,
    ), meta


def compact_document_output(
    output: DiscussionGraphOutput,
    *,
    node_cap: int,
    edge_cap: int,
    dedup_threshold: float = DEFAULT_NODE_DEDUP_THRESHOLD,
) -> tuple[DiscussionGraphOutput, dict[str, Any]]:
    nodes, edges, id_map, meta = _compact_nodes_edges(
        output.nodes,
        output.edges,
        priorities=None,
        node_cap=node_cap,
        edge_cap=edge_cap,
        dedup_threshold=dedup_threshold,
    )
    issues: list[DiscussionIssue] = []
    seen_issue_keys: set[tuple[str, tuple[str, ...], str]] = set()
    for issue in output.issues:
        mapped = list(dict.fromkeys(id_map[x] for x in issue.node_ids if x in id_map))
        if not mapped:
            continue
        row = issue.model_copy(deep=True)
        row.node_ids = mapped
        key = (str(row.issue_type), tuple(sorted(mapped)), row.title.strip().lower())
        if key in seen_issue_keys:
            continue
        seen_issue_keys.add(key)
        row.id = f"i{len(issues) + 1}"
        issues.append(row)
    return DiscussionGraphOutput(
        paragraph_summary=output.paragraph_summary,
        nodes=nodes,
        edges=edges,
        issues=issues,
        overall_assessment=output.overall_assessment,
    ), meta


def _compact_graph_json(state: SharedReasoningGraph6, *, include_source_alignment: bool = False) -> str:
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


def build_specialist_prompt(specialist: SpecialistName, state: SharedReasoningGraph6) -> tuple[str, str]:
    rule = _SPECIALIST_RULES[specialist]
    system = (
        f"You are the {specialist.capitalize()} Agent in a fixed six-agent graph-native pilot. "
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



def _collect_assumption_candidates(findings: list[PilotFinding]) -> list[AssumptionCandidate]:
    """Collect only explicit Logic-agent missing-premise proposals.

    The Assumption Agent is not called for a vague low-confidence inference.
    A proposal must be tied to a concrete graph edge, explicitly labelled as a
    missing/unjustified premise, or accompanied by a request_missing_premise
    patch. This keeps the call conditional without losing genuine cases where
    Logic used a broader label such as unsupported_inference.
    """

    rows: list[AssumptionCandidate] = []
    seen: set[tuple[str, str]] = set()
    for finding in findings:
        if not finding.missing_node_proposals:
            continue
        explicit_patch = any(
            patch.operation == "request_missing_premise" for patch in finding.patches
        )
        explicit_label = finding.canonical_issue_type in {
            "missing_premise", "unjustified_assumption"
        }
        for proposal in finding.missing_node_proposals:
            text_key = re.sub(r"\s+", " ", proposal.text.lower()).strip()
            edge_tied = bool(proposal.required_for_edge_id)
            if not text_key or not (edge_tied or explicit_label or explicit_patch):
                continue
            key = (proposal.required_for_edge_id, text_key)
            if key in seen:
                continue
            seen.add(key)
            rows.append(AssumptionCandidate(
                proposal_id=f"a{len(rows) + 1}",
                logic_finding_id=finding.id,
                text=proposal.text,
                required_for_edge_id=proposal.required_for_edge_id,
                rationale=proposal.rationale,
                confidence=proposal.confidence,
            ))
    return rows


def build_assumption_prompt(
    state: SharedReasoningGraph6,
    candidates: list[AssumptionCandidate],
) -> tuple[str, str]:
    system = (
        "You are Agent 4 of 6: the Assumption Agent. Evaluate only the missing-premise candidates proposed by the Logic Agent. "
        "A missing premise is not automatically an error. Distinguish definitions and ordinary background knowledge from uncertain or critical unsupported assumptions. "
        "Use the shared graph and quoted node text; do not solve the original document from scratch and do not invent new unrelated assumptions. "
        f"Allowed disposition vocabulary only: {', '.join(ASSUMPTION_DISPOSITION_VOCAB)}. "
        "Use accepted_definition for meaning fixed by a term or rule; accepted_background for stable ordinary background knowledge; "
        "supported_by_context or explicitly_supported when graph nodes justify it; plausible_but_unverified when reasonable but not established; "
        "unsupported_critical_assumption when the conclusion depends on a substantive unsupported premise; overly_strong_assumption for a stronger premise than needed; "
        "circular_assumption when it presupposes the conclusion; irrelevant_assumption when the Logic proposal is unnecessary. "
        "Set proposed_edge_status to valid, conditional, insufficient_information, or invalid as warranted."
    )
    payload = {
        "graph": json.loads(_compact_graph_json(state, include_source_alignment=True)),
        "assumption_candidates": [row.model_dump() for row in candidates],
    }
    return system, "Assess these proposed assumptions:\n" + json.dumps(
        payload, ensure_ascii=False, separators=(",", ":")
    )


def _validate_assumption_assessments(
    review: PilotAssumptionReviewOutput,
    candidates: list[AssumptionCandidate],
    state: SharedReasoningGraph6,
) -> list[AssumptionAssessment]:
    by_id = {row.proposal_id: row for row in candidates}
    rows: list[AssumptionAssessment] = []
    seen: set[str] = set()
    for assessment in review.assessments:
        candidate = by_id.get(assessment.proposal_id)
        if candidate is None or assessment.proposal_id in seen:
            continue
        seen.add(assessment.proposal_id)
        assessment.logic_finding_id = candidate.logic_finding_id
        assessment.required_for_edge_id = candidate.required_for_edge_id
        assessment.supporting_node_ids = [
            x for x in dict.fromkeys(assessment.supporting_node_ids) if x in state.node_ids()
        ]
        rows.append(assessment)
    return rows


_ACCEPTED_ASSUMPTIONS = {
    "accepted_definition",
    "accepted_background",
    "supported_by_context",
    "explicitly_supported",
}
_INVALID_ASSUMPTIONS = {
    "unsupported_critical_assumption",
    "overly_strong_assumption",
    "circular_assumption",
}


def _apply_assumption_assessments(
    findings: list[PilotFinding],
    candidates: list[AssumptionCandidate],
    assessments: list[AssumptionAssessment],
) -> list[PilotFinding]:
    candidate_by_id = {row.proposal_id: row for row in candidates}
    assessments_by_finding: dict[str, list[AssumptionAssessment]] = {}
    for assessment in assessments:
        assessments_by_finding.setdefault(assessment.logic_finding_id, []).append(assessment)

    result: list[PilotFinding] = []
    for finding in findings:
        related = assessments_by_finding.get(finding.id, [])
        if not related:
            result.append(finding)
            continue

        dispositions = {row.disposition for row in related}
        # A Logic finding that exists only because of acceptable background or
        # an unnecessary proposal is removed rather than producing a false Red.
        if dispositions and dispositions <= (_ACCEPTED_ASSUMPTIONS | {"irrelevant_assumption"}):
            if finding.canonical_issue_type in {"missing_premise", "unjustified_assumption"}:
                continue
            finding.missing_node_proposals = []
            result.append(finding)
            continue

        if "circular_assumption" in dispositions:
            finding.canonical_issue_type = "circular_reasoning"
            finding.public_issue_type = CANONICAL_TO_PUBLIC["circular_reasoning"]
            finding.verdict = "invalid"
            finding.severity = "high"
            finding.title = "The inference depends on a circular assumption"
        elif dispositions & {"unsupported_critical_assumption", "overly_strong_assumption"}:
            finding.canonical_issue_type = "unjustified_assumption"
            finding.public_issue_type = CANONICAL_TO_PUBLIC["unjustified_assumption"]
            finding.verdict = "invalid"
            finding.severity = "high" if "unsupported_critical_assumption" in dispositions else "medium"
            finding.title = "The inference depends on an unjustified assumption"
        elif "plausible_but_unverified" in dispositions:
            finding.verdict = "uncertain"
            finding.severity = "medium" if finding.severity == "high" else finding.severity
            finding.title = "The inference is conditional on an unverified assumption"
            finding.explanation = (
                finding.explanation.rstrip() + " The missing premise is plausible, but the supplied text does not establish it."
            ).strip()
            finding.confidence = min(
                finding.confidence,
                max(row.confidence for row in related if row.disposition == "plausible_but_unverified"),
            )
        result.append(finding)
    return result


def _normalize_finding_targets(finding: PilotFinding, state: SharedReasoningGraph6) -> PilotFinding | None:
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


def _valid_findings(review: PilotSpecialistReviewOutput, state: SharedReasoningGraph6) -> list[PilotFinding]:
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


def _important_judge_targets(state: SharedReasoningGraph6) -> set[str]:
    """Return graph elements whose status can directly affect a conclusion."""

    conclusion_ids = {node.id for node in state.nodes if node.role == "conclusion"}
    important: set[str] = set(conclusion_ids)
    for edge in state.edges:
        if edge.target in conclusion_ids:
            important.add(edge.id)
            important.add(edge.source)
    return important


def _judge_trigger_reasons(
    findings: list[PilotFinding],
    state: SharedReasoningGraph6,
) -> list[str]:
    """Use the Judge for meaningful ambiguity, without requiring a hard clash.

    The trigger is intentionally moderate: unresolved medium/high findings,
    overlapping labels on the same target, proposed graph revisions, or a
    low-confidence high-impact finding can all merit adjudication. A merely
    low-confidence minor finding does not.
    """

    reasons: list[str] = []
    important_targets = _important_judge_targets(state)
    threshold = _bounded_env_float(
        "DISCUSSION_JUDGE_CONFIDENCE_THRESHOLD", 0.65, minimum=0.4, maximum=0.9
    )

    if any(
        finding.verdict == "uncertain" and finding.severity in {"high", "medium"}
        for finding in findings
    ):
        reasons.append("material_uncertainty")

    by_target: dict[tuple[str, ...], list[PilotFinding]] = {}
    for finding in findings:
        target = _finding_target_key(finding)
        if target:
            by_target.setdefault(target, []).append(finding)
    if any(
        len(rows) > 1 and (
            len({row.canonical_issue_type for row in rows}) > 1
            or len({row.verdict for row in rows}) > 1
        )
        for rows in by_target.values()
    ):
        reasons.append("overlapping_or_conflicting_findings")

    graph_revision_ops = {"change_edge_type", "qualify_claim"}
    if any(
        patch.operation in graph_revision_ops
        for finding in findings
        for patch in finding.patches
    ):
        reasons.append("graph_revision_proposed")

    if any(
        finding.confidence < threshold
        and finding.severity in {"high", "medium"}
        and bool((set(finding.node_ids) | set(finding.edge_ids)) & important_targets)
        for finding in findings
    ):
        reasons.append("low_confidence_high_impact")

    return list(dict.fromkeys(reasons))


def _needs_judge(findings: list[PilotFinding], state: SharedReasoningGraph6) -> bool:
    return bool(_judge_trigger_reasons(findings, state))


def build_judge_prompt(state: SharedReasoningGraph6, findings: list[PilotFinding]) -> tuple[str, str]:
    system = (
        "You are Agent 6 of 6: the conditional Judge. You do not re-read or solve the original text from scratch. "
        "Adjudicate only materially uncertain, overlapping, conflicting, or graph-revising candidate findings. "
        "Accept supported findings, reject speculative or duplicate ones, or revise them to a narrower target or better label. "
        "Do not add unrelated findings. Preserve the fixed canonical and public vocabularies. "
        f"Canonical vocabulary: {', '.join(CANONICAL_ISSUE_VOCAB)}. "
        f"Public vocabulary: {', '.join(PUBLIC_ISSUE_VOCAB)}."
    )
    payload = {
        "graph": json.loads(_compact_graph_json(state, include_source_alignment=True)),
        "assumption_assessments": [row.model_dump() for row in state.assumption_assessments],
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


def run_graph_native_6agents_chunk(
    text: str,
    *,
    model: str,
    reasoning_effort: str,
    max_output_tokens: int,
    custom_instruction: str,
    client: Any,
    judge_enabled: bool = True,
) -> tuple[DiscussionGraphOutput, list[Any], float, dict[str, Any]]:
    """Capped Compiler -> parallel Evidence/Logic/Target -> conditional Assumption -> conditional Judge.

    The public DiscussionGraphOutput is unchanged. Node budgets, deduplication,
    source alignments, assumption assessments, parallel traces, and judge traces
    remain internal. This is the general Discussion Lab path; no RAGQA-specific
    source/response dual-graph logic is introduced here.
    """

    node_cap = _bounded_env_int(
        "DISCUSSION_MAX_NODES_PER_CHUNK", DEFAULT_COMPILER_NODE_CAP,
        minimum=8, maximum=ABSOLUTE_COMPILER_NODE_HARD_CAP,
    )
    edge_cap = _bounded_env_int(
        "DISCUSSION_MAX_EDGES_PER_CHUNK", DEFAULT_COMPILER_EDGE_CAP,
        minimum=8, maximum=ABSOLUTE_COMPILER_EDGE_HARD_CAP,
    )
    dedup_threshold = _bounded_env_float(
        "DISCUSSION_NODE_DEDUP_THRESHOLD", DEFAULT_NODE_DEDUP_THRESHOLD,
        minimum=0.85, maximum=1.0,
    )

    all_responses: list[Any] = []
    total_latency = 0.0
    private_trace: dict[str, Any] = {
        "architecture": PILOT_ARCHITECTURE_NAME,
        "agent_count": PILOT_AGENT_COUNT,
        "agents": list(PILOT_AGENT_NAMES),
        "execution": "parallel_core_specialists",
        "stages": [],
        "compiler": {
            "version": BALANCED_COMPILER_VERSION,
            "contract": "smallest semantically complete proposition; local repair only; no whole-graph rewrite",
        },
        "compiler_limits": {
            "node_cap": node_cap,
            "edge_cap": edge_cap,
            "dedup_threshold": dedup_threshold,
        },
        "vocabulary": {
            "node_roles": list(NODE_ROLE_VOCAB),
            "edge_relations": list(EDGE_RELATION_VOCAB),
            "validation_statuses": list(get_args(ValidationStatus)),
            "canonical_issue_types": list(CANONICAL_ISSUE_VOCAB),
            "assumption_dispositions": list(ASSUMPTION_DISPOSITION_VOCAB),
        },
    }

    compiler_system, compiler_user = build_compiler_prompt(
        text, custom_instruction, node_cap=node_cap, edge_cap=edge_cap
    )
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

    initial_compiler_diagnostics = _balanced_compiler_diagnostics(compiled_raw, text)
    compiler_refinement: dict[str, Any] = {
        "attempted": False,
        "mode": "none",
        "compiler_version": BALANCED_COMPILER_VERSION,
        "initial_diagnostics": initial_compiler_diagnostics,
        "final_diagnostics": initial_compiler_diagnostics,
    }
    if initial_compiler_diagnostics["needs_refinement"]:
        repair_system, repair_user = _build_balanced_repair_prompt(
            text, compiled_raw, initial_compiler_diagnostics
        )
        repair_raw, repair_responses, repair_latency = _call_structured_stage(
            client=client,
            model=model,
            reasoning_effort=reasoning_effort,
            max_output_tokens=max(1200, min(2600, max_output_tokens)),
            system=repair_system,
            user=repair_user,
            schema=DiscussionCompilerRepairOutput,
        )
        assert isinstance(repair_raw, DiscussionCompilerRepairOutput)
        compiled_raw = _apply_balanced_compiler_repair(compiled_raw, repair_raw)
        all_responses.extend(repair_responses)
        total_latency += repair_latency
        final_diagnostics = _balanced_compiler_diagnostics(compiled_raw, text)
        compiler_refinement = {
            "attempted": True,
            "mode": "local_node_patch",
            "compiler_version": BALANCED_COMPILER_VERSION,
            "problem_node_ids": list(initial_compiler_diagnostics.get("problem_node_ids") or []),
            "replacement_groups": len(repair_raw.replacements),
            "added_nodes": len(repair_raw.added_nodes),
            "dropped_nodes": len(repair_raw.drop_node_ids),
            "initial_diagnostics": initial_compiler_diagnostics,
            "final_diagnostics": final_diagnostics,
            "remaining_quality_warning": bool(final_diagnostics["needs_refinement"]),
        }
        private_trace["stages"].append({
            "stage": "agent:compiler_local_repair",
            "status": "ok",
            "api_calls": len(repair_responses),
            "latency_ms": repair_latency,
            "problem_node_ids": list(initial_compiler_diagnostics.get("problem_node_ids") or []),
            "remaining_quality_warning": bool(final_diagnostics["needs_refinement"]),
        })

    compiled_raw, compiler_compaction = compact_compiler_output(
        compiled_raw,
        node_cap=node_cap,
        edge_cap=edge_cap,
        dedup_threshold=dedup_threshold,
    )
    private_trace["balanced_compiler_version"] = BALANCED_COMPILER_VERSION
    private_trace["compiler_refinement"] = compiler_refinement
    private_trace["stages"].append({
        "stage": "agent:compiler",
        "status": "ok",
        "api_calls": len(responses),
        "latency_ms": latency,
        "compiler_version": BALANCED_COMPILER_VERSION,
        "refinement": compiler_refinement,
        "compaction": compiler_compaction,
        "omitted_content_summary": compiled_raw.omitted_content_summary,
    })

    state = SharedReasoningGraph6(
        source_text=text,
        nodes=compiled_raw.nodes,
        edges=compiled_raw.edges,
        source_alignment=_resolve_source_alignment(text, compiled_raw.nodes),
    )

    def run_specialist_isolated(specialist: SpecialistName) -> dict[str, Any]:
        system, user = build_specialist_prompt(specialist, state)
        try:
            review_raw, stage_responses, stage_latency = _call_structured_stage(
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
            return {
                "specialist": specialist,
                "status": "ok",
                "review": review_raw,
                "responses": stage_responses,
                "latency_ms": stage_latency,
            }
        except Exception as exc:
            return {
                "specialist": specialist,
                "status": "failed",
                "review": None,
                "responses": [],
                "latency_ms": 0.0,
                "error": f"{type(exc).__name__}: {exc}",
            }

    core_specialists: tuple[SpecialistName, ...] = ("evidence", "logic", "target")
    parallel_started = time.perf_counter()
    specialist_results: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=3, thread_name_prefix="discussion-agent") as executor:
        future_map = {
            executor.submit(run_specialist_isolated, specialist): specialist
            for specialist in core_specialists
        }
        for future in as_completed(future_map):
            specialist = future_map[future]
            try:
                specialist_results[specialist] = future.result()
            except Exception as exc:
                specialist_results[specialist] = {
                    "specialist": specialist,
                    "status": "failed",
                    "review": None,
                    "responses": [],
                    "latency_ms": 0.0,
                    "error": f"{type(exc).__name__}: {exc}",
                }
    parallel_wall_latency = round((time.perf_counter() - parallel_started) * 1000, 3)
    total_latency += parallel_wall_latency
    private_trace["stages"].append({
        "stage": "parallel:evidence_logic_target",
        "status": "ok",
        "specialists": list(core_specialists),
        "wall_latency_ms": parallel_wall_latency,
    })

    all_findings: list[PilotFinding] = []
    logic_findings: list[PilotFinding] = []
    for specialist in core_specialists:
        result = specialist_results[specialist]
        all_responses.extend(result["responses"])
        review = result.get("review")
        if result["status"] == "ok" and isinstance(review, PilotSpecialistReviewOutput):
            state.apply_review(review)
            all_findings.extend(review.findings)
            if specialist == "logic":
                logic_findings = list(review.findings)
            private_trace["stages"].append({
                "stage": f"agent:{specialist}",
                "status": "ok",
                "api_calls": len(result["responses"]),
                "finding_count": len(review.findings),
                "individual_latency_ms": result["latency_ms"],
                "executed_in_parallel": True,
            })
        else:
            private_trace["stages"].append({
                "stage": f"agent:{specialist}",
                "status": "failed",
                "error": result.get("error", "unknown specialist failure"),
                "executed_in_parallel": True,
            })

    assumption_candidates = _collect_assumption_candidates(logic_findings)
    if assumption_candidates:
        system, user = build_assumption_prompt(state, assumption_candidates)
        try:
            assumption_raw, responses, latency = _call_structured_stage(
                client=client,
                model=model,
                reasoning_effort=reasoning_effort,
                max_output_tokens=max(1400, min(4200, max_output_tokens)),
                system=system,
                user=user,
                schema=PilotAssumptionReviewOutput,
            )
            assert isinstance(assumption_raw, PilotAssumptionReviewOutput)
            assessments = _validate_assumption_assessments(
                assumption_raw, assumption_candidates, state
            )
            state.assumption_assessments = assessments
            all_responses.extend(responses)
            total_latency += latency
            all_findings = _apply_assumption_assessments(
                all_findings, assumption_candidates, assessments
            )
            private_trace["stages"].append({
                "stage": "agent:assumption",
                "status": "ok",
                "api_calls": len(responses),
                "latency_ms": latency,
                "candidate_count": len(assumption_candidates),
                "assessment_count": len(assessments),
                "trigger": "explicit_logic_missing_premise_proposal",
            })
        except Exception as exc:
            private_trace["stages"].append({
                "stage": "agent:assumption",
                "status": "failed",
                "candidate_count": len(assumption_candidates),
                "error": f"{type(exc).__name__}: {exc}",
            })
    else:
        private_trace["stages"].append({
            "stage": "agent:assumption",
            "status": "not_needed",
            "api_calls": 0,
            "candidate_count": 0,
            "trigger": "no_explicit_logic_missing_premise_proposal",
        })

    all_findings = _deduplicate_findings(all_findings)
    judge_reasons = _judge_trigger_reasons(all_findings, state) if judge_enabled else []

    if judge_enabled and all_findings and judge_reasons:
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
                "latency_ms": latency,
                "trigger_reasons": judge_reasons,
            })
        except Exception as exc:
            private_trace["stages"].append({
                "stage": "agent:judge",
                "status": "failed",
                "trigger_reasons": judge_reasons,
                "error": f"{type(exc).__name__}: {exc}",
            })
    else:
        private_trace["stages"].append({
            "stage": "agent:judge",
            "status": "not_needed" if judge_enabled else "disabled",
            "api_calls": 0,
            "trigger_reasons": judge_reasons,
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
    private_trace["assumption_candidates"] = [row.model_dump() for row in assumption_candidates]
    private_trace["assumption_assessments"] = [row.model_dump() for row in state.assumption_assessments]
    private_trace["missing_premise_candidate_count"] = len(assumption_candidates)
    private_trace["judge_trigger_reasons"] = judge_reasons
    private_trace["final_agent_finding_count"] = len(public_issues)

    output = DiscussionGraphOutput(
        paragraph_summary=compiled_raw.paragraph_summary,
        nodes=compiled_raw.nodes,
        edges=compiled_raw.edges,
        issues=public_issues,
        overall_assessment="potential_issue" if public_issues else "internally_consistent",
    )
    return output, all_responses, round(total_latency, 3), private_trace

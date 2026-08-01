from __future__ import annotations

import hashlib
import json
import os
import re
import time
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from .openai_runner import ALLOWED_REASONING_EFFORTS, DEFAULT_MODEL, _load_local_env, _usage_dict
from .graph_metrics import calculate_graph_metrics


DEFAULT_DISCUSSION_CHUNK_CHARS = 24000
MIN_DISCUSSION_CHUNK_CHARS = 4000


NodeRole = Literal[
    "observation", "evidence", "claim", "mechanism", "limitation", "conclusion",
    "study_design", "analysis_method", "eligibility_criterion", "selection_criterion",
    "exposure_definition",
]
AssertionType = Literal[
    "descriptive", "association", "causal", "temporal", "necessity", "sufficiency",
    "exclusivity", "scope", "safety", "statistical", "clinical_significance",
    "effect_magnitude", "study_design", "analysis_method", "eligibility", "selection",
    "mediator", "competing_event", "estimand", "reproducibility", "multiplicity",
    "subgroup_heterogeneity", "noninferiority", "equivalence", "other",
]
CausalRole = Literal[
    "none", "exposure", "outcome", "mediator", "confounder", "collider",
    "selection_variable", "competing_event", "effect_modifier",
]
MethodologicalRole = Literal[
    "none", "design_feature", "analysis_choice", "eligibility_rule", "selection_rule",
    "exposure_definition", "outcome_definition", "adjustment_variable", "estimand_definition",
]
Polarity = Literal["positive", "negative", "uncertain"]
Certainty = Literal[
    "observed", "reported", "suggests", "may", "likely", "concludes", "establishes",
    "proves", "uncertain",
]
EdgeRelation = Literal[
    "supports", "supports_weakly", "contradicts", "limits", "causes", "mediates",
    "precedes", "follows", "necessary_for", "not_necessary_for", "sufficient_for",
    "exclusive_through", "generalizes_to", "same_claim_as", "qualifies", "evidence_for",
    "defines_population", "defines_time_zero", "defines_exposure", "adjusts_for",
    "conditions_on", "excludes", "competes_with", "measured_after", "does_not_establish",
]
IssueType = Literal[
    "direct_contradiction", "semantic_contradiction", "magnitude_inflation",
    "causal_overclaim", "temporal_mechanism_conflict", "time_zero_mismatch",
    "temporal_scope_extrapolation", "scope_overreach", "necessity_violation",
    "exclusivity_conflict", "evidence_strength_mismatch", "unsupported_generalization",
    "unsupported_mechanism", "design_claim_mismatch", "subgroup_significance_fallacy",
    "unsupported_effect_heterogeneity", "attrition_bias", "informative_missingness",
    "landmark_selection_bias", "post_treatment_adjustment", "estimand_mismatch",
    "collider_bias_risk", "noninferiority_interpretation_error", "equivalence_fallacy",
    "competing_risk_misclassification", "multiplicity_risk", "selective_outcome_reporting",
    "reproducibility_conflict", "surrogate_to_clinical_overreach",
    # Backward-compatible v024 label:
    "temporal_inversion", "other",
]
IssueFamily = Literal[
    "consistency", "causal_strength", "temporal_logic", "scope_generalization",
    "statistical_interpretation", "study_design", "missing_data_selection",
    "estimand_method", "reproducibility_reporting", "other",
]
Severity = Literal["high", "medium", "low"]
OverallAssessment = Literal[
    "internally_consistent", "formal_conflict", "unsupported_conclusion",
    "methodological_risk", "mixed_concerns", "potential_issue", "clear_conflict",
]


class DiscussionNode(BaseModel):
    id: str = Field(description="Stable sequential id such as d1, d2, d3")
    sentence_index: int = Field(ge=1)
    source_text: str = Field(description="Exact source span copied from the paragraph; do not paraphrase")
    plain_meaning: str = Field(description="Short normalized reader-friendly meaning")
    role: NodeRole
    assertion_type: AssertionType
    polarity: Polarity
    certainty: Certainty
    subject: str = ""
    predicate: str = ""
    object: str = ""
    population_scope: str = "not specified"
    time_scope: str = "not specified"
    analysis_population: str = "not specified"
    estimand: str = "not specified"
    causal_role: CausalRole = "none"
    methodological_role: MethodologicalRole = "none"
    numeric_mentions: list[str] = Field(default_factory=list)
    inferred_details: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    why_it_matters: str = ""


class DiscussionEdge(BaseModel):
    id: str = Field(description="Stable sequential id such as e1, e2")
    source: str
    target: str
    relation: EdgeRelation
    rationale: str
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)


class DiscussionIssue(BaseModel):
    id: str = Field(description="Stable sequential id such as i1, i2")
    issue_type: IssueType
    severity: Severity
    title: str
    node_ids: list[str]
    explanation: str
    logical_pattern: str
    suggested_revision: str = ""
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)


class DiscussionGraphOutput(BaseModel):
    paragraph_summary: str
    nodes: list[DiscussionNode]
    edges: list[DiscussionEdge]
    issues: list[DiscussionIssue]
    overall_assessment: OverallAssessment


def build_discussion_prompt(text: str, custom_instruction: str = "") -> dict[str, str]:
    system = (
        "You audit the internal logical, statistical, methodological, and epistemic structure of a scientific Discussion-style paragraph. "
        "Do not fact-check references and do not use outside knowledge. Analyze only what the supplied text itself says. "
        "Extract atomic public claims as typed nodes and relations as edges. "
        "SOURCE FIDELITY IS MANDATORY: source_text must be an exact contiguous quote copied from the supplied paragraph. "
        "Do not add words such as statistically, clinically meaningful, robust, proven, or all patients unless those words are present in the source span. "
        "Preserve every number, percentage, time point, subgroup label, analysis name, and uncertainty phrase exactly in source_text. "
        "Use plain_meaning for normalization or interpretation, and place any inference not literally stated in inferred_details. "
        "Classify both the discourse function and methodological role. Study design, eligibility rules, exposure definitions, selection criteria, and analysis choices are not biological mechanisms. "
        "Use mechanism only for a claimed biological or causal pathway. "
        "Recognize these issue families when supported by the text: direct or semantic contradiction; magnitude inflation; association-to-causation overclaim; "
        "temporal mechanism conflict; time-zero mismatch; temporal scope extrapolation; necessity/exclusivity conflict; subgroup significance fallacy; unsupported effect heterogeneity; "
        "selective attrition/completer bias; landmark survivor selection; adjustment for a post-treatment mediator; estimand mismatch; collider-selection risk; "
        "noninferiority/equivalence interpretation error; competing-event misclassification; multiplicity/selective reporting; failed replication versus reproducibility; "
        "and biomarker/surrogate evidence generalized to clinical therapeutic benefit. "
        "Do not manufacture a contradiction merely because evidence is weak. Distinguish: (1) formal conflict, (2) structurally unsupported conclusion, "
        "(3) structurally identifiable methodological risk, and (4) a model-suggested concern requiring human review. "
        "Split compound conclusions into atomic logical components only when needed, but preserve their shared source span and avoid duplicate nodes that merely restate an included claim. "
        "Use sequential ids d1, d2, ...; edges e1, e2, ...; issues i1, i2, .... "
        "The output is an inspectable claim graph, not private chain-of-thought. Explanations must be concise and reader-facing."
    )
    if custom_instruction.strip():
        system += "\nAdditional instruction: " + custom_instruction.strip()
    user = (
        "Analyze this paragraph for its internal reasoning structure:\n\n"
        f"{text.strip()}\n\n"
        "Return the typed claim graph and identify only issues supported by the paragraph's own structure."
    )
    return {"system": system, "user": user}


def _parsed_from_response(response: Any) -> DiscussionGraphOutput:
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
                return parsed if isinstance(parsed, DiscussionGraphOutput) else DiscussionGraphOutput.model_validate(parsed)
    output_parsed = getattr(response, "output_parsed", None)
    if output_parsed is not None:
        return output_parsed if isinstance(output_parsed, DiscussionGraphOutput) else DiscussionGraphOutput.model_validate(output_parsed)
    output_text = str(getattr(response, "output_text", "") or "").strip()
    if output_text:
        return DiscussionGraphOutput.model_validate_json(output_text)
    if refusal_messages:
        raise ValueError("OpenAI model refusal: " + " | ".join(refusal_messages))
    raise ValueError("OpenAI response contained no parsed Discussion graph")


def _norm_space(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _norm_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value or "").lower()).strip("_")


def _sentences(text: str) -> list[str]:
    return [x.strip() for x in re.split(r"(?<=[.!?])\s+", _norm_space(text)) if x.strip()]


def _best_source_match(source_text: str, paragraph: str) -> tuple[str, str, float]:
    """Return (status, matched span, similarity)."""
    source = _norm_space(source_text)
    paragraph_norm = _norm_space(paragraph)
    if not source:
        return "unmatched", "", 0.0
    if source in paragraph_norm:
        return "exact", source, 1.0
    candidates = _sentences(paragraph_norm)
    if not candidates:
        return "unmatched", "", 0.0
    best = max(candidates, key=lambda x: SequenceMatcher(None, source.lower(), x.lower()).ratio())
    ratio = SequenceMatcher(None, source.lower(), best.lower()).ratio()
    if ratio >= 0.82:
        return "paraphrased", best, ratio
    if ratio >= 0.55:
        return "partial", best, ratio
    return "unmatched", best, ratio


def _extract_numeric_mentions(text: str) -> list[str]:
    patterns = re.findall(
        r"\b(?:p\s*[=<>]\s*)?\d+(?:\.\d+)?%?|\b\d+(?:\.\d+)?\s*(?:hours?|days?|weeks?|months?|years?|points?)\b",
        str(text or ""),
        flags=re.I,
    )
    seen: list[str] = []
    for item in patterns:
        value = _norm_space(item)
        if value and value not in seen:
            seen.append(value)
    return seen


def _combined_text(*parts: Any) -> str:
    return " ".join(_norm_space(str(x)) for x in parts if x is not None).lower()


def _reclassify_node(row: dict[str, Any]) -> None:
    text = _combined_text(row.get("source_text"), row.get("plain_meaning"), row.get("predicate"), row.get("object"))

    # Methodological roles take precedence over biological mechanism labels.
    if re.search(r"required for inclusion|eligible for|eligibility|included only|excluded from|were not analyzed", text):
        row["role"] = "eligibility_criterion" if "required for inclusion" in text or "eligible" in text else "selection_criterion"
        row["assertion_type"] = "eligibility" if row["role"] == "eligibility_criterion" else "selection"
        row["methodological_role"] = "eligibility_rule" if row["role"] == "eligibility_criterion" else "selection_rule"
        row["causal_role"] = "selection_variable"
    elif re.search(r"exposure was assigned|classified as exposed|medication use recorded|exposure definition", text):
        row["role"] = "exposure_definition"
        row["assertion_type"] = "analysis_method"
        row["methodological_role"] = "exposure_definition"
    elif re.search(r"adjusted for|controlled for|censored|kaplan.?meier|fine.?gray|cause-specific|primary model|primary analysis", text):
        row["role"] = "analysis_method"
        row["assertion_type"] = "analysis_method"
        row["methodological_role"] = "analysis_choice"
    elif re.search(r"retrospective|not randomized|nonrandomized|not designed|study design|prespecified|post hoc", text):
        if row.get("role") not in {"conclusion", "evidence"}:
            row["role"] = "study_design"
            row["assertion_type"] = "study_design"
            row["methodological_role"] = "design_feature"

    if re.search(r"post-treatment|after (?:therapy|treatment) began|three months after", text) and re.search(r"adjusted|controlled", text):
        row["assertion_type"] = "analysis_method"
        row["methodological_role"] = "adjustment_variable"
    if re.search(r"affected by treatment|treatment .* influenced .*response|mediator", text):
        row["assertion_type"] = "mediator"
        row["causal_role"] = "mediator"
    if re.search(r"death.*before.*recurrence|competing event|prevents subsequent recurrence", text):
        row["assertion_type"] = "competing_event"
        row["causal_role"] = "competing_event"
    if re.search(r"interaction|effect.*similar.*groups|subgroup", text):
        if row.get("assertion_type") in {"descriptive", "statistical", "other"}:
            row["assertion_type"] = "subgroup_heterogeneity"
    if re.search(r"noninferior|noninferiority", text):
        row["assertion_type"] = "noninferiority"
    if re.search(r"equivalent|equivalence|equally effective", text):
        row["assertion_type"] = "equivalence"
    if re.search(r"replicat|reproducib|validation cohort", text):
        row["assertion_type"] = "reproducibility"
    if re.search(r"multiple comparisons|multiplicity|twenty outcomes|20 outcomes|primary endpoint", text):
        row["assertion_type"] = "multiplicity"
    if re.search(r"modest|large|magnitude", text):
        if row.get("assertion_type") in {"descriptive", "clinical_significance", "other"}:
            row["assertion_type"] = "effect_magnitude"

    # An exclusive population claim is not automatically a biological mechanism.
    if row.get("role") == "mechanism" and re.search(r"only in (?:men|women|patients)|all patients|entire population|subgroup", text) and not re.search(r"mechanism|pathway|through|mediates", text):
        row["role"] = "conclusion" if row.get("certainty") in {"concludes", "establishes", "proves"} else "claim"
        row["assertion_type"] = "scope"


def _issue_family(issue_type: str) -> str:
    mapping = {
        "direct_contradiction": "consistency", "semantic_contradiction": "consistency",
        "magnitude_inflation": "statistical_interpretation", "causal_overclaim": "causal_strength",
        "temporal_mechanism_conflict": "temporal_logic", "temporal_inversion": "temporal_logic",
        "time_zero_mismatch": "temporal_logic", "temporal_scope_extrapolation": "temporal_logic",
        "scope_overreach": "scope_generalization", "unsupported_generalization": "scope_generalization",
        "necessity_violation": "consistency", "exclusivity_conflict": "consistency",
        "evidence_strength_mismatch": "statistical_interpretation",
        "unsupported_mechanism": "causal_strength", "design_claim_mismatch": "study_design",
        "subgroup_significance_fallacy": "statistical_interpretation",
        "unsupported_effect_heterogeneity": "statistical_interpretation",
        "attrition_bias": "missing_data_selection", "informative_missingness": "missing_data_selection",
        "landmark_selection_bias": "missing_data_selection", "post_treatment_adjustment": "estimand_method",
        "estimand_mismatch": "estimand_method", "collider_bias_risk": "missing_data_selection",
        "noninferiority_interpretation_error": "statistical_interpretation",
        "equivalence_fallacy": "statistical_interpretation",
        "competing_risk_misclassification": "estimand_method", "multiplicity_risk": "reproducibility_reporting",
        "selective_outcome_reporting": "reproducibility_reporting",
        "reproducibility_conflict": "reproducibility_reporting",
        "surrogate_to_clinical_overreach": "causal_strength",
    }
    return mapping.get(issue_type, "other")


def _retag_issue(issue: dict[str, Any], paragraph: str, nodes_by_id: dict[str, dict[str, Any]]) -> None:
    related = [nodes_by_id[x] for x in issue.get("node_ids", []) if x in nodes_by_id]
    # Retag from the issue-specific evidence, not from every phrase in the paragraph.
    # The paragraph is used later by deterministic augmentation to add genuinely missing issues.
    text = _combined_text(issue.get("title"), issue.get("explanation"), issue.get("logical_pattern"), *[n.get("plain_meaning") for n in related])
    current = issue.get("issue_type", "other")

    if "not replicated" in text or ("validation cohort" in text and "reproduc" in text):
        current = "reproducibility_conflict"
    elif re.search(r"multiple comparisons|multiplicity|twenty outcomes|20 outcomes|nominal p", text):
        if re.search(r"robust|reproducible|therapeutic effect|only positive|one exploratory", text):
            current = "multiplicity_risk" if "multiple" in text or "twenty" in text or "20 outcomes" in text else current
    elif re.search(r"kaplan.?meier|death[s]? before recurrence|competing", text) and re.search(r"censor", text):
        current = "competing_risk_misclassification"
    elif re.search(r"noninferior|noninferiority", text) and re.search(r"equivalent|equally effective|equivalence", text):
        current = "noninferiority_interpretation_error"
    elif re.search(r"not statistically significant", text) and re.search(r"equivalent|equally effective|equivalence", text):
        current = "equivalence_fallacy"
    elif re.search(r"interaction", text) and re.search(r"men|women|subgroup|sex", text):
        current = "subgroup_significance_fallacy"
    elif re.search(r"similar (?:estimated )?effects|no significant .*interaction", text) and re.search(r"only in|men exclusively|women receive no benefit|heterogeneity", text):
        current = "unsupported_effect_heterogeneity"
    elif re.search(r"discontinued|dropout|completed|completer", text) and re.search(r"excluded|omitted|final analysis", text):
        current = "attrition_bias"
    elif re.search(r"landmark", text) and re.search(r"from (?:the time of )?diagnosis|beginning at diagnosis|preceding year", text):
        current = "time_zero_mismatch"
    elif re.search(r"landmark", text) and re.search(r"surviv|early deaths|excluded", text):
        current = "landmark_selection_bias"
    elif re.search(r"adjusted for|controlled for", text) and re.search(r"after (?:therapy|treatment) began|post-treatment|treatment response", text):
        current = "post_treatment_adjustment"
    elif re.search(r"total effect|conditional|null estimate|direct effect|estimand", text):
        current = "estimand_mismatch"
    elif re.search(r"intensive.?care|icu", text) and re.search(r"selection|included only|restrict|condition", text) and re.search(r"severity", text):
        current = "collider_bias_risk"
    elif re.search(r"modest", text) and re.search(r"large", text):
        current = "magnitude_inflation"
    elif re.search(r"within 24 hours|two weeks later|occurs after|immediate mechanism", text):
        current = "temporal_mechanism_conflict"
    elif re.search(r"not necessary", text) and re.search(r"exclusive|sole pathway|entirely dependent", text):
        current = "necessity_violation"
    elif re.search(r"short-term|four-week|long-term", text):
        current = "temporal_scope_extrapolation"
    elif re.search(r"biomarker|surrogate", text) and re.search(r"therapeutic|clinical benefit|survival|prevents", text):
        current = "surrogate_to_clinical_overreach"
    elif current == "exclusivity_conflict" and re.search(r"men|women|subgroup|all patients|population", text) and not re.search(r"mechanism|pathway|through|mediates", text):
        current = "unsupported_effect_heterogeneity"
    elif current == "scope_overreach" and re.search(r"study design|not designed|equivalence trial", text):
        current = "design_claim_mismatch"
    elif current == "temporal_inversion" and re.search(r"landmark|exposure classification|time zero", text):
        current = "time_zero_mismatch"

    issue["issue_type"] = current
    issue["issue_family"] = _issue_family(current)


def _node_ids_matching(nodes: list[dict[str, Any]], *patterns: str) -> list[str]:
    ids: list[str] = []
    for node in nodes:
        text = _combined_text(node.get("source_text"), node.get("plain_meaning"), node.get("role"), node.get("assertion_type"))
        if any(re.search(pattern, text) for pattern in patterns):
            ids.append(node["id"])
    return ids


def _append_issue_if_missing(
    issues: list[dict[str, Any]], *, issue_type: str, severity: str, title: str,
    node_ids: list[str], explanation: str, logical_pattern: str, suggested_revision: str = "",
) -> None:
    if any(x.get("issue_type") == issue_type for x in issues):
        return
    node_ids = list(dict.fromkeys(node_ids))
    issues.append({
        "id": f"i{len(issues)+1}", "issue_type": issue_type, "severity": severity,
        "title": title, "node_ids": node_ids, "explanation": explanation,
        "logical_pattern": logical_pattern, "suggested_revision": suggested_revision,
        "confidence": 0.9, "issue_family": _issue_family(issue_type),
        "generated_by": "v027_structural_pattern",
    })


def _augment_structural_issues(paragraph: str, nodes: list[dict[str, Any]], issues: list[dict[str, Any]]) -> None:
    text = _combined_text(paragraph)

    if re.search(r"significant.*(?:men|subgroup)", text) and re.search(r"not statistically significant.*(?:women|subgroup)", text) and re.search(r"interaction.*not statistically significant", text):
        ids = _node_ids_matching(nodes, r"men|women|interaction|only")
        _append_issue_if_missing(
            issues, issue_type="subgroup_significance_fallacy", severity="high",
            title="Subgroup-specific significance is mistaken for a subgroup difference", node_ids=ids,
            explanation="Significance in one subgroup and non-significance in another do not establish that subgroup effects differ; the paragraph also reports no significant interaction.",
            logical_pattern="significant in A + nonsignificant in B ≠ significant A–B interaction",
        )

    if re.search(r"discontinued|dropout", text) and re.search(r"excluded|omitted", text) and re.search(r"adverse effects|lack of efficacy|insufficient efficacy", text):
        ids = _node_ids_matching(nodes, r"completed|discontinued|excluded|adverse|efficacy|nearly all")
        _append_issue_if_missing(
            issues, issue_type="attrition_bias", severity="high",
            title="Unfavorable outcomes were selectively removed from the analyzed sample", node_ids=ids,
            explanation="Completer-only evidence is generalized despite selective exclusion of patients with adverse effects or lack of efficacy.",
            logical_pattern="informative dropout + completer analysis → selected outcome estimate",
        )

    if "landmark" in text and re.search(r"from (?:the time of )?diagnosis|beginning at diagnosis", text):
        ids = _node_ids_matching(nodes, r"landmark|diagnosis|exposure|surviv|all patients")
        _append_issue_if_missing(
            issues, issue_type="time_zero_mismatch", severity="high",
            title="Post-landmark exposure is used to support benefit beginning at diagnosis", node_ids=ids,
            explanation="Exposure defined at the landmark cannot establish a treatment effect during the preceding period.",
            logical_pattern="exposure defined at t1 → conclusion asserted from t0",
        )
    if "landmark" in text and re.search(r"survived|died during the first year|early deaths", text):
        ids = _node_ids_matching(nodes, r"surviv|died|excluded|all patients|landmark")
        _append_issue_if_missing(
            issues, issue_type="landmark_selection_bias", severity="high",
            title="Landmark-survivor evidence is generalized to patients excluded before the landmark", node_ids=ids,
            explanation="The analyzed population excludes early deaths, but the conclusion applies to all patients.",
            logical_pattern="survive-to-landmark selection → inference restricted to landmark survivors",
        )

    if re.search(r"adjusted for .*response|controlled for .*response", text) and re.search(r"after (?:therapy|treatment) began|three months after", text) and re.search(r"affected by treatment|treatment .* affected|strongly affected", text):
        ids = _node_ids_matching(nodes, r"adjust|controlled|response|mortality|null|no survival benefit")
        _append_issue_if_missing(
            issues, issue_type="post_treatment_adjustment", severity="high",
            title="The primary model conditions on a treatment-induced post-treatment variable", node_ids=ids,
            explanation="If response lies on the treatment pathway, adjusting for it may remove part of the total effect.",
            logical_pattern="Treatment → mediator → outcome; model adjusts for mediator",
        )
        _append_issue_if_missing(
            issues, issue_type="estimand_mismatch", severity="medium",
            title="A conditional estimate is interpreted as the total treatment effect", node_ids=ids,
            explanation="The response-adjusted estimate targets a different estimand from an unqualified claim of no total survival benefit.",
            logical_pattern="conditional/direct effect estimate ≠ total effect estimate",
        )

    if re.search(r"included only .*intensive care|only patients admitted to the intensive care|icu", text) and re.search(r"both .* increased .* admission|biomarker.*admission", text) and re.search(r"severe .* mortality|severity .* mortality", text):
        ids = _node_ids_matching(nodes, r"intensive|icu|biomarker|severe|mortality")
        _append_issue_if_missing(
            issues, issue_type="collider_bias_risk", severity="medium",
            title="Conditioning on intensive-care admission creates a collider structure", node_ids=ids,
            explanation="Admission is affected by both the biomarker and disease severity, while severity predicts mortality; restricting to admitted patients may induce a spurious biomarker–mortality association.",
            logical_pattern="Biomarker → selection ← severity → outcome; analysis conditions on selection",
        )

    if re.search(r"noninferiority", text) and re.search(r"equally effective|equivalence|equivalent", text):
        ids = _node_ids_matching(nodes, r"noninferior|confidence interval|equivalence|equal|not statistically significant")
        _append_issue_if_missing(
            issues, issue_type="noninferiority_interpretation_error", severity="high",
            title="Failure to establish noninferiority is converted into an equality conclusion", node_ids=ids,
            explanation="The confidence interval permits unacceptable inferiority and the study was not an equivalence trial, so equal effectiveness is not established.",
            logical_pattern="noninferiority not shown + nonsignificance ≠ equivalence",
        )

    if re.search(r"deaths? before recurrence", text) and re.search(r"censor", text) and re.search(r"kaplan.?meier", text):
        ids = _node_ids_matching(nodes, r"recurrence|death|censor|kaplan|hazard|prevents")
        _append_issue_if_missing(
            issues, issue_type="competing_risk_misclassification", severity="high",
            title="Deaths before recurrence are treated as ordinary censoring", node_ids=ids,
            explanation="Death prevents subsequent recurrence and is a competing event; a Kaplan–Meier analysis that censors such deaths does not directly estimate the competing-risk cumulative incidence.",
            logical_pattern="competing event censored as noninformative loss → estimand mismatch",
        )

    if re.search(r"(?:twenty|20) .*outcomes", text) and re.search(r"without .*multiple comparisons|not adjust.*multiple comparisons", text):
        ids = _node_ids_matching(nodes, r"20|twenty|multiple|nominal|primary endpoint|robust|reproduc")
        _append_issue_if_missing(
            issues, issue_type="multiplicity_risk", severity="high",
            title="A nominal result is interpreted without accounting for multiplicity", node_ids=ids,
            explanation="Many outcomes were tested without a prespecified primary endpoint or multiplicity adjustment, so one nominal p value does not by itself establish a robust effect.",
            logical_pattern="many unadjusted tests + isolated nominal p value → elevated false-positive risk",
        )
    if re.search(r"not replicated|failed .*validation|did not recur in the validation", text) and re.search(r"reproducible", text):
        ids = _node_ids_matching(nodes, r"replicat|validation|reproduc")
        _append_issue_if_missing(
            issues, issue_type="reproducibility_conflict", severity="high",
            title="The reproducibility claim conflicts with failed validation", node_ids=ids,
            explanation="The paragraph says the positive finding did not replicate and then describes the effect as reproducible.",
            logical_pattern="failed replication + reproducible conclusion",
        )
    if re.search(r"exploratory biomarker", text) and re.search(r"therapeutic effect|clinical benefit", text):
        ids = _node_ids_matching(nodes, r"biomarker|therapeutic|clinical|robust")
        _append_issue_if_missing(
            issues, issue_type="surrogate_to_clinical_overreach", severity="high",
            title="An exploratory biomarker signal is generalized to therapeutic benefit", node_ids=ids,
            explanation="The sole positive result is a biomarker, while the conclusion claims a broad therapeutic effect.",
            logical_pattern="surrogate/biomarker result → broad clinical therapeutic conclusion",
        )


def _validate_issue_structure(issue: dict[str, Any], nodes_by_id: dict[str, dict[str, Any]], edges: list[dict[str, Any]], paragraph: str) -> tuple[str, list[str]]:
    issue_nodes = [nodes_by_id[x] for x in issue.get("node_ids", []) if x in nodes_by_id]
    node_ids = {x["id"] for x in issue_nodes}
    issue_edges = [e for e in edges if e.get("source") in node_ids and e.get("target") in node_ids]
    relations = {e.get("relation") for e in issue_edges}
    kinds = {n.get("assertion_type") for n in issue_nodes}
    certainties = {n.get("certainty") for n in issue_nodes}
    text = _combined_text(paragraph, issue.get("title"), issue.get("explanation"), issue.get("logical_pattern"), *[n.get("plain_meaning") for n in issue_nodes])
    reasons: list[str] = []
    kind = issue.get("issue_type")

    formal_types = {"direct_contradiction", "semantic_contradiction", "necessity_violation", "exclusivity_conflict", "reproducibility_conflict"}
    unsupported_types = {
        "magnitude_inflation", "causal_overclaim", "temporal_mechanism_conflict", "temporal_scope_extrapolation",
        "scope_overreach", "evidence_strength_mismatch", "unsupported_generalization", "unsupported_mechanism",
        "design_claim_mismatch", "subgroup_significance_fallacy", "unsupported_effect_heterogeneity",
        "noninferiority_interpretation_error", "equivalence_fallacy", "surrogate_to_clinical_overreach",
    }
    method_types = {
        "time_zero_mismatch", "attrition_bias", "informative_missingness", "landmark_selection_bias",
        "post_treatment_adjustment", "estimand_mismatch", "collider_bias_risk",
        "competing_risk_misclassification", "multiplicity_risk", "selective_outcome_reporting",
    }

    if kind in formal_types:
        if "contradicts" in relations:
            reasons.append("관련 Node 사이에 직접적인 contradicts Edge가 있습니다.")
        if kind == "necessity_violation" and re.search(r"not necessary|without .*response", text) and re.search(r"exclusive|sole pathway|entirely dependent", text):
            reasons.append("필요하지 않다는 근거와 배타적/필수 경로 결론이 함께 존재합니다.")
        if kind == "reproducibility_conflict" and re.search(r"not replicated|failed .*validation|did not recur", text) and "reproduc" in text:
            reasons.append("재현 실패와 reproducible 결론이 같은 문단에 명시되어 있습니다.")
        if reasons:
            return "formal_conflict", reasons

    if kind in method_types:
        if kind == "attrition_bias" and re.search(r"discontinued|dropout", text) and re.search(r"excluded|omitted", text):
            reasons.append("탈락자 제외와 불리한 탈락 사유가 함께 명시되어 있습니다.")
        elif kind == "landmark_selection_bias" and "landmark" in text and re.search(r"surviv|early death|excluded", text):
            reasons.append("landmark 생존 조건과 분석 제외가 명시되어 있습니다.")
        elif kind == "time_zero_mismatch" and "landmark" in text and re.search(r"from .*diagnosis|beginning at diagnosis", text):
            reasons.append("노출 정의 시점과 결론의 추적 시작점이 다릅니다.")
        elif kind == "post_treatment_adjustment" and re.search(r"adjusted|controlled", text) and re.search(r"after treatment|post-treatment|response", text):
            reasons.append("치료 후 변수에 대한 보정이 명시되어 있습니다.")
        elif kind == "estimand_mismatch" and re.search(r"conditional|adjusted|null estimate|total effect", text):
            reasons.append("조건부 추정치와 전체효과 결론의 차이가 명시되어 있습니다.")
        elif kind == "collider_bias_risk" and re.search(r"included only|restrict|condition", text) and re.search(r"admission|selection", text):
            reasons.append("공통 결과인 선택 변수에 조건을 거는 구조가 있습니다.")
        elif kind == "competing_risk_misclassification" and re.search(r"death.*before recurrence", text) and re.search(r"censor", text):
            reasons.append("재발 전 사망을 censoring으로 처리한 구조가 명시되어 있습니다.")
        elif kind == "multiplicity_risk" and re.search(r"multiple comparisons|many outcomes|twenty|20 outcomes", text):
            reasons.append("다수 결과와 다중비교 미보정이 명시되어 있습니다.")
        if reasons:
            return "structural_methodological_risk", reasons

    if kind in unsupported_types or kind in {"temporal_inversion", "unsupported_generalization"}:
        if kind == "subgroup_significance_fallacy" and "interaction" in text:
            reasons.append("하위집단별 유의성과 상호작용 검정이 구분되어 있습니다.")
        elif kind in {"noninferiority_interpretation_error", "equivalence_fallacy"} and re.search(r"noninferior|equivalence|equally effective", text):
            reasons.append("비열등성/동등성 설계와 결론의 불일치가 문단에 명시되어 있습니다.")
        elif kind == "magnitude_inflation" and re.search(r"modest", text) and re.search(r"large", text):
            reasons.append("동일 효과가 modest와 large로 다르게 표현됩니다.")
        elif kind == "temporal_mechanism_conflict" and re.search(r"before|after|later|within", text) and re.search(r"immediate mechanism|caus", text):
            reasons.append("관찰된 시간 순서와 즉각적 기전 결론이 연결되어 있습니다.")
        elif kind == "causal_overclaim" and (certainties & {"establishes", "proves"} or re.search(r"protects|prevents|causes", text)) and ("association" in kinds or "associated" in text):
            reasons.append("연관성 근거와 확정적 인과 결론이 함께 있습니다.")
        elif kind in {"scope_overreach", "unsupported_generalization"}:
            scopes = {_norm_key(n.get("population_scope", "")) for n in issue_nodes if _norm_key(n.get("population_scope", "")) not in {"", "not_specified"}}
            if len(scopes) >= 2 or re.search(r"subset|completer|icu|survivor|subgroup", text) and re.search(r"all patients|entire population|nearly all", text):
                reasons.append("근거 집단과 결론 집단의 범위가 다릅니다.")
        elif kind == "evidence_strength_mismatch" and (certainties & {"establishes", "proves"} or re.search(r"definitive|robust", text)):
            reasons.append("제한적·탐색적 근거보다 강한 결론 표현이 사용됩니다.")
        elif kind == "surrogate_to_clinical_overreach" and re.search(r"biomarker|surrogate", text) and re.search(r"therapeutic|clinical benefit", text):
            reasons.append("바이오마커 결과와 광범위한 임상효과 결론이 연결되어 있습니다.")
        if reasons:
            return "rule_confirmed_unsupported", reasons

    if issue_edges:
        reasons.append("관련 Node 사이에 구조적 관계가 존재하지만 현재 규칙만으로 결론을 확정하지는 못합니다.")
    return "model_suggested_concern", reasons


def _group_issues(issues: list[dict[str, Any]], nodes_by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    conclusion_ids = {n["id"] for n in nodes_by_id.values() if n.get("role") == "conclusion"}
    for issue in issues:
        shared_conclusions = sorted(set(issue.get("node_ids", [])) & conclusion_ids)
        anchor = shared_conclusions[0] if shared_conclusions else (issue.get("node_ids") or [issue["id"]])[0]
        key = (issue.get("issue_family", "other"), anchor)
        grouped.setdefault(key, []).append(issue)

    rank = {"high": 3, "medium": 2, "low": 1}
    validation_rank = {
        "formal_conflict": 4, "rule_confirmed_unsupported": 3,
        "structural_methodological_risk": 2, "model_suggested_concern": 1,
    }
    rows: list[dict[str, Any]] = []
    for index, ((family, anchor), members) in enumerate(grouped.items(), 1):
        strongest = max(members, key=lambda x: rank.get(x.get("severity", "low"), 0))
        validation = max(members, key=lambda x: validation_rank.get(x.get("verification_level", "model_suggested_concern"), 0)).get("verification_level")
        rows.append({
            "id": f"g{index}",
            "family": family,
            "anchor_node_id": anchor,
            "severity": strongest.get("severity", "medium"),
            "title": strongest.get("title", family),
            "summary": " ".join(dict.fromkeys(_norm_space(x.get("explanation", "")) for x in members if x.get("explanation")))[:700],
            "issue_ids": [x["id"] for x in members],
            "verification_level": validation,
            "sub_findings": [
                {"issue_id": x["id"], "issue_type": x.get("issue_type"), "title": x.get("title"), "severity": x.get("severity")}
                for x in members
            ],
        })
    return rows


def _recalculate_assessment(issues: list[dict[str, Any]]) -> str:
    levels = {x.get("verification_level") for x in issues}
    if "formal_conflict" in levels and len(levels - {"formal_conflict"}) > 0:
        return "mixed_concerns"
    if "formal_conflict" in levels:
        return "formal_conflict"
    if "structural_methodological_risk" in levels and "rule_confirmed_unsupported" in levels:
        return "mixed_concerns"
    if "structural_methodological_risk" in levels:
        return "methodological_risk"
    if "rule_confirmed_unsupported" in levels:
        return "unsupported_conclusion"
    if issues:
        return "potential_issue"
    return "internally_consistent"


def normalize_discussion_output(parsed: DiscussionGraphOutput | dict[str, Any], *, input_text: str = "") -> dict[str, Any]:
    data = parsed if isinstance(parsed, DiscussionGraphOutput) else DiscussionGraphOutput.model_validate(parsed)
    paragraph = _norm_space(input_text)
    node_map: dict[str, str] = {}
    nodes: list[dict[str, Any]] = []
    warnings: list[str] = []

    for index, node in enumerate(data.nodes, 1):
        new_id = f"d{index}"
        old_id = str(node.id or new_id)
        node_map[old_id] = new_id
        row = node.model_dump()
        row["id"] = new_id
        row["source_text"] = _norm_space(row.get("source_text", ""))
        row["plain_meaning"] = _norm_space(row.get("plain_meaning", ""))
        row["normalized_claim"] = row["plain_meaning"]
        _reclassify_node(row)

        status, matched, similarity = _best_source_match(row["source_text"], paragraph) if paragraph else ("not_checked", row["source_text"], 1.0)
        row["source_fidelity_status"] = status
        row["matched_source_span"] = matched
        row["source_similarity"] = round(similarity, 4)
        row["source_span_exact"] = status == "exact"
        if status in {"paraphrased", "partial", "unmatched"}:
            warnings.append(f"{new_id}: source_text가 원문 직접 인용이 아닐 수 있습니다 ({status}, similarity={similarity:.2f}).")
        source_for_numbers = matched if matched else row["source_text"]
        row["numeric_mentions"] = list(dict.fromkeys((row.get("numeric_mentions") or []) + _extract_numeric_mentions(source_for_numbers)))
        row["inferred_details"] = [str(x).strip() for x in row.get("inferred_details", []) if str(x).strip()]
        nodes.append(row)

    known_node_ids = {n["id"] for n in nodes}
    edges: list[dict[str, Any]] = []
    for index, edge in enumerate(data.edges, 1):
        row = edge.model_dump()
        row["id"] = f"e{index}"
        row["source"] = node_map.get(row["source"], row["source"])
        row["target"] = node_map.get(row["target"], row["target"])
        if row["source"] not in known_node_ids or row["target"] not in known_node_ids:
            warnings.append(f"Dropped {row['id']} because it referenced an unknown node")
            continue
        edges.append(row)

    issues: list[dict[str, Any]] = []
    for index, issue in enumerate(data.issues, 1):
        row = issue.model_dump()
        row["id"] = f"i{index}"
        row["node_ids"] = [node_map.get(x, x) for x in row["node_ids"] if node_map.get(x, x) in known_node_ids]
        row["generated_by"] = "model"
        issues.append(row)

    nodes_by_id = {node["id"]: node for node in nodes}
    for issue in issues:
        _retag_issue(issue, paragraph, nodes_by_id)
    _augment_structural_issues(paragraph, nodes, issues)

    # Reassign stable ids after deterministic augmentation.
    for index, issue in enumerate(issues, 1):
        issue["id"] = f"i{index}"
        issue.setdefault("issue_family", _issue_family(issue.get("issue_type", "other")))
        verification_level, validation_reasons = _validate_issue_structure(issue, nodes_by_id, edges, paragraph)
        issue["verification_level"] = verification_level
        issue["validation_reasons"] = validation_reasons
        # Backward-compatible field with clearer values in v027.
        issue["validation_status"] = verification_level

    issue_node_ids = {node_id for issue in issues for node_id in issue.get("node_ids", [])}
    for node in nodes:
        node["has_issue"] = node["id"] in issue_node_ids
        node["issue_ids"] = [issue["id"] for issue in issues if node["id"] in issue.get("node_ids", [])]
        node["display_status"] = "issue" if node["has_issue"] else node["role"]

    issue_groups = _group_issues(issues, nodes_by_id)
    counts: dict[str, int] = {}
    family_counts: dict[str, int] = {}
    verification_counts: dict[str, int] = {}
    for issue in issues:
        counts[issue["issue_type"]] = counts.get(issue["issue_type"], 0) + 1
        family = issue.get("issue_family", "other")
        family_counts[family] = family_counts.get(family, 0) + 1
        level = issue.get("verification_level", "model_suggested_concern")
        verification_counts[level] = verification_counts.get(level, 0) + 1

    assessment = _recalculate_assessment(issues)
    fidelity_warning_count = sum(1 for n in nodes if n.get("source_fidelity_status") in {"paraphrased", "partial", "unmatched"})
    return {
        "schema_version": "0.27.0",
        "paragraph_summary": data.paragraph_summary,
        "model_overall_assessment": data.overall_assessment,
        "overall_assessment": assessment,
        "nodes": nodes,
        "edges": edges,
        "issues": issues,
        "issue_groups": issue_groups,
        "warnings": warnings,
        "summary": {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "issue_count": len(issues),
            "issue_group_count": len(issue_groups),
            "high_severity_count": sum(1 for x in issues if x["severity"] == "high"),
            "issue_type_counts": counts,
            "issue_family_counts": family_counts,
            "verification_level_counts": verification_counts,
            "formal_conflict_count": verification_counts.get("formal_conflict", 0),
            "rule_confirmed_unsupported_count": verification_counts.get("rule_confirmed_unsupported", 0),
            "structural_methodological_risk_count": verification_counts.get("structural_methodological_risk", 0),
            "model_suggested_concern_count": verification_counts.get("model_suggested_concern", 0),
            "source_fidelity_warning_count": fidelity_warning_count,
            # Backward compatibility:
            "structurally_supported_issue_count": sum(1 for x in issues if x.get("verification_level") != "model_suggested_concern"),
            "model_flag_only_count": verification_counts.get("model_suggested_concern", 0),
        },
        "graph_metrics": calculate_graph_metrics(nodes, edges, issues=issues),
    }


def analyze_structured_discussion(payload: dict[str, Any]) -> dict[str, Any]:
    """Offline entry point used by tests and imports of pre-extracted graph JSON."""
    structured = payload.get("structured_output", payload)
    input_text = str(payload.get("input_text") or payload.get("text") or "")
    return normalize_discussion_output(structured, input_text=input_text)



def _env_int(name: str, default: int, *, minimum: int | None = None) -> int:
    raw = str(os.getenv(name, "") or "").strip()
    value = default if not raw else int(raw)
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _split_oversized_segment(segment: str, max_chars: int) -> list[str]:
    """Split one oversized paragraph without silently dropping any text."""
    sentences = [x.strip() for x in re.split(r"(?<=[.!?])\s+", segment.strip()) if x.strip()]
    if not sentences:
        sentences = [segment.strip()]
    pieces: list[str] = []
    current = ""
    for sentence in sentences:
        if len(sentence) <= max_chars:
            candidate = sentence if not current else current + " " + sentence
            if len(candidate) <= max_chars:
                current = candidate
                continue
            if current:
                pieces.append(current)
            current = sentence
            continue
        if current:
            pieces.append(current)
            current = ""
        remaining = sentence
        while len(remaining) > max_chars:
            cut = remaining.rfind(" ", 0, max_chars + 1)
            if cut < max_chars // 2:
                cut = max_chars
            pieces.append(remaining[:cut].strip())
            remaining = remaining[cut:].strip()
        if remaining:
            current = remaining
    if current:
        pieces.append(current)
    return [x for x in pieces if x]


def split_discussion_text(text: str, *, max_chars: int | None = None) -> list[str]:
    """Create sentence/paragraph-aware chunks. There is no default total-input cap."""
    text = str(text or "").strip()
    if not text:
        return []
    limit = int(max_chars or _env_int("DISCUSSION_CHUNK_CHARS", DEFAULT_DISCUSSION_CHUNK_CHARS, minimum=MIN_DISCUSSION_CHUNK_CHARS))
    if limit < MIN_DISCUSSION_CHUNK_CHARS:
        raise ValueError(f"Discussion chunk size must be at least {MIN_DISCUSSION_CHUNK_CHARS} characters")

    paragraphs = [x.strip() for x in re.split(r"\n\s*\n+", text) if x.strip()]
    if not paragraphs:
        paragraphs = [text]
    segments: list[str] = []
    for paragraph in paragraphs:
        if len(paragraph) <= limit:
            segments.append(paragraph)
        else:
            segments.extend(_split_oversized_segment(paragraph, limit))

    chunks: list[str] = []
    current = ""
    for segment in segments:
        candidate = segment if not current else current + "\n\n" + segment
        if len(candidate) <= limit:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = segment
    if current:
        chunks.append(current)
    return chunks


def _call_discussion_model(
    text: str, *, model: str, reasoning_effort: str, max_output_tokens: int,
    custom_instruction: str, client: Any,
) -> tuple[DiscussionGraphOutput, Any, float]:
    prompt = build_discussion_prompt(text, custom_instruction)
    started = time.perf_counter()
    response = client.responses.parse(
        model=model,
        reasoning={"effort": reasoning_effort},
        max_output_tokens=max_output_tokens,
        store=False,
        input=[
            {"role": "system", "content": prompt["system"]},
            {"role": "user", "content": prompt["user"]},
        ],
        text_format=DiscussionGraphOutput,
    )
    latency_ms = round((time.perf_counter() - started) * 1000, 3)
    return _parsed_from_response(response), response, latency_ms


def _merge_parsed_chunks(parsed_chunks: list[DiscussionGraphOutput], chunks: list[str]) -> DiscussionGraphOutput:
    nodes: list[DiscussionNode] = []
    edges: list[DiscussionEdge] = []
    issues: list[DiscussionIssue] = []
    summaries: list[str] = []
    sentence_offset = 0
    for chunk_index, (parsed, chunk_text) in enumerate(zip(parsed_chunks, chunks), 1):
        summaries.append(parsed.paragraph_summary)
        node_map: dict[str, str] = {}
        for local_index, node in enumerate(parsed.nodes, 1):
            new_id = f"c{chunk_index}_d{local_index}"
            node_map[node.id] = new_id
            row = node.model_copy(deep=True)
            row.id = new_id
            row.sentence_index = sentence_offset + max(1, int(node.sentence_index))
            nodes.append(row)
        for local_index, edge in enumerate(parsed.edges, 1):
            if edge.source not in node_map or edge.target not in node_map:
                continue
            row = edge.model_copy(deep=True)
            row.id = f"c{chunk_index}_e{local_index}"
            row.source = node_map[edge.source]
            row.target = node_map[edge.target]
            edges.append(row)
        for local_index, issue in enumerate(parsed.issues, 1):
            row = issue.model_copy(deep=True)
            row.id = f"c{chunk_index}_i{local_index}"
            row.node_ids = [node_map[x] for x in issue.node_ids if x in node_map]
            issues.append(row)
        sentence_offset += max(1, len(_sentences(chunk_text)))

    # The deterministic whole-document postprocessor runs after this merge and can
    # add cross-chunk issue patterns. LLM-extracted edges remain provenance-safe
    # within their source chunk rather than inventing cross-chunk links.
    return DiscussionGraphOutput(
        paragraph_summary=" ".join(dict.fromkeys(_norm_space(x) for x in summaries if _norm_space(x)))[:6000],
        nodes=nodes,
        edges=edges,
        issues=issues,
        overall_assessment="potential_issue",
    )


def _sum_usage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for row in rows:
        for key, value in row.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                result[key] = result.get(key, 0) + value
    return result


def generate_discussion_graph(
    text: str,
    *,
    model: str | None = None,
    reasoning_effort: str = "low",
    max_output_tokens: int = 7000,
    custom_instruction: str = "",
    client: Any = None,
    architecture_mode: str | None = None,
) -> dict[str, Any]:
    _load_local_env()
    text = str(text or "")
    if not text.strip():
        raise ValueError("분석할 Discussion 문단을 입력하세요.")

    # Default 0 means no application-level total character limit. Very long
    # documents are analyzed in sentence/paragraph-aware chunks instead.
    configured_total_limit = _env_int("DISCUSSION_MAX_INPUT_CHARS", 0, minimum=0)
    if configured_total_limit and len(text) > configured_total_limit:
        raise ValueError(
            f"입력이 관리자가 설정한 한도({configured_total_limit:,}자)를 초과했습니다. "
            "DISCUSSION_MAX_INPUT_CHARS=0이면 애플리케이션 글자 수 제한이 해제됩니다."
        )

    model = str(model or os.getenv("OPENAI_MODEL") or DEFAULT_MODEL).strip()
    if not re.fullmatch(r"[A-Za-z0-9_.:\-]+", model):
        raise ValueError("model contains unsupported characters")
    reasoning_effort = str(reasoning_effort or "low").lower().strip()
    if reasoning_effort not in ALLOWED_REASONING_EFFORTS:
        raise ValueError("reasoning_effort must be low, medium, or high")
    max_output_tokens = int(max_output_tokens)
    if max_output_tokens < 1000 or max_output_tokens > 30000:
        raise ValueError("max_output_tokens must be between 1000 and 30000")

    if client is None:
        if not os.getenv("OPENAI_API_KEY", "").strip():
            raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다. .env를 확인하고 서버를 다시 시작하세요.")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ValueError("openai Python package가 없습니다. RUN_WINDOWS.bat를 다시 실행하세요.") from exc
        client = OpenAI()

    chunk_chars = _env_int("DISCUSSION_CHUNK_CHARS", DEFAULT_DISCUSSION_CHUNK_CHARS, minimum=MIN_DISCUSSION_CHUNK_CHARS)
    chunks = split_discussion_text(text, max_chars=chunk_chars)
    configured_max_chunks = _env_int("DISCUSSION_MAX_CHUNKS", 0, minimum=0)
    if configured_max_chunks and len(chunks) > configured_max_chunks:
        raise ValueError(
            f"입력이 {len(chunks)}개 chunk로 분할되어 관리자가 설정한 최대 {configured_max_chunks}개를 초과했습니다. "
            "DISCUSSION_MAX_CHUNKS=0이면 chunk 수 제한이 해제됩니다."
        )

    architecture_mode = str(
        architecture_mode or os.getenv("DISCUSSION_ARCHITECTURE") or "graph_native_6agents_balanced"
    ).strip().lower()
    if architecture_mode not in {
        "graph_native_6agents_balanced", "graph_native_6agents_pilot", "graph_native_5agents_pilot",
        "graph_native_multi_agent", "legacy_single_pass",
    }:
        raise ValueError(
            "architecture_mode must be graph_native_6agents_balanced, graph_native_6agents_pilot, "
            "graph_native_5agents_pilot, graph_native_multi_agent, or legacy_single_pass"
        )
    judge_enabled = str(os.getenv("DISCUSSION_JUDGE_ENABLED", "1")).strip().lower() not in {"0", "false", "no", "off"}

    parsed_chunks: list[DiscussionGraphOutput] = []
    responses: list[Any] = []
    latencies: list[float] = []
    chunk_response_groups: list[list[Any]] = []
    architecture_warnings: list[str] = []
    architecture_traces: list[dict[str, Any]] = []
    for index, chunk in enumerate(chunks, 1):
        chunk_instruction = custom_instruction
        if len(chunks) > 1:
            prefix = (
                f"This is document chunk {index} of {len(chunks)}. Analyze only claims stated in this chunk. "
                "Do not assume content from omitted chunks."
            )
            chunk_instruction = prefix + (" " + custom_instruction.strip() if custom_instruction.strip() else "")

        if architecture_mode == "legacy_single_pass":
            parsed, response, latency_ms = _call_discussion_model(
                chunk,
                model=model,
                reasoning_effort=reasoning_effort,
                max_output_tokens=max_output_tokens,
                custom_instruction=chunk_instruction,
                client=client,
            )
            stage_responses = [response]
        else:
            try:
                # Lazy imports avoid circular imports while preserving the
                # long-standing public DiscussionGraphOutput schema.
                if architecture_mode in {"graph_native_6agents_balanced", "graph_native_6agents_pilot"}:
                    from .discussion_architecture_6agents import run_graph_native_6agents_chunk

                    parsed, stage_responses, latency_ms, _private_trace = run_graph_native_6agents_chunk(
                        chunk,
                        model=model,
                        reasoning_effort=reasoning_effort,
                        max_output_tokens=max_output_tokens,
                        custom_instruction=chunk_instruction,
                        client=client,
                        judge_enabled=judge_enabled,
                    )
                elif architecture_mode == "graph_native_5agents_pilot":
                    from .discussion_architecture_5agents import run_graph_native_5agents_chunk

                    parsed, stage_responses, latency_ms, _private_trace = run_graph_native_5agents_chunk(
                        chunk,
                        model=model,
                        reasoning_effort=reasoning_effort,
                        max_output_tokens=max_output_tokens,
                        custom_instruction=chunk_instruction,
                        client=client,
                        judge_enabled=judge_enabled,
                    )
                else:
                    from .discussion_architecture import run_graph_native_chunk

                    parsed, stage_responses, latency_ms, _private_trace = run_graph_native_chunk(
                        chunk,
                        model=model,
                        reasoning_effort=reasoning_effort,
                        max_output_tokens=max_output_tokens,
                        custom_instruction=chunk_instruction,
                        client=client,
                        judge_enabled=judge_enabled,
                    )
                architecture_traces.append(_private_trace)
            except Exception as exc:
                # Compatibility-first fallback: an internal architecture-stage
                # failure must not break Discussion Lab or its public output.
                architecture_warnings.append(
                    f"Chunk {index}: graph-native architecture failed and used legacy fallback "
                    f"({type(exc).__name__}: {exc})."
                )
                parsed, response, latency_ms = _call_discussion_model(
                    chunk,
                    model=model,
                    reasoning_effort=reasoning_effort,
                    max_output_tokens=max_output_tokens,
                    custom_instruction=chunk_instruction,
                    client=client,
                )
                stage_responses = [response]

        parsed_chunks.append(parsed)
        chunk_response_groups.append(stage_responses)
        responses.extend(stage_responses)
        latencies.append(latency_ms)

    parsed_document = parsed_chunks[0] if len(parsed_chunks) == 1 else _merge_parsed_chunks(parsed_chunks, chunks)
    document_compaction: dict[str, Any] | None = None
    if architecture_mode in {"graph_native_6agents_balanced", "graph_native_6agents_pilot"}:
        from .discussion_architecture_6agents import (
            DEFAULT_DOCUMENT_EDGE_CAP, DEFAULT_DOCUMENT_NODE_CAP,
            DEFAULT_NODE_DEDUP_THRESHOLD, compact_document_output,
        )
        document_node_cap = _env_int(
            "DISCUSSION_MAX_DOCUMENT_NODES", DEFAULT_DOCUMENT_NODE_CAP, minimum=16
        )
        document_edge_cap = _env_int(
            "DISCUSSION_MAX_DOCUMENT_EDGES", DEFAULT_DOCUMENT_EDGE_CAP, minimum=16
        )
        raw_threshold = str(os.getenv("DISCUSSION_NODE_DEDUP_THRESHOLD", "") or "").strip()
        dedup_threshold = DEFAULT_NODE_DEDUP_THRESHOLD if not raw_threshold else float(raw_threshold)
        dedup_threshold = max(0.85, min(1.0, dedup_threshold))
        parsed_document, document_compaction = compact_document_output(
            parsed_document,
            node_cap=document_node_cap,
            edge_cap=document_edge_cap,
            dedup_threshold=dedup_threshold,
        )
        if document_compaction.get("duplicate_node_count", 0):
            architecture_warnings.append(
                f"중복 의미 Node {document_compaction['duplicate_node_count']}개를 canonical Node로 병합했습니다."
            )
        if document_compaction.get("pruned_node_count", 0):
            architecture_warnings.append(
                f"긴 문서 Graph 안정성을 위해 중요도가 낮은 Node {document_compaction['pruned_node_count']}개를 생략했습니다."
            )
    normalized = normalize_discussion_output(parsed_document, input_text=text)
    input_hash = hashlib.sha256(_norm_space(text).encode("utf-8")).hexdigest()[:12]
    usages = [_usage_dict(response) for response in responses]
    chunk_usages = [_sum_usage([_usage_dict(response) for response in group]) for group in chunk_response_groups]
    returned_models = list(dict.fromkeys(str(getattr(response, "model", model)) for response in responses))
    response_ids = [str(getattr(response, "id", "")) for response in responses]
    normalized.update({
        "provider": "openai",
        "api": "Responses API",
        "model_requested": model,
        "model_returned": returned_models[0] if len(returned_models) == 1 else returned_models,
        "response_id": response_ids[0] if len(response_ids) == 1 else response_ids,
        "reasoning_effort": reasoning_effort,
        "latency_ms": round(sum(latencies), 3),
        "chunk_latencies_ms": latencies,
        "usage": _sum_usage(usages),
        "chunk_usages": chunk_usages,
        "input_text": text,
        "input_hash": input_hash,
        "input_preview": _norm_space(text)[:180],
        "input_char_count": len(text),
        "analysis_mode": "single_pass" if len(chunks) == 1 else "auto_chunked",
        "chunk_count": len(chunks),
        "chunk_char_limit": chunk_chars,
        "chunk_char_counts": [len(x) for x in chunks],
        "api_call_count": len(responses),
        "application_input_limit": configured_total_limit or None,
        "verification_engine": "typed_graph_structural_rules",
        "z3_used": False,
        "prompt_version": (
            "discussion_graph_native_6agents_balanced_compiler_v045"
            if architecture_mode in {"graph_native_6agents_balanced", "graph_native_6agents_pilot"}
            else (
                "discussion_graph_native_5agents_pilot_v1"
                if architecture_mode == "graph_native_5agents_pilot"
                else (
                    "discussion_graph_native_multi_agent_v1"
                    if architecture_mode == "graph_native_multi_agent"
                    else "discussion_internal_logic_graph_v4_unbounded_chunking"
                )
            )
        ),
        "discussion_compiler": (
            "v043-derived-balanced-complete-proposition-local-repair"
            if architecture_mode in {"graph_native_6agents_balanced", "graph_native_6agents_pilot"}
            else "legacy"
        ),
        "discussion_agent_pipeline": (
            ["compiler", "evidence", "logic", "target", "assumption_conditional", "judge_conditional"]
            if architecture_mode in {"graph_native_6agents_balanced", "graph_native_6agents_pilot"}
            else []
        ),
        "architecture_traces": architecture_traces,
    })
    if architecture_warnings:
        normalized.setdefault("warnings", []).extend(architecture_warnings)
    if len(chunks) > 1:
        normalized.setdefault("warnings", []).append(
            "긴 문서는 자동 분할되어 분석되었습니다. LLM Edge는 각 chunk 내부 출처에만 연결하고, "
            "문서 전체 deterministic pattern 검사가 병합 후 다시 실행됩니다."
        )
    return normalized

def run_discussion_lab(payload: dict[str, Any], *, output_root: Path, client: Any = None) -> dict[str, Any]:
    result = generate_discussion_graph(
        str(payload.get("text") or payload.get("paragraph") or ""),
        model=payload.get("model"),
        reasoning_effort=str(payload.get("reasoning_effort") or "low"),
        max_output_tokens=int(payload.get("max_output_tokens") or 7000),
        custom_instruction=str(payload.get("custom_instruction") or ""),
        client=client,
        architecture_mode=(str(payload.get("architecture_mode")) if payload.get("architecture_mode") else None),
    )
    run_id = f"discussion_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    result["run_id"] = run_id
    (run_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def list_discussion_runs(output_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not output_root.exists():
        return rows
    for path in sorted(output_root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        result_path = path / "result.json"
        if not path.is_dir() or not result_path.exists():
            continue
        try:
            data = json.loads(result_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        rows.append({
            "run_id": path.name,
            "paragraph_summary": data.get("paragraph_summary"),
            "input_preview": data.get("input_preview") or _norm_space(data.get("input_text", ""))[:180],
            "input_hash": data.get("input_hash", ""),
            "overall_assessment": data.get("overall_assessment"),
            "issue_count": (data.get("summary") or {}).get("issue_count", 0),
            "issue_group_count": (data.get("summary") or {}).get("issue_group_count", 0),
            "high_severity_count": (data.get("summary") or {}).get("high_severity_count", 0),
            "input_char_count": data.get("input_char_count", len(str(data.get("input_text") or ""))),
            "analysis_mode": data.get("analysis_mode", "single_pass"),
            "chunk_count": data.get("chunk_count", 1),
            "z3_used": bool(data.get("z3_used", False)),
        })
    return rows


def load_discussion_run(output_root: Path, run_id: str) -> dict[str, Any]:
    path = (output_root / run_id / "result.json").resolve()
    root = output_root.resolve()
    if root not in path.parents or not path.exists():
        raise ValueError("Unknown Discussion Lab run")
    return json.loads(path.read_text(encoding="utf-8"))


def discussion_sample() -> str:
    return (
        "Treatment G reduced inflammatory activity and was followed by slower fibrosis progression. "
        "The antifibrotic effect was observed equally in patients with and without a measurable inflammatory response, "
        "suggesting that inflammation reduction was not necessary for benefit. Nevertheless, because inflammation declined "
        "before fibrosis improved, the results prove that Treatment G prevents fibrosis exclusively through suppression of inflammation."
    )

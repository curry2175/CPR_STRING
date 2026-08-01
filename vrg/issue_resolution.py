from __future__ import annotations

import copy
import json
import re
from typing import Any

from .graph_metrics import calculate_graph_metrics


SEMANTICS_VERSION = "v031_active_acknowledged_resolved"


def _norm(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _sentences(text: str) -> list[str]:
    return [x.strip() for x in re.split(r"(?<=[.!?])\s+", _norm(text)) if x.strip()]


# These patterns describe an unsafe *conclusion*, not merely the presence of a
# methodological risk in the study design. They are deliberately evaluated in
# conclusion-like/final sentences only.
RISKY_CONCLUSION_PATTERNS: dict[str, tuple[str, ...]] = {
    "attrition_bias": (r"well tolerated", r"nearly all", r"all treated patients"),
    "informative_missingness": (r"well tolerated", r"nearly all", r"all treated patients"),
    "collider_bias_risk": (r"protects? against", r"causal protective", r"all patients"),
    "scope_overreach": (r"all patients", r"entire patient population", r"nearly all", r"across the entire", r"consistent .* across"),
    "causal_overclaim": (r"\bproves?\b", r"\bestablish(?:es|ed)?\b.*\bcaus", r"directly causes?", r"protects? against", r"\bprevents?\b", r"\bconfirms?\b.*mechanism"),
    "time_zero_mismatch": (r"from (?:the time of )?diagnosis", r"beginning at diagnosis"),
    "landmark_selection_bias": (r"all patients", r"from (?:the time of )?diagnosis"),
    "multiplicity_risk": (r"robust", r"definitive", r"therapeutic effect", r"statistically significant benefit"),
    "reproducibility_conflict": (r"reproducible", r"replicated", r"robust"),
    "surrogate_to_clinical_overreach": (r"therapeutic effect", r"clinical benefit", r"prevents?", r"survival benefit"),
    "necessity_violation": (r"necessary", r"exclusively", r"sole pathway", r"entirely dependent"),
    "exclusivity_conflict": (r"exclusively", r"sole pathway", r"only in"),
    "noninferiority_interpretation_error": (r"equally effective", r"equivalent", r"equivalence"),
    "equivalence_fallacy": (r"equally effective", r"equivalent", r"equivalence"),
    "post_treatment_adjustment": (r"no survival benefit", r"has no .*benefit", r"because .*adjusted .*null"),
    "estimand_mismatch": (r"no survival benefit", r"has no .*benefit", r"total effect .* absent"),
    "magnitude_inflation": (r"\blarge\b", r"definitively", r"substantial .* benefit"),
    "evidence_strength_mismatch": (r"definitively", r"\bproves?\b", r"\bdemonstrates?\b", r"\bestablish(?:es|ed)?\b", r"statistically significant benefit"),
    "unsupported_generalization": (r"all patients", r"entire patient population", r"consistent .* across", r"nearly all"),
    "subgroup_significance_fallacy": (r"only in men", r"only in women", r"no benefit in women", r"no benefit in men"),
    "unsupported_effect_heterogeneity": (r"only in men", r"only in women", r"no benefit in women", r"no benefit in men", r"sex-specific"),
    "temporal_mechanism_conflict": (r"immediate mechanism", r"confirms?.*mechanism", r"responsible for"),
    "temporal_scope_extrapolation": (r"long-term", r"permanent", r"sustained benefit"),
    "competing_risk_misclassification": (r"prevents? .*recurrence", r"recurrence prevention", r"eliminates? recurrence"),
    "reproducibility_conflict": (r"robust", r"reproducible"),
}


GENERAL_SAFE_PATTERNS: tuple[str, ...] = (
    r"cannot (?:by itself )?(?:establish|infer|demonstrate|prove|support)",
    r"does not (?:establish|demonstrate|prove|support|confirm)",
    r"do not (?:establish|demonstrate|prove|support|confirm)",
    r"is not established",
    r"are not established",
    r"not established",
    r"should not be generalized",
    r"cannot be generalized",
    r"no .* generalization .* is established",
    r"no causal .* is established",
    r"direct causation is not established",
    r"remains? uncertain",
    r"is uncertain",
    r"hypothesis-generating",
    r"may (?:distort|bias|reflect)",
    r"cannot be excluded",
    r"not an equivalence trial",
    r"did not meet .* noninferiority",
    r"does not establish equal effectiveness",
    r"not statistically significant .* interaction",
    r"data do not establish",
    r"applies only",
    r"limited to",
    r"associated with.*(?:but|however)",
)


ISSUE_ACKNOWLEDGEMENT_PATTERNS: dict[str, tuple[str, ...]] = {
    "attrition_bias": (r"informative attrition", r"45% discontinued", r"excluded", r"dropout"),
    "informative_missingness": (r"informative attrition", r"discontinued", r"excluded"),
    "collider_bias_risk": (r"conditioning on .* may distort", r"collider", r"influenced .* admission"),
    "competing_risk_misclassification": (r"death is a competing event", r"competing event", r"cause-specific .* not significantly different"),
    "time_zero_mismatch": (r"applies only after the landmark", r"cannot establish .* from diagnosis"),
    "landmark_selection_bias": (r"patients who survived", r"died before", r"applies only after the landmark"),
    "multiplicity_risk": (r"without .* multiplicity adjustment", r"not adjusted for multiple comparisons", r"exploratory"),
    "reproducibility_conflict": (r"not replicated", r"failed .* validation"),
    "surrogate_to_clinical_overreach": (r"exploratory biomarker", r"does not establish .* therapeutic"),
    "necessity_violation": (r"not established as necessary", r"not necessary", r"with and without"),
    "noninferiority_interpretation_error": (r"did not meet .* noninferiority", r"not an equivalence trial"),
    "equivalence_fallacy": (r"not an equivalence trial", r"does not establish equal effectiveness"),
    "post_treatment_adjustment": (r"conditional effect", r"post-treatment", r"affected by .* treatment"),
    "estimand_mismatch": (r"conditional effect", r"total .* benefit", r"different estimand"),
    "magnitude_inflation": (r"modest improvement", r"possibility of no effect", r"uncertain"),
    "evidence_strength_mismatch": (r"possibility of no effect", r"exploratory", r"cannot be excluded", r"not established"),
    "subgroup_significance_fallacy": (r"interaction was not statistically significant", r"effect estimates were similar"),
    "unsupported_effect_heterogeneity": (r"interaction was not statistically significant", r"effect estimates were similar", r"do not establish a sex-specific"),
    "temporal_mechanism_conflict": (r"did not begin to decline until", r"timing does not support", r"mechanism remains uncertain"),
    "causal_overclaim": (r"associated with", r"residual confounding", r"reverse causation", r"causation is not established"),
    "scope_overreach": (r"should not be generalized", r"applies only", r"no .* generalization"),
    "unsupported_generalization": (r"should not be generalized", r"applies only", r"no .* generalization"),
}


def _contains_any(text: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(pattern, text, flags=re.I) for pattern in patterns)


def _is_safe_sentence(sentence: str, issue_type: str) -> bool:
    text = sentence.lower()
    if _contains_any(text, GENERAL_SAFE_PATTERNS):
        return True
    return _contains_any(text, ISSUE_ACKNOWLEDGEMENT_PATTERNS.get(issue_type, ())) and bool(
        re.search(r"cannot|does not|do not|not establish|uncertain|should not|may distort|applies only|limited", text)
    )


def _conclusion_candidates(paragraph: str, nodes: list[dict[str, Any]]) -> list[str]:
    sentences = _sentences(paragraph)
    candidates: list[str] = sentences[-2:] if len(sentences) >= 2 else sentences
    for node in nodes:
        role = str(node.get("role") or "")
        certainty = str(node.get("certainty") or "")
        text = _norm(node.get("source_text"))
        if text and (role in {"conclusion", "claim"} or certainty in {"concludes", "establishes", "proves"}):
            candidates.append(text)
    # Preserve order while removing duplicates.
    return list(dict.fromkeys(x for x in candidates if x))


def _risk_claim_present(issue_type: str, conclusion_sentences: list[str]) -> bool:
    patterns = RISKY_CONCLUSION_PATTERNS.get(issue_type)
    if not patterns:
        # Unknown/model-only concern: it is not actionable unless a strong final
        # assertion is present. This avoids converting missing detail into error.
        patterns = (r"\bproves?\b", r"\bdefinitively\b", r"\bestablish(?:es|ed)?\b", r"\ball patients\b", r"\bexclusively\b")
    for sentence in conclusion_sentences:
        if _is_safe_sentence(sentence, issue_type):
            continue
        if _contains_any(sentence.lower(), patterns):
            return True
    return False


def _acknowledged(issue_type: str, paragraph: str) -> bool:
    text = paragraph.lower()
    return _contains_any(text, ISSUE_ACKNOWLEDGEMENT_PATTERNS.get(issue_type, ())) or _contains_any(text, GENERAL_SAFE_PATTERNS)


def _resolved(issue_type: str, conclusion_sentences: list[str], risk_present: bool) -> bool:
    safe_final = any(_is_safe_sentence(sentence, issue_type) for sentence in conclusion_sentences)
    # A methodological pattern that is explicitly acknowledged and is not used
    # to support an unsafe conclusion is a resolved/contextual risk, not a defect.
    return safe_final and not risk_present



def _node_ids_for_patterns(nodes: list[dict[str, Any]], patterns: tuple[str, ...]) -> list[str]:
    result: list[str] = []
    for node in nodes:
        text = _norm(node.get("source_text")).lower()
        if any(re.search(pattern, text, flags=re.I) for pattern in patterns):
            result.append(str(node.get("id")))
    return list(dict.fromkeys(x for x in result if x))


def _append_synthetic_issue(
    issues: list[dict[str, Any]],
    nodes: list[dict[str, Any]],
    *,
    issue_type: str,
    title: str,
    explanation: str,
    patterns: tuple[str, ...],
    severity: str = "high",
) -> None:
    if any(str(x.get("issue_type")) == issue_type for x in issues):
        return
    issues.append({
        "id": f"i{len(issues)+1}",
        "issue_type": issue_type,
        "issue_family": "v031_resolution_audit",
        "severity": severity,
        "title": title,
        "node_ids": _node_ids_for_patterns(nodes, patterns),
        "explanation": explanation,
        "logical_pattern": "risk pattern + unsafe conclusion",
        "suggested_revision": "Align the conclusion with the stated evidence, population, timing, and uncertainty.",
        "confidence": 0.95,
        "generated_by": "v031_local_semantic_refresh",
        "verification_level": "structural_methodological_risk",
    })


def ensure_structural_issues(graph: dict[str, Any], paragraph: str) -> dict[str, Any]:
    """Add local issue candidates when reusing a cached graph card.

    Candidate generation is intentionally permissive; apply_resolution_semantics
    subsequently decides whether each pattern is an active defect or an
    acknowledged/resolved risk.
    """
    result = copy.deepcopy(graph)
    nodes = result.get("nodes") or []
    issues = list(result.get("issues") or [])
    text = paragraph.lower()
    final = " ".join(_sentences(paragraph)[-2:]).lower()

    if re.search(r"discontinued|dropout", text) and re.search(r"excluded|omitted", text):
        _append_synthetic_issue(issues, nodes, issue_type="attrition_bias", title="Potential informative attrition", explanation="Participants with unfavorable outcomes may have been excluded from the analyzed sample.", patterns=(r"discontinued|dropout|excluded", r"well tolerated|nearly all"))
    if re.search(r"intensive.?care|\bicu\b", text) and re.search(r"admission", text) and re.search(r"severity|severe", text):
        _append_synthetic_issue(issues, nodes, issue_type="collider_bias_risk", title="Conditioning on ICU admission may induce selection bias", explanation="ICU admission may be influenced by both exposure and severity while severity predicts outcome.", patterns=(r"intensive|icu|admission", r"severity|severe|mortality", r"biomarker"), severity="medium")
    if re.search(r"death.*before.*recurrence|competing event", text) and re.search(r"censor|kaplan", text):
        _append_synthetic_issue(issues, nodes, issue_type="competing_risk_misclassification", title="Competing deaths may be treated as ordinary censoring", explanation="Death precludes recurrence and changes the estimand of a Kaplan–Meier analysis.", patterns=(r"death|competing|censor|kaplan", r"recurrence|prevents"))
    if "landmark" in text and re.search(r"diagnosis|first year|surviv", text):
        _append_synthetic_issue(issues, nodes, issue_type="time_zero_mismatch", title="Landmark exposure may be extrapolated to diagnosis", explanation="Exposure defined at a later landmark cannot establish benefit before that time.", patterns=(r"landmark|diagnosis|exposure", r"surviv|died|all patients"))
        _append_synthetic_issue(issues, nodes, issue_type="landmark_selection_bias", title="Landmark survivors may not represent all patients", explanation="Patients excluded before the landmark cannot be covered by an all-patient conclusion.", patterns=(r"surviv|died|excluded|landmark", r"all patients"))
    if re.search(r"multiple comparisons|multiplicity|twenty outcomes|20 outcomes", text):
        _append_synthetic_issue(issues, nodes, issue_type="multiplicity_risk", title="Multiplicity may weaken an isolated finding", explanation="Many unadjusted tests increase false-positive risk.", patterns=(r"multiple|multiplicity|twenty|20|exploratory", r"robust|therapeutic|significant"))
    if re.search(r"not replicated|validation", text):
        _append_synthetic_issue(issues, nodes, issue_type="reproducibility_conflict", title="Validation may conflict with reproducibility language", explanation="A failed or absent replication cannot support a reproducibility claim.", patterns=(r"replicat|validation", r"reproduc|robust"))
    if re.search(r"biomarker|surrogate", text) and re.search(r"therapeutic effect|clinical benefit", final):
        _append_synthetic_issue(issues, nodes, issue_type="surrogate_to_clinical_overreach", title="Biomarker evidence may be generalized to clinical benefit", explanation="A surrogate signal alone may not establish therapeutic clinical benefit.", patterns=(r"biomarker|surrogate", r"therapeutic|clinical benefit"))
    if re.search(r"with and without|not necessary|inflammation", text) and re.search(r"exclusive|sole pathway|entirely", final):
        _append_synthetic_issue(issues, nodes, issue_type="necessity_violation", title="Non-necessity conflicts with an exclusive mechanism claim", explanation="An effect observed without the proposed mediator is inconsistent with exclusivity.", patterns=(r"not necessary|with and without", r"exclusive|sole|proves"))
    if re.search(r"noninferior|noninferiority", text) and re.search(r"equal|equival", text):
        _append_synthetic_issue(issues, nodes, issue_type="noninferiority_interpretation_error", title="Noninferiority results may be converted into equivalence", explanation="Failure or nonsignificance in a noninferiority setting does not establish equality.", patterns=(r"noninfer|confidence interval", r"equal|equival"))
        _append_synthetic_issue(issues, nodes, issue_type="equivalence_fallacy", title="Nonsignificance may be mistaken for equivalence", explanation="A nonsignificant comparison is not evidence of equivalence.", patterns=(r"not statistically significant|nonsignificant", r"equal|equival"))
    if re.search(r"adjusted for .*response|adjustment for response|response measured .* after", text):
        _append_synthetic_issue(issues, nodes, issue_type="post_treatment_adjustment", title="A post-treatment response may be conditioned on", explanation="Adjustment for a treatment-induced response may remove part of the total effect.", patterns=(r"response|adjust", r"mortality|null|benefit"))
        _append_synthetic_issue(issues, nodes, issue_type="estimand_mismatch", title="Conditional and total effects may be conflated", explanation="A response-adjusted estimate may target a different estimand from total benefit.", patterns=(r"conditional|total|adjust|response", r"no .* benefit|null"), severity="medium")
    if re.search(r"modest|possibility of no effect", text) and re.search(r"large|definitively|statistically significant benefit", final):
        _append_synthetic_issue(issues, nodes, issue_type="magnitude_inflation", title="Effect magnitude may be inflated", explanation="A modest or uncertain estimate is stronger in the conclusion than in the evidence.", patterns=(r"modest|no effect|confidence interval", r"large|definitively|significant"))
        _append_synthetic_issue(issues, nodes, issue_type="evidence_strength_mismatch", title="Conclusion certainty may exceed evidence", explanation="Exploratory or uncertain evidence does not support a definitive conclusion.", patterns=(r"exploratory|uncertain|no effect", r"definitively|establish|demonstrate"))
    if re.search(r"retrospective|nonrandomized|residual confounding|reverse causation", text) and re.search(r"caus|proves?|establish", final):
        _append_synthetic_issue(issues, nodes, issue_type="causal_overclaim", title="Observational association may be stated causally", explanation="Residual confounding or reverse causation limits causal interpretation.", patterns=(r"retrospective|confounding|associated", r"caus|proves|establish"))
        _append_synthetic_issue(issues, nodes, issue_type="evidence_strength_mismatch", title="Causal certainty may exceed observational evidence", explanation="Measured-variable adjustment does not by itself eliminate residual confounding.", patterns=(r"confounding|associated", r"caus|establish"))
    if re.search(r"post hoc|small .*subset|subgroup", text) and re.search(r"entire patient population|all patients|consistent .* across", final):
        _append_synthetic_issue(issues, nodes, issue_type="scope_overreach", title="A subgroup result may be generalized beyond its scope", explanation="A small or post hoc subgroup does not establish benefit across the full population.", patterns=(r"post hoc|subset|subgroup|full population", r"entire|all patients|consistent"))
        _append_synthetic_issue(issues, nodes, issue_type="evidence_strength_mismatch", title="Exploratory subgroup evidence may be overstated", explanation="Post hoc subset evidence is exploratory.", patterns=(r"post hoc|exploratory|small", r"demonstrate|establish|consistent"))
    if re.search(r"interaction.*not statistically significant|effect estimates were similar", text) and re.search(r"only in men|only in women|no benefit in", final):
        _append_synthetic_issue(issues, nodes, issue_type="subgroup_significance_fallacy", title="Within-subgroup significance may be mistaken for interaction", explanation="Significance in one subgroup and not another does not prove a subgroup difference.", patterns=(r"men|women|interaction|similar", r"only in|no benefit"))
        _append_synthetic_issue(issues, nodes, issue_type="unsupported_effect_heterogeneity", title="Sex-specific heterogeneity may be unsupported", explanation="Similar estimates and a nonsignificant interaction do not support exclusivity.", patterns=(r"interaction|similar|men|women", r"only in|no benefit"))
    if re.search(r"within 24 hours|two weeks later|did not begin to decline", text) and re.search(r"immediate mechanism|confirms?.*mechanism", final):
        _append_synthetic_issue(issues, nodes, issue_type="temporal_mechanism_conflict", title="Mechanism timing may conflict with the clinical response", explanation="The proposed mediator changes after the clinical effect begins.", patterns=(r"24 hours|two weeks|decline", r"immediate mechanism|confirms"))
    if re.search(r"associated with", text) and re.search(r"protects? against|prevents?|directly causes?|proves?|establish.*caus", final):
        _append_synthetic_issue(issues, nodes, issue_type="causal_overclaim", title="Association may be converted into causation", explanation="The conclusion uses causal language stronger than the reported association.", patterns=(r"associated", r"protect|prevent|caus|prove|establish"))
    if re.search(r"included only|among .*patients|small .*subset|completers|landmark", text) and re.search(r"all patients|entire patient population|nearly all", final):
        _append_synthetic_issue(issues, nodes, issue_type="scope_overreach", title="Conclusion may exceed the analyzed population", explanation="The conclusion extends beyond the population represented by the evidence.", patterns=(r"included only|among|subset|completer|landmark", r"all patients|entire|nearly all"))

    result["issues"] = issues
    return result

def apply_resolution_semantics(graph: dict[str, Any], paragraph: str) -> dict[str, Any]:
    """Annotate graph issues as active, acknowledged, resolved, or contextual.

    The function does not treat a methodological pattern by itself as an error.
    An issue becomes actionable only when the final/conclusion-like text makes an
    unsafe claim and the paragraph does not resolve it with an explicit qualifier.
    """
    result = ensure_structural_issues(graph, paragraph)
    nodes = result.get("nodes") or []
    edges = result.get("edges") or []
    issues = result.get("issues") or []
    conclusions = _conclusion_candidates(paragraph, nodes)

    active: list[dict[str, Any]] = []
    resolved_rows: list[dict[str, Any]] = []
    for issue in issues:
        issue_type = str(issue.get("issue_type") or "other")
        risk_present = _risk_claim_present(issue_type, conclusions)
        acknowledged = _acknowledged(issue_type, paragraph)
        resolved = _resolved(issue_type, conclusions, risk_present)
        actionable = bool(risk_present and not resolved)
        if actionable:
            state = "active"
        elif resolved:
            state = "resolved"
        elif acknowledged:
            state = "acknowledged"
        else:
            state = "contextual"

        issue["pattern_present"] = True
        issue["acknowledged_by_text"] = acknowledged
        issue["resolved_by_qualification"] = resolved
        issue["unsafe_conclusion_present"] = risk_present
        issue["actionable_defect"] = actionable
        issue["issue_state"] = state
        issue["resolution_semantics_version"] = SEMANTICS_VERSION
        if actionable:
            issue["resolution_reason"] = "An unsafe conclusion remains active after considering explicit qualifications."
            active.append(issue)
        else:
            if resolved:
                reason = "The final/conclusion-like text explicitly limits or rejects the unsafe inference."
            elif acknowledged:
                reason = "The paragraph acknowledges the risk, and no matching unsafe conclusion was identified."
            else:
                reason = "A possible pattern was noted, but no matching unsafe conclusion was identified."
            issue["resolution_reason"] = reason
            resolved_rows.append(issue)

    result["issues"] = issues
    result["actionable_issues"] = active
    result["resolved_or_contextual_issues"] = resolved_rows
    result["resolution_semantics_version"] = SEMANTICS_VERSION
    result["overall_assessment"] = "actionable_defect" if active else "no_actionable_defect"
    summary = dict(result.get("summary") or {})
    summary.update({
        "issue_count": len(issues),
        "actionable_issue_count": len(active),
        "resolved_or_contextual_issue_count": len(resolved_rows),
        "active_issue_type_counts": dict(_count_types(active)),
        "issue_state_counts": dict(_count_states(issues)),
    })
    result["summary"] = summary
    # Integrity should reflect active defects, not acknowledged/resolved risks.
    result["graph_metrics"] = calculate_graph_metrics(nodes, edges, issues=active)
    return result


def _count_types(issues: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for issue in issues:
        key = str(issue.get("issue_type") or "other")
        counts[key] = counts.get(key, 0) + 1
    return counts


def _count_states(issues: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for issue in issues:
        key = str(issue.get("issue_state") or "unknown")
        counts[key] = counts.get(key, 0) + 1
    return counts


def graph_from_card(card: str, paragraph: str = "") -> dict[str, Any] | None:
    """Reconstruct the reusable graph subset from a v030 graph card.

    This permits v031 to reuse already-paid graph extraction outputs. The card
    contains the node/edge fields needed by the downstream benchmark; v031 then
    recomputes deterministic issue states locally.
    """
    if "GRAPH STRUCTURE" not in str(card or ""):
        return None
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    for raw in str(card).splitlines():
        node_match = re.match(
            r'^- \[(?P<id>[^\]]+)\] role=(?P<role>\S+) assertion=(?P<assertion>\S+) certainty=(?P<certainty>\S+) quote=(?P<quote>.+)$',
            raw.strip(),
        )
        if node_match:
            quote_raw = node_match.group("quote")
            try:
                quote = json.loads(quote_raw)
            except Exception:
                quote = quote_raw.strip('"')
            nodes.append({
                "id": node_match.group("id"),
                "role": node_match.group("role"),
                "assertion_type": node_match.group("assertion"),
                "certainty": node_match.group("certainty"),
                "source_text": _norm(quote),
                "plain_meaning": _norm(quote),
                "source_fidelity_status": "not_checked",
            })
            continue
        edge_match = re.match(
            r'^- (?P<source>\S+) --(?P<relation>[^-]+)--> (?P<target>\S+):\s*(?P<rationale>.*)$',
            raw.strip(),
        )
        if edge_match:
            edges.append({
                "id": f"e{len(edges)+1}",
                "source": edge_match.group("source"),
                "target": edge_match.group("target"),
                "relation": edge_match.group("relation"),
                "rationale": _norm(edge_match.group("rationale")),
                "confidence": 1.0,
            })
    if not nodes:
        return None
    return {
        "schema_version": "0.31.0-reconstructed-card",
        "nodes": nodes,
        "edges": edges,
        "issues": [],
        "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        "api_call_count": 0,
        "latency_ms": 0.0,
        "cache_reconstructed": True,
        "graph_metrics": calculate_graph_metrics(nodes, edges, issues=[]),
        "summary": {"node_count": len(nodes), "edge_count": len(edges)},
    }


def contains_unsafe_claim(text: str, issue_type: str) -> bool:
    """Return True only when risky wording is asserted rather than negated/qualified."""
    for sentence in _sentences(text):
        if _is_safe_sentence(sentence, issue_type):
            continue
        patterns = RISKY_CONCLUSION_PATTERNS.get(issue_type, ())
        if patterns and _contains_any(sentence.lower(), patterns):
            return True
    return False

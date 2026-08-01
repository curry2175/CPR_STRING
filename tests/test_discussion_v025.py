from __future__ import annotations

import json
from pathlib import Path

from vrg.discussion_graph import (
    DiscussionGraphOutput,
    DiscussionIssue,
    DiscussionNode,
    analyze_structured_discussion,
)

ROOT = Path(__file__).resolve().parents[1]


def _output_for_text(text: str, *, issues=None, nodes=None):
    if nodes is None:
        sentences = [x.strip() for x in text.split(".") if x.strip()]
        nodes = [
            DiscussionNode(
                id=f"x{i}", sentence_index=i, source_text=sentence + ".",
                plain_meaning=sentence, role="observation", assertion_type="descriptive",
                polarity="positive", certainty="reported",
            )
            for i, sentence in enumerate(sentences, 1)
        ]
    return DiscussionGraphOutput(
        paragraph_summary="Regression fixture",
        nodes=nodes,
        edges=[],
        issues=issues or [],
        overall_assessment="potential_issue",
    )


def _issue_types(result):
    return {x["issue_type"] for x in result["issues"]}


def test_v025_regression_corpus_contains_all_collected_error_families():
    path = ROOT / "data" / "discussion_regression_cases_v025.jsonl"
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    assert len(rows) == 13
    ids = {x["id"] for x in rows}
    assert {"attrition_bias", "landmark_scope", "post_treatment_adjustment", "collider_selection", "competing_risk", "multiplicity_reproducibility"}.issubset(ids)


def test_v025_source_fidelity_and_number_preservation():
    text = "However, 45% of participants discontinued follow-up."
    node = DiscussionNode(
        id="x", sentence_index=1,
        source_text="Almost half of participants discontinued follow-up.",
        plain_meaning="Many participants discontinued.", role="observation",
        assertion_type="statistical", polarity="positive", certainty="reported",
    )
    result = analyze_structured_discussion({"input_text": text, "structured_output": _output_for_text(text, nodes=[node]).model_dump()})
    row = result["nodes"][0]
    assert row["source_fidelity_status"] in {"paraphrased", "partial"}
    assert row["matched_source_span"] == text
    assert row["numeric_mentions"] == ["45%"]
    assert result["summary"]["source_fidelity_warning_count"] == 1


def test_v025_exact_source_span_is_distinguished_from_normalized_claim():
    text = "Treatment H remained associated with lower mortality."
    node = DiscussionNode(
        id="x", sentence_index=1, source_text=text,
        plain_meaning="Treatment H had an adjusted mortality association.", role="evidence",
        assertion_type="association", polarity="positive", certainty="reported",
    )
    result = analyze_structured_discussion({"input_text": text, "structured_output": _output_for_text(text, nodes=[node]).model_dump()})
    row = result["nodes"][0]
    assert row["source_span_exact"] is True
    assert row["source_text"] == text
    assert row["normalized_claim"] != row["source_text"]


def test_v025_eligibility_rule_is_not_biological_mechanism():
    text = "Survival to one year was required for inclusion in the landmark analysis."
    node = DiscussionNode(
        id="x", sentence_index=1, source_text=text, plain_meaning=text,
        role="mechanism", assertion_type="necessity", polarity="positive", certainty="reported",
    )
    result = analyze_structured_discussion({"input_text": text, "structured_output": _output_for_text(text, nodes=[node]).model_dump()})
    row = result["nodes"][0]
    assert row["role"] == "eligibility_criterion"
    assert row["methodological_role"] == "eligibility_rule"
    assert row["causal_role"] == "selection_variable"


def test_v025_post_treatment_adjustment_and_estimand_are_added_structurally():
    text = (
        "Treatment N was associated with lower mortality before adjustment. "
        "The primary analysis adjusted for treatment response measured three months after therapy began. "
        "Treatment response was strongly affected by Treatment N and predicted mortality. "
        "The authors concluded that Treatment N has no survival benefit because the adjusted estimate was null."
    )
    result = analyze_structured_discussion({"input_text": text, "structured_output": _output_for_text(text).model_dump()})
    assert {"post_treatment_adjustment", "estimand_mismatch"}.issubset(_issue_types(result))
    levels = {x["issue_type"]: x["verification_level"] for x in result["issues"]}
    assert levels["post_treatment_adjustment"] == "structural_methodological_risk"


def test_v025_landmark_time_zero_and_survivor_selection_are_separate():
    text = (
        "Among patients who survived one year, exposure was assigned at the one-year landmark. "
        "Patients who died before the landmark were excluded. "
        "The authors concluded benefit from diagnosis in all patients."
    )
    result = analyze_structured_discussion({"input_text": text, "structured_output": _output_for_text(text).model_dump()})
    assert {"time_zero_mismatch", "landmark_selection_bias"}.issubset(_issue_types(result))


def test_v025_competing_event_is_not_reduced_to_generic_limitation():
    text = (
        "Deaths before recurrence were treated as censoring in the Kaplan-Meier analysis. "
        "The authors concluded that treatment prevents recurrence."
    )
    result = analyze_structured_discussion({"input_text": text, "structured_output": _output_for_text(text).model_dump()})
    assert "competing_risk_misclassification" in _issue_types(result)
    issue = next(x for x in result["issues"] if x["issue_type"] == "competing_risk_misclassification")
    assert issue["verification_level"] == "structural_methodological_risk"


def test_v025_noninferiority_is_not_labeled_scope_overreach():
    text = (
        "Treatment O did not meet the noninferiority margin. The study was not an equivalence trial. "
        "The comparison was not statistically significant, so the authors concluded equal effectiveness."
    )
    generic = DiscussionIssue(
        id="z", issue_type="scope_overreach", severity="high", title="Conclusion exceeds design",
        node_ids=[], explanation="The equivalence conclusion exceeds the study design.",
        logical_pattern="nonsignificance to equivalence",
    )
    result = analyze_structured_discussion({"input_text": text, "structured_output": _output_for_text(text, issues=[generic]).model_dump()})
    assert "noninferiority_interpretation_error" in _issue_types(result)
    assert not any(x["issue_type"] == "scope_overreach" for x in result["issues"])


def test_v025_subgroup_exclusive_claim_is_not_mechanism_conflict():
    text = (
        "Treatment K was significant in men but not statistically significant in women. "
        "The treatment-by-sex interaction was not statistically significant. "
        "The authors concluded that Treatment K is effective only in men."
    )
    issue = DiscussionIssue(
        id="z", issue_type="exclusivity_conflict", severity="high", title="Male-only efficacy",
        node_ids=[], explanation="Effective exclusively in men despite no significant interaction.",
        logical_pattern="exclusive to men",
    )
    result = analyze_structured_discussion({"input_text": text, "structured_output": _output_for_text(text, issues=[issue]).model_dump()})
    assert "subgroup_significance_fallacy" in _issue_types(result)
    assert "exclusivity_conflict" not in _issue_types(result)


def test_v025_multiplicity_replication_and_surrogate_are_distinct_issues():
    text = (
        "The study evaluated 20 outcomes without adjusting for multiple comparisons. "
        "One exploratory biomarker had a nominal p value of 0.04 but was not replicated in the validation cohort. "
        "The authors concluded a robust and reproducible therapeutic effect."
    )
    result = analyze_structured_discussion({"input_text": text, "structured_output": _output_for_text(text).model_dump()})
    assert {"multiplicity_risk", "reproducibility_conflict", "surrogate_to_clinical_overreach"}.issubset(_issue_types(result))
    reproducibility = next(x for x in result["issues"] if x["issue_type"] == "reproducibility_conflict")
    assert reproducibility["verification_level"] == "formal_conflict"


def test_v025_issue_groups_collapse_related_findings_by_family_and_anchor():
    text = "Benefit was not necessary through inflammation, but the conclusion called inflammation the exclusive mechanism."
    nodes = [
        DiscussionNode(id="a", sentence_index=1, source_text=text, plain_meaning="Inflammation was not necessary.", role="evidence", assertion_type="necessity", polarity="negative", certainty="reported"),
        DiscussionNode(id="b", sentence_index=1, source_text=text, plain_meaning="Inflammation was the exclusive mechanism.", role="conclusion", assertion_type="exclusivity", polarity="positive", certainty="proves"),
    ]
    issues = [
        DiscussionIssue(id="x", issue_type="necessity_violation", severity="high", title="Necessity conflict", node_ids=["a", "b"], explanation="Not necessary but exclusive.", logical_pattern="not necessary + exclusive"),
        DiscussionIssue(id="y", issue_type="exclusivity_conflict", severity="high", title="Exclusive pathway conflict", node_ids=["a", "b"], explanation="Benefit without pathway conflicts with exclusivity.", logical_pattern="without M + only M"),
    ]
    result = analyze_structured_discussion({"input_text": text, "structured_output": _output_for_text(text, nodes=nodes, issues=issues).model_dump()})
    assert result["summary"]["issue_count"] == 2
    assert result["summary"]["issue_group_count"] == 1
    assert len(result["issue_groups"][0]["sub_findings"]) == 2


def test_v025_discussion_ui_clears_stale_results_and_shows_source_fidelity():
    page = (ROOT / "static" / "discussion_lab.html").read_text(encoding="utf-8")
    assert "Discussion Reasoning Lab · v028" in page
    assert "이전 결과는 표시하지 않습니다" in page
    assert "input_hash" in page
    assert "원문 직접 인용" in page
    assert "시스템 정규화 의미" in page
    assert "구조적으로 확인된 방법론 위험" in page

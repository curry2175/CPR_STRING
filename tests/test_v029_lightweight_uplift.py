from __future__ import annotations

import json
from pathlib import Path

from vrg.lightweight_uplift import (
    build_graph_card,
    load_uplift_benchmark,
    paired_uplift,
    score_method,
)


def _case(label: str = "flawed"):
    return {
        "label": label,
        "text": "The study was observational. Residual confounding cannot be excluded. The result proves that treatment directly causes lower mortality.",
        "gold_issue_types": ["causal_overclaim"] if label == "flawed" else [],
        "gold_source_spans": ["Residual confounding cannot be excluded"] if label == "flawed" else [],
        "gold_target_conclusion": "The result proves that treatment directly causes lower mortality." if label == "flawed" else "",
        "gold_safe_conclusion": "Treatment was associated with lower mortality, but direct causation is not established.",
    }


def test_v029_benchmark_is_balanced_across_three_variants():
    path = Path(__file__).resolve().parents[1] / "data" / "discussion_uplift_benchmark_v029.jsonl"
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    assert len(rows) == 78
    for variant in ("base", "distance_8", "distance_16"):
        subset = [x for x in rows if x["variant"] == variant]
        assert len(subset) == 26
        assert sum(x["label"] == "clean" for x in subset) == 13
        assert sum(x["label"] == "flawed" for x in subset) == 13


def test_load_uplift_benchmark_balances_labels_and_variants():
    path = Path(__file__).resolve().parents[1] / "data" / "discussion_uplift_benchmark_v029.jsonl"
    rows, info = load_uplift_benchmark(path, variants=["base", "distance_16"], limit=12, seed=3)
    assert len(rows) == 12
    assert set(x["variant"] for x in rows) == {"base", "distance_16"}
    assert sum(x["label"] == "clean" for x in rows) == 6
    assert sum(x["label"] == "flawed" for x in rows) == 6
    assert info["selected"] == 12


def test_graph_card_separates_structure_and_verified_insights():
    graph = {
        "nodes": [{"id": "d1", "role": "limitation", "assertion_type": "study_design", "certainty": "observed", "source_text": "Residual confounding cannot be excluded."}],
        "edges": [{"source": "d1", "relation": "limits", "target": "d2", "rationale": "Limits causal certainty"}],
        "issues": [{"issue_type": "causal_overclaim", "severity": "high", "title": "Causal overclaim", "node_ids": ["d1"], "explanation": "Association is stated as causation", "suggested_revision": "Use association language"}],
        "graph_metrics": {"node_count": 2, "edge_count": 1, "max_depth": 1},
    }
    structure = build_graph_card(graph, include_verified_insights=False)
    insight = build_graph_card(graph, include_verified_insights=True)
    assert "GRAPH STRUCTURE" in structure
    assert "DETERMINISTIC / TYPED-GRAPH INSIGHTS" not in structure
    assert "causal_overclaim" in insight


def test_strict_success_requires_detection_type_and_both_localizations():
    case = _case("flawed")
    output = {
        "predicted_problem": True,
        "predicted_issue_types": ["causal_overclaim"],
        "vulnerable_conclusion": "The result proves that treatment directly causes lower mortality.",
        "evidence_spans": ["Residual confounding cannot be excluded"],
        "revised_conclusion": "Treatment was associated with lower mortality, but direct causation is not established.",
    }
    scores = score_method(output, case)
    assert scores["strict_audit_success"] is True
    output["evidence_spans"] = []
    scores = score_method(output, case)
    assert scores["strict_audit_success"] is False


def test_clean_strict_success_rewards_no_false_alarm():
    case = _case("clean")
    output = {"predicted_problem": False, "predicted_issue_types": [], "vulnerable_conclusion": "", "evidence_spans": [], "revised_conclusion": ""}
    assert score_method(output, case)["strict_audit_success"] is True


def test_paired_uplift_counts_corrected_and_regressed_strict_cases():
    rows = [
        {"methods": {"small_direct": {"scores": {"strict_audit_success": False}}, "small_graph_insight": {"scores": {"strict_audit_success": True}}}},
        {"methods": {"small_direct": {"scores": {"strict_audit_success": True}}, "small_graph_insight": {"scores": {"strict_audit_success": False}}}},
        {"methods": {"small_direct": {"scores": {"strict_audit_success": False}}, "small_graph_insight": {"scores": {"strict_audit_success": True}}}},
    ]
    result = paired_uplift(rows, "small_graph_insight", seed=1)
    assert result["corrected_direct_failures"] == 2
    assert result["regressed_direct_successes"] == 1
    assert result["net_strict_success_gain"] == 1


def test_paired_uplift_skips_missing_scores_without_crashing():
    rows = [
        {
            "methods": {
                "small_direct": {"scores": {"strict_audit_success": True}},
                "small_graph_insight": {
                    "status": "error",
                    "scores": None,
                    "error": {"error_type": "ValidationError", "error": "EOF while parsing"},
                },
            }
        },
        {
            "methods": {
                "small_direct": {"scores": {"strict_audit_success": False}},
                "small_graph_insight": {"scores": {"strict_audit_success": True}},
            }
        },
    ]
    result = paired_uplift(rows, "small_graph_insight", seed=1)
    assert result["n_total"] == 2
    assert result["n"] == 1
    assert result["n_missing"] == 1
    assert result["corrected_direct_failures"] == 1

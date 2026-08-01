from __future__ import annotations

import json
from pathlib import Path

from vrg.comparative_evaluation import (
    BenchmarkCase,
    _discussion_method_summary,
    _summarize_audit_method,
    paired_method_comparison,
    stratified_binary_sample,
)


def _case(i: int, label: str) -> BenchmarkCase:
    return BenchmarkCase(
        dataset="x", case_id=str(i), task="t", instruction="i",
        context="c", question="q", gold_answer=label, raw_record={},
    )


def test_stratified_binary_sample_balances_labels():
    cases = [_case(i, "Yes") for i in range(8)] + [_case(100 + i, "No") for i in range(8)]
    selected, info = stratified_binary_sample(cases, 10, seed=7)
    assert len(selected) == 10
    assert info["yes"] == 5
    assert info["no"] == 5
    assert info["balanced"] is True


def test_paired_method_comparison_correction_and_regression():
    rows = [
        {"gold_answer": "Yes", "methods": {"direct": {"answer_correct": False}, "graph_repair": {"answer_correct": True}}},
        {"gold_answer": "No", "methods": {"direct": {"answer_correct": True}, "graph_repair": {"answer_correct": False}}},
        {"gold_answer": "No", "methods": {"direct": {"answer_correct": False}, "graph_repair": {"answer_correct": True}}},
        {"gold_answer": "Yes", "methods": {"direct": {"answer_correct": True}, "graph_repair": {"answer_correct": True}}},
    ]
    result = paired_method_comparison(rows, "graph_repair")
    assert result["corrected_initial_errors"] == 2
    assert result["regressed_initially_correct"] == 1
    assert result["net_correction"] == 1
    assert result["accuracy_delta_percentage_points"] == 25.0


def test_reasoning_audit_summary_reports_clean_false_positive_and_localization():
    rows = [
        {"gold_fault": False, "expected_root_nodes": [], "methods": {"graph_verifier": {"predicted_fault": False, "predicted_root_nodes": [], "usage": {}, "api_calls": 0}}},
        {"gold_fault": False, "expected_root_nodes": [], "methods": {"graph_verifier": {"predicted_fault": True, "predicted_root_nodes": ["s1"], "usage": {}, "api_calls": 0}}},
        {"gold_fault": True, "expected_root_nodes": ["s2"], "methods": {"graph_verifier": {"predicted_fault": True, "predicted_root_nodes": ["s2"], "usage": {}, "api_calls": 0}}},
    ]
    result = _summarize_audit_method(rows, "graph_verifier")
    assert result["tp"] == 1
    assert result["fp"] == 1
    assert result["clean_false_positive_rate_percent"] == 50.0
    assert result["root_exact_localization_percent"] == 100.0


def test_discussion_summary_counts_clean_false_positive_and_issue_types():
    rows = [
        {"gold_problem": False, "gold_issue_types": [], "gold_source_spans": [], "text": "Clean text.",
         "methods": {"discussion_graph": {"predicted_problem": False, "predicted_issue_types": [], "source_spans": [], "usage": {}, "api_calls": 1}}},
        {"gold_problem": True, "gold_issue_types": ["causal_overclaim"], "gold_source_spans": ["proves causation"], "text": "The result proves causation.",
         "methods": {"discussion_graph": {"predicted_problem": True, "predicted_issue_types": ["causal_overclaim"], "source_spans": ["proves causation"], "usage": {}, "api_calls": 1}}},
    ]
    result = _discussion_method_summary(rows, "discussion_graph")
    assert result["detection_f1_percent"] == 100.0
    assert result["issue_type_micro_f1_percent"] == 100.0
    assert result["source_localization_hit_percent"] == 100.0


def test_discussion_benchmark_has_matched_clean_and_flawed_pairs():
    path = Path(__file__).resolve().parents[1] / "data" / "discussion_audit_benchmark_v028.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 26
    by_pair = {}
    for row in rows:
        by_pair.setdefault(row["pair_id"], set()).add(row["label"])
    assert len(by_pair) == 13
    assert all(labels == {"clean", "flawed"} for labels in by_pair.values())

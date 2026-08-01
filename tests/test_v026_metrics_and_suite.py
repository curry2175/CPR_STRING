from __future__ import annotations

import json
from pathlib import Path

from vrg.discussion_graph import DiscussionGraphOutput, DiscussionNode, analyze_structured_discussion
from vrg.evaluation_suite import BenchmarkCase, _binary_label, _macro_f1, summarize_results
from vrg.graph_metrics import calculate_graph_metrics

ROOT = Path(__file__).resolve().parents[1]


def test_v026_graph_depth_width_and_scores():
    nodes = [
        {"id": "e1", "role": "evidence", "source_fidelity_status": "exact"},
        {"id": "e2", "role": "evidence", "source_fidelity_status": "exact"},
        {"id": "c1", "role": "claim", "source_fidelity_status": "exact"},
        {"id": "z1", "role": "conclusion", "source_fidelity_status": "exact"},
    ]
    edges = [
        {"source": "e1", "target": "c1", "relation": "supports", "confidence": 0.9},
        {"source": "e2", "target": "c1", "relation": "supports", "confidence": 0.9},
        {"source": "c1", "target": "z1", "relation": "supports", "confidence": 0.9},
    ]
    result = calculate_graph_metrics(nodes, edges)
    assert result["structure"]["maximum_depth"] == 2
    assert result["structure"]["maximum_width"] == 2
    assert result["structure"]["width_profile"] == [2, 1, 1]
    assert result["scores"]["grounding"] == 100.0
    assert result["scores"]["fidelity"] == 100.0


def test_v026_cycle_is_collapsed_for_depth():
    nodes = [{"id": x, "role": "claim"} for x in ("a", "b", "c")]
    edges = [
        {"source": "a", "target": "b", "relation": "supports", "confidence": 0.9},
        {"source": "b", "target": "a", "relation": "supports", "confidence": 0.9},
        {"source": "b", "target": "c", "relation": "supports", "confidence": 0.9},
    ]
    result = calculate_graph_metrics(nodes, edges)
    assert result["structure"]["has_cycle"] is True
    assert result["structure"]["maximum_depth"] == 1


def test_v026_discussion_output_contains_graph_metrics():
    text = "Treatment reduced inflammation. The authors concluded that treatment improved outcomes."
    output = DiscussionGraphOutput(
        paragraph_summary="fixture",
        nodes=[
            DiscussionNode(id="x1", sentence_index=1, source_text="Treatment reduced inflammation.", plain_meaning="Inflammation decreased.", role="evidence", assertion_type="descriptive", polarity="positive", certainty="observed"),
            DiscussionNode(id="x2", sentence_index=2, source_text="The authors concluded that treatment improved outcomes.", plain_meaning="Treatment improved outcomes.", role="conclusion", assertion_type="causal", polarity="positive", certainty="concludes"),
        ],
        edges=[], issues=[], overall_assessment="internally_consistent",
    )
    result = analyze_structured_discussion({"input_text": text, "structured_output": output.model_dump()})
    assert result["schema_version"] == "0.27.0"
    assert "graph_metrics" in result
    assert result["graph_metrics"]["size"]["node_count"] == 2


def test_v026_strict_binary_label_policy():
    assert _binary_label("Yes") == "Yes"
    assert _binary_label("false") == "No"
    assert _binary_label("maybe") is None
    assert _binary_label("Unknown") is None


def test_v026_summary_has_dataset_metrics():
    rows = [
        {"dataset": "legalbench", "gold_answer": "Yes", "predicted_answer": "Yes", "answer_correct": True, "usage": {}, "graph_metrics": {"structure": {"maximum_depth": 2, "maximum_width": 2}, "scores": {"complexity": 50, "grounding": 80, "integrity": 100, "fidelity": 90}}},
        {"dataset": "legalbench", "gold_answer": "No", "predicted_answer": "Yes", "answer_correct": False, "usage": {}, "graph_metrics": {"structure": {"maximum_depth": 1, "maximum_width": 1}, "scores": {"complexity": 30, "grounding": 70, "integrity": 90, "fidelity": 100}}},
    ]
    summary = summarize_results(rows, [])
    assert summary["datasets"]["legalbench"]["accuracy_percent"] == 50.0
    assert summary["datasets"]["legalbench"]["mean_graph_depth"] == 1.5
    assert summary["strict_binary_policy"]["pubmedqa_maybe_excluded"] is True


def test_v026_ui_exposes_semantic_tabs_metrics_and_suite_page():
    page = (ROOT / "static" / "discussion_lab.html").read_text(encoding="utf-8")
    assert "Discussion Reasoning Lab · v028" in page
    assert "Semantic representation" in page
    assert "Width profile" in page
    assert "Graph 구조 Metric" in page
    suite = (ROOT / "static" / "suite_evaluation.html").read_text(encoding="utf-8")
    assert "Three-Dataset Evaluation · v028" in suite
    assert "ProofWriter Unknown" in suite
    assert "PubMedQA Maybe" in suite


def test_v026_cli_and_windows_runner_exist():
    assert (ROOT / "run_three_dataset_evaluation.py").exists()
    assert (ROOT / "RUN_THREE_DATASET_EVALUATION_WINDOWS.bat").exists()


def test_v026_legalbench_prefers_official_hf_test_split(monkeypatch, tmp_path):
    from vrg import evaluation_suite as suite

    calls = []

    def fake_cached(url, cache_path, refresh=False):
        calls.append(url)
        if "/splits?" in url:
            return {"splits": [
                {"config": "hearsay", "split": "train"},
                {"config": "hearsay", "split": "test"},
            ]}
        assert "split=test" in url
        if "offset=0" in url:
            return {"rows": [
                {"row_idx": 8, "row": {"answer": "Yes", "text": "An out-of-court statement offered for its truth."}},
                {"row_idx": 9, "row": {"answer": "Maybe", "text": "Excluded by strict binary policy."}},
            ]}
        return {"rows": []}

    monkeypatch.setattr(suite, "_cached_json_request", fake_cached)
    monkeypatch.setattr(suite, "_legalbench_readme", lambda *args, **kwargs: "Classify hearsay.")
    cases = suite._legalbench_cases_from_hf(tmp_path, limit=10, task_names=["hearsay"], refresh=False)
    assert len(cases) == 1
    assert cases[0].gold_answer == "Yes"
    assert cases[0].metadata["source_split"] == "test"
    assert any("split=test" in call for call in calls)


def test_v026_legalbench_requested_unknown_task_is_rejected(monkeypatch, tmp_path):
    from vrg import evaluation_suite as suite

    monkeypatch.setattr(
        suite,
        "_cached_json_request",
        lambda *args, **kwargs: {"splits": [{"config": "hearsay", "split": "test"}]},
    )
    try:
        suite._legalbench_cases_from_hf(tmp_path, limit=5, task_names=["not_a_task"], refresh=False)
    except ValueError as exc:
        assert "not found" in str(exc)
    else:
        raise AssertionError("unknown LegalBench task should fail explicitly")

from __future__ import annotations

from pathlib import Path

from vrg.ragtruth_catch_review import _report_html


def _sample_result() -> dict:
    source = {
        "nodes": [{"id": "S1", "node_type": "source_fact", "text": "The capital is Paris.", "evidence_ids": ["e1"]}],
        "edges": [],
    }
    response = {
        "nodes": [{"id": "R1", "node_type": "claim", "text": "The capital is Lyon.", "normalized_claim": "The capital is Lyon.", "sentence_id": "a1"}],
        "edges": [],
    }
    return {
        "candidate": {
            "case_id": "demo",
            "catch_reason": "direct_case_id_review",
            "task_instruction": "Answer from the source.",
            "response": "The capital is Lyon.",
            "gold_labels": [{"start": 15, "end": 19, "text": "Lyon"}],
            "evidence_card": {"text": "e1: The capital is Paris.", "units": [{"id": "e1", "text": "The capital is Paris."}]},
            "raw": {"scores": {"char_f1": 0.0}, "predicted_spans": []},
            "balanced_dual_graph": {
                "scores": {"char_f1": 1.0},
                "predicted_spans": [{"start": 15, "end": 19, "text": "Lyon"}],
                "source_graph": source,
                "response_graph": response,
                "alignments": [{"response_node_id": "R1", "source_node_ids": ["S1"], "relation": "contradicted_by", "confidence": 0.99, "explanation": "Mismatch"}],
            },
        },
        "node_comparisons": [{
            "node_id": "R1", "sentence_id": "a1", "text": "The capital is Lyon.", "normalized_claim": "The capital is Lyon.",
            "overlaps_gold": True, "overlaps_dual_prediction": True,
            "balanced_relation": "contradicted_by", "balanced_confidence": 0.99, "balanced_explanation": "Mismatch",
            "source_node_ids": ["S1"], "source_nodes": source["nodes"],
            "six_agent_verdict": "contradicted_by", "six_agent_confidence": 0.99, "six_agent_explanation": "Mismatch",
            "six_agent_changed_dimensions": ["entity"],
        }],
        "validated_source_graph": source,
        "validated_response_graph": response,
        "source_six_agent": {},
        "response_six_agent": {},
        "cross_graph": {"final_verdicts": [{"response_node_id": "R1", "source_node_ids": ["S1"], "verdict": "contradicted_by", "confidence": 0.99, "explanation": "Mismatch"}]},
    }


def test_v048_report_contains_discussion_lab_style_graph_views(tmp_path: Path):
    report = _report_html(_sample_result())
    assert "Interactive Source / Response Graph" in report
    assert "Source Graph" in report
    assert "Response Graph" in report
    assert "Cross comparison" in report
    assert "6-Agent validated" in report
    assert "Balanced candidate" in report
    assert "id='caseGraph'" in report
    assert "const GRAPH_DATA=" in report
    path = tmp_path / "report.html"
    path.write_text(report, encoding="utf-8")
    assert path.stat().st_size > 10000

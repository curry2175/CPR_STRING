from __future__ import annotations

from reproject_alignment_gate import _reproject_row
from vrg.ragtruth_dual_graph import (
    GRAPH_METHOD,
    RAW_METHOD,
    AlignmentRecord,
    DualGraphAlignmentOutput,
    ResponseAtomicityCheck,
    ResponseClaimGraphOutput,
    ResponseClaimNode,
    ResponseCoverageCheck,
    _component_key,
    _predictions_from_alignment,
)


def _graph(response: str) -> ResponseClaimGraphOutput:
    return ResponseClaimGraphOutput(
        nodes=[
            ResponseClaimNode(
                id="R1",
                sentence_id="a1",
                node_type="causal_claim",
                text=response,
                normalized_claim=response,
                claim_form="complete_sentence",
                evaluation_eligible=True,
                atomicity_check=ResponseAtomicityCheck(),
            )
        ],
        edges=[],
        coverage_check=ResponseCoverageCheck(),
    )


def test_v049_balanced_recall_emits_partial_claim_suppressed_by_v046():
    response = "Changes are typically due to water loss, not fat loss."
    graph = _graph(response)
    alignment = DualGraphAlignmentOutput(
        alignments=[
            AlignmentRecord(
                response_node_id="R1",
                relation="partially_supported_by",
                problem_text="typically due to water loss, not fat loss",
                label_type="unsupported",
                confidence=0.68,
                explanation="Water loss is not in the source.",
            )
        ]
    )
    old_predictions, _ = _predictions_from_alignment(
        alignment, graph, response, gate_profile="v046_conservative"
    )
    new_predictions, details = _predictions_from_alignment(
        alignment, graph, response, gate_profile="v049_balanced_recall"
    )
    assert old_predictions == []
    assert len(new_predictions) == 1
    assert "water loss" in new_predictions[0]["text"]
    assert details["alignment_gate_profile"] == "v049_balanced_recall"


def test_v046_cache_key_can_be_preserved_while_gate_changes():
    payload = {"case_id": "14396", "model": "gpt-5.4-nano"}
    legacy = _component_key(
        "alignment",
        payload,
        prompt_version="v046-dual-graph-conservative-factuality-gated-alignment",
    )
    same_legacy = _component_key(
        "alignment",
        payload,
        prompt_version="v046-dual-graph-conservative-factuality-gated-alignment",
    )
    new_prompt = _component_key(
        "alignment",
        payload,
        prompt_version="v049-dual-graph-balanced-recall-alignment",
    )
    assert legacy == same_legacy
    assert legacy != new_prompt


def test_offline_reprojection_changes_dual_score_without_changing_raw():
    response = "Changes are typically due to water loss, not fat loss."
    graph = _graph(response)
    alignment = DualGraphAlignmentOutput(
        alignments=[
            AlignmentRecord(
                response_node_id="R1",
                relation="partially_supported_by",
                problem_text="typically due to water loss, not fat loss",
                label_type="unsupported",
                confidence=0.68,
                explanation="unsupported causal attribution",
            )
        ]
    )
    raw_scores = {
        "char_precision": 1.0,
        "char_recall": 0.0,
        "char_f1": 0.0,
        "gold_has_hallucination": True,
        "predicted_has_hallucination": False,
    }
    row = {
        "case_id": "x1",
        "response": response,
        "gold_labels": [
            {
                "start": response.index("water loss"),
                "end": response.index("water loss") + len("water loss"),
                "text": "water loss",
                "label_type": "unsupported",
            }
        ],
        "methods": {
            RAW_METHOD: {"status": "ok", "predicted_spans": [], "scores": raw_scores},
            GRAPH_METHOD: {
                "status": "ok",
                "predicted_spans": [],
                "scores": raw_scores,
                "details": {},
                "generation_records": [
                    {"component": "response_graph", "status": "ok", "parsed": graph.model_dump()},
                    {"component": "alignment", "status": "ok", "parsed": alignment.model_dump()},
                ],
            },
        },
    }
    updated = _reproject_row(row, "v049_balanced_recall")
    assert updated is not None
    assert updated["methods"][RAW_METHOD]["scores"] == raw_scores
    assert updated["methods"][GRAPH_METHOD]["scores"]["char_recall"] > 0

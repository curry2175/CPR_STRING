from __future__ import annotations

from vrg.ragtruth_dual_graph import (
    AlignmentRecord,
    DualGraphAlignmentOutput,
    ResponseAtomicityCheck,
    ResponseClaimGraphOutput,
    ResponseClaimNode,
    _catch_reason,
    _predictions_from_alignment,
)


def _graph(text: str) -> ResponseClaimGraphOutput:
    return ResponseClaimGraphOutput(
        nodes=[ResponseClaimNode(
            id="R1",
            sentence_id="a1",
            node_type="claim",
            text=text,
            normalized_claim=text,
            inherited_context="",
            claim_form="complete_sentence",
            evaluation_eligible=True,
            atomicity_check=ResponseAtomicityCheck(),
        )],
        edges=[],
    )


def test_v046_safe_inference_and_generic_advice_never_emit_spans():
    response = "Research before your trip."
    graph = _graph(response)
    for relation in ("safe_inference", "generic_advice", "not_factual", "uncertain"):
        output = DualGraphAlignmentOutput(alignments=[AlignmentRecord(
            response_node_id="R1",
            source_node_ids=[],
            relation=relation,
            problem_text=response,
            label_type="unsupported",
            confidence=0.99,
            explanation="test",
        )])
        predictions, details = _predictions_from_alignment(output, graph, response)
        assert predictions == []
        assert details["alignments"][0]["submission_gate"] == "non_hallucination_relation"


def test_v046_low_confidence_unsupported_is_suppressed():
    response = "The device uses heat."
    graph = _graph(response)
    output = DualGraphAlignmentOutput(alignments=[AlignmentRecord(
        response_node_id="R1",
        source_node_ids=[],
        relation="not_found_in_source",
        problem_text=response,
        label_type="unsupported",
        confidence=0.70,
        explanation="not confident enough",
    )])
    predictions, details = _predictions_from_alignment(output, graph, response)
    assert predictions == []
    assert details["alignments"][0]["submission_gate"] == "suppressed_low_confidence_or_no_error_label"


def test_v046_high_confidence_contradiction_emits_span():
    response = "The capital is Lyon."
    graph = _graph(response)
    output = DualGraphAlignmentOutput(alignments=[AlignmentRecord(
        response_node_id="R1",
        source_node_ids=["S1"],
        relation="contradicted_by",
        problem_text="Lyon",
        label_type="contradiction",
        confidence=0.95,
        explanation="source says Paris",
    )])
    predictions, _ = _predictions_from_alignment(output, graph, response)
    assert len(predictions) == 1
    assert predictions[0]["text"] == "Lyon"


def test_v046_catch_reason_strict_and_material_uplift():
    assert _catch_reason(
        {"gold_has_hallucination": True, "predicted_has_hallucination": False, "char_f1": 0.0, "char_recall": 0.0},
        {"gold_has_hallucination": True, "predicted_has_hallucination": True, "char_f1": 0.8, "char_recall": 0.9},
    ) == "raw_detection_miss_dual_hit"
    assert _catch_reason(
        {"gold_has_hallucination": True, "predicted_has_hallucination": True, "char_f1": 0.2, "char_recall": 0.3},
        {"gold_has_hallucination": True, "predicted_has_hallucination": True, "char_f1": 0.6, "char_recall": 0.7},
    ) == "dual_material_uplift"


def test_v046_selected_catch_runs_six_agent_review_and_writes_html(tmp_path):
    import json
    import threading
    from types import SimpleNamespace

    from vrg.ragtruth_catch_review import run_selected_catch_review
    from vrg.ragtruth_six_agent_dual_graph import (
        CrossEvidenceMatchOutput,
        CrossLogicOutput,
        CrossLogicVerdict,
        CrossMatchRecord,
        GraphSpecialistReviewOutput,
    )

    class Responses:
        def __init__(self):
            self.calls = []
            self.lock = threading.Lock()

        def parse(self, **kwargs):
            with self.lock:
                self.calls.append(kwargs)
                n = len(self.calls)
            output_type = kwargs["text_format"]
            system = kwargs["input"][0]["content"]
            if output_type is GraphSpecialistReviewOutput:
                specialist = "evidence" if "Evidence Agent" in system else "logic" if "Logic Agent" in system else "target"
                parsed = GraphSpecialistReviewOutput(specialist=specialist)
            elif output_type is CrossEvidenceMatchOutput:
                parsed = CrossEvidenceMatchOutput(matches=[CrossMatchRecord(
                    response_node_id="R1", candidate_source_node_ids=["S1"],
                    candidate_evidence_ids=["e1"], match_type="semantic", confidence=0.99,
                )])
            elif output_type is CrossLogicOutput:
                parsed = CrossLogicOutput(verdicts=[CrossLogicVerdict(
                    response_node_id="R1", source_node_ids=["S1"], verdict="contradicted_by",
                    confidence=0.99, explanation="Paris contradicts Lyon", changed_dimensions=["entity"],
                )])
            elif output_type is DualGraphAlignmentOutput:
                parsed = DualGraphAlignmentOutput(alignments=[AlignmentRecord(
                    response_node_id="R1", source_node_ids=["S1"], relation="contradicted_by",
                    problem_text="Lyon", label_type="contradiction", confidence=0.99,
                )])
            else:
                raise AssertionError(output_type)
            item = SimpleNamespace(type="output_text", parsed=parsed)
            usage = SimpleNamespace(model_dump=lambda: {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15})
            return SimpleNamespace(
                id=f"resp_{n}", model=kwargs["model"],
                output=[SimpleNamespace(type="message", content=[item])], usage=usage,
            )

    class Client:
        def __init__(self):
            self.responses = Responses()

    benchmark = tmp_path / "benchmark"
    cases = benchmark / "catch_candidates" / "cases"
    cases.mkdir(parents=True)
    candidate = {
        "case_id": "c1", "source_id": "s1", "task_type": "QA", "task_instruction": "Answer from source.",
        "response": "The capital of France is Lyon.",
        "gold_labels": [{"start": 25, "end": 29, "text": "Lyon", "label_type": "contradiction"}],
        "catch_reason": "raw_detection_miss_dual_hit",
        "evidence_card": {"text": "e1: The capital of France is Paris.", "units": [{"id": "e1", "text": "The capital of France is Paris."}]},
        "raw": {"predicted_spans": [], "scores": {"char_f1": 0.0}},
        "balanced_dual_graph": {
            "predicted_spans": [{"start": 25, "end": 29, "text": "Lyon", "label_type": "contradiction"}],
            "scores": {"char_f1": 1.0},
            "source_graph": {"nodes": [{"id": "S1", "node_type": "source_fact", "text": "The capital of France is Paris.", "evidence_ids": ["e1"]}], "edges": [], "summary": ""},
            "response_graph": {"nodes": [{"id": "R1", "sentence_id": "a1", "node_type": "claim", "text": "The capital of France is Lyon.", "normalized_claim": "The capital of France is Lyon", "inherited_context": "", "claim_form": "complete_sentence", "evaluation_eligible": True, "atomicity_check": {"single_verdict_possible": True, "contains_multiple_independent_claims": False, "split_required": False, "note": ""}, "start": 0, "end": 30, "resolved_text": "The capital of France is Lyon."}], "edges": [], "coverage_check": {"all_factual_clauses_covered": True, "omitted_factual_text": []}},
            "alignments": [{"response_node_id": "R1", "source_node_ids": ["S1"], "relation": "contradicted_by", "problem_text": "Lyon", "label_type": "contradiction", "confidence": 0.99, "explanation": "source says Paris"}],
        },
    }
    (cases / "c1.json").write_text(json.dumps(candidate), encoding="utf-8")
    result = run_selected_catch_review(
        benchmark_run_dir=benchmark, case_id="c1", output_root=tmp_path / "reviews",
        cache_path=tmp_path / "cache.json", client=Client(),
    )
    review_dir = tmp_path / "reviews" / result["run_id"]
    assert (review_dir / "result.json").exists()
    assert (review_dir / "report.html").exists()
    assert result["api_calls_this_run"] == 9
    assert result["six_agent_predicted_spans"][0]["text"] == "Lyon"
